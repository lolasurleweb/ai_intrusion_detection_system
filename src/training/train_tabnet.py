import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import uuid
from datetime import datetime
from itertools import product

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score

from src.utils.io import load_pickle
from src.training.evaluate_tabnet import (
    evaluate_cross_validation_results,
    plot_training_history
)

from pytorch_tabnet.metrics import Metric

class CostMetric(Metric):
    def __init__(self):
        self._name = "cost_metric"
        self.alpha = 2
        self.beta = 1
        self.optimum = 0
        self.greater_is_better = False
        self._maximize = False

    def __call__(self, y_true, y_score):
        proba_pos = y_score[:, 1]
        thresholds = np.linspace(0.01, 0.99, 50)
        best_cost = float("inf")

        for t in thresholds:
            y_pred = (proba_pos > t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            pos_total = tp + fn
            neg_total = tn + fp

            fn_rate = fn / pos_total if pos_total > 0 else 0
            fp_rate = fp / neg_total if neg_total > 0 else 0

            cost = self.alpha * fn_rate + self.beta * fp_rate

            best_cost = min(best_cost, cost)

        return best_cost

def save_model(clf, path="src/models/tabnet"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    suffix = f"{timestamp}_{uid}"

    model_path = f"{path}_model_{suffix}"
    metadata_path = "models/final_model_metadata.json"

    clf.save_model(model_path)

    metadata = {
        "model_path": model_path,
        "timestamp": timestamp,
        "uid": uid,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Modell gespeichert: {model_path}")
    print(f"[✓] Metadaten gespeichert: {metadata_path}")


def train_tabnet(X_train, y_train, X_val, y_val, params, alpha, beta, fold_idx=None):
    clf = TabNetClassifier(
        **params,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42
    )

    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_train.values, y_train.values), (X_val.values, y_val.values)],
        eval_name=["train", "val"],
        eval_metric=["logloss", CostMetric],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    if fold_idx is not None:
        plot_training_history(clf, save_path=f"reports/figures/training_history_fold_{fold_idx+1}.png")

    return clf

def run_training():
    print("[✓] Lade Trainingsdaten...")
    trainval_df = load_pickle("data/processed/train_val_pool.pkl")
    X_full = trainval_df.drop(columns=["attack_detected"])
    y_full = trainval_df["attack_detected"]

    search_space = {
        "n_d": [8, 16],
        "n_a": [8, 16],
        "mask_type": ["sparsemax", "entmax"],
        "lambda_sparse": [1e-3, 1e-4]
    }
    alpha, beta = 2, 1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configs = list(product(*search_space.values()))
    print(f"Starte Grid Search mit {len(configs)} Konfigurationen...")

    early_stop_rounds = 5
    cost_tolerance = 1e-3
    bad_config_counter = 0

    best_config = None
    lowest_avg_cost = float("inf")
    best_fold_metrics = None
    best_fold_models = None

    for i, values in enumerate(configs):
        params = dict(zip(search_space.keys(), values))
        print(f"\n[{i+1}/{len(configs)}] Konfiguration: {params}")

        fold_costs, fold_metrics = [], []
        fold_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

            clf = train_tabnet(
                X_train, y_train, X_val, y_val,
                params, alpha=alpha, beta=beta,
                fold_idx=fold_idx
            )

            y_proba = clf.predict_proba(X_val.values)[:, 1]
            thresholds = np.linspace(0.01, 0.99, 50)
            best_cost = float("inf")

            for t in thresholds:
                y_pred = (y_proba > t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                pos_total = tp + fn
                neg_total = tn + fp
                fn_rate = fn / pos_total if pos_total > 0 else 0
                fp_rate = fp / neg_total if neg_total > 0 else 0
                cost = alpha * fn_rate + beta * fp_rate
                best_cost = min(best_cost, cost)

            fold_costs.append(best_cost)
            fold_metrics.append({
                "cost": best_cost,
                "f1": f1_score(y_val, (y_proba > 0.5).astype(int)),
                "auc": roc_auc_score(y_val, y_proba),
                "precision": precision_score(y_val, (y_proba > 0.5).astype(int)),
                "recall": recall_score(y_val, (y_proba > 0.5).astype(int))
            })
            fold_models.append((clf, best_cost))

            print(f"  → Fold {fold_idx+1}: Cost={best_cost:.2f}")

        avg_cost = np.mean(fold_costs)
        std_cost = np.std(fold_costs)
        print(f"→ Durchschnittliche Kosten: {avg_cost:.4f} ± {std_cost:.4f}")

        if avg_cost < (lowest_avg_cost - cost_tolerance):
            best_config = params
            best_fold_metrics = fold_metrics
            best_fold_models = fold_models
            lowest_avg_cost = avg_cost
            bad_config_counter = 0
            print("[✓] Neue beste Konfiguration gefunden.")
        else:
            bad_config_counter += 1
            print(f"[!] Keine Verbesserung. bad_config_counter = {bad_config_counter}")

        if bad_config_counter >= early_stop_rounds:
            print(f"[✘] Früher Abbruch der Grid Search nach {i+1} Konfigurationen.")
            break

    print("\nGrid Search abgeschlossen.")
    print(f"Beste Konfiguration: {best_config}")
    print(f"Kosten: {lowest_avg_cost:.2f} (Durchschnitt über 5 Folds)")

    evaluate_cross_validation_results(
        best_fold_metrics,
        best_config=best_config
    )

    best_fold_idx = np.argmin([m[1] for m in best_fold_models])
    final_clf, best_cost = best_fold_models[best_fold_idx]
    print(f"[✓] Bestes Fold-Modell: Fold {best_fold_idx+1} mit Cost {best_cost:.2f}")

    final_clf.forward_masks = True
    save_model(final_clf, path="models/tabnet_final")
    print("[✓] Finalmodell gespeichert.")