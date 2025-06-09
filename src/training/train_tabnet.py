import os
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
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from src.utils.io import load_pickle
from src.training.evaluate_tabnet import (
    evaluate_cross_validation_results,
    plot_training_history
)
from sklearn.metrics import confusion_matrix
from pytorch_tabnet.metrics import Metric

class CostScore(Metric):
    def __init__(self, alpha=5.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self._name = "custom_cost"
        self._maximize = False

    def __call__(self, y_true, y_score):
        if y_score.ndim == 2:
            y_pred = np.argmax(y_score, axis=1)
        else:
            y_pred = (y_score > 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fnr = fn / (tp + fn + 1e-7)
        fpr = fp / (fp + tn + 1e-7)
        return self.alpha * fnr + self.beta * fpr

def save_fold_models(fold_models, params, fold_metrics, path_prefix="models/tabnet_cv"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    base_path = f"{path_prefix}_{timestamp}_{uid}"
    os.makedirs(base_path, exist_ok=True)

    model_paths = []

    for i, (clf, cost) in enumerate(fold_models):
        model_path = f"{base_path}/fold_{i+1}"
        clf.save_model(model_path)
        model_paths.append(model_path)

    metadata = {
        "timestamp": timestamp,
        "uid": uid,
        "base_path": base_path,
        "model_paths": model_paths,
        "hyperparameters": params,
        "validation_metrics": fold_metrics
    }

    with open(f"{base_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Alle Fold-Modelle gespeichert unter: {base_path}")
    print(f"[✓] Metadaten gespeichert unter: {base_path}/metadata.json")


def train_tabnet(X_train, y_train, X_val, y_val, params, fold_idx=None):
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
        eval_metric=["logloss", CostScore],
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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configs = list(product(*search_space.values()))
    print(f"Starte Grid Search mit {len(configs)} Konfigurationen...")

    early_stop_rounds = 5
    bad_config_counter = 0

    best_config = None
    lowest_avg_cost = float("inf")
    best_fold_metrics = None
    best_fold_models = None

    for i, values in enumerate(configs):
        params = dict(zip(search_space.keys(), values))
        print(f"\n[{i+1}/{len(configs)}] Konfiguration: {params}")

        fold_costs, fold_metrics, fold_models = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

            clf = train_tabnet(X_train, y_train, X_val, y_val, params, fold_idx=fold_idx)

            y_val_pred = clf.predict(X_val.values)
            y_val_proba = clf.predict_proba(X_val.values)[:, 1]

            val_cost = clf.history["val_custom_cost"][-1]

            fold_costs.append(val_cost)
            fold_metrics.append({
                "cost": val_cost,
                "logloss": clf.history["val_logloss"][-1],
                "f1": f1_score(y_val, y_val_pred),
                "auc": roc_auc_score(y_val, y_val_proba),
                "precision": precision_score(y_val, y_val_pred),
                "recall": recall_score(y_val, y_val_pred)
            })
            fold_models.append((clf, val_cost))

            print(f"  → Fold {fold_idx+1}: Val-Cost = {val_cost:.4f}")

        avg_cost = np.mean(fold_costs)
        std_cost = np.std(fold_costs)
        print(f"→ Durchschnittlicher Cost: {avg_cost:.4f} ± {std_cost:.4f}")

        if avg_cost < lowest_avg_cost:
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
    print(f"CustomCost: {lowest_avg_cost:.4f} (Durchschnitt über 5 Folds)")

    save_fold_models(
    fold_models=best_fold_models,
    params=best_config,
    fold_metrics=best_fold_metrics,
    path_prefix="models/tabnet_cv"
    )

    print("[✓] Finalmodelle gespeichert.")
