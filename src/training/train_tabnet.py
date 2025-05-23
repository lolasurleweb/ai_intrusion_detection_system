import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import confusion_matrix
from src.utils.io import load_classic, load_train_val_test_pool
import json
from src.training.evaluate_tabnet import evaluate_tabnet_model
import uuid
from datetime import datetime
from itertools import product
from sklearn.model_selection import StratifiedKFold

def compute_optimal_threshold_by_cost_function(y_true, y_proba, alpha, beta, save_plot_path=None):
    thresholds = np.linspace(0.0, 1.0, 200)
    costs, fns, fps = [], [], []

    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cost = alpha * fn + beta * fp
        costs.append(cost)
        fns.append(fn)
        fps.append(fp)

    min_idx = np.argmin(costs)
    best_threshold = thresholds[min_idx]
    best_cost = costs[min_idx]

    # Plot optional
    if save_plot_path:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, costs, label='Gesamtkosten')
        plt.axvline(best_threshold, linestyle='--', color='gray', label=f"Optimaler Schwellenwert: {best_threshold:.2f}")
        plt.xlabel("Threshold")
        plt.ylabel("Geschätzte Kosten")
        plt.title("Kostenbasierte Schwellenwertoptimierung")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close()
        print(f"[✓] Kostenplot gespeichert unter: {save_plot_path}")

    return best_threshold, best_cost



def save_model_and_threshold(clf, threshold, path="src/models/tabnet"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:6]
    suffix = f"{timestamp}_{uid}"

    model_path = f"{path}_model_{suffix}.zip"
    threshold_path = f"{path}_threshold_{suffix}.json"

    clf.save_model(model_path)
    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(threshold)}, f)

    print(f"[✓] Modell gespeichert unter: {model_path}")
    print(f"[✓] Threshold gespeichert unter: {threshold_path}")    

def train_tabnet(X_train, y_train, X_val, y_val, params, threshold_plot_path, alpha, beta):
    clf = TabNetClassifier(
        **params,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42
    )

    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    y_proba_val = clf.predict_proba(X_val.values)[:, 1]
    threshold, cost = compute_optimal_threshold_by_cost_function(
        y_val, y_proba_val, alpha=alpha, beta=beta,
        save_plot_path=threshold_plot_path
    )
    print(f"[✓] Kostenminimaler Schwellenwert: {threshold:.2f} (Kosten: {cost:.2f})")

    return clf, threshold, cost

def run_training():
    print("[✓] Lade Trainingsdaten für Grid Search...")
    X_full, y_full = load_train_val_test_pool()

    search_space = {
        "n_d": [8, 16],
        "n_a": [8, 16],
        "mask_type": ["sparsemax", "entmax"],
        "lambda_sparse": [1e-3, 1e-4]
    }

    alpha = 2
    beta = 1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configs = list(product(*search_space.values()))

    lowest_avg_cost = float("inf")
    best_config = None
    best_model = None
    best_threshold = None

    print(f"Starte Grid Search mit {len(configs)} Hyperparameter-Kombinationen...")

    for i, values in enumerate(configs):
        params = dict(zip(search_space.keys(), values))
        print(f"\n[{i+1}/{len(configs)}] Teste Config: {params}")

        fold_costs = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

            clf, threshold, cost = train_tabnet(
                X_train, y_train, X_val, y_val,
                params=params,
                threshold_plot_path=None,
                alpha=alpha,
                beta=beta
            )

            fold_costs.append(cost)
            print(f"  → Fold {fold_idx+1} Cost: {cost:.2f}")

        avg_cost = np.mean(fold_costs)
        print(f"➤ Durchschnittliche Kosten: {avg_cost:.2f}")

        if avg_cost < lowest_avg_cost:
            lowest_avg_cost = avg_cost
            best_config = params
            best_model = clf
            best_threshold = threshold
            print("Neue beste Konfiguration gefunden!")

    print("\nGrid Search abgeschlossen.")
    print("Beste Konfiguration:")
    print(best_config)
    print(f"Minimale durchschnittliche Kosten: {lowest_avg_cost:.2f}")

    if best_model:
        print("\n[✓] Lade klassische Splits für finales Training...")
        from src.utils.io import load_classic
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_classic()

        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)

        print("[✓] Trainiere finales Modell mit bester Konfiguration...")
        clf = TabNetClassifier(
            **best_config,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.95},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            seed=42
        )

        clf.fit(
            X_train=X_train_val.values, y_train=y_train_val.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        clf.forward_masks = True  # Aktiviert Speicherung der Feature Masks

        save_model_and_threshold(clf, best_threshold, path="models/tabnet")
        print("[✓] Finales Modell und validierter Threshold gespeichert.")

        print("\n[✓] Starte finale Evaluation auf dem unberührten Test-Set...")
        from src.training.evaluate_tabnet import evaluate_tabnet_model
        evaluate_tabnet_model(
            clf,
            threshold=best_threshold,
            X_test=X_test,
            y_test=y_test,
            feature_names=X_train.columns.tolist()
        )
