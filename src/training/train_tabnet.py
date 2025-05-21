import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from src.utils.io import load_classic
import json
from src.training.evaluate_tabnet import evaluate_tabnet_model

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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


def save_model_and_threshold(clf, threshold, path="src/models/"):
    clf.save_model(f"{path}_model.zip")
    with open(f"{path}_threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)
    print(f"[✓] Modell und Threshold gespeichert unter: {path}tabnet.zip / {path}threshold.json")


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
    from itertools import product

    search_space = {
        "n_d": [8, 16],
        "n_a": [8, 16],
        "mask_type": ["sparsemax", "entmax"],
        "lambda_sparse": [1e-3, 1e-4]
    }

    alpha = 2  # Kosten eines False Negatives (verpasster Angriff)
    beta = 1    # Kosten eines False Positives (Fehlalarm)

    configs = list(product(*search_space.values()))
    lowest_cost = float("inf")
    best_config = None
    best_model = None
    best_threshold = 0.5

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_classic()
    feature_names = X_train.columns.tolist()

    print(f"Starte Grid Search mit {len(configs)} Konfigurationen...")

    for i, values in enumerate(configs):
        params = dict(zip(search_space.keys(), values))
        print(f"\n[{i+1}/{len(configs)}] Teste Config: {params}")

        clf, threshold, cost = train_tabnet(
            X_train, y_train, X_val, y_val, params,
            threshold_plot_path=f"reports/figures/threshold_cost_val_{i}.png",
            alpha=alpha, beta=beta
        )

        print(f"Kosten auf Val: {cost:.2f} (Threshold: {threshold:.2f})")

        if cost < lowest_cost:
            lowest_cost = cost
            best_config = params
            best_model = clf
            best_threshold = threshold
            print("Neue beste Konfiguration gefunden!")

    print("Grid Search abgeschlossen.")
    print("Beste Konfiguration:")
    print(best_config)
    print(f"Minimale erwartete Kosten auf Val: {lowest_cost:.2f}")

    if best_model:
        save_model_and_threshold(best_model, best_threshold, path="models/tabnet")
        evaluate_tabnet_model(best_model, best_threshold, X_test, y_test, feature_names)