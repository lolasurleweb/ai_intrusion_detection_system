import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from src.utils.io import load_classic
import json
from src.training.evaluate_tabnet import evaluate_tabnet_model

def compute_optimal_threshold(y_true, y_proba, save_plot_path):
    thresholds = np.linspace(0.1, 0.9, 50)
    recalls, precisions, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_proba > t).astype(int)
        recalls.append(recall_score(y_true, y_pred_t))
        precisions.append(precision_score(y_true, y_pred_t))
        f1s.append(f1_score(y_true, y_pred_t))

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, f1s, label="F1-Score")
    plt.axvline(best_threshold, color='gray', linestyle='--', label=f"Best Threshold: {best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep (Validation Set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)
    plt.close()

    return best_threshold, best_f1

def save_model_and_threshold(clf, threshold, path="src/models/"):
    clf.save_model(f"{path}_model.zip")
    with open(f"{path}_threshold.json", "w") as f:
        json.dump({"threshold": float(threshold)}, f)
    print(f"[✓] Modell und Threshold gespeichert unter: {path}tabnet.zip / {path}threshold.json")


def train_tabnet(X_train, y_train, X_val, y_val, params, threshold_plot_path):
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
        eval_metric=["balanced_accuracy"],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    y_proba_val = clf.predict_proba(X_val.values)[:, 1]
    threshold, f1 = compute_optimal_threshold(y_val, y_proba_val, save_plot_path=threshold_plot_path)
    return clf, threshold, f1

def run_training():
    from itertools import product

    search_space = {
        "n_d": [8, 16],
        "n_a": [8, 16],
        "mask_type": ["sparsemax", "entmax"],
        "lambda_sparse": [1e-3, 1e-4]
    }

    configs = list(product(*search_space.values()))
    best_f1 = 0
    best_config = None
    best_model = None
    best_threshold = 0.5

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_classic()
    feature_names = X_train.columns.tolist()

    print(f"Starte Grid Search mit {len(configs)} Konfigurationen...")

    for i, values in enumerate(configs):
        params = dict(zip(search_space.keys(), values))
        print(f"\n[{i+1}/{len(configs)}] Teste Config: {params}")

        clf, threshold, f1 = train_tabnet(X_train, y_train, X_val, y_val, params, threshold_plot_path=f"reports/figures/threshold_sweep_val_{i}.png")

        print(f"F1 auf Val: {f1:.4f} (Threshold: {threshold:.2f})")

        if f1 > best_f1:
            best_f1 = f1
            best_config = params
            best_model = clf
            best_threshold = threshold
            print("Neue beste Konfiguration gefunden!")

    print("Grid Search abgeschlossen.")
    print("Beste Konfiguration:")
    print(best_config)
    print(f"Beste F1 auf Validierung: {best_f1:.4f}")

    if best_model:
        save_model_and_threshold(best_model, best_threshold, path="models/tabnet")
        evaluate_tabnet_model(best_model, best_threshold, X_test, y_test, feature_names)
