import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import uuid
from datetime import datetime
from itertools import product

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score, precision_score,
    recall_score, cohen_kappa_score, roc_auc_score
)

from src.utils.io import load_train_val_test_pool
from src.training.evaluate_tabnet import (
    plot_tabnet_feature_importance,
    evaluate_cross_validation_results,
    save_instance_level_explanations,
    compute_and_plot_permutation_importance,
    save_confusion_matrix
)


def compute_optimal_threshold_by_cost_function(y_true, y_proba, alpha, beta, save_plot_path=None):
    thresholds = np.linspace(0.0, 1.0, 200)
    costs = []

    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = alpha * fn + beta * fp
        costs.append(cost)

    min_idx = np.argmin(costs)
    best_threshold = thresholds[min_idx]
    best_cost = costs[min_idx]

    if save_plot_path:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, costs, label='Gesamtkosten')
        plt.axvline(best_threshold, linestyle='--', color='gray', label=f"Optimal: {best_threshold:.2f}")
        plt.xlabel("Threshold")
        plt.ylabel("Kosten")
        plt.title("Kostenbasierte Threshold-Optimierung")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close()
        print(f"[✓] Kostenplot gespeichert: {save_plot_path}")

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

    print(f"[✓] Modell gespeichert: {model_path}")
    print(f"[✓] Threshold gespeichert: {threshold_path}")


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
        y_val, y_proba_val, alpha, beta, save_plot_path=threshold_plot_path
    )

    print(f"[✓] Optimaler Threshold: {threshold:.2f}, Kosten: {cost:.2f}")
    return clf, threshold, cost


def run_training():
    print("[✓] Lade Trainingsdaten...")
    X_full, y_full = load_train_val_test_pool()

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

    best_config, best_threshold = None, None
    lowest_avg_cost = float("inf")
    best_metrics, best_fold_metrics = None, None

    for i, values in enumerate(configs):
        params = dict(zip(search_space.keys(), values))
        print(f"\n[{i+1}/{len(configs)}] Konfiguration: {params}")

        fold_costs, aucs, f1s, fold_metrics = [], [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
            X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
            y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

            clf, threshold, cost = train_tabnet(
                X_train, y_train, X_val, y_val,
                params, threshold_plot_path=None, alpha=alpha, beta=beta
            )

            y_proba = clf.predict_proba(X_val.values)[:, 1]
            y_pred = (y_proba > threshold).astype(int)

            fold_costs.append(cost)
            aucs.append(roc_auc_score(y_val, y_proba))
            f1s.append(f1_score(y_val, y_pred))

            fold_metrics.append({
                "cost": cost,
                "auc": aucs[-1],
                "f1": f1s[-1],
                "accuracy": accuracy_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "kappa": cohen_kappa_score(y_val, y_pred)
            })

            print(f"  → Fold {fold_idx+1}: Cost={cost:.2f}, AUC={aucs[-1]:.3f}, F1={f1s[-1]:.3f}")

        avg_cost = np.mean(fold_costs)
        if avg_cost < lowest_avg_cost:
            best_config = params
            best_threshold = threshold
            lowest_avg_cost = avg_cost
            best_metrics = {
                "cost": (np.mean(fold_costs), np.std(fold_costs)),
                "auc": (np.mean(aucs), np.std(aucs)),
                "f1": (np.mean(f1s), np.std(f1s))
            }
            best_fold_metrics = fold_metrics

    print("\nGrid Search abgeschlossen.")
    print(f"Beste Konfiguration: {best_config}")
    print(f"Kosten: {best_metrics['cost'][0]:.2f} ± {best_metrics['cost'][1]:.2f}")
    print(f"AUC:    {best_metrics['auc'][0]:.3f} ± {best_metrics['auc'][1]:.3f}")
    print(f"F1:     {best_metrics['f1'][0]:.3f} ± {best_metrics['f1'][1]:.3f}")

    evaluate_cross_validation_results(best_fold_metrics)

    print("\n[✓] Trainiere finales Modell auf Trainings- und Validierungsdaten mit Early-Stopping...")
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )

    final_clf = TabNetClassifier(
        **best_config,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42
    )

    final_clf.fit(
        X_train=X_train_final.values,
        y_train=y_train_final.values,
        eval_set=[(X_val_final.values, y_val_final.values)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    final_clf.forward_masks = True
    y_proba_full = final_clf.predict_proba(X_full.values)[:, 1]
    y_pred_full = (y_proba_full > best_threshold).astype(int)

    save_model_and_threshold(final_clf, threshold=best_threshold, path="models/tabnet_final")

    print("[✓] Generiere lokale Erklärungen für positive IDS-Warnungen...")
    save_instance_level_explanations(
        clf=final_clf,
        X=X_full,
        y_proba=y_proba_full,
        y_pred=y_pred_full,
        feature_names=X_full.columns.tolist(),
        threshold=best_threshold,
        save_path="reports/explanations/final_model_instance_level.json",
        top_k=5,
        only_positive_predictions=True,
        include_scores=True
    )

    print("[✓] Visualisiere TabNet Feature-Masken...")
    plot_tabnet_feature_importance(
        clf=final_clf,
        feature_names=X_full.columns.tolist(),
        save_path="reports/figures/tabnet_feature_masks_final.png"
    )

    print("[✓] Berechne Permutation Importance...")
    compute_and_plot_permutation_importance(
        clf=final_clf,
        X_val=X_full.sample(frac=0.25, random_state=42),
        y_val=y_full.loc[X_full.sample(frac=0.25, random_state=42).index],
        feature_names=X_full.columns.tolist(),
        save_path="reports/figures/permutation_importance_final.png",
        scoring='f1'
    )

    print("[✓] Speichere Confusion Matrix...")
    save_confusion_matrix(
        y_true=y_full,
        y_pred=y_pred_full,
        save_path="reports/figures/confusion_matrix_final.png"
    )

    print("[✓] Fertig.")
