import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from src.utils.io import load_classic


def plot_tabnet_feature_importance(clf, feature_names, save_path=None):
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title("TabNet Feature Importance (via Feature Masks)")
    plt.xlabel("Importance")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[✓] Feature-Masken-Plot gespeichert unter: {save_path}")
    else:
        plt.show()


def compute_and_plot_permutation_importance(clf, X_val, y_val, feature_names, save_path=None, scoring='accuracy'):
    class SklearnLikeWrapper:
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            return self

        def predict(self, X):
            if isinstance(X, pd.DataFrame):
                X = X.values.astype(np.float32)
            return self.model.predict(X)

    wrapped_model = SklearnLikeWrapper(clf)

    result = permutation_importance(
        wrapped_model, X_val, y_val,
        scoring=scoring, n_repeats=10, random_state=42
    )

    sorted_idx = result.importances_mean.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], result.importances_mean[sorted_idx])
    plt.title("Permutation Importance")
    plt.xlabel(f"Mean decrease in {scoring}")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[✓] Permutation-Plot gespeichert unter: {save_path}")
    else:
        plt.show()

    return result


def save_metrics_as_table(accuracy, precision, recall, f1, kappa, save_path):
    df = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score", "Cohen's Kappa"],
        "Wert": [accuracy, precision, recall, f1, kappa]
    })

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title("Klassische Metriken")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"[✓] Metrik-Tabelle gespeichert unter: {save_path}")
    plt.close()


def save_confusion_matrix(y_true, y_pred, save_path):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"[✓] Confusion Matrix gespeichert unter: {save_path}")
    plt.close()


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
    plt.title("Schwellenwert-Analyse (Threshold vs. Metriken)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)
    print(f"[✓] Threshold-Sweep gespeichert unter: {save_plot_path}")
    plt.close()

    return best_threshold, best_f1


def train_and_evaluate():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_classic()
    feature_names = X_train.columns.tolist()

    print("[1] Starte Training...")
    clf = TabNetClassifier()
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
        drop_last=False,
    )

    print("[2] Berechne Wahrscheinlichkeiten und finde optimalen Threshold...")
    y_proba = clf.predict_proba(X_test.values)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 50)
    recalls, precisions, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_proba > t).astype(int)
        recalls.append(recall_score(y_test, y_pred_t))
        precisions.append(precision_score(y_test, y_pred_t))
        f1s.append(f1_score(y_test, y_pred_t))

    best_threshold = thresholds[np.argmax(f1s)]
    print(f"[✓] Optimaler Threshold (F1-basiert): {best_threshold:.2f}")

    # Finales Vorhersagen mit optimiertem Threshold
    y_pred = (y_proba > best_threshold).astype(int)

    # Klassische Metriken
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    # Speichern der Metriktabelle
    save_metrics_as_table(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        kappa=kappa,
        save_path="reports/figures/classic_metrics.png"
    )

    # Confusion Matrix
    save_confusion_matrix(
        y_test,
        y_pred,
        save_path="reports/figures/confusion_matrix.png"
    )

    # Threshold-Analyse-Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, f1s, label="F1-Score")
    plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best F1 @ {best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Schwellenwert-Analyse (Threshold vs. Metriken)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/figures/threshold_sweep.png")
    print("[✓] Threshold-Sweep gespeichert unter: reports/figures/threshold_sweep.png")

    print("[3] Speicher Explainability-Plots...")
    plot_tabnet_feature_importance(
        clf,
        feature_names,
        save_path="reports/figures/tabnet_feature_masks.png"
    )

    compute_and_plot_permutation_importance(
        clf,
        X_test.astype(np.float32).values,
        y_test.values,
        feature_names,
        save_path="reports/figures/permutation_importance.png"
    )

    print("[✓] Training und Evaluation abgeschlossen.")
