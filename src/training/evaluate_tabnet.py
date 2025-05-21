import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

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

def save_instance_level_explanations(clf, X, y_proba, y_pred, feature_names, threshold, save_path):
    feature_masks, _ = clf.explain(X.values)  # <-- Fix: Tuple entpacken
    instance_feature_importances = feature_masks.mean(axis=1)

    top_k = 5
    results = []

    for idx, (proba, pred, row) in enumerate(zip(y_proba, y_pred, instance_feature_importances)):
        top_features = [feature_names[i] for i in np.argsort(row)[::-1][:top_k]]
        results.append({
            "index": int(idx),
            "predicted_proba": float(proba),
            "predicted_label": int(pred),
            "threshold": float(threshold),
            "top_features": top_features
        })

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Erklärungen gespeichert unter: {save_path}")


def evaluate_tabnet_model(clf, threshold, X_test, y_test, feature_names):
    y_proba_test = clf.predict_proba(X_test.values)[:, 1]
    y_pred_test = (y_proba_test > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    save_metrics_as_table(acc, prec, rec, f1, kappa, "reports/figures/classic_metrics_test.png")
    save_confusion_matrix(y_test, y_pred_test, "reports/figures/confusion_matrix_test.png")

    print("\nFinale Evaluation auf dem Testset:")
    print(classification_report(y_test, y_pred_test))

    plot_tabnet_feature_importance(clf, feature_names, save_path="reports/figures/tabnet_feature_masks_test.png")
    compute_and_plot_permutation_importance(clf, X_test, y_test, feature_names, save_path="reports/figures/permutation_importance_test.png")

    save_instance_level_explanations(
        clf, X_test, y_proba_test, y_pred_test, feature_names, threshold,
        save_path="reports/feature_reasons/test_predictions_with_explanations.json"
    )