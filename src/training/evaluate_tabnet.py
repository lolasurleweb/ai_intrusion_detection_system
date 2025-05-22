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
import shap

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

def compute_and_plot_permutation_importance(clf, X_val, y_val, feature_names, save_path=None, scoring='recall'):
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

def explain_instance_tabnet_local(model, instance: pd.DataFrame, feature_names: list, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    model.device = "cpu"
    model.forward_masks = True
    model.eval()

    instance_np = instance.values.astype(np.float32)
    _ = model.predict(instance_np)

    if not hasattr(model, "explain_outputs") or model.explain_outputs is None:
        raise RuntimeError("TabNet explain_outputs wurde nicht gesetzt. Sicherstellen, dass model.forward_masks=True und model.predict() verwendet wird.")

    # explain_outputs ist ein Tupel (masks, masks_for_each_step)
    decision_masks = model.explain_outputs[0]  # Shape: (1, n_steps, n_features)
    aggregate_mask = np.sum(decision_masks, axis=1).squeeze()  # Shape: (n_features,)

    mask_norm = aggregate_mask / aggregate_mask.sum()

    sorted_idx = np.argsort(mask_norm)[::-1]
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_importances = mask_norm[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel("Relative Importance (TabNet Feature Mask)")
    plt.title("Lokale Erklärung für 1 Instanz")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[✓] Lokale Erklärung gespeichert unter: {save_path}")
        plt.close()
    else:
        plt.show()

    return sorted_features, sorted_importances



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

    print("[✓] Erzeuge lokale Erklärung für eine Beispielinstanz...")
    suspicious_idx = np.argmax(y_proba_test)  # höchste Angriffswahrscheinlichkeit
    suspicious_instance = X_test.iloc[[suspicious_idx]]
    explain_instance_tabnet_local(
        clf,
        suspicious_instance,
        feature_names,
        save_path="reports/figures/local_explanation_example.png"
    )

