import pandas as pd
import numpy as np
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    cohen_kappa_score
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
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
        print(f"Feature-Masken-Plot gespeichert unter: {save_path}")
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


def train_and_evaluate():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_classic()
    feature_names = X_train.columns.tolist()

    print("[1] Starte Training...")
    clf = TabNetClassifier()
    clf.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=["val"],
        eval_metric=["accuracy"],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    print("[2] Evaluierung auf Testdaten...")
    y_pred = clf.predict(X_test.values)

    print("Klassische Metriken:")
    print(f"Accuracy:       {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision:      {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:         {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:       {f1_score(y_test, y_pred):.4f}")
    print(f"Cohen's Kappa:  {cohen_kappa_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("[3] Speicher Explainability-Plots...")
    plot_tabnet_feature_importance(
        clf,
        feature_names,
        save_path="/home/lola/ai_intrusion_detection_system/reports/figures/tabnet_feature_masks.png"
    )

    X_test_np = X_test.astype(np.float32).values
    y_test_np = y_test.values

    compute_and_plot_permutation_importance(
        clf,
        X_test_np,
        y_test_np,
        feature_names,
        save_path="/home/lola/ai_intrusion_detection_system/reports/figures/permutation_importance.png"
    )

    print("[✓] Training und Evaluation abgeschlossen.")