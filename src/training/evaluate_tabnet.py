import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance


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


def save_instance_level_explanations(
    clf,
    X,
    y_proba,
    y_pred,
    feature_names,
    threshold,
    save_path,
    top_k=5,
    only_positive_predictions=False,
    include_scores=True
):
    
    feature_masks, _ = clf.explain(X.values.astype(np.float32))  
    instance_feature_importances = feature_masks 

    results = []

    for idx, (proba, pred, importance_vector) in enumerate(zip(y_proba, y_pred, instance_feature_importances)):
        if only_positive_predictions and pred != 1:
            continue

        sorted_idx = np.argsort(importance_vector)[::-1][:top_k]

        if include_scores:
            top_features = [
                {"feature": feature_names[i], "importance": float(round(importance_vector[i], 5))}
                for i in sorted_idx
            ]
        else:
            top_features = [feature_names[i] for i in sorted_idx]

        results.append({
            "index": int(idx),
            "predicted_proba": float(round(proba, 5)),
            "predicted_label": int(pred),
            "threshold": float(round(threshold, 5)),
            "top_features": top_features
        })

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✓] Lokale Erklärungen gespeichert unter: {save_path}")

def evaluate_cross_validation_results(fold_metrics, save_dir="reports/cv_summary"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(fold_metrics)
    df_mean_std = df.agg(['mean', 'std']).T.round(4)
    df_mean_std.columns = ['Mittelwert', 'Standardabweichung']
    df_mean_std.index.name = 'Metrik'

    df_mean_std.to_csv(Path(save_dir) / "cv_metrics.csv")
    print(f"[✓] Cross-Validation-Metriken gespeichert unter: {save_dir}/cv_metrics.csv")


    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', figsize=(10, 6), title="Metriken pro Fold")
    plt.xticks(ticks=np.arange(len(df)), labels=[f"Fold {i+1}" for i in range(len(df))], rotation=0)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "metrics_per_fold.png")
    print(f"[✓] Plot gespeichert unter: {save_dir}/metrics_per_fold.png")
    plt.close()

    
    df.boxplot(figsize=(10, 6))
    plt.title("Verteilung der Metriken über Folds")
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "metrics_boxplot.png")
    print(f"[✓] Boxplot gespeichert unter: {save_dir}/metrics_boxplot.png")
    plt.close()
