import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance
import seaborn as sns
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetClassifier
from src.utils.io import load_pickle


def save_confusion_matrix(y_true, y_pred, save_path):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"[✓] Confusion Matrix gespeichert unter: {save_path}")
    plt.close()

def plot_bar_and_violin(fold_metrics_df, save_path="reports/figures/cv_summary_bar_and_violin.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Barplot: Kosten pro Fold
    fold_metrics_df.plot(kind="bar", ax=axes[0], title="Kosten pro Fold", legend=False)
    axes[0].set_xticks(range(len(fold_metrics_df)))
    axes[0].set_xticklabels([f"Fold {i+1}" for i in range(len(fold_metrics_df))])
    axes[0].set_ylabel("Cost")

    # Violinplot: Verteilung der Kosten
    sns.violinplot(data=fold_metrics_df, y="cost", ax=axes[1], inner="box", linewidth=1)
    axes[1].set_title("Kostenverteilung (Violin)")
    axes[1].set_ylabel("Cost")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✓] Kombiplot gespeichert: {save_path}")
    plt.close()

def evaluate_cross_validation_results(
    fold_metrics,
    best_config=None,
    best_threshold=None,
    save_dir="reports/cv_summary"
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. CSV mit Kosten pro Fold
    df = pd.DataFrame(fold_metrics)[["cost"]]
    df.index.name = "fold"
    df.to_csv(Path(save_dir) / "fold_costs.csv")
    print(f"[✓] Fold-Kosten gespeichert unter: {save_dir}/fold_costs.csv")

    # 2. Summary JSON mit Threshold, Config, Statistik
    summary = {
        "mean_cost": round(df["cost"].mean(), 4),
        "std_cost": round(df["cost"].std(), 4),
        "threshold": round(best_threshold, 4) if best_threshold is not None else None,
        "best_config": best_config,
        "n_folds": len(df),
        "timestamp": datetime.now().isoformat()
    }

    with open(Path(save_dir) / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Summary gespeichert unter: {save_dir}/cv_summary.json")

    # 3. Bar- + Violinplot
    plot_bar_and_violin(df, save_path=Path(save_dir) / "cv_metrics_combined_violin.png")


def plot_tabnet_feature_importance(clf, X, feature_names, save_path=None):
    importances = clf.explain(X.values.astype(np.float32))[0].mean(axis=0)
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

def plot_training_history(clf, save_path=None):
    history = clf.history
    epochs = range(1, len(history['loss']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss'], label="Train Loss")
    
    if 'val_cost_metric' in history:
        plt.plot(epochs, history['val_cost_metric'], label="Validation Cost")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Trainingsverlauf")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[✓] Trainingsverlauf gespeichert: {save_path}")
    else:
        plt.show()


def run_final_test_model():
    print("[✓] Lade Holdout-Testdaten...")
    df_test = load_pickle("data/processed/test_holdout.pkl")
    y_test = df_test["attack_detected"]
    X_test = df_test.drop(columns=["attack_detected"])

    print("[✓] Lade finales Modell und Threshold...")
    model_path = "models/tabnet_final_model_20250531-102653_f169c4.zip.zip"
    threshold_path = "models/tabnet_final_threshold_20250531-102653_f169c4.json"

    clf = TabNetClassifier()
    clf.load_model(model_path)


    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    print(f"[✓] Verwende Threshold: {threshold:.2f}")
    y_proba = clf.predict_proba(X_test.values)[:, 1]
    y_pred = (y_proba > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    alpha, beta = 2, 1
    cost = alpha * fn + beta * fp

    metrics = {
        "threshold": round(threshold, 4),
        "cost": cost,
        "f1": round(f1_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "kappa": round(cohen_kappa_score(y_test, y_pred), 4)
    }

    pd.DataFrame.from_dict(metrics, orient="index", columns=["Wert"]).to_csv(
        "reports/final_test_metrics.csv")
    print("[✓] Test-Metriken gespeichert unter: reports/final_test_metrics.csv")

    print("[✓] Speichere Confusion Matrix...")
    save_confusion_matrix(y_test, y_pred, "reports/figures/confusion_matrix_test.png")

    print("[✓] Visualisiere Feature-Masken...")
    plot_tabnet_feature_importance(clf, X_test, X_test.columns.tolist(), save_path="reports/figures/tabnet_feature_masks_test.png")


    print("[✓] Berechne Permutation Importance...")
    compute_and_plot_permutation_importance(
        clf, X_val=X_test, y_val=y_test,
        feature_names=X_test.columns.tolist(),
        save_path="reports/figures/permutation_importance_test.png",
        scoring='f1'
    )

    print("[✓] Speichere lokale Erklärungen für positive IDS-Warnungen...")
    save_instance_level_explanations(
        clf=clf,
        X=X_test,
        y_proba=y_proba,
        y_pred=y_pred,
        feature_names=X_test.columns.tolist(),
        threshold=threshold,
        save_path="reports/explanations/test_instance_level.json",
        top_k=5,
        only_positive_predictions=True,
        include_scores=True
    )

    print("[✓] Testevaluation abgeschlossen.")
