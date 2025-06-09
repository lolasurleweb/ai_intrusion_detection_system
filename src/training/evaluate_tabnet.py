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
from sklearn.manifold import TSNE

def save_confusion_matrix(y_true, y_pred, save_path):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    disp.ax_.set_title("")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"[\u2713] Confusion Matrix gespeichert unter: {save_path}")
    plt.close()


def plot_bar_and_violin(df, save_path="reports/figures/cv_summary_bar_and_violin.png"):
    metrics = [col for col in df.columns if col != "fold"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sns.violinplot(data=df, y=metric, ax=ax, inner="box", linewidth=1)
        ax.set_title(f"{metric} (Violin)")
        ax.set_ylabel(metric)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[\u2713] Violinplot gespeichert: {save_path}")
    plt.close()


def evaluate_cross_validation_results(
    fold_metrics,
    best_config=None,
    best_threshold=None,
    save_dir="reports/cv_summary"
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(fold_metrics)
    df.index.name = "fold"
    df.reset_index(inplace=True)
    df.to_csv(Path(save_dir) / "fold_metrics.csv", index=False)
    print(f"[\u2713] Fold-Metriken gespeichert unter: {save_dir}/fold_metrics.csv")

    summary = {
        "mean_cost": round(df["cost"].mean(), 4),
        "std_cost": round(df["cost"].std(), 4),
        "mean_f1": round(df["f1"].mean(), 4),
        "mean_auc": round(df["auc"].mean(), 4),
        "mean_precision": round(df["precision"].mean(), 4),
        "mean_recall": round(df["recall"].mean(), 4),
        "threshold": round(best_threshold, 4) if best_threshold is not None else None,
        "best_config": best_config,
        "n_folds": len(df),
        "timestamp": datetime.now().isoformat()
    }

    with open(Path(save_dir) / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[\u2713] Summary gespeichert unter: {save_dir}/cv_summary.json")

    plot_bar_and_violin(df.drop(columns="fold"), save_path=Path(save_dir) / "cv_metrics_violin.png")

def plot_training_history(clf, save_path=None):
    history = clf.history.history

    train_loss = history.get("train_logloss")
    val_cost = history.get("val_cost_metric")

    assert isinstance(train_loss, list), "train_logloss nicht vorhanden oder kein list"
    if val_cost is not None:
        assert isinstance(val_cost, list), "val_cost_metric nicht vom Typ list"

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train LogLoss")
    if val_cost:
        plt.plot(epochs, val_cost, label="Validation Cost")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Trainingsverlauf")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[✓] Trainingsverlauf gespeichert: {save_path}")
    else:
        plt.show()

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
            "top_features": top_features
        })

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[✓] Lokale Erklärungen gespeichert unter: {save_path}")

def analyze_model_errors(clf, X_test, y_test, y_pred, y_proba, feature_names,
                         explanation_dir="reports/explanations/",
                         figure_dir="reports/figures/",
                         top_k_features=5):

    Path(explanation_dir).mkdir(parents=True, exist_ok=True)
    Path(figure_dir).mkdir(parents=True, exist_ok=True)

    mask_fn = (y_test == 1) & (y_pred == 0)
    mask_fp = (y_test == 0) & (y_pred == 1)

    save_instance_level_explanations(
        clf=clf,
        X=X_test[mask_fn],
        y_proba=y_proba[mask_fn],
        y_pred=y_pred[mask_fn],
        feature_names=feature_names,
        save_path=f"{explanation_dir}/false_negatives.json",
        top_k=top_k_features,
        only_positive_predictions=False
    )

    save_instance_level_explanations(
        clf=clf,
        X=X_test[mask_fp],
        y_proba=y_proba[mask_fp],
        y_pred=y_pred[mask_fp],
        feature_names=feature_names,
        save_path=f"{explanation_dir}/false_positives.json",
        top_k=top_k_features,
        only_positive_predictions=False
    )

    print("[✓] Lokale Erklärungen für FP und FN gespeichert.")

    feature_importances = clf.explain(X_test.values.astype(np.float32))[0].mean(axis=0)
    top_k_idx = np.argsort(feature_importances)[::-1][:top_k_features]
    top_k_feature_names = [feature_names[i] for i in top_k_idx]

    df = X_test.copy()
    df["y_true"] = y_test
    df["y_pred"] = y_pred

    df["Fehlerklasse"] = "TN"
    df.loc[(y_test == 1) & (y_pred == 1), "Fehlerklasse"] = "TP"
    df.loc[(y_test == 1) & (y_pred == 0), "Fehlerklasse"] = "FN"
    df.loc[(y_test == 0) & (y_pred == 1), "Fehlerklasse"] = "FP"

    for feature in top_k_feature_names:
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=df, x="Fehlerklasse", y=feature, palette="Set2", inner="box")
        plt.title(f"Verteilung von '{feature}' nach Fehlerklasse")
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/error_violin_{feature}.png")
        plt.close()
        print(f"[✓] Violinplot gespeichert: {figure_dir}/error_violin_{feature}.png")

    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_embedded = tsne.fit_transform(X_test.values.astype(np.float32))

        df_tsne = pd.DataFrame(X_embedded, columns=["TSNE-1", "TSNE-2"])
        df_tsne["Fehlerklasse"] = df["Fehlerklasse"]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_tsne, x="TSNE-1", y="TSNE-2", hue="Fehlerklasse", alpha=0.8, palette="Set1")
        plt.title("t-SNE Projektion nach Fehlerklasse")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/tsne_error_projection.png")
        plt.close()
        print(f"[✓] t-SNE Visualisierung gespeichert: {figure_dir}/tsne_error_projection.png")

    except Exception as e:
        print(f"[!] t-SNE konnte nicht berechnet werden: {e}")

def run_final_test_model():
    print("[✓] Lade Holdout-Testdaten...")
    df_test = load_pickle("data/processed/test_holdout.pkl")
    y_test = df_test["attack_detected"]
    X_test = df_test.drop(columns=["attack_detected"])

    print("[✓] Lade finales Modell...")
    metadata_path = "models/final_model_metadata.json"
    if Path(metadata_path).exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        model_path = meta["model_path"]
    else:
        raise FileNotFoundError("Metadata-Datei zum Laden des finalen Modells fehlt.")

    clf = TabNetClassifier()
    clf.load_model(model_path + ".zip")

    y_proba = clf.predict_proba(X_test.values)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)  # Standard-Threshold

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    alpha, beta = 2, 1
    pos_total = tp + fn
    neg_total = tn + fp

    fn_rate = fn / pos_total if pos_total > 0 else 0
    fp_rate = fp / neg_total if neg_total > 0 else 0
    cost = alpha * fn_rate + beta * fp_rate

    metrics = {
        "cost": cost,
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4)
    }

    pd.DataFrame.from_dict(metrics, orient="index", columns=["Wert"]).to_csv(
        "reports/final_test_metrics.csv")
    print("[✓] Test-Metriken gespeichert unter: reports/final_test_metrics.csv")

    print("[✓] Speichere Confusion Matrix...")
    save_confusion_matrix(y_test, y_pred, "reports/figures/confusion_matrix_test.png")

    print("[✓] Visualisiere Feature-Masken...")
    plot_tabnet_feature_importance(clf, X_test, X_test.columns.tolist(), 
                                   save_path="reports/figures/tabnet_feature_masks_test.png")

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
        save_path="reports/explanations/test_instance_level.json",
        top_k=5,
        only_positive_predictions=True,
        include_scores=True
    )

    print("[✓] Visualisiere Grenzen des Modells")
    analyze_model_errors(
        clf=clf,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        feature_names=X_test.columns.tolist(),
        top_k_features=5
    )

    print("[✓] Testevaluation abgeschlossen.")