from glob import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from matplotlib.patches import Rectangle
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_auc_score, accuracy_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

from pytorch_tabnet.tab_model import TabNetClassifier
from src.utils.io import load_pickle

def save_confusion_matrix(y_true, y_pred, save_path):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    disp.ax_.set_title("")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.savefig(save_path)
    print(f"[✓] Confusion Matrix gespeichert unter: {save_path}")
    plt.close(fig)

def compute_null_distribution(mean_mask, labels, n_iter=1000, seed=42):
    np.random.seed(seed)
    null_diffs = []

    for _ in range(n_iter):
        permuted = np.random.permutation(labels)
        group1 = mean_mask[permuted == 0]
        group2 = mean_mask[permuted == 1]

        if len(group1) == 0 or len(group2) == 0:
            continue

        diff = group1.mean(axis=0) - group2.mean(axis=0)
        null_diffs.append(diff)

    return np.array(null_diffs)

def compute_feature_mask_statistics(mean_mask, idx, feature_names):
    if len(idx) == 0:
        return np.zeros(len(feature_names)), np.zeros(len(feature_names))
    subset = mean_mask[idx]
    return subset.mean(axis=0), subset.std(axis=0)

def compute_significance(mask_group1, mask_group2, feature_names, diff_label, save_dir):
    labels = np.array([0] * len(mask_group1) + [1] * len(mask_group2))
    masks_combined = np.vstack([mask_group1, mask_group2])

    null_diffs = compute_null_distribution(masks_combined, labels, n_iter=1000)
    lower_ci = np.percentile(null_diffs, 2.5, axis=0)
    upper_ci = np.percentile(null_diffs, 97.5, axis=0)

    real_diff = mask_group2.mean(axis=0) - mask_group1.mean(axis=0)
    significant = (real_diff < lower_ci) | (real_diff > upper_ci)

    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_names,
        real_diff,
        color=["#1f77b4" if sig else "#d3d3d3" for sig in significant]
    )
    plt.axvline(0, color='gray', linestyle='--')
    plt.tight_layout()
    filepath = os.path.join(save_dir, f"diff_{diff_label.replace(' ', '_').lower()}_significant.png")
    plt.savefig(filepath)
    plt.close()

    return pd.DataFrame({
        "feature": feature_names,
        f"diff_{diff_label.lower().replace(' ', '_')}": real_diff,
        f"ci_lower_{diff_label.lower().replace(' ', '_')}": lower_ci,
        f"ci_upper_{diff_label.lower().replace(' ', '_')}": upper_ci,
        f"significant_{diff_label.lower().replace(' ', '_')}": significant
    })

def plot_feature_mask_bar(mean_mask, std_mask, cls_name, feature_names, save_dir):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, mean_mask, xerr=std_mask, ecolor="red", capsize=4)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f"featuremask_{cls_name.lower()}.png")
    plt.savefig(filepath)
    plt.close()

def plot_feature_mask_heatmap(df_masks, save_dir, std_threshold=0.5):
    df_all = df_masks.set_index("feature")

    annot = pd.DataFrame(index=df_all.index)
    for cls in ["tp", "fn", "fp", "tn"]:
        mean = df_all[f"mean_{cls}"]
        std = df_all[f"std_{cls}"]
        annot[cls.upper()] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean, std)]

    heatmap_data = df_all[[f"mean_tp", f"mean_fn", f"mean_fp", f"mean_tn"]]
    heatmap_data.columns = ["TP", "FN", "FP", "TN"]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        heatmap_data,
        annot=annot,
        fmt="",
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Feature Importance (Mean)"}
    )
    plt.ylabel("Feature")
    plt.xlabel("Fehlerklasse")

    for row_idx, feature in enumerate(df_all.index):
        for col_idx, cls in enumerate(["tp", "fn", "fp", "tn"]):
            std_val = df_all.loc[feature, f"std_{cls}"]
            if std_val > std_threshold:
                ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "feature_mask_heatmap_instability.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"[✓] Instabilitäts-Heatmap gespeichert unter: {heatmap_path}")

def analyze_feature_masks(models, X_test, y_test, y_pred, save_dir="reports/figures/feature_masks"):
    os.makedirs(save_dir, exist_ok=True)
    feature_names = X_test.columns.tolist()

    all_masks = [model.explain(X_test.values.astype(np.float32))[0] for model in models]
    mean_mask = np.mean(all_masks, axis=0)

    y_test = np.array(y_test)
    idx = {
        "tp": np.where((y_test == 1) & (y_pred == 1))[0],
        "fp": np.where((y_test == 0) & (y_pred == 1))[0],
        "fn": np.where((y_test == 1) & (y_pred == 0))[0],
        "tn": np.where((y_test == 0) & (y_pred == 0))[0]
    }

    stats = {
        cls: compute_feature_mask_statistics(mean_mask, idx[cls], feature_names)
        for cls in idx
    }

    df_masks = pd.DataFrame({
        "feature": feature_names,
        **{f"mean_{cls}": stats[cls][0] for cls in stats},
        **{f"std_{cls}": stats[cls][1] for cls in stats},
    })

    df_fn_tp = compute_significance(mean_mask[idx["tp"]], mean_mask[idx["fn"]], feature_names, "FN TP", save_dir)
    df_fp_tn = compute_significance(mean_mask[idx["tn"]], mean_mask[idx["fp"]], feature_names, "FP TN", save_dir)

    df_combined = df_fn_tp.merge(df_fp_tn, on="feature")
    df_masks.to_csv(os.path.join(save_dir, "feature_mask_means_stds.csv"), index=False)
    df_combined.to_csv(os.path.join(save_dir, "feature_mask_significance.csv"), index=False)

    for cls in stats:
        plot_feature_mask_bar(stats[cls][0], stats[cls][1], cls.upper(), feature_names, save_dir)

    plot_feature_mask_heatmap(df_masks, save_dir)
    print("[✓] Alle Feature-Masken und Signifikanzplots wurden gespeichert.")

def analyze_ensemble_uncertainty(models, X_test, y_test, y_pred, save_dir="reports/figures/ensemble_uncertainty"):
    os.makedirs(save_dir, exist_ok=True)

    print("[✓] Berechne predict_proba pro Fold-Modell...")
    probas = np.array([model.predict_proba(X_test.values)[:, 1] for model in models])  # [n_models, n_samples]

    print("[✓] Berechne Streuung und Mittelwert je Instanz...")
    proba_std = np.std(probas, axis=0)    # [n_samples]
    proba_mean = np.mean(probas, axis=0)  # [n_samples]

    df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "proba_mean": proba_mean,
        "proba_std": proba_std
    })

    df["error_type"] = "Correct"
    df.loc[(df.y_true == 1) & (df.y_pred == 0), "error_type"] = "False Negative"
    df.loc[(df.y_true == 0) & (df.y_pred == 1), "error_type"] = "False Positive"

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="error_type", y="proba_std", inner="box", cut=0)
    plt.ylabel("Standardabweichung der Modell-Vorhersagen")
    plt.xlabel("Fehlerklasse")
    plt.tight_layout()
    path_violin = os.path.join(save_dir, "ensemble_std_per_error_type_violin.png")
    plt.savefig(path_violin)
    plt.close()
    print(f"[✓] Violinplot gespeichert unter: {path_violin}")

def plot_3d_interactive(data_3d, y_true, y_pred, save_path="tsne_3d_interactive.html"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = []
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            labels.append("TP")
        elif yt == 1 and yp == 0:
            labels.append("FN")
        elif yt == 0 and yp == 1:
            labels.append("FP")
        else:
            labels.append("TN")

    df = pd.DataFrame(data_3d, columns=["Dim 1", "Dim 2", "Dim 3"])
    df["Label"] = labels

    color_map = {
        "TP": "#4C72B0",
        "FN": "#55A868",
        "FP": "#C44E52",
        "TN": "#DD8452"
    }

    fig = px.scatter_3d(
        df,
        x="Dim 1",
        y="Dim 2",
        z="Dim 3",
        color="Label",
        color_discrete_map=color_map,
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(title="Interaktives t-SNE 3D Embedding")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"[✓] Interaktiver t-SNE-Plot gespeichert unter: {save_path}")

def visualize_embeddings_3d(mean_mask, y_true, y_pred, save_dir="reports/figures/embeddings_3d"):
    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    label_colors = {
    "TP": "#4C72B0",     
    "FN": "#55A868",     
    "FP": "#C44E52",    
    "TN": "#DD8452"    
    }
    
    def plot_3d(data_3d, title, name):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        for label in ["TP", "FN", "FP", "TN"]:
            color = label_colors[label]
            if label == "TP":
                idx = np.where((y_true == 1) & (y_pred == 1))
            elif label == "FN":
                idx = np.where((y_true == 1) & (y_pred == 0))
            elif label == "FP":
                idx = np.where((y_true == 0) & (y_pred == 1))
            else:  # TN
                idx = np.where((y_true == 0) & (y_pred == 0))

            ax.scatter(
                data_3d[idx, 0],
                data_3d[idx, 1],
                data_3d[idx, 2],
                label=label,
                alpha=0.6,
                s=30,
                c=color
            )

        ax.set_title(title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"embedding_3d_{name}.png"))
        plt.close()

    print("[✓] Starte PCA 3D...")
    pca_3d = PCA(n_components=3).fit_transform(mean_mask)
    plot_3d(pca_3d, "PCA 3D Embedding", "pca")

    print("[✓] Starte t-SNE 3D...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30).fit_transform(mean_mask)
    plot_3d(tsne_3d, "t-SNE 3D Embedding", "tsne")

    print("[✓] Starte UMAP 3D...")
    reducer = umap.UMAP(n_components=3, random_state=42)
    umap_3d = reducer.fit_transform(mean_mask)
    plot_3d(umap_3d, "UMAP 3D Embedding", "umap")

    print("[✓] Starte t-SNE 3D (Interaktive Version)...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30).fit_transform(mean_mask)
    plot_3d(tsne_3d, "t-SNE 3D Embedding", "tsne")
    plot_3d_interactive(tsne_3d, y_true, y_pred, save_path=os.path.join(save_dir, "tsne_3d_interactive.html"))

    print(f"[✓] 3D-Visualisierungen gespeichert unter: {save_dir}")

def compare_fn_tp_feature_means(X_test, y_true, y_pred, save_path="reports/figures/fn_tp_feature_diff.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fn_idx = (y_true == 1) & (y_pred == 0)
    tp_idx = (y_true == 1) & (y_pred == 1)

    if fn_idx.sum() == 0 or tp_idx.sum() == 0:
        print("[!] Nicht genügend FN oder TP vorhanden.")
        return

    df_fn = X_test.loc[fn_idx]
    df_tp = X_test.loc[tp_idx]

    fn_means = df_fn.mean()
    tp_means = df_tp.mean()
    diff = fn_means - tp_means

    df_plot = pd.DataFrame({
        "FN mean": fn_means,
        "TP mean": tp_means,
        "Difference (FN - TP)": diff
    }).sort_values("Difference (FN - TP)", key=np.abs, ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    df_plot["Difference (FN - TP)"].plot(kind="barh", color="#4C72B0")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Differenz")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] FN vs. TP Featurevergleich gespeichert unter: {save_path}")

def plot_tp_kmeans_clusters_interactive(tsne_3d, y_true, y_pred, cluster_labels, save_path="reports/figures/embeddings_3d/tp_kmeans_clusters_3d.html"):
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    if len(tp_idx) != len(cluster_labels):
        print("[!] Länge der Clusterlabels stimmt nicht mit TP-Anzahl überein.")
        return

    df = pd.DataFrame(tsne_3d[tp_idx], columns=["Dim 1", "Dim 2", "Dim 3"])
    df["Cluster"] = cluster_labels.astype(str)

    fig = px.scatter_3d(
        df,
        x="Dim 1", y="Dim 2", z="Dim 3",
        color="Cluster",
        title="True Positives – KMeans Cluster im t-SNE-Raum",
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=4))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"[✓] Interaktive TP-KMeans-Cluster-Visualisierung gespeichert unter: {save_path}")

def analyze_tp_clusters(tsne_3d, y_true, y_pred, X_test, save_path=None):
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    if len(tp_idx) == 0:
        print("[!] Keine True Positives vorhanden.")
        return

    tp_embeds = tsne_3d[tp_idx]
    scaled_tp = StandardScaler().fit_transform(tp_embeds)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(scaled_tp)

    plot_tp_kmeans_clusters_interactive(
    tsne_3d=tsne_3d,
    y_true=y_true,
    y_pred=y_pred,
    cluster_labels=labels,
    save_path="reports/figures/embeddings_3d/tp_kmeans_clusters_3d.html"
    )

    df_tp = X_test.iloc[tp_idx].copy()
    df_tp["cluster"] = labels

    mean_df = df_tp.groupby("cluster").mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(mean_df.T, cmap="viridis", annot=True, fmt=".2f")
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[✓] KMeans TP-Cluster-Heatmap gespeichert unter: {save_path}")
    else:
        plt.show()

    plt.close()

def run_final_test_model_ensemble(alpha=2, beta=1):
    print("[\u2713] Lade Holdout-Testdaten...")
    df_test = load_pickle("data/processed/test_holdout.pkl")
    y_test = df_test["attack_detected"]
    X_test = df_test.drop(columns=["attack_detected"])

    print("[\u2713] Suche Ensemble-Metadaten...")
    metadata_path = sorted(glob("models/tabnet_cv_*/metadata.json"))[-1]
    with open(metadata_path) as f:
        meta = json.load(f)

    model_paths = meta["model_paths"]
    models = []

    print("[\u2713] Lade Ensemble-Modelle...")
    for path in model_paths:
        model_file = f"{path}.zip"
        clf = TabNetClassifier()
        clf.load_model(model_file)
        models.append(clf)

    print(f"[\u2713] {len(models)} Fold-Modelle erfolgreich geladen.")

    print("[\u2713] Ensemble-Inferenz...")
    y_proba_matrix = np.array([model.predict_proba(X_test.values)[:, 1] for model in models])
    y_proba = y_proba_matrix.mean(axis=0)
    y_pred = (y_proba > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
    cost = alpha * fn_rate + beta * fp_rate

    metrics = {
        "cost": cost,
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }

    pd.DataFrame.from_dict({k: [round(v, 4)] for k, v in metrics.items()}).T.to_csv(
        "reports/final_test_metrics_ensemble.csv", header=["Wert"]
    )
    print("[\u2713] Test-Metriken gespeichert.")

    print("[\u2713] Speichere Confusion Matrix...")
    save_confusion_matrix(y_test, y_pred, "reports/figures/confusion_matrix_test_ensemble.png")

    print("[\u2713] Starte Feature-Mask-Analyse...")
    analyze_feature_masks(models, X_test, y_test.values, y_pred)

    print("[\u2713] Starte Konsistenzanalyse des Ensembles...")
    analyze_ensemble_uncertainty(models, X_test, y_test.values, y_pred)

    print("[\u2713] Berechne mittlere Feature-Masken...")
    mean_mask = np.mean([model.explain(X_test.values.astype(np.float32))[0] for model in models], axis=0)

    print("[\u2713] Starte t-SNE 3D...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30).fit_transform(mean_mask)

    plot_3d_interactive(tsne_3d, y_test, y_pred, save_path="reports/figures/embeddings_3d/tsne_3d_interactive.html")

    print("[\u2713] Starte TP-Clusteranalyse...")
    analyze_tp_clusters(
        tsne_3d, y_test.values, y_pred, X_test,
        save_path="reports/figures/tp_clusters_tsne_kmeans.png"
    )

    print("[\u2713] Starte Vergleich FN vs. TP...")
    compare_fn_tp_feature_means(X_test, y_test, y_pred)

    print("[\u2713] Testevaluation abgeschlossen.")