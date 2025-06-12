from glob import glob
import json
import os
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, 
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score
)
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
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

import os
import numpy as np
import matplotlib.pyplot as plt

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

    # Plot
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

def plot_feature_mask_heatmap(df_masks, df_significance, save_dir):
    df_all = df_masks.merge(df_significance, on="feature").set_index("feature")

    annot = pd.DataFrame(index=df_all.index)
    for cls in ["tp", "fn", "fp", "tn"]:
        mean = df_all[f"mean_{cls}"]
        std = df_all[f"std_{cls}"]
        annot[cls.upper()] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean, std)]

    heatmap_data = df_all[[f"mean_tp", f"mean_fn", f"mean_fp", f"mean_tn"]]
    heatmap_data.columns = ["TP", "FN", "FP", "TN"]

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, annot=annot, fmt="", cmap="Blues", linewidths=0.5,
                     cbar_kws={"label": "Feature Importance (Mean)"})
    plt.ylabel("Feature")
    plt.xlabel("Fehlerklasse")

    # Signifikanzrahmen einzeichnen
    sig_map = {
        (i, 1): sig for i, sig in enumerate(df_all["significant_fn_tp"])  # FN
    }
    sig_map.update({
        (i, 2): sig for i, sig in enumerate(df_all["significant_fp_tn"])  # FP
    })
    for (y, x), sig in sig_map.items():
        if sig:
            ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, "feature_mask_heatmap_significance.png")
    plt.savefig(heatmap_path)
    plt.close()

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

    plot_feature_mask_heatmap(df_masks, df_combined, save_dir)
    print("[✓] Alle Feature-Masken und Signifikanzplots wurden gespeichert.")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Violinplot der Streuung pro Fehlerklasse
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="error_type", y="proba_std", inner="box", cut=0)
    plt.ylabel("Standardabweichung der Modell-Vorhersagen")
    plt.xlabel("Fehlerklasse")
    plt.tight_layout()
    path_violin = os.path.join(save_dir, "ensemble_std_per_error_type_violin.png")
    plt.savefig(path_violin)
    plt.close()
    print(f"[✓] Violinplot gespeichert unter: {path_violin}")

def run_final_test_model_ensemble(alpha=2, beta=1):
    print("[✓] Lade Holdout-Testdaten...")
    df_test = load_pickle("data/processed/test_holdout.pkl")
    y_test = df_test["attack_detected"]
    X_test = df_test.drop(columns=["attack_detected"])

    print("[✓] Suche Ensemble-Metadaten...")
    matches = sorted(glob("models/tabnet_cv_*/metadata.json"))
    if not matches:
        raise FileNotFoundError("Keine metadata.json gefunden unter models/tabnet_cv_*/")
    metadata_path = matches[-1]

    with open(metadata_path) as f:
        meta = json.load(f)

    model_paths = meta["model_paths"]
    models = []
    print("[✓] Lade Ensemble-Modelle...")
    for path in model_paths:
        model_file = f"{path}.zip"
        if not Path(model_file).exists():
            raise FileNotFoundError(f"Modell-Datei fehlt: {model_file}")
        clf = TabNetClassifier()
        clf.load_model(model_file)
        models.append(clf)

    print(f"[✓] {len(models)} Fold-Modelle erfolgreich geladen.")

    print("[✓] Ensemble-Inferenz...")
    y_proba_matrix = np.array([model.predict_proba(X_test.values)[:, 1] for model in models])
    y_proba = y_proba_matrix.mean(axis=0)
    y_pred = (y_proba > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    pos_total = tp + fn
    neg_total = tn + fp
    fn_rate = fn / pos_total if pos_total > 0 else 0
    fp_rate = fp / neg_total if neg_total > 0 else 0
    cost = alpha * fn_rate + beta * fp_rate

    metrics_raw = {
        "cost": cost,
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba)
    }

    metrics_rounded = {k: round(v, 4) for k, v in metrics_raw.items()}

    pd.DataFrame.from_dict(metrics_rounded, orient="index", columns=["Wert"]).to_csv(
        "reports/final_test_metrics_ensemble.csv"
    )
    print("[✓] Test-Metriken gespeichert unter: reports/final_test_metrics_ensemble.csv")

    print("[✓] Speichere Confusion Matrix...")
    save_confusion_matrix(y_test, y_pred, "reports/figures/confusion_matrix_test_ensemble.png")

    print("[✓] Starte Feature-Mask-Analyse...")
    analyze_feature_masks(models, X_test, y_test.values, y_pred)

    print("[✓] Starte Konsistenzanalyse des Ensembles...")
    analyze_ensemble_uncertainty(models, X_test, y_test.values, y_pred)

    print("[✓] Testevaluation des Ensembles abgeschlossen.")