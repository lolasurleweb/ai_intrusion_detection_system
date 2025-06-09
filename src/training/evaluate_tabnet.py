from glob import glob
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def analyze_feature_masks(models, X_test, y_test, y_pred, save_dir="reports/figures/feature_masks"):
    os.makedirs(save_dir, exist_ok=True)
    feature_names = X_test.columns.tolist()

    # Schritt 1: Ensemble-Mittelwert der Feature Masks sammeln
    all_masks = []
    for model in models:
        mask, _ = model.explain(X_test.values.astype(np.float32))  # Achtung: explain() statt predict()
        all_masks.append(mask)

    mean_mask = np.mean(all_masks, axis=0)  # shape: [n_samples, n_features]

    # Schritt 2: Fehlerklassen-Indizes
    y_test = np.array(y_test)
    tp_idx = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]
    tn_idx = np.where((y_test == 0) & (y_pred == 0))[0]

    # Schritt 3: Klassenspezifische Mittelwerte
    mean_mask_tp = mean_mask[tp_idx].mean(axis=0) if len(tp_idx) > 0 else np.zeros(len(feature_names))
    mean_mask_fp = mean_mask[fp_idx].mean(axis=0) if len(fp_idx) > 0 else np.zeros(len(feature_names))
    mean_mask_fn = mean_mask[fn_idx].mean(axis=0) if len(fn_idx) > 0 else np.zeros(len(feature_names))
    mean_mask_tn = mean_mask[tn_idx].mean(axis=0) if len(tn_idx) > 0 else np.zeros(len(feature_names))

    # Schritt 4: Differenzplots
    def plot_difference(diff, title, filename):
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, diff)
        plt.title(title)
        plt.axvline(0, color='gray', linestyle='--')
        plt.tight_layout()
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"[✓] Plot gespeichert unter: {filepath}")

    plot_difference(mean_mask_fn - mean_mask_tp,
                    "Feature-Mask-Differenz (False Negatives - True Positives)",
                    "diff_fn_vs_tp.png")

    plot_difference(mean_mask_fp - mean_mask_tn,
                    "Feature-Mask-Differenz (False Positives - True Negatives)",
                    "diff_fp_vs_tn.png")

    # Schritt 5: Absolute Klassenmasken
    for cls_name, mask in zip(
        ["TP", "FP", "FN", "TN"],
        [mean_mask_tp, mean_mask_fp, mean_mask_fn, mean_mask_tn]
    ):
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, mask)
        plt.title(f"Durchschnittliche Feature-Masken für {cls_name}")
        plt.tight_layout()
        filepath = os.path.join(save_dir, f"featuremask_{cls_name.lower()}.png")
        plt.savefig(filepath)
        plt.close()
        print(f"[✓] Plot gespeichert unter: {filepath}")

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

    print("[✓] Testevaluation des Ensembles abgeschlossen.")