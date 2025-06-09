from glob import glob
import json
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

    print("[✓] Testevaluation des Ensembles abgeschlossen.")