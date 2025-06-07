from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    cohen_kappa_score
)
from src.utils.io import load_pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_dummy_baseline(save_path="reports/dummy_metrics.csv",
                       matrix_path="reports/figures/dummy_confusion_matrix.png",
                       strategy="stratified",
                       alpha=2, beta=1):
    
    print(f"[✓] Starte Dummy-Baseline mit Strategie: {strategy}")

    df_test = load_pickle("data/processed/test_holdout.pkl")
    y_test = df_test["attack_detected"]
    X_test = df_test.drop(columns=["attack_detected"])

    dummy = DummyClassifier(strategy=strategy, random_state=42)
    dummy.fit(X_test, y_test)
    y_pred = dummy.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    pos_total = tp + fn
    neg_total = tn + fp

    fn_rate = fn / pos_total if pos_total > 0 else 0
    fp_rate = fp / neg_total if neg_total > 0 else 0

    cost = alpha * fn_rate + beta * fp_rate

    y_proba = dummy.predict_proba(X_test)[:, 1]

    # Metriken berechnen
    metrics = {
        "strategy": strategy,
        "cost": round(cost, 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "kappa": round(cohen_kappa_score(y_test, y_pred), 4),
    }

    df_metrics = pd.DataFrame([metrics])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(save_path, index=False)
    print(f"[✓] Dummy-Ergebnisse gespeichert unter: {save_path}")
    print(df_metrics.to_string(index=False))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    disp.ax_.set_title("Dummy Confusion Matrix")
    plt.tight_layout()
    Path(matrix_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(matrix_path)
    print(f"[✓] Confusion Matrix gespeichert unter: {matrix_path}")
    plt.close()
