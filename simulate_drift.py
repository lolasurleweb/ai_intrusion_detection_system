from pytorch_tabnet.tab_model import TabNetClassifier
import json
from src.utils.io import load_time
from src.drift.drift_detection import PerformanceDriftDetector
from src.drift.replay_buffer import ReplayBuffer
from src.drift.finetuning import fine_tune_tabnet
from src.training.train_tabnet import compute_optimal_threshold_by_cost_function
from sklearn.metrics import accuracy_score, roc_auc_score
from src.data.preprocessing import NUMERICAL_FEATURES
import numpy as np
import pandas as pd
import joblib

def simulate_stream(X, y, batch_size=128):
    for i in range(0, len(X), batch_size):
        yield X.iloc[i:i+batch_size], y.iloc[i:i+batch_size]

def run_drift_simulation():
    print("[1] Lade Modell und Threshold...")
    clf = TabNetClassifier()
    clf.load_model("models/tabnet_model_20250522-144359_f81eba.zip.zip")

    with open("models/tabnet_threshold_20250522-144359_f81eba.json") as f:
        threshold = json.load(f)["threshold"]

    print("[2] Lade Zeitsplits...")
    (X_early, y_early), (X_mid, y_mid), (X_late, y_late) = load_time()

    X_stream = pd.concat([X_mid, X_late], axis=0).reset_index(drop=True)
    y_stream = pd.concat([y_mid, y_late], axis=0).reset_index(drop=True)

    scaler = joblib.load("data/processed/scaler.pkl")

    print("[3] Starte Drift-Überwachung...")
    drift_detector = PerformanceDriftDetector(acc_threshold=0.85)
    replay_buffer = ReplayBuffer(max_size=500)

    drift_events = 0
    fine_tune_events = 0
    reoptimize_every = 5
    retraining_counter = 0

    for i, (X_batch, y_batch) in enumerate(simulate_stream(X_stream, y_stream)):

        X_batch = X_batch.copy()
        X_batch[NUMERICAL_FEATURES] = scaler.transform(X_batch[NUMERICAL_FEATURES])

        y_proba = clf.predict_proba(X_batch.values)[:, 1]
        y_pred = (y_proba > threshold).astype(int)

        acc = accuracy_score(y_batch, y_pred)
        auc = roc_auc_score(y_batch, y_proba)

        print(f"[{i}] Accuracy={acc:.3f}, AUC={auc:.3f}")

        if drift_detector.update(acc, auc):
            print(f"Drift erkannt bei Batch {i}")
            drift_events += 1

            # Trenne TP und FP explizit
            predicted_positive = y_pred == 1
            tp_mask = (predicted_positive) & (y_batch == 1)
            fp_mask = (predicted_positive) & (y_batch == 0)

            X_tp = X_batch[tp_mask]
            y_tp = y_batch[tp_mask]

            X_fp = X_batch[fp_mask]
            y_fp = y_batch[fp_mask]

            X_alarm = pd.concat([X_tp, X_fp])
            y_alarm = pd.concat([y_tp, y_fp])

            print(f"[Batch {i}] Fine-Tune: {len(X_tp)} TP, {len(X_fp)} FP")

            if not X_alarm.empty:
                X_ft, y_ft = replay_buffer.sample(n_old=50, X_new=X_alarm, y_new=y_alarm)
                clf = fine_tune_tabnet(clf, X_ft, y_ft, epochs=10)
                replay_buffer.add_batch(X_alarm, y_alarm)

                fine_tune_events += 1
                retraining_counter += 1
                print(f"✓ Fine-Tuning durchgeführt bei Batch {i}")

                # Optional: Reoptimiere Threshold – nur wenn beide Klassen vorhanden sind
                if retraining_counter % reoptimize_every == 0:
                    if len(y_ft) >= 30 and y_ft.nunique() == 2:
                        y_ft_proba = clf.predict_proba(X_ft.values)[:, 1]
                        new_threshold, _ = compute_optimal_threshold_by_cost_function(
                            y_true=y_ft,
                            y_proba=y_ft_proba,
                            alpha=2,
                            beta=1
                        )
                        threshold = new_threshold
                        print(f"[✓] Threshold neu gesetzt auf {threshold:.3f}")
                    else:
                        print("[i] Threshold nicht angepasst: unzureichende Klassenvielfalt oder Samplegröße")

    print("\n[✓] Drift-Handling abgeschlossen.")
    print(f"Erkannte Drift-Warnungen: {drift_events}")
    print(f"Durchgeführte Fine-Tunings: {fine_tune_events}")

if __name__ == "__main__":
    run_drift_simulation()