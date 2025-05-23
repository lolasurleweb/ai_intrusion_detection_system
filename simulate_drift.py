from pytorch_tabnet.tab_model import TabNetClassifier
import json
from src.utils.io import load_time
from src.drift.drift_detection import PerformanceDriftDetector
from src.drift.replay_buffer import ReplayBuffer
from src.drift.finetuning import fine_tune_tabnet
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd

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

    print("[3] Starte Drift-Überwachung...")
    drift_detector = PerformanceDriftDetector(acc_threshold=0.85)
    replay_buffer = ReplayBuffer(max_size=500)

    drift_events = 0
    fine_tune_events = 0

    for i, (X_batch, y_batch) in enumerate(simulate_stream(X_stream, y_stream)):

        y_proba = clf.predict_proba(X_batch.values)[:, 1]
        y_pred = (y_proba > threshold).astype(int)

        acc = accuracy_score(y_batch, y_pred)
        auc = roc_auc_score(y_batch, y_proba)

        print(f"[{i}] Accuracy={acc:.3f}, AUC={auc:.3f}")

        if drift_detector.update(acc, auc):
            print(f"Drift erkannt bei Batch {i}")
            drift_events += 1

            alarms = y_pred == 1
            X_alarm = X_batch[alarms]
            y_alarm = y_batch[alarms]

            if not X_alarm.empty:
                X_ft, y_ft = replay_buffer.sample(n_old=50, X_new=X_alarm, y_new=y_alarm)
                clf = fine_tune_tabnet(clf, X_ft, y_ft, epochs=10)
                replay_buffer.add_batch(X_alarm, y_alarm)

                fine_tune_events += 1
                print(f"✓ Fine-Tuning durchgeführt bei Batch {i}")

    print("\n[✓] Drift-Handling abgeschlossen.")
    print(f"Erkannte Drift-Warnungen: {drift_events}")
    print(f"Durchgeführte Fine-Tunings: {fine_tune_events}")


if __name__ == "__main__":
    run_drift_simulation()
