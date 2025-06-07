from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier
import json
from src.utils.io import load_pickle
from src.drift.drift_detection import PerformanceDriftDetector
from src.drift.replay_buffer import ReplayBuffer
from src.drift.finetuning import fine_tune_tabnet
from src.training.train_tabnet import compute_optimal_threshold_by_cost_function
import numpy as np
import pandas as pd


def simulate_stream(X, y, batch_size=128):
    for i in range(0, len(X), batch_size):
        yield X.iloc[i:i+batch_size], y.iloc[i:i+batch_size]


def run_deployment_simulation():
    print("[1] Lade finales Modell und Threshold...")

    metadata_path = Path("models/final_model_metadata.json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        model_path = meta["model_path"] + ".zip"
        threshold_path = meta["threshold_path"]
    else:
        raise FileNotFoundError("Metadata-Datei zum Laden des finalen Modells fehlt.")

    clf = TabNetClassifier()
    clf.load_model(model_path)

    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    print("[2] Lade Holdout-Test-Kosten...")
    test_cost = None
    with open("reports/final_test_metrics.csv") as f:
        for line in f:
            if "cost" in line.lower():
                test_cost = float(line.strip().split(",")[-1])
                break

    if test_cost is None:
        raise ValueError("Testkosten konnten nicht aus der final_test_metrics.csv gelesen werden.")

    print("[3] Lade Zeitsplits aus Pickles...")
    df_early = load_pickle("data/processed/drift_sim_early.pkl")
    df_mid = load_pickle("data/processed/drift_sim_mid.pkl")
    df_late = load_pickle("data/processed/drift_sim_late.pkl")

    X_early = df_early.drop(columns=["attack_detected"])
    y_early = df_early["attack_detected"]

    X_mid = df_mid.drop(columns=["attack_detected"])
    y_mid = df_mid["attack_detected"]

    X_late = df_late.drop(columns=["attack_detected"])
    y_late = df_late["attack_detected"]

    X_stream = pd.concat([X_mid, X_late], axis=0).reset_index(drop=True)
    y_stream = pd.concat([y_mid, y_late], axis=0).reset_index(drop=True)

    print("[4] Starte Drift-Überwachung...")
    drift_detector = PerformanceDriftDetector(ref_cost=test_cost, rel_increase=0.15)
    replay_buffer = ReplayBuffer(max_size=500)

    drift_events = 0
    fine_tune_events = 0
    reoptimize_every = 1
    retraining_counter = 0

    for i, (X_batch, y_batch) in enumerate(simulate_stream(X_stream, y_stream)):

        y_proba = clf.predict_proba(X_batch.values)[:, 1]
        y_pred = (y_proba > threshold).astype(int)

        cost = 2 * np.sum((y_pred == 0) & (y_batch == 1)) + 1 * np.sum((y_pred == 1) & (y_batch == 0))
        cost /= len(y_batch)

        print(f"[{i}] Cost={cost:.3f}")

        if drift_detector.update(cost):
            print(f"Drift erkannt bei Batch {i}")
            drift_events += 1

            predicted_positive = y_pred == 1
            tp_mask = (predicted_positive) & (y_batch == 1)
            fp_mask = (predicted_positive) & (y_batch == 0)

            X_tp = X_batch[tp_mask]
            y_tp = y_batch[tp_mask]

            X_fp = X_batch[fp_mask]
            y_fp = y_batch[fp_mask]

            if not X_tp.empty:
                n_tp = min(len(X_tp), 30)
                n_fp = min(len(X_fp), 10)

                X_tp_sampled = X_tp.sample(n=n_tp, random_state=42)
                y_tp_sampled = y_tp.loc[X_tp_sampled.index]

                if n_fp > 0:
                    X_fp_sampled = X_fp.sample(n=n_fp, random_state=42)
                    y_fp_sampled = y_fp.loc[X_fp_sampled.index]
                else:
                    X_fp_sampled = pd.DataFrame(columns=X_tp.columns)
                    y_fp_sampled = pd.Series(dtype=int)

                X_alarm = pd.concat([X_tp_sampled, X_fp_sampled])
                y_alarm = pd.concat([y_tp_sampled, y_fp_sampled])

                print(f"[Batch {i}] → Fine-Tuning mit {len(X_tp_sampled)} TP, {len(X_fp_sampled)} FP")

                X_ft, y_ft = replay_buffer.sample(n_old=50, X_new=X_alarm, y_new=y_alarm)
                clf = fine_tune_tabnet(clf, X_ft, y_ft, epochs=10)
                replay_buffer.add_batch(X_alarm, y_alarm)

                fine_tune_events += 1
                retraining_counter += 1
                print(f"✓ Fine-Tuning durchgeführt bei Batch {i}")

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

    print("\n[✓] Einsatzsimulation abgeschlossen.")
    print(f"Erkannte Drift-Warnungen: {drift_events}")
    print(f"Durchgeführte Fine-Tunings: {fine_tune_events}")


if __name__ == "__main__":
    run_deployment_simulation()