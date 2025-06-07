from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier
import json
from src.utils.io import load_pickle
from src.drift.replay_buffer import ReplayBuffer
from river.drift import ADWIN
import numpy as np
import pandas as pd

def simulate_stream(clf, threshold, data_splits):
    print("[3] Initialisiere ADWIN, Replay Buffer, Logging...")
    adwin = ADWIN(delta=0.002)
    replay_buffer = ReplayBuffer(maxlen=500)

    log = []
    drift_points = []

    print("[4] Starte Simulation...")
    stream = pd.concat(data_splits, ignore_index=True)

    for i, row in stream.iterrows():
        X_inst = row.drop("attack_detected").to_frame().T
        true_label = row["attack_detected"]

        # Vorhersage & Wahrscheinlichkeit
        proba = clf.predict_proba(X_inst.values)[0, 1]
        pred = int(proba > threshold)

        # Erklärung bei Warnung
        if pred == 1:
            masks, _ = clf.explain(X_inst.values.astype(np.float32))
            top_features = np.argsort(masks[0])[::-1][:5]
            explanation = {
                X_inst.columns[i]: float(masks[0][i]) for i in top_features
            }

            # Feedback (SOC)
            replay_buffer.append(X_inst, true_label)

            log.append({
                "index": i,
                "pred": pred,
                "proba": float(proba),
                "true_label": int(true_label),
                "explanation": explanation
            })

        # Monitoring (unabhängig von pred)
        is_correct = int(pred == true_label)
        adwin.update(is_correct)

        if adwin.drift_detected:
            print(f"[!] Drift erkannt bei Instanz {i}. Starte Finetuning...")
            drift_points.append(i)

            # Finetuning
            if len(replay_buffer) > 10:
                X_replay, y_replay = replay_buffer.get()
                clf.fit(
                    X_train=X_replay.values, y_train=y_replay.values,
                    eval_set=[(X_replay.values, y_replay.values)],
                    eval_name=["replay"],
                    max_epochs=20, patience=5, batch_size=512,
                    eval_metric=["logloss"]
                )
                adwin.reset()
                print("[✓] Finetuning abgeschlossen.")
            else:
                print("[!] Zu wenig Daten im Replay Buffer – Finetuning übersprungen.")

    # Speichern der Logs
    Path("reports/deployment").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_json("reports/deployment/warnings_log.json", orient="records", indent=2)
    pd.Series(drift_points).to_csv("reports/deployment/drift_points.csv", index=False)
    print("[✓] Simulation abgeschlossen. Logs gespeichert.")

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

    print("[2] Lade Zeitsplits aus Pickles...")
    df_early = load_pickle("data/processed/drift_sim_early.pkl")
    df_mid = load_pickle("data/processed/drift_sim_mid.pkl")
    df_late = load_pickle("data/processed/drift_sim_late.pkl")

    simulate_stream(clf, threshold, [df_early, df_mid, df_late])
