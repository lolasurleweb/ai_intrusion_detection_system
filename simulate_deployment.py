from pathlib import Path
from pytorch_tabnet.tab_model import TabNetClassifier
import json
from src.utils.io import load_pickle
from src.drift.replay_buffer import ReplayBuffer
from src.training.train_tabnet import CostMetric
import numpy as np
import pandas as pd

def cost_function(y_true, y_proba):
    y_score = np.vstack([1 - np.array(y_proba), np.array(y_proba)]).T
    metric = CostMetric()
    return metric(y_true, y_score)

def simulate_stream(clf, threshold, X_stream, y_stream, buffer, stream_name,
                    baseline_cost, cost_window=50, degrade_thresh=0.05):
    print(f"Starte Stream-Simulation mit Monitoring: {stream_name}")
    alert_count = 0
    recent_probs = []
    recent_trues = []

    print("[0] Lade erwartete Spaltenstruktur...")
    with open("data/processed/columns.json") as f:
        expected_columns = json.load(f)

    expected_columns = [str(col) for col in expected_columns if col != "attack_detected"]

    for i in range(len(X_stream)):
        x = X_stream.iloc[[i]].copy()
        x = x[[col for col in expected_columns if col in x.columns]]

        y_true = y_stream.iloc[i]

        try:
            prob = clf.predict_proba(x.to_numpy())[0][1]
        except Exception as e:
            print(f"[Fehler bei Instanz {i}] ➜ {e}")
            continue

        pred = int(prob >= threshold)

        recent_probs.append(prob)
        recent_trues.append(y_true)

        # Warnung + Erklärung
        if pred == 1:
            alert_count += 1
            explanation = x.iloc[0].sort_values(ascending=False).head(3).index.tolist()
            print(f"[ALERT #{alert_count}] Instanz {i} (p={prob:.2f}) ➜ Top Features: {explanation}")
            buffer.add(x.iloc[0].to_dict(), y_true)

        # Drift-Monitoring
        if len(recent_trues) >= cost_window:
            cost = cost_function(recent_trues, recent_probs)
            degradation = (cost - baseline_cost) / baseline_cost
            print(f"[Monitoring] Instanz {i}: Cost={cost:.2f} (Δ {degradation*100:.2f}%)")

            if degradation > degrade_thresh:
                print("Drift erkannt.")
                return "drift_detected"

            # Sliding Window
            recent_probs.pop(0)
            recent_trues.pop(0)

    print(f"[Stream-Ende] {alert_count} Alerts im Stream '{stream_name}' – kein Drift erkannt.")
    return "no_drift"



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

    X_early, y_early = df_early.drop(columns=["attack_detected"]), df_early["attack_detected"]
    X_mid, y_mid = df_mid.drop(columns=["attack_detected"]), df_mid["attack_detected"]
    X_late, y_late = df_late.drop(columns=["attack_detected"]), df_late["attack_detected"]

    print("[4] Initialisiere Replay Buffer...")
    replay_buffer = ReplayBuffer(maxlen=500)

    for X_split, y_split, name in zip(
        [X_early, X_mid, X_late],
        [y_early, y_mid, y_late],
        ["Early Drift", "Mid Drift", "Late Drift"]
    ):
        result = simulate_stream(
            clf=clf,
            threshold=threshold,
            X_stream=X_split,
            y_stream=y_split,
            buffer=replay_buffer,
            stream_name=name,
            baseline_cost=test_cost,
            cost_window=50,
            degrade_thresh=0.05
        )

        if result == "drift_detected":
            print(f"[Action] ➜ Fine-Tuning sollte jetzt erfolgen (noch nicht implementiert).")
            break

    print(f"Simulation abgeschlossen. Replay Buffer enthält {len(replay_buffer.buffer)} Instanzen.")