import json
from pathlib import Path
import pickle
import numpy as np
from src.training.evaluate_tabnet import save_instance_level_explanations
from src.utils.io import load_pickle
from pytorch_tabnet.tab_model import TabNetClassifier
from river.drift import ADWIN

def flip_labels():
    with open("data/processed/drift_sim_late.pkl", "rb") as f:
        df_late = pickle.load(f)

    # Flippe 50 % der Labels
    flip_frac = 0.5
    n_total = len(df_late)
    n_flip = int(flip_frac * n_total)
    np.random.seed(42)
    flip_indices = np.random.choice(df_late.index, size=n_flip, replace=False)

    df_late.loc[flip_indices, "attack_detected"] = 1 - df_late.loc[flip_indices, "attack_detected"]

    with open("data/processed/drift_sim_late.pkl", "wb") as f:
        pickle.dump(df_late, f)

    print(f"[✓] {n_flip} Labels in 'drift_sim_late.pkl' wurden erfolgreich geflippt.")


def finetune_on_replay_buffer(clf, buffer, model_save_path):
    X_buffer, y_buffer = zip(*buffer)
    X_buffer = np.stack(X_buffer)
    y_buffer = np.array(y_buffer)

    print(f"[→] Finetuning auf {len(buffer)} Instanzen...")
    clf.fit(X_buffer, y_buffer, max_epochs=10, patience=3, loss_fn="binary_focal_loss")
    clf.save_model(model_save_path)
    print(f"[✓] Finetuned Modell gespeichert unter: {model_save_path}")

def run_deployment_loop(df, clf, feature_names, explanation_path,
                        adwin, replay_buffer, drift_flag, min_samples_to_refit, model_save_path, threshold=0.5):

    X_all = df.drop(columns=["attack_detected"])
    y_true = df["attack_detected"].values
    X_np = X_all.values.astype(np.float32)

    y_proba = clf.predict_proba(X_np)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    save_instance_level_explanations(
        clf, X_all, y_proba, y_pred, feature_names, explanation_path,
        only_positive_predictions=True
    )

    for i in range(len(df)):
        if y_pred[i] == 1:
            # Alarm ausgelöst → SOC-Feedback simulieren
            x_instance = X_all.iloc[i]
            true_label = y_true[i]
            replay_buffer.append((x_instance.values, true_label))
            adwin.update(int(y_pred[i] == true_label))  # 1 = korrekt, 0 = falsch

            if adwin.drift_detected:
                print("[!] Drift erkannt!")
                drift_flag[0] = True

        if drift_flag[0] and len(replay_buffer) >= min_samples_to_refit:
            finetune_on_replay_buffer(clf, replay_buffer, model_save_path)
            drift_flag[0] = False
            replay_buffer.clear()

def run_deployment_simulation(threshold=0.5):
    # Modell laden
    metadata_path = Path("models/final_model_metadata.json")
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        model_path = meta["model_path"] + ".zip"
    else:
        raise FileNotFoundError("Metadata-Datei zum Laden des finalen Modells fehlt.")

    clf = TabNetClassifier()
    clf.load_model(model_path)

    # Daten laden
    print("[1] Lade Daten...")
    df_early = load_pickle("data/processed/drift_sim_early.pkl")
    df_mid = load_pickle("data/processed/drift_sim_mid.pkl")
    flip_labels()
    df_late = load_pickle("data/processed/drift_sim_late.pkl")

    feature_names = df_early.drop(columns=["attack_detected"]).columns.tolist()

    # Initialisierung
    adwin = ADWIN()
    #print("[DEBUG] ADWIN Methoden:", dir(adwin))
    replay_buffer = []
    drift_flag = [False]
    min_samples_to_refit = 50
    model_save_path = "models/finetuned_model"

    # Deployment-Simulation
    print("\n[2] Simuliere Deployment: early")
    run_deployment_loop(df_early, clf, feature_names, "explanations/early.json",
                        adwin, replay_buffer, drift_flag, min_samples_to_refit, model_save_path, threshold=threshold)

    print("\n[3] Simuliere Deployment: mid")
    run_deployment_loop(df_mid, clf, feature_names, "explanations/mid.json",
                        adwin, replay_buffer, drift_flag, min_samples_to_refit, model_save_path, threshold=threshold)

    print("\n[4] Simuliere Deployment: late")
    run_deployment_loop(df_late, clf, feature_names, "explanations/late.json",
                        adwin, replay_buffer, drift_flag, min_samples_to_refit, model_save_path, threshold=threshold)

    print("\n[✓] Deployment-Simulation abgeschlossen.")
