import json
import pickle
import numpy as np
from pathlib import Path
from glob import glob
from src.utils.io import load_pickle
from pytorch_tabnet.tab_model import TabNetClassifier
from river.drift import ADWIN

def load_ensemble_models():
    matches = sorted(glob("models/tabnet_cv_*/metadata.json"))
    if not matches:
        raise FileNotFoundError("Keine metadata.json gefunden.")
    metadata_path = matches[-1]

    with open(metadata_path) as f:
        meta = json.load(f)

    model_paths = meta["model_paths"]
    models = []
    for path in model_paths:
        clf = TabNetClassifier()
        clf.load_model(f"{path}.zip")
        models.append(clf)
    return models, model_paths

def ensemble_predict(models, X_np, threshold=0.5):
    y_proba_matrix = np.array([model.predict_proba(X_np)[:, 1] for model in models])
    y_proba = y_proba_matrix.mean(axis=0)
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, y_proba

def finetune_ensemble_models(models, buffer, save_paths):
    X_buffer, y_buffer = zip(*buffer)
    X_buffer = np.stack(X_buffer)
    y_buffer = np.array(y_buffer)

    print(f"[→] Finetuning auf {len(buffer)} Instanzen...")

    for i, (model, path) in enumerate(zip(models, save_paths), start=1):
        print(f"  • Modell {i} wird feinjustiert...")
        model.fit(X_buffer, y_buffer, max_epochs=10, patience=3, loss_fn="binary_focal_loss")
        model.save_model(path)
        print(f"    ↳ Gespeichert unter: {path}.zip")

    print("[✓] Alle Ensemble-Modelle wurden feinjustiert.")

def run_deployment_loop_ensemble(df, models, feature_names, adwin, replay_buffer,
                                  drift_flag, min_samples_to_refit, model_paths, threshold=0.5):

    X_all = df.drop(columns=["attack_detected"])
    y_true = df["attack_detected"].values
    X_np = X_all.values.astype(np.float32)

    y_pred, y_proba = ensemble_predict(models, X_np, threshold=threshold)

    for i in range(len(df)):
        if y_pred[i] == 1:
            x_instance = X_all.iloc[i]
            true_label = y_true[i]
            replay_buffer.append((x_instance.values, true_label))
            adwin.update(int(y_pred[i] == true_label))  # 1 = korrekt, 0 = falsch

            if adwin.drift_detected:
                print("[!] Drift erkannt!")
                drift_flag[0] = True

        if drift_flag[0] and len(replay_buffer) >= min_samples_to_refit:
            finetune_ensemble_models(models, replay_buffer, model_paths)
            drift_flag[0] = False
            replay_buffer.clear()

def flip_labels():
    with open("data/processed/drift_sim_late.pkl", "rb") as f:
        df_late = pickle.load(f)
    np.random.seed(42)
    flip_indices = np.random.choice(df_late.index, size=int(0.5 * len(df_late)), replace=False)
    df_late.loc[flip_indices, "attack_detected"] = 1 - df_late.loc[flip_indices, "attack_detected"]
    with open("data/processed/drift_sim_late.pkl", "wb") as f:
        pickle.dump(df_late, f)
    print(f"[✓] {len(flip_indices)} Labels in 'drift_sim_late.pkl' wurden erfolgreich geflippt.")

def run_deployment_simulation_ensemble(threshold=0.5):
    print("[1] Lade Ensemble-Modelle...")
    models, model_paths = load_ensemble_models()

    print("[2] Lade Drift-Simulationsdaten...")
    df_early = load_pickle("data/processed/drift_sim_early.pkl")
    df_mid = load_pickle("data/processed/drift_sim_mid.pkl")
    flip_labels()
    df_late = load_pickle("data/processed/drift_sim_late.pkl")

    feature_names = df_early.drop(columns=["attack_detected"]).columns.tolist()

    adwin = ADWIN()
    replay_buffer = []
    drift_flag = [False]
    min_samples_to_refit = 50

    print("\n[3] Simuliere Deployment: early")
    run_deployment_loop_ensemble(df_early, models, feature_names,
                                  adwin, replay_buffer, drift_flag,
                                  min_samples_to_refit, model_paths, threshold)

    print("\n[4] Simuliere Deployment: mid")
    run_deployment_loop_ensemble(df_mid, models, feature_names,
                                  adwin, replay_buffer, drift_flag,
                                  min_samples_to_refit, model_paths, threshold)

    print("\n[5] Simuliere Deployment: late")
    run_deployment_loop_ensemble(df_late, models, feature_names,
                                  adwin, replay_buffer, drift_flag,
                                  min_samples_to_refit, model_paths, threshold)

    print("\n[✓] Ensemble-Deployment-Simulation abgeschlossen.")
