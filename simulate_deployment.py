import json
import pickle
import numpy as np
from glob import glob
from collections import deque

import pandas as pd
from src.utils.io import load_pickle
from pytorch_tabnet.tab_model import TabNetClassifier
from river.drift import ADWIN

def load_ensemble_models():
    matches = sorted(glob("models/tabnet_bayesopt_*/metadata.json"))
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

def ensemble_predict(models, X, threshold=0.5, uncertainty_threshold=0.15):
    X_np = X.values.astype(np.float32)
    y_proba_matrix = np.array([model.predict_proba(X_np)[:, 1] for model in models])
    y_proba_mean = y_proba_matrix.mean(axis=0)
    y_std_per_sample = y_proba_matrix.std(axis=0)

    y_pred = np.where(
        y_std_per_sample > uncertainty_threshold,
        1,
        (y_proba_mean >= threshold).astype(int)
    )
    return y_pred, y_proba_mean

def get_ensemble_explanation(models, X_instance_df):
    X_np = X_instance_df.values.astype(np.float32)

    all_masks = [model.explain(X_np)[0][0] for model in models]
    avg_mask = np.mean(all_masks, axis=0)

    if avg_mask.ndim == 1:
        feature_importance = avg_mask 
    elif avg_mask.ndim == 2:
        feature_importance = avg_mask[0]
    else:
        raise ValueError(f"Unerwartete Maskenform: {avg_mask.shape}")

    return dict(zip(X_instance_df.columns, feature_importance))

def finetune_ensemble_models(models, buffer, save_paths, feature_names):
    X_buffer, y_buffer = zip(*buffer)
    X_df = pd.DataFrame(X_buffer, columns=feature_names).astype(np.float32)
    y_buffer = np.array(y_buffer)

    print(f"[→] Finetuning auf {len(buffer)} Instanzen...")

    for i, (model, path) in enumerate(zip(models, save_paths), start=1):
        print(f"  • Modell {i} wird feinjustiert...")
        model.fit(
            X_train=X_df.values,
            y_train=y_buffer,
            max_epochs=10,
            patience=3,
            loss_fn="binary_focal_loss"
        )
        model.save_model(path)
        print(f"    ↳ Gespeichert unter: {path}.zip")

    print("[✓] Alle Ensemble-Modelle wurden feinjustiert.")

def run_deployment_loop_ensemble(df, models, feature_names, adwin, replay_buffer,
                                 drift_flag, min_samples_to_refit, model_paths, threshold, stats):

    X_all = df.drop(columns=["attack_detected"])
    y_true = df["attack_detected"].values

    y_pred, y_proba = ensemble_predict(models, X_all, threshold=threshold)

    for i in range(len(df)):
        x_instance = X_all.iloc[i]
        true_label = y_true[i]

        # Update ADWIN mit dem absoluten Fehler zwischen der Prädiktionswahrscheinlichkeit und dem tatsächlichen Label
        error = abs(y_proba[i] - true_label)
        adwin.update(error)

        # Bonus: Drift-Erkennung nur einmal pro Detektion
        if adwin.drift_detected and not drift_flag[0]:
            print(f"[!] Drift erkannt bei Instanz {i}")
            print(f"↳ ADWIN-Fehlerschätzer: {adwin.estimation:.3f}")
            drift_flag[0] = True
            stats["drift_detected_count"] += 1

        if y_pred[i] == 1:
            x_instance_df = X_all.iloc[[i]]  # Als DataFrame für explain
            explanation = get_ensemble_explanation(models, x_instance_df)

            print(f"\n[ALARM] Angriff erkannt bei Instanz {i}")
            print("↳ Erklärung (Top-Features):")
            top_k = sorted(explanation.items(), key=lambda x: -x[1])[:5]
            for feat, val in top_k:
                print(f"  • {feat}: {val:.4f}")

            replay_buffer.append((x_instance_df.values.flatten(), true_label))
            stats["total_replay_samples"] += 1

        if drift_flag[0] and len(replay_buffer) >= min_samples_to_refit:
            finetune_ensemble_models(models, replay_buffer, model_paths, feature_names)
            drift_flag[0] = False
            replay_buffer.clear()
            stats["finetune_count"] += 1

def flip_labels(path: str, flip_ratio: float):
    with open(path, "rb") as f:
        df = pickle.load(f)
    np.random.seed(42)
    n_flip = int(flip_ratio * len(df))
    flip_indices = np.random.choice(df.index, size=n_flip, replace=False)
    df.loc[flip_indices, "attack_detected"] = 1 - df.loc[flip_indices, "attack_detected"]
    with open(path, "wb") as f:
        pickle.dump(df, f)
    print(f"[✓] {n_flip} Labels in '{path}' wurden erfolgreich geflippt.")

def run_deployment_simulation_ensemble(threshold=0.5):
    print("[1] Lade Ensemble-Modelle...")
    models, model_paths = load_ensemble_models()

    print("[2] Lade Drift-Simulationsdaten...")
    df_early = load_pickle("data/processed/drift_sim_early.pkl")

    flip_labels("data/processed/drift_sim_mid.pkl", flip_ratio=0.4)
    df_mid = load_pickle("data/processed/drift_sim_mid.pkl")

    flip_labels("data/processed/drift_sim_late.pkl", flip_ratio=0.8)
    df_late = load_pickle("data/processed/drift_sim_late.pkl")

    df_stream = pd.concat([df_early, df_mid, df_late], ignore_index=True)

    feature_names = df_stream.drop(columns=["attack_detected"]).columns.tolist()

    adwin = ADWIN(delta=0.005)
    replay_buffer = []
    drift_flag = [False]
    min_samples_to_refit = 50

    stats = {
        "drift_detected_count": 0,
        "finetune_count": 0,
        "total_replay_samples": 0
    }

    print("\n[3] Simuliere Deployment über gesamten Datenstrom")
    run_deployment_loop_ensemble(df_stream, models, feature_names,
                                 adwin, replay_buffer, drift_flag,
                                 min_samples_to_refit, model_paths, threshold,
                                 stats)

    print("\n[📊] Zusammenfassung der Deployment-Simulation:")
    print(f"    ↳ Drift erkannt:           {stats['drift_detected_count']}x")
    print(f"    ↳ Finetuning durchgeführt: {stats['finetune_count']}x")
    print(f"    ↳ Gesammelte SOC-Feedbacks im Replay Buffer: {stats['total_replay_samples']}")
