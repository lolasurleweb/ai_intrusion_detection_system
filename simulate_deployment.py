import json
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

def inject_label_drift(df, drift_fraction=0.4, bidirectional=True, random_state=42):
    df = df.copy()
    np.random.seed(random_state)

    # 0 → 1
    harmless_indices = df[df["attack_detected"] == 0].index
    n_to_flip_0_to_1 = int(len(harmless_indices) * drift_fraction)
    indices_to_flip_0_to_1 = np.random.choice(harmless_indices, size=n_to_flip_0_to_1, replace=False)
    df.loc[indices_to_flip_0_to_1, "attack_detected"] = 1

    print(f"{n_to_flip_0_to_1} harmlose Instanzen wurden in Angriffe umetikettiert (0 → 1).")

    if bidirectional:
        # 1 → 0
        attack_indices = df[df["attack_detected"] == 1].index
        # Achtung: Teile der 1er-Instanzen könnten gerade erst umetikettiert worden sein!
        # Daher neu bestimmen, aber nur solche, die *nicht* gerade geändert wurden
        remaining_attack_indices = list(set(attack_indices) - set(indices_to_flip_0_to_1))
        n_to_flip_1_to_0 = int(len(remaining_attack_indices) * drift_fraction)
        indices_to_flip_1_to_0 = np.random.choice(remaining_attack_indices, size=n_to_flip_1_to_0, replace=False)
        df.loc[indices_to_flip_1_to_0, "attack_detected"] = 0

        print(f"{n_to_flip_1_to_0} Angriffe wurden in harmlose Instanzen umetikettiert (1 → 0).")

    return df

def finetune_tabnet_model(model, X, y, max_epochs=20, patience=5):
    model.fit(
        X_train=X,
        y_train=y,
        eval_set=[(X, y)],
        eval_name=["replay"],
        eval_metric=["accuracy"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=512,
        virtual_batch_size=128,
        drop_last=False,
        from_unsupervised=False
    )
    return model

def finetune_tabnet_model(model, X, y, max_epochs=20, patience=5):
    model.fit(
        X_train=X,
        y_train=y,
        eval_set=[(X, y)],
        eval_name=["replay"],
        eval_metric=["accuracy"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=512,
        virtual_batch_size=128,
        drop_last=False,
        from_unsupervised=None
    )
    return model

def finetune_tabnet_model(model, X, y, max_epochs=20, patience=5):
    model.fit(
        X_train=X,
        y_train=y,
        eval_set=[(X, y)],
        eval_name=["replay"],
        eval_metric=["logloss"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=512,
        virtual_batch_size=128,
        drop_last=False,
        from_unsupervised=None
    )
    return model

def run_deployment_simulation_ensemble(threshold=0.5):
    print("[1] Lade Ensemble-Modelle...")
    models, model_paths = load_ensemble_models()

    print("[2] Bereite Drift-Simulationsdaten vor...")
    df_early = load_pickle("data/processed/drift_sim_early.pkl")
    df_late = load_pickle("data/processed/drift_sim_late.pkl")

    df_late_drifted = inject_label_drift(df_late, drift_fraction=0.4)
    df_stream = pd.concat([df_early, df_late_drifted], ignore_index=True)
    feature_names = df_stream.drop(columns=["attack_detected"]).columns.tolist()

    print("[3] Starte Inferenzloop mit Drift-Überwachung und Finetuning...")
    replay_buffer = deque(maxlen=10_000)
    adwin = ADWIN(delta=0.002)
    drift_detected_indices = []

    low_threshold = 0.35  # Schwelle für "sehr sicher kein Angriff"
    finetuned_at = []


    for idx, row in df_stream.iterrows():
        X_instance = row[feature_names].to_frame().T
        y_true = row["attack_detected"]

        # Vorhersage
        y_pred, y_proba = ensemble_predict(models, X_instance, threshold=threshold)
        y_pred_label = y_pred[0]

        # Fehler an ADWIN weitergeben
        if (y_pred_label == 1 and y_true in [0, 1]) or (y_pred_label == 0 and y_true == 1):
            error = int(y_pred_label != y_true)
            adwin.update(error)

        # Wenn Drift erkannt: Finetuning durchführen
        if adwin.drift_detected:
            print(f"[DRIFT] Konzeptdrift erkannt bei Index {idx}")
            drift_detected_indices.append(idx)

            df_replay = pd.DataFrame(replay_buffer)
            if df_replay.empty:
                print("Kein Finetuning durchgeführt – Replay Buffer ist leer.")
            else:
                class_counts = df_replay["attack_detected"].value_counts().to_dict()
                if len(class_counts) < 2 or min(class_counts.values()) < 50:
                    print(f"Nicht genug Beispiele für Finetuning (Verteilung: {class_counts})")
                else:
                    # Schwächere Klasse bestimmen
                    minority_class = min(class_counts, key=class_counts.get)
                    majority_class = 1 - minority_class
                    n_samples = class_counts[minority_class]

                    # LIFO-Auswahl: Letzte n Instanzen je Klasse
                    minority_df = df_replay[df_replay["attack_detected"] == minority_class].iloc[-n_samples:]
                    majority_df = df_replay[df_replay["attack_detected"] == majority_class].iloc[-n_samples:]

                    df_balanced = pd.concat([minority_df, majority_df], ignore_index=True)
                    df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle

                    X_replay = df_balanced.drop(columns=["attack_detected"]).values.astype(np.float32)
                    y_replay = df_balanced["attack_detected"].values

                    print(f"Finetuning mit {n_samples} + {n_samples} Instanzen (Klassen 0/1)")
                    
                    print("[4] Finetuning startet...")
                    for i, model in enumerate(models):
                        print(f"    → Modell {i+1}/5 wird angepasst...")
                        finetune_tabnet_model(model, X_replay, y_replay)
                    print("Finetuning abgeschlossen – Modelle werden weiterverwendet.")
                    finetuned_at.append(idx)
                    adwin = ADWIN(delta=0.002)
                    print("ADWIN wurde zurückgesetzt.")

        # === Feedback-Simulation ===
        if y_pred_label == 1:
            print(f"[ALARM] Alarm bei Index {idx}")
            print(f"         → Vorhersage-Wahrscheinlichkeit: {y_proba[0]:.4f}")

            explanation = get_ensemble_explanation(models, X_instance)
            top_features = sorted(explanation.items(), key=lambda x: -x[1])[:3]
            print("         → Wichtigste Features laut Ensemble:")
            for feat, score in top_features:
                print(f"           {feat}: {score:.4f}")

        elif y_pred_label == 0 and y_true == 1:
            print(f"[MISS] Angriff wurde übersehen bei Index {idx}")
            print(f"        → Vorhersage-Wahrscheinlichkeit: {y_proba[0]:.4f}")

        # === Replay-Logik ===
        if y_pred_label == 1:
            instance_for_buffer = row[feature_names + ["attack_detected"]].to_dict()
            replay_buffer.append(instance_for_buffer)
        elif y_pred_label == 0 and y_true == 1:
            instance_for_buffer = row[feature_names + ["attack_detected"]].to_dict()
            replay_buffer.append(instance_for_buffer)
        elif y_pred_label == 0 and y_proba[0] < low_threshold:
            instance_for_buffer = row[feature_names + ["attack_detected"]].to_dict()
            replay_buffer.append(instance_for_buffer)

    print(f"\n[5] Inferenz abgeschlossen.")
    df_replay_final = pd.DataFrame(replay_buffer)
    class_counts = df_replay_final["attack_detected"].value_counts().to_dict()
    print("\n[6] Klassenverteilung im Replay Buffer:")
    print(f"    Klasse 0 (kein Angriff): {class_counts.get(0, 0)}")
    print(f"    Klasse 1 (Angriff):     {class_counts.get(1, 0)}")

    print(f"    Anzahl erkannter Drifts: {len(drift_detected_indices)}")
    if drift_detected_indices:
        print(f"    Detektierte Driftpositionen: {drift_detected_indices}")

    if finetuned_at:
        print(f"\n[7] Finetuning wurde durchgeführt bei folgenden Indizes:")
        print(f"    {finetuned_at}")
    else:
        print("\n[7] Kein Finetuning wurde durchgeführt.")

