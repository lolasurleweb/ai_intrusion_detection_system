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

def inject_label_drift(df, drift_fraction=0.4, random_state=42):
    df = df.copy()
    np.random.seed(random_state)

    harmless_indices = df[df["attack_detected"] == 0].index
    n_to_flip = int(len(harmless_indices) * drift_fraction)

    indices_to_flip = np.random.choice(harmless_indices, size=n_to_flip, replace=False)
    df.loc[indices_to_flip, "attack_detected"] = 1

    print(f"{n_to_flip} harmlose Instanzen wurden in Angriffe umetikettiert.")
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

from collections import deque
from river.drift import ADWIN
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import pandas as pd

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

from collections import deque
from river.drift import ADWIN
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import pandas as pd

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

def run_deployment_simulation_ensemble(threshold=0.5):
    print("[1] Lade Ensemble-Modelle...")
    models, model_paths = load_ensemble_models()

    print("[2] Bereite Drift-Simulationsdaten vor...")
    df_early = load_pickle("data/processed/drift_sim_early.pkl")
    df_mid = load_pickle("data/processed/drift_sim_mid.pkl")
    df_late = load_pickle("data/processed/drift_sim_late.pkl")

    df_mid_drifted = inject_label_drift(df_mid, drift_fraction=0.4)
    df_stream = pd.concat([df_early, df_mid_drifted], ignore_index=True)
    feature_names = df_stream.drop(columns=["attack_detected"]).columns.tolist()

    print("[3] Starte Inferenzloop mit Drift-Überwachung und Finetuning...")
    replay_buffer = deque(maxlen=10_000)
    adwin = ADWIN(delta=0.002)
    drift_detected_indices = []

    for idx, row in df_stream.iterrows():
        X_instance = row[feature_names].to_frame().T
        y_true = row["attack_detected"]

        # Vorhersage
        y_pred, y_proba = ensemble_predict(models, X_instance, threshold=threshold)
        y_pred_label = y_pred[0]

        # Fehler an ADWIN weitergeben
        error = int(y_pred_label != y_true)
        adwin.update(error)

        # Wenn Drift erkannt: Finetuning durchführen
        if adwin.drift_detected:
            print(f"\n⚠️ [DRIFT] Konzeptdrift erkannt bei Index {idx}")
            drift_detected_indices.append(idx)

            df_replay = pd.DataFrame(replay_buffer)
            if df_replay.empty:
                print("⚠️ Kein Finetuning durchgeführt – Replay Buffer ist leer.")
            else:
                X_replay = df_replay.drop(columns=["attack_detected"]).values.astype(np.float32)
                y_replay = df_replay["attack_detected"].values

                print("[4] Finetuning startet...")
                for i, model in enumerate(models):
                    print(f"    → Modell {i+1}/5 wird angepasst...")
                    finetune_tabnet_model(model, X_replay, y_replay)
                print("✅ Finetuning abgeschlossen – Modelle werden weiterverwendet.")

                # ADWIN zurücksetzen für nächste Driftüberwachung
                adwin = ADWIN(delta=0.002)
                print("🔁 ADWIN wurde zurückgesetzt.")

        # Alarmfall
        if y_pred_label == 1:
            print(f"[ALARM] Angriff erkannt bei Index {idx}")
            print(f"         → Vorhersage-Wahrscheinlichkeit: {y_proba[0]:.4f}")
            explanation = get_ensemble_explanation(models, X_instance)
            top_features = sorted(explanation.items(), key=lambda x: -x[1])[:3]
            print("         → Wichtigste Features laut Ensemble:")
            for feat, score in top_features:
                print(f"           {feat}: {score:.4f}")

            # Speichern im Replay Buffer
            instance_for_buffer = row[feature_names + ["attack_detected"]].to_dict()
            replay_buffer.append(instance_for_buffer)

    print(f"\n[5] Inferenz abgeschlossen. Größe des Replay Buffers: {len(replay_buffer)}")
    print(f"    Anzahl erkannter Drifts: {len(drift_detected_indices)}")
    if drift_detected_indices:
        print(f"    Detektierte Driftpositionen: {drift_detected_indices}")
