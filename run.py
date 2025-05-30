import argparse
import joblib
import pandas as pd

from src.data.preprocessing import (
    load_and_clean_data,
    log_transform,
    encode_categorical,
    split_for_training_and_drift,
    scale_numerical,
    TARGET_COL
)
from src.utils.io import save_pickle
from src.training.train_tabnet import run_training
from src.utils.seeding import set_seed
from simulate_drift import run_drift_simulation
import json

def preprocess():

    print("[1] Lade und bereinige Daten...")
    df = load_and_clean_data("data/raw/cybersecurity_intrusion_data.csv")
    df = log_transform(df, "session_duration")
    df_encoded = encode_categorical(df)
    df_encoded[TARGET_COL] = df[TARGET_COL].values

    print("[2] Splitte in Drift-Simulation und Trainingsdaten...")
    df_trainvaltest, drift_splits = split_for_training_and_drift(df_encoded, TARGET_COL)

    print("[3] Setze Feature-Spalten (Dummy-Kompatibilität)...")
    all_dfs = [df_trainvaltest] + list(drift_splits.values())
    all_columns = pd.concat(all_dfs).columns.tolist()

    # Speichere Spaltenstruktur für späteren Inferenzlauf
    with open("data/processed/columns.json", "w") as f:
        json.dump(all_columns, f)
        print(f"[✓] Spaltenstruktur gespeichert: data/processed/columns.json")

    # Reindexiere alle Splits auf vollständiges Spaltenset
    df_trainvaltest = df_trainvaltest.reindex(columns=all_columns, fill_value=0)
    for k in drift_splits:
        drift_splits[k] = drift_splits[k].reindex(columns=all_columns, fill_value=0)

    print("[4] Skaliere Trainingsdaten und speichere Scaler...")
    df_trainvaltest_scaled, scaler = scale_numerical(df_trainvaltest.drop(columns=[TARGET_COL]), fit=True)

    # Spaltenreihenfolge sicherstellen
    df_trainvaltest_scaled = df_trainvaltest_scaled.reindex(columns=[col for col in all_columns if col != TARGET_COL])

    # Zielspalte hinzufügen und Index-Synchronisation prüfen
    df_trainvaltest_scaled[TARGET_COL] = df_trainvaltest[TARGET_COL].values
    assert all(df_trainvaltest_scaled.index == df_trainvaltest.index), "[Fehler] Index stimmt nach Skalierung nicht überein!"

    joblib.dump(scaler, "data/processed/scaler.pkl")
    save_pickle(df_trainvaltest_scaled, "data/processed/train_val_test_pool.pkl")

    print("[5] Skaliere und speichere Drift-Splits...")
    for name, df_part in drift_splits.items():
        df_scaled, _ = scale_numerical(df_part.drop(columns=[TARGET_COL]), scaler=scaler, fit=False)
        df_scaled = df_scaled.reindex(columns=[col for col in all_columns if col != TARGET_COL])
        df_scaled[TARGET_COL] = df_part[TARGET_COL].values
        assert all(df_scaled.index == df_part.index), f"[Fehler] Index stimmt für Drift-Split '{name}' nicht überein!"
        save_pickle(df_scaled, f"data/processed/drift_sim_{name}.pkl")

    print("[✓] Preprocessing abgeschlossen.")


def main():
    parser = argparse.ArgumentParser(description="Cybersecurity ML-Pipeline")
    parser.add_argument("step", choices=["preprocess", "train", "simulate_drift"],
                        help="W\u00e4hle den Teil der Pipeline, den du ausf\u00fchren willst.")
    args = parser.parse_args()

    if args.step == "preprocess":
        preprocess()
    elif args.step == "train":
        set_seed(42)
        run_training()
    elif args.step == "simulate_drift":
        run_drift_simulation()

if __name__ == "__main__":
    main()
