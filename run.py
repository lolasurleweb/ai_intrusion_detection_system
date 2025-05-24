import argparse

import joblib
from src.data.preprocessing import (
    load_and_clean_data,
    log_transform,
    encode_categorical,     
    create_time_split,
    scale_and_save_splits,
    TARGET_COL,
    scale_numerical                 
)
from src.utils.io import save_pickle

from src.training.train_tabnet import (
    run_training
)
from src.utils.seeding import set_seed
from simulate_drift import run_drift_simulation


def preprocess():
    print("[1] Lade und bereinige Daten...")
    df = load_and_clean_data("data/raw/cybersecurity_intrusion_data.csv")
    df = log_transform(df, "session_duration")
    df_encoded = encode_categorical(df)
    df_encoded[TARGET_COL] = df[TARGET_COL].values
    df_encoded["session_id"] = df["session_id"].values
    expected_columns = df_encoded.columns.tolist()

    print("[2] Sortiere nach Zeit...")
    df_encoded = df_encoded.sort_values("session_id").reset_index(drop=True)
    df_encoded.drop(columns=["session_id"], inplace=True)

    print("[3] Split in Early vs Mid+Late...")
    n = len(df_encoded)
    df_early = df_encoded.iloc[:int(n * 0.33)].copy()
    df_mid_late = df_encoded.iloc[int(n * 0.33):].copy()

    df_early = df_early.reindex(columns=expected_columns, fill_value=0)
    df_mid_late = df_mid_late.reindex(columns=expected_columns, fill_value=0)

    print("[4] Speichere Pool (Mid + Late) für Cross-Validation...")
    save_pickle(df_mid_late, "data/processed/train_val_test_pool.pkl")

    print("[5] Skaliere Mid + Late für Scaler-Anpassung...")
    df_mid_late_scaled, scaler = scale_numerical(df_mid_late.drop(columns=[TARGET_COL]), fit=True)
    joblib.dump(scaler, "data/processed/scaler.pkl")

    print("[6] Erzeuge und speichere Time-Splits mit gleichem Scaler...")
    (X_early, y_early), (X_mid, y_mid), (X_late, y_late) = create_time_split(df_early)
    scale_and_save_splits({
        "early": (X_early, y_early),
        "mid": (X_mid, y_mid),
        "late": (X_late, y_late)
    }, scaler=scaler)

    print("[✓] Preprocessing abgeschlossen.")


def main():
    parser = argparse.ArgumentParser(description="Cybersecurity ML-Pipeline")
    parser.add_argument("step", choices=["preprocess", "load_classic", "load_time", "train", "simulate_drift"],
                        help="Wähle den Teil der Pipeline, den du ausführen willst.")
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