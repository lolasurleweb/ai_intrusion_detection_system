import argparse
from src.data.preprocessing import (
    load_and_clean_data,
    log_transform,
    encode_categorical,     
    create_classic_split,
    create_time_split,
    scale_and_save_splits,
    TARGET_COL                 
)
from src.models.train_tabnet import (
    load_classic,
    load_time,
)

def preprocess():
    print("[1] Lade und bereinige Daten...")
    df = load_and_clean_data("data/raw/cybersecurity_intrusion_data.csv")
    df = log_transform(df, "session_duration")

    df_encoded = encode_categorical(df)
    df_encoded[TARGET_COL] = df[TARGET_COL].values
    df_encoded['session_id'] = df['session_id'].values

    print("[2] Sortiere nach Zeit (session_id)...")
    df_encoded = df_encoded.sort_values("session_id").reset_index(drop=True)

    n = len(df_encoded)
    df_early = df_encoded.iloc[:int(n * 0.33)].copy()
    df_mid_late = df_encoded.iloc[int(n * 0.33):].copy()

    print("[3] Erzeuge klassischen Split (nur mid + late)...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_classic_split(df_mid_late)
    scale_and_save_splits({
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    })

    print("[4] Erzeuge zeitlichen Split (nur early)...")
    (X_early, y_early), (X_mid, y_mid), (X_late, y_late) = create_time_split(df_early)
    scale_and_save_splits({
        "early": (X_early, y_early),
        "mid":   (X_mid,   y_mid),
        "late":  (X_late,  y_late),
    })

    print("[✓] Preprocessing abgeschlossen.")

def load_classic_splits():
    print("[2] Lade klassisch segmentierte Daten...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_classic()
    print("Train Shape:", X_train.shape)
    print("Val Shape:", X_val.shape)
    print("Test Shape:", X_test.shape)


def load_time_splits():
    print("[3] Lade zeitlich segmentierte Daten...")
    (X_early, y_early), (X_mid, y_mid), (X_late, y_late) = load_time()
    print("Early Shape:", X_early.shape)
    print("Mid Shape:", X_mid.shape)
    print("Late Shape:", X_late.shape)


def main():
    parser = argparse.ArgumentParser(description="Cybersecurity ML-Pipeline")
    parser.add_argument("step", choices=["preprocess", "load_classic", "load_time"],
                        help="Wähle den Teil der Pipeline, den du ausführen willst.")
    args = parser.parse_args()

    if args.step == "preprocess":
        preprocess()
    elif args.step == "load_classic":
        load_classic_splits()
    elif args.step == "load_time":
        load_time_splits()

if __name__ == "__main__":
    main()