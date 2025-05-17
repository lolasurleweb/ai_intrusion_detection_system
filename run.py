import argparse
from src.data.preprocessing import (
    load_and_clean_data,
    log_transform,
    encode_and_scale,
    create_splits,
    save_splits
)
from src.models.train_tabnet import load_train_val_test


def preprocess():
    print("[1] Lade und bereinige Daten...")
    df = load_and_clean_data("data/raw/cybersecurity_intrusion_data.csv")
    df = log_transform(df, "session_duration")
    X, y, scaler = encode_and_scale(df)
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    print("[✓] Preprocessing abgeschlossen.")

def load_splits():
    print("[2] Lade vorbereitete Daten...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()
    print("Train Shape:", X_train.shape)
    print("Val Shape:", X_val.shape)
    print("Test Shape:", X_test.shape)


def main():
    parser = argparse.ArgumentParser(description="Cybersecurity ML-Pipeline")
    parser.add_argument("step", choices=["preprocess", "load_splits"],
                        help="Wähle den Teil der Pipeline, den du ausführen willst.")
    args = parser.parse_args()

    if args.step == "preprocess":
        preprocess()
    elif args.step == "load_splits":
        load_splits()

if __name__ == "__main__":
    main()
