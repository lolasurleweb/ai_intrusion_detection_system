import pandas as pd
from pathlib import Path

def load_train_val_test(data_dir='data/processed'):
    data_dir = Path(data_dir)

    train_df = pd.read_csv(data_dir / "train_data.csv")
    val_df = pd.read_csv(data_dir / "val_data.csv")
    test_df = pd.read_csv(data_dir / "test_data.csv")

    X_train = train_df.drop(columns=["attack_detected"])
    y_train = train_df["attack_detected"]

    X_val = val_df.drop(columns=["attack_detected"])
    y_val = val_df["attack_detected"]

    X_test = test_df.drop(columns=["attack_detected"])
    y_test = test_df["attack_detected"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)