import pandas as pd
from pathlib import Path

def split_xy(df):
    return df.drop(columns=["attack_detected", "session_id"], errors='ignore'), df["attack_detected"]

def load_classic(data_dir='data/processed'):
    data_dir = Path(data_dir)

    df_train = pd.read_csv(data_dir / "train.csv")
    df_val   = pd.read_csv(data_dir / "val.csv")
    df_test  = pd.read_csv(data_dir / "test.csv")

    return split_xy(df_train), split_xy(df_val), split_xy(df_test)

def load_time(data_dir='data/processed'):
    data_dir = Path(data_dir)

    df_early = pd.read_csv(data_dir / "early.csv")
    df_mid   = pd.read_csv(data_dir / "mid.csv")
    df_late  = pd.read_csv(data_dir / "late.csv")

    return split_xy(df_early), split_xy(df_mid), split_xy(df_late)
