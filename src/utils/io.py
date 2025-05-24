import pandas as pd
from pathlib import Path
import pickle

def split_xy(df):
    return df.drop(columns=["attack_detected", "session_id"], errors='ignore'), df["attack_detected"]

def load_time(data_dir='data/processed'):
    data_dir = Path(data_dir)
    df_early = pd.read_csv(data_dir / "early.csv")
    df_mid   = pd.read_csv(data_dir / "mid.csv")
    df_late  = pd.read_csv(data_dir / "late.csv")
    return split_xy(df_early), split_xy(df_mid), split_xy(df_late)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_train_val_test_pool():
    df = load_pickle("data/processed/train_val_test_pool.pkl")
    return split_xy(df)