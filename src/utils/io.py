import pandas as pd
from pathlib import Path
import pickle

def split_xy(df):
    return df.drop(columns=["attack_detected", "session_id"], errors='ignore'), df["attack_detected"]

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_train_val_test_pool():
    df = load_pickle("data/processed/train_val_test_pool.pkl")
    return split_xy(df)