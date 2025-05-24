import os
import pandas as pd
import numpy as np
import joblib
from src.data.preprocessing import NUMERICAL_FEATURES


def compare_weights(suffix):
    import torch
    before = torch.load(f"checkpoints/before_{suffix}.pt")
    after = torch.load(f"checkpoints/after_{suffix}.pt")

    for k in before:
        diff = torch.norm(before[k] - after[k]).item()
        print(f"{k}: Δ={diff:.6f}")


def check_scaling_consistency(csv_paths, scaler_path):
    if not os.path.exists(scaler_path):
        print(f"[!] Scaler nicht gefunden unter: {scaler_path}")
        return

    scaler = joblib.load(scaler_path)

    print("[QA] Vergleiche Feature-Statistiken mit train-Scaler:\n")
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"[!] Datei fehlt: {path}")
            continue

        df = pd.read_csv(path)
        means = df[NUMERICAL_FEATURES].mean()
        stds = df[NUMERICAL_FEATURES].std()

        print(f"--- {os.path.basename(path)} ---")
        for i, feature in enumerate(NUMERICAL_FEATURES):
            target_mean = 0  # nach Standardisierung
            target_std = 1

            m_diff = abs(means[feature] - target_mean)
            s_diff = abs(stds[feature] - target_std)

            status = "OK" if m_diff < 0.1 and s_diff < 0.1 else "WARN"
            print(f"{feature:<25} mean Δ={m_diff:.2f}, std Δ={s_diff:.2f} [{status}]")
        print()


if __name__ == "__main__":
    compare_weights("dein_suffix")  # optional

    check_scaling_consistency(
        csv_paths=[
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv",
            "data/processed/early.csv",
            "data/processed/mid.csv",
            "data/processed/late.csv",
        ],
        scaler_path="data/processed/scaler.pkl"
    )
