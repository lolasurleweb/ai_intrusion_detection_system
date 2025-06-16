import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json
from src.utils.io import save_pickle
import joblib

NUMERICAL_FEATURES = [
    'network_packet_size',
    'login_attempts',
    'ip_reputation_score',
    'failed_logins',
    'session_duration_log'
]

CATEGORICAL_FEATURES = ['protocol_type', 'encryption_used', 'browser_type']
BINARY_FEATURES = ['unusual_time_access']
TARGET_COL = 'attack_detected'

def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "session_id" in df.columns:
        df = df.drop(columns=["session_id"])
    df['encryption_used'] = df['encryption_used'].fillna('None')
    return df

def log_transform(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col + '_log'] = np.log1p(df[col])
    df.drop(columns=[col], inplace=True)
    return df

def encode_categorical(df: pd.DataFrame, save_path=None) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
    
    if save_path:
        with open(save_path, "w") as f:
            json.dump(df_encoded.columns.tolist(), f)
    
    return df_encoded.astype({col: float for col in df_encoded.select_dtypes('bool').columns})

def scale_numerical(df: pd.DataFrame, scaler: StandardScaler = None, fit=True):
    df_scaled = df.copy()
    if fit:
        scaler = StandardScaler()
        df_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(df_scaled[NUMERICAL_FEATURES])
    else:
        df_scaled[NUMERICAL_FEATURES] = scaler.transform(df_scaled[NUMERICAL_FEATURES])
    return df_scaled, scaler

def split_for_training_and_drift(df: pd.DataFrame, target_col: str, seed: int = 42):
    df_shuffled = shuffle(df, random_state=seed).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_drift = int(n_total * 0.33)
    df_drift = df_shuffled.iloc[:n_drift].copy()
    df_trainvaltest = df_shuffled.iloc[n_drift:].copy()

    y_drift = df_drift[target_col]

    # Neu: Nur zwei Splits
    df_early, df_late = train_test_split(
        df_drift, test_size=0.5, stratify=y_drift, random_state=seed
    )

    drift_splits = {
        "early": df_early.reset_index(drop=True),
        "late": df_late.reset_index(drop=True)
    }

    return df_trainvaltest.reset_index(drop=True), drift_splits

def split_train_val_test_holdout(df: pd.DataFrame, target_col: str, test_size: float = 0.2, seed: int = 42):
    df_class_0 = df[df[target_col] == 0]
    df_class_1 = df[df[target_col] == 1]
    
    n_min = min(len(df_class_0), len(df_class_1))
    
    df_balanced = pd.concat([
        df_class_0.sample(n=n_min, random_state=seed),
        df_class_1.sample(n=n_min, random_state=seed)
    ]).reset_index(drop=True)

    df_balanced = shuffle(df_balanced, random_state=seed).reset_index(drop=True)

    y = df_balanced[target_col]
    X = df_balanced.drop(columns=[target_col])

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    df_pool = X_pool.copy()
    df_pool[target_col] = y_pool

    df_test = X_test.copy()
    df_test[target_col] = y_test

    return df_pool.reset_index(drop=True), df_test.reset_index(drop=True)

def preprocess():

    print("[1] Lade und bereinige Daten...")
    df = load_and_clean_data("data/raw/cybersecurity_intrusion_data.csv")
    df = log_transform(df, "session_duration")

    df_encoded = encode_categorical(df)
    df_encoded[TARGET_COL] = df[TARGET_COL].values

    print("[2a] Splitte in Drift-Simulation und Trainingsdaten...")
    df_trainvaltest, drift_splits = split_for_training_and_drift(df_encoded, TARGET_COL)

    print("[2b] Splitte Train-Val-Test-Pool in Training/Validation und Testset...")
    df_trainval, df_test = split_train_val_test_holdout(df_trainvaltest, TARGET_COL)

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
    df_trainval_scaled, scaler = scale_numerical(df_trainval.drop(columns=[TARGET_COL]), fit=True)
    df_test_scaled, _ = scale_numerical(df_test.drop(columns=[TARGET_COL]), scaler=scaler, fit=False)

    df_trainval_scaled = df_trainval_scaled.reindex(columns=[col for col in all_columns if col != TARGET_COL])
    df_test_scaled = df_test_scaled.reindex(columns=[col for col in all_columns if col != TARGET_COL])

    df_trainval_scaled[TARGET_COL] = df_trainval[TARGET_COL].values
    df_test_scaled[TARGET_COL] = df_test[TARGET_COL].values

    joblib.dump(scaler, "data/processed/scaler.pkl")
    save_pickle(df_trainval_scaled, "data/processed/train_val_pool.pkl")
    save_pickle(df_test_scaled, "data/processed/test_holdout.pkl")
    print("[✓] Train/Val/Test Splits gespeichert.")

    print("[5] Skaliere und speichere Drift-Splits...")
    for name, df_part in drift_splits.items():
        df_scaled, _ = scale_numerical(df_part.drop(columns=[TARGET_COL]), scaler=scaler, fit=False)
        df_scaled = df_scaled.reindex(columns=[col for col in all_columns if col != TARGET_COL])
        df_scaled[TARGET_COL] = df_part[TARGET_COL].values
        assert all(df_scaled.index == df_part.index), f"[Fehler] Index stimmt für Drift-Split '{name}' nicht überein!"
        save_pickle(df_scaled, f"data/processed/drift_sim_{name}.pkl")

    print("[✓] Preprocessing abgeschlossen.")