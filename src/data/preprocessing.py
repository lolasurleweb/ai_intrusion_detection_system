import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import json

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

def create_time_split(df: pd.DataFrame, seed=42):
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_temp, X_late, y_temp, y_late = train_test_split(X, y, test_size=1/3, stratify=y, random_state=seed)
    X_early, X_mid, y_early, y_mid = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed)

    return (X_early, y_early), (X_mid, y_mid), (X_late, y_late)


def split_for_training_and_drift(df: pd.DataFrame, target_col: str, seed: int = 42):
    df_shuffled = shuffle(df, random_state=seed).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_drift = int(n_total * 0.33)

    df_drift = df_shuffled.iloc[:n_drift].copy()
    df_trainvaltest = df_shuffled.iloc[n_drift:].copy()

    n_drift_split = len(df_drift) // 3
    df_early = df_drift.iloc[:n_drift_split].copy()
    df_mid = df_drift.iloc[n_drift_split:2 * n_drift_split].copy()
    df_late = df_drift.iloc[2 * n_drift_split:].copy()

    drift_splits = {
        "early": df_early,
        "mid": df_mid,
        "late": df_late
    }

    return df_trainvaltest, drift_splits

