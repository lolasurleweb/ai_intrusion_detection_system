import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.io import save_pickle

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

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
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

def scale_and_save_splits(splits: dict, scaler=None, path_prefix='data/processed/'):
    reference_columns = None

    for name, (X, y) in splits.items():
        if reference_columns is None:
            reference_columns = list(X.columns)
        else:
            assert list(X.columns) == reference_columns, f"Feature mismatch in split {name}"

        X_scaled, _ = scale_numerical(X, scaler=scaler, fit=False)
        df = pd.concat([X_scaled, y], axis=1)
        df.to_csv(f"{path_prefix}{name}.csv", index=False)
