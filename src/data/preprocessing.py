import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=['session_id'], inplace=True, errors='ignore')
    df['encryption_used'] = df['encryption_used'].fillna('None')
    return df

def log_transform(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col + '_log'] = np.log1p(df[col])
    df.drop(columns=[col], inplace=True)
    return df

def encode_and_scale(df: pd.DataFrame, target_col: str = 'attack_detected'):
    df = df.copy()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    numerical_features = ['network_packet_size', 'login_attempts', 'ip_reputation_score', 'failed_logins', 'session_duration_log']
    scaler = StandardScaler()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    return X, y, scaler

def create_splits(X, y, test_size=0.2, val_size=0.25, seed=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, path_prefix='data/processed/'):
    pd.concat([X_train, y_train], axis=1).to_csv(f'{path_prefix}train_data.csv', index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(f'{path_prefix}val_data.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f'{path_prefix}test_data.csv', index=False)
