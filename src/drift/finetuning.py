from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import torch
import uuid
from src.data.preprocessing import NUMERICAL_FEATURES
import joblib
import os

suffix = uuid.uuid4().hex[:6]

def fine_tune_tabnet(model, X_ft, y_ft, epochs=10, batch_size=512):
    os.makedirs("checkpoints", exist_ok=True)

    scaler = joblib.load("data/processed/scaler.pkl")
    X_ft[NUMERICAL_FEATURES] = scaler.transform(X_ft[NUMERICAL_FEATURES])

    X_ft = X_ft.astype(np.float32)
    y_ft = y_ft.astype(int)

    torch.save(model.network.state_dict(), f"checkpoints/before_{suffix}.pt")

    model.fit(
        X_train=X_ft.values,
        y_train=y_ft.values,
        max_epochs=epochs,
        patience=epochs,
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=True
    )

    torch.save(model.network.state_dict(), f"checkpoints/after_{suffix}.pt")

    return model
