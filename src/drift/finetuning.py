from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import torch
import uuid

suffix = uuid.uuid4().hex[:6]

def fine_tune_tabnet(model, X_ft, y_ft, epochs=10, batch_size=512):
    X_ft = X_ft.astype(np.float32)
    y_ft = y_ft.astype(int)

    torch.save(model.network.state_dict(), f"checkpoints/before_{suffix}.pt")

    model.fit(
        X_train=X_ft.values,
        y_train=y_ft.values,
        max_epochs=epochs,
        from_unsupervised=True,
        patience=epochs,
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    torch.save(model.network.state_dict(), f"checkpoints/after_{suffix}.pt")

    return model
