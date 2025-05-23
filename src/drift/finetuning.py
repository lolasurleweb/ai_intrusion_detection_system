from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def fine_tune_tabnet(model, X_ft, y_ft, epochs=10, batch_size=512):
    model.fit(
        X_train=X_ft.values,
        y_train=y_ft.values,
        max_epochs=epochs,
        patience=epochs,  # kein Early Stopping
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        from_unsupervised=None
    )
    return model