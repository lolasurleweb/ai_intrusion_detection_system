from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np

def fine_tune_tabnet(model, X_ft, y_ft, epochs=10, batch_size=512):
    X_ft = X_ft.astype(np.float32)
    y_ft = y_ft.astype(int)

    model.fit(
        X_train=X_ft.values,
        y_train=y_ft.values,
        max_epochs=epochs,
        patience=epochs,
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    return model
