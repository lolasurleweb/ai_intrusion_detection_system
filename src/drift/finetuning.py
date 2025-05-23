from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def fine_tune_tabnet(model, X_ft, y_ft, epochs=10, batch_size=512):
    # Optional: Learning Rate Reduktion
    fine_tune_model = TabNetClassifier(
        input_dim=X_ft.shape[1],
        output_dim=2,
        n_d=model.n_d,
        n_a=model.n_a,
        n_steps=model.n_steps,
        gamma=model.gamma,
        lambda_sparse=model.lambda_sparse,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=[],
        mask_type=model.mask_type,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-3),
        scheduler_params={"step_size": 5, "gamma": 0.95},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42
    )

    fine_tune_model.network.load_state_dict(model.network.state_dict())  # Übernehme alten Zustand

    fine_tune_model.fit(
        X_train=X_ft.values, y_train=y_ft.values,
        max_epochs=epochs,
        patience=5,
        batch_size=batch_size,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    return fine_tune_model