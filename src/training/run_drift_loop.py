from src.training.drift_detection import PerformanceDriftDetector
from src.training.replay_buffer import ReplayBuffer
from src.training.finetuning import fine_tune_tabnet

drift_detector = PerformanceDriftDetector(acc_threshold=0.86)
replay_buffer = ReplayBuffer(max_size=500)

# In Evaluation-Schleife
for i, (X_val_batch, y_val_batch) in enumerate(streaming_val_data):

    y_proba = model.predict_proba(X_val_batch.values)[:, 1]
    y_pred = (y_proba > threshold).astype(int)

    acc = (y_pred == y_val_batch.values).mean()
    auc = roc_auc_score(y_val_batch, y_proba)

    if drift_detector.update(acc, auc):
        print(f"Drift erkannt bei Batch {i} mit Accuracy {acc:.3f}, AUC {auc:.3f}")

        # Simuliere Label Feedback nur bei Alarms
        alarms = y_pred == 1
        if alarms.sum() > 0:
            X_alarm = X_val_batch[alarms]
            y_alarm = y_val_batch[alarms]

            X_ft, y_ft = replay_buffer.sample(n_old=50, X_new=X_alarm, y_new=y_alarm)

            # Fine-Tune
            model = fine_tune_tabnet(model, X_ft, y_ft)
            print("✓ Modell nach Driftwarnung feinjustiert.")

            # Buffer updaten
            replay_buffer.add_batch(X_alarm, y_alarm)
