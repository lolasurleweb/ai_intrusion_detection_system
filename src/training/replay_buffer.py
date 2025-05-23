from collections import deque
import numpy as np
import pandas as pd

class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add_batch(self, X_new, y_new):
        for x, y in zip(X_new.iterrows(), y_new):
            self.buffer.append((x[1], y))

    def sample(self, n_old, X_new, y_new):
        # Aktuelle neue Daten (z. B. Alarmfeedback)
        df_new = pd.DataFrame(X_new)
        y_new = pd.Series(y_new)

        # Alte aus dem Buffer
        if len(self.buffer) < n_old:
            old_samples = list(self.buffer)
        else:
            old_samples = np.random.choice(list(self.buffer), size=n_old, replace=False)

        if old_samples:
            X_old, y_old = zip(*old_samples)
            X_old = pd.DataFrame(X_old)
            y_old = pd.Series(y_old)
        else:
            X_old = pd.DataFrame(columns=X_new.columns)
            y_old = pd.Series(dtype=int)

        # Kombinieren
        X_combined = pd.concat([X_old, df_new], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_old, y_new], axis=0).reset_index(drop=True)

        return X_combined, y_combined
