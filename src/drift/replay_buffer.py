from collections import deque
import numpy as np
import pandas as pd
import random

class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add_batch(self, X_new, y_new):
        for x, y in zip(X_new.iterrows(), y_new):
            self.buffer.append((x[1], y))

    def sample(self, n_old, X_new, y_new):
        # Neue aktuelle Alarmdaten
        df_new = pd.DataFrame(X_new)
        y_new = pd.Series(y_new)

        # Alte Bufferdaten nach Label trennen
        buffer_list = list(self.buffer)
        old_tp = [(x, y) for (x, y) in buffer_list if y == 1]
        old_fp = [(x, y) for (x, y) in buffer_list if y == 0]

        # 3:1-Verhältnis – 75 % TP, 25 % FP
        n_tp = min(len(old_tp), int(n_old * 0.75))
        n_fp = min(len(old_fp), n_old - n_tp)

        sampled_tp = random.sample(old_tp, k=n_tp) if n_tp > 0 else []
        sampled_fp = random.sample(old_fp, k=n_fp) if n_fp > 0 else []

        sampled = sampled_tp + sampled_fp
        random.shuffle(sampled)

        if sampled:
            X_old, y_old = zip(*sampled)
            X_old = pd.DataFrame(X_old)
            y_old = pd.Series(y_old)
        else:
            X_old = pd.DataFrame(columns=X_new.columns)
            y_old = pd.Series(dtype=int)

        # Kombinieren mit aktuellen Daten
        X_combined = pd.concat([X_old, df_new], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_old, y_new], axis=0).reset_index(drop=True)

        return X_combined, y_combined
