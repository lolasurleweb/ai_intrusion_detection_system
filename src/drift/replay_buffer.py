from collections import deque
import pandas as pd

class ReplayBuffer:
    def __init__(self, maxlen=500):
        self.buffer = deque(maxlen=maxlen)

    def add(self, x, y):
        self.buffer.append((x, y))

    def get_data(self):
        if len(self.buffer) == 0:
            return pd.DataFrame(), pd.Series(dtype=int)
        X, y = zip(*self.buffer)
        return pd.DataFrame(X), pd.Series(y)