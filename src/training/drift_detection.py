from river.drift import ADWIN
import numpy as np

class PerformanceDriftDetector:
    def __init__(self, acc_threshold=0.85, min_instances=30):
        self.adwin_auc = ADWIN()
        self.adwin_acc = ADWIN()
        self.acc_threshold = acc_threshold
        self.min_instances = min_instances
        self.warnings = []

    def update(self, acc, auc):
        self.adwin_auc.update(auc)
        self.adwin_acc.update(acc)

        drift_acc = self.adwin_acc.change_detected and len(self.adwin_acc.window) > self.min_instances
        drift_auc = self.adwin_auc.change_detected and len(self.adwin_auc.window) > self.min_instances
        drift_hard = acc < self.acc_threshold

        if drift_acc or drift_auc or drift_hard:
            self.warnings.append((acc, auc))
            return True

        return False
