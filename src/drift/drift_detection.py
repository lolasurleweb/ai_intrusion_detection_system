class PerformanceDriftDetector:
    def __init__(self, ref_cost: float, rel_increase: float = 0.15):
        self.ref_cost = ref_cost
        self.rel_increase = rel_increase
        self.warnings = []

    def update(self, current_cost: float) -> bool:
        drift = current_cost > self.ref_cost * (1 + self.rel_increase)
        if drift:
            self.warnings.append(current_cost)
        return drift