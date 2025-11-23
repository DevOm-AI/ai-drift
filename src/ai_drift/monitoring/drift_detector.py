import pandas as pd
import numpy as np
from typing import Dict, List


class DriftDetector:
    def __init__(self, reference: pd.DataFrame):
        self.reference = reference.copy()
        self.numeric_features = reference.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = reference.select_dtypes(exclude=[np.number]).columns.tolist()

    def compute_basic_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Return mean and std for numeric features."""
        stats = {}
        for col in self.numeric_features:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std() if df[col].std() != 0 else 1.0)
            }
        return stats

    def z_score_drift(self, current: pd.DataFrame, threshold: float = 0.8) -> List[str]:
        """Detect drift using simple z-score between reference and current means."""
        drifted_features = []
        ref_stats = self.compute_basic_stats(self.reference)
        cur_stats = self.compute_basic_stats(current)

        for col in self.numeric_features:
            ref_mean = ref_stats[col]["mean"]
            ref_std = ref_stats[col]["std"]
            cur_mean = cur_stats[col]["mean"]

            z = abs((cur_mean - ref_mean) / ref_std)
            if z > threshold:
                drifted_features.append(col)

        return drifted_features



if __name__ == "__main__":
    ref = pd.read_csv("src/ai_drift/data/reference.csv")
    cur = pd.read_csv("src/ai_drift/data/current_sudden.csv")

    detector = DriftDetector(ref)
    drifted = detector.z_score_drift(cur)

    print("Drifted Features:", drifted)
