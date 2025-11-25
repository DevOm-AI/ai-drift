import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.stats import ks_2samp
from scipy.stats import chi2_contingency

from alibi_detect.cd import KSDrift

# from .dashboard import DriftDashboard
# from .dashboard_cyberpunk import CyberpunkDashboard
from .dashboard_professional import ProfessionalDashboard


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
    

    def ks_test_drift(self, current: pd.DataFrame, p_threshold: float = 0.05):
        """
        Detect drift using Kolmogorov-Smirnov test (numeric features).
        Returns features where p-value < p_threshold.
        """
        drifted = []

        for col in self.numeric_features:
            ref_values = self.reference[col].dropna()
            cur_values = current[col].dropna()

            stat, p_value = ks_2samp(ref_values, cur_values)

            if p_value < p_threshold:
                drifted.append(col)

        return drifted

    def chi_square_drift(self, current: pd.DataFrame, p_threshold: float = 0.05):
        """
        Detect drift for categorical features using Chi-square test.
        Returns categorical columns where p-value < p_threshold.
        """
        drifted = []

        for col in self.categorical_features:
            # Create contingency table of value counts
            ref_counts = self.reference[col].value_counts()
            cur_counts = current[col].value_counts()

            # Align categories
            combined_index = ref_counts.index.union(cur_counts.index)
            ref_aligned = ref_counts.reindex(combined_index, fill_value=0)
            cur_aligned = cur_counts.reindex(combined_index, fill_value=0)

            # Build contingency table
            contingency_table = [ref_aligned.values, cur_aligned.values]

            chi2, p_value, _, _ = chi2_contingency(contingency_table)

            if p_value < p_threshold:
                drifted.append(col)

        return drifted



    def detect_drift(self, current: pd.DataFrame):
        """
        Runs all drift tests and returns a unified drift report.
        """
        report = {
            "z_score_drift": self.z_score_drift(current),
            "ks_test_drift": self.ks_test_drift(current),
            "chi_square_drift": self.chi_square_drift(current),
        }

        # Combine unique drifted features
        all_drift = set(
            report["z_score_drift"]
            + report["ks_test_drift"]
            + report["chi_square_drift"]
        )

        report["all_drifted_features"] = list(all_drift)

        return report
        

    def alibi_drift(self, current: pd.DataFrame):
        results = {}

        for col in self.numeric_features:
            ref_vals = self.reference[col].values.reshape(-1, 1)
            cur_vals = current[col].values.reshape(-1, 1)

            cd = KSDrift(ref_vals, p_val=0.05)
            preds = cd.predict(cur_vals)

            is_drift  = bool(preds["data"]["is_drift"])
            p_value = float(preds["data"]["p_val"].item())

            results[col] = {
                "is_drift": is_drift,
                "p_value": p_value
            }

        return results


    def save_evidently_dashboard(self, report, path="artifacts/evidently_report.html"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        report.save(path)
        print(f"Evidently dashboard saved at {path}")




if __name__ == "__main__":
    ref = pd.read_csv("src/ai_drift/data/reference.csv")
    cur = pd.read_csv("src/ai_drift/data/current_sudden.csv")

    detector = DriftDetector(ref)

    unified = detector.detect_drift(cur)
    alibi = detector.alibi_drift(cur)

    dash = ProfessionalDashboard(detector)
    dash.save_html(ref, cur, unified, alibi, path="artifacts/drift_dashboard_professional.html")

