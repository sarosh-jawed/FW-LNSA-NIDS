from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.manuscript_analysis import (
    bootstrap_paired_ci,
    build_calibration_reliability,
    holm_adjust,
    paired_fw_lnsa_analysis,
    rank_biserial_from_differences,
)


class ManuscriptAnalysisTests(unittest.TestCase):
    def test_holm_adjustment_is_monotonic_in_sorted_order(self) -> None:
        p = np.array([0.01, 0.04, 0.03])
        adjusted = holm_adjust(p)
        self.assertTrue(np.all(adjusted >= p))
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-12))

    def test_rank_biserial_direction(self) -> None:
        self.assertGreater(rank_biserial_from_differences([1, 2, 3]), 0)
        self.assertLess(rank_biserial_from_differences([-1, -2, -3]), 0)
        self.assertEqual(rank_biserial_from_differences([0, 0]), 0)

    def test_bootstrap_ci_is_reproducible(self) -> None:
        first = bootstrap_paired_ci([0.1, 0.2, 0.3], iterations=500, random_seed=9)
        second = bootstrap_paired_ci([0.1, 0.2, 0.3], iterations=500, random_seed=9)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.2)
        self.assertGreaterEqual(first[1], 0.2)

    def _fw_frame(self) -> pd.DataFrame:
        rows = []
        for seed in [100, 101, 102]:
            for method, f1, fpr in [("hamming", 0.5, 0.02), ("weighted_similarity", 0.7, 0.015)]:
                rows.append(
                    {
                        "dataset_key": "nsl_kdd",
                        "dataset": "NSL-KDD",
                        "method": method,
                        "method_name": "Hamming" if method == "hamming" else "Weighted Similarity",
                        "fs_size": 20,
                        "feature_set": "FS-20",
                        "target_fpr": 0.01,
                        "seed": seed,
                        "test_f1": f1 + seed / 100000,
                        "test_recall": f1,
                        "test_fpr": fpr,
                        "test_balanced_accuracy": f1,
                        "test_mcc": f1 - 0.1,
                        "retention_rate": 0.5,
                        "test_detection_time_sec": 0.1,
                        "test_prediction_records_per_sec": 1000,
                        "serialized_model_bytes": 100,
                        "test_peak_prediction_rss_delta_mb": 1.0,
                        "validation_fpr": 0.01,
                    }
                )
        return pd.DataFrame(rows)

    def test_paired_analysis_reports_weighted_wins(self) -> None:
        result = paired_fw_lnsa_analysis(self._fw_frame(), bootstrap_iterations=200)
        f1 = result[result["metric"] == "test_f1"].iloc[0]
        self.assertEqual(f1["pairs"], 3)
        self.assertEqual(f1["weighted_wins"], 3)
        self.assertGreater(f1["mean_difference_weighted_minus_hamming"], 0)

    def test_calibration_table_marks_target_miss(self) -> None:
        fw = self._fw_frame()
        baseline = pd.DataFrame(
            [
                {
                    "dataset_key": "nsl_kdd",
                    "dataset": "NSL-KDD",
                    "model_name": "Logistic Regression",
                    "feature_set": "FS-20",
                    "fs_size": 20,
                    "target_fpr": 0.01,
                    "seed": 100,
                    "validation_fpr": 0.009,
                    "test_fpr": 0.02,
                    "test_f1": 0.8,
                    "test_recall": 0.8,
                    "test_balanced_accuracy": 0.8,
                    "test_mcc": 0.6,
                }
            ]
        )
        table = build_calibration_reliability(fw, baseline)
        baseline_row = table[table["approach"] == "Baseline"].iloc[0]
        self.assertFalse(bool(baseline_row["test_target_met"]))
        self.assertAlmostEqual(baseline_row["test_absolute_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
