"""Validation tests for checkpoints 1 to 6.

These tests use small synthetic data so they can run in Colab or locally before
raw NSL-KDD and CICIDS2017 files are available.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.detector_generation import (
    build_detector_pool,
    negative_selection,
    predict_with_detectors,
)
from src.feature_selection import mutual_information_feature_selection
from src.matching import (
    hamming_distance_matrix,
    jaccard_similarity_matrix,
    weighted_hamming_distance_matrix,
    weighted_smc_similarity_matrix,
)
from src.preprocessing import prepare_nsl_kdd, to_binary_label
from src.representation import binary_median_split, validate_binary_matrix


class CoreModuleTests(unittest.TestCase):
    def test_label_conversion_is_case_insensitive(self) -> None:
        self.assertEqual(to_binary_label("normal"), 0)
        self.assertEqual(to_binary_label(" Normal "), 0)
        self.assertEqual(to_binary_label("neptune"), 1)

    def test_nsl_kdd_preprocessing_removes_label_leakage(self) -> None:
        normal_row = [0, "tcp", "http", "SF"] + [0] * 37 + ["normal", 20]
        attack_row = [1, "udp", "private", "REJ"] + [1] * 37 + ["neptune", 18]

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_file = Path(tmp_dir) / "KDDTrain+.txt"
            test_file = Path(tmp_dir) / "KDDTest+.txt"
            pd.DataFrame([normal_row, attack_row]).to_csv(train_file, header=False, index=False)
            pd.DataFrame([normal_row, attack_row]).to_csv(test_file, header=False, index=False)

            dataset = prepare_nsl_kdd(train_file, test_file)

        self.assertEqual(dataset.y_train.tolist(), [0, 1])
        self.assertNotIn("label", dataset.X_train.columns)
        self.assertNotIn("difficulty", dataset.X_train.columns)
        self.assertEqual(dataset.train_attack_categories.tolist(), ["Normal", "DoS"])
        self.assertEqual(dataset.X_train.shape[1], dataset.X_test.shape[1])

    def test_feature_selection_and_binary_representation(self) -> None:
        X_train = pd.DataFrame(
            {
                "a": [0.0, 0.1, 0.9, 1.0],
                "b": [1.0, 0.9, 0.1, 0.0],
                "c": [0.2, 0.2, 0.2, 0.2],
            }
        )
        y_train = np.array([0, 0, 1, 1])
        result = mutual_information_feature_selection(X_train, y_train, fs_size=2, random_seed=42)
        self.assertEqual(len(result.selected_features), 2)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0)

        representation = binary_median_split(X_train, X_train, result.selected_features)
        validate_binary_matrix(representation.X_train_bin, name="X_train_bin")
        self.assertEqual(representation.X_train_bin.shape, (4, 2))

    def test_matching_scores_are_correct(self) -> None:
        records = np.array([[1, 0, 1], [0, 0, 1]], dtype=np.int8)
        detectors = np.array([[1, 1, 1]], dtype=np.int8)
        weights = np.array([0.5, 0.25, 0.25])

        np.testing.assert_array_equal(hamming_distance_matrix(records, detectors).ravel(), [1, 2])
        np.testing.assert_allclose(weighted_hamming_distance_matrix(records, detectors, weights).ravel(), [0.25, 0.75])
        np.testing.assert_allclose(weighted_smc_similarity_matrix(records, detectors, weights).ravel(), [0.75, 0.25])
        np.testing.assert_allclose(jaccard_similarity_matrix(records, detectors).ravel(), [2 / 3, 1 / 3])

    def test_detector_generation_and_prediction(self) -> None:
        self_space = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int8)
        candidates = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=np.int8)
        pool = negative_selection(candidates, self_space, method="hamming", threshold=0)
        self.assertEqual(pool.retained_count, 2)
        self.assertEqual(pool.rejected_count, 1)

        X_test = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.int8)
        predictions = predict_with_detectors(X_test, pool, detection_threshold=0)
        self.assertEqual(predictions.tolist(), [1, 0])

    def test_weighted_smc_pool_uses_similarity_thresholds(self) -> None:
        self_space = np.array([[0, 0, 0]], dtype=np.int8)
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        pool = build_detector_pool(
            self_space,
            n_detectors=20,
            vector_length=3,
            method="weighted_smc",
            self_threshold=1.0,
            seed=42,
            weights=weights,
        )
        self.assertGreaterEqual(pool.candidate_count, pool.retained_count)


if __name__ == "__main__":
    unittest.main()
