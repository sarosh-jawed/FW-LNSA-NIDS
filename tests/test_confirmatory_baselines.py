from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.confirmatory_baselines import (
    _attack_scores,
    _fit_and_score,
    load_manuscript_config,
    load_locked_feature_sets,
    summarize_confirmatory_baselines,
    verify_confirmatory_contract,
)
from src.baselines import build_baseline_model
from src.preprocessing import PreparedDataset
from src.utils import stable_feature_selection_signature, stable_partition_signature
from sklearn.preprocessing import MinMaxScaler


class ConfirmatoryBaselineTests(unittest.TestCase):
    def _dataset(self) -> PreparedDataset:
        X_train = pd.DataFrame({"a": [0.0, 0.1, 0.8, 0.9], "b": [0.0, 0.2, 0.7, 1.0]})
        X_validation = pd.DataFrame({"a": [0.05, 0.15, 0.75, 0.95], "b": [0.1, 0.1, 0.8, 0.9]})
        X_test = pd.DataFrame({"a": [0.02, 0.25, 0.7, 0.98], "b": [0.0, 0.3, 0.6, 1.0]})
        labels = pd.Series(["normal", "normal", "attack", "attack"])
        return PreparedDataset(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=np.array([0, 0, 1, 1], dtype=np.int8),
            y_validation=np.array([0, 0, 1, 1], dtype=np.int8),
            y_test=np.array([0, 0, 1, 1], dtype=np.int8),
            train_original_labels=labels,
            validation_original_labels=labels.copy(),
            test_original_labels=labels.copy(),
            train_attack_categories=pd.Series(["Normal", "Normal", "Attack", "Attack"]),
            validation_attack_categories=pd.Series(["Normal", "Normal", "Attack", "Attack"]),
            test_attack_categories=pd.Series(["Normal", "Normal", "Attack", "Attack"]),
            scaler=MinMaxScaler(),
            metadata={"dataset": "Synthetic"},
        )

    def test_profile_loads(self) -> None:
        config = load_manuscript_config("configs/manuscript_readiness.yaml", profile_name="smoke")
        self.assertEqual(config["profile_name"], "smoke")
        self.assertEqual(config["baseline"]["seeds"], [100])

    def test_supervised_scores_are_attack_oriented(self) -> None:
        dataset = self._dataset()
        model = build_baseline_model("logistic_regression", {}, seed=100)
        validation_scores, test_scores, resources = _fit_and_score(
            "logistic_regression",
            model,
            dataset.X_train,
            dataset.y_train,
            dataset.X_validation,
            dataset.X_test,
        )
        self.assertEqual(validation_scores.shape, (4,))
        self.assertEqual(test_scores.shape, (4,))
        self.assertGreater(validation_scores[2:].mean(), validation_scores[:2].mean())
        self.assertGreater(resources["serialized_model_bytes"], 0)

    def test_isolation_forest_scores_are_attack_oriented(self) -> None:
        dataset = self._dataset()
        model = build_baseline_model("isolation_forest", {"n_estimators": 5, "n_jobs": 1}, seed=100)
        model.fit(dataset.X_train.loc[dataset.y_train == 0])
        scores = _attack_scores("isolation_forest", model, dataset.X_test)
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.isfinite(scores).all())

    def test_contract_verifies_hashes_and_feature_order(self) -> None:
        dataset = self._dataset()
        features = ["a", "b"]
        weights = np.array([0.6, 0.4])
        mi = np.array([0.3, 0.2])
        signature = stable_feature_selection_signature(features, mi, weights)
        feature_sets = {
            2: SimpleNamespace(
                selected_features=features,
                feature_selection_hash=signature,
            )
        }
        partitions = {
            "train_partition_hash": stable_partition_signature(dataset.X_train, dataset.y_train, dataset.train_original_labels),
            "validation_partition_hash": stable_partition_signature(dataset.X_validation, dataset.y_validation, dataset.validation_original_labels),
            "test_partition_hash": stable_partition_signature(dataset.X_test, dataset.y_test, dataset.test_original_labels),
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "synthetic" / "manifests").mkdir(parents=True)
            (root / "synthetic" / "tables").mkdir(parents=True)
            manifest = {
                "dataset_key": "synthetic",
                "partition_hashes": partitions,
                "feature_selection_hashes": {"FS-2": signature},
                "test_metrics_used_for_selection": False,
            }
            (root / "synthetic" / "manifests" / "execution_manifest.json").write_text(json.dumps(manifest))
            pd.DataFrame(
                [
                    {"fs_size": 2, "rank": 1, "feature": "a", "normalized_weight": 0.6, "feature_selection_hash": signature},
                    {"fs_size": 2, "rank": 2, "feature": "b", "normalized_weight": 0.4, "feature_selection_hash": signature},
                ]
            ).to_csv(root / "synthetic" / "tables" / "selected_features.csv", index=False)
            contract = verify_confirmatory_contract("synthetic", dataset, feature_sets, root)
            self.assertEqual(contract["partition_hashes"], partitions)

    def test_baseline_summary_preserves_target_operating_points(self) -> None:
        rows = []
        for seed in [100, 101]:
            rows.append(
                {
                    "dataset_key": "x",
                    "dataset": "X",
                    "model": "logistic_regression",
                    "model_name": "Logistic Regression",
                    "model_family": "Linear",
                    "feature_set": "FS-10",
                    "fs_size": 10,
                    "target_fpr": 0.01,
                    "model_parameters_json": "{}",
                    "feature_selection_hash": "f",
                    "train_partition_hash": "tr",
                    "validation_partition_hash": "va",
                    "test_partition_hash": "te",
                    "seed": seed,
                    "validation_accuracy": 0.8,
                    "validation_precision": 0.8,
                    "validation_recall": 0.8,
                    "validation_f1": 0.8,
                    "validation_fpr": 0.01,
                    "validation_balanced_accuracy": 0.8,
                    "validation_mcc": 0.6,
                    "validation_pr_auc": 0.8,
                    "validation_roc_auc": 0.8,
                    "test_accuracy": 0.75,
                    "test_precision": 0.75,
                    "test_recall": 0.75,
                    "test_f1": 0.75,
                    "test_fpr": 0.02,
                    "test_balanced_accuracy": 0.75,
                    "test_mcc": 0.5,
                    "test_pr_auc": 0.75,
                    "test_roc_auc": 0.75,
                    "calibration_fpr": 0.01,
                    "fit_time_sec": 1.0,
                    "validation_score_time_sec": 0.1,
                    "test_score_time_sec": 0.1,
                    "total_time_sec": 1.2,
                    "validation_records_per_sec": 100.0,
                    "test_records_per_sec": 100.0,
                    "peak_memory_increase_mb": 1.0,
                    "serialized_model_bytes": 1000,
                }
            )
        summary = summarize_confirmatory_baselines(pd.DataFrame(rows))
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["runs"], 2)
        self.assertAlmostEqual(summary.iloc[0]["test_f1_mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
