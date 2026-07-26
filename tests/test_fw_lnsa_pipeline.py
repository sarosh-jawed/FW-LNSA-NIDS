"""Validation tests for the FW-LNSA model and NSL-KDD experiment pipeline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import ConfigurationError, load_yaml_config, resolve_profile
from src.evaluation import attack_category_analysis, evaluate_binary_classification
from src.experiments import run_prepared_nsl_kdd_experiments
from src.fw_lnsa import FWLNSA
from src.preprocessing import NSL_KDD_COLUMNS, PreparedDataset


class FWLNSAPipelineTests(unittest.TestCase):
    def _prepared_dataset(self) -> PreparedDataset:
        X_train = pd.DataFrame(
            {
                "f1": [0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 0.9, 0.9, 0.2, 0.8, 0.3, 0.7],
                "f2": [0.0, 0.1, 0.0, 0.1, 1.0, 0.9, 1.0, 0.9, 0.2, 0.8, 0.3, 0.7],
                "f3": [0.1, 0.0, 0.1, 0.0, 0.9, 1.0, 0.9, 1.0, 0.2, 0.8, 0.3, 0.7],
                "f4": [0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 0.9, 0.9, 0.3, 0.7, 0.2, 0.8],
                "f5": [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.4, 0.6, 0.4, 0.6],
                "f6": [0.0, 0.2, 0.0, 0.2, 1.0, 0.8, 1.0, 0.8, 0.3, 0.7, 0.4, 0.6],
            }
        )
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=np.int8)
        X_test = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 0.1, 0.9, 0.2, 0.8],
                "f2": [0.0, 1.0, 0.1, 0.9, 0.2, 0.8],
                "f3": [0.0, 1.0, 0.1, 0.9, 0.2, 0.8],
                "f4": [0.0, 1.0, 0.1, 0.9, 0.2, 0.8],
                "f5": [0.2, 0.8, 0.2, 0.8, 0.4, 0.6],
                "f6": [0.0, 1.0, 0.2, 0.8, 0.3, 0.7],
            }
        )
        y_test = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
        scaler = MinMaxScaler().fit(X_train)
        return PreparedDataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_original_labels=pd.Series(["normal"] * 6 + ["attack"] * 6),
            test_original_labels=pd.Series(["normal", "neptune", "normal", "satan", "normal", "guess_passwd"]),
            train_attack_categories=pd.Series(["Normal"] * 6 + ["DoS"] * 6),
            test_attack_categories=pd.Series(["Normal", "DoS", "Normal", "Probe", "Normal", "R2L"]),
            scaler=scaler,
            metadata={"dataset": "NSL-KDD", "train_records": 12, "test_records": 6},
        )

    def _experiment_config(self, output_dir: str) -> dict:
        return {
            "profile_name": "test",
            "dataset": {"name": "nsl_kdd", "task": "binary"},
            "paths": {
                "train_file": "unused",
                "test_file": "unused",
                "output_dir": output_dir,
            },
            "preprocessing": {"scale": True, "max_self_samples": 6},
            "feature_selection": {
                "method": "mutual_information",
                "feature_sizes": [4],
                "main_feature_size": 4,
                "random_seed": 42,
            },
            "representation": {"method": "binary_median_split"},
            "experiment": {
                "methods": ["hamming", "weighted_hamming", "weighted_smc", "jaccard"],
                "detector_budgets": [32],
                "seeds": [42],
                "deduplicate_candidates": False,
                "prediction_chunk_size": 10,
            },
            "method_settings": {
                "hamming": {
                    "threshold_scale": "ratio",
                    "self_thresholds": [0.0],
                    "detection_thresholds": [0.0],
                },
                "weighted_hamming": {
                    "threshold_scale": "direct",
                    "self_thresholds": [0.0],
                    "detection_thresholds": [0.0],
                },
                "weighted_smc": {
                    "threshold_scale": "direct",
                    "self_thresholds": [1.0],
                    "detection_thresholds": [1.0],
                },
                "jaccard": {
                    "threshold_scale": "direct",
                    "self_thresholds": [1.0],
                    "detection_thresholds": [1.0],
                },
            },
            "selection": {"balanced_fpr_limit": 0.10},
            "outputs": {
                "results_table": "nsl_kdd_fw_lnsa_results.csv",
                "method_summary": "nsl_kdd_matching_method_summary.csv",
                "balanced_configs": "nsl_kdd_best_balanced_configs.csv",
                "attack_category_analysis": "nsl_kdd_attack_category_analysis.csv",
                "selected_features": "nsl_kdd_selected_features.csv",
            },
        }

    def test_weighted_smc_model_fit_predict_and_scores(self) -> None:
        X_train = np.array(
            [[0, 0, 0, 0]] * 4 + [[1, 1, 1, 1]] * 4,
            dtype=np.int8,
        )
        y_train = np.array([0] * 4 + [1] * 4, dtype=np.int8)
        weights = np.full(4, 0.25)

        model = FWLNSA(
            method="weighted_smc",
            n_detectors=100,
            self_threshold=1.0,
            detection_threshold=1.0,
            random_seed=42,
            max_self_samples=4,
        )
        model.fit(X_train, y_train, feature_weights=weights)
        predictions = model.predict(np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int8))
        scores = model.decision_scores(np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int8))
        summary = model.get_detector_summary()

        self.assertTrue(model.is_fitted)
        self.assertEqual(predictions.tolist(), [0, 1])
        self.assertTrue(np.all((scores >= 0.0) & (scores <= 1.0)))
        self.assertGreater(summary["retained_detectors"], 0)
        self.assertIsNotNone(summary["detection_time_sec"])

    def test_binary_metrics_and_category_analysis(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1], dtype=np.int8)
        y_pred = np.array([0, 1, 1, 0, 1], dtype=np.int8)
        metrics = evaluate_binary_classification(y_true, y_pred)
        self.assertEqual((metrics.tn, metrics.fp, metrics.fn, metrics.tp), (1, 1, 1, 2))
        self.assertAlmostEqual(metrics.fpr, 0.5)
        self.assertAlmostEqual(metrics.recall, 2 / 3)

        table = attack_category_analysis(
            y_true,
            y_pred,
            ["Normal", "Normal", "DoS", "Probe", "DoS"],
        )
        normal = table[table["category"] == "Normal"].iloc[0]
        dos = table[table["category"] == "DoS"].iloc[0]
        self.assertEqual(normal["false_alarms"], 1)
        self.assertEqual(dos["detected_attacks"], 2)

    def test_experiment_pipeline_saves_required_tables(self) -> None:
        dataset = self._prepared_dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = run_prepared_nsl_kdd_experiments(
                dataset,
                self._experiment_config(tmp_dir),
                save_outputs=True,
            )

            self.assertEqual(len(outputs.results), 4)
            self.assertEqual(set(outputs.results["method"]), {"hamming", "weighted_hamming", "weighted_smc", "jaccard"})
            self.assertEqual(len(outputs.balanced_configs), 4)
            self.assertFalse(outputs.attack_category_analysis.empty)
            self.assertEqual(len(outputs.selected_features), 4)
            for path in outputs.output_paths.values():
                self.assertTrue(path.exists(), path)

    def test_configuration_profiles_are_loaded_and_validated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "nsl_kdd_fw_lnsa.yaml"
        loaded = load_yaml_config(config_path)
        smoke = resolve_profile(loaded, "smoke")
        full = resolve_profile(loaded, "full")
        self.assertEqual(smoke["experiment"]["detector_budgets"], [50])
        self.assertIn("weighted_smc", full["experiment"]["methods"])
        self.assertEqual(full["feature_selection"]["main_feature_size"], 20)

        loaded["profiles"]["smoke"]["experiment"]["methods"] = ["unknown"]
        with self.assertRaises(ConfigurationError):
            resolve_profile(loaded, "smoke")

    def test_command_line_dry_run_with_small_nsl_kdd_files(self) -> None:
        normal_row = [0, "tcp", "http", "SF"] + [0] * 37 + ["normal", 20]
        attack_row = [1, "udp", "private", "REJ"] + [1] * 37 + ["neptune", 18]

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_file = Path(tmp_dir) / "KDDTrain+.txt"
            test_file = Path(tmp_dir) / "KDDTest+.txt"
            pd.DataFrame([normal_row, attack_row] * 4, columns=NSL_KDD_COLUMNS).to_csv(
                train_file,
                header=False,
                index=False,
            )
            pd.DataFrame([normal_row, attack_row], columns=NSL_KDD_COLUMNS).to_csv(
                test_file,
                header=False,
                index=False,
            )

            repo_root = Path(__file__).resolve().parents[1]
            command = [
                sys.executable,
                "scripts/run_nsl_kdd_fw_lnsa.py",
                "--config",
                "configs/nsl_kdd_fw_lnsa.yaml",
                "--profile",
                "smoke",
                "--train-file",
                str(train_file),
                "--test-file",
                str(test_file),
                "--output-dir",
                str(Path(tmp_dir) / "results"),
                "--dry-run",
            ]
            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Configuration and data paths are valid.", completed.stdout)


if __name__ == "__main__":
    unittest.main()
