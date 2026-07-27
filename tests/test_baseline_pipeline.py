"""Validation tests for baseline models and fair FW-LNSA comparisons."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler

from src.baselines import (
    build_fw_lnsa_comparison,
    run_prepared_baseline_experiments,
    run_prepared_baseline_grid,
    summarize_baseline_results,
)
from src.config import (
    ConfigurationError,
    load_yaml_config,
    resolve_baseline_profile,
)
from src.experiments import run_prepared_nsl_kdd_experiments
from src.preprocessing import NSL_KDD_COLUMNS, PreparedDataset


class BaselinePipelineTests(unittest.TestCase):
    def _prepared_dataset(self) -> PreparedDataset:
        rng = np.random.default_rng(42)
        normal_train = rng.normal(0.20, 0.04, size=(30, 6))
        attack_train = rng.normal(0.80, 0.04, size=(30, 6))
        normal_test = rng.normal(0.20, 0.04, size=(10, 6))
        attack_test = rng.normal(0.80, 0.04, size=(10, 6))
        X_train = pd.DataFrame(
            np.vstack([normal_train, attack_train]),
            columns=[f"f{i}" for i in range(1, 7)],
        )
        X_test = pd.DataFrame(
            np.vstack([normal_test, attack_test]),
            columns=X_train.columns,
        )
        y_train = np.array([0] * 30 + [1] * 30, dtype=np.int8)
        y_test = np.array([0] * 10 + [1] * 10, dtype=np.int8)
        scaler = MinMaxScaler().fit(X_train)
        return PreparedDataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_original_labels=pd.Series(["normal"] * 30 + ["attack"] * 30),
            test_original_labels=pd.Series(["normal"] * 10 + ["neptune"] * 5 + ["satan"] * 5),
            train_attack_categories=pd.Series(["Normal"] * 30 + ["DoS"] * 30),
            test_attack_categories=pd.Series(["Normal"] * 10 + ["DoS"] * 5 + ["Probe"] * 5),
            scaler=scaler,
            metadata={"dataset": "NSL-KDD", "train_records": 60, "test_records": 20},
        )

    def _baseline_config(self, output_dir: str) -> dict:
        return {
            "profile_name": "test",
            "feature_selection": {
                "method": "mutual_information",
                "feature_sizes": [4],
                "main_feature_size": 4,
                "random_seed": 42,
                "max_samples": None,
            },
            "experiment": {"seeds": [42]},
            "models": {
                "logistic_regression": {
                    "enabled": True,
                    "parameters": {"solver": "liblinear", "max_iter": 500},
                },
                "decision_tree": {
                    "enabled": True,
                    "parameters": {"max_depth": 5, "min_samples_leaf": 1},
                },
                "random_forest": {
                    "enabled": True,
                    "parameters": {"n_estimators": 10, "n_jobs": 1},
                },
                "isolation_forest": {
                    "enabled": True,
                    "parameters": {"n_estimators": 10, "n_jobs": 1},
                },
            },
            "selection": {"balanced_fpr_limit": 0.10},
            "outputs": {
                "nsl_kdd": {
                    "baseline_results": "nsl_kdd_baseline_results.csv",
                    "comparison_table": "nsl_kdd_fw_lnsa_vs_baselines.csv",
                    "attack_category_analysis": "nsl_kdd_baseline_attack_category_analysis.csv",
                    "selected_features": "nsl_kdd_baseline_selected_features.csv",
                }
            },
        }

    def _fw_config(self, output_dir: str) -> dict:
        return {
            "profile_name": "test",
            "dataset": {"name": "nsl_kdd", "task": "binary"},
            "paths": {"train_file": "unused", "test_file": "unused", "output_dir": output_dir},
            "preprocessing": {"scale": True, "max_self_samples": 30},
            "feature_selection": {
                "method": "mutual_information",
                "feature_sizes": [4],
                "main_feature_size": 4,
                "random_seed": 42,
                "max_samples": None,
            },
            "representation": {"method": "binary_median_split"},
            "experiment": {
                "methods": ["weighted_smc"],
                "detector_budgets": [20],
                "seeds": [42],
                "deduplicate_candidates": True,
                "prediction_chunk_size": 10,
            },
            "method_settings": {
                "weighted_smc": {
                    "threshold_scale": "direct",
                    "self_thresholds": [0.8],
                    "detection_thresholds": [0.75],
                }
            },
            "selection": {"balanced_fpr_limit": 0.10},
            "outputs": {
                "results_table": "nsl_kdd_fw_lnsa_results.csv",
                "method_summary": "summary.csv",
                "balanced_configs": "balanced.csv",
                "attack_category_analysis": "categories.csv",
                "selected_features": "features.csv",
            },
        }

    def _write_cic_archive(self, root: Path) -> Path:
        archive_path = root / "MachineLearningCSV.zip"
        columns = [" Flow Duration ", "Flow Bytes/s", "Packet Length", "Label"]
        rows = []
        labels = ["BENIGN", "DDoS", "PortScan", "FTP-Patator"]
        for index in range(80):
            label = labels[index % len(labels)]
            base = 0.1 if label == "BENIGN" else 0.8
            rows.append([base + index * 0.0001, base * 10, index % 13, label])
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "MachineLearningCVE/sample.csv",
                pd.DataFrame(rows, columns=columns).to_csv(index=False),
            )
        return archive_path

    def test_baseline_models_are_deterministic_and_isolation_forest_uses_normal_only(self) -> None:
        dataset = self._prepared_dataset()
        config = self._baseline_config("unused")
        first, _, _ = run_prepared_baseline_grid(dataset, config, dataset_key="nsl_kdd")
        second, _, _ = run_prepared_baseline_grid(dataset, config, dataset_key="nsl_kdd")

        metric_columns = ["accuracy", "precision", "recall", "f1", "fpr", "fnr"]
        pd.testing.assert_frame_equal(
            first.sort_values("model")[metric_columns].reset_index(drop=True),
            second.sort_values("model")[metric_columns].reset_index(drop=True),
        )
        isolation = first[first["model"] == "isolation_forest"].iloc[0]
        self.assertEqual(isolation["training_records_used"], 30)
        self.assertEqual(isolation["normal_training_records_used"], 30)
        self.assertEqual(isolation["attack_training_records_used"], 0)
        self.assertTrue((first["tn"] + first["fp"] + first["fn"] + first["tp"] == 20).all())

    def test_seed_repeats_are_aggregated_under_one_model_configuration(self) -> None:
        dataset = self._prepared_dataset()
        config = self._baseline_config("unused")
        config["experiment"]["seeds"] = [42, 43]
        for model_name, settings in config["models"].items():
            settings["enabled"] = model_name == "decision_tree"

        results, _, _ = run_prepared_baseline_grid(
            dataset,
            config,
            dataset_key="nsl_kdd",
        )
        summary = summarize_baseline_results(results)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["runs"], 2)
        self.assertEqual(summary.iloc[0]["seeds"], "42|43")

    def test_baseline_outputs_and_resource_columns(self) -> None:
        dataset = self._prepared_dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = run_prepared_baseline_experiments(
                dataset,
                self._baseline_config(tmp_dir),
                dataset_key="nsl_kdd",
                output_dir=tmp_dir,
                fw_lnsa_results_path=Path(tmp_dir) / "missing.csv",
            )
            self.assertEqual(len(outputs.results), 4)
            self.assertEqual(len(outputs.output_paths), 4)
            self.assertFalse(outputs.attack_category_analysis.empty)
            self.assertIn("peak_memory_increase_mb", outputs.results.columns)
            self.assertIn("serialized_model_size_kb", outputs.results.columns)
            self.assertIn("train_partition_hash", outputs.results.columns)
            self.assertTrue((outputs.results["serialized_model_size_kb"] > 0).all())
            for path in outputs.output_paths.values():
                self.assertTrue(path.exists(), path)

    def test_fw_lnsa_results_include_partition_signatures(self) -> None:
        dataset = self._prepared_dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = run_prepared_nsl_kdd_experiments(
                dataset,
                self._fw_config(tmp_dir),
                save_outputs=False,
            )
            self.assertIn("train_partition_hash", outputs.results.columns)
            self.assertIn("test_partition_hash", outputs.results.columns)
            self.assertEqual(outputs.results.iloc[0]["train_records"], 60)
            self.assertEqual(outputs.results.iloc[0]["test_records"], 20)

    def test_comparison_includes_only_compatible_fw_lnsa_rows(self) -> None:
        dataset = self._prepared_dataset()
        config = self._baseline_config("unused")
        baseline_results, _, _ = run_prepared_baseline_grid(dataset, config, dataset_key="nsl_kdd")
        first = baseline_results.iloc[0]
        fw_row = {
            "profile": "test",
            "method": "weighted_smc",
            "method_name": "Weighted SMC",
            "method_role": "main_operational_variant",
            "feature_set": "FS-4",
            "fs_size": 4,
            "seed": 42,
            "detector_budget": 100,
            "self_threshold_config": 0.8,
            "detection_threshold_config": 0.75,
            "threshold_scale": "direct",
            "train_records": 60,
            "test_records": 20,
            "train_partition_hash": first["train_partition_hash"],
            "test_partition_hash": first["test_partition_hash"],
            "accuracy": 0.9,
            "precision": 0.9,
            "recall": 0.9,
            "f1": 0.9,
            "fpr": 0.05,
            "fnr": 0.1,
            "total_time_sec": 0.2,
            "generation_time_sec": 0.1,
            "detection_time_sec": 0.1,
            "retained_detectors": 50,
            "detector_memory_bytes": 1000,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fw_path = Path(tmp_dir) / "fw.csv"
            pd.DataFrame([fw_row]).to_csv(fw_path, index=False)
            comparison = build_fw_lnsa_comparison(
                baseline_results,
                fw_lnsa_results_path=fw_path,
                fpr_limit=0.10,
            )
            self.assertIn("FW-LNSA", set(comparison["approach"]))
            self.assertIn("Baseline", set(comparison["approach"]))

            fw_row["train_partition_hash"] = "wrong"
            pd.DataFrame([fw_row]).to_csv(fw_path, index=False)
            mismatch = build_fw_lnsa_comparison(
                baseline_results,
                fw_lnsa_results_path=fw_path,
                fpr_limit=0.10,
            )
            self.assertEqual(set(mismatch["approach"]), {"Baseline"})
            self.assertTrue((mismatch["comparison_status"] == "no_compatible_fw_lnsa_rows").all())

    def test_baseline_configuration_profiles_are_validated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline_models.yaml"
        loaded = load_yaml_config(config_path)
        smoke = resolve_baseline_profile(loaded, "smoke")
        full = resolve_baseline_profile(loaded, "full")
        self.assertEqual(smoke["feature_selection"]["feature_sizes"], [20])
        self.assertEqual(full["experiment"]["seeds"], [42, 43, 44, 45, 46])
        self.assertTrue(smoke["models"]["isolation_forest"]["enabled"])

        loaded["profiles"]["smoke"]["feature_selection"] = {
            "feature_sizes": [],
            "main_feature_size": 20,
        }
        with self.assertRaises(ConfigurationError):
            resolve_baseline_profile(loaded, "smoke")

    def test_command_line_runs_small_nsl_kdd_and_cicids2017(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        normal_row = [0, "tcp", "http", "SF"] + [0] * 37 + ["normal", 20]
        attack_row = [1, "udp", "private", "REJ"] + [1] * 37 + ["neptune", 18]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            train_file = root / "KDDTrain+.txt"
            test_file = root / "KDDTest+.txt"
            pd.DataFrame([normal_row, attack_row] * 12, columns=NSL_KDD_COLUMNS).to_csv(
                train_file, header=False, index=False
            )
            pd.DataFrame([normal_row, attack_row] * 4, columns=NSL_KDD_COLUMNS).to_csv(
                test_file, header=False, index=False
            )
            cic_archive = self._write_cic_archive(root)

            baseline_config = {
                "active_profile": "smoke",
                "datasets": ["nsl_kdd", "cicids2017"],
                "dataset_configs": {
                    "nsl_kdd": str(repo_root / "configs" / "nsl_kdd_fw_lnsa.yaml"),
                    "cicids2017": str(repo_root / "configs" / "cicids2017_fw_lnsa.yaml"),
                },
                "dataset_profiles": {"nsl_kdd": "smoke", "cicids2017": "smoke"},
                "feature_selection": {
                    "method": "mutual_information",
                    "feature_sizes": [3],
                    "main_feature_size": 3,
                    "random_seed": 42,
                    "max_samples": None,
                },
                "experiment": {"seeds": [42]},
                "models": {
                    "logistic_regression": {
                        "enabled": True,
                        "parameters": {"solver": "liblinear", "max_iter": 200},
                    },
                    "decision_tree": {"enabled": False, "parameters": {}},
                    "random_forest": {"enabled": False, "parameters": {}},
                    "isolation_forest": {"enabled": False, "parameters": {}},
                },
                "selection": {"balanced_fpr_limit": 0.10},
                "outputs": {
                    "nsl_kdd": {
                        "baseline_results": "nsl.csv",
                        "comparison_table": "nsl_compare.csv",
                        "attack_category_analysis": "nsl_categories.csv",
                    },
                    "cicids2017": {
                        "baseline_results": "cic.csv",
                        "comparison_table": "cic_compare.csv",
                        "attack_category_analysis": "cic_categories.csv",
                    },
                },
                "profiles": {"smoke": {}},
            }
            config_path = root / "baseline.yaml"
            config_path.write_text(yaml.safe_dump(baseline_config, sort_keys=False))

            nsl_command = [
                sys.executable,
                "scripts/run_baselines.py",
                "--config",
                str(config_path),
                "--profile",
                "smoke",
                "--dataset",
                "nsl_kdd",
                "--nsl-train-file",
                str(train_file),
                "--nsl-test-file",
                str(test_file),
                "--output-dir",
                str(root / "nsl_results"),
            ]
            nsl = subprocess.run(
                nsl_command, cwd=repo_root, capture_output=True, text=True, check=False
            )
            self.assertEqual(nsl.returncode, 0, nsl.stderr)
            self.assertIn("Baseline run completed", nsl.stdout)

            cic_command = [
                sys.executable,
                "scripts/run_baselines.py",
                "--config",
                str(config_path),
                "--profile",
                "smoke",
                "--dataset",
                "cicids2017",
                "--cic-raw-dir",
                str(root),
                "--cic-archive-file",
                str(cic_archive),
                "--output-dir",
                str(root / "cic_results"),
            ]
            cic = subprocess.run(
                cic_command, cwd=repo_root, capture_output=True, text=True, check=False
            )
            self.assertEqual(cic.returncode, 0, cic.stderr)
            self.assertIn("Baseline run completed", cic.stdout)


if __name__ == "__main__":
    unittest.main()
