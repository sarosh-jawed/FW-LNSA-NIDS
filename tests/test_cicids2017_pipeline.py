"""Validation tests for CICIDS2017 cleaning and FW-LNSA experiments."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ConfigurationError, load_yaml_config, resolve_profile
from src.experiments import run_cicids2017_experiments
from src.preprocessing import (
    clean_cicids2017_dataframe,
    discover_cicids2017_sources,
    map_cicids2017_attack_category,
    prepare_cicids2017,
)


class CICIDS2017PipelineTests(unittest.TestCase):
    def _write_archive(self, directory: Path) -> Path:
        archive_path = directory / "MachineLearningCSV.zip"
        columns = [" Flow Duration ", " Flow Bytes/s", "Packet Count", "Flag Value", " Label"]

        rows_a = []
        rows_b = []
        labels = ["BENIGN", "DDoS", "PortScan", "Bot", "FTP-Patator", "Web Attack Brute Force"]
        for index in range(72):
            label = labels[index % len(labels)]
            base = 0.1 if label == "BENIGN" else 0.8
            row = [base + index * 0.001, base * 10, index % 11, index % 2, label]
            (rows_a if index < 36 else rows_b).append(row)

        rows_a.append(rows_a[0].copy())
        rows_a.append([1.0, "Infinity", 2, 1, "DDoS"])
        rows_b.append([2.0, "not-a-number", 3, 0, "PortScan"])
        rows_b.append([3.0, 2.0, 4, 1, None])

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, rows in (("MachineLearningCVE/day_a.csv", rows_a), ("MachineLearningCVE/day_b.csv", rows_b)):
                csv_text = pd.DataFrame(rows, columns=columns).to_csv(index=False)
                archive.writestr(name, csv_text)
        return archive_path

    def _config(self, raw_dir: Path, archive_path: Path, output_dir: Path) -> dict:
        return {
            "profile_name": "test",
            "dataset": {"name": "cicids2017", "task": "binary"},
            "paths": {
                "raw_dir": str(raw_dir),
                "archive_file": str(archive_path),
                "output_dir": str(output_dir),
            },
            "preprocessing": {
                "label_column": "Label",
                "benign_label": "BENIGN",
                "chunk_size": 20,
                "max_chunks_per_file": None,
                "drop_duplicates": True,
                "scale": True,
                "max_self_samples": 20,
                "sampling": {
                    "max_benign_records": 20,
                    "max_records_per_attack_label": 12,
                    "random_seed": 42,
                },
                "split": {
                    "strategy": "stratified_random",
                    "stratify_by": "original_label",
                    "test_size": 0.25,
                    "random_seed": 42,
                },
            },
            "feature_selection": {
                "method": "mutual_information",
                "feature_sizes": [4],
                "main_feature_size": 4,
                "random_seed": 42,
                "max_samples": None,
            },
            "representation": {"method": "binary_median_split"},
            "experiment": {
                "methods": ["hamming", "weighted_hamming", "weighted_smc"],
                "detector_budgets": [24],
                "seeds": [42],
                "deduplicate_candidates": True,
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
            "selection": {"balanced_fpr_limit": 0.20},
            "outputs": {
                "results_table": "cicids2017_fw_lnsa_results.csv",
                "method_summary": "cicids2017_matching_method_summary.csv",
                "balanced_configs": "cicids2017_best_balanced_configs.csv",
                "attack_category_analysis": "cicids2017_attack_category_analysis.csv",
                "selected_features": "cicids2017_selected_features.csv",
                "data_quality_report": "cicids2017_data_quality_report.csv",
            },
        }

    def test_cleaning_handles_whitespace_nonfinite_invalid_and_duplicates(self) -> None:
        raw = pd.DataFrame(
            {
                " Feature A ": [1.0, 1.0, np.inf, "bad", 5.0],
                "Feature B": [2.0, 2.0, 3.0, 4.0, 5.0],
                " Label ": ["BENIGN", "BENIGN", "DDoS", "PortScan", None],
            }
        )
        X, y, labels = clean_cicids2017_dataframe(raw)
        self.assertEqual(len(X), 3)
        self.assertEqual(y.tolist(), [0, 1, 1])
        self.assertEqual(labels.tolist(), ["BENIGN", "DDoS", "PortScan"])
        self.assertTrue(np.isfinite(X.to_numpy()).all())
        self.assertNotIn("Label", X.columns)

    def test_archive_discovery_preparation_and_leakage_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            archive_path = self._write_archive(root)
            sources = discover_cicids2017_sources(root, archive_file=archive_path)
            self.assertEqual(len(sources), 2)

            prepared = prepare_cicids2017(
                root,
                archive_file=archive_path,
                chunk_size=20,
                max_chunks_per_file=None,
                max_benign_records=20,
                max_records_per_attack_label=12,
                test_size=0.25,
            )
            self.assertGreater(len(prepared.X_train), 0)
            self.assertGreater(len(prepared.X_test), 0)
            self.assertEqual(list(prepared.X_train.columns), list(prepared.X_test.columns))
            self.assertTrue(np.isfinite(prepared.X_train.to_numpy()).all())
            self.assertTrue(np.isfinite(prepared.X_test.to_numpy()).all())
            self.assertNotIn("Label", prepared.X_train.columns)
            self.assertIsNotNone(prepared.data_quality_report)
            summary = prepared.data_quality_report.query("record_type == 'summary'").iloc[0]
            self.assertGreaterEqual(int(summary["duplicate_rows_removed"]), 1)
            self.assertGreaterEqual(int(summary["nonfinite_values_replaced"]), 1)
            self.assertGreaterEqual(int(summary["nonnumeric_values_coerced"]), 1)

    def test_global_cap_sampling_preserves_a_bounded_natural_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            archive_path = self._write_archive(root)
            prepared = prepare_cicids2017(
                root,
                archive_file=archive_path,
                chunk_size=20,
                sampling_strategy="global_cap",
                max_total_records=30,
                max_benign_records=1,
                max_records_per_attack_label=1,
                test_size=0.30,
                stratify_by="binary",
            )
            self.assertEqual(
                int(prepared.metadata["sampled_records"]),
                30,
            )
            self.assertEqual(prepared.metadata["sampling_strategy"], "global_cap")
            self.assertEqual(len(prepared.X_train) + len(prepared.X_test), 30)

    def test_attack_category_mapping(self) -> None:
        self.assertEqual(map_cicids2017_attack_category("BENIGN"), "Normal")
        self.assertEqual(map_cicids2017_attack_category("DoS Hulk"), "DoS/DDoS")
        self.assertEqual(map_cicids2017_attack_category("FTP-Patator"), "Brute Force")
        self.assertEqual(map_cicids2017_attack_category("Web Attack XSS"), "Web Attack")

    def test_configuration_profiles_are_validated(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "cicids2017_fw_lnsa.yaml"
        loaded = load_yaml_config(config_path)
        smoke = resolve_profile(loaded, "smoke")
        full = resolve_profile(loaded, "full")
        self.assertEqual(smoke["preprocessing"]["max_chunks_per_file"], 3)
        self.assertEqual(full["feature_selection"]["main_feature_size"], 20)
        self.assertIn("weighted_smc", full["experiment"]["methods"])

        loaded["profiles"]["smoke"]["preprocessing"]["split"] = {"test_size": 2.0}
        with self.assertRaises(ConfigurationError):
            resolve_profile(loaded, "smoke")

    def test_experiment_pipeline_saves_required_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            archive_path = self._write_archive(root)
            output_dir = root / "results"
            outputs = run_cicids2017_experiments(
                self._config(root, archive_path, output_dir),
            )
            self.assertEqual(len(outputs.results), 3)
            self.assertEqual(
                set(outputs.results["method"]),
                {"hamming", "weighted_hamming", "weighted_smc"},
            )
            self.assertFalse(outputs.data_quality_report.empty)
            self.assertEqual(len(outputs.output_paths), 6)
            for path in outputs.output_paths.values():
                self.assertTrue(path.exists(), path)

    def test_command_line_dry_run_with_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            archive_path = self._write_archive(root)
            repo_root = Path(__file__).resolve().parents[1]
            command = [
                sys.executable,
                "scripts/run_cicids2017_fw_lnsa.py",
                "--config",
                "configs/cicids2017_fw_lnsa.yaml",
                "--profile",
                "smoke",
                "--raw-dir",
                str(root),
                "--archive-file",
                str(archive_path),
                "--output-dir",
                str(root / "results"),
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
            self.assertIn("Discovered CSV sources: 2", completed.stdout)
            self.assertIn("data sources are valid", completed.stdout)


if __name__ == "__main__":
    unittest.main()
