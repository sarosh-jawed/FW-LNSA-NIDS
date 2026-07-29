"""Validation tests for the final FW-LNSA confirmatory protocol."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.confirmatory_protocol import (
    calibrate_detection_threshold,
    expected_confirmatory_plan,
    load_confirmatory_config,
    prepare_confirmatory_dataset,
    prepare_confirmatory_feature_sets,
    run_confirmatory_dataset,
    run_tuning_stage,
)
from src.matching import (
    weighted_binary_similarity_matrix,
    weighted_hamming_distance_matrix,
)
from src.preprocessing import (
    map_nsl_kdd_attack_category,
    prepare_nsl_kdd,
    read_nsl_kdd_file,
)


class ConfirmatoryProtocolTests(unittest.TestCase):
    def _write_nsl_files(self, directory: Path) -> tuple[Path, Path]:
        rows = []
        labels = ["normal", "neptune", "portsweep", "guess_passwd", "buffer_overflow"]
        for index in range(100):
            label = labels[index % len(labels)]
            base = 0 if label == "normal" else 1
            row = [index % 3, "tcp" if index % 2 == 0 else "udp", "http", "SF"]
            row += [base + (index % 7)] * 37
            row += [label, 20]
            rows.append(row)
        test_rows = []
        for index in range(40):
            label = "Normal" if index % 2 == 0 else "Attack"
            base = 0 if label == "Normal" else 1
            row = [index % 2, "tcp", "http", "SF"] + [base + (index % 5)] * 37 + [label]
            test_rows.append(row)
        train = directory / "KDDTrain+.txt"
        test = directory / "KDDTest+.txt"
        pd.DataFrame(rows).to_csv(train, header=False, index=False)
        pd.DataFrame(test_rows).to_csv(test, header=False, index=False, sep="\t")
        return train, test

    def _write_cic_archive(self, directory: Path) -> Path:
        archive_path = directory / "MachineLearningCSV.zip"
        columns = [" Flow Duration ", " Flow Bytes/s", "Packet Count", "Flag Value", " Label"]
        labels = ["BENIGN", "DDoS", "PortScan", "Bot", "FTP-Patator"]
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for day in range(2):
                rows = []
                for index in range(100):
                    label = labels[index % len(labels)]
                    base = 0.1 if label == "BENIGN" else 0.8
                    rows.append([base + index * 0.001, base * 10, index % 11, index % 2, label])
                archive.writestr(
                    f"MachineLearningCVE/day_{day}.csv",
                    pd.DataFrame(rows, columns=columns).to_csv(index=False),
                )
        return archive_path

    def _config(self, root: Path, train: Path, test: Path, archive: Path) -> dict:
        config = load_confirmatory_config(
            Path(__file__).resolve().parents[1] / "configs" / "confirmatory_protocol.yaml",
            profile_name="smoke",
        )
        config["paths"]["output_dir"] = str(root / "outputs")
        config["datasets"]["nsl_kdd"]["train_file"] = str(train)
        config["datasets"]["nsl_kdd"]["test_file"] = str(test)
        config["datasets"]["cicids2017"]["raw_dir"] = str(root)
        config["datasets"]["cicids2017"]["archive_file"] = str(archive)
        config["datasets"]["cicids2017"]["max_chunks_per_file"] = None
        return config

    def test_weighted_similarity_is_complement_of_weighted_hamming(self) -> None:
        records = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.int8)
        detectors = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int8)
        weights = np.array([0.2, 0.3, 0.5])
        distance = weighted_hamming_distance_matrix(records, detectors, weights)
        similarity = weighted_binary_similarity_matrix(records, detectors, weights)
        np.testing.assert_allclose(similarity, 1.0 - distance, atol=1e-12)

    def test_calibration_meets_discrete_fpr_targets(self) -> None:
        y = np.array([0, 0, 0, 0, 1, 1], dtype=np.int8)
        distance_scores = np.array([0, 1, 2, 3, 0, 1], dtype=float)
        result = calibrate_detection_threshold(
            distance_scores,
            y,
            method="hamming",
            target_fpr=0.25,
        )
        self.assertLessEqual(result.achieved_fpr, 0.25)
        self.assertEqual(result.false_alarms, 1)

        similarity_scores = 1.0 - distance_scores / 3.0
        result_similarity = calibrate_detection_threshold(
            similarity_scores,
            y,
            method="weighted_similarity",
            target_fpr=0.25,
        )
        self.assertLessEqual(result_similarity.achieved_fpr, 0.25)
        self.assertEqual(result_similarity.false_alarms, 1)

    def test_nsl_schema_and_category_integrity(self) -> None:
        self.assertEqual(map_nsl_kdd_attack_category("neptune."), "DoS")
        self.assertEqual(map_nsl_kdd_attack_category("portsweep"), "Probe")
        self.assertEqual(map_nsl_kdd_attack_category("guess_passwd"), "R2L")
        self.assertEqual(map_nsl_kdd_attack_category("rootkit"), "U2R")
        self.assertEqual(map_nsl_kdd_attack_category("Attack"), "Attack (Unspecified)")

        with tempfile.TemporaryDirectory() as tmp:
            train, test = self._write_nsl_files(Path(tmp))
            test_frame = read_nsl_kdd_file(test)
            self.assertEqual(test_frame.shape[1], 43)
            self.assertTrue(test_frame["difficulty"].isna().all())
            dataset = prepare_nsl_kdd(
                train,
                test,
                validation_size=0.20,
                validation_seed=2026,
            )
            self.assertTrue(dataset.has_validation)
            self.assertEqual(dataset.metadata["test_attack_category_resolution"], "binary_only")
            self.assertGreater(len(dataset.X_train), len(dataset.X_validation))
            self.assertEqual(dataset.X_train.shape[1], dataset.X_test.shape[1])
            self.assertEqual(dataset.X_train.shape[1], dataset.X_validation.shape[1])

    def test_profiles_match_final_confirmatory_design(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "confirmatory_protocol.yaml"
        final = load_confirmatory_config(config_path, profile_name="confirmatory")
        self.assertEqual(final["experiment"]["detector_budgets"], [500, 1000, 2500, 5000])
        self.assertEqual(final["experiment"]["feature_sizes"], [10, 20])
        self.assertEqual(final["experiment"]["tuning_seeds"], [42, 43, 44, 45, 46])
        self.assertEqual(final["experiment"]["confirmatory_seeds"], list(range(100, 120)))
        self.assertEqual(final["experiment"]["target_fprs"], [0.10, 0.05, 0.01])
        plan = expected_confirmatory_plan(final)
        self.assertEqual(plan["locked_configurations_per_dataset"], 12)
        self.assertEqual(plan["confirmatory_rows_per_dataset"], 240)

    def test_tuning_is_independent_of_test_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = self._write_nsl_files(root)
            archive = self._write_cic_archive(root)
            config = self._config(root, train, test, archive)
            dataset = prepare_confirmatory_dataset("nsl_kdd", config)
            feature_sets = prepare_confirmatory_feature_sets(dataset, config)
            first = run_tuning_stage("nsl_kdd", dataset, config, feature_sets)

            changed = dataset.__class__(
                **{
                    **dataset.__dict__,
                    "y_test": 1 - dataset.y_test,
                }
            )
            second = run_tuning_stage("nsl_kdd", changed, config, feature_sets)
            columns = [
                "method",
                "fs_size",
                "seed",
                "detector_budget",
                "self_threshold_config",
                "target_fpr",
                "validation_f1",
                "validation_fpr",
                "detection_threshold",
            ]
            pd.testing.assert_frame_equal(first[columns], second[columns])
            self.assertTrue((first["test_partition_accessed"] == False).all())

    def test_end_to_end_smoke_is_resumable_and_writes_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = self._write_nsl_files(root)
            archive = self._write_cic_archive(root)
            config = self._config(root, train, test, archive)

            outputs = run_confirmatory_dataset("nsl_kdd", config, stage="all")
            self.assertEqual(len(outputs.tuning_results), 2)
            self.assertEqual(len(outputs.locked_configurations), 2)
            self.assertEqual(len(outputs.final_seed_results), 2)
            self.assertTrue((outputs.final_seed_results["test_metrics_used_for_selection"] == False).all())
            lock_path = outputs.output_paths["configuration_lock"]
            self.assertTrue(lock_path.exists())
            with lock_path.open("r", encoding="utf-8") as handle:
                lock = json.load(handle)
            self.assertFalse(lock["calibration"]["test_metrics_used_for_selection"])

            resumed = run_confirmatory_dataset("nsl_kdd", config, stage="all")
            self.assertEqual(len(resumed.tuning_results), 2)
            self.assertEqual(len(resumed.final_seed_results), 2)
            self.assertEqual(
                resumed.manifest["partition_hashes"],
                outputs.manifest["partition_hashes"],
            )

    def test_resume_rejects_changed_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = self._write_nsl_files(root)
            archive = self._write_cic_archive(root)
            config = self._config(root, train, test, archive)
            outputs = run_confirmatory_dataset("nsl_kdd", config, stage="all")
            self.assertTrue(outputs.output_paths["execution_state"].exists())

            changed = json.loads(json.dumps(config))
            changed["protocol"]["max_self_samples"] = 999
            with self.assertRaisesRegex(RuntimeError, "different execution identity"):
                run_confirmatory_dataset("nsl_kdd", changed, stage="all")

    def test_cicids2017_three_way_preparation_uses_training_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = self._write_nsl_files(root)
            archive = self._write_cic_archive(root)
            config = self._config(root, train, test, archive)
            dataset = prepare_confirmatory_dataset("cicids2017", config)
            self.assertTrue(dataset.has_validation)
            self.assertGreater(len(dataset.X_train), 0)
            self.assertGreater(len(dataset.X_validation), 0)
            self.assertGreater(len(dataset.X_test), 0)
            self.assertEqual(dataset.X_train.shape[1], dataset.X_validation.shape[1])
            self.assertEqual(dataset.X_train.shape[1], dataset.X_test.shape[1])
            self.assertIn("validation_records", dataset.metadata)
            self.assertIsNotNone(dataset.data_quality_report)


if __name__ == "__main__":
    unittest.main()
