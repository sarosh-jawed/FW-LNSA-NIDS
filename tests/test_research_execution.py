from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.baselines import run_prepared_baseline_grid
from src.config import load_yaml_config, resolve_baseline_profile, resolve_profile
from src.experiments import run_prepared_fw_lnsa_experiments
from src.preprocessing import PreparedDataset
from src.research_execution import (
    _prepare_dataset_cached,
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    build_execution_plan,
    load_jsonl_frame,
    load_research_execution_config,
)
from src.statistical_analysis import (
    aggregate_fw_results,
    analyze_research_results,
    feature_set_ablation,
    hamming_weight_ablation,
    mean_std_ci,
    select_fpr_controlled_configs,
)


class ResearchExecutionTests(unittest.TestCase):
    def _dataset(self) -> PreparedDataset:
        rng = np.random.default_rng(42)
        normal_train = rng.normal(0.15, 0.03, size=(30, 6))
        attack_train = rng.normal(0.85, 0.03, size=(30, 6))
        normal_test = rng.normal(0.15, 0.03, size=(12, 6))
        attack_test = rng.normal(0.85, 0.03, size=(12, 6))
        X_train = pd.DataFrame(
            np.vstack([normal_train, attack_train]),
            columns=[f"f{i}" for i in range(6)],
        )
        X_test = pd.DataFrame(
            np.vstack([normal_test, attack_test]),
            columns=X_train.columns,
        )
        y_train = np.array([0] * 30 + [1] * 30, dtype=np.int8)
        y_test = np.array([0] * 12 + [1] * 12, dtype=np.int8)
        return PreparedDataset(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_original_labels=pd.Series(["normal"] * 30 + ["attack"] * 30),
            test_original_labels=pd.Series(["normal"] * 12 + ["attack"] * 12),
            train_attack_categories=pd.Series(["Normal"] * 30 + ["DoS"] * 30),
            test_attack_categories=pd.Series(["Normal"] * 12 + ["DoS"] * 12),
            scaler=MinMaxScaler().fit(X_train),
            metadata={"dataset": "NSL-KDD"},
        )

    def _fw_config(self, output_dir: str) -> dict:
        return {
            "profile_name": "research",
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
                "seeds": [42, 43],
                "deduplicate_candidates": True,
                "prediction_chunk_size": 20,
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
                "results_table": "fw.csv",
                "method_summary": "summary.csv",
                "balanced_configs": "balanced.csv",
                "attack_category_analysis": "categories.csv",
                "selected_features": "features.csv",
            },
        }

    def _baseline_config(self) -> dict:
        return {
            "profile_name": "research",
            "feature_selection": {
                "method": "mutual_information",
                "feature_sizes": [4],
                "main_feature_size": 4,
                "random_seed": 42,
                "max_samples": None,
            },
            "experiment": {"seeds": [42, 43]},
            "models": {
                "decision_tree": {
                    "enabled": True,
                    "parameters": {"max_depth": 3},
                }
            },
            "selection": {"balanced_fpr_limit": 0.10},
        }

    def test_fw_grid_resumes_without_repeating_completed_rows(self) -> None:
        dataset = self._dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._fw_config(tmp_dir)
            first = run_prepared_fw_lnsa_experiments(
                dataset,
                config,
                max_runs=1,
                save_outputs=False,
            )
            new_rows: list[dict] = []
            resumed = run_prepared_fw_lnsa_experiments(
                dataset,
                config,
                save_outputs=False,
                existing_results=first.results,
                result_callback=new_rows.append,
            )
            self.assertEqual(len(first.results), 1)
            self.assertEqual(len(resumed.results), 2)
            self.assertEqual(len(new_rows), 1)
            self.assertEqual(set(resumed.results["seed"]), {42, 43})

    def test_detection_thresholds_reuse_one_detector_pool_and_score_pass(self) -> None:
        dataset = self._dataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = self._fw_config(tmp_dir)
            config["experiment"]["seeds"] = [42]
            config["method_settings"]["weighted_smc"]["detection_thresholds"] = [0.75, 0.70]
            outputs = run_prepared_fw_lnsa_experiments(
                dataset,
                config,
                save_outputs=False,
            )
            self.assertEqual(len(outputs.results), 2)
            self.assertEqual(outputs.results["detector_pool_key"].nunique(), 1)
            self.assertTrue(outputs.results["shared_score_evaluation"].all())
            self.assertEqual(outputs.results["generation_time_sec"].nunique(), 1)
            self.assertEqual(outputs.results["detection_time_sec"].nunique(), 1)

    def test_baseline_grid_resumes_and_preserves_category_rows(self) -> None:
        dataset = self._dataset()
        config = self._baseline_config()
        first_results, first_categories, _ = run_prepared_baseline_grid(
            dataset,
            config,
            dataset_key="nsl_kdd",
            max_runs=1,
        )
        callbacks: list[tuple[dict, pd.DataFrame]] = []
        resumed_results, resumed_categories, _ = run_prepared_baseline_grid(
            dataset,
            config,
            dataset_key="nsl_kdd",
            existing_results=first_results,
            existing_category_analysis=first_categories,
            result_callback=lambda row, categories: callbacks.append((row, categories)),
        )
        self.assertEqual(len(resumed_results), 2)
        self.assertEqual(len(callbacks), 1)
        self.assertGreater(len(resumed_categories), len(first_categories))
        self.assertEqual(set(resumed_results["seed"]), {42, 43})

    def test_statistical_summaries_use_seed_repeats(self) -> None:
        rows = []
        for method, offset in (("hamming", 0.0), ("weighted_hamming", 0.05)):
            for fs_size in (10, 20):
                for seed in (42, 43, 44):
                    rows.append(
                        {
                            "dataset": "NSL-KDD",
                            "profile": "research",
                            "method": method,
                            "method_name": method,
                            "method_role": "test",
                            "feature_set": f"FS-{fs_size}",
                            "fs_size": fs_size,
                            "detector_budget": 500,
                            "self_threshold_config": 0.1,
                            "detection_threshold_config": 0.2,
                            "threshold_scale": "direct",
                            "selected_features": "x",
                            "feature_selection_hash": f"h{fs_size}",
                            "train_records": 100,
                            "test_records": 50,
                            "train_partition_hash": "train",
                            "test_partition_hash": "test",
                            "seed": seed,
                            "accuracy": 0.7 + offset,
                            "precision": 0.7 + offset,
                            "recall": 0.7 + offset,
                            "f1": 0.7 + offset,
                            "fpr": 0.08,
                            "fnr": 0.3 - offset,
                            "total_time_sec": 1.0,
                            "retained_detectors": 100,
                        }
                    )
        frame = pd.DataFrame(rows)
        summary = aggregate_fw_results(frame)
        self.assertEqual(len(summary), 4)
        self.assertTrue((summary["seed_count"] == 3).all())
        selected = select_fpr_controlled_configs(summary, fpr_limit=0.10)
        self.assertEqual(len(selected), 4)
        ablation = hamming_weight_ablation(frame)
        self.assertEqual(len(ablation), 6)
        self.assertTrue(np.allclose(ablation["delta_f1"], 0.05))
        fs_ablation = feature_set_ablation(summary)
        self.assertEqual(len(fs_ablation), 2)
        ci = mean_std_ci([1.0, 2.0, 3.0])
        self.assertLess(ci["ci_low"], ci["mean"])
        self.assertGreater(ci["ci_high"], ci["mean"])

    def test_analysis_outputs_are_saved(self) -> None:
        fw_rows = []
        baseline_rows = []
        for dataset in ("NSL-KDD", "CICIDS2017"):
            for method in ("hamming", "weighted_hamming", "weighted_smc"):
                for fs_size in (10, 20):
                    for seed in (42, 43):
                        fw_rows.append(
                            {
                                "dataset": dataset,
                                "profile": "research",
                                "method": method,
                                "method_name": method,
                                "method_role": "test",
                                "feature_set": f"FS-{fs_size}",
                                "fs_size": fs_size,
                                "detector_budget": 500,
                                "self_threshold_config": 0.1,
                                "detection_threshold_config": 0.2,
                                "threshold_scale": "direct",
                                "selected_features": "x",
                                "feature_selection_hash": f"h{fs_size}",
                                "train_records": 100,
                                "test_records": 50,
                                "train_partition_hash": "train",
                                "test_partition_hash": "test",
                                "seed": seed,
                                "accuracy": 0.8,
                                "precision": 0.8,
                                "recall": 0.8,
                                "f1": 0.8,
                                "fpr": 0.05,
                                "fnr": 0.2,
                                "total_time_sec": 1.0,
                                "retained_detectors": 100,
                            }
                        )
            for model in (
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "isolation_forest",
            ):
                for fs_size in (10, 20):
                    for seed in (42, 43):
                        baseline_rows.append(
                            {
                                "dataset_key": dataset.lower().replace("-", "_"),
                                "dataset": dataset,
                                "profile": "research",
                                "model": model,
                                "model_name": model,
                                "model_family": "test",
                                "feature_set": f"FS-{fs_size}",
                                "fs_size": fs_size,
                                "model_parameters_json": "{}",
                                "selected_features": "x",
                                "feature_selection_hash": f"h{fs_size}",
                                "train_records": 100,
                                "test_records": 50,
                                "train_partition_hash": "train",
                                "test_partition_hash": "test",
                                "seed": seed,
                                "accuracy": 0.85,
                                "precision": 0.85,
                                "recall": 0.85,
                                "f1": 0.85,
                                "fpr": 0.04,
                                "fnr": 0.15,
                                "total_time_sec": 0.5,
                            }
                        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = analyze_research_results(
                fw_results=pd.DataFrame(fw_rows),
                baseline_results=pd.DataFrame(baseline_rows),
                output_dir=tmp_dir,
                expected={
                    "datasets": ["NSL-KDD", "CICIDS2017"],
                    "methods": ["hamming", "weighted_hamming", "weighted_smc"],
                    "feature_sizes": [10, 20],
                    "detector_budgets": [500],
                    "seeds": [42, 43],
                    "baseline_models": [
                        "logistic_regression",
                        "decision_tree",
                        "random_forest",
                        "isolation_forest",
                    ],
                },
            )
            self.assertIn("readiness_report", outputs.tables)
            self.assertTrue(outputs.tables["readiness_report"]["passed"].all())
            for path in outputs.output_paths.values():
                self.assertTrue(path.exists())

    def test_repository_research_profiles_match_final_plan(self) -> None:
        root = Path(__file__).resolve().parents[1]
        nsl = resolve_profile(load_yaml_config(root / "configs/nsl_kdd_fw_lnsa.yaml"), "research")
        cic = resolve_profile(load_yaml_config(root / "configs/cicids2017_fw_lnsa.yaml"), "research")
        baselines = resolve_baseline_profile(
            load_yaml_config(root / "configs/baseline_models.yaml"), "research"
        )
        execution = load_research_execution_config(root / "configs/research_execution.yaml")
        for config in (nsl, cic):
            self.assertEqual(config["feature_selection"]["feature_sizes"], [10, 20])
            self.assertEqual(config["experiment"]["seeds"], [42, 43, 44, 45, 46])
            self.assertEqual(
                config["experiment"]["methods"],
                ["hamming", "weighted_hamming", "weighted_smc"],
            )
        self.assertEqual(baselines["experiment"]["seeds"], [42, 43, 44, 45, 46])
        self.assertIn("analysis", execution["execution"]["stages"])

        plan = build_execution_plan(
            execution_config=execution,
            profile="research",
            repo_root=root,
        ).set_index("stage")
        self.assertEqual(int(plan.loc["nsl_kdd_fw_lnsa", "expected_runs"]), 280)
        self.assertEqual(int(plan.loc["cicids2017_fw_lnsa", "expected_runs"]), 240)
        self.assertEqual(int(plan.loc["nsl_kdd_baselines", "expected_runs"]), 40)
        self.assertEqual(int(plan.loc["cicids2017_baselines", "expected_runs"]), 40)

    def test_atomic_writes_replace_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "rows.csv"
            json_path = Path(tmp_dir) / "state.json"
            atomic_write_csv(pd.DataFrame({"x": [1]}), csv_path)
            atomic_write_csv(pd.DataFrame({"x": [2]}), csv_path)
            atomic_write_json({"status": "running"}, json_path)
            atomic_write_json({"status": "complete"}, json_path)
            self.assertEqual(pd.read_csv(csv_path).iloc[0]["x"], 2)
            self.assertIn("complete", json_path.read_text())

    def test_progress_journal_resumes_and_ignores_partial_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            journal = Path(tmp_dir) / "rows.jsonl"
            append_jsonl({"seed": 42, "f1": 0.8}, journal)
            append_jsonl({"seed": 43, "f1": 0.9}, journal)
            with journal.open("a", encoding="utf-8") as handle:
                handle.write('{"seed": 44')
            loaded = load_jsonl_frame(journal)
            self.assertEqual(loaded["seed"].tolist(), [42, 43])
            self.assertEqual(loaded["f1"].tolist(), [0.8, 0.9])

    def test_prepared_dataset_cache_is_reused(self) -> None:
        config = {
            "dataset": {"name": "nsl_kdd"},
            "paths": {"train_file": "train", "test_file": "test"},
            "preprocessing": {"scale": True},
        }
        execution = {"execution": {"cache_dir": "cache"}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            prepared_source = self._dataset()
            with patch(
                "src.research_execution._prepare_dataset",
                return_value=prepared_source,
            ) as prepare:
                first, first_path, first_hit = _prepare_dataset_cached(
                    dataset_name="nsl_kdd",
                    config=config,
                    output_dir=Path(tmp_dir),
                    execution_config=execution,
                    dataset_hashes={"train": "a", "test": "b"},
                )
                second, second_path, second_hit = _prepare_dataset_cached(
                    dataset_name="nsl_kdd",
                    config=config,
                    output_dir=Path(tmp_dir),
                    execution_config=execution,
                    dataset_hashes={"train": "a", "test": "b"},
                )
            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(first_path, second_path)
            self.assertEqual(len(first.X_train), len(second.X_train))
            pd.testing.assert_frame_equal(first.X_train, second.X_train)
            prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
