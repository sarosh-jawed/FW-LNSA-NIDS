"""Baseline models and fair comparison utilities for FW-LNSA research.

All models use the same prepared train and test partitions as FW-LNSA. Feature
selection is fitted on training data only. Isolation Forest is trained only on
normal training traffic so it remains a genuine anomaly-detection baseline.
"""

from __future__ import annotations

import json
import math
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .evaluation import attack_category_analysis, evaluate_binary_classification
from .experiments import _feature_selection_sample
from .feature_selection import FeatureSelectionResult, mutual_information_feature_selection
from .preprocessing import PreparedDataset
from .utils import (
    ensure_dir,
    save_dataframe,
    stable_feature_selection_signature,
    stable_partition_signature,
)

BASELINE_MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "isolation_forest": "Isolation Forest",
}

BASELINE_MODEL_FAMILIES = {
    "logistic_regression": "supervised_linear",
    "decision_tree": "supervised_tree",
    "random_forest": "supervised_ensemble",
    "isolation_forest": "unsupervised_anomaly",
}

ProgressCallback = Callable[[int, int, dict[str, Any]], None]


@dataclass(frozen=True)
class BaselineFeatureSet:
    """Selected continuous features used by one baseline feature setting."""

    fs_size: int
    selection: FeatureSelectionResult
    X_train: pd.DataFrame
    X_test: pd.DataFrame


@dataclass(frozen=True)
class BaselineOutputs:
    """In-memory baseline tables and their saved file paths."""

    results: pd.DataFrame
    comparison: pd.DataFrame
    attack_category_analysis: pd.DataFrame
    selected_features: pd.DataFrame
    output_paths: dict[str, Path]


class _PeakMemoryMonitor:
    """Sample process memory while a model is fitting and predicting."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss = 0
        self.peak_rss = 0

    def _sample(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)
            except psutil.Error:
                return
            self._stop_event.wait(self.interval_seconds)

    def __enter__(self) -> "_PeakMemoryMonitor":
        self.start_rss = self._process.memory_info().rss
        self.peak_rss = self.start_rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)
        except psutil.Error:
            pass

    @property
    def peak_process_memory_mb(self) -> float:
        return self.peak_rss / (1024 ** 2)

    @property
    def peak_memory_increase_mb(self) -> float:
        return max(0.0, self.peak_rss - self.start_rss) / (1024 ** 2)


def _normalize_model_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Convert YAML values into parameters accepted by scikit-learn."""

    normalized = dict(parameters)
    for key, value in list(normalized.items()):
        if isinstance(value, str) and value.lower() == "none":
            normalized[key] = None
    return normalized


def build_baseline_model(
    model_name: str,
    parameters: Mapping[str, Any],
    *,
    seed: int,
) -> object:
    """Build one configured baseline model with a reproducible random seed."""

    params = _normalize_model_parameters(parameters)

    if model_name == "logistic_regression":
        params.setdefault("solver", "liblinear")
        params.setdefault("max_iter", 1000)
        params.setdefault("class_weight", "balanced")
        params["random_state"] = seed
        return LogisticRegression(**params)

    if model_name == "decision_tree":
        params.setdefault("class_weight", "balanced")
        params["random_state"] = seed
        return DecisionTreeClassifier(**params)

    if model_name == "random_forest":
        params.setdefault("n_estimators", 200)
        params.setdefault("class_weight", "balanced_subsample")
        params.setdefault("n_jobs", -1)
        params["random_state"] = seed
        return RandomForestClassifier(**params)

    if model_name == "isolation_forest":
        params.setdefault("n_estimators", 200)
        params.setdefault("contamination", "auto")
        params.setdefault("n_jobs", -1)
        params["random_state"] = seed
        return IsolationForest(**params)

    raise ValueError(f"Unsupported baseline model: {model_name}")


def prepare_baseline_feature_sets(
    dataset: PreparedDataset,
    feature_sizes: Sequence[int],
    *,
    random_seed: int,
    max_selection_samples: int | None,
) -> dict[int, BaselineFeatureSet]:
    """Fit mutual-information selection once and keep continuous features."""

    selection_X, selection_y = _feature_selection_sample(
        dataset,
        max_samples=max_selection_samples,
        random_seed=random_seed,
    )

    prepared: dict[int, BaselineFeatureSet] = {}
    for fs_size in feature_sizes:
        selection = mutual_information_feature_selection(
            selection_X,
            selection_y,
            fs_size=int(fs_size),
            random_seed=random_seed,
        )
        columns = list(selection.selected_features)
        prepared[int(fs_size)] = BaselineFeatureSet(
            fs_size=int(fs_size),
            selection=selection,
            X_train=dataset.X_train.loc[:, columns].copy(),
            X_test=dataset.X_test.loc[:, columns].copy(),
        )
    return prepared


def _fit_predict_one(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and predict while recording timing, memory, and model size."""

    train_mask = np.ones(len(y_train), dtype=bool)
    if model_name == "isolation_forest":
        train_mask = y_train == 0
        if not np.any(train_mask):
            raise ValueError("Isolation Forest requires normal training records.")

    with _PeakMemoryMonitor() as memory_monitor:
        fit_start = time.perf_counter()
        if model_name == "isolation_forest":
            model.fit(X_train.loc[train_mask])
        else:
            model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        raw_predictions = model.predict(X_test)
        predict_time = time.perf_counter() - predict_start

    if model_name == "isolation_forest":
        predictions = np.where(np.asarray(raw_predictions) == -1, 1, 0).astype(np.int8)
    else:
        predictions = np.asarray(raw_predictions, dtype=np.int8)

    serialized_size_kb = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024
    resources = {
        "fit_time_sec": float(fit_time),
        "prediction_time_sec": float(predict_time),
        "total_time_sec": float(fit_time + predict_time),
        "peak_process_memory_mb": float(memory_monitor.peak_process_memory_mb),
        "peak_memory_increase_mb": float(memory_monitor.peak_memory_increase_mb),
        "serialized_model_size_kb": float(serialized_size_kb),
        "training_records_available": int(len(y_train)),
        "training_records_used": int(np.sum(train_mask)),
        "normal_training_records_used": int(np.sum(y_train[train_mask] == 0)),
        "attack_training_records_used": int(np.sum(y_train[train_mask] == 1)),
    }
    return predictions, resources


def _selected_feature_table(
    prepared_sets: Mapping[int, BaselineFeatureSet],
    *,
    dataset_name: str,
    profile_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fs_size, prepared in prepared_sets.items():
        selected_scores = prepared.selection.selected_scores.reset_index(drop=True)
        for rank, selected_row in selected_scores.iterrows():
            rows.append(
                {
                    "dataset": dataset_name,
                    "profile": profile_name,
                    "feature_set": f"FS-{fs_size}",
                    "fs_size": fs_size,
                    "rank": int(rank) + 1,
                    "feature": str(selected_row["feature"]),
                    "mutual_information_score": float(selected_row["mi_score"]),
                    "normalized_weight": float(selected_row["normalized_weight"]),
                }
            )
    return pd.DataFrame(rows)


def run_prepared_baseline_grid(
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    *,
    dataset_key: str,
    max_runs: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all configured models, feature sets, and seeds on one dataset."""

    feature_config = config["feature_selection"]
    prepared_sets = prepare_baseline_feature_sets(
        dataset,
        feature_config["feature_sizes"],
        random_seed=int(feature_config["random_seed"]),
        max_selection_samples=feature_config.get("max_samples"),
    )

    enabled_models = [
        name
        for name, model_config in config["models"].items()
        if bool(model_config.get("enabled", False))
    ]
    total_runs = (
        len(enabled_models)
        * len(feature_config["feature_sizes"])
        * len(config["experiment"]["seeds"])
    )
    run_limit = total_runs if max_runs is None else min(total_runs, int(max_runs))

    train_hash = stable_partition_signature(
        dataset.X_train,
        dataset.y_train,
        dataset.train_original_labels,
    )
    test_hash = stable_partition_signature(
        dataset.X_test,
        dataset.y_test,
        dataset.test_original_labels,
    )

    rows: list[dict[str, Any]] = []
    category_tables: list[pd.DataFrame] = []
    profile_name = str(config["profile_name"])
    dataset_name = str(dataset.metadata.get("dataset", dataset_key))

    for fs_size in feature_config["feature_sizes"]:
        prepared = prepared_sets[int(fs_size)]
        for model_name in enabled_models:
            model_config = config["models"][model_name]
            for seed in config["experiment"]["seeds"]:
                if len(rows) >= run_limit:
                    selected = _selected_feature_table(
                        prepared_sets,
                        dataset_name=dataset_name,
                        profile_name=profile_name,
                    )
                    return pd.DataFrame(rows), pd.concat(category_tables, ignore_index=True), selected

                model = build_baseline_model(
                    model_name,
                    model_config.get("parameters", {}),
                    seed=int(seed),
                )
                predictions, resources = _fit_predict_one(
                    model_name,
                    model,
                    prepared.X_train,
                    dataset.y_train,
                    prepared.X_test,
                )
                metrics = evaluate_binary_classification(dataset.y_test, predictions)
                parameters = dict(model.get_params(deep=False))
                parameters.pop("random_state", None)

                feature_selection_hash = stable_feature_selection_signature(
                    prepared.selection.selected_features,
                    prepared.selection.selected_scores["mi_score"].to_numpy(dtype=float),
                    prepared.selection.weights,
                )

                row: dict[str, Any] = {
                    "dataset_key": dataset_key,
                    "dataset": dataset_name,
                    "profile": profile_name,
                    "approach": "Baseline",
                    "model": model_name,
                    "model_name": BASELINE_MODEL_NAMES[model_name],
                    "model_family": BASELINE_MODEL_FAMILIES[model_name],
                    "feature_set": f"FS-{int(fs_size)}",
                    "fs_size": int(fs_size),
                    "seed": int(seed),
                    "feature_selection_method": "mutual_information",
                    "feature_selection_seed": int(feature_config["random_seed"]),
                    "selected_features": "|".join(prepared.selection.selected_features),
                    "feature_selection_hash": feature_selection_hash,
                    "model_parameters_json": json.dumps(parameters, sort_keys=True, default=str),
                    "train_records": int(len(dataset.X_train)),
                    "test_records": int(len(dataset.X_test)),
                    "train_normal_records": int(np.sum(dataset.y_train == 0)),
                    "train_attack_records": int(np.sum(dataset.y_train == 1)),
                    "test_normal_records": int(np.sum(dataset.y_test == 0)),
                    "test_attack_records": int(np.sum(dataset.y_test == 1)),
                    "train_partition_hash": train_hash,
                    "test_partition_hash": test_hash,
                    **resources,
                    **metrics.to_dict(),
                }
                rows.append(row)

                categories = dataset.test_attack_categories
                if categories is not None:
                    category_tables.append(
                        attack_category_analysis(
                            dataset.y_test,
                            predictions,
                            categories,
                            run_metadata={
                                "dataset_key": dataset_key,
                                "dataset": dataset_name,
                                "profile": profile_name,
                                "model": model_name,
                                "model_name": BASELINE_MODEL_NAMES[model_name],
                                "model_family": BASELINE_MODEL_FAMILIES[model_name],
                                "feature_set": f"FS-{int(fs_size)}",
                                "fs_size": int(fs_size),
                                "seed": int(seed),
                                "train_partition_hash": train_hash,
                                "test_partition_hash": test_hash,
                            },
                        )
                    )

                if progress_callback is not None:
                    progress_callback(len(rows), run_limit, row)

    selected = _selected_feature_table(
        prepared_sets,
        dataset_name=dataset_name,
        profile_name=profile_name,
    )
    category_frame = (
        pd.concat(category_tables, ignore_index=True)
        if category_tables
        else pd.DataFrame()
    )
    return pd.DataFrame(rows), category_frame, selected


def _aggregate_metric(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=0))


def summarize_baseline_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated baseline runs without mixing model settings."""

    if results.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_columns = [
        "dataset_key",
        "dataset",
        "profile",
        "model",
        "model_name",
        "model_family",
        "feature_set",
        "fs_size",
        "model_parameters_json",
        "selected_features",
        "feature_selection_hash",
        "train_records",
        "test_records",
        "train_partition_hash",
        "test_partition_hash",
    ]
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "fit_time_sec",
        "prediction_time_sec",
        "total_time_sec",
        "peak_memory_increase_mb",
        "peak_process_memory_mb",
        "serialized_model_size_kb",
    ]

    for keys, group in results.groupby(group_columns, dropna=False, sort=False):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "approach": "Baseline",
                "runs": int(len(group)),
                "seeds": "|".join(map(str, sorted(group["seed"].unique()))),
                "comparison_status": "baseline_result",
                "fpr_constraint_met": False,
            }
        )
        for metric in metric_columns:
            mean, std = _aggregate_metric(group, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def _select_fw_lnsa_summaries(
    fw_results: pd.DataFrame,
    *,
    profile_name: str,
    feature_sizes: set[int],
    fpr_limit: float,
    train_partition_hash: str,
    test_partition_hash: str,
    train_records: int,
    test_records: int,
    feature_hashes_by_size: Mapping[int, str],
) -> tuple[pd.DataFrame, str]:
    """Select one stable FW-LNSA configuration per method and feature set."""

    if fw_results.empty:
        return pd.DataFrame(), "fw_lnsa_results_empty"

    required = {"method", "method_name", "feature_set", "fs_size", "seed", "fpr", "recall", "f1"}
    if not required.issubset(fw_results.columns):
        return pd.DataFrame(), "fw_lnsa_results_schema_incompatible"

    compatible = fw_results.copy()
    if "profile" in compatible.columns:
        compatible = compatible[compatible["profile"].astype(str) == profile_name]
    compatible = compatible[compatible["fs_size"].astype(int).isin(feature_sizes)]

    if "train_partition_hash" in compatible.columns and "test_partition_hash" in compatible.columns:
        compatible = compatible[
            (compatible["train_partition_hash"].astype(str) == train_partition_hash)
            & (compatible["test_partition_hash"].astype(str) == test_partition_hash)
        ]
        compatibility_reason = "partition_hash_match"
    elif "train_records" in compatible.columns and "test_records" in compatible.columns:
        compatible = compatible[
            (compatible["train_records"].astype(int) == train_records)
            & (compatible["test_records"].astype(int) == test_records)
        ]
        compatibility_reason = "record_count_match_hash_unavailable"
    else:
        return pd.DataFrame(), "fw_lnsa_partition_metadata_unavailable"

    if compatible.empty:
        return pd.DataFrame(), "no_compatible_fw_lnsa_rows"

    if "feature_selection_hash" in compatible.columns:
        feature_mask = np.zeros(len(compatible), dtype=bool)
        for fs_size, expected_hash in feature_hashes_by_size.items():
            feature_mask |= (
                (compatible["fs_size"].astype(int).to_numpy() == int(fs_size))
                & (compatible["feature_selection_hash"].astype(str).to_numpy() == expected_hash)
            )
        compatible = compatible.loc[feature_mask]
        compatibility_reason = f"{compatibility_reason};feature_selection_hash_match"
    elif "selected_features" in compatible.columns:
        compatibility_reason = f"{compatibility_reason};feature_hash_unavailable"
    else:
        compatibility_reason = f"{compatibility_reason};feature_metadata_unavailable"

    if compatible.empty:
        return pd.DataFrame(), "no_compatible_feature_selection_rows"

    hyperparameter_columns = [
        column
        for column in [
            "method",
            "method_name",
            "method_role",
            "feature_set",
            "fs_size",
            "selected_features",
            "feature_selection_hash",
            "detector_budget",
            "self_threshold_config",
            "detection_threshold_config",
            "threshold_scale",
            "train_records",
            "test_records",
            "train_partition_hash",
            "test_partition_hash",
        ]
        if column in compatible.columns
    ]
    metric_columns = [
        column
        for column in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "fpr",
            "fnr",
            "total_time_sec",
            "generation_time_sec",
            "detection_time_sec",
            "retained_detectors",
            "detector_memory_bytes",
        ]
        if column in compatible.columns
    ]

    grouped_rows: list[dict[str, Any]] = []
    for keys, group in compatible.groupby(hyperparameter_columns, dropna=False, sort=False):
        row = dict(zip(hyperparameter_columns, keys))
        row["runs"] = int(len(group))
        row["seeds"] = "|".join(map(str, sorted(group["seed"].unique())))
        for metric in metric_columns:
            mean, std = _aggregate_metric(group, metric)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        row["fpr_constraint_met"] = bool(row.get("fpr_mean", math.inf) <= fpr_limit)
        grouped_rows.append(row)

    grouped = pd.DataFrame(grouped_rows)
    selected_rows: list[pd.Series] = []
    for (_method, _fs_size), group in grouped.groupby(["method", "fs_size"], sort=False):
        controlled = group[group["fpr_constraint_met"]]
        candidate = controlled if not controlled.empty else group
        candidate = candidate.sort_values(
            ["recall_mean", "f1_mean", "accuracy_mean", "total_time_sec_mean"],
            ascending=[False, False, False, True],
        )
        selected_rows.append(candidate.iloc[0])

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["approach"] = "FW-LNSA"
    selected["model"] = selected["method"]
    selected["model_name"] = selected["method_name"]
    selected["model_family"] = "detector_based"
    selected["profile"] = profile_name
    selected["comparison_status"] = compatibility_reason
    return selected, compatibility_reason


def build_fw_lnsa_comparison(
    baseline_results: pd.DataFrame,
    *,
    fw_lnsa_results_path: str | Path,
    fpr_limit: float,
) -> pd.DataFrame:
    """Combine compatible baseline summaries with selected FW-LNSA rows."""

    baseline_summary = summarize_baseline_results(baseline_results)
    if baseline_summary.empty:
        return baseline_summary

    fw_path = Path(fw_lnsa_results_path)
    baseline_summary["fpr_constraint_met"] = (
        baseline_summary["fpr_mean"] <= float(fpr_limit)
    )
    baseline_summary["comparison_status"] = (
        "fw_lnsa_results_not_found" if not fw_path.exists() else "baseline_result"
    )
    if not fw_path.exists():
        return baseline_summary

    fw_results = pd.read_csv(fw_path)
    first = baseline_summary.iloc[0]
    fw_summary, status = _select_fw_lnsa_summaries(
        fw_results,
        profile_name=str(first["profile"]),
        feature_sizes=set(baseline_summary["fs_size"].astype(int)),
        fpr_limit=float(fpr_limit),
        train_partition_hash=str(first["train_partition_hash"]),
        test_partition_hash=str(first["test_partition_hash"]),
        train_records=int(first["train_records"]),
        test_records=int(first["test_records"]),
        feature_hashes_by_size={
            int(row.fs_size): str(row.feature_selection_hash)
            for row in baseline_summary[["fs_size", "feature_selection_hash"]]
            .drop_duplicates()
            .itertuples(index=False)
        },
    )
    if fw_summary.empty:
        baseline_summary["comparison_status"] = status
        return baseline_summary

    all_columns = sorted(set(baseline_summary.columns) | set(fw_summary.columns))
    baseline_aligned = baseline_summary.reindex(columns=all_columns)
    fw_aligned = fw_summary.reindex(columns=all_columns)
    combined = pd.concat([baseline_aligned, fw_aligned], ignore_index=True)
    return combined.sort_values(
        ["feature_set", "approach", "f1_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def save_baseline_outputs(
    *,
    results: pd.DataFrame,
    comparison: pd.DataFrame,
    category_analysis: pd.DataFrame,
    selected_features: pd.DataFrame,
    output_dir: str | Path,
    output_names: Mapping[str, str],
) -> dict[str, Path]:
    """Save baseline tables under results/tables."""

    table_dir = ensure_dir(Path(output_dir) / "tables")
    output_paths = {
        "baseline_results": save_dataframe(
            results,
            table_dir / str(output_names["baseline_results"]),
        ),
        "comparison_table": save_dataframe(
            comparison,
            table_dir / str(output_names["comparison_table"]),
        ),
        "attack_category_analysis": save_dataframe(
            category_analysis,
            table_dir / str(output_names["attack_category_analysis"]),
        ),
    }
    if output_names.get("selected_features"):
        output_paths["selected_features"] = save_dataframe(
            selected_features,
            table_dir / str(output_names["selected_features"]),
        )
    return output_paths


def run_prepared_baseline_experiments(
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    *,
    dataset_key: str,
    output_dir: str | Path,
    fw_lnsa_results_path: str | Path,
    max_runs: int | None = None,
    progress_callback: ProgressCallback | None = None,
    save_outputs: bool = True,
) -> BaselineOutputs:
    """Run and optionally save a complete baseline package for one dataset."""

    results, categories, selected_features = run_prepared_baseline_grid(
        dataset,
        config,
        dataset_key=dataset_key,
        max_runs=max_runs,
        progress_callback=progress_callback,
    )
    comparison = build_fw_lnsa_comparison(
        results,
        fw_lnsa_results_path=fw_lnsa_results_path,
        fpr_limit=float(config["selection"]["balanced_fpr_limit"]),
    )

    output_paths: dict[str, Path] = {}
    if save_outputs:
        output_paths = save_baseline_outputs(
            results=results,
            comparison=comparison,
            category_analysis=categories,
            selected_features=selected_features,
            output_dir=output_dir,
            output_names=config["outputs"][dataset_key],
        )

    return BaselineOutputs(
        results=results,
        comparison=comparison,
        attack_category_analysis=categories,
        selected_features=selected_features,
        output_paths=output_paths,
    )
