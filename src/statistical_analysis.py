"""Statistical summaries and controlled ablations for FW-LNSA research results.

The functions in this module operate only on saved experiment rows. They do not
fit models. This separation keeps the analysis reproducible and allows tables to
be regenerated without repeating expensive detector generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .utils import ensure_dir, save_dataframe

METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "fnr",
    "total_time_sec",
)

FW_GROUP_COLUMNS = (
    "dataset",
    "profile",
    "method",
    "method_name",
    "method_role",
    "feature_set",
    "fs_size",
    "detector_budget",
    "self_threshold_config",
    "detection_threshold_config",
    "threshold_scale",
    "selected_features",
    "feature_selection_hash",
    "train_records",
    "test_records",
    "train_partition_hash",
    "test_partition_hash",
)

BASELINE_GROUP_COLUMNS = (
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
)


@dataclass(frozen=True)
class ResearchAnalysisOutputs:
    """Generated publication-support tables and their saved paths."""

    tables: dict[str, pd.DataFrame]
    output_paths: dict[str, Path]


def _available(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)


def mean_std_ci(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    """Return mean, sample standard deviation, and two-sided t confidence bounds."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    count = len(array)
    if count == 0:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    mean = float(array.mean())
    if count == 1:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}

    std = float(array.std(ddof=1))
    standard_error = std / np.sqrt(count)
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=count - 1))
    margin = critical * standard_error
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def _aggregate_repeated_runs(
    results: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    confidence: float,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    groups = _available(results, group_columns)
    if not groups:
        raise ValueError("No grouping columns are available in the result table.")

    rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(groups, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["runs"] = int(len(group))
        if "seed" in group.columns:
            seeds = sorted(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int).unique())
            row["seeds"] = "|".join(map(str, seeds))
            row["seed_count"] = len(seeds)
        for metric in metric_columns:
            summary = mean_std_ci(_numeric_values(group, metric), confidence=confidence)
            for suffix, value in summary.items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_fw_results(
    results: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate FW-LNSA seed repeats without mixing hyperparameter settings."""

    metrics = list(METRIC_COLUMNS) + [
        "generation_time_sec",
        "detection_time_sec",
        "retained_detectors",
        "detector_memory_bytes",
        "retention_rate",
    ]
    return _aggregate_repeated_runs(
        results,
        group_columns=FW_GROUP_COLUMNS,
        metric_columns=_available(results, metrics),
        confidence=confidence,
    )


def aggregate_baseline_results(
    results: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate baseline seed repeats without mixing model configurations."""

    metrics = list(METRIC_COLUMNS) + [
        "fit_time_sec",
        "prediction_time_sec",
        "peak_memory_increase_mb",
        "peak_process_memory_mb",
        "serialized_model_size_kb",
    ]
    return _aggregate_repeated_runs(
        results,
        group_columns=BASELINE_GROUP_COLUMNS,
        metric_columns=_available(results, metrics),
        confidence=confidence,
    )


def select_fpr_controlled_configs(
    summary: pd.DataFrame,
    *,
    fpr_limit: float,
) -> pd.DataFrame:
    """Select stable configurations using aggregated seed results."""

    if summary.empty:
        return pd.DataFrame()
    required = {"dataset", "method", "fs_size", "f1_mean", "recall_mean", "fpr_mean"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"FW-LNSA summary is missing columns: {sorted(missing)}")

    selected_rows: list[pd.Series] = []
    for (_dataset, _method, _fs_size), group in summary.groupby(
        ["dataset", "method", "fs_size"], sort=False
    ):
        controlled = group[group["fpr_mean"] <= float(fpr_limit)]
        candidates = controlled if not controlled.empty else group
        selection_rule = (
            f"Aggregated FPR <= {fpr_limit:.2f}"
            if not controlled.empty
            else f"Fallback best F1 because aggregated FPR exceeded {fpr_limit:.2f}"
        )
        sort_columns = ["f1_mean", "recall_mean", "fpr_mean"]
        ascending = [False, False, True]
        if "total_time_sec_mean" in candidates.columns:
            sort_columns.append("total_time_sec_mean")
            ascending.append(True)
        selected = candidates.sort_values(sort_columns, ascending=ascending).iloc[0].copy()
        selected["selection_rule"] = selection_rule
        selected["fpr_constraint_met"] = bool(selected["fpr_mean"] <= float(fpr_limit))
        selected_rows.append(selected)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _metric_deltas(merged: pd.DataFrame, left_suffix: str, right_suffix: str) -> pd.DataFrame:
    for metric in METRIC_COLUMNS:
        left = f"{metric}_{left_suffix}"
        right = f"{metric}_{right_suffix}"
        if left in merged.columns and right in merged.columns:
            merged[f"delta_{metric}"] = merged[right] - merged[left]
    return merged


def hamming_weight_ablation(results: pd.DataFrame) -> pd.DataFrame:
    """Pair Hamming and Weighted Hamming rows at identical settings and seeds."""

    if results.empty or "method" not in results.columns:
        return pd.DataFrame()
    left = results[results["method"] == "hamming"].copy()
    right = results[results["method"] == "weighted_hamming"].copy()
    if left.empty or right.empty:
        return pd.DataFrame()

    pair_columns = _available(
        results,
        [
            "dataset",
            "profile",
            "fs_size",
            "feature_set",
            "seed",
            "detector_budget",
            "self_threshold_config",
            "detection_threshold_config",
            "train_partition_hash",
            "test_partition_hash",
        ],
    )
    keep_metrics = _available(results, METRIC_COLUMNS)
    merged = left[pair_columns + keep_metrics].merge(
        right[pair_columns + keep_metrics],
        on=pair_columns,
        how="inner",
        suffixes=("_hamming", "_weighted_hamming"),
        validate="one_to_one",
    )
    return _metric_deltas(merged, "hamming", "weighted_hamming")


def matching_formulation_ablation(best_configs: pd.DataFrame) -> pd.DataFrame:
    """Compare selected Weighted Hamming and Weighted SMC configurations."""

    if best_configs.empty:
        return pd.DataFrame()
    subset = best_configs[
        best_configs["method"].isin(["weighted_hamming", "weighted_smc"])
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    index_columns = _available(subset, ["dataset", "profile", "feature_set", "fs_size"])
    value_columns = _available(
        subset,
        ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "fpr_mean", "total_time_sec_mean"],
    )
    pivot = subset.pivot_table(
        index=index_columns,
        columns="method",
        values=value_columns,
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{method}" for metric, method in pivot.columns]
    result = pivot.reset_index()
    for metric in ["accuracy", "precision", "recall", "f1", "fpr", "total_time_sec"]:
        left = f"{metric}_mean_weighted_hamming"
        right = f"{metric}_mean_weighted_smc"
        if left in result.columns and right in result.columns:
            result[f"delta_{metric}"] = result[right] - result[left]
    return result


def feature_set_ablation(summary: pd.DataFrame) -> pd.DataFrame:
    """Pair FS-10 and FS-20 under the same method and detector settings."""

    if summary.empty or not {10, 20}.issubset(set(summary["fs_size"].astype(int))):
        return pd.DataFrame()
    fs10 = summary[summary["fs_size"].astype(int) == 10].copy()
    fs20 = summary[summary["fs_size"].astype(int) == 20].copy()
    pair_columns = _available(
        summary,
        [
            "dataset",
            "profile",
            "method",
            "detector_budget",
            "self_threshold_config",
            "detection_threshold_config",
            "threshold_scale",
            "train_partition_hash",
            "test_partition_hash",
        ],
    )
    metrics = _available(summary, [f"{metric}_mean" for metric in METRIC_COLUMNS])
    merged = fs10[pair_columns + metrics].merge(
        fs20[pair_columns + metrics],
        on=pair_columns,
        how="inner",
        suffixes=("_fs10", "_fs20"),
    )
    for metric in METRIC_COLUMNS:
        left = f"{metric}_mean_fs10"
        right = f"{metric}_mean_fs20"
        if left in merged.columns and right in merged.columns:
            merged[f"delta_{metric}"] = merged[right] - merged[left]
    return merged


def sensitivity_summary(results: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Aggregate metric behavior across one controlled sensitivity variable."""

    if results.empty or variable not in results.columns:
        return pd.DataFrame()
    group_columns = _available(
        results,
        ["dataset", "profile", "method", "method_name", "feature_set", "fs_size", variable],
    )
    return _aggregate_repeated_runs(
        results,
        group_columns=group_columns,
        metric_columns=_available(results, METRIC_COLUMNS + ("retained_detectors",)),
        confidence=0.95,
    )


def seed_stability_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Report stability across seeds for each method and feature set."""

    if results.empty:
        return pd.DataFrame()
    group_columns = _available(
        results,
        ["dataset", "profile", "method", "method_name", "feature_set", "fs_size"],
    )
    summary = _aggregate_repeated_runs(
        results,
        group_columns=group_columns,
        metric_columns=_available(results, ["f1", "recall", "fpr", "total_time_sec"]),
        confidence=0.95,
    )
    for metric in ("f1", "recall", "fpr", "total_time_sec"):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in summary.columns and std_col in summary.columns:
            denominator = summary[mean_col].abs().replace(0, np.nan)
            summary[f"{metric}_coefficient_of_variation"] = summary[std_col] / denominator
    return summary


def cross_dataset_summary(best_configs: pd.DataFrame) -> pd.DataFrame:
    """Align best FW-LNSA configurations across datasets for generalization review."""

    if best_configs.empty:
        return pd.DataFrame()
    columns = _available(
        best_configs,
        [
            "dataset",
            "profile",
            "method",
            "method_name",
            "feature_set",
            "fs_size",
            "detector_budget",
            "self_threshold_config",
            "detection_threshold_config",
            "runs",
            "seed_count",
            "accuracy_mean",
            "accuracy_std",
            "precision_mean",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "fpr_mean",
            "fpr_std",
            "total_time_sec_mean",
            "retained_detectors_mean",
            "fpr_constraint_met",
            "selection_rule",
        ],
    )
    return best_configs.loc[:, columns].sort_values(
        ["method", "fs_size", "dataset"], ignore_index=True
    )


def build_readiness_report(
    fw_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    *,
    expected: Mapping[str, Any],
) -> pd.DataFrame:
    """Check whether saved results satisfy the planned publication coverage."""

    rows: list[dict[str, Any]] = []
    expected_seeds = set(map(int, expected.get("seeds", [])))
    expected_feature_sizes = set(map(int, expected.get("feature_sizes", [])))
    expected_methods = set(map(str, expected.get("methods", [])))
    expected_budgets = set(map(int, expected.get("detector_budgets", [])))
    expected_models = set(map(str, expected.get("baseline_models", [])))
    datasets = list(expected.get("datasets", ["NSL-KDD", "CICIDS2017"]))

    for dataset_name in datasets:
        fw_dataset = fw_results[fw_results.get("dataset", pd.Series(dtype=str)).astype(str) == dataset_name]
        baseline_dataset = baseline_results[
            baseline_results.get("dataset", pd.Series(dtype=str)).astype(str) == dataset_name
        ]

        fw_partitions = set()
        baseline_partitions = set()
        if {"train_partition_hash", "test_partition_hash"}.issubset(fw_dataset.columns):
            fw_partitions = set(
                zip(
                    fw_dataset["train_partition_hash"].astype(str),
                    fw_dataset["test_partition_hash"].astype(str),
                )
            )
        if {"train_partition_hash", "test_partition_hash"}.issubset(
            baseline_dataset.columns
        ):
            baseline_partitions = set(
                zip(
                    baseline_dataset["train_partition_hash"].astype(str),
                    baseline_dataset["test_partition_hash"].astype(str),
                )
            )

        feature_hash_match = True
        for fs_size in expected_feature_sizes:
            fw_hashes = set(
                fw_dataset.loc[
                    pd.to_numeric(fw_dataset.get("fs_size"), errors="coerce") == fs_size,
                    "feature_selection_hash",
                ].astype(str)
            ) if "feature_selection_hash" in fw_dataset.columns else set()
            baseline_hashes = set(
                baseline_dataset.loc[
                    pd.to_numeric(baseline_dataset.get("fs_size"), errors="coerce") == fs_size,
                    "feature_selection_hash",
                ].astype(str)
            ) if "feature_selection_hash" in baseline_dataset.columns else set()
            feature_hash_match = feature_hash_match and bool(
                fw_hashes and baseline_hashes and fw_hashes == baseline_hashes
            )

        checks = {
            "FW-LNSA methods": expected_methods.issubset(set(fw_dataset.get("method", []))),
            "Feature sets": expected_feature_sizes.issubset(
                set(pd.to_numeric(fw_dataset.get("fs_size", []), errors="coerce").dropna().astype(int))
            ),
            "Detector budgets": expected_budgets.issubset(
                set(pd.to_numeric(fw_dataset.get("detector_budget", []), errors="coerce").dropna().astype(int))
            ),
            "FW-LNSA seeds": expected_seeds.issubset(
                set(pd.to_numeric(fw_dataset.get("seed", []), errors="coerce").dropna().astype(int))
            ),
            "Baseline models": expected_models.issubset(set(baseline_dataset.get("model", []))),
            "Baseline seeds": expected_seeds.issubset(
                set(pd.to_numeric(baseline_dataset.get("seed", []), errors="coerce").dropna().astype(int))
            ),
            "Partition compatibility": bool(
                fw_partitions and baseline_partitions and fw_partitions == baseline_partitions
            ),
            "Feature-selection compatibility": feature_hash_match,
        }
        for check, passed in checks.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "check": check,
                    "passed": bool(passed),
                    "status": "complete" if passed else "incomplete",
                }
            )
    return pd.DataFrame(rows)


def analyze_research_results(
    *,
    fw_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    output_dir: str | Path,
    fpr_limit: float = 0.10,
    confidence: float = 0.95,
    expected: Mapping[str, Any] | None = None,
) -> ResearchAnalysisOutputs:
    """Generate and save all repeated-run, ablation, and sensitivity tables."""

    fw_summary = aggregate_fw_results(fw_results, confidence=confidence)
    baseline_summary = aggregate_baseline_results(baseline_results, confidence=confidence)
    best_configs = select_fpr_controlled_configs(fw_summary, fpr_limit=fpr_limit)

    tables = {
        "fw_summary": fw_summary,
        "baseline_summary": baseline_summary,
        "best_fpr_controlled": best_configs,
        "hamming_weight_ablation": hamming_weight_ablation(fw_results),
        "matching_formulation_ablation": matching_formulation_ablation(best_configs),
        "feature_set_ablation": feature_set_ablation(fw_summary),
        "detector_budget_sensitivity": sensitivity_summary(fw_results, "detector_budget"),
        "self_threshold_sensitivity": sensitivity_summary(fw_results, "self_threshold_config"),
        "detection_threshold_sensitivity": sensitivity_summary(
            fw_results, "detection_threshold_config"
        ),
        "seed_stability": seed_stability_summary(fw_results),
        "cross_dataset_summary": cross_dataset_summary(best_configs),
    }
    if expected is not None:
        tables["readiness_report"] = build_readiness_report(
            fw_results,
            baseline_results,
            expected=expected,
        )

    table_dir = ensure_dir(Path(output_dir) / "tables")
    filenames = {
        "fw_summary": "research_fw_lnsa_summary.csv",
        "baseline_summary": "research_baseline_summary.csv",
        "best_fpr_controlled": "research_best_fpr_controlled.csv",
        "hamming_weight_ablation": "research_hamming_weight_ablation.csv",
        "matching_formulation_ablation": "research_matching_formulation_ablation.csv",
        "feature_set_ablation": "research_feature_set_ablation.csv",
        "detector_budget_sensitivity": "research_detector_budget_sensitivity.csv",
        "self_threshold_sensitivity": "research_self_threshold_sensitivity.csv",
        "detection_threshold_sensitivity": "research_detection_threshold_sensitivity.csv",
        "seed_stability": "research_seed_stability.csv",
        "cross_dataset_summary": "research_cross_dataset_summary.csv",
        "readiness_report": "research_readiness_report.csv",
    }
    output_paths = {
        name: save_dataframe(frame, table_dir / filenames[name])
        for name, frame in tables.items()
    }
    return ResearchAnalysisOutputs(tables=tables, output_paths=output_paths)
