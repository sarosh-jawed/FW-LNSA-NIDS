"""Statistical audit and manuscript evidence generation.

The analysis operates only on completed confirmatory artifacts and compatible
validation-calibrated baselines. It creates publication tables, paired tests,
calibration-transfer diagnostics, reproducibility checks, figures, and a
self-contained writer handoff package without copying raw datasets.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .research_execution import atomic_write_csv, atomic_write_json, environment_manifest
from .utils import ensure_dir, stable_mapping_signature


PRIMARY_METRICS = (
    "test_f1",
    "test_recall",
    "test_fpr",
    "test_balanced_accuracy",
    "test_mcc",
)
SECONDARY_METRICS = (
    "retention_rate",
    "test_detection_time_sec",
    "test_prediction_records_per_sec",
    "serialized_model_bytes",
    "test_peak_prediction_rss_delta_mb",
)
LOWER_IS_BETTER = {
    "test_fpr",
    "test_detection_time_sec",
    "serialized_model_bytes",
    "test_peak_prediction_rss_delta_mb",
}


@dataclass(frozen=True)
class ManuscriptAnalysisOutputs:
    tables: dict[str, pd.DataFrame]
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    manifest: dict[str, Any]
    manifest_path: Path


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def mean_std_ci(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(array.mean())
    if len(array) == 1:
        return {"count": 1, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(array.std(ddof=1))
    critical = float(stats.t.ppf((1 + confidence) / 2, len(array) - 1))
    margin = critical * std / math.sqrt(len(array))
    return {"count": len(array), "mean": mean, "std": std, "ci_low": mean - margin, "ci_high": mean + margin}


def bootstrap_paired_ci(
    differences: Sequence[float],
    *,
    confidence: float = 0.95,
    iterations: int = 5000,
    random_seed: int = 2026,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1 or np.allclose(values, values[0]):
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(random_seed)
    samples = rng.choice(values, size=(int(iterations), len(values)), replace=True).mean(axis=1)
    alpha = 1 - confidence
    return (
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1 - alpha / 2)),
    )


def rank_biserial_from_differences(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & ~np.isclose(values, 0.0)]
    if len(values) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    total = positive + negative
    return float((positive - negative) / total) if total else 0.0


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if len(finite_indices) == 0:
        return adjusted
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _safe_wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    finite = differences[np.isfinite(differences)]
    if len(finite) == 0:
        return np.nan, np.nan
    if np.allclose(finite, 0.0):
        return 0.0, 1.0
    result = stats.wilcoxon(finite, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def paired_fw_lnsa_analysis(
    final_seed_results: pd.DataFrame,
    *,
    metrics: Sequence[str] = PRIMARY_METRICS + SECONDARY_METRICS,
    confidence: float = 0.95,
    bootstrap_iterations: int = 5000,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Compare locked Hamming and weighted-similarity procedures by seed."""

    required = {"dataset_key", "method", "fs_size", "target_fpr", "seed"}
    missing = required - set(final_seed_results.columns)
    if missing:
        raise ValueError(f"Confirmatory results are missing columns: {sorted(missing)}")
    hamming = final_seed_results[final_seed_results["method"] == "hamming"].copy()
    weighted = final_seed_results[final_seed_results["method"] == "weighted_similarity"].copy()
    pair_keys = ["dataset_key", "dataset", "fs_size", "feature_set", "target_fpr", "seed"]
    available_metrics = [metric for metric in metrics if metric in final_seed_results.columns]
    merged = hamming[pair_keys + available_metrics].merge(
        weighted[pair_keys + available_metrics],
        on=pair_keys,
        how="inner",
        suffixes=("_hamming", "_weighted"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    group_keys = ["dataset_key", "dataset", "fs_size", "feature_set", "target_fpr"]
    for keys, group in merged.groupby(group_keys, sort=False, dropna=False):
        base = dict(zip(group_keys, keys))
        for metric in available_metrics:
            h = pd.to_numeric(group[f"{metric}_hamming"], errors="coerce").to_numpy(dtype=float)
            w = pd.to_numeric(group[f"{metric}_weighted"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(h) & np.isfinite(w)
            raw_diff = w[valid] - h[valid]
            favorable = -raw_diff if metric in LOWER_IS_BETTER else raw_diff
            ci_low, ci_high = bootstrap_paired_ci(
                raw_diff,
                confidence=confidence,
                iterations=bootstrap_iterations,
                random_seed=random_seed + len(rows),
            )
            statistic, p_value = _safe_wilcoxon(raw_diff)
            tolerance = 1e-12
            rows.append({
                **base,
                "metric": metric,
                "pairs": int(len(raw_diff)),
                "hamming_mean": float(np.mean(h[valid])) if np.any(valid) else np.nan,
                "weighted_mean": float(np.mean(w[valid])) if np.any(valid) else np.nan,
                "mean_difference_weighted_minus_hamming": float(np.mean(raw_diff)) if len(raw_diff) else np.nan,
                "median_difference_weighted_minus_hamming": float(np.median(raw_diff)) if len(raw_diff) else np.nan,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_effect": rank_biserial_from_differences(favorable),
                "weighted_wins": int(np.sum(favorable > tolerance)),
                "ties": int(np.sum(np.abs(favorable) <= tolerance)),
                "weighted_losses": int(np.sum(favorable < -tolerance)),
                "preferred_direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p"] = holm_adjust(result["p_value"].to_numpy(dtype=float))
        result["significant_after_holm"] = result["holm_adjusted_p"] < 0.05
    return result


def _aggregate_seed_results(
    frame: pd.DataFrame,
    *,
    approach: str,
    label_column: str,
    confidence: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    group_columns = ["dataset_key", "dataset", label_column, "feature_set", "fs_size", "target_fpr"]
    metrics = [
        "test_accuracy", "test_precision", "test_recall", "test_f1", "test_fpr",
        "test_specificity", "test_balanced_accuracy", "test_mcc", "test_pr_auc",
        "test_roc_auc", "validation_fpr", "calibration_fpr",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, keys))
        row["approach"] = approach
        row["runs"] = int(len(group))
        row["seeds"] = "|".join(map(str, sorted(group["seed"].astype(int).unique())))
        for metric in metrics:
            if metric not in group.columns:
                continue
            summary = mean_std_ci(_numeric(group[metric]), confidence)
            for suffix, value in summary.items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_performance_summary(
    fw_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    fw = fw_results.copy()
    fw["procedure"] = fw["method_name"].fillna(fw["method"])
    baseline = baseline_results.copy()
    baseline["procedure"] = baseline["model_name"].fillna(baseline["model"])
    return pd.concat(
        [
            _aggregate_seed_results(fw, approach="FW-LNSA", label_column="procedure", confidence=confidence),
            _aggregate_seed_results(baseline, approach="Baseline", label_column="procedure", confidence=confidence),
        ],
        ignore_index=True,
        sort=False,
    )


def build_calibration_reliability(
    fw_results: pd.DataFrame, baseline_results: pd.DataFrame
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frame, approach, label in [
        (fw_results, "FW-LNSA", "method_name"),
        (baseline_results, "Baseline", "model_name"),
    ]:
        if frame.empty:
            continue
        subset = frame.copy()
        subset["approach"] = approach
        subset["procedure"] = subset[label]
        subset["validation_fpr_observed"] = pd.to_numeric(subset["validation_fpr"], errors="coerce")
        subset["test_fpr_observed"] = pd.to_numeric(subset["test_fpr"], errors="coerce")
        subset["validation_absolute_error"] = (subset["validation_fpr_observed"] - subset["target_fpr"]).abs()
        subset["test_absolute_error"] = (subset["test_fpr_observed"] - subset["target_fpr"]).abs()
        subset["test_minus_validation_fpr"] = subset["test_fpr_observed"] - subset["validation_fpr_observed"]
        subset["test_target_met"] = subset["test_fpr_observed"] <= subset["target_fpr"] + 1e-12
        rows.append(subset[[
            "dataset_key", "dataset", "approach", "procedure", "feature_set", "fs_size",
            "target_fpr", "seed", "validation_fpr_observed", "test_fpr_observed",
            "validation_absolute_error", "test_absolute_error", "test_minus_validation_fpr",
            "test_target_met", "test_f1", "test_recall", "test_balanced_accuracy", "test_mcc",
        ]])
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_fs_ablation(fw_results: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_key", "dataset", "method", "method_name", "target_fpr", "seed"]
    metrics = ["test_f1", "test_recall", "test_fpr", "test_balanced_accuracy", "test_mcc"]
    fs10 = fw_results[fw_results["fs_size"].astype(int) == 10][keys + metrics]
    fs20 = fw_results[fw_results["fs_size"].astype(int) == 20][keys + metrics]
    merged = fs10.merge(fs20, on=keys, suffixes=("_fs10", "_fs20"), validate="one_to_one")
    for metric in metrics:
        merged[f"delta_{metric}_fs20_minus_fs10"] = merged[f"{metric}_fs20"] - merged[f"{metric}_fs10"]
    return merged


def build_efficiency_summary(fw_results: pd.DataFrame, baseline_results: pd.DataFrame) -> pd.DataFrame:
    fw = fw_results.copy()
    fw["approach"] = "FW-LNSA"
    fw["procedure"] = fw["method_name"]
    fw["model_size_bytes"] = pd.to_numeric(fw["serialized_model_bytes"], errors="coerce")
    fw["prediction_records_per_sec"] = pd.to_numeric(fw["test_prediction_records_per_sec"], errors="coerce")
    fw["prediction_time_sec"] = pd.to_numeric(fw["test_detection_time_sec"], errors="coerce")
    fw["memory_delta_mb"] = pd.to_numeric(fw["test_peak_prediction_rss_delta_mb"], errors="coerce")
    fw["detector_retention"] = pd.to_numeric(fw["retention_rate"], errors="coerce")

    baseline = baseline_results.copy()
    baseline["approach"] = "Baseline"
    baseline["procedure"] = baseline["model_name"]
    baseline["model_size_bytes"] = pd.to_numeric(baseline["serialized_model_bytes"], errors="coerce")
    baseline["prediction_records_per_sec"] = pd.to_numeric(baseline["test_records_per_sec"], errors="coerce")
    baseline["prediction_time_sec"] = pd.to_numeric(baseline["test_score_time_sec"], errors="coerce")
    baseline["memory_delta_mb"] = pd.to_numeric(baseline["peak_memory_increase_mb"], errors="coerce")
    baseline["detector_retention"] = np.nan

    combined = pd.concat([fw, baseline], ignore_index=True, sort=False)
    groups = ["dataset_key", "dataset", "approach", "procedure", "feature_set", "fs_size", "target_fpr"]
    rows: list[dict[str, Any]] = []
    for keys, group in combined.groupby(groups, sort=False, dropna=False):
        row = dict(zip(groups, keys))
        row["runs"] = int(len(group))
        for metric in ["model_size_bytes", "prediction_records_per_sec", "prediction_time_sec", "memory_delta_mb", "detector_retention"]:
            summary = mean_std_ci(_numeric(group[metric]))
            for suffix, value in summary.items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def combine_category_summaries(
    fw_category: pd.DataFrame, baseline_category: pd.DataFrame
) -> pd.DataFrame:
    fw = fw_category.copy()
    if not fw.empty:
        fw["approach"] = "FW-LNSA"
        fw["procedure"] = fw["method_name"]
    baseline = baseline_category.copy()
    if not baseline.empty:
        baseline["approach"] = "Baseline"
        baseline["procedure"] = baseline["model_name"]
    columns = [
        "dataset_key", "dataset", "approach", "procedure", "feature_set", "fs_size",
        "target_fpr", "category", "category_recall_mean", "category_recall_std",
        "false_alarm_rate_mean", "false_alarm_rate_std", "runs",
    ]
    frames = []
    for frame in (fw, baseline):
        if not frame.empty:
            frames.append(frame.reindex(columns=columns))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def audit_results(
    *,
    confirmatory_root: str | Path,
    baseline_root: str | Path,
    manuscript_config: Mapping[str, Any],
    fw_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, evidence: Any, severity: str = "error") -> None:
        checks.append({
            "check": name,
            "passed": bool(passed),
            "severity": severity,
            "evidence": json.dumps(evidence, sort_keys=True, default=str),
        })

    readiness = manuscript_config["readiness"]
    confirmatory_root = Path(confirmatory_root)
    baseline_root = Path(baseline_root)
    expected_seeds = set(map(int, manuscript_config["baseline"]["seeds"]))
    expected_targets = set(round(float(x), 12) for x in manuscript_config["baseline"]["target_fprs"])

    for dataset_key in manuscript_config["baseline"]["datasets"]:
        ds_fw = fw_results[fw_results["dataset_key"] == dataset_key]
        expected_fw = int(readiness["expected_confirmatory_rows_per_dataset"])
        record(f"{dataset_key}: confirmatory row count", len(ds_fw) == expected_fw, {"actual": len(ds_fw), "expected": expected_fw})
        record(
            f"{dataset_key}: confirmatory seed coverage",
            set(ds_fw["seed"].astype(int).unique()) == expected_seeds,
            sorted(ds_fw["seed"].astype(int).unique()),
        )
        record(
            f"{dataset_key}: confirmatory result identity unique",
            not ds_fw.duplicated(["method", "fs_size", "target_fpr", "seed", "configuration_id"]).any(),
            int(ds_fw.duplicated(["method", "fs_size", "target_fpr", "seed", "configuration_id"]).sum()),
        )
        record(
            f"{dataset_key}: primary metrics complete",
            not ds_fw[list(PRIMARY_METRICS)].isna().any().any(),
            ds_fw[list(PRIMARY_METRICS)].isna().sum().to_dict(),
        )
        record(
            f"{dataset_key}: test not used for FW-LNSA selection",
            not ds_fw["test_metrics_used_for_selection"].astype(bool).any(),
            ds_fw["test_metrics_used_for_selection"].value_counts(dropna=False).to_dict(),
        )

        manifest_path = confirmatory_root / dataset_key / "manifests" / "execution_manifest.json"
        lock_path = confirmatory_root / dataset_key / "manifests" / "configuration_lock.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        record(
            f"{dataset_key}: configuration lock integrity",
            len(lock.get("locked_configurations", [])) == int(readiness["expected_locked_configurations_per_dataset"]),
            {"lock_id": lock.get("lock_id"), "rows": len(lock.get("locked_configurations", []))},
        )
        record(
            f"{dataset_key}: tuning and confirmatory seeds disjoint",
            not (set(lock.get("tuning_seeds", [])) & set(lock.get("confirmatory_seeds", []))),
            {"tuning": lock.get("tuning_seeds", []), "confirmatory": lock.get("confirmatory_seeds", [])},
        )
        record(
            f"{dataset_key}: manifest completed counts",
            manifest.get("completed", {}).get("confirmatory_rows") == expected_fw,
            manifest.get("completed", {}),
        )

        ds_base = baseline_results[baseline_results["dataset_key"] == dataset_key]
        expected_base = (
            int(readiness["expected_baseline_models"])
            * int(readiness["expected_baseline_feature_sets"])
            * len(expected_seeds)
            * int(readiness["expected_target_fprs"])
        )
        record(f"{dataset_key}: baseline row count", len(ds_base) == expected_base, {"actual": len(ds_base), "expected": expected_base})
        record(
            f"{dataset_key}: baseline identity unique",
            not ds_base.duplicated(["model", "fs_size", "target_fpr", "seed"]).any(),
            int(ds_base.duplicated(["model", "fs_size", "target_fpr", "seed"]).sum()),
        )
        record(
            f"{dataset_key}: baseline target coverage",
            set(round(float(x), 12) for x in ds_base["target_fpr"].unique()) == expected_targets,
            sorted(ds_base["target_fpr"].unique()),
        )
        record(
            f"{dataset_key}: test not used for baseline selection",
            not ds_base["test_metrics_used_for_selection"].astype(bool).any(),
            ds_base["test_metrics_used_for_selection"].value_counts(dropna=False).to_dict(),
        )
        for hash_column in ["train_partition_hash", "validation_partition_hash", "test_partition_hash", "feature_selection_hash"]:
            fw_hashes = set(ds_fw[hash_column].astype(str).unique())
            base_hashes = set(ds_base[hash_column].astype(str).unique())
            record(
                f"{dataset_key}: compatible {hash_column}",
                base_hashes.issubset(fw_hashes),
                {"fw": sorted(fw_hashes), "baseline": sorted(base_hashes)},
            )

        baseline_manifest = baseline_root / "confirmatory_baselines" / dataset_key / "manifests" / "execution_manifest.json"
        record(
            f"{dataset_key}: baseline manifest present",
            baseline_manifest.exists(),
            str(baseline_manifest),
        )

    return pd.DataFrame(checks)


def _save_table_and_latex(frame: pd.DataFrame, name: str, table_dir: Path, latex_dir: Path) -> tuple[Path, Path]:
    csv_path = atomic_write_csv(frame, table_dir / f"{name}.csv")
    latex_path = latex_dir / f"{name}.tex"
    latex_path.write_text(
        frame.to_latex(index=False, escape=True, na_rep="--", float_format=lambda value: f"{value:.4f}"),
        encoding="utf-8",
    )
    return csv_path, latex_path


def _save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> Path:
    path = figure_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_publication_figures(
    performance: pd.DataFrame,
    calibration: pd.DataFrame,
    paired: pd.DataFrame,
    efficiency: pd.DataFrame,
    categories: pd.DataFrame,
    figure_dir: str | Path,
) -> dict[str, Path]:
    output = ensure_dir(figure_dir)
    paths: dict[str, Path] = {}

    if not performance.empty:
        subset = performance.dropna(subset=["test_fpr_mean", "test_f1_mean"]).copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        for (dataset, procedure), group in subset.groupby(["dataset", "procedure"], sort=False):
            ax.plot(group["test_fpr_mean"], group["test_f1_mean"], marker="o", label=f"{dataset}: {procedure}")
        ax.set_xlabel("Observed test false-positive rate")
        ax.set_ylabel("Mean test F1")
        ax.set_title("Confirmatory F1 versus observed false-positive rate")
        ax.legend(fontsize=7)
        paths["f1_vs_fpr"] = _save_figure(fig, output, "f1_vs_observed_fpr")

        fig, ax = plt.subplots(figsize=(8, 5))
        for (dataset, procedure), group in subset.groupby(["dataset", "procedure"], sort=False):
            ax.plot(group["test_fpr_mean"], group["test_recall_mean"], marker="o", label=f"{dataset}: {procedure}")
        ax.set_xlabel("Observed test false-positive rate")
        ax.set_ylabel("Mean test recall")
        ax.set_title("Confirmatory recall versus observed false-positive rate")
        ax.legend(fontsize=7)
        paths["recall_vs_fpr"] = _save_figure(fig, output, "recall_vs_observed_fpr")

    if not paired.empty:
        subset = paired[paired["metric"] == "test_f1"].copy()
        if not subset.empty:
            labels = subset.apply(lambda row: f"{row['dataset']} {row['feature_set']} FPR={row['target_fpr']:.2f}", axis=1)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(labels, subset["mean_difference_weighted_minus_hamming"])
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("Mean paired F1 difference: weighted minus Hamming")
            ax.set_title("Paired confirmatory effect of feature-weighted matching")
            paths["paired_f1"] = _save_figure(fig, output, "paired_weighted_vs_hamming_f1")

    if not calibration.empty:
        summary = calibration.groupby(["dataset", "approach", "procedure", "target_fpr"], dropna=False)[["validation_fpr_observed", "test_fpr_observed"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 6))
        for (dataset, approach), group in summary.groupby(["dataset", "approach"], sort=False):
            ax.scatter(group["validation_fpr_observed"], group["test_fpr_observed"], label=f"{dataset}: {approach}")
        limit = max(0.1, float(summary[["validation_fpr_observed", "test_fpr_observed"]].max().max()) * 1.05)
        ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Validation FPR")
        ax.set_ylabel("Test FPR")
        ax.set_title("Validation-to-test false-positive calibration transfer")
        ax.legend(fontsize=8)
        paths["calibration_transfer"] = _save_figure(fig, output, "validation_vs_test_fpr")

    if not efficiency.empty:
        subset = efficiency.dropna(subset=["model_size_bytes_mean", "prediction_records_per_sec_mean"])
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for (dataset, approach), group in subset.groupby(["dataset", "approach"], sort=False):
                ax.scatter(group["model_size_bytes_mean"] / 1024, group["prediction_records_per_sec_mean"], label=f"{dataset}: {approach}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Mean serialized model size (KiB, log scale)")
            ax.set_ylabel("Mean prediction throughput (records/s, log scale)")
            ax.set_title("Model footprint and prediction throughput")
            ax.legend(fontsize=8)
            paths["size_throughput"] = _save_figure(fig, output, "model_size_vs_throughput")

    if not categories.empty:
        subset = categories[
            (categories["dataset_key"] == "cicids2017")
            & (categories["feature_set"] == "FS-20")
            & (categories["target_fpr"].astype(float) == 0.01)
        ].dropna(subset=["category_recall_mean"])
        if not subset.empty:
            pivot = subset.pivot_table(index="category", columns="procedure", values="category_recall_mean", aggfunc="first")
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Mean category recall")
            ax.set_title("CICIDS2017 attack-category recall at target FPR 0.01")
            ax.legend(fontsize=7)
            paths["category_recall"] = _save_figure(fig, output, "cicids2017_category_recall")

    return paths


def analyze_and_save_manuscript_evidence(
    *,
    confirmatory_root: str | Path,
    baseline_root: str | Path,
    output_root: str | Path,
    manuscript_config: Mapping[str, Any],
) -> ManuscriptAnalysisOutputs:
    confirmatory_root = Path(confirmatory_root)
    baseline_root = Path(baseline_root)
    output_root = ensure_dir(output_root)
    table_dir = ensure_dir(output_root / "publication_tables")
    seed_table_dir = ensure_dir(output_root / "seed_level_tables")
    latex_dir = ensure_dir(output_root / "latex_tables")
    figure_dir = ensure_dir(output_root / "publication_figures")

    fw_frames = []
    fw_categories = []
    baseline_frames = []
    baseline_categories = []
    selected_features = []
    data_quality = []
    for dataset_key in manuscript_config["baseline"]["datasets"]:
        fw_frames.append(pd.read_csv(confirmatory_root / dataset_key / "tables" / "final_seed_results.csv"))
        fw_categories.append(pd.read_csv(confirmatory_root / dataset_key / "tables" / "category_summary.csv"))
        selected = pd.read_csv(confirmatory_root / dataset_key / "tables" / "selected_features.csv")
        selected.insert(0, "dataset_key", dataset_key)
        selected_features.append(selected)
        quality_path = confirmatory_root / dataset_key / "tables" / "data_quality_report.csv"
        if quality_path.exists():
            quality = pd.read_csv(quality_path)
            quality.insert(0, "dataset_key", dataset_key)
            data_quality.append(quality)
        base_table = baseline_root / "confirmatory_baselines" / dataset_key / "tables"
        baseline_frames.append(pd.read_csv(base_table / "baseline_seed_results.csv"))
        baseline_categories.append(pd.read_csv(base_table / "baseline_category_summary.csv"))

    fw_results = pd.concat(fw_frames, ignore_index=True)
    fw_category = pd.concat(fw_categories, ignore_index=True)
    baseline_results = pd.concat(baseline_frames, ignore_index=True)
    baseline_category = pd.concat(baseline_categories, ignore_index=True)

    stats_config = manuscript_config["statistics"]
    confidence = float(stats_config.get("confidence", 0.95))
    paired = paired_fw_lnsa_analysis(
        fw_results,
        metrics=tuple(stats_config.get("primary_metrics", PRIMARY_METRICS))
        + tuple(stats_config.get("secondary_metrics", SECONDARY_METRICS)),
        confidence=confidence,
        bootstrap_iterations=int(stats_config.get("bootstrap_iterations", 5000)),
        random_seed=int(stats_config.get("random_seed", 2026)),
    )
    performance = build_performance_summary(fw_results, baseline_results, confidence=confidence)
    calibration = build_calibration_reliability(fw_results, baseline_results)
    fs_ablation = build_fs_ablation(fw_results)
    efficiency = build_efficiency_summary(fw_results, baseline_results)
    category = combine_category_summaries(fw_category, baseline_category)
    selected = pd.concat(selected_features, ignore_index=True)
    quality = pd.concat(data_quality, ignore_index=True) if data_quality else pd.DataFrame()
    audit = audit_results(
        confirmatory_root=confirmatory_root,
        baseline_root=baseline_root,
        manuscript_config=manuscript_config,
        fw_results=fw_results,
        baseline_results=baseline_results,
    )
    calibration_summary = calibration.groupby(
        ["dataset_key", "dataset", "approach", "procedure", "feature_set", "fs_size", "target_fpr"],
        dropna=False,
    ).agg(
        runs=("seed", "count"),
        validation_fpr_mean=("validation_fpr_observed", "mean"),
        test_fpr_mean=("test_fpr_observed", "mean"),
        test_absolute_error_mean=("test_absolute_error", "mean"),
        test_target_met_rate=("test_target_met", "mean"),
        test_f1_mean=("test_f1", "mean"),
        test_recall_mean=("test_recall", "mean"),
    ).reset_index()

    tables = {
        "confirmatory_performance": performance,
        "paired_weighted_vs_hamming": paired,
        "calibration_reliability_seed_level": calibration,
        "calibration_reliability_summary": calibration_summary,
        "feature_set_ablation_seed_level": fs_ablation,
        "efficiency_summary": efficiency,
        "attack_category_summary": category,
        "selected_features_and_weights": selected,
        "data_quality_provenance": quality,
        "final_readiness_audit": audit,
        "fw_lnsa_seed_results": fw_results,
        "baseline_seed_results": baseline_results,
    }
    table_paths: dict[str, Path] = {}
    seed_level_names = {
        "calibration_reliability_seed_level",
        "feature_set_ablation_seed_level",
        "fw_lnsa_seed_results",
        "baseline_seed_results",
    }
    for name, frame in tables.items():
        if name in seed_level_names:
            csv_path = atomic_write_csv(frame, seed_table_dir / f"{name}.csv")
            table_paths[name] = csv_path
        else:
            csv_path, latex_path = _save_table_and_latex(frame, name, table_dir, latex_dir)
            table_paths[name] = csv_path
            table_paths[f"{name}_latex"] = latex_path

    figures = generate_publication_figures(performance, calibration, paired, efficiency, category, figure_dir)
    passed = bool(not audit.empty and audit.loc[audit["severity"] == "error", "passed"].all())
    manifest = {
        "analysis_version": "1.0",
        "profile": manuscript_config["profile_name"],
        "config_hash": stable_mapping_signature(manuscript_config),
        "execution_environment": environment_manifest(Path.cwd()),
        "input_rows": {
            "fw_lnsa": int(len(fw_results)),
            "baselines": int(len(baseline_results)),
            "fw_categories": int(len(fw_category)),
            "baseline_categories": int(len(baseline_category)),
        },
        "paired_tests": int(len(paired)),
        "audit_checks": int(len(audit)),
        "audit_passed": passed,
        "test_metrics_used_for_selection": False,
        "tables": {key: str(path) for key, path in table_paths.items()},
        "figures": {key: str(path) for key, path in figures.items()},
    }
    manifest_path = output_root / "analysis_manifest.json"
    atomic_write_json(manifest, manifest_path)
    return ManuscriptAnalysisOutputs(
        tables=tables,
        table_paths=table_paths,
        figure_paths=figures,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def _copy_tree_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def _author_table(writer_config: Mapping[str, Any]) -> str:
    rows = ["| Author | Role | Affiliation | Email | ORCID |", "|---|---|---|---|---|"]
    for author in writer_config.get("authors", []):
        rows.append(
            f"| {author.get('name', '')} | {author.get('role', '')} | {author.get('affiliation', '')} | "
            f"{author.get('email', '')} | {author.get('orcid', '')} |"
        )
    return "\n".join(rows)


def build_writer_handoff(
    *,
    analysis_root: str | Path,
    confirmatory_root: str | Path,
    baseline_root: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
    manuscript_config: Mapping[str, Any],
    executed_notebook: str | Path | None = None,
) -> Path:
    """Create a focused writer package containing evidence, not raw datasets."""

    analysis_root = Path(analysis_root)
    confirmatory_root = Path(confirmatory_root)
    baseline_root = Path(baseline_root)
    repository_root = Path(repository_root)
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    evidence = ensure_dir(output / "evidence")
    reproducibility = ensure_dir(output / "reproducibility")

    manifest = json.loads((analysis_root / "analysis_manifest.json").read_text(encoding="utf-8"))
    if not bool(manifest.get("audit_passed", False)):
        raise RuntimeError("The manuscript handoff cannot be built because the final audit failed.")

    _copy_tree_if_exists(analysis_root / "publication_tables", evidence / "tables_csv")
    _copy_tree_if_exists(analysis_root / "seed_level_tables", evidence / "seed_level_csv")
    _copy_tree_if_exists(analysis_root / "latex_tables", evidence / "tables_latex")
    _copy_tree_if_exists(analysis_root / "publication_figures", evidence / "figures")
    shutil.copy2(analysis_root / "analysis_manifest.json", reproducibility / "analysis_manifest.json")
    shutil.copy2(repository_root / "configs" / "confirmatory_protocol.yaml", reproducibility / "confirmatory_protocol.yaml")
    shutil.copy2(repository_root / "configs" / "manuscript_readiness.yaml", reproducibility / "manuscript_readiness.yaml")
    shutil.copy2(repository_root / "requirements.txt", reproducibility / "requirements.txt")
    lock_requirements = repository_root / "requirements-research-lock.txt"
    if lock_requirements.exists():
        shutil.copy2(lock_requirements, reproducibility / "requirements-research-lock.txt")

    for dataset_key in manuscript_config["baseline"]["datasets"]:
        ds_dir = ensure_dir(reproducibility / dataset_key)
        for name in ["execution_manifest.json", "configuration_lock.json", "execution_state.json"]:
            source = confirmatory_root / dataset_key / "manifests" / name
            if source.exists():
                shutil.copy2(source, ds_dir / f"fw_lnsa_{name}")
        baseline_manifest = baseline_root / "confirmatory_baselines" / dataset_key / "manifests" / "execution_manifest.json"
        if baseline_manifest.exists():
            shutil.copy2(baseline_manifest, ds_dir / "baseline_execution_manifest.json")

    if executed_notebook is not None and Path(executed_notebook).exists():
        shutil.copy2(executed_notebook, reproducibility / Path(executed_notebook).name)

    writer = manuscript_config.get("writer", {})
    title = writer.get("project_title", "FW-LNSA")
    target_journal = writer.get("target_journal", "TO BE CONFIRMED")
    authors = _author_table(writer)

    documents: dict[str, str] = {
        "00_START_HERE.md": f"""# Writer handoff: {title}\n\nThis package is the only result package that should be used for manuscript writing. Begin with `06_RESULTS_INTERPRETATION.md`. Main tables are under `evidence/tables_csv/` and `evidence/tables_latex/`; seed-level audit data are isolated under `evidence/seed_level_csv/`.\n\n**Target journal:** {target_journal}\n\n## Evidence status\n\n- FW-LNSA uses a locked train/validation/test protocol.\n- Configuration and threshold selection use training and validation data only.\n- Confirmatory seeds are 100 through 119.\n- Classical baselines reuse the exact partition and feature-selection hashes.\n- All final audit checks passed before this package was created.\n- Raw datasets are intentionally excluded.\n\n## Author metadata\n\n{authors}\n\nAny field marked `TO BE CONFIRMED` must be finalized by Sarosh and Dr. Suraiya before submission, not inferred by the writer.\n""",
        "01_PROJECT_CONTEXT.md": """# Project context\n\nThe project evaluates a feature-weighted lightweight negative selection algorithm for anomaly-based network intrusion detection on NSL-KDD and CICIDS2017. The framework selects compact feature sets using mutual information, transforms them into a binary median-split representation, generates negative-selection detectors, and compares ordinary Hamming matching with training-derived feature-weighted similarity.\n\nThe final design emphasizes lightweight detector representations, target-FPR validation calibration, unseen-seed confirmatory evaluation, cross-dataset behavior, and complete reproducibility evidence.\n""",
        "02_RESEARCH_QUESTIONS.md": """# Research questions\n\n1. Does training-derived feature weighting improve confirmatory detection performance relative to unweighted Hamming matching?\n2. How does the effect depend on feature-set size, dataset, and target false-positive operating point?\n3. How reliably do validation-calibrated target FPRs transfer to untouched test partitions?\n4. How does FW-LNSA compare with Logistic Regression, Decision Tree, Random Forest, and Isolation Forest under identical partitions and selected features?\n5. What tradeoffs arise among detection quality, detector retention, runtime, throughput, memory, and model size?\n6. Which attack categories remain difficult under strict false-positive constraints?\n""",
        "03_CONTRIBUTIONS_AND_NOVELTY.md": """# Contributions and defensible novelty\n\nThe paper should frame novelty as the integrated protocol and evidence, not claim that feature selection, negative selection, or weighted matching is individually unprecedented.\n\nDefensible contributions:\n\n1. A lightweight negative-selection framework that directly embeds training-derived mutual-information weights in detector matching.\n2. A controlled ablation between Hamming and weighted similarity under the same feature-selection and data-partition contract.\n3. Validation-only calibration at explicit target-FPR operating points followed by untouched-test evaluation.\n4. An unseen-seed confirmatory design that separates development seeds from final evaluation seeds.\n5. Hash-verified, partition-compatible comparisons with four classical baselines.\n6. Cross-dataset evidence covering predictive quality, calibration transfer, attack-category behavior, detector retention, throughput, memory, and model footprint.\n7. A resumable and auditable open implementation with configuration locks, manifests, dataset fingerprints, and publication-ready evidence.\n\nAvoid absolute claims such as “the first weighted NSA” unless a systematic literature search directly supports them.\n""",
        "04_METHODS_CONTRACT.md": """# Methods contract\n\n- Training data: feature selection, medians, detector generation, and model fitting.\n- Validation data: target-FPR threshold calibration and FW-LNSA configuration selection.\n- Test data: one locked confirmatory evaluation per unseen seed and operating point.\n- Tuning seeds: 42–46.\n- Confirmatory seeds: 100–119.\n- Feature sets: FS-10 and FS-20.\n- Target FPRs: 0.10, 0.05, and 0.01.\n- FW-LNSA methods: Hamming and feature-weighted similarity.\n- Baselines: Logistic Regression, Decision Tree, Random Forest, and Isolation Forest.\n- Statistical comparison: paired Wilcoxon signed-rank tests, rank-biserial effects, bootstrap confidence intervals, Holm correction, and win/tie/loss counts.\n\nAll exact settings and hashes are preserved in `reproducibility/`.\n""",
        "05_DATASET_AND_PREPROCESSING_DETAILS.md": """# Dataset and preprocessing details\n\nUse the execution manifests for exact record counts and hashes. NSL-KDD uses the official KDDTrain+ and KDDTest+ files. CICIDS2017 uses the official MachineLearningCSV archive, duplicate removal, non-finite replacement, global-cap sampling, and stratified train/validation/test splits.\n\nCategorical encoding and scaling are fitted on training data only. Mutual-information feature selection is fitted on training data only.\n\nNSL-KDD category evidence in this run is binary-only (`Attack (Unspecified)`) because the supplied test labels do not support defensible DoS, Probe, R2L, and U2R reporting. Do not infer those families.\n""",
        "06_RESULTS_INTERPRETATION.md": """# Results interpretation guide\n\nUse `confirmatory_performance.csv` for the main result table, `paired_weighted_vs_hamming.csv` for the central ablation, and `calibration_reliability_summary.csv` for target-FPR transfer.\n\nThe correct framing is conditional rather than universal. Feature weighting should be described by dataset, FS size, and operating point. Report unfavorable or null effects alongside favorable effects.\n\nUse adjusted p-values together with effect sizes and confidence intervals. Statistical significance alone is insufficient. For FPR, runtime, model size, and memory, lower values are favorable. For F1, recall, balanced accuracy, MCC, retention, and throughput, higher values are favorable.\n\nAttack-category results must be interpreted with support counts and evidence status. Rare or weakly supported categories belong in limitations.\n""",
        "07_SUPPORTED_CLAIMS.md": """# Supported claim boundaries\n\nClaims may be made when directly supported by the supplied tables:\n\n- The implementation followed a locked three-way protocol and did not use test metrics for selection.\n- Baselines and FW-LNSA share verified partitions and feature-selection hashes.\n- Feature weighting produced dataset-, representation-, and operating-point-dependent effects.\n- Validation-calibrated FPR did not always transfer perfectly to the test partition.\n- FW-LNSA has measurable detector-retention and lightweight-model characteristics.\n- CICIDS2017 contains attack categories that remain difficult under strict FPR constraints.\n- Results are reproducible from the exact commit, configurations, manifests, and dataset fingerprints.\n""",
        "08_PROHIBITED_OR_UNSUPPORTED_CLAIMS.md": """# Prohibited or unsupported claims\n\nDo not claim:\n\n- Perfect intrusion detection.\n- Universal superiority over all baselines or all datasets.\n- That feature weighting always improves every metric.\n- That target validation FPR guarantees the same test FPR.\n- Detailed NSL-KDD attack-family recall when the evidence is binary-only.\n- Real-time production readiness based only on offline benchmark timing.\n- Generalization to encrypted traffic, zero-day attacks, live networks, or datasets not evaluated.\n- That any single component is the first of its kind without a completed systematic literature review.\n""",
        "09_LIMITATIONS_AND_THREATS_TO_VALIDITY.md": """# Limitations and threats to validity\n\n- NSL-KDD is an older benchmark and has limited realism for current traffic.\n- CICIDS2017 is more modern but remains a controlled laboratory dataset.\n- CICIDS2017 sampling caps computational cost and may alter very rare-class prevalence.\n- Binary median-split representation sacrifices information for compactness.\n- Validation-to-test calibration can shift under distribution differences.\n- Some discrete detector-score distributions make target-FPR calibration coarse.\n- NSL-KDD category resolution is binary-only in the supplied test file.\n- Confirmatory results cover 20 unseen seeds, not every possible random initialization.\n- Classical baseline hyperparameters are fixed, not exhaustively optimized. This avoids test leakage but may not represent each baseline’s absolute maximum potential.\n- Offline runtime does not by itself establish live deployment performance.\n""",
        "10_RELATED_WORK_MATRIX.md": """# Related-work matrix instructions\n\nThe included `10_RELATED_WORK_MATRIX.csv` is a structured starter. Tanazzah should update it during the literature review without changing the verified experimental claims. Each paper should be classified by dataset, negative-selection variant, feature-selection method, weighting mechanism, threshold calibration, repeated seeds, statistical testing, efficiency evidence, and reproducibility artifacts.\n\nThe novelty statement must be revised against the completed matrix before submission.\n""",
        "11_TABLE_AND_FIGURE_MAP.md": """# Table and figure map\n\nSuggested main-text tables:\n\n1. Dataset and protocol summary.\n2. Confirmatory FW-LNSA performance by dataset, feature set, and target FPR.\n3. Paired Hamming versus weighted-similarity analysis.\n4. Compatible baseline comparison.\n5. Validation-to-test FPR calibration reliability.\n6. Efficiency and model-footprint comparison.\n\nSuggested main-text figures:\n\n1. FW-LNSA architecture and three-way evaluation protocol.\n2. F1 versus observed FPR.\n3. Paired weighted-versus-Hamming F1 differences.\n4. Validation FPR versus test FPR.\n5. Model size versus prediction throughput.\n6. CICIDS2017 attack-category recall.\n\nUse the remaining detailed tables as supplementary material.\n""",
        "12_REPRODUCIBILITY_STATEMENT.md": """# Reproducibility statement\n\nThe repository preserves deterministic configurations, fixed development and confirmatory seed sets, dataset SHA-256 fingerprints, train/validation/test partition hashes, feature-selection hashes, configuration locks, row-level progress journals, execution manifests, environment versions, validation suites, and an executed final notebook. Raw benchmark datasets are excluded from the writer handoff and should be obtained from their official sources.\n""",
        "13_DATA_AND_CODE_AVAILABILITY.md": """# Data and code availability\n\nCode and configuration files are maintained in the private FW-LNSA-NIDS GitHub repository during manuscript preparation. The final paper should provide the release/tag or archived DOI selected by the authors. NSL-KDD and CICIDS2017 must be cited and obtained from their official distribution sources. No raw dataset is redistributed in the writer package.\n""",
        "14_AUTHORSHIP_AND_CREDIT_ROLES.md": f"""# Authorship and CRediT roles\n\n{authors}\n\nSuggested roles to confirm with all authors before submission:\n\n- Sarosh Jawed: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review and editing.\n- Tanazzah: Investigation, Literature review, Writing – original draft, Writing – review and editing.\n- Suraiya Akter: Supervision, Conceptualization, Methodology, Validation, Writing – review and editing, Project administration.\n\nThese are proposed roles and must be approved by the authors.\n""",
        "15_SUGGESTED_PAPER_STRUCTURE.md": """# Suggested paper structure\n\n1. Introduction\n2. Related Work\n3. Research Gap and Contributions\n4. Materials and Methods\n   - Datasets\n   - Leakage-safe preprocessing\n   - Feature selection and weighting\n   - FW-LNSA detector generation and matching\n   - Validation-calibrated operating points\n   - Confirmatory protocol and baselines\n   - Metrics and statistical analysis\n5. Results\n   - Main confirmatory results\n   - Hamming versus weighted matching\n   - Baseline comparisons\n   - Calibration reliability\n   - Efficiency and retention\n   - Category analysis\n6. Discussion\n7. Threats to Validity and Limitations\n8. Conclusion\n9. Data and Code Availability\n""",
    }
    for name, content in documents.items():
        (output / name).write_text(content, encoding="utf-8")

    related_work = pd.DataFrame(
        columns=[
            "citation_key", "year", "method", "datasets", "feature_selection",
            "feature_weighting", "negative_selection_variant", "target_fpr_calibration",
            "independent_validation", "unseen_seed_confirmation", "statistical_tests",
            "efficiency_metrics", "reproducible_code", "gap_relative_to_this_study",
        ]
    )
    related_work.to_csv(output / "10_RELATED_WORK_MATRIX.csv", index=False)

    references = """@inproceedings{forrest1994selfnonself,\n  title={Self-nonself discrimination in a computer},\n  author={Forrest, Stephanie and Perelson, Alan S. and Allen, Lawrence and Cherukuri, Rajesh},\n  booktitle={Proceedings of the 1994 IEEE Symposium on Research in Security and Privacy},\n  year={1994},\n  pages={202--212},\n  doi={10.1109/RISP.1994.296580}\n}\n\n@article{hofmeyr2000architecture,\n  title={Architecture for an artificial immune system},\n  author={Hofmeyr, Steven A. and Forrest, Stephanie},\n  journal={Evolutionary Computation},\n  volume={8},\n  number={4},\n  pages={443--473},\n  year={2000},\n  doi={10.1162/106365600568257}\n}\n\n@inproceedings{tavallaee2009detailed,\n  title={A detailed analysis of the KDD CUP 99 data set},\n  author={Tavallaee, Mahbod and Bagheri, Ebrahim and Lu, Wei and Ghorbani, Ali A.},\n  booktitle={2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications},\n  year={2009}\n}\n\n@inproceedings{sharafaldin2018toward,\n  title={Toward generating a new intrusion detection dataset and intrusion traffic characterization},\n  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A.},\n  booktitle={Proceedings of the 4th International Conference on Information Systems Security and Privacy},\n  pages={108--116},\n  year={2018},\n  doi={10.5220/0006639801080116}\n}\n\n@article{pedregosa2011scikit,\n  title={Scikit-learn: Machine learning in Python},\n  author={Pedregosa, Fabian and others},\n  journal={Journal of Machine Learning Research},\n  volume={12},\n  pages={2825--2830},\n  year={2011}\n}\n\n@article{holm1979simple,\n  title={A simple sequentially rejective multiple test procedure},\n  author={Holm, Sture},\n  journal={Scandinavian Journal of Statistics},\n  volume={6},\n  number={2},\n  pages={65--70},\n  year={1979}\n}\n\n@article{wilcoxon1945individual,\n  title={Individual comparisons by ranking methods},\n  author={Wilcoxon, Frank},\n  journal={Biometrics Bulletin},\n  volume={1},\n  number={6},\n  pages={80--83},\n  year={1945}\n}\n"""
    (output / "references.bib").write_text(references, encoding="utf-8")

    package_manifest = {
        "package_version": "1.0",
        "analysis_manifest": manifest,
        "contains_raw_datasets": False,
        "intended_recipient": "Tanazzah and the manuscript authors",
        "files": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
    }
    atomic_write_json(package_manifest, output / "PACKAGE_MANIFEST.json")
    return output
