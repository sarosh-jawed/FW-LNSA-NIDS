"""Reproducible experiment runners and result-table generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .evaluation import attack_category_analysis, evaluate_binary_classification
from .feature_selection import FeatureSelectionResult, mutual_information_feature_selection
from .fw_lnsa import FWLNSA
from .preprocessing import PreparedDataset, prepare_cicids2017, prepare_nsl_kdd
from .representation import BinaryRepresentationResult, binary_median_split
from .utils import ensure_dir, save_dataframe

ProgressCallback = Callable[[int, int, dict[str, Any]], None]

METHOD_ROLES = {
    "hamming": "classical_baseline",
    "weighted_hamming": "feature_weight_ablation",
    "weighted_smc": "main_operational_variant",
    "jaccard": "optional_diagnostic",
}


@dataclass(frozen=True)
class PreparedFeatureSet:
    """Selected features, weights, and binary train/test matrices."""

    fs_size: int
    selection: FeatureSelectionResult
    representation: BinaryRepresentationResult


@dataclass(frozen=True)
class ExperimentOutputs:
    """In-memory tables and saved output paths from one runner call."""

    results: pd.DataFrame
    method_summary: pd.DataFrame
    balanced_configs: pd.DataFrame
    attack_category_analysis: pd.DataFrame
    selected_features: pd.DataFrame
    data_quality_report: pd.DataFrame
    output_paths: dict[str, Path]


def _display_name(method: str) -> str:
    return {
        "hamming": "Hamming",
        "weighted_hamming": "Weighted Hamming",
        "weighted_smc": "Weighted SMC",
        "jaccard": "Jaccard",
    }[method]


def _threshold_value(method: str, configured: float, fs_size: int, scale: str) -> float:
    if method == "hamming":
        if scale != "ratio":
            raise ValueError("Hamming thresholds must be configured as ratios.")
        return float(int(math.ceil(float(configured) * fs_size)))
    if scale != "direct":
        raise ValueError(f"{method} thresholds must use direct values.")
    return float(configured)


def _feature_selection_sample(
    dataset: PreparedDataset,
    *,
    max_samples: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a deterministic class-aware sample for faster exploratory profiles."""

    if max_samples is None or len(dataset.X_train) <= max_samples:
        return dataset.X_train, dataset.y_train

    rng = np.random.default_rng(random_seed)
    selected_parts: list[np.ndarray] = []
    for label in (0, 1):
        label_indices = np.flatnonzero(dataset.y_train == label)
        target = max(1, round(max_samples * len(label_indices) / len(dataset.y_train)))
        target = min(target, len(label_indices))
        selected_parts.append(rng.choice(label_indices, size=target, replace=False))

    selected = np.concatenate(selected_parts)
    if len(selected) > max_samples:
        selected = rng.choice(selected, size=max_samples, replace=False)
    rng.shuffle(selected)
    return dataset.X_train.iloc[selected].reset_index(drop=True), dataset.y_train[selected]


def prepare_feature_sets(
    dataset: PreparedDataset,
    feature_sizes: list[int],
    *,
    random_seed: int,
    max_selection_samples: int | None = None,
) -> dict[int, PreparedFeatureSet]:
    """Fit feature selection and binary representation once per feature size."""

    selection_X, selection_y = _feature_selection_sample(
        dataset,
        max_samples=max_selection_samples,
        random_seed=random_seed,
    )

    prepared: dict[int, PreparedFeatureSet] = {}
    for fs_size in feature_sizes:
        selection = mutual_information_feature_selection(
            selection_X,
            selection_y,
            fs_size=fs_size,
            random_seed=random_seed,
        )
        representation = binary_median_split(
            dataset.X_train,
            dataset.X_test,
            selection.selected_features,
        )
        prepared[fs_size] = PreparedFeatureSet(
            fs_size=fs_size,
            selection=selection,
            representation=representation,
        )
    return prepared


def _estimate_total_runs(config: Mapping[str, Any]) -> int:
    feature_sizes = config["feature_selection"]["feature_sizes"]
    experiment = config["experiment"]
    method_settings = config["method_settings"]

    total = 0
    for _fs_size in feature_sizes:
        for method in experiment["methods"]:
            settings = method_settings[method]
            total += (
                len(experiment["seeds"])
                * len(experiment["detector_budgets"])
                * len(settings["self_thresholds"])
                * len(settings["detection_thresholds"])
            )
    return total


def _run_one(
    prepared: PreparedFeatureSet,
    dataset: PreparedDataset,
    *,
    method: str,
    seed: int,
    detector_budget: int,
    self_threshold_config: float,
    detection_threshold_config: float,
    threshold_scale: str,
    max_self_samples: int | None,
    deduplicate_candidates: bool,
    prediction_chunk_size: int,
    profile_name: str,
) -> tuple[dict[str, Any], FWLNSA, np.ndarray]:
    fs_size = prepared.fs_size
    self_threshold = _threshold_value(
        method,
        self_threshold_config,
        fs_size,
        threshold_scale,
    )
    detection_threshold = _threshold_value(
        method,
        detection_threshold_config,
        fs_size,
        threshold_scale,
    )

    model = FWLNSA(
        method=method,
        n_detectors=detector_budget,
        self_threshold=self_threshold,
        detection_threshold=detection_threshold,
        random_seed=seed,
        max_self_samples=max_self_samples,
        deduplicate_candidates=deduplicate_candidates,
        prediction_chunk_size=prediction_chunk_size,
    )
    model.fit(
        prepared.representation.X_train_bin,
        dataset.y_train,
        feature_weights=prepared.selection.weights,
    )
    predictions = model.predict(prepared.representation.X_test_bin)
    metrics = evaluate_binary_classification(dataset.y_test, predictions)
    detector_summary = model.get_detector_summary()

    row = {
        "profile": profile_name,
        "dataset": str(dataset.metadata.get("dataset", "unknown")),
        "method": method,
        "method_name": _display_name(method),
        "method_role": METHOD_ROLES[method],
        "feature_set": f"FS-{fs_size}",
        "fs_size": fs_size,
        "seed": seed,
        "detector_budget": detector_budget,
        "self_threshold_config": float(self_threshold_config),
        "detection_threshold_config": float(detection_threshold_config),
        "threshold_scale": threshold_scale,
        **detector_summary,
        **metrics.to_dict(),
    }
    return row, model, predictions


def run_experiment_grid(
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    *,
    max_runs: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[int, PreparedFeatureSet]]:
    """Run the configured method, threshold, budget, seed, and feature grid."""

    feature_config = config["feature_selection"]
    experiment = config["experiment"]
    preprocessing = config["preprocessing"]
    method_settings = config["method_settings"]

    prepared_sets = prepare_feature_sets(
        dataset,
        list(feature_config["feature_sizes"]),
        random_seed=int(feature_config.get("random_seed", 42)),
        max_selection_samples=feature_config.get("max_samples"),
    )

    estimated_total = _estimate_total_runs(config)
    run_limit = estimated_total if max_runs is None else min(max_runs, estimated_total)
    rows: list[dict[str, Any]] = []

    for fs_size in feature_config["feature_sizes"]:
        prepared = prepared_sets[int(fs_size)]
        for method in experiment["methods"]:
            settings = method_settings[method]
            for seed in experiment["seeds"]:
                for detector_budget in experiment["detector_budgets"]:
                    for self_value in settings["self_thresholds"]:
                        for detection_value in settings["detection_thresholds"]:
                            if len(rows) >= run_limit:
                                return pd.DataFrame(rows), prepared_sets

                            row, _model, _predictions = _run_one(
                                prepared,
                                dataset,
                                method=method,
                                seed=int(seed),
                                detector_budget=int(detector_budget),
                                self_threshold_config=float(self_value),
                                detection_threshold_config=float(detection_value),
                                threshold_scale=str(settings["threshold_scale"]),
                                max_self_samples=preprocessing.get("max_self_samples"),
                                deduplicate_candidates=bool(
                                    experiment.get("deduplicate_candidates", False)
                                ),
                                prediction_chunk_size=int(
                                    experiment.get("prediction_chunk_size", 250)
                                ),
                                profile_name=str(config.get("profile_name", "unknown")),
                            )
                            rows.append(row)
                            if progress_callback is not None:
                                progress_callback(len(rows), run_limit, row)

    return pd.DataFrame(rows), prepared_sets


def summarize_methods(results: pd.DataFrame) -> pd.DataFrame:
    """Create a method and feature-set summary without mixing run settings."""

    if results.empty:
        return pd.DataFrame()

    summary = (
        results.groupby(
            ["method", "method_name", "method_role", "feature_set", "fs_size"],
            as_index=False,
        )
        .agg(
            runs=("method", "size"),
            best_accuracy=("accuracy", "max"),
            best_precision=("precision", "max"),
            best_recall=("recall", "max"),
            best_f1=("f1", "max"),
            lowest_fpr=("fpr", "min"),
            average_f1=("f1", "mean"),
            average_fpr=("fpr", "mean"),
            average_retained_detectors=("retained_detectors", "mean"),
            average_generation_time_sec=("generation_time_sec", "mean"),
            average_detection_time_sec=("detection_time_sec", "mean"),
            average_total_time_sec=("total_time_sec", "mean"),
        )
        .sort_values(["fs_size", "best_f1"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


def select_balanced_configs(
    results: pd.DataFrame,
    *,
    fpr_limit: float,
    main_feature_size: int,
) -> pd.DataFrame:
    """Select one controlled-FPR configuration for each matching method."""

    if results.empty:
        return pd.DataFrame()

    main_results = results[results["fs_size"] == main_feature_size].copy()
    selected_rows: list[pd.Series] = []

    for _method, group in main_results.groupby("method", sort=False):
        controlled = group[group["fpr"] <= fpr_limit].copy()
        if controlled.empty:
            candidates = group.copy()
            selection_rule = f"Fallback best F1 because no run met FPR <= {fpr_limit:.2f}"
        else:
            candidates = controlled
            selection_rule = f"Best balanced configuration with FPR <= {fpr_limit:.2f}"

        ordered = candidates.sort_values(
            ["f1", "recall", "fpr", "total_time_sec", "retained_detectors"],
            ascending=[False, False, True, True, True],
        )
        selected = ordered.iloc[0].copy()
        selected["selection_rule"] = selection_rule
        selected_rows.append(selected)

    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_selected_feature_table(
    prepared_sets: Mapping[int, PreparedFeatureSet],
) -> pd.DataFrame:
    """Create a publication-support table of selected features and weights."""

    rows: list[pd.DataFrame] = []
    for fs_size, prepared in prepared_sets.items():
        table = prepared.selection.selected_scores.copy()
        table.insert(0, "feature_set", f"FS-{fs_size}")
        table.insert(1, "fs_size", fs_size)
        table.insert(2, "rank", np.arange(1, len(table) + 1))
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def analyze_balanced_attack_categories(
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    balanced_configs: pd.DataFrame,
    prepared_sets: Mapping[int, PreparedFeatureSet],
) -> pd.DataFrame:
    """Rebuild selected models and report category behavior for each method."""

    if balanced_configs.empty or dataset.test_attack_categories is None:
        return pd.DataFrame()

    preprocessing = config["preprocessing"]
    experiment = config["experiment"]
    rows: list[pd.DataFrame] = []

    for _, selected in balanced_configs.iterrows():
        fs_size = int(selected["fs_size"])
        prepared = prepared_sets[fs_size]
        method = str(selected["method"])

        model = FWLNSA(
            method=method,
            n_detectors=int(selected["detector_budget"]),
            self_threshold=float(selected["self_threshold"]),
            detection_threshold=float(selected["detection_threshold"]),
            random_seed=int(selected["seed"]),
            max_self_samples=preprocessing.get("max_self_samples"),
            deduplicate_candidates=bool(experiment.get("deduplicate_candidates", False)),
            prediction_chunk_size=int(experiment.get("prediction_chunk_size", 250)),
        )
        model.fit(
            prepared.representation.X_train_bin,
            dataset.y_train,
            feature_weights=prepared.selection.weights,
        )
        predictions = model.predict(prepared.representation.X_test_bin)

        metadata = {
            "method": method,
            "method_name": _display_name(method),
            "method_role": METHOD_ROLES[method],
            "selection_type": "balanced_main_feature_set",
            "feature_set": f"FS-{fs_size}",
            "seed": int(selected["seed"]),
            "detector_budget": int(selected["detector_budget"]),
            "self_threshold": float(selected["self_threshold"]),
            "detection_threshold": float(selected["detection_threshold"]),
        }
        rows.append(
            attack_category_analysis(
                dataset.y_test,
                predictions,
                dataset.test_attack_categories,
                run_metadata=metadata,
            )
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_experiment_outputs(
    *,
    results: pd.DataFrame,
    method_summary: pd.DataFrame,
    balanced_configs: pd.DataFrame,
    category_analysis: pd.DataFrame,
    selected_features: pd.DataFrame,
    data_quality_report: pd.DataFrame | None,
    config: Mapping[str, Any],
) -> dict[str, Path]:
    """Save configured FW-LNSA tables for either supported dataset."""

    output_dir = ensure_dir(config["paths"]["output_dir"])
    tables_dir = ensure_dir(output_dir / "tables")
    output_names = config["outputs"]

    paths = {
        "results_table": save_dataframe(
            results,
            tables_dir / output_names["results_table"],
        ),
        "method_summary": save_dataframe(
            method_summary,
            tables_dir / output_names["method_summary"],
        ),
        "balanced_configs": save_dataframe(
            balanced_configs,
            tables_dir / output_names["balanced_configs"],
        ),
        "attack_category_analysis": save_dataframe(
            category_analysis,
            tables_dir / output_names["attack_category_analysis"],
        ),
        "selected_features": save_dataframe(
            selected_features,
            tables_dir / output_names["selected_features"],
        ),
    }
    quality_name = output_names.get("data_quality_report")
    if quality_name and data_quality_report is not None:
        paths["data_quality_report"] = save_dataframe(
            data_quality_report,
            tables_dir / quality_name,
        )
    return paths


def run_prepared_fw_lnsa_experiments(
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    *,
    max_runs: int | None = None,
    progress_callback: ProgressCallback | None = None,
    save_outputs: bool = True,
) -> ExperimentOutputs:
    """Run a complete FW-LNSA experiment package from a prepared dataset."""

    results, prepared_sets = run_experiment_grid(
        dataset,
        config,
        max_runs=max_runs,
        progress_callback=progress_callback,
    )
    method_summary = summarize_methods(results)
    balanced_configs = select_balanced_configs(
        results,
        fpr_limit=float(config["selection"]["balanced_fpr_limit"]),
        main_feature_size=int(config["feature_selection"]["main_feature_size"]),
    )
    category_analysis = analyze_balanced_attack_categories(
        dataset,
        config,
        balanced_configs,
        prepared_sets,
    )
    selected_features = build_selected_feature_table(prepared_sets)

    output_paths: dict[str, Path] = {}
    if save_outputs:
        output_paths = save_experiment_outputs(
            results=results,
            method_summary=method_summary,
            balanced_configs=balanced_configs,
            category_analysis=category_analysis,
            selected_features=selected_features,
            data_quality_report=dataset.data_quality_report,
            config=config,
        )

    return ExperimentOutputs(
        results=results,
        method_summary=method_summary,
        balanced_configs=balanced_configs,
        attack_category_analysis=category_analysis,
        selected_features=selected_features,
        data_quality_report=(
            dataset.data_quality_report.copy()
            if dataset.data_quality_report is not None
            else pd.DataFrame()
        ),
        output_paths=output_paths,
    )



def run_prepared_nsl_kdd_experiments(
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    *,
    max_runs: int | None = None,
    progress_callback: ProgressCallback | None = None,
    save_outputs: bool = True,
) -> ExperimentOutputs:
    """Backward-compatible NSL-KDD wrapper for the generic prepared runner."""

    return run_prepared_fw_lnsa_experiments(
        dataset,
        config,
        max_runs=max_runs,
        progress_callback=progress_callback,
        save_outputs=save_outputs,
    )


def run_nsl_kdd_experiments(
    config: Mapping[str, Any],
    *,
    max_runs: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ExperimentOutputs:
    """Load NSL-KDD, run experiments, and save all required tables."""

    paths = config["paths"]
    dataset = prepare_nsl_kdd(
        paths["train_file"],
        paths["test_file"],
        scale=bool(config["preprocessing"].get("scale", True)),
    )
    return run_prepared_fw_lnsa_experiments(
        dataset,
        config,
        max_runs=max_runs,
        progress_callback=progress_callback,
        save_outputs=True,
    )



def run_cicids2017_experiments(
    config: Mapping[str, Any],
    *,
    max_runs: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ExperimentOutputs:
    """Load CICIDS2017, run FW-LNSA experiments, and save all tables."""

    paths = config["paths"]
    preprocessing = config["preprocessing"]
    sampling = preprocessing["sampling"]
    split = preprocessing["split"]

    dataset = prepare_cicids2017(
        paths["raw_dir"],
        archive_file=paths.get("archive_file"),
        label_column=str(preprocessing.get("label_column", "Label")),
        benign_label=str(preprocessing.get("benign_label", "BENIGN")),
        chunk_size=int(preprocessing.get("chunk_size", 25_000)),
        max_chunks_per_file=preprocessing.get("max_chunks_per_file"),
        drop_duplicates=bool(preprocessing.get("drop_duplicates", True)),
        sampling_strategy=str(sampling.get("strategy", "per_label_cap")),
        max_benign_records=int(sampling.get("max_benign_records", 1)),
        max_records_per_attack_label=int(sampling.get("max_records_per_attack_label", 1)),
        max_total_records=sampling.get("max_total_records"),
        sampling_seed=int(sampling.get("random_seed", 42)),
        test_size=float(split.get("test_size", 0.30)),
        split_seed=int(split.get("random_seed", 42)),
        stratify_by=str(split.get("stratify_by", "original_label")),
        scale=bool(preprocessing.get("scale", True)),
    )
    return run_prepared_fw_lnsa_experiments(
        dataset,
        config,
        max_runs=max_runs,
        progress_callback=progress_callback,
        save_outputs=True,
    )
