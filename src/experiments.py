"""Reproducible experiment runners and result-table generation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .evaluation import attack_category_analysis, evaluate_binary_classification
from .feature_selection import (
    FeatureSelectionResult,
    compute_mutual_information_scores,
    select_top_features,
)
from .fw_lnsa import FWLNSA
from .preprocessing import PreparedDataset, prepare_cicids2017, prepare_nsl_kdd
from .representation import BinaryRepresentationResult, binary_median_split
from .utils import (
    ensure_dir,
    save_dataframe,
    stable_feature_selection_signature,
    stable_partition_signature,
)

ProgressCallback = Callable[[int, int, dict[str, Any]], None]
ResultCallback = Callable[[dict[str, Any]], None]

FW_RUN_KEY_COLUMNS = (
    "profile",
    "method",
    "fs_size",
    "seed",
    "detector_budget",
    "self_threshold_config",
    "detection_threshold_config",
)

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

    # Mutual-information scores are independent of the requested feature-set
    # size. Compute them once, then derive nested FS-10 and FS-20 selections
    # from the same ranked table. This removes duplicate fitting and guarantees
    # that feature-set ablations differ only in representation size.
    score_table = compute_mutual_information_scores(
        selection_X,
        selection_y,
        random_seed=random_seed,
    )

    prepared: dict[int, PreparedFeatureSet] = {}
    for fs_size in feature_sizes:
        selection = select_top_features(score_table, fs_size=fs_size)
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


def _normalize_key_value(value: Any) -> Any:
    """Normalize values used in deterministic run identity keys."""

    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 12)
    return str(value)


def fw_run_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the stable identity of one FW-LNSA experiment row."""

    return tuple(_normalize_key_value(row[column]) for column in FW_RUN_KEY_COLUMNS)


def iter_fw_run_specs(config: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield configured runs in the exact deterministic execution order."""

    feature_config = config["feature_selection"]
    experiment = config["experiment"]
    method_settings = config["method_settings"]
    profile_name = str(config.get("profile_name", "unknown"))

    for fs_size in feature_config["feature_sizes"]:
        for method in experiment["methods"]:
            settings = method_settings[method]
            for seed in experiment["seeds"]:
                for detector_budget in experiment["detector_budgets"]:
                    for self_value in settings["self_thresholds"]:
                        for detection_value in settings["detection_thresholds"]:
                            yield {
                                "profile": profile_name,
                                "method": str(method),
                                "fs_size": int(fs_size),
                                "seed": int(seed),
                                "detector_budget": int(detector_budget),
                                "self_threshold_config": float(self_value),
                                "detection_threshold_config": float(detection_value),
                                "threshold_scale": str(settings["threshold_scale"]),
                            }


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
    train_partition_hash: str,
    test_partition_hash: str,
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

    feature_selection_hash = stable_feature_selection_signature(
        prepared.selection.selected_features,
        prepared.selection.selected_scores["mi_score"].to_numpy(dtype=float),
        prepared.selection.weights,
    )

    row = {
        "profile": profile_name,
        "dataset": str(dataset.metadata.get("dataset", "unknown")),
        "method": method,
        "method_name": _display_name(method),
        "method_role": METHOD_ROLES[method],
        "feature_set": f"FS-{fs_size}",
        "fs_size": fs_size,
        "selected_features": "|".join(prepared.selection.selected_features),
        "feature_selection_hash": feature_selection_hash,
        "seed": seed,
        "detector_budget": detector_budget,
        "self_threshold_config": float(self_threshold_config),
        "detection_threshold_config": float(detection_threshold_config),
        "threshold_scale": threshold_scale,
        "train_records": int(len(dataset.X_train)),
        "test_records": int(len(dataset.X_test)),
        "train_partition_hash": train_partition_hash,
        "test_partition_hash": test_partition_hash,
        **detector_summary,
        **metrics.to_dict(),
    }
    return row, model, predictions


def _run_detection_threshold_group(
    prepared: PreparedFeatureSet,
    dataset: PreparedDataset,
    *,
    method: str,
    seed: int,
    detector_budget: int,
    self_threshold_config: float,
    detection_threshold_configs: list[float],
    threshold_scale: str,
    max_self_samples: int | None,
    deduplicate_candidates: bool,
    prediction_chunk_size: int,
    profile_name: str,
    train_partition_hash: str,
    test_partition_hash: str,
) -> list[dict[str, Any]]:
    """Fit one detector pool and evaluate every detection threshold on its scores.

    Detector generation and nearest-detector scoring do not depend on the final
    detection threshold. Reusing them avoids repeating identical computation
    while preserving one result row for every configured threshold.
    """

    fs_size = prepared.fs_size
    self_threshold = _threshold_value(
        method,
        self_threshold_config,
        fs_size,
        threshold_scale,
    )
    configured_thresholds = [float(value) for value in detection_threshold_configs]
    actual_thresholds = [
        _threshold_value(method, value, fs_size, threshold_scale)
        for value in configured_thresholds
    ]

    model = FWLNSA(
        method=method,
        n_detectors=detector_budget,
        self_threshold=self_threshold,
        detection_threshold=actual_thresholds[0],
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

    score_started = time.perf_counter()
    scores = model.decision_scores(prepared.representation.X_test_bin)
    detection_time = float(time.perf_counter() - score_started)
    fit_summary = model.get_detector_summary()
    generation_time = float(fit_summary["generation_time_sec"])

    feature_selection_hash = stable_feature_selection_signature(
        prepared.selection.selected_features,
        prepared.selection.selected_scores["mi_score"].to_numpy(dtype=float),
        prepared.selection.weights,
    )
    pool_key = (
        f"{profile_name}|{method}|FS-{fs_size}|seed={seed}|budget={detector_budget}|"
        f"self={float(self_threshold_config):.12g}"
    )

    rows: list[dict[str, Any]] = []
    distance_method = method in {"hamming", "weighted_hamming"}
    for configured, actual in zip(configured_thresholds, actual_thresholds):
        if distance_method:
            predictions = (scores <= actual).astype(np.int8)
        else:
            predictions = (scores >= actual).astype(np.int8)
        metrics = evaluate_binary_classification(dataset.y_test, predictions)
        detector_summary = dict(fit_summary)
        detector_summary["detection_threshold"] = float(actual)
        detector_summary["detection_time_sec"] = detection_time
        detector_summary["total_time_sec"] = generation_time + detection_time

        rows.append(
            {
                "profile": profile_name,
                "dataset": str(dataset.metadata.get("dataset", "unknown")),
                "method": method,
                "method_name": _display_name(method),
                "method_role": METHOD_ROLES[method],
                "feature_set": f"FS-{fs_size}",
                "fs_size": fs_size,
                "selected_features": "|".join(prepared.selection.selected_features),
                "feature_selection_hash": feature_selection_hash,
                "seed": seed,
                "detector_budget": detector_budget,
                "self_threshold_config": float(self_threshold_config),
                "detection_threshold_config": float(configured),
                "threshold_scale": threshold_scale,
                "detector_pool_key": pool_key,
                "shared_score_evaluation": True,
                "train_records": int(len(dataset.X_train)),
                "test_records": int(len(dataset.X_test)),
                "train_partition_hash": train_partition_hash,
                "test_partition_hash": test_partition_hash,
                **detector_summary,
                **metrics.to_dict(),
            }
        )
    return rows


def run_experiment_grid(
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    *,
    max_runs: int | None = None,
    progress_callback: ProgressCallback | None = None,
    existing_results: pd.DataFrame | None = None,
    result_callback: ResultCallback | None = None,
) -> tuple[pd.DataFrame, dict[int, PreparedFeatureSet]]:
    """Run the configured grid with safe row-level resume support."""

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

    train_partition_hash = stable_partition_signature(
        dataset.X_train, dataset.y_train, dataset.train_original_labels
    )
    test_partition_hash = stable_partition_signature(
        dataset.X_test, dataset.y_test, dataset.test_original_labels
    )

    all_specs = list(iter_fw_run_specs(config))
    run_limit = len(all_specs) if max_runs is None else min(int(max_runs), len(all_specs))
    selected_specs = all_specs[:run_limit]
    allowed_keys = {fw_run_key(spec) for spec in selected_specs}

    rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[Any, ...]] = set()
    if existing_results is not None and not existing_results.empty:
        missing_columns = set(FW_RUN_KEY_COLUMNS) - set(existing_results.columns)
        if missing_columns:
            raise ValueError(
                "Existing FW-LNSA results cannot be resumed because run identity "
                f"columns are missing: {sorted(missing_columns)}"
            )
        for existing_row in existing_results.to_dict(orient="records"):
            key = fw_run_key(existing_row)
            if key in allowed_keys and key not in completed_keys:
                rows.append(dict(existing_row))
                completed_keys.add(key)

    if progress_callback is not None and rows:
        progress_callback(len(rows), run_limit, rows[-1])

    grouped_specs: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for spec in selected_specs:
        group_key = (
            spec["profile"],
            spec["method"],
            spec["fs_size"],
            spec["seed"],
            spec["detector_budget"],
            _normalize_key_value(spec["self_threshold_config"]),
            spec["threshold_scale"],
        )
        grouped_specs.setdefault(group_key, []).append(spec)

    for group in grouped_specs.values():
        missing_specs = [spec for spec in group if fw_run_key(spec) not in completed_keys]
        if not missing_specs:
            continue

        first = missing_specs[0]
        generated_rows = _run_detection_threshold_group(
            prepared_sets[int(first["fs_size"])],
            dataset,
            method=str(first["method"]),
            seed=int(first["seed"]),
            detector_budget=int(first["detector_budget"]),
            self_threshold_config=float(first["self_threshold_config"]),
            detection_threshold_configs=[
                float(spec["detection_threshold_config"]) for spec in missing_specs
            ],
            threshold_scale=str(first["threshold_scale"]),
            max_self_samples=preprocessing.get("max_self_samples"),
            deduplicate_candidates=bool(experiment.get("deduplicate_candidates", False)),
            prediction_chunk_size=int(experiment.get("prediction_chunk_size", 250)),
            profile_name=str(config.get("profile_name", "unknown")),
            train_partition_hash=train_partition_hash,
            test_partition_hash=test_partition_hash,
        )
        for row in generated_rows:
            key = fw_run_key(row)
            rows.append(row)
            completed_keys.add(key)
            if result_callback is not None:
                result_callback(row)
            if progress_callback is not None:
                progress_callback(len(rows), run_limit, row)

    order = {fw_run_key(spec): index for index, spec in enumerate(selected_specs)}
    rows.sort(key=lambda row: order[fw_run_key(row)])
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
    existing_results: pd.DataFrame | None = None,
    result_callback: ResultCallback | None = None,
) -> ExperimentOutputs:
    """Run a complete FW-LNSA experiment package from a prepared dataset."""

    results, prepared_sets = run_experiment_grid(
        dataset,
        config,
        max_runs=max_runs,
        progress_callback=progress_callback,
        existing_results=existing_results,
        result_callback=result_callback,
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
    existing_results: pd.DataFrame | None = None,
    result_callback: ResultCallback | None = None,
) -> ExperimentOutputs:
    """Backward-compatible NSL-KDD wrapper for the generic prepared runner."""

    return run_prepared_fw_lnsa_experiments(
        dataset,
        config,
        max_runs=max_runs,
        progress_callback=progress_callback,
        save_outputs=save_outputs,
        existing_results=existing_results,
        result_callback=result_callback,
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
