"""Validation-calibrated confirmatory protocol for FW-LNSA.

This module separates model development from final evaluation. Detector and
feature training use only the training partition. Operating thresholds and
configuration choices use only validation data. The final test partition is
read only after a configuration has been locked.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .config import ConfigurationError, deep_merge, load_yaml_config
from .evaluation import attack_category_analysis, evaluate_binary_classification
from .feature_selection import compute_mutual_information_scores, select_top_features
from .fw_lnsa import DISTANCE_METHODS, FWLNSA
from .preprocessing import PreparedDataset, prepare_cicids2017, prepare_nsl_kdd
from .representation import BinaryRepresentationResult, binary_median_split
from .research_execution import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    environment_manifest,
    file_sha256,
    load_jsonl_frame,
)
from .utils import (
    ensure_dir,
    stable_feature_selection_signature,
    stable_mapping_signature,
    stable_partition_signature,
)


PROTOCOL_VERSION = "1.0"
CANONICAL_METHODS = {"hamming", "weighted_similarity"}
ProgressCallback = Callable[[str, int, int, Mapping[str, Any]], None]


@dataclass(frozen=True)
class CalibrationResult:
    """Validation-derived operating threshold and observed false-positive rate."""

    threshold: float
    target_fpr: float
    achieved_fpr: float
    benign_records: int
    false_alarms: int
    rule: str
    degenerate_scores: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_threshold": self.threshold,
            "target_fpr": self.target_fpr,
            "calibration_fpr": self.achieved_fpr,
            "calibration_benign_records": self.benign_records,
            "calibration_false_alarms": self.false_alarms,
            "calibration_rule": self.rule,
            "calibration_degenerate_scores": self.degenerate_scores,
        }


@dataclass(frozen=True)
class ConfirmatoryFeatureSet:
    """Training-fitted feature selection and binary representation."""

    fs_size: int
    selected_features: list[str]
    weights: np.ndarray
    score_table: pd.DataFrame
    representation: BinaryRepresentationResult
    feature_selection_hash: str


@dataclass(frozen=True)
class ConfirmatoryRunOutputs:
    """Saved tables and manifests produced for one dataset."""

    tuning_results: pd.DataFrame
    tuning_summary: pd.DataFrame
    locked_configurations: pd.DataFrame
    final_seed_results: pd.DataFrame
    final_summary: pd.DataFrame
    category_seed_results: pd.DataFrame
    category_summary: pd.DataFrame
    selected_features: pd.DataFrame
    manifest: dict[str, Any]
    output_paths: dict[str, Path]


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{key}' must be a mapping.")
    return value


def _require_list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"'{key}' must be a non-empty list.")
    return value


def validate_confirmatory_config(config: Mapping[str, Any]) -> None:
    """Validate the confirmatory protocol before data loading."""

    datasets = _require_mapping(config, "datasets")
    for dataset_name in ("nsl_kdd", "cicids2017"):
        if dataset_name not in datasets or not isinstance(datasets[dataset_name], Mapping):
            raise ConfigurationError(f"datasets.{dataset_name} is required.")

    protocol = _require_mapping(config, "protocol")
    validation_size = protocol.get("validation_size")
    if not isinstance(validation_size, (int, float)) or not 0.0 < float(validation_size) < 1.0:
        raise ConfigurationError("protocol.validation_size must be between 0 and 1.")

    experiment = _require_mapping(config, "experiment")
    methods = [str(value) for value in _require_list(experiment, "methods")]
    unknown = set(methods) - CANONICAL_METHODS
    if unknown:
        raise ConfigurationError(
            "Confirmatory methods must be hamming and weighted_similarity. "
            f"Unsupported values: {sorted(unknown)}"
        )
    feature_sizes = _require_list(experiment, "feature_sizes")
    budgets = _require_list(experiment, "detector_budgets")
    tuning_seeds = _require_list(experiment, "tuning_seeds")
    confirmatory_seeds = _require_list(experiment, "confirmatory_seeds")
    target_fprs = _require_list(experiment, "target_fprs")
    if any(not isinstance(value, int) or value <= 0 for value in feature_sizes):
        raise ConfigurationError("experiment.feature_sizes must contain positive integers.")
    if any(not isinstance(value, int) or value <= 0 for value in budgets):
        raise ConfigurationError("experiment.detector_budgets must contain positive integers.")
    if any(not isinstance(value, int) for value in tuning_seeds + confirmatory_seeds):
        raise ConfigurationError("All experiment seeds must be integers.")
    if set(tuning_seeds) & set(confirmatory_seeds):
        raise ConfigurationError("Tuning and confirmatory seeds must not overlap.")
    if any(not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0 for value in target_fprs):
        raise ConfigurationError("experiment.target_fprs must contain probabilities.")

    self_thresholds = _require_mapping(experiment, "self_thresholds")
    for method in methods:
        settings = self_thresholds.get(method)
        if not isinstance(settings, Mapping):
            raise ConfigurationError(f"experiment.self_thresholds.{method} is required.")
        values = settings.get("values")
        if not isinstance(values, list) or not values:
            raise ConfigurationError(
                f"experiment.self_thresholds.{method}.values must be non-empty."
            )
        scale = str(settings.get("scale", ""))
        expected = "ratio" if method == "hamming" else "direct"
        if scale != expected:
            raise ConfigurationError(
                f"experiment.self_thresholds.{method}.scale must be '{expected}'."
            )

    feature_selection = _require_mapping(config, "feature_selection")
    if str(feature_selection.get("method", "")) != "mutual_information":
        raise ConfigurationError("Only mutual_information feature selection is supported.")

    output_dir = _require_mapping(config, "paths").get("output_dir")
    if not output_dir:
        raise ConfigurationError("paths.output_dir is required.")


def resolve_confirmatory_profile(
    config: Mapping[str, Any],
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Apply one confirmatory execution profile and validate it."""

    base = {key: value for key, value in config.items() if key != "profiles"}
    profiles = config.get("profiles", {})
    selected_name = profile_name or str(base.get("active_profile", "smoke"))
    if not isinstance(profiles, Mapping) or selected_name not in profiles:
        available = ", ".join(sorted(map(str, profiles))) if isinstance(profiles, Mapping) else "none"
        raise ConfigurationError(
            f"Unknown confirmatory profile '{selected_name}'. Available profiles: {available}."
        )
    override = profiles[selected_name]
    if not isinstance(override, Mapping):
        raise ConfigurationError(f"Profile '{selected_name}' must be a mapping.")
    resolved = deep_merge(base, override)
    resolved["profile_name"] = selected_name
    resolved["protocol_version"] = str(resolved.get("protocol_version", PROTOCOL_VERSION))
    validate_confirmatory_config(resolved)
    return resolved


def load_confirmatory_config(
    path: str | Path,
    *,
    profile_name: str | None = None,
) -> dict[str, Any]:
    return resolve_confirmatory_profile(load_yaml_config(path), profile_name)


def _threshold_value(method: str, configured: float, fs_size: int, scale: str) -> float:
    if method == "hamming":
        if scale != "ratio":
            raise ValueError("Hamming self thresholds must use ratio scale.")
        return float(int(math.ceil(float(configured) * fs_size)))
    if scale != "direct":
        raise ValueError("Weighted similarity self thresholds must use direct scale.")
    return float(configured)


def _attack_scores(method: str, native_scores: np.ndarray) -> np.ndarray:
    values = np.asarray(native_scores, dtype=float)
    return -values if method in DISTANCE_METHODS else values


def predict_from_native_scores(
    scores: Sequence[float],
    *,
    method: str,
    threshold: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if method in DISTANCE_METHODS:
        return (values <= float(threshold)).astype(np.int8)
    return (values >= float(threshold)).astype(np.int8)


def calibrate_detection_threshold(
    validation_scores: Sequence[float],
    y_validation: Sequence[int],
    *,
    method: str,
    target_fpr: float,
) -> CalibrationResult:
    """Choose the least restrictive validation threshold meeting target FPR.

    Only benign validation scores determine the threshold. Attack validation
    records are used later to rank candidate configurations, never to set the
    false-positive operating point itself.
    """

    if not 0.0 <= float(target_fpr) <= 1.0:
        raise ValueError("target_fpr must be between 0 and 1.")
    scores = np.asarray(validation_scores, dtype=float)
    labels = np.asarray(y_validation, dtype=np.int8)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError("Validation scores and labels must be aligned vectors.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Validation labels must contain only 0 and 1.")

    benign = scores[labels == 0]
    if len(benign) == 0:
        raise ValueError("Threshold calibration requires benign validation records.")
    finite = benign[np.isfinite(benign)]
    degenerate = finite.size == 0 or np.unique(finite).size <= 1
    allowed_false_alarms = int(math.floor(float(target_fpr) * len(benign) + 1e-12))

    if method in DISTANCE_METHODS:
        if finite.size == 0:
            threshold = 0.0
        else:
            unique, counts = np.unique(finite, return_counts=True)
            cumulative = np.cumsum(counts)
            valid = unique[cumulative <= allowed_false_alarms]
            threshold = (
                float(valid[-1])
                if len(valid)
                else float(np.nextafter(unique[0], -np.inf))
            )
        rule = "largest distance threshold with empirical benign FPR at or below target"
    else:
        if finite.size == 0:
            threshold = 1.0
        else:
            unique, counts = np.unique(finite, return_counts=True)
            descending = unique[::-1]
            cumulative = np.cumsum(counts[::-1])
            valid = descending[cumulative <= allowed_false_alarms]
            threshold = (
                float(valid[-1])
                if len(valid)
                else float(np.nextafter(unique[-1], np.inf))
            )
        rule = "smallest similarity threshold with empirical benign FPR at or below target"

    benign_predictions = predict_from_native_scores(
        benign,
        method=method,
        threshold=threshold,
    )
    false_alarms = int(np.sum(benign_predictions == 1))
    achieved = float(false_alarms / len(benign))
    if achieved > float(target_fpr) + 1e-12:
        raise RuntimeError("Calibrated threshold exceeded its target false-positive rate.")

    return CalibrationResult(
        threshold=float(threshold),
        target_fpr=float(target_fpr),
        achieved_fpr=achieved,
        benign_records=int(len(benign)),
        false_alarms=false_alarms,
        rule=rule,
        degenerate_scores=bool(degenerate),
    )


def _mean_std_ci(values: Iterable[float], confidence: float) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(np.mean(array))
    if len(array) == 1:
        return {"count": 1, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(np.std(array, ddof=1))
    margin = float(stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1) * std / math.sqrt(len(array)))
    return {
        "count": int(len(array)),
        "mean": mean,
        "std": std,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def _aggregate_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
    *,
    confidence: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_key, group in frame.groupby(list(group_columns), dropna=False, sort=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_columns, key_values))
        for metric in metric_columns:
            summary = _mean_std_ci(group[metric].to_numpy(dtype=float), confidence)
            for suffix, value in summary.items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_confirmatory_feature_sets(
    dataset: PreparedDataset,
    config: Mapping[str, Any],
) -> dict[int, ConfirmatoryFeatureSet]:
    """Fit feature ranking and binary medians on training data only."""

    if not dataset.has_validation or dataset.X_validation is None:
        raise ValueError("The confirmatory protocol requires a validation partition.")
    feature_config = config["feature_selection"]
    X_selection = dataset.X_train
    y_selection = dataset.y_train
    max_samples = feature_config.get("max_samples")
    random_seed = int(feature_config.get("random_seed", 42))
    if max_samples is not None and len(X_selection) > int(max_samples):
        rng = np.random.default_rng(random_seed)
        selected_parts: list[np.ndarray] = []
        for label in (0, 1):
            indices = np.flatnonzero(y_selection == label)
            target = max(1, round(int(max_samples) * len(indices) / len(y_selection)))
            selected_parts.append(rng.choice(indices, size=min(target, len(indices)), replace=False))
        selected = np.concatenate(selected_parts)
        if len(selected) > int(max_samples):
            selected = rng.choice(selected, size=int(max_samples), replace=False)
        rng.shuffle(selected)
        X_selection = X_selection.iloc[selected].reset_index(drop=True)
        y_selection = y_selection[selected]

    score_table = compute_mutual_information_scores(
        X_selection,
        y_selection,
        random_seed=random_seed,
    )
    prepared: dict[int, ConfirmatoryFeatureSet] = {}
    for fs_size in config["experiment"]["feature_sizes"]:
        selection = select_top_features(score_table, int(fs_size))
        representation = binary_median_split(
            dataset.X_train,
            dataset.X_test,
            selection.selected_features,
            X_validation=dataset.X_validation,
        )
        signature = stable_feature_selection_signature(
            selection.selected_features,
            selection.selected_scores["mi_score"].to_numpy(dtype=float),
            selection.weights,
        )
        prepared[int(fs_size)] = ConfirmatoryFeatureSet(
            fs_size=int(fs_size),
            selected_features=selection.selected_features,
            weights=selection.weights,
            score_table=selection.selected_scores.copy(),
            representation=representation,
            feature_selection_hash=signature,
        )
    return prepared


def _partition_hashes(dataset: PreparedDataset) -> dict[str, str]:
    if not dataset.has_validation or dataset.X_validation is None or dataset.y_validation is None:
        raise ValueError("Validation partition is missing.")
    return {
        "train_partition_hash": stable_partition_signature(
            dataset.X_train,
            dataset.y_train,
            dataset.train_original_labels,
        ),
        "validation_partition_hash": stable_partition_signature(
            dataset.X_validation,
            dataset.y_validation,
            dataset.validation_original_labels,
        ),
        "test_partition_hash": stable_partition_signature(
            dataset.X_test,
            dataset.y_test,
            dataset.test_original_labels,
        ),
    }


def prepare_confirmatory_dataset(
    dataset_name: str,
    config: Mapping[str, Any],
) -> PreparedDataset:
    """Prepare one dataset according to the locked three-way protocol."""

    dataset_config = config["datasets"][dataset_name]
    protocol = config["protocol"]
    validation_size = float(protocol["validation_size"])
    validation_seed = int(protocol["validation_seed"])

    if dataset_name == "nsl_kdd":
        return prepare_nsl_kdd(
            dataset_config["train_file"],
            dataset_config["test_file"],
            scale=bool(dataset_config.get("scale", True)),
            validation_size=validation_size,
            validation_seed=validation_seed,
            validation_stratify_by=str(dataset_config.get("stratify_by", "original_label")),
        )
    if dataset_name == "cicids2017":
        sampling = dataset_config["sampling"]
        return prepare_cicids2017(
            dataset_config["raw_dir"],
            archive_file=dataset_config.get("archive_file"),
            label_column=str(dataset_config.get("label_column", "Label")),
            benign_label=str(dataset_config.get("benign_label", "BENIGN")),
            chunk_size=int(dataset_config.get("chunk_size", 25_000)),
            max_chunks_per_file=dataset_config.get("max_chunks_per_file"),
            drop_duplicates=bool(dataset_config.get("drop_duplicates", True)),
            sampling_strategy=str(sampling.get("strategy", "global_cap")),
            max_benign_records=int(sampling.get("max_benign_records", 1)),
            max_records_per_attack_label=int(sampling.get("max_records_per_attack_label", 1)),
            max_total_records=sampling.get("max_total_records"),
            sampling_seed=int(sampling.get("random_seed", 42)),
            test_size=float(dataset_config.get("test_size", 0.30)),
            split_seed=int(dataset_config.get("split_seed", 42)),
            stratify_by=str(dataset_config.get("stratify_by", "original_label")),
            scale=bool(dataset_config.get("scale", True)),
            validation_size=validation_size,
            validation_seed=validation_seed,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _method_display(method: str) -> str:
    return "Hamming" if method == "hamming" else "FW-LNSA Weighted Similarity"


def _weights_for_method(method: str, feature_set: ConfirmatoryFeatureSet) -> np.ndarray | None:
    return feature_set.weights if method == "weighted_similarity" else None


def _tuning_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["dataset_key"]),
        str(row["method"]),
        int(row["fs_size"]),
        int(row["seed"]),
        int(row["detector_budget"]),
        round(float(row["self_threshold_config"]), 12),
        round(float(row["target_fpr"]), 12),
    )


def _final_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["dataset_key"]),
        str(row["method"]),
        int(row["fs_size"]),
        int(row["seed"]),
        round(float(row["target_fpr"]), 12),
        str(row["configuration_id"]),
    )


def _dataset_file_fingerprints(dataset_name: str, config: Mapping[str, Any]) -> dict[str, str]:
    dataset_config = config["datasets"][dataset_name]
    if dataset_name == "nsl_kdd":
        paths = [Path(dataset_config["train_file"]), Path(dataset_config["test_file"])]
    else:
        archive = dataset_config.get("archive_file")
        paths = [Path(archive)] if archive else sorted(Path(dataset_config["raw_dir"]).rglob("*.csv"))
    return {str(path): file_sha256(path) for path in paths}


def _execution_environment() -> dict[str, Any]:
    """Capture stable software and hardware provenance for resume safety."""

    environment = environment_manifest(Path(__file__).resolve().parents[1])
    environment.pop("captured_utc", None)
    return environment


def _execution_state(
    dataset_name: str,
    dataset: PreparedDataset,
    feature_sets: Mapping[int, ConfirmatoryFeatureSet],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable identity for one resumable execution directory."""

    return {
        "protocol_version": str(config.get("protocol_version", PROTOCOL_VERSION)),
        "profile": str(config.get("profile_name", "unknown")),
        "dataset_key": dataset_name,
        "config_hash": stable_mapping_signature(dict(config)),
        "dataset_file_sha256": _dataset_file_fingerprints(dataset_name, config),
        "partition_hashes": _partition_hashes(dataset),
        "feature_selection_hashes": {
            f"FS-{size}": feature_set.feature_selection_hash
            for size, feature_set in feature_sets.items()
        },
        "execution_environment": _execution_environment(),
    }


def _validate_or_create_execution_state(
    state_path: Path,
    expected_state: Mapping[str, Any],
    *,
    progress_paths: Sequence[Path],
) -> None:
    """Reject unsafe resume attempts before reading saved result rows."""

    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            saved_state = json.load(handle)
        if saved_state != dict(expected_state):
            changed = [
                key
                for key in sorted(set(saved_state) | set(expected_state))
                if saved_state.get(key) != expected_state.get(key)
            ]
            raise RuntimeError(
                "The confirmatory output directory belongs to a different "
                f"execution identity. Changed fields: {', '.join(changed)}. "
                "Use a new output directory instead of mixing results."
            )
        return

    if any(path.exists() for path in progress_paths):
        raise RuntimeError(
            "Saved progress exists without an execution-state manifest. "
            "Use a new output directory so provenance cannot be ambiguous."
        )
    atomic_write_json(dict(expected_state), state_path)


def _base_run_metadata(
    dataset_name: str,
    config: Mapping[str, Any],
    feature_set: ConfirmatoryFeatureSet,
    partition_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "protocol_version": str(config.get("protocol_version", PROTOCOL_VERSION)),
        "profile": str(config.get("profile_name", "unknown")),
        "config_hash": stable_mapping_signature(dict(config)),
        "dataset_key": dataset_name,
        "dataset": "NSL-KDD" if dataset_name == "nsl_kdd" else "CICIDS2017",
        "fs_size": int(feature_set.fs_size),
        "feature_set": f"FS-{feature_set.fs_size}",
        "selected_features": "|".join(feature_set.selected_features),
        "feature_selection_hash": feature_set.feature_selection_hash,
        **dict(partition_hashes),
    }


def _fit_validation_pool(
    dataset: PreparedDataset,
    feature_set: ConfirmatoryFeatureSet,
    config: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    detector_budget: int,
    self_threshold_config: float,
    self_threshold_scale: str,
) -> tuple[FWLNSA, np.ndarray, dict[str, Any]]:
    if feature_set.representation.X_validation_bin is None or dataset.y_validation is None:
        raise ValueError("Validation representation is missing.")
    actual_self_threshold = _threshold_value(
        method,
        self_threshold_config,
        feature_set.fs_size,
        self_threshold_scale,
    )
    protocol = config["protocol"]
    model = FWLNSA(
        method=method,
        n_detectors=int(detector_budget),
        self_threshold=actual_self_threshold,
        detection_threshold=0.0,
        random_seed=int(seed),
        max_self_samples=protocol.get("max_self_samples"),
        deduplicate_candidates=bool(protocol.get("deduplicate_candidates", False)),
        prediction_chunk_size=int(protocol.get("prediction_chunk_size", 250)),
    )
    model.fit(
        feature_set.representation.X_train_bin,
        dataset.y_train,
        feature_weights=_weights_for_method(method, feature_set),
    )
    validation_scores, validation_resources = model.score_with_resources(
        feature_set.representation.X_validation_bin
    )
    return model, validation_scores, validation_resources


def run_tuning_stage(
    dataset_name: str,
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    feature_sets: Mapping[int, ConfirmatoryFeatureSet],
    *,
    existing_results: pd.DataFrame | None = None,
    result_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_callback: ProgressCallback | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Run validation-only configuration tuning with row-level resume."""

    if dataset.y_validation is None:
        raise ValueError("Validation labels are required.")
    experiment = config["experiment"]
    partition_hashes = _partition_hashes(dataset)
    completed: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    if existing_results is not None and not existing_results.empty:
        for row in existing_results.to_dict(orient="records"):
            key = _tuning_key(row)
            if key not in completed:
                completed.add(key)
                rows.append(dict(row))

    pool_specs: list[dict[str, Any]] = []
    for method in experiment["methods"]:
        threshold_settings = experiment["self_thresholds"][method]
        for fs_size in experiment["feature_sizes"]:
            for seed in experiment["tuning_seeds"]:
                for budget in experiment["detector_budgets"]:
                    for self_value in threshold_settings["values"]:
                        pool_specs.append(
                            {
                                "method": str(method),
                                "fs_size": int(fs_size),
                                "seed": int(seed),
                                "detector_budget": int(budget),
                                "self_threshold_config": float(self_value),
                                "self_threshold_scale": str(threshold_settings["scale"]),
                            }
                        )
    expected_rows = len(pool_specs) * len(experiment["target_fprs"])
    row_limit = expected_rows if max_rows is None else min(int(max_rows), expected_rows)

    for spec in pool_specs:
        missing_targets = [
            float(target)
            for target in experiment["target_fprs"]
            if _tuning_key(
                {
                    "dataset_key": dataset_name,
                    **spec,
                    "target_fpr": float(target),
                }
            )
            not in completed
        ]
        if not missing_targets:
            continue
        if len(rows) >= row_limit:
            break

        feature_set = feature_sets[int(spec["fs_size"])]
        model, validation_scores, validation_resources = _fit_validation_pool(
            dataset,
            feature_set,
            config,
            method=str(spec["method"]),
            seed=int(spec["seed"]),
            detector_budget=int(spec["detector_budget"]),
            self_threshold_config=float(spec["self_threshold_config"]),
            self_threshold_scale=str(spec["self_threshold_scale"]),
        )
        fit_summary = model.get_detector_summary()
        base = _base_run_metadata(dataset_name, config, feature_set, partition_hashes)

        for target_fpr in missing_targets:
            if len(rows) >= row_limit:
                break
            calibration = calibrate_detection_threshold(
                validation_scores,
                dataset.y_validation,
                method=str(spec["method"]),
                target_fpr=target_fpr,
            )
            predictions = predict_from_native_scores(
                validation_scores,
                method=str(spec["method"]),
                threshold=calibration.threshold,
            )
            metrics = evaluate_binary_classification(
                dataset.y_validation,
                predictions,
                y_score=_attack_scores(str(spec["method"]), validation_scores),
            )
            row = {
                **base,
                "stage": "tuning",
                "method": str(spec["method"]),
                "method_name": _method_display(str(spec["method"])),
                "seed": int(spec["seed"]),
                "detector_budget": int(spec["detector_budget"]),
                "self_threshold_config": float(spec["self_threshold_config"]),
                "self_threshold_scale": str(spec["self_threshold_scale"]),
                **fit_summary,
                "self_threshold": float(fit_summary["self_threshold"]),
                **calibration.to_dict(),
                **{f"validation_{key}": value for key, value in metrics.to_dict().items()},
                **{f"validation_{key}": value for key, value in validation_resources.items()},
                "test_partition_accessed": False,
            }
            key = _tuning_key(row)
            rows.append(row)
            completed.add(key)
            if result_callback is not None:
                result_callback(row)
            if progress_callback is not None:
                progress_callback("tuning", len(rows), row_limit, row)

    order_columns = [
        "method",
        "fs_size",
        "seed",
        "detector_budget",
        "self_threshold_config",
        "target_fpr",
    ]
    return pd.DataFrame(rows).sort_values(order_columns).reset_index(drop=True)


def summarize_tuning_results(
    tuning_results: pd.DataFrame,
    *,
    confidence: float,
) -> pd.DataFrame:
    if tuning_results.empty:
        return pd.DataFrame()
    groups = [
        "dataset_key",
        "dataset",
        "method",
        "method_name",
        "fs_size",
        "feature_set",
        "detector_budget",
        "self_threshold_config",
        "self_threshold_scale",
        "self_threshold",
        "target_fpr",
        "feature_selection_hash",
        "train_partition_hash",
        "validation_partition_hash",
        "test_partition_hash",
    ]
    metrics = [
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_fpr",
        "validation_specificity",
        "validation_balanced_accuracy",
        "validation_mcc",
        "validation_pr_auc",
        "validation_roc_auc",
        "calibration_fpr",
        "retained_detectors",
        "retention_rate",
        "generation_time_sec",
        "validation_detection_time_sec",
        "validation_prediction_records_per_sec",
        "detector_array_bytes",
        "serialized_model_bytes",
        "peak_fit_rss_delta_mb",
        "validation_peak_prediction_rss_delta_mb",
    ]
    summary = _aggregate_metrics(
        tuning_results,
        groups,
        metrics,
        confidence=confidence,
    )
    seed_counts = (
        tuning_results.groupby(groups, dropna=False)["seed"]
        .nunique()
        .rename("tuning_seed_count")
        .reset_index()
    )
    return summary.merge(seed_counts, on=groups, how="left")


def lock_configurations(
    tuning_summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Lock one validation-selected configuration per method, FS, and FPR."""

    if tuning_summary.empty:
        raise ValueError("Tuning summary is empty.")
    expected_seeds = len(config["experiment"]["tuning_seeds"])
    complete = tuning_summary[tuning_summary["tuning_seed_count"] == expected_seeds].copy()
    if complete.empty:
        raise ValueError("No configuration contains every configured tuning seed.")

    selected_rows: list[pd.Series] = []
    for _, group in complete.groupby(
        ["dataset_key", "method", "fs_size", "target_fpr"],
        sort=False,
    ):
        controlled = group[
            group["validation_fpr_mean"] <= group["target_fpr"] + 1e-12
        ].copy()
        if controlled.empty:
            raise RuntimeError("No validation-controlled configuration met its target FPR.")
        ordered = controlled.sort_values(
            [
                "validation_f1_mean",
                "validation_recall_mean",
                "validation_fpr_mean",
                "generation_time_sec_mean",
                "serialized_model_bytes_mean",
            ],
            ascending=[False, False, True, True, True],
        )
        selected = ordered.iloc[0].copy()
        selected["selection_rule"] = (
            "Highest mean validation F1 at calibrated FPR, followed by recall, "
            "lower FPR, runtime, and serialized size"
        )
        selected["configuration_id"] = stable_mapping_signature(
            {
                "dataset": selected["dataset_key"],
                "method": selected["method"],
                "fs_size": int(selected["fs_size"]),
                "target_fpr": float(selected["target_fpr"]),
                "detector_budget": int(selected["detector_budget"]),
                "self_threshold_config": float(selected["self_threshold_config"]),
                "self_threshold_scale": selected["self_threshold_scale"],
            }
        )[:16]
        selected_rows.append(selected)
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def build_configuration_lock(
    dataset_name: str,
    dataset: PreparedDataset,
    feature_sets: Mapping[int, ConfirmatoryFeatureSet],
    locked_configurations: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    partition_hashes = _partition_hashes(dataset)
    lock = {
        "protocol_version": str(config.get("protocol_version", PROTOCOL_VERSION)),
        "profile": str(config.get("profile_name", "unknown")),
        "dataset_key": dataset_name,
        "dataset_metadata": dataset.metadata,
        "dataset_file_sha256": _dataset_file_fingerprints(dataset_name, config),
        "config_hash": stable_mapping_signature(dict(config)),
        "partition_hashes": partition_hashes,
        "feature_selection_hashes": {
            f"FS-{size}": feature_set.feature_selection_hash
            for size, feature_set in feature_sets.items()
        },
        "tuning_seeds": list(config["experiment"]["tuning_seeds"]),
        "confirmatory_seeds": list(config["experiment"]["confirmatory_seeds"]),
        "execution_environment": _execution_environment(),
        "calibration": {
            "source": "benign validation scores only",
            "target_fprs": list(config["experiment"]["target_fprs"]),
            "test_metrics_used_for_selection": False,
        },
        "locked_configurations": locked_configurations.to_dict(orient="records"),
    }
    lock["lock_id"] = stable_mapping_signature(lock)
    return lock


def validate_configuration_lock(
    lock: Mapping[str, Any],
    dataset_name: str,
    dataset: PreparedDataset,
    feature_sets: Mapping[int, ConfirmatoryFeatureSet],
    config: Mapping[str, Any],
) -> None:
    if str(lock.get("dataset_key")) != dataset_name:
        raise ValueError("Configuration lock belongs to a different dataset.")
    if str(lock.get("config_hash")) != stable_mapping_signature(dict(config)):
        raise ValueError("Configuration changed after tuning. Create a new output directory.")
    if dict(lock.get("partition_hashes", {})) != _partition_hashes(dataset):
        raise ValueError("Prepared partitions do not match the configuration lock.")
    expected_features = {
        f"FS-{size}": feature_set.feature_selection_hash
        for size, feature_set in feature_sets.items()
    }
    if dict(lock.get("feature_selection_hashes", {})) != expected_features:
        raise ValueError("Feature selection does not match the configuration lock.")


def run_final_stage(
    dataset_name: str,
    dataset: PreparedDataset,
    config: Mapping[str, Any],
    feature_sets: Mapping[int, ConfirmatoryFeatureSet],
    lock: Mapping[str, Any],
    *,
    existing_results: pd.DataFrame | None = None,
    existing_categories: pd.DataFrame | None = None,
    result_callback: Callable[[Mapping[str, Any]], None] | None = None,
    category_callback: Callable[[Mapping[str, Any]], None] | None = None,
    progress_callback: ProgressCallback | None = None,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate locked procedures on the untouched test partition."""

    validate_configuration_lock(lock, dataset_name, dataset, feature_sets, config)
    if dataset.y_validation is None:
        raise ValueError("Validation labels are required.")
    partition_hashes = _partition_hashes(dataset)
    selected = pd.DataFrame(lock["locked_configurations"])
    confirmatory_seeds = list(config["experiment"]["confirmatory_seeds"])
    expected_rows = len(selected) * len(confirmatory_seeds)
    row_limit = expected_rows if max_rows is None else min(int(max_rows), expected_rows)

    rows: list[dict[str, Any]] = []
    completed: set[tuple[Any, ...]] = set()
    if existing_results is not None and not existing_results.empty:
        for row in existing_results.to_dict(orient="records"):
            key = _final_key(row)
            if key not in completed:
                completed.add(key)
                rows.append(dict(row))

    category_rows = (
        existing_categories.to_dict(orient="records")
        if existing_categories is not None and not existing_categories.empty
        else []
    )
    category_completed = {
        (_final_key(row), str(row["category"])) for row in category_rows
    }

    # Configurations with the same pool settings share a fitted model and score
    # pass. Different target FPRs are calibrated from the same validation scores.
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for seed in confirmatory_seeds:
        for record in selected.to_dict(orient="records"):
            key = (
                str(record["method"]),
                int(record["fs_size"]),
                int(seed),
                int(record["detector_budget"]),
                round(float(record["self_threshold_config"]), 12),
                str(record["self_threshold_scale"]),
            )
            grouped.setdefault(key, []).append(record)

    for group_key, records in grouped.items():
        method, fs_size, seed, budget, self_value, self_scale = group_key
        missing = [
            record
            for record in records
            if _final_key(
                {
                    "dataset_key": dataset_name,
                    "method": method,
                    "fs_size": fs_size,
                    "seed": seed,
                    "target_fpr": record["target_fpr"],
                    "configuration_id": record["configuration_id"],
                }
            )
            not in completed
        ]
        if not missing:
            continue
        if len(rows) >= row_limit:
            break

        feature_set = feature_sets[int(fs_size)]
        model, validation_scores, validation_resources = _fit_validation_pool(
            dataset,
            feature_set,
            config,
            method=str(method),
            seed=int(seed),
            detector_budget=int(budget),
            self_threshold_config=float(self_value),
            self_threshold_scale=str(self_scale),
        )
        test_scores, test_resources = model.score_with_resources(
            feature_set.representation.X_test_bin
        )
        fit_summary = model.get_detector_summary()
        base = _base_run_metadata(dataset_name, config, feature_set, partition_hashes)

        for locked_record in missing:
            if len(rows) >= row_limit:
                break
            calibration = calibrate_detection_threshold(
                validation_scores,
                dataset.y_validation,
                method=str(method),
                target_fpr=float(locked_record["target_fpr"]),
            )
            validation_predictions = predict_from_native_scores(
                validation_scores,
                method=str(method),
                threshold=calibration.threshold,
            )
            test_predictions = predict_from_native_scores(
                test_scores,
                method=str(method),
                threshold=calibration.threshold,
            )
            validation_metrics = evaluate_binary_classification(
                dataset.y_validation,
                validation_predictions,
                y_score=_attack_scores(str(method), validation_scores),
            )
            test_metrics = evaluate_binary_classification(
                dataset.y_test,
                test_predictions,
                y_score=_attack_scores(str(method), test_scores),
            )
            row = {
                **base,
                "stage": "confirmatory",
                "lock_id": str(lock["lock_id"]),
                "configuration_id": str(locked_record["configuration_id"]),
                "method": str(method),
                "method_name": _method_display(str(method)),
                "seed": int(seed),
                "detector_budget": int(budget),
                "self_threshold_config": float(self_value),
                "self_threshold_scale": str(self_scale),
                **fit_summary,
                "self_threshold": float(fit_summary["self_threshold"]),
                **calibration.to_dict(),
                **{f"validation_{key}": value for key, value in validation_metrics.to_dict().items()},
                **{f"test_{key}": value for key, value in test_metrics.to_dict().items()},
                **{f"validation_{key}": value for key, value in validation_resources.items()},
                **{f"test_{key}": value for key, value in test_resources.items()},
                "configuration_selected_on_validation": True,
                "test_metrics_used_for_selection": False,
                "test_evaluations_for_seed": 1,
            }
            key = _final_key(row)
            rows.append(row)
            completed.add(key)
            if result_callback is not None:
                result_callback(row)

            if dataset.test_attack_categories is not None:
                category_frame = attack_category_analysis(
                    dataset.y_test,
                    test_predictions,
                    dataset.test_attack_categories,
                    run_metadata={
                        "dataset_key": dataset_name,
                        "dataset": base["dataset"],
                        "method": str(method),
                        "method_name": _method_display(str(method)),
                        "feature_set": f"FS-{fs_size}",
                        "fs_size": int(fs_size),
                        "seed": int(seed),
                        "target_fpr": float(locked_record["target_fpr"]),
                        "configuration_id": str(locked_record["configuration_id"]),
                        "lock_id": str(lock["lock_id"]),
                    },
                )
                for category_row in category_frame.to_dict(orient="records"):
                    category_key = (key, str(category_row["category"]))
                    if category_key in category_completed:
                        continue
                    category_rows.append(category_row)
                    category_completed.add(category_key)
                    if category_callback is not None:
                        category_callback(category_row)

            if progress_callback is not None:
                progress_callback("confirmatory", len(rows), row_limit, row)

    final_frame = pd.DataFrame(rows)
    if not final_frame.empty:
        final_frame = final_frame.sort_values(
            ["method", "fs_size", "target_fpr", "seed"]
        ).reset_index(drop=True)
    category_frame = pd.DataFrame(category_rows)
    if not category_frame.empty:
        category_frame = category_frame.sort_values(
            ["method", "fs_size", "target_fpr", "seed", "category"]
        ).reset_index(drop=True)
    return final_frame, category_frame


def summarize_final_results(
    final_results: pd.DataFrame,
    *,
    confidence: float,
) -> pd.DataFrame:
    if final_results.empty:
        return pd.DataFrame()
    groups = [
        "dataset_key",
        "dataset",
        "method",
        "method_name",
        "fs_size",
        "feature_set",
        "target_fpr",
        "configuration_id",
        "detector_budget",
        "self_threshold_config",
        "self_threshold_scale",
        "feature_selection_hash",
        "train_partition_hash",
        "validation_partition_hash",
        "test_partition_hash",
        "lock_id",
    ]
    metrics = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_fpr",
        "test_specificity",
        "test_balanced_accuracy",
        "test_mcc",
        "test_pr_auc",
        "test_roc_auc",
        "calibration_fpr",
        "retained_detectors",
        "retention_rate",
        "generation_time_sec",
        "test_detection_time_sec",
        "test_prediction_records_per_sec",
        "test_prediction_latency_per_1000_ms",
        "detector_array_bytes",
        "feature_weights_bytes",
        "serialized_model_bytes",
        "peak_fit_rss_delta_mb",
        "test_peak_prediction_rss_delta_mb",
    ]
    summary = _aggregate_metrics(
        final_results,
        groups,
        metrics,
        confidence=confidence,
    )
    seed_counts = (
        final_results.groupby(groups, dropna=False)["seed"]
        .nunique()
        .rename("confirmatory_seed_count")
        .reset_index()
    )
    return summary.merge(seed_counts, on=groups, how="left")


def summarize_category_results(
    category_results: pd.DataFrame,
    *,
    confidence: float,
) -> pd.DataFrame:
    if category_results.empty:
        return pd.DataFrame()
    groups = [
        "dataset_key",
        "dataset",
        "method",
        "method_name",
        "fs_size",
        "feature_set",
        "target_fpr",
        "configuration_id",
        "category",
        "lock_id",
    ]
    metrics = [
        "category_recall",
        "false_alarm_rate",
        "detected_attacks",
        "missed_attacks",
        "false_alarms",
    ]
    summary = _aggregate_metrics(
        category_results,
        groups,
        metrics,
        confidence=confidence,
    )
    records = (
        category_results.groupby(groups, dropna=False)
        .agg(
            test_records_per_seed=("total_records", "first"),
            confirmatory_seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    summary = summary.merge(records, on=groups, how="left")
    summary["category_evidence_status"] = np.where(
        summary["test_records_per_seed"] == 0,
        "no_test_observations",
        np.where(
            summary["test_records_per_seed"] < 20,
            "limited_sample",
            "reportable",
        ),
    )
    return summary


def _selected_feature_table(feature_sets: Mapping[int, ConfirmatoryFeatureSet]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fs_size, feature_set in feature_sets.items():
        table = feature_set.score_table.copy()
        table.insert(0, "feature_set", f"FS-{fs_size}")
        table.insert(1, "fs_size", fs_size)
        table.insert(2, "rank", np.arange(1, len(table) + 1))
        table["feature_selection_hash"] = feature_set.feature_selection_hash
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def expected_confirmatory_plan(config: Mapping[str, Any]) -> dict[str, int]:
    experiment = config["experiment"]
    self_counts = sum(
        len(experiment["self_thresholds"][method]["values"])
        for method in experiment["methods"]
    )
    tuning_fits = (
        len(experiment["feature_sizes"])
        * len(experiment["detector_budgets"])
        * len(experiment["tuning_seeds"])
        * self_counts
    )
    tuning_rows = tuning_fits * len(experiment["target_fprs"])
    locked_configurations = (
        len(experiment["methods"])
        * len(experiment["feature_sizes"])
        * len(experiment["target_fprs"])
    )
    final_rows = locked_configurations * len(experiment["confirmatory_seeds"])
    return {
        "tuning_model_fits_per_dataset": int(tuning_fits),
        "tuning_rows_per_dataset": int(tuning_rows),
        "locked_configurations_per_dataset": int(locked_configurations),
        "confirmatory_rows_per_dataset": int(final_rows),
    }


def run_confirmatory_dataset(
    dataset_name: str,
    config: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    stage: str = "all",
    max_tuning_rows: int | None = None,
    max_final_rows: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ConfirmatoryRunOutputs:
    """Run or resume the complete protocol for one dataset."""

    if stage not in {"tune", "confirm", "all"}:
        raise ValueError("stage must be tune, confirm, or all.")
    root = ensure_dir(output_root or config["paths"]["output_dir"])
    dataset_root = ensure_dir(root / dataset_name)
    tables_dir = ensure_dir(dataset_root / "tables")
    progress_dir = ensure_dir(dataset_root / "progress")
    manifests_dir = ensure_dir(dataset_root / "manifests")

    dataset = prepare_confirmatory_dataset(dataset_name, config)
    feature_sets = prepare_confirmatory_feature_sets(dataset, config)
    confidence = float(config["protocol"].get("confidence", 0.95))

    tuning_journal = progress_dir / "tuning_rows.jsonl"
    final_journal = progress_dir / "confirmatory_rows.jsonl"
    category_journal = progress_dir / "confirmatory_category_rows.jsonl"
    lock_path = manifests_dir / "configuration_lock.json"
    state_path = manifests_dir / "execution_state.json"
    expected_state = _execution_state(dataset_name, dataset, feature_sets, config)
    _validate_or_create_execution_state(
        state_path,
        expected_state,
        progress_paths=[tuning_journal, final_journal, category_journal, lock_path],
    )

    tuning_results = load_jsonl_frame(tuning_journal)
    if stage in {"tune", "all"}:
        tuning_results = run_tuning_stage(
            dataset_name,
            dataset,
            config,
            feature_sets,
            existing_results=tuning_results,
            result_callback=lambda row: append_jsonl(row, tuning_journal),
            progress_callback=progress_callback,
            max_rows=max_tuning_rows,
        )
    tuning_summary = summarize_tuning_results(tuning_results, confidence=confidence)

    plan = expected_confirmatory_plan(config)
    expected_tuning = plan["tuning_rows_per_dataset"]
    lock: dict[str, Any] = {}
    locked_configurations = pd.DataFrame()
    if len(tuning_results) == expected_tuning:
        locked_configurations = lock_configurations(tuning_summary, config)
        lock = build_configuration_lock(
            dataset_name,
            dataset,
            feature_sets,
            locked_configurations,
            config,
        )
        atomic_write_json(lock, lock_path)
    elif lock_path.exists():
        with lock_path.open("r", encoding="utf-8") as handle:
            lock = json.load(handle)
        locked_configurations = pd.DataFrame(lock["locked_configurations"])

    final_results = load_jsonl_frame(final_journal)
    category_results = load_jsonl_frame(category_journal)
    if stage in {"confirm", "all"}:
        if not lock:
            raise RuntimeError(
                f"Tuning is incomplete: {len(tuning_results)}/{expected_tuning} rows. "
                "Complete tuning before final evaluation."
            )
        final_results, category_results = run_final_stage(
            dataset_name,
            dataset,
            config,
            feature_sets,
            lock,
            existing_results=final_results,
            existing_categories=category_results,
            result_callback=lambda row: append_jsonl(row, final_journal),
            category_callback=lambda row: append_jsonl(row, category_journal),
            progress_callback=progress_callback,
            max_rows=max_final_rows,
        )

    final_summary = summarize_final_results(final_results, confidence=confidence)
    category_summary = summarize_category_results(category_results, confidence=confidence)
    selected_features = _selected_feature_table(feature_sets)

    output_paths = {
        "tuning_results": atomic_write_csv(tuning_results, tables_dir / "tuning_results.csv"),
        "tuning_summary": atomic_write_csv(tuning_summary, tables_dir / "tuning_summary.csv"),
        "locked_configurations": atomic_write_csv(
            locked_configurations,
            tables_dir / "locked_configurations.csv",
        ),
        "final_seed_results": atomic_write_csv(
            final_results,
            tables_dir / "final_seed_results.csv",
        ),
        "final_summary": atomic_write_csv(final_summary, tables_dir / "final_summary.csv"),
        "category_seed_results": atomic_write_csv(
            category_results,
            tables_dir / "category_seed_results.csv",
        ),
        "category_summary": atomic_write_csv(
            category_summary,
            tables_dir / "category_summary.csv",
        ),
        "selected_features": atomic_write_csv(
            selected_features,
            tables_dir / "selected_features.csv",
        ),
    }
    if dataset.data_quality_report is not None:
        output_paths["data_quality_report"] = atomic_write_csv(
            dataset.data_quality_report,
            tables_dir / "data_quality_report.csv",
        )

    manifest = {
        **expected_state,
        "dataset_metadata": dataset.metadata,
        "plan": plan,
        "completed": {
            "tuning_rows": int(len(tuning_results)),
            "locked_configurations": int(len(locked_configurations)),
            "confirmatory_rows": int(len(final_results)),
            "category_rows": int(len(category_results)),
        },
        "configuration_lock": str(lock_path) if lock else None,
        "test_metrics_used_for_selection": False,
    }
    manifest_path = manifests_dir / "execution_manifest.json"
    atomic_write_json(manifest, manifest_path)
    output_paths["manifest"] = manifest_path
    output_paths["execution_state"] = state_path
    if lock:
        output_paths["configuration_lock"] = lock_path

    return ConfirmatoryRunOutputs(
        tuning_results=tuning_results,
        tuning_summary=tuning_summary,
        locked_configurations=locked_configurations,
        final_seed_results=final_results,
        final_summary=final_summary,
        category_seed_results=category_results,
        category_summary=category_summary,
        selected_features=selected_features,
        manifest=manifest,
        output_paths=output_paths,
    )
