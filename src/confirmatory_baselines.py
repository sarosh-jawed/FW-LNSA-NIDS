"""Confirmatory classical baselines aligned to the locked FW-LNSA protocol.

Every model is trained on the confirmatory training partition. A decision
threshold is calibrated using benign validation scores only, then the locked
procedure is evaluated once on the untouched test partition. The module also
verifies partition and feature-selection hashes against the completed
FW-LNSA confirmatory artifacts before any baseline result is accepted.
"""

from __future__ import annotations

import json
import math
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
from scipy import stats

from .baselines import BASELINE_MODEL_FAMILIES, BASELINE_MODEL_NAMES, build_baseline_model
from .config import ConfigurationError, deep_merge, load_yaml_config
from .confirmatory_protocol import (
    calibrate_detection_threshold,
    load_confirmatory_config,
    prepare_confirmatory_dataset,
)
from .evaluation import attack_category_analysis, evaluate_binary_classification
from .research_execution import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    environment_manifest,
    file_sha256,
    load_jsonl_frame,
)
from .utils import ensure_dir, stable_mapping_signature, stable_partition_signature


BASELINE_PROTOCOL_VERSION = "1.0"
BASELINE_RESULT_KEY = ("dataset_key", "model", "fs_size", "seed", "target_fpr")
ProgressCallback = Callable[[int, int, Mapping[str, Any]], None]


@dataclass(frozen=True)
class ConfirmatoryBaselineOutputs:
    seed_results: pd.DataFrame
    summary: pd.DataFrame
    category_seed_results: pd.DataFrame
    category_summary: pd.DataFrame
    selected_features: pd.DataFrame
    manifest: dict[str, Any]
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class LockedFeatureSet:
    fs_size: int
    selected_features: list[str]
    weights: np.ndarray
    feature_selection_hash: str


class _PeakMemoryMonitor:
    """Sample process RSS while one baseline is fitted and scored."""

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PeakMemoryMonitor":
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            except psutil.Error:
                return

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.interval * 4))
        try:
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        except psutil.Error:
            pass

    @property
    def peak_mb(self) -> float:
        return float(self.peak_rss / (1024 ** 2))

    @property
    def delta_mb(self) -> float:
        return float(max(0, self.peak_rss - self.start_rss) / (1024 ** 2))


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{key}' must be a mapping.")
    return value


def validate_manuscript_config(config: Mapping[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    for key in ("confirmatory_config", "confirmatory_results_dir", "manuscript_results_dir", "handoff_dir"):
        if not paths.get(key):
            raise ConfigurationError(f"paths.{key} is required.")

    baseline = _require_mapping(config, "baseline")
    datasets = baseline.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ConfigurationError("baseline.datasets must be a non-empty list.")
    if set(datasets) - {"nsl_kdd", "cicids2017"}:
        raise ConfigurationError("baseline.datasets contains an unsupported dataset.")
    for key in ("feature_sizes", "seeds", "target_fprs"):
        values = baseline.get(key)
        if not isinstance(values, list) or not values:
            raise ConfigurationError(f"baseline.{key} must be a non-empty list.")
    if any(int(value) <= 0 for value in baseline["feature_sizes"]):
        raise ConfigurationError("baseline.feature_sizes must contain positive integers.")
    if any(not isinstance(value, int) for value in baseline["seeds"]):
        raise ConfigurationError("baseline.seeds must contain integers.")
    if any(not 0 <= float(value) <= 1 for value in baseline["target_fprs"]):
        raise ConfigurationError("baseline.target_fprs must contain probabilities.")
    models = _require_mapping(baseline, "models")
    supported = {"logistic_regression", "decision_tree", "random_forest", "isolation_forest"}
    if set(models) - supported:
        raise ConfigurationError(f"Unsupported baseline models: {sorted(set(models) - supported)}")
    if not any(bool(settings.get("enabled", False)) for settings in models.values() if isinstance(settings, Mapping)):
        raise ConfigurationError("At least one confirmatory baseline model must be enabled.")


def resolve_manuscript_profile(
    config: Mapping[str, Any], profile_name: str | None = None
) -> dict[str, Any]:
    base = {key: value for key, value in config.items() if key != "profiles"}
    profiles = config.get("profiles", {})
    selected = profile_name or str(base.get("active_profile", "smoke"))
    if not isinstance(profiles, Mapping) or selected not in profiles:
        available = ", ".join(sorted(map(str, profiles))) if isinstance(profiles, Mapping) else "none"
        raise ConfigurationError(f"Unknown manuscript profile '{selected}'. Available: {available}.")
    override = profiles[selected]
    if not isinstance(override, Mapping):
        raise ConfigurationError(f"Profile '{selected}' must be a mapping.")
    resolved = deep_merge(base, override)
    resolved["profile_name"] = selected
    validate_manuscript_config(resolved)
    return resolved


def load_manuscript_config(
    path: str | Path, *, profile_name: str | None = None
) -> dict[str, Any]:
    return resolve_manuscript_profile(load_yaml_config(path), profile_name)


def _partition_hashes(dataset: Any) -> dict[str, str]:
    if dataset.X_validation is None or dataset.y_validation is None:
        raise ValueError("Confirmatory baselines require a validation partition.")
    return {
        "train_partition_hash": stable_partition_signature(
            dataset.X_train, dataset.y_train, dataset.train_original_labels
        ),
        "validation_partition_hash": stable_partition_signature(
            dataset.X_validation, dataset.y_validation, dataset.validation_original_labels
        ),
        "test_partition_hash": stable_partition_signature(
            dataset.X_test, dataset.y_test, dataset.test_original_labels
        ),
    }


def _read_manifest(confirmatory_root: Path, dataset_key: str) -> dict[str, Any]:
    path = confirmatory_root / dataset_key / "manifests" / "execution_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Confirmatory manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dataset_key") != dataset_key:
        raise ValueError(f"Confirmatory manifest dataset mismatch for {dataset_key}.")
    if bool(manifest.get("test_metrics_used_for_selection", True)):
        raise ValueError("Confirmatory manifest indicates test metrics were used for selection.")
    return manifest


def verify_dataset_fingerprints(
    dataset_key: str,
    confirmatory_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Match current raw files to the SHA-256 values used by confirmation."""

    dataset_config = confirmatory_config["datasets"][dataset_key]
    if dataset_key == "nsl_kdd":
        current_paths = [
            Path(dataset_config["train_file"]),
            Path(dataset_config["test_file"]),
        ]
    else:
        current_paths = [Path(dataset_config["archive_file"])]
    actual = {str(path): file_sha256(path) for path in current_paths}
    expected_hashes = sorted(map(str, manifest.get("dataset_file_sha256", {}).values()))
    if sorted(actual.values()) != expected_hashes:
        raise ValueError(
            f"Dataset fingerprint mismatch for {dataset_key}. "
            "The baseline inputs differ from the confirmatory run."
        )
    return actual


def load_locked_feature_sets(
    confirmatory_root: str | Path, dataset_key: str
) -> dict[int, LockedFeatureSet]:
    """Load the exact selected features saved by the confirmatory run.

    Baselines deliberately consume the locked artifact instead of recomputing
    mutual information. This prevents library-version drift from silently
    changing the comparison feature space.
    """

    root = Path(confirmatory_root)
    manifest = _read_manifest(root, dataset_key)
    path = root / dataset_key / "tables" / "selected_features.csv"
    frame = pd.read_csv(path)
    required = {"fs_size", "rank", "feature", "normalized_weight", "feature_selection_hash"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Locked selected-feature table is missing: {sorted(missing)}")
    output: dict[int, LockedFeatureSet] = {}
    for fs_size, group in frame.groupby(frame["fs_size"].astype(int), sort=True):
        ordered = group.sort_values("rank")
        hashes = ordered["feature_selection_hash"].astype(str).unique()
        if len(hashes) != 1:
            raise ValueError(f"Multiple feature-selection hashes found for FS-{fs_size}.")
        expected = manifest.get("feature_selection_hashes", {}).get(f"FS-{int(fs_size)}")
        if expected != hashes[0]:
            raise ValueError(f"Saved feature-selection hash mismatch for {dataset_key} FS-{fs_size}.")
        output[int(fs_size)] = LockedFeatureSet(
            fs_size=int(fs_size),
            selected_features=ordered["feature"].astype(str).tolist(),
            weights=ordered["normalized_weight"].astype(float).to_numpy(),
            feature_selection_hash=str(hashes[0]),
        )
    return output


def verify_confirmatory_contract(
    dataset_key: str,
    dataset: Any,
    feature_sets: Mapping[int, Any],
    confirmatory_root: str | Path,
) -> dict[str, Any]:
    """Verify recreated partitions and selected features against saved evidence."""

    root = Path(confirmatory_root)
    manifest = _read_manifest(root, dataset_key)
    actual_partitions = _partition_hashes(dataset)
    expected_partitions = manifest.get("partition_hashes", {})
    if actual_partitions != expected_partitions:
        raise ValueError(
            f"Partition hash mismatch for {dataset_key}. Baselines would not be comparable."
        )

    actual_features = {
        f"FS-{size}": feature_set.feature_selection_hash
        for size, feature_set in sorted(feature_sets.items())
    }
    expected_features = manifest.get("feature_selection_hashes", {})
    for key, expected in expected_features.items():
        if actual_features.get(key) != expected:
            raise ValueError(
                f"Feature-selection hash mismatch for {dataset_key} {key}."
            )

    for size, feature_set in feature_sets.items():
        missing_columns = set(feature_set.selected_features) - set(dataset.X_train.columns)
        if missing_columns:
            raise ValueError(
                f"Locked features are missing from the recreated {dataset_key} partition: "
                f"{sorted(missing_columns)}"
            )

    return {
        "manifest": manifest,
        "partition_hashes": actual_partitions,
        "feature_selection_hashes": actual_features,
    }


def _attack_scores(model_name: str, model: object, X: pd.DataFrame) -> np.ndarray:
    """Return scores oriented so larger values indicate a stronger attack."""

    if model_name == "isolation_forest":
        return -np.asarray(model.decision_function(X), dtype=float)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        classes = list(getattr(model, "classes_", [0, 1]))
        try:
            attack_index = classes.index(1)
        except ValueError as exc:
            raise ValueError("Supervised baseline did not learn the attack class.") from exc
        return probabilities[:, attack_index]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _fit_and_score(
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_mask = np.ones(len(y_train), dtype=bool)
    if model_name == "isolation_forest":
        train_mask = y_train == 0
        if not np.any(train_mask):
            raise ValueError("Isolation Forest requires benign training records.")

    with _PeakMemoryMonitor() as memory:
        fit_start = time.perf_counter()
        if model_name == "isolation_forest":
            model.fit(X_train.loc[train_mask])
        else:
            model.fit(X_train, y_train)
        fit_time = time.perf_counter() - fit_start

        validation_start = time.perf_counter()
        validation_scores = _attack_scores(model_name, model, X_validation)
        validation_time = time.perf_counter() - validation_start

        test_start = time.perf_counter()
        test_scores = _attack_scores(model_name, model, X_test)
        test_time = time.perf_counter() - test_start

    resources = {
        "fit_time_sec": float(fit_time),
        "validation_score_time_sec": float(validation_time),
        "test_score_time_sec": float(test_time),
        "total_time_sec": float(fit_time + validation_time + test_time),
        "validation_records_per_sec": float(len(X_validation) / validation_time) if validation_time else math.inf,
        "test_records_per_sec": float(len(X_test) / test_time) if test_time else math.inf,
        "peak_process_memory_mb": memory.peak_mb,
        "peak_memory_increase_mb": memory.delta_mb,
        "serialized_model_bytes": int(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))),
        "training_records_available": int(len(y_train)),
        "training_records_used": int(np.sum(train_mask)),
        "normal_training_records_used": int(np.sum(y_train[train_mask] == 0)),
        "attack_training_records_used": int(np.sum(y_train[train_mask] == 1)),
    }
    return validation_scores, test_scores, resources


def _result_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for column in BASELINE_RESULT_KEY:
        value = row[column]
        if isinstance(value, (float, np.floating)):
            value = round(float(value), 12)
        elif isinstance(value, (int, np.integer)):
            value = int(value)
        else:
            value = str(value)
        values.append(value)
    return tuple(values)


def _iter_specs(config: Mapping[str, Any], dataset_key: str) -> Iterable[dict[str, Any]]:
    baseline = config["baseline"]
    enabled = [
        name
        for name, settings in baseline["models"].items()
        if isinstance(settings, Mapping) and bool(settings.get("enabled", False))
    ]
    for fs_size in baseline["feature_sizes"]:
        for model_name in enabled:
            for seed in baseline["seeds"]:
                yield {
                    "dataset_key": dataset_key,
                    "model": model_name,
                    "fs_size": int(fs_size),
                    "seed": int(seed),
                }


def _mean_std_ci(values: Sequence[float], confidence: float) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(array.mean())
    if len(array) == 1:
        return {"count": 1, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(array.std(ddof=1))
    margin = float(stats.t.ppf((1 + confidence) / 2, len(array) - 1) * std / math.sqrt(len(array)))
    return {"count": len(array), "mean": mean, "std": std, "ci_low": mean - margin, "ci_high": mean + margin}


def summarize_confirmatory_baselines(
    results: pd.DataFrame, *, confidence: float = 0.95
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    groups = [
        "dataset_key", "dataset", "model", "model_name", "model_family",
        "feature_set", "fs_size", "target_fpr", "model_parameters_json",
        "feature_selection_hash", "train_partition_hash",
        "validation_partition_hash", "test_partition_hash",
    ]
    metrics = [
        "validation_accuracy", "validation_precision", "validation_recall",
        "validation_f1", "validation_fpr", "validation_balanced_accuracy",
        "validation_mcc", "validation_pr_auc", "validation_roc_auc",
        "test_accuracy", "test_precision", "test_recall", "test_f1",
        "test_fpr", "test_balanced_accuracy", "test_mcc", "test_pr_auc",
        "test_roc_auc", "calibration_fpr", "fit_time_sec",
        "validation_score_time_sec", "test_score_time_sec", "total_time_sec",
        "validation_records_per_sec", "test_records_per_sec",
        "peak_memory_increase_mb", "serialized_model_bytes",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(groups, dropna=False, sort=False):
        row = dict(zip(groups, keys))
        row["approach"] = "Baseline"
        row["runs"] = int(len(group))
        row["seeds"] = "|".join(map(str, sorted(group["seed"].astype(int).unique())))
        for metric in metrics:
            summary = _mean_std_ci(pd.to_numeric(group[metric], errors="coerce").dropna().tolist(), confidence)
            for suffix, value in summary.items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_baseline_categories(
    categories: pd.DataFrame, *, confidence: float = 0.95
) -> pd.DataFrame:
    if categories.empty:
        return pd.DataFrame()
    groups = [
        "dataset_key", "dataset", "model", "model_name", "feature_set",
        "fs_size", "target_fpr", "category",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in categories.groupby(groups, dropna=False, sort=False):
        row = dict(zip(groups, keys))
        row["runs"] = int(len(group))
        for metric in ("category_recall", "false_alarm_rate", "detected_attacks", "missed_attacks", "false_alarms"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().tolist()
            summary = _mean_std_ci(values, confidence)
            for suffix, value in summary.items():
                row[f"{metric}_{suffix}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run_confirmatory_baselines_dataset(
    dataset_key: str,
    manuscript_config: Mapping[str, Any],
    *,
    confirmatory_results_root: str | Path,
    output_root: str | Path | None = None,
    confirmatory_config_path: str | Path | None = None,
    dataset_path_overrides: Mapping[str, str | Path] | None = None,
    max_model_fits: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ConfirmatoryBaselineOutputs:
    """Run or resume all fixed classical baselines for one dataset."""

    baseline_config = manuscript_config["baseline"]
    if dataset_key not in baseline_config["datasets"]:
        raise ValueError(f"Dataset '{dataset_key}' is not enabled in this profile.")

    confirmatory_path = Path(
        confirmatory_config_path or manuscript_config["paths"]["confirmatory_config"]
    )
    confirmatory_config = load_confirmatory_config(confirmatory_path, profile_name="confirmatory")
    overrides = dict(dataset_path_overrides or {})
    if dataset_key == "nsl_kdd":
        if overrides.get("nsl_train") is not None:
            confirmatory_config["datasets"]["nsl_kdd"]["train_file"] = str(overrides["nsl_train"])
        if overrides.get("nsl_test") is not None:
            confirmatory_config["datasets"]["nsl_kdd"]["test_file"] = str(overrides["nsl_test"])
    elif dataset_key == "cicids2017":
        if overrides.get("cic_raw_dir") is not None:
            confirmatory_config["datasets"]["cicids2017"]["raw_dir"] = str(overrides["cic_raw_dir"])
        if overrides.get("cic_archive") is not None:
            confirmatory_config["datasets"]["cicids2017"]["archive_file"] = str(overrides["cic_archive"])
    dataset = prepare_confirmatory_dataset(dataset_key, confirmatory_config)
    feature_sets = load_locked_feature_sets(confirmatory_results_root, dataset_key)
    contract = verify_confirmatory_contract(
        dataset_key, dataset, feature_sets, confirmatory_results_root
    )
    dataset_fingerprints = verify_dataset_fingerprints(
        dataset_key, confirmatory_config, contract["manifest"]
    )

    root = ensure_dir(output_root or manuscript_config["paths"]["manuscript_results_dir"])
    dataset_root = ensure_dir(root / "confirmatory_baselines" / dataset_key)
    tables_dir = ensure_dir(dataset_root / "tables")
    progress_dir = ensure_dir(dataset_root / "progress")
    manifests_dir = ensure_dir(dataset_root / "manifests")
    journal = progress_dir / "baseline_rows.jsonl"
    category_journal = progress_dir / "baseline_category_rows.jsonl"

    existing = load_jsonl_frame(journal)
    rows = existing.to_dict(orient="records") if not existing.empty else []
    completed = {_result_key(row) for row in rows}
    existing_categories = load_jsonl_frame(category_journal)
    category_rows = existing_categories.to_dict(orient="records") if not existing_categories.empty else []
    category_completed = {
        (
            str(row["dataset_key"]),
            str(row["model"]),
            int(row["fs_size"]),
            int(row["seed"]),
            round(float(row["target_fpr"]), 12),
        )
        for row in category_rows
        if all(key in row for key in ["dataset_key", "model", "fs_size", "seed", "target_fpr"])
    }

    specs = list(_iter_specs(manuscript_config, dataset_key))
    if max_model_fits is not None:
        specs = specs[: max(0, int(max_model_fits))]
    total_result_rows = len(specs) * len(baseline_config["target_fprs"])

    for fit_index, spec in enumerate(specs, start=1):
        pending_targets = []
        for target in baseline_config["target_fprs"]:
            target_value = float(target)
            result_missing = _result_key({**spec, "target_fpr": target_value}) not in completed
            category_identity = (
                dataset_key,
                str(spec["model"]),
                int(spec["fs_size"]),
                int(spec["seed"]),
                round(target_value, 12),
            )
            category_missing = category_identity not in category_completed
            if result_missing or category_missing:
                pending_targets.append(target_value)
        if not pending_targets:
            continue

        fs_size = int(spec["fs_size"])
        feature_set = feature_sets[fs_size]
        selected = list(feature_set.selected_features)
        X_train = dataset.X_train.loc[:, selected]
        X_validation = dataset.X_validation.loc[:, selected]
        X_test = dataset.X_test.loc[:, selected]
        model_name = str(spec["model"])
        settings = baseline_config["models"][model_name]
        model = build_baseline_model(
            model_name, settings.get("parameters", {}), seed=int(spec["seed"])
        )
        validation_scores, test_scores, resources = _fit_and_score(
            model_name,
            model,
            X_train,
            dataset.y_train,
            X_validation,
            X_test,
        )
        parameters = dict(model.get_params(deep=False))
        parameters_json = json.dumps(parameters, sort_keys=True, default=str)
        model_hash = stable_mapping_signature(parameters)

        for target_fpr in pending_targets:
            calibration = calibrate_detection_threshold(
                validation_scores,
                dataset.y_validation,
                method="weighted_similarity",
                target_fpr=target_fpr,
            )
            validation_predictions = (validation_scores >= calibration.threshold).astype(np.int8)
            test_predictions = (test_scores >= calibration.threshold).astype(np.int8)
            validation_metrics = evaluate_binary_classification(
                dataset.y_validation, validation_predictions, y_score=validation_scores
            )
            test_metrics = evaluate_binary_classification(
                dataset.y_test, test_predictions, y_score=test_scores
            )
            row = {
                "baseline_protocol_version": BASELINE_PROTOCOL_VERSION,
                "profile": manuscript_config["profile_name"],
                "dataset_key": dataset_key,
                "dataset": dataset.metadata.get("dataset", dataset_key),
                "approach": "Baseline",
                "model": model_name,
                "model_name": BASELINE_MODEL_NAMES[model_name],
                "model_family": BASELINE_MODEL_FAMILIES[model_name],
                "feature_set": f"FS-{fs_size}",
                "fs_size": fs_size,
                "seed": int(spec["seed"]),
                "target_fpr": target_fpr,
                "feature_selection_hash": feature_set.feature_selection_hash,
                "selected_features": "|".join(selected),
                "model_parameters_json": parameters_json,
                "model_configuration_hash": model_hash,
                **contract["partition_hashes"],
                **calibration.to_dict(),
                **{f"validation_{key}": value for key, value in validation_metrics.to_dict().items()},
                **{f"test_{key}": value for key, value in test_metrics.to_dict().items()},
                **resources,
                "configuration_selected_on_validation": False,
                "threshold_selected_on_validation": True,
                "test_metrics_used_for_selection": False,
                "test_evaluations_for_seed": 1,
            }
            result_identity = _result_key(row)
            if result_identity not in completed:
                rows.append(row)
                completed.add(result_identity)
                append_jsonl(row, journal)

            category_identity = (
                dataset_key, model_name, fs_size, int(spec["seed"]), round(target_fpr, 12)
            )
            if dataset.test_attack_categories is not None and category_identity not in category_completed:
                category = attack_category_analysis(
                    dataset.y_test,
                    test_predictions,
                    dataset.test_attack_categories,
                    run_metadata={
                        "dataset_key": dataset_key,
                        "dataset": dataset.metadata.get("dataset", dataset_key),
                        "model": model_name,
                        "model_name": BASELINE_MODEL_NAMES[model_name],
                        "feature_set": f"FS-{fs_size}",
                        "fs_size": fs_size,
                        "seed": int(spec["seed"]),
                        "target_fpr": target_fpr,
                        **contract["partition_hashes"],
                    },
                )
                for category_row in category.to_dict(orient="records"):
                    category_rows.append(category_row)
                    append_jsonl(category_row, category_journal)
                category_completed.add(category_identity)

            if progress_callback is not None:
                progress_callback(len(rows), total_result_rows, row)

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["model", "fs_size", "target_fpr", "seed"]).reset_index(drop=True)
    categories = pd.DataFrame(category_rows)
    if not categories.empty:
        categories = categories.drop_duplicates(
            ["dataset_key", "model", "fs_size", "seed", "target_fpr", "category"],
            keep="last",
        ).sort_values(["model", "fs_size", "target_fpr", "seed", "category"]).reset_index(drop=True)

    confidence = float(manuscript_config["statistics"].get("confidence", 0.95))
    summary = summarize_confirmatory_baselines(results, confidence=confidence)
    category_summary = summarize_baseline_categories(categories, confidence=confidence)
    selected_rows: list[dict[str, Any]] = []
    for size, feature_set in sorted(feature_sets.items()):
        for rank, feature in enumerate(feature_set.selected_features, start=1):
            selected_rows.append({
                "dataset_key": dataset_key,
                "feature_set": f"FS-{size}",
                "fs_size": size,
                "rank": rank,
                "feature": feature,
                "normalized_weight": float(feature_set.weights[rank - 1]),
                "feature_selection_hash": feature_set.feature_selection_hash,
            })
    selected_features = pd.DataFrame(selected_rows)

    output_paths = {
        "seed_results": atomic_write_csv(results, tables_dir / "baseline_seed_results.csv"),
        "summary": atomic_write_csv(summary, tables_dir / "baseline_summary.csv"),
        "category_seed_results": atomic_write_csv(categories, tables_dir / "baseline_category_seed_results.csv"),
        "category_summary": atomic_write_csv(category_summary, tables_dir / "baseline_category_summary.csv"),
        "selected_features": atomic_write_csv(selected_features, tables_dir / "selected_features.csv"),
    }
    expected = len(list(_iter_specs(manuscript_config, dataset_key))) * len(baseline_config["target_fprs"])
    manifest = {
        "baseline_protocol_version": BASELINE_PROTOCOL_VERSION,
        "profile": manuscript_config["profile_name"],
        "dataset_key": dataset_key,
        "confirmatory_contract": {
            "source_manifest": contract["manifest"],
            "partition_hashes": contract["partition_hashes"],
            "feature_selection_hashes": contract["feature_selection_hashes"],
            "dataset_file_sha256": dataset_fingerprints,
        },
        "baseline_config_hash": stable_mapping_signature(baseline_config),
        "execution_environment": environment_manifest(Path.cwd()),
        "completed_rows": int(len(results)),
        "expected_rows": int(expected),
        "test_metrics_used_for_selection": False,
    }
    manifest_path = manifests_dir / "execution_manifest.json"
    atomic_write_json(manifest, manifest_path)
    output_paths["manifest"] = manifest_path

    return ConfirmatoryBaselineOutputs(
        seed_results=results,
        summary=summary,
        category_seed_results=categories,
        category_summary=category_summary,
        selected_features=selected_features,
        manifest=manifest,
        output_paths=output_paths,
    )
