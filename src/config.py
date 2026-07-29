"""Configuration loading and validation for FW-LNSA experiments.

Experiment choices live in YAML so every saved table can be traced to a concrete
configuration. Validation runs before dataset loading or model fitting, which
prevents long research jobs from failing because of an avoidable typo.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

SUPPORTED_METHODS = {
    "hamming",
    "weighted_hamming",
    "weighted_similarity",
    "weighted_smc",
    "jaccard",
}
DISTANCE_METHODS = {"hamming", "weighted_hamming"}
SIMILARITY_METHODS = {"weighted_similarity", "weighted_smc", "jaccard"}


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the document root."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ConfigurationError("The configuration root must be a YAML mapping.")
    return loaded


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_profile(config: Mapping[str, Any], profile_name: str | None = None) -> dict[str, Any]:
    """Return a complete configuration with one execution profile applied."""

    base = {key: value for key, value in config.items() if key != "profiles"}
    profiles = config.get("profiles", {})
    selected_name = profile_name or str(base.get("active_profile", "smoke"))

    if not isinstance(profiles, Mapping) or selected_name not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles)) or "none"
        raise ConfigurationError(
            f"Unknown execution profile '{selected_name}'. Available profiles: {available}."
        )
    selected = profiles[selected_name]
    if not isinstance(selected, Mapping):
        raise ConfigurationError(f"Profile '{selected_name}' must be a mapping.")

    resolved = deep_merge(base, selected)
    resolved["profile_name"] = selected_name

    dataset = _require_mapping(resolved, "dataset")
    dataset_name = str(dataset.get("name", "")).lower()
    if dataset_name == "nsl_kdd":
        validate_nsl_kdd_config(resolved)
    elif dataset_name == "cicids2017":
        validate_cicids2017_config(resolved)
    else:
        raise ConfigurationError(
            "dataset.name must be either 'nsl_kdd' or 'cicids2017'."
        )
    return resolved


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{key}' must be a mapping.")
    return value


def _require_nonempty_list(config: Mapping[str, Any], key: str) -> list[Any]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"'{key}' must be a non-empty list.")
    return value


def _validate_probability_list(values: list[Any], name: str) -> None:
    for value in values:
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ConfigurationError(f"All values in '{name}' must be between 0 and 1.")


def _validate_common_experiment_config(config: Mapping[str, Any]) -> None:
    preprocessing = _require_mapping(config, "preprocessing")
    max_self_samples = preprocessing.get("max_self_samples")
    if max_self_samples is not None and (
        not isinstance(max_self_samples, int) or max_self_samples <= 0
    ):
        raise ConfigurationError(
            "preprocessing.max_self_samples must be a positive integer or null."
        )

    feature_selection = _require_mapping(config, "feature_selection")
    feature_sizes = _require_nonempty_list(feature_selection, "feature_sizes")
    if any(not isinstance(size, int) or size <= 0 for size in feature_sizes):
        raise ConfigurationError(
            "feature_selection.feature_sizes must contain positive integers."
        )
    main_size = feature_selection.get("main_feature_size")
    if main_size not in feature_sizes:
        raise ConfigurationError(
            "feature_selection.main_feature_size must be listed in feature_sizes."
        )

    experiment = _require_mapping(config, "experiment")
    methods = _require_nonempty_list(experiment, "methods")
    unknown_methods = set(methods) - SUPPORTED_METHODS
    if unknown_methods:
        raise ConfigurationError(f"Unsupported methods: {sorted(unknown_methods)}")

    budgets = _require_nonempty_list(experiment, "detector_budgets")
    if any(not isinstance(budget, int) or budget <= 0 for budget in budgets):
        raise ConfigurationError(
            "experiment.detector_budgets must contain positive integers."
        )
    seeds = _require_nonempty_list(experiment, "seeds")
    if any(not isinstance(seed, int) for seed in seeds):
        raise ConfigurationError("experiment.seeds must contain integers.")

    method_settings = _require_mapping(config, "method_settings")
    for method in methods:
        settings = method_settings.get(method)
        if not isinstance(settings, Mapping):
            raise ConfigurationError(f"method_settings.{method} is required.")
        self_thresholds = settings.get("self_thresholds")
        detection_thresholds = settings.get("detection_thresholds")
        if not isinstance(self_thresholds, list) or not self_thresholds:
            raise ConfigurationError(
                f"method_settings.{method}.self_thresholds must be non-empty."
            )
        if not isinstance(detection_thresholds, list) or not detection_thresholds:
            raise ConfigurationError(
                f"method_settings.{method}.detection_thresholds must be non-empty."
            )
        _validate_probability_list(
            self_thresholds,
            f"method_settings.{method}.self_thresholds",
        )
        _validate_probability_list(
            detection_thresholds,
            f"method_settings.{method}.detection_thresholds",
        )
        threshold_scale = str(settings.get("threshold_scale", "direct"))
        if method == "hamming" and threshold_scale != "ratio":
            raise ConfigurationError("Hamming thresholds must use threshold_scale: ratio.")
        if method != "hamming" and threshold_scale != "direct":
            raise ConfigurationError(
                f"{method} thresholds must use threshold_scale: direct."
            )

    selection = _require_mapping(config, "selection")
    fpr_limit = selection.get("balanced_fpr_limit")
    if not isinstance(fpr_limit, (int, float)) or not 0.0 <= float(fpr_limit) <= 1.0:
        raise ConfigurationError(
            "selection.balanced_fpr_limit must be between 0 and 1."
        )


def _validate_outputs(config: Mapping[str, Any], required: set[str]) -> None:
    outputs = _require_mapping(config, "outputs")
    missing = sorted(key for key in required if not outputs.get(key))
    if missing:
        raise ConfigurationError(f"Missing output filenames: {missing}")


def validate_nsl_kdd_config(config: Mapping[str, Any]) -> None:
    """Validate settings required by the NSL-KDD FW-LNSA runner."""

    dataset = _require_mapping(config, "dataset")
    if str(dataset.get("name", "")).lower() != "nsl_kdd":
        raise ConfigurationError("dataset.name must be 'nsl_kdd'.")

    paths = _require_mapping(config, "paths")
    for path_key in ("train_file", "test_file", "output_dir"):
        if not paths.get(path_key):
            raise ConfigurationError(f"paths.{path_key} is required.")

    _validate_common_experiment_config(config)
    _validate_outputs(
        config,
        {
            "results_table",
            "method_summary",
            "balanced_configs",
            "attack_category_analysis",
            "selected_features",
        },
    )


def validate_cicids2017_config(config: Mapping[str, Any]) -> None:
    """Validate settings required by the CICIDS2017 FW-LNSA runner."""

    dataset = _require_mapping(config, "dataset")
    if str(dataset.get("name", "")).lower() != "cicids2017":
        raise ConfigurationError("dataset.name must be 'cicids2017'.")

    paths = _require_mapping(config, "paths")
    if not paths.get("raw_dir"):
        raise ConfigurationError("paths.raw_dir is required.")
    if not paths.get("output_dir"):
        raise ConfigurationError("paths.output_dir is required.")

    preprocessing = _require_mapping(config, "preprocessing")
    chunk_size = preprocessing.get("chunk_size")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ConfigurationError("preprocessing.chunk_size must be a positive integer.")
    max_chunks = preprocessing.get("max_chunks_per_file")
    if max_chunks is not None and (not isinstance(max_chunks, int) or max_chunks <= 0):
        raise ConfigurationError(
            "preprocessing.max_chunks_per_file must be a positive integer or null."
        )
    if not preprocessing.get("label_column"):
        raise ConfigurationError("preprocessing.label_column is required.")
    if not preprocessing.get("benign_label"):
        raise ConfigurationError("preprocessing.benign_label is required.")

    sampling = _require_mapping(preprocessing, "sampling")
    strategy = sampling.get("strategy", "per_label_cap")
    if strategy not in {"per_label_cap", "global_cap"}:
        raise ConfigurationError(
            "preprocessing.sampling.strategy must be 'per_label_cap' or 'global_cap'."
        )
    for key in ("max_benign_records", "max_records_per_attack_label"):
        value = sampling.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ConfigurationError(f"preprocessing.sampling.{key} must be positive.")
    if strategy == "per_label_cap":
        for key in ("max_benign_records", "max_records_per_attack_label"):
            if not isinstance(sampling.get(key), int) or sampling.get(key) <= 0:
                raise ConfigurationError(
                    f"preprocessing.sampling.{key} is required for per_label_cap sampling."
                )
    if strategy == "global_cap":
        total = sampling.get("max_total_records")
        if not isinstance(total, int) or total <= 0:
            raise ConfigurationError(
                "preprocessing.sampling.max_total_records is required for global_cap sampling."
            )
    if not isinstance(sampling.get("random_seed"), int):
        raise ConfigurationError("preprocessing.sampling.random_seed must be an integer.")

    split = _require_mapping(preprocessing, "split")
    test_size = split.get("test_size")
    if not isinstance(test_size, (int, float)) or not 0.0 < float(test_size) < 1.0:
        raise ConfigurationError("preprocessing.split.test_size must be between 0 and 1.")
    if not isinstance(split.get("random_seed"), int):
        raise ConfigurationError("preprocessing.split.random_seed must be an integer.")
    if split.get("stratify_by") not in {"original_label", "binary"}:
        raise ConfigurationError(
            "preprocessing.split.stratify_by must be 'original_label' or 'binary'."
        )

    _validate_common_experiment_config(config)
    _validate_outputs(
        config,
        {
            "results_table",
            "method_summary",
            "balanced_configs",
            "attack_category_analysis",
            "selected_features",
            "data_quality_report",
        },
    )

SUPPORTED_BASELINE_MODELS = {
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "isolation_forest",
}


def resolve_baseline_profile(
    config: Mapping[str, Any],
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Apply one baseline execution profile and validate the result."""

    base = {key: value for key, value in config.items() if key != "profiles"}
    profiles = config.get("profiles", {})
    selected_name = profile_name or str(base.get("active_profile", "smoke"))

    if not isinstance(profiles, Mapping) or selected_name not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles)) or "none"
        raise ConfigurationError(
            f"Unknown baseline profile '{selected_name}'. Available profiles: {available}."
        )
    selected = profiles[selected_name]
    if not isinstance(selected, Mapping):
        raise ConfigurationError(f"Baseline profile '{selected_name}' must be a mapping.")

    resolved = deep_merge(base, selected)
    resolved["profile_name"] = selected_name
    validate_baseline_config(resolved)
    return resolved


def validate_baseline_config(config: Mapping[str, Any]) -> None:
    """Validate settings required by the baseline experiment runner."""

    datasets = _require_nonempty_list(config, "datasets")
    invalid_datasets = sorted(set(map(str, datasets)) - {"nsl_kdd", "cicids2017"})
    if invalid_datasets:
        raise ConfigurationError(
            f"Unsupported baseline datasets: {', '.join(invalid_datasets)}."
        )

    dataset_configs = _require_mapping(config, "dataset_configs")
    dataset_profiles = _require_mapping(config, "dataset_profiles")
    for dataset_name in datasets:
        if not dataset_configs.get(dataset_name):
            raise ConfigurationError(
                f"dataset_configs.{dataset_name} must reference a dataset YAML file."
            )
        if not dataset_profiles.get(dataset_name):
            raise ConfigurationError(
                f"dataset_profiles.{dataset_name} must name a dataset execution profile."
            )

    feature_selection = _require_mapping(config, "feature_selection")
    feature_sizes = feature_selection.get("feature_sizes")
    if not isinstance(feature_sizes, list) or not feature_sizes:
        raise ConfigurationError("feature_selection.feature_sizes must be a non-empty list.")
    if any(not isinstance(value, int) or value <= 0 for value in feature_sizes):
        raise ConfigurationError("All baseline feature sizes must be positive integers.")
    main_size = feature_selection.get("main_feature_size")
    if main_size not in feature_sizes:
        raise ConfigurationError(
            "feature_selection.main_feature_size must appear in feature_sizes."
        )
    if not isinstance(feature_selection.get("random_seed"), int):
        raise ConfigurationError("feature_selection.random_seed must be an integer.")
    inherit_sampling = feature_selection.get("inherit_dataset_sampling", True)
    if not isinstance(inherit_sampling, bool):
        raise ConfigurationError(
            "feature_selection.inherit_dataset_sampling must be true or false."
        )
    max_samples = feature_selection.get("max_samples")
    if max_samples is not None and (not isinstance(max_samples, int) or max_samples <= 0):
        raise ConfigurationError("feature_selection.max_samples must be positive or null.")

    experiment = _require_mapping(config, "experiment")
    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ConfigurationError("experiment.seeds must be a non-empty list of integers.")

    models = _require_mapping(config, "models")
    enabled_models: list[str] = []
    for model_name, model_config in models.items():
        if model_name not in SUPPORTED_BASELINE_MODELS:
            raise ConfigurationError(f"Unsupported baseline model '{model_name}'.")
        if not isinstance(model_config, Mapping):
            raise ConfigurationError(f"models.{model_name} must be a mapping.")
        if bool(model_config.get("enabled", False)):
            enabled_models.append(model_name)
        parameters = model_config.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ConfigurationError(f"models.{model_name}.parameters must be a mapping.")
    if not enabled_models:
        raise ConfigurationError("At least one baseline model must be enabled.")

    selection = _require_mapping(config, "selection")
    fpr_limit = selection.get("balanced_fpr_limit")
    if not isinstance(fpr_limit, (int, float)) or not 0.0 <= float(fpr_limit) <= 1.0:
        raise ConfigurationError("selection.balanced_fpr_limit must be between 0 and 1.")

    outputs = _require_mapping(config, "outputs")
    for dataset_name in datasets:
        dataset_outputs = outputs.get(dataset_name)
        if not isinstance(dataset_outputs, Mapping):
            raise ConfigurationError(f"outputs.{dataset_name} must be a mapping.")
        required = {
            "baseline_results",
            "comparison_table",
            "attack_category_analysis",
        }
        missing = sorted(required - set(dataset_outputs))
        if missing:
            raise ConfigurationError(
                f"outputs.{dataset_name} is missing: {', '.join(missing)}."
            )
