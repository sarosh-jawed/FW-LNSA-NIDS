"""Configuration loading and validation for FW-LNSA experiments.

The repository keeps experiment choices in YAML so every published table can be
traced to a concrete configuration. This module intentionally validates the
important research controls before a long run starts.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

SUPPORTED_METHODS = {"hamming", "weighted_hamming", "weighted_smc", "jaccard"}
DISTANCE_METHODS = {"hamming", "weighted_hamming"}
SIMILARITY_METHODS = {"weighted_smc", "jaccard"}


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
    validate_nsl_kdd_config(resolved)
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


def validate_nsl_kdd_config(config: Mapping[str, Any]) -> None:
    """Validate settings required by the NSL-KDD FW-LNSA runner."""

    dataset = _require_mapping(config, "dataset")
    if str(dataset.get("name", "")).lower() != "nsl_kdd":
        raise ConfigurationError("dataset.name must be 'nsl_kdd'.")

    paths = _require_mapping(config, "paths")
    for path_key in ("train_file", "test_file", "output_dir"):
        if not paths.get(path_key):
            raise ConfigurationError(f"paths.{path_key} is required.")

    preprocessing = _require_mapping(config, "preprocessing")
    max_self_samples = preprocessing.get("max_self_samples")
    if max_self_samples is not None and (not isinstance(max_self_samples, int) or max_self_samples <= 0):
        raise ConfigurationError("preprocessing.max_self_samples must be a positive integer or null.")

    feature_selection = _require_mapping(config, "feature_selection")
    feature_sizes = _require_nonempty_list(feature_selection, "feature_sizes")
    if any(not isinstance(size, int) or size <= 0 for size in feature_sizes):
        raise ConfigurationError("feature_selection.feature_sizes must contain positive integers.")

    main_size = feature_selection.get("main_feature_size")
    if main_size not in feature_sizes:
        raise ConfigurationError("feature_selection.main_feature_size must be listed in feature_sizes.")

    experiment = _require_mapping(config, "experiment")
    methods = _require_nonempty_list(experiment, "methods")
    unknown_methods = set(methods) - SUPPORTED_METHODS
    if unknown_methods:
        raise ConfigurationError(f"Unsupported methods: {sorted(unknown_methods)}")

    budgets = _require_nonempty_list(experiment, "detector_budgets")
    if any(not isinstance(budget, int) or budget <= 0 for budget in budgets):
        raise ConfigurationError("experiment.detector_budgets must contain positive integers.")

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
            raise ConfigurationError(f"method_settings.{method}.self_thresholds must be non-empty.")
        if not isinstance(detection_thresholds, list) or not detection_thresholds:
            raise ConfigurationError(
                f"method_settings.{method}.detection_thresholds must be non-empty."
            )
        _validate_probability_list(self_thresholds, f"method_settings.{method}.self_thresholds")
        _validate_probability_list(
            detection_thresholds,
            f"method_settings.{method}.detection_thresholds",
        )

        threshold_scale = str(settings.get("threshold_scale", "direct"))
        if method == "hamming" and threshold_scale != "ratio":
            raise ConfigurationError("Hamming thresholds must use threshold_scale: ratio.")
        if method != "hamming" and threshold_scale != "direct":
            raise ConfigurationError(f"{method} thresholds must use threshold_scale: direct.")

    selection = _require_mapping(config, "selection")
    fpr_limit = selection.get("balanced_fpr_limit")
    if not isinstance(fpr_limit, (int, float)) or not 0.0 <= float(fpr_limit) <= 1.0:
        raise ConfigurationError("selection.balanced_fpr_limit must be between 0 and 1.")

    outputs = _require_mapping(config, "outputs")
    required_outputs = {
        "results_table",
        "method_summary",
        "balanced_configs",
        "attack_category_analysis",
        "selected_features",
    }
    missing_outputs = sorted(key for key in required_outputs if not outputs.get(key))
    if missing_outputs:
        raise ConfigurationError(f"Missing output filenames: {missing_outputs}")
