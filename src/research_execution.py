"""Resumable research execution for NSL-KDD, CICIDS2017, and baselines.

The execution layer stores one row after every completed run. A Colab restart can
therefore continue from the exact remaining configuration instead of repeating
the full grid. Fingerprints prevent accidental resume with changed data or YAML.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import yaml
from joblib import dump as joblib_dump
from joblib import load as joblib_load

from .baselines import (
    BaselineOutputs,
    iter_baseline_run_specs,
    run_prepared_baseline_experiments,
)
from .config import load_yaml_config, resolve_baseline_profile, resolve_profile
from .experiments import (
    ExperimentOutputs,
    iter_fw_run_specs,
    run_prepared_fw_lnsa_experiments,
)
from .preprocessing import PreparedDataset, prepare_cicids2017, prepare_nsl_kdd
from .statistical_analysis import ResearchAnalysisOutputs, analyze_research_results
from .utils import ensure_dir

SUPPORTED_STAGES = (
    "nsl_kdd_fw_lnsa",
    "cicids2017_fw_lnsa",
    "nsl_kdd_baselines",
    "cicids2017_baselines",
    "analysis",
)


@dataclass(frozen=True)
class StageResult:
    """Result metadata returned after one research stage."""

    stage: str
    status: str
    completed_runs: int
    expected_runs: int
    output_paths: dict[str, Path]
    state_path: Path
    manifest_path: Path


def build_execution_plan(
    *,
    execution_config: Mapping[str, Any],
    profile: str,
    repo_root: str | Path,
    stages: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return exact configured run counts without loading either dataset."""

    root = Path(repo_root).resolve()
    selected = list(stages or execution_config["execution"]["stages"])
    rows: list[dict[str, Any]] = []
    dataset_configs: dict[str, dict[str, Any]] = {}
    for dataset_name in ("nsl_kdd", "cicids2017"):
        dataset_configs[dataset_name] = _resolve_dataset_config(
            execution_config,
            dataset_name=dataset_name,
            profile=profile,
            repo_root=root,
            output_dir=root / "results",
            path_overrides=None,
        )

    for stage in SUPPORTED_STAGES:
        if stage not in selected:
            continue
        if stage.endswith("_fw_lnsa"):
            dataset_name = stage.removesuffix("_fw_lnsa")
            config = dataset_configs[dataset_name]
            specs = list(iter_fw_run_specs(config))
            detector_pool_groups = {
                (
                    spec["profile"],
                    spec["method"],
                    spec["fs_size"],
                    spec["seed"],
                    spec["detector_budget"],
                    round(float(spec["self_threshold_config"]), 12),
                )
                for spec in specs
            }
            rows.append(
                {
                    "stage": stage,
                    "profile": profile,
                    "expected_runs": len(specs),
                    "unique_model_fits": len(detector_pool_groups),
                    "feature_sizes": "|".join(
                        map(str, config["feature_selection"]["feature_sizes"])
                    ),
                    "seeds": "|".join(map(str, config["experiment"]["seeds"])),
                    "methods_or_models": "|".join(config["experiment"]["methods"]),
                }
            )
        elif stage.endswith("_baselines"):
            dataset_name = stage.removesuffix("_baselines")
            baseline = _resolve_baseline_config(
                execution_config,
                profile=profile,
                repo_root=root,
                dataset_config=dataset_configs[dataset_name],
            )
            enabled = [
                name
                for name, value in baseline["models"].items()
                if bool(value.get("enabled", False))
            ]
            specs = list(iter_baseline_run_specs(baseline, dataset_key=dataset_name))
            rows.append(
                {
                    "stage": stage,
                    "profile": profile,
                    "expected_runs": len(specs),
                    "unique_model_fits": len(specs),
                    "feature_sizes": "|".join(
                        map(str, baseline["feature_selection"]["feature_sizes"])
                    ),
                    "seeds": "|".join(map(str, baseline["experiment"]["seeds"])),
                    "methods_or_models": "|".join(enabled),
                }
            )
        else:
            rows.append(
                {
                    "stage": stage,
                    "profile": profile,
                    "expected_runs": 1,
                    "unique_model_fits": 0,
                    "feature_sizes": "",
                    "seeds": "",
                    "methods_or_models": "saved-result analysis",
                }
            )
    return pd.DataFrame(rows)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_research_execution_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the repository-level research execution YAML."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Research execution configuration not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Research execution configuration must be a YAML mapping.")

    required = {"dataset_configs", "baseline_config", "execution", "analysis"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Research execution configuration is missing: {sorted(missing)}")
    configured_stages = config["execution"].get("stages", [])
    unknown = set(configured_stages) - set(SUPPORTED_STAGES)
    if unknown:
        raise ValueError(f"Unsupported research execution stages: {sorted(unknown)}")
    return config


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def mapping_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a dataset or configuration file without loading it into memory."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "PyYAML", "psutil"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_commit(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def environment_manifest(repo_root: Path) -> dict[str, Any]:
    """Capture enough environment metadata to audit a research run."""

    return {
        "captured_utc": utc_now(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_versions": _package_versions(),
        "git_commit": _git_commit(repo_root),
    }


def atomic_write_json(value: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(destination)
    return destination


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)
    return destination


def append_jsonl(record: Mapping[str, Any], path: str | Path) -> Path:
    """Append one durable progress record without rewriting prior rows."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True, default=str, separators=(",", ":"))
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def load_jsonl_frame(path: str | Path) -> pd.DataFrame:
    """Load progress records and ignore only an interrupted final line."""

    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return pd.DataFrame()

    lines = source.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(
                f"Progress journal contains an invalid record at line {index + 1}: {source}"
            )
        if not isinstance(value, dict):
            raise ValueError(
                f"Progress journal record {index + 1} is not an object: {source}"
            )
        records.append(value)
    return pd.DataFrame(records)


def _dataset_files(dataset_name: str, config: Mapping[str, Any]) -> list[Path]:
    paths = config["paths"]
    if dataset_name == "nsl_kdd":
        return [Path(paths["train_file"]), Path(paths["test_file"])]
    archive = paths.get("archive_file")
    if archive and Path(archive).exists():
        return [Path(archive)]
    raw_dir = Path(paths["raw_dir"])
    return sorted(raw_dir.rglob("*.csv"))


def _dataset_fingerprint(dataset_name: str, config: Mapping[str, Any]) -> dict[str, str]:
    files = _dataset_files(dataset_name, config)
    if not files:
        raise FileNotFoundError(f"No dataset files were found for {dataset_name}.")
    return {str(path.resolve()): file_sha256(path) for path in files}


def _prepare_dataset(dataset_name: str, config: Mapping[str, Any]) -> PreparedDataset:
    paths = config["paths"]
    preprocessing = config["preprocessing"]
    if dataset_name == "nsl_kdd":
        return prepare_nsl_kdd(
            paths["train_file"],
            paths["test_file"],
            scale=bool(preprocessing.get("scale", True)),
        )

    sampling = preprocessing["sampling"]
    split = preprocessing["split"]
    return prepare_cicids2017(
        paths["raw_dir"],
        archive_file=paths.get("archive_file"),
        label_column=str(preprocessing.get("label_column", "Label")),
        benign_label=str(preprocessing.get("benign_label", "BENIGN")),
        chunk_size=int(preprocessing.get("chunk_size", 25000)),
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


def _prepare_dataset_cached(
    *,
    dataset_name: str,
    config: Mapping[str, Any],
    output_dir: Path,
    execution_config: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
) -> tuple[PreparedDataset, Path, bool]:
    """Load or build a persistent prepared dataset for repeated Colab stages."""

    execution = execution_config["execution"]
    cache_dir = ensure_dir(output_dir / str(execution.get("cache_dir", "cache")))
    source_paths = {
        key: config["paths"].get(key)
        for key in ("train_file", "test_file", "raw_dir", "archive_file")
        if config["paths"].get(key)
    }
    cache_fingerprint = mapping_hash(
        {
            "dataset_name": dataset_name,
            "dataset": config.get("dataset", {}),
            "preprocessing": config.get("preprocessing", {}),
            "source_paths": source_paths,
            "dataset_hashes": dict(dataset_hashes),
        }
    )
    cache_path = cache_dir / f"{dataset_name}_{cache_fingerprint[:16]}.joblib"
    metadata_path = cache_path.with_suffix(".json")

    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == cache_fingerprint:
            loaded = joblib_load(cache_path)
            if not isinstance(loaded, PreparedDataset):
                raise TypeError(f"Prepared dataset cache is invalid: {cache_path}")
            return loaded, cache_path, True

    dataset = _prepare_dataset(dataset_name, config)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    joblib_dump(dataset, temporary, compress=3)
    temporary.replace(cache_path)
    atomic_write_json(
        {
            "dataset_name": dataset_name,
            "fingerprint": cache_fingerprint,
            "created_utc": utc_now(),
            "records": {
                "train": int(len(dataset.X_train)),
                "test": int(len(dataset.X_test)),
                "features": int(dataset.X_train.shape[1]),
            },
            "dataset_hashes": dict(dataset_hashes),
        },
        metadata_path,
    )
    return dataset, cache_path, False


def _resolve_dataset_config(
    execution_config: Mapping[str, Any],
    *,
    dataset_name: str,
    profile: str,
    repo_root: Path,
    output_dir: Path,
    path_overrides: Mapping[str, str | Path] | None,
) -> dict[str, Any]:
    config_path = _resolve_path(execution_config["dataset_configs"][dataset_name], repo_root)
    config = resolve_profile(load_yaml_config(config_path), profile)
    config["paths"]["output_dir"] = str(output_dir)

    if path_overrides:
        for key, value in path_overrides.items():
            if value is not None:
                config["paths"][key] = str(value)

    for key in ("train_file", "test_file", "raw_dir", "archive_file"):
        value = config["paths"].get(key)
        if value:
            config["paths"][key] = str(_resolve_path(value, repo_root))
    return config


def _resolve_baseline_config(
    execution_config: Mapping[str, Any],
    *,
    profile: str,
    repo_root: Path,
    dataset_config: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_path = _resolve_path(execution_config["baseline_config"], repo_root)
    baseline = resolve_baseline_profile(load_yaml_config(baseline_path), profile)
    aligned = deepcopy(baseline)
    if bool(aligned["feature_selection"].get("inherit_dataset_sampling", True)):
        dataset_features = dataset_config["feature_selection"]
        aligned["feature_selection"]["random_seed"] = int(
            dataset_features.get("random_seed", 42)
        )
        aligned["feature_selection"]["max_samples"] = dataset_features.get("max_samples")
    return aligned


def _state_paths(output_dir: Path, stage: str, execution_config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    execution = execution_config["execution"]
    progress_dir = ensure_dir(output_dir / str(execution.get("progress_dir", "progress")))
    manifest_dir = ensure_dir(output_dir / str(execution.get("manifest_dir", "manifests")))
    return (
        progress_dir / f"{stage}_rows.jsonl",
        progress_dir / f"{stage}_categories.jsonl",
        manifest_dir / f"{stage}_manifest.json",
    )


def _validate_or_initialize_state(
    *,
    stage: str,
    fingerprint: str,
    expected_runs: int,
    manifest_path: Path,
    resume: bool,
    environment: Mapping[str, Any],
    dataset_hashes: Mapping[str, str],
    config_hash: str,
) -> dict[str, Any]:
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if resume:
            if previous.get("fingerprint") != fingerprint:
                raise ValueError(
                    f"Cannot resume {stage}: configuration or dataset fingerprint changed. "
                    "Use a new output directory or remove the old progress files."
                )
            return previous

    state = {
        "stage": stage,
        "status": "running",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "completed_utc": None,
        "expected_runs": int(expected_runs),
        "completed_runs": 0,
        "fingerprint": fingerprint,
        "config_hash": config_hash,
        "dataset_hashes": dict(dataset_hashes),
        "environment": dict(environment),
    }
    atomic_write_json(state, manifest_path)
    return state


def _progress_printer(current: int, total: int, row: Mapping[str, Any]) -> None:
    name = row.get("method_name", row.get("model_name", "run"))
    feature_set = row.get("feature_set", "")
    seed = row.get("seed", "")
    f1 = float(row.get("f1", float("nan")))
    fpr = float(row.get("fpr", float("nan")))
    print(f"[{current}/{total}] {name} {feature_set} seed={seed} F1={f1:.4f} FPR={fpr:.4f}")


def _completed_stage_result(
    *,
    stage: str,
    state: Mapping[str, Any],
    expected_runs: int,
    manifest_path: Path,
) -> StageResult | None:
    """Return a stage result when every expected row and output already exists."""

    if state.get("status") != "complete":
        return None
    if int(state.get("completed_runs", -1)) != int(expected_runs):
        return None
    raw_paths = state.get("output_paths", {})
    if not isinstance(raw_paths, Mapping) or not raw_paths:
        return None
    output_paths = {name: Path(str(path)) for name, path in raw_paths.items()}
    if not all(path.exists() for path in output_paths.values()):
        return None
    return StageResult(
        stage=stage,
        status="complete",
        completed_runs=expected_runs,
        expected_runs=expected_runs,
        output_paths=output_paths,
        state_path=manifest_path,
        manifest_path=manifest_path,
    )


def run_fw_lnsa_stage(
    *,
    dataset_name: str,
    execution_config: Mapping[str, Any],
    profile: str,
    repo_root: str | Path,
    output_dir: str | Path,
    path_overrides: Mapping[str, str | Path] | None = None,
    resume: bool = True,
    max_runs: int | None = None,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = _progress_printer,
) -> StageResult:
    """Run one dataset's FW-LNSA grid with atomic saved run rows."""

    root = Path(repo_root).resolve()
    output_root = ensure_dir(output_dir)
    stage = f"{dataset_name}_fw_lnsa"
    config = _resolve_dataset_config(
        execution_config,
        dataset_name=dataset_name,
        profile=profile,
        repo_root=root,
        output_dir=output_root,
        path_overrides=path_overrides,
    )
    expected_runs = len(list(iter_fw_run_specs(config)))
    if max_runs is not None:
        expected_runs = min(expected_runs, int(max_runs))

    rows_path, _categories_path, manifest_path = _state_paths(output_root, stage, execution_config)
    config_hash = mapping_hash(config)
    dataset_hashes = _dataset_fingerprint(dataset_name, config)
    fingerprint = mapping_hash(
        {"stage": stage, "config_hash": config_hash, "dataset_hashes": dataset_hashes}
    )
    state = _validate_or_initialize_state(
        stage=stage,
        fingerprint=fingerprint,
        expected_runs=expected_runs,
        manifest_path=manifest_path,
        resume=resume,
        environment=environment_manifest(root),
        dataset_hashes=dataset_hashes,
        config_hash=config_hash,
    )
    state["expected_runs"] = expected_runs
    if resume:
        completed_stage = _completed_stage_result(
            stage=stage,
            state=state,
            expected_runs=expected_runs,
            manifest_path=manifest_path,
        )
        if completed_stage is not None:
            return completed_stage

    if not resume:
        rows_path.unlink(missing_ok=True)
    existing = load_jsonl_frame(rows_path) if resume else pd.DataFrame()
    saved_rows = existing.to_dict(orient="records")
    state["completed_runs"] = len(saved_rows)
    state["updated_utc"] = utc_now()
    atomic_write_json(state, manifest_path)

    def save_row(row: dict[str, Any]) -> None:
        append_jsonl(row, rows_path)
        saved_rows.append(dict(row))
        state["completed_runs"] = len(saved_rows)
        state["updated_utc"] = utc_now()
        atomic_write_json(state, manifest_path)

    dataset, cache_path, cache_hit = _prepare_dataset_cached(
        dataset_name=dataset_name,
        config=config,
        output_dir=output_root,
        execution_config=execution_config,
        dataset_hashes=dataset_hashes,
    )
    state["prepared_dataset_cache"] = str(cache_path)
    state["prepared_dataset_cache_hit"] = cache_hit
    atomic_write_json(state, manifest_path)
    outputs: ExperimentOutputs = run_prepared_fw_lnsa_experiments(
        dataset,
        config,
        max_runs=max_runs,
        progress_callback=progress_callback,
        save_outputs=True,
        existing_results=existing,
        result_callback=save_row,
    )

    completed = len(outputs.results)
    status = "complete" if completed == expected_runs else "partial"
    state.update(
        {
            "status": status,
            "completed_runs": completed,
            "updated_utc": utc_now(),
            "completed_utc": utc_now() if status == "complete" else None,
            "output_paths": {name: str(path) for name, path in outputs.output_paths.items()},
        }
    )
    atomic_write_json(state, manifest_path)
    return StageResult(
        stage=stage,
        status=status,
        completed_runs=completed,
        expected_runs=expected_runs,
        output_paths=outputs.output_paths,
        state_path=manifest_path,
        manifest_path=manifest_path,
    )


def run_baseline_stage(
    *,
    dataset_name: str,
    execution_config: Mapping[str, Any],
    profile: str,
    repo_root: str | Path,
    output_dir: str | Path,
    path_overrides: Mapping[str, str | Path] | None = None,
    resume: bool = True,
    max_runs: int | None = None,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = _progress_printer,
) -> StageResult:
    """Run one dataset's baseline grid with result and category progress."""

    root = Path(repo_root).resolve()
    output_root = ensure_dir(output_dir)
    stage = f"{dataset_name}_baselines"
    dataset_config = _resolve_dataset_config(
        execution_config,
        dataset_name=dataset_name,
        profile=profile,
        repo_root=root,
        output_dir=output_root,
        path_overrides=path_overrides,
    )
    baseline_config = _resolve_baseline_config(
        execution_config,
        profile=profile,
        repo_root=root,
        dataset_config=dataset_config,
    )
    expected_runs = len(list(iter_baseline_run_specs(baseline_config, dataset_key=dataset_name)))
    if max_runs is not None:
        expected_runs = min(expected_runs, int(max_runs))

    rows_path, categories_path, manifest_path = _state_paths(output_root, stage, execution_config)
    config_hash = mapping_hash(
        {"dataset_config": dataset_config, "baseline_config": baseline_config}
    )
    dataset_hashes = _dataset_fingerprint(dataset_name, dataset_config)
    fingerprint = mapping_hash(
        {"stage": stage, "config_hash": config_hash, "dataset_hashes": dataset_hashes}
    )
    state = _validate_or_initialize_state(
        stage=stage,
        fingerprint=fingerprint,
        expected_runs=expected_runs,
        manifest_path=manifest_path,
        resume=resume,
        environment=environment_manifest(root),
        dataset_hashes=dataset_hashes,
        config_hash=config_hash,
    )
    state["expected_runs"] = expected_runs
    if resume:
        completed_stage = _completed_stage_result(
            stage=stage,
            state=state,
            expected_runs=expected_runs,
            manifest_path=manifest_path,
        )
        if completed_stage is not None:
            return completed_stage

    if not resume:
        rows_path.unlink(missing_ok=True)
        categories_path.unlink(missing_ok=True)
    existing = load_jsonl_frame(rows_path) if resume else pd.DataFrame()
    existing_categories = load_jsonl_frame(categories_path) if resume else pd.DataFrame()
    saved_rows = existing.to_dict(orient="records")
    saved_categories = (
        [existing_categories] if not existing_categories.empty else []
    )
    state["completed_runs"] = len(saved_rows)
    state["updated_utc"] = utc_now()
    atomic_write_json(state, manifest_path)

    def save_row(row: dict[str, Any], categories: pd.DataFrame) -> None:
        append_jsonl(row, rows_path)
        saved_rows.append(dict(row))
        if not categories.empty:
            for category_row in categories.to_dict(orient="records"):
                append_jsonl(category_row, categories_path)
            saved_categories.append(categories.copy())
        state["completed_runs"] = len(saved_rows)
        state["updated_utc"] = utc_now()
        atomic_write_json(state, manifest_path)

    dataset, cache_path, cache_hit = _prepare_dataset_cached(
        dataset_name=dataset_name,
        config=dataset_config,
        output_dir=output_root,
        execution_config=execution_config,
        dataset_hashes=dataset_hashes,
    )
    state["prepared_dataset_cache"] = str(cache_path)
    state["prepared_dataset_cache_hit"] = cache_hit
    atomic_write_json(state, manifest_path)
    fw_results_path = (
        output_root / "tables" / str(dataset_config["outputs"]["results_table"])
    )
    outputs: BaselineOutputs = run_prepared_baseline_experiments(
        dataset,
        baseline_config,
        dataset_key=dataset_name,
        output_dir=output_root,
        fw_lnsa_results_path=fw_results_path,
        max_runs=max_runs,
        progress_callback=progress_callback,
        save_outputs=True,
        existing_results=existing,
        existing_category_analysis=existing_categories,
        result_callback=save_row,
    )

    completed = len(outputs.results)
    status = "complete" if completed == expected_runs else "partial"
    state.update(
        {
            "status": status,
            "completed_runs": completed,
            "updated_utc": utc_now(),
            "completed_utc": utc_now() if status == "complete" else None,
            "output_paths": {name: str(path) for name, path in outputs.output_paths.items()},
        }
    )
    atomic_write_json(state, manifest_path)
    return StageResult(
        stage=stage,
        status=status,
        completed_runs=completed,
        expected_runs=expected_runs,
        output_paths=outputs.output_paths,
        state_path=manifest_path,
        manifest_path=manifest_path,
    )


def _read_required_tables(paths: Sequence[Path]) -> pd.DataFrame:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Research analysis requires completed result tables: "
            + ", ".join(str(path) for path in missing)
        )
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def run_analysis_stage(
    *,
    execution_config: Mapping[str, Any],
    profile: str,
    repo_root: str | Path,
    output_dir: str | Path,
) -> StageResult:
    """Build repeated-run, ablation, sensitivity, and readiness tables."""

    root = Path(repo_root).resolve()
    output_root = ensure_dir(output_dir)
    stage = "analysis"
    table_dir = output_root / "tables"

    dataset_configs = {
        name: _resolve_dataset_config(
            execution_config,
            dataset_name=name,
            profile=profile,
            repo_root=root,
            output_dir=output_root,
            path_overrides=None,
        )
        for name in ("nsl_kdd", "cicids2017")
    }
    baseline_config = resolve_baseline_profile(
        load_yaml_config(_resolve_path(execution_config["baseline_config"], root)),
        profile,
    )

    fw_paths = [
        table_dir / str(dataset_configs[name]["outputs"]["results_table"])
        for name in ("nsl_kdd", "cicids2017")
    ]
    baseline_paths = [
        table_dir / str(baseline_config["outputs"][name]["baseline_results"])
        for name in ("nsl_kdd", "cicids2017")
    ]
    fw_results = _read_required_tables(fw_paths)
    baseline_results = _read_required_tables(baseline_paths)

    analysis_config = execution_config["analysis"]
    outputs: ResearchAnalysisOutputs = analyze_research_results(
        fw_results=fw_results,
        baseline_results=baseline_results,
        output_dir=output_root,
        fpr_limit=float(analysis_config.get("fpr_limit", 0.10)),
        confidence=float(analysis_config.get("confidence", 0.95)),
        expected=analysis_config.get("expected"),
    )

    _rows_path, _categories_path, manifest_path = _state_paths(
        output_root, stage, execution_config
    )
    state = {
        "stage": stage,
        "status": "complete",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "completed_utc": utc_now(),
        "profile": profile,
        "environment": environment_manifest(root),
        "input_tables": [str(path) for path in fw_paths + baseline_paths],
        "output_paths": {name: str(path) for name, path in outputs.output_paths.items()},
    }
    atomic_write_json(state, manifest_path)
    return StageResult(
        stage=stage,
        status="complete",
        completed_runs=1,
        expected_runs=1,
        output_paths=outputs.output_paths,
        state_path=manifest_path,
        manifest_path=manifest_path,
    )


def run_research_suite(
    *,
    execution_config: Mapping[str, Any],
    profile: str,
    repo_root: str | Path,
    output_dir: str | Path,
    stages: Sequence[str] | None = None,
    nsl_train_file: str | Path | None = None,
    nsl_test_file: str | Path | None = None,
    cic_raw_dir: str | Path | None = None,
    cic_archive_file: str | Path | None = None,
    resume: bool = True,
    max_runs: int | None = None,
) -> list[StageResult]:
    """Run selected stages in dependency-safe order."""

    selected = list(stages or execution_config["execution"]["stages"])
    unknown = set(selected) - set(SUPPORTED_STAGES)
    if unknown:
        raise ValueError(f"Unsupported stages: {sorted(unknown)}")

    results: list[StageResult] = []
    for stage in SUPPORTED_STAGES:
        if stage not in selected:
            continue
        if stage == "nsl_kdd_fw_lnsa":
            results.append(
                run_fw_lnsa_stage(
                    dataset_name="nsl_kdd",
                    execution_config=execution_config,
                    profile=profile,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    path_overrides={
                        "train_file": nsl_train_file,
                        "test_file": nsl_test_file,
                    },
                    resume=resume,
                    max_runs=max_runs,
                )
            )
        elif stage == "cicids2017_fw_lnsa":
            results.append(
                run_fw_lnsa_stage(
                    dataset_name="cicids2017",
                    execution_config=execution_config,
                    profile=profile,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    path_overrides={
                        "raw_dir": cic_raw_dir,
                        "archive_file": cic_archive_file,
                    },
                    resume=resume,
                    max_runs=max_runs,
                )
            )
        elif stage == "nsl_kdd_baselines":
            results.append(
                run_baseline_stage(
                    dataset_name="nsl_kdd",
                    execution_config=execution_config,
                    profile=profile,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    path_overrides={
                        "train_file": nsl_train_file,
                        "test_file": nsl_test_file,
                    },
                    resume=resume,
                    max_runs=max_runs,
                )
            )
        elif stage == "cicids2017_baselines":
            results.append(
                run_baseline_stage(
                    dataset_name="cicids2017",
                    execution_config=execution_config,
                    profile=profile,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    path_overrides={
                        "raw_dir": cic_raw_dir,
                        "archive_file": cic_archive_file,
                    },
                    resume=resume,
                    max_runs=max_runs,
                )
            )
        else:
            results.append(
                run_analysis_stage(
                    execution_config=execution_config,
                    profile=profile,
                    repo_root=repo_root,
                    output_dir=output_dir,
                )
            )
    return results
