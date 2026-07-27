"""Run reproducible baseline models for NSL-KDD and CICIDS2017."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import run_prepared_baseline_experiments
from src.config import (
    load_yaml_config,
    resolve_baseline_profile,
    resolve_profile,
)
from src.preprocessing import (
    discover_cicids2017_sources,
    prepare_cicids2017,
    prepare_nsl_kdd,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fair classical baselines using FW-LNSA data partitions."
    )
    parser.add_argument("--config", required=True, help="Baseline YAML configuration.")
    parser.add_argument("--profile", default=None, help="smoke, research, or full.")
    parser.add_argument(
        "--dataset",
        choices=["all", "nsl_kdd", "cicids2017"],
        default="all",
        help="Run one dataset or every configured dataset.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--nsl-train-file", default=None)
    parser.add_argument("--nsl-test-file", default=None)
    parser.add_argument("--cic-raw-dir", default=None)
    parser.add_argument("--cic-archive-file", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_dataset_config(
    baseline_config: Mapping[str, Any],
    dataset_name: str,
) -> dict[str, Any]:
    config_path = Path(str(baseline_config["dataset_configs"][dataset_name]))
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    loaded = load_yaml_config(config_path)
    profile_name = str(baseline_config["dataset_profiles"][dataset_name])
    return resolve_profile(loaded, profile_name)


def _apply_overrides(
    dataset_name: str,
    dataset_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.output_dir:
        dataset_config["paths"]["output_dir"] = args.output_dir
    if dataset_name == "nsl_kdd":
        if args.nsl_train_file:
            dataset_config["paths"]["train_file"] = args.nsl_train_file
        if args.nsl_test_file:
            dataset_config["paths"]["test_file"] = args.nsl_test_file
    else:
        if args.cic_raw_dir:
            dataset_config["paths"]["raw_dir"] = args.cic_raw_dir
        if args.cic_archive_file:
            dataset_config["paths"]["archive_file"] = args.cic_archive_file
    return dataset_config


def _align_feature_selection_with_dataset(
    baseline_config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse FW-LNSA feature-selection sampling controls for fair comparison."""

    aligned = deepcopy(dict(baseline_config))
    if bool(aligned["feature_selection"].get("inherit_dataset_sampling", True)):
        dataset_feature_config = dataset_config["feature_selection"]
        aligned["feature_selection"]["random_seed"] = int(
            dataset_feature_config.get("random_seed", 42)
        )
        aligned["feature_selection"]["max_samples"] = dataset_feature_config.get(
            "max_samples"
        )
    return aligned


def _print_plan(
    dataset_name: str,
    baseline_config: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
) -> None:
    enabled_models = [
        name
        for name, settings in baseline_config["models"].items()
        if bool(settings.get("enabled", False))
    ]
    print("\nBaseline experiment plan")
    print(f"Dataset: {dataset_name}")
    print(f"Baseline profile: {baseline_config['profile_name']}")
    print(f"Dataset profile: {dataset_config['profile_name']}")
    print(f"Feature sizes: {baseline_config['feature_selection']['feature_sizes']}")
    print(
        "Feature-selection sample limit: "
        f"{baseline_config['feature_selection'].get('max_samples')}"
    )
    print(f"Models: {enabled_models}")
    print(f"Seeds: {baseline_config['experiment']['seeds']}")
    print(f"Output directory: {Path(dataset_config['paths']['output_dir']).resolve()}")


def _validate_data_sources(dataset_name: str, dataset_config: Mapping[str, Any]) -> None:
    paths = dataset_config["paths"]
    if dataset_name == "nsl_kdd":
        train_file = Path(paths["train_file"])
        test_file = Path(paths["test_file"])
        if not train_file.exists():
            raise FileNotFoundError(f"NSL-KDD training file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"NSL-KDD testing file not found: {test_file}")
        print(f"Training file: {train_file.resolve()}")
        print(f"Testing file: {test_file.resolve()}")
    else:
        sources = discover_cicids2017_sources(
            paths["raw_dir"],
            archive_file=paths.get("archive_file"),
        )
        print(f"Discovered CICIDS2017 CSV sources: {len(sources)}")


def _prepare_dataset(dataset_name: str, config: Mapping[str, Any]):
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
        max_records_per_attack_label=int(
            sampling.get("max_records_per_attack_label", 1)
        ),
        max_total_records=sampling.get("max_total_records"),
        sampling_seed=int(sampling.get("random_seed", 42)),
        test_size=float(split.get("test_size", 0.30)),
        split_seed=int(split.get("random_seed", 42)),
        stratify_by=str(split.get("stratify_by", "original_label")),
        scale=bool(preprocessing.get("scale", True)),
    )


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    loaded = load_yaml_config(config_path)
    baseline_config = resolve_baseline_profile(loaded, args.profile)

    configured_datasets = list(baseline_config["datasets"])
    datasets = configured_datasets if args.dataset == "all" else [args.dataset]
    missing = [name for name in datasets if name not in configured_datasets]
    if missing:
        raise ValueError(f"Dataset is not enabled in this profile: {missing[0]}")

    for dataset_name in datasets:
        dataset_config = _apply_overrides(
            dataset_name,
            _load_dataset_config(baseline_config, dataset_name),
            args,
        )
        run_config = _align_feature_selection_with_dataset(
            baseline_config, dataset_config
        )
        _print_plan(dataset_name, run_config, dataset_config)
        _validate_data_sources(dataset_name, dataset_config)

        if args.dry_run:
            print("Configuration and data sources are valid.")
            continue

        dataset = _prepare_dataset(dataset_name, dataset_config)
        output_dir = Path(dataset_config["paths"]["output_dir"])
        fw_output_name = dataset_config["outputs"]["results_table"]
        fw_results_path = output_dir / "tables" / fw_output_name

        def progress(current: int, total: int, row: dict[str, Any]) -> None:
            print(
                f"[{current}/{total}] {row['model_name']} {row['feature_set']} "
                f"seed={row['seed']} F1={row['f1']:.4f} FPR={row['fpr']:.4f}"
            )

        outputs = run_prepared_baseline_experiments(
            dataset,
            run_config,
            dataset_key=dataset_name,
            output_dir=output_dir,
            fw_lnsa_results_path=fw_results_path,
            max_runs=args.max_runs,
            progress_callback=progress,
            save_outputs=True,
        )
        print("\nBaseline run completed.")
        print(f"Result rows: {len(outputs.results)}")
        print("Saved tables:")
        for name, path in outputs.output_paths.items():
            print(f"  {name}: {path.resolve()}")
        if not outputs.comparison.empty:
            status_counts = outputs.comparison["comparison_status"].value_counts().to_dict()
            print(f"Comparison status: {status_counts}")


if __name__ == "__main__":
    main()
