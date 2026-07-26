"""Run reproducible CICIDS2017 FW-LNSA experiments.

Examples:
    python scripts/run_cicids2017_fw_lnsa.py --config configs/cicids2017_fw_lnsa.yaml
    python scripts/run_cicids2017_fw_lnsa.py --config configs/cicids2017_fw_lnsa.yaml --profile research
    python scripts/run_cicids2017_fw_lnsa.py --config configs/cicids2017_fw_lnsa.yaml --profile full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_yaml_config, resolve_profile
from src.experiments import run_cicids2017_experiments
from src.preprocessing import discover_cicids2017_sources


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FW-LNSA matching experiments on CICIDS2017.",
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "research", "full"],
        help="Execution profile. The YAML active profile is used when omitted.",
    )
    parser.add_argument("--raw-dir", help="Optional extracted CICIDS2017 directory override.")
    parser.add_argument("--archive-file", help="Optional MachineLearningCSV.zip override.")
    parser.add_argument("--output-dir", help="Optional result-directory override.")
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Optional safety limit for experiment runs. Intended for debugging only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and discover CSV sources without fitting models.",
    )
    return parser


def _apply_path_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.raw_dir:
        config["paths"]["raw_dir"] = args.raw_dir
    if args.archive_file:
        config["paths"]["archive_file"] = args.archive_file
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir

    config["paths"]["raw_dir"] = str(_resolve_repo_path(config["paths"]["raw_dir"]))
    archive_value = config["paths"].get("archive_file")
    if archive_value:
        config["paths"]["archive_file"] = str(_resolve_repo_path(archive_value))
    config["paths"]["output_dir"] = str(_resolve_repo_path(config["paths"]["output_dir"]))


def _print_plan(config: dict[str, Any], source_count: int) -> None:
    preprocessing = config["preprocessing"]
    sampling = preprocessing["sampling"]
    print("FW-LNSA CICIDS2017 experiment plan")
    print(f"Profile: {config['profile_name']}")
    print(f"Raw directory: {config['paths']['raw_dir']}")
    print(f"Archive file: {config['paths'].get('archive_file')}")
    print(f"Discovered CSV sources: {source_count}")
    print(f"Output directory: {config['paths']['output_dir']}")
    print(f"Chunk size: {preprocessing['chunk_size']}")
    print(f"Maximum chunks per file: {preprocessing.get('max_chunks_per_file')}")
    strategy = sampling.get("strategy", "per_label_cap")
    print(f"Sampling strategy: {strategy}")
    if strategy == "global_cap":
        print(f"Maximum sampled records: {sampling['max_total_records']}")
    else:
        print(
            "Sampling quotas: "
            f"BENIGN={sampling['max_benign_records']}, "
            f"each attack label={sampling['max_records_per_attack_label']}"
        )
    print(f"Feature sizes: {config['feature_selection']['feature_sizes']}")
    print(f"Methods: {config['experiment']['methods']}")
    print(f"Detector budgets: {config['experiment']['detector_budgets']}")
    print(f"Seeds: {config['experiment']['seeds']}")


def _progress(current: int, total: int, row: dict[str, Any]) -> None:
    print(
        f"[{current}/{total}] "
        f"{row['method_name']} {row['feature_set']} "
        f"seed={row['seed']} budget={row['detector_budget']} "
        f"F1={row['f1']:.4f} FPR={row['fpr']:.4f}"
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive.")

    raw_config = load_yaml_config(_resolve_repo_path(args.config))
    config = resolve_profile(raw_config, args.profile)
    _apply_path_overrides(config, args)

    sources = discover_cicids2017_sources(
        config["paths"]["raw_dir"],
        archive_file=config["paths"].get("archive_file"),
    )
    _print_plan(config, len(sources))

    if args.dry_run:
        print("Configuration and CICIDS2017 data sources are valid.")
        return

    outputs = run_cicids2017_experiments(
        config,
        max_runs=args.max_runs,
        progress_callback=_progress,
    )

    print("\nExperiment run completed.")
    print(f"Result rows: {len(outputs.results)}")
    print("Saved tables:")
    for name, path in outputs.output_paths.items():
        print(f"  {name}: {path}")

    if not outputs.data_quality_report.empty:
        summary = outputs.data_quality_report[
            outputs.data_quality_report["record_type"] == "summary"
        ]
        if not summary.empty:
            row = summary.iloc[0]
            print("\nData quality summary:")
            print(f"  Rows read: {int(row['rows_read'])}")
            print(f"  Duplicates removed: {int(row['duplicate_rows_removed'])}")
            print(f"  Nonfinite values replaced: {int(row['nonfinite_values_replaced'])}")
            print(f"  Sampled records: {int(row['rows_retained_in_sample'])}")
            print(f"  Train records: {int(row['train_records'])}")
            print(f"  Test records: {int(row['test_records'])}")

    if not outputs.balanced_configs.empty:
        columns = ["method_name", "feature_set", "recall", "f1", "fpr", "total_time_sec"]
        print("\nBalanced configurations:")
        print(outputs.balanced_configs[columns].to_string(index=False))


if __name__ == "__main__":
    main()
