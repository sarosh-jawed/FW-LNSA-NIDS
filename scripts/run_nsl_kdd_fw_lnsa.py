"""Run reproducible NSL-KDD FW-LNSA experiments.

Examples:
    python scripts/run_nsl_kdd_fw_lnsa.py --config configs/nsl_kdd_fw_lnsa.yaml
    python scripts/run_nsl_kdd_fw_lnsa.py --config configs/nsl_kdd_fw_lnsa.yaml --profile research
    python scripts/run_nsl_kdd_fw_lnsa.py --config configs/nsl_kdd_fw_lnsa.yaml --profile full
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
from src.experiments import run_nsl_kdd_experiments


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FW-LNSA matching experiments on NSL-KDD.",
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "research", "full"],
        help="Execution profile. The YAML active profile is used when omitted.",
    )
    parser.add_argument("--train-file", help="Optional NSL-KDD training-file override.")
    parser.add_argument("--test-file", help="Optional NSL-KDD test-file override.")
    parser.add_argument("--output-dir", help="Optional result-directory override.")
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Optional safety limit for experiment runs. Intended for debugging only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and paths without fitting models.",
    )
    return parser


def _apply_path_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.train_file:
        config["paths"]["train_file"] = args.train_file
    if args.test_file:
        config["paths"]["test_file"] = args.test_file
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir

    config["paths"]["train_file"] = str(_resolve_repo_path(config["paths"]["train_file"]))
    config["paths"]["test_file"] = str(_resolve_repo_path(config["paths"]["test_file"]))
    config["paths"]["output_dir"] = str(_resolve_repo_path(config["paths"]["output_dir"]))


def _print_plan(config: dict[str, Any]) -> None:
    print("FW-LNSA NSL-KDD experiment plan")
    print(f"Profile: {config['profile_name']}")
    print(f"Training data: {config['paths']['train_file']}")
    print(f"Testing data: {config['paths']['test_file']}")
    print(f"Output directory: {config['paths']['output_dir']}")
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
    _print_plan(config)

    train_path = Path(config["paths"]["train_file"])
    test_path = Path(config["paths"]["test_file"])
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "NSL-KDD data files are missing. Expected: " + ", ".join(missing)
        )

    if args.dry_run:
        print("Configuration and data paths are valid.")
        return

    outputs = run_nsl_kdd_experiments(
        config,
        max_runs=args.max_runs,
        progress_callback=_progress,
    )

    print("\nExperiment run completed.")
    print(f"Result rows: {len(outputs.results)}")
    print("Saved tables:")
    for name, path in outputs.output_paths.items():
        print(f"  {name}: {path}")

    if not outputs.balanced_configs.empty:
        columns = ["method_name", "feature_set", "recall", "f1", "fpr", "total_time_sec"]
        print("\nBalanced configurations:")
        print(outputs.balanced_configs[columns].to_string(index=False))


if __name__ == "__main__":
    main()
