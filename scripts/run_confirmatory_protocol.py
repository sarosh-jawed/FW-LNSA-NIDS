"""Run the validation-calibrated FW-LNSA confirmatory protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.confirmatory_protocol import (  # noqa: E402
    expected_confirmatory_plan,
    load_confirmatory_config,
    run_confirmatory_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune FW-LNSA on validation data, lock configurations, and evaluate "
            "the locked procedures on untouched test partitions."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/confirmatory_protocol.yaml",
        help="Path to the confirmatory YAML configuration.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Execution profile. Use smoke for verification or confirmatory for final runs.",
    )
    parser.add_argument(
        "--dataset",
        choices=["nsl_kdd", "cicids2017", "all"],
        default="all",
    )
    parser.add_argument(
        "--stage",
        choices=["tune", "confirm", "all"],
        default="all",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--nsl-train", default=None)
    parser.add_argument("--nsl-test", default=None)
    parser.add_argument("--cic-raw-dir", default=None)
    parser.add_argument("--cic-archive", default=None)
    parser.add_argument("--max-tuning-rows", type=int, default=None)
    parser.add_argument("--max-final-rows", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _apply_path_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.nsl_train:
        config["datasets"]["nsl_kdd"]["train_file"] = args.nsl_train
    if args.nsl_test:
        config["datasets"]["nsl_kdd"]["test_file"] = args.nsl_test
    if args.cic_raw_dir:
        config["datasets"]["cicids2017"]["raw_dir"] = args.cic_raw_dir
    if args.cic_archive:
        config["datasets"]["cicids2017"]["archive_file"] = args.cic_archive


def _progress(stage: str, current: int, total: int, row: Mapping[str, Any]) -> None:
    print(
        f"[{stage} {current}/{total}] {row.get('dataset_key')} "
        f"{row.get('method')} FS-{row.get('fs_size')} seed={row.get('seed')} "
        f"target_fpr={float(row.get('target_fpr', 0.0)):.2f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = load_confirmatory_config(args.config, profile_name=args.profile)
    _apply_path_overrides(config, args)

    plan = expected_confirmatory_plan(config)
    print(f"Profile: {config['profile_name']}")
    for key, value in plan.items():
        print(f"{key}: {value}")
    if args.plan_only:
        return

    datasets = ["nsl_kdd", "cicids2017"] if args.dataset == "all" else [args.dataset]
    for dataset_name in datasets:
        print(f"\nRunning {dataset_name} confirmatory protocol")
        outputs = run_confirmatory_dataset(
            dataset_name,
            config,
            output_root=config["paths"]["output_dir"],
            stage=args.stage,
            max_tuning_rows=args.max_tuning_rows,
            max_final_rows=args.max_final_rows,
            progress_callback=_progress,
        )
        completed = outputs.manifest["completed"]
        print(
            f"Completed {dataset_name}: tuning={completed['tuning_rows']}, "
            f"locked={completed['locked_configurations']}, "
            f"final={completed['confirmatory_rows']}"
        )
        for name, path in outputs.output_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
