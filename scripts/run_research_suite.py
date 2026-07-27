"""Run the resumable FW-LNSA research suite locally or in Google Colab."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research_execution import (
    SUPPORTED_STAGES,
    build_execution_plan,
    load_research_execution_config,
    run_research_suite,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable NSL-KDD, CICIDS2017, baseline, and analysis jobs."
    )
    parser.add_argument(
        "--config",
        default="configs/research_execution.yaml",
        help="Research execution YAML.",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "research", "full"],
        default="research",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stage names or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Persistent result directory. Use Google Drive in Colab.",
    )
    parser.add_argument("--nsl-train-file", default=None)
    parser.add_argument("--nsl-test-file", default=None)
    parser.add_argument("--cic-raw-dir", default=None)
    parser.add_argument("--cic-archive-file", default=None)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit each model stage for validation. Do not use for final research runs.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing saved run rows and start the selected stages again.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print exact run counts without loading data or fitting models.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _run_isolated_stages(args: argparse.Namespace, stages: list[str]) -> None:
    """Run each selected stage in a fresh process to release dataset memory."""

    ordered_stages = [stage for stage in SUPPORTED_STAGES if stage in stages]
    for stage in ordered_stages:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--profile",
            str(args.profile),
            "--stages",
            stage,
            "--output-dir",
            str(args.output_dir),
            "--in-process",
        ]
        optional_paths = {
            "--nsl-train-file": args.nsl_train_file,
            "--nsl-test-file": args.nsl_test_file,
            "--cic-raw-dir": args.cic_raw_dir,
            "--cic-archive-file": args.cic_archive_file,
        }
        for flag, value in optional_paths.items():
            if value is not None:
                command.extend([flag, str(value)])
        if args.max_runs is not None:
            command.extend(["--max-runs", str(args.max_runs)])
        if args.no_resume:
            command.append("--no-resume")

        print(f"\nStarting isolated stage: {stage}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    print("\nAll selected research stages completed.")


def main() -> None:
    args = _parser().parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive.")

    config_path = _resolve(args.config)
    assert config_path is not None
    execution_config = load_research_execution_config(config_path)
    if args.stages == "all":
        stages = list(execution_config["execution"]["stages"])
    else:
        stages = [value.strip() for value in args.stages.split(",") if value.strip()]
    unknown = set(stages) - set(SUPPORTED_STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    output_dir = _resolve(args.output_dir)
    assert output_dir is not None

    if args.plan_only:
        plan = build_execution_plan(
            execution_config=execution_config,
            profile=args.profile,
            repo_root=REPO_ROOT,
            stages=stages,
        )
        print(plan.to_string(index=False))
        print(f"\nTotal configured result rows: {int(plan['expected_runs'].sum())}")
        print(f"Total unique model fits: {int(plan['unique_model_fits'].sum())}")
        return

    if len(stages) > 1 and not args.in_process:
        _run_isolated_stages(args, stages)
        return

    print("FW-LNSA research execution")
    print(f"Profile: {args.profile}")
    print(f"Stages: {stages}")
    print(f"Output directory: {output_dir}")
    print(f"Resume enabled: {not args.no_resume}")

    results = run_research_suite(
        execution_config=execution_config,
        profile=args.profile,
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        stages=stages,
        nsl_train_file=_resolve(args.nsl_train_file),
        nsl_test_file=_resolve(args.nsl_test_file),
        cic_raw_dir=_resolve(args.cic_raw_dir),
        cic_archive_file=_resolve(args.cic_archive_file),
        resume=not args.no_resume,
        max_runs=args.max_runs,
    )

    print("\nResearch execution finished.")
    for result in results:
        print(
            f"{result.stage}: {result.status} "
            f"({result.completed_runs}/{result.expected_runs})"
        )
        for name, path in result.output_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
