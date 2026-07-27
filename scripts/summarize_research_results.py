"""Regenerate publication-support analysis from completed result CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research_execution import load_research_execution_config, run_analysis_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize FW-LNSA research results.")
    parser.add_argument("--config", default="configs/research_execution.yaml")
    parser.add_argument("--profile", choices=["research", "full"], default="research")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    result = run_analysis_stage(
        execution_config=load_research_execution_config(config_path),
        profile=args.profile,
        repo_root=REPO_ROOT,
        output_dir=output_dir,
    )
    print(f"Analysis status: {result.status}")
    for name, path in result.output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
