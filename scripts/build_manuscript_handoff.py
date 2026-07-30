"""Build the final focused evidence package for the manuscript writer."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.confirmatory_baselines import load_manuscript_config
from src.manuscript_analysis import build_writer_handoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/manuscript_readiness.yaml")
    parser.add_argument("--profile", default="final")
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--confirmatory-results-dir", default=None)
    parser.add_argument("--baseline-results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--executed-notebook", default=None)
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_manuscript_config(args.config, profile_name=args.profile)
    manuscript_root = Path(config["paths"]["manuscript_results_dir"])
    analysis_root = Path(args.analysis_dir or manuscript_root / "analysis")
    confirmatory_root = Path(args.confirmatory_results_dir or config["paths"]["confirmatory_results_dir"])
    baseline_root = Path(args.baseline_results_dir or manuscript_root)
    output_dir = Path(args.output_dir or config["paths"]["handoff_dir"])
    handoff = build_writer_handoff(
        analysis_root=analysis_root,
        confirmatory_root=confirmatory_root,
        baseline_root=baseline_root,
        output_dir=output_dir,
        repository_root=Path.cwd(),
        manuscript_config=config,
        executed_notebook=args.executed_notebook,
    )
    print(f"Writer handoff created: {handoff}")
    if args.make_zip:
        archive = shutil.make_archive(str(handoff), "zip", root_dir=handoff)
        print(f"Writer handoff archive: {archive}")


if __name__ == "__main__":
    main()
