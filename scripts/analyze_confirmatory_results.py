"""Create final statistical audit, publication tables, and figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.confirmatory_baselines import load_manuscript_config
from src.manuscript_analysis import analyze_and_save_manuscript_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/manuscript_readiness.yaml")
    parser.add_argument("--profile", default="final")
    parser.add_argument("--confirmatory-results-dir", default=None)
    parser.add_argument("--baseline-results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_manuscript_config(args.config, profile_name=args.profile)
    confirmatory_root = Path(args.confirmatory_results_dir or config["paths"]["confirmatory_results_dir"])
    baseline_root = Path(args.baseline_results_dir or config["paths"]["manuscript_results_dir"])
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["paths"]["manuscript_results_dir"]) / "analysis"
    )
    outputs = analyze_and_save_manuscript_evidence(
        confirmatory_root=confirmatory_root,
        baseline_root=baseline_root,
        output_root=output_root,
        manuscript_config=config,
    )
    print(f"Audit passed: {outputs.manifest['audit_passed']}")
    print(f"Analysis manifest: {outputs.manifest_path}")
    for name, path in outputs.figure_paths.items():
        print(f"  figure {name}: {path}")


if __name__ == "__main__":
    main()
