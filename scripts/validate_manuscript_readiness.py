"""Validate code suites and, when present, the final manuscript evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default="results_manuscript/analysis")
    parser.add_argument("--code-only", action="store_true")
    return parser.parse_args()


def run_tests() -> None:
    patterns = [
        "test_core_modules.py",
        "test_fw_lnsa_pipeline.py",
        "test_cicids2017_pipeline.py",
        "test_baseline_pipeline.py",
        "test_research_execution.py",
        "test_confirmatory_protocol.py",
        "test_confirmatory_baselines.py",
        "test_manuscript_analysis.py",
        "test_manuscript_handoff.py",
    ]
    for pattern in patterns:
        print(f"Running {pattern}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"],
            check=True,
        )


def validate_outputs(analysis_dir: Path) -> None:
    manifest_path = analysis_dir / "analysis_manifest.json"
    audit_path = analysis_dir / "publication_tables" / "final_readiness_audit.csv"
    if not manifest_path.exists() or not audit_path.exists():
        raise FileNotFoundError(
            "Final analysis outputs are missing. Run analyze_confirmatory_results.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = pd.read_csv(audit_path)
    failed = audit[(audit["severity"] == "error") & (~audit["passed"].astype(bool))]
    if not bool(manifest.get("audit_passed", False)) or not failed.empty:
        raise RuntimeError(f"Manuscript readiness audit failed:\n{failed.to_string(index=False)}")
    print(f"Final readiness audit passed: {len(audit)} checks.")


def main() -> None:
    args = parse_args()
    run_tests()
    if not args.code_only:
        validate_outputs(Path(args.analysis_dir))
    print("Manuscript readiness validation passed.")


if __name__ == "__main__":
    main()
