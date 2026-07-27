"""Run automated tests for the resumable research execution and analysis layer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_research_execution",
        "-v",
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print("Research execution validation passed.")


if __name__ == "__main__":
    main()
