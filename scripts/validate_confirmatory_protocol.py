"""Run the validation-calibrated protocol test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_file = repo_root / "tests" / "test_confirmatory_protocol.py"
    if not test_file.exists():
        raise FileNotFoundError(f"Validation test file not found: {test_file}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_confirmatory_protocol.py",
            "-v",
        ],
        cwd=repo_root,
        check=True,
    )
    print("Confirmatory protocol validation passed.")


if __name__ == "__main__":
    main()
