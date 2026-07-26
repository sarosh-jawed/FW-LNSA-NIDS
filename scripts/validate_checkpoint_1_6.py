"""Run checkpoint 1 to 6 validation tests.

Usage:
    python scripts/validate_checkpoint_1_6.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_file = repo_root / "tests" / "test_checkpoint_1_6_modules.py"

    if not test_file.exists():
        raise FileNotFoundError(f"Validation test file not found: {test_file}")

    command = [sys.executable, "-m", "unittest", "tests/test_checkpoint_1_6_modules.py", "-v"]
    completed = subprocess.run(command, cwd=repo_root, check=False)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    print("Checkpoint 1 to 6 validation passed.")


if __name__ == "__main__":
    main()
