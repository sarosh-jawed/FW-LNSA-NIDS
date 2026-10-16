#!/usr/bin/env python3
"""All new research jobs require an explicit execution profile."""

from pathlib import Path
import os
import sys

for variable in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.final_validation.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
