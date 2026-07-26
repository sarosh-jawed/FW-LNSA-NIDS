"""Run NSL-KDD FW-LNSA experiments.

Implementation target:
    python scripts/run_nsl_kdd_fw_lnsa.py --config configs/nsl_kdd_fw_lnsa.yaml
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"Config found: {config_path}")
    print("NSL-KDD FW-LNSA runner placeholder. Implement src modules before running experiments.")


if __name__ == "__main__":
    main()
