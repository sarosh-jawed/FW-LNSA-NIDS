"""Run baseline model experiments."""

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
    print("Baseline runner placeholder. Implement src/baselines.py first.")


if __name__ == "__main__":
    main()
