"""Regenerate publication figures from saved result CSV files."""

from pathlib import Path


def main() -> None:
    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Looking for tables in: {tables_dir}")
    print(f"Figures will be saved to: {figures_dir}")
    print("Figure generation placeholder. Implement after final CSV schemas are locked.")


if __name__ == "__main__":
    main()
