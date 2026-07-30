"""Run hash-compatible validation-calibrated classical baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.confirmatory_baselines import load_manuscript_config, run_confirmatory_baselines_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/manuscript_readiness.yaml")
    parser.add_argument("--profile", default="final")
    parser.add_argument("--dataset", choices=["nsl_kdd", "cicids2017", "all"], default="all")
    parser.add_argument("--confirmatory-results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--confirmatory-config", default=None)
    parser.add_argument("--nsl-train", default=None)
    parser.add_argument("--nsl-test", default=None)
    parser.add_argument("--cic-raw-dir", default=None)
    parser.add_argument("--cic-archive", default=None)
    parser.add_argument("--max-model-fits", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_manuscript_config(args.config, profile_name=args.profile)
    confirmatory_root = Path(args.confirmatory_results_dir or config["paths"]["confirmatory_results_dir"])
    output_root = Path(args.output_dir or config["paths"]["manuscript_results_dir"])
    datasets = config["baseline"]["datasets"] if args.dataset == "all" else [args.dataset]

    for dataset_key in datasets:
        print(f"Running confirmatory baselines for {dataset_key}.", flush=True)
        outputs = run_confirmatory_baselines_dataset(
            dataset_key,
            config,
            confirmatory_results_root=confirmatory_root,
            output_root=output_root,
            confirmatory_config_path=args.confirmatory_config,
            dataset_path_overrides={
                "nsl_train": args.nsl_train,
                "nsl_test": args.nsl_test,
                "cic_raw_dir": args.cic_raw_dir,
                "cic_archive": args.cic_archive,
            },
            max_model_fits=args.max_model_fits,
            progress_callback=lambda completed, expected, row: print(
                f"[{completed}/{expected}] {row['model']} {row['feature_set']} "
                f"seed={row['seed']} target_fpr={row['target_fpr']:.2f}",
                flush=True,
            ),
        )
        print(
            f"Completed {dataset_key}: {len(outputs.seed_results)}/{outputs.manifest['expected_rows']} rows."
        )
        for name, path in outputs.output_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
