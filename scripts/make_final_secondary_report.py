#!/usr/bin/env python3
"""Descriptive secondary summaries; never redefine original significance family."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from src.final_validation.evidence import read_csv  # noqa: E402
from src.final_validation.io import Job, sha256, table, write_json  # noqa: E402
from src.manuscript_analysis import mean_std_ci, bootstrap_paired_ci  # noqa: E402


def build(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for dataset in ["nsl_kdd", "cicids2017"]:
        path = root / (dataset + "_development") / "validation_sensitivity.csv"
        if not path.exists():
            continue
        frame = read_csv(path)
        frame = frame[frame.experiment.str.startswith("feature_count_")]
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, g in frame.groupby("method"):
            summary = g.groupby("k").validation_f1.agg(["mean", "std"]).reset_index()
            ax.errorbar(
                summary.k,
                summary["mean"],
                yerr=summary["std"].fillna(0),
                fmt="o-" if method == "weighted_similarity" else "s--",
                color="black" if method == "weighted_similarity" else ".5",
                capsize=3,
                label=method,
            )
        ax.set_xlabel("Selected feature count k")
        ax.set_ylabel("Validation F1, mean ± SD (5 development seeds)")
        ax.set_title(dataset + " • secondary k sensitivity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / (dataset + "_feature_count.pdf"), bbox_inches="tight")
        fig.savefig(
            output / (dataset + "_feature_count.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    path = root / "sample120k/test_seed_results.csv"
    if path.exists():
        frame = read_csv(path)
        provenance_path = root / "sample120k/sample_row_provenance.csv"
        if provenance_path.exists():
            provenance = read_csv(provenance_path)
            new_sets = {
                name: set(zip(group["__source_file__"], group["__source_row__"]))
                for name, group in provenance.groupby("partition")
            }
            overlap = []
            for original_partition in ["train", "validation", "test"]:
                old_path = (
                    root
                    / "cicids2017_data"
                    / (original_partition + "_row_signatures.csv")
                )
                if not old_path.exists():
                    continue
                old = read_csv(old_path)
                old_keys = set(
                    zip(old.source_file, old.source_row_zero_based_excluding_header)
                )
                for new_partition, new_keys in new_sets.items():
                    shared = len(old_keys & new_keys)
                    overlap.append(
                        {
                            "original_60k_partition": original_partition,
                            "new_120k_partition": new_partition,
                            "shared_rows": shared,
                            "original_partition_records": len(old_keys),
                            "fraction_of_original_partition": shared / len(old_keys),
                            "interpretation": "cross-study repartitioning overlap; not independent replication",
                        }
                    )
            table(output / "sample120k_cross_study_partition_overlap.csv", overlap)
        rows = []
        for (method, target), group in frame.groupby(["method", "target_fpr"]):
            for metric in [
                "test_f1",
                "test_recall",
                "test_fpr",
                "validation_fpr",
                "retained_detectors",
                "retention_rate",
            ]:
                rows.append(
                    {
                        "study": "SECONDARY_120K",
                        "method": method,
                        "target_fpr": target,
                        "metric": metric,
                        **mean_std_ci(group[metric]),
                    }
                )
        table(output / "sample120k_metric_summary.csv", rows)
        h = frame[frame.method == "hamming"]
        w = frame[frame.method == "weighted_similarity"]
        pairs = h.merge(
            w,
            on=["seed", "fs_size", "target_fpr"],
            suffixes=("_h", "_w"),
            validate="one_to_one",
        )
        paired = []
        for i, metric in enumerate(["test_f1", "test_recall", "test_fpr"]):
            diff = pairs[metric + "_w"].to_numpy() - pairs[metric + "_h"].to_numpy()
            low, high = bootstrap_paired_ci(
                diff, iterations=5000, random_seed=1202026 + i
            )
            paired.append(
                {
                    "study": "SECONDARY_120K",
                    "metric": metric,
                    "pairs": len(diff),
                    "weighted_minus_hamming_mean": diff.mean(),
                    "paired_percentile_95ci_low": low,
                    "paired_percentile_95ci_high": high,
                    "positive_differences": int((diff > 1e-12).sum()),
                    "ties": int((np.abs(diff) <= 1e-12).sum()),
                    "negative_differences": int((diff < -1e-12).sum()),
                    "bootstrap_seed": 1202026 + i,
                    "bootstrap_iterations": 5000,
                    "inferential_p_values": "NOT_REPORTED_DESCRIPTIVE_SECONDARY_STUDY",
                }
            )
        table(output / "sample120k_paired_descriptive.csv", paired)
        compliance = []
        for method, g in frame.groupby("method"):
            compliance.append(
                {
                    "method": method,
                    "seeds": len(g),
                    "test_target_met_count": int((g.test_fpr <= 0.05 + 1e-12).sum()),
                    "mean_absolute_target_error": float(
                        (g.test_fpr - 0.05).abs().mean()
                    ),
                    "mean_validation_fpr": g.validation_fpr.mean(),
                    "mean_test_fpr": g.test_fpr.mean(),
                }
            )
        table(output / "sample120k_calibration.csv", compliance)
        write_json(
            output / "secondary_statistical_contract.json",
            {
                "study": "120k_v1 secondary descriptive",
                "confidence": 0.95,
                "resampling_unit": "20 paired detector seeds300-319, fixed new120k data partitions",
                "bootstrap_iterations": 5000,
                "bootstrap_seed": "1202026+metric index",
                "p_values": "None; original120-test family unchanged",
                "overlap": "The120k cohort contains all60k original sampled records; repartitioning can move old test rows into secondary training and old train rows into secondary test. This does not alter primary models/results and is not independent replication.",
                "preprocessing_selection": "New train only; deterministic new validation grid; test never used in selection",
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--profile", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    dependencies = {
        str(p.relative_to(args.root)): sha256(p)
        for pattern in [
            "*_development/validation_sensitivity.csv",
            "sample120k/test_seed_results.csv",
        ]
        for p in args.root.glob(pattern)
    }
    Job(
        args.root,
        "secondary_report",
        {"script_sha256": sha256(__file__), "dependencies": dependencies},
    ).run(lambda p: build(args.root, p), synthetic=args.profile == "smoke")


if __name__ == "__main__":
    main()
