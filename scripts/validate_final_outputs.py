#!/usr/bin/env python3
"""Numeric/cardinality/seed scientific gates, independent of model fitting."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.final_validation.evidence import read_csv  # noqa: E402
from src.final_validation.io import table, write_json  # noqa: E402


def validate(root, accepted, profile="full"):
    root, accepted = Path(root), Path(accepted)
    checks = []

    def check(name, condition, detail=""):
        checks.append(
            {"check": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    def read(relative):
        path = root / relative
        if not path.is_file():
            checks.append(
                {
                    "check": relative,
                    "status": "PENDING",
                    "detail": "Output not available",
                }
            )
            return None
        return read_csv(path)

    original = read_csv(accepted / "evidence/seed_level_csv/fw_lnsa_seed_results.csv")
    integrity = read("evidence/integrity_checks.csv")
    if integrity is not None:
        check("accepted_evidence_checks", bool(integrity.passed.all()))
    reconciliation = read("evidence/statistical_reconciliation.csv")
    if reconciliation is not None:
        check(
            "120_test_statistics_reconciled",
            bool(reconciliation.matches_at_1e_12.all()),
        )
    if profile == "full":
        for dataset in ["nsl_kdd", "cicids2017"]:
            audit = read(dataset + "_data/reconstruction_checks.csv")
            if audit is not None:
                check(
                    dataset + ":partition_and_features",
                    len(audit) == 5 and bool(audit.passed.all()),
                )
            for k in [10, 20]:
                patterns = read(f"{dataset}_data/fs{k}_patterns.csv")
                if patterns is not None:
                    check(
                        f"{dataset}:FS{k}:patterns",
                        len(patterns) == 3
                        and bool((patterns.unique_patterns <= patterns.records).all())
                        and bool(patterns.constant_bits.between(0, k).all()),
                    )
            stability = read(dataset + "_stability/selection_frequency.csv")
            if stability is not None:
                check(
                    dataset + ":30_stability_resamples",
                    set(stability.resamples) == {30},
                )
                check(
                    dataset + ":selection_frequency",
                    bool(
                        np.allclose(
                            stability.selected_count / 30, stability.selection_frequency
                        )
                    ),
                )
                for k, g in stability.groupby("k"):
                    check(
                        f"{dataset}:top{k}:total_selection_count",
                        g.selected_count.sum() == k * 30,
                    )
            pairs = read(dataset + "_stability/pairwise_stability.csv")
            if pairs is not None:
                check(
                    dataset + ":870_stability_pair_rows",
                    len(pairs) == 870 and pairs.jaccard.between(0, 1).all(),
                )
            development = read(dataset + "_development/validation_sensitivity.csv")
            if development is not None:
                check(
                    dataset + ":development_row_count",
                    len(development) == (90 if dataset == "nsl_kdd" else 100),
                )
                check(
                    dataset + ":development_unique_keys",
                    not development.duplicated(["experiment", "method", "seed"]).any(),
                )
                check(
                    dataset + ":development_seeds",
                    set(development.seed) == set(range(42, 47)),
                )
                check(
                    dataset + ":development_validation_ceiling",
                    bool((development.validation_fpr <= 0.05 + 1e-12).all()),
                )
                check(
                    dataset + ":validation_only_scope",
                    set(development.scope) == {"SECONDARY_DEVELOPMENT_VALIDATION_ONLY"},
                )
            replay = read(dataset + "_replay/replay_checks.csv")
            if replay is not None:
                keys = ["dataset_key", "method", "fs_size", "target_fpr", "seed"]
                check(dataset + ":240_replay_rows", len(replay) == 240)
                check(
                    dataset + ":replay_unique_keys", not replay.duplicated(keys).any()
                )
                merged = replay.merge(
                    original[original.dataset_key == dataset],
                    on=keys,
                    suffixes=("_new", "_original"),
                    validate="one_to_one",
                )
                check(dataset + ":replay_all_keys", len(merged) == 240)
                for field in [
                    "test_tn",
                    "test_fp",
                    "test_fn",
                    "test_tp",
                    "test_f1",
                    "test_recall",
                    "test_fpr",
                    "retained_detectors",
                    "unique_self_patterns",
                    "candidate_detectors",
                    "self_space_records_used",
                ]:
                    check(
                        dataset + ":replay:" + field,
                        bool(
                            np.allclose(
                                merged[field + "_new"],
                                merged[field + "_original"],
                                atol=1e-12,
                                rtol=0,
                            )
                        ),
                    )
                check(
                    dataset + ":frozen_thresholds",
                    bool(
                        np.allclose(
                            replay.threshold,
                            merged.detection_threshold_original,
                            atol=1e-12,
                            rtol=0,
                        )
                    ),
                )
            cats = read(dataset + "_replay/category_seed_recalls.csv")
            if cats is not None:
                check(
                    dataset + ":category_zero_support_missing",
                    bool(cats.loc[cats.support == 0, "recall"].isna().all()),
                )
                check(
                    dataset + ":category_recalls_range",
                    bool(cats.loc[cats.support > 0, "recall"].between(0, 1).all()),
                )
                if dataset == "nsl_kdd":
                    check(
                        "NSL:binary_test_categories_only",
                        set(cats.category) == {"Attack (Unspecified)"},
                    )
        bench = read("benchmark/benchmark_repeats.csv")
        if bench is not None:
            check("benchmark:80_repeats", len(bench) == 80)
            check(
                "benchmark:finite_positive_timing",
                bool(
                    np.isfinite(bench.elapsed_sec).all()
                    and (bench.elapsed_sec > 0).all()
                ),
            )
            for key, g in bench.groupby(["dataset_key", "method", "target_fpr"]):
                check(f"benchmark:{key}:10_repeats", len(g) == 10)
                check(
                    f"benchmark:{key}:frozen_predictions",
                    g.prediction_sha256.nunique() == 1,
                )
            check(
                "benchmark:rss_absolute_and_delta",
                bool((bench.peak_rss_mb >= bench.prediction_rss_delta_mb).all()),
            )
        secondary = read("sample120k/test_seed_results.csv")
        if secondary is not None:
            check("120k:40_rows", len(secondary) == 40)
            check(
                "120k:evaluation_seeds300_319",
                set(secondary.seed) == set(range(300, 320)),
            )
            check(
                "120k:unique_keys", not secondary.duplicated(["method", "seed"]).any()
            )
            check(
                "120k:validation_ceiling",
                bool((secondary.validation_fpr <= 0.05 + 1e-12).all()),
            )
        tuning = read("sample120k/tuning.csv")
        if tuning is not None:
            check("120k:120_tuning_rows", len(tuning) == 120)
            check("120k:tuning_seeds200_204", set(tuning.seed) == set(range(200, 205)))
    else:
        # Smoke software path: never manufacture a full empirical cardinality gate.
        check(
            "synthetic_scope_only",
            True,
            "Full real-data cardinality checks intentionally not applied to synthetic fixtures",
        )
    table(root / "scientific_qa.csv", checks)
    counts = pd.Series([r["status"] for r in checks]).value_counts().to_dict()
    write_json(
        root / "scientific_qa.json",
        {
            "profile": profile,
            "counts": counts,
            "checks": checks,
            "scope": "Numeric/key/range checks; favorable scientific outcomes are not required",
        },
    )
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=["smoke", "full"])
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    checks = validate(args.root, args.accepted, args.profile)
    print(pd.DataFrame(checks).to_string(index=False))
    if any(
        r["status"] == "FAIL" or (r["status"] == "PENDING" and not args.allow_pending)
        for r in checks
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
