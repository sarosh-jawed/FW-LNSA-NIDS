#!/usr/bin/env python3
"""Reconcile an archived full execution without fitting any model."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.final_validation.evidence import audit_evidence, read_csv  # noqa: E402
from scripts.verify_final_validation import verify_export  # noqa: E402

BASE = "8496b94717c85566748c767c387f999d76bdc9a7"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_lineage(repository, source_hashes):
    """Permit only exact, hash-pinned, reconstructable static cleanup edits."""
    repository = Path(repository)
    ledger_path = repository / "docs/release/SOURCE_CLEANUP_LEDGER.json"
    ledger = {r["path"]: r for r in json.loads(ledger_path.read_text())}
    rows = []
    for rel, expected in source_hashes.items():
        path = repository / rel
        content = path.read_bytes() if path.is_file() else b""
        actual = hashlib.sha256(content).hexdigest()
        exact = actual == expected
        restored = False
        if not exact and rel in ledger:
            edit = ledger[rel]
            start = edit["byte_offset"]
            before, after = edit["before"].encode(), edit["after"].encode()
            recovered = content[:start] + before + content[start + len(after) :]
            restored = (
                actual == edit["release_sha256"]
                and expected == edit["executed_sha256"]
                and content[start : start + len(after)] == after
                and hashlib.sha256(recovered).hexdigest() == expected
            )
        rows.append(
            {
                "path": rel,
                "executed_sha256": expected,
                "release_sha256": actual,
                "byte_identical": exact,
                "documented_static_cleanup": restored,
                "matches": exact or restored,
            }
        )
    return rows


def reconcile(export, accepted, repository, output):
    export, accepted, repository, output = map(
        Path, (export, accepted, repository, output)
    )
    output.mkdir(parents=True, exist_ok=True)
    verification = verify_export(export)
    if not verification["eligible_for_independent_final_review"]:
        raise ValueError("Full real-data gates are incomplete")
    checks = []

    def check(name, condition, detail=""):
        checks.append(
            {"check": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    identity = verification["source_identity"]
    source_rows = source_lineage(repository, identity["source_hashes"])
    check(
        "all_executed_source_files",
        all(r["matches"] for r in source_rows),
        f"{len(source_rows)} files: exact hashes or explicit reconstructable static cleanup",
    )
    check(
        "accepted_package_identity",
        digest(accepted / "PACKAGE_MANIFEST.json") == identity["accepted"],
    )
    reference = repository / "docs/final_revision/accepted_file_hashes.json"
    audit_evidence(accepted, output / "recomputed_original", reference)
    original = read_csv(accepted / "evidence/seed_level_csv/fw_lnsa_seed_results.csv")
    originals = {
        "fw": original,
        "reference": read_csv(
            accepted / "evidence/seed_level_csv/baseline_seed_results.csv"
        ),
    }
    keys = ["dataset_key", "fs_size", "method", "target_fpr", "seed"]
    with zipfile.ZipFile(export) as z:

        def frame(name):
            with z.open(name) as f:
                return pd.read_csv(f, float_precision="round_trip")

        def data(name):
            return json.loads(z.read(name))

        check(
            "export_payload_hash_count",
            len(data("EXPORT_MANIFEST.json")["files"]) == 2269,
            "This reviewed export has 2269 hashed payload files",
        )
        check("structured_failures_empty", data("structured_failures.json") == [])
        qa = data("scientific_qa.json")
        check("scientific_qa_88_pass", qa["counts"] == {"PASS": 88})
        for name in [
            "paired_120_regenerated.csv",
            "statistical_reconciliation.csv",
            "all_predictive_metrics.csv",
            "calibration_traceability.csv",
        ]:
            a = read_csv(output / "recomputed_original" / name)
            b = frame("evidence/" + name)
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=0, atol=1e-12)
            check("independent_original:" + name, True)
        # Holm is independently recalculated from raw p values in one 120-test family.
        paired = frame("evidence/paired_120_regenerated.csv")
        order = np.argsort(paired.p_value.to_numpy(), kind="stable")
        raw = paired.p_value.to_numpy()[order]
        adjusted = np.minimum(
            1, np.maximum.accumulate(raw * np.arange(len(raw), 0, -1))
        )
        restored = np.empty(len(order))
        restored[order] = adjusted
        check(
            "holm_family_120",
            len(paired) == 120
            and np.allclose(restored, paired.holm_adjusted_p, atol=1e-12, rtol=0),
        )
        replay_caveats = []
        for dataset in ["nsl_kdd", "cicids2017"]:
            replay = frame(dataset + "_replay/replay_checks.csv")
            merged = replay.merge(
                original,
                on=keys,
                suffixes=("_replay", "_original"),
                validate="one_to_one",
            )
            check(dataset + ":240_replay_rows", len(merged) == 240)
            for part in ["test", "validation"]:
                for metric in ["tn", "fp", "fn", "tp", "f1", "recall", "fpr"]:
                    col = part + "_" + metric
                    check(
                        dataset + ":replay:" + col,
                        np.allclose(
                            merged[col + "_replay"],
                            merged[col + "_original"],
                            atol=1e-12,
                            rtol=0,
                        ),
                    )
            for threshold in ["threshold", "validation_threshold", "test_threshold"]:
                check(
                    dataset + ":final:" + threshold,
                    np.allclose(
                        merged[threshold],
                        merged.detection_threshold_original,
                        atol=1e-12,
                        rtol=0,
                    ),
                )
            stale = ~np.isclose(
                merged.detection_threshold_replay, merged.threshold, atol=1e-12, rtol=0
            )
            replay_caveats.append(
                {
                    "dataset_key": dataset,
                    "stale_fit_summary_threshold_rows": int(stale.sum()),
                    "total_rows": len(merged),
                    "authoritative_final_threshold_field": "threshold",
                    "reason": "fit_summary detection_threshold captures pre-calibration default; threshold/validation_threshold/test_threshold agree with accepted calibrated cutoff",
                }
            )
            check(
                dataset + ":reconstruction",
                frame(dataset + "_data/reconstruction_checks.csv").passed.all(),
            )
            dev = frame(dataset + "_development/validation_sensitivity.csv")
            check(
                dataset + ":development_rows",
                len(dev) == (90 if dataset == "nsl_kdd" else 100),
            )
            check(dataset + ":development_seeds", set(dev.seed) == set(range(42, 47)))
            check(
                dataset + ":validation_only",
                not any(c.startswith("test_") for c in dev.columns),
            )
            for (experiment, method), group in dev.groupby(["experiment", "method"]):
                check(
                    f"{dataset}:development:{experiment}:{method}",
                    len(group) == 5 and set(group.seed) == set(range(42, 47)),
                )
            mi = frame(dataset + "_stability/resample_mi_scores.csv")
            check(
                dataset + ":30_training_resamples",
                set(mi.resample_seed) == set(range(3000, 3030)),
            )
        for name, df in originals.items():
            for part in ["validation", "test"]:
                tn, fp, fn, tp = [
                    df[part + "_" + c].to_numpy() for c in ["tn", "fp", "fn", "tp"]
                ]
                for metric, actual in [
                    ("f1", 2 * tp / (2 * tp + fp + fn)),
                    ("recall", tp / (tp + fn)),
                    ("fpr", fp / (tn + fp)),
                ]:
                    check(
                        f"{name}:{part}:confusion_{metric}",
                        np.allclose(
                            actual, df[part + "_" + metric], atol=1e-12, rtol=0
                        ),
                    )
        secondary = frame("sample120k/test_seed_results.csv")
        check(
            "120k:40_paired_seed_rows",
            len(secondary) == 40
            and not secondary.duplicated(["method", "seed"]).any()
            and set(secondary.seed) == set(range(300, 320)),
        )
        check("120k:120_tuning_rows", len(frame("sample120k/tuning.csv")) == 120)
        from src.manuscript_analysis import bootstrap_paired_ci

        h = secondary[secondary.method == "hamming"].set_index("seed").sort_index()
        w = (
            secondary[secondary.method == "weighted_similarity"]
            .set_index("seed")
            .sort_index()
        )
        for i, metric in enumerate(["test_f1", "test_recall", "test_fpr"]):
            diff = w[metric].to_numpy() - h[metric].to_numpy()
            low, high = bootstrap_paired_ci(
                diff, iterations=5000, random_seed=1202026 + i
            )
            saved = (
                frame("secondary_report/sample120k_paired_descriptive.csv")
                .set_index("metric")
                .loc[metric]
            )
            check(
                "120k:paired:" + metric,
                np.allclose(
                    [diff.mean(), low, high],
                    [
                        saved.weighted_minus_hamming_mean,
                        saved.paired_percentile_95ci_low,
                        saved.paired_percentile_95ci_high,
                    ],
                    atol=1e-12,
                    rtol=0,
                ),
            )
        provenance = frame("sample120k/sample_row_provenance.csv")
        new = {
            k: set(zip(g.__source_file__, g.__source_row__))
            for k, g in provenance.groupby("partition")
        }
        overlaps = frame(
            "secondary_report/sample120k_cross_study_partition_overlap.csv"
        )
        for part in ["train", "validation", "test"]:
            old = frame("cicids2017_data/" + part + "_row_signatures.csv")
            old_set = set(
                zip(old.source_file, old.source_row_zero_based_excluding_header)
            )
            check("120k:nested:" + part, old_set.issubset(set.union(*new.values())))
            for new_part, new_set in new.items():
                saved = overlaps[
                    (overlaps.original_60k_partition == part)
                    & (overlaps.new_120k_partition == new_part)
                ].iloc[0]
                check(
                    f"120k:overlap:{part}:{new_part}",
                    len(old_set & new_set) == saved.shared_rows,
                )
        repeats = frame("benchmark/benchmark_repeats.csv")
        summary = frame("benchmark/benchmark_summary.csv")
        check("benchmark:80_rows", len(repeats) == 80)
        for key, group in repeats.groupby(["dataset_key", "method", "target_fpr"]):
            check(
                "benchmark:repeat_keys:" + str(key),
                len(group) == 10
                and set(group.repeat) == set(range(10))
                and group.prediction_sha256.nunique() == 1,
            )
            saved = summary[
                (summary.dataset_key == key[0])
                & (summary.method == key[1])
                & np.isclose(summary.target_fpr, key[2])
            ].iloc[0]
            for metric in [
                "elapsed_sec",
                "throughput_records_per_sec",
                "peak_rss_mb",
                "prediction_rss_delta_mb",
                "process_high_water_rss_mb",
                "detector_state_pickle_bytes",
                "deployed_pipeline_pickle_bytes",
            ]:
                actual = [
                    group[metric].median(),
                    group[metric].quantile(0.25),
                    group[metric].quantile(0.75),
                    group[metric].min(),
                    group[metric].max(),
                ]
                expected = [
                    saved[metric + "_" + s]
                    for s in ["median", "q1", "q3", "min", "max"]
                ]
                check(
                    "benchmark:summary:" + str(key) + ":" + metric,
                    np.allclose(actual, expected, atol=1e-9, rtol=1e-12),
                )
        benchmark_caveat = {
            "sampled_timed_peak_rss_mib_min": float(repeats.peak_rss_mb.min()),
            "sampled_timed_peak_rss_mib_max": float(repeats.peak_rss_mb.max()),
            "lifetime_high_water_mib_unique": sorted(
                repeats.process_high_water_rss_mb.unique().tolist()
            ),
            "interpretation": "Lifetime high-water RSS includes process launch, imports, loading and warmup history; it is not the sampled timed-stage peak. Uniform high-water values do not establish identical timed-stage allocation. Do not substitute it for timed-stage RSS.",
        }
    pd.DataFrame(checks).to_csv(
        output / "release_reconciliation_checks.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(source_rows).to_csv(
        output / "executed_source_reconciliation.csv", index=False, lineterminator="\n"
    )
    result = {
        "status": "VERIFIED_FOR_MANUSCRIPT_PREPARATION"
        if all(c["status"] == "PASS" for c in checks)
        else "FAIL",
        "base_commit": BASE,
        "export_sha256": digest(export),
        "checks": len(checks),
        "pass": sum(c["status"] == "PASS" for c in checks),
        "executed_source_files": len(source_rows),
        "source_byte_identical": sum(r["byte_identical"] for r in source_rows),
        "source_static_cleanup_verified": sum(
            r["documented_static_cleanup"] for r in source_rows
        ),
        "accepted_fingerprint_files": len(json.loads(reference.read_text())),
        "replay_metadata_caveat": replay_caveats,
        "benchmark_scope_caveat": benchmark_caveat,
        "notebook_identity_limit": "Colab export hashes all scientific source/configuration files, but does not hash notebook 09. Delivered notebook bytes match the prior delivered notebook; exact uploaded notebook-file identity is not independently provable from this export.",
        "submission_signoff": "NOT_GRANTED: author metadata, reference checks, journal requirements and owner publication remain pending",
    }
    (output / "RELEASE_READINESS_GATE.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    if result["status"] == "FAIL":
        raise ValueError("Release reconciliation failed; inspect checks")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--accepted", type=Path, required=True)
    p.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(
        json.dumps(
            reconcile(args.export, args.accepted, args.repository, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
