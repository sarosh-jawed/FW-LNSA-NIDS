"""Original evidence regression checks and separately versioned statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..manuscript_analysis import paired_fw_lnsa_analysis, mean_std_ci, holm_adjust
from .io import sha256, table, write_json

KEYS = ["dataset_key", "method", "fs_size", "target_fpr", "seed"]


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def verify_accepted_fingerprints(accepted, hash_reference):
    """Always verify immutable inputs, including before a cached-job resume."""
    accepted = Path(accepted)
    expected = json.loads(Path(hash_reference).read_text())
    checks = []
    for relative, digest in expected.items():
        path = accepted / relative
        passed = path.is_file() and sha256(path) == digest
        checks.append({"check": "accepted_bytes:" + relative, "passed": passed})
        if not passed:
            raise ValueError("Accepted evidence fingerprint mismatch: " + relative)
    return checks


def audit_evidence(accepted, output, hash_reference):
    accepted, output = Path(accepted), Path(output)
    checks = verify_accepted_fingerprints(accepted, hash_reference)
    fw = read_csv(accepted / "evidence/seed_level_csv/fw_lnsa_seed_results.csv")
    baseline = read_csv(accepted / "evidence/seed_level_csv/baseline_seed_results.csv")
    for name, frame, key, expected_rows in [
        ("fw", fw, KEYS, 480),
        (
            "baseline",
            baseline,
            ["dataset_key", "model", "fs_size", "target_fpr", "seed"],
            960,
        ),
    ]:
        checks.append(
            {"check": f"{name}:row_count", "passed": len(frame) == expected_rows}
        )
        checks.append(
            {"check": f"{name}:unique_keys", "passed": not frame.duplicated(key).any()}
        )
        group = [c for c in key if c != "seed"]
        checks.append(
            {
                "check": f"{name}:every_seed_100_119",
                "passed": all(
                    set(g.seed) == set(range(100, 120)) for _, g in frame.groupby(group)
                ),
            }
        )
        for metric in ["test_f1", "test_recall", "test_fpr", "validation_fpr"]:
            checks.append(
                {
                    "check": f"{name}:{metric}:range",
                    "passed": bool(frame[metric].between(0, 1).all()),
                }
            )
        for partition in ["validation", "test"]:
            tn, fp, fn, tp = (
                frame[f"{partition}_{v}"].to_numpy() for v in ["tn", "fp", "fn", "tp"]
            )
            for metric, calculated in [
                ("fpr", fp / (tn + fp)),
                ("recall", tp / (tp + fn)),
                (
                    "f1",
                    np.divide(
                        2 * tp,
                        2 * tp + fp + fn,
                        out=np.zeros(len(tp), float),
                        where=(2 * tp + fp + fn) != 0,
                    ),
                ),
            ]:
                checks.append(
                    {
                        "check": f"{name}:{partition}_{metric}:confusion",
                        "passed": bool(
                            np.allclose(
                                calculated,
                                frame[f"{partition}_{metric}"],
                                atol=1e-12,
                                rtol=0,
                            )
                        ),
                    }
                )
        checks.append(
            {
                "check": f"{name}:validation_ceiling",
                "passed": bool(
                    (frame.validation_fpr <= frame.target_fpr + 1e-12).all()
                ),
            }
        )
    for dataset, group in fw.groupby("dataset_key"):
        manifest = json.loads(
            (
                accepted / f"reproducibility/{dataset}/fw_lnsa_execution_manifest.json"
            ).read_text()
        )
        lock = json.loads(
            (
                accepted / f"reproducibility/{dataset}/fw_lnsa_configuration_lock.json"
            ).read_text()
        )
        checks.append(
            {
                "check": f"{dataset}:lock_count",
                "passed": len(lock["locked_configurations"]) == 12,
            }
        )
        for _, row in group.iterrows():
            selected = [
                r
                for r in lock["locked_configurations"]
                if r["configuration_id"] == row.configuration_id
            ]
            if (
                len(selected) != 1
                or int(selected[0]["detector_budget"]) != int(row.detector_budget)
                or float(selected[0]["self_threshold_config"])
                != float(row.self_threshold_config)
            ):
                raise ValueError("Seed row disagrees with configuration lock")
        for part in ["train", "validation", "test"]:
            c = part + "_partition_hash"
            checks.append(
                {
                    "check": f"{dataset}:{c}",
                    "passed": set(group[c]) == {manifest["partition_hashes"][c]},
                }
            )
        for k, g in group.groupby("fs_size"):
            checks.append(
                {
                    "check": f"{dataset}:FS{k}:feature_hash",
                    "passed": set(g.feature_selection_hash)
                    == {manifest["feature_selection_hashes"][f"FS-{k}"]},
                }
            )
    original = read_csv(accepted / "evidence/tables_csv/paired_weighted_vs_hamming.csv")
    regenerated = paired_fw_lnsa_analysis(fw)
    table(output / "paired_120_regenerated.csv", regenerated)
    join = ["dataset_key", "fs_size", "target_fpr", "metric"]
    merged = original.merge(
        regenerated,
        on=join,
        suffixes=("_original", "_regenerated"),
        validate="one_to_one",
    )
    reconciliation = []
    for metric in [
        "mean_difference_weighted_minus_hamming",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "p_value",
        "holm_adjusted_p",
        "weighted_wins",
        "ties",
        "weighted_losses",
    ]:
        delta = (merged[metric + "_regenerated"] - merged[metric + "_original"]).abs()
        for i, row in merged.iterrows():
            reconciliation.append(
                {
                    **{k: row[k] for k in join},
                    "field": metric,
                    "original": row[metric + "_original"],
                    "regenerated": row[metric + "_regenerated"],
                    "abs_difference": delta.iloc[i],
                    "matches_at_1e_12": bool(delta.iloc[i] <= 1e-12),
                }
            )
    recon = table(output / "statistical_reconciliation.csv", reconciliation)
    checks.append({"check": "120_statistical_rows", "passed": len(merged) == 120})
    checks.append(
        {
            "check": "paired_statistics_reproduce",
            "passed": bool(recon.matches_at_1e_12.all()),
        }
    )
    # A 12-test family is a sensitivity only; it never replaces original 120.
    alternative = regenerated[regenerated.metric == "test_f1"].copy()
    alternative["holm_12_sensitivity_only"] = holm_adjust(alternative.p_value)
    alternative["holm_120_original_family"] = alternative.holm_adjusted_p
    table(output / "holm_12_vs_120_sensitivity.csv", alternative)
    calibration = []
    for approach, frame, name_col in [
        ("FW-LNSA", fw, "method"),
        ("Reference", baseline, "model"),
    ]:
        for key, g in frame.groupby(["dataset_key", name_col, "fs_size", "target_fpr"]):
            row = dict(zip(["dataset_key", "model", "fs_size", "target_fpr"], key))
            row.update(approach=approach, seeds=len(g))
            for part in ["validation", "test"]:
                fpr = g[part + "_fpr"]
                met = fpr <= g.target_fpr + 1e-12
                row[part + "_target_met_count"] = int(met.sum())
                row[part + "_target_met_denominator"] = len(g)
                # Exact binomial CI is descriptive across shared-data seeds.
                low = (
                    0.0
                    if met.sum() == 0
                    else stats.beta.ppf(0.025, met.sum(), len(g) - met.sum() + 1)
                )
                high = (
                    1.0
                    if met.sum() == len(g)
                    else stats.beta.ppf(0.975, met.sum() + 1, len(g) - met.sum())
                )
                row[part + "_seed_compliance_ci_low"] = low
                row[part + "_seed_compliance_ci_high"] = high
                for metric, values in [
                    ("fpr", fpr),
                    ("absolute_target_error", (fpr - g.target_fpr).abs()),
                ]:
                    row.update(
                        {
                            part + "_" + metric + "_" + k: v
                            for k, v in mean_std_ci(values).items()
                        }
                    )
            for field in [
                "detection_threshold",
                "retained_detectors",
                "retention_rate",
            ]:
                if field in g:
                    row.update(
                        {
                            field + "_median": g[field].median(),
                            field + "_min": g[field].min(),
                            field + "_max": g[field].max(),
                        }
                    )
            calibration.append(row)
    table(output / "calibration_traceability.csv", calibration)
    metrics = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_fpr",
        "test_specificity",
        "test_balanced_accuracy",
        "test_mcc",
        "test_roc_auc",
        "test_pr_auc",
    ]
    supplement = []
    for approach, frame, name_col in [
        ("FW-LNSA", fw, "method"),
        ("Reference", baseline, "model"),
    ]:
        for key, g in frame.groupby(["dataset_key", name_col, "fs_size", "target_fpr"]):
            for metric in metrics:
                supplement.append(
                    {
                        **dict(
                            zip(["dataset_key", "model", "fs_size", "target_fpr"], key)
                        ),
                        "approach": approach,
                        "metric": metric,
                        **mean_std_ci(g[metric]),
                    }
                )
    table(output / "all_predictive_metrics.csv", supplement)
    table(output / "integrity_checks.csv", checks)
    write_json(
        output / "statistical_contract.json",
        {
            "original_family": 120,
            "metrics_per_condition": 10,
            "conditions": 12,
            "bootstrap_iterations": 5000,
            "bootstrap_seed": "2026 + sequential output row index",
            "interval": "paired percentile bootstrap of mean difference, 95%",
            "mean_intervals": "Student t, 19 df",
            "wilcoxon": "scipy.stats.wilcoxon(zero_method=wilcox, alternative=two-sided, method=auto); allclose-to-zero shortcut returns (0,1)",
            "near_zero_caveat": "Historical allclose shortcut uses numpy default 1e-8; rank effect excludes isclose-zero; win/tie/loss tolerance 1e-12. Retained exactly.",
            "uncertainty": "Detector seed randomness conditional on common fixed data and validation, not sampling/domain uncertainty",
            "accepted_sha256": sha256(accepted / "PACKAGE_MANIFEST.json"),
        },
    )
    if not all(r["passed"] for r in checks):
        raise ValueError("Evidence regression failed; see integrity_checks.csv")
