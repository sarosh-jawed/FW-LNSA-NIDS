#!/usr/bin/env python3
"""Build a small public artifact set from immutable accepted and Colab archives.

No model fitting, threshold selection or raw-data loading occurs here.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.final_validation.evidence import read_csv, verify_accepted_fingerprints  # noqa: E402
from scripts.verify_final_validation import verify_export  # noqa: E402
from scripts.publication_figures import make_figures  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrated_replay(frame):
    """Export the final cutoff, never the stale pre-calibration fit summary."""
    result = frame.copy()
    if not np.allclose(result.threshold, result.test_threshold, rtol=0, atol=1e-12):
        raise ValueError("Replay final thresholds disagree")
    result["calibrated_detection_threshold"] = result.threshold
    return result.drop(columns=["detection_threshold", "threshold"], errors="ignore")


def build(export, accepted, output):
    export, accepted, output = map(Path, (export, accepted, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty; preserve earlier artifacts")
    verification = verify_export(export)
    if not verification["eligible_for_independent_final_review"]:
        raise ValueError("Full verified empirical gates required")
    repository = Path(__file__).resolve().parents[1]
    verify_accepted_fingerprints(
        accepted, repository / "docs/final_revision/accepted_file_hashes.json"
    )
    if (
        sha(accepted / "PACKAGE_MANIFEST.json")
        != verification["source_identity"]["accepted"]
    ):
        raise ValueError("Accepted archive does not match executed source identity")
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    tables.mkdir()
    provenance = {}
    with zipfile.ZipFile(export) as archive:

        def read(name):
            with archive.open(name) as f:
                return pd.read_csv(f, float_precision="round_trip")

        def metadata(name):
            return json.loads(archive.read(name))

        def source(scope, name):
            digest = (
                hashlib.sha256(archive.read(name)).hexdigest()
                if scope == "colab_export"
                else sha(accepted / name)
            )
            return {"archive": scope, "path": name, "sha256": digest}

        def save(
            name,
            frame,
            sources,
            transformation="Exact selected columns; original numeric precision retained",
        ):
            path = tables / (name + ".csv")
            frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
            provenance[str(path.relative_to(output))] = {
                "sources": [source(*s) for s in sources],
                "transformation": transformation,
                "rows": len(frame),
                "columns": list(frame.columns),
            }
            return frame

        for name, original in [
            ("original_fw_seed_metrics", "fw_lnsa_seed_results.csv"),
            ("original_reference_seed_metrics", "baseline_seed_results.csv"),
        ]:
            relative = "evidence/seed_level_csv/" + original
            frame = read_csv(accepted / relative)
            keep = [
                c
                for c in frame
                if c
                in [
                    "dataset_key",
                    "method",
                    "model",
                    "fs_size",
                    "target_fpr",
                    "seed",
                    "configuration_id",
                    "detector_budget",
                    "self_threshold",
                    "detection_threshold",
                    "retained_detectors",
                    "retention_rate",
                ]
                or (
                    c.startswith(("test_", "validation_"))
                    and any(
                        c.endswith("_" + m)
                        for m in [
                            "tn",
                            "fp",
                            "fn",
                            "tp",
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "fpr",
                            "specificity",
                            "balanced_accuracy",
                            "mcc",
                            "roc_auc",
                            "pr_auc",
                        ]
                    )
                )
            ]
            save(name, frame[keep], [("accepted", relative)])
        for name, rel in {
            "original_predictive_metrics": "evidence/all_predictive_metrics.csv",
            "original_paired_tests_120": "evidence/paired_120_regenerated.csv",
            "calibration": "evidence/calibration_traceability.csv",
            "category_recall": "cicids2017_replay/category_summary.csv",
            "cicids_category_map": "cicids2017_data/category_map.csv",
            "controlled_benchmark": "benchmark/benchmark_summary.csv",
            "benchmark_repeats": "benchmark/benchmark_repeats.csv",
            "sample120k_metrics": "secondary_report/sample120k_metric_summary.csv",
            "sample120k_paired": "secondary_report/sample120k_paired_descriptive.csv",
            "sample120k_overlap": "secondary_report/sample120k_cross_study_partition_overlap.csv",
            "sample120k_composition": "sample120k/composition.csv",
            "sample120k_calibration": "secondary_report/sample120k_calibration.csv",
            "no_port_features": "cicids2017_development/no_destination_port_features.csv",
        }.items():
            save(name, read(rel), [("colab_export", rel)])
        performance_rel = "evidence/tables_csv/confirmatory_performance.csv"
        save(
            "original_performance",
            read_csv(accepted / performance_rel),
            [("accepted", performance_rel)],
        )
        secondary = read("sample120k/test_seed_results.csv")
        keep = [
            c
            for c in secondary
            if c
            in [
                "method",
                "seed",
                "fs_size",
                "target_fpr",
                "configuration_id",
                "detector_budget",
                "self_threshold",
                "detection_threshold",
                "retained_detectors",
                "retention_rate",
            ]
            or (
                c.startswith(("test_", "validation_"))
                and any(
                    c.endswith("_" + m)
                    for m in ["f1", "recall", "fpr", "tn", "fp", "fn", "tp"]
                )
            )
        ]
        save(
            "sample120k_seed_metrics",
            secondary[keep],
            [("colab_export", "sample120k/test_seed_results.csv")],
        )
        lock = metadata("sample120k/configuration_lock.json")
        lockcols = [
            "method",
            "fs_size",
            "target_fpr",
            "detector_budget",
            "self_threshold_config",
            "self_threshold_scale",
            "self_threshold",
            "configuration_id",
        ]
        save(
            "sample120k_locks",
            pd.DataFrame(lock["locked_configurations"])[lockcols],
            [("colab_export", "sample120k/configuration_lock.json")],
        )
        for name, files in {
            "partition_predictor_overlap": [
                (d + "_data/predictor_overlap.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
            ],
            "partition_pattern_overlap": [
                (f"{d}_data/fs{k}_pattern_overlap.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
                for k in [10, 20]
            ],
            "dataset_composition": [
                (d + "_data/composition.csv", d) for d in ["nsl_kdd", "cicids2017"]
            ],
            "binary_patterns": [
                (f"{d}_data/fs{k}_patterns.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
                for k in [10, 20]
            ],
            "binary_bits": [
                (f"{d}_data/fs{k}_bits.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
                for k in [10, 20]
            ],
            "features_weights": [
                (f"{d}_data/fs{k}_features_weights.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
                for k in [10, 20]
            ],
            "feature_stability": [
                (d + "_stability/selection_frequency.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
            ],
            "development_summary": [
                (d + "_development/validation_sensitivity_summary.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
            ],
            "development_seed_metrics": [
                (d + "_development/validation_sensitivity.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
            ],
            "mrmr_ranking": [
                (d + "_development/mrmr_ranking.csv", d)
                for d in ["nsl_kdd", "cicids2017"]
            ],
        }.items():
            frames = []
            for rel, dataset in files:
                f = read(rel).assign(dataset_key=dataset)
                if "_data/fs" in rel:
                    k = int(Path(rel).name.split("_")[0][2:])
                    f["fs_size"] = k
                    if name == "features_weights":
                        f["rank"] = np.arange(1, len(f) + 1)
                if name == "development_seed_metrics":
                    # Detailed timing/fit state belongs in the private archive.
                    f = f[
                        [
                            c
                            for c in f
                            if c
                            in [
                                "dataset_key",
                                "method",
                                "seed",
                                "experiment",
                                "k",
                                "self_cap",
                                "target_fpr",
                                "retained_detectors",
                                "retention_rate",
                                "unique_self_patterns",
                                "detection_threshold",
                            ]
                            or c
                            in ["validation_f1", "validation_recall", "validation_fpr"]
                        ]
                    ]
                frames.append(f)
            save(
                name,
                pd.concat(frames, ignore_index=True),
                [("colab_export", rel) for rel, _ in files],
                "Concatenate dataset-tagged aggregate or detector-seed diagnostics; no network-record rows",
            )
        pairs = []
        summary = []
        for dataset in ["nsl_kdd", "cicids2017"]:
            rel = dataset + "_stability/pairwise_stability.csv"
            f = read(rel).assign(dataset_key=dataset)
            pairs.append(f)
            for k, g in f.groupby("k"):
                summary.append(
                    {
                        "dataset_key": dataset,
                        "k": k,
                        "resamples": 30,
                        "dependent_pairwise_comparisons": len(g),
                        "mean_jaccard": g.jaccard.mean(),
                        "std_jaccard": g.jaccard.std(),
                        "min_jaccard": g.jaccard.min(),
                        "max_jaccard": g.jaccard.max(),
                        "mean_full_rank_spearman": g.spearman_full_ranking.mean(),
                    }
                )
        sources = [
            ("colab_export", d + "_stability/pairwise_stability.csv")
            for d in ["nsl_kdd", "cicids2017"]
        ]
        save("stability_pairwise", pd.concat(pairs, ignore_index=True), sources)
        save(
            "stability_summary",
            pd.DataFrame(summary),
            sources,
            "Descriptive means and sample SD of dependent pairwise statistics; not independent inferential replicates",
        )
        replay = []
        lockrows = []
        for d in ["nsl_kdd", "cicids2017"]:
            rel = d + "_replay/replay_checks.csv"
            f = calibrated_replay(read(rel))
            cols = [
                "dataset_key",
                "configuration_id",
                "fs_size",
                "method",
                "target_fpr",
                "seed",
                "candidate_detectors",
                "retained_detectors",
                "retention_rate",
                "unique_self_patterns",
                "unique_mature_detectors",
                "mature_duplicate_count",
                "self_used_pre_dedup",
                "calibrated_detection_threshold",
                "validation_unique_finite_scores",
                "test_unique_finite_scores",
                "validation_records_at_threshold",
                "test_records_at_threshold",
            ]
            replay.append(f[cols])
            locked = json.loads(
                (
                    accepted / f"reproducibility/{d}/fw_lnsa_configuration_lock.json"
                ).read_text()
            )
            lockrows.extend(locked["locked_configurations"])
        save(
            "replay_diagnostics",
            pd.concat(replay, ignore_index=True),
            [
                ("colab_export", d + "_replay/replay_checks.csv")
                for d in ["nsl_kdd", "cicids2017"]
            ],
            "Use calibrated final threshold field; omit stale pre-calibration detection_threshold metadata and uncomputed diagnostic AUC columns",
        )
        lockframe = pd.DataFrame(lockrows)
        keep = [
            c
            for c in [
                "dataset_key",
                "method",
                "fs_size",
                "target_fpr",
                "detector_budget",
                "self_threshold_config",
                "self_threshold_scale",
                "self_threshold",
                "configuration_id",
            ]
            if c in lockframe
        ]
        save(
            "original_configuration_locks",
            lockframe[keep],
            [
                ("accepted", f"reproducibility/{d}/fw_lnsa_configuration_lock.json")
                for d in ["nsl_kdd", "cicids2017"]
            ],
        )
        meta = {
            "datasets": {
                d: metadata(d + "_data/dataset_metadata.json")
                for d in ["nsl_kdd", "cicids2017"]
            },
            "secondary120k": lock["dataset_metadata"],
            "partition_and_feature_fingerprints": {
                d: read(d + "_data/reconstruction_checks.csv").to_dict("records")
                for d in ["nsl_kdd", "cicids2017"]
            },
            "raw_input_hashes": {
                d: read(d + "_data/raw_hash_checks.csv").to_dict("records")
                for d in ["nsl_kdd", "cicids2017"]
            },
            "environment": metadata("environment.json"),
            "original_statistics": metadata("evidence/statistical_contract.json"),
            "secondary_statistics": metadata(
                "secondary_report/secondary_statistical_contract.json"
            ),
            "blocked": [
                r for r in metadata("status.json")["jobs"] if r["status"] == "BLOCKED"
            ],
        }
        # Only an explicit allowlist of aggregate fields is published.
        (output / "METHODS_METADATA.json").write_text(json.dumps(meta, indent=2) + "\n")
        (output / "EXECUTED_SOURCE_IDENTITY.json").write_text(
            json.dumps(metadata("execution_identity.json"), indent=2) + "\n"
        )
        schema = read("cicids2017_data/cicids_exact_schema.csv")
        save(
            "cicids_input_columns",
            schema[
                [
                    "file",
                    "input_column",
                    "normalized_column",
                    "retained_as_predictor",
                    "reason",
                    "destination_port_retained",
                ]
            ],
            [("colab_export", "cicids2017_data/cicids_exact_schema.csv")],
        )
        cleaning = read("cicids2017_data/cleaning_report.csv").drop(
            columns=["archive_member"], errors="ignore"
        )
        save(
            "cicids_cleaning",
            cleaning,
            [("colab_export", "cicids2017_data/cleaning_report.csv")],
        )
        # Final review status never overwrites the original execution dashboard.
        (output / "STATUS.json").write_text(
            json.dumps(
                {
                    "handoff_status": "VERIFIED_FOR_MANUSCRIPT_PREPARATION",
                    "empirical_jobs": 12,
                    "scientific_qa_pass": 88,
                    "expensive_studies_rerun": False,
                    "manuscript_modified": False,
                    "github_pushed": False,
                    "submission_final": False,
                    "colab_export_sha256": sha(export),
                },
                indent=2,
            )
            + "\n"
        )
    figure_inventory = make_figures(output)
    (output / "TABLE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    pd.DataFrame(figure_inventory).to_csv(
        output / "FIGURE_INVENTORY.csv", index=False, lineterminator="\n"
    )
    files = {
        p.relative_to(output).as_posix(): sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    manifest = {
        "schema": "fw_lnsa_publication_v1",
        "colab_export_sha256": sha(export),
        "accepted_package_manifest_sha256": sha(accepted / "PACKAGE_MANIFEST.json"),
        "software_license": "Repository MIT license; no third-party raw data redistributed",
        "regeneration": "scripts/build_publication_artifacts.py --export ARCHIVE --accepted ACCEPTED_ROOT --output NEW_EMPTY_FOLDER",
        "files": files,
    }
    (output / "PUBLICATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True, type=Path)
    p.add_argument("--accepted", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = build(args.export, args.accepted, args.output)
    print("Curated publication files:", len(result["files"]))


if __name__ == "__main__":
    main()
