"""Dependency-ordered CLI orchestration with honest experiment status."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import traceback
import zipfile

import numpy as np
import pandas as pd
import yaml

from ..confirmatory_protocol import (
    prepare_confirmatory_feature_sets,
    run_tuning_stage,
    summarize_tuning_results,
    build_configuration_lock,
    run_final_stage,
)
from ..preprocessing import PreparedDataset
from ..research_execution import append_jsonl, load_jsonl_frame
from sklearn.preprocessing import MinMaxScaler
from . import VERSION
from .benchmark import benchmark
from .data import configured_original, data_audit, raw_preflight
from .diagnostics import feature_stability
from .evidence import audit_evidence, read_csv, verify_accepted_fingerprints
from .experiments import development, replay
from .io import Job, environment, sha256, signature, table, write_json

JOB_NAMES = [
    "evidence",
    "data",
    "stability",
    "development",
    "replay",
    "benchmark",
    "sample120k",
    "report",
]


def synthetic_dataset():
    rng = np.random.default_rng(991)
    frames = []
    labels = []
    for size in [160, 80, 90]:
        y = np.tile([0, 1], size // 2 + 1)[:size]
        values = rng.normal(size=(size, 35))
        values[:, 0] += y * 2
        values[:, 1] = values[:, 0]
        values[:, 2] = 0
        values[:, 3] = rng.random(size) < 0.02
        frames.append(pd.DataFrame(values, columns=[f"f{i}" for i in range(35)]))
        labels.append(y)
    scaler = MinMaxScaler().fit(frames[0])
    frames = [pd.DataFrame(scaler.transform(x), columns=x.columns) for x in frames]
    original = [pd.Series(np.where(y == 0, "normal", "anomaly")) for y in labels]
    category = [
        pd.Series(np.where(y == 0, "Normal", "Attack (Unspecified)")) for y in labels
    ]
    return PreparedDataset(
        X_train=frames[0],
        X_validation=frames[1],
        X_test=frames[2],
        y_train=labels[0],
        y_validation=labels[1],
        y_test=labels[2],
        train_original_labels=original[0],
        validation_original_labels=original[1],
        test_original_labels=original[2],
        train_attack_categories=category[0],
        validation_attack_categories=category[1],
        test_attack_categories=category[2],
        scaler=scaler,
        metadata={"dataset": "SYNTHETIC"},
    )


def stability(data, output, settings):
    seeds = list(range(3000, 3000 + settings["resamples"]))
    rankings = []
    for seed in seeds:
        job = Job(
            output,
            f"resample_{seed}",
            {
                "seed": seed,
                "fraction": 0.8,
                "scope": "fixed original training preprocessing",
                "rows": len(data.y_train),
                "version": VERSION,
            },
        )

        def execute(p):
            long, _, _ = feature_stability(data.X_train, data.y_train, seeds=[seed])
            table(p / "ranking.csv", long)

        job.run(execute, synthetic=settings["synthetic"])
        rankings.append(read_csv(job.path / "ranking.csv"))
    long, frequency, pairs = feature_stability(
        data.X_train, data.y_train, seeds=seeds, rankings=rankings
    )
    table(Path(output) / "resample_mi_scores.csv", long)
    table(Path(output) / "selection_frequency.csv", frequency)
    table(Path(output) / "pairwise_stability.csv", pairs)
    write_json(
        Path(output) / "contract.json",
        {
            "resamples": len(seeds),
            "seeds": seeds,
            "fraction": 0.8,
            "without_replacement": True,
            "stratified": "binary class",
            "preprocessing_scope": "original full-training encoding/scaling/imputation held fixed; MI only refit within training resamples",
            "scope": "conditional stability, not independently refitted preprocessing",
            "ties": "MI descending, lexical feature name",
        },
    )


def sample120k(config, output, settings):
    """New train/validation/test study; strict deterministic selection tie rules."""
    cfg = copy.deepcopy(config)
    cfg["datasets"]["cicids2017"]["sampling"]["max_total_records"] = 120000
    cfg["experiment"]["feature_sizes"] = [20]
    cfg["experiment"]["target_fprs"] = [0.05]
    cfg["experiment"]["tuning_seeds"] = list(range(200, 205))
    cfg["experiment"]["confirmatory_seeds"] = list(range(300, 320))
    cfg["paths"]["output_dir"] = str(output)
    cfg["secondary_study"] = "120k_v1"
    if settings["synthetic"]:
        cfg["experiment"]["detector_budgets"] = [20]
        cfg["experiment"]["tuning_seeds"] = [200]
        cfg["experiment"]["confirmatory_seeds"] = [300]
        data = synthetic_dataset()
        source = Path(output) / "synthetic_input_fingerprint.json"
        write_json(source, {"synthetic_fixture": 991, "rows": [160, 80, 90]})
        cfg["datasets"]["cicids2017"]["archive_file"] = str(source)
    else:
        from ..confirmatory_protocol import prepare_confirmatory_dataset

        data = prepare_confirmatory_dataset("cicids2017", cfg)
    fs = prepare_confirmatory_feature_sets(data, cfg)
    write_json(Path(output) / "secondary_protocol.json", cfg)
    write_json(Path(output) / "composition_metadata.json", data.metadata)
    from ..confirmatory_protocol import _partition_hashes

    write_json(Path(output) / "partition_hashes.json", _partition_hashes(data))
    table(Path(output) / "features.csv", fs[20].score_table)
    composition = []
    for name in ["train", "validation", "test"]:
        labels = getattr(data, name + "_original_labels")
        cats = getattr(data, name + "_attack_categories")
        for (label, category), group in pd.DataFrame(
            {"original_label": labels, "category": cats}
        ).groupby(["original_label", "category"]):
            composition.append(
                {
                    "partition": name,
                    "original_label": label,
                    "category": category,
                    "records": len(group),
                }
            )
    table(Path(output) / "composition.csv", composition)
    if data.row_provenance:
        sidecars = []
        for partition, frame in data.row_provenance.items():
            frame = frame.copy()
            frame["partition"] = partition
            sidecars.append(frame)
        rows = pd.concat(sidecars, ignore_index=True)
        table(Path(output) / "sample_row_provenance.csv", rows)
        original = Path(output).parent / "cicids2017_data"
        keys = set(zip(rows["__source_file__"], rows["__source_row__"]))
        overlap = []
        for part in ["train", "validation", "test"]:
            path = original / (part + "_row_signatures.csv")
            if path.exists():
                old = read_csv(path)
                oldkeys = set(
                    zip(old.source_file, old.source_row_zero_based_excluding_header)
                )
                overlap.append(
                    {
                        "original_60k_partition": part,
                        "rows_also_in_new_120k_sample": len(oldkeys & keys),
                        "independent_external_sample": False,
                    }
                )
        table(Path(output) / "overlap_with_original_60k.csv", overlap)
    # Existing implementation provides row-by-row JSONL partial recovery.
    tuning_path = Path(output) / "tuning.jsonl"
    tuning = run_tuning_stage(
        "cicids2017",
        data,
        cfg,
        fs,
        existing_results=load_jsonl_frame(tuning_path),
        result_callback=lambda row: append_jsonl(row, tuning_path),
    )
    summary = summarize_tuning_results(tuning, confidence=0.95)
    # Original runtime tie break was timing-sensitive. New secondary tie rule
    # uses deterministic budget and tolerance only, prespecified in config.
    selected = []
    for _, g in summary.groupby(["dataset_key", "method", "fs_size", "target_fpr"]):
        g = g[g.validation_fpr_mean <= g.target_fpr + 1e-12]
        row = (
            g.sort_values(
                [
                    "validation_f1_mean",
                    "validation_recall_mean",
                    "validation_fpr_mean",
                    "detector_budget",
                    "self_threshold_config",
                    "method",
                ],
                ascending=[False, False, True, True, True, True],
            )
            .iloc[0]
            .copy()
        )
        # Build same configuration ID contract as original without its timing tie.
        from ..utils import stable_mapping_signature

        row["configuration_id"] = stable_mapping_signature(
            {
                "dataset": row.dataset_key,
                "method": row.method,
                "fs_size": int(row.fs_size),
                "target_fpr": float(row.target_fpr),
                "detector_budget": int(row.detector_budget),
                "self_threshold_config": float(row.self_threshold_config),
                "self_threshold_scale": row.self_threshold_scale,
            }
        )[:16]
        row["selection_rule"] = (
            "secondary deterministic validation F1/recall/FPR/budget/tolerance/method"
        )
        selected.append(row)
    selected = pd.DataFrame(selected)
    lock = build_configuration_lock("cicids2017", data, fs, selected, cfg)
    write_json(Path(output) / "configuration_lock.json", lock)
    final_path = Path(output) / "test.jsonl"
    cats_path = Path(output) / "categories.jsonl"
    final, categories = run_final_stage(
        "cicids2017",
        data,
        cfg,
        fs,
        lock,
        existing_results=load_jsonl_frame(final_path),
        existing_categories=load_jsonl_frame(cats_path),
        result_callback=lambda row: append_jsonl(row, final_path),
        category_callback=lambda row: append_jsonl(row, cats_path),
    )
    table(Path(output) / "tuning.csv", tuning)
    table(Path(output) / "test_seed_results.csv", final)
    table(Path(output) / "category_seed_results.csv", categories)
    expected = len(cfg["experiment"]["confirmatory_seeds"]) * 2
    if len(final) != expected:
        raise ValueError(f"Incomplete 120k study: {len(final)}/{expected}")


def dashboard(root):
    rows = []
    for name in (
        ["evidence"]
        + [
            f"{key}_{stage}"
            for key in ["nsl_kdd", "cicids2017"]
            for stage in ["data", "stability", "development", "replay"]
        ]
        + ["benchmark", "sample120k", "report"]
    ):
        p = Path(root) / name / "manifest.json"
        state = (
            json.loads(p.read_text())
            if p.exists()
            else {"status": "PENDING_USER_COLAB_RUN"}
        )
        rows.append(
            {"job": name, "status": state["status"], "error": state.get("error", "")}
        )
    rows.append(
        {
            "job": "nsl_kdd_attack_family",
            "status": "BLOCKED",
            "error": "Strict named-label alignment failed:306service entries differ in candidate mirrors; binary-only retained",
        }
    )
    rows.append(
        {
            "job": "scenario_holdout",
            "status": "BLOCKED",
            "error": "Optional extension lacks comparable prespecified class/scenario contract",
        }
    )
    table(Path(root) / "status.csv", rows)
    write_json(
        Path(root) / "status.json",
        {
            "release_status": "PRELIMINARY",
            "version": VERSION,
            "jobs": rows,
            "finalization": "Requires full empirical jobs, user execution/upload and independent subsequent reconciliation. Never automatic manuscript approval.",
        },
    )
    return rows


def export_results(root, destination):
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    exported = {}
    excluded = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.resolve() == destination or p.name.endswith(".tmp"):
                continue
            rel = p.relative_to(root).as_posix()
            if p.suffix in [".pkl", ".npy", ".npz", ".joblib"]:
                excluded.append(rel)
                continue
            z.write(p, rel)
            exported[rel] = sha256(p)
        z.writestr(
            "EXPORT_MANIFEST.json",
            json.dumps(
                {
                    "files": exported,
                    "excluded_private_files": excluded,
                    "scope": "No raw numeric dataset or executable pickle; all retained public job outputs hashed",
                },
                indent=2,
            ),
        )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "full"], required=True)
    parser.add_argument(
        "--accepted",
        type=Path,
        required=True,
        help="Unpacked original writer_handoff directory",
    )
    parser.add_argument(
        "--data-root", type=Path, help="Parent of nsl_kdd and cicids2017 folders"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--jobs", nargs="+", choices=JOB_NAMES + ["all"], default=["all"]
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["nsl_kdd", "cicids2017"],
        default=["nsl_kdd", "cicids2017"],
    )
    parser.add_argument("--include-120k", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args(argv)
    settings = yaml.safe_load(Path("configs/final_validation.yaml").read_text())[
        "profiles"
    ][args.profile]
    settings["profile"] = args.profile
    args.output.mkdir(parents=True, exist_ok=True)
    verify_accepted_fingerprints(
        args.accepted, "docs/final_revision/accepted_file_hashes.json"
    )
    config = configured_original(args.data_root or ".")
    write_json(args.output / "environment.json", environment())
    source_files = (
        list(Path("src").rglob("*.py"))
        + list(Path("scripts").glob("*.py"))
        + list(Path("configs").glob("*.yaml"))
        + [
            Path("requirements-final-validation-lock.txt"),
            Path("docs/final_revision/accepted_file_hashes.json"),
        ]
    )
    identity = {
        "version": VERSION,
        "profile": args.profile,
        "accepted": sha256(args.accepted / "PACKAGE_MANIFEST.json"),
        "packages": environment()["packages"],
        "source_hashes": {str(p): sha256(p) for p in source_files},
    }
    identity_path = args.output / "execution_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(
            "Source/profile/environment changed: choose a new versioned output root. Never overwrite prior results."
        )
    write_json(identity_path, identity)
    jobs = set(JOB_NAMES if "all" in args.jobs else args.jobs)
    if not args.include_120k and "sample120k" not in args.jobs:
        jobs.discard("sample120k")
    failures = []
    try:
        if "evidence" in jobs:
            Job(
                args.output,
                "evidence",
                {"version": VERSION, "accepted_sha": identity["accepted"]},
            ).run(
                lambda p: audit_evidence(
                    args.accepted, p, "docs/final_revision/accepted_file_hashes.json"
                )
            )
        for key in args.datasets:
            data = None
            features = None
            if jobs & {"data", "stability", "development", "replay"}:
                try:
                    if settings["synthetic"]:
                        data = synthetic_dataset()
                        features = prepare_confirmatory_feature_sets(data, config)
                        Job(
                            args.output,
                            key + "_data",
                            {"version": VERSION, "synthetic": True},
                        ).run(
                            lambda p: write_json(
                                p / "synthetic_metadata.json", data.metadata
                            ),
                            synthetic=True,
                        )
                    else:
                        raw_preflight(config, args.accepted, key)
                        job = Job(
                            args.output,
                            key + "_data",
                            {
                                "data_contract": VERSION,
                                "raw": "original",
                                "dataset": key,
                            },
                        )

                        def execute(p):
                            d, f = data_audit(key, config, args.accepted, p)
                            (p / "prepared.pkl").write_bytes(
                                pickle.dumps((d, f), protocol=5)
                            )

                        job.run(execute)
                        # Trust only cache bytes verified by Job.valid().
                        data, features = pickle.loads(
                            (job.path / "prepared.pkl").read_bytes()
                        )
                    for stage, callback in [
                        ("stability", lambda p: stability(data, p, settings)),
                        (
                            "development",
                            lambda p: development(
                                key, data, features, config, args.accepted, p, settings
                            ),
                        ),
                        (
                            "replay",
                            lambda p: replay(
                                key, data, features, args.accepted, p, settings
                            ),
                        ),
                    ]:
                        if stage not in jobs:
                            continue
                        if settings["synthetic"] and stage == "replay":
                            # Original pools cannot be reconciled to synthetic data.
                            from .smoke import smoke_replay

                            def callback(p):
                                return smoke_replay(data, features, p)

                        job = Job(
                            args.output,
                            key + "_" + stage,
                            {
                                "version": VERSION,
                                "settings": settings,
                                "dataset": key,
                                "data_hash": signature(data.metadata),
                            },
                        )
                        try:
                            job.run(callback, synthetic=settings["synthetic"])
                        except Exception as e:
                            failures.append(
                                {
                                    "job": key + "_" + stage,
                                    "error": str(e),
                                    "traceback": traceback.format_exc(),
                                }
                            )
                except Exception as e:
                    failures.append(
                        {
                            "job": key + "_data",
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
            data = features = None
        if "benchmark" in jobs:
            try:
                Job(
                    args.output,
                    "benchmark",
                    {"version": VERSION, "repeats": settings["benchmark_repeats"]},
                ).run(
                    lambda p: benchmark(
                        args.output,
                        p,
                        settings["benchmark_repeats"],
                        settings["synthetic"],
                        args.accepted,
                    ),
                    synthetic=settings["synthetic"],
                )
            except Exception as e:
                failures.append(
                    {
                        "job": "benchmark",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
        if "sample120k" in jobs:
            try:
                if not settings["synthetic"]:
                    raw_preflight(config, args.accepted, "cicids2017")
                Job(
                    args.output,
                    "sample120k",
                    {
                        "version": VERSION,
                        "contract": "120k_v1",
                        "synthetic": settings["synthetic"],
                    },
                ).run(
                    lambda p: sample120k(config, p, settings),
                    synthetic=settings["synthetic"],
                )
            except Exception as e:
                failures.append(
                    {
                        "job": "sample120k",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
        if "report" in jobs:
            from .reporting import report

            # Reports regenerate when more jobs complete; never gate their own
            # dependency updates through a stale completed-job checkpoint.
            report(args.output, args.accepted)
    finally:
        rows = dashboard(args.output)
        write_json(args.output / "structured_failures.json", failures)
        if args.export:
            export_results(args.output, args.export)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    if failures:
        raise SystemExit(
            "Some jobs failed; inspect structured_failures.json and resume after correcting the cause"
        )


if __name__ == "__main__":
    main()
