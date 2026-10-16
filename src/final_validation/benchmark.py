"""One measured repeat per clean subprocess, frozen binary-input predictions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

import numpy as np

from ..confirmatory_protocol import predict_from_native_scores
from ..utils import PeakRSSMonitor
from .io import Job, environment, table, write_json


def worker(state_path, output, warmups=1):
    # Load only trusted locally generated state; never unpickle third-party files.
    artifact = pickle.loads(Path(state_path).read_bytes())
    model = artifact["model"]
    X = artifact["X"]
    threshold = artifact["threshold"]

    def predict():
        return predict_from_native_scores(
            model.decision_scores(X), method=model.method, threshold=threshold
        )

    for _ in range(warmups):
        warm = predict()
        if not np.array_equal(warm, artifact["expected_predictions"]):
            raise ValueError("Warm-up prediction hash mismatch")
    with PeakRSSMonitor(interval_seconds=0.001) as monitor:
        start = time.perf_counter_ns()
        pred = predict()
        elapsed = (time.perf_counter_ns() - start) / 1e9
    if not np.array_equal(pred, artifact["expected_predictions"]):
        raise ValueError("Measured prediction hash mismatch")
    rss = monitor.result
    state = {
        "method": model.method,
        "self_threshold": model.self_threshold,
        "detection_threshold": model.detection_threshold,
        "detectors": model.detector_pool_.detectors,
        "weights": model.feature_weights_,
        "vector_length": model.vector_length_,
    }
    from .deployment import DeploymentPipeline

    state_meta = artifact["preprocessing"]
    if "encoded_columns" in state_meta:
        deployed = DeploymentPipeline(
            model=model,
            scaler=state_meta["scaler"],
            encoded_columns=state_meta["encoded_columns"],
            selected_features=state_meta["selected_features"],
            binary_medians=state_meta["medians"],
            imputation_medians=state_meta["imputation_medians"],
            dataset_key=artifact["base"]["dataset_key"],
        )
    else:
        deployed = {"model": model, "preprocessing": state_meta}  # synthetic fixture
    write_json(
        output,
        {
            **artifact["base"],
            "elapsed_sec": elapsed,
            "records": len(X),
            "throughput_records_per_sec": len(X) / elapsed,
            "peak_rss_mb": rss.peak_rss_mb,
            "baseline_rss_mb": rss.baseline_rss_mb,
            "prediction_rss_delta_mb": rss.peak_delta_mb,
            "process_high_water_rss_mb": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            / (1024 if sys.platform != "darwin" else 1024**2),
            "rss_floor_flag": bool(rss.peak_delta_mb <= 0.1),
            "rss_floor_interpretation": "sampled near-zero increment is allocator reuse/resolution, not zero total memory",
            "detector_state_pickle_bytes": len(
                pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
            ),
            "deployed_pipeline_pickle_bytes": len(
                pickle.dumps(deployed, protocol=pickle.HIGHEST_PROTOCOL)
            ),
            "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
            "scope": "binary scoring + threshold only; excludes preprocessing/load/fit/IO; 1 warm-up; independent process per repeat",
            "environment": environment(),
        },
    )


def prepare_strict_states(root, accepted):
    """Add FS20,target.01,seed100 cases to probe historical high-RSS conditions."""
    from .experiments import fit_pool, validate_pool, locked_records
    from .evidence import read_csv
    from .io import sha256

    root, accepted = Path(root), Path(accepted)
    historical = read_csv(accepted / "evidence/seed_level_csv/fw_lnsa_seed_results.csv")
    for dataset_key in ["nsl_kdd", "cicids2017"]:
        cache = root / (dataset_key + "_data") / "prepared.pkl"
        manifest = json.loads((cache.parent / "manifest.json").read_text())
        if sha256(cache) != manifest["outputs"]["prepared.pkl"]:
            raise ValueError("Private prepared-state hash mismatch")
        dataset, features = pickle.loads(cache.read_bytes())
        feature = features[20]
        for record in locked_records(accepted, dataset_key):
            if record["fs_size"] != 20 or not np.isclose(record["target_fpr"], 0.01):
                continue
            job = Job(
                root / "benchmark_strict_states",
                dataset_key + "_" + record["configuration_id"],
                {
                    "configuration_id": record["configuration_id"],
                    "seed": 100,
                    "prepared_sha256": sha256(cache),
                },
            )

            def execute(p):
                model = fit_pool(dataset, feature, record, 100)
                validate_pool(model, dataset, feature, 0.01)
                scores = model.decision_scores(feature.representation.X_test_bin)
                predictions = predict_from_native_scores(
                    scores, method=model.method, threshold=model.detection_threshold
                )
                from ..evaluation import evaluate_binary_classification

                metrics = evaluate_binary_classification(
                    dataset.y_test, predictions
                ).to_dict()
                original = historical[
                    (historical.dataset_key == dataset_key)
                    & (historical.configuration_id == record["configuration_id"])
                    & (historical.seed == 100)
                ].iloc[0]
                for field in ["tn", "fp", "fn", "tp", "f1", "recall", "fpr"]:
                    if not np.isclose(
                        metrics[field], original["test_" + field], atol=1e-12, rtol=0
                    ):
                        raise ValueError(
                            "Strict frozen benchmark reconstruction mismatch"
                        )
                base = {
                    "dataset_key": dataset_key,
                    "configuration_id": record["configuration_id"],
                    "method": record["method"],
                    "fs_size": 20,
                    "target_fpr": 0.01,
                    "seed": 100,
                }
                artifact = {
                    "model": model,
                    "X": feature.representation.X_test_bin,
                    "threshold": model.detection_threshold,
                    "expected_predictions": predictions,
                    "preprocessing": {
                        "scaler": dataset.scaler,
                        "imputation_medians": dataset.imputation_medians,
                        "selected_features": feature.selected_features,
                        "weights": feature.weights,
                        "medians": feature.representation.medians,
                        "encoded_columns": list(dataset.X_train.columns),
                    },
                    "base": base,
                }
                (p / "benchmark_state.pkl").write_bytes(
                    pickle.dumps(artifact, protocol=5)
                )
                table(
                    p / "frozen_reconstruction.csv",
                    [{**base, **metrics, "matches_original": True}],
                )

            job.run(execute)


def benchmark(root, output, repeats=10, synthetic=False, accepted=None):
    if not synthetic:
        if accepted is None:
            raise ValueError(
                "Accepted evidence required for strict frozen benchmark cases"
            )
        prepare_strict_states(root, accepted)
    states = sorted(Path(root).glob("*_replay/units/*/benchmark_state.pkl"))
    states += sorted(Path(root).glob("benchmark_strict_states/*/benchmark_state.pkl"))
    if not states:
        raise FileNotFoundError("No verified frozen replay benchmark states")
    results = []
    for state in states:
        name = state.parent.name
        for repeat in range(repeats):
            job = Job(
                output,
                f"{state.parents[2].name}_{name}_{repeat}",
                {
                    "state_hash": hashlib.sha256(state.read_bytes()).hexdigest(),
                    "repeat": repeat,
                    "warmup": 1,
                    "batch": 250,
                    "threads": 1,
                },
            )

            def execute(p):
                env = {
                    **os.environ,
                    **{
                        k: "1"
                        for k in [
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                        ]
                    },
                }
                command = [
                    sys.executable,
                    "-m",
                    "src.final_validation.benchmark",
                    "--worker",
                    str(state.resolve()),
                    "--output",
                    str((p / "repeat.json").resolve()),
                ]
                subprocess.run(command, check=True, env=env, timeout=1800)

            job.run(execute, synthetic=synthetic)
            row = json.loads((job.path / "repeat.json").read_text())
            row.pop("environment")
            results.append({**row, "repeat": repeat})
    frame = table(Path(output) / "benchmark_repeats.csv", results)
    summaries = []
    for key, g in frame.groupby(["dataset_key", "fs_size", "method", "target_fpr"]):
        row = dict(zip(["dataset_key", "fs_size", "method", "target_fpr"], key))
        row["repeats"] = len(g)
        for field in [
            "elapsed_sec",
            "throughput_records_per_sec",
            "peak_rss_mb",
            "prediction_rss_delta_mb",
            "process_high_water_rss_mb",
            "detector_state_pickle_bytes",
            "deployed_pipeline_pickle_bytes",
        ]:
            row.update(
                {
                    field + "_median": g[field].median(),
                    field + "_q1": g[field].quantile(0.25),
                    field + "_q3": g[field].quantile(0.75),
                    field + "_min": g[field].min(),
                    field + "_max": g[field].max(),
                }
            )
        summaries.append(row)
    table(Path(output) / "benchmark_summary.csv", summaries)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worker(args.worker, args.output)
