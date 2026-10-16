"""Synthetic integration fixtures, never manuscript evidence."""

from pathlib import Path
import pickle

import numpy as np

from .diagnostics import category_recalls, scores_diagnostic
from .experiments import fit_pool, validate_pool
from .io import table, write_json


def smoke_replay(dataset, features, output):
    rows = []
    for method in ["hamming", "weighted_similarity"]:
        record = {
            "method": method,
            "detector_budget": 20,
            "self_threshold_config": 0.05 if method == "hamming" else 0.85,
            "self_threshold_scale": "ratio" if method == "hamming" else "direct",
        }
        model = fit_pool(dataset, features[20], record, 100, cap=100)
        metrics, scores, pred = validate_pool(model, dataset, features[20])
        again = pickle.loads(pickle.dumps(model, protocol=5))
        if not np.array_equal(
            scores, again.decision_scores(features[20].representation.X_validation_bin)
        ):
            raise ValueError("Serialization round trip failed")
        base = {
            "dataset_key": "SYNTHETIC",
            "method": method,
            "fs_size": 20,
            "target_fpr": 0.05,
            "seed": 100,
        }
        rows.append(
            {
                **base,
                **metrics,
                **scores_diagnostic(
                    scores, dataset.y_validation, model.detection_threshold, method
                ),
            }
        )
        # Same path contract as real replay; benchmark synthetic only.
        p = Path(output) / "units" / method
        p.mkdir(parents=True, exist_ok=True)
        (p / "benchmark_state.pkl").write_bytes(
            pickle.dumps(
                {
                    "model": model,
                    "X": features[20].representation.X_validation_bin,
                    "threshold": model.detection_threshold,
                    "expected_predictions": pred,
                    "preprocessing": {
                        "scaler": dataset.scaler,
                        "features": features[20].selected_features,
                        "medians": features[20].representation.medians,
                    },
                    "base": base,
                },
                protocol=5,
            )
        )
    table(Path(output) / "synthetic_replay.csv", rows)
    table(
        Path(output) / "category_missing_support_test.csv",
        category_recalls(
            dataset.y_validation,
            pred,
            dataset.validation_original_labels,
            dataset.validation_attack_categories,
            {"Attack (Unspecified)", "AbsentCategory"},
        ),
    )
    write_json(
        Path(output) / "scope.json",
        {"status": "VERIFIED_SYNTHETIC_ONLY", "original_metric_comparison": False},
    )
