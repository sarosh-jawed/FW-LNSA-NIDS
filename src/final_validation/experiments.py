"""Validation-only sensitivity and frozen primary-pool diagnostic replay."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..confirmatory_protocol import (
    _threshold_value,
    calibrate_detection_threshold,
    predict_from_native_scores,
    prepare_confirmatory_feature_sets,
)
from ..evaluation import evaluate_binary_classification
from ..feature_selection import normalize_feature_weights
from ..fw_lnsa import FWLNSA
from ..representation import binary_median_split
from .diagnostics import (
    category_recalls,
    mrmr_ranking,
    scores_diagnostic,
    score_unique_patterns,
)
from .evidence import read_csv
from .io import Job, table, write_json


def fit_pool(dataset, feature, record, seed, *, cap=10000, budget=None):
    model = FWLNSA(
        method=record["method"],
        n_detectors=int(record["detector_budget"] if budget is None else budget),
        self_threshold=_threshold_value(
            record["method"],
            float(record["self_threshold_config"]),
            feature.fs_size,
            record["self_threshold_scale"],
        ),
        detection_threshold=0.0,
        random_seed=seed,
        max_self_samples=cap,
        deduplicate_candidates=False,
        prediction_chunk_size=250,
    )
    model.fit(
        feature.representation.X_train_bin,
        dataset.y_train,
        feature_weights=feature.weights
        if record["method"] == "weighted_similarity"
        else None,
    )
    return model


def validate_pool(model, dataset, feature, target=0.05, *, memoize=False):
    scores = (
        score_unique_patterns(model, feature.representation.X_validation_bin)
        if memoize
        else model.decision_scores(feature.representation.X_validation_bin)
    )
    calibration = calibrate_detection_threshold(
        scores, dataset.y_validation, method=model.method, target_fpr=target
    )
    model.detection_threshold = calibration.threshold
    predictions = predict_from_native_scores(
        scores, method=model.method, threshold=calibration.threshold
    )
    metrics = evaluate_binary_classification(
        dataset.y_validation, predictions
    ).to_dict()
    if metrics["fpr"] > target + 1e-12:
        raise ValueError("Validation FPR ceiling violated")
    return {"validation_" + k: v for k, v in metrics.items()}, scores, predictions


def locked_records(accepted, dataset):
    lock = json.loads(
        (
            Path(accepted)
            / f"reproducibility/{dataset}/fw_lnsa_configuration_lock.json"
        ).read_text()
    )
    return lock["locked_configurations"]


def development(dataset_key, dataset, features, config, accepted, output, settings):
    output = Path(output)
    records = [
        r
        for r in locked_records(accepted, dataset_key)
        if r["fs_size"] == 20 and np.isclose(r["target_fpr"], 0.05)
    ]
    synthetic = settings.get("synthetic", False)
    variants = []
    # E: original locked FS20 at three caps. H: k sensitivity reuses original
    # FS20 tolerance ratio/direct value and budgets, no configuration selection.
    expanded = copy.deepcopy(config)
    expanded["experiment"]["feature_sizes"] = [5, 10, 15, 20, 30]
    counts = prepare_confirmatory_feature_sets(dataset, expanded)
    for k, feature in counts.items():
        variants.append((f"feature_count_{k}", feature, dataset, 10000))
    for cap in [2500, 5000, 10000]:
        variants.append((f"self_cap_{cap}", features[20], dataset, cap))
    # C: mRMR uses train only. Full training-fitted preprocessing is fixed.
    ranking, redundancy = mrmr_ranking(dataset.X_train, dataset.y_train)
    table(output / "mrmr_ranking.csv", ranking)
    table(
        output / "mrmr_redundancy_nats.csv",
        redundancy.rename_axis("feature").reset_index(),
    )
    top = ranking.head(20)
    names = top.feature.tolist()
    feature = replace(
        features[20],
        selected_features=names,
        weights=normalize_feature_weights(top.relevance_nats),
        representation=binary_median_split(
            dataset.X_train, dataset.X_test, names, X_validation=dataset.X_validation
        ),
        feature_selection_hash="secondary_mrmr",
    )
    variants.append(("mrmr_20", feature, dataset, 10000))
    # Redundancy for original selected feature sets, Spearman and discrete MI.
    for k, original in features.items():
        table(
            output / f"fs{k}_spearman.csv",
            dataset.X_train[original.selected_features]
            .corr(method="spearman")
            .rename_axis("feature")
            .reset_index(),
        )
    if dataset_key == "cicids2017" and "Destination Port" in dataset.X_train:
        noport = replace(
            dataset,
            X_train=dataset.X_train.drop(columns="Destination Port"),
            X_validation=dataset.X_validation.drop(columns="Destination Port"),
            X_test=dataset.X_test.drop(columns="Destination Port"),
        )
        noport_features = prepare_confirmatory_feature_sets(noport, config)
        table(
            output / "no_destination_port_features.csv", noport_features[20].score_table
        )
        variants.append(("no_destination_port_20", noport_features[20], noport, 10000))
    rows = []
    for name, feature, data, cap in variants:
        for record in records:
            for seed in settings["development_seeds"]:
                identity = {
                    "experiment": name,
                    "dataset": dataset_key,
                    "record": record["configuration_id"],
                    "seed": seed,
                    "cap": cap,
                    "features": feature.selected_features,
                    "weights": feature.weights.tolist(),
                    "settings": settings,
                }
                unit = Job(
                    output / "units", f"{name}_{record['method']}_{seed}", identity
                )

                def execute(p):
                    model = fit_pool(
                        data,
                        feature,
                        record,
                        seed,
                        cap=cap,
                        budget=20 if synthetic else None,
                    )
                    start = time.perf_counter()
                    metrics, scores, pred = validate_pool(model, data, feature)
                    duration = time.perf_counter() - start
                    row = {
                        "experiment": name,
                        "dataset_key": dataset_key,
                        "method": record["method"],
                        "seed": seed,
                        "k": feature.fs_size,
                        "self_cap": cap,
                        "target_fpr": 0.05,
                        "scope": "SECONDARY_DEVELOPMENT_VALIDATION_ONLY",
                        "validation_score_wall_sec": duration,
                        "validation_records_per_sec": len(pred) / duration,
                        **model.fit_summary_.to_dict(),
                        **metrics,
                        **scores_diagnostic(
                            scores,
                            data.y_validation,
                            model.detection_threshold,
                            model.method,
                        ),
                    }
                    table(p / "result.csv", [row])

                unit.run(execute, synthetic=synthetic)
                rows.extend(read_csv(unit.path / "result.csv").to_dict("records"))
    table(output / "validation_sensitivity.csv", rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(
        ["experiment", "dataset_key", "method", "k", "self_cap"], dropna=False
    )[
        [
            "validation_f1",
            "validation_recall",
            "validation_fpr",
            "retention_rate",
            "retained_detectors",
            "validation_records_per_sec",
        ]
    ].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    table(output / "validation_sensitivity_summary.csv", summary.reset_index())
    write_json(
        output / "selection_boundary.json",
        {
            "test_scored": False,
            "configuration": "original FS20 locks reused, no grid search; mRMR and no-port are development representation diagnostics only",
            "preprocessing": "Original training-fitted encoding, imputation and scaling fixed; MI/medians recalculated on training when representation changes",
            "budget_smoke_only": 20 if synthetic else None,
        },
    )


def replay(dataset_key, dataset, features, accepted, output, settings):
    """Reconstruct locked pools, verify predictions/confusions, add diagnostics.

    Primary table is read-only. No threshold or configuration is selected on test.
    Per-seed/config checkpoints retain one result and all category rows atomically.
    """
    output = Path(output)
    fw = read_csv(Path(accepted) / "evidence/seed_level_csv/fw_lnsa_seed_results.csv")
    records = locked_records(accepted, dataset_key)
    all_rows = []
    all_categories = []
    original_labels = dataset.test_original_labels
    categories = dataset.test_attack_categories
    # NSL test has binary-only labels: training family names do not establish
    # zero test support. Only observed test binary category is reportable.
    universe = (
        set(categories)
        if dataset_key == "nsl_kdd"
        else (
            set(dataset.train_attack_categories)
            | set(dataset.validation_attack_categories)
            | set(categories)
        )
    )
    universe -= {"Normal", "Benign"}
    for record in records:
        feature = features[int(record["fs_size"])]
        for seed in settings["replay_seeds"]:
            identity = {
                "dataset": dataset_key,
                "config": record["configuration_id"],
                "seed": seed,
                "feature_hash": feature.feature_selection_hash,
            }
            unit = Job(
                output / "units", f"{record['configuration_id']}_{seed}", identity
            )

            def execute(p):
                model = fit_pool(dataset, feature, record, seed)
                metrics, validation_scores, _ = validate_pool(
                    model, dataset, feature, float(record["target_fpr"]), memoize=True
                )
                test_scores = score_unique_patterns(
                    model, feature.representation.X_test_bin
                )
                pred = predict_from_native_scores(
                    test_scores,
                    method=model.method,
                    threshold=model.detection_threshold,
                )
                test_metrics = evaluate_binary_classification(
                    dataset.y_test, pred
                ).to_dict()
                historical = fw[
                    (fw.dataset_key == dataset_key)
                    & (fw.configuration_id == record["configuration_id"])
                    & (fw.seed == seed)
                ].iloc[0]
                for field in ["tn", "fp", "fn", "tp", "f1", "recall", "fpr"]:
                    if not np.isclose(
                        test_metrics[field],
                        historical["test_" + field],
                        rtol=0,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            f"Frozen replay mismatch {dataset_key}/{seed}/{field}"
                        )
                if not np.isclose(
                    model.detection_threshold,
                    historical.detection_threshold,
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ValueError("Frozen threshold mismatch")
                base = {
                    "dataset_key": dataset_key,
                    "configuration_id": record["configuration_id"],
                    "fs_size": feature.fs_size,
                    "method": record["method"],
                    "target_fpr": record["target_fpr"],
                    "seed": seed,
                }
                # Coverage is descriptive exact nearest-detector score support,
                # not a volume estimate for the full 2^k non-self space.
                selfspace = feature.representation.X_train_bin[dataset.y_train == 0]
                used = model._sample_self_space(selfspace)
                detector_keys = [bytes(x) for x in model.detector_pool_.detectors]
                row = {
                    **base,
                    **model.fit_summary_.to_dict(),
                    "threshold": model.detection_threshold,
                    "replay_matches_original": True,
                    "candidate_deduplication": False,
                    "self_deduplication": True,
                    "unique_mature_detectors": len(set(detector_keys)),
                    "mature_duplicate_count": len(detector_keys)
                    - len(set(detector_keys)),
                    "self_sample_without_replacement": True,
                    "self_used_pre_dedup": len(used),
                    **{"test_" + k: v for k, v in test_metrics.items()},
                    **metrics,
                    **{
                        "validation_" + k: v
                        for k, v in scores_diagnostic(
                            validation_scores,
                            dataset.y_validation,
                            model.detection_threshold,
                            model.method,
                        ).items()
                    },
                    **{
                        "test_" + k: v
                        for k, v in scores_diagnostic(
                            test_scores,
                            dataset.y_test,
                            model.detection_threshold,
                            model.method,
                        ).items()
                    },
                }
                table(p / "replay.csv", [row])
                cats = category_recalls(
                    dataset.y_test, pred, original_labels, categories, universe
                )
                table(p / "categories.csv", [{**base, **r} for r in cats])
                if (
                    feature.fs_size == 20
                    and np.isclose(record["target_fpr"], 0.05)
                    and seed == 100
                ):
                    # Private benchmark state; never publish pickle/raw records.
                    artifact = {
                        "model": model,
                        "X": feature.representation.X_test_bin,
                        "threshold": model.detection_threshold,
                        "expected_predictions": pred,
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

            unit.run(execute)
            all_rows.extend(read_csv(unit.path / "replay.csv").to_dict("records"))
            all_categories.extend(
                read_csv(unit.path / "categories.csv").to_dict("records")
            )
    table(output / "replay_checks.csv", all_rows)
    table(output / "category_seed_recalls.csv", all_categories)
    # Zero-support categories stay missing, never coerced to zero.
    cat = pd.DataFrame(all_categories)
    group = ["dataset_key", "fs_size", "method", "target_fpr", "category"]
    summary = []
    from ..manuscript_analysis import mean_std_ci

    for key, g in cat.groupby(group):
        support = int(g.support.iloc[0])
        summary.append(
            {
                **dict(zip(group, key)),
                "support": support,
                "status": "ESTIMABLE" if support else "NOT_ESTIMABLE",
                **mean_std_ci(g.recall),
            }
        )
    table(output / "category_summary.csv", summary)
