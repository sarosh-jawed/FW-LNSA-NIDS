#!/usr/bin/env python3
"""Strict candidate-canonical audit; no raw-file edits or attack-family inference."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src.final_validation.data import configured_original  # noqa: E402
from src.final_validation.io import sha256, table, write_json  # noqa: E402
from src.confirmatory_protocol import prepare_confirmatory_dataset  # noqa: E402
from src.preprocessing import (
    NSL_KDD_FEATURE_COLUMNS,
    NSL_KDD_CATEGORICAL_COLUMNS,
    read_nsl_kdd_file,
    to_binary_label,
)  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical1", required=True, type=Path)
    parser.add_argument("--canonical2", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--accepted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if sha256(args.canonical1) != sha256(args.canonical2):
        raise ValueError("Candidate mirrors disagree; no family claims")
    config = configured_original(args.data_root)
    dataset = prepare_confirmatory_dataset("nsl_kdd", config)
    original = read_nsl_kdd_file(config["datasets"]["nsl_kdd"]["test_file"])
    candidate = read_nsl_kdd_file(args.canonical1)
    if len(original) != len(candidate):
        raise ValueError("Row-count mismatch")
    checks = []
    mismatches = []
    for feature in NSL_KDD_FEATURE_COLUMNS:
        mask = original[feature] != candidate[feature]
        checks.append(
            {
                "feature": feature,
                "row_mismatches": int(mask.sum()),
                "exact_values_equal": not mask.any(),
            }
        )
        if mask.any():
            counts = (
                pd.DataFrame(
                    {
                        "accepted_value": original.loc[mask, feature],
                        "candidate_value": candidate.loc[mask, feature],
                    }
                )
                .value_counts()
                .reset_index(name="rows")
            )
            mismatches.extend(
                [{"feature": feature, **row} for row in counts.to_dict("records")]
            )
    table(args.output / "predictor_alignment_41.csv", checks)
    table(args.output / "mismatch_value_summary.csv", mismatches)
    labels_equal = np.array_equal(
        original.label.map(to_binary_label), candidate.label.map(to_binary_label)
    )
    exact = all(r["exact_values_equal"] for r in checks) and labels_equal
    raw = candidate.drop(columns=["label", "difficulty"])
    encoded = pd.get_dummies(
        raw, columns=NSL_KDD_CATEGORICAL_COLUMNS, dtype=float
    ).reindex(columns=dataset.X_train.columns, fill_value=0.0)
    transformed = pd.DataFrame(
        dataset.scaler.transform(encoded), columns=dataset.X_train.columns
    )
    primary = pd.read_csv(
        args.accepted / "evidence/seed_level_csv/fw_lnsa_seed_results.csv"
    )
    primary = primary[primary.dataset_key == "nsl_kdd"]
    projections = []
    for k, g in primary.groupby("fs_size"):
        features = g.selected_features.iloc[0].split("|")
        a = dataset.X_test[features].to_numpy()
        b = transformed[features].to_numpy()
        projections.append(
            {
                "fs_size": int(k),
                "selected_features": len(features),
                "rows": len(a),
                "identical_selected_values": bool(np.array_equal(a, b)),
                "different_selected_cells": int((a != b).sum()),
                "scope": "PRIMARY SELECTED PROJECTION ONLY; does not waive raw41-column family alignment rule",
            }
        )
    table(args.output / "primary_selected_projection_equivalence.csv", projections)
    write_json(
        args.output / "alignment_manifest.json",
        {
            "status": "STRICT_ALIGNMENT_PASSED"
            if exact
            else "BLOCKED_STRICT_ALIGNMENT_FAILED",
            "accepted_test_sha256": sha256(config["datasets"]["nsl_kdd"]["test_file"]),
            "candidate1_sha256": sha256(args.canonical1),
            "candidate2_sha256": sha256(args.canonical2),
            "binary_labels_equal_in_order": bool(labels_equal),
            "canonical_attack_names_transferred": False,
            "primary_projection_unchanged": all(
                r["identical_selected_values"] for r in projections
            ),
            "interpretation": "A raw service mismatch is surfaced, never silently corrected. Primary prediction inputs may be unaffected, but strict raw alignment remains required for family labels.",
        },
    )
    print(
        json.dumps(
            json.loads((args.output / "alignment_manifest.json").read_text()), indent=2
        )
    )


if __name__ == "__main__":
    main()
