"""Reuse original preparation and selection, add observational sidecars."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from ..confirmatory_protocol import (
    load_confirmatory_config,
    prepare_confirmatory_dataset,
    prepare_confirmatory_feature_sets,
    _partition_hashes,
)
from ..preprocessing import (
    read_nsl_kdd_file,
    NSL_KDD_FEATURE_COLUMNS,
    map_cicids2017_attack_category,
)
from .diagnostics import binary_diagnostics
from .io import sha256, table, write_json


def configured_original(data_root):
    config = load_confirmatory_config(
        "configs/confirmatory_protocol.yaml", profile_name="confirmatory"
    )
    root = Path(data_root).resolve()
    config["datasets"]["nsl_kdd"]["train_file"] = str(root / "nsl_kdd/KDDTrain+.txt")
    config["datasets"]["nsl_kdd"]["test_file"] = str(root / "nsl_kdd/KDDTest+.txt")
    config["datasets"]["cicids2017"]["raw_dir"] = str(root / "cicids2017")
    config["datasets"]["cicids2017"]["archive_file"] = str(
        root / "cicids2017/MachineLearningCSV.zip"
    )
    return config


def raw_preflight(config, accepted, dataset):
    manifest = json.loads(
        (
            Path(accepted)
            / f"reproducibility/{dataset}/fw_lnsa_execution_manifest.json"
        ).read_text()
    )
    original = {Path(k).name: v for k, v in manifest["dataset_file_sha256"].items()}
    paths = [
        config["datasets"][dataset][key]
        for key in (
            ["train_file", "test_file"] if dataset == "nsl_kdd" else ["archive_file"]
        )
    ]
    results = []
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Raw dataset absent: {path}")
        actual = sha256(path)
        if actual != original[Path(path).name]:
            raise ValueError(
                f"Raw hash mismatch: {Path(path).name}; historical reconstruction refused"
            )
        results.append(
            {
                "file": Path(path).name,
                "expected_sha256": original[Path(path).name],
                "actual_sha256": actual,
                "passed": True,
            }
        )
    return results


def data_audit(dataset_key, config, accepted, output):
    output = Path(output)
    table(output / "raw_hash_checks.csv", raw_preflight(config, accepted, dataset_key))
    dataset = prepare_confirmatory_dataset(dataset_key, config)
    features = prepare_confirmatory_feature_sets(dataset, config)
    original = json.loads(
        (
            Path(accepted)
            / f"reproducibility/{dataset_key}/fw_lnsa_execution_manifest.json"
        ).read_text()
    )
    checks = []
    for name, value in _partition_hashes(dataset).items():
        checks.append(
            {
                "type": "partition",
                "key": name,
                "expected": original["partition_hashes"][name],
                "actual": value,
                "passed": value == original["partition_hashes"][name],
            }
        )
    for k, feature in features.items():
        expected = original["feature_selection_hashes"][f"FS-{k}"]
        checks.append(
            {
                "type": "features",
                "key": f"FS-{k}",
                "expected": expected,
                "actual": feature.feature_selection_hash,
                "passed": expected == feature.feature_selection_hash,
            }
        )
        feature.score_table.to_csv(output / f"fs{k}_features_weights.csv", index=False)
        representation = feature.representation
        partitions = {
            "train": (representation.X_train_bin, dataset.y_train),
            "validation": (representation.X_validation_bin, dataset.y_validation),
            "test": (representation.X_test_bin, dataset.y_test),
        }
        summary, bits, overlap = binary_diagnostics(
            partitions, feature.selected_features
        )
        table(output / f"fs{k}_patterns.csv", summary)
        table(output / f"fs{k}_bits.csv", bits)
        table(output / f"fs{k}_pattern_overlap.csv", overlap)
    table(output / "reconstruction_checks.csv", checks)
    if not all(c["passed"] for c in checks):
        raise ValueError(
            "Reconstruction differs from original partitions/features; no locked replay permitted"
        )
    composition = []
    overlaps = []
    sets = {}
    for name in ["train", "validation", "test"]:
        X = getattr(dataset, "X_" + name)
        y = getattr(dataset, "y_" + name)
        labels = getattr(dataset, name + "_original_labels")
        categories = getattr(dataset, name + "_attack_categories")
        hashes = pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)
        sets[name] = set(hashes)
        # Hashed observational rows, no predictors exported.
        rows = pd.DataFrame(
            {
                "position": np.arange(len(X)),
                "predictor_hash_uint64": hashes,
                "binary_label": y,
                "original_label": labels,
                "category": categories,
            }
        )
        if dataset.row_provenance:
            provenance = dataset.row_provenance[name].rename(
                columns={
                    "__source_file__": "source_file",
                    "__source_row__": "source_row_zero_based_excluding_header",
                    "__priority__": "priority_uint64",
                }
            )
            rows = pd.concat(
                [
                    rows,
                    provenance[
                        [
                            "source_file",
                            "source_row_zero_based_excluding_header",
                            "priority_uint64",
                        ]
                    ],
                ],
                axis=1,
            )
        else:
            rows["source_file"] = "KDDTest+.txt" if name == "test" else "KDDTrain+.txt"
        table(output / f"{name}_row_signatures.csv", rows)
        for fields in [
            ["binary_label"],
            ["original_label", "category"],
            ["source_file", "binary_label"],
        ]:
            counts = (
                rows.groupby(fields, dropna=False).size().reset_index(name="records")
            )
            for row in counts.to_dict("records"):
                composition.append(
                    {"partition": name, "grouping": "|".join(fields), **row}
                )
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlaps.append(
            {
                "partition_a": a,
                "partition_b": b,
                "shared_predictor_hashes": len(sets[a] & sets[b]),
                "hash_algorithm": "pandas uint64; collision limitation; diagnostic only",
            }
        )
    table(output / "composition.csv", composition)
    table(output / "predictor_overlap.csv", overlaps)
    write_json(output / "dataset_metadata.json", dataset.metadata)
    if dataset.data_quality_report is not None:
        table(output / "cleaning_report.csv", dataset.data_quality_report)
    if dataset_key == "nsl_kdd":
        schemas = []
        for split in ["train", "test"]:
            raw = read_nsl_kdd_file(config["datasets"][dataset_key][split + "_file"])
            path = Path(config["datasets"][dataset_key][split + "_file"])
            first = path.open().readline().strip()
            schemas.append(
                {
                    "split": split,
                    "columns": len(first.split(","))
                    if "," in first
                    else len(first.split("\t")),
                    "rows": len(raw),
                    "unique_labels": "|".join(sorted(raw.label.astype(str).unique())),
                    "sha256": sha256(path),
                    "difficulty_excluded": True,
                    "family_resolution": dataset.metadata.get(
                        "test_attack_category_resolution"
                    )
                    if split == "test"
                    else "named_training_attacks",
                }
            )
        table(output / "nsl_schema_provenance.csv", schemas)
        # Optional canonical test file must pass every aligned feature and label.
        candidate = config["datasets"][dataset_key].get("canonical_test_file")
        status = {
            "status": "BLOCKED",
            "reason": "Supplied test has binary labels; independent canonical attack-name file absent. No family inference.",
        }
        if candidate:
            canonical = read_nsl_kdd_file(candidate)
            given = read_nsl_kdd_file(config["datasets"][dataset_key]["test_file"])
            if len(canonical) != len(given) or not canonical[
                NSL_KDD_FEATURE_COLUMNS
            ].equals(given[NSL_KDD_FEATURE_COLUMNS]):
                raise ValueError("Canonical test row/predictor alignment failed")
            from ..preprocessing import to_binary_label

            if not np.array_equal(
                canonical.label.map(to_binary_label), given.label.map(to_binary_label)
            ):
                raise ValueError("Canonical binary labels disagree")
            if set(canonical.label.astype(str)) <= {"normal", "anomaly", "0", "1"}:
                raise ValueError("Canonical file does not contain attack names")
            status = {
                "status": "ALIGNED",
                "canonical_sha256": sha256(candidate),
                "feature_columns_checked": 41,
                "rows": len(canonical),
                "reason": "Labels may be used only for separately labeled family analysis",
            }
        write_json(output / "canonical_family_alignment.json", status)
    else:
        archive = Path(config["datasets"][dataset_key]["archive_file"])
        schema = []
        with zipfile.ZipFile(archive) as z:
            for member in sorted(n for n in z.namelist() if n.lower().endswith(".csv")):
                with z.open(member) as f:
                    columns = pd.read_csv(f, nrows=0).columns
                info = z.getinfo(member)
                for column in columns:
                    name = " ".join(column.strip().split())
                    schema.append(
                        {
                            "file": Path(member).name,
                            "archive_member": member,
                            "archive_crc32": info.CRC,
                            "archive_uncompressed_bytes": info.file_size,
                            "input_column": column,
                            "normalized_column": name,
                            "retained_as_predictor": name in dataset.X_train.columns,
                            "reason": "Label excluded"
                            if name.lower() == "label"
                            else "numeric predictor"
                            if name in dataset.X_train.columns
                            else "removed/all missing",
                            "destination_port_retained": name == "Destination Port"
                            and name in dataset.X_train.columns,
                        }
                    )
        table(output / "cicids_exact_schema.csv", schema)
        labels = sorted(
            set(dataset.train_original_labels)
            | set(dataset.validation_original_labels)
            | set(dataset.test_original_labels)
        )
        table(
            output / "category_map.csv",
            [
                {
                    "original_label": label,
                    "category": map_cicids2017_attack_category(label),
                    "train_support": int(
                        (dataset.train_original_labels == label).sum()
                    ),
                    "validation_support": int(
                        (dataset.validation_original_labels == label).sum()
                    ),
                    "test_support": int((dataset.test_original_labels == label).sum()),
                }
                for label in labels
            ],
        )
        write_json(
            output / "leakage_contract.json",
            {
                "destination_port_retained": "Destination Port"
                in dataset.X_train.columns,
                "explicit_identity_drop_list": ["Label"],
                "IPs_timestamps_flow_ID": "Absent from supplied MachineLearningCSV schema, not explicitly filtered by original general loader",
                "helper_fields_excluded": "all names starting __",
                "duplicates": "globally before split, numeric predictors + normalized original label, pandas uint64 hash; not near-duplicate or temporal deduplication",
                "sample": "smallest 60000 priority keys = pandas row hash XOR seed constant; deterministic hash-priority, not pandas sample or first 60000. 30000/7500 quotas INACTIVE under global_cap. Seed 42.",
                "split": "row-wise original-label stratified with rare pooling, sample then test 30% seed42 then validation20% of development seed2026",
                "source_rows": "fixed observational offsets track pre-filter positions, zero-based excluding header",
                "independence": "Same capture/day may occur in all partitions; no independent deployment validation",
            },
        )
    return dataset, features
