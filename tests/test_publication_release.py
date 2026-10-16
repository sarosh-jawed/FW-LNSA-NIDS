"""Release-contract checks on curated evidence, independent of raw datasets."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from scripts.build_publication_artifacts import calibrated_replay, build
from scripts.check_release_cleanliness import inspect_text, inspect_files
from scripts.reconcile_release_evidence import source_lineage

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "results/publication"


def test_calibrated_replay_uses_final_cutoff():
    frame = pd.DataFrame(
        {"detection_threshold": [0.0], "threshold": [1.0], "test_threshold": [1.0]}
    )
    result = calibrated_replay(frame)
    assert result.calibrated_detection_threshold.tolist() == [1.0]
    assert "detection_threshold" not in result and "threshold" not in result


def test_calibrated_replay_rejects_inconsistent_final_cutoff():
    with pytest.raises(ValueError, match="thresholds disagree"):
        calibrated_replay(pd.DataFrame({"threshold": [1.0], "test_threshold": [0.0]}))


def test_cleanliness_detects_encoded_punctuation_and_private_state(tmp_path):
    assert inspect_text(json.dumps({"text": chr(0x2014)}), "document.json")
    (tmp_path / "checkpoint.pkl").write_bytes(b"state")
    assert inspect_files(tmp_path, ["checkpoint.pkl"])["status"] == "FAIL"


def test_cleanliness_detects_personal_path_without_rejecting_colab_defaults():
    assert inspect_text("C:" + "\\Users\\example\\project", "note.md")
    assert inspect_text("/workspace/" + "scratch/example", "note.md")
    assert not inspect_text(
        "/content/drive/MyDrive/FW-LNSA-final-validation", "note.md"
    )


def test_builder_preserves_existing_output(tmp_path):
    (tmp_path / "keep.txt").write_text("accepted")
    with pytest.raises(ValueError, match="new or empty"):
        build("absent.zip", "absent", tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "accepted"


def test_every_executed_scientific_source_still_matches():
    identity = json.loads((PUBLIC / "EXECUTED_SOURCE_IDENTITY.json").read_text())
    assert len(identity["source_hashes"]) == 62
    rows = source_lineage(ROOT, identity["source_hashes"])
    assert all(r["matches"] for r in rows)
    assert sum(r["byte_identical"] for r in rows) == 58
    assert sum(r["documented_static_cleanup"] for r in rows) == 4


def test_curated_original_seed_metrics_and_holm_family():
    for file, rows, model in [
        ("original_fw_seed_metrics", 480, "method"),
        ("original_reference_seed_metrics", 960, "model"),
    ]:
        df = pd.read_csv(
            PUBLIC / "tables" / (file + ".csv"), float_precision="round_trip"
        )
        assert len(df) == rows
        assert not df.duplicated(
            ["dataset_key", model, "fs_size", "target_fpr", "seed"]
        ).any()
        for _, g in df.groupby(["dataset_key", model, "fs_size", "target_fpr"]):
            assert set(g.seed) == set(range(100, 120))
        for part in ["test", "validation"]:
            tp, fp, fn, tn = [df[part + "_" + c] for c in ["tp", "fp", "fn", "tn"]]
            assert np.allclose(
                2 * tp / (2 * tp + fp + fn), df[part + "_f1"], rtol=0, atol=1e-12
            )
            assert np.allclose(fp / (fp + tn), df[part + "_fpr"], rtol=0, atol=1e-12)
    paired = pd.read_csv(
        PUBLIC / "tables/original_paired_tests_120.csv", float_precision="round_trip"
    )
    assert len(paired) == 120
    order = np.argsort(paired.p_value.to_numpy(), kind="stable")
    expected = np.minimum(
        1,
        np.maximum.accumulate(paired.p_value.to_numpy()[order] * np.arange(120, 0, -1)),
    )
    assert np.allclose(
        expected, paired.holm_adjusted_p.to_numpy()[order], rtol=0, atol=1e-12
    )


def test_public_inventory_has_no_unmanifested_files():
    manifest = json.loads((PUBLIC / "PUBLICATION_MANIFEST.json").read_text())
    actual = {
        p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*") if p.is_file()
    }
    assert actual == set(manifest["files"]) | {"PUBLICATION_MANIFEST.json"}
    for rel, expected in manifest["files"].items():
        assert hashlib.sha256((PUBLIC / rel).read_bytes()).hexdigest() == expected
    assert inspect_files(PUBLIC, sorted(actual))["status"] == "PASS"
    from PIL import Image

    for path in (PUBLIC / "figures").glob("*.png"):
        with Image.open(path) as image:
            image.load()
            assert min(image.info["dpi"]) >= 299


def test_primary_weights_and_missing_category_contracts():
    weights = pd.read_csv(
        PUBLIC / "tables/features_weights.csv", float_precision="round_trip"
    )
    for (_, k), g in weights.groupby(["dataset_key", "fs_size"]):
        assert len(g) == k and g.feature.nunique() == k
        assert set(g["rank"]) == set(range(1, k + 1))
        assert np.isclose(g.normalized_weight.sum(), 1, rtol=0, atol=1e-12)
    categories = pd.read_csv(PUBLIC / "tables/category_recall.csv")
    absent = categories[categories.category.isin(["Heartbleed", "Infiltration"])]
    assert (absent.support == 0).all()
    assert (absent.status == "NOT_ESTIMABLE").all()
    assert absent["mean"].isna().all()


def test_adverse_secondary_result_and_overlap_are_preserved():
    development = pd.read_csv(PUBLIC / "tables/development_seed_metrics.csv")
    adverse = development[
        (development.dataset_key == "cicids2017")
        & (development.experiment == "mrmr_20")
        & (development.method == "weighted_similarity")
    ]
    assert len(adverse) == 5
    assert (
        (adverse[["validation_f1", "validation_recall", "validation_fpr"]] == 0)
        .all()
        .all()
    )
    paired = (
        pd.read_csv(PUBLIC / "tables/sample120k_paired.csv")
        .set_index("metric")
        .loc["test_f1"]
    )
    assert paired.paired_percentile_95ci_low < 0 < paired.paired_percentile_95ci_high
    overlap = pd.read_csv(PUBLIC / "tables/sample120k_overlap.csv")
    assert overlap.shared_rows.sum() == 60000
    assert (
        overlap[
            (overlap.original_60k_partition == "test")
            & (overlap.new_120k_partition == "train")
        ].shared_rows.item()
        == 6810
    )
    methods = json.loads((PUBLIC / "METHODS_METADATA.json").read_text())
    assert {r["job"] for r in methods["blocked"]} == {
        "nsl_kdd_attack_family",
        "scenario_holdout",
    }


def test_source_cleanup_ledger_rejects_unlisted_edit(tmp_path):
    import shutil

    rows = json.loads((ROOT / "docs/release/SOURCE_CLEANUP_LEDGER.json").read_text())
    row = rows[0]
    (tmp_path / "docs/release").mkdir(parents=True)
    (tmp_path / "docs/release/SOURCE_CLEANUP_LEDGER.json").write_text(json.dumps([row]))
    target = tmp_path / row["path"]
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / row["path"], target)
    assert source_lineage(tmp_path, {row["path"]: row["executed_sha256"]})[0]["matches"]
    target.write_text(target.read_text() + "\n# unexpected additional edit\n")
    assert not source_lineage(tmp_path, {row["path"]: row["executed_sha256"]})[0][
        "matches"
    ]
