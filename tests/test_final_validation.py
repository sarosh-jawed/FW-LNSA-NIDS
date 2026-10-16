from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.final_validation.io import Job, binary_matrix, table, write_json
from src.final_validation.diagnostics import (
    binary_diagnostics,
    category_recalls,
    feature_stability,
    mrmr_ranking,
    sample_training,
)
from src.final_validation.runner import synthetic_dataset
from src.final_validation.data import raw_preflight
from src.confirmatory_protocol import (
    calibrate_detection_threshold,
    predict_from_native_scores,
)
from src.detector_generation import negative_selection
from src.feature_selection import normalize_feature_weights
from src.fw_lnsa import FWLNSA
from src.manuscript_analysis import _safe_wilcoxon, holm_adjust
from src.preprocessing import _clean_cicids2017_chunk, _safe_stratification_labels
from src.utils import current_process


def test_checkpoint_success_resume(tmp_path):
    calls = []
    job = Job(tmp_path, "unit", {"seed": 42})

    def action(p):
        calls.append(True)
        table(p / "data.csv", [{"a": 1}])

    job.run(action, synthetic=True)
    job.run(action, synthetic=True)
    assert len(calls) == 1
    assert json.loads(job.manifest.read_text())["status"] == "VERIFIED_SYNTHETIC_ONLY"


def test_checkpoint_hash_mismatch(tmp_path):
    job = Job(tmp_path, "unit", {})
    job.run(lambda p: write_json(p / "data.json", {"x": 1}))
    (job.path / "data.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        job.valid()


def test_checkpoint_identity_mismatch(tmp_path):
    Job(tmp_path, "unit", {"seed": 42}).run(lambda p: write_json(p / "data.json", {}))
    with pytest.raises(ValueError, match="identity mismatch"):
        Job(tmp_path, "unit", {"seed": 43}).valid()


def test_partial_failure_recovery(tmp_path):
    job = Job(tmp_path, "unit", {})

    def fail(p):
        table(p / "partial.csv", [{"x": 1}])
        raise RuntimeError("disconnect")

    with pytest.raises(RuntimeError):
        job.run(fail)
    assert json.loads(job.manifest.read_text())["status"] == "FAILED"
    job.run(lambda p: table(p / "partial.csv", [{"x": 2}]))
    assert pd.read_csv(job.path / "partial.csv").x.iloc[0] == 2


@pytest.mark.parametrize("values", [[[0.5]], [[256]], [[np.nan]], [0, 1]])
def test_validate_before_cast(values):
    with pytest.raises(ValueError):
        binary_matrix(values)


def test_constant_duplicate_patterns():
    parts = {
        "train": (np.array([[0, 0], [0, 0], [1, 0]]), np.array([0, 1, 1])),
        "validation": (np.array([[1, 0], [1, 1]]), np.array([0, 1])),
        "test": (np.array([[0, 0], [1, 1]]), np.array([0, 1])),
    }
    summary, bits, overlap = binary_diagnostics(parts, ["a", "b"])
    assert summary.iloc[0].unique_patterns == 2
    assert summary.iloc[0].constant_bits == 1
    assert summary.iloc[0].benign_attack_shared_patterns == 1
    assert summary.iloc[2].records_unseen_in_train == 1
    assert len(bits) == 6 and len(overlap) == 3


def test_rare_category_missing_is_nan():
    row = category_recalls(
        [0, 1], [0, 1], ["benign", "DoS"], ["Normal", "DoS"], ["Absent"]
    )[0]
    assert row["status"] == "NOT_ESTIMABLE" and np.isnan(row["recall"])


def test_provenance_offsets_after_filter_and_duplicates():
    raw = pd.DataFrame(
        {
            "x": [1, 2, 2, np.nan, 4],
            "Label": ["BENIGN", None, "BENIGN", "BENIGN", "DoS Hulk"],
        }
    )
    cleaned, _ = _clean_cicids2017_chunk(
        raw,
        label_column="Label",
        benign_label="BENIGN",
        source_name="example.csv",
        row_offset=100,
        seen_hashes={
            int(
                pd.util.hash_pandas_object(
                    pd.DataFrame({"x": [2.0], "__label__": ["BENIGN"]}), index=False
                ).iloc[0]
            )
        },
        random_seed=42,
        drop_duplicates=True,
    )
    assert cleaned.__source_row__.tolist() == [100, 104]


def test_empty_detector_pool_and_serialization():
    X = np.zeros((10, 3), dtype=np.int8)
    y = np.zeros(10, dtype=np.int8)
    model = FWLNSA(method="hamming", n_detectors=8, self_threshold=3).fit(X, y)
    assert model.detector_pool_.retained_count == 0
    assert np.all(np.isinf(model.decision_scores(X)))
    assert np.all(model.predict(X) == 0)
    assert np.array_equal(
        pickle.loads(pickle.dumps(model)).predict(X), model.predict(X)
    )


def test_candidate_self_duplicates():
    pool = negative_selection(
        np.array([[1, 1], [1, 1], [0, 0]]),
        np.array([[0, 0], [0, 0]]),
        method="hamming",
        threshold=0,
    )
    assert pool.unique_self_patterns == 1 and pool.retained_count == 2


def test_inclusive_score_ties_and_floor():
    score = np.array([0.0, 0.5, 0.5, 1.0])
    y = np.zeros(4, dtype=np.int8)
    cal = calibrate_detection_threshold(
        score, y, method="weighted_similarity", target_fpr=0.5
    )
    assert cal.achieved_fpr == 0.25 and cal.threshold == 1
    assert (
        predict_from_native_scores([1], method="weighted_similarity", threshold=1)[0]
        == 1
    )


def test_all_tie_wilcoxon():
    assert _safe_wilcoxon(np.zeros(20)) == (0.0, 1.0)
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_degenerate_weights():
    assert np.allclose(normalize_feature_weights([0, 0]), [0.5, 0.5])


def test_training_resample_reproducible_and_rare_guard():
    data = synthetic_dataset()
    a, ya = sample_training(data.X_train, data.y_train, 3000)
    b, yb = sample_training(data.X_train, data.y_train, 3000)
    pd.testing.assert_frame_equal(a, b)
    assert np.array_equal(ya, yb) and len(a) == 128
    with pytest.raises(ValueError):
        sample_training(pd.DataFrame({"a": [1, 2]}), [0, 1], 1)


def test_feature_stability_reproducible():
    data = synthetic_dataset()
    a = feature_stability(data.X_train, data.y_train, seeds=[3000, 3001])
    b = feature_stability(data.X_train, data.y_train, seeds=[3000, 3001])
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x, y)
    assert a[1].selection_frequency.between(0, 1).all()
    assert a[2].jaccard.between(0, 1).all()


def test_mrmr_duplicate_and_lexical_tie():
    X = pd.DataFrame(
        {"z": [0, 0, 1, 1] * 10, "a": [0, 0, 1, 1] * 10, "b": [1, 0, 1, 0] * 10}
    )
    ranking, red = mrmr_ranking(X, [0, 0, 1, 1] * 10)
    assert ranking.feature.iloc[0] == "a"
    assert np.isclose(red.loc["z", "a"], np.log(2))
    assert (
        ranking.feature.iloc[1] == "b"
    )  # duplicate relevance minus redundancy = 0; lexical b wins


def test_absent_dataset(tmp_path):
    config = {
        "datasets": {
            "nsl_kdd": {
                "train_file": str(tmp_path / "absent"),
                "test_file": str(tmp_path / "absent2"),
            }
        }
    }
    path = tmp_path / "accepted/reproducibility/nsl_kdd/fw_lnsa_execution_manifest.json"
    write_json(
        path, {"dataset_file_sha256": {"absent": "missing", "absent2": "missing"}}
    )
    with pytest.raises(FileNotFoundError):
        raw_preflight(config, tmp_path / "accepted", "nsl_kdd")


def test_raw_hash_mismatch(tmp_path):
    raw = tmp_path / "file"
    raw.write_text("wrong")
    config = {"datasets": {"nsl_kdd": {"train_file": str(raw), "test_file": str(raw)}}}
    path = tmp_path / "accepted/reproducibility/nsl_kdd/fw_lnsa_execution_manifest.json"
    write_json(path, {"dataset_file_sha256": {"file": "0" * 64}})
    with pytest.raises(ValueError, match="hash mismatch"):
        raw_preflight(config, tmp_path / "accepted", "nsl_kdd")


def test_rare_stratification_labels():
    labels = pd.Series(["BENIGN"] * 10 + ["DoS"] * 5 + ["RareA", "RareB"])
    strat, _, _ = _safe_stratification_labels(
        labels, np.array([0] * 10 + [1] * 7), strategy="original_label"
    )
    assert len(strat) == 17


def test_current_process_matches_proc_self():
    process = current_process()
    if Path("/proc/self/stat").exists():
        assert process.pid == int(Path("/proc/self/stat").read_text().split(" ", 1)[0])
    assert process.memory_info().rss > 0


def test_complete_deployment_transform_and_pickle():
    from src.final_validation.deployment import DeploymentPipeline
    from src.confirmatory_protocol import (
        load_confirmatory_config,
        prepare_confirmatory_feature_sets,
    )

    data = synthetic_dataset()
    features = prepare_confirmatory_feature_sets(
        data,
        load_confirmatory_config(
            "configs/confirmatory_protocol.yaml", profile_name="confirmatory"
        ),
    )[20]
    model = FWLNSA(method="hamming", n_detectors=20, self_threshold=1).fit(
        features.representation.X_train_bin, data.y_train
    )
    pipeline = DeploymentPipeline(
        model,
        data.scaler,
        list(data.X_train.columns),
        features.selected_features,
        features.representation.medians,
        None,
        "nsl_kdd",
    )
    raw = pd.DataFrame(
        data.scaler.inverse_transform(data.X_test), columns=data.X_test.columns
    )
    assert np.array_equal(pipeline.transform(raw), features.representation.X_test_bin)
    restored = pickle.loads(pickle.dumps(pipeline, protocol=5))
    assert np.array_equal(restored.predict(raw), pipeline.predict(raw))


@pytest.mark.parametrize("method", ["hamming", "weighted_similarity"])
def test_duplicate_pattern_replay_exact_scores_and_thresholds(method):
    from src.final_validation.diagnostics import score_unique_patterns

    rng = np.random.default_rng(777)
    unique = rng.integers(0, 2, size=(310, 10), dtype=np.int8)
    X = unique[rng.integers(0, len(unique), size=2000)]
    y = np.tile([0, 1], 1000)
    weights = np.arange(1, 11, dtype=float)
    weights /= weights.sum()
    model = FWLNSA(
        method=method,
        n_detectors=60,
        self_threshold=0 if method == "hamming" else 0.99,
        random_seed=50,
    ).fit(X, y, feature_weights=weights if method == "weighted_similarity" else None)
    full = model.decision_scores(X)
    memo = score_unique_patterns(model, X)
    assert np.array_equal(full, memo)
    fullcal = calibrate_detection_threshold(full, y, method=method, target_fpr=0.05)
    memocal = calibrate_detection_threshold(memo, y, method=method, target_fpr=0.05)
    assert fullcal.threshold == memocal.threshold
    assert np.array_equal(
        predict_from_native_scores(full, method=method, threshold=fullcal.threshold),
        predict_from_native_scores(memo, method=method, threshold=memocal.threshold),
    )


def test_accepted_input_seal_rejects_changed_content_on_resume(tmp_path):
    from src.final_validation.evidence import verify_accepted_fingerprints
    from src.final_validation.io import sha256

    accepted = tmp_path / "accepted"
    accepted.mkdir()
    evidence = accepted / "seed.csv"
    evidence.write_text("seed,f1\n100,0.5\n")
    reference = tmp_path / "expected.json"
    write_json(reference, {"seed.csv": sha256(evidence)})
    assert verify_accepted_fingerprints(accepted, reference)[0]["passed"]
    evidence.write_text("seed,f1\n100,0.9\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_accepted_fingerprints(accepted, reference)


def test_accepted_input_seal_rejects_missing_file(tmp_path):
    from src.final_validation.evidence import verify_accepted_fingerprints

    reference = tmp_path / "expected.json"
    write_json(reference, {"missing.csv": "0" * 64})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_accepted_fingerprints(tmp_path, reference)
