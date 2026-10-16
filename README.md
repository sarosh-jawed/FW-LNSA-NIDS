# FW-LNSA-NIDS

FW-LNSA is a label-informed, feature-weighted negative selection framework for network intrusion detection. It combines training-only mutual-information ranking and weights, median binary encoding, benign self-space detector generation, and validation-only false-positive-rate calibration.

The preserved confirmatory study compares separately validation-selected Hamming and weighted configurations on NSL-KDD and CICIDS2017. The comparison does not isolate the weighting formula. NSL-KDD improvements are conditional; CICIDS2017 FS10 utility is near zero, no CICIDS2017 F1 comparison survives the original 120-test Holm correction, and classical trees substantially outperform FW-LNSA on CICIDS2017. Compact serialized detector state does not establish low latency, low process memory or real-time deployment readiness.

## Verified research artifacts

- [Publication tables, figures and source manifests](results/publication/)
- [Methods and scientific limitations](docs/release/METHODS_AND_LIMITATIONS.md)
- [Release reconciliation and readiness](docs/release/RELEASE_READINESS_GATE.json)
- [All 42 supervisor-comment dispositions](docs/release/SUPERVISOR_COMMENT_DISPOSITION.csv)
- [Exact reproduction instructions](REPRODUCE.md)

The full Colab execution uses `PROFILE="full"` and `INCLUDE_120K=True`. Its SHA-256 is `4ddf1e02e621d0d7e0c046d1aadea62f5dd295bbbf1d2f348adc24e60ea722ec`. All 2,269 archived payload hashes, 12 required real-data jobs and 88 scientific QA checks pass. Release reconciliation checks every one of the 62 executed source/configuration hashes and preserves the original evidence.

The handoff is verified for manuscript preparation. This is not journal submission approval. The two blocked extensions, NSL named attack-family recovery and comparable scenario/day-held-out validation, remain blocked. Permanent archive creation and owner-controlled GitHub publication are separate actions.

## Reproduce publication artifacts without retraining

Use Python 3.12 and `requirements-release-lock.txt`. Supply the reviewed Colab export ZIP and the original extracted `writer_handoff` directory, both outside the repository:

```bash
python scripts/reconcile_release_evidence.py --export COLAB_EXPORT.zip --accepted ACCEPTED_ROOT --output PRIVATE_REVIEW_DIR
python scripts/build_publication_artifacts.py --export COLAB_EXPORT.zip --accepted ACCEPTED_ROOT --output NEW_EMPTY_ARTIFACT_DIR
python -m pytest -q
python scripts/check_release_cleanliness.py
```

The publication builder reads existing evidence only. It does not fit models, choose thresholds or load raw datasets. `TABLE_PROVENANCE.json` maps curated tables to archived input hashes; `FIGURE_INVENTORY.csv` supplies captions, placement and CSV row/column selectors. All ten final figures are vector PDFs with 600-dpi PNG companions. Use them at double-column width.

## Repository organization

| Directory | Purpose |
|---|---|
| `src/` | Original research implementation and modular final validation |
| `configs/` | Explicit experiment profiles and locked protocol definitions |
| `scripts/` | Experiment CLIs, archive reconciliation and publication regeneration |
| `tests/` | Model behavior, integrity, checkpoint, serialization and release contracts |
| `notebooks/09_final_research_validation.ipynb` | Complete Colab workflow used for final validation |
| `results/publication/` | Curated aggregate and detector-seed evidence, vector artwork and hashes |
| `docs/release/` | Current source audit, methods, limitations and comment dispositions |
| `docs/final_revision/accepted_file_hashes.json` | Immutable accepted-input seal used by executed validation code |

Notebooks 01-08 remain historical provenance. They are not alternative final evidence sources. The final manuscript can be prepared entirely from the current handoff and curated publication directory.

## Scientific boundaries

Destination Port was retained in CICIDS2017. Its random flow split permits same-capture dependence. The 120k secondary cohort contains all original 60k records and repartitions them; it is sample-size/repartitioning sensitivity, not independent external validation. Its paired F1 interval crosses zero and the weighted method has a substantially larger observed FPR. Weighted mRMR produced no detections in the tested development configuration.

The NSL binary test differs from candidate named-label mirrors in 306 service values. Both selected primary prediction projections are identical, but strict raw-feature alignment fails; family labels must not be transferred. All uncertainty across seeds is conditional on fixed data and calibration partitions.

## Data and licensing

Software is distributed under the existing MIT license. No raw third-party dataset, fitted checkpoint, network-record provenance table, private execution archive or personal path is included in the public release. Curated seed rows represent model runs, not individual network records. Dataset access and use remain subject to the original providers' terms. The public manifests identify required input bytes without redistributing them.

The pinned accepted commit and dereferenced `paper-evidence-v1.0` tag are `8496b94717c85566748c767c387f999d76bdc9a7`. Original results and tag remain unchanged. This release integration does not push, merge, retag or assign a DOI.

Final static cleanup is explicit in `docs/release/SOURCE_CLEANUP_LEDGER.json`: 58 of the 62 executed files remain byte-identical. Four files have only an unused-import removal, duplicate-import removal or unused-local-binding rename. The configuration-key access is retained. Exact edit offsets and before/after hashes reconstruct the executed source bytes; algorithms and numerical contracts are unchanged. The unmodified executed overlay is retained in the private audit archive.
