# Reproduce the verified FW-LNSA research release

## Environment

Use Python 3.12. The recorded Colab run used Python 3.12.13; release checks use Python 3.12.14 with the same scientific package versions. `requirements-final-validation-lock.txt` is the executed scientific lock. `requirements-release-lock.txt` additionally pins the complete environment used for publication regeneration. This added lock does not change the executed scientific versions.

```bash
python3.12 -m venv .venv-final-validation
.venv-final-validation/bin/python -m pip install -r requirements-release-lock.txt
.venv-final-validation/bin/python -m pip check
.venv-final-validation/bin/python -m pytest -q
.venv-final-validation/bin/python scripts/check_release_cleanliness.py
```

Windows PowerShell uses `.\.venv-final-validation\Scripts\python.exe` instead of the Unix executable above. Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to `1` before numerical jobs.

## Verify archives and regenerate publication artifacts

Keep the reviewed Colab export and the original `writer_handoff.zip` outside the Git working tree. Extract the latter once to `ACCEPTED_ROOT`; its root contains `PACKAGE_MANIFEST.json`, `evidence/` and `reproducibility/`. The immutable fingerprint seal verifies all 59 accepted files. The reviewed Colab archive has SHA-256 `4ddf1e02e621d0d7e0c046d1aadea62f5dd295bbbf1d2f348adc24e60ea722ec`.

```bash
python scripts/reconcile_release_evidence.py --export COLAB_EXPORT.zip --accepted ACCEPTED_ROOT --output PRIVATE_REVIEW_DIR
python scripts/build_publication_artifacts.py --export COLAB_EXPORT.zip --accepted ACCEPTED_ROOT --output NEW_EMPTY_ARTIFACT_DIR
```

Use a new empty output directory; the builder refuses to overwrite existing artifacts. No fitting or raw-data loading occurs. The archive must pass every payload hash, nested manifest and full-profile completion gate. Reconciliation independently regenerates original statistics, recalculates Holm over 120 tests, checks all 480 replay confusion/threshold rows, recomputes secondary intervals and overlap counts, and reconstructs all benchmark summary quantiles from 80 repeat rows.

`PUBLICATION_MANIFEST.json` hashes every generated publication file. `TABLE_PROVENANCE.json` identifies exact archive members and their hashes. `FIGURE_INVENTORY.csv` identifies plot row keys, numerical columns, captions and manuscript placement. PDF creation timestamps are removed for deterministic regeneration. Regeneration is byte-checked in the release environment; other font/rendering environments may require comparison of numerical data and figure geometry rather than identical PNG bytes.

## Repeat empirical validation only when scientifically necessary

The full user Colab run is complete. It does not need another run for release integration. The following commands document full empirical reproduction, not an instruction to repeat it now.

Raw expected names below `DATA_ROOT` are `nsl_kdd/KDDTrain+.txt`, `nsl_kdd/KDDTest+.txt` and `cicids2017/MachineLearningCSV.zip`. Raw hashes must match the preserved execution manifests; there is no substitute-file override.

```bash
python scripts/run_final_validation.py --profile smoke --accepted ACCEPTED_ROOT --output PRIVATE_SMOKE_DIR --jobs all --include-120k
python scripts/run_final_validation.py --profile full --accepted ACCEPTED_ROOT --data-root DATA_ROOT --output PRIVATE_NEW_FULL_DIR --jobs all --include-120k
python scripts/make_final_secondary_report.py --root PRIVATE_NEW_FULL_DIR --profile full
python scripts/validate_final_outputs.py --root PRIVATE_NEW_FULL_DIR --accepted ACCEPTED_ROOT --profile full
python scripts/run_final_validation.py --profile full --accepted ACCEPTED_ROOT --data-root DATA_ROOT --output PRIVATE_NEW_FULL_DIR --jobs report --export PRIVATE_EXPORT.zip
python scripts/verify_final_validation.py --export PRIVATE_EXPORT.zip --output PRIVATE_CHECKS.json
```

The CLI requires an explicit profile. The original `active_profile: smoke` is not the primary study. A full run includes 30 training resamples per dataset, 190 development rows, 480 frozen replays, 80 benchmark repeats, 120 secondary tuning rows and 40 secondary test rows. Private unit checkpoints resume only under an identical source/profile/environment identity. This release adds publication scripts, so any genuinely new empirical execution requires a new output directory; do not attach the old Colab checkpoint directory to a changed source tree.

Notebook 09 is byte-identical to the previously delivered notebook; its six source cells remain unchanged. Its setup expects the legacy filename `FW_LNSA_Final_Research_Code_GitHub_Ready.zip`. For a future clean Colab reproduction, make a copy of the new release ZIP with that legacy filename before uploading it, together with the original starter and data ZIPs. Both packages have the same application/overlay interface. The notebook uses the executed scientific lock. Its uploaded notebook-file hash was not captured by Colab's export, so exact identity of the user's uploaded notebook cannot independently be established; all 62 recorded scientific source/configuration hashes are verified.

## Interpretation and privacy

The original 120-test Holm family is unchanged. A 12-F1 family is only a separately labeled sensitivity in the private export. Secondary analyses never replace the primary locks. Seed intervals describe detector randomness conditional on shared fixed data, not deployment or record-sampling uncertainty.

In replay diagnostics, `threshold`, `validation_threshold` and `test_threshold` are the final calibrated cutoff. The raw replay fit-summary field `detection_threshold` retains its pre-calibration default. The public curation explicitly exports `calibrated_detection_threshold` from the final field and omits uncomputed diagnostic AUC columns; accepted primary AUC values remain in the original-metrics table.

Benchmark timing measures full-record binary scoring plus thresholding, batch size 250, one warmup and ten measured clean processes per case. It excludes preprocessing, model loading, fitting and I/O. Sampled timed-stage RSS, RSS increment and lifetime high-water RSS have different scopes. Detector-state pickle size excludes preprocessing; fitted deployment-state pickle size includes it but excludes library executables. Never infer end-to-end latency or real-time performance from these measurements.

Do not commit raw data, row signatures, sample provenance, pickles, caches or full exports. `.gitignore` and the release cleanliness check protect these classes. The private handoff contains audit records deliberately excluded from GitHub. Permanent archive creation, references, author declarations and journal submission remain owner/author actions.

Final static cleanup is explicit in `docs/release/SOURCE_CLEANUP_LEDGER.json`: 58 of the 62 executed files remain byte-identical. Four files have only an unused-import removal, duplicate-import removal or unused-local-binding rename. The configuration-key access is retained. Exact edit offsets and before/after hashes reconstruct the executed source bytes; algorithms and numerical contracts are unchanged. The unmodified executed overlay is retained in the private audit archive.

Native Windows publication review can run `python -m pytest -q tests/test_publication_release.py`, full Ruff, bytecode compilation and the cleanliness check. The complete suite imports the preserved Unix `resource` benchmark module and belongs on Linux/Colab/WSL. Native Windows full-suite compatibility is not claimed. See the owner PowerShell guide for the portable review route.
