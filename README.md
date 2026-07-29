# FW-LNSA-NIDS

## Overview

FW-LNSA-NIDS is the implementation repository for **FW-LNSA**, a feature-weighted lightweight Negative Selection Algorithm framework for anomaly-based network intrusion detection.

The project follows Artificial Immune System logic: normal traffic is treated as **self**, anomalous or attack traffic is treated as **non-self**, candidate detectors that match self are rejected, and retained detectors are used to identify anomalous records.

The main experimental direction is to evaluate whether feature-importance weighting can improve detector matching behavior while keeping the method lightweight, reproducible, and interpretable.

## Research Contribution

This repository implements FW-LNSA, a feature-weighted lightweight negative selection framework for anomaly-based network intrusion detection. The confirmatory protocol integrates training-derived feature weights into binary detector similarity, calibrates operating thresholds on validation data, and evaluates locked procedures on untouched test partitions.

The final research framing is:

- **Main method:** FW-LNSA, Feature-Weighted Lightweight Negative Selection Algorithm.
- **Canonical weighted formulation:** feature-weighted binary similarity.
- **Classical comparison:** unweighted Hamming NSA.
- **Equivalence verification:** Weighted Hamming, because normalized weighted similarity equals one minus normalized Weighted Hamming distance.
- **Secondary extension:** TP-FW-LNSA remains a pilot or future direction.

The repository does not claim that Weighted Hamming or weighted binary similarity is a newly invented metric. The contribution is the feature-weighted NSA framework, validation-calibrated low-FPR protocol, detector-efficiency analysis, and reproducible cross-dataset study of matching, representation size, detector budget, false positives, stability, and attack-category behavior.

## Repository Structure

```text
FW-LNSA-NIDS/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_nsl_kdd_exploration.ipynb
│   ├── 02_fw_lnsa_nsl_kdd_reproducibility.ipynb
│   ├── 03_cicids2017_exploration.ipynb
│   ├── 04_nsl_kdd_research_execution.ipynb
│   ├── 05_cicids2017_research_execution.ipynb
│   ├── 06_baseline_and_ablation_research_execution.ipynb
│   └── 07_confirmatory_research_execution.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── representation.py
│   ├── detector_generation.py
│   ├── matching.py
│   ├── fw_lnsa.py
│   ├── two_phase.py
│   ├── baselines.py
│   ├── evaluation.py
│   ├── experiments.py
│   ├── research_execution.py
│   ├── statistical_analysis.py
│   ├── confirmatory_protocol.py
│   └── utils.py
├── configs/
│   ├── nsl_kdd_fw_lnsa.yaml
│   ├── cicids2017_fw_lnsa.yaml
│   ├── baseline_models.yaml
│   ├── research_execution.yaml
│   └── confirmatory_protocol.yaml
├── scripts/
│   ├── run_nsl_kdd_fw_lnsa.py
│   ├── run_cicids2017_fw_lnsa.py
│   ├── run_baselines.py
│   ├── make_figures.py
│   ├── validate_core_modules.py
│   ├── validate_fw_lnsa_pipeline.py
│   ├── validate_cicids2017_pipeline.py
│   ├── validate_research_execution.py
│   ├── run_research_suite.py
│   ├── summarize_research_results.py
│   ├── make_research_figures.py
│   ├── run_confirmatory_protocol.py
│   └── validate_confirmatory_protocol.py
├── tests/
│   ├── test_core_modules.py
│   ├── test_fw_lnsa_pipeline.py
│   ├── test_cicids2017_pipeline.py
│   ├── test_baseline_pipeline.py
│   ├── test_research_execution.py
│   └── test_confirmatory_protocol.py
├── results/
│   ├── tables/
│   └── figures/
├── docs/
│   ├── Simulation_Report_1.docx
│   ├── Simulation_Report_2.docx
│   └── methodology_notes/
└── manuscript/
    ├── overleaf_source/
    └── figures/
```

## Datasets

Raw datasets are **not included** in this repository.

Required datasets before manuscript writing:

1. **NSL-KDD**
   - Expected files:
     - `KDDTrain+.txt`
     - `KDDTest+.txt`
   - Purpose:
     - controlled feasibility benchmark
     - FS-10 / FS-20 comparison
     - family-level attack analysis when the supplied test file preserves detailed attack names
   - Data-integrity note:
     - some redistributed `KDDTest+.txt` files contain only `Normal` and `Attack` labels
     - the pipeline reports `Attack (Unspecified)` for those files rather than inventing DoS, Probe, R2L, or U2R labels

2. **CICIDS2017**
   - Expected input:
     - the official `MachineLearningCSV.zip` archive, or the extracted machine-learning CSV files
   - Purpose:
     - modern validation dataset
     - binary benign vs attack evaluation
     - attack-family analysis using the preserved original labels
   - Reproducibility controls:
     - chunked archive streaming
     - exact-duplicate removal before splitting
     - deterministic per-label sampling
     - stratified train/test splitting
     - train-only imputation and scaling
     - saved data-quality report

Place local dataset files under:

```text
data/raw/nsl_kdd/
data/raw/cicids2017/
```

Those raw directories should remain ignored by Git. Use `data/README.md` to document download sources, expected filenames, and preprocessing decisions.

## Installation

Create a clean Python environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Reproduce Results

Validate the reusable modules and the integrated FW-LNSA pipeline first:

```bash
python scripts/validate_core_modules.py
python scripts/validate_fw_lnsa_pipeline.py
python scripts/validate_cicids2017_pipeline.py
python scripts/validate_baseline_pipeline.py
python scripts/validate_research_execution.py
```

The default NSL-KDD command uses the smoke profile so the complete pipeline can be checked safely before longer research runs:

```bash
python scripts/run_nsl_kdd_fw_lnsa.py --config configs/nsl_kdd_fw_lnsa.yaml
```

Use the research profile for a controlled reduced grid:

```bash
python scripts/run_nsl_kdd_fw_lnsa.py --config configs/nsl_kdd_fw_lnsa.yaml --profile research
```

Use the full profile only in Colab or another suitable compute environment after the smoke and research profiles pass:

```bash
python scripts/run_nsl_kdd_fw_lnsa.py --config configs/nsl_kdd_fw_lnsa.yaml --profile full
```

The default CICIDS2017 command uses a bounded smoke profile that streams directly from the official archive:

```bash
python scripts/run_cicids2017_fw_lnsa.py --config configs/cicids2017_fw_lnsa.yaml --profile smoke
```

Use the research profile in Colab after validation passes:

```bash
python scripts/run_cicids2017_fw_lnsa.py --config configs/cicids2017_fw_lnsa.yaml --profile research
```

Use the full profile only after reviewing the research profile outputs:

```bash
python scripts/run_cicids2017_fw_lnsa.py --config configs/cicids2017_fw_lnsa.yaml --profile full
```

Baseline smoke tests for both datasets run with:

```bash
python scripts/run_baselines.py --config configs/baseline_models.yaml --profile smoke --dataset all
```

Run one dataset when a shorter verification is useful:

```bash
python scripts/run_baselines.py --config configs/baseline_models.yaml --profile smoke --dataset nsl_kdd
python scripts/run_baselines.py --config configs/baseline_models.yaml --profile smoke --dataset cicids2017
```

Use the research and full profiles only after the matching FW-LNSA profile has
created compatible result tables:

```bash
python scripts/run_baselines.py --config configs/baseline_models.yaml --profile research --dataset all
python scripts/run_baselines.py --config configs/baseline_models.yaml --profile full --dataset all
```

Publication figures should be regenerated from saved CSV files with:

```bash
python scripts/make_figures.py
```

### Resumable Colab research execution

The final research profiles should be run through the persistent execution
layer. Each completed run is appended immediately to a durable JSONL progress
journal, and the saved manifest records
the configuration hash, dataset hash, Python environment, package versions, and
Git commit. A resumed job is rejected when its data or configuration no longer
matches the saved state.

Review the exact configured workload before starting Colab:

```bash
python scripts/run_research_suite.py --profile research --plan-only
```

The current research profile contains 280 NSL-KDD FW-LNSA runs, 240
CICIDS2017 FW-LNSA runs, and 40 baseline runs per dataset. The analysis stage
then regenerates all repeated-run and ablation tables from the saved CSV files.
Detection thresholds with the same detector-generation settings reuse one
detector pool and one score pass, reducing the two FW-LNSA grids to 120 unique
detector fits each. Prepared datasets are also cached for baseline reuse. When
multiple stages are selected, the command-line runner starts each stage in a
fresh Python process so dataset memory is released before the next stage begins.

Run the final NSL-KDD research profile:

```bash
python scripts/run_research_suite.py \
  --profile research \
  --stages nsl_kdd_fw_lnsa \
  --output-dir /path/to/persistent/results \
  --nsl-train-file /path/to/KDDTrain+.txt \
  --nsl-test-file /path/to/KDDTest+.txt
```

Run the final CICIDS2017 research profile:

```bash
python scripts/run_research_suite.py \
  --profile research \
  --stages cicids2017_fw_lnsa \
  --output-dir /path/to/persistent/results \
  --cic-archive-file /path/to/MachineLearningCSV.zip \
  --cic-raw-dir /path/to/cicids2017
```

After both FW-LNSA runs finish, run all required baselines and the final
ablation package:

```bash
python scripts/run_research_suite.py \
  --profile research \
  --stages nsl_kdd_baselines,cicids2017_baselines,analysis \
  --output-dir /path/to/persistent/results \
  --nsl-train-file /path/to/KDDTrain+.txt \
  --nsl-test-file /path/to/KDDTest+.txt \
  --cic-archive-file /path/to/MachineLearningCSV.zip \
  --cic-raw-dir /path/to/cicids2017
```

The three Colab notebooks in `notebooks/04` through `notebooks/06` provide the
same commands with Google Drive persistence and private GitHub access through
Colab Secrets.


### Validation-calibrated confirmatory protocol

The exploratory research grid is preserved for sensitivity analysis, but final
paper claims must use the three-way confirmatory protocol. The protocol fits
preprocessing, feature selection, binary medians, and detector pools on the
training partition. It calibrates target-FPR operating thresholds and selects
configurations on validation data. Only then does it evaluate the locked
procedure on the untouched test partition.

Validate the implementation:

```bash
python scripts/validate_confirmatory_protocol.py
```

Review the exact workload without loading data:

```bash
python scripts/run_confirmatory_protocol.py \
  --config configs/confirmatory_protocol.yaml \
  --profile confirmatory \
  --plan-only
```

Run a local or Colab smoke verification first:

```bash
python scripts/run_confirmatory_protocol.py \
  --config configs/confirmatory_protocol.yaml \
  --profile smoke \
  --dataset all \
  --stage all
```

Run final validation tuning into a new persistent output directory:

```bash
python scripts/run_confirmatory_protocol.py \
  --config configs/confirmatory_protocol.yaml \
  --profile confirmatory \
  --dataset all \
  --stage tune \
  --output-dir results_confirmatory
```

Inspect both `locked_configurations.csv` files before final evaluation. Then run:

```bash
python scripts/run_confirmatory_protocol.py \
  --config configs/confirmatory_protocol.yaml \
  --profile confirmatory \
  --dataset all \
  --stage confirm \
  --output-dir results_confirmatory
```

The final profile uses:

- FS-10 and FS-20
- Detector budgets 500, 1000, 2500, and 5000
- Tuning seeds 42 through 46
- Unseen confirmatory seeds 100 through 119
- Target FPR procedures 0.10, 0.05, and 0.01
- Hamming and FW-LNSA weighted similarity

Each dataset saves row-level progress journals, dataset fingerprints, partition
hashes, feature-selection hashes, a validation-selected configuration lock,
seed-level final results, aggregated 95% confidence intervals, category tables,
and an execution manifest. The test metrics are explicitly excluded from
configuration selection.

The Colab workflow is available in:

```text
notebooks/07_confirmatory_research_execution.ipynb
```

The exact methodological contract is documented in:

```text
docs/methodology_notes/confirmatory_protocol.md
```

## Main Experiments

### NSL-KDD FW-LNSA

Required output files:

```text
results/tables/nsl_kdd_fw_lnsa_results.csv
results/tables/nsl_kdd_matching_method_summary.csv
results/tables/nsl_kdd_best_balanced_configs.csv
results/tables/nsl_kdd_attack_category_analysis.csv
results/figures/nsl_kdd_method_comparison.png
results/figures/nsl_kdd_attack_category_recall.png
results/figures/nsl_kdd_runtime_detector_budget.png
```

Primary settings:

| Component | Decision |
|---|---|
| Feature sets | FS-10, FS-20; FS-All optional |
| Main feature set | FS-20 |
| Feature selection | Mutual information with binary indicators treated as discrete |
| Representation | Binary median split |
| Detector budgets | 500, 1000, 2500, 5000 |
| Tuning seeds | 42, 43, 44, 45, 46 |
| Confirmatory seeds | 100 through 119 |
| Final methods | Hamming NSA, FW-LNSA weighted similarity |
| Equivalence verification | Weighted Hamming |
| Diagnostic only | Jaccard in exploratory analyses |
| Metrics | Accuracy, precision, recall, F1, FPR, specificity, balanced accuracy, MCC, PR-AUC, ROC-AUC, runtime, throughput, model size, memory, retained detectors |

### CICIDS2017 FW-LNSA

FW-LNSA output files:

```text
results/tables/cicids2017_fw_lnsa_results.csv
results/tables/cicids2017_matching_method_summary.csv
results/tables/cicids2017_best_balanced_configs.csv
results/tables/cicids2017_attack_category_analysis.csv
results/tables/cicids2017_selected_features.csv
results/tables/cicids2017_data_quality_report.csv
```

The data pipeline documents and validates:

- whitespace and duplicate-column cleanup
- missing labels and all-missing feature rows
- `inf`, `-inf`, `Infinity`, and invalid numeric values
- exact duplicate rows before train/test splitting
- `BENIGN` versus attack conversion
- preserved original attack labels and stable attack families
- deterministic class-aware sampling
- stratified train/test splitting with rare labels pooled only for split safety
- training-only median imputation and scaling
- source-file and aggregate data-quality counts

Baseline runs reuse the same dataset profile, train/test partition, feature-selection
seed, and feature-selection sample limit as the matching FW-LNSA run. Every result
stores SHA-256 partition signatures so incompatible runs are not silently combined.

## Baseline Models

Implemented baselines:

| Baseline | Training rule | Role |
|---|---|---|
| Logistic Regression | Full labeled training partition | Interpretable linear supervised baseline |
| Decision Tree | Full labeled training partition | Interpretable nonlinear supervised baseline |
| Random Forest | Full labeled training partition | Strong tabular supervised baseline |
| Isolation Forest | Normal training traffic only | Unsupervised anomaly-detection baseline |

All baselines use continuous scaled FS-10 or FS-20 features selected by mutual
information on training data only. The comparison protocol records accuracy,
precision, recall, F1, FPR, FNR, fit time, prediction time, peak memory increase,
process memory, serialized model size, model parameters, seeds, and partition
signatures.

Generated tables:

```text
results/tables/nsl_kdd_baseline_results.csv
results/tables/nsl_kdd_fw_lnsa_vs_baselines.csv
results/tables/nsl_kdd_baseline_attack_category_analysis.csv
results/tables/nsl_kdd_baseline_selected_features.csv
results/tables/cicids2017_baseline_results.csv
results/tables/cicids2017_fw_lnsa_vs_baselines.csv
results/tables/cicids2017_baseline_attack_category_analysis.csv
results/tables/cicids2017_baseline_selected_features.csv
```

A comparison table includes FW-LNSA rows only when the profile, partition hashes,
record counts, and feature setting are compatible. This prevents smoke, research,
and full-profile results from being mixed accidentally.

XGBoost and LightGBM remain optional future additions. They are not required for
the core paper comparison and should not delay final experiments.

The paper should not claim that FW-LNSA must outperform every supervised model.
The expected contribution is a lightweight, interpretable, detector-based AIS
framework with measurable tradeoffs in recall, FPR, runtime, and detector retention.

## Results Summary

This section should be updated only after final scripts regenerate the tables.

The research analysis layer creates these publication-support tables:

```text
results/tables/research_fw_lnsa_summary.csv
results/tables/research_baseline_summary.csv
results/tables/research_best_fpr_controlled.csv
results/tables/research_hamming_weight_ablation.csv
results/tables/research_matching_formulation_ablation.csv
results/tables/research_feature_set_ablation.csv
results/tables/research_detector_budget_sensitivity.csv
results/tables/research_self_threshold_sensitivity.csv
results/tables/research_detection_threshold_sensitivity.csv
results/tables/research_seed_stability.csv
results/tables/research_cross_dataset_summary.csv
results/tables/research_readiness_report.csv
```

Repeated-run summaries report mean, sample standard deviation, and 95 percent
confidence intervals. Balanced configurations are selected from aggregated
seed results, rather than from one favorable seed. Mutual-information scores
are computed once per prepared partition and reused to create nested FS-10 and
FS-20 selections.

Final confirmatory summary template:

| Dataset | Method | Feature Set | Target FPR | Recall | F1 | Observed FPR | Throughput | Retained Detectors |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NSL-KDD | FW-LNSA weighted similarity | FS-20 | TBD | TBD | TBD | TBD | TBD | TBD |
| NSL-KDD | Hamming NSA | FS-20 | TBD | TBD | TBD | TBD | TBD | TBD |
| CICIDS2017 | FW-LNSA weighted similarity | FS-20 | TBD | TBD | TBD | TBD | TBD | TBD |
| CICIDS2017 | Hamming NSA | FS-20 | TBD | TBD | TBD | TBD | TBD | TBD |

Weighted Hamming is reported only as a mathematical and software-equivalence check. It is not treated as a separate contribution or independently tuned final method.

## Citation

If you use this repository, cite the future manuscript when available.

Temporary citation placeholder:

```bibtex
@misc{jawed_fw_lnsa_nids_2026,
  title  = {FW-LNSA-NIDS: Feature-Weighted Lightweight Negative Selection Algorithm for Network Intrusion Detection},
  author = {Jawed, Sarosh and [Second Author] and Akter, Suraiya},
  year   = {2026},
  note   = {GitHub repository}
}
```

## Authors

- **Sarosh Jawed** - Researcher / first author; implementation, experiments, repository, and manuscript drafting.
- **Tanazzah** - Second author; planned support for experiments, baseline comparisons, validation, or manuscript support.
- **Dr. Suraiya Akter** - Advisor / third author; research supervision and methodological guidance.

Authorship and task ownership should be documented through commits, experiment logs, result files, and writing contributions.

## License

This repository is released under the MIT License unless changed by the authors before submission.

Dataset files are not covered by this repository license. Follow each dataset provider's terms and citation requirements.
