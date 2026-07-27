# FW-LNSA-NIDS

## Overview

FW-LNSA-NIDS is the implementation repository for **FW-LNSA**, a feature-weighted lightweight Negative Selection Algorithm framework for anomaly-based network intrusion detection.

The project follows Artificial Immune System logic: normal traffic is treated as **self**, anomalous or attack traffic is treated as **non-self**, candidate detectors that match self are rejected, and retained detectors are used to identify anomalous records.

The main experimental direction is to evaluate whether feature-importance weighting can improve detector matching behavior while keeping the method lightweight, reproducible, and interpretable.

## Research Contribution

This repository implements FW-LNSA, a feature-weighted lightweight negative selection framework for anomaly-based network intrusion detection. The framework integrates feature-importance weights directly into detector matching and evaluates Hamming, Weighted Hamming, and Weighted SMC matching under controlled detector budgets, thresholds, seeds, runtime, retained-detector, and attack-category analyses.

The paper contribution should be framed carefully:

- **Main method:** FW-LNSA, Feature-Weighted Lightweight Negative Selection Algorithm.
- **Main operational matching variant:** Weighted SMC inside FW-LNSA.
- **Ablation / comparison:** Weighted Hamming.
- **Classical comparison:** Standard Hamming.
- **Optional diagnostic:** Jaccard similarity.
- **Secondary extension:** TP-FW-LNSA as a two-phase pilot, not the main title unless strengthened later.

This repository does **not** claim that Weighted Hamming or Weighted SMC is a newly invented distance/similarity measure. The contribution is the controlled feature-weighted NSA framework, reproducible evaluation protocol, and analysis of matching behavior, false positives, detector retention, runtime, and attack-category performance.

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
│   └── 03_cicids2017_exploration.ipynb
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
│   └── utils.py
├── configs/
│   ├── nsl_kdd_fw_lnsa.yaml
│   ├── cicids2017_fw_lnsa.yaml
│   └── baseline_models.yaml
├── scripts/
│   ├── run_nsl_kdd_fw_lnsa.py
│   ├── run_cicids2017_fw_lnsa.py
│   ├── run_baselines.py
│   ├── make_figures.py
│   ├── validate_core_modules.py
│   ├── validate_fw_lnsa_pipeline.py
│   └── validate_cicids2017_pipeline.py
├── tests/
│   ├── test_core_modules.py
│   ├── test_fw_lnsa_pipeline.py
│   ├── test_cicids2017_pipeline.py
│   └── test_baseline_pipeline.py
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
     - attack-category analysis for Normal, DoS, Probe, R2L, and U2R

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
| Feature selection | Mutual information |
| Representation | Binary median split |
| Detector budgets | 500, 1000, 2500, 5000 |
| Seeds | 42, 43, 44, 45, 46 |
| Matching methods | Hamming, Weighted Hamming, Weighted SMC |
| Diagnostic only | Jaccard |
| Metrics | Accuracy, precision, recall, F1, FPR, runtime, retained detectors |

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
- stratified train/test splitting
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

Planned summary table:

| Dataset | Method | Feature Set | Recall | F1 | FPR | Runtime | Retained Detectors |
|---|---|---|---:|---:|---:|---:|---:|
| NSL-KDD | Weighted SMC FW-LNSA | FS-20 | TBD | TBD | TBD | TBD | TBD |
| NSL-KDD | Weighted Hamming | FS-20 | TBD | TBD | TBD | TBD | TBD |
| NSL-KDD | Hamming | FS-20 | TBD | TBD | TBD | TBD | TBD |
| CICIDS2017 | Weighted SMC FW-LNSA | FS-20 | TBD | TBD | TBD | TBD | TBD |

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
