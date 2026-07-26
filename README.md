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
│   └── make_figures.py
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
   - Expected format:
     - machine-learning CSV flow files
   - Purpose:
     - modern validation dataset
     - binary benign vs attack evaluation
     - optional attack-category analysis if labels are clean enough

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

After the implementation modules are completed, the main NSL-KDD FW-LNSA experiment should run with:

```bash
python scripts/run_nsl_kdd_fw_lnsa.py --config configs/nsl_kdd_fw_lnsa.yaml
```

The CICIDS2017 experiment should run with:

```bash
python scripts/run_cicids2017_fw_lnsa.py --config configs/cicids2017_fw_lnsa.yaml
```

Baseline models should run with:

```bash
python scripts/run_baselines.py --config configs/baseline_models.yaml
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

Required output files:

```text
results/tables/cicids2017_fw_lnsa_results.csv
results/tables/cicids2017_baseline_results.csv
results/tables/cicids2017_fw_lnsa_vs_baselines.csv
results/figures/cicids2017_method_comparison.png
results/figures/cicids2017_fw_lnsa_vs_baselines.png
```

CICIDS2017 preprocessing must document handling of:

- whitespace in column names
- NaN values
- `inf`, `-inf`, and `Infinity`
- duplicate or invalid rows
- BENIGN vs attack label conversion
- possible class imbalance
- train-only scaling

## Baseline Models

Required baselines:

| Baseline | Role |
|---|---|
| Logistic Regression | Simple interpretable supervised baseline |
| Decision Tree | Fast interpretable supervised baseline |
| Random Forest | Strong tabular supervised baseline |
| Isolation Forest | Unsupervised anomaly-detection baseline |
| XGBoost / LightGBM | Optional stronger supervised baseline |

The paper should not claim that FW-LNSA must outperform every supervised model. The expected claim is that FW-LNSA provides a lightweight, interpretable, detector-based AIS framework with measurable tradeoffs in recall, FPR, runtime, and detector retention.

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

- **Sarosh Jawed** — Researcher / first author; implementation, experiments, repository, and manuscript drafting.
- **Tanazzah** — Second author; planned support for experiments, baseline comparisons, validation, or manuscript support.
- **Dr. Suraiya Akter** — Advisor / third author; research supervision and methodological guidance.

Authorship and task ownership should be documented through commits, experiment logs, result files, and writing contributions.

## License

This repository is released under the MIT License unless changed by the authors before submission.

Dataset files are not covered by this repository license. Follow each dataset provider's terms and citation requirements.
