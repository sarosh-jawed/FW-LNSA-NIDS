# Data Directory

Raw datasets are not committed to this repository.

## Expected local layout

```text
data/
├── raw/
│   ├── nsl_kdd/
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   └── cicids2017/
│       ├── MachineLearningCSV.zip
│       └── <optional extracted CSV files>
├── interim/
└── processed/
```

## NSL-KDD

Expected files:

- `KDDTrain+.txt`
- `KDDTest+.txt`

Implemented processing controls:

- assign 41 NSL-KDD feature names plus `label` and `difficulty`
- convert `normal` to 0 and all attacks to 1
- preserve original labels for attack-category analysis
- map detailed labels into Normal, DoS, Probe, R2L, and U2R
- report `Attack (Unspecified)` when a redistributed test file contains binary-only attack labels
- split `KDDTrain+` into detector-training and validation partitions
- keep `KDDTest+` untouched until the locked confirmatory evaluation
- one-hot encode `protocol_type`, `service`, and `flag` using training vocabulary only
- fit imputation and scaling only on detector-training data

## CICIDS2017

Expected input:

- official `MachineLearningCSV.zip`, or its extracted CSV flow files

Implemented processing controls:

- stream CSV members from the ZIP archive in bounded chunks
- strip whitespace and disambiguate duplicate column names
- detect the label column case-insensitively
- replace nonfinite values and coerce invalid numeric values to missing
- remove rows with missing labels or no usable feature values
- remove exact duplicate feature-label rows before splitting
- convert `BENIGN` to 0 and all attacks to 1
- preserve original labels and map them to stable attack families
- use deterministic per-label sampling for smoke validation
- use deterministic bounded natural-distribution sampling for research runs
- create a reproducible development/test split, pooling only labels too rare to stratify safely
- split development data into detector-training and validation partitions
- fit imputation medians and scaling only on detector-training data
- reserve the untouched test partition for locked confirmatory evaluation
- save a source-level and aggregate data-quality report

## Dataset rule

Do not upload large raw datasets to GitHub unless the dataset license clearly allows redistribution. Prefer scripts/configuration plus clear download instructions.

## Google Drive layout for Colab

The research notebooks expect:

```text
MyDrive/FW-LNSA-NIDS/
├── data/
│   ├── nsl_kdd/
│   │   ├── KDDTrain+.txt
│   │   └── KDDTest+.txt
│   └── cicids2017/
│       └── MachineLearningCSV.zip
├── results_research/
└── results_confirmatory/
```

Keep exploratory and confirmatory outputs in separate directories. The
confirmatory directory stores tuning journals, validation-selected locks, final
seed results, manifests, tables, and packaged outputs so work survives Colab
runtime resets.
