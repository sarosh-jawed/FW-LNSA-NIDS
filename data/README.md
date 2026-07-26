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

Processing notes to document later:

- assign 41 NSL-KDD feature names plus `label` and `difficulty`
- convert `normal` to 0 and all attacks to 1
- preserve original labels for attack-category analysis
- map attacks into Normal, DoS, Probe, R2L, and U2R
- one-hot encode `protocol_type`, `service`, and `flag`
- fit scaling only on training data

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
- use deterministic per-label sampling to control memory and class imbalance
- create a reproducible stratified train/test split
- fit imputation medians and scaling only on training data
- save a source-level and aggregate data-quality report

## Dataset rule

Do not upload large raw datasets to GitHub unless the dataset license clearly allows redistribution. Prefer scripts/configuration plus clear download instructions.
