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
│       └── <CICIDS2017 machine-learning CSV files>
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

- machine-learning CSV flow files

Processing notes to document later:

- strip whitespace from column names
- replace `Infinity`, `inf`, and `-inf`
- handle NaN values
- remove invalid rows and leakage-prone identifiers if present
- convert `BENIGN` to 0 and all attacks to 1
- preserve original attack labels where possible
- fit scaling only on training data

## Dataset rule

Do not upload large raw datasets to GitHub unless the dataset license clearly allows redistribution. Prefer scripts/configuration plus clear download instructions.
