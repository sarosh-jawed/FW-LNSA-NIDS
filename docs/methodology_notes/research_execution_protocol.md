# Final Research Execution Protocol

## Scope

The final evaluation uses FW-LNSA as the main framework. Weighted SMC is the main operational matching variant. Weighted Hamming is the feature-weighting ablation, and Hamming is the classical internal NSA comparison. Jaccard remains optional and is excluded from the required research profile.

## Datasets

- NSL-KDD uses the official training and testing files.
- CICIDS2017 uses the official machine-learning CSV archive.
- CICIDS2017 research runs use bounded natural-distribution sampling rather than per-label smoke quotas.
- Preprocessing, feature selection, imputation, scaling, and binary representation are fitted using training data only.
- Binary one-hot indicators are treated as discrete variables during mutual-information estimation, while continuous measurements remain continuous.
- Rare CICIDS2017 labels are pooled only for split stratification when necessary. Their original labels and attack families remain unchanged for reporting.

## Repeated-run design

The research profile uses:

- FS-10 and FS-20
- seeds 42, 43, 44, 45, and 46
- Hamming, Weighted Hamming, and Weighted SMC
- detector budgets 500 and 1000 for the first research pass
- controlled self and detection threshold grids
- Logistic Regression, Decision Tree, Random Forest, and Isolation Forest baselines

The larger full profile should run only after the research profile has been reviewed for memory use, runtime, class coverage, and result stability.

## Fair comparison rules

- Baselines inherit the matching dataset profile.
- Baselines use the same train and test partitions as FW-LNSA.
- Baselines use the same mutual-information feature selection and FS-10 or FS-20 features.
- Randomized methods use the same five seeds.
- Isolation Forest trains only on normal training records.
- Smoke rows are never compared with research or full rows.
- Partition hashes and feature-selection hashes must match before FW-LNSA and baseline rows are combined.

## Saved-run recovery

Each completed model run is appended immediately to a durable JSONL journal in the persistent output directory. An interrupted final journal line is ignored safely during resume. The execution manifest stores configuration hashes, dataset hashes, package versions, Python information, and the Git commit. A resumed run is rejected when its configuration or dataset fingerprint differs from the saved state.

Prepared train and test partitions are cached in the persistent output directory. NSL-KDD and CICIDS2017 baselines therefore reuse the exact prepared partitions produced for the matching FW-LNSA profile without rescanning or reprocessing the raw dataset. When several stages are requested in one command, each stage runs in a fresh Python process so large dataset objects are released before the next stage starts.

Detection thresholds that share a feature set, method, seed, detector budget, and self threshold reuse one fitted detector pool and one anomaly-score pass. Every threshold still receives its own result row and metrics. This removes repeated computation without changing the experimental grid. Mutual-information scores are also computed once per prepared partition and reused for nested FS-10 and FS-20 selections.

## Statistical reporting

Final paper tables should report mean, sample standard deviation, and 95 percent confidence intervals across seeds. Balanced configurations are selected from aggregated repeated runs, not from a single favorable seed.

Required analyses are:

1. Hamming versus Weighted Hamming
2. Weighted Hamming versus Weighted SMC
3. FS-10 versus FS-20
4. Detector-budget sensitivity
5. Self-threshold sensitivity
6. Detection-threshold sensitivity
7. Seed stability
8. NSL-KDD versus CICIDS2017 generalization

## Interpretation rule

The paper should not claim that the weighting formula or similarity measure is newly invented. The defensible contribution is the feature-weighted lightweight NSA framework, its direct integration of feature importance into detector matching, and its controlled analysis of accuracy, recall, false positives, retained detectors, runtime, attack-category behavior, and cross-dataset stability.
