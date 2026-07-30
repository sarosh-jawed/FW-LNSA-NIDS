# Manuscript Readiness Protocol

## Purpose

This protocol closes the gap between completed FW-LNSA confirmation and paper
writing. It prevents incompatible exploratory baselines, uncorrected multiple
tests, and undocumented result selection from entering the manuscript.

## Confirmatory-compatible baselines

The baseline runner recreates the original train, validation, and test
partitions from the completed confirmatory configuration. Before model fitting,
it verifies:

- source dataset SHA-256 fingerprints
- train, validation, and test partition hashes
- the exact saved FS-10 and FS-20 feature lists
- feature-selection hashes
- the original confirmatory manifest status

The saved feature lists are consumed directly rather than recomputing mutual
information. This avoids silent feature-space drift across library versions.

Logistic Regression, Decision Tree, Random Forest, and Isolation Forest are
trained on the training partition. For supervised models, attack probability is
used as the operating score. For Isolation Forest, the anomaly score is oriented
so larger values indicate stronger attack evidence. A threshold is calibrated
from benign validation scores for each target FPR. Test labels and test metrics
are never used for threshold or model selection.

## Statistical analysis

Hamming and weighted similarity are paired by dataset, feature set, target FPR,
and unseen confirmatory seed. The final analysis reports:

- paired mean and median differences
- bootstrap 95 percent confidence intervals
- Wilcoxon signed-rank tests
- rank-biserial effect sizes
- Holm-adjusted p-values
- weighted-method win, tie, and loss counts

FPR, runtime, model size, and memory are interpreted as lower-is-better metrics.
F1, recall, balanced accuracy, MCC, retention, and throughput are interpreted as
higher-is-better metrics.

## Final audit

The readiness gate checks row counts, seed coverage, duplicate identities,
missing metrics, lock integrity, tuning and confirmatory seed separation,
dataset and partition compatibility, feature-selection compatibility, baseline
completion, target-FPR coverage, and the prohibition on test-based selection.
The writer package cannot be built when an error-level audit check fails.

## Writer handoff

The handoff contains only verified evidence needed for manuscript writing:

- final summary tables
- seed-level audit data in a separate folder
- LaTeX table exports
- publication figures
- methods and novelty boundaries
- supported and prohibited claims
- limitations and threats to validity
- manifests, configuration locks, and environment details
- the executed final notebook when available

Raw datasets are excluded because they are large, licensed separately, and not
needed by the writer.
