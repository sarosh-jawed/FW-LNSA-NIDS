# Validation-Calibrated Confirmatory Protocol

## Purpose

The confirmatory protocol prevents final-test leakage and turns the exploratory
FW-LNSA grid into a defensible final evaluation. It uses three partitions:

1. **Training:** preprocessing, feature selection, binary medians, and detector generation.
2. **Validation:** target-FPR calibration and configuration selection.
3. **Test:** one final evaluation of each locked procedure for each unseen seed.

The test partition is never used to choose a feature set, detector budget,
self threshold, detection threshold, or matching method.

## Canonical methods

The final protocol evaluates:

- **Hamming:** unweighted classical NSA comparison.
- **FW-LNSA weighted similarity:** feature-weighted binary similarity.

For normalized weights and binary vectors:

\[
S_w(x,d)=\sum_i w_i\mathbf{1}[x_i=d_i]
       =1-\sum_i w_i\mathbf{1}[x_i\ne d_i]
       =1-D_w(x,d).
\]

Weighted Hamming is therefore retained as an equivalence test, not as an
independent novelty claim. The historical `weighted_smc` name remains supported
for backward compatibility with exploratory outputs.

## Configuration selection

For every candidate detector pool, benign validation scores calibrate the least
restrictive threshold whose empirical false-positive rate does not exceed the
target. Validation attacks are then used to rank candidate detector budgets and
self thresholds. The ranking order is:

1. Mean validation F1
2. Mean validation recall
3. Lower validation FPR
4. Lower generation time
5. Smaller serialized model size

The final configuration is locked separately for each dataset, matching method,
feature-set size, and target FPR.

## Seed separation

- Tuning seeds: `42, 43, 44, 45, 46`
- Confirmatory seeds: `100` through `119`

The seed sets do not overlap. Final summaries report mean, sample standard
deviation, and 95% confidence intervals across the 20 unseen detector seeds.

## Operating points

The final protocol reports three validation-calibrated operating points:

- FPR at or below 0.10
- FPR at or below 0.05
- FPR at or below 0.01

These are procedures, not test-selected thresholds. Each confirmatory seed
calibrates its numeric threshold from its validation score distribution before
the test partition is scored.

## Efficiency evidence

Every final row records:

- Candidate and retained detector counts
- Detector retention rate
- Detector-array bytes
- Feature-weight bytes
- Serialized model bytes
- Detector-generation time
- Detection time
- Records processed per second
- Latency per 1,000 records
- Peak RSS increase during fitting and scoring

These measurements support a quantitative lightweight claim rather than a
qualitative label.

## Attack-category reporting

Category results are saved for every confirmatory seed and then aggregated.
Each category table includes test count, mean recall, standard deviation, 95%
confidence interval, detected count, missed count, and evidence status.
Categories with fewer than 20 test observations are marked as limited. Categories
with no test observations are not assigned a percentage.

Some redistributed NSL-KDD test files contain only `Normal` and `Attack` labels.
In that case attacks are reported as `Attack (Unspecified)`. The code does not
invent DoS, Probe, R2L, or U2R labels that are absent from the source file.

## Reproducibility controls

The protocol saves:

- Dataset SHA-256 fingerprints
- Training, validation, and test partition hashes
- Feature-selection hashes
- Configuration hash
- Validation-selected configuration lock
- Row-level JSONL progress journals
- Seed-level final results
- Seed-level and aggregated category results
- Execution manifest

A configuration or partition mismatch invalidates the saved lock and requires a
new output directory.
