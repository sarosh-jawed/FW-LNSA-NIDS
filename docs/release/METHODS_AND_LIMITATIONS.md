# Verified methods, corrections and claim boundaries

Status: VERIFIED_FOR_MANUSCRIPT_PREPARATION. The accepted evidence is unchanged. Original confirmation and newly executed robustness analyses have distinct estimands. This document supplies factual writing notes; it does not edit the manuscript or grant submission approval.

## Data and representation

NSL-KDD uses 125,973 supplied development records: 100,778 training and 25,195 validation, stratified by original labels with seed 2026. The separate supplied binary test has 22,544 records and 42 columns. Label and difficulty fields are excluded from predictors. Training-fitted one-hot columns and MinMax scaling are applied unchanged to validation/test; test extrapolation is not clipped.

The accepted test is a binary-label mirror. Two matching named-label candidate mirrors differ from it in 306 service-field entries, `Attack4` versus `imap4`; the other 40 predictors and binary labels align in row order. The primary FS10 and FS20 transformed prediction matrices are identical with zero differing cells. The strict 41-predictor family-transfer gate nevertheless fails. Do not infer or impute test attack-family labels or silently correct the accepted source.

CICIDS2017 uses the supplied eight MachineLearningCSV files, with 78 encoded predictors. Destination Port is retained. Only the label and internal provenance helpers are excluded. IP, timestamp and flow-ID fields are absent in these inputs; a dedicated exclusion filter did not remove them. The exact included columns and per-file cleaning counts are in `results/publication/tables/cicids_input_columns.csv` and `cicids_cleaning.csv`.

Cleaning coerces numeric values, replaces nonfinite cells by missing values, drops missing-label/all-predictor-missing rows, and deduplicates numeric predictors plus normalized original label using uint64 row hashes. Hash-collision risk is not mathematically zero; this does not remove near-duplicate or correlated flows. Training medians impute missing cells, and scaling is trained only on training data.

After cleaning and before splitting, the smallest 60,000 deterministic row-hash priorities are selected with sampling seed 42. The configured 30,000-benign and 7,500-per-attack caps are inactive under `global_cap`. The sample has 49,661 normal and 10,339 attack records. Test is 30% with seed 42; validation is 20% of development with seed 2026, original-label stratification and rare pooling. Training is 33,600 (27,810 normal/5,790 attack); validation 8,400 (6,953/1,447); test 18,000 (14,898/3,102). Exact category, original-label and source-file composition is supplied. Random flow splitting permits same-capture/day dependence; exact predictor-hash overlap checks do not establish scenario independence.

Primary MI uses `mutual_info_classif`, random state 42, one computational job and default three-neighbor continuous estimation. An explicit discrete mask marks a column discrete when all its values are close to 0 or 1. This includes one-hot indicators and other binary-like columns; dense default continuous treatment is not used. NSL primary MI uses a stratified training-only subsample up to 50,000; CICIDS primary MI uses its 33,600 training records. Primary ranking uses descending pandas MI sort without a lexical tie rule. Selected nonnegative MI scores are normalized to sum one, with a uniform fallback if all selected scores are zero. The complete method is label-informed.

For selected feature j, binary encoding is

\[
b_j(x)=\mathbf{1}\{x_j>m_j\},\qquad m_j=\operatorname{median}_{x\in T}x_j,
\qquad w_j=\frac{\max(\widehat I_j,0)}{\sum_\ell\max(\widehat I_\ell,0)}.
\]

Equality to the median maps to zero. Medians use all training rows and remain fixed on validation/test.

## Detector algorithm and calibration

For k bits, Hamming distance and weighted bit-equality similarity are

\[
D_H(x,d)=\sum_{j=1}^{k}\mathbf{1}\{x_j\ne d_j\},\qquad
S_W(x,d)=\sum_{j=1}^{k}w_j\mathbf{1}\{x_j=d_j\}.
\]

Only normal training rows define self. When needed, `default_rng(detector_seed)` samples 10,000 self-row indices without replacement; unique binary self patterns are obtained after sampling. Candidate generation separately initializes a generator with the same seed, not an independently spawned RNG stream. It draws one fixed candidate batch; candidate duplicates remain, and there is no replenish-to-mature-count loop.

Retain a Hamming candidate iff every self distance exceeds `ceil(ratio*k)`. Retain a weighted candidate iff every self similarity is strictly less than the configured tolerance. At prediction, native Hamming score is the minimum distance to a mature detector; weighted score is maximum similarity. Attack decisions are inclusive: Hamming distance <= cutoff, weighted similarity >= cutoff. Empty-pool and degenerate/tied-score behavior follow the existing implementation and tests.

For each fitted pool, benign validation scores select the most permissive attainable inclusive cutoff respecting `floor(target*number_of_benign_validation_records)`. Tied scores are never randomly split; `nextafter` excludes an infeasible boundary when needed. Validation attacks inform configuration ranking, not the benign FPR cutoff. Each confirmatory seed is recalibrated on the same fixed validation partition before test scoring. These are operational FPR thresholds, not calibrated probabilities; ECE and Brier score are not analyses in this study.

The original development grid uses budgets 500/1,000/2,500/5,000; Hamming tolerance ratios 0.05/0.10/0.15 and weighted similarities 0.85/0.80/0.75; k10/20; seeds 42-46; and targets 0.01/0.05/0.10. This yields 240 pool fits and 720 target rows per dataset. Selection orders mean validation F1, recall, lower FPR, lower measured generation time and lower serialized size. The timing tie break is hardware-sensitive. Full rejected-configuration historical journals were unavailable; saved original locks remain authoritative. Do not invent missing historical grid timings or claim a fresh grid must select the same lock.

Logistic Regression, Decision Tree, Random Forest and Isolation Forest are untuned reference models with the preserved hyperparameters. Isolation Forest uses benign training only and negative `decision_function`; external validation calibration supersedes its internal contamination offset. Common splits do not imply equal optimization effort.

## Statistical methodology and correction

The original Holm family is **120 tests: 12 dataset/k/target conditions times ten metrics**. The 12 F1 contrasts are a subset. All original adjusted values are preserved, including five significant NSL F1 contrasts and no significant CICIDS F1 contrast. A 12-F1-only sensitivity is not a replacement family.

Mean intervals are 95% Student-t across 20 detector seeds, with 19 degrees of freedom. Paired mean-difference intervals use 5,000 paired-seed percentile-bootstrap replicates and seed `2026 + sequential output row index`. Wilcoxon is two-sided with `zero_method="wilcox"` and `method="auto"`; a historical `numpy.allclose` zero shortcut returns p=1. Near-zero tolerance in that shortcut/rank-effect handling differs from the 1e-12 win/tie/loss rule; exact historical behavior is preserved. Neither t intervals nor paired intervals quantify across-domain or record-sampling uncertainty.

Calibration tables contain target-met counts/20, exact binomial descriptive seed-compliance intervals, absolute target error and threshold ranges. Seeds reuse common validation data, so those intervals do not establish independent operational reliability. Conditional seed variation must not be described as deployment uncertainty.

## Secondary analyses

Thirty stratified 80% training-only resamples use seeds 3000-3029. Original training-fitted preprocessing is fixed; MI is refitted in each resample with lexical exact-score ties. NSL resample MI can use more rows than the original 50k cap. These are conditional feature-stability diagnostics, not a full preprocessing-refit study. Pairwise Jaccard values share resamples; their 435 pairs per k are dependent.

Feature-count k5/10/15/20/30 and self-cap 2.5k/5k/10k diagnostics use development seeds 42-46, target 0.05 and frozen original FS20 budget/tolerance. They do not reselect primary test configurations. No-port diagnostics remove Destination Port and refit training-only MI. mRMR-MID maximizes relevance minus mean selected-feature redundancy, using empirical discrete MI in nats for both terms, training quantile bins up to ten and lexical ties. It changes selection/estimator/representation jointly. Reference: Peng, Long and Ding, IEEE TPAMI 27(8), 1226-1238 (2005), DOI 10.1109/TPAMI.2005.159. Weighted mRMR produced no CICIDS detections at the tested calibrated point; this adverse result remains.

The 120k study uses a separately fingerprinted nested hash-priority cohort, FS20, target 0.05, tuning seeds 200-204 and evaluation seeds 300-319. Its validation tie breaks are deterministic and exclude runtime. Both selected budgets are 5,000; Hamming self distance is 3 and weighted tolerance 0.75. Its 20-seed mean F1 values are 0.468781 and 0.526021. Paired difference is 0.057240, 95% percentile interval [-0.003229, 0.115110], with 13 wins/0 ties/7 losses. Mean observed FPR is 0.005483 versus 0.041481. No new inferential p family is reported.

All original 60k records occur in 120k. Repartitioning places 6,810 original test rows in secondary training and 1,823 in secondary validation. Primary models/locks/test outcomes remain unchanged, but the secondary study is not independent confirmation. Do not claim matched-observed-FPR superiority, external validation or an isolated weighting-formula effect.

CICIDS category recall is reconstructed with frozen primary predictions. Heartbleed and Infiltration have zero test support and are `NOT_ESTIMABLE`. The supplied original-label map includes Web Attack-Brute Force in Brute Force. Rare-label pooling used for split stratification does not make absent test attacks observed.

## Efficiency, metadata and limits

Controlled benchmarking uses 80 measured processes, eight FS20 cases (two datasets/methods/targets 0.01/0.05), seed 100, batch size 250, one warmup per process and ten measurements per case. CPU is AMD EPYC 7B12, eight visible CPUs, one computational thread; Python 3.12.13, Linux 6.6.122+, CPU-only algorithm. Colab tier and exclusive machine access are not established. Timing includes full-record binary scoring and threshold comparison, excluding preprocessing/loading/fitting/I/O. It does not use duplicate-pattern replay acceleration.

RSS columns ending `_mb` use MiB (2^20 bytes). Timed-stage absolute sampled peak RSS ranges from 175.332031 to 466.910156 MiB across these repeats. Small RSS increments reflect allocation reuse/sampling resolution, not zero total memory. Lifetime high-water RSS is uniformly 853.621094 MiB in this run; it reflects earlier process history and must not be substituted for the timed-stage measurement. No causal explanation of that uniformity or exact remeasurement of the historical 279 MB outlier is claimed.

Original model size is a pickle dictionary of detector state, weights, thresholds and vector length; it excludes fitted preprocessing. The new complete deployment pickle additionally contains the model and fitted encoder-column metadata, scaler, imputations and medians. Neither includes library executables. Report bytes/KiB and the serialization scope, not a universal deployment footprint.

The diagnostic replay CSV also contains a stale fit-summary `detection_threshold` initialized before calibration. The final `threshold`, `validation_threshold` and `test_threshold` match the accepted cutoff in every replay. Public curation uses `calibrated_detection_threshold` from the final field. Diagnostic AUC columns were not computed; they are omitted from public replay tables. Original accepted score-based AUC values remain available and unchanged.

NSL family recovery and comparable scenario/day-held-out evaluation remain blocked. Data dependence, unavailable historical tuning journals, untuned references, hardware-specific measurements and seed-conditional uncertainty remain limitations. Author metadata, funding/conflicts/CRediT, bibliography, current journal rules and permanent archive/DOI require author action. No manuscript or historical tag was changed and no publication was pushed.

The final source integration includes four documented static cleanups, with exact reversibility and original/release hashes in SOURCE_CLEANUP_LEDGER.json. These remove unused or duplicate imports and rename an unused binding while retaining its configuration access. The original 62-file source identity matched in full before cleanup; 58 remain byte-identical in the release. No scientific computation or selection rule changed.

For the algorithm box, distinguish the outer procedure selection from each frozen-pool run: select detector budget/tolerance with development seeds and validation objectives first; then for each confirmatory seed generate one training pool, calibrate its benign-validation cutoff, freeze state, and evaluate test. Report Hamming attack-oriented ranking score as the negative native minimum distance for AUC, and weighted score as maximum similarity. A native distance cutoff is not interchangeable with an attack-oriented ranking score.

Aggregate full-predictor overlap checks find five shared predictor hashes between NSL training/validation, 542 between training/test and 114 between validation/test. These are counts of distinct shared uint64 predictor hashes, not counts of labeled duplicate records. Supplied NSL partition membership is preserved, but predictor-value separation is not complete. CICIDS has zero shared full-predictor hashes in each within-study partition pair after its cleaning/splitting. Neither observation establishes independence of near-duplicate or same-capture records. Exact counts are in `partition_predictor_overlap.csv`; binary-pattern overlap is a separate representation diagnostic. The named-label source audit is preserved in `docs/release/nsl_provenance` and is separate from the Colab experiment.

The paired percentile intervals are unadjusted intervals for individual mean differences. They are not simultaneous family-wise intervals, so excluding zero in one such interval does not imply significance after the 120-test Holm correction. Display paired interval endpoints to six decimals to avoid negative-zero rounding of small endpoints.
