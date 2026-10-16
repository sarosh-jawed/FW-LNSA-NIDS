#!/usr/bin/env python3
"""Prepare compact manuscript table data from the curated publication directory."""

from pathlib import Path
import argparse
import json
import pandas as pd


def build(publication, output):
    publication, output = Path(publication), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty table folder")
    output.mkdir(parents=True, exist_ok=True)

    def read(n):
        return pd.read_csv(
            publication / "tables" / (n + ".csv"), float_precision="round_trip"
        )

    entries = []
    pieces = [
        "# Verified manuscript table data\n\nValues below are display-rounded. Exact unrounded numbers and source hashes are in publication/tables and TABLE_PROVENANCE.json. All uncertainty is conditional on shared fixed data. These are table fragments for Tanazzah to incorporate; no manuscript was edited. Do not automatically include every preparation table in the paper. Use the section guide to keep the main text focused.\n"
    ]

    def table(name, title, frame, source, note):
        frame.to_csv(output / (name + ".csv"), index=False, lineterminator="\n")
        (output / (name + ".tex")).write_text(
            frame.to_latex(index=False, escape=True, float_format="%.6g")
        )
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            lines.append(
                "| "
                + " | ".join(
                    (f"{x:.6g}" if isinstance(x, float) else str(x)).replace("|", "/")
                    for x in row
                )
                + " |"
            )
        text = "\n".join(lines) + "\n"
        (output / (name + ".md")).write_text(text)
        pieces.append(f"\n## {name}: {title}\n\n{text}\nSource: {source}. {note}\n")
        entries.append({"table": name, "title": title, "source": source, "note": note})

    meta = json.loads((publication / "METHODS_METADATA.json").read_text())
    data = []
    for d, m in list(meta["datasets"].items()) + [
        ("cicids2017_secondary120k", meta["secondary120k"])
    ]:
        for part in ["train", "validation", "test"]:
            data.append(
                {
                    "Dataset": d,
                    "Split": part,
                    "Records": m[part + "_records"],
                    "Normal": m[part + "_normal_records"],
                    "Attack": m[part + "_attack_records"],
                }
            )
    table(
        "T01_partition_composition",
        "Dataset partition counts",
        pd.DataFrame(data),
        "METHODS_METADATA.json datasets and secondary120k",
        "The 120k rows belong to a separate nested secondary cohort. Original NSL train/validation are drawn from the supplied development file; NSL test is the supplied separate binary file.",
    )
    features = read("features_weights")
    for d in ["nsl_kdd", "cicids2017"]:
        f = features[
            (features.dataset_key == d) & (features.fs_size == 20)
        ].sort_values("rank")
        frame = pd.DataFrame(
            {
                "Rank": f["rank"],
                "Feature": f.feature,
                "MI": f.mi_score.map(lambda x: f"{x:.6f}"),
                "Weight": f.normalized_weight.map(lambda x: f"{x:.6f}"),
                "Discrete": f.is_discrete,
            }
        )
        table(
            "T02_features_" + d,
            "Complete FS20 features and MI weights: " + d,
            frame,
            "tables/features_weights.csv; dataset_key=" + d + ", fs_size=20",
            "Training-only MI. Weights are renormalized within the selected set; full precision is retained in the source CSV. No supplementary table is required.",
        )
    p = read("original_paired_tests_120")
    p = p[p.metric == "test_f1"]
    frame = pd.DataFrame(
        {
            "Dataset": p.dataset_key.replace(
                {"nsl_kdd": "NSL", "cicids2017": "CICIDS"}
            ),
            "k": p.fs_size,
            "Target": p.target_fpr.map(lambda x: f"{x:.2f}"),
            "H F1": p.hamming_mean.map(lambda x: f"{x:.4f}"),
            "W F1": p.weighted_mean.map(lambda x: f"{x:.4f}"),
            "Delta": p.mean_difference_weighted_minus_hamming.map(lambda x: f"{x:.6f}"),
            "95% paired CI": [
                f"[{a:.6f}, {b:.6f}]"
                for a, b in zip(p.bootstrap_ci_low, p.bootstrap_ci_high)
            ],
            "Holm p": p.holm_adjusted_p.map(lambda x: f"{x:.6g}"),
            "W/T/L": [
                f"{a}/{b}/{c}"
                for a, b, c in zip(p.weighted_wins, p.ties, p.weighted_losses)
            ],
        }
    )
    table(
        "T03_original_paired_f1",
        "Original paired F1 comparisons",
        frame,
        "tables/original_paired_tests_120.csv; metric=test_f1",
        "Holm adjusts the complete original family of 120 tests. Five NSL F1 contrasts have adjusted p < 0.05; none on CICIDS. Intervals are 5000 paired bootstrap resamples of 20 seeds.",
    )
    perf = read("original_performance")
    f = perf[perf.approach == "FW-LNSA"]

    def compact(f):
        return pd.DataFrame(
            {
                "Dataset": f.dataset_key,
                "Method": f.procedure.str.replace(
                    "FW-LNSA Weighted Similarity", "Weighted", regex=False
                ),
                "k": f.fs_size,
                "Target": f.target_fpr.map(lambda x: f"{x:.2f}"),
                "Test FPR": f.test_fpr_mean.map(lambda x: f"{x:.6f}"),
                "Recall": f.test_recall_mean.map(lambda x: f"{x:.4f}"),
                "F1": f.test_f1_mean.map(lambda x: f"{x:.4f}"),
                "95% F1 CI": [
                    f"[{a:.6f}, {b:.6f}]"
                    for a, b in zip(f.test_f1_ci_low, f.test_f1_ci_high)
                ],
            }
        )

    table(
        "T04_original_performance",
        "Original absolute performance",
        compact(f),
        "tables/original_performance.csv; approach=FW-LNSA",
        "Mean and Student-t 95% seed intervals. Observed FPR differs across methods; these are not matched-observed-FPR comparisons.",
    )
    f = perf[(perf.fs_size == 20) & (perf.target_fpr == 0.05)]
    table(
        "T05_reference_context",
        "Reference-model context at FS20 and target 0.05",
        compact(f),
        "tables/original_performance.csv; fs_size=20,target_fpr=0.05",
        "Reference models use prespecified, untuned hyperparameters. No claim of equal optimization effort. Other targets/features and all ten metrics are in the curated CSVs.",
    )
    f = read("calibration")
    f = f[f.approach == "FW-LNSA"]
    frame = pd.DataFrame(
        {
            "Dataset": f.dataset_key,
            "Method": f.model,
            "k": f.fs_size,
            "Target": f.target_fpr.map(lambda x: f"{x:.2f}"),
            "Val FPR": f.validation_fpr_mean.map(lambda x: f"{x:.6f}"),
            "Test FPR": f.test_fpr_mean.map(lambda x: f"{x:.6f}"),
            "Test met/20": [f"{n}/20" for n in f.test_target_met_count],
            "Abs error": f.test_absolute_target_error_mean.map(lambda x: f"{x:.6f}"),
            "Threshold median": f.detection_threshold_median.map(lambda x: f"{x:.6g}"),
        }
    )
    table(
        "T06_calibration",
        "Target transfer and seed-specific thresholds",
        frame,
        "tables/calibration.csv; approach=FW-LNSA",
        "Counts and compliance intervals are descriptive across shared-data seeds. The full source contains min/max cutoffs, exact binomial compliance intervals and reference models. Hamming and weighted thresholds have different score orientations.",
    )
    f = read("controlled_benchmark")
    frame = pd.DataFrame(
        {
            "Dataset": f.dataset_key,
            "Method": f.method.replace(
                {"weighted_similarity": "Weighted", "hamming": "Hamming"}
            ),
            "Target": f.target_fpr.map(lambda x: f"{x:.2f}"),
            "Records/s median [IQR]": [
                f"{m:.0f} [{a:.0f}, {b:.0f}]"
                for m, a, b in zip(
                    f.throughput_records_per_sec_median,
                    f.throughput_records_per_sec_q1,
                    f.throughput_records_per_sec_q3,
                )
            ],
            "Peak RSS MiB": [
                f"{m:.2f} [{a:.2f}, {b:.2f}]"
                for m, a, b in zip(
                    f.peak_rss_mb_median, f.peak_rss_mb_q1, f.peak_rss_mb_q3
                )
            ],
            "Detector KiB": (f.detector_state_pickle_bytes_median / 1024).map(
                lambda x: f"{x:.2f}"
            ),
            "Pipeline KiB": (f.deployed_pipeline_pickle_bytes_median / 1024).map(
                lambda x: f"{x:.2f}"
            ),
        }
    )
    table(
        "T07_controlled_efficiency",
        "Controlled efficiency and model-size accounting",
        frame,
        "tables/controlled_benchmark.csv",
        "Ten measured processes per case, each with one warmup; FS20 seed100, batch250, CPU only, one computational thread. Binary scoring/thresholding excludes preprocessing/load/fit/I/O. Peak RSS is timed-stage absolute sampled RSS. Lifetime high-water is separately 853.621094 MiB in this run; it includes earlier process history and is not the timed-stage peak. All sizes are pickle state, not installed executable footprint.",
    )
    f = read("development_summary")
    f = f[
        (f.dataset_key == "cicids2017")
        & f.experiment.isin(["feature_count_20", "no_destination_port_20", "mrmr_20"])
    ]

    def devcompact(f):
        return pd.DataFrame(
            {
                "Dataset": f.dataset_key,
                "Experiment": f.experiment,
                "Method": f.method,
                "F1 mean +/- SD": [
                    f"{m:.6f} +/- {s:.6f}"
                    for m, s in zip(f.validation_f1_mean, f.validation_f1_std)
                ],
                "Recall": f.validation_recall_mean.map(lambda x: f"{x:.6f}"),
                "FPR": f.validation_fpr_mean.map(lambda x: f"{x:.6f}"),
                "Retention": f.retention_rate_mean.map(lambda x: f"{x:.4f}"),
            }
        )

    table(
        "T08_representation_diagnostics",
        "CICIDS development representation contrasts",
        devcompact(f),
        "tables/development_summary.csv; cicids2017, feature_count_20/no_destination_port_20/mrmr_20",
        "Five development seeds, target0.05, frozen original FS20 budget/tolerance. Weighted mRMR has zero detections. Port removal refits the ranking; this is not a causal leakage estimate.",
    )
    f = read("sample120k_metrics")
    rows = []
    for method, g in f.groupby("method"):
        g = g.set_index("metric")
        rows.append(
            {
                "Method": method,
                "F1": f"{g.loc['test_f1', 'mean']:.6f}",
                "95% F1 CI": f"[{g.loc['test_f1', 'ci_low']:.6f}, {g.loc['test_f1', 'ci_high']:.6f}]",
                "Recall": f"{g.loc['test_recall', 'mean']:.6f}",
                "Observed FPR": f"{g.loc['test_fpr', 'mean']:.6f}",
            }
        )
    table(
        "T09_secondary120k",
        "Nested 120k sample-size/repartitioning sensitivity",
        pd.DataFrame(rows),
        "tables/sample120k_metrics.csv, sample120k_paired.csv and sample120k_overlap.csv",
        "20 evaluation seeds300-319 after tuning seeds200-204. Paired F1 difference0.0572397687, 95% percentile interval[-0.0032292264,0.1151099189], W/T/L13/0/7. All original60k records recur; 6810 originaltest rows enter secondarytrain and1823 secondaryvalidation. No independent validation or matched-FPR superiority claim.",
    )
    f = read("category_recall")
    f = f[(f.fs_size == 20) & (f.target_fpr == 0.05)]
    rows = []
    for category, g in f.groupby("category"):
        row = {"Category": category, "Support": int(g.support.iloc[0])}
        for method in ["hamming", "weighted_similarity"]:
            q = g[g.method == method].iloc[0]
            row[method] = (
                "NOT_ESTIMABLE"
                if q.support == 0
                else f"{q['mean']:.4f} [{q.ci_low:.4f}, {q.ci_high:.4f}]"
            )
        rows.append(row)
    table(
        "T10_category_recall",
        "CICIDS category recall with support",
        pd.DataFrame(rows),
        "tables/category_recall.csv; fs_size=20,target_fpr=0.05",
        "Means and exact unclipped95%seed intervals. No test support means not estimable, not zero recall. Original-label/category mappings are in cicids_category_map.csv.",
    )
    f = read("stability_summary")
    table(
        "T11_stability",
        "Training feature-selection stability",
        f,
        "tables/stability_summary.csv",
        "30 training-only resamples per dataset; 435 dependent pairs per k. Jaccard mean/SD/min/max are descriptive; no independent435-replicate CI is implied.",
    )
    f = read("development_summary")
    table(
        "T12_feature_count",
        "Development feature-count sensitivity",
        devcompact(f[f.experiment.str.startswith("feature_count_")]),
        "tables/development_summary.csv; experiment starts feature_count_",
        "Five development seeds; k5/10/15/20/30; no primary test reselection.",
    )
    table(
        "T13_self_cap",
        "Development self-cap sensitivity",
        devcompact(f[f.experiment.str.startswith("self_cap_")]),
        "tables/development_summary.csv; experiment starts self_cap_",
        "Caps2500/5000/10000 apply before self deduplication. Five development seeds; primary locks unchanged.",
    )
    (output / "MANUSCRIPT_TABLES.md").write_text("\n".join(pieces))
    (output / "TABLE_INDEX.json").write_text(json.dumps(entries, indent=2) + "\n")
    print("Manuscript preparation tables:", len(entries))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publication", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    build(a.publication, a.output)


if __name__ == "__main__":
    main()
