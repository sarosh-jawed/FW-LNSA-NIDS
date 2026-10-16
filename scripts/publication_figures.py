"""Vector artwork from curated verified CSV tables, with no scientific refitting."""

from pathlib import Path
from io import BytesIO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

DATASETS = {"nsl_kdd": "NSL-KDD", "cicids2017": "CICIDS2017"}
METHODS = {"hamming": "Hamming", "weighted_similarity": "Weighted"}


def make_figures(root):
    root = Path(root)
    folder = root / "figures"
    folder.mkdir()
    inventory = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1,
            "savefig.facecolor": "white",
        }
    )

    def read(name):
        return pd.read_csv(
            root / "tables" / (name + ".csv"), float_precision="round_trip"
        )

    def save(fig, stem, placement, caption, sources, rows, priority="Main text"):
        fig.savefig(
            folder / (stem + ".pdf"), metadata={"CreationDate": None, "ModDate": None}
        )
        # Encode fully in memory before writing, then verify the saved image.
        # This also detects truncated multi-chunk PNG writes on hosted filesystems.
        from PIL import Image

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=600)
        content = buffer.getvalue()
        path = folder / (stem + ".png")
        if path.write_bytes(content) != len(content) or path.read_bytes() != content:
            raise OSError("Incomplete PNG write: " + path.name)
        with Image.open(path) as rendered:
            rendered.load()
        inventory.append(
            {
                "figure_id": stem.split("_")[0],
                "pdf": "figures/" + stem + ".pdf",
                "png": "figures/" + stem + ".png",
                "recommended_placement": placement,
                "priority": priority,
                "caption": caption,
                "underlying_csv": ";".join("tables/" + s + ".csv" for s in sources),
                "source_rows_and_columns": rows,
                "recommended_width": "Double column, 183 mm; do not shrink to one column",
                "png_dpi": 600,
            }
        )
        plt.close(fig)

    # The workflow follows implemented train/validation/test dependencies.
    fig, ax = plt.subplots(figsize=(7.2, 6), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = {
        "fit": (
            0.025,
            0.79,
            0.45,
            0.18,
            "TRAIN ONLY\nEncode, impute and scale predictors.\nFit label-informed MI rank/weights\nand selected-feature medians.",
        ),
        "bits": (
            0.525,
            0.79,
            0.45,
            0.18,
            "FROZEN REPRESENTATION\nApply training transformations to all splits.\nBinary bit = 1 if x > train median.\nNo test-driven feature selection.",
        ),
        "pool": (
            0.025,
            0.49,
            0.45,
            0.23,
            "TRAIN-ONLY DETECTOR POOL\nSample normal rows without replacement\n(up to 10,000), then unique self patterns.\nGenerate one seeded candidate batch.\nReject self matches; retain candidate\nduplicates. No replenishment loop.",
        ),
        "dev": (
            0.525,
            0.49,
            0.45,
            0.23,
            "DEVELOPMENT: SEEDS 42-46\nCalibrate on benign validation scores.\nSelect each method's budget/tolerance\nusing validation F1 and ordered tie breaks.\nLock configurations before confirmation.",
        ),
        "confirm": (
            0.025,
            0.22,
            0.45,
            0.20,
            "CONFIRMATION: SEEDS 100-119\nFit new training pools at locked settings.\nCalibrate each seed's threshold on the\nsame benign validation partition.\nInclusive score ties remain grouped.",
        ),
        "test": (
            0.525,
            0.22,
            0.45,
            0.20,
            "HELD-OUT TEST\nFreeze preprocessing, pool and threshold.\nScore and compute per-seed metrics.\nPair methods by seed; preserve original\n120-test Holm family and seed intervals.",
        ),
    }
    for x, y, w, h, text in boxes.values():
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.01",
                facecolor=".965",
                edgecolor=".25",
                linewidth=0.8,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.45,
        )

    def arrow(a, b):
        ax.annotate(
            "", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": ".2", "lw": 1}
        )

    arrow((0.475, 0.88), (0.525, 0.88))
    arrow((0.75, 0.79), (0.75, 0.755))
    ax.plot([0.75, 0.25], [0.755, 0.755], color=".2", lw=1)
    arrow((0.25, 0.755), (0.25, 0.72))
    arrow((0.475, 0.605), (0.525, 0.605))
    arrow((0.75, 0.49), (0.75, 0.455))
    ax.plot([0.75, 0.25], [0.455, 0.455], color=".2", lw=1)
    arrow((0.25, 0.455), (0.25, 0.42))
    arrow((0.475, 0.32), (0.525, 0.32))
    ax.text(
        0.5,
        0.11,
        "Threshold calibration controls empirical benign-validation FPR, not probability calibration.\nWeighted and Hamming settings are selected separately; this is not an isolated formula ablation.",
        ha="center",
        va="center",
        fontsize=8.5,
        linespacing=1.5,
    )
    save(
        fig,
        "F01_implemented_workflow",
        "Methodology, fig:fw_lnsa_architecture",
        "Implemented original FW-LNSA workflow. Training labels inform mutual-information ranking and weights; only detector generation uses normal traffic. Development configurations and seed-specific validation thresholds are frozen before test scoring. Candidate duplicates are retained. ECE and Brier score are not computed, and detector scores are not probabilities.",
        ["original_configuration_locks"],
        "Source: src/preprocessing.py, src/feature_selection.py, src/fw_lnsa.py, src/detector_generation.py, src/confirmatory_protocol.py; original_configuration_locks.csv (all rows)",
    )

    p = read("original_paired_tests_120")
    p = p[p.metric == "test_f1"]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.55), sharex=True, layout="constrained"
    )
    for ax, (d, g) in zip(axes, p.groupby("dataset_key", sort=True)):
        g = g.sort_values(["fs_size", "target_fpr"])
        y = np.arange(len(g))
        v = g.mean_difference_weighted_minus_hamming
        ax.errorbar(
            v,
            y,
            xerr=[v - g.bootstrap_ci_low, g.bootstrap_ci_high - v],
            fmt="o",
            color="black",
            capsize=2.5,
            markersize=4,
        )
        ax.axvline(0, ls=":", color=".45", lw=0.8)
        ax.set_yticks(
            y, [f"FS-{r.fs_size}, {r.target_fpr:.2f}" for r in g.itertuples()]
        )
        ax.set_title(DATASETS[d])
        ax.set_xlabel("Weighted minus Hamming F1")
    save(
        fig,
        "F02_primary_paired_f1",
        "Results, subsec:paired_analysis",
        "Original paired mean F1 differences and 95% percentile bootstrap intervals (5,000 paired detector-seed resamples, 20 seeds per condition). Labels give feature count and nominal validation FPR target. Significance uses the preserved 120-test Holm family, not a 12-F1 family. No CICIDS2017 F1 comparison has adjusted p < 0.05. The configurations were selected separately.",
        ["original_paired_tests_120"],
        "metric=test_f1; dataset_key,fs_size,target_fpr; mean_difference_weighted_minus_hamming,bootstrap_ci_low,bootstrap_ci_high",
    )

    perf = read("original_performance")
    fw = perf[perf.approach == "FW-LNSA"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), layout="constrained")
    for col, d in enumerate(["nsl_kdd", "cicids2017"]):
        for row, metric in enumerate(["test_f1", "test_recall"]):
            ax = axes[row, col]
            for (method, k), g in fw[fw.dataset_key == d].groupby(
                ["procedure", "fs_size"]
            ):
                weighted = "Weighted" in method
                mean = g[metric + "_mean"].to_numpy()
                ax.errorbar(
                    g.test_fpr_mean,
                    mean,
                    yerr=[mean - g[metric + "_ci_low"], g[metric + "_ci_high"] - mean],
                    fmt="o" if weighted else "s",
                    fillstyle="full" if k == 20 else "none",
                    color="black" if k == 20 else ".45",
                    capsize=2,
                    markersize=4,
                    label=f"{'W' if weighted else 'H'}, FS-{k}",
                )
            ax.set_title(DATASETS[d] if row == 0 else "")
            ax.set_xlabel("Observed test FPR")
            ax.set_ylabel("F1" if row == 0 else "Recall")
            ax.set_ylim(
                -0.035, max(0.4 if d == "cicids2017" else 0.85, ax.get_ylim()[1])
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside upper center", frameon=False)
    save(
        fig,
        "F03_primary_utility_and_false_alarms",
        "Results, confirmatory performance",
        "Original mean F1 and recall versus observed test FPR. Vertical bars are 95% Student-t intervals across 20 detector seeds on fixed partitions; horizontal positions are mean FPR, not matched-FPR operating points. W denotes weighted matching and H Hamming; open markers denote FS10 and filled markers FS20. Near-zero CICIDS2017 FS10 utility and conditional NSL-KDD gains are retained.",
        ["original_performance"],
        "approach=FW-LNSA; dataset_key,procedure,fs_size,target_fpr; test_fpr_mean,test_f1_mean/ci_low/ci_high,test_recall_mean/ci_low/ci_high",
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    for ax, d in zip(axes, ["nsl_kdd", "cicids2017"]):
        g = fw[fw.dataset_key == d]
        for (method, k), q in g.groupby(["procedure", "fs_size"]):
            x = q.validation_fpr_mean.to_numpy()
            y = q.test_fpr_mean.to_numpy()
            ax.errorbar(
                x,
                y,
                xerr=[
                    np.maximum(0, x - q.validation_fpr_ci_low),
                    np.maximum(0, q.validation_fpr_ci_high - x),
                ],
                yerr=[
                    np.maximum(0, y - q.test_fpr_ci_low),
                    np.maximum(0, q.test_fpr_ci_high - y),
                ],
                fmt="o" if "Weighted" in method else "s",
                fillstyle="full" if k == 20 else "none",
                color="black" if k == 20 else ".5",
                capsize=2,
                markersize=3.5,
                label=f"{'W' if 'Weighted' in method else 'H'}, FS-{k}",
            )
        upper = max(0.11, ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, upper], [0, upper], ":", color=".5")
        ax.set_xlim(-0.003, upper)
        ax.set_ylim(-0.003, upper)
        ax.set_title(DATASETS[d])
        ax.set_xlabel("Mean validation FPR")
        ax.set_ylabel("Mean test FPR")
        ax.legend(ncol=2, loc="upper left")
    save(
        fig,
        "F04_validation_to_test_fpr",
        "Results, calibration reliability",
        "Original validation and test FPR means with marginal 95% detector-seed Student-t intervals. The dotted line marks equal FPR, not a model or guarantee. Shared fixed calibration data induce dependence between seeds. The full calibration table supplies target-met counts out of 20 and absolute target errors; nominal ceilings are not claims of exact test FPR.",
        ["original_performance", "calibration"],
        "approach=FW-LNSA; test_fpr_mean/ci_low/ci_high,validation_fpr_mean/ci_low/ci_high",
    )

    patterns = read("binary_patterns")
    pairs = read("stability_pairwise")
    fig = plt.figure(figsize=(7.2, 5.3), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    for col, d in enumerate(["nsl_kdd", "cicids2017"]):
        ax = fig.add_subplot(gs[0, col])
        for k, g in patterns[patterns.dataset_key == d].groupby("fs_size"):
            g = g.set_index("partition").loc[["train", "validation", "test"]]
            ax.plot(
                np.arange(3),
                100 * g.unique_fraction,
                "o-" if k == 20 else "s--",
                color="black" if k == 20 else ".5",
                markersize=4,
                label=f"FS-{k}",
            )
        ax.set_xticks(range(3), ["Train", "Validation", "Test"])
        ax.set_ylabel("Unique bit patterns / records (%)")
        ax.set_title(DATASETS[d])
        ax.legend()
    ax = fig.add_subplot(gs[1, :])
    values = []
    labels = []
    for d in ["nsl_kdd", "cicids2017"]:
        for k in [10, 20]:
            values.append(
                pairs[(pairs.dataset_key == d) & (pairs.k == k)].jaccard.to_numpy()
            )
            labels.append(DATASETS[d] + f"\nFS-{k}")
    ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": ".85", "linewidth": 0.8},
        medianprops={"color": "black"},
        flierprops={
            "marker": ".",
            "markersize": 3,
            "markerfacecolor": ".3",
            "markeredgecolor": ".3",
        },
    )
    ax.set_ylabel("Training-resample Jaccard")
    ax.set_ylim(min(np.min(v) for v in values) - 0.025, 1.015)
    save(
        fig,
        "F05_representation_and_stability",
        "Results/Discussion, representation and selection diagnostics",
        "Descriptive representation compression (top) and top-k feature-set overlap across 30 stratified 80% training-only resamples (bottom). Training-fitted preprocessing is held fixed while MI is refitted. Each box summarizes 435 dependent pairwise Jaccard values, not 435 independent replications. CICIDS2017 FS20 has 415 unique training patterns among 33,600 rows. These associations do not establish a causal failure mechanism.",
        ["binary_patterns", "stability_pairwise"],
        "binary_patterns: unique_fraction by dataset_key,fs_size,partition; stability_pairwise: jaccard by dataset_key,k",
    )

    dev = read("development_summary")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), layout="constrained")
    for col, d in enumerate(["nsl_kdd", "cicids2017"]):
        for row, prefix, xcol in [
            (0, "feature_count_", "k"),
            (1, "self_cap_", "self_cap"),
        ]:
            ax = axes[row, col]
            g = dev[(dev.dataset_key == d) & dev.experiment.str.startswith(prefix)]
            for method, q in g.groupby("method"):
                q = q.sort_values(xcol)
                x = q[xcol].to_numpy() / (1000 if row else 1)
                ax.errorbar(
                    x,
                    q.validation_f1_mean,
                    yerr=q.validation_f1_std,
                    fmt="o-" if method == "weighted_similarity" else "s--",
                    color="black" if method == "weighted_similarity" else ".5",
                    capsize=2,
                    markersize=4,
                    label=METHODS[method],
                )
            ax.set_title(DATASETS[d] if row == 0 else "")
            ax.set_xlabel(
                "Selected feature count k" if row == 0 else "Self-row cap (thousands)"
            )
            ax.set_ylabel("Validation F1 (mean +/- SD)")
            ax.set_ylim(-0.07, 1.05)
            if row == 0:
                ax.set_xticks([5, 10, 15, 20, 30])
                ax.legend()
            else:
                ax.set_xticks([2.5, 5, 10])
    save(
        fig,
        "F06_training_only_sensitivities",
        "Results, separate development robustness subsection",
        "Development-only feature-count (top) and self-cap (bottom) sensitivities at validation target 0.05, five seeds 42-46, mean +/- sample SD. Original FS20 detector budget and tolerance are reused; features and weights are trained without held-out test selection. The cap applies to normal rows before self-pattern deduplication. These exploratory curves do not retrospectively justify primary feature counts or replace confirmatory locks.",
        ["development_summary"],
        "experiment starts feature_count_ or self_cap_; dataset_key,method,k,self_cap; validation_f1_mean/std",
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
    exps = ["feature_count_20", "no_destination_port_20", "mrmr_20"]
    g = dev[(dev.dataset_key == "cicids2017") & dev.experiment.isin(exps)]
    for ax, metric, title in zip(
        axes, ["validation_f1", "validation_fpr"], ["Validation F1", "Validation FPR"]
    ):
        for method, offset, marker, color in [
            ("hamming", -0.09, "s", ".5"),
            ("weighted_similarity", 0.09, "o", "black"),
        ]:
            q = g[g.method == method].set_index("experiment").loc[exps]
            ax.errorbar(
                np.arange(3) + offset,
                q[metric + "_mean"],
                yerr=q[metric + "_std"],
                fmt=marker,
                color=color,
                capsize=3,
                label=METHODS[method],
            )
        ax.set_xticks(range(3), ["Original MI", "No port + MI", "mRMR (MID)"])
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylabel(title + " (mean +/- SD)")
        ax.legend()
    save(
        fig,
        "F07_cicids_representation_diagnostics",
        "Results/Discussion, CICIDS negative-result analysis",
        "CICIDS2017 FS20 development diagnostics at validation target 0.05 (five seeds; mean +/- SD). Port removal refits training-only MI ranking. mRMR uses empirical discretized MI relevance minus mean redundancy with lexical ties. The frozen original detector budget/tolerance is reused; these are joint representation/configuration diagnostics, not causal leakage or weighting-only ablations. Weighted mRMR produces no detections: zero recall and FPR. This adverse result is retained.",
        ["development_summary"],
        "dataset_key=cicids2017; experiment=feature_count_20/no_destination_port_20/mrmr_20; validation_f1_mean/std,validation_fpr_mean/std",
    )

    cats = read("category_recall")
    cats = cats[(cats.fs_size == 20) & np.isclose(cats.target_fpr, 0.05)]
    labels = sorted(cats.category.unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.7), layout="constrained")
    supports = []
    for method, offset, marker, color in [
        ("hamming", -0.13, "s", ".5"),
        ("weighted_similarity", 0.13, "o", "black"),
    ]:
        q = cats[cats.method == method].set_index("category").loc[labels]
        valid = q.support.to_numpy() > 0
        x = q["mean"].to_numpy()
        lo = np.maximum(0, q.ci_low.to_numpy())
        hi = np.minimum(1, q.ci_high.to_numpy())
        ax.errorbar(
            x[valid],
            np.arange(len(q))[valid] + offset,
            xerr=[
                np.maximum(0, x[valid] - lo[valid]),
                np.maximum(0, hi[valid] - x[valid]),
            ],
            fmt=marker,
            color=color,
            capsize=2,
            markersize=4,
            label=METHODS[method],
        )
        supports = q.support.tolist()
    ax.set_yticks(
        range(len(labels)), [f"{c} (n={int(n)})" for c, n in zip(labels, supports)]
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.015, 1.01)
    ax.set_xlabel("Test recall (95% detector-seed interval)")
    ax.legend(loc="lower right")
    for i, n in enumerate(supports):
        if n == 0:
            ax.text(0.15, i, "NOT ESTIMABLE", va="center", fontsize=8.5, color=".35")
    save(
        fig,
        "F08_cicids_category_recall",
        "Results, attack-category analysis",
        "Original-model CICIDS2017 FS20 category recall at nominal target 0.05, recovered by verified frozen replay (20 seeds). Error bars are 95% seed Student-t intervals, displayed within [0,1]; exact unclipped endpoints remain in the CSV. Support is test-record count, shared across seeds. Heartbleed and Infiltration have no test support and are not estimable. Methods have different observed FPRs. Original-label mapping is supplied in the category-map table.",
        ["category_recall", "cicids_category_map"],
        "fs_size=20,target_fpr=0.05; method,category; support,mean,ci_low,ci_high",
    )

    bench = (
        read("controlled_benchmark")
        .sort_values(["dataset_key", "target_fpr", "method"])
        .reset_index(drop=True)
    )
    y = np.arange(len(bench))
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 4.3), sharey=True, layout="constrained"
    )
    for ax, metric, label in [
        (
            axes[0],
            "throughput_records_per_sec",
            "Scoring throughput\n(records/s, log scale)",
        ),
        (axes[2], "peak_rss_mb", "Timed-stage peak RSS\n(MiB)"),
    ]:
        med = bench[metric + "_median"].to_numpy()
        lo = bench[metric + "_q1"].to_numpy()
        hi = bench[metric + "_q3"].to_numpy()
        ax.errorbar(
            med,
            y,
            xerr=[med - lo, hi - med],
            fmt="o",
            color="black",
            capsize=2,
            markersize=3.5,
        )
        ax.set_xlabel(label)
    axes[0].set_xscale("log")
    axes[0].set_yticks(
        y,
        [
            f"{'CICIDS' if r.dataset_key == 'cicids2017' else 'NSL'} {METHODS[r.method]}\nTarget {r.target_fpr:.2f}"
            for r in bench.itertuples()
        ],
    )
    axes[0].invert_yaxis()
    axes[1].plot(
        bench.detector_state_pickle_bytes_median / 1024,
        y - 0.11,
        "s",
        ms=3.5,
        color=".5",
        label="Detector state",
    )
    axes[1].plot(
        bench.deployed_pipeline_pickle_bytes_median / 1024,
        y + 0.11,
        "o",
        ms=3.5,
        color="black",
        label="Full pipeline",
    )
    axes[1].set_xlabel("Serialized state\n(KiB)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.18), fontsize=8.5)
    save(
        fig,
        "F09_controlled_efficiency",
        "Results, efficiency trade-offs",
        "Controlled full-record scoring for eight frozen FS20 cases, seed 100. Timing and sampled peak RSS show medians and IQRs from ten fresh processes per case, each with one warmup. Timing includes binary scoring and thresholding, and excludes preprocessing, loading, fit and I/O. CPU: AMD EPYC 7B12; one computational thread. Detector-state pickle and complete fitted deployment-state pickle are distinct; library executables are excluded. RSS is absolute timed-stage process memory, not its near-zero increment or the process lifetime high-water mark. These measurements do not establish real-time deployment performance.",
        ["controlled_benchmark"],
        "dataset_key,method,target_fpr; throughput_records_per_sec_median/q1/q3,peak_rss_mb_median/q1/q3,detector_state_pickle_bytes_median,deployed_pipeline_pickle_bytes_median",
    )

    secondary = read("sample120k_metrics")
    raw = read("sample120k_seed_metrics")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["test_f1", "test_recall", "test_fpr"],
        ["Test F1", "Test recall", "Observed test FPR"],
    ):
        for x, method in enumerate(["hamming", "weighted_similarity"]):
            values = raw[raw.method == method].sort_values("seed")[metric].to_numpy()
            q = secondary[
                (secondary.method == method) & (secondary.metric == metric)
            ].iloc[0]
            ax.scatter(
                x + np.linspace(-0.09, 0.09, len(values)),
                values,
                s=8,
                color=".7",
                zorder=1,
            )
            ax.errorbar(
                x,
                q["mean"],
                yerr=[[q["mean"] - q.ci_low], [q.ci_high - q["mean"]]],
                fmt="s" if x == 0 else "o",
                color="black",
                capsize=3,
                ms=4,
                zorder=2,
            )
        ax.set_xticks([0, 1], ["Hamming", "Weighted"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel(title)
    save(
        fig,
        "F10_secondary_120k",
        "Results, separate sample-size/repartitioning sensitivity",
        "Secondary nested 120k cohort, FS20, validation target 0.05, fresh evaluation seeds 300-319. Gray dots are detector-seed results; black marks and bars are means and 95% seed Student-t intervals. The paired F1 difference is 0.057240, with a 95% paired bootstrap interval [-0.003229, 0.115110]. Observed test FPR is 0.005483 (Hamming) versus 0.041481 (weighted). All original 60k records are nested in the cohort, and 6,810 original test rows enter new training. This is not independent external validation or a matched-observed-FPR comparison.",
        [
            "sample120k_seed_metrics",
            "sample120k_metrics",
            "sample120k_paired",
            "sample120k_overlap",
        ],
        "sample120k_seed_metrics: seed=300..319,method,test_f1/recall/fpr; sample120k_metrics: mean,ci_low,ci_high",
    )
    return inventory
