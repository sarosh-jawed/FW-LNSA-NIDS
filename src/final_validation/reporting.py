"""Publication table data and grayscale vector figures; manuscript is untouched."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evidence import read_csv
from .io import sha256, table, write_json


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(folder / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def report(root, accepted):
    root, accepted = Path(root), Path(accepted)
    out = root / "report"
    out.mkdir(exist_ok=True)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    performance = read_csv(
        accepted / "evidence/tables_csv/confirmatory_performance.csv"
    )
    paired = read_csv(accepted / "evidence/tables_csv/paired_weighted_vs_hamming.csv")
    p = paired[paired.metric == "test_f1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for ax, (dataset, g) in zip(axes, p.groupby("dataset_key", sort=True)):
        g = g.sort_values(["fs_size", "target_fpr"])
        y = np.arange(len(g))
        v = g.mean_difference_weighted_minus_hamming
        ax.errorbar(
            v,
            y,
            xerr=[v - g.bootstrap_ci_low, g.bootstrap_ci_high - v],
            fmt="o",
            color="black",
            capsize=3,
        )
        ax.axvline(0, color=".6", ls="--")
        ax.set_yticks(
            y, [f"FS-{r.fs_size}, target {r.target_fpr:g}" for r in g.itertuples()]
        )
        ax.set_title(dataset)
        ax.set_xlabel("Weighted − Hamming F1 (95% paired CI)")
    save(fig, figures, "paired_f1_original_120_family")
    fw = performance[performance.approach == "FW-LNSA"]
    for metric, title in [("test_f1", "F1"), ("test_recall", "Recall")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, (dataset, g) in zip(axes, fw.groupby("dataset_key", sort=True)):
            for (method, size), q in g.groupby(["procedure", "fs_size"]):
                marker = "o" if "Weighted" in method else "s"
                color = "black" if size == 20 else ".55"
                ax.errorbar(
                    q.test_fpr_mean,
                    q[metric + "_mean"],
                    yerr=[
                        np.maximum(0, q[metric + "_mean"] - q[metric + "_ci_low"]),
                        np.maximum(0, q[metric + "_ci_high"] - q[metric + "_mean"]),
                    ],
                    fmt=marker,
                    color=color,
                    label=f"{method}, FS-{size}",
                    capsize=2,
                )
            ax.set_title(dataset)
            ax.set_xlabel("Realized test FPR")
            ax.set_ylabel(title + " (95% seed t interval)")
            ax.legend(fontsize=7)
        save(fig, figures, "original_" + metric + "_vs_realized_fpr")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (dataset, g) in zip(axes, fw.groupby("dataset_key", sort=True)):
        for method, q in g.groupby("procedure"):
            ax.scatter(
                q.validation_fpr_mean,
                q.test_fpr_mean,
                marker="o" if "Weighted" in method else "s",
                facecolors="none" if "Weighted" in method else ".3",
                edgecolors="black",
                label=method,
            )
        maximum = max(g.validation_fpr_mean.max(), g.test_fpr_mean.max()) * 1.1
        ax.plot([0, maximum], [0, maximum], ls="--", color=".6")
        ax.set_title(dataset)
        ax.set_xlabel("Mean validation FPR")
        ax.set_ylabel("Mean test FPR")
        ax.legend(fontsize=8)
    save(fig, figures, "original_validation_vs_test_fpr")
    efficiency = read_csv(accepted / "evidence/tables_csv/efficiency_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for (dataset, method), g in efficiency[efficiency.approach == "FW-LNSA"].groupby(
        ["dataset_key", "procedure"]
    ):
        ax.scatter(
            g.model_size_bytes_mean / 1024,
            g.prediction_records_per_sec_mean,
            marker="o" if "Weighted" in method else "s",
            facecolors="none" if dataset == "cicids2017" else ".3",
            edgecolors="black",
            label=f"{dataset} {method}",
        )
    ax.set_xlabel("Original detector-state pickle size (KiB)")
    ax.set_ylabel("Original mean throughput (records/s)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("Historical single-run resource evidence; not a controlled rerun")
    save(fig, figures, "historical_storage_throughput_context")
    for key in ["nsl_kdd", "cicids2017"]:
        files = list((root / (key + "_data")).glob("fs*_patterns.csv"))
        if files:
            fig, ax = plt.subplots(figsize=(6, 4))
            for path in files:
                g = read_csv(path)
                ax.plot(
                    g.partition,
                    g.unique_fraction,
                    "o--" if "10" in path.stem else "s-",
                    color="black",
                    label=path.stem.split("_")[0].upper(),
                )
            ax.set_ylabel("Unique bit patterns / records")
            ax.set_title(key + " • descriptive representation compression")
            ax.legend()
            save(fig, figures, key + "_binary_patterns")
        freqpath = root / (key + "_stability") / "selection_frequency.csv"
        if freqpath.exists():
            g = read_csv(freqpath)
            g = (
                g[g.k == 20]
                .sort_values("selection_frequency", ascending=False)
                .head(25)
                .sort_values("selection_frequency")
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(g.feature, g.selection_frequency, color=".35")
            ax.set_xlim(0, 1.02)
            ax.set_xlabel("Top-20 selection frequency")
            ax.set_title(key + " • training-only stability")
            save(fig, figures, key + "_feature_stability")
        catpath = root / (key + "_replay") / "category_summary.csv"
        if catpath.exists():
            g = read_csv(catpath)
            g = g[
                (g.fs_size == 20)
                & np.isclose(g.target_fpr, 0.05)
                & (g.method == "weighted_similarity")
            ]
            if len(g):
                fig, ax = plt.subplots(figsize=(8, 4))
                g = g.sort_values("category")
                x = np.arange(len(g))
                valid = g.support > 0
                means = g.loc[valid, "mean"]
                low = g.loc[valid, "ci_low"]
                high = g.loc[valid, "ci_high"]
                ax.errorbar(
                    x[valid],
                    means,
                    yerr=[np.maximum(0, means - low), np.maximum(0, high - means)],
                    fmt="o",
                    color="black",
                    capsize=3,
                )
                ax.set_xticks(
                    x,
                    [f"{r.category}\n(n={r.support})" for r in g.itertuples()],
                    rotation=25,
                    ha="right",
                )
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel("Recall, 95% seed t interval")
                ax.set_title(f"{key}, weighted FS-20, target .05")
                save(fig, figures, key + "_category_support_recall")
    controlled_path = root / "benchmark/benchmark_summary.csv"
    if controlled_path.exists():
        controlled = read_csv(controlled_path)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        case_labels = []
        for row in controlled.itertuples():
            x = row.detector_state_pickle_bytes_median / 1024
            y = row.throughput_records_per_sec_median
            ax.errorbar(
                x,
                y,
                xerr=[
                    [max(0, x - row.detector_state_pickle_bytes_q1 / 1024)],
                    [max(0, row.detector_state_pickle_bytes_q3 / 1024 - x)],
                ],
                yerr=[
                    [max(0, y - row.throughput_records_per_sec_q1)],
                    [max(0, row.throughput_records_per_sec_q3 - y)],
                ],
                fmt="o" if row.method == "weighted_similarity" else "s",
                color="black",
                capsize=3,
            )
            case_number = len(case_labels) + 1
            case_labels.append(
                f"{case_number}: {row.dataset_key}, {row.method}, {row.target_fpr:g}"
            )
            ax.annotate(
                str(case_number),
                (x, y),
                fontsize=8,
                xytext=(5, 9 if row.target_fpr == 0.01 else -13),
                textcoords="offset points",
            )
        ax.set_xlabel("Detector-state pickle size (KiB), median/IQR")
        ax.set_ylabel("Controlled throughput (records/s), median/IQR")
        ax.set_yscale("log")
        ax.set_title("FS-20, targets .01/.05, seed 100; 10 repeats per condition")
        ax.text(
            1.03,
            1,
            "\n".join(case_labels),
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.margins(x=0.08, y=0.18)
        save(fig, figures, "controlled_storage_throughput")

    # Save portable publication table data in CSV, Markdown and LaTeX fragments.
    tables = {
        "primary_performance_unchanged": performance,
        "paired_statistics_original_120": paired,
    }
    for name, path in [
        ("calibration_traceability", root / "evidence/calibration_traceability.csv"),
        ("controlled_benchmark", root / "benchmark/benchmark_summary.csv"),
    ]:
        if path.exists():
            tables[name] = read_csv(path)
    for name, frame in tables.items():
        table(out / (name + ".csv"), frame)
        # Avoid optional tabulate dependency and preserve unrounded CSV values.
        with (out / (name + ".md")).open("w") as f:
            f.write(
                "| "
                + " | ".join(frame.columns)
                + " |\n| "
                + " | ".join(["---"] * len(frame.columns))
                + " |\n"
            )
            for row in frame.itertuples(index=False, name=None):
                f.write(
                    "| " + " | ".join(str(v).replace("|", "/") for v in row) + " |\n"
                )
        (out / (name + ".tex")).write_text(
            frame.to_latex(index=False, float_format="%.6g", escape=True)
        )
    for dataset_key, group in fw.groupby("dataset_key"):
        compact = pd.DataFrame(
            {
                "Method": group.procedure.replace({"Weighted Similarity": "Weighted"}),
                "k": group.fs_size,
                "Target": group.target_fpr,
                "Val FPR": group.validation_fpr_mean,
                "Test FPR": group.test_fpr_mean,
                "Recall": group.test_recall_mean,
                "F1": group.test_f1_mean,
                "95percent CI": [
                    f"[{low:.4f}, {high:.4f}]"
                    for low, high in zip(group.test_f1_ci_low, group.test_f1_ci_high)
                ],
            }
        )
        table(out / ("primary_compact_" + dataset_key + ".csv"), compact)
        (out / ("primary_compact_" + dataset_key + ".tex")).write_text(
            compact.to_latex(index=False, float_format="%.4f", escape=True)
        )
    compact_pair = pd.DataFrame(
        {
            "Dataset": p.dataset_key,
            "k": p.fs_size,
            "Target": p.target_fpr,
            "Delta F1": p.mean_difference_weighted_minus_hamming,
            "95percent paired CI": [
                f"[{low:.4f}, {high:.4f}]"
                for low, high in zip(p.bootstrap_ci_low, p.bootstrap_ci_high)
            ],
            "Holm120 p": p.holm_adjusted_p,
            "W/T/L": [
                f"{w}/{t}/{loss}"
                for w, t, loss in zip(p.weighted_wins, p.ties, p.weighted_losses)
            ],
        }
    )
    table(out / "paired_f1_compact.csv", compact_pair)
    (out / "paired_f1_compact.tex").write_text(
        compact_pair.to_latex(index=False, float_format="%.6g", escape=True)
    )

    files = {
        p.relative_to(out).as_posix(): sha256(p)
        for p in out.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(
        out / "manifest.json",
        {
            "identity": "regenerated_report",
            "status": (
                "VERIFIED_SYNTHETIC_ONLY"
                if (root / "execution_identity.json").exists()
                and json.loads((root / "execution_identity.json").read_text()).get(
                    "profile"
                )
                == "smoke"
                else "VERIFIED_REAL_DATA"
            ),
            "outputs": files,
            "evidence_scope": "Primary source files read-only; secondary plots appear only for available completed jobs. Synthetic output roots must never be used for manuscript.",
        },
    )
