"""Create publication-ready figures from final research summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def make_figures(output_dir: Path) -> list[Path]:
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    best = _read_optional(table_dir / "research_best_fpr_controlled.csv")
    budget = _read_optional(table_dir / "research_detector_budget_sensitivity.csv")
    feature = _read_optional(table_dir / "research_feature_set_ablation.csv")

    paths: list[Path] = []

    if not best.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (dataset, method), group in best.groupby(["dataset", "method_name"]):
            ax.scatter(group["fpr_mean"], group["recall_mean"], label=f"{dataset}: {method}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("Recall")
        ax.set_title("FW-LNSA balanced recall and false positive tradeoff")
        ax.legend(fontsize=8)
        path = figure_dir / "research_recall_fpr_tradeoff.png"
        _save(fig, path)
        paths.append(path)

    if not budget.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (dataset, method), group in budget.groupby(["dataset", "method_name"]):
            ordered = group.sort_values("detector_budget")
            ax.plot(
                ordered["detector_budget"],
                ordered["total_time_sec_mean"],
                marker="o",
                label=f"{dataset}: {method}",
            )
        ax.set_xlabel("Detector budget")
        ax.set_ylabel("Mean total runtime in seconds")
        ax.set_title("Runtime sensitivity to detector budget")
        ax.legend(fontsize=8)
        path = figure_dir / "research_runtime_detector_budget.png"
        _save(fig, path)
        paths.append(path)

    if not feature.empty and "delta_f1" in feature.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = feature["dataset"].astype(str) + ": " + feature["method"].astype(str)
        ax.axhline(0, linewidth=1)
        ax.scatter(range(len(feature)), feature["delta_f1"])
        ax.set_xticks(range(len(feature)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("F1 difference, FS-20 minus FS-10")
        ax.set_title("Feature-set ablation")
        path = figure_dir / "research_fs20_vs_fs10.png"
        _save(fig, path)
        paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final FW-LNSA research figures.")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    paths = make_figures(Path(args.output_dir))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
