"""Descriptive representation audits and training-only feature robustness."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split

from ..feature_selection import compute_mutual_information_scores
from .io import binary_matrix


def pattern_keys(X):
    return [bytes(row) for row in np.packbits(binary_matrix(X), axis=1)]


def binary_diagnostics(partitions, features):
    summaries, bits, overlaps = [], [], []
    sets = {}
    train_set = set(pattern_keys(partitions["train"][0]))
    for name, (X, y) in partitions.items():
        X = binary_matrix(X)
        y = np.asarray(y)
        if len(X) != len(y) or not np.all(np.isin(y, [0, 1])):
            raise ValueError("Invalid aligned binary labels")
        keys = pattern_keys(X)
        counts = pd.Series(keys).value_counts()
        sets[name] = set(keys)
        benign = set(k for k, v in zip(keys, y) if v == 0)
        attack = set(k for k, v in zip(keys, y) if v == 1)
        prevalence = X.mean(axis=0)
        summaries.append(
            {
                "partition": name,
                "records": len(X),
                "unique_patterns": len(counts),
                "largest_pattern_count": int(counts.max()),
                "unique_fraction": len(counts) / len(X),
                "constant_bits": int(((prevalence == 0) | (prevalence == 1)).sum()),
                "rare_bits_at_1pct": int(
                    ((prevalence <= 0.01) | (prevalence >= 0.99)).sum()
                ),
                "benign_attack_shared_patterns": len(benign & attack),
                "records_unseen_in_train": sum(k not in train_set for k in keys),
                "descriptive_only": True,
            }
        )
        for j, feature in enumerate(features):
            row = {"partition": name, "feature": feature, "prevalence": prevalence[j]}
            for label in [0, 1]:
                row[f"class_{label}_support"] = int((y == label).sum())
                row[f"class_{label}_prevalence"] = (
                    float(X[y == label, j].mean()) if (y == label).any() else np.nan
                )
            bits.append(row)
    for a, b in combinations(sets, 2):
        overlaps.append(
            {
                "partition_a": a,
                "partition_b": b,
                "shared_patterns": len(sets[a] & sets[b]),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(bits), pd.DataFrame(overlaps)


def sample_training(X, y, seed, fraction=0.8):
    """Fixed-size stratified sample, without replacement; no held-out records."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    if np.min(counts) < 2:
        raise ValueError(
            "Each class needs at least two records for stratified resampling"
        )
    indices, _ = train_test_split(
        np.arange(len(y)), train_size=fraction, stratify=y, random_state=seed
    )
    return X.iloc[indices].reset_index(drop=True), y[indices]


def feature_stability(X, y, *, seeds, sizes=(10, 20), rankings=None):
    scores = [] if rankings is None else rankings
    for seed in seeds if rankings is None else []:
        sub, labels = sample_training(X, y, seed)
        ranking = compute_mutual_information_scores(sub, labels, random_seed=seed)
        # Secondary tie rule, distinct from historical pandas sort: lexical.
        ranking = ranking.sort_values(["mi_score", "feature"], ascending=[False, True])
        ranking["rank"] = np.arange(1, len(ranking) + 1)
        ranking["resample_seed"] = seed
        scores.append(ranking)
    long = pd.concat(scores, ignore_index=True)
    freq = []
    for size in sizes:
        for feature, group in long.groupby("feature"):
            freq.append(
                {
                    "k": size,
                    "feature": feature,
                    "selected_count": int((group["rank"] <= size).sum()),
                    "resamples": len(seeds),
                    "selection_frequency": float((group["rank"] <= size).mean()),
                    "mi_mean": group.mi_score.mean(),
                    "mi_std": group.mi_score.std(),
                    "mi_min": group.mi_score.min(),
                    "mi_max": group.mi_score.max(),
                }
            )
    pairs = []
    for a, b in combinations(seeds, 2):
        x = long[long.resample_seed == a].set_index("feature")
        z = long[long.resample_seed == b].set_index("feature").reindex(x.index)
        for size in sizes:
            sa = set(x.index[x["rank"] <= size])
            sb = set(z.index[z["rank"] <= size])
            pairs.append(
                {
                    "seed_a": a,
                    "seed_b": b,
                    "k": size,
                    "jaccard": len(sa & sb) / len(sa | sb),
                    "spearman_full_ranking": spearmanr(x["rank"], z["rank"]).statistic,
                }
            )
    return long, pd.DataFrame(freq), pd.DataFrame(pairs)


def mrmr_ranking(X, y, *, bins=10, seed=4242):
    """MID (Peng-style relevance minus mean redundancy), all MI in nats.

    Each numeric feature is discretized by training quantiles (<=10 bins;
    duplicate boundaries collapse); binary indicators remain discrete. Both
    I(X_j;Y) and I(X_j;X_s) use sklearn mutual_info_score plug-in contingency
    estimators on identical discretized variables. Thus unlike mixing Pearson
    correlation with continuous MI, subtraction has common units. Lexical ties.
    """
    del seed  # estimator is deterministic, recorded API seed has no effect
    columns = {}
    for name in sorted(X.columns):
        values = X[name].to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        columns[name] = np.searchsorted(edges, values, side="right")
    disc = pd.DataFrame(columns, index=X.index)
    names = list(disc.columns)
    relevance = {name: mutual_info_score(disc[name], y) for name in names}
    redundancy = pd.DataFrame(0.0, index=names, columns=names)
    for a, b in combinations(names, 2):
        redundancy.loc[a, b] = redundancy.loc[b, a] = mutual_info_score(
            disc[a], disc[b]
        )
    selected, rows = [], []
    while len(selected) < len(names):
        remaining = [n for n in names if n not in selected]
        red = {
            n: float(redundancy.loc[n, selected].mean()) if selected else 0.0
            for n in remaining
        }
        best = sorted(remaining, key=lambda n: (-(relevance[n] - red[n]), n))[0]
        rows.append(
            {
                "feature": best,
                "rank": len(selected) + 1,
                "relevance_nats": relevance[best],
                "mean_redundancy_nats": red[best],
                "mid_objective": relevance[best] - red[best],
            }
        )
        selected.append(best)
    return pd.DataFrame(rows), redundancy


def scores_diagnostic(scores, y, threshold, method):
    scores = np.asarray(scores)
    finite = scores[np.isfinite(scores)]
    unique, counts = np.unique(finite, return_counts=True)
    return {
        "unique_finite_scores": len(unique),
        "nonfinite_scores": int((~np.isfinite(scores)).sum()),
        "max_score_tie_count": int(counts.max()) if len(counts) else 0,
        "records_at_threshold": int((scores == threshold).sum()),
        "benign_at_threshold": int(((scores == threshold) & (y == 0)).sum()),
        "minimum_score_step": float(np.diff(unique).min())
        if len(unique) > 1
        else np.nan,
        "threshold": threshold,
        "method": method,
    }


def category_recalls(y, predictions, original_labels, category_labels, universe):
    rows = []
    original_labels = np.asarray(original_labels)
    category_labels = np.asarray(category_labels)
    for category in sorted(universe):
        mask = (category_labels == category) & (np.asarray(y) == 1)
        support = int(mask.sum())
        hits = int(np.asarray(predictions)[mask].sum())
        rows.append(
            {
                "category": category,
                "support": support,
                "detected": hits,
                "recall": hits / support if support else np.nan,
                "status": "ESTIMABLE" if support else "NOT_ESTIMABLE",
                "original_labels_present": "|".join(sorted(set(original_labels[mask]))),
            }
        )
    return rows


def score_unique_patterns(model, X):
    """Reuse deterministic scores for duplicate bit rows, then restore row order.

    Diagnostic computation only; never used for resource benchmarks or main
    model inference. The original per-pattern scorer and feature summation
    remain unchanged. Golden confusion/threshold checks gate real-data replay.
    """
    X = binary_matrix(X)
    unique, inverse = np.unique(X, axis=0, return_inverse=True)
    return model.decision_scores(unique)[inverse]
