"""Reaction-level research metrics with bootstrap intervals (B6).

Primary indicators (G3-P1 §6.1): reaction-equal signed-progress MAE,
top-1/top-3 localization error in normalized arc length excluding the exact
TS anchor, same-reaction pair concordance, direction cosine + coverage, and
coordinate-activity PR-AUC/F1 + coverage.  Bootstrap confidence intervals use
the reaction group as the resampling unit — frames are never treated as
independent samples.  These metrics prove IRC teacher tasks only; they say
nothing about refinement success, off-path transfer, or QC cost.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

import numpy as np

DEFAULT_BOOTSTRAP_ITERATIONS: Final[int] = 1000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 42
LOCALIZATION_TOP_K: Final[int] = 3


def _reaction_stat(
    per_reaction: Mapping[str, Mapping[str, list[float]]],
    key: str,
) -> list[float]:
    return [
        float(np.mean(values[key]))
        for values in per_reaction.values()
        if values.get(key)
    ]


def progress_metrics(per_reaction: Mapping[str, Mapping[str, list[float]]]) -> dict[str, object]:
    """Reaction-equal MAE of s and d plus TS-anchor-excluded localization."""
    abs_s_errors: list[float] = []
    abs_d_errors: list[float] = []
    top1: list[float] = []
    top3: list[float] = []
    for values in per_reaction.values():
        s = np.asarray(values.get("s", []), dtype=float)
        s_hat = np.asarray(values.get("s_hat", []), dtype=float)
        d = np.asarray(values.get("d", []), dtype=float)
        if s.size and s_hat.size and s.size == s_hat.size:
            abs_s_errors.append(float(np.mean(np.abs(s_hat - s))))
            abs_d_errors.append(float(np.mean(np.abs(np.abs(s_hat) - np.abs(s)))))
        if s.size >= 2 and s_hat.size == s.size:
            nonzero = np.abs(s) > 1e-9
            if nonzero.any():
                order = np.argsort(np.abs(s_hat[nonzero]))
                sorted_d = np.abs(s[nonzero])[order]
                top1.append(float(sorted_d[0]))
                top3.append(float(np.min(sorted_d[:LOCALIZATION_TOP_K])))
    return {
        "n_reactions_s": len(abs_s_errors),
        "mae_s": float(np.mean(abs_s_errors)) if abs_s_errors else float("nan"),
        "mae_d": float(np.mean(abs_d_errors)) if abs_d_errors else float("nan"),
        "n_reactions_localization": len(top1),
        "localization_top1_mean_d": float(np.mean(top1)) if top1 else float("nan"),
        "localization_top3_mean_d": float(np.mean(top3)) if top3 else float("nan"),
    }


def pair_concordance(
    per_reaction: Mapping[str, Mapping[str, list[float]]],
    margin: float,
) -> dict[str, object]:
    """Fraction of same-reaction pairs where the predicted score agrees.

    A pair ``(i, j)`` with ``d_i + margin < d_j`` is concordant when
    ``score_i > score_j``; the rate is computed per reaction then averaged
    reaction-equal.
    """
    rates: list[float] = []
    n_pairs_total = 0
    for values in per_reaction.values():
        d = np.asarray(values.get("d", []), dtype=float)
        score = np.asarray(values.get("scores", []), dtype=float)
        if d.size < 2 or score.size != d.size:
            continue
        concordant = 0
        n_pairs = 0
        for i in range(d.size):
            for j in range(d.size):
                if i != j and d[i] + margin < d[j]:
                    n_pairs += 1
                    if score[i] > score[j]:
                        concordant += 1
        if n_pairs:
            rates.append(concordant / n_pairs)
            n_pairs_total += n_pairs
    return {
        "pair_concordance": float(np.mean(rates)) if rates else float("nan"),
        "n_pairs": n_pairs_total,
        "n_reactions_pairs": len(rates),
    }


def direction_metrics(per_reaction: Mapping[str, Mapping[str, list[float]]]) -> dict[str, object]:
    cosines: list[float] = []
    for values in per_reaction.values():
        cos = values.get("cos", [])
        if cos:
            cosines.append(float(np.mean(cos)))
    all_cos = [
        c
        for values in per_reaction.values()
        for c in values.get("cos", [])
    ]
    return {
        "direction_cosine_mean": float(np.mean(cosines)) if cosines else float("nan"),
        "direction_frame_cosine_mean": float(np.mean(all_cos)) if all_cos else float("nan"),
        "n_reactions_direction": len(cosines),
        "n_valid_frames": len(all_cos),
    }


def activity_metrics(per_reaction: Mapping[str, Mapping[str, list[float]]]) -> dict[str, object]:
    """PR-AUC (average precision), F1@0.5, and coverage for activity labels."""
    y_all: list[float] = []
    p_all: list[float] = []
    per_reaction_ap: list[float] = []
    for values in per_reaction.values():
        y = np.asarray(values.get("act_y", []), dtype=float)
        p = np.asarray(values.get("act_p", []), dtype=float)
        if y.size and p.size == y.size:
            y_all.extend(y.tolist())
            p_all.extend(p.tolist())
            if 0.0 < y.mean() < 1.0:
                per_reaction_ap.append(_average_precision(y, p))
    if not y_all:
        return {"activity_pr_auc": float("nan"), "activity_f1": float("nan"), "n_activity_labels": 0}
    y = np.asarray(y_all)
    p = np.asarray(p_all)
    pred = p >= 0.5
    tp = float(np.sum(pred & (y >= 0.5)))
    fp = float(np.sum(pred & (y < 0.5)))
    fn = float(np.sum(~pred & (y >= 0.5)))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    return {
        "activity_pr_auc": _average_precision(y, p),
        "activity_pr_auc_reaction_mean": (
            float(np.mean(per_reaction_ap)) if per_reaction_ap else float("nan")
        ),
        "activity_f1": f1,
        "n_activity_labels": len(y_all),
        "activity_positive_rate": float(np.mean(y >= 0.5)),
    }


def _average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Step-interpolated average precision (no sklearn dependency)."""
    order = np.argsort(-p)
    y_sorted = y[order] >= 0.5
    n_positive = float(y_sorted.sum())
    if n_positive == 0:
        return float("nan")
    cum_hits = np.cumsum(y_sorted)
    precision = cum_hits / (np.arange(len(y_sorted)) + 1)
    return float(np.sum(precision * y_sorted) / n_positive)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic=np.mean,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Percentile bootstrap over reaction-level statistic values."""
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if array.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    stats = np.asarray(
        [statistic(array[rng.integers(0, array.size, array.size)]) for _ in range(iterations)]
    )
    return {
        "mean": float(statistic(array)),
        "ci95_low": float(np.percentile(stats, 2.5)),
        "ci95_high": float(np.percentile(stats, 97.5)),
        "n": int(array.size),
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "activity_metrics",
    "bootstrap_ci",
    "direction_metrics",
    "pair_concordance",
    "progress_metrics",
]
