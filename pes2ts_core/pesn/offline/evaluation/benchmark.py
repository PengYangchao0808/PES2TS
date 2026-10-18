"""Frozen-weight benchmark over exported teacher views (B6).

The benchmark loads frozen model weights (checkpoint or bundle) and computes
the research metrics on a declared evaluation view, together with the
same-input baselines (constant / rule / MLP when its frozen state exists).
There is no fit call, no early stop, and no training-mode forward here; the
sealed evaluation adds only a different reaction view and separate
permissions, never a different code path through the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.models.baselines import (
    ConstantProgressBaseline,
    RuleProgressBaseline,
)
from pes2ts_core.pesn.offline.evaluation.metrics import (
    bootstrap_ci,
    activity_metrics,
    direction_metrics,
    pair_concordance,
    progress_metrics,
)
from pes2ts_core.pesn.offline.training.loader import (
    ReactionData,
    build_pesn_input_for_frame,
)
from pes2ts_core.pesn.offline.training.losses import predict_batch_outputs
from pes2ts_core.pesn.features.graph import build_features

BENCHMARK_SCHEMA: Final[str] = "pesn_benchmark_report_v1"


class BenchmarkError(RuntimeError):
    """Typed benchmark failure (fit attempted / missing frozen weights)."""


def _collect_per_reaction(
    predictor_fn,
    reactions: Sequence[ReactionData],
    *,
    batch_reactions: int = 8,
) -> dict[str, dict[str, list[float]]]:
    per_reaction: dict[str, dict[str, list[float]]] = {}
    for start in range(0, len(reactions), batch_reactions):
        chunk = reactions[start : start + batch_reactions]
        samples = []
        for reaction in chunk:
            for frame in range(reaction.n_frames):
                if not reaction.s_mask[frame]:
                    continue
                prediction_input = build_pesn_input_for_frame(reaction, frame)
                samples.append((reaction, frame, build_features(prediction_input)))
        if not samples:
            continue
        outputs = predictor_fn([features for _, _, features in samples])
        s_hat = np.asarray(outputs.get("signed_progress", []), dtype=float)
        score = np.asarray(outputs.get("path_score", []), dtype=float)
        act_logits = outputs.get("activity_logits")
        act_probs = (
            1.0 / (1.0 + np.exp(-np.asarray(act_logits, dtype=float)))
            if act_logits is not None
            else None
        )
        u_flat = (
            np.asarray(outputs.get("direction_components"), dtype=float)
            if outputs.get("direction_components") is not None
            else None
        )
        driver_counts = outputs.get("driver_counts") or [0] * len(samples)
        start_d = 0
        for local_id, (reaction, frame, _) in enumerate(samples):
            store = per_reaction.setdefault(
                reaction.reaction_id,
                {"s": [], "s_hat": [], "d": [], "cos": [], "act_y": [], "act_p": [], "scores": []},
            )
            if local_id < s_hat.size:
                store["s"].append(float(reaction.s[frame]))
                store["s_hat"].append(float(s_hat[local_id]))
                store["d"].append(float(reaction.d[frame]))
                store["scores"].append(float(score[local_id]))
            count = int(driver_counts[local_id])
            end_d = start_d + count
            if u_flat is not None and reaction.dir_mask[frame] and count > 0:
                segment = u_flat[start_d:end_d]
                norm = float(np.linalg.norm(segment))
                if norm > 1e-6:
                    t_hat = segment / norm
                    store["cos"].append(
                        float(np.dot(t_hat, reaction.direction[frame]))
                    )
            if act_probs is not None and reaction.act_mask[frame] and count > 0:
                store["act_y"].extend(reaction.activity[frame].tolist())
                store["act_p"].extend(act_probs[start_d:end_d].tolist())
            start_d = end_d
    return per_reaction


def _full_metrics(per_reaction: Mapping[str, Mapping[str, list[float]]], margin: float) -> dict[str, object]:
    return {
        **progress_metrics(per_reaction),
        **direction_metrics(per_reaction),
        **activity_metrics(per_reaction),
        **pair_concordance(per_reaction, margin),
    }


def run_frozen_benchmark(
    model,
    reactions: Sequence[ReactionData],
    *,
    eval_name: str,
    pair_margin: float = 0.05,
    baseline_mlp=None,
) -> dict[str, object]:
    """Frozen evaluation of the model + same-input baselines on one view."""

    def model_fn(features_list):
        return predict_batch_outputs(model, features_list)

    model_view = _collect_per_reaction(model_fn, reactions)
    report: dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA,
        "eval_name": eval_name,
        "n_reactions": len(reactions),
        "model": _full_metrics(model_view, pair_margin),
        "baseline_constant": _full_metrics(
            _collect_per_reaction(
                lambda fl: ConstantProgressBaseline().predict_batch(fl), reactions
            ),
            pair_margin,
        ),
        "baseline_rule": _full_metrics(
            _collect_per_reaction(
                lambda fl: RuleProgressBaseline().predict_batch(fl), reactions
            ),
            pair_margin,
        ),
    }
    if baseline_mlp is not None:
        report["baseline_mlp"] = _full_metrics(
            _collect_per_reaction(
                lambda fl: baseline_mlp.predict_batch(fl), reactions
            ),
            pair_margin,
        )
    per_reaction_mae = [
        float(np.mean(np.abs(np.asarray(v["s_hat"]) - np.asarray(v["s"]))))
        for v in model_view.values()
        if v.get("s") and v.get("s_hat")
    ]
    report["progress_mae_bootstrap"] = bootstrap_ci(per_reaction_mae)
    return report


def write_benchmark_report(report: Mapping[str, object], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(report), indent=2))
    return target


__all__ = [
    "BENCHMARK_SCHEMA",
    "BenchmarkError",
    "run_frozen_benchmark",
    "write_benchmark_report",
]
