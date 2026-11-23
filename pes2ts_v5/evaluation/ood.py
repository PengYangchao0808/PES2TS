"""OOD evaluation (V5.7-B T5): ID / Scaffold-OOD / Family-OOD / Size-OOD.

Split semantics come from the dataset's ``splits.json`` (reaction-level Group
split, plan §2.4).  Degradations are reported relative to the ID split.
"""

from __future__ import annotations

import numpy as np

from pes2ts_v5.evaluation.single_step import evaluate_single_step

DEFAULT_SPLITS = ("id", "scaffold_ood", "family_ood", "size_ood")


def evaluate_ood(
    model,
    dataset,
    *,
    device="cpu",
    batch_size: int = 8,
    max_batches: int | None = None,
    split_metric: str = "move_rmse",
) -> dict:
    available = [s for s in DEFAULT_SPLITS if len(dataset.reaction_ids_for_split(s)) > 0]
    if not available:
        available = [s for s in ("valid", "id") if len(dataset.reaction_ids_for_split(s)) > 0]
    per_split = {}
    for split in available:
        per_split[split] = evaluate_single_step(
            model, dataset, split, device=device, batch_size=batch_size,
            max_batches=max_batches)
    base = per_split.get("id", {}).get(split_metric)
    degradation = {}
    for split, metrics in per_split.items():
        value = metrics.get(split_metric)
        if base and np.isfinite(base) and base > 0 and value is not None and np.isfinite(value):
            degradation[split] = float(value / base - 1.0)
        else:
            degradation[split] = float("nan")
    return {
        "splits_evaluated": available,
        "per_split": per_split,
        "degradation_vs_id": degradation,
        "metric": split_metric,
        "note": "Group-split OOD; no frame-level random split (plan §2.4)",
    }
