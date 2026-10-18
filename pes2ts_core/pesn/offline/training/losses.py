"""Masked multi-task teacher loss (B4, G3-P1 §5.2).

``L = λs·Huber(ŝ, s) + λd·Huber(d̂, d) + λr·softplus[-(f_i − f_j)/τ]
   + λt·(1 − t̂·t*) + λa·BCE(ẑ, z) + λsym·Huber(−ŝ_swap, s)``

Every term averages over its *valid* samples only; missing labels never
back-propagate zeros.  ``d̂ = |ŝ|`` structurally (consistency is a
diagnostic, not a second task).  Direction components are normalized per
sample with a validity check (near-zero norms are excluded, not squashed).
Pairs are same-reaction with ``d_i + margin < d_j``.  A batch whose tasks are
all missing is reported as a typed skip by the trainer, never silently
zero-lossed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

from pes2ts_core.pesn.offline.training.loader import TrainBatch

DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "s": 1.0,
    "d": 0.5,
    "rank": 1.0,
    "direction": 0.5,
    "activity": 0.2,
    "symmetry": 0.1,
}
DEFAULT_TAU: Final[float] = 1.0
DEFAULT_PAIR_MARGIN: Final[float] = 0.05
HUBER_DELTA: Final[float] = 1.0
DIRECTION_MIN_NORM: Final[float] = 1e-6


@dataclass(slots=True)
class LossParts:
    total: object
    s: object
    d: object
    rank: object
    direction: object
    activity: object
    symmetry: object
    n_valid_s: int
    n_valid_dir: int
    n_valid_act: int
    n_pairs: int


class LossError(RuntimeError):
    """Typed loss-computation failure (e.g., an entirely empty batch)."""


def compute_teacher_loss(
    model,
    batch: TrainBatch,
    *,
    weights: dict[str, float] = None,
    tau: float = DEFAULT_TAU,
    apply_symmetry: bool = True,
) -> LossParts:
    """Forward + masked multi-task loss for one training batch."""
    from pes2ts_core.pesn.features.graph import swap_task_graph_features

    torch = model.torch
    weights = weights or dict(DEFAULT_WEIGHTS)
    samples = batch.samples
    if not samples:
        raise LossError("empty batch")

    outputs = model.forward([sample.features for sample in samples])
    s_hat = outputs["signed_progress"]
    score = outputs["path_score"]

    s_target = torch.as_tensor(
        [sample.s for sample in samples], dtype=torch.float32
    )
    valid = torch.as_tensor([sample.valid for sample in samples], dtype=torch.bool)

    if valid.any():
        huber = torch.nn.functional.huber_loss(
            s_hat[valid], s_target[valid], delta=HUBER_DELTA
        )
        loss_s = huber
        loss_d = torch.nn.functional.huber_loss(
            torch.abs(s_hat[valid]), torch.abs(s_target[valid]), delta=HUBER_DELTA
        )
    else:
        loss_s = s_hat.sum() * 0.0
        loss_d = loss_s

    if batch.pairs:
        pair_i = torch.as_tensor([i for i, _ in batch.pairs], dtype=torch.long)
        pair_j = torch.as_tensor([j for _, j in batch.pairs], dtype=torch.long)
        loss_rank = torch.nn.functional.softplus(
            -((score[pair_i] - score[pair_j]) / tau)
        ).mean()
    else:
        loss_rank = score.sum() * 0.0

    u_flat = outputs["direction_components"]
    driver_graph_id = outputs["driver_graph_id"]
    driver_counts = outputs["driver_counts"]
    n_graphs = len(samples)
    unit_flat = torch.zeros_like(u_flat)
    dir_valid_flat = torch.zeros_like(u_flat, dtype=torch.bool)
    dir_target_flat = torch.zeros_like(u_flat)
    act_target_flat = torch.zeros_like(u_flat)
    act_valid_flat = torch.zeros_like(u_flat, dtype=torch.bool)
    start = 0
    for graph_id in range(n_graphs):
        count = int(driver_counts[graph_id].item())
        end = start + count
        if count == 0:
            continue
        segment = u_flat[start:end]
        norm = segment.norm()
        sample = samples[graph_id]
        if sample.dir_valid and float(norm.item()) > DIRECTION_MIN_NORM:
            unit_flat[start:end] = segment / norm
            dir_valid_flat[start:end] = True
        if sample.dir_valid:
            dir_target_flat[start:end] = torch.as_tensor(
                sample.direction, dtype=torch.float32
            )
        if sample.act_valid:
            act_target_flat[start:end] = torch.as_tensor(
                sample.activity, dtype=torch.float32
            )
            act_valid_flat[start:end] = True
        start = end

    if dir_valid_flat.any():
        # Per-sample full cosine: for each graph segment the unit prediction
        # is dotted with the unit target, and the loss averages
        # (1 - cos_theta) over VALID SAMPLES.  A parallel vector therefore
        # has exactly zero loss regardless of the coordinate count; the
        # earlier flat elementwise mean scaled with the segment length.
        cos_per_sample = unit_flat * dir_target_flat
        sample_losses = []
        start = 0
        for graph_id in range(n_graphs):
            count = int(driver_counts[graph_id].item())
            end = start + count
            segment_valid = dir_valid_flat[start:end]
            if count and segment_valid.any():
                cos_theta = cos_per_sample[start:end][segment_valid].sum()
                sample_losses.append(1.0 - cos_theta)
            start = end
        loss_direction = (
            torch.stack(sample_losses).mean() if sample_losses
            else u_flat.sum() * 0.0
        )
    else:
        loss_direction = u_flat.sum() * 0.0

    if act_valid_flat.any():
        loss_activity = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["activity_logits"][act_valid_flat],
            act_target_flat[act_valid_flat],
        )
    else:
        loss_activity = u_flat.sum() * 0.0

    loss_symmetry = s_hat.sum() * 0.0
    if apply_symmetry and weights.get("symmetry", 0.0) > 0 and valid.any():
        swapped = [
            swap_task_graph_features(sample.features) for sample in samples
        ]
        swapped_out = model.forward(swapped)
        loss_symmetry = torch.nn.functional.huber_loss(
            -swapped_out["signed_progress"][valid],
            s_target[valid],
            delta=HUBER_DELTA,
        )

    total = (
        weights["s"] * loss_s
        + weights["d"] * loss_d
        + weights["rank"] * loss_rank
        + weights["direction"] * loss_direction
        + weights["activity"] * loss_activity
        + weights["symmetry"] * loss_symmetry
    )
    return LossParts(
        total=total,
        s=loss_s,
        d=loss_d,
        rank=loss_rank,
        direction=loss_direction,
        activity=loss_activity,
        symmetry=loss_symmetry,
        n_valid_s=int(valid.sum().item()),
        n_valid_dir=int(dir_valid_flat.sum().item()),
        n_valid_act=int(act_valid_flat.sum().item()),
        n_pairs=len(batch.pairs),
    )


def predict_batch_outputs(model, features_list: Sequence) -> dict[str, object]:
    """No-grad forward used by evaluation (numpy outputs, masks separate)."""
    torch = model.torch
    model.train_mode(False)
    samples_outputs = model.forward(list(features_list))
    n_graphs = len(features_list)
    driver_counts = samples_outputs["driver_counts"]
    with torch.no_grad():
        s_hat = samples_outputs["signed_progress"].cpu().numpy()
        score = samples_outputs["path_score"].cpu().numpy()
        u = samples_outputs["direction_components"].cpu().numpy()
        act_logits = samples_outputs["activity_logits"].cpu().numpy()
    counts = [int(driver_counts[g].item()) for g in range(n_graphs)]
    return {
        "signed_progress": s_hat,
        "path_score": score,
        "direction_components": u,
        "activity_logits": act_logits,
        "driver_counts": counts,
    }


__all__ = [
    "DEFAULT_PAIR_MARGIN",
    "DEFAULT_TAU",
    "DEFAULT_WEIGHTS",
    "LossError",
    "LossParts",
    "compute_teacher_loss",
    "predict_batch_outputs",
]
