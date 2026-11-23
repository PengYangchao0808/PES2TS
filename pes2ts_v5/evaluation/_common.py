"""Shared evaluation helpers: device move, masked metrics, numpy AUC, windows."""

from __future__ import annotations

import numpy as np
import torch

from pes2ts_v5.data.trajectories import collate_samples


def to_device(batch: dict, device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    diff = (pred - target) ** 2
    diff = diff.mean(dim=-1) if diff.dim() == pred.dim() and pred.dim() > mask.dim() else diff
    return float(torch.sqrt((diff * mask).sum() / mask.sum().clamp(min=1)).item())


def masked_cosine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum().item() == 0:
        return float("nan")
    p = pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos = (p * t).sum(dim=-1)
    return float(((cos * mask).sum() / mask.sum().clamp(min=1)).item())


def binary_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    """Rank-based ROC-AUC without sklearn; returns NaN for single-class input."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    keep = np.isfinite(scores) & np.isfinite(targets)
    scores, targets = scores[keep], targets[keep]
    n_pos = int((targets > 0.5).sum())
    n_neg = int(targets.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.shape[0]:
        j = i
        while j + 1 < sorted_scores.shape[0] and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    pos_rank_sum = ranks[targets > 0.5].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def window_sample(dataset, traj_idx: int, end: int, indices: list[int]) -> dict:
    """Build one single-step sample from selected history indices (all <= end)."""
    seq = dataset.build_sequence(traj_idx)
    t = int(end)
    keep = sorted({int(i) for i in indices if 0 <= int(i) <= t})
    if not keep or keep[-1] != t:
        keep.append(t)
    sample = {
        "traj_idx": seq["traj_idx"],
        "trajectory_id": seq["trajectory_id"],
        "reaction_id": seq["reaction_id"],
        "reaction_row": seq["reaction_row"],
        "direction": seq["direction"],
        "static_feats": seq["static_feats"],
        "length": len(keep),
    }
    time_keys = ("frame_feats", "delta_s", "frame_mask", "q", "s", "move_target",
                 "move_mask", "process_label", "process_mask", "process_progress",
                 "progress_mask", "trust_label", "trust_mask")
    for key in time_keys:
        sample[key] = np.asarray(seq[key])[keep]
    return sample


def collate_window(samples: list[dict]) -> dict:
    return collate_samples(samples)
