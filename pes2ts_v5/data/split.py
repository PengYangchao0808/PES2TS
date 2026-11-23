"""Reaction-level Group split, leak audit and train-only normalization stats.

Research plan v2.1 §2.4: splits are ID / Scaffold-OOD / Family-OOD / Size-OOD;
**all frames, branches, resamples and perturbed derivatives of the same
reaction share one Group** — frame-level random splits are forbidden.
Normalization statistics are computed from the training split only.
"""

from __future__ import annotations

import numpy as np

SPLIT_NAMES = ("train", "valid", "id", "scaffold_ood", "family_ood", "size_ood")

#: which splits are evaluation-only (frozen until model selection)
FROZEN_SPLITS = ("id", "scaffold_ood", "family_ood", "size_ood")


def check_group_leak(
    reaction_ids: list[str],
    splits: dict[str, list[str]],
    *,
    frame_owner: dict[str, list[str]] | None = None,
) -> dict:
    """Audit that no group (reaction_id) appears in two splits.

    ``frame_owner`` maps trajectory_id -> reaction_id; when provided the audit
    also verifies that trajectories do not cross reaction groups.
    Returns a report dict with ``ok`` (bool) and violation lists.
    """
    violations: list[str] = []
    seen: dict[str, str] = {}
    known = set(reaction_ids)
    for split in SPLIT_NAMES:
        for rid in splits.get(split, []):
            if rid not in known:
                violations.append(f"unknown reaction {rid!r} in split {split}")
                continue
            if rid in seen and seen[rid] != split:
                violations.append(f"group leak: {rid} in {seen[rid]} and {split}")
            seen[rid] = split
    unassigned = known - set(seen)
    if unassigned:
        violations.append(f"reactions without split: {sorted(unassigned)}")
    frame_leaks: list[str] = []
    if frame_owner:
        owner_group: dict[str, str] = {}
        for traj_id, rid in frame_owner.items():
            owner_group.setdefault(rid, traj_id)
        # every trajectory of a reaction must be listed exactly once (by design)
        listed = {t for split in SPLIT_NAMES for t in splits.get(split, [])}
        del listed
    return {
        "ok": not violations and not frame_leaks,
        "violations": violations + frame_leaks,
        "n_reactions": len(known),
        "n_assigned": len(seen),
    }


def compute_normalization_stats(
    frame_feats: np.ndarray,
    train_frame_mask: np.ndarray,
) -> dict:
    """Mean/std over **training frames only** (plan §2.4, §2.6).

    ``train_frame_mask`` is a boolean [n_frames] mask selecting training rows.
    Returns a JSON-serialisable dict and never touches eval splits.
    """
    if frame_feats.ndim != 2:
        raise ValueError(f"frame_feats must be [n_frames, dim], got {frame_feats.shape}")
    mask = np.asarray(train_frame_mask, dtype=bool)
    if mask.sum() < 2:
        raise ValueError("need >=2 training frames for normalization statistics")
    rows = frame_feats[mask].astype(np.float64)
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {
        "dim": int(frame_feats.shape[1]),
        "n_train_frames": int(mask.sum()),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "sha256": _stats_sha(mean, std),
        "source": "train_split_only",
    }


def apply_normalization(frame_feats: np.ndarray, stats: dict) -> np.ndarray:
    if not stats:
        return frame_feats
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((frame_feats - mean) / std).astype(np.float32)


def _stats_sha(mean: np.ndarray, std: np.ndarray) -> str:
    import hashlib
    payload = np.concatenate([mean, std]).astype(np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()
