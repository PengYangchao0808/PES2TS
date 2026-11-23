"""Irregular-sampling stress test (V5.7-B T3): regular / sparse / random gaps /
missing history.

Each condition rebuilds the same single-step windows from a trajectory with a
different history sampling pattern; the path-interval ``delta_s`` of merged
steps is summed so information stays physically consistent.
"""

from __future__ import annotations

import numpy as np

from pes2ts_v5.evaluation._common import collate_window, to_device, window_sample
from pes2ts_v5.evaluation.single_step import evaluate_model_batches

CONDITIONS = ("regular", "sparse", "random_gaps", "missing_history")


def _window_indices(condition: str, t: int, history_len: int, rng: np.random.Generator,
                    gap_fraction: float = 0.3, keep_missing: int = 3) -> list[int]:
    start = max(0, t - history_len + 1)
    full = list(range(start, t + 1))
    if condition == "regular":
        return full
    if condition == "sparse":
        picked = [i for i in full if (t - i) % 2 == 0]
        return sorted(set(picked) | {t})
    if condition == "random_gaps":
        picked = [i for i in full if i == t or rng.random() > gap_fraction]
        return sorted(set(picked) | {t})
    if condition == "missing_history":
        return sorted({t, *full[-keep_missing:]})
    raise ValueError(f"unknown condition {condition!r}")


def _merge_delta_s(sample: dict, indices: list[int], t: int) -> None:
    """Sum delta_s over dropped intervals for retained frames (physical consistency)."""
    seq_ds = sample["delta_s"]
    if len(indices) < 2:
        return
    merged = []
    for pos, idx in enumerate(indices):
        nxt = indices[pos + 1] if pos + 1 < len(indices) else idx
        lo, hi = sorted((idx, nxt))
        merged.append(float(seq_ds[lo:hi + 1].sum()) if idx != nxt else float(seq_ds[idx]))
    sample["delta_s"] = np.asarray(merged, dtype=np.float32)
    sample["delta_s"][0] = 0.0


def evaluate_irregular_sampling(
    model,
    dataset,
    split: str = "id",
    *,
    device="cpu",
    max_trajectories: int = 12,
    history_len: int = 8,
    seed: int = 17,
    gap_fraction: float = 0.3,
) -> dict:
    idx = dataset.trajectory_indices(split, seed=seed)[:max_trajectories]
    rng = np.random.default_rng(seed)
    results: dict[str, dict] = {}
    for condition in CONDITIONS:
        samples = []
        for traj_idx in idx:
            seq = dataset.build_sequence(traj_idx)
            t = max(1, seq["length"] - 1)
            keep = _window_indices(condition, t, history_len, rng, gap_fraction)
            sample = window_sample(dataset, traj_idx, t, keep)
            _merge_delta_s(sample, keep, t)
            samples.append(sample)
        if not samples:
            results[condition] = {"move_rmse": float("nan"), "move_cosine": float("nan")}
            continue
        batch = collate_window(samples)
        batch = to_device(batch, device)
        # single-step metrics only over the final valid frame
        for key in ("frame_mask", "move_mask", "process_mask", "progress_mask", "trust_mask"):
            mask = batch[key].clone()
            only_last = torch.zeros_like(mask)
            only_last[:, -1] = mask[:, -1]
            batch[key] = only_last
        metrics = evaluate_model_batches(model, [(batch, [])], device=device)
        results[condition] = {
            "move_rmse": metrics["move_rmse"],
            "move_cosine": metrics["move_cosine"],
            "process_acc": metrics["process_acc"],
            "n": metrics["n_move"],
        }
    base = results.get("regular", {}).get("move_rmse")
    degradation = {}
    if base and np.isfinite(base) and base > 0:
        for condition in CONDITIONS:
            value = results[condition]["move_rmse"]
            degradation[condition] = float(value / base - 1.0) if np.isfinite(value) else float("nan")
    else:
        degradation = {c: float("nan") for c in CONDITIONS}
    return {
        "split": split,
        "history_len": history_len,
        "conditions": results,
        "error_worsening_ratio": degradation,
        "n_trajectories": len(idx),
    }


# torch imported lazily below to keep module docstring ordering readable
import torch  # noqa: E402  (needed for mask ops above)
