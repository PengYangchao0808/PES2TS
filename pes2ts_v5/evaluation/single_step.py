"""Single-step evaluation (V5.7-B T1): Move / Process / Trust metrics on a split.

This is the primary *offline prediction* metric set; it is explicitly **not** a
QC success or rollout-success metric.
"""

from __future__ import annotations

import numpy as np
import torch

from pes2ts_v5.evaluation._common import (
    binary_auc,
    masked_cosine,
    masked_rmse,
    to_device,
)


def _empty_metrics() -> dict:
    return {
        "n_batches": 0, "n_frames": 0, "n_move": 0, "n_process": 0,
        "n_progress": 0, "n_trust": 0,
        "move_rmse": float("nan"), "move_cosine": float("nan"),
        "process_acc": float("nan"), "process_auc": float("nan"),
        "progress_mae": float("nan"), "trust_auc": float("nan"),
        "mask_coverage": {},
    }


@torch.no_grad()
def evaluate_model_batches(
    model,
    batches,
    *,
    device="cpu",
) -> dict:
    """Accumulate single-step metrics over an iterator of ``(batch, meta)``."""
    model.eval()
    move_pred, move_tgt, move_mask = [], [], []
    proc_pred, proc_tgt, proc_mask = [], [], []
    prog_pred, prog_tgt, prog_mask = [], [], []
    trust_pred, trust_tgt, trust_mask = [], [], []
    n_frames = 0
    n_batches = 0
    for batch, _meta in batches:
        batch = to_device(batch, device)
        out = model(batch)
        valid = batch["frame_mask"]
        n_frames += int(valid.sum().item())
        n_batches += 1
        m = batch["move_mask"] & valid
        if m.any():
            move_pred.append(out["move"][m].cpu())
            move_tgt.append(batch["move_target"][m].cpu())
            move_mask.append(torch.ones(int(m.sum().item()), dtype=torch.bool))
        pm = batch["process_mask"] & valid
        if pm.any():
            proc_pred.append(torch.softmax(out["process_logits"], dim=-1)[pm].cpu())
            proc_tgt.append(batch["process_label"][pm].cpu())
            proc_mask.append(torch.ones(int(pm.sum().item()), dtype=torch.bool))
        gm = batch["progress_mask"] & valid
        if gm.any():
            prog_pred.append(out["process_progress"][gm].cpu())
            prog_tgt.append(batch["process_progress"][gm].cpu())
            prog_mask.append(torch.ones(int(gm.sum().item()), dtype=torch.bool))
        tm = batch["trust_mask"] & valid
        if tm.any():
            trust_pred.append(torch.sigmoid(out["trust_logit"])[tm].cpu())
            trust_tgt.append(batch["trust_label"][tm].cpu())
            trust_mask.append(torch.ones(int(tm.sum().item()), dtype=torch.bool))
    metrics = _empty_metrics()
    metrics["n_batches"] = n_batches
    metrics["n_frames"] = n_frames
    if move_pred:
        pred = torch.cat(move_pred)
        tgt = torch.cat(move_tgt)
        mask = torch.cat(move_mask)
        metrics["n_move"] = int(mask.numel())
        metrics["move_rmse"] = masked_rmse(pred, tgt, mask)
        metrics["move_cosine"] = masked_cosine(pred, tgt, mask)
    if proc_pred:
        prob = torch.cat(proc_pred)
        tgt = torch.cat(proc_tgt).numpy()
        metrics["n_process"] = int(tgt.shape[0])
        metrics["process_acc"] = float((prob.argmax(dim=-1).numpy() == tgt).mean())
        near = prob[:, 1].numpy()
        metrics["process_auc"] = binary_auc(near, (tgt == 1).astype(np.float64))
    if prog_pred:
        pred = torch.cat(prog_pred)
        tgt = torch.cat(prog_tgt)
        mask = torch.cat(prog_mask)
        metrics["n_progress"] = int(mask.numel())
        metrics["progress_mae"] = float(((pred - tgt).abs() * mask).sum().item()
                                        / max(int(mask.sum().item()), 1))
    if trust_pred:
        pred = torch.cat(trust_pred).numpy()
        tgt = torch.cat(trust_tgt).numpy()
        metrics["n_trust"] = int(tgt.shape[0])
        metrics["trust_auc"] = binary_auc(pred, (tgt >= 0.5).astype(np.float64))
    metrics["mask_coverage"] = {
        "move": metrics["n_move"] / max(n_frames, 1),
        "process": metrics["n_process"] / max(n_frames, 1),
        "trust": metrics["n_trust"] / max(n_frames, 1),
    }
    return metrics


def evaluate_single_step(
    model,
    dataset,
    split: str = "id",
    *,
    device="cpu",
    batch_size: int = 8,
    max_batches: int | None = None,
    fraction: float = 1.0,
    seed: int = 17,
) -> dict:
    batches = dataset.iter_batches(
        split, batch_size=batch_size, shuffle=False, seed=seed,
        fraction=fraction, max_batches=max_batches)
    metrics = evaluate_model_batches(model, batches, device=device)
    metrics["split"] = split
    return metrics
