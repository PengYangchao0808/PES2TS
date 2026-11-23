"""Prediction heads (Move / Process / Trust) + masked multi-task losses.

Output heads (plan §1.2): Move is a controlled-reaction-coordinate displacement,
Process is path progress + TS-neighbourhood class, Trust is an **offline risk
proxy** (never an xTB success probability).  Missing labels are masked, never
filled with constants.

The individual masked terms (:func:`move_huber_loss`, :func:`process_ce_loss`,
:func:`progress_huber_loss`, :func:`trust_bce_loss`) are the single source of
truth used both by :func:`compute_losses` (legacy multi-task dict) and by the
named loss components in :mod:`pes2ts_v5.losses.components`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        q_dim: int = 3,
        process_classes: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        mid = max(8, hidden_dim // 2)
        self.move = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, q_dim),
        )
        self.process_cls = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.GELU(), nn.Linear(mid, process_classes))
        self.process_prog = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.GELU(), nn.Linear(mid, 1))
        self.trust = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.GELU(), nn.Linear(mid, 1))

    def forward(self, y: torch.Tensor) -> dict:
        return {
            "move": self.move(y),                                  # [B,T,q]
            "process_logits": self.process_cls(y),                  # [B,T,C]
            "process_progress": self.process_prog(y).squeeze(-1),   # [B,T]
            "trust_logit": self.trust(y).squeeze(-1),               # [B,T]
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over mask; returns a graph-connected zero when the mask is empty."""
    m = mask.to(values.dtype)
    denom = m.sum().clamp(min=1.0)
    return (values * m).sum() / denom


def _valid_mask(batch: dict, valid_mask: torch.Tensor | None) -> torch.Tensor:
    return valid_mask if valid_mask is not None else batch["frame_mask"]


def move_huber_loss(out: dict, batch: dict, *,
                    valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked Smooth-L1 on the reaction-coordinate displacement."""
    mask = batch["move_mask"] & _valid_mask(batch, valid_mask)
    diff = F.smooth_l1_loss(out["move"], batch["move_target"], reduction="none")
    return _masked_mean(diff.mean(dim=-1), mask)


def process_ce_loss(out: dict, batch: dict, *,
                    valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked cross-entropy on the path-stage class."""
    mask = batch["process_mask"] & _valid_mask(batch, valid_mask)
    if not mask.any():
        return out["process_logits"].sum() * 0.0
    ce = F.cross_entropy(
        out["process_logits"].reshape(-1, out["process_logits"].shape[-1]),
        batch["process_label"].reshape(-1).clamp(min=0),
        reduction="none").reshape(mask.shape)
    return _masked_mean(ce, mask)


def progress_huber_loss(out: dict, batch: dict, *,
                        valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked Smooth-L1 on normalised path progress."""
    mask = batch["progress_mask"] & _valid_mask(batch, valid_mask)
    diff = F.smooth_l1_loss(out["process_progress"], batch["process_progress"],
                            reduction="none")
    return _masked_mean(diff, mask)


def trust_bce_loss(out: dict, batch: dict, *,
                   valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Masked BCE-with-logits on the offline trust/risk proxy label."""
    mask = batch["trust_mask"] & _valid_mask(batch, valid_mask)
    bce = F.binary_cross_entropy_with_logits(
        out["trust_logit"], batch["trust_label"].clamp(0.0, 1.0), reduction="none")
    return _masked_mean(bce, mask)


def compute_losses(
    out: dict,
    batch: dict,
    *,
    weights: dict | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict:
    """Masked multi-task loss.  ``valid_mask`` defaults to ``batch['frame_mask']``."""
    weights = {"move": 1.0, "process": 0.5, "progress": 0.2, "trust": 0.2,
               **(weights or {})}
    valid = _valid_mask(batch, valid_mask)
    move_mask = batch["move_mask"] & valid
    process_mask = batch["process_mask"] & valid
    progress_mask = batch["progress_mask"] & valid
    trust_mask = batch["trust_mask"] & valid

    move_loss = move_huber_loss(out, batch, valid_mask=valid)
    process_loss = process_ce_loss(out, batch, valid_mask=valid)
    progress_loss = progress_huber_loss(out, batch, valid_mask=valid)
    trust_loss = trust_bce_loss(out, batch, valid_mask=valid)

    total = (weights["move"] * move_loss
             + weights["process"] * process_loss
             + weights["progress"] * progress_loss
             + weights["trust"] * trust_loss)
    return {
        "total": total,
        "move": move_loss,
        "process": process_loss,
        "progress": progress_loss,
        "trust": trust_loss,
        "n_move": int(move_mask.sum().item()),
        "n_process": int(process_mask.sum().item()),
        "n_progress": int(progress_mask.sum().item()),
        "n_trust": int(trust_mask.sum().item()),
    }


__all__ = [
    "MultiHead",
    "compute_losses",
    "move_huber_loss",
    "process_ce_loss",
    "progress_huber_loss",
    "trust_bce_loss",
]
