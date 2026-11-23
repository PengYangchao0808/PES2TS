"""Default encoder: low-dimensional reaction descriptor (plan §3.3 / §7).

The V5.6 primary screen intentionally uses cheap chemistry descriptors rather
than a full 3D graph encoder; StarEGNN v2 is kept as an optional stub for the
V5.7-A Encoder ablation (see :mod:`pes2ts_v5.models.encoders.stregnn_v2`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DescriptorEncoder(nn.Module):
    """Per-frame descriptor projection + reaction-level static conditioning.

    Input : ``frame_feats [B, T, F]``, ``static_feats [B, S]``
    Output: ``h [B, T, hidden_dim]`` (causal: per-frame + static only)
    """

    def __init__(
        self,
        frame_feat_dim: int,
        static_feat_dim: int,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.frame_feat_dim = int(frame_feat_dim)
        self.static_feat_dim = int(static_feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.frame_proj = nn.Sequential(
            nn.Linear(frame_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(static_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, frame_feats: torch.Tensor, static_feats: torch.Tensor) -> torch.Tensor:
        x = self.frame_proj(frame_feats)
        s = self.static_proj(static_feats).unsqueeze(1)
        return self.norm(self.drop(x + s))
