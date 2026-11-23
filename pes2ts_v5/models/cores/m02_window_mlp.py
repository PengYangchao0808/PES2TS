"""M02 Window-MLP — fixed causal window then MLP (plan §3.1)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase, causal_left_pad


class WindowMLP(CoreBase):
    name = "M02"
    paradigm = "feedforward"
    stateful = False
    note = "fixed causal window (conv kernel) + MLP; history feature test"

    def __init__(
        self,
        hidden_dim: int,
        layers: int = 2,
        window: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.window = int(window)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=self.window)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, lengths=None, delta_s=None, state=None):
        # [B,T,H] -> [B,H,T], causal conv over the previous `window` frames
        z = causal_left_pad(x, self.window - 1).transpose(1, 2)
        z = torch.nn.functional.gelu(self.conv(z)).transpose(1, 2)
        return x + self.mlp(z), None
