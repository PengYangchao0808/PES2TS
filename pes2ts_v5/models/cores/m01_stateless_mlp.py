"""M01 Stateless MLP — no memory, single-frame lower bound (plan §3.1)."""

from __future__ import annotations

import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class StatelessMLP(CoreBase):
    name = "M01"
    paradigm = "feedforward"
    stateful = False
    note = "single-frame lower bound; no history"

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(int(layers)):
            blocks += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                       nn.Dropout(dropout)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x, lengths=None, delta_s=None, state=None):
        return self.net(x), None
