"""M03 Causal TCN — dilated causal convolution, multi-scale local history."""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase, causal_left_pad


class CausalTCN(CoreBase):
    name = "M03"
    paradigm = "cnn"
    stateful = False
    note = "dilated causal conv stack (dilations 1,2,4,8 cycling)"

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        dilations = [1, 2, 4, 8]
        blocks = []
        for i in range(int(layers)):
            d = dilations[i % len(dilations)]
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=d),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            self.add_module(f"block{i}", blocks[-1])
        self.dilations = [dilations[i % len(dilations)] for i in range(int(layers))]
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, lengths=None, delta_s=None, state=None):
        z = x
        for i, d in enumerate(self.dilations):
            block = getattr(self, f"block{i}")
            zt = causal_left_pad(z, d).transpose(1, 2)
            z = z + block(zt).transpose(1, 2)
        return self.out(z), None
