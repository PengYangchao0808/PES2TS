"""M06 LSTM — classic state memory control (torch.nn.LSTM)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class LSTMCore(CoreBase):
    name = "M06"
    paradigm = "rnn"
    stateful = True
    note = "torch.nn.LSTM; state = (h, c) each [layers, B, H]"

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=self.layers,
                           batch_first=True,
                           dropout=dropout if self.layers > 1 else 0.0)

    def init_state(self, batch_size: int, device=None):
        device = device or next(self.parameters()).device
        z = torch.zeros(self.layers, batch_size, self.hidden_dim, device=device)
        return (z, z.clone())

    def forward(self, x, lengths=None, delta_s=None, state=None):
        y, (h_n, c_n) = self.rnn(x, state)
        return y, (h_n, c_n)
