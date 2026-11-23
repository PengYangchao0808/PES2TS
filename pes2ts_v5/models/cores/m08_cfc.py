"""M08 CfC — closed-form continuous-time cell (SIMPLIFIED implementation).

Closed-form gated leaky integration driven by the observed path interval
``delta_s`` (treated as an irregular time step).  This preserves the CfC
inductive bias (continuous-time, input-dependent time constant) without the
official mixed closed-form / ODE solver; plan §3.4 requires CfC to be compared
against the **Time-aware GRU** (M05), which shares the same available time
information.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class _CfCCell(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.w_tau = nn.Linear(2 * hidden_dim, hidden_dim)
        self.w_out = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x_t, h, dt):
        z = torch.cat([x_t, h], dim=-1)
        tau = torch.sigmoid(self.w_tau(z)) + 1e-3          # per-unit time constant
        decay = torch.exp(-dt * tau)                        # closed-form decay
        h_target = torch.tanh(self.w_out(z))
        return decay * h + (1.0 - decay) * h_target


class CfCSmall(CoreBase):
    name = "M08"
    paradigm = "continuous-rnn"
    stateful = True
    note = "SIMPLIFIED closed-form continuous-time cell (exponential gate on delta_s)"

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.cells = nn.ModuleList([_CfCCell(hidden_dim) for _ in range(self.layers)])
        self.drop = nn.Dropout(dropout)

    def init_state(self, batch_size: int, device=None):
        device = device or next(self.parameters()).device
        return tuple(torch.zeros(batch_size, self.hidden_dim, device=device)
                     for _ in range(self.layers))

    def forward(self, x, lengths=None, delta_s=None, state=None):
        b, t, _ = x.shape
        if state is None:
            hs = [x.new_zeros(b, self.hidden_dim) for _ in range(self.layers)]
        else:
            hs = list(state)
        if delta_s is None:
            dt = x.new_ones(b, t)
        else:
            dt = delta_s.to(x.dtype).clamp(min=0.0)
        outs = []
        for ti in range(t):
            inp = x[:, ti]
            for li, cell in enumerate(self.cells):
                hs[li] = cell(inp, hs[li], dt[:, ti:ti + 1])
                inp = self.drop(hs[li])
            outs.append(inp)
        return torch.stack(outs, dim=1), tuple(hs)
