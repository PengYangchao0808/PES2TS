"""M09 LTC — liquid time-constant network (SIMPLIFIED exponential-Euler cell)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class _LTCCell(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_tau = nn.Linear(2 * hidden_dim, hidden_dim)
        self.w_syn = nn.Linear(2 * hidden_dim, hidden_dim)
        self.w_out = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x_t, h, dt):
        z = torch.cat([x_t, h], dim=-1)
        tau = torch.nn.functional.softplus(self.w_tau(z)) + 0.1     # liquid time constant
        syn = torch.tanh(self.w_syn(z))                             # synaptic drive
        decay = torch.exp(-dt / tau)                                # stable exponential Euler
        h_new = decay * h + (1.0 - decay) * tau * syn
        return torch.tanh(self.w_out(torch.cat([h_new, x_t], dim=-1)) + h_new)


class LTCSmall(CoreBase):
    name = "M09"
    paradigm = "continuous-rnn"
    stateful = True
    note = "SIMPLIFIED liquid time-constant cell with exponential-Euler integration"

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.cells = nn.ModuleList([_LTCCell(hidden_dim) for _ in range(self.layers)])
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
