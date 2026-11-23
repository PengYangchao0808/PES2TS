"""M05 Time-aware GRU — explicit irregular path-interval input (plan §3.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class _TimeAwareGRUCell(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.wz = nn.Linear(2 * hidden_dim + 1, hidden_dim)
        self.wr = nn.Linear(2 * hidden_dim + 1, hidden_dim)
        self.wh = nn.Linear(2 * hidden_dim + 1, hidden_dim)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor,
                dt_feat: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x_t, h, dt_feat], dim=-1)
        z = torch.sigmoid(self.wz(cat))
        r = torch.sigmoid(self.wr(cat))
        n = torch.tanh(self.wh(torch.cat([x_t, r * h, dt_feat], dim=-1)))
        return (1.0 - z) * h + z * n


class TimeAwareGRU(CoreBase):
    name = "M05"
    paradigm = "continuous-rnn"
    stateful = True
    note = "custom GRU cell with log1p(delta_s) as explicit time feature"

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.cells = nn.ModuleList([_TimeAwareGRUCell(hidden_dim)
                                    for _ in range(self.layers)])
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
            hs = [h for h in state]
        if delta_s is None:
            dt = x.new_ones(b, t)
        else:
            dt = delta_s.to(x.dtype).clamp(min=0.0)
        dt_feat = torch.log1p(dt).unsqueeze(-1)  # [B,T,1]
        outs = []
        for ti in range(t):
            inp = x[:, ti]
            d = dt_feat[:, ti]
            for li, cell in enumerate(self.cells):
                hs[li] = cell(inp, hs[li], d)
                inp = self.drop(hs[li])
            outs.append(inp)
        return torch.stack(outs, dim=1), tuple(hs)
