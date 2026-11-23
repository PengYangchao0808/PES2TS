"""M13 S4D — structured diagonal state-space layer (SIMPLIFIED real-diagonal).

Simplification: the S4D-Lin complex diagonal kernel is represented with a
**real negative diagonal** and ZOH-style discretization (no complex rotation).
It keeps the linear-state-dynamics inductive bias (plan §3.1) without external
`mamba_ssm`/`s4` packages.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class _S4DLayer(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        # real negative diagonal A = -k, k = 1..N; strictly negative so the ZOH
        # input map (a_bar - 1) / a is well defined (no division by zero)
        idx = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.log_A = nn.Parameter(-idx.unsqueeze(0).expand(
            hidden_dim, state_dim).clone())
        self.B = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.02)
        self.log_dt = nn.Parameter(torch.zeros(hidden_dim))
        self.d_skip = nn.Parameter(torch.ones(hidden_dim))
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        b, t, h = x.shape
        u = self.in_proj(x)
        a = self.log_A
        dt = torch.exp(self.log_dt)                     # [H]
        a_bar = torch.exp(dt.unsqueeze(-1) * a)         # [H,S]
        b_bar = (a_bar - 1.0) / a                       # ZOH input matrix [H,S]
        state = u.new_zeros(b, h, self.state_dim)
        ys = []
        for ti in range(t):
            state = a_bar.unsqueeze(0) * state + b_bar.unsqueeze(0) * u[:, ti].unsqueeze(-1)
            ys.append((state * self.C.unsqueeze(0)).sum(dim=-1) + self.d_skip * u[:, ti])
        return self.out(torch.stack(ys, dim=1))


class S4D(CoreBase):
    name = "M13"
    paradigm = "ssm"
    stateful = False
    note = "SIMPLIFIED S4D: real-diagonal linear state dynamics, ZOH discretization"

    def __init__(self, hidden_dim: int, layers: int = 2, state_dim: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.state_dim = int(state_dim)
        self.blocks = nn.ModuleList([_S4DLayer(hidden_dim, self.state_dim)
                                     for _ in range(self.layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths=None, delta_s=None, state=None):
        z = x
        for block in self.blocks:
            z = z + self.drop(block(z))
        return z, None
