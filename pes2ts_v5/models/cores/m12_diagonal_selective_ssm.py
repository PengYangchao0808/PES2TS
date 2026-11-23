"""M12 Diagonal Selective SSM — Mamba-lite diagonal scan (SIMPLIFIED)."""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase, causal_left_pad


class _DiagonalSelectiveLayer(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2,
                              groups=hidden_dim)
        self.dt_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dt_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.b_proj = nn.Linear(hidden_dim, state_dim)
        self.c_proj = nn.Linear(hidden_dim, state_dim)
        self.a_log = nn.Parameter(torch.zeros(hidden_dim, state_dim))
        # S4D-Lin-like negative real initialization (real diagonal simplification)
        idx = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.register_buffer("a_init", -torch.log(idx).unsqueeze(0).expand(
            hidden_dim, state_dim).clone())
        with torch.no_grad():
            self.a_log.copy_(self.a_init)
        self.d_skip = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        b, t, h = x.shape
        u = self.in_proj(x)
        u = torch.nn.functional.gelu(self.conv(causal_left_pad(u, 1).transpose(1, 2))
                                     .transpose(1, 2)) + u
        dt = torch.nn.functional.softplus(self.dt_proj(u) + self.dt_bias)
        a = -torch.exp(self.a_log)                       # [H, S] negative
        state = u.new_zeros(b, h, self.state_dim)
        ys = []
        for ti in range(t):
            d = dt[:, ti, :]                              # [B,H]
            a_bar = torch.exp(d.unsqueeze(-1) * a)        # [B,H,S]
            b_proj = self.b_proj(u[:, ti])                # [B,S]
            c_proj = self.c_proj(u[:, ti])                # [B,S]
            state = a_bar * state + b_proj.unsqueeze(1) * u[:, ti].unsqueeze(-1)
            y = (state * c_proj.unsqueeze(1)).sum(dim=-1) + self.d_skip * u[:, ti]
            ys.append(y)
        return self.out_proj(torch.stack(ys, dim=1))


class DiagonalSelectiveSSM(CoreBase):
    name = "M12"
    paradigm = "ssm"
    stateful = False
    note = "SIMPLIFIED selective diagonal SSM (input-dependent dt/B/C, python scan)"

    def __init__(self, hidden_dim: int, layers: int = 2, state_dim: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.state_dim = int(state_dim)
        self.blocks = nn.ModuleList([
            _DiagonalSelectiveLayer(hidden_dim, self.state_dim)
            for _ in range(self.layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths=None, delta_s=None, state=None):
        # stateless-over-window implementation for training; for single-step
        # rollout the caller re-feeds the recent history (documented in rollout_q)
        z = x
        for block in self.blocks:
            z = z + self.drop(block(z))
        return z, None
