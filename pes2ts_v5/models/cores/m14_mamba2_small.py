"""M14 Mamba-2 Small — SIMPLIFIED selective SSM block (no hardware scan).

Simplification: depthwise causal conv + input-dependent (dt, B, C) diagonal
selective scan implemented as a plain Python time loop, plus a gated MLP
branch.  No chunked parallel scan, no tensor-core matmul, no `mamba_ssm`
dependency; parameter scale is configurable for the fairness protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase, causal_left_pad


class _Mamba2Block(nn.Module):
    def __init__(self, hidden_dim: int, expand: int = 2, state_dim: int = 16,
                 d_conv: int = 2):
        super().__init__()
        inner = hidden_dim * expand
        self.inner = inner
        self.state_dim = state_dim
        self.in_proj = nn.Linear(hidden_dim, 2 * inner)
        self.conv = nn.Conv1d(inner, inner, kernel_size=d_conv, groups=inner)
        self.x_proj = nn.Linear(inner, 1 + 2 * state_dim)
        self.dt_bias = nn.Parameter(torch.zeros(inner))
        idx = torch.arange(1, state_dim + 1, dtype=torch.float32)
        self.a_log = nn.Parameter(-torch.log(idx).unsqueeze(0).expand(inner, state_dim).clone())
        self.d_skip = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        b, t, _ = x.shape
        u, gate = self.in_proj(x).chunk(2, dim=-1)          # [B,T,inner] each
        u = torch.nn.functional.silu(
            self.conv(causal_left_pad(u, 1).transpose(1, 2)).transpose(1, 2))
        dt_bc = self.x_proj(u)
        dt = torch.nn.functional.softplus(dt_bc[..., :1] + self.dt_bias)  # [B,T,inner]
        bc = dt_bc[..., 1:]
        B_t, C_t = bc[..., :self.state_dim], bc[..., self.state_dim:]
        a = -torch.exp(self.a_log)                          # [inner,S]
        state = u.new_zeros(b, self.inner, self.state_dim)
        ys = []
        for ti in range(t):
            d = dt[:, ti, :]                                # [B,inner]
            a_bar = torch.exp(d.unsqueeze(-1) * a)          # [B,inner,S]
            state = a_bar * state + B_t[:, ti].unsqueeze(1) * u[:, ti].unsqueeze(-1)
            ys.append((state * C_t[:, ti].unsqueeze(1)).sum(dim=-1)
                      + self.d_skip * u[:, ti])
        y = torch.stack(ys, dim=1) * torch.nn.functional.silu(gate)
        return self.norm(x + self.out_proj(y))


class Mamba2Small(CoreBase):
    name = "M14"
    paradigm = "ssm"
    stateful = False
    note = "SIMPLIFIED Mamba-2: selective scan (python loop) + gated MLP branch"

    def __init__(self, hidden_dim: int, layers: int = 2, expand: int = 2,
                 state_dim: int = 16, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.blocks = nn.ModuleList([
            _Mamba2Block(hidden_dim, expand=expand, state_dim=state_dim)
            for _ in range(self.layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths=None, delta_s=None, state=None):
        z = x
        for block in self.blocks:
            z = z + self.drop(block(z))
        return z, None
