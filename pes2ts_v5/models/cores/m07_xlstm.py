"""M07 xLSTM (SIMPLIFIED) — sLSTM-style exponential-gated memory.

Simplification (documented, not hidden): only the sLSTM scalar-memory branch is
implemented (exponential input/forget gates + normalizer state).  The mLSTM
matrix-memory branch and the official xLSTM block stacking are **not**
implemented.  No external `xlstm`/`mamba_ssm` dependency is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class _SLSTMCell(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.wi = nn.Linear(2 * hidden_dim, hidden_dim)
        self.wf = nn.Linear(2 * hidden_dim, hidden_dim)
        self.wo = nn.Linear(2 * hidden_dim, hidden_dim)
        self.wz = nn.Linear(2 * hidden_dim, hidden_dim)
        self.eps = 1e-6

    def forward(self, x_t, c, m):
        cat = torch.cat([x_t, c], dim=-1)
        i = torch.exp(self.wi(cat).clamp(max=5.0))
        f = torch.exp(self.wf(cat).clamp(max=5.0))
        z = torch.tanh(self.wz(cat))
        o = torch.sigmoid(self.wo(cat))
        m_new = torch.maximum(torch.log(f + self.eps) + m, torch.log(i + self.eps))
        c_new = f * c + i * z
        c_new = torch.clamp(c_new, -1e3, 1e3)
        h_new = o * c_new * torch.exp(-m_new)
        h_new = torch.clamp(h_new, -10.0, 10.0)
        return h_new, (c_new, m_new)


class XLSTMSmall(CoreBase):
    name = "M07"
    paradigm = "gated-memory"
    stateful = True
    note = "SIMPLIFIED xLSTM: sLSTM-only exponential gating, no mLSTM matrix memory"

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.cells = nn.ModuleList([_SLSTMCell(hidden_dim) for _ in range(self.layers)])
        self.drop = nn.Dropout(dropout)

    def init_state(self, batch_size: int, device=None):
        device = device or next(self.parameters()).device
        zero = torch.zeros(batch_size, self.hidden_dim, device=device)
        return tuple((zero.clone(), zero.clone()) for _ in range(self.layers))

    def forward(self, x, lengths=None, delta_s=None, state=None):
        b, t, _ = x.shape
        if state is None:
            zero = x.new_zeros(b, self.hidden_dim)
            states = [(zero, zero.clone()) for _ in range(self.layers)]
        else:
            states = list(state)
        outs = []
        for ti in range(t):
            inp = x[:, ti]
            new_states = []
            for li, cell in enumerate(self.cells):
                c, m = states[li]
                h, (c_new, m_new) = cell(inp, c, m)
                new_states.append((c_new, m_new))
                inp = self.drop(h)
            states = new_states
            outs.append(inp)
        return torch.stack(outs, dim=1), tuple(states)
