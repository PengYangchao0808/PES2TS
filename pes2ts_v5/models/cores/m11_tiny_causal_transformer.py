"""M11 Tiny Causal Transformer — self-attention with strict causal mask."""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores.base import CoreBase


class TinyCausalTransformer(CoreBase):
    name = "M11"
    paradigm = "attention"
    stateful = False
    note = "Tiny TransformerEncoder, causal mask, learned positional embedding"

    def __init__(self, hidden_dim: int, layers: int = 2, n_heads: int = 4,
                 max_len: int = 64, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if hidden_dim % n_heads != 0:
            for h in (4, 2, 1):
                if hidden_dim % h == 0:
                    n_heads = h
                    break
        self.n_heads = int(n_heads)
        self.max_len = int(max_len)
        self.in_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, hidden_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=self.n_heads,
            dim_feedforward=2 * hidden_dim, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers),
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, lengths=None, delta_s=None, state=None):
        b, t, _ = x.shape
        if t > self.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.max_len}")
        h = self.in_proj(x) + self.pos[:, :t]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        mask = torch.zeros(t, t, dtype=x.dtype, device=x.device)
        mask = mask.masked_fill(causal, float("-inf"))
        out = self.encoder(h, mask=mask)
        return self.norm(out + h), None
