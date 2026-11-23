"""StarEGNN v2 — OPTIONAL / EXPERIMENTAL encoder stub (V5.7-A Encoder ablation).

Status: **stub-quality**, not part of the V5.6-A primary screen.  It is a
dependency-free (no torch_geometric) EGNN-lite over a dense distance graph:
radial-basis edge messages + node updates + masked mean pooling.  It exists so
the Encoder ablation can be wired early; it must not be reported as a trained
or validated architecture until actually run in V5.7-A.

Input tensors are padded atom tensors (max_atoms), produced by an adapter from
the V5.5 dataset contract (V5.5 does not ship padded atom tensors by default).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _RadialBasis(nn.Module):
    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, float(cutoff), n_rbf)
        self.register_buffer("centers", centers, persistent=True)
        self.width = float(cutoff) / max(n_rbf - 1, 1)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((dist.unsqueeze(-1) - self.centers) / self.width) ** 2)


class StarEGNNv2Encoder(nn.Module):
    """EGNN-lite: per-frame node features + radial-basis message passing."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        n_atom_types: int = 16,
        n_layers: int = 2,
        n_rbf: int = 16,
        cutoff: float = 5.0,
        max_atoms: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.max_atoms = int(max_atoms)
        self.atom_embed = nn.Embedding(int(n_atom_types), hidden_dim)
        self.rbf = _RadialBasis(n_rbf, cutoff)
        self.edge_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim + n_rbf, hidden_dim),
                          nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_layers)])
        self.node_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                          nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        coords: torch.Tensor,       # [B, T, A, 3]
        atom_types: torch.Tensor,   # [B, T, A] int64 (0 = padded)
        atom_mask: torch.Tensor,    # [B, T, A] bool
    ) -> torch.Tensor:              # [B, T, hidden_dim]
        b, t, a, _ = coords.shape
        flat_mask = atom_mask.reshape(-1, a)
        h = self.atom_embed(atom_types.reshape(-1, a).clamp(0, self.atom_embed.num_embeddings - 1))
        pos = coords.reshape(-1, a, 3)
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        dist = torch.linalg.norm(diff, dim=-1)
        rbf = self.rbf(dist)
        pair_mask = flat_mask[:, :, None] & flat_mask[:, None, :]
        for li in range(self.n_layers):
            h_i = h[:, :, None, :].expand(-1, a, a, -1)
            h_j = h[:, None, :, :].expand(-1, a, a, -1)
            msg = self.edge_mlp[li](torch.cat([h_i, h_j, rbf], dim=-1))
            msg = msg * pair_mask.unsqueeze(-1).to(msg.dtype)
            agg = msg.sum(dim=2) / pair_mask.sum(dim=2, keepdim=True).clamp(min=1).to(msg.dtype)
            h = h + self.drop(self.node_mlp[li](torch.cat([h, agg], dim=-1)))
        denom = flat_mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        pooled = (h * flat_mask.unsqueeze(-1).to(h.dtype)).sum(dim=1) / denom
        return self.out_proj(pooled).reshape(b, t, self.hidden_dim)
