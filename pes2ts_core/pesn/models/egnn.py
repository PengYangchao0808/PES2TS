"""Equivariant EGNN encoder for the direct policy (WP2, plan v3 §6).

Architecture (first-version candidates, plan §6): hidden 128, 4 layers,
32-bin Gaussian radial basis, 5 A cutoff; the light control is 64/3/32.
Equivariance: node scalars ``h`` are invariant; an equivariant vector
channel ``v`` accumulates unit relative directions gated by scalar
messages, so vector readouts rotate with the input geometry while scalar
readouts stay unchanged. Task edges (formed/broken/order_changed) ride the
same message passing with their own role embedding and are never
distance-cut.

Torch is resolved lazily through ``require_torch()`` and the module is
built inside a factory (same discipline as ``models/encoder.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from pes2ts_core.pesn.features.direct_graph import N_TASK_ROLES
from pes2ts_core.pesn.models import require_torch

EGNN_VERSION: Final[str] = "pesn_direct_egnn_v1.0"
MAX_Z: Final[int] = 100


@dataclass(frozen=True, slots=True)
class EGNNConfig:
    hidden: int = 128
    n_layers: int = 4
    n_basis: int = 32
    cutoff: float = 5.0
    light: bool = False

    def __post_init__(self) -> None:
        if self.light:
            object.__setattr__(self, "hidden", min(self.hidden, 64))
            object.__setattr__(self, "n_layers", min(self.n_layers, 3))


class _TorchEncoderFactory:
    """Builds the torch.nn.Module once torch is confirmed available."""

    def build(self, torch, config: EGNNConfig):
        nn = torch.nn
        hidden = config.hidden

        class _EGNN(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.config = config
                inner.z_embed = nn.Embedding(MAX_Z + 1, hidden)
                inner.basis_proj = nn.Linear(config.n_basis, hidden)
                inner.role_embed = nn.Embedding(2 ** N_TASK_ROLES, hidden, padding_idx=0)
                inner.grad_proj = nn.Linear(2, hidden)   # [|g|, |g|^2] invariants
                inner.grad_gate = nn.Linear(hidden, 1)   # equivariant vector seed
                inner.scalar_proj = nn.Linear(6, hidden)
                inner.edge_mlp = nn.ModuleList(
                    [nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden))
                     for _ in range(config.n_layers)])
                inner.node_mlp = nn.ModuleList(
                    [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden))
                     for _ in range(config.n_layers)])
                inner.vector_gate = nn.ModuleList(
                    [nn.Linear(hidden, 1) for _ in range(config.n_layers)])
                inner.norm_h = nn.ModuleList(
                    [nn.LayerNorm(hidden) for _ in range(config.n_layers)])

            def forward(inner, batch: dict) -> dict:
                h = inner.z_embed(batch["atomic_numbers"].clamp(max=MAX_Z))
                grad = batch["gradient"]
                # scalar channel: strictly invariant gradient summaries only
                g_invariant = torch.cat(
                    [grad.norm(dim=1, keepdim=True),
                     (grad * grad).sum(dim=1, keepdim=True)], dim=1)
                h = h + inner.grad_proj(g_invariant)
                scalars = batch["graph_scalars"]
                h = h + inner.scalar_proj(scalars[batch["node_graph_id"]])

                coords = batch["coords"]
                src, dst = batch["edge_index"][0], batch["edge_index"][1]
                rel = coords[src] - coords[dst]
                dist = rel.norm(dim=1, keepdim=True).clamp(min=1e-6)
                edge_static = inner.basis_proj(batch["edge_basis"]) + inner.role_embed(
                    inner._role_ids(batch))
                # equivariant vector channel seeded with the gradient
                # (gate from invariant scalars -> stays a 3-vector under rotation)
                v = grad * torch.sigmoid(inner.grad_gate(h))
                for layer in range(config.n_layers):
                    msg_in = torch.cat([h[src], h[dst], edge_static], dim=1)
                    messages = inner.edge_mlp[layer](msg_in)
                    agg = torch.zeros_like(h).index_add_(0, dst, messages)
                    h = inner.norm_h[layer](
                        h + inner.node_mlp[layer](torch.cat([h, agg], dim=1)))
                    gate = torch.sigmoid(inner.vector_gate[layer](messages))
                    v = v.index_add(0, dst, (rel / dist) * gate)
                return {"h": h, "v": v, "coords": coords,
                        "n_graphs": batch["n_graphs"],
                        "node_graph_id": batch["node_graph_id"],
                        "node_counts": batch["node_counts"]}

            def _role_ids(inner, batch: dict):
                roles = batch["edge_task_roles"]
                weights = (2 ** torch.arange(N_TASK_ROLES, device=roles.device)).unsqueeze(0)
                return (roles * weights).sum(dim=1).long()

        return _EGNN()


class DirectEGNNEncoder:
    """Wrapper mirroring ``SmallGraphEncoder``'s module-holder interface."""

    def __init__(self, config: EGNNConfig | None = None) -> None:
        self.torch = require_torch()
        self.config = config or EGNNConfig()
        self.module = _TorchEncoderFactory().build(self.torch, self.config)

    def __call__(self, batch: dict) -> dict:
        return self.module(batch)

    def parameters(self):
        return self.module.parameters()

    def train_mode(self, flag: bool) -> None:
        self.module.train(flag)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state) -> None:
        self.module.load_state_dict(state)

    def to(self, device) -> "DirectEGNNEncoder":
        self.module.to(device)
        return self

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())


def collate_direct_features(torch, features_list: list) -> dict:
    """Concatenated-graph batch (offset edge indices, per-node graph ids)."""
    offsets = np.cumsum([0] + [f.n_atoms for f in features_list]).tolist()
    n_total = int(offsets[-1])
    edge_blocks = []
    node_graph_id = np.zeros(n_total, dtype=np.int64)
    for gi, f in enumerate(features_list):
        edges = f.edge_index + offsets[gi]
        edge_blocks.append(edges)
        node_graph_id[offsets[gi]:offsets[gi + 1]] = gi
    edge_index = (np.concatenate(edge_blocks, axis=1) if edge_blocks
                  else np.zeros((2, 0), np.int64))
    T = torch
    return {
        "atomic_numbers": T.as_tensor(
            np.concatenate([f.atomic_numbers for f in features_list]).astype(np.int64)),
        "coords": T.as_tensor(
            np.concatenate([f.coordinates for f in features_list]).astype(np.float32)),
        "gradient": T.as_tensor(
            np.concatenate([f.gradient for f in features_list]).astype(np.float32)),
        "edge_index": T.as_tensor(edge_index.astype(np.int64)),
        "edge_basis": T.as_tensor(
            np.concatenate([f.edge_basis for f in features_list]).astype(np.float32)),
        "edge_task_roles": T.as_tensor(
            np.concatenate([f.edge_task_roles for f in features_list]).astype(np.float32)),
        "graph_scalars": T.as_tensor(
            np.stack([f.graph_scalars for f in features_list]).astype(np.float32)),
        "node_graph_id": T.as_tensor(node_graph_id),
        "node_counts": T.as_tensor(np.diff(np.asarray(offsets)).astype(np.int64)),
        "n_graphs": len(features_list),
    }


__all__ = [
    "EGNN_CONFIG_LIGHT",
    "EGNNConfig",
    "EGNN_VERSION",
    "DirectEGNNEncoder",
    "collate_direct_features",
]


EGNN_CONFIG_LIGHT: Final[EGNNConfig] = EGNNConfig(light=True)
