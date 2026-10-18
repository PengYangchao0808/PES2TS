"""Small graph message-passing encoder (B4, torch-backed).

Design (G3-P1 §5.1 v1 config): hidden 64, 3 interaction layers, 16-bin
Gaussian distance basis, 5 A spatial cutoff, typed task edges that never
disappear at any distance.  Messages are built from invariant edge features
(Gaussian distance basis + edge-type embedding + bond-order scalar); node
updates use residual + LayerNorm.  The readout mean-pools node states per
graph and exposes per-driver-edge features (endpoint node states + edge
attribute + current scaled coordinate value) for the direction/activity
heads — every output is a scalar or low-dim internal-coordinate quantity, so
no Cartesian equivariant head is needed for rigid invariance.

A whole batch is processed as concatenated graphs with offset edge indices
(vectorized ``index_add_`` aggregation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from pes2ts_core.pesn.models import require_torch

#: Reference v1 configuration (G3-P1 §5.1); recorded in run records.
DEFAULT_HIDDEN: Final[int] = 64
DEFAULT_N_LAYERS: Final[int] = 3
N_EDGE_ROLES: Final[int] = 6
MAX_ATOMIC_NUMBER: Final[int] = 118


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    """Batched encoder outputs as torch tensors."""

    pooled: object  # (B, hidden)
    driver_edge_feat: object  # (D_total, 3*hidden + 1)
    driver_graph_id: object  # (D_total,) long
    driver_counts: object  # (B,) long


class _TorchEncoderFactory:
    """Builds the torch.nn.Module once torch is confirmed available."""

    def build(self, torch, hidden: int, n_layers: int, n_basis: int):
        nn = torch.nn

        class _Encoder(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.z_embed = nn.Embedding(MAX_ATOMIC_NUMBER + 1, hidden)
                inner.role_embed = nn.Embedding(N_EDGE_ROLES, hidden)
                inner.dist_proj = nn.Linear(n_basis, hidden)
                inner.order_proj = nn.Linear(2, hidden)
                inner.layers = nn.ModuleList()
                for _ in range(n_layers):
                    inner.layers.append(
                        nn.ModuleDict(
                            {
                                "edge_mlp": nn.Sequential(
                                    nn.Linear(3 * hidden, hidden),
                                    nn.SiLU(),
                                    nn.Linear(hidden, hidden),
                                ),
                                "norm": nn.LayerNorm(hidden),
                                "node_mlp": nn.Sequential(
                                    nn.Linear(hidden, hidden),
                                    nn.SiLU(),
                                    nn.Linear(hidden, hidden),
                                ),
                            }
                        )
                    )

            def forward(inner, batch: dict[str, object]) -> EncoderOutput:
                h = inner.z_embed(batch["atomic_numbers"])
                role_matrix = inner.role_embed.weight
                edge_attr = (
                    inner.dist_proj(batch["edge_basis"])
                    + batch["edge_roles"] @ role_matrix
                    + inner.order_proj(
                        torch.stack([batch["edge_r_order"], batch["edge_p_order"]], dim=-1)
                    )
                )
                edge_src = batch["edge_src"]
                edge_dst = batch["edge_dst"]
                for layer in inner.layers:
                    messages = layer["edge_mlp"](
                        torch.cat([h[edge_src], h[edge_dst], edge_attr], dim=-1)
                    )
                    aggregated = h.new_zeros(h.shape)
                    aggregated.index_add_(0, edge_dst, messages)
                    h = layer["norm"](h + layer["node_mlp"](aggregated))

                n_graphs = int(batch["n_graphs"])
                node_counts = torch.bincount(
                    batch["node_graph_id"], minlength=n_graphs
                ).to(h.dtype)
                pooled = h.new_zeros((n_graphs, h.shape[-1]))
                pooled.index_add_(0, batch["node_graph_id"], h)
                pooled = pooled / node_counts.unsqueeze(-1).clamp(min=1.0)

                driver_slot = batch["driver_edge_slot"]
                driver_feat = torch.cat(
                    [
                        h[edge_src[driver_slot]],
                        h[edge_dst[driver_slot]],
                        edge_attr[driver_slot],
                        batch["driver_q_scaled"].unsqueeze(-1),
                    ],
                    dim=-1,
                )
                return EncoderOutput(
                    pooled=pooled,
                    driver_edge_feat=driver_feat,
                    driver_graph_id=batch["driver_graph_id"],
                    driver_counts=batch["driver_counts"],
                )

        return _Encoder()


class SmallGraphEncoder:
    """Batch graph encoder over runtime :class:`GraphFeatures`."""

    def __init__(
        self,
        *,
        hidden: int = DEFAULT_HIDDEN,
        n_layers: int = DEFAULT_N_LAYERS,
        n_basis: int = 16,
    ) -> None:
        self.torch = require_torch()
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_basis = n_basis
        self.module = _TorchEncoderFactory().build(self.torch, hidden, n_layers, n_basis)

    def parameters(self):
        return self.module.parameters()

    def train_mode(self, flag: bool) -> None:
        self.module.train(flag)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state) -> None:
        self.module.load_state_dict(state)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    def collate(self, features_list: Sequence[object]) -> dict[str, object]:
        torch = self.torch
        z_chunks: list[np.ndarray] = []
        src_chunks: list[np.ndarray] = []
        dst_chunks: list[np.ndarray] = []
        roles_chunks: list[np.ndarray] = []
        basis_chunks: list[np.ndarray] = []
        r_order_chunks: list[np.ndarray] = []
        p_order_chunks: list[np.ndarray] = []
        node_id_chunks: list[np.ndarray] = []
        driver_slot_chunks: list[np.ndarray] = []
        driver_id_chunks: list[np.ndarray] = []
        driver_q_chunks: list[np.ndarray] = []
        driver_counts: list[int] = []
        offset = 0
        for graph_id, features in enumerate(features_list):
            n_atoms = features.n_atoms
            z_chunks.append(features.atomic_numbers)
            node_id_chunks.append(np.full(n_atoms, graph_id, dtype=np.int64))
            src_chunks.append(features.edge_index[0] + offset)
            dst_chunks.append(features.edge_index[1] + offset)
            roles_chunks.append(features.edge_roles)
            basis_chunks.append(features.edge_distance_basis)
            r_order_chunks.append(features.edge_r_order)
            p_order_chunks.append(features.edge_p_order)
            driver_slot_chunks.append(features.driver_edge_index)
            n_driver = len(features.driver_edge_index)
            driver_id_chunks.append(np.full(n_driver, graph_id, dtype=np.int64))
            driver_q_chunks.append(features.driver_q_scaled)
            driver_counts.append(n_driver)
            offset += n_atoms

        def _long(chunks: list[np.ndarray]) -> object:
            return torch.as_tensor(np.concatenate(chunks), dtype=torch.long)

        def _float(chunks: list[np.ndarray]) -> object:
            return torch.as_tensor(np.concatenate(chunks), dtype=torch.float32)

        if z_chunks and basis_chunks:
            basis_width = np.concatenate(basis_chunks).shape[1]
            if basis_width != self.n_basis:
                raise ValueError(
                    f"feature basis width {basis_width} != encoder n_basis {self.n_basis}"
                )
        return {
            "atomic_numbers": _long(z_chunks),
            "edge_src": _long(src_chunks),
            "edge_dst": _long(dst_chunks),
            "edge_roles": _float(roles_chunks),
            "edge_basis": _float(basis_chunks),
            "edge_r_order": _float(r_order_chunks),
            "edge_p_order": _float(p_order_chunks),
            "node_graph_id": _long(node_id_chunks),
            "n_graphs": len(features_list),
            "driver_edge_slot": _long(driver_slot_chunks),
            "driver_graph_id": _long(driver_id_chunks),
            "driver_counts": torch.as_tensor(driver_counts, dtype=torch.long),
            "driver_q_scaled": _float(driver_q_chunks),
        }

    def forward(self, features_list: Sequence[object]) -> EncoderOutput:
        return self.module(self.collate(features_list))

    def __call__(self, features_list: Sequence[object]) -> EncoderOutput:
        return self.forward(features_list)


__all__ = [
    "DEFAULT_HIDDEN",
    "DEFAULT_N_LAYERS",
    "EncoderOutput",
    "SmallGraphEncoder",
]
