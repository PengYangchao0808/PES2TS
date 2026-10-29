"""V5.4.1 WP3b — E1 ``gin_e_chemical``: ONE GINE-chem layer over the
legal ``G_chem`` view (plan §6.7 E1 / §6.6.4; registry family
``gin_e_chemical``, id ``E1``).

Formula — frozen §6.7, width = :data:`~...interfaces.HIDDEN_V541` = 48
(E5 later builds the same class at width 24), exactly ONE message
update, epsilon fixed at 0::

    P_e   : linear 8 → width                    (chem-edge projection)
    m_ji  = ReLU( h_j + P_e e_ji )              (j = neighbour, i = receiver)
    f     = MLP( (1 + ε) · h + Σ_j m_ji ),  ε = 0, FIXED
    MLP   : Linear(width→width) → ReLU → Linear(width→width)

with the SELF term EXPLICIT and OUTSIDE the sum (no graph self-loop is
ever added; the provider refuses self loops), so an isolated node
aggregates exactly ``f = MLP(h)``.  The output is the residual
``delta_node`` — the outer §5.2 ``W``/``alpha`` gate is NOT duplicated
inside the candidate.

Scope discipline — this module is NEW and touches nothing else:

* **view** — ONLY the WP3a ``G_chem`` provider (``build_chem_graph``):
  the legal source/target covalent union plus reaction-edit pairs,
  deduplicated per undirected pair and expanded to BOTH message
  directions with bitwise-identical attributes (§6.6.3/§6.6.4).  The
  signed bond delta therefore never flips with message direction — only
  a reversed INTENT changes it;
* **inputs** — only the frozen 8-field chem attributes
  (``chem_edge_fields``): never a distance, never an F2-masked
  ``edge_attr`` slot, never ``pos`` ⇒ ``delta_node`` is an invariant
  scalar channel (E(3)-safe by construction);
* **math** — pure PyTorch, no dropout / BatchNorm / graph pooling /
  coordinate update (plan §6.7 公共项); aggregation is a graph-local
  ``index_add_`` so it is permutation-equivariant and never mixes graphs;
* **diagnostics** — ``diagnostics_enabled`` (default True) is forwarded
  to the provider; ``last_diagnostics`` holds detached Python floats
  (provider ``counts`` + ``n_nodes`` / ``n_edges`` / ``zero_neighbor`` /
  ``zero_neighbor_frac``) and is ``{}`` when disabled.  No autograd
  graph and no view is ever retained between forwards;
* **registration** — importing this module installs the real E1 builder
  with ``register_embedding("E1", ..., replace=True)`` (the WP3 path).

E1 has NO degree statistics (no degree scaler) — the inner-train degree
stats of §6.7 are E2-only.
"""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.v541.embedding_registry import register_embedding
from pes2ts_core.pesn.models.v541.embeddings.chemical_common import (
    ChemicalEmbeddingBase,
    scatter_sum,
    two_layer_mlp,
)
from pes2ts_core.pesn.models.v541.feature_providers import build_chem_graph
from pes2ts_core.pesn.models.v541.feature_providers.common import CHEM_EDGE_DIM
from pes2ts_core.pesn.models.v541.feature_providers.legal_neighbors import (
    ChemGraphView,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    NodeDeltaEmbedding,
)

#: Registry id implemented by this module (frozen campaign row).
EMBEDDING_ID_E1: Final[str] = "E1"

#: Frozen §6.7 GINE epsilon — FIXED at 0, never learned, never a parameter.
GINE_EPSILON: Final[float] = 0.0


class GINEChemicalEmbedding(ChemicalEmbeddingBase):
    """E1: one GINE-chem layer → ``delta_node``.

    ``forward(h_legacy, batch)`` rebuilds ``G_chem`` from the
    forward-view batch (lazy, no cache — §6.6.6), then evaluates the
    formula above.  :meth:`aggregate_delta` is the pure view→delta math
    (provider call factored out) so tests can feed a hand-built view.
    """

    def __init__(
            self,
            width: int = HIDDEN_V541,
            *,
            diagnostics_enabled: bool = True,
    ) -> None:
        super().__init__(
            width=width, diagnostics_enabled=diagnostics_enabled)
        # P_e: linear projection of the frozen 8-field chem-edge attrs.
        self.pe_proj = nn.Linear(CHEM_EDGE_DIM, self.width)
        # atomwise MLP: width → width → width (ReLU middle, linear out).
        self.out_mlp = two_layer_mlp(self.width, self.width, self.width)
        # ε = 0 fixed (a plain constant — NOT a parameter/buffer).
        self.eps = GINE_EPSILON

    # -- pure view → delta math --------------------------------------------
    def aggregate_delta(
            self, h_legacy: torch.Tensor, view: ChemGraphView,
    ) -> torch.Tensor:
        """``f`` from an ALREADY-BUILT ``G_chem`` view (no provider call).

        Message direction convention (identical to the E3/E4 candidates):
        ``edge_index[0]`` is the RECEIVER and ``edge_index[1]`` the
        neighbour; because the view expands every pair to BOTH directions
        with bitwise-identical attributes, each node aggregates each
        neighbour exactly once and the result is independent of which
        row is called receiver.
        """
        n_nodes = int(h_legacy.shape[0])
        receiver = view.edge_index[0]
        neighbour = view.edge_index[1]
        pe = self.pe_proj(view.edge_attr.to(dtype=h_legacy.dtype))  # [E, w]
        # ReLU(h_j + P_e e_ji): neighbour's h, pair attrs (same both ways)
        message = torch.relu(h_legacy[neighbour] + pe)             # [E, w]
        aggregated = scatter_sum(n_nodes, receiver, message)       # Σ_j
        # explicit self term OUTSIDE the sum; no graph self-loop exists
        x = (1.0 + self.eps) * h_legacy + aggregated
        return self.out_mlp(x)

    # -- NodeDeltaEmbedding contract ---------------------------------------
    def forward(
            self, h_legacy: torch.Tensor,
            batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        view = build_chem_graph(
            batch, diagnostics=self.diagnostics_enabled)
        delta = self.aggregate_delta(h_legacy, view)
        own: dict[str, float] = {}
        if self.diagnostics_enabled:
            # in-degree of the bidirectional view = incident chem pairs
            degree = torch.bincount(
                view.edge_index[0], minlength=int(h_legacy.shape[0]))
            zero = int((degree == 0).sum())
            n_nodes = int(h_legacy.shape[0])
            own = {
                "n_nodes": float(n_nodes),
                "n_edges": float(int(view.edge_index.shape[1])),
                "zero_neighbor": float(zero),
                "zero_neighbor_frac": float(zero) / float(max(n_nodes, 1)),
            }
        self._record_diagnostics(view.counts, **own)
        return delta


def build_gin_e_chemical() -> NodeDeltaEmbedding:
    """Zero-arg registry builder for E1 (width 48, frozen defaults)."""
    return GINEChemicalEmbedding()


# WP3 registration at import time (plan §5.1; the main agent wires this
# module through ``embeddings/__init__.py`` — this file does not).
register_embedding(EMBEDDING_ID_E1, build_gin_e_chemical, replace=True)

__all__ = [
    "EMBEDDING_ID_E1",
    "GINE_EPSILON",
    "GINEChemicalEmbedding",
    "build_gin_e_chemical",
]
