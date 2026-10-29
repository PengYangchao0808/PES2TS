"""V5.4.1 WP3b — E3 ``schnet_spatial``: ONE SchNet-like continuous-filter
update over the legal ``G_spatial`` view (plan §6.7 E3 / §6.6.2 / §6.6.1;
registry family ``schnet_spatial``, id ``E3``).

Formula — frozen §6.7, width = :data:`~...interfaces.HIDDEN_V541` = 48,
exactly ONE message update, no coordinate update::

    c(r) = 0.5 · (cos(π · r / 4.5) + 1)      for r < 4.5 Å, else 0
    F    : RBF12(12) → width → width         (ReLU middle, linear out)
    A    : linear width → width
    m_i  = Σ_{j ∈ N_spatial(i)}  A(h_j) ⊙ F(RBF12(r_ji)) · c(r_ji)
    f    = atomwise_MLP(m)                   (width → width → width)

Scope discipline — this module is NEW and touches nothing else:

* **view** — only the WP3a ``G_spatial`` provider
  (``build_spatial_graph``): the legacy radius selection recomputed from
  the CURRENT ``pos`` (Å).  Covalent pairs kept beyond the cutoff are
  never spatial edges (§6.6.2); the ≤32 cap with 1e-6 Å tie retention
  may realise a higher degree and is never re-truncated by atom id;
* **F2 untouched** — distances/RBF come from ``pos``; the F2-masked
  first 12 ``edge_attr`` slots are NEVER read and NEVER written (the
  input batch is read-only, §6.6.1);
* **geometry** — only DISTANCE-derived scalar weights enter the message:
  no vector output, no coordinate update, no raw-coordinate feature, so
  ``delta_node`` is an invariant scalar channel (E(3)-safe by
  construction);
* **empty view** — a node with no spatial neighbour aggregates exactly
  ``m = 0`` (``scatter_sum`` on an empty index); the atomwise MLP is then
  evaluated at 0 (no neighbour-specific parameter is added);
* **diagnostics** — ``diagnostics_enabled`` (default True) is forwarded
  to the provider; ``last_diagnostics`` holds detached Python floats
  (provider ``counts`` + ``n_nodes`` / ``n_edges`` / ``zero_neighbor``)
  and is ``{}`` when disabled.  No autograd graph is ever retained;
* **registration** — importing this module installs the real E3 builder
  with ``register_embedding("E3", ..., replace=True)`` (the WP3 path);
  no placeholder math exists anywhere in the repo for E3.

Not claimed: a complete SchNet reproduction (no energy/force head, no
multi-layer stack, no discrete-neighbour smoothness — plan §6.7 [R8]).
"""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.v541.embedding_registry import register_embedding
from pes2ts_core.pesn.models.v541.embeddings.geometric_common import (
    ENVELOPE_CUTOFF,
    GeometricEmbeddingBase,
    cosine_envelope,
    scatter_sum,
    two_layer_mlp,
)
from pes2ts_core.pesn.models.v541.feature_providers import build_spatial_graph
from pes2ts_core.pesn.models.v541.feature_providers.common import (
    DEFAULT_CHUNK_SIZE,
    RBF_N,
)
from pes2ts_core.pesn.models.v541.feature_providers.legal_neighbors import (
    SpatialGraphView,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    NodeDeltaEmbedding,
)

#: Registry id implemented by this module (frozen campaign row).
EMBEDDING_ID_E3: Final[str] = "E3"


class SchnetSpatialEmbedding(GeometricEmbeddingBase):
    """E3: one SchNet-like continuous-filter update → ``delta_node``.

    ``forward(h_legacy, batch)`` rebuilds ``G_spatial`` from the
    forward-view batch (lazy, no cache — §6.6.6), then evaluates the
    formula above.  :meth:`aggregate_delta` is the pure view→delta math
    (provider call factored out) so tests can exercise a fixed view
    directly; E5 does NOT reuse this module (§6.7 E5 = GINE24 + Angle24).
    """

    def __init__(
            self,
            width: int = HIDDEN_V541,
            *,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            envelope_cutoff: float = ENVELOPE_CUTOFF,
            diagnostics_enabled: bool = True,
    ) -> None:
        super().__init__(
            width=width, chunk_size=chunk_size,
            diagnostics_enabled=diagnostics_enabled)
        self.envelope_cutoff = float(envelope_cutoff)
        # A(h_j): linear width → width (§6.7 E3).
        self.atom_proj = nn.Linear(self.width, self.width)
        # F(RBF12): 12 → width → width, ReLU middle, linear output.
        self.filter_mlp = two_layer_mlp(RBF_N, self.width, self.width)
        # atomwise MLP(m): width → width → width.
        self.out_mlp = two_layer_mlp(self.width, self.width, self.width)

    # -- pure view → delta math --------------------------------------------
    def aggregate_delta(
            self, h_legacy: torch.Tensor, view: SpatialGraphView,
    ) -> torch.Tensor:
        """``f`` from an ALREADY-BUILT ``G_spatial`` view (no provider call)."""
        n_nodes = int(h_legacy.shape[0])
        # Directed layout of the bidirectional view: edge (i → j) means
        # node i RECEIVES the message of neighbour j (both directions of
        # every pair exist, so each node aggregates each neighbour once).
        receiver = view.edge_index[0]
        neighbor = view.edge_index[1]
        rbf = view.rbf12.to(dtype=h_legacy.dtype)        # [E, 12] recomputed
        dist = view.distance.to(dtype=h_legacy.dtype)    # [E] current Å
        envelope = cosine_envelope(dist, self.envelope_cutoff)   # [E]
        # distance-only scalar weight: F(RBF12) ⊙ c(r)
        weight = self.filter_mlp(rbf) * envelope.unsqueeze(-1)
        # A(h_j) ⊙ weight  (valid is all-true by provider contract; kept
        # so an unforeseen invalid row could never inject a non-finite)
        message = (self.atom_proj(h_legacy[neighbor]) * weight
                   * view.valid.to(dtype=h_legacy.dtype).unsqueeze(-1))
        m = scatter_sum(n_nodes, receiver, message)      # m = 0 if no neighbour
        return self.out_mlp(m)

    # -- NodeDeltaEmbedding contract ---------------------------------------
    def forward(
            self, h_legacy: torch.Tensor,
            batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        view = build_spatial_graph(
            batch, chunk_size=self.chunk_size,
            diagnostics=self.diagnostics_enabled)
        delta = self.aggregate_delta(h_legacy, view)
        own: dict[str, float] = {}
        if self.diagnostics_enabled:
            degree = torch.bincount(
                view.edge_index[0], minlength=int(h_legacy.shape[0]))
            own = {
                "n_nodes": float(h_legacy.shape[0]),
                "n_edges": float(int(view.edge_index.shape[1])),
                "zero_neighbor": float(int((degree == 0).sum())),
            }
        self._record_diagnostics(view.counts, **own)
        return delta


def build_schnet_spatial() -> NodeDeltaEmbedding:
    """Zero-arg registry builder for E3 (width 48, frozen defaults)."""
    return SchnetSpatialEmbedding()


# WP3 registration at import time (plan §5.1; the main agent wires this
# module through ``embeddings/__init__.py`` — this file does not).
register_embedding(EMBEDDING_ID_E3, build_schnet_spatial, replace=True)

__all__ = [
    "EMBEDDING_ID_E3",
    "SchnetSpatialEmbedding",
    "build_schnet_spatial",
]
