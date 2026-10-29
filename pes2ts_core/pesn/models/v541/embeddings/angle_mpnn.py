"""V5.4.1 WP3b — E4 ``angle_mpnn``: ONE Angle-MPNN layer over the
preregistered ``G_angle`` triplets (plan §6.7 E4 / §6.3; registry family
``angle_mpnn``, id ``E4``).

Formula — frozen §6.7, width = :data:`~...interfaces.HIDDEN_V541` = 48
(E5 later builds the same class at width 24), exactly ONE layer, no
directed-edge hidden state is persisted::

    a_ij   = MLP([ h_j , P_e(e_ji) , RBF12(r_ji) ])         → width
    t_ijk  = MLP([ h_i , a_ij + a_ik , a_ij ⊙ a_ik ,
                    1 , c , c² , c³ ])                       → width
    agg_i  = Σ_{{j,k} unordered pairs of centre i} t_ijk
    f      = out_MLP(agg) ⊙ center_has_angle

with every MLP ``Linear→ReLU→Linear`` (hidden width = width, ReLU in the
middle, LINEAR output) and ``P_e : 8 → width`` a plain linear projection
of the frozen 8-field chem-edge attributes.  ``c = cos θ_jik`` and the
``[1, c, c², c³]`` basis come from the WP3a angle provider.

Scope discipline — this module is NEW and touches nothing else:

* **view** — ONLY the preregistered ``G_angle`` triplets
  (``build_angle_triplets``): chem/intent neighbour pool (§6.3.2), ≤12
  neighbours per centre, ``reject_boundary_tie_shell``, per-pair
  two-side RBF12 + 8 chem fields + clamped cosine.  This is an
  angle-inspired NODE model — NOT DimeNet's spherical-harmonic/Bessel
  directed-edge model (§6.7 [R9]);
* **unordered pair symmetry** — ``j/k`` enter ONLY through the
  commutative sum ``a_ij + a_ik`` and product ``a_ij ⊙ a_ik``, so
  swapping the two sides is STRICTLY (bitwise) invariant; atom ids are
  never concatenated into any feature and never used for ranking;
* **missing angle** — a centre with no usable triplet aggregates exactly
  ``agg = 0`` (``scatter_sum`` on an empty index) and the EXPLICIT
  ``center_has_angle`` mask gates the output, so no learned
  missing-angle token (and certainly no atom-id-dependent one) exists;
* **F2 untouched / units** — both side distances and RBF12 are recomputed
  from the CURRENT ``pos`` (Å) by the provider; the F2-masked first 12
  ``edge_attr`` slots are NEVER read and NEVER written (§6.6.1);
* **diagnostics** — ``diagnostics_enabled`` (default True) is forwarded
  to the provider; ``last_diagnostics`` holds detached Python floats
  (provider ``counts`` + ``n_nodes`` / ``n_triplets`` / ``zero_angle``)
  and is ``{}`` when disabled.  No autograd graph is ever retained;
* **registration** — importing this module installs the real E4 builder
  with ``register_embedding("E4", ..., replace=True)`` (the WP3 path).

Degenerate geometry (zero-length angle side) is refused by the provider
with its typed ``V541_PROVIDER_ZERO_DISTANCE`` error — this module never
silently drops a triplet.
"""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.v541.embedding_registry import register_embedding
from pes2ts_core.pesn.models.v541.embeddings.geometric_common import (
    COS_BASIS_DIM,
    GeometricEmbeddingBase,
    scatter_sum,
    two_layer_mlp,
)
from pes2ts_core.pesn.models.v541.feature_providers import build_angle_triplets
from pes2ts_core.pesn.models.v541.feature_providers.angle_triplets import (
    AngleTripletView,
)
from pes2ts_core.pesn.models.v541.feature_providers.common import (
    CHEM_EDGE_DIM,
    DEFAULT_CHUNK_SIZE,
    RBF_N,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    NodeDeltaEmbedding,
)

#: Registry id implemented by this module (frozen campaign row).
EMBEDDING_ID_E4: Final[str] = "E4"


class AngleMPNNEmbedding(GeometricEmbeddingBase):
    """E4: one Angle-MPNN layer → ``delta_node``.

    ``forward(h_legacy, batch)`` rebuilds ``G_angle`` from the
    forward-view batch (lazy, no cache — §6.6.6), then evaluates the
    formula above.  :meth:`aggregate_delta` is the pure view→delta math
    (provider call factored out) so tests can feed a hand-swapped view
    and prove the strict ``j/k`` symmetry.
    """

    def __init__(
            self,
            width: int = HIDDEN_V541,
            *,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            diagnostics_enabled: bool = True,
    ) -> None:
        super().__init__(
            width=width, chunk_size=chunk_size,
            diagnostics_enabled=diagnostics_enabled)
        # P_e: linear projection of the frozen 8-field chem-edge attrs.
        self.pe_proj = nn.Linear(CHEM_EDGE_DIM, self.width)
        # a_ij MLP: [h_j (width), P_e e (width), RBF12 (12)] → width → width.
        self.pair_mlp = two_layer_mlp(
            self.width + self.width + RBF_N, self.width, self.width)
        # triplet MLP: [h_i, a_ij+a_ik, a_ij⊙a_ik, 1,c,c²,c³] → width → width.
        self.triplet_mlp = two_layer_mlp(
            self.width + self.width + self.width + COS_BASIS_DIM,
            self.width, self.width)
        # output MLP: width → width → width (applied to the aggregate).
        self.out_mlp = two_layer_mlp(self.width, self.width, self.width)

    # -- pure view → delta math --------------------------------------------
    def _side_embedding(
            self, h_legacy: torch.Tensor, view: AngleTripletView, side: int,
    ) -> torch.Tensor:
        """``a`` for ONE side of every triplet (``side`` ∈ {0, 1})."""
        neighbor = view.side_index[:, side]
        chem = view.side_chem_attr[:, side].to(dtype=h_legacy.dtype)
        rbf = view.side_rbf12[:, side].to(dtype=h_legacy.dtype)
        features = torch.cat(
            (h_legacy[neighbor], self.pe_proj(chem), rbf), dim=-1)
        return self.pair_mlp(features)

    def aggregate_delta(
            self, h_legacy: torch.Tensor, view: AngleTripletView,
    ) -> torch.Tensor:
        """``f`` from an ALREADY-BUILT ``G_angle`` view (no provider call)."""
        n_nodes = int(h_legacy.shape[0])
        # Both sides run through the SAME pair MLP; every later use is
        # commutative in (a_ij, a_ik) ⇒ strict j/k swap invariance.
        a0 = self._side_embedding(h_legacy, view, 0)
        a1 = self._side_embedding(h_legacy, view, 1)
        pair_sum = a0 + a1
        pair_prod = a0 * a1
        center = view.center_index
        basis = view.cos_basis.to(dtype=h_legacy.dtype)   # [T, 4] = [1,c,c²,c³]
        triplet_features = torch.cat(
            (h_legacy[center], pair_sum, pair_prod, basis), dim=-1)
        message = self.triplet_mlp(triplet_features)      # [T, width]
        agg = scatter_sum(n_nodes, center, message)       # 0 where no angle
        # explicit mask: a no-angle centre contributes exactly 0 (no
        # learned missing-angle token, no atom-id feature).
        mask = view.center_has_angle.to(dtype=h_legacy.dtype).unsqueeze(-1)
        return self.out_mlp(agg) * mask

    # -- NodeDeltaEmbedding contract ---------------------------------------
    def forward(
            self, h_legacy: torch.Tensor,
            batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        view = build_angle_triplets(
            batch, chunk_size=self.chunk_size,
            diagnostics=self.diagnostics_enabled)
        delta = self.aggregate_delta(h_legacy, view)
        own: dict[str, float] = {}
        if self.diagnostics_enabled:
            own = {
                "n_nodes": float(h_legacy.shape[0]),
                "n_triplets": float(int(view.center_index.shape[0])),
                "zero_angle": float(int(
                    (~view.center_has_angle).sum().item())),
            }
        self._record_diagnostics(view.counts, **own)
        return delta


def build_angle_mpnn() -> NodeDeltaEmbedding:
    """Zero-arg registry builder for E4 (width 48, frozen defaults)."""
    return AngleMPNNEmbedding()


# WP3 registration at import time (plan §5.1; the main agent wires this
# module through ``embeddings/__init__.py`` — this file does not).
register_embedding(EMBEDDING_ID_E4, build_angle_mpnn, replace=True)

__all__ = [
    "EMBEDDING_ID_E4",
    "AngleMPNNEmbedding",
    "build_angle_mpnn",
]
