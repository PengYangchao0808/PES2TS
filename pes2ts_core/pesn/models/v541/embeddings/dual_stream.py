"""V5.4.1 WP3b — E5 ``dual_stream``: GINE24 ‖ Angle24 dual-stream residual
(plan §6.7 E5; registry family ``dual_stream``, id ``E5``, frozen row
``internal_view: E1_and_E4_shared_inputs``,
``conditional_parent: (E1, E4)`` — this module implements the MATH only
and makes no launch/qualification claim; §9.2 conditioning is a campaign
decision, not a property of this file).

Formula — frozen §6.7 E5, outer width =
:data:`~...interfaces.HIDDEN_V541` = 48, branch width 24, ONE external
``W``/``alpha`` gate (the §5.2 residual gate is NOT duplicated here):

    h_c   = Linear_chem (h_legacy)            # 48 → 24, bias
    h_a   = Linear_angle(h_legacy)            # 48 → 24, bias, INDEPENDENT
    f_c   = GINE-chem(h_c , G_chem)           # E1 math at width 24
    f_a   = Angle-MPNN(h_a, G_angle)          # E4 math at width 24
    f     = W_fuse · [ f_c ‖ f_a ]            # Linear 48 → 48, bias

Each stream reuses the EXACT E1/E4 operator classes at width 24
(:class:`~...embeddings.gin_e_chemical.GINEChemicalEmbedding` /
:class:`~...embeddings.angle_mpnn.AngleMPNNEmbedding`) through their
pure ``aggregate_delta(view)`` interface — no operator is reimplemented
here, and each view is built EXACTLY once per forward (``G_chem`` once
for the chem stream, ``G_angle`` once for the angle stream; the angle
provider's own internal ``G_chem`` build is the unchanged WP3a
behaviour).  The two input projections are SEPARATE ``nn.Linear``
modules with independent weights; the concat order is frozen as
``[f_c ‖ f_a]`` (chem block first).

Scope discipline — this module is NEW and touches nothing else:

* **views** — ONLY the two preregistered WP3a views the E1/E4 candidates
  already consume (``build_chem_graph`` / ``build_angle_triplets``); no
  new forward key, no supervision/metadata read, no F2-masked
  ``edge_attr`` slot is ever read (distances/attrs come from the current
  ``pos`` in Å / the frozen 8 chem fields);
* **no internal gate** — the output is the raw fused ``delta_node``;
  the single external ``W``/``alpha`` of plan §5.2 lives in
  :class:`~...v541.model.V541Model` and is never mirrored inside E5;
* **no weight copying** — the module is constructed fresh by the
  registry builder; ``V541Model`` builds it under
  ``torch.random.fork_rng`` with the independent
  ``embedding_init_seed`` (plan §5.4), so no parent/legacy weight is
  ever copied and the legacy RNG stream is untouched;
* **diagnostics** — ``diagnostics_enabled`` (default True) is forwarded
  verbatim to BOTH providers; ``last_diagnostics`` holds detached
  Python floats only: the shared vocabulary ``n_nodes`` / ``n_edges`` /
  ``n_triplets`` / ``zero_neighbor(_frac)`` / ``zero_angle(_frac)`` plus
  the per-source provider counts under the ``chem.`` / ``angle.``
  prefixes.  No autograd graph and no view is ever retained between
  forwards (plan §6.6.6 — views are rebuilt lazily every call);
* **inherited legality** — the angle stream inherits E4's strict
  ``j/k`` unordered-pair symmetry, the explicit ``center_has_angle``
  mask and the provider's typed ``V541_PROVIDER_ZERO_DISTANCE`` refusal;
  the chem stream inherits E1's explicit self term and direction-
  symmetric signed attrs;
* **registration** — importing this module installs the real E5 builder
  with ``register_embedding("E5", ..., replace=True)`` (the WP3 path);
  the main agent wires this file through ``embeddings/__init__.py``
  (this file does not).  Importing it also transitively imports the E1
  and E4 modules (their import-time registrations are the same real WP3
  builders).
"""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.v541.embedding_registry import register_embedding
from pes2ts_core.pesn.models.v541.embeddings.angle_mpnn import (
    AngleMPNNEmbedding,
)
from pes2ts_core.pesn.models.v541.embeddings.gin_e_chemical import (
    GINEChemicalEmbedding,
)
from pes2ts_core.pesn.models.v541.feature_providers import (
    build_angle_triplets,
    build_chem_graph,
)
from pes2ts_core.pesn.models.v541.feature_providers.angle_triplets import (
    AngleTripletView,
)
from pes2ts_core.pesn.models.v541.feature_providers.common import (
    DEFAULT_CHUNK_SIZE,
)
from pes2ts_core.pesn.models.v541.feature_providers.legal_neighbors import (
    ChemGraphView,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    NodeDeltaEmbedding,
    V541InterfaceError,
)

#: Registry id implemented by this module (frozen campaign row).
EMBEDDING_ID_E5: Final[str] = "E5"

#: Frozen §6.7 E5 branch width (GINE24 / Angle24).
BRANCH_WIDTH_E5: Final[int] = 24

#: Diagnostics prefix of the ``G_chem`` source counts.
CHEM_PREFIX: Final[str] = "chem."
#: Diagnostics prefix of the ``G_angle`` source counts.
ANGLE_PREFIX: Final[str] = "angle."


class DualStreamEmbedding(NodeDeltaEmbedding):
    """E5: Linear48→24 per stream, GINE24 ‖ Angle24, concat, fusion 48→48.

    ``forward(h_legacy, batch)`` builds ``G_chem`` and ``G_angle`` from
    the forward-view batch (lazy, no cache — §6.6.6), then evaluates the
    formula above.  :meth:`aggregate_delta` is the pure view→delta math
    (provider calls factored out) so tests can feed hand-built /
    hand-swapped views; :meth:`stream_deltas` exposes the two 24-wide
    stream outputs for stream-isolation checks.
    """

    def __init__(
            self,
            width: int = HIDDEN_V541,
            *,
            branch_width: int = BRANCH_WIDTH_E5,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            diagnostics_enabled: bool = True,
    ) -> None:
        super().__init__()
        if int(width) < 1 or int(branch_width) < 1:
            raise V541InterfaceError(
                "V541_EMBEDDING_WIDTH",
                f"width/branch_width must be >= 1, got {width!r}/{branch_width!r}")
        self.width = int(width)
        self.branch_width = int(branch_width)
        self.chunk_size = int(chunk_size)
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self._last_diagnostics: dict[str, float] = {}
        # Independent per-stream input projections (§6.7 E5: 分别 Linear).
        self.chem_in = nn.Linear(self.width, self.branch_width)
        self.angle_in = nn.Linear(self.width, self.branch_width)
        # The streams REUSE the exact E1/E4 operator classes at width 24
        # (pure aggregate_delta — no operator is duplicated here).
        self.chem_branch = GINEChemicalEmbedding(width=self.branch_width)
        self.angle_branch = AngleMPNNEmbedding(
            width=self.branch_width, chunk_size=self.chunk_size)
        # E5 owns diagnostics end-to-end; the sub-branches' forward (and
        # therefore their own recording) is never invoked.
        self.chem_branch.diagnostics_enabled = False
        self.angle_branch.diagnostics_enabled = False
        # Fusion: frozen concat order [f_c ‖ f_a] → 48.
        self.fusion = nn.Linear(2 * self.branch_width, self.width)

    @property
    def last_diagnostics(self) -> dict[str, float]:
        """Detached float diagnostics of the last forward (always a copy)."""
        return dict(self._last_diagnostics)

    # -- pure view → delta math --------------------------------------------
    def stream_deltas(
            self, h_legacy: torch.Tensor, chem_view: ChemGraphView,
            angle_view: AngleTripletView,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(f_c, f_a)`` — each ``[N, branch_width]``, views consumed once."""
        h_chem = self.chem_in(h_legacy)             # [N, 24]
        h_angle = self.angle_in(h_legacy)           # [N, 24] (independent W)
        f_chem = self.chem_branch.aggregate_delta(h_chem, chem_view)
        f_angle = self.angle_branch.aggregate_delta(h_angle, angle_view)
        return f_chem, f_angle

    def aggregate_delta(
            self, h_legacy: torch.Tensor, chem_view: ChemGraphView,
            angle_view: AngleTripletView,
    ) -> torch.Tensor:
        """``f`` from ALREADY-BUILT views (no provider call).

        Concat order is FROZEN as ``[f_c ‖ f_a]``; the fusion is a plain
        linear map — no activation, no normalisation, no gate.
        """
        f_chem, f_angle = self.stream_deltas(h_legacy, chem_view, angle_view)
        return self.fusion(torch.cat((f_chem, f_angle), dim=-1))

    # -- NodeDeltaEmbedding contract ---------------------------------------
    def forward(
            self, h_legacy: torch.Tensor,
            batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # Each view is built EXACTLY ONCE per forward and feeds its own
        # stream — no duplicated provider operator.
        chem_view = build_chem_graph(
            batch, diagnostics=self.diagnostics_enabled)
        angle_view = build_angle_triplets(
            batch, chunk_size=self.chunk_size,
            diagnostics=self.diagnostics_enabled)
        delta = self.aggregate_delta(h_legacy, chem_view, angle_view)
        if not self.diagnostics_enabled:
            self._last_diagnostics = {}
            return delta
        n_nodes = int(h_legacy.shape[0])
        # in-degree of the bidirectional chem view = incident chem pairs
        degree = torch.bincount(
            chem_view.edge_index[0], minlength=n_nodes)
        zero = int((degree == 0).sum())
        zero_angle = int((~angle_view.center_has_angle).sum())
        own: dict[str, float] = {
            "n_nodes": float(n_nodes),
            "n_edges": float(int(chem_view.edge_index.shape[1])),
            "n_triplets": float(int(angle_view.center_index.shape[0])),
            "zero_neighbor": float(zero),
            "zero_neighbor_frac": float(zero) / float(max(n_nodes, 1)),
            "zero_angle": float(zero_angle),
            "zero_angle_frac": float(zero_angle) / float(max(n_nodes, 1)),
        }
        prefixed: dict[str, float] = {
            f"{CHEM_PREFIX}{k}": float(v) for k, v in chem_view.counts.items()}
        prefixed.update({
            f"{ANGLE_PREFIX}{k}": float(v) for k, v in angle_view.counts.items()})
        self._last_diagnostics = {**prefixed, **own}
        return delta


def build_dual_stream() -> NodeDeltaEmbedding:
    """Zero-arg registry builder for E5 (48 outer / 24 per stream, frozen defaults)."""
    return DualStreamEmbedding()


# WP3 registration at import time (plan §5.1; the main agent wires this
# module through ``embeddings/__init__.py`` — this file does not).
register_embedding(EMBEDDING_ID_E5, build_dual_stream, replace=True)

__all__ = [
    "ANGLE_PREFIX",
    "BRANCH_WIDTH_E5",
    "CHEM_PREFIX",
    "EMBEDDING_ID_E5",
    "DualStreamEmbedding",
    "build_dual_stream",
]
