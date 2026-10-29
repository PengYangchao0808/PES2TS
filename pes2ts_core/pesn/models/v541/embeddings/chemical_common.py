"""V5.4.1 WP3b — shared primitives for the chemical candidates E1/E2.

Bounded scope (plan §5.1 layout; this file is NEW): it only adds
``pes2ts_core/pesn/models/v541/embeddings/chemical_common.py``.  It does
not modify ``V541Model``, the embedding registry, the WP3a feature
providers, config, docs or any legacy file, and it ships no E3/E4/E5/E6
math.

Contents (consumed by ``gin_e_chemical.py`` (E1) and ``pna_chemical.py``
(E2)):

* :func:`two_layer_mlp` — the frozen §6.7 default MLP shape
  ``Linear(in, width) → ReLU → Linear(width, out)`` (hidden width, ReLU
  in the middle, LINEAR output, no dropout/normalisation).  Kept local
  (byte-identical to the geometric twin) so the chemical candidates do
  not depend on another candidate's module;
* :func:`scatter_sum` — bounded, graph-local ``index_add_`` aggregation
  (per-node rows only; never mixes graphs);
* :class:`ChemicalEmbeddingBase` — the ``NodeDeltaEmbedding`` base with
  the WP3a diagnostics switch (``diagnostics_enabled``, forwarded to
  ``build_chem_graph``) and a ``last_diagnostics`` mapping of DETACHED
  Python floats.  No autograd graph is ever retained by a diagnostic
  (the ``v541.interfaces`` convention) and no view/tensor is cached
  between forwards (plan §6.6.6: rebuild lazily every call);
* the typed degree-stats refusals shared by E2 (and reusable by WP4):
  :class:`V541DegreeStatsMissingError` (``V541_DEGREE_STATS_MISSING``)
  and :class:`V541DegreeStatsInvalidError`
  (``V541_DEGREE_STATS_INVALID``), both subclasses of the frozen WP2
  ``V541InterfaceError`` taxonomy.

The E1/E2 candidates read ONLY the frozen 8-field chem-edge attributes
of the WP3a ``G_chem`` view (config ``chem_edge_fields``) — never a
distance, never an F2-masked ``edge_attr`` slot, never ``pos`` — so
their ``delta_node`` is an invariant scalar channel (E(3)-safe by
construction).
"""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.v541.feature_providers.common import CHEM_EDGE_DIM
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    NodeDeltaEmbedding,
    V541InterfaceError,
)

#: Frozen chem-edge width (WP3a fixture freeze; config ``chem_edge_dim``).
CHEM_EDGE_FIELDS_DIM: Final[int] = CHEM_EDGE_DIM          # 8


# ---------------------------------------------------------------------------
# typed refusals (degree stats — E2 / WP4 loader surface)
# ---------------------------------------------------------------------------
class V541DegreeStatsMissingError(V541InterfaceError):
    """E2 PNA forward before ``set_degree_stats`` (plan §6.7 / §6.6)."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "V541_DEGREE_STATS_MISSING",
            detail or ("E2 PNA degree scalers need frozen inner-train "
                       "degree stats; call set_degree_stats(mu_log, "
                       "stats_sha256) after loading — forward never "
                       "computes stats itself (plan §6.7)"))


class V541DegreeStatsInvalidError(V541InterfaceError):
    """Degree stats present but unusable (mu_log non-finite / ≤ 0, or a
    malformed provenance digest) — e.g. a tampered checkpoint."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "V541_DEGREE_STATS_INVALID",
            detail or ("degree stats are set but invalid: mu_log must be "
                       "finite and > 0 (mean of log1p(degree) over a "
                       "training population with at least one edge) and "
                       "stats_sha256 must be a 64-hex digest"))


# ---------------------------------------------------------------------------
# frozen §6.7 building blocks
# ---------------------------------------------------------------------------
def two_layer_mlp(in_dim: int, width: int, out_dim: int) -> nn.Sequential:
    """The frozen §6.7 default MLP: ``Linear→ReLU→Linear`` (no extras).

    ``width`` is the hidden width (48 for E1/E2; E5 reuses 24), the
    middle activation is ReLU and the OUTPUT layer is LINEAR (no
    activation, no dropout, no BatchNorm — plan §6.7 公共项).
    """
    return nn.Sequential(
        nn.Linear(int(in_dim), int(width)),
        nn.ReLU(),
        nn.Linear(int(width), int(out_dim)),
    )


def scatter_sum(
        n_nodes: int, index: torch.Tensor, values: torch.Tensor,
) -> torch.Tensor:
    """``out[i] = sum_{rows with index == i} values`` → ``[n_nodes, width]``.

    Pure ``index_add_`` over a fresh zero tensor: differentiable,
    permutation-equivariant, and a no-op for an empty ``index`` (the
    zero-neighbour case).  Rows only ever address nodes of THIS batch,
    so no aggregation can cross a graph boundary.
    """
    out = values.new_zeros((int(n_nodes), int(values.shape[-1])))
    out.index_add_(0, index, values)
    return out


# ---------------------------------------------------------------------------
# diagnostics base (mirrors the WP3a switch semantics of the geometric base)
# ---------------------------------------------------------------------------
class ChemicalEmbeddingBase(NodeDeltaEmbedding):
    """Common ``NodeDeltaEmbedding`` surface for E1/E2.

    * ``diagnostics_enabled`` (bool, default ``True``) — the WP3a switch,
      forwarded verbatim to ``build_chem_graph``.  ``False`` ⇒ the
      provider skips every count-only host sync AND this base records
      ``last_diagnostics == {}`` (no count is computed here either);
    * ``last_diagnostics`` — a COPY of the detached Python-float mapping
      recorded by the LAST forward (provider ``counts`` plus the
      candidate's own keys such as ``n_edges`` / ``zero_neighbor_frac``).
      Plain floats only: no tensor, no autograd graph is ever stored and
      no view is cached between forwards.
    """

    def __init__(
            self,
            *,
            width: int = HIDDEN_V541,
            diagnostics_enabled: bool = True,
    ) -> None:
        super().__init__()
        if int(width) < 1:
            raise V541InterfaceError(
                "V541_EMBEDDING_WIDTH", f"width must be >= 1, got {width!r}")
        self.width = int(width)
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self._last_diagnostics: dict[str, float] = {}

    @property
    def last_diagnostics(self) -> dict[str, float]:
        """Detached float diagnostics of the last forward (always a copy)."""
        return dict(self._last_diagnostics)

    def _record_diagnostics(
            self,
            provider_counts: Mapping[str, float],
            **own: float,
    ) -> None:
        """Store the detached float diagnostics of THIS forward.

        ``provider_counts`` is the provider's ``counts`` mapping (empty
        when ``diagnostics_enabled`` is False); ``own`` are the
        candidate's own count keys.  Everything is coerced to ``float``
        so no tensor — and therefore no autograd graph — can be retained.
        """
        if not self.diagnostics_enabled:
            self._last_diagnostics = {}
            return
        merged: dict[str, float] = {
            k: float(v) for k, v in provider_counts.items()}
        merged.update({k: float(v) for k, v in own.items()})
        self._last_diagnostics = merged


__all__ = [
    "CHEM_EDGE_FIELDS_DIM",
    "ChemicalEmbeddingBase",
    "V541DegreeStatsInvalidError",
    "V541DegreeStatsMissingError",
    "scatter_sum",
    "two_layer_mlp",
]
