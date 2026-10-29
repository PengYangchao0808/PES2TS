"""V5.4.1 WP3b — shared primitives for the geometric candidates E3/E4.

Bounded scope (plan §5.1 layout; this file is NEW): it only adds
``pes2ts_core/pesn/models/v541/embeddings/geometric_common.py``.  It does
not modify ``V541Model``, the embedding registry, the WP3a feature
providers, config, docs or any legacy file, and it ships no E1/E2/E5 math.

Contents (all consumed by ``schnet_spatial.py`` (E3) and
``angle_mpnn.py`` (E4)):

* :func:`cosine_envelope` — the frozen §6.7 E3 envelope
  ``c(r) = 0.5·(cos(π·r/4.5) + 1)`` for ``r < 4.5 Å`` and EXACTLY ``0``
  for ``r ≥ 4.5 Å`` (the G_spatial tie-retention can emit pairs a hair
  beyond the cutoff, so the ``r ≥ 4.5`` branch must be explicit);
* :func:`two_layer_mlp` — the frozen §6.7 default MLP shape
  ``Linear(in, width) → ReLU → Linear(width, out)`` (hidden width, ReLU
  in the middle, LINEAR output, no dropout/normalisation);
* :func:`scatter_sum` — bounded, graph-local ``index_add_`` aggregation
  (per-node rows only; never mixes graphs);
* :class:`GeometricEmbeddingBase` — the ``NodeDeltaEmbedding`` base with
  the WP3a diagnostics switch (``diagnostics_enabled``, forwarded to the
  providers) and a ``last_diagnostics`` mapping of DETACHED Python
  floats.  No autograd graph is ever retained by a diagnostic (the
  ``v541.interfaces`` convention).

Units: distances are the CURRENT frame in Å exactly as the providers
recompute them from ``pos``; nothing here rescales coordinates and
nothing here reads the F2-masked ``edge_attr`` RBF slots.
"""

from __future__ import annotations

import math
from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.v541.feature_providers.common import (
    DEFAULT_CHUNK_SIZE,
    RBF_N,
    RBF_CENTER_END,
)
from pes2ts_core.pesn.models.v541.interfaces import (
    HIDDEN_V541,
    NodeDeltaEmbedding,
)

#: Frozen cosine-envelope cutoff (Å) — identical to the legacy spatial
#: cutoff and the RBF centre span (spec v3 ``RBF_CUTOFF`` = 4.5 Å).
ENVELOPE_CUTOFF: Final[float] = RBF_CENTER_END          # 4.5
#: Frozen cosine-envelope scale: ``π / cutoff`` inside ``cos``.
ENVELOPE_PI: Final[float] = math.pi
#: Frozen cos-basis width for E4: ``[1, c, c², c³]`` (plan §6.7 E4).
COS_BASIS_DIM: Final[int] = 4


def cosine_envelope(
        distance: torch.Tensor,
        cutoff: float = ENVELOPE_CUTOFF,
) -> torch.Tensor:
    """``c(r) = 0.5·(cos(π·r/cutoff) + 1)`` for ``r < cutoff``, else ``0``.

    The E3 continuous filter weight (plan §6.7).  Properties frozen here
    and asserted by the WP3b tests:

    * ``c(0) = 1``, ``c(cutoff/2) = 0.5``, ``c(r ≥ cutoff) = 0`` EXACTLY
      (the ``r ≥ cutoff`` branch is a hard zero, not a cosine tail);
    * continuous at ``r = cutoff`` (the cosine itself is 0 there), so the
      discrete ``r < 4.5 Å`` neighbour cut introduces no jump beyond the
      plan's documented ``[R8]`` caveat;
    * the cosine branch is evaluated on ``min(r, cutoff)`` so the
      backward pass of an at/above-cutoff pair never differentiates
      through a clamped argument (gradient exactly 0 there, no NaN);
    * finite for every finite input; dtype/device follow ``distance``.
    """
    r = distance
    bounded = r.clamp(max=float(cutoff))
    inside = 0.5 * (torch.cos(ENVELOPE_PI * bounded / float(cutoff)) + 1.0)
    return torch.where(r < float(cutoff), inside, torch.zeros_like(inside))


def two_layer_mlp(in_dim: int, width: int, out_dim: int) -> nn.Sequential:
    """The frozen §6.7 default MLP: ``Linear→ReLU→Linear`` (no extras).

    ``width`` is the hidden width (48 for E3/E4; E5 reuses 24), the
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
    zero-neighbour / zero-angle cases).  Rows only ever address nodes of
    THIS batch, so no aggregation can cross a graph boundary.
    """
    out = values.new_zeros((int(n_nodes), int(values.shape[-1])))
    out.index_add_(0, index, values)
    return out


class GeometricEmbeddingBase(NodeDeltaEmbedding):
    """Common ``NodeDeltaEmbedding`` surface for E3/E4.

    * ``diagnostics_enabled`` (bool, default ``True``) — the WP3a switch,
      forwarded verbatim to every provider call.  ``False`` ⇒ the
      providers skip every count-only host sync AND this base records
      ``last_diagnostics == {}`` (no count is computed here either);
    * ``last_diagnostics`` — a COPY of the detached Python-float mapping
      recorded by the LAST forward (provider ``counts`` plus the
      candidate's own keys such as ``zero_neighbor`` / ``zero_angle``).
      Plain floats only: no tensor, no autograd graph is ever stored.
    """

    def __init__(
            self,
            *,
            width: int = HIDDEN_V541,
            chunk_size: int = DEFAULT_CHUNK_SIZE,
            diagnostics_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.chunk_size = int(chunk_size)
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
        merged: dict[str, float] = {k: float(v) for k, v in provider_counts.items()}
        merged.update({k: float(v) for k, v in own.items()})
        self._last_diagnostics = merged


__all__ = [
    "COS_BASIS_DIM",
    "ENVELOPE_CUTOFF",
    "ENVELOPE_PI",
    "GeometricEmbeddingBase",
    "RBF_N",
    "cosine_envelope",
    "scatter_sum",
    "two_layer_mlp",
]
