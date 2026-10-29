"""V5.4.1 WP3b — E2 ``pna_chemical``: ONE PNA layer over the legal
``G_chem`` view (plan §6.7 E2 / §6.6.4; registry family ``pna_chemical``,
id ``E2``).

Formula — frozen §6.7, width = :data:`~...interfaces.HIDDEN_V541` = 48,
exactly ONE message update, ``towers = 1``::

    P_e    : linear 8 → width                        (chem-edge projection)
    m_ji   = MLP([ h_i , h_j , P_e e_ji ]) → width   (i = receiver, j = neighbour)
    agg    ∈ { mean, max, min, std(population) } over incoming m_ji
    scaler ∈ { identity, amplification, attenuation } per node degree d
    feat   = concat over (agg × scaler) ⊕ self h      → 13·width
    f      = MLP(feat) → width

Degree scalers (frozen inner-train statistics, plan §6.7)::

    identity    = 1
    amplification = log1p(d) / mu_log     for d > 0, else 1
    attenuation   = mu_log / log1p(d)     for d > 0, else 1   (reciprocal)

with ``mu_log = mean_inner_train(log1p(d))`` computed OFFLINE only (see
``pes2ts_core/pesn/offline/training/v541_degree_stats.py``).  This module
NEVER computes stats in forward and NEVER reads a split: the frozen
values arrive through :meth:`set_degree_stats`, whose ``mu_log`` /
``stats_sha256`` are registered BUFFERS, so both the number and its
provenance survive ``state_dict`` round-trips (plan §7.3).  Refusals:

* forward before ``set_degree_stats`` → typed
  ``V541_DEGREE_STATS_MISSING``;
* ``set_degree_stats`` (or a tampered checkpoint) with a non-finite /
  ≤ 0 ``mu_log`` or a malformed digest → typed
  ``V541_DEGREE_STATS_INVALID``;
* an all-edgeless training population is refused by the OFFLINE helper
  (``V541_DEGREE_STATS_NO_EDGES``) — ``mu_log`` would be 0 and the
  amplification scaler undefined.

Aggregation details (frozen, §6.8 row 3):

* zero-neighbour node → mean/max/min/std ALL exactly 0 and the SELF
  feature is preserved (no learned missing-neighbour token);
* single-neighbour node → population std is exactly 0 (guarded sqrt:
  ``var`` is clamped at 0 and the ``var > 0`` branch is selected with a
  safe stand-in, so the ``1/(2√0)`` singularity never reaches the
  backward graph → finite gradients);
* ``max``/``min`` use ``scatter_reduce(..., include_self=False)`` over a
  zeros init: nodes WITH messages get the true extremum of their
  messages only, nodes WITHOUT messages keep the zeros init.

Scope discipline — this module is NEW and touches nothing else: view is
ONLY the WP3a ``G_chem`` provider, inputs are only the frozen 8-field
chem attributes (never ``pos``, never an F2-masked slot ⇒ invariant
scalar delta), math is pure PyTorch with no dropout/BN/pooling/coordinate
update, ``diagnostics_enabled`` is forwarded to the provider and
``last_diagnostics`` holds detached Python floats only (no view/graph is
cached between forwards).  Importing this module installs the real E2
builder with ``register_embedding("E2", ..., replace=True)``.
"""

from __future__ import annotations

import math
from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.v541.embedding_registry import register_embedding
from pes2ts_core.pesn.models.v541.embeddings.chemical_common import (
    ChemicalEmbeddingBase,
    V541DegreeStatsInvalidError,
    V541DegreeStatsMissingError,
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
EMBEDDING_ID_E2: Final[str] = "E2"

#: Frozen aggregator order (plan §6.7: mean/max/min/std).
AGGREGATORS: Final[tuple[str, ...]] = ("mean", "max", "min", "std")
#: Frozen scaler order (plan §6.7: identity/amplification/attenuation).
SCALERS: Final[tuple[str, ...]] = ("identity", "amplification", "attenuation")

#: PNA towers — frozen to 1 (single-tower combination only, §6.7).
PNA_TOWERS: Final[int] = 1


def _population_std(
        total: torch.Tensor, total_sq: torch.Tensor, count: torch.Tensor,
) -> torch.Tensor:
    """Guarded population std from per-node sums.

    ``var = E[x²] − (E[x])²`` clamped at 0; the sqrt only evaluates on
    the ``var > 0`` branch (a safe stand-in feeds the other branch), so
    the ``1/(2√0)`` singularity never enters the backward graph and a
    single-neighbour node yields EXACTLY 0 with a finite gradient.
    """
    denom = count.clamp(min=1).unsqueeze(-1)
    mean = total / denom
    mean_sq = total_sq / denom
    var = (mean_sq - mean * mean).clamp(min=0.0)
    positive = var > 0
    safe_var = torch.where(positive, var, torch.ones_like(var))
    return torch.where(positive, torch.sqrt(safe_var), torch.zeros_like(var))


class PNAChemicalEmbedding(ChemicalEmbeddingBase):
    """E2: one PNA layer → ``delta_node`` (stats supplied, never fitted)."""

    def __init__(
            self,
            width: int = HIDDEN_V541,
            *,
            diagnostics_enabled: bool = True,
    ) -> None:
        super().__init__(
            width=width, diagnostics_enabled=diagnostics_enabled)
        if PNA_TOWERS != 1:                      # pragma: no cover - frozen
            raise AssertionError("frozen towers=1")
        # P_e: linear projection of the frozen 8-field chem-edge attrs.
        self.pe_proj = nn.Linear(CHEM_EDGE_DIM, self.width)
        # message MLP: [h_i, h_j, P_e e] → width → width.
        self.msg_mlp = two_layer_mlp(3 * self.width, self.width, self.width)
        # readout: 12 (agg × scaler) blocks ⊕ self → width → width.
        self.readout = two_layer_mlp(
            (len(AGGREGATORS) * len(SCALERS) + 1) * self.width,
            self.width, self.width)
        # frozen degree statistics (offline inner-train fit; buffers so
        # both the value and its provenance checkpoint, plan §7.3)
        self.register_buffer(
            "mu_log", torch.tensor(float("nan"), dtype=torch.float32))
        self.register_buffer(
            "stats_sha256", torch.zeros(32, dtype=torch.uint8))
        self.register_buffer(
            "stats_ready", torch.tensor(0, dtype=torch.uint8))
        # host-side mirrors: keep the per-forward state check free of any
        # GPU→host sync (buffers stay the checkpointed source of truth;
        # mirrors refresh at set/load time only).
        self._stats_state = "unset"               # unset | ready | invalid
        self._mu_log_value = float("nan")

    # -- frozen stats intake (the ONLY writer besides load) -----------------
    @staticmethod
    def _digest_bytes(stats_sha256: object) -> bytes:
        if isinstance(stats_sha256, str):
            text = stats_sha256.strip().lower()
            if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
                raise V541DegreeStatsInvalidError(
                    f"stats_sha256 must be 64 hex chars, got "
                    f"{len(stats_sha256)!r} chars")
            return bytes.fromhex(text)
        if isinstance(stats_sha256, (bytes, bytearray, memoryview)):
            raw = bytes(stats_sha256)
            if len(raw) != 32:
                raise V541DegreeStatsInvalidError(
                    f"stats_sha256 bytes must be 32 long, got {len(raw)}")
            return raw
        if isinstance(stats_sha256, torch.Tensor):
            flat = stats_sha256.detach().reshape(-1).to(torch.uint8)
            if int(flat.numel()) != 32:
                raise V541DegreeStatsInvalidError(
                    f"stats_sha256 tensor must hold 32 bytes, got "
                    f"{int(flat.numel())}")
            return bytes(flat.cpu().tolist())
        raise V541DegreeStatsInvalidError(
            f"stats_sha256 must be hex str / 32 bytes, got "
            f"{type(stats_sha256).__name__}")

    @staticmethod
    def _validated_mu(mu_log: object) -> float:
        try:
            mu = float(mu_log)                 # 0-dim tensors OK
        except (TypeError, ValueError) as exc:
            raise V541DegreeStatsInvalidError(
                f"mu_log must be a real number, got {mu_log!r}") from exc
        if not math.isfinite(mu) or mu <= 0.0:
            raise V541DegreeStatsInvalidError(
                f"mu_log must be finite and > 0 (mean of log1p(degree) "
                f"over a population with at least one edge), got {mu!r}")
        return mu

    def _refresh_stats_mirrors(self) -> None:
        """Re-derive the host-side state from the (possibly loaded) buffers."""
        ready = bool(int(self.stats_ready))
        mu = float(self.mu_log)
        sha_set = bool(self.stats_sha256.any())
        if not ready:
            self._stats_state = "unset"
            self._mu_log_value = float("nan")
        elif math.isfinite(mu) and mu > 0.0 and sha_set:
            self._stats_state = "ready"
            self._mu_log_value = mu
        else:
            self._stats_state = "invalid"
            self._mu_log_value = mu

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        # Buffers are the checkpointed truth; refresh the host mirrors so
        # a plain state_dict round-trip re-arms the module without a
        # mandatory set_degree_stats re-call.
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self._refresh_stats_mirrors()

    def set_degree_stats(self, mu_log: object, stats_sha256: object) -> None:
        """Freeze the inner-train degree statistics (plan §6.7 / §6.6).

        ``mu_log`` must be finite and > 0; ``stats_sha256`` a 64-hex
        digest (or its 32 raw bytes).  Invalid input is a typed refusal
        and leaves any previously set stats untouched.
        """
        mu = self._validated_mu(mu_log)
        digest = self._digest_bytes(stats_sha256)
        self.mu_log.fill_(mu)
        self.stats_sha256.copy_(
            torch.tensor(list(digest), dtype=torch.uint8))
        self.stats_ready.fill_(1)
        self._refresh_stats_mirrors()

    @property
    def degree_stats_state(self) -> str:
        """``"unset" | "ready" | "invalid"`` (host-side, no sync)."""
        return self._stats_state

    @property
    def degree_stats_provenance(self) -> dict[str, object]:
        """Detached provenance snapshot (mu_log + hex digest + state)."""
        digest = bytes(self.stats_sha256.cpu().tolist())
        return {
            "state": self._stats_state,
            "mu_log": self._mu_log_value,
            "stats_sha256": digest.hex() if any(digest) else None,
        }

    def _require_stats(self) -> float:
        if self._stats_state == "unset":
            raise V541DegreeStatsMissingError()
        if self._stats_state == "invalid":
            raise V541DegreeStatsInvalidError(
                f"degree stats buffers are set but invalid "
                f"(mu_log={self._mu_log_value!r})")
        return self._mu_log_value

    # -- pure view → delta math --------------------------------------------
    @staticmethod
    def _degree_scalers(
            degree: torch.Tensor, mu_log: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(identity, amplification, attenuation)`` per node.

        ``d = 0`` explicitly yields (1, 1, 1); the attenuation
        denominator is swapped to 1 where ``d == 0`` so no ``x/0``
        intermediate ever exists (plan §6.7 d=0 rule).
        """
        positive = degree > 0
        ones = torch.ones_like(degree)
        log1p_d = torch.log1p(degree)
        safe_log = torch.where(positive, log1p_d, torch.ones_like(log1p_d))
        identity = ones
        amplification = torch.where(positive, log1p_d / mu_log, ones)
        attenuation = torch.where(positive, mu_log / safe_log, ones)
        return identity, amplification, attenuation

    def aggregate_delta(
            self, h_legacy: torch.Tensor, view: ChemGraphView,
            mu_log: float,
    ) -> torch.Tensor:
        """``f`` from an ALREADY-BUILT ``G_chem`` view (no provider call).

        Message direction convention matches the other candidates:
        ``edge_index[0]`` receives from ``edge_index[1]``; the view is
        bidirectional with bitwise-identical attributes, so each node
        aggregates each neighbour exactly once regardless of the naming.
        """
        n_nodes = int(h_legacy.shape[0])
        width = self.width
        receiver = view.edge_index[0]
        neighbour = view.edge_index[1]
        pe = self.pe_proj(view.edge_attr.to(dtype=h_legacy.dtype))  # [E, w]
        # m_ji = MLP([h_i, h_j, P_e e_ji]) — receiver first (§6.7)
        message = self.msg_mlp(torch.cat(
            (h_legacy[receiver], h_legacy[neighbour], pe), dim=-1))  # [E, w]

        count = torch.bincount(receiver, minlength=n_nodes)          # [n]
        total = scatter_sum(n_nodes, receiver, message)
        total_sq = scatter_sum(n_nodes, receiver, message * message)
        denom = count.clamp(min=1).unsqueeze(-1).to(message.dtype)
        mean = total / denom
        if message.numel():
            index = receiver.unsqueeze(-1).expand_as(message)
            zeros = message.new_zeros((n_nodes, width))
            maximum = zeros.scatter_reduce(
                0, index, message, reduce="amax", include_self=False)
            minimum = zeros.scatter_reduce(
                0, index, message, reduce="amin", include_self=False)
        else:
            maximum = message.new_zeros((n_nodes, width))
            minimum = message.new_zeros((n_nodes, width))
        std = _population_std(total, total_sq, count.to(message.dtype))
        aggregates = {"mean": mean, "max": maximum,
                      "min": minimum, "std": std}

        degree = count.to(message.dtype)
        identity, amplification, attenuation = self._degree_scalers(
            degree, float(mu_log))
        scalers = {"identity": identity, "amplification": amplification,
                   "attenuation": attenuation}

        blocks: list[torch.Tensor] = []
        for agg_name in AGGREGATORS:
            agg = aggregates[agg_name]
            for scaler_name in SCALERS:
                blocks.append(agg * scalers[scaler_name].unsqueeze(-1))
        blocks.append(h_legacy)                     # explicit self feature
        features = torch.cat(blocks, dim=-1)        # [n, 13·width]
        return self.readout(features)

    # -- NodeDeltaEmbedding contract ---------------------------------------
    def forward(
            self, h_legacy: torch.Tensor,
            batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # typed refusal BEFORE any provider work — stats are never fitted
        # in forward and no split is ever read here (plan §6.7 / §6.6).
        mu_log = self._require_stats()
        view = build_chem_graph(
            batch, diagnostics=self.diagnostics_enabled)
        delta = self.aggregate_delta(h_legacy, view, mu_log)
        own: dict[str, float] = {}
        if self.diagnostics_enabled:
            degree = torch.bincount(
                view.edge_index[0], minlength=int(h_legacy.shape[0]))
            zero = int((degree == 0).sum())
            n_nodes = int(h_legacy.shape[0])
            own = {
                "n_nodes": float(n_nodes),
                "n_edges": float(int(view.edge_index.shape[1])),
                "zero_neighbor": float(zero),
                "zero_neighbor_frac": float(zero) / float(max(n_nodes, 1)),
                "mu_log": float(mu_log),
            }
        self._record_diagnostics(view.counts, **own)
        return delta


def build_pna_chemical() -> NodeDeltaEmbedding:
    """Zero-arg registry builder for E2 (width 48, frozen defaults)."""
    return PNAChemicalEmbedding()


# WP3 registration at import time (plan §5.1; the main agent wires this
# module through ``embeddings/__init__.py`` — this file does not).
register_embedding(EMBEDDING_ID_E2, build_pna_chemical, replace=True)

__all__ = [
    "AGGREGATORS",
    "EMBEDDING_ID_E2",
    "PNA_TOWERS",
    "PNAChemicalEmbedding",
    "SCALERS",
    "build_pna_chemical",
]
