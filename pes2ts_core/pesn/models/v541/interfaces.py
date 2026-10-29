"""V5.4.1 model-layer interfaces (WP2, plan §5 / ADR-0032).

Frozen, forward-compatible surface that WP3 candidate embeddings build
against.  NOTHING here implements E1–E5 — those ids are RESERVED and raise
the typed refusal :class:`V541EmbeddingNotImplementedError` until their
real WP3 modules land (plan §5.1: ``painn_scalar``/E6 additionally requires
separate approval).  No placeholder/fake candidate math is shipped.

What WP2 freezes here (stable for WP3/WP4):

* **error taxonomy** — every refusal carries a stable ``code`` +
  ``detail`` (repo convention, cf. ``KeyAtomV3ContractError`` /
  ``KeyAtomV3InvalidInputError``);
* **residual-embedding contract** — :class:`NodeDeltaEmbedding`:
  ``f_phi(h_legacy, V_legal) -> delta_node [N,48]`` (or a mapping with
  ``delta_node`` plus optional ``node_vector``); the legacy 22-channel
  ``edge_scalar`` is NEVER a candidate output (plan §5.2: 化学/角度内部
  视图只影响 ``delta_node``) and ``node_vector`` is NOT consumed by the
  core this stage (E0–E5; E6 is scalar-projected diagnostics only);
* **diagnostics vocabulary** — :data:`DIAGNOSTIC_KEYS_V541`: counts and
  RMS values stored ONLY as detached Python floats, never as tensors with
  a live autograd graph;
* **parameter-name anchors** — :data:`ALPHA_PARAM_KEY` /
  :data:`W_OUT_PARAM_KEY` (the ``alpha`` scalar and the 48→48 ``W`` map),
  so WP3/WP4 tooling can address the gate without guessing names.

Input legality itself is NOT redefined here: the model reuses the frozen
v3 forward-view whitelist (``check_forward_view_keys_v3``) and the typed
invalid classes of ``StarEGNNV3`` so V541 refusals stay wire-compatible
with the R3/V5.3-MP harnesses."""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.key_atom_egnn_v3 import Z_DIM_V3

#: Registered V541 model version (campaign ``schema pes2ts_v541_campaign_v1``
#: model layer; bumped only with a new ADR — WP2 baseline).
V541_SCHEMA_VERSION: Final[str] = "pesn_key_atom_v541_model_v1"

#: Frozen node hidden width (= R3 ``StarEGNNV3`` hidden; plan §5.2 W is
#: 48→48).
HIDDEN_V541: Final[int] = 48

#: Frozen z width (re-exported for convenience; same as ``Z_DIM_V3``).
Z_DIM_V541: Final[int] = Z_DIM_V3

#: Residual gate parameter name inside the state dict.
ALPHA_PARAM_KEY: Final[str] = "alpha"
#: Residual projection W (48→48, no bias) parameter name.
W_OUT_PARAM_KEY: Final[str] = "w_out.weight"

#: Frozen E0 anchor: the joint build has exactly the legacy parameter count
#: (E0 adds ZERO parameters, plan §5.2/§5.4).
MEASURED_PARAMS_V541_E0_JOINT: Final[int] = 57_619

#: Stable diagnostic keys recorded per forward (all detached floats).
#: WP3/WP4 monitors may rely on this set; unknown extra keys are allowed
#: but these must never disappear.
DIAGNOSTIC_KEYS_V541: Final[tuple[str, ...]] = (
    "n_nodes", "n_edges", "n_graphs", "n_keys",
    "alpha", "rms_h_legacy", "rms_delta_node", "rms_alpha_W_f",
    "rms_node_vector", "node_vector_consumed",
)

#: E0 has no branch (plan §5.2: E0 数学上等价于 delta=0, 不使用极小非零
#: alpha 冒充 parity — it simply does not exist).
EMBEDDING_ID_E0: Final[str] = "E0"


# ---------------------------------------------------------------------------
# error taxonomy (typed refusals; every class carries ``code``/``detail``)
# ---------------------------------------------------------------------------
class V541InterfaceError(RuntimeError):
    """Base typed refusal of the V5.4.1 model interface layer."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class V541EmbeddingUnknownError(V541InterfaceError):
    """Embedding id outside the frozen E0–E6 registry."""


class V541EmbeddingNotImplementedError(V541InterfaceError):
    """Registered candidate id (E1–E6) without a real WP3 implementation.

    Typed refusal — never return a stand-in computation (plan §5.1: no
    placeholder candidates; parity failure must not be masked).
    """


class V541LegalityError(V541InterfaceError):
    """Candidate branch output out of the frozen residual contract."""


class V541InitError(V541InterfaceError):
    """Construction-time contract violation (e.g. missing init seed)."""


class V541LoadError(V541InterfaceError):
    """Legacy state-dict load/audit refusal (plan §5.3 strict-load cert)."""


# ---------------------------------------------------------------------------
# residual-embedding contract (plan §5.2)
# ---------------------------------------------------------------------------
class NodeDeltaEmbedding(nn.Module):
    """Base class for ``f_phi(h_legacy, V_legal)`` residual candidates.

    Contract (frozen by WP2; WP3 subclasses implement it):

    * ``forward(h_legacy, batch)`` receives the LEGACY node init
      ``h_legacy [N, 48]`` — NOT detached (``f_phi_init_from_h_legacy``;
      gradients may flow into the legacy node-init pathway) — plus the
      read-only forward-view ``batch`` mapping (whitelist already
      enforced by :class:`~pes2ts_core.pesn.models.v541.model.V541Model`);
    * the return value is either a tensor ``delta_node [N, 48]`` or a
      mapping containing ``"delta_node"`` (required) and optionally
      ``"node_vector"`` (recorded as diagnostics ONLY — the fixed core
      does not consume vectors this stage, plan §5.2 / config
      ``core_consumption.node_vector_consumed: false``);
    * candidates must NEVER return edge scalars: the core always receives
      the legacy 22-channel edge input in legacy order (a returned
      ``edge_scalar`` entry is a typed refusal);
    * the operator must be E(3)-equivariant in the backbone sense: the
      delta is an INVARIANT scalar-channel tensor (it may only depend on
      invariant features — node scalars, distances, angles — never on
      raw coordinates as vectors).
    """

    def forward(  # pragma: no cover - abstract
            self, h_legacy: torch.Tensor,
            batch: Mapping[str, torch.Tensor]) -> torch.Tensor | Mapping[str, torch.Tensor]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# diagnostics (detached scalars only — never cache autograd graphs)
# ---------------------------------------------------------------------------
def detached_rms(t: torch.Tensor) -> float:
    """RMS of ``t`` as a detached Python float (no graph retained)."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(t.detach().to(torch.float64) ** 2)).item())


def detached_scalar(t: torch.Tensor) -> float:
    """A 0-dim/1-element tensor as a detached Python float."""
    return float(t.detach().reshape(-1)[0].to(torch.float64).item())


def count_scalar(t: torch.Tensor | int) -> float:
    """A count as a Python float (diagnostics are uniformly floats)."""
    return float(t) if isinstance(t, int) else float(t.shape[0])


# ---------------------------------------------------------------------------
# branch-output normalization (typed legality of f_phi outputs)
# ---------------------------------------------------------------------------
def normalize_branch_output(
        raw: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        n_nodes: int,
        hidden: int = HIDDEN_V541,
        ref_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Validate a candidate output; return ``(delta_node, node_vector)``.

    Typed refusals (``V541LegalityError``): ``V541_BRANCH_OUTPUT``,
    ``V541_DELTA_SHAPE``, ``V541_DELTA_DTYPE``, ``V541_DELTA_NONFINITE``,
    ``V541_EDGE_SCALAR_FORBIDDEN``, ``V541_NODE_VECTOR_SHAPE``.
    """
    node_vector: torch.Tensor | None = None
    if isinstance(raw, torch.Tensor):
        delta = raw
    elif isinstance(raw, Mapping):
        if "edge_scalar" in raw:
            raise V541LegalityError(
                "V541_EDGE_SCALAR_FORBIDDEN",
                "candidates may not supply edge scalars; the core consumes "
                "the legacy 22-channel edge input in legacy order (plan §5.2)")
        if "delta_node" not in raw:
            raise V541LegalityError(
                "V541_BRANCH_OUTPUT",
                "mapping output must contain 'delta_node' [N,48]")
        delta = raw["delta_node"]
        if "node_vector" in raw and raw["node_vector"] is not None:
            node_vector = raw["node_vector"]
    else:
        raise V541LegalityError(
            "V541_BRANCH_OUTPUT",
            f"f_phi must return a tensor or a mapping, got {type(raw).__name__}")

    if not isinstance(delta, torch.Tensor):
        raise V541LegalityError(
            "V541_BRANCH_OUTPUT",
            f"delta_node must be a tensor, got {type(delta).__name__}")
    if tuple(delta.shape) != (int(n_nodes), int(hidden)):
        raise V541LegalityError(
            "V541_DELTA_SHAPE",
            f"delta_node must be [{n_nodes},{hidden}], got {tuple(delta.shape)}")
    if delta.dtype != ref_dtype:
        raise V541LegalityError(
            "V541_DELTA_DTYPE",
            f"delta_node dtype {delta.dtype} != legacy dtype {ref_dtype}")
    if not bool(torch.isfinite(delta).all()):
        raise V541LegalityError(
            "V541_DELTA_NONFINITE", "delta_node contains non-finite values")

    if node_vector is not None:
        if not isinstance(node_vector, torch.Tensor):
            raise V541LegalityError(
                "V541_BRANCH_OUTPUT",
                f"node_vector must be a tensor, got {type(node_vector).__name__}")
        if node_vector.shape[:1] != (int(n_nodes),) or node_vector.shape[-1] != 3:
            raise V541LegalityError(
                "V541_NODE_VECTOR_SHAPE",
                f"node_vector must be [N,...,3], got {tuple(node_vector.shape)}")
        if not bool(torch.isfinite(node_vector).all()):
            raise V541LegalityError(
                "V541_DELTA_NONFINITE", "node_vector contains non-finite values")
    return delta, node_vector


__all__ = [
    "ALPHA_PARAM_KEY",
    "DIAGNOSTIC_KEYS_V541",
    "EMBEDDING_ID_E0",
    "HIDDEN_V541",
    "MEASURED_PARAMS_V541_E0_JOINT",
    "V541_SCHEMA_VERSION",
    "W_OUT_PARAM_KEY",
    "Z_DIM_V541",
    "NodeDeltaEmbedding",
    "V541EmbeddingNotImplementedError",
    "V541EmbeddingUnknownError",
    "V541InitError",
    "V541InterfaceError",
    "V541LegalityError",
    "V541LoadError",
    "count_scalar",
    "detached_rms",
    "detached_scalar",
    "normalize_branch_output",
]
