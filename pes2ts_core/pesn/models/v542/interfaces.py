"""V5.4.2 unified Backbone interface (ADR-0033 §2.1; WP1 lane A).

What WP1 freezes here (stable for WP2 candidates A1–A5 and WP3+ tooling):

* :class:`BackboneOutput` — the ONLY backbone return type: a frozen,
  slotted dataclass with exactly TWO functional fields ``z: [G, 64]``
  (graph order = ``batch["graph_index"]``, tanh-saturated) and
  ``edge_msg: [E, 48]`` (the last-layer scalar message per directed input
  edge, in ``batch["edge_index"]`` column order, WITH the autograd graph).
  There is deliberately NO ``u`` (MOVE is produced by the shared
  :class:`~pes2ts_core.pesn.models.v542.decoder.DirectMoveDecoder`) and
  NO ``node_h`` (node-state diagnostics go through a separate hook, never
  through the interface);
* frozen constants — the common candidate widths (hidden 48 / z 64 /
  edge_msg 48 / node input 21), the A0 migration invariants
  (``A0_ACTIVE_PARAMS = 55_392``, ``A0_JOINT_PARAMS = 57_619``), the
  decoder/head sizes (49 / 2,178) and the 70k active-parameter hard
  budget (ADR-0033 §2.1: 参数硬上限);
* :class:`Backbone` — a minimal Protocol every candidate backbone must
  satisfy (``forward(batch) -> BackboneOutput`` + an exact
  ``active_param_count`` property) plus the
  :func:`validate_backbone_output` shape/finiteness helper that
  :class:`~pes2ts_core.pesn.models.v542.model.V542Model` runs in its
  forward (config ``interface.wrapper_checks``: dataclass does NOT do
  full isfinite checks itself);
* the error taxonomy — typed refusals carrying a stable ``code`` +
  ``detail`` (repo convention, cf. :class:`KeyAtomV3ContractError` /
  ``V541InterfaceError``).

Discipline: NOTHING here implements A1–A5; this module only freezes the
contract surface.  ``torch.compile`` is FORBIDDEN for the whole V5.4.2
model layer (ADR-0033 §2.1: hook/parity 失效) — no code in this package
calls it, and :class:`~pes2ts_core.pesn.models.v542.model.V542Model`
refuses a compiled module at construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Protocol, runtime_checkable

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.key_atom_egnn_v3 import Z_DIM_V3

#: Registered V5.4.2 model-layer version (bumped only with a new ADR).
V542_SCHEMA_VERSION: Final[str] = "pesn_key_atom_v542_model_v1"

#: Frozen node hidden width (= R3 ``StarEGNNV3`` hidden; ADR-0033 §2.1).
HIDDEN_V542: Final[int] = 48
#: Frozen z width (re-exported; same as ``Z_DIM_V3``).
Z_DIM_V542: Final[int] = Z_DIM_V3
#: Frozen edge-message width (= hidden; the decoder's ``w_v`` is 48→1).
EDGE_MSG_DIM_V542: Final[int] = 48
#: Frozen node input width (element 16 ⊕ is_key/degK/role 5 = 21).
NODE_INPUT_DIM_V542: Final[int] = 21

#: MOVE teacher reference step (Å) — non-learnable (ADR-0033 §2.1).
A_REF_V542: Final[float] = 0.08

#: Exact parameter counts (frozen migration invariants, ADR-0033 §2.1).
N_DECODER_PARAMS_V542: Final[int] = 49        # w_v = Linear(48,1): 48+1
N_PROCESS_PARAMS_V542: Final[int] = 2_178     # ReactionProcessHeadV53MP
A0_ACTIVE_PARAMS_V542: Final[int] = 55_392    # StarEGNNV3 55,441 − w_v 49
A0_JOINT_PARAMS_V542: Final[int] = 57_619     # 55,392 + 49 + 2,178
#: Hard construction-time budget for ANY candidate backbone (active params).
ACTIVE_PARAM_BUDGET_V542: Final[int] = 70_000

#: External five-key task surface of contract §3.2 (never extended).
FORWARD_KEYS_V542: Final[tuple[str, ...]] = (
    "u", "z", "q", "endpoint_logit", "order_score")
#: ``forward_move`` view keys (MOVE-only; no head math).
FORWARD_MOVE_KEYS_V542: Final[tuple[str, ...]] = ("u", "z")


# ---------------------------------------------------------------------------
# error taxonomy (typed refusals; every class carries ``code``/``detail``)
# ---------------------------------------------------------------------------
class V542InterfaceError(RuntimeError):
    """Base typed refusal of the V5.4.2 model interface layer."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class V542BackboneOutputError(V542InterfaceError):
    """Backbone output outside the frozen two-field interface."""


class V542ParamBudgetError(V542InterfaceError):
    """Parameter count violates a frozen budget/invariant."""


class V542LoadError(V542InterfaceError):
    """Strict-load audit refusal (missing/unexpected/shape/dtype)."""


class V542InitError(V542InterfaceError):
    """Construction-time contract violation (shared-module mismatch etc.)."""


class V542ForwardKeyError(V542InterfaceError):
    """Forward key outside the frozen ``FORWARD_VIEW_KEYS_V3`` whitelist."""


# ---------------------------------------------------------------------------
# the unified backbone output (ADR-0033 §2.1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BackboneOutput:
    """Unified backbone return: exactly two functional fields.

    ``z: [G, 64]`` — invariant graph summary in ``batch["graph_index"]``
    graph order, tanh-saturated (identical semantics to the legacy
    ``StarEGNNV3`` z head output).  ``edge_msg: [E, 48]`` — the LAST-layer
    scalar message of every directed input edge, row-for-row aligned with
    the columns of ``batch["edge_index"]``, carrying the live autograd
    graph (no detach: A0 training backprops through it into the backbone).

    Frozen + slotted: no ``u`` (the unified decoder owns MOVE), no
    ``node_h`` (diagnostics use a separate hook), no extra fields ever —
    field drift is a ``V542_INTERFACE`` schema violation (ADR-0033 §2.4).
    """

    z: torch.Tensor
    edge_msg: torch.Tensor

    @property
    def n_graphs(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_msg.shape[0])


def validate_backbone_output(
        output: BackboneOutput,
        *,
        n_graphs: int,
        n_edges: int,
        hidden: int = HIDDEN_V542,
        z_dim: int = Z_DIM_V542,
) -> None:
    """Typed shape/dtype/finiteness check of a :class:`BackboneOutput`.

    Config ``interface.wrapper_checks: shape_and_isfinite_in_V542Model_
    forward``: the dataclass itself does NOT run full isfinite checks;
    :meth:`V542Model.forward` calls this helper instead.  Refusals:
    ``V542_BACKBONE_OUTPUT`` (not a BackboneOutput), ``V542_BACKBONE_Z_
    SHAPE``, ``V542_BACKBONE_EDGE_SHAPE``, ``V542_BACKBONE_DTYPE``,
    ``V542_BACKBONE_NONFINITE``.
    """
    if not isinstance(output, BackboneOutput):
        raise V542BackboneOutputError(
            "V542_BACKBONE_OUTPUT",
            f"backbone must return BackboneOutput, got {type(output).__name__}")
    z, edge_msg = output.z, output.edge_msg
    if not isinstance(z, torch.Tensor) or not isinstance(edge_msg, torch.Tensor):
        raise V542BackboneOutputError(
            "V542_BACKBONE_OUTPUT",
            "BackboneOutput fields must be tensors "
            f"(z={type(z).__name__}, edge_msg={type(edge_msg).__name__})")
    if tuple(z.shape) != (int(n_graphs), int(z_dim)):
        raise V542BackboneOutputError(
            "V542_BACKBONE_Z_SHAPE",
            f"z must be [{n_graphs},{z_dim}], got {tuple(z.shape)}")
    if tuple(edge_msg.shape) != (int(n_edges), int(hidden)):
        raise V542BackboneOutputError(
            "V542_BACKBONE_EDGE_SHAPE",
            f"edge_msg must be [{n_edges},{hidden}], got {tuple(edge_msg.shape)}")
    if z.dtype != torch.float32 or edge_msg.dtype != torch.float32:
        raise V542BackboneOutputError(
            "V542_BACKBONE_DTYPE",
            f"fp32 required (z={z.dtype}, edge_msg={edge_msg.dtype})")
    if not bool(torch.isfinite(z).all()) or \
            not bool(torch.isfinite(edge_msg).all()):
        raise V542BackboneOutputError(
            "V542_BACKBONE_NONFINITE",
            "z/edge_msg contain non-finite values")


# ---------------------------------------------------------------------------
# the Backbone protocol (candidate surface for A1–A5)
# ---------------------------------------------------------------------------
@runtime_checkable
class Backbone(Protocol):
    """Minimal contract every V5.4.2 backbone satisfies.

    * ``forward(batch) -> BackboneOutput`` — consumes the frozen
      forward-view batch (``FORWARD_VIEW_KEYS_V3``) and returns the
      two-field output; input legality stays the legacy typed-invalid
      taxonomy (this protocol does not redefine it);
    * ``active_param_count`` — the EXACT number of trainable backbone
      parameters EXCLUDING any decoder-owned shared parameter (A0:
      55,392 = 55,441 − shared ``w_v`` 49).  The construction-time budget
      check (``≤ ACTIVE_PARAM_BUDGET_V542``) is each backbone's own
      responsibility and must assert the exact value, never an estimate.

    ``torch.compile`` may never be applied to a Backbone implementation
    (ADR-0033 §2.1).
    """

    def __call__(self, batch: Mapping[str, torch.Tensor]) -> BackboneOutput: ...

    @property
    def active_param_count(self) -> int: ...


def assert_active_param_budget(count: int, *, candidate_id: str) -> int:
    """Assert the frozen 70k active-parameter hard budget; return ``count``."""
    if int(count) > ACTIVE_PARAM_BUDGET_V542:
        raise V542ParamBudgetError(
            "V542_PARAM_BUDGET",
            f"candidate {candidate_id} active params {count} > "
            f"backbone_active_max {ACTIVE_PARAM_BUDGET_V542}")
    return int(count)


def assert_a0_param_invariants(*, active: int, joint: int) -> None:
    """Assert the A0 migration invariants (ADR-0033 §2.1: 迁移不变量).

    ``active == 55_392`` (StarEGNNV3 55,441 minus the 49 shared ``w_v``)
    and ``joint == 57_619`` (active + 49 decoder + 2,178 process head).
    """
    if int(active) != A0_ACTIVE_PARAMS_V542:
        raise V542ParamBudgetError(
            "V542_PARAM_BUDGET",
            f"A0 active params {active} != {A0_ACTIVE_PARAMS_V542} "
            "(StarEGNNV3 55,441 − shared w_v 49)")
    if int(joint) != A0_JOINT_PARAMS_V542:
        raise V542ParamBudgetError(
            "V542_PARAM_BUDGET",
            f"A0 joint params {joint} != {A0_JOINT_PARAMS_V542} "
            f"({A0_ACTIVE_PARAMS_V542} + {N_DECODER_PARAMS_V542} + "
            f"{N_PROCESS_PARAMS_V542})")


def assert_no_torch_compile(module: nn.Module) -> None:
    """Refuse a ``torch.compile``-wrapped module anywhere in the tree.

    ADR-0033 §2.1: torch_compile = false — dynamo breaks the temporary
    forward-hook capture and the bitwise parity harness.  The check is
    structural (``_orig_mod`` marks ``OptimizedModule``), so it catches
    compile-wrapped children too, not just the root.
    """
    for name, child in module.named_modules():
        if hasattr(child, "_orig_mod"):
            raise V542InterfaceError(
                "V542_TORCH_COMPILE_FORBIDDEN",
                f"module {name or type(child).__name__!r} is torch.compile-"
                "wrapped; V5.4.2 forbids torch.compile (ADR-0033 §2.1)")


__all__ = [
    "A0_ACTIVE_PARAMS_V542",
    "A0_JOINT_PARAMS_V542",
    "ACTIVE_PARAM_BUDGET_V542",
    "A_REF_V542",
    "EDGE_MSG_DIM_V542",
    "FORWARD_KEYS_V542",
    "FORWARD_MOVE_KEYS_V542",
    "HIDDEN_V542",
    "N_DECODER_PARAMS_V542",
    "N_PROCESS_PARAMS_V542",
    "NODE_INPUT_DIM_V542",
    "V542_SCHEMA_VERSION",
    "Z_DIM_V542",
    "Backbone",
    "BackboneOutput",
    "V542BackboneOutputError",
    "V542ForwardKeyError",
    "V542InitError",
    "V542InterfaceError",
    "V542LoadError",
    "V542ParamBudgetError",
    "assert_active_param_budget",
    "assert_a0_param_invariants",
    "assert_no_torch_compile",
    "validate_backbone_output",
]
