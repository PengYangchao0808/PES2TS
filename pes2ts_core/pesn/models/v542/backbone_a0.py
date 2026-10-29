"""V5.4.2 ``A0LegacyBackbone`` — the A0 adapter over legacy StarEGNNV3
(ADR-0033 §2.2; config ``backbone_contract.a0_migration``).

Migration discipline (NO legacy file is modified):

* the class SUBCLASSES :class:`StarEGNNV3`; the core loop is literally
  ``super().forward(batch)`` — the legacy ``u`` is computed by that call
  and then DISCARDED (MOVE now belongs to the unified
  :class:`~pes2ts_core.pesn.models.v542.decoder.DirectMoveDecoder`);
* ``edge_msg`` capture: a temporary forward hook on
  ``message_mlp[-1]`` (the last-layer ``nn.Sequential``) records its
  output — the layer-3 message tensor in INPUT edge order, shape
  ``[E, 48]``, NOT detached (autograd graph stays live).  The hook is
  removed in a ``finally`` (even on typed-invalid aborts) and the
  forward asserts it fired EXACTLY once;
* ``w_v`` is SHARED with the decoder by module object: after
  ``super().__init__()`` the legacy's own fresh ``w_v`` Linear is
  REPLACED by the decoder's module object, so ``backbone.w_v is
  decoder.w_v`` holds, ``named_parameters()`` dedupes the 49 params, and
  the optimizer can never double-update them;
* parameter accounting — the full build keeps the legacy 55,441 total
  (the shared ``w_v`` still lives under ``backbone.`` for state-dict
  continuity); the ADR's migration invariant is
  ``active_param_count = 55,441 − 49 = 55,392`` (decoder-owned params
  excluded) and ``active + 49 + 2,178 == 57,619`` joint
  (``assert_a0_joint_invariants``).

Forward output is the frozen two-field
:class:`~pes2ts_core.pesn.models.v542.interfaces.BackboneOutput`
(``z`` as-is from the legacy head; ``edge_msg`` from the hook)."""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.contracts.key_atom_v3 import EDGE_INDEX_KEY
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
from pes2ts_core.pesn.models.v542.decoder import (
    DirectMoveDecoder,
    assert_decoder_a_ref_frozen,
)
from pes2ts_core.pesn.models.v542.interfaces import (
    A0_ACTIVE_PARAMS_V542,
    A0_JOINT_PARAMS_V542,
    HIDDEN_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    V542InitError,
    V542InterfaceError,
    BackboneOutput,
    assert_a0_param_invariants,
)

#: Registered A0 adapter version (campaign backbone family A0).
A0_BACKBONE_VERSION: Final[str] = "pesn_key_atom_v542_a0_adapter_v1"

#: Full legacy parameter count of ``StarEGNNV3`` (E2-frozen).
_LEGACY_TOTAL_PARAMS: Final[int] = 55_441


class A0LegacyBackbone(StarEGNNV3):
    """A0 adapter: legacy Star 3-layer EGNN behind the unified interface.

    Args:
        move_decoder: the SHARED decoder whose ``w_v`` module object this
            backbone adopts (``self.w_v is move_decoder.w_v``).  Required —
            constructing an A0 backbone without the decoder would silently
            fork MOVE ownership and break the param-group/id invariants.
            The decoder's ``a_ref`` must be the FROZEN constant 0.08
            (gate-1 B1 typed refusal otherwise).
    """

    #: Registered adapter version.
    v542_version: Final[str] = A0_BACKBONE_VERSION

    def __init__(self, move_decoder: DirectMoveDecoder) -> None:
        if not isinstance(move_decoder, DirectMoveDecoder):
            raise V542InitError(
                "V542_A0_DECODER",
                "A0LegacyBackbone requires the shared DirectMoveDecoder "
                f"(got {type(move_decoder).__name__})")
        # Gate-1 B1: the adopted decoder must carry the FROZEN a_ref=0.08.
        assert_decoder_a_ref_frozen(move_decoder)
        # Legacy construction FIRST (identical submodule names / RNG
        # consumption / parameter set as StarEGNNV3).
        super().__init__()
        # Adopt the decoder's w_v module OBJECT (not a copy): the fresh
        # legacy Linear allocated by super().__init__() is dropped and the
        # shared object takes its slot, so state-dict key "backbone.w_v.*"
        # stays valid while ``backbone.w_v is decoder.w_v`` holds.
        self.w_v = move_decoder.w_v
        assert self.w_v is move_decoder.w_v

        total = sum(int(p.numel()) for p in self.parameters())
        if total != _LEGACY_TOTAL_PARAMS:
            raise V542InitError(
                "V542_A0_PARAMS",
                f"A0 full build has {total} params, legacy total is "
                f"{_LEGACY_TOTAL_PARAMS}")
        assert_a0_param_invariants(
            active=self.active_param_count,
            joint=self.active_param_count + N_DECODER_PARAMS_V542
            + N_PROCESS_PARAMS_V542)

    # -- accounting ---------------------------------------------------------
    @property
    def active_param_count(self) -> int:
        """Trainable backbone params EXCLUDING the decoder-owned ``w_v``.

        A0 migration invariant: 55,441 legacy total − 49 shared ``w_v``
        = **55,392** (ADR-0033 §2.1).  The 49 ``w_v`` params are counted
        ONCE, in the decoder's group.
        """
        return (sum(int(p.numel()) for p in self.parameters())
                - N_DECODER_PARAMS_V542)

    # -- unified forward ----------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> BackboneOutput:
        """Legacy forward verbatim + ``edge_msg`` hook capture → the interface.

        The legacy ``{"u", "z"}`` return is consumed here: ``z`` passes
        through, ``u`` is compute-and-discard (the unified decoder owns
        MOVE; ADR-0033 §2.2).  The temporary hook fires exactly once
        (the last-layer Sequential runs once per backbone forward) and is
        removed in ``finally`` — including on typed-invalid aborts.
        """
        captured: list[torch.Tensor] = []

        def _capture(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            # No detach: edge_msg keeps the autograd graph (training
            # backprops MOVE through the decoder into the backbone).
            captured.append(output)

        handle = self.message_mlp[-1].register_forward_hook(_capture)
        try:
            legacy_out = super().forward(batch)   # core loop untouched
        finally:
            handle.remove()

        if len(captured) != 1:
            # message_mlp[-1] must run exactly once per forward; zero or
            # multiple captures mean the legacy core loop drifted.
            raise V542InterfaceError(
                "V542_A0_HOOK",
                f"edge_msg hook fired {len(captured)} times, expected exactly 1")
        edge_msg = captured[0]

        n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
        if tuple(edge_msg.shape) != (n_edges, HIDDEN_V542):
            raise V542InterfaceError(
                "V542_A0_EDGE_MSG",
                f"captured edge_msg has shape {tuple(edge_msg.shape)}, "
                f"expected ({n_edges}, {HIDDEN_V542}) in input edge order")

        return BackboneOutput(z=legacy_out["z"], edge_msg=edge_msg)


def assert_a0_joint_invariants(backbone: A0LegacyBackbone) -> dict[str, int]:
    """Prove ``active + 49 + 2,178 == 57,619`` on a live A0 backbone.

    Returns the auditable counts dict ``{"active", "decoder", "process",
    "joint"}``; raises :class:`V542ParamBudgetError`-typed refusals via
    :func:`assert_a0_param_invariants` on any drift.
    """
    if not isinstance(backbone, A0LegacyBackbone):
        raise V542InitError(
            "V542_A0_JOINT",
            f"expected A0LegacyBackbone, got {type(backbone).__name__}")
    active = backbone.active_param_count
    joint = (active + N_DECODER_PARAMS_V542 + N_PROCESS_PARAMS_V542)
    assert_a0_param_invariants(active=active, joint=joint)
    return {
        "active": int(active),
        "decoder": int(N_DECODER_PARAMS_V542),
        "process": int(N_PROCESS_PARAMS_V542),
        "joint": int(joint),
    }


__all__ = [
    "A0_BACKBONE_VERSION",
    "A0LegacyBackbone",
    "assert_a0_joint_invariants",
]
