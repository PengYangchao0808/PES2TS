"""V5.4.2 ``V542Model`` — backbone + unified MOVE decoder + shared PROCESS
head (ADR-0033 §2.1; config ``backbone_contract``).

Assembly (frozen)::

    batch ─► backbone (A0LegacyBackbone) ─► BackboneOutput(z, edge_msg)
                                              │                │
                                              ▼                ▼
                                    process_head (z)   move_decoder (edge_msg,…)
                                      q/endpoint/order      u

* ``process_head`` is :class:`ReactionProcessHeadV53MP` reused UNCHANGED
  (2,178 params; reads ONLY ``z``; ``order_score IS q`` by the head's
  frozen wiring);
* the external forward surface is EXACTLY the five contract keys
  ``{u, z, q, endpoint_logit, order_score}`` — no extra external keys are
  ever emitted (config ``process_head.new_external_keys_forbidden``);
  ``forward_move`` stays ``{u, z}``;
* ``param_groups_by_id()`` groups parameters BY ``id()`` —
  ``core = backbone ∪ move_decoder`` (lr 1e-4), ``head = process_head``
  (lr 3e-4).  Name-prefix classifiers are FORBIDDEN here (ADR-0033 §2.2:
  ``move_decoder.*`` would trip ``PARAM_GROUP_UNCLASSIFIED`` in the V541
  scheme); every unique parameter must be classified EXACTLY once;
* ``torch.compile`` is rejected at construction (assert-no-compiled
  walk) and never called anywhere in this package (ADR-0033 §2.1).

The wrapper performs the config ``interface.wrapper_checks``: shape and
isfinite validation of the BackboneOutput in forward (the dataclass
itself stays check-light)."""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_INDEX_KEY,
    GRAPH_CV_DELTA_KEY,
    NODE_DEGK_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import ReactionProcessHeadV53MP
from pes2ts_core.pesn.models.v542.backbone_a0 import A0LegacyBackbone
from pes2ts_core.pesn.models.v542.decoder import (
    DirectMoveDecoder,
    assert_decoder_a_ref_frozen,
)
from pes2ts_core.pesn.models.v542.interfaces import (
    A0_ACTIVE_PARAMS_V542,
    A0_JOINT_PARAMS_V542,
    FORWARD_KEYS_V542,
    FORWARD_MOVE_KEYS_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    V542InitError,
    V542_SCHEMA_VERSION,
    assert_no_torch_compile,
    validate_backbone_output,
)

#: Registered V5.4.2 model version.
V542_MODEL_VERSION: Final[str] = V542_SCHEMA_VERSION


class V542Model(nn.Module):
    """Unified V5.4.2 model: A0 backbone + shared decoder + process head.

    Args:
        backbone: an EXISTING :class:`A0LegacyBackbone` shared BY
            REFERENCE (same policy as the legacy heads models); ``None``
            builds a fresh A0 adapter around ``move_decoder``.
        move_decoder: the SHARED :class:`DirectMoveDecoder`.  When
            ``backbone`` is provided it MUST share this decoder's
            ``w_v`` (``backbone.w_v is move_decoder.w_v``) — any other
            wiring is a typed refusal (double ``w_v`` would double-update
            in the optimizer and break the 57,619 joint invariant).
            The decoder must be a :class:`DirectMoveDecoder` carrying the
            FROZEN ``a_ref = 0.08`` (gate-1 B1 typed refusals
            ``V542_DECODER_TYPE`` / ``V542_DECODER_A_REF``).
        process_head: an EXISTING
            :class:`ReactionProcessHeadV53MP` shared BY REFERENCE;
            ``None`` builds a fresh one.

    Construction never calls ``torch.compile`` and refuses a compiled
    module anywhere in the assembled tree.
    """

    #: Registered model version.
    v542_version: Final[str] = V542_MODEL_VERSION

    def __init__(
            self,
            backbone: A0LegacyBackbone | None = None,
            *,
            move_decoder: DirectMoveDecoder | None = None,
            process_head: ReactionProcessHeadV53MP | None = None,
    ) -> None:
        super().__init__()
        decoder = move_decoder if move_decoder is not None else DirectMoveDecoder()
        if not isinstance(decoder, DirectMoveDecoder):
            raise V542InitError(
                "V542_DECODER_TYPE",
                "move_decoder must be the shared DirectMoveDecoder "
                f"(got {type(decoder).__name__})")
        # Gate-1 B1: the shared decoder must carry the FROZEN a_ref=0.08.
        assert_decoder_a_ref_frozen(decoder)
        if backbone is not None:
            if not isinstance(backbone, A0LegacyBackbone):
                raise V542InitError(
                    "V542_BACKBONE_TYPE",
                    f"backbone must be an A0LegacyBackbone, got "
                    f"{type(backbone).__name__}")
            if backbone.w_v is not decoder.w_v:
                raise V542InitError(
                    "V542_WV_NOT_SHARED",
                    "backbone.w_v is not the decoder's w_v module object — "
                    "the shared-w_v invariant (backbone.w_v is move_decoder."
                    "w_v) must hold; pass the decoder that owns the backbone")
            self.backbone = backbone
        else:
            self.backbone = A0LegacyBackbone(decoder)
        self.move_decoder = decoder
        self.process_head = (process_head if process_head is not None
                             else ReactionProcessHeadV53MP())
        if not isinstance(self.process_head, ReactionProcessHeadV53MP):
            raise V542InitError(
                "V542_PROCESS_HEAD",
                "process_head must be the frozen ReactionProcessHeadV53MP "
                f"(got {type(self.process_head).__name__})")

        # Structural refusal of torch.compile (never called; never allowed).
        assert_no_torch_compile(self)

        # -- frozen invariants at build (config param_budget) --------------
        groups = self.param_groups_by_id()
        core_params = sum(int(p.numel()) for p in groups["core"])
        head_params = sum(int(p.numel()) for p in groups["head"])
        if core_params != A0_ACTIVE_PARAMS_V542 + N_DECODER_PARAMS_V542:
            raise V542InitError(
                "V542_PARAM_BUDGET",
                f"core group has {core_params} params, expected "
                f"{A0_ACTIVE_PARAMS_V542}+{N_DECODER_PARAMS_V542}="
                f"{A0_ACTIVE_PARAMS_V542 + N_DECODER_PARAMS_V542}")
        if head_params != N_PROCESS_PARAMS_V542:
            raise V542InitError(
                "V542_PARAM_BUDGET",
                f"head group has {head_params} params, expected "
                f"{N_PROCESS_PARAMS_V542}")

    # -- param groups BY id (ADR-0033 §2.2; config init_policy.param_groups)
    def param_groups_by_id(self) -> dict[str, list[nn.Parameter]]:
        """``{"core": backbone ∪ move_decoder, "head": process_head}``.

        Grouping is BY ``id()`` — the shared ``w_v`` parameters appear
        EXACTLY once (in ``core``) even though two modules reference
        them.  Every unique parameter of this model must be classified
        exactly once; anything unclassified or double-classified is a
        typed refusal (never silently guessed, plan §7.1 discipline).
        """
        core: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in (self.backbone, self.move_decoder):
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    core.append(p)
        head: list[nn.Parameter] = []
        head_ids: set[int] = set()
        for p in self.process_head.parameters():
            if id(p) in head_ids:
                raise V542InitError(
                    "V542_PARAM_GROUP_DUPLICATE",
                    "process_head exposes the same parameter object twice")
            head_ids.add(id(p))
            head.append(p)
        overlap = seen & head_ids
        if overlap:
            raise V542InitError(
                "V542_PARAM_GROUP_OVERLAP",
                f"{len(overlap)} parameter object(s) shared between "
                "core and head groups")
        # Every unique model parameter classified exactly once.
        all_ids = {id(p) for p in self.parameters()}
        unclassified = all_ids - seen - head_ids
        if unclassified:
            raise V542InitError(
                "V542_PARAM_UNCLASSIFIED",
                f"{len(unclassified)} model parameter(s) belong to neither "
                "core (backbone ∪ move_decoder) nor head (process_head)")
        extra = (seen | head_ids) - all_ids
        if extra:
            raise V542InitError(
                "V542_PARAM_GROUP_EXTRA",
                f"{len(extra)} grouped parameter(s) are not model parameters")
        return {"core": core, "head": head}

    def param_group_report(self) -> list[dict[str, object]]:
        """JSON-safe group summary (counts by id-grouping; lr comes from
        the campaign config, not from the model)."""
        groups = self.param_groups_by_id()
        return [
            {"name": name,
             "n_tensors": len(groups[name]),
             "n_params": sum(int(p.numel()) for p in groups[name])}
            for name in ("core", "head")
        ]

    # -- accounting ---------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        """Deduped joint count (shared ``w_v`` counted ONCE): 57,619 for A0."""
        return sum(int(p.numel()) for p in self.parameters())

    def parameter_report(self) -> dict[str, int]:
        """Exact per-component counts + the frozen invariant check."""
        groups = self.param_groups_by_id()
        active = self.backbone.active_param_count
        joint = self.n_parameters
        report = {
            "backbone_active": int(active),
            "decoder": sum(int(p.numel()) for p in groups["core"])
            - int(active),
            "process_head": sum(int(p.numel()) for p in groups["head"]),
            "joint": int(joint),
        }
        expected = {
            "backbone_active": A0_ACTIVE_PARAMS_V542,
            "decoder": N_DECODER_PARAMS_V542,
            "process_head": N_PROCESS_PARAMS_V542,
            "joint": A0_JOINT_PARAMS_V542,
        }
        if report != expected:
            raise V542InitError(
                "V542_PARAM_BUDGET",
                f"parameter report {report} != frozen invariants {expected}")
        return report

    # -- forward ------------------------------------------------------------
    def _decode_u(self, batch: Mapping[str, torch.Tensor],
                  edge_msg: torch.Tensor) -> torch.Tensor:
        return self.move_decoder(
            edge_msg,
            batch[NODE_POS_KEY],
            batch[EDGE_INDEX_KEY],
            batch[NODE_KEY_SET_KEY],
            batch[NODE_DEGK_KEY],
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """External surface = EXACTLY ``{u, z, q, endpoint_logit, order_score}``.

        ``order_score IS q`` — the same tensor object produced by the
        head's frozen shared-scalar wiring is returned under both keys
        (asserted with ``is`` in the tests/parity harness).
        """
        out = self.backbone(batch)
        n_graphs = int(batch[GRAPH_CV_DELTA_KEY].shape[0])
        n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
        validate_backbone_output(out, n_graphs=n_graphs, n_edges=n_edges)

        u = self._decode_u(batch, out.edge_msg)
        head = self.process_head(out.z)
        result = {
            "u": u,
            "z": out.z,
            "q": head["q"],
            "endpoint_logit": head["endpoint_logit"],
            "order_score": head["order_score"],
        }
        # The external surface is frozen; drift here breaks every
        # trainer/evaluator binding (config new_external_keys_forbidden).
        if set(result) != set(FORWARD_KEYS_V542):
            raise V542InitError(
                "V542_FORWARD_KEYS",
                f"external keys {sorted(result)} != frozen "
                f"{sorted(FORWARD_KEYS_V542)}")
        return result

    def forward_move(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Backbone + decoder only: ``{u, z}`` — no PROCESS head math."""
        out = self.backbone(batch)
        n_graphs = int(batch[GRAPH_CV_DELTA_KEY].shape[0])
        n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
        validate_backbone_output(out, n_graphs=n_graphs, n_edges=n_edges)
        result = {"u": self._decode_u(batch, out.edge_msg), "z": out.z}
        if set(result) != set(FORWARD_MOVE_KEYS_V542):  # pragma: no cover - structural
            raise V542InitError(
                "V542_FORWARD_KEYS",
                f"forward_move keys {sorted(result)} != frozen "
                f"{sorted(FORWARD_MOVE_KEYS_V542)}")
        return result


def measured_params_v542_joint() -> int:
    """Exact ``sum(p.numel())`` of the default A0 joint build (fresh)."""
    return V542Model().n_parameters


__all__ = [
    "V542_MODEL_VERSION",
    "V542Model",
    "measured_params_v542_joint",
]
