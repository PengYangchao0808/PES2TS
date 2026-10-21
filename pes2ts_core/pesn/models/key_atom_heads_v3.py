"""V5.2 auxiliary heads (Progress ``q`` + TSStop logit) on StarEGNNV3.

Model version ``pesn_key_atom_v52_heads_v3`` (campaign ``v52_training.heads``
FROZEN; 30h plan §5.1).  The R3 backbone
(:class:`~pes2ts_core.pesn.models.key_atom_egnn_v3.StarEGNNV3`) is shared
UNCHANGED — its frozen ``z`` graph summary ``[G, 64]`` feeds both heads::

    (X_current, G_reaction) → StarEGNNV3 → u [ΣK,3], z [G,64]
                                              ├→ MoveHead   : u (backbone)
                                              ├→ Progress   : q = tanh(Linear(64,32)→SiLU→Linear(32,1))
                                              └→ TSStop     : logit scalar (Linear(64,1))

Frozen architecture (campaign yaml ``v52_training.heads``):

* ``progress``: exactly ``q = tanh(Linear(64, 32) → SiLU → Linear(32, 1))``
  — output in ``[-1, 1]``, per graph;
* ``ts_stop``: a bare scalar LOGIT per graph (``Linear(64, 1)``, NO sigmoid
  in the module — the frozen training recipe uses ``BCEWithLogitsLoss``,
  plan §5.1: 不要训练时预先 sigmoid 两次).  The yaml freezes the stop head
  only as "logit scalar", so the minimal legal wiring is a single affine
  map from ``z``;
* heads consume ONLY the backbone's invariant ``z`` — they never read or
  write node states, coordinates or the ``u`` pathway, and never feed back
  into the encoder (plan §5.1: ``q`` 不能反向作为模型的历史状态输入).

**Attachability guarantee**: constructing :class:`KeyAtomV52Model` (with or
without an existing backbone) and running the heads leaves the backbone —
and therefore the Move output ``u`` — BIT-for-bit unchanged
(``torch.equal``); asserted explicitly in
``tests/pesn/test_r3_v52_heads.py``.  ``forward`` still enforces the v3
forward-view contract (``check_forward_view_keys_v3``): supervision keys
(including the V5.2 labels) can never reach any module here.

Parameter accounting (exact; reported separately from the backbone):

* backbone ``StarEGNNV3`` = ``MEASURED_PARAMS_V3 = 55,441`` (E2-frozen);
* ``ProgressHeadV3`` = 64·32+32 + 32·1+1 = **2,113**;
* ``TSStopHeadV3`` = 64·1+1 = **65**;
* joint build = 55,441 + 2,178 = **57,619** (the campaign's “~59k + 少量头
  扩展” class; well inside the 70k construction budget).

V5.3-MP section (MoveProcessContract v1 §6 + §3.2 outputs; V5.3-MP plan
§3 目标架构 / §8.1 — added WITHOUT changing any V5.2 class behavior):

* version ``pesn_key_atom_v53mp_heads_v1`` (contract §6 版本 row);
* :class:`ReactionProcessHeadV53MP` — the shared ``ReactionProcessHead``
  reading the frozen backbone ``z [G, 64]`` and emitting the PROCESS trio
  of contract §3.2: ``q_pred`` (tanh, A=−1/TS=0/B=+1), a bare
  ``endpoint_logit`` (BCEWithLogits; 训练时禁止双重 sigmoid) and
  ``order_score``.  **Wiring choice**: ``order_score IS q`` — this build
  SHARES the ordering score with q (contract §6 explicitly allows
  “可与 q 共享或独立”); one scalar head serving both directed ordering
  (§5.2 primary supervision) and q regression (§5.1 auxiliary) so the two
  can never drift apart in sign/scale.  The V5.3 loss (plan §6) supervises
  the shared scalar via ``λ_r·L_ranking + λ_q·L_progress``;
* :class:`EndpointHeadV53MP` — the V5.3 endpoint-completion logit head,
  shape-identical to the old stop head (``Linear(64, 1)``, NO sigmoid)
  but with NEW label semantics (``endpoint_label_spec_v1``, contract
  §5.3); it is NOT the legacy TS-stop head;
* :class:`KeyAtomV53MPModel` — backbone + ``process_head`` only (no
  separate progress/ts_stop children); shares an EXISTING backbone BY
  REFERENCE exactly like :class:`KeyAtomV52Model`, so the
  **Attachability guarantee** (contract §6) holds: constructing the model
  and running the heads leaves ``u`` bit-for-bit unchanged
  (``torch.equal``, asserted in ``tests/pesn/test_r3_v53mp_heads.py``);
* legacy discipline: :attr:`TSStopHeadV3.legacy_frozen = True` marks the
  V5.2 stop head as **LEGACY_TS_STOP** (contract §2/§6 弃用 row): V5.2
  semantics, only for loading old checkpoints — forbidden for new
  training or new inference.  Nothing is renamed or removed (V5.2
  checkpoints/tests keep loading);
* parameter accounting (exact, asserted in
  ``tests/pesn/test_r3_v53mp_heads.py``): backbone 55,441 +
  (ProgressHeadV3 2,113 + EndpointHeadV53MP 65 = 2,178) =
  **57,619** — identical joint count to V5.2 because the endpoint head is
  shape-identical to the old stop head; only the SUPERVISION semantics
  change (loss module ``key_atom_loss_v53mp_v3``)."""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.key_atom_egnn_v3 import Z_DIM_V3, StarEGNNV3

#: Registered heads-model version (campaign ``v52_training.heads``).
V52_HEADS_VERSION: Final[str] = "pesn_key_atom_v52_heads_v3"

#: Frozen progress-head hidden width (yaml: ``Linear(64,32) SiLU Linear(32,1)``).
PROGRESS_HIDDEN_V52: Final[int] = 32

#: Exact parameter counts (asserted by ``tests/pesn/test_r3_v52_heads.py``).
MEASURED_PARAMS_PROGRESS_HEAD_V52: Final[int] = 2_113   # (64·32+32)+(32+1)
MEASURED_PARAMS_TS_STOP_HEAD_V52: Final[int] = 65      # (64+1)
MEASURED_PARAMS_HEADS_V52: Final[int] = (
    MEASURED_PARAMS_PROGRESS_HEAD_V52 + MEASURED_PARAMS_TS_STOP_HEAD_V52)
MEASURED_PARAMS_V52_JOINT: Final[int] = 55_441 + MEASURED_PARAMS_HEADS_V52


class ProgressHeadV3(nn.Module):
    """Frozen V5.2 progress head: ``q = tanh(Linear(64,32)→SiLU→Linear(32,1))``."""

    def __init__(self, z_dim: int = Z_DIM_V3, hidden: int = PROGRESS_HIDDEN_V52) -> None:
        super().__init__()
        if int(z_dim) != Z_DIM_V3 or int(hidden) != PROGRESS_HIDDEN_V52:
            raise RuntimeError(
                f"{V52_HEADS_VERSION} freezes the progress head to "
                f"Linear({Z_DIM_V3},{PROGRESS_HIDDEN_V52})→SiLU→"
                f"Linear({PROGRESS_HIDDEN_V52},1) (got {z_dim}/{hidden})")
        self.fc1 = nn.Linear(Z_DIM_V3, PROGRESS_HIDDEN_V52)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(PROGRESS_HIDDEN_V52, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z [G,64] → q [G]`` in ``[-1, 1]`` (tanh)."""
        if z.dim() != 2 or int(z.shape[1]) != Z_DIM_V3:
            raise RuntimeError(
                f"progress head expects z [G,{Z_DIM_V3}], got {tuple(z.shape)}")
        return torch.tanh(self.fc2(self.act(self.fc1(z)))).squeeze(-1)


class TSStopHeadV3(nn.Module):
    """Frozen V5.2 TSStop head: bare scalar logit per graph (no sigmoid).

    LEGACY_TS_STOP: V5.2 semantics; only for loading old checkpoints;
    forbidden for new training (contract §2/§6).  See
    :data:`LEGACY_TS_STOP_HEAD` and :attr:`legacy_frozen`.
    """

    #: LEGACY marker (contract §2 弃用词 ``LEGACY_TS_STOP``): this class may
    #: only load old V5.2 checkpoints — new training/inference must use
    #: :class:`EndpointHeadV53MP` instead.
    legacy_frozen = True

    def __init__(self, z_dim: int = Z_DIM_V3) -> None:
        super().__init__()
        if int(z_dim) != Z_DIM_V3:
            raise RuntimeError(
                f"{V52_HEADS_VERSION} freezes the stop head to "
                f"Linear({Z_DIM_V3},1) (got z_dim={z_dim})")
        self.fc = nn.Linear(Z_DIM_V3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z [G,64] → stop_logit [G]`` (raw logit; sigmoid belongs to eval)."""
        if z.dim() != 2 or int(z.shape[1]) != Z_DIM_V3:
            raise RuntimeError(
                f"stop head expects z [G,{Z_DIM_V3}], got {tuple(z.shape)}")
        return self.fc(z).squeeze(-1)


class KeyAtomV52Model(nn.Module):
    """StarEGNNV3 backbone + frozen V5.2 heads (shared ``z``; move untouched).

    ``backbone`` may be an EXISTING (e.g. R3-pretrained) ``StarEGNNV3``; it
    is then shared BY REFERENCE — construction performs no copy and no
    in-place edit, so pretrained weights and the ``u`` pathway are
    bit-for-bit preserved (the backbone's own forward is called verbatim).
    """

    def __init__(self, backbone: StarEGNNV3 | None = None) -> None:
        super().__init__()
        if backbone is not None and not isinstance(backbone, StarEGNNV3):
            raise TypeError(
                f"backbone must be a StarEGNNV3 (got {type(backbone).__name__})")
        self.backbone = backbone if backbone is not None else StarEGNNV3()
        self.progress_head = ProgressHeadV3()
        self.ts_stop_head = TSStopHeadV3()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Full V5.2 forward: ``{"u": [ΣK,3], "z": [G,64], "q": [G], "stop_logit": [G]}``."""
        out = self.backbone(batch)
        z = out["z"]
        return {
            "u": out["u"],
            "z": z,
            "q": self.progress_head(z),
            "stop_logit": self.ts_stop_head(z),
        }

    def forward_move(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Backbone-only view (R3 M0-equivalent): ``{"u", "z"}`` — no head math.

        Bit-for-bit identical to calling the backbone directly (delegated
        call; asserted by the backbone-unchanged test).
        """
        return self.backbone(batch)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def head_parameter_counts(self) -> dict[str, int]:
        """Exact per-head parameter counts (reported separately from backbone)."""
        return {
            "backbone": self.backbone.n_parameters,
            "progress_head": sum(
                p.numel() for p in self.progress_head.parameters()),
            "ts_stop_head": sum(
                p.numel() for p in self.ts_stop_head.parameters()),
        }


def measured_params_v52_joint() -> int:
    """Exact ``sum(p.numel())`` of the default joint build (fresh build)."""
    return KeyAtomV52Model().n_parameters


# ---------------------------------------------------------------------------
# V5.3-MP heads (MoveProcessContract v1 §6; V5.3-MP plan §3 / §8.1)
# ---------------------------------------------------------------------------
#: Registered heads-model version for the V5.3-MP campaign (contract §6).
V53MP_HEADS_VERSION: Final[str] = "pesn_key_atom_v53mp_heads_v1"

#: Legacy marker name for the frozen V5.2 stop head (contract §2/§6:
#: ``LEGACY_TS_STOP`` = read old checkpoints only; 禁止新训练/新推理).
LEGACY_TS_STOP_HEAD: Final[str] = "TSStopHeadV3"

#: Exact V5.3-MP parameter counts (asserted by
#: ``tests/pesn/test_r3_v53mp_heads.py``).
MEASURED_PARAMS_ENDPOINT_HEAD_V53MP: Final[int] = 65    # (64+1)
MEASURED_PARAMS_HEADS_V53MP: Final[int] = (
    MEASURED_PARAMS_PROGRESS_HEAD_V52 + MEASURED_PARAMS_ENDPOINT_HEAD_V53MP)
MEASURED_PARAMS_V53MP_JOINT: Final[int] = (
    55_441 + MEASURED_PARAMS_HEADS_V53MP)


class EndpointHeadV53MP(nn.Module):
    """V5.3 endpoint-completion head: bare scalar logit per graph.

    ``Linear(Z_DIM_V3, 1)`` from the shared backbone ``z`` — the SAME
    minimal legal wiring as the frozen V5.2 stop head (contract §6: 端点
    输出 ``endpoint_logit [G]``, BCEWithLogits; 训练时禁止双重 sigmoid),
    but with NEW V5.3 label semantics (``endpoint_label_spec_v1``,
    contract §5.3) — hence a separate class: :class:`TSStopHeadV3` stays
    LEGACY.  NO sigmoid inside the module.
    """

    def __init__(self, z_dim: int = Z_DIM_V3) -> None:
        super().__init__()
        if int(z_dim) != Z_DIM_V3:
            raise RuntimeError(
                f"{V53MP_HEADS_VERSION} freezes the endpoint head to "
                f"Linear({Z_DIM_V3},1) (got z_dim={z_dim})")
        self.fc = nn.Linear(Z_DIM_V3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z [G,64] → endpoint_logit [G]`` (raw logit; sigmoid at eval)."""
        if z.dim() != 2 or int(z.shape[1]) != Z_DIM_V3:
            raise RuntimeError(
                f"endpoint head expects z [G,{Z_DIM_V3}], got {tuple(z.shape)}")
        return self.fc(z).squeeze(-1)


class ReactionProcessHeadV53MP(nn.Module):
    """Shared ``ReactionProcessHead`` (contract §2/§6): q + endpoint + order.

    Reads ONLY the frozen backbone ``z [G, 64]`` and returns::

        {"q": [G], "endpoint_logit": [G], "order_score": [G]}

    Sub-heads: :class:`ProgressHeadV3` (reused UNCHANGED — contract §6
    “旧进度头 … 不得改变输入/输出语义”) for ``q`` and
    :class:`EndpointHeadV53MP` for ``endpoint_logit``.

    **Frozen wiring choice**: ``order_score IS q`` — the directed-ordering
    score SHARES the progress scalar (contract §3.2: ``order_score`` 可与
    q 共享或独立; this build shares).  Ranking supervision
    (contract §5.2, plan §6) and q regression therefore act on ONE head,
    so ordering and progress can never disagree in sign or scale.
    """

    def __init__(self) -> None:
        super().__init__()
        self.progress_head = ProgressHeadV3()
        self.endpoint_head = EndpointHeadV53MP()

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """``z [G,64] → {"q", "endpoint_logit", "order_score"}`` (each [G])."""
        q = self.progress_head(z)
        endpoint_logit = self.endpoint_head(z)
        return {
            "q": q,
            "endpoint_logit": endpoint_logit,
            "order_score": q,   # shared by construction (contract §6)
        }


class KeyAtomV53MPModel(nn.Module):
    """StarEGNNV3 backbone + shared V5.3-MP PROCESS head (move untouched).

    Like :class:`KeyAtomV52Model`, ``backbone`` may be an EXISTING (e.g.
    R3-pretrained) ``StarEGNNV3``; it is then shared BY REFERENCE — no
    copy, no in-place edit, pretrained weights and the ``u`` pathway are
    bit-for-bit preserved (the backbone's own forward is called verbatim).
    The only child besides the backbone is ``process_head`` (contract §6:
    no separate ts_stop head — the legacy stop path is LEGACY_TS_STOP).
    """

    def __init__(self, backbone: StarEGNNV3 | None = None) -> None:
        super().__init__()
        if backbone is not None and not isinstance(backbone, StarEGNNV3):
            raise TypeError(
                f"backbone must be a StarEGNNV3 (got {type(backbone).__name__})")
        self.backbone = backbone if backbone is not None else StarEGNNV3()
        self.process_head = ReactionProcessHeadV53MP()

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Full V5.3-MP forward: ``{"u", "z", "q", "endpoint_logit",
        "order_score"}`` — outputs of contract §3.2 minus the online-only
        ``move_flag``/``selected_coords`` (computed by the controller)."""
        out = self.backbone(batch)
        z = out["z"]
        head = self.process_head(z)
        return {
            "u": out["u"],
            "z": z,
            "q": head["q"],
            "endpoint_logit": head["endpoint_logit"],
            "order_score": head["order_score"],
        }

    def forward_move(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Backbone-only view (R3 M0-equivalent): ``{"u", "z"}`` — no head math.

        Bit-for-bit identical to calling the backbone directly (delegated
        call; asserted by ``tests/pesn/test_r3_v53mp_heads.py``).
        """
        return self.backbone(batch)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def head_parameter_counts(self) -> dict[str, int]:
        """Exact per-head parameter counts (reported separately from backbone)."""
        return {
            "backbone": self.backbone.n_parameters,
            "process_head": sum(
                p.numel() for p in self.process_head.parameters()),
            "process_head_progress": sum(
                p.numel() for p in self.process_head.progress_head.parameters()),
            "process_head_endpoint": sum(
                p.numel() for p in self.process_head.endpoint_head.parameters()),
        }


def measured_params_v53mp_joint() -> int:
    """Exact ``sum(p.numel())`` of the default V5.3-MP joint build."""
    return KeyAtomV53MPModel().n_parameters


__all__ = [
    "V52_HEADS_VERSION",
    "PROGRESS_HIDDEN_V52",
    "MEASURED_PARAMS_PROGRESS_HEAD_V52",
    "MEASURED_PARAMS_TS_STOP_HEAD_V52",
    "MEASURED_PARAMS_HEADS_V52",
    "MEASURED_PARAMS_V52_JOINT",
    "ProgressHeadV3",
    "TSStopHeadV3",
    "KeyAtomV52Model",
    "measured_params_v52_joint",
    "V53MP_HEADS_VERSION",
    "LEGACY_TS_STOP_HEAD",
    "MEASURED_PARAMS_ENDPOINT_HEAD_V53MP",
    "MEASURED_PARAMS_HEADS_V53MP",
    "MEASURED_PARAMS_V53MP_JOINT",
    "EndpointHeadV53MP",
    "ReactionProcessHeadV53MP",
    "KeyAtomV53MPModel",
    "measured_params_v53mp_joint",
]
