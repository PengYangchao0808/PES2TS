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
  扩展” class; well inside the 70k construction budget)."""

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
    """Frozen V5.2 TSStop head: bare scalar logit per graph (no sigmoid)."""

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
]
