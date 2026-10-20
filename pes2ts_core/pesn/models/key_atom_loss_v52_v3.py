"""V5.2 joint loss (functions only — W-LOSS owns the training loop).

Schema ``pesn_key_atom_loss_v52_v3``.  Normative sources (FROZEN): campaign
yaml ``v52_training`` (loss / stop_labels / qlabel / stop_class_weights /
lambda_mono=0 / lambda_path=0) and 30h plan §5.3–§5.4.

Frozen semantics:

* ``L_v52 = L_move + λ_q·L_q + λ_m·L_TSStop`` with ``λ_mono = λ_path = 0``
  (this campaign; the parameters exist only to be asserted zero — monotonic
  / path consistency are diagnostics, plan §5.4);
* ``L_move`` is the FULL frozen R3 loss
  (:func:`pes2ts_core.pesn.models.key_atom_losses_v3.key_atom_loss_v3`,
  B0-F + optional S/D bucket via its own args) — its ``total`` is reused
  verbatim as the move term;
* ``L_q = weighted_mean(Huber_0.1(q_pred − q_true))`` over q-valid rows
  only; curve-balanced aggregation is the sampler's job upstream (30h plan
  §4.1 batch composition), the loss applies ``frame_weight`` (label
  quality) exactly once, mirroring the move reduction;
* ``L_TSStop = weighted_mean(BCEWithLogits(stop_logit, y_stop))`` — raw
  logits (no pre-sigmoid), ``Mask`` rows NEVER enter the numerator OR the
  denominator; stop positive/negative class weights are FROZEN values
  (:class:`StopClassWeightsV52`) estimated from INNER-TRAIN rows only
  (:func:`estimate_stop_class_weights_v52`) — dev/confirm statistics may
  never flow back into training (T3.3);
* each head term is exactly 0 (not NaN) when its λ is 0 or when no row is
  eligible; gradients to the shared encoder flow only through the terms
  whose λ > 0 (no unexpected q/stop → action path exists by construction:
  the heads read ``z`` only, never ``u``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.models.key_atom_losses_v3 import (
    KeyAtomLossV3Config,
    key_atom_loss_v3,
)

V52_LOSS_SCHEMA: Final[str] = "pesn_key_atom_loss_v52_v3"

#: Frozen q-loss Huber scale (plan §5.4: Huber_{0.1}).
Q_HUBER_DELTA_V52: Final[float] = 0.1

#: Frozen stop-label windows (campaign ``v52_training.stop_labels``; the
#: yaml values 0.04/0.12 Å are re-exported here so the loss/label layers
#: share ONE constant — 禁止偷扩窗).
STOP_POSITIVE_WINDOW_ANGSTROM: Final[float] = 0.04
STOP_NEGATIVE_WINDOW_ANGSTROM: Final[float] = 0.12

#: Frozen off-switches (campaign ``v52_training.lambda_mono/lambda_path``).
LAMBDA_MONO_V52: Final[float] = 0.0
LAMBDA_PATH_V52: Final[float] = 0.0


class KeyAtomV52LossError(RuntimeError):
    """Typed refusal of the V5.2 loss layer."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise KeyAtomV52LossError(code, detail)


# ---------------------------------------------------------------------------
# Frozen stop class weights (inner-train ONLY, then frozen — T3.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StopClassWeightsV52:
    """Frozen BCE class weighting for the TSStop head.

    ``pos_weight`` is applied as the standard ``BCEWithLogitsLoss``
    ``pos_weight`` (negatives weight 1.0); ``n_pos``/``n_neg``/``source``
    record the INNER-TRAIN provenance so audits can re-derive the value.
    """

    pos_weight: float
    n_pos: int
    n_neg: int
    source: str = "inner-train"

    def __post_init__(self) -> None:
        if self.n_pos < 0 or self.n_neg < 0:
            raise KeyAtomV52LossError(
                "STOP_WEIGHTS_COUNTS", f"negative counts in {self}")
        if not (float(self.pos_weight) >= 0.0):
            raise KeyAtomV52LossError(
                "STOP_WEIGHTS_POS", f"pos_weight must be >= 0, got {self.pos_weight}")

    def as_dict(self) -> dict:
        return {"pos_weight": float(self.pos_weight), "n_pos": int(self.n_pos),
                "n_neg": int(self.n_neg), "source": str(self.source)}


def estimate_stop_class_weights_v52(
    y_stop: torch.Tensor,
    eligible: torch.Tensor,
    *,
    source: str = "inner-train",
) -> StopClassWeightsV52:
    """Estimate ``pos_weight = n_neg / n_pos`` on ELIGIBLE (unmasked) rows.

    ``eligible`` must already be the inner-train slice (Mask rows and any
    non-inner-train rows excluded upstream — this function cannot see them,
    which is the structural guarantee behind T3.3).  ``n_pos == 0`` yields
    ``pos_weight = 1.0`` with zero counts recorded — the caller is expected
    to raise ``insufficient_stop_positives`` from the label gate instead of
    fabricating a ratio.
    """
    y = y_stop.reshape(-1).to(dtype=torch.bool)
    mask = eligible.reshape(-1).to(dtype=torch.bool)
    _require(int(y.numel()) == int(mask.numel()), "STOP_WEIGHTS_SHAPE",
             f"y_stop {tuple(y.shape)} vs eligible {tuple(mask.shape)}")
    yv = y[mask]
    n_pos = int((yv == 1).sum())
    n_neg = int((yv == 0).sum())
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    return StopClassWeightsV52(
        pos_weight=float(pos_weight), n_pos=n_pos, n_neg=n_neg, source=source)


# ---------------------------------------------------------------------------
# Head losses
# ---------------------------------------------------------------------------
def _frame_broadcast(
    rows: int, frame_weight: torch.Tensor | None, device, dtype,
) -> torch.Tensor | None:
    if frame_weight is None:
        return None
    w = frame_weight.to(device=device, dtype=dtype).reshape(-1)
    _require(int(w.numel()) == rows, "FRAME_WEIGHT_SHAPE",
             f"frame_weight must be [{rows}], got {int(w.numel())}")
    return w


def progress_loss_v52(
    q_pred: torch.Tensor,
    q_true: torch.Tensor,
    q_mask: torch.Tensor,
    *,
    frame_weight: torch.Tensor | None = None,
    huber_delta: float = Q_HUBER_DELTA_V52,
) -> dict:
    """``L_q = weighted_mean(Huber_δ(q_pred − q_true))`` over q-valid rows.

    ``q_mask`` false ⇒ the row is out of numerator AND denominator (zero-arc
    / masked curves never divide by zero).  Empty ⇒ the term is exactly 0.
    """
    qp = q_pred.reshape(-1).to(dtype=torch.float32)
    qt = q_true.reshape(-1).to(dtype=torch.float32)
    _require(int(qp.numel()) == int(qt.numel()), "Q_SHAPE",
             f"q_pred {tuple(q_pred.shape)} vs q_true {tuple(q_true.shape)}")
    keep = q_mask.reshape(-1).to(dtype=torch.bool)
    _require(int(keep.numel()) == int(qp.numel()), "Q_MASK_SHAPE",
             f"q_mask {tuple(q_mask.shape)} vs q_pred {tuple(q_pred.shape)}")
    _require(bool(torch.isfinite(qp[keep]).all()) and bool(torch.isfinite(qt[keep]).all()),
             "Q_NONFINITE", "non-finite q values on a valid row")
    w = _frame_broadcast(int(qp.numel()), frame_weight, qp.device, qp.dtype)
    keep_f = keep.to(dtype=qp.dtype)
    w_row = keep_f if w is None else keep_f * w
    err = torch.abs(qp - qt)
    quad = 0.5 * err * err
    linear = float(huber_delta) * (err - 0.5 * float(huber_delta))
    term = torch.where(err <= float(huber_delta), quad, linear)
    denom = w_row.sum()
    n_valid = int(keep.sum())
    if float(denom) == 0.0 or n_valid == 0:
        loss = torch.zeros((), dtype=qp.dtype, device=qp.device)
    else:
        loss = (w_row * term).sum() / denom
    return {"l_q": loss, "n_q_rows": n_valid,
            "lambda_note": "curve-balanced aggregation is sampler-side"}


def ts_stop_loss_v52(
    stop_logit: torch.Tensor,
    y_stop: torch.Tensor,
    stop_mask: torch.Tensor,
    class_weights: StopClassWeightsV52,
    *,
    frame_weight: torch.Tensor | None = None,
) -> dict:
    """``L_TSStop = weighted_mean(BCEWithLogits)``; Mask rows never in the
    denominator.  Per-row effective weight = ``frame_weight × (pos_weight
    if positive else 1)``; the denominator sums the SAME effective weights
    over unmasked rows (Mask excluded — campaign ``stop_class_weights``).
    """
    logit = stop_logit.reshape(-1).to(dtype=torch.float32)
    y = y_stop.reshape(-1).to(dtype=torch.float32)
    _require(int(logit.numel()) == int(y.numel()), "STOP_SHAPE",
             f"stop_logit {tuple(stop_logit.shape)} vs y_stop {tuple(y_stop.shape)}")
    keep = stop_mask.reshape(-1).to(dtype=torch.bool)
    _require(int(keep.numel()) == int(logit.numel()), "STOP_MASK_SHAPE",
             f"stop_mask {tuple(stop_mask.shape)} vs stop_logit {tuple(stop_logit.shape)}")
    yv = y[keep]
    _require(bool(((yv == 0.0) | (yv == 1.0)).all()), "STOP_TARGET",
             "y_stop must be binary (0/1) on unmasked rows")
    _require(bool(torch.isfinite(logit[keep]).all()), "STOP_NONFINITE",
             "non-finite stop_logit on an unmasked row")
    w = _frame_broadcast(int(logit.numel()), frame_weight, logit.device, logit.dtype)
    pos_w = float(class_weights.pos_weight)
    class_row = torch.where(y > 0.5, torch.full_like(y, pos_w), torch.ones_like(y))
    keep_f = keep.to(dtype=logit.dtype)
    w_row = keep_f * class_row if w is None else keep_f * w * class_row
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    denom = w_row.sum()
    n_rows = int(keep.sum())
    if float(denom) == 0.0 or n_rows == 0:
        loss = torch.zeros((), dtype=logit.dtype, device=logit.device)
    else:
        loss = (w_row * bce).sum() / denom
    return {"l_ts_stop": loss, "n_stop_rows": n_rows,
            "pos_weight": pos_w}


# ---------------------------------------------------------------------------
# Total V5.2 loss
# ---------------------------------------------------------------------------
def key_atom_loss_v52(
    u: torch.Tensor,
    y: torch.Tensor,
    degK: torch.Tensor,
    key_ptr: torch.Tensor,
    pos: torch.Tensor,
    key_index: torch.Tensor,
    pair_index: torch.Tensor | None = None,
    *,
    q_pred: torch.Tensor | None = None,
    q_true: torch.Tensor | None = None,
    q_mask: torch.Tensor | None = None,
    stop_logit: torch.Tensor | None = None,
    y_stop: torch.Tensor | None = None,
    stop_mask: torch.Tensor | None = None,
    stop_class_weights: StopClassWeightsV52 | None = None,
    lambda_q: float = 0.0,
    lambda_stop: float = 0.0,
    lambda_mono: float = LAMBDA_MONO_V52,
    lambda_path: float = LAMBDA_PATH_V52,
    config: KeyAtomLossV3Config | None = None,
    lambda_cos: float = 0.0,
    frame_mask: torch.Tensor | None = None,
    frame_weight: torch.Tensor | None = None,
    pair_edit: torch.Tensor | None = None,
    lambda_strong: float = 0.0,
    lambda_dead: float = 0.0,
    aux_arm: str | None = None,
) -> dict:
    """``L_v52 = L_move + λ_q·L_q + λ_m·L_TSStop`` (λ_mono=λ_path=0 asserted).

    All ``u/y/degK/key_ptr/pos/key_index/pair_index`` and move-only kwargs
    are passed through UNCHANGED to the frozen R3 loss; its ``total`` is
    reused verbatim as ``L_move``.  Head tensors are per-GRAPH ([G]) and
    must align with ``key_ptr``'s graph count.  Setting ``lambda_q>0``
    without q tensors (or ``lambda_stop>0`` without stop tensors) is a
    typed refusal — never a silent zero.
    """
    _require(float(lambda_mono) == LAMBDA_MONO_V52, "LAMBDA_MONO",
             f"campaign freezes lambda_mono={LAMBDA_MONO_V52}; got {lambda_mono}")
    _require(float(lambda_path) == LAMBDA_PATH_V52, "LAMBDA_PATH",
             f"campaign freezes lambda_path={LAMBDA_PATH_V52}; got {lambda_path}")
    _require(float(lambda_q) >= 0.0 and float(lambda_stop) >= 0.0, "LAMBDA_NEG",
             "lambda_q/lambda_stop must be >= 0")

    move = key_atom_loss_v3(
        u, y, degK, key_ptr, pos, key_index, pair_index,
        config=config, lambda_cos=lambda_cos, frame_mask=frame_mask,
        frame_weight=frame_weight, pair_edit=pair_edit,
        lambda_strong=lambda_strong, lambda_dead=lambda_dead,
        aux_arm=aux_arm)

    n_graphs = int(key_ptr.shape[0]) - 1
    device = u.device
    dtype = u.dtype

    l_q = torch.zeros((), dtype=dtype, device=device)
    l_stop = torch.zeros((), dtype=dtype, device=device)
    n_q_rows = 0
    n_stop_rows = 0
    if float(lambda_q) != 0.0:
        _require(q_pred is not None and q_true is not None and q_mask is not None,
                 "Q_HEADS_MISSING",
                 "lambda_q > 0 requires q_pred/q_true/q_mask")
        for name, t in (("q_pred", q_pred), ("q_true", q_true), ("q_mask", q_mask)):
            _require(int(t.reshape(-1).numel()) == n_graphs, "Q_GRAPH_COUNT",
                     f"{name} must have {n_graphs} rows (one per graph), got "
                     f"{int(t.reshape(-1).numel())}")
        q_out = progress_loss_v52(q_pred, q_true, q_mask, frame_weight=frame_weight)
        l_q = q_out["l_q"].to(dtype=dtype)
        n_q_rows = int(q_out["n_q_rows"])
    if float(lambda_stop) != 0.0:
        _require(stop_logit is not None and y_stop is not None
                 and stop_mask is not None and stop_class_weights is not None,
                 "STOP_HEADS_MISSING",
                 "lambda_stop > 0 requires stop_logit/y_stop/stop_mask/"
                 "stop_class_weights")
        for name, t in (("stop_logit", stop_logit), ("y_stop", y_stop),
                        ("stop_mask", stop_mask)):
            _require(int(t.reshape(-1).numel()) == n_graphs, "STOP_GRAPH_COUNT",
                     f"{name} must have {n_graphs} rows (one per graph), got "
                     f"{int(t.reshape(-1).numel())}")
        stop_out = ts_stop_loss_v52(
            stop_logit, y_stop, stop_mask, stop_class_weights,
            frame_weight=frame_weight)
        l_stop = stop_out["l_ts_stop"].to(dtype=dtype)
        n_stop_rows = int(stop_out["n_stop_rows"])

    total = move["loss"] + float(lambda_q) * l_q + float(lambda_stop) * l_stop
    _require(bool(torch.isfinite(total)), "NONFINITE_LOSS",
             "V5.2 total loss is not finite")
    return {
        "loss": total,
        "l_move": move["loss"],
        "l_q": l_q,
        "l_ts_stop": l_stop,
        "lambda_q": float(lambda_q),
        "lambda_stop": float(lambda_stop),
        "lambda_mono": 0.0,
        "lambda_path": 0.0,
        "n_q_rows": n_q_rows,
        "n_stop_rows": n_stop_rows,
        "stop_class_weights": (stop_class_weights.as_dict()
                               if stop_class_weights is not None else None),
        "move": move,
        "schema": V52_LOSS_SCHEMA,
    }


__all__ = [
    "V52_LOSS_SCHEMA",
    "Q_HUBER_DELTA_V52",
    "STOP_POSITIVE_WINDOW_ANGSTROM",
    "STOP_NEGATIVE_WINDOW_ANGSTROM",
    "LAMBDA_MONO_V52",
    "LAMBDA_PATH_V52",
    "KeyAtomV52LossError",
    "StopClassWeightsV52",
    "estimate_stop_class_weights_v52",
    "progress_loss_v52",
    "ts_stop_loss_v52",
    "key_atom_loss_v52",
]
