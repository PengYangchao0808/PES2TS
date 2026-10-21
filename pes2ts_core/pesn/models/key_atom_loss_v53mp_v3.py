"""V5.3-MP joint MOVE–PROCESS loss (functions only — W-LOSS owns the loop).

Schema ``pesn_key_atom_loss_v53mp_v3``.  Normative sources (FROZEN):
MoveProcessContract v1 §7 (损失, active-normative for V5.3-MP) + §3.2
(outputs: ``q_pred`` tanh / ``endpoint_logit`` raw / ``order_score``) +
§5 (PROCESS 标签: progress / directed ordering / endpoint completion),
and the V5.3-MP plan §6 (损失、采样与训练) / §8.1.

Frozen formula (contract §7)::

    L_total = L_Move + λ_r·L_ranking + λ_q·L_progress + λ_e·L_endpoint

Frozen semantics:

* ``L_Move`` — the FROZEN R3 total
  (:func:`pes2ts_core.pesn.models.key_atom_losses_v3.key_atom_loss_v3`,
  atom + pair + cosine + strong + dead) reused VERBATIM: every move
  argument (``u/y/degK/key_ptr/pos/key_index/pair_index`` + config,
  λ_cos, frame_mask, frame_weight, pair_edit, λ_strong, λ_dead, aux_arm)
  is passed through UNCHANGED, exactly like
  :func:`pes2ts_core.pesn.models.key_atom_loss_v52.key_atom_loss_v52`;
* ``L_ranking`` — PRIMARY supervision (contract §5.2: 排序为主监督;
  plan §6).  For each of the ``P`` graph-local pairs ``(i, j)`` — ``j``
  LATER along the intent than ``i``, both indexing the batch's ``G``
  graph rows — ``softplus((m − (q_j − q_i)) / m)`` with the frozen
  margin ``m = RANK_MARGIN_V53MP = 0.1`` (q units; q ∈ [−1, 1] via tanh),
  averaged over pairs.  Pairs may cross the TS and are never restricted
  to one side.  **Pair masking/sampling is the trainer's job upstream**
  — the loss consumes ``order_pairs`` as given.  ``P == 0`` (empty) or
  ``λ_r == 0`` ⇒ the term is EXACTLY 0.0 (never NaN);
* ``L_progress`` — auxiliary Huber-0.1 masked weighted mean over q-valid
  rows, reused READ-ONLY via :func:`progress_loss_v52` (same module as
  V5.2 — not modified here).  ``q_mask`` false ⇒ the row is out of
  numerator AND denominator;
* ``L_endpoint`` — BCEWithLogits with a SCALAR ``endpoint_pos_weight``
  (contract §7: BCEWithLogits + 类别权重 + 独立 Mask), adapted from
  :func:`ts_stop_loss_v52`'s weighted-mean code path WITHOUT touching the
  v52 module.  Per-row effective weight = ``frame_weight × (pos_weight
  if positive else 1)``; Mask rows NEVER enter the numerator OR the
  denominator; empty ⇒ EXACTLY 0.0;
* typed refusals (:class:`KeyAtomV53MPLossError`, mirroring
  :class:`KeyAtomV52LossError` with ``.code``/``.detail``): any
  ``λ < 0``; ``λ > 0`` without the term's tensors (never a silent
  zero); head tensors must be per-graph ``[G]`` rows aligned with
  ``key_ptr``'s graph count; malformed ``order_pairs`` (dtype/shape/
  bounds); non-finite head values;
* ``frame_weight`` applies exactly ONCE per head term (every head term is
  a weighted mean over its own effective row/pair weights — uniform
  rescales cancel), mirroring the move reduction (see the v52 module);
* the total must be finite (asserted); gradients to the shared encoder
  flow only through the terms whose λ > 0 (no q/endpoint → action path
  exists by construction: the heads read ``z`` only, never ``u``).
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.models.key_atom_loss_v52_v3 import progress_loss_v52
from pes2ts_core.pesn.models.key_atom_losses_v3 import (
    KeyAtomLossV3Config,
    key_atom_loss_v3,
)

V53MP_LOSS_SCHEMA: Final[str] = "pesn_key_atom_loss_v53mp_v3"

#: Frozen ranking margin (q units; contract §5.2 pairwise margin, plan §6).
RANK_MARGIN_V53MP: Final[float] = 0.1

#: Frozen q-loss Huber scale (shared with V5.2: plan §5.4 → §6 Huber_0.1).
Q_HUBER_DELTA_V53MP: Final[float] = 0.1


class KeyAtomV53MPLossError(RuntimeError):
    """Typed refusal of the V5.3-MP loss layer (mirrors KeyAtomV52LossError)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise KeyAtomV53MPLossError(code, detail)


def _frame_broadcast(
    rows: int, frame_weight: torch.Tensor | None, device, dtype,
) -> torch.Tensor | None:
    if frame_weight is None:
        return None
    w = frame_weight.to(device=device, dtype=dtype).reshape(-1)
    _require(int(w.numel()) == rows, "FRAME_WEIGHT_SHAPE",
             f"frame_weight must be [{rows}], got {int(w.numel())}")
    return w


# ---------------------------------------------------------------------------
# Head losses
# ---------------------------------------------------------------------------
def ranking_loss_v53mp(
    q_pred: torch.Tensor,
    order_pairs: torch.Tensor,
    *,
    margin: float = RANK_MARGIN_V53MP,
    frame_weight: torch.Tensor | None = None,
) -> dict:
    """``L_ranking = weighted_mean(softplus((m − (q_j − q_i))/m))`` over pairs.

    ``order_pairs`` is a ``LongTensor [2, P]`` of GRAPH-LOCAL row indices:
    row 0 = frame ``i`` (earlier along the intent), row 1 = frame ``j``
    (later; contract §5.2 requires ``q̂(X_j) > q̂(X_i)``).  Indices
    reference the batch's ``G`` graph rows, i.e. the rows of ``q_pred``.

    Pair masking/sampling is the trainer's job upstream — pairs are
    consumed as given.  ``P == 0`` ⇒ exactly 0.0 (never NaN).  When
    ``frame_weight`` is given, the pair weight is the MEAN of its two
    frames' weights (both frames' label quality apply) and enters the
    weighted mean once — uniform rescales cancel, mirroring the move
    reduction.
    """
    _require(float(margin) > 0.0, "RANK_MARGIN",
             f"rank_margin must be > 0, got {margin}")
    _require(order_pairs is not None, "RANK_PAIRS_MISSING",
             "ranking loss requires order_pairs")
    _require(isinstance(order_pairs, torch.Tensor), "RANK_PAIRS_TYPE",
             f"order_pairs must be a torch.Tensor, got {type(order_pairs).__name__}")
    _require(order_pairs.dtype in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
        "RANK_PAIRS_DTYPE",
        f"order_pairs must be an integer (Long) tensor, got {order_pairs.dtype}")
    _require(order_pairs.dim() == 2 and int(order_pairs.shape[0]) == 2,
             "RANK_PAIRS_SHAPE",
             f"order_pairs must be [2, P], got {tuple(order_pairs.shape)}")

    qp = q_pred.reshape(-1).to(dtype=torch.float32)
    _require(bool(torch.isfinite(qp).all()), "RANK_NONFINITE",
             "non-finite q_pred on a graph row")
    n_rows = int(qp.numel())

    pairs = order_pairs.long()
    if int(pairs.numel()) == 0:
        return {"l_rank": torch.zeros((), dtype=qp.dtype, device=qp.device),
                "n_rank_pairs": 0}
    pi, pj = pairs[0], pairs[1]
    _require(bool((pi >= 0).all() and (pi < n_rows).all()
                  and (pj >= 0).all() and (pj < n_rows).all()),
             "RANK_PAIRS_BOUNDS",
             f"order_pairs indices must lie in [0, {n_rows}) "
             "(graph rows of the batch)")

    m = float(margin)
    term = F.softplus((m - (qp[pj] - qp[pi])) / m)
    w = _frame_broadcast(n_rows, frame_weight, qp.device, qp.dtype)
    if w is None:
        loss = term.mean()
    else:
        pair_w = 0.5 * (w[pi] + w[pj])
        denom = pair_w.sum()
        if float(denom) == 0.0:
            loss = torch.zeros((), dtype=qp.dtype, device=qp.device)
        else:
            loss = (pair_w * term).sum() / denom
    return {"l_rank": loss, "n_rank_pairs": int(pi.numel())}


def endpoint_loss_v53mp(
    endpoint_logit: torch.Tensor,
    y_endpoint: torch.Tensor,
    endpoint_mask: torch.Tensor,
    *,
    endpoint_pos_weight: float = 1.0,
    frame_weight: torch.Tensor | None = None,
) -> dict:
    """``L_endpoint = weighted_mean(BCEWithLogits(endpoint_logit, y))``.

    Scalar ``endpoint_pos_weight`` plays the role of the frozen class
    weighting (contract §7; V5.2 used a dataclass — the V5.3 value is a
    single ``float`` argument, campaign-tunable).  Per-row effective
    weight = ``frame_weight × (pos_weight if positive else 1)``; the
    denominator sums the SAME effective weights over unmasked rows only —
    Mask rows NEVER enter the numerator OR the denominator; empty ⇒
    exactly 0.0.  Adapted from ``ts_stop_loss_v52``'s code path; the v52
    module is NOT modified.
    """
    _require(float(endpoint_pos_weight) >= 0.0, "ENDPOINT_POS_WEIGHT",
             f"endpoint_pos_weight must be >= 0, got {endpoint_pos_weight}")
    logit = endpoint_logit.reshape(-1).to(dtype=torch.float32)
    y = y_endpoint.reshape(-1).to(dtype=torch.float32)
    _require(int(logit.numel()) == int(y.numel()), "ENDPOINT_SHAPE",
             f"endpoint_logit {tuple(endpoint_logit.shape)} vs "
             f"y_endpoint {tuple(y_endpoint.shape)}")
    keep = endpoint_mask.reshape(-1).to(dtype=torch.bool)
    _require(int(keep.numel()) == int(logit.numel()), "ENDPOINT_MASK_SHAPE",
             f"endpoint_mask {tuple(endpoint_mask.shape)} vs "
             f"endpoint_logit {tuple(endpoint_logit.shape)}")
    yv = y[keep]
    _require(bool(((yv == 0.0) | (yv == 1.0)).all()), "ENDPOINT_TARGET",
             "y_endpoint must be binary (0/1) on unmasked rows")
    _require(bool(torch.isfinite(logit[keep]).all()), "ENDPOINT_NONFINITE",
             "non-finite endpoint_logit on an unmasked row")
    w = _frame_broadcast(int(logit.numel()), frame_weight,
                         logit.device, logit.dtype)
    pos_w = float(endpoint_pos_weight)
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
    return {"l_endpoint": loss, "n_endpoint_rows": n_rows,
            "pos_weight": pos_w}


# ---------------------------------------------------------------------------
# Total V5.3-MP loss
# ---------------------------------------------------------------------------
def key_atom_loss_v53mp(
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
    order_pairs: torch.Tensor | None = None,
    endpoint_logit: torch.Tensor | None = None,
    y_endpoint: torch.Tensor | None = None,
    endpoint_mask: torch.Tensor | None = None,
    endpoint_pos_weight: float = 1.0,
    lambda_rank: float = 0.0,
    lambda_q: float = 0.0,
    lambda_endpoint: float = 0.0,
    rank_margin: float = RANK_MARGIN_V53MP,
    config: KeyAtomLossV3Config | None = None,
    lambda_cos: float = 0.0,
    frame_mask: torch.Tensor | None = None,
    frame_weight: torch.Tensor | None = None,
    pair_edit: torch.Tensor | None = None,
    lambda_strong: float = 0.0,
    lambda_dead: float = 0.0,
    aux_arm: str | None = None,
) -> dict:
    """``L_total = L_move + λ_r·L_ranking + λ_q·L_progress + λ_e·L_endpoint``.

    All ``u/y/degK/key_ptr/pos/key_index/pair_index`` and move-only kwargs
    are passed through UNCHANGED to the frozen R3 loss; its ``total`` is
    reused verbatim as ``L_move`` (contract §7: L_Move = R3 已验证组合).
    Head tensors are per-GRAPH ([G]) and must align with ``key_ptr``'s
    graph count.  ``order_pairs [2, P]`` is graph-local (row0 = earlier
    frame ``i``, row1 = later frame ``j`` along the intent).  Setting any
    λ > 0 without that term's tensors is a typed refusal — never a
    silent zero; λ < 0 is refused; the total must be finite.

    Returns exactly::

        {"loss", "l_move", "l_rank", "l_q", "l_endpoint",
         "n_rank_pairs", "n_q_rows", "n_endpoint_rows",
         "lambda_rank", "lambda_q", "lambda_endpoint",
         "move": <full move subdict>, "schema": V53MP_LOSS_SCHEMA}
    """
    _require(float(lambda_rank) >= 0.0 and float(lambda_q) >= 0.0
             and float(lambda_endpoint) >= 0.0, "LAMBDA_NEG",
             "lambda_rank/lambda_q/lambda_endpoint must be >= 0")
    _require(float(endpoint_pos_weight) >= 0.0, "ENDPOINT_POS_WEIGHT",
             f"endpoint_pos_weight must be >= 0, got {endpoint_pos_weight}")
    _require(float(rank_margin) > 0.0, "RANK_MARGIN",
             f"rank_margin must be > 0, got {rank_margin}")

    move = key_atom_loss_v3(
        u, y, degK, key_ptr, pos, key_index, pair_index,
        config=config, lambda_cos=lambda_cos, frame_mask=frame_mask,
        frame_weight=frame_weight, pair_edit=pair_edit,
        lambda_strong=lambda_strong, lambda_dead=lambda_dead,
        aux_arm=aux_arm)

    n_graphs = int(key_ptr.shape[0]) - 1
    device = u.device
    dtype = u.dtype

    l_rank = torch.zeros((), dtype=dtype, device=device)
    l_q = torch.zeros((), dtype=dtype, device=device)
    l_endpoint = torch.zeros((), dtype=dtype, device=device)
    n_rank_pairs = 0
    n_q_rows = 0
    n_endpoint_rows = 0

    if float(lambda_rank) != 0.0:
        _require(q_pred is not None, "RANK_HEADS_MISSING",
                 "lambda_rank > 0 requires q_pred")
        _require(order_pairs is not None, "RANK_PAIRS_MISSING",
                 "lambda_rank > 0 requires order_pairs")
        _require(int(q_pred.reshape(-1).numel()) == n_graphs, "Q_GRAPH_COUNT",
                 f"q_pred must have {n_graphs} rows (one per graph), got "
                 f"{int(q_pred.reshape(-1).numel())}")
        rank_out = ranking_loss_v53mp(
            q_pred, order_pairs, margin=rank_margin,
            frame_weight=frame_weight)
        l_rank = rank_out["l_rank"].to(dtype=dtype)
        n_rank_pairs = int(rank_out["n_rank_pairs"])

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

    if float(lambda_endpoint) != 0.0:
        _require(endpoint_logit is not None and y_endpoint is not None
                 and endpoint_mask is not None, "ENDPOINT_HEADS_MISSING",
                 "lambda_endpoint > 0 requires endpoint_logit/y_endpoint/"
                 "endpoint_mask")
        for name, t in (("endpoint_logit", endpoint_logit),
                        ("y_endpoint", y_endpoint),
                        ("endpoint_mask", endpoint_mask)):
            _require(int(t.reshape(-1).numel()) == n_graphs,
                     "ENDPOINT_GRAPH_COUNT",
                     f"{name} must have {n_graphs} rows (one per graph), got "
                     f"{int(t.reshape(-1).numel())}")
        ep_out = endpoint_loss_v53mp(
            endpoint_logit, y_endpoint, endpoint_mask,
            endpoint_pos_weight=endpoint_pos_weight,
            frame_weight=frame_weight)
        l_endpoint = ep_out["l_endpoint"].to(dtype=dtype)
        n_endpoint_rows = int(ep_out["n_endpoint_rows"])

    total = (move["loss"] + float(lambda_rank) * l_rank
             + float(lambda_q) * l_q + float(lambda_endpoint) * l_endpoint)
    _require(bool(torch.isfinite(total)), "NONFINITE_LOSS",
             "V5.3-MP total loss is not finite")
    return {
        "loss": total,
        "l_move": move["loss"],
        "l_rank": l_rank,
        "l_q": l_q,
        "l_endpoint": l_endpoint,
        "n_rank_pairs": int(n_rank_pairs),
        "n_q_rows": int(n_q_rows),
        "n_endpoint_rows": int(n_endpoint_rows),
        "lambda_rank": float(lambda_rank),
        "lambda_q": float(lambda_q),
        "lambda_endpoint": float(lambda_endpoint),
        "move": move,
        "schema": V53MP_LOSS_SCHEMA,
    }


__all__ = [
    "V53MP_LOSS_SCHEMA",
    "RANK_MARGIN_V53MP",
    "Q_HUBER_DELTA_V53MP",
    "KeyAtomV53MPLossError",
    "endpoint_loss_v53mp",
    "key_atom_loss_v53mp",
    "ranking_loss_v53mp",
]
