"""V5.3-MP joint loss tests: ranking math, mask semantics, typed refusals.

Frozen sources under test: MoveProcessContract v1 §7 (损失) + §5 (PROCESS
标签) and V5.3-MP plan §6.  Covers: λ=0 ⇒ head terms EXACTLY 0 and total
bit-identical to the frozen R3 move loss; λ>0 without tensors ⇒ typed
refusal (:class:`KeyAtomV53MPLossError` with ``.code``); λ<0 refusal;
pairwise ranking hand values + order-satisfied ordering; empty
``order_pairs`` ⇒ exactly 0.0; endpoint Mask rows excluded from BOTH
numerator and denominator (manual BCE comparison); q-mask semantics;
non-finite head inputs refused; gradient flow through L_ranking + L_q +
L_endpoint; frame_weight applied exactly once (uniform-scale
cancellation); exact return-schema keys.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from pes2ts_core.pesn.models.key_atom_loss_v53mp_v3 import (  # noqa: E402
    Q_HUBER_DELTA_V53MP,
    RANK_MARGIN_V53MP,
    V53MP_LOSS_SCHEMA,
    KeyAtomV53MPLossError,
    endpoint_loss_v53mp,
    key_atom_loss_v53mp,
    ranking_loss_v53mp,
)
from pes2ts_core.pesn.models.key_atom_loss_v52_v3 import (  # noqa: E402
    progress_loss_v52,
)
from pes2ts_core.pesn.models.key_atom_losses_v3 import (  # noqa: E402
    key_atom_loss_v3,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_intent  # noqa: E402
from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_teacher  # noqa: E402

RESULT_KEYS = {
    "loss", "l_move", "l_rank", "l_q", "l_endpoint",
    "n_rank_pairs", "n_q_rows", "n_endpoint_rows",
    "lambda_rank", "lambda_q", "lambda_endpoint",
    "move", "schema",
}


def _move_args(n_graphs: int = 2, seed: int = 0):
    """Minimal legal move-loss tensors for the SN2 K-set (3 keys/graph)."""
    g = int(n_graphs)
    n_keys = 3 * g
    torch.manual_seed(seed)
    u = torch.randn(n_keys, 3) * 0.01
    y = torch.randn(n_keys, 3) * 0.01
    degK = torch.ones(n_keys)
    key_ptr = torch.arange(0, n_keys + 1, 3, dtype=torch.long)
    pos = torch.zeros(n_keys, 3)
    pos[:, 0] = torch.arange(n_keys, dtype=torch.float32) * 1.5
    key_index = torch.arange(n_keys, dtype=torch.long)
    pair_index = torch.tensor(
        [[3 * i, 3 * i + 1] for i in range(g)], dtype=torch.long).t().contiguous()
    return dict(u=u, y=y, degK=degK, key_ptr=key_ptr, pos=pos,
                key_index=key_index, pair_index=pair_index)


def _move_kwargs(n_graphs: int = 2) -> dict:
    return dict(frame_weight=torch.ones(int(n_graphs)),
                frame_mask=torch.ones(int(n_graphs), dtype=torch.bool))


def _q_head(n_graphs: int = 2):
    q_pred = torch.tensor([0.4, -0.2] * (int(n_graphs) // 2))[:n_graphs]
    q_true = torch.tensor([0.1, 0.1] * (int(n_graphs) // 2))[:n_graphs]
    q_mask = torch.ones(n_graphs, dtype=torch.bool)
    return q_pred, q_true, q_mask


def _endpoint_head(n_graphs: int = 2):
    logit = torch.tensor([1.0, -1.0] * (int(n_graphs) // 2))[:n_graphs]
    y = torch.tensor([1.0, 0.0] * (int(n_graphs) // 2))[:n_graphs]
    mask = torch.tensor([True, False] * (int(n_graphs) // 2))[:n_graphs]
    return logit, y, mask


# ---------------------------------------------------------------------------
# λ=0 ⇒ exact-zero head terms, total == frozen move total
# ---------------------------------------------------------------------------
def test_zero_lambdas_total_is_move() -> None:
    args = _move_args()
    move_only = key_atom_loss_v3(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"], **_move_kwargs())
    joint = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        lambda_rank=0.0, lambda_q=0.0, lambda_endpoint=0.0, **_move_kwargs())
    # λ=0 ⇒ head terms skipped entirely; total bit-identical to R3 move loss
    assert torch.equal(joint["loss"], move_only["loss"])
    assert torch.equal(joint["l_move"], move_only["loss"])
    assert float(joint["l_rank"]) == 0.0
    assert float(joint["l_q"]) == 0.0
    assert float(joint["l_endpoint"]) == 0.0
    assert joint["n_rank_pairs"] == 0
    assert joint["n_q_rows"] == 0
    assert joint["n_endpoint_rows"] == 0
    assert joint["schema"] == V53MP_LOSS_SCHEMA == "pesn_key_atom_loss_v53mp_v3"


def test_zero_lambdas_ignore_absent_head_tensors() -> None:
    """λ=0 must NOT demand tensors (opposite of the typed refusal)."""
    args = _move_args()
    out = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"], **_move_kwargs())
    assert set(out) == RESULT_KEYS
    assert out["schema"] == V53MP_LOSS_SCHEMA
    assert isinstance(out["move"], dict) and out["move"]["schema"] == "pesn_key_atom_loss_v3"


def test_return_keys_are_exact() -> None:
    args = _move_args()
    q_pred, q_true, q_mask = _q_head()
    logit, y_ep, ep_mask = _endpoint_head()
    out = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_pred, q_true=q_true, q_mask=q_mask,
        order_pairs=torch.tensor([[0], [1]], dtype=torch.long),
        endpoint_logit=logit, y_endpoint=y_ep, endpoint_mask=ep_mask,
        lambda_rank=0.1, lambda_q=0.05, lambda_endpoint=0.05,
        **_move_kwargs())
    assert set(out) == RESULT_KEYS


# ---------------------------------------------------------------------------
# Typed refusals
# ---------------------------------------------------------------------------
def test_missing_tensors_are_typed_refusal() -> None:
    args = _move_args()
    base = (args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"])
    kw = _move_kwargs()

    with pytest.raises(KeyAtomV53MPLossError) as ei:   # no q_pred
        key_atom_loss_v53mp(
            *base, order_pairs=torch.tensor([[0], [1]], dtype=torch.long),
            lambda_rank=0.1, **kw)
    assert ei.value.code == "RANK_HEADS_MISSING"

    q_pred, _, _ = _q_head()
    with pytest.raises(KeyAtomV53MPLossError) as ei:   # no order_pairs
        key_atom_loss_v53mp(*base, q_pred=q_pred, lambda_rank=0.1, **kw)
    assert ei.value.code == "RANK_PAIRS_MISSING"

    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(*base, lambda_q=0.1, **kw)   # no q tensors
    assert ei.value.code == "Q_HEADS_MISSING"

    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(*base, lambda_endpoint=0.1, **kw)  # no ep tensors
    assert ei.value.code == "ENDPOINT_HEADS_MISSING"

    # partial q / endpoint tensors are still refused
    with pytest.raises(KeyAtomV53MPLossError):
        key_atom_loss_v53mp(*base, q_pred=q_pred,
                            lambda_q=0.1, **kw)          # missing q_true/q_mask
    logit, y_ep, _ = _endpoint_head()
    with pytest.raises(KeyAtomV53MPLossError):
        key_atom_loss_v53mp(*base, endpoint_logit=logit, y_endpoint=y_ep,
                            lambda_endpoint=0.1, **kw)   # missing endpoint_mask


def test_negative_lambda_refused() -> None:
    args = _move_args()
    base = (args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"])
    kw = _move_kwargs()
    for bad in ({"lambda_rank": -0.1}, {"lambda_q": -0.1},
                {"lambda_endpoint": -0.1}):
        with pytest.raises(KeyAtomV53MPLossError) as ei:
            key_atom_loss_v53mp(*base, **bad, **kw)
        assert ei.value.code == "LAMBDA_NEG"


def test_graph_count_mismatch_refused() -> None:
    args = _move_args()                     # 2 graphs
    kw = _move_kwargs()
    base = (args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"])
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(
            *base, q_pred=torch.zeros(3), q_true=torch.zeros(3),
            q_mask=torch.ones(3, dtype=torch.bool), lambda_q=0.1, **kw)
    assert ei.value.code == "Q_GRAPH_COUNT"
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(
            *base, endpoint_logit=torch.zeros(3), y_endpoint=torch.zeros(3),
            endpoint_mask=torch.ones(3, dtype=torch.bool),
            lambda_endpoint=0.1, **kw)
    assert ei.value.code == "ENDPOINT_GRAPH_COUNT"
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(
            *base, q_pred=torch.zeros(3),
            order_pairs=torch.tensor([[0], [1]], dtype=torch.long),
            lambda_rank=0.1, **kw)         # ranking also needs [G] q_pred
    assert ei.value.code == "Q_GRAPH_COUNT"


def test_invalid_rank_margin_and_pos_weight_refused() -> None:
    assert RANK_MARGIN_V53MP == 0.1
    args = _move_args()
    base = (args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"])
    kw = _move_kwargs()
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(*base, rank_margin=0.0, **kw)
    assert ei.value.code == "RANK_MARGIN"
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(*base, endpoint_pos_weight=-1.0, **kw)
    assert ei.value.code == "ENDPOINT_POS_WEIGHT"


def test_nonfinite_head_inputs_refused() -> None:
    args = _move_args()
    kw = _move_kwargs()
    base = (args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"])
    q_bad = torch.tensor([float("inf"), 0.0])
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(
            *base, q_pred=q_bad,
            order_pairs=torch.tensor([[0], [1]], dtype=torch.long),
            lambda_rank=0.1, **kw)
    assert ei.value.code == "RANK_NONFINITE"
    logit_bad = torch.tensor([float("nan"), 0.0])
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        key_atom_loss_v53mp(
            *base, endpoint_logit=logit_bad, y_endpoint=torch.tensor([1.0, 0.0]),
            endpoint_mask=torch.ones(2, dtype=torch.bool),
            lambda_endpoint=0.1, **kw)
    assert ei.value.code == "ENDPOINT_NONFINITE"
    # move path: non-finite u is still refused by the frozen R3 loss
    u_bad = args["u"].clone()
    u_bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        key_atom_loss_v53mp(
            u_bad, args["y"], args["degK"], args["key_ptr"], args["pos"],
            args["key_index"], args["pair_index"], **kw)


# ---------------------------------------------------------------------------
# L_ranking math + ordering semantics
# ---------------------------------------------------------------------------
def test_ranking_hand_value() -> None:
    q_pred = torch.tensor([0.0, 1.0])
    pairs = torch.tensor([[0], [1]], dtype=torch.long)
    out = ranking_loss_v53mp(q_pred, pairs)
    # softplus((0.1 − 1.0)/0.1) = softplus(−9) = log1p(e^−9)
    expect = math.log1p(math.exp(-9.0))
    assert float(out["l_rank"]) == pytest.approx(expect, rel=1e-3)
    assert out["n_rank_pairs"] == 1


def test_ranking_satisfied_order_losses_less_than_violated() -> None:
    pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    q_good = torch.tensor([-0.8, 0.0, 0.8])      # j later ⇒ q_j > q_i
    q_bad = torch.tensor([0.8, 0.0, -0.8])       # reversed
    good = ranking_loss_v53mp(q_good, pairs)
    bad = ranking_loss_v53mp(q_bad, pairs)
    assert float(good["l_rank"]) < float(bad["l_rank"])
    assert float(good["l_rank"]) < 0.1       # satisfied pairs ≈ near-zero
    assert good["n_rank_pairs"] == bad["n_rank_pairs"] == 2
    # same ordering at the joint-loss level (λ_r > 0)
    args = _move_args(n_graphs=3)
    kw = _move_kwargs(n_graphs=3)
    res_good = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_good, order_pairs=pairs, lambda_rank=1.0, **kw)
    res_bad = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_bad, order_pairs=pairs, lambda_rank=1.0, **kw)
    assert float(res_good["l_rank"]) < float(res_bad["l_rank"])
    assert float(res_good["loss"]) < float(res_bad["loss"])


def test_empty_order_pairs_exact_zero() -> None:
    # standalone
    out = ranking_loss_v53mp(torch.tensor([0.2, -0.3]),
                             torch.zeros(2, 0, dtype=torch.long))
    assert float(out["l_rank"]) == 0.0 and out["n_rank_pairs"] == 0
    # joint: λ_r > 0 with P == 0 ⇒ term exactly 0 (not an error, not NaN)
    args = _move_args()
    q_pred, _, _ = _q_head()
    res = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_pred, order_pairs=torch.zeros(2, 0, dtype=torch.long),
        lambda_rank=0.5, **_move_kwargs())
    assert float(res["l_rank"]) == 0.0
    assert res["n_rank_pairs"] == 0
    assert bool(torch.isfinite(res["loss"]))


def test_order_pairs_validation() -> None:
    q = torch.tensor([0.0, 0.5, -0.5])
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        ranking_loss_v53mp(q, torch.tensor([[0.0], [1.0]]))   # float dtype
    assert ei.value.code == "RANK_PAIRS_DTYPE"
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        ranking_loss_v53mp(q, torch.tensor([[0, 1, 2]]))      # [1,3] not [2,P]
    assert ei.value.code == "RANK_PAIRS_SHAPE"
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        ranking_loss_v53mp(q, torch.tensor([[0], [3]]))       # 3 >= G
    assert ei.value.code == "RANK_PAIRS_BOUNDS"
    with pytest.raises(KeyAtomV53MPLossError) as ei:
        ranking_loss_v53mp(q, torch.tensor([[-1], [0]]))      # negative
    assert ei.value.code == "RANK_PAIRS_BOUNDS"
    with pytest.raises(KeyAtomV53MPLossError):
        ranking_loss_v53mp(q, None)                           # missing


# ---------------------------------------------------------------------------
# L_progress (reused Huber-0.1) + q-mask semantics
# ---------------------------------------------------------------------------
def test_q_mask_semantics_via_joint() -> None:
    args = _move_args()
    q_pred = torch.tensor([0.0, 5.0])        # masked row has a huge error
    q_true = torch.tensor([0.0, 0.0])
    q_mask = torch.tensor([True, False])
    res = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_pred, q_true=q_true, q_mask=q_mask, lambda_q=0.1,
        **_move_kwargs())
    assert float(res["l_q"]) == 0.0
    assert res["n_q_rows"] == 1
    # frame_weight of the masked row is also irrelevant
    res2 = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_pred, q_true=q_true, q_mask=q_mask, lambda_q=0.1,
        frame_weight=torch.tensor([1.0, 999.0]),
        frame_mask=torch.ones(2, dtype=torch.bool))
    assert float(res2["l_q"]) == 0.0
    assert res2["n_q_rows"] == 1


def test_q_term_reuses_v52_progress_math() -> None:
    """λ_q>0 ⇒ l_q identical to progress_loss_v52 (read-only reuse)."""
    args = _move_args()
    q_pred, q_true, q_mask = _q_head()
    kw = _move_kwargs()
    res = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_pred, q_true=q_true, q_mask=q_mask, lambda_q=0.25, **kw)
    expect = progress_loss_v52(q_pred, q_true, q_mask,
                               frame_weight=kw["frame_weight"])
    assert float(res["l_q"]) == pytest.approx(float(expect["l_q"]), rel=1e-7)
    assert res["n_q_rows"] == expect["n_q_rows"]
    assert Q_HUBER_DELTA_V53MP == 0.1


# ---------------------------------------------------------------------------
# L_endpoint (BCEWithLogits; Mask never in denominator)
# ---------------------------------------------------------------------------
def _bce(logit: float, target: float) -> float:
    return float(F.binary_cross_entropy_with_logits(
        torch.tensor([logit]), torch.tensor([target])))


def test_endpoint_mask_rows_excluded_manual_bce() -> None:
    logit = torch.tensor([2.0, -1.0, 60.0])
    y = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.tensor([True, True, False])     # third row Masked
    pos_weight = 2.0
    out = endpoint_loss_v53mp(logit, y, mask, endpoint_pos_weight=pos_weight)
    manual = (pos_weight * _bce(2.0, 1.0) + 1.0 * _bce(-1.0, 0.0)) / (pos_weight + 1.0)
    assert out["n_endpoint_rows"] == 2
    assert float(out["l_endpoint"]) == pytest.approx(manual, rel=1e-6)
    # had the Mask row been (wrongly) included it would dominate the mean
    out_all = endpoint_loss_v53mp(
        logit, y, torch.ones(3, dtype=torch.bool), endpoint_pos_weight=pos_weight)
    assert float(out_all["l_endpoint"]) > 0.1


def test_endpoint_mask_extreme_row_contributes_nothing() -> None:
    logit = torch.tensor([50.0, -50.0, 0.0])
    y = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.tensor([True, True, False])     # masked, wrong side, extreme
    out = endpoint_loss_v53mp(logit, y, mask, endpoint_pos_weight=3.0)
    assert out["n_endpoint_rows"] == 2
    assert float(out["l_endpoint"]) < 1e-6


def test_endpoint_all_masked_exact_zero() -> None:
    out = endpoint_loss_v53mp(torch.zeros(2), torch.tensor([1.0, 0.0]),
                              torch.zeros(2, dtype=torch.bool))
    assert float(out["l_endpoint"]) == 0.0 and out["n_endpoint_rows"] == 0


def test_endpoint_pos_weight_reweights_classes() -> None:
    logit = torch.tensor([2.0, 0.0])
    y = torch.tensor([1.0, 0.0])
    mask = torch.ones(2, dtype=torch.bool)
    w1 = endpoint_loss_v53mp(logit, y, mask, endpoint_pos_weight=1.0)
    w3 = endpoint_loss_v53mp(logit, y, mask, endpoint_pos_weight=3.0)
    expect1 = (_bce(2.0, 1.0) + _bce(0.0, 0.0)) / 2.0
    expect3 = (3.0 * _bce(2.0, 1.0) + _bce(0.0, 0.0)) / 4.0
    assert float(w1["l_endpoint"]) == pytest.approx(expect1, rel=1e-6)
    assert float(w3["l_endpoint"]) == pytest.approx(expect3, rel=1e-6)
    assert float(w3["l_endpoint"]) != pytest.approx(float(w1["l_endpoint"]),
                                                    rel=1e-3)


def test_endpoint_rejects_non_binary_targets() -> None:
    with pytest.raises(KeyAtomV53MPLossError):
        endpoint_loss_v53mp(torch.zeros(2), torch.tensor([0.5, 0.0]),
                            torch.ones(2, dtype=torch.bool))


def test_endpoint_term_reuses_v52_bce_path_joint() -> None:
    args = _move_args()
    logit, y_ep, ep_mask = _endpoint_head()
    kw = _move_kwargs()
    res = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        endpoint_logit=logit, y_endpoint=y_ep, endpoint_mask=ep_mask,
        endpoint_pos_weight=2.0, lambda_endpoint=0.3, **kw)
    expect = endpoint_loss_v53mp(logit, y_ep, ep_mask,
                                 endpoint_pos_weight=2.0,
                                 frame_weight=kw["frame_weight"])
    assert float(res["l_endpoint"]) == pytest.approx(
        float(expect["l_endpoint"]), rel=1e-7)
    assert res["n_endpoint_rows"] == 1          # one row Masked
    assert float(res["loss"]) == pytest.approx(
        float(res["l_move"]) + 0.3 * float(res["l_endpoint"]), rel=1e-6)


# ---------------------------------------------------------------------------
# Composition + frame_weight-once + gradient flow
# ---------------------------------------------------------------------------
def test_composition_formula_all_terms() -> None:
    args = _move_args()
    q_pred, q_true, q_mask = _q_head()
    logit, y_ep, _ = _endpoint_head()
    ep_mask = torch.ones(2, dtype=torch.bool)
    pairs = torch.tensor([[0], [1]], dtype=torch.long)
    kw = _move_kwargs()
    res = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_pred, q_true=q_true, q_mask=q_mask,
        order_pairs=pairs,
        endpoint_logit=logit, y_endpoint=y_ep, endpoint_mask=ep_mask,
        lambda_rank=0.2, lambda_q=0.05, lambda_endpoint=0.1, **kw)
    l_rank = float(ranking_loss_v53mp(q_pred, pairs,
                                      frame_weight=kw["frame_weight"])["l_rank"])
    l_q = float(progress_loss_v52(q_pred, q_true, q_mask,
                                  frame_weight=kw["frame_weight"])["l_q"])
    l_ep = float(endpoint_loss_v53mp(logit, y_ep, ep_mask,
                                     frame_weight=kw["frame_weight"])["l_endpoint"])
    expect = (float(res["l_move"]) + 0.2 * l_rank
              + 0.05 * l_q + 0.1 * l_ep)
    assert float(res["l_rank"]) == pytest.approx(l_rank, rel=1e-7)
    assert float(res["l_q"]) == pytest.approx(l_q, rel=1e-7)
    assert float(res["l_endpoint"]) == pytest.approx(l_ep, rel=1e-7)
    assert float(res["loss"]) == pytest.approx(expect, rel=1e-6)
    assert (res["lambda_rank"], res["lambda_q"], res["lambda_endpoint"]) == (
        0.2, 0.05, 0.1)
    assert res["n_q_rows"] == 2 and res["n_rank_pairs"] == 1


def test_frame_weight_applied_once_uniform_scale_cancels() -> None:
    """Uniform rescale of ALL frame weights must cancel everywhere."""
    args = _move_args()
    q_pred, q_true, q_mask = _q_head()
    logit, y_ep, ep_mask = _endpoint_head()
    pairs = torch.tensor([[0], [1]], dtype=torch.long)
    head = dict(q_pred=q_pred, q_true=q_true, q_mask=q_mask,
                order_pairs=pairs, endpoint_logit=logit, y_endpoint=y_ep,
                endpoint_mask=ep_mask,
                lambda_rank=0.2, lambda_q=0.05, lambda_endpoint=0.1)
    base = dict(frame_weight=torch.ones(2),
                frame_mask=torch.ones(2, dtype=torch.bool))
    doubled = dict(frame_weight=2.0 * torch.ones(2),
                   frame_mask=torch.ones(2, dtype=torch.bool))
    r1 = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"], **head, **base)
    r2 = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"], **head, **doubled)
    for key in ("l_rank", "l_q", "l_endpoint", "l_move", "loss"):
        assert float(r1[key]) == pytest.approx(float(r2[key]), rel=1e-6), key


def test_gradient_flows_through_all_head_terms() -> None:
    args = _move_args()
    q_pred = torch.tensor([0.3, -0.1], requires_grad=True)
    q_true = torch.tensor([0.1, 0.1])
    q_mask = torch.ones(2, dtype=torch.bool)
    pairs = torch.tensor([[0], [1]], dtype=torch.long)
    logit = torch.tensor([0.5, -0.5], requires_grad=True)
    y_ep = torch.tensor([1.0, 0.0])
    ep_mask = torch.ones(2, dtype=torch.bool)
    res = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_pred, q_true=q_true, q_mask=q_mask,
        order_pairs=pairs,
        endpoint_logit=logit, y_endpoint=y_ep, endpoint_mask=ep_mask,
        lambda_rank=0.5, lambda_q=0.1, lambda_endpoint=0.2,
        **_move_kwargs())
    assert bool(torch.isfinite(res["loss"]))
    res["loss"].backward()
    assert q_pred.grad is not None and float(q_pred.grad.abs().sum()) > 0.0
    assert logit.grad is not None and float(logit.grad.abs().sum()) > 0.0


def test_gradients_reach_model_heads_end_to_end() -> None:
    """Full path: StarEGNNV3+process_head forward → L_v53mp → backward."""
    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
    from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch

    torch.manual_seed(11)
    model = KeyAtomV53MPModel(StarEGNNV3())
    batch = make_v3_batch(n_graphs=2)
    fwd = forward_view_v3(batch)
    out = model(fwd)
    n_graphs = int(out["z"].shape[0])          # bidirectional fixture → 2×curves
    args = _move_args(n_graphs=n_graphs)
    y_ep = torch.tensor([1.0, 0.0] * (n_graphs // 2))
    q_true = torch.tensor([-0.5, 0.5] * (n_graphs // 2))
    order_pairs = torch.tensor(
        [[2 * k for k in range(n_graphs // 2)],
         [2 * k + 1 for k in range(n_graphs // 2)]], dtype=torch.long)
    res = key_atom_loss_v53mp(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=out["q"], q_true=q_true,
        q_mask=torch.ones(n_graphs, dtype=torch.bool),
        order_pairs=order_pairs,
        endpoint_logit=out["endpoint_logit"], y_endpoint=y_ep,
        endpoint_mask=torch.ones(n_graphs, dtype=torch.bool),
        lambda_rank=0.1, lambda_q=0.05, lambda_endpoint=0.1,
        **_move_kwargs(n_graphs=n_graphs))
    assert bool(torch.isfinite(res["loss"]))
    assert res["schema"] == V53MP_LOSS_SCHEMA
    res["loss"].backward()
    assert model.process_head.progress_head.fc1.weight.grad is not None
    assert model.process_head.endpoint_head.fc.weight.grad is not None


def test_teacher_intent_import_unused_guard() -> None:
    # keep the fixture imports honest (they document the SN2 K-set provenance)
    intent = make_sn2_intent("forward")
    assert make_sn2_teacher("forward") is not None
    assert len(intent.key_rows()) == 3
