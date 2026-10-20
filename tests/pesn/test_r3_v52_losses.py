"""V5.2 joint loss tests: L_q math, stop mask semantics, weight freezing.

Frozen campaign ``v52_training`` (loss / stop_class_weights / lambda_mono=0
/ lambda_path=0) + 30h plan §5.3–§5.4.  Covers: Huber_0.1 q math on hand
values, Mask rows excluded from BOTH numerator and denominator, class
weights estimated from inner-train ONLY then frozen (T3.3), λ=0 exact-zero
skip, L_v52 = L_move + λq·L_q + λm·L_TSStop composition, λ_mono/λ_path
refusal, and pass-through of the frozen R3 move loss.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from pes2ts_core.pesn.models.key_atom_loss_v52_v3 import (  # noqa: E402
    LAMBDA_MONO_V52,
    LAMBDA_PATH_V52,
    Q_HUBER_DELTA_V52,
    KeyAtomV52LossError,
    StopClassWeightsV52,
    estimate_stop_class_weights_v52,
    key_atom_loss_v52,
    progress_loss_v52,
    ts_stop_loss_v52,
)
from pes2ts_core.pesn.models.key_atom_losses_v3 import (  # noqa: E402
    key_atom_loss_v3,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_sn2_intent,
    make_sn2_teacher,
)


def _move_args(n_graphs: int = 2, seed: int = 0):
    """Minimal legal move-loss tensors for the SN2 K-set (3 keys/graph)."""
    g = int(n_graphs)
    n_keys = 3 * g
    torch.manual_seed(seed)
    u = torch.randn(n_keys, 3) * 0.01
    y = torch.randn(n_keys, 3) * 0.01
    degK = torch.ones(n_keys)
    key_ptr = torch.arange(0, n_keys + 1, 3, dtype=torch.long)
    # a small key-atom geometry (3 atoms per graph, spread on x)
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


# ---------------------------------------------------------------------------
# L_q (Huber_0.1, weighted mean, mask semantics)
# ---------------------------------------------------------------------------
def test_progress_loss_hand_value() -> None:
    q_pred = torch.tensor([0.2, -0.3])
    q_true = torch.tensor([0.0, 0.0])
    q_mask = torch.tensor([True, True])
    out = progress_loss_v52(q_pred, q_true, q_mask)
    # both |e| > δ=0.1 → linear regime: δ(e − δ/2)
    expect = (Q_HUBER_DELTA_V52 * (0.2 - Q_HUBER_DELTA_V52 / 2)
              + Q_HUBER_DELTA_V52 * (0.3 - Q_HUBER_DELTA_V52 / 2)) / 2
    assert float(out["l_q"]) == pytest.approx(expect, rel=1e-6)
    assert out["n_q_rows"] == 2


def test_progress_loss_quadratic_regime() -> None:
    out = progress_loss_v52(torch.tensor([0.05]), torch.tensor([0.0]),
                            torch.tensor([True]))
    # |e|=0.05 <= δ=0.1 → quadratic: e²/2
    assert float(out["l_q"]) == pytest.approx(0.05 ** 2 / 2, rel=1e-6)


def test_progress_loss_linear_regime() -> None:
    out = progress_loss_v52(torch.tensor([1.0]), torch.tensor([0.0]),
                            torch.tensor([True]))
    # |e|=1 > δ: term = δ(e − δ/2) = 0.1·0.95
    assert float(out["l_q"]) == pytest.approx(0.095, rel=1e-6)


def test_progress_loss_mask_rows_out_of_denominator() -> None:
    # masked row has an enormous error — must NOT affect the loss at all
    q_pred = torch.tensor([0.0, 100.0])
    q_true = torch.tensor([0.0, 0.0])
    mask = torch.tensor([True, False])
    out = progress_loss_v52(q_pred, q_true, mask)
    assert float(out["l_q"]) == 0.0
    assert out["n_q_rows"] == 1
    # frame_weight of the masked row is also irrelevant
    out2 = progress_loss_v52(q_pred, q_true, mask,
                             frame_weight=torch.tensor([1.0, 999.0]))
    assert float(out2["l_q"]) == float(out["l_q"])


def test_progress_loss_all_masked_is_exact_zero() -> None:
    out = progress_loss_v52(torch.zeros(3), torch.zeros(3),
                            torch.zeros(3, dtype=torch.bool))
    assert float(out["l_q"]) == 0.0 and out["n_q_rows"] == 0


def test_progress_loss_frame_weight_once() -> None:
    q_pred = torch.tensor([0.2, 0.0])
    q_true = torch.tensor([0.0, 0.0])
    mask = torch.tensor([True, True])
    w = torch.tensor([2.0, 2.0])
    out = progress_loss_v52(q_pred, q_true, mask, frame_weight=w)
    base = float(progress_loss_v52(q_pred, q_true, mask)["l_q"])
    # uniform rescale of all weights cancels in the weighted mean
    assert float(out["l_q"]) == pytest.approx(base, rel=1e-6)


# ---------------------------------------------------------------------------
# TSStop loss (BCEWithLogits; Mask never in denominator; class weights)
# ---------------------------------------------------------------------------
def test_stop_loss_matches_manual_bce() -> None:
    logit = torch.tensor([2.0, -1.0])
    y = torch.tensor([1.0, 0.0])
    mask = torch.tensor([True, True])
    weights = StopClassWeightsV52(pos_weight=1.0, n_pos=1, n_neg=1)
    out = ts_stop_loss_v52(logit, y, mask, weights)
    manual = float(torch.nn.functional.binary_cross_entropy_with_logits(
        logit, y, reduction="mean"))
    assert float(out["l_ts_stop"]) == pytest.approx(manual, rel=1e-6)


def test_stop_loss_pos_weight_reweights_classes_in_mean() -> None:
    # class weights act through the weighted MEAN (denominator = Σ effective
    # weights over unmasked rows): they re-weight the RELATIVE class
    # influence, so use an asymmetric pair to observe the effect
    logit = torch.tensor([2.0, 0.0])
    y = torch.tensor([1.0, 0.0])
    mask = torch.tensor([True, True])
    b_pos = float(torch.nn.functional.softplus(torch.tensor(-2.0)))
    b_neg = float(torch.nn.functional.softplus(torch.tensor(0.0)))
    w1 = ts_stop_loss_v52(logit, y, mask,
                          StopClassWeightsV52(1.0, 1, 1))
    assert float(w1["l_ts_stop"]) == pytest.approx(
        (b_pos + b_neg) / 2, rel=1e-6)
    w3 = ts_stop_loss_v52(logit, y, mask,
                          StopClassWeightsV52(3.0, 1, 1))
    assert float(w3["l_ts_stop"]) == pytest.approx(
        (3.0 * b_pos + b_neg) / 4.0, rel=1e-6)
    assert float(w3["l_ts_stop"]) != pytest.approx(float(w1["l_ts_stop"]),
                                                   rel=1e-3)


def test_stop_loss_uniform_class_batch_cancels_pos_weight() -> None:
    # documented property of the frozen weighted-mean reduction: with all
    # rows sharing one class the pos_weight cancels between numerator and
    # denominator (relative weighting only — never a free loss scale)
    logit = torch.tensor([2.0])
    y = torch.tensor([1.0])
    mask = torch.tensor([True])
    w1 = ts_stop_loss_v52(logit, y, mask,
                          StopClassWeightsV52(1.0, 1, 1))
    w3 = ts_stop_loss_v52(logit, y, mask,
                          StopClassWeightsV52(3.0, 1, 1))
    assert float(w3["l_ts_stop"]) == pytest.approx(float(w1["l_ts_stop"]),
                                                   rel=1e-6)


def test_stop_loss_mask_never_in_denominator() -> None:
    logit = torch.tensor([50.0, -50.0, 0.0])
    y = torch.tensor([1.0, 0.0, 1.0])
    mask = torch.tensor([True, True, False])   # third row Masked
    weights = StopClassWeightsV52(pos_weight=2.0, n_pos=5, n_neg=1)
    out = ts_stop_loss_v52(logit, y, mask, weights)
    assert out["n_stop_rows"] == 2
    # both unmasked rows are confidently correct → near-zero loss; the
    # masked row (also "wrong side" extreme) contributes NOTHING
    assert float(out["l_ts_stop"]) < 1e-6
    out_all = ts_stop_loss_v52(
        logit, y, torch.tensor([True, True, True]), weights)
    # had the Mask row been (wrongly) included it would dominate the mean
    assert float(out_all["l_ts_stop"]) > 0.1


def test_stop_loss_all_masked_exact_zero() -> None:
    out = ts_stop_loss_v52(torch.zeros(2), torch.tensor([1.0, 0.0]),
                           torch.zeros(2, dtype=torch.bool),
                           StopClassWeightsV52(1.0, 1, 1))
    assert float(out["l_ts_stop"]) == 0.0 and out["n_stop_rows"] == 0


def test_stop_loss_rejects_non_binary_targets() -> None:
    with pytest.raises(KeyAtomV52LossError):
        ts_stop_loss_v52(torch.zeros(2), torch.tensor([0.5, 0.0]),
                         torch.ones(2, dtype=torch.bool),
                         StopClassWeightsV52(1.0, 1, 1))


# ---------------------------------------------------------------------------
# class-weight estimation: inner-train only, then frozen (T3.3)
# ---------------------------------------------------------------------------
def test_estimate_stop_class_weights_ratio() -> None:
    y = torch.tensor([1, 1, 0, 0, 0, 0])
    eligible = torch.ones(6, dtype=torch.bool)
    w = estimate_stop_class_weights_v52(y, eligible, source="inner-train f0")
    assert (w.n_pos, w.n_neg) == (2, 4)
    assert w.pos_weight == pytest.approx(2.0)
    assert w.source == "inner-train f0"


def test_estimate_stop_class_weights_eligible_slices_only() -> None:
    """Structural T3.3 guarantee: rows outside ``eligible`` are invisible.

    The estimator receives ONLY the inner-train slice — dev/confirm rows
    cannot flow back because they are never passed in.
    """
    y_train = torch.tensor([1, 0, 0])
    elig_train = torch.ones(3, dtype=torch.bool)
    w_dev = estimate_stop_class_weights_v52(y_train, elig_train)
    # pretend dev/confirm had a wildly different ratio — it is never seen
    assert w_dev.n_pos + w_dev.n_neg == 3
    assert w_dev.pos_weight == pytest.approx(2.0)   # n_neg/n_pos = 2/1
    # frozen dataclass: cannot be mutated after estimation
    with pytest.raises(Exception):
        w_dev.pos_weight = 99.0  # type: ignore[misc]


def test_estimate_stop_class_weights_zero_positives_never_fabricates() -> None:
    y = torch.tensor([0, 0, 0])
    w = estimate_stop_class_weights_v52(y, torch.ones(3, dtype=torch.bool))
    assert w.n_pos == 0 and w.pos_weight == 1.0   # flagged upstream, no ratio
    # Mask rows (eligible=False) never counted
    y2 = torch.tensor([1, 0, 1, 0])
    elig2 = torch.tensor([False, True, False, True])
    w2 = estimate_stop_class_weights_v52(y2, elig2)
    assert (w2.n_pos, w2.n_neg) == (0, 2)


def test_stop_class_weights_immutable_and_validated() -> None:
    w = StopClassWeightsV52(pos_weight=1.5, n_pos=3, n_neg=2)
    with pytest.raises(Exception):
        w.n_pos = 5  # type: ignore[misc]
    with pytest.raises(KeyAtomV52LossError):
        StopClassWeightsV52(pos_weight=-1.0, n_pos=1, n_neg=1)
    with pytest.raises(KeyAtomV52LossError):
        StopClassWeightsV52(pos_weight=1.0, n_pos=-1, n_neg=1)


# ---------------------------------------------------------------------------
# L_v52 composition + frozen off-switches + move pass-through
# ---------------------------------------------------------------------------
def test_v52_total_is_move_when_lambdas_zero() -> None:
    args = _move_args()
    move_only = key_atom_loss_v3(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"], **_move_kwargs())
    joint = key_atom_loss_v52(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        lambda_q=0.0, lambda_stop=0.0, **_move_kwargs())
    # λ=0 ⇒ head terms skipped entirely; total bit-identical to R3 move loss
    assert torch.equal(joint["loss"], move_only["loss"])
    assert float(joint["l_q"]) == 0.0 and float(joint["l_ts_stop"]) == 0.0
    assert torch.equal(joint["l_move"], move_only["loss"])


def test_v52_composition_formula() -> None:
    args = _move_args()
    q_pred = torch.tensor([0.4, -0.2])
    q_true = torch.tensor([0.1, 0.1])
    q_mask = torch.tensor([True, True])
    logit = torch.tensor([1.0, -1.0])
    y_stop = torch.tensor([1.0, 0.0])
    stop_mask = torch.tensor([True, False])   # one Mask row
    weights = StopClassWeightsV52(pos_weight=2.0, n_pos=4, n_neg=4)
    kwargs = _move_kwargs()
    joint = key_atom_loss_v52(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=q_pred, q_true=q_true, q_mask=q_mask,
        stop_logit=logit, y_stop=y_stop, stop_mask=stop_mask,
        stop_class_weights=weights,
        lambda_q=0.05, lambda_stop=0.02, **kwargs)
    l_q = float(progress_loss_v52(q_pred, q_true, q_mask,
                                  frame_weight=kwargs["frame_weight"])["l_q"])
    l_s = float(ts_stop_loss_v52(logit, y_stop, stop_mask, weights,
                                 frame_weight=kwargs["frame_weight"]
                                 )["l_ts_stop"])
    expect = float(joint["l_move"]) + 0.05 * l_q + 0.02 * l_s
    assert float(joint["loss"]) == pytest.approx(expect, rel=1e-6)
    assert joint["n_stop_rows"] == 1


def test_v52_lambda_mono_path_frozen_zero() -> None:
    assert LAMBDA_MONO_V52 == 0.0 and LAMBDA_PATH_V52 == 0.0
    args = _move_args()
    with pytest.raises(KeyAtomV52LossError):
        key_atom_loss_v52(
            args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"],
            lambda_mono=0.1, **_move_kwargs())
    with pytest.raises(KeyAtomV52LossError):
        key_atom_loss_v52(
            args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"],
            lambda_path=0.1, **_move_kwargs())


def test_v52_heads_missing_is_typed_refusal() -> None:
    args = _move_args()
    with pytest.raises(KeyAtomV52LossError):
        key_atom_loss_v52(
            args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"],
            lambda_q=0.01, **_move_kwargs())     # no q tensors given
    with pytest.raises(KeyAtomV52LossError):
        key_atom_loss_v52(
            args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"],
            lambda_stop=0.01, **_move_kwargs())  # no stop tensors given


def test_v52_graph_count_mismatch_refused() -> None:
    args = _move_args()
    with pytest.raises(KeyAtomV52LossError):
        key_atom_loss_v52(
            args["u"], args["y"], args["degK"], args["key_ptr"],
            args["pos"], args["key_index"], args["pair_index"],
            q_pred=torch.zeros(3), q_true=torch.zeros(3),
            q_mask=torch.ones(3, dtype=torch.bool),
            lambda_q=0.01, **_move_kwargs())     # 3 rows for 2 graphs


def test_v52_end_to_end_through_model() -> None:
    """Full path: StarEGNNV3+heads forward → L_v52 → backward, all finite."""
    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV52Model
    from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch

    torch.manual_seed(11)
    model = KeyAtomV52Model(StarEGNNV3())
    batch = make_v3_batch(n_graphs=2)
    fwd = forward_view_v3(batch)
    out = model(fwd)
    n_graphs = int(out["z"].shape[0])          # bidirectional fixture → 2×curves
    args = _move_args(n_graphs=n_graphs)
    y_stop = torch.tensor([1.0, 0.0] * (n_graphs // 2))
    q_true = torch.tensor([-0.5, 0.5] * (n_graphs // 2))
    weights = estimate_stop_class_weights_v52(
        y_stop, torch.ones(n_graphs, dtype=torch.bool),
        source="fixture inner-train")
    result = key_atom_loss_v52(
        args["u"], args["y"], args["degK"], args["key_ptr"], args["pos"],
        args["key_index"], args["pair_index"],
        q_pred=out["q"], q_true=q_true,
        q_mask=torch.ones(n_graphs, dtype=torch.bool),
        stop_logit=out["stop_logit"], y_stop=y_stop,
        stop_mask=torch.ones(n_graphs, dtype=torch.bool),
        stop_class_weights=weights,
        lambda_q=0.05, lambda_stop=0.02, **_move_kwargs(n_graphs=n_graphs))
    assert bool(torch.isfinite(result["loss"]))
    result["loss"].backward()
    assert model.progress_head.fc1.weight.grad is not None
    assert result["schema"] == "pesn_key_atom_loss_v52_v3"


def test_teacher_intent_import_unused_guard() -> None:
    # keep the fixture imports honest (they document the SN2 K-set provenance)
    intent = make_sn2_intent("forward")
    assert make_sn2_teacher("forward") is not None
    assert len(intent.key_rows()) == 3
