"""V5.4.3 WP1 — the ``q0_raw`` outlet and the ``tanh`` identity.

Plan §5.2 (``q0_raw [G] fp32`` row: "``progress_head.fc2`` 输出（hook）"),
§5.8 (implementation note: "``q0_raw`` 用 ``progress_head.fc2`` 的 forward
hook 捕获；``q`` 仍由 ``process_head(z)`` 产生；两者逐元素验证" and the
equality list "``tanh(q0_raw_V543) == q0_V543``"), §7.1 S0-A (阻断门,
"CPU bitwise；GPU ≤1e-6"), §11.5 (``test_v543_q0_raw_tanh_identity``).

What is proven here:

* the outlet is a TEMPORARY forward hook on
  ``process_head.progress_head.fc2``: it fires exactly once, is removed in
  a ``finally`` (also on exception) and never detaches the output;
* the captured fc2 output is ``[G,1]`` and the frozen packet field is the
  squeezed ``[G]`` column — a squeeze, never a reshape/sum, so the identity
  below stays bit-exact;
* ``q0_raw`` is the PRE-tanh scalar: ``torch.tanh(q0_raw)`` equals the
  head's own ``q`` bitwise on CPU (``torch.equal``; atol 0.0) and within
  1e-6 on GPU — the plan §5.8 tolerance is never loosened;
* the identity holds at tanh saturation (extreme logits) and on a synthetic
  ``ProgressHeadV3`` as well as on a full ``CoarseEncoder`` forward;
* the parity helpers refuse shape/dtype drift and any out-of-tolerance
  element with a typed ``V543_PARITY_*`` code.

CPU-fast, synthetic tensors only."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.models.key_atom_egnn_v3 import Z_DIM_V3  # noqa: E402
from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    KeyAtomV53MPModel,
    ProgressHeadV3,
    ReactionProcessHeadV53MP,
)
from pes2ts_core.pesn.models.v543 import (  # noqa: E402
    PARITY_ATOL_CPU,
    PARITY_ATOL_GPU,
    CoarseEncoder,
    V543ParityError,
    assert_q0_raw_matches_q,
    compare_parity,
    parity_atol,
    single_fire_hook,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_gate_batch,
)

SEED = 29
N_GRAPHS = 3


@pytest.fixture()
def head() -> ProgressHeadV3:
    torch.manual_seed(SEED)
    return ProgressHeadV3()


@pytest.fixture()
def z() -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(N_GRAPHS, Z_DIM_V3)


@pytest.fixture()
def encoder() -> CoarseEncoder:
    torch.manual_seed(SEED)
    legacy = KeyAtomV53MPModel()
    return CoarseEncoder.load(
        {"model": {k: v.clone() for k, v in legacy.state_dict().items()}}, SEED)


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


# ---------------------------------------------------------------------------
# the raw outlet on a synthetic head
# ---------------------------------------------------------------------------
def test_fc2_hook_captures_pre_tanh_q0_raw(head, z) -> None:
    with torch.no_grad():
        with single_fire_hook(head.fc2, "progress_fc2") as captured:
            q = head(z)
    assert len(captured) == 1
    raw2d = captured[0]
    # fc2 is Linear(32,1): the outlet shape is [G,1] (plan §5.2 squeeze)
    assert tuple(raw2d.shape) == (N_GRAPHS, 1)
    assert raw2d.dtype == torch.float32
    q0_raw = raw2d.squeeze(-1)
    assert tuple(q0_raw.shape) == (N_GRAPHS,)
    # plan §5.8 identity 4, CPU bitwise
    assert torch.equal(torch.tanh(q0_raw), q)
    # and it really is PRE-tanh: raw != q (unless every logit is 0)
    assert not torch.equal(q0_raw, q)


def test_q0_raw_matches_q_helper_accepts_identity(head, z) -> None:
    with torch.no_grad():
        with single_fire_hook(head.fc2, "progress_fc2") as captured:
            q = head(z)
    row = assert_q0_raw_matches_q(captured[0].squeeze(-1), q,
                                 context="synthetic ProgressHeadV3")
    assert row["ok"] is True
    assert row["bitwise"] is True
    assert row["max_abs_diff"] == 0.0
    assert row["atol"] == PARITY_ATOL_CPU == 0.0
    assert row["n_elements"] == N_GRAPHS


def test_q0_raw_matches_q_helper_refuses_perturbation(head, z) -> None:
    with torch.no_grad():
        with single_fire_hook(head.fc2, "progress_fc2") as captured:
            q = head(z)
    q0_raw = captured[0].squeeze(-1)
    drifted = q + 1e-6                      # beyond the CPU bitwise bound
    with pytest.raises(V543ParityError) as exc:
        assert_q0_raw_matches_q(q0_raw, drifted)
    assert exc.value.code == "V543_PARITY_Q0_RAW_TANH"
    assert "atol=0.0e+00" in exc.value.detail


def test_tanh_identity_holds_at_saturation() -> None:
    # extreme pre-tanh logits (tanh clamps to exactly ±1.0): the identity is
    # an elementwise tanh of the SAME tensor, so it stays bit-exact
    raw = torch.tensor([-1e3, -40.0, -1e-8, 0.0, 1e-8, 40.0, 1e3])
    q = torch.tanh(raw)
    row = assert_q0_raw_matches_q(raw, q)
    assert row["bitwise"] is True
    assert torch.equal(torch.tanh(raw), q)


# ---------------------------------------------------------------------------
# the outlet on a real encoder forward
# ---------------------------------------------------------------------------
def test_encoder_packet_q0_raw_equals_manual_fc2(encoder, batch) -> None:
    encoder.eval()
    with torch.no_grad():
        packet = encoder(batch)
        head = encoder.process_head.progress_head
        manual = head.fc2(head.act(head.fc1(packet.z))).squeeze(-1)
    assert torch.equal(packet.q0_raw, manual)
    assert tuple(packet.q0_raw.shape) == (packet.n_graphs,)
    # packet.q0 is DERIVED and bitwise the head's q
    assert torch.equal(packet.q0, torch.tanh(packet.q0_raw))


def test_encoder_q0_raw_tanh_identity_on_fixture(encoder, batch) -> None:
    encoder.eval()
    with torch.no_grad():
        packet = encoder(batch)
        head_out = encoder.process_head(packet.z)
    assert torch.equal(torch.tanh(packet.q0_raw), head_out["q"])
    assert head_out["order_score"] is head_out["q"]
    row = assert_q0_raw_matches_q(packet.q0_raw, head_out["q"],
                                 context="CoarseEncoder forward")
    assert row["bitwise"] is True and row["atol"] == 0.0


# ---------------------------------------------------------------------------
# hook lifecycle (plan §5.8: finally-removal + exactly-once)
# ---------------------------------------------------------------------------
def test_single_fire_hook_removes_handle_and_counts(head) -> None:
    assert len(head.fc2._forward_hooks) == 0
    with single_fire_hook(head.fc2, "progress_fc2") as captured:
        assert len(head.fc2._forward_hooks) == 1
        head(torch.zeros(N_GRAPHS, Z_DIM_V3))
    assert len(head.fc2._forward_hooks) == 0
    assert len(captured) == 1


def test_single_fire_hook_removes_handle_on_exception(head) -> None:
    with pytest.raises(RuntimeError):
        with single_fire_hook(head.fc2, "progress_fc2") as captured:
            head(torch.zeros(N_GRAPHS, Z_DIM_V3))
            raise RuntimeError("boom")
    assert len(head.fc2._forward_hooks) == 0
    assert len(captured) == 1          # fired once before the abort


def test_process_head_forward_fires_fc2_once(z) -> None:
    torch.manual_seed(SEED)
    process_head = ReactionProcessHeadV53MP()
    with torch.no_grad():
        with single_fire_hook(
                process_head.progress_head.fc2, "progress_fc2") as captured:
            out = process_head(z)
    assert len(captured) == 1
    assert torch.equal(torch.tanh(captured[0].squeeze(-1)), out["q"])


# ---------------------------------------------------------------------------
# parity helpers: tolerances and typed refusals (plan §5.8)
# ---------------------------------------------------------------------------
def test_parity_tolerances_are_the_frozen_plan_values() -> None:
    assert PARITY_ATOL_CPU == 0.0
    assert PARITY_ATOL_GPU == 1e-6
    cpu = torch.zeros(4)
    assert parity_atol(cpu, cpu) == 0.0
    assert parity_atol(cpu, cpu, atol=1e-3) == 1e-3


def test_compare_parity_audit_row_and_bitwise_flag() -> None:
    a = torch.tensor([0.1, -0.2, 0.3])
    row = compare_parity("z", a.clone(), a)
    assert row == {
        "name": "z", "bitwise": True, "max_abs_diff": 0.0, "atol": 0.0,
        "device": "cpu", "n_elements": 3, "ok": True}
    # an explicit GPU-class tolerance admits a tiny drift
    drift = a + 1e-7
    loose = compare_parity("z", drift, a, atol=PARITY_ATOL_GPU)
    assert loose["bitwise"] is False
    assert 0.0 < loose["max_abs_diff"] <= PARITY_ATOL_GPU
    # and the default CPU bound refuses the same drift
    with pytest.raises(V543ParityError) as exc:
        compare_parity("z", drift, a)
    assert exc.value.code == "V543_PARITY_Z"


def test_compare_parity_refuses_shape_and_dtype_drift() -> None:
    with pytest.raises(V543ParityError) as exc:
        compare_parity("u0", torch.zeros(3, 3), torch.zeros(2, 3))
    assert exc.value.code == "V543_PARITY_SHAPE"
    with pytest.raises(V543ParityError) as exc:
        compare_parity("e0", torch.zeros(2, dtype=torch.float64),
                       torch.zeros(2))
    assert exc.value.code == "V543_PARITY_DTYPE"
    # non-finite values can never pass a parity check
    with pytest.raises(V543ParityError):
        compare_parity("q", torch.tensor([float("nan")]), torch.tensor([0.0]))
