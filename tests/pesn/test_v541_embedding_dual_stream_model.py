"""V5.4.1 WP3b — model-level battery for E5 ``dual_stream`` (plan §5.2
gate mechanics re-exercised with the REAL candidate, §6.7 E5).

WP2 froze the branch mechanics with a test-only dummy and WP3b re-ran
them against E1–E4; this file runs the same guarantees against E5:

* ``alpha == 0`` ⇒ five outputs (and ``forward_move``) bitwise equal the
  legacy model sharing the same backbone + process head;
* ``alpha != 0`` ⇒ the gate is real (outputs move) and the normative
  E(3) + permutation harness passes: translation, rotation, reflection,
  within-graph permutation;
* supervision keys are refused BEFORE any branch math and the temporary
  ``node_proj`` hook is always removed;
* cold start: ``alpha`` learns first, branch grads are exactly zero;
  after ``alpha != 0`` every branch parameter gets a finite nonzero grad;
* INDEPENDENT RNG (plan §5.4): the branch build consumes the global
  stream exactly like a pure legacy build (``fork_rng``), the same
  ``embedding_init_seed`` reproduces the branch bit-for-bit, and the
  parent/legacy weights are NEVER copied into the branch;
* parameter accounting with the EXTERNAL ``W``/``alpha`` reported
  separately from the E5 branch itself (E5 adds no internal gate);
* the branch's ``diagnostics_enabled`` switch yields detached floats
  (shared keys + ``chem.``/``angle.`` prefixes) or ``{}``;
* independent vs mixed batch predictions agree (key-aligned, fp32
  tolerance) — no cross-graph contamination through either stream.

Registry discipline: the E5 module is imported INSIDE an autouse fixture
(its import registers ``E5``) and the registry is restored afterwards,
so the WP2 "not implemented" tests keep passing in the same session.
"""

from __future__ import annotations

import pytest
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel
from pes2ts_core.pesn.models.v541 import (
    DIAGNOSTIC_KEYS_V541,
    HIDDEN_V541,
    MEASURED_PARAMS_V541_E0_JOINT,
    V541Model,
    embedding_spec,
    is_embedding_implemented,
    registry_restore,
    registry_snapshot,
)
from pes2ts_core.pesn.models.v541.embedding_registry import build_embedding
from scripts.agent_ops.feature_gate.equivariance_harness import (
    check_permutation_equivariance,
    check_rotation_equivariance,
    check_translation_invariance,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (
    make_sn2_sample,
    make_v3_batch,
)
from tests.pesn.test_v541_embedding_dual_stream_fixtures import (
    ABS_TOL,
    E5_BRANCH_PARAMS_CANDIDATE,
    REL_TOL,
    assert_close,
    load_e5,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    forward_batch,
    sphere_center_batch,
)

SEED = 5
INIT_SEED = 7
GATE = 0.25
EID = "E5"
FIVE_OUTPUTS = ("u", "z", "q", "endpoint_logit", "order_score")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def candidate():
    """Import (⇒ register E5) inside each test, restore afterwards."""
    snap = registry_snapshot()
    module = load_e5()
    yield module
    registry_restore(snap)


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_batch(n_graphs=2))


def _model(*, alpha: float = 0.0, seed: int = SEED,
           init_seed: int = INIT_SEED) -> V541Model:
    torch.manual_seed(seed)
    model = V541Model(embedding_id=EID, embedding_init_seed=init_seed)
    if alpha:
        model.alpha.data.fill_(float(alpha))
    return model.eval()


def _legacy_view(model: V541Model) -> KeyAtomV53MPModel:
    view = KeyAtomV53MPModel(model.backbone)
    view.process_head = model.process_head
    return view.eval()


def _loss(model: V541Model, batch) -> torch.Tensor:
    out = model(batch)
    return out["u"].sum() + out["z"].sum() + out["q"].sum()


# ---------------------------------------------------------------------------
# registry / construction / accounting
# ---------------------------------------------------------------------------
def test_registry_installs_real_candidate(candidate) -> None:
    assert is_embedding_implemented(EID) is True
    built = build_embedding(EID)
    assert isinstance(built, candidate.DualStreamEmbedding)
    assert sum(p.numel() for p in built.parameters()) > 0
    assert candidate.EMBEDDING_ID_E5 == EID
    spec = embedding_spec(EID)
    assert spec.family == "dual_stream"
    assert spec.output == "residual_node_delta"
    assert spec.conditional_parent == ("E1", "E4")


def test_branch_state_dict_and_split_param_accounting() -> None:
    """Parameter totals reported SEPARATELY: the E5 branch itself, then
    the external §5.2 ``W``/``alpha`` gate, then the joint."""
    model = _model()
    keys = set(model.state_dict())
    assert {"alpha", "w_out.weight"} <= keys
    branch_keys = {k for k in keys if k.startswith("embedding.")}
    assert branch_keys
    assert any(k.startswith("embedding.chem_branch.") for k in branch_keys)
    assert any(k.startswith("embedding.angle_branch.") for k in branch_keys)
    assert "embedding.chem_in.weight" in keys
    assert "embedding.angle_in.weight" in keys
    assert "embedding.fusion.weight" in keys
    legacy_keys = set(_legacy_view(model).state_dict())
    assert legacy_keys < keys
    counts = model.head_parameter_counts()
    # (a) the branch alone — E5 adds NO internal gate
    assert counts["embedding"] == E5_BRANCH_PARAMS_CANDIDATE
    assert counts["embedding"] == sum(
        p.numel() for p in model.embedding.parameters())
    # (b) the EXTERNAL gate on top (the single §5.2 W/alpha)
    assert counts["w_out"] == HIDDEN_V541 ** 2
    assert counts["alpha"] == 1
    # (c) the joints
    assert (counts["backbone"] + counts["process_head"]
            + counts["embedding"] + counts["w_out"] + counts["alpha"]
            == model.n_parameters)
    assert model.n_parameters == (
        MEASURED_PARAMS_V541_E0_JOINT + counts["embedding"]
        + counts["w_out"] + counts["alpha"])
    # the residual gate itself starts at exactly zero (plan §5.2)
    assert float(model.alpha.detach()) == 0.0
    assert tuple(model.w_out.weight.shape) == (HIDDEN_V541, HIDDEN_V541)


# ---------------------------------------------------------------------------
# alpha == 0: bitwise legacy parity (§5.3 certificate, re-run on E5)
# ---------------------------------------------------------------------------
def test_alpha_zero_five_outputs_bitwise(batch) -> None:
    model = _model()
    legacy = _legacy_view(model)
    with torch.no_grad():
        out_v = model(batch)
        out_l = legacy(batch)
    assert set(out_v) == set(out_l) == set(FIVE_OUTPUTS)
    for key in FIVE_OUTPUTS:
        assert torch.equal(out_v[key], out_l[key]), key
    assert torch.equal(out_v["order_score"], out_v["q"])


def test_alpha_zero_forward_move_bitwise(batch) -> None:
    model = _model()
    legacy = _legacy_view(model)
    with torch.no_grad():
        move_v = model.forward_move(batch)
        move_l = legacy.forward_move(batch)
        move_bb = model.backbone(batch)
    assert set(move_v) == {"u", "z"}
    for src in (move_l, move_bb):
        assert torch.equal(move_v["u"], src["u"])
        assert torch.equal(move_v["z"], src["z"])


def test_alpha_zero_leaves_frozen_diagnostics_key_set(batch) -> None:
    model = _model()
    with torch.no_grad():
        model(batch)
    assert set(model.diagnostics) == set(DIAGNOSTIC_KEYS_V541)
    assert model.diagnostics["alpha"] == 0.0
    assert model.diagnostics["rms_delta_node"] > 0.0   # the branch RAN
    assert model.diagnostics["node_vector_consumed"] == 0.0


# ---------------------------------------------------------------------------
# alpha != 0: the gate is real + normative E(3)/permutation harness
# ---------------------------------------------------------------------------
def test_nonzero_gate_changes_outputs(batch) -> None:
    model = _model(alpha=GATE)
    legacy = _legacy_view(model)
    with torch.no_grad():
        out_v = model(batch)
        out_l = legacy(batch)
    assert not torch.equal(out_v["z"], out_l["z"])     # gate is real
    assert not torch.equal(out_v["u"], out_l["u"])


def test_nonzero_gate_translation_invariance(batch) -> None:
    model = _model(alpha=GATE)
    report = check_translation_invariance(
        model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report


def test_nonzero_gate_rotation_equivariance(batch) -> None:
    model = _model(alpha=GATE)
    report = check_rotation_equivariance(
        model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report


def test_nonzero_gate_permutation_equivariance(batch) -> None:
    model = _model(alpha=GATE)
    report = check_permutation_equivariance(
        model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report


def test_nonzero_gate_reflection_equivariance(batch) -> None:
    """Improper orthogonal transform (det = −1): polar ``u`` obeys
    ``u(R x) = R u(x)`` at the normative tolerance."""
    model = _model(alpha=GATE)
    R = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    rot = dict(batch)
    rot["pos"] = batch["pos"] @ R.T
    with torch.no_grad():
        a = model(batch)
        b = model(rot)
    errors = ((a["u"] @ R.T) - b["u"]).norm(dim=-1)
    scale = float(a["u"].abs().max())
    assert float(errors.max()) <= 1e-6 + 1e-5 * scale, float(errors.max())


def test_nonzero_gate_geometry_sensitive_f2_untouched() -> None:
    """Identical F2 topology, different current radii: the prediction
    moves at a live gate; the batch is never mutated."""
    near, far = sphere_center_batch([1.0, 1.0, 1.0]), sphere_center_batch(
        [1.2, 1.2, 1.2])
    assert torch.equal(near["edge_index"], far["edge_index"])
    assert torch.equal(near["edge_attr"], far["edge_attr"])
    assert bool((near["edge_attr"][:, :12] == 0).all())
    assert bool((far["edge_attr"][:, :12] == 0).all())

    model = _model(alpha=GATE)
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in near.items()}
    with torch.no_grad():
        out_near = model(near)
        out_far = model(far)
    assert not torch.allclose(out_near["z"], out_far["z"],
                              rtol=REL_TOL, atol=ABS_TOL)
    assert float((out_near["z"] - out_far["z"]).abs().max()) > 1e-4
    for key, value in near.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key]), key


# ---------------------------------------------------------------------------
# leak + hook lifecycle (WP2 guarantees, re-run on the real candidate)
# ---------------------------------------------------------------------------
def test_supervision_leak_refused_before_branch_math(batch) -> None:
    model = _model(alpha=GATE)
    leaked = dict(batch)
    leaked["y_teacher"] = torch.zeros(int(batch["key_index"].numel()), 3)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        model(leaked)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    assert len(model.backbone.node_proj._forward_hooks) == 0
    assert model.diagnostics == {}                    # never populated
    with torch.no_grad():
        out = model(batch)
    assert tuple(out["u"].shape) == (int(batch["key_index"].numel()), 3)


def test_hook_removed_after_normal_forward(batch) -> None:
    model = _model(alpha=GATE)
    assert len(model.backbone.node_proj._forward_hooks) == 0
    with torch.no_grad():
        model(batch)
        model.forward_move(batch)
    assert len(model.backbone.node_proj._forward_hooks) == 0


def test_metadata_intervention_does_not_change_output(batch) -> None:
    model = _model(alpha=GATE)
    alt = dict(batch)
    alt["endpoint_names"] = ("X_LABEL", "Y_LABEL")
    with torch.no_grad():
        a = model(batch)
        b = model(alt)
    for key in FIVE_OUTPUTS:
        assert torch.equal(a[key], b[key]), key


# ---------------------------------------------------------------------------
# gradients (plan §5.2 cold start, re-run on E5)
# ---------------------------------------------------------------------------
def test_cold_start_alpha_grad_only(batch) -> None:
    model = _model()                               # alpha = 0
    _loss(model, batch).backward()
    assert model.alpha.grad is not None
    assert abs(float(model.alpha.grad)) > 0.0        # alpha learns at step 0
    assert model.w_out.weight.grad is not None
    assert torch.equal(model.w_out.weight.grad,
                       torch.zeros_like(model.w_out.weight.grad))
    for p in model.embedding.parameters():
        assert p.grad is not None                    # in the graph…
        assert torch.equal(p.grad, torch.zeros_like(p.grad))  # …× alpha


def test_after_alpha_nonzero_branch_grads_flow_and_are_finite(
        batch) -> None:
    model = _model()
    _loss(model, batch).backward()                  # cold step
    model.zero_grad(set_to_none=True)
    model.alpha.data.fill_(0.1)
    _loss(model, batch).backward()
    assert float(model.w_out.weight.grad.abs().sum()) > 0.0
    branch_grown = 0
    for pname, p in model.embedding.named_parameters():
        assert p.grad is not None, pname
        assert bool(torch.isfinite(p.grad).all()), pname
        if float(p.grad.abs().sum()) > 0.0:
            branch_grown += 1
    assert branch_grown > 0, "no branch parameter received a gradient"
    assert abs(float(model.alpha.grad)) > 0.0
    # both streams carried gradient (both views were live on this batch)
    chem_grown = any(
        float(p.grad.abs().sum()) > 0.0
        for n, p in model.embedding.chem_branch.named_parameters())
    angle_grown = any(
        float(p.grad.abs().sum()) > 0.0
        for n, p in model.embedding.angle_branch.named_parameters())
    assert chem_grown and angle_grown
    # legacy pathway keeps receiving gradients too
    assert model.backbone.node_proj.weight.grad is not None
    assert bool(torch.isfinite(model.backbone.node_proj.weight.grad).all())
    assert model.process_head.progress_head.fc1.weight.grad is not None


def test_position_gradients_reach_geometry_and_are_finite(batch) -> None:
    """The branch reads the CURRENT geometry (angle RBF/cosine): ``pos``
    must receive a finite (and nonzero) gradient through E5."""
    pos = batch["pos"].detach().clone().requires_grad_(True)
    work = dict(batch)
    work["pos"] = pos
    model = _model(alpha=GATE)
    out = model(work)
    (out["z"].sum() + out["q"].sum()).backward()
    assert pos.grad is not None
    assert bool(torch.isfinite(pos.grad).all())
    assert float(pos.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# independent RNG (plan §5.4) + no parent weight copy
# ---------------------------------------------------------------------------
def test_branch_init_preserves_legacy_rng_sequence() -> None:
    torch.manual_seed(123)
    legacy = KeyAtomV53MPModel()
    state_legacy = torch.random.get_rng_state()
    torch.manual_seed(123)
    branch = V541Model(embedding_id=EID, embedding_init_seed=999)
    state_branch = torch.random.get_rng_state()
    assert torch.equal(state_legacy, state_branch)
    # ⇒ the legacy weights themselves are bitwise identical too
    assert all(torch.equal(legacy.state_dict()[k], branch.state_dict()[k])
               for k in legacy.state_dict())


def test_branch_init_reproducible_per_seed() -> None:
    a = _model(init_seed=11)
    b = _model(init_seed=11)
    c = _model(init_seed=12)
    assert torch.equal(a.w_out.weight, b.w_out.weight)
    assert all(torch.equal(pa, pb) for pa, pb in
               zip(a.embedding.parameters(), b.embedding.parameters()))
    assert any(not torch.equal(pa, pc) for pa, pc in
               zip(a.embedding.parameters(), c.embedding.parameters()))


def test_branch_build_consumes_no_global_rng() -> None:
    torch.manual_seed(321)
    V541Model(embedding_id=EID, embedding_init_seed=42)
    branch_state = torch.random.get_rng_state()
    torch.manual_seed(321)
    KeyAtomV53MPModel()
    legacy_state = torch.random.get_rng_state()
    assert torch.equal(branch_state, legacy_state)


def test_parent_weights_are_never_copied_into_branch() -> None:
    """No branch parameter is a copy of any legacy parameter (shape-
    compatible pairs must differ), and the legacy core is untouched by
    branch construction under a different seed stream."""
    model = _model()
    legacy_sd = model.legacy_state_dict()
    for pname, p in model.embedding.named_parameters():
        for lname, t in legacy_sd.items():
            if tuple(p.shape) == tuple(t.shape):
                assert not torch.equal(p.detach(), t), (pname, lname)
    # the external W is a fresh orthogonal init, not an identity/legacy copy
    assert not torch.equal(
        model.w_out.weight.detach(),
        torch.eye(HIDDEN_V541, dtype=model.w_out.weight.dtype))


# ---------------------------------------------------------------------------
# diagnostics switch (model-level)
# ---------------------------------------------------------------------------
def test_diagnostics_switch_model_and_branch(batch) -> None:
    model = _model(alpha=GATE)
    with torch.no_grad():
        model(batch)
    # branch diagnostics: detached floats, shared keys + source prefixes
    branch_diag = model.embedding.last_diagnostics
    for key in ("n_edges", "n_triplets", "zero_neighbor_frac",
                "zero_angle_frac", "chem.n_chem_pairs", "angle.n_triplets"):
        assert key in branch_diag, key
        assert isinstance(branch_diag[key], float)
    # model-level switch: frozen key set, then empty when disabled
    assert set(model.diagnostics) == set(DIAGNOSTIC_KEYS_V541)
    model.diagnostics_enabled = False
    with torch.no_grad():
        model(batch)
    assert model.diagnostics == {}
    # branch-level switch (the candidate's own WP3a flag)
    model.embedding.diagnostics_enabled = False
    with torch.no_grad():
        model(batch)
    assert model.embedding.last_diagnostics == {}
    model.embedding.diagnostics_enabled = True
    with torch.no_grad():
        model(batch)
    assert model.embedding.last_diagnostics


# ---------------------------------------------------------------------------
# §6.8 row 6: independent vs mixed batch predictions
# ---------------------------------------------------------------------------
def test_independent_vs_mixed_batch_predictions() -> None:
    batch_a = forward_batch([make_sn2_sample("forward")])
    batch_b = forward_batch([make_sn2_sample("reverse", jitter_seed=7)])
    batch_ab = forward_batch([
        make_sn2_sample("forward"), make_sn2_sample("reverse", jitter_seed=7)])
    n_a = int(batch_a["pos"].shape[0])

    model = _model(alpha=GATE)
    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
        out_ab = model(batch_ab)

    # per-graph outputs: graph 0 is A, graph 1 is B in the mixed batch
    assert_close(out_a["z"], out_ab["z"][:1], "E5 mixed z(A)")
    assert_close(out_b["z"], out_ab["z"][1:], "E5 mixed z(B)")
    for key in ("q", "endpoint_logit", "order_score"):
        assert_close(out_a[key], out_ab[key][:1], f"E5 mixed {key}(A)")
        assert_close(out_b[key], out_ab[key][1:], f"E5 mixed {key}(B)")

    # per-key outputs: align rows by the key atom's label (index)
    key_a = batch_a["key_index"]
    key_ab = batch_ab["key_index"]
    in_a = batch_ab["graph_index"][key_ab] == 0
    assert torch.equal(key_ab[in_a], key_a)          # same keys, same order
    assert_close(out_a["u"], out_ab["u"][in_a], "E5 mixed u(A)")
    in_b = batch_ab["graph_index"][key_ab] == 1
    key_b = batch_b["key_index"]
    assert torch.equal(key_ab[in_b] - n_a, key_b)
    assert_close(out_b["u"], out_ab["u"][in_b], "E5 mixed u(B)")


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_eval_determinism(batch) -> None:
    model = _model(alpha=GATE)
    with torch.no_grad():
        a = model(batch)
        b = model(batch)
    for key in FIVE_OUTPUTS:
        assert torch.equal(a[key], b[key]), key
