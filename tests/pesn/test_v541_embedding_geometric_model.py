"""V5.4.1 WP3b — model-level battery for the geometric candidates
E3 ``schnet_spatial`` / E4 ``angle_mpnn`` (plan §5.2 gate mechanics
re-exercised with REAL candidates, §6.8 rows 1/4/5/6).

WP2 already froze the branch mechanics with a test-only dummy; this file
runs the same guarantees against the real WP3b candidates:

* ``alpha == 0`` ⇒ five outputs (and ``forward_move``) bitwise equal the
  legacy model sharing the same backbone + process head;
* ``alpha != 0`` ⇒ the gate is real (outputs move), and the normative
  E(3) + permutation harness passes: translation, rotation, reflection,
  within-graph permutation;
* F2 geometry sensitivity at a NON-zero gate: two batches with identical
  F2 topology (same ``edge_attr``, same ``edge_index``, RBF slots zero)
  but different current distances produce different model outputs, and
  the input batch is never mutated;
* supervision keys are refused BEFORE any branch math and the temporary
  ``node_proj`` hook is always removed;
* cold start: ``alpha`` learns first, branch grads are exactly zero;
  after ``alpha != 0`` every branch parameter gets a finite nonzero grad;
* diagnostics: the model's frozen key set is untouched, and the branch's
  ``diagnostics_enabled`` switch yields detached floats or ``{}``;
* independent vs mixed batch predictions agree (key-aligned, fp32
  tolerance) — no cross-graph contamination through the branch.

Registry discipline: both candidate modules are imported INSIDE an
autouse fixture (their imports register ``E3``/``E4``) and the registry
is restored afterwards, so the WP2 "not implemented" tests keep passing
in the same session.
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
from tests.pesn.test_v541_embedding_geometric_fixtures import (
    ABS_TOL,
    REL_TOL,
    assert_close,
    load_e3,
    load_e4,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    bonded_pair_batch,
    forward_batch,
    sphere_center_batch,
)

SEED = 5
INIT_SEED = 7
GATE = 0.25
CANDIDATES = ("E3", "E4")
FIVE_OUTPUTS = ("u", "z", "q", "endpoint_logit", "order_score")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def candidates():
    """Import (⇒ register E3/E4) inside each test, restore afterwards."""
    snap = registry_snapshot()
    modules = {"E3": load_e3(), "E4": load_e4()}
    yield modules
    registry_restore(snap)


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_batch(n_graphs=2))


def _model(embedding_id: str, *, alpha: float = 0.0,
           seed: int = SEED, init_seed: int = INIT_SEED) -> V541Model:
    torch.manual_seed(seed)
    model = V541Model(embedding_id=embedding_id, embedding_init_seed=init_seed)
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


def _geometry_pair(embedding_id: str) -> tuple[dict, dict]:
    """Two batches with IDENTICAL F2 topology but different distances."""
    if embedding_id == "E3":
        return bonded_pair_batch(1.0), bonded_pair_batch(2.0)
    return sphere_center_batch([1.0, 1.0, 1.0]), sphere_center_batch(
        [1.2, 1.2, 1.2])


# ---------------------------------------------------------------------------
# registry / construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_registry_installs_real_candidate(eid: str) -> None:
    assert is_embedding_implemented(eid) is True
    built = build_embedding(eid)
    assert isinstance(built, torch.nn.Module)
    assert sum(p.numel() for p in built.parameters()) > 0
    spec = embedding_spec(eid)
    assert spec.family == ("schnet_spatial" if eid == "E3" else "angle_mpnn")
    assert spec.output == "residual_node_delta"
    assert spec.in_main_matrix is True


@pytest.mark.parametrize("eid", CANDIDATES)
def test_branch_state_dict_and_param_accounting(eid: str) -> None:
    model = _model(eid)
    keys = set(model.state_dict())
    assert {"alpha", "w_out.weight"} <= keys
    branch_keys = {k for k in keys if k.startswith("embedding.")}
    assert branch_keys
    legacy_keys = set(_legacy_view(model).state_dict())
    assert legacy_keys < keys
    counts = model.head_parameter_counts()
    assert counts["embedding"] == sum(
        p.numel() for p in model.embedding.parameters())
    assert counts["w_out"] == HIDDEN_V541 ** 2
    assert counts["alpha"] == 1
    assert (counts["backbone"] + counts["process_head"]
            + counts["embedding"] + counts["w_out"] + counts["alpha"]
            == model.n_parameters)
    # the residual gate itself starts at exactly zero (plan §5.2)
    assert float(model.alpha.detach()) == 0.0
    assert tuple(model.w_out.weight.shape) == (HIDDEN_V541, HIDDEN_V541)


# ---------------------------------------------------------------------------
# alpha == 0: bitwise legacy parity (§5.3 certificate, re-run on E3/E4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_alpha_zero_five_outputs_bitwise(eid: str, batch) -> None:
    model = _model(eid)
    legacy = _legacy_view(model)
    with torch.no_grad():
        out_v = model(batch)
        out_l = legacy(batch)
    assert set(out_v) == set(out_l) == set(FIVE_OUTPUTS)
    for key in FIVE_OUTPUTS:
        assert torch.equal(out_v[key], out_l[key]), key
    assert torch.equal(out_v["order_score"], out_v["q"])


@pytest.mark.parametrize("eid", CANDIDATES)
def test_alpha_zero_forward_move_bitwise(eid: str, batch) -> None:
    model = _model(eid)
    legacy = _legacy_view(model)
    with torch.no_grad():
        move_v = model.forward_move(batch)
        move_l = legacy.forward_move(batch)
        move_bb = model.backbone(batch)
    assert set(move_v) == {"u", "z"}
    for src in (move_l, move_bb):
        assert torch.equal(move_v["u"], src["u"])
        assert torch.equal(move_v["z"], src["z"])


@pytest.mark.parametrize("eid", CANDIDATES)
def test_alpha_zero_leaves_frozen_diagnostics_key_set(eid: str, batch) -> None:
    model = _model(eid)
    with torch.no_grad():
        model(batch)
    assert set(model.diagnostics) == set(DIAGNOSTIC_KEYS_V541)
    assert model.diagnostics["alpha"] == 0.0
    assert model.diagnostics["rms_delta_node"] > 0.0   # the branch RAN
    assert model.diagnostics["node_vector_consumed"] == 0.0


# ---------------------------------------------------------------------------
# alpha != 0: the gate is real + normative E(3)/permutation harness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_nonzero_gate_changes_outputs(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    legacy = _legacy_view(model)
    with torch.no_grad():
        out_v = model(batch)
        out_l = legacy(batch)
    assert not torch.equal(out_v["z"], out_l["z"])     # gate is real
    assert not torch.equal(out_v["u"], out_l["u"])


@pytest.mark.parametrize("eid", CANDIDATES)
def test_nonzero_gate_translation_invariance(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    report = check_translation_invariance(
        model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report


@pytest.mark.parametrize("eid", CANDIDATES)
def test_nonzero_gate_rotation_equivariance(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    report = check_rotation_equivariance(
        model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report


@pytest.mark.parametrize("eid", CANDIDATES)
def test_nonzero_gate_permutation_equivariance(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    report = check_permutation_equivariance(
        model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report


@pytest.mark.parametrize("eid", CANDIDATES)
def test_nonzero_gate_reflection_equivariance(eid: str, batch) -> None:
    """Improper orthogonal transform (det = −1): polar ``u`` obeys
    ``u(R x) = R u(x)`` at the normative tolerance."""
    model = _model(eid, alpha=GATE)
    R = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    rot = dict(batch)
    rot["pos"] = batch["pos"] @ R.T
    with torch.no_grad():
        a = model(batch)
        b = model(rot)
    errors = ((a["u"] @ R.T) - b["u"]).norm(dim=-1)
    scale = float(a["u"].abs().max())
    assert float(errors.max()) <= 1e-6 + 1e-5 * scale, float(errors.max())


# ---------------------------------------------------------------------------
# §6.8 row 1: F2 mask untouched, distance changes, NON-zero gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_nonzero_gate_geometry_sensitive_f2_untouched(eid: str) -> None:
    near, far = _geometry_pair(eid)
    # identical F2 topology: same edges, same packed attrs, RBF slots zero
    assert torch.equal(near["edge_index"], far["edge_index"])
    assert torch.equal(near["edge_attr"], far["edge_attr"])
    assert bool((near["edge_attr"][:, :12] == 0).all())
    assert bool((far["edge_attr"][:, :12] == 0).all())

    model = _model(eid, alpha=GATE)
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in near.items()}
    with torch.no_grad():
        out_near = model(near)
        out_far = model(far)
    # the current distance really moves the prediction at a live gate
    assert not torch.allclose(out_near["z"], out_far["z"],
                              rtol=REL_TOL, atol=ABS_TOL)
    assert float((out_near["z"] - out_far["z"]).abs().max()) > 1e-4
    # and the candidate never wrote back into the F2 batch
    for key, value in near.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key]), key
    # the per-graph scalar head moves too (the gate is real end-to-end);
    # compared BITWISE because the real shift (~5e-6) sits inside the
    # fp32 tolerance used for the z comparison above.
    assert not torch.equal(out_near["q"], out_far["q"])


# ---------------------------------------------------------------------------
# leak + hook lifecycle (WP2 guarantees, re-run on real candidates)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_supervision_leak_refused_before_branch_math(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    leaked = dict(batch)
    leaked["y_teacher"] = torch.zeros(int(batch["key_index"].numel()), 3)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        model(leaked)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    assert len(model.backbone.node_proj._forward_hooks) == 0
    assert model.diagnostics == {}                    # never populated
    # the model still works afterwards (no leaked state)
    with torch.no_grad():
        out = model(batch)
    assert tuple(out["u"].shape) == (int(batch["key_index"].numel()), 3)


@pytest.mark.parametrize("eid", CANDIDATES)
def test_hook_removed_after_normal_forward(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    assert len(model.backbone.node_proj._forward_hooks) == 0
    with torch.no_grad():
        model(batch)
        model.forward_move(batch)
    assert len(model.backbone.node_proj._forward_hooks) == 0


@pytest.mark.parametrize("eid", CANDIDATES)
def test_metadata_intervention_does_not_change_output(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    alt = dict(batch)
    alt["endpoint_names"] = ("X_LABEL", "Y_LABEL")
    with torch.no_grad():
        a = model(batch)
        b = model(alt)
    for key in FIVE_OUTPUTS:
        assert torch.equal(a[key], b[key]), key


# ---------------------------------------------------------------------------
# gradients (plan §5.2 cold start, re-run on real candidates)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_cold_start_alpha_grad_only(eid: str, batch) -> None:
    model = _model(eid)                             # alpha = 0
    _loss(model, batch).backward()
    assert model.alpha.grad is not None
    assert abs(float(model.alpha.grad)) > 0.0        # alpha learns at step 0
    assert model.w_out.weight.grad is not None
    assert torch.equal(model.w_out.weight.grad,
                       torch.zeros_like(model.w_out.weight.grad))
    for p in model.embedding.parameters():
        assert p.grad is not None                    # in the graph…
        assert torch.equal(p.grad, torch.zeros_like(p.grad))  # …× alpha


@pytest.mark.parametrize("eid", CANDIDATES)
def test_after_alpha_nonzero_branch_grads_flow_and_are_finite(
        eid: str, batch) -> None:
    model = _model(eid)
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
    # legacy pathway keeps receiving gradients too
    assert model.backbone.node_proj.weight.grad is not None
    assert bool(torch.isfinite(model.backbone.node_proj.weight.grad).all())
    assert model.process_head.progress_head.fc1.weight.grad is not None


@pytest.mark.parametrize("eid", CANDIDATES)
def test_position_gradients_reach_geometry_and_are_finite(
        eid: str, batch) -> None:
    """The branch reads the CURRENT geometry: ``pos`` must receive a
    finite (and nonzero) gradient through the candidate."""
    pos = batch["pos"].detach().clone().requires_grad_(True)
    work = dict(batch)
    work["pos"] = pos
    model = _model(eid, alpha=GATE)
    out = model(work)
    (out["z"].sum() + out["q"].sum()).backward()
    assert pos.grad is not None
    assert bool(torch.isfinite(pos.grad).all())
    assert float(pos.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# diagnostics switch (model-level)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_diagnostics_switch_model_and_branch(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    with torch.no_grad():
        model(batch)
    # branch diagnostics: detached floats with the candidate's own keys
    branch_diag = model.embedding.last_diagnostics
    own_keys = ("zero_neighbor",) if eid == "E3" else ("zero_angle",
                                                       "n_triplets")
    for key in own_keys:
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
@pytest.mark.parametrize("eid", CANDIDATES)
def test_independent_vs_mixed_batch_predictions(eid: str) -> None:
    batch_a = forward_batch([make_sn2_sample("forward")])
    batch_b = forward_batch([make_sn2_sample("reverse", jitter_seed=7)])
    batch_ab = forward_batch([
        make_sn2_sample("forward"), make_sn2_sample("reverse", jitter_seed=7)])
    n_a = int(batch_a["pos"].shape[0])

    model = _model(eid, alpha=GATE)
    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
        out_ab = model(batch_ab)

    # per-graph outputs: graph 0 is A, graph 1 is B in the mixed batch
    assert_close(out_a["z"], out_ab["z"][:1], f"{eid} mixed z(A)")
    assert_close(out_b["z"], out_ab["z"][1:], f"{eid} mixed z(B)")
    for key in ("q", "endpoint_logit", "order_score"):
        assert_close(out_a[key], out_ab[key][:1], f"{eid} mixed {key}(A)")
        assert_close(out_b[key], out_ab[key][1:], f"{eid} mixed {key}(B)")

    # per-key outputs: align rows by the key atom's label (index)
    key_a = batch_a["key_index"]
    key_ab = batch_ab["key_index"]
    in_a = batch_ab["graph_index"][key_ab] == 0
    assert torch.equal(key_ab[in_a], key_a)          # same keys, same order
    assert_close(out_a["u"], out_ab["u"][in_a], f"{eid} mixed u(A)")
    in_b = batch_ab["graph_index"][key_ab] == 1
    key_b = batch_b["key_index"]
    assert torch.equal(key_ab[in_b] - n_a, key_b)
    assert_close(out_b["u"], out_ab["u"][in_b], f"{eid} mixed u(B)")


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_eval_determinism(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    with torch.no_grad():
        a = model(batch)
        b = model(batch)
    for key in FIVE_OUTPUTS:
        assert torch.equal(a[key], b[key]), key
