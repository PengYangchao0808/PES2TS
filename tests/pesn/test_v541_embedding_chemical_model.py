"""V5.4.1 WP3b — model-level battery for the chemical candidates
E1 ``gin_e_chemical`` / E2 ``pna_chemical`` (plan §5.2 gate mechanics
re-exercised with REAL candidates, §6.8 rows 1/5/6).

WP2 already froze the branch mechanics with a test-only dummy; this file
runs the same guarantees against the real WP3b chemical candidates:

* ``alpha == 0`` ⇒ five outputs (and ``forward_move``) bitwise equal the
  legacy model sharing the same backbone + process head;
* ``alpha != 0`` ⇒ the gate is real (outputs move), and the normative
  E(3) + permutation harness passes: translation, rotation, reflection,
  within-graph permutation;
* the branch ``delta_node`` is an INVARIANT scalar channel (E1/E2 read
  no geometry): it is bitwise pos-invariant, and since F2 masks every
  distance feature AND the branch recomputes nothing geometric, the
  whole model's invariant scalars (z/q) are bitwise pos-invariant too
  (only E3/E4 recompute geometry, §6.6.1);
* supervision keys are refused BEFORE any branch math and the temporary
  ``node_proj`` hook is always removed;
* cold start: ``alpha`` learns first, branch grads are exactly zero;
  after ``alpha != 0`` every branch parameter gets a finite nonzero grad;
* the model's frozen diagnostics key set is untouched, and the branch's
  ``diagnostics_enabled`` switch yields detached floats or ``{}``;
* independent vs mixed batch predictions agree (key-aligned, fp32
  tolerance) — no cross-graph contamination through the branch;
* E2-specific: a branch model refuses forward until its frozen degree
  stats arrive via ``set_degree_stats`` (typed refusal), and the stats
  buffers round-trip through ``state_dict``.

Registry discipline: both candidate modules are imported INSIDE an
autouse fixture (their imports register ``E1``/``E2``) and the registry
is restored afterwards, so the WP2 "not implemented" tests keep passing
in the same session.
"""

from __future__ import annotations

import math

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
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch
from tests.pesn.test_v541_embedding_chemical_fixtures import (
    load_e1,
    load_e2,
)
from tests.pesn.test_v541_feature_providers_fixtures import (
    forward_batch,
    transform_forward_batch,
)

SEED = 5
INIT_SEED = 7
GATE = 0.25
CANDIDATES = ("E1", "E2")
FIVE_OUTPUTS = ("u", "z", "q", "endpoint_logit", "order_score")
#: frozen E2 stats for the model battery (mu_log = ln2 > 0)
MU_LN2 = math.log(2.0)
SHA_OK = "ab" * 32


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def candidates():
    """Import (⇒ register E1/E2) inside each test, restore afterwards."""
    snap = registry_snapshot()
    modules = {"E1": load_e1(), "E2": load_e2()}
    yield modules
    registry_restore(snap)


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_batch(n_graphs=2))


def _model(embedding_id: str, *, alpha: float = 0.0,
           seed: int = SEED, init_seed: int = INIT_SEED) -> V541Model:
    torch.manual_seed(seed)
    model = V541Model(embedding_id=embedding_id, embedding_init_seed=init_seed)
    if embedding_id == "E2":
        # E2 consumes FROZEN inner-train degree stats (never fitted here)
        model.embedding.set_degree_stats(MU_LN2, SHA_OK)
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
# registry / construction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_registry_installs_real_candidate(eid: str, candidates) -> None:
    assert is_embedding_implemented(eid) is True
    built = build_embedding(eid)
    assert isinstance(built, torch.nn.Module)
    assert sum(p.numel() for p in built.parameters()) > 0
    spec = embedding_spec(eid)
    assert spec.family == ("gin_e_chemical" if eid == "E1" else "pna_chemical")
    assert spec.output == "residual_node_delta"
    assert spec.in_main_matrix is True
    # the module's frozen registry id constant matches the campaign row
    module_id = (candidates["E1"].EMBEDDING_ID_E1 if eid == "E1"
                 else candidates["E2"].EMBEDDING_ID_E2)
    assert module_id == eid


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
    # E2's degree stats are BUFFERS (checkpointable provenance, §7.3)
    if eid == "E2":
        assert {"embedding.mu_log", "embedding.stats_sha256",
                "embedding.stats_ready"} <= keys


# ---------------------------------------------------------------------------
# alpha == 0: bitwise legacy parity (§5.3 certificate, re-run on E1/E2)
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
# §6.8 row 1 (chemical variant): the branch delta is an INVARIANT scalar
# channel — bitwise pos-invariant — while the legacy pathway still moves
# the model outputs with the current geometry.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eid", CANDIDATES)
def test_branch_delta_bitwise_pos_invariant(eid: str, batch) -> None:
    model = _model(eid, alpha=GATE)
    moved = transform_forward_batch(
        batch, rotate=None, translate=torch.tensor([3.25, -1.5, 0.75]))
    h = model._compute_h_legacy(batch)
    with torch.no_grad():
        d_a = model.embedding(h, batch)
        d_b = model.embedding(h, moved)
    assert torch.equal(d_a, d_b)                  # no geometry is read
    # E1/E2 are pure chemical-topology candidates: at F2 every distance
    # feature is masked AND the branch recomputes nothing geometric, so
    # the whole model's invariant scalars are pos-invariant too (only
    # E3/E4 recompute geometry, §6.6.1).
    with torch.no_grad():
        out_a = model(batch)
        out_b = model(moved)
    assert torch.equal(out_a["z"], out_b["z"])
    assert torch.equal(out_a["q"], out_b["q"])


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
    """The branch is geometry-blind; the legacy equivariant pathway is
    not: ``pos`` must receive a finite (and nonzero) gradient once the
    vector head is in the loss."""
    pos = batch["pos"].detach().clone().requires_grad_(True)
    work = dict(batch)
    work["pos"] = pos
    model = _model(eid, alpha=GATE)
    out = model(work)
    (out["z"].sum() + out["q"].sum() + out["u"].sum()).backward()
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
    for key in ("n_nodes", "n_edges", "zero_neighbor", "zero_neighbor_frac"):
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
def test_independent_vs_mixed_batch(eid: str) -> None:
    from scripts.agent_ops.feature_gate.v3_fixtures import make_sn2_sample

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
    assert torch.allclose(out_a["z"], out_ab["z"][:1], rtol=1e-5, atol=1e-6)
    assert torch.allclose(out_b["z"], out_ab["z"][1:], rtol=1e-5, atol=1e-6)
    # per-key outputs: align rows by the key atom's label (index)
    key_a = batch_a["key_index"]
    key_ab = batch_ab["key_index"]
    in_a = batch_ab["graph_index"][key_ab] == 0
    assert torch.equal(key_ab[in_a], key_a)          # same keys, same order
    assert torch.allclose(out_a["u"], out_ab["u"][in_a], rtol=1e-5, atol=1e-6)
    in_b = batch_ab["graph_index"][key_ab] == 1
    key_b = batch_b["key_index"]
    assert torch.equal(key_ab[in_b] - n_a, key_b)
    assert torch.allclose(out_b["u"], out_ab["u"][in_b], rtol=1e-5, atol=1e-6)


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


# ---------------------------------------------------------------------------
# E2-specific: the frozen-stats contract at MODEL level
# ---------------------------------------------------------------------------
def test_e2_model_refuses_forward_without_stats(batch) -> None:
    torch.manual_seed(SEED)
    model = V541Model(embedding_id="E2", embedding_init_seed=INIT_SEED)
    model.alpha.data.fill_(GATE)
    assert model.embedding.degree_stats_state == "unset"
    with pytest.raises(Exception) as exc:
        model(batch)
    assert getattr(exc.value, "code", "") == "V541_DEGREE_STATS_MISSING"
    # after the runner loads the frozen stats the model runs
    model.embedding.set_degree_stats(MU_LN2, SHA_OK)
    with torch.no_grad():
        out = model(batch)
    assert set(out) == set(FIVE_OUTPUTS)


def test_e1_model_needs_no_degree_stats(batch) -> None:
    torch.manual_seed(SEED)
    model = V541Model(embedding_id="E1", embedding_init_seed=INIT_SEED)
    model.alpha.data.fill_(GATE)
    assert not hasattr(model.embedding, "set_degree_stats")
    with torch.no_grad():
        out = model(batch)
    assert set(out) == set(FIVE_OUTPUTS)
