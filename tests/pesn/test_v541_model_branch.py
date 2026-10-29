"""V5.4.1 WP2 — residual-branch mechanics with a TEST-ONLY dummy embedding
(plan §5.2/§5.4; ADR-0032).

The real candidates E1–E5 are NOT implemented in WP2 (typed refusal, see
``test_v541_model.py``).  These tests exercise the FROZEN branch
mechanics — controlled ``node_proj`` forward-hook injection, the unified
residual ``h = h_legacy + alpha·W·f``, independent fork_rng init, hook
cleanup, detached diagnostics — through a test-registered, E(3)-safe
dummy embedding (per-node MLP over ``h_legacy`` only, so it cannot break
symmetry by construction).

Guarantees under test:

* ``alpha == 0`` ⇒ five outputs bitwise equal the legacy model sharing
  the same backbone + process head; ``forward_move`` too;
* the precomputed ``h_legacy`` is bitwise what the backbone's own
  ``node_proj`` emits (the mirror is exact — no core-loop copy drifts);
* ``alpha`` gets a NONZERO gradient at cold start while ``W``/``f`` are
  zero (expected first step); after ``alpha != 0`` the whole branch
  receives nonzero gradients;
* with ``alpha != 0`` the outputs change (the gate is real) yet
  translation invariance / rotation+reflection equivariance / within-
  graph permutation equivariance hold at the normative PD-2 tolerances;
* branch init under ``torch.random.fork_rng`` never perturbs the legacy
  RNG sequence; same seed ⇒ identical branch weights;
* the hook is ALWAYS removed (normal return and typed-invalid abort),
  branch outputs are legality-checked (shape/dtype/finiteness/edge
  refusal), ``node_vector`` is recorded but never consumed, and
  supervision keys are refused before any branch math."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel  # noqa: E402
from pes2ts_core.pesn.models.v541 import (  # noqa: E402
    DIAGNOSTIC_KEYS_V541,
    HIDDEN_V541,
    NodeDeltaEmbedding,
    V541LegalityError,
    V541Model,
    register_embedding,
    registry_restore,
    registry_snapshot,
)
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    check_permutation_equivariance,
    check_rotation_equivariance,
    check_translation_invariance,
    _random_rotation,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch  # noqa: E402

SEED = 5
INIT_SEED = 7
GATE = 0.25
TEST_ID = "TEST_INV_MLP"
FIVE_OUTPUTS = ("u", "z", "q", "endpoint_logit", "order_score")


# ---------------------------------------------------------------------------
# test-only dummy embedding (E(3)-safe: per-node MLP over h_legacy only)
# ---------------------------------------------------------------------------
class InvariantMLPEmbedding(NodeDeltaEmbedding):
    """Dummy ``f_phi``: per-node MLP on ``h_legacy`` (invariant ⇒ safe)."""

    def __init__(self, hidden: int = HIDDEN_V541) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden))

    def forward(self, h_legacy, batch):        # batch deliberately unread
        return self.net(h_legacy)


def _build_dummy() -> NodeDeltaEmbedding:
    return InvariantMLPEmbedding()


@pytest.fixture(autouse=True)
def _registry():
    snap = registry_snapshot()
    register_embedding(TEST_ID, _build_dummy)
    yield
    registry_restore(snap)


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_batch(n_graphs=2))


def _branch_model(alpha: float = 0.0, *, seed: int = SEED,
                  init_seed: int = INIT_SEED) -> V541Model:
    torch.manual_seed(seed)
    model = V541Model(embedding_id=TEST_ID, embedding_init_seed=init_seed)
    if alpha:
        model.alpha.data.fill_(float(alpha))
    return model.eval()


def _legacy_view(model: V541Model) -> KeyAtomV53MPModel:
    """A legacy model sharing the SAME backbone + process head by reference."""
    view = KeyAtomV53MPModel(model.backbone)
    view.process_head = model.process_head
    return view.eval()


def _n_hooks(model: V541Model) -> int:
    return len(model.backbone.node_proj._forward_hooks)


# ---------------------------------------------------------------------------
# construction: W non-zero (orthogonal), alpha zero, layout, params
# ---------------------------------------------------------------------------
def test_branch_parameters_contract() -> None:
    model = _branch_model()
    assert model.has_branch is True and model.is_e0 is False
    # alpha: single scalar, exactly zero-initialized
    assert model.alpha.shape == ()
    assert float(model.alpha.detach()) == 0.0
    # W: 48→48, no bias, NON-zero and orthogonal init (plan §5.2)
    assert model.w_out.weight.shape == (HIDDEN_V541, HIDDEN_V541)
    assert model.w_out.bias is None
    assert float(model.w_out.weight.detach().abs().sum()) > 0.0
    gram = model.w_out.weight @ model.w_out.weight.T
    assert torch.allclose(gram, torch.eye(HIDDEN_V541), atol=1e-5)
    # state dict: legacy keys unchanged + exactly the three new anchors
    legacy_keys = set(_legacy_view(model).state_dict())
    v541_keys = set(model.state_dict())
    assert legacy_keys <= v541_keys
    new_keys = v541_keys - legacy_keys
    assert new_keys == {"alpha", "w_out.weight",
                        *[k for k in v541_keys if k.startswith("embedding.")]}
    n_branch = sum(p.numel() for p in model.embedding.parameters())
    assert model.n_parameters == 57619 + n_branch + HIDDEN_V541 ** 2 + 1
    counts = model.head_parameter_counts()
    assert counts["embedding"] == n_branch
    assert counts["w_out"] == HIDDEN_V541 ** 2
    assert counts["alpha"] == 1
    assert counts["backbone"] + counts["process_head"] + sum(
        counts[k] for k in ("embedding", "w_out", "alpha")
    ) == model.n_parameters


# ---------------------------------------------------------------------------
# alpha == 0: bitwise legacy parity (five outputs + forward_move)
# ---------------------------------------------------------------------------
def test_alpha_zero_five_outputs_bitwise(batch) -> None:
    model = _branch_model()
    legacy = _legacy_view(model)
    with torch.no_grad():
        out_v = model(batch)
        out_l = legacy(batch)
    assert set(out_v) == set(out_l) == set(FIVE_OUTPUTS)
    for key in FIVE_OUTPUTS:
        assert torch.equal(out_v[key], out_l[key]), key
    assert torch.equal(out_v["order_score"], out_v["q"])


def test_alpha_zero_forward_move_bitwise(batch) -> None:
    model = _branch_model()
    legacy = _legacy_view(model)
    with torch.no_grad():
        move_v = model.forward_move(batch)
        move_l = legacy.forward_move(batch)
        move_bb = model.backbone(batch)
    assert set(move_v) == {"u", "z"}
    for src in (move_l, move_bb):
        assert torch.equal(move_v["u"], src["u"])
        assert torch.equal(move_v["z"], src["z"])


def test_h_legacy_mirror_is_bitwise_exact(batch) -> None:
    """The mirrored node-init line reproduces the backbone's own
    ``node_proj`` output bit-for-bit (no core-loop duplication drift)."""
    model = _branch_model()
    captured: dict[str, torch.Tensor] = {}

    def _capture(module, inputs, output):
        captured["h"] = output.clone()

    handle = model.backbone.node_proj.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()
    with torch.no_grad():
        h_pre = model._compute_h_legacy(batch)
    assert torch.equal(h_pre, captured["h"])


def test_eval_determinism_branch(batch) -> None:
    model = _branch_model(alpha=GATE)
    with torch.no_grad():
        a = model(batch)
        b = model(batch)
    for key in FIVE_OUTPUTS:
        assert torch.equal(a[key], b[key]), key


# ---------------------------------------------------------------------------
# alpha != 0: the gate is real, E(3) + permutation still hold
# ---------------------------------------------------------------------------
def test_nonzero_gate_changes_outputs(batch) -> None:
    model = _branch_model(alpha=GATE)
    legacy = _legacy_view(model)
    with torch.no_grad():
        out_v = model(batch)
        out_l = legacy(batch)
    assert not torch.equal(out_v["u"], out_l["u"])
    assert not torch.equal(out_v["z"], out_l["z"])
    assert torch.equal(out_v["order_score"], out_v["q"])   # shared scalar kept
    # non-key atoms still have no direct action rows
    assert tuple(out_v["u"].shape) == (int(batch["key_index"].numel()), 3)


def test_nonzero_gate_translation_invariance(batch) -> None:
    model = _branch_model(alpha=GATE)
    report = check_translation_invariance(model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report
    # invariants (z/q/endpoint) are bitwise stable under a pure translation:
    # h_legacy never reads absolute coordinates
    shifted = dict(batch)
    shifted["pos"] = batch["pos"] + torch.tensor([1.7, -0.3, 2.2])
    with torch.no_grad():
        a = model(batch)
        b = model(shifted)
    for key in ("z", "q", "endpoint_logit", "order_score"):
        assert torch.equal(a[key], b[key]), key


def test_nonzero_gate_rotation_equivariance(batch) -> None:
    model = _branch_model(alpha=GATE)
    report = check_rotation_equivariance(model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report


def test_nonzero_gate_reflection_equivariance(batch) -> None:
    """Improper orthogonal transform (det = -1): polar ``u`` obeys
    ``u(R x) = R u(x)`` at the normative tolerance."""
    model = _branch_model(alpha=GATE)
    R = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    rot = dict(batch)
    rot["pos"] = batch["pos"] @ R.T
    with torch.no_grad():
        a = model(batch)
        b = model(rot)
    errors = ((a["u"] @ R.T) - b["u"]).norm(dim=-1)
    scale = float(a["u"].abs().max())
    assert float(errors.max()) <= 1e-6 + 1e-5 * scale, errors.max()


def test_nonzero_gate_permutation_equivariance(batch) -> None:
    model = _branch_model(alpha=GATE)
    report = check_permutation_equivariance(
        model, batch, preset="30h_v1.1_normative")
    assert report["passed"], report


def test_rotation_invariant_outputs_stable(batch) -> None:
    """z/q/endpoint are invariants: a rotation moves them only at float noise."""
    model = _branch_model(alpha=GATE)
    R = _random_rotation(3)
    rot = dict(batch)
    rot["pos"] = batch["pos"] @ R.T
    with torch.no_grad():
        a = model(batch)
        b = model(rot)
    for key in ("z", "q", "endpoint_logit"):
        delta = float((a[key] - b[key]).abs().max())
        assert delta <= 1e-6 + 1e-5 * float(a[key].abs().max()), (key, delta)


# ---------------------------------------------------------------------------
# gradients (plan §5.2: alpha learns first; W/f start at zero by math)
# ---------------------------------------------------------------------------
def _loss(model: V541Model, batch) -> torch.Tensor:
    out = model(batch)
    return out["u"].sum() + out["z"].sum() + out["q"].sum()


def test_cold_start_gradients_alpha_only(batch) -> None:
    model = _branch_model()
    loss = _loss(model, batch)
    loss.backward()
    assert model.alpha.grad is not None
    assert abs(float(model.alpha.grad)) > 0.0            # alpha learns at step 0
    assert model.w_out.weight.grad is not None
    assert torch.equal(model.w_out.weight.grad,
                       torch.zeros_like(model.w_out.weight.grad))
    for p in model.embedding.parameters():
        assert p.grad is not None                        # in graph…
        assert torch.equal(p.grad, torch.zeros_like(p.grad))  # …× alpha = 0


def test_after_alpha_nonzero_branch_grads_flow(batch) -> None:
    model = _branch_model()
    _loss(model, batch).backward()                       # cold step
    model.zero_grad(set_to_none=True)
    model.alpha.data.fill_(0.1)
    _loss(model, batch).backward()
    assert float(model.w_out.weight.grad.abs().sum()) > 0.0
    branch_grown = any(
        float(p.grad.abs().sum()) > 0.0 for p in model.embedding.parameters())
    assert branch_grown
    assert abs(float(model.alpha.grad)) > 0.0


def test_legacy_grads_still_reach_backbone_and_heads(batch) -> None:
    model = _branch_model(alpha=GATE)
    _loss(model, batch).backward()
    assert model.backbone.z_head.weight.grad is not None
    assert model.process_head.progress_head.fc1.weight.grad is not None
    assert model.backbone.node_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# independent RNG (plan §5.4 step 3): fork_rng keeps the legacy sequence
# ---------------------------------------------------------------------------
def test_branch_init_preserves_legacy_rng_sequence() -> None:
    torch.manual_seed(123)
    legacy = KeyAtomV53MPModel()
    state_legacy = torch.random.get_rng_state()
    torch.manual_seed(123)
    branch = V541Model(embedding_id=TEST_ID, embedding_init_seed=999)
    state_branch = torch.random.get_rng_state()
    assert torch.equal(state_legacy, state_branch)
    # ⇒ the legacy weights themselves are bitwise identical too
    assert all(torch.equal(legacy.state_dict()[k], branch.state_dict()[k])
               for k in legacy.state_dict())


def test_branch_init_reproducible_per_seed() -> None:
    a = _branch_model(init_seed=11)
    b = _branch_model(init_seed=11)
    c = _branch_model(init_seed=12)
    assert torch.equal(a.w_out.weight, b.w_out.weight)
    assert all(torch.equal(pa, pb) for pa, pb in
               zip(a.embedding.parameters(), b.embedding.parameters()))
    same_seed_differs = not torch.equal(a.w_out.weight, c.w_out.weight)
    branch_params_differ = any(
        not torch.equal(pa, pc) for pa, pc in
        zip(a.embedding.parameters(), c.embedding.parameters()))
    assert same_seed_differs or branch_params_differ


def test_branch_build_consumes_no_global_rng() -> None:
    """Construction draws RNG only inside the fork: the global stream ends
    exactly where a pure legacy build would leave it (plan §5.4 step 3)."""
    torch.manual_seed(321)
    V541Model(embedding_id=TEST_ID, embedding_init_seed=42)
    branch_state = torch.random.get_rng_state()
    torch.manual_seed(321)
    KeyAtomV53MPModel()
    legacy_state = torch.random.get_rng_state()
    assert torch.equal(branch_state, legacy_state)


# ---------------------------------------------------------------------------
# hook lifecycle + forward whitelist + branch legality
# ---------------------------------------------------------------------------
def test_hook_removed_after_normal_forward(batch) -> None:
    model = _branch_model(alpha=GATE)
    assert _n_hooks(model) == 0
    with torch.no_grad():
        model(batch)
        model.forward_move(batch)
    assert _n_hooks(model) == 0


def test_hook_removed_after_typed_invalid_abort(batch) -> None:
    model = _branch_model(alpha=GATE)
    bad = dict(batch)
    bad["pos"] = torch.full_like(batch["pos"], float("nan"))
    with pytest.raises(Exception) as exc:
        model(bad)
    assert getattr(exc.value, "code", "") == "NONFINITE_INPUT"
    assert _n_hooks(model) == 0
    # and the model still works afterwards (no leaked state)
    with torch.no_grad():
        out = model(batch)
    assert tuple(out["u"].shape) == (int(batch["key_index"].numel()), 3)


def test_branch_z_range_typed_invalid_before_elem_emb(batch) -> None:
    model = _branch_model(alpha=GATE)
    bad = dict(batch)
    bad["z"] = batch["z"].clone()
    bad["z"][0] = 300
    with pytest.raises(Exception) as exc:
        model(bad)
    assert getattr(exc.value, "code", "") == "Z_OUT_OF_RANGE"
    assert _n_hooks(model) == 0


def test_branch_supervision_key_refused_before_any_branch_math(batch) -> None:
    model = _branch_model(alpha=GATE)
    leaked = dict(batch)
    leaked["y_teacher"] = torch.zeros(int(batch["key_index"].numel()), 3)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        model(leaked)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    assert _n_hooks(model) == 0
    # diagnostics were never populated by the refused call
    assert model.diagnostics == {}


def test_branch_metadata_intervention_does_not_change_output(batch) -> None:
    model = _branch_model(alpha=GATE)
    alt = dict(batch)
    alt["endpoint_names"] = ("X_LABEL", "Y_LABEL")
    with torch.no_grad():
        a = model(batch)
        b = model(alt)
    for key in FIVE_OUTPUTS:
        assert torch.equal(a[key], b[key]), key


def _model_with(embedding_cls, *, alpha: float = GATE) -> V541Model:
    test_id = f"TEST_{embedding_cls.__name__.upper()}"
    register_embedding(test_id, embedding_cls, replace=True)
    torch.manual_seed(SEED)
    model = V541Model(embedding_id=test_id, embedding_init_seed=INIT_SEED)
    if alpha:
        model.alpha.data.fill_(float(alpha))
    return model.eval()


def test_branch_output_shape_refusal(batch) -> None:
    class TooNarrow(NodeDeltaEmbedding):
        def forward(self, h_legacy, batch):
            return h_legacy[:, :8]

    model = _model_with(TooNarrow)
    with pytest.raises(V541LegalityError) as exc:
        model(batch)
    assert exc.value.code == "V541_DELTA_SHAPE"
    assert _n_hooks(model) == 0


def test_branch_output_dtype_refusal(batch) -> None:
    class HalfDelta(NodeDeltaEmbedding):
        def forward(self, h_legacy, batch):
            return torch.tanh(h_legacy).to(torch.float16)

    model = _model_with(HalfDelta)
    with pytest.raises(V541LegalityError) as exc:
        model(batch)
    assert exc.value.code == "V541_DELTA_DTYPE"


def test_branch_output_nonfinite_refusal(batch) -> None:
    class NonFinite(NodeDeltaEmbedding):
        def forward(self, h_legacy, batch):
            out = h_legacy.clone()
            out[0, 0] = float("nan")
            return out

    model = _model_with(NonFinite)
    with pytest.raises(V541LegalityError) as exc:
        model(batch)
    assert exc.value.code == "V541_DELTA_NONFINITE"
    assert _n_hooks(model) == 0


def test_edge_scalar_output_forbidden(batch) -> None:
    """Candidates may never supply edge scalars — the core keeps the
    legacy 22-channel edge input in legacy order (plan §5.2)."""

    class WithEdge(NodeDeltaEmbedding):
        def forward(self, h_legacy, batch):
            return {
                "delta_node": h_legacy,
                "edge_scalar": torch.zeros(int(batch["edge_index"].shape[1]), 22),
            }

    model = _model_with(WithEdge)
    with pytest.raises(V541LegalityError) as exc:
        model(batch)
    assert exc.value.code == "V541_EDGE_SCALAR_FORBIDDEN"
    assert _n_hooks(model) == 0


def test_node_vector_recorded_but_never_consumed(batch) -> None:
    class WithVector(NodeDeltaEmbedding):
        def forward(self, h_legacy, batch):
            return {
                "delta_node": h_legacy,
                "node_vector": torch.ones(int(h_legacy.shape[0]), 3),
            }

    model = _model_with(WithVector, alpha=0.0)
    with torch.no_grad():
        out = model(batch)
    legacy = _legacy_view(model)
    with torch.no_grad():
        ref = legacy(batch)
    for key in FIVE_OUTPUTS:                     # alpha=0 ⇒ vector is inert
        assert torch.equal(out[key], ref[key]), key
    diag = model.diagnostics
    assert diag["node_vector_consumed"] == 0.0   # core never consumes it
    assert diag["rms_node_vector"] > 0.0         # …but it is DIAGNOSED
    assert set(diag) == set(DIAGNOSTIC_KEYS_V541)


def test_diagnostics_are_detached_floats(batch) -> None:
    model = _branch_model(alpha=GATE)
    with torch.no_grad():
        model(batch)
    diag = model.diagnostics
    assert set(diag) == set(DIAGNOSTIC_KEYS_V541)
    for key, value in diag.items():
        assert isinstance(value, float), (key, type(value))
        assert value == value and value not in (float("inf"),)  # finite
    assert diag["alpha"] == GATE
    assert diag["n_nodes"] == float(batch["pos"].shape[0])
    assert diag["n_edges"] == float(batch["edge_index"].shape[1])
    assert diag["n_keys"] == float(batch["key_index"].numel())
    assert diag["rms_h_legacy"] > 0.0
    assert diag["rms_alpha_W_f"] > 0.0            # alpha·W·f is live
    # returning a snapshot never aliases live state
    model.alpha.data.fill_(0.0)
    assert diag["alpha"] == GATE                  # old snapshot unchanged


def test_diagnostics_can_be_disabled_for_training(batch) -> None:
    model = _branch_model(alpha=GATE)
    model.diagnostics_enabled = False
    with torch.no_grad():
        model(batch)
    assert model.diagnostics == {}


def test_e0_diagnostics_empty(batch) -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    e0.eval()
    with torch.no_grad():
        e0(batch)
    assert e0.diagnostics == {}
