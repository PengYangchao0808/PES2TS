"""V5.4.1 WP2 — E0 parity + interface certificate (plan §5.1–§5.5, ADR-0032).

Headline guarantees under test:

* ``V541Model()`` (E0) keeps the legacy submodule/state-dict naming
  (``backbone.*`` + ``process_head.*``), adds ZERO parameters and ZERO
  extra RNG, and its five outputs are BIT-for-bit identical to
  ``KeyAtomV53MPModel`` under the same seed (plan §5.3 items 1–5);
* strict-load certificate: explicit audit (missing/unexpected/shape/
  dtype all reviewed) then ``strict=True`` load of a legacy checkpoint;
* input legality parity: NaN/Z-range typed invalid + forward-view
  whitelist refusal are the SAME typed errors as the legacy model;
  supervision/metadata keys can never reach the branch or the outputs;
* E1–E5 (and E6) raise the typed NOT-IMPLEMENTED refusal — no fake
  candidate math exists in WP2;
* registry contract: E0 has no branch and can never be registered.

Fixture-level only (``not_admissible_until_gate_pass``); no training, no
dev/confirm access."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import (  # noqa: E402
    KeyAtomV3InvalidInputError,
    StarEGNNV3,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    MEASURED_PARAMS_V53MP_JOINT,
    V53MP_HEADS_VERSION,
    KeyAtomV53MPModel,
)
from pes2ts_core.pesn.models.v541 import (  # noqa: E402
    ALPHA_PARAM_KEY,
    LEGACY_STATE_PREFIXES,
    MEASURED_PARAMS_V541_E0_JOINT,
    V541_SCHEMA_VERSION,
    W_OUT_PARAM_KEY,
    V541EmbeddingNotImplementedError,
    V541EmbeddingUnknownError,
    V541InitError,
    V541LoadError,
    V541Model,
    EMBEDDING_SPECS,
    FROZEN_EMBEDDING_IDS,
    embedding_spec,
    implemented_embedding_ids,
    is_embedding_implemented,
    is_embedding_known,
    measured_params_v541_e0,
    register_embedding,
    registry_restore,
    registry_snapshot,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch  # noqa: E402

SEED = 17
FIVE_OUTPUTS = ("u", "z", "q", "endpoint_logit", "order_score")


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_batch(n_graphs=2))


@pytest.fixture(autouse=True)
def _clean_registry():
    """No test-only registration may leak into other tests/runs."""
    snap = registry_snapshot()
    yield
    registry_restore(snap)


def _candidate_state(eid: str) -> str:
    """Registry state of one frozen candidate: implemented | pending | broken.

    The registry answers DYNAMICALLY by importing the BUILT-IN WP3 module on
    demand, so this helper follows whatever has actually landed.  ``broken``
    means the landed module raises its TYPED import failure (never a raw
    exception, and never silently downgraded to ``pending``).
    """
    from pes2ts_core.pesn.models.v541 import embedding_registry as reg

    try:
        return "implemented" if reg.is_embedding_implemented(eid) else "pending"
    except Exception as exc:  # noqa: BLE001 - must be the typed import failure
        assert getattr(exc, "code", "") == "V541_EMBEDDING_IMPORT_FAILED", \
            "unexpected failure mode for {}: {!r}".format(eid, exc)
        return "broken"


# ---------------------------------------------------------------------------
# Registered surface (§5.1 module boundary / frozen ids)
# ---------------------------------------------------------------------------
def test_version_constants() -> None:
    assert V541_SCHEMA_VERSION == "pesn_key_atom_v541_model_v1"
    assert V541Model.v541_version == V541_SCHEMA_VERSION


def test_frozen_registry_ids_match_campaign_yaml() -> None:
    assert FROZEN_EMBEDDING_IDS == ("E0", "E1", "E2", "E3", "E4", "E5", "E6")
    assert set(EMBEDDING_SPECS) == set(FROZEN_EMBEDDING_IDS)
    # WP3 landed the real candidates: the registry answers DYNAMICALLY (it
    # lazily imports the built-in module), so E0–E4 are implemented and E6
    # (diagnostic-only) never is.  E5 is queried separately — the parallel
    # WP3 lane is landing dual_stream.py right now.
    assert is_embedding_implemented("E0") is True
    for eid in ("E1", "E2", "E3", "E4"):
        assert _candidate_state(eid) == "implemented", eid
    assert _candidate_state("E6") == "pending"
    assert _candidate_state("E5") in {"implemented", "pending", "broken"}
    spec = embedding_spec("E1")
    assert spec.family == "gin_e_chemical"
    # the STATIC flag is WP2-era metadata and stays frozen; the dynamic
    # is_embedding_implemented() above is the authoritative answer
    assert spec.implemented is False
    assert embedding_spec("E6").diagnostic_only is True


# ---------------------------------------------------------------------------
# E0: state-dict naming + zero new parameters (plan §5.3 item 1)
# ---------------------------------------------------------------------------
def test_e0_state_keys_identical_to_legacy() -> None:
    torch.manual_seed(SEED)
    legacy = KeyAtomV53MPModel()
    torch.manual_seed(SEED)
    e0 = V541Model()
    legacy_keys = set(legacy.state_dict())
    e0_keys = set(e0.state_dict())
    assert e0_keys == legacy_keys
    assert all(k.startswith(LEGACY_STATE_PREFIXES) for k in e0_keys)
    # no gate/branch keys leak into the E0 build
    assert not any(k == ALPHA_PARAM_KEY or k.startswith(W_OUT_PARAM_KEY)
                   for k in e0_keys)
    assert not any(k.startswith("embedding") for k in e0_keys)
    # submodule naming preserved (legacy loaders address these paths)
    assert isinstance(e0.backbone, StarEGNNV3)
    assert type(e0.process_head) is type(legacy.process_head)


def test_e0_zero_new_parameters() -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    assert e0.n_parameters == MEASURED_PARAMS_V53MP_JOINT
    assert e0.n_parameters == MEASURED_PARAMS_V541_E0_JOINT == 57_619
    assert measured_params_v541_e0() == 57_619
    assert e0.head_parameter_counts() == KeyAtomV53MPModel().head_parameter_counts()
    assert e0.has_branch is False and e0.is_e0 is True
    assert e0.alpha is None and e0.w_out is None and e0.embedding is None


def test_e0_zero_extra_rng_same_seed_bitwise_state() -> None:
    """E0 construction consumes exactly the legacy RNG sequence."""
    torch.manual_seed(SEED)
    KeyAtomV53MPModel()                        # consumes the legacy stream
    state_legacy = torch.random.get_rng_state()
    torch.manual_seed(SEED)
    V541Model()                                # E0 build
    state_e0 = torch.random.get_rng_state()
    assert torch.equal(state_legacy, state_e0)
    # and therefore identical weights under the same seed
    torch.manual_seed(SEED)
    legacy2 = KeyAtomV53MPModel()
    torch.manual_seed(SEED)
    e0 = V541Model()
    assert all(torch.equal(legacy2.state_dict()[k], e0.state_dict()[k])
               for k in legacy2.state_dict())


# ---------------------------------------------------------------------------
# E0: five-output bitwise parity + forward_move + q==order_score (§5.3)
# ---------------------------------------------------------------------------
def test_e0_five_outputs_bitwise_parity(batch) -> None:
    torch.manual_seed(SEED)
    legacy = KeyAtomV53MPModel()
    legacy.eval()
    torch.manual_seed(SEED)
    e0 = V541Model()
    e0.eval()
    with torch.no_grad():
        out_legacy = legacy(batch)
        out_e0 = e0(batch)
    assert set(out_legacy) == set(out_e0) == set(FIVE_OUTPUTS)
    for key in FIVE_OUTPUTS:
        assert torch.equal(out_legacy[key], out_e0[key]), key
    assert torch.equal(out_e0["order_score"], out_e0["q"])


def test_e0_forward_move_equals_backbone_direct(batch) -> None:
    torch.manual_seed(SEED)
    backbone = StarEGNNV3()
    e0 = V541Model(backbone)                   # shared by reference
    e0.eval()
    with torch.no_grad():
        out_bb = backbone(batch)
        out_move = e0.forward_move(batch)
    assert set(out_move) == {"u", "z"}
    assert torch.equal(out_move["u"], out_bb["u"])
    assert torch.equal(out_move["z"], out_bb["z"])
    # and equals the legacy forward_move view
    legacy = KeyAtomV53MPModel(backbone)
    with torch.no_grad():
        legacy_move = legacy.forward_move(batch)
    assert torch.equal(out_move["u"], legacy_move["u"])
    assert torch.equal(out_move["z"], legacy_move["z"])


def test_e0_eval_determinism_bitwise(batch) -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    e0.eval()
    with torch.no_grad():
        a = e0(batch)
        b = e0(batch)
    for key in FIVE_OUTPUTS:
        assert torch.equal(a[key], b[key]), key


def test_e0_non_key_atoms_have_no_direct_action(batch) -> None:
    """``u`` has exactly one row per key atom — non-key atoms carry no
    direct action output (MoveProcessContract §3.2 shape semantics)."""
    torch.manual_seed(SEED)
    e0 = V541Model()
    e0.eval()
    with torch.no_grad():
        out = e0(batch)
    n_keys = int(batch["key_index"].numel())
    n_atoms = int(batch["pos"].shape[0])
    assert n_keys < n_atoms                     # fixture has spectator atoms
    assert tuple(out["u"].shape) == (n_keys, 3)


# ---------------------------------------------------------------------------
# Input legality parity (§5.3 item 5/6; GeometryPolicyContract §S4.3)
# ---------------------------------------------------------------------------
def test_e0_nan_pos_typed_invalid(batch) -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    bad = dict(batch)
    bad["pos"] = torch.full_like(batch["pos"], float("nan"))
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        e0(bad)
    assert exc.value.code == "NONFINITE_INPUT"


def test_e0_z_range_typed_invalid(batch) -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    bad = dict(batch)
    bad["z"] = batch["z"].clone()
    bad["z"][0] = 200
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        e0(bad)
    assert exc.value.code == "Z_OUT_OF_RANGE"
    bad["z"] = batch["z"].clone()
    bad["z"][0] = 0                             # padding row is not input
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        e0(bad)
    assert exc.value.code == "Z_PADDING"


def test_e0_supervision_key_refused(batch) -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    leaked = dict(batch)
    leaked["y_teacher"] = torch.zeros(int(batch["key_index"].numel()), 3)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        e0(leaked)
    assert exc.value.code == "FORWARD_VIEW_LEAK"


def test_e0_metadata_intervention_does_not_change_output(batch) -> None:
    """``endpoint_names`` is whitelisted metadata: no numeric feature reads
    it, so intervening on it can never move an output."""
    torch.manual_seed(SEED)
    e0 = V541Model()
    e0.eval()
    alt = dict(batch)
    alt["endpoint_names"] = ("ALPHA_LABEL", "BETA_LABEL")
    with torch.no_grad():
        a = e0(batch)
        b = e0(alt)
    for key in FIVE_OUTPUTS:
        assert torch.equal(a[key], b[key]), key


def test_e0_nonfinite_degenerate_features_typed_invalid(batch) -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    bad = dict(batch)
    bad["degK"] = torch.full_like(batch["degK"], float("inf"))
    with pytest.raises(KeyAtomV3InvalidInputError) as exc:
        e0(bad)
    assert exc.value.code == "NONFINITE_INPUT"


# ---------------------------------------------------------------------------
# Registry dispatch (landed WP3 candidates build; pending ids REFUSE)
# ---------------------------------------------------------------------------
def test_main_matrix_candidates_are_landed() -> None:
    """E1–E4 resolve through the lazy BUILT-IN map; E6 never builds."""
    for eid in ("E1", "E2", "E3", "E4"):
        assert _candidate_state(eid) == "implemented", eid
    assert _candidate_state("E6") == "pending"


@pytest.mark.parametrize("eid", ["E1", "E2", "E3", "E4", "E5", "E6"])
def test_registry_dispatch_matches_the_landed_module(eid) -> None:
    """One dispatch table, three honest outcomes — no fake builder.

    * implemented → the REAL branch builds (no registry edit needed);
    * pending (module not landed) → typed ``V541_EMBEDDING_NOT_IMPLEMENTED``;
    * broken (import failed INSIDE a landed module) → typed
      ``V541_EMBEDDING_IMPORT_FAILED`` — never reported as ``pending``.
    """
    from pes2ts_core.pesn.models.v541 import NodeDeltaEmbedding, build_embedding

    state = _candidate_state(eid)
    if state == "broken":
        with pytest.raises(Exception) as exc:
            build_embedding(eid)
        assert exc.value.code == "V541_EMBEDDING_IMPORT_FAILED"
        return
    if state == "pending":
        assert eid in ("E5", "E6"), \
            "{} has a BUILT-IN module and must not be pending".format(eid)
        with pytest.raises(V541EmbeddingNotImplementedError) as exc:
            if eid == "E6":
                build_embedding(eid)          # diagnostic-only, never builds
            else:
                V541Model(embedding_id=eid, embedding_init_seed=1234)
        assert exc.value.code == "V541_EMBEDDING_NOT_IMPLEMENTED"
        assert "WP3" in exc.value.detail
        with pytest.raises(V541EmbeddingNotImplementedError):
            build_embedding(eid)
        return
    assert eid != "E6", "E6 is diagnostic-only and must never build"
    built = build_embedding(eid)
    assert isinstance(built, NodeDeltaEmbedding)
    assert is_embedding_implemented(eid) is True


def test_unknown_embedding_id_refused() -> None:
    with pytest.raises(V541EmbeddingUnknownError) as exc:
        V541Model(embedding_id="E9")
    assert exc.value.code == "V541_EMBEDDING_UNKNOWN"
    assert is_embedding_known("E9") is False
    with pytest.raises(V541EmbeddingUnknownError):
        embedding_spec("nope")


def test_branch_build_requires_explicit_init_seed() -> None:
    """Plan §5.4 step 3: candidate init must run on an independent seed."""
    from pes2ts_core.pesn.models.v541 import NodeDeltaEmbedding

    # a still-pending id refuses as NOT-IMPLEMENTED even before the seed
    # check — never a missing-seed confusion for an unbuildable candidate
    assert _candidate_state("E6") == "pending"
    with pytest.raises(V541EmbeddingNotImplementedError):
        V541Model(embedding_id="E6")

    class _Dummy(NodeDeltaEmbedding):
        def forward(self, h_legacy, batch):  # pragma: no cover
            return h_legacy

    snap = registry_snapshot()
    try:
        register_embedding("TEST_STUB", _Dummy)
        with pytest.raises(V541InitError) as exc:
            V541Model(embedding_id="TEST_STUB")
        assert exc.value.code == "V541_INIT_SEED_REQUIRED"
    finally:
        registry_restore(snap)


def test_registry_guards() -> None:
    from pes2ts_core.pesn.models.v541 import build_embedding

    assert build_embedding("E0") is None        # E0: no branch, ever
    with pytest.raises(Exception) as exc:
        register_embedding("E0", lambda: None)
    assert getattr(exc.value, "code", "") == "V541_EMBEDDING_E0_NO_BRANCH"
    with pytest.raises(Exception) as exc:
        register_embedding("E1", lambda: None)  # reserved: WP3 + replace=True
    assert getattr(exc.value, "code", "") == "V541_EMBEDDING_FROZEN"
    # E1 is a REAL landed implementation now; E6 (diagnostic-only) is not
    assert is_embedding_implemented("E1") is True
    assert is_embedding_implemented("E6") is False


# ---------------------------------------------------------------------------
# BUILT-IN lazy registration (no fake builders, no swallowed ImportError)
# ---------------------------------------------------------------------------
def test_builtin_lazy_registration_is_isolated() -> None:
    """Dropping the builder table still resolves the real E1/E2 modules."""
    from pes2ts_core.pesn.models.v541 import NodeDeltaEmbedding
    from pes2ts_core.pesn.models.v541 import embedding_registry as reg

    snap = reg.registry_snapshot()
    try:
        reg.registry_restore({})
        assert "E1" not in reg.registry_snapshot()
        assert reg.is_embedding_implemented("E1") is True   # lazy import
        assert "E1" in reg.registry_snapshot()              # now registered
        assert isinstance(reg.build_embedding("E2"), NodeDeltaEmbedding)
        # a pending id stays pending and E6 is a typed refusal
        if _candidate_state("E5") != "implemented":
            with pytest.raises(Exception) as exc:
                reg.build_embedding("E5")
            assert exc.value.code in {
                "V541_EMBEDDING_NOT_IMPLEMENTED",
                "V541_EMBEDDING_IMPORT_FAILED"}
        with pytest.raises(V541EmbeddingNotImplementedError):
            reg.build_embedding("E6")
    finally:
        reg.registry_restore(snap)


def test_builtin_e5_resolves_or_typed_refusal() -> None:
    """E5 resolves the moment ``dual_stream.py`` lands — no registry edit.

    Until a file exists it is the typed not-implemented refusal; a landed
    module must BUILD (it is never reported as pending).
    """
    from pes2ts_core.pesn.models.v541 import NodeDeltaEmbedding
    from pes2ts_core.pesn.models.v541 import embedding_registry as reg

    snap = reg.registry_snapshot()
    try:
        reg.registry_restore({})
        try:
            built = reg.build_embedding("E5")
        except V541EmbeddingNotImplementedError as exc:
            assert exc.code == "V541_EMBEDDING_NOT_IMPLEMENTED"
            assert "WP3" in exc.value.detail
            assert reg.is_embedding_implemented("E5") is False
            return
        assert isinstance(built, NodeDeltaEmbedding)
        assert reg.is_embedding_implemented("E5") is True
    finally:
        reg.registry_restore(snap)


def test_broken_builtin_module_is_not_reported_unimplemented(
        monkeypatch) -> None:
    """An ImportError from INSIDE a landed module must NOT become 'pending'."""
    import sys
    import types

    from pes2ts_core.pesn.models.v541 import embedding_registry as reg

    # a stand-in gin_e_chemical module WITHOUT the registry builder: the
    # static ``from ... import build_gin_e_chemical`` then raises ImportError
    broken = types.ModuleType(
        "pes2ts_core.pesn.models.v541.embeddings.gin_e_chemical")
    monkeypatch.setitem(
        sys.modules, "pes2ts_core.pesn.models.v541.embeddings.gin_e_chemical",
        broken)
    snap = reg.registry_snapshot()
    try:
        reg.registry_restore({k: v for k, v in snap.items() if k != "E1"})
        with pytest.raises(reg.V541InterfaceError) as exc:
            reg.ensure_embedding("E1")
        assert exc.value.code == "V541_EMBEDDING_IMPORT_FAILED"
        with pytest.raises(reg.V541InterfaceError):
            reg.build_embedding("E1")
    finally:
        reg.registry_restore(snap)


def test_registry_uses_no_dynamic_import_machinery() -> None:
    """The runtime-set boundary: no importlib / __import__ in the registry."""
    import ast

    from pes2ts_core.pesn.models.v541 import embedding_registry as reg

    tree = ast.parse(Path(reg.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in ("importlib", "subprocess")
                       for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in (
                "importlib", "subprocess")
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            assert name not in ("eval", "exec", "__import__")


# ---------------------------------------------------------------------------
# Strict-load certificate (plan §5.3 item 1)
# ---------------------------------------------------------------------------
def _legacy_payload(seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    legacy = KeyAtomV53MPModel()
    return {"schema": V53MP_HEADS_VERSION, "model": legacy.state_dict(),
            "update": 123, "best_val": 0.5}


def test_e0_strict_load_of_legacy_payload(batch) -> None:
    payload = _legacy_payload()
    torch.manual_seed(SEED + 1)                 # different fresh weights
    e0 = V541Model()
    report = e0.load_legacy_state_dict(payload, strict=True)
    audit = report["audit"]
    assert audit["ok"] is True
    assert audit["missing"] == [] and audit["unexpected"] == []
    assert audit["shape_mismatch"] == [] and audit["dtype_mismatch"] == []
    assert report["missing_keys"] == [] and report["unexpected_keys"] == []
    # loaded weights now bitwise equal the checkpoint
    state = payload["model"]
    assert all(torch.equal(e0.state_dict()[k], state[k]) for k in state)
    # and forward matches the original legacy model bitwise
    torch.manual_seed(SEED)
    reference = KeyAtomV53MPModel()
    reference.eval()
    e0.eval()
    with torch.no_grad():
        out_ref = reference(batch)
        out_e0 = e0(batch)
    for key in FIVE_OUTPUTS:
        assert torch.equal(out_ref[key], out_e0[key]), key


def test_e0_strict_load_accepts_raw_state_dict() -> None:
    payload = _legacy_payload()
    torch.manual_seed(SEED)
    e0 = V541Model()
    report = e0.load_legacy_state_dict(payload["model"], strict=True)
    assert report["audit"]["ok"] is True


@pytest.mark.parametrize("mutation,expected_code", [
    ("missing", "missing"),
    ("unexpected", "unexpected"),
    ("shape", "shape_mismatch"),
    ("dtype", "dtype_mismatch"),
])
def test_audit_detects_all_four_defect_classes(mutation, expected_code) -> None:
    payload = _legacy_payload()
    state = dict(payload["model"])
    key = "process_head.endpoint_head.fc.weight"
    if mutation == "missing":
        state.pop(key)
    elif mutation == "unexpected":
        state["backbone.not_a_real.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state[key] = state[key][:, :10]      # [1,64] -> [1,10]
    elif mutation == "dtype":
        state[key] = state[key].to(torch.float16)
    torch.manual_seed(SEED)
    e0 = V541Model()
    audit = e0.audit_legacy_state_dict(state)
    assert audit["ok"] is False
    assert audit[expected_code], audit
    with pytest.raises(V541LoadError) as exc:
        e0.load_legacy_state_dict(state, strict=True)
    assert exc.value.code == "V541_LOAD_AUDIT"
    if mutation in ("missing", "unexpected"):
        # non-strict tolerates key-level gaps (the WP3 partial-load path)
        report = e0.load_legacy_state_dict(state, strict=False)
        assert report["audit"]["ok"] is False
    else:
        # torch hard-fails on shape/dtype even with strict=False → typed error
        with pytest.raises(V541LoadError) as exc:
            e0.load_legacy_state_dict(state, strict=False)
        assert exc.value.code == "V541_LOAD_SHAPE"


def test_load_payload_type_refusals() -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    with pytest.raises(V541LoadError) as exc:
        e0.load_legacy_state_dict(["not", "a", "mapping"], strict=True)
    assert exc.value.code == "V541_LOAD_PAYLOAD"
    with pytest.raises(V541LoadError) as exc:
        e0.load_legacy_state_dict({"model": {"a": 1.0}}, strict=True)
    assert exc.value.code == "V541_LOAD_PAYLOAD"


def test_legacy_state_dict_view_and_prefixes() -> None:
    torch.manual_seed(SEED)
    e0 = V541Model()
    legacy_view = e0.legacy_state_dict()
    assert set(legacy_view) == set(e0.state_dict())   # E0: view == full
    assert all(k.startswith(LEGACY_STATE_PREFIXES) for k in legacy_view)


def test_v52_import_compat_regression() -> None:
    """Nothing renamed/removed: legacy classes still construct & forward."""
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV52Model
    torch.manual_seed(SEED)
    model = KeyAtomV52Model()
    assert isinstance(model.backbone, StarEGNNV3)
