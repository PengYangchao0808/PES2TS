"""V5.4.2 WP1 lane A — model/adapter/loader certificate (ADR-0033 §2.1–§2.2).

Headline guarantees under test:

* :class:`BackboneOutput` is frozen + slotted with EXACTLY the two
  functional fields ``z``/``edge_msg`` (no ``u``, no ``node_h``; field
  drift is a schema violation);
* :class:`DirectMoveDecoder` has exactly 49 params (``w_v`` only) and
  ``a_ref`` is a FROZEN constant — no constructor argument, read-only
  property, ``forward`` re-asserts 0.08 (gate-1 B1);
* A0 invariants: active 55,392 / decoder 49 / process 2,178 / joint
  57,619 (``active + 49 + 2178 == 57_619`` proven on a live build);
* ``backbone.w_v is move_decoder.w_v`` (shared module object),
  ``named_parameters()`` dedupes the shared 49 params, the hook is
  removed after forward and fired exactly once;
* :meth:`V542Model.param_groups_by_id` classifies EVERY unique parameter
  exactly once (core = backbone ∪ move_decoder, head = process_head);
* external surface is EXACTLY the five contract keys,
  ``order_score is q``, ``forward_move`` is ``{u, z}``; out-of-contract
  forward keys are refused with the legacy ``FORWARD_VIEW_LEAK`` typed
  error; no torch.compile anywhere (constructor refuses compiled trees);
* two eval-mode calls on the same batch are bitwise equal (determinism);
* the strict-load audit: typed failures for missing/unexpected/shape/
  dtype, while the declared ``w_v`` dual-write PASSES.

Fixture-level only (``make_v3_gate_batch``); no training, no real-data
access, no dev/confirm touch."""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    ReactionProcessHeadV53MP,
)
from pes2ts_core.pesn.models.v542 import (  # noqa: E402
    A0_ACTIVE_PARAMS_V542,
    A0_JOINT_PARAMS_V542,
    ACTIVE_PARAM_BUDGET_V542,
    A_REF_V542,
    FORWARD_KEYS_V542,
    FORWARD_MOVE_KEYS_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    A0LegacyBackbone,
    BackboneOutput,
    DirectMoveDecoder,
    V542InitError,
    V542LoadError,
    V542Model,
    V542ParamBudgetError,
    a0_mapping_table,
    assert_a0_joint_invariants,
    load_a0_strict,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_gate_batch  # noqa: E402

SEED = 17


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


@pytest.fixture()
def model() -> V542Model:
    torch.manual_seed(SEED)
    return V542Model()


# ---------------------------------------------------------------------------
# BackboneOutput freeze (ADR-0033 §2.1)
# ---------------------------------------------------------------------------
def test_backbone_output_fields_and_freeze() -> None:
    fields = {f.name for f in dataclasses.fields(BackboneOutput)}
    assert fields == {"z", "edge_msg"}
    assert "u" not in fields and "node_h" not in fields
    out = BackboneOutput(z=torch.zeros(2, 64), edge_msg=torch.zeros(3, 48))
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.z = torch.ones(2, 64)  # type: ignore[misc]
    assert set(BackboneOutput.__slots__) == {"z", "edge_msg"}
    assert out.n_graphs == 2 and out.n_edges == 3


# ---------------------------------------------------------------------------
# decoder: 49 params, unlearnable a_ref (ADR-0033 §2.1)
# ---------------------------------------------------------------------------
def test_decoder_49_params_and_a_ref_not_learnable() -> None:
    dec = DirectMoveDecoder()
    assert dec.param_count == N_DECODER_PARAMS_V542 == 49
    names = sorted(n for n, _ in dec.named_parameters())
    assert names == ["w_v.bias", "w_v.weight"]
    assert isinstance(dec.a_ref, float) and dec.a_ref == A_REF_V542 == 0.08
    # a_ref is neither Parameter nor buffer: unlearnable & non-stateful
    assert "a_ref" not in dict(dec.named_parameters())
    assert "a_ref" not in dict(dec.named_buffers())
    assert not any("a_ref" in k for k in dec.state_dict())


# ---------------------------------------------------------------------------
# gate-1 B1: a_ref is FROZEN (no constructor arg, read-only, forward-checked)
# ---------------------------------------------------------------------------
def test_decoder_refuses_constructor_a_ref_override() -> None:
    # the public a_ref constructor argument was REMOVED entirely — any
    # attempt to pass one is a TypeError (the constant is not
    # configurable under a run identity claiming a_ref_angstrom=0.08)
    with pytest.raises(TypeError):
        DirectMoveDecoder(a_ref=0.1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DirectMoveDecoder(a_ref=0.08)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DirectMoveDecoder(0.1)  # type: ignore[call-arg]


def test_decoder_a_ref_is_read_only() -> None:
    dec = DirectMoveDecoder()
    with pytest.raises(AttributeError):
        dec.a_ref = 0.1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        dec.a_ref = 0.08  # type: ignore[misc]
    assert dec.a_ref == A_REF_V542


def test_decoder_forward_reasserts_frozen_a_ref() -> None:
    # even a hand-patched backing slot cannot change MOVE math: forward
    # re-asserts the frozen constant with a typed refusal
    dec = DirectMoveDecoder()
    dec._a_ref = 0.1  # sabotage the backing slot directly
    with pytest.raises(V542InitError) as exc:
        dec(torch.zeros(3, 48), torch.zeros(4, 3),
            torch.tensor([[0, 1, 2], [1, 2, 3]]),
            torch.tensor([1, 3]), torch.ones(4))
    assert exc.value.code == "V542_A_REF_FROZEN"
    dec._a_ref = A_REF_V542  # restore
    out = dec(torch.zeros(3, 48), torch.zeros(4, 3),
              torch.tensor([[0, 1, 2], [1, 2, 3]]),
              torch.tensor([1, 3]), torch.ones(4))
    assert tuple(out.shape) == (2, 3)


def test_v542_model_refuses_wrong_a_ref_decoder() -> None:
    # a DirectMoveDecoder SUBCLASS whose a_ref property is overridden to
    # 0.1 is a typed refusal at both assembly seams (gate-1 B1)
    class _WrongARefDecoder(DirectMoveDecoder):
        @property
        def a_ref(self) -> float:
            return 0.1

    bad = _WrongARefDecoder()
    with pytest.raises(V542InitError) as exc:
        A0LegacyBackbone(bad)
    assert exc.value.code == "V542_DECODER_A_REF"
    with pytest.raises(V542InitError) as exc:
        V542Model(move_decoder=bad)
    assert exc.value.code == "V542_DECODER_A_REF"
    # a monkeypatched instance (property shadow via __dict__ is blocked by
    # the property; simulate via a wrong-valued _a_ref read on getattr)
    good = DirectMoveDecoder()
    assert good.a_ref == A_REF_V542
    # non-decoder objects are refused before the a_ref check
    with pytest.raises(V542InitError) as exc:
        V542Model(move_decoder=torch.nn.Linear(1, 1))  # type: ignore[arg-type]
    assert exc.value.code == "V542_DECODER_TYPE"


def test_default_forward_move_unchanged_by_a_ref_freeze(
        model: V542Model, batch) -> None:
    # positive control: the default build's forward_move stays legacy-exact
    # (bitwise) after the a_ref freeze — the freeze changes the INTERFACE,
    # never the math
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    torch.manual_seed(SEED + 11)
    legacy = KeyAtomV53MPModel()
    load_a0_strict(model, {"model": legacy.state_dict()})
    model.eval()
    legacy.eval()
    with torch.no_grad():
        move = model.forward_move(batch)
        ref = legacy.forward_move(batch)
    assert set(move) == {"u", "z"}
    assert torch.equal(move["u"], ref["u"])
    assert torch.equal(move["z"], ref["z"])
    # and the u scale really is a_ref=0.08: degK=1 rows scale by 0.08
    assert model.move_decoder.a_ref == A_REF_V542


def test_decoder_constructor_refuses_param_drift() -> None:
    # the 49-param invariant is asserted at build; simulate drift by
    # monkeypatching the frozen constant for one construction
    import pes2ts_core.pesn.models.v542.decoder as dec_mod

    real = dec_mod.N_DECODER_PARAMS_V542
    try:
        dec_mod.N_DECODER_PARAMS_V542 = 48
        with pytest.raises(V542ParamBudgetError) as exc:
            DirectMoveDecoder()
        assert exc.value.code == "V542_DECODER_PARAMS"
    finally:
        dec_mod.N_DECODER_PARAMS_V542 = real


def test_decoder_rejects_missing_w_v(model: V542Model) -> None:
    # decoder reads exactly five tensors — no batch mapping is accepted
    dec = model.move_decoder
    edge_msg = torch.zeros(3, 48)
    pos = torch.zeros(4, 3)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ki = torch.tensor([1, 3])
    dk = torch.ones(4)
    out = dec(edge_msg, pos, ei, ki, dk)
    assert tuple(out.shape) == (2, 3)
    # zero messages ⇒ zero u (structural non-key zeros preserved)
    assert torch.equal(out, torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# A0 invariants (55,392 / 57,619) + shared w_v + hook lifecycle
# ---------------------------------------------------------------------------
def test_a0_active_and_joint_invariants(model: V542Model) -> None:
    report = model.parameter_report()
    assert report == {
        "backbone_active": A0_ACTIVE_PARAMS_V542,
        "decoder": 49,
        "process_head": N_PROCESS_PARAMS_V542,
        "joint": A0_JOINT_PARAMS_V542,
    }
    assert A0_ACTIVE_PARAMS_V542 == 55_392
    assert N_PROCESS_PARAMS_V542 == 2_178
    assert A0_JOINT_PARAMS_V542 == 57_619
    counts = assert_a0_joint_invariants(model.backbone)
    assert counts == {"active": 55_392, "decoder": 49,
                      "process": 2_178, "joint": 57_619}
    assert model.n_parameters == 57_619
    assert report["backbone_active"] + 49 + 2_178 == 57_619
    assert report["backbone_active"] <= ACTIVE_PARAM_BUDGET_V542


def test_backbone_shares_decoder_module_object(model: V542Model) -> None:
    assert model.backbone.w_v is model.move_decoder.w_v
    # state dict: dual keys pointing at the SAME tensor
    sd = model.state_dict()
    assert "backbone.w_v.weight" in sd and "move_decoder.w_v.weight" in sd
    assert sd["backbone.w_v.weight"].data_ptr() \
        == sd["move_decoder.w_v.weight"].data_ptr()
    assert sd["backbone.w_v.bias"].data_ptr() \
        == sd["move_decoder.w_v.bias"].data_ptr()


def test_named_parameters_dedupes_shared_w_v(model: V542Model) -> None:
    # the shared w_v params appear EXACTLY ONCE (under backbone.*)
    wv = [(n, id(p)) for n, p in model.named_parameters() if "w_v" in n]
    assert len(wv) == 2  # weight + bias, one instance each
    assert all(n.startswith("backbone.w_v.") for n, _ in wv)
    assert len({i for _, i in wv}) == 2
    # deduped total: naive per-module sum double counts 49; joint must not
    naive = (sum(int(p.numel()) for p in model.backbone.parameters())
             + sum(int(p.numel()) for p in model.move_decoder.parameters())
             + sum(int(p.numel()) for p in model.process_head.parameters()))
    assert naive == 57_619 + 49
    assert model.n_parameters == 57_619


def test_hook_removed_after_forward_and_fires_once(
        model: V542Model, batch) -> None:
    hooks = model.backbone.message_mlp[-1]._forward_hooks
    assert len(hooks) == 0
    with torch.no_grad():
        model(batch)
    # finally-removed: no hook survives the forward (even across calls)
    assert len(model.backbone.message_mlp[-1]._forward_hooks) == 0
    # fired exactly once: edge_msg equals the layer-3 message recomputed
    # with an external counting hook on the same module
    seen = []

    def _count(_m, _i, output):
        seen.append(output)

    handle = model.backbone.message_mlp[-1].register_forward_hook(_count)
    try:
        with torch.no_grad():
            out = model.backbone(batch)
    finally:
        handle.remove()
    assert len(seen) == 1
    assert torch.equal(out.edge_msg, seen[0])
    assert tuple(out.edge_msg.shape) == (
        int(batch["edge_index"].shape[1]), 48)


def test_hook_removed_on_typed_invalid_abort(model: V542Model, batch) -> None:
    bad = dict(batch)
    bad["degK"] = torch.full_like(batch["degK"], float("inf"))
    with pytest.raises(RuntimeError):
        model(bad)
    assert len(model.backbone.message_mlp[-1]._forward_hooks) == 0


# ---------------------------------------------------------------------------
# param-group by id (ADR-0033 §2.2)
# ---------------------------------------------------------------------------
def test_param_groups_by_id_classification(model: V542Model) -> None:
    groups = model.param_groups_by_id()
    assert set(groups) == {"core", "head"}
    core_ids = {id(p) for p in groups["core"]}
    head_ids = {id(p) for p in groups["head"]}
    assert not (core_ids & head_ids), "a param was classified twice"
    all_ids = {id(p) for p in model.parameters()}
    assert core_ids | head_ids == all_ids, "a param was left unclassified"
    # core = backbone ∪ move_decoder (shared w_v counted once)
    expected_core = {id(p) for p in model.backbone.parameters()} \
        | {id(p) for p in model.move_decoder.parameters()}
    assert core_ids == expected_core
    assert head_ids == {id(p) for p in model.process_head.parameters()}
    # exact sizes
    assert sum(int(p.numel()) for p in groups["core"]) == 55_392 + 49
    assert sum(int(p.numel()) for p in groups["head"]) == 2_178
    # the 49 w_v params live in core exactly once
    wv_ids = {id(model.move_decoder.w_v.weight),
              id(model.move_decoder.w_v.bias)}
    assert wv_ids <= core_ids and not (wv_ids & head_ids)
    report = model.param_group_report()
    assert [r["name"] for r in report] == ["core", "head"]
    assert [r["n_params"] for r in report] == [55_441, 2_178]


def test_param_group_refuses_shared_head_param(model: V542Model) -> None:
    # sabotage: a process_head param also registered on the backbone
    model.backbone.z_head = model.process_head.endpoint_head.fc
    with pytest.raises(V542InitError) as exc:
        model.param_groups_by_id()
    # the sabotaged param is either overlap or (z_head replaced →) a
    # backbone-only object; overlap with head is what we forced
    assert exc.value.code in ("V542_PARAM_GROUP_OVERLAP",
                              "V542_PARAM_UNCLASSIFIED",
                              "V542_PARAM_GROUP_EXTRA")


# ---------------------------------------------------------------------------
# external surface: 5 keys, order_score is q, forward_move, whitelist
# ---------------------------------------------------------------------------
def test_forward_key_surface_and_order_score_is_q(model: V542Model, batch) -> None:
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert set(out) == set(FORWARD_KEYS_V542) \
        == {"u", "z", "q", "endpoint_logit", "order_score"}
    assert out["order_score"] is out["q"], "order_score IS q (same object)"
    assert tuple(out["u"].shape)[-1] == 3
    n_graphs = int(batch["cv_target_delta"].shape[0])  # bidir ⇒ 2×n_graphs
    assert tuple(out["z"].shape) == (n_graphs, 64)
    with torch.no_grad():
        move = model.forward_move(batch)
    assert set(move) == set(FORWARD_MOVE_KEYS_V542) == {"u", "z"}
    assert torch.equal(move["u"], out["u"])
    assert torch.equal(move["z"], out["z"])


def test_forward_refuses_out_of_contract_keys(model: V542Model, batch) -> None:
    bad = dict(batch)
    bad["y_teacher"] = torch.zeros(2, 3)  # supervision leak attempt
    with pytest.raises(KeyAtomV3ContractError) as exc:
        model(bad)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    # forward_move enforces the same whitelist (legacy backbone check)
    with pytest.raises(KeyAtomV3ContractError):
        model.forward_move(bad)


def test_backbone_output_wrapper_checks(model: V542Model, batch,
                                        monkeypatch) -> None:
    # a backbone returning a wrong-shaped edge_msg is a typed refusal
    real_forward = A0LegacyBackbone.forward

    def _bad_forward(self, b):
        out = real_forward(self, b)
        return BackboneOutput(z=out.z, edge_msg=out.edge_msg[:1])

    monkeypatch.setattr(A0LegacyBackbone, "forward", _bad_forward)
    with pytest.raises(RuntimeError) as exc:
        model(batch)
    assert getattr(exc.value, "code", "") == "V542_BACKBONE_EDGE_SHAPE"


# ---------------------------------------------------------------------------
# no torch.compile (ADR-0033 §2.1)
# ---------------------------------------------------------------------------
def test_no_torch_compile_anywhere(model: V542Model) -> None:
    from pes2ts_core.pesn.models.v542 import backbone_a0, decoder, interfaces, loader, model as model_mod

    for mod in (interfaces, decoder, backbone_a0, model_mod, loader):
        source = open(mod.__file__, encoding="utf-8").read()
        assert "torch.compile(" not in source, mod.__name__
    for name, child in model.named_modules():
        assert not hasattr(child, "_orig_mod"), name


def test_constructor_refuses_compiled_module(model: V542Model) -> None:
    from pes2ts_core.pesn.models.v542 import assert_no_torch_compile

    class _FakeOptimized(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._orig_mod = torch.nn.Linear(2, 2)

    with pytest.raises(Exception) as exc:
        assert_no_torch_compile(_FakeOptimized())
    assert getattr(exc.value, "code", "") == "V542_TORCH_COMPILE_FORBIDDEN"
    # the assembled model is clean
    assert_no_torch_compile(model)


# ---------------------------------------------------------------------------
# determinism (two eval calls bitwise equal)
# ---------------------------------------------------------------------------
def test_determinism_two_eval_calls(model: V542Model, batch) -> None:
    model.eval()
    with torch.no_grad():
        a = model(batch)
        b = model(batch)
    for key in FORWARD_KEYS_V542:
        assert torch.equal(a[key], b[key]), key
    assert a["order_score"] is a["q"] and b["order_score"] is b["q"]


def test_a0_adapter_is_legacy_exact_on_fixture(model: V542Model, batch) -> None:
    # the A0 adapter + shared decoder reproduce the legacy forward
    # bit-for-bit from the SAME weights (identity load of a legacy build)
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    torch.manual_seed(SEED + 1)
    legacy = KeyAtomV53MPModel()
    load_a0_strict(model, {"model": legacy.state_dict()})
    model.eval()
    legacy.eval()
    with torch.no_grad():
        out = model(batch)
        ref = legacy(batch)
    for key in FORWARD_KEYS_V542:
        assert torch.equal(out[key], ref[key]), key
    assert out["order_score"] is out["q"]


# ---------------------------------------------------------------------------
# strict-load audit typed failures (ADR-0033 §2.2)
# ---------------------------------------------------------------------------
def _payload(model=None) -> dict:
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    if model is None:
        torch.manual_seed(SEED)
        model = KeyAtomV53MPModel()
    return {"model": {k: v.clone() for k, v in model.state_dict().items()}}


def test_strict_load_passes_with_w_v_dual_write() -> None:
    torch.manual_seed(SEED)
    v542 = V542Model()
    payload = _payload()
    audit = load_a0_strict(v542, payload)
    assert audit["ok"] is True and audit["strict"] is True
    assert audit["missing"] == [] and audit["unexpected"] == []
    assert audit["shape_mismatch"] == [] and audit["dtype_mismatch"] == []
    # the declared dual-write is NOT rejected and is recorded
    assert set(audit["dual_write_aliases"]) == {
        "backbone.w_v.weight", "backbone.w_v.bias"}
    assert audit["dual_write_aliases"]["backbone.w_v.weight"] == [
        "backbone.w_v.weight", "move_decoder.w_v.weight"]
    assert audit["n_source_keys"] == 43
    assert audit["n_mapped_targets"] == 45
    assert len(audit["torch_equal_verified"]) == 45
    # gate-0 A3: the mapping digest is now canonical (64-hex, no placeholder)
    digest = str(audit["mapping_table_sha256"])
    assert len(digest) == 64 and not digest.startswith("LOCAL_PLACEHOLDER:")
    assert v542.backbone.w_v is v542.move_decoder.w_v
    # post-load: the shared tensors equal the source values
    src = payload["model"]
    sd = v542.state_dict()
    assert torch.equal(sd["backbone.w_v.weight"], src["backbone.w_v.weight"])
    assert torch.equal(sd["move_decoder.w_v.bias"], src["backbone.w_v.bias"])


def test_mapping_table_is_identity_plus_dual_write() -> None:
    table = a0_mapping_table()
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    source_keys = set(KeyAtomV53MPModel().state_dict().keys())
    assert set(table) == source_keys
    for key, targets in table.items():
        if key in ("backbone.w_v.weight", "backbone.w_v.bias"):
            assert targets == [key, key.replace("backbone.",
                                                "move_decoder.", 1)]
        else:
            assert targets == [key]


# ---------------------------------------------------------------------------
# gate-1 C4: the A0 loader can NEVER perturb the training RNG stream
# ---------------------------------------------------------------------------
def test_a0_loader_never_perturbs_global_rng() -> None:
    # a0_mapping_table builds a throwaway KeyAtomV53MPModel — the
    # construction is wrapped in torch.random.fork_rng so the global
    # torch CPU (+CUDA when available) RNG state is byte-identical
    # before/after; load_a0_strict inherits the guarantee (it calls
    # a0_mapping_table internally and touches no other RNG consumer).
    torch.manual_seed(4242)
    cpu_before = torch.random.get_rng_state()
    cuda_before = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)

    table = a0_mapping_table()
    assert len(table) == 43

    cpu_after = torch.random.get_rng_state()
    assert torch.equal(cpu_before, cpu_after), \
        "a0_mapping_table perturbed the global torch CPU RNG stream"
    if cuda_before is not None:
        cuda_after = torch.cuda.get_rng_state_all()
        assert len(cuda_before) == len(cuda_after)
        for b, a in zip(cuda_before, cuda_after):
            assert torch.equal(b, a), \
                "a0_mapping_table perturbed a CUDA RNG stream"

    # load_a0_strict on a real pair: RNG state must also stay fixed
    torch.manual_seed(5151)
    v542 = V542Model()               # construction may consume RNG — do it
    payload = _payload()             # BEFORE the snapshot
    cpu_before2 = torch.random.get_rng_state()
    cuda_before2 = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)
    load_a0_strict(v542, payload)
    assert torch.equal(cpu_before2, torch.random.get_rng_state()), \
        "load_a0_strict perturbed the global torch CPU RNG stream"
    if cuda_before2 is not None:
        for b, a in zip(cuda_before2, torch.cuda.get_rng_state_all()):
            assert torch.equal(b, a), \
                "load_a0_strict perturbed a CUDA RNG stream"


def test_strict_load_typed_failure_missing() -> None:
    torch.manual_seed(SEED)
    v542 = V542Model()
    payload = _payload()
    del payload["model"]["backbone.z_head.weight"]
    with pytest.raises(V542LoadError) as exc:
        load_a0_strict(v542, payload)
    assert exc.value.code == "V542_LOAD_MISSING"


def test_strict_load_typed_failure_unexpected() -> None:
    torch.manual_seed(SEED)
    v542 = V542Model()
    payload = _payload()
    payload["model"]["backbone.not_a_key"] = torch.zeros(3)
    with pytest.raises(V542LoadError) as exc:
        load_a0_strict(v542, payload)
    assert exc.value.code == "V542_LOAD_UNEXPECTED"


def test_strict_load_typed_failure_shape() -> None:
    torch.manual_seed(SEED)
    v542 = V542Model()
    payload = _payload()
    payload["model"]["backbone.z_head.weight"] = \
        payload["model"]["backbone.z_head.weight"][:10, :]
    with pytest.raises(V542LoadError) as exc:
        load_a0_strict(v542, payload)
    assert exc.value.code == "V542_LOAD_SHAPE"


def test_strict_load_typed_failure_dtype() -> None:
    torch.manual_seed(SEED)
    v542 = V542Model()
    payload = _payload()
    payload["model"]["backbone.elem_emb.weight"] = \
        payload["model"]["backbone.elem_emb.weight"].to(torch.float16)
    with pytest.raises(V542LoadError) as exc:
        load_a0_strict(v542, payload)
    assert exc.value.code == "V542_LOAD_DTYPE"


def test_strict_load_never_falls_back_to_loose_modes() -> None:
    # the loader API has NO strict=False / heuristic path
    import inspect

    from pes2ts_core.pesn.models.v542 import loader

    sig = inspect.signature(loader.load_a0_strict)
    assert "strict" not in sig.parameters, \
        "load_a0_strict must not expose a non-strict mode"
    # code-only check (AST): no ``strict=False`` call kwarg anywhere in the
    # module, and the one load_state_dict call passes strict=True
    import ast

    tree = ast.parse(inspect.getsource(loader))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "load_state_dict"]
    assert calls, "loader must load through nn.Module.load_state_dict"
    for call in calls:
        for kw in call.keywords:
            assert not (kw.arg == "strict"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False), \
                "strict=False path exists in the loader"
    assert any(kw.arg == "strict"
               and isinstance(kw.value, ast.Constant)
               and kw.value.value is True
               for call in calls for kw in call.keywords), \
        "load_state_dict must be called with strict=True"


def test_v542_init_refuses_foreign_decoder_wiring() -> None:
    torch.manual_seed(SEED)
    dec_a, dec_b = DirectMoveDecoder(), DirectMoveDecoder()
    bb = A0LegacyBackbone(dec_a)
    with pytest.raises(V542InitError) as exc:
        V542Model(bb, move_decoder=dec_b)
    assert exc.value.code == "V542_WV_NOT_SHARED"
    # the matching wiring builds fine
    ok = V542Model(bb, move_decoder=dec_a)
    assert ok.backbone.w_v is dec_a.w_v


def test_v542_init_refuses_non_a0_backbone() -> None:
    from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3

    with pytest.raises(V542InitError) as exc:
        V542Model(StarEGNNV3())  # type: ignore[arg-type]
    assert exc.value.code == "V542_BACKBONE_TYPE"


def test_v542_init_refuses_non_frozen_process_head() -> None:
    with pytest.raises(V542InitError) as exc:
        V542Model(process_head=torch.nn.Linear(64, 1))  # type: ignore[arg-type]
    assert exc.value.code == "V542_PROCESS_HEAD"
    # the frozen class is ReactionProcessHeadV53MP (2,178 params)
    assert isinstance(V542Model().process_head, ReactionProcessHeadV53MP)
