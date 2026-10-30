"""V5.4.2 WP2 lane P2-0 — shared candidate scaffolding certificate
(ADR-0033 §2.3; config ``training_contract.init_policy``).

Headline guarantees under test:

* the FROZEN seed recipe — ``seed_init`` determinism, ``digest[:8]``
  ≡ ``hexdigest[:16]`` equivalence, the ``{run_seed, init_source_id}``
  template, the A1-C→A1 / A5-C→A5 overrides and the ``% 2147483647``
  range bound;
* SHA-pinned R3 loading — the REAL checkpoint in this environment loads
  (37 backbone keys) while a wrong expected SHA and a missing path are
  typed refusals;
* ``construction_scope`` restores the caller's torch RNG stream
  byte-identically on normal exit AND on exception, seeds exactly
  ``seed_init`` inside, and ``build_process_head`` is RNG-neutral while
  returning the frozen 2,178-parameter head;
* the strict R3 mapping audit — A1's mapping set (21 keys) applied to
  the REAL R3 payload, typed failures for missing target / unexpected
  source / shape / dtype / undeclared double consumption, a declared
  dual-write accepted exactly once;
* shared ``w_v`` identity + ``torch.equal``, exact-zero ``zero_init_``,
  the 70k candidate budget gate, and the backbone contract check
  (accepts a real ``A0LegacyBackbone``, rejects wrong-shape /
  finite-violating / detached fakes and a compiled tree).

Fixture-level only (``make_v3_gate_batch``); no training, no real-data
access, no dev/confirm touch."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    ReactionProcessHeadV53MP,
)
from pes2ts_core.pesn.models.v542 import (  # noqa: E402
    A0_ACTIVE_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    A0LegacyBackbone,
    BackboneOutput,
    DirectMoveDecoder,
    V542BackboneOutputError,
    V542InitError,
    V542InterfaceError,
)
from pes2ts_core.pesn.models.v542.candidates import (  # noqa: E402
    INIT_SEED_TEMPLATE,
    INIT_SOURCE_ID_OVERRIDES,
    R3_CHECKPOINT_PATH,
    R3_CHECKPOINT_SHA256,
    MappingAudit,
    V542CandidateBudgetError,
    V542InitMapError,
    V542InitSeedError,
    apply_r3_mapping,
    apply_r3_w_v,
    apply_shared_w_v,
    assert_candidate_budget,
    build_process_head,
    check_backbone_contract,
    construction_scope,
    finish_training_seed,
    load_r3_backbone_state,
    resolve_init_source_id,
    seed_init,
    zero_init_,
)
from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_gate_batch  # noqa: E402

SEED = 17
#: Frozen seed modulus of the recipe (``% 2147483647``).
SEED_MODULUS = 2_147_483_647

#: A1's frozen semantic mapping prefixes (config
#: ``init_policy.init_mapping.r3_semantic_mapping.A1``): R3 同名同形.
A1_MAPPING_PREFIXES = (
    "elem_emb", "node_proj", "z_head",
    "message_mlp.0", "message_mlp.1",
    "node_mlp.0", "node_mlp.1",
)


@pytest.fixture()
def r3_state():
    return load_r3_backbone_state()


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


def _a1_mapping(source_state) -> dict[str, str]:
    """A1's mapping set: identity over the seven frozen prefixes."""
    mapping: dict[str, str] = {}
    for key in sorted(source_state):
        if any(key == p or key.startswith(p + ".")
               for p in A1_MAPPING_PREFIXES):
            mapping[key] = key
    return mapping


def _consumed(source_state, mapping) -> list[str]:
    """Payload keys NOT consumed by ``mapping`` (must be declared)."""
    return [k for k in sorted(source_state) if k not in mapping]


# ---------------------------------------------------------------------------
# frozen seed recipe (config init_policy.candidate_fresh_init)
# ---------------------------------------------------------------------------
def test_seed_recipe_constants_are_frozen() -> None:
    assert INIT_SEED_TEMPLATE == "v542-init|{run_seed}|{init_source_id}"
    assert INIT_SOURCE_ID_OVERRIDES == {"A1-C": "A1", "A5-C": "A5"}
    assert R3_CHECKPOINT_PATH == (
        "outputs/pesn/30h_campaign/r3_runs/5372d9b12807c3f0/"
        "checkpoint_best.pt")
    assert R3_CHECKPOINT_SHA256 == (
        "a2e46b6ad65cc18996d6c1660c5ae24c723397ca22526e52fa34e430afd0417f")


def test_seed_recipe_determinism_hexdigest_equivalence_and_range() -> None:
    # digest[:8] big-endian ≡ hexdigest[:16] big-endian (frozen
    # seed_recipe_algorithm), evaluated for the whole run-seed ×
    # candidate-id grid.
    for run_seed in (17, 29, 43):
        for init_source_id in ("A0", "A1", "A2", "A3", "A4", "A5",
                               "A1-C", "A5-C"):
            message = f"v542-init|{run_seed}|{init_source_id}"
            digest = hashlib.sha256(message.encode("utf-8")).digest()
            from_digest = (int.from_bytes(digest[:8], "big")
                           % SEED_MODULUS)
            from_hex = (int(hashlib.sha256(message.encode("utf-8")).hexdigest()[:16], 16)
                        % SEED_MODULUS)
            assert from_digest == from_hex
            value = seed_init(run_seed, init_source_id)
            assert value == from_digest == from_hex
            assert isinstance(value, int)
            assert 0 <= value < SEED_MODULUS          # range bound
    # determinism + stream separation
    assert seed_init(17, "A1") == seed_init(17, "A1")
    assert seed_init(17, "A1") != seed_init(29, "A1")
    assert seed_init(17, "A1") != seed_init(17, "A5")


def test_resolve_init_source_id_overrides_and_default() -> None:
    assert resolve_init_source_id("A1-C") == "A1"
    assert resolve_init_source_id("A5-C") == "A5"
    for candidate_id in ("A0", "A1", "A2", "A3", "A4", "A5"):
        assert resolve_init_source_id(candidate_id) == candidate_id
    # the override is what puts A1-C on A1's stream
    assert seed_init(17, resolve_init_source_id("A1-C")) == seed_init(17, "A1")
    assert seed_init(17, resolve_init_source_id("A1-C")) != seed_init(17, "A1-C")
    assert seed_init(17, resolve_init_source_id("A5-C")) == seed_init(17, "A5")
    assert seed_init(17, resolve_init_source_id("A5-C")) != seed_init(17, "A5-C")


def test_seed_recipe_typed_refusals() -> None:
    with pytest.raises(V542InitSeedError) as exc:
        seed_init("17", "A1")            # type: ignore[arg-type]
    assert exc.value.code == "V542_INIT_SEED_TYPE"
    with pytest.raises(V542InitSeedError) as exc:
        seed_init(-1, "A1")
    assert exc.value.code == "V542_INIT_SEED_TYPE"
    with pytest.raises(V542InitSeedError) as exc:
        seed_init(True, "A1")            # bool is not a seed
    assert exc.value.code == "V542_INIT_SEED_TYPE"
    with pytest.raises(V542InitSeedError) as exc:
        seed_init(17, "")                # type: ignore[arg-type]
    assert exc.value.code == "V542_INIT_SEED_TYPE"
    with pytest.raises(V542InitSeedError) as exc:
        resolve_init_source_id("")       # type: ignore[arg-type]
    assert exc.value.code == "V542_INIT_SOURCE_ID"
    with pytest.raises(V542InitSeedError) as exc:
        resolve_init_source_id(None)     # type: ignore[arg-type]
    assert exc.value.code == "V542_INIT_SOURCE_ID"


# ---------------------------------------------------------------------------
# SHA-pinned R3 loading (config backbone_init; ADR-0033 §2.3)
# ---------------------------------------------------------------------------
def test_load_r3_backbone_state_real_checkpoint(r3_state) -> None:
    assert len(r3_state) == 37
    assert all(isinstance(v, torch.Tensor) for v in r3_state.values())
    assert r3_state["z_head.weight"].shape == (64, 96)
    assert r3_state["w_v.weight"].shape == (1, 48)
    assert r3_state["w_v.bias"].shape == (1,)
    assert r3_state["elem_emb.weight"].shape == (119, 16)
    assert r3_state["node_proj.weight"].shape == (48, 21)
    # the pinned digest really is the SHA-256 over the live file bytes
    digest = hashlib.sha256(Path(R3_CHECKPOINT_PATH).read_bytes()).hexdigest()
    assert digest == R3_CHECKPOINT_SHA256


def test_load_r3_backbone_state_refuses_wrong_sha() -> None:
    with pytest.raises(V542InitError) as exc:
        load_r3_backbone_state(expected_sha256="0" * 64)
    assert exc.value.code == "V542_R3_INIT_SHA256"
    # typed refusal, not a bare RuntimeError
    assert isinstance(exc.value, V542InterfaceError)
    assert "0" * 64 in exc.value.detail          # the rejected expectation


def test_load_r3_backbone_state_refuses_missing_path(tmp_path) -> None:
    missing = tmp_path / "never_written" / "checkpoint_best.pt"
    with pytest.raises(V542InitError) as exc:
        load_r3_backbone_state(path=str(missing))
    assert exc.value.code == "V542_R3_INIT_MISSING"
    assert isinstance(exc.value, V542InterfaceError)


# ---------------------------------------------------------------------------
# construction_scope: RNG isolation + exact seed_init inside
# ---------------------------------------------------------------------------
def test_construction_scope_restores_rng_on_normal_exit() -> None:
    torch.manual_seed(4242)
    before = torch.random.get_rng_state().clone()
    with construction_scope(17, "A1"):
        torch.randn(32)                    # consume the scoped stream
    assert torch.equal(before, torch.random.get_rng_state()), \
        "construction_scope perturbed the caller's RNG stream"


def test_construction_scope_restores_rng_on_exception() -> None:
    torch.manual_seed(777)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with construction_scope(29, "A5"):
            torch.randn(8)
            raise RuntimeError("boom")
    assert torch.equal(before, torch.random.get_rng_state()), \
        "construction_scope did not restore the RNG stream on exception"


def test_construction_scope_seeds_exactly_seed_init() -> None:
    with construction_scope(17, "A1") as seed:
        assert seed == seed_init(17, "A1")
        assert isinstance(seed, int)
        inside = torch.randn(8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_init(17, "A1"))
        reference = torch.randn(8)
    assert torch.equal(inside, reference), \
        "inside-scope stream is not seed_init(run_seed, init_source_id)"
    # and a module built inside sees the same stream
    with construction_scope(43, "A4"):
        built_in = torch.nn.Linear(5, 5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_init(43, "A4"))
        reference_linear = torch.nn.Linear(5, 5)
    assert torch.equal(built_in.weight, reference_linear.weight)
    assert torch.equal(built_in.bias, reference_linear.bias)


# ---------------------------------------------------------------------------
# deterministic process-head anchor (config process_head_anchor)
# ---------------------------------------------------------------------------
def test_build_process_head_is_rng_neutral_and_has_2178_params() -> None:
    torch.manual_seed(31337)
    before = torch.random.get_rng_state().clone()
    head = build_process_head(17)
    assert torch.equal(before, torch.random.get_rng_state()), \
        "build_process_head perturbed the caller's RNG stream"
    assert isinstance(head, ReactionProcessHeadV53MP)
    n_params = sum(int(p.numel()) for p in head.parameters())
    assert n_params == N_PROCESS_PARAMS_V542 == 2_178

    # anchor determinism: same seed → identical weights; other seed →
    # (essentially surely) different weights
    head_again = build_process_head(17)
    state_a = head.state_dict()
    state_b = head_again.state_dict()
    assert set(state_a) == set(state_b)
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    head_other = build_process_head(29)
    state_c = head_other.state_dict()
    assert any(not torch.equal(state_a[k], state_c[k]) for k in state_a)

    # typed refusal on an invalid anchor seed
    with pytest.raises(V542InitSeedError) as exc:
        build_process_head(-3)
    assert exc.value.code == "V542_INIT_SEED_TYPE"


def test_finish_training_seed_resets_global_rng() -> None:
    torch.manual_seed(999)
    torch.randn(16)                        # perturb the stream
    finish_training_seed(17)
    assert int(torch.initial_seed()) == 17
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        reference = torch.randn(6)
    assert torch.equal(torch.randn(6), reference)
    # typed refusal on a seed torch would silently remap
    with pytest.raises(V542InitSeedError) as exc:
        finish_training_seed(-1)
    assert exc.value.code == "V542_INIT_SEED_TYPE"


# ---------------------------------------------------------------------------
# strict R3 mapping audit (config init_policy.mapping / mapping_audit)
# ---------------------------------------------------------------------------
def test_apply_r3_mapping_a1_happy_path(r3_state) -> None:
    mapping = _a1_mapping(r3_state)
    assert len(mapping) == 21, mapping
    assert set(mapping) == {
        "elem_emb.weight",
        "node_proj.weight", "node_proj.bias",
        "z_head.weight", "z_head.bias",
        "message_mlp.0.0.weight", "message_mlp.0.0.bias",
        "message_mlp.0.2.weight", "message_mlp.0.2.bias",
        "message_mlp.1.0.weight", "message_mlp.1.0.bias",
        "message_mlp.1.2.weight", "message_mlp.1.2.bias",
        "node_mlp.0.0.weight", "node_mlp.0.0.bias",
        "node_mlp.0.2.weight", "node_mlp.0.2.bias",
        "node_mlp.1.0.weight", "node_mlp.1.0.bias",
        "node_mlp.1.2.weight", "node_mlp.1.2.bias",
    }
    consumed = _consumed(r3_state, mapping)
    assert len(consumed) == 16              # 21 mapped + 16 declared = 37

    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    pre = {k: v.clone() for k, v in module.state_dict().items()}
    audit = apply_r3_mapping(module, r3_state, mapping,
                             consumed_source_keys=consumed)
    assert isinstance(audit, MappingAudit)
    assert audit.missing == ()
    assert audit.unexpected == ()
    assert audit.shape_mismatch == ()
    assert audit.dtype_mismatch == ()
    assert audit.dual_write_aliases == ()
    assert len(audit.torch_equal_verified) == 21
    assert set(audit.torch_equal_verified) == set(mapping)
    assert audit.source_keys == tuple(sorted(r3_state))
    assert set(audit.target_keys) == set(mapping)
    assert str(audit.mapping_table_sha256).startswith("LOCAL_PLACEHOLDER:")
    # mapped slots now equal the real R3 tensors, byte-for-byte
    post = module.state_dict()
    for target in mapping:
        assert torch.equal(post[target], r3_state[target])
    # every UNmapped slot is untouched (fresh init preserved)
    for key in pre:
        if key not in mapping:
            assert torch.equal(post[key], pre[key]), key
    # the digest is a pure function of the effective table
    audit_again = apply_r3_mapping(module, r3_state, mapping,
                                   consumed_source_keys=consumed)
    assert audit_again.mapping_table_sha256 == audit.mapping_table_sha256


def test_apply_r3_mapping_missing_target(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    pre = {k: v.clone() for k, v in module.state_dict().items()}
    mapping = {"ghost.target.weight": "elem_emb.weight"}
    consumed = [k for k in sorted(r3_state) if k != "elem_emb.weight"]
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, r3_state, mapping,
                         consumed_source_keys=consumed)
    assert exc.value.code == "V542_INIT_MAP_MISSING"
    assert "ghost.target.weight" in exc.value.detail
    # nothing was copied before the refusal
    post = module.state_dict()
    assert all(torch.equal(post[k], pre[k]) for k in pre)


def test_apply_r3_mapping_missing_source(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    mapping = {"node_proj.bias": "ghost.source"}
    consumed = list(sorted(r3_state))
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, r3_state, mapping,
                         consumed_source_keys=consumed)
    assert exc.value.code == "V542_INIT_MAP_MISSING"
    assert "ghost.source" in exc.value.detail


def test_apply_r3_mapping_unexpected_source(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    drifted = dict(r3_state)
    drifted["ghost.weight"] = torch.zeros(4)   # payload key nobody consumes
    mapping = _a1_mapping(r3_state)
    consumed = _consumed(r3_state, mapping)
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, drifted, mapping,
                         consumed_source_keys=consumed)
    assert exc.value.code == "V542_INIT_MAP_UNEXPECTED"
    assert "ghost.weight" in exc.value.detail
    # an undeclared leftover payload key is ALSO unexpected (strict
    # full-accounting reading of mapping_audit)
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, r3_state, mapping,
                         consumed_source_keys=consumed[:-1])
    assert exc.value.code == "V542_INIT_MAP_UNEXPECTED"
    assert consumed[-1] in exc.value.detail


def test_apply_r3_mapping_shape_mismatch_no_copy(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    pre = {k: v.clone() for k, v in module.state_dict().items()}
    drifted = dict(r3_state)
    drifted["z_head.weight"] = r3_state["z_head.weight"][:10, :]
    mapping = _a1_mapping(r3_state)
    consumed = _consumed(r3_state, mapping)
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, drifted, mapping,
                         consumed_source_keys=consumed)
    assert exc.value.code == "V542_INIT_MAP_SHAPE"
    assert "z_head.weight" in exc.value.detail
    # no copy_ on mismatch: the whole module is untouched
    post = module.state_dict()
    assert all(torch.equal(post[k], pre[k]) for k in pre)


def test_apply_r3_mapping_dtype_mismatch_no_copy(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    pre = {k: v.clone() for k, v in module.state_dict().items()}
    drifted = dict(r3_state)
    drifted["elem_emb.weight"] = r3_state["elem_emb.weight"].to(torch.float16)
    mapping = _a1_mapping(r3_state)
    consumed = _consumed(r3_state, mapping)
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, drifted, mapping,
                         consumed_source_keys=consumed)
    assert exc.value.code == "V542_INIT_MAP_DTYPE"
    assert "elem_emb.weight" in exc.value.detail
    post = module.state_dict()
    assert all(torch.equal(post[k], pre[k]) for k in pre)


def test_apply_r3_mapping_declared_dual_write_accepted(r3_state) -> None:
    # the R3 bias tensors are independent random draws — prove the copy
    # really rewrote the second slot from the FIRST source
    assert not torch.equal(r3_state["node_mlp.0.0.bias"],
                           r3_state["node_mlp.1.0.bias"])
    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    mapping = {"node_mlp.0.0.bias": "node_mlp.0.0.bias"}
    dual = (("node_mlp.0.0.bias", "node_mlp.1.0.bias"),)
    consumed = [k for k in sorted(r3_state) if k != "node_mlp.0.0.bias"]
    audit = apply_r3_mapping(module, r3_state, mapping,
                             consumed_source_keys=consumed,
                             dual_write=dual)
    assert audit.missing == () and audit.unexpected == ()
    assert audit.shape_mismatch == () and audit.dtype_mismatch == ()
    assert audit.dual_write_aliases == (("node_mlp.0.0.bias",
                                         "node_mlp.1.0.bias"),)
    assert set(audit.torch_equal_verified) == {
        "node_mlp.0.0.bias", "node_mlp.1.0.bias"}
    post = module.state_dict()
    # the dual target equals the declared SOURCE (not its own payload)
    assert torch.equal(post["node_mlp.1.0.bias"],
                       r3_state["node_mlp.0.0.bias"])
    assert not torch.equal(post["node_mlp.1.0.bias"],
                           r3_state["node_mlp.1.0.bias"])
    assert torch.equal(post["node_mlp.0.0.bias"],
                       r3_state["node_mlp.0.0.bias"])
    # declaring the SAME pair twice is refused ("exactly once")
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, r3_state, mapping,
                         consumed_source_keys=consumed,
                         dual_write=(dual[0], dual[0]))
    assert exc.value.code == "V542_INIT_MAP_DOUBLE_CONSUME"


def test_apply_r3_mapping_undeclared_double_consumption(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    # two targets fed from ONE source with NO declared dual_write
    mapping = {"node_mlp.0.0.bias": "node_mlp.0.0.bias",
               "node_mlp.1.0.bias": "node_mlp.0.0.bias"}
    consumed = [k for k in sorted(r3_state) if k != "node_mlp.0.0.bias"]
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, r3_state, mapping,
                         consumed_source_keys=consumed)
    assert exc.value.code == "V542_INIT_MAP_DOUBLE_CONSUME"
    assert "node_mlp.0.0.bias" in exc.value.detail
    # the SAME construction passes once the dual_write is declared:
    # express it the frozen way — one mapping entry + one pair
    mapping_ok = {"node_mlp.0.0.bias": "node_mlp.0.0.bias"}
    dual = (("node_mlp.0.0.bias", "node_mlp.1.0.bias"),)
    audit = apply_r3_mapping(module, r3_state, mapping_ok,
                             consumed_source_keys=consumed,
                             dual_write=dual)
    assert audit.dual_write_aliases == dual


def test_apply_r3_mapping_rejects_target_conflict(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        module = StarEGNNV3()
    # one target (norm_h.1.weight) fed from TWO different sources: the
    # identity mapping entry AND the declared dual-write pair
    mapping = {"norm_h.0.weight": "norm_h.0.weight",
               "norm_h.1.weight": "norm_h.1.weight"}
    dual = (("norm_h.0.weight", "norm_h.1.weight"),)
    consumed = [k for k in sorted(r3_state)
                if k not in ("norm_h.0.weight", "norm_h.1.weight")]
    # consumption itself is legal here (norm_h.0: 1 mapping + 1 declared
    # pair = 2 ≤ 1+1; norm_h.1: 1 ≤ 1) — only the target clash fails
    with pytest.raises(V542InitMapError) as exc:
        apply_r3_mapping(module, r3_state, mapping,
                         consumed_source_keys=consumed,
                         dual_write=dual)
    assert exc.value.code == "V542_INIT_MAP_TARGET_CONFLICT"
    assert "norm_h.1.weight" in exc.value.detail


# ---------------------------------------------------------------------------
# shared w_v from R3 (config construction_order item 4)
# ---------------------------------------------------------------------------
def test_apply_r3_w_v_loads_fresh_decoder_from_sha_pinned_r3(r3_state) -> None:
    # default source=None: the SHA-pinned R3 gate runs on every call
    with torch.random.fork_rng(devices=[]):
        decoder = DirectMoveDecoder()
    random_before = decoder.w_v.weight.detach().clone()
    apply_r3_w_v(decoder)
    assert not torch.equal(decoder.w_v.weight, random_before)  # loaded
    assert torch.equal(decoder.w_v.weight, r3_state["w_v.weight"])
    assert torch.equal(decoder.w_v.bias, r3_state["w_v.bias"])


def test_apply_r3_w_v_uses_injected_source(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        decoder = DirectMoveDecoder()
    injected = dict(r3_state)
    injected["w_v.weight"] = torch.full_like(injected["w_v.weight"], 0.5)
    injected["w_v.bias"] = torch.full_like(injected["w_v.bias"], -0.25)
    apply_r3_w_v(decoder, injected)
    assert torch.equal(decoder.w_v.weight, injected["w_v.weight"])
    assert torch.equal(decoder.w_v.bias, injected["w_v.bias"])
    # and the pinned values are NOT used when a source is injected
    assert not torch.equal(decoder.w_v.weight, r3_state["w_v.weight"])


def test_apply_r3_w_v_typed_refusals(r3_state) -> None:
    # non-decoder seam
    with pytest.raises(V542InitError) as exc:
        apply_r3_w_v(torch.nn.Linear(48, 1))     # type: ignore[arg-type]
    assert exc.value.code == "V542_DECODER_TYPE"
    # non-mapping source
    with pytest.raises(V542InitError) as exc:
        apply_r3_w_v(DirectMoveDecoder(), source=[1, 2, 3])  # type: ignore[arg-type]
    assert exc.value.code == "V542_WV_SOURCE"
    # source without the w_v tensors
    with pytest.raises(V542InitError) as exc:
        apply_r3_w_v(DirectMoveDecoder(), source={"other": torch.zeros(2)})
    assert exc.value.code == "V542_WV_SOURCE"
    # shape drift is refused BEFORE any copy (no heuristics)
    bad_shape = dict(r3_state)
    bad_shape["w_v.weight"] = torch.zeros(1, 12)
    decoder = DirectMoveDecoder()
    with pytest.raises(V542InitError) as exc:
        apply_r3_w_v(decoder, bad_shape)
    assert exc.value.code == "V542_WV_MISMATCH"
    # the decoder was untouched by the refused load
    assert decoder.w_v.weight.shape == (1, 48)
    # dtype drift is refused the same way
    bad_dtype = dict(r3_state)
    bad_dtype["w_v.bias"] = r3_state["w_v.bias"].to(torch.float64)
    with pytest.raises(V542InitError) as exc:
        apply_r3_w_v(DirectMoveDecoder(), bad_dtype)
    assert exc.value.code == "V542_WV_MISMATCH"


def test_apply_shared_w_v_delegates_to_apply_r3_w_v(r3_state) -> None:
    # apply_shared_w_v first asserts backbone.w_v is move_decoder.w_v,
    # then delegates the load to apply_r3_w_v (same source semantics)
    with torch.random.fork_rng(devices=[]):
        decoder = DirectMoveDecoder()
        backbone = A0LegacyBackbone(decoder)
    injected = dict(r3_state)
    injected["w_v.weight"] = torch.zeros_like(injected["w_v.weight"])
    injected["w_v.bias"] = torch.zeros_like(injected["w_v.bias"])
    apply_shared_w_v(backbone, decoder, source=injected)
    assert torch.equal(decoder.w_v.weight, injected["w_v.weight"])
    assert torch.equal(backbone.w_v.weight, injected["w_v.weight"])


def test_apply_shared_w_v_identity_and_torch_equal(r3_state) -> None:
    with torch.random.fork_rng(devices=[]):
        decoder = DirectMoveDecoder()
        backbone = A0LegacyBackbone(decoder)
    apply_shared_w_v(backbone, decoder)
    assert backbone.w_v is decoder.w_v            # shared module object
    assert torch.equal(decoder.w_v.weight, r3_state["w_v.weight"])
    assert torch.equal(decoder.w_v.bias, r3_state["w_v.bias"])
    assert torch.equal(backbone.w_v.weight, r3_state["w_v.weight"])
    assert torch.equal(backbone.w_v.bias, r3_state["w_v.bias"])
    # state-dict dual keys point at the SAME storage
    sd = backbone.state_dict()
    assert sd["w_v.weight"].data_ptr() == decoder.w_v.weight.data_ptr()


def test_apply_shared_w_v_refuses_forked_wiring() -> None:
    with torch.random.fork_rng(devices=[]):
        decoder_a = DirectMoveDecoder()
        backbone_a = A0LegacyBackbone(decoder_a)
        decoder_b = DirectMoveDecoder()
        backbone_b = A0LegacyBackbone(decoder_b)
    with pytest.raises(V542InitError) as exc:
        apply_shared_w_v(backbone_b, decoder_a)   # backbone_b owns decoder_b
    assert exc.value.code == "V542_WV_NOT_SHARED"
    with pytest.raises(V542InitError) as exc:
        apply_shared_w_v(backbone_a, torch.nn.Linear(48, 1))  # type: ignore[arg-type]
    assert exc.value.code == "V542_DECODER_TYPE"


# ---------------------------------------------------------------------------
# per-candidate zero-init (config per_candidate_zero_init)
# ---------------------------------------------------------------------------
def test_zero_init_zeros_params_and_keeps_forward_semantics() -> None:
    torch.manual_seed(SEED)
    slot = torch.nn.Linear(12, 48)
    zero_init_(slot)
    assert torch.equal(slot.weight, torch.zeros(48, 12))
    assert torch.equal(slot.bias, torch.zeros(48))
    assert slot.weight.dtype == torch.float32 and slot.bias.dtype == torch.float32
    x = torch.randn(3, 12)
    out = slot(x)
    assert tuple(out.shape) == (3, 48)           # shape unaffected
    assert out.dtype == x.dtype == slot.weight.dtype  # dtype unaffected
    assert torch.equal(out, torch.zeros(3, 48))  # exact zeros propagate
    # typed refusal for a non-Linear slot
    with pytest.raises(V542InitError) as exc:
        zero_init_(torch.nn.LayerNorm(8))        # type: ignore[arg-type]
    assert exc.value.code == "V542_ZERO_INIT_TYPE"


# ---------------------------------------------------------------------------
# candidate budget (config param_budget.backbone_active_max = 70,000)
# ---------------------------------------------------------------------------
def test_assert_candidate_budget_measures_and_returns() -> None:
    class _AtCap(torch.nn.Module):
        @property
        def active_param_count(self) -> int:
            return 70_000

    assert assert_candidate_budget(_AtCap()) == 70_000
    assert assert_candidate_budget(_AtCap(), max_active=70_000) == 70_000
    # custom cap is honoured
    class _Over(torch.nn.Module):
        @property
        def active_param_count(self) -> int:
            return 70_001

    with pytest.raises(V542CandidateBudgetError) as exc:
        assert_candidate_budget(_Over())
    assert exc.value.code == "V542_CANDIDATE_BUDGET"
    assert "70001" in exc.value.detail
    assert assert_candidate_budget(_Over(), max_active=80_000) == 70_001
    # raw module fallback: sum(p.numel())
    linear = torch.nn.Linear(10, 10)             # 110 params
    assert assert_candidate_budget(linear) == 110
    with pytest.raises(V542CandidateBudgetError) as exc:
        assert_candidate_budget(linear, expected_active=111)
    assert exc.value.code == "V542_CANDIDATE_ACTIVE_MISMATCH"
    # exact expected match returns the measured count on a REAL backbone
    with torch.random.fork_rng(devices=[]):
        backbone = A0LegacyBackbone(DirectMoveDecoder())
    measured = assert_candidate_budget(backbone, expected_active=55_392)
    assert measured == A0_ACTIVE_PARAMS_V542 == 55_392
    with pytest.raises(V542CandidateBudgetError) as exc:
        assert_candidate_budget(backbone, expected_active=41_232)
    assert exc.value.code == "V542_CANDIDATE_ACTIVE_MISMATCH"


# ---------------------------------------------------------------------------
# backbone contract check (config backbone_contract.interface)
# ---------------------------------------------------------------------------
def test_check_backbone_contract_accepts_real_a0(batch) -> None:
    with torch.random.fork_rng(devices=[]):
        backbone = A0LegacyBackbone(DirectMoveDecoder())
    assert check_backbone_contract(backbone, batch) is None


def test_check_backbone_contract_rejects_wrong_shape(batch) -> None:
    class _WrongShape(torch.nn.Module):
        def forward(self, b):
            g = int(b["cv_target_delta"].shape[0])
            e = int(b["edge_index"].shape[1])
            return BackboneOutput(
                z=torch.zeros(g, 32, requires_grad=True),
                edge_msg=torch.zeros(e, 48, requires_grad=True))

    with pytest.raises(V542BackboneOutputError) as exc:
        check_backbone_contract(_WrongShape(), batch)
    assert exc.value.code == "V542_BACKBONE_Z_SHAPE"


def test_check_backbone_contract_rejects_non_finite(batch) -> None:
    class _NonFinite(torch.nn.Module):
        def forward(self, b):
            g = int(b["cv_target_delta"].shape[0])
            e = int(b["edge_index"].shape[1])
            return BackboneOutput(
                z=torch.full((g, 64), float("inf")),
                edge_msg=torch.zeros(e, 48))

    with pytest.raises(V542BackboneOutputError) as exc:
        check_backbone_contract(_NonFinite(), batch)
    assert exc.value.code == "V542_BACKBONE_NONFINITE"


def test_check_backbone_contract_rejects_detached_graph(batch) -> None:
    class _Detached(torch.nn.Module):
        def forward(self, b):
            g = int(b["cv_target_delta"].shape[0])
            e = int(b["edge_index"].shape[1])
            return BackboneOutput(z=torch.zeros(g, 64),
                                  edge_msg=torch.zeros(e, 48))

    with pytest.raises(V542BackboneOutputError) as exc:
        check_backbone_contract(_Detached(), batch)
    assert exc.value.code == "V542_BACKBONE_NO_GRAD"


def test_check_backbone_contract_rejects_compiled_tree(batch) -> None:
    class _CompiledTree(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._orig_mod = torch.nn.Linear(2, 2)

        def forward(self, b):
            raise AssertionError("a compiled tree must never run here")

    with pytest.raises(V542InterfaceError) as exc:
        check_backbone_contract(_CompiledTree(), batch)
    assert exc.value.code == "V542_TORCH_COMPILE_FORBIDDEN"


def test_candidates_common_never_calls_torch_compile() -> None:
    import inspect

    from pes2ts_core.pesn.models.v542.candidates import common

    assert "torch.compile(" not in inspect.getsource(common)


# ---------------------------------------------------------------------------
# the frozen construction order is a documented, testable contract
# ---------------------------------------------------------------------------
def test_construction_scope_docstring_freezes_order() -> None:
    import inspect

    from pes2ts_core.pesn.models.v542.candidates import common

    scope_doc = inspect.getdoc(common.construction_scope) or ""
    for step in ("1. fresh backbone modules",
                 "2. R3 tensors mapped",
                 "3. special parameters zero-initialized",
                 "4. ``DirectMoveDecoder``",
                 "5. ``ReactionProcessHeadV53MP``",
                 "6. AFTER leaving the scope"):
        assert step in scope_doc, step
    assert "finish_training_seed" in scope_doc
