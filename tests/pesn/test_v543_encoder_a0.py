"""V5.4.3 WP1 — ``CoarseEncoder`` (A0 adapter) certificate.

Plan §4.4 (Encoder 只运行一次), §5.8 (A0 迁移与零残差 parity: three hooks,
``finally`` removal, exactly-once firing, no detach), §7.1 S0-A (阻断门),
§11.1/§11.2 (``coarse_encoder.py``: strict load + 三个 hook +
``u0/q0_raw/e0``; ``CoarsePacket``), §11.3 (禁止 strict=False /
torch.compile / 修改 V542 文件), §11.5 (``test_v543_hook_fires_once``).

Headline guarantees under test:

* the frozen V543 constants re-express the V5.4.2 semantics (64/48/0.08/
  55,392/49/2,178/57,619) and the plan §5.8 tolerances (CPU 0.0 bitwise,
  GPU 1e-6);
* :class:`CoarsePacket` is frozen + slotted with exactly six fields, no
  ``pos``/supervision/context field; shape/dtype/finiteness drift is a
  typed refusal;
* the encoder is strict-loadable by the FROZEN V5.4.2 loader: 43 source
  keys → 45 mapped targets (the declared ``w_v`` dual-write), no
  ``strict=False`` path exists, and the load never perturbs the global
  torch RNG (``fork_rng`` discipline);
* ``forward`` reproduces the V5.4.2 reference (``V542Model``) BITWISE on
  the SAME weights: ``u0 == u``, ``z == z``, ``q0 == q``,
  ``endpoint0 == endpoint_logit``, ``edge_msg`` equal, and
  ``tanh(q0_raw) == q``;
* all three hooks (``message_mlp[-1]`` / ``norm_h[-1]`` /
  ``progress_head.fc2``) fire EXACTLY once and are removed afterwards —
  including on a typed-invalid abort;
* ``SUPERVISION_KEYS_V3`` members are refused before any compute; other
  out-of-contract keys keep the legacy ``FORWARD_VIEW_LEAK`` refusal;
* captured tensors keep their autograd graph (no detach);
* no ``torch.compile`` anywhere in this package.

Fixture-level only (``make_v3_gate_batch``) — CPU-fast, no real data, no
training.  The real-checkpoint S0-A leg lives in
``tests/pesn/test_v543_encoder_parity_realdata.py`` (``realdata`` marker)."""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import Z_DIM_V3  # noqa: E402
from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    KeyAtomV53MPModel,
    ReactionProcessHeadV53MP,
)
from pes2ts_core.pesn.models.v542 import (  # noqa: E402
    A0_ACTIVE_PARAMS_V542,
    A0_JOINT_PARAMS_V542,
    A_REF_V542,
    EDGE_MSG_DIM_V542,
    HIDDEN_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    A0LegacyBackbone,
    DirectMoveDecoder,
    V542LoadError,
    V542Model,
    load_a0_strict,
)
from pes2ts_core.pesn.models.v543 import (  # noqa: E402
    A_REF_V543,
    COARSE_PACKET_KEYS,
    COARSE_PACKET_SCHEMA,
    DECODER_CONTEXT_SCHEMA_V543,
    EDGE_MSG_DIM_V543,
    ENCODER_ACTIVE_PARAMS_V543,
    ENCODER_FORWARD_CALLS_PER_VIEW,
    ENCODER_JOINT_PARAMS_V543,
    ENCODER_Z_DIM_V543,
    HIDDEN_V543,
    HOOK_FIRES_PER_FORWARD,
    HOOK_LABELS,
    KEY_S_DIM_V543,
    MAPPING_MODE_FROZEN,
    MAPPING_MODE_IDENTITY,
    N_DECODER_PARAMS_V543,
    N_PROCESS_PARAMS_V543,
    PARITY_ATOL_CPU,
    PARITY_ATOL_GPU,
    PAYLOAD_LAYOUTS_V543,
    PAYLOAD_LAYOUT_LEGACY_43,
    PAYLOAD_LAYOUT_V542_A0_45,
    PRECISION_VERSION_V543,
    CoarseEncoder,
    CoarsePacket,
    V543ContractError,
    V543HookError,
    V543InitError,
    V543LoadError,
    V543SupervisionLeakError,
    assert_encoder_packet_parity,
    classify_payload_layout,
    forked_global_rng,
    identity_mapping_table,
    measured_encoder_params,
    resolve_a0_mapping,
    validate_coarse_packet,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_gate_batch,
)

SEED = 17


@pytest.fixture()
def batch():
    return forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=SEED))


@pytest.fixture()
def legacy_payload() -> dict:
    """A deterministic legacy 43-key joint state dict (V53-MP A0 form)."""
    torch.manual_seed(SEED)
    legacy = KeyAtomV53MPModel()
    return {"model": {k: v.clone() for k, v in legacy.state_dict().items()}}


@pytest.fixture()
def v542_payload() -> dict:
    """A deterministic V5.4.2 A0 45-key payload (``w_v`` dual-write material).

    This is the REAL S0-A asset form (``pesn_v542_train_v1``): the joint
    state dict of :class:`V542Model`, where ``backbone.w_v.*`` and
    ``move_decoder.w_v.*`` both exist (one shared tensor object).
    """
    torch.manual_seed(SEED)
    model = V542Model()
    return {"model": {k: v.clone() for k, v in model.state_dict().items()}}


def _materialized_45(legacy_payload: dict) -> dict:
    """Legacy 43-key dict + the materialized ``w_v`` dual-write (45 keys)."""
    state = dict(legacy_payload["model"])
    state["move_decoder.w_v.weight"] = legacy_payload["model"][
        "backbone.w_v.weight"].clone()
    state["move_decoder.w_v.bias"] = legacy_payload["model"][
        "backbone.w_v.bias"].clone()
    return {"model": state}


@pytest.fixture()
def encoder(legacy_payload) -> CoarseEncoder:
    return CoarseEncoder.load(legacy_payload, SEED)


# ---------------------------------------------------------------------------
# frozen constants (plan §5.2 / §5.3 / §5.8 / §6.4)
# ---------------------------------------------------------------------------
def test_contract_constants_pin_v542_semantics() -> None:
    assert ENCODER_Z_DIM_V543 == Z_DIM_V3 == 64
    assert KEY_S_DIM_V543 == HIDDEN_V543 == HIDDEN_V542 == 48
    assert EDGE_MSG_DIM_V543 == EDGE_MSG_DIM_V542 == 48
    assert A_REF_V543 == A_REF_V542 == 0.08
    assert ENCODER_ACTIVE_PARAMS_V543 == A0_ACTIVE_PARAMS_V542 == 55_392
    assert N_DECODER_PARAMS_V543 == N_DECODER_PARAMS_V542 == 49
    assert N_PROCESS_PARAMS_V543 == N_PROCESS_PARAMS_V542 == 2_178
    assert ENCODER_JOINT_PARAMS_V543 == A0_JOINT_PARAMS_V542 == 57_619
    # plan §5.8 tolerances: CPU bitwise, GPU ≤1e-6 (never loosened)
    assert PARITY_ATOL_CPU == 0.0
    assert PARITY_ATOL_GPU == 1e-6
    # plan §5.2 schema/precision literals the WP2 Bridge must emit
    assert DECODER_CONTEXT_SCHEMA_V543 == "decoder_context_v1"
    assert PRECISION_VERSION_V543 == "fp32_v1"
    # plan §4.4 / §5.8 hook disciplines
    assert ENCODER_FORWARD_CALLS_PER_VIEW == 1
    assert HOOK_FIRES_PER_FORWARD == 1
    assert HOOK_LABELS == ("message_mlp_last", "norm_h_last", "progress_fc2")


# ---------------------------------------------------------------------------
# CoarsePacket contract (plan §11.2)
# ---------------------------------------------------------------------------
def test_coarse_packet_fields_frozen_and_slotted() -> None:
    fields = {f.name for f in dataclasses.fields(CoarsePacket)}
    assert fields == set(COARSE_PACKET_KEYS)
    assert fields == {"z", "edge_msg", "node_h", "u0", "q0_raw", "e0"}
    for forbidden in ("pos", "u", "q", "edge_index", "key_index"):
        assert forbidden not in fields
    assert set(CoarsePacket.__slots__) == set(COARSE_PACKET_KEYS)
    packet = CoarsePacket(
        z=torch.zeros(2, 64), edge_msg=torch.zeros(3, 48),
        node_h=torch.zeros(5, 48), u0=torch.zeros(2, 3),
        q0_raw=torch.zeros(2), e0=torch.zeros(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet.z = torch.ones(2, 64)  # type: ignore[misc]
    assert packet.n_graphs == 2 and packet.n_keys == 2
    assert packet.n_nodes == 5 and packet.n_edges == 3
    # ``q0`` is DERIVED (never stored): tanh of the raw scalar
    raw = torch.tensor([-3.0, 0.0, 0.5])
    derived = CoarsePacket(
        z=torch.zeros(2, 64), edge_msg=torch.zeros(0, 48),
        node_h=torch.zeros(5, 48), u0=torch.zeros(2, 3),
        q0_raw=raw, e0=torch.zeros(2))
    assert torch.equal(derived.q0, torch.tanh(raw))
    assert COARSE_PACKET_SCHEMA == "pesn_v543_coarse_packet_v1"


def test_validate_coarse_packet_typed_refusals() -> None:
    good = CoarsePacket(
        z=torch.zeros(2, 64), edge_msg=torch.zeros(3, 48),
        node_h=torch.zeros(5, 48), u0=torch.zeros(2, 3),
        q0_raw=torch.zeros(2), e0=torch.zeros(2))
    validate_coarse_packet(good, n_graphs=2, n_nodes=5, n_keys=2, n_edges=3)
    # (a) wrong batch size in z
    with pytest.raises(V543ContractError) as exc:
        validate_coarse_packet(good, n_graphs=3, n_nodes=5, n_keys=2, n_edges=3)
    assert exc.value.code == "V543_PACKET_Z_SHAPE"
    # (b) wrong u0 row count
    with pytest.raises(V543ContractError) as exc:
        validate_coarse_packet(good, n_graphs=2, n_nodes=5, n_keys=4, n_edges=3)
    assert exc.value.code == "V543_PACKET_U0_SHAPE"
    # (c) wrong dtype
    bad_dtype = dataclasses.replace(good, e0=torch.zeros(2, dtype=torch.float64))
    with pytest.raises(V543ContractError) as exc:
        validate_coarse_packet(bad_dtype, n_graphs=2, n_nodes=5, n_keys=2,
                               n_edges=3)
    assert exc.value.code == "V543_PACKET_DTYPE"
    # (d) non-finite value
    bad_value = dataclasses.replace(
        good, q0_raw=torch.tensor([0.0, float("nan")]))
    with pytest.raises(V543ContractError) as exc:
        validate_coarse_packet(bad_value, n_graphs=2, n_nodes=5, n_keys=2,
                               n_edges=3)
    assert exc.value.code == "V543_PACKET_NONFINITE"
    # (e) not a packet at all
    with pytest.raises(V543ContractError) as exc:
        validate_coarse_packet(object(), n_graphs=2, n_nodes=5, n_keys=2,
                               n_edges=3)  # type: ignore[arg-type]
    assert exc.value.code == "V543_PACKET_TYPE"


# ---------------------------------------------------------------------------
# accounting (plan §6.4 / §11.2)
# ---------------------------------------------------------------------------
def test_encoder_param_invariants_and_state_dict_keys(encoder) -> None:
    report = measured_encoder_params()
    assert report == {
        "backbone_active": 55_392,
        "decoder": 49,
        "process_head": 2_178,
        "joint": 57_619,
    }
    assert encoder.active_param_count == ENCODER_ACTIVE_PARAMS_V543
    assert sum(int(p.numel()) for p in encoder.parameters()) == \
        ENCODER_JOINT_PARAMS_V543
    # the shared w_v is ONE module object; its 49 params counted once
    assert encoder.backbone.w_v is encoder.move_decoder.w_v
    # the A0 state-dict surface the frozen loader mapping targets:
    keys = set(encoder.state_dict())
    assert len(keys) == 45
    assert {"backbone.w_v.weight", "move_decoder.w_v.weight"} <= keys
    assert {k for k in keys if k.startswith("process_head.")} == {
        "process_head.progress_head.fc1.weight",
        "process_head.progress_head.fc1.bias",
        "process_head.progress_head.fc2.weight",
        "process_head.progress_head.fc2.bias",
        "process_head.endpoint_head.fc.weight",
        "process_head.endpoint_head.fc.bias",
    }
    assert len([k for k in keys if k.startswith("backbone.")]) == 37


# ---------------------------------------------------------------------------
# strict load, legacy 43-key layout: 43 source keys -> 45 targets
# (plan §7.1 S0-A; §11.3)
# ---------------------------------------------------------------------------
def test_load_strict_45_keys_from_mock_payload(legacy_payload) -> None:
    encoder = CoarseEncoder.load(legacy_payload, SEED)
    audit = encoder.load_audit
    assert audit is not None
    assert audit["ok"] is True and audit["strict"] is True
    assert audit["payload_layout"] == PAYLOAD_LAYOUT_LEGACY_43
    assert audit["mapping_mode"] == MAPPING_MODE_FROZEN
    assert audit["n_source_keys"] == 43
    assert audit["n_mapped_targets"] == 45
    assert audit["torch_equal_verified"] == 45
    inner = audit["load_audit"]
    assert inner["missing"] == [] and inner["unexpected"] == []
    assert inner["shape_mismatch"] == [] and inner["dtype_mismatch"] == []
    assert len(str(audit["mapping_table_sha256"])) == 64
    # the declared w_v dual-write is part of the frozen mapping
    assert set(inner["dual_write_aliases"]) == {
        "backbone.w_v.weight", "backbone.w_v.bias"}
    # every source tensor landed bit-for-bit
    src = legacy_payload["model"]
    state = encoder.state_dict()
    for key, value in src.items():
        assert torch.equal(state[key], value), key
    assert torch.equal(state["move_decoder.w_v.weight"],
                       src["backbone.w_v.weight"])


# ---------------------------------------------------------------------------
# strict load, V5.4.2 A0 45-key layout: identity mapping 45 -> 45
# (the REAL S0-A asset form; see module docstring #2 of coarse_encoder)
# ---------------------------------------------------------------------------
def test_load_v542_a0_45_keys_identity_path(v542_payload) -> None:
    encoder = CoarseEncoder.load(v542_payload, SEED)
    audit = encoder.load_audit
    assert audit is not None
    assert audit["ok"] is True and audit["strict"] is True
    assert audit["payload_layout"] == PAYLOAD_LAYOUT_V542_A0_45
    assert audit["mapping_mode"] == MAPPING_MODE_IDENTITY
    assert audit["n_source_keys"] == 45
    assert audit["n_mapped_targets"] == 45
    assert audit["torch_equal_verified"] == 45
    inner = audit["load_audit"]
    assert inner["missing"] == [] and inner["unexpected"] == []
    assert inner["shape_mismatch"] == [] and inner["dtype_mismatch"] == []
    assert len(str(audit["mapping_table_sha256"])) == 64
    # the identity table declares NO alias (the dual-write is already
    # materialized in the payload); the frozen loader still re-checks that
    # the two w_v slot families hold equal values below
    assert inner["dual_write_aliases"] == {}
    # every source tensor landed bit-for-bit (45/45, both w_v slot families)
    src = v542_payload["model"]
    state = encoder.state_dict()
    assert set(state) == set(src)
    for key, value in src.items():
        assert torch.equal(state[key], value), key
    assert torch.equal(state["move_decoder.w_v.weight"],
                       src["move_decoder.w_v.weight"])
    assert encoder.backbone.w_v is encoder.move_decoder.w_v
    described = encoder.describe()
    assert described["payload_layout"] == PAYLOAD_LAYOUT_V542_A0_45
    assert described["mapping_mode"] == MAPPING_MODE_IDENTITY


def test_load_v542_45_keys_reproduces_v542_forward(v542_payload, batch) -> None:
    """The 45-key identity path is a VALUE load, not a fork: full parity."""
    torch.manual_seed(SEED + 5)
    reference = V542Model()
    load_a0_strict(reference, v542_payload,
                   resolve_a0_mapping(
                       v542_payload["model"],
                       target_keys=set(reference.state_dict()))[1])
    encoder = CoarseEncoder.load(v542_payload, SEED)
    encoder.eval()
    reference.eval()
    with torch.no_grad():
        packet = encoder(batch)
        ref = reference(batch)
        ref_packet = reference.backbone(batch)
    rows = assert_encoder_packet_parity(
        packet, z=ref["z"], u0=ref["u"], q=ref["q"],
        endpoint_logit=ref["endpoint_logit"],
        edge_msg=ref_packet.edge_msg,
        context="45-key identity load vs v542 model")
    for name, row in rows.items():
        assert row["bitwise"] is True, name


def test_both_layouts_load_the_same_values(legacy_payload) -> None:
    """Legacy 43-key and materialized 45-key forms → identical encoders."""
    from_legacy = CoarseEncoder.load(legacy_payload, SEED)
    from_45 = CoarseEncoder.load(_materialized_45(legacy_payload), SEED)
    legacy_state = from_legacy.state_dict()
    state_45 = from_45.state_dict()
    assert set(legacy_state) == set(state_45)
    for key in legacy_state:
        assert torch.equal(legacy_state[key], state_45[key]), key
    # only the ROUTE differs, and both are recorded
    assert from_legacy.load_audit["payload_layout"] == PAYLOAD_LAYOUT_LEGACY_43
    assert from_legacy.load_audit["mapping_mode"] == MAPPING_MODE_FROZEN
    assert from_45.load_audit["payload_layout"] == PAYLOAD_LAYOUT_V542_A0_45
    assert from_45.load_audit["mapping_mode"] == MAPPING_MODE_IDENTITY
    assert (from_legacy.load_audit["encoder_ckpt_sha256"]
            != from_45.load_audit["encoder_ckpt_sha256"])


def test_layout_classification_and_identity_table(legacy_payload,
                                                  v542_payload) -> None:
    target_keys = set(CoarseEncoder().state_dict())
    assert classify_payload_layout(
        legacy_payload["model"], target_keys=target_keys) == \
        PAYLOAD_LAYOUT_LEGACY_43
    assert classify_payload_layout(
        v542_payload["model"], target_keys=target_keys) == \
        PAYLOAD_LAYOUT_V542_A0_45
    layout, table = resolve_a0_mapping(
        legacy_payload["model"], target_keys=target_keys)
    assert (layout, table) == (PAYLOAD_LAYOUT_LEGACY_43, None)
    layout, table = resolve_a0_mapping(
        v542_payload["model"], target_keys=target_keys)
    assert layout == PAYLOAD_LAYOUT_V542_A0_45
    assert table == identity_mapping_table(v542_payload["model"])
    assert len(table) == 45
    assert all(targets == [key] for key, targets in table.items())
    assert PAYLOAD_LAYOUTS_V543 == (
        PAYLOAD_LAYOUT_LEGACY_43, PAYLOAD_LAYOUT_V542_A0_45)


def test_load_refuses_unknown_payload_layout(legacy_payload) -> None:
    """A key set matching NEITHER frozen layout is a typed refusal."""
    broken = {k: v for k, v in legacy_payload["model"].items()
              if k != "backbone.z_head.weight"}          # 42 keys
    with pytest.raises(V543LoadError) as exc:
        CoarseEncoder.load({"model": broken}, SEED)
    assert exc.value.code == "V543_CKPT_LAYOUT"
    assert "backbone.z_head.weight" in exc.value.detail
    assert PAYLOAD_LAYOUT_LEGACY_43 in exc.value.detail
    assert PAYLOAD_LAYOUT_V542_A0_45 in exc.value.detail


def test_load_strict_audit_still_runs_inside_a_valid_layout(
        legacy_payload, v542_payload) -> None:
    """Layout detection never bypasses the frozen V5.4.2 strict audit."""
    # (a) shape drift inside the legacy 43-key layout
    shaped = dict(legacy_payload["model"])
    shaped["backbone.z_head.weight"] = shaped[
        "backbone.z_head.weight"].t().contiguous()
    with pytest.raises(V542LoadError) as exc:
        CoarseEncoder.load({"model": shaped}, SEED)
    assert exc.value.code == "V542_LOAD_SHAPE"
    # (b) dtype drift inside the V542 45-key layout
    typed = dict(v542_payload["model"])
    typed["process_head.progress_head.fc2.weight"] = typed[
        "process_head.progress_head.fc2.weight"].to(torch.float64)
    with pytest.raises(V542LoadError) as exc:
        CoarseEncoder.load({"model": typed}, SEED)
    assert exc.value.code == "V542_LOAD_DTYPE"


def test_load_records_ckpt_sha_and_describe(legacy_payload) -> None:
    encoder = CoarseEncoder.load(legacy_payload, 29)
    audit = encoder.load_audit
    assert audit is not None
    assert len(str(audit["encoder_ckpt_sha256"])) == 64
    assert audit["seed"] == 29
    # the load audit names the mapping route AND the RNG-fork coverage
    assert audit["payload_layout"] == PAYLOAD_LAYOUT_LEGACY_43
    assert audit["mapping_mode"] == MAPPING_MODE_FROZEN
    assert len(str(audit["mapping_table_sha256"])) == 64
    assert audit["rng_fork"]["cpu"] is True
    assert audit["rng_fork"]["cuda_devices"] == (
        list(range(torch.cuda.device_count()))
        if torch.cuda.is_available() else [])
    described = encoder.describe()
    assert described["version"] == "pesn_key_atom_v543_a0_coarse_encoder_v1"
    assert described["packet_fields"] == list(COARSE_PACKET_KEYS)
    assert [h["label"] for h in described["hooks"]] == list(HOOK_LABELS)
    assert described["encoder_ckpt_sha256"] == audit["encoder_ckpt_sha256"]
    assert described["payload_layout"] == PAYLOAD_LAYOUT_LEGACY_43
    assert described["mapping_mode"] == MAPPING_MODE_FROZEN
    assert described["parity_atol"] == {"cpu": 0.0, "gpu": 1e-6}
    # a fresh (unloaded) encoder reports no checkpoint / no layout
    fresh = CoarseEncoder().describe()
    assert fresh["encoder_ckpt_sha256"] is None
    assert fresh["payload_layout"] is None and fresh["mapping_mode"] is None


def test_load_refuses_absent_checkpoint(tmp_path) -> None:
    missing = tmp_path / "checkpoint_best.pt"
    with pytest.raises(V543LoadError) as exc:
        CoarseEncoder.load(str(missing), SEED)
    assert exc.value.code == "V543_CKPT_MISSING"


def test_load_never_perturbs_global_rng(legacy_payload, v542_payload) -> None:
    """CPU **and** every visible CUDA generator must be restored byte-exactly.

    Regression for the leak found on the ``pes2ts-gpu`` host: ``load`` seeds
    explicitly (``torch.manual_seed`` → also ``cuda.manual_seed_all``), so a
    CPU-only ``fork_rng(devices=[])`` let the seed escape into all CUDA
    generator states.  The fork now covers every visible CUDA device
    (module docstring #7 of ``coarse_encoder``).
    """
    for payload in (legacy_payload, v542_payload):
        torch.manual_seed(4242)
        cpu_before = torch.random.get_rng_state()
        cuda_before = (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None)
        CoarseEncoder.load(payload, SEED)
        assert torch.equal(torch.random.get_rng_state(), cpu_before)
        if cuda_before:  # CUDA hosts with >= 1 visible device
            after = torch.cuda.get_rng_state_all()
            assert len(after) == len(cuda_before)
            for index, (left, right) in enumerate(zip(cuda_before, after)):
                assert torch.equal(left, right), (
                    f"CUDA RNG state of device {index} leaked by load")


def test_forked_global_rng_restores_all_generators() -> None:
    """The RNG-fork primitive itself: CPU + CUDA, with a sensitivity control."""
    torch.manual_seed(2024)
    cpu_before = torch.random.get_rng_state()
    cuda_before = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)
    with forked_global_rng(seed=99) as devices:
        # inside the fork: the seed IS applied (construction determinism)
        assert not torch.equal(torch.random.get_rng_state(), cpu_before)
        expected_devices = (list(range(torch.cuda.device_count()))
                            if torch.cuda.is_available() else [])
        assert devices == expected_devices
        _ = torch.randn(4)                       # the body may draw freely
        if cuda_before:
            _ = torch.randn(4, device="cuda")
    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    if cuda_before:  # CUDA hosts with >= 1 visible device
        after = torch.cuda.get_rng_state_all()
        for index, (left, right) in enumerate(zip(cuda_before, after)):
            assert torch.equal(left, right), f"device {index} leaked"
        # SENSITIVITY CONTROL: this test would have caught the CPU-only fork
        # bug — seeding outside a widened fork DOES move the CUDA states.
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1234)
            leaked = torch.cuda.get_rng_state_all()
            assert any(not torch.equal(left, right)
                       for left, right in zip(cuda_before, leaked)), \
                "negative control failed: a CPU-only fork did not leak"
        finally:
            torch.cuda.set_rng_state_all(cuda_before)
            torch.set_rng_state(cpu_before)


# ---------------------------------------------------------------------------
# forward: packet shapes + V542 bitwise parity on shared weights
# ---------------------------------------------------------------------------
def test_forward_packet_shapes(encoder, batch) -> None:
    encoder.eval()
    with torch.no_grad():
        packet = encoder(batch)
    n_graphs = int(batch["cv_target_delta"].shape[0])
    assert isinstance(packet, CoarsePacket)
    assert tuple(packet.z.shape) == (n_graphs, ENCODER_Z_DIM_V543)
    assert tuple(packet.q0_raw.shape) == (n_graphs,)
    assert tuple(packet.e0.shape) == (n_graphs,)
    assert packet.n_graphs == int(packet.z.shape[0])
    assert tuple(packet.node_h.shape) == (
        int(batch["pos"].shape[0]), HIDDEN_V543)
    assert tuple(packet.edge_msg.shape) == (
        int(batch["edge_index"].shape[1]), EDGE_MSG_DIM_V543)
    assert tuple(packet.u0.shape) == (int(batch["key_index"].shape[0]), 3)


def test_forward_parity_with_v542_model_on_shared_weights(
        legacy_payload, batch) -> None:
    """plan §5.8 equalities 1/2/3/4/5 on the SAME weights (CPU bitwise)."""
    encoder = CoarseEncoder.load(legacy_payload, SEED)
    reference = V542Model()
    load_a0_strict(reference, legacy_payload)
    encoder.eval()
    reference.eval()
    with torch.no_grad():
        packet = encoder(batch)
        ref = reference(batch)
        ref_packet = reference.backbone(batch)
    rows = assert_encoder_packet_parity(
        packet,
        z=ref["z"],
        u0=ref["u"],
        q=ref["q"],
        endpoint_logit=ref["endpoint_logit"],
        edge_msg=ref_packet.edge_msg,
        context="v543 encoder vs v542 model",
    )
    for name, row in rows.items():
        assert row["bitwise"] is True, name
        assert row["max_abs_diff"] == 0.0, name
        assert row["atol"] == 0.0, name
    # order_score IS q on the reference side (unchanged V542 wiring)
    assert ref["order_score"] is ref["q"]
    # the packet's derived q0 is bitwise the reference q
    assert torch.equal(packet.q0, ref["q"])


# ---------------------------------------------------------------------------
# hooks: exactly once, always removed (plan §5.8; §11.5)
# ---------------------------------------------------------------------------
def _hook_modules(encoder: CoarseEncoder):
    return (encoder.backbone.message_mlp[-1],
            encoder.backbone.norm_h[-1],
            encoder.process_head.progress_head.fc2)


def test_hooks_fire_once_and_are_removed(encoder, batch) -> None:
    modules = _hook_modules(encoder)
    assert all(len(m._forward_hooks) == 0 for m in modules)
    encoder.eval()
    with torch.no_grad():
        encoder(batch)
    # finally-removed: nothing survives the forward
    assert all(len(m._forward_hooks) == 0 for m in modules)
    assert encoder.last_hook_fires == {
        "message_mlp_last": 1, "norm_h_last": 1, "progress_fc2": 1}
    assert encoder.cumulative_hook_fires == encoder.last_hook_fires
    # independent count on the message module (the backbone's own hook is
    # separate and must not change the legacy count)
    seen = []
    handle = encoder.backbone.message_mlp[-1].register_forward_hook(
        lambda _m, _i, out: seen.append(out))
    try:
        with torch.no_grad():
            packet = encoder(batch)
    finally:
        handle.remove()
    assert len(seen) == 1
    assert torch.equal(packet.edge_msg, seen[0])
    assert encoder.last_hook_fires == {
        "message_mlp_last": 1, "norm_h_last": 1, "progress_fc2": 1}


def test_hooks_removed_on_typed_invalid_abort(encoder, batch) -> None:
    modules = _hook_modules(encoder)
    bad = dict(batch)
    bad["degK"] = torch.full_like(batch["degK"], float("inf"))
    before = encoder.forward_count
    with pytest.raises(RuntimeError):
        encoder(bad)
    assert all(len(m._forward_hooks) == 0 for m in modules)
    # an aborted forward is NOT a forward (plan §4.4 counter discipline)
    assert encoder.forward_count == before
    assert encoder.last_hook_fires == dict.fromkeys(HOOK_LABELS, 0)


def test_forward_count_and_reset(encoder, batch) -> None:
    encoder.eval()
    assert encoder.forward_count == 0
    with torch.no_grad():
        encoder(batch)
        encoder(batch)
    assert encoder.forward_count == 2
    assert encoder.cumulative_hook_fires == {
        "message_mlp_last": 2, "norm_h_last": 2, "progress_fc2": 2}
    encoder.reset_counters()
    assert encoder.forward_count == 0
    assert encoder.cumulative_hook_fires == dict.fromkeys(HOOK_LABELS, 0)


# ---------------------------------------------------------------------------
# supervision refusal (plan §5.6) + legacy whitelist
# ---------------------------------------------------------------------------
def test_forward_refuses_supervision_keys(encoder, batch) -> None:
    for leak in ("y_teacher", "pair_edit", "frame_weight", "d0"):
        bad = dict(batch)
        bad[leak] = torch.zeros(2)
        with pytest.raises(V543SupervisionLeakError) as exc:
            encoder(bad)
        assert exc.value.code == "V543_SUPERVISION_LEAK"
        assert leak in exc.value.detail
    # refused BEFORE any compute: no counter moved, no hook left behind
    assert encoder.forward_count == 0
    assert all(len(m._forward_hooks) == 0 for m in _hook_modules(encoder))


def test_forward_refuses_other_out_of_contract_keys(encoder, batch) -> None:
    bad = dict(batch)
    bad["branch_id"] = torch.zeros(2)
    with pytest.raises(KeyAtomV3ContractError) as exc:
        encoder(bad)
    assert exc.value.code == "FORWARD_VIEW_LEAK"
    assert encoder.forward_count == 0


# ---------------------------------------------------------------------------
# determinism + autograd (plan §5.8: hooks never detach)
# ---------------------------------------------------------------------------
def test_forward_determinism_two_eval_calls(encoder, batch) -> None:
    encoder.eval()
    with torch.no_grad():
        first = encoder(batch)
        second = encoder(batch)
    for field in COARSE_PACKET_KEYS:
        assert torch.equal(getattr(first, field), getattr(second, field)), field


def test_forward_keeps_autograd_graph(encoder, batch) -> None:
    encoder.train()
    packet = encoder(batch)
    assert packet.edge_msg.requires_grad is True
    assert packet.node_h.requires_grad is True
    assert packet.q0_raw.requires_grad is True
    assert packet.e0.requires_grad is True
    assert packet.u0.requires_grad is True
    assert packet.edge_msg.grad_fn is not None
    assert packet.node_h.grad_fn is not None
    # a backward really reaches backbone + head parameters (no detach)
    loss = (packet.u0.sum() + packet.z.sum() + packet.node_h.sum()
            + packet.q0_raw.sum() + packet.e0.sum())
    loss.backward()
    grad_modules = {
        name for name, p in encoder.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0.0}
    assert "backbone.message_mlp.0.0.weight" in grad_modules
    assert "process_head.progress_head.fc2.weight" in grad_modules


# ---------------------------------------------------------------------------
# construction refusals (plan §11.3: no copied w_v, no torch.compile)
# ---------------------------------------------------------------------------
def test_encoder_refuses_foreign_decoder_and_unshared_w_v() -> None:
    with pytest.raises(V543InitError) as exc:
        CoarseEncoder(move_decoder=object())  # type: ignore[arg-type]
    assert exc.value.code == "V543_ENCODER_DECODER_TYPE"

    torch.manual_seed(SEED)
    other_decoder = DirectMoveDecoder()
    own_decoder = DirectMoveDecoder()
    backbone = A0LegacyBackbone(own_decoder)
    with pytest.raises(V543InitError) as exc:
        CoarseEncoder(backbone, move_decoder=other_decoder)
    assert exc.value.code == "V543_ENCODER_WV_NOT_SHARED"

    with pytest.raises(V543InitError) as exc:
        CoarseEncoder(backbone, move_decoder=own_decoder,
                      process_head=object())  # type: ignore[arg-type]
    assert exc.value.code == "V543_ENCODER_PROCESS_HEAD"
    # the valid wiring builds
    encoder = CoarseEncoder(backbone, move_decoder=own_decoder)
    assert encoder.backbone.w_v is own_decoder.w_v


def test_encoder_refuses_compiled_tree() -> None:
    torch.manual_seed(SEED)
    decoder = DirectMoveDecoder()
    backbone = A0LegacyBackbone(decoder)
    backbone.z_head._orig_mod = torch.nn.Linear(2, 2)
    with pytest.raises(Exception) as exc:
        CoarseEncoder(backbone, move_decoder=decoder)
    assert getattr(exc.value, "code", "") == "V542_TORCH_COMPILE_FORBIDDEN"
    del backbone.z_head._orig_mod
    # a clean tree passes
    encoder = CoarseEncoder(backbone, move_decoder=decoder)
    for name, child in encoder.named_modules():
        assert not hasattr(child, "_orig_mod"), name


def test_no_torch_compile_in_v543_sources() -> None:
    from pes2ts_core.pesn.models.v543 import coarse_encoder, contracts

    for module in (contracts, coarse_encoder):
        source = open(module.__file__, encoding="utf-8").read()
        assert "torch.compile(" not in source, module.__name__


def test_hook_error_is_typed_on_missing_fire() -> None:
    from pes2ts_core.pesn.models.v543 import assert_single_fire

    with pytest.raises(V543HookError) as exc:
        assert_single_fire([], "progress_fc2")
    assert exc.value.code == "V543_HOOK_FIRE_COUNT"
    assert "0 time(s)" in exc.value.detail


def test_process_head_is_frozen_v53mp(encoder) -> None:
    assert isinstance(encoder.process_head, ReactionProcessHeadV53MP)
    assert encoder.process_head.progress_head.fc2.out_features == 1
    assert sum(int(p.numel()) for p in encoder.process_head.parameters()) == 2_178
