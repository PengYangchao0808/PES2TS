"""V5.4.3 WP1 — REAL-checkpoint A0 parity (S0-A, plan §7.1 / §5.8).

Marked ``realdata``: pytest.ini's default ``addopts`` EXCLUDES this file
(``-m "not realdata ..."``), so it only runs when explicitly requested,
inside the GPU host (WSL ``pes2ts-gpu``)::

    pytest -m realdata tests/pesn/test_v543_encoder_parity_realdata.py -q

Assets (plan §7.1 S0-A mapping ``V543_M0_Sxx`` ← ``V542_A0_Sxx``)::

    outputs/pesn/v542_backbone_50k/runs/V542_A0_S17/checkpoint_best.pt
    outputs/pesn/v542_backbone_50k/runs/V542_A0_S29/checkpoint_best.pt
    outputs/pesn/v542_backbone_50k/runs/V542_A0_S43/checkpoint_best.pt

Per seed this file proves the plan §5.8 equalities — ``u0 == u_V542``,
``z == z_V542``, ``q0 == q_V542``, ``tanh(q0_raw) == q0``,
``endpoint0 == endpoint_logit_V542``, ``order_score is q`` — plus the
plan §11.5 hook disciplines, on:

1. **8 fixture geometric batches** (real weights, synthetic geometry) —
   always runs when the checkpoint is present; this is the fast
   certification leg;
2. **>= 8 REAL inner-val batches** (the frozen V542 data view via
   ``scripts.agent_ops.v542_a0_parity.inner_val_batches``) — the full
   S0-A leg; skips with an explicit reason when the pinned data view is
   not mounted in this environment.

Environment discipline: a checkpoint that is ABSENT is a clean skip; a
checkpoint that is present and fails parity FAILS HARD (no tolerance
loosening, no "approximate" pass, plan §7.1 S0-A "不得用「近似相等」放行").

Payload form: the S0-A assets are the V5.4.2 **45-key** joint state dicts
(``payload["model"]``, schema ``pesn_v542_train_v1``, the ``w_v``
dual-write already materialized).  They are therefore loaded through the
IDENTITY mapping that ``resolve_a0_mapping`` selects — the same mapping is
handed to the frozen ``V542Model`` reference, so one checkpoint is loaded
into both targets through one recorded table (``coarse_encoder`` module
docstring #2)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.models.v543 import (  # noqa: E402
    MAPPING_MODE_IDENTITY,
    PARITY_ATOL_CPU,
    PAYLOAD_LAYOUT_V542_A0_45,
    CoarseEncoder,
    assert_encoder_packet_parity,
    assert_q0_raw_matches_q,
    resolve_a0_mapping,
)
from pes2ts_core.pesn.models.v542 import (  # noqa: E402
    V542Model,
    load_a0_strict,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    make_v3_gate_batch,
)

pytestmark = pytest.mark.realdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_TEMPLATE = "outputs/pesn/v542_backbone_50k/runs/V542_A0_S{seed}/checkpoint_best.pt"
SEEDS = (17, 29, 43)
#: Fixture batches per seed (the plan's ">= 8 batches" cadence on synthetic
#: geometry; the real inner-val leg below uses the frozen data view).
FIXTURE_BATCHES = 8
DATA_VIEW = Path("/pes2ts_data/v53mp_50k_data")


def _checkpoint_path(seed: int) -> Path:
    return PROJECT_ROOT / CKPT_TEMPLATE.format(seed=int(seed))


def _require_checkpoint(seed: int) -> Path:
    path = _checkpoint_path(seed)
    if not path.is_file():
        pytest.skip(f"A0 checkpoint absent in this environment: {path}")
    return path


@pytest.fixture(scope="module")
def payloads() -> dict[int, dict]:
    """Strict-agnostic raw payload per seed (absent seeds are dropped)."""
    loaded: dict[int, dict] = {}
    for seed in SEEDS:
        path = _checkpoint_path(seed)
        if not path.is_file():
            continue
        loaded[seed] = torch.load(str(path), map_location="cpu",
                                  weights_only=False)
    if not loaded:
        pytest.skip("no V542_A0 checkpoint present under "
                    f"{PROJECT_ROOT / 'outputs/pesn/v542_backbone_50k/runs'}")
    return loaded


def _state_of(payload: dict) -> dict:
    return payload["model"] if "model" in payload else payload


def _load_reference(payload: dict):
    """Strict-load the V5.4.2 reference through the SAME mapping as the encoder.

    The S0-A assets are 45-key ``v542_a0_joint_45`` payloads (the ``w_v``
    dual-write already materialized), so the frozen 43-key dual-write
    mapping must NOT be used for them — see ``coarse_encoder`` docstring #2.
    """
    reference = V542Model()
    _layout, table = resolve_a0_mapping(
        _state_of(payload), target_keys=set(reference.state_dict()))
    load_a0_strict(reference, payload, table)
    return reference


def _pair(seed: int, payload: dict):
    """``(CoarseEncoder, V542Model)`` from ONE checkpoint, both strict-loaded."""
    encoder = CoarseEncoder.load(payload, int(seed))
    reference = _load_reference(payload)
    encoder.eval()
    reference.eval()
    return encoder, reference


@pytest.mark.parametrize("seed", SEEDS)
def test_real_checkpoint_strict_load_audit(seed: int, payloads) -> None:
    _require_checkpoint(seed)
    if seed not in payloads:  # pragma: no cover - guarded by the skip above
        pytest.skip(f"payload for seed {seed} unavailable")
    encoder = CoarseEncoder.load(payloads[seed], seed)
    audit = encoder.load_audit
    assert audit is not None
    assert audit["ok"] is True and audit["strict"] is True
    # the real S0-A assets are the V5.4.2 45-key joint form
    assert audit["payload_layout"] == PAYLOAD_LAYOUT_V542_A0_45
    assert audit["mapping_mode"] == MAPPING_MODE_IDENTITY
    assert audit["n_source_keys"] == 45
    assert audit["n_mapped_targets"] == 45
    assert audit["torch_equal_verified"] == 45
    assert len(str(audit["encoder_ckpt_sha256"])) == 64
    inner = audit["load_audit"]
    assert inner["missing"] == [] and inner["unexpected"] == []
    assert inner["shape_mismatch"] == [] and inner["dtype_mismatch"] == []
    # the same checkpoint strict-loads into the V542 reference with the SAME
    # mapping digest (one checkpoint, one mapping, two targets)
    reference = V542Model()
    _layout, table = resolve_a0_mapping(
        _state_of(payloads[seed]), target_keys=set(reference.state_dict()))
    v542_audit = load_a0_strict(reference, payloads[seed], table)
    assert v542_audit["mapping_table_sha256"] == inner["mapping_table_sha256"]


@pytest.mark.parametrize("seed", SEEDS)
def test_real_checkpoint_weight_identity(seed: int, payloads) -> None:
    _require_checkpoint(seed)
    if seed not in payloads:  # pragma: no cover
        pytest.skip(f"payload for seed {seed} unavailable")
    encoder, reference = _pair(seed, payloads[seed])
    enc_state = encoder.state_dict()
    ref_state = reference.state_dict()
    assert set(enc_state) == set(ref_state)
    for key in enc_state:
        assert torch.equal(enc_state[key], ref_state[key]), key
    assert encoder.backbone.w_v is encoder.move_decoder.w_v


@pytest.mark.parametrize("seed", SEEDS)
def test_real_checkpoint_forward_parity_on_fixtures(seed: int, payloads) -> None:
    """plan §5.8 equalities 1/2/3/4/5 on 8 batches (CPU bitwise)."""
    _require_checkpoint(seed)
    if seed not in payloads:  # pragma: no cover
        pytest.skip(f"payload for seed {seed} unavailable")
    encoder, reference = _pair(seed, payloads[seed])
    n_checked = 0
    for index in range(FIXTURE_BATCHES):
        batch = forward_view_v3(
            make_v3_gate_batch(n_graphs=2, jitter_seed=seed + index))
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
            context=f"seed {seed} fixture batch {index}",
        )
        for name, row in rows.items():
            assert row["bitwise"] is True, (seed, index, name)
            assert row["max_abs_diff"] == 0.0, (seed, index, name)
        assert ref["order_score"] is ref["q"]
        n_checked += 1
    assert n_checked == FIXTURE_BATCHES
    assert encoder.forward_count == FIXTURE_BATCHES


@pytest.mark.parametrize("seed", SEEDS)
def test_real_checkpoint_q0_raw_tanh_identity(seed: int, payloads) -> None:
    _require_checkpoint(seed)
    if seed not in payloads:  # pragma: no cover
        pytest.skip(f"payload for seed {seed} unavailable")
    encoder, reference = _pair(seed, payloads[seed])
    batch = forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=seed))
    with torch.no_grad():
        packet = encoder(batch)
        ref = reference(batch)
    row = assert_q0_raw_matches_q(packet.q0_raw, ref["q"],
                                 context=f"seed {seed} real checkpoint")
    assert row["bitwise"] is True
    assert row["atol"] == PARITY_ATOL_CPU == 0.0
    assert torch.equal(packet.q0, ref["q"])


@pytest.mark.parametrize("seed", SEEDS)
def test_real_checkpoint_hooks_fire_once(seed: int, payloads) -> None:
    _require_checkpoint(seed)
    if seed not in payloads:  # pragma: no cover
        pytest.skip(f"payload for seed {seed} unavailable")
    encoder, _reference = _pair(seed, payloads[seed])
    modules = (encoder.backbone.message_mlp[-1],
               encoder.backbone.norm_h[-1],
               encoder.process_head.progress_head.fc2)
    assert all(len(m._forward_hooks) == 0 for m in modules)
    batch = forward_view_v3(make_v3_gate_batch(n_graphs=2, jitter_seed=seed))
    with torch.no_grad():
        encoder(batch)
    assert all(len(m._forward_hooks) == 0 for m in modules)
    assert encoder.last_hook_fires == {
        "message_mlp_last": 1, "norm_h_last": 1, "progress_fc2": 1}


def _require_data_view() -> None:
    if not (DATA_VIEW / "data_manifest.json").is_file() or \
            not (DATA_VIEW / "split_manifest_v53mp.json").is_file():
        pytest.skip("frozen 50K data view unavailable in this environment "
                    f"({DATA_VIEW})")


@pytest.mark.parametrize("seed", SEEDS)
def test_real_inner_val_batches_parity(seed: int, payloads) -> None:
    """The FULL S0-A leg: >= 8 REAL inner-val batches, one seed at a time."""
    _require_checkpoint(seed)
    if seed not in payloads:  # pragma: no cover
        pytest.skip(f"payload for seed {seed} unavailable")
    _require_data_view()
    pytest.importorskip("yaml")
    from scripts.agent_ops.v542_a0_parity import (
        DEFAULT_CONFIG,
        MIN_INNER_VAL_BATCHES,
        inner_val_batches,
        load_pinned_assets,
    )

    cfg, _assets = load_pinned_assets(PROJECT_ROOT / DEFAULT_CONFIG)
    batches = inner_val_batches(cfg, seed)
    assert len(batches) >= MIN_INNER_VAL_BATCHES == 8

    encoder, reference = _pair(seed, payloads[seed])
    for index, batch in enumerate(batches):
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
            context=f"seed {seed} inner-val batch {index}",
        )
        for name, row in rows.items():
            assert row["bitwise"] is True, (seed, index, name)
    assert encoder.forward_count == len(batches)
