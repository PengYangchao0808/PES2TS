"""V5.4.3 WP2 — REAL-DATA pair-set source audit (plan §5.6 :431–435).

Marked ``realdata``: ``pytest.ini``'s default ``addopts`` EXCLUDE this file
(``-m "not realdata ..."``), so it only runs when explicitly requested, inside
the GPU host (WSL ``pes2ts-gpu``)::

    pytest -m realdata tests/pesn/test_v543_bridge_pair_audit_realdata.py -q

This is the plan §5.6 pair-source audit, verbatim:

    在 10,000 个采样帧上，比较 Bridge 从 forward edges 重构的 pair 集合与监督
    ``pair_index`` 的集合/计数/端点。期望：集合完全一致（因为二者都来自同一
    intent 的 changed pairs）；若不一致，说明 forward 视图缺信息或重构规则错，
    必须阻断。审计代码只读监督字段用于**比对**，不进入 context、不进入模型；
    审计报告单独存放，SHA 进 run identity。

Discipline:

* the Bridge sees ONLY ``forward_view_v3(batch)`` — the supervision tensor is
  passed SEPARATELY to :func:`~pes2ts_core.pesn.models.v543.latent_bridge.
  audit_pair_sets`, so the model path stays structurally supervision free
  (``build_context`` never receives the raw batch, and the raw batch would be
  REFUSED anyway — see ``test_v543_latent_bridge.py``);
* every audited frame must match, frame by frame.  A mismatch FAILS HARD
  (plan §5.6 :434 "必须阻断") — no tolerance, no averaging away of failures;
* frames are the frozen 50K INNER-VAL positions of the pinned split contract
  (the same ``inner_val_selection`` positions the V542 A0 S0-A leg uses), built
  through the same ``_batch_from_positions`` + ``collate_v3`` path as training,
  so the audited feature rows are the production ones;
* if the inner-val slice cannot supply the requested frame budget the test
  SKIPS with an explicit note instead of pretending the audit was full
  ("全量太慢则……分批/抽样并注明");
* no encoder checkpoint is needed: the audited quantity is a pure function of
  the forward features.  A ``CoarsePacket``-shaped synthetic packet is used to
  additionally prove that ``build_context`` works on REAL forward views.

Assets (frozen data view; the checkpoint is NOT required here)::

    /pes2ts_data/v53mp_50k_data/data_manifest.json
    /pes2ts_data/v53mp_50k_data/teacher_manifest.json
    /pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json

Set ``V543_PAIR_AUDIT_OUT=<path>`` to persist the audit record JSON (the plan
requires the report to be stored separately and its SHA to enter the run
identity)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.models.v543 import (  # noqa: E402
    ENCODER_Z_DIM_V543,
    KEY_S_DIM_V543,
    CoarsePacket,
    LatentBridge,
    audit_pair_sets,
    build_context,
    validate_decoder_context,
)

pytestmark = pytest.mark.realdata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_VIEW = Path("/pes2ts_data/v53mp_50k_data")
#: The plan §5.6 audit size.
TARGET_FRAMES = 10_000
#: The FROZEN lower bound of ``inner_val_selection`` (``MIN_INNER_VAL_BATCHES``
#: in ``scripts/agent_ops/v542_a0_parity.py``): asking for fewer batches is a
#: typed ``V542_PARITY_BATCHES`` refusal, so every budget keeps ``n_batches``
#: at least 8 and only the per-batch width shrinks.
MIN_BATCHES = 8
#: Budget fallbacks ``(n_batches, positions_per_batch)`` — the first one the
#: frozen inner-val slice can satisfy is used (frames = n_batches × positions).
BUDGETS = ((64, 64), (64, 32), (32, 32), (16, 32), (MIN_BATCHES, 32))
SEEDS = (17, 29, 43)


def _require_data_view() -> None:
    for name in ("data_manifest.json", "teacher_manifest.json",
                 "split_manifest_v53mp.json"):
        if not (DATA_VIEW / name).is_file():
            pytest.skip("frozen 50K data view unavailable in this environment "
                        f"({DATA_VIEW / name})")


#: Memo of built frame batches: ``(seed, n_batches, positions_per_batch)`` →
#: raw collated batches.  The budget probe and the audit share the same build.
_BATCH_MEMO: dict[tuple[int, int, int], list[dict]] = {}


def _raw_frame_batches(
        seed: int, n_batches: int, positions_per_batch: int) -> list[dict]:
    """``n_batches`` inner-val batches WITH supervision (audit input only).

    Built through the frozen selection + the production
    ``_batch_from_positions`` path, so the rows are exactly what training sees.
    """
    key = (int(seed), int(n_batches), int(positions_per_batch))
    if key in _BATCH_MEMO:
        return _BATCH_MEMO[key]
    pytest.importorskip("yaml")
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _FeatureLRU,
        _batch_from_positions,
    )
    from scripts.agent_ops.v542_a0_parity import (
        DEFAULT_CONFIG,
        _open_frozen_dataset,
        inner_val_selection,
        load_pinned_assets,
    )

    cfg, _assets = load_pinned_assets(PROJECT_ROOT / DEFAULT_CONFIG)
    selection = inner_val_selection(
        cfg, int(seed),
        n_batches=int(n_batches),
        positions_per_batch=int(positions_per_batch))
    positions = [int(p) for p in selection["positions"]]
    dataset = _open_frozen_dataset(cfg)
    lru = _FeatureLRU(dataset, capacity=len(positions) + 64)
    batches: list[dict] = []
    for start in range(0, len(positions), int(positions_per_batch)):
        chunk = positions[start:start + int(positions_per_batch)]
        batch, _records = _batch_from_positions(dataset, lru, chunk, "cpu")
        batches.append(batch)
    _BATCH_MEMO[key] = batches
    return batches


def _first_available_budget() -> tuple[int, int]:
    """The largest ``(n_batches, positions_per_batch)`` the slice can serve."""
    last_error: str | None = None
    for n_batches, positions_per_batch in BUDGETS:
        try:
            _raw_frame_batches(SEEDS[0], n_batches, positions_per_batch)
        except Exception as error:  # noqa: BLE001 - reported in the skip reason
            last_error = f"{type(error).__name__}: {error}"
            continue
        return int(n_batches), int(positions_per_batch)
    pytest.skip("the frozen inner-val slice cannot serve any audit budget "
                f"{BUDGETS} in this environment (last error: {last_error})")


def _packet_from_view(view: dict, *, seed: int = 17) -> CoarsePacket:
    """A synthetic encoder output whose SHAPES match ``view`` exactly."""
    generator = torch.Generator().manual_seed(seed)
    n_nodes = int(view["pos"].shape[0])
    n_edges = int(view["edge_index"].shape[1])
    n_keys = int(view["key_index"].shape[0])
    n_graphs = int(view["cv_target_delta"].shape[0])
    return CoarsePacket(
        z=torch.randn(n_graphs, ENCODER_Z_DIM_V543, generator=generator),
        edge_msg=torch.zeros(n_edges, 48),
        node_h=torch.randn(n_nodes, KEY_S_DIM_V543, generator=generator),
        u0=0.5 * torch.randn(n_keys, 3, generator=generator),
        q0_raw=torch.randn(n_graphs, generator=generator),
        e0=torch.randn(n_graphs, generator=generator),
    )


@pytest.fixture(scope="module")
def bridge() -> LatentBridge:
    return LatentBridge()


@pytest.fixture(scope="module")
def budget() -> tuple[int, int]:
    _require_data_view()
    return _first_available_budget()


def test_pair_set_source_audit_10000_frames(
        bridge: LatentBridge, budget: tuple[int, int]) -> None:
    """The full §5.6 audit: 10,000 frames, set/count/endpoint equality."""
    n_batches, positions_per_batch = budget
    audited_frames = 0
    audited_batches = 0
    records: list[dict[str, object]] = []
    totals = {
        "n_bridge_pairs": 0,
        "n_supervised_pairs": 0,
        "n_missing_from_bridge": 0,
        "n_extra_in_bridge": 0,
        "n_nonkey_changed_edges": 0,
    }

    for seed in SEEDS:
        if audited_frames >= TARGET_FRAMES:
            break
        for batch in _raw_frame_batches(seed, n_batches, positions_per_batch):
            if audited_frames >= TARGET_FRAMES:
                break
            view = forward_view_v3(batch)      # supervision stripped HERE
            record = audit_pair_sets(
                bridge,
                view,
                supervised_pair_index=batch["pair_index"],   # audit-only
            )
            records.append(record)
            audited_frames += int(len(batch["n_nodes"]))
            audited_batches += 1
            for name in ("n_bridge_pairs", "n_supervised_pairs"):
                totals[name] += int(record[name])
            for name in ("n_missing_from_bridge", "n_extra_in_bridge"):
                totals[name] += int(record[name])
            totals["n_nonkey_changed_edges"] += int(
                record["bridge_stats"]["n_nonkey_changed_edges"])

    assert audited_frames > 0, "the inner-val view produced no frames"
    assert audited_batches == len(records)
    # every individual frame must match — no failure may be averaged away
    assert totals["n_missing_from_bridge"] == 0
    assert totals["n_extra_in_bridge"] == 0
    assert totals["n_bridge_pairs"] == totals["n_supervised_pairs"]
    # a changed pair whose endpoint is not a key would be invisible to the
    # K-pair set; the v3 builder guarantees there are none (key_rows covers
    # every changed-bond atom) — the audit records the count instead of
    # assuming it.
    assert totals["n_nonkey_changed_edges"] == 0

    report = {
        "schema": "pesn_v543_pair_set_audit_report_v1",
        "plan_section": "§5.6 :431-435",
        "bridge_spec_sha256": bridge.bridge_spec_sha256(),
        "target_frames": TARGET_FRAMES,
        "audited_frames": audited_frames,
        "audited_batches": audited_batches,
        "seeds": list(SEEDS),
        "budget": {"n_batches": n_batches,
                   "positions_per_batch": positions_per_batch},
        "totals": totals,
        "bridge_stats_last": records[-1]["bridge_stats"],
        "ok": True,
    }
    out = os.environ.get("V543_PAIR_AUDIT_OUT")
    if out:
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True),
                          encoding="utf-8")

    if audited_frames < TARGET_FRAMES:
        pytest.skip(
            f"the frozen inner-val slice supplied only {audited_frames} of the "
            f"{TARGET_FRAMES} frames the plan §5.6 audit asks for (budget "
            f"{n_batches}×{positions_per_batch} × {len(SEEDS)} seeds); every "
            "audited frame MATCHED — raise BUDGETS to complete the full-size "
            "audit")


def test_real_forward_view_builds_a_context(
        bridge: LatentBridge, budget: tuple[int, int]) -> None:
    """``build_context`` on REAL forward features (packet shapes synthesised).

    The frozen selection refuses ``n_batches < MIN_BATCHES`` (8), so the whole
    budget is used — and it is the SAME memoised build the audit already made,
    so this costs nothing extra.
    """
    n_batches, positions_per_batch = budget
    assert n_batches >= MIN_BATCHES
    batches = _raw_frame_batches(SEEDS[0], n_batches, positions_per_batch)
    assert batches, "no inner-val batch available"
    checked = 0
    for batch in batches:
        view = forward_view_v3(batch)
        packet = _packet_from_view(view)
        context = build_context(packet, view)
        validate_decoder_context(context)
        assert context.schema == "decoder_context_v1"
        assert context.bridge_spec_sha256 == bridge.bridge_spec_sha256()
        assert context.pair_s.shape == (
            int(context.intent_pair_index.shape[1]), 48)
        assert int(context.token_ptr[-1]) == (
            int(context.key_ptr[-1]) + int(context.pair_ptr[-1])
            + int(context.key_ptr.numel()) - 1)
        assert bool(torch.isfinite(context.key_v).all())
        assert bool(torch.isfinite(context.pair_s).all())
        # the reconstructed set equals supervision on this real batch too
        audit_pair_sets(bridge, view,
                        supervised_pair_index=batch["pair_index"])
        checked += 1
    assert checked >= 1
