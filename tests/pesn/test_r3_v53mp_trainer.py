"""V5.3-MP trainer tests (mission: key_atom_v53mp_train_v3; plan §8.2).

CPU-fast gates for the Lane C trainer (MoveProcessContract v1 §4–§9 +
V5.3-MP plan §5–§7):

* arm/λ validation (A move-only, B rank+q, C all three; typed refusals);
* curve-level pool split (deterministic, disjoint) + the CONFIRM firewall
  (folds 3/4 never enter train / inner-val / dev position sets);
* directed-ordering pair sampler: (curve, intent direction) grouping,
  q-gap enforcement, cross-TS pairs allowed, seeded cap, empty ⇒ [2, 0];
* frame_mask: terminal/masked/conflict rows are excluded from L_move
  (l_move on the full batch == loss computed on ok rows only);
* R3 backbone init loads VERBATIM; LEGACY_TS_STOP heads refused;
* smoke run arm A: move-only loss path, process heads bit-unchanged;
* smoke run arm C: q/endpoint present in the forward, positive
  l_rank/l_q/l_endpoint at update 1, grad-monitor rows in
  train_log.jsonl, frozen τ_end block, MOVE-guard block vs r3_eval_ckpt;
* resume: 4-update run + resume to 8 == an uninterrupted 8-update run
  (draws are pure functions of (seed, update), so this is exact).

A tiny synthetic six-table fixture (5 curves x 7 nodes, SN2 chemistry
from ``v3_fixtures``) carries mixed label statuses; endpoint labels are
constructed directly as a self-contained table (Lane B's builder is not
required by the trainer's pinned interface).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
    collate_v3,
    to_torch_batch,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.training import key_atom_v53mp_train_v3 as tr  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import set_seed  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    V53MP_SCHEMA,
    V53MPTrainerConfig,
    V53MPTrainerError,
    sample_order_pairs,
    train_run_v53mp,
    v53mp_pool_split,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    ATOMIC_NUMBERS_SN2,
    BASE_XYZ_SN2,
    make_sn2_intent,
    make_sn2_teacher,
)

N_NODES = 7
#: pool curves (fold 1), dev curves (fold 0) — confirm curve added per test
DEFAULT_FOLDS = {"cv0": 1, "cv1": 1, "cv2": 1, "cv_dev0": 0, "cv_dev1": 0}
POOL_CURVES = ("cv0", "cv1", "cv2")
#: label_status overrides: (curve, node, direction) -> status
STATUS_OVERRIDES = {
    ("cv0", N_NODES - 1, "reverse"): "terminal",
    ("cv1", 2, "forward"): "terminal",
    ("cv1", N_NODES - 2, "reverse"): "masked",
}


# ---------------------------------------------------------------------------
# lane readiness (parallel Lane A APIs; skipped cleanly until they land)
# ---------------------------------------------------------------------------


def _lane_status() -> tuple[bool, str]:
    import importlib

    try:
        heads = importlib.import_module(
            "pes2ts_core.pesn.models.key_atom_heads_v3")
        loss = importlib.import_module(
            "pes2ts_core.pesn.models.key_atom_loss_v53mp_v3")
    except Exception as exc:            # pragma: no cover - lane in flight
        return False, "import failed: {}".format(exc)
    missing = [name for mod, name in (
        (heads, "KeyAtomV53MPModel"), (loss, "key_atom_loss_v53mp"))
        if not hasattr(mod, name)]
    if missing:
        return False, "pinned API missing: {}".format(missing)
    return True, ""


@pytest.fixture()
def lanes_ready():
    ok, why = _lane_status()
    if not ok:
        pytest.skip("pinned Lane A API not landed yet ({})".format(why))


# ---------------------------------------------------------------------------
# synthetic six-table fixture (mirror of tests/pesn fixture patterns)
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _curve_frames(curve_idx: int, n_nodes: int) -> list:
    """Deterministic monotone path: Br approaches C, O approaches C."""
    rng = np.random.default_rng(1234 + int(curve_idx))
    frames = []
    for k in range(n_nodes):
        t = k / max(n_nodes - 1, 1)
        xyz = np.array(BASE_XYZ_SN2, dtype=np.float64)
        xyz[1, 0] -= 1.1 * t          # Br toward C (bond breaking)
        xyz[2, 0] += 1.1 * t           # O toward C (bond forming)
        xyz += rng.normal(scale=0.01, size=xyz.shape)
        frames.append(xyz)
    return frames


def _write_fixture_tables(tables_dir: Path, folds: dict) -> None:
    """Write the PD-1 six-table layout for ``folds`` (multi-curve, appended)."""
    out = Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json = json.dumps(list(ATOMIC_NUMBERS_SN2))
    curve_rows, node_rows, intent_rows, edge_rows, quality_rows, sample_rows = \
        [], [], [], [], [], []
    for curve_idx, (cid, fold) in enumerate(sorted(folds.items())):
        fwd = make_sn2_intent("forward", curve_id=cid)
        rev = fwd.reversed_intent()
        frames = _curve_frames(curve_idx, N_NODES)
        ts_index = N_NODES // 2
        curve_rows.append((cid, "canon::{}".format(cid), N_NODES,
                           2 * 4, ts_index, 0, N_NODES - 1))
        for frame, xyz in enumerate(frames):
            node_rows.append((
                cid, frame, json.dumps(xyz.tolist()), 0.0, 0.0,
                -1.0 + 2.0 * frame / max(N_NODES - 1, 1),
                frame == ts_index, frame in (0, N_NODES - 1), atomic_json))
        for it in (fwd, rev):
            intent_rows.append((
                it.intent_id, cid, it.direction, it.reverse_of_intent_id,
                it.intent_pair_id, "false",
                json.dumps([list(b) for b in it.g_src_bonds]),
                json.dumps([list(b) for b in it.g_tgt_bonds]),
                "[]", json.dumps({"target_deltas": list(it.cv_deltas)}),
                "", "", "v3"))
            for edge_index, (src, dst) in enumerate(
                    ((0, 1), (1, 2), (2, 3), (3, 4))):
                flat = list(np.asarray(
                    make_sn2_teacher(it.direction), dtype=np.float64).ravel())
                packed = hashlib.sha256(
                    b"fixture-atom-map").hexdigest()
                edge_rows.append((
                    cid, it.intent_id, edge_index, src, dst,
                    json.dumps(flat), "[6,3]", f"[0,{len(flat)})",
                    packed, hashlib.sha256(
                        json.dumps(flat).encode()).hexdigest(),
                    "angstrom", "true"))
        quality_rows.append((cid, "ok"))
        n_samples = N_NODES * 2
        cw = 1.0 / n_samples
        for frame in range(N_NODES):
            for it in (fwd, rev):
                status = STATUS_OVERRIDES.get(
                    (cid, frame, it.direction), "ok")
                target = json.dumps(
                    [list(r) for r in make_sn2_teacher(it.direction)])
                sample_rows.append((
                    f"{cid}:{frame}:{it.intent_id}:0.08", cid, frame,
                    it.intent_id, 0.08, 0.0, 0.0, 0.0,
                    1.0, cw, 1.0, status,
                    hashlib.sha256(b"feature").hexdigest(),
                    hashlib.sha256(b"label").hexdigest(),
                    "false", "local_frame", len(it.key_rows()) - 1,
                    target, 0.0, 0, "false", int(fold)))

    _write_csv(out / "irc_curve.csv",
               ("curve_id", "canonical_irc_id", "n_nodes", "n_edges",
                "ts_index", "endpoint_a_index", "endpoint_b_index"),
               curve_rows)
    _write_csv(out / "irc_node.csv",
               ("curve_id", "node_index", "x", "y", "z", "progress",
                "is_ts", "is_endpoint", "atomic_numbers"), node_rows)
    _write_csv(out / "reaction_intent.csv",
               ("intent_id", "curve_id", "direction", "reverse_of_intent_id",
                "intent_pair_id", "intent_underdetermined", "g_src", "g_tgt",
                "delta_b_intent", "cv_spec", "intent_role", "rules",
                "intent_version"), intent_rows)
    _write_csv(out / "irc_edge.csv",
               ("curve_id", "intent_id", "edge_index", "src_node_index",
                "dst_node_index", "delta_xyz_flat", "delta_xyz_shape",
                "delta_xyz_offsets", "atom_map_sha256", "delta_xyz_sha256",
                "units", "same_ref_frame"), edge_rows)
    _write_csv(out / "curve_quality.csv",
               ("curve_id", "quality_status"), quality_rows)
    _write_csv(out / "sample_view.csv",
               ("sample_id", "curve_id", "node_index", "intent_id", "d0",
                "x", "y", "z", "label_weight", "curve_weight",
                "sample_weight", "label_status", "feature_sha", "label_sha",
                "sensitivity_only", "gauge_ref", "target_node_index",
                "target_action_u", "target_progress_q", "stop_label", "mask",
                "fold"), sample_rows)
    table_names = ("irc_curve.csv", "irc_node.csv", "irc_edge.csv",
                   "reaction_intent.csv", "curve_quality.csv",
                   "sample_view.csv")
    (out / "data_manifest.json").write_text(json.dumps(
        {"files": [{"path": n, "sha256": _sha256(out / n)}
                   for n in table_names]},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# self-contained endpoint label table (pinned field surface)
# ---------------------------------------------------------------------------


@dataclass
class _EpRow:
    endpoint_label: int        # +1 / 0 / -1
    d_end_angstrom: float
    bond_edit_complete: bool
    geometry_legal: bool
    ts_row: bool
    start_row: bool
    terminal_row: bool
    degenerate_curve: bool


class _EpTable:
    def __init__(self, by_sample) -> None:
        self.by_sample = by_sample


def _build_endpoint_table(dataset: KeyAtomDatasetV3) -> _EpTable:
    rows = {}
    for rec in dataset.samples:
        node = int(rec.node_index)
        fwd = rec.intent.direction == "forward"
        end_node = N_NODES - 1 if fwd else 0
        start_node = 0 if fwd else N_NODES - 1
        ts_node = dataset.curve_ts_index(rec.curve_id)
        status_ok = str(rec.label_status) == "ok"
        if not status_ok:
            label = -1                       # terminal/masked/conflict ⇒ Mask
        elif node == ts_node:
            label = 0                        # TS always negative (§5.3)
        elif node == end_node:
            label = 1                        # target endpoint for this intent
        elif node == start_node:
            label = 0                        # start row: negative
        elif node == 2 and fwd:
            label = -1                       # ambiguous band ⇒ Mask
        else:
            label = 0                        # mid rows: not yet reached
        rows[rec.sample_id] = _EpRow(
            endpoint_label=label,
            d_end_angstrom=0.15 * abs(node - end_node),
            bond_edit_complete=(label == 1),
            geometry_legal=True,
            ts_row=(node == ts_node),
            start_row=(node == start_node),
            terminal_row=(str(rec.label_status) == "terminal"),
            degenerate_curve=False,
        )
    return _EpTable(rows)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _build_env(tmp_path: Path, folds: dict) -> SimpleNamespace:
    tables = tmp_path / "tables"
    _write_fixture_tables(tables, folds)
    dataset = KeyAtomDatasetV3(
        tables,
        label_statuses=("ok", "terminal", "masked", "conflict"))
    labels = dataset.v52_labels()
    endpoint = _build_endpoint_table(dataset)
    dev_pos = [i for i, r in enumerate(dataset.samples)
               if int(getattr(r, "fold", -1)) == 0]
    pool_pos = [i for i, r in enumerate(dataset.samples)
                if int(getattr(r, "fold", -1)) == 1]
    return SimpleNamespace(
        ds=dataset, labels=labels, endpoint=endpoint,
        dev_pos=dev_pos, pool_pos=pool_pos, folds=dict(folds))


@pytest.fixture()
def env(tmp_path):
    return _build_env(tmp_path, DEFAULT_FOLDS)


def _smoke_cfg(**overrides) -> V53MPTrainerConfig:
    base = dict(
        arm="A", seed=17, backbone_init="scratch",
        freeze_encoder_updates=4, joint_updates=4, smoke_updates=8,
        val_every=4, grad_monitor_every=4, batch_curves=2,
        frames_per_curve=4, device="cpu")
    base.update(overrides)
    return V53MPTrainerConfig(**base)


def _read_log(out: Path) -> list:
    path = out / "train_log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# config validation (arm / λ contracts)
# ---------------------------------------------------------------------------


def test_config_arm_and_lambda_validation() -> None:
    # arm A: pure move
    ok_a = V53MPTrainerConfig(arm="A", seed=17)
    assert ok_a.lambda_rank == 0.0 and ok_a.lambda_q == 0.0
    assert not ok_a.train_rank and not ok_a.train_endpoint
    # arm B: rank + q, NO endpoint
    ok_b = V53MPTrainerConfig(
        arm="B", seed=17, lambda_rank=0.1, lambda_q=0.05)
    assert ok_b.train_rank and ok_b.train_q and not ok_b.train_endpoint
    # arm C: all three
    ok_c = V53MPTrainerConfig(
        arm="C", seed=17, lambda_rank=0.1, lambda_q=0.05,
        lambda_endpoint=0.2)
    assert ok_c.train_endpoint
    # mismatches are TYPED refusals
    with pytest.raises(V53MPTrainerError) as exc:
        V53MPTrainerConfig(arm="A", seed=17, lambda_q=0.05)
    assert exc.value.code == "LAMBDA_ARM_MISMATCH"
    with pytest.raises(V53MPTrainerError) as exc:
        V53MPTrainerConfig(arm="B", seed=17, lambda_rank=0.1)   # q missing
    assert exc.value.code == "LAMBDA_ARM_MISMATCH"
    with pytest.raises(V53MPTrainerError) as exc:
        V53MPTrainerConfig(
            arm="B", seed=17, lambda_rank=0.1, lambda_q=0.05,
            lambda_endpoint=0.1)
    assert exc.value.code == "LAMBDA_ARM_MISMATCH"
    with pytest.raises(V53MPTrainerError) as exc:
        V53MPTrainerConfig(arm="C", seed=17, lambda_rank=0.1,
                           lambda_q=0.05)                       # e missing
    assert exc.value.code == "LAMBDA_ARM_MISMATCH"
    # λ range (0, 0.5] when active
    with pytest.raises(V53MPTrainerError) as exc:
        V53MPTrainerConfig(arm="C", seed=17, lambda_rank=0.1,
                           lambda_q=0.05, lambda_endpoint=0.9)
    assert exc.value.code == "LAMBDA_RANGE"
    with pytest.raises(V53MPTrainerError) as exc:
        V53MPTrainerConfig(arm="C", seed=17, lambda_rank=0.9,
                           lambda_q=0.05, lambda_endpoint=0.1)
    assert exc.value.code == "LAMBDA_RANGE"
    # unknown arm / backbone init
    with pytest.raises(V53MPTrainerError) as exc:
        V53MPTrainerConfig(arm="D", seed=17)
    assert exc.value.code == "ARM"
    with pytest.raises(V53MPTrainerError) as exc:
        V53MPTrainerConfig(arm="A", seed=17, backbone_init="r4")
    assert exc.value.code == "BACKBONE_INIT"
    # arm A effective freeze is vacuous (no active head)
    assert ok_a.effective_freeze_updates == 0
    assert ok_b.effective_freeze_updates == ok_b.freeze_encoder_updates


# ---------------------------------------------------------------------------
# pool split + confirm firewall
# ---------------------------------------------------------------------------


def test_v53mp_pool_split_curve_level_and_deterministic(env) -> None:
    samples = env.ds.samples
    train1, val1 = v53mp_pool_split(
        samples, pool_curve_ids=POOL_CURVES, seed=17,
        inner_val_fraction=0.2)
    train2, val2 = v53mp_pool_split(
        samples, pool_curve_ids=POOL_CURVES, seed=17,
        inner_val_fraction=0.2)
    assert (train1, val1) == (train2, val2)          # seeded, deterministic
    train_curves = {samples[i].curve_id for i in train1}
    val_curves = {samples[i].curve_id for i in val1}
    assert train_curves and val_curves
    assert not (train_curves & val_curves)           # curve-level disjoint
    assert train_curves | val_curves == set(POOL_CURVES)
    assert len(val_curves) == 1                      # 0.2 * 3 -> 1
    # a different seed may reshuffle but never overlaps
    train3, val3 = v53mp_pool_split(
        samples, pool_curve_ids=POOL_CURVES, seed=29,
        inner_val_fraction=0.2)
    assert not ({samples[i].curve_id for i in train3}
                & {samples[i].curve_id for i in val3})


def test_pool_split_refuses_confirm_fold_curve(tmp_path) -> None:
    folds = dict(DEFAULT_FOLDS)
    folds["cv_conf"] = 3
    env_c = _build_env(tmp_path, folds)
    with pytest.raises(V53MPTrainerError) as exc:
        v53mp_pool_split(
            env_c.ds.samples,
            pool_curve_ids=(*POOL_CURVES, "cv_conf"),
            seed=17, inner_val_fraction=0.2)
    assert exc.value.code == "CONFIRM_POSITIONS"


def test_train_run_refuses_confirm_dev_positions(tmp_path) -> None:
    """(no lanes needed: the firewall fires BEFORE any model import)"""
    folds = dict(DEFAULT_FOLDS)
    folds["cv_conf"] = 3
    env_c = _build_env(tmp_path, folds)
    confirm_pos = [i for i, r in enumerate(env_c.ds.samples)
                   if r.curve_id == "cv_conf"]
    cfg = _smoke_cfg(smoke_updates=1)
    with pytest.raises(V53MPTrainerError) as exc:
        train_run_v53mp(
            env_c.ds, env_c.labels, env_c.endpoint, cfg,
            tmp_path / "run_dev", dev_positions=confirm_pos,
            train_positions=env_c.pool_pos, resume=False)
    assert exc.value.code == "CONFIRM_POSITIONS"
    # ... and a confirm curve leaking into the POOL is refused too
    with pytest.raises(V53MPTrainerError) as exc:
        train_run_v53mp(
            env_c.ds, env_c.labels, env_c.endpoint, cfg,
            tmp_path / "run_pool", dev_positions=env_c.dev_pos,
            resume=False)
    assert exc.value.code == "CONFIRM_POSITIONS"


def test_train_run_refuses_curve_incomplete_pool(tmp_path, env) -> None:
    # 2 full curves + a PARTIAL third curve: the curve-level split covers
    # strictly more positions than the given pool -> typed refusal
    part = [i for i, r in enumerate(env.ds.samples)
            if r.curve_id in ("cv0", "cv1")]
    part += [i for i, r in enumerate(env.ds.samples)
             if r.curve_id == "cv2"][:3]
    cfg = _smoke_cfg(smoke_updates=1)
    with pytest.raises(V53MPTrainerError) as exc:
        train_run_v53mp(
            env.ds, env.labels, env.endpoint, cfg,
            tmp_path / "run_incomplete",
            dev_positions=env.dev_pos,
            train_positions=part, resume=False)
    assert exc.value.code == "TRAIN_POOL_NOT_CURVE_COMPLETE"


# ---------------------------------------------------------------------------
# ordering pair sampler
# ---------------------------------------------------------------------------


def _batch_for(env, positions) -> dict:
    built = [build_sample_features_v3(
        env.ds.to_sample_v3(env.ds.samples[p]), level="F2", intent_off=False)
        for p in positions]
    return to_torch_batch(collate_v3(built), device="cpu")


def _positions_for(env, curve: str, direction: str, limit=None) -> list:
    out = [i for i, r in enumerate(env.ds.samples)
           if r.curve_id == curve and r.intent.direction == direction]
    out.sort(key=lambda i: env.ds.samples[i].node_index)
    return out if limit is None else out[:limit]


def test_sample_order_pairs_grouping_gap_and_cap(env) -> None:
    labels = env.labels
    fwd = _positions_for(env, "cv0", "forward")
    batch = _batch_for(env, fwd)
    pairs = sample_order_pairs(
        batch, labels, order_min_gap=0.05, max_pairs=1000,
        rng=np.random.RandomState(0))
    assert pairs.dtype == torch.int64 and pairs.shape[0] == 2
    assert int(pairs.shape[1]) > 0
    sids = [str(s) for s in batch["sample_ids"]]
    cross_ts = False
    for a, b in zip(pairs[0].tolist(), pairs[1].tolist()):
        lab_a, lab_b = labels.by_sample[sids[a]], labels.by_sample[sids[b]]
        assert lab_a.curve_id == lab_b.curve_id        # same curve ONLY
        assert lab_a.direction == lab_b.direction      # same intent ONLY
        assert lab_b.q > lab_a.q + 0.05                # q gap enforced
        if lab_a.q * lab_b.q < 0.0:
            cross_ts = True                            # cross-TS allowed
    assert cross_ts, "cross-TS pairs must be constructed (contract §5.2)"
    # cap respected (seeded subsample)
    capped = sample_order_pairs(
        batch, labels, order_min_gap=0.05, max_pairs=3,
        rng=np.random.RandomState(0))
    assert int(capped.shape[1]) == 3
    # mixing both directions keeps pairs INSIDE one direction
    both = _batch_for(env, fwd + _positions_for(env, "cv0", "reverse"))
    pairs2 = sample_order_pairs(
        both, labels, order_min_gap=0.05, max_pairs=1000,
        rng=np.random.RandomState(1))
    sids2 = [str(s) for s in both["sample_ids"]]
    for a, b in zip(pairs2[0].tolist(), pairs2[1].tolist()):
        assert labels.by_sample[sids2[a]].direction == \
            labels.by_sample[sids2[b]].direction
    # impossible gap => EMPTY [2, 0] (the loss yields exactly 0)
    empty = sample_order_pairs(
        batch, labels, order_min_gap=5.0, max_pairs=1000,
        rng=np.random.RandomState(0))
    assert empty.shape == (2, 0)


# ---------------------------------------------------------------------------
# frame_mask: non-ok rows excluded from L_move
# ---------------------------------------------------------------------------


def _move_args(batch: dict, u: torch.Tensor, frame_mask=None) -> dict:
    from pes2ts_core.pesn.models.key_atom_loss_v53mp_v3 import (
        key_atom_loss_v53mp,
    )
    from pes2ts_core.pesn.models.key_atom_losses_v3 import (
        KeyAtomLossV3Config,
    )

    n_keys = batch["n_keys"].long()
    key_ptr = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(n_keys, dim=0)])
    return key_atom_loss_v53mp(
        u, batch["y_teacher"],
        batch["degK"][batch["key_index"]],
        key_ptr, batch["pos"], batch["key_index"], batch["pair_index"],
        frame_mask=frame_mask, frame_weight=batch["frame_weight"],
        lambda_cos=0.02, config=KeyAtomLossV3Config())


def test_frame_mask_excludes_non_ok_rows_from_move(env, lanes_ready) -> None:
    # graphs: 0 = ok, 1 = terminal, 2 = ok (cv1 forward: node0 / node2 / node3)
    positions = _positions_for(env, "cv1", "forward")[:4]
    recs = env.ds.records(positions)
    statuses = [r.label_status for r in recs]
    assert "terminal" in statuses, "fixture must contain a terminal row"
    keep = [i for i, r in enumerate(recs) if r.label_status == "ok"]
    assert 0 in keep and len(keep) >= 2
    full = _batch_for(env, positions)
    ok_pos = [positions[i] for i in keep]
    ok = _batch_for(env, ok_pos)

    # deterministic non-degenerate errors, DIFFERENT per graph
    n_keys = full["n_keys"].long()
    ptr = torch.cat([torch.zeros(1, dtype=torch.long),
                     torch.cumsum(n_keys, dim=0)])
    delta = torch.zeros_like(full["y_teacher"])
    for g in range(len(n_keys)):
        delta[ptr[g]:ptr[g + 1]] = 0.05 + 0.01 * g
    u_full = full["y_teacher"] + delta
    frame_mask = tr._frame_mask_for(recs, "cpu")
    assert frame_mask.tolist() == [r.label_status == "ok" for r in recs]

    out_full = _move_args(full, u_full, frame_mask=frame_mask)
    u_ok = torch.cat([u_full[ptr[g]:ptr[g + 1]] for g in keep])
    out_ok = _move_args(ok, u_ok)
    assert float(out_full["l_move"]) > 1e-4          # non-degenerate
    assert float(out_full["l_move"]) == pytest.approx(
        float(out_ok["l_move"]), rel=1e-6, abs=1e-9)
    assert float(out_full["loss"]) == float(out_full["l_move"])  # λ = 0


# ---------------------------------------------------------------------------
# backbone init / legacy refusal
# ---------------------------------------------------------------------------


def test_r3_backbone_verbatim_and_legacy_refusal(tmp_path, lanes_ready) -> None:
    set_seed(5)
    r3 = StarEGNNV3()
    ckpt = tmp_path / "r3.pt"
    torch.save({"model": r3.state_dict()}, str(ckpt))
    cfg = V53MPTrainerConfig(arm="A", seed=17, backbone_init="r3")
    model = tr.load_model_v53mp(cfg, r3_ckpt=ckpt, device="cpu")
    back = {k[len("backbone."):]: v for k, v in model.state_dict().items()
            if k.startswith("backbone.")}
    assert set(back) == set(r3.state_dict())
    for key, tensor in r3.state_dict().items():
        assert torch.equal(back[key], tensor), key     # VERBATIM load
    counts = model.head_parameter_counts()
    assert counts["backbone"] == 55441
    assert counts["process_head"] > 0
    tr.assert_no_legacy_head(model)                   # V5.3 model is clean
    # LEGACY_TS_STOP (legacy_frozen=True) is refused, typed
    from pes2ts_core.pesn.models.key_atom_heads_v3 import TSStopHeadV3

    holder = torch.nn.Module()
    holder.legacy_head = TSStopHeadV3()
    with pytest.raises(V53MPTrainerError) as exc:
        tr.assert_no_legacy_head(holder)
    assert exc.value.code == "LEGACY_HEAD"
    # backbone_init="r3" without a checkpoint is a typed refusal
    with pytest.raises(V53MPTrainerError) as exc:
        tr.load_model_v53mp(cfg, r3_ckpt=None, device="cpu")
    assert exc.value.code == "R3_CHECKPOINT_MISSING"


# ---------------------------------------------------------------------------
# smoke run: arm A (move-only)
# ---------------------------------------------------------------------------


def test_smoke_arm_a_move_only(tmp_path, env, lanes_ready) -> None:
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    out = tmp_path / "run_a"
    cfg = _smoke_cfg(arm="A")
    metrics = train_run_v53mp(
        env.ds, env.labels, env.endpoint, cfg, out,
        dev_positions=env.dev_pos, resume=False)
    assert metrics["schema"] == V53MP_SCHEMA
    assert metrics["arm"] == "A"
    assert metrics["run_outcome"] in ("ok", "scientific_failure")
    assert (out / "checkpoint_best.pt").is_file()
    assert (out / "checkpoint_last.pt").is_file()
    assert (out / "metrics.json").is_file()
    assert metrics["checkpoint_best_sha256"] and metrics["seeds"]
    assert metrics["head_params"]["process_head"] > 0
    # no guard baseline given ⇒ guard block present, baseline metrics None
    assert metrics["move_guard_vs_r3"].get("rmse_degradation_pct") is None
    # move-only loss path: every process term is EXACTLY 0 in the log
    train_lines = [e for e in _read_log(out) if "loss" in e]
    assert train_lines
    for line in train_lines:
        assert line["l_rank"] == 0.0
        assert line["l_q"] == 0.0
        assert line["l_endpoint"] == 0.0
    # process heads never trained (bitwise unchanged from seed-exact init)
    set_seed(cfg.seed)
    ref = KeyAtomV53MPModel()
    state = torch.load(str(out / "checkpoint_best.pt"), map_location="cpu",
                       weights_only=False)["model"]
    ref_state = ref.state_dict()
    for key, tensor in ref_state.items():
        if key.startswith("backbone."):
            continue
        assert key in state and torch.equal(state[key], tensor), key
    # ... while the shared encoder DID train (this is a real run)
    assert any(not torch.equal(state[key], tensor)
               for key, tensor in ref_state.items()
               if key.startswith("backbone."))


# ---------------------------------------------------------------------------
# smoke run: arm C (full PROCESS)
# ---------------------------------------------------------------------------


def test_smoke_arm_c_process_and_metrics(tmp_path, env, lanes_ready) -> None:
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    out = tmp_path / "run_c"
    cfg = _smoke_cfg(
        arm="C", lambda_rank=0.1, lambda_q=0.05, lambda_endpoint=0.1,
        freeze_encoder_updates=2, joint_updates=4, smoke_updates=6,
        val_every=3, grad_monitor_every=2)
    # forward contract: q / endpoint_logit present and per-graph
    set_seed(29)
    probe = KeyAtomV53MPModel()
    from scripts.agent_ops.feature_gate.v3_fixtures import make_v3_batch

    batch = make_v3_batch(n_graphs=2)
    fwd = probe(forward_view_v3(batch))
    assert "q" in fwd and "endpoint_logit" in fwd and "order_score" in fwd
    assert fwd["endpoint_logit"].shape == (int(batch["n_keys"].shape[0]),)
    assert fwd["q"].shape == (int(batch["n_keys"].shape[0]),)

    r3_eval = tmp_path / "r3_eval.pt"
    torch.save({"model": StarEGNNV3().state_dict()}, str(r3_eval))
    metrics = train_run_v53mp(
        env.ds, env.labels, env.endpoint, cfg, out,
        dev_positions=env.dev_pos, r3_eval_ckpt=str(r3_eval), resume=False)
    # positive process terms at update 1 (fixture guarantees pairs + rows)
    line1 = next(e for e in _read_log(out)
                 if e.get("update") == 1 and "loss" in e)
    assert line1["l_rank"] > 0.0
    assert line1["l_q"] > 0.0
    assert line1["l_endpoint"] > 0.0
    assert line1["n_rank_pairs"] > 0
    assert line1["n_endpoint_rows"] > 0
    # gradient monitoring rows appear in train_log.jsonl
    grad_rows = [e for e in _read_log(out) if "grad_monitor" in e]
    assert grad_rows, "grad_monitor_every=2 must emit monitor rows"
    # the first rows fall in the freeze stage (encoder frozen -> flagged);
    # joint-phase rows must carry the actual encoder grad diagnostics
    joint_rows = [r["grad_monitor"] for r in grad_rows
                  if not r["grad_monitor"]["encoder_frozen"]]
    assert joint_rows, "joint-phase monitor rows must report grad norms"
    row = joint_rows[0]
    assert "move" in row["grad_norms"] and "rank" in row["grad_norms"]
    assert "q" in row["grad_norms"] and "endpoint" in row["grad_norms"]
    assert row["encoder_share"] is not None and row["encoder_share"] >= 0.0
    # update 2 sits in the freeze stage: recorded as encoder_frozen (no cost)
    assert any(r["grad_monitor"]["encoder_frozen"] for r in grad_rows)
    # τ_end block frozen in metrics, 0 <= tau <= 1
    tau = metrics["tau_end_calibration"]
    assert tau is not None and tau["n_inner_val_unmasked"] > 0
    assert tau["tau_end"] is not None
    assert 0.0 <= float(tau["tau_end"]) <= 1.0
    assert tau["target_precision"] == cfg.tau_target_precision
    assert tau["achieved_precision"] is not None
    # checkpoint payload carries the frozen τ_end too
    best = torch.load(str(out / "checkpoint_best.pt"), map_location="cpu",
                      weights_only=False)
    assert best["tau_end_calibration"]["tau_end"] == tau["tau_end"]
    # MOVE guard block present (r3_eval_ckpt given)
    guard = metrics["move_guard_vs_r3"]
    assert guard["rmse_degradation_pct"] is not None
    assert guard["within_rmse_guard"] in (True, False, None)
    assert metrics["run_outcome"] in ("ok", "scientific_failure")
    # contract §8: reported separately, no composite score
    assert "q_head" in metrics and "ordering" in metrics
    assert "endpoint_head" in metrics and "move" in metrics
    assert "composite" not in json.dumps(metrics)
    assert metrics["endpoint_head"]["auroc"] is None or \
        0.0 <= metrics["endpoint_head"]["auroc"] <= 1.0
    assert metrics["endpoint_head"]["auprc"] is None or \
        0.0 <= metrics["endpoint_head"]["auprc"] <= 1.0
    assert metrics["ordering"]["n_pairs_full"] >= 0
    # endpoint pos_weight estimated from inner-train, recorded with counts
    epw = metrics["endpoint_pos_weight"]
    assert epw["n_pos"] > 0 and "inner-train" in epw["source"]


# ---------------------------------------------------------------------------
# resume: 4 + resume-to-8 == fresh 8 (pure per-update draws)
# ---------------------------------------------------------------------------


def _loss_by_update(out: Path) -> dict:
    return {e["update"]: e["loss"] for e in _read_log(out)
            if "loss" in e and "update" in e}


def _val_by_update(out: Path) -> dict:
    return {e["update"]: e["inner_val_loss"] for e in _read_log(out)
            if "inner_val_loss" in e and "update" in e}


def test_resume_identical_to_fresh_run(tmp_path, env, lanes_ready) -> None:
    common = dict(
        arm="B", seed=17, backbone_init="scratch",
        lambda_rank=0.1, lambda_q=0.05, lambda_endpoint=0.0,
        freeze_encoder_updates=2, val_every=1, batch_curves=2,
        frames_per_curve=4, device="cpu")
    dir_resumed = tmp_path / "resumed"
    dir_fresh = tmp_path / "fresh"
    # leg 1: 4 updates
    cfg4 = V53MPTrainerConfig(
        **{**common, "joint_updates": 2, "smoke_updates": 4})
    run1 = train_run_v53mp(
        env.ds, env.labels, env.endpoint, cfg4, dir_resumed,
        dev_positions=env.dev_pos, resume=False)
    assert run1["selected_update"] in (1, 2, 3, 4)
    # leg 2: SAME out_dir, resume to 8
    cfg8 = V53MPTrainerConfig(
        **{**common, "joint_updates": 6, "smoke_updates": 8})
    resumed = train_run_v53mp(
        env.ds, env.labels, env.endpoint, cfg8, dir_resumed,
        dev_positions=env.dev_pos, resume=True)
    # leg 3: uninterrupted 8-update run, same seed
    fresh = train_run_v53mp(
        env.ds, env.labels, env.endpoint, cfg8, dir_fresh,
        dev_positions=env.dev_pos, resume=False)

    assert resumed["last_train_loss"] == fresh["last_train_loss"]
    assert resumed["best_inner_val_loss"] == fresh["best_inner_val_loss"]
    assert resumed["selected_update"] == fresh["selected_update"]
    assert resumed["best_update"] == fresh["best_update"]
    # inner-val loss recorded at EVERY update in both runs -> exact match
    val_r, val_f = _val_by_update(dir_resumed), _val_by_update(dir_fresh)
    assert set(val_r) == set(val_f) == set(range(1, 9))
    assert val_r == val_f
    # train-loss lines (logged at update==1 and run end) match exactly;
    # the resumed dir additionally carries leg-1's end-of-run line at 4
    loss_r, loss_f = _loss_by_update(dir_resumed), _loss_by_update(dir_fresh)
    assert set(loss_r) == {1, 4, 8} and set(loss_f) == {1, 8}
    assert loss_r[1] == loss_f[1] and loss_r[8] == loss_f[8]
    # final model weights are bitwise identical
    state_r = torch.load(str(dir_resumed / "checkpoint_best.pt"),
                         map_location="cpu", weights_only=False)["model"]
    state_f = torch.load(str(dir_fresh / "checkpoint_best.pt"),
                         map_location="cpu", weights_only=False)["model"]
    assert set(state_r) == set(state_f)
    for key in state_r:
        assert torch.equal(state_r[key], state_f[key]), key
    # resume identity refusal: a different arm cannot resume into the dir
    cfg_c = V53MPTrainerConfig(
        arm="C", seed=17, backbone_init="scratch",
        lambda_rank=0.1, lambda_q=0.05, lambda_endpoint=0.1,
        freeze_encoder_updates=2, joint_updates=6, smoke_updates=8,
        val_every=1, batch_curves=2, frames_per_curve=4, device="cpu")
    with pytest.raises(V53MPTrainerError) as exc:
        train_run_v53mp(
            env.ds, env.labels, env.endpoint, cfg_c, dir_resumed,
            dev_positions=env.dev_pos, resume=True)
    assert exc.value.code == "RESUME_IDENTITY_MISMATCH"


def test_feature_lru_zero_pads_missing_teacher(env, lanes_ready) -> None:
    """Real-data regression: terminal/masked rows carry y_teacher=None
    (unlike this fixture's always-teacher rows); the trainer's feature LRU
    must zero-pad so mixed batches never lose the key."""
    import dataclasses

    base = env.ds.samples[0]
    rec_none = dataclasses.replace(base, target_action_u=None)

    class _FakeDS:
        def __init__(self, ds, rec):
            self._ds, self._rec = ds, rec

        def records(self, _idx):
            return [self._rec]

        def to_sample_v3(self, rec):
            return self._ds.to_sample_v3(rec)

    lru = tr._FeatureLRU(_FakeDS(env.ds, rec_none), capacity=4)
    built = lru.get(0)
    n_key = len(base.intent.key_rows())
    assert built["y_teacher"] is not None
    assert built["y_teacher"].shape == (n_key, 3)
    assert float(np.abs(built["y_teacher"]).max()) == 0.0
    # collate keeps y_teacher when mixed with a teacher-bearing sample
    ok_built = build_sample_features_v3(
        env.ds.to_sample_v3(env.ds.samples[1]), level="F2", intent_off=False)
    batch = to_torch_batch(collate_v3([built, ok_built]), device="cpu")
    assert "y_teacher" in batch
    assert batch["y_teacher"].shape[0] == n_key * 2


# ---------------------------------------------------------------------------
# CUDA memory suite: fraction validation, application, metrics keys
# (CPU-only — torch.cuda entry points are monkeypatched; no GPU required)
# ---------------------------------------------------------------------------


def test_config_cuda_memory_fraction_validation() -> None:
    # default: None (zero behaviour change)
    assert V53MPTrainerConfig(arm="A", seed=17).cuda_memory_fraction is None
    # 0 < f <= 1 accepted
    ok = V53MPTrainerConfig(arm="A", seed=17, cuda_memory_fraction=0.17)
    assert ok.cuda_memory_fraction == 0.17
    assert V53MPTrainerConfig(
        arm="A", seed=17, cuda_memory_fraction=1.0).cuda_memory_fraction == 1.0
    # boundary / out-of-range values are TYPED refusals
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(V53MPTrainerError) as exc:
            V53MPTrainerConfig(
                arm="A", seed=17, cuda_memory_fraction=bad)
        assert exc.value.code == "CUDA_MEMORY_FRACTION", bad


def test_apply_cuda_memory_fraction_calls_set_fraction(monkeypatch) -> None:
    calls = []

    def _record(name):
        def _fn(*args, **kwargs):
            calls.append((name, args, kwargs))
        return _fn

    monkeypatch.setattr(tr.torch.cuda, "init", _record("init"))
    monkeypatch.setattr(
        tr.torch.cuda, "set_per_process_memory_fraction",
        _record("set_per_process_memory_fraction"))

    # default None: the fraction setter is NEVER called (pre-cap behaviour)
    cfg_none = V53MPTrainerConfig(arm="A", seed=17, device="cuda")
    assert tr.apply_cuda_memory_fraction(cfg_none) is None
    assert calls == []

    # fraction 0.17 on bare "cuda": init + setter with device index 0
    cfg = V53MPTrainerConfig(
        arm="A", seed=17, device="cuda", cuda_memory_fraction=0.17)
    assert tr.apply_cuda_memory_fraction(cfg) == 0.17
    assert [c[0] for c in calls] == [
        "init", "set_per_process_memory_fraction"]
    frac_call = calls[1]
    assert frac_call[1] == (0.17, 0)       # (fraction, device_index=0)

    # explicit device index is passed through verbatim
    calls.clear()
    cfg1 = V53MPTrainerConfig(
        arm="A", seed=17, device="cuda:1", cuda_memory_fraction=0.17)
    assert tr.apply_cuda_memory_fraction(cfg1) == 0.17
    assert calls[1][1] == (0.17, 1)        # device index 1

    # CPU device: NEVER touches torch.cuda even with a fraction set
    calls.clear()
    cfg_cpu = V53MPTrainerConfig(
        arm="A", seed=17, device="cpu", cuda_memory_fraction=0.17)
    assert tr.apply_cuda_memory_fraction(cfg_cpu) is None
    assert calls == []


def test_reset_peak_memory_stats_noop_on_cpu(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        tr.torch.cuda, "reset_peak_memory_stats",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    tr._reset_peak_memory_stats("cpu")          # explicit no-op, no crash
    assert calls == []
    tr._reset_peak_memory_stats("cuda:2")       # forwarded verbatim on CUDA
    assert calls == [(("cuda:2",), {})]


def test_gpu_memory_metrics_keys_int_or_none_on_cpu() -> None:
    # CPU device: 0 bytes / None fraction (explicit contract, never crashes)
    cfg = V53MPTrainerConfig(arm="A", seed=17, device="cpu")
    mem = tr._gpu_memory_metrics(cfg)
    assert set(mem) == {
        "gpu_max_memory_allocated_bytes",
        "gpu_max_memory_reserved_bytes",
        "gpu_memory_fraction_applied",
    }
    assert mem["gpu_max_memory_allocated_bytes"] == 0
    assert mem["gpu_max_memory_reserved_bytes"] == 0
    assert mem["gpu_memory_fraction_applied"] is None
    assert isinstance(mem["gpu_max_memory_allocated_bytes"], int)
    assert isinstance(mem["gpu_max_memory_reserved_bytes"], int)
    # a fraction on a CPU device is never "applied"
    cfg_frac = V53MPTrainerConfig(
        arm="A", seed=17, device="cpu", cuda_memory_fraction=0.17)
    assert tr._gpu_memory_metrics(cfg_frac)["gpu_memory_fraction_applied"] is None


def test_smoke_metrics_carry_gpu_memory_keys(tmp_path, env, lanes_ready) -> None:
    out = tmp_path / "run_gpu_mem"
    cfg = _smoke_cfg(arm="A", smoke_updates=1, val_every=1)
    metrics = train_run_v53mp(
        env.ds, env.labels, env.endpoint, cfg, out,
        dev_positions=env.dev_pos, resume=False)
    # top-level keys present with the CPU contract values (int/None)
    assert isinstance(metrics["gpu_max_memory_allocated_bytes"], int)
    assert isinstance(metrics["gpu_max_memory_reserved_bytes"], int)
    assert metrics["gpu_max_memory_allocated_bytes"] == 0
    assert metrics["gpu_max_memory_reserved_bytes"] == 0
    assert metrics["gpu_memory_fraction_applied"] is None
    # persisted metrics.json carries the same top-level keys
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert payload["gpu_max_memory_allocated_bytes"] == 0
    assert payload["gpu_max_memory_reserved_bytes"] == 0
    assert payload["gpu_memory_fraction_applied"] is None
