"""V5.3-MP S0-D sharded lazy dataset tests (plan §4.4).

Synthetic two-curve six-table fixture (torch-free).  Cover the S0-D hard
gates that are testable without real data:

* **shard equivalence** — after :func:`write_geometry_shards`, loading
  through the shard fast path is EXACTLY equivalent to a full CSV load:
  same sample order, same ``frame_xyz`` bytes, same ``v52_labels`` q /
  d_key_ts / stop values;
* **determinism** — re-running the shard writer reproduces byte-
  identical shards (fixed zip timestamps, sorted keys, float64);
* **laziness + LRU bound** — the geometry cache is empty at init and
  never exceeds ``cache_curves`` (eviction verified with
  ``cache_curves=1``);
* **invariants** — curve-weight Σ=1.0 verification and manifest/shard
  sha checks still fail closed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
    SHARD_INDEX_FILENAME,
    SHARD_INDEX_VERSION,
    KeyAtomDatasetV3,
    KeyAtomDatasetV3Error,
    ShardedKeyAtomDatasetV3,
    write_geometry_shards,
)

ALL_STATUSES = ("ok", "terminal", "masked", "conflict")
N_FRAMES = 4
T_VALUES = (0.0, 0.35, 0.7, 1.0)
TS_INDEX = 2
CURVE_IDS = ("curve_000", "curve_001")

ATOMIC_NUMBERS = (6, 35, 8, 1, 1, 1)
G_SRC_BONDS = ((0, 1, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0))
G_TGT_BONDS = ((0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0))
O_DIR = np.array([-0.8, 0.6, 0.0])
H_FIXED = np.array([[0.2, 1.0, 0.2], [0.2, -0.9, 0.5], [-0.3, -0.2, -1.0]])


def _frame(t: float) -> np.ndarray:
    xyz = np.zeros((6, 3))
    xyz[1] = np.array([1.9 + 1.7 * t, 0.0, 0.0])
    xyz[2] = O_DIR * (3.4 - 2.0 * t)
    xyz[3:6] = H_FIXED
    return xyz


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        writer.writerows(rows)


def _write_fixture(tables_dir: Path) -> Path:
    """Two curves × 4 nodes, manifest included, fold in sample_view."""
    out = Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json = json.dumps(list(ATOMIC_NUMBERS))
    curve_rows, node_rows, intent_rows, edge_rows, quality_rows, sample_rows = \
        [], [], [], [], [], []
    for c, cid in enumerate(CURVE_IDS):
        curve_rows.append((cid, f"canon::{cid}", N_FRAMES, 2 * (N_FRAMES - 1),
                           TS_INDEX, 0, N_FRAMES - 1))
        for node, t in enumerate(T_VALUES):
            node_rows.append((
                cid, node, json.dumps(_frame(t).tolist()), 0.0, 0.0,
                -1.0 + 2.0 * node / (N_FRAMES - 1), node == TS_INDEX,
                node in (0, N_FRAMES - 1), atomic_json))
        g_src = json.dumps([list(b) for b in G_SRC_BONDS])
        g_tgt = json.dumps([list(b) for b in G_TGT_BONDS])
        cv_spec = json.dumps({
            "target_deltas": [0.05, -0.05],
            "coordinates": [[0, 1], [0, 2]],
            "kind": "graph_edit_bond_lengths"})
        for direction, iid, rev in (("forward", f"{cid}::fwd", f"{cid}::rev"),
                                    ("reverse", f"{cid}::rev", f"{cid}::fwd")):
            intent_rows.append((
                iid, cid, direction, rev, f"{cid}::pair", "false",
                g_src if direction == "forward" else g_tgt,
                g_tgt if direction == "forward" else g_src,
                "[]", cv_spec, "", "", "v3"))
            edge_rows.append((
                cid, iid, 0, 0, 1, json.dumps([0.0] * 9), "[3,3]", "[0,9)",
                "0" * 64, "1" * 64, "angstrom", "true"))
        quality_rows.append((cid, "ok"))
        cw = 1.0 / (N_FRAMES * 2)
        target_u = json.dumps([0.01, 0.0, 0.0, 0.02, 0.0, 0.0,
                               -0.03, 0.0, 0.0])
        for node in range(N_FRAMES):
            for iid in (f"{cid}::fwd", f"{cid}::rev"):
                sample_rows.append((
                    f"{cid}:{node}:{iid}:0.08", cid, node, iid, 0.08,
                    0.0, 0.0, 0.0, 1.0, cw, 1.0, "ok",
                    "f" * 64, "l" * 64, "false", "local_frame", 2,
                    target_u, 0.0, 0, "false", c))

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
    names = ("irc_curve.csv", "irc_node.csv", "irc_edge.csv",
             "reaction_intent.csv", "curve_quality.csv", "sample_view.csv")
    (out / "data_manifest.json").write_text(json.dumps({"files": [
        {"path": n,
         "sha256": hashlib.sha256((out / n).read_bytes()).hexdigest()}
        for n in names]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


@pytest.fixture()
def tables_dir(tmp_path: Path) -> Path:
    return _write_fixture(tmp_path / "data")


def _refresh_manifest(dir_path: Path) -> None:
    names = ("irc_curve.csv", "irc_node.csv", "irc_edge.csv",
             "reaction_intent.csv", "curve_quality.csv", "sample_view.csv")
    (dir_path / "data_manifest.json").write_text(json.dumps({"files": [
        {"path": n,
         "sha256": hashlib.sha256((dir_path / n).read_bytes()).hexdigest()}
        for n in names]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# shard round-trip: EXACT equivalence with the full CSV load
# ---------------------------------------------------------------------------
def test_shard_roundtrip_equivalence(tables_dir: Path) -> None:
    full = KeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    lazy_csv = ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    assert lazy_csv.uses_shards is False          # CSV fallback before shards

    result = write_geometry_shards(tables_dir)
    assert result["version"] == SHARD_INDEX_VERSION
    assert result["n_curves"] == len(CURVE_IDS)
    assert result["n_shards"] == 1
    index = json.loads(
        (tables_dir / SHARD_INDEX_FILENAME).read_text(encoding="utf-8"))
    assert index["version"] == SHARD_INDEX_VERSION
    for entry in index["shards"]:
        actual = hashlib.sha256(
            (tables_dir / entry["path"]).read_bytes()).hexdigest()
        assert entry["sha256"] == actual
        assert entry["format"] == "npz"

    lazy = ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    assert lazy.uses_shards is True

    # same sample order, same identity fields
    assert [r.sample_id for r in lazy.samples] == \
           [r.sample_id for r in full.samples]
    assert [r.sample_id for r in lazy.records([3, 0, 1])] == \
           [full.samples[i].sample_id for i in (3, 0, 1)]
    for a, b in zip(lazy.samples, full.samples):
        assert (a.curve_id, a.node_index, a.intent.intent_id, a.d0,
                a.label_status, a.label_weight, a.curve_weight,
                a.sample_weight, a.fold, a.target_node_index) == \
               (b.curve_id, b.node_index, b.intent.intent_id, b.d0,
                b.label_status, b.label_weight, b.curve_weight,
                b.sample_weight, b.fold, b.target_node_index)
        assert a.intent == b.intent

    # EXACT frame bytes for every (curve, node)
    for cid in CURVE_IDS:
        for node in range(N_FRAMES):
            ref = np.asarray(full.frame_xyz(cid, node), dtype=np.float64)
            got = np.asarray(lazy.frame_xyz(cid, node), dtype=np.float64)
            assert ref.tobytes() == got.tobytes(), (cid, node)

    # identical v52 label math (q / d_key_ts / stop), float-exact
    v52_full = full.v52_labels()
    v52_lazy = lazy.v52_labels()
    assert lazy.v52_labels() is v52_lazy          # computed once, cached
    assert set(v52_full.by_sample) == set(v52_lazy.by_sample)
    for sid in sorted(v52_full.by_sample):
        a, b = v52_lazy.by_sample[sid], v52_full.by_sample[sid]
        assert a.q == b.q and a.q_mask == b.q_mask
        assert a.d_key_ts == b.d_key_ts and a.stop == b.stop
    assert set(v52_full.curves) == set(v52_lazy.curves)
    for cid in v52_full.curves:
        assert v52_full.curves[cid].arc_s == v52_lazy.curves[cid].arc_s
        assert v52_full.curves[cid].q_canonical == \
            v52_lazy.curves[cid].q_canonical

    # to_sample_v3 assembles the same buildable sample
    for rec_full, rec_lazy in zip(full.samples, lazy.samples):
        s_full = full.to_sample_v3(rec_full)
        s_lazy = lazy.to_sample_v3(rec_lazy)
        assert s_lazy.xyz == s_full.xyz
        assert s_lazy.frame_weight == s_full.frame_weight
        assert s_lazy.sample_id == s_full.sample_id


def test_write_geometry_shards_deterministic(tables_dir: Path) -> None:
    out_a = tables_dir.parent / "shards_a"
    out_b = tables_dir.parent / "shards_b"
    res_a = write_geometry_shards(tables_dir, out_dir=out_a)
    res_b = write_geometry_shards(tables_dir, out_dir=out_b)
    assert [e["sha256"] for e in res_a["shards"]] == \
           [e["sha256"] for e in res_b["shards"]]
    assert (out_a / SHARD_INDEX_FILENAME).read_bytes() == \
           (out_b / SHARD_INDEX_FILENAME).read_bytes()


def test_tampered_shard_fails_closed(tables_dir: Path) -> None:
    write_geometry_shards(tables_dir)
    shard = next(tables_dir.glob("shard_*.npz"))
    shard.write_bytes(b"not a zip file at all")
    lazy = ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    # verification happens on first geometry load, BEFORE any parse
    # (fail-closed on sha; no lazy fallback to CSV after corruption)
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        lazy.frame_xyz(CURVE_IDS[0], 0)
    assert exc.value.code in ("SHARD_SHA", "SHARD_READ")


# ---------------------------------------------------------------------------
# laziness + bounded LRU
# ---------------------------------------------------------------------------
def test_lazy_until_first_frame_access(tables_dir: Path) -> None:
    lazy = ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    assert lazy.n_cached_curves == 0              # no geometry at init
    assert len(lazy) == len(CURVE_IDS) * N_FRAMES * 2
    assert lazy.samples and lazy.intents and lazy.folds is not None
    assert lazy.n_cached_curves == 0              # light index only
    lazy.frame_xyz(CURVE_IDS[0], 0)
    assert lazy.n_cached_curves == 1


def test_lru_bound_respected(tables_dir: Path) -> None:
    lazy = ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES,
                                   cache_curves=1)
    lazy.frame_xyz(CURVE_IDS[0], 0)
    assert lazy.cached_curve_ids == (CURVE_IDS[0],)
    lazy.frame_xyz(CURVE_IDS[1], 0)               # evicts curve_000
    assert lazy.cached_curve_ids == (CURVE_IDS[1],)
    lazy.frame_xyz(CURVE_IDS[0], 1)               # evicts curve_001
    assert lazy.cached_curve_ids == (CURVE_IDS[0],)
    # random access over everything, twice — bound never exceeded
    for _ in range(2):
        for cid in CURVE_IDS:
            for node in range(N_FRAMES):
                lazy.frame_xyz(cid, node)
                assert lazy.n_cached_curves <= 1
    # frames are still byte-correct after all the eviction churn
    full = KeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    for cid in CURVE_IDS:
        for node in range(N_FRAMES):
            ref = np.asarray(full.frame_xyz(cid, node), dtype=np.float64)
            got = np.asarray(lazy.frame_xyz(cid, node), dtype=np.float64)
            assert ref.tobytes() == got.tobytes()


def test_lru_bound_on_shard_path(tables_dir: Path) -> None:
    write_geometry_shards(tables_dir)
    lazy = ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES,
                                   cache_curves=1)
    for cid in reversed(CURVE_IDS):
        for node in range(N_FRAMES):
            lazy.frame_xyz(cid, node)
            assert lazy.n_cached_curves <= 1
    assert lazy.cached_curve_ids == (CURVE_IDS[0],)   # last touched wins


# ---------------------------------------------------------------------------
# invariants still enforced on the light index
# ---------------------------------------------------------------------------
def test_curve_weight_verification_enforced(tables_dir: Path) -> None:
    target = tables_dir / "sample_view.csv"
    rows = list(csv.reader(target.open(encoding="utf-8")))
    header, body = rows[0], rows[1:]
    cw_col = header.index("curve_weight")
    body[0][cw_col] = str(float(body[0][cw_col]) * 2.0)   # break Σ = 1.0
    with target.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([header] + body)
    _refresh_manifest(tables_dir)
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    assert exc.value.code == "CURVE_WEIGHT_SUM"


def test_manifest_sha_still_fail_closed(tables_dir: Path) -> None:
    with (tables_dir / "sample_view.csv").open("a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    assert exc.value.code == "MANIFEST_SHA"


def test_split_manifest_fold_filter(tables_dir: Path) -> None:
    (tables_dir / "split_manifest_r3.json").write_text(
        json.dumps({"curve_folds": {CURVE_IDS[0]: 0, CURVE_IDS[1]: 1}}),
        encoding="utf-8")
    fold0 = ShardedKeyAtomDatasetV3(tables_dir, fold=0,
                                    label_statuses=ALL_STATUSES)
    assert {r.curve_id for r in fold0.samples} == {CURVE_IDS[0]}
    assert all(r.fold == 0 for r in fold0.samples)
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ShardedKeyAtomDatasetV3(tables_dir, fold=2,
                                label_statuses=ALL_STATUSES)
    assert exc.value.code == "NO_SAMPLES"


def test_light_record_exposes_supervision_fields(tables_dir: Path) -> None:
    lazy = ShardedKeyAtomDatasetV3(tables_dir, label_statuses=ALL_STATUSES)
    rec = lazy.samples[0]
    assert rec.target_action_u is not None and len(rec.target_action_u) == 3
    assert rec.target_node_index == 2
    assert rec.atomic_numbers == ATOMIC_NUMBERS
    sample = lazy.to_sample_v3(rec)
    assert len(sample.xyz) == len(ATOMIC_NUMBERS)
    assert sample.y_teacher == rec.target_action_u
    assert sample.frame_weight == rec.label_weight
    assert lazy.curve_ts_index(rec.curve_id) == TS_INDEX
    assert lazy.curve_endpoints(rec.curve_id) == (0, N_FRAMES - 1)
