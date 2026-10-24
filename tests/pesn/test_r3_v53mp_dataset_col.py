"""Loader v2 (columnar + memmap) tests — plan §4 / S4 lane I1.

Fixture strategy: reuse the two-curve six-table builder from
``tests/pesn/test_r3_v53mp_dataset`` and layer the v2 artifacts on top
(geometry shards → v1 light cache → label cache → ``idx_v2`` build), so the
mini-tables exercise the SAME pipeline the real 20k tables use.

Covered hard gates:

* **builder→reader roundtrip** — every ``SampleRecordV3Lite`` field and
  every v52/endpoint label field of the columnar reader equals the v1
  reference (exact for identity/intent fields; ``float32``-cast for the
  plan-§4.1 float32-stored groups: weights + ``target_action_u``);
* **lazy single-pass** — record construction happens ONLY on access (a
  monkeypatched construction counter pins that ``len()`` / property reads
  construct nothing, ``samples[i]`` constructs exactly one, iteration does
  NOT accumulate across passes);
* **binding fail-closed** — data-manifest drift, missing idx files and
  fold/status/direction mismatches are typed refusals, never silent serves;
* **records()/geometry/features equivalence vs v1** on the same fixture;
* **label lazy-view attribute parity** — by_sample/curves/window/count
  surface mirrors the eager ``load_label_cache`` tables.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3Error,
    ShardedKeyAtomDatasetV3,
    write_geometry_shards,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    IDX2_SCHEMA,
    ColumnarKeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    build_endpoint_labels_spec_v1,
)
from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (  # noqa: E402
    save_label_cache,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
    collate_v3,
)
from tests.pesn.test_r3_v53mp_dataset import (  # noqa: E402
    ALL_STATUSES,
    CURVE_IDS,
    N_FRAMES,
    _write_fixture,
)

_BUILDER_PATH = (REPO / "scripts" / "agent_ops" / "data_gate" /
                 "r3_v53mp_build_idx_v2.py")


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "r3_v53mp_build_idx_v2_under_test", _BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_v2_stack(tables_dir: Path) -> tuple[ShardedKeyAtomDatasetV3, dict]:
    """shards → v1 light cache → label cache → idx_v2; returns (v1, summary)."""
    write_geometry_shards(tables_dir)
    v1 = ShardedKeyAtomDatasetV3(
        tables_dir, label_statuses=ALL_STATUSES, verify_manifest=False,
        light_cache=True)
    labels = v1.v52_labels()
    endpoint = build_endpoint_labels_spec_v1(v1, labels)
    save_label_cache(tables_dir, labels, endpoint,
                     dataset_manifest_sha256=v1.manifest_sha256)
    builder = _load_builder()
    summary = builder.build_idx_v2(
        tables_dir, tables_dir / "idx_v2", verify_sources=False)
    return v1, summary


@pytest.fixture()
def v2_stack(tmp_path: Path):
    tables = _write_fixture(tmp_path / "data")
    v1, summary = _build_v2_stack(tables)
    v2 = ColumnarKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    # eager reference tables straight from the cache loader path
    from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (
        load_label_cache,
    )
    labels_ref, ep_ref = load_label_cache(
        tables, dataset_manifest_sha256=v1.manifest_sha256)
    labels_v2, ep_v2 = v2.label_tables()
    return {
        "tables": tables, "v1": v1, "v2": v2, "summary": summary,
        "labels_ref": labels_ref, "ep_ref": ep_ref,
        "labels_v2": labels_v2, "ep_v2": ep_v2,
    }


# ---------------------------------------------------------------------------
# builder → reader roundtrip
# ---------------------------------------------------------------------------
def test_builder_summary_schema(v2_stack) -> None:
    summary = v2_stack["summary"]
    assert summary["schema"] == IDX2_SCHEMA
    assert summary["n_samples"] == len(v2_stack["v1"])
    assert summary["n_curves"] == len(CURVE_IDS)
    assert summary["core_index_bytes_per_sample"] == 43  # 44B plan target −1
    curves_json = json.loads(
        (v2_stack["tables"] / "idx_v2" / "curves.json").read_text("utf-8"))
    assert curves_json["schema"] == IDX2_SCHEMA
    binding = curves_json["binding"]
    assert binding["n_samples"] == summary["n_samples"]
    assert binding["fold"] is None
    manifest_sha = hashlib.sha256(
        (v2_stack["tables"] / "data_manifest.json").read_bytes()).hexdigest()
    assert binding["data_manifest_sha256"] == manifest_sha
    # source identity is bound (size/mtime always; sha256 when the builder
    # ran with source verification — the fixture skips it for speed)
    entry = binding["source_light_pickle"]
    assert entry["size"] > 0 and entry["mtime_ns"] > 0
    assert entry["sha256"] in ("",) or len(entry["sha256"]) == 64
    assert len(binding["source_labels_npz"]["path"]) > 0


def test_roundtrip_sample_fields(v2_stack) -> None:
    v1, v2 = v2_stack["v1"], v2_stack["v2"]
    assert len(v1) == len(v2)
    assert len(v2.records()) == len(v1.samples)
    for a, c in zip(v1.samples, v2.samples):
        assert a.sample_id == c.sample_id
        assert a.curve_id == c.curve_id
        assert a.node_index == c.node_index
        assert a.intent == c.intent           # full IntentRecordV3 equality
        assert a.d0 == c.d0
        assert a.label_status == c.label_status
        assert a.atomic_numbers == c.atomic_numbers
        assert a.target_node_index == c.target_node_index
        assert a.fold == c.fold
        # plan-§4.1 float32-stored groups: equality at consumed precision
        for field in ("label_weight", "curve_weight", "sample_weight"):
            assert np.float32(getattr(a, field)) == np.float32(getattr(c, field))
        ta, tc = a.target_action_u, c.target_action_u
        assert (ta is None) == (tc is None)
        if ta is not None:
            assert np.array_equal(np.asarray(ta, dtype=np.float32),
                                  np.asarray(tc, dtype=np.float32))


def test_roundtrip_label_fields(v2_stack) -> None:
    labels_ref, labels_v2 = v2_stack["labels_ref"], v2_stack["labels_v2"]
    ep_ref, ep_v2 = v2_stack["ep_ref"], v2_stack["ep_v2"]
    for rec in v2_stack["v2"].samples:
        sid = rec.sample_id
        assert labels_v2.by_sample[sid] == labels_ref.by_sample[sid]
        assert ep_v2.by_sample[sid] == ep_ref.by_sample[sid]
    for cid in CURVE_IDS:
        assert labels_v2.curves[cid] == labels_ref.curves[cid]
    assert ep_v2.positive_window == ep_ref.positive_window
    assert ep_v2.negative_window == ep_ref.negative_window
    assert len(labels_v2.by_sample) == len(labels_ref.by_sample)
    assert len(ep_v2.by_sample) == len(ep_ref.by_sample)
    assert set(labels_v2.zero_arc_curves) == set(labels_ref.zero_arc_curves)
    assert (set(labels_v2.q_masked_sample_ids())
            == set(labels_ref.q_masked_sample_ids()))
    assert ep_v2.n_rows == ep_ref.n_rows
    assert ep_v2.count_label(1) == ep_ref.count_label(1)


# ---------------------------------------------------------------------------
# lazy sequence: single-pass, no accumulation, bounded construction
# ---------------------------------------------------------------------------
def test_lazy_sequence_no_bulk_materialization(v2_stack) -> None:
    v2 = v2_stack["v2"]
    counter = {"n": 0}
    original = v2._record_at

    def counting(row: int):
        counter["n"] += 1
        return original(row)

    v2._record_at = counting          # instance-level monkeypatch
    try:
        n = len(v2.samples)
        assert counter["n"] == 0, "len() must not construct records"
        _ = v2.samples
        assert counter["n"] == 0, "property access must not construct records"
        rec = v2.samples[3]
        assert counter["n"] == 1, "index access constructs exactly one record"
        assert rec.sample_id
        for _ in v2.samples:
            pass
        assert counter["n"] == 1 + n, "one full pass = n constructions"
        for _ in v2.samples:
            pass
        assert counter["n"] == 1 + 2 * n, \
            "a second pass must not reuse nor accumulate cached records"
        assert v2.records([0, 1]) and counter["n"] == 3 + 2 * n
    finally:
        v2._record_at = original


def test_lazy_sequence_iteration_stable(v2_stack) -> None:
    v2 = v2_stack["v2"]
    first = [r.sample_id for r in v2.samples]
    second = [r.sample_id for r in v2.samples]
    assert first == second
    assert v2.samples[0].sample_id == first[0]
    assert v2.samples[-1].sample_id == first[-1]
    with pytest.raises(IndexError):
        _ = v2.samples[len(v2)]


# ---------------------------------------------------------------------------
# binding / integrity fail-closed
# ---------------------------------------------------------------------------
def test_binding_manifest_drift_refused(v2_stack) -> None:
    tables = v2_stack["tables"]
    manifest = tables / "data_manifest.json"
    manifest.write_text(manifest.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ColumnarKeyAtomDatasetV3(
            tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    assert exc.value.code == "IDX2_BINDING"


def test_binding_missing_idx_refused(v2_stack) -> None:
    tables = v2_stack["tables"]
    shutil.rmtree(tables / "idx_v2" / "samples")
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ColumnarKeyAtomDatasetV3(
            tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    assert exc.value.code in ("IDX2_MISSING", "IDX2_SHA")


def test_idx_tamper_sha_refused(v2_stack) -> None:
    tables = v2_stack["tables"]
    target = tables / "idx_v2" / "samples" / "curve_idx.npy"
    payload = bytearray(target.read_bytes())
    payload[-8] ^= 0xFF
    target.write_bytes(bytes(payload))
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ColumnarKeyAtomDatasetV3(
            tables, label_statuses=ALL_STATUSES, verify_manifest=True)
    assert exc.value.code in ("IDX2_SHA", "IDX2_SHAPE")


def test_binding_filter_mismatch_refused(v2_stack) -> None:
    tables = v2_stack["tables"]
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ColumnarKeyAtomDatasetV3(
            tables, label_statuses=("ok",), verify_manifest=False)
    assert exc.value.code == "IDX2_BINDING"
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ColumnarKeyAtomDatasetV3(
            tables, label_statuses=ALL_STATUSES, verify_manifest=False,
            fold=0)
    assert exc.value.code == "IDX2_BINDING"


def test_source_drift_stat_refused(v2_stack) -> None:
    tables = v2_stack["tables"]
    pkl = tables / ".light_index_v53mp.pkl"
    pkl.write_bytes(pkl.read_bytes() + b"\x00")   # size+mtime change
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ColumnarKeyAtomDatasetV3(
            tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    assert exc.value.code == "IDX2_BINDING"


# ---------------------------------------------------------------------------
# records()/geometry/features equivalence vs v1
# ---------------------------------------------------------------------------
def test_records_and_geometry_equivalence(v2_stack) -> None:
    v1, v2 = v2_stack["v1"], v2_stack["v2"]
    for p in (0, 3, 7, 15):
        r1 = v1.records([p])[0]
        r2 = v2.records([p])[0]
        assert r1.sample_id == r2.sample_id
        assert r1.intent == r2.intent
        s1 = v1.to_sample_v3(r1)
        s2 = v2.to_sample_v3(r2)
        assert s1.xyz == s2.xyz
        assert s1.sample_id == s2.sample_id
        assert np.float32(s1.frame_weight) == np.float32(s2.frame_weight)
        b1 = collate_v3([build_sample_features_v3(s1)])
        b2 = collate_v3([build_sample_features_v3(s2)])
        assert set(b1) == set(b2)
        for key in b1:
            va, vb = b1[key], b2[key]
            if isinstance(va, np.ndarray):
                assert va.dtype == vb.dtype, key
                assert np.array_equal(va, vb), key
            else:
                assert va == vb, (key, va, vb)
    for cid in CURVE_IDS:
        for node in range(N_FRAMES):
            a = np.asarray(v1.frame_xyz(cid, node), dtype=np.float64)
            c = np.asarray(v2.frame_xyz(cid, node), dtype=np.float64)
            assert a.tobytes() == c.tobytes(), (cid, node)
        assert v1.curve_ts_index(cid) == v2.curve_ts_index(cid)
        assert v1.curve_endpoints(cid) == v2.curve_endpoints(cid)
        assert v1.curve_energies(cid) == v2.curve_energies(cid)


def test_sampler_positional_equality(v2_stack) -> None:
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        V53MPTrainerConfig,
        _CurveDirIndex,
        _curve_batch_positions,
        _update_rng,
    )
    v1, v2 = v2_stack["v1"], v2_stack["v2"]
    positions = list(range(len(v1)))
    idx1 = _CurveDirIndex.build(v1.samples, positions)
    idx2 = _CurveDirIndex.build(v2.samples, positions)
    assert idx1.curves == idx2.curves
    assert idx1.pools.keys() == idx2.pools.keys()
    for key in idx1.pools:
        assert idx1.pools[key] == idx2.pools[key]
    config = V53MPTrainerConfig(arm="A", seed=17, backbone_init="r3")
    for update in range(1, 21):
        b1 = _curve_batch_positions(
            v1.samples, positions, config=config,
            rng=_update_rng(17, update, salt=0), index=idx1)
        b2 = _curve_batch_positions(
            v2.samples, positions, config=config,
            rng=_update_rng(17, update, salt=0), index=idx2)
        assert b1 == b2, update


def test_v52_labels_fallback_identical(v2_stack) -> None:
    """dataset.v52_labels() (recompute path) still works on the col loader."""
    v1, v2 = v2_stack["v1"], v2_stack["v2"]
    ref = v1.v52_labels()
    got = v2.v52_labels()
    assert set(ref.by_sample) == set(got.by_sample)
    for sid in ref.by_sample:
        a, b = ref.by_sample[sid], got.by_sample[sid]
        assert a.q == b.q and a.q_mask == b.q_mask
        assert a.d_key_ts == b.d_key_ts and a.stop == b.stop
    for cid in ref.curves:
        assert ref.curves[cid].arc_s == got.curves[cid].arc_s
        assert (ref.curves[cid].q_canonical == got.curves[cid].q_canonical)


def test_verify_sources_full_hash_passes(v2_stack) -> None:
    """verify_sources=True re-hashes the sources and still opens."""
    v2 = ColumnarKeyAtomDatasetV3(
        v2_stack["tables"], label_statuses=ALL_STATUSES,
        verify_manifest=False, verify_sources=True)
    assert len(v2) == len(v2_stack["v1"])


def test_label_curve_rekeying_order_divergent(tmp_path: Path, monkeypatch) -> None:
    """Regression: label-cache curve order (sorted) != first-appearance order.

    The real 20k tables exposed a rekeying bug the symmetric mini-fixture
    (curve_000 < curve_001 in both orders) could not.  Force divergence by
    naming the second curve such that sorted order flips vs appearance order.
    """
    import tests.pesn.test_r3_v53mp_dataset as base
    monkeypatch.setattr(base, "CURVE_IDS", ("curve_b", "curve_a"))
    tables = _write_fixture(tmp_path / "data")
    v1, _ = _build_v2_stack(tables)
    v2 = ColumnarKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=False)
    labels_ref, _ep_ref = __import__(
        "pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp",
        fromlist=["load_label_cache"]).load_label_cache(
            tables, dataset_manifest_sha256=v1.manifest_sha256)
    labels_v2, _ep_v2 = v2.label_tables()
    # appearance order: curve_b first; sorted order: curve_a first
    assert v2._curve_ids[0] == "curve_b"
    for cid in ("curve_a", "curve_b"):
        assert labels_v2.curves[cid] == labels_ref.curves[cid], cid
    for rec in v2.samples:
        assert (labels_v2.by_sample[rec.sample_id]
                == labels_ref.by_sample[rec.sample_id])
