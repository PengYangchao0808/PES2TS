"""Geometry memmap-pack tests (S4 plan §5.2 / lane I2).

Synthetic mini-fixture coverage on top of the two-curve six-table builder
from ``tests/pesn/test_r3_v53mp_dataset``:

* **builder→reader roundtrip** — after ``write_geometry_shards`` +
  ``build_geometry_pack``, ``ColumnarKeyAtomDatasetV3(use_geometry_pack=
  True)`` serves frame_xyz / to_sample_v3 bytes IDENTICAL to the shard-LRU
  path (``use_geometry_pack=False``), for every node of every curve;
* **binding fail-closed** — a drifted shard_index / data_manifest sha, a
  missing coords file, a truncated coords file and a tampered coords file
  are all typed refusals, never silent wrong geometry;
* **absence keeps the shard path** — no ``geom/`` directory ⇒ the pack is
  inactive and the shard LRU serves everything (v1 behaviour preserved);
* **feature equality** — collated F2 features from the pack path equal
  the shard path array-by-array.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
    collate_v3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3Error,
    write_geometry_shards,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    GEOM_DIRNAME,
    GEOM_OFFSETS_FILENAME,
    ColumnarKeyAtomDatasetV3,
)
from scripts.agent_ops.data_gate.r3_v53mp_build_geometry_pack import (  # noqa: E402
    build_geometry_pack,
)
from tests.pesn.test_r3_v53mp_dataset import (  # noqa: E402
    ALL_STATUSES,
    CURVE_IDS,
    N_FRAMES,
    _write_fixture,
)


def _build_stack(tables_dir: Path) -> dict:
    """shards → idx_v2 (via the I1 builder) → geom pack."""
    write_geometry_shards(tables_dir)
    # idx_v2 is required by the columnar reader; reuse the I1 builder
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "i1_builder",
        REPO / "scripts/agent_ops/data_gate/r3_v53mp_build_idx_v2.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        ShardedKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
        build_endpoint_labels_spec_v1,
    )
    from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (
        save_label_cache,
    )
    v1 = ShardedKeyAtomDatasetV3(
        tables_dir, label_statuses=ALL_STATUSES, verify_manifest=False,
        light_cache=True)
    labels = v1.v52_labels()
    endpoint = build_endpoint_labels_spec_v1(v1, labels)
    save_label_cache(tables_dir, labels, endpoint,
                     dataset_manifest_sha256=v1.manifest_sha256)
    builder.build_idx_v2(tables_dir, tables_dir / "idx_v2",
                         verify_sources=False)
    summary = build_geometry_pack(tables_dir, tables_dir / GEOM_DIRNAME)
    return {"summary": summary, "v1": v1}


@pytest.fixture()
def stack(tmp_path: Path) -> dict:
    tables = _write_fixture(tmp_path / "data")
    out = _build_stack(tables)
    out["tables"] = tables
    return out


def _open(tables: Path, **kwargs) -> ColumnarKeyAtomDatasetV3:
    return ColumnarKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=False,
        **kwargs)


# ---------------------------------------------------------------------------
# builder → reader roundtrip (bytes identical to the shard path)
# ---------------------------------------------------------------------------
def test_geometry_pack_roundtrip_bytes_equal(stack) -> None:
    tables = stack["tables"]
    summary = stack["summary"]
    assert summary["n_curves"] == len(CURVE_IDS)
    assert summary["total_atom_rows"] > 0
    ds_pack = _open(tables, use_geometry_pack=True)
    ds_shard = _open(tables, use_geometry_pack=False)
    assert ds_pack.geom_pack_binding is not None
    assert ds_shard.geom_pack_binding is None
    # every node of every curve, byte-identical
    for cid in CURVE_IDS:
        for node in range(N_FRAMES):
            got = np.asarray(ds_pack.frame_xyz(cid, node), dtype=np.float64)
            want = np.asarray(ds_shard.frame_xyz(cid, node), dtype=np.float64)
            assert got.shape == want.shape
            assert got.tobytes() == want.tobytes(), (cid, node)
    # to_sample_v3 → collated features: array-by-array equality
    for pos in range(len(ds_pack)):
        rec_p = ds_pack.records([pos])[0]
        rec_s = ds_shard.records([pos])[0]
        assert rec_p.sample_id == rec_s.sample_id
        b_p = collate_v3([build_sample_features_v3(
            ds_pack.to_sample_v3(rec_p), level="F2", intent_off=False)])
        b_s = collate_v3([build_sample_features_v3(
            ds_shard.to_sample_v3(rec_s), level="F2", intent_off=False)])
        assert set(b_p) == set(b_s)
        for key in b_p:
            va, vb = b_p[key], b_s[key]
            if isinstance(va, np.ndarray):
                assert va.dtype == vb.dtype and np.array_equal(va, vb), key
            else:
                assert va == vb, key


def test_geometry_pack_layout_and_binding(stack) -> None:
    tables = stack["tables"]
    doc = json.loads((tables / GEOM_DIRNAME / GEOM_OFFSETS_FILENAME)
                     .read_text(encoding="utf-8"))
    assert doc["schema"] == "v53mp_geom_pack_v1"
    binding = doc["binding"]
    assert len(binding["shard_index_sha256"]) == 64
    assert len(binding["data_manifest_sha256"]) == 64
    assert len(binding["coords_sha256"]) == 64
    assert binding["n_curves"] == len(CURVE_IDS)
    # offsets are atom-row offsets, contiguous and consistent
    cursor = 0
    coords = np.load(tables / GEOM_DIRNAME / "coords.f64.npy",
                     mmap_mode="r", allow_pickle=False)
    assert coords.shape == (binding["total_atom_rows"], 3)
    assert coords.dtype == np.float64
    for cid in sorted(doc["curves"]):
        entry = doc["curves"][cid]
        assert entry["offset"] == cursor
        assert entry["n_nodes"] > 0 and entry["n_atoms"] > 0
        cursor += entry["n_nodes"] * entry["n_atoms"]
    assert cursor == binding["total_atom_rows"]
    # reader agrees with the on-disk table
    ds = _open(tables)
    assert ds.geom_pack_binding["coords_sha256"] == binding["coords_sha256"]


# ---------------------------------------------------------------------------
# binding fail-closed
# ---------------------------------------------------------------------------
def test_geometry_pack_binding_drift_refused(stack) -> None:
    tables = stack["tables"]
    # tamper the shard index (content changes its sha)
    index = tables / "shard_index_v53mp.json"
    doc = json.loads(index.read_text(encoding="utf-8"))
    doc["tampered"] = True
    index.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        _open(tables)
    assert exc.value.code == "GEOM_BINDING"


def test_geometry_pack_manifest_drift_refused(stack) -> None:
    tables = stack["tables"]
    manifest = tables / "data_manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8")
    # the idx binding ALSO checks the manifest first — so the refusal may
    # come from either binding; both are typed and fail-closed
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        _open(tables)
    assert exc.value.code in ("IDX2_BINDING", "GEOM_BINDING")


def test_geometry_pack_missing_coords_refused(stack) -> None:
    tables = stack["tables"]
    (tables / GEOM_DIRNAME / "coords.f64.npy").unlink()
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        _open(tables)
    assert exc.value.code == "GEOM_MISSING"


def test_geometry_pack_tampered_coords_refused(stack) -> None:
    tables = stack["tables"]
    coords = tables / GEOM_DIRNAME / "coords.f64.npy"
    payload = bytearray(coords.read_bytes())
    payload[-8] ^= 0xFF
    coords.write_bytes(bytes(payload))
    # same size, different bytes → the sha check refuses (verify on)
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        ColumnarKeyAtomDatasetV3(
            tables, label_statuses=ALL_STATUSES, verify_manifest=True)
    assert exc.value.code == "GEOM_SHA"


def test_geometry_pack_coverage_gap_refused(stack) -> None:
    tables = stack["tables"]
    offsets = tables / GEOM_DIRNAME / GEOM_OFFSETS_FILENAME
    doc = json.loads(offsets.read_text(encoding="utf-8"))
    doc["curves"].pop(CURVE_IDS[0])
    offsets.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        _open(tables)
    assert exc.value.code == "GEOM_COVERAGE"


# ---------------------------------------------------------------------------
# absence keeps the shard path (v1 behaviour preserved)
# ---------------------------------------------------------------------------
def test_geometry_pack_absent_falls_back_to_shards(tmp_path: Path) -> None:
    tables = _write_fixture(tmp_path / "data")
    write_geometry_shards(tables)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "i1_builder_b",
        REPO / "scripts/agent_ops/data_gate/r3_v53mp_build_idx_v2.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        ShardedKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
        build_endpoint_labels_spec_v1,
    )
    from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (
        save_label_cache,
    )
    v1 = ShardedKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=False,
        light_cache=True)
    labels = v1.v52_labels()
    save_label_cache(tables, labels, build_endpoint_labels_spec_v1(v1, labels),
                     dataset_manifest_sha256=v1.manifest_sha256)
    builder.build_idx_v2(tables, tables / "idx_v2", verify_sources=False)
    ds = _open(tables)                     # no geom/ directory at all
    assert ds.geom_pack_binding is None
    # shard path still serves byte-correct frames
    ref = np.asarray(v1.frame_xyz(CURVE_IDS[0], 0), dtype=np.float64)
    got = np.asarray(ds.frame_xyz(CURVE_IDS[0], 0), dtype=np.float64)
    assert ref.tobytes() == got.tobytes()


def test_geometry_pack_disabled_by_flag(stack) -> None:
    tables = stack["tables"]
    ds = _open(tables, use_geometry_pack=False)
    assert ds.geom_pack_binding is None
    assert (tables / GEOM_DIRNAME / GEOM_OFFSETS_FILENAME).is_file()


def test_geometry_pack_never_touches_lru(stack) -> None:
    """The pack path bypasses the geometry LRU entirely (cold cache)."""
    ds = _open(stack["tables"])
    assert ds.n_cached_curves == 0
    ds.frame_xyz(CURVE_IDS[0], 0)
    ds.frame_xyz(CURVE_IDS[1], N_FRAMES - 1)
    # no curve ever enters the LRU on the pack path
    assert ds.n_cached_curves == 0
    assert ds.cached_curve_ids == ()


def test_geometry_pack_missing_dir_is_not_an_error(tmp_path: Path) -> None:
    """geom/ absent entirely → reader opens fine, pack inactive."""
    tables = _write_fixture(tmp_path / "data")
    write_geometry_shards(tables)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "i1_builder_c",
        REPO / "scripts/agent_ops/data_gate/r3_v53mp_build_idx_v2.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        ShardedKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
        build_endpoint_labels_spec_v1,
    )
    from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (
        save_label_cache,
    )
    v1 = ShardedKeyAtomDatasetV3(
        tables, label_statuses=ALL_STATUSES, verify_manifest=False,
        light_cache=True)
    labels = v1.v52_labels()
    save_label_cache(tables, labels, build_endpoint_labels_spec_v1(v1, labels),
                     dataset_manifest_sha256=v1.manifest_sha256)
    builder.build_idx_v2(tables, tables / "idx_v2", verify_sources=False)
    assert not (tables / GEOM_DIRNAME).exists()
    ds = _open(tables)
    assert ds.geom_pack_binding is None
    shutil.rmtree(tables / "idx_v2", ignore_errors=True)
