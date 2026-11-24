"""Tests for the read-only Zone-2 readers (``pes2ts_data.readers``)."""

from __future__ import annotations

import csv
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pes2ts_data.readers import (
    CACHE_JSON,
    CACHE_NPZ,
    CACHE_SCHEMA,
    D0_PRIMARY_ANGSTROM,
    LIGHT_CACHE_JSON,
    LIGHT_CACHE_PKL,
    LIGHT_CACHE_SCHEMA,
    READER_VERSION,
    IdxV2Dataset,
    IdxV2Error,
    IdxV2Reader,
    KeyAtomStreamView,
    LabelTables,
    LightIndexState,
    ReaderError,
    idx_v2_available,
    light_index_available,
    light_index_binding,
    load_label_cache,
    load_light_index,
    read_light_index_metadata,
)
from pes2ts_data.readers._common import (
    LABEL_STATUSES,
    TABLE_FILENAMES,
    sha256_file,
    verify_data_manifest,
)

CURVES = {
    "C1": {"canonical_irc_id": "I1", "n_nodes": 3, "ts_index": 1,
           "atoms": [6, 6, 6], "fold": 0},
    "C2": {"canonical_irc_id": "I2", "n_nodes": 2, "ts_index": 0,
           "atoms": [1, 1], "fold": 1},
}


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _build_tables(tmp_path: Path) -> Path:
    """Write the six frozen-shape CSV tables plus manifests under tmp_path."""
    tables = tmp_path / "tables"
    tables.mkdir(parents=True)
    _write_csv(tables / TABLE_FILENAMES["irc_curve"],
               ("curve_id", "canonical_irc_id", "n_nodes", "ts_index"),
               [(cid, meta["canonical_irc_id"], meta["n_nodes"], meta["ts_index"])
                for cid, meta in CURVES.items()])
    node_rows = [(cid, node, 0.0, 0.0, 0.0, json.dumps(meta["atoms"]))
                 for cid, meta in CURVES.items() for node in range(meta["n_nodes"])]
    _write_csv(tables / TABLE_FILENAMES["irc_node"],
               ("curve_id", "node_index", "x", "y", "z", "atomic_numbers"),
               node_rows)
    _write_csv(tables / TABLE_FILENAMES["irc_edge"],
               ("curve_id", "intent_id", "edge_index", "delta_xyz_flat"), [])
    _write_csv(tables / TABLE_FILENAMES["curve_quality"],
               ("curve_id", "quality_status"), [("C1", "ok"), ("C2", "ok")])
    intent_rows = []
    for cid, meta in CURVES.items():
        payload = json.dumps({"atoms": meta["atoms"]})
        intent_rows.extend([
            (f"{cid}-f", cid, "forward", f"{cid}-r", f"{cid}-pair", payload, payload, "[]"),
            (f"{cid}-r", cid, "reverse", f"{cid}-f", f"{cid}-pair", payload, payload, "[]"),
        ])
    _write_csv(tables / TABLE_FILENAMES["reaction_intent"],
               ("intent_id", "curve_id", "direction", "reverse_of_intent_id",
                "intent_pair_id", "g_src", "g_tgt", "cv_spec"), intent_rows)
    sample_header = ("sample_id", "curve_id", "node_index", "intent_id", "d0",
                     "label_weight", "curve_weight", "sample_weight",
                     "label_status", "fold")
    sample_rows = [
        ("S1", "C1", 0, "C1-f", 0.08, 1.0, 0.5, 1.0, "ok", 0),
        ("S2", "C1", 1, "C1-f", 0.08, 1.0, 0.5, 1.0, "ok", 0),
        ("S3", "C1", 0, "C1-r", 0.08, 1.0, 0.0, 1.0, "ok", 0),
        ("S4", "C1", 2, "C1-f", 0.10, 1.0, 5.0, 1.0, "ok", 0),
        ("S5", "C2", 0, "C2-f", 0.08, 1.0, 1.0, 1.0, "masked", 1),
    ]
    _write_csv(tables / TABLE_FILENAMES["sample_view"], sample_header, sample_rows)
    manifest = {"files": {
        filename: sha256_file(tables / filename)
        for filename in TABLE_FILENAMES.values()}}
    (tables / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tables / "split_manifest_r3.json").write_text(
        json.dumps({"curve_folds": {cid: meta["fold"] for cid, meta in CURVES.items()}}),
        encoding="utf-8")
    return tables


@pytest.fixture()
def tables_dir(tmp_path: Path) -> Path:
    return _build_tables(tmp_path)


# ---------------------------------------------------------------------------
# Six-table streaming view
# ---------------------------------------------------------------------------

def test_stream_view_light_tables(tables_dir: Path):
    view = KeyAtomStreamView(tables_dir)
    assert view.curve_ids == ["C1", "C2"]
    assert view.n_curves == 2
    assert view.n_nodes == 5
    assert view.node_indices("C1") == [0, 1, 2]
    assert tuple(view.node_window("C1", 1, 3).tolist()) == (1, 2)
    assert view.atomic_numbers("C1") == (6, 6, 6)
    assert view.curve_meta("C1")["ts_index"] == "1"
    assert view.fold_of("C2") == 1
    assert view.intent("C1", "C1-r").direction == "reverse"
    assert view.intent("C1", "C1-r").reverse_of_intent_id == "C1-f"


def test_stream_view_manifest_verification(tables_dir: Path):
    view = KeyAtomStreamView(tables_dir, verify_manifest=True)
    assert view.manifest_sha256 == sha256_file(tables_dir / "data_manifest.json")
    assert sha256_file(tables_dir / "data_manifest.json") == verify_data_manifest(tables_dir)

    bad = tables_dir / "data_manifest.json"
    bad.write_text(json.dumps({"files": {"irc_curve.csv": "0" * 64}}), encoding="utf-8")
    with pytest.raises(ReaderError) as excinfo:
        KeyAtomStreamView(tables_dir, verify_manifest=True)
    assert excinfo.value.code == "MANIFEST_SHA"


def test_stream_columns_apply_frozen_survival_filters(tables_dir: Path):
    view = KeyAtomStreamView(tables_dir)
    columns = view.stream_columns()
    # S4 (non-primary d0) is dropped; S1/S2/S3/S5 survive (all statuses kept).
    assert columns.n_rows == 4
    assert columns.curve_idx.tolist() == [0, 0, 0, 1]
    assert columns.node_index.tolist() == [0, 1, 0, 0]
    assert columns.direction.tolist() == [0, 0, 1, 0]
    assert columns.fold.tolist() == [0, 0, 0, 1]
    assert columns.weight_curves_checked == 2
    # S4's curve_weight (5.0) must not enter the invariant (accumulated only
    # over the primary-d0 group).
    assert abs(float(columns.curve_idx.shape[0]) - 4.0) < 1e-12


def test_stream_view_fold_filter(tables_dir: Path):
    view = KeyAtomStreamView(tables_dir, fold=1)
    columns = view.stream_columns()
    assert columns.curve_idx.tolist() == [1]
    with pytest.raises(ReaderError) as excinfo:
        KeyAtomStreamView(tables_dir, fold=7).stream_columns()
    # Every curve of the manifest has fold != 7, so no rows survive the filter.
    assert excinfo.value.code == "NO_SAMPLES"


def test_stream_view_label_joins(tables_dir: Path):
    view = KeyAtomStreamView(tables_dir)
    rows = list(view.iter_sample_rows())
    assert len(rows) == 5 and rows[0]["sample_id"] == "S1"
    joined = list(view.iter_label_joined_rows())
    assert joined[0]["direction"] == "forward"
    assert joined[0]["node_index"] == 0
    assert joined[2]["direction"] == "reverse"

    labels = _load_fixture_labels(tables_dir)
    with_labels = list(view.iter_label_joined_rows(labels))
    assert with_labels[0]["labels"]["q"] == pytest.approx(0.1)
    assert with_labels[4]["labels"]["endpoint_label"] == -2


def test_stream_view_typed_refusals(tmp_path: Path):
    with pytest.raises(ReaderError) as excinfo:
        KeyAtomStreamView(tmp_path / "missing")
    assert excinfo.value.code == "TABLES_DIR"

    tables = _build_tables(tmp_path / "a")
    (tables / TABLE_FILENAMES["irc_edge"]).unlink()
    with pytest.raises(ReaderError) as excinfo:
        KeyAtomStreamView(tables)
    assert excinfo.value.code == "TABLE_MISSING"

    tables = _build_tables(tmp_path / "b")
    with (tables / TABLE_FILENAMES["irc_curve"]).open("a", encoding="utf-8") as handle:
        handle.write("C1,I1,3,1\n")
    with pytest.raises(ReaderError) as excinfo:
        KeyAtomStreamView(tables)
    assert excinfo.value.code == "CURVE_ID_DUP"

    tables = _build_tables(tmp_path / "c")
    with (tables / TABLE_FILENAMES["sample_view"]).open("w", encoding="utf-8") as handle:
        handle.write("sample_id,curve_id\nS1,C1\n")
    with pytest.raises(ReaderError) as excinfo:
        KeyAtomStreamView(tables).stream_columns()
    assert excinfo.value.code == "TABLE_COLUMNS"


# ---------------------------------------------------------------------------
# Label cache
# ---------------------------------------------------------------------------

def _fixture_manifest_sha(tables_dir: Path) -> str:
    return sha256_file(tables_dir / "data_manifest.json")


def _write_fixture_labels(tables_dir: Path, *, stale: bool = False) -> None:
    doc = {
        "schema": CACHE_SCHEMA,
        "dataset_manifest_sha256": ("0" * 64 if stale
                                    else _fixture_manifest_sha(tables_dir)),
        "n_samples": 5,
        "n_curves": 1,
        "curve_meta": [{"curve_id": "C1", "ts_index": 1,
                        "q_zero_arc_reason": "", "n": 3}],
        "endpoint_windows": {"positive_window_angstrom": 0.2,
                             "negative_window_angstrom": 0.2},
    }
    arrays = {
        "sample_ids": np.asarray([f"S{index}" for index in range(1, 6)], dtype="U64"),
        "q": np.asarray([0.1, 0.2, 0.3, 0.4, np.nan]),
        "q_mask": np.asarray([True, True, True, True, False]),
        "stop": np.asarray([1, 1, 1, 1, -2], dtype=np.int8),
        "d_key_ts": np.asarray([0.2, 0.2, 0.2, 0.2, np.nan]),
        "fold": np.asarray([0, 0, 0, 0, -1], dtype=np.int64),
        "label_status": np.asarray(["ok", "ok", "ok", "ok", "masked"], dtype="U16"),
        "direction": np.asarray(["forward", "forward", "reverse", "forward",
                                 "forward"], dtype="U8"),
        "curve_of": np.asarray(["C1"] * 5, dtype="U24"),
        "node_index": np.asarray([0, 1, 0, 2, 0], dtype=np.int64),
        "intent_of": np.asarray(["C1-f", "C1-f", "C1-r", "C1-f", "C1-f"],
                                dtype="U40"),
        "ep_label": np.asarray([1, 1, 0, 1, -2], dtype=np.int8),
        "d_end": np.asarray([0.3, 0.3, 0.2, 0.3, np.nan]),
        "bond_edit_complete": np.asarray([True, True, True, True, False]),
        "geometry_legal": np.asarray([True] * 5),
        "ts_row": np.asarray([True, False, False, True, False]),
        "start_row": np.asarray([True, False, False, False, False]),
        "terminal_row": np.asarray([False, False, False, True, False]),
        "degenerate_curve": np.asarray([False, False, False, False, True]),
        "energy_ev_within_window": np.asarray([1, 1, 1, 1, -1], dtype=np.int8),
        "d_key_ts_ep": np.asarray([0.2, 0.2, 0.2, 0.2, 0.4]),
        "node_order": np.asarray([0, 1, 2], dtype=np.int64),
        "arc_s": np.asarray([0.0, 0.1, 0.2]),
        "q_canonical": np.asarray([0.5, 0.0, -0.5]),
    }
    np.savez(tables_dir / CACHE_NPZ, **arrays)
    (tables_dir / CACHE_JSON).write_text(json.dumps(doc), encoding="utf-8")


def _load_fixture_labels(tables_dir: Path) -> LabelTables:
    _write_fixture_labels(tables_dir)
    return load_label_cache(tables_dir)


def test_label_cache_round_trip(tables_dir: Path):
    labels = _load_fixture_labels(tables_dir)
    assert isinstance(labels, LabelTables)
    assert labels.n_samples == 5 and labels.n_curves == 1
    assert labels.sample_ids == ["S1", "S2", "S3", "S4", "S5"]
    sample = labels.sample("S1")
    assert sample["q"] == pytest.approx(0.1)
    assert sample["q_mask"] is True
    assert sample["endpoint_label"] == 1
    assert sample["energy_ev_within_window"] is True
    masked = labels.sample("S5")
    assert masked["fold"] is None
    assert masked["q"] is None
    assert masked["endpoint_label"] == -2
    curve = labels.curve("C1")
    assert curve.arc_s == (0.0, 0.1, 0.2)
    assert curve.q_canonical == (0.5, 0.0, -0.5)
    assert labels.positive_window == pytest.approx(0.2)


def test_label_cache_typed_refusals(tables_dir: Path, tmp_path: Path):
    with pytest.raises(ReaderError) as excinfo:
        load_label_cache(tmp_path / "missing")
    assert excinfo.value.code == "CACHE_MISSING"

    _write_fixture_labels(tables_dir)
    (tables_dir / CACHE_JSON).write_text(
        json.dumps({"schema": "wrong"}), encoding="utf-8")
    with pytest.raises(ReaderError) as excinfo:
        load_label_cache(tables_dir)
    assert excinfo.value.code == "CACHE_SCHEMA"

    _write_fixture_labels(tables_dir, stale=True)
    with pytest.raises(ReaderError) as excinfo:
        load_label_cache(tables_dir)
    assert excinfo.value.code == "CACHE_STALE"

    # Structural load without binding verification still succeeds.
    assert load_label_cache(tables_dir, verify_manifest=False).n_samples == 5

    _write_fixture_labels(tables_dir)
    data = dict(np.load(tables_dir / CACHE_NPZ))
    data.pop("ep_label")
    np.savez(tables_dir / CACHE_NPZ, **data)
    with pytest.raises(ReaderError) as excinfo:
        load_label_cache(tables_dir)
    assert excinfo.value.code == "CACHE_ARRAYS"


# ---------------------------------------------------------------------------
# Light index
# ---------------------------------------------------------------------------

def _light_binding(tables_dir: Path, fold: int | None = None) -> dict:
    return light_index_binding(
        data_manifest_sha256=_fixture_manifest_sha(tables_dir),
        split_manifest_sha256=sha256_file(tables_dir / "split_manifest_r3.json"),
        d0=D0_PRIMARY_ANGSTROM,
        label_statuses=LABEL_STATUSES,
        directions=("forward", "reverse"),
        fold=fold,
    )


def _write_light_index(tables_dir: Path, *, binding: dict | None = None,
                       state: dict | None = None) -> dict:
    binding = binding or _light_binding(tables_dir)
    state = state or {
        "node_index": {"C1": {0, 1, 2}},
        "curve_atomic": {"C1": (6, 6, 6)},
        "curve_energy": {"C1": {0: 0.0, 1: 0.1}},
        "samples": [{"sample_id": "S1"}],
        "weight_totals": {"C1": 1.0},
    }
    (tables_dir / LIGHT_CACHE_JSON).write_text(
        json.dumps({"binding": binding, "n_samples": len(state["samples"])}),
        encoding="utf-8")
    with (tables_dir / LIGHT_CACHE_PKL).open("wb") as handle:
        pickle.dump(state, handle)
    return binding


def test_light_index_round_trip(tables_dir: Path):
    binding = _write_light_index(tables_dir)
    assert light_index_available(tables_dir)
    state = load_light_index(tables_dir, expected_binding=binding)
    assert isinstance(state, LightIndexState)
    assert state.n_samples == 1
    assert state.node_index["C1"] == {0, 1, 2}
    assert state.curve_atomic["C1"] == (6, 6, 6)
    assert state.weight_totals["C1"] == pytest.approx(1.0)
    metadata = read_light_index_metadata(tables_dir)
    assert metadata["binding"]["schema"] == LIGHT_CACHE_SCHEMA


def test_light_index_typed_refusals(tables_dir: Path):
    with pytest.raises(ReaderError) as excinfo:
        load_light_index(tables_dir)
    assert excinfo.value.code == "LIGHT_INDEX_MISSING"

    binding = _write_light_index(tables_dir)
    with pytest.raises(ReaderError) as excinfo:
        load_light_index(tables_dir, expected_binding={**binding, "d0": 0.09})
    assert excinfo.value.code == "LIGHT_INDEX_STALE"

    (tables_dir / LIGHT_CACHE_JSON).write_text(
        json.dumps({"binding": {"schema": "wrong"}}), encoding="utf-8")
    with pytest.raises(ReaderError) as excinfo:
        load_light_index(tables_dir)
    assert excinfo.value.code == "LIGHT_INDEX_SCHEMA"


def _dump_with_fake_module(path: Path, module_name: str, cls_name: str,
                           state: dict) -> None:
    module = types.ModuleType(module_name)
    cls = type(cls_name, (), {})
    cls.__module__ = module_name
    cls.__qualname__ = cls_name
    setattr(module, cls_name, cls)
    sys.modules[module_name] = module
    try:
        instance = cls()
        instance.sample_id = "S1"
        instance.node_index = 2
        state = dict(state)
        state["samples"] = [instance]
        with path.open("wb") as handle:
            pickle.dump(state, handle)
    finally:
        sys.modules.pop(module_name, None)


def test_light_index_shims_removed_frozen_classes(tables_dir: Path):
    binding = _light_binding(tables_dir)
    base_state = {
        "node_index": {"C1": {0}}, "curve_atomic": {"C1": (6,)},
        "curve_energy": {"C1": {}}, "samples": [], "weight_totals": {"C1": 1.0},
    }
    _write_light_index(tables_dir, binding=binding, state=base_state)
    _dump_with_fake_module(
        tables_dir / LIGHT_CACHE_PKL,
        "pes2ts_core.pesn.offline.data._removed_standin",
        "SampleRecordV3Lite", base_state)
    state = load_light_index(tables_dir)
    sample = state.samples[0]
    assert sample.sample_id == "S1" and sample.node_index == 2


def test_light_index_refuses_untrusted_pickle_classes(tables_dir: Path):
    binding = _light_binding(tables_dir)
    base_state = {
        "node_index": {}, "curve_atomic": {}, "curve_energy": {},
        "samples": [], "weight_totals": {},
    }
    _write_light_index(tables_dir, binding=binding, state=base_state)
    _dump_with_fake_module(tables_dir / LIGHT_CACHE_PKL,
                           "evil_module_outside_roots", "Payload", base_state)
    with pytest.raises(ReaderError) as excinfo:
        load_light_index(tables_dir)
    assert excinfo.value.code == "LIGHT_INDEX_UNREADABLE"


# ---------------------------------------------------------------------------
# idx_v2 memmap reader + numpy dataset view
# ---------------------------------------------------------------------------

def _write_idx_v2(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    curves_doc = {
        "schema": "v53mp_idx_v2",
        "created_at": "2026-11-01T00:00:00+00:00",
        "builder_git_commit": "abcdef123456",
        "binding": {"n_curves": 2, "n_intents": 2, "n_samples": 5,
                    "data_manifest_sha256": "d" * 64},
        "curve_ids": ["RXN_0000000001", "RXN_0000000002"],
        "curves": [
            {"n_nodes": 3, "ts_index": 1, "atomic_numbers": [6, 6, 6],
             "endpoint_a": 0, "endpoint_b": 2, "weight_total": 1.0,
             "has_energy": False},
            {"n_nodes": 2, "ts_index": 0, "atomic_numbers": [1, 1],
             "endpoint_a": 0, "endpoint_b": 1, "weight_total": 1.0,
             "has_energy": False},
        ],
        "label_status_table": ["conflict", "masked", "ok", "terminal"],
        "files": {"curves.json": {"sha256": "e" * 64}},
    }
    (root / "curves.json").write_text(json.dumps(curves_doc), encoding="utf-8")
    groups = {
        "label_curves": {
            "q_canonical": np.asarray([0.2, 0.1, 0.0, 0.5, 0.9], dtype=np.float32),
            "arc_s": np.asarray([0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float32),
            "node_order": np.asarray([0, 1, 2, 1, 0], dtype=np.int64),
            "offsets": np.asarray([0, 3, 5], dtype=np.int64),
        },
        "samples": {
            "curve_idx": np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
            "intent_idx": np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
            "node_index": np.asarray([0, 1, 2, 0, 1], dtype=np.int64),
            "label_status": np.asarray([2, 2, 2, 2, 2], dtype=np.int64),
            "fold": np.asarray([0, 1, 2, 0, 1], dtype=np.int64),
            "d0": np.asarray([0.08] * 5, dtype=np.float32),
            "sample_weight": np.ones(5, dtype=np.float32),
            "label_weight": np.ones(5, dtype=np.float32),
            "curve_weight": np.asarray([1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                                       dtype=np.float32),
        },
        "labels_ep": {
            "ts_row": np.asarray([True, False, False, True, False]),
            "start_row": np.asarray([True, False, False, True, False]),
            "terminal_row": np.asarray([False, False, True, False, True]),
            "d_key_ts_ep": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            "degenerate_curve": np.zeros(5, dtype=bool),
            "ep_label": np.asarray([1, 0, -1, 1, -1], dtype=np.int8),
            "geometry_legal": np.ones(5, dtype=bool),
        },
        "labels_v52": {
            "q": np.asarray([0.2, 0.1, 0.0, 0.5, 0.9], dtype=np.float32),
            "q_mask": np.ones(5, dtype=bool),
        },
    }
    for group, members in groups.items():
        (root / group).mkdir()
        for name, array in members.items():
            np.save(root / group / f"{name}.npy", array)
    return root


@pytest.fixture()
def idx_root(tmp_path: Path) -> Path:
    return _write_idx_v2(tmp_path / "idx_v2")


def test_idx_v2_reader_available_and_counts(idx_root: Path):
    assert idx_v2_available(idx_root) is True
    assert idx_v2_available(idx_root / "nope") is False
    reader = IdxV2Reader(idx_root)
    assert reader.n_curves == 2
    assert reader.n_samples == 5
    assert reader.n_nodes == 5
    assert reader.counts() == {"n_curves": 2, "n_intents": 2,
                               "n_frames": 5, "n_samples": 5}
    doc = reader.manifest_doc()
    assert doc["schema"] == "v53mp_idx_v2"
    assert doc["dataset_version"].startswith("idx_v2-2026-11-01")
    assert len(doc["dataset_hash"]) == 64


def test_idx_v2_curve_arrays_respect_node_order(idx_root: Path):
    reader = IdxV2Reader(idx_root)
    nodes = reader.curve_node_arrays(0)
    assert nodes["q"].tolist() == pytest.approx([0.2, 0.1, 0.0])
    assert nodes["s"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    reordered = reader.curve_node_arrays(1)
    # stored rows are node_order [1, 0] -> output is by node index.
    assert reordered["q"].tolist() == pytest.approx([0.9, 0.5])
    assert reader.sample_rows(0) == (0, 3)
    assert reader.sample_rows(1) == (3, 5)


def test_idx_v2_reader_typed_refusals(tmp_path: Path, idx_root: Path):
    with pytest.raises(IdxV2Error) as excinfo:
        IdxV2Reader(tmp_path / "missing")
    assert excinfo.value.code == "IDX_V2_MISSING"

    bad = _write_idx_v2(tmp_path / "bad_schema")
    (bad / "curves.json").write_text(json.dumps({"schema": "v0"}), encoding="utf-8")
    with pytest.raises(IdxV2Error) as excinfo:
        IdxV2Reader(bad)
    assert excinfo.value.code == "IDX_V2_SCHEMA"

    incomplete = _write_idx_v2(tmp_path / "incomplete")
    (incomplete / "labels_ep" / "d_key_ts_ep.npy").unlink()
    with pytest.raises(IdxV2Error) as excinfo:
        IdxV2Reader(incomplete)
    assert excinfo.value.code == "IDX_V2_ARRAY"


def test_idx_v2_dataset_builds_v55_shaped_sequences(idx_root: Path):
    reader = IdxV2Reader(idx_root)
    dataset = IdxV2Dataset(reader)
    counts = dataset.counts
    assert counts["n_curves"] == reader.n_curves
    assert counts["n_trajectories"] >= 1
    assert dataset.frame_feat_dim == 7 and dataset.static_feat_dim == 8

    sequence = dataset.build_sequence(0)
    assert sequence["frame_feats"].shape == (3, 7)
    assert sequence["static_feats"].shape == (8,)
    assert sequence["move_target"].shape == (3,)
    assert sequence["move_mask"].shape == (3,)
    assert sequence["process_label"].shape == (3,)
    assert sequence["trust_mask"].shape == (3,)
    assert sequence["length"] == 3

    chunks = list(dataset.iter_sequences("train", max_samples=2))
    assert chunks and len(chunks[0][0]) >= 1
    assert chunks[0][1][0]["reaction_id"].startswith("RXN_")


def test_readers_are_torch_free():
    readers_dir = Path(__file__).resolve().parents[2] / "pes2ts_data" / "readers"
    for path in readers_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import torch" not in text, path
        assert "from torch" not in text, path


def test_reader_version():
    assert READER_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# Regression: max_curves vs global trajectory indexing (real-data smoke bug)
# ---------------------------------------------------------------------------

def _write_idx_v2_multi(root: Path, *, n_curves: int = 6,
                        nodes_per_curve: int = 2) -> Path:
    """Synthetic idx_v2 with ``n_curves`` one-intent curves (seed-85 split mix)."""
    root.mkdir(parents=True, exist_ok=True)
    n_nodes = n_curves * nodes_per_curve
    curve_ids = [f"RXN_{index:010d}" for index in range(n_curves)]
    curves_doc = {
        "schema": "v53mp_idx_v2",
        "created_at": "2026-11-01T00:00:00+00:00",
        "builder_git_commit": "abcdef123456",
        "binding": {"n_curves": n_curves, "n_intents": n_curves,
                    "n_samples": n_curves,
                    "data_manifest_sha256": "d" * 64},
        "curve_ids": curve_ids,
        "curves": [
            {"n_nodes": nodes_per_curve, "ts_index": index % nodes_per_curve,
             "atomic_numbers": [6] * nodes_per_curve,
             "endpoint_a": 0, "endpoint_b": nodes_per_curve - 1,
             "weight_total": 1.0, "has_energy": False}
            for index in range(n_curves)],
        "label_status_table": ["conflict", "masked", "ok", "terminal"],
        "files": {},
    }
    (root / "curves.json").write_text(json.dumps(curves_doc), encoding="utf-8")
    groups = {
        "label_curves": {
            "q_canonical": np.linspace(0.2, -0.2, n_nodes, dtype=np.float32),
            "arc_s": np.tile(np.asarray([0.0, 1.0], dtype=np.float32), n_curves),
            "node_order": np.tile(
                np.arange(nodes_per_curve, dtype=np.int64), n_curves),
            "offsets": np.arange(n_curves + 1, dtype=np.int64) * nodes_per_curve,
        },
        "samples": {
            "curve_idx": np.arange(n_curves, dtype=np.int64),
            "intent_idx": np.arange(n_curves, dtype=np.int64),
            "node_index": np.zeros(n_curves, dtype=np.int64),
            "label_status": np.full(n_curves, 2, dtype=np.int64),
            "fold": np.arange(n_curves, dtype=np.int64) % 3,
            "d0": np.full(n_curves, 0.08, dtype=np.float32),
            "sample_weight": np.ones(n_curves, dtype=np.float32),
            "label_weight": np.ones(n_curves, dtype=np.float32),
            "curve_weight": np.ones(n_curves, dtype=np.float32),
        },
        "labels_ep": {
            "ts_row": np.ones(n_curves, dtype=bool),
            "start_row": np.zeros(n_curves, dtype=bool),
            "terminal_row": np.zeros(n_curves, dtype=bool),
            "d_key_ts_ep": np.linspace(0.1, 0.5, n_curves, dtype=np.float32),
            "degenerate_curve": np.zeros(n_curves, dtype=bool),
            "ep_label": np.ones(n_curves, dtype=np.int8),
            "geometry_legal": np.ones(n_curves, dtype=bool),
        },
        "labels_v52": {
            "q": np.zeros(n_curves, dtype=np.float32),
            "q_mask": np.ones(n_curves, dtype=bool),
        },
    }
    for group, members in groups.items():
        (root / group).mkdir()
        for name, array in members.items():
            np.save(root / group / f"{name}.npy", array)
    return root


def test_max_curves_global_index_space_is_consistent(tmp_path: Path):
    """Capped split population must never shift or overflow trajectory indices.

    Regression for the real-data smoke bug: with ``data.limit`` -> ``max_curves``
    the v5 adapter maps a per-split position with ``offset = sum(previous split
    trajectory counts)``; if Zone 2 resolved indices by "first split with
    ``index < len``", the offset of a later split exceeded every local length and
    raised ``TRAJ_RANGE`` (e.g. index 16 with 8 curves).
    """
    reader = IdxV2Reader(_write_idx_v2_multi(tmp_path / "idx_v2"))
    dataset = IdxV2Dataset(reader, seed=85,
                           max_curves={"train": 2, "valid": 1, "id": 1})
    offsets = dataset.trajectory_offsets()
    assert offsets == {"train": 0, "valid": 2, "id": 3}
    assert dataset.n_trajectories == 4

    # The v5 adapter computes the same offsets from the per-split trajectory
    # lists (``pes2ts_v5.data.idx_v2_source._split_offsets``); pin the contract.
    adapter_offsets: dict[str, int] = {}
    running = 0
    for split, refs in dataset._trajs.items():
        adapter_offsets[split] = running
        running += len(refs)
    assert adapter_offsets == offsets

    for split in ("train", "valid", "id"):
        positions = dataset.trajectory_indices(split)
        curve_rows = dataset._splits[split]
        assert positions == list(range(len(positions)))
        assert len(positions) == len(curve_rows)
        for position in positions:
            sequence = dataset.build_sequence(offsets[split] + position)
            assert sequence["traj_idx"] == offsets[split] + position
            assert sequence["reaction_row"] == curve_rows[position]
            assert dataset.trajectory_split(offsets[split] + position) == split

        # fraction < 1, shuffled order and max_samples variants
        fraction_positions = dataset.trajectory_indices(split, fraction=0.5,
                                                        seed=17)
        assert all(0 <= position < len(positions) for position in fraction_positions)
        for position in fraction_positions:
            dataset.build_sequence(offsets[split] + position)

        rng = np.random.default_rng(17)
        shuffled = [positions[i] for i in rng.permutation(len(positions))]
        for position in shuffled:
            dataset.build_sequence(offsets[split] + position)

        global_indices = dataset.global_trajectory_indices(split, fraction=0.5)
        assert all(offsets[split] <= index < offsets[split] + len(positions)
                   for index in global_indices)

        samples, metas = next(dataset.iter_sequences(
            split, fraction=0.5, shuffle=True, max_samples=1))
        assert len(samples) <= 1 and len(metas) == len(samples)
        for sample in samples:
            assert 0 <= sample["traj_idx"] < dataset.n_trajectories
            rebuilt = dataset.build_sequence(sample["traj_idx"])
            assert rebuilt["trajectory_id"] == sample["trajectory_id"]

    # Out-of-space indices stay typed and never reach the numpy arrays.
    for bad in (dataset.n_trajectories, dataset.n_trajectories + 16, -1):
        with pytest.raises(IdxV2Error) as excinfo:
            dataset.build_sequence(bad)
        assert excinfo.value.code == "TRAJ_RANGE"


