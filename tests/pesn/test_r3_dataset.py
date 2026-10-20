"""R3 dataset v3 tests: six-table fixtures, intent keys, weights, manifests.

Fixture-level (E1 delivers the real tables in parallel); all results are
``not_admissible_until_gate_pass`` until G0+G1.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    forward_view_v3,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
    collate_v3,
    to_torch_batch,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3  # noqa: E402
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    REQUIRED_COLUMNS,
    KeyAtomDatasetV3,
    KeyAtomDatasetV3Error,
    build_feature_batches,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (  # noqa: E402
    write_fixture_tables,
)

N_FRAMES = 5


@pytest.fixture()
def tables_dir(tmp_path: Path) -> Path:
    write_fixture_tables(tmp_path / "data", n_frames=N_FRAMES)
    return tmp_path / "data"


@pytest.fixture()
def dataset(tables_dir: Path) -> KeyAtomDatasetV3:
    return KeyAtomDatasetV3(tables_dir)


def test_schema_mirror_is_subset_of_e1_schema() -> None:
    """REQUIRED_COLUMNS must stay a subset of the E1 r3_6table_v2 schema."""
    from scripts.agent_ops.data_gate import r3_table_schema

    for table, cols in REQUIRED_COLUMNS.items():
        assert table in r3_table_schema.TABLES, table
        schema_cols = set(r3_table_schema.TABLES[table].column_names)
        missing = set(cols) - schema_cols
        assert not missing, (table, missing)


def test_dataset_loads_fixture_tables(dataset) -> None:
    assert len(dataset) == N_FRAMES * 2          # nodes × both intents
    assert len(dataset.intents) == 2
    (curve_id, intent_id), intent = next(iter(dataset.intents.items()))
    assert intent.direction in ("forward", "reverse")
    assert intent.key_rows() == (0, 1, 2)        # C, Br, O


def test_intent_keying_and_pair(dataset) -> None:
    fwd = [i for (_, _), i in dataset.intents.items() if i.direction == "forward"]
    rev = [i for (_, _), i in dataset.intents.items() if i.direction == "reverse"]
    assert len(fwd) == len(rev) == 1
    assert fwd[0].reverse_of_intent_id == rev[0].intent_id
    assert rev[0].reverse_of_intent_id == fwd[0].intent_id


def test_curve_weight_invariant_enforced(dataset) -> None:
    per_curve: dict[str, float] = {}
    for rec in dataset.samples:
        per_curve[rec.curve_id] = per_curve.get(rec.curve_id, 0.0) \
            + rec.curve_weight
    for total in per_curve.values():
        assert total == pytest.approx(1.0, abs=1e-9)


def test_record_to_sample_and_feature_build(dataset) -> None:
    rec = dataset.samples[0]
    sample = dataset.to_sample_v3(rec)
    assert sample.frame_weight == rec.label_weight   # loss applies it once
    assert sample.d0 == 0.08
    feats = build_sample_features_v3(sample)
    assert feats["pos"].shape == (6, 3)
    assert feats["key_index"].tolist() == [0, 1, 2]
    assert feats["y_teacher"] is not None


def test_build_feature_batches_shapes(dataset) -> None:
    batch = build_feature_batches(dataset, [0, 1, 2, 3])
    n_keys = int(batch["key_index"].numel())
    assert batch["pos"].shape[0] == 4 * 6
    assert batch["y_teacher"].shape == (n_keys, 3)
    assert batch["cv_target_delta"].shape == (4, 4)
    # model consumes ONLY the forward view
    model = StarEGNNV3()
    out = model(forward_view_v3(batch))
    assert out["u"].shape == (n_keys, 3)


def test_forward_view_excludes_dataset_supervision(dataset) -> None:
    batch = build_feature_batches(dataset, [0, 1])
    fwd = forward_view_v3(batch)
    for key in ("y_teacher", "pair_index", "pair_edit", "frame_weight",
                "d0", "sample_ids", "curve_ids", "intent_ids"):
        assert key not in fwd, key


def test_fold_filter(dataset) -> None:
    kept = KeyAtomDatasetV3(
        dataset.tables_dir, fold=0)
    assert len(kept) == len(dataset)
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        KeyAtomDatasetV3(dataset.tables_dir, fold=1)
    assert exc.value.code == "NO_SAMPLES"


def test_manifest_sha_enforced(tmp_path: Path) -> None:
    write_fixture_tables(tmp_path / "data", n_frames=2)
    # tamper with a table AFTER the manifest was written
    target = tmp_path / "data" / "sample_view.csv"
    with target.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        KeyAtomDatasetV3(tmp_path / "data")
    assert exc.value.code == "MANIFEST_SHA"


def _refresh_manifest(tables_dir: Path) -> None:
    """Recompute data_manifest.json after a test tampers with a table."""
    import hashlib

    names = ("irc_curve.csv", "irc_node.csv", "irc_edge.csv",
             "reaction_intent.csv", "curve_quality.csv", "sample_view.csv")
    manifest = {"files": [
        {"path": name,
         "sha256": hashlib.sha256((tables_dir / name).read_bytes()).hexdigest()}
        for name in names]}
    (tables_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_non_primary_d0_dropped(tmp_path: Path) -> None:
    write_fixture_tables(tmp_path / "data", n_frames=2)
    target = tmp_path / "data" / "sample_view.csv"
    rows = list(csv.reader(target.open(encoding="utf-8")))
    header, body = rows[0], rows[1:]
    d0_col = header.index("d0")
    extra = list(body[0])
    extra[0] = "sensitivity-row"
    extra[d0_col] = "0.16"
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(body + [extra])
    _refresh_manifest(tmp_path / "data")
    ds = KeyAtomDatasetV3(tmp_path / "data")
    assert all(rec.d0 == 0.08 for rec in ds.samples)
    assert len(ds) == 4   # 2 frames × 2 intents; the 0.16 row is dropped


def test_masked_status_filtered_by_default(tmp_path: Path) -> None:
    write_fixture_tables(tmp_path / "data", n_frames=2)
    ds_ok = KeyAtomDatasetV3(tmp_path / "data")
    ds_all = KeyAtomDatasetV3(tmp_path / "data", label_statuses=(
        "ok", "masked", "terminal", "conflict"))
    assert len(ds_all) >= len(ds_ok)


def test_missing_table_typed_refusal(tmp_path: Path) -> None:
    write_fixture_tables(tmp_path / "data", n_frames=2)
    (tmp_path / "data" / "irc_edge.csv").unlink()
    # manifest check trips first (fail-closed), or the table-presence check
    with pytest.raises(KeyAtomDatasetV3Error) as exc:
        KeyAtomDatasetV3(tmp_path / "data")
    assert exc.value.code in ("MANIFEST_FILE", "TABLE_MISSING")


# ---------------------------------------------------------------------------
# E1 six-table encodings (G0 delivery: dict graphs, coords_flat, flat u,
# assignments-style split manifest)
# ---------------------------------------------------------------------------
def _write_e1_style_tables(tmp_path: Path) -> Path:
    import hashlib
    import struct as _struct

    out = tmp_path / "e1data"
    out.mkdir(parents=True, exist_ok=True)
    atoms = [6, 8, 1, 1]
    xyz = [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [-0.5, 0.9, 0.2], [-0.5, -0.9, 0.2]]
    g_src = {"atoms": atoms, "bonds": [[0, 1, 1.0], [0, 2, 1.0], [0, 3, 1.0]]}
    g_tgt = {"atoms": atoms, "bonds": [[0, 1, 2.0], [0, 2, 1.0], [0, 3, 1.0]]}
    cv_spec = {"coordinates": [[0, 1]], "d0_angstrom": 0.08,
               "kind": "graph_edit_bond_lengths"}

    def _csv(name, header, rows):
        with (out / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    _csv("irc_curve.csv",
         ("curve_id", "canonical_irc_id", "n_nodes", "n_edges", "ts_index"),
         [("C1", "canon::C1", 3, 2, 1)])
    _csv("irc_node.csv",
         ("curve_id", "node_index", "x", "y", "z", "progress", "is_ts",
          "is_endpoint", "atomic_numbers", "coords_flat", "coords_shape"),
         [("C1", f, 0.0, 0.0, 0.0, -1 + f, f == 1, f in (0, 2),
           json.dumps(atoms), json.dumps([v for row in xyz for v in row]),
           "[4,3]") for f in range(3)])
    _csv("reaction_intent.csv",
         ("intent_id", "curve_id", "direction", "reverse_of_intent_id",
          "intent_pair_id", "intent_underdetermined", "g_src", "g_tgt",
          "delta_b_intent", "cv_spec", "intent_role", "rules",
          "intent_version"),
         [("C1_fwd", "C1", "forward", "C1_rev", "p1", "false",
           json.dumps(g_src), json.dumps(g_tgt), "[]", json.dumps(cv_spec),
           "", "", "v3"),
          ("C1_rev", "C1", "reverse", "C1_fwd", "p1", "false",
           json.dumps(g_tgt), json.dumps(g_src), "[]", json.dumps(cv_spec),
           "", "", "v3")])
    _csv("irc_edge.csv",
         ("curve_id", "intent_id", "edge_index", "src_node_index",
          "dst_node_index", "delta_xyz_flat", "delta_xyz_shape",
          "delta_xyz_offsets", "atom_map_sha256", "delta_xyz_sha256",
          "units", "same_ref_frame"),
         [("C1", "C1_fwd", 0, 0, 1, json.dumps([0.0] * 12), "[4,3]",
           "[0,12)", hashlib.sha256(b"m").hexdigest(),
           hashlib.sha256(_struct.pack("<12d", *([0.0] * 12))).hexdigest(),
           "angstrom", "true")])
    _csv("curve_quality.csv", ("curve_id", "quality_status"), [("C1", "ok")])
    rows = []
    for f in range(3):
        for iid in ("C1_fwd", "C1_rev"):
            rows.append((
                f"C1:{f}:{iid}:0.08", "C1", f, iid, 0.08, 0.0, 0.0, 0.0,
                1.0, 0.5 if f == 0 else 0.0, 1.0, "ok", "f" * 64, "l" * 64,
                "false", "local_frame", 1,
                json.dumps([0.01, -0.005, 0.0, 0.02, 0.0, 0.0]), 0.0, 0,
                "false", 0, "train_eligible"))
    _csv("sample_view.csv",
         ("sample_id", "curve_id", "node_index", "intent_id", "d0", "x", "y",
          "z", "label_weight", "curve_weight", "sample_weight", "label_status",
          "feature_sha", "label_sha", "sensitivity_only", "gauge_ref",
          "target_node_index", "target_action_u", "target_progress_q",
          "stop_label", "mask", "fold", "disposition"), rows)
    names = ("irc_curve.csv", "irc_node.csv", "irc_edge.csv",
             "reaction_intent.csv", "curve_quality.csv", "sample_view.csv")
    (out / "data_manifest.json").write_text(json.dumps({"files": [
        {"path": n, "sha256": hashlib.sha256((out / n).read_bytes()).hexdigest()}
        for n in names]}), encoding="utf-8")
    (out / "split_manifest_r3.json").write_text(json.dumps({
        "version": "r3_split_manifest_v2", "folds": [0, 1],
        "assignments": [{"curve_id": "C1", "fold": 0,
                         "near_dup_group": "C1"}]}), encoding="utf-8")
    return out


def test_e1_encoding_tables_load_end_to_end(tmp_path: Path) -> None:
    """Dict-graph g_src/g_tgt, coords_flat geometry, flat target_action_u and
    an assignments-style split manifest — the G0 E1 delivery encodings."""
    ds = KeyAtomDatasetV3(_write_e1_style_tables(tmp_path))
    assert len(ds) == 6
    assert ds.folds == {"C1": 0}
    rec = ds.samples[0]
    s = ds.to_sample_v3(rec)
    assert s.atomic_numbers == (6, 8, 1, 1)
    assert len(s.xyz) == 4                       # coords_flat [4,3]
    assert s.intent.key_rows() == (0, 1)         # C=O edit (+ cv atoms)
    assert s.y_teacher is not None and len(s.y_teacher) == 2  # flat u parse
    feats = build_sample_features_v3(s)
    assert feats["pos"].shape == (4, 3)


def test_derived_cv_deltas_follow_contract_direction_rule() -> None:
    """Δξ = −ΔB for bond-length CVs without explicit target_deltas
    (contract §3.1: breaking ⇒ distance grows, forming ⇒ shrinks)."""
    row = {
        "intent_id": "i", "curve_id": "c", "direction": "forward",
        "reverse_of_intent_id": "r", "intent_pair_id": "p",
        "intent_underdetermined": "false",
        "g_src": json.dumps({"atoms": [6, 8], "bonds": [[0, 1, 1.0]]}),
        "g_tgt": json.dumps({"atoms": [6, 8], "bonds": [[0, 1, 2.0]]}),
        "cv_spec": json.dumps({"coordinates": [[0, 1]], "kind":
                               "graph_edit_bond_lengths"}),
    }
    from pes2ts_core.pesn.contracts.key_atom_v3 import IntentRecordV3

    intent = IntentRecordV3.from_table_row(row)
    assert intent.cv_deltas == (-1.0,)           # forming ⇒ Δξ < 0
    assert intent.signed_intent_sign() == -1.0
    assert intent.reversed_intent().cv_deltas == (1.0,)


def test_explicit_cv_deltas_take_precedence() -> None:
    row = {
        "intent_id": "i", "curve_id": "c", "direction": "forward",
        "reverse_of_intent_id": "r", "intent_pair_id": "p",
        "g_src": json.dumps({"atoms": [6, 8], "bonds": [[0, 1, 1.0]]}),
        "g_tgt": json.dumps({"atoms": [6, 8], "bonds": [[0, 1, 2.0]]}),
        "cv_spec": json.dumps({"coordinates": [[0, 1]],
                               "target_deltas": [0.05, 0.1]}),
    }
    from pes2ts_core.pesn.contracts.key_atom_v3 import IntentRecordV3

    intent = IntentRecordV3.from_table_row(row)
    assert intent.cv_deltas == (0.05, 0.1)
    assert intent.signed_intent_sign() == 1.0
