"""V5.3-MP ``endpoint_label_spec_v1`` tests (plan §4.3 rules + §4.5 coverage).

Torch-free, numpy-only unit/integration tests on a synthetic six-table
fixture with CONTROLLED per-frame geometry (an SN2-like C–Br + O path
whose C–Br/C–O distances interpolate from the A start to the B target
endpoint).  Cover: target-endpoint positive, start-endpoint negative,
TS hard-negative, neighborhood positive (``d_end <= w_pos`` + completed
edits + legal geometry), middle-window Mask, terminal-only-at-endpoint,
degenerate-curve masking (auto-detected AND ``DEGENERATE_TS_CURVES``
union), the terminal-less load refusal, the plan §4.5 coverage report,
``bond_edit_complete`` failing on a still-long forming bond, collision
(``geometry_legal`` False) vetoing a near-endpoint positive, and the
diagnostic-only energy flag.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import KeyAtomDatasetV3
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
    DEGENERATE_TS_CURVES,
    ENDPOINT_LABEL_SPEC_V1,
    TS_HARD_NEGATIVE_WINDOW_ANGSTROM_V1,
    W_NEG_ANGSTROM_V1,
    W_POS_ANGSTROM_V1,
    EndpointLabelSpecV1Error,
    EndpointLabelTableV1,
    build_endpoint_labels_spec_v1,
    bond_edit_complete_v1,
    geometry_legal_v1,
    signed_edit_pairs_v1,
)

ALL_STATUSES = ("ok", "terminal", "masked", "conflict")

# ---------------------------------------------------------------------------
# Synthetic six-table fixture with controlled per-frame geometry
# ---------------------------------------------------------------------------
ATOMIC_NUMBERS = (6, 35, 8, 1, 1, 1)          # C Br O H H H
#: interpolation parameters t per node: A start (0.0) … TS (0.5) … B (1.0)
T_VALUES = (0.0, 0.2, 0.4, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0)
TS_INDEX = 3                                   # t = 0.5
NODE_B = len(T_VALUES) - 1                     # target endpoint of forward
O_DIR = np.array([-0.8, 0.6, 0.0])
H_FIXED = np.array([[0.2, 1.0, 0.2], [0.2, -0.9, 0.5], [-0.3, -0.2, -1.0]])
G_SRC_BONDS = ((0, 1, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0))
G_TGT_BONDS = ((0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0), (0, 5, 1.0))


def _frame(t: float, *, collide: bool = False) -> np.ndarray:
    """One fixture frame: C–Br stretches 1.9→3.6 Å, C–O shrinks 3.4→1.4 Å."""
    xyz = np.zeros((6, 3))
    xyz[1] = np.array([1.9 + 1.7 * t, 0.0, 0.0])        # Br along +x
    xyz[2] = O_DIR * (3.4 - 2.0 * t)                    # O along O_DIR
    xyz[3:6] = H_FIXED
    if collide:
        xyz[5] = np.array([-0.15, -0.1, -0.5])          # H on top of C: 0.53 Å
    return xyz


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        writer.writerows(rows)


def _write_fixture(
    tables_dir: Path,
    curves: list[dict],
    energies: dict[str, list[float]] | None = None,
) -> None:
    """Write six tables (no manifests) with per-curve control dicts.

    Each curve dict: ``curve_id``, ``fold``, optional ``collide_nodes``,
    ``degenerate`` (TS frame := B frame) and ``statuses`` mapping
    ``(node, direction) -> label_status``.
    """
    out = Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(T_VALUES)
    atomic_json = json.dumps(list(ATOMIC_NUMBERS))

    curve_rows, node_rows, intent_rows, edge_rows, quality_rows, sample_rows = \
        [], [], [], [], [], []
    for spec in curves:
        cid = spec["curve_id"]
        fold = spec.get("fold", 0)
        collide_nodes = set(spec.get("collide_nodes", ()))
        statuses = spec.get("statuses", {})
        curve_rows.append((cid, f"canon::{cid}", n, 2 * (n - 1), TS_INDEX,
                           0, n - 1))
        for node, t in enumerate(T_VALUES):
            if spec.get("degenerate") and node == TS_INDEX:
                xyz = _frame(1.0)              # TS frame := B frame
            else:
                xyz = _frame(t, collide=node in collide_nodes)
            energy = ""
            if energies and cid in energies:
                energy = repr(energies[cid][node])
            node_rows.append((
                cid, node, json.dumps(xyz.tolist()), 0.0, 0.0,
                -1.0 + 2.0 * node / max(n - 1, 1),
                node == TS_INDEX, node in (0, n - 1), atomic_json, energy))
        g_src = json.dumps([list(b) for b in G_SRC_BONDS])
        g_tgt = json.dumps([list(b) for b in G_TGT_BONDS])
        cv_spec = json.dumps({
            "target_deltas": [0.05, -0.05],
            "coordinates": [[0, 1], [0, 2]],
            "kind": "graph_edit_bond_lengths"})
        intents = []
        for direction, iid, rev in (("forward", f"{cid}::fwd", f"{cid}::rev"),
                                    ("reverse", f"{cid}::rev", f"{cid}::fwd")):
            intents.append((iid, direction, rev))
            intent_rows.append((
                iid, cid, direction, rev, f"{cid}::pair", "false",
                g_src if direction == "forward" else g_tgt,
                g_tgt if direction == "forward" else g_src,
                "[]", cv_spec, "", "", "v3"))
            edge_rows.append((
                cid, iid, 0, 0, 1, json.dumps([0.0] * 9), "[3,3]", "[0,9)",
                "0" * 64, "1" * 64, "angstrom", "true"))
        quality_rows.append((cid, "ok"))

        n_samples = n * 2
        cw = 1.0 / n_samples
        target_u = json.dumps([0.01, 0.0, 0.0, 0.02, 0.0, 0.0,
                               -0.03, 0.0, 0.0])
        for node in range(n):
            for iid, direction, _rev in intents:
                status = statuses.get((node, direction), "ok")
                sample_rows.append((
                    f"{cid}:{node}:{iid}:0.08", cid, node, iid, 0.08,
                    0.0, 0.0, 0.0, 1.0, cw, 1.0, status,
                    "f" * 64, "l" * 64, "false", "local_frame", 2,
                    target_u, 0.0, 0, "false", fold))

    _write_csv(out / "irc_curve.csv",
               ("curve_id", "canonical_irc_id", "n_nodes", "n_edges",
                "ts_index", "endpoint_a_index", "endpoint_b_index"),
               curve_rows)
    _write_csv(out / "irc_node.csv",
               ("curve_id", "node_index", "x", "y", "z", "progress",
                "is_ts", "is_endpoint", "atomic_numbers", "energy"),
               node_rows)
    _write_csv(out / "reaction_intent.csv",
               ("intent_id", "curve_id", "direction", "reverse_of_intent_id",
                "intent_pair_id", "intent_underdetermined", "g_src", "g_tgt",
                "delta_b_intent", "cv_spec", "intent_role", "rules",
                "intent_version"),
               intent_rows)
    _write_csv(out / "irc_edge.csv",
               ("curve_id", "intent_id", "edge_index", "src_node_index",
                "dst_node_index", "delta_xyz_flat", "delta_xyz_shape",
                "delta_xyz_offsets", "atom_map_sha256", "delta_xyz_sha256",
                "units", "same_ref_frame"),
               edge_rows)
    _write_csv(out / "curve_quality.csv",
               ("curve_id", "quality_status"), quality_rows)
    _write_csv(out / "sample_view.csv",
               ("sample_id", "curve_id", "node_index", "intent_id", "d0",
                "x", "y", "z", "label_weight", "curve_weight",
                "sample_weight", "label_status", "feature_sha", "label_sha",
                "sensitivity_only", "gauge_ref", "target_node_index",
                "target_action_u", "target_progress_q", "stop_label", "mask",
                "fold"),
               sample_rows)


CURVE_FAMILIES = {
    "curve_main": "famA",
    "curve_deg": "famA",
    "RXN_0000001174": "famB",
}


@pytest.fixture()
def fixture_dir(tmp_path: Path) -> Path:
    _write_fixture(tmp_path / "data", [
        {"curve_id": "curve_main", "fold": 0,
         "collide_nodes": (7,),
         "statuses": {(8, "forward"): "terminal",
                      (2, "forward"): "terminal"}},
        {"curve_id": "curve_deg", "fold": 1, "degenerate": True},
        # union-forced degeneracy: normal geometry, but the curve id is
        # listed in DEGENERATE_TS_CURVES ⇒ every row must be Mask+flagged
        {"curve_id": "RXN_0000001174", "fold": 0},
    ])
    return tmp_path / "data"


@pytest.fixture()
def full_dataset(fixture_dir: Path) -> KeyAtomDatasetV3:
    return KeyAtomDatasetV3(fixture_dir, label_statuses=ALL_STATUSES)


@pytest.fixture()
def label_table(full_dataset: KeyAtomDatasetV3) -> EndpointLabelTableV1:
    return build_endpoint_labels_spec_v1(
        full_dataset, full_dataset.v52_labels(),
        curve_families=CURVE_FAMILIES)


def _lab(table: EndpointLabelTableV1, cid: str, node: int,
         direction: str):
    matches = [l for l in table.by_sample.values()
               if l.curve_id == cid and l.node_index == node
               and l.direction == direction]
    assert len(matches) == 1
    return matches[0]


# ---------------------------------------------------------------------------
# plan §4.3 rule-by-rule assertions (curve_main, forward = A→B)
# ---------------------------------------------------------------------------
def test_target_endpoint_frame_positive(label_table) -> None:
    fwd = _lab(label_table, "curve_main", NODE_B, "forward")
    assert fwd.endpoint_label == 1
    assert fwd.d_end_angstrom == pytest.approx(0.0, abs=1e-12)
    assert fwd.start_row is False and fwd.ts_row is False
    rev = _lab(label_table, "curve_main", 0, "reverse")
    assert rev.endpoint_label == 1
    assert rev.d_end_angstrom == pytest.approx(0.0, abs=1e-12)


def test_start_endpoint_negative(label_table) -> None:
    fwd = _lab(label_table, "curve_main", 0, "forward")
    assert fwd.endpoint_label == 0 and fwd.start_row is True
    rev = _lab(label_table, "curve_main", NODE_B, "reverse")
    assert rev.endpoint_label == 0 and rev.start_row is True


def test_ts_row_always_negative(label_table) -> None:
    for direction in ("forward", "reverse"):
        ts = _lab(label_table, "curve_main", TS_INDEX, direction)
        assert ts.ts_row is True
        assert ts.endpoint_label == 0
        assert ts.d_key_ts_angstrom <= TS_HARD_NEGATIVE_WINDOW_ANGSTROM_V1


def test_neighborhood_positive(label_table) -> None:
    near = _lab(label_table, "curve_main", 6, "forward")   # t = 0.90
    assert near.d_end_angstrom <= W_POS_ANGSTROM_V1
    assert near.bond_edit_complete is True
    assert near.geometry_legal is True
    assert near.endpoint_label == 1


def test_middle_window_mask(label_table) -> None:
    mid = _lab(label_table, "curve_main", 4, "forward")    # t = 0.70
    assert W_POS_ANGSTROM_V1 < mid.d_end_angstrom <= W_NEG_ANGSTROM_V1
    assert mid.endpoint_label == -1


def test_terminal_positive_only_at_endpoint(label_table) -> None:
    at_endpoint = _lab(label_table, "curve_main", NODE_B, "forward")
    assert at_endpoint.terminal_row is True
    assert at_endpoint.endpoint_label == 1
    mid_terminal = _lab(label_table, "curve_main", 2, "forward")
    assert mid_terminal.terminal_row is True
    assert mid_terminal.d_end_angstrom > W_POS_ANGSTROM_V1
    assert mid_terminal.endpoint_label == -1     # terminal ≠ endpoint


def test_degenerate_curves_masked_and_flagged(label_table) -> None:
    degenerate_ids = {"curve_deg", "RXN_0000001174"}
    assert set(DEGENERATE_TS_CURVES) >= {"RXN_0000001174"}
    for lab in label_table.by_sample.values():
        if lab.curve_id in degenerate_ids:
            assert lab.degenerate_curve is True
            assert lab.endpoint_label == -1
        else:
            assert lab.degenerate_curve is False
    # auto-detection: curve_deg is NOT in DEGENERATE_TS_CURVES but its
    # TS frame sits at d_end = 0 of the B endpoint
    ts = _lab(label_table, "curve_deg", TS_INDEX, "forward")
    assert ts.d_end_angstrom <= W_POS_ANGSTROM_V1


def test_builder_refuses_dataset_without_terminal(
        fixture_dir: Path) -> None:
    ok_only = KeyAtomDatasetV3(fixture_dir, label_statuses=("ok",))
    v52 = ok_only.v52_labels()
    with pytest.raises(EndpointLabelSpecV1Error) as exc:
        build_endpoint_labels_spec_v1(ok_only, v52)
    assert exc.value.code == "TERMINAL_NOT_LOADED"


def test_builder_refuses_partial_status_set(fixture_dir: Path) -> None:
    no_conflict = KeyAtomDatasetV3(
        fixture_dir, label_statuses=("ok", "terminal", "masked"))
    v52 = no_conflict.v52_labels()
    with pytest.raises(EndpointLabelSpecV1Error) as exc:
        build_endpoint_labels_spec_v1(no_conflict, v52)
    assert exc.value.code == "STATUS_SET_NOT_LOADED"


def test_bond_edit_incomplete_when_forming_bond_long(label_table) -> None:
    # t=0.70: C–O = 2.00 Å > 1.40 + 0.30 tol ⇒ forming edit incomplete
    mid = _lab(label_table, "curve_main", 4, "forward")
    assert mid.bond_edit_complete is False
    near = _lab(label_table, "curve_main", 6, "forward")
    assert near.bond_edit_complete is True


def test_collision_frame_masked_despite_near_endpoint(label_table) -> None:
    # t=0.95, H forced onto C: d_end <= w_pos and edits complete, but the
    # frame collides ⇒ geometry_legal False ⇒ Mask (plan §4.3 Mask 2)
    hit = _lab(label_table, "curve_main", 7, "forward")
    assert hit.d_end_angstrom <= W_POS_ANGSTROM_V1
    assert hit.bond_edit_complete is True
    assert hit.geometry_legal is False
    assert hit.endpoint_label == -1
    healthy = _lab(label_table, "curve_main", 7, "reverse")
    assert healthy.geometry_legal is False    # legality is intent-free


# ---------------------------------------------------------------------------
# plan §4.5 coverage / hard-negative report
# ---------------------------------------------------------------------------
def test_report_counts_and_coverage(label_table) -> None:
    rep = label_table.report()
    assert rep["spec"] == ENDPOINT_LABEL_SPEC_V1
    assert rep["n_rows"] == 54                     # 3 curves × 9 nodes × 2
    assert rep["n_pos"] == 3 and rep["n_neg"] == 10 and rep["n_mask"] == 41
    assert rep["n_pos"] + rep["n_neg"] + rep["n_mask"] == rep["n_rows"]
    assert rep["n_curves"] == 3
    assert rep["per_curve_positive"] == {
        "RXN_0000001174": 0, "curve_deg": 0, "curve_main": 3}
    assert rep["n_curves_with_positive"] == 1
    assert rep["effective_positive_coverage"] == pytest.approx(1 / 3)
    assert rep["degenerate_curves"] == ["RXN_0000001174", "curve_deg"]
    # TS-window rows: curve_main TS (2 intents) + RXN TS (2 intents) +
    # curve_deg TS and B endpoint (its TS frame IS the B frame, so the B
    # node has d_key_ts = 0 as well); only curve_main's TS rows may be
    # negative — degenerate curves mask theirs
    assert rep["n_ts_window_rows"] == 8
    assert rep["n_ts_negative"] == 2
    assert rep["n_start_rows"] == 6
    assert rep["n_start_negative"] == 2
    assert rep["n_terminal_rows"] == 2
    assert rep["n_terminal_positive"] == 1
    assert rep["per_fold"] == {
        "0": {"n_rows": 36, "n_pos": 3, "n_neg": 10, "n_mask": 23},
        "1": {"n_rows": 18, "n_pos": 0, "n_neg": 0, "n_mask": 18},
    }
    assert rep["per_family"]["famA"] == {
        "n_curves": 2, "n_curves_with_positive": 1,
        "effective_positive_coverage": 0.5,
        "n_rows": 36, "n_pos": 3, "n_neg": 10, "n_mask": 23}
    assert rep["per_family"]["famB"]["effective_positive_coverage"] == 0.0
    # deterministic ordering
    rows = label_table.to_rows()
    assert [r["sample_id"] for r in rows] == sorted(
        r["sample_id"] for r in rows)


def test_report_without_families(full_dataset) -> None:
    table = build_endpoint_labels_spec_v1(
        full_dataset, full_dataset.v52_labels())
    assert table.report()["per_family"] is None


# ---------------------------------------------------------------------------
# diagnostic-only energy flag (plan §4.3 正例 3; v1: never changes labels)
# ---------------------------------------------------------------------------
def test_energy_flag_none_without_column(full_dataset) -> None:
    table = build_endpoint_labels_spec_v1(
        full_dataset, full_dataset.v52_labels())
    assert all(l.energy_ev_within_window is None
               for l in table.by_sample.values())


def test_energy_flag_computed_but_labels_unchanged(tmp_path) -> None:
    curve_specs = [
        {"curve_id": "curve_main", "fold": 0,
         "statuses": {(8, "forward"): "terminal",
                      (2, "forward"): "terminal"}},
        {"curve_id": "curve_deg", "fold": 1, "degenerate": True},
        {"curve_id": "RXN_0000001174", "fold": 0},
    ]
    _write_fixture(tmp_path / "plain", curve_specs)
    plain = KeyAtomDatasetV3(tmp_path / "plain", label_statuses=ALL_STATUSES)
    plain_table = build_endpoint_labels_spec_v1(plain, plain.v52_labels())

    # E(t) = -100 + 0.01·(1−t) Hartree ⇒ node0 ΔE = 6.28 kcal/mol (out),
    # node1 ≈ 5.02 (out), node6 ≈ 0.63 (in), node8 = 0 (in)
    energies = {"curve_main": [-100.0 + 0.01 * (1 - t) for t in T_VALUES]}
    _write_fixture(tmp_path / "energetic", curve_specs, energies=energies)
    ener = KeyAtomDatasetV3(tmp_path / "energetic",
                            label_statuses=ALL_STATUSES)
    ener_table = build_endpoint_labels_spec_v1(ener, ener.v52_labels())

    assert _lab(ener_table, "curve_main", 0, "forward") \
        .energy_ev_within_window is False
    assert _lab(ener_table, "curve_main", 1, "forward") \
        .energy_ev_within_window is False
    assert _lab(ener_table, "curve_main", 6, "forward") \
        .energy_ev_within_window is True
    assert _lab(ener_table, "curve_main", NODE_B, "forward") \
        .energy_ev_within_window is True
    # curve_deg ships no energies ⇒ diagnostic stays None
    assert _lab(ener_table, "curve_deg", 4, "forward") \
        .energy_ev_within_window is None
    # DIAGNOSTIC-ONLY: labels are bit-identical with vs without energies
    assert {sid: l.endpoint_label
            for sid, l in plain_table.by_sample.items()} == \
           {sid: l.endpoint_label
            for sid, l in ener_table.by_sample.items()}


# ---------------------------------------------------------------------------
# unit-level helpers
# ---------------------------------------------------------------------------
def test_geometry_legal_v1_collision_and_nonfinite() -> None:
    good = _frame(0.5)
    assert geometry_legal_v1(good) is True
    assert geometry_legal_v1(_frame(0.5, collide=True)) is False
    bad = good.copy()
    bad[0, 0] = np.nan
    assert geometry_legal_v1(bad) is False


def test_signed_edit_pairs_v1_forward_and_reverse() -> None:
    from pes2ts_core.pesn.contracts.key_atom_v3 import IntentRecordV3
    row = {
        "intent_id": "i", "curve_id": "c", "direction": "forward",
        "reverse_of_intent_id": "r", "intent_pair_id": "p",
        "intent_underdetermined": "false",
        "g_src": json.dumps({"atoms": list(ATOMIC_NUMBERS),
                             "bonds": [list(b) for b in G_SRC_BONDS]}),
        "g_tgt": json.dumps({"atoms": list(ATOMIC_NUMBERS),
                             "bonds": [list(b) for b in G_TGT_BONDS]}),
        "cv_spec": json.dumps({"target_deltas": [0.05, -0.05],
                               "coordinates": [[0, 1], [0, 2]]}),
    }
    fwd = IntentRecordV3.from_table_row(row)
    edits = signed_edit_pairs_v1(fwd)
    # (0,1) breaking C–Br; (0,2) forming C–O; C–H spectators excluded
    assert [(i, j, d) for i, j, _s, _t, d in edits] == [(0, 1, -1.0),
                                                        (0, 2, 1.0)]
    near_b = _frame(0.9)
    assert bond_edit_complete_v1(fwd, near_b, _frame(1.0)) is True
    assert bond_edit_complete_v1(fwd, _frame(0.4), _frame(1.0)) is False
    # reverse intent swaps forming/breaking roles
    rev = fwd.reversed_intent()
    assert [(i, j, d) for i, j, _s, _t, d in signed_edit_pairs_v1(rev)] \
        == [(0, 1, 1.0), (0, 2, -1.0)]
    assert bond_edit_complete_v1(rev, _frame(0.1), _frame(0.0)) is True
