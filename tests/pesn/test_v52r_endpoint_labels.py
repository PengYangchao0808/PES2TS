"""V5.2-R D0 endpoint-stop label tests (schema pes2ts_fullpath_nav_labels_v1).

Unit tests are numpy-only; real-data tests load the PD-1 six-table delivery
under ``outputs/pesn/30h_campaign/data`` (skipped gracefully when absent)
with ALL label statuses and re-verify the acceptance numbers of the V5.2
task-correction audit 20261021 (contract §5.2/§5.3; V5.2-R plan §2/§3 D0).
"""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import KeyAtomDatasetV3
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
    DEGENERATE_TS_CURVES,
    ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM,
    ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM,
    EndpointStopLabelError,
    EndpointStopLabelTable,
    build_endpoint_stop_labels,
    endpoint_stop_label_from_distance,
    target_endpoint_node,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "outputs/pesn/30h_campaign/data"
HAVE_TABLES = TABLES_DIR.is_dir()

ALL_STATUSES = ("ok", "terminal", "masked", "conflict")


# ----------------------------------------------------------------- unit tests


def test_window_edges_inclusive() -> None:
    """0.10 → 1 and 0.30 → 0 are inclusive; ±1e-9 falls to Mask."""
    w_pos = ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM
    w_neg = ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM
    assert endpoint_stop_label_from_distance(w_pos) == 1
    assert endpoint_stop_label_from_distance(w_pos + 1e-9) == -1
    assert endpoint_stop_label_from_distance(w_neg) == 0
    assert endpoint_stop_label_from_distance(w_neg - 1e-9) == -1
    assert endpoint_stop_label_from_distance(0.0) == 1
    assert endpoint_stop_label_from_distance(10.0) == 0


def test_window_order_violation_raises() -> None:
    with pytest.raises(EndpointStopLabelError) as exc:
        endpoint_stop_label_from_distance(
            0.1, positive_window=0.30, negative_window=0.10)
    assert exc.value.code == "ENDPOINT_WINDOW_ORDER"


def test_target_endpoint_node_from_order() -> None:
    order = tuple(range(5))
    assert target_endpoint_node(order, "forward") == 4
    assert target_endpoint_node(order, "reverse") == 0

    class _FakeCurve:
        node_order = (3, 2, 1, 0)      # reversed path order

    assert target_endpoint_node(_FakeCurve(), "forward") == 0
    assert target_endpoint_node(_FakeCurve(), "reverse") == 3
    with pytest.raises(EndpointStopLabelError):
        target_endpoint_node(order, "sideways")


# --------------------------------------------------------------- real-data


@pytest.fixture(scope="module")
def full_dataset() -> KeyAtomDatasetV3:
    if not HAVE_TABLES:
        pytest.skip("six-table delivery not present")
    return KeyAtomDatasetV3(TABLES_DIR, label_statuses=ALL_STATUSES)


@pytest.fixture(scope="module")
def stop_table(full_dataset: KeyAtomDatasetV3) -> EndpointStopLabelTable:
    return build_endpoint_stop_labels(full_dataset, full_dataset.v52_labels())


def test_build_refuses_dataset_without_terminal(
        full_dataset: KeyAtomDatasetV3) -> None:
    """Core bug fix: a terminal-less load MUST be refused."""
    ok_only = KeyAtomDatasetV3(TABLES_DIR, label_statuses=("ok",))
    v52 = ok_only.v52_labels()
    with pytest.raises(EndpointStopLabelError) as exc:
        build_endpoint_stop_labels(ok_only, v52)
    assert exc.value.code == "TERMINAL_NOT_LOADED"


@pytest.mark.skipif(not HAVE_TABLES, reason="six-table delivery not present")
def test_total_rows_and_terminal_inclusion(
        stop_table: EndpointStopLabelTable) -> None:
    print(f"actual rows: {stop_table.n_rows}")
    assert stop_table.n_rows == 12034
    # terminal (no_future_contact) rows must be present in the Stop table
    assert stop_table.count_terminal_rows() == 1123
    assert stop_table.count_move_status("ok") == 9655


@pytest.mark.skipif(not HAVE_TABLES, reason="six-table delivery not present")
def test_window_label_counts(
        stop_table: EndpointStopLabelTable) -> None:
    """0.10/0.30 → positives ≈1781 (1770–1795), negatives ≈8399, mask ≈1854."""
    print(f"pos/neg/mask = {stop_table.n_positive}/{stop_table.n_negative}"
          f"/{stop_table.n_mask}")
    assert 1770 <= stop_table.n_positive <= 1795
    assert 8380 <= stop_table.n_negative <= 8410
    assert 1840 <= stop_table.n_mask <= 1870
    assert stop_table.n_positive + stop_table.n_negative + stop_table.n_mask \
        == stop_table.n_rows


@pytest.mark.skipif(not HAVE_TABLES, reason="six-table delivery not present")
def test_start_endpoint_false_positives(
        stop_table: EndpointStopLabelTable) -> None:
    """Departing rows (forward@A, reverse@B) must never be labeled 1."""
    start_rows = [
        lab for lab in stop_table.by_sample.values()
        if (lab.direction == "forward"
            and lab.node_index == stop_table.curves[lab.curve_id].endpoint_a_node)
        or (lab.direction == "reverse"
            and lab.node_index == stop_table.curves[lab.curve_id].endpoint_b_node)]
    assert len(start_rows) == 100
    fps = [lab for lab in start_rows if lab.endpoint_stop_label == 1]
    print(f"start FP = {len(fps)}; "
          f"d_end min = {min(l.d_end_angstrom for l in start_rows):.3f}")
    assert fps == []
    assert min(l.d_end_angstrom for l in start_rows) >= \
        ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM


@pytest.mark.skipif(not HAVE_TABLES, reason="six-table delivery not present")
def test_ts_false_positives_are_only_flagged_degenerates(
        stop_table: EndpointStopLabelTable) -> None:
    """Exactly the 2 degenerate curves' TS rows may be positive; all TS rows
    of those curves carry degenerate_flag (≤4 rows)."""
    ts_rows = [
        lab for lab in stop_table.by_sample.values()
        if lab.node_index == stop_table.curves[lab.curve_id].ts_index]
    assert len(ts_rows) == 100
    ts_fp = [lab for lab in ts_rows if lab.endpoint_stop_label == 1]
    fp_curves = sorted({lab.curve_id for lab in ts_fp})
    fp_detail = [(lab.curve_id, lab.direction, round(lab.d_end_angstrom, 3))
                 for lab in ts_fp]
    print(f"TS FP: {fp_detail}")
    assert fp_curves == sorted(DEGENERATE_TS_CURVES)
    for lab in ts_fp:
        assert lab.degenerate_flag
    degenerate_rows = [
        lab for lab in stop_table.by_sample.values() if lab.degenerate_flag]
    assert len(degenerate_rows) <= 4
    assert {lab.curve_id for lab in degenerate_rows} == set(DEGENERATE_TS_CURVES)
    assert all(lab.node_index == stop_table.curves[lab.curve_id].ts_index
               for lab in degenerate_rows)


@pytest.mark.skipif(not HAVE_TABLES, reason="six-table delivery not present")
def test_intent_contrast_at_endpoints(
        stop_table: EndpointStopLabelTable) -> None:
    """Arriving intent (forward@B, reverse@A) → 1; departing → 0."""
    by_key = {
        (lab.curve_id, lab.node_index, lab.direction): lab
        for lab in stop_table.by_sample.values()}
    violations = []
    for cid, c in stop_table.curves.items():
        checks = (
            (c.endpoint_b_node, "forward", 1),
            (c.endpoint_b_node, "reverse", 0),
            (c.endpoint_a_node, "forward", 0),
            (c.endpoint_a_node, "reverse", 1),
        )
        for node, direction, expect in checks:
            lab = by_key[(cid, node, direction)]
            if lab.endpoint_stop_label != expect:
                violations.append((cid, node, direction,
                                   lab.endpoint_stop_label, expect))
    print(f"contrast violations: {violations[:5]} (total {len(violations)})")
    assert violations == []


@pytest.mark.skipif(not HAVE_TABLES, reason="six-table delivery not present")
def test_move_status_mapping_counts(
        stop_table: EndpointStopLabelTable) -> None:
    assert stop_table.count_move_status("ok") == 9655
    assert stop_table.count_move_status("no_future_contact") == 1123
    assert stop_table.count_move_status("masked") == 1188
    assert stop_table.count_move_status("conflict") == 68


@pytest.mark.skipif(not HAVE_TABLES, reason="six-table delivery not present")
def test_per_curve_positive_stats(
        stop_table: EndpointStopLabelTable) -> None:
    vals = sorted(stop_table.positive_counts_by_curve().values())
    med = statistics.median(vals)
    print(f"per-curve positives min/med/max = {vals[0]}/{med}/{vals[-1]}")
    assert (vals[0], med, vals[-1]) == (9, 32.0, 90)


@pytest.mark.skipif(not HAVE_TABLES, reason="six-table delivery not present")
def test_as_dict_roundtrip(stop_table: EndpointStopLabelTable) -> None:
    doc = stop_table.as_dict()
    assert doc["schema"] == "pes2ts_fullpath_nav_labels_v1"
    assert doc["n_rows"] == 12034
    assert len(doc["rows"]) == 12034
    assert doc["n_positive"] == stop_table.n_positive
    keys = [r["sample_id"] for r in doc["rows"]]
    assert keys == sorted(keys)
