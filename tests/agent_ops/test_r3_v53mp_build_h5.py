"""Tests for the V5.3-MP S0-B H5 build (r3_v53mp_build_tables_h5.py).

Synthetic H-transfer fixture: H0-H1 + H2 -> H0 + H1-H2 with the TS at
frame 0 and the archive's ``ts_first_dual_branch`` layout ([TS, seg1 -> end1,
TS', seg2 -> end2]).  Covers segment splitting, orientation (comparative
edit-direction score + G1 cross-check), canonical curve construction, teacher
involution, and the row emission for the six tables.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC = importlib.util.spec_from_file_location(
    "r3_v53mp_build_tables_h5",
    REPO_ROOT / "scripts/agent_ops/data_gate/r3_v53mp_build_tables_h5.py",
)
build_h5 = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault(SPEC.name, build_h5)
SPEC.loader.exec_module(build_h5)


# ---------------------------------------------------------------------------
# fixture: H-exchange IRC
# ---------------------------------------------------------------------------


def _interp(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None, None]
    return (1 - t) * a[None] + t * b[None]


def _fixture(n_seg: int = 30):
    """R: H0-H1 bonded (0.74 A), H2 far.  P: H1-H2 bonded, H0 far."""
    r_end = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [6.0, 0.0, 0.0]])
    ts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    p_end = np.array([[6.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    seg1 = _interp(ts, r_end, n_seg)          # TS -> R end
    seg2 = _interp(ts, p_end, n_seg)          # TS' -> P end
    coords = np.concatenate([seg1, seg2], axis=0)   # TS shared twice
    energies = np.concatenate([
        np.linspace(-1.0, -2.0, n_seg), np.linspace(-1.0, -2.1, n_seg)])
    p1_doc = {
        "mapping": {"map_to_atoms": [
            {"map": i + 1, "ts_irc_index": i, "element": "H"}
            for i in range(3)]},
        "bond_events": {"broken": [{"atoms": [1, 2], "order_r": 1.0,
                                    "order_p": None}],
                        "formed": [{"atoms": [2, 3], "order_r": None,
                                    "order_p": 1.0}]},
        "graph": {"elements": {"1": "H", "2": "H", "3": "H"},
                  "r_bonds": [[1, 2, 1.0]],
                  "p_bonds": [[2, 3, 1.0]]},
    }
    ts_info = {"charge": 0, "multiplicity": 1, "atomic_numbers": [1, 1, 1]}
    return coords, energies, p1_doc, ts_info


def _task_dict(tmp_path: Path, **over):
    coords, energies, p1_doc, ts_info = _fixture()
    p1_path = tmp_path / "RXN_0000000001.json"
    p1_path.write_text(json.dumps(p1_doc), encoding="utf-8")
    task = {
        "rid": "RXN_0000000001",
        "coordinates": coords,
        "energies": energies,
        "atomic_numbers": (1, 1, 1),
        "p1_doc": p1_doc,
        "p1_path": str(p1_path),
        "ts_info": ts_info,
        "fold": 0,
        "g1_orientation": "R_first",
        "frames_sha256": "0" * 64,
        "energies_sha256": "0" * 64,
    }
    task.update(over)
    return task


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_split_segments_happy(tmp_path):
    coords, energies, _, _ = _fixture()
    brk, seg1, seg2 = build_h5._split_segments(coords)
    assert brk == 29
    assert seg1.shape[0] == 30 and seg2.shape[0] == 30
    np.testing.assert_allclose(seg1[0], coords[0])


def test_split_segments_rejects_no_break(tmp_path):
    coords = np.cumsum(np.ones((40, 3, 3)) * 0.01, axis=0)
    with pytest.raises(build_h5.BuildError) as exc:
        build_h5._split_segments(coords)
    assert exc.value.reason == "segment_break_not_resolved"


def test_split_segments_rejects_short(tmp_path):
    coords, _, _, _ = _fixture()
    with pytest.raises(build_h5.BuildError) as exc:
        build_h5._split_segments(coords[:4])
    assert exc.value.reason == "too_few_frames"


def test_process_curve_structure(tmp_path):
    out = build_h5._process_curve(_task_dict(tmp_path))
    rows = out["rows"]
    assert out["excluded"] is None
    # 59 = 30 (R side) + 29 (P side past the shared TS)
    assert out["ledger"]["n_nodes"] == 59
    assert out["ledger"]["ts_index"] == 29
    assert len(rows["irc_node.csv"]) == 59
    assert len(rows["sample_view.csv"]) == 118
    assert len(rows["reaction_intent.csv"]) == 2
    assert len(rows["irc_edge.csv"]) == 2 * 58
    # orientation evidence recorded
    assert out["ledger"]["orientation"]["method"] == "edit_direction_comparative"
    assert out["ledger"]["orientation"]["g1_disagreement"] is False


def test_process_curve_orientation_g1_conflict_refused(tmp_path):
    task = _task_dict(tmp_path, g1_orientation="P_first")
    with pytest.raises(build_h5.BuildError) as exc:
        build_h5._process_curve(task)
    assert exc.value.reason == "orientation_g1_conflict"


def test_process_curve_z_mismatch_refused(tmp_path):
    task = _task_dict(tmp_path)
    task["ts_info"] = dict(task["ts_info"], atomic_numbers=[1, 1, 8])
    with pytest.raises(build_h5.BuildError) as exc:
        build_h5._process_curve(task)
    assert exc.value.reason == "z_mismatch_h5_ts"


def test_teacher_terminal_structure(tmp_path):
    """forward terminal at the P end, reverse terminal at the R end."""
    out = build_h5._process_curve(_task_dict(tmp_path))
    rows = [json.loads(r) for r in []]  # placeholder
    # parse sample_view fragment rows (CSV text)
    import csv as _csv
    import io as _io
    sample_rows = list(_csv.reader(
        _io.StringIO("".join(out["rows"]["sample_view.csv"]))))
    header = None
    recs = {}
    for row in sample_rows:
        sid, curve_id, node, intent_id = row[0], row[1], int(row[2]), row[3]
        status = row[11]
        recs[(node, intent_id)] = status
    n = 59
    assert recs[(0, "RXN_0000000001_rev")] == "terminal"
    assert recs[(n - 1, "RXN_0000000001_fwd")] == "terminal"
    assert recs[(0, "RXN_0000000001_fwd")] == "ok"
    assert recs[(n - 1, "RXN_0000000001_rev")] == "ok"


def test_edge_involution(tmp_path):
    """reverse irc_edge deltas are exact negations of forward deltas."""
    out = build_h5._process_curve(_task_dict(tmp_path))
    import csv as _csv
    import io as _io
    edge_rows = list(_csv.reader(
        _io.StringIO("".join(out["rows"]["irc_edge.csv"]))))
    fwd = [r for r in edge_rows if r[1].endswith("_fwd")]
    rev = [r for r in edge_rows if r[1].endswith("_rev")]
    assert len(fwd) == len(rev) == 58
    for fr, rr in zip(fwd, rev):
        d_f = np.asarray(json.loads(fr[5]))
        d_r = np.asarray(json.loads(rr[5]))
        np.testing.assert_allclose(d_f, -d_r, atol=1e-6)
        # PD-9: reverse edge goes dst<-src reversed
        assert int(fr[3]) == int(rr[4]) and int(fr[4]) == int(rr[3])


def test_build_error_is_typed():
    err = build_h5.BuildError("reason_code", "detail")
    assert err.reason == "reason_code"
    assert "detail" in str(err)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
