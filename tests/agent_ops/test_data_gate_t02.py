"""ORA-2 gate tests for the R3 data gate round 2 (WP-D2), schema v2.

Covers the new machine gates T0.2a (intent involution), T0.2b (reversed
single-step teacher), T0.3a (d0), T0.3b (gauge), T0.5 (truth-access audit)
plus the PD-9 flat ``delta_xyz`` encoding checks inside T0.1c and the
label/weight invariants inside T0.1a — one fixture per blocker, each failing
*exactly* its expected test ids.  The round-1 contract tests live in
``test_data_schema_validator.py``.

Dry-run scope: no truth-data access, no ACP/engine calls, no network; fixture
mutation happens only under pytest ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops.data_gate import r3_table_schema as schema  # noqa: E402
from scripts.agent_ops.data_gate import validate_r3_tables as v  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "data_fixtures"
ALL_IDS = list(v.TEST_IDS)

# fixture -> exactly the gate ids that must fail (all others must pass)
NEW_VIOLATION_FIXTURES = [
    ("broken_involution", {"T0.2a"}),
    ("reverse_teacher_mismatch", {"T0.2b"}),
    ("weight_sum_mismatch", {"T0.1a"}),
    ("legacy_delta_columns", {"T0.1a", "T0.1c", "T0.2b"}),
    ("d0_unflagged", {"T0.3a"}),
    ("gauge_unresolved", {"T0.3b"}),
]


def statuses(report):
    return {t["id"]: t["status"] for t in report["tests"]}


def get_test(report, test_id):
    return next(t for t in report["tests"] if t["id"] == test_id)


def copy_fixture(fixture: str, tmp_path: Path, name: str = "tables") -> Path:
    dst = tmp_path / name
    shutil.copytree(FIXTURES / fixture, dst)
    return dst


def edit_csv(tables_dir: Path, filename: str, fn) -> None:
    path = tables_dir / filename
    df = pd.read_csv(path)
    fn(df)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# schema v2 surface (PD-1 / PD-9 / ORA-2)
# ---------------------------------------------------------------------------

def test_schema_version_and_pd9_edge_columns():
    assert schema.SCHEMA_VERSION == "r3_6table_v2"
    assert schema.VALIDATOR_VERSION.startswith("0.2")
    edge = schema.TABLES["irc_edge"]
    for col in ("delta_xyz_flat", "delta_xyz_shape", "delta_xyz_offsets",
                "atom_map_sha256", "delta_xyz_sha256", "units",
                "same_ref_frame", "intent_id"):
        assert col in edge.column_names, col
    for legacy in ("delta_dx", "delta_dy", "delta_dz"):
        assert legacy not in edge.column_names
    assert edge.primary_key == ("curve_id", "intent_id", "edge_index")


def test_schema_intent_pairing_and_weight_columns():
    ri = schema.TABLES["reaction_intent"]
    for col in ("reverse_of_intent_id", "intent_pair_id",
                "intent_underdetermined"):
        assert col in ri.column_names, col
    assert ("curve_id", "intent_id") in ri.unique
    sv = schema.TABLES["sample_view"]
    for col in ("label_weight", "curve_weight", "sample_weight",
                "label_status", "feature_sha", "label_sha",
                "sensitivity_only", "gauge_ref"):
        assert col in sv.column_names, col
    assert tuple(sv.column("label_status").enum) == schema.LABEL_STATUS_VALUES
    assert tuple(sv.column("gauge_ref").enum) == schema.GAUGE_REF_VALUES


def test_schema_normative_constants_and_notes():
    assert schema.CURVE_WEIGHT_TARGET == 1.0
    assert schema.PRIMARY_D0_ANGSTROM == 0.08
    assert schema.DELTA_EPS == pytest.approx(1e-6)
    assert schema.CANONICAL_UNITS == "angstrom"
    assert schema.TRUTH_ACCESSOR_ALLOWLIST == (
        "truth_reader", "pes2ts_core.g0.truth.truth_reader")
    joined = "\n".join(schema.SCHEMA_NOTES)
    for token in ("PD-9", "ORA-2", "float64", "intent_pairing",
                  "label_weights", "truth_access", "intent_scoped_edges"):
        assert token in joined, token


# ---------------------------------------------------------------------------
# violation fixtures → exactly the expected gate ids fail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture,failing", NEW_VIOLATION_FIXTURES)
def test_v2_violation_fixture_fails_exactly_expected_tests(fixture, failing, tmp_path):
    report = v.run_validation(FIXTURES / fixture, tmp_path / "report.json")
    st = statuses(report)
    for tid in ALL_IDS:
        if tid in failing:
            assert st[tid] == "fail", (tid, st)
        else:
            assert st[tid] == "pass", (tid, st)
    assert report["exit_code"] == 1


def test_broken_involution_reasons(tmp_path):
    report = v.run_validation(FIXTURES / "broken_involution",
                              tmp_path / "report.json")
    t = get_test(report, "T0.2a")
    reasons = {b["reason"] for b in t["bad_rows"]}
    assert "self_reverse:c0_rev" in reasons
    assert t["counts"]["n_curves_with_intents"] == 2
    assert t["counts"]["n_curves_exempt_underdetermined"] == 0
    # c1 pair is untouched and must be fine
    assert t["details"]["per_curve"]["c1"]["pair_ok"] is True
    assert t["details"]["per_curve"]["c0"]["pair_ok"] is False


def test_intent_underdetermined_exempts_pairing(tmp_path):
    dst = copy_fixture("broken_involution", tmp_path)

    def _flag(df):
        df.loc[df["curve_id"] == "c0", "intent_underdetermined"] = True

    edit_csv(dst, "reaction_intent.csv", _flag)
    report = v.run_validation(dst, tmp_path / "report.json")
    t = get_test(report, "T0.2a")
    assert t["status"] == "pass"
    assert t["counts"]["n_curves_exempt_underdetermined"] == 1
    assert "exempt" in (t.get("note") or "")


def test_reverse_teacher_mismatch_reason(tmp_path):
    report = v.run_validation(FIXTURES / "reverse_teacher_mismatch",
                              tmp_path / "report.json")
    t = get_test(report, "T0.2b")
    assert t["counts"]["n_violations"] == 1
    bad = t["bad_rows"][0]
    assert bad["reason"] == "delta_not_negated"
    assert bad["curve_id"] == "c1"
    assert bad["edge_index"] == "5"
    assert bad["max_abs_sum"] > schema.DELTA_EPS
    assert t["counts"]["n_forward_edges_checked"] == 18


def test_reference_frame_flag_mismatch_is_violation(tmp_path):
    dst = copy_fixture("clean", tmp_path)

    def _flag(df):
        idx = df.index[(df["curve_id"] == "c0") & (df["intent_id"] == "c0_rev")
                       & (df["edge_index"] == 4)][0]
        df.loc[idx, "same_ref_frame"] = False

    edit_csv(dst, "irc_edge.csv", _flag)
    report = v.run_validation(dst, tmp_path / "report.json")
    t = get_test(report, "T0.2b")
    assert t["status"] == "fail"
    assert {b["reason"] for b in t["bad_rows"]} == {"reference_frame_mismatch"}


# ---------------------------------------------------------------------------
# PD-9 encoding checks (T0.1c)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,fn,expected_reason",
    [
        ("sha",
         lambda df: df.__setitem__(
             "delta_xyz_sha256",
             ["0" * 64 if i else "f" * 64 for i in range(len(df))]),
         "delta_sha_mismatch"),
        ("length",
         lambda df: df.__setitem__(
             "delta_xyz_flat",
             [json.dumps(json.loads(df["delta_xyz_flat"][0])[:-1])
              if i == 0 else df["delta_xyz_flat"][i]
              for i in range(len(df))]),
         "flat_length_ne_shape"),
        ("offsets",
         lambda df: df.__setitem__(
             "delta_xyz_offsets",
             ["[0,6]" if i == 1 else ("[10000,10006]" if i == 0
                                      else df["delta_xyz_offsets"][i])
              for i in range(len(df))]),
         "offsets_not_monotone"),
        ("units",
         lambda df: df.__setitem__("units", ["bohr"] * len(df)),
         "units_not_angstrom"),
    ],
)
def test_pd9_tampering_fails_t0_1c(name, fn, expected_reason, tmp_path):
    dst = copy_fixture("clean", tmp_path, name=f"tables_{name}")
    edit_csv(dst, "irc_edge.csv", fn)
    report = v.run_validation(dst, tmp_path / f"report_{name}.json")
    t = get_test(report, "T0.1c")
    assert t["status"] == "fail", t
    by_reason = t["counts"]["pd9_violations_by_reason"]
    assert by_reason.get(expected_reason, 0) >= 1, by_reason
    # coordinates are clean in these copies: only PD-9 rows may fail
    assert t["counts"]["n_non_finite_coordinates"] == 0


def test_offset_span_mismatch_detected(tmp_path):
    dst = copy_fixture("clean", tmp_path)

    def _span(df):
        vals = []
        for i in range(len(df)):
            if i == 2:
                vals.append("[0,5]")  # span 5 != len(flat) 6
            else:
                vals.append(df["delta_xyz_offsets"][i])
        df["delta_xyz_offsets"] = vals

    edit_csv(dst, "irc_edge.csv", _span)
    report = v.run_validation(dst, tmp_path / "report.json")
    t = get_test(report, "T0.1c")
    assert t["counts"]["pd9_violations_by_reason"].get("offset_span_ne_flat_length") == 1


# ---------------------------------------------------------------------------
# weights (T0.1a) and d0/gauge (T0.3a/T0.3b)
# ---------------------------------------------------------------------------

def test_weight_sum_invariant_details(tmp_path):
    report = v.run_validation(FIXTURES / "weight_sum_mismatch",
                              tmp_path / "report.json")
    t = get_test(report, "T0.1a")
    assert t["counts"]["n_weight_violations"] == 1
    bad = [b for b in t["bad_rows"]
           if b["reason"] == "curve_weight_sum_mismatch"]
    assert bad[0]["curve_id"] == "c1"
    assert bad[0]["sum_curve_weight"] == pytest.approx(1.45)
    assert t["details"]["curve_weight_target"] == 1.0
    # FK integrity must stay green in this fixture
    assert t["counts"]["n_orphan_rows"] == 0


def test_weight_sum_non_finite_detected(tmp_path):
    dst = copy_fixture("clean", tmp_path)

    def _nan(df):
        df.loc[0, "curve_weight"] = float("nan")

    edit_csv(dst, "sample_view.csv", _nan)
    report = v.run_validation(dst, tmp_path / "report.json")
    t = get_test(report, "T0.1a")
    assert t["counts"]["n_weight_violations"] == 1
    assert any(b["reason"] == "non_finite_curve_weight" for b in t["bad_rows"])


def test_d0_gate_counts(tmp_path):
    clean = v.run_validation(FIXTURES / "clean", tmp_path / "clean.json")
    assert get_test(clean, "T0.3a")["counts"]["n_main_d0_rows"] == 40

    report = v.run_validation(FIXTURES / "d0_unflagged",
                              tmp_path / "report.json")
    t = get_test(report, "T0.3a")
    assert t["status"] == "fail"
    assert t["counts"]["n_main_d0_rows"] == 40
    assert t["counts"]["n_sensitivity_rows"] == 0
    reasons = [b["reason"] for b in t["bad_rows"]]
    assert reasons.count("unflagged_nonprimary_d0") == 4
    assert all(abs(b["d0"] - 0.05) < 1e-12 for b in t["bad_rows"]
               if b["reason"] == "unflagged_nonprimary_d0")


def test_gauge_gate_reasons(tmp_path):
    report = v.run_validation(FIXTURES / "gauge_unresolved",
                              tmp_path / "report.json")
    t = get_test(report, "T0.3b")
    assert t["status"] == "fail"
    reasons = {b["reason"] for b in t["bad_rows"]}
    assert reasons == {"unresolved_gauge_not_masked", "bad_gauge_ref"}
    assert t["counts"]["n_unresolved_gauge_rows"] == 2
    # label_status enum violation is not involved here
    assert "bad_label_status" not in reasons


# ---------------------------------------------------------------------------
# T0.5 — truth-access audit
# ---------------------------------------------------------------------------

def test_truth_log_auto_resolved_from_tables_dir(tmp_path):
    report = v.run_validation(FIXTURES / "clean", tmp_path / "report.json")
    t = get_test(report, "T0.5")
    assert t["status"] == "pass"
    assert t["counts"]["n_entries"] == 2
    assert t["details"]["strict"] is False
    assert str(FIXTURES / "clean" / "truth_access_log.jsonl") == t["details"]["log"]


def test_truth_log_missing_strict_is_blocked(tmp_path):
    report = v.run_validation(FIXTURES / "clean", tmp_path / "report.json",
                              truth_log=tmp_path / "nope.jsonl")
    t = get_test(report, "T0.5")
    assert t["status"] == "blocked"  # fail-closed: blocked, never silently green
    assert "missing" in (t.get("note") or "")
    assert report["exit_code"] == 1
    assert report["truth_log"] == str(tmp_path / "nope.jsonl")


def test_truth_log_not_run_when_absent(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = v.run_validation(empty, tmp_path / "report.json")
    t = get_test(report, "T0.5")
    assert t["status"] == "not_run"
    assert "--truth-log" in (t.get("note") or "")


def test_truth_log_violations_fail(tmp_path):
    log = tmp_path / "truth_access_log.jsonl"
    log.write_text(
        json.dumps({"ts": "t0", "accessor": "truth_reader",
                    "reaction_id": "r1"}) + "\n"
        + json.dumps({"ts": "t1", "accessor": "pes2ts_core.utils.evil_reader",
                      "reaction_id": "r1"}) + "\n"
        + json.dumps({"ts": "t2", "function": "read_direct_hdf5"}) + "\n"
        + "not-json-at-all\n",
        encoding="utf-8",
    )
    report = v.run_validation(FIXTURES / "clean", tmp_path / "report.json",
                              truth_log=log)
    t = get_test(report, "T0.5")
    assert t["status"] == "fail"
    assert t["counts"]["n_entries"] == 4
    reasons = [b["reason"] for b in t["bad_rows"]]
    assert reasons.count("accessor_not_allowlisted") == 1
    assert reasons.count("missing_accessor_field") == 1
    assert sum(1 for r in reasons if r.startswith("unparseable_entry")) == 1
    assert t["counts"]["allowlist"] == list(schema.TRUTH_ACCESSOR_ALLOWLIST)


def test_cli_truth_log_option(tmp_path):
    missing = tmp_path / "missing.jsonl"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.data_gate.validate_r3_tables",
         "--tables-dir", str(FIXTURES / "clean"),
         "--report", str(tmp_path / "report.json"),
         "--truth-log", str(missing)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    t5 = [t for t in data["tests"] if t["id"] == "T0.5"][0]
    assert t5["status"] == "blocked"


@pytest.mark.parametrize("fixture", [f for f, _ in NEW_VIOLATION_FIXTURES])
def test_cli_new_violation_fixtures_exit_1(fixture, tmp_path):
    report = tmp_path / "cli_report.json"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.data_gate.validate_r3_tables",
         "--tables-dir", str(FIXTURES / fixture), "--report", str(report)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["schema_version"] == "r3_6table_v2"
    assert [t["id"] for t in data["tests"]] == ALL_IDS
