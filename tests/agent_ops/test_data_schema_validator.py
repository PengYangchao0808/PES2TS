"""Tests for the R3 data-table schema + Data Gate validator (T0.1 / T0.4).

Round 2 (WP-D2): updated for schema ``r3_6table_v2`` (PD-9 flat ``delta_xyz``,
intent pairing, label/weight columns, d0/gauge rules, T0.5 truth audit). The
ORA-2-specific gates live in ``test_data_gate_t02.py``; this file keeps the
round-1 contract coverage (T0.1a/b/c, T0.4a/b, blocked/not_run semantics,
CLI exit codes) green against the v2 fixtures.

Fixture sets under ``tests/agent_ops/data_fixtures/`` (each also carries a
``truth_access_log.jsonl`` allow-listed truth log for T0.5):

* ``clean``         — 2 curves × 10 nodes, forward+reverse intents, PD-9 edges
                      in both directions, weights summing to 1.0 per
                      (curve_id, d0); all 10 gate tests pass.
* ``orphan_fk``     — orphan rows for node→curve, edge→node (both directions),
                      sample→intent / curve / node ⇒ only ``T0.1a`` fails.
* ``duplicate_ts``  — two ``is_ts`` nodes on one curve ⇒ only ``T0.1b`` fails.
* ``nan_split_leak``— NaN/inf coordinates **and** one curve listed in two folds
                      ⇒ ``T0.1c`` + ``T0.4a`` fail.

Dry-run scope: no truth-data access, no ACP/engine calls, no network.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops.data_gate import r3_table_schema as schema  # noqa: E402
from scripts.agent_ops.data_gate import validate_r3_tables as v  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "data_fixtures"
ALL_IDS = list(v.TEST_IDS)
SIX_TABLES = {
    "irc_curve",
    "irc_node",
    "irc_edge",
    "reaction_intent",
    "curve_quality",
    "sample_view",
}


def run(fixture: str, tmp_path: Path, name: str = "report.json", **kw):
    return v.run_validation(FIXTURES / fixture, tmp_path / name, **kw)


def statuses(report):
    return {t["id"]: t["status"] for t in report["tests"]}


def get_test(report, test_id):
    return next(t for t in report["tests"] if t["id"] == test_id)


# ---------------------------------------------------------------------------
# schema module
# ---------------------------------------------------------------------------

def test_schema_defines_exactly_the_six_30h_tables():
    assert set(schema.TABLE_ORDER) == SIX_TABLES
    assert set(schema.TABLES) == SIX_TABLES
    assert schema.TABLE_ORDER == tuple(schema.TABLES)  # declared order
    for name, tsch in schema.TABLES.items():
        assert tsch.filename == f"{name}.csv"
        assert tsch.primary_key
        assert tsch.required_columns, name


def test_schema_records_doc_discrepancies():
    """5-table vs 6-table doc disagreement must be machine-recorded."""
    assert schema.SCHEMA_NOTES, "schema discrepancies must be recorded"
    joined = "\n".join(schema.SCHEMA_NOTES)
    assert "table_count" in joined
    assert "six" in joined and "five" in joined
    assert "split_manifest" in joined  # §3.1 vs §7.1 filename disagreement
    assert schema.SCHEMA_SOURCE


def test_schema_keys_and_sample_identity():
    assert schema.TABLES["irc_node"].primary_key == ("curve_id", "node_index")
    assert ("sample_view", ("curve_id", "node_index"), "irc_node",
            ("curve_id", "node_index")) in schema.COMPOSITE_FK
    sv = schema.TABLES["sample_view"]
    assert {"curve_id", "node_index", "intent_id", "d0"} <= set(sv.column_names)
    assert ("curve_id", "node_index", "intent_id", "d0") in sv.unique
    # d0 must not be part of curve identity (plan §3.1)
    assert "d0" not in schema.TABLES["irc_curve"].column_names
    # FK declarations for the checks T0.1a performs
    assert ("curve_id", "irc_curve", "curve_id") in schema.TABLES["irc_node"].fk_columns
    assert ("intent_id", "reaction_intent", "intent_id") in sv.fk_columns
    assert ("intent_id", "reaction_intent", "intent_id") in schema.TABLES["irc_edge"].fk_columns


# ---------------------------------------------------------------------------
# clean fixtures → all pass
# ---------------------------------------------------------------------------

def test_clean_fixture_all_tests_pass(tmp_path):
    report = run("clean", tmp_path)
    st = statuses(report)
    assert st == {tid: "pass" for tid in ALL_IDS}, st
    assert report["exit_code"] == 0
    assert report["summary"]["all_pass"] is True
    assert report["summary"]["pass"] == len(ALL_IDS)
    assert report["schema_version"] == schema.SCHEMA_VERSION == "r3_6table_v2"
    for t in report["tests"]:
        assert t["evidence"] and t["evidence"].startswith(str(tmp_path))
        assert t["inputs"], f"{t['id']} must record the files it read"
    # report embeds schema provenance + discrepancy notes
    assert report["schema_source"] and report["schema_notes"]
    assert (tmp_path / "report.json").is_file()


def test_clean_split_manifest_near_dup_groups_present(tmp_path):
    report = run("clean", tmp_path)
    t = get_test(report, "T0.4a")
    assert t["details"]["near_dup_column_present"] is True
    assert t["counts"]["n_near_dup_groups"] == 2
    assert t["counts"]["n_violations"] == 0


def test_clean_weight_and_pd9_counts(tmp_path):
    report = run("clean", tmp_path)
    t1a = get_test(report, "T0.1a")
    assert t1a["counts"]["n_weight_groups"] == 2  # (c0, 0.08) and (c1, 0.08)
    assert t1a["counts"]["n_weight_violations"] == 0
    assert t1a["counts"]["n_orphan_rows"] == 0
    assert t1a["details"]["curve_weight_target"] == 1.0
    t1c = get_test(report, "T0.1c")
    assert t1c["counts"]["n_edges_pd9_checked"] == 36  # 2 curves x 2 intents x 9
    assert t1c["counts"]["n_pd9_violations"] == 0
    t5 = get_test(report, "T0.5")
    assert t5["counts"]["n_entries"] == 2
    assert t5["counts"]["n_violations"] == 0


# ---------------------------------------------------------------------------
# violation fixtures → exactly the expected test ids fail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture,failing",
    [
        ("orphan_fk", {"T0.1a"}),
        ("duplicate_ts", {"T0.1b"}),
        ("nan_split_leak", {"T0.1c", "T0.4a"}),
    ],
)
def test_violation_fixture_fails_exactly_expected_tests(fixture, failing, tmp_path):
    report = run(fixture, tmp_path)
    st = statuses(report)
    for tid in ALL_IDS:
        if tid in failing:
            assert st[tid] == "fail", (tid, st)
        else:
            assert st[tid] == "pass", (tid, st)
    assert report["exit_code"] == 1
    assert report["summary"]["all_pass"] is False


def test_orphan_fixture_counts_and_evidence(tmp_path):
    report = run("orphan_fk", tmp_path)
    t = get_test(report, "T0.1a")
    assert t["counts"]["n_orphan_rows"] == 6
    assert t["counts"]["n_relations_checked"] == t["counts"]["n_relations_total"]
    # FK orphans only — the weight invariant must stay green in this fixture
    assert t["counts"]["n_weight_violations"] == 0
    assert t["counts"]["n_integrity_violations"] == 0
    relations = {b["relation"] for b in t["bad_rows"]}
    assert any(r.startswith("irc_node.") for r in relations)  # node→curve
    assert any("dst_node_index" in r for r in relations)      # edge→node
    assert any("intent_id" in r for r in relations)           # sample→intent
    assert all("value" in b for b in t["bad_rows"])


def test_duplicate_ts_fixture_reports_reason(tmp_path):
    report = run("duplicate_ts", tmp_path)
    t = get_test(report, "T0.1b")
    assert t["counts"]["n_curves_duplicate_ts"] == 1
    assert t["counts"]["n_curves_ts_side_order_illegal"] == 0
    assert any("duplicate_ts" in b["reason"] for b in t["bad_rows"])
    assert t["details"]["per_curve"]["c1"]["ts_side_order_legal"] is True


def test_nan_and_leak_fixture_bad_rows_never_dropped(tmp_path):
    report = run("nan_split_leak", tmp_path)
    t1c = get_test(report, "T0.1c")
    assert t1c["counts"]["n_non_finite_coordinates"] == 2
    assert t1c["counts"]["n_pd9_violations"] == 0
    tables = {b["table"] for b in t1c["bad_rows"]}
    assert tables == {"irc_node", "sample_view"}
    t4a = get_test(report, "T0.4a")
    assert t4a["counts"]["n_violations"] == 1
    assert t4a["bad_rows"][0]["reason"] == "curve_in_two_folds"
    assert t4a["bad_rows"][0]["curve_id"] == "c0"
    # near-dup column intentionally absent from this fixture
    assert t4a["details"]["near_dup_column_present"] is False


# ---------------------------------------------------------------------------
# blocked / not_run semantics
# ---------------------------------------------------------------------------

def test_missing_inputs_are_blocked_not_passed(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    shutil.copy(FIXTURES / "clean" / "irc_curve.csv", partial / "irc_curve.csv")
    report = v.run_validation(partial, tmp_path / "report.json")
    st = statuses(report)
    for tid in ALL_IDS:
        if tid == "T0.5":
            continue  # no truth log in this dir → not_run (nothing to audit)
        assert st[tid] == "blocked", (tid, st)
    assert st["T0.5"] == "not_run"
    assert report["exit_code"] == 1
    assert report["summary"]["all_pass"] is False


def test_tests_filter_marks_others_not_run(tmp_path):
    report = run("clean", tmp_path, only_tests=["T0.1a"])
    st = statuses(report)
    assert st["T0.1a"] == "pass"
    assert st["T0.4b"] == "not_run"
    assert st["T0.5"] == "not_run"
    assert report["exit_code"] == 1  # not_run is not a pass


def test_status_enum_is_complete():
    assert set(v.STATUS_VALUES) == {"pass", "fail", "blocked", "not_run"}
    assert set(v.TEST_IDS) == {
        "T0.1a", "T0.1b", "T0.1c", "T0.2a", "T0.2b", "T0.3a", "T0.3b",
        "T0.4a", "T0.4b", "T0.5",
    }


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture,expected_code",
    [
        ("clean", 0),
        ("orphan_fk", 1),
        ("duplicate_ts", 1),
        ("nan_split_leak", 1),
    ],
)
def test_cli_exit_codes(fixture, expected_code, tmp_path):
    report = tmp_path / "cli_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.agent_ops.data_gate.validate_r3_tables",
            "--tables-dir",
            str(FIXTURES / fixture),
            "--report",
            str(report),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == expected_code, proc.stdout + proc.stderr
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["exit_code"] == expected_code
    assert [t["id"] for t in data["tests"]] == ALL_IDS
    if expected_code == 0:
        assert all(t["status"] == "pass" for t in data["tests"])
    else:
        assert any(t["status"] != "pass" for t in data["tests"])
