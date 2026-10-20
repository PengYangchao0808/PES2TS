"""E4/G3a label-gate report generator tests (fixture + REAL six tables).

Runs ``scripts/agent_ops/feature_gate/e4_label_gate.py`` mechanics on the
deterministic v3 fixture tables and — when present in this checkout — the
REAL E1 six tables (read-only; located via the git common dir / env
``PES2TS_E1_TABLES``).  The real-data section is the E4 label-gate evidence
run (CPU-only); it SKIPS (never fails) when the tables are absent so the
suite stays green in table-less checkouts.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from scripts.agent_ops.feature_gate.e4_label_gate import (  # noqa: E402
    DEV_FOLDS,
    INSUFFICIENT_MIN_POS,
    REPORT_SCHEMA,
    build_e4_label_report,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _locate_real_tables() -> Path | None:
    """E1 six tables: env override, worktree/main-tree campaign dir."""
    env = os.environ.get("PES2TS_E1_TABLES")
    if env and Path(env).is_dir():
        return Path(env)
    candidates = [PROJECT_ROOT / "outputs/pesn/30h_campaign/data"]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        main_tree = Path(common).resolve().parent
        candidates.append(main_tree / "outputs/pesn/30h_campaign/data")
    except (subprocess.CalledProcessError, OSError):
        pass
    for cand in candidates:
        if (cand / "sample_view.csv").is_file():
            return cand
    return None


@pytest.fixture()
def fixture_tables(tmp_path):
    from scripts.agent_ops.feature_gate.v3_fixtures import write_fixture_tables
    write_fixture_tables(tmp_path, n_frames=9, n_curves=3, fold=0)
    return tmp_path


# ---------------------------------------------------------------------------
# fixture mechanics
# ---------------------------------------------------------------------------
def test_fixture_report_structure(fixture_tables) -> None:
    report = build_e4_label_report(fixture_tables)
    assert report["schema"] == REPORT_SCHEMA
    # provenance: six table shas + zero truth reads
    assert len(report["provenance"]["table_sha256"]) == 6
    assert report["provenance"]["truth_reads"] == 0
    # q section basics
    q = report["q_labels"]
    assert q["n_zero_arc_curves"] == 0 and q["zero_arc_curves"] == []
    assert q["coverage"]["q_valid"] > 0
    # integrity: reverse negation exact on fixture
    checks = {c["id"]: c for c in report["integrity_checks"]}
    assert checks["reverse_q_negation_exact"]["pass"]
    assert checks["q_anchors_exact"]["pass"]
    assert checks["q_in_range"]["pass"]
    assert checks["stop_domain_1_0_mask"]["pass"]
    # wiring: bitwise backbone guarantee reported true
    wiring = report["head_wiring"]
    assert wiring["backbone_params"] == 55441
    assert wiring["progress_head_params"] == 2113
    assert wiring["ts_stop_head_params"] == 65
    assert wiring["joint_params"] == 57619
    assert wiring["u_bitwise_backbone_unchanged"] is True
    assert wiring["backbone_weights_unchanged_after_attach"] is True
    # move non-degradation hook: design only, never run
    assert report["move_non_degradation"]["status"] == "not_run"


def test_fixture_report_stop_coverage_per_fold(fixture_tables) -> None:
    report = build_e4_label_report(fixture_tables)
    coverage = report["stop_labels"]["coverage_per_fold"]
    assert coverage, "no fold coverage rows"
    for row in coverage:
        ok = row["ok"]
        assert ok["n"] == ok["pos"] + ok["neg"] + ok["mask"]
        assert row["all"]["n"] == (row["all"]["pos"] + row["all"]["neg"]
                                   + row["all"]["mask"])
    # fixture stop coverage sums to total sample count
    total = sum(row["all"]["n"] for row in coverage)
    assert total == report["dataset"]["n_samples"]


def test_fixture_class_weights_inner_train_only(fixture_tables) -> None:
    report = build_e4_label_report(fixture_tables)
    weights = report["stop_labels"]["class_weights_frozen"]
    dev_entries = [w for w in weights if w.get("fold") in DEV_FOLDS]
    assert dev_entries, "no per-fold frozen class weights"
    for w in dev_entries:
        inner = tuple(f for f in DEV_FOLDS if f != w["fold"])
        assert tuple(w["folds"]) == inner       # fold f excluded (inner-train)
        assert w["frozen"] is True
        assert w["n_pos"] + w["n_neg"] > 0 or w["pos_weight"] == 1.0
    conf = [w for w in weights if "confirm_for" in w]
    assert conf and tuple(conf[0]["folds"]) == DEV_FOLDS


def test_cli_writes_report_file(fixture_tables, tmp_path, capsys) -> None:
    out = tmp_path / "e4" / "e4_label_report.json"
    rc = main(["--fixture", "--report", str(out)])
    assert rc == 0 and out.is_file()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["fixture_mode"] is True
    assert doc["admissibility"] == "fixture_only_not_admissible"
    printed = json.loads(capsys.readouterr().out)
    assert printed["report"] == str(out)


def test_insufficient_flag_rule_constants_reported(fixture_tables) -> None:
    report = build_e4_label_report(fixture_tables)
    ins = report["stop_labels"]["insufficient_stop_positives"]
    assert ins["rule_constants"]["min_pos"] == INSUFFICIENT_MIN_POS
    assert ins["rule_constants"]["frozen"] is False   # engineering precheck
    # tiny fixtures must flag insufficient positives (honest, no padding)
    assert ins["flagged"] is True
    assert report["gate"]["proposed_state"] in ("partial", "fail")


# ---------------------------------------------------------------------------
# REAL six tables (E4 label-gate evidence; CPU)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_tables() -> Path:
    path = _locate_real_tables()
    if path is None:
        pytest.skip("E1 six tables not present (set PES2TS_E1_TABLES)")
    return path


@pytest.fixture(scope="module")
def real_report(real_tables):
    return build_e4_label_report(real_tables)


def test_real_report_core_counts(real_report) -> None:
    ds = real_report["dataset"]
    assert ds["n_samples"] == 12034          # E1 gate denominator (G0 PASS)
    assert ds["by_status"]["ok"] == 9655
    assert ds["n_curves"] == 50
    assert real_report["provenance"]["truth_reads"] == 0
    assert len(real_report["provenance"]["table_sha256"]) == 6


def test_real_report_integrity(real_report) -> None:
    checks = {c["id"]: c for c in real_report["integrity_checks"]}
    for cid in ("q_in_range", "q_anchors_exact",
                "reverse_q_negation_exact", "stop_domain_1_0_mask"):
        assert checks[cid]["pass"], cid
    rev = checks["reverse_q_negation_exact"]["evidence"]
    assert rev["pairs"] > 4000
    assert rev["max_abs_q_fwd_plus_q_rev"] == 0.0   # exact by construction


def test_real_report_zero_arc_list_and_q_coverage(real_report) -> None:
    q = real_report["q_labels"]
    assert q["n_zero_arc_curves"] == 0 and q["zero_arc_curves"] == []
    assert q["coverage"]["q_valid"] == q["coverage"]["q_valid"]
    for row in q["per_fold"]:
        assert row["n_ok"] == row["q_valid"] + row["q_masked"]
        stats = row["q_stats"]
        assert -1.0 - 1e-9 <= stats["min"] <= stats["max"] <= 1.0 + 1e-9


def test_real_report_stop_coverage_and_weights(real_report) -> None:
    stop = real_report["stop_labels"]
    total_ok = 0
    for row in stop["coverage_per_fold"]:
        ok = row["ok"]
        assert ok["n"] == ok["pos"] + ok["neg"] + ok["mask"]
        assert ok["pos"] > 0 and ok["neg"] > 0      # every fold has both
        total_ok += ok["n"]
    assert total_ok == 9655
    for w in stop["class_weights_frozen"]:
        assert w["n_pos"] > 0 and w["n_neg"] > 0
        assert w["pos_weight"] > 0
        assert w["frozen"] is True
    # real dev data has enough positives under the declared precheck
    assert stop["insufficient_stop_positives"]["flagged"] is False
    assert real_report["gate"]["proposed_state"] == "pass"


def test_real_report_head_wiring_bitwise(real_report) -> None:
    wiring = real_report["head_wiring"]
    assert wiring["u_bitwise_backbone_unchanged"] is True
    assert wiring["backbone_weights_unchanged_after_attach"] is True
    assert wiring["real_batch_shapes_ok"] is True
    assert wiring["real_batch_n_graphs"] >= 1


def test_real_report_move_hook_never_run(real_report) -> None:
    hook = real_report["move_non_degradation"]
    assert hook["status"] == "not_run"
    assert "R3" in hook["reason"]
    assert "pass_condition" in hook["protocol"]
