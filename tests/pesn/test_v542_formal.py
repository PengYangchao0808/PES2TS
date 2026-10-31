"""V5.4.2 WP5 formal runner + A0 sanity-band tests (lane A, P5).

CPU-fast gates for ``scripts/agent_ops/v542_formal.py`` (plan §8,
ADR-0033 §2.2/§2.6):

* **formal validation gate** — the runner validates the frozen config in
  FORMAL mode FIRST; a validator rejection surfaces as a typed refusal
  with the VERBATIM code (``V542_RECORD_MISSING`` when the governance
  record instances are absent, ``V542_RECORD_MISMATCH`` when a config
  copy drifts from the frozen record spec);
* **T0-before-T1 ordering** — T1 refuses (typed
  ``V542_FORMAL_T0_INCOMPLETE``) until all three T0 A0 formal runs are
  ``done`` in the ledger; T1 additionally requires the
  ``V542-A0-sanity-band`` PASS record (absent ⇒ typed
  ``V542_SANITY_BAND_ABSENT``; breach ⇒ typed
  ``V542_SANITY_BAND_BREACH``, no launch);
* **A4 never runs** — ``V542_A4_S17`` (and A4 S29/S43) are always
  ``gate_skipped`` with the frozen WP4 ruling reason
  ``diag_best_inner_val_drop_lt_5pct_vs_initial`` and
  ``new_training=0``, independent of cost/diagnostic gates and even
  under ``--only``; a hypothetical A4/A1-C/A5-C/conditional launch is a
  typed ``V542_FORMAL_LAUNCH_FORBIDDEN`` refusal;
* **conditional skip reasons** — T2 A5 ⇒
  ``a5_not_activated_no_A1_A4_h3_qualified_candidate_after_top2_seeds``;
  T3 top-2 rows ⇒ ``top2_not_frozen``; T3 A5 supplement ⇒
  ``a5_supplement_parent_gate_unmet``; T-ATTR ⇒
  ``a1_not_in_top2_and_not_selected`` / ``a5_not_selected_backbone`` —
  all recorded ``gate_skipped`` with ``new_training=0`` in the
  append-only ledger;
* **per-run execution** — a tiny (test-dose) T0-style run writes the
  ledger row (outcome/new_training/identity block + digest/wall+core
  seconds/final inner-val loss/reason/timestamp), the checkpoints and
  the per-run ``run_manifest.json``; the ledger is append-only with
  monotone ``seq``;
* **resume** — a second leg resumes from ``checkpoint_last.pt`` (same
  identity digest, manifest ``resume`` event); an identity mismatch is
  the trainer's typed ``invalid_run`` refusal WITHOUT touching the
  checkpoint (byte-identical);
* **sanity band** — the exact frozen rule on PER-ENDPOINT SEED MEANS:
  RMSE/zero relative ≤ 2% AND AUPRC/q absolute ≤ 0.01 (pass/breach/
  seed-mean-vs-per-seed/boundary cases); the record is written with the
  verdict, the numbers, the evaluator identity and the source run
  identities; a breach record states T1 stays blocked with the frozen
  audit categories + identity-layer bumps and can never unblock T1;
* **discipline** — no ``torch.compile``; ``--dry-run`` performs no
  training and no writes.

The tiny-run tests reuse the REAL six-table fixture env from the frozen
V5.3-MP trainer tests (``_build_env``); all doses are tiny test-only
overrides recorded honestly (``is_formal_schedule=false``) — the frozen
config, thresholds and trainer math are never modified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml  # noqa: E402  (existing repo dependency)

from pes2ts_core.pesn.models.v542.identity import (  # noqa: E402
    canonical_sha256,
)
from scripts.agent_ops.v542_formal import (  # noqa: E402
    A0_SANITY_BAND_GATE,
    EVALUATOR_IDENTITY,
    FORMAL_STAGES,
    REASON_A1C_SKIP,
    REASON_A4_GATE_SKIPPED,
    REASON_A5C_SKIP,
    REASON_A5_NOT_ACTIVATED,
    REASON_A5_NOT_EVALUATED,
    REASON_TOP2_NOT_FROZEN,
    REASON_TOP2_NOT_SELECTED,
    V542FormalError,
    _assert_launch_allowed,
    append_ledger_row,
    compute_sanity_band,
    decide_stage_runs,
    execute_formal_run,
    filter_only,
    formal_gate,
    latest_ledger_state,
    locate_v541_e0_reference,
    main,
    read_band_record,
    read_ledger,
    run_sanity_band,
    run_stage,
    stage_rows,
    status_report,
)
from scripts.agent_ops.v542_formal import (  # noqa: E402
    REASON_A5_SUPPLEMENT as REASON_A5_SUPPLEMENT_SKIP,
)
from tests.pesn.test_r3_v53mp_trainer import (  # noqa: E402
    DEFAULT_FOLDS,
    _build_env,
)

CONFIG_PATH = REPO / "config/campaigns/pes2ts_v542_backbone_50k.yaml"
IDENTITY: Dict[str, str] = {
    "source_manifest_sha256": "a" * 64,
    "config_sha256": "b" * 64,
    "common_protocol_sha256": "c" * 64,
}


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg() -> Dict[str, Any]:
    from scripts.agent_ops.v542_formal import load_formal_config

    return load_formal_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """The REAL fixture env, built once for the tiny-run tests."""
    return _build_env(tmp_path_factory.mktemp("v542formal"),
                      DEFAULT_FOLDS)


def _t0_done(campaign_root: Path) -> None:
    """Simulate a completed T0 A0x3 in the append-only ledger."""
    for seed in (17, 29, 43):
        append_ledger_row(campaign_root, {
            "run_id": f"V542_A0_S{seed}", "stage": "T0", "backbone": "A0",
            "seed": seed, "outcome": "done", "new_training": 1,
            "reason": "formal_run_complete",
            "identity": dict(IDENTITY),
            "identity_sha256": "f" * 64, "wall_s": 1.0, "core_s": 1.0,
            "final_inner_val_loss": 0.5, "is_formal_schedule": True,
            "device": "cpu",
        })


def _band_pass_record() -> Dict[str, Any]:
    return {"schema": "pes2ts_v542_a0_sanity_band_v1",
            "gate": A0_SANITY_BAND_GATE, "verdict": "pass",
            "t1_status": "unblocked", "breached_endpoints": []}


def _write_band(campaign_root: Path, record: Dict[str, Any]) -> None:
    path = campaign_root / "plan" / "a0_sanity_band.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def _a0_endpoints(over: Optional[Dict[int, Dict[str, float]]] = None
                  ) -> Dict[int, Dict[str, float]]:
    base = {17: {"rmse_zero": 0.905, "endpoint_auprc": 0.276,
                 "q_corr": 0.9965},
            29: {"rmse_zero": 0.915, "endpoint_auprc": 0.271,
                 "q_corr": 0.9955},
            43: {"rmse_zero": 0.895, "endpoint_auprc": 0.279,
                 "q_corr": 0.9970}}
    base.update(over or {})
    return base


def _v541_reference() -> Dict[int, Dict[str, Any]]:
    refs = {}
    for seed, rmse, auprc, q, ckpt in (
            (17, 0.901, 0.274, 0.9964, 37000),
            (29, 0.908, 0.274, 0.9964, 36400),
            (43, 0.899, 0.275, 0.9963, 37800)):
        refs[seed] = {
            "endpoints": {"rmse_zero": rmse, "endpoint_auprc": auprc,
                          "q_corr": q},
            "provenance": {
                "metrics_path": f"fake/V541_T0R_E0_S{seed}/metrics.json",
                "metrics_sha256": "0" * 64,
                "checkpoint_path": f"fake/V541_T0R_E0_S{seed}/ckpt",
                "checkpoint_sha256_pinned": "1" * 64,
                "run_id": f"V541_T0R_E0_S{seed}",
                "source_tier": "v541_e0_runs_final_37900",
                "run_updates": 37900, "checkpoint_update": ckpt,
            },
        }
    return refs


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# formal validation gate (typed refusals, verbatim codes)
# ---------------------------------------------------------------------------
def test_formal_gate_refuses_when_record_instances_are_missing(tmp_path):
    # the frozen validator's record_root seam: no instances under root
    # ⇒ typed V542_RECORD_MISSING, surfaced VERBATIM by the runner.
    with pytest.raises(V542FormalError) as excinfo:
        formal_gate(CONFIG_PATH, record_root=tmp_path)
    assert excinfo.value.code == "V542_RECORD_MISSING"
    assert "formal launch is typed-blocked by the frozen config validator" \
        in excinfo.value.detail


def test_formal_gate_refuses_a_config_drifting_from_the_record_spec(
        tmp_path, capsys):
    # a config copy whose record instance path drifts from the frozen
    # spec ⇒ typed V542_RECORD_MISMATCH through the CLI (exit 2, no
    # traceback).
    doc = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    doc = copy.deepcopy(doc)
    doc["governance"]["records"]["a0_reuse_decision_record"]["instance"] = (
        "outputs/pesn/v542_backbone_50k/plan/elsewhere.json")
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(doc, sort_keys=False),
                       encoding="utf-8")
    code = main(["run", "--stage", "T0", "--dry-run",
                 "--config", str(drifted),
                 "--campaign-root", str(tmp_path / "campaign")])
    assert code == 2
    err = capsys.readouterr().err
    assert "V542_RECORD_MISMATCH" in err
    assert "Traceback" not in err


def test_formal_gate_passes_on_the_frozen_config():
    report = formal_gate(CONFIG_PATH)
    assert report["mode"] == "formal"
    assert len(report["checks"]) == 28


# ---------------------------------------------------------------------------
# T0-before-T1 ordering + sanity-band hard precondition
# ---------------------------------------------------------------------------
def test_t1_refuses_until_t0_a0x3_is_done(tmp_path):
    from scripts.agent_ops.v542_formal import load_formal_config
    cfg = load_formal_config(CONFIG_PATH)
    with pytest.raises(V542FormalError) as excinfo:
        run_stage("T1", cfg=cfg, campaign_root=tmp_path,
                  formal_validated=True)
    assert excinfo.value.code == "V542_FORMAL_T0_INCOMPLETE"
    assert "V542_A0_S17" in excinfo.value.detail
    # nothing was written: no ledger, no band, no run dirs
    assert not (tmp_path / "run_ledger.jsonl").exists()
    assert not (tmp_path / "runs").exists()


def test_t1_refuses_while_any_single_t0_seed_is_missing(tmp_path):
    from scripts.agent_ops.v542_formal import load_formal_config
    cfg = load_formal_config(CONFIG_PATH)
    _t0_done(tmp_path)
    # invalidate S43: a fresh ledger state must still refuse
    rows = read_ledger(tmp_path)
    rows[-1]["outcome"] = "invalid_run"
    (tmp_path / "run_ledger.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(V542FormalError) as excinfo:
        run_stage("T1", cfg=cfg, campaign_root=tmp_path,
                  formal_validated=True)
    assert excinfo.value.code == "V542_FORMAL_T0_INCOMPLETE"
    assert "V542_A0_S43" in excinfo.value.detail


def test_t1_refuses_without_a_sanity_band_record(tmp_path):
    from scripts.agent_ops.v542_formal import load_formal_config
    cfg = load_formal_config(CONFIG_PATH)
    _t0_done(tmp_path)
    with pytest.raises(V542FormalError) as excinfo:
        run_stage("T1", cfg=cfg, campaign_root=tmp_path,
                  only=["V542_A1_S17"], formal_validated=True)
    assert excinfo.value.code == "V542_SANITY_BAND_ABSENT"
    assert A0_SANITY_BAND_GATE in excinfo.value.detail


def test_t1_refuses_on_a_breach_band_record(tmp_path):
    from scripts.agent_ops.v542_formal import load_formal_config
    cfg = load_formal_config(CONFIG_PATH)
    _t0_done(tmp_path)
    _write_band(tmp_path, {
        "schema": "pes2ts_v542_a0_sanity_band_v1",
        "gate": A0_SANITY_BAND_GATE, "verdict": "breach",
        "t1_status": "blocked_sanity_band_breach",
        "breached_endpoints": ["rmse_zero"]})
    with pytest.raises(V542FormalError) as excinfo:
        run_stage("T1", cfg=cfg, campaign_root=tmp_path,
                  only=["V542_A1_S17"], formal_validated=True)
    assert excinfo.value.code == "V542_SANITY_BAND_BREACH"
    assert "rmse_zero" in excinfo.value.detail
    assert "exception" in excinfo.value.detail    # no exception signing


# ---------------------------------------------------------------------------
# A4 never runs; conditional rows stay gate_skipped
# ---------------------------------------------------------------------------
def test_a4_is_always_gate_skipped_with_the_frozen_reason(cfg, tmp_path):
    _t0_done(tmp_path)
    _write_band(tmp_path, _band_pass_record())
    plan = run_stage("T1", cfg=cfg, campaign_root=tmp_path,
                     only=["V542_A4_S17"], formal_validated=True)
    assert plan["executed"] == []
    assert plan["gate_skipped_recorded"] == ["V542_A4_S17"]
    row = read_ledger(tmp_path)[-1]
    assert row["run_id"] == "V542_A4_S17"
    assert row["outcome"] == "gate_skipped"
    assert row["reason"] == REASON_A4_GATE_SKIPPED
    assert row["new_training"] == 0
    assert row["identity_sha256"] is None
    assert not (tmp_path / "runs" / "V542_A4_S17").exists()


def test_a4_decision_never_consults_diagnostic_or_cost(cfg, tmp_path):
    # unit level: with the STAGE gates satisfied (T0 done + band pass),
    # the A4 branch still fires BEFORE any diagnostic/cost lookup — an
    # empty campaign root cannot verify either gate, yet A4 is decided.
    state = {f"V542_A0_S{s}": {"outcome": "done"} for s in (17, 29, 43)}
    rows = [row for row in stage_rows(cfg, "T1") if row.backbone == "A4"]
    decisions = decide_stage_runs(
        "T1", rows, state=state, band=_band_pass_record(), cfg=cfg,
        campaign_root=tmp_path / "no-such-dir")
    assert len(decisions) == 1
    assert decisions[0].action == "gate_skipped"
    assert decisions[0].reason == REASON_A4_GATE_SKIPPED
    assert decisions[0].new_training == 0


def test_a4_t3_rows_keep_the_binding_reason(cfg, tmp_path):
    _write_band(tmp_path, _band_pass_record())
    decisions = decide_stage_runs(
        "T3", [r for r in stage_rows(cfg, "T3") if r.backbone == "A4"],
        state={}, band=None, cfg=cfg, campaign_root=tmp_path)
    assert [d.reason for d in decisions] == [REASON_A4_GATE_SKIPPED] * 2
    assert all(d.new_training == 0 for d in decisions)


def test_t1_candidates_launch_when_gates_hold(cfg, tmp_path):
    # diagnostic pointer present ⇒ A1/A2/A3 advance (frozen WP4 v2
    # report) and the signed cost ratios are all <= 2.0 ⇒ launch.
    _t0_done(tmp_path)
    _write_band(tmp_path, _band_pass_record())
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir(parents=True)
    repo_pointer = (REPO / "outputs/pesn/v542_backbone_50k/diagnostics"
                    / "diagnostic_report_current.json")
    (diag_dir / "diagnostic_report_current.json").write_bytes(
        repo_pointer.read_bytes())
    rows = [r for r in stage_rows(cfg, "T1") if r.backbone != "A4"]
    decisions = decide_stage_runs(
        "T1", rows, state=latest_ledger_state(read_ledger(tmp_path)),
        band=read_band_record(tmp_path), cfg=cfg, campaign_root=tmp_path)
    assert [d.action for d in decisions] == ["launch"] * 3
    assert [d.run_id for d in decisions] == [
        "V542_A1_S17", "V542_A2_S17", "V542_A3_S17"]
    assert all(d.new_training == 1 for d in decisions)


def test_t2_t3_attr_skip_reasons_and_zero_new_training(cfg, tmp_path):
    plan = run_stage("T2", cfg=cfg, campaign_root=tmp_path,
                     formal_validated=True)
    assert plan["gate_skipped_recorded"] == ["V542_A5_S17"]
    # WP6: T3 top-2 rows cannot even be decided before the seed17 screen
    # report exists — typed refusal, nothing written.
    with pytest.raises(V542FormalError) as excinfo:
        run_stage("T3", cfg=cfg, campaign_root=tmp_path,
                  formal_validated=True)
    assert excinfo.value.code == "V542_TOP2_SCREEN_ABSENT"
    assert not (tmp_path / "run_ledger.jsonl").read_text().strip().splitlines()[3:]
    # with a frozen screen that selects nothing, every A1-A3 T3 row is
    # gate_skipped as "not in the frozen top2" (still zero training).
    _write_screen_report(tmp_path, top2=[])
    plan = run_stage("T3", cfg=cfg, campaign_root=tmp_path,
                     formal_validated=True)
    assert len(plan["gate_skipped_recorded"]) == 10
    plan = run_stage("T-ATTR", cfg=cfg, campaign_root=tmp_path,
                     formal_validated=True)
    assert plan["gate_skipped_recorded"] == [
        "V542_A1C_S17", "V542_A5C_S17"]
    assert plan["executed"] == []

    by_id = {r["run_id"]: r for r in read_ledger(tmp_path)}
    assert by_id["V542_A5_S17"]["reason"] == REASON_A5_NOT_EVALUATED
    for run_id in ("V542_A1_S29", "V542_A1_S43", "V542_A2_S29",
                   "V542_A2_S43", "V542_A3_S29", "V542_A3_S43"):
        assert by_id[run_id]["reason"] == REASON_TOP2_NOT_SELECTED
    for run_id in ("V542_A5_S29", "V542_A5_S43"):
        assert by_id[run_id]["reason"] == REASON_A5_SUPPLEMENT_SKIP
    for run_id in ("V542_A4_S29", "V542_A4_S43"):
        assert by_id[run_id]["reason"] == REASON_A4_GATE_SKIPPED
    assert by_id["V542_A1C_S17"]["reason"] == REASON_A1C_SKIP
    assert by_id["V542_A5C_S17"]["reason"] == REASON_A5C_SKIP
    assert all(row["outcome"] == "gate_skipped"
               and row["new_training"] == 0
               for row in by_id.values())


def _write_screen_report(root, *, top2):
    """Synthetic WP6 seed17 screen report for the T3 gate tests."""
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "pes2ts_v542_seed17_screen_report_v1",
        "top2": list(top2),
        "top2_count": len(top2),
        "top2_freeze_precondition": {
            "asserted": True,
            "a0_s17_done": True,
            "sanity_band_gate": "V542-A0-sanity-band",
            "sanity_band_verdict": "pass",
        },
    }
    (reports / "seed17_screen_report.json").write_text(
        json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def test_t3_launches_only_the_frozen_top2(cfg, tmp_path):
    """Decisions only (no training): top2 membership is the T3 gate."""
    _write_screen_report(tmp_path, top2=["A2", "A3"])
    decisions = decide_stage_runs(
        "T3", stage_rows(cfg, "T3"), state={}, band=None, cfg=cfg,
        campaign_root=tmp_path)
    by_id = {d.run_id: d for d in decisions}
    for backbone in ("A2", "A3"):
        for seed in (29, 43):
            d = by_id[f"V542_{backbone}_S{seed}"]
            assert d.action == "launch" and d.new_training == 1
    for seed in (29, 43):
        assert by_id[f"V542_A1_S{seed}"].reason == \
            REASON_TOP2_NOT_SELECTED
        assert by_id[f"V542_A4_S{seed}"].reason == \
            REASON_A4_GATE_SKIPPED
        assert by_id[f"V542_A5_S{seed}"].reason == \
            REASON_A5_SUPPLEMENT_SKIP
    # the freeze precondition must be asserted
    _write_screen_report(tmp_path, top2=["A2", "A3"])
    doc_path = tmp_path / "reports/seed17_screen_report.json"
    doc = json.loads(doc_path.read_text())
    doc["top2_freeze_precondition"]["asserted"] = False
    doc_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(V542FormalError) as excinfo:
        decide_stage_runs("T3", stage_rows(cfg, "T3"), state={},
                          band=None, cfg=cfg, campaign_root=tmp_path)
    assert excinfo.value.code == "V542_TOP2_SCREEN_ABSENT"


def test_conditional_launches_are_structurally_forbidden(cfg):
    from scripts.agent_ops.v542_formal import RunRow

    for run_id, stage, backbone in (
            ("V542_A4_S17", "T1", "A4"),
            ("V542_A4_S29", "T3", "A4"),
            ("V542_A5_S29", "T3", "A5"),
            ("V542_A1C_S17", "T-ATTR", "A1-C"),
            ("V542_A5C_S17", "T-ATTR", "A5-C")):
        row = RunRow(run_id, stage, backbone, 17, "conditional", False)
        with pytest.raises(V542FormalError) as excinfo:
            _assert_launch_allowed(row)
        assert excinfo.value.code == "V542_FORMAL_LAUNCH_FORBIDDEN"
        assert run_id in excinfo.value.detail
    # WP6: T3 top-2 rows of A1-A3 and the T2 A5 full-dose row are
    # structurally allowed — their frozen predicates are the substantive
    # gates, asserted separately.
    for backbone in ("A1", "A2", "A3"):
        _assert_launch_allowed(
            RunRow(f"V542_{backbone}_S29", "T3", backbone, 29,
                   "conditional", True))
    _assert_launch_allowed(
        RunRow("V542_A5_S17", "T2", "A5", 17, "conditional", True))


def test_unknown_stage_and_run_ids_are_typed_refusals(cfg):
    assert FORMAL_STAGES == ("T0", "T1", "T2", "T3", "T-ATTR")
    with pytest.raises(V542FormalError) as excinfo:
        stage_rows(cfg, "T-DIAG")
    assert excinfo.value.code == "V542_FORMAL_STAGE_UNKNOWN"
    with pytest.raises(V542FormalError) as excinfo:
        filter_only(stage_rows(cfg, "T0"), ["V542_A0_S99"])
    assert excinfo.value.code == "V542_FORMAL_RUN_UNKNOWN"


# ---------------------------------------------------------------------------
# per-run execution: ledger row / checkpoints / manifest / resume
# ---------------------------------------------------------------------------
def test_two_step_t0_run_writes_ledger_checkpoints_and_identity(
        cfg, env, tmp_path):
    row = stage_rows(cfg, "T0")[0]
    result = execute_formal_run(
        env, row, cfg_mapping=cfg, config_path=str(CONFIG_PATH),
        campaign_root=tmp_path, device="cpu", identity=dict(IDENTITY),
        test_dose=(1, 1))
    ledger_row = result["ledger_row"]
    assert ledger_row["run_id"] == "V542_A0_S17"
    assert ledger_row["stage"] == "T0"
    assert ledger_row["outcome"] == "done"
    assert ledger_row["new_training"] == 1
    assert ledger_row["reason"] == "formal_run_complete"
    # identity block + digest (canonical recipe)
    assert ledger_row["identity"]["source_manifest_sha256"] == \
        IDENTITY["source_manifest_sha256"]
    assert "schema" in ledger_row["identity"]     # canonical overlay
    assert ledger_row["identity_sha256"] == \
        canonical_sha256(ledger_row["identity"])
    # wall/core seconds, final inner-val loss, timestamp
    assert isinstance(ledger_row["wall_s"], (int, float))
    assert isinstance(ledger_row["core_s"], (int, float))
    assert ledger_row["final_inner_val_loss"] == pytest.approx(
        result["metrics"]["best_inner_val_loss"])
    assert ledger_row["ts"]
    assert ledger_row["torch_compile"] is False
    # test-dose honesty: a tiny run is NEVER a formal-schedule attempt
    assert ledger_row["is_formal_schedule"] is False
    assert result["manifest"]["dose"] == {
        "w0_updates": 1, "w1_updates": 1, "total_updates": 2,
        "val_every": 100}

    run_dir = tmp_path / "runs" / "V542_A0_S17"
    assert (run_dir / "checkpoint_last.pt").is_file()
    assert (run_dir / "checkpoint_best.pt").is_file()
    assert (run_dir / "run_manifest.json").is_file()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(
        encoding="utf-8"))
    assert manifest["outcome"] == "done"
    assert manifest["identity_sha256"] == ledger_row["identity_sha256"]
    assert manifest["ledger_seq"] == ledger_row["seq"]
    assert manifest["torch_compile"] is False
    assert manifest["qc_calls"] == 0
    assert manifest["confirm_read"] is False


def test_resume_path_continues_and_identity_mismatch_is_refused(
        cfg, env, tmp_path):
    row = stage_rows(cfg, "T0")[0]
    common = dict(cfg_mapping=cfg, config_path=str(CONFIG_PATH),
                  campaign_root=tmp_path, device="cpu",
                  test_dose=(1, 1))
    leg1 = execute_formal_run(env, row, identity=dict(IDENTITY),
                              test_smoke_updates=1, **common)
    assert leg1["metrics"]["updates_run"] == 1
    ckpt = tmp_path / "runs" / "V542_A0_S17" / "checkpoint_last.pt"

    leg2 = execute_formal_run(env, row, identity=dict(IDENTITY),
                              test_smoke_updates=2, **common)
    assert leg2["ledger_row"]["outcome"] == "done"
    assert leg2["ledger_row"]["identity_sha256"] == \
        leg1["ledger_row"]["identity_sha256"]
    manifest_events = [
        json.loads(line)["event"]
        for line in (tmp_path / "runs" / "V542_A0_S17" / "manifest.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "resume" in manifest_events

    # identity mismatch (a caller-identity key that survives the
    # canonical overlay) ⇒ typed invalid_run refusal, checkpoint
    # byte-identical, ledger gains an invalid_run row.
    digest_before = _sha256_file(ckpt)
    bad_identity = dict(IDENTITY)
    bad_identity["source_manifest_sha256"] = "d" * 64
    with pytest.raises(V542FormalError) as excinfo:
        execute_formal_run(env, row, identity=bad_identity,
                           test_smoke_updates=3, **common)
    assert excinfo.value.code == "RESUME_IDENTITY_MISMATCH"
    assert excinfo.value.outcome == "invalid_run"
    assert _sha256_file(ckpt) == digest_before
    last = read_ledger(tmp_path)[-1]
    assert last["outcome"] == "invalid_run"
    assert last["new_training"] == 0
    assert "RESUME_IDENTITY_MISMATCH" in last["reason"]


def test_ledger_is_append_only_with_monotone_seq(cfg, env, tmp_path):
    row = stage_rows(cfg, "T0")[0]
    for _ in range(2):
        execute_formal_run(
            env, row, cfg_mapping=cfg, config_path=str(CONFIG_PATH),
            campaign_root=tmp_path, device="cpu", identity=dict(IDENTITY),
            test_dose=(1, 1), test_smoke_updates=1)
    ledger = tmp_path / "run_ledger.jsonl"
    prefix = ledger.read_bytes()
    rows = read_ledger(tmp_path)
    assert [r["seq"] for r in rows] == [1, 2]
    assert rows[0]["schema"] == rows[1]["schema"] == \
        "pes2ts_v542_run_ledger_v1"
    execute_formal_run(
        env, row, cfg_mapping=cfg, config_path=str(CONFIG_PATH),
        campaign_root=tmp_path, device="cpu", identity=dict(IDENTITY),
        test_dose=(1, 1), test_smoke_updates=2)
    after = ledger.read_bytes()
    assert after[:len(prefix)] == prefix      # APPEND-only, never rewrite
    assert [r["seq"] for r in read_ledger(tmp_path)] == [1, 2, 3]


def test_run_stage_records_gate_skips_and_is_idempotent(cfg, tmp_path):
    plan = run_stage("T-ATTR", cfg=cfg, campaign_root=tmp_path,
                     formal_validated=True)
    assert plan["gate_skipped_recorded"] == [
        "V542_A1C_S17", "V542_A5C_S17"]
    plan2 = run_stage("T-ATTR", cfg=cfg, campaign_root=tmp_path,
                      formal_validated=True)
    # second pass records nothing new (append-only, no duplicates)
    assert plan2["gate_skipped_recorded"] == []
    assert plan2["already_recorded"] == ["V542_A1C_S17", "V542_A5C_S17"]
    assert len(read_ledger(tmp_path)) == 2


# ---------------------------------------------------------------------------
# sanity band: the exact frozen rule on seed means
# ---------------------------------------------------------------------------
def test_sanity_band_pass_and_breach_with_the_exact_rule():
    a0 = _a0_endpoints()
    e0 = {seed: entry["endpoints"]
          for seed, entry in _v541_reference().items()}
    band = compute_sanity_band(a0, e0)
    assert band["verdict"] == "pass"
    assert band["breached_endpoints"] == []
    assert band["rule"]["rmse_zero_rel_max_pct"] == 2.0
    assert band["rule"]["auprc_abs_max"] == 0.01
    assert band["rule"]["q_corr_abs_max"] == 0.01

    # a large rmse deviation breaches exactly that endpoint
    bad = _a0_endpoints({43: {"rmse_zero": 1.05, "endpoint_auprc": 0.279,
                                "q_corr": 0.9970}})
    band = compute_sanity_band(bad, e0)
    assert band["verdict"] == "breach"
    assert band["breached_endpoints"] == ["rmse_zero"]

    # an AUPRC seed-mean shift of >0.01 breaches AUPRC only
    bad = _a0_endpoints({17: {"rmse_zero": 0.905, "endpoint_auprc": 0.310,
                                "q_corr": 0.9965}})
    band = compute_sanity_band(bad, e0)
    assert band["breached_endpoints"] == ["endpoint_auprc"]

    # a q seed-mean shift of >0.01 breaches q only
    bad = _a0_endpoints({17: {"rmse_zero": 0.905, "endpoint_auprc": 0.276,
                                "q_corr": 0.9600}})
    band = compute_sanity_band(bad, e0)
    assert band["breached_endpoints"] == ["q_corr"]


def test_sanity_band_is_a_seed_mean_not_a_per_seed_check():
    # S17 alone deviates 3% (> 2% per-seed) but the THREE-SEED MEAN stays
    # within the band ⇒ pass.  This pins the fix-5 C5 seed-mean
    # semantics (not per-seed checks).
    a0 = _a0_endpoints({17: {"rmse_zero": 0.927, "endpoint_auprc": 0.276,
                               "q_corr": 0.9965}})
    e0 = {seed: entry["endpoints"]
          for seed, entry in _v541_reference().items()}
    per_seed_out = abs(0.927 - 0.901) / 0.901 * 100.0
    assert per_seed_out > 2.0                    # a per-seed check FAILs
    band = compute_sanity_band(a0, e0)
    assert band["verdict"] == "pass"             # the seed mean PASSES
    rmse = band["endpoints"]["rmse_zero"]
    assert rmse["a0_seed_mean"] == pytest.approx((0.927 + 0.915 + 0.895) / 3)
    assert abs(rmse["relative_diff_pct"]) <= 2.0
    # every per-seed value is reported for audit
    assert set(rmse["a0_by_seed"]) == {"17", "29", "43"}


def test_sanity_band_boundary_is_inclusive():
    import math

    e0 = {seed: entry["endpoints"]
          for seed, entry in _v541_reference().items()}
    # inside-the-band values a hair under each limit ⇒ PASS (the rule is
    # <= inclusive); a hair over each limit ⇒ BREACH of that endpoint.
    inside = {seed: {
        "rmse_zero": math.nextafter(e0[seed]["rmse_zero"] * 1.02,
                                    e0[seed]["rmse_zero"]),
        "endpoint_auprc": math.nextafter(e0[seed]["endpoint_auprc"] + 0.01,
                                         e0[seed]["endpoint_auprc"]),
        "q_corr": math.nextafter(e0[seed]["q_corr"] - 0.01,
                                 e0[seed]["q_corr"]),
    } for seed in (17, 29, 43)}
    band = compute_sanity_band(inside, e0)
    assert band["verdict"] == "pass"
    assert band["breached_endpoints"] == []

    over = {seed: {
        "rmse_zero": math.nextafter(e0[seed]["rmse_zero"] * 1.02,
                                    math.inf),
        "endpoint_auprc": math.nextafter(e0[seed]["endpoint_auprc"] + 0.01,
                                         math.inf),
        "q_corr": math.nextafter(e0[seed]["q_corr"] - 0.01, -math.inf),
    } for seed in (17, 29, 43)}
    band = compute_sanity_band(over, e0)
    assert band["verdict"] == "breach"
    assert set(band["breached_endpoints"]) == {
        "rmse_zero", "endpoint_auprc", "q_corr"}


def test_sanity_band_refuses_missing_or_non_finite_inputs():
    e0 = {seed: entry["endpoints"]
          for seed, entry in _v541_reference().items()}
    with pytest.raises(V542FormalError) as excinfo:
        compute_sanity_band(_a0_endpoints({43: None}), e0)
    assert excinfo.value.code == "V542_BAND_SEEDS_INCOMPLETE"
    broken = _a0_endpoints()
    broken[29]["endpoint_auprc"] = float("nan")
    with pytest.raises(V542FormalError) as excinfo:
        compute_sanity_band(broken, e0)
    assert excinfo.value.code == "V542_BAND_SEEDS_INCOMPLETE"
    # string seed keys are accepted (JSON round-trips)
    str_a0 = {str(k): v for k, v in _a0_endpoints().items()}
    str_e0 = {str(k): v for k, v in e0.items()}
    assert compute_sanity_band(str_a0, str_e0)["verdict"] == "pass"


def test_sanity_band_record_write_and_breach_content(cfg, tmp_path):
    _t0_done(tmp_path)
    e0 = _v541_reference()
    plan = run_sanity_band(cfg, campaign_root=tmp_path,
                           a0_endpoints=_a0_endpoints(), v541_reference=e0,
                           a0_identities={"V542_A0_S17": "1" * 64})
    assert plan["verdict"] == "pass"
    record = read_band_record(tmp_path)
    assert record["verdict"] == "pass"
    assert record["t1_status"] == "unblocked"
    assert record["gate"] == A0_SANITY_BAND_GATE
    assert record["evaluator_identity"] == EVALUATOR_IDENTITY
    assert [entry["run_id"] for entry in record["a0_source_runs"]] == [
        "V542_A0_S17", "V542_A0_S29", "V542_A0_S43"]
    assert record["a0_source_runs"][0]["identity_sha256"] == "1" * 64
    assert [entry["run_id"] for entry in record["v541_e0_reference"]] == [
        "V541_T0R_E0_S17", "V541_T0R_E0_S29", "V541_T0R_E0_S43"]
    # provenance of the V541 archive is carried, never fabricated
    assert record["v541_e0_reference"][0]["metrics_sha256"] == "0" * 64

    # breach: verdict=breach is written (never a pass), the breached
    # endpoints are listed, and T1 stays blocked with the frozen audit
    # categories + identity-layer bumps.
    bad = _a0_endpoints({43: {"rmse_zero": 1.05, "endpoint_auprc": 0.279,
                                "q_corr": 0.9970}})
    plan = run_sanity_band(cfg, campaign_root=tmp_path, a0_endpoints=bad,
                           v541_reference=e0)
    assert plan["verdict"] == "breach"
    record = read_band_record(tmp_path)
    assert record["verdict"] == "breach"         # never overwritten w/ pass
    assert record["breached_endpoints"] == ["rmse_zero"]
    assert record["t1_status"] == "blocked_sanity_band_breach"
    note = record["breach_note"]
    for token in ("parity", "process-head anchor", "R3 mapping", "sampler",
                  "evaluator", "code", "data", "initialization",
                  "source_tree_fingerprint", "data_manifest_sha256",
                  "init_mapping_sha256", "process_head_anchor_sha256",
                  "invalid_run", "no sanity exception"):
        assert token in note
    # and the breach record now blocks T1 for real
    with pytest.raises(V542FormalError) as excinfo:
        run_stage("T1", cfg=cfg, campaign_root=tmp_path,
                  only=["V542_A1_S17"], formal_validated=True)
    assert excinfo.value.code == "V542_SANITY_BAND_BREACH"


def test_sanity_band_requires_t0_before_evaluating(cfg, tmp_path):
    with pytest.raises(V542FormalError) as excinfo:
        run_sanity_band(cfg, campaign_root=tmp_path,
                        a0_endpoints=_a0_endpoints(),
                        v541_reference=_v541_reference())
    assert excinfo.value.code == "V542_SANITY_BAND_T0_INCOMPLETE"


def test_v541_reference_is_located_with_provenance(tmp_path):
    # synthetic V541 archive + synthetic pinned config assets: the
    # reference is LOCATED (paths + SHAs + pin cross-check), never
    # fabricated; a pin mismatch is a typed refusal.
    root = tmp_path / "v541_runs"
    assets = []
    for seed, ckpt_sha in ((17, "a" * 64), (29, "b" * 64), (43, "c" * 64)):
        run_dir = root / f"V541_T0R_E0_S{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(json.dumps({
            "run_outcome": "done",
            "checkpoint_best_sha256": ckpt_sha,
            "move": {"key_rmse_over_zero": {"median_rmse_over_zero": 0.9}},
            "endpoint_head": {"auprc": 0.27},
            "q_head": {"corr_curve_median": 0.99},
        }), encoding="utf-8")
        assets.append({"id": f"V541_E0_S{seed}",
                       "path": f"outputs/.../S{seed}/checkpoint_best.pt",
                       "sha256": ckpt_sha,
                       "source_tier": "v541_e0_runs_final_37900",
                       "run_updates": 37900, "checkpoint_update": 37000})
    cfg = {"backbone_contract": {"a0_migration": {"parity": {
        "assets": assets}}}}
    refs = locate_v541_e0_reference(cfg, v541_runs_root=root)
    assert sorted(refs) == [17, 29, 43]
    assert refs[17]["endpoints"] == {
        "rmse_zero": 0.9, "endpoint_auprc": 0.27, "q_corr": 0.99}
    prov = refs[17]["provenance"]
    assert prov["checkpoint_sha256_pinned"] == "a" * 64
    assert len(prov["metrics_sha256"]) == 64

    # a checkpoint-SHA pin mismatch is typed, never silently accepted
    cfg["backbone_contract"]["a0_migration"]["parity"]["assets"][0][
        "sha256"] = "e" * 64
    with pytest.raises(V542FormalError) as excinfo:
        locate_v541_e0_reference(cfg, v541_runs_root=root)
    assert excinfo.value.code == "V542_BAND_REFERENCE_INVALID"


# ---------------------------------------------------------------------------
# status / dry-run / discipline
# ---------------------------------------------------------------------------
def test_status_reports_the_ledger_state(cfg, tmp_path):
    _t0_done(tmp_path)
    _write_band(tmp_path, _band_pass_record())
    report = status_report(cfg, campaign_root=tmp_path)
    assert report["ledger_rows"] == 3
    assert report["sanity_band"]["verdict"] == "pass"
    by_id = {entry["run_id"]: entry for entry in report["runs"]}
    assert len(by_id) == 26                       # the whole frozen matrix
    assert by_id["V542_A0_S17"]["outcome"] == "done"
    assert by_id["V542_A0_S17"]["new_training"] == 1
    assert by_id["V542_A0_S17"]["identity_sha256"] == "f" * 64
    assert by_id["V542_A0_S17"]["stage"] == "T0"
    assert by_id["V542_A4_S17"]["outcome"] is None  # never decided yet


def test_dry_run_performs_no_training_and_no_writes(tmp_path, capsys):
    root = tmp_path / "campaign"
    code = main(["run", "--stage", "T0", "--dry-run", "--device", "cpu",
                 "--config", str(CONFIG_PATH), "--campaign-root",
                 str(root)])
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["training_performed"] is False
    assert [step["run_id"] for step in plan["steps"]] == [
        "V542_A0_S17", "V542_A0_S29", "V542_A0_S43"]
    assert all(step["action"] == "launch" for step in plan["steps"])
    assert plan["torch_compile"] is False
    assert not root.exists()                      # NOTHING was written

    code = main(["sanity-band", "--dry-run", "--device", "cpu",
                 "--config", str(CONFIG_PATH), "--campaign-root",
                 str(root)])
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["training_performed"] is False
    assert plan["gate"] == A0_SANITY_BAND_GATE
    assert plan["rule"]["rmse_zero_rel_max_pct"] == 2.0
    assert not root.exists()


def test_formal_run_never_calls_torch_compile(cfg, env, tmp_path,
                                              monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("torch.compile is forbidden (ADR-0033 §2.1)")

    monkeypatch.setattr(torch, "compile", _boom)
    row = stage_rows(cfg, "T0")[0]
    result = execute_formal_run(
        env, row, cfg_mapping=cfg, config_path=str(CONFIG_PATH),
        campaign_root=tmp_path, device="cpu", identity=dict(IDENTITY),
        test_dose=(1, 1))
    assert result["ledger_row"]["outcome"] == "done"
    assert result["manifest"]["torch_compile"] is False
    assert result["metrics"]["schedule"]["torch_compile"] is False


def test_cli_stage_gate_skips_are_recorded(tmp_path, capsys):
    # `run --stage T-ATTR` on the real config records both frozen skip
    # reasons under a tmp campaign root without any training.
    root = tmp_path / "campaign"
    code = main(["run", "--stage", "T-ATTR", "--device", "cpu",
                 "--config", str(CONFIG_PATH), "--campaign-root",
                 str(root)])
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["training_performed"] is False
    assert plan["gate_skipped_recorded"] == [
        "V542_A1C_S17", "V542_A5C_S17"]
    rows = [json.loads(line) for line in
            (root / "run_ledger.jsonl").read_text(encoding="utf-8")
            .splitlines() if line.strip()]
    assert [r["reason"] for r in rows] == [
        REASON_A1C_SKIP, REASON_A5C_SKIP]
    assert all(r["new_training"] == 0 for r in rows)
