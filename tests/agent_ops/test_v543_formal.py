"""V5.4.3 WP8 formal runner tests (``scripts/agent_ops/v543_formal.py``).

CPU-fast, synthetic: the runner's ``context_builder`` / ``trainer`` /
``evaluator`` seams are injected, so runplan projection, the pin, the formal
gate, stage orchestration, resume refusal, artifact archiving, the 18-run
ceiling and the single-lane lock are all exercised WITHOUT training and
without data (the plan explicitly forbids launching formal runs from
``--stop-after``-style shortcuts, so no test here ever trains for real).

What is proven here:

* the runplan projects ``run_matrix`` + the pinned T5 finalist list into
  T3(7)/T4(7)/T5(≤4) = ≤18 rows at the frozen 37,900-update dose, and refuses
  more than 2 finalists per track / more than 18 rows / non-formal decoders;
* ``freeze-runplan`` writes ``plan/runplan.json`` + the pin whose
  ``runplan_sha256`` is the sha256 of the document's canonical file bytes
  (the frozen ``governance.records`` sha rule), and any later edit is
  ``V543_FORMAL_RUNPLAN_TAMPERED``; a config change after the freeze is
  ``V543_FORMAL_RUNPLAN_DRIFT``;
* the formal gate cannot be bypassed: with the committed config (null budget)
  ``run-stage`` refuses with the validator's own ``V543_BUDGET_NULL`` BEFORE
  any runplan/training work; a budget-filled config passes and then the
  orchestration runs;
* one formal run archives ``checkpoint_*``/``train_log.jsonl``/
  ``run_records.jsonl``/``metrics.json``/``identity.json``/``eval_report.json``
  /``run_manifest.json``, appends exactly one ledger row with
  ``new_training=1`` and carries the candidate bundle schema the statistics
  lane consumes;
* a resume-identity mismatch is ``V543_FORMAL_RESUME_IDENTITY`` with
  ``new_training=0`` (invalid_run, checkpoint untouched);
* the 18-run ceiling, the ``--only`` filter and ``T0`` (zero trainings) are
  enforced, and the single formal lane is held while a run trains.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import v543_campaign as campaign  # noqa: E402
from scripts.agent_ops import v543_formal as f  # noqa: E402

CONFIG = REPO_ROOT / "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture()
def cfg_host() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def budget_filled_config(tmp_path: Path) -> Path:
    """A copy of the committed config with the WP7 budget table signed.

    Only ``budget.per_candidate_core_time`` differs; the §9.5 instrument, the
    thresholds and the signoff mirror stay byte-identical, so the frozen
    validator's ``--formal`` gate passes on this file.
    """
    doc = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    doc["budget"]["per_candidate_core_time"] = {
        decoder_id: {"core_seconds": 40.0}
        for decoder_id in campaign.FORMAL_DECODERS}
    path = tmp_path / "pes2ts_v543_dual_decoder_50k.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def null_budget_config(tmp_path: Path) -> Path:
    """A copy of the committed config with the WP7 budget table UN-signed.

    The committed config has carried the measured WP7 budget since
    2026-11-03 (formal gate open), so the "null budget blocks the gate"
    tests exercise this copy instead — the refusal path itself is the
    frozen validator's, unchanged by the fill.
    """
    doc = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    doc["budget"]["per_candidate_core_time"] = None
    doc["budget"]["per_candidate_status"] = "pending_wp7_profiling"
    path = tmp_path / "pes2ts_v543_null_budget.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def finalists_file(tmp_path: Path, *, move=("M2", "M1"),
                   process=("P4", "P1")) -> Path:
    path = tmp_path / "t5_finalists.json"
    path.write_text(json.dumps({
        "campaign_id": "PES2TS_V543_DUAL_DECODER_50K",
        "finalists": {"move": list(move), "process": list(process)},
    }), encoding="utf-8")
    return path


class FakeCtx:
    def __init__(self, decoder_id: str, seed: int) -> None:
        self.decoder_id = decoder_id
        self.track = campaign.track_of(decoder_id)
        self.seed = int(seed)
        self.encoder_ckpt_sha256 = "enc-sha"
        self.execution_device = "cpu"
        self.checkpoint = {
            "path": f"outputs/pesn/v542_backbone_50k/runs/V542_A0_S{seed}"
                    "/checkpoint_best.pt",
            "sha256": f"a0-sha-{seed}"}
        self.params = {"active": {"n_trainable_params": 36000}}
        self.labels_v52 = None
        self.endpoint_labels = None
        self.model = None
        self.source = None


def fake_builder(**_kwargs):
    return FakeCtx(str(_kwargs["decoder_id"]), int(_kwargs["seed"]))


def fake_trainer(*, fail=None, lock_probe_root=None, **_kwargs):
    def _train(ctx, *, run_dir, device="cpu"):
        if lock_probe_root is not None:
            # the formal lane must hold the single-lane lock while training
            with pytest.raises(campaign.V543CampaignError) as exc:
                campaign.LaneLock(lock_probe_root).acquire()
            assert exc.value.code == "V543_CAMPAIGN_LANE_BUSY"
        if fail is not None:
            raise fail
        out = Path(run_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_log.jsonl").write_text(
            json.dumps({"event": "train", "update": f.DOSE,
                        "loss": 0.1, "grad_norm": 0.5,
                        "seconds": 0.02}) + "\n", encoding="utf-8")
        (out / "run_records.jsonl").write_text(
            json.dumps({"event": "validate", "update": f.DOSE,
                        "inner_val_loss": 0.25}) + "\n", encoding="utf-8")
        for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
            (out / name).write_bytes(b"fake-checkpoint")
        (out / "metrics.json").write_text(json.dumps({"outcome": "done"}),
                                          encoding="utf-8")
        return {
            "metrics": {
                "outcome": "done", "updates": f.DOSE,
                "is_formal_schedule": True, "best_val_loss": 0.25,
                "best_update": f.DOSE, "wall_seconds": 12.5,
                "seconds_per_update": 0.00033, "encoder_forward_count": 1000,
                "cache": {"enabled": True, "hits": 10, "misses": 5,
                          "builds": 1, "hit_rate": 0.66},
                "params": {"active": {"n_trainable_params": 36000}},
                "loss_coefficients": {"lambda_pair": 0.5},
                "resumed": False,
            },
            "identity": {"seed": int(ctx.seed),
                         "decoder_id": ctx.decoder_id},
            "identity_sha256": f"trainer-{ctx.decoder_id}-S{ctx.seed}",
            "params": {"active": {"n_trainable_params": 36000}},
            "wall_seconds": 12.5,
            "resume": {"requested": False},
        }

    return _train


def fake_evaluator(**_kwargs):
    def _eval(ctx, *, run_id, seed, stage, decoder_id, track, env, cfg,
              encoder_checkpoint, device="cpu"):
        return {
            "schema": f.CANDIDATE_BUNDLE_SCHEMA,
            "run_id": run_id,
            "candidate_id": decoder_id,
            "track": track,
            "seed": int(seed),
            "stage": stage,
            "evaluator": {"module": f.EVALUATOR_MODULE_REL,
                          "code_sha256": "eval-code",
                          "spec_sha256": "eval-spec",
                          "report_schema": f.EVAL_REPORT_SCHEMA},
            "bridge_spec_sha256": "bridge-sha",
            "checkpoint": dict(encoder_checkpoint),
            "eval_report": {"schema": f.EVAL_REPORT_SCHEMA,
                            "per_curve": [], "pooled": {}},
            "bootstrap_inputs": None,
        }

    return _eval


def _frozen(tmp_path: Path, cfg: dict, *, finalists=None,
            config_path: Path | None = None) -> dict:
    plan = f.build_runplan(cfg, t5_finalists=finalists,
                           config_path=config_path or CONFIG)
    return f.freeze_runplan(plan, campaign_root=tmp_path)


# ---------------------------------------------------------------------------
# runplan projection
# ---------------------------------------------------------------------------
def test_runplan_nominal_is_fourteen_rows(cfg):
    plan = f.build_runplan(cfg, config_path=CONFIG)
    rows = f.runplan_rows(plan)
    assert len(rows) == 14
    assert plan["run_matrix"]["t5_finalists_pinned"] is False
    assert plan["run_matrix"]["rows_total"] == 14
    assert [row.run_id for row in f.stage_rows(plan, "T3")] == [
        "V543_M1_S17", "V543_M2_S17", "V543_M3_S17",
        "V543_P1_S17", "V543_P2_S17", "V543_P3_S17", "V543_P4_S17"]
    assert [row.seed for row in f.stage_rows(plan, "T4")] == [29] * 7
    assert all(row.dose == f.DOSE for row in rows)
    assert plan["dose"]["updates"] == 37900
    assert plan["run_matrix"]["total_formal_updates_ceiling"] == 682200


def test_runplan_with_finalists_is_eighteen_rows(cfg, tmp_path):
    finalists = f.load_t5_finalists(finalists_file(tmp_path))
    plan = f.build_runplan(cfg, t5_finalists=finalists, config_path=CONFIG)
    rows = f.runplan_rows(plan)
    assert len(rows) == 18
    assert plan["run_matrix"]["t5_finalists_pinned"] is True
    t5 = f.stage_rows(plan, "T5")
    assert [row.run_id for row in t5] == [
        "V543_M2_S43", "V543_M1_S43", "V543_P4_S43", "V543_P1_S43"]
    assert all(row.seed == 43 for row in t5)
    assert plan["run_matrix"]["t5_rows"] == 4


def test_runplan_matches_the_frozen_expectations(cfg):
    """Every projected row must exist in the validator's appendix-C table."""
    from scripts.agent_ops import v543_config_schema as v

    plan = f.build_runplan(cfg, config_path=CONFIG)
    for row in f.runplan_rows(plan):
        assert row.run_id in v.EXPECTED_RUN_IDS
        assert v.EXPECTED_RUN_TABLE[row.run_id]["dose"] == row.dose
        assert v.EXPECTED_RUN_TABLE[row.run_id]["seed"] == row.seed


@pytest.mark.parametrize("move,process,code", [
    (("M1", "M2", "M3"), ("P1",), "V543_FORMAL_T5_FINALISTS"),
    (("M1", "M1"), ("P1",), "V543_FORMAL_T5_FINALISTS"),
    (("P1",), ("M1",), "V543_FORMAL_T5_FINALISTS"),
    (("M4",), ("P1",), "V543_FORMAL_T5_FINALISTS"),
])
def test_t5_finalists_validation(tmp_path, move, process, code):
    with pytest.raises(f.V543FormalError) as exc:
        f.load_t5_finalists(finalists_file(tmp_path, move=move,
                                           process=process))
    assert exc.value.code == code


def test_runplan_refuses_a_non_formal_decoder(cfg):
    matrix = copy.deepcopy(cfg)
    matrix["run_matrix"]["t3_seed17"] = ["M1", "M2", "M0", "P1", "P2", "P3", "P4"]
    with pytest.raises(f.V543FormalError) as exc:
        f.build_runplan(matrix, config_path=CONFIG)
    assert exc.value.code == f.CODE_RUN_MATRIX


def test_runplan_accepts_the_full_eighteen_row_matrix(cfg_host):
    matrix = copy.deepcopy(cfg_host)
    finalists = {"move": ["M2", "M1"], "process": ["P4", "P1"]}
    plan = f.build_runplan(matrix, t5_finalists=finalists, config_path=CONFIG)
    assert len(f.runplan_rows(plan)) == 18
    assert plan["run_matrix"]["rows_total"] == 18


def test_runplan_refuses_more_than_the_ceiling(cfg, monkeypatch):
    """The ``<= 18`` guard is enforced on the projected row count."""
    monkeypatch.setattr(f, "FORMAL_RUNS_MAX", 17)
    with pytest.raises(f.V543FormalError) as exc:
        f.build_runplan(cfg, t5_finalists={
            "move": ["M2", "M1"], "process": ["P4", "P1"]},
            config_path=CONFIG)
    assert exc.value.code == f.CODE_RUN_MATRIX


def test_stage_rows_rejects_an_unknown_stage(cfg):
    plan = f.build_runplan(cfg, config_path=CONFIG)
    with pytest.raises(f.V543FormalError) as exc:
        f.stage_rows(plan, "T7")
    assert exc.value.code == f.CODE_STAGE_UNKNOWN
    with pytest.raises(f.V543FormalError):
        f.stage_rows(plan, "T1")


def test_only_filter(cfg):
    plan = f.build_runplan(cfg, config_path=CONFIG)
    rows = f.stage_rows(plan, "T3")
    kept = f.filter_only(rows, ["V543_M1_S17"])
    assert [row.run_id for row in kept] == ["V543_M1_S17"]
    with pytest.raises(f.V543FormalError) as exc:
        f.filter_only(rows, ["V543_P4_S29"])
    assert exc.value.code == f.CODE_RUN_UNKNOWN


# ---------------------------------------------------------------------------
# pin freeze / verification
# ---------------------------------------------------------------------------
def test_freeze_writes_the_pin_with_the_frozen_sha_rule(cfg, tmp_path):
    pin = _frozen(tmp_path, cfg)
    pin_file = tmp_path / "plan/runplan.pin.json"
    doc_file = tmp_path / "plan/runplan.json"
    assert pin_file.is_file() and doc_file.is_file()
    assert pin["schema"] == f.PIN_SCHEMA
    for field in ("campaign_id", "config_path", "schema_id", "seeds",
                  "runplan_sha256", "frozen_on", "frozen_by"):
        assert pin.get(field) not in (None, "", [])
    assert pin["runplan_sha256"] == campaign.sha256_file(doc_file)
    assert pin["rows_total"] == 14
    verified = f.verify_runplan_pin(tmp_path, config_path=CONFIG)
    assert verified["runplan_sha256"] == pin["runplan_sha256"]


def test_pin_records_the_t5_pin_state(cfg, tmp_path):
    finalists = f.load_t5_finalists(finalists_file(tmp_path))
    pin = _frozen(tmp_path, cfg, finalists=finalists)
    assert pin["t5_finalists_pinned"] is True
    assert pin["t5_rows"] == 4
    assert pin["rows_total"] == 18


def test_runplan_tamper_is_refused(cfg, tmp_path):
    _frozen(tmp_path, cfg)
    doc_file = tmp_path / "plan/runplan.json"
    document = json.loads(doc_file.read_text(encoding="utf-8"))
    document["rows"][0]["dose"] = 100
    doc_file.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(f.V543FormalError) as exc:
        f.verify_runplan_pin(tmp_path, config_path=CONFIG)
    assert exc.value.code == f.CODE_RUNPLAN_TAMPERED


def test_config_drift_after_the_freeze_is_refused(cfg, tmp_path):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n# drift\n",
                           encoding="utf-8")
    with pytest.raises(f.V543FormalError) as exc:
        f.verify_runplan_pin(tmp_path, config_path=config_path)
    assert exc.value.code == f.CODE_RUNPLAN_DRIFT


def test_missing_runplan_is_refused(tmp_path):
    with pytest.raises(f.V543FormalError) as exc:
        f.load_runplan(tmp_path)
    assert exc.value.code == f.CODE_RUNPLAN_MISSING
    with pytest.raises(f.V543FormalError):
        f.verify_runplan_pin(tmp_path, config_path=CONFIG)


# ---------------------------------------------------------------------------
# formal gate
# ---------------------------------------------------------------------------
def test_formal_gate_refuses_the_committed_null_budget(tmp_path):
    with pytest.raises(f.V543FormalError) as exc:
        f.formal_gate(null_budget_config(tmp_path))
    assert exc.value.code == "V543_BUDGET_NULL"


def test_run_stage_is_blocked_by_the_gate_before_any_runplan_work(tmp_path):
    """No runplan is frozen, yet the refusal is the BUDGET one (gate first)."""
    config_path = null_budget_config(tmp_path)
    cfg = f.load_formal_config(config_path)
    with pytest.raises(f.V543FormalError) as exc:
        f.run_stage("T3", cfg=cfg, config_path=config_path,
                    campaign_root=tmp_path, dry_run=True)
    assert exc.value.code == "V543_BUDGET_NULL"
    assert not (tmp_path / "plan").exists()


def test_budget_filled_config_passes_the_gate(tmp_path):
    config_path = budget_filled_config(tmp_path)
    report = f.formal_gate(config_path)
    assert report["mode"] == "formal"
    assert report["t1_config_ready"] is True


# ---------------------------------------------------------------------------
# stage execution (seams: no real training)
# ---------------------------------------------------------------------------
def _run_stage(tmp_path, cfg, cfg_path, rows_stage="T3", **kwargs):
    return f.run_stage(
        rows_stage, cfg=cfg, config_path=cfg_path, campaign_root=tmp_path,
        device="cpu", env=object(),
        context_builder=kwargs.pop("context_builder", fake_builder),
        trainer=kwargs.pop("trainer", fake_trainer()),
        evaluator=kwargs.pop("evaluator", fake_evaluator()), **kwargs)


def test_formal_run_archives_every_artifact(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    result = _run_stage(tmp_path, cfg, config_path, only=["V543_M1_S17"])
    assert result["training_performed"] is True
    summary = result["executed"][0]
    assert summary["run_id"] == "V543_M1_S17"
    assert summary["outcome"] == "done"
    assert summary["reason"] == f.REASON_FORMAL_DONE
    assert summary["new_training"] == 1
    run_dir = tmp_path / "runs" / "V543_M1_S17"
    record = json.loads((run_dir / "run_manifest.json").read_text(
        encoding="utf-8"))
    assert record["dose"]["updates"] == f.DOSE
    assert record["dose"]["is_formal_schedule"] is True
    assert record["eval"]["report_schema"] == f.EVAL_REPORT_SCHEMA
    for name in ("checkpoint_best.pt", "checkpoint_last.pt",
                 "train_log.jsonl", "run_records.jsonl", "metrics.json",
                 "identity.json", "eval_report.json", "run_manifest.json",
                 "run_record.json"):
        assert (run_dir / name).is_file(), name
    identity = json.loads((run_dir / "identity.json").read_text(
        encoding="utf-8"))
    assert identity["identity_sha256"]
    assert identity["encoder"]["sha256"] == "a0-sha-17"
    assert identity["runplan"]["runplan_sha256"]
    assert identity["evaluator"]["report_schema"] == f.EVAL_REPORT_SCHEMA
    ledger = f.read_ledger(tmp_path)
    assert len(ledger) == 1
    assert ledger[0]["new_training"] == 1
    assert ledger[0]["outcome"] == "done"
    assert f.new_training_count(f.latest_ledger_state(ledger)) == 1


def test_candidate_bundle_carries_the_stats_lane_contract(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    _run_stage(tmp_path, cfg, config_path, only=["V543_P1_S17"])
    bundle = json.loads((tmp_path / "runs/V543_P1_S17/eval_report.json")
                        .read_text(encoding="utf-8"))
    assert bundle["schema"] == f.CANDIDATE_BUNDLE_SCHEMA
    for key in ("run_id", "candidate_id", "track", "seed", "stage",
                "eval_report", "bootstrap_inputs"):
        assert key in bundle
    assert bundle["candidate_id"] == "P1"
    assert bundle["track"] == "process"
    assert bundle["seed"] == 17
    assert bundle["stage"] == "T3"
    for key in ("code_sha256", "spec_sha256", "report_schema"):
        assert key in bundle["evaluator"]
    assert set(bundle["checkpoint"]) == {"path", "sha256"}
    assert bundle["eval_report"]["schema"] == f.EVAL_REPORT_SCHEMA


def test_resume_identity_mismatch_is_an_invalid_run(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    failure = f.V543FormalError(f.CODE_RESUME_IDENTITY,
                               "checkpoint_last belongs to a different run",
                               outcome="invalid_run")
    with pytest.raises(f.V543FormalError) as exc:
        _run_stage(tmp_path, cfg, config_path, only=["V543_M2_S17"],
                   trainer=fake_trainer(fail=failure))
    assert exc.value.code == f.CODE_RESUME_IDENTITY
    assert exc.value.outcome == "invalid_run"
    ledger = f.read_ledger(tmp_path)
    assert len(ledger) == 1
    assert ledger[0]["outcome"] == "invalid_run"
    assert ledger[0]["new_training"] == 0
    record = json.loads((tmp_path / "runs/V543_M2_S17/run_manifest.json")
                        .read_text(encoding="utf-8"))
    assert record["outcome"] == "invalid_run"
    assert record["new_training"] == 0


def test_trainer_refusal_is_mapped_to_the_formal_lane(tmp_path, cfg):
    """A campaign-seam identity refusal keeps its typed code and outcome."""
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)

    def trainer(ctx, *, run_dir, device="cpu"):
        raise campaign.V543CampaignError(
            "V543_CAMPAIGN_RESUME_IDENTITY", "mismatch", outcome="invalid_run")

    with pytest.raises(f.V543FormalError) as exc:
        _run_stage(tmp_path, cfg, config_path, only=["V543_M3_S17"],
                   trainer=trainer)
    assert exc.value.code == f.CODE_RESUME_IDENTITY
    assert f.read_ledger(tmp_path)[0]["new_training"] == 0


def test_evaluator_failure_is_archived_and_refused(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)

    def evaluator(**_kwargs):
        def _eval(ctx, **inner):
            raise RuntimeError("evaluator exploded")

        return _eval

    with pytest.raises(f.V543FormalError) as exc:
        _run_stage(tmp_path, cfg, config_path, only=["V543_M1_S17"],
                   evaluator=evaluator())
    assert exc.value.code == f.CODE_EVAL_FAILED
    record = json.loads((tmp_path / "runs/V543_M1_S17/run_manifest.json")
                        .read_text(encoding="utf-8"))
    assert record["eval"]["evaluated"] is False
    assert "evaluator exploded" in record["eval"]["error"]
    assert record["new_training"] == 1
    assert not (tmp_path / "runs/V543_M1_S17/eval_report.json").exists()


def test_already_done_rows_are_skipped_idempotently(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    _run_stage(tmp_path, cfg, config_path, only=["V543_M1_S17"])
    second = _run_stage(tmp_path, cfg, config_path, only=["V543_M1_S17"])
    assert second["training_performed"] is False
    assert second["steps"][0]["action"] == "skip"
    assert second["steps"][0]["reason"] == f.REASON_ALREADY_DONE
    assert len(f.read_ledger(tmp_path)) == 1


def test_dry_run_performs_no_training_and_writes_no_ledger(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    result = _run_stage(tmp_path, cfg, config_path, dry_run=True)
    assert result["dry_run"] is True
    assert result["training_performed"] is False
    assert len(result["steps"]) == 7
    assert all(step["action"] == "launch" for step in result["steps"])
    assert not f.read_ledger(tmp_path)
    assert not (tmp_path / "runs").exists()


def test_eighteen_run_ceiling_is_enforced(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    for index in range(18):
        f.append_ledger_row(tmp_path, {
            "run_id": f"V543_FAKE_S{index}", "outcome": "done",
            "new_training": 1, "stage": "T3", "seed": 17})
    with pytest.raises(f.V543FormalError) as exc:
        _run_stage(tmp_path, cfg, config_path, only=["V543_M1_S17"])
    assert exc.value.code == f.CODE_RUN_MATRIX


def test_t0_trains_nothing_and_needs_the_baseline(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    with pytest.raises(f.V543FormalError) as exc:
        f.run_stage("T0", cfg=cfg, config_path=config_path,
                    campaign_root=tmp_path, env=object(),
                    context_builder=fake_builder, trainer=fake_trainer(),
                    evaluator=fake_evaluator())
    assert exc.value.code == f.CODE_BASELINE_MISSING
    baseline = tmp_path / f.BASELINE_REL
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(json.dumps({"schema": "pesn_v543_m0p0_baseline_v1"}),
                        encoding="utf-8")
    result = f.run_stage("T0", cfg=cfg, config_path=config_path,
                         campaign_root=tmp_path, env=object(),
                         context_builder=fake_builder, trainer=fake_trainer(),
                         evaluator=fake_evaluator())
    assert result["reason"] == f.REASON_T0_VERIFIED
    assert result["training_performed"] is False
    assert result["verified"] is True
    assert not f.read_ledger(tmp_path)


def test_training_holds_the_single_lane(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    _run_stage(tmp_path, cfg, config_path, only=["V543_M1_S17"],
               trainer=fake_trainer(lock_probe_root=tmp_path))


def test_run_stage_requires_a_verified_pin(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    doc_file = tmp_path / "plan/runplan.json"
    document = json.loads(doc_file.read_text(encoding="utf-8"))
    document["rows"][0]["decoder_id"] = "M2"
    doc_file.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(f.V543FormalError) as exc:
        _run_stage(tmp_path, cfg, config_path, dry_run=True)
    assert exc.value.code == f.CODE_RUNPLAN_TAMPERED


# ---------------------------------------------------------------------------
# status / list
# ---------------------------------------------------------------------------
def test_status_reports_the_ledger_and_the_gate(tmp_path, cfg):
    config_path = budget_filled_config(tmp_path)
    _frozen(tmp_path, cfg, config_path=config_path)
    _run_stage(tmp_path, cfg, config_path, only=["V543_M1_S17"])
    report = f.status_report(campaign_root=tmp_path, config_path=config_path)
    assert report["n_core"] == 1
    assert report["formal_runs_max"] == 18
    assert report["runplan"]["rows"] == 14
    assert report["runplan_pin"]["verified"] is True
    assert report["gate"]["passed"] is True
    assert "V543_M1_S17" in report["done"]
    assert report["per_run"]["V543_M1_S17"]["eval_report"]


def test_status_surfaces_the_null_budget_gate(tmp_path):
    report = f.status_report(campaign_root=tmp_path,
                             config_path=null_budget_config(tmp_path))
    assert report["gate"]["checked"] is True
    assert report["gate"]["passed"] is False
    assert report["gate"]["code"] == "V543_BUDGET_NULL"


def test_list_prints_the_pinned_matrix(cfg, capsys, monkeypatch, tmp_path):
    finalists = finalists_file(tmp_path)
    # the CLI resolves the campaign root from --campaign-root, so freeze first
    plan = f.build_runplan(cfg, t5_finalists=f.load_t5_finalists(finalists),
                           config_path=CONFIG)
    f.freeze_runplan(plan, campaign_root=tmp_path)
    code = f.main(["list", "--campaign-root", str(tmp_path),
                   "--t5-finalists", str(finalists)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows_total"] == 18
    assert payload["t5_pinned"] is True


def test_runplan_cli_is_a_dry_run(capsys, tmp_path):
    code = f.main(["runplan", "--campaign-root", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert len(payload["rows"]) == 14
    assert not (tmp_path / "plan").exists()


def test_cli_refuses_the_null_budget_before_training(capsys, tmp_path):
    code = f.main(["run-stage", "T3", "--campaign-root", str(tmp_path),
                   "--config", str(null_budget_config(tmp_path)),
                   "--dry-run"])
    assert code == 2
    assert "V543_BUDGET_NULL" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
