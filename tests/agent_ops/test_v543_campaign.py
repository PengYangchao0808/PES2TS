"""V5.4.3 WP7 campaign harness tests (``scripts/agent_ops/v543_campaign.py``).

CPU-fast, synthetic: every test injects the ``context_builder`` /
``leg_runner`` seams of the harness, so the smoke / profile / diagnostic
orchestration, the §8.1 pass criteria, the §8.3/§8.5 resource circuit breakers,
the single-lane lock and the WP7 budget artifact are exercised WITHOUT torch,
without data and without a GPU.

What is proven here:

* the frozen operational guards refuse any drift (single lane / swap / OOM /
  confirm / QC / host limit) BEFORE anything runs;
* a smoke passes only when loss is finite, the gradient is alive, the packet
  cache is wired and the kill→resume cycle continues under ONE identity digest
  (each failure mode is a separate case, and the failing record is archived);
* the profile artifact carries the exact per-candidate core time, the
  measurement-environment SHA and the M0/P0 denominator rule, and refuses a
  partial / zero measurement (the values the ``--formal`` gate will type as
  ``V543_BUDGET_INVALID``);
* resource circuit breakers (host RSS, OOM, swap, wall clock) and the
  technical-result discipline (``rank_excluded`` / ``scientific_result``) are
  typed refusals;
* the single-lane lock is exclusive and re-acquirable;
* ``--formal`` delegates to the frozen validator (the real config is
  ``V543_BUDGET_NULL``) and ``--dry-run`` writes nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import v543_campaign as c  # noqa: E402

CONFIG = REPO_ROOT / "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml"


# ---------------------------------------------------------------------------
# fixtures: a synthetic context + leg runner (the two documented seams)
# ---------------------------------------------------------------------------
class FakeCtx:
    def __init__(self, decoder_id: str, seed: int) -> None:
        self.decoder_id = decoder_id
        self.track = c.track_of(decoder_id)
        self.seed = int(seed)
        self.encoder_ckpt_sha256 = "enc-sha"
        self.execution_device = "cpu"
        self.checkpoint = {"path": "/ckpt/a0.pt", "sha256": "a0-sha"}
        self.params = {"active": {"n_trainable_params": 36000}}
        self.labels_v52 = None
        self.endpoint_labels = None
        self.model = None
        self.source = None


class FakeRunner:
    """Writes the trainer's own files and returns the metrics a real leg does.

    ``losses`` / ``grads`` are consumed update-by-update; ``identity`` is the
    run identity digest both legs must share; ``resume_start`` is the first
    update the resumed leg writes.
    """

    def __init__(self, *, kill_at: int, updates: int, losses=None,
                 grads=None, identity: str = "id-1",
                 resume_identity: str | None = None,
                 cache: dict | None = None,
                 resume_start: int | None = None) -> None:
        self.kill_at = int(kill_at)
        self.updates = int(updates)
        self.losses = losses
        self.grads = grads
        self.identity = identity
        self.resume_identity = resume_identity or identity
        self.cache = cache or {"enabled": True, "mode": "frozen_encoder_only",
                               "hits": 0, "misses": 64, "builds": 2,
                               "hit_rate": 0.0, "hit_rate_target": 0.8,
                               "identity": {"encoder_ckpt_sha256": "enc-sha"}}
        self.resume_start = (self.kill_at + 1 if resume_start is None
                             else int(resume_start))
        self.calls: list[dict] = []

    def __call__(self, ctx, *, out_dir, stop_after, device="cpu", resume=False):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        start = self.resume_start if resume else 1
        written = 0
        with (out / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            for update in range(start, int(stop_after) + 1):
                loss = (self.losses[update - 1] if self.losses
                        else 1.0 / (1.0 + update))
                grad = (self.grads[update - 1] if self.grads else 0.5)
                handle.write(json.dumps({
                    "event": "train", "update": update, "loss": loss,
                    "grad_norm": grad, "seconds": 0.01}) + "\n")
                written += 1
        with (out / "run_records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "validate",
                                     "update": int(stop_after),
                                     "inner_val_loss": 0.5}) + "\n")
        for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
            (out / name).write_bytes(b"fake-checkpoint")
        identity_sha = self.resume_identity if resume else self.identity
        self.calls.append({"resume": bool(resume), "written": written,
                           "start": start})
        return {
            "metrics": {
                "outcome": "done", "updates": int(stop_after),
                "is_formal_schedule": False,
                "cache": dict(self.cache),
                "encoder_forward_count": 64,
                "params": {"active": {"n_trainable_params": 36000}},
            },
            "identity": {"seed": int(ctx.seed),
                         "decoder_id": ctx.decoder_id},
            "identity_sha256": identity_sha,
            "trainer_config": {"seed": int(ctx.seed)},
            "resume": {"requested": bool(resume)},
            "params": {"active": {"n_trainable_params": 36000}},
            "wall_seconds": 0.5,
        }


def fake_builder(**_kwargs):
    return FakeCtx(str(_kwargs["decoder_id"]), int(_kwargs["seed"]))


@pytest.fixture()
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _mut(cfg: dict, fn) -> dict:
    import copy

    out = copy.deepcopy(cfg)
    fn(out)
    return out


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path,value", [
    (("governance", "authorization", "single_lane"), False),
    (("governance", "authorization", "swap_allowed"), True),
    (("governance", "authorization", "confirm_authorized"), True),
    (("governance", "authorization", "sealed_authorized"), True),
    (("governance", "authorization", "new_qc_calls"), 1),
    (("budget", "single_lane"), False),
    (("budget", "swap_max"), 1),
    (("budget", "oom_kill_max"), 1),
    (("budget", "host_limit_gib"), 16),
    (("resources", "host_limit_gib"), 16),
    (("resources", "single_lane"), False),
    (("resources", "swap_allowed"), True),
])
def test_guards_refuse_drift(cfg, path, value):
    mutated = _mut(cfg, lambda doc: _set(doc, path, value))
    with pytest.raises(c.V543CampaignError) as exc:
        c.assert_operational_guards(mutated)
    assert exc.value.code == "V543_CAMPAIGN_GUARD"


def _set(doc: dict, path, value) -> None:
    node = doc
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def test_guards_accept_the_frozen_config(cfg):
    guard = c.assert_operational_guards(cfg)
    assert guard["single_lane"] is True
    assert guard["swap_max"] == 0
    assert guard["oom_kill_max"] == 0
    assert guard["host_limit_gib"] == 12
    assert guard["confirm_read"] is False
    assert guard["new_qc_calls"] == 0


def test_resolve_decoders_rejects_conditional_and_control():
    with pytest.raises(c.V543CampaignError) as exc:
        c.resolve_decoders(["M4"])
    assert exc.value.code == "V543_CAMPAIGN_DECODER_UNKNOWN"
    for outside in ("M5", "P5", "P4-C", "M9"):
        with pytest.raises(c.V543CampaignError):
            c.resolve_decoders([outside])
    assert c.resolve_decoders(["M1", "P1"]) == ["M1", "P1"]
    assert c.resolve_decoders(None) == list(c.FORMAL_DECODERS)


def test_track_of():
    assert c.track_of("M1") == "move"
    assert c.track_of("P4") == "process"
    with pytest.raises(c.V543CampaignError):
        c.track_of("Q1")


def test_cli_seed_and_device_validation():
    assert c.validate_cli_seed(17) == 17
    for bad in (99, "x"):
        with pytest.raises(c.V543CampaignError) as exc:
            c.validate_cli_seed(bad)
        assert exc.value.code == "V543_CAMPAIGN_SEED_FROZEN"
    assert c.validate_cli_device("cpu") == "cpu"
    with pytest.raises(c.V543CampaignError) as exc:
        c.validate_cli_device("tpu")
    assert exc.value.code == "V543_CAMPAIGN_DEVICE_UNKNOWN"


def test_technical_result_discipline():
    c.assert_technical_result({"rank_excluded": True,
                               "scientific_result": False})
    for row in ({"rank_excluded": False, "scientific_result": False},
                {"rank_excluded": True, "scientific_result": True}):
        with pytest.raises(c.V543CampaignError) as exc:
            c.assert_technical_result(row)
        assert exc.value.code == "V543_CAMPAIGN_SCIENCE_LEAK"


# ---------------------------------------------------------------------------
# single-lane lock
# ---------------------------------------------------------------------------
def test_lane_lock_is_exclusive_and_reacquirable(tmp_path):
    first = c.LaneLock(tmp_path).acquire()
    try:
        with pytest.raises(c.V543CampaignError) as exc:
            c.LaneLock(tmp_path).acquire()
        assert exc.value.code == "V543_CAMPAIGN_LANE_BUSY"
        assert exc.value.outcome == "resource_failure"
    finally:
        first.release()
    second = c.LaneLock(tmp_path).acquire()
    second.release()
    assert (tmp_path / "plan" / "lane.lock").is_file()


def test_lane_lock_context_manager(tmp_path):
    with c.LaneLock(tmp_path):
        assert (tmp_path / "plan" / "lane.lock").is_file()
    with c.LaneLock(tmp_path):
        pass


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
def _smoke(tmp_path, cfg, *, runner=None, updates=6, kill_at=3, **kwargs):
    runner = runner or FakeRunner(kill_at=kill_at, updates=updates)
    return c.smoke_candidate(
        object(), "M1", cfg=cfg, campaign_root=tmp_path, seed=17,
        updates=updates, kill_at=kill_at, device="cpu",
        cfg_path=CONFIG, context_builder=fake_builder, leg_runner=runner,
        **kwargs)


def test_smoke_passes_and_archives_the_record(tmp_path, cfg):
    row = _smoke(tmp_path, cfg)
    assert row["schema"] == c.RUN_RECORD_SCHEMA
    assert row["kind"] == "smoke"
    assert row["run_id"] == "V543_SMOKE_M1_S17"
    assert row["track"] == "move"
    assert row["checks"]["passed"] is True
    assert row["checks"]["failures"] == []
    for criterion in ("loss_finite", "grad_alive", "cache_wired",
                      "resume_proven"):
        assert row["checks"]["criteria"][criterion]["passed"] is True
    assert row["rank_excluded"] is True
    assert row["scientific_result"] is False
    assert row["is_formal_schedule"] is False
    assert row["dose"]["is_formal_schedule"] is False
    assert row["reason_code"] == c.REASON_SMOKE_OK
    assert row["resume"]["first_resumed_update"] == 4
    assert row["shas"]["checkpoint_best_sha256"]
    assert row["environment_sha256"]
    run_dir = Path(row["run_dir"])
    assert (run_dir / "run_record.json").is_file()
    report = tmp_path / "smoke" / "smoke_report.jsonl"
    rows = c.read_jsonl(report)
    assert len(rows) == 1 and rows[0]["run_id"] == row["run_id"]


def test_smoke_fails_on_nonfinite_loss(tmp_path, cfg):
    runner = FakeRunner(kill_at=3, updates=6,
                        losses=[0.1, float("nan"), 0.1, 0.1, 0.1, 0.1])
    with pytest.raises(c.V543CampaignError) as exc:
        _smoke(tmp_path, cfg, runner=runner)
    assert exc.value.code == "V543_CAMPAIGN_SMOKE_FAILED"
    assert exc.value.outcome == "invalid_run"
    record = json.loads((tmp_path / "smoke" / "V543_SMOKE_M1_S17"
                         / "run_record.json").read_text(encoding="utf-8"))
    assert record["checks"]["passed"] is False
    assert "loss_finite" in record["checks"]["failures"]


def test_smoke_fails_when_the_gradient_is_dead(tmp_path, cfg):
    runner = FakeRunner(kill_at=3, updates=6, grads=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(c.V543CampaignError) as exc:
        _smoke(tmp_path, cfg, runner=runner)
    assert exc.value.code == "V543_CAMPAIGN_SMOKE_FAILED"
    record = json.loads((tmp_path / "smoke" / "V543_SMOKE_M1_S17"
                         / "run_record.json").read_text(encoding="utf-8"))
    assert "grad_alive" in record["checks"]["failures"]


def test_smoke_fails_when_the_cache_is_disabled(tmp_path, cfg):
    runner = FakeRunner(kill_at=3, updates=6,
                        cache={"enabled": False, "hits": 0, "misses": 0,
                               "builds": 0, "hit_rate": 0.0})
    with pytest.raises(c.V543CampaignError):
        _smoke(tmp_path, cfg, runner=runner)
    record = json.loads((tmp_path / "smoke" / "V543_SMOKE_M1_S17"
                         / "run_record.json").read_text(encoding="utf-8"))
    assert "cache_wired" in record["checks"]["failures"]


def test_smoke_fails_when_the_resume_identity_differs(tmp_path, cfg):
    runner = FakeRunner(kill_at=3, updates=6, identity="id-1",
                        resume_identity="id-2")
    with pytest.raises(c.V543CampaignError):
        _smoke(tmp_path, cfg, runner=runner)
    record = json.loads((tmp_path / "smoke" / "V543_SMOKE_M1_S17"
                         / "run_record.json").read_text(encoding="utf-8"))
    assert "resume_proven" in record["checks"]["failures"]
    assert record["resume"]["identity_match"] is False


def test_smoke_fails_when_the_resume_skips_updates(tmp_path, cfg):
    runner = FakeRunner(kill_at=3, updates=6, resume_start=5)
    with pytest.raises(c.V543CampaignError):
        _smoke(tmp_path, cfg, runner=runner)
    record = json.loads((tmp_path / "smoke" / "V543_SMOKE_M1_S17"
                         / "run_record.json").read_text(encoding="utf-8"))
    assert record["checks"]["criteria"]["resume_proven"]["value"][
        "first_resumed_update"] == 5


@pytest.mark.parametrize("kill_at", [0, 3])
def test_smoke_kill_at_bounds(tmp_path, cfg, kill_at):
    with pytest.raises(c.V543CampaignError) as exc:
        _smoke(tmp_path, cfg, updates=3, kill_at=kill_at)
    assert exc.value.code == "V543_CAMPAIGN_KILL_AT"


def test_smoke_seams_receive_both_legs(tmp_path, cfg):
    runner = FakeRunner(kill_at=2, updates=5)
    _smoke(tmp_path, cfg, runner=runner, updates=5, kill_at=2)
    assert [call["resume"] for call in runner.calls] == [False, True]
    assert runner.calls[1]["start"] == 3


# ---------------------------------------------------------------------------
# profile + WP7 budget artifact
# ---------------------------------------------------------------------------
def _profile_all(tmp_path, cfg):
    rows = []
    for decoder_id in c.PROFILE_DECODERS:
        runner = FakeRunner(kill_at=1, updates=c.PROFILE_WINDOW_UPDATES)
        rows.append(c.profile_candidate(
            object(), decoder_id, cfg=cfg, campaign_root=tmp_path, seed=17,
            warmup=c.PROFILE_WARMUP_UPDATES,
            steady=c.PROFILE_STEADY_UPDATES, device="cpu", cfg_path=CONFIG,
            context_builder=fake_builder, leg_runner=runner))
    artifact = c.build_budget_profile(
        rows, cfg=cfg, warmup=c.PROFILE_WARMUP_UPDATES,
        steady=c.PROFILE_STEADY_UPDATES,
        environment=c.environment_record("cpu"), cfg_path=CONFIG)
    return rows, artifact


def test_profile_row_reports_the_window_and_core_time(tmp_path, cfg):
    row = c.profile_candidate(
        object(), "P1", cfg=cfg, campaign_root=tmp_path, seed=17, device="cpu",
        cfg_path=CONFIG, context_builder=fake_builder,
        leg_runner=FakeRunner(kill_at=1, updates=c.PROFILE_WINDOW_UPDATES))
    assert row["kind"] == "profile"
    assert row["window"]["warmup_updates"] == 20
    assert row["window"]["steady_updates"] == 40
    assert row["window"]["measured_updates"] == 40
    assert row["core_time"]["core_seconds"] == pytest.approx(0.4)
    assert row["core_time"]["core_seconds_per_update"] == pytest.approx(0.01)
    assert row["core_time"]["updates_per_second"] == pytest.approx(100.0)
    assert row["rank_excluded"] is True
    assert row["budget_written"] is False
    assert "per_candidate_core_time" not in row
    assert row["environment_sha256"]


def test_budget_artifact_covers_the_seven_candidates(tmp_path, cfg):
    rows, artifact = _profile_all(tmp_path, cfg)
    assert artifact["schema"] == c.BUDGET_PROFILE_SCHEMA
    measurements = artifact["measurements"]
    table = measurements["per_candidate_core_time"]
    assert set(table) == set(c.FORMAL_DECODERS)
    for decoder_id, entry in table.items():
        assert entry["core_seconds"] > 0
        assert entry["track"] == c.track_of(decoder_id)
        assert entry["core_ratio_vs_denominator"] is not None
        assert entry["denominator_decoder_id"] == (
            "M0" if c.track_of(decoder_id) == "move" else "P0")
    assert measurements["denominators"]["move"]["decoder_id"] == "M0"
    assert measurements["denominators"]["process"]["decoder_id"] == "P0"
    assert measurements["environment_sha256"]
    assert artifact["measurements_sha256"]
    assert artifact["ready_for_budget_fill"] is True
    # the shape the frozen validator accepts: {decoder: {core_seconds: >0}}
    for entry in table.values():
        assert isinstance(entry["core_seconds"], float)


def test_budget_artifact_refuses_a_partial_measurement(tmp_path, cfg):
    rows, _artifact = _profile_all(tmp_path, cfg)
    partial = [row for row in rows if row["decoder_id"] != "P4"]
    with pytest.raises(c.V543CampaignError) as exc:
        c.build_budget_profile(partial, cfg=cfg, warmup=20, steady=40,
                               environment=c.environment_record("cpu"),
                               cfg_path=CONFIG)
    assert exc.value.code == "V543_CAMPAIGN_BUDGET_PROFILE_INCOMPLETE"


def test_budget_artifact_refuses_a_missing_denominator(tmp_path, cfg):
    rows, _artifact = _profile_all(tmp_path, cfg)
    without_baseline = [row for row in rows if row["decoder_id"] != "P0"]
    with pytest.raises(c.V543CampaignError) as exc:
        c.build_budget_profile(without_baseline, cfg=cfg, warmup=20, steady=40,
                               environment=c.environment_record("cpu"),
                               cfg_path=CONFIG)
    assert exc.value.code == "V543_CAMPAIGN_BUDGET_PROFILE_INCOMPLETE"


def test_budget_artifact_refuses_a_non_positive_measurement(tmp_path, cfg):
    rows, _artifact = _profile_all(tmp_path, cfg)
    broken = [dict(row) for row in rows]
    for row in broken:
        if row["decoder_id"] == "M2":
            row["core_time"] = dict(row["core_time"], core_seconds=0.0)
    with pytest.raises(c.V543CampaignError) as exc:
        c.build_budget_profile(broken, cfg=cfg, warmup=20, steady=40,
                               environment=c.environment_record("cpu"),
                               cfg_path=CONFIG)
    assert exc.value.code == "V543_CAMPAIGN_BUDGET_PROFILE_INVALID"


def test_profile_window_must_contain_steady_updates(tmp_path, cfg):
    with pytest.raises(c.V543CampaignError) as exc:
        c.profile_candidate(
            object(), "M1", cfg=cfg, campaign_root=tmp_path, seed=17,
            warmup=0, steady=0, device="cpu", cfg_path=CONFIG,
            context_builder=fake_builder,
            leg_runner=FakeRunner(kill_at=1, updates=2))
    assert exc.value.code == "V543_CAMPAIGN_PROFILE_WINDOW"


# ---------------------------------------------------------------------------
# resource circuit breakers
# ---------------------------------------------------------------------------
def test_resource_breakers():
    c.assert_resources(host_peak_mb=1024.0, oom_kills=0, swap_used=0,
                       wall_seconds=10.0)
    with pytest.raises(c.V543CampaignError) as exc:
        c.assert_resources(host_peak_mb=13000.0, oom_kills=0, swap_used=0,
                           wall_seconds=10.0)
    assert exc.value.outcome == "resource_failure"
    with pytest.raises(c.V543CampaignError):
        c.assert_resources(host_peak_mb=1.0, oom_kills=1, swap_used=0,
                           wall_seconds=1.0)
    with pytest.raises(c.V543CampaignError):
        c.assert_resources(host_peak_mb=1.0, oom_kills=0, swap_used=1,
                           wall_seconds=1.0)
    with pytest.raises(c.V543CampaignError):
        c.assert_resources(host_peak_mb=1.0, oom_kills=0, swap_used=0,
                           wall_seconds=c.MAX_WALL_SECONDS + 1.0)


def test_smoke_records_the_resource_checks(tmp_path, cfg):
    row = _smoke(tmp_path, cfg)
    resources = row["resources"]
    assert resources["host_limit_gib"] == 12
    assert resources["oom_kill_count"] == 0
    assert resources["swap_used"] == 0
    assert resources["checks"]["host_limit_mb"] == 12288.0


# ---------------------------------------------------------------------------
# diagnostic
# ---------------------------------------------------------------------------
def test_diagnostic_screen_allowlist_excludes_h3():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    screens = c.diagnostic_screens(cfg)
    assert set(screens) == {"diagnostic_loss_drop_min_pct"}
    assert all(not key.startswith(("move_h3", "process_h3", "h3_"))
               for key in screens)


def test_diagnostic_applies_only_the_loss_drop_screen():
    train_rows = [{"loss": 1.0, "grad_norm": 0.5, "seconds": 0.01},
                  {"loss": 0.5, "grad_norm": 0.4, "seconds": 0.01}]
    val_rows = [{"inner_val_loss": 1.0}, {"inner_val_loss": 0.90}]
    verdict = c.evaluate_diagnostic(val_rows=val_rows,
                                    train_rows=train_rows,
                                    loss_drop_min_pct=5.0)
    assert verdict["passed"] is True
    assert verdict["loss_drop_pct"] == pytest.approx(10.0)
    assert (verdict["checks"]["evaluator_side_screens"]["passed"] is None)
    flat = c.evaluate_diagnostic(val_rows=[{"inner_val_loss": 1.0},
                                           {"inner_val_loss": 0.99}],
                                 train_rows=train_rows,
                                 loss_drop_min_pct=5.0)
    assert flat["passed"] is False
    assert flat["failures"] == ["loss_drop"]


# ---------------------------------------------------------------------------
# formal gate delegation + dry-run
# ---------------------------------------------------------------------------
def test_formal_gate_delegates_to_the_frozen_validator(tmp_path):
    # The committed config has carried the measured WP7 budget since
    # 2026-11-03 (gate open), so the delegation test exercises a null-budget
    # copy — the refusal path itself is the frozen validator's, unchanged.
    import yaml as _yaml

    doc = _yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    doc["budget"]["per_candidate_core_time"] = None
    doc["budget"]["per_candidate_status"] = "pending_wp7_profiling"
    null_path = tmp_path / "pes2ts_v543_null_budget.yaml"
    null_path.write_text(
        _yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    with pytest.raises(c.V543CampaignError) as exc:
        c.formal_gate(null_path)
    assert exc.value.code == "V543_BUDGET_NULL"


def test_dry_runs_write_nothing(tmp_path, cfg):
    plan = c.plan_smoke(cfg=cfg, campaign_root=tmp_path, decoders=["M1"],
                        seed=17, updates=25, kill_at=12)
    assert plan["dry_run"] is True and plan["training_performed"] is False
    assert plan["steps"][0]["run_id"] == "V543_SMOKE_M1_S17"
    profile_plan = c.plan_profile(cfg=cfg, campaign_root=tmp_path)
    assert profile_plan["window"]["total_updates"] == 60
    assert [step["decoder_id"] for step in profile_plan["steps"]] == list(
        c.PROFILE_DECODERS)
    assert not (tmp_path / "smoke").exists()
    assert not (tmp_path / "profile").exists()
    assert not (tmp_path / "plan").exists()


def test_main_dry_run_prints_and_writes_nothing(tmp_path, capsys):
    code = c.main(["smoke", "--dry-run", "--decoder", "M1", "--updates", "25",
                   "--kill-at", "12", "--campaign-root", str(tmp_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "smoke" and payload["dry_run"] is True
    assert not (tmp_path / "smoke").exists()


def test_environment_sha_is_stable_and_pid_free():
    env = c.environment_record("cpu")
    first = c.environment_sha256(env)
    env["pid"] = env.get("pid", 0) + 1
    assert c.environment_sha256(env) == first
