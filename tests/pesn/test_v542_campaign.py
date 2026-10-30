"""V5.4.2 WP4 campaign technical-measurement harness tests (P3 lane C).

CPU-fast gates for ``scripts/agent_ops/v542_campaign.py`` (plan §7):

* **profile** — a 2+2-step warm-up variant runs on the REAL fixture env
  and emits every required field (``core_seconds`` / ``wall_seconds`` /
  ``peak_vram_mb`` / ``host_rss_mb`` / ``edges_per_second`` / device /
  torch+CUDA versions / exact config+source identity digests) with the
  warm-up updates excluded — and NEVER touches the budget (no
  ``per_candidate_core_time`` anywhere; the frozen budget record bytes
  stay identical);
* **smoke** — 2 updates, kill at step 1, resume and complete: the
  kill-and-resume cycle is proven (manifest ``resume`` event, identity
  digest match across legs), the JSONL record is APPEND-only and every
  row is ``rank_excluded=true`` / ``scientific_result=false`` (typed
  ``V542_CAMPAIGN_SCIENCE_LEAK`` refusal otherwise);
* **diagnostic** — each of the four frozen ENGINEERING screens is applied
  correctly on synthetic metric dicts (pass/fail/boundary plus the
  non-finite case) and H3-style effect gating is refused (typed
  ``V542_DIAG_H3_GATE_FORBIDDEN``); a tiny real A0 control run writes a
  decision row with an explicit reason code; the non-finite gradient
  screen CONSUMES the trainer run-record ``grad_finite_all_finite``
  field (false → eliminate with the frozen reason code; true → advance;
  absent → ``evaluated=false`` + ``gradient_screen_unavailable``, and
  advancing is refused unless ``--allow-unevaluated-grad-screen`` —
  which records the override and is refused in formal mode);
* **typed refusals** — ``--formal`` is typed-blocked by the frozen
  validator (``V542_BUDGET_NULL``), ``--dry-run`` performs no training,
  profiling a diagnostic variant without the conditional flag is
  refused, forbidden config overrides are refused; an invalid ``--seed``
  (not in {17,29,43}), an unavailable ``--device cuda`` and an unknown
  candidate id are typed CLI refusals (non-zero exit, no traceback),
  while unexpected exceptions still surface (never swallowed);
* **discipline** — no ``torch.compile``; single-lane / no-swap /
  confirm-never-read / zero-QC guards asserted on the real config AND on
  every output row.

The fixture environment is the REAL six-table path reused from the frozen
V5.3-MP trainer tests (``_build_env``); all step counts are tiny and come
from harness parameters / test-only run-length overrides — the frozen
config and trainer math are never modified.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.offline.training.v541_identity import (  # noqa: E402
    sha256_json,
)
from scripts.agent_ops.v542_campaign import (  # noqa: E402
    CAMPAIGN_SCHEMA,
    MAIN_BACKBONES,
    REASON_DIAG_ALL_PASS,
    REASON_DIAG_DEAD_FRACTION,
    REASON_DIAG_GRAD_UNAVAILABLE,
    REASON_DIAG_LOSS_DROP,
    REASON_DIAG_NON_FINITE,
    REASON_DIAG_RMSE_ZERO,
    V542CampaignError,
    _trainer_config,
    apply_diagnostic_screens,
    assert_operational_guards,
    assert_technical_result,
    diagnostic_metrics_from_run,
    diagnostic_thresholds,
    formal_gate,
    load_campaign_config,
    main,
    plan_diagnostic,
    plan_profile,
    plan_smoke,
    resolve_candidates,
    run_diagnostic,
    run_profile,
    run_smoke,
    validate_cli_device,
    validate_cli_seed,
)
from tests.pesn.test_r3_v53mp_trainer import (  # noqa: E402
    DEFAULT_FOLDS,
    _build_env,
)

SEED = 17
#: A small but plausible run-identity SHA block (the CLI lane assembles
#: the real 14-field one; the harness records/compares whatever it is given).
IDENTITY: Dict[str, str] = {
    "source_manifest_sha256": "a" * 64,
    "config_sha256": "b" * 64,
    "common_protocol_sha256": "c" * 64,
}

CONFIG_PATH = REPO / "config/campaigns/pes2ts_v542_backbone_50k.yaml"
BUDGET_TEMPLATE = REPO / (
    "config/campaigns/v542_templates/budget_policy.template.json")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def config() -> Dict[str, Any]:
    return load_campaign_config(CONFIG_PATH)


@pytest.fixture()
def env(tmp_path):
    """The REAL fixture env: six-table CSVs → KeyAtomDatasetV3 → labels."""
    return _build_env(tmp_path, DEFAULT_FOLDS)


def _identities(candidates) -> Dict[str, Dict[str, str]]:
    return {c: dict(IDENTITY) for c in candidates}


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _base_metrics(**over: Any) -> Dict[str, Any]:
    """A passing synthetic diagnostic metric dict (15% loss drop)."""
    base: Dict[str, Any] = {
        "initial_inner_val_total_loss": 1.0,
        "best_inner_val_total_loss": 0.85,
        "loss_finite": True,
        "grad_finite": True,
        "dead_fraction": 0.10,
        "rmse_zero": 1.0,
    }
    base.update(over)
    return base


CONTROL = {"dead_fraction": 0.10, "rmse_zero": 1.0}


# ---------------------------------------------------------------------------
# operational guards: single lane / no swap / confirm / QC
# ---------------------------------------------------------------------------
def test_operational_guards_accept_the_real_config(config):
    guards = assert_operational_guards(config)
    assert guards == {
        "single_lane": True,
        "swap_max": 0,
        "oom_kill_max": 0,
        "confirm_read": False,
        "sealed_authorized": False,
        "new_qc_calls": 0,
    }


@pytest.mark.parametrize("mutate, fragment", [
    (lambda cfg: cfg["governance"]["authorization"].update(
        {"single_lane": False}), "single_lane"),
    (lambda cfg: cfg["governance"]["authorization"].update(
        {"swap_allowed": True}), "swap"),
    (lambda cfg: cfg["governance"]["authorization"].update(
        {"confirm_authorized": True}), "confirm"),
    (lambda cfg: cfg["governance"]["authorization"].update(
        {"sealed_authorized": True}), "sealed"),
    (lambda cfg: cfg["governance"]["authorization"].update(
        {"new_qc_calls": 1}), "new_qc_calls"),
    (lambda cfg: cfg["budget"].update({"single_lane": False}), "single_lane"),
    (lambda cfg: cfg["budget"].update({"swap_max": 1}), "swap_max"),
    (lambda cfg: cfg["budget"].update({"oom_kill_max": 1}), "oom_kill_max"),
])
def test_operational_guards_refuse_drift(config, mutate, fragment):
    mutated = copy.deepcopy(config)
    mutate(mutated)
    with pytest.raises(V542CampaignError) as excinfo:
        assert_operational_guards(mutated)
    assert excinfo.value.code == "V542_CAMPAIGN_GUARD"
    assert fragment in excinfo.value.detail


def test_resolve_candidates_refuses_unflagged_diagnostic_variants():
    assert resolve_candidates(None, conditional=False) == list(MAIN_BACKBONES)
    assert resolve_candidates(["A0", "A1-C"], conditional=True) == \
        ["A0", "A1-C"]
    with pytest.raises(V542CampaignError) as excinfo:
        resolve_candidates(["A0", "A1-C"], conditional=False)
    assert excinfo.value.code == "V542_CAMPAIGN_CONDITIONAL_REQUIRED"
    with pytest.raises(V542CampaignError) as excinfo:
        resolve_candidates(["A5-C"], conditional=False)
    assert excinfo.value.code == "V542_CAMPAIGN_CONDITIONAL_REQUIRED"
    with pytest.raises(V542CampaignError) as excinfo:
        resolve_candidates(["A9"], conditional=False)
    assert excinfo.value.code == "V542_CAMPAIGN_CANDIDATE_UNKNOWN"


# ---------------------------------------------------------------------------
# PROFILE: 2+2 warm-up variant, required fields, budget untouched
# ---------------------------------------------------------------------------
def test_profile_two_plus_two_emits_required_fields_without_touching_budget(
        env, config, tmp_path):
    root = tmp_path / "campaign"
    budget_before = _sha256_file(BUDGET_TEMPLATE)
    config_before = _sha256_file(CONFIG_PATH)

    rows = run_profile(
        env, config=config, campaign_root=root, candidates=["A0"],
        identity_by_candidate=_identities(["A0"]), device="cpu", seed=SEED,
        w0=2, w1=2, warmup=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == CAMPAIGN_SCHEMA
    assert row["kind"] == "profile"
    assert row["run_id"] == "V542_PROFILE_A0_S17"
    assert row["candidate_id"] == "A0" and row["seed"] == SEED

    # --- required measurement fields --------------------------------------
    for key in ("core_seconds", "wall_seconds", "peak_vram_mb",
                "host_rss_mb", "edges_per_second", "device",
                "torch_version", "cuda_version", "config_identity_sha256",
                "config_full_sha256", "identity", "identity_sha256"):
        assert key in row, key
    assert row["core_seconds"] > 0.0
    assert row["wall_seconds"] > 0.0
    assert row["host_rss_mb"] > 0.0
    assert row["peak_vram_mb"] is None          # CPU run: no VRAM
    assert row["edges_measured"] > 0
    assert row["edges_per_second"] > 0.0
    assert row["device"] == "cpu"
    assert row["torch_version"] == str(torch.__version__)
    assert len(row["config_identity_sha256"]) == 64
    assert len(row["config_full_sha256"]) == 64

    # --- run identity block attached ---------------------------------------
    assert row["identity"] == IDENTITY
    assert row["identity_sha256"] == sha256_json(IDENTITY)

    # --- warm-up excluded: 2 W0 + 2 W1 with 1 warm-up => measured [2,3,4] --
    assert row["window"]["total_updates"] == 4
    assert row["window"]["warmup_updates"] == 1
    assert row["window"]["measured_updates"] == [2, 3, 4]
    assert row["window"]["w0_measured_updates"] == 1
    assert row["window"]["w1_measured_updates"] == 2
    assert row["w0_core_seconds"] > 0.0 and row["w1_core_seconds"] > 0.0

    # --- technical measurement, never a scientific result ------------------
    assert row["rank_excluded"] is True
    assert row["scientific_result"] is False
    assert row["torch_compile"] is False
    assert row["run_outcome"] == "done"
    assert row["is_formal_schedule"] is False
    assert row["reason_code"] == "profile_ok"

    # --- guard block on the row -------------------------------------------
    guards = row["guards"]
    assert guards["single_lane"] is True and guards["swap_max"] == 0
    assert guards["confirm_read"] is False and guards["new_qc_calls"] == 0
    assert row["single_lane"] is True
    assert row["confirm_read"] is False
    assert row["new_qc_calls"] == 0

    # --- budget NEVER touched ---------------------------------------------
    assert row["budget_written"] is False
    assert "per_candidate_core_time" not in row
    assert _sha256_file(BUDGET_TEMPLATE) == budget_before
    assert _sha256_file(CONFIG_PATH) == config_before

    # --- append-only report -------------------------------------------------
    report = root / "profile" / "profile_report.jsonl"
    assert report.is_file()
    stored = _read_jsonl(report)
    assert len(stored) == 1 and stored[0]["run_id"] == "V542_PROFILE_A0_S17"


def test_profile_refuses_diagnostic_variant_without_conditional(env, config,
                                                                tmp_path):
    with pytest.raises(V542CampaignError) as excinfo:
        run_profile(
            env, config=config, campaign_root=tmp_path / "campaign",
            candidates=["A0", "A1-C"], conditional=False,
            identity_by_candidate=_identities(["A0", "A1-C"]), device="cpu")
    assert excinfo.value.code == "V542_CAMPAIGN_CONDITIONAL_REQUIRED"
    # the refusal happens BEFORE any training/writes
    assert not (tmp_path / "campaign").exists()
    with pytest.raises(V542CampaignError) as excinfo:
        plan_profile(config=config, campaign_root=tmp_path / "campaign",
                     candidates=["A5-C"], conditional=False)
    assert excinfo.value.code == "V542_CAMPAIGN_CONDITIONAL_REQUIRED"


def test_profile_conditional_variants_are_accepted_with_the_flag(config):
    plan = plan_profile(config=config, campaign_root="/tmp/never",
                        candidates=["A1-C", "A5-C"], conditional=True)
    assert [s["candidate_id"] for s in plan["steps"]] == ["A1-C", "A5-C"]
    assert all(s["conditional"] for s in plan["steps"])


def test_profile_override_allowlist_refuses_forbidden_knobs(config):
    with pytest.raises(V542CampaignError) as excinfo:
        _trainer_config(config, candidate_id="A0", seed=SEED, freeze=1,
                        joint=1, smoke_updates=None, device="cpu",
                        overrides={"lambda_q": 0.4})
    assert excinfo.value.code == "V542_CAMPAIGN_OVERRIDE_FORBIDDEN"
    with pytest.raises(V542CampaignError) as excinfo:
        _trainer_config(config, candidate_id="A0", seed=SEED, freeze=1,
                        joint=1, smoke_updates=None, device="cpu",
                        overrides={"grad_clip": 99.0})
    assert excinfo.value.code == "V542_CAMPAIGN_OVERRIDE_FORBIDDEN"
    # run-length knobs remain legal
    cfg = _trainer_config(config, candidate_id="A0", seed=SEED, freeze=1,
                          joint=1, smoke_updates=None, device="cpu",
                          overrides={"val_every": 1})
    assert cfg.val_every == 1


# ---------------------------------------------------------------------------
# SMOKE: 2 steps, kill at 1, resume, append-only, rank_excluded
# ---------------------------------------------------------------------------
def test_smoke_two_steps_kills_at_one_resumes_and_completes(env, config,
                                                            tmp_path):
    root = tmp_path / "campaign"
    rows = run_smoke(
        env, config=config, campaign_root=root, candidates=["A0"],
        identity_by_candidate=_identities(["A0"]), device="cpu", seed=SEED,
        updates=2, kill_at=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "smoke" and row["matrix"] == "main"
    assert row["run_id"] == "V542_SMOKE_A0_S17"
    assert row["dose"] == 2
    assert row["run_outcome"] == "done"

    # --- kill-and-resume cycle proven --------------------------------------
    kr = row["kill_resume"]
    assert kr["kill_at"] == 1
    assert kr["leg1"]["dose"] == 1 and kr["leg1"]["updates_run"] == 1
    assert kr["leg2"]["dose"] == 2 and kr["leg2"]["updates_run"] == 1
    assert kr["leg2"]["resumed"] is True
    assert kr["resume_events"] >= 1
    # the trainer-side digest is over the CANONICAL identity block
    # (caller SHA block + canonical run-identity fields), not the raw dict;
    # both legs agreeing on it is asserted inside smoke_candidate
    assert len(kr["identity_sha256"]) == 64
    manifest = _read_jsonl(Path(kr["manifest_path"]))
    assert [r["event"] for r in manifest].count("resume") == 1

    # --- technical measurement --------------------------------------------
    assert row["rank_excluded"] is True
    assert row["scientific_result"] is False
    assert row["reason_code"] == "smoke_kill_resume_proven"
    assert row["guards"]["single_lane"] is True
    assert row["guards"]["swap_max"] == 0
    assert row["confirm_read"] is False and row["new_qc_calls"] == 0
    assert row["identity"] == IDENTITY

    # --- APPEND-only report ------------------------------------------------
    report = root / "smoke" / "smoke_report.jsonl"
    first_bytes = report.read_bytes()
    assert len(_read_jsonl(report)) == 1

    rows2 = run_smoke(
        env, config=config, campaign_root=root, candidates=["A0"],
        identity_by_candidate=_identities(["A0"]), device="cpu", seed=SEED,
        updates=2, kill_at=1)
    assert len(rows2) == 1
    assert report.read_bytes().startswith(first_bytes)   # never rewritten
    assert len(_read_jsonl(report)) == 2


def test_smoke_conditional_variants_recorded_separately(env, config,
                                                        tmp_path):
    root = tmp_path / "campaign"
    rows = run_smoke(
        env, config=config, campaign_root=root, candidates=["A1-C"],
        conditional=True, identity_by_candidate=_identities(["A1-C"]),
        device="cpu", seed=SEED, updates=2, kill_at=1)
    assert len(rows) == 1 and rows[0]["matrix"] == "conditional"
    assert rows[0]["candidate_id"] == "A1-C"
    conditional_report = root / "smoke" / "smoke_conditional_report.jsonl"
    main_report = root / "smoke" / "smoke_report.jsonl"
    assert len(_read_jsonl(conditional_report)) == 1
    assert not main_report.exists()   # the main matrix stays clean


def test_smoke_typed_refusal_when_reported_as_a_scientific_result():
    with pytest.raises(V542CampaignError) as excinfo:
        assert_technical_result(
            {"rank_excluded": False, "scientific_result": False})
    assert excinfo.value.code == "V542_CAMPAIGN_SCIENCE_LEAK"
    with pytest.raises(V542CampaignError) as excinfo:
        assert_technical_result(
            {"rank_excluded": True, "scientific_result": True})
    assert excinfo.value.code == "V542_CAMPAIGN_SCIENCE_LEAK"
    assert_technical_result(
        {"rank_excluded": True, "scientific_result": False})


# ---------------------------------------------------------------------------
# DIAGNOSTIC: all four engineering screens + non-finite + H3 refusal
# ---------------------------------------------------------------------------
def test_diagnostic_screen_pass_advances_to_full_dose():
    decision = apply_diagnostic_screens(_base_metrics(), CONTROL)
    assert decision["decision"] == "advance_full_dose"
    assert decision["reason_code"] == REASON_DIAG_ALL_PASS
    assert decision["failing_screens"] == []
    assert decision["h3_effect_thresholds_applied"] is False
    assert decision["ranking_applied"] is False
    assert decision["rank_excluded"] is True
    assert decision["elimination_only"] is True
    assert [s["screen"] for s in decision["screens"]] == [
        "non_finite_loss_or_gradient",
        "best_inner_val_total_loss_drop",
        "dead_fraction_vs_a0_at_2000",
        "rmse_zero_vs_a0_at_2000"]
    # no ranking surface can leak into the decision record
    dumped = json.dumps(decision)
    assert '"rank"' not in dumped and "winner" not in dumped


@pytest.mark.parametrize("metrics, reason", [
    (_base_metrics(loss_finite=False), REASON_DIAG_NON_FINITE),
    (_base_metrics(grad_finite=False), REASON_DIAG_NON_FINITE),
    # best inner-val improved only 2% (< 5%) over the initial evaluation
    (_base_metrics(best_inner_val_total_loss=0.98), REASON_DIAG_LOSS_DROP),
    # dead fraction 0.31 > A0@2000 (0.10) + 0.20
    (_base_metrics(dead_fraction=0.31), REASON_DIAG_DEAD_FRACTION),
    # RMSE/zero degraded 11% (> 10%) vs A0@2000
    (_base_metrics(rmse_zero=1.11), REASON_DIAG_RMSE_ZERO),
])
def test_diagnostic_screens_eliminate_on_each_frozen_rule(metrics, reason):
    decision = apply_diagnostic_screens(metrics, CONTROL)
    assert decision["decision"] == "eliminated"
    assert decision["reason_code"] == reason
    assert decision["failing_screens"]


def test_diagnostic_screens_boundary_values_pass():
    # exactly 5.0% drop and exactly +0.20 dead margin land ON the limit
    # (inclusive pass); RMSE is kept clearly inside 10% because float
    # 1.10-1.00 lands microscopically ABOVE the decimal 10% boundary
    decision = apply_diagnostic_screens(
        _base_metrics(best_inner_val_total_loss=0.95, dead_fraction=0.30,
                      rmse_zero=1.09), CONTROL)
    assert decision["decision"] == "advance_full_dose"
    assert decision["reason_code"] == REASON_DIAG_ALL_PASS
    statuses = {s["screen"]: s["status"] for s in decision["screens"]}
    assert set(statuses.values()) == {"pass"}


def test_diagnostic_screens_record_not_evaluated_without_evaluator_metrics():
    metrics = {"initial_inner_val_total_loss": 1.0,
               "best_inner_val_total_loss": 0.80,
               "loss_finite": True, "grad_finite": None,
               "dead_fraction": None, "rmse_zero": None}
    control = {"dead_fraction": None, "rmse_zero": None}
    # the gradient half is NOT measured (no trainer grad field, no
    # evaluator grad_finite): advancing to full dose is REFUSED (typed)
    # unless the explicit override is given.
    with pytest.raises(V542CampaignError) as excinfo:
        apply_diagnostic_screens(metrics, control)
    assert excinfo.value.code == "V542_DIAG_GRAD_SCREEN_UNEVALUATED"
    # the override advances WITH the unavailability recorded on the row
    decision = apply_diagnostic_screens(
        metrics, control, allow_unevaluated_grad_screen=True)
    assert decision["decision"] == "advance_full_dose"
    assert decision["gradient_screen_unavailable"] is True
    assert decision["unevaluated_grad_screen_override"] is True
    by_screen = {s["screen"]: s for s in decision["screens"]}
    assert by_screen["non_finite_loss_or_gradient"]["evaluated"] is False
    assert by_screen["non_finite_loss_or_gradient"]["passed"] is True
    assert by_screen["non_finite_loss_or_gradient"]["reason_code"] == \
        "grad_screen_unavailable_trainer_records_lack_field"
    assert by_screen["dead_fraction_vs_a0_at_2000"]["evaluated"] is False
    assert by_screen["rmse_zero_vs_a0_at_2000"]["evaluated"] is False
    assert by_screen["dead_fraction_vs_a0_at_2000"]["reason_code"] == \
        "screen_not_evaluated_metrics_absent"


# ---------------------------------------------------------------------------
# C3 — the gradient screen CONSUMES the trainer run-record flag
# ``grad_finite_all_finite`` (never optional evaluator metrics only)
# ---------------------------------------------------------------------------
def _write_synthetic_run(run_dir: Path, *, grad_records=None,
                         losses=(0.5,), inner_val=(1.0, 0.85)) -> Path:
    """A minimal run-artifact set: inner-val records in run_records.jsonl
    (+ optional trainer ``grad_finite_all_finite`` rows) and train rows
    in train_log.jsonl."""
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [{"event": "inner_val", "inner_val_total_loss": loss}
               for loss in inner_val]
    records.extend(grad_records or [])
    (run_dir / "run_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    trains = [{"event": "train", "loss": loss} for loss in losses]
    (run_dir / "train_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in trains) + "\n", encoding="utf-8")
    return run_dir


def test_diagnostic_grad_screen_false_in_run_records_eliminates(tmp_path):
    run_dir = _write_synthetic_run(
        tmp_path / "run", grad_records=[
            {"event": "stage_switch", "grad_finite_all_finite": False,
             "grad_nonfinite_params": ["core.lin0.weight", "head.w_v"]}])
    metrics = diagnostic_metrics_from_run(run_dir)
    assert metrics["grad_finite_all_finite"] is False
    assert metrics["grad_nonfinite_params"] == [
        "core.lin0.weight", "head.w_v"]
    # evaluator metrics are IRRELEVANT for the gradient half: the
    # trainer run record alone drives the screen
    decision = apply_diagnostic_screens(metrics, CONTROL)
    assert decision["decision"] == "eliminated"
    assert decision["reason_code"] == REASON_DIAG_NON_FINITE
    grad_screen = next(s for s in decision["screens"]
                       if s["screen"] == "non_finite_loss_or_gradient")
    assert grad_screen["evaluated"] is True
    assert grad_screen["grad_finite_all_finite"] is False
    assert grad_screen["grad_nonfinite_params"] == [
        "core.lin0.weight", "head.w_v"]


def test_diagnostic_grad_screen_true_in_run_records_advances(tmp_path):
    run_dir = _write_synthetic_run(
        tmp_path / "run", grad_records=[
            {"event": "train", "grad_finite_all_finite": True},
            {"event": "inner_val", "inner_val_total_loss": 0.90,
             "grad_finite_all_finite": True}])
    metrics = diagnostic_metrics_from_run(run_dir)
    assert metrics["grad_finite_all_finite"] is True
    decision = apply_diagnostic_screens(metrics, CONTROL)
    assert decision["decision"] == "advance_full_dose"
    assert decision["reason_code"] == REASON_DIAG_ALL_PASS
    assert decision["gradient_screen_unavailable"] is False
    assert decision["unevaluated_grad_screen_override"] is False


def test_diagnostic_grad_screen_missing_field_flags_unavailable(tmp_path):
    run_dir = _write_synthetic_run(tmp_path / "run")   # no grad field
    metrics = diagnostic_metrics_from_run(run_dir)
    assert metrics["grad_finite_all_finite"] is None
    assert metrics["grad_nonfinite_params"] is None
    # cannot advance without the explicit override ...
    with pytest.raises(V542CampaignError) as excinfo:
        apply_diagnostic_screens(metrics, CONTROL)
    assert excinfo.value.code == "V542_DIAG_GRAD_SCREEN_UNEVALUATED"
    # ... and cannot eliminate either: the screen is evaluated=false with
    # the explicit reason code and the decision row is flagged
    decision = apply_diagnostic_screens(
        metrics, CONTROL, allow_unevaluated_grad_screen=True)
    assert decision["decision"] == "advance_full_dose"
    assert decision["gradient_screen_unavailable"] is True
    assert decision["unevaluated_grad_screen_override"] is True
    grad_screen = next(s for s in decision["screens"]
                       if s["screen"] == "non_finite_loss_or_gradient")
    assert grad_screen["evaluated"] is False
    assert grad_screen["passed"] is True
    assert grad_screen["reason_code"] == REASON_DIAG_GRAD_UNAVAILABLE


def test_diagnostic_grad_override_is_typed_recorded_and_refused_formally(
        tmp_path, capsys):
    # the override records itself on the decision row (typed + recorded)
    decision = apply_diagnostic_screens(
        _base_metrics(grad_finite_all_finite=None, grad_finite=None),
        CONTROL, allow_unevaluated_grad_screen=True)
    assert decision["unevaluated_grad_screen_override"] is True
    assert decision["gradient_screen_unavailable"] is True
    # ... and is REFUSED in formal mode
    code = main(["diagnostic", "--formal",
                 "--allow-unevaluated-grad-screen",
                 "--campaign-root", str(tmp_path / "campaign"),
                 "--config", str(CONFIG_PATH)])
    assert code == 2
    err = capsys.readouterr().err
    assert "V542_CAMPAIGN_GRAD_OVERRIDE_FORMAL_FORBIDDEN" in err
    assert not (tmp_path / "campaign").exists()


def test_diagnostic_refuses_h3_style_effect_gating():
    with pytest.raises(V542CampaignError) as excinfo:
        apply_diagnostic_screens(
            _base_metrics(), CONTROL,
            h3_effect_thresholds={"h3_move_rmse_rel_min_pct": 3.0})
    assert excinfo.value.code == "V542_DIAG_H3_GATES_FORBIDDEN"
    # smuggling an H3/effect key through the thresholds mapping is refused too
    with pytest.raises(V542CampaignError) as excinfo:
        apply_diagnostic_screens(
            _base_metrics(), CONTROL,
            {"loss_drop_min_pct": 5.0, "h3_move_rmse_rel_min_pct": 3.0})
    assert excinfo.value.code == "V542_DIAG_H3_GATE_FORBIDDEN_THRESHOLD"


def test_diagnostic_thresholds_mirror_the_signed_config(config):
    assert diagnostic_thresholds(config) == {
        "loss_drop_min_pct": 5.0,
        "dead_fraction_margin_vs_a0": 0.20,
        "rmse_zero_rel_max_pct": 10.0,
    }


def test_diagnostic_runner_tiny_a0_control_writes_decision(env, config,
                                                           tmp_path):
    root = tmp_path / "campaign"
    rows = run_diagnostic(
        env, config=config, campaign_root=root, candidates=["A0"],
        identity_by_candidate=_identities(["A0"]), device="cpu", seed=SEED,
        updates=2, w0=1, w1=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "diagnostic"
    assert row["run_id"] == "V542_DIAG_A0_S17"
    assert row["is_control"] is True
    assert row["control_run_id"] == "V542_DIAG_A0_S17"
    assert row["dose"] == {"w0_updates": 1, "w1_updates": 1,
                           "total_updates": 2}
    assert row["identity"] == IDENTITY
    assert row["run_outcome"] == "done"
    assert row["rank_excluded"] is True
    assert row["scientific_result"] is False

    # the single inner-val sweep is both initial and best → 0% drop → the
    # frozen "not learning" screen eliminates with an explicit reason code
    decision = row["decision"]
    assert decision["decision"] == "eliminated"
    assert decision["reason_code"] == REASON_DIAG_LOSS_DROP
    assert decision["h3_effect_thresholds_applied"] is False
    assert decision["ranking_applied"] is False
    assert row["reason_code"] == REASON_DIAG_LOSS_DROP
    assert row["diag_metrics"]["loss_finite"] is True

    report = root / "diagnostics" / "diagnostic_report.jsonl"
    assert len(_read_jsonl(report)) == 1


def test_diagnostic_requires_the_a0_control(config):
    with pytest.raises(V542CampaignError) as excinfo:
        plan_diagnostic(config=config, campaign_root="/tmp/never",
                        candidates=["A1", "A2"])
    assert excinfo.value.code == "V542_CAMPAIGN_CANDIDATES"


# ---------------------------------------------------------------------------
# --formal typed-blocked / --dry-run performs no training
# ---------------------------------------------------------------------------
def test_formal_is_refused_by_the_technical_harness():
    # WP4 record instances exist and the formal gate passes, so this
    # harness must delegate: it never executes formal runs.
    with pytest.raises(V542CampaignError) as excinfo:
        formal_gate(CONFIG_PATH)
    assert excinfo.value.code == "V542_CAMPAIGN_FORMAL_DELEGATED"


def test_formal_cli_is_typed_blocked(tmp_path, capsys):
    code = main(["smoke", "--formal",
                 "--campaign-root", str(tmp_path / "campaign"),
                 "--config", str(CONFIG_PATH)])
    assert code == 2
    err = capsys.readouterr().err
    assert "V542_CAMPAIGN_FORMAL_DELEGATED" in err
    assert not (tmp_path / "campaign").exists()


@pytest.mark.parametrize("command, extra", [
    ("profile", ["--w0", "2", "--w1", "2", "--warmup", "1"]),
    ("smoke", ["--updates", "2", "--kill-at", "1"]),
    ("diagnostic", ["--updates", "2", "--w0", "1", "--w1", "1"]),
])
def test_dry_run_prints_the_plan_without_training(tmp_path, capsys, command,
                                                  extra):
    root = tmp_path / "campaign"
    code = main([command, "--dry-run", "--device", "cpu",
                 "--campaign-root", str(root), "--config",
                 str(CONFIG_PATH)] + extra)
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["kind"] == command
    assert plan["dry_run"] is True
    assert plan["training_performed"] is False
    assert plan["steps"]
    assert plan["guards"]["single_lane"] is True
    assert not root.exists()          # NOTHING was written

    if command == "profile":
        assert plan["window"] == {"w0_updates": 2, "w1_updates": 2,
                                  "warmup_updates": 1, "total_updates": 4}
        assert plan["budget_written"] is False
    elif command == "smoke":
        assert plan["kill_at"] == 1
        assert plan["steps"][0]["legs"] == [
            {"leg": 1, "dose": 1, "resume": False},
            {"leg": 2, "dose": 2, "resume": True}]
        assert plan["rank_excluded"] is True
    else:
        assert plan["control_run_id"] == "V542_DIAG_A0_S17"
        assert plan["steps"][0]["candidate_id"] == "A0"
        assert plan["steps"][0]["is_control"] is True
        assert plan["h3_effect_gates_applied"] is False
        assert plan["ranking_applied"] is False


# ---------------------------------------------------------------------------
# no torch.compile anywhere in the campaign path
# ---------------------------------------------------------------------------
def test_campaign_never_calls_torch_compile(env, config, tmp_path,
                                            monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("torch.compile is forbidden (ADR-0033 §2.1)")

    monkeypatch.setattr(torch, "compile", _boom)
    rows = run_profile(
        env, config=config, campaign_root=tmp_path / "campaign",
        candidates=["A0"], identity_by_candidate=_identities(["A0"]),
        device="cpu", seed=SEED, w0=1, w1=1, warmup=0)
    assert rows[0]["run_outcome"] == "done"
    assert rows[0]["torch_compile"] is False


# ---------------------------------------------------------------------------
# C5 — CLI robustness: typed seed/device validation + top-level handler
# ---------------------------------------------------------------------------
def test_validate_cli_seed_accepts_frozen_seeds_and_refuses_others():
    assert validate_cli_seed(17) == 17
    assert validate_cli_seed(29) == 29
    assert validate_cli_seed(43) == 43
    assert validate_cli_seed("17") == 17
    for bad in (18, 99, 0, -1, "abc"):
        with pytest.raises(V542CampaignError) as excinfo:
            validate_cli_seed(bad)
        assert excinfo.value.code == "V542_CAMPAIGN_SEED_FROZEN"


def test_validate_cli_device_cpu_ok_cuda_unavailable_refused(monkeypatch):
    assert validate_cli_device("cpu") == "cpu"
    # simulate a CPU-only host: --device cuda is a typed refusal
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(V542CampaignError) as excinfo:
        validate_cli_device("cuda")
    assert excinfo.value.code == "V542_CAMPAIGN_DEVICE_UNAVAILABLE"
    with pytest.raises(V542CampaignError) as excinfo:
        validate_cli_device("cuda:0")
    assert excinfo.value.code == "V542_CAMPAIGN_DEVICE_UNAVAILABLE"
    # an unknown device string is its own typed refusal
    with pytest.raises(V542CampaignError) as excinfo:
        validate_cli_device("mps")
    assert excinfo.value.code == "V542_CAMPAIGN_DEVICE_UNKNOWN"


def test_cli_bad_seed_is_a_typed_refusal_exit_nonzero(tmp_path, capsys):
    code = main(["diagnostic", "--dry-run", "--seed", "99",
                 "--campaign-root", str(tmp_path / "campaign"),
                 "--config", str(CONFIG_PATH)])
    assert code == 2
    err = capsys.readouterr().err
    assert "V542_CAMPAIGN_SEED_FROZEN" in err
    assert "Traceback" not in err
    assert not (tmp_path / "campaign").exists()


def test_cli_bad_device_is_a_typed_refusal_exit_nonzero(tmp_path, capsys,
                                                        monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    code = main(["diagnostic", "--dry-run", "--device", "cuda",
                 "--campaign-root", str(tmp_path / "campaign"),
                 "--config", str(CONFIG_PATH)])
    assert code == 2
    err = capsys.readouterr().err
    assert "V542_CAMPAIGN_DEVICE_UNAVAILABLE" in err
    assert "Traceback" not in err
    assert not (tmp_path / "campaign").exists()


def test_cli_unknown_candidate_is_a_typed_refusal_exit_nonzero(tmp_path,
                                                               capsys):
    code = main(["diagnostic", "--dry-run", "--candidates", "A9",
                 "--campaign-root", str(tmp_path / "campaign"),
                 "--config", str(CONFIG_PATH)])
    assert code == 2
    err = capsys.readouterr().err
    assert "V542_CAMPAIGN_CANDIDATE_UNKNOWN" in err
    assert "Traceback" not in err


def test_cli_unexpected_exceptions_are_not_swallowed(monkeypatch, tmp_path):
    # a non-typed exception must SURFACE (re-raised), never be converted
    # into a silent success or a swallowed refusal.
    def _boom(*_args, **_kwargs):
        raise ValueError("unexpected non-typed failure")

    monkeypatch.setattr(
        "scripts.agent_ops.v542_campaign.load_campaign_config", _boom)
    with pytest.raises(ValueError, match="unexpected non-typed failure"):
        main(["diagnostic", "--dry-run", "--config", str(CONFIG_PATH),
              "--campaign-root", str(tmp_path / "campaign")])


def test_cli_trainer_error_is_converted_to_a_typed_refusal(tmp_path, capsys,
                                                           monkeypatch):
    from pes2ts_core.pesn.offline.training.v542_train import V542TrainerError

    def _boom(*_args, **_kwargs):
        raise V542TrainerError("CUDA_UNAVAILABLE",
                               "device=cuda but CUDA is not available")

    monkeypatch.setattr(
        "scripts.agent_ops.v542_campaign.load_campaign_env", _boom)
    code = main(["diagnostic", "--campaign-root", str(tmp_path / "campaign"),
                 "--config", str(CONFIG_PATH)])
    assert code == 2
    err = capsys.readouterr().err
    assert "V542_CAMPAIGN_TRAINER_REFUSAL" in err
    assert "CUDA_UNAVAILABLE" in err
    assert "Traceback" not in err
    assert not (tmp_path / "campaign").exists()


def test_identity_digest_uses_the_single_canonical_recipe():
    """Gate-3 residual A1: the harness must not use the lenient ``str()`` recipe."""
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256
    from scripts.agent_ops.v542_campaign import _identity_sha256

    payload = {"b": 2, "a": [1, "x", {"c": True}]}
    assert _identity_sha256(payload) == canonical_sha256(payload)
    # key order must not matter
    assert _identity_sha256({"a": [1, "x", {"c": True}], "b": 2}) == \
        canonical_sha256(payload)
    # non-JSON values are refused, not silently coerced with str()
    with pytest.raises(Exception) as excinfo:
        _identity_sha256({"bad": object()})
    assert "V542_IDENTITY_SERIALIZE" in str(excinfo.value)
