"""V5.4.3 WP9 — T7 integration rollout tests (``scripts/agent_ops/
v543_rollout.py``).

CPU-fast and synthetic by default: the controller rules of §4.6 / §10.3, the
four-combination stage matrix of §10.1, the champion/baseline resolution off a
select report, the two-expert T7 assembly, the frame patching arithmetic of
§4.3 and the §9.5 dev one-shot gate are all exercised WITHOUT data, WITHOUT
training and WITHOUT touching the 589 dev curves.

The one heavy leg is marked ``realdata`` (pytest.ini's default ``addopts``
excludes it), and it is the plan's own end-to-end proof that stage B0 IS the
frozen M0/P0 baseline:

    pytest -m realdata tests/agent_ops/test_v543_rollout.py -q

What is proven here (plan §10 :1044–1082, §9.5 :1035–1041, §11.5
``test_v543_no_move_contract``):

* ``move_flag`` is the §4.6 :253–259 transcription: TS rows always move, the
  ONLY No-Move is "calibrated p_end + consistent chemical state", ``q`` never
  participates and an uncalibrated ``tau_end`` makes No-Move impossible;
* every §10.2 :1071 No-Move audit field is present and no No-Move reason can
  mention ``q`` (structural, not a comment);
* ``B0/BM/BP/BMP`` route the M* / P* champions exactly as the §10.1 table says;
* a ``champion: null`` select entry keeps the track's M0/P0 (§8.2 :816) and a
  selected champion without a checkpoint is the typed
  ``V543_ROLLOUT_CHAMPION_CHECKPOINT`` refusal;
* the T7 assembly turns BOTH experts on (§4.3 :226) while still asserting the
  §6.4 budget, which the single-active-track ``assemble_v543`` cannot express;
* the dev mode is gated: ``confirm`` always refuses, ``dev`` refuses without
  ``--dev-onetime``, a missing dev loader refuses BEFORE the one shot is
  burned, the first visit appends exactly one ledger line and a second visit is
  ``V543_ROLLOUT_DEV_ALREADY_USED``;
* stage predictions patch the frozen frames and a zero residual reproduces
  them bit-for-bit (which is why B0 can be compared to the frozen baseline);
* the report document carries the §10.1 integration-gate rules, the controller
  block, the dev bookkeeping and the B0 consistency block.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
numpy = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import v543_rollout as r  # noqa: E402

np = numpy


# ---------------------------------------------------------------------------
# §4.6 / §10.3 — the controller
# ---------------------------------------------------------------------------
def test_move_flag_is_the_plan_transcription() -> None:
    # TS rows ALWAYS move, whatever p_end/tau say
    assert r.move_flag(0.99, 0.5, True, True) is True
    assert r.move_flag(0.0, 0.5, False, True) is True
    # the only legal No-Move: p_end >= tau_end AND chemical_state_ok
    assert r.move_flag(0.9, 0.5, True, False) is False
    assert r.move_flag(0.5, 0.5, True, False) is False
    # a consistent state without the calibrated endpoint probability keeps moving
    assert r.move_flag(0.2, 0.5, True, False) is True
    # an inconsistent chemical state can never stop the walk
    assert r.move_flag(0.9, 0.5, False, False) is True
    # uncalibrated tau => No-Move is impossible (§1.5 :105)
    assert r.move_flag(0.9, None, True, False) is True
    assert r.move_flag(None, 0.5, True, False) is True


def test_controller_acceptance_cases_from_section_10_3() -> None:
    """§10.3 :1076–1079 — the four ``test_v543_no_move_contract`` cases."""
    # q=+1, p_end low -> must MOVE
    decision = r.controller_step(p_end=0.01, tau_end=0.5, chemical_state_ok=True,
                                 is_ts=False, q_observed=1.0)
    assert decision["move"] is True
    assert decision["q_used_as_trigger"] is False
    # q=-1, p_end high and consistent -> No-Move allowed
    decision = r.controller_step(p_end=0.95, tau_end=0.5,
                                 chemical_state_ok=True, is_ts=False,
                                 q_observed=-1.0)
    assert decision["move"] is False
    assert decision["reason"] in r.NO_MOVE_REASONS
    # TS row -> always MOVE, even with q=-1 and p_end high
    decision = r.controller_step(p_end=0.99, tau_end=0.5,
                                 chemical_state_ok=True, is_ts=True,
                                 q_observed=-1.0)
    assert decision["move"] is True
    assert decision["reason"] == "ts_row_must_move"
    # p_end high but the chemical state disagrees -> must MOVE
    decision = r.controller_step(p_end=0.99, tau_end=0.5,
                                 chemical_state_ok=False, is_ts=False,
                                 q_observed=-1.0)
    assert decision["move"] is True
    assert decision["reason"] == "chemical_state_not_consistent_must_move"


def test_no_move_never_mentions_q_and_records_the_audit_fields() -> None:
    decision = r.controller_step(p_end=0.8, tau_end=0.5, chemical_state_ok=True,
                                 is_ts=False, q_observed=-0.7)
    assert decision["move"] is False
    assert "q" not in decision["reason"]
    # §10.2 :1071 — every No-Move records these three fields
    assert set(decision["inputs"]) == {"p_end", "tau_end", "chemical_state_ok",
                                       "is_ts"}
    assert decision["q_observed_not_a_trigger"] == -0.7
    for reason in r.NO_MOVE_REASONS:
        assert "q" not in reason


def test_uncalibrated_tau_reports_that_no_move_is_impossible() -> None:
    decision = r.controller_step(p_end=0.999, tau_end=None,
                                 chemical_state_ok=True, is_ts=False,
                                 q_observed=-1.0)
    assert decision["move"] is True
    assert decision["reason"] == "tau_end_uncalibrated_no_move_impossible"
    assert decision["inputs"]["tau_end"] is None


def test_default_chemical_state_check_is_unavailable_not_invented() -> None:
    ok, reason = r.default_chemical_state_check(object())
    assert ok is False
    assert "unavailable" in reason


# ---------------------------------------------------------------------------
# §10.1 — the four combinations
# ---------------------------------------------------------------------------
def test_stage_plans_match_the_section_10_1_table() -> None:
    plans = r.stage_plans()
    assert tuple(plans) == r.STAGES == ("B0", "BM", "BP", "BMP")
    assert plans["B0"].champion_tracks == ()
    assert plans["BM"].champion_tracks == ("move",)
    assert plans["BP"].champion_tracks == ("process",)
    assert plans["BMP"].champion_tracks == ("move", "process")
    described = plans["BMP"].describe()
    assert described["move"] == "M*" and described["process"] == "P*"
    assert plans["BM"].describe()["process"] == "P0"
    assert plans["B0"].describe()["purpose"] == "double baseline (M0 + P0)"


def test_stage_plan_refuses_an_unknown_combination() -> None:
    with pytest.raises(r.RolloutError) as error:
        r.stage_plan("B0+")
    assert error.value.code == r.CODE_STAGE


# ---------------------------------------------------------------------------
# champion resolution (§8.2 :816 / §10.1)
# ---------------------------------------------------------------------------
def _select_document(*, move: object, process: object) -> dict:
    return {
        "schema": "pesn_v543_stats_select_report_v1",
        "tracks": {
            "move": {"track": "move", "champion": move,
                     "retained_baseline": None if move else "M0"},
            "process": {"track": "process", "champion": process,
                        "retained_baseline": None if process else "P0"},
        },
    }


def _write_select(tmp_path: Path, document: dict) -> dict:
    path = tmp_path / "track_champions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return r.load_select_report(path)


def test_select_report_must_carry_the_frozen_schema(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"schema": "something_else", "tracks": {}}),
                    encoding="utf-8")
    with pytest.raises(r.RolloutError) as error:
        r.load_select_report(path)
    assert error.value.code == r.CODE_SELECT_SCHEMA


def test_a_null_champion_retains_the_track_baseline(tmp_path: Path) -> None:
    select = _write_select(tmp_path, _select_document(move=None, process=None))
    move = r.resolve_champion(select, "move", seed=17,
                              campaign_root=tmp_path / "campaign")
    process = r.resolve_champion(select, "process", seed=17,
                                 campaign_root=tmp_path / "campaign")
    assert move.decoder_id == "M0" and process.decoder_id == "P0"
    assert move.source == process.source == "retained_baseline"
    assert move.checkpoint_path is None
    assert move.select_report_sha256 == select["sha256"]


def test_a_selected_champion_without_a_checkpoint_is_refused(tmp_path: Path) -> None:
    select = _write_select(tmp_path, _select_document(move="M1", process=None))
    with pytest.raises(r.RolloutError) as error:
        r.resolve_champion(select, "move", seed=17,
                           campaign_root=tmp_path / "campaign")
    assert error.value.code == r.CODE_CHAMPION_CHECKPOINT


def test_a_selected_champion_resolves_its_run_checkpoint(tmp_path: Path) -> None:
    select = _write_select(tmp_path, _select_document(move="M1", process="P2"))
    campaign_root = tmp_path / "campaign"
    for decoder_id in ("M1", "P2"):
        run_dir = campaign_root / "runs" / (
            r.RUN_ID_TEMPLATE.format(decoder_id=decoder_id, seed=17))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / r.CHAMPION_CHECKPOINT_NAME).write_bytes(b"weights")
    move = r.resolve_champion(select, "move", seed=17,
                              campaign_root=campaign_root)
    process = r.resolve_champion(select, "process", seed=17,
                                 campaign_root=campaign_root)
    assert move.decoder_id == "M1" and move.source == "champion"
    assert process.decoder_id == "P2"
    assert len(str(move.checkpoint_sha256)) == 64
    assert move.checkpoint_sha256 == process.checkpoint_sha256


# ---------------------------------------------------------------------------
# the T7 assembly (§4.3 :226 / §6.4)
# ---------------------------------------------------------------------------
class _FakeMoveDecoder(torch.nn.Module):
    """A protocol-clean synthetic MOVE decoder (no data/loss surface)."""

    decoder_id = "M1"
    track = "move"
    model_version = "fake_m1_v1"

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 1, bias=False)

    def forward(self, ctx):                       # noqa: D401 - protocol
        return torch.zeros_like(ctx.u0)


class _FakeProcessDecoder(torch.nn.Module):
    decoder_id = "P1"
    track = "process"
    model_version = "fake_p1_v1"

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(1, 1, bias=False)

    def forward(self, ctx):                       # noqa: D401 - protocol
        from pes2ts_core.pesn.models.v543 import ProcessResidual

        return ProcessResidual(delta_q_raw=torch.zeros_like(ctx.q0_raw),
                               delta_e=torch.zeros_like(ctx.endpoint0_logit))


def _synthetic_m0p0():
    from pes2ts_core.pesn.models.v543 import CoarseEncoder, LatentBridge
    from pes2ts_core.pesn.models.v543.registry import build_baseline

    encoder = CoarseEncoder.load(
        REPO_ROOT / "outputs/pesn/v542_backbone_50k/runs/V542_A0_S17/"
                   "checkpoint_best.pt", 17)
    encoder.eval()
    bridge = LatentBridge()
    return encoder, bridge, build_baseline("move"), build_baseline("process")


def test_t7_assembly_enables_both_experts_and_keeps_the_budget_checks() -> None:
    encoder, bridge, _m0, _p0 = _synthetic_m0p0()
    registry = r.candidate_registry()
    move = registry.build("M1", track="move")
    process = registry.build("P1", track="process")
    assembly = r.build_rollout_assembly(
        encoder, bridge=bridge, move=move, move_id="M1",
        process=process, process_id="P1")
    assert assembly.move.decoder_id == "M1"
    assert assembly.process.decoder_id == "P1"
    # both slots are non-baseline: exactly what assemble_v543 refuses
    assert assembly.move.is_baseline is False
    assert assembly.process.is_baseline is False
    assert all(not p.requires_grad for p in assembly.encoder.parameters())
    report = r._assert_rollout_budget(assembly)
    assert report["total_model_params"] <= r.PARAM_GLOBAL_CAP_V543
    assert report["bridge_trainable"] == 0
    assert 34_200 <= report["move_params"] <= 37_800
    assert 34_200 <= report["process_params"] <= 37_800


def test_t7_assembly_still_enforces_the_section_6_4_param_band() -> None:
    encoder, bridge, _m0, _p0 = _synthetic_m0p0()
    with pytest.raises(r.RolloutError) as error:
        r.build_rollout_assembly(encoder, bridge=bridge, move=_FakeMoveDecoder(),
                                 move_id="M1", process=_FakeProcessDecoder(),
                                 process_id="P1")
    assert error.value.code == "V543_REGISTRY_PARAMS_BAND"


def test_t7_assembly_refuses_a_non_encoder() -> None:
    _encoder, bridge, m0, p0 = _synthetic_m0p0()
    with pytest.raises(r.RolloutError) as error:
        r.build_rollout_assembly(object(), bridge=bridge, move=m0, move_id="M0",
                                 process=p0, process_id="P0")
    assert error.value.code == r.CODE_ASSEMBLY


# ---------------------------------------------------------------------------
# §9.5 — the dev one-shot gate
# ---------------------------------------------------------------------------
def test_confirm_mode_is_sealed() -> None:
    with pytest.raises(r.RolloutError) as error:
        r.assert_mode_allowed(r.MODE_CONFIRM, dev_onetime=True)
    assert error.value.code == r.CODE_CONFIRM_SEALED
    assert "confirm_read=false" in str(error.value)


def test_dev_mode_requires_the_explicit_flag() -> None:
    with pytest.raises(r.RolloutError) as error:
        r.assert_mode_allowed(r.MODE_DEV, dev_onetime=False)
    assert error.value.code == r.CODE_DEV_FLAG_REQUIRED
    assert r.assert_mode_allowed(r.MODE_DEV, dev_onetime=True) == r.MODE_DEV
    assert r.assert_mode_allowed(r.MODE_INNER_VAL, dev_onetime=False) == \
        r.MODE_INNER_VAL


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(r.RolloutError) as error:
        r.assert_mode_allowed("dev2", dev_onetime=True)
    assert error.value.code == r.CODE_MODE


def test_dev_ledger_is_one_shot(tmp_path: Path) -> None:
    ledger = r.DevAccessLedger(tmp_path / "dev_ledger.jsonl")
    assert ledger.already_used() is False
    entry = ledger.claim(actor="test", purpose="t7", seed=17, mode="dev",
                         budget={"eval_curves_requested": 1})
    assert entry["split"] == "dev" and entry["one_shot"] is True
    assert entry["confirm_read"] is False and entry["new_QC_calls"] == 0
    assert len(str(entry["entry_sha256"])) == 64
    assert ledger.already_used() is True
    with pytest.raises(r.RolloutError) as error:
        ledger.claim(actor="test", purpose="t7", seed=17, mode="dev",
                     budget={})
    assert error.value.code == r.CODE_DEV_ALREADY_USED
    # append-only: the first claim is still the only dev line
    assert len(ledger.dev_uses()) == 1


def test_missing_dev_loader_refuses_before_burning_the_one_shot(
        tmp_path: Path) -> None:
    ledger_path = tmp_path / "dev_ledger.jsonl"
    with pytest.raises(r.RolloutError) as error:
        r.require_dev_frame_loader(None)
    assert error.value.code == r.CODE_DEV_LOADER_UNAVAILABLE
    assert not ledger_path.exists()
    assert r.DevAccessLedger(ledger_path).already_used() is False


# ---------------------------------------------------------------------------
# frame patching / the walk / the report
# ---------------------------------------------------------------------------
def _frame(*, curve="c1", direction="forward", is_ts=False, is_start=False,
           endpoint_logit=0.0, q0_raw=0.3):
    from pes2ts_core.pesn.offline.evaluation.v543_eval import EvalFrame

    u = np.zeros((2, 3), dtype=np.float64)
    return EvalFrame(
        curve_id=curve, u=u, y=u.copy(), pair_slots=np.zeros((0, 2),
                                                             dtype=np.int64),
        pair_offsets=np.zeros((0, 3), dtype=np.float64),
        direction=direction, stratum="pre_TS", intent_class="both",
        q_pred=math.tanh(q0_raw), q_true=0.5, q_mask=True,
        endpoint_logit=endpoint_logit, endpoint_label=1, endpoint_mask=True,
        is_start_row=is_start, is_ts_row=is_ts, move_eligible=True, u0=u.copy(),
        q0_raw=q0_raw, endpoint0_logit=endpoint_logit, key_v=None)


def test_zero_prediction_reproduces_the_frozen_frame_bit_for_bit() -> None:
    frames = [_frame(curve="c1"), _frame(curve="c1", is_ts=True)]
    predictions = [r.FramePrediction(u_rows=np.asarray(frame.u),
                                     delta_q_raw=0.0, delta_e=0.0, n_keys=2)
                   for frame in frames]
    stage = r.apply_predictions(
        frames, predictions, stage="B0",
        meta={"n_batches": 1, "encoder_forward_count": 1, "wall_seconds": 0.1,
              "max_abs_delta_q_raw": 0.0, "max_abs_delta_e": 0.0,
              "max_abs_delta_u": 0.0})
    assert stage.identity_vs_zero_residual is True
    for before, after in zip(frames, stage.frames):
        assert np.array_equal(np.asarray(before.u), np.asarray(after.u))
        assert float(before.q_pred) == float(after.q_pred)
        assert float(before.endpoint_logit) == float(after.endpoint_logit)


def test_predictions_must_match_the_frame_set() -> None:
    frames = [_frame()]
    with pytest.raises(r.RolloutError) as error:
        r.apply_predictions(frames, [], stage="B0",
                            meta={"n_batches": 0, "encoder_forward_count": 0,
                                  "wall_seconds": 0.0,
                                  "max_abs_delta_q_raw": 0.0,
                                  "max_abs_delta_e": 0.0,
                                  "max_abs_delta_u": 0.0})
    assert error.value.code == r.CODE_FRAMES


def test_walk_keeps_moving_when_tau_is_uncalibrated() -> None:
    frames = [_frame(curve="c1", is_start=True),
              _frame(curve="c1"), _frame(curve="c1", is_ts=True)]
    meta = {"n_batches": 1, "encoder_forward_count": 1, "wall_seconds": 0.1,
            "max_abs_delta_q_raw": 0.0, "max_abs_delta_e": 0.0,
            "max_abs_delta_u": 0.0}
    predictions = [r.FramePrediction(u_rows=np.asarray(frame.u),
                                     delta_q_raw=0.0, delta_e=0.0, n_keys=2)
                   for frame in frames]
    stage = r.apply_predictions(frames, predictions, stage="B0", meta=meta)
    walk = r.run_walk(stage, max_steps=8, tau_end=None)
    assert walk.n_groups == 1
    assert walk.n_no_move == 0
    assert walk.n_ts_row_reached == 1
    assert walk.n_passed_ts == 0
    assert walk.n_terminal_frame_reached == 1
    assert walk.success_rate == 1.0
    assert walk.controller_reasons == {
        "tau_end_uncalibrated_no_move_impossible": 2,
        # a TS row keeps moving and says so (§4.3 :227)
        "ts_row_must_move": 1}
    described = walk.describe()
    assert described["collision"]["status"] == "not_measured"
    assert described["endpoint_arrival"]["status"].startswith(
        "not_applicable_tau_end_uncalibrated")


def test_walk_stops_on_the_legal_no_move_and_counts_the_misstop() -> None:
    frames = [_frame(curve="c1", is_start=True, endpoint_logit=99.0)]
    meta = {"n_batches": 1, "encoder_forward_count": 1, "wall_seconds": 0.1,
            "max_abs_delta_q_raw": 0.0, "max_abs_delta_e": 0.0,
            "max_abs_delta_u": 0.0}
    predictions = [r.FramePrediction(u_rows=np.asarray(frames[0].u),
                                     delta_q_raw=0.0, delta_e=0.0, n_keys=2)]
    stage = r.apply_predictions(frames, predictions, stage="B0", meta=meta)
    walk = r.run_walk(stage, max_steps=4, tau_end=0.5,
                      chemical_state_check=lambda _f: (True, "injected"))
    assert walk.n_no_move == 1
    # a No-Move on a start row is a misstop (§10.1 :1059)
    assert walk.n_misstop_start == 1


def test_report_document_carries_the_gate_controller_and_dev_blocks() -> None:
    from pes2ts_core.pesn.offline.evaluation.v543_eval import evaluate_v543

    frames = [_frame(curve="c1", is_start=True, endpoint_logit=0.2),
              _frame(curve="c1", is_ts=True, endpoint_logit=0.1)]
    meta = {"n_batches": 1, "encoder_forward_count": 1, "wall_seconds": 0.1,
            "max_abs_delta_q_raw": 0.0, "max_abs_delta_e": 0.0,
            "max_abs_delta_u": 0.0}
    predictions = [r.FramePrediction(u_rows=np.asarray(frame.u),
                                     delta_q_raw=0.0, delta_e=0.0, n_keys=2)
                   for frame in frames]
    result = r.SeedStageResults(seed=17, base_frames=frames,
                               positions_rule={"rule": "synthetic"},
                               positions_sha256="0" * 64, train_q_mean=0.0)
    for stage in r.STAGES:
        stage_frames = r.apply_predictions(frames, predictions, stage=stage,
                                          meta=meta)
        result.stages[stage] = stage_frames
        result.reports[stage] = evaluate_v543(stage_frames.frames,
                                             train_q_mean=0.0)
        result.walks[stage] = r.run_walk(stage_frames, max_steps=4,
                                        tau_end=None)
        result.decoders[stage] = {"stage": stage}
    resolution = r.ChampionResolution(
        track="move", decoder_id="M0", source="retained_baseline",
        checkpoint_path=None, checkpoint_sha256=None,
        select_report="select.json", select_report_sha256="0" * 64, seed=17)
    document = r.build_report(
        mode=r.MODE_INNER_VAL, seeds=[17], campaign={"campaign_id": "SYNTH"},
        checkpoints={17: {"path": "synthetic", "sha256": "0" * 64}},
        results=[result],
        move_resolution_by_seed={17: resolution},
        process_resolution_by_seed={17: r.ChampionResolution(
            track="process", decoder_id="P0", source="retained_baseline",
            checkpoint_path=None, checkpoint_sha256=None,
            select_report="select.json", select_report_sha256="0" * 64,
            seed=17)},
        budget={"eval_curves_requested": 1, "frames_per_curve_requested": 2,
                "frames_per_batch": 32, "tau_end": None,
                "max_steps": 4},
        data_block={"mode": r.MODE_INNER_VAL}, baseline_document=None,
        baseline_sha256=None)
    assert document["schema"] == r.ROLLOUT_SCHEMA
    assert document["stages"] == list(r.STAGES)
    assert set(document["integration_gate"]["rules"]) == set(
        r.INTEGRATION_GATE_RULES)
    assert document["integration_gate"]["checks"][
        "collision_not_increased"]["status"] == "not_measurable"
    assert document["controller"]["q_used_as_trigger"] is False
    assert document["controller"]["parameter_count"] == 0
    assert document["dev_access"]["confirm_read"] is False
    assert document["dev_access"]["new_QC_calls"] == 0
    per_seed = document["per_seed"]["17"]
    for stage in r.STAGES:
        block = per_seed["stages"][stage]
        assert block["eval_report"]["schema"] == "pesn_v543_eval_report_v1"
        assert "rollout" in block and "frames" in block
    # with no baseline document, B0 has no consistency block to lie about
    assert per_seed["stages"]["B0"]["b0_vs_frozen_baseline"] is None


def test_b0_consistency_check_refuses_a_different_report() -> None:
    baseline = {"schema": "pesn_v543_m0p0_baseline_v1",
                "per_seed": {"17": {"per_curve": [{"curve_id": "c1"}],
                                    "pooled": {"move": {}},
                                    "validity": {}}}}
    with pytest.raises(r.RolloutError) as error:
        r.assert_b0_matches_frozen_baseline(
            stage_report={"per_curve": [{"curve_id": "other"}],
                          "pooled": {"move": {}}, "validity": {}},
            baseline=baseline, seed=17, budget_matches=True)
    assert error.value.code == r.CODE_B0_BASELINE_MISMATCH
    # a different budget is reported as not_checked, never as a pass
    result = r.assert_b0_matches_frozen_baseline(
        stage_report={"per_curve": [], "pooled": {}, "validity": {}},
        baseline=baseline, seed=17, budget_matches=False)
    assert result["checked"] is False and result["identical"] is None


# ---------------------------------------------------------------------------
# the realdata end-to-end leg (§10.2 / §9.5 :1037 — B0 IS the baseline)
# ---------------------------------------------------------------------------
@pytest.mark.realdata
def test_inner_val_rollout_b0_reproduces_the_frozen_baseline(
        tmp_path: Path) -> None:
    import argparse

    args = argparse.Namespace(
        out=str(tmp_path / "t7_rollout.json"),
        config=r.DEFAULT_CONFIG, mode=r.MODE_INNER_VAL, dev_onetime=False,
        dev_ledger=str(tmp_path / "dev_ledger.jsonl"),
        select_report=str(REPO_ROOT / r.CAMPAIGN_ROOT / r.DEFAULT_SELECT_REL),
        baseline=None, seeds=[17], eval_curves=1024, frames_per_curve=8,
        max_steps=r.DEFAULT_MAX_STEPS, no_basis=False, no_strata=False,
        move_champion=None, process_champion=None,
        move_champion_checkpoint=None, process_champion_checkpoint=None,
        actor="pytest")
    document = r.run(args)
    block = document["per_seed"]["17"]["stages"]["B0"]
    assert block["b0_vs_frozen_baseline"]["checked"] is True
    assert block["b0_vs_frozen_baseline"]["identical"] is True
    assert block["frames"]["identity_vs_zero_residual"] is True
    # the placeholder champions (no qualified candidate) keep M0/P0, so every
    # stage is the zero residual and the walk never stops on its own
    for stage in r.STAGES:
        assert document["per_seed"]["17"]["stages"][stage][
            "frames"]["identity_vs_zero_residual"] is True
        assert document["per_seed"]["17"]["stages"][stage]["rollout"][
            "n_no_move"] == 0
