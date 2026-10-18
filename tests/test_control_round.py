"""End-to-end control-round closed loop on synthetic fixtures (offline only).

Exercises the full ADR-0017 package-C pipeline with the offline fake ACP
transport: collect (rule + bounded exploration) -> train (conservative Q +
restricted imitation) -> freeze (pesn_model_bundle_v2) -> execute (frozen
Q-greedy re-run).  Every phase writes typed artifacts; no real QC runs, so
the status claim of this closure is fixture_passed, never smoke_passed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.generation.campaign import load_campaign_manifest  # noqa: E402
from pes2ts_core.generation.control_round import (  # noqa: E402
    ControlRoundError,
    build_selection_manifest,
    load_control_round_config,
    run_control_collection,
    run_control_execution,
    run_control_freeze,
    run_control_training,
)
from pes2ts_core.pesn.offline.transitions import (  # noqa: E402
    TRANSITIONS_FILE_NAME,
    iter_transition_docs,
    load_qualified_transitions,
)
from tests.test_control_step import make_transport  # noqa: E402

REACTION_ID = "RXN_CONTROL_ROUND_TEST"
ELEMENTS = ["C", "H", "H"]
R_COORDS = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]
P_COORDS = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [4.0, 0.0, 0.0]]


def _plan_document() -> dict:
    return {
        "schema_version": "pes2ts_connectivity_continuation_plan_v1",
        "reaction_id": REACTION_ID,
        "elements": list(ELEMENTS),
        "atom_map_order": [1, 2, 3],
        "charge": 0,
        "multiplicity": 1,
        "active_edits": [{"edit_kind": "broken", "maps": [1, 2]}],
        "drivers": [
            {
                "id": "B_1_2", "kind": "distance", "atoms": [0, 1], "maps": [1, 2],
                "role": "driver", "unit": "angstrom",
                "lambda_values": [0.0, 1.0], "values": [1.5, 2.5],
            }
        ],
        "execution_direction": "R_to_P", "origin_endpoint": "R",
        "schedule_source": "round_fixture",
        "start_geometry": [row[:] for row in R_COORDS],
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _build_round_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    legacy_root = tmp_path / "legacy"
    _write(legacy_root / REACTION_ID / "plan.json", _plan_document())

    snapshot_path = tmp_path / "snapshots" / f"{REACTION_ID}.json"
    _write(snapshot_path, {
        "schema_version": "p0_demo24_record_snapshot_v1",
        "reaction_id": REACTION_ID,
        "r_coordinates": R_COORDS,
        "p_coordinates": P_COORDS,
    })

    manifest_body = {
        "schema_version": "pes2ts_campaign_manifest_v1",
        "campaign_label": "control_round_fixture",
        "cases": [{
            "reaction_id": REACTION_ID,
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": "a" * 64,
        }],
        "frozen_parameters": {
            "continuation_policy": {}, "local_policy": {}, "branch_policy": {},
            "candidate_top_k": 3,
        },
        "budget": {
            "max_frames_per_path": 80, "max_attempts_per_path": 160,
            "max_gradient_calls_per_path": 2000, "max_total_gradient_calls": 60000,
            "max_wall_seconds_per_path": 600.0, "max_session_wall_seconds": 86400.0,
        },
        "acp_wiring": {"root": "unused", "python": None, "config_path": None},
        "review_status": {REACTION_ID: "needs_review"},
        "pilot_cases": [], "reference_failures": {},
    }
    manifest_path = tmp_path / "manifests" / "campaign.json"
    _write(manifest_path, manifest_body)
    load_campaign_manifest(manifest_path)  # validates the fixture

    selection_path = tmp_path / "manifests" / "selection.json"
    build_selection_manifest(
        reaction_ids=[REACTION_ID],
        campaign_manifest_path=manifest_path,
        output_path=selection_path,
        note="synthetic fixture",
    )

    config_path = tmp_path / "control_round.yaml"
    config_path.write_text(
        "\n".join([
            "schema_version: pesn_control_round_config_v1",
            "seed: 42",
            "paths:",
            f"  rounds_root: {tmp_path / 'rounds'}",
            f"  artifacts_root: {tmp_path / 'artifacts'}",
            "environment:",
            "  n_grid_points: 5",
            "  max_subdivisions: 2",
            "  max_rejections_per_target: 8",
            "  max_host_requests: 50",
            "  max_wall_seconds: 60.0",
            "residual:",
            "  epsilon_angstrom: 0.1",
            "candidates:",
            "  model_scales: [0.5, 1.0]",
            "  n_single_coord_perturbs: 2",
            "  max_candidates: 8",
            "  exploration_probability: 0.5",
            "  exploration_seed: 42",
            "reward:",
            "  candidate_bonus: 1.0",
            "  cost_reference_seconds: 100.0",
            "  hard_failure_penalty: 0.25",
            "control_train:",
            "  max_epochs: 3",
            "  patience: 3",
            "  batch_transitions: 8",
            "  cql_alpha: 1.0",
            "  imitation_weight: 0.5",
            "  freeze_encoder: true",
            "model:",
            "  hidden: 32",
            "  n_layers: 2",
            "  n_basis: 16",
            "  use_coordinate_attention: false",
            "bundle:",
            "  model_id: pesn-control-fixture-v0",
        ]),
        encoding="utf-8",
    )
    return manifest_path, legacy_root, selection_path, config_path, snapshot_path


def test_full_control_round_closed_loop(tmp_path):
    (manifest_path, legacy_root, selection_path, config_path, _snapshot) = _build_round_inputs(tmp_path)
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, calls = make_transport()
    assert calls == []

    collect = run_control_collection(
        round_dir=round_dir,
        manifest_path=manifest_path,
        legacy_root=legacy_root,
        selection_path=selection_path,
        config=config,
        transport=transport,
    )
    assert collect["n_cases"] == 1
    assert collect["environment_id"] == "E_native"
    assert (round_dir / "environment_freeze.json").is_file()
    assert (round_dir / "cases" / REACTION_ID / "control" / "control_scan_result.json").is_file()
    transitions = list(
        iter_transition_docs(round_dir / "transitions" / TRANSITIONS_FILE_NAME)
    )
    assert len(transitions) >= 4
    for doc in transitions:
        assert doc["environment_hash"] == collect["environment_hash"]

    train = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
    )
    assert train["transition_store"]["n_transitions"] == len(transitions)
    assert train["encoder_frozen"] is True
    assert train["n_parameters"] > 0
    run_dirs = list((round_dir / "runs").iterdir())
    assert any((run_dir / "checkpoint_best.pt").is_file() for run_dir in run_dirs)

    freeze = run_control_freeze(
        round_dir=round_dir, config=config, selection_path=selection_path,
    )
    bundle_dir = Path(freeze["bundle_path"])
    manifest = json.loads((bundle_dir / "model_bundle.json").read_text())
    assert manifest["schema_version"] == "pesn_model_bundle_v2"
    assert manifest["capabilities"]["q_proxy"] == "research_only"
    assert manifest["capabilities"]["production_control"] == "disabled"

    execute = run_control_execution(
        round_dir=round_dir,
        manifest_path=manifest_path,
        legacy_root=legacy_root,
        selection_path=selection_path,
        config=config,
        bundle_dir=bundle_dir,
        transport=transport,
    )
    assert execute["n_cases"] == 1
    execute_summary_path = round_dir / "execute" / "residual_q" / "execute_summary.json"
    assert execute_summary_path.is_file()
    assert len(calls) > len(transitions)  # both phases really executed


def test_training_refuses_foreign_environment_transitions(tmp_path):
    (manifest_path, legacy_root, selection_path, config_path, _snapshot) = _build_round_inputs(tmp_path)
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    freeze_doc = json.loads(
        (round_dir / "environment_freeze.json").read_text(encoding="utf-8")
    )
    qualified = load_qualified_transitions(
        round_dir, environment_hash="0" * 64, allowed_reactions=[REACTION_ID]
    )
    assert qualified == []
    # Tamper the round's frozen environment identity: the store must refuse
    # to qualify the transitions under the new hash and training must stop.
    freeze_doc["environment_hash"] = "0" * 64
    _write(round_dir / "environment_freeze.json", freeze_doc)
    with pytest.raises(ControlRoundError, match="no qualified transitions"):
        run_control_training(
            round_dir=round_dir, config=config, selection_path=selection_path
        )


def test_execute_refuses_bundle_without_q_support(tmp_path, monkeypatch):
    (manifest_path, legacy_root, selection_path, config_path, _snapshot) = _build_round_inputs(tmp_path)
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
    )
    freeze = run_control_freeze(
        round_dir=round_dir, config=config, selection_path=selection_path,
    )
    bundle_dir = Path(freeze["bundle_path"])
    manifest = json.loads((bundle_dir / "model_bundle.json").read_text())
    manifest["capabilities"]["q_proxy"] = "disabled"
    _write(bundle_dir / "model_bundle.json", manifest)
    checksums = json.loads((bundle_dir / "checksums.json").read_text())
    checksums["model_bundle.json"] = __import__("hashlib").sha256(
        (bundle_dir / "model_bundle.json").read_bytes()
    ).hexdigest()
    _write(bundle_dir / "checksums.json", checksums)
    with pytest.raises(ControlRoundError, match="q_proxy"):
        run_control_execution(
            round_dir=round_dir, manifest_path=manifest_path,
            legacy_root=legacy_root, selection_path=selection_path,
            config=config, bundle_dir=bundle_dir, transport=transport,
        )


# ---------------------------------------------------------------------------
# Demo24 plan §5 counterexample coverage
# ---------------------------------------------------------------------------

SECOND_REACTION = "RXN_CONTROL_ROUND_TEST2"


def _build_two_case_inputs(tmp_path: Path, *, drop_second_plan: bool):
    """Two-case fixture; optionally the second case has no plan at all."""
    (manifest_path, legacy_root, selection_path, config_path, _snap) = _build_round_inputs(tmp_path)
    legacy_root2 = tmp_path / "legacy2"
    _write(legacy_root2 / REACTION_ID / "plan.json", _plan_document())
    plan2 = _plan_document()
    plan2["reaction_id"] = SECOND_REACTION
    plan2["start_geometry"] = [row[:] for row in R_COORDS]
    if not drop_second_plan:
        _write(legacy_root2 / SECOND_REACTION / "plan.json", plan2)

    snapshot2 = tmp_path / "snapshots" / f"{SECOND_REACTION}.json"
    _write(snapshot2, {
        "schema_version": "p0_demo24_record_snapshot_v1",
        "reaction_id": SECOND_REACTION,
        "r_coordinates": R_COORDS,
        "p_coordinates": P_COORDS,
    })
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"].append({
        "reaction_id": SECOND_REACTION,
        "snapshot_path": str(snapshot2),
        "snapshot_sha256": "b" * 64,
    })
    manifest["review_status"][SECOND_REACTION] = "needs_review"
    from pes2ts_core.generation.planning.synchronized_path import digest

    manifest["content_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "content_sha256"}
    )
    manifest_path2 = tmp_path / "manifests" / "campaign2.json"
    _write(manifest_path2, manifest)

    selection2 = tmp_path / "manifests" / "selection2.json"
    build_selection_manifest(
        reaction_ids=[REACTION_ID, SECOND_REACTION],
        campaign_manifest_path=manifest_path2,
        output_path=selection2,
        note="two-case fixture",
    )
    return manifest_path2, legacy_root2, selection2, config_path


def test_single_bad_case_does_not_abort_batch(tmp_path):
    (manifest_path, legacy_root, selection_path, config_path) = _build_two_case_inputs(
        tmp_path, drop_second_plan=True
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, _ = make_transport()
    summary = run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    assert summary["n_cases"] == 2
    by_id = {row["reaction_id"]: row for row in summary["cases"]}
    assert by_id[REACTION_ID]["status"] in ("complete_interval", "partial_prefix")
    assert by_id[SECOND_REACTION]["status"] == "rejected_input"
    assert (by_id[SECOND_REACTION]["failure_code"] or "").startswith("CASE_FAILURE")
    assert (round_dir / "cases" / REACTION_ID / "control" / "control_scan_result.json").is_file()


def test_resume_does_not_rebill_or_duplicate(tmp_path):
    (manifest_path, legacy_root, selection_path, config_path) = _build_two_case_inputs(
        tmp_path, drop_second_plan=False
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, calls = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    n_calls_first = len(calls)
    transitions_path = round_dir / "transitions" / TRANSITIONS_FILE_NAME
    n_transitions_first = len(transitions_path.read_text().strip().splitlines())

    summary = run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    assert summary["n_executed"] == 0
    assert summary["n_resumed"] == 2
    assert len(calls) == n_calls_first
    assert len(transitions_path.read_text().strip().splitlines()) == n_transitions_first
    ledger = (round_dir / "cost_ledger.jsonl").read_text().strip().splitlines()
    resumed_rows = [json.loads(line) for line in ledger if json.loads(line)["resumed"]]
    assert len(resumed_rows) == 2
    assert all(row["wall_seconds"] == 0.0 for row in resumed_rows)


def test_bc_only_training_never_touches_q_head(tmp_path):
    (manifest_path, legacy_root, selection_path, config_path) = _build_two_case_inputs(
        tmp_path, drop_second_plan=False
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )

    import torch as torch_mod

    from pes2ts_core.pesn.offline.control_trainer import ControlTrainer, ControlTrainSettings

    freeze_doc = json.loads(
        (round_dir / "environment_freeze.json").read_text(encoding="utf-8")
    )
    load = __import__(
        "pes2ts_core.pesn.offline.transitions", fromlist=["load_transition_set"]
    ).load_transition_set(
        round_dir,
        environment_hash=freeze_doc["environment_hash"],
        allowed_reactions=json.loads(selection_path.read_text())["reaction_ids"],
    )
    assert load.qualified

    def run(mode):
        settings = ControlTrainSettings.from_config({
            **config, "control_train": {**config["control_train"], "training_mode": mode},
        })
        trainer = ControlTrainer(
            settings=settings, run_record=None,
            config_sha256="0" * 64, transitions_digest="0" * 64,
        )
        q_before = {
            k: v.clone() for k, v in trainer.model.heads.module.state_dict().items()
            if k.startswith("q_driver") or k.startswith("q_final")
        }
        trainer.train(load.qualified)
        q_after = trainer.model.heads.module.state_dict()
        unchanged = all(torch_mod.equal(q_before[k], q_after[k]) for k in q_before)
        return trainer, unchanged

    _, bc_unchanged = run("bc_only")
    cql_trainer, cql_unchanged = run("cql")
    assert bc_unchanged is True
    assert cql_unchanged is False


def test_td_target_respects_recorded_bootstrap_mask(tmp_path):
    from pes2ts_core.pesn.offline.control_trainer import (
        ControlBatch,
        ControlTrainer,
        ControlTrainSettings,
        _successor_context_index,
    )

    (manifest_path, legacy_root, selection_path, config_path) = _build_two_case_inputs(
        tmp_path, drop_second_plan=False
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    freeze_doc = json.loads(
        (round_dir / "environment_freeze.json").read_text(encoding="utf-8")
    )
    from pes2ts_core.pesn.offline.transitions import load_transition_set

    transitions = load_transition_set(
        round_dir,
        environment_hash=freeze_doc["environment_hash"],
        allowed_reactions=json.loads(selection_path.read_text())["reaction_ids"],
    ).qualified
    transitions = sorted(transitions, key=lambda t: (t.reaction_id, t.doc["decision_index"]))
    assert len(transitions) >= 3
    successors = _successor_context_index(transitions)

    # Force the final transition into a budget-truncated state: the recorded
    # mask (False) must win — the TD target is exactly r, no bootstrapping
    # from the fabricated zero residual, despite the "truncated" status name.
    final = transitions[-1]
    final.doc["episode_status"] = "truncated_budget"
    final.doc["bootstrap_mask"] = False
    ongoing = transitions[-2]
    assert ongoing.bootstrap_mask is True

    settings = ControlTrainSettings.from_config(dict(config))
    trainer = ControlTrainer(
        settings=settings, run_record=None,
        config_sha256="0" * 64, transitions_digest="0" * 64,
    )
    torch = trainer.torch

    captured_targets: list[float] = []

    real_smooth_l1 = torch.nn.functional.smooth_l1_loss

    def spy_smooth_l1(pred, target, *args, **kwargs):
        captured_targets.extend(target.tolist())
        return real_smooth_l1(pred, target, *args, **kwargs)

    batch = ControlBatch(
        transitions=[ongoing, final],
        next_contexts=[
            successors.get(
                (
                    ongoing.episode_successor_key_base,
                    int(ongoing.doc["decision_index"]) + 1,
                )
            ),
            successors.get(
                (
                    final.episode_successor_key_base,
                    int(final.doc["decision_index"]) + 1,
                )
            ),
        ],
    )
    assert batch.next_contexts[0] is not None
    monkeypatch_obj = torch.nn.functional
    original = monkeypatch_obj.smooth_l1_loss
    monkeypatch_obj.smooth_l1_loss = spy_smooth_l1
    try:
        losses = trainer.compute_losses(batch)
    finally:
        monkeypatch_obj.smooth_l1_loss = original
    assert float(losses["total"].item()) == float(losses["total"].item())
    rewards = [ongoing.reward, final.reward]
    # masked-off final: y == r exactly
    assert captured_targets[1] == pytest.approx(rewards[1], abs=1e-6)
    # bootstrapped ongoing: y == r + max_a' Q_target(S', a') from the target net
    assert captured_targets[0] != pytest.approx(rewards[0], abs=1e-6)


def test_transition_store_rejections_and_dedup(tmp_path):
    from pes2ts_core.pesn.offline.transitions import (
        load_transition_set,
        iter_transition_docs,
    )

    (manifest_path, legacy_root, selection_path, config_path) = _build_two_case_inputs(
        tmp_path, drop_second_plan=False
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    freeze_doc = json.loads(
        (round_dir / "environment_freeze.json").read_text(encoding="utf-8")
    )
    transitions_path = round_dir / "transitions" / TRANSITIONS_FILE_NAME
    docs = list(iter_transition_docs(transitions_path))

    # valid-mixing attempt: relabel one transition to a foreign reaction
    foreign = dict(docs[0])
    foreign["reaction_id"] = "RXN_NOT_IN_TRAIN_SELECTION"
    foreign["transition_id"] = foreign["transition_id"] + "-foreign"
    with transitions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(foreign, sort_keys=True) + "\n")
        handle.write(json.dumps(docs[1], sort_keys=True) + "\n")  # duplicate id

    load = load_transition_set(
        round_dir,
        environment_hash=freeze_doc["environment_hash"],
        allowed_reactions=json.loads(selection_path.read_text())["reaction_ids"],
    )
    assert load.rejected.get("reaction_not_allowed") == 1
    assert load.rejected.get("duplicate_transition_id") == 1
    ids = [t.doc["transition_id"] for t in load.qualified]
    assert len(ids) == len(set(ids))


def test_environment_drift_refused_between_phases(tmp_path):
    (manifest_path, legacy_root, selection_path, config_path) = _build_two_case_inputs(
        tmp_path, drop_second_plan=False
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    drifted = json.loads(json.dumps(config))
    drifted["environment"]["max_wall_seconds"] = 123.0
    with pytest.raises(ControlRoundError, match="environment drift"):
        run_control_collection(
            round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
            selection_path=selection_path, config=drifted, transport=transport,
        )


def test_report_keeps_full_denominator(tmp_path):
    from pes2ts_core.generation.control_round import run_control_report

    (manifest_path, legacy_root, selection_path, config_path) = _build_two_case_inputs(
        tmp_path, drop_second_plan=False
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config, transport=transport,
    )
    run_control_training(round_dir=round_dir, config=config, selection_path=selection_path)
    freeze = run_control_freeze(round_dir=round_dir, config=config, selection_path=selection_path)
    run_control_execution(
        round_dir=round_dir, manifest_path=manifest_path, legacy_root=legacy_root,
        selection_path=selection_path, config=config,
        bundle_dir=freeze["bundle_path"], transport=transport,
        max_cases=1,
    )
    report = run_control_report(round_dir=round_dir, execute_selection_path=selection_path)
    assert report["n_denominator"] == 2
    rows = report["cases"]
    assert len(rows) == 6
    for group in ("rule_zero", "bc_residual"):
        entries = [r for r in rows if r["policy_group"] == group]
        assert all(r["status"] == "not_run" for r in entries)
    residual_q = [r for r in rows if r["policy_group"] == "residual_q"]
    statuses = {r["reaction_id"]: r["status"] for r in residual_q}
    assert statuses[REACTION_ID] != "not_run"
    assert statuses[SECOND_REACTION] == "not_attempted"


def test_model_residual_policy_wiring():
    from pes2ts_core.generation.planning.policy_adapter import (
        CandidatePolicyConfig,
        ModelResidualPolicy,
    )

    config = CandidatePolicyConfig()

    class State:
        pass

    state = State()
    state.control = build_test_control_like()
    state.spec = build_spec_like(state.control)
    state.target_lambda = 0.5

    policy = ModelResidualPolicy(config, lambda _s: [0.0])
    decision = policy.decide(state)
    # zero proposal dedups onto rule_zero; executed action stays the baseline
    assert decision.executed_candidate_index == 0

    policy_nonzero = ModelResidualPolicy(config, lambda _s: [0.8])
    decision_nonzero = policy_nonzero.decide(state)
    executed = decision_nonzero.candidates[decision_nonzero.executed_candidate_index]
    assert abs(executed[0] - 0.8) < 1e-9
    assert decision_nonzero.candidate_sources[decision_nonzero.executed_candidate_index] == "model_residual"

    policy_none = ModelResidualPolicy(config, lambda _s: None)
    decision_none = policy_none.decide(state)
    assert decision_none.executed_candidate_index == 0
    assert all(u == (0.0,) * len(u) for u in (
        decision_none.candidates[decision_none.executed_candidate_index],
    ))


def build_test_control_like():
    from tests.test_control_step import build_test_control

    return build_test_control()


def build_spec_like(control):
    from pes2ts_core.generation.planning.policy_adapter import build_residual_spec

    return build_residual_spec(control, epsilon_angstrom=0.1)
