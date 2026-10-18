"""Control-state snapshot + replay-branch tests (WP-3 §6.1 / IRC plan §5).

Covers: per-decision snapshot capture during collect-style runs; strict
validator rejections; zero/nonzero replay branches isolated from the round
store and episode index; budget inheritance + typed budget_stop;
missing-history refusal; parent-state digest chaining; and replay
determinism under the analytic fake-ACP transport.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pes2ts_core.generation.planning.control_replay import (
    BRANCH_OUTCOME_ACCEPTED,
    BRANCH_OUTCOME_BUDGET_STOP,
    ControlReplayError,
    load_snapshot,
    require_complete_history,
    run_replay_branch,
)
from pes2ts_core.generation.planning.control_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    ControlSnapshotError,
    snapshot_content_sha256,
    validate_snapshot,
)
from pes2ts_core.generation.planning.control_step import (
    persist_control_outputs,
    run_control_scan,
)
from pes2ts_core.generation.planning.policy_adapter import (
    CandidatePolicyConfig,
    EpsilonExplorePolicy,
    build_residual_spec,
)
from pes2ts_core.pesn.contracts.control import PolicyRewardSpec
from tests.test_control_round import (  # noqa: E402 - fixture reuse
    REACTION_ID,
    _build_two_case_inputs,
)
from tests.test_control_step import (  # noqa: E402 - fixture reuse
    ORIGIN_GEOMETRY,
    build_test_control,
    make_reward_spec,
    make_spec,
    make_transport,
    scan_policy,
    zero_decision,
)

from pes2ts_core.generation.planning.chemical_control import compile_chemical_control


def _analytic_snapshot_run(tmp_path: Path, *, snapshot_dir: Path | None = None):
    control = build_test_control()
    spec = make_spec(control)
    transport, calls = make_transport()
    result = run_control_scan(
        control=control,
        origin_geometry=ORIGIN_GEOMETRY,
        transport=transport,
        save_dir=tmp_path / "case",
        spec=spec,
        reward_spec=make_reward_spec(),
        decide=zero_decision,
        policy=scan_policy(),
        reaction_id=control.reaction_id,
        episode_tag="e000",
        round_id="r1",
        snapshot_dir=snapshot_dir,
    )
    return control, spec, result, calls


class TestSnapshotCapture:
    def test_snapshot_written_per_decision_during_collect(self, tmp_path):
        """Collect-style run writes one validated snapshot per decision point."""
        (manifest_path, legacy_root, _sel, _cfg) = _build_two_case_inputs(
            tmp_path, drop_second_plan=False
        )
        assert manifest_path.is_file()
        plan = json.loads((legacy_root / REACTION_ID / "plan.json").read_text())
        case_snapshot = json.loads(
            (tmp_path / "snapshots" / f"{REACTION_ID}.json").read_text()
        )
        control = compile_chemical_control(
            plan,
            reaction_id=REACTION_ID,
            n_grid_points=5,
            endpoint_geometry_r=case_snapshot["r_coordinates"],
            endpoint_geometry_p=case_snapshot["p_coordinates"],
        )
        spec = build_residual_spec(control, epsilon_angstrom=0.10)
        transport, _ = make_transport()
        policy = EpsilonExplorePolicy(CandidatePolicyConfig(exploration_seed=42))
        snapshot_dir = tmp_path / "snapshots_out"
        sink_docs: list[dict] = []
        result = run_control_scan(
            control=control,
            origin_geometry=plan["start_geometry"],
            transport=transport,
            save_dir=tmp_path / "case",
            spec=spec,
            reward_spec=make_reward_spec(),
            decide=policy.decide,
            policy=scan_policy(),
            reaction_id=REACTION_ID,
            episode_tag="e000",
            round_id="r1",
            snapshot_dir=snapshot_dir,
            snapshot_sink=sink_docs.append,
        )
        assert result["status"] in ("complete_interval", "partial_prefix")
        files = sorted(snapshot_dir.glob("d*.json"))
        assert len(files) >= 4  # grid 0.25/0.5/0.75/1.0 at minimum
        docs = [json.loads(path.read_text()) for path in files]
        for doc in docs:
            validate_snapshot(doc)
            assert doc["schema_version"] == SNAPSHOT_SCHEMA_VERSION
            assert doc["episode"]["reaction_id"] == REACTION_ID
            assert doc["episode"]["episode_id"] == f"{REACTION_ID}:e000"
            assert doc["episode"]["round_id"] == "r1"
            assert doc["episode"]["branch"] is None
            assert doc["control_identity"]["policy_id"] == "epsilon_explore_v1"
            assert doc["control_identity"]["environment_hash"]
            assert doc["control_identity"]["reward_spec_sha256"]
            assert doc["control_identity"]["spec_digest"]
            assert doc["control_identity"]["control_digest"]
            assert doc["control_identity"]["code_identity"]
            assert doc["physical_state"]["pending"]  # decision points have work left
            assert doc["rng"]["kind"] == "generator_state"
            assert doc["rng"]["exploration_seed"] == 42
            assert doc["rng"]["generator_state"] is not None
            assert doc["rng"]["determinism_note"]
        assert len(sink_docs) == len(files)
        for path, doc, sink_doc in zip(files, docs, sink_docs):
            assert json.loads(path.read_text()) == doc
            assert sink_doc == doc

    def test_validator_rejects_tampered_and_incomplete_docs(self, tmp_path):
        control, spec, _result, _calls = _analytic_snapshot_run(
            tmp_path, snapshot_dir=tmp_path / "snapshots"
        )
        del spec, control
        snapshot = load_snapshot(tmp_path / "snapshots" / "d000.json")

        tampered = copy.deepcopy(snapshot)
        tampered["unknown_top_level"] = 1
        with pytest.raises(ControlSnapshotError) as excinfo:
            validate_snapshot(tampered)
        assert excinfo.value.code == "UNKNOWN_KEY"

        tampered_section = copy.deepcopy(snapshot)
        tampered_section["budget"]["smuggled"] = 1
        with pytest.raises(ControlSnapshotError) as excinfo:
            validate_snapshot(tampered_section)
        assert excinfo.value.code == "UNKNOWN_KEY"

        wrong_schema = copy.deepcopy(snapshot)
        wrong_schema["schema_version"] = "pes2ts_control_state_snapshot_v0"
        with pytest.raises(ControlSnapshotError) as excinfo:
            validate_snapshot(wrong_schema)
        assert excinfo.value.code == "SCHEMA_UNKNOWN"

        missing_section = copy.deepcopy(snapshot)
        del missing_section["history"]
        with pytest.raises(ControlSnapshotError) as excinfo:
            validate_snapshot(missing_section)
        assert excinfo.value.code == "MISSING_KEY"

        missing_leaf = copy.deepcopy(snapshot)
        del missing_leaf["history"]["subdivision"]
        with pytest.raises(ControlSnapshotError) as excinfo:
            validate_snapshot(missing_leaf)
        assert excinfo.value.code == "MISSING_KEY"

        bad_pending = copy.deepcopy(snapshot)
        bad_pending["physical_state"]["pending"][0]["rejections"] = "many"
        with pytest.raises(ControlSnapshotError) as excinfo:
            validate_snapshot(bad_pending)
        assert excinfo.value.code in ("VALUE_INVALID", "TYPE_INVALID")

        # load_snapshot wraps validator errors as typed replay errors
        bad_path = tmp_path / "tampered.json"
        bad_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ControlReplayError) as excinfo:
            load_snapshot(bad_path)
        assert excinfo.value.code == "UNKNOWN_KEY"


class TestReplayBranch:
    def test_zero_residual_branch_isolated_from_round_store(self, tmp_path):
        control, spec, result, _calls = _analytic_snapshot_run(
            tmp_path, snapshot_dir=tmp_path / "snapshots"
        )
        case_dir = tmp_path / "case"
        persist_control_outputs(case_dir, result)

        round_root = tmp_path / "round"
        transitions_path = round_root / "transitions" / "transitions.jsonl"
        transitions_path.parent.mkdir(parents=True, exist_ok=True)
        transitions_path.write_bytes(b"SENTINEL-TRANSITIONS\n")
        episode_index_path = round_root / "transitions" / "episode_index.json"
        episode_index_path.write_text(
            json.dumps({"schema_version": "pesn_control_episode_index_v1",
                        "episodes": [{"episode_id": "sentinel"}]}),
            encoding="utf-8",
        )
        before = sorted(str(p) for p in round_root.rglob("*"))
        transitions_bytes = transitions_path.read_bytes()
        episode_index_bytes = episode_index_path.read_bytes()

        snapshot = load_snapshot(tmp_path / "snapshots" / "d001.json")
        branch_dir = tmp_path / "branch_b00"
        branch_transport, branch_calls = make_transport()
        branch = run_replay_branch(
            control=control,
            snapshot=snapshot,
            candidate_u=(0.0,),
            candidate_source="rule_zero",
            transport=branch_transport,
            save_dir=branch_dir,
            reward_spec=make_reward_spec(),
            spec=spec,
            policy=scan_policy(),
        )

        assert branch["schema_version"] == "pes2ts_control_replay_branch_v1"
        assert branch["branch"]["branch_label"] == "b00"
        assert branch["branch"]["source_episode_id"] == f"{control.reaction_id}:e000"
        assert branch["branch"]["source_decision_index"] == snapshot["episode"]["decision_index"]
        assert branch["branch"]["source_snapshot_sha256"] == snapshot_content_sha256(snapshot)
        assert branch["executed_candidate"]["residual_u"] == [0.0]
        assert branch["executed_candidate"]["source_label"] == "rule_zero"

        outcome = branch["outcome"]
        assert outcome["status"] == BRANCH_OUTCOME_ACCEPTED
        assert outcome["accepted"] is True
        assert outcome["target_resolved"] is True
        target_lambda = snapshot["physical_state"]["target_lambda"]
        assert outcome["target_lambda"] == pytest.approx(target_lambda)

        # exactly one logical-target continuation: all calls at the open target
        assert len(branch_calls) == 1
        expected_target = 1.5 + 1.0 * target_lambda
        assert branch_calls[0]["target"] == pytest.approx(expected_target)

        cost = branch["cost"]
        assert cost["scope"] == "branch_only"
        assert cost["host_requests"] == len(branch_calls) == 1
        assert cost["wall_seconds"] >= 0.0
        assert cost["n_optimization_cycles"] >= 0
        assert branch["budget_after"]["inherited_from_snapshot"] is True
        assert branch["budget_after"]["request_count"] == (
            snapshot["budget"]["request_count"] + cost["host_requests"]
        )
        assert branch["budget_after"]["remaining_host_requests"] == (
            snapshot["budget"]["max_host_requests"]
            - branch["budget_after"]["request_count"]
        )
        assert branch["accepted_frames"]
        assert branch["artifacts"][0] == "control_replay_branch.json"
        assert (branch_dir / "control_replay_branch.json").is_file()

        # diagnostic branches NEVER touch the round store or episode index
        assert transitions_path.read_bytes() == transitions_bytes
        assert episode_index_path.read_bytes() == episode_index_bytes
        after = sorted(str(p) for p in round_root.rglob("*"))
        assert after == before

    def test_branch_budget_inheritance_and_budget_stop(self, tmp_path):
        control, spec, _result, _calls = _analytic_snapshot_run(
            tmp_path, snapshot_dir=tmp_path / "snapshots"
        )
        snapshot = load_snapshot(tmp_path / "snapshots" / "d001.json")

        accepted_branch = run_replay_branch(
            control=control,
            snapshot=snapshot,
            candidate_u=(0.0,),
            candidate_source="rule_zero",
            transport=make_transport()[0],
            save_dir=tmp_path / "branch_inherited",
            reward_spec=make_reward_spec(),
            spec=spec,
            policy=scan_policy(),
        )
        assert accepted_branch["outcome"]["status"] == BRANCH_OUTCOME_ACCEPTED
        after = accepted_branch["budget_after"]
        assert after["snapshot_request_count"] == snapshot["budget"]["request_count"]
        assert after["request_count"] == (
            snapshot["budget"]["request_count"]
            + accepted_branch["cost"]["host_requests"]
        )
        assert after["remaining_host_requests"] == (
            snapshot["budget"]["max_host_requests"] - after["request_count"]
        )
        assert after["snapshot_wall_elapsed_seconds"] == pytest.approx(
            snapshot["budget"]["wall_elapsed_seconds"]
        )
        assert after["wall_elapsed_seconds"] >= after["snapshot_wall_elapsed_seconds"]
        assert after["remaining_wall_seconds"] <= (
            snapshot["budget"]["max_wall_seconds"]
            - snapshot["budget"]["wall_elapsed_seconds"]
        )

        exhausted = copy.deepcopy(snapshot)
        exhausted["budget"]["request_count"] = exhausted["budget"]["max_host_requests"]
        validate_snapshot(exhausted)
        stop_calls: list = []
        stop_transport, stop_calls_ref = make_transport()

        def recording_transport(request, attempt_dir, execution_id, attempt_id):
            stop_calls.append(attempt_dir)
            return stop_transport(request, attempt_dir, execution_id, attempt_id)

        branch_stop = run_replay_branch(
            control=control,
            snapshot=exhausted,
            candidate_u=(0.0,),
            candidate_source="rule_zero",
            transport=recording_transport,
            save_dir=tmp_path / "branch_stop",
            reward_spec=make_reward_spec(),
            spec=spec,
            policy=scan_policy(),
        )
        assert branch_stop["outcome"]["status"] == BRANCH_OUTCOME_BUDGET_STOP
        assert branch_stop["outcome"]["failure_code"] == "REQUEST_BUDGET_EXHAUSTED"
        assert branch_stop["cost"]["host_requests"] == 0
        assert stop_calls == []
        assert branch_stop["budget_after"]["request_count"] == exhausted["budget"]["request_count"]
        assert branch_stop["budget_after"]["remaining_host_requests"] == 0
        assert branch_stop["accepted_frames"] == []

        wall_exhausted = copy.deepcopy(snapshot)
        wall_exhausted["budget"]["wall_elapsed_seconds"] = (
            wall_exhausted["budget"]["max_wall_seconds"]
        )
        validate_snapshot(wall_exhausted)
        branch_wall_stop = run_replay_branch(
            control=control,
            snapshot=wall_exhausted,
            candidate_u=(0.0,),
            candidate_source="rule_zero",
            transport=make_transport()[0],
            save_dir=tmp_path / "branch_wall_stop",
            reward_spec=make_reward_spec(),
            spec=spec,
            policy=scan_policy(),
        )
        assert branch_wall_stop["outcome"]["status"] == BRANCH_OUTCOME_BUDGET_STOP
        assert branch_wall_stop["outcome"]["failure_code"] == "WALL_BUDGET_EXHAUSTED"
        assert branch_wall_stop["cost"]["host_requests"] == 0

    def test_missing_history_refused(self, tmp_path):
        control, spec, _result, _calls = _analytic_snapshot_run(
            tmp_path, snapshot_dir=tmp_path / "snapshots"
        )
        snapshot = load_snapshot(tmp_path / "snapshots" / "d000.json")

        broken = copy.deepcopy(snapshot)
        del broken["budget"]["request_count"]
        with pytest.raises(ControlReplayError) as excinfo:
            run_replay_branch(
                control=control,
                snapshot=broken,
                candidate_u=(0.0,),
                candidate_source="rule_zero",
                transport=make_transport()[0],
                save_dir=tmp_path / "b_missing",
                reward_spec=make_reward_spec(),
                spec=spec,
                policy=scan_policy(),
            )
        assert excinfo.value.code == "missing_history"

        no_pending = copy.deepcopy(snapshot)
        del no_pending["physical_state"]["pending"]
        with pytest.raises(ControlReplayError) as excinfo:
            require_complete_history(no_pending)
        assert excinfo.value.code == "missing_history"

        null_budget = copy.deepcopy(snapshot)
        null_budget["budget"]["max_host_requests"] = None
        with pytest.raises(ControlReplayError) as excinfo:
            require_complete_history(null_budget)
        assert excinfo.value.code == "missing_history"

        no_subdivision = copy.deepcopy(snapshot)
        del no_subdivision["history"]["subdivision"]["min_scale"]
        with pytest.raises(ControlReplayError) as excinfo:
            require_complete_history(no_subdivision)
        assert excinfo.value.code == "missing_history"

        complete = copy.deepcopy(snapshot)
        require_complete_history(complete)  # a valid snapshot passes

    def test_snapshot_chain_links_consecutive_digests(self, tmp_path):
        _control, _spec, _result, _calls = _analytic_snapshot_run(
            tmp_path, snapshot_dir=tmp_path / "snapshots"
        )
        d000 = json.loads((tmp_path / "snapshots" / "d000.json").read_text())
        d001 = json.loads((tmp_path / "snapshots" / "d001.json").read_text())
        assert d000["parent_state_sha256"] is not None
        assert d001["parent_state_sha256"] == snapshot_content_sha256(d000)
        d002_path = tmp_path / "snapshots" / "d002.json"
        if d002_path.is_file():
            d002 = json.loads(d002_path.read_text())
            assert d002["parent_state_sha256"] == snapshot_content_sha256(d001)

    def test_replay_determinism_and_candidate_paths(self, tmp_path):
        control, spec, _result, _calls = _analytic_snapshot_run(
            tmp_path, snapshot_dir=tmp_path / "snapshots"
        )
        snapshot = load_snapshot(tmp_path / "snapshots" / "d001.json")

        branch_a = run_replay_branch(
            control=control,
            snapshot=snapshot,
            candidate_u=(0.0,),
            candidate_source="rule_zero",
            transport=make_transport()[0],
            save_dir=tmp_path / "det_a",
            reward_spec=make_reward_spec(),
            spec=spec,
            policy=scan_policy(),
        )
        branch_b = run_replay_branch(
            control=control,
            snapshot=snapshot,
            candidate_u=(0.0,),
            candidate_source="rule_zero",
            transport=make_transport()[0],
            save_dir=tmp_path / "det_b",
            reward_spec=make_reward_spec(),
            spec=spec,
            policy=scan_policy(),
        )
        assert branch_a["outcome"]["status"] == branch_b["outcome"]["status"] == (
            BRANCH_OUTCOME_ACCEPTED
        )
        assert [f["geometry"] for f in branch_a["accepted_frames"]] == [
            f["geometry"] for f in branch_b["accepted_frames"]
        ]
        assert branch_a["cost"]["host_requests"] == branch_b["cost"]["host_requests"]

        target_lambda = snapshot["physical_state"]["target_lambda"]
        nonzero_transport, nonzero_calls = make_transport()
        branch_nonzero = run_replay_branch(
            control=control,
            snapshot=snapshot,
            candidate_u=(0.5,),
            candidate_source="model_residual",
            transport=nonzero_transport,
            save_dir=tmp_path / "branch_nonzero",
            reward_spec=make_reward_spec(),
            spec=spec,
            policy=scan_policy(),
        )
        assert branch_nonzero["outcome"]["status"] == BRANCH_OUTCOME_ACCEPTED
        assert branch_nonzero["executed_candidate"]["residual_u"] == [0.5]
        assert len(nonzero_calls) == 1
        nominal = 1.5 + 1.0 * target_lambda
        assert nonzero_calls[0]["target"] == pytest.approx(nominal + 0.5 * 0.10)
        assert branch_nonzero["quality"]["max_abs_deviation"] == pytest.approx(0.05)

        with pytest.raises(ControlReplayError) as excinfo:
            run_replay_branch(
                control=control,
                snapshot=snapshot,
                candidate_u=(5.0,),
                candidate_source="model_residual",
                transport=make_transport()[0],
                save_dir=tmp_path / "branch_illegal",
                reward_spec=make_reward_spec(),
                spec=spec,
                policy=scan_policy(),
            )
        assert excinfo.value.code == "candidate_illegal"

        tampered = copy.deepcopy(snapshot)
        tampered["control_identity"]["control_digest"] = "0" * 64
        with pytest.raises(ControlReplayError) as excinfo:
            run_replay_branch(
                control=control,
                snapshot=tampered,
                candidate_u=(0.0,),
                candidate_source="rule_zero",
                transport=make_transport()[0],
                save_dir=tmp_path / "branch_mismatch",
                reward_spec=make_reward_spec(),
                spec=spec,
                policy=scan_policy(),
            )
        assert excinfo.value.code == "control_mismatch"

        tampered_env = copy.deepcopy(snapshot)
        tampered_env["control_identity"]["environment_hash"] = "f" * 64
        with pytest.raises(ControlReplayError) as excinfo:
            run_replay_branch(
                control=control,
                snapshot=tampered_env,
                candidate_u=(0.0,),
                candidate_source="rule_zero",
                transport=make_transport()[0],
                save_dir=tmp_path / "branch_env",
                reward_spec=make_reward_spec(),
                spec=spec,
                policy=scan_policy(),
            )
        assert excinfo.value.code == "environment_mismatch"


class TestScanWithoutSnapshotsUnchanged:
    def test_no_snapshot_args_behaves_as_before(self, tmp_path):
        control, spec, result, calls = _analytic_snapshot_run(
            tmp_path, snapshot_dir=None
        )
        assert result["schema_version"] == "pes2ts_native_scan_result_v3"
        assert result["status"] == "complete_interval"
        assert len(result["frames"]) == 5
        assert len(result["_transitions"]) == 4
        assert len(calls) == 4
        assert not (tmp_path / "snapshots").exists()
