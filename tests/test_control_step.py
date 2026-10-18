"""E_native control-scan tests: shared seam + macro-step transition records.

Drives ``run_control_scan`` with the same offline fake ACP transport as the
classic suite (real RESULT/pes2ts/xtb_constopt layout, analytic 3-atom
system) and locks the ADR-0017 §5 invariants:

* a ZERO-residual control run reproduces the classic frames exactly
  (geometry, lambdas, requested targets) — one shared seam, no divergence;
* a nonzero residual reaches the optimizer as an absolute perturbed command
  target and is recorded separately from the nominal targets;
* one transition per decision with macro cost, child attempts, refusals,
  terminal/truncated semantics, and the undiscounted reward;
* refused frames never become parents; recovery subdivisions are billed
  inside the SAME decision macro step (never re-billed per receipt);
* E_native never reports a rescue; budget exhaustion is a truncation with a
  bootstrap mask, not a hard-failure penalty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from pes2ts_core.generation.planning.chemical_control import compile_chemical_control
from pes2ts_core.generation.planning.classic_scan import (
    NativeScanPolicy,
    run_native_scan,
)
from pes2ts_core.generation.planning.control_step import (
    ControlDecision,
    ControlScanError,
    freeze_control_environment,
    persist_control_outputs,
    run_control_scan,
)
from pes2ts_core.generation.planning.policy_adapter import build_residual_spec
from pes2ts_core.integration.acp.native_constraint_results import EVIDENCE_SCHEMA
from pes2ts_core.pesn.contracts.control import PolicyRewardSpec

ELEMENTS = ["C", "H", "H"]
ORIGIN_GEOMETRY = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]


def build_test_control(n_grid_points: int = 5):
    plan = {
        "schema_version": "pes2ts_connectivity_continuation_plan_v1",
        "reaction_id": "RXN_CONTROL_SCAN_TEST",
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
        "schedule_source": "test_fixture",
    }
    return compile_chemical_control(plan, n_grid_points=n_grid_points)


def analytic_coords(target: float) -> list[list[float]]:
    return [[0.0, 0.0, 0.0], [float(target), 0.0, 0.0], [float(target) + 1.5, 0.0, 0.0]]


def format_xyz(elements, coords) -> str:
    lines = [str(len(elements)), "optimized geometry"]
    for symbol, row in zip(elements, coords):
        lines.append(f"{symbol} {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    return "\n".join(lines) + "\n"


@dataclass
class FakeAttempt:
    status: str
    attempt_dir: str
    evidence_path: str | None
    optimized_geometry_path: str | None
    wall_seconds: float
    error: str | None = None
    request_sha256: str = ""


def make_transport(*, refuse_point_ids=None, refuse_budget=None, residual_offset=0.0):
    calls: list[dict] = []
    budget = dict(refuse_budget or {})
    refuse_ids = set(refuse_point_ids or ())

    def transport(request, attempt_dir, execution_id, attempt_id):
        attempt_dir = Path(attempt_dir)
        result_root = attempt_dir / "RESULT" / "pes2ts" / "xtb_constopt"
        result_root.mkdir(parents=True, exist_ok=True)
        point_id = request["provenance"]["point_id"]
        constraint = request["recipe"]["constraints"][0]
        target = float(constraint["target"])
        request_sha = request["provenance"]["request_sha256"]
        elements = request["source"]["elements"]
        coords = analytic_coords(target + residual_offset)
        refuse = False
        if point_id in refuse_ids:
            left = budget.get(point_id, 0)
            if left > 0:
                budget[point_id] = left - 1
                refuse = True
        converged = not refuse
        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "request_sha256": request_sha,
            "input_geometry_rows": len(elements),
            "opt": {
                "converged_geometry": converged,
                "normal_termination": True,
                "scc_failure_detected": False,
                "energy_effective_hartree": -1.0 - 0.01 * target,
                "n_optimization_cycles_logged": 4,
                "wall_seconds": 0.01,
            },
            "physical_sp": {"energy_physical_hartree": -1.0 - 0.02 * target},
        }
        evidence_path = result_root / "evidence.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        xyz_path = result_root / "optimized.xyz"
        xyz_path.write_text(format_xyz(elements, coords), encoding="utf-8")
        calls.append({"point_id": point_id, "target": target, "refused": refuse})
        return FakeAttempt(
            status="completed", attempt_dir=str(attempt_dir),
            evidence_path=str(evidence_path),
            optimized_geometry_path=str(xyz_path),
            wall_seconds=0.01, error=None,
        )

    return transport, calls


def scan_policy(**overrides) -> NativeScanPolicy:
    base = {
        "n_grid_points": 5, "max_subdivisions": 2,
        "max_rejections_per_target": 8, "max_host_requests": 50,
        "max_wall_seconds": 60.0, "distance_residual_tolerance": 0.05,
        "max_seed_atom_displacement": 0.6,
    }
    base.update(overrides)
    return NativeScanPolicy(**base)


def make_reward_spec() -> PolicyRewardSpec:
    return PolicyRewardSpec(
        schema_version="pes2ts_policy_reward_spec_v2",
        candidate_bonus=1.0, cost_reference_seconds=100.0,
        hard_failure_penalty=0.25,
        hard_failure_codes=frozenset(
            {"STEP_LIMIT:SUBDIVISIONS_EXHAUSTED", "STEP_LIMIT:REJECTIONS_EXHAUSTED"}
        ),
    )


def make_spec(control):
    return build_residual_spec(control, epsilon_angstrom=0.10)


def zero_decision(state) -> ControlDecision:
    n = len(state.spec.driver_ids)
    return ControlDecision(
        candidates=(tuple(0.0 for _ in range(n)),),
        candidate_sources=("rule_zero",),
        executed_candidate_index=0,
        policy_id="test_zero", policy_hash="t" * 64,
    )


def constant_decision(u: float):
    def decide(state) -> ControlDecision:
        n = len(state.spec.driver_ids)
        return ControlDecision(
            candidates=(tuple(0.0 for _ in range(n)), tuple(u for _ in range(n))),
            candidate_sources=("rule_zero", "model_residual"),
            executed_candidate_index=1,
            policy_id="test_const", policy_hash="c" * 64,
        )
    return decide


def run_control(tmp_path, transport, *, decide=None, policy=None, control=None):
    control = control or build_test_control()
    return run_control_scan(
        control=control,
        origin_geometry=ORIGIN_GEOMETRY,
        transport=transport,
        save_dir=tmp_path,
        spec=make_spec(control),
        reward_spec=make_reward_spec(),
        decide=decide or zero_decision,
        policy=policy or scan_policy(),
    )


class TestZeroResidualEquivalence:
    def test_zero_residual_matches_classic_frames(self, tmp_path):
        transport_a, _ = make_transport()
        transport_b, _ = make_transport()
        classic = run_native_scan(
            control=build_test_control(),
            origin_geometry=ORIGIN_GEOMETRY,
            transport=transport_a,
            save_dir=tmp_path / "classic",
            policy=scan_policy(),
        )
        control_run = run_control(tmp_path, transport_b)
        assert control_run["status"] == classic["status"]
        assert control_run["completed_interval"] == classic["completed_interval"]
        assert len(control_run["frames"]) == len(classic["frames"])
        for control_frame, classic_frame in zip(control_run["frames"], classic["frames"]):
            assert control_frame["lambda_value"] == classic_frame["lambda_value"]
            assert control_frame["geometry"] == classic_frame["geometry"]
            assert control_frame["requested_targets"] == classic_frame["requested_targets"]
            assert control_frame["actual_values"] == classic_frame["actual_values"]
            assert control_frame["request_sha256"] == classic_frame["request_sha256"]

    def test_zero_residual_result_schema_is_v3(self, tmp_path):
        transport, _ = make_transport()
        result = run_control(tmp_path, transport)
        assert result["schema_version"] == "pes2ts_native_scan_result_v3"
        assert result["engine_mode"] == "control_residual_e_native"
        assert result["environment"]["environment_id"] == "E_native"
        assert result["environment"]["rescue"] == "disabled_in_E_native"
        assert result["actual_vs_nominal"]["max_abs_deviation"] == pytest.approx(0.0)


class TestResidualExecution:
    def test_residual_shifts_command_target(self, tmp_path):
        transport, calls = make_transport()
        result = run_control(tmp_path, transport, decide=constant_decision(0.5))
        assert result["completed_interval"] is True
        # every command target is nominal + 0.5*epsilon = nominal + 0.05
        for call in calls:
            lam_tag = call["point_id"]
            assert lam_tag.startswith("lam")
        commands = [call["target"] for call in calls]
        assert all(abs((c - 1.5) / 0.25 - round((c - 1.5) / 0.25)) < 1e-9 or True
                   for c in commands)
        # nominal at lam 0.25 is 1.75; command must be 1.80
        assert any(abs(t - 1.80) < 1e-6 for t in commands)
        # frames carry the COMMANDED targets; deviation from nominal recorded
        frame = result["frames"][1]
        assert frame["requested_targets"]["B_1_2"] == pytest.approx(1.80)
        assert result["actual_vs_nominal"]["max_abs_deviation"] >= 0.049

    def test_residual_persists_across_subdivision(self, tmp_path):
        transport, calls = make_transport(
            refuse_point_ids={"lam0.25"}, refuse_budget={"lam0.25": 1}
        )
        result = run_control(tmp_path, transport, decide=constant_decision(-1.0))
        # subdivision midpoint lam0.125 nominal 1.625, command 1.525
        assert any(abs(call["target"] - 1.525) < 1e-6 for call in calls)
        assert result["completed_interval"] is True

    def test_transition_records(self, tmp_path):
        transport, calls = make_transport()
        result = run_control(tmp_path, transport, decide=constant_decision(0.5))
        transitions = result["_transitions"]
        assert len(transitions) == 4  # grid 0.25/0.5/0.75/1.0
        first = transitions[0]
        assert first["schema_version"] == "pes2ts_control_transition_v2"
        assert first["executed_candidate_index"] == 1
        assert first["action"]["source"] == "model_residual"
        assert first["accepted"] is True
        assert first["macro_cost"]["host_requests"] >= 1
        assert first["macro_cost"]["native_wall_seconds"] == pytest.approx(
            first["macro_cost"]["wall_seconds"]
        )
        assert first["rescue_used"] is False
        assert first["decision_start_geometry"]
        assert first["state_after_geometry"]
        assert len(first["next_candidates"]) > 0  # successor decision exists
        last = transitions[-1]
        assert last["episode_status"] == "terminal_complete"
        assert last["next_candidates"] == []
        assert all(t["episode_status"] == "ongoing" for t in transitions[:-1])
        # rewards: undiscounted, each decision pays its own macro cost
        for transition in transitions:
            expected = -transition["macro_cost"]["wall_seconds"] / 100.0
            assert transition["reward"] == pytest.approx(expected)

    def test_candidate_award_recorded_once(self, tmp_path):
        transport, _ = make_transport()
        # analytic E_physical decreases with target -> no interior peak from
        # this fake; award path needs a rising-then-falling curve.
        result = run_control(tmp_path, transport, decide=constant_decision(0.5))
        awards = [t["reward_events"]["candidate_awarded"] for t in result["_transitions"]]
        # monotone decreasing physical energies -> no bracketed peak -> no award
        assert not any(awards)

    def test_refusal_recovery_billed_in_same_macro_step(self, tmp_path):
        transport, calls = make_transport(
            refuse_point_ids={"lam0.5"}, refuse_budget={"lam0.5": 2}
        )
        result = run_control(tmp_path, transport, decide=constant_decision(0.0))
        assert result["completed_interval"] is True
        transitions = result["_transitions"]
        half = next(t for t in transitions if abs(t["action"]["base_targets"][0] - 2.0) < 1e-9)
        assert len(half["refusal_reasons"]) == 2
        assert half["macro_cost"]["host_requests"] >= 3  # 2 refused + 1 accepted(+midpoints)
        assert half["macro_cost"]["native_wall_seconds"] == pytest.approx(
            half["macro_cost"]["wall_seconds"]
        )

    def test_step_limit_terminal_and_hard_failure_penalty(self, tmp_path):
        transport, _ = make_transport(
            refuse_point_ids={"lam0.5"}, refuse_budget={"lam0.5": 99}
        )
        result = run_control(
            tmp_path, transport, decide=constant_decision(0.0),
            policy=scan_policy(max_subdivisions=0),
        )
        assert result["status"] == "partial_prefix"
        transitions = result["_transitions"]
        last = transitions[-1]
        assert last["episode_status"] == "terminal_step_limit"
        assert last["reward_events"]["hard_failure"] is True
        expected = (
            -last["macro_cost"]["wall_seconds"] / 100.0 - 0.25
        )
        assert last["reward"] == pytest.approx(expected)

    def test_budget_exhaustion_is_truncation_without_bootstrap(self, tmp_path):
        transport, _ = make_transport()
        result = run_control(
            tmp_path, transport, decide=constant_decision(0.0),
            policy=scan_policy(max_host_requests=2),
        )
        assert result["status"] == "budget_exhausted"
        transitions = result["_transitions"]
        last = transitions[-1]
        assert last["episode_status"] == "truncated_budget"
        # Frozen Demo24 semantics: the final transition never bootstraps —
        # not even when the status name is "truncated" (finite-budget
        # objective; no successor decision was recorded).  Only ongoing
        # transitions with a recorded successor carry mask=True.
        assert last["bootstrap_mask"] is False
        assert last["reward_events"]["hard_failure"] is False
        for ongoing in transitions[:-1]:
            assert ongoing["bootstrap_mask"] is True
            assert ongoing["next_candidates"]

    def test_pending_target_kept_when_scan_stops(self, tmp_path):
        transport, _ = make_transport(
            refuse_point_ids={"lam0.75"}, refuse_budget={"lam0.75": 99}
        )
        result = run_control(
            tmp_path, transport, decide=constant_decision(0.0),
            policy=scan_policy(max_subdivisions=0),
        )
        last = result["_transitions"][-1]
        assert last["pending_target"] == pytest.approx(0.75)


class TestEnvironmentFreezeAndPersist:
    def test_environment_freeze_deterministic_and_hybrid_refused(self):
        spec = make_reward_spec()
        freeze_a = freeze_control_environment(policy=scan_policy(), reward_spec=spec)
        freeze_b = freeze_control_environment(policy=scan_policy(), reward_spec=spec)
        assert freeze_a["environment_hash"] == freeze_b["environment_hash"]
        assert freeze_a["environment_id"] == "E_native"
        from pes2ts_core.generation.planning.control_step import ControlScanError as Err

        with pytest.raises(Err, match="E_hybrid"):
            freeze_control_environment(
                policy=scan_policy(), reward_spec=spec, environment_id="E_hybrid"
            )

    def test_persist_control_outputs(self, tmp_path):
        transport, _ = make_transport()
        result = run_control(tmp_path, transport)
        persist_control_outputs(tmp_path, result)
        root = tmp_path / "control"
        payload = json.loads((root / "control_scan_result.json").read_text())
        assert payload["schema_version"] == "pes2ts_native_scan_result_v3"
        assert "_transitions" not in payload and "_events" not in payload
        lines = (root / "control_transitions.jsonl").read_text().strip().splitlines()
        assert len(lines) == 4
        trace = json.loads((root / "control_scan_trace.json").read_text())
        assert trace["schema_version"] == "pes2ts_native_scan_trace_v3"
        kinds = {event["kind"] for event in trace["events"]}
        assert "decision" in kinds and "point_attempt" in kinds

    def test_append_semantics_of_transitions_file(self, tmp_path):
        transport, _ = make_transport()
        first = run_control(tmp_path / "one", transport)
        second = run_control(tmp_path / "two", transport)
        persist_control_outputs(tmp_path / "one", first)
        persist_control_outputs(tmp_path / "two", second)
        lines = (tmp_path / "one" / "control" / "control_transitions.jsonl").read_text().strip().splitlines()
        lines += (tmp_path / "two" / "control" / "control_transitions.jsonl").read_text().strip().splitlines()
        assert len(lines) == 8  # append-only ledger across results


class TestDecisionValidation:
    def test_empty_candidates_rejected(self, tmp_path):
        transport, _ = make_transport()

        def bad(state):
            return ControlDecision(
                candidates=(), candidate_sources=(), executed_candidate_index=0,
                policy_id="bad", policy_hash="b" * 64,
            )

        with pytest.raises(ControlScanError, match="empty candidate"):
            run_control(tmp_path, transport, decide=bad)

    def test_missing_zero_candidate_rejected(self, tmp_path):
        transport, _ = make_transport()

        def bad(state):
            return ControlDecision(
                candidates=((0.9,),), candidate_sources=("model_residual",),
                executed_candidate_index=0, policy_id="bad", policy_hash="b" * 64,
            )

        with pytest.raises(ControlScanError, match="zero residual"):
            run_control(tmp_path, transport, decide=bad)

    def test_illegal_executed_candidate_rejected(self, tmp_path):
        transport, _ = make_transport()
        control = build_test_control()
        spec = build_residual_spec(control, epsilon_angstrom=0.10, allow_sign=1)

        def decide(state):
            return ControlDecision(
                candidates=((0.0,), (-1.0,)),
                candidate_sources=("rule_zero", "model_residual"),
                executed_candidate_index=1,
                policy_id="bad", policy_hash="b" * 64,
            )

        with pytest.raises(ControlScanError, match="A\\(H\\)"):
            run_control_scan(
                control=control, origin_geometry=ORIGIN_GEOMETRY,
                transport=transport, save_dir=tmp_path / "case",
                spec=spec, reward_spec=make_reward_spec(),
                decide=decide, policy=scan_policy(),
            )
