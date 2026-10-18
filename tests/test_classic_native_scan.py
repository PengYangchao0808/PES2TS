"""Classic native per-point scan controller tests (G2-AB3 WP-5).

Drives ``run_native_scan`` with an offline fake transport that writes the real
ACP product layout (``RESULT/pes2ts/xtb_constopt/evidence.json`` +
``optimized.xyz``, schema ``pes2ts_xtb_constrained_opt_evidence_v1``) and an
analytic 3-atom H-A-B system (driver distance solved in closed form).

Invariants locked here:
* refused/unconverged frames NEVER become the parent frame;
* subdivision keeps the original target queued — lambda 0.5 is retried and
  eventually accepted, or the scan stops typed with STEP_LIMIT;
* ``completed_interval`` only when every planned grid lambda has an accepted
  frame;
* budgets stop the scan with ``budget_exhausted``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.generation.planning.chemical_control import compile_chemical_control
from pes2ts_core.generation.planning.classic_scan import (
    STATUS_BUDGET_EXHAUSTED,
    STATUS_COMPLETE,
    STATUS_PARTIAL_PREFIX,
    STATUS_REJECTED_INPUT,
    NativeScanError,
    NativeScanPolicy,
    run_native_scan,
)
from pes2ts_core.integration.acp.native_constraint_results import EVIDENCE_SCHEMA

# 3-atom analytic system: C at origin, driver H at (d, 0, 0), spectator H at
# (d + 1.5, 0, 0).  Driver distance d is exact by construction; the spectator
# contact d(H1, H2) = 1.5 A stays outside every covalent/bonded threshold for
# the whole scan range d in [1.5, 2.5], so chemistry checks never trip on
# non-driver pairs.
ELEMENTS = ["C", "H", "H"]
ORIGIN_GEOMETRY = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]]


def build_test_control(n_grid_points: int = 5):
    plan = {
        "schema_version": "pes2ts_connectivity_continuation_plan_v1",
        "reaction_id": "RXN_NATIVE_SCAN_TEST",
        "elements": list(ELEMENTS),
        "atom_map_order": [1, 2, 3],
        "charge": 0,
        "multiplicity": 1,
        "active_edits": [{"edit_kind": "broken", "maps": [1, 2]}],
        "drivers": [
            {
                "id": "B_1_2",
                "kind": "distance",
                "atoms": [0, 1],
                "maps": [1, 2],
                "role": "driver",
                "unit": "angstrom",
                "lambda_values": [0.0, 1.0],
                "values": [1.5, 2.5],
            }
        ],
        "execution_direction": "R_to_P",
        "origin_endpoint": "R",
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
    """Fake ACP transport; refuses listed point_ids while budget lasts.

    ``residual_offset`` shifts the optimized driver distance away from the
    requested target (chemistry/residual rejection path).
    """
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
        calls.append(
            {
                "point_id": point_id,
                "target": target,
                "refused": refuse,
                "attempt_dir": str(attempt_dir),
                "request_sha256": request_sha,
            }
        )
        return FakeAttempt(
            status="completed",
            attempt_dir=str(attempt_dir),
            evidence_path=str(evidence_path),
            optimized_geometry_path=str(xyz_path),
            wall_seconds=0.01,
            error=None,
        )

    return transport, calls, budget


def scan_policy(**overrides) -> NativeScanPolicy:
    base = {
        "n_grid_points": 5,
        "max_subdivisions": 2,
        "max_rejections_per_target": 8,
        "max_host_requests": 50,
        "max_wall_seconds": 60.0,
        "distance_residual_tolerance": 0.05,
        "max_seed_atom_displacement": 0.6,
    }
    base.update(overrides)
    return NativeScanPolicy(**base)


def run(tmp_path, transport, *, policy=None, control=None, origin=None):
    return run_native_scan(
        control=control or build_test_control(),
        origin_geometry=origin if origin is not None else ORIGIN_GEOMETRY,
        transport=transport,
        save_dir=tmp_path / "case",
        policy=policy or scan_policy(),
    )


def accepted_dirs(result) -> set[str]:
    return {f["attempt_dir"] for f in result["frames"] if f["attempt_dir"]}


def lambdas(result) -> list[float]:
    return [f["lambda_value"] for f in result["frames"]]


def test_refused_frames_never_become_parent(tmp_path):
    transport, calls, _ = make_transport(
        refuse_point_ids={"lam0.5"}, refuse_budget={"lam0.5": 99}
    )
    result = run(tmp_path, transport, policy=scan_policy(max_subdivisions=0))
    assert result["status"] == STATUS_PARTIAL_PREFIX
    assert result["failure_code"] == "STEP_LIMIT:SUBDIVISIONS_EXHAUSTED"
    assert result["completed_interval"] is False

    refused = {c["attempt_dir"] for c in calls if c["refused"]}
    assert refused, "the fake transport must actually refuse lam0.5"
    assert refused.isdisjoint(accepted_dirs(result))

    assert 0.5 not in lambdas(result)
    events = result["_events"]
    assert any(
        e["kind"] == "refusal" and abs(e["lambda"] - 0.5) < 1e-9 for e in events
    )
    accepted_at_half = [
        e for e in events if e["kind"] == "accepted" and abs(e["lambda"] - 0.5) < 1e-9
    ]
    assert accepted_at_half == []
    # Parent chain only references already-accepted frames.
    for frame in result["frames"][1:]:
        assert frame["parent_frame_index"] is not None
        assert 0 <= frame["parent_frame_index"] < frame["frame_index"]
    # No refused attempt dir is ever used as an accepted frame's attempt.
    for frame in result["frames"][1:]:
        assert frame["attempt_dir"] not in refused


def test_subdivision_retries_original_target_until_accepted(tmp_path):
    transport, calls, _ = make_transport(
        refuse_point_ids={"lam0.5"}, refuse_budget={"lam0.5": 2}
    )
    result = run(tmp_path, transport, policy=scan_policy(max_subdivisions=2))
    events = result["_events"]
    attempts_at_half = [
        e for e in events if e["kind"] == "point_attempt" and abs(e["lambda"] - 0.5) < 1e-9
    ]
    # The original target 0.5 is attempted at least twice (two refusals).
    assert len(attempts_at_half) >= 2
    outcomes = [e["outcome"] for e in attempts_at_half]
    assert outcomes.count("rejected_not_converged") >= 2
    assert outcomes[-1] == "computed"
    # Eventually accepted, and the whole grid completes.
    assert any(
        e["kind"] == "accepted" and abs(e["lambda"] - 0.5) < 1e-9 for e in events
    )
    assert result["status"] == STATUS_COMPLETE
    assert result["completed_interval"] is True
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert any(abs(x - lam) < 1e-9 for x in lambdas(result))
    # Subdivision midpoints really were inserted between 0.25 and 0.5.
    midpoints = [x for x in lambdas(result) if 0.25 < x < 0.5 - 1e-12]
    assert midpoints
    # The accepted 0.5 frame is not one of the refused attempts.
    refused = {c["attempt_dir"] for c in calls if c["refused"]}
    frame_half = next(f for f in result["frames"] if abs(f["lambda_value"] - 0.5) < 1e-9)
    assert frame_half["attempt_dir"] not in refused
    # Actual driver values met the targets within the residual tolerance.
    for frame in result["frames"][1:]:
        for key, actual in frame["actual_values"].items():
            assert abs(actual - frame["requested_targets"][key]) <= 0.05 + 1e-9
    assert result["energy_channel"]["physical_available"] is True


def test_completed_interval_requires_full_grid_coverage(tmp_path):
    transport, _, _ = make_transport(
        refuse_point_ids={"lam0.75"}, refuse_budget={"lam0.75": 99}
    )
    result = run(tmp_path, transport, policy=scan_policy(max_subdivisions=0))
    assert result["status"] == STATUS_PARTIAL_PREFIX
    assert result["completed_interval"] is False
    assert 0.75 not in lambdas(result)
    # Prefix 0.0/0.25/0.5 is present but the interval is still incomplete.
    for lam in (0.0, 0.25, 0.5):
        assert any(abs(x - lam) < 1e-9 for x in lambdas(result))
    assert result["failure_code"] == "STEP_LIMIT:SUBDIVISIONS_EXHAUSTED"


def test_host_request_budget_exhaustion_is_typed(tmp_path):
    transport, calls, _ = make_transport()
    result = run(tmp_path, transport, policy=scan_policy(max_host_requests=2))
    assert result["status"] == STATUS_BUDGET_EXHAUSTED
    assert result["failure_code"] == "REQUEST_BUDGET_EXHAUSTED"
    assert result["completed_interval"] is False
    assert result["cost"]["host_requests"] == 2
    assert len(calls) == 2
    # Origin + two accepted grid points only.
    assert len(result["frames"]) == 3
    assert lambdas(result)[0] == 0.0


def test_residual_violation_is_refused_not_accepted(tmp_path):
    # Optimized distance off by 0.20 A > tolerance 0.05 -> chemistry refusal.
    transport, calls, _ = make_transport(residual_offset=0.20)
    result = run(tmp_path, transport, policy=scan_policy(max_subdivisions=0))
    assert result["status"] == STATUS_PARTIAL_PREFIX
    assert result["completed_interval"] is False
    assert result["failure_code"] == "STEP_LIMIT:SUBDIVISIONS_EXHAUSTED"
    assert lambdas(result) == [0.0]
    assert calls[0]["target"] == pytest.approx(1.75)
    assert any(
        "residual" in str(e.get("reason", "")) for e in result["_events"]
        if e["kind"] == "refusal"
    )


def test_origin_geometry_mismatch_and_nonfinite_inputs(tmp_path):
    transport, _, _ = make_transport()
    with pytest.raises(NativeScanError, match="origin geometry"):
        run(tmp_path, transport, origin=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    bad = [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [3.0, 0.0, 0.0]]
    result = run(tmp_path, transport, origin=bad)
    assert result["status"] == STATUS_REJECTED_INPUT
    assert result["failure_code"] == "INPUT_GEOMETRY_INVALID"
    assert result["frames"] == [] or result["n_accepted_frames_beyond_origin"] == 0


def test_policy_payload_round_trip():
    policy = scan_policy(max_host_requests=7)
    payload = policy.to_payload()
    assert payload["max_host_requests"] == 7
    assert payload["n_grid_points"] == 5
    assert NativeScanPolicy(**payload).max_host_requests == 7
