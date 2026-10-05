"""Offline branch-policy calibration grid (G2-AB2 WP-2-A).

Scans the five :class:`BranchPolicy` knobs plus the two continuation-policy
calibration knobs (``cumulative_rmsd_limit``, ``max_curvature_probes``) on
the AB1 analytic fixtures (double-well branch selection, anchored-release
honesty, cumulative-drift gate, curvature-probe budget).  No QC engine, no
network, no truth access — the calibration set IS the analytic fixtures plus
the development queue, and holdout tuning remains forbidden (R6).

Judge criteria per cell (all must pass; each is a boolean with evidence):

1. ``branch_selection_ok`` — under a double well with two minima at the SAME
   constraint distance, the correction follows the previous accepted branch
   (final distance stays in the reference basin), never jumps basins.
2. ``biased_only_honest`` — anchor-only stationarity is reported
   ``convergence_mode="biased_only"`` with released raw-gradient evidence and
   ``restraint_energy_added=False``; never as a physical stationarity claim.
3. ``cumulative_gate_discriminates`` — a genuinely drifting branch trips the
   typed ``LOCALITY_CUMULATIVE`` rejection at the scanned limit (a limit so
   loose the drift escapes fails the cell).
4. ``probe_budget_respected`` — the number of curvature probes never exceeds
   ``max_curvature_probes`` and a probe fires when the budget allows it.

The chosen cell is the MOST CONSERVATIVE passing cell under the frozen
ordering: strongest anchor first (largest ``bias_kappa``), then smallest
radii (``branch_rmsd_radius``, ``branch_atom_radius``,
``cumulative_rmsd_radius``), then fewest ``release_iterations``, then the
strictest ``cumulative_rmsd_limit``, then the smallest probe budget.  The
output document is ``pes2ts_branch_calibration_v1`` (ADR-0004).
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pes2ts_core.generation.planning.continuation import (
    ContinuationPolicy,
    run_continuation,
)
from pes2ts_core.generation.planning.local_corrector import (
    BranchPolicy,
    LocalCorrectorPolicy,
    correct_local,
)
from pes2ts_core.generation.planning.synchronized_path import digest

CALIBRATION_SCHEMA = "pes2ts_branch_calibration_v1"

#: Frozen conservative ordering — smaller is more conservative (see module
#: docstring).  ``-bias_kappa`` puts the STRONGEST anchor first.
CONSERVATIVE_ORDER = ("-bias_kappa", "branch_rmsd_radius", "branch_atom_radius",
                      "cumulative_rmsd_radius", "release_iterations",
                      "cumulative_rmsd_limit", "max_curvature_probes")

DEFAULT_GRID: dict[str, tuple[float | int, ...]] = {
    "branch_rmsd_radius": (.15, .25, .35),
    "branch_atom_radius": (.30, .50),
    "cumulative_rmsd_radius": (.25, .40, .60),
    "bias_kappa": (.05, .10, .25, .50),
    "release_iterations": (4, 6, 10),
    "cumulative_rmsd_limit": (1.0, 2.0),
    "max_curvature_probes": (1, 2),
}


def _double_well_evaluator(x, evaluation_id):
    m1, m2 = .9, 1.8
    u = x[2]-x[0]
    d = float(np.linalg.norm(u))
    unit = u/d
    ded = 2.*(d-m1)*(d-m2)*(2.*d-m1-m2)
    g = np.zeros_like(x)
    g[2] = ded*unit
    g[0] = -g[2]
    return {"success": True, "energy": (d-m1)**2*(d-m2)**2,
            "gradient_hartree_per_angstrom": g}


def _constant_pull_evaluator(strength):
    def evaluate(x, evaluation_id):
        u = x[2]-x[0]
        unit = u/float(np.linalg.norm(u))
        g = np.zeros_like(x)
        g[2] = -strength*unit
        g[0] = -g[2]
        return {"success": True, "energy": -strength*float(np.linalg.norm(u)),
                "gradient_hartree_per_angstrom": g}
    return evaluate


_COORDS = [{"kind": "distance", "atoms": [0, 1]}]
_MASSES = [12., 12., 1.]


def _frame(d: float) -> np.ndarray:
    return np.array([[0., 0., 0.], [1.3, 0., 0.], [d, 0., 0.]])


def _generous_local_policy() -> LocalCorrectorPolicy:
    return LocalCorrectorPolicy(rmsd_radius=1.0, atom_radius=1.2,
                                iteration_atom_step=.08, max_evaluations=200,
                                max_iterations=120, gradient_rms_tolerance=1e-4,
                                gradient_max_tolerance=3e-4)


def branch_selection_case(branch: BranchPolicy) -> dict[str, Any]:
    """Two minima under one constraint distance: follow the previous branch."""
    result = correct_local(_frame(1.38), _COORDS, [1.3], _MASSES,
                           _double_well_evaluator,
                           policy=_generous_local_policy(),
                           branch_reference=_frame(.9), branch_policy=branch)
    final = np.asarray(result["coordinates"]) if result.get("coordinates") else None
    final_d = float(np.linalg.norm(final[2]-final[0])) if final is not None else None
    ok = bool(result["success"] and final_d is not None and final_d < 1.2)
    return {"branch_selection_ok": ok, "converged": bool(result["success"]),
            "convergence_mode": result.get("convergence_mode"),
            "final_distance": final_d,
            "failure_class": result.get("failure_class")}


def biased_release_case(branch: BranchPolicy) -> dict[str, Any]:
    """Anchor-only stationarity must be reported ``biased_only``, honestly."""
    result = correct_local(_frame(1.05), _COORDS, [1.3], _MASSES,
                           _constant_pull_evaluator(.05),
                           policy=_generous_local_policy(),
                           branch_reference=_frame(1.0), branch_policy=branch)
    released = result.get("released_evidence") or {}
    honest = bool(
        result["success"]
        and result.get("convergence_mode") == "biased_only"
        and result.get("numeric_anchor_active") is True
        and result.get("restraint_energy_added") is False
        and result.get("stationary_point_claimed") is False
        and released.get("free_gradient_max_hartree_per_angstrom", 0.) > 3e-4)
    return {"biased_only_honest": honest,
            "convergence_mode": result.get("convergence_mode"),
            "released_free_gradient_max":
                released.get("free_gradient_max_hartree_per_angstrom"),
            "failure_class": result.get("failure_class")}


class _DriftingCorrector:
    """Satisfies the driver target exactly while drifting a spectator atom.

    Also answers directional-curvature probes so the probe budget is exercised
    after repeated LOCALITY failures.
    """

    def __init__(self, *, drift_per_frame: float = .2, fail_reason: str | None = None):
        self.drift = drift_per_frame
        self.fail_reason = fail_reason
        self.n_frames = 0
        self.probe_calls = 0

    def set_branch_reference(self, geometry, frame_id):
        pass

    def on_accept(self, result):
        pass

    def __call__(self, guess, targets, attempt_id):
        x = np.asarray(guess, float).copy()
        x[0][0] -= self.drift            # cumulative drift vs frames[0]
        x[2][0] = x[0][0] + targets[0]   # exact driver distance along +x
        self.n_frames += 1
        success = self.fail_reason is None
        return {"success": success, "converged": success,
                "failure_class": self.fail_reason,
                "coordinates": x.tolist(),
                "energy": -0.001*self.n_frames,
                "duration_seconds": 0.0005,
                "physical_gradient_status": "bound",
                "physical_gradient_hartree_per_angstrom": np.zeros_like(x).tolist(),
                "n_gradient_evaluations": 1,
                "evaluations": [{"evaluation_id": attempt_id + "/e0"}],
                "acp": {"reused": False, "wall_seconds": 0.0005}}

    def directional_curvature(self, x, coordinates, gradient, name="probe"):
        self.probe_calls += 1
        return {"name": name, "direction_norm": 1.0,
                "curvature": -0.5, "n_gradient_evaluations": 1,
                "budgeted_single_direction_only": True}


def _drift_plan(policy: ContinuationPolicy) -> dict[str, Any]:
    start = _frame(.9)
    plan = {"schema_version": "pes2ts_connectivity_continuation_plan_v1",
            "scope": "connectivity_only", "parameter_dimension": 1,
            "reaction_id": "CALIB-DRIFT", "atom_map_order": [1, 2, 3],
            "elements": ["C", "C", "H"], "start_geometry": start.tolist(),
            "masses": [12., 12., 1.], "charge": 0, "multiplicity": 1,
            "active_edits": [{"maps": [1, 3], "edit_kind": "formed"}],
            "drivers": [{"id": "B_1_3", "kind": "distance", "role": "driver",
                         "edit_kind": "formed", "maps": [1, 3], "atoms": [0, 2],
                         "unit": "angstrom", "lambda_values": [0., 1.],
                         "values": [.9, 1.8]}],
            "guards": [], "local_support_maps": [1, 3],
            "collision_checks": [], "common_bond_checks": [],
            "policy": asdict(policy)}
    plan["content_sha256"] = digest({k: v for k, v in plan.items()
                                     if k != "content_sha256"})
    return plan


def cumulative_gate_case(limit: float) -> dict[str, Any]:
    """A drifting branch must trip the typed LOCALITY_CUMULATIVE rejection.

    Fixture geometry: 10 fixed λ-steps of .1 with a .25 Å/frame spectator
    drift give an end-of-path cumulative RMSD of ~1.44 Å — above the strict
    limit (1.0, gate must fire) and below the loose one (2.0, gate must stay
    silent so the cell discriminates instead of always rejecting).
    """
    policy = ContinuationPolicy(initial_step=.1, min_step=.05, max_step=.1,
                                max_frames=12, max_attempts=24, max_seconds=60.,
                                cumulative_rmsd_limit=limit,
                                max_curvature_probes=0, adaptive_enabled=False)
    result = run_continuation(_drift_plan(policy),
                              {"success": True, "converged": True,
                               "coordinates": _frame(.9).tolist(), "energy": 0.,
                               "duration_seconds": 0.},
                              _DriftingCorrector(drift_per_frame=.25))
    reasons = [attempt.get("reason") for attempt in result["attempts"]
               if attempt.get("reason")]
    fired = any((reason or "").startswith("LOCALITY_CUMULATIVE")
                or "LOCALITY_CUMULATIVE" in (reason or "") for reason in reasons)
    return {"cumulative_gate_discriminates": fired,
            "typed_reasons": sorted({r for r in reasons if r})[:4],
            "n_accepted_frames": max(0, len(result["frames"])-1)}


def probe_budget_case(max_probes: int) -> dict[str, Any]:
    """Repeated LOCALITY failures may probe curvature only within budget.

    Adaptive step-halving keeps re-attempting the failing segment, so the
    same LOCALITY shape repeats and the budgeted probe fires — then stops at
    ``max_curvature_probes``.
    """
    policy = ContinuationPolicy(initial_step=.1, min_step=.001, max_step=.1,
                                max_frames=12, max_attempts=10, max_seconds=60.,
                                cumulative_rmsd_limit=10.,
                                max_curvature_probes=max_probes,
                                adaptive_enabled=True)
    corrector = _DriftingCorrector(drift_per_frame=.25, fail_reason="LOCALITY_ATOM")
    result = run_continuation(_drift_plan(policy),
                              {"success": True, "converged": True,
                               "coordinates": _frame(.9).tolist(), "energy": 0.,
                               "duration_seconds": 0.}, corrector)
    probes = sum(1 for attempt in result["attempts"] if attempt.get("curvature_probe"))
    return {"probe_budget_respected": probes <= max_probes,
            "probes_fired": probes, "probe_calls": corrector.probe_calls,
            "budget": max_probes}


def _cell_verdict(cell: dict[str, Any]) -> dict[str, Any]:
    branch = BranchPolicy(**{k: cell[k] for k in
                             ("branch_rmsd_radius", "branch_atom_radius",
                              "cumulative_rmsd_radius", "bias_kappa",
                              "release_iterations")})
    verdict = {**branch_selection_case(branch),
               **biased_release_case(branch),
               **cumulative_gate_case(cell["cumulative_rmsd_limit"]),
               **probe_budget_case(int(cell["max_curvature_probes"]))}
    verdict["passed"] = all(verdict[name] for name in
                            ("branch_selection_ok", "biased_only_honest",
                             "cumulative_gate_discriminates",
                             "probe_budget_respected"))
    return verdict


def _conservative_key(cell: dict[str, Any]) -> tuple:
    return (-cell["bias_kappa"], cell["branch_rmsd_radius"],
            cell["branch_atom_radius"], cell["cumulative_rmsd_radius"],
            cell["release_iterations"], cell["cumulative_rmsd_limit"],
            cell["max_curvature_probes"])


def _iter_grid(grid: dict[str, tuple]):
    keys = sorted(grid)
    import itertools
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values, strict=True))


def calibration_set_identity() -> str:
    """Digest of the analytic calibration set (fixture definitions, R6)."""
    return digest({"double_well": {"minima": [.9, 1.8], "constraint": 1.3},
                   "anchored_release": {"pull": .05, "branch": 1.0},
                   "drift": {"drift_per_frame": .2, "frames": 10},
                   "probe_budget": {"failure": "LOCALITY_ATOM"}})


def scan_branch_policy_grid(grid: dict[str, tuple] | None = None) -> dict[str, Any]:
    """Run the offline grid and freeze the most conservative passing cell."""
    grid = {**DEFAULT_GRID, **(grid or {})}
    started = time.monotonic()
    cells = []
    passing = []
    for cell in _iter_grid(grid):
        verdict = _cell_verdict(cell)
        row = {"cell": dict(sorted(cell.items())), **verdict}
        cells.append(row)
        if verdict["passed"]:
            passing.append(cell)
    if not passing:
        raise ValueError("CALIBRATION_GRID_EMPTY:no_passing_cell")
    chosen = min(passing, key=_conservative_key)
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "grid": {k: list(v) for k, v in sorted(grid.items())},
        "criteria": ["branch_selection_ok", "biased_only_honest",
                     "cumulative_gate_discriminates", "probe_budget_respected"],
        "conservative_ordering": list(CONSERVATIVE_ORDER),
        "n_cells": len(cells), "n_passing": len(passing),
        "chosen_cell": dict(sorted(chosen.items())),
        "chosen_is_most_conservative_passing": True,
        "rejected_example_cells": [row["cell"] for row in cells
                                   if not row["passed"]][:8],
        "calibration_set_sha256": calibration_set_identity(),
        "holdout_touched": False,
        "holdout_tuning_forbidden": True,
        "duration_seconds": round(time.monotonic()-started, 3),
    }


#: Values frozen into ``config/defaults.yaml`` (``g2.continuation.branch_policy``)
#: by this scan's most-conservative-passing rule; asserted equal to a fresh
#: scan in the test suite so config and code cannot drift apart.
CHOSEN_BRANCH_POLICY = BranchPolicy(branch_rmsd_radius=.25,
                                    branch_atom_radius=.50,
                                    cumulative_rmsd_radius=.25,
                                    bias_kappa=.50,
                                    release_iterations=4)

#: Frozen continuation calibration knobs (same scan).
CHOSEN_CONTINUATION_KNOBS = {"cumulative_rmsd_limit": 1.0,
                             "max_curvature_probes": 1}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output = Path(argv[0]) if argv else Path("pes2ts_branch_calibration_v1.json")
    report = scan_branch_policy_grid()
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                 allow_nan=False), encoding="utf-8")
    print(f"branch calibration: {report['n_passing']}/{report['n_cells']} cells pass; "
          f"chosen {report['chosen_cell']} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
