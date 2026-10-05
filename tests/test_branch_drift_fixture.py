"""G2-AB1 WP-2 branch-stability fixtures on analytic PESs.

Two documented failure modes of the old corrector (B1-2):

1. branch selection — with two minima under the SAME constraint distance, the
   correction must follow the previous accepted branch, not jump basins;
2. honest release — a frame stationary only under the numeric anchor must be
   reported ``biased_only`` with released raw free-gradient evidence that
   exceeds tolerance; the electronic energy never carries a restraint.
"""
from __future__ import annotations

import numpy as np
import pytest

from pes2ts_core.generation.planning.local_corrector import (
    BranchPolicy, LocalCorrectorPolicy, correct_local,
)

COORDS = [{"kind": "distance", "atoms": [0, 1]}]
MASSES = [12., 12., 1.]


def frame(d):
    # Collinear: the printed atom-2 coordinate IS the true 0-2 distance.
    return np.array([[0., 0., 0.], [1.3, 0., 0.], [d, 0., 0.]])


def double_well_evaluator(calls):
    m1, m2 = .9, 1.8

    def evaluate(x, evaluation_id):
        calls.append(np.asarray(x, float).copy())
        u = x[2]-x[0]
        d = float(np.linalg.norm(u))
        unit = u/d
        ded = 2.*(d-m1)*(d-m2)*(2.*d-m1-m2)
        g = np.zeros_like(x)
        g[2] = ded*unit
        g[0] = -g[2]
        return {"success": True, "energy": (d-m1)**2*(d-m2)**2,
                "gradient_hartree_per_angstrom": g}
    return evaluate


def constant_pull_evaluator(strength, calls):
    def evaluate(x, evaluation_id):
        calls.append(np.asarray(x, float).copy())
        u = x[2]-x[0]
        unit = u/float(np.linalg.norm(u))
        g = np.zeros_like(x)
        g[2] = -strength*unit
        g[0] = -g[2]
        return {"success": True, "energy": -strength*float(np.linalg.norm(u)),
                "gradient_hartree_per_angstrom": g}
    return evaluate


def generous_policy(**overrides):
    base = dict(rmsd_radius=1.0, atom_radius=1.2, iteration_atom_step=.08,
                max_evaluations=200, max_iterations=120)
    base.update(overrides)
    return LocalCorrectorPolicy(**base)


def test_two_minima_same_constraint_follows_previous_branch():
    calls = []
    branch = frame(.9)
    # Guess sits past the barrier on the far basin's side: raw physical
    # descent alone would slide to d~1.8; the frozen branch anchor pulls the
    # correction back onto the previous accepted branch.
    guess = frame(1.38)
    result = correct_local(guess, COORDS, [1.3], MASSES, double_well_evaluator(calls),
                           policy=generous_policy(),
                           branch_reference=branch,
                           branch_policy=BranchPolicy(branch_rmsd_radius=.35,
                                                      branch_atom_radius=.5,
                                                      cumulative_rmsd_radius=.45,
                                                      bias_kappa=.5, release_iterations=10))
    assert result["success"], result["failure_class"]
    assert result["convergence_mode"] == "physical"
    final = np.asarray(result["coordinates"])
    final_d = float(np.linalg.norm(final[2]-final[0]))
    assert final_d == pytest.approx(.9, abs=.05)
    assert final_d < 1.2
    # Every evaluated trial stayed inside the branch radius of the reference.
    for x in calls:
        assert np.linalg.norm(x[2]-x[0]) < .9+.5+1e-9


def test_anchor_only_stationarity_is_reported_biased_only_with_released_evidence():
    calls = []
    branch = frame(1.0)
    guess = frame(1.05)
    result = correct_local(guess, COORDS, [1.3], MASSES, constant_pull_evaluator(.05, calls),
                           policy=generous_policy(gradient_rms_tolerance=1e-4,
                                                  gradient_max_tolerance=3e-4),
                           branch_reference=branch,
                           branch_policy=BranchPolicy(branch_rmsd_radius=.3,
                                                      branch_atom_radius=.5,
                                                      cumulative_rmsd_radius=.45,
                                                      bias_kappa=.5, release_iterations=12))
    assert result["success"], result["failure_class"]
    assert result["convergence_mode"] == "biased_only"
    assert result["numeric_anchor_active"] is True
    assert not result["restraint_energy_added"]
    released = result["released_evidence"]
    assert released is not None
    # The released raw physical free gradient still exceeds tolerance.
    assert released["free_gradient_max_hartree_per_angstrom"] > 3e-4
    assert result["quality"]["free_gradient_max_hartree_per_angstrom"] > 3e-4
    assert result["stationary_point_claimed"] is False


def test_biased_only_frame_stays_inside_branch_radius():
    calls = []
    branch = frame(1.0)
    guess = frame(1.05)
    result = correct_local(guess, COORDS, [1.3], MASSES, constant_pull_evaluator(.05, calls),
                           policy=generous_policy(),
                           branch_reference=branch,
                           branch_policy=BranchPolicy(branch_rmsd_radius=.3,
                                                      branch_atom_radius=.5,
                                                      cumulative_rmsd_radius=.45,
                                                      bias_kappa=.5, release_iterations=12))
    assert result["convergence_mode"] == "biased_only"
    from pes2ts_core.generation.planning.local_corrector import motion_evidence
    for x in calls:
        motion = motion_evidence(x, branch)
        assert motion["maximum_atom_step_angstrom"] < .5+1e-9
        assert motion["rmsd_angstrom"] < .3+1e-9


def test_branch_radius_violation_is_typed_locality_branch_failure():
    calls = []
    branch = frame(1.0)
    # Strong pull, weak anchor: every downhill trial leaves the branch radius.
    result = correct_local(frame(1.05), COORDS, [1.3], MASSES,
                           constant_pull_evaluator(.5, calls),
                           policy=generous_policy(),
                           branch_reference=branch,
                           branch_policy=BranchPolicy(branch_rmsd_radius=.05,
                                                      branch_atom_radius=.08,
                                                      cumulative_rmsd_radius=.4,
                                                      bias_kappa=.02, release_iterations=4))
    assert not result["success"]
    assert result["failure_class"].startswith("LOCALITY")
    assert result["convergence_mode"] is None


def test_no_anchor_without_branch_reference_keeps_legacy_behaviour():
    calls = []
    # No branch reference: no anchor, biased_only unreachable, physical only.
    result = correct_local(frame(1.0), COORDS, [1.3], MASSES,
                           double_well_evaluator(calls), policy=generous_policy())
    assert result["success"]
    assert result["convergence_mode"] == "physical"
    assert result["numeric_anchor_active"] is False
    assert result["released_evidence"] is None
    assert result["branch_locality_reference"] is None
