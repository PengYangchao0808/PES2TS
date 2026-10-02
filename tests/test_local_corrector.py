"""Scientific failure modes of local physical-gradient correction."""
import numpy as np
import pytest

from pes2ts_core.generation.planning.local_corrector import LocalCorrectorPolicy, correct_local, tangent_projector
from pes2ts_core.generation.planning.structure_checks import structure_issues


COORDS = [{"kind": "distance", "atoms": [0, 1]}]
X = np.array([[0., 0., 0.], [1., 0., 0.], [.2, 1., .3]])


def harmonic_evaluator(reference, calls):
    def evaluate(x, name):
        calls.append(x.copy())
        # Rotation/translation invariant, with a unique local angle minimum.
        v = x[2]-x[0]
        length = np.linalg.norm(v)
        target = np.linalg.norm(reference[2]-reference[0])
        error = length-target
        g = np.zeros_like(x)
        g[2] = error*v/length
        g[0] = -g[2]
        return {"success": True, "energy": .5*error**2, "gradient_hartree_per_angstrom": g}
    return evaluate


def test_local_correction_reaches_physical_stationarity_without_restraint():
    desired = X.copy()
    desired[2] *= 1.05
    calls = []
    result = correct_local(X, COORDS, [1.], [12., 12., 12.], harmonic_evaluator(desired, calls))
    assert result["success"]
    assert result["quality"]["free_gradient_max_hartree_per_angstrom"] < .001
    assert abs(np.linalg.norm(np.asarray(result["coordinates"])[0]-np.asarray(result["coordinates"])[1])-1.) < 1e-5
    assert not result["restraint_energy_added"]
    assert result["n_gradient_evaluations"] > 1


def test_no_trial_can_escape_fixed_neighbourhood_and_boundary_is_not_success():
    desired = X*3
    calls = []
    policy = LocalCorrectorPolicy(rmsd_radius=.02, atom_radius=.04, iteration_atom_step=.01,
                                  max_evaluations=70, max_iterations=50)
    result = correct_local(X, COORDS, [1.], [12., 12., 12.], harmonic_evaluator(desired, calls), policy=policy)
    from pes2ts_core.generation.planning.local_corrector import motion_evidence
    assert not result["success"]
    assert result["failure_class"] in {"LOCALITY_LIMIT", "LOCAL_ITERATION_LIMIT", "LOCAL_BUDGET_EXHAUSTED"}
    assert all(motion_evidence(x, X)["rmsd_angstrom"] < .02 for x in calls)
    assert all(motion_evidence(x, X)["maximum_atom_step_angstrom"] < .04 for x in calls)
    assert result["quality"]["free_gradient_max_hartree_per_angstrom"] > .001


def test_constraint_force_is_not_confused_with_free_force():
    x = X.copy()
    def evaluate(x, name):
        v = x[1]-x[0]
        g = np.zeros_like(x)
        g[1] = v; g[0] = -v
        return {"success": True, "energy": .5*np.sum(v*v), "gradient_hartree_per_angstrom": g}
    result = correct_local(x, COORDS, [1.], [12., 12., 12.], evaluate)
    assert result["success"] and result["n_gradient_evaluations"] == 1
    assert result["quality"]["physical_gradient_norm_hartree_per_angstrom"] > 1.
    assert not result["stationary_point_claimed"]


def test_rank_deficiency_and_failed_gradients_cannot_be_accepted():
    evaluate = harmonic_evaluator(X, [])
    result = correct_local(X, COORDS*2, [1., 1.], [12., 12., 12.], evaluate)
    assert result["failure_class"] == "DEPENDENT_CONSTRAINTS"
    result = correct_local(X, COORDS, [1.], [12., 12., 12.],
                           lambda *args: {"success": False, "failure_class": "SCF_FAILED"})
    assert result["failure_class"] == "SCF_FAILED" and not result["success"]


def test_spectator_bond_and_stereo_are_online_checks():
    plan = {"common_bond_checks": [{"atoms": [0, 1], "minimum_distance": .5, "maximum_distance": 2.}]}
    x = X.copy(); x[1] *= 4
    assert structure_issues(x, plan)[0]["reason"] == "SPECTATOR_BOND_GEOMETRY"
    tetra = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,-1.]])
    plan = {"spectator_stereo_checks": [{"atoms": [0,1,2,3], "maps": [1,2,3,4],
                                         "reference_sign": 1., "minimum_volume": .01}]}
    assert structure_issues(tetra, plan)[0]["reason"] == "SPECTATOR_STEREO_CHANGED"


@pytest.mark.parametrize("mutation", [{"rmsd_radius": 0.}, {"max_evaluations": 1.5},
                                      {"gradient_rms_tolerance": float("nan")}])
def test_local_policy_rejects_invalid_budgets(mutation):
    with pytest.raises(ValueError):
        LocalCorrectorPolicy(**mutation)


def test_tangent_projector_preserves_physical_fragment_motion():
    p, evidence = tangent_projector(X, COORDS)
    assert evidence["constraint_rank"] == 1
    assert np.allclose(p@p, p)
    # Third atom relative motion is free, rather than removed as a fragment gauge.
    displacement = np.zeros_like(X); displacement[2, 1] = 1.
    assert np.linalg.norm(p@displacement.ravel()) > .2
