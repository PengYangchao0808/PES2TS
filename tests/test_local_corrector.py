"""Scientific failure modes of local physical-gradient correction."""
import re
import numpy as np
import pytest

from pes2ts_core.generation.planning.local_corrector import (
    LocalCorrectorPolicy, correct_local, evaluation_content_digest, tangent_projector,
)
from pes2ts_core.generation.planning.structure_checks import structure_issues


COORDS = [{"kind": "distance", "atoms": [0, 1]}]
X = np.array([[0., 0., 0.], [1., 0., 0.], [.2, 1., .3]])
EVALUATION_ID_PATTERN = re.compile(r"trial-\d{4}/eval-[0-9a-f]{16}")


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
    # G2-AB1 WP-2: the old LOCALITY_LIMIT is split into typed LOCALITY_* codes.
    assert result["failure_class"].startswith(("LOCALITY", "LOCAL_"))
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
    assert result["failure_class"] == "LOCALITY_RANK:DEPENDENT_CONSTRAINTS"
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


def recording_evaluator(reference, ids, calls):
    evaluate = harmonic_evaluator(reference, calls)

    def wrapped(x, evaluation_id):
        ids.append(evaluation_id)
        return evaluate(x, evaluation_id)
    return wrapped


def test_evaluation_ids_are_content_addressed_and_replay_stable():
    desired = X.copy()
    desired[2] *= 1.05
    ids_first, ids_replay, calls = [], [], []
    correct_local(X, COORDS, [1.], [12., 12., 12.],
                  recording_evaluator(desired, ids_first, calls))
    correct_local(X, COORDS, [1.], [12., 12., 12.],
                  recording_evaluator(desired, ids_replay, calls))
    assert ids_first
    assert ids_first == ids_replay
    assert all(EVALUATION_ID_PATTERN.fullmatch(i) for i in ids_first)


def test_changed_content_gets_a_fresh_evaluation_address_instead_of_a_collision():
    desired = X.copy()
    desired[2] *= 1.05
    ids_a, ids_b, calls = [], [], []
    correct_local(X, COORDS, [1.], [12., 12., 12.],
                  recording_evaluator(desired, ids_a, calls))
    shifted = X.copy(); shifted[2, 0] += .1
    correct_local(shifted, COORDS, [1.], [12., 12., 12.],
                  recording_evaluator(desired, ids_b, calls))
    assert ids_a[0].split("/")[0] == ids_b[0].split("/")[0] == "trial-0000"
    assert ids_a[0].split("/")[1] != ids_b[0].split("/")[1]


def test_method_context_and_operation_change_the_content_digest():
    base = evaluation_content_digest(X, COORDS, [1.])
    assert len(base) == 16
    assert evaluation_content_digest(X, COORDS, [1.]) == base
    assert evaluation_content_digest(X, COORDS, [1.], {"method": "HF-3c"}) != base
    assert evaluation_content_digest(X, COORDS, [1.], operation="landing_free_opt") != base
    assert evaluation_content_digest(X*1.01, COORDS, [1.]) != base
    assert evaluation_content_digest(X, COORDS, [1.2]) != base


def test_typed_locality_codes_split_step_atom_branch_and_cumulative():
    from pes2ts_core.generation.planning.local_corrector import BranchPolicy
    far = X*3
    loose = dict(rmsd_radius=5., atom_radius=5., iteration_atom_step=.05,
                 max_evaluations=40, max_iterations=30)
    # Cumulative gate tightest: motion away from the correction start is typed
    # LOCALITY_CUMULATIVE even though predictor radii are generous.
    result = correct_local(X, COORDS, [1.], [12., 12., 12.], harmonic_evaluator(far, []),
                           policy=LocalCorrectorPolicy(**loose),
                           branch_policy=BranchPolicy(cumulative_rmsd_radius=.05))
    assert result["failure_class"] == "LOCALITY_CUMULATIVE"
    # Predictor per-atom gate tightest -> LOCALITY_ATOM.
    result = correct_local(X, COORDS, [1.], [12., 12., 12.], harmonic_evaluator(far, []),
                           policy=LocalCorrectorPolicy(rmsd_radius=5., atom_radius=.03,
                                                       iteration_atom_step=.02,
                                                       max_evaluations=40, max_iterations=30))
    assert result["failure_class"] == "LOCALITY_ATOM"
    # Predictor aggregate RMSD gate tightest -> LOCALITY_STEP.
    result = correct_local(X, COORDS, [1.], [12., 12., 12.], harmonic_evaluator(far, []),
                           policy=LocalCorrectorPolicy(rmsd_radius=.01, atom_radius=5.,
                                                       iteration_atom_step=.02,
                                                       max_evaluations=40, max_iterations=30))
    assert result["failure_class"] == "LOCALITY_STEP"
    # Branch gate tightest -> LOCALITY_BRANCH.
    result = correct_local(X, COORDS, [1.], [12., 12., 12.], harmonic_evaluator(far, []),
                           policy=LocalCorrectorPolicy(**loose),
                           branch_reference=X.copy(),
                           branch_policy=BranchPolicy(branch_rmsd_radius=.02,
                                                      branch_atom_radius=.02,
                                                      cumulative_rmsd_radius=5.,
                                                      bias_kappa=.01, release_iterations=2))
    assert result["failure_class"] == "LOCALITY_BRANCH"
