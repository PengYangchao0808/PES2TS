import copy
from dataclasses import asdict
import numpy as np
import pytest

from pes2ts_core.contracts import validate_document
from pes2ts_core.generation.planning.candidate_selection import (
    GRADIENT_SEED_PROPOSALS_SCHEMA, bind_candidate, continuation_path_bundle,
    rank_continuation_candidates, seed_proposal_from_gradient_ranking,
)
from pes2ts_core.generation.planning.continuation import ContinuationPolicy, geometry_quality, run_continuation
from pes2ts_core.generation.planning.synchronized_path import digest
from test_connectivity_continuation import example, physical_backend


def path():
    plan, initial = example(max_frames=8)
    result = run_continuation(plan, initial, physical_backend)
    for i,frame in enumerate(result["frames"]):
        frame["energy_hartree"] = -(i-3.)**2
        frame["physical_gradient_status"] = "bound"
        frame["physical_gradient_hartree_per_angstrom"] = [[-.01*(i+1), 0, 0], [.01*(i+1), 0, 0]]
    return plan, result


def test_partial_prefix_has_hash_bound_gradient_candidates():
    plan, result = path()
    ranking = rank_continuation_candidates(result, plan)
    assert not result["completed_interval"]
    assert ranking["candidates"]
    assert ranking["candidates"][0]["frame_index"] == 2
    assert ranking["candidates"][0]["geometry_sha256"] == digest(result["frames"][2]["geometry"])
    assert not ranking["reference_geometry_used"]
    assert not ranking["candidates"][0]["reaction_connection_verified"]


def test_legacy_gradient_frame_key_is_normalized_and_tampering_rejected():
    plan, result = path()
    candidate = bind_candidate({"selected_frame": 3}, result, plan)
    assert candidate["frame_index"] == 3
    altered = copy.deepcopy(result)
    altered["frames"][3]["geometry"][0][0] += .1
    with pytest.raises(ValueError, match="CANDIDATE_GEOMETRY"):
        bind_candidate(candidate, altered, plan)


def test_missing_gradient_is_not_silently_ranked_as_zero():
    plan, result = path()
    for frame in result["frames"]:
        frame.pop("physical_gradient_hartree_per_angstrom", None)
    assert not rank_continuation_candidates(result, plan)["candidates"]


def test_online_gradient_gate_rejects_unconverged_and_missing_force():
    plan, initial = example(require_physical_gradient=True, initial_step=.005, min_step=.005)
    result = run_continuation(plan, initial, physical_backend)
    assert result["status"] == "STEP_LIMIT:PHYSICAL_GRADIENT_MISSING"
    # A third atom introduces a genuine unconstrained deformation.
    x = [[0.,0.,0.], [1.,0.,0.], [.3,1.,.4]]
    plan["elements"] += ["H"]; plan["atom_map_order"] += [3]; plan["masses"] += [1.]
    plan["start_geometry"] = x; initial["coordinates"] = x
    plan["content_sha256"] = digest({k:v for k,v in plan.items() if k != "content_sha256"})
    def backend(*args):
        record = physical_backend(*args)
        record.update(physical_gradient_status="bound", physical_gradient_hartree_per_angstrom=[[0.,0.,0.], [0.,0.,0.], [0.,1.,0.]])
        return record
    result = run_continuation(plan, initial, backend)
    assert result["status"] == "STEP_LIMIT:FREE_GRADIENT_NOT_CONVERGED"


def test_small_target_increment_requires_residual_below_increment_scale():
    previous = np.array([[0., 0., 0.], [1., 0., 0.]])
    x = np.array([[0., 0., 0.], [1.002, 0., 0.]])
    coordinates = [{"kind":"distance", "atoms":[0,1]}]
    policy = ContinuationPolicy(relative_constraint_tolerance=.2)
    quality = geometry_quality(previous, x, coordinates, [1.001], policy, [1.])
    assert quality["effective_constraint_tolerances"] == pytest.approx([.0002])
    assert quality["reason"] == "CONSTRAINT_RESIDUAL"


def test_every_candidate_carries_the_canonical_seed_proposal_fields():
    plan, result = path()
    ranking = rank_continuation_candidates(result, plan)
    assert ranking["schema_version"] == GRADIENT_SEED_PROPOSALS_SCHEMA
    for candidate in ranking["candidates"]:
        assert candidate["evidence_class"]
        assert candidate["physical_gradient_norm_hartree_per_bohr"] > 0
        assert candidate["free_gradient_norm_hartree_per_bohr"] >= 0
        assert "reconstructed_energy_slope_hartree_per_lambda" in candidate
        assert candidate["plan_sha256"] == plan["content_sha256"]
        assert candidate["source_path_sha256"] == digest(result)
        assert candidate["completed_interval"] is False
        assert candidate["reference_geometry_used"] is False
        assert candidate["stationary_point_verified"] is False
        assert candidate["reaction_connection_verified"] is False


def test_partial_continuation_projects_into_usable_bundle_with_report_only_interval():
    plan, result = path()
    bundle = continuation_path_bundle(result, plan, case_id="case:unit")
    assert not validate_document(bundle)
    assert bundle["status"] == "usable"
    projection = bundle["extensions"]["pes2ts.continuation_projection.v1"]
    assert projection["completed_interval"] is False
    assert projection["completed_interval_is_report_only"] is True
    assert projection["path_integrity_reported_separately"] is True
    assert projection["plan_sha256"] == plan["content_sha256"]


def test_gradient_ranking_converts_into_a_formal_accepted_seed_proposal():
    plan, result = path()
    ranking = rank_continuation_candidates(result, plan)
    bundle = continuation_path_bundle(result, plan, case_id="case:unit")
    proposal = seed_proposal_from_gradient_ranking(bundle, ranking)
    assert not validate_document(proposal)
    assert proposal["status"] == "accepted"
    assert proposal["selection_source"] == "ranking"
    selected = proposal["selected_frames"][0]
    assert selected["geometry_sha256"] == ranking["candidates"][0]["geometry_sha256"]
    assert selected["completed_interval"] is False
    assert selected["reference_geometry_used"] is False
    extension = proposal["extensions"]["pes2ts.gradient_seed_proposals.v1"]
    assert extension["preparation_layer_status"] == "absent"


def test_conversion_rejects_foreign_geometry_and_missing_candidates():
    plan, result = path()
    ranking = rank_continuation_candidates(result, plan)
    bundle = continuation_path_bundle(result, plan, case_id="case:unit")
    forged = copy.deepcopy(ranking)
    forged["candidates"][0]["geometry_sha256"] = "0"*64
    with pytest.raises(ValueError, match="CANDIDATE_GEOMETRY_NOT_IN_PATH_BUNDLE"):
        seed_proposal_from_gradient_ranking(bundle, forged)
    with pytest.raises(ValueError, match="NO_BOUND_CANDIDATE"):
        seed_proposal_from_gradient_ranking(bundle, ranking, rank=99)
    with pytest.raises(ValueError, match="INVALID_GRADIENT_SEED_PROPOSALS_SCHEMA"):
        seed_proposal_from_gradient_ranking(bundle, {"schema_version": "ranking.json"})
