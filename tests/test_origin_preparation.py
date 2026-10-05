"""Origin preparation repairs and the versioned endpoint-method evidence (WP-3)."""
import numpy as np
import pytest

from pes2ts_core.generation.planning.origin_preparation import (
    ENDPOINT_METHOD_EVIDENCE_SCHEMA, apply_input_repair, prepare_origin,
    repair_component_overlap, torsion_hypotheses,
)

# H2-like fixture: atoms 0-1 bonded, spectator H (atom 2) bonded to atom 0.
ELEMENTS = ["H", "H", "H"]
BONDS = [[0, 1], [0, 2]]
X = np.array([[0., 0., 0.], [0.74, 0., 0.], [0., 0.62, 0.]])


def test_component_repair_preserves_each_component_internal_coordinates():
    x=np.array([[0.,0.,0.],[1.,0.,0.],[.3,0.,0.]])
    y,evidence=repair_component_overlap(x,["C","H","Cl"],[[0,1],[2]])
    assert evidence["internal_geometry_preserved"]
    assert np.allclose(y[1]-y[0],x[1]-x[0])
    assert np.linalg.norm(y[0]-y[2])>2.


def test_torsion_hypothesis_preserves_bonds_and_does_not_rotate_ring():
    x=np.array([[0.,1.,0.],[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]])
    bonds=[[0,1],[1,2],[2,3]]
    rows=torsion_hypotheses(x,bonds)
    assert len(rows)==2
    for y,record in rows:
        assert all(np.isclose(np.linalg.norm(y[a]-y[b]),np.linalg.norm(x[a]-x[b])) for a,b in bonds)
    assert not torsion_hypotheses(x,bonds+[[0,3]])


def identity_preserving_evaluator():
    def evaluate_free(geometry, evaluation_id):
        return {"success": True, "failure_class": None,
                "coordinates": np.asarray(geometry, float).tolist(),
                "energy": -1., "gradient_hartree_per_angstrom": None,
                "duration_seconds": .01, "acp_receipt_ref": "acp:origin-1"}
    return evaluate_free


def test_successful_preparation_emits_endpoint_method_evidence():
    evidence = prepare_origin(X, ELEMENTS, BONDS, identity_preserving_evaluator(),
                              method_context={"method": "GFN2-xTB"})
    assert evidence["schema_version"] == ENDPOINT_METHOD_EVIDENCE_SCHEMA
    assert evidence["status"] == "prepared"
    assert evidence["stationarity_status"] == "not_certified"
    assert evidence["minimum_status"] == "unknown"
    assert evidence["identity_status"] == "connection_preserved"
    assert evidence["frequency_ref"] == {"value": None,
                                         "reason": "frequency_not_computed_by_origin_preparation"}
    assert evidence["acp_receipt_ref"] == "acp:origin-1"
    assert evidence["prepared_geometry_hash"]


def test_free_optimization_that_breaks_a_bond_is_typed_connection_lost():
    # Free optimization pulls atom 1 away until the original 0-1 bond is gone
    # (the RXN_0000079731 N8-H16 pattern: converged, but connection lost).
    broken = X.copy()
    broken[1] = [2.8, 0., 0.]
    def evaluator(geometry, evaluation_id):
        return {"success": True, "failure_class": None, "coordinates": broken.tolist(),
                "energy": -1.1, "duration_seconds": .01}
    evidence = prepare_origin(X, ELEMENTS, BONDS, evaluator)
    assert evidence["status"] == "failed"
    assert evidence["failure_code"] == "ORIGIN_CONNECTION_LOST"
    lost = evidence["failure_detail"]["lost_connections"]
    assert lost and lost[0]["atoms"] == [0, 1]
    assert lost[0]["distance_before_angstrom"] == pytest.approx(.74, abs=.02)
    assert lost[0]["distance_after_angstrom"] == pytest.approx(2.8, abs=.01)
    assert evidence["failure_detail"]["source_geometry_hash"] == evidence["source_geometry_hash"]
    assert evidence["minimum_status"] == "unknown"


def test_overlapped_input_is_a_terminal_invalid_origin():
    overlapped = X.copy()
    overlapped[2] = [0.74, 0.05, 0.]
    def evaluator(geometry, evaluation_id):
        raise AssertionError("an overlapped input must never reach the optimizer")
    evidence = prepare_origin(overlapped, ELEMENTS, BONDS, evaluator)
    assert evidence["status"] == "invalid_input"
    assert evidence["failure_code"] == "ORIGIN_OVERLAP_INPUT"
    assert evidence["failure_detail"]["collisions"]
    assert evidence["prepared_geometry_hash"] is None
    # A bonded pair squeezed to near-coincidence is overlap input as well.
    collapsed = X.copy()
    collapsed[2] = [0., 0.02, 0.]
    evidence = prepare_origin(collapsed, ELEMENTS, BONDS, evaluator)
    assert evidence["failure_code"] == "ORIGIN_OVERLAP_INPUT"
    assert evidence["failure_detail"]["collapsed_bonds"]


def test_scf_failure_and_budget_exhaustion_are_typed():
    def scf(geometry, evaluation_id):
        return {"success": False, "failure_class": "SCF_NOT_CONVERGED", "duration_seconds": 1.}
    evidence = prepare_origin(X, ELEMENTS, BONDS, scf)
    assert evidence["failure_code"] == "ORIGIN_SCF_BRANCH"

    def slow(geometry, evaluation_id):
        return {"success": True, "failure_class": None, "coordinates": np.asarray(geometry).tolist(),
                "energy": -1., "duration_seconds": 30.}
    evidence = prepare_origin(X, ELEMENTS, BONDS, slow, budget_seconds=10.)
    assert evidence["failure_code"] == "ORIGIN_BUDGET_EXHAUSTED"
    assert evidence["failure_detail"]["observed_seconds"] > evidence["failure_detail"]["allowed_seconds"]


def test_repair_produces_new_identity_and_never_overwrites_the_endpoint():
    x = np.array([[0., 0., 0.], [1., 0., 0.], [.3, 0., 0.]])
    repaired, repair_evidence = repair_component_overlap(x, ["C", "H", "Cl"], [[0, 1], [2]])
    record = apply_input_repair(x, repaired, "component_overlap_separation", repair_evidence)
    provenance = record["provenance"]
    assert provenance["source_geometry_hash"] != provenance["repaired_geometry_hash"]
    assert provenance["original_geometry_unchanged"] is True
    assert np.allclose(x, np.array([[0., 0., 0.], [1., 0., 0.], [.3, 0., 0.]]))
    assert provenance["repair_kind"] == "component_overlap_separation"
    # A repaired input is a NEW origin candidate, not the original endpoint:
    # it must pass the same screens under its own identity.
    assert np.linalg.norm(np.asarray(record["geometry"])[0]-np.asarray(record["geometry"])[2]) > 2.
