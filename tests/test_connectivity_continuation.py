"""Failure-injection and scope regressions for the second-round executor."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.generation.planning.connectivity import connectivity_view
from pes2ts_core.generation.planning.connectivity_plan import build_plan, choose_origin
from pes2ts_core.generation.planning.continuation import (
    ContinuationPolicy, project_geometry, run_continuation, validate_plan,
)
from pes2ts_core.generation.planning.graph_rebuild import load_endpoint_materials_from_export, rebuild_endpoint_graphs
from pes2ts_core.generation.planning.primary import select_primary_candidate
from pes2ts_core.generation.planning.synchronized_path import digest


def example(**policy):
    x = [[0., 0., 0.], [1., 0., 0.]]
    plan = {"scope": "connectivity_only", "parameter_dimension": 1, "reaction_id": "unit",
            "atom_map_order": [1, 2], "elements": ["H", "H"], "masses": [1., 1.],
            "start_geometry": x, "active_edits": [{"maps": [1, 2], "edit_kind": "broken"}],
            "drivers": [{"id": "b", "kind": "distance", "atoms": [0, 1], "maps": [1, 2],
                         "edit_kind": "broken", "lambda_values": [0., 1.], "values": [1., 2.]}],
            "guards": [], "policy": asdict(ContinuationPolicy(**policy)), "content_sha256": "unit"}
    initial = {"success": True, "converged": True, "coordinates": x, "energy": 0.}
    plan["content_sha256"] = digest({k: v for k, v in plan.items() if k != "content_sha256"})
    return plan, initial


def physical_backend(guess, targets, attempt):
    return {"success": True, "converged": True, "coordinates": guess.tolist(), "energy": float(targets[0]), "duration_seconds": .1}


def test_rejection_backtracks_to_last_accepted_state():
    plan, initial = example(initial_step=.08, max_step=.08)
    calls = []
    def backend(guess, targets, attempt):
        calls.append(targets[0])
        result = physical_backend(guess, targets, attempt)
        if len(calls) == 1:
            # Constraint residual is invalid although the backend says converged.
            result["coordinates"] = [[0, 0, 0], [8, 0, 0]]
        return result
    result = run_continuation(plan, initial, backend)
    assert result["status"] == "completed"
    assert calls[:2] == pytest.approx([1.08, 1.04])
    assert not result["attempts"][0]["accepted"]
    assert result["frames"][1]["lambda"] == pytest.approx(.04)
    assert result["attempts"][1]["parent_frame_id"] == "accepted-0000"


def test_small_global_rmsd_does_not_hide_one_mobile_atom():
    from pes2ts_core.generation.planning.continuation import geometry_quality
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))*3
    y = x.copy(); y[-1, 2] += 1.
    coords = [{"kind": "distance", "atoms": [0, 1]}]
    target = [np.linalg.norm(x[0]-x[1])]
    q = geometry_quality(x, y, coords, target, ContinuationPolicy())
    assert q["rmsd_angstrom"] < .3
    assert q["reason"] == "PATH_DISCONTINUITY"


def test_minimum_step_has_a_finite_failure():
    plan, initial = example(initial_step=.01, min_step=.005)
    result = run_continuation(plan, initial, lambda *a: {"success": False, "failure_class": "SCF_FAILED"})
    assert len(result["attempts"]) == 2
    assert len(result["frames"]) == 1
    assert result["status"] == "STEP_LIMIT:SCF_FAILED"


def test_fixed_control_uses_previous_frame_and_stops_on_rejection():
    p, initial = example(predictor_enabled=False, adaptive_enabled=False)
    guesses = []
    def failed(guess, targets, attempt):
        guesses.append(guess.tolist())
        return {"success": False, "failure_class": "SCF_FAILED"}
    result = run_continuation(p, initial, failed)
    assert guesses == [initial["coordinates"]]
    assert len(result["attempts"]) == 1
    assert result["status"] == "FIXED_STEP_FAILED:SCF_FAILED"


def test_budget_does_not_count_short_continuous_path_as_complete():
    plan, initial = example(max_frames=3)
    result = run_continuation(plan, initial, physical_backend)
    assert result["status"] == "BUDGET_EXHAUSTED"
    assert not result["completed_interval"]


@pytest.mark.parametrize("energy", [None, float("nan"), float("inf")])
def test_nonfinite_energy_cannot_be_accepted(energy):
    plan, initial = example()
    def backend(*args):
        return {**physical_backend(*args), "energy": energy}
    result = run_continuation(plan, initial, backend)
    assert result["status"] == "NONFINITE_ENERGY"
    assert len(result["frames"]) == 1


def test_final_remainder_below_minimum_step_can_finish():
    plan, initial = example(initial_step=.3, max_step=.3, min_step=.1)
    result = run_continuation(plan, initial, physical_backend)
    assert result["completed_interval"]
    assert result["frames"][-1]["lambda"] == 1.


def test_nonlinear_projection_reaches_both_targets():
    coords = [{"kind": "distance", "atoms": [0, 1]}, {"kind": "distance", "atoms": [0, 2]}]
    x = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    y, evidence = project_geometry(x, coords, [1.1, 1.2], [12., 1., 12.], ContinuationPolicy())
    assert evidence["reason"] is None
    assert [np.linalg.norm(y[0]-y[i]) for i in (1, 2)] == pytest.approx([1.1, 1.2], abs=1e-5)


def test_contradictory_constraints_are_not_hidden_by_pseudoinverse():
    coords = [{"kind": "distance", "atoms": [0, 1]}]*2
    y, evidence = project_geometry(np.array([[0., 0., 0.], [1., 0., 0.]]), coords, [1., 2.], [1., 1.], ContinuationPolicy())
    assert y is None and evidence["reason"] == "CONSTRAINT_INFEASIBLE"


@pytest.mark.parametrize("mutation", ["order", "extra", "missing", "index"])
def test_scope_rejects_order_changes_extra_pairs_and_mapping_errors(mutation):
    p, _ = example()
    if mutation == "order": p["drivers"][0]["edit_kind"] = "order_changed"
    elif mutation == "extra": p["drivers"].append(copy.deepcopy(p["drivers"][0]))
    elif mutation == "missing": p["drivers"] = []
    else: p["drivers"][0]["maps"] = [2, 1]
    with pytest.raises(ValueError): validate_plan(p)


def test_distance_guard_cannot_smuggle_order_change_back_in():
    p, _ = example()
    g = copy.deepcopy(p["drivers"][0]); g["id"] = "g"
    p["guards"] = [g]
    with pytest.raises(ValueError, match="DISTANCE_GUARD"): validate_plan(p)


def test_all_demo24_plans_have_exact_fb_scope_and_archived_derivatives():
    counts = {"formed": 0, "broken": 0, "order_changed": 0}
    paths = sorted((Path(__file__).parent/"fixtures/p0_demo24/records").glob("RXN_*.json"))
    assert len(paths) == 24
    for path in paths:
        s = json.loads(path.read_text(encoding="utf-8"))
        b = rebuild_endpoint_graphs(s["reaction_smiles"], load_endpoint_materials_from_export(s))
        graph = build_reaction_edit_graph(b)
        view = connectivity_view(graph)
        assert all(e.edit_kind != "order_changed" for e in view.edits)
        for e in graph.edits: counts[e.edit_kind] += 1
        origin = choose_origin(s, b)
        if origin["status"] != "ready":
            assert s["reaction_id"] == "RXN_0000155302"
            continue
        plan = build_plan(s, b, origin, s["r_coordinates" if origin["side"] == "R" else "p_coordinates"])
        assert {tuple(d["maps"]) for d in plan["drivers"]} == {e.pair for e in view.edits}
        assert len(plan["derived_changes_archive"]) == sum(e.edit_kind == "order_changed" for e in graph.edits)
    assert counts == {"formed": 40, "broken": 31, "order_changed": 44}


def test_primary_promotes_a_single_fb_driver_without_demanding_order_changes():
    p = Path(__file__).parent/"fixtures/p0_demo24/records/RXN_0000026256.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    b = rebuild_endpoint_graphs(s["reaction_smiles"], load_endpoint_materials_from_export(s))
    view = connectivity_view(build_reaction_edit_graph(b))
    assert len(view.edits) == 1
    drivers = [{"kind": "B", "maps": list(view.edits[0].pair)}]
    result = select_primary_candidate([{"candidate_id": "fb", "drivers": drivers, "failure_reasons": []}], b)
    assert result["primary"]["drivers"] == drivers


def test_phase_schedule_has_exact_endpoints_and_fb_only_scope():
    p = Path(__file__).parent/"fixtures/p0_demo24/records/RXN_0000007104.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    b = rebuild_endpoint_graphs(s["reaction_smiles"], load_endpoint_materials_from_export(s))
    origin = choose_origin(s, b)
    x = s["r_coordinates" if origin["side"] == "R" else "p_coordinates"]
    linear = build_plan(s, b, origin, x)
    phased = build_plan(s, b, origin, x, phase_profile=[.4, .6])
    for a, d in zip(linear["drivers"], phased["drivers"]):
        assert d["values"][0] == pytest.approx(a["values"][0])
        assert d["values"][-1] == pytest.approx(a["values"][-1])
        assert len(d["values"]) == 101
        assert d["edit_kind"] in {"formed", "broken"}


def test_redundant_guard_is_rejected_before_execution():
    # Identical angle rows are not independent even though each is local.
    p, _ = example()
    p["atom_map_order"] = [1, 2, 3]
    p["elements"] = ["C", "C", "H"]
    p["masses"] = [12., 12., 1.]
    p["start_geometry"] = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]]
    p["local_support_maps"] = [3]
    g = {"id": "a", "kind": "angle", "maps": [1, 2, 3], "atoms": [0, 1, 2], "lambda_values": [0., 1.], "values": [90., 90.]}
    p["guards"] = [g, {**g, "id": "duplicate_angle"}]
    with pytest.raises(ValueError, match="DEPENDENT_OR_SINGULAR_GUARD"):
        validate_plan(p)


def test_frozen_plan_tampering_is_rejected():
    p, initial = example()
    p["drivers"][0]["values"][-1] = 3.
    with pytest.raises(ValueError, match="FROZEN_PLAN_HASH_MISMATCH"):
        run_continuation(p, initial, physical_backend)


def test_missing_fb_cannot_fall_back_to_incomplete_legacy_primary():
    path = Path(__file__).parent/"fixtures/p0_demo24/records/RXN_0000007104.json"
    s = json.loads(path.read_text(encoding="utf-8"))
    b = rebuild_endpoint_graphs(s["reaction_smiles"], load_endpoint_materials_from_export(s))
    view = connectivity_view(build_reaction_edit_graph(b))
    c = {"candidate_id": "incomplete", "drivers": [{"kind": "B", "maps": list(view.edits[0].pair)}]}
    result = select_primary_candidate([c], b, connectivity_only=True)
    assert result["primary"] is None
    assert result["reason"] == "INCOMPLETE_CONNECTIVITY_DRIVER_COVERAGE"
