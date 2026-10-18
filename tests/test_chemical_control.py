"""Chemical-control compilation tests (G2-AB3 WP-2).

Uses the REAL frozen continuation plan, copied into
``tests/fixtures/evidence/g2t_campaign_s3_RXN_0000007104_plan.json`` as the
happy-path fixture (12 atoms, distance drivers (0,10) and (2,10), active
K={0,2,10}, maps {1,3,11}, charge 0, multiplicity 1) plus typed
``ChemicalControlError`` regressions.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pes2ts_core.generation.planning.chemical_control import (
    ChemicalControlError,
    compile_chemical_control,
    control_targets_payload,
    load_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_PLAN_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "evidence"
    / "g2t_campaign_s3_RXN_0000007104_plan.json"
)


@pytest.fixture(scope="module")
def real_plan():
    return json.loads(REAL_PLAN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real_control(real_plan):
    return compile_chemical_control(real_plan, n_grid_points=3)


def test_load_plan_reads_the_real_frozen_plan(real_plan):
    plan = load_plan(REAL_PLAN_PATH)
    assert plan["schema_version"] == "pes2ts_connectivity_continuation_plan_v1"
    assert plan["reaction_id"] == "RXN_0000007104"
    assert plan["charge"] == 0
    assert plan["multiplicity"] == 1
    assert len(plan["elements"]) == 12
    assert plan == real_plan


def test_compile_happy_path_binds_real_plan_identity(real_control, real_plan):
    assert real_control.reaction_id == "RXN_0000007104"
    assert real_control.elements == tuple(real_plan["elements"])
    assert real_control.atom_map_order == tuple(real_plan["atom_map_order"])
    assert real_control.charge == 0
    assert real_control.multiplicity == 1
    assert real_control.execution_direction == "P_to_R"
    assert real_control.origin_endpoint == "P"
    assert real_control.plan_sha256 == real_plan["content_sha256"]
    assert real_control.lambda_grid == (0.0, 0.5, 1.0)
    assert real_control.lambda_grid[0] == 0.0
    assert real_control.lambda_grid[-1] == 1.0
    assert real_control.guard_atoms == ()
    assert real_control.schedule_source == "endpoint_graph_and_geometry"
    assert len(real_control.atom_order_sha256) == 64


def test_k_includes_all_driver_atoms_and_edit_maps(real_control):
    driver_atoms = {atom for driver in real_control.drivers for atom in driver.atoms}
    assert driver_atoms == {0, 2, 10}
    assert real_control.active_atom_indices == (0, 2, 10)
    driver_maps = {m for driver in real_control.drivers for m in driver.maps}
    assert driver_maps == {1, 3, 11}
    assert real_control.active_atom_maps == (1, 3, 11)
    for driver in real_control.drivers:
        for atom in driver.atoms:
            assert atom in real_control.active_atom_indices


def test_targets_at_interpolates_linearly_between_endpoints(real_control, real_plan):
    expected = tuple(
        0.5 * (driver["values"][0] + driver["values"][1])
        for driver in real_plan["drivers"]
    )
    assert real_control.targets_at(0.5) == pytest.approx(expected, abs=1e-12)
    # Endpoints reproduce the frozen plan values exactly.
    assert real_control.targets_at(0.0) == pytest.approx(
        tuple(d["values"][0] for d in real_plan["drivers"]), abs=1e-12
    )
    assert real_control.targets_at(1.0) == pytest.approx(
        tuple(d["values"][1] for d in real_plan["drivers"]), abs=1e-12
    )
    mid = real_control.targets_at(0.25)
    assert mid[0] == pytest.approx(
        real_plan["drivers"][0]["values"][0]
        + 0.25 * (real_plan["drivers"][0]["values"][1] - real_plan["drivers"][0]["values"][0]),
        abs=1e-12,
    )


def test_driver_metadata_preserved(real_control):
    assert [d.driver_id for d in real_control.drivers] == ["B_1_11", "B_3_11"]
    assert real_control.drivers[0].atoms == (0, 10)
    assert real_control.drivers[1].atoms == (2, 10)
    assert real_control.drivers[0].maps == (1, 11)
    assert real_control.drivers[1].maps == (3, 11)
    for driver in real_control.drivers:
        assert driver.kind == "distance"
        assert driver.unit == "angstrom"
        assert driver.role == "driver"
        assert driver.lambda_values == (0.0, 1.0)


def test_control_targets_payload_is_serializable_driver_rows(real_control):
    rows = control_targets_payload(real_control, 0.5)
    assert len(rows) == 2
    assert rows[0]["driver_id"] == "B_1_11"
    assert rows[0]["kind"] == "distance"
    assert rows[0]["atoms"] == [0, 10]
    assert rows[0]["unit"] == "angstrom"
    assert rows[0]["target"] == pytest.approx(real_control.targets_at(0.5)[0], abs=1e-12)


def test_reaction_id_override(real_plan):
    control = compile_chemical_control(real_plan, reaction_id="RXN_OVERRIDE", n_grid_points=2)
    assert control.reaction_id == "RXN_OVERRIDE"
    assert control.lambda_grid == (0.0, 1.0)


def test_active_edit_without_driver_is_typed_error(real_plan):
    plan = copy.deepcopy(real_plan)
    plan["active_edits"] = list(plan["active_edits"]) + [
        {"edit_kind": "formed", "maps": [6, 7]}
    ]
    with pytest.raises(ChemicalControlError, match="active edits without a driver"):
        compile_chemical_control(plan, n_grid_points=3)


def test_active_edit_maps_must_be_covered_by_driver_maps(real_plan):
    plan = copy.deepcopy(real_plan)
    # Move driver B_1_11 off edit pair (1, 11); edit (1, 11) loses its driver.
    plan["drivers"][0]["maps"] = [2, 10]
    with pytest.raises(ChemicalControlError, match="active edits without a driver"):
        compile_chemical_control(plan, n_grid_points=3)


def test_non_driver_role_is_typed_error(real_plan):
    plan = copy.deepcopy(real_plan)
    plan["drivers"][0]["role"] = "guard"
    with pytest.raises(ChemicalControlError, match="only driver-role"):
        compile_chemical_control(plan, n_grid_points=3)


@pytest.mark.parametrize("n_grid_points", [0, 1, 2.5, True])
def test_n_grid_points_must_be_int_at_least_two(real_plan, n_grid_points):
    with pytest.raises(ChemicalControlError, match="n_grid_points"):
        compile_chemical_control(real_plan, n_grid_points=n_grid_points)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: plan.__setitem__("schema_version", "pes2ts_something_else_v1"),
        lambda plan: plan.clear() or plan.__setitem__("schema_version", "nope"),
    ],
)
def test_foreign_schema_is_typed_error(real_plan, mutate):
    plan = copy.deepcopy(real_plan)
    mutate(plan)
    with pytest.raises(ChemicalControlError, match="refusing to compile foreign"):
        compile_chemical_control(plan, n_grid_points=3)


def test_missing_charge_or_multiplicity_is_typed_error(real_plan):
    plan = copy.deepcopy(real_plan)
    del plan["charge"]
    with pytest.raises(ChemicalControlError, match="charge/multiplicity"):
        compile_chemical_control(plan, n_grid_points=3)


def test_empty_elements_or_map_order_mismatch_is_typed_error(real_plan):
    plan = copy.deepcopy(real_plan)
    plan["elements"] = []
    with pytest.raises(ChemicalControlError, match="elements"):
        compile_chemical_control(plan, n_grid_points=3)
    plan = copy.deepcopy(real_plan)
    plan["atom_map_order"] = plan["atom_map_order"][:-1]
    with pytest.raises(ChemicalControlError, match="atom_map_order"):
        compile_chemical_control(plan, n_grid_points=3)


def test_driver_atom_must_be_valid_0_based_position(real_plan):
    plan = copy.deepcopy(real_plan)
    plan["drivers"][0]["atoms"] = [12, 10]  # 12 is out of range for 12 atoms
    with pytest.raises(ChemicalControlError, match="0-based position"):
        compile_chemical_control(plan, n_grid_points=3)
    plan = copy.deepcopy(real_plan)
    plan["drivers"][0]["atoms"] = [True, 10]  # bool is not an accepted index
    with pytest.raises(ChemicalControlError, match="0-based position"):
        compile_chemical_control(plan, n_grid_points=3)


def test_load_plan_rejects_foreign_and_unreadable_files(tmp_path, real_plan):
    foreign = tmp_path / "foreign.json"
    foreign.write_text(
        json.dumps({"schema_version": "not_a_plan_v1"}), encoding="utf-8"
    )
    with pytest.raises(ChemicalControlError, match="does not satisfy"):
        load_plan(foreign)
    with pytest.raises(ChemicalControlError, match="unreadable"):
        load_plan(tmp_path / "missing.json")
    good = tmp_path / "good.json"
    good.write_text(json.dumps(real_plan), encoding="utf-8")
    assert load_plan(good) == real_plan
