"""Frozen ACP ``XtbConstrainedOpt`` request-contract tests (G2-AB3 WP-4).

Offline tests for ``build_constraint_request`` / ``validate_constraint_request``
/ ``request_sha256``: 0-based atom positions, unit enforcement, driver-role
separation, ``fix_atoms`` validation, provenance digests, and
geometry/elements length binding.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from pes2ts_core.integration.acp.xtb_constraints_request import (
    XTC_REQUEST_SCHEMA,
    ConstraintRequestError,
    DriverTarget,
    atom_order_digest,
    build_constraint_request,
    geometry_digest,
    request_sha256,
    validate_constraint_request,
)

ELEMENTS = ["C", "H", "H"]
GEOMETRY = [[0.0, 0.0, 0.0], [2.03, 0.0, 0.0], [3.5, 0.0, 0.0]]

# Real-plan-shaped 12-atom order (map 11 -> 0-based position 10).
PLAN_ELEMENTS = ["N", "C", "N", "C", "C", "Cl", "Cl", "Cl", "N", "O", "H", "H"]
PLAN_GEOMETRY = [[float(i), 0.0, 0.0] for i in range(12)]


def _distance_target(target=2.0, **overrides) -> DriverTarget:
    payload = {"driver_id": "d0", "kind": "distance", "atoms": (0, 1),
               "target": target, "unit": "angstrom", "role": "driver"}
    payload.update(overrides)
    return DriverTarget(**payload)


def _build(elements=ELEMENTS, geometry=GEOMETRY, targets=None, *,
           reaction_id="RXN_TEST", **overrides):
    return build_constraint_request(
        geometry=geometry,
        elements=elements,
        charge=0,
        multiplicity=1,
        targets=targets if targets is not None else [_distance_target()],
        reaction_id=reaction_id,
        point_id="lam0.5",
        **overrides,
    )


def _canonical_sha256(payload) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_happy_path_request_schema_and_zero_based_atoms():
    request = _build()
    assert request["schema_version"] == XTC_REQUEST_SCHEMA
    assert request["reaction_id"] == "RXN_TEST"
    assert request["source"]["elements"] == ELEMENTS
    assert request["source"]["charge"] == 0
    assert request["source"]["multiplicity"] == 1
    constraint = request["recipe"]["constraints"][0]
    # 0-based positions in the frozen atom order, stored verbatim.
    assert constraint["atoms"] == [0, 1]
    assert constraint["kind"] == "distance"
    assert constraint["unit"] == "angstrom"
    assert constraint["target"] == 2.0
    assert request["recipe"]["gfn_level"] == 2
    assert request["recipe"]["force_constant"] == 50.0
    assert request["recipe"]["fix_atoms"] is None
    validate_constraint_request(request)


def test_request_sha256_binds_payload_provenance():
    request = _build()
    provenance = request["provenance"]
    assert set(provenance) >= {
        "point_id", "atom_order_sha256", "seed_geometry_sha256", "request_sha256",
    }
    assert provenance["point_id"] == "lam0.5"
    assert provenance["atom_order_sha256"] == atom_order_digest(ELEMENTS, 3)
    assert provenance["seed_geometry_sha256"] == geometry_digest(GEOMETRY, ELEMENTS)
    # The digest binds the payload WITHOUT provenance (freeze-at-build).
    frozen = {key: value for key, value in request.items() if key != "provenance"}
    assert provenance["request_sha256"] == _canonical_sha256(frozen)
    assert request_sha256(request) == provenance["request_sha256"]
    # Deterministic rebuild -> same digest; any payload change -> new digest.
    assert request_sha256(_build()) == request_sha256(request)
    assert request_sha256(_build(force_constant=25.0)) != request_sha256(request)
    assert request_sha256(_build(targets=[_distance_target(2.5)])) != request_sha256(request)


def test_request_sha256_missing_provenance_raises():
    with pytest.raises(ConstraintRequestError, match="provenance"):
        request_sha256({"schema_version": XTC_REQUEST_SCHEMA})


def test_distance_unit_must_be_angstrom():
    with pytest.raises(ConstraintRequestError, match="requires unit"):
        _build(targets=[_distance_target(unit="bohr")])
    with pytest.raises(ConstraintRequestError, match="requires unit"):
        _build(targets=[_distance_target(kind="angle", atoms=(0, 1, 2),
                                         unit="angstrom")])


def test_angle_and_dihedral_units_are_degree():
    angle = DriverTarget("a0", "angle", (0, 1, 2), 109.5, "degree")
    request = _build(targets=[angle])
    assert request["recipe"]["constraints"][0]["unit"] == "degree"
    validate_constraint_request(request)
    dihedral = DriverTarget("t0", "dihedral", (0, 1, 2, 0), 180.0, "degree")
    with pytest.raises(ConstraintRequestError, match="distinct atoms"):
        _build(targets=[dihedral])


def test_only_driver_role_may_compile_to_potentials():
    with pytest.raises(ConstraintRequestError, match="role='driver'"):
        _build(targets=[_distance_target(role="guard")])


def test_fix_atoms_are_sorted_deduped_and_range_checked():
    request = _build(fix_atoms=[2, 0, 2])
    assert request["recipe"]["fix_atoms"] == [0, 2]
    with pytest.raises(ConstraintRequestError, match="0-based position"):
        _build(fix_atoms=[5])
    with pytest.raises(ConstraintRequestError, match="0-based position"):
        _build(fix_atoms=[True])
    with pytest.raises(ConstraintRequestError, match="0-based position"):
        _build(fix_atoms=[0.0])


def test_geometry_and_elements_length_must_match():
    with pytest.raises(ConstraintRequestError, match="geometry has"):
        _build(geometry=GEOMETRY[:-1])
    with pytest.raises(ConstraintRequestError, match="non-empty"):
        _build(elements=[])


def test_targets_shape_unit_and_arity_violations():
    with pytest.raises(ConstraintRequestError, match="at least one driver"):
        _build(targets=[])
    with pytest.raises(ConstraintRequestError, match="unsupported coordinate kind"):
        _build(targets=[DriverTarget("x", "coordinate", (0, 1), 1.0, "angstrom")])
    with pytest.raises(ConstraintRequestError, match="exactly 2 distinct atoms"):
        _build(targets=[DriverTarget("x", "distance", (0,), 1.0, "angstrom")])
    with pytest.raises(ConstraintRequestError, match="positive"):
        _build(targets=[_distance_target(0.0)])
    with pytest.raises(ConstraintRequestError, match="positive"):
        _build(targets=[_distance_target(-1.0)])
    with pytest.raises(ConstraintRequestError, match="0-based position"):
        _build(targets=[DriverTarget("x", "distance", (0, 3), 1.0, "angstrom")])
    with pytest.raises(ConstraintRequestError, match="0-based position"):
        _build(targets=[DriverTarget("x", "distance", (0, True), 1.0, "angstrom")])


def test_real_plan_driver_atoms_are_zero_based_positions():
    targets = [
        DriverTarget("B_1_11", "distance", (0, 10), 2.7210149455364996, "angstrom"),
        DriverTarget("B_3_11", "distance", (2, 10), 1.006732710451488, "angstrom"),
    ]
    request = _build(elements=PLAN_ELEMENTS, geometry=PLAN_GEOMETRY, targets=targets,
                     reaction_id="RXN_0000007104")
    constraints = request["recipe"]["constraints"]
    assert [c["atoms"] for c in constraints] == [[0, 10], [2, 10]]
    # Map id 11 binds to 0-based position 10 — never map_id, never map_id-1
    # misapplied: position 10 is the 11th atom in the frozen order.
    assert PLAN_ELEMENTS[10] == "H"
    validate_constraint_request(request)
    assert len(request["provenance"]["atom_order_sha256"]) == 64


def test_provenance_optional_digests_are_carried_when_given():
    request = _build(plan_sha256="p" * 64, parent_frame_sha256="q" * 64)
    assert request["provenance"]["plan_sha256"] == "p" * 64
    assert request["provenance"]["parent_frame_sha256"] == "q" * 64
    plain = _build()
    assert "plan_sha256" not in plain["provenance"]


def test_digest_helpers_are_deterministic_and_round_to_six_decimals():
    assert atom_order_digest(ELEMENTS, 3) == atom_order_digest(list(ELEMENTS), 3)
    assert atom_order_digest(ELEMENTS, 3) != atom_order_digest(ELEMENTS, 4)
    near_a = [[0.0, 0.0, 0.0], [2.0000001, 0.0, 0.0]]
    near_b = [[0.0, 0.0, 0.0], [2.0000004, 0.0, 0.0]]
    far = [[0.0, 0.0, 0.0], [2.000002, 0.0, 0.0]]
    assert geometry_digest(near_a, ["C", "H"]) == geometry_digest(near_b, ["C", "H"])
    assert geometry_digest(near_a, ["C", "H"]) != geometry_digest(far, ["C", "H"])
    assert geometry_digest(near_a, ["C", "H"]) != geometry_digest(near_a, ["H", "C"])


def test_validate_constraint_request_rejects_broken_payloads():
    valid = _build()
    validate_constraint_request(valid)

    broken = dict(valid)
    with pytest.raises(ConstraintRequestError, match="must be a JSON object"):
        validate_constraint_request(["not", "a", "dict"])
    broken_schema = dict(valid, schema_version="pes2ts_other_v1")
    with pytest.raises(ConstraintRequestError, match="schema_version"):
        validate_constraint_request(broken_schema)
    broken_source = dict(valid, source={"elements": ELEMENTS})
    with pytest.raises(ConstraintRequestError, match="source"):
        validate_constraint_request(broken_source)
    short_geometry = dict(valid)
    short_geometry["source"] = {
        "elements": ELEMENTS, "geometry": GEOMETRY[:-1],
        "charge": 0, "multiplicity": 1,
    }
    with pytest.raises(ConstraintRequestError, match="geometry must match"):
        validate_constraint_request(short_geometry)
    bad_row = dict(valid)
    bad_row["source"] = {
        "elements": ELEMENTS, "geometry": [[0, 0], GEOMETRY[1], GEOMETRY[2]],
        "charge": 0, "multiplicity": 1,
    }
    with pytest.raises(ConstraintRequestError, match=r"\[x, y, z\]"):
        validate_constraint_request(bad_row)
    no_constraints = dict(valid, recipe=dict(valid["recipe"], constraints=[]))
    with pytest.raises(ConstraintRequestError, match="constraints"):
        validate_constraint_request(no_constraints)
    bad_unit = json.loads(json.dumps(valid))
    bad_unit["recipe"]["constraints"][0]["unit"] = "bohr"
    with pytest.raises(ConstraintRequestError, match="requires unit"):
        validate_constraint_request(bad_unit)
    bad_atom = json.loads(json.dumps(valid))
    bad_atom["recipe"]["constraints"][0]["atoms"] = [0, 9]
    with pytest.raises(ConstraintRequestError, match="out of range"):
        validate_constraint_request(bad_atom)
    bad_target = json.loads(json.dumps(valid))
    bad_target["recipe"]["constraints"][0]["target"] = 0.0
    with pytest.raises(ConstraintRequestError, match="positive"):
        validate_constraint_request(bad_target)
