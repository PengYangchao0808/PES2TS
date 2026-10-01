"""Offline tests for the PathRequest / NEB protocol (plan todo 26).

Locks the todo-26 acceptance surface:

* ``PathRequest`` carries **both endpoint geometries in the same frozen atom
  order** (materials ``atom_map_ids`` claim or the bundle conservation
  order) plus image-chain params (``n_images`` + optional spring/backend);
* missing endpoint geometry and atom-order mismatch are **typed refusals** —
  never fabricated coordinates, never a "NEB distance coordinate" projected
  onto older scan-shaped contracts;
* NEB and Scan are accounted in **separate cost/success channels**: entering
  NEB records a native-scan-branch exit and never counts as a scan success;
* a NEB-found intermediate minimum carries the structural caveat and the
  typed flag for adjacent-segment handling (todo 27 staging);
* ``PathRequest.to_path_candidate`` produces a contracts_v2
  ``PathCandidateV1`` payload that ``plan_freeze.freeze_path_candidate_plan``
  accepts (sealed plan, ``validate_v2_document`` + ``verify_generation_plan``
  clean) and that ``compile_orca.compile_orca_path_request`` compiles into
  the ORCA NEB input shape under a probed capability;
* determinism: identical inputs produce byte-identical documents.

Every fixture is synthetic/in-memory (mapped SMILES + whitelist materials);
no test reads the real ``data/`` tree.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from test_compile_orca import _smoke  # noqa: E402
from test_selector_core import (  # noqa: E402
    SINGLE_BOND_P,
    SINGLE_BOND_R,
    SINGLE_BOND_SMILES,
    _bundle,
    _config,
    _materials,
)

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.scan_strategy.compile_orca import (
    CODE_CAPABILITY_UNPROBED,
    OrcaCompileError,
    compile_orca_path_request,
)
from pes2ts_core.scan_strategy.contracts_v2 import (
    CANDIDATE_KIND_PATH,
    PATH_METHOD_KINDS,
    validate_v2_document,
)
from pes2ts_core.scan_strategy.path_request import (
    BRANCH_NATIVE_SCAN_EXITED,
    CHANNEL_NEB,
    CHANNEL_SCAN,
    CODE_ATOM_ORDER_MISMATCH,
    CODE_ENDPOINT_GEOMETRY_MISSING,
    CODE_IMAGE_CHAIN_INVALID,
    FABRICATED_DISTANCE_KEYS,
    FORBIDDEN_PATH_KEYS,
    METHOD_CHANNELS,
    NEB_ENTER_EXITS_SCAN_BRANCH,
    SINGLE_ELEMENTARY_PROCESS_GUARANTEED,
    PathRequest,
    PathRequestError,
    enter_neb_channels,
    flag_intermediate_minimum,
    make_path_request,
    path_request_from_doc,
    record_channel_outcome,
    request_json,
)
from pes2ts_core.scan_strategy.plan_freeze import (
    PlanFreezeError,
    freeze_path_candidate_plan,
    verify_generation_plan,
)
from pes2ts_core.utils.hashing import stable_json_dumps

SHA_A = "a" * 64
FORBIDDEN: frozenset[str] = frozenset(
    {key.lower() for key in FORBIDDEN_TRUTH_KEYS}
    | {key.lower() for key in FORBIDDEN_EXPORT_KEYS}
)


def _request(**overrides: Any) -> PathRequest:
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    materials = _materials(SINGLE_BOND_R, SINGLE_BOND_P)
    return make_path_request(bundle, materials, _config(), 5, **overrides)


# ---------------------------------------------------------------------------
# Happy path: same atom order at both endpoints + image chain params.
# ---------------------------------------------------------------------------
def test_path_request_same_atom_order_endpoints_and_image_chain() -> None:
    request = _request(reaction_id="RXN_0000000001")
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    assert request.atom_rows == tuple(bundle.conservation.map_ids)
    assert request.n_atoms == len(request.atom_rows)
    assert len(request.elements) == request.n_atoms
    reactant = request.endpoint_geometries["reactant"]
    product = request.endpoint_geometries["product"]
    assert len(reactant) == request.n_atoms == len(product)
    # Both blocks are row-aligned to the SAME frozen order (row i = atom_rows[i]).
    for position, map_id in enumerate(request.atom_rows):
        assert reactant[position] == SINGLE_BOND_R[map_id]
        assert product[position] == SINGLE_BOND_P[map_id]
    assert request.image_chain.n_images == 5
    assert request.method_kind == "NEB" and request.method_kind in PATH_METHOD_KINDS
    doc = request.to_doc()
    assert doc["image_chain"]["n_images"] == 5
    assert doc["atom_rows"] == list(request.atom_rows)
    assert set(doc["endpoint_geometries"]) == {"reactant", "product"}


def test_path_request_materials_atom_map_ids_define_frozen_order() -> None:
    """A non-map-sorted atom_map_ids claim is the frozen order for BOTH sides."""
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    order = [4, 3, 2, 1]
    materials = _materials(SINGLE_BOND_R, SINGLE_BOND_P)
    materials["atom_map_ids"] = order
    # Aligned coordinate lists must follow the same claimed order.
    materials["r_coordinates"] = [list(SINGLE_BOND_R[m]) for m in order]
    materials["p_coordinates"] = [list(SINGLE_BOND_P[m]) for m in order]
    request = make_path_request(bundle, materials, _config(), 4)
    assert request.atom_rows == (4, 3, 2, 1)
    assert request.endpoint_geometries["reactant"][0] == SINGLE_BOND_R[4]
    assert request.endpoint_geometries["product"][0] == SINGLE_BOND_P[4]
    assert request.endpoint_geometries["reactant"][3] == SINGLE_BOND_R[1]
    # Permutation is NOT a mismatch — same order, both sides, by construction.
    map_keyed = _materials(SINGLE_BOND_R, SINGLE_BOND_P)
    map_keyed["atom_map_ids"] = order
    request2 = make_path_request(bundle, map_keyed, _config(), 4)
    assert request2.endpoint_geometries == request.endpoint_geometries


def test_path_request_recovery_protocol_and_policy() -> None:
    config = _config(
        path_request={"deep_intermediate_threshold": 1.5},
    )
    request = make_path_request(
        _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P)),
        _materials(SINGLE_BOND_R, SINGLE_BOND_P),
        config,
        {"n_images": 7, "spring_constant": 0.05, "neb_backend_id": "orca"},
    )
    assert request.image_chain.spring_constant == 0.05
    assert request.image_chain.neb_backend_id == "orca"
    assert request.policy["max_path_neb_candidates"] == 1
    assert request.policy["deep_intermediate_threshold"] == 1.5
    protocol = request.recovery_protocol.to_doc()
    assert protocol["per_image_fields"] == [
        "per_image_energy",
        "gradient_availability",
        "convergence",
    ]
    caveat = protocol["intermediate_minimum_policy"]
    assert caveat["single_elementary_process_guaranteed"] is False
    assert caveat["handled_by"] == "intermediate_evidence_staging"


def test_path_request_yaml_numeric_string_policy() -> None:
    """YAML 1.1 quirk: deep_intermediate_threshold may arrive as a string."""
    config = {
        "scan_strategy": {
            "max_total_candidates": {"scan": 6, "path_neb": 2},
            "path_request": {"deep_intermediate_threshold": "1.5"},
        }
    }
    request = make_path_request(
        _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P)),
        _materials(SINGLE_BOND_R, SINGLE_BOND_P),
        config,
        3,
    )
    assert request.policy["deep_intermediate_threshold"] == 1.5
    assert request.policy["max_path_neb_candidates"] == 2


# ---------------------------------------------------------------------------
# Typed refusals — no fabrication.
# ---------------------------------------------------------------------------
def test_refuses_missing_endpoint_geometry_materials_none() -> None:
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    with pytest.raises(PathRequestError) as excinfo:
        make_path_request(bundle, None, _config(), 5)
    assert excinfo.value.code == CODE_ENDPOINT_GEOMETRY_MISSING
    assert "never fabricated" in str(excinfo.value)


def test_refuses_missing_endpoint_geometry_partial_maps() -> None:
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    materials = _materials(SINGLE_BOND_R, SINGLE_BOND_P)
    del materials["p_coordinates"][4]
    with pytest.raises(PathRequestError) as excinfo:
        make_path_request(bundle, materials, _config(), 5)
    assert excinfo.value.code == CODE_ENDPOINT_GEOMETRY_MISSING
    assert "4" in str(excinfo.value)


def test_refuses_atom_order_mismatch_extra_or_missing_maps() -> None:
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    materials = _materials(SINGLE_BOND_R, SINGLE_BOND_P)
    materials["atom_map_ids"] = [1, 2, 3]  # bundle has maps 1..4
    with pytest.raises(PathRequestError) as excinfo:
        make_path_request(bundle, materials, _config(), 5)
    assert excinfo.value.code == CODE_ATOM_ORDER_MISMATCH
    assert "missing" in str(excinfo.value)


def test_refuses_atom_order_mismatch_duplicate_maps() -> None:
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    materials = _materials(SINGLE_BOND_R, SINGLE_BOND_P)
    materials["atom_map_ids"] = [1, 2, 2, 4]
    with pytest.raises(PathRequestError) as excinfo:
        make_path_request(bundle, materials, _config(), 5)
    assert excinfo.value.code == CODE_ATOM_ORDER_MISMATCH


def test_refusal_path_returns_no_partial_request() -> None:
    """A refusal is an exception — no PathRequest doc is produced at all."""
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    with pytest.raises(PathRequestError):
        make_path_request(bundle, None, _config(), 5)
    # Aligned lists without atom_map_ids are a schema refusal, not a guess.
    materials = _materials(SINGLE_BOND_R, SINGLE_BOND_P)
    materials["r_coordinates"] = [list(v) for v in SINGLE_BOND_R.values()]
    with pytest.raises(PathRequestError) as excinfo:
        make_path_request(bundle, materials, _config(), 5)
    assert excinfo.value.code == "MATERIALS_SCHEMA_INVALID"


def test_refuses_invalid_image_chain() -> None:
    bundle = _bundle(SINGLE_BOND_SMILES, _materials(SINGLE_BOND_R, SINGLE_BOND_P))
    materials = _materials(SINGLE_BOND_R, SINGLE_BOND_P)
    with pytest.raises(PathRequestError) as excinfo:
        make_path_request(bundle, materials, _config(), 0)
    assert excinfo.value.code == CODE_IMAGE_CHAIN_INVALID
    with pytest.raises(PathRequestError) as excinfo:
        make_path_request(bundle, materials, _config(), {"n_images": 3, "spring_constant": -1.0})
    assert excinfo.value.code == CODE_IMAGE_CHAIN_INVALID


def test_no_fabricated_distance_coordinate_keys() -> None:
    """Path docs/candidates never carry scan-shaped or distance-coordinate keys."""
    request = _request()
    for document in (request.to_doc(), request.to_path_candidate(candidate_id="c-1")):
        flat_keys = set(document.keys())
        assert not (flat_keys & FABRICATED_DISTANCE_KEYS)
        serialized = stable_json_dumps(document)
        for banned in FABRICATED_DISTANCE_KEYS:
            assert f'"{banned}"' not in serialized


# ---------------------------------------------------------------------------
# Method channels: NEB and Scan accounted separately.
# ---------------------------------------------------------------------------
def test_entering_neb_exits_native_scan_branch_on_separate_channels() -> None:
    channels = enter_neb_channels()
    assert tuple(c.method_kind for c in channels) == METHOD_CHANNELS
    by_kind = {c.method_kind: c for c in channels}
    neb, scan = by_kind[CHANNEL_NEB], by_kind[CHANNEL_SCAN]
    assert neb.n_entered == 1
    assert neb.branch == BRANCH_NATIVE_SCAN_EXITED
    assert "never scan successes" in neb.cost_note or "NEB channel" in neb.cost_note
    assert scan.n_entered == 0
    assert scan.n_succeeded == 0
    assert scan.branch == BRANCH_NATIVE_SCAN_EXITED
    assert NEB_ENTER_EXITS_SCAN_BRANCH in neb.cost_note


def test_neb_success_never_counts_as_scan_success() -> None:
    channels = enter_neb_channels()
    updated = record_channel_outcome(channels, CHANNEL_NEB, succeeded=True)
    by_kind = {c.method_kind: c for c in updated}
    assert by_kind[CHANNEL_NEB].n_succeeded == 1
    assert by_kind[CHANNEL_SCAN].n_succeeded == 0
    assert by_kind[CHANNEL_SCAN].n_entered == 0
    # Input tuple is untouched (immutable channels).
    assert {c.method_kind: c for c in channels}[CHANNEL_NEB].n_succeeded == 0


def test_path_request_doc_carries_separate_channels() -> None:
    doc = _request().to_doc()
    assert set(doc["method_channels"]) == {CHANNEL_NEB, CHANNEL_SCAN}
    assert doc["method_channels"][CHANNEL_NEB]["n_entered"] == 1
    assert doc["method_channels"][CHANNEL_SCAN]["n_succeeded"] == 0


# ---------------------------------------------------------------------------
# Intermediate-minimum typed flag (design §8.2; staging owned by todo 27).
# ---------------------------------------------------------------------------
def test_intermediate_minimum_flag_present_and_typed() -> None:
    flag = flag_intermediate_minimum([0.0, -0.4, 0.2, 1.0])
    assert flag.interior_minimum is True
    assert flag.image_index == 1
    assert flag.depth == pytest.approx(0.4)
    assert flag.deep_intermediate is False
    assert flag.adjacent_segment_required is False
    assert flag.single_elementary_process_guaranteed is False
    assert flag.reason_code == "NEB_INTERMEDIATE_MINIMUM"
    assert flag.next_stage == "intermediate_evidence_staging"


def test_deep_intermediate_flag_requires_adjacent_segment_handling() -> None:
    flag = flag_intermediate_minimum([0.0, -2.0, 0.1, 1.5], deep_threshold=0.5)
    assert flag.deep_intermediate is True
    assert flag.adjacent_segment_required is True
    assert flag.reason_code == "DEEP_INTERMEDIATE_MINIMUM"
    assert flag.single_elementary_process_guaranteed is False


def test_monotonic_profile_has_no_interior_minimum() -> None:
    flag = flag_intermediate_minimum([0.0, 0.5, 1.0, 2.0])
    assert flag.interior_minimum is False
    assert flag.adjacent_segment_required is False
    assert flag.reason_code == "NO_INTERIOR_MINIMUM"


# ---------------------------------------------------------------------------
# PathCandidateV1 freeze compatibility + ORCA NEB compile shape.
# ---------------------------------------------------------------------------
def test_path_candidate_freeze_compatibility() -> None:
    request = _request(reaction_id="RXN_0000000001")
    candidate = request.to_path_candidate(candidate_id="cand-path-26")
    plan = freeze_path_candidate_plan(
        candidate,
        reaction_id="RXN_0000000001",
        case_id="case-0001",
        split="train",
        source_case_sha256=SHA_A,
        endpoint_graph_sha256=request.bundle_content_sha256,
    )
    assert validate_v2_document(plan) == []
    assert verify_generation_plan(plan) == []
    frozen = plan["candidates"][0]
    assert frozen["candidate_kind"] == CANDIDATE_KIND_PATH
    assert frozen["n_atoms"] == request.n_atoms
    assert frozen["endpoint_geometries"]["reactant"] == [
        list(row) for row in request.endpoint_geometries["reactant"]
    ]
    assert frozen["image_chain"]["n_images"] == 5
    assert frozen["method_kind"] == "NEB"
    assert frozen["extensions"]["path_request"]["atom_rows"] == list(request.atom_rows)
    assert plan["extensions"]["freeze"]["atom_order"] == list(request.atom_rows)
    assert plan["compiled"]["kind"] == "recipe"
    assert plan["status"] == "frozen"


def test_path_candidate_freeze_deterministic_and_refuses_dirty_candidates() -> None:
    request = _request(reaction_id="RXN_0000000001")
    candidate = request.to_path_candidate(candidate_id="cand-path-26")
    plan_a = freeze_path_candidate_plan(
        candidate, reaction_id="RXN_0000000001", case_id="case-0001",
        source_case_sha256=SHA_A,
    )
    plan_b = freeze_path_candidate_plan(
        copy.deepcopy(candidate), reaction_id="RXN_0000000001", case_id="case-0001",
        source_case_sha256=SHA_A,
    )
    assert plan_a["content_sha256"] == plan_b["content_sha256"]
    dirty = copy.deepcopy(candidate)
    dirty["failure_reasons"] = [{"code": "SCF_FAILED", "detail": "x"}]
    with pytest.raises(PlanFreezeError) as excinfo:
        freeze_path_candidate_plan(
            dirty, reaction_id="RXN_0000000001", case_id="case-0001",
            source_case_sha256=SHA_A,
        )
    assert excinfo.value.code == "CANDIDATE_NOT_CLEAN"
    wrong_kind = copy.deepcopy(candidate)
    wrong_kind["candidate_kind"] = "ScanCandidateV2"
    with pytest.raises(PlanFreezeError) as excinfo:
        freeze_path_candidate_plan(
            wrong_kind, reaction_id="RXN_0000000001", case_id="case-0001",
            source_case_sha256=SHA_A,
        )
    assert excinfo.value.code == "PATH_CANDIDATE_INVALID"


def test_compile_orca_path_request_neb_shape(tmp_path) -> None:
    capability = _smoke(tmp_path)
    request = _request(reaction_id="RXN_0000000001")
    compiled = compile_orca_path_request(request, capability)
    assert compiled.mode == "PATH_NEB"
    assert compiled.total_points == 5
    assert compiled.atom_rows == request.atom_rows
    fragments = compiled.geom_fragments
    assert any("Path" in f and "n_images 5" in f for f in fragments)
    assert sum(1 for f in fragments if f.startswith("* xyz")) == 2
    assert compiled.recipe["image_chain"]["n_images"] == 5
    assert compiled.recipe["path_request"]["request_id"] == request.request_id
    assert "recovery_protocol" in compiled.recipe
    # Deterministic hashes over the same request.
    again = compile_orca_path_request(request, capability)
    assert again.point_input_sha256 == compiled.point_input_sha256


def test_compile_orca_path_request_capability_gate_first(tmp_path) -> None:
    capability = _smoke(tmp_path, probed=False)
    request = _request()
    with pytest.raises(OrcaCompileError) as excinfo:
        compile_orca_path_request(request, capability)
    assert excinfo.value.code == CODE_CAPABILITY_UNPROBED


# ---------------------------------------------------------------------------
# Determinism, purity, round-trip.
# ---------------------------------------------------------------------------
def test_determinism_same_inputs_byte_identical_documents() -> None:
    request_a = _request(reaction_id="RXN_0000000001")
    request_b = _request(reaction_id="RXN_0000000001")
    assert request_json(request_a) == request_json(request_b)
    assert request_a.to_doc() == request_b.to_doc()
    assert (
        request_a.to_path_candidate(candidate_id="c-1")
        == request_b.to_path_candidate(candidate_id="c-1")
    )


def test_rehydrate_round_trip_is_stable() -> None:
    request = _request(reaction_id="RXN_0000000001")
    revived = path_request_from_doc(request.to_doc())
    assert request_json(revived) == request_json(request)


def test_path_request_documents_carry_no_forbidden_truth_keys() -> None:
    request = _request(reaction_id="RXN_0000000001")
    for document in (request.to_doc(), request.to_path_candidate(candidate_id="c-1")):
        def _walk(value: Any, path: str = "$") -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    assert str(key).lower() not in FORBIDDEN, f"{path}.{key}"
                    assert str(key).lower() not in FORBIDDEN_PATH_KEYS, f"{path}.{key}"
                    _walk(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    _walk(child, f"{path}[{index}]")

        _walk(document)
