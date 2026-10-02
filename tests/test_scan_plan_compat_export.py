"""Offline P1 acceptance for the legacy single-B compatibility export (todo 18).

Locks the ``pes2ts_core/generation/planning/compat_export.py`` bridge:

* a synthetic ready case → selector → freeze-shaped SINGLE_1D plan exports a
  v1 ``ScanPlan`` that passes ``contracts.validate_document``/``dumps_document``
  and the adapter preflight ``scan_plan_to_acp_request``;
* the exported plan roundtrips losslessly through the real
  ``integration.acp.cli_backend`` against the fake ``acp.cli`` process
  (request payload correct, artifacts read back);
* multi-coordinate (COUPLED/SCHEDULED), A/D driver kinds, non-uniform point
  lists, PathCandidateV1, and non-frozen/proposal inputs are **typed
  refusals** (``CompatExportError`` ⊂ ``ACPMappingError``) — never projected;
* 3..101 point bounds are honored and export is deterministic;
* needs_review proposals never produce a ready export, including every
  Demo24 fixture record (human review is not lifted by this plan).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from rdkit import rdBase

from pes2ts_core.contracts import dumps_document, make_document, seal_document, validate_document
from pes2ts_core.integration.acp.adapter import ACPMappingError, scan_plan_to_acp_request
from pes2ts_core.integration.acp.cli_backend import (
    ACPCLIBackend,
    collect_cli_path_bundle,
    cli_result_to_execution_record,
    validate_cli_result,
)
from pes2ts_core.integration.acp.quality import apply_scan_path_quality
from pes2ts_core.generation.planning.compat_export import (
    DRIVER_KIND_UNSUPPORTED,
    INDEX_ORDER_MISMATCH,
    MULTI_COORDINATE_UNSUPPORTED,
    NEEDS_REVIEW_OR_PROPOSAL_REFUSED,
    NONUNIFORM_POINTS,
    PATH_CANDIDATE_UNSUPPORTED,
    PLAN_NOT_FROZEN,
    POINTS_OUT_OF_BOUNDS,
    SOURCE_CASE_BINDING_MISMATCH,
    CompatExportError,
    export_legacy_scan_plan,
)
from pes2ts_core.generation.planning.contracts_v2 import (
    make_generation_plan,
    make_strategy_proposal,
    validate_v2_document,
)
from pes2ts_core.generation.planning.graph_rebuild import (
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)
from pes2ts_core.generation.planning.selector import propose_strategies
from pes2ts_core.utils.hashing import stable_json_dumps

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "p0_demo24"

SINGLE_BOND_SMILES = "[CH3:1][CH3:2].[CH3:3][CH3:4]>>[CH3:1][CH2:2][CH2:3][CH3:4]"
R_XYZ = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0), 3: (6.0, 3.0, 0.0), 4: (7.54, 3.0, 0.0)}
P_XYZ = {1: (0.0, 0.0, 0.0), 2: (1.54, 0.0, 0.0), 3: (3.08, 0.0, 0.0), 4: (4.62, 2.0, 0.0)}


def _config() -> dict[str, Any]:
    return {
        "scan_strategy": {
            "context_radius": 2,
            "max_scan_coordinates": 3,
            "point_limits": {"baseline": 9, "max": 101},
            "schedule_budget": 3,
            "direction_budget": 2,
            "assembly_candidate_budget": 2,
            "max_total_candidates": {"scan": 6, "path_neb": 1},
            "max_step_by_kind": {"distance": 0.2, "angle": 10.0, "dihedral": 10.0},
            "jacobian_condition_number_max": 1.0e6,
            "element_pair_bond_thresholds": {"tolerance": 0.45},
            "special_domain_policy": "unsupported",
        },
        "g2": {"validity": {"collision_min_distance": 0.8}},
    }


def _ready_case() -> dict[str, Any]:
    return make_document(
        "ReactionCase",
        "case:compat-single-bond",
        "ready",
        dataset_version="synthetic-compat-v1",
        reaction_id="RXN_COMPAT_0001",
        case_id="case:compat-single-bond",
        split="train",
        atoms=[
            {"atom_map_id": 1, "element": "C"},
            {"atom_map_id": 2, "element": "C"},
            {"atom_map_id": 3, "element": "C"},
            {"atom_map_id": 4, "element": "C"},
        ],
        reactant={"charge": 0, "multiplicity": 1,
                  "geometry": [list(R_XYZ[i]) for i in (1, 2, 3, 4)]},
        product={"charge": 0, "multiplicity": 1,
                 "geometry": [list(P_XYZ[i]) for i in (1, 2, 3, 4)]},
        edits=[{"kind": "formed", "atom_map_ids": [2, 3]}],
        hydrogen_transfers=[],
        source={"dataset": "synthetic", "mapping_provenance": "endpoint_only"},
    )


def _selector_bundle(case: dict[str, Any]):
    order = [atom["atom_map_id"] for atom in case["atoms"]]
    export_like: dict[str, dict[int, dict[str, Any]]] = {"r": {}, "p": {}}
    for side_key, rows in (("r", case["reactant"]["geometry"]), ("p", case["product"]["geometry"])):
        for index, map_id in enumerate(order):
            export_like[side_key][map_id] = {"element": "C", "coordinates": list(rows[index])}
    bundle = rebuild_endpoint_graphs(SINGLE_BOND_SMILES, export_like)
    materials = {
        "r_coordinates": {map_id: list(case["reactant"]["geometry"][i]) for i, map_id in enumerate(order)},
        "p_coordinates": {map_id: list(case["product"]["geometry"][i]) for i, map_id in enumerate(order)},
        "endpoint_electronic": {
            "reactant": {"charge": 0, "multiplicity": 1},
            "product": {"charge": 0, "multiplicity": 1},
        },
    }
    return bundle, materials


def _propose(case: dict[str, Any]) -> dict[str, Any]:
    bundle, materials = _selector_bundle(case)
    return propose_strategies(
        bundle,
        materials,
        _config(),
        reaction_id=case["reaction_id"],
        case_id=case["case_id"],
        split=case["split"],
        source_case_sha256=case["content_sha256"],
    )


def _exportable_proposal_candidates(
    proposal: Mapping[str, Any],
    *,
    prefer_start: str | None = None,
    schedule_kind: str | None = "linear",
) -> list[Mapping[str, Any]]:
    picked: list[Mapping[str, Any]] = []
    for candidate in proposal.get("candidates") or ():
        if candidate.get("mode") != "SINGLE_1D":
            continue
        drivers = candidate.get("drivers") or []
        if len(drivers) != 1 or drivers[0].get("kind") != "B":
            continue
        if schedule_kind is not None and candidate.get("schedule_kind") not in (None, schedule_kind):
            continue
        codes = {row.get("code") for row in candidate.get("failure_reasons") or [] if isinstance(row, Mapping)}
        if "PRUNED_BY_BUDGET" in codes:
            continue
        if prefer_start is not None and candidate.get("start_endpoint") != prefer_start:
            continue
        picked.append(candidate)
    return picked


def _freeze_shaped_plan(
    proposal: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    prefer_start: str | None = None,
    plan_id: str | None = None,
) -> dict[str, Any]:
    candidates = _exportable_proposal_candidates(proposal, prefer_start=prefer_start)
    if not candidates:
        candidates = _exportable_proposal_candidates(proposal, prefer_start=None)
    if not candidates:
        raise AssertionError("selector proposal carries no exportable SINGLE_1D B-driver candidate")
    prop_cand = dict(candidates[0])
    frozen_candidate = {key: value for key, value in prop_cand.items() if key != "capability_check"}
    frozen_candidate["candidate_kind"] = "ScanCandidateV2"
    frozen_candidate["failure_reasons"] = []
    pid = plan_id or f"plan-compat-{proposal['content_sha256'][:12]}"
    return make_generation_plan(
        pid,
        "frozen",
        plan_id=pid,
        reaction_id=case["reaction_id"],
        case_id=case["case_id"],
        split=case["split"],
        plan_version=1,
        source_proposal_sha256=proposal["content_sha256"],
        source_case_sha256=case["content_sha256"],
        policy_hashes={"selector": proposal["content_sha256"][:16], "compat_export": "p1"},
        backend={"engine": "xtb", "adapter_version": "acp-adapter-v0", "method": "GFN2-xTB"},
        candidate_graph={
            "nodes": [{"node_id": frozen_candidate["candidate_id"], "state": "frozen", "terminal": True}],
            "edges": [],
        },
        candidates=[frozen_candidate],
        budget=dict(prop_cand["budget"]),
        compiled={
            "kind": "recipe",
            "recipe": {"note": "P1 compat-export acceptance; physical points compiled by compat_export"},
        },
        quality_tests=[],
    )


def _hand_materials(
    *,
    atom_map_ids: tuple[int, ...] = (1, 2, 3, 4),
    content_sha256: str = "a" * 64,
    reaction_id: str = "RXN_COMPAT_HAND",
    case_id: str = "case:compat-hand",
    split: str = "train",
    dataset_version: str = "synthetic-compat-v1",
    r_coordinates: Mapping[int, tuple[float, float, float]] | None = None,
    p_coordinates: Mapping[int, tuple[float, float, float]] | None = None,
) -> dict[str, Any]:
    r_coordinates = r_coordinates or R_XYZ
    p_coordinates = p_coordinates or P_XYZ
    return {
        "atom_map_ids": list(atom_map_ids),
        "content_sha256": content_sha256,
        "reaction_id": reaction_id,
        "case_id": case_id,
        "split": split,
        "dataset_version": dataset_version,
        "r_coordinates": {map_id: list(r_coordinates[map_id]) for map_id in atom_map_ids},
        "p_coordinates": {map_id: list(p_coordinates[map_id]) for map_id in atom_map_ids},
    }


def _hand_plan(
    mats: Mapping[str, Any],
    *,
    mode: str = "SINGLE_1D",
    drivers: list[dict[str, Any]] | None = None,
    lambda_values: list[float] | None = None,
    start_endpoint: str = "R",
    direction: str = "R_to_P",
    plan_status: str = "frozen",
    plan_id: str = "plan-compat-hand",
    backend: Mapping[str, Any] | None = None,
    candidate_kind: str = "ScanCandidateV2",
) -> dict[str, Any]:
    if drivers is None:
        drivers = [{"kind": "B", "maps": [2, 3], "unit": "angstrom", "schedule_values": None, "index0": None}]
    if lambda_values is None:
        lambda_values = [index / 8 for index in range(9)]
    candidate: dict[str, Any] = {
        "candidate_kind": candidate_kind,
        "candidate_id": "cand-hand-000",
        "mode": mode,
        "start_endpoint": start_endpoint,
        "direction": direction,
        "assembly_id": "asm-hand",
        "anchor_reason": "HAND_TEST",
        "drivers": drivers,
        "monitors": [],
        "guards": [],
        "event_coverage": [],
        "lambda_values": lambda_values,
        "schedule_id": "sched:linear",
        "required_capabilities": [mode],
        "budget": {"max_attempts": 1, "max_cpu_hours": 1.0, "max_wall_seconds": 60},
        "failure_reasons": [],
    }
    if mode == "SCHEDULED_1D":
        candidate["schedule_kind"] = "linear"
    if candidate_kind == "PathCandidateV1":
        candidate = {
            "candidate_kind": "PathCandidateV1",
            "candidate_id": "cand-hand-000",
            "n_atoms": 4,
            "endpoint_geometries": {
                "reactant": [list(R_XYZ[i]) for i in (1, 2, 3, 4)],
                "product": [list(P_XYZ[i]) for i in (1, 2, 3, 4)],
            },
            "image_chain": {"n_images": 5},
            "start_endpoint": start_endpoint,
            "direction": direction,
            "anchor_reason": "HAND_TEST",
            "failure_reasons": [],
        }
    resolved_backend = dict(backend or {"engine": "xtb", "adapter_version": "acp-adapter-v0", "method": "GFN2-xTB"})
    return make_generation_plan(
        plan_id,
        plan_status,
        plan_id=plan_id,
        reaction_id=mats["reaction_id"],
        case_id=mats["case_id"],
        split=mats["split"],
        plan_version=1,
        source_proposal_sha256="b" * 64,
        source_case_sha256=mats["content_sha256"],
        policy_hashes={"compat_export": "hand"},
        backend=resolved_backend,
        candidate_graph={
            "nodes": [{"node_id": candidate["candidate_id"], "state": "frozen", "terminal": True}],
            "edges": [],
        },
        candidates=[candidate],
        budget={"max_attempts": 1, "max_cpu_hours": 1.0, "max_wall_seconds": 60},
        compiled={"kind": "recipe", "recipe": {"note": "hand-built compat-export acceptance plan"}},
        quality_tests=[],
    )


def _case_from_mats(mats: Mapping[str, Any]) -> dict[str, Any]:
    order = list(mats["atom_map_ids"])
    return make_document(
        "ReactionCase",
        mats["case_id"],
        "ready",
        dataset_version=mats["dataset_version"],
        reaction_id=mats["reaction_id"],
        case_id=mats["case_id"],
        split=mats["split"],
        atoms=[{"atom_map_id": map_id, "element": "C"} for map_id in order],
        reactant={"charge": 0, "multiplicity": 1,
                  "geometry": [list(mats["r_coordinates"][map_id]) for map_id in order]},
        product={"charge": 0, "multiplicity": 1,
                 "geometry": [list(mats["p_coordinates"][map_id]) for map_id in order]},
        edits=[{"kind": "formed", "atom_map_ids": [2, 3]}],
        hydrogen_transfers=[],
        source={"dataset": "synthetic", "mapping_provenance": "endpoint_only"},
    )


def _fake_acp(tmp_path: Path) -> Path:
    root = tmp_path / "fake_acp"
    package = root / "src" / "acp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(r'''import json, pathlib, sys, time
args = sys.argv
request = json.loads(pathlib.Path(args[args.index("--scan-config") + 1]).read_text(encoding="utf-8"))
output = pathlib.Path(args[args.index("--output") + 1])
delay = request.get("protocol", {}).get("test_delay_seconds", 0)
time.sleep(delay)
if request.get("protocol", {}).get("test_exit_code"):
    raise SystemExit(int(request["protocol"]["test_exit_code"]))
work = output / "WORK" / "07_PATH" / "pes_scan_001"
result = output / "RESULT" / "pes_search"
frames_dir = work / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
result.mkdir(parents=True, exist_ok=True)
source = request["source"]["xyz_text"].splitlines()
n_atoms = int(source[0])
symbols = [line.split()[0] for line in source[2:2+n_atoms]]
xyz = [[float(x) for x in line.split()[1:4]] for line in source[2:2+n_atoms]]
coord = request["coordinate"]
frames = []
for i in range(coord["n_points"]):
    target = coord["start"] + (coord["end"] - coord["start"]) * i / (coord["n_points"] - 1)
    geom = [row[:] for row in xyz]
    a, b = coord["atoms"]
    geom[b] = [geom[a][0] + target, geom[a][1], geom[a][2]]
    frame_name = f"frame_{i:04d}.xyz"
    frame_path = frames_dir / frame_name
    frame_path.write_text(str(n_atoms) + "\nfake ACP frame\n" + "".join(
        f"{symbols[j]} {geom[j][0]:.8f} {geom[j][1]:.8f} {geom[j][2]:.8f}\n" for j in range(n_atoms)), encoding="utf-8")
    frames.append({"index":i, "target_coordinate":target, "actual_coordinate":target,
        "geometry_path":"frames/" + frame_name, "scan_energy_hartree":-20.0 + i/100,
        "single_point_energy_hartree":None, "optimization_converged":True,
        "constraint_residuals":{"driver-01":0.0}, "retry_history":[],
        "optimizer_level":{"method":request["protocol"]["scan_optimizer"]["method"],
            "basis":request["protocol"]["scan_optimizer"].get("basis"),
            "dispersion":request["protocol"]["scan_optimizer"].get("dispersion"),
            "solvent_model":request["protocol"]["scan_optimizer"].get("solvent_model", "none"),
            "solvent":request["protocol"]["scan_optimizer"].get("solvent"),
            "ri_approximation":request["protocol"]["scan_optimizer"].get("ri_approximation", "none")},
        "optimizer_engine":"orca"})
profile = {"schema_version":"pes_profile_v2", "workflow":"PESsearch", "mode":"bond_length_scan",
    "status":"completed", "scan_dir":"WORK/07_PATH/pes_scan_001", "frames":frames}
(result / "pes_profile.json").write_text(json.dumps(profile), encoding="utf-8")
(result / "pes_recommendations.json").write_text(json.dumps({"schema_version":"pes_recommendations_v1"}), encoding="utf-8")
manifest = {"version":2, "task_id":"", "workflow":"PESsearch", "status":"completed", "products":[
    {"id":"pes_profile", "label":"profile", "path":"pes_search/pes_profile.json", "kind":"pes_profile"},
    {"id":"pes_recommendations", "label":"recommendations", "path":"pes_search/pes_recommendations.json", "kind":"report"}]}
(output / "RESULT" / "result_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
''', encoding="utf-8")
    return root


def test_compat_export_error_is_acp_mapping_error_family() -> None:
    assert issubclass(CompatExportError, ACPMappingError)


# ---------------------------------------------------------------------------
# Happy path: selector → freeze-shaped SINGLE_1D plan → legacy ScanPlan.
# ---------------------------------------------------------------------------
def test_single_1d_export_passes_contracts_validate_and_adapter_preflight() -> None:
    case = _ready_case()
    proposal = _propose(case)
    assert validate_v2_document(proposal) == []
    plan_v2 = _freeze_shaped_plan(proposal, case, prefer_start="R")
    assert validate_v2_document(plan_v2) == []

    plan = export_legacy_scan_plan(plan_v2, case, _config())

    assert validate_document(plan) == []
    assert dumps_document(plan)
    assert plan["status"] == "ready"
    assert plan["atom_map_ids"] == [1, 2, 3, 4]
    assert plan["n_atoms"] == 4
    assert plan["source_case_sha256"] == case["content_sha256"]
    candidate = plan["candidates"][0]
    assert candidate["coordinates"][0]["unit"] == "angstrom"
    assert len(candidate["coordinates"]) == 1
    points = candidate["coordinates"][0]["points"]
    assert 3 <= len(points) <= 101
    assert candidate["retry_policy"]["failure_policy"] == "retry_previous"
    assert candidate["method"]["engine"] in {"xtb", "orca"}

    preflight = scan_plan_to_acp_request(case, plan)
    request = preflight["scan_request"]
    assert request["mode"] == "bond_length_scan"
    assert request["coordinate"]["atoms"] == candidate["coordinates"][0]["atom_indices"]
    assert request["coordinate"]["n_points"] == len(points)
    assert abs(request["coordinate"]["start"] - points[0]) < 1e-9
    assert abs(request["coordinate"]["end"] - points[-1]) < 1e-9


def test_exported_scanplan_roundtrips_fake_acp_losslessly(tmp_path: Path) -> None:
    case = _ready_case()
    proposal = _propose(case)
    plan_v2 = _freeze_shaped_plan(proposal, case, prefer_start="R")
    plan = export_legacy_scan_plan(plan_v2, case, _config())
    assert validate_document(plan) == []
    preflight = scan_plan_to_acp_request(case, plan)
    request = preflight["scan_request"]
    points = plan["candidates"][0]["coordinates"][0]["points"]
    indices = plan["candidates"][0]["coordinates"][0]["atom_indices"]

    root = _fake_acp(tmp_path)
    backend = ACPCLIBackend(acp_root=root, python_executable=sys.executable)
    result = backend.run_scan(
        execution_id="execution-compat-1",
        attempt_id="attempt-001",
        scan_request=request,
        output_root=tmp_path / "run",
        timeout_seconds=10,
    )
    assert result.status == "completed" and result.returncode == 0

    verified = validate_cli_result(result.attempt_dir)
    frames = verified["profile"]["frames"]
    assert len(frames) == len(points)
    assert request["coordinate"]["atoms"] == indices

    execution = cli_result_to_execution_record(
        case=case,
        plan=plan,
        candidate_id=plan["candidates"][0]["candidate_id"],
        result=result,
    )
    path = collect_cli_path_bundle(
        output_dir=result.attempt_dir,
        case=case,
        plan=plan,
        execution_id=result.execution_id,
    )
    assert len(path["frames"]) == len(points)
    assert path["atom_map_ids"] == plan["atom_map_ids"]
    targets = [frame["target_coordinate"] for frame in path["frames"]]
    assert all(abs(target - point) < 1e-4 for target, point in zip(targets, points, strict=True))
    assert all(
        frame["geometry_ref"].startswith("RESULT/pes2ts/frames/")
        for frame in path["frames"]
    )

    quality = apply_scan_path_quality(plan=plan, execution=execution, path=path)
    assessment = quality["quality_assessment"]
    assert quality["path_bundle"]["status"] == "usable"
    assert assessment["status"] == "usable"
    assert assessment["checks"]["target_coordinates_match_plan"] is True
    assert assessment["checks"]["no_atom_collisions"] is True


def test_export_is_deterministic() -> None:
    case = _ready_case()
    proposal = _propose(case)
    plan_v2 = _freeze_shaped_plan(proposal, case)
    first = export_legacy_scan_plan(plan_v2, case, _config())
    second = export_legacy_scan_plan(plan_v2, case, _config())
    assert first["content_sha256"] == second["content_sha256"]
    strip = lambda doc: {key: value for key, value in doc.items() if key != "created_at"}
    assert stable_json_dumps(strip(first)) == stable_json_dumps(strip(second))


# ---------------------------------------------------------------------------
# Typed rejections: multi-coordinate, A/D kinds, non-uniform, bounds, paths.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["COUPLED_1D", "SCHEDULED_1D"])
def test_multi_coordinate_plan_is_typed_rejection_not_projection(mode: str) -> None:
    mats = _hand_materials()
    drivers = [
        {"kind": "B", "maps": [1, 2], "unit": "angstrom", "schedule_values": None, "index0": None},
        {"kind": "B", "maps": [3, 4], "unit": "angstrom", "schedule_values": None, "index0": None},
    ]
    plan_v2 = _hand_plan(mats, mode=mode, drivers=drivers)
    assert validate_v2_document(plan_v2) == []

    with pytest.raises(ACPMappingError) as excinfo:
        export_legacy_scan_plan(plan_v2, mats, _config())
    error = excinfo.value
    assert isinstance(error, CompatExportError)
    assert error.code == MULTI_COORDINATE_UNSUPPORTED
    assert "never" in str(error).lower() or "SINGLE_1D" in str(error)


@pytest.mark.parametrize("kind,maps,unit", [
    ("A", [1, 2, 3], "degree"),
    ("D", [1, 2, 3, 4], "degree"),
])
def test_ad_driver_kinds_are_typed_rejection(kind: str, maps: list[int], unit: str) -> None:
    mats = _hand_materials()
    drivers = [{"kind": kind, "maps": maps, "unit": unit, "schedule_values": None, "index0": None}]
    plan_v2 = _hand_plan(mats, drivers=drivers)
    assert validate_v2_document(plan_v2) == []

    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(plan_v2, mats, _config())
    assert excinfo.value.code == DRIVER_KIND_UNSUPPORTED
    assert kind in str(excinfo.value)


def test_nonuniform_schedule_values_are_typed_rejection() -> None:
    mats = _hand_materials()
    drivers = [{
        "kind": "B",
        "maps": [2, 3],
        "unit": "angstrom",
        "schedule_values": [0.0, 0.1, 0.9, 1.0],
        "index0": None,
    }]
    plan_v2 = _hand_plan(
        mats,
        drivers=drivers,
        lambda_values=[0.0, 1 / 3, 2 / 3, 1.0],
    )
    assert validate_v2_document(plan_v2) == []

    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(plan_v2, mats, _config())
    assert excinfo.value.code == NONUNIFORM_POINTS


def test_nonuniform_lambda_grid_is_typed_rejection() -> None:
    mats = _hand_materials()
    plan_v2 = _hand_plan(mats, lambda_values=[0.0, 0.1, 0.9, 1.0])
    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(plan_v2, mats, _config())
    assert excinfo.value.code == NONUNIFORM_POINTS


@pytest.mark.parametrize("n_points,expected", [
    (2, POINTS_OUT_OF_BOUNDS),
    (102, POINTS_OUT_OF_BOUNDS),
])
def test_point_count_bounds_are_honored(n_points: int, expected: str) -> None:
    mats = _hand_materials()
    lambda_values = [index / (n_points - 1) for index in range(n_points)]
    plan_v2 = _hand_plan(mats, lambda_values=lambda_values)
    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(plan_v2, mats, _config())
    assert excinfo.value.code == expected


@pytest.mark.parametrize("n_points", [3, 101])
def test_point_count_bounds_within_contract_pass_preflight(n_points: int) -> None:
    case = _case_from_mats(_hand_materials())
    mats = _hand_materials(content_sha256=case["content_sha256"])
    lambda_values = [index / (n_points - 1) for index in range(n_points)]
    plan_v2 = _hand_plan(mats, lambda_values=lambda_values)
    plan = export_legacy_scan_plan(plan_v2, case, _config())
    assert validate_document(plan) == []
    assert len(plan["candidates"][0]["coordinates"][0]["points"]) == n_points
    preflight = scan_plan_to_acp_request(case, plan)
    assert preflight["scan_request"]["coordinate"]["n_points"] == n_points


def test_path_candidate_is_typed_rejection() -> None:
    mats = _hand_materials()
    plan_v2 = _hand_plan(mats, candidate_kind="PathCandidateV1")
    assert validate_v2_document(plan_v2) == []
    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(plan_v2, mats, _config())
    assert excinfo.value.code == PATH_CANDIDATE_UNSUPPORTED


def test_non_frozen_plan_is_refused() -> None:
    mats = _hand_materials()
    plan_v2 = _hand_plan(mats)
    tampered = {**plan_v2, "status": "superseded"}
    tampered = seal_document(tampered)
    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(tampered, mats, _config())
    # contracts_v2 currently flags a supersedes-less superseded plan at
    # validation (optional-field gap); either typed refusal proves the
    # export never yields a ready ScanPlan from a non-frozen document.
    assert excinfo.value.code in {PLAN_NOT_FROZEN, "PLAN_DOCUMENT_INVALID"}


def test_source_case_sha256_mismatch_is_typed_refusal() -> None:
    mats = _hand_materials(content_sha256="a" * 64)
    plan_v2 = _hand_plan(mats)
    other = dict(mats)
    other["content_sha256"] = "c" * 64
    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(plan_v2, other, _config())
    assert excinfo.value.code == SOURCE_CASE_BINDING_MISMATCH


def test_atom_order_feeds_indices_and_index0_must_agree() -> None:
    mats = _hand_materials(atom_map_ids=(3, 1, 2, 4), content_sha256="d" * 64)
    plan_v2 = _hand_plan(mats)
    plan = export_legacy_scan_plan(plan_v2, mats, _config())
    assert plan["atom_map_ids"] == [3, 1, 2, 4]
    assert plan["candidates"][0]["coordinates"][0]["atom_indices"] == [2, 0]

    stale_drivers = [{
        "kind": "B",
        "maps": [2, 3],
        "unit": "angstrom",
        "schedule_values": None,
        "index0": [0, 1],
    }]
    stale_plan = _hand_plan(mats, drivers=stale_drivers, plan_id="plan-compat-stale-index")
    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(stale_plan, mats, _config())
    assert excinfo.value.code == INDEX_ORDER_MISMATCH


# ---------------------------------------------------------------------------
# needs_review: proposals never produce a ready export.
# ---------------------------------------------------------------------------
def test_needs_review_proposal_never_produces_ready_export() -> None:
    mats = _hand_materials()
    proposal = make_strategy_proposal(
        "rxn:compat-needs-review",
        "needs_review",
        reaction_id=mats["reaction_id"],
        case_id=mats["case_id"],
        split=mats["split"],
        source_case_sha256=mats["content_sha256"],
        graph_input={
            "endpoint_graph_sha256": "e" * 64,
            "normalization_version": "v1",
            "mapping_equivalence": {"map_ids": [1, 2, 3, 4], "basis": "test"},
        },
        graph_features={
            "edit_components": [],
            "context_support": {},
            "events": [],
            "typed_couplings": {},
        },
        family="HAND_TEST",
        motif_tags=[],
        rule_trace=[{"rule_id": "R_HAND_TEST"}],
        epistemic_status="endpoint_hypothesis",
        candidates=[],
        reasons=[{"code": "ROUTE_NEEDS_REVIEW", "detail": "human chemistry review required"}],
        blocking_reasons=[],
    )
    assert validate_v2_document(proposal) == []
    assert proposal["execution_eligible"] is False

    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(proposal, mats, _config())
    assert excinfo.value.code == NEEDS_REVIEW_OR_PROPOSAL_REFUSED
    assert "needs_review" in str(excinfo.value) or "review" in str(excinfo.value).lower()


def _demo24_materials(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    maps = [int(map_id) for map_id in snapshot["maps"]]
    return {
        "r_coordinates": {
            map_id: [float(x) for x in snapshot["r_coordinates"][index]]
            for index, map_id in enumerate(maps)
        },
        "p_coordinates": {
            map_id: [float(x) for x in snapshot["p_coordinates"][index]]
            for index, map_id in enumerate(maps)
        },
        "endpoint_electronic": snapshot["endpoint_electronic"],
    }


@pytest.mark.parametrize(
    "record_path",
    sorted((FIXTURE_ROOT / "records").glob("RXN_*.json")),
    ids=lambda path: path.stem,
)
def test_demo24_never_exports_ready_from_needs_review_proposals(record_path: Path) -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    pinned = str(manifest["rdkit_version"])
    if rdBase.rdkitVersion != pinned:
        pytest.fail(f"rdkit {rdBase.rdkitVersion} != fixture pin {pinned} (never-skip gate)")
    from pes2ts_core.config_loader import load_config

    snapshot = json.loads(record_path.read_text(encoding="utf-8"))
    export_materials = load_endpoint_materials_from_export(snapshot)
    bundle = rebuild_endpoint_graphs(str(snapshot["reaction_smiles"]), export_materials)
    materials = _demo24_materials(snapshot)
    proposal = propose_strategies(
        bundle,
        materials,
        load_config(),
        reaction_id=str(snapshot["reaction_id"]),
        split=str(snapshot.get("split", "unassigned")),
    )
    assert proposal["execution_eligible"] is False

    with pytest.raises(CompatExportError) as excinfo:
        export_legacy_scan_plan(proposal, materials, load_config())
    assert excinfo.value.code == NEEDS_REVIEW_OR_PROPOSAL_REFUSED
