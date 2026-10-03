"""X3'-B result-plane projection tests (ADR-0001/0002).

Covers:

* ``trajectory_record_to_path_bundle`` — a canonical ``TrajectoryRecord``
  (hand-built to mirror ``XtbPathACPBackend.run`` output, and once driven
  through the backend's fake-ACP seam) projects into a sealed v1
  ``PathBundle`` with the contract's method/frames/status/provenance shape,
  unit-honest energy re-mapping, and zero evaluation/label keys.
* The quality/ranking plane accepts the projected bundle: the frozen v1
  contract validators pass, ``integration/acp/quality.py`` classifies it
  without raising, and ``rank_path_bundle`` consumes its frames.
* ``build_unified_gate`` / ``write_unified_gate`` — the additive
  ``unified_gate.json`` mapping the xTB population to ``XTB_PATH``: pure,
  deterministic, and never rewriting ``g2_eligible.json`` /
  ``g2_scan_ready.json``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from pes2ts_core.contracts import (
    FORBIDDEN_TRUTH_KEYS,
    ContractError,
    dumps_document,
    loads_document,
    make_document,
    validate_document,
)
from pes2ts_core.generation.execution.protocol import METHOD_XTB_PATH
from pes2ts_core.generation.execution.record import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    TrajectoryFrame,
    TrajectoryRecord,
)
from pes2ts_core.generation.execution.xtb_path.projection import (
    PROJECTION_EXTENSION_KEY,
    SCHEMA_UNIFIED_GATE,
    ProjectionError,
    build_unified_gate,
    trajectory_record_to_path_bundle,
    write_unified_gate,
)
from pes2ts_core.integration.acp.trajectory import HARTREE_TO_KCAL_MOL
from pes2ts_core.ranking import rank_path_bundle
from pes2ts_core.utils.hashing import stable_json_dumps
from pes2ts_core.utils.jsonio import read_json

REACTION_ID = "RXN_0000000001"
CASE_ID = "case:xtb-path-projection-v1"
PLAN_ID = "plan:xtb-path-projection-v1"
CANDIDATE_ID = "candidate:xtb-path-1"
EXECUTION_ID = "execution:xtb-path-1"
ACP_TASK_ID = "acp-task:xtb-path-1"
PLAN_SHA256 = "ab" * 32
REQUEST_SHA256 = "cd" * 32
RAW_TRAJ_SHA256 = "ef" * 32
MANIFEST_SHA256 = "12" * 32
#: Native seam energies (relative kcal/mol) as the XTB_PATH backend records them.
NATIVE_ENERGIES = (0.0, 2.5, 1.0)
FRAME_GEOMETRIES = (
    ((0.0, 0.0, 0.0), (0.74, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.12, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.50, 0.0, 0.0)),
)


def _make_case(reaction_id: str = REACTION_ID) -> dict:
    return make_document(
        "ReactionCase",
        CASE_ID if reaction_id == REACTION_ID else f"case:{reaction_id}",
        "ready",
        dataset_version="synthetic-v1",
        reaction_id=reaction_id,
        case_id=CASE_ID if reaction_id == REACTION_ID else f"case:{reaction_id}",
        split="train",
        atoms=[
            {"atom_map_id": 1, "element": "H"},
            {"atom_map_id": 2, "element": "H"},
        ],
        reactant={
            "charge": 0,
            "multiplicity": 1,
            "geometry": [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
        },
        product={
            "charge": 0,
            "multiplicity": 1,
            "geometry": [[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]],
        },
        edits=[{"kind": "order_changed", "atom_map_ids": [1, 2]}],
        hydrogen_transfers=[],
        source={"dataset": "synthetic", "mapping_provenance": "endpoint_only"},
    )


def _make_record(
    *,
    status: str = STATUS_COMPLETED,
    method: str = METHOD_XTB_PATH,
    energies=NATIVE_ENERGIES,
    native_unit: str = "relative_kcal_per_mol",
    with_geometry: bool = True,
    plan_sha256: str | None = PLAN_SHA256,
) -> TrajectoryRecord:
    frames = []
    for index, energy in enumerate(energies):
        geometry = FRAME_GEOMETRIES[index % len(FRAME_GEOMETRIES)] if with_geometry else None
        frames.append(
            TrajectoryFrame(
                frame_index=index,
                energy_hartree=energy,
                geometry=geometry,
                converged=True,
            )
        )
    if status == STATUS_FAILED:
        frames = []
    acp_block = {
        "direction": "forward",
        "returncode": 0 if status != STATUS_FAILED else 9,
        "timed_out": False,
        "request_sha256": REQUEST_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "acp_execution_id": "pes2ts-xb-path-0000",
        "acp_attempt_id": "attempt-0000",
        "wall_seconds": 12.5,
        "acp_status": "completed" if status != STATUS_FAILED else "failed",
        "acp_reused": False,
        "acp_attempt_dir": "C:/scratch/run/WORK",
        "acp_log_ref": "WORK/acp.log",
        "acp_error": None,
        "native_frame_energy_unit": native_unit,
        "raw_trajectory_path": "C:/scratch/run/RESULT/pes_search/xtbpath.xyz",
        "raw_trajectory_sha256": RAW_TRAJ_SHA256 if status != STATUS_FAILED else None,
        "executable_sha256": None,
        "argv": None,
        "xtb_version_line": None,
        "seed_supported": None,
        "seed": None,
        "omp_num_threads": None,
    }
    if status == STATUS_FAILED:
        acp_block["failure_code"] = "G2_XTB_FAILED"
    provenance = {
        "acp": acp_block,
        "request_sha256": REQUEST_SHA256,
        "raw_trajectory_sha256": RAW_TRAJ_SHA256 if status != STATUS_FAILED else None,
    }
    return TrajectoryRecord(
        reaction_id=REACTION_ID,
        method=method,
        status=status,
        frames=tuple(frames),
        plan_sha256=plan_sha256,
        provenance=provenance,
    )


def _plan_ref(**overrides) -> dict:
    ref = {
        "plan_id": PLAN_ID,
        "candidate_id": CANDIDATE_ID,
        "execution_id": EXECUTION_ID,
        "acp_task_id": ACP_TASK_ID,
        "atom_map_ids": [1, 2],
    }
    ref.update(overrides)
    return ref


def _make_plan(case: dict, *, candidate_id: str = CANDIDATE_ID) -> dict:
    points = [0.74, 1.12, 1.50]
    return make_document(
        "ScanPlan",
        PLAN_ID,
        "ready",
        plan_id=PLAN_ID,
        experiment_id="experiment:xtb-path-projection-v1",
        dataset_version="synthetic-v1",
        reaction_id=case["reaction_id"],
        case_id=case["case_id"],
        split="train",
        plan_version=1,
        atom_map_ids=[1, 2],
        n_atoms=2,
        candidate_strategy="one_dimensional",
        source_case_sha256=case["content_sha256"],
        plan_frozen=True,
        candidates=[
            {
                "candidate_id": candidate_id,
                "coordinates": [
                    {
                        "coordinate_id": "coord-1",
                        "role": "driver",
                        "unit": "angstrom",
                        "atom_map_ids": [1, 2],
                        "atom_indices": [0, 1],
                        "points": points,
                    }
                ],
            }
        ],
    )


def _make_execution(case: dict, plan: dict, *, candidate_id: str = CANDIDATE_ID) -> dict:
    return make_document(
        "ExecutionRecord",
        EXECUTION_ID,
        "completed",
        reaction_id=case["reaction_id"],
        case_id=case["case_id"],
        plan_id=plan["plan_id"],
        candidate_id=candidate_id,
        request_id="request:xtb-path-1",
        acp_task_id=ACP_TASK_ID,
        attempts=[
            {"attempt_id": "attempt:xtb-path-1", "status": "completed", "cpu_seconds": 3.0}
        ],
        total_cpu_seconds=3.0,
        cost_complete=True,
        failure_retained=False,
        work_ref="WORK/pes2ts",
        result_ref="RESULT/pes2ts",
    )


def _extension(bundle: dict) -> dict:
    return bundle["extensions"][PROJECTION_EXTENSION_KEY]


def _collect_keys(value, out: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            out.add(str(key).lower())
            _collect_keys(child, out)
    elif isinstance(value, list):
        for item in value:
            _collect_keys(item, out)


# ---------------------------------------------------------------------------
# trajectory_record_to_path_bundle
# ---------------------------------------------------------------------------
def test_completed_record_projects_to_valid_path_bundle() -> None:
    record = _make_record()
    bundle = trajectory_record_to_path_bundle(record, case=_make_case(), plan_ref=_plan_ref())

    assert validate_document(bundle) == []
    assert bundle["schema_name"] == "PathBundle"
    assert bundle["reaction_id"] == REACTION_ID
    assert bundle["case_id"] == CASE_ID
    assert bundle["plan_id"] == PLAN_ID
    assert bundle["candidate_id"] == CANDIDATE_ID
    assert bundle["execution_id"] == EXECUTION_ID
    assert bundle["acp_task_id"] == ACP_TASK_ID
    assert bundle["atom_map_ids"] == [1, 2]
    assert bundle["n_atoms"] == 2
    assert bundle["status"] == "unchecked"
    assert bundle["supersedes"] is None

    frames = bundle["frames"]
    assert len(frames) == len(NATIVE_ENERGIES)
    for index, (frame, native_energy, geometry) in enumerate(
        zip(frames, NATIVE_ENERGIES, FRAME_GEOMETRIES, strict=True)
    ):
        assert frame["frame_index"] == index
        assert frame["frame_id"] == f"{EXECUTION_ID}:f{index:05d}"
        assert frame["atom_map_ids"] == [1, 2]
        assert frame["elements"] == ["H", "H"]
        assert frame["geometry"] == [list(row) for row in geometry]
        channel = frame["energies"]["scan_electronic"]
        assert channel["unit"] == "hartree"
        assert channel["method_id"] == METHOD_XTB_PATH
        assert channel["value"] == pytest.approx(native_energy / HARTREE_TO_KCAL_MOL)

    # Method / plan binding / provenance live in the namespaced extension.
    ext = _extension(bundle)
    assert ext["method"] == METHOD_XTB_PATH
    assert ext["trajectory_status"] == STATUS_COMPLETED
    assert ext["plan_sha256"] == PLAN_SHA256
    provenance = ext["provenance"]
    assert provenance["request_sha256"] == REQUEST_SHA256
    assert provenance["raw_trajectory_sha256"] == RAW_TRAJ_SHA256
    acp = provenance["acp"]
    assert acp["acp_execution_id"] == "pes2ts-xb-path-0000"
    assert acp["acp_attempt_id"] == "attempt-0000"
    assert acp["direction"] == "forward"
    assert acp["native_frame_energy_unit"] == "relative_kcal_per_mol"
    assert acp["manifest_sha256"] == MANIFEST_SHA256
    assert isinstance(acp["acp_block_sha256"], str) and len(acp["acp_block_sha256"]) == 64
    # Volatile scheduler metadata is excluded from the sealed document.
    assert "wall_seconds" not in acp
    assert "acp_attempt_dir" not in acp
    assert "raw_trajectory_path" not in acp

    # Sealed-document round trip (contract validators + digest).
    assert loads_document(dumps_document(bundle))["content_sha256"] == bundle["content_sha256"]


@pytest.mark.parametrize(
    "record_status, expected_bundle_status",
    [
        (STATUS_COMPLETED, "unchecked"),
        (STATUS_PARTIAL, "needs_review"),
        (STATUS_FAILED, "unusable"),
    ],
)
def test_status_mapping(record_status: str, expected_bundle_status: str) -> None:
    record = _make_record(status=record_status)
    bundle = trajectory_record_to_path_bundle(record, case=_make_case(), plan_ref=_plan_ref())

    assert bundle["status"] == expected_bundle_status
    assert _extension(bundle)["trajectory_status"] == record_status
    assert validate_document(bundle) == []
    if record_status == STATUS_FAILED:
        assert bundle["frames"] == []


def test_energy_unit_remap_documents_conversion() -> None:
    record = _make_record(energies=(627.5094740631,), native_unit="relative_kcal_per_mol")
    # Single-frame record: reuse the two-atom geometry.
    bundle = trajectory_record_to_path_bundle(record, case=_make_case(), plan_ref=_plan_ref())

    channel = bundle["frames"][0]["energies"]["scan_electronic"]
    assert channel["value"] == pytest.approx(1.0)
    remap = _extension(bundle)["energy_unit_remap"]
    assert remap["native_frame_energy_unit"] == "relative_kcal_per_mol"
    assert remap["contract_energy_unit"] == "hartree"
    assert remap["converted"] is True
    assert remap["hartree_per_kcal_mol"] == HARTREE_TO_KCAL_MOL

    hartree_record = _make_record(energies=(1.25,), native_unit="hartree")
    hartree_bundle = trajectory_record_to_path_bundle(
        hartree_record, case=_make_case(), plan_ref=_plan_ref()
    )
    assert hartree_bundle["frames"][0]["energies"]["scan_electronic"]["value"] == pytest.approx(1.25)
    assert _extension(hartree_bundle)["energy_unit_remap"]["converted"] is False


def test_projection_carries_no_evaluation_keys() -> None:
    record = _make_record()
    bundle = trajectory_record_to_path_bundle(record, case=_make_case(), plan_ref=_plan_ref())

    seen: set[str] = set()
    _collect_keys(bundle, seen)
    assert not (seen & FORBIDDEN_TRUTH_KEYS), seen & FORBIDDEN_TRUTH_KEYS
    # The contract validator agrees (production-input truth scan).
    assert validate_document(bundle, production_input=True) == []


def test_plan_ref_none_stays_deterministic_and_contract_valid() -> None:
    record = _make_record()
    first = trajectory_record_to_path_bundle(record, case=_make_case())
    second = trajectory_record_to_path_bundle(record, case=_make_case())

    assert validate_document(first) == []
    assert first["content_sha256"] == second["content_sha256"]
    assert first["plan_id"] == f"plan:{PLAN_SHA256}"
    assert first["candidate_id"] == REACTION_ID
    assert first["execution_id"] == "pes2ts-xb-path-0000"
    assert first["acp_task_id"] is None
    assert first["frames"] == second["frames"]


def test_projection_is_deterministic_for_identical_inputs() -> None:
    record = _make_record()
    case = _make_case()
    ref = _plan_ref()
    first = trajectory_record_to_path_bundle(record, case=case, plan_ref=ref)
    second = trajectory_record_to_path_bundle(record, case=case, plan_ref=ref)

    assert first["content_sha256"] == second["content_sha256"]
    assert stable_json_dumps({k: v for k, v in first.items() if k != "created_at"}) == stable_json_dumps(
        {k: v for k, v in second.items() if k != "created_at"}
    )


def test_projection_rejects_case_reaction_mismatch() -> None:
    record = _make_record()
    other_case = _make_case(reaction_id="RXN_OTHER")
    with pytest.raises(ProjectionError, match="does not match"):
        trajectory_record_to_path_bundle(record, case=other_case, plan_ref=_plan_ref())


def test_projection_rejects_missing_geometry() -> None:
    record = _make_record(with_geometry=False)
    with pytest.raises(ProjectionError, match="no geometry"):
        trajectory_record_to_path_bundle(record, case=_make_case(), plan_ref=_plan_ref())


def test_projection_rejects_unknown_record_status() -> None:
    record = _make_record()
    broken = TrajectoryRecord(
        reaction_id=record.reaction_id,
        method=record.method,
        status="sideways",
        frames=record.frames,
        plan_sha256=record.plan_sha256,
        provenance=record.provenance,
    )
    with pytest.raises(ProjectionError, match="record.status"):
        trajectory_record_to_path_bundle(broken, case=_make_case(), plan_ref=_plan_ref())


def test_projection_rejects_plan_ref_atom_map_mismatch() -> None:
    record = _make_record()
    with pytest.raises(ProjectionError, match="atom_map_ids"):
        trajectory_record_to_path_bundle(
            record, case=_make_case(), plan_ref=_plan_ref(atom_map_ids=[2, 1])
        )


def test_projection_rejects_invalid_case() -> None:
    record = _make_record()
    case = _make_case()
    tampered = copy.deepcopy(case)
    tampered["atoms"][0]["atom_map_id"] = 99
    tampered = {**tampered}  # keep sealed digest stale on purpose
    with pytest.raises(ProjectionError, match="ReactionCase"):
        trajectory_record_to_path_bundle(record, case=tampered, plan_ref=_plan_ref())


# ---------------------------------------------------------------------------
# Quality / ranking plane consumption.
# ---------------------------------------------------------------------------
def test_contract_validators_and_ranking_accept_projected_bundle() -> None:
    record = _make_record()
    case = _make_case()
    plan = _make_plan(case)
    execution = _make_execution(case, plan)
    bundle = trajectory_record_to_path_bundle(record, case=case, plan_ref=_plan_ref())

    # 1. Frozen v1 contract validators accept the projection.
    assert validate_document(bundle) == []
    reloaded = loads_document(dumps_document(bundle))
    assert reloaded["object_id"] == bundle["object_id"]

    # 2. The ranking plane consumes the projected frames.
    proposal = rank_path_bundle(bundle, rule="highest_scan_energy")
    assert proposal["status"] == "needs_review"  # projected status is unchecked
    assert proposal["path_status"] == "unchecked"
    assert proposal["selected_frames"]
    assert proposal["selected_frames"][0]["frame_id"] == f"{EXECUTION_ID}:f00001"
    assert proposal["selected_frames"][0]["score"] == pytest.approx(
        max(NATIVE_ENERGIES) / HARTREE_TO_KCAL_MOL
    )
    assert validate_document(proposal) == []


def test_acp_quality_layer_accepts_projected_bundle() -> None:
    from pes2ts_core.integration.acp.quality import (
        apply_scan_path_quality,
        assess_scan_path_quality,
    )

    record = _make_record()
    case = _make_case()
    plan = _make_plan(case)
    execution = _make_execution(case, plan)
    bundle = trajectory_record_to_path_bundle(record, case=case, plan_ref=_plan_ref())

    # XTB_PATH frames carry no frozen scan target coordinates, so the ACP
    # scan-quality layer must ACCEPT the document (no ContractError) and
    # classify it as needs_review — never as physically validated.
    report = assess_scan_path_quality(plan=plan, execution=execution, path=bundle)
    assert report["status"] == "needs_review"
    assert report["physical_validation"] == "not_run"
    assert report["observed_frame_count"] == len(NATIVE_ENERGIES)
    assert report["input_sha256"]["path"] == bundle["content_sha256"]

    applied = apply_scan_path_quality(plan=plan, execution=execution, path=bundle)
    assert applied["path_bundle"]["status"] == "needs_review"
    assert applied["path_bundle"]["content_sha256"] != bundle["content_sha256"]
    assert validate_document(applied["path_bundle"]) == []


def test_backend_produced_record_projects_without_plan_ref(tmp_path: Path) -> None:
    """Project a record that ``XtbPathACPBackend.run`` produced via fake ACP."""
    from test_g2_acp_backend import _fake_acp, _materials, _plan

    from pes2ts_core.generation.execution.xtb_path.acp_backend import XtbPathACPBackend

    root = _fake_acp(tmp_path)
    backend = XtbPathACPBackend()
    record = backend.run(_plan(), _materials(tmp_path, root))

    assert record.status == STATUS_COMPLETED
    assert record.method == METHOD_XTB_PATH
    assert record.provenance["acp"]["native_frame_energy_unit"] == "relative_kcal_per_mol"

    bundle = trajectory_record_to_path_bundle(record, case=_make_case())

    assert validate_document(bundle) == []
    assert bundle["status"] == "unchecked"
    assert _extension(bundle)["method"] == METHOD_XTB_PATH
    assert _extension(bundle)["plan_sha256"] == "ab" * 32
    assert _extension(bundle)["provenance"]["request_sha256"] == (
        record.provenance["request_sha256"]
    )
    assert _extension(bundle)["provenance"]["raw_trajectory_sha256"] == (
        record.provenance["raw_trajectory_sha256"]
    )
    acp = _extension(bundle)["provenance"]["acp"]
    assert acp["acp_execution_id"] == record.provenance["acp"]["acp_execution_id"]
    assert acp["direction"] == "forward"
    # Native kcal/mol values are re-mapped into the contract hartree channel.
    for frame, native in zip(bundle["frames"], NATIVE_ENERGIES, strict=True):
        assert frame["energies"]["scan_electronic"]["value"] == pytest.approx(
            native / HARTREE_TO_KCAL_MOL
        )
    seen: set[str] = set()
    _collect_keys(bundle, seen)
    assert not (seen & FORBIDDEN_TRUTH_KEYS)


# ---------------------------------------------------------------------------
# Unified gate (additive X3' projection).
# ---------------------------------------------------------------------------
def test_build_unified_gate_maps_population_to_xtb_path() -> None:
    ids = ["RXN_0000000001", "RXN_0000000002", "RXN_0000000003"]
    gate = build_unified_gate(eligible_ids=ids, plan_ref={"plan_id": PLAN_ID, "plan_sha256": PLAN_SHA256})

    assert gate["schema_version"] == SCHEMA_UNIFIED_GATE
    assert gate["method"] == METHOD_XTB_PATH
    assert gate["n_ids"] == 3
    assert gate["ids"] == ids
    assert gate["ids_sha256"] == hashlib.sha256(
        stable_json_dumps(ids).encode("utf-8")
    ).hexdigest()
    assert gate["plan_ref"] == {"plan_id": PLAN_ID, "plan_sha256": PLAN_SHA256}
    assert gate["supersedes"] is None
    assert gate["additive"] is True


def test_build_unified_gate_defaults_method_and_is_deterministic() -> None:
    ids = ["RXN_A", "RXN_B"]
    first = build_unified_gate(eligible_ids=ids)
    second = build_unified_gate(eligible_ids=ids)

    assert first["method"] == METHOD_XTB_PATH
    assert first == second
    assert first["plan_ref"] is None


def test_build_unified_gate_is_additive_and_never_touches_legacy_lists(
    tmp_path: Path,
) -> None:
    legacy_eligible = tmp_path / "g2_eligible.json"
    legacy_scan_ready = tmp_path / "g2_scan_ready.json"
    legacy_eligible_payload = {
        "schema_version": "g1_manifest_v1",
        "reaction_ids": ["RXN_0000000001", "RXN_0000000002"],
    }
    legacy_scan_ready_payload = {
        "schema_version": "g1_v2_gate_v1",
        "reaction_ids": ["RXN_0000000001"],
        "n_eligible": 1,
    }
    legacy_eligible.write_text(
        stable_json_dumps(legacy_eligible_payload), encoding="utf-8"
    )
    legacy_scan_ready.write_text(
        stable_json_dumps(legacy_scan_ready_payload), encoding="utf-8"
    )
    eligible_bytes = legacy_eligible.read_bytes()
    scan_ready_bytes = legacy_scan_ready.read_bytes()

    ids = ["RXN_0000000001", "RXN_0000000002"]
    gate = build_unified_gate(eligible_ids=ids)
    gate_path = tmp_path / "unified_gate.json"
    write_unified_gate(gate_path, gate)

    # The gate is a separate additive artifact.
    assert gate_path.is_file()
    written = json.loads(gate_path.read_text(encoding="utf-8"))
    assert written == gate
    assert written["supersedes"] is None
    assert written["additive"] is True
    # Legacy G2 lists are byte-identical: the gate never rewrites them.
    assert legacy_eligible.read_bytes() == eligible_bytes
    assert legacy_scan_ready.read_bytes() == scan_ready_bytes
    # The existing G2 reader shape is untouched.
    assert json.loads(legacy_eligible.read_text(encoding="utf-8"))[
        "reaction_ids"
    ] == legacy_eligible_payload["reaction_ids"]


def test_build_unified_gate_preserves_population_order_and_binds_digest() -> None:
    ids = ["RXN_0000000003", "RXN_0000000001", "RXN_0000000002"]
    gate = build_unified_gate(eligible_ids=ids)

    assert gate["ids"] == ids  # caller order preserved, never re-sorted
    assert gate["n_ids"] == len(ids)
    reordered = build_unified_gate(eligible_ids=list(reversed(ids)))
    assert reordered["ids_sha256"] != gate["ids_sha256"]


@pytest.mark.parametrize(
    "eligible_ids, match",
    [
        (["RXN_1", "RXN_1"], "duplicate"),
        (["RXN_1", ""], "non-empty"),
        (["RXN_1", 2], "non-empty"),
        ("RXN_1", "sequence"),
    ],
)
def test_build_unified_gate_rejects_bad_populations(eligible_ids, match: str) -> None:
    with pytest.raises(ProjectionError, match=match):
        build_unified_gate(eligible_ids=eligible_ids)


def test_build_unified_gate_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="unknown execution method"):
        build_unified_gate(eligible_ids=["RXN_1"], method="NOT_A_METHOD")


def test_write_unified_gate_round_trip_and_validation(tmp_path: Path) -> None:
    ids = ["RXN_0000000001"]
    gate = build_unified_gate(eligible_ids=ids)
    path = tmp_path / "nested" / "unified_gate.json"
    write_unified_gate(path, gate)

    assert read_json(path) == gate

    # Atomic overwrite leaves a valid document, not a partial write.
    gate2 = build_unified_gate(eligible_ids=ids + ["RXN_0000000002"])
    write_unified_gate(path, gate2)
    assert read_json(path) == gate2
    assert not list(path.parent.glob("*.tmp-*"))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda g: {**g, "schema_version": "wrong"}, "schema_version"),
        (lambda g: {**g, "method": "NOPE"}, "unknown execution method"),
        (lambda g: {**g, "n_ids": 99}, "n_ids"),
        (lambda g: {**g, "ids": []}, "n_ids"),
        (lambda g: {**g, "ids_sha256": "0" * 64}, "ids_sha256"),
        (lambda g: {**g, "supersedes": "g2_eligible.json"}, "additive"),
        (lambda g: {**g, "additive": False}, "additive"),
    ],
)
def test_write_unified_gate_rejects_malformed_gate(tmp_path: Path, mutate, match: str) -> None:
    gate = build_unified_gate(eligible_ids=["RXN_1"])
    with pytest.raises(ProjectionError, match=match):
        write_unified_gate(tmp_path / "unified_gate.json", mutate(gate))


def test_contract_error_from_bad_case_document_is_typed() -> None:
    record = _make_record()
    with pytest.raises(ProjectionError, match="ReactionCase"):
        trajectory_record_to_path_bundle(record, case={"schema_name": "ReactionCase"})


def test_dumps_document_rejects_tampered_projected_bundle() -> None:
    record = _make_record()
    bundle = trajectory_record_to_path_bundle(record, case=_make_case(), plan_ref=_plan_ref())
    tampered = copy.deepcopy(bundle)
    tampered["frames"][0]["energies"]["scan_electronic"]["value"] += 0.5
    with pytest.raises(ContractError, match="content_sha256"):
        dumps_document(tampered)
