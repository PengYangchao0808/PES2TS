"""Result-plane projection of the canonical TrajectoryRecord (ADR-0002 X3'-B).

The ranking/quality plane consumes frozen v1 contract objects
(``PathBundle`` / ``SeedProposal`` / ``ValidationResult``), while every
generation backend returns one canonical :class:`TrajectoryRecord`
(ADR-0001 execution seam).  This module is the pure bridge between the two:

* :func:`trajectory_record_to_path_bundle` projects a ``TrajectoryRecord``
  plus its ReactionCase / plan reference into a sealed v1 ``PathBundle``
  document, following the projection style of
  ``integration/acp/adapter.py::acp_result_to_path_bundle`` and
  ``integration/acp/trajectory.py``.  It is chemistry-free: no
  evaluation/label fields are ever derived, and the frozen v1 contract
  validators reject any that appear.
* :func:`build_unified_gate` / :func:`write_unified_gate` emit the additive
  ``unified_gate.json`` artifact that maps the xTB-eligible population to the
  ``XTB_PATH`` method.  The gate is a **superset projection only**: it never
  reads, rewrites, or replaces the legacy G2 lists ``g2_eligible.json`` /
  ``g2_scan_ready.json``, which the existing G2 reader keeps consuming
  unchanged.

Determinism / provenance contract (ADR-0002 hashes-not-state):

* ``plan_sha256`` and the provenance digests (``request_sha256`` /
  ``manifest_sha256`` / ``raw_trajectory_sha256``) are carried into the
  projected document so scientific content is bound to its inputs.
* Volatile scheduler metadata inside ``TrajectoryRecord.provenance["acp"]``
  (machine-local attempt directories, log refs, wall time) is **not** embedded
  in the sealed ``PathBundle``; only the ACP block's identity fields, typed
  outcome fields, and an ``acp_block_sha256`` digest over the full block are
  projected, so two projections of the same scientific content stay
  byte-comparable apart from the contract ``created_at`` stamp.

Unit honesty (X3' re-map, no guessing): the XTB_PATH backend stores parsed
``xtbpath.xyz`` comment energies (relative kcal/mol) into the seam field
``TrajectoryFrame.energy_hartree`` and records the native unit in
``provenance["acp"]["native_frame_energy_unit"]``.  This projection re-maps
them into the v1 contract's ``hartree`` energy channel using that recorded
native unit; the remap itself is documented in the bundle extensions.

This module performs no I/O: it never opens geometry files (frames carry
inline coordinates, the shape the v1 contract requires) and the unified-gate
writer delegates atomic publication to :mod:`pes2ts_core.utils.jsonio`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pes2ts_core.contracts import ContractError, dumps_document, make_document
from pes2ts_core.generation.execution.protocol import METHOD_XTB_PATH, validate_method
from pes2ts_core.generation.execution.record import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    TrajectoryRecord,
)
from pes2ts_core.integration.acp.trajectory import HARTREE_TO_KCAL_MOL
from pes2ts_core.utils.hashing import stable_json_dumps
from pes2ts_core.utils.jsonio import write_json

#: Namespaced extension key carrying method/provenance for the projected bundle.
PROJECTION_EXTENSION_KEY: Final[str] = "pes2ts.trajectory_projection.v1"
#: Schema version stamped into every ``unified_gate.json`` artifact.
SCHEMA_UNIFIED_GATE: Final[str] = "pes2ts_unified_gate_v1"
#: Native frame-energy unit recorded by the XTB_PATH ACP backend.
NATIVE_UNIT_RELATIVE_KCAL: Final[str] = "relative_kcal_per_mol"
#: Contract energy unit demanded by the v1 PathBundle frame channel.
CONTRACT_ENERGY_UNIT: Final[str] = "hartree"
#: Energy channel name the v1 ranking/quality plane consumes.
ENERGY_CHANNEL: Final[str] = "scan_electronic"
#: Record status -> v1 PathBundle status.  A completed trajectory is projected
#: ``unchecked``: the projection never claims usability; the quality plane
#: decides ``usable``/``needs_review``/``unusable`` afterwards.
STATUS_MAP: Final[Mapping[str, str]] = {
    STATUS_COMPLETED: "unchecked",
    STATUS_PARTIAL: "needs_review",
    STATUS_FAILED: "unusable",
}
#: Provenance keys lifted verbatim from ``record.provenance``.
_PROVENANCE_DIGEST_KEYS: Final[tuple[str, ...]] = (
    "request_sha256",
    "raw_trajectory_sha256",
)
#: ACP-block identity/outcome keys projected into the sealed bundle.  Volatile
#: scheduler metadata (attempt dirs, log refs, wall time, free-text errors) is
#: deliberately excluded and only bound via ``acp_block_sha256``.
_ACP_IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "acp_execution_id",
    "acp_attempt_id",
    "direction",
    "acp_status",
    "native_frame_energy_unit",
    "failure_code",
    "manifest_sha256",
    "returncode",
    "timed_out",
    "acp_reused",
)


class ProjectionError(ValueError):
    """The TrajectoryRecord / case / plan reference cannot be projected."""


def _require_record(record: TrajectoryRecord) -> TrajectoryRecord:
    if not isinstance(record, TrajectoryRecord):
        raise ProjectionError("record must be a TrajectoryRecord")
    if not isinstance(record.reaction_id, str) or not record.reaction_id:
        raise ProjectionError("record.reaction_id must be a non-empty string")
    if not isinstance(record.method, str) or not record.method:
        raise ProjectionError("record.method must be a non-empty string")
    if record.status not in STATUS_MAP:
        raise ProjectionError(
            f"record.status {record.status!r} is not one of {tuple(STATUS_MAP)}"
        )
    return record


def _validated_case(case: Mapping[str, Any], *, reaction_id: str) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        raise ProjectionError("case must be a mapping")
    document = dict(case)
    try:
        dumps_document(document)
    except ContractError as exc:
        raise ProjectionError(f"ReactionCase: {exc}") from exc
    if document.get("reaction_id") != reaction_id:
        raise ProjectionError(
            f"case.reaction_id {document.get('reaction_id')!r} does not match "
            f"record.reaction_id {reaction_id!r}"
        )
    atoms = document.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ProjectionError("case.atoms must be a non-empty array")
    atom_map_ids: list[int] = []
    elements: list[str] = []
    for index, atom in enumerate(atoms):
        if not isinstance(atom, Mapping):
            raise ProjectionError(f"case.atoms[{index}] must be an object")
        atom_map_id = atom.get("atom_map_id")
        element = atom.get("element")
        if isinstance(atom_map_id, bool) or not isinstance(atom_map_id, int) or atom_map_id < 1:
            raise ProjectionError(f"case.atoms[{index}].atom_map_id must be a positive integer")
        if not isinstance(element, str) or not element:
            raise ProjectionError(f"case.atoms[{index}].element must be a non-empty string")
        atom_map_ids.append(atom_map_id)
        elements.append(element)
    if len(set(atom_map_ids)) != len(atom_map_ids):
        raise ProjectionError("case.atoms atom_map_id values must be unique")
    return {
        "document": document,
        "case_id": document["case_id"],
        "atom_map_ids": atom_map_ids,
        "elements": elements,
    }


def _plan_ref_field(plan_ref: Mapping[str, Any] | None, key: str) -> Any:
    if plan_ref is None:
        return None
    if not isinstance(plan_ref, Mapping):
        raise ProjectionError("plan_ref must be a mapping or None")
    return plan_ref.get(key)


def _json_safe_plan_ref(plan_ref: Mapping[str, Any] | None) -> Any:
    """Return a deterministic JSON-safe copy of *plan_ref* (or ``None``)."""
    if plan_ref is None:
        return None
    if not isinstance(plan_ref, Mapping):
        raise ProjectionError("plan_ref must be a mapping or None")
    try:
        return json.loads(stable_json_dumps(dict(plan_ref)))
    except (TypeError, ValueError) as exc:
        raise ProjectionError(f"plan_ref must be JSON-serializable: {exc}") from exc


def _candidate_id_from_plan_ref(plan_ref: Mapping[str, Any] | None) -> str | None:
    candidate_id = _plan_ref_field(plan_ref, "candidate_id")
    if isinstance(candidate_id, str) and candidate_id:
        return candidate_id
    if plan_ref is None or not isinstance(plan_ref, Mapping):
        return None
    candidates = plan_ref.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, Mapping):
            nested = first.get("candidate_id")
            if isinstance(nested, str) and nested:
                return nested
    return None


def _acp_identity_projection(record: TrajectoryRecord) -> dict[str, Any]:
    """Project ``provenance["acp"]`` to identity/outcome fields + block digest."""
    provenance = record.provenance if isinstance(record.provenance, Mapping) else {}
    acp_source = provenance.get("acp")
    identity: dict[str, Any] = {}
    if isinstance(acp_source, Mapping):
        for key in _ACP_IDENTITY_KEYS:
            if key in acp_source:
                identity[key] = acp_source[key]
        identity["acp_block_sha256"] = hashlib.sha256(
            stable_json_dumps(dict(acp_source)).encode("utf-8")
        ).hexdigest()
    return identity


def _native_frame_energy_unit(record: TrajectoryRecord) -> str:
    provenance = record.provenance if isinstance(record.provenance, Mapping) else {}
    acp_source = provenance.get("acp")
    if isinstance(acp_source, Mapping):
        unit = acp_source.get("native_frame_energy_unit")
        if isinstance(unit, str) and unit:
            return unit
    return CONTRACT_ENERGY_UNIT


def _remap_energy(value: float, native_unit: str) -> float:
    """Map a seam energy value into the v1 contract's hartree channel."""
    if native_unit == NATIVE_UNIT_RELATIVE_KCAL:
        return value / HARTREE_TO_KCAL_MOL
    return value


def _project_frames(
    record: TrajectoryRecord,
    *,
    execution_id: str,
    atom_map_ids: list[int],
    elements: list[str],
    native_unit: str,
) -> list[dict[str, Any]]:
    n_atoms = len(atom_map_ids)
    projected: list[dict[str, Any]] = []
    for index, frame in enumerate(record.frames):
        if frame.geometry is None:
            raise ProjectionError(
                f"frame {index} carries no geometry; PathBundle frames require inline coordinates"
            )
        geometry = [list(row) for row in frame.geometry]
        if len(geometry) != n_atoms or any(len(row) != 3 for row in geometry):
            raise ProjectionError(
                f"frame {index} geometry must be {n_atoms}x3 coordinates in case atom order"
            )
        for row in geometry:
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ProjectionError(f"frame {index} geometry coordinates must be numeric")
        energies: dict[str, Any] = {}
        if frame.energy_hartree is not None:
            if isinstance(frame.energy_hartree, bool) or not isinstance(
                frame.energy_hartree, (int, float)
            ):
                raise ProjectionError(f"frame {index} energy_hartree must be numeric or null")
            energies[ENERGY_CHANNEL] = {
                "value": _remap_energy(float(frame.energy_hartree), native_unit),
                "unit": CONTRACT_ENERGY_UNIT,
                "method_id": record.method,
            }
        projected.append(
            {
                "frame_id": f"{execution_id}:f{index:05d}",
                "frame_index": index,
                "atom_map_ids": list(atom_map_ids),
                "elements": list(elements),
                "geometry": geometry,
                "geometry_ref": "",
                "energies": energies,
                "converged": frame.converged,
            }
        )
    return projected


def trajectory_record_to_path_bundle(
    record: TrajectoryRecord,
    *,
    case: Mapping[str, Any],
    plan_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one canonical TrajectoryRecord into a sealed v1 PathBundle.

    Parameters
    ----------
    record:
        The execution-seam record returned by an ``ExecutionBackend``
        (for example ``XtbPathACPBackend.run``).
    case:
        The sealed v1 ``ReactionCase`` the record executed.  Its
        ``reaction_id`` must match the record; its ``atoms`` supply the
        ``atom_map_ids`` / ``elements`` order every projected frame carries.
    plan_ref:
        Optional mapping supplying plan identity: ``plan_id``,
        ``candidate_id`` (or a ScanPlan-style ``candidates[0].candidate_id``),
        ``execution_id``, ``acp_task_id``, and optionally ``atom_map_ids``
        cross-checked against the case.  Missing fields fall back to
        deterministic record-derived identities so the projection stays pure
        and total.

    Returns
    -------
    dict
        A sealed ``PathBundle`` document accepted by
        ``pes2ts_core.contracts.validate_document`` / ``dumps_document``.
        The ``method``, ``plan_sha256``, and provenance (including the ACP
        block identity + digest) live under the namespaced extension
        ``pes2ts.trajectory_projection.v1``; frame energies are re-mapped to
        the contract ``hartree`` channel using the recorded native unit.
        No evaluation/label fields are derived or copied.
    """
    record = _require_record(record)
    case_info = _validated_case(case, reaction_id=record.reaction_id)
    if plan_ref is not None and not isinstance(plan_ref, Mapping):
        raise ProjectionError("plan_ref must be a mapping or None")

    plan_maps = _plan_ref_field(plan_ref, "atom_map_ids")
    if plan_maps is not None and list(plan_maps) != case_info["atom_map_ids"]:
        raise ProjectionError("plan_ref.atom_map_ids does not match case atom order")

    plan_sha256 = record.plan_sha256
    if not isinstance(plan_sha256, str) or not plan_sha256:
        plan_sha256 = None
    plan_id = _plan_ref_field(plan_ref, "plan_id")
    if not isinstance(plan_id, str) or not plan_id:
        plan_id = f"plan:{plan_sha256}" if plan_sha256 else f"plan:{record.reaction_id}"
    candidate_id = _candidate_id_from_plan_ref(plan_ref)
    if candidate_id is None:
        # No ScanPlan candidate identity in the execution seam: the projected
        # bundle binds to the executed reaction itself (deterministic, never a
        # silent default that could collide with a real candidate id).
        candidate_id = record.reaction_id

    provenance = record.provenance if isinstance(record.provenance, Mapping) else {}
    acp_identity = _acp_identity_projection(record)
    execution_id = _plan_ref_field(plan_ref, "execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        execution_id = acp_identity.get("acp_execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        execution_id = f"exec:{plan_sha256}" if plan_sha256 else f"exec:{record.reaction_id}"
    acp_task_id = _plan_ref_field(plan_ref, "acp_task_id")
    if acp_task_id is not None and (
        not isinstance(acp_task_id, str) or not acp_task_id
    ):
        raise ProjectionError("plan_ref.acp_task_id must be a non-empty string or None")

    native_unit = _native_frame_energy_unit(record)
    frames = _project_frames(
        record,
        execution_id=execution_id,
        atom_map_ids=case_info["atom_map_ids"],
        elements=case_info["elements"],
        native_unit=native_unit,
    )

    provenance_projection: dict[str, Any] = {"acp": acp_identity}
    for key in _PROVENANCE_DIGEST_KEYS:
        provenance_projection[key] = provenance.get(key)

    remapped = native_unit == NATIVE_UNIT_RELATIVE_KCAL
    extensions = {
        PROJECTION_EXTENSION_KEY: {
            "method": record.method,
            "trajectory_status": record.status,
            "trajectory_schema_version": record.schema_version,
            "plan_sha256": plan_sha256,
            "provenance": provenance_projection,
            "energy_unit_remap": {
                "native_frame_energy_unit": native_unit,
                "contract_energy_unit": CONTRACT_ENERGY_UNIT,
                "converted": remapped,
                "hartree_per_kcal_mol": HARTREE_TO_KCAL_MOL if remapped else None,
            },
        }
    }

    try:
        return make_document(
            "PathBundle",
            f"path:{execution_id}",
            STATUS_MAP[record.status],
            reaction_id=record.reaction_id,
            case_id=case_info["case_id"],
            plan_id=plan_id,
            candidate_id=candidate_id,
            execution_id=execution_id,
            acp_task_id=acp_task_id,
            atom_map_ids=case_info["atom_map_ids"],
            n_atoms=len(case_info["atom_map_ids"]),
            energy_reference="relative_hartree" if remapped else CONTRACT_ENERGY_UNIT,
            frames=frames,
            supersedes=None,
            extensions=extensions,
        )
    except ContractError as exc:
        raise ProjectionError(f"projected PathBundle violates the v1 contract: {exc}") from exc


def build_unified_gate(
    *,
    eligible_ids: Sequence[str],
    method: str = METHOD_XTB_PATH,
    plan_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the additive ``unified_gate.json`` document (X3' unified gate).

    Maps the xTB-eligible reaction population onto the ``XTB_PATH`` execution
    method so the ranking/quality plane can consume the unified execution seam
    **without** touching the legacy G2 lists.  This function is pure: it never
    opens ``g2_eligible.json`` / ``g2_scan_ready.json`` and never rewrites
    them; the existing G2 reader keeps consuming those files unchanged.  The
    gate is therefore strictly additive — a superset projection alongside the
    legacy lists, never a replacement.

    Parameters
    ----------
    eligible_ids:
        Ordered reaction-id population (callers pass the canonical sorted
        list, e.g. the ids from ``g2_eligible.json`` / ``g2_scan_ready.json``).
        Order is preserved and bound by ``ids_sha256``.
    method:
        Execution method the population is mapped to.  Defaults to
        ``XTB_PATH``; must be a known seam method.
    plan_ref:
        Optional JSON-safe mapping carrying plan identity / provenance for the
        gate (for example ``plan_id`` and ``plan_sha256``).

    Returns
    -------
    dict
        ``{"schema_version", "method", "plan_ref", "n_ids", "ids",
        "ids_sha256", "supersedes", "additive"}``.  ``supersedes`` is always
        ``None`` and ``additive`` is always ``True``: the gate supersedes no
        legacy artifact.
    """
    validate_method(method)
    if isinstance(eligible_ids, (str, bytes)) or not isinstance(eligible_ids, Sequence):
        raise ProjectionError("eligible_ids must be a sequence of reaction-id strings")
    ids: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(eligible_ids):
        if not isinstance(raw, str) or not raw:
            raise ProjectionError(f"eligible_ids[{index}] must be a non-empty string")
        if raw in seen:
            raise ProjectionError(f"eligible_ids contains a duplicate id: {raw!r}")
        seen.add(raw)
        ids.append(raw)
    gate_plan_ref = _json_safe_plan_ref(plan_ref)
    ids_sha256 = hashlib.sha256(stable_json_dumps(ids).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_UNIFIED_GATE,
        "method": method,
        "plan_ref": gate_plan_ref,
        "n_ids": len(ids),
        "ids": ids,
        "ids_sha256": ids_sha256,
        "supersedes": None,
        "additive": True,
    }


def write_unified_gate(path: str | Path, gate: Mapping[str, Any]) -> None:
    """Atomically publish *gate* as ``unified_gate.json`` via ``utils/jsonio``.

    Validates the gate shape first so a malformed document is never published.
    The write is atomic (temp file + ``os.replace``); a failure leaves any
    pre-existing target untouched.  Legacy G2 list files are never written by
    this function.
    """
    if not isinstance(gate, Mapping):
        raise ProjectionError("gate must be a mapping")
    if gate.get("schema_version") != SCHEMA_UNIFIED_GATE:
        raise ProjectionError(
            f"gate.schema_version must be {SCHEMA_UNIFIED_GATE!r}, got {gate.get('schema_version')!r}"
        )
    method = gate.get("method")
    if not isinstance(method, str) or not method:
        raise ProjectionError("gate.method must be a non-empty string")
    try:
        validate_method(method)
    except ValueError as exc:
        raise ProjectionError(str(exc)) from exc
    n_ids = gate.get("n_ids")
    ids = gate.get("ids")
    if isinstance(n_ids, bool) or not isinstance(n_ids, int) or n_ids < 0:
        raise ProjectionError("gate.n_ids must be a non-negative integer")
    if not isinstance(ids, list) or len(ids) != n_ids:
        raise ProjectionError("gate.ids must be a list matching gate.n_ids")
    ids_sha256 = gate.get("ids_sha256")
    if not isinstance(ids_sha256, str) or ids_sha256 != hashlib.sha256(
        stable_json_dumps(ids).encode("utf-8")
    ).hexdigest():
        raise ProjectionError("gate.ids_sha256 does not match gate.ids")
    if gate.get("supersedes") is not None or gate.get("additive") is not True:
        raise ProjectionError("gate must stay additive (supersedes=None, additive=True)")
    write_json(path, dict(gate))


__all__ = [
    "CONTRACT_ENERGY_UNIT",
    "ENERGY_CHANNEL",
    "NATIVE_UNIT_RELATIVE_KCAL",
    "PROJECTION_EXTENSION_KEY",
    "SCHEMA_UNIFIED_GATE",
    "STATUS_MAP",
    "ProjectionError",
    "build_unified_gate",
    "trajectory_record_to_path_bundle",
    "write_unified_gate",
]
