"""Multi-coordinate / A/D / non-uniform ACP request and result projections (todo 20).

New module: the existing single-distance adapter
(``adapter.scan_plan_to_acp_request``) stays byte-frozen. This module:

- **Request payloads.** ``multicoord_request_payload`` projects a todo-19
  ``CompiledRequest`` (plus case materials and an effective capability) into an
  ACP-facing multi-coordinate request payload that carries **ALL** drivers with
  kind + atom maps + zero-based indices + units + per-point values — never only
  the first driver. Covers 2–3 coordinate simultaneous scans (``COUPLED_1D``),
  A/D kinds, and non-uniform / point-wise schedules (``SCHEDULED_1D``).
- **Result parse surface.** ``parse_multicoord_result`` →
  ``MulticoordAttemptResult``: per-frame all-driver target/actual/residual
  values with explicit completeness accounting (todo 21 extends recovery on
  this surface; a missing driver value is recorded as incomplete, never
  silently dropped and never first-driver-only).
- **Capability gate first.** Before any payload is built,
  ``capabilities.capability_check`` runs against the effective capability. The
  shipped default registry (adapter ``SINGLE_1D`` / ``B`` only, todo 16)
  refuses multi-coordinate / A/D / non-uniform requests with typed
  ``BACKEND_CAPABILITY_MISSING`` (declared-but-unprobed → ``CAPABILITY_UNPROBED``).
  Explicit refusal; no downgrade projection.

No submission: payloads are data projections only and never launch ACP.
"""

# allow: SIZE_OK — plan-named todo-20 single ACP-extension module (payload
# projection + result parse surface + capability gate); precedent
# compile_orca.py (todo 19) / compat_export.py (todo 18).

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pes2ts_core.integration.acp.adapter import ACPMappingError
from pes2ts_core.scan_strategy.capabilities import (
    BACKEND_CAPABILITY_MISSING,
    EffectiveCapability,
    capability_check,
)
from pes2ts_core.scan_strategy.compile_orca import (
    RECIPE_PER_POINT,
    CompiledCoordinate,
    CompiledRequest,
    CODE_ATOM_ORDER_INVALID,
    CODE_CAPABILITY_UNPROBED,
    CODE_COORDINATE_VALUE_INVALID,
    CODE_CUSTOM_SCHEDULE_UNSUPPORTED,
    CODE_PER_POINT_CONSTRAINTS_UNSUPPORTED,
    CODE_SIMUL_SCAN_UNSUPPORTED,
)
from pes2ts_core.utils.hashing import stable_json_dumps

__all__ = [
    "ADAPTER_VERSION_MULTICOORD",
    "CODE_BACKEND_CAPABILITY_MISSING",
    "CODE_CAPABILITY_UNPROBED",
    "CODE_COMPILED_REQUEST_INVALID",
    "CODE_FRAMES_INVALID",
    "CODE_MATERIALS_INVALID",
    "SCHEMA_MULTICOORD_REQUEST",
    "SCHEMA_MULTICOORD_RESULT",
    "MulticoordAdapterError",
    "MulticoordAttemptResult",
    "MulticoordCaseMaterials",
    "MulticoordFrame",
    "MulticoordRequestConfig",
    "driver_id_for",
    "multicoord_request_payload",
    "parse_multicoord_case_materials",
    "parse_multicoord_request_config",
    "parse_multicoord_result",
]

# ---------------------------------------------------------------------------
# Vocabulary.
# ---------------------------------------------------------------------------
ADAPTER_VERSION_MULTICOORD: str = "pes2ts_acp_multicoord_v1"
SCHEMA_MULTICOORD_REQUEST: str = "pes2ts_acp_multicoord_request_v1"
SCHEMA_MULTICOORD_RESULT: str = "pes2ts_acp_multicoord_result_v1"

CODE_BACKEND_CAPABILITY_MISSING: str = BACKEND_CAPABILITY_MISSING
CODE_COMPILED_REQUEST_INVALID: str = "COMPILED_REQUEST_INVALID"
CODE_MATERIALS_INVALID: str = "MATERIALS_INVALID"
CODE_FRAMES_INVALID: str = "FRAMES_INVALID"


class MulticoordAdapterError(ACPMappingError):
    """Typed multicoord ACP refusal; ``code`` is a stable machine-readable token.

    Inherits the ACP mapping error family so existing ``ACPMappingError``
    refusal surfaces keep catching projection failures, while ``code``
    distinguishes capability refusals from shape/binding refusals.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


# ---------------------------------------------------------------------------
# Boundary value objects (parse-don't-validate).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MulticoordCaseMaterials:
    """Whitelisted case projection consumed by the multicoord payload builder.

    Geometries are aligned to ``atom_map_ids`` order (the frozen common atom
    order), not map-sorted — matching how ``CompiledRequest.atom_rows`` and the
    ACP xyz blocks are assembled.
    """

    atom_map_ids: tuple[int, ...]
    elements: tuple[str, ...]
    r_geometry: tuple[tuple[float, float, float], ...]
    p_geometry: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int


@dataclass(frozen=True, slots=True)
class MulticoordRequestConfig:
    """Effective capability plus optional protocol hints for payload building."""

    capability: EffectiveCapability
    engine: str | None = None


def driver_id_for(index: int) -> str:
    """Stable per-driver identifier shared by payload and result surfaces."""
    return f"driver-{index + 1:02d}"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _parse_xyz_block(raw: Any, path: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise MulticoordAdapterError(CODE_MATERIALS_INVALID, f"{path} must be a non-empty coordinate array")
    rows: list[tuple[float, float, float]] = []
    for index, row in enumerate(raw):
        if isinstance(row, Mapping):
            row = row.get("coordinates")
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise MulticoordAdapterError(
                CODE_MATERIALS_INVALID, f"{path}[{index}] must be one 3D coordinate"
            )
        try:
            xyz = (float(row[0]), float(row[1]), float(row[2]))
        except (TypeError, ValueError) as exc:
            raise MulticoordAdapterError(
                CODE_MATERIALS_INVALID, f"{path}[{index}] coordinates must be numbers"
            ) from exc
        if any(not math.isfinite(component) for component in xyz):
            raise MulticoordAdapterError(
                CODE_MATERIALS_INVALID, f"{path}[{index}] coordinates must be finite"
            )
        rows.append(xyz)
    return tuple(rows)


def _parse_endpoint_charge_multiplicity(raw: Any, side: str) -> tuple[int, int]:
    if not isinstance(raw, Mapping):
        raise MulticoordAdapterError(CODE_MATERIALS_INVALID, f"endpoint {side} must be an object")
    charge = raw.get("charge", 0)
    multiplicity = raw.get("multiplicity", 1)
    if isinstance(charge, bool) or not isinstance(charge, int):
        raise MulticoordAdapterError(CODE_MATERIALS_INVALID, f"endpoint {side}.charge must be an integer")
    if isinstance(multiplicity, bool) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise MulticoordAdapterError(
            CODE_MATERIALS_INVALID, f"endpoint {side}.multiplicity must be an integer >= 1"
        )
    return charge, multiplicity


def parse_multicoord_case_materials(raw: Any) -> MulticoordCaseMaterials:
    """Parse a sealed ReactionCase or a projection mapping into materials.

    Accepted shapes:

    - **ReactionCase document**: ``atoms`` (map/element rows) plus
      ``reactant``/``product`` endpoint objects with ``geometry`` lists aligned
      to ``atoms`` order and endpoint ``charge``/``multiplicity``.
    - **Projection mapping**: ``atom_map_ids`` + ``elements`` +
      ``r_coordinates``/``reactant`` (or ``r``) + ``p_coordinates``/``product``
      (or ``p``) + ``charge``/``multiplicity`` (or ``endpoint_electronic``).
    """
    if not isinstance(raw, Mapping):
        raise MulticoordAdapterError(CODE_MATERIALS_INVALID, "case materials must be a mapping")
    if "atoms" in raw and ("reactant" in raw or "product" in raw):
        atoms = raw.get("atoms")
        if not isinstance(atoms, (list, tuple)) or not atoms:
            raise MulticoordAdapterError(CODE_MATERIALS_INVALID, "case atoms must be a non-empty array")
        map_ids: list[int] = []
        elements: list[str] = []
        for index, atom in enumerate(atoms):
            if not isinstance(atom, Mapping):
                raise MulticoordAdapterError(CODE_MATERIALS_INVALID, f"atoms[{index}] must be an object")
            map_id = atom.get("atom_map_id")
            element = atom.get("element")
            if isinstance(map_id, bool) or not isinstance(map_id, int) or map_id <= 0:
                raise MulticoordAdapterError(
                    CODE_MATERIALS_INVALID, f"atoms[{index}].atom_map_id must be a positive integer"
                )
            if not isinstance(element, str) or not element:
                raise MulticoordAdapterError(
                    CODE_MATERIALS_INVALID, f"atoms[{index}].element must be a non-empty string"
                )
            map_ids.append(map_id)
            elements.append(element)
        r_raw = raw.get("reactant")
        p_raw = raw.get("product")
        r_geometry = _parse_xyz_block(
            r_raw.get("geometry") if isinstance(r_raw, Mapping) else None, "reactant.geometry"
        )
        p_geometry = _parse_xyz_block(
            p_raw.get("geometry") if isinstance(p_raw, Mapping) else None, "product.geometry"
        )
        r_charge, r_mult = _parse_endpoint_charge_multiplicity(r_raw, "reactant")
        p_charge, p_mult = _parse_endpoint_charge_multiplicity(p_raw, "product")
        if (r_charge, r_mult) != (p_charge, p_mult):
            raise MulticoordAdapterError(
                CODE_MATERIALS_INVALID,
                "endpoint charge/multiplicity differ between R and P — electronic-state "
                "routing belongs upstream, not in the ACP projection",
            )
        charge, multiplicity = r_charge, r_mult
    else:
        raw_maps = raw.get("atom_map_ids")
        raw_elements = raw.get("elements")
        if not isinstance(raw_maps, (list, tuple)) or not raw_maps:
            raise MulticoordAdapterError(
                CODE_MATERIALS_INVALID, "projection materials require a non-empty atom_map_ids array"
            )
        if not isinstance(raw_elements, (list, tuple)) or not raw_elements:
            raise MulticoordAdapterError(
                CODE_MATERIALS_INVALID, "projection materials require a non-empty elements array"
            )
        map_ids = []
        for index, map_id in enumerate(raw_maps):
            if isinstance(map_id, bool) or not isinstance(map_id, int) or map_id <= 0:
                raise MulticoordAdapterError(
                    CODE_MATERIALS_INVALID, f"atom_map_ids[{index}] must be a positive integer"
                )
            map_ids.append(map_id)
        elements = []
        for index, element in enumerate(raw_elements):
            if not isinstance(element, str) or not element:
                raise MulticoordAdapterError(
                    CODE_MATERIALS_INVALID, f"elements[{index}] must be a non-empty string"
                )
            elements.append(element)
        r_key = next((key for key in ("r_coordinates", "reactant", "r") if key in raw), None)
        p_key = next((key for key in ("p_coordinates", "product", "p") if key in raw), None)
        if r_key is None or p_key is None:
            raise MulticoordAdapterError(
                CODE_MATERIALS_INVALID,
                "projection materials require both R and P geometry blocks "
                "(r_coordinates/reactant and p_coordinates/product)",
            )
        r_geometry = _parse_xyz_block(raw[r_key], str(r_key))
        p_geometry = _parse_xyz_block(raw[p_key], str(p_key))
        electronic = raw.get("endpoint_electronic")
        if isinstance(electronic, Mapping):
            reactant_side = electronic.get("reactant", electronic.get("R"))
            product_side = electronic.get("product", electronic.get("P"))
            r_charge, r_mult = _parse_endpoint_charge_multiplicity(reactant_side, "reactant")
            p_charge, p_mult = _parse_endpoint_charge_multiplicity(product_side, "product")
            if (r_charge, r_mult) != (p_charge, p_mult):
                raise MulticoordAdapterError(
                    CODE_MATERIALS_INVALID, "endpoint_electronic R/P charge or multiplicity differ"
                )
            charge, multiplicity = r_charge, r_mult
        else:
            charge_raw = raw.get("charge", 0)
            multiplicity_raw = raw.get("multiplicity", 1)
            if isinstance(charge_raw, bool) or not isinstance(charge_raw, int):
                raise MulticoordAdapterError(
                    CODE_MATERIALS_INVALID, "materials.charge must be an integer"
                )
            if (isinstance(multiplicity_raw, bool) or not isinstance(multiplicity_raw, int)
                    or multiplicity_raw < 1):
                raise MulticoordAdapterError(
                    CODE_MATERIALS_INVALID, "materials.multiplicity must be an integer >= 1"
                )
            charge, multiplicity = charge_raw, multiplicity_raw
    if len(set(map_ids)) != len(map_ids):
        raise MulticoordAdapterError(CODE_MATERIALS_INVALID, "atom_map_ids must be unique")
    if len(elements) != len(map_ids):
        raise MulticoordAdapterError(
            CODE_MATERIALS_INVALID,
            f"elements has {len(elements)} entries but atom_map_ids has {len(map_ids)}",
        )
    if len(r_geometry) != len(map_ids) or len(p_geometry) != len(map_ids):
        raise MulticoordAdapterError(
            CODE_MATERIALS_INVALID,
            "R and P geometries must each contain exactly one coordinate per atom_map_id",
        )
    return MulticoordCaseMaterials(
        atom_map_ids=tuple(map_ids),
        elements=tuple(elements),
        r_geometry=r_geometry,
        p_geometry=p_geometry,
        charge=charge,
        multiplicity=multiplicity,
    )


def parse_multicoord_request_config(raw: Any) -> MulticoordRequestConfig:
    """Parse the config boundary: an ``EffectiveCapability`` or a mapping."""
    if isinstance(raw, MulticoordRequestConfig):
        return raw
    if isinstance(raw, EffectiveCapability):
        return MulticoordRequestConfig(capability=raw)
    if not isinstance(raw, Mapping):
        raise MulticoordAdapterError(
            CODE_COMPILED_REQUEST_INVALID,
            "config must be an EffectiveCapability or a mapping with a capability key",
        )
    capability = raw.get("capability")
    if not isinstance(capability, EffectiveCapability):
        raise MulticoordAdapterError(
            CODE_COMPILED_REQUEST_INVALID,
            "config.capability must be an EffectiveCapability from capabilities.effective_capability",
        )
    engine = raw.get("engine")
    if engine is not None and (not isinstance(engine, str) or not engine):
        raise MulticoordAdapterError(
            CODE_COMPILED_REQUEST_INVALID, "config.engine must be null or a non-empty string"
        )
    return MulticoordRequestConfig(capability=capability, engine=str(engine) if engine else None)


# ---------------------------------------------------------------------------
# Capability gate (before any projection; never optimistic).
# ---------------------------------------------------------------------------
def _capability_gate(capability: EffectiveCapability, compiled: CompiledRequest) -> None:
    """Refuse beyond the effective capability; explicit codes, no downgrade."""
    kinds = [row.kind for row in compiled.coordinates]
    n_coords = len(kinds)
    check = capability_check(capability, compiled.mode, kinds, n_coords)
    if check["status"] != "pass":
        missing = "; ".join(str(item) for item in check.get("missing") or ())
        detail = missing or f"mode {compiled.mode!r} is not supported by the effective capability"
        if check["status"] == "unknown":
            raise MulticoordAdapterError(
                CODE_CAPABILITY_UNPROBED,
                f"{detail} — declared but unprobed modes can never project to a ready request",
            )
        raise MulticoordAdapterError(CODE_BACKEND_CAPABILITY_MISSING, detail)
    # Projection-layer shape requirements beyond mode/kind/arity vocabulary.
    if compiled.simultaneous and not capability.constraint_support.simul_scan:
        raise MulticoordAdapterError(
            CODE_SIMUL_SCAN_UNSUPPORTED,
            "compiled request is simultaneous but effective constraint_support.simul_scan is false",
        )
    if compiled.recipe_kind == RECIPE_PER_POINT:
        if not capability.custom_schedule_support:
            raise MulticoordAdapterError(
                CODE_CUSTOM_SCHEDULE_UNSUPPORTED,
                "compiled per-point recipe requires effective custom_schedule_support",
            )
        if not capability.constraint_support.per_point_constraints:
            raise MulticoordAdapterError(
                CODE_PER_POINT_CONSTRAINTS_UNSUPPORTED,
                "compiled per-point recipe requires effective per_point_constraints",
            )


# ---------------------------------------------------------------------------
# Request payload projection.
# ---------------------------------------------------------------------------
def _validate_compiled(compiled: Any) -> CompiledRequest:
    if not isinstance(compiled, CompiledRequest):
        raise MulticoordAdapterError(
            CODE_COMPILED_REQUEST_INVALID,
            "compiled must be a scan_strategy.compile_orca.CompiledRequest artifact",
        )
    if not compiled.coordinates:
        raise MulticoordAdapterError(
            CODE_COMPILED_REQUEST_INVALID,
            "compiled request carries no scan coordinates — PATH/NEB artifacts are not "
            "multicoord scan payloads (todo 26 owns the path request protocol)",
        )
    for row in compiled.coordinates:
        if not isinstance(row, CompiledCoordinate):
            raise MulticoordAdapterError(
                CODE_COMPILED_REQUEST_INVALID, "compiled.coordinates entries must be CompiledCoordinate"
            )
        if len(row.values) != compiled.total_points:
            raise MulticoordAdapterError(
                CODE_COMPILED_REQUEST_INVALID,
                f"driver {row.kind}{list(row.maps)} has {len(row.values)} values but "
                f"total_points={compiled.total_points}",
            )
    return compiled


def _driver_record(row: CompiledCoordinate, index: int) -> dict[str, Any]:
    """One payload driver record: kind + maps + indices + units + per-point values."""
    return {
        "coordinate_id": driver_id_for(index),
        "kind": row.kind,
        "atom_map_ids": list(row.maps),
        "atom_indices": list(row.indices),
        "unit": row.unit,
        "start_value": row.start_value,
        "end_value": row.end_value,
        "values": list(row.values),
        "schedule_values": list(row.schedule_values),
    }


def _xyz_text(
    geometry: Sequence[Sequence[float]],
    elements: Sequence[str],
    comment: str,
) -> str:
    lines = [str(len(elements)), comment]
    lines.extend(
        f"{symbol} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}"
        for symbol, xyz in zip(elements, geometry, strict=True)
    )
    return "\n".join(lines) + "\n"


def multicoord_request_payload(
    compiled: Any,
    case_materials: Any,
    config: Any,
) -> dict[str, Any]:
    """Project a compiled multi-coordinate scan into an ACP request payload.

    The payload carries **every** driver of ``compiled.coordinates`` with kind,
    atom maps, zero-based indices, units, and per-point values — never only the
    first driver. The capability gate runs before any projection: the shipped
    default registry (adapter SINGLE_1D / B only) refuses with typed
    ``BACKEND_CAPABILITY_MISSING``; declared-but-unprobed combinations refuse
    with ``CAPABILITY_UNPROBED``. This function builds data only and never
    submits a job.
    """
    compiled_request = _validate_compiled(compiled)
    materials = (
        case_materials
        if isinstance(case_materials, MulticoordCaseMaterials)
        else parse_multicoord_case_materials(case_materials)
    )
    request_config = parse_multicoord_request_config(config)
    _capability_gate(request_config.capability, compiled_request)
    if compiled_request.atom_rows != materials.atom_map_ids:
        raise MulticoordAdapterError(
            CODE_ATOM_ORDER_INVALID,
            f"compiled atom_rows {list(compiled_request.atom_rows)} do not match case materials "
            f"atom_map_ids {list(materials.atom_map_ids)} — indices resolve through the common "
            "atom order, never map−1",
        )
    recipe = compiled_request.recipe if isinstance(compiled_request.recipe, Mapping) else {}
    start_endpoint = str(recipe.get("start_endpoint") or "R")
    if start_endpoint not in ("R", "P"):
        raise MulticoordAdapterError(
            CODE_COORDINATE_VALUE_INVALID,
            f"compiled recipe start_endpoint={start_endpoint!r}; expected 'R' or 'P'",
        )
    geometry = materials.r_geometry if start_endpoint == "R" else materials.p_geometry
    charge = recipe.get("charge", materials.charge)
    multiplicity = recipe.get("multiplicity", materials.multiplicity)
    xyz_text = _xyz_text(
        geometry, materials.elements, f"PES2TS multicoord {compiled_request.candidate_id}"
    )
    coordinates = [
        _driver_record(row, index) for index, row in enumerate(compiled_request.coordinates)
    ]
    driver_ids = [record["coordinate_id"] for record in coordinates]
    payload: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION_MULTICOORD,
        "schema_version": SCHEMA_MULTICOORD_REQUEST,
        "metadata": {
            "candidate_id": compiled_request.candidate_id,
            "mode": compiled_request.mode,
            "compiled_kind": compiled_request.compiled_kind,
            "recipe_kind": compiled_request.recipe_kind,
            "compile_schema_version": compiled_request.schema_version,
            "simultaneous": compiled_request.simultaneous,
            "total_points": compiled_request.total_points,
            "n_drivers": len(coordinates),
            "driver_ids": driver_ids,
            "atom_map_ids": list(materials.atom_map_ids),
            "elements": list(materials.elements),
            "start_endpoint": start_endpoint,
            "charge": charge,
            "multiplicity": multiplicity,
            "method": recipe.get("method"),
            "nprocs": recipe.get("nprocs"),
            "geometry_sha256": recipe.get("geometry_sha256"),
        },
        "source": {
            "source_type": "xyz_text",
            "xyz_text": xyz_text,
            "charge": charge,
            "multiplicity": multiplicity,
        },
        "coordinates": coordinates,
        "lambda_values": list(compiled_request.lambda_values),
        "point_input_sha256": list(compiled_request.point_input_sha256),
        "geom_fragments": list(compiled_request.geom_fragments),
        "recipe": dict(recipe) if isinstance(recipe, Mapping) else {},
        "protocol": {
            "scan_type": "multicoordinate_scan",
            "mode": compiled_request.mode,
            "n_drivers": len(coordinates),
            "simultaneous": compiled_request.simultaneous,
            "recipe_kind": compiled_request.recipe_kind,
            "custom_schedule": compiled_request.recipe_kind == RECIPE_PER_POINT,
            "name": "pes2ts_acp_multicoord_v1",
        },
    }
    if request_config.engine is not None:
        payload["protocol"]["scan_driver"] = {
            "software": request_config.engine,
            "mode": (
                "per_point_constraints"
                if compiled_request.recipe_kind == RECIPE_PER_POINT
                else "relaxed_scan"
            ),
        }
    return payload


# ---------------------------------------------------------------------------
# Result parse surface (todo 21 extends recovery on this contract).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MulticoordFrame:
    """One parsed attempt frame with per-driver target/actual/residual values.

    ``missing_driver_ids`` names drivers whose target or actual value is
    absent — explicit incompleteness, never a silent drop and never a
    first-driver projection. Residuals are optional per native scan
    semantics (absent residual → ``None``, not incomplete by itself).
    """

    frame_index: int
    converged: bool | None
    target_by_driver: dict[str, float | None]
    actual_by_driver: dict[str, float | None]
    residual_by_driver: dict[str, float | None]
    scan_energy_hartree: float | None
    single_point_energy_hartree: float | None
    geometry_path: str
    retry_history: tuple[dict[str, Any], ...]
    missing_driver_ids: tuple[str, ...]
    complete: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "converged": self.converged,
            "target_by_driver": dict(self.target_by_driver),
            "actual_by_driver": dict(self.actual_by_driver),
            "residual_by_driver": dict(self.residual_by_driver),
            "scan_energy_hartree": self.scan_energy_hartree,
            "single_point_energy_hartree": self.single_point_energy_hartree,
            "geometry_path": self.geometry_path,
            "retry_history": [dict(item) for item in self.retry_history],
            "missing_driver_ids": list(self.missing_driver_ids),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class MulticoordAttemptResult:
    """Parsed multicoord attempt: every driver recovered on every frame.

    ``complete`` is true only when every parsed frame carries target and
    actual values for every driver. Zero frames is explicitly incomplete.
    """

    schema_version: str
    candidate_id: str
    mode: str
    execution_id: str
    acp_task_id: str | None
    driver_ids: tuple[str, ...]
    driver_units: dict[str, str]
    n_frames: int
    complete: bool
    frames: tuple[MulticoordFrame, ...]

    def to_doc(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "execution_id": self.execution_id,
            "acp_task_id": self.acp_task_id,
            "driver_ids": list(self.driver_ids),
            "driver_units": dict(self.driver_units),
            "n_frames": self.n_frames,
            "complete": self.complete,
            "frames": [frame.to_record() for frame in self.frames],
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_doc())


def _driver_value_map(raw: Any, driver_ids: Sequence[str], path: str) -> dict[str, float | None]:
    """Parse a per-driver value mapping; missing drivers become None entries."""
    if raw is None:
        return {driver_id: None for driver_id in driver_ids}
    if not isinstance(raw, Mapping):
        raise MulticoordAdapterError(
            CODE_FRAMES_INVALID, f"{path} must be null or an object keyed by driver id"
        )
    out: dict[str, float | None] = {}
    for driver_id in driver_ids:
        value = _finite_number(raw.get(driver_id))
        if driver_id in raw and raw.get(driver_id) is not None and value is None:
            raise MulticoordAdapterError(
                CODE_FRAMES_INVALID, f"{path}[{driver_id}] must be null or a finite number"
            )
        out[driver_id] = value
    return out


def parse_multicoord_result(
    *,
    compiled: Any,
    execution_id: str,
    acp_task_id: str | None,
    raw_frames: Sequence[Any],
) -> MulticoordAttemptResult:
    """Parse per-frame all-driver actual values from a multicoord attempt.

    ``raw_frames`` is the ACP profile-like frame list. Each frame may carry
    ``target_coordinates`` / ``actual_coordinates`` / ``constraint_residuals``
    as ``{driver_id: value|None}`` objects (driver ids from
    :func:`driver_id_for`, matching ``multicoord_request_payload``). A frame
    missing any driver's target or actual value is marked incomplete —
    todo 21 extends recovery (retry history, monitors, diagnostics) on this
    surface without changing its refusal semantics.
    """
    compiled_request = _validate_compiled(compiled)
    if not isinstance(execution_id, str) or not execution_id:
        raise MulticoordAdapterError(CODE_FRAMES_INVALID, "execution_id must be a non-empty string")
    if not isinstance(raw_frames, (list, tuple)):
        raise MulticoordAdapterError(CODE_FRAMES_INVALID, "raw_frames must be an array")
    driver_ids = tuple(driver_id_for(index) for index in range(len(compiled_request.coordinates)))
    driver_units = {driver_id: row.unit for driver_id, row in zip(
        driver_ids, compiled_request.coordinates, strict=True
    )}
    parsed: list[MulticoordFrame] = []
    seen_indices: set[int] = set()
    for position, raw in enumerate(raw_frames):
        if not isinstance(raw, Mapping):
            raise MulticoordAdapterError(CODE_FRAMES_INVALID, f"raw_frames[{position}] must be an object")
        index = raw.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise MulticoordAdapterError(
                CODE_FRAMES_INVALID, f"raw_frames[{position}].index must be a non-negative integer"
            )
        if index in seen_indices:
            raise MulticoordAdapterError(
                CODE_FRAMES_INVALID, f"raw_frames[{position}].index={index} is duplicated"
            )
        seen_indices.add(index)
        converged_raw = raw.get("optimization_converged", raw.get("converged"))
        if converged_raw is not None and not isinstance(converged_raw, bool):
            raise MulticoordAdapterError(
                CODE_FRAMES_INVALID, f"raw_frames[{position}].converged must be null or a boolean"
            )
        retry_history_raw = raw.get("retry_history", [])
        if not isinstance(retry_history_raw, (list, tuple)):
            raise MulticoordAdapterError(
                CODE_FRAMES_INVALID, f"raw_frames[{position}].retry_history must be an array"
            )
        retry_history: list[dict[str, Any]] = []
        for retry_index, retry in enumerate(retry_history_raw):
            if not isinstance(retry, Mapping):
                raise MulticoordAdapterError(
                    CODE_FRAMES_INVALID,
                    f"raw_frames[{position}].retry_history[{retry_index}] must be an object",
                )
            retry_history.append(dict(retry))
        targets = _driver_value_map(
            raw.get("target_coordinates"), driver_ids, f"raw_frames[{position}].target_coordinates"
        )
        actuals = _driver_value_map(
            raw.get("actual_coordinates"), driver_ids, f"raw_frames[{position}].actual_coordinates"
        )
        residuals = _driver_value_map(
            raw.get("constraint_residuals"), driver_ids,
            f"raw_frames[{position}].constraint_residuals",
        )
        missing = tuple(
            driver_id
            for driver_id in driver_ids
            if targets.get(driver_id) is None or actuals.get(driver_id) is None
        )
        geometry_path = raw.get("geometry_path", "")
        if geometry_path is None:
            geometry_path = ""
        if not isinstance(geometry_path, str):
            raise MulticoordAdapterError(
                CODE_FRAMES_INVALID, f"raw_frames[{position}].geometry_path must be a string"
            )
        parsed.append(
            MulticoordFrame(
                frame_index=index,
                converged=converged_raw,
                target_by_driver=targets,
                actual_by_driver=actuals,
                residual_by_driver=residuals,
                scan_energy_hartree=_finite_number(raw.get("scan_energy_hartree")),
                single_point_energy_hartree=_finite_number(
                    raw.get("single_point_energy_hartree", raw.get("refined_energy_hartree"))
                ),
                geometry_path=geometry_path,
                retry_history=tuple(retry_history),
                missing_driver_ids=missing,
                complete=not missing,
            )
        )
    parsed.sort(key=lambda frame: frame.frame_index)
    complete = bool(parsed) and all(frame.complete for frame in parsed)
    return MulticoordAttemptResult(
        schema_version=SCHEMA_MULTICOORD_RESULT,
        candidate_id=compiled_request.candidate_id,
        mode=compiled_request.mode,
        execution_id=execution_id,
        acp_task_id=acp_task_id,
        driver_ids=driver_ids,
        driver_units=driver_units,
        n_frames=len(parsed),
        complete=complete,
        frames=tuple(parsed),
    )
