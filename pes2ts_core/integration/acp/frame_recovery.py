"""Per-frame ALL-driver target/actual/residual/monitor recovery (todo 21).

Design §11.1 (``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``):
every frame records ``λ / stage_id / geometry_ref / energy_channel /
convergence / all_driver_targets / all_driver_actuals / residuals /
monitor_values / non_target_contacts / electronic_state_diagnostics /
retry_history``.  Single-point refinement and scan energies live in separate
channels.

New module — the frozen single-distance surfaces (``trajectory.py``,
``quality.py``, ``contracts.py`` PathBundle validation, ``adapter.py``) are
**not** modified.  This module:

- **Consumes the todo-20 parse surface.**  ``recover_frames`` takes a
  ``MulticoordAttemptResult`` (every driver parsed per frame), the todo-19
  ``CompiledRequest`` (schedule-λ targets per driver per point), and optional
  coordinate-pool monitor records.  Targets come from the compiled schedule;
  actuals come from the parsed frame; residuals are computed as
  ``target − actual`` for **every** driver — never the first driver only.
- **Incomplete marking.**  A frame missing ANY driver's residual (target or
  actual absent, so the residual cannot be computed) is marked ``incomplete``
  and listed in ``incomplete_frame_indices`` — never silently passed.  The
  per-frame ``missing_driver_ids`` keeps todo-20 semantics (target or actual
  absent).
- **Monitor / contact / diagnostics / retry channels.**  Monitor quantities
  are measured from per-frame geometry for every supplied pool record (B
  distance, A angle, D dihedral); non-target contacts are nonbonded atom
  pairs below a threshold that are not part of any driver coordinate;
  electronic-state diagnostics is an explicit placeholder structured channel;
  retry history is carried through per frame.
- **PathBundle v2 extension.**  ``to_path_bundle_extension`` emits an
  additive block for ``PathBundle.extensions["pes2ts.frame_recovery.v2"]``.
  Old single-driver field semantics stay intact (``target_coordinate`` /
  ``actual_coordinate`` / ``constraint_residuals`` keep their historical
  meaning, derived from the first driver when several are present);
  multi-driver data is added under new named fields only — additive, never an
  overwrite.

No submission: recovery is data projection over already-parsed attempt
results and never launches ACP/ORCA.
"""

# allow: SIZE_OK — plan-named todo-21 single frame-recovery module (design
# §11.1 per-frame surface + PathBundle v2 extension); precedent
# multicoord.py (todo 20) / compile_orca.py (todo 19).

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.integration.acp.multicoord import (
    MulticoordAttemptResult,
    MulticoordFrame,
    driver_id_for,
)
from pes2ts_core.scan_strategy.compile_orca import (
    CompiledCoordinate,
    CompiledRequest,
)
from pes2ts_core.scan_strategy.coordinate_pool import (
    KIND_A,
    KIND_B,
    KIND_D,
)
from pes2ts_core.scan_strategy.geometry_feasibility import (
    bond_angle_deg,
    dihedral_deg,
)
from pes2ts_core.utils.hashing import stable_json_dumps

__all__ = [
    "CODE_COMPILED_INVALID",
    "CODE_DRIVER_COVERAGE",
    "CODE_DRIVER_MISMATCH",
    "CODE_FRAME_INDEX_INVALID",
    "CODE_GEOMETRY_INVALID",
    "CODE_MONITOR_INVALID",
    "CODE_RESULT_INVALID",
    "CODE_THRESHOLD_INVALID",
    "DEFAULT_CONTACT_THRESHOLD_ANGSTROM",
    "ENERGY_CHANNEL_NONE",
    "ENERGY_CHANNEL_SCAN",
    "ENERGY_CHANNEL_SINGLE_POINT",
    "PATH_BUNDLE_EXTENSION_KEY",
    "SCHEMA_FRAME_RECOVERY",
    "SCHEMA_PATH_BUNDLE_EXTENSION",
    "ContactRecord",
    "EnergyChannelRow",
    "FrameRecoveryError",
    "FrameRecoveryRecord",
    "FrameRecoveryReport",
    "MonitorRecord",
    "MonitorSpec",
    "recover_frames",
    "to_path_bundle_extension",
]

# ---------------------------------------------------------------------------
# Vocabulary.
# ---------------------------------------------------------------------------
SCHEMA_FRAME_RECOVERY: Final[str] = "pes2ts_acp_frame_recovery_v1"
SCHEMA_PATH_BUNDLE_EXTENSION: Final[str] = "pes2ts_path_bundle_extension_v2"
PATH_BUNDLE_EXTENSION_KEY: Final[str] = "pes2ts.frame_recovery.v2"

ENERGY_CHANNEL_SCAN: Final[str] = "scan"
ENERGY_CHANNEL_SINGLE_POINT: Final[str] = "single_point"
ENERGY_CHANNEL_NONE: Final[str] = "none"

#: Non-target near-contact threshold (Å).  Distinct from quality.py's
#: ``collision_threshold_angstrom`` (0.45): contacts here flag unintended
#: close approaches, not hard overlaps.
DEFAULT_CONTACT_THRESHOLD_ANGSTROM: Final[float] = 0.8

#: Electronic-state diagnostics placeholder status.
ELECTRONIC_STATE_STATUS: Final[str] = "not_recorded"

CODE_RESULT_INVALID: Final[str] = "FRAME_RESULT_INVALID"
CODE_COMPILED_INVALID: Final[str] = "COMPILED_REQUEST_INVALID"
CODE_DRIVER_MISMATCH: Final[str] = "DRIVER_MISMATCH"
CODE_DRIVER_COVERAGE: Final[str] = "DRIVER_COVERAGE_INCOMPLETE"
CODE_MONITOR_INVALID: Final[str] = "MONITOR_INVALID"
CODE_GEOMETRY_INVALID: Final[str] = "GEOMETRY_INVALID"
CODE_FRAME_INDEX_INVALID: Final[str] = "FRAME_INDEX_INVALID"
CODE_THRESHOLD_INVALID: Final[str] = "CONTACT_THRESHOLD_INVALID"


class FrameRecoveryError(ValueError):
    """Typed recovery refusal; ``code`` is a stable machine-readable token."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Boundary value objects (parse-don't-validate).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MonitorSpec:
    """One coordinate-pool record recovered per frame (map-keyed atoms)."""

    coordinate_id: str
    kind: str
    atom_maps: tuple[int, ...]
    units: str
    role: str


@dataclass(frozen=True, slots=True)
class MonitorRecord:
    """One monitor quantity measured (or explicitly unmeasured) at a frame."""

    coordinate_id: str
    kind: str
    atom_maps: tuple[int, ...]
    units: str
    role: str
    value: float | None
    measured: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "coordinate_id": self.coordinate_id,
            "kind": self.kind,
            "atom_maps": list(self.atom_maps),
            "units": self.units,
            "role": self.role,
            "value": self.value,
            "measured": self.measured,
        }


@dataclass(frozen=True, slots=True)
class ContactRecord:
    """One non-target near contact (atom pair below threshold, not a driver)."""

    atom_index_a: int
    atom_index_b: int
    map_a: int
    map_b: int
    distance_angstrom: float

    def to_record(self) -> dict[str, Any]:
        return {
            "atom_index_a": self.atom_index_a,
            "atom_index_b": self.atom_index_b,
            "map_a": self.map_a,
            "map_b": self.map_b,
            "distance_angstrom": self.distance_angstrom,
        }


@dataclass(frozen=True, slots=True)
class EnergyChannelRow:
    """One energy observation tagged with its channel — never merged."""

    energy_channel: str
    value: float | None
    method_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "energy_channel": self.energy_channel,
            "value": self.value,
            "method_id": self.method_id,
        }


@dataclass(frozen=True, slots=True)
class FrameRecoveryRecord:
    """One frame with every driver's target/actual/residual plus side channels."""

    frame_index: int
    converged: bool | None
    lambda_value: float | None
    stage_id: str | None
    geometry_ref: str
    targets_by_driver: dict[str, float | None]
    actuals_by_driver: dict[str, float | None]
    residuals_by_driver: dict[str, float | None]
    backend_residuals_by_driver: dict[str, float | None]
    missing_driver_ids: tuple[str, ...]
    missing_residual_driver_ids: tuple[str, ...]
    incomplete: bool
    monitors: tuple[MonitorRecord, ...]
    non_target_contacts: tuple[ContactRecord, ...]
    contacts_measured: bool
    electronic_state_diagnostics: dict[str, Any]
    retry_history: tuple[dict[str, Any], ...]
    energies: tuple[EnergyChannelRow, ...]

    def to_doc(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "converged": self.converged,
            "lambda_value": self.lambda_value,
            "stage_id": self.stage_id,
            "geometry_ref": self.geometry_ref,
            "targets_by_driver": dict(self.targets_by_driver),
            "actuals_by_driver": dict(self.actuals_by_driver),
            "residuals_by_driver": dict(self.residuals_by_driver),
            "backend_residuals_by_driver": dict(self.backend_residuals_by_driver),
            "missing_driver_ids": list(self.missing_driver_ids),
            "missing_residual_driver_ids": list(self.missing_residual_driver_ids),
            "incomplete": self.incomplete,
            "monitors": [row.to_record() for row in self.monitors],
            "non_target_contacts": [row.to_record() for row in self.non_target_contacts],
            "contacts_measured": self.contacts_measured,
            "electronic_state_diagnostics": dict(self.electronic_state_diagnostics),
            "retry_history": [dict(item) for item in self.retry_history],
            "energies": [row.to_record() for row in self.energies],
        }


@dataclass(frozen=True, slots=True)
class FrameRecoveryReport:
    """Recovered per-frame all-driver surface of one multicoord attempt."""

    schema_version: str
    candidate_id: str
    mode: str
    execution_id: str
    acp_task_id: str | None
    driver_ids: tuple[str, ...]
    driver_units: dict[str, str]
    n_frames: int
    complete: bool
    frames: tuple[FrameRecoveryRecord, ...]
    incomplete_frame_indices: tuple[int, ...]
    monitor_coordinate_ids: tuple[str, ...]
    electronic_state_diagnostics: dict[str, Any]
    source: dict[str, Any]

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
            "frames": [frame.to_doc() for frame in self.frames],
            "incomplete_frame_indices": list(self.incomplete_frame_indices),
            "monitor_coordinate_ids": list(self.monitor_coordinate_ids),
            "electronic_state_diagnostics": dict(self.electronic_state_diagnostics),
            "source": dict(self.source),
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_doc())


# ---------------------------------------------------------------------------
# Boundary parsing helpers.
# ---------------------------------------------------------------------------
def _validate_compiled(compiled: Any) -> CompiledRequest:
    if not isinstance(compiled, CompiledRequest):
        raise FrameRecoveryError(
            CODE_COMPILED_INVALID,
            "compiled must be a scan_strategy.compile_orca.CompiledRequest",
        )
    if not compiled.coordinates:
        raise FrameRecoveryError(
            CODE_COMPILED_INVALID,
            "compiled request carries no driver coordinates to recover",
        )
    for row in compiled.coordinates:
        if not isinstance(row, CompiledCoordinate):
            raise FrameRecoveryError(
                CODE_COMPILED_INVALID, "compiled.coordinates entries must be CompiledCoordinate"
            )
        if not row.values:
            raise FrameRecoveryError(
                CODE_COMPILED_INVALID,
                f"compiled driver {row.kind}{list(row.maps)} carries no per-point values",
            )
    return compiled


def _validate_result(result: Any) -> MulticoordAttemptResult:
    if not isinstance(result, MulticoordAttemptResult):
        raise FrameRecoveryError(
            CODE_RESULT_INVALID,
            "multicoord_result must be a MulticoordAttemptResult from "
            "multicoord.parse_multicoord_result (todo 20 surface)",
        )
    if not result.driver_ids:
        raise FrameRecoveryError(CODE_RESULT_INVALID, "multicoord_result carries no drivers")
    return result


def _parse_monitor(raw: Any, position: int) -> MonitorSpec:
    if isinstance(raw, MonitorSpec):
        return raw
    if isinstance(raw, Mapping):
        coordinate_id = raw.get("coordinate_id")
        kind = raw.get("kind")
        atom_maps = raw.get("atom_maps")
        units = raw.get("units")
        role = raw.get("role", "monitor")
    else:
        coordinate_id = getattr(raw, "coordinate_id", None)
        kind = getattr(raw, "kind", None)
        atom_maps = getattr(raw, "atom_maps", None)
        units = getattr(raw, "units", None)
        role = getattr(raw, "role", "monitor")
    if not isinstance(coordinate_id, str) or not coordinate_id:
        raise FrameRecoveryError(
            CODE_MONITOR_INVALID, f"monitors[{position}].coordinate_id must be a non-empty string"
        )
    if kind not in (KIND_B, KIND_A, KIND_D):
        raise FrameRecoveryError(
            CODE_MONITOR_INVALID,
            f"monitors[{position}].kind must be one of B/A/D, got {kind!r}",
        )
    if not isinstance(atom_maps, (list, tuple)) or not atom_maps:
        raise FrameRecoveryError(
            CODE_MONITOR_INVALID, f"monitors[{position}].atom_maps must be a non-empty array"
        )
    maps: list[int] = []
    for map_index, map_id in enumerate(atom_maps):
        if not _is_int(map_id) or map_id <= 0:
            raise FrameRecoveryError(
                CODE_MONITOR_INVALID,
                f"monitors[{position}].atom_maps[{map_index}] must be a positive integer",
            )
        maps.append(int(map_id))
    arity = {KIND_B: 2, KIND_A: 3, KIND_D: 4}[kind]
    if len(maps) != arity:
        raise FrameRecoveryError(
            CODE_MONITOR_INVALID,
            f"monitors[{position}] kind {kind} requires {arity} atom maps, got {len(maps)}",
        )
    if len(set(maps)) != len(maps):
        raise FrameRecoveryError(
            CODE_MONITOR_INVALID, f"monitors[{position}].atom_maps must be unique"
        )
    if not isinstance(units, str) or not units:
        raise FrameRecoveryError(
            CODE_MONITOR_INVALID, f"monitors[{position}].units must be a non-empty string"
        )
    if not isinstance(role, str) or not role:
        role = "monitor"
    return MonitorSpec(
        coordinate_id=coordinate_id,
        kind=kind,
        atom_maps=tuple(maps),
        units=units,
        role=role,
    )


def _normalize_geometry(
    raw: Any, atom_rows: tuple[int, ...], path: str
) -> dict[int, tuple[float, float, float]] | None:
    """Map frame geometry to ``map_id → xyz``; ``None`` when unavailable."""
    if raw is None:
        return None
    n_atoms = len(atom_rows)
    if isinstance(raw, Mapping):
        out: dict[int, tuple[float, float, float]] = {}
        for map_id in atom_rows:
            row = raw.get(map_id, raw.get(str(map_id)))
            xyz = _xyz_row(row, f"{path}[map {map_id}]")
            if xyz is None:
                raise FrameRecoveryError(
                    CODE_GEOMETRY_INVALID, f"{path}[map {map_id}] must be a finite 3D coordinate"
                )
            out[map_id] = xyz
        return out
    if not isinstance(raw, (list, tuple)) or len(raw) != n_atoms:
        raise FrameRecoveryError(
            CODE_GEOMETRY_INVALID,
            f"{path} must be an N×3 coordinate block aligned to compiled.atom_rows "
            f"or a map-keyed mapping (N={n_atoms})",
        )
    out = {}
    for position, map_id in enumerate(atom_rows):
        xyz = _xyz_row(raw[position], f"{path}[{position}]")
        if xyz is None:
            raise FrameRecoveryError(
                CODE_GEOMETRY_INVALID, f"{path}[{position}] must be a finite 3D coordinate"
            )
        out[map_id] = xyz
    return out


def _xyz_row(raw: Any, path: str) -> tuple[float, float, float] | None:
    if isinstance(raw, Mapping):
        raw = raw.get("coordinates")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None
    try:
        xyz = (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(component) for component in xyz):
        return None
    return xyz


def _electronic_state_placeholder(compiled: CompiledRequest) -> dict[str, Any]:
    recipe = compiled.recipe if isinstance(compiled.recipe, Mapping) else {}
    charge = recipe.get("charge")
    multiplicity = recipe.get("multiplicity")
    return {
        "status": ELECTRONIC_STATE_STATUS,
        "channel": "electronic_state_diagnostics",
        "charge": charge if _is_int(charge) else None,
        "multiplicity": (
            multiplicity
            if _is_int(multiplicity) and multiplicity >= 1
            else None
        ),
        "spin_contamination": None,
        "notes": (
            "placeholder structured channel; backend electronic-state "
            "diagnostics are not yet wired (design §11.1)"
        ),
    }


def _energy_rows(frame: MulticoordFrame) -> tuple[EnergyChannelRow, ...]:
    scan_value = frame.scan_energy_hartree
    sp_value = frame.single_point_energy_hartree
    if scan_value is None and sp_value is None:
        return (EnergyChannelRow(ENERGY_CHANNEL_NONE, None),)
    rows: list[EnergyChannelRow] = []
    if scan_value is not None:
        rows.append(EnergyChannelRow(ENERGY_CHANNEL_SCAN, scan_value))
    if sp_value is not None:
        rows.append(EnergyChannelRow(ENERGY_CHANNEL_SINGLE_POINT, sp_value))
    return tuple(rows)


def _measure_monitor(
    spec: MonitorSpec, geometry: dict[int, tuple[float, float, float]] | None
) -> MonitorRecord:
    if geometry is None:
        return MonitorRecord(
            coordinate_id=spec.coordinate_id,
            kind=spec.kind,
            atom_maps=spec.atom_maps,
            units=spec.units,
            role=spec.role,
            value=None,
            measured=False,
        )
    points = [geometry[map_id] for map_id in spec.atom_maps]
    if spec.kind == KIND_B:
        value = math.dist(points[0], points[1])
    elif spec.kind == KIND_A:
        value = bond_angle_deg(points[0], points[1], points[2])
    else:
        value = dihedral_deg(points[0], points[1], points[2], points[3])
    return MonitorRecord(
        coordinate_id=spec.coordinate_id,
        kind=spec.kind,
        atom_maps=spec.atom_maps,
        units=spec.units,
        role=spec.role,
        value=value,
        measured=True,
    )


def _target_map_pairs(compiled: CompiledRequest) -> set[frozenset[int]]:
    pairs: set[frozenset[int]] = set()
    for row in compiled.coordinates:
        maps = list(row.maps)
        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                pairs.add(frozenset((maps[i], maps[j])))
    return pairs


def _non_target_contacts(
    geometry: dict[int, tuple[float, float, float]] | None,
    atom_rows: tuple[int, ...],
    target_pairs: set[frozenset[int]],
    threshold: float,
) -> tuple[ContactRecord, ...]:
    if geometry is None:
        return ()
    index_by_map = {map_id: index for index, map_id in enumerate(atom_rows)}
    contacts: list[ContactRecord] = []
    for i, map_a in enumerate(atom_rows):
        for map_b in atom_rows[i + 1 :]:
            if frozenset((map_a, map_b)) in target_pairs:
                continue
            distance = math.dist(geometry[map_a], geometry[map_b])
            if distance < threshold:
                contacts.append(
                    ContactRecord(
                        atom_index_a=index_by_map[map_a],
                        atom_index_b=index_by_map[map_b],
                        map_a=map_a,
                        map_b=map_b,
                        distance_angstrom=distance,
                    )
                )
    return tuple(contacts)


def _assert_all_drivers_covered(record: FrameRecoveryRecord, driver_ids: tuple[str, ...]) -> None:
    """Tripwire: recovery outputs must carry EVERY driver, never first-only."""
    expected = set(driver_ids)
    for field_name, mapping in (
        ("targets_by_driver", record.targets_by_driver),
        ("actuals_by_driver", record.actuals_by_driver),
        ("residuals_by_driver", record.residuals_by_driver),
        ("backend_residuals_by_driver", record.backend_residuals_by_driver),
    ):
        if set(mapping) != expected:
            raise FrameRecoveryError(
                CODE_DRIVER_COVERAGE,
                f"frame {record.frame_index} {field_name} drivers "
                f"{sorted(mapping)} != all drivers {sorted(expected)}",
            )


# ---------------------------------------------------------------------------
# Recovery entry point.
# ---------------------------------------------------------------------------
def recover_frames(
    multicoord_result: Any,
    compiled: Any,
    monitors: Sequence[Any] | None = None,
    *,
    geometries: Mapping[int, Any] | None = None,
    contact_threshold_angstrom: float = DEFAULT_CONTACT_THRESHOLD_ANGSTROM,
) -> FrameRecoveryReport:
    """Recover per-frame all-driver targets/actuals/residuals plus side channels.

    Targets are the compiled schedule-λ values (todo 19 ``CompiledCoordinate``
    values at each frame index; parsed targets are the fallback when the
    compiled table does not cover that index).  Actuals come from the todo-20
    parsed frame.  Residuals are computed as ``target − actual`` for every
    driver.  A frame missing ANY driver's residual is marked ``incomplete``
    (never silently passed); ``missing_driver_ids`` keeps todo-20 semantics
    (target or actual absent).

    ``monitors`` are coordinate-pool records (``CoordinateRecord`` or mappings
    with ``coordinate_id``/``kind``/``atom_maps``/``units``/``role``); values
    are measured from per-frame ``geometries`` when supplied (else explicitly
    unmeasured).  Non-target contacts are nonbonded atom pairs below
    ``contact_threshold_angstrom`` that are not part of any driver coordinate.
    Electronic-state diagnostics is a placeholder structured channel; retry
    history is carried per frame.  Single-point and scan energies are stored
    in separate channel rows.
    """
    result = _validate_result(multicoord_result)
    compiled_request = _validate_compiled(compiled)
    if result.candidate_id != compiled_request.candidate_id:
        raise FrameRecoveryError(
            CODE_DRIVER_MISMATCH,
            f"result candidate_id {result.candidate_id!r} != compiled "
            f"{compiled_request.candidate_id!r}",
        )
    expected_driver_ids = tuple(
        driver_id_for(index) for index in range(len(compiled_request.coordinates))
    )
    if result.driver_ids != expected_driver_ids:
        raise FrameRecoveryError(
            CODE_DRIVER_MISMATCH,
            f"result drivers {list(result.driver_ids)} != compiled drivers "
            f"{list(expected_driver_ids)} — recovery consumes every compiled "
            "driver, never a first-driver projection",
        )
    if not isinstance(contact_threshold_angstrom, (int, float)) or isinstance(
        contact_threshold_angstrom, bool
    ):
        raise FrameRecoveryError(
            CODE_THRESHOLD_INVALID, "contact_threshold_angstrom must be a finite number"
        )
    threshold = float(contact_threshold_angstrom)
    if not math.isfinite(threshold) or threshold <= 0:
        raise FrameRecoveryError(
            CODE_THRESHOLD_INVALID, "contact_threshold_angstrom must be finite and positive"
        )
    monitor_specs = tuple(
        _parse_monitor(raw, position) for position, raw in enumerate(monitors or ())
    )
    if geometries is not None and not isinstance(geometries, Mapping):
        raise FrameRecoveryError(
            CODE_GEOMETRY_INVALID, "geometries must be a mapping keyed by frame_index or None"
        )
    atom_rows = compiled_request.atom_rows
    target_pairs = _target_map_pairs(compiled_request)
    electronic_state = _electronic_state_placeholder(compiled_request)
    driver_ids = result.driver_ids
    driver_units = dict(result.driver_units)
    n_points = len(compiled_request.lambda_values)

    records: list[FrameRecoveryRecord] = []
    seen_indices: set[int] = set()
    for frame in result.frames:
        if frame.frame_index in seen_indices:
            raise FrameRecoveryError(
                CODE_FRAME_INDEX_INVALID,
                f"duplicate frame_index {frame.frame_index} in recovery input",
            )
        seen_indices.add(frame.frame_index)
        index = frame.frame_index
        lambda_value = (
            compiled_request.lambda_values[index] if 0 <= index < n_points else None
        )
        targets: dict[str, float | None] = {}
        actuals: dict[str, float | None] = {}
        residuals: dict[str, float | None] = {}
        missing_driver: list[str] = []
        missing_residual: list[str] = []
        for position, driver_id in enumerate(driver_ids):
            row = compiled_request.coordinates[position]
            compiled_target = (
                _finite_number(row.values[index])
                if 0 <= index < len(row.values)
                else None
            )
            parsed_target = _finite_number(frame.target_by_driver.get(driver_id))
            target = compiled_target if compiled_target is not None else parsed_target
            actual = _finite_number(frame.actual_by_driver.get(driver_id))
            residual = (target - actual) if target is not None and actual is not None else None
            targets[driver_id] = target
            actuals[driver_id] = actual
            residuals[driver_id] = residual
            if target is None or actual is None:
                missing_driver.append(driver_id)
            if residual is None:
                missing_residual.append(driver_id)
        backend_residuals = {
            driver_id: _finite_number(frame.residual_by_driver.get(driver_id))
            for driver_id in driver_ids
        }
        geometry = None
        if geometries is not None:
            geometry = _normalize_geometry(
                geometries.get(index), atom_rows, f"geometries[{index}]"
            )
        monitor_rows = tuple(_measure_monitor(spec, geometry) for spec in monitor_specs)
        contacts = _non_target_contacts(geometry, atom_rows, target_pairs, threshold)
        record = FrameRecoveryRecord(
            frame_index=index,
            converged=frame.converged,
            lambda_value=lambda_value,
            stage_id=None,
            geometry_ref=frame.geometry_path,
            targets_by_driver=targets,
            actuals_by_driver=actuals,
            residuals_by_driver=residuals,
            backend_residuals_by_driver=backend_residuals,
            missing_driver_ids=tuple(missing_driver),
            missing_residual_driver_ids=tuple(missing_residual),
            incomplete=bool(missing_residual),
            monitors=monitor_rows,
            non_target_contacts=contacts,
            contacts_measured=geometry is not None,
            electronic_state_diagnostics=dict(electronic_state),
            retry_history=tuple(dict(item) for item in frame.retry_history),
            energies=_energy_rows(frame),
        )
        _assert_all_drivers_covered(record, driver_ids)
        records.append(record)
    records.sort(key=lambda row: row.frame_index)
    incomplete_indices = tuple(row.frame_index for row in records if row.incomplete)
    complete = bool(records) and not incomplete_indices
    return FrameRecoveryReport(
        schema_version=SCHEMA_FRAME_RECOVERY,
        candidate_id=compiled_request.candidate_id,
        mode=compiled_request.mode,
        execution_id=result.execution_id,
        acp_task_id=result.acp_task_id,
        driver_ids=driver_ids,
        driver_units=driver_units,
        n_frames=len(records),
        complete=complete,
        frames=tuple(records),
        incomplete_frame_indices=incomplete_indices,
        monitor_coordinate_ids=tuple(spec.coordinate_id for spec in monitor_specs),
        electronic_state_diagnostics=dict(electronic_state),
        source={
            "compiled_schema_version": compiled_request.schema_version,
            "compiled_candidate_id": compiled_request.candidate_id,
            "compiled_mode": compiled_request.mode,
            "result_schema_version": result.schema_version,
            "result_candidate_id": result.candidate_id,
            "execution_id": result.execution_id,
            "n_drivers": len(driver_ids),
            "driver_ids": list(driver_ids),
            "n_points": n_points,
        },
    )


# ---------------------------------------------------------------------------
# PathBundle v2 extension (additive; old single-driver fields unchanged).
# ---------------------------------------------------------------------------
def _extension_frame_record(
    record: FrameRecoveryRecord, first_driver: str
) -> dict[str, Any]:
    """One extension frame: old single-driver semantics + new named fields."""
    return {
        "frame_index": record.frame_index,
        "converged": record.converged,
        "lambda_value": record.lambda_value,
        "stage_id": record.stage_id,
        "geometry_ref": record.geometry_ref,
        # OLD PathBundle frame field semantics — first driver, unchanged shape
        # (single-driver bundles carry exactly these values historically).
        "target_coordinate": record.targets_by_driver.get(first_driver),
        "actual_coordinate": record.actuals_by_driver.get(first_driver),
        "constraint_residuals": dict(record.backend_residuals_by_driver),
        # NEW additive named fields — never overwrite the old ones above.
        "all_driver_targets": dict(record.targets_by_driver),
        "all_driver_actuals": dict(record.actuals_by_driver),
        "all_driver_residuals": dict(record.residuals_by_driver),
        "missing_driver_ids": list(record.missing_driver_ids),
        "missing_residual_driver_ids": list(record.missing_residual_driver_ids),
        "incomplete": record.incomplete,
        "monitors": [row.to_record() for row in record.monitors],
        "non_target_contacts": [row.to_record() for row in record.non_target_contacts],
        "contacts_measured": record.contacts_measured,
        "electronic_state_diagnostics": dict(record.electronic_state_diagnostics),
        "retry_history": [dict(item) for item in record.retry_history],
        "energy_channels": [row.to_record() for row in record.energies],
    }


def to_path_bundle_extension(report: Any) -> dict[str, Any]:
    """Project a recovery report as a PathBundle ``extensions`` block (v2).

    Consumers attach the returned dict at
    ``path_bundle["extensions"]["pes2ts.frame_recovery.v2"]``.  Old
    single-driver PathBundle frame field semantics stay intact: each extension
    frame carries ``target_coordinate`` / ``actual_coordinate`` /
    ``constraint_residuals`` with their historical meaning (first driver when
    several are present).  Multi-driver data is added under new named fields
    only — additive, never an overwrite of the v1 frame fields inside the
    PathBundle itself.
    """
    if not isinstance(report, FrameRecoveryReport):
        raise FrameRecoveryError(
            CODE_RESULT_INVALID, "report must be a FrameRecoveryReport from recover_frames"
        )
    if not report.driver_ids:
        raise FrameRecoveryError(CODE_DRIVER_COVERAGE, "report carries no drivers")
    first_driver = report.driver_ids[0]
    frames = [_extension_frame_record(record, first_driver) for record in report.frames]
    for frame_record, record in zip(frames, report.frames, strict=True):
        for driver_id in report.driver_ids:
            if (driver_id not in frame_record["all_driver_targets"]
                    or driver_id not in frame_record["all_driver_actuals"]
                    or driver_id not in frame_record["all_driver_residuals"]):
                raise FrameRecoveryError(
                    CODE_DRIVER_COVERAGE,
                    f"extension frame {record.frame_index} missing driver {driver_id}",
                )
    return {
        "schema_version": SCHEMA_PATH_BUNDLE_EXTENSION,
        "recovery_schema_version": report.schema_version,
        "extension_key": PATH_BUNDLE_EXTENSION_KEY,
        "candidate_id": report.candidate_id,
        "mode": report.mode,
        "execution_id": report.execution_id,
        "acp_task_id": report.acp_task_id,
        "n_drivers": len(report.driver_ids),
        "driver_ids": list(report.driver_ids),
        "driver_units": dict(report.driver_units),
        "n_frames": report.n_frames,
        "complete": report.complete,
        "incomplete_frame_indices": list(report.incomplete_frame_indices),
        "monitor_coordinate_ids": list(report.monitor_coordinate_ids),
        "electronic_state_diagnostics": dict(report.electronic_state_diagnostics),
        "frames": frames,
        "source": dict(report.source),
        "semantics": {
            "old_single_driver_fields": [
                "target_coordinate", "actual_coordinate", "constraint_residuals",
            ],
            "new_named_fields": [
                "all_driver_targets", "all_driver_actuals", "all_driver_residuals",
                "missing_driver_ids", "missing_residual_driver_ids", "incomplete",
                "monitors", "non_target_contacts", "electronic_state_diagnostics",
                "retry_history", "energy_channels", "lambda_value", "stage_id",
            ],
            "additive_only": True,
            "residual_definition": "target_minus_actual",
            "incomplete_rule": (
                "frame incomplete when ANY driver residual is missing "
                "(target or actual absent); never silently passed"
            ),
        },
    }
