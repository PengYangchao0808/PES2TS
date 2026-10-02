"""ORCA Scan / Constraints / Path compiler for frozen plan candidates (todo 19).

Implements design §8 (mode table) and §9.2 (compilation sketch) of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``:

- **Capability gate first, compilation second.**  Every request is checked
  against the effective backend capability (``capabilities.capability_check``)
  *before* any syntax is produced.  Unsupported / unprobed requests raise
  :class:`OrcaCompileError` with a machine-readable ``code`` and produce no
  compilation output.  Multi-coordinate ``Simul_Scan`` additionally requires
  an effective ``constraint_support.simul_scan`` claim *and* that
  ``COUPLED_1D`` is enabled by passing probe receipts (todo 16); unprobed →
  ``SIMUL_SCAN_UNPROBED``.
- **0-based indices from the common atom order.**  Driver atom map ids are
  resolved to ORCA 0-based indices through the frozen ``atom_rows`` order
  supplied by the plan — never ``map − 1``.  A frozen ``driver.index0`` that
  disagrees with the order is a typed refusal.
- **Units.**  ``B`` distances compile in Å (``angstrom``), ``A``/``D`` in
  degrees (``degree``); a unit/kind mismatch is refused.  ``D`` interpolation
  uses the shortest-arc delta (periodic handling) exactly as todo 13 defined.
- **Modes.**  ``SINGLE_1D`` (linear) → one native ``%geom Scan`` coordinate;
  ``COUPLED_1D`` (2–3 linear coordinates) → ``%geom Scan`` per coordinate plus
  ``Simul_Scan true`` (same λ index); ``SCHEDULED_1D`` custom lists and any
  non-linear schedule → sealed per-point ``%geom Constraints`` generation
  recipe (one input per λ point, per-point sha256 recorded — not inline file
  dumps).  ``PathCandidateV1`` → minimal ``%geom Path`` shape (endpoint xyz
  blocks + image count; elaborated by todo 26).
- **All drivers recovered.**  Every driver coordinate of the candidate appears
  in the compiled request; there is no first-driver projection.
- **No implicit grids.**  Only the schedule's λ-derived points are compiled
  (``total_points == len(lambda_values)``).

``CompiledRequest`` is the artifact contract for todos 20/21/23: it carries
the ``%geom`` syntax fragment strings, the per-point input sha256 list, the
total point count, and the coordinate table (kind, maps, indices, units,
start/end, per-point values).
"""

# allow: SIZE_OK — plan-named todo-19 single compiler module (design §8 mode
# table + §9.2 compilation boundary); precedent: compat_export.py (todo 18) /
# capabilities.py (todo 16) / schedules.py (todo 14).

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.generation.planning.capabilities import (
    BACKEND_CAPABILITY_MISSING,
    EffectiveCapability,
    capability_check,
)
from pes2ts_core.generation.planning.contracts_v2 import (
    CANDIDATE_KIND_PATH,
    CANDIDATE_KIND_SCAN,
    DRIVER_KINDS,
    MODE_COUPLED_1D,
    MODE_SCHEDULED_1D,
    MODE_SINGLE_1D,
    PATH_METHOD_KINDS,
)
from pes2ts_core.generation.planning.coordinate_pool import (
    KIND_A,
    KIND_B,
    KIND_D,
    KIND_UNITS,
    UNIT_ANGSTROM,
    UNIT_DEGREE,
)
from pes2ts_core.generation.planning.geometry_feasibility import (
    bond_angle_deg,
    dihedral_deg,
    shortest_arc_delta,
)
from pes2ts_core.utils.hashing import sha256_bytes, stable_json_dumps

__all__ = [
    "CODE_ATOM_ORDER_INVALID",
    "CODE_BACKEND_CAPABILITY_MISSING",
    "CODE_CAPABILITY_UNPROBED",
    "CODE_COORDINATE_VALUE_INVALID",
    "CODE_CUSTOM_SCHEDULE_UNSUPPORTED",
    "CODE_DRIVER_ARITY_INVALID",
    "CODE_DRIVER_MAPS_INVALID",
    "CODE_DRIVER_MAP_UNKNOWN",
    "CODE_GEOMETRY_REQUIRED",
    "CODE_INDEX_ORDER_MISMATCH",
    "CODE_LAMBDA_GRID_INVALID",
    "CODE_PATH_GEOMETRY_INVALID",
    "CODE_PATH_METHOD_INVALID",
    "CODE_PER_POINT_CONSTRAINTS_UNSUPPORTED",
    "CODE_POINT_WINDOW_INVALID",
    "CODE_SCHEDULE_VALUE_MISMATCH",
    "CODE_SIMUL_SCAN_UNPROBED",
    "CODE_SIMUL_SCAN_UNSUPPORTED",
    "CODE_UNKNOWN_CANDIDATE_KIND",
    "CODE_UNIT_UNSUPPORTED",
    "CompiledCoordinate",
    "CompiledRequest",
    "OrcaCompileError",
    "SCHEMA_ORCA_COMPILE",
    "compile_orca",
    "compile_orca_path_request",
]

# ---------------------------------------------------------------------------
# Vocabulary and typed refusal codes.
# ---------------------------------------------------------------------------
SCHEMA_ORCA_COMPILE: Final[str] = "g1_orca_compile_v1"

CODE_BACKEND_CAPABILITY_MISSING: Final[str] = BACKEND_CAPABILITY_MISSING
CODE_SIMUL_SCAN_UNPROBED: Final[str] = "SIMUL_SCAN_UNPROBED"
CODE_SIMUL_SCAN_UNSUPPORTED: Final[str] = "SIMUL_SCAN_UNSUPPORTED"
CODE_CAPABILITY_UNPROBED: Final[str] = "CAPABILITY_UNPROBED"
CODE_CUSTOM_SCHEDULE_UNSUPPORTED: Final[str] = "CUSTOM_SCHEDULE_UNSUPPORTED"
CODE_PER_POINT_CONSTRAINTS_UNSUPPORTED: Final[str] = "PER_POINT_CONSTRAINTS_UNSUPPORTED"
CODE_UNKNOWN_CANDIDATE_KIND: Final[str] = "UNKNOWN_CANDIDATE_KIND"
CODE_ATOM_ORDER_INVALID: Final[str] = "ATOM_ORDER_INVALID"
CODE_DRIVER_MAPS_INVALID: Final[str] = "DRIVER_MAPS_INVALID"
CODE_DRIVER_MAP_UNKNOWN: Final[str] = "DRIVER_MAP_UNKNOWN"
CODE_DRIVER_ARITY_INVALID: Final[str] = "DRIVER_ARITY_INVALID"
CODE_INDEX_ORDER_MISMATCH: Final[str] = "INDEX_ORDER_MISMATCH"
CODE_UNIT_UNSUPPORTED: Final[str] = "UNIT_UNSUPPORTED"
CODE_GEOMETRY_REQUIRED: Final[str] = "GEOMETRY_REQUIRED"
CODE_COORDINATE_VALUE_INVALID: Final[str] = "COORDINATE_VALUE_INVALID"
CODE_LAMBDA_GRID_INVALID: Final[str] = "LAMBDA_GRID_INVALID"
CODE_SCHEDULE_VALUE_MISMATCH: Final[str] = "SCHEDULE_VALUE_MISMATCH"
CODE_POINT_WINDOW_INVALID: Final[str] = "POINT_WINDOW_INVALID"
CODE_PATH_GEOMETRY_INVALID: Final[str] = "PATH_GEOMETRY_INVALID"
CODE_PATH_METHOD_INVALID: Final[str] = "PATH_METHOD_INVALID"

#: Driver kind → arity (B=2, A=3, D=4 atoms).
KIND_ARITY: Final[dict[str, int]] = {KIND_B: 2, KIND_A: 3, KIND_D: 4}

#: Physical-value rounding (byte-stable serialization, todo-18 precedent).
VALUE_DECIMALS: Final[int] = 6

#: Schedule linearity tolerance (s(λ) vs λ).
LINEARITY_TOL: Final[float] = 1e-9

#: Per-point recipe identifier (sealed generation recipe, not inline files).
RECIPE_PER_POINT: Final[str] = "orca_per_point_constraints_v1"
RECIPE_NATIVE_SCAN: Final[str] = "orca_native_scan_v1"
RECIPE_PATH_NEB: Final[str] = "orca_path_neb_v1"


class OrcaCompileError(ValueError):
    """Typed compile refusal; ``code`` is a stable machine-readable token."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


# ---------------------------------------------------------------------------
# Compiled artifacts.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CompiledCoordinate:
    """One compiled driver coordinate (every driver, never projected away)."""

    kind: str
    maps: tuple[int, ...]
    indices: tuple[int, ...]
    unit: str
    start_value: float
    end_value: float
    values: tuple[float, ...]
    schedule_values: tuple[float, ...]

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "kind": self.kind,
            "maps": list(self.maps),
            "indices": list(self.indices),
            "unit": self.unit,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "values": list(self.values),
            "schedule_values": list(self.schedule_values),
        }


@dataclass(frozen=True, slots=True)
class CompiledRequest:
    """Sealed compilation artifact of one frozen plan candidate.

    ``geom_fragments`` are the ``%geom`` syntax fragment strings;
    ``point_input_sha256`` is the per-point input hash list (one entry for
    native Scan/Path inputs, ``total_points`` entries for per-point
    Constraints recipes); ``coordinates`` carries the full driver table with
    0-based indices resolved through the common atom order.
    """

    schema_version: str
    candidate_id: str
    mode: str
    compiled_kind: str
    recipe_kind: str
    geom_fragments: tuple[str, ...]
    point_input_sha256: tuple[str, ...]
    total_points: int
    coordinates: tuple[CompiledCoordinate, ...]
    simultaneous: bool
    lambda_values: tuple[float, ...]
    atom_rows: tuple[int, ...]
    recipe: dict[str, Any]

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection (deterministic; no truth-derived keys)."""
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "compiled_kind": self.compiled_kind,
            "recipe_kind": self.recipe_kind,
            "geom_fragments": list(self.geom_fragments),
            "point_input_sha256": list(self.point_input_sha256),
            "total_points": self.total_points,
            "coordinates": [row.to_record() for row in self.coordinates],
            "simultaneous": self.simultaneous,
            "lambda_values": list(self.lambda_values),
            "atom_rows": list(self.atom_rows),
            "recipe": self.recipe,
        }

    def to_json(self) -> str:
        """Canonical serialization (determinism test surface)."""
        return stable_json_dumps(self.to_doc())


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    """Deterministic compact float rendering for ORCA syntax fragments."""
    rounded = round(float(value), VALUE_DECIMALS)
    if rounded == 0.0:
        return "0"
    text = f"{rounded:.{VALUE_DECIMALS}f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-", "-0"} else text


def _round_value(value: float) -> float:
    return round(float(value), VALUE_DECIMALS)


def _normalize_dihedral(value: float) -> float:
    """Normalize a dihedral into (−180, 180]."""
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if normalized <= -180.0:
        normalized += 360.0
    return normalized


def _interpolate(kind: str, start: float, end: float, s_value: float) -> float:
    """Return q(s) = start + s·Δ with the shortest arc for dihedrals."""
    if kind == KIND_D:
        delta = shortest_arc_delta(start, end)
        return _normalize_dihedral(start + s_value * delta)
    return start + s_value * (end - start)


def _is_linear(s_values: Sequence[float], lambdas: Sequence[float]) -> bool:
    """True when every s(λ) equals λ within :data:`LINEARITY_TOL`."""
    return all(
        abs(float(s) - float(lam)) <= LINEARITY_TOL
        for s, lam in zip(s_values, lambdas, strict=True)
    )


def _parse_atom_rows(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise OrcaCompileError(CODE_ATOM_ORDER_INVALID, "atom_rows must be a non-empty array")
    try:
        order = tuple(int(map_id) for map_id in raw)
    except (TypeError, ValueError) as exc:
        raise OrcaCompileError(CODE_ATOM_ORDER_INVALID, "atom_rows entries must be integers") from exc
    if any(map_id <= 0 for map_id in order) or len(set(order)) != len(order):
        raise OrcaCompileError(
            CODE_ATOM_ORDER_INVALID,
            f"atom_rows={order!r} must be unique positive map ids (the common frozen order)",
        )
    return order


def _normalize_geometry(
    raw: Any,
) -> dict[str, dict[int, tuple[float, float, float]]]:
    """Parse endpoint geometry into ``{"R": …, "P": …}`` map→xyz blocks."""
    if not isinstance(raw, Mapping):
        raise OrcaCompileError(CODE_GEOMETRY_REQUIRED, "geometry mapping with R/P coordinate blocks is required")
    out: dict[str, dict[int, tuple[float, float, float]]] = {}
    for canonical, keys in (("R", ("R", "r", "reactant")), ("P", ("P", "p", "product"))):
        block: Any = None
        for key in keys:
            if key in raw:
                block = raw[key]
                break
        if block is None:
            raise OrcaCompileError(CODE_GEOMETRY_REQUIRED, f"geometry is missing the {canonical} coordinate block")
        if not isinstance(block, Mapping):
            raise OrcaCompileError(CODE_GEOMETRY_REQUIRED, f"geometry.{canonical} must be a map→coordinate object")
        parsed: dict[int, tuple[float, float, float]] = {}
        for map_key, value in block.items():
            try:
                map_id = int(map_key)
            except (TypeError, ValueError) as exc:
                raise OrcaCompileError(
                    CODE_GEOMETRY_REQUIRED, f"geometry.{canonical} key {map_key!r} is not an integer map id"
                ) from exc
            if isinstance(value, Mapping):
                value = value.get("coordinates")
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise OrcaCompileError(
                    CODE_GEOMETRY_REQUIRED, f"geometry.{canonical}[{map_id}] must be one 3D coordinate"
                )
            try:
                xyz = (float(value[0]), float(value[1]), float(value[2]))
            except (TypeError, ValueError) as exc:
                raise OrcaCompileError(
                    CODE_GEOMETRY_REQUIRED, f"geometry.{canonical}[{map_id}] coordinates must be numbers"
                ) from exc
            if any(not math.isfinite(component) for component in xyz):
                raise OrcaCompileError(
                    CODE_GEOMETRY_REQUIRED, f"geometry.{canonical}[{map_id}] coordinates must be finite"
                )
            parsed[map_id] = xyz
        out[canonical] = parsed
    return out


def _geometry_sha256(geometry: Mapping[str, Mapping[int, tuple[float, float, float]]]) -> str:
    payload = {
        side: {
            str(map_id): [_round_value(c) for c in coords]
            for map_id, coords in sorted(block.items())
        }
        for side, block in sorted(geometry.items())
    }
    return sha256_bytes(stable_json_dumps(payload).encode("utf-8"))


def _resolve_driver(
    driver: Any,
    map_to_index: Mapping[int, int],
    path: str,
) -> tuple[str, tuple[int, ...], tuple[int, ...], str, tuple[float, ...] | None]:
    """Parse one driver payload → (kind, maps, indices, unit, schedule_values)."""
    if not isinstance(driver, Mapping):
        raise OrcaCompileError(CODE_DRIVER_MAPS_INVALID, f"{path} must be an object")
    kind = driver.get("kind")
    if kind not in DRIVER_KINDS:
        raise OrcaCompileError(
            CODE_DRIVER_MAPS_INVALID, f"{path}.kind={kind!r}; expected one of {', '.join(DRIVER_KINDS)}"
        )
    raw_maps = driver.get("maps")
    if not isinstance(raw_maps, (list, tuple)) or not raw_maps:
        raise OrcaCompileError(CODE_DRIVER_MAPS_INVALID, f"{path}.maps must be a non-empty array")
    try:
        maps = tuple(int(map_id) for map_id in raw_maps)
    except (TypeError, ValueError) as exc:
        raise OrcaCompileError(CODE_DRIVER_MAPS_INVALID, f"{path}.maps entries must be integers") from exc
    arity = KIND_ARITY[str(kind)]
    if len(maps) != arity:
        raise OrcaCompileError(
            CODE_DRIVER_ARITY_INVALID,
            f"{path}: kind {kind} requires {arity} atom maps, got {len(maps)} ({maps!r})",
        )
    if len(set(maps)) != len(maps) or any(map_id <= 0 for map_id in maps):
        raise OrcaCompileError(
            CODE_DRIVER_MAPS_INVALID, f"{path}.maps={maps!r} must be distinct positive integers"
        )
    unknown = [map_id for map_id in maps if map_id not in map_to_index]
    if unknown:
        raise OrcaCompileError(
            CODE_DRIVER_MAP_UNKNOWN,
            f"{path}: maps {unknown} are not in the frozen atom_rows order — indices "
            "resolve through the common atom order, never map−1",
        )
    computed = tuple(map_to_index[map_id] for map_id in maps)
    index0 = driver.get("index0")
    if index0 is not None:
        if not isinstance(index0, (list, tuple)):
            raise OrcaCompileError(CODE_INDEX_ORDER_MISMATCH, f"{path}.index0 must be null or an integer array")
        try:
            frozen = tuple(int(value) for value in index0)
        except (TypeError, ValueError) as exc:
            raise OrcaCompileError(CODE_INDEX_ORDER_MISMATCH, f"{path}.index0 entries must be integers") from exc
        if frozen != computed:
            raise OrcaCompileError(
                CODE_INDEX_ORDER_MISMATCH,
                f"{path}.index0={frozen!r} disagrees with atom_rows indices {computed!r}",
            )
    unit = driver.get("unit")
    expected_unit = KIND_UNITS[str(kind)]
    if unit != expected_unit:
        raise OrcaCompileError(
            CODE_UNIT_UNSUPPORTED,
            f"{path}.unit={unit!r}; kind {kind} compiles in {expected_unit!r}",
        )
    raw_schedule = driver.get("schedule_values")
    schedule_values: tuple[float, ...] | None
    if raw_schedule is None:
        schedule_values = None
    else:
        if not isinstance(raw_schedule, (list, tuple)):
            raise OrcaCompileError(
                CODE_SCHEDULE_VALUE_MISMATCH, f"{path}.schedule_values must be null or an array"
            )
        try:
            schedule_values = tuple(float(value) for value in raw_schedule)
        except (TypeError, ValueError) as exc:
            raise OrcaCompileError(
                CODE_SCHEDULE_VALUE_MISMATCH, f"{path}.schedule_values entries must be numbers"
            ) from exc
        if any(not math.isfinite(value) for value in schedule_values):
            raise OrcaCompileError(
                CODE_SCHEDULE_VALUE_MISMATCH, f"{path}.schedule_values entries must be finite"
            )
    return str(kind), maps, computed, str(unit), schedule_values


def _endpoint_q(
    kind: str,
    maps: tuple[int, ...],
    geometry: Mapping[str, Mapping[int, tuple[float, float, float]]],
    start_endpoint: str,
) -> tuple[float, float]:
    """Return (q_start, q_end) measured from endpoint geometry."""
    if start_endpoint == "R":
        sides = (geometry["R"], geometry["P"])
    elif start_endpoint == "P":
        sides = (geometry["P"], geometry["R"])
    else:
        raise OrcaCompileError(
            CODE_COORDINATE_VALUE_INVALID, f"start_endpoint={start_endpoint!r}; expected 'R' or 'P'"
        )
    values: list[float] = []
    for coords in sides:
        for map_id in maps:
            if map_id not in coords:
                raise OrcaCompileError(
                    CODE_GEOMETRY_REQUIRED, f"geometry is missing atom map {map_id}"
                )
        points = [coords[map_id] for map_id in maps]
        if kind == KIND_B:
            values.append(math.dist(points[0], points[1]))
        elif kind == KIND_A:
            values.append(bond_angle_deg(points[0], points[1], points[2]))
        else:
            values.append(dihedral_deg(points[0], points[1], points[2], points[3]))
    if kind == KIND_D:
        return _normalize_dihedral(values[0]), _normalize_dihedral(values[1])
    return values[0], values[1]


def _check_point_window(n_points: int, capability: EffectiveCapability) -> None:
    baseline = capability.point_limits.baseline
    maximum = capability.point_limits.max
    if baseline > maximum:
        raise OrcaCompileError(
            CODE_POINT_WINDOW_INVALID,
            f"effective point_limits are inverted (baseline={baseline} > max={maximum})",
        )
    if n_points < baseline or n_points > maximum:
        raise OrcaCompileError(
            CODE_POINT_WINDOW_INVALID,
            f"n_points={n_points} outside effective window [{baseline}, {maximum}]",
        )


def _xyz_block(
    rows: Sequence[Sequence[float]],
    charge: int,
    multiplicity: int,
    elements: Sequence[str] | None,
) -> str:
    lines = [f"* xyz {charge} {multiplicity}"]
    for index, row in enumerate(rows):
        symbol = elements[index] if elements is not None and index < len(elements) else "X"
        lines.append(f"{symbol} {_fmt(row[0])} {_fmt(row[1])} {_fmt(row[2])}")
    lines.append("*")
    return "\n".join(lines)


def _constraint_block(
    coordinates: Sequence[CompiledCoordinate], point_index: int
) -> str:
    lines = ["%geom", "Constraints"]
    for coord in coordinates:
        indices = " ".join(str(index) for index in coord.indices)
        lines.append(
            "{ " + f"{coord.kind} {indices} {_fmt(coord.values[point_index])}" + " }"
        )
    lines.append("end")
    lines.append("end")
    return "\n".join(lines)


def _native_scan_block(
    coordinates: Sequence[CompiledCoordinate],
    n_points: int,
    simultaneous: bool,
) -> str:
    steps = n_points - 1
    lines = ["%geom"]
    for coord in coordinates:
        indices = " ".join(str(index) for index in coord.indices)
        lines.append(
            f"Scan {coord.kind} {indices} = {_fmt(coord.start_value)}, "
            f"{_fmt(coord.end_value)}, {steps} end"
        )
    if simultaneous:
        lines.append("Simul_Scan true")
    lines.append("end")
    return "\n".join(lines)


def _path_block(n_images: int) -> str:
    return "\n".join(["%geom", "Path", f"n_images {n_images}", "end"])


def _input_text(
    *,
    candidate_id: str,
    point_label: str,
    method: str,
    charge: int,
    multiplicity: int,
    nprocs: int,
    geometry_sha: str,
    atom_rows: Sequence[int],
    geom_block: str,
) -> str:
    return "\n".join(
        [
            f"pes2ts-orca-compile {candidate_id} {point_label}",
            f"! {method}",
            "%pal",
            f" nprocs {nprocs}",
            "end",
            f"# geometry_sha256={geometry_sha} atom_rows={list(atom_rows)}",
            "# xyz assembled by the adapter from the frozen atom order + endpoint geometry",
            geom_block,
            "",
        ]
    )


def _refuse_capability(
    capability: EffectiveCapability, mode: str, kinds: Sequence[str], n_coords: int
) -> None:
    """Capability gate: raise the typed refusal *before* any compilation."""
    check = capability_check(capability, mode, kinds, n_coords)
    if check["status"] == "pass":
        return
    missing = "; ".join(str(item) for item in check.get("missing") or ())
    if check["status"] == "unknown":
        if mode == MODE_COUPLED_1D:
            raise OrcaCompileError(
                CODE_SIMUL_SCAN_UNPROBED,
                "COUPLED_1D Simul_Scan is declared but unprobed — a passing probe "
                f"receipt covering Simul_Scan is required ({missing})",
            )
        raise OrcaCompileError(
            CODE_CAPABILITY_UNPROBED,
            f"mode {mode!r} is declared but not enabled by probe receipts ({missing})",
        )
    raise OrcaCompileError(BACKEND_CAPABILITY_MISSING, missing or f"mode {mode!r} not supported")


# ---------------------------------------------------------------------------
# Public compiler.
# ---------------------------------------------------------------------------
def compile_orca(
    candidate: Mapping[str, Any],
    atom_rows: Sequence[int],
    capability: EffectiveCapability,
    *,
    geometry: Mapping[str, Any] | None = None,
    elements: Sequence[str] | None = None,
    method: str = "B3LYP-D3",
    charge: int = 0,
    multiplicity: int = 1,
    nprocs: int = 4,
) -> CompiledRequest:
    """Compile one frozen plan candidate into an ORCA request artifact.

    ``candidate`` is a frozen-plan ``ScanCandidateV2`` or ``PathCandidateV1``
    payload; ``atom_rows`` is the plan's common frozen atom order (map ids);
    ``capability`` is the effective backend capability (todo 16).  Raises
    :class:`OrcaCompileError` (typed ``code``) on every refusal — the
    capability gate runs before any syntax is produced.
    """
    if not isinstance(candidate, Mapping):
        raise OrcaCompileError(CODE_UNKNOWN_CANDIDATE_KIND, "candidate must be a mapping")
    order = _parse_atom_rows(atom_rows)
    map_to_index = {map_id: position for position, map_id in enumerate(order)}
    kind_tag = candidate.get("candidate_kind")
    candidate_id = str(candidate.get("candidate_id") or "cand-unknown")
    if kind_tag == CANDIDATE_KIND_PATH:
        return _compile_path(
            candidate,
            candidate_id,
            capability,
            order,
            elements=elements,
            method=method,
            charge=charge,
            multiplicity=multiplicity,
            nprocs=nprocs,
        )
    if kind_tag != CANDIDATE_KIND_SCAN:
        raise OrcaCompileError(
            CODE_UNKNOWN_CANDIDATE_KIND,
            f"candidate_kind={kind_tag!r}; expected {CANDIDATE_KIND_SCAN!r} or {CANDIDATE_KIND_PATH!r}",
        )
    return _compile_scan(
        candidate,
        candidate_id,
        capability,
        order,
        map_to_index,
        geometry=geometry,
        method=method,
        charge=charge,
        multiplicity=multiplicity,
        nprocs=nprocs,
    )


def _compile_scan(
    candidate: Mapping[str, Any],
    candidate_id: str,
    capability: EffectiveCapability,
    order: tuple[int, ...],
    map_to_index: Mapping[int, int],
    *,
    geometry: Mapping[str, Any] | None,
    method: str,
    charge: int,
    multiplicity: int,
    nprocs: int,
) -> CompiledRequest:
    mode = str(candidate.get("mode") or "")
    raw_drivers = candidate.get("drivers")
    if not isinstance(raw_drivers, (list, tuple)) or not raw_drivers:
        raise OrcaCompileError(CODE_DRIVER_MAPS_INVALID, "drivers must be a non-empty array")
    drivers = list(raw_drivers)
    n_coords = len(drivers)

    # --- capability gate FIRST (shape extraction only; no compilation) -----
    gate_kinds: list[str] = []
    for index, driver in enumerate(drivers):
        if not isinstance(driver, Mapping) or driver.get("kind") not in DRIVER_KINDS:
            raise OrcaCompileError(
                CODE_DRIVER_MAPS_INVALID,
                f"drivers[{index}].kind must be one of {', '.join(DRIVER_KINDS)}",
            )
        gate_kinds.append(str(driver["kind"]))
    _refuse_capability(capability, mode, gate_kinds, n_coords)

    if n_coords > 3:
        raise OrcaCompileError(
            CODE_BACKEND_CAPABILITY_MISSING,
            f"n_coords={n_coords} exceeds the native scan upper bound 3 "
            "(design §9.1: the contracts-surface ≤4 is not executable capability)",
        )

    # --- per-mode capability extras (still gate, before compilation) -------
    multi = n_coords >= 2
    if mode == MODE_SINGLE_1D and multi:
        raise OrcaCompileError(
            CODE_BACKEND_CAPABILITY_MISSING, "SINGLE_1D requires exactly one driver"
        )
    if mode in (MODE_COUPLED_1D, MODE_SCHEDULED_1D) and not (2 <= n_coords <= 3):
        raise OrcaCompileError(
            CODE_BACKEND_CAPABILITY_MISSING,
            f"{mode} requires 2..3 drivers, got {n_coords}",
        )
    if mode not in (MODE_SINGLE_1D, MODE_COUPLED_1D, MODE_SCHEDULED_1D):
        raise OrcaCompileError(
            CODE_UNKNOWN_CANDIDATE_KIND,
            f"mode={mode!r}; expected one of SINGLE_1D/COUPLED_1D/SCHEDULED_1D",
        )

    # --- driver parsing (indices from atom_rows, units, schedules) --------
    parsed = [
        _resolve_driver(driver, map_to_index, f"drivers[{index}]")
        for index, driver in enumerate(drivers)
    ]
    lambdas_raw = candidate.get("lambda_values")
    if not isinstance(lambdas_raw, (list, tuple)) or not lambdas_raw:
        raise OrcaCompileError(CODE_LAMBDA_GRID_INVALID, "lambda_values must be a non-empty array")
    try:
        lambdas = tuple(float(value) for value in lambdas_raw)
    except (TypeError, ValueError) as exc:
        raise OrcaCompileError(CODE_LAMBDA_GRID_INVALID, "lambda_values entries must be numbers") from exc
    if any(not math.isfinite(value) for value in lambdas):
        raise OrcaCompileError(CODE_LAMBDA_GRID_INVALID, "lambda_values entries must be finite")
    n_points = len(lambdas)
    _check_point_window(n_points, capability)

    schedule_lists: list[tuple[float, ...]] = []
    for index, (_kind, _maps, _indices, _unit, schedule_values) in enumerate(parsed):
        if schedule_values is None:
            schedule_lists.append(lambdas)
        else:
            if len(schedule_values) != n_points:
                raise OrcaCompileError(
                    CODE_SCHEDULE_VALUE_MISMATCH,
                    f"drivers[{index}].schedule_values has {len(schedule_values)} entries "
                    f"but lambda_values has {n_points}",
                )
            schedule_lists.append(schedule_values)
    all_linear = all(_is_linear(values, lambdas) for values in schedule_lists)

    # --- compilation-strategy selection (mode-driven, never implicit) ------
    simultaneous = False
    per_point = False
    if mode == MODE_COUPLED_1D:
        if not capability.constraint_support.simul_scan:
            raise OrcaCompileError(
                CODE_SIMUL_SCAN_UNSUPPORTED,
                "effective constraint_support.simul_scan is false — multi-coordinate "
                "Simul_Scan compilation is not available on this backend",
            )
        if not all_linear:
            raise OrcaCompileError(
                CODE_SIMUL_SCAN_UNSUPPORTED,
                "COUPLED_1D compiles as Simul_Scan over the shared linear λ grid; "
                "non-linear per-driver schedules must be SCHEDULED_1D per-point "
                "compilation (no implicit grids)",
            )
        if not capability.constraint_support.native_scan:
            raise OrcaCompileError(
                CODE_BACKEND_CAPABILITY_MISSING,
                "effective constraint_support.native_scan is false",
            )
        simultaneous = True
    elif mode == MODE_SCHEDULED_1D:
        per_point = True
        if not capability.custom_schedule_support:
            raise OrcaCompileError(
                CODE_CUSTOM_SCHEDULE_UNSUPPORTED,
                "effective custom_schedule_support is false — custom λ lists cannot compile",
            )
        if not capability.constraint_support.per_point_constraints:
            raise OrcaCompileError(
                CODE_PER_POINT_CONSTRAINTS_UNSUPPORTED,
                "effective constraint_support.per_point_constraints is false — "
                "SCHEDULED_1D requires per-point %geom Constraints compilation",
            )
    else:  # SINGLE_1D
        if all_linear:
            if not capability.constraint_support.native_scan:
                raise OrcaCompileError(
                    CODE_BACKEND_CAPABILITY_MISSING,
                    "effective constraint_support.native_scan is false",
                )
        else:
            per_point = True
            if not capability.custom_schedule_support:
                raise OrcaCompileError(
                    CODE_CUSTOM_SCHEDULE_UNSUPPORTED,
                    "non-linear single-driver schedules require custom_schedule_support",
                )
            if not capability.constraint_support.per_point_constraints:
                raise OrcaCompileError(
                    CODE_PER_POINT_CONSTRAINTS_UNSUPPORTED,
                    "non-linear single-driver schedules require per_point_constraints",
                )

    start_endpoint = str(candidate.get("start_endpoint") or "R")
    if start_endpoint not in ("R", "P"):
        raise OrcaCompileError(
            CODE_COORDINATE_VALUE_INVALID, f"start_endpoint={start_endpoint!r}; expected 'R' or 'P'"
        )
    if geometry is None:
        raise OrcaCompileError(
            CODE_GEOMETRY_REQUIRED,
            "endpoint geometry (R/P map→xyz blocks) is required to compile physical "
            "constraint values",
        )
    geometry_norm = _normalize_geometry(geometry)
    geom_sha = _geometry_sha256(geometry_norm)

    coordinates: list[CompiledCoordinate] = []
    for (kind, maps, indices, unit, schedule_values), s_list in zip(
        parsed, schedule_lists, strict=True
    ):
        q_start, q_end = _endpoint_q(kind, maps, geometry_norm, start_endpoint)
        if not (math.isfinite(q_start) and math.isfinite(q_end)):
            raise OrcaCompileError(
                CODE_COORDINATE_VALUE_INVALID, f"non-finite endpoint values for kind {kind} maps {maps}"
            )
        if kind == KIND_D:
            q_start = _normalize_dihedral(q_start)
            q_end = _normalize_dihedral(q_end)
        values = tuple(
            _round_value(_interpolate(kind, q_start, q_end, s_value))
            for s_value in s_list
        )
        coordinates.append(
            CompiledCoordinate(
                kind=kind,
                maps=maps,
                indices=indices,
                unit=unit,
                start_value=_round_value(q_start),
                end_value=_round_value(q_end),
                values=values,
                schedule_values=tuple(_round_value(s) for s in s_list),
            )
        )

    # --- artifact assembly -------------------------------------------------
    if per_point:
        fragments = tuple(
            _constraint_block(coordinates, point_index) for point_index in range(n_points)
        )
        hashes: list[str] = []
        for point_index in range(n_points):
            text = _input_text(
                candidate_id=candidate_id,
                point_label=f"point {point_index + 1}/{n_points}",
                method=method,
                charge=charge,
                multiplicity=multiplicity,
                nprocs=nprocs,
                geometry_sha=geom_sha,
                atom_rows=order,
                geom_block=fragments[point_index],
            )
            hashes.append(sha256_bytes(text.encode("utf-8")))
        recipe: dict[str, Any] = {
            "recipe_kind": RECIPE_PER_POINT,
            "method": method,
            "charge": charge,
            "multiplicity": multiplicity,
            "nprocs": nprocs,
            "geometry_sha256": geom_sha,
            "atom_rows": list(order),
            "mode": mode,
            "start_endpoint": start_endpoint,
            "lambda_values": list(lambdas),
            "coordinates": [row.to_record() for row in coordinates],
            "generation": (
                "point i input = method/pal header + %geom Constraints block with "
                "coordinates[k].values[i]; XYZ assembled by the adapter from the "
                "frozen atom_rows order and endpoint geometry"
            ),
        }
        compiled_kind = "recipe"
        recipe_kind = RECIPE_PER_POINT
        simultaneous_flag = False
    else:
        fragment = _native_scan_block(coordinates, n_points, simultaneous)
        fragments = (fragment,)
        text = _input_text(
            candidate_id=candidate_id,
            point_label="scan",
            method=method,
            charge=charge,
            multiplicity=multiplicity,
            nprocs=nprocs,
            geometry_sha=geom_sha,
            atom_rows=order,
            geom_block=fragment,
        )
        hashes = [sha256_bytes(text.encode("utf-8"))]
        recipe = {
            "recipe_kind": RECIPE_NATIVE_SCAN,
            "method": method,
            "charge": charge,
            "multiplicity": multiplicity,
            "nprocs": nprocs,
            "geometry_sha256": geom_sha,
            "atom_rows": list(order),
            "mode": mode,
            "start_endpoint": start_endpoint,
            "lambda_values": list(lambdas),
            "coordinates": [row.to_record() for row in coordinates],
            "simul_scan": simultaneous,
            "steps": n_points - 1,
            "generation": (
                "single ORCA input with %geom Scan per driver "
                "(steps = total_points − 1); Simul_Scan true when simultaneous"
            ),
        }
        compiled_kind = "hashes"
        recipe_kind = RECIPE_NATIVE_SCAN
        simultaneous_flag = simultaneous

    return CompiledRequest(
        schema_version=SCHEMA_ORCA_COMPILE,
        candidate_id=candidate_id,
        mode=mode,
        compiled_kind=compiled_kind,
        recipe_kind=recipe_kind,
        geom_fragments=fragments,
        point_input_sha256=tuple(hashes),
        total_points=n_points,
        coordinates=tuple(coordinates),
        simultaneous=simultaneous_flag,
        lambda_values=lambdas,
        atom_rows=order,
        recipe=recipe,
    )


def _compile_path(
    candidate: Mapping[str, Any],
    candidate_id: str,
    capability: EffectiveCapability,
    order: tuple[int, ...],
    *,
    elements: Sequence[str] | None,
    method: str,
    charge: int,
    multiplicity: int,
    nprocs: int,
) -> CompiledRequest:
    method_kind = candidate.get("method_kind")
    if method_kind is not None and method_kind not in PATH_METHOD_KINDS:
        raise OrcaCompileError(
            CODE_PATH_METHOD_INVALID,
            f"method_kind={method_kind!r}; expected one of {', '.join(PATH_METHOD_KINDS)}",
        )
    # capability gate FIRST
    _refuse_capability(capability, "PATH_NEB", (), 0)

    n_atoms_raw = candidate.get("n_atoms")
    if not isinstance(n_atoms_raw, int) or isinstance(n_atoms_raw, bool) or n_atoms_raw < 1:
        raise OrcaCompileError(CODE_PATH_GEOMETRY_INVALID, "n_atoms must be a positive integer")
    n_atoms = n_atoms_raw
    if len(order) != n_atoms:
        raise OrcaCompileError(
            CODE_PATH_GEOMETRY_INVALID,
            f"atom_rows has {len(order)} entries but n_atoms={n_atoms}",
        )
    geometries = candidate.get("endpoint_geometries")
    if not isinstance(geometries, Mapping):
        raise OrcaCompileError(
            CODE_PATH_GEOMETRY_INVALID, "endpoint_geometries must be a reactant/product object"
        )
    blocks: dict[str, list[list[float]]] = {}
    for side in ("reactant", "product"):
        raw_block = geometries.get(side)
        if not isinstance(raw_block, (list, tuple)) or len(raw_block) != n_atoms:
            raise OrcaCompileError(
                CODE_PATH_GEOMETRY_INVALID,
                f"endpoint_geometries.{side} must contain exactly n_atoms={n_atoms} rows",
            )
        rows: list[list[float]] = []
        for index, row in enumerate(raw_block):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise OrcaCompileError(
                    CODE_PATH_GEOMETRY_INVALID,
                    f"endpoint_geometries.{side}[{index}] must be one 3D coordinate",
                )
            try:
                xyz = [float(row[0]), float(row[1]), float(row[2])]
            except (TypeError, ValueError) as exc:
                raise OrcaCompileError(
                    CODE_PATH_GEOMETRY_INVALID,
                    f"endpoint_geometries.{side}[{index}] coordinates must be numbers",
                ) from exc
            if any(not math.isfinite(component) for component in xyz):
                raise OrcaCompileError(
                    CODE_PATH_GEOMETRY_INVALID,
                    f"endpoint_geometries.{side}[{index}] coordinates must be finite",
                )
            rows.append(xyz)
        blocks[side] = rows
    image_chain = candidate.get("image_chain")
    if not isinstance(image_chain, Mapping):
        raise OrcaCompileError(CODE_PATH_GEOMETRY_INVALID, "image_chain must be an object")
    n_images = image_chain.get("n_images")
    if not isinstance(n_images, int) or isinstance(n_images, bool) or n_images < 1:
        raise OrcaCompileError(
            CODE_PATH_GEOMETRY_INVALID, "image_chain.n_images must be a positive integer"
        )

    endpoint_sha = sha256_bytes(stable_json_dumps(blocks).encode("utf-8"))
    path_fragment = _path_block(n_images)
    reactant_block = _xyz_block(blocks["reactant"], charge, multiplicity, elements)
    product_block = _xyz_block(blocks["product"], charge, multiplicity, elements)
    fragments = (path_fragment, reactant_block, product_block)
    text = "\n".join(
        [
            f"pes2ts-orca-compile {candidate_id} path",
            f"! {method}",
            "%pal",
            f" nprocs {nprocs}",
            "end",
            f"# endpoint_sha256={endpoint_sha} n_images={n_images} atom_rows={list(order)}",
            path_fragment,
            "# reactant endpoint",
            reactant_block,
            "# product endpoint",
            product_block,
            "",
        ]
    )
    point_hash = sha256_bytes(text.encode("utf-8"))
    recipe: dict[str, Any] = {
        "recipe_kind": RECIPE_PATH_NEB,
        "method": method,
        "method_kind": method_kind or "NEB",
        "charge": charge,
        "multiplicity": multiplicity,
        "nprocs": nprocs,
        "n_atoms": n_atoms,
        "n_images": n_images,
        "atom_rows": list(order),
        "endpoint_sha256": endpoint_sha,
        "endpoint_geometries": blocks,
        "generation": (
            "minimal NEB shape (todo 26 elaborates): endpoint xyz blocks in the "
            "frozen atom order + %geom Path n_images; elements bind at the adapter"
        ),
    }
    return CompiledRequest(
        schema_version=SCHEMA_ORCA_COMPILE,
        candidate_id=candidate_id,
        mode="PATH_NEB",
        compiled_kind="hashes",
        recipe_kind=RECIPE_PATH_NEB,
        geom_fragments=fragments,
        point_input_sha256=(point_hash,),
        total_points=n_images,
        coordinates=(),
        simultaneous=False,
        lambda_values=(),
        atom_rows=order,
        recipe=recipe,
    )


def _path_atom_rows_tuple(raw: Any) -> tuple[int, ...]:
    """Parse a frozen atom-order array into a positive-int tuple."""
    if not isinstance(raw, (list, tuple)) or not raw:
        raise OrcaCompileError(
            CODE_PATH_GEOMETRY_INVALID,
            "atom_rows must be a non-empty array of positive integers",
        )
    out: list[int] = []
    for entry in raw:
        if isinstance(entry, bool) or not isinstance(entry, int) or entry <= 0:
            raise OrcaCompileError(
                CODE_PATH_GEOMETRY_INVALID, "atom_rows entries must be positive integers"
            )
        out.append(entry)
    return tuple(out)


def _path_extension_block(candidate: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the namespaced ``extensions.path_request`` block when present."""
    extensions = candidate.get("extensions")
    if not isinstance(extensions, Mapping):
        return None
    block = extensions.get("path_request")
    return block if isinstance(block, Mapping) else None


def _json_scalar_map(raw: Any) -> dict[str, Any]:
    """Project a mapping into a JSON-scalar dict (stable-hash safe)."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
    return out


def _xyz_rows(raw: Any, where: str) -> list[list[float]]:
    """Parse an endpoint geometry block into rows of three finite floats."""
    if not isinstance(raw, (list, tuple)):
        raise OrcaCompileError(CODE_PATH_GEOMETRY_INVALID, f"{where} must be an array")
    rows: list[list[float]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise OrcaCompileError(
                CODE_PATH_GEOMETRY_INVALID, f"{where}[{index}] must be one 3D coordinate"
            )
        coords: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise OrcaCompileError(
                    CODE_PATH_GEOMETRY_INVALID,
                    f"{where}[{index}] coordinates must be numbers",
                )
            coords.append(float(item))
        rows.append(coords)
    return rows


def compile_orca_path_request(
    path_request: Any,
    capability: EffectiveCapability,
    *,
    elements: Sequence[str] | None = None,
    method: str = "B3LYP-D3",
    charge: int = 0,
    multiplicity: int = 1,
    nprocs: int = 4,
) -> CompiledRequest:
    """Compile a PathRequest (todo 26) into the ORCA NEB input shape.

    Additive entry point: projects the request through
    ``PathRequest.to_path_candidate`` (duck-typed — no import of
    ``path_request`` from this module), runs the todo-19 capability-gated
    ``compile_orca`` NEB path unchanged, then binds the frozen image-chain
    parameters and recovery protocol into the recipe and the hashed input
    text.  The ``%geom Path`` fragment keeps the minimal ``n_images`` shape;
    spring/backend parameters are frozen in the recipe for the backend —
    this module never invents unverified ORCA keywords.

    ``path_request`` may be a PathRequest dataclass (exposes
    ``to_path_candidate``/``atom_rows``/``elements``) or a PathCandidateV1
    mapping carrying ``extensions.path_request.atom_rows``.
    """
    candidate: dict[str, Any]
    order: tuple[int, ...]
    request_block: Mapping[str, Any] | None
    request_id: str
    to_candidate = getattr(path_request, "to_path_candidate", None)
    if callable(to_candidate):
        request_id = str(getattr(path_request, "request_id", "") or "path-request")
        raw_candidate = to_candidate(candidate_id=request_id)
        if not isinstance(raw_candidate, Mapping):
            raise OrcaCompileError(
                CODE_UNKNOWN_CANDIDATE_KIND, "to_path_candidate() must return a mapping"
            )
        candidate = dict(raw_candidate)
        order = _path_atom_rows_tuple(getattr(path_request, "atom_rows", ()))
        if elements is None:
            raw_elements = getattr(path_request, "elements", ()) or ()
            elements = tuple(str(entry) for entry in raw_elements)
        request_block = _path_extension_block(candidate)
    elif isinstance(path_request, Mapping) and path_request.get(
        "candidate_kind"
    ) == CANDIDATE_KIND_PATH:
        candidate = dict(path_request)
        request_block = _path_extension_block(candidate)
        raw_order = request_block.get("atom_rows") if request_block is not None else None
        order = _path_atom_rows_tuple(raw_order)
        raw_id = request_block.get("request_id") if request_block is not None else None
        request_id = str(raw_id or candidate.get("candidate_id") or "path-request")
    else:
        raise OrcaCompileError(
            CODE_UNKNOWN_CANDIDATE_KIND,
            "path_request must be a PathRequest or a PathCandidateV1 mapping",
        )
    if not order:
        raise OrcaCompileError(CODE_PATH_GEOMETRY_INVALID, "atom_rows must be non-empty")
    base = compile_orca(
        candidate,
        order,
        capability,
        elements=elements,
        method=method,
        charge=charge,
        multiplicity=multiplicity,
        nprocs=nprocs,
    )
    recipe = dict(base.recipe)
    raw_chain = candidate.get("image_chain")
    image_chain_doc = _json_scalar_map(raw_chain)
    recipe["image_chain"] = image_chain_doc
    schema_version = "g1_path_request_v1"
    if request_block is not None and request_block.get("schema_version") is not None:
        schema_version = str(request_block.get("schema_version"))
    recipe["path_request"] = {
        "schema_version": schema_version,
        "request_id": request_id,
    }
    if request_block is not None:
        recovery = request_block.get("recovery_protocol")
        if isinstance(recovery, Mapping):
            recipe["recovery_protocol"] = _json_scalar_map(recovery)
        channels = request_block.get("method_channels")
        if isinstance(channels, Mapping):
            recipe["method_channels"] = _json_scalar_map(channels)
    raw_blocks = recipe.get("endpoint_geometries")
    if not isinstance(raw_blocks, Mapping):
        raise OrcaCompileError(
            CODE_PATH_GEOMETRY_INVALID, "compiled recipe lost endpoint_geometries"
        )
    reactant_rows = _xyz_rows(raw_blocks.get("reactant"), "endpoint_geometries.reactant")
    product_rows = _xyz_rows(raw_blocks.get("product"), "endpoint_geometries.product")
    n_images = int(base.total_points)
    path_fragment = _path_block(n_images)
    reactant_block = _xyz_block(reactant_rows, charge, multiplicity, elements)
    product_block = _xyz_block(product_rows, charge, multiplicity, elements)
    endpoint_sha = str(recipe.get("endpoint_sha256") or "")
    text = "\n".join(
        [
            f"pes2ts-orca-compile {base.candidate_id} path",
            f"! {method}",
            "%pal",
            f" nprocs {nprocs}",
            "end",
            f"# endpoint_sha256={endpoint_sha} n_images={n_images} atom_rows={list(order)}",
            f"# path_request {schema_version} "
            f"{request_id} image_chain={stable_json_dumps(image_chain_doc)}",
            path_fragment,
            "# reactant endpoint",
            reactant_block,
            "# product endpoint",
            product_block,
            "",
        ]
    )
    point_hash = sha256_bytes(text.encode("utf-8"))
    return CompiledRequest(
        schema_version=SCHEMA_ORCA_COMPILE,
        candidate_id=base.candidate_id,
        mode="PATH_NEB",
        compiled_kind="hashes",
        recipe_kind=RECIPE_PATH_NEB,
        geom_fragments=base.geom_fragments,
        point_input_sha256=(point_hash,),
        total_points=n_images,
        coordinates=(),
        simultaneous=False,
        lambda_values=(),
        atom_rows=order,
        recipe=recipe,
    )
