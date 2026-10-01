"""Geometry feasibility gates, Jacobian rank, and point-count rule (todo 13).

Implements design §6.3 (geometry feasibility) and §8.3 (points & budget) of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``:

- **Coordinate definition validation** — kind/units/index correctness, no
  duplicate atoms inside a coordinate, B distances positive, A/D non-degenerate
  at both endpoint geometries (A angle away from 0/180°, D well-defined: no
  collinear central-bond triples), and D periodic unwrapping (Δdihedral via
  shortest arc in (−180°, 180°]; ±180° handled, never a raw ±360° jump).
- **Endpoint graph/geometry cross-checks** — element-pair bond thresholds from
  Cordero covalent-radius sums plus ``scan_strategy.element_pair_bond_thresholds.tolerance``
  (todo 2 config key); wrong broken-bond endpoints (R-graph bond whose R
  geometry is already separated) and formed bonds missing at P; cross-component
  atom overlap; triangle inequality / distance-geometry realizability for
  multi-B sets (H2 three-distance case included).
- **Unit-scaled Jacobian** — J = ∂q/∂x for the candidate's driver coordinates
  at both endpoint geometries, central finite differences, each q-row scaled
  by ``max_step_by_kind[kind]`` so B (Å) and A/D (°) rows are comparable;
  rank(J) must equal n_coords and κ(J) ≤ ``jacobian_condition_number_max``.
- **Rigid-assembly / constraint-projection precheck** — Cartesian linear
  interpolation between the two endpoint geometries (cheap geometric precheck
  only, never a computed PES): interpolated driver values stay in physical
  ranges and no severe nonbonded clash appears along the path; start-point
  target coordinates must match the starting geometry (no teleporting scan).
- **Point count rule** — ``N_required = 1 + max_j ceil(total_variation(q_j) /
  max_step_j)`` over the λ-parameterized path (linear interpolation default;
  D total variation uses the shortest-arc Δ), ``N = max(N_baseline,
  N_required)``; N above ``point_limits.max`` yields the typed verdict
  ``POINT_BUDGET_EXCEEDED`` plus a suggestion (segmentation / method change /
  reject — the decision is todo 17's; here only the typed verdict).

**Guardrail (plan Must-NOT-do):** the legacy 0.45–8 Å crude distance range
from ``planning.py`` is *never* a qualification gate here. Bondedness uses
Cordero radius sums + configured tolerance only.

Pure functions only: no file IO, no config loading (the config mapping is an
input), no data-tree reads, no truth. Inputs are the todo-12 coordinate pool,
the todo-5 endpoint graph bundle, and map-keyed endpoint coordinates
(``EndpointMaterials`` or ``{"r": ..., "p": ...}``).
"""

# noqa: SIZE_OK — plan-named todo-13 single module (design §6.3 geometry gates
# + §8.3 point-count rule + unit-scaled Jacobian + path precheck); precedent:
# coordinate_pool.py (todo 12), registry.py (todo 11), contracts_v2.py (todo 3).

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.scan_strategy.coordinate_pool import (
    KIND_A,
    KIND_B,
    KIND_D,
    UNIT_ANGSTROM,
    UNIT_DEGREE,
    CoordinatePool,
    CoordinateRecord,
    DriverSetCandidate,
    EndpointMaterials,
)
from pes2ts_core.utils.hashing import stable_json_dumps

# ---------------------------------------------------------------------------
# Schema + vocabulary.
# ---------------------------------------------------------------------------
SCHEMA_GEOMETRY_FEASIBILITY: Final[str] = "g1_geometry_feasibility_v1"

CHECK_COORDINATE_DEFINITION: Final[str] = "coordinate_definition"
CHECK_ENDPOINT_DEGENERACY: Final[str] = "endpoint_degeneracy"
CHECK_DIHEDRAL_UNWRAP: Final[str] = "dihedral_periodic_unwrap"
CHECK_ENDPOINT_BOND_STATE: Final[str] = "endpoint_bond_state"
CHECK_CROSS_COMPONENT_OVERLAP: Final[str] = "cross_component_overlap"
CHECK_TRIANGLE_INEQUALITY: Final[str] = "triangle_inequality"
CHECK_JACOBIAN_RANK: Final[str] = "jacobian_rank"
CHECK_JACOBIAN_CONDITION: Final[str] = "jacobian_condition"
CHECK_PATH_PRECHECK: Final[str] = "path_precheck"
CHECK_START_GEOMETRY_MATCH: Final[str] = "start_geometry_match"
CHECK_POINT_COUNT: Final[str] = "point_count"

CHECK_ORDER: Final[tuple[str, ...]] = (
    CHECK_COORDINATE_DEFINITION,
    CHECK_ENDPOINT_DEGENERACY,
    CHECK_DIHEDRAL_UNWRAP,
    CHECK_ENDPOINT_BOND_STATE,
    CHECK_CROSS_COMPONENT_OVERLAP,
    CHECK_TRIANGLE_INEQUALITY,
    CHECK_JACOBIAN_RANK,
    CHECK_JACOBIAN_CONDITION,
    CHECK_PATH_PRECHECK,
    CHECK_START_GEOMETRY_MATCH,
    CHECK_POINT_COUNT,
)

STATUS_PASS: Final[str] = "pass"
STATUS_FAIL: Final[str] = "fail"
STATUS_SKIPPED: Final[str] = "skipped"

CODE_COORDINATE_SCHEMA_INVALID: Final[str] = "COORDINATE_SCHEMA_INVALID"
CODE_DUPLICATE_ATOMS: Final[str] = "DUPLICATE_ATOMS_IN_COORDINATE"
CODE_UNKNOWN_MAP_ID: Final[str] = "UNKNOWN_MAP_ID"
CODE_UNIT_MISMATCH: Final[str] = "UNIT_MISMATCH"
CODE_B_DISTANCE_NONPOSITIVE: Final[str] = "B_DISTANCE_NONPOSITIVE"
CODE_ANGLE_DEGENERATE: Final[str] = "ANGLE_DEGENERATE"
CODE_DIHEDRAL_UNDEFINED: Final[str] = "DIHEDRAL_UNDEFINED"
CODE_BROKEN_BOND_ALREADY_SEPARATED: Final[str] = "BROKEN_BOND_ALREADY_SEPARATED"
CODE_FORMED_BOND_MISSING_AT_P: Final[str] = "FORMED_BOND_MISSING_AT_P"
CODE_BONDED_DISTANCE_IMPLAUSIBLE: Final[str] = "BONDED_DISTANCE_IMPLAUSIBLE"
CODE_CROSS_COMPONENT_OVERLAP: Final[str] = "CROSS_COMPONENT_OVERLAP"
CODE_TRIANGLE_INEQUALITY_VIOLATION: Final[str] = "TRIANGLE_INEQUALITY_VIOLATION"
CODE_JACOBIAN_RANK_DEFICIENT: Final[str] = "JACOBIAN_RANK_DEFICIENT"
CODE_JACOBIAN_ILL_CONDITIONED: Final[str] = "JACOBIAN_ILL_CONDITIONED"
CODE_INTERPOLATION_INFEASIBLE: Final[str] = "INTERPOLATION_INFEASIBLE"
CODE_PATH_COLLISION: Final[str] = "PATH_COLLISION"
CODE_START_TELEPORT: Final[str] = "START_TELEPORT"
CODE_POINT_BUDGET_EXCEEDED: Final[str] = "POINT_BUDGET_EXCEEDED"
CODE_MATERIALS_REQUIRED: Final[str] = "MATERIALS_REQUIRED"
CODE_POLICY_INVALID: Final[str] = "POLICY_INVALID"
CODE_START_ENDPOINT_INVALID: Final[str] = "START_ENDPOINT_INVALID"

SUGGESTION_POINT_BUDGET: Final[str] = (
    "segment_scan_or_change_path_method_or_reject; decision owned by todo 17"
)

# --- Tolerances (TODO: calibrate — 待校准) ---------------------------------
ANGLE_DEGENERACY_TOL_DEG: Final[float] = 1.0
D_COLLINEAR_SIN_TOL: Final[float] = 1.0e-3
B_POSITIVE_EPS: Final[float] = 1.0e-8
BONDED_LOWER_FRACTION: Final[float] = 0.5
BONDED_LOWER_FLOOR: Final[float] = 0.4
JACOBIAN_FD_STEP_ANG: Final[float] = 1.0e-4
TRIANGLE_EPS_ANG: Final[float] = 1.0e-6
PATH_PRECHECK_LAMBDAS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
DEFAULT_COLLISION_MIN_DISTANCE: Final[float] = 0.8

DEFAULT_POINT_BASELINE: Final[int] = 9
DEFAULT_POINT_MAX: Final[int] = 101
DEFAULT_MAX_STEP_DISTANCE: Final[float] = 0.2
DEFAULT_MAX_STEP_ANGLE: Final[float] = 10.0
DEFAULT_MAX_STEP_DIHEDRAL: Final[float] = 10.0
DEFAULT_JACOBIAN_CONDITION_MAX: Final[float] = 1.0e6
DEFAULT_BOND_TOLERANCE: Final[float] = 0.45

_START_ALIASES: Final[dict[str, str]] = {
    "R": "R", "r": "R", "reactant": "R",
    "P": "P", "p": "P", "product": "P",
}


# ---------------------------------------------------------------------------
# Policy (config boundary).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeasibilityPolicy:
    """Versioned numeric policy for geometry feasibility + point count."""

    point_baseline: int
    point_max: int
    max_step_distance: float
    max_step_angle: float
    max_step_dihedral: float
    jacobian_condition_number_max: float
    bond_tolerance: float
    collision_min_distance: float

    def max_step_for(self, kind: str) -> float:
        """Return the configured maximum step for a coordinate kind."""
        if kind == KIND_B:
            return self.max_step_distance
        if kind == KIND_A:
            return self.max_step_angle
        if kind == KIND_D:
            return self.max_step_dihedral
        raise ValueError(f"POLICY_INVALID: unknown coordinate kind {kind!r}")


def policy_from_config(config: Mapping[str, Any]) -> FeasibilityPolicy:
    """Parse ``scan_strategy`` feasibility keys from a loaded config mapping.

    Missing keys fall back to the documented defaults mirroring
    ``config/defaults.yaml`` (todo 2). Present-but-invalid values raise
    ``ValueError`` with a ``POLICY_INVALID:`` prefix.
    """
    section = config.get("scan_strategy")
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError("POLICY_INVALID: scan_strategy must be a mapping")
    limits = section.get("point_limits")
    if limits is None:
        limits = {}
    if not isinstance(limits, Mapping):
        raise ValueError("POLICY_INVALID: scan_strategy.point_limits must be a mapping")
    steps = section.get("max_step_by_kind")
    if steps is None:
        steps = {}
    if not isinstance(steps, Mapping):
        raise ValueError(
            "POLICY_INVALID: scan_strategy.max_step_by_kind must be a mapping"
        )
    thresholds = section.get("element_pair_bond_thresholds")
    if thresholds is None:
        thresholds = {}
    if not isinstance(thresholds, Mapping):
        raise ValueError(
            "POLICY_INVALID: scan_strategy.element_pair_bond_thresholds must be a mapping"
        )

    baseline = _int_key(limits, "baseline", DEFAULT_POINT_BASELINE)
    point_max = _int_key(limits, "max", DEFAULT_POINT_MAX)
    if baseline < 1:
        raise ValueError("POLICY_INVALID: point_limits.baseline must be >= 1")
    if point_max < baseline:
        raise ValueError(
            "POLICY_INVALID: point_limits.max must be >= point_limits.baseline"
        )
    max_step_distance = _float_key(
        steps, "distance", DEFAULT_MAX_STEP_DISTANCE, "max_step_by_kind.distance"
    )
    max_step_angle = _float_key(
        steps, "angle", DEFAULT_MAX_STEP_ANGLE, "max_step_by_kind.angle"
    )
    max_step_dihedral = _float_key(
        steps, "dihedral", DEFAULT_MAX_STEP_DIHEDRAL, "max_step_by_kind.dihedral"
    )
    jmax = _float_key(
        section,
        "jacobian_condition_number_max",
        DEFAULT_JACOBIAN_CONDITION_MAX,
        "jacobian_condition_number_max",
    )
    bond_tolerance = _float_key(
        thresholds,
        "tolerance",
        DEFAULT_BOND_TOLERANCE,
        "element_pair_bond_thresholds.tolerance",
    )
    return FeasibilityPolicy(
        point_baseline=baseline,
        point_max=point_max,
        max_step_distance=max_step_distance,
        max_step_angle=max_step_angle,
        max_step_dihedral=max_step_dihedral,
        jacobian_condition_number_max=jmax,
        bond_tolerance=bond_tolerance,
        collision_min_distance=_collision_min_distance(config),
    )


def _collision_min_distance(config: Mapping[str, Any]) -> float:
    """Read g2.validity.collision_min_distance, else the documented default."""
    g2 = config.get("g2")
    if not isinstance(g2, Mapping):
        return DEFAULT_COLLISION_MIN_DISTANCE
    validity = g2.get("validity")
    if not isinstance(validity, Mapping):
        return DEFAULT_COLLISION_MIN_DISTANCE
    raw = validity.get("collision_min_distance", DEFAULT_COLLISION_MIN_DISTANCE)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(
            "POLICY_INVALID: g2.validity.collision_min_distance must be numeric"
        )
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "POLICY_INVALID: g2.validity.collision_min_distance must be finite and > 0"
        )
    return value


def _int_key(section: Mapping[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"POLICY_INVALID: {key} must be an integer")
    if isinstance(raw, str):
        # YAML 1.1 quirk: plain ints stay ints, but be strict about strings.
        try:
            return int(raw, 10)
        except ValueError:
            raise ValueError(f"POLICY_INVALID: {key} must be an integer") from None
    if not isinstance(raw, int):
        raise ValueError(f"POLICY_INVALID: {key} must be an integer")
    return raw


def _float_key(
    section: Mapping[str, Any], key: str, default: float, label: str
) -> float:
    raw = section.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"POLICY_INVALID: {label} must be numeric")
    if isinstance(raw, str):
        # YAML 1.1 parses "1.0e6" as a string — coerce at the config boundary.
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"POLICY_INVALID: {label} must be numeric") from None
    elif isinstance(raw, int | float):
        value = float(raw)
    else:
        raise ValueError(f"POLICY_INVALID: {label} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"POLICY_INVALID: {label} must be finite and > 0")
    return value


# ---------------------------------------------------------------------------
# Report dataclasses.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CheckResult:
    """One feasibility check verdict."""

    check_id: str
    status: str
    detail: str

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {"check_id": self.check_id, "status": self.status, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class FeasibilityFailure:
    """One typed rejection with the check that produced it."""

    check_id: str
    code: str
    detail: str
    suggestion: str | None = None

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "check_id": self.check_id,
            "code": self.code,
            "detail": self.detail,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True, slots=True)
class FeasibilityReport:
    """Outcome of :func:`assess_geometry_feasibility` for one candidate."""

    schema_version: str
    ok: bool
    candidate_id: str
    start_endpoint: str
    checks: tuple[CheckResult, ...]
    failures: tuple[FeasibilityFailure, ...]
    n_points: int | None
    n_points_required: int | None
    n_points_baseline: int
    point_budget_exceeded: bool
    total_variation_by_coord: tuple[tuple[str, float], ...]
    jacobian_rank_by_endpoint: tuple[tuple[str, int, int], ...]
    jacobian_condition_max: float | None

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection; never contains truth-derived keys."""
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "candidate_id": self.candidate_id,
            "start_endpoint": self.start_endpoint,
            "checks": [check.to_record() for check in self.checks],
            "failures": [failure.to_record() for failure in self.failures],
            "n_points": self.n_points,
            "n_points_required": self.n_points_required,
            "n_points_baseline": self.n_points_baseline,
            "point_budget_exceeded": self.point_budget_exceeded,
            "total_variation_by_coord": [
                {"coordinate_id": cid, "total_variation": value}
                for cid, value in self.total_variation_by_coord
            ],
            "jacobian_rank_by_endpoint": [
                {"endpoint": endpoint, "rank": rank, "n_coords": n_coords}
                for endpoint, rank, n_coords in self.jacobian_rank_by_endpoint
            ],
            "jacobian_condition_max": self.jacobian_condition_max,
        }

    def to_json(self) -> str:
        """Canonical serialization (determinism test surface)."""
        return stable_json_dumps(self.to_doc())


# ---------------------------------------------------------------------------
# Public geometry primitives (shared with tests and later todos).
# ---------------------------------------------------------------------------
def bond_length(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> float:
    """Return the Euclidean distance between two points (angstrom)."""
    return math.dist(first, second)


def bond_angle_deg(
    first: tuple[float, float, float],
    vertex: tuple[float, float, float],
    third: tuple[float, float, float],
) -> float:
    """Return the angle first–vertex–third in degrees, in [0, 180]."""
    v1 = (first[0] - vertex[0], first[1] - vertex[1], first[2] - vertex[2])
    v2 = (third[0] - vertex[0], third[1] - vertex[1], third[2] - vertex[2])
    n1 = math.sqrt(sum(c * c for c in v1))
    n2 = math.sqrt(sum(c * c for c in v2))
    if n1 <= B_POSITIVE_EPS or n2 <= B_POSITIVE_EPS:
        return 0.0
    cosine = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def dihedral_deg(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    """Return the signed dihedral p0–p1–p2–p3 in degrees, in (−180, 180]."""
    b0 = (p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2])
    b1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    b2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
    b1_norm = _normalize(b1)
    b1_dot_b0 = _dot(b0, b1_norm)
    b1_dot_b2 = _dot(b2, b1_norm)
    v = (
        b0[0] - b1_dot_b0 * b1_norm[0],
        b0[1] - b1_dot_b0 * b1_norm[1],
        b0[2] - b1_dot_b0 * b1_norm[2],
    )
    w = (
        b2[0] - b1_dot_b2 * b1_norm[0],
        b2[1] - b1_dot_b2 * b1_norm[1],
        b2[2] - b1_dot_b2 * b1_norm[2],
    )
    x = _dot(v, w)
    y = _dot(_cross(b1_norm, v), w)
    return math.degrees(math.atan2(y, x))


def shortest_arc_delta(start_deg: float, end_deg: float) -> float:
    """Return the shortest-arc dihedral difference end−start in [−180, 180).

    350° → −10° is coterminal (Δ = 0, never a raw −360° jump); 170° → −170°
    crosses the ±180° cut and yields +20°, not −340°.
    """
    return (end_deg - start_deg + 180.0) % 360.0 - 180.0


def dihedral_is_undefined(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> bool:
    """True when a collinear triple makes the p0–p1–p2–p3 dihedral undefined."""
    b1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    b0 = (p0[0] - p1[0], p0[1] - p1[1], p0[2] - p1[2])
    b2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
    n_b1 = math.sqrt(sum(c * c for c in b1))
    n_b0 = math.sqrt(sum(c * c for c in b0))
    n_b2 = math.sqrt(sum(c * c for c in b2))
    if n_b1 <= B_POSITIVE_EPS or n_b0 <= B_POSITIVE_EPS or n_b2 <= B_POSITIVE_EPS:
        return True
    sin_first = math.sqrt(sum(c * c for c in _cross(b0, b1))) / (n_b0 * n_b1)
    sin_second = math.sqrt(sum(c * c for c in _cross(b2, b1))) / (n_b2 * n_b1)
    return sin_first < D_COLLINEAR_SIN_TOL or sin_second < D_COLLINEAR_SIN_TOL


def jacobian_diagnostics(j_matrix: np.ndarray) -> tuple[int, float]:
    """Return (rank, condition number) of a unit-scaled Jacobian matrix.

    Condition number is σ_max/σ_min; rank-deficient matrices report ``inf``.
    """
    if j_matrix.size == 0:
        return 0, math.inf
    singular = np.linalg.svd(j_matrix, compute_uv=False)
    if singular.size == 0 or not np.all(np.isfinite(singular)):
        return 0, math.inf
    sigma_max = float(singular[0])
    sigma_min = float(singular[-1])
    if sigma_max <= 0.0:
        return 0, math.inf
    tolerance = max(j_matrix.shape) * float(np.finfo(float).eps) * sigma_max
    rank = int(np.count_nonzero(singular > tolerance))
    if sigma_min <= tolerance:
        return rank, math.inf
    return rank, sigma_max / sigma_min


# ---------------------------------------------------------------------------
# Materials boundary (parse-don't-validate).
# ---------------------------------------------------------------------------
def normalize_endpoint_materials(
    materials: EndpointMaterials | Mapping[str, Any] | None,
) -> EndpointMaterials | None:
    """Normalize untrusted materials input into typed endpoint coordinates."""
    if materials is None:
        return None
    if isinstance(materials, EndpointMaterials):
        return materials
    if not isinstance(materials, Mapping):
        raise ValueError(
            "MATERIALS_SCHEMA_INVALID: expected EndpointMaterials, mapping with "
            "'r'/'p' keys, or None"
        )
    sides: dict[str, dict[int, tuple[float, float, float]]] = {}
    for side in ("r", "p"):
        raw_side = materials.get(side)
        if raw_side is None:
            continue
        if not isinstance(raw_side, Mapping):
            raise ValueError(
                f"MATERIALS_SCHEMA_INVALID: side {side!r} must be a map→xyz mapping"
            )
        parsed: dict[int, tuple[float, float, float]] = {}
        for key, value in raw_side.items():
            map_id = int(key)
            parsed[map_id] = _parse_xyz(value, side, map_id)
        sides[side] = parsed
    if not sides:
        raise ValueError("MATERIALS_SCHEMA_INVALID: no 'r'/'p' coordinate blocks")
    return EndpointMaterials(
        r_coordinates=sides.get("r", {}),
        p_coordinates=sides.get("p", {}),
    )


def _parse_xyz(value: object, side: str, map_id: int) -> tuple[float, float, float]:
    """Parse one finite xyz triple or raise a typed schema error."""
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(
            f"MATERIALS_SCHEMA_INVALID: {side}[{map_id}].coordinates must be length-3"
        )
    coords: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError(
                f"MATERIALS_SCHEMA_INVALID: {side}[{map_id}].coordinates must be numeric"
            )
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(
                f"MATERIALS_SCHEMA_INVALID: {side}[{map_id}].coordinates must be finite"
            )
        coords.append(number)
    return (coords[0], coords[1], coords[2])


# ---------------------------------------------------------------------------
# Graph / element helpers.
# ---------------------------------------------------------------------------
def elements_by_map(bundle: EndpointGraphBundle) -> dict[int, str]:
    """Return map_id → element symbol from both endpoint graphs."""
    elements = {int(node.map_id): str(node.element) for node in bundle.r_graph.nodes}
    for node in bundle.p_graph.nodes:
        elements.setdefault(int(node.map_id), str(node.element))
    return elements


def bonded_pairs(graph: Any) -> set[tuple[int, int]]:
    """Return the sorted map pairs bonded in one side graph."""
    out: set[tuple[int, int]] = set()
    for edge in graph.edges:
        a, b = int(edge.map_a), int(edge.map_b)
        out.add((a, b) if a <= b else (b, a))
    return out


def component_of_map(graph: Any) -> dict[int, int]:
    """Return map_id → component_id for one side graph."""
    mapping: dict[int, int] = {}
    for component in graph.components:
        for map_id in component.map_ids:
            mapping[int(map_id)] = int(component.component_id)
    return mapping


def covalent_radius_sum(first: str, second: str) -> float:
    """Return the Cordero radius sum for two element symbols (fallback 1.0 Å each)."""
    try:
        z1 = ELEMENT_SYMBOLS.index(first) + 1
        z2 = ELEMENT_SYMBOLS.index(second) + 1
    except ValueError:
        return 2.0
    return COVALENT_RADII.get(z1, 1.0) + COVALENT_RADII.get(z2, 1.0)


# ---------------------------------------------------------------------------
# Coordinate evaluation + Jacobian.
# ---------------------------------------------------------------------------
def evaluate_coordinate(
    record: CoordinateRecord,
    xyz: Mapping[int, tuple[float, float, float]],
) -> float | None:
    """Evaluate one pool coordinate against map-keyed coordinates (native units)."""
    points = [xyz.get(map_id) for map_id in record.atom_maps]
    if any(point is None for point in points):
        return None
    resolved = [point for point in points if point is not None]
    if record.kind == KIND_B:
        return bond_length(resolved[0], resolved[1])
    if record.kind == KIND_A:
        return bond_angle_deg(resolved[0], resolved[1], resolved[2])
    if record.kind == KIND_D:
        return dihedral_deg(resolved[0], resolved[1], resolved[2], resolved[3])
    return None


def _evaluate_all(
    records: Sequence[CoordinateRecord], xyz: Mapping[int, tuple[float, float, float]]
) -> np.ndarray | None:
    values: list[float] = []
    for record in records:
        value = evaluate_coordinate(record, xyz)
        if value is None or not math.isfinite(value):
            return None
        values.append(value)
    return np.asarray(values, dtype=float)


def unit_scaled_jacobian(
    records: Sequence[CoordinateRecord],
    xyz: Mapping[int, tuple[float, float, float]],
    policy: FeasibilityPolicy,
) -> np.ndarray | None:
    """Return the unit-scaled Jacobian J = ∂q/∂x (central finite differences).

    Only atoms referenced by *records* contribute columns. Each q-row is
    divided by ``max_step_for(kind)`` so B (Å) and A/D (°) rows share one
    scale; rank is invariant under this row scaling, while the condition
    number becomes unit-fair.
    """
    atoms = sorted({map_id for record in records for map_id in record.atom_maps})
    if not records or not atoms:
        return None
    if any(map_id not in xyz for map_id in atoms):
        return None
    j_matrix = np.zeros((len(records), 3 * len(atoms)), dtype=float)
    step = JACOBIAN_FD_STEP_ANG
    for col, map_id in enumerate(atoms):
        base = xyz[map_id]
        for dim in range(3):
            plus = dict(xyz)
            minus = dict(xyz)
            p_list = [base[0], base[1], base[2]]
            m_list = [base[0], base[1], base[2]]
            p_list[dim] += step
            m_list[dim] -= step
            plus[map_id] = (p_list[0], p_list[1], p_list[2])
            minus[map_id] = (m_list[0], m_list[1], m_list[2])
            q_plus = _evaluate_all(records, plus)
            q_minus = _evaluate_all(records, minus)
            if q_plus is None or q_minus is None:
                return None
            j_matrix[:, 3 * col + dim] = (q_plus - q_minus) / (2.0 * step)
    for row, record in enumerate(records):
        j_matrix[row, :] = j_matrix[row, :] / policy.max_step_for(record.kind)
    return j_matrix


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def assess_geometry_feasibility(
    candidate: DriverSetCandidate,
    pool: CoordinatePool,
    bundle: EndpointGraphBundle,
    materials: EndpointMaterials | Mapping[str, Any] | None,
    config: Mapping[str, Any],
    *,
    start_endpoint: str = "R",
) -> FeasibilityReport:
    """Assess geometric feasibility of one driver-set candidate (design §6.3/§8.3).

    Parameters
    ----------
    candidate:
        Driver-set candidate from the todo-12 coordinate pool.
    pool:
        The coordinate pool the candidate's driver ids resolve against.
    bundle:
        Endpoint graph bundle (elements, bonds, components).
    materials:
        Map-keyed endpoint coordinates (R and P) in a common frame.
        ``None`` skips geometry-dependent checks with a typed
        ``MATERIALS_REQUIRED`` failure — geometric feasibility cannot be
        claimed without geometry.
    config:
        Loaded configuration mapping (``scan_strategy`` section consumed).
    start_endpoint:
        Which side the scan starts from (``"R"``/``"reactant"`` or
        ``"P"``/``"product"``). Used for the start-geometry match and report;
        bond-state and degeneracy checks are direction-independent graph
        semantics and run on both sides.
    """
    start = _normalize_start_endpoint(start_endpoint)
    driver_records = _resolve_drivers(candidate, pool)
    materials_norm = _safe_normalize_materials(materials)
    try:
        policy = policy_from_config(config)
    except ValueError as error:
        return _policy_failure_report(candidate, start, str(error))

    elements = elements_by_map(bundle)
    r_bonded = bonded_pairs(bundle.r_graph)
    p_bonded = bonded_pairs(bundle.p_graph)
    all_bonded = r_bonded | p_bonded
    broken_pairs = r_bonded - p_bonded
    formed_pairs = p_bonded - r_bonded
    r_components = component_of_map(bundle.r_graph)
    p_components = component_of_map(bundle.p_graph)
    r_xyz = None if materials_norm is None else dict(materials_norm.r_coordinates)
    p_xyz = None if materials_norm is None else dict(materials_norm.p_coordinates)

    checks: list[CheckResult] = []
    failures: list[FeasibilityFailure] = []

    def record(check_id: str, status: str, detail: str) -> None:
        checks.append(CheckResult(check_id=check_id, status=status, detail=detail))

    def fail(
        check_id: str, code: str, detail: str, suggestion: str | None = None
    ) -> None:
        failures.append(
            FeasibilityFailure(
                check_id=check_id, code=code, detail=detail, suggestion=suggestion
            )
        )

    # --- 1. Coordinate definition (no geometry required) --------------------
    schema_issues = _check_coordinate_definition(driver_records, elements, bundle)
    if schema_issues:
        record(CHECK_COORDINATE_DEFINITION, STATUS_FAIL, "; ".join(schema_issues))
        for issue in schema_issues:
            fail(CHECK_COORDINATE_DEFINITION, _schema_issue_code(issue), issue)
    else:
        record(
            CHECK_COORDINATE_DEFINITION,
            STATUS_PASS,
            f"{len(driver_records)} driver coordinate(s) schema-valid",
        )

    geometry_ready = materials_norm is not None and r_xyz is not None and p_xyz is not None
    if not geometry_ready:
        reason = "endpoint materials missing; geometry checks skipped"
        for check_id in CHECK_ORDER[1:]:
            record(check_id, STATUS_SKIPPED, reason)
        fail(CHECK_COORDINATE_DEFINITION, CODE_MATERIALS_REQUIRED, reason)
        return _build_report(
            candidate=candidate,
            start=start,
            checks=checks,
            failures=failures,
            n_points=None,
            n_points_required=None,
            policy=policy,
            total_variation=(),
            jacobian_ranks=(),
            jacobian_condition_max=None,
        )

    r_xyz_resolved = r_xyz if r_xyz is not None else {}
    p_xyz_resolved = p_xyz if p_xyz is not None else {}

    # --- 2. Endpoint degeneracy --------------------------------------------
    degeneracy_issues = _check_endpoint_degeneracy(
        driver_records, r_xyz_resolved, p_xyz_resolved
    )
    if degeneracy_issues:
        record(CHECK_ENDPOINT_DEGENERACY, STATUS_FAIL, "; ".join(degeneracy_issues))
        for issue in degeneracy_issues:
            fail(CHECK_ENDPOINT_DEGENERACY, _degeneracy_code(issue), issue)
    else:
        record(
            CHECK_ENDPOINT_DEGENERACY,
            STATUS_PASS,
            "B>0 and A/D non-degenerate at both endpoint geometries",
        )

    # --- 3. Dihedral periodic unwrap ---------------------------------------
    unwrap_detail = _dihedral_unwrap_detail(driver_records)
    record(CHECK_DIHEDRAL_UNWRAP, STATUS_PASS, unwrap_detail)

    # --- 4. Endpoint bond state (Cordero thresholds, NOT the 0.45–8 range) --
    bond_issues = _check_endpoint_bond_state(
        elements=elements,
        r_xyz=r_xyz_resolved,
        p_xyz=p_xyz_resolved,
        r_bonded=r_bonded,
        p_bonded=p_bonded,
        broken_pairs=broken_pairs,
        formed_pairs=formed_pairs,
        policy=policy,
    )
    if bond_issues:
        record(CHECK_ENDPOINT_BOND_STATE, STATUS_FAIL, "; ".join(bond_issues))
        for issue in bond_issues:
            fail(CHECK_ENDPOINT_BOND_STATE, _bond_state_code(issue), issue)
    else:
        record(
            CHECK_ENDPOINT_BOND_STATE,
            STATUS_PASS,
            "graph-bonded pairs within Cordero radius sum + tolerance on both sides",
        )

    # --- 5. Cross-component overlap ----------------------------------------
    overlap_issues = _check_cross_component_overlap(
        r_xyz=r_xyz_resolved,
        p_xyz=p_xyz_resolved,
        r_components=r_components,
        p_components=p_components,
        all_bonded=all_bonded,
        policy=policy,
    )
    if overlap_issues:
        record(CHECK_CROSS_COMPONENT_OVERLAP, STATUS_FAIL, "; ".join(overlap_issues))
        for issue in overlap_issues:
            fail(CHECK_CROSS_COMPONENT_OVERLAP, CODE_CROSS_COMPONENT_OVERLAP, issue)
    else:
        record(
            CHECK_CROSS_COMPONENT_OVERLAP,
            STATUS_PASS,
            "no cross-component nonbonded pair below the clash floor",
        )

    # --- 6. Triangle inequality / distance geometry ------------------------
    triangle_issues = _check_triangle_inequality(
        driver_records, r_xyz_resolved, p_xyz_resolved
    )
    if triangle_issues:
        record(CHECK_TRIANGLE_INEQUALITY, STATUS_FAIL, "; ".join(triangle_issues))
        for issue in triangle_issues:
            fail(CHECK_TRIANGLE_INEQUALITY, CODE_TRIANGLE_INEQUALITY_VIOLATION, issue)
    else:
        record(
            CHECK_TRIANGLE_INEQUALITY,
            STATUS_PASS,
            "multi-B driver triples satisfy triangle inequalities at both endpoints",
        )

    # --- 7/8. Unit-scaled Jacobian rank + condition -------------------------
    jacobian_ranks: list[tuple[str, int, int]] = []
    jacobian_condition_max: float | None = None
    n_coords = len(driver_records)
    rank_failures: list[str] = []
    condition_issues: list[str] = []
    max_kappa = 0.0
    for endpoint, xyz in (("R", r_xyz_resolved), ("P", p_xyz_resolved)):
        j_matrix = unit_scaled_jacobian(driver_records, xyz, policy)
        if j_matrix is None:
            rank_failures.append(
                f"{endpoint}: jacobian not evaluable (missing map or non-finite q)"
            )
            jacobian_ranks.append((endpoint, -1, n_coords))
            continue
        rank, kappa = jacobian_diagnostics(j_matrix)
        jacobian_ranks.append((endpoint, rank, n_coords))
        max_kappa = max(max_kappa, kappa) if math.isfinite(kappa) else math.inf
        if rank != n_coords:
            rank_failures.append(f"{endpoint}: rank {rank} != n_coords {n_coords}")
        if not math.isfinite(kappa) or kappa > policy.jacobian_condition_number_max:
            condition_issues.append(
                f"{endpoint}: kappa {kappa:.6g} > max "
                f"{policy.jacobian_condition_number_max:.6g}"
            )
    if rank_failures:
        record(CHECK_JACOBIAN_RANK, STATUS_FAIL, "; ".join(rank_failures))
        for issue in rank_failures:
            fail(CHECK_JACOBIAN_RANK, CODE_JACOBIAN_RANK_DEFICIENT, issue)
    else:
        record(
            CHECK_JACOBIAN_RANK,
            STATUS_PASS,
            f"rank(J) == n_coords ({n_coords}) at R and P",
        )
    if condition_issues:
        record(CHECK_JACOBIAN_CONDITION, STATUS_FAIL, "; ".join(condition_issues))
        for issue in condition_issues:
            fail(CHECK_JACOBIAN_CONDITION, CODE_JACOBIAN_ILL_CONDITIONED, issue)
    else:
        record(
            CHECK_JACOBIAN_CONDITION,
            STATUS_PASS,
            f"max kappa {max_kappa:.6g} <= {policy.jacobian_condition_number_max:.6g}",
        )
    if math.isfinite(max_kappa):
        jacobian_condition_max = max_kappa

    # --- 9. Path precheck (Cartesian linear interpolation) ------------------
    path_issues = _check_path_precheck(
        driver_records=driver_records,
        r_xyz=r_xyz_resolved,
        p_xyz=p_xyz_resolved,
        all_bonded=all_bonded,
        policy=policy,
    )
    if path_issues:
        record(CHECK_PATH_PRECHECK, STATUS_FAIL, "; ".join(path_issues))
        for issue in path_issues:
            code = (
                CODE_PATH_COLLISION
                if issue.startswith("collision")
                else CODE_INTERPOLATION_INFEASIBLE
            )
            fail(CHECK_PATH_PRECHECK, code, issue)
    else:
        record(
            CHECK_PATH_PRECHECK,
            STATUS_PASS,
            "interpolated driver values physical and clash-free at interior λ",
        )

    # --- 10. Start-geometry match (no teleporting scan) --------------------
    start_xyz = r_xyz_resolved if start == "R" else p_xyz_resolved
    teleport_issues = _check_start_geometry_match(driver_records, start_xyz, start, policy)
    if teleport_issues:
        record(CHECK_START_GEOMETRY_MATCH, STATUS_FAIL, "; ".join(teleport_issues))
        for issue in teleport_issues:
            fail(CHECK_START_GEOMETRY_MATCH, CODE_START_TELEPORT, issue)
    else:
        record(
            CHECK_START_GEOMETRY_MATCH,
            STATUS_PASS,
            f"driver start values consistent with {start} geometry",
        )

    # --- 11. Point count rule ----------------------------------------------
    total_variation, point_issues, n_required, n_points, budget_exceeded = (
        _check_point_count(
            driver_records=driver_records,
            r_xyz=r_xyz_resolved,
            p_xyz=p_xyz_resolved,
            policy=policy,
        )
    )
    if point_issues:
        record(CHECK_POINT_COUNT, STATUS_FAIL, "; ".join(point_issues))
        for issue in point_issues:
            fail(
                CHECK_POINT_COUNT,
                CODE_POINT_BUDGET_EXCEEDED,
                issue,
                suggestion=SUGGESTION_POINT_BUDGET,
            )
    else:
        record(
            CHECK_POINT_COUNT,
            STATUS_PASS,
            f"N={n_points} = max(baseline {policy.point_baseline}, "
            f"required {n_required})",
        )

    return _build_report(
        candidate=candidate,
        start=start,
        checks=checks,
        failures=failures,
        n_points=n_points,
        n_points_required=n_required,
        policy=policy,
        total_variation=total_variation,
        jacobian_ranks=tuple(jacobian_ranks),
        jacobian_condition_max=jacobian_condition_max,
        point_budget_exceeded=budget_exceeded,
    )


# ---------------------------------------------------------------------------
# Report builder + policy failure path.
# ---------------------------------------------------------------------------
def _build_report(
    *,
    candidate: DriverSetCandidate,
    start: str,
    checks: Sequence[CheckResult],
    failures: Sequence[FeasibilityFailure],
    n_points: int | None,
    n_points_required: int | None,
    policy: FeasibilityPolicy,
    total_variation: Sequence[tuple[str, float]],
    jacobian_ranks: Sequence[tuple[str, int, int]],
    jacobian_condition_max: float | None,
    point_budget_exceeded: bool = False,
) -> FeasibilityReport:
    return FeasibilityReport(
        schema_version=SCHEMA_GEOMETRY_FEASIBILITY,
        ok=not failures,
        candidate_id=candidate.candidate_id,
        start_endpoint=start,
        checks=tuple(checks),
        failures=tuple(failures),
        n_points=n_points,
        n_points_required=n_points_required,
        n_points_baseline=policy.point_baseline,
        point_budget_exceeded=point_budget_exceeded,
        total_variation_by_coord=tuple(total_variation),
        jacobian_rank_by_endpoint=tuple(jacobian_ranks),
        jacobian_condition_max=jacobian_condition_max,
    )


def _policy_failure_report(
    candidate: DriverSetCandidate, start: str, message: str
) -> FeasibilityReport:
    detail = message.removeprefix("POLICY_INVALID:").strip()
    checks = tuple(
        CheckResult(check_id=check_id, status=STATUS_SKIPPED, detail=f"policy invalid: {detail}")
        for check_id in CHECK_ORDER
    )
    failures = (
        FeasibilityFailure(
            check_id=CHECK_COORDINATE_DEFINITION,
            code=CODE_POLICY_INVALID,
            detail=message,
        ),
    )
    return FeasibilityReport(
        schema_version=SCHEMA_GEOMETRY_FEASIBILITY,
        ok=False,
        candidate_id=candidate.candidate_id,
        start_endpoint=start,
        checks=checks,
        failures=failures,
        n_points=None,
        n_points_required=None,
        n_points_baseline=DEFAULT_POINT_BASELINE,
        point_budget_exceeded=False,
        total_variation_by_coord=(),
        jacobian_rank_by_endpoint=(),
        jacobian_condition_max=None,
    )


# ---------------------------------------------------------------------------
# Check implementations.
# ---------------------------------------------------------------------------
def _normalize_start_endpoint(start_endpoint: str) -> str:
    normalized = _START_ALIASES.get(start_endpoint)
    if normalized is None:
        raise ValueError(
            f"{CODE_START_ENDPOINT_INVALID}: {start_endpoint!r}; "
            "expected R/reactant or P/product"
        )
    return normalized


def _safe_normalize_materials(
    materials: EndpointMaterials | Mapping[str, Any] | None,
) -> EndpointMaterials | None:
    try:
        return normalize_endpoint_materials(materials)
    except ValueError:
        return None


def _resolve_drivers(
    candidate: DriverSetCandidate, pool: CoordinatePool
) -> list[CoordinateRecord]:
    records: list[CoordinateRecord] = []
    for coordinate_id in candidate.driver_coordinate_ids:
        try:
            record = pool.coordinate(coordinate_id)
        except KeyError:
            continue
        records.append(record)
    return records


def _check_coordinate_definition(
    records: Sequence[CoordinateRecord],
    elements: Mapping[int, str],
    bundle: EndpointGraphBundle,
) -> list[str]:
    """Validate kind/units/arity/duplicates/map existence of driver coordinates."""
    issues: list[str] = []
    expected_arity = {KIND_B: 2, KIND_A: 3, KIND_D: 4}
    expected_units = {KIND_B: UNIT_ANGSTROM, KIND_A: UNIT_DEGREE, KIND_D: UNIT_DEGREE}
    bundle_maps = {int(node.map_id) for node in bundle.r_graph.nodes}
    bundle_maps |= {int(node.map_id) for node in bundle.p_graph.nodes}
    for record in records:
        label = record.coordinate_id
        if record.kind not in expected_arity:
            issues.append(f"{label}: unknown kind {record.kind!r}")
            continue
        if record.units != expected_units[record.kind]:
            issues.append(
                f"{label}: units {record.units!r} != {expected_units[record.kind]!r} "
                f"for kind {record.kind}"
            )
        arity = expected_arity[record.kind]
        if len(record.atom_maps) != arity:
            issues.append(
                f"{label}: kind {record.kind} requires {arity} atom maps, "
                f"got {len(record.atom_maps)}"
            )
        if len(set(record.atom_maps)) != len(record.atom_maps):
            issues.append(f"{label}: duplicate atoms in coordinate {record.atom_maps}")
        for map_id in record.atom_maps:
            if map_id not in bundle_maps:
                issues.append(f"{label}: unknown map id {map_id} (not in endpoint graphs)")
            elif map_id not in elements:
                issues.append(f"{label}: map id {map_id} has no element")
    return issues


def _check_endpoint_degeneracy(
    records: Sequence[CoordinateRecord],
    r_xyz: Mapping[int, tuple[float, float, float]],
    p_xyz: Mapping[int, tuple[float, float, float]],
) -> list[str]:
    """B distances positive; A away from 0/180°; D well-defined at both ends."""
    issues: list[str] = []
    for record in records:
        for side, xyz in (("R", r_xyz), ("P", p_xyz)):
            points = [xyz.get(map_id) for map_id in record.atom_maps]
            if any(point is None for point in points):
                issues.append(
                    f"{record.coordinate_id}@{side}: missing coordinates for "
                    f"{record.atom_maps}"
                )
                continue
            resolved = [point for point in points if point is not None]
            if record.kind == KIND_B:
                distance = bond_length(resolved[0], resolved[1])
                if distance <= B_POSITIVE_EPS:
                    issues.append(
                        f"{record.coordinate_id}@{side}: B distance {distance:.6g} "
                        "not positive"
                    )
            elif record.kind == KIND_A:
                angle = bond_angle_deg(resolved[0], resolved[1], resolved[2])
                if (
                    angle <= ANGLE_DEGENERACY_TOL_DEG
                    or angle >= 180.0 - ANGLE_DEGENERACY_TOL_DEG
                ):
                    issues.append(
                        f"{record.coordinate_id}@{side}: A angle {angle:.4f}° "
                        f"within {ANGLE_DEGENERACY_TOL_DEG}° of 0/180 (degenerate)"
                    )
            elif record.kind == KIND_D:
                if dihedral_is_undefined(resolved[0], resolved[1], resolved[2], resolved[3]):
                    issues.append(
                        f"{record.coordinate_id}@{side}: D dihedral undefined "
                        "(collinear triple on the central bond)"
                    )
    return issues


def _dihedral_unwrap_detail(records: Sequence[CoordinateRecord]) -> str:
    """Report shortest-arc Δdihedral per D driver (±180 handled, no raw jump)."""
    deltas: list[str] = []
    for record in records:
        if record.kind != KIND_D:
            continue
        r_value = record.origin.r_value
        p_value = record.origin.p_value
        if r_value is None or p_value is None:
            deltas.append(f"{record.coordinate_id}:Δ=from_geometry")
            continue
        delta = shortest_arc_delta(float(r_value), float(p_value))
        naive = float(p_value) - float(r_value)
        deltas.append(f"{record.coordinate_id}:Δ={delta:.4f}°(naive {naive:.4f}°)")
    if not deltas:
        return "no D drivers in candidate"
    return "shortest-arc Δdihedral: " + ", ".join(deltas)


def _check_endpoint_bond_state(
    *,
    elements: Mapping[int, str],
    r_xyz: Mapping[int, tuple[float, float, float]],
    p_xyz: Mapping[int, tuple[float, float, float]],
    r_bonded: set[tuple[int, int]],
    p_bonded: set[tuple[int, int]],
    broken_pairs: set[tuple[int, int]],
    formed_pairs: set[tuple[int, int]],
    policy: FeasibilityPolicy,
) -> list[str]:
    """Cordero radius-sum + tolerance bondedness; wrong broken/formed endpoints."""
    issues: list[str] = []
    for pair in sorted(r_bonded | p_bonded):
        first, second = pair
        if (
            first not in r_xyz
            or second not in r_xyz
            or first not in p_xyz
            or second not in p_xyz
        ):
            issues.append(f"bonded pair {first}-{second}: missing endpoint coordinates")
            continue
        if first not in elements or second not in elements:
            issues.append(f"bonded pair {first}-{second}: missing element")
            continue
        radius_sum = covalent_radius_sum(elements[first], elements[second])
        upper = radius_sum + policy.bond_tolerance
        lower = max(BONDED_LOWER_FLOOR, BONDED_LOWER_FRACTION * radius_sum)
        d_r = bond_length(r_xyz[first], r_xyz[second])
        d_p = bond_length(p_xyz[first], p_xyz[second])
        if pair in broken_pairs:
            # R-side bond must still exist at the R endpoint geometry.
            if d_r > upper:
                issues.append(
                    f"broken bond {first}-{second}: R distance {d_r:.4f} Å > "
                    f"{upper:.4f} Å (already separated at R — wrong endpoint)"
                )
            if d_r < lower:
                issues.append(
                    f"broken bond {first}-{second}: R distance {d_r:.4f} Å < "
                    f"{lower:.4f} Å (implausibly short)"
                )
        elif pair in formed_pairs:
            if d_p > upper:
                issues.append(
                    f"formed bond {first}-{second}: P distance {d_p:.4f} Å > "
                    f"{upper:.4f} Å (bond missing at P)"
                )
            if d_p < lower:
                issues.append(
                    f"formed bond {first}-{second}: P distance {d_p:.4f} Å < "
                    f"{lower:.4f} Å (implausibly short)"
                )
        else:
            if d_r > upper:
                issues.append(
                    f"bonded pair {first}-{second}: R distance {d_r:.4f} Å > "
                    f"{upper:.4f} Å"
                )
            if d_r < lower:
                issues.append(
                    f"bonded pair {first}-{second}: R distance {d_r:.4f} Å < "
                    f"{lower:.4f} Å"
                )
            if d_p > upper:
                issues.append(
                    f"bonded pair {first}-{second}: P distance {d_p:.4f} Å > "
                    f"{upper:.4f} Å"
                )
            if d_p < lower:
                issues.append(
                    f"bonded pair {first}-{second}: P distance {d_p:.4f} Å < "
                    f"{lower:.4f} Å"
                )
    return issues


def _check_cross_component_overlap(
    *,
    r_xyz: Mapping[int, tuple[float, float, float]],
    p_xyz: Mapping[int, tuple[float, float, float]],
    r_components: Mapping[int, int],
    p_components: Mapping[int, int],
    all_bonded: set[tuple[int, int]],
    policy: FeasibilityPolicy,
) -> list[str]:
    """Nonbonded pairs in different components below the clash floor."""
    issues: list[str] = []
    for side, xyz, components in (
        ("R", r_xyz, r_components),
        ("P", p_xyz, p_components),
    ):
        maps = sorted(m for m in xyz if m in components)
        for i, first in enumerate(maps):
            for second in maps[i + 1 :]:
                if (first, second) in all_bonded:
                    continue
                if components[first] == components[second]:
                    continue
                distance = bond_length(xyz[first], xyz[second])
                if distance < policy.collision_min_distance:
                    issues.append(
                        f"{side}: cross-component pair {first}-{second} at "
                        f"{distance:.4f} Å < {policy.collision_min_distance} Å"
                    )
    return issues


def _planned_b_value(
    record: CoordinateRecord,
    side: str,
    xyz: Mapping[int, tuple[float, float, float]],
) -> float | None:
    """Return the planned B value for one side (origin record preferred).

    Conformational / hydrogen-event records may carry explicit r_value /
    p_value planned targets; those are the scan's constraint values and are
    what the realizability precheck must judge. Without a recorded value the
    endpoint geometry measurement is the planned start/end by construction.
    """
    recorded = record.origin.r_value if side == "R" else record.origin.p_value
    if recorded is not None:
        value = float(recorded)
        return value if math.isfinite(value) else None
    return evaluate_coordinate(record, xyz)


def _check_triangle_inequality(
    records: Sequence[CoordinateRecord],
    r_xyz: Mapping[int, tuple[float, float, float]],
    p_xyz: Mapping[int, tuple[float, float, float]],
) -> list[str]:
    """Triangle inequality on planned B values for fully covered atom triples.

    Endpoint-measured distances of a realizable geometry can never violate
    Euclidean triangle inequalities; the meaningful failure is a candidate
    whose *planned* driver constraints are jointly unrealizable (H2
    three-distance sets especially). Recorded origin values are preferred;
    geometry measurements fill in when no planned value exists.
    """
    issues: list[str] = []
    b_records: dict[tuple[int, int], CoordinateRecord] = {}
    for record in records:
        if record.kind == KIND_B and len(record.atom_maps) == 2:
            a, b = record.atom_maps
            key = (a, b) if a <= b else (b, a)
            b_records.setdefault(key, record)
    if len(b_records) < 3:
        return issues
    atoms = sorted({map_id for pair in b_records for map_id in pair})
    for i, first in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            second = atoms[j]
            for k in range(j + 1, len(atoms)):
                third = atoms[k]
                needed = {
                    tuple(sorted((first, second))),
                    tuple(sorted((second, third))),
                    tuple(sorted((first, third))),
                }
                if not needed <= set(b_records):
                    continue
                ordered_pairs = (
                    tuple(sorted((first, second))),
                    tuple(sorted((second, third))),
                    tuple(sorted((first, third))),
                )
                for side, xyz in (("R", r_xyz), ("P", p_xyz)):
                    values: list[float | None] = []
                    for pair in ordered_pairs:
                        record = b_records[pair]
                        if record.atom_maps[0] not in xyz or record.atom_maps[1] not in xyz:
                            values.append(None)
                        else:
                            values.append(_planned_b_value(record, side, xyz))
                    if any(value is None for value in values):
                        continue
                    d_first_second = float(values[0])  # type: ignore[arg-type]
                    d_second_third = float(values[1])  # type: ignore[arg-type]
                    d_first_third = float(values[2])  # type: ignore[arg-type]
                    label = f"{first}-{second}-{third}"
                    if d_first_second + d_second_third < d_first_third - TRIANGLE_EPS_ANG:
                        issues.append(
                            f"{side}: triangle {label} violated "
                            f"({d_first_second:.4f}+{d_second_third:.4f} "
                            f"< {d_first_third:.4f})"
                        )
                    if d_first_second + d_first_third < d_second_third - TRIANGLE_EPS_ANG:
                        issues.append(
                            f"{side}: triangle {label} violated "
                            f"({d_first_second:.4f}+{d_first_third:.4f} "
                            f"< {d_second_third:.4f})"
                        )
                    if d_second_third + d_first_third < d_first_second - TRIANGLE_EPS_ANG:
                        issues.append(
                            f"{side}: triangle {label} violated "
                            f"({d_second_third:.4f}+{d_first_third:.4f} "
                            f"< {d_first_second:.4f})"
                        )
    return issues


def _check_path_precheck(
    *,
    driver_records: Sequence[CoordinateRecord],
    r_xyz: Mapping[int, tuple[float, float, float]],
    p_xyz: Mapping[int, tuple[float, float, float]],
    all_bonded: set[tuple[int, int]],
    policy: FeasibilityPolicy,
) -> list[str]:
    """Cartesian linear interpolation precheck: physical ranges + clash-free.

    Cheap geometric precheck only (design §6.3.5) — never a computed PES.
    """
    issues: list[str] = []
    if not driver_records:
        return issues
    maps = sorted({map_id for record in driver_records for map_id in record.atom_maps})
    if any(map_id not in r_xyz or map_id not in p_xyz for map_id in maps):
        issues.append("interpolation: missing endpoint coordinates for driver atoms")
        return issues
    for lam in PATH_PRECHECK_LAMBDAS:
        xyz: dict[int, tuple[float, float, float]] = {}
        for map_id in maps:
            r_point = r_xyz[map_id]
            p_point = p_xyz[map_id]
            xyz[map_id] = (
                (1.0 - lam) * r_point[0] + lam * p_point[0],
                (1.0 - lam) * r_point[1] + lam * p_point[1],
                (1.0 - lam) * r_point[2] + lam * p_point[2],
            )
        for record in driver_records:
            value = evaluate_coordinate(record, xyz)
            if value is None or not math.isfinite(value):
                issues.append(
                    f"λ={lam}: {record.coordinate_id} undefined on interpolated geometry"
                )
                continue
            if record.kind == KIND_B and value <= B_POSITIVE_EPS:
                issues.append(
                    f"λ={lam}: {record.coordinate_id} B={value:.6g} not positive"
                )
            elif record.kind == KIND_A and (
                value <= ANGLE_DEGENERACY_TOL_DEG
                or value >= 180.0 - ANGLE_DEGENERACY_TOL_DEG
            ):
                issues.append(
                    f"λ={lam}: {record.coordinate_id} A={value:.4f}° degenerate"
                )
            elif record.kind == KIND_D and dihedral_is_undefined(
                xyz[record.atom_maps[0]],
                xyz[record.atom_maps[1]],
                xyz[record.atom_maps[2]],
                xyz[record.atom_maps[3]],
            ):
                issues.append(
                    f"λ={lam}: {record.coordinate_id} D undefined (collinear)"
                )
        # Severe nonbonded clash on the interpolated geometry (any pair of
        # driver atoms not bonded in R∪P).
        for i, first in enumerate(maps):
            for second in maps[i + 1 :]:
                if (first, second) in all_bonded:
                    continue
                distance = bond_length(xyz[first], xyz[second])
                if distance < policy.collision_min_distance:
                    issues.append(
                        f"collision: λ={lam} nonbonded pair {first}-{second} at "
                        f"{distance:.4f} Å < {policy.collision_min_distance} Å"
                    )
    return issues


def _check_start_geometry_match(
    records: Sequence[CoordinateRecord],
    start_xyz: Mapping[int, tuple[float, float, float]],
    start: str,
    policy: FeasibilityPolicy,
) -> list[str]:
    """Start-point target coordinates must match the starting geometry.

    Conformational D records carry recorded r_value/p_value; a mismatch
    beyond the per-kind max step is a teleporting-scan violation. Edit-driven
    coordinates have no recorded planned start (their start IS the geometry),
    which is explicitly consistent by construction.
    """
    issues: list[str] = []
    for record in records:
        if record.origin.source_kind != "conformational_difference":
            continue
        recorded = record.origin.r_value if start == "R" else record.origin.p_value
        if recorded is None:
            continue
        measured = evaluate_coordinate(record, start_xyz)
        if measured is None:
            issues.append(
                f"{record.coordinate_id}: start-geometry value undefined on {start}"
            )
            continue
        tolerance = policy.max_step_for(record.kind)
        if abs(measured - float(recorded)) > tolerance:
            issues.append(
                f"{record.coordinate_id}: recorded {start} value {recorded:.4f}° "
                f"vs geometry {measured:.4f}° (Δ>{tolerance:.4f}) — teleporting scan"
            )
    return issues


def _check_point_count(
    *,
    driver_records: Sequence[CoordinateRecord],
    r_xyz: Mapping[int, tuple[float, float, float]],
    p_xyz: Mapping[int, tuple[float, float, float]],
    policy: FeasibilityPolicy,
) -> tuple[
    tuple[tuple[str, float], ...], list[str], int, int, bool
]:
    """N_required = 1 + max_j ceil(TV_j / max_step_j); N = max(baseline, N)."""
    total_variation: list[tuple[str, float]] = []
    n_required = 1
    for record in driver_records:
        q_start = evaluate_coordinate(record, r_xyz)
        q_end = evaluate_coordinate(record, p_xyz)
        if q_start is None or q_end is None:
            total_variation.append((record.coordinate_id, math.inf))
            continue
        if record.kind == KIND_D:
            variation = abs(shortest_arc_delta(q_start, q_end))
        else:
            variation = abs(q_end - q_start)
        total_variation.append((record.coordinate_id, variation))
        max_step = policy.max_step_for(record.kind)
        # ceil with a tiny guard so exact binary multiples are not inflated.
        steps = max(0, math.ceil(variation / max_step - 1e-9))
        n_required = max(n_required, 1 + steps)
    n_points = max(policy.point_baseline, n_required)
    budget_exceeded = n_points > policy.point_max
    issues: list[str] = []
    if budget_exceeded:
        worst = max(
            total_variation,
            key=lambda row: (
                math.ceil(row[1] / _step_of(driver_records, row[0], policy) - 1e-9)
                if math.isfinite(row[1])
                else -1
            ),
            default=None,
        )
        worst_text = (
            f"worst {worst[0]} TV={worst[1]:.4f}" if worst is not None else "no drivers"
        )
        issues.append(
            f"N_required={n_required} > point_limits.max={policy.point_max} "
            f"({worst_text}); typed verdict only — segmentation/method/reject "
            "decision owned by todo 17"
        )
    return tuple(total_variation), issues, n_required, n_points, budget_exceeded


def _step_of(
    records: Sequence[CoordinateRecord], coordinate_id: str, policy: FeasibilityPolicy
) -> float:
    for record in records:
        if record.coordinate_id == coordinate_id:
            return policy.max_step_for(record.kind)
    return policy.max_step_distance


# ---------------------------------------------------------------------------
# Issue → typed failure code mapping.
# ---------------------------------------------------------------------------
def _schema_issue_code(issue: str) -> str:
    if "duplicate atoms" in issue:
        return CODE_DUPLICATE_ATOMS
    if "unknown map id" in issue:
        return CODE_UNKNOWN_MAP_ID
    if "units" in issue:
        return CODE_UNIT_MISMATCH
    return CODE_COORDINATE_SCHEMA_INVALID


def _degeneracy_code(issue: str) -> str:
    if "B distance" in issue:
        return CODE_B_DISTANCE_NONPOSITIVE
    if "A angle" in issue:
        return CODE_ANGLE_DEGENERATE
    if "D dihedral" in issue:
        return CODE_DIHEDRAL_UNDEFINED
    return CODE_COORDINATE_SCHEMA_INVALID


def _bond_state_code(issue: str) -> str:
    if issue.startswith("broken bond"):
        return CODE_BROKEN_BOND_ALREADY_SEPARATED
    if issue.startswith("formed bond"):
        return CODE_FORMED_BOND_MISSING_AT_P
    return CODE_BONDED_DISTANCE_IMPLAUSIBLE


# ---------------------------------------------------------------------------
# Small vector helpers.
# ---------------------------------------------------------------------------
def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


__all__ = [
    "ANGLE_DEGENERACY_TOL_DEG",
    "CHECK_COORDINATE_DEFINITION",
    "CHECK_CROSS_COMPONENT_OVERLAP",
    "CHECK_DIHEDRAL_UNWRAP",
    "CHECK_ENDPOINT_BOND_STATE",
    "CHECK_ENDPOINT_DEGENERACY",
    "CHECK_JACOBIAN_CONDITION",
    "CHECK_JACOBIAN_RANK",
    "CHECK_PATH_PRECHECK",
    "CHECK_POINT_COUNT",
    "CHECK_START_GEOMETRY_MATCH",
    "CHECK_TRIANGLE_INEQUALITY",
    "CODE_ANGLE_DEGENERATE",
    "CODE_B_DISTANCE_NONPOSITIVE",
    "CODE_BROKEN_BOND_ALREADY_SEPARATED",
    "CODE_CROSS_COMPONENT_OVERLAP",
    "CODE_DIHEDRAL_UNDEFINED",
    "CODE_FORMED_BOND_MISSING_AT_P",
    "CODE_INTERPOLATION_INFEASIBLE",
    "CODE_JACOBIAN_ILL_CONDITIONED",
    "CODE_JACOBIAN_RANK_DEFICIENT",
    "CODE_MATERIALS_REQUIRED",
    "CODE_PATH_COLLISION",
    "CODE_POINT_BUDGET_EXCEEDED",
    "CODE_POLICY_INVALID",
    "CODE_START_ENDPOINT_INVALID",
    "CODE_START_TELEPORT",
    "CODE_TRIANGLE_INEQUALITY_VIOLATION",
    "CODE_UNIT_MISMATCH",
    "CODE_UNKNOWN_MAP_ID",
    "CODE_DUPLICATE_ATOMS",
    "CODE_COORDINATE_SCHEMA_INVALID",
    "DEFAULT_MAX_STEP_ANGLE",
    "DEFAULT_MAX_STEP_DIHEDRAL",
    "DEFAULT_MAX_STEP_DISTANCE",
    "DEFAULT_POINT_BASELINE",
    "DEFAULT_POINT_MAX",
    "FeasibilityFailure",
    "FeasibilityPolicy",
    "FeasibilityReport",
    "SCHEMA_GEOMETRY_FEASIBILITY",
    "SUGGESTION_POINT_BUDGET",
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_SKIPPED",
    "CheckResult",
    "assess_geometry_feasibility",
    "bond_angle_deg",
    "bond_length",
    "bonded_pairs",
    "covalent_radius_sum",
    "component_of_map",
    "dihedral_deg",
    "dihedral_is_undefined",
    "elements_by_map",
    "evaluate_coordinate",
    "jacobian_diagnostics",
    "normalize_endpoint_materials",
    "policy_from_config",
    "shortest_arc_delta",
    "unit_scaled_jacobian",
]
