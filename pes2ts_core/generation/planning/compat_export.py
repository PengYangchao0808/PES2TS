"""Legacy single-B compatibility export for frozen GenerationPlanV2 docs.

P1 bridge (design §10.1, §13.2 P1; plan todo 18): project a frozen
``g1_generation_plan_v2`` document whose every candidate is a
``ScanCandidateV2`` SINGLE_1D with exactly one ``B`` distance driver onto the
old v1 ``ScanPlan`` shape that passes ``pes2ts_core.contracts.validate_document``
and ``pes2ts_core.integration.acp.adapter.scan_plan_to_acp_request``.

Guardrails (never silently degraded):

- Multi-coordinate plans (``COUPLED_1D`` / ``SCHEDULED_1D``, >1 driver),
  ``A``/``D`` driver kinds, non-uniform point lists, and ``PathCandidateV1``
  payloads are **explicitly rejected** with :class:`CompatExportError`
  (an ``ACPMappingError`` subclass carrying a machine-readable ``code``).
  The export never projects a multi-coordinate plan onto its first driver.
- ``needs_review`` / proposal-stage documents never produce a ready export:
  only a sealed, ``status="frozen"`` ``g1_generation_plan_v2`` document is
  exportable.  The 24 Demo24 cases remain ``needs_review``; real ``ready``
  requires human review that this plan does not lift.
- ``source_case_sha256`` must bind to the case snapshot digest supplied as
  ``materials``; identity fields (``reaction_id``/``case_id``/``split``)
  must agree between plan and materials.  Driver indices are computed from
  the frozen atom order of that snapshot (or taken from a consistent frozen
  ``driver.index0``); a disagreement is a typed refusal.
- Point counts honor the v1/ACP contract bounds 3..101, and the computed
  point list must be uniform within 1e-6 — the exact predicate the adapter
  applies (``ACP PesScanRequest`` represents uniform points only).

``materials`` is parsed at the boundary: a sealed ``ReactionCase`` document
or a projection mapping with ``atom_map_ids``, identity fields,
``content_sha256``, and ``r_coordinates``/``p_coordinates`` (map -> xyz).
Physical scan points are compiled here as
``q(λ) = q_start + s(λ)·(q_end − q_start)`` from endpoint geometry
(``driver.schedule_values`` when present, else the shared ``lambda_values``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pes2ts_core.contracts import make_document, seal_document
from pes2ts_core.integration.acp.adapter import ACPMappingError
from pes2ts_core.generation.planning.contracts_v2 import (
    CANDIDATE_KIND_PATH,
    CANDIDATE_KIND_SCAN,
    MODE_SINGLE_1D,
    OBJECT_GENERATION_PLAN,
    OBJECT_STRATEGY_PROPOSAL,
    SCHEMA_GENERATION_PLAN,
    validate_v2_document,
)

# allow: SIZE_OK — plan-named todo-18 single module (v2→v1 legacy single-B
# export bridge + materials boundary parse + typed refusal vocabulary);
# precedent: contracts_v2.py / selector.py / registry.py (todo 3/17/11).

__all__ = [
    "CompatExportError",
    "LegacyExportMaterials",
    "export_legacy_scan_plan",
]

# ---------------------------------------------------------------------------
# Typed refusal vocabulary.
# ---------------------------------------------------------------------------
NOT_A_PLAN_DOCUMENT = "NOT_A_PLAN_DOCUMENT"
NEEDS_REVIEW_OR_PROPOSAL_REFUSED = "NEEDS_REVIEW_OR_PROPOSAL_REFUSED"
PLAN_NOT_FROZEN = "PLAN_NOT_FROZEN"
PLAN_DOCUMENT_INVALID = "PLAN_DOCUMENT_INVALID"
PLAN_DIGEST_MISMATCH = "PLAN_DIGEST_MISMATCH"
IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
SOURCE_CASE_BINDING_MISMATCH = "SOURCE_CASE_BINDING_MISMATCH"
MATERIALS_INVALID = "MATERIALS_INVALID"
EMPTY_CANDIDATES = "EMPTY_CANDIDATES"
MULTI_COORDINATE_UNSUPPORTED = "MULTI_COORDINATE_UNSUPPORTED"
DRIVER_KIND_UNSUPPORTED = "DRIVER_KIND_UNSUPPORTED"
UNIT_UNSUPPORTED = "UNIT_UNSUPPORTED"
DRIVER_MAPS_INVALID = "DRIVER_MAPS_INVALID"
DRIVER_MAP_UNKNOWN = "DRIVER_MAP_UNKNOWN"
INDEX_ORDER_MISMATCH = "INDEX_ORDER_MISMATCH"
PATH_CANDIDATE_UNSUPPORTED = "PATH_CANDIDATE_UNSUPPORTED"
UNKNOWN_CANDIDATE_KIND = "UNKNOWN_CANDIDATE_KIND"
INVALID_START_ENDPOINT = "INVALID_START_ENDPOINT"
NO_LAMBDA_GRID = "NO_LAMBDA_GRID"
SCHEDULE_VALUE_COUNT_MISMATCH = "SCHEDULE_VALUE_COUNT_MISMATCH"
GEOMETRY_MISSING = "GEOMETRY_MISSING"
SCAN_RANGE_DEGENERATE = "SCAN_RANGE_DEGENERATE"
POINTS_OUT_OF_BOUNDS = "POINTS_OUT_OF_BOUNDS"
POINTS_INVALID = "POINTS_INVALID"
NONUNIFORM_POINTS = "NONUNIFORM_POINTS"
BACKEND_INVALID = "BACKEND_INVALID"
BACKEND_ENGINE_UNSUPPORTED = "BACKEND_ENGINE_UNSUPPORTED"
BACKEND_METHOD_MISSING = "BACKEND_METHOD_MISSING"

DEFAULT_EXPERIMENT_ID = "scan-strategy-compat-v1"
CANDIDATE_STRATEGY = "graph_scan_single_1d_compat_v1"
#: Same uniformity predicate as ``scan_plan_to_acp_request`` (adapter).
POINT_UNIFORMITY_TOL = 1e-6
POINT_MIN = 3
POINT_MAX = 101


class CompatExportError(ACPMappingError):
    """Typed refusal for the legacy single-B compatibility export.

    Subclasses ``ACPMappingError`` so callers can catch the adapter's
    baseline-mapping error family, while ``code`` names the exact refusal.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        message = code if not detail else f"{code}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class LegacyExportMaterials:
    """Case-snapshot projection that the v1 export binds to."""

    atom_map_ids: tuple[int, ...]
    dataset_version: str
    reaction_id: str
    case_id: str
    split: str
    content_sha256: str
    r_coordinates: dict[int, tuple[float, float, float]]
    p_coordinates: dict[int, tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class _CompatConfig:
    experiment_id: str
    failure_policy: str = "retry_previous"
    optimizer_retries: int = 1
    reuse_previous_geometry: bool = True


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _parse_xyz_block(raw: Any, path: str) -> dict[int, tuple[float, float, float]]:
    if not isinstance(raw, Mapping):
        raise CompatExportError(MATERIALS_INVALID, f"{path}: must be an object keyed by atom map id")
    parsed: dict[int, tuple[float, float, float]] = {}
    for key, value in raw.items():
        try:
            map_id = int(key)
        except (TypeError, ValueError) as exc:
            raise CompatExportError(MATERIALS_INVALID, f"{path}: map id {key!r} is not an integer") from exc
        if map_id <= 0:
            raise CompatExportError(MATERIALS_INVALID, f"{path}: map id {map_id} must be positive")
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise CompatExportError(MATERIALS_INVALID, f"{path}[{map_id}]: expected one 3D coordinate")
        try:
            xyz = (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError) as exc:
            raise CompatExportError(MATERIALS_INVALID, f"{path}[{map_id}]: coordinates must be numbers") from exc
        if any(not math.isfinite(component) for component in xyz):
            raise CompatExportError(MATERIALS_INVALID, f"{path}[{map_id}]: coordinates must be finite")
        parsed[map_id] = xyz
    return parsed


def _parse_materials(raw: Any) -> LegacyExportMaterials:
    """Parse a ReactionCase document or a case-snapshot projection."""
    if not isinstance(raw, Mapping):
        raise CompatExportError(MATERIALS_INVALID, "materials must be a ReactionCase document or projection mapping")
    if raw.get("schema_name") == "ReactionCase":
        atoms = raw.get("atoms")
        if not isinstance(atoms, list) or not atoms:
            raise CompatExportError(MATERIALS_INVALID, "ReactionCase materials require a non-empty atoms list")
        try:
            order = tuple(int(atom["atom_map_id"]) for atom in atoms)
        except (KeyError, TypeError, ValueError) as exc:
            raise CompatExportError(MATERIALS_INVALID, "ReactionCase atoms must carry integer atom_map_id values") from exc
        if len(set(order)) != len(order) or any(map_id <= 0 for map_id in order):
            raise CompatExportError(MATERIALS_INVALID, "ReactionCase atom_map_id values must be unique positive integers")

        def _side_geometry(side: str) -> dict[int, tuple[float, float, float]]:
            block = raw.get(side)
            if not isinstance(block, Mapping):
                raise CompatExportError(MATERIALS_INVALID, f"ReactionCase materials require a {side} block")
            rows = block.get("geometry")
            if not isinstance(rows, list) or len(rows) != len(order):
                raise CompatExportError(MATERIALS_INVALID, f"ReactionCase {side}.geometry must cover every atom")
            coords: dict[int, tuple[float, float, float]] = {}
            for index, row in enumerate(rows):
                if not isinstance(row, (list, tuple)) or len(row) != 3:
                    raise CompatExportError(MATERIALS_INVALID, f"ReactionCase {side}.geometry[{index}]: expected one 3D coordinate")
                try:
                    xyz = (float(row[0]), float(row[1]), float(row[2]))
                except (TypeError, ValueError) as exc:
                    raise CompatExportError(MATERIALS_INVALID, f"ReactionCase {side}.geometry[{index}]: coordinates must be numbers") from exc
                if any(not math.isfinite(component) for component in xyz):
                    raise CompatExportError(MATERIALS_INVALID, f"ReactionCase {side}.geometry[{index}]: coordinates must be finite")
                coords[order[index]] = xyz
            return coords

        identity: dict[str, str] = {}
        for field in ("dataset_version", "reaction_id", "case_id", "split"):
            value = raw.get(field)
            if not isinstance(value, str) or not value:
                raise CompatExportError(MATERIALS_INVALID, f"ReactionCase materials require {field}")
            identity[field] = value
        if raw.get("content_sha256") is not None and not _is_sha256(raw.get("content_sha256")):
            raise CompatExportError(MATERIALS_INVALID, "ReactionCase content_sha256 must be a lowercase SHA256")
        return LegacyExportMaterials(
            atom_map_ids=order,
            dataset_version=identity["dataset_version"],
            reaction_id=identity["reaction_id"],
            case_id=identity["case_id"],
            split=identity["split"],
            content_sha256=str(raw.get("content_sha256") or ""),
            r_coordinates=_side_geometry("reactant"),
            p_coordinates=_side_geometry("product"),
        )

    raw_order = raw.get("atom_map_ids")
    if not isinstance(raw_order, list) or not raw_order:
        raise CompatExportError(MATERIALS_INVALID, "materials projection requires a non-empty atom_map_ids list")
    try:
        order = tuple(int(map_id) for map_id in raw_order)
    except (TypeError, ValueError) as exc:
        raise CompatExportError(MATERIALS_INVALID, "materials atom_map_ids must be integers") from exc
    if len(set(order)) != len(order) or any(map_id <= 0 for map_id in order):
        raise CompatExportError(MATERIALS_INVALID, "materials atom_map_ids must be unique positive integers")
    identity: dict[str, str] = {}
    for field in ("dataset_version", "reaction_id", "case_id", "split"):
        value = raw.get(field)
        if not isinstance(value, str) or not value:
            raise CompatExportError(MATERIALS_INVALID, f"materials projection requires {field}")
        identity[field] = value
    content_sha = raw.get("content_sha256")
    if not _is_sha256(content_sha):
        raise CompatExportError(MATERIALS_INVALID, "materials projection requires a lowercase SHA256 content_sha256")
    r_coordinates = _parse_xyz_block(raw.get("r_coordinates"), "r_coordinates")
    p_coordinates = _parse_xyz_block(raw.get("p_coordinates"), "p_coordinates")
    for side, coords in (("r_coordinates", r_coordinates), ("p_coordinates", p_coordinates)):
        missing = [map_id for map_id in order if map_id not in coords]
        if missing:
            raise CompatExportError(MATERIALS_INVALID, f"{side}: missing geometry for maps {missing}")
    return LegacyExportMaterials(
        atom_map_ids=order,
        dataset_version=identity["dataset_version"],
        reaction_id=identity["reaction_id"],
        case_id=identity["case_id"],
        split=identity["split"],
        content_sha256=str(content_sha),
        r_coordinates=r_coordinates,
        p_coordinates=p_coordinates,
    )


def _compat_config(config: Mapping[str, Any] | None) -> _CompatConfig:
    """Read compat-export policy from config; defaults are deterministic."""
    section: Mapping[str, Any] = {}
    top_experiment: Any = None
    if isinstance(config, Mapping):
        top_experiment = config.get("experiment_id")
        raw = config.get("scan_strategy")
        if isinstance(raw, Mapping):
            inner = raw.get("compat_export")
            if isinstance(inner, Mapping):
                section = inner
    experiment_id = top_experiment
    if not isinstance(experiment_id, str) or not experiment_id:
        experiment_id = section.get("experiment_id", DEFAULT_EXPERIMENT_ID)
    if not isinstance(experiment_id, str) or not experiment_id:
        experiment_id = DEFAULT_EXPERIMENT_ID
    failure_policy = "retry_previous"
    optimizer_retries = 1
    reuse_previous_geometry = True
    retry = section.get("retry_policy")
    if isinstance(retry, Mapping):
        if isinstance(retry.get("failure_policy"), str) and retry["failure_policy"]:
            failure_policy = retry["failure_policy"]
        raw_retries = retry.get("optimizer_retries")
        if isinstance(raw_retries, int) and not isinstance(raw_retries, bool) and raw_retries >= 0:
            optimizer_retries = raw_retries
        if isinstance(retry.get("reuse_previous_geometry"), bool):
            reuse_previous_geometry = retry["reuse_previous_geometry"]
    return _CompatConfig(
        experiment_id=experiment_id,
        failure_policy=failure_policy,
        optimizer_retries=optimizer_retries,
        reuse_previous_geometry=reuse_previous_geometry,
    )


def _require_plan_doc(doc: Any) -> dict[str, Any]:
    """Boundary-parse the v2 plan document; refuse proposals and non-frozen plans."""
    if not isinstance(doc, Mapping):
        raise CompatExportError(NOT_A_PLAN_DOCUMENT, "export input must be a document object")
    schema = doc.get("schema_name")
    if schema == OBJECT_STRATEGY_PROPOSAL:
        raise CompatExportError(
            NEEDS_REVIEW_OR_PROPOSAL_REFUSED,
            "strategy proposals are never execution-eligible; a needs_review "
            "proposal never produces a ready legacy export — freeze a "
            "g1_generation_plan_v2 after human review and backend smoke first",
        )
    if schema != OBJECT_GENERATION_PLAN or doc.get("schema_version") != SCHEMA_GENERATION_PLAN:
        raise CompatExportError(
            NOT_A_PLAN_DOCUMENT,
            f"expected schema_name={OBJECT_GENERATION_PLAN!r} schema_version={SCHEMA_GENERATION_PLAN!r}, got {schema!r}/{doc.get('schema_version')!r}",
        )
    issues = validate_v2_document(dict(doc))
    if issues:
        raise CompatExportError(PLAN_DOCUMENT_INVALID, "; ".join(issues[:5]))
    if seal_document(dict(doc))["content_sha256"] != doc.get("content_sha256"):
        raise CompatExportError(PLAN_DIGEST_MISMATCH, "document content_sha256 does not match its sealed content")
    if doc.get("status") != "frozen":
        raise CompatExportError(PLAN_NOT_FROZEN, f"status={doc.get('status')!r}; only frozen plans export")
    return dict(doc)


def _bind_plan_to_materials(plan_v2: Mapping[str, Any], mats: LegacyExportMaterials) -> None:
    for field, value in (
        ("reaction_id", mats.reaction_id),
        ("case_id", mats.case_id),
        ("split", mats.split),
    ):
        if plan_v2.get(field) != value:
            raise CompatExportError(
                IDENTITY_MISMATCH,
                f"{field}: plan={plan_v2.get(field)!r} materials={value!r}",
            )
    if plan_v2.get("source_case_sha256") != mats.content_sha256:
        raise CompatExportError(
            SOURCE_CASE_BINDING_MISMATCH,
            "plan.source_case_sha256 does not bind to the case snapshot digest in materials",
        )


def _pair_distance(
    coords: Mapping[int, tuple[float, float, float]],
    maps: list[int],
) -> float:
    first = coords[maps[0]]
    second = coords[maps[1]]
    return math.sqrt(sum((first[axis] - second[axis]) ** 2 for axis in range(3)))


def _endpoint_distances(
    mats: LegacyExportMaterials,
    start_endpoint: str,
    maps: list[int],
) -> tuple[float, float]:
    if start_endpoint == "R":
        sides = (mats.r_coordinates, mats.p_coordinates)
    elif start_endpoint == "P":
        sides = (mats.p_coordinates, mats.r_coordinates)
    else:
        raise CompatExportError(INVALID_START_ENDPOINT, f"start_endpoint={start_endpoint!r}; expected 'R' or 'P'")
    for coords in sides:
        for map_id in maps:
            if map_id not in coords:
                raise CompatExportError(GEOMETRY_MISSING, f"materials geometry lacks atom map {map_id}")
    return _pair_distance(sides[0], maps), _pair_distance(sides[1], maps)


def _check_points(points: list[float]) -> None:
    count = len(points)
    if count < POINT_MIN or count > POINT_MAX:
        raise CompatExportError(
            POINTS_OUT_OF_BOUNDS,
            f"n_points={count}; the legacy contract requires {POINT_MIN}..{POINT_MAX} scan points",
        )
    if any(not math.isfinite(point) for point in points):
        raise CompatExportError(POINTS_INVALID, "scan points must be finite numbers")
    expected = [
        points[0] + (points[-1] - points[0]) * index / (count - 1)
        for index in range(count)
    ]
    if any(abs(actual - want) > POINT_UNIFORMITY_TOL for actual, want in zip(points, expected, strict=True)):
        raise CompatExportError(
            NONUNIFORM_POINTS,
            "ACP PesScanRequest represents uniform points only; a non-uniform "
            "plan needs an adapter extension and is never linearized here",
        )


def _export_candidate(
    plan_v2: Mapping[str, Any],
    candidate: Any,
    mats: LegacyExportMaterials,
    opts: _CompatConfig,
    index: int,
) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise CompatExportError(UNKNOWN_CANDIDATE_KIND, f"candidates[{index}] must be an object")
    kind = candidate.get("candidate_kind")
    if kind == CANDIDATE_KIND_PATH:
        raise CompatExportError(
            PATH_CANDIDATE_UNSUPPORTED,
            "the legacy single-B ScanPlan cannot represent PathCandidateV1/NEB payloads",
        )
    if kind != CANDIDATE_KIND_SCAN:
        raise CompatExportError(UNKNOWN_CANDIDATE_KIND, f"candidates[{index}].candidate_kind={kind!r}")
    mode = candidate.get("mode")
    if mode != MODE_SINGLE_1D:
        raise CompatExportError(
            MULTI_COORDINATE_UNSUPPORTED,
            f"mode={mode!r}; the legacy export accepts only SINGLE_1D and never "
            "projects multi-coordinate (COUPLED/SCHEDULED) plans onto a first driver",
        )
    drivers = candidate.get("drivers")
    if not isinstance(drivers, list) or len(drivers) != 1:
        count = len(drivers) if isinstance(drivers, list) else drivers
        raise CompatExportError(
            MULTI_COORDINATE_UNSUPPORTED,
            f"n_drivers={count!r}; exactly one distance driver is required",
        )
    driver = drivers[0]
    if not isinstance(driver, Mapping):
        raise CompatExportError(DRIVER_MAPS_INVALID, "the single driver must be an object")
    driver_kind = driver.get("kind")
    if driver_kind != "B":
        raise CompatExportError(
            DRIVER_KIND_UNSUPPORTED,
            f"driver kind={driver_kind!r}; the legacy contract exports distance (B) drivers only",
        )
    if driver.get("unit") != "angstrom":
        raise CompatExportError(UNIT_UNSUPPORTED, f"driver unit={driver.get('unit')!r}; expected 'angstrom'")
    raw_maps = driver.get("maps")
    if not isinstance(raw_maps, list) or len(raw_maps) != 2:
        raise CompatExportError(DRIVER_MAPS_INVALID, f"driver maps={raw_maps!r}; a B driver needs exactly two atom maps")
    try:
        maps = [int(map_id) for map_id in raw_maps]
    except (TypeError, ValueError) as exc:
        raise CompatExportError(DRIVER_MAPS_INVALID, "driver maps must be integers") from exc
    if maps[0] == maps[1] or any(map_id <= 0 for map_id in maps):
        raise CompatExportError(DRIVER_MAPS_INVALID, f"driver maps={maps!r}; maps must be distinct positive integers")
    order = list(mats.atom_map_ids)
    map_to_index = {map_id: position for position, map_id in enumerate(order)}
    unknown = [map_id for map_id in maps if map_id not in map_to_index]
    if unknown:
        raise CompatExportError(DRIVER_MAP_UNKNOWN, f"driver maps {unknown} are not in the frozen atom order")
    computed = [map_to_index[map_id] for map_id in maps]
    index0 = driver.get("index0")
    if index0 is not None:
        if not isinstance(index0, (list, tuple)):
            raise CompatExportError(INDEX_ORDER_MISMATCH, "driver.index0 must be null or an integer array")
        try:
            frozen = [int(value) for value in index0]
        except (TypeError, ValueError) as exc:
            raise CompatExportError(INDEX_ORDER_MISMATCH, "driver.index0 must be integers") from exc
        if frozen != computed:
            raise CompatExportError(
                INDEX_ORDER_MISMATCH,
                f"driver.index0={frozen!r} disagrees with the frozen atom order indices {computed!r}",
            )
        indices = frozen
    else:
        indices = computed
    raw_lambdas = candidate.get("lambda_values")
    if not isinstance(raw_lambdas, list) or not raw_lambdas:
        raise CompatExportError(NO_LAMBDA_GRID, "candidate.lambda_values must be a non-empty array")
    try:
        lambdas = [float(value) for value in raw_lambdas]
    except (TypeError, ValueError) as exc:
        raise CompatExportError(NO_LAMBDA_GRID, "candidate.lambda_values must be numbers") from exc
    schedule_values = driver.get("schedule_values")
    if schedule_values is None:
        s_values = lambdas
    else:
        if not isinstance(schedule_values, (list, tuple)):
            raise CompatExportError(SCHEDULE_VALUE_COUNT_MISMATCH, "driver.schedule_values must be null or an array")
        try:
            s_values = [float(value) for value in schedule_values]
        except (TypeError, ValueError) as exc:
            raise CompatExportError(SCHEDULE_VALUE_COUNT_MISMATCH, "driver.schedule_values must be numbers") from exc
        if len(s_values) != len(lambdas):
            raise CompatExportError(
                SCHEDULE_VALUE_COUNT_MISMATCH,
                f"schedule_values has {len(s_values)} entries but lambda_values has {len(lambdas)}",
            )
    start_endpoint = candidate.get("start_endpoint")
    if start_endpoint not in ("R", "P"):
        raise CompatExportError(INVALID_START_ENDPOINT, f"start_endpoint={start_endpoint!r}")
    q_start, q_end = _endpoint_distances(mats, str(start_endpoint), maps)
    if q_start == q_end:
        raise CompatExportError(
            SCAN_RANGE_DEGENERATE,
            f"endpoint distances for maps {maps} are identical ({q_start:.6g}); nothing to scan",
        )
    # Freeze the endpoints at contract precision first, then interpolate —
    # the resulting list satisfies the adapter's uniformity predicate by
    # construction (same formula, same rounded endpoints).
    p_start = round(q_start, 6)
    p_end = round(q_end, 6)
    points = [round(p_start + s_value * (p_end - p_start), 6) for s_value in s_values]
    _check_points(points)
    budget = candidate.get("budget")
    if not isinstance(budget, Mapping):
        raise CompatExportError(BACKEND_INVALID, "candidate.budget must be an object")
    attempts = budget.get("max_attempts", 1)
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
        attempts = 1
    retry_policy = {
        "failure_policy": opts.failure_policy,
        "scan_retry_count": max(0, attempts - 1),
        "optimizer_retries": opts.optimizer_retries,
        "reuse_previous_geometry": opts.reuse_previous_geometry,
    }
    backend = plan_v2.get("backend")
    if not isinstance(backend, Mapping):
        raise CompatExportError(BACKEND_INVALID, "plan.backend must be an object")
    engine = str(backend.get("engine") or "").lower()
    if engine not in {"orca", "xtb"}:
        raise CompatExportError(
            BACKEND_ENGINE_UNSUPPORTED,
            f"backend engine={backend.get('engine')!r}; the ACP baseline supports orca/xtb only",
        )
    method_name = backend.get("method")
    if not isinstance(method_name, str) or not method_name:
        raise CompatExportError(BACKEND_METHOD_MISSING, "plan.backend.method must be a non-empty string")
    method = {"engine": engine, "method": method_name}
    observer_pairs: set[tuple[int, int]] = set()
    driver_pair = set(maps)
    monitors = candidate.get("monitors")
    if isinstance(monitors, list):
        for monitor in monitors:
            if not isinstance(monitor, Mapping):
                continue
            monitor_maps = monitor.get("maps")
            if not isinstance(monitor_maps, list) or len(monitor_maps) != 2:
                continue
            try:
                pair = tuple(sorted((int(monitor_maps[0]), int(monitor_maps[1]))))
            except (TypeError, ValueError):
                continue
            if set(pair) != driver_pair:
                observer_pairs.add(pair)
    plan_id = str(plan_v2["plan_id"])
    candidate_id = f"{plan_id}:c{index + 1:03d}"
    start_name = "reactant" if start_endpoint == "R" else "product"
    direction = candidate.get("direction")
    if direction not in ("R_to_P", "P_to_R"):
        raise CompatExportError(IDENTITY_MISMATCH, f"direction={direction!r}; expected 'R_to_P' or 'P_to_R'")
    return {
        "rank": index + 1,
        "start_endpoint": start_name,
        "direction": direction,
        "observer_pairs_map": [list(pair) for pair in sorted(observer_pairs)],
        "retry_policy": retry_policy,
        "method": method,
        "budget": {
            "max_attempts": int(budget.get("max_attempts", 1) or 0),
            "max_cpu_hours": float(budget.get("max_cpu_hours", 0.0) or 0.0),
            "max_wall_seconds": int(budget.get("max_wall_seconds", 0) or 0),
        },
        "coordinates": [
            {
                "coordinate_id": "driver-01",
                "role": "driver",
                "atom_map_ids": maps,
                "atom_indices": indices,
                "unit": "angstrom",
                "direction": "stretch" if points[-1] >= points[0] else "contract",
                "points": points,
            }
        ],
        "fallback_candidate_ids": [],
        "candidate_id": candidate_id,
        "request_id": f"{candidate_id}:attempt1",
    }


def export_legacy_scan_plan(
    generation_plan_v2_doc: Any,
    materials: Any,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a frozen SINGLE_1D single-B ``g1_generation_plan_v2`` to a v1 ScanPlan.

    Returns a sealed v1 ``ScanPlan`` document (status ``ready``) that passes
    ``contracts.validate_document``/``dumps_document`` and the adapter preflight
    ``scan_plan_to_acp_request``.  Every refusal raises
    :class:`CompatExportError` (an ``ACPMappingError``) with a typed ``code``;
    no multi-coordinate, A/D, non-uniform, or needs_review input is ever
    silently projected onto the legacy single-B shape.
    """
    plan_v2 = _require_plan_doc(generation_plan_v2_doc)
    mats = _parse_materials(materials)
    _bind_plan_to_materials(plan_v2, mats)
    opts = _compat_config(config)
    raw_candidates = plan_v2.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise CompatExportError(EMPTY_CANDIDATES, "frozen plan carries no candidates to export")
    v1_candidates = [
        _export_candidate(plan_v2, candidate, mats, opts, index)
        for index, candidate in enumerate(raw_candidates)
    ]
    plan_id = str(plan_v2["plan_id"])
    return make_document(
        "ScanPlan",
        plan_id,
        "ready",
        experiment_id=opts.experiment_id,
        dataset_version=mats.dataset_version,
        reaction_id=str(plan_v2["reaction_id"]),
        case_id=str(plan_v2["case_id"]),
        plan_id=plan_id,
        split=str(plan_v2["split"]),
        plan_version=int(plan_v2["plan_version"]),
        atom_map_ids=list(mats.atom_map_ids),
        n_atoms=len(mats.atom_map_ids),
        candidate_strategy=CANDIDATE_STRATEGY,
        candidates=v1_candidates,
        source_case_sha256=str(plan_v2["source_case_sha256"]),
        plan_frozen=True,
    )
