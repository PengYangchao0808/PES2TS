"""PathRequest / NEB protocol for the graph scan-strategy selector (design §8).

Implements the ``PATH_REQUIRED`` protocol of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`` (§8 mode table,
§8.2 staged-evidence rule, §9.1 NEB capability row):

- :func:`make_path_request` builds a typed :class:`PathRequest` from an
  :class:`EndpointGraphBundle`, whitelist endpoint materials, the
  ``scan_strategy`` config section, and explicit image-chain parameters.
  **Both endpoint geometry blocks are row-aligned to one frozen
  ``atom_rows`` order** (materials ``atom_map_ids`` when present, otherwise
  the bundle's conservation order); a missing endpoint geometry or an
  atom-order mismatch is a typed refusal — coordinates are **never**
  fabricated, and this protocol never invents a "NEB distance coordinate"
  for older contracts.
- **NEB and Scan are accounted separately** (design §8.3/§13.4): cost and
  success live per ``method_kind`` channel (``NEB`` | ``scan``).  A reaction
  that enters NEB is recorded as *exiting* the native scan branch; NEB
  outcomes never increment scan-channel successes.
- **NEB intermediate minima are not a guaranteed single elementary process**
  (design §8.2): the caveat is structural (:class:`RecoveryProtocol`
  policy block), and :func:`flag_intermediate_minimum` turns a recovered
  energy profile into a typed :class:`IntermediateMinimumFlag` for the
  adjacent-segment handling owned by the intermediate-staging todo.
- :meth:`PathRequest.to_path_candidate` projects the request into the
  contracts_v2 ``PathCandidateV1`` discriminated-union payload that
  ``plan_freeze.freeze_path_candidate_plan`` consumes.
"""

# allow: SIZE_OK — plan-named todo-26 single protocol module (PathRequest +
# typed refusals + method channels + intermediate flag + candidate
# projection); precedent contracts_v2.py / plan_freeze.py / compile_orca.py.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS as _V1_TRUTH_KEYS
from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS as _V2_EXPORT_KEYS
from pes2ts_core.generation.planning.contracts_v2 import (
    CANDIDATE_KIND_PATH,
    DIRECTIONS,
    ENDPOINTS,
    PATH_METHOD_KINDS,
)
from pes2ts_core.generation.planning.coordinate_pool import EndpointMaterials
from pes2ts_core.utils.hashing import stable_json_dumps

# ---------------------------------------------------------------------------
# Schema vocabulary and typed refusal codes.
# ---------------------------------------------------------------------------
SCHEMA_PATH_REQUEST: Final[str] = "g1_path_request_v1"
OBJECT_PATH_REQUEST: Final[str] = "PathRequest"
PATH_REQUEST_VERSION: Final[str] = "path_request_protocol_v1"

CODE_PATH_REQUEST_INVALID: Final[str] = "PATH_REQUEST_INVALID"
CODE_BUNDLE_INVALID: Final[str] = "BUNDLE_INVALID"
CODE_MATERIALS_INVALID: Final[str] = "MATERIALS_SCHEMA_INVALID"
CODE_ENDPOINT_GEOMETRY_MISSING: Final[str] = "ENDPOINT_GEOMETRY_MISSING"
CODE_ATOM_ORDER_MISMATCH: Final[str] = "ATOM_ORDER_MISMATCH"
CODE_IMAGE_CHAIN_INVALID: Final[str] = "IMAGE_CHAIN_INVALID"

#: Method-channel vocabulary — NEB and Scan are accounted separately.
CHANNEL_SCAN: Final[str] = "scan"
CHANNEL_NEB: Final[str] = "NEB"
METHOD_CHANNELS: Final[tuple[str, ...]] = (CHANNEL_NEB, CHANNEL_SCAN)
#: Branch marker recorded on both channels when a reaction enters NEB.
BRANCH_NATIVE_SCAN_EXITED: Final[str] = "native_scan_branch_exited"
BRANCH_NATIVE_SCAN: Final[str] = "native_scan_branch"
#: Entering NEB exits the native scan branch; a NEB outcome is never a
#: scan-channel success (design §13.4 cost/success split).
NEB_ENTER_EXITS_SCAN_BRANCH: Final[str] = (
    "entering NEB exits the native scan branch; NEB cost/success is "
    "accounted only on the NEB channel"
)
SCAN_CHANNEL_COST_NOTE: Final[str] = (
    "scan-channel cost/success stops at the branch exit; NEB attempts are "
    "not scan successes"
)

#: Per-image data the NEB recovery protocol collects (design §11.1/§9.1).
RECOVERY_PER_IMAGE_FIELDS: Final[tuple[str, ...]] = (
    "per_image_energy",
    "gradient_availability",
    "convergence",
)
#: Structural caveat: a NEB-found intermediate minimum never guarantees a
#: single elementary process; deep intermediates require adjacent-segment
#: handling (design §8.2; staging owned by todo 27).
SINGLE_ELEMENTARY_PROCESS_GUARANTEED: Final[bool] = False
INTERMEDIATE_NEXT_STAGE: Final[str] = "intermediate_evidence_staging"
INTERMEDIATE_MINIMUM_CODE: Final[str] = "NEB_INTERMEDIATE_MINIMUM"
DEEP_INTERMEDIATE_CODE: Final[str] = "DEEP_INTERMEDIATE_MINIMUM"
ON_INTERMEDIATE_MINIMUM: Final[str] = "flag_for_adjacent_segment_handling"
ON_DEEP_INTERMEDIATE: Final[str] = "adjacent_segment_handling_required"

RECIPE_PATH_REQUEST_V1: Final[str] = "pes2ts_path_request_recipe_v1"
NEB_BACKEND_ID_DEFAULT: Final[str] = "orca"
ANCHOR_REASON_DEFAULT: Final[str] = "path_request_protocol_v1"
DEFAULT_METHOD_KIND: Final[str] = PATH_METHOD_KINDS[0]

#: Endpoint side key -> materials coordinate key (both endpoint blocks).
_ENDPOINT_SIDES: Final[tuple[tuple[str, str], ...]] = (
    ("reactant", "r_coordinates"),
    ("product", "p_coordinates"),
)

#: Every key that must never appear in a PathRequest document or projection.
FORBIDDEN_PATH_KEYS: Final[frozenset[str]] = frozenset(
    {key.lower() for key in _V1_TRUTH_KEYS}
    | {key.lower() for key in _V2_EXPORT_KEYS}
)
#: Keys that would mean "fabricated NEB distance coordinate for an older
#: scan-shaped contract" — this protocol refuses to produce them.
FABRICATED_DISTANCE_KEYS: Final[frozenset[str]] = frozenset(
    {"neb_distance_coordinate", "neb_distance_coordinates", "distance_coordinate",
     "drivers", "lambda_values", "schedule_id", "schedule_kind", "assembly_id",
     "mode"}
)


class PathRequestError(ValueError):
    """Typed PathRequest refusal; ``code`` is a stable machine token."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


# ---------------------------------------------------------------------------
# Image-chain parameters (design §8 PATH_REQUIRED row: n_images + spring /
# NEB method params; backend-id recorded for non-ORCA PATH algorithms).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ImageChainParams:
    """Frozen NEB image-chain parameters for one PathRequest."""

    n_images: int
    spring_constant: float | None = None
    neb_backend_id: str | None = None

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection; optional params appear only when set."""
        doc: dict[str, Any] = {"n_images": self.n_images}
        if self.spring_constant is not None:
            doc["spring_constant"] = self.spring_constant
        if self.neb_backend_id is not None:
            doc["neb_backend_id"] = self.neb_backend_id
        return doc


def _positive_int(raw: Any, field: str, code: str = CODE_IMAGE_CHAIN_INVALID) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise PathRequestError(code, f"{field} must be a positive integer")
    return raw


def _optional_positive_float(raw: Any, field: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PathRequestError(CODE_IMAGE_CHAIN_INVALID, f"{field} must be a number")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise PathRequestError(CODE_IMAGE_CHAIN_INVALID, f"{field} must be finite and > 0")
    return value


def _parse_image_chain(raw: Any) -> ImageChainParams:
    """Parse image-chain params from an int, mapping, or dataclass."""
    if isinstance(raw, ImageChainParams):
        return raw
    if isinstance(raw, bool):
        raise PathRequestError(CODE_IMAGE_CHAIN_INVALID, "image chain must be an object or int")
    if isinstance(raw, int):
        return ImageChainParams(n_images=_positive_int(raw, "n_images"))
    if not isinstance(raw, Mapping):
        raise PathRequestError(
            CODE_IMAGE_CHAIN_INVALID, "image_chain_params must be a mapping, int, or ImageChainParams"
        )
    n_images = _positive_int(raw.get("n_images"), "image_chain.n_images")
    spring = _optional_positive_float(raw.get("spring_constant"), "image_chain.spring_constant")
    backend_raw = raw.get("neb_backend_id")
    backend: str | None
    if backend_raw is None:
        backend = None
    elif isinstance(backend_raw, str) and backend_raw.strip():
        backend = backend_raw.strip()
    else:
        raise PathRequestError(
            CODE_IMAGE_CHAIN_INVALID, "image_chain.neb_backend_id must be a non-empty string"
        )
    return ImageChainParams(n_images=n_images, spring_constant=spring, neb_backend_id=backend)


# ---------------------------------------------------------------------------
# Method channels — NEB and Scan accounted separately (design §8.3/§13.4).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MethodChannelAccounting:
    """Immutable cost/success channel for one ``method_kind``.

    Channels are never merged: a reaction that enters NEB records a branch
    exit on the scan channel and counts its cost/success only on the NEB
    channel.  ``n_cpu_hours`` stays ``0.0`` until a backend measures CPU
    time (ACP does not currently report it — §13.4 requires measured cost,
    never a guessed denominator).
    """

    method_kind: str
    n_entered: int
    n_succeeded: int
    n_failed: int
    branch: str
    cost_note: str
    n_cpu_hours: float = 0.0

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection of one channel."""
        return {
            "method_kind": self.method_kind,
            "n_entered": self.n_entered,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "branch": self.branch,
            "cost_note": self.cost_note,
            "n_cpu_hours": self.n_cpu_hours,
        }


def enter_neb_channels() -> tuple[MethodChannelAccounting, ...]:
    """Accounting snapshot for a reaction *entering* NEB.

    Returns one entry per channel in :data:`METHOD_CHANNELS` order:
    ``NEB`` records the entry; ``scan`` records the branch exit with zero
    success — entering NEB is never a scan success.
    """
    neb = MethodChannelAccounting(
        method_kind=CHANNEL_NEB,
        n_entered=1,
        n_succeeded=0,
        n_failed=0,
        branch=BRANCH_NATIVE_SCAN_EXITED,
        cost_note=NEB_ENTER_EXITS_SCAN_BRANCH,
    )
    scan = MethodChannelAccounting(
        method_kind=CHANNEL_SCAN,
        n_entered=0,
        n_succeeded=0,
        n_failed=0,
        branch=BRANCH_NATIVE_SCAN_EXITED,
        cost_note=SCAN_CHANNEL_COST_NOTE,
    )
    return (neb, scan)


def record_channel_outcome(
    channels: Sequence[MethodChannelAccounting],
    method_kind: str,
    *,
    succeeded: bool,
) -> tuple[MethodChannelAccounting, ...]:
    """Return a new channel tuple with one outcome recorded on ``method_kind``.

    Only the addressed channel's counters change; every other channel is
    copied verbatim (test lock: a NEB success never increments scan).
    """
    if method_kind not in METHOD_CHANNELS:
        raise PathRequestError(
            CODE_PATH_REQUEST_INVALID,
            f"method_kind={method_kind!r}; expected one of {', '.join(METHOD_CHANNELS)}",
        )
    out: list[MethodChannelAccounting] = []
    for channel in channels:
        if channel.method_kind != method_kind:
            out.append(channel)
            continue
        if succeeded:
            out.append(
                MethodChannelAccounting(
                    method_kind=channel.method_kind,
                    n_entered=channel.n_entered,
                    n_succeeded=channel.n_succeeded + 1,
                    n_failed=channel.n_failed,
                    branch=channel.branch,
                    cost_note=channel.cost_note,
                    n_cpu_hours=channel.n_cpu_hours,
                )
            )
        else:
            out.append(
                MethodChannelAccounting(
                    method_kind=channel.method_kind,
                    n_entered=channel.n_entered,
                    n_succeeded=channel.n_succeeded,
                    n_failed=channel.n_failed + 1,
                    branch=channel.branch,
                    cost_note=channel.cost_note,
                    n_cpu_hours=channel.n_cpu_hours,
                )
            )
    return tuple(out)


def _channels_doc(channels: Sequence[MethodChannelAccounting]) -> dict[str, Any]:
    return {channel.method_kind: channel.to_doc() for channel in channels}


# ---------------------------------------------------------------------------
# Recovery protocol + intermediate-minimum typed flag (design §8.2/§11.1).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RecoveryProtocol:
    """What per-image NEB data the backend must recover, plus the
    intermediate-minimum policy frozen into every PathRequest."""

    per_image_fields: tuple[str, ...] = RECOVERY_PER_IMAGE_FIELDS

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection including the structural caveat."""
        return {
            "per_image_fields": list(self.per_image_fields),
            "intermediate_minimum_policy": {
                "single_elementary_process_guaranteed": SINGLE_ELEMENTARY_PROCESS_GUARANTEED,
                "on_intermediate_minimum": ON_INTERMEDIATE_MINIMUM,
                "on_deep_intermediate": ON_DEEP_INTERMEDIATE,
                "deep_threshold_source": "scan_strategy.path_request.deep_intermediate_threshold",
                "handled_by": INTERMEDIATE_NEXT_STAGE,
            },
        }


@dataclass(frozen=True, slots=True)
class IntermediateMinimumFlag:
    """Typed flag for a NEB energy profile that contains an interior minimum.

    A NEB-found intermediate minimum never guarantees a single elementary
    process; a *deep* interior minimum additionally requires adjacent-segment
    handling (design §8.2).  This module only produces the typed flag —
    staging/plan versioning is owned by the intermediate-evidence todo.
    """

    interior_minimum: bool
    image_index: int | None
    energy_profile: tuple[float, ...]
    depth: float | None
    deep_intermediate: bool
    adjacent_segment_required: bool
    reason_code: str
    single_elementary_process_guaranteed: bool = SINGLE_ELEMENTARY_PROCESS_GUARANTEED
    next_stage: str = INTERMEDIATE_NEXT_STAGE

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection of the flag."""
        return {
            "interior_minimum": self.interior_minimum,
            "image_index": self.image_index,
            "energy_profile": list(self.energy_profile),
            "depth": self.depth,
            "deep_intermediate": self.deep_intermediate,
            "adjacent_segment_required": self.adjacent_segment_required,
            "reason_code": self.reason_code,
            "single_elementary_process_guaranteed": self.single_elementary_process_guaranteed,
            "next_stage": self.next_stage,
        }


def _finite_float_tuple(raw: Any) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise PathRequestError(
            CODE_PATH_REQUEST_INVALID, "energy profile must be a non-empty array of numbers"
        )
    out: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID, "energy profile entries must be numbers"
            )
        value = float(item)
        if not math.isfinite(value):
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID, "energy profile entries must be finite"
            )
        out.append(value)
    return tuple(out)


def flag_intermediate_minimum(
    energies: Sequence[float],
    *,
    deep_threshold: float | None = None,
) -> IntermediateMinimumFlag:
    """Classify a recovered NEB energy profile into a typed flag.

    An *interior local minimum* (image ``i`` with ``e[i] <= e[i-1]`` and
    ``e[i] <= e[i+1]``, strict on at least one side) sets
    ``interior_minimum=True`` and always clears the single-elementary-process
    guarantee.  ``deep_intermediate`` additionally requires
    ``deep_threshold`` (uncalibrated by default — design has no frozen
    number) and ``depth >= deep_threshold``; deep intermediates set
    ``adjacent_segment_required=True`` for the todo-27 staging stage.
    """
    profile = _finite_float_tuple(energies)
    interior_index: int | None = None
    interior_energy: float | None = None
    for index in range(1, len(profile) - 1):
        current, prev, nxt = profile[index], profile[index - 1], profile[index + 1]
        if current <= prev and current <= nxt and (current < prev or current < nxt):
            if interior_energy is None or current < interior_energy:
                interior_energy = current
                interior_index = index
    if interior_index is None:
        return IntermediateMinimumFlag(
            interior_minimum=False,
            image_index=None,
            energy_profile=profile,
            depth=None,
            deep_intermediate=False,
            adjacent_segment_required=False,
            reason_code="NO_INTERIOR_MINIMUM",
        )
    depth = min(profile[interior_index - 1], profile[interior_index + 1]) - float(
        profile[interior_index]
    )
    deep = deep_threshold is not None and depth >= deep_threshold
    return IntermediateMinimumFlag(
        interior_minimum=True,
        image_index=interior_index,
        energy_profile=profile,
        depth=depth,
        deep_intermediate=deep,
        adjacent_segment_required=deep,
        reason_code=DEEP_INTERMEDIATE_CODE if deep else INTERMEDIATE_MINIMUM_CODE,
    )


# ---------------------------------------------------------------------------
# Materials + bundle parsing at the single trust boundary.
# ---------------------------------------------------------------------------
def _parse_xyz(raw: Any, where: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise PathRequestError(CODE_MATERIALS_INVALID, f"{where} must be a length-3 coordinate")
    coords: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PathRequestError(
                CODE_MATERIALS_INVALID, f"{where} coordinates must be numbers"
            )
        value = float(item)
        if not math.isfinite(value):
            raise PathRequestError(
                CODE_MATERIALS_INVALID, f"{where} coordinates must be finite"
            )
        coords.append(value)
    return (coords[0], coords[1], coords[2])


def _parse_map_id(raw: Any, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise PathRequestError(CODE_ATOM_ORDER_MISMATCH, f"{where} must be a positive integer")
    return raw


def _coerce_finite_float(raw: Any, field: str, *, minimum: float | None = None) -> float:
    """Narrow an untrusted numeric field to a finite float (boundary parse).

    Numeric strings are accepted (YAML 1.1 quirk: ``1.0e6``-style values load
    as ``str`` — precedent geometry_feasibility/direction_assembly todo 13/15).
    """
    value: float
    if isinstance(raw, bool):
        raise PathRequestError(CODE_PATH_REQUEST_INVALID, f"{field} must be a number")
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError as exc:
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID, f"{field} must be a number"
            ) from exc
    else:
        raise PathRequestError(CODE_PATH_REQUEST_INVALID, f"{field} must be a number")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        bound = "" if minimum is None else f" and >= {minimum}"
        raise PathRequestError(
            CODE_PATH_REQUEST_INVALID, f"{field} must be finite{bound}"
        )
    return value


def _doc_count(raw: Any, field: str) -> int:
    """Narrow a document counter to a non-negative int (0 when absent)."""
    if raw is None:
        return 0
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise PathRequestError(
            CODE_PATH_REQUEST_INVALID, f"{field} must be a non-negative integer"
        )
    return raw


def _side_coordinates(raw: Any, key: str) -> dict[int, tuple[float, float, float]]:
    """Normalize one materials side into a map-keyed coordinate table."""
    if raw is None:
        raise PathRequestError(
            CODE_ENDPOINT_GEOMETRY_MISSING,
            f"materials.{key}: side block missing — endpoint geometry is required",
        )
    if isinstance(raw, Mapping):
        parsed: dict[int, tuple[float, float, float]] = {}
        for map_key, value in raw.items():
            map_id = _parse_map_id(map_key, f"materials.{key} key")
            if isinstance(value, Mapping):
                value = value.get("coordinates")
            parsed[map_id] = _parse_xyz(value, f"materials.{key}[{map_id}]")
        return parsed
    if isinstance(raw, (list, tuple)):
        parsed = {}
        for index, row in enumerate(raw):
            if not isinstance(row, Mapping):
                raise PathRequestError(
                    CODE_MATERIALS_INVALID,
                    f"materials.{key}[{index}]: row must be a mapping with map+coordinates",
                )
            map_id = _parse_map_id(row.get("map"), f"materials.{key}[{index}].map")
            if map_id in parsed:
                raise PathRequestError(
                    CODE_ATOM_ORDER_MISMATCH,
                    f"materials.{key}: duplicate map {map_id} in row list",
                )
            parsed[map_id] = _parse_xyz(
                row.get("coordinates"), f"materials.{key}[{index}].coordinates"
            )
        return parsed
    raise PathRequestError(
        CODE_MATERIALS_INVALID,
        f"materials.{key}: expected a map-keyed mapping or a row list",
    )


@dataclass(frozen=True, slots=True)
class _ParsedMaterials:
    """Whitelist-parsed endpoint materials: map-keyed coordinates per side."""

    r_coordinates: Mapping[int, tuple[float, float, float]]
    p_coordinates: Mapping[int, tuple[float, float, float]]
    atom_map_ids: tuple[int, ...] | None


def _parse_materials(materials: Any) -> _ParsedMaterials:
    """Parse materials into typed per-side coordinate tables (boundary once)."""
    if materials is None:
        raise PathRequestError(
            CODE_ENDPOINT_GEOMETRY_MISSING,
            "endpoint materials required — missing endpoint geometry is a typed "
            "refusal; coordinates are never fabricated",
        )
    if isinstance(materials, EndpointMaterials):
        return _ParsedMaterials(
            r_coordinates=dict(materials.r_coordinates),
            p_coordinates=dict(materials.p_coordinates),
            atom_map_ids=None,
        )
    if not isinstance(materials, Mapping):
        raise PathRequestError(
            CODE_MATERIALS_INVALID,
            "materials must be an EndpointMaterials or a mapping with r/p coordinate blocks",
        )
    atom_map_ids: tuple[int, ...] | None = None
    raw_order = materials.get("atom_map_ids")
    if raw_order is not None:
        if not isinstance(raw_order, (list, tuple)) or not raw_order:
            raise PathRequestError(
                CODE_ATOM_ORDER_MISMATCH, "materials.atom_map_ids must be a non-empty array"
            )
        parsed_order: list[int] = []
        for index, entry in enumerate(raw_order):
            parsed_order.append(_parse_map_id(entry, f"materials.atom_map_ids[{index}]"))
        if len(set(parsed_order)) != len(parsed_order):
            raise PathRequestError(
                CODE_ATOM_ORDER_MISMATCH, "materials.atom_map_ids must be unique"
            )
        atom_map_ids = tuple(parsed_order)

    def _resolve(key: str, alt: str) -> dict[int, tuple[float, float, float]]:
        raw = materials.get(key, materials.get(alt))
        if raw is None:
            raise PathRequestError(
                CODE_ENDPOINT_GEOMETRY_MISSING,
                f"materials.{key}: side block missing — endpoint geometry is required",
            )
        if isinstance(raw, (list, tuple)) and atom_map_ids is not None:
            if len(raw) != len(atom_map_ids):
                raise PathRequestError(
                    CODE_MATERIALS_INVALID,
                    f"materials.{key}: aligned coordinate list length {len(raw)} "
                    f"!= atom_map_ids length {len(atom_map_ids)}",
                )
            return {
                map_id: _parse_xyz(row, f"materials.{key}[{position}]")
                for position, (map_id, row) in enumerate(zip(atom_map_ids, raw, strict=True))
            }
        if isinstance(raw, (list, tuple)):
            if raw and all(isinstance(row, Mapping) for row in raw):
                return _side_coordinates(raw, key)
            raise PathRequestError(
                CODE_MATERIALS_INVALID,
                f"materials.{key}: aligned coordinate lists require materials.atom_map_ids",
            )
        return _side_coordinates(raw, key)

    return _ParsedMaterials(
        r_coordinates=_resolve("r_coordinates", "r"),
        p_coordinates=_resolve("p_coordinates", "p"),
        atom_map_ids=atom_map_ids,
    )


def _bundle_map_ids(bundle: EndpointGraphBundle) -> tuple[int, ...]:
    conservation = bundle.conservation
    maps = tuple(conservation.map_ids)
    if not maps:
        raise PathRequestError(CODE_BUNDLE_INVALID, "bundle conservation.map_ids is empty")
    r_maps = frozenset(node.map_id for node in bundle.r_graph.nodes)
    p_maps = frozenset(node.map_id for node in bundle.p_graph.nodes)
    expected = frozenset(maps)
    if r_maps != expected or p_maps != expected:
        raise PathRequestError(
            CODE_ATOM_ORDER_MISMATCH,
            "bundle R/P graph map sets disagree with conservation.map_ids — "
            "endpoint atom order is not a common frozen order",
        )
    return maps


def _bundle_elements(
    bundle: EndpointGraphBundle, atom_rows: Sequence[int]
) -> tuple[str, ...]:
    by_map = {node.map_id: node.element for node in bundle.r_graph.nodes}
    missing = sorted(set(atom_rows) - by_map.keys())
    if missing:
        raise PathRequestError(
            CODE_ATOM_ORDER_MISMATCH, f"bundle is missing elements for maps {missing}"
        )
    return tuple(by_map[map_id] for map_id in atom_rows)


def _resolve_atom_rows(
    bundle_maps: tuple[int, ...], materials: _ParsedMaterials
) -> tuple[int, ...]:
    """Frozen atom order: materials ``atom_map_ids`` claim, else bundle order."""
    if materials.atom_map_ids is None:
        return bundle_maps
    claimed = set(materials.atom_map_ids)
    expected = set(bundle_maps)
    if claimed != expected:
        missing = sorted(expected - claimed)
        extra = sorted(claimed - expected)
        raise PathRequestError(
            CODE_ATOM_ORDER_MISMATCH,
            f"materials.atom_map_ids does not cover the bundle atom set "
            f"(missing={missing}, extra={extra})",
        )
    return materials.atom_map_ids


def _endpoint_block(
    coordinates: Mapping[int, tuple[float, float, float]],
    atom_rows: Sequence[int],
    side: str,
) -> tuple[tuple[float, float, float], ...]:
    missing = sorted(set(atom_rows) - coordinates.keys())
    if missing:
        raise PathRequestError(
            CODE_ENDPOINT_GEOMETRY_MISSING,
            f"endpoint {side} geometry missing maps {missing} — refusal; "
            "coordinates are never fabricated",
        )
    return tuple(coordinates[map_id] for map_id in atom_rows)


def _policy_from_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract the path-request policy slice from the scan_strategy section."""
    policy: dict[str, Any] = {}
    if config is None:
        return policy
    raw = config.get("scan_strategy")
    if not isinstance(raw, Mapping):
        return policy
    caps = raw.get("max_total_candidates")
    if isinstance(caps, Mapping):
        raw_neb = caps.get("path_neb")
        if isinstance(raw_neb, int) and not isinstance(raw_neb, bool):
            policy["max_path_neb_candidates"] = raw_neb
    section = raw.get("path_request")
    if isinstance(section, Mapping) and section.get("deep_intermediate_threshold") is not None:
        policy["deep_intermediate_threshold"] = _coerce_finite_float(
            section["deep_intermediate_threshold"],
            "scan_strategy.path_request.deep_intermediate_threshold",
            minimum=0.0,
        )
    return policy


def _request_id(bundle_sha: str, reaction_id: str | None) -> str:
    if reaction_id:
        return f"{reaction_id}:{PATH_REQUEST_VERSION}"
    return f"pathreq:{bundle_sha[:16]}"


# ---------------------------------------------------------------------------
# PathRequest document.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PathRequest:
    """Independent NEB path request: both endpoints in one frozen atom order."""

    schema_version: str
    request_id: str
    reaction_id: str | None
    bundle_content_sha256: str
    atom_rows: tuple[int, ...]
    n_atoms: int
    elements: tuple[str, ...]
    endpoint_geometries: Mapping[str, tuple[tuple[float, float, float], ...]]
    image_chain: ImageChainParams
    method_kind: str
    recovery_protocol: RecoveryProtocol
    method_channels: tuple[MethodChannelAccounting, ...]
    policy: Mapping[str, Any]

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe PathRequest document (deterministic, no volatile keys)."""
        return {
            "schema_name": OBJECT_PATH_REQUEST,
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "reaction_id": self.reaction_id,
            "bundle_content_sha256": self.bundle_content_sha256,
            "atom_rows": list(self.atom_rows),
            "n_atoms": self.n_atoms,
            "elements": list(self.elements),
            "endpoint_geometries": {
                side: [list(row) for row in self.endpoint_geometries[side]]
                for side, _ in _ENDPOINT_SIDES
            },
            "image_chain": self.image_chain.to_doc(),
            "method_kind": self.method_kind,
            "recovery_protocol": self.recovery_protocol.to_doc(),
            "method_channels": _channels_doc(self.method_channels),
            "policy": dict(self.policy),
        }

    def to_path_candidate(
        self,
        *,
        candidate_id: str,
        start_endpoint: str = "R",
        direction: str | None = None,
        anchor_reason: str = ANCHOR_REASON_DEFAULT,
        budget: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Project into a contracts_v2 ``PathCandidateV1`` payload.

        The payload is exactly what ``plan_freeze.freeze_path_candidate_plan``
        consumes: same-atom-order endpoint geometries, image-chain params,
        NEB method kind, and a namespaced ``extensions.path_request`` block
        carrying the frozen atom order + channel accounting for freeze/audit.
        """
        if start_endpoint not in ENDPOINTS:
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID, f"start_endpoint={start_endpoint!r}; expected R or P"
            )
        resolved_direction = direction or ("R_to_P" if start_endpoint == "R" else "P_to_R")
        if resolved_direction not in DIRECTIONS:
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID,
                f"direction={resolved_direction!r}; expected one of {', '.join(DIRECTIONS)}",
            )
        if (start_endpoint == "R") != (resolved_direction == "R_to_P"):
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID,
                f"direction={resolved_direction!r} must start from start_endpoint={start_endpoint!r}",
            )
        candidate: dict[str, Any] = {
            "candidate_kind": CANDIDATE_KIND_PATH,
            "candidate_id": candidate_id,
            "n_atoms": self.n_atoms,
            "endpoint_geometries": {
                "reactant": [list(row) for row in self.endpoint_geometries["reactant"]],
                "product": [list(row) for row in self.endpoint_geometries["product"]],
            },
            "image_chain": self.image_chain.to_doc(),
            "start_endpoint": start_endpoint,
            "direction": resolved_direction,
            "anchor_reason": anchor_reason,
            "failure_reasons": [],
            "method_kind": self.method_kind,
            "required_capabilities": ["PATH_NEB"],
            "extensions": {
                "path_request": {
                    "schema_version": self.schema_version,
                    "request_id": self.request_id,
                    "atom_rows": list(self.atom_rows),
                    "bundle_content_sha256": self.bundle_content_sha256,
                    "method_channels": _channels_doc(self.method_channels),
                    "recovery_protocol": self.recovery_protocol.to_doc(),
                }
            },
        }
        if budget is not None:
            candidate["budget"] = dict(budget)
        return candidate


def make_path_request(
    bundle: Any,
    materials: Any,
    config: Mapping[str, Any] | None,
    image_chain_params: Any,
    *,
    reaction_id: str | None = None,
) -> PathRequest:
    """Build a typed NEB PathRequest from bundle + materials + config.

    ``bundle`` must be an :class:`EndpointGraphBundle`; ``materials`` must
    whitelist-provide endpoint coordinates for every atom of the bundle's
    frozen order (map-keyed blocks, or aligned lists + ``atom_map_ids``).
    Refusals are typed (:class:`PathRequestError`): missing endpoint
    geometry → ``ENDPOINT_GEOMETRY_MISSING``; atom-order mismatch →
    ``ATOM_ORDER_MISMATCH``; bad image chain → ``IMAGE_CHAIN_INVALID``.
    Nothing is fabricated on any refusal path.
    """
    if not isinstance(bundle, EndpointGraphBundle):
        raise PathRequestError(
            CODE_BUNDLE_INVALID, "bundle must be an EndpointGraphBundle (todo 5 graph contract)"
        )
    bundle_maps = _bundle_map_ids(bundle)
    parsed = _parse_materials(materials)
    atom_rows = _resolve_atom_rows(bundle_maps, parsed)
    n_atoms = len(atom_rows)
    elements = _bundle_elements(bundle, atom_rows)
    endpoint_geometries = {
        "reactant": _endpoint_block(parsed.r_coordinates, atom_rows, "reactant"),
        "product": _endpoint_block(parsed.p_coordinates, atom_rows, "product"),
    }
    image_chain = _parse_image_chain(image_chain_params)
    policy = _policy_from_config(config)
    return PathRequest(
        schema_version=SCHEMA_PATH_REQUEST,
        request_id=_request_id(bundle.content_sha256, reaction_id),
        reaction_id=reaction_id,
        bundle_content_sha256=bundle.content_sha256,
        atom_rows=atom_rows,
        n_atoms=n_atoms,
        elements=elements,
        endpoint_geometries=endpoint_geometries,
        image_chain=image_chain,
        method_kind=DEFAULT_METHOD_KIND,
        recovery_protocol=RecoveryProtocol(),
        method_channels=enter_neb_channels(),
        policy=policy,
    )


def path_request_from_doc(doc: Mapping[str, Any]) -> PathRequest:
    """Rehydrate a :class:`PathRequest` from its ``to_doc()`` projection."""
    if not isinstance(doc, Mapping) or doc.get("schema_name") != OBJECT_PATH_REQUEST:
        raise PathRequestError(
            CODE_PATH_REQUEST_INVALID, "document is not a PathRequest projection"
        )
    if doc.get("schema_version") != SCHEMA_PATH_REQUEST:
        raise PathRequestError(
            CODE_PATH_REQUEST_INVALID,
            f"schema_version={doc.get('schema_version')!r}; expected {SCHEMA_PATH_REQUEST!r}",
        )
    raw_channels = doc.get("method_channels")
    if not isinstance(raw_channels, Mapping):
        raise PathRequestError(CODE_PATH_REQUEST_INVALID, "method_channels must be an object")
    channels: list[MethodChannelAccounting] = []
    for kind in METHOD_CHANNELS:
        row = raw_channels.get(kind)
        if not isinstance(row, Mapping):
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID, f"method_channels.{kind} missing"
            )
        channels.append(
            MethodChannelAccounting(
                method_kind=str(row.get("method_kind") or kind),
                n_entered=_doc_count(row.get("n_entered"), f"method_channels.{kind}.n_entered"),
                n_succeeded=_doc_count(
                    row.get("n_succeeded"), f"method_channels.{kind}.n_succeeded"
                ),
                n_failed=_doc_count(row.get("n_failed"), f"method_channels.{kind}.n_failed"),
                branch=str(row.get("branch") or ""),
                cost_note=str(row.get("cost_note") or ""),
                n_cpu_hours=(
                    0.0
                    if row.get("n_cpu_hours") is None
                    else _coerce_finite_float(
                        row.get("n_cpu_hours"), f"method_channels.{kind}.n_cpu_hours"
                    )
                ),
            )
        )
    raw_geometries = doc.get("endpoint_geometries")
    if not isinstance(raw_geometries, Mapping):
        raise PathRequestError(
            CODE_PATH_REQUEST_INVALID, "endpoint_geometries must be an object"
        )
    geometries: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for side, _ in _ENDPOINT_SIDES:
        block = raw_geometries.get(side)
        if not isinstance(block, (list, tuple)):
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID, f"endpoint_geometries.{side} must be an array"
            )
        geometries[side] = tuple(
            _parse_xyz(row, f"endpoint_geometries.{side}") for row in block
        )
    raw_chain = doc.get("image_chain")
    image_chain = _parse_image_chain(raw_chain)
    atom_rows_raw = doc.get("atom_rows")
    if not isinstance(atom_rows_raw, (list, tuple)) or not atom_rows_raw:
        raise PathRequestError(CODE_PATH_REQUEST_INVALID, "atom_rows must be a non-empty array")
    atom_rows = tuple(_parse_map_id(entry, "atom_rows entry") for entry in atom_rows_raw)
    raw_n_atoms = doc.get("n_atoms")
    n_atoms = len(atom_rows)
    if raw_n_atoms is not None:
        declared = _doc_count(raw_n_atoms, "n_atoms")
        if declared != n_atoms:
            raise PathRequestError(
                CODE_PATH_REQUEST_INVALID,
                f"n_atoms={declared} disagrees with atom_rows length {n_atoms}",
            )
    elements_raw = doc.get("elements")
    if not isinstance(elements_raw, (list, tuple)) or len(elements_raw) != len(atom_rows):
        raise PathRequestError(
            CODE_PATH_REQUEST_INVALID, "elements must align with atom_rows"
        )
    element_symbols = tuple(str(entry) for entry in elements_raw)
    raw_policy = doc.get("policy")
    policy: dict[str, Any] = dict(raw_policy) if isinstance(raw_policy, Mapping) else {}
    return PathRequest(
        schema_version=str(doc.get("schema_version")),
        request_id=str(doc.get("request_id") or ""),
        reaction_id=None if doc.get("reaction_id") is None else str(doc.get("reaction_id")),
        bundle_content_sha256=str(doc.get("bundle_content_sha256") or ""),
        atom_rows=atom_rows,
        n_atoms=n_atoms,
        elements=element_symbols,
        endpoint_geometries=geometries,
        image_chain=image_chain,
        method_kind=str(doc.get("method_kind") or DEFAULT_METHOD_KIND),
        recovery_protocol=RecoveryProtocol(),
        method_channels=tuple(channels),
        policy=policy,
    )


def request_json(request: PathRequest) -> str:
    """Deterministic stable-JSON serialization of one PathRequest."""
    return stable_json_dumps(request.to_doc())


__all__ = [
    "ANCHOR_REASON_DEFAULT",
    "BRANCH_NATIVE_SCAN",
    "BRANCH_NATIVE_SCAN_EXITED",
    "CODE_ATOM_ORDER_MISMATCH",
    "CODE_BUNDLE_INVALID",
    "CODE_ENDPOINT_GEOMETRY_MISSING",
    "CODE_IMAGE_CHAIN_INVALID",
    "CODE_MATERIALS_INVALID",
    "CODE_PATH_REQUEST_INVALID",
    "CHANNEL_NEB",
    "CHANNEL_SCAN",
    "DEEP_INTERMEDIATE_CODE",
    "DEFAULT_METHOD_KIND",
    "FABRICATED_DISTANCE_KEYS",
    "FORBIDDEN_PATH_KEYS",
    "ImageChainParams",
    "INTERMEDIATE_MINIMUM_CODE",
    "INTERMEDIATE_NEXT_STAGE",
    "IntermediateMinimumFlag",
    "METHOD_CHANNELS",
    "NEB_ENTER_EXITS_SCAN_BRANCH",
    "OBJECT_PATH_REQUEST",
    "ON_DEEP_INTERMEDIATE",
    "ON_INTERMEDIATE_MINIMUM",
    "PATH_REQUEST_VERSION",
    "RECIPE_PATH_REQUEST_V1",
    "RECOVERY_PER_IMAGE_FIELDS",
    "SCHEMA_PATH_REQUEST",
    "SINGLE_ELEMENTARY_PROCESS_GUARANTEED",
    "PathRequest",
    "PathRequestError",
    "RecoveryProtocol",
    "enter_neb_channels",
    "flag_intermediate_minimum",
    "make_path_request",
    "path_request_from_doc",
    "record_channel_outcome",
    "request_json",
]
