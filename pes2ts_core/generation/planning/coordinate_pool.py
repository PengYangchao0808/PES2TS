"""Coordinate pool with driver/monitor/guard/target_test roles (todo 12).

Implements design §6.1–§6.2 of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``:

- **Coordinate generation from the P0 layer + RouteResult (todo 11).**
  F/B edits → bond-distance ``B`` records; H events → two *separate*
  partner distances (donor-H and acceptor-H — never one B = d1−d2),
  the D···H···A angle ``∠DHA`` as ``A``, and optionally the D–A distance;
  ring-closure / attack geometry → local ``A``/``D`` as appropriate;
  conformational differences → ``D`` from endpoint geometry comparison
  (requires ``materials``); O (order-change) edits → monitoring coordinates
  **by default** (never drivers without explicit geometric nomination).
- **Four roles recorded separately** (§6.1): ``driver`` (λ-constrained),
  ``monitor`` (measured, never compiled into an ORCA constraint), ``guard``
  (auxiliary orientation constraint; **defaults to empty** and counts toward
  constraint rank when present), ``target_test`` (coordinates whose final
  values decide target-path compatibility; todo 22 consumes).
- **Event coverage per driver SET** (§6.2): ``direct`` (event has ≥1
  driver), ``coupled_monitor`` (not driven, but monitors of a candidate whose
  drivers cover other events link it via shared centre / same H / same ring /
  aromatic region), ``uncovered``.  Any ``uncovered`` event blocks the
  candidate from the default automatic scan set (typed reason; the candidate
  is still listed as a non-default / ablation probe).
- **All edits monitored in every candidate**: drivers monitor themselves;
  every edit pair has at least one driver-or-monitor coordinate in each
  candidate's role assignment.
- **Determinism**: coordinates ordered by ``(atom_maps, kind, role, origin)``,
  candidates by driver-id tuples; ``to_json`` via ``stable_json_dumps``.

Coordinates reference atoms by **map ids**; map→index resolution is todo 19's
compile-time job.  This module never compiles constraints, never reads the
data trees, and never touches truth.  Inputs are read-only imports of the
todo-5/7/8/9/11 contracts.
"""

# noqa: SIZE_OK — plan-named todo-12 single module (design §6.1–§6.2 coordinate
# pool + four roles + per-candidate event coverage + driver-set enumeration);
# precedent: registry.py (todo 11), event_coupling.py (todo 9),
# contracts_v2.py (todo 3).

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

from pes2ts_core.g1.endpoint_context import EndpointContext
from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.g1.event_coupling import (
    EVENT_H2,
    EVENT_H_TRANSFER,
    EVENT_RING,
    EventCouplingGraph,
    EventNode,
)
from pes2ts_core.g1.reaction_edit_graph import ReactionEditGraph
from pes2ts_core.g1.v2_schema import HH_EVENT_KINDS
from pes2ts_core.generation.planning.contracts_v2 import (
    COVERAGE_KINDS,
    DRIVER_KINDS,
    MAX_SCAN_DRIVERS,
)
from pes2ts_core.generation.planning.registry import RouteResult, StrategyCandidate
from pes2ts_core.utils.hashing import stable_json_dumps

# ---------------------------------------------------------------------------
# Schema + vocabulary.
# ---------------------------------------------------------------------------
SCHEMA_COORDINATE_POOL: Final[str] = "g1_coordinate_pool_v1"

#: Design §6.1 roles.
ROLE_DRIVER: Final[str] = "driver"
ROLE_MONITOR: Final[str] = "monitor"
ROLE_GUARD: Final[str] = "guard"
ROLE_TARGET_TEST: Final[str] = "target_test"
ROLES: Final[tuple[str, ...]] = (ROLE_DRIVER, ROLE_MONITOR, ROLE_GUARD, ROLE_TARGET_TEST)

#: Coordinate kinds (single source: contracts_v2.DRIVER_KINDS).
KIND_B: Final[str] = "B"
KIND_A: Final[str] = "A"
KIND_D: Final[str] = "D"
KINDS: Final[tuple[str, ...]] = DRIVER_KINDS

UNIT_ANGSTROM: Final[str] = "angstrom"
UNIT_DEGREE: Final[str] = "degree"
KIND_UNITS: Final[dict[str, str]] = {
    KIND_B: UNIT_ANGSTROM,
    KIND_A: UNIT_DEGREE,
    KIND_D: UNIT_DEGREE,
}

#: Design §6.2 coverage vocabulary (single source: contracts_v2.COVERAGE_KINDS).
COVERAGE_DIRECT: Final[str] = COVERAGE_KINDS[0]
COVERAGE_COUPLED_MONITOR: Final[str] = COVERAGE_KINDS[1]
COVERAGE_UNCOVERED: Final[str] = COVERAGE_KINDS[2]

#: Typed origin source kinds (which edit / H-event / region produced a record).
ORIGIN_EDIT: Final[str] = "edit"
ORIGIN_HYDROGEN_EVENT: Final[str] = "hydrogen_event"
ORIGIN_RING_GEOMETRY: Final[str] = "ring_geometry"
ORIGIN_CONFORMATIONAL: Final[str] = "conformational_difference"
ORIGIN_KINDS: Final[tuple[str, ...]] = (
    ORIGIN_EDIT,
    ORIGIN_HYDROGEN_EVENT,
    ORIGIN_RING_GEOMETRY,
    ORIGIN_CONFORMATIONAL,
)

#: Hydrogen-event geometry kinds.
GEOM_PARTNER: Final[str] = "partner_distance"
GEOM_ANGLE_DHA: Final[str] = "angle_DHA"
GEOM_DISTANCE_DA: Final[str] = "distance_DA"
GEOM_HH: Final[str] = "hh_distance"

#: Ring / attack geometry kinds.
GEOM_ATTACK_ANGLE: Final[str] = "attack_angle"
GEOM_ATTACK_TORSION: Final[str] = "attack_torsion"

#: Hydrogen partner roles (design: donor-H and acceptor-H stay separate).
PARTNER_DONOR: Final[str] = "donor"
PARTNER_ACCEPTOR: Final[str] = "acceptor"

#: Coupled-monitor evidence rules (design §6.2 "同中心/同 H/同环/区域").
EVID_SHARED_ATOM: Final[str] = "shared_support_atom"
EVID_SAME_AROMATIC_REGION: Final[str] = "same_aromatic_region"
EVID_SAME_RING_REGION: Final[str] = "same_ring_region"
EVID_SAME_HYDROGEN: Final[str] = "same_hydrogen"
EVID_SHARED_EDIT: Final[str] = "shared_edit_membership"

#: Typed non-default reasons (candidate stays listed, never silently dropped).
REASON_UNCOVERED_EVENT: Final[str] = "UNCOVERED_EVENT"
REASON_NO_DRIVER_COORDINATES: Final[str] = "NO_DRIVER_COORDINATES"
REASON_ROUTE_REVIEW_REQUIRED: Final[str] = "ROUTE_REVIEW_REQUIRED"

#: Order-change nomination note (design §6.2: O is monitoring by default).
NOTE_ORDER_CHANGE_MONITOR: Final[str] = "order_change_monitor_by_default"

#: Conformational-difference thresholds (TODO: calibrate — 待校准).
CONFORMATIONAL_DELTA_MIN_DEG: Final[float] = 10.0
MAX_CONFORMATIONAL_DRIVERS: Final[int] = MAX_SCAN_DRIVERS

#: Structural driver-set enumeration caps (finite candidate tree, todo 17 prunes).
COMBINATION_PAIR_CAP: Final[int] = 8
COMBINATION_TRIPLE_CAP: Final[int] = 4

_KIND_RANK: Final[dict[str, int]] = {name: i for i, name in enumerate(KINDS)}
_ROLE_RANK: Final[dict[str, int]] = {name: i for i, name in enumerate(ROLES)}
_ORIGIN_RANK: Final[dict[str, int]] = {
    ORIGIN_HYDROGEN_EVENT: 0,
    ORIGIN_RING_GEOMETRY: 1,
    ORIGIN_CONFORMATIONAL: 2,
    ORIGIN_EDIT: 3,
}
_ELEMENT_H: Final[str] = "H"
_RING_BASIS_SHARED_REGION: Final[str] = "shared_ring_region"


# ---------------------------------------------------------------------------
# Dataclasses.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EndpointMaterials:
    """Map-keyed endpoint coordinates (angstrom) for geometry comparison."""

    r_coordinates: Mapping[int, tuple[float, float, float]]
    p_coordinates: Mapping[int, tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class OriginRecord:
    """Typed origin: which edit / H-event / region produced the coordinate."""

    source_kind: str
    edit_pair: tuple[int, int] | None = None
    edit_kind: str | None = None
    event_id: str | None = None
    hydrogen_map: int | None = None
    partner_map: int | None = None
    partner_role: str | None = None
    geometry_kind: str | None = None
    r_value: float | None = None
    p_value: float | None = None
    note: str | None = None

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection; stable field set."""
        return {
            "source_kind": self.source_kind,
            "edit_pair": None if self.edit_pair is None else [self.edit_pair[0], self.edit_pair[1]],
            "edit_kind": self.edit_kind,
            "event_id": self.event_id,
            "hydrogen_map": self.hydrogen_map,
            "partner_map": self.partner_map,
            "partner_role": self.partner_role,
            "geometry_kind": self.geometry_kind,
            "r_value": self.r_value,
            "p_value": self.p_value,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class CoordinateRecord:
    """One pool coordinate.  Atoms referenced by map ids (todo 19 resolves)."""

    coordinate_id: str
    role: str
    kind: str
    atom_maps: tuple[int, ...]
    units: str
    event_ids: tuple[str, ...]
    origin: OriginRecord

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "coordinate_id": self.coordinate_id,
            "role": self.role,
            "kind": self.kind,
            "atom_maps": list(self.atom_maps),
            "units": self.units,
            "event_ids": list(self.event_ids),
            "origin": self.origin.to_record(),
        }


@dataclass(frozen=True, slots=True)
class EventCoverage:
    """Coverage of one event by one candidate driver set (design §6.2)."""

    event_id: str
    coverage: str
    driver_coordinate_ids: tuple[str, ...]
    monitor_coordinate_ids: tuple[str, ...]
    coupling_evidence: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection (contracts_v2 event_coverage-compatible core)."""
        return {
            "event_id": self.event_id,
            "coverage": self.coverage,
            "driver_coordinate_ids": list(self.driver_coordinate_ids),
            "monitor_coordinate_ids": list(self.monitor_coordinate_ids),
            "coupling_evidence": list(self.coupling_evidence),
        }


@dataclass(frozen=True, slots=True)
class DriverSetCandidate:
    """One finite driver-coordinate-set candidate with role assignment."""

    candidate_id: str
    driver_coordinate_ids: tuple[str, ...]
    monitor_coordinate_ids: tuple[str, ...]
    guard_coordinate_ids: tuple[str, ...]
    target_test_coordinate_ids: tuple[str, ...]
    event_coverage: tuple[EventCoverage, ...]
    has_uncovered_event: bool
    default_automatic: bool
    non_default_reason: str | None
    route_strategy_id: str | None
    route_review_required: bool
    n_drivers: int
    constraint_rank: int
    edits_monitored: bool

    def coverage_of(self, event_id: str) -> EventCoverage | None:
        """Return the coverage row of one event, or ``None``."""
        for row in self.event_coverage:
            if row.event_id == event_id:
                return row
        return None

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "candidate_id": self.candidate_id,
            "driver_coordinate_ids": list(self.driver_coordinate_ids),
            "monitor_coordinate_ids": list(self.monitor_coordinate_ids),
            "guard_coordinate_ids": list(self.guard_coordinate_ids),
            "target_test_coordinate_ids": list(self.target_test_coordinate_ids),
            "event_coverage": [row.to_record() for row in self.event_coverage],
            "has_uncovered_event": self.has_uncovered_event,
            "default_automatic": self.default_automatic,
            "non_default_reason": self.non_default_reason,
            "route_strategy_id": self.route_strategy_id,
            "route_review_required": self.route_review_required,
            "n_drivers": self.n_drivers,
            "constraint_rank": self.constraint_rank,
            "edits_monitored": self.edits_monitored,
        }


@dataclass(frozen=True, slots=True)
class CoordinatePool:
    """Complete coordinate pool + finite driver-set candidates of one reaction."""

    schema_version: str
    reaction_id: str | None
    coordinates: tuple[CoordinateRecord, ...]
    driver_candidates: tuple[DriverSetCandidate, ...]
    all_edits_monitored: bool

    def candidates(self) -> tuple[DriverSetCandidate, ...]:
        """Return the finite driver-set candidates (1–3 drivers each)."""
        return self.driver_candidates

    def coordinates_by_role(self, role: str) -> tuple[CoordinateRecord, ...]:
        """Return every coordinate of one role, in pool order."""
        if role not in ROLES:
            raise ValueError(f"UNKNOWN_COORDINATE_ROLE: {role!r}; expected one of {ROLES}")
        return tuple(c for c in self.coordinates if c.role == role)

    def coordinate(self, coordinate_id: str) -> CoordinateRecord:
        """Return one coordinate by id; raises ``KeyError`` when absent."""
        for record in self.coordinates:
            if record.coordinate_id == coordinate_id:
                return record
        raise KeyError(coordinate_id)

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection; never contains truth-derived keys."""
        return {
            "schema_version": self.schema_version,
            "reaction_id": self.reaction_id,
            "all_edits_monitored": self.all_edits_monitored,
            "coordinates": [record.to_record() for record in self.coordinates],
            "driver_candidates": [
                candidate.to_record() for candidate in self.driver_candidates
            ],
        }

    def to_json(self) -> str:
        """Canonical serialization (determinism test surface)."""
        return stable_json_dumps(self.to_doc())


class _Draft(NamedTuple):
    """Pre-id coordinate draft used during generation and merge."""

    kind: str
    atom_maps: tuple[int, ...]
    role: str
    event_ids: tuple[str, ...]
    origin: OriginRecord


# ---------------------------------------------------------------------------
# Public constraint-rank helper (design §6.1: guards count; monitors never).
# ---------------------------------------------------------------------------
def constraint_rank(n_drivers: int, n_guards: int) -> int:
    """Return the geometric constraint rank of a candidate role assignment.

    Design §6.1: every λ-varying quantity counts as a driver; static guards
    also count toward rank and over-constraint checks; monitors never count.
    """
    if n_drivers < 0 or n_guards < 0:
        raise ValueError("CONSTRAINT_RANK_NEGATIVE: drivers and guards must be >= 0")
    return n_drivers + n_guards


# ---------------------------------------------------------------------------
# Public builder.
# ---------------------------------------------------------------------------
def build_coordinate_pool(
    route_result: RouteResult,
    edit_graph: ReactionEditGraph,
    context: EndpointContext,
    coupling: EventCouplingGraph,
    bundle: EndpointGraphBundle,
    materials: EndpointMaterials | Mapping[str, Any] | None = None,
    *, connectivity_only: bool = False,
) -> CoordinatePool:
    """Build the coordinate pool + driver-set candidates of one reaction.

    Positional API per todo 12.  ``materials`` is optional and only required
    for conformational-difference ``D`` generation (endpoint geometry
    comparison); it accepts a typed :class:`EndpointMaterials`, a
    ``{"r": ..., "p": ...}`` mapping of map→xyz, or ``None``.
    """
    elements = _elements_by_map(bundle)
    membership = coupling.membership_map()
    events = tuple(coupling.events)
    events_by_id = {event.event_id: event for event in events}
    edit_pairs = tuple(
        sorted((int(edit.pair[0]), int(edit.pair[1])) for edit in edit_graph.edits)
    )
    kind_by_pair = {
        (int(edit.pair[0]), int(edit.pair[1])): str(edit.edit_kind)
        for edit in edit_graph.edits
    }
    normalized_materials = _normalize_materials(materials)

    drafts: list[_Draft] = []
    _emit_edit_coordinates(drafts, edit_graph, membership)
    _emit_hydrogen_coordinates(drafts, coupling, elements, membership)
    _emit_ring_coordinates(drafts, events, bundle)
    _emit_conformational_coordinates(drafts, bundle, elements, normalized_materials)
    if connectivity_only:
        active_pairs = {e.pair for e in edit_graph.edits if e.edit_kind in {"formed", "broken"}}
        # Geometry suggestions remain passive; guards require a frozen promotion.
        drafts = [d._replace(role=ROLE_MONITOR)
                  if d.role == ROLE_DRIVER and (d.kind != KIND_B or tuple(sorted(d.atom_maps)) not in active_pairs)
                  else d for d in drafts if d.origin.edit_kind != "order_changed"]
    records = _merge_and_assign_ids(drafts)

    candidates = _build_driver_candidates(
        route_result,
        records,
        events,
        events_by_id,
        membership,
        edit_graph,
        context,
        edit_pairs,
        kind_by_pair,
        elements,
    )

    pool_pairs = _pairs_covered_by_roles(records, {ROLE_DRIVER, ROLE_MONITOR})
    all_edits_monitored = set(edit_pairs) <= pool_pairs if edit_pairs else True
    if candidates and not all(c.edits_monitored for c in candidates):
        all_edits_monitored = False

    return CoordinatePool(
        schema_version=SCHEMA_COORDINATE_POOL,
        reaction_id=route_result.reaction_id,
        coordinates=records,
        driver_candidates=candidates,
        all_edits_monitored=all_edits_monitored,
    )


# ---------------------------------------------------------------------------
# Materials boundary (parse-don't-validate; local because todo-5 materials
# schema requires an element field and this pool only needs coordinates).
# ---------------------------------------------------------------------------
def _normalize_materials(
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
# Coordinate emission (design §6.2 generation rules).
# ---------------------------------------------------------------------------
def _emit_edit_coordinates(
    drafts: list[_Draft],
    edit_graph: ReactionEditGraph,
    membership: Mapping[tuple[int, int], tuple[str, ...]],
) -> None:
    """F/B edits → B driver-nominated; O edits → B monitor-only by default."""
    for edit in edit_graph.edits:
        pair = (int(edit.pair[0]), int(edit.pair[1]))
        if pair[0] > pair[1]:
            pair = (pair[1], pair[0])
        event_ids = tuple(sorted(membership.get(pair, ())))
        kind = str(edit.edit_kind)
        origin = OriginRecord(source_kind=ORIGIN_EDIT, edit_pair=pair, edit_kind=kind)
        if kind == "order_changed":
            drafts.append(
                _Draft(
                    KIND_B,
                    pair,
                    ROLE_MONITOR,
                    event_ids,
                    OriginRecord(
                        source_kind=ORIGIN_EDIT,
                        edit_pair=pair,
                        edit_kind=kind,
                        note=NOTE_ORDER_CHANGE_MONITOR,
                    ),
                )
            )
            continue
        # formed / broken: bond distance is the natural geometric driver.
        drafts.append(_Draft(KIND_B, pair, ROLE_DRIVER, event_ids, origin))
        # Target-test companion: final bond distance decides completion (todo 22).
        drafts.append(_Draft(KIND_B, pair, ROLE_TARGET_TEST, event_ids, origin))


def _emit_hydrogen_coordinates(
    drafts: list[_Draft],
    coupling: EventCouplingGraph,
    elements: Mapping[int, str],
    membership: Mapping[tuple[int, int], tuple[str, ...]],
) -> None:
    """H events → separate partner distances + ∠DHA + optional D–A.

    Hard rule (design §6.2 / todo 14 failure scenario): donor-H and acceptor-H
    stay **two separate B records**; a single difference coordinate
    ``B = d1 − d2`` is never generated.
    """
    h_events = tuple(
        event
        for event in coupling.events
        if event.event_type in (EVENT_H_TRANSFER, EVENT_H2)
    )
    for change in coupling.hydrogen_events:
        kind = str(change["kind"])
        h_atom = int(change["h"])
        src_raw = change["from"]
        dst_raw = change["to"]
        src = None if src_raw is None else int(src_raw)
        dst = None if dst_raw is None else int(dst_raw)
        event_ids = tuple(
            sorted(
                event.event_id
                for event in h_events
                if h_atom in event.support_atom_maps
            )
        )

        if kind in HH_EVENT_KINDS:
            if src is not None and str(change["from_state"]) == "hh":
                pair = _sorted_pair(h_atom, src)
                drafts.append(
                    _Draft(
                        KIND_B,
                        pair,
                        ROLE_DRIVER,
                        _pair_events(pair, membership, event_ids),
                        OriginRecord(
                            source_kind=ORIGIN_HYDROGEN_EVENT,
                            event_id=event_ids[0] if event_ids else None,
                            hydrogen_map=h_atom,
                            partner_map=src,
                            partner_role=PARTNER_DONOR,
                            geometry_kind=GEOM_HH,
                        ),
                    )
                )
            if dst is not None and str(change["to_state"]) == "hh":
                pair = _sorted_pair(h_atom, dst)
                drafts.append(
                    _Draft(
                        KIND_B,
                        pair,
                        ROLE_DRIVER,
                        _pair_events(pair, membership, event_ids),
                        OriginRecord(
                            source_kind=ORIGIN_HYDROGEN_EVENT,
                            event_id=event_ids[0] if event_ids else None,
                            hydrogen_map=h_atom,
                            partner_map=dst,
                            partner_role=PARTNER_ACCEPTOR,
                            geometry_kind=GEOM_HH,
                        ),
                    )
                )
            for partner, state_key, role in (
                (src, "from_state", PARTNER_DONOR),
                (dst, "to_state", PARTNER_ACCEPTOR),
            ):
                if partner is None or str(change[state_key]) == "hh":
                    continue
                pair = _sorted_pair(h_atom, partner)
                drafts.append(
                    _Draft(
                        KIND_B,
                        pair,
                        ROLE_DRIVER,
                        _pair_events(pair, membership, event_ids),
                        OriginRecord(
                            source_kind=ORIGIN_HYDROGEN_EVENT,
                            event_id=event_ids[0] if event_ids else None,
                            hydrogen_map=h_atom,
                            partner_map=partner,
                            partner_role=role,
                            geometry_kind=GEOM_PARTNER,
                        ),
                    )
                )
            continue

        # transfer / release / capture: two-partner geometry.
        if src is not None:
            pair = _sorted_pair(h_atom, src)
            drafts.append(
                _Draft(
                    KIND_B,
                    pair,
                    ROLE_DRIVER,
                    _pair_events(pair, membership, event_ids),
                    OriginRecord(
                        source_kind=ORIGIN_HYDROGEN_EVENT,
                        event_id=event_ids[0] if event_ids else None,
                        hydrogen_map=h_atom,
                        partner_map=src,
                        partner_role=PARTNER_DONOR,
                        geometry_kind=GEOM_PARTNER,
                    ),
                )
            )
        if dst is not None:
            pair = _sorted_pair(h_atom, dst)
            drafts.append(
                _Draft(
                    KIND_B,
                    pair,
                    ROLE_DRIVER,
                    _pair_events(pair, membership, event_ids),
                    OriginRecord(
                        source_kind=ORIGIN_HYDROGEN_EVENT,
                        event_id=event_ids[0] if event_ids else None,
                        hydrogen_map=h_atom,
                        partner_map=dst,
                        partner_role=PARTNER_ACCEPTOR,
                        geometry_kind=GEOM_PARTNER,
                    ),
                )
            )
            # Arrival test: the P-bound partner distance decides H transfer.
            drafts.append(
                _Draft(
                    KIND_B,
                    pair,
                    ROLE_TARGET_TEST,
                    _pair_events(pair, membership, event_ids),
                    OriginRecord(
                        source_kind=ORIGIN_HYDROGEN_EVENT,
                        event_id=event_ids[0] if event_ids else None,
                        hydrogen_map=h_atom,
                        partner_map=dst,
                        partner_role=PARTNER_ACCEPTOR,
                        geometry_kind=GEOM_PARTNER,
                    ),
                )
            )
        if (
            src is not None
            and dst is not None
            and elements.get(src, _ELEMENT_H) != _ELEMENT_H
            and elements.get(dst, _ELEMENT_H) != _ELEMENT_H
        ):
            # ∠DHA angle + optional D–A distance (still separate records).
            drafts.append(
                _Draft(
                    KIND_A,
                    (src, h_atom, dst),
                    ROLE_DRIVER,
                    event_ids,
                    OriginRecord(
                        source_kind=ORIGIN_HYDROGEN_EVENT,
                        event_id=event_ids[0] if event_ids else None,
                        hydrogen_map=h_atom,
                        partner_map=dst,
                        partner_role=PARTNER_ACCEPTOR,
                        geometry_kind=GEOM_ANGLE_DHA,
                    ),
                )
            )
            da_pair = _sorted_pair(src, dst)
            drafts.append(
                _Draft(
                    KIND_B,
                    da_pair,
                    ROLE_DRIVER,
                    event_ids,
                    OriginRecord(
                        source_kind=ORIGIN_HYDROGEN_EVENT,
                        event_id=event_ids[0] if event_ids else None,
                        hydrogen_map=h_atom,
                        partner_map=dst,
                        geometry_kind=GEOM_DISTANCE_DA,
                    ),
                )
            )


def _pair_events(
    pair: tuple[int, int],
    membership: Mapping[tuple[int, int], tuple[str, ...]],
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    """Return owning event ids of an edit pair, else the H-event fallback."""
    owners = membership.get(pair)
    if owners:
        return tuple(sorted(owners))
    return fallback


def _emit_ring_coordinates(
    drafts: list[_Draft],
    events: Sequence[EventNode],
    bundle: EndpointGraphBundle,
) -> None:
    """Ring closure / attack geometry → local A or D as appropriate."""
    r_adj = _adjacency(bundle)
    p_adj = _adjacency(bundle, side="p")
    genuine = tuple(
        event
        for event in events
        if event.event_type == EVENT_RING and _is_genuine_ring_event(event)
    )
    for event in genuine:
        pairs = tuple(sorted(event.edit_pairs))
        if not pairs:
            continue
        chosen = _preferred_ring_pair(pairs, bundle)
        if chosen is None:
            continue
        pair, edit_kind = chosen
        adj = r_adj if edit_kind == "formed" else p_adj
        first, second = pair
        nbr_first = sorted(x for x in adj.get(first, ()) if x != second)
        nbr_second = sorted(x for x in adj.get(second, ()) if x != first)
        if nbr_first and nbr_second:
            drafts.append(
                _Draft(
                    KIND_D,
                    (nbr_first[0], first, second, nbr_second[0]),
                    ROLE_DRIVER,
                    (event.event_id,),
                    OriginRecord(
                        source_kind=ORIGIN_RING_GEOMETRY,
                        event_id=event.event_id,
                        edit_pair=pair,
                        geometry_kind=GEOM_ATTACK_TORSION,
                    ),
                )
            )
        elif nbr_first:
            drafts.append(
                _Draft(
                    KIND_A,
                    (nbr_first[0], first, second),
                    ROLE_DRIVER,
                    (event.event_id,),
                    OriginRecord(
                        source_kind=ORIGIN_RING_GEOMETRY,
                        event_id=event.event_id,
                        edit_pair=pair,
                        geometry_kind=GEOM_ATTACK_ANGLE,
                    ),
                )
            )
        elif nbr_second:
            drafts.append(
                _Draft(
                    KIND_A,
                    (first, second, nbr_second[0]),
                    ROLE_DRIVER,
                    (event.event_id,),
                    OriginRecord(
                        source_kind=ORIGIN_RING_GEOMETRY,
                        event_id=event.event_id,
                        edit_pair=pair,
                        geometry_kind=GEOM_ATTACK_ANGLE,
                    ),
                )
            )


def _preferred_ring_pair(
    pairs: Sequence[tuple[int, int]], bundle: EndpointGraphBundle
) -> tuple[tuple[int, int], str] | None:
    """Pick the closure/opening pair that carries attack geometry."""
    kind_by_pair = _edit_kinds_from_graphs(bundle)
    for pair in pairs:
        kind = kind_by_pair.get(pair)
        if kind in ("formed", "broken"):
            return pair, kind
    return None


def _emit_conformational_coordinates(
    drafts: list[_Draft],
    bundle: EndpointGraphBundle,
    elements: Mapping[int, str],
    materials: EndpointMaterials | None,
) -> None:
    """Conformational differences → D from endpoint geometry comparison."""
    if materials is None:
        return
    if not materials.r_coordinates or not materials.p_coordinates:
        return
    torsions = _heavy_torsions(bundle, elements)
    scored: list[tuple[float, tuple[int, int, int, int], float, float]] = []
    for torsion in torsions:
        r_value = _dihedral_at(materials.r_coordinates, torsion)
        p_value = _dihedral_at(materials.p_coordinates, torsion)
        if r_value is None or p_value is None:
            continue
        delta = _angle_delta_deg(r_value, p_value)
        if delta >= CONFORMATIONAL_DELTA_MIN_DEG:
            scored.append((delta, torsion, r_value, p_value))
    scored.sort(key=lambda row: (-row[0], row[1]))
    for _delta, torsion, r_value, p_value in scored[:MAX_CONFORMATIONAL_DRIVERS]:
        origin = OriginRecord(
            source_kind=ORIGIN_CONFORMATIONAL,
            geometry_kind="torsion_difference",
            r_value=round(r_value, 6),
            p_value=round(p_value, 6),
        )
        drafts.append(_Draft(KIND_D, torsion, ROLE_DRIVER, (), origin))
        drafts.append(_Draft(KIND_D, torsion, ROLE_TARGET_TEST, (), origin))


# ---------------------------------------------------------------------------
# Merge + deterministic id assignment.
# ---------------------------------------------------------------------------
def _merge_and_assign_ids(drafts: Sequence[_Draft]) -> tuple[CoordinateRecord, ...]:
    """Merge duplicate (kind, maps, role) drafts and assign stable ids."""
    merged: dict[tuple[str, tuple[int, ...], str], _Draft] = {}
    for draft in drafts:
        key = (draft.kind, draft.atom_maps, draft.role)
        previous = merged.get(key)
        if previous is None:
            merged[key] = draft
            continue
        event_ids = tuple(sorted(set(previous.event_ids) | set(draft.event_ids)))
        origin = (
            previous.origin
            if _origin_rank(previous.origin) <= _origin_rank(draft.origin)
            else draft.origin
        )
        merged[key] = _Draft(draft.kind, draft.atom_maps, draft.role, event_ids, origin)
    ordered = sorted(
        merged.values(),
        key=lambda draft: (
            draft.atom_maps,
            _KIND_RANK[draft.kind],
            _ROLE_RANK[draft.role],
            _origin_sort_key(draft.origin),
        ),
    )
    return tuple(
        CoordinateRecord(
            coordinate_id=f"coord-{index:04d}",
            role=draft.role,
            kind=draft.kind,
            atom_maps=draft.atom_maps,
            units=KIND_UNITS[draft.kind],
            event_ids=draft.event_ids,
            origin=draft.origin,
        )
        for index, draft in enumerate(ordered, start=1)
    )


def _origin_rank(origin: OriginRecord) -> int:
    return _ORIGIN_RANK.get(origin.source_kind, len(_ORIGIN_RANK))


def _origin_sort_key(origin: OriginRecord) -> tuple[Any, ...]:
    return (
        origin.source_kind,
        origin.edit_pair if origin.edit_pair is not None else (),
        origin.edit_kind if origin.edit_kind is not None else "",
        origin.event_id if origin.event_id is not None else "",
        origin.hydrogen_map if origin.hydrogen_map is not None else 0,
        origin.partner_map if origin.partner_map is not None else 0,
        origin.partner_role if origin.partner_role is not None else "",
        origin.geometry_kind if origin.geometry_kind is not None else "",
        origin.note if origin.note is not None else "",
    )


# ---------------------------------------------------------------------------
# Driver-set candidates + event coverage (design §6.2).
# ---------------------------------------------------------------------------
def _build_driver_candidates(
    route_result: RouteResult,
    records: tuple[CoordinateRecord, ...],
    events: Sequence[EventNode],
    events_by_id: Mapping[str, EventNode],
    membership: Mapping[tuple[int, int], tuple[str, ...]],
    edit_graph: ReactionEditGraph,
    context: EndpointContext,
    edit_pairs: tuple[tuple[int, int], ...],
    kind_by_pair: Mapping[tuple[int, int], str],
    elements: Mapping[int, str],
) -> tuple[DriverSetCandidate, ...]:
    """Enumerate finite 1–3 driver sets per route candidate with coverage."""
    driver_coords = [record for record in records if record.role == ROLE_DRIVER]
    if route_result.outcome in ("typed_rejection", "special_domain_exit"):
        route_candidates: tuple[StrategyCandidate, ...] = ()
    else:
        route_candidates = route_result.strategy_candidates

    seen: dict[frozenset[str], DriverSetCandidate] = {}
    out: list[DriverSetCandidate] = []
    for route_candidate in route_candidates:
        component_events = set(route_candidate.component_events)
        if component_events:
            relevant = [
                record
                for record in driver_coords
                if set(record.event_ids) & component_events
            ]
        else:
            # No component events (stereo / pure-conformational branch):
            # every driver-nominated coordinate is structurally relevant.
            relevant = list(driver_coords)
        if not relevant:
            continue
        planned = int(route_candidate.n_drivers_planned)
        if planned <= 0:
            continue
        max_size = max(1, min(MAX_SCAN_DRIVERS, planned))
        combos = _structural_combinations(relevant, max_size)
        for combo in combos:
            driver_ids = tuple(sorted(record.coordinate_id for record in combo))
            key = frozenset(driver_ids)
            if key in seen:
                continue
            coverage = _compute_event_coverage(
                driver_ids,
                records,
                events,
                events_by_id,
                membership,
                edit_graph,
                context,
                elements,
            )
            has_uncovered = any(
                row.coverage == COVERAGE_UNCOVERED for row in coverage
            )
            n_drivers = len(driver_ids)
            guards: tuple[str, ...] = ()
            rank = constraint_rank(n_drivers, len(guards))
            if has_uncovered:
                reason: str | None = REASON_UNCOVERED_EVENT
            elif n_drivers < 1:
                reason = REASON_NO_DRIVER_COORDINATES
            elif route_candidate.review_required:
                reason = REASON_ROUTE_REVIEW_REQUIRED
            else:
                reason = None
            driver_set = set(driver_ids)
            monitor_ids = tuple(
                sorted(
                    record.coordinate_id
                    for record in records
                    if record.role == ROLE_MONITOR
                    or (record.role == ROLE_DRIVER and record.coordinate_id not in driver_set)
                )
            )
            target_test_ids = tuple(
                sorted(
                    record.coordinate_id
                    for record in records
                    if record.role == ROLE_TARGET_TEST
                )
            )
            monitored_pairs = _pairs_covered_by_roles(
                records, {ROLE_DRIVER, ROLE_MONITOR}, restrict=driver_set | set(monitor_ids)
            )
            candidate = DriverSetCandidate(
                candidate_id="cand:" + "+".join(driver_ids),
                driver_coordinate_ids=driver_ids,
                monitor_coordinate_ids=monitor_ids,
                guard_coordinate_ids=guards,
                target_test_coordinate_ids=target_test_ids,
                event_coverage=coverage,
                has_uncovered_event=has_uncovered,
                default_automatic=reason is None,
                non_default_reason=reason,
                route_strategy_id=str(route_candidate.strategy_id),
                route_review_required=bool(route_candidate.review_required),
                n_drivers=n_drivers,
                constraint_rank=rank,
                edits_monitored=set(edit_pairs) <= monitored_pairs if edit_pairs else True,
            )
            seen[key] = candidate
            out.append(candidate)
    out.sort(key=lambda candidate: (candidate.driver_coordinate_ids, candidate.candidate_id))
    return tuple(out)


def _structural_combinations(
    coords: Sequence[CoordinateRecord], max_size: int
) -> list[tuple[CoordinateRecord, ...]]:
    """Return structurally meaningful 1–max_size driver combinations."""
    singles = [(record,) for record in coords]
    pairs: list[tuple[CoordinateRecord, ...]] = []
    for i, left in enumerate(coords):
        for right in coords[i + 1 :]:
            if _coords_linked(left, right):
                pairs.append((left, right))
    triples: list[tuple[CoordinateRecord, ...]] = []
    if max_size >= 3:
        for i, first in enumerate(coords):
            for j in range(i + 1, len(coords)):
                second = coords[j]
                for k in range(j + 1, len(coords)):
                    third = coords[k]
                    if _triple_connected(first, second, third):
                        triples.append((first, second, third))
    return singles + pairs[:COMBINATION_PAIR_CAP] + triples[:COMBINATION_TRIPLE_CAP]


def _coords_linked(left: CoordinateRecord, right: CoordinateRecord) -> bool:
    """True when two coordinates share an event or an atom map."""
    if set(left.event_ids) & set(right.event_ids):
        return True
    return bool(set(left.atom_maps) & set(right.atom_maps))


def _triple_connected(
    first: CoordinateRecord, second: CoordinateRecord, third: CoordinateRecord
) -> bool:
    """True when the three coordinates form a connected link graph."""
    links = (
        _coords_linked(first, second),
        _coords_linked(first, third),
        _coords_linked(second, third),
    )
    return sum(links) >= 2


def _compute_event_coverage(
    driver_ids: tuple[str, ...],
    records: Sequence[CoordinateRecord],
    events: Sequence[EventNode],
    events_by_id: Mapping[str, EventNode],
    membership: Mapping[tuple[int, int], tuple[str, ...]],
    edit_graph: ReactionEditGraph,
    context: EndpointContext,
    elements: Mapping[int, str],
) -> tuple[EventCoverage, ...]:
    """Compute per-driver-set event coverage (design §6.2 vocabulary)."""
    by_id = {record.coordinate_id: record for record in records}
    driver_set = set(driver_ids)
    drivers = [by_id[cid] for cid in driver_ids]
    monitors = [
        record
        for record in records
        if record.role == ROLE_MONITOR
        or (record.role == ROLE_DRIVER and record.coordinate_id not in driver_set)
    ]
    direct_event_ids = {event_id for record in drivers for event_id in record.event_ids}
    rows: list[EventCoverage] = []
    for event in sorted(events, key=lambda node: node.event_id):
        driver_refs = tuple(
            sorted(
                record.coordinate_id
                for record in drivers
                if event.event_id in record.event_ids
            )
        )
        monitor_refs = tuple(
            sorted(
                record.coordinate_id
                for record in monitors
                if event.event_id in record.event_ids
            )
        )
        if driver_refs:
            rows.append(
                EventCoverage(
                    event_id=event.event_id,
                    coverage=COVERAGE_DIRECT,
                    driver_coordinate_ids=driver_refs,
                    monitor_coordinate_ids=monitor_refs,
                    coupling_evidence=(),
                )
            )
            continue
        if not monitor_refs:
            rows.append(
                EventCoverage(
                    event_id=event.event_id,
                    coverage=COVERAGE_UNCOVERED,
                    driver_coordinate_ids=(),
                    monitor_coordinate_ids=(),
                    coupling_evidence=(),
                )
            )
            continue
        evidence: list[str] = []
        for driven_id in sorted(direct_event_ids):
            driven_event = events_by_id.get(driven_id)
            if driven_event is None:
                continue
            evidence.extend(
                _structural_links(
                    event,
                    driven_event,
                    membership=membership,
                    edit_graph=edit_graph,
                    context=context,
                    elements=elements,
                )
            )
        unique_evidence = tuple(sorted(set(evidence)))
        rows.append(
            EventCoverage(
                event_id=event.event_id,
                coverage=(
                    COVERAGE_COUPLED_MONITOR
                    if unique_evidence
                    else COVERAGE_UNCOVERED
                ),
                driver_coordinate_ids=(),
                monitor_coordinate_ids=monitor_refs,
                coupling_evidence=unique_evidence,
            )
        )
    return tuple(rows)


def _regions_of(
    pairs: Sequence[tuple[int, int]],
    region_by_pair: Mapping[tuple[int, int], str | None],
) -> set[str]:
    """Return the non-None aromatic region ids of one event's edit pairs."""
    regions: set[str] = set()
    for pair in pairs:
        region = region_by_pair.get(pair)
        if region is not None:
            regions.add(region)
    return regions


def _structural_links(
    event: EventNode,
    driven: EventNode,
    *,
    membership: Mapping[tuple[int, int], tuple[str, ...]],
    edit_graph: ReactionEditGraph,
    context: EndpointContext,
    elements: Mapping[int, str],
) -> tuple[str, ...]:
    """Return typed coupling evidence between two events (design §6.2)."""
    evidence: list[str] = []
    shared_atoms = sorted(set(event.support_atom_maps) & set(driven.support_atom_maps))
    if shared_atoms:
        preview = ",".join(str(atom) for atom in shared_atoms[:3])
        evidence.append(f"{EVID_SHARED_ATOM}:{preview}")
    shared_hydrogens = [
        atom for atom in shared_atoms if elements.get(atom, _ELEMENT_H) == _ELEMENT_H
    ]
    if shared_hydrogens:
        evidence.append(f"{EVID_SAME_HYDROGEN}:{shared_hydrogens[0]}")
    region_by_pair = {
        (int(edit.pair[0]), int(edit.pair[1])): edit.aromatic_region
        for edit in edit_graph.edits
    }
    event_regions = _regions_of(event.edit_pairs, region_by_pair)
    driven_regions = _regions_of(driven.edit_pairs, region_by_pair)
    shared_regions = sorted(event_regions & driven_regions)
    if shared_regions:
        evidence.append(f"{EVID_SAME_AROMATIC_REGION}:{shared_regions[0]}")
    ring_groups_by_pair = {
        (int(group_edit[0]), int(group_edit[1])): group.group_id
        for group in context.ring_groups
        for group_edit in group.edit_pairs
    }
    event_groups = {
        ring_groups_by_pair[pair]
        for pair in event.edit_pairs
        if pair in ring_groups_by_pair
    }
    driven_groups = {
        ring_groups_by_pair[pair]
        for pair in driven.edit_pairs
        if pair in ring_groups_by_pair
    }
    shared_groups = sorted(event_groups & driven_groups)
    if shared_groups:
        evidence.append(f"{EVID_SAME_RING_REGION}:{shared_groups[0]}")
    for pair, owners in membership.items():
        if event.event_id in owners and driven.event_id in owners:
            evidence.append(f"{EVID_SHARED_EDIT}:{pair[0]}-{pair[1]}")
            break
    return tuple(evidence)


def _pairs_covered_by_roles(
    records: Sequence[CoordinateRecord],
    roles: set[str],
    restrict: set[str] | None = None,
) -> set[tuple[int, int]]:
    """Return edit-like pairs covered by coordinates of the given roles."""
    covered: set[tuple[int, int]] = set()
    for record in records:
        if record.role not in roles:
            continue
        if restrict is not None and record.coordinate_id not in restrict:
            continue
        if record.origin.edit_pair is not None:
            covered.add(record.origin.edit_pair)
        elif record.kind == KIND_B and len(record.atom_maps) == 2:
            covered.add(_sorted_pair(record.atom_maps[0], record.atom_maps[1]))
    return covered


# ---------------------------------------------------------------------------
# Graph / geometry helpers.
# ---------------------------------------------------------------------------
def _elements_by_map(bundle: EndpointGraphBundle) -> dict[int, str]:
    elements = {int(node.map_id): str(node.element) for node in bundle.r_graph.nodes}
    for node in bundle.p_graph.nodes:
        elements.setdefault(int(node.map_id), str(node.element))
    return elements


def _edit_kinds_from_graphs(bundle: EndpointGraphBundle) -> dict[tuple[int, int], str]:
    """Re-derive F/B/O kinds from endpoint graphs (ring pair classification)."""
    r_orders = _side_orders(bundle.r_graph)
    p_orders = _side_orders(bundle.p_graph)
    pairs = set(r_orders) | set(p_orders)
    kinds: dict[tuple[int, int], str] = {}
    for pair in pairs:
        r_order = r_orders.get(pair)
        p_order = p_orders.get(pair)
        if r_order is None and p_order is not None:
            kinds[pair] = "formed"
        elif r_order is not None and p_order is None:
            kinds[pair] = "broken"
        elif r_order is not None and p_order is not None and r_order != p_order:
            kinds[pair] = "order_changed"
    return kinds


def _side_orders(graph: Any) -> dict[tuple[int, int], float]:
    orders: dict[tuple[int, int], float] = {}
    for edge in graph.edges:
        pair = _sorted_pair(int(edge.map_a), int(edge.map_b))
        order = 1.5 if edge.aromatic else float(edge.bond_order)
        orders[pair] = min(orders.get(pair, order), order)
    return orders


def _adjacency(bundle: EndpointGraphBundle, side: str = "r") -> dict[int, list[int]]:
    graph = bundle.r_graph if side == "r" else bundle.p_graph
    adjacency: dict[int, list[int]] = {}
    for edge in graph.edges:
        a, b = int(edge.map_a), int(edge.map_b)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    for members in adjacency.values():
        members.sort()
    return adjacency


def _heavy_torsions(
    bundle: EndpointGraphBundle, elements: Mapping[int, str]
) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate canonical heavy-atom torsions over the R∪P edge union."""
    edges: set[tuple[int, int]] = set()
    for graph in (bundle.r_graph, bundle.p_graph):
        for edge in graph.edges:
            pair = _sorted_pair(int(edge.map_a), int(edge.map_b))
            if (
                elements.get(pair[0], _ELEMENT_H) != _ELEMENT_H
                and elements.get(pair[1], _ELEMENT_H) != _ELEMENT_H
            ):
                edges.add(pair)
    adjacency: dict[int, list[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    for members in adjacency.values():
        members.sort()
    torsions: set[tuple[int, int, int, int]] = set()
    for b in sorted(adjacency):
        for c in adjacency[b]:
            if c <= b:
                continue
            for a in adjacency[b]:
                if a == c:
                    continue
                for d in adjacency[c]:
                    if d in (a, b, c):
                        continue
                    quad = (a, b, c, d)
                    torsions.add(min(quad, (d, c, b, a)))
    return tuple(sorted(torsions))


def _dihedral_at(
    coordinates: Mapping[int, tuple[float, float, float]],
    torsion: tuple[int, int, int, int],
) -> float | None:
    p0 = coordinates.get(torsion[0])
    p1 = coordinates.get(torsion[1])
    p2 = coordinates.get(torsion[2])
    p3 = coordinates.get(torsion[3])
    if p0 is None or p1 is None or p2 is None or p3 is None:
        return None
    return _dihedral(p0, p1, p2, p3)


def _dihedral(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    """Return the signed dihedral angle in degrees for four 3D points."""
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


def _angle_delta_deg(left: float, right: float) -> float:
    """Return the absolute wrapped angular difference in degrees."""
    return abs((left - right + 180.0) % 360.0 - 180.0)


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


def _sorted_pair(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


def _is_genuine_ring_event(event: EventNode) -> bool:
    """Distinguish real ring motifs from todo-8 singleton support-path noise."""
    if len(event.edit_pairs) >= 2:
        return True
    metadata = dict(event.metadata)
    return str(metadata.get("basis", "")) == _RING_BASIS_SHARED_REGION


__all__ = [
    "COVERAGE_COUPLED_MONITOR",
    "COVERAGE_DIRECT",
    "COVERAGE_UNCOVERED",
    "COMBINATION_PAIR_CAP",
    "COMBINATION_TRIPLE_CAP",
    "CONFORMATIONAL_DELTA_MIN_DEG",
    "EVID_SAME_AROMATIC_REGION",
    "EVID_SAME_HYDROGEN",
    "EVID_SAME_RING_REGION",
    "EVID_SHARED_ATOM",
    "EVID_SHARED_EDIT",
    "GEOM_ANGLE_DHA",
    "GEOM_ATTACK_ANGLE",
    "GEOM_ATTACK_TORSION",
    "GEOM_DISTANCE_DA",
    "GEOM_HH",
    "GEOM_PARTNER",
    "KINDS",
    "KIND_A",
    "KIND_B",
    "KIND_D",
    "KIND_UNITS",
    "MAX_CONFORMATIONAL_DRIVERS",
    "NOTE_ORDER_CHANGE_MONITOR",
    "ORIGIN_CONFORMATIONAL",
    "ORIGIN_EDIT",
    "ORIGIN_HYDROGEN_EVENT",
    "ORIGIN_KINDS",
    "ORIGIN_RING_GEOMETRY",
    "PARTNER_ACCEPTOR",
    "PARTNER_DONOR",
    "REASON_NO_DRIVER_COORDINATES",
    "REASON_ROUTE_REVIEW_REQUIRED",
    "REASON_UNCOVERED_EVENT",
    "ROLE_DRIVER",
    "ROLE_GUARD",
    "ROLE_MONITOR",
    "ROLE_TARGET_TEST",
    "ROLES",
    "SCHEMA_COORDINATE_POOL",
    "UNIT_ANGSTROM",
    "UNIT_DEGREE",
    "CoordinatePool",
    "CoordinateRecord",
    "DriverSetCandidate",
    "EndpointMaterials",
    "EventCoverage",
    "OriginRecord",
    "build_coordinate_pool",
    "constraint_rank",
]
