"""Direction & multi-component assembly policy (todo 15, design §7).

Implements the layered start-direction comparison and the multi-component
assembly policy of ``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``
§7, the direction-independent anchor analysis in
``docs/design/G1_v2_方向无关成键锚点与双向扫描策略.md`` §2/§4, and the demoted
hybrid five-rule anchors from
``docs/design/G1_v2_多成键与少成键方向对比.md`` §4.

Layered comparison (the ONLY selection mechanism, in order):

1. electronic-state / geometry review priority — an endpoint without complete
   review and a usable target definition can never start a scan;
2. chosen-driver start point already bonded + reduces fragment placement
   degrees-of-freedom — this layer outranks layer 3;
3. configuration / orientation / undriven-event coverage — preferred BEFORE
   F/B totals;
4. remaining ties keep BOTH directions (bidirectional candidates).

F>B "stretch from P" / B>F "stretch from R" and the hybrid five-rule edge
selection are **cheap priors and anchor heuristics only**: they are recorded
in ``anchor_reason`` / ``rule_trace`` for ``LOCAL_CONNECTIVITY`` and
``CONNECTIVITY_EXCHANGE`` strategy candidates and never act as a second
selection mechanism (guardrail: selection = :func:`compare_directions` only).

There is **no fixed "always P first"** (or R first).  Reverse curves are
never spliced without verification — this module never claims curve
equivalence (``SPLICE_POLICY`` below; splicing accounting is todo 28).

Multi-component assembly: rigid alignment reuses the G2 Kabsch primitives
(:func:`pes2ts_core.g2.endpoints.kabsch_transform` /
:func:`pes2ts_core.g2.endpoints.place_single`); finite approach
configurations translate driver-relevant fragments along the driver axis;
a nonbonded collision check gates readiness; spectator components keep their
stored placement and are flagged, never silently dropped.  Source files that
store components in independent physical frames are detected as typed
``ASSEMBLY_FRAME_MISMATCH`` records and never silently accepted.

Equivalence properties (locked by tests): template R/P swap and rigid
endpoint motions do not change equivalent strategy output modulo direction
labels; F=B ties yield bidirectional candidates; assembly failure yields
not-ready with a typed reason.
"""

# noqa: SIZE_OK -- plan-named todo-15 single module (layered direction
# comparison + demoted five-rule anchors + multi-component assembly policy);
# precedent: registry.py (todo 11), coordinate_pool.py (todo 12),
# geometry_feasibility.py (todo 13).

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.g2.endpoints import (
    ComponentGeometry,
    place_single,
)
from pes2ts_core.scan_strategy.contracts_v2 import DIRECTIONS, ENDPOINTS
from pes2ts_core.scan_strategy.registry import (
    STRATEGY_CONNECTIVITY_EXCHANGE,
    STRATEGY_LOCAL_CONNECTIVITY,
)
from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps

SCHEMA_DIRECTION_ASSEMBLY: Final[str] = "g1_direction_assembly_v1"

#: Reverse curves are never spliced without verification (todo 28 owns the
#: splicing accounting; this module never claims equivalence either way).
SPLICE_POLICY: Final[str] = "reverse_curves_never_spliced_without_verification"

#: The one and only direction-selection mechanism in this module.
SELECTION_MECHANISM: Final[str] = "layered_direction_comparison_v1"

# Layer identifiers recorded in ``AnchorReason.layer``.
LAYER_1: Final[str] = "L1"
LAYER_2: Final[str] = "L2"
LAYER_3: Final[str] = "L3"
LAYER_4: Final[str] = "L4"
LAYER_PRIOR: Final[str] = "prior"
LAYER_HYBRID: Final[str] = "hybrid_anchor"
LAYER_ASSEMBLY: Final[str] = "assembly"

# Typed anchor-reason codes.
ANCHOR_LAYER1_REVIEW_PRIORITY: Final[str] = "LAYER1_REVIEW_PRIORITY"
ANCHOR_DRIVER_BONDED_AT_START: Final[str] = "LAYER2_DRIVER_BONDED_AT_START"
ANCHOR_REDUCES_FRAGMENT_DOF: Final[str] = "LAYER2_REDUCES_FRAGMENT_DOF"
ANCHOR_CONFIGURATION_OK: Final[str] = "LAYER3_CONFIGURATION_OK"
ANCHOR_ORIENTATION_OK: Final[str] = "LAYER3_ORIENTATION_OK"
ANCHOR_UNDRIVEN_EVENT_COVERAGE_OK: Final[str] = "LAYER3_UNDRIVEN_EVENT_COVERAGE_OK"
ANCHOR_TIE_BIDIRECTIONAL: Final[str] = "LAYER4_TIE_BIDIRECTIONAL"
ANCHOR_ASSEMBLY_NOT_EVALUATED: Final[str] = "ASSEMBLY_NOT_EVALUATED"
PRIOR_FB_STRETCH_FROM_ANCHOR: Final[str] = "PRIOR_FB_STRETCH_FROM_ANCHOR"
PRIOR_FB_NONE: Final[str] = "PRIOR_FB_NONE"

# Demoted hybrid five-rule anchors (G1_v2_多成键与少成键方向对比.md §4).
HYBRID_RULE_TIE_GEOMETRY: Final[str] = "hybrid_rule_tie_geometry"
HYBRID_RULE_MORE_BOND_DEFAULT: Final[str] = "hybrid_rule_more_bond_default"
HYBRID_RULE_FEWER_BOND_EVALUATED: Final[str] = "hybrid_rule_fewer_bond_evaluated"
HYBRID_RULE_REPRESENTATIVE_OR_PATH: Final[str] = "hybrid_rule_representative_or_path"
HYBRID_RULE_MINIMAL_CONSTRAINTS: Final[str] = "hybrid_rule_minimal_constraints"
HYBRID_RULE_IDS: Final[tuple[str, ...]] = (
    HYBRID_RULE_TIE_GEOMETRY,
    HYBRID_RULE_MORE_BOND_DEFAULT,
    HYBRID_RULE_FEWER_BOND_EVALUATED,
    HYBRID_RULE_REPRESENTATIVE_OR_PATH,
    HYBRID_RULE_MINIMAL_CONSTRAINTS,
)
#: Strategies whose candidates carry the demoted five-rule anchors.
ANCHOR_APPLICABLE_STRATEGIES: Final[frozenset[str]] = frozenset({
    STRATEGY_LOCAL_CONNECTIVITY,
    STRATEGY_CONNECTIVITY_EXCHANGE,
})

# Typed assembly / readiness codes.
CODE_ASSEMBLY_FRAME_MISMATCH: Final[str] = "ASSEMBLY_FRAME_MISMATCH"
CODE_ASSEMBLY_COLLISION: Final[str] = "ASSEMBLY_COLLISION"
CODE_ASSEMBLY_FAILED: Final[str] = "ASSEMBLY_FAILED"
CODE_ASSEMBLY_READY: Final[str] = "ASSEMBLY_READY"
CODE_SPECTATOR_PRESERVED: Final[str] = "SPECTATOR_PRESERVED"
CODE_NO_USABLE_ENDPOINT: Final[str] = "NO_USABLE_ENDPOINT"

DEFAULT_COLLISION_MIN_DISTANCE: Final[float] = 0.8
DEFAULT_BOND_TOLERANCE: Final[float] = 0.45
#: Stored bonded distance may exceed radius_sum+bond_tolerance by this much
#: before the component is flagged as written across independent frames.
DEFAULT_FRAME_MISMATCH_TOLERANCE: Final[float] = 1.5
DEFAULT_APPROACH_TARGET_DISTANCES: Final[tuple[float, ...]] = (2.5, 3.0, 3.5)
DEFAULT_MIN_ANCHOR_MAPS: Final[int] = 3
DEFAULT_ANCHOR_TOLERANCE: Final[float] = 1e-3

Coordinate = tuple[float, float, float]
MapPair = tuple[int, int]

_START_ALIASES: Final[dict[str, str]] = {
    "R": "R", "r": "R", "reactant": "R",
    "P": "P", "p": "P", "product": "P",
}


class EditCountsSource(Protocol):
    """Structural view of a reaction edit graph's exclusive edit counts."""

    @property
    def edit_counts(self) -> Mapping[str, int]: ...


# ---------------------------------------------------------------------------
# Vocabulary records.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FBCounts:
    """Exclusive F/B/O edit counts (edit graph semantics, todo 7)."""

    formed: int
    broken: int
    order_changed: int

    @property
    def is_tie(self) -> bool:
        return self.formed == self.broken

    @property
    def lo(self) -> int:
        return min(self.formed, self.broken)

    @property
    def hi(self) -> int:
        return max(self.formed, self.broken)

    def to_doc(self) -> dict[str, JSONValue]:
        doc: dict[str, JSONValue] = {
            "formed": self.formed,
            "broken": self.broken,
            "order_changed": self.order_changed,
        }
        return doc


@dataclass(frozen=True, slots=True)
class AnchorReason:
    """One typed trace entry explaining a direction / anchor / assembly fact."""

    code: str
    layer: str
    endpoint: str | None = None
    detail: str = ""

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "code": self.code,
            "layer": self.layer,
            "endpoint": self.endpoint,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class EndpointProfile:
    """Per-endpoint evaluation inputs for the layered direction comparison.

    Layer 1 fields gate usability; layer 2 fields describe the chosen driver's
    start-point state; layer 3 fields describe configuration / orientation /
    undriven-event coverage.  ``endpoint`` ∈ {R, P}.
    """

    endpoint: str
    electronic_review_complete: bool = True
    geometry_review_complete: bool = True
    has_usable_target_definition: bool = True
    driver_bonded_at_start: bool = False
    reduces_fragment_dof: bool = False
    configuration_ok: bool = False
    orientation_ok: bool = False
    undriven_event_coverage_ok: bool = False

    def __post_init__(self) -> None:
        if self.endpoint not in ENDPOINTS:
            raise ValueError(
                f"ENDPOINT_INVALID: {self.endpoint!r}; expected one of {list(ENDPOINTS)}"
            )

    def review_ok(self) -> bool:
        """Electronic-state + geometry review completed for this endpoint."""
        return self.electronic_review_complete and self.geometry_review_complete

    def target_definition_ok(self) -> bool:
        return self.has_usable_target_definition

    def layer1_ok(self) -> bool:
        """Combined layer-1 usability (review complete AND target defined)."""
        return self.review_ok() and self.target_definition_ok()

    def layer2_score(self) -> int:
        return (1 if self.driver_bonded_at_start else 0) + (
            1 if self.reduces_fragment_dof else 0
        )

    def layer3_score(self) -> int:
        return (
            (1 if self.configuration_ok else 0)
            + (1 if self.orientation_ok else 0)
            + (1 if self.undriven_event_coverage_ok else 0)
        )


@dataclass(frozen=True, slots=True)
class DirectionComparison:
    """Outcome of the layered comparison (design §7)."""

    preferred: str | None
    decided_layer: int | None
    bidirectional: bool
    reasons: tuple[AnchorReason, ...]
    layer1_eligible: tuple[str, ...]

    @property
    def blocked_no_endpoint(self) -> bool:
        return not self.layer1_eligible

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "selection_mechanism": SELECTION_MECHANISM,
            "preferred": self.preferred,
            "decided_layer": self.decided_layer,
            "bidirectional": self.bidirectional,
            "layer1_eligible": list(self.layer1_eligible),
            "reasons": [reason.to_doc() for reason in self.reasons],
        }


@dataclass(frozen=True, slots=True)
class DriverBondSpec:
    """One driver bond nomination (map ids; B kind for cross-component checks)."""

    atom_maps: tuple[int, ...]
    kind: str = "B"
    driver_id: str | None = None


@dataclass(frozen=True, slots=True)
class DriverSetSpec:
    """A driver-set nomination with its route strategy (for anchor applicability)."""

    driver_set_id: str
    route_strategy_id: str | None = None
    driver_bonds: tuple[DriverBondSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentMaterial:
    """One source component: tag + map-keyed stored coordinates (+ elements)."""

    tag: str
    coordinates: Mapping[int, Coordinate]
    elements: Mapping[int, str] = field(default_factory=dict)
    contains_edit_atoms: bool = False


@dataclass(frozen=True, slots=True)
class SideMaterial:
    """All stored components of one endpoint side."""

    endpoint: str
    components: tuple[ComponentMaterial, ...]

    def __post_init__(self) -> None:
        if self.endpoint not in ENDPOINTS:
            raise ValueError(
                f"ENDPOINT_INVALID: {self.endpoint!r}; expected one of {list(ENDPOINTS)}"
            )


@dataclass(frozen=True, slots=True)
class PlacementRecord:
    """One rigid placement of a component into the assembly frame."""

    component_tag: str
    onto_tag: str
    n_shared: int
    basis: str
    rmsd_shared: float | None
    anchor_rank_ok: bool | None
    frame_ambiguous: bool

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "component_tag": self.component_tag,
            "onto_tag": self.onto_tag,
            "n_shared": self.n_shared,
            "basis": self.basis,
            "rmsd_shared": self.rmsd_shared,
            "anchor_rank_ok": self.anchor_rank_ok,
            "frame_ambiguous": self.frame_ambiguous,
        }


@dataclass(frozen=True, slots=True)
class ApproachConfiguration:
    """One finite approach configuration: translate a fragment along a driver axis."""

    configuration_id: str
    driver_set_id: str | None
    driver_atom_maps: tuple[int, ...]
    moved_component_tag: str
    fixed_component_tag: str
    moved_maps: tuple[int, ...]
    target_distance: float
    axis: Coordinate
    shift: float
    collision_ok: bool = False
    min_nonbonded_distance: float | None = None

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "configuration_id": self.configuration_id,
            "driver_set_id": self.driver_set_id,
            "driver_atom_maps": list(self.driver_atom_maps),
            "moved_component_tag": self.moved_component_tag,
            "fixed_component_tag": self.fixed_component_tag,
            "moved_maps": list(self.moved_maps),
            "target_distance": self.target_distance,
            "axis": list(self.axis),
            "shift": self.shift,
            "collision_ok": self.collision_ok,
            "min_nonbonded_distance": self.min_nonbonded_distance,
        }


@dataclass(frozen=True, slots=True)
class FrameMismatchRecord:
    """Typed misalignment detection: source components in independent frames."""

    component_tag: str
    code: str
    detail: str
    spectator_preserved: bool

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "component_tag": self.component_tag,
            "code": self.code,
            "detail": self.detail,
            "spectator_preserved": self.spectator_preserved,
        }


@dataclass(frozen=True, slots=True)
class AssemblyRecord:
    """Deterministic assembly outcome for one start side / direction."""

    assembly_id: str
    endpoint: str
    direction: str
    anchor_component_tag: str
    placements: tuple[PlacementRecord, ...]
    approach_configurations: tuple[ApproachConfiguration, ...]
    spectator_tags: tuple[str, ...]
    frame_mismatches: tuple[FrameMismatchRecord, ...]
    min_nonbonded_distance: float | None
    n_severe_contacts: int
    collision_ok: bool
    ready: bool
    failure_codes: tuple[str, ...]
    coordinates: Mapping[int, Coordinate]

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "assembly_id": self.assembly_id,
            "endpoint": self.endpoint,
            "direction": self.direction,
            "anchor_component_tag": self.anchor_component_tag,
            "placements": [p.to_doc() for p in self.placements],
            "approach_configurations": [c.to_doc() for c in self.approach_configurations],
            "spectator_tags": list(self.spectator_tags),
            "frame_mismatches": [m.to_doc() for m in self.frame_mismatches],
            "min_nonbonded_distance": self.min_nonbonded_distance,
            "n_severe_contacts": self.n_severe_contacts,
            "collision_ok": self.collision_ok,
            "ready": self.ready,
            "failure_codes": list(self.failure_codes),
            "coordinates": {str(k): list(v) for k, v in sorted(self.coordinates.items())},
        }


@dataclass(frozen=True, slots=True)
class DirectionCandidate:
    """One direction-annotated candidate (todo 17 consumes this shape)."""

    start_endpoint: str
    direction: str
    assembly_id: str | None
    anchor_reason: tuple[AnchorReason, ...]
    route_strategy_id: str | None
    driver_set_id: str | None
    rule_trace: tuple[str, ...]
    bidirectional: bool
    assembly: AssemblyRecord | None
    target_assembly: AssemblyRecord | None
    ready: bool
    readiness_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.start_endpoint not in ENDPOINTS:
            raise ValueError(
                f"ENDPOINT_INVALID: {self.start_endpoint!r}; "
                f"expected one of {list(ENDPOINTS)}"
            )
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"DIRECTION_INVALID: {self.direction!r}; expected one of {list(DIRECTIONS)}"
            )
        expected = "R_to_P" if self.start_endpoint == "R" else "P_to_R"
        if self.direction != expected:
            raise ValueError(
                f"DIRECTION_MISMATCH: direction {self.direction!r} must start from "
                f"start_endpoint {self.start_endpoint!r} (expected {expected!r})"
            )

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "start_endpoint": self.start_endpoint,
            "direction": self.direction,
            "assembly_id": self.assembly_id,
            "anchor_reason": [reason.to_doc() for reason in self.anchor_reason],
            "route_strategy_id": self.route_strategy_id,
            "driver_set_id": self.driver_set_id,
            "rule_trace": list(self.rule_trace),
            "bidirectional": self.bidirectional,
            "ready": self.ready,
            "readiness_blockers": list(self.readiness_blockers),
            "equivalence_claimed": False,
            "splice_policy": SPLICE_POLICY,
            "assembly": None if self.assembly is None else self.assembly.to_doc(),
            "target_assembly": (
                None if self.target_assembly is None else self.target_assembly.to_doc()
            ),
        }


@dataclass(frozen=True, slots=True)
class DirectionAssemblyInputs:
    """Bundle of read-only inputs for :func:`resolve_direction_assembly`."""

    reaction_id: str | None
    fb_counts: FBCounts
    r_profile: EndpointProfile
    p_profile: EndpointProfile
    driver_sets: tuple[DriverSetSpec, ...] = ()
    r_side: SideMaterial | None = None
    p_side: SideMaterial | None = None
    #: Bonded map pairs in the union of the R and P bond graphs (collision
    #: "nonbonded" = bonded in neither side).
    bonded_pairs: frozenset[MapPair] = frozenset()
    elements: Mapping[int, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DirectionAssemblyResult:
    """Full todo-15 output: comparison + ordered direction candidates."""

    schema_version: str
    reaction_id: str | None
    fb_counts: FBCounts
    comparison: DirectionComparison
    direction_candidates: tuple[DirectionCandidate, ...]

    def to_doc(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "reaction_id": self.reaction_id,
            "splice_policy": SPLICE_POLICY,
            "selection_mechanism": SELECTION_MECHANISM,
            "fb_counts": self.fb_counts.to_doc(),
            "comparison": self.comparison.to_doc(),
            "direction_candidates": [c.to_record() for c in self.direction_candidates],
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_doc())


@dataclass(frozen=True, slots=True)
class DirectionAssemblyPolicy:
    """Numeric policy for assembly + approach enumeration."""

    collision_min_distance: float
    bond_tolerance: float
    frame_mismatch_tolerance: float
    approach_target_distances: tuple[float, ...]
    min_anchor_maps: int
    anchor_tolerance: float


def policy_from_config(config: Mapping[str, Any] | None) -> DirectionAssemblyPolicy:
    """Parse policy values from a loaded config mapping (YAML-safe coercion).

    Missing keys fall back to the documented defaults.  Present-but-invalid
    values raise ``ValueError`` with a ``POLICY_INVALID:`` prefix.  Float
    values accept numeric strings (PyYAML 1.1 quirk on ``1.0e6``-style
    scalars).
    """
    section: Mapping[str, Any] = {}
    if config is not None:
        raw_section = config.get("scan_strategy")
        if isinstance(raw_section, Mapping):
            inner = raw_section.get("direction_assembly")
            if isinstance(inner, Mapping):
                section = inner
    g2: Mapping[str, Any] = {}
    validity: Mapping[str, Any] = {}
    if config is not None:
        raw_g2 = config.get("g2")
        if isinstance(raw_g2, Mapping):
            g2 = raw_g2
            raw_validity = g2.get("validity")
            if isinstance(raw_validity, Mapping):
                validity = raw_validity
    collision_raw = validity.get("collision_min_distance", DEFAULT_COLLISION_MIN_DISTANCE)
    collision = _float_value(collision_raw, "g2.validity.collision_min_distance")
    if collision <= 0:
        raise ValueError("POLICY_INVALID: collision_min_distance must be > 0")
    bond = _float_value(
        section.get("bond_tolerance", DEFAULT_BOND_TOLERANCE),
        "scan_strategy.direction_assembly.bond_tolerance",
    )
    frame_tol = _float_value(
        section.get("frame_mismatch_tolerance", DEFAULT_FRAME_MISMATCH_TOLERANCE),
        "scan_strategy.direction_assembly.frame_mismatch_tolerance",
    )
    if frame_tol < 0:
        raise ValueError("POLICY_INVALID: frame_mismatch_tolerance must be >= 0")
    targets_raw = section.get("approach_target_distances", DEFAULT_APPROACH_TARGET_DISTANCES)
    if not isinstance(targets_raw, (list, tuple)) or not targets_raw:
        raise ValueError(
            "POLICY_INVALID: approach_target_distances must be a non-empty list"
        )
    targets = tuple(
        _float_value(item, "scan_strategy.direction_assembly.approach_target_distances")
        for item in targets_raw
    )
    if any(t <= 0 for t in targets):
        raise ValueError("POLICY_INVALID: approach target distances must be > 0")
    min_anchor = section.get("min_anchor_maps", DEFAULT_MIN_ANCHOR_MAPS)
    if isinstance(min_anchor, bool) or not isinstance(min_anchor, int) or min_anchor < 1:
        raise ValueError("POLICY_INVALID: min_anchor_maps must be a positive integer")
    anchor_tol = _float_value(
        section.get("anchor_tolerance", DEFAULT_ANCHOR_TOLERANCE),
        "scan_strategy.direction_assembly.anchor_tolerance",
    )
    return DirectionAssemblyPolicy(
        collision_min_distance=collision,
        bond_tolerance=bond,
        frame_mismatch_tolerance=frame_tol,
        approach_target_distances=targets,
        min_anchor_maps=min_anchor,
        anchor_tolerance=anchor_tol,
    )


def _float_value(raw: Any, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise ValueError(f"POLICY_INVALID: {label} must be numeric")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"POLICY_INVALID: {label} must be numeric") from None
    if not math.isfinite(value):
        raise ValueError(f"POLICY_INVALID: {label} must be finite")
    return value


def fb_counts_from_edit_graph(edit_graph: EditCountsSource) -> FBCounts:
    """Extract exclusive F/B/O counts from a reaction edit graph (todo 7)."""
    counts = edit_graph.edit_counts
    return FBCounts(
        formed=int(counts.get("formed", 0)),
        broken=int(counts.get("broken", 0)),
        order_changed=int(counts.get("order_changed", 0)),
    )


# ---------------------------------------------------------------------------
# Demoted hybrid five-rule anchors (heuristic trace only).
# ---------------------------------------------------------------------------
def hybrid_rule_ids_for(fb: FBCounts) -> tuple[str, ...]:
    """Return the demoted five-rule anchor ids applicable to *fb*.

    These anchors come from the hybrid edge-selection analysis
    (G1_v2_多成键与少成键方向对比.md §4) and are recorded in ``rule_trace``
    for ``LOCAL_CONNECTIVITY`` / ``CONNECTIVITY_EXCHANGE`` candidates.  They
    are NOT a selection mechanism: :func:`compare_directions` decides, these
    only annotate.
    """
    if fb.is_tie:
        first = HYBRID_RULE_TIE_GEOMETRY
    elif fb.hi <= 3:
        first = HYBRID_RULE_MORE_BOND_DEFAULT
    elif 1 <= fb.lo <= 3:
        first = HYBRID_RULE_FEWER_BOND_EVALUATED
    else:
        first = HYBRID_RULE_REPRESENTATIVE_OR_PATH
    return (first, HYBRID_RULE_MINIMAL_CONSTRAINTS)


def hybrid_anchor_reasons(fb: FBCounts) -> tuple[AnchorReason, ...]:
    """Typed anchor-reason entries for the demoted five-rule anchors."""
    return tuple(
        AnchorReason(
            code=rule_id,
            layer=LAYER_HYBRID,
            endpoint=None,
            detail=(
                "demoted hybrid five-rule anchor (heuristic trace only; "
                "layered comparison remains the selection mechanism)"
            ),
        )
        for rule_id in hybrid_rule_ids_for(fb)
    )


def fb_prior_reason(fb: FBCounts) -> AnchorReason:
    """Cheap F/B prior entry — recorded, never a hard direction rule."""
    if fb.formed > fb.broken:
        return AnchorReason(
            code=PRIOR_FB_STRETCH_FROM_ANCHOR,
            layer=LAYER_PRIOR,
            endpoint="P",
            detail="F>B cheap prior: stretch from bonded-anchor P (never a hard rule)",
        )
    if fb.broken > fb.formed:
        return AnchorReason(
            code=PRIOR_FB_STRETCH_FROM_ANCHOR,
            layer=LAYER_PRIOR,
            endpoint="R",
            detail="B>F cheap prior: stretch from bonded-anchor R (never a hard rule)",
        )
    return AnchorReason(
        code=PRIOR_FB_NONE,
        layer=LAYER_PRIOR,
        endpoint=None,
        detail="F=B: bond-count prior cannot pick a side (tie goes to layers/tie rule)",
    )


# ---------------------------------------------------------------------------
# Layered direction comparison (design §7 — the only selection mechanism).
# ---------------------------------------------------------------------------
def compare_directions(
    r_profile: EndpointProfile, p_profile: EndpointProfile
) -> DirectionComparison:
    """Compare R vs P with the layered order of design §7.

    Layer 1 has two parts: (hard) BOTH sides must carry a usable target
    definition — otherwise no direction can start; (preference) the side
    with completed electronic/geometry review wins when exactly one side is
    reviewed.  Layer 2 (driver bonded at start + fragment DOF reduction)
    outranks layer 3 (configuration / orientation / undriven-event coverage);
    F/B totals are never consulted here.  Full ties keep both directions.
    """
    if r_profile.endpoint != "R" or p_profile.endpoint != "P":
        raise ValueError(
            "ENDPOINT_INVALID: compare_directions expects "
            "r_profile.endpoint='R' and p_profile.endpoint='P'"
        )
    if not (r_profile.target_definition_ok() and p_profile.target_definition_ok()):
        return DirectionComparison(
            preferred=None,
            decided_layer=1,
            bidirectional=False,
            reasons=(
                AnchorReason(
                    code=CODE_NO_USABLE_ENDPOINT,
                    layer=LAYER_1,
                    endpoint=None,
                    detail=(
                        "both sides must carry a usable target definition "
                        "(design §7 layer 1 hard gate); at least one side lacks it"
                    ),
                ),
            ),
            layer1_eligible=(),
        )
    r_review, p_review = r_profile.review_ok(), p_profile.review_ok()
    if r_review != p_review:
        winner = "R" if r_review else "P"
        return DirectionComparison(
            preferred=winner,
            decided_layer=1,
            bidirectional=False,
            reasons=(
                AnchorReason(
                    code=ANCHOR_LAYER1_REVIEW_PRIORITY,
                    layer=LAYER_1,
                    endpoint=winner,
                    detail=(
                        "completed electronic/geometry review priority "
                        "(design §7 layer 1 preference)"
                    ),
                ),
            ),
            layer1_eligible=(winner,),
        )
    eligible = ("R", "P")
    r2, p2 = r_profile.layer2_score(), p_profile.layer2_score()
    if r2 != p2:
        winner = "R" if r2 > p2 else "P"
        profile = r_profile if winner == "R" else p_profile
        reasons = []
        if profile.driver_bonded_at_start:
            reasons.append(
                AnchorReason(
                    code=ANCHOR_DRIVER_BONDED_AT_START,
                    layer=LAYER_2,
                    endpoint=winner,
                    detail="chosen driver is already bonded at this start point",
                )
            )
        if profile.reduces_fragment_dof:
            reasons.append(
                AnchorReason(
                    code=ANCHOR_REDUCES_FRAGMENT_DOF,
                    layer=LAYER_2,
                    endpoint=winner,
                    detail="start side reduces fragment placement degrees-of-freedom",
                )
            )
        reasons.append(
            AnchorReason(
                code=ANCHOR_DRIVER_BONDED_AT_START
                if profile.driver_bonded_at_start
                else ANCHOR_REDUCES_FRAGMENT_DOF,
                layer=LAYER_2,
                endpoint=winner,
                detail=f"layer-2 score {r2 if winner == 'R' else p2} vs "
                f"{p2 if winner == 'R' else r2} (layer 2 outranks layer 3)",
            )
        )
        return DirectionComparison(
            preferred=winner,
            decided_layer=2,
            bidirectional=False,
            reasons=tuple(reasons),
            layer1_eligible=eligible,
        )
    r3, p3 = r_profile.layer3_score(), p_profile.layer3_score()
    if r3 != p3:
        winner = "R" if r3 > p3 else "P"
        profile = r_profile if winner == "R" else p_profile
        reasons = []
        if profile.configuration_ok:
            reasons.append(
                AnchorReason(
                    code=ANCHOR_CONFIGURATION_OK,
                    layer=LAYER_3,
                    endpoint=winner,
                    detail="starting configuration quality preferred (before F/B totals)",
                )
            )
        if profile.orientation_ok:
            reasons.append(
                AnchorReason(
                    code=ANCHOR_ORIENTATION_OK,
                    layer=LAYER_3,
                    endpoint=winner,
                    detail="local attack orientation preferred (before F/B totals)",
                )
            )
        if profile.undriven_event_coverage_ok:
            reasons.append(
                AnchorReason(
                    code=ANCHOR_UNDRIVEN_EVENT_COVERAGE_OK,
                    layer=LAYER_3,
                    endpoint=winner,
                    detail="undriven-event coverage preferred (before F/B totals)",
                )
            )
        reasons.append(
            AnchorReason(
                code=ANCHOR_CONFIGURATION_OK
                if profile.configuration_ok
                else (
                    ANCHOR_ORIENTATION_OK
                    if profile.orientation_ok
                    else ANCHOR_UNDRIVEN_EVENT_COVERAGE_OK
                ),
                layer=LAYER_3,
                endpoint=winner,
                detail=f"layer-3 score {r3 if winner == 'R' else p3} vs "
                f"{p3 if winner == 'R' else r3}",
            )
        )
        return DirectionComparison(
            preferred=winner,
            decided_layer=3,
            bidirectional=False,
            reasons=tuple(reasons),
            layer1_eligible=eligible,
        )
    return DirectionComparison(
        preferred=None,
        decided_layer=4,
        bidirectional=True,
        reasons=(
            AnchorReason(
                code=ANCHOR_TIE_BIDIRECTIONAL,
                layer=LAYER_4,
                endpoint=None,
                detail=(
                    "layers 1-3 tied; both directions kept (bidirectional "
                    "candidates; F/B priors recorded but not decisive)"
                ),
            ),
        ),
        layer1_eligible=eligible,
    )


# ---------------------------------------------------------------------------
# Multi-component assembly.
# ---------------------------------------------------------------------------
def _normalize_endpoint(endpoint: str) -> str:
    normalized = _START_ALIASES.get(endpoint)
    if normalized is None:
        raise ValueError(
            f"ENDPOINT_INVALID: {endpoint!r}; expected one of {list(ENDPOINTS)}"
        )
    return normalized


def _ordered_components(side: SideMaterial) -> list[ComponentMaterial]:
    """Deterministic component order: largest first, then tag ascending."""
    return sorted(side.components, key=lambda c: (-len(c.coordinates), c.tag))


def _qualified(endpoint: str, tag: str) -> str:
    return f"{endpoint}:{tag}"


def _shared_count(first: ComponentMaterial, second: ComponentMaterial) -> int:
    return len(set(first.coordinates) & set(second.coordinates))


def _dist(first: Coordinate, second: Coordinate) -> float:
    return math.dist(first, second)


def _unit_axis(from_coord: Coordinate, to_coord: Coordinate, distance: float) -> Coordinate:
    if distance == 0.0:
        return (1.0, 0.0, 0.0)
    return (
        float(to_coord[0] - from_coord[0]) / distance,
        float(to_coord[1] - from_coord[1]) / distance,
        float(to_coord[2] - from_coord[2]) / distance,
    )


def _shift(coord: Coordinate, axis: Coordinate, shift: float) -> Coordinate:
    return (
        float(coord[0] + axis[0] * shift),
        float(coord[1] + axis[1] * shift),
        float(coord[2] + axis[2] * shift),
    )


def _covalent_radius(element: str) -> float | None:
    try:
        z = ELEMENT_SYMBOLS.index(element) + 1
    except ValueError:
        return None
    return COVALENT_RADII.get(z)


def _bonded_geometry_mismatches(
    component: ComponentMaterial,
    qualified_tag: str,
    bonded_pairs: Collection[MapPair],
    policy: DirectionAssemblyPolicy,
) -> list[FrameMismatchRecord]:
    """Detect a component whose stored bonded atoms sit in independent frames."""
    records: list[FrameMismatchRecord] = []
    maps = set(component.coordinates)
    elements = component.elements
    for first, second in sorted(bonded_pairs):
        if first not in maps or second not in maps:
            continue
        e1 = elements.get(first)
        e2 = elements.get(second)
        if e1 is None or e2 is None:
            continue
        r1 = _covalent_radius(e1)
        r2 = _covalent_radius(e2)
        if r1 is None or r2 is None:
            continue
        distance = _dist(component.coordinates[first], component.coordinates[second])
        limit = r1 + r2 + policy.bond_tolerance + policy.frame_mismatch_tolerance
        if distance > limit:
            records.append(
                FrameMismatchRecord(
                    component_tag=qualified_tag,
                    code=CODE_ASSEMBLY_FRAME_MISMATCH,
                    detail=(
                        f"bonded pair ({first},{second}) stored at {distance:.3f} A "
                        f"> limit {limit:.3f} A — component written across "
                        f"independent physical frames; stored pose never accepted"
                    ),
                    spectator_preserved=False,
                )
            )
    return records


def _place_components(
    endpoint: str,
    start_comps: list[ComponentMaterial],
    other_side: SideMaterial | None,
    policy: DirectionAssemblyPolicy,
) -> tuple[dict[str, dict[int, Coordinate]], list[PlacementRecord], dict[str, bool], str]:
    """Rigidly place start-side components into one assembly frame.

    The largest start-side component anchors the frame (stored coordinates);
    other components are placed through cross-side shared-map Kabsch fits
    (G2 ``place_single`` fallback chain) whenever a placed opposite-side
    component shares maps with them.  Components never placed keep their
    stored coordinates and are flagged frame-ambiguous.
    """
    anchor = start_comps[0]
    anchor_tag = _qualified(endpoint, anchor.tag)
    placed: dict[str, dict[int, Coordinate]] = {anchor.tag: dict(anchor.coordinates)}
    ambiguous: dict[str, bool] = {anchor.tag: False}
    placements: list[PlacementRecord] = []
    others: list[ComponentMaterial] = [] if other_side is None else list(other_side.components)
    placed_other: dict[str, dict[int, Coordinate]] = {}

    progress = True
    while progress:
        progress = False
        for other in others:
            if other.tag in placed_other:
                continue
            options: list[tuple[int, str, ComponentMaterial]] = []
            for start in start_comps:
                if start.tag not in placed:
                    continue
                n = _shared_count(other, start)
                if n > 0:
                    options.append((n, start.tag, start))
            if not options:
                continue
            _, onto_tag, onto = min(options, key=lambda item: (-item[0], item[1]))
            shared = sorted(set(other.coordinates) & set(placed[onto_tag]))
            moving = ComponentGeometry(
                _qualified(other_side.endpoint if other_side else "P", other.tag),
                {m: other.coordinates[m] for m in sorted(other.coordinates)},
            )
            target = ComponentGeometry(
                _qualified(endpoint, onto_tag),
                {m: placed[onto_tag][m] for m in sorted(placed[onto_tag])},
            )
            result, placement = place_single(
                moving,
                target,
                anchors=shared,
                anchor_tolerance=policy.anchor_tolerance,
                min_anchor_maps=policy.min_anchor_maps,
            )
            placed_other[other.tag] = result
            placements.append(
                PlacementRecord(
                    component_tag=_qualified(other_side.endpoint if other_side else "P", other.tag),
                    onto_tag=_qualified(endpoint, onto_tag),
                    n_shared=placement.n_shared,
                    basis=placement.basis,
                    rmsd_shared=placement.rmsd_shared,
                    anchor_rank_ok=placement.anchor_rank_ok,
                    frame_ambiguous=False,
                )
            )
            progress = True
        for start in start_comps:
            if start.tag in placed:
                continue
            options = []
            for other in others:
                if other.tag not in placed_other:
                    continue
                n = _shared_count(start, other)
                if n > 0:
                    options.append((n, other.tag, other))
            if not options:
                continue
            _, onto_other_tag, _ = min(options, key=lambda item: (-item[0], item[1]))
            shared = sorted(set(start.coordinates) & set(placed_other[onto_other_tag]))
            moving = ComponentGeometry(
                _qualified(endpoint, start.tag),
                {m: start.coordinates[m] for m in sorted(start.coordinates)},
            )
            target = ComponentGeometry(
                _qualified(other_side.endpoint if other_side else "P", onto_other_tag),
                {m: placed_other[onto_other_tag][m] for m in sorted(placed_other[onto_other_tag])},
            )
            result, placement = place_single(
                moving,
                target,
                anchors=shared,
                anchor_tolerance=policy.anchor_tolerance,
                min_anchor_maps=policy.min_anchor_maps,
            )
            placed[start.tag] = result
            ambiguous[start.tag] = False
            placements.append(
                PlacementRecord(
                    component_tag=_qualified(endpoint, start.tag),
                    onto_tag=_qualified(other_side.endpoint if other_side else "P", onto_other_tag),
                    n_shared=placement.n_shared,
                    basis=placement.basis,
                    rmsd_shared=placement.rmsd_shared,
                    anchor_rank_ok=placement.anchor_rank_ok,
                    frame_ambiguous=False,
                )
            )
            progress = True

    for start in start_comps:
        if start.tag not in placed:
            placed[start.tag] = dict(start.coordinates)
            ambiguous[start.tag] = True
    for other in others:
        if other.tag not in placed_other:
            placed_other[other.tag] = dict(other.coordinates)
    return placed, placements, ambiguous, anchor_tag


def _nonbonded_metrics(
    coords: Mapping[int, Coordinate],
    maps: Sequence[int],
    bonded_pairs: Collection[MapPair],
    collision_min: float,
) -> tuple[float | None, int]:
    bonded = set(bonded_pairs)
    bonded |= {(b, a) for a, b in bonded_pairs}
    min_distance: float | None = None
    n_severe = 0
    for i, first in enumerate(maps):
        for second in maps[i + 1 :]:
            if (first, second) in bonded:
                continue
            distance = _dist(coords[first], coords[second])
            if min_distance is None or distance < min_distance:
                min_distance = distance
            if distance < collision_min:
                n_severe += 1
    return min_distance, n_severe


def _approach_configurations(
    assembly_id: str,
    endpoint: str,
    start_tag_of: Mapping[int, str],
    placed_coords: Mapping[int, Coordinate],
    anchor_tag: str,
    driver_sets: Sequence[DriverSetSpec],
    policy: DirectionAssemblyPolicy,
) -> list[ApproachConfiguration]:
    """Finite approach configurations: translate fragments along driver axes."""
    configs: list[ApproachConfiguration] = []
    index = 0
    for dset in driver_sets:
        for bond in dset.driver_bonds:
            if bond.kind != "B" or len(bond.atom_maps) != 2:
                continue
            first, second = bond.atom_maps
            tag_first = start_tag_of.get(first)
            tag_second = start_tag_of.get(second)
            if tag_first is None or tag_second is None or tag_first == tag_second:
                continue
            if tag_first == anchor_tag:
                fixed_map, moved_map = first, second
            elif tag_second == anchor_tag:
                fixed_map, moved_map = second, first
            else:
                maps_first = sorted(m for m, t in start_tag_of.items() if t == tag_first)
                maps_second = sorted(m for m, t in start_tag_of.items() if t == tag_second)
                if (maps_first[0], tag_first) <= (maps_second[0], tag_second):
                    fixed_map, moved_map = first, second
                else:
                    fixed_map, moved_map = second, first
            moved_tag = start_tag_of[moved_map]
            fixed_tag = start_tag_of[fixed_map]
            moved_maps = tuple(sorted(m for m, t in start_tag_of.items() if t == moved_tag))
            current = _dist(placed_coords[moved_map], placed_coords[fixed_map])
            axis = _unit_axis(placed_coords[fixed_map], placed_coords[moved_map], current)
            for target in sorted(policy.approach_target_distances):
                configs.append(
                    ApproachConfiguration(
                        configuration_id=f"{assembly_id}-cfg-{index:03d}",
                        driver_set_id=dset.driver_set_id,
                        driver_atom_maps=bond.atom_maps,
                        moved_component_tag=_qualified(endpoint, moved_tag),
                        fixed_component_tag=_qualified(endpoint, fixed_tag),
                        moved_maps=moved_maps,
                        target_distance=target,
                        axis=axis,
                        shift=target - current,
                    )
                )
                index += 1
    return configs


def assemble_start_side(
    side: SideMaterial,
    *,
    assembly_id: str,
    direction: str,
    other_side: SideMaterial | None = None,
    driver_sets: Sequence[DriverSetSpec] = (),
    bonded_pairs: Collection[MapPair] = (),
    elements: Mapping[int, str] | None = None,
    policy: DirectionAssemblyPolicy | None = None,
) -> AssemblyRecord:
    """Assemble one start side: rigid placement + approach configs + checks.

    Deterministic ordered enumeration: anchor = largest component (tag
    ascending tie-break); placements follow the shared-map BFS order;
    approach configurations follow (driver-set order, driver-bond order,
    target distance ascending).
    """
    endpoint = _normalize_endpoint(side.endpoint)
    if direction != ("R_to_P" if endpoint == "R" else "P_to_R"):
        raise ValueError(
            f"DIRECTION_MISMATCH: {direction!r} does not start from {endpoint!r}"
        )
    if not side.components:
        return AssemblyRecord(
            assembly_id=assembly_id,
            endpoint=endpoint,
            direction=direction,
            anchor_component_tag="",
            placements=(),
            approach_configurations=(),
            spectator_tags=(),
            frame_mismatches=(),
            min_nonbonded_distance=None,
            n_severe_contacts=0,
            collision_ok=False,
            ready=False,
            failure_codes=(CODE_ASSEMBLY_FAILED,),
            coordinates={},
        )
    active_policy = policy if policy is not None else policy_from_config(None)
    element_map = dict(elements) if elements is not None else {}
    for component in side.components:
        element_map.update(component.elements)
    start_comps = _ordered_components(side)
    placed, placements, ambiguous, anchor_tag = _place_components(
        endpoint, start_comps, other_side, active_policy
    )

    failure_codes: list[str] = []
    frame_mismatches: list[FrameMismatchRecord] = []
    spectator_tags: list[str] = []
    comp_of_map: dict[int, str] = {}
    for component in start_comps:
        qualified = _qualified(endpoint, component.tag)
        comp_of_map.update({m: component.tag for m in component.coordinates})
        mismatches = _bonded_geometry_mismatches(
            component, qualified, bonded_pairs, active_policy
        )
        frame_mismatches.extend(mismatches)
        if ambiguous.get(component.tag, False):
            is_spectator = not component.contains_edit_atoms
            if is_spectator:
                spectator_tags.append(qualified)
            frame_mismatches.append(
                FrameMismatchRecord(
                    component_tag=qualified,
                    code=CODE_ASSEMBLY_FRAME_MISMATCH,
                    detail=(
                        "component never rigidly linked to the anchor frame "
                        "(source independent-frame placement); stored pose "
                        "preserved only when spectator, never silently accepted "
                        "as a common physical frame"
                    ),
                    spectator_preserved=is_spectator,
                )
            )

    flat_coords: dict[int, Coordinate] = {}
    for component in start_comps:
        flat_coords.update(placed[component.tag])
    maps = sorted(flat_coords)
    for map_, coord in flat_coords.items():
        if not all(math.isfinite(value) for value in coord):
            failure_codes.append(CODE_ASSEMBLY_FAILED)
            break

    if any(m.code == CODE_ASSEMBLY_FRAME_MISMATCH and not m.spectator_preserved for m in frame_mismatches):
        if CODE_ASSEMBLY_FRAME_MISMATCH not in failure_codes:
            failure_codes.append(CODE_ASSEMBLY_FRAME_MISMATCH)

    start_tag_of = {m: comp_of_map[m] for m in maps}
    configs = _approach_configurations(
        assembly_id,
        endpoint,
        start_tag_of,
        flat_coords,
        anchor_tag,
        driver_sets,
        active_policy,
    )

    base_min, base_severe = _nonbonded_metrics(
        flat_coords, maps, bonded_pairs, active_policy.collision_min_distance
    )
    evaluated_configs: list[ApproachConfiguration] = []
    best_config_ok: ApproachConfiguration | None = None
    for config in configs:
        moved_coords = dict(flat_coords)
        for m in config.moved_maps:
            moved_coords[m] = _shift(moved_coords[m], config.axis, config.shift)
        c_min, c_severe = _nonbonded_metrics(
            moved_coords, maps, bonded_pairs, active_policy.collision_min_distance
        )
        updated = ApproachConfiguration(
            configuration_id=config.configuration_id,
            driver_set_id=config.driver_set_id,
            driver_atom_maps=config.driver_atom_maps,
            moved_component_tag=config.moved_component_tag,
            fixed_component_tag=config.fixed_component_tag,
            moved_maps=config.moved_maps,
            target_distance=config.target_distance,
            axis=config.axis,
            shift=config.shift,
            collision_ok=c_severe == 0,
            min_nonbonded_distance=c_min,
        )
        evaluated_configs.append(updated)
        if updated.collision_ok and (
            best_config_ok is None
            or (updated.min_nonbonded_distance or 0.0)
            > (best_config_ok.min_nonbonded_distance or 0.0)
        ):
            best_config_ok = updated

    collision_ok = base_severe == 0 or best_config_ok is not None
    if not collision_ok:
        failure_codes.append(CODE_ASSEMBLY_COLLISION)

    if base_severe == 0:
        min_nonbonded = base_min
        n_severe = 0
        collision_ok_final = True
    elif best_config_ok is not None:
        min_nonbonded = best_config_ok.min_nonbonded_distance
        n_severe = 0
        collision_ok_final = True
    else:
        min_nonbonded = base_min
        n_severe = base_severe
        collision_ok_final = False

    ready = not failure_codes
    if ready:
        failure_codes_out: tuple[str, ...] = (CODE_ASSEMBLY_READY,)
    else:
        failure_codes_out = tuple(dict.fromkeys(failure_codes))

    return AssemblyRecord(
        assembly_id=assembly_id,
        endpoint=endpoint,
        direction=direction,
        anchor_component_tag=anchor_tag,
        placements=tuple(placements),
        approach_configurations=tuple(evaluated_configs),
        spectator_tags=tuple(spectator_tags),
        frame_mismatches=tuple(frame_mismatches),
        min_nonbonded_distance=min_nonbonded,
        n_severe_contacts=n_severe,
        collision_ok=collision_ok_final,
        ready=ready,
        failure_codes=failure_codes_out,
        coordinates=flat_coords,
    )


# ---------------------------------------------------------------------------
# Full resolve pipeline.
# ---------------------------------------------------------------------------
def _assembly_reasons(record: AssemblyRecord | None, endpoint: str) -> list[AnchorReason]:
    if record is None:
        return []
    out: list[AnchorReason] = [
        AnchorReason(
            code=CODE_ASSEMBLY_READY if record.ready else record.failure_codes[0],
            layer=LAYER_ASSEMBLY,
            endpoint=endpoint,
            detail=(
                f"assembly {record.assembly_id}: "
                f"min_nonbonded={record.min_nonbonded_distance}"
            ),
        )
    ]
    for mismatch in record.frame_mismatches:
        out.append(
            AnchorReason(
                code=mismatch.code,
                layer=LAYER_ASSEMBLY,
                endpoint=endpoint,
                detail=mismatch.detail,
            )
        )
    if record.spectator_tags:
        out.append(
            AnchorReason(
                code=CODE_SPECTATOR_PRESERVED,
                layer=LAYER_ASSEMBLY,
                endpoint=endpoint,
                detail=(
                    "spectator components preserved (stored placement, flagged): "
                    + ", ".join(record.spectator_tags)
                ),
            )
        )
    return out


def _assembly_blockers(record: AssemblyRecord | None) -> list[str]:
    if record is None or record.ready:
        return []
    return [code for code in record.failure_codes if code != CODE_ASSEMBLY_READY]


def _direction_for(start: str) -> str:
    return "R_to_P" if start == "R" else "P_to_R"


def _other_endpoint(start: str) -> str:
    return "P" if start == "R" else "R"


def _side_for(inputs: DirectionAssemblyInputs, endpoint: str) -> SideMaterial | None:
    return inputs.r_side if endpoint == "R" else inputs.p_side


def resolve_direction_assembly(
    inputs: DirectionAssemblyInputs,
    *,
    config: Mapping[str, Any] | None = None,
) -> DirectionAssemblyResult:
    """Resolve direction candidates for one reaction (todo 15 public entry).

    Pipeline: layered comparison (only selection mechanism) → cheap F/B priors
    recorded → deterministic start-endpoint enumeration (preferred only, or
    R then P for ties / blocked layer 1) → per-start multi-component assembly
    → per driver-set candidates with demoted hybrid anchors where applicable.
    """
    policy = policy_from_config(config)
    comparison = compare_directions(inputs.r_profile, inputs.p_profile)
    prior = fb_prior_reason(inputs.fb_counts)
    hybrid = hybrid_anchor_reasons(inputs.fb_counts)
    hard_blocked = comparison.blocked_no_endpoint

    if comparison.preferred is not None:
        starts: tuple[str, ...] = (comparison.preferred,)
    else:
        # ties and layer-1 hard blocks both enumerate R then P deterministically
        starts = ("R", "P")

    driver_sets: Sequence[DriverSetSpec] = inputs.driver_sets
    if not driver_sets:
        driver_sets = (DriverSetSpec(driver_set_id="ds-000", route_strategy_id=None),)

    candidates: list[DirectionCandidate] = []
    for start in starts:
        direction = _direction_for(start)
        profile = inputs.r_profile if start == "R" else inputs.p_profile
        other_endpoint = _other_endpoint(start)
        other_side = _side_for(inputs, other_endpoint)
        side = _side_for(inputs, start)
        assembly_id = f"asm-{start}-000"
        target_assembly_id = f"asm-{other_endpoint}-000"
        assembly: AssemblyRecord | None = None
        if side is not None:
            assembly = assemble_start_side(
                side,
                assembly_id=assembly_id,
                direction=direction,
                other_side=other_side,
                driver_sets=driver_sets,
                bonded_pairs=inputs.bonded_pairs,
                elements=inputs.elements,
                policy=policy,
            )
        # The target endpoint's assembly also gates readiness: scan end values
        # require both sides' geometry to assemble (design §7).
        target_assembly: AssemblyRecord | None = None
        if other_side is not None:
            target_assembly = assemble_start_side(
                other_side,
                assembly_id=target_assembly_id,
                direction=_direction_for(other_endpoint),
                other_side=side,
                driver_sets=driver_sets,
                bonded_pairs=inputs.bonded_pairs,
                elements=inputs.elements,
                policy=policy,
            )

        shared_reasons: list[AnchorReason] = list(comparison.reasons)
        if hard_blocked:
            shared_reasons.append(
                AnchorReason(
                    code=CODE_NO_USABLE_ENDPOINT,
                    layer=LAYER_1,
                    endpoint=start,
                    detail=(
                        "layer-1 hard gate: at least one side lacks a usable "
                        "target definition; no direction can start"
                    ),
                )
            )
        elif comparison.preferred is not None and start == comparison.preferred:
            pass  # comparison reasons already name the winner
        elif comparison.bidirectional:
            shared_reasons.append(
                AnchorReason(
                    code=ANCHOR_TIE_BIDIRECTIONAL,
                    layer=LAYER_4,
                    endpoint=start,
                    detail="tie bidirectional candidate kept (fixed order R then P)",
                )
            )
        if not profile.layer1_ok():
            shared_reasons.append(
                AnchorReason(
                    code=CODE_NO_USABLE_ENDPOINT,
                    layer=LAYER_1,
                    endpoint=start,
                    detail="endpoint lacks complete review or usable target definition",
                )
            )
        if assembly is None:
            shared_reasons.append(
                AnchorReason(
                    code=ANCHOR_ASSEMBLY_NOT_EVALUATED,
                    layer=LAYER_ASSEMBLY,
                    endpoint=start,
                    detail="no start-side material provided; assembly not evaluated here",
                )
            )
        if target_assembly is None:
            shared_reasons.append(
                AnchorReason(
                    code=ANCHOR_ASSEMBLY_NOT_EVALUATED,
                    layer=LAYER_ASSEMBLY,
                    endpoint=other_endpoint,
                    detail="no target-side material provided; assembly not evaluated here",
                )
            )

        for dset in driver_sets:
            strategy_id = dset.route_strategy_id
            rule_trace: list[str] = [SELECTION_MECHANISM, SPLICE_POLICY]
            reasons = list(shared_reasons)
            reasons.append(prior)
            if strategy_id in ANCHOR_APPLICABLE_STRATEGIES:
                rule_trace.extend(hybrid_rule_ids_for(inputs.fb_counts))
                reasons.extend(hybrid)
            else:
                rule_trace.append("hybrid_anchors_not_applicable_to_strategy")
            reasons.extend(_assembly_reasons(assembly, start))
            reasons.extend(_assembly_reasons(target_assembly, other_endpoint))

            blockers: list[str] = []
            if hard_blocked or not profile.layer1_ok():
                blockers.append(CODE_NO_USABLE_ENDPOINT)
            blockers.extend(_assembly_blockers(assembly))
            blockers.extend(_assembly_blockers(target_assembly))
            ready = not blockers
            candidates.append(
                DirectionCandidate(
                    start_endpoint=start,
                    direction=direction,
                    assembly_id=None if assembly is None else assembly.assembly_id,
                    anchor_reason=tuple(reasons),
                    route_strategy_id=strategy_id,
                    driver_set_id=dset.driver_set_id,
                    rule_trace=tuple(dict.fromkeys(rule_trace)),
                    bidirectional=comparison.bidirectional,
                    assembly=assembly,
                    target_assembly=target_assembly,
                    ready=ready,
                    readiness_blockers=tuple(dict.fromkeys(blockers)),
                )
            )

    return DirectionAssemblyResult(
        schema_version=SCHEMA_DIRECTION_ASSEMBLY,
        reaction_id=inputs.reaction_id,
        fb_counts=inputs.fb_counts,
        comparison=comparison,
        direction_candidates=tuple(candidates),
    )


__all__ = [
    "ANCHOR_APPLICABLE_STRATEGIES",
    "ANCHOR_ASSEMBLY_NOT_EVALUATED",
    "ANCHOR_CONFIGURATION_OK",
    "ANCHOR_DRIVER_BONDED_AT_START",
    "ANCHOR_LAYER1_REVIEW_PRIORITY",
    "ANCHOR_ORIENTATION_OK",
    "ANCHOR_REDUCES_FRAGMENT_DOF",
    "ANCHOR_TIE_BIDIRECTIONAL",
    "ANCHOR_UNDRIVEN_EVENT_COVERAGE_OK",
    "CODE_ASSEMBLY_COLLISION",
    "CODE_ASSEMBLY_FAILED",
    "CODE_ASSEMBLY_FRAME_MISMATCH",
    "CODE_ASSEMBLY_READY",
    "CODE_NO_USABLE_ENDPOINT",
    "CODE_SPECTATOR_PRESERVED",
    "DEFAULT_APPROACH_TARGET_DISTANCES",
    "DEFAULT_ANCHOR_TOLERANCE",
    "DEFAULT_BOND_TOLERANCE",
    "DEFAULT_COLLISION_MIN_DISTANCE",
    "DEFAULT_FRAME_MISMATCH_TOLERANCE",
    "DEFAULT_MIN_ANCHOR_MAPS",
    "HYBRID_RULE_FEWER_BOND_EVALUATED",
    "HYBRID_RULE_IDS",
    "HYBRID_RULE_MINIMAL_CONSTRAINTS",
    "HYBRID_RULE_MORE_BOND_DEFAULT",
    "HYBRID_RULE_REPRESENTATIVE_OR_PATH",
    "HYBRID_RULE_TIE_GEOMETRY",
    "LAYER_1",
    "LAYER_2",
    "LAYER_3",
    "LAYER_4",
    "LAYER_ASSEMBLY",
    "LAYER_HYBRID",
    "LAYER_PRIOR",
    "PRIOR_FB_NONE",
    "PRIOR_FB_STRETCH_FROM_ANCHOR",
    "SCHEMA_DIRECTION_ASSEMBLY",
    "SELECTION_MECHANISM",
    "SPLICE_POLICY",
    "ApproachConfiguration",
    "AnchorReason",
    "AssemblyRecord",
    "ComponentMaterial",
    "DirectionAssemblyInputs",
    "DirectionAssemblyPolicy",
    "DirectionAssemblyResult",
    "DirectionCandidate",
    "DirectionComparison",
    "DriverBondSpec",
    "DriverSetSpec",
    "EditCountsSource",
    "EndpointProfile",
    "FBCounts",
    "FrameMismatchRecord",
    "PlacementRecord",
    "SideMaterial",
    "assemble_start_side",
    "compare_directions",
    "fb_counts_from_edit_graph",
    "fb_prior_reason",
    "hybrid_anchor_reasons",
    "hybrid_rule_ids_for",
    "policy_from_config",
    "resolve_direction_assembly",
]
