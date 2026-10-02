"""Graph-structural scan-strategy registry and motif routing (todo 11).

Implements design §5.1 routing order and the §5.2 eleven-strategy registry of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``:

1. Watershed / domain errors first (conservation, ``SPECIAL_DOMAIN``,
   ``SPECIAL_ELECTRONIC_STATE_REQUIRED``) → typed exits.
2. ``NO_REACTION_CHANGE`` when there is no connectivity/electronic/stereo/
   conformational target change.
3. Event construction is *consumed* from todo 9's typed event coupling graph
   H (``EventCouplingGraph``); this module never re-derives events.
4. No-connectivity-change branch: pure order changes route to the delocalized
   strategy (``AROMATIC_COUPLED``); non-aromatic O networks without an
   interpretable geometric driver route to ``NETWORK_PATH`` / review.
5. Per strong-component motif matching → 1/2/3-driver strategy candidates.
6. Direction/schedule hooks (todos 15/14) and the budget gate (todo 17) are
   explicit no-op extension points recorded in ``rule_trace``.

Registry entry vocabulary is design §5.2 verbatim: ``LOCAL_CONNECTIVITY``,
``H_TRANSFER``, ``CONNECTIVITY_EXCHANGE``, ``H_TRANSFER_COUPLED``,
``RING_COUPLED``, ``H2_EVENT``, ``AROMATIC_COUPLED``,
``MULTI_EVENT_CONNECTED``, ``CONFORMATION_STEREO``, ``NETWORK_PATH``,
``SPECIAL_DOMAIN``.  ``PATH_REQUIRED`` (design §8 mode table) is not a §5.2
registry id; path-routed candidates carry ``NETWORK_PATH`` here and feed the
PathRequest protocol of todo 26.

Hard rules enforced by this module:

- Motif matching is **graph-structural only**.  Strata are cohort labels
  (G0/G1), never routing inputs; the selector API rejects a ``stratum``
  argument with ``ValueError("STRATUM_NOT_SUPPORTED...")``.
- Mutually exclusive major classes must not swallow labels: a reaction with
  H transfer AND connectivity exchange routes to a multi-event strategy with
  **both** labels preserved on the candidate and in the case trace.
- Every input receives **exactly one** disposition:
  ``executable_candidate_set`` / ``needs_review`` / ``special_domain_exit`` /
  ``typed_rejection`` — never a silent empty result.
- Proposal-level routing only: no coordinates, no schedules, no ORCA/ACP
  compilation (todos 12–14 own those stages).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pes2ts_core.g1.endpoint_context import EndpointContext
from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.g1.event_coupling import (
    EVENT_CONNECTIVITY_EXCHANGE,
    EVENT_CONTEXT_NEAR,
    EVENT_DELOCALIZED,
    EVENT_H2,
    EVENT_H_TRANSFER,
    EVENT_RING,
    EVENT_SHARED_CENTER,
    EventCouplingGraph,
    EventNode,
)
from pes2ts_core.g1.reaction_edit_graph import ReactionEditGraph
from pes2ts_core.utils.hashing import stable_json_dumps

# allow: SIZE_OK — plan-named todo-11 module (design §5.2 registry table +
# §5.1 routing order + RouteResult contract for todos 12–17); precedent:
# contracts_v2.py / event_coupling.py / endpoint_context.py single modules.

# ---------------------------------------------------------------------------
# Schema + outcome vocabulary.
# ---------------------------------------------------------------------------
SCHEMA_ROUTE_RESULT: Final[str] = "g1_strategy_route_result_v1"

OUTCOME_EXECUTABLE_CANDIDATE_SET: Final[str] = "executable_candidate_set"
OUTCOME_NEEDS_REVIEW: Final[str] = "needs_review"
OUTCOME_SPECIAL_DOMAIN_EXIT: Final[str] = "special_domain_exit"
OUTCOME_TYPED_REJECTION: Final[str] = "typed_rejection"
OUTCOMES: Final[tuple[str, ...]] = (
    OUTCOME_EXECUTABLE_CANDIDATE_SET,
    OUTCOME_NEEDS_REVIEW,
    OUTCOME_SPECIAL_DOMAIN_EXIT,
    OUTCOME_TYPED_REJECTION,
)

#: Typed rejection codes (watershed / no-reaction branches).
REJ_ELEMENT_IMBALANCE: Final[str] = "ELEMENT_IMBALANCE"
REJ_ISOTOPE_IMBALANCE: Final[str] = "ISOTOPE_IMBALANCE"
REJ_NO_REACTION_CHANGE: Final[str] = "NO_REACTION_CHANGE"

#: Special-domain exit codes (design §3.2 / §5.2 SPECIAL_DOMAIN).
SPEC_DOMAIN: Final[str] = "SPECIAL_DOMAIN"
SPEC_ELECTRONIC_STATE: Final[str] = "SPECIAL_ELECTRONIC_STATE_REQUIRED"

#: Raised (never returned) when the selector API receives a ``stratum`` kwarg.
STRATUM_NOT_SUPPORTED: Final[str] = "STRATUM_NOT_SUPPORTED"

#: Design §5.2 registry ids, verbatim and in table order.
STRATEGY_LOCAL_CONNECTIVITY: Final[str] = "LOCAL_CONNECTIVITY"
STRATEGY_H_TRANSFER: Final[str] = "H_TRANSFER"
STRATEGY_CONNECTIVITY_EXCHANGE: Final[str] = "CONNECTIVITY_EXCHANGE"
STRATEGY_H_TRANSFER_COUPLED: Final[str] = "H_TRANSFER_COUPLED"
STRATEGY_RING_COUPLED: Final[str] = "RING_COUPLED"
STRATEGY_H2_EVENT: Final[str] = "H2_EVENT"
STRATEGY_AROMATIC_COUPLED: Final[str] = "AROMATIC_COUPLED"
STRATEGY_MULTI_EVENT_CONNECTED: Final[str] = "MULTI_EVENT_CONNECTED"
STRATEGY_CONFORMATION_STEREO: Final[str] = "CONFORMATION_STEREO"
STRATEGY_NETWORK_PATH: Final[str] = "NETWORK_PATH"
STRATEGY_SPECIAL_DOMAIN: Final[str] = "SPECIAL_DOMAIN"

#: The eleven §5.2 main strategies in design-table order.
STRATEGY_IDS: Final[tuple[str, ...]] = (
    STRATEGY_LOCAL_CONNECTIVITY,
    STRATEGY_H_TRANSFER,
    STRATEGY_CONNECTIVITY_EXCHANGE,
    STRATEGY_H_TRANSFER_COUPLED,
    STRATEGY_RING_COUPLED,
    STRATEGY_H2_EVENT,
    STRATEGY_AROMATIC_COUPLED,
    STRATEGY_MULTI_EVENT_CONNECTED,
    STRATEGY_CONFORMATION_STEREO,
    STRATEGY_NETWORK_PATH,
    STRATEGY_SPECIAL_DOMAIN,
)

#: Motif-matching specificity order (§5.1 step 5).  ``SPECIAL_DOMAIN`` is a
#: watershed exit, never motif-matched; it is excluded here on purpose.
STRATEGY_ROUTING_ORDER: Final[tuple[str, ...]] = (
    STRATEGY_H2_EVENT,
    STRATEGY_H_TRANSFER_COUPLED,
    STRATEGY_H_TRANSFER,
    STRATEGY_CONNECTIVITY_EXCHANGE,
    STRATEGY_RING_COUPLED,
    STRATEGY_AROMATIC_COUPLED,
    STRATEGY_LOCAL_CONNECTIVITY,
    STRATEGY_MULTI_EVENT_CONNECTED,
    STRATEGY_CONFORMATION_STEREO,
    STRATEGY_NETWORK_PATH,
)
_STRATEGY_RANK: Final[dict[str, int]] = {
    name: i for i, name in enumerate(STRATEGY_ROUTING_ORDER)
}

# ---------------------------------------------------------------------------
# Registry rule ids (proposal-level trace vocabulary; distinct from the
# todo-9 event-coupling rule ids, which are echoed separately as R_EVENT_*).
# ---------------------------------------------------------------------------
R_WATERSHED_ELEMENT_CONSERVATION: Final[str] = "R_WATERSHED_ELEMENT_CONSERVATION"
R_WATERSHED_ISOTOPE_CONSERVATION: Final[str] = "R_WATERSHED_ISOTOPE_CONSERVATION"
R_WATERSHED_SPECIAL_DOMAIN_ELEMENT: Final[str] = "R_WATERSHED_SPECIAL_DOMAIN_ELEMENT"
R_WATERSHED_SPECIAL_ELECTRONIC_STATE: Final[str] = (
    "R_WATERSHED_SPECIAL_ELECTRONIC_STATE"
)
R_NO_REACTION_CHANGE: Final[str] = "R_NO_REACTION_CHANGE"

R_BRANCH_DELOCALIZED_ORDER_ONLY: Final[str] = "R_BRANCH_DELOCALIZED_ORDER_ONLY"
R_BRANCH_CONFORMATION_STEREO: Final[str] = "R_BRANCH_CONFORMATION_STEREO"
R_BRANCH_O_NETWORK_REVIEW: Final[str] = "R_BRANCH_O_NETWORK_REVIEW"

R_MOTIF_LOCAL_CONNECTIVITY_FORMED: Final[str] = "R_MOTIF_LOCAL_CONNECTIVITY_FORMED"
R_MOTIF_LOCAL_CONNECTIVITY_BROKEN: Final[str] = "R_MOTIF_LOCAL_CONNECTIVITY_BROKEN"
R_MOTIF_H_TRANSFER_PURE: Final[str] = "R_MOTIF_H_TRANSFER_PURE"
R_MOTIF_H_TRANSFER_COUPLED: Final[str] = "R_MOTIF_H_TRANSFER_COUPLED"
R_MOTIF_H2_EVENT: Final[str] = "R_MOTIF_H2_EVENT"
R_MOTIF_CONNECTIVITY_EXCHANGE: Final[str] = "R_MOTIF_CONNECTIVITY_EXCHANGE"
R_MOTIF_RING_CLOSURE: Final[str] = "R_MOTIF_RING_CLOSURE"
R_MOTIF_RING_OPENING: Final[str] = "R_MOTIF_RING_OPENING"
R_MOTIF_RING_COUPLED: Final[str] = "R_MOTIF_RING_COUPLED"
R_MOTIF_AROMATIC_WITH_EVENTS: Final[str] = "R_MOTIF_AROMATIC_WITH_EVENTS"
R_MOTIF_AROMATIC_ORDER_ONLY: Final[str] = "R_MOTIF_AROMATIC_ORDER_ONLY"
R_MOTIF_MULTI_EVENT_CONNECTED: Final[str] = "R_MOTIF_MULTI_EVENT_CONNECTED"
R_MOTIF_CONFORMATION_STEREO: Final[str] = "R_MOTIF_CONFORMATION_STEREO"
R_MOTIF_NETWORK_PATH_MULTI_COMPONENT: Final[str] = (
    "R_MOTIF_NETWORK_PATH_MULTI_COMPONENT"
)
R_MOTIF_NETWORK_PATH_FALLBACK: Final[str] = "R_MOTIF_NETWORK_PATH_FALLBACK"

#: Event-type labels echoed into traces so multi-label mechanisms survive
#: routing (design §5: "主策略 + 多个结构标签"; §5.1 step 5).
_EVENT_RULE_BY_TYPE: Final[dict[str, str]] = {
    EVENT_H_TRANSFER: "R_EVENT_H_TRANSFER",
    EVENT_H2: "R_EVENT_H2_EVENT",
    EVENT_CONNECTIVITY_EXCHANGE: "R_EVENT_CONNECTIVITY_EXCHANGE",
    EVENT_RING: "R_EVENT_RING_REORGANIZATION",
    EVENT_DELOCALIZED: "R_EVENT_DELocalIZED_REGION",
    EVENT_SHARED_CENTER: "R_EVENT_SHARED_CENTER",
}

#: Extension-point markers (owning todos named explicitly; no-ops at todo 11).
R_HOOK_DIRECTION_TODO15_PENDING: Final[str] = "R_HOOK_DIRECTION_TODO15_PENDING"
R_HOOK_SCHEDULE_TODO14_PENDING: Final[str] = "R_HOOK_SCHEDULE_TODO14_PENDING"
R_HOOK_BUDGET_TODO17_PENDING: Final[str] = "R_HOOK_BUDGET_TODO17_PENDING"

# ---------------------------------------------------------------------------
# SPECIAL_DOMAIN detection policy.
# ---------------------------------------------------------------------------
#: Conservative organic element set for first-pass special-domain detection.
#:
#: Design §12 states that Demo24 covers only C/H/N/O/F/Cl small-molecule
#: singlets and that the design does **not** thereby validate metals,
#: organometallics, catalysis, radicals or solution multi-body reactions; it
#: does not pin a machine-checkable metal list.  Policy (recorded here as a
#: named constant per the todo-11 contract): any atom whose element falls
#: outside this organic set triggers the typed ``SPECIAL_DOMAIN`` unsupported
#: exit (full required-representation/method detail is todo 29's deliverable;
#: this router emits the exit code + reason + minimal requirement list only).
ORGANIC_ELEMENTS: Final[frozenset[str]] = frozenset(
    {"H", "C", "N", "O", "F", "Cl", "S", "P", "Br", "I", "B", "Si"}
)

#: Minimal required representations/methods listed at the router exit
#: (todo 29 expands per subdomain).
_SPECIAL_DOMAIN_REPRESENTATIONS: Final[tuple[str, ...]] = (
    "coordination/metal-aware molecular representation",
    "periodic/slab representation for surface models",
    "spin-adapted electronic-state representation for electron exchange",
    "excited-state representation for state crossings",
)
_SPECIAL_DOMAIN_METHODS: Final[tuple[str, ...]] = (
    "validated special-domain backend/plugin (never generic covalent-radius rules)",
)
_SPECIAL_ELECTRONIC_REPRESENTATIONS: Final[tuple[str, ...]] = (
    "equal particle-number endpoints with explicit charge/multiplicity ledger",
    "cross-spin-surface or excited-state workflow (e.g. MECP) when surfaces differ",
)
_SPECIAL_ELECTRONIC_METHODS: Final[tuple[str, ...]] = (
    "dedicated electronic-state flow; ordinary NEB is not a universal fallback",
)

# Edit-kind + event-type constants (strings match todo-7/9 vocabularies).
_KIND_FORMED: Final[str] = "formed"
_KIND_BROKEN: Final[str] = "broken"
_KIND_ORDER: Final[str] = "order_changed"
_ELEMENT_H: Final[str] = "H"
_MAX_DRIVERS: Final[int] = 3


# ---------------------------------------------------------------------------
# Registry dataclasses.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StrategySpec:
    """One §5.2 registry row.

    ``required_graph_pattern`` is the predicate description over the P0 layer
    objects (``EndpointGraphBundle`` / ``ReactionEditGraph`` /
    ``EndpointContext`` / ``EventCouplingGraph``).  ``escalation_fallback``
    names the next strategy(ies) when this pattern fails or the candidate set
    is empty downstream (todos 13/16/17 consume it; recorded on every
    candidate here).
    """

    strategy_id: str
    required_graph_pattern: str
    preferred_drivers: str
    monitoring_targets: str
    escalation_fallback: tuple[str, ...]


#: Design §5.2 table, row order preserved.
STRATEGY_REGISTRY: Final[dict[str, StrategySpec]] = {
    STRATEGY_LOCAL_CONNECTIVITY: StrategySpec(
        strategy_id=STRATEGY_LOCAL_CONNECTIVITY,
        required_graph_pattern=(
            "one local connectivity change (exactly one formed or broken "
            "edit in the routed component); local order changes allowed"
        ),
        preferred_drivers="one formed/broken B; ring closure/opening also applicable",
        monitoring_targets=(
            "all O, local ring geometry, non-target connectivity and stereo"
        ),
        escalation_fallback=(STRATEGY_NETWORK_PATH,),
    ),
    STRATEGY_H_TRANSFER: StrategySpec(
        strategy_id=STRATEGY_H_TRANSFER,
        required_graph_pattern=(
            "one complete heavy-atom D-H->A event (H_TRANSFER) with no "
            "additional connectivity change in the component"
        ),
        preferred_drivers=(
            "first try one H-A or D-H B; the other H distance is a monitor"
        ),
        monitoring_targets=(
            "both H distances, D-A, angle DHA, adjacent electronic rearrangement"
        ),
        escalation_fallback=(STRATEGY_H_TRANSFER_COUPLED, STRATEGY_NETWORK_PATH),
    ),
    STRATEGY_CONNECTIVITY_EXCHANGE: StrategySpec(
        strategy_id=STRATEGY_CONNECTIVITY_EXCHANGE,
        required_graph_pattern=(
            "formed and broken edits sharing a non-H center atom with full "
            "local context in one component"
        ),
        preferred_drivers="forming B + breaking B",
        monitoring_targets="conjugated O, central chirality, approach direction",
        escalation_fallback=(STRATEGY_NETWORK_PATH,),
    ),
    STRATEGY_H_TRANSFER_COUPLED: StrategySpec(
        strategy_id=STRATEGY_H_TRANSFER_COUPLED,
        required_graph_pattern=(
            "H transfer event(s) connected (strong component) to heavy-atom "
            "connectivity/ring events"
        ),
        preferred_drivers=(
            "main heavy-atom B + one H B; add a second H B when needed (<=3)"
        ),
        monitoring_targets=(
            "undriven H distances, all heavy-atom targets, regional electronic state"
        ),
        escalation_fallback=(STRATEGY_NETWORK_PATH,),
    ),
    STRATEGY_RING_COUPLED: StrategySpec(
        strategy_id=STRATEGY_RING_COUPLED,
        required_graph_pattern=(
            "multiple edits connected through the same local ring support "
            "structure (RING_REORGANIZATION with >=2 connectivity edits, or "
            "ring event(s) coupled to other heavy-atom events)"
        ),
        preferred_drivers="1-3 key connection B; A/D when justified",
        monitoring_targets="remaining edits, ring distortion, configuration, collisions",
        escalation_fallback=(STRATEGY_NETWORK_PATH,),
    ),
    STRATEGY_H2_EVENT: StrategySpec(
        strategy_id=STRATEGY_H2_EVENT,
        required_graph_pattern=(
            "H-H and related X-H connectivity changes (H2_EVENT node in the "
            "strong component)"
        ),
        preferred_drivers="H-H + 1-2 related X-H B",
        monitoring_targets=(
            "all former partners, H2 position/orientation, additional heavy-atom events"
        ),
        escalation_fallback=(STRATEGY_NETWORK_PATH,),
    ),
    STRATEGY_AROMATIC_COUPLED: StrategySpec(
        strategy_id=STRATEGY_AROMATIC_COUPLED,
        required_graph_pattern=(
            "regional electronic rearrangement (DELOCALIZED_REGION) "
            "accompanied by real connectivity/H/ring events; pure order "
            "changes without such events stay review-only"
        ),
        preferred_drivers=(
            "real connectivity at the region boundary or H; one explained A/D if needed"
        ),
        monitoring_targets=(
            "region bond-length distribution, planarity, electronic bond-order "
            "indicators, all boundary connections"
        ),
        escalation_fallback=(STRATEGY_NETWORK_PATH,),
    ),
    STRATEGY_MULTI_EVENT_CONNECTED: StrategySpec(
        strategy_id=STRATEGY_MULTI_EVENT_CONNECTED,
        required_graph_pattern=(
            "multiple events connected in one strong component with unknown "
            "timing (not explained by a more specific motif)"
        ),
        preferred_drivers="evidence-covered <=3 B/A/D combinations",
        monitoring_targets=(
            "all unconstrained events, especially leaving bonds / second H bonds"
        ),
        escalation_fallback=(STRATEGY_NETWORK_PATH,),
    ),
    STRATEGY_CONFORMATION_STEREO: StrategySpec(
        strategy_id=STRATEGY_CONFORMATION_STEREO,
        required_graph_pattern=(
            "no connectivity difference, or conformation is a clear preceding "
            "step (stereo/attribute target change without F/B edits)"
        ),
        preferred_drivers="A or D; chirality inversion needs geometric verification",
        monitoring_targets="target configuration, skeleton topology, non-target contacts",
        escalation_fallback=(STRATEGY_NETWORK_PATH,),
    ),
    STRATEGY_NETWORK_PATH: StrategySpec(
        strategy_id=STRATEGY_NETWORK_PATH,
        required_graph_pattern=(
            "multiple strong components, dense rearrangement, driver overflow, "
            "or no suitable local coordinate"
        ),
        preferred_drivers="dual-end image chain; never invent a single-bond driver",
        monitoring_targets="all events and endpoint identity",
        escalation_fallback=(),  # terminal strategy: review when endpoints/e-state fail
    ),
    STRATEGY_SPECIAL_DOMAIN: StrategySpec(
        strategy_id=STRATEGY_SPECIAL_DOMAIN,
        required_graph_pattern=(
            "coordination/metal, surface/periodic, electron exchange, or "
            "excited-state input needing special representation/methods"
        ),
        preferred_drivers="validated special-domain plugin/backend only",
        monitoring_targets="domain-relevant targets",
        escalation_fallback=(),  # explicit unsupported exit, never generic rules
    ),
}


# ---------------------------------------------------------------------------
# Routing result dataclasses.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """One ordered proposal-level strategy candidate (no coordinates yet)."""

    strategy_id: str
    rule_trace: tuple[str, ...]
    pattern_evidence: dict[str, Any]
    review_required: bool
    component_events: tuple[str, ...]
    labels: tuple[str, ...]
    escalation_fallback: tuple[str, ...]
    n_drivers_planned: int

    def to_record(self) -> dict[str, Any]:
        """JSON-safe projection (deterministic via ``stable_json_dumps``)."""
        return {
            "strategy_id": self.strategy_id,
            "rule_trace": list(self.rule_trace),
            "pattern_evidence": dict(self.pattern_evidence),
            "review_required": self.review_required,
            "component_events": list(self.component_events),
            "labels": list(self.labels),
            "escalation_fallback": list(self.escalation_fallback),
            "n_drivers_planned": self.n_drivers_planned,
        }


@dataclass(frozen=True, slots=True)
class RouteResult:
    """Exactly-one-disposition routing output for one mapped reaction."""

    schema_version: str
    outcome: str
    strategy_candidates: tuple[StrategyCandidate, ...]
    labels: tuple[str, ...]
    rule_trace: tuple[str, ...]
    rejection_code: str | None = None
    rejection_reason: str | None = None
    special_domain: dict[str, Any] | None = field(default=None)
    reaction_id: str | None = None

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection; never contains truth-derived keys."""
        return {
            "schema_version": self.schema_version,
            "reaction_id": self.reaction_id,
            "outcome": self.outcome,
            "rejection_code": self.rejection_code,
            "rejection_reason": self.rejection_reason,
            "special_domain": (
                None if self.special_domain is None else dict(self.special_domain)
            ),
            "labels": list(self.labels),
            "rule_trace": list(self.rule_trace),
            "strategy_candidates": [
                candidate.to_record() for candidate in self.strategy_candidates
            ],
        }

    def to_json(self) -> str:
        """Canonical serialization (determinism test surface)."""
        return stable_json_dumps(self.to_doc())


# ---------------------------------------------------------------------------
# Extension points — explicit no-ops, owning todos named in docstrings.
# ---------------------------------------------------------------------------
def apply_direction_hooks(
    candidate: StrategyCandidate, *, policy: Mapping[str, Any] | None = None
) -> StrategyCandidate:
    """Direction/assembly hook — owning todo: 15.

    No-op at proposal-routing stage (todo 11): direction, start endpoint,
    assembly id and anchor reason are decided by todo 15's layered direction
    policy over coordinate candidates.  The pending marker is recorded in the
    candidate ``rule_trace`` so downstream traces show the hook was invoked.
    """
    del policy
    if R_HOOK_DIRECTION_TODO15_PENDING in candidate.rule_trace:
        return candidate
    return _with_extra_trace(candidate, R_HOOK_DIRECTION_TODO15_PENDING)


def apply_schedule_hooks(
    candidate: StrategyCandidate, *, policy: Mapping[str, Any] | None = None
) -> StrategyCandidate:
    """Schedule hook — owning todo: 14.

    No-op at proposal-routing stage (todo 11): linear / event_A_early /
    event_A_late / smoothstep schedules and the finite candidate tree are
    generated by todo 14 from coordinate drivers that do not exist yet.
    """
    del policy
    if R_HOOK_SCHEDULE_TODO14_PENDING in candidate.rule_trace:
        return candidate
    return _with_extra_trace(candidate, R_HOOK_SCHEDULE_TODO14_PENDING)


def apply_budget_gate(
    candidates: Sequence[StrategyCandidate], *, policy: Mapping[str, Any] | None = None
) -> tuple[StrategyCandidate, ...]:
    """Budget gate — owning todo: 17.

    No-op at proposal-routing stage (todo 11): ``deduplicate_rank_and_budget``
    and release-gate semantics belong to todo 17's selector core.  The pending
    marker is appended to every candidate trace.
    """
    del policy
    gated: list[StrategyCandidate] = []
    for candidate in candidates:
        if R_HOOK_BUDGET_TODO17_PENDING in candidate.rule_trace:
            gated.append(candidate)
        else:
            gated.append(_with_extra_trace(candidate, R_HOOK_BUDGET_TODO17_PENDING))
    return tuple(gated)


def _with_extra_trace(candidate: StrategyCandidate, rule: str) -> StrategyCandidate:
    """Return *candidate* with *rule* appended to its trace (idempotent)."""
    return StrategyCandidate(
        strategy_id=candidate.strategy_id,
        rule_trace=(*candidate.rule_trace, rule),
        pattern_evidence=dict(candidate.pattern_evidence),
        review_required=candidate.review_required,
        component_events=candidate.component_events,
        labels=candidate.labels,
        escalation_fallback=candidate.escalation_fallback,
        n_drivers_planned=candidate.n_drivers_planned,
    )


# ---------------------------------------------------------------------------
# Internal routing helpers.
# ---------------------------------------------------------------------------
def _sorted_pairs(pairs: Sequence[Any]) -> tuple[tuple[int, int], ...]:
    normalized = sorted((int(a), int(b)) for a, b in pairs)
    return tuple(normalized)


def _elements_by_map(bundle: EndpointGraphBundle) -> dict[int, str]:
    elements: dict[int, str] = {}
    for node in bundle.r_graph.nodes:
        elements[int(node.map_id)] = str(node.element)
    for node in bundle.p_graph.nodes:
        elements.setdefault(int(node.map_id), str(node.element))
    return elements


def _non_organic_elements(bundle: EndpointGraphBundle) -> tuple[str, ...]:
    found = {
        element
        for element in _elements_by_map(bundle).values()
        if element not in ORGANIC_ELEMENTS
    }
    return tuple(sorted(found))


def _event_rule(event_type: str) -> str:
    return _EVENT_RULE_BY_TYPE.get(event_type, f"R_EVENT_{event_type}")


_RING_BASIS_SHARED_REGION: Final[str] = "shared_ring_region"


def _is_genuine_ring_event(event: EventNode) -> bool:
    """Distinguish real ring motifs from todo-8 singleton noise groups.

    ``build_endpoint_context`` (todo 8) attaches a singleton ring group with
    basis ``support_path_overlap`` to *every* support-path edit, so a plain
    single-bond formation also yields a ``RING_REORGANIZATION`` event node.
    Routing treats a ring event as a genuine ring motif only when it carries
    the ``shared_ring_region`` basis (edge lies in an endpoint ring region) or
    groups >=2 edits (multi-edit ring closure/opening).
    """
    if len(event.edit_pairs) >= 2:
        return True
    metadata = dict(event.metadata)
    return str(metadata.get("basis", "")) == _RING_BASIS_SHARED_REGION


def _kind_by_pair(edit_graph: ReactionEditGraph) -> dict[tuple[int, int], str]:
    kinds: dict[tuple[int, int], str] = {}
    for edit in edit_graph.edits:
        pair = (int(edit.pair[0]), int(edit.pair[1]))
        kinds[pair if pair[0] <= pair[1] else (pair[1], pair[0])] = str(
            edit.edit_kind
        )
    return kinds


def _candidate(
    strategy_id: str,
    *,
    rules: Sequence[str],
    evidence: Mapping[str, Any],
    review_required: bool,
    component_events: Sequence[str],
    labels: Sequence[str],
    n_drivers_planned: int,
) -> StrategyCandidate:
    spec = STRATEGY_REGISTRY[strategy_id]
    event_rules = tuple(
        _event_rule(label) for label in sorted(set(labels))
    )
    rule_trace = tuple(dict.fromkeys([*rules, *event_rules]))
    return StrategyCandidate(
        strategy_id=strategy_id,
        rule_trace=rule_trace,
        pattern_evidence=dict(evidence),
        review_required=review_required,
        component_events=tuple(component_events),
        labels=tuple(sorted(set(labels))),
        escalation_fallback=spec.escalation_fallback,
        n_drivers_planned=n_drivers_planned,
    )


def _match_component(
    *,
    component_events: Sequence[EventNode],
    component_pairs: tuple[tuple[int, int], ...],
    kind_by_pair: Mapping[tuple[int, int], str],
    elements: Mapping[int, str],
) -> StrategyCandidate | None:
    """Match one strong component to its primary §5.2 strategy.

    Returns ``None`` only when the component carries no events and no edits
    (defensive; the caller then emits the NETWORK_PATH fallback so the result
    is never silently empty).  Multi-label preservation: every event type in
    the component lands in ``labels`` and in the candidate rule trace.
    """
    types = {event.event_type for event in component_events}
    event_ids = tuple(event.event_id for event in component_events)
    genuine_ring_events = tuple(
        event
        for event in component_events
        if event.event_type == EVENT_RING and _is_genuine_ring_event(event)
    )
    # Labels carry genuine structural event types only; todo-8 singleton
    # support-path ring groups are membership noise, not §4.3 ring evidence.
    labels = {
        event.event_type
        for event in component_events
        if event.event_type != EVENT_CONTEXT_NEAR
        and (event.event_type != EVENT_RING or _is_genuine_ring_event(event))
    }
    kinds = [kind_by_pair.get(pair, "") for pair in component_pairs]
    fb_pairs = tuple(
        pair
        for pair, kind in zip(component_pairs, kinds, strict=True)
        if kind in (_KIND_FORMED, _KIND_BROKEN)
    )
    formed_pairs = tuple(p for p, k in zip(component_pairs, kinds, strict=True) if k == _KIND_FORMED)
    broken_pairs = tuple(p for p, k in zip(component_pairs, kinds, strict=True) if k == _KIND_BROKEN)
    n_fb = len(fb_pairs)
    heavy_fb = tuple(
        pair
        for pair in fb_pairs
        if elements.get(pair[0], _ELEMENT_H) != _ELEMENT_H
        and elements.get(pair[1], _ELEMENT_H) != _ELEMENT_H
    )
    heavy_types = {
        event.event_type
        for event in component_events
        if event.event_type
        not in {EVENT_H_TRANSFER, EVENT_CONTEXT_NEAR, EVENT_RING}
        or (
            event.event_type == EVENT_RING and _is_genuine_ring_event(event)
        )
    }
    n_genuine_rings = len(genuine_ring_events)

    evidence: dict[str, Any] = {
        "event_ids": list(event_ids),
        "event_types": sorted(types),
        "edit_pairs": [list(pair) for pair in component_pairs],
        "fb_pairs": [list(pair) for pair in fb_pairs],
    }

    def _build(
        strategy_id: str,
        rules: Sequence[str],
        *,
        review: bool,
        drivers: int,
    ) -> StrategyCandidate:
        return _candidate(
            strategy_id,
            rules=rules,
            evidence=evidence,
            review_required=review,
            component_events=event_ids,
            labels=labels,
            n_drivers_planned=drivers,
        )

    # --- specificity order (registry STRATEGY_ROUTING_ORDER) ---------------
    if EVENT_H2 in types:
        return _build(
            STRATEGY_H2_EVENT,
            (R_MOTIF_H2_EVENT,),
            review=True,
            drivers=_MAX_DRIVERS,
        )

    if EVENT_H_TRANSFER in types:
        n_h_transfer = sum(
            1 for event in component_events if event.event_type == EVENT_H_TRANSFER
        )
        if n_h_transfer >= 2 or heavy_types or heavy_fb:
            # Multiple coupled H transfers (shared centre) or H coupled to
            # heavy-atom events → the multi-event H strategy; both/all labels
            # stay preserved on the candidate (design §5: multi-label).
            return _build(
                STRATEGY_H_TRANSFER_COUPLED,
                (R_MOTIF_H_TRANSFER_COUPLED,),
                review=True,
                drivers=min(_MAX_DRIVERS, 2),
            )
        return _build(
            STRATEGY_H_TRANSFER,
            (R_MOTIF_H_TRANSFER_PURE,),
            review=False,
            drivers=1,
        )

    # Connectivity exchange: formed+broken sharing a non-H atom.
    exchange_shared = False
    for formed in formed_pairs:
        for broken in broken_pairs:
            shared = {formed[0], formed[1]} & {broken[0], broken[1]}
            if shared and all(
                elements.get(atom, _ELEMENT_H) != _ELEMENT_H for atom in shared
            ):
                exchange_shared = True
                break
        if exchange_shared:
            break
    if exchange_shared:
        return _build(
            STRATEGY_CONNECTIVITY_EXCHANGE,
            (R_MOTIF_CONNECTIVITY_EXCHANGE,),
            review=False,
            drivers=2,
        )

    if genuine_ring_events:
        other_heavy = {
            event.event_type
            for event in component_events
            if event.event_type
            not in {
                EVENT_RING,
                EVENT_DELOCALIZED,
                EVENT_SHARED_CENTER,
                EVENT_CONTEXT_NEAR,
            }
        }
        if n_fb >= 2 or other_heavy or n_genuine_rings >= 2:
            ring_rules: list[str] = [R_MOTIF_RING_COUPLED]
            if formed_pairs and not broken_pairs:
                ring_rules.append(R_MOTIF_RING_CLOSURE)
            elif broken_pairs and not formed_pairs:
                ring_rules.append(R_MOTIF_RING_OPENING)
            return _build(
                STRATEGY_RING_COUPLED,
                ring_rules,
                review=False,
                drivers=min(_MAX_DRIVERS, max(1, n_fb)),
            )
        # Single-bond ring closure/opening → LOCAL_CONNECTIVITY (design §5.2:
        # "环开闭也可适用") with the ring rule preserved in the trace.
        ring_rules = [R_MOTIF_LOCAL_CONNECTIVITY_FORMED if formed_pairs else R_MOTIF_LOCAL_CONNECTIVITY_BROKEN]
        if formed_pairs:
            ring_rules.append(R_MOTIF_RING_CLOSURE)
        else:
            ring_rules.append(R_MOTIF_RING_OPENING)
        return _build(
            STRATEGY_LOCAL_CONNECTIVITY,
            ring_rules,
            review=False,
            drivers=1,
        )

    if EVENT_DELOCALIZED in types:
        real_types = {
            event.event_type
            for event in component_events
            if event.event_type
            in {EVENT_H_TRANSFER, EVENT_H2, EVENT_CONNECTIVITY_EXCHANGE}
            or (
                event.event_type == EVENT_RING and _is_genuine_ring_event(event)
            )
        }
        if real_types:
            return _build(
                STRATEGY_AROMATIC_COUPLED,
                (R_MOTIF_AROMATIC_WITH_EVENTS,),
                review=False,
                drivers=2,
            )
        return _build(
            STRATEGY_AROMATIC_COUPLED,
            (R_MOTIF_AROMATIC_ORDER_ONLY,),
            review=True,
            drivers=1,
        )

    if n_fb == 1:
        rule = (
            R_MOTIF_LOCAL_CONNECTIVITY_FORMED
            if formed_pairs
            else R_MOTIF_LOCAL_CONNECTIVITY_BROKEN
        )
        return _build(
            STRATEGY_LOCAL_CONNECTIVITY,
            (rule,),
            review=False,
            drivers=1,
        )

    if EVENT_SHARED_CENTER in types or len(labels) >= 2 or n_fb >= 2:
        return _build(
            STRATEGY_MULTI_EVENT_CONNECTED,
            (R_MOTIF_MULTI_EVENT_CONNECTED,),
            review=True,
            drivers=min(_MAX_DRIVERS, max(2, n_fb)),
        )

    return None


# ---------------------------------------------------------------------------
# Public routing API.
# ---------------------------------------------------------------------------
def route_strategies(
    *,
    bundle: EndpointGraphBundle,
    edit_graph: ReactionEditGraph,
    context: EndpointContext,
    coupling: EventCouplingGraph,
    reaction_id: str | None = None,
    policy: Mapping[str, Any] | None = None,
    stratum: str | None = None,
) -> RouteResult:
    """Route one mapped reaction to ordered §5.2 strategy candidates.

    Keyword-only on purpose: the P0 layers are explicit dependencies, and the
    rejected ``stratum`` keyword documents the hard no-stratum-dispatch rule
    (design §5.1 pseudocode comment "不按 stratum 分派").

    Returns a :class:`RouteResult` whose ``outcome`` is exactly one of
    ``executable_candidate_set`` / ``needs_review`` / ``special_domain_exit``
    / ``typed_rejection``.  Raises ``ValueError("STRATUM_NOT_SUPPORTED...")``
    when ``stratum`` is not ``None``.
    """
    if stratum is not None:
        raise ValueError(
            f"{STRATUM_NOT_SUPPORTED}: motif matching is graph-structural only; "
            "strata are cohort labels (G0/G1) and never routing inputs "
            "(design §5.1)"
        )

    trace: list[str] = []
    labels: set[str] = set()

    def _reject(code: str, reason: str) -> RouteResult:
        return RouteResult(
            schema_version=SCHEMA_ROUTE_RESULT,
            outcome=OUTCOME_TYPED_REJECTION,
            strategy_candidates=(),
            labels=tuple(sorted(labels)),
            rule_trace=tuple(trace),
            rejection_code=code,
            rejection_reason=reason,
            special_domain=None,
            reaction_id=reaction_id,
        )

    def _special(code: str, reason: str, representations: tuple[str, ...],
                 methods: tuple[str, ...]) -> RouteResult:
        return RouteResult(
            schema_version=SCHEMA_ROUTE_RESULT,
            outcome=OUTCOME_SPECIAL_DOMAIN_EXIT,
            strategy_candidates=(),
            labels=tuple(sorted(labels)),
            rule_trace=tuple(trace),
            rejection_code=code,
            rejection_reason=reason,
            special_domain={
                "code": code,
                "reason": reason,
                "required_representations": list(representations),
                "required_methods": list(methods),
            },
            reaction_id=reaction_id,
        )

    # ------------------------------------------------------------------
    # §5.1 step 1: watershed / domain errors first.
    # ------------------------------------------------------------------
    conservation = bundle.conservation
    if not conservation.element_conserved:
        trace.append(R_WATERSHED_ELEMENT_CONSERVATION)
        return _reject(
            REJ_ELEMENT_IMBALANCE,
            "R/P element inventory differs per map id; not a scan-routing input",
        )
    if not conservation.isotope_conserved:
        trace.append(R_WATERSHED_ISOTOPE_CONSERVATION)
        return _reject(
            REJ_ISOTOPE_IMBALANCE,
            "R/P isotope inventory differs per map id; not a scan-routing input",
        )

    non_organic = _non_organic_elements(bundle)
    if non_organic:
        trace.append(R_WATERSHED_SPECIAL_DOMAIN_ELEMENT)
        return _special(
            SPEC_DOMAIN,
            (
                "elements outside the conservative organic set "
                f"{sorted(ORGANIC_ELEMENTS)}: {list(non_organic)}; "
                "special-domain routing required"
            ),
            _SPECIAL_DOMAIN_REPRESENTATIONS,
            _SPECIAL_DOMAIN_METHODS,
        )

    r_features, p_features = context.endpoints
    r_charge, p_charge = r_features.charge, p_features.charge
    r_mult, p_mult = r_features.multiplicity, p_features.multiplicity
    charge_differs = (
        r_charge is not None and p_charge is not None and r_charge != p_charge
    )
    mult_differs = (
        r_mult is not None and p_mult is not None and r_mult != p_mult
    )
    if charge_differs or mult_differs:
        trace.append(R_WATERSHED_SPECIAL_ELECTRONIC_STATE)
        return _special(
            SPEC_ELECTRONIC_STATE,
            (
                f"endpoint electronic state differs (charge R={r_charge}/P={p_charge}, "
                f"multiplicity R={r_mult}/P={p_mult}); dedicated electronic-state flow"
            ),
            _SPECIAL_ELECTRONIC_REPRESENTATIONS,
            _SPECIAL_ELECTRONIC_METHODS,
        )

    # ------------------------------------------------------------------
    # §5.1 step 2: NO_REACTION_CHANGE.
    # ------------------------------------------------------------------
    edits = edit_graph.edits
    attr_changes = edit_graph.atom_attribute_changes
    has_stereo_attr = any(
        (record.get("R") or {}).get("chiral_tag")
        != (record.get("P") or {}).get("chiral_tag")
        for record in attr_changes
    )
    if not edits and not attr_changes:
        trace.append(R_NO_REACTION_CHANGE)
        return _reject(
            REJ_NO_REACTION_CHANGE,
            "no connectivity/electronic/stereo/conformational target change",
        )

    # ------------------------------------------------------------------
    # §5.1 steps 3-4: event construction consumed from todo 9 H; then the
    # no-connectivity-change branch (pure order changes → delocalized).
    # ------------------------------------------------------------------
    kind_by_pair = _kind_by_pair(edit_graph)
    elements = _elements_by_map(bundle)
    fb_edits = [
        pair
        for pair, kind in kind_by_pair.items()
        if kind in (_KIND_FORMED, _KIND_BROKEN)
    ]
    membership = coupling.membership_map()
    events_by_id = {event.event_id: event for event in coupling.events}

    if not fb_edits:
        delocalized_events = coupling.events_by_type(EVENT_DELOCALIZED)
        aromatic_delocalized = [
            event
            for event in delocalized_events
            if any(
                kind_by_pair.get(pair) == _KIND_ORDER
                and _edit_aromatic_region(edit_graph, pair) is not None
                for pair in event.edit_pairs
            )
        ]
        if aromatic_delocalized:
            region_ids = sorted(
                {
                    region
                    for event in aromatic_delocalized
                    for pair in event.edit_pairs
                    if (region := _edit_aromatic_region(edit_graph, pair)) is not None
                }
            )
            for event in aromatic_delocalized:
                labels.add(EVENT_DELOCALIZED)
            trace.append(R_BRANCH_DELOCALIZED_ORDER_ONLY)
            candidate = _candidate(
                STRATEGY_AROMATIC_COUPLED,
                rules=(R_BRANCH_DELOCALIZED_ORDER_ONLY, R_MOTIF_AROMATIC_ORDER_ONLY),
                evidence={
                    "branch": "no_connectivity_change",
                    "order_changed_pairs": [
                        list(pair)
                        for pair, kind in sorted(kind_by_pair.items())
                        if kind == _KIND_ORDER
                    ],
                    "aromatic_region_ids": region_ids,
                    "event_ids": [e.event_id for e in aromatic_delocalized],
                },
                review_required=True,
                component_events=tuple(e.event_id for e in aromatic_delocalized),
                labels=labels,
                n_drivers_planned=1,
            )
            candidates = _finalize((candidate,), trace, policy)
            return _finish(candidates, trace, labels, reaction_id)
        if has_stereo_attr:
            trace.append(R_BRANCH_CONFORMATION_STEREO)
            candidate = _candidate(
                STRATEGY_CONFORMATION_STEREO,
                rules=(R_BRANCH_CONFORMATION_STEREO, R_MOTIF_CONFORMATION_STEREO),
                evidence={
                    "branch": "no_connectivity_change",
                    "stereo_attribute_changes": [
                        dict(record) for record in attr_changes
                    ],
                },
                review_required=True,
                component_events=(),
                labels=labels,
                n_drivers_planned=1,
            )
            candidates = _finalize((candidate,), trace, policy)
            return _finish(candidates, trace, labels, reaction_id)
        trace.append(R_BRANCH_O_NETWORK_REVIEW)
        candidate = _candidate(
            STRATEGY_NETWORK_PATH,
            rules=(R_BRANCH_O_NETWORK_REVIEW, R_MOTIF_NETWORK_PATH_FALLBACK),
            evidence={
                "branch": "no_connectivity_change",
                "order_changed_pairs": [
                    list(pair)
                    for pair, kind in sorted(kind_by_pair.items())
                    if kind == _KIND_ORDER
                ],
                "reason": "O network without interpretable local geometric driver",
            },
            review_required=True,
            component_events=(),
            labels=labels,
            n_drivers_planned=0,
        )
        candidates = _finalize((candidate,), trace, policy)
        return _finish(candidates, trace, labels, reaction_id)

    # ------------------------------------------------------------------
    # §5.1 step 5: per-event-component routing.
    # ------------------------------------------------------------------
    component_candidates: list[StrategyCandidate] = []
    matched_any = False
    for component in coupling.strong_components:
        if not component:
            continue
        component_pairs_set: set[tuple[int, int]] = set()
        for pair, owners in membership.items():
            if set(owners) & set(component):
                component_pairs_set.add(pair)
        component_pairs = tuple(sorted(component_pairs_set))
        component_event_nodes = tuple(
            events_by_id[event_id]
            for event_id in component
            if event_id in events_by_id
        )
        for event in component_event_nodes:
            if event.event_type == EVENT_CONTEXT_NEAR:
                continue
            if event.event_type == EVENT_RING and not _is_genuine_ring_event(event):
                continue
            labels.add(event.event_type)
        matched = _match_component(
            component_events=component_event_nodes,
            component_pairs=component_pairs,
            kind_by_pair=kind_by_pair,
            elements=elements,
        )
        if matched is None:
            continue
        matched_any = True
        component_candidates.append(matched)

    if len(coupling.strong_components) >= 2:
        trace.append(R_MOTIF_NETWORK_PATH_MULTI_COMPONENT)
        multi_labels = set(labels)
        component_candidates.append(
            _candidate(
                STRATEGY_NETWORK_PATH,
                rules=(R_MOTIF_NETWORK_PATH_MULTI_COMPONENT,),
                evidence={
                    "n_strong_components": len(coupling.strong_components),
                    "strong_components": [
                        list(component) for component in coupling.strong_components
                    ],
                },
                review_required=True,
                component_events=tuple(
                    sorted({e for c in coupling.strong_components for e in c})
                ),
                labels=multi_labels,
                n_drivers_planned=0,
            )
        )

    if not matched_any and not component_candidates:
        # Defensive: edits exist but no component matched — never silent empty.
        trace.append(R_MOTIF_NETWORK_PATH_FALLBACK)
        component_candidates.append(
            _candidate(
                STRATEGY_NETWORK_PATH,
                rules=(R_MOTIF_NETWORK_PATH_FALLBACK,),
                evidence={
                    "reason": "no local motif matched the routed components",
                    "fb_pairs": [list(pair) for pair in sorted(fb_edits)],
                },
                review_required=True,
                component_events=(),
                labels=labels,
                n_drivers_planned=0,
            )
        )

    for candidate in component_candidates:
        trace.extend(
            rule
            for rule in candidate.rule_trace
            if rule.startswith("R_MOTIF_") and rule not in trace
        )

    candidates = _finalize(component_candidates, trace, policy)
    return _finish(candidates, trace, labels, reaction_id)


def _edit_aromatic_region(
    edit_graph: ReactionEditGraph, pair: tuple[int, int]
) -> str | None:
    """Return the aromatic-region id of one edit pair, or ``None``."""
    for edit in edit_graph.edits:
        edit_pair = (int(edit.pair[0]), int(edit.pair[1]))
        if edit_pair == pair or edit_pair == (pair[1], pair[0]):
            region = edit.aromatic_region
            return None if region is None else str(region)
    return None


def _finalize(
    candidates: Sequence[StrategyCandidate],
    trace: list[str],
    policy: Mapping[str, Any] | None,
) -> tuple[StrategyCandidate, ...]:
    """Apply todo-14/15/17 extension hooks (no-ops) in fixed order."""
    directed = tuple(
        apply_direction_hooks(candidate, policy=policy) for candidate in candidates
    )
    scheduled = tuple(
        apply_schedule_hooks(candidate, policy=policy) for candidate in directed
    )
    gated = apply_budget_gate(scheduled, policy=policy)
    for marker in (
        R_HOOK_DIRECTION_TODO15_PENDING,
        R_HOOK_SCHEDULE_TODO14_PENDING,
        R_HOOK_BUDGET_TODO17_PENDING,
    ):
        if marker not in trace:
            trace.append(marker)
    # Deterministic order: routing-order rank, then component min event id,
    # then strategy id (stable map/event ordering breaks technical ties).
    return tuple(
        sorted(
            gated,
            key=lambda c: (
                _STRATEGY_RANK.get(c.strategy_id, len(STRATEGY_ROUTING_ORDER)),
                c.component_events,
                c.strategy_id,
            ),
        )
    )


def _finish(
    candidates: tuple[StrategyCandidate, ...],
    trace: Sequence[str],
    labels: set[str],
    reaction_id: str | None,
) -> RouteResult:
    """Assign the exactly-one disposition from the routed candidate set."""
    all_labels = set(labels)
    for candidate in candidates:
        all_labels.update(candidate.labels)
    if any(not candidate.review_required for candidate in candidates):
        outcome = OUTCOME_EXECUTABLE_CANDIDATE_SET
    else:
        outcome = OUTCOME_NEEDS_REVIEW
    return RouteResult(
        schema_version=SCHEMA_ROUTE_RESULT,
        outcome=outcome,
        strategy_candidates=candidates,
        labels=tuple(sorted(all_labels)),
        rule_trace=tuple(trace),
        rejection_code=None,
        rejection_reason=None,
        special_domain=None,
        reaction_id=reaction_id,
    )


__all__ = [
    "ORGANIC_ELEMENTS",
    "OUTCOME_EXECUTABLE_CANDIDATE_SET",
    "OUTCOME_NEEDS_REVIEW",
    "OUTCOME_SPECIAL_DOMAIN_EXIT",
    "OUTCOME_TYPED_REJECTION",
    "OUTCOMES",
    "R_BRANCH_CONFORMATION_STEREO",
    "R_BRANCH_DELOCALIZED_ORDER_ONLY",
    "R_BRANCH_O_NETWORK_REVIEW",
    "R_HOOK_BUDGET_TODO17_PENDING",
    "R_HOOK_DIRECTION_TODO15_PENDING",
    "R_HOOK_SCHEDULE_TODO14_PENDING",
    "R_MOTIF_AROMATIC_ORDER_ONLY",
    "R_MOTIF_AROMATIC_WITH_EVENTS",
    "R_MOTIF_CONNECTIVITY_EXCHANGE",
    "R_MOTIF_CONFORMATION_STEREO",
    "R_MOTIF_H2_EVENT",
    "R_MOTIF_H_TRANSFER_COUPLED",
    "R_MOTIF_H_TRANSFER_PURE",
    "R_MOTIF_LOCAL_CONNECTIVITY_BROKEN",
    "R_MOTIF_LOCAL_CONNECTIVITY_FORMED",
    "R_MOTIF_MULTI_EVENT_CONNECTED",
    "R_MOTIF_NETWORK_PATH_FALLBACK",
    "R_MOTIF_NETWORK_PATH_MULTI_COMPONENT",
    "R_MOTIF_RING_CLOSURE",
    "R_MOTIF_RING_COUPLED",
    "R_MOTIF_RING_OPENING",
    "R_NO_REACTION_CHANGE",
    "R_WATERSHED_ELEMENT_CONSERVATION",
    "R_WATERSHED_ISOTOPE_CONSERVATION",
    "R_WATERSHED_SPECIAL_DOMAIN_ELEMENT",
    "R_WATERSHED_SPECIAL_ELECTRONIC_STATE",
    "REJ_ELEMENT_IMBALANCE",
    "REJ_ISOTOPE_IMBALANCE",
    "REJ_NO_REACTION_CHANGE",
    "SCHEMA_ROUTE_RESULT",
    "SPEC_DOMAIN",
    "SPEC_ELECTRONIC_STATE",
    "STRATEGY_AROMATIC_COUPLED",
    "STRATEGY_CONNECTIVITY_EXCHANGE",
    "STRATEGY_CONFORMATION_STEREO",
    "STRATEGY_H2_EVENT",
    "STRATEGY_H_TRANSFER",
    "STRATEGY_H_TRANSFER_COUPLED",
    "STRATEGY_IDS",
    "STRATEGY_LOCAL_CONNECTIVITY",
    "STRATEGY_MULTI_EVENT_CONNECTED",
    "STRATEGY_NETWORK_PATH",
    "STRATEGY_REGISTRY",
    "STRATEGY_RING_COUPLED",
    "STRATEGY_ROUTING_ORDER",
    "STRATEGY_SPECIAL_DOMAIN",
    "STRATUM_NOT_SUPPORTED",
    "RouteResult",
    "StrategyCandidate",
    "StrategySpec",
    "apply_budget_gate",
    "apply_direction_hooks",
    "apply_schedule_hooks",
    "route_strategies",
]
