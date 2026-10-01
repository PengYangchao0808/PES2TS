"""Special-domain exits + electronic-state routing (plan todo 29).

Full detail layer over todo 11's ``registry`` special-domain exit codes
(design §3.2 / §5.2 / §12 of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``):

- ``SPECIAL_DOMAIN`` subtypes: ``coordination_metal``, ``surface_periodic``,
  ``electron_exchange``, ``excited_state``.  Each returns a typed
  :class:`SpecialDomainExit` carrying required representations, required
  method families, why the generic selector is invalid, and a pointer to a
  future specialized module (exits only — specialized methods are **not**
  implemented here).
- ``SPECIAL_ELECTRONIC_STATE_REQUIRED`` sub-exits: total-charge difference,
  particle-number difference, cross-spin-surface, excited-state crossing.
  Each names requirement **lists** (typed :class:`Requirement` records), never
  implementations.
- Ordinary NEB is **not** a universal fallback for any of these: routing a
  special-domain/electronic-state input to ordinary NEB is refused with a
  typed :class:`NebRefusal`.
- Generic covalent-radius rules must **not** be applied to metals or any
  non-organic element (:func:`generic_covalent_radius_sum` refuses).
- Coverage declaration records that Demo24 has evidence only for C/H/N/O/F/Cl;
  metals and every special domain have **no** evidence (design §12).

Element classification is a single source: ``registry.ORGANIC_ELEMENTS`` is
imported, never redefined.  This module only *reads* the todo-11
:class:`RouteResult`; it never modifies ``registry`` and never emits generic
strategy candidates.
"""

# allow: SIZE_OK — plan-named todo-29 single detail-layer module (four
# SPECIAL_DOMAIN subtypes + four electronic-state sub-exits + typed
# requirement records + NEB/covalent guards + coverage declaration);
# precedent registry.py / plan_freeze.py / path_request.py.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle
from pes2ts_core.scan_strategy.geometry_feasibility import covalent_radius_sum
from pes2ts_core.scan_strategy.registry import (
    ORGANIC_ELEMENTS,
    OUTCOME_SPECIAL_DOMAIN_EXIT,
    SPEC_DOMAIN,
    SPEC_ELECTRONIC_STATE,
    RouteResult,
)
from pes2ts_core.utils.hashing import stable_json_dumps

# ---------------------------------------------------------------------------
# Schema vocabulary.
# ---------------------------------------------------------------------------
SCHEMA_SPECIAL_DOMAIN_EXIT: Final[str] = "g1_special_domain_exit_v1"
SCHEMA_NEB_REFUSAL: Final[str] = "g1_ordinary_neb_refusal_v1"
SCHEMA_COVERAGE_DECLARATION: Final[str] = "g1_special_domain_coverage_v1"

# ---------------------------------------------------------------------------
# SPECIAL_DOMAIN subtypes (design §5.2 first registry row, closed set).
# ---------------------------------------------------------------------------
SUBTYPE_COORDINATION_METAL: Final[str] = "coordination_metal"
SUBTYPE_SURFACE_PERIODIC: Final[str] = "surface_periodic"
SUBTYPE_ELECTRON_EXCHANGE: Final[str] = "electron_exchange"
SUBTYPE_EXCITED_STATE: Final[str] = "excited_state"
SPECIAL_DOMAIN_SUBTYPES: Final[tuple[str, ...]] = (
    SUBTYPE_COORDINATION_METAL,
    SUBTYPE_SURFACE_PERIODIC,
    SUBTYPE_ELECTRON_EXCHANGE,
    SUBTYPE_EXCITED_STATE,
)

# ---------------------------------------------------------------------------
# SPECIAL_ELECTRONIC_STATE_REQUIRED sub-exits (design §3.2, closed set).
# ---------------------------------------------------------------------------
ESUB_TOTAL_CHARGE_DIFFERENCE: Final[str] = "total_charge_difference"
ESUB_PARTICLE_NUMBER_DIFFERENCE: Final[str] = "particle_number_difference"
ESUB_CROSS_SPIN_SURFACE: Final[str] = "cross_spin_surface"
ESUB_EXCITED_STATE_CROSSING: Final[str] = "excited_state_crossing"
SPECIAL_ELECTRONIC_STATE_SUBTYPES: Final[tuple[str, ...]] = (
    ESUB_TOTAL_CHARGE_DIFFERENCE,
    ESUB_PARTICLE_NUMBER_DIFFERENCE,
    ESUB_CROSS_SPIN_SURFACE,
    ESUB_EXCITED_STATE_CROSSING,
)

# ---------------------------------------------------------------------------
# Guard codes (ordinary NEB / covalent-radius).
# ---------------------------------------------------------------------------
ORDINARY_NEB_REFUSED_SPECIAL_DOMAIN: Final[str] = (
    "ORDINARY_NEB_REFUSED_SPECIAL_DOMAIN"
)
ORDINARY_NEB_REFUSED_ELECTRONIC_STATE: Final[str] = (
    "ORDINARY_NEB_REFUSED_ELECTRONIC_STATE"
)
COVALENT_RADIUS_RULE_FORBIDDEN_FOR_METAL: Final[str] = (
    "COVALENT_RADIUS_RULE_FORBIDDEN_FOR_METAL"
)

ATTEMPTED_METHOD_ORDINARY_NEB: Final[str] = "ordinary_NEB"
#: Explicit statement carried on every exit (design §3.2): ordinary NEB is
#: never a universal fallback for special domains or electronic-state cases.
ORDINARY_NEB_NOT_UNIVERSAL_FALLBACK: Final[str] = (
    "ordinary NEB is not a universal fallback for this domain/electronic state"
)

# Requirement-record categories (data, not prose).
CATEGORY_REPRESENTATION: Final[str] = "representation"
CATEGORY_METHOD_FAMILY: Final[str] = "method_family"

# Future specialized modules — pointers only; nothing is implemented here.
POINTER_METAL_COORDINATION: Final[str] = (
    "pes2ts_core/scan_strategy/specialized/metal_coordination.py (future)"
)
POINTER_SURFACE_PERIODIC: Final[str] = (
    "pes2ts_core/scan_strategy/specialized/surface_periodic.py (future)"
)
POINTER_ELECTRON_EXCHANGE: Final[str] = (
    "pes2ts_core/scan_strategy/specialized/electron_exchange.py (future)"
)
POINTER_EXCITED_STATE: Final[str] = (
    "pes2ts_core/scan_strategy/specialized/excited_state.py (future)"
)
POINTER_ELECTRONIC_STATE_FLOW: Final[str] = (
    "pes2ts_core/scan_strategy/specialized/electronic_state_flow.py (future)"
)

#: Why the generic §5.2 selector + ordinary NEB cannot serve these domains.
WHY_METAL_INVALID: Final[str] = (
    "generic covalent-radius bond rules and single-surface NEB are invalid "
    "for coordination/metal systems; bonding, spin, and relativistic effects "
    "need a validated special-domain backend (design §5.2 SPECIAL_DOMAIN)"
)
WHY_SURFACE_INVALID: Final[str] = (
    "finite-molecule graph routing and ordinary NEB are invalid for "
    "surface/periodic models; periodic boundary conditions and slab "
    "representations are required (design §5.2 SPECIAL_DOMAIN)"
)
WHY_ELECTRON_EXCHANGE_INVALID: Final[str] = (
    "single-reference single-surface routing is invalid for explicit "
    "electron-exchange systems; spin-adapted electronic-state treatment is "
    "required (design §5.2 SPECIAL_DOMAIN / §3.2)"
)
WHY_EXCITED_STATE_INVALID: Final[str] = (
    "ground-state scan routing and ordinary NEB are invalid for "
    "excited-state crossings; state-specific methods and seam location are "
    "required (design §3.2 / §5.2)"
)
WHY_CHARGE_INVALID: Final[str] = (
    "total-charge difference between endpoints makes an ordinary neutral "
    "single-surface scan invalid; a dedicated charged-species flow is "
    "required (design §3.2)"
)
WHY_PARTICLE_INVALID: Final[str] = (
    "particle-number difference between endpoints makes ordinary NEB "
    "invalid; an electron attachment/detachment workflow is required "
    "(design §3.2)"
)
WHY_SPIN_INVALID: Final[str] = (
    "cross-spin-surface endpoints make ordinary single-surface NEB invalid; "
    "multi-state/spin-flip/MECP-class methods are required (design §3.2)"
)
WHY_EXCITED_CROSSING_INVALID: Final[str] = (
    "excited-state crossing endpoints make ground-state NEB invalid; "
    "excited-state methods and seam optimization are required (design §3.2)"
)


# ---------------------------------------------------------------------------
# Typed requirement records (data, not prose).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Requirement:
    """One required representation or method family (typed record)."""

    requirement_id: str
    category: str
    label: str
    detail: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "category": self.category,
            "label": self.label,
            "detail": self.detail,
        }


def _rep(req_id: str, label: str, detail: str = "") -> Requirement:
    return Requirement(
        requirement_id=req_id,
        category=CATEGORY_REPRESENTATION,
        label=label,
        detail=detail,
    )


def _mth(req_id: str, label: str, detail: str = "") -> Requirement:
    return Requirement(
        requirement_id=req_id,
        category=CATEGORY_METHOD_FAMILY,
        label=label,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Special-domain catalog (requirement lists as typed records).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SpecialDomainSpec:
    """Catalog row for one special-domain subtype or electronic sub-exit."""

    subtype: str
    code: str
    required_representations: tuple[Requirement, ...]
    required_methods: tuple[Requirement, ...]
    why_generic_selector_invalid: str
    future_module_pointer: str

    def to_doc(self) -> dict[str, Any]:
        return {
            "subtype": self.subtype,
            "code": self.code,
            "required_representations": [
                req.to_record() for req in self.required_representations
            ],
            "required_methods": [req.to_record() for req in self.required_methods],
            "why_generic_selector_invalid": self.why_generic_selector_invalid,
            "future_module_pointer": self.future_module_pointer,
        }


SPECIAL_DOMAIN_SPECS: Final[dict[str, SpecialDomainSpec]] = {
    SUBTYPE_COORDINATION_METAL: SpecialDomainSpec(
        subtype=SUBTYPE_COORDINATION_METAL,
        code=SPEC_DOMAIN,
        required_representations=(
            _rep(
                "rep_coordination_bonds",
                "coordination/metal-aware molecular representation",
                "explicit metal–ligand bond types; oxidation/spin ledger",
            ),
            _rep(
                "rep_basis_treatment",
                "metal-appropriate basis/ECP and correlated treatment",
                "named as requirement; never implied by generic organic defaults",
            ),
        ),
        required_methods=(
            _mth(
                "mth_validated_special_backend",
                "validated special-domain backend/plugin for metal systems",
                "capability must be probed; unknown = unsupported",
            ),
            _mth(
                "mth_no_generic_covalent_radius",
                "generic covalent-radius bond rules forbidden for metals",
                "design §5.2 SPECIAL_DOMAIN escalation column",
            ),
        ),
        why_generic_selector_invalid=WHY_METAL_INVALID,
        future_module_pointer=POINTER_METAL_COORDINATION,
    ),
    SUBTYPE_SURFACE_PERIODIC: SpecialDomainSpec(
        subtype=SUBTYPE_SURFACE_PERIODIC,
        code=SPEC_DOMAIN,
        required_representations=(
            _rep(
                "rep_periodic_cell",
                "periodic/slab representation (cell, k-points, vacuum)",
                "finite-molecule graph routing does not encode periodicity",
            ),
            _rep(
                "rep_surface_site",
                "surface site and coverage representation",
                "adsorption site identity is part of the computational input",
            ),
        ),
        required_methods=(
            _mth(
                "mth_periodic_backend",
                "validated periodic/slab backend for surface models",
                "capability must be probed; unknown = unsupported",
            ),
        ),
        why_generic_selector_invalid=WHY_SURFACE_INVALID,
        future_module_pointer=POINTER_SURFACE_PERIODIC,
    ),
    SUBTYPE_ELECTRON_EXCHANGE: SpecialDomainSpec(
        subtype=SUBTYPE_ELECTRON_EXCHANGE,
        code=SPEC_DOMAIN,
        required_representations=(
            _rep(
                "rep_spin_adapted",
                "spin-adapted electronic-state representation for electron exchange",
                "charge/spin ledger per endpoint; not reducible to bond edits",
            ),
        ),
        required_methods=(
            _mth(
                "mth_electron_exchange_methods",
                "validated electron-exchange method family (named requirement)",
                "e.g. multi-state / constrained-DFT-class treatments as requirements",
            ),
        ),
        why_generic_selector_invalid=WHY_ELECTRON_EXCHANGE_INVALID,
        future_module_pointer=POINTER_ELECTRON_EXCHANGE,
    ),
    SUBTYPE_EXCITED_STATE: SpecialDomainSpec(
        subtype=SUBTYPE_EXCITED_STATE,
        code=SPEC_DOMAIN,
        required_representations=(
            _rep(
                "rep_excited_state",
                "excited-state representation (state character, multiplicity)",
                "state identity is computational input, not a scan coordinate",
            ),
        ),
        required_methods=(
            _mth(
                "mth_excited_state_methods",
                "validated excited-state method family (named requirement)",
                "e.g. TD-DFT/CASSCF-class treatments as requirements",
            ),
        ),
        why_generic_selector_invalid=WHY_EXCITED_STATE_INVALID,
        future_module_pointer=POINTER_EXCITED_STATE,
    ),
}

SPECIAL_ELECTRONIC_STATE_SPECS: Final[dict[str, SpecialDomainSpec]] = {
    ESUB_TOTAL_CHARGE_DIFFERENCE: SpecialDomainSpec(
        subtype=ESUB_TOTAL_CHARGE_DIFFERENCE,
        code=SPEC_ELECTRONIC_STATE,
        required_representations=(
            _rep(
                "rep_charge_ledger",
                "equal particle-number endpoints with explicit total-charge ledger",
                "R/P total charges recorded per endpoint; difference is typed",
            ),
        ),
        required_methods=(
            _mth(
                "mth_constrained_dft_family",
                "constrained-DFT / charged-species method family (requirement list)",
                "named as requirement; no implementation in this module",
            ),
            _mth(
                "mth_charged_endpoint_preparation",
                "charged-endpoint preparation and charge-conserving review",
            ),
        ),
        why_generic_selector_invalid=WHY_CHARGE_INVALID,
        future_module_pointer=POINTER_ELECTRONIC_STATE_FLOW,
    ),
    ESUB_PARTICLE_NUMBER_DIFFERENCE: SpecialDomainSpec(
        subtype=ESUB_PARTICLE_NUMBER_DIFFERENCE,
        code=SPEC_ELECTRONIC_STATE,
        required_representations=(
            _rep(
                "rep_particle_number_ledger",
                "particle-number (electron-count) ledger per endpoint",
                "explicit electron counts; difference is typed, never silent",
            ),
        ),
        required_methods=(
            _mth(
                "mth_particle_number_workflow",
                "electron attachment/detachment workflow family (requirement list)",
                "named as requirement; no implementation in this module",
            ),
            _mth(
                "mth_multireference_open_shell",
                "multi-reference/open-shell treatment when character requires it",
            ),
        ),
        why_generic_selector_invalid=WHY_PARTICLE_INVALID,
        future_module_pointer=POINTER_ELECTRONIC_STATE_FLOW,
    ),
    ESUB_CROSS_SPIN_SURFACE: SpecialDomainSpec(
        subtype=ESUB_CROSS_SPIN_SURFACE,
        code=SPEC_ELECTRONIC_STATE,
        required_representations=(
            _rep(
                "rep_spin_surfaces",
                "cross-spin-surface endpoints (distinct multiplicities)",
                "R/P multiplicities recorded; ordinary single-surface NEB invalid",
            ),
        ),
        required_methods=(
            _mth(
                "mth_multi_state_methods",
                "multi-state / multi-configurational method family (requirement list)",
                "named as requirement; no implementation in this module",
            ),
            _mth(
                "mth_spin_flip_methods",
                "spin-flip method family (requirement list)",
                "named as requirement; no implementation in this module",
            ),
            _mth(
                "mth_mecp_seam",
                "MECP / seam-optimization method family (requirement list)",
                "design §3.2 references ORCA MECP as a backend capability, not a plan",
            ),
        ),
        why_generic_selector_invalid=WHY_SPIN_INVALID,
        future_module_pointer=POINTER_ELECTRONIC_STATE_FLOW,
    ),
    ESUB_EXCITED_STATE_CROSSING: SpecialDomainSpec(
        subtype=ESUB_EXCITED_STATE_CROSSING,
        code=SPEC_ELECTRONIC_STATE,
        required_representations=(
            _rep(
                "rep_state_crossing",
                "excited-state crossing endpoints (state character per side)",
                "state identity recorded; ground-state NEB invalid",
            ),
        ),
        required_methods=(
            _mth(
                "mth_excited_state_crossing_methods",
                "excited-state + seam-location method family (requirement list)",
                "named as requirement; no implementation in this module",
            ),
        ),
        why_generic_selector_invalid=WHY_EXCITED_CROSSING_INVALID,
        future_module_pointer=POINTER_ELECTRONIC_STATE_FLOW,
    ),
}


# ---------------------------------------------------------------------------
# Coverage declaration (design §12): Demo24 evidence is organic-only.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CoverageDeclaration:
    """Which element/domains have ANY evidence; metals = none."""

    evidence_source: str
    domains_with_evidence: tuple[str, ...]
    domains_without_evidence: tuple[str, ...]
    note: str

    def to_doc(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_COVERAGE_DECLARATION,
            "evidence_source": self.evidence_source,
            "domains_with_evidence": list(self.domains_with_evidence),
            "domains_without_evidence": list(self.domains_without_evidence),
            "note": self.note,
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_doc())


#: Design §12: Demo24 covers only C/H/N/O/F/Cl small-molecule singlets.
#: Metals and every special domain have NO evidence — never claim broader
#: coverage from this evidence set.
DEMO24_COVERAGE: Final[CoverageDeclaration] = CoverageDeclaration(
    evidence_source="demo24_strategy_evidence (design §12)",
    domains_with_evidence=("C", "H", "N", "O", "F", "Cl"),
    domains_without_evidence=(
        "coordination_metal",
        "surface_periodic",
        "electron_exchange",
        "excited_state",
        "organometallic",
        "catalysis",
        "radical",
        "solution_multi_body",
        "metal",
        "Fe",
        "all_other_elements_outside_C_H_N_O_F_Cl",
    ),
    note=(
        "Demo24 evidence covers only C/H/N/O/F/Cl small-molecule singlets; "
        "metals and the special domains have NO evidence and must never be "
        "claimed as covered by this design result (design §12)."
    ),
)


def element_has_demo24_evidence(element: str) -> bool:
    """True only for the Demo24 element set C/H/N/O/F/Cl."""
    return element in DEMO24_COVERAGE.domains_with_evidence


# ---------------------------------------------------------------------------
# Domain flags + typed exit records.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DomainFlags:
    """Explicit special-domain flags the caller supplies (no silent guesses)."""

    surface_or_periodic: bool = False
    electron_exchange: bool = False
    excited_state: bool = False

    def any_set(self) -> bool:
        return (
            self.surface_or_periodic or self.electron_exchange or self.excited_state
        )


@dataclass(frozen=True, slots=True)
class SpecialDomainExit:
    """Typed special-domain / electronic-state exit (exits only)."""

    code: str
    subtype: str
    reason: str
    spec: SpecialDomainSpec
    ordinary_neb_permitted: bool = False
    coverage_note: str = DEMO24_COVERAGE.note
    reaction_id: str | None = None

    @property
    def required_representations(self) -> tuple[Requirement, ...]:
        return self.spec.required_representations

    @property
    def required_methods(self) -> tuple[Requirement, ...]:
        return self.spec.required_methods

    @property
    def why_generic_selector_invalid(self) -> str:
        return self.spec.why_generic_selector_invalid

    @property
    def future_module_pointer(self) -> str:
        return self.spec.future_module_pointer

    @property
    def required_method_ids(self) -> tuple[str, ...]:
        return tuple(req.requirement_id for req in self.spec.required_methods)

    @property
    def required_representation_ids(self) -> tuple[str, ...]:
        return tuple(req.requirement_id for req in self.spec.required_representations)

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection (deterministic via ``stable_json_dumps``)."""
        return {
            "schema_version": SCHEMA_SPECIAL_DOMAIN_EXIT,
            "reaction_id": self.reaction_id,
            "code": self.code,
            "subtype": self.subtype,
            "reason": self.reason,
            "ordinary_neb_permitted": self.ordinary_neb_permitted,
            "ordinary_neb_note": ORDINARY_NEB_NOT_UNIVERSAL_FALLBACK,
            "required_representations": [
                req.to_record() for req in self.spec.required_representations
            ],
            "required_methods": [
                req.to_record() for req in self.spec.required_methods
            ],
            "why_generic_selector_invalid": self.spec.why_generic_selector_invalid,
            "future_module_pointer": self.spec.future_module_pointer,
            "coverage_note": self.coverage_note,
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_doc())


def _build_exit(
    spec: SpecialDomainSpec,
    reason: str,
    *,
    reaction_id: str | None = None,
) -> SpecialDomainExit:
    return SpecialDomainExit(
        code=spec.code,
        subtype=spec.subtype,
        reason=reason,
        spec=spec,
        ordinary_neb_permitted=False,
        coverage_note=DEMO24_COVERAGE.note,
        reaction_id=reaction_id,
    )


# ---------------------------------------------------------------------------
# Element helpers (single source: registry.ORGANIC_ELEMENTS).
# ---------------------------------------------------------------------------
def elements_from_bundle(bundle: EndpointGraphBundle) -> tuple[str, ...]:
    """Sorted unique elements present on either endpoint graph."""
    found: set[str] = set()
    for node in bundle.r_graph.nodes:
        found.add(str(node.element))
    for node in bundle.p_graph.nodes:
        found.add(str(node.element))
    return tuple(sorted(found))


def non_organic_elements(elements: Sequence[str]) -> tuple[str, ...]:
    """Elements outside ``registry.ORGANIC_ELEMENTS`` (sorted, unique)."""
    return tuple(sorted({str(e) for e in elements if str(e) not in ORGANIC_ELEMENTS}))


# ---------------------------------------------------------------------------
# Classification entry points.
# ---------------------------------------------------------------------------
def classify_special_domain(
    *,
    elements: Sequence[str] | None = None,
    bundle: EndpointGraphBundle | None = None,
    domain_flags: DomainFlags | None = None,
    reaction_id: str | None = None,
) -> tuple[SpecialDomainExit, ...]:
    """Classify ``SPECIAL_DOMAIN`` subtypes for one input.

    Returns one typed exit per applicable subtype (closed set).  Non-organic
    elements yield ``coordination_metal``; explicit flags add
    ``surface_periodic`` / ``electron_exchange`` / ``excited_state``.
    Empty tuple when nothing triggers.
    """
    flags = domain_flags or DomainFlags()
    if elements is None and bundle is not None:
        elements = elements_from_bundle(bundle)

    exits: list[SpecialDomainExit] = []
    non_org = non_organic_elements(elements or ())
    if non_org:
        exits.append(
            _build_exit(
                SPECIAL_DOMAIN_SPECS[SUBTYPE_COORDINATION_METAL],
                (
                    "non-organic elements "
                    f"{list(non_org)} require special-domain routing; "
                    f"Demo24 evidence covers only "
                    f"{list(DEMO24_COVERAGE.domains_with_evidence)}"
                ),
                reaction_id=reaction_id,
            )
        )
    if flags.surface_or_periodic:
        exits.append(
            _build_exit(
                SPECIAL_DOMAIN_SPECS[SUBTYPE_SURFACE_PERIODIC],
                "surface/periodic flag set; special-domain routing required",
                reaction_id=reaction_id,
            )
        )
    if flags.electron_exchange:
        exits.append(
            _build_exit(
                SPECIAL_DOMAIN_SPECS[SUBTYPE_ELECTRON_EXCHANGE],
                "electron-exchange flag set; special-domain routing required",
                reaction_id=reaction_id,
            )
        )
    if flags.excited_state:
        exits.append(
            _build_exit(
                SPECIAL_DOMAIN_SPECS[SUBTYPE_EXCITED_STATE],
                "excited-state flag set; special-domain routing required",
                reaction_id=reaction_id,
            )
        )
    return tuple(exits)


def classify_electronic_state(
    *,
    r_charge: int | None = None,
    p_charge: int | None = None,
    r_multiplicity: int | None = None,
    p_multiplicity: int | None = None,
    r_electron_count: int | None = None,
    p_electron_count: int | None = None,
    r_excited_state: bool = False,
    p_excited_state: bool = False,
    reaction_id: str | None = None,
) -> tuple[SpecialDomainExit, ...]:
    """Classify ``SPECIAL_ELECTRONIC_STATE_REQUIRED`` sub-exits.

    Design §3.2 closed set: total-charge difference / particle-number
    difference / cross-spin-surface / excited-state crossing.  Multiple
    sub-exits may apply; all are returned in design order.  Empty tuple when
    the endpoints are electronic-state compatible (or fields are unknown).
    """
    exits: list[SpecialDomainExit] = []
    if (
        r_charge is not None
        and p_charge is not None
        and r_charge != p_charge
    ):
        exits.append(
            _build_exit(
                SPECIAL_ELECTRONIC_STATE_SPECS[ESUB_TOTAL_CHARGE_DIFFERENCE],
                (
                    f"total charge differs (R={r_charge}, P={p_charge}); "
                    "dedicated charged-species flow required"
                ),
                reaction_id=reaction_id,
            )
        )
    if (
        r_electron_count is not None
        and p_electron_count is not None
        and r_electron_count != p_electron_count
    ):
        exits.append(
            _build_exit(
                SPECIAL_ELECTRONIC_STATE_SPECS[ESUB_PARTICLE_NUMBER_DIFFERENCE],
                (
                    f"particle number differs (R={r_electron_count}, "
                    f"P={p_electron_count}); "
                    "electron attachment/detachment workflow required"
                ),
                reaction_id=reaction_id,
            )
        )
    if (
        r_multiplicity is not None
        and p_multiplicity is not None
        and r_multiplicity != p_multiplicity
    ):
        exits.append(
            _build_exit(
                SPECIAL_ELECTRONIC_STATE_SPECS[ESUB_CROSS_SPIN_SURFACE],
                (
                    f"multiplicity differs (R={r_multiplicity}, "
                    f"P={p_multiplicity}); cross-spin-surface flow required"
                ),
                reaction_id=reaction_id,
            )
        )
    if r_excited_state or p_excited_state:
        exits.append(
            _build_exit(
                SPECIAL_ELECTRONIC_STATE_SPECS[ESUB_EXCITED_STATE_CROSSING],
                "excited-state endpoint flag set; seam/crossing flow required",
                reaction_id=reaction_id,
            )
        )
    return tuple(exits)


def detail_for_route_result(
    route_result: RouteResult,
    *,
    elements: Sequence[str] | None = None,
    bundle: EndpointGraphBundle | None = None,
    domain_flags: DomainFlags | None = None,
    r_charge: int | None = None,
    p_charge: int | None = None,
    r_multiplicity: int | None = None,
    p_multiplicity: int | None = None,
    r_electron_count: int | None = None,
    p_electron_count: int | None = None,
    r_excited_state: bool = False,
    p_excited_state: bool = False,
) -> tuple[SpecialDomainExit, ...]:
    """Expand a todo-11 ``special_domain_exit`` RouteResult into full detail.

    Returns ``()`` for ordinary (non-special) outcomes.  For a
    ``SPECIAL_DOMAIN`` route, classifies subtypes from elements/flags; for
    ``SPECIAL_ELECTRONIC_STATE_REQUIRED``, classifies electronic sub-exits
    from charge/multiplicity/electron-count flags.  When a
    ``SPECIAL_DOMAIN`` route cannot be refined (no elements/bundle/flags),
    a single ``coordination_metal`` exit records that refinement inputs were
    not supplied — never a generic strategy candidate.
    """
    if route_result.outcome != OUTCOME_SPECIAL_DOMAIN_EXIT:
        return ()

    code = route_result.rejection_code
    if code == SPEC_ELECTRONIC_STATE:
        return classify_electronic_state(
            r_charge=r_charge,
            p_charge=p_charge,
            r_multiplicity=r_multiplicity,
            p_multiplicity=p_multiplicity,
            r_electron_count=r_electron_count,
            p_electron_count=p_electron_count,
            r_excited_state=r_excited_state,
            p_excited_state=p_excited_state,
            reaction_id=route_result.reaction_id,
        )

    if code == SPEC_DOMAIN:
        exits = classify_special_domain(
            elements=elements,
            bundle=bundle,
            domain_flags=domain_flags,
            reaction_id=route_result.reaction_id,
        )
        if exits:
            return exits
        return (
            _build_exit(
                SPECIAL_DOMAIN_SPECS[SUBTYPE_COORDINATION_METAL],
                (
                    "registry SPECIAL_DOMAIN exit without refinement inputs; "
                    "supply elements/bundle/domain_flags for exact subtype "
                    "(never a generic candidate)"
                ),
                reaction_id=route_result.reaction_id,
            ),
        )

    return ()


# ---------------------------------------------------------------------------
# Ordinary-NEB refusal guards (typed; never a silent fallback).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class NebRefusal:
    """Typed refusal of ordinary NEB for a special-domain/e-state input."""

    code: str
    exit_code: str
    exit_subtype: str
    attempted_method: str
    reason: str
    required_method_ids: tuple[str, ...]

    def to_doc(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_NEB_REFUSAL,
            "code": self.code,
            "exit_code": self.exit_code,
            "exit_subtype": self.exit_subtype,
            "attempted_method": self.attempted_method,
            "reason": self.reason,
            "required_method_ids": list(self.required_method_ids),
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_doc())


class OrdinaryNebRefusedError(ValueError):
    """Raised when ordinary NEB is proposed for a special-domain input."""

    def __init__(self, refusal: NebRefusal) -> None:
        self.refusal = refusal
        super().__init__(f"{refusal.code}: {refusal.reason}")


def refuse_ordinary_neb(
    exit: SpecialDomainExit, *, attempted_method: str = ATTEMPTED_METHOD_ORDINARY_NEB
) -> NebRefusal:
    """Typed refusal: ordinary NEB is not a universal fallback for *exit*."""
    refusal_code = (
        ORDINARY_NEB_REFUSED_ELECTRONIC_STATE
        if exit.code == SPEC_ELECTRONIC_STATE
        else ORDINARY_NEB_REFUSED_SPECIAL_DOMAIN
    )
    method_ids = exit.required_method_ids
    method_text = ", ".join(method_ids) if method_ids else "none recorded"
    reason = (
        f"{ORDINARY_NEB_NOT_UNIVERSAL_FALLBACK}: {exit.subtype} "
        f"({exit.code}); attempted={attempted_method}; "
        f"required method families: {method_text}"
    )
    return NebRefusal(
        code=refusal_code,
        exit_code=exit.code,
        exit_subtype=exit.subtype,
        attempted_method=attempted_method,
        reason=reason,
        required_method_ids=method_ids,
    )


def assert_ordinary_neb_not_used(
    exit: SpecialDomainExit, *, attempted_method: str = ATTEMPTED_METHOD_ORDINARY_NEB
) -> None:
    """Raise :class:`OrdinaryNebRefusedError` — ordinary NEB is refused."""
    raise OrdinaryNebRefusedError(
        refuse_ordinary_neb(exit, attempted_method=attempted_method)
    )


def guard_route_for_ordinary_neb(
    route_result: RouteResult,
    *,
    exits: Sequence[SpecialDomainExit] = (),
    attempted_method: str = ATTEMPTED_METHOD_ORDINARY_NEB,
) -> NebRefusal | None:
    """Guard a route about to be sent to ordinary NEB.

    Returns ``None`` for ordinary routes (NEB/path remains a legal explicit
    escalation per design §5.1 step 7).  Returns a typed :class:`NebRefusal`
    when the route is a special-domain/electronic-state exit — ordinary NEB
    is refused, never a silent fallback.
    """
    if route_result.outcome != OUTCOME_SPECIAL_DOMAIN_EXIT:
        return None
    if exits:
        return refuse_ordinary_neb(exits[0], attempted_method=attempted_method)
    code = route_result.rejection_code or SPEC_DOMAIN
    refusal_code = (
        ORDINARY_NEB_REFUSED_ELECTRONIC_STATE
        if code == SPEC_ELECTRONIC_STATE
        else ORDINARY_NEB_REFUSED_SPECIAL_DOMAIN
    )
    detail = ""
    if route_result.special_domain is not None:
        detail = f"; route detail: {route_result.special_domain.get('reason', '')}"
    reason = (
        f"{ORDINARY_NEB_NOT_UNIVERSAL_FALLBACK}: route code {code}; "
        f"attempted={attempted_method}; route to a dedicated "
        f"special-domain/electronic-state flow{detail}"
    )
    return NebRefusal(
        code=refusal_code,
        exit_code=code,
        exit_subtype="unrefined",
        attempted_method=attempted_method,
        reason=reason,
        required_method_ids=(),
    )


# ---------------------------------------------------------------------------
# Covalent-radius rule guard (design §5.2: never for metals/special domains).
# ---------------------------------------------------------------------------
class CovalentRadiusRuleError(ValueError):
    """Raised when generic covalent-radius rules hit a non-organic element."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class CovalentRadiusGuardResult:
    """Verdict for generic covalent-radius rule applicability."""

    permitted: bool
    purpose: str
    non_organic_elements: tuple[str, ...]
    code: str | None
    detail: str

    def to_doc(self) -> dict[str, Any]:
        return {
            "permitted": self.permitted,
            "purpose": self.purpose,
            "non_organic_elements": list(self.non_organic_elements),
            "code": self.code,
            "detail": self.detail,
        }


def covalent_radius_rule_check(
    elements: Sequence[str], *, purpose: str
) -> CovalentRadiusGuardResult:
    """Whether generic covalent-radius rules may be applied to *elements*.

    Permitted only when every element is in ``registry.ORGANIC_ELEMENTS``.
    Any metal or other non-organic element → refused with
    ``COVALENT_RADIUS_RULE_FORBIDDEN_FOR_METAL``.
    """
    non_org = non_organic_elements(elements)
    if non_org:
        return CovalentRadiusGuardResult(
            permitted=False,
            purpose=purpose,
            non_organic_elements=non_org,
            code=COVALENT_RADIUS_RULE_FORBIDDEN_FOR_METAL,
            detail=(
                "generic covalent-radius rules are forbidden for special-domain "
                f"elements {list(non_org)} (design §5.2 SPECIAL_DOMAIN); "
                f"purpose={purpose}"
            ),
        )
    return CovalentRadiusGuardResult(
        permitted=True,
        purpose=purpose,
        non_organic_elements=(),
        code=None,
        detail=(
            "organic element set "
            f"{sorted(ORGANIC_ELEMENTS)}; generic covalent-radius rule permitted"
        ),
    )


def generic_covalent_radius_sum(first: str, second: str, *, purpose: str) -> float:
    """Generic covalent-radius sum — refuses metals/non-organic elements.

    Reuses ``geometry_feasibility.covalent_radius_sum`` for organic pairs
    (single source of radius values) but **refuses before consulting the
    table** when either element is outside the organic set.
    """
    guard = covalent_radius_rule_check((first, second), purpose=purpose)
    if not guard.permitted:
        raise CovalentRadiusRuleError(
            COVALENT_RADIUS_RULE_FORBIDDEN_FOR_METAL, guard.detail
        )
    return covalent_radius_sum(first, second)


def special_domain_policy_from_config(config: Mapping[str, Any] | None) -> str:
    """Return ``scan_strategy.special_domain_policy`` (default ``unsupported``)."""
    if config is None:
        return "unsupported"
    section = config.get("scan_strategy")
    if not isinstance(section, Mapping):
        return "unsupported"
    raw = section.get("special_domain_policy", "unsupported")
    return str(raw)
