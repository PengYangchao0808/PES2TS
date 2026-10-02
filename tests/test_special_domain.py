"""Offline tests for special-domain / electronic-state routing (todo 29).

Locks design §3.2 / §5.2 / §12:

* metal input → typed ``SPECIAL_DOMAIN`` exit (subtype + representation/
  method requirement records), never a generic strategy candidate;
* cross-spin-surface input → ``SPECIAL_ELECTRONIC_STATE_REQUIRED`` sub-exit
  with ordinary NEB refused (typed);
* surface/periodic and electron-exchange / excited-state subtypes typed;
* coverage declaration records metals as **no-evidence** (Demo24 = C/H/N/O/F/Cl);
* generic covalent-radius rule application to metal input is refused;
* determinism of exit serialization.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.endpoint_context import EndpointElectronic
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning.registry import (
    ORGANIC_ELEMENTS,
    OUTCOME_EXECUTABLE_CANDIDATE_SET,
    OUTCOME_SPECIAL_DOMAIN_EXIT,
    SPEC_DOMAIN,
    SPEC_ELECTRONIC_STATE,
)
from pes2ts_core.generation.planning.special_domain import (
    ATTEMPTED_METHOD_ORDINARY_NEB,
    CATEGORY_METHOD_FAMILY,
    CATEGORY_REPRESENTATION,
    COVALENT_RADIUS_RULE_FORBIDDEN_FOR_METAL,
    DEMO24_COVERAGE,
    ESUB_CROSS_SPIN_SURFACE,
    ESUB_EXCITED_STATE_CROSSING,
    ESUB_PARTICLE_NUMBER_DIFFERENCE,
    ESUB_TOTAL_CHARGE_DIFFERENCE,
    ORDINARY_NEB_NOT_UNIVERSAL_FALLBACK,
    ORDINARY_NEB_REFUSED_ELECTRONIC_STATE,
    ORDINARY_NEB_REFUSED_SPECIAL_DOMAIN,
    SPECIAL_DOMAIN_SUBTYPES,
    SPECIAL_ELECTRONIC_STATE_SUBTYPES,
    SUBTYPE_COORDINATION_METAL,
    SUBTYPE_ELECTRON_EXCHANGE,
    SUBTYPE_EXCITED_STATE,
    SUBTYPE_SURFACE_PERIODIC,
    CovalentRadiusRuleError,
    DomainFlags,
    OrdinaryNebRefusedError,
    assert_ordinary_neb_not_used,
    classify_electronic_state,
    classify_special_domain,
    covalent_radius_rule_check,
    detail_for_route_result,
    element_has_demo24_evidence,
    elements_from_bundle,
    generic_covalent_radius_sum,
    guard_route_for_ordinary_neb,
    non_organic_elements,
    refuse_ordinary_neb,
)
from test_strategy_registry import (
    _electronic_mismatch,
    _hand_bundle,
    _metal_input,
    _p0,
    _route,
)

FORBIDDEN = {key.lower() for key in FORBIDDEN_TRUTH_KEYS} | {
    key.lower() for key in FORBIDDEN_EXPORT_KEYS
} | {"endpoint_match", "orientation", "irc_evidence"}


def _walk_keys(value: object, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key).lower())
            _walk_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, found)


def _cross_spin_p0() -> dict[str, Any]:
    """Organic endpoints, same charge, different multiplicities (cross-spin)."""
    return _p0(
        [(1, 2, 1)],
        [(1, 2, 1), (1, 3, 1)],
        elements={1: "C", 2: "C", 3: "C"},
        endpoint_electronic={
            "reactant": EndpointElectronic(charge=0, multiplicity=1),
            "product": EndpointElectronic(charge=0, multiplicity=3),
        },
    )


# ---------------------------------------------------------------------------
# Vocabulary / catalog integrity.
# ---------------------------------------------------------------------------
def test_special_domain_subtype_closed_set() -> None:
    assert SPECIAL_DOMAIN_SUBTYPES == (
        "coordination_metal",
        "surface_periodic",
        "electron_exchange",
        "excited_state",
    )


def test_electronic_state_subtype_closed_set() -> None:
    assert SPECIAL_ELECTRONIC_STATE_SUBTYPES == (
        "total_charge_difference",
        "particle_number_difference",
        "cross_spin_surface",
        "excited_state_crossing",
    )


def test_organic_elements_reused_from_registry_single_source() -> None:
    """Element classification is imported from registry — never redefined."""
    assert "C" in ORGANIC_ELEMENTS and "H" in ORGANIC_ELEMENTS
    assert "Fe" not in ORGANIC_ELEMENTS
    assert "Cl" in ORGANIC_ELEMENTS
    assert non_organic_elements(["C", "H", "Fe", "O"]) == ("Fe",)
    assert non_organic_elements(["C", "H", "O"]) == ()


# ---------------------------------------------------------------------------
# Metal input → typed SPECIAL_DOMAIN exit (NOT a generic candidate).
# ---------------------------------------------------------------------------
def test_metal_input_route_is_special_exit_not_generic_candidate() -> None:
    p0 = _metal_input()  # Fe–C hand-built fixture
    result = _route(p0)
    assert result.outcome == OUTCOME_SPECIAL_DOMAIN_EXIT
    assert result.rejection_code == SPEC_DOMAIN
    assert result.strategy_candidates == ()  # never a generic candidate


def test_metal_input_detail_layer_subtype_and_requirements() -> None:
    p0 = _metal_input()
    result = _route(p0, reaction_id="RXN_METAL")
    elements = elements_from_bundle(p0["bundle"])
    assert "Fe" in elements
    exits = detail_for_route_result(result, elements=elements)
    assert len(exits) == 1
    exit = exits[0]
    assert exit.code == SPEC_DOMAIN
    assert exit.subtype == SUBTYPE_COORDINATION_METAL
    assert exit.ordinary_neb_permitted is False
    assert exit.required_representations
    assert exit.required_methods
    for req in (*exit.required_representations, *exit.required_methods):
        assert req.requirement_id
        assert req.category in (CATEGORY_REPRESENTATION, CATEGORY_METHOD_FAMILY)
        assert req.label
    assert "metal" in exit.why_generic_selector_invalid.lower()
    assert "specialized" in exit.future_module_pointer
    assert "Fe" in exit.reason
    doc = exit.to_doc()
    assert doc["subtype"] == SUBTYPE_COORDINATION_METAL
    assert doc["ordinary_neb_permitted"] is False
    assert doc["required_representations"][0]["requirement_id"]


def test_detail_layer_empty_for_ordinary_route() -> None:
    p0 = _p0(
        [(1, 2, 1)],
        [(1, 2, 1), (1, 3, 1)],
        elements={1: "C", 2: "C", 3: "C"},
    )
    result = _route(p0)
    assert result.outcome == OUTCOME_EXECUTABLE_CANDIDATE_SET
    assert detail_for_route_result(result, elements=["C", "C"]) == ()


def test_detail_layer_unrefined_special_domain_fallback_still_typed() -> None:
    p0 = _metal_input()
    result = _route(p0)
    exits = detail_for_route_result(result)  # no elements supplied
    assert len(exits) == 1
    assert exits[0].subtype == SUBTYPE_COORDINATION_METAL
    assert exits[0].ordinary_neb_permitted is False
    assert "refinement" in exits[0].reason


# ---------------------------------------------------------------------------
# SPECIAL_DOMAIN subtypes: surface/periodic, electron-exchange, excited-state.
# ---------------------------------------------------------------------------
def test_surface_periodic_flag_typed_exit() -> None:
    exits = classify_special_domain(
        elements=["C", "H"],
        domain_flags=DomainFlags(surface_or_periodic=True),
    )
    assert len(exits) == 1
    exit = exits[0]
    assert exit.subtype == SUBTYPE_SURFACE_PERIODIC
    assert exit.code == SPEC_DOMAIN
    assert exit.required_representations
    assert exit.required_methods
    assert "periodic" in exit.why_generic_selector_invalid.lower()
    assert exit.ordinary_neb_permitted is False


def test_electron_exchange_flag_typed_exit() -> None:
    exits = classify_special_domain(
        elements=["C", "H"],
        domain_flags=DomainFlags(electron_exchange=True),
    )
    assert exits[0].subtype == SUBTYPE_ELECTRON_EXCHANGE
    assert exits[0].code == SPEC_DOMAIN
    assert exits[0].required_methods


def test_excited_state_flag_typed_exit() -> None:
    exits = classify_special_domain(
        elements=["C", "H"],
        domain_flags=DomainFlags(excited_state=True),
    )
    assert exits[0].subtype == SUBTYPE_EXCITED_STATE
    assert exits[0].code == SPEC_DOMAIN
    assert exits[0].required_methods


def test_metal_surface_multiple_subtypes_all_typed() -> None:
    exits = classify_special_domain(
        elements=["Fe", "C"],
        domain_flags=DomainFlags(surface_or_periodic=True),
    )
    subtypes = [exit.subtype for exit in exits]
    assert subtypes == [SUBTYPE_COORDINATION_METAL, SUBTYPE_SURFACE_PERIODIC]
    for exit in exits:
        assert exit.ordinary_neb_permitted is False
        assert exit.required_representations
        assert exit.required_methods


def test_classify_special_domain_empty_when_organic_no_flags() -> None:
    assert classify_special_domain(elements=["C", "H", "O"]) == ()
    bundle = _hand_bundle([(1, 2, 1)], [(1, 2, 1)], elements={1: "C", 2: "C"})
    assert elements_from_bundle(bundle) == ("C",)
    assert classify_special_domain(bundle=bundle) == ()


# ---------------------------------------------------------------------------
# Cross-spin-surface → SPECIAL_ELECTRONIC_STATE_REQUIRED; ordinary NEB refused.
# ---------------------------------------------------------------------------
def test_cross_spin_surface_route_special_electronic_state_exit() -> None:
    p0 = _cross_spin_p0()
    result = _route(p0)
    assert result.outcome == OUTCOME_SPECIAL_DOMAIN_EXIT
    assert result.rejection_code == SPEC_ELECTRONIC_STATE
    assert result.strategy_candidates == ()

    exits = detail_for_route_result(
        result,
        r_charge=0,
        p_charge=0,
        r_multiplicity=1,
        p_multiplicity=3,
    )
    assert len(exits) == 1
    exit = exits[0]
    assert exit.code == SPEC_ELECTRONIC_STATE
    assert exit.subtype == ESUB_CROSS_SPIN_SURFACE
    assert exit.ordinary_neb_permitted is False
    method_ids = exit.required_method_ids
    assert "mth_multi_state_methods" in method_ids
    assert "mth_spin_flip_methods" in method_ids
    assert "mth_mecp_seam" in method_ids


def test_cross_spin_surface_ordinary_neb_refused_typed() -> None:
    exits = classify_electronic_state(
        r_charge=0, p_charge=0, r_multiplicity=1, p_multiplicity=3
    )
    assert exits[0].subtype == ESUB_CROSS_SPIN_SURFACE
    refusal = refuse_ordinary_neb(exits[0])
    assert refusal.code == ORDINARY_NEB_REFUSED_ELECTRONIC_STATE
    assert refusal.exit_code == SPEC_ELECTRONIC_STATE
    assert refusal.exit_subtype == ESUB_CROSS_SPIN_SURFACE
    assert refusal.attempted_method == ATTEMPTED_METHOD_ORDINARY_NEB
    assert ORDINARY_NEB_NOT_UNIVERSAL_FALLBACK in refusal.reason
    assert "mth_multi_state_methods" in refusal.required_method_ids
    assert "NEB" in refusal.reason

    with pytest.raises(OrdinaryNebRefusedError) as excinfo:
        assert_ordinary_neb_not_used(exits[0])
    assert excinfo.value.refusal.code == ORDINARY_NEB_REFUSED_ELECTRONIC_STATE


def test_guard_route_for_ordinary_neb_refuses_special_route() -> None:
    p0 = _cross_spin_p0()
    result = _route(p0)
    refusal = guard_route_for_ordinary_neb(
        result,
        exits=classify_electronic_state(
            r_charge=0, p_charge=0, r_multiplicity=1, p_multiplicity=3
        ),
    )
    assert refusal is not None
    assert refusal.code == ORDINARY_NEB_REFUSED_ELECTRONIC_STATE

    # Route-level guard without refined exits still refuses (typed).
    refusal_bare = guard_route_for_ordinary_neb(result)
    assert refusal_bare is not None
    assert refusal_bare.code == ORDINARY_NEB_REFUSED_ELECTRONIC_STATE
    assert refusal_bare.exit_subtype == "unrefined"


def test_guard_route_allows_ordinary_route_for_neb_escalation() -> None:
    """Ordinary routes are not blocked — NEB remains a legal explicit escalation."""
    p0 = _p0(
        [(1, 2, 1)],
        [(1, 2, 1), (1, 3, 1)],
        elements={1: "C", 2: "C", 3: "C"},
    )
    result = _route(p0)
    assert guard_route_for_ordinary_neb(result) is None


# ---------------------------------------------------------------------------
# Other electronic-state sub-exits (typed requirement lists).
# ---------------------------------------------------------------------------
def test_total_charge_difference_sub_exit() -> None:
    exits = classify_electronic_state(r_charge=0, p_charge=1)
    assert len(exits) == 1
    exit = exits[0]
    assert exit.subtype == ESUB_TOTAL_CHARGE_DIFFERENCE
    assert exit.code == SPEC_ELECTRONIC_STATE
    assert "constrained_dft" in exit.required_method_ids[0]
    assert exit.ordinary_neb_permitted is False


def test_particle_number_difference_sub_exit() -> None:
    exits = classify_electronic_state(r_electron_count=10, p_electron_count=11)
    assert exits[0].subtype == ESUB_PARTICLE_NUMBER_DIFFERENCE
    assert exits[0].required_method_ids


def test_excited_state_crossing_sub_exit() -> None:
    exits = classify_electronic_state(p_excited_state=True)
    assert exits[0].subtype == ESUB_EXCITED_STATE_CROSSING
    assert exits[0].code == SPEC_ELECTRONIC_STATE
    assert exits[0].required_methods


def test_charge_and_multiplicity_differ_yields_two_sub_exits() -> None:
    exits = classify_electronic_state(
        r_charge=0, p_charge=1, r_multiplicity=1, p_multiplicity=2
    )
    subtypes = [exit.subtype for exit in exits]
    assert subtypes == [
        ESUB_TOTAL_CHARGE_DIFFERENCE,
        ESUB_CROSS_SPIN_SURFACE,
    ]


def test_registry_electronic_mismatch_expands_to_sub_exits() -> None:
    p0 = _electronic_mismatch()  # charge 0→1, mult 1→2
    result = _route(p0)
    assert result.rejection_code == SPEC_ELECTRONIC_STATE
    exits = detail_for_route_result(
        result, r_charge=0, p_charge=1, r_multiplicity=1, p_multiplicity=2
    )
    subtypes = [exit.subtype for exit in exits]
    assert ESUB_TOTAL_CHARGE_DIFFERENCE in subtypes
    assert ESUB_CROSS_SPIN_SURFACE in subtypes
    for exit in exits:
        assert exit.ordinary_neb_permitted is False


def test_electronic_state_compatible_returns_empty() -> None:
    assert (
        classify_electronic_state(
            r_charge=0, p_charge=0, r_multiplicity=1, p_multiplicity=1
        )
        == ()
    )
    assert classify_electronic_state() == ()


# ---------------------------------------------------------------------------
# Coverage declaration: metals = no evidence; Demo24 = C/H/N/O/F/Cl only.
# ---------------------------------------------------------------------------
def test_coverage_declaration_demo24_organic_only() -> None:
    cov = DEMO24_COVERAGE
    assert set(cov.domains_with_evidence) == {"C", "H", "N", "O", "F", "Cl"}
    assert "Fe" in cov.domains_without_evidence
    assert "metal" in cov.domains_without_evidence
    assert "coordination_metal" in cov.domains_without_evidence
    assert "surface_periodic" in cov.domains_without_evidence
    assert cov.note
    doc = cov.to_doc()
    assert doc["schema_version"] == "g1_special_domain_coverage_v1"
    assert "Fe" in doc["domains_without_evidence"]


def test_element_has_demo24_evidence_boundary() -> None:
    assert element_has_demo24_evidence("C") is True
    assert element_has_demo24_evidence("Cl") is True
    assert element_has_demo24_evidence("Fe") is False
    assert element_has_demo24_evidence("S") is False


def test_metal_exit_carries_no_evidence_coverage_note() -> None:
    exits = classify_special_domain(elements=["Fe", "C"])
    note = exits[0].coverage_note.lower()
    assert "no evidence" in note
    assert "c/h/n/o/f/cl" in note


# ---------------------------------------------------------------------------
# Covalent-radius guard: generic rules MUST NOT apply to metals.
# ---------------------------------------------------------------------------
def test_covalent_radius_rule_refused_for_metal_input() -> None:
    guard = covalent_radius_rule_check(["Fe", "C"], purpose="bond feasibility")
    assert guard.permitted is False
    assert guard.code == COVALENT_RADIUS_RULE_FORBIDDEN_FOR_METAL
    assert guard.non_organic_elements == ("Fe",)

    with pytest.raises(CovalentRadiusRuleError) as excinfo:
        generic_covalent_radius_sum("Fe", "C", purpose="bond feasibility")
    assert excinfo.value.code == COVALENT_RADIUS_RULE_FORBIDDEN_FOR_METAL
    assert "forbidden" in excinfo.value.detail


def test_covalent_radius_rule_permitted_for_organic_pairs() -> None:
    guard = covalent_radius_rule_check(["C", "H"], purpose="bond feasibility")
    assert guard.permitted is True
    assert guard.code is None
    value = generic_covalent_radius_sum("C", "H", purpose="bond feasibility")
    assert value > 0


def test_covalent_radius_rule_refused_for_each_non_organic_element() -> None:
    for element in ("Fe", "Na", "Se"):
        guard = covalent_radius_rule_check(["C", element], purpose="test")
        assert guard.permitted is False
        assert guard.non_organic_elements == (element,)


# ---------------------------------------------------------------------------
# Guard: metal special-domain route → ordinary NEB refused (typed).
# ---------------------------------------------------------------------------
def test_metal_route_ordinary_neb_refused() -> None:
    p0 = _metal_input()
    result = _route(p0)
    exits = detail_for_route_result(result, elements=["Fe", "C", "C"])
    refusal = guard_route_for_ordinary_neb(result, exits=exits)
    assert refusal is not None
    assert refusal.code == ORDINARY_NEB_REFUSED_SPECIAL_DOMAIN
    assert refusal.exit_subtype == SUBTYPE_COORDINATION_METAL

    with pytest.raises(OrdinaryNebRefusedError) as excinfo:
        assert_ordinary_neb_not_used(exits[0])
    assert "not a universal fallback" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Determinism + purity.
# ---------------------------------------------------------------------------
def test_exit_serialization_deterministic() -> None:
    exits = classify_special_domain(elements=["Fe", "C"])
    assert exits[0].to_json() == exits[0].to_json()
    e_exits = classify_electronic_state(
        r_charge=0, p_charge=0, r_multiplicity=1, p_multiplicity=3
    )
    assert e_exits[0].to_json() == e_exits[0].to_json()
    assert DEMO24_COVERAGE.to_json() == DEMO24_COVERAGE.to_json()


def test_exit_docs_carry_no_forbidden_truth_keys() -> None:
    samples: list[dict[str, Any]] = [
        classify_special_domain(elements=["Fe", "C"])[0].to_doc(),
        classify_special_domain(
            elements=["C"], domain_flags=DomainFlags(surface_or_periodic=True)
        )[0].to_doc(),
        classify_electronic_state(
            r_charge=0, p_charge=1, r_multiplicity=1, p_multiplicity=3
        )[0].to_doc(),
        DEMO24_COVERAGE.to_doc(),
        refuse_ordinary_neb(
            classify_electronic_state(r_multiplicity=1, p_multiplicity=3)[0]
        ).to_doc(),
        covalent_radius_rule_check(["Fe"], purpose="t").to_doc(),
    ]
    for doc in samples:
        found: set[str] = set()
        _walk_keys(doc, found)
        assert not (found & FORBIDDEN), sorted(found & FORBIDDEN)


def test_route_result_detail_roundtrip_via_registry_json() -> None:
    """Registry special exit + detail layer both serialize deterministically."""
    p0 = _metal_input()
    first = _route(p0, reaction_id="RXN_DET").to_json()
    second = _route(p0, reaction_id="RXN_DET").to_json()
    assert first == second
    exits = detail_for_route_result(
        _route(p0, reaction_id="RXN_DET"), elements=["Fe", "C", "C"]
    )
    doc = json.loads(exits[0].to_json())
    assert doc["code"] == SPEC_DOMAIN
    assert doc["subtype"] == SUBTYPE_COORDINATION_METAL
    assert doc["ordinary_neb_permitted"] is False
