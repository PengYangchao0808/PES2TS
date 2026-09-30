from __future__ import annotations

import pytest

from pes2ts_core.contracts import ContractError, seal_document, validate_document
from pes2ts_core.demo import synthetic_objects
from pes2ts_core.g1.reaction_case import resolve_endpoint_multiplicity


def _export(reactant_multiplicities: list[int | None]) -> dict:
    components = {}
    atom_rows = []
    for index, multiplicity in enumerate(reactant_multiplicities):
        tag = f"R{index}"
        components[tag] = {"atomic_numbers": [6], "charge": 0, "multiplicity": multiplicity}
        atom_rows.append({"r_component": tag, "p_component": "P0"})
    components["P0"] = {"atomic_numbers": [6] * len(reactant_multiplicities),
                         "charge": 0, "multiplicity": 1}
    return {"atom_rows": atom_rows, "components": components,
            "charge_total_reactants": 0, "charge_total_products": 0,
            "multiplicity_max": 42}


@pytest.mark.parametrize(
    ("component_spins", "expected", "status"),
    [([1], 1, "resolved"), ([1, 1], 1, "resolved"),
     ([2, 1], 2, "resolved"), ([2, 2], None, "unresolved"),
     ([None], None, "unresolved")],
)
def test_endpoint_spin_resolution_is_conservative(component_spins, expected, status):
    result = resolve_endpoint_multiplicity(_export(component_spins), "reactant")
    assert result["multiplicity"] == expected
    assert result["status"] == status


def test_endpoint_spin_resolution_rejects_inconsistent_source_charge():
    export = _export([1])
    export["components"]["R0"]["charge"] = 1
    with pytest.raises(ContractError, match="do not sum"):
        resolve_endpoint_multiplicity(export, "reactant")


def test_endpoint_spin_resolution_rejects_component_atom_count_mismatch():
    export = _export([1])
    export["components"]["R0"]["atomic_numbers"] = [6, 1]
    with pytest.raises(ContractError, match="atom count disagrees"):
        resolve_endpoint_multiplicity(export, "reactant")


def _case_with_spin_provenance() -> dict:
    case = synthetic_objects()["ReactionCase"]
    case["source"]["g1_export_sha256"] = "a" * 64
    case["source"]["g1_payload_sha256"] = "b" * 64
    case["source"]["spin_provenance"] = {
        side: {"side": side, "multiplicity": 1, "status": "resolved",
               "reason": "all_source_components_are_singlets",
               "source_field": "sanitized_g1_export.components",
               "components": [{"component_id": prefix + "0", "multiplicity": 1,
                               "charge": 0, "atom_count": len(case["atoms"])}]}
        for side, prefix in (("reactant", "R"), ("product", "P"))
    }
    return seal_document(case)


def test_contract_accepts_consistent_source_spin_provenance():
    assert validate_document(_case_with_spin_provenance()) == []


def test_contract_rejects_endpoint_spin_that_disagrees_with_provenance():
    case = _case_with_spin_provenance()
    case["reactant"]["multiplicity"] = 2
    problems = validate_document(seal_document(case))
    assert any("reactant.multiplicity: disagrees with source spin provenance" in issue
               for issue in problems)


def test_contract_recomputes_ambiguity_and_component_charge_totals():
    case = _case_with_spin_provenance()
    case["source"]["spin_provenance"]["reactant"]["components"] = [
        {"component_id": "R0", "multiplicity": 2, "charge": 0, "atom_count": 1},
        {"component_id": "R1", "multiplicity": 2, "charge": 1, "atom_count": 2},
    ]
    problems = validate_document(seal_document(case))
    assert any("charges do not sum" in issue for issue in problems)
    assert any("spin resolution" in issue for issue in problems)
