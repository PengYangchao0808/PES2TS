"""Strict endpoint-only conversion from the current G1 v2 export to ReactionCase."""
from __future__ import annotations

import hashlib
from typing import Any

from pes2ts_core.contracts import ContractError, make_document
from pes2ts_core.utils.hashing import stable_json_dumps

ALLOWED_EXPORT_FIELDS = frozenset({
    "schema_version", "reaction_id", "dataset_version", "mapping_provenance", "atom_order", "maps", "elements",
    "r_atomic_numbers", "r_coordinates", "p_atomic_numbers", "p_coordinates", "atom_rows", "components",
    "charge_total_reactants", "charge_total_products", "multiplicity_max", "edits", "hydrogen_partner_changes",
    "aromatic_regions", "reaction_center", "classification", "status_summary", "generated_at",
})


def resolve_endpoint_multiplicity(export: dict[str, Any], side: str) -> dict[str, Any]:
    """Resolve whole-endpoint spin only when source components determine it.

    Component multiplicities come from the endpoint-only G0 inventory.  A
    whole endpoint has a unique multiplicity when it contains at most one
    non-singlet component; two or more non-singlet components have unresolved
    spin coupling and are deliberately left null.  ``multiplicity_max`` is
    never used because a maximum is not a total-system spin state.
    """
    if side not in {"reactant", "product"}:
        raise ContractError("side must be reactant or product")
    prefix = "R" if side == "reactant" else "P"
    component_key = f"{prefix.lower()}_component"
    rows = export.get("atom_rows")
    components = export.get("components")
    if not isinstance(rows, list) or not rows or not isinstance(components, dict):
        raise ContractError("endpoint spin resolution requires source atom_rows and components")
    tags: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get(component_key), str):
            raise ContractError(f"atom_rows[{index}].{component_key} must be a component ID")
        tag = row[component_key]
        if not tag.startswith(prefix):
            raise ContractError(f"atom_rows[{index}].{component_key} has the wrong endpoint prefix")
        tags.append(tag)
    component_ids = sorted(set(tags))
    declared_side_ids = sorted(tag for tag in components if isinstance(tag, str) and tag.startswith(prefix))
    if declared_side_ids != component_ids:
        raise ContractError(f"source component IDs for {side} disagree with atom_rows")
    component_records = []
    for tag in component_ids:
        component = components.get(tag)
        if not isinstance(component, dict):
            raise ContractError(f"source component {tag} is missing")
        atom_count = sum(value == tag for value in tags)
        atomic_numbers = component.get("atomic_numbers")
        if not isinstance(atomic_numbers, list) or len(atomic_numbers) != atom_count:
            raise ContractError(f"source component {tag} atom count disagrees with atom_rows")
        multiplicity = component.get("multiplicity")
        charge = component.get("charge")
        if multiplicity is not None and (not isinstance(multiplicity, int) or isinstance(multiplicity, bool)
                                         or multiplicity < 1):
            raise ContractError(f"source component {tag} multiplicity must be null or a positive integer")
        valid_multiplicity = (isinstance(multiplicity, int) and not isinstance(multiplicity, bool)
                              and multiplicity > 0)
        valid_charge = isinstance(charge, int) and not isinstance(charge, bool)
        component_records.append({"component_id": tag,
                                  "multiplicity": multiplicity if valid_multiplicity else None,
                                  "charge": charge if valid_charge else None,
                                  "atom_count": atom_count})
    endpoint_charge = export.get("charge_total_reactants" if side == "reactant" else "charge_total_products")
    if not isinstance(endpoint_charge, int) or isinstance(endpoint_charge, bool):
        raise ContractError(f"source total charge for {side} must be an integer")
    if any(record["charge"] is None for record in component_records):
        raise ContractError(f"source component charge for {side} must be an integer")
    if sum(record["charge"] for record in component_records) != endpoint_charge:
        raise ContractError(f"source component charges do not sum to the {side} endpoint charge")

    unknown = [record for record in component_records if record["multiplicity"] is None]
    non_singlets = [record for record in component_records if record["multiplicity"] not in (None, 1)]
    if unknown:
        multiplicity, status, reason = None, "unresolved", "component_multiplicity_missing"
    elif len(non_singlets) > 1:
        multiplicity, status, reason = None, "unresolved", "multiple_non_singlet_components_require_spin_coupling"
    elif non_singlets:
        multiplicity, status, reason = non_singlets[0]["multiplicity"], "resolved", "source_components_define_total_spin"
    else:
        multiplicity, status, reason = 1, "resolved", "all_source_components_are_singlets"
    return {"side": side, "multiplicity": multiplicity, "status": status,
            "reason": reason, "components": component_records,
            "source_field": "sanitized_g1_export.components"}


def reaction_case_from_g1_export(export: dict[str, Any], *, split: str, reactant_multiplicity: int | None,
                                 product_multiplicity: int | None, provenance_sha256: str | None = None,
                                 review_reasons: list[str] | None = None,
                                 spin_provenance: dict[str, Any] | None = None,
                                 source_export_sha256: str | None = None) -> dict[str, Any]:
    """Convert G1 export with a positive whitelist and explicit spin review state.

    IRC-derived keys in legacy exports are ignored. Multiplicity is never
    inferred from ``multiplicity_max`` because it is not a whole-system spin.
    Missing multiplicities produce a non-executable ``needs_review`` case.
    """
    unknown = set(export) - ALLOWED_EXPORT_FIELDS
    required = {"reaction_id", "dataset_version", "maps", "elements", "r_coordinates", "p_coordinates", "edits", "hydrogen_partner_changes"}
    missing = required - set(export)
    if unknown:
        raise ContractError(f"unrecognized G1 export fields: {sorted(unknown)}")
    if missing:
        raise ContractError(f"G1 export is missing fields: {sorted(missing)}")
    multiplicities = (reactant_multiplicity, product_multiplicity)
    if any(value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1)
           for value in multiplicities):
        raise ContractError("endpoint multiplicities must be positive integers or unresolved None")
    maps, elements = export["maps"], export["elements"]
    if len(maps) != len(elements) or len(set(maps)) != len(maps):
        raise ContractError("G1 map and element arrays do not align")
    payload = {key: export[key] for key in sorted(required)}
    payload_hash = hashlib.sha256(stable_json_dumps(payload).encode()).hexdigest()
    source_hash = provenance_sha256 or payload_hash
    full_export_hash = (source_export_sha256
                        or hashlib.sha256(stable_json_dumps(export).encode()).hexdigest())
    for name, digest in (("source_export_sha256", full_export_hash), ("provenance_sha256", source_hash)):
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ContractError(f"{name} must be a lowercase SHA256 digest")
    case_id = "case:" + hashlib.sha256(stable_json_dumps({"rid": export["reaction_id"], "sha": source_hash}).encode()).hexdigest()[:16]
    edits = [{"kind": item["edit_kind"], "atom_map_ids": list(item["pair"]),
              "reactant_bond_order": item.get("r_bond_order"), "product_bond_order": item.get("p_bond_order")}
             for item in export["edits"]]
    h_transfers = [{"hydrogen_map_id": item["h"], "old_partner_map_id": item.get("from"),
                    "new_partner_map_id": item.get("to"), "kind": item["kind"]}
                   for item in export["hydrogen_partner_changes"]]
    source = {"dataset_version": export["dataset_version"], "g1_schema_version": export["schema_version"],
              "g1_export_sha256": full_export_hash, "g1_payload_sha256": payload_hash,
              "mapping_provenance": export.get("mapping_provenance"),
              "fields_whitelisted": True}
    if spin_provenance is not None:
        source["spin_provenance"] = spin_provenance
    missing_spin = [side for side, value in zip(("reactant", "product"), multiplicities, strict=True)
                    if value is None]
    case_review_reasons = list(review_reasons or [])
    case_review_reasons.extend(
        f"{side} endpoint spin multiplicity requires chemistry review" for side in missing_spin
    )
    status = "needs_review" if case_review_reasons else "ready"
    fields = dict(dataset_version=export["dataset_version"],
        reaction_id=export["reaction_id"], case_id=case_id, split=split,
        atoms=[{"atom_map_id": atom_map, "element": element} for atom_map, element in zip(maps, elements, strict=True)],
        reactant={"charge": export["charge_total_reactants"], "multiplicity": reactant_multiplicity,
                  "geometry": export["r_coordinates"]},
        product={"charge": export["charge_total_products"], "multiplicity": product_multiplicity,
                 "geometry": export["p_coordinates"]}, edits=edits, hydrogen_transfers=h_transfers,
        source=source)
    if case_review_reasons:
        fields["review_reasons"] = case_review_reasons
    return make_document("ReactionCase", case_id, status, **fields)
