"""Assembly of one versioned G1 reaction-change document.

Wires the batch-1 pieces for one inventory row (parse -> map/row checks ->
bond-change detail -> preview cross-check -> component matching -> index
tables) under first-failure-wins validation: a rejected reaction carries one
typed code plus a machine-readable detail and empty component tables, so an
invalid index can never enter QC.  Persisted component tables use
reaction-global map numbers composed from the matcher's inventory-local rows
(``global = map_lists[rxn_index][q]``).  Bond geometry is a soft per-component
annotation; only a component whose stored bonds exceed the covalent limit in
more than half of its bonds rejects with ``G1_BOND_GEOMETRY``.  The private
``_union_edges`` carrier is consumed by :func:`reaction_center` and never
persisted.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Final

from rdkit import RDLogger

from pes2ts_core.g0.fetch import dataset_version
from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g0.strata import compute_bond_changes_preview
from pes2ts_core.g1.blocks import SideSpec, ambiguity, side_blocks
from pes2ts_core.g1.bond_changes import (
    BondChanges, ChangedBond, compute_bond_changes, crosscheck_against_preview, reaction_center,
)
from pes2ts_core.g1.index_map import build_side_indexes
from pes2ts_core.g1.parse import (
    ReactionParseError, parse_reaction, row_consistency_checks, side_map_checks,
)
from pes2ts_core.g1.skeleton import ComponentMismatchError, match_side_components
from pes2ts_core.utils.hashing import JSONValue, sha256_bytes

# RDKit parse failures are surfaced as typed rejections instead of stderr noise.
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

#: Schema version stamped into every reaction-change document.
CHANGE_SCHEMA_VERSION: Final[str] = "g1_change_v1"
#: The seven boolean category keys of the document's ``categories`` block.
CATEGORY_NAMES: Final[tuple[str, ...]] = (
    "pure_formed", "pure_broken", "both", "order_change_only",
    "has_order_change", "h_migration", "multi_component",
)
#: Preview counter names copied into the document's ``preview_counters``.
PREVIEW_COUNTERS: Final[tuple[str, ...]] = (
    "n_bonds_formed", "n_bonds_broken", "n_bond_order_changed",
)
#: G1 config defaults (``config["g1"]``).
DEFAULT_MATCH_CAP: Final[int] = 10000
DEFAULT_MAX_CANDIDATES: Final[int] = 64
DEFAULT_BOND_TOLERANCE: Final[float] = 0.45
DEFAULT_NEIGHBORHOOD_SHELL: Final[int] = 1


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``g1`` configuration block (or an empty mapping)."""
    g1_config = config.get("g1")
    return g1_config if isinstance(g1_config, Mapping) else {}


def _changed_bond(bond: ChangedBond) -> dict[str, JSONValue]:
    """Return the JSON view of one mapped bond difference."""
    return {"atoms": [bond.atoms[0], bond.atoms[1]], "order_r": bond.order_r, "order_p": bond.order_p}


def _categories(changes: BondChanges, n_reactant: int, n_product: int) -> dict[str, JSONValue]:
    """Return the seven documented category booleans of one document.

    ``order_change_only`` means every formed/broken map pair is also an
    order-changed pair (the bond graph is unchanged; only orders moved), which
    is the only meaningful reading under the documented non-exclusive lists.
    """
    formed, broken = len(changes["formed"]), len(changes["broken"])
    order = len(changes["order_changed"])
    formed_pairs = {bond.atoms for bond in changes["formed"]}
    broken_pairs = {bond.atoms for bond in changes["broken"]}
    order_pairs = {bond.atoms for bond in changes["order_changed"]}
    return {
        "pure_formed": formed > 0 and broken == 0 and order == 0,
        "pure_broken": broken > 0 and formed == 0 and order == 0,
        "both": formed > 0 and broken > 0,
        "order_change_only": order > 0 and formed_pairs <= order_pairs and broken_pairs <= order_pairs,
        "has_order_change": order > 0,
        "h_migration": bool(changes["hydrogen_migration"]),
        "multi_component": n_reactant > 1 or n_product > 1,
    }


def _side_records(row: Mapping[str, Any], prefix: str) -> list[Mapping[str, Any]]:
    """Return the inventory component records of one side, in stored order."""
    return [c for c in row["components"] if str(c["tag"]).startswith(prefix)]


def _base_document(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Return the complete document skeleton of one reaction (rejected default)."""
    version = dataset_version(config)
    text = str(row["reaction_smiles"])
    return {
        "schema_version": CHANGE_SCHEMA_VERSION, "dataset_version": version,
        "reaction_id": str(row["reaction_id"]), "generated_at": _now(),
        "source": {"dataset_version": version, "reaction_smiles_sha256": sha256_bytes(text.encode("utf-8"))},
        "mapping": {"n_atoms": 0, "maps_unique": False, "map_sets_equal": False, "charge_consistent": False,
                    "spin_consistent": False, "n_reactant_components": 0, "n_product_components": 0},
        "bond_changes": {"formed": [], "broken": [], "order_changed": [], "hydrogen_migration": [],
                         "preview_counters": {name: 0 for name in PREVIEW_COUNTERS}, "preview_match": False},
        "reaction_center": {"core": [], "with_shell": []},
        "categories": {name: False for name in CATEGORY_NAMES},
        "reactants": [], "products": [],
        "ambiguity": {"index": "unique", "pairing": "unique"},
        "validation": {"status": "rejected", "failure_code": None, "failure_detail": None},
    }


def _reject(document: dict[str, JSONValue], code: RejectionCode, detail: str) -> dict[str, JSONValue]:
    """Finalize *document* as rejected: one code, one detail, empty component tables."""
    document["reactants"] = []
    document["products"] = []
    document["validation"] = {"status": "rejected", "failure_code": code.value, "failure_detail": detail}
    return document


def build_reaction_change(row: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Assemble the versioned reaction-change document of one inventory row.

    Validation order is first-failure-wins: parse/map/row errors reject with
    ``G1_MAP_ERROR``, a preview mismatch with ``G1_PREVIEW_CONFLICT``, an
    unchanged bond graph with ``G1_NO_BOND_CHANGE``, an unmatched component
    with ``G1_COMPONENT_MISMATCH``, a failed element cross-check with
    ``G1_INDEX_MISMATCH``, and only a component whose stored bonds fail
    geometry in more than half of its bonds with ``G1_BOND_GEOMETRY``.  A
    rejected document keeps the full schema with empty reactant/product tables.
    """
    settings = _settings(config)
    shell = int(settings.get("neighborhood_shell", DEFAULT_NEIGHBORHOOD_SHELL))
    match_cap = int(settings.get("match_cap", DEFAULT_MATCH_CAP))
    max_candidates = int(settings.get("max_candidates", DEFAULT_MAX_CANDIDATES))
    tolerance = float(settings.get("bond_tolerance", DEFAULT_BOND_TOLERANCE))
    document = _base_document(row, config)
    try:
        reactants, products = parse_reaction(str(row["reaction_smiles"]))
    except ReactionParseError as exc:
        return _reject(document, RejectionCode.G1_MAP_ERROR, exc.reason)
    map_checks = side_map_checks(reactants, products)
    mapping: dict[str, JSONValue] = {
        "n_atoms": int(map_checks["n_atoms"]), "maps_unique": bool(map_checks["maps_unique"]),
        "map_sets_equal": bool(map_checks["map_sets_equal"]), "charge_consistent": False,
        "spin_consistent": False, "n_reactant_components": len(reactants.components),
        "n_product_components": len(products.components),
    }
    document["mapping"] = mapping
    if not map_checks["map_sets_equal"]:
        return _reject(document, RejectionCode.G1_MAP_ERROR, "map_sets_differ")
    if not map_checks["maps_unique"]:
        return _reject(document, RejectionCode.G1_MAP_ERROR, "duplicate maps")
    row_checks = row_consistency_checks(row)
    mapping["charge_consistent"] = bool(row_checks["charge_consistent"])
    mapping["spin_consistent"] = bool(row_checks["spin_consistent"])
    if not (row_checks["charge_consistent"] and row_checks["spin_consistent"]):
        return _reject(document, RejectionCode.G1_MAP_ERROR, "charge_or_spin_inconsistent")
    changes = compute_bond_changes(reactants.mols, products.mols, reactants.map_lists, products.map_lists)
    preview = compute_bond_changes_preview(row)
    preview_match = crosscheck_against_preview(row, changes)
    formed: list[JSONValue] = [_changed_bond(bond) for bond in changes["formed"]]
    broken: list[JSONValue] = [_changed_bond(bond) for bond in changes["broken"]]
    order_changed: list[JSONValue] = [_changed_bond(bond) for bond in changes["order_changed"]]
    migrations: list[JSONValue] = [
        {"h": entry["h"], "from": entry["from"], "to": entry["to"]}
        for entry in changes["hydrogen_migration"]
    ]
    document["bond_changes"] = {
        "formed": formed, "broken": broken, "order_changed": order_changed,
        "hydrogen_migration": migrations,
        "preview_counters": {name: int(preview[name]) for name in PREVIEW_COUNTERS},
        "preview_match": preview_match,
    }
    if not preview_match:
        return _reject(document, RejectionCode.G1_PREVIEW_CONFLICT, "preview_counters_differ")
    if not (changes["formed"] or changes["broken"] or changes["order_changed"]):
        return _reject(document, RejectionCode.G1_NO_BOND_CHANGE, "formed=0 broken=0 order_changed=0")
    center = reaction_center(changes, shell)
    core: list[JSONValue] = [atom for atom in center["core"]]
    with_shell: list[JSONValue] = [atom for atom in center["with_shell"]]
    document["reaction_center"] = {"core": core, "with_shell": with_shell}
    document["categories"] = _categories(changes, len(reactants.components), len(products.components))
    reactant_records, product_records = _side_records(row, "R"), _side_records(row, "P")
    try:
        reactant_matches = match_side_components(reactants.mols, reactants.map_lists, reactant_records, match_cap=match_cap, max_candidates=max_candidates)
    except ComponentMismatchError as exc:
        return _reject(document, RejectionCode.G1_COMPONENT_MISMATCH, f"reactant component {exc.rxn_index} unmatched")
    try:
        product_matches = match_side_components(products.mols, products.map_lists, product_records, match_cap=match_cap, max_candidates=max_candidates)
    except ComponentMismatchError as exc:
        return _reject(document, RejectionCode.G1_COMPONENT_MISMATCH, f"product component {exc.rxn_index} unmatched")
    reactant_indexes = build_side_indexes(reactant_matches, reactant_records, max_candidates=max_candidates)
    product_indexes = build_side_indexes(product_matches, product_records, max_candidates=max_candidates)
    for index in (*reactant_indexes, *product_indexes):
        if not index.element_check:
            return _reject(document, RejectionCode.G1_INDEX_MISMATCH, f"{index.tag}: map-to-element cross-check failed")
    reactant_side = SideSpec({m.tag: m for m in reactant_matches}, reactants.map_lists,
                              {str(r["tag"]): r for r in reactant_records}, tolerance)
    product_side = SideSpec({m.tag: m for m in product_matches}, products.map_lists,
                             {str(r["tag"]): r for r in product_records}, tolerance)
    reactant_blocks, hard = side_blocks(reactant_indexes, reactant_side)
    if hard:
        return _reject(document, RejectionCode.G1_BOND_GEOMETRY, hard)
    product_blocks, hard = side_blocks(product_indexes, product_side)
    if hard:
        return _reject(document, RejectionCode.G1_BOND_GEOMETRY, hard)
    document["reactants"], document["products"] = reactant_blocks, product_blocks
    document["ambiguity"] = ambiguity([*reactant_indexes, *product_indexes], [*reactant_matches, *product_matches])
    document["validation"] = {"status": "valid", "failure_code": None, "failure_detail": None}
    return document


__all__ = [
    "CATEGORY_NAMES", "CHANGE_SCHEMA_VERSION", "DEFAULT_BOND_TOLERANCE", "DEFAULT_MATCH_CAP",
    "DEFAULT_MAX_CANDIDATES", "DEFAULT_NEIGHBORHOOD_SHELL", "PREVIEW_COUNTERS", "build_reaction_change",
]
