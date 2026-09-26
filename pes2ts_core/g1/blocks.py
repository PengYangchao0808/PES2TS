"""Component-block serialization for the G1 reaction-change document.

The matcher (:mod:`pes2ts_core.g1.skeleton`) works in inventory-local XYZ-row
space, while the persisted tables must speak reaction-global map numbers:
:func:`component_block` composes each query atom position with the side's
per-component map list (``global = map_lists[rxn_index][q]``) for the chosen
rows and for every stored candidate.  Geometry is a soft per-component verdict
(:func:`pes2ts_core.g1.index_map.bond_geometry_check` plus the failing-bond
fraction of :func:`failing_bond_count`, which drives the >50% hard rejection).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from pes2ts_core.g1.index_map import COVALENT_RADII, ComponentIndex, bond_geometry_check
from pes2ts_core.g1.skeleton import ComponentMatch
from pes2ts_core.utils.hashing import JSONValue


@dataclass(frozen=True, slots=True)
class SideSpec:
    """Everything needed to serialize one matched side's component blocks."""

    matches_by_tag: Mapping[str, ComponentMatch]
    map_lists: tuple[tuple[int, ...], ...]
    records_by_tag: Mapping[str, Mapping[str, Any]]
    tolerance: float


def failing_bond_count(record: Mapping[str, Any], tolerance: float) -> tuple[int, int]:
    """Return ``(n_bonds, n_failing)`` of one component's stored bonds.

    A bond fails when its XYZ distance exceeds the Cordero covalent-radius sum
    plus *tolerance*; out-of-range bond maps are skipped (the element
    cross-check rejects such components first).
    """
    numbers = [int(number) for number in record["atomic_numbers"]]
    coordinates = [[float(value) for value in row] for row in record["coordinates"]]
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(record["smiles"]), params)
    if mol is None or mol.GetNumAtoms() == 0:
        return 0, 0
    n_bonds = n_failing = 0
    for bond in mol.GetBonds():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        first = int(bond.GetBeginAtom().GetAtomMapNum())  # pyright: ignore[reportUnknownMemberType]
        second = int(bond.GetEndAtom().GetAtomMapNum())  # pyright: ignore[reportUnknownMemberType]
        if not (1 <= first <= len(numbers) and 1 <= second <= len(numbers)):
            continue
        n_bonds += 1
        limit = COVALENT_RADII[numbers[first - 1]] + COVALENT_RADII[numbers[second - 1]] + tolerance
        if math.dist(coordinates[first - 1], coordinates[second - 1]) > limit:
            n_failing += 1
    return n_bonds, n_failing


def component_block(index: ComponentIndex, side: SideSpec) -> tuple[dict[str, JSONValue], str]:
    """Return one component's document block and its hard-reject detail ("" = none).

    Every map number is translated into the reaction-global space: the first
    isomorphism gives each stored row its query atom position, and each
    candidate pair keeps its tuple position as the query atom index.
    """
    record = side.records_by_tag[index.tag]
    match = side.matches_by_tag[index.tag]
    maps = side.map_lists[match.rxn_index]
    position_of_row = [0] * len(match.isomorphisms[0])
    for position, row_index in enumerate(match.isomorphisms[0]):
        position_of_row[row_index] = position
    rows: list[JSONValue] = [
        {"local_index": row["local_index"], "map": maps[position_of_row[int(row["local_index"])]],
         "element": row["element"], "global_index": row["global_index"]}
        for row in index.rows
    ]
    candidates: list[JSONValue] = [
        [{"map": maps[position], "local_index": local} for position, (_m, local) in enumerate(candidate)]
        for candidate in index.candidates
    ]
    geometry_ok, detail = bond_geometry_check(record, tolerance=side.tolerance)
    hard = ""
    if not geometry_ok:
        n_bonds, n_failing = failing_bond_count(record, side.tolerance)
        if n_bonds and n_failing * 2 > n_bonds:
            hard = f"{index.tag}: {n_failing}/{n_bonds} stored bonds exceed the covalent-radius limit"
    block: dict[str, JSONValue] = {
        "tag": index.tag, "index_base": index.index_base, "n_atoms": index.n_atoms,
        "status": index.status, "n_candidates": index.n_candidates,
        "truncated": index.status == "truncated", "element_check": index.element_check,
        "geometry_check_ok": geometry_ok, "geometry_worst": detail,
        "rows": rows, "candidates": candidates,
    }
    return block, hard


def side_blocks(indexes: Sequence[ComponentIndex], side: SideSpec) -> tuple[list[JSONValue], str]:
    """Return every component block of one side plus the first hard-reject detail."""
    blocks: list[JSONValue] = []
    for index in indexes:
        block, hard = component_block(index, side)
        if hard:
            return [], hard
        blocks.append(block)
    return blocks, ""


def ambiguity(indexes: Sequence[ComponentIndex], matches: Sequence[ComponentMatch]) -> dict[str, JSONValue]:
    """Return the worst index status and the pairing-ambiguity flag."""
    statuses = {index.status for index in indexes}
    if "truncated" in statuses:
        index_status = "truncated"
    elif "symmetric_ambiguous" in statuses:
        index_status = "ambiguous"
    else:
        index_status = "unique"
    pairing = "ambiguous" if any(len(match.tag_candidates) > 1 for match in matches) else "unique"
    return {"index": index_status, "pairing": pairing}


__all__ = ["SideSpec", "ambiguity", "component_block", "failing_bond_count", "side_blocks"]
