"""Endpoint-only mapped-graph payload extracted once for the P1 document.

The P2 classifier needs the R/P bond graphs in reaction-map space (context
shells, ring counts, component counts) but must not re-parse or re-derive
anything from the truth side.  This module therefore turns the *mapped
reaction SMILES* -- endpoint data the inventory already owns -- into a small
deterministic payload stored inside the P1 document: the element of every
map number, the bond list of each side with bond orders read through the
same raw-graph view the G1 diff uses (aromatic bonds stay ``1.5``), and the
component ordinal of every map.  No truth accessor is touched here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

from rdkit import RDLogger

from pes2ts_core.g1.parse import ReactionParseError, parse_reaction

RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

logger = logging.getLogger(__name__)

#: Bond-order rendering of aromatic bonds (both sides parsed identically).
AROMATIC_ORDER: Final[float] = 1.5


def graph_payload(reaction_smiles: str) -> dict[str, Any] | None:
    """Return the map-space graph payload of one mapped reaction SMILES.

    The payload carries ``elements`` (map -> element symbol, agreeing across
    sides by construction), ``r_bonds``/``p_bonds`` (sorted
    ``[map_a, map_b, order]`` triples), and ``r_components``/
    ``p_components`` (map -> zero-based ordinal of appearance).  Returns
    ``None`` when the SMILES does not parse under the strict G1 rules; the
    caller maps that to the typed ``source_structure_mismatch`` status.
    """
    try:
        reactants, products = parse_reaction(reaction_smiles)
    except ReactionParseError:
        return None
    elements: dict[int, str] = {}
    payload: dict[str, Any] = {
        "elements": elements, "r_bonds": [], "p_bonds": [],
        "r_components": {}, "p_components": {},
    }
    for side_key, side in (("r", reactants), ("p", products)):
        bonds: list[tuple[int, int, float]] = []
        components: dict[int, int] = {}
        for ordinal, (mol, map_list) in enumerate(zip(side.mols, side.map_lists, strict=True)):
            for atom, map_number in zip(mol.GetAtoms(), map_list, strict=True):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                elements[int(map_number)] = str(atom.GetSymbol())  # pyright: ignore[reportUnknownMemberType]
                components[int(map_number)] = ordinal
            for bond in mol.GetBonds():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                first = int(map_list[bond.GetBeginAtomIdx()])  # pyright: ignore[reportUnknownMemberType]
                second = int(map_list[bond.GetEndAtomIdx()])  # pyright: ignore[reportUnknownMemberType
                pair = (first, second) if first <= second else (second, first)
                bonds.append((pair[0], pair[1], float(bond.GetBondTypeAsDouble())))  # pyright: ignore[reportUnknownMemberType]
        payload[f"{side_key}_bonds"] = sorted(bonds)
        payload[f"{side_key}_components"] = dict(sorted(components.items()))
    return payload


def payload_elements_by_map(payload: Mapping[str, Any]) -> dict[int, str]:
    """Return the payload's map -> element mapping keyed by integer map."""
    return {int(map_number): str(symbol) for map_number, symbol in payload["elements"].items()}


__all__ = ["AROMATIC_ORDER", "graph_payload", "payload_elements_by_map"]
