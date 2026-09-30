"""Whitelisted endpoint-materials schema for EndpointGraphBundle construction.

Untrusted endpoint materials cross the trust boundary exactly once here:
map-keyed (or map-list) rows of ``element`` + ``coordinates`` are parsed into
typed ``(element, coordinates)`` rows and required to bijection-match the
mapped reaction SMILES.  Violations raise :class:`EndpointGraphError` with a
stable machine-readable ``code`` — never a partial result (design §3.1).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

SIDES: Final[tuple[str, ...]] = ("r", "p")
MATERIALS_TOP_KEYS: Final[frozenset[str]] = frozenset(SIDES)
MATERIALS_ATOM_KEYS: Final[frozenset[str]] = frozenset({"map", "element", "coordinates"})

CODE_MAP_INVALID: Final[str] = "MAP_INVALID"
CODE_MAP_AMBIGUOUS: Final[str] = "MAP_AMBIGUOUS"
CODE_MATERIALS_SCHEMA: Final[str] = "MATERIALS_SCHEMA_INVALID"
CODE_H_INVENTORY: Final[str] = "H_INVENTORY_MISMATCH"

REASON_ORDER_BINDING: Final[str] = "order_based_binding_forbidden"
REASON_DUPLICATE_MAP: Final[str] = "duplicate_map_in_materials"
REASON_MAP_SET_MISMATCH: Final[str] = "map_set_mismatch"
REASON_EXTRA_MAP: Final[str] = "extra_materials_map"
REASON_ELEMENT_MISMATCH: Final[str] = "materials_element_mismatch"
REASON_UNKNOWN_TOP_KEY: Final[str] = "unknown_top_level_key"
REASON_UNKNOWN_ATOM_KEY: Final[str] = "unknown_atom_key"
REASON_COORDINATES_INVALID: Final[str] = "coordinates_invalid"
REASON_ENTRY_INVALID: Final[str] = "atom_entry_invalid"
REASON_SIDE_INVALID: Final[str] = "side_materials_invalid"
REASON_EXPLICIT_H: Final[str] = "explicit_h_inventory_mismatch"

#: One validated materials row: (element symbol, (x, y, z) in angstrom).
MaterialRow = tuple[str, tuple[float, float, float]]


class EndpointGraphError(ValueError):
    """Typed validation failure; ``code`` is a stable machine-readable token."""

    def __init__(self, code: str, reasons: Sequence[str]) -> None:
        self.code = code
        self.reasons = tuple(reasons)
        super().__init__(f"{code}: {'; '.join(self.reasons)}")


def _coerce_map_id(key: object) -> int:
    """Coerce a materials map key (int or decimal string) to a positive int."""
    if isinstance(key, bool) or not isinstance(key, int | str):
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_ENTRY_INVALID])
    try:
        map_id = int(key)
    except ValueError as error:
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_ENTRY_INVALID]) from error
    if map_id <= 0 or (isinstance(key, str) and str(map_id) != key.strip()):
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_ENTRY_INVALID])
    return map_id


def _parse_coordinates(value: object) -> tuple[float, float, float]:
    """Parse a 3-finite-float coordinate triple or raise a schema error."""
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_COORDINATES_INVALID])
    coords: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_COORDINATES_INVALID])
        number = float(item)
        if not math.isfinite(number):
            raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_COORDINATES_INVALID])
        coords.append(number)
    return (coords[0], coords[1], coords[2])


def _parse_atom_entry(raw: object) -> MaterialRow:
    """Parse one materials atom entry into (element, coordinates)."""
    if not isinstance(raw, Mapping):
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_ENTRY_INVALID])
    if set(raw) - MATERIALS_ATOM_KEYS:
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_UNKNOWN_ATOM_KEY])
    element = raw.get("element")
    if not isinstance(element, str) or not element:
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_ENTRY_INVALID])
    if "coordinates" not in raw:
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_COORDINATES_INVALID])
    return element, _parse_coordinates(raw["coordinates"])


def normalize_side_materials(side_value: object, side: str) -> dict[int, MaterialRow]:
    """Normalize one side's materials (map-dict or map-list form) to typed rows."""
    entries: list[tuple[object, object]] = []
    if isinstance(side_value, Mapping):
        entries = list(side_value.items())
    elif isinstance(side_value, list):
        for index, raw in enumerate(side_value):
            if not isinstance(raw, Mapping):
                raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_ENTRY_INVALID])
            if set(raw) - MATERIALS_ATOM_KEYS:
                raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_UNKNOWN_ATOM_KEY])
            if "map" not in raw:
                raise EndpointGraphError(
                    CODE_MAP_AMBIGUOUS, [f"{side}[{index}]: {REASON_ORDER_BINDING}"]
                )
            entries.append((raw["map"], raw))
    else:
        raise EndpointGraphError(CODE_MATERIALS_SCHEMA, [REASON_SIDE_INVALID])
    materials: dict[int, MaterialRow] = {}
    for key, raw in entries:
        map_id = _coerce_map_id(key)
        if map_id in materials:
            raise EndpointGraphError(
                CODE_MAP_INVALID, [f"{side}: {REASON_DUPLICATE_MAP} map={map_id}"]
            )
        materials[map_id] = _parse_atom_entry(raw)
    return materials


def require_bijection(
    materials: Mapping[int, MaterialRow],
    node_elements: Mapping[int, str],
    side: str,
) -> None:
    """Require materials ↔ SMILES-map bijection with element agreement.

    ``node_elements`` maps every SMILES map number of that side to its element
    symbol; missing material maps that are all hydrogens raise
    ``H_INVENTORY_MISMATCH`` (classified by the caller-supplied elements).
    """
    graph_maps = set(node_elements)
    material_maps = set(materials)
    missing = sorted(graph_maps - material_maps)
    extra = sorted(material_maps - graph_maps)
    if extra:
        raise EndpointGraphError(
            CODE_MAP_INVALID, [f"{side}: {REASON_EXTRA_MAP} maps={extra}"]
        )
    if missing:
        details = [f"map={m}" for m in missing]
        if all(node_elements[m] == "H" for m in missing):
            raise EndpointGraphError(
                CODE_H_INVENTORY, [f"{side}: {REASON_EXPLICIT_H} {details}"]
            )
        raise EndpointGraphError(
            CODE_MAP_INVALID, [f"{side}: {REASON_MAP_SET_MISMATCH} {details}"]
        )
    mismatches = [
        f"map={m} materials={materials[m][0]} smiles={node_elements[m]}"
        for m in sorted(graph_maps)
        if materials[m][0] != node_elements[m]
    ]
    if mismatches:
        raise EndpointGraphError(
            CODE_MAP_INVALID, [f"{side}: {REASON_ELEMENT_MISMATCH} {'; '.join(mismatches)}"]
        )


__all__ = [
    "CODE_H_INVENTORY",
    "CODE_MAP_AMBIGUOUS",
    "CODE_MAP_INVALID",
    "CODE_MATERIALS_SCHEMA",
    "MATERIALS_ATOM_KEYS",
    "MATERIALS_TOP_KEYS",
    "REASON_COORDINATES_INVALID",
    "REASON_DUPLICATE_MAP",
    "REASON_ELEMENT_MISMATCH",
    "REASON_ENTRY_INVALID",
    "REASON_EXPLICIT_H",
    "REASON_EXTRA_MAP",
    "REASON_MAP_SET_MISMATCH",
    "REASON_ORDER_BINDING",
    "REASON_SIDE_INVALID",
    "REASON_UNKNOWN_ATOM_KEY",
    "REASON_UNKNOWN_TOP_KEY",
    "EndpointGraphError",
    "MaterialRow",
    "SIDES",
    "normalize_side_materials",
    "require_bijection",
]
