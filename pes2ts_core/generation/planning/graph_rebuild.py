"""Rebuild complete R/P bond graphs from whitelisted in-memory inputs only.

Sources are exactly two whitelisted inputs supplied by the caller: a mapped
reaction SMILES string (inventory / candidate CSV) and endpoint materials
(map-keyed element + coordinates rows, e.g. from sanitized export-contract
documents).  Graph construction delegates to the frozen G1 graph contract
:func:`pes2ts_core.g1.endpoint_graph.build_endpoint_graph_bundle`; this module
adds (a) validation/normalization of the materials through
:mod:`pes2ts_core.g1.endpoint_materials`, (b) binding of source hashes
(reaction SMILES, normalized materials, map-space graph payload) into a
provenance block on the rebuilt bundle document, and (c) a schema-generic
loader that turns a parsed export-contract document into the materials
mapping.  The module performs no filesystem access — callers parse documents
wherever they choose.  Rebuilt documents never carry truth-derived keys
(design interface commitments, 全图来源 row).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.g1.endpoint_graph import EndpointGraphBundle, build_endpoint_graph_bundle
from pes2ts_core.g1.endpoint_materials import (
    CODE_MATERIALS_SCHEMA,
    MATERIALS_TOP_KEYS,
    REASON_SIDE_INVALID,
    REASON_UNKNOWN_TOP_KEY,
    SIDES,
    EndpointGraphError,
    MaterialRow,
    normalize_side_materials,
)
from pes2ts_core.g1.graph_payload import graph_payload
from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps

# noqa: SIZE_OK — task-specified single module owning the whitelisted-input
# rebuild boundary (rebuild API + its export-doc materials loader); precedent
# endpoint_graph.py (frozen schema + builder in one module).

#: Provenance marker recorded on every rebuilt bundle document.
REBUILD_SOURCE_KIND: Final[str] = "whitelisted_endpoint_rebuild"
#: Document key holding the source-hash binding block.
PROVENANCE_KEY: Final[str] = "provenance"

#: Typed failure codes raised by this module's input boundaries.
CODE_SMILES_INVALID: Final[str] = "REBUILD_SMILES_INVALID"
CODE_EXPORT_SCHEMA: Final[str] = "EXPORT_SCHEMA_INVALID"

REASON_UNPARSEABLE: Final[str] = "unparseable_reaction_smiles"
REASON_MAPS_INVALID: Final[str] = "maps_invalid"
REASON_COORDS_SHAPE: Final[str] = "coordinates_shape_invalid"
REASON_COORD_VALUE: Final[str] = "coordinate_value_invalid"
REASON_ELEMENT_SOURCE_MISSING: Final[str] = "element_source_missing"

#: Export-doc side -> coordinate-list key (schema-generic across contract trees).
COORDINATE_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("r", "r_coordinates"),
    ("p", "p_coordinates"),
)


class RebuildInputError(ValueError):
    """Typed rebuild-input failure; ``code`` is a stable machine-readable token."""

    def __init__(self, code: str, reasons: Sequence[str]) -> None:
        self.code = code
        self.reasons = tuple(reasons)
        super().__init__(f"{code}: {'; '.join(self.reasons)}")


@dataclass(frozen=True, slots=True)
class RebuiltEndpointGraphBundle(EndpointGraphBundle):
    """EndpointGraphBundle plus whitelisted-source hash bindings.

    A rebuild is a pure function of (reaction SMILES, endpoint materials).
    The three digests record exactly which sources produced the bundle:
    the literal SMILES string, the canonical normalized materials rows, and
    the map-space graph payload derived from the SMILES.  ``content_sha256``
    remains the chemistry identity from the frozen G1 graph contract.
    """

    reaction_smiles_sha256: str
    endpoint_materials_sha256: str
    graph_payload_sha256: str

    def to_doc(self) -> dict[str, Any]:
        """Return the bundle document plus the provenance source-hash block."""
        # Explicit parent call: zero-arg super() is unreliable inside methods
        # of a slots=True dataclass subclass (the decorator recreates the
        # class object, orphaning the method's __class__ cell).
        doc = EndpointGraphBundle.to_doc(self)
        doc[PROVENANCE_KEY] = {
            "source_kind": REBUILD_SOURCE_KIND,
            "reaction_smiles_sha256": self.reaction_smiles_sha256,
            "endpoint_materials_sha256": self.endpoint_materials_sha256,
            "graph_payload_sha256": self.graph_payload_sha256,
        }
        return doc


def _normalize_materials(
    endpoint_materials: Mapping[str, Any],
) -> dict[str, dict[int, MaterialRow]]:
    """Validate/normalize both materials sides at the single trust boundary."""
    unknown = sorted(str(key) for key in endpoint_materials if key not in MATERIALS_TOP_KEYS)
    if unknown:
        raise EndpointGraphError(
            CODE_MATERIALS_SCHEMA, [f"{REASON_UNKNOWN_TOP_KEY}: {unknown}"]
        )
    missing = [side for side in SIDES if side not in endpoint_materials]
    if missing:
        raise EndpointGraphError(
            CODE_MATERIALS_SCHEMA, [f"{REASON_SIDE_INVALID}: missing {missing}"]
        )
    return {
        side: normalize_side_materials(endpoint_materials[side], side) for side in SIDES
    }


def _materials_sha256(
    normalized: Mapping[str, Mapping[int, MaterialRow]],
) -> str:
    """Digest the canonical (side, map, element, coordinates) row projection."""
    canonical: list[JSONValue] = []
    for side in sorted(normalized):
        for map_id in sorted(normalized[side]):
            element, coordinates = normalized[side][map_id]
            canonical.append([side, map_id, element, list(coordinates)])
    return hashlib.sha256(stable_json_dumps(canonical).encode("utf-8")).hexdigest()


def _payload_canonical(payload: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Project the map-space payload into a JSON-safe canonical form."""
    return {
        "elements": {str(key): value for key, value in payload["elements"].items()},
        "r_bonds": [list(bond) for bond in payload["r_bonds"]],
        "p_bonds": [list(bond) for bond in payload["p_bonds"]],
        "r_components": {
            str(key): value for key, value in payload["r_components"].items()
        },
        "p_components": {
            str(key): value for key, value in payload["p_components"].items()
        },
    }


def _payload_sha256(reaction_smiles: str) -> str:
    """Digest the canonical map-space payload of the mapped reaction SMILES."""
    payload = graph_payload(reaction_smiles)
    if payload is None:
        raise RebuildInputError(CODE_SMILES_INVALID, [REASON_UNPARSEABLE])
    canonical = _payload_canonical(payload)
    return hashlib.sha256(stable_json_dumps(canonical).encode("utf-8")).hexdigest()


def rebuild_endpoint_graphs(
    reaction_smiles: str, endpoint_materials: Mapping[str, Any]
) -> RebuiltEndpointGraphBundle:
    """Rebuild the complete R/P bond graph from whitelisted inputs.

    Both materials sides are validated/normalized through
    :mod:`pes2ts_core.g1.endpoint_materials`; graph construction delegates to
    :func:`pes2ts_core.g1.endpoint_graph.build_endpoint_graph_bundle` (strict
    both-sides explicit-H parse, element/isotope conservation, explicit-H
    inventory, geometry binding).  The returned bundle binds the source
    hashes of the reaction SMILES, the normalized materials, and the
    map-space graph payload into its document's provenance block.

    Raises
    ------
    RebuildInputError
        ``REBUILD_SMILES_INVALID`` when the SMILES does not parse under the
        strict G1 rules.
    EndpointGraphError
        Materials schema / bijection / conservation violations (typed codes
        from the frozen G1 graph contract).
    """
    payload_digest = _payload_sha256(reaction_smiles)
    normalized = _normalize_materials(endpoint_materials)
    base = build_endpoint_graph_bundle(reaction_smiles, endpoint_materials)
    return RebuiltEndpointGraphBundle(
        schema_version=base.schema_version,
        aromatic_model=base.aromatic_model,
        rdkit_version=base.rdkit_version,
        normalization_version=base.normalization_version,
        r_graph=base.r_graph,
        p_graph=base.p_graph,
        r_geometry=base.r_geometry,
        p_geometry=base.p_geometry,
        conservation=base.conservation,
        content_sha256=base.content_sha256,
        reaction_smiles_sha256=hashlib.sha256(reaction_smiles.encode("utf-8")).hexdigest(),
        endpoint_materials_sha256=_materials_sha256(normalized),
        graph_payload_sha256=payload_digest,
    )


def _export_map_ids(export_doc: Mapping[str, Any]) -> tuple[int, ...]:
    """Extract the positive-unique map ids in map-ascending coordinate order."""
    raw_maps = export_doc.get("maps")
    if not isinstance(raw_maps, list) or not raw_maps:
        raise RebuildInputError(CODE_EXPORT_SCHEMA, [REASON_MAPS_INVALID])
    map_ids: list[int] = []
    for raw in raw_maps:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise RebuildInputError(CODE_EXPORT_SCHEMA, [REASON_MAPS_INVALID])
        map_ids.append(raw)
    if len(set(map_ids)) != len(map_ids):
        raise RebuildInputError(CODE_EXPORT_SCHEMA, [REASON_MAPS_INVALID])
    return tuple(map_ids)


def _export_elements(
    export_doc: Mapping[str, Any], map_ids: Sequence[int]
) -> dict[int, str]:
    """Resolve the map -> element table from atom_rows or the aligned elements list."""
    atom_rows = export_doc.get("atom_rows")
    if isinstance(atom_rows, list):
        by_map: dict[int, str] = {}
        complete = True
        for row in atom_rows:
            if not isinstance(row, Mapping) or "map" not in row or "element" not in row:
                complete = False
                break
            raw_map = row["map"]
            element = row["element"]
            if (
                isinstance(raw_map, bool)
                or not isinstance(raw_map, int)
                or raw_map <= 0
                or not isinstance(element, str)
                or not element
            ):
                complete = False
                break
            by_map[raw_map] = element
        if complete and all(map_id in by_map for map_id in map_ids):
            return {map_id: by_map[map_id] for map_id in map_ids}
    elements = export_doc.get("elements")
    if isinstance(elements, list) and len(elements) == len(map_ids):
        rows: dict[int, str] = {}
        for map_id, raw in zip(map_ids, elements, strict=True):
            if not isinstance(raw, str) or not raw:
                raise RebuildInputError(
                    CODE_EXPORT_SCHEMA, [REASON_ELEMENT_SOURCE_MISSING]
                )
            rows[map_id] = raw
        return rows
    raise RebuildInputError(CODE_EXPORT_SCHEMA, [REASON_ELEMENT_SOURCE_MISSING])


def _coordinate_row(raw: object, coord_key: str, map_id: int) -> list[float]:
    """Parse one export coordinate row into three finite floats."""
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        raise RebuildInputError(
            CODE_EXPORT_SCHEMA,
            [f"{REASON_COORDS_SHAPE}: {coord_key} map={map_id}"],
        )
    coords: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise RebuildInputError(
                CODE_EXPORT_SCHEMA, [f"{REASON_COORD_VALUE}: {coord_key} map={map_id}"]
            )
        number = float(item)
        if not math.isfinite(number):
            raise RebuildInputError(
                CODE_EXPORT_SCHEMA, [f"{REASON_COORD_VALUE}: {coord_key} map={map_id}"]
            )
        coords.append(number)
    return coords


def load_endpoint_materials_from_export(export_doc: Mapping[str, Any]) -> dict[
    str, dict[int, dict[str, Any]]
]:
    """Build endpoint materials from a parsed sanitized export-contract doc.

    Accepts any parsed document carrying the export-contract coordinate
    schema — ``maps`` (positive unique ids, map-ascending), aligned
    ``r_coordinates``/``p_coordinates`` rows, and an element source via
    ``atom_rows`` (map -> element) or the aligned ``elements`` list.  The
    caller chooses which tree the parsed document came from; this function
    is schema-generic and touches only those whitelisted fields, ignoring
    every other key the document may carry.

    Returns the materials mapping ``{side: {map: {"element", "coordinates"}}}``
    that :func:`pes2ts_core.g1.endpoint_graph.build_endpoint_graph_bundle`
    accepts.  Pure in-memory; no filesystem access.

    Raises
    ------
    RebuildInputError
        ``EXPORT_SCHEMA_INVALID`` with a stable reason token for any
        malformed field.
    """
    map_ids = _export_map_ids(export_doc)
    elements_by_map = _export_elements(export_doc, map_ids)
    materials: dict[str, dict[int, dict[str, Any]]] = {}
    for side, coord_key in COORDINATE_KEYS:
        rows = export_doc.get(coord_key)
        if not isinstance(rows, list) or len(rows) != len(map_ids):
            raise RebuildInputError(
                CODE_EXPORT_SCHEMA,
                [f"{REASON_COORDS_SHAPE}: {coord_key} expected {len(map_ids)} rows"],
            )
        side_materials: dict[int, dict[str, Any]] = {}
        for map_id, raw in zip(map_ids, rows, strict=True):
            side_materials[map_id] = {
                "element": elements_by_map[map_id],
                "coordinates": _coordinate_row(raw, coord_key, map_id),
            }
        materials[side] = side_materials
    return materials


__all__ = [
    "CODE_EXPORT_SCHEMA",
    "CODE_SMILES_INVALID",
    "COORDINATE_KEYS",
    "PROVENANCE_KEY",
    "REBUILD_SOURCE_KIND",
    "REASON_COORD_VALUE",
    "REASON_COORDS_SHAPE",
    "REASON_ELEMENT_SOURCE_MISSING",
    "REASON_MAPS_INVALID",
    "REASON_UNPARSEABLE",
    "RebuildInputError",
    "RebuiltEndpointGraphBundle",
    "load_endpoint_materials_from_export",
    "rebuild_endpoint_graphs",
]
