"""G1 endpoint-graph adaptation into the predictor's plain-data contracts (B1).

Reads the existing G1 authority artifacts (P1 mapping documents and the G1 v2
whitelisted export) and converts them into the runtime
:class:`~pes2ts_core.pesn.contracts.inputs.TaskGraph` /
:class:`~pes2ts_core.pesn.contracts.inputs.CoordinateSpec` shapes.  This
adapter never re-decides edits, roles, or permissions — it only converts and
numerically prepares what the topology authority already froze.

Archive invariant reused here (re-verified per reaction by G1 P1): IRC/TS atom
rows follow the reaction map-number order, so atom row ``i`` corresponds to
map ``i + 1`` and map-space pairs translate to row pairs by subtracting one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Mapping

from pes2ts_core.pesn.contracts.inputs import (
    CoordinateDef,
    CoordinateSpec,
    CoordinateSpec as _CoordSpec,
    TaskEdit,
    TaskGraph,
    BondTerm,
)
from pes2ts_core.utils.jsonio import read_json

ADAPTER_SCHEMA_VERSION: Final[str] = "pesn_endpoint_adapter_v1"
#: Provenance string carried into every adapted task graph.
TASK_SOURCE: Final[str] = "g1_p1_mapping+g1v2_export"
#: v1 distance-coordinate scale (angstrom) and reference; frozen in LabelSpec.
DRIVER_SCALE_A: Final[float] = 1.0
DRIVER_REF_Q: Final[float] = 0.0
COORDINATE_SPEC_VERSION: Final[str] = "pesn_driver_coords_v1"

EDIT_KIND_TO_ROLE: Final[dict[str, str]] = {
    "formed": "driver",
    "broken": "driver",
    "order_changed": "driver",
}


class EndpointAdapterError(RuntimeError):
    """The authority documents cannot be adapted (typed, per reaction)."""


def shard_name(reaction_id: str, shard_size: int) -> str:
    numeric = int(reaction_id.rsplit("_", 1)[-1])
    return f"{numeric // shard_size:05d}"


def load_authority_document(
    interim_dir: str | Path, reaction_id: str, *, shard_size: int
) -> dict[str, Any]:
    """Read the P1 mapping + v2 export documents of one reaction."""
    interim = Path(interim_dir)
    shard = shard_name(reaction_id, shard_size)
    p1_path = interim / "g1_truth" / "p1_mapping" / shard / f"{reaction_id}.json"
    export_path = interim / "g1_v2" / "export" / shard / f"{reaction_id}.json"
    for path in (p1_path, export_path):
        if not path.is_file():
            raise EndpointAdapterError(f"authority document missing: {path}")
    return {"p1": read_json(p1_path), "export": read_json(export_path)}


def adapt_reaction(docs: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the authority documents into one adapter view (map space)."""
    p1 = docs["p1"]
    export = docs["export"]
    graph = p1.get("graph")
    irc_validation = p1.get("irc_validation")
    if not isinstance(graph, dict) or not isinstance(irc_validation, dict):
        raise EndpointAdapterError("p1 document lacks graph or irc_validation")
    elements_raw = graph.get("elements")
    if not isinstance(elements_raw, dict) or not elements_raw:
        raise EndpointAdapterError("p1 graph has no elements mapping")
    # JSON object keys arrive as strings; normalize to map numbers once.
    elements = {int(k): v for k, v in elements_raw.items()}
    maps = sorted(int(m) for m in elements)
    if maps != list(range(1, len(maps) + 1)):
        raise EndpointAdapterError("map numbers are not a dense 1..n sequence")
    r_bonds = [
        (int(a), int(b), float(order))
        for a, b, order in (tuple(e) for e in graph.get("r_bonds", []))
    ]
    p_bonds = [
        (int(a), int(b), float(order))
        for a, b, order in (tuple(e) for e in graph.get("p_bonds", []))
    ]
    edits = []
    for record in export.get("edits", []):
        pair = record.get("pair")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise EndpointAdapterError(f"export edit pair malformed: {record!r}")
        edits.append(
            {
                "kind": str(record["edit_kind"]),
                "maps": (int(pair[0]), int(pair[1])),
                "r_order": record.get("r_bond_order"),
                "p_order": record.get("p_bond_order"),
            }
        )
    classification = export.get("classification")
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "reaction_id": str(p1["reaction_id"]),
        "atomic_numbers_by_map": [int(_element_to_z(str(elements[m]))) for m in maps],
        "atom_map": maps,
        "r_bonds": r_bonds,
        "p_bonds": p_bonds,
        "edits": edits,
        "l0u_family": str(classification.get("l0u_cluster_id"))
        if isinstance(classification, dict)
        else None,
        "l0_family": str(classification.get("l0_cluster_id"))
        if isinstance(classification, dict)
        else None,
        "charge_total_reactants": export.get("charge_total_reactants"),
        "multiplicity_max": export.get("multiplicity_max"),
        "p1_status": p1.get("status"),
        "layout": irc_validation.get("layout"),
        "ts_frame_index": irc_validation.get("ts_frame_index"),
        "splice_index": irc_validation.get("splice_index"),
        "orientation": irc_validation.get("orientation"),
        "endpoint_match": irc_validation.get("endpoint_match"),
        "quality_flags": irc_validation.get("quality_flags"),
        "mapping_provenance": export.get("mapping_provenance"),
    }


_ELEMENT_TO_Z: Final[dict[str, int]] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33,
    "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41,
    "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
    "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81,
    "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
}


def _element_to_z(element: str) -> int:
    z = _ELEMENT_TO_Z.get(element)
    if z is None:
        raise EndpointAdapterError(f"unknown element symbol {element!r}")
    return z


def to_task_graph(adapter: Mapping[str, Any]) -> TaskGraph:
    """Convert the map-space adapter view into a row-space TaskGraph."""
    maps = list(adapter["atom_map"])
    map_to_row = {m: i for i, m in enumerate(maps)}
    atomic_numbers = tuple(int(z) for z in adapter["atomic_numbers_by_map"])

    def _bond_terms(bonds) -> tuple[BondTerm, ...]:
        return tuple(
            BondTerm(i=map_to_row[int(a)], j=map_to_row[int(b)], order=float(order))
            for a, b, order in bonds
        )

    edits = tuple(
        TaskEdit(
            kind=str(edit["kind"]),
            i=map_to_row[edit["maps"][0]],
            j=map_to_row[edit["maps"][1]],
            reactant_order=float(edit["r_order"]) if edit["r_order"] is not None else 0.0,
            product_order=float(edit["p_order"]) if edit["p_order"] is not None else 0.0,
        )
        for edit in adapter["edits"]
    )
    graph = TaskGraph(
        schema_version="pesn_task_graph_v1",
        atomic_numbers=atomic_numbers,
        atom_map=tuple(maps),
        reactant_bonds=_bond_terms(adapter["r_bonds"]),
        product_bonds=_bond_terms(adapter["p_bonds"]),
        edits=edits,
        source=TASK_SOURCE,
        provenance={
            "mapping_provenance": str(adapter.get("mapping_provenance") or ""),
            "p1_status": str(adapter.get("p1_status") or ""),
        },
    )
    graph.validate()
    return graph


def to_coordinate_spec(adapter: Mapping[str, Any]) -> _CoordSpec:
    """Driver distance coordinates for every declared edit (v1: F/B/O only)."""
    maps = list(adapter["atom_map"])
    map_to_row = {m: i for i, m in enumerate(maps)}
    coords = []
    for index, edit in enumerate(adapter["edits"]):
        role = EDIT_KIND_TO_ROLE.get(str(edit["kind"]))
        if role is None:
            continue
        a, b = edit["maps"]
        coords.append(
            CoordinateDef(
                kind="distance",
                atoms=(map_to_row[a], map_to_row[b]),
                units="angstrom",
                role=role,
                scale_a=DRIVER_SCALE_A,
                ref_q=DRIVER_REF_Q,
                label=f"{edit['kind']}:{a}-{b}#{index}",
            )
        )
    spec = CoordinateSpec(
        schema_version="pesn_coordinate_spec_v1",
        coords=tuple(coords),
        spec_version=COORDINATE_SPEC_VERSION,
    )
    spec.validate(len(maps))
    return spec


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "COORDINATE_SPEC_VERSION",
    "DRIVER_REF_Q",
    "DRIVER_SCALE_A",
    "TASK_SOURCE",
    "EndpointAdapterError",
    "adapt_reaction",
    "load_authority_document",
    "shard_name",
    "to_coordinate_spec",
    "to_task_graph",
]
