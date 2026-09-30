"""EndpointGraphBundle: complete R/P bond graphs from whitelisted materials.

Both sides of a mapped reaction SMILES are parsed through the strict G1 path
(:func:`pes2ts_core.g1.parse.parse_reaction`, ``removeHs=False``) — same code
path both sides, no tautomer normalization, aromatic bonds stay unkekulized at
order 1.5 (consistent with :mod:`pes2ts_core.g1.bond_changes`).  Whitelisted
endpoint materials are validated in :mod:`pes2ts_core.g1.endpoint_materials`
into a bijection with the SMILES maps; element/isotope conservation and the
explicit-H inventory are checked here; ``content_sha256`` is invariant to
template atom ordering and rigid translation of endpoint coordinates.  The
output document never contains truth-derived keys (design §3.1-§3.2, §13.1).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import rdkit
from rdkit import RDLogger

from pes2ts_core.g1.endpoint_materials import (
    CODE_H_INVENTORY,
    CODE_MAP_AMBIGUOUS,
    CODE_MAP_INVALID,
    CODE_MATERIALS_SCHEMA,
    MATERIALS_TOP_KEYS,
    REASON_EXPLICIT_H,
    REASON_MAP_SET_MISMATCH,
    REASON_SIDE_INVALID,
    REASON_UNKNOWN_TOP_KEY,
    EndpointGraphError,
    MaterialRow,
    SIDES,
    normalize_side_materials,
    require_bijection,
)
from pes2ts_core.g1.parse import SideComponents, parse_reaction
from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps

# noqa: SIZE_OK — design §13.1 single graph-contract module (frozen bundle
# schema + builder + identity); materials trust boundary already split out.

RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

SCHEMA_VERSION: Final[str] = "g1_endpoint_graph_v1"
AROMATIC_MODEL: Final[str] = "rdkit_default_unkekulized"
NORMALIZATION_VERSION: Final[str] = "endpoint_graph_norm_v1"
GEOMETRY_UNIT: Final[str] = "angstrom"

CODE_ELEMENT_IMBALANCE: Final[str] = "ELEMENT_IMBALANCE"
CODE_ISOTOPE_IMBALANCE: Final[str] = "ISOTOPE_IMBALANCE"
REASON_ELEMENT_IMBALANCE: Final[str] = "element_imbalance"
REASON_ISOTOPE_IMBALANCE: Final[str] = "isotope_imbalance"


@dataclass(frozen=True, slots=True)
class AtomNode:
    """One mapped atom: map_id/element/isotope/charge/radicals/H-neighbors/aromatic/stereo."""

    map_id: int
    element: str
    isotope: int
    formal_charge: int
    radical_electrons: int
    explicit_H_neighbors: int
    aromatic: bool
    stereo: str


@dataclass(frozen=True, slots=True)
class BondEdge:
    """One mapped bond: unordered pair (map_a < map_b) + type/order/aromatic/stereo."""

    map_a: int
    map_b: int
    connection_type: str
    bond_order: float
    aromatic: bool
    stereo: str


@dataclass(frozen=True, slots=True)
class GraphComponent:
    """Connected component; ``component_id`` ordered by smallest map."""

    component_id: int
    map_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SideGraph:
    """One endpoint graph: nodes sorted by map, edges in canonical order."""

    nodes: tuple[AtomNode, ...]
    edges: tuple[BondEdge, ...]
    components: tuple[GraphComponent, ...]


@dataclass(frozen=True, slots=True)
class GeometryRef:
    """Geometry binding: unit + SHA256 over centroid-centered map-sorted coordinates."""

    unit: str
    coordinates_sha256: str
    n_points: int


@dataclass(frozen=True, slots=True)
class ConservationReport:
    """R→P conservation outcome and per-side inventory counts."""

    map_ids: tuple[int, ...]
    element_conserved: bool
    isotope_conserved: bool
    explicit_h_count_r: int
    explicit_h_count_p: int
    n_atoms_r: int
    n_atoms_p: int
    n_components_r: int
    n_components_p: int


@dataclass(frozen=True, slots=True)
class EndpointGraphBundle:
    """Complete R/P endpoint graphs + conservation + identity hash (design §3.1)."""

    schema_version: str
    aromatic_model: str
    rdkit_version: str
    normalization_version: str
    r_graph: SideGraph
    p_graph: SideGraph
    r_geometry: GeometryRef
    p_geometry: GeometryRef
    conservation: ConservationReport
    content_sha256: str

    def to_doc(self) -> dict[str, Any]:
        """Return the JSON-safe document; no truth-derived keys ever."""
        cons = self.conservation
        return {
            "schema_version": self.schema_version,
            "aromatic_model": self.aromatic_model,
            "rdkit_version": self.rdkit_version,
            "normalization_version": self.normalization_version,
            "content_sha256": self.content_sha256,
            "r_graph": _graph_doc(self.r_graph),
            "p_graph": _graph_doc(self.p_graph),
            "r_geometry": _geometry_doc(self.r_geometry),
            "p_geometry": _geometry_doc(self.p_geometry),
            "conservation": {
                "map_ids": list(cons.map_ids),
                "element_conserved": cons.element_conserved,
                "isotope_conserved": cons.isotope_conserved,
                "explicit_h_count_r": cons.explicit_h_count_r,
                "explicit_h_count_p": cons.explicit_h_count_p,
                "n_atoms_r": cons.n_atoms_r,
                "n_atoms_p": cons.n_atoms_p,
                "n_components_r": cons.n_components_r,
                "n_components_p": cons.n_components_p,
            },
        }


def _graph_doc(graph: SideGraph) -> dict[str, Any]:
    """JSON-safe projection of one side graph (hash input and document share it)."""
    return {
        "nodes": [
            {
                "map_id": n.map_id, "element": n.element, "isotope": n.isotope,
                "formal_charge": n.formal_charge, "radical_electrons": n.radical_electrons,
                "explicit_H_neighbors": n.explicit_H_neighbors, "aromatic": n.aromatic,
                "stereo": n.stereo,
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "map_a": e.map_a, "map_b": e.map_b, "connection_type": e.connection_type,
                "bond_order": e.bond_order, "aromatic": e.aromatic, "stereo": e.stereo,
            }
            for e in graph.edges
        ],
        "components": [
            {"component_id": c.component_id, "map_ids": list(c.map_ids)}
            for c in graph.components
        ],
    }


def _geometry_doc(ref: GeometryRef) -> dict[str, Any]:
    """JSON-safe projection of one geometry binding."""
    return {"unit": ref.unit, "coordinates_sha256": ref.coordinates_sha256, "n_points": ref.n_points}


def _find(parent: dict[int, int], node: int) -> int:
    """Union-find root with path halving."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _components(map_ids: Sequence[int], edges: Sequence[tuple[int, int]]) -> tuple[GraphComponent, ...]:
    """Connected components ordered by their smallest map number."""
    parent = {m: m for m in map_ids}
    for left, right in edges:
        root_left, root_right = _find(parent, left), _find(parent, right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)
    groups: dict[int, list[int]] = {}
    for map_id in parent:
        groups.setdefault(_find(parent, map_id), []).append(map_id)
    return tuple(
        GraphComponent(index, tuple(sorted(group)))
        for index, group in enumerate(sorted(groups.values()))
    )


def _side_graph(side: SideComponents) -> tuple[SideGraph, dict[int, AtomNode]]:
    """Build one side's graph from the strict parse; nodes/edges canonically sorted."""
    nodes_by_map: dict[int, AtomNode] = {}
    edge_keys: list[tuple[int, int, str, float, bool, str]] = []
    for mol, map_list in zip(side.mols, side.map_lists, strict=True):
        for atom, map_number in zip(mol.GetAtoms(), map_list, strict=True):
            map_id = int(map_number)
            nodes_by_map[map_id] = AtomNode(
                map_id=map_id,
                element=str(atom.GetSymbol()),
                isotope=int(atom.GetIsotope()),
                formal_charge=int(atom.GetFormalCharge()),
                radical_electrons=int(atom.GetNumRadicalElectrons()),
                explicit_H_neighbors=sum(
                    1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 1
                ),
                aromatic=bool(atom.GetIsAromatic()),
                stereo=str(atom.GetChiralTag()),
            )
        for bond in mol.GetBonds():
            first = int(map_list[bond.GetBeginAtomIdx()])
            second = int(map_list[bond.GetEndAtomIdx()])
            pair = (first, second) if first < second else (second, first)
            edge_keys.append((
                pair[0], pair[1], str(bond.GetBondType()),
                float(bond.GetBondTypeAsDouble()), bool(bond.GetIsAromatic()),
                str(bond.GetStereo()),
            ))
    map_ids = sorted(nodes_by_map)
    edges = tuple(BondEdge(*record) for record in sorted(edge_keys))
    graph = SideGraph(
        nodes=tuple(nodes_by_map[m] for m in map_ids),
        edges=edges,
        components=_components(map_ids, [(e.map_a, e.map_b) for e in edges]),
    )
    return graph, nodes_by_map


def _check_map_sets(r_nodes: Mapping[int, AtomNode], p_nodes: Mapping[int, AtomNode]) -> None:
    """Require identical R/P map sets; classify hydrogen-only gaps as H inventory."""
    only_r = sorted(set(r_nodes) - set(p_nodes))
    only_p = sorted(set(p_nodes) - set(r_nodes))
    if not only_r and not only_p:
        return
    missing = only_r + only_p
    elements = [r_nodes[m].element if m in r_nodes else p_nodes[m].element for m in missing]
    details = [f"map={m}" for m in missing]
    if all(element == "H" for element in elements):
        raise EndpointGraphError(
            CODE_H_INVENTORY,
            [f"smiles: {REASON_EXPLICIT_H} R_only={only_r} P_only={only_p} {details}"],
        )
    raise EndpointGraphError(
        CODE_MAP_INVALID,
        [f"smiles: {REASON_MAP_SET_MISMATCH} R_only={only_r} P_only={only_p}"],
    )


def _conservation(
    r_nodes: Mapping[int, AtomNode],
    p_nodes: Mapping[int, AtomNode],
    r_graph: SideGraph,
    p_graph: SideGraph,
) -> ConservationReport:
    """Check element/isotope conservation across R→P; raise typed errors."""
    common = sorted(set(r_nodes) & set(p_nodes))
    element_bad = [
        f"map={m} R={r_nodes[m].element} P={p_nodes[m].element}"
        for m in common
        if r_nodes[m].element != p_nodes[m].element
    ]
    if element_bad:
        raise EndpointGraphError(
            CODE_ELEMENT_IMBALANCE, [f"{REASON_ELEMENT_IMBALANCE} {'; '.join(element_bad)}"]
        )
    isotope_bad = [
        f"map={m} R_isotope={r_nodes[m].isotope} P_isotope={p_nodes[m].isotope}"
        for m in common
        if r_nodes[m].isotope != p_nodes[m].isotope
    ]
    if isotope_bad:
        raise EndpointGraphError(
            CODE_ISOTOPE_IMBALANCE, [f"{REASON_ISOTOPE_IMBALANCE} {'; '.join(isotope_bad)}"]
        )
    return ConservationReport(
        map_ids=tuple(common),
        element_conserved=True,
        isotope_conserved=True,
        explicit_h_count_r=sum(1 for node in r_nodes.values() if node.element == "H"),
        explicit_h_count_p=sum(1 for node in p_nodes.values() if node.element == "H"),
        n_atoms_r=len(r_nodes),
        n_atoms_p=len(p_nodes),
        n_components_r=len(r_graph.components),
        n_components_p=len(p_graph.components),
    )


def _geometry_ref(coordinates: Mapping[int, tuple[float, float, float]]) -> GeometryRef:
    """Bind geometry via SHA256 over centroid-centered, map-sorted coordinates."""
    maps = sorted(coordinates)
    count = len(maps)
    center = tuple(sum(coordinates[m][axis] for m in maps) / count for axis in range(3))
    centered: list[JSONValue] = []
    for m in maps:
        row: list[JSONValue] = [m]
        for axis in range(3):
            row.append(round(coordinates[m][axis] - center[axis], 8))
        centered.append(row)
    digest = hashlib.sha256(stable_json_dumps(centered).encode()).hexdigest()
    return GeometryRef(unit=GEOMETRY_UNIT, coordinates_sha256=digest, n_points=count)


def build_endpoint_graph_bundle(
    reaction_smiles: str, endpoint_materials: Mapping[str, Any]
) -> EndpointGraphBundle:
    """Build and validate the complete R/P endpoint graph bundle.

    ``endpoint_materials`` must whitelist-map exactly the SMILES map numbers
    per side (``r``/``p``) to ``{"element": str, "coordinates": [x, y, z]}``
    entries — dict keyed by map id, or a list of entries carrying a ``map``
    key.  Raises :class:`EndpointGraphError` with a typed ``code`` on any
    conservation, bijection, or schema violation; never returns a partial
    bundle.
    """
    reactants, products = parse_reaction(reaction_smiles)
    r_graph, r_nodes = _side_graph(reactants)
    p_graph, p_nodes = _side_graph(products)
    _check_map_sets(r_nodes, p_nodes)
    unknown_top = [str(k) for k in endpoint_materials if k not in MATERIALS_TOP_KEYS]
    if unknown_top:
        raise EndpointGraphError(
            CODE_MATERIALS_SCHEMA, [f"{REASON_UNKNOWN_TOP_KEY}: {sorted(unknown_top)}"]
        )
    missing_sides = [side for side in SIDES if side not in endpoint_materials]
    if missing_sides:
        raise EndpointGraphError(
            CODE_MATERIALS_SCHEMA, [f"{REASON_SIDE_INVALID}: missing {missing_sides}"]
        )
    r_materials: dict[int, MaterialRow] = normalize_side_materials(endpoint_materials["r"], "r")
    p_materials: dict[int, MaterialRow] = normalize_side_materials(endpoint_materials["p"], "p")
    require_bijection(r_materials, {m: node.element for m, node in r_nodes.items()}, "r")
    require_bijection(p_materials, {m: node.element for m, node in p_nodes.items()}, "p")
    conservation = _conservation(r_nodes, p_nodes, r_graph, p_graph)
    r_geometry = _geometry_ref({m: row[1] for m, row in r_materials.items()})
    p_geometry = _geometry_ref({m: row[1] for m, row in p_materials.items()})
    content: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "aromatic_model": AROMATIC_MODEL,
        "rdkit_version": rdkit.__version__,
        "normalization_version": NORMALIZATION_VERSION,
        "r_graph": _graph_doc(r_graph),
        "p_graph": _graph_doc(p_graph),
        "r_geometry": _geometry_doc(r_geometry),
        "p_geometry": _geometry_doc(p_geometry),
        "conservation": {
            "map_ids": list(conservation.map_ids),
            "element_conserved": conservation.element_conserved,
            "isotope_conserved": conservation.isotope_conserved,
            "explicit_h_count_r": conservation.explicit_h_count_r,
            "explicit_h_count_p": conservation.explicit_h_count_p,
            "n_atoms_r": conservation.n_atoms_r,
            "n_atoms_p": conservation.n_atoms_p,
            "n_components_r": conservation.n_components_r,
            "n_components_p": conservation.n_components_p,
        },
    }
    content_sha256 = hashlib.sha256(stable_json_dumps(content).encode()).hexdigest()
    return EndpointGraphBundle(
        schema_version=SCHEMA_VERSION,
        aromatic_model=AROMATIC_MODEL,
        rdkit_version=rdkit.__version__,
        normalization_version=NORMALIZATION_VERSION,
        r_graph=r_graph,
        p_graph=p_graph,
        r_geometry=r_geometry,
        p_geometry=p_geometry,
        conservation=conservation,
        content_sha256=content_sha256,
    )


__all__ = [
    "AROMATIC_MODEL", "CODE_ELEMENT_IMBALANCE", "CODE_H_INVENTORY",
    "CODE_ISOTOPE_IMBALANCE", "CODE_MAP_AMBIGUOUS", "CODE_MAP_INVALID",
    "CODE_MATERIALS_SCHEMA", "GEOMETRY_UNIT", "NORMALIZATION_VERSION",
    "SCHEMA_VERSION", "AtomNode", "BondEdge", "ConservationReport",
    "EndpointGraphBundle", "EndpointGraphError", "GeometryRef",
    "GraphComponent", "SideGraph", "build_endpoint_graph_bundle",
]
