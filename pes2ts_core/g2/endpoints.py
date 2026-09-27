"""G2 endpoint construction: atom tables, rigid alignment, and BFS assembly.

This module turns a G1 reaction-change document plus its inventory row into
map-ordered per-side atom tables, validates the cross-side contract, derives
map-space bond-pair sets, provides the deterministic 1-to-1 placement
primitive, and assembles multi-component endpoints through a cross-side
bipartite BFS with candidate scoring, changed-pair separation, and the
terminal collision check.  Nothing here reads truth data or spawns processes.
"""

# allow: SIZE_OK -- the plan contract fixes every endpoint-construction
# primitive (tables, validation, bond pairs, anchors, Kabsch, placement, BFS
# assembly, separation, scoring, file writing) in this single module; tasks 2
# and 3 both extend the same file by contract.

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.g1.parse import ReactionParseError, SideComponents, parse_reaction
from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps
from pes2ts_core.utils.jsonio import atomic_writer

#: Minimum anchor-map count before a Kabsch fit is attempted (plan contract).
MIN_ANCHOR_MAPS: Final[int] = 3
#: Default second-singular-value threshold (plan ``g2.assembly.anchor_tolerance``).
DEFAULT_ANCHOR_TOLERANCE: Final[float] = 1e-3
#: Placement basis: rigid fit on shared maps outside ``reaction_center.with_shell``.
BASIS_ANCHOR_MAPS: Final[str] = "anchor_maps"
#: Placement basis: rigid fit on every shared map (anchor set unusable).
BASIS_ALL_SHARED_MAPS: Final[str] = "all_shared_maps"
#: Placement basis: centroid alignment only (degenerate or too-few shared maps).
BASIS_TRANSLATION_ONLY: Final[str] = "degenerate_translation"
#: Assembly group basis: the group holding the candidate anchor keeps stored coords.
BASIS_CANDIDATE_ANCHOR_FRAME: Final[str] = "candidate_anchor_stored_frame"
#: Assembly group basis: a disconnected spectator group keeps per-group frames.
BASIS_STORED_FRAME_PER_GROUP: Final[str] = "stored_frame_per_group"
#: Separation floor (plan ``g2.assembly.forming_min_distance``).
DEFAULT_FORMING_MIN_DISTANCE: Final[float] = 2.0
#: Separation target floor (plan ``g2.assembly.forming_target_distance``).
DEFAULT_FORMING_TARGET_DISTANCE: Final[float] = 3.0
#: Bond threshold tolerance over the covalent-radius sum (``g2.assembly.bond_tolerance``).
DEFAULT_BOND_TOLERANCE: Final[float] = 0.45
#: Non-bonded collision floor (``g2.validity.collision_min_distance``).
DEFAULT_COLLISION_MIN_DISTANCE: Final[float] = 0.8
#: Extra distance added to the radius sum for separation targets.
SEPARATION_CLEARANCE: Final[float] = 0.5
#: Recorded provenance of the accepted charge/multiplicity pair.
MULTIPLICITY_BASIS: Final[str] = "g1_valid_invariant"

Coordinate = tuple[float, float, float]
MapPair = tuple[int, int]
Node = tuple[str, str]
ArrayF64 = NDArray[np.float64]


class EndpointError(Exception):
    """A typed endpoint-construction violation.

    :attr:`code` is always a :class:`RejectionCode` (endpoint-stage code:
    ``G2_ENDPOINT_MISMATCH``) and :attr:`detail` names the violated check.
    :attr:`evidence` optionally carries structured proof (e.g. the collision
    metrics and all candidate scores for ``G2_ASSEMBLY_COLLISION``).
    """

    def __init__(
        self,
        code: RejectionCode,
        detail: str,
        evidence: Mapping[str, JSONValue] | None = None,
    ) -> None:
        self.code = code
        self.detail = detail
        self.evidence = None if evidence is None else dict(evidence)
        super().__init__(f"{code.value}: {detail}")


@dataclass(frozen=True, slots=True)
class SideAtom:
    """One atom of one side, keyed by reaction-global map."""

    map: int
    element: str
    component: str
    local_index: int
    global_index: int
    coordinate: Coordinate

    def as_record(self) -> dict[str, JSONValue]:
        """Return the JSON-serializable record (sorted-key dump is stable)."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComponentGeometry:
    """Map-keyed coordinates of one component in the frame of its side."""

    tag: str
    coords: Mapping[int, Coordinate]


@dataclass(frozen=True, slots=True)
class Placement:
    """The deterministic record of one component placed onto another."""

    component: str
    onto: str
    n_shared: int
    basis: str
    rmsd_shared: float
    anchor_rank_ok: bool

    def as_record(self) -> dict[str, JSONValue]:
        """Return the JSON-serializable record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KabschResult:
    """The rigid transform mapping *moving* onto *target* (plus its RMSD)."""

    rotation: ArrayF64
    translation: ArrayF64
    rmsd: float


def build_side_atoms(
    blocks: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
) -> list[SideAtom]:
    """Translate G1 component-block rows into one map-sorted atom table.

    Each row's coordinates come from the inventory component's
    ``coordinates[local_index]``; every row element is cross-checked against
    the component ``atomic_numbers`` and every coordinate must be finite.
    Raises :class:`EndpointError` with code ``G2_ENDPOINT_MISMATCH`` on any
    violation (missing component, element mismatch, non-finite or malformed
    coordinates, duplicate maps).
    """
    by_tag = {str(record["tag"]): record for record in components}
    atoms: list[SideAtom] = []
    seen: set[int] = set()
    for block in blocks:
        tag = str(block["tag"])
        record = by_tag.get(tag)
        if record is None:
            raise EndpointError(RejectionCode.G2_ENDPOINT_MISMATCH, f"{tag}: no inventory component")
        numbers = [int(value) for value in record["atomic_numbers"]]
        coordinates = record["coordinates"]
        for row in block["rows"]:
            local = int(row["local_index"])
            map_ = int(row["map"])
            element = str(row["element"])
            if not 0 <= local < len(numbers) or len(coordinates) <= local:
                raise EndpointError(
                    RejectionCode.G2_ENDPOINT_MISMATCH,
                    f"{tag}: map {map_} local_index {local} out of range",
                )
            if not 1 <= numbers[local] <= len(ELEMENT_SYMBOLS):
                raise EndpointError(
                    RejectionCode.G2_ENDPOINT_MISMATCH,
                    f"{tag}: map {map_} atomic number {numbers[local]} unknown",
                )
            expected = ELEMENT_SYMBOLS[numbers[local] - 1]
            if element != expected:
                raise EndpointError(
                    RejectionCode.G2_ENDPOINT_MISMATCH,
                    f"{tag}: map {map_} element {element} != atomic_numbers[{local}]={expected}",
                )
            xyz = coordinates[local]
            if len(xyz) != 3 or not all(math.isfinite(float(value)) for value in xyz):
                raise EndpointError(
                    RejectionCode.G2_ENDPOINT_MISMATCH,
                    f"{tag}: map {map_} coordinates not three finite values",
                )
            if map_ in seen:
                raise EndpointError(RejectionCode.G2_ENDPOINT_MISMATCH, f"{tag}: duplicate map {map_}")
            seen.add(map_)
            atoms.append(
                SideAtom(
                    map=map_, element=element, component=tag, local_index=local,
                    global_index=int(row["global_index"]),
                    coordinate=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                )
            )
    atoms.sort(key=lambda atom: atom.map)
    return atoms


def validate_sides(
    reactant_atoms: Sequence[SideAtom],
    product_atoms: Sequence[SideAtom],
    row: Mapping[str, Any],
) -> dict[str, JSONValue]:
    """Enforce the cross-side contract and the G1-valid invariant.

    Map sets must be equal, every map must carry the same element on both
    sides, atom totals must match, and the stored totals must satisfy
    ``charge_total_reactants == charge_total_products == 0`` with
    ``multiplicity_max == 1`` (only Q=0/M=1 is accepted; no spin guessing).
    Returns the accepted ``{"charge", "multiplicity", "multiplicity_basis"}``
    record or raises :class:`EndpointError` with code ``G2_ENDPOINT_MISMATCH``.
    """
    r_by_map = {atom.map: atom for atom in reactant_atoms}
    p_by_map = {atom.map: atom for atom in product_atoms}
    if set(r_by_map) != set(p_by_map):
        raise EndpointError(
            RejectionCode.G2_ENDPOINT_MISMATCH,
            f"map_sets_differ: only_r={sorted(set(r_by_map) - set(p_by_map))} "
            f"only_p={sorted(set(p_by_map) - set(r_by_map))}",
        )
    if len(r_by_map) != len(reactant_atoms) or len(p_by_map) != len(product_atoms):
        raise EndpointError(RejectionCode.G2_ENDPOINT_MISMATCH, "duplicate maps within one side")
    if len(reactant_atoms) != len(product_atoms):
        raise EndpointError(
            RejectionCode.G2_ENDPOINT_MISMATCH,
            f"atom_count_mismatch: R={len(reactant_atoms)} P={len(product_atoms)}",
        )
    mismatched = sorted(m for m, atom in r_by_map.items() if p_by_map[m].element != atom.element)
    if mismatched:
        first = mismatched[0]
        raise EndpointError(
            RejectionCode.G2_ENDPOINT_MISMATCH,
            f"element_mismatch: map {first} R={r_by_map[first].element} P={p_by_map[first].element}",
        )
    charge_r = int(row["charge_total_reactants"])
    charge_p = int(row["charge_total_products"])
    if charge_r != charge_p:
        raise EndpointError(
            RejectionCode.G2_ENDPOINT_MISMATCH,
            f"charge_mismatch: R={charge_r} P={charge_p}",
        )
    multiplicity = int(row["multiplicity_max"])
    if charge_r != 0:
        raise EndpointError(RejectionCode.G2_ENDPOINT_MISMATCH, f"charge_not_neutral: Q={charge_r}")
    if multiplicity != 1:
        raise EndpointError(
            RejectionCode.G2_ENDPOINT_MISMATCH, f"multiplicity_not_singlet: M={multiplicity}"
        )
    return {"charge": charge_r, "multiplicity": multiplicity, "multiplicity_basis": MULTIPLICITY_BASIS}


def _side_pairs(side: SideComponents) -> frozenset[MapPair]:
    pairs: set[MapPair] = set()
    for mol, maps in zip(side.mols, side.map_lists):
        for bond in mol.GetBonds():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            first = maps[int(bond.GetBeginAtomIdx())]  # pyright: ignore[reportUnknownArgumentType]
            second = maps[int(bond.GetEndAtomIdx())]
            pairs.add((first, second) if first < second else (second, first))
    return frozenset(pairs)


def side_bond_pairs(reaction_smiles: str) -> tuple[frozenset[MapPair], frozenset[MapPair]]:
    """Return the (reactant, product) bonded map-pair sets in map space.

    Reuses :func:`pes2ts_core.g1.parse.parse_reaction`; the sets feed the
    collision check where "non-bonded" means bonded in neither R nor P.
    """
    reactants, products = parse_reaction(reaction_smiles)
    return _side_pairs(reactants), _side_pairs(products)


def anchor_maps(document: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the sorted shared maps outside ``reaction_center.with_shell``.

    These spectator maps are the preferred Kabsch basis: the reaction center
    itself is left free to relax between the sides.
    """
    def _maps(blocks: Any) -> set[int]:
        return {int(row["map"]) for block in blocks for row in block["rows"]}

    shared = _maps(document["reactants"]) & _maps(document["products"])
    shell = {int(map_) for map_ in document["reaction_center"]["with_shell"]}
    return tuple(sorted(shared - shell))


def kabsch_transform(moving: ArrayF64, target: ArrayF64) -> KabschResult:
    """Fit the deterministic rigid transform taking *moving* onto *target*.

    SVD-based Kabsch with reflection correction (the diagonal sign keeps
    ``det(rotation) = +1``); same input always yields the same output.
    """
    if moving.ndim != 2 or moving.shape[1] != 3 or moving.shape != target.shape:
        raise ValueError(f"expected matching (N, 3) arrays, got {moving.shape} vs {target.shape}")
    moving_center = moving.mean(axis=0)
    target_center = target.mean(axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = (moving - moving_center).T @ (target - target_center)
    if not bool(np.all(np.isfinite(covariance))):
        raise ValueError("kabsch input yields a non-finite covariance (scale overflow)")
    u, _, vt = np.linalg.svd(covariance)
    correction = 1.0 if float(np.linalg.det(vt.T @ u.T)) >= 0.0 else -1.0
    rotation = vt.T @ np.diag([1.0, 1.0, correction]) @ u.T
    translation = target_center - rotation @ moving_center
    applied = moving @ rotation.T + translation
    residual = applied - target
    rmsd = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return KabschResult(rotation=rotation, translation=translation, rmsd=rmsd)


def _rank_ok(points: ArrayF64, tolerance: float) -> bool:
    """True when the centered point set has rank >= 2 above *tolerance*."""
    if points.shape[0] < 2:
        return False
    centered = points - points.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    return bool(s[1] > tolerance and s[0] > tolerance)


def place_single(
    moving: ComponentGeometry,
    onto: ComponentGeometry,
    anchors: Collection[int],
    *,
    anchor_tolerance: float = DEFAULT_ANCHOR_TOLERANCE,
    min_anchor_maps: int = MIN_ANCHOR_MAPS,
) -> tuple[dict[int, Coordinate], Placement]:
    """Place *moving* onto *onto* through the basis fallback chain.

    Basis: shared anchor maps (>= ``min_anchor_maps`` and rank >= 2), else all
    shared maps (same gates), else degenerate translation (centroid alignment
    only).  Every moving coordinate is transformed; returns the placed
    coordinates and the :class:`Placement` record.  Raises ``ValueError``
    when the components share no map (a caller contract violation, never a
    data rejection).
    """
    shared = sorted(set(moving.coords) & set(onto.coords))
    if not shared:
        raise ValueError(f"no shared maps between {moving.tag!r} and {onto.tag!r}")
    moving_points = np.array([moving.coords[map_] for map_ in shared])
    onto_points = np.array([onto.coords[map_] for map_ in shared])
    anchor_shared = [map_ for map_ in shared if map_ in set(anchors)]
    anchor_rank_ok = False
    result: KabschResult | None = None
    basis = BASIS_TRANSLATION_ONLY
    if len(anchor_shared) >= min_anchor_maps:
        anchor_rank_ok = _rank_ok(np.array([moving.coords[m] for m in anchor_shared]), anchor_tolerance)
        if anchor_rank_ok:
            basis = BASIS_ANCHOR_MAPS
            result = kabsch_transform(
                np.array([moving.coords[m] for m in anchor_shared]),
                np.array([onto.coords[m] for m in anchor_shared]),
            )
    if result is None and len(shared) >= min_anchor_maps and _rank_ok(moving_points, anchor_tolerance):
        basis = BASIS_ALL_SHARED_MAPS
        result = kabsch_transform(moving_points, onto_points)
    if result is None:
        rotation = np.eye(3)
        translation = onto_points.mean(axis=0) - moving_points.mean(axis=0)
    else:
        rotation, translation = result.rotation, result.translation
    placed = {
        map_: tuple(float(value) for value in rotation @ np.array(coord) + translation)
        for map_, coord in moving.coords.items()
    }
    applied = moving_points @ rotation.T + translation
    residual = applied - onto_points
    rmsd_shared = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    placement = Placement(
        component=moving.tag, onto=onto.tag, n_shared=len(shared), basis=basis,
        rmsd_shared=rmsd_shared, anchor_rank_ok=anchor_rank_ok,
    )
    return placed, placement


def endpoints_record(
    validation: Mapping[str, JSONValue],
    reactant_atoms: Sequence[SideAtom],
    product_atoms: Sequence[SideAtom],
    placements: Sequence[Placement] = (),
) -> dict[str, JSONValue]:
    """Assemble the deterministic, JSON-serializable endpoints record.

    Plain data only (maps pre-sorted, standard float repr), so
    ``json.dumps(record, sort_keys=True)`` is byte-identical across runs on
    the same input.
    """
    return {
        "validation": dict(validation),
        "reactant_atoms": [atom.as_record() for atom in reactant_atoms],
        "product_atoms": [atom.as_record() for atom in product_atoms],
        "placements": [placement.as_record() for placement in placements],
    }


def _covalent_radius(element: str) -> float:
    """Return the Cordero radius for *element* via the element→Z table."""
    z = ELEMENT_SYMBOLS.index(element) + 1
    try:
        return COVALENT_RADII[z]
    except KeyError:
        raise EndpointError(
            RejectionCode.G2_ASSEMBLY_FAILED, f"no covalent radius for element {element}"
        ) from None


def separate_changed_pairs(
    pairs: Collection[MapPair],
    *,
    side: str,
    kind: str,
    coords: Mapping[int, Coordinate],
    component_of_map: Mapping[int, str],
    group_seed_of_component: Mapping[str, str],
    elements: Mapping[int, str],
    forming_min_distance: float = DEFAULT_FORMING_MIN_DISTANCE,
    forming_target_distance: float = DEFAULT_FORMING_TARGET_DISTANCE,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
) -> tuple[dict[int, Coordinate], list[dict[str, JSONValue]]]:
    """Separate changed key pairs lying in different placed components.

    Side-aware and direction-independent: the R assembly evaluates ``formed``
    pairs and the P assembly evaluates ``broken`` pairs, through the same
    code path.  A pair triggers when its atoms sit in DIFFERENT components
    and the placed distance is below ``max(forming_min_distance, radius_sum +
    bond_tolerance)``; the non-seed component then translates along the pair
    axis to ``max(forming_target_distance, radius_sum + bond_tolerance +
    SEPARATION_CLEARANCE)``.  When neither component is its group's seed, the
    component holding the larger map moves.  Returns the updated coordinates
    (new dict, input untouched) and one deterministic record per evaluated
    pair in ascending map order.
    """
    moved_coords = {map_: coord for map_, coord in coords.items()}
    records: list[dict[str, JSONValue]] = []
    for first, second in sorted(tuple(pair) for pair in pairs):
        radius_sum = _covalent_radius(elements[first]) + _covalent_radius(elements[second])
        threshold = max(forming_min_distance, radius_sum + bond_tolerance)
        distance = _dist_coords(moved_coords[first], moved_coords[second])
        component_first = component_of_map[first]
        component_second = component_of_map[second]
        triggered = component_first != component_second and distance < threshold
        if not triggered:
            records.append(
                {
                    "side": side, "kind": kind, "atoms": [first, second],
                    "separation_evaluated": True, "triggered": False,
                    "threshold": threshold, "d_placed": distance,
                }
            )
            continue
        target = max(forming_target_distance, radius_sum + bond_tolerance + SEPARATION_CLEARANCE)
        seed_first = group_seed_of_component.get(component_first) == component_first
        seed_second = group_seed_of_component.get(component_second) == component_second
        if seed_first and not seed_second:
            moved_component = component_second
        elif seed_second and not seed_first:
            moved_component = component_first
        else:
            moved_component = component_second if second > first else component_first
        fixed_map, moved_map = (first, second) if moved_component == component_second else (second, first)
        unit = _unit_vector(moved_coords[moved_map], moved_coords[fixed_map], distance)
        shift = target - distance
        for map_, component in component_of_map.items():
            if component == moved_component:
                moved_coords[map_] = _shift_coord(moved_coords[map_], unit, shift)
        records.append(
            {
                "side": side, "kind": kind, "atoms": [first, second],
                "separation_evaluated": True, "triggered": True,
                "threshold": threshold, "target": target,
                "d_before": distance, "d_after": _dist_coords(
                    moved_coords[first], moved_coords[second]
                ),
            }
        )
    return moved_coords, records


def _dist_coords(first: Coordinate, second: Coordinate) -> float:
    return math.dist(first, second)


def _unit_vector(moved: Coordinate, fixed: Coordinate, distance: float) -> Coordinate:
    return tuple((moved[i] - fixed[i]) / distance for i in range(3))


def _shift_coord(coord: Coordinate, unit: Coordinate, shift: float) -> Coordinate:
    return tuple(coord[i] + unit[i] * shift for i in range(3))


@dataclass(frozen=True, slots=True)
class EndpointAssembly:
    """The deterministic result of one endpoint assembly."""

    reaction_id: str
    reactant_coords: dict[int, Coordinate]
    product_coords: dict[int, Coordinate]
    elements: dict[int, str]
    record: dict[str, JSONValue] = field(default_factory=dict)
    winner_index: int = 0


def _config_block(config: Mapping[str, Any] | None, section: str) -> Mapping[str, Any]:
    blocks = (config or {}).get("g2", {})
    return blocks.get(section, {}) if isinstance(blocks, Mapping) else {}


def _connected_groups(
    nodes: Sequence[Node], shared_counts: Mapping[tuple[Node, Node], int]
) -> list[set[Node]]:
    parent: dict[Node, Node] = {node: node for node in nodes}

    def find(node: Node) -> Node:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (first, second), count in shared_counts.items():
        if count > 0:
            parent[find(first)] = find(second)
    grouped: dict[Node, set[Node]] = {}
    for node in nodes:
        grouped.setdefault(find(node), set()).add(node)
    return list(grouped.values())


def _node_sort_key(node: Node) -> tuple[int, str]:
    return (0 if node[0] == "R" else 1, node[1])


def _bfs_place(
    anchor: Node,
    maps_by_node: Mapping[Node, set[int]],
    stored_by_node: Mapping[Node, Mapping[int, Coordinate]],
    shared_counts: Mapping[tuple[Node, Node], int],
    anchors: tuple[int, ...],
    anchor_tolerance: float,
    min_anchor_maps: int,
) -> tuple[dict[Node, dict[int, Coordinate]], list[Placement]]:
    """Place every node reachable from *anchor* into the anchor's frame."""
    placed: dict[Node, dict[int, Coordinate]] = {anchor: dict(stored_by_node[anchor])}
    placements: list[Placement] = []
    while True:
        best: tuple[tuple[int, int, str], Node, Node, int] | None = None
        for node in sorted(set(maps_by_node) - set(placed), key=_node_sort_key):
            edges = [
                (shared_counts[(node, target)], target)
                for target in placed
                if shared_counts.get((node, target), 0) > 0
            ]
            if not edges:
                continue
            shared_count, target = max(edges, key=lambda item: (item[0], -_node_sort_key(item[1])[0], item[1][1]))
            key = (-shared_count, _node_sort_key(node)[0], node[1])
            if best is None or key < best[0]:
                best = (key, node, target, shared_count)
        if best is None:
            return placed, placements
        _, node, target, shared_count = best
        moving = ComponentGeometry(
            f"{node[0]}:{node[1]}",
            {map_: stored_by_node[node][map_] for map_ in sorted(maps_by_node[node])},
        )
        onto = ComponentGeometry(
            f"{target[0]}:{target[1]}",
            {map_: placed[target][map_] for map_ in sorted(maps_by_node[target])},
        )
        placed_coords, placement = place_single(
            moving, onto, anchors,
            anchor_tolerance=anchor_tolerance, min_anchor_maps=min_anchor_maps,
        )
        placement = Placement(
            component=f"{node[0]}:{node[1]}", onto=f"{target[0]}:{target[1]}",
            n_shared=shared_count, basis=placement.basis,
            rmsd_shared=placement.rmsd_shared, anchor_rank_ok=placement.anchor_rank_ok,
        )
        placements.append(placement)
        placed[node] = placed_coords


def _check_finite(coords: Mapping[int, Coordinate], label: str) -> None:
    for map_, coord in coords.items():
        if not all(math.isfinite(value) for value in coord):
            raise EndpointError(
                RejectionCode.G2_ASSEMBLY_FAILED,
                f"{label}: non-finite placed coordinate for map {map_}",
            )


def _nonbonded_pairs(maps: Sequence[int], bonded: frozenset[MapPair]) -> list[MapPair]:
    return [pair for pair in combinations(sorted(maps), 2) if pair not in bonded]


def _assemble_endpoints(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    config: Mapping[str, Any] | None,
) -> EndpointAssembly:
    assembly_cfg = _config_block(config, "assembly")
    validity_cfg = _config_block(config, "validity")
    min_anchor_maps = int(assembly_cfg.get("min_anchor_maps", MIN_ANCHOR_MAPS))
    anchor_tolerance = float(assembly_cfg.get("anchor_tolerance", DEFAULT_ANCHOR_TOLERANCE))
    forming_min = float(assembly_cfg.get("forming_min_distance", DEFAULT_FORMING_MIN_DISTANCE))
    forming_target = float(assembly_cfg.get("forming_target_distance", DEFAULT_FORMING_TARGET_DISTANCE))
    bond_tolerance = float(assembly_cfg.get("bond_tolerance", DEFAULT_BOND_TOLERANCE))
    collision_min = float(validity_cfg.get("collision_min_distance", DEFAULT_COLLISION_MIN_DISTANCE))

    atoms_r = build_side_atoms(document["reactants"], row["components"])
    atoms_p = build_side_atoms(document["products"], row["components"])
    validation = validate_sides(atoms_r, atoms_p, row)
    r_pairs, p_pairs = side_bond_pairs(str(document["reaction_smiles"]))
    formed = sorted(p_pairs - r_pairs)
    broken = sorted(r_pairs - p_pairs)

    r_tags = sorted({atom.component for atom in atoms_r})
    p_tags = sorted({atom.component for atom in atoms_p})
    maps_by_node: dict[Node, set[int]] = {}
    for atom in atoms_r:
        maps_by_node.setdefault(("R", atom.component), set()).add(atom.map)
    for atom in atoms_p:
        maps_by_node.setdefault(("P", atom.component), set()).add(atom.map)
    stored_by_node: dict[Node, dict[int, Coordinate]] = {}
    for atom in atoms_r:
        stored_by_node.setdefault(("R", atom.component), {})[atom.map] = atom.coordinate
    for atom in atoms_p:
        stored_by_node.setdefault(("P", atom.component), {})[atom.map] = atom.coordinate
    shared_counts: dict[tuple[Node, Node], int] = {}
    for r_node in maps_by_node:
        if r_node[0] != "R":
            continue
        for p_node in maps_by_node:
            if p_node[0] != "P":
                continue
            count = len(maps_by_node[r_node] & maps_by_node[p_node])
            if count > 0:
                shared_counts[(r_node, p_node)] = count
                shared_counts[(p_node, r_node)] = count

    sizes: dict[Node, int] = {node: len(maps) for node, maps in maps_by_node.items()}
    r_ranked = sorted(r_tags, key=lambda tag: (-sizes[("R", tag)], tag))
    candidates: list[Node] = [("R", r_ranked[0])]
    if len(r_ranked) >= 2:
        candidates.append(("R", r_ranked[1]))
    if len(p_tags) >= 2:
        p_ranked = sorted(p_tags, key=lambda tag: (-sizes[("P", tag)], tag))
        candidates.append(("P", p_ranked[0]))
    anchors = anchor_maps(document)

    node_list = sorted(maps_by_node, key=_node_sort_key)
    groups = sorted(_connected_groups(node_list, shared_counts), key=lambda group: sorted(map(_node_sort_key, group)))
    maps = sorted({atom.map for atom in atoms_r})
    elements = {atom.map: atom.element for atom in atoms_r}
    all_maps = {atom.map: atom for atom in atoms_r}
    candidate_records: list[dict[str, JSONValue]] = []
    scored: list[tuple[tuple[int, float, float, int], dict[str, JSONValue], dict[str, JSONValue], dict[Node, dict[int, Coordinate]]]] = []
    for index, anchor in enumerate(candidates):
        placed, placements = _bfs_place(
            anchor, maps_by_node, stored_by_node, shared_counts, anchors,
            anchor_tolerance, min_anchor_maps,
        )
        ordered_groups = [next(group for group in groups if anchor in group)]
        ordered_groups += [group for group in groups if anchor not in group]
        group_records: list[dict[str, JSONValue]] = []
        seed_of_component: dict[str, str] = {}
        side_coords: dict[str, dict[int, Coordinate]] = {"R": {}, "P": {}}
        component_of_map_sides: dict[str, dict[int, str]] = {"R": {}, "P": {}}
        for group_id, group in enumerate(ordered_groups):
            is_anchored = anchor in group
            nodes_sorted = sorted(group, key=_node_sort_key)
            seed_node = anchor if is_anchored else nodes_sorted[0]
            group_records.append(
                {
                    "group_id": f"G{group_id}",
                    "anchor_component": f"{seed_node[0]}:{seed_node[1]}",
                    "assembly_basis": (
                        BASIS_CANDIDATE_ANCHOR_FRAME if is_anchored else BASIS_STORED_FRAME_PER_GROUP
                    ),
                    "global_frame_basis": (
                        BASIS_CANDIDATE_ANCHOR_FRAME if is_anchored else BASIS_STORED_FRAME_PER_GROUP
                    ),
                    "frame_ambiguous": not is_anchored,
                    "nodes": [[side, tag] for side, tag in nodes_sorted],
                }
            )
            for node in nodes_sorted:
                key = f"{node[0]}:{node[1]}"
                seed_of_component[key] = f"{seed_node[0]}:{seed_node[1]}"
                for map_ in sorted(maps_by_node[node]):
                    if is_anchored:
                        coord = placed[node][map_]
                    else:
                        coord = stored_by_node[("R", _r_tag_of(map_, atoms_r))][map_]
                    side_coords[node[0]][map_] = coord
                    component_of_map_sides[node[0]][map_] = key
        for side_key in ("R", "P"):
            _check_finite(side_coords[side_key], f"candidate {index} {side_key} assembly")
        separations: list[dict[str, JSONValue]] = []
        r_coords, r_records = separate_changed_pairs(
            formed, side="R", kind="formed", coords=side_coords["R"],
            component_of_map=component_of_map_sides["R"],
            group_seed_of_component=seed_of_component, elements=elements,
            forming_min_distance=forming_min, forming_target_distance=forming_target,
            bond_tolerance=bond_tolerance,
        )
        p_coords, p_records = separate_changed_pairs(
            broken, side="P", kind="broken", coords=side_coords["P"],
            component_of_map=component_of_map_sides["P"],
            group_seed_of_component=seed_of_component, elements=elements,
            forming_min_distance=forming_min, forming_target_distance=forming_target,
            bond_tolerance=bond_tolerance,
        )
        separations = r_records + p_records

        nonbonded = _nonbonded_pairs(maps, r_pairs | p_pairs)
        severe_pairs: list[dict[str, JSONValue]] = []
        min_distance: float | None = None
        for side_key, coords in (("R", r_coords), ("P", p_coords)):
            for pair in nonbonded:
                distance = _dist_coords(coords[pair[0]], coords[pair[1]])
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                if distance < collision_min:
                    severe_pairs.append(
                        {"side": side_key, "atoms": list(pair), "distance": distance}
                    )
        formed_excess = 0.0
        for first, second in formed:
            radius_sum = _covalent_radius(elements[first]) + _covalent_radius(elements[second])
            formed_excess += max(0.0, (radius_sum + bond_tolerance) - _dist_coords(r_coords[first], r_coords[second]))
        metrics = {
            "n_severe_contacts": len(severe_pairs),
            "min_nonbonded_distance": min_distance,
            "formed_excess": formed_excess,
            "collision_min_distance": collision_min,
            "severe_pairs": sorted(severe_pairs, key=lambda item: (item["side"], item["atoms"])),
        }
        score = (
            metrics["n_severe_contacts"],
            0.0 if min_distance is None else -round(min_distance, 6),
            formed_excess,
            index,
        )
        frame_ambiguous = any(group["frame_ambiguous"] for group in group_records)
        candidate_record = {
            "candidate_index": index,
            "anchor_component": f"{anchor[0]}:{anchor[1]}",
            "frame_ambiguous": frame_ambiguous,
            "metrics": metrics,
            "score": list(score),
        }
        candidate_records.append(candidate_record)
        scored.append((score, candidate_record, {"groups": group_records, "placements": [p.as_record() for p in placements], "separations": separations, "metrics": metrics}, {"R": r_coords, "P": p_coords}))

    scored.sort(key=lambda item: item[0])
    winner_score, winner_candidate, winner_parts, winner_coords = scored[0]
    if winner_score[0] > 0:
        raise EndpointError(
            RejectionCode.G2_ASSEMBLY_COLLISION,
            f"non-bonded pair below {collision_min} A in the chosen assembly",
            evidence={"metrics": winner_candidate["metrics"], "candidates": candidate_records},
        )
    record: dict[str, JSONValue] = {
        "reaction_id": str(document["reaction_id"]),
        "multiplicity_basis": validation["multiplicity_basis"],
        "groups": winner_parts["groups"],
        "placements": winner_parts["placements"],
        "separations": winner_parts["separations"],
        "metrics": winner_parts["metrics"],
        "candidates": candidate_records,
        "reactant_atoms": [atom.as_record() for atom in atoms_r],
        "product_atoms": [atom.as_record() for atom in atoms_p],
    }
    return EndpointAssembly(
        reaction_id=str(document["reaction_id"]),
        reactant_coords={map_: winner_coords["R"][map_] for map_ in maps},
        product_coords={map_: winner_coords["P"][map_] for map_ in maps},
        elements=elements,
        record=record,
        winner_index=winner_candidate["candidate_index"],
    )


def _r_tag_of(map_: int, atoms: Sequence[SideAtom]) -> str:
    for atom in atoms:
        if atom.map == map_:
            return atom.component
    raise EndpointError(RejectionCode.G2_ASSEMBLY_FAILED, f"map {map_} missing on the reactant side")


def assemble_endpoints(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> EndpointAssembly:
    """Assemble the deterministic multi-component endpoints for one reaction.

    Builds the candidate anchor list (largest R component, second-largest R,
    largest P when a side has >= 2 components), runs the cross-side bipartite
    BFS rigid placement for every candidate, separates changed key pairs per
    side, scores every candidate by
    ``(n_severe_contacts, -min_nonbonded, formed_excess, index)`` and selects
    the lexicographic minimum.  A chosen assembly that still holds a
    non-bonded pair below ``collision_min_distance`` raises
    ``G2_ASSEMBLY_COLLISION`` with the full metrics and candidate scores as
    evidence; structural malformation raises ``G2_ASSEMBLY_FAILED``.
    ``G2_ENDPOINT_MISMATCH`` from the table/validation layer propagates
    untouched.
    """
    try:
        return _assemble_endpoints(document, row, config)
    except EndpointError:
        raise
    except ReactionParseError as error:
        raise EndpointError(
            RejectionCode.G2_ASSEMBLY_FAILED, f"unparseable reaction smiles: {error}"
        ) from None
    except np.linalg.LinAlgError as error:
        raise EndpointError(
            RejectionCode.G2_ASSEMBLY_FAILED, f"degenerate placement transform: {error}"
        ) from None
    except ValueError as error:
        raise EndpointError(
            RejectionCode.G2_ASSEMBLY_FAILED, f"degenerate placement transform: {error}"
        ) from None
    except KeyError as error:
        raise EndpointError(
            RejectionCode.G2_ASSEMBLY_FAILED, f"malformed document: missing key {error}"
        ) from None


def _xyz_bytes(reaction_id: str, suffix: str, coords: Mapping[int, Coordinate], elements: Mapping[int, str]) -> bytes:
    lines = [str(len(coords)), f"{reaction_id} {suffix}"]
    lines += [
        f"{elements[map_]} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}" for map_, coord in sorted(coords.items())
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_endpoint_files(directory: str | Path, assembly: EndpointAssembly) -> tuple[Path, ...]:
    """Write ``endpoints.json`` plus ``R.xyz``/``P.xyz`` atomically.

    JSON goes through :func:`stable_json_dumps` (sorted keys, compact
    separators) inside :func:`atomic_writer`; XYZ files carry the atoms in
    ascending map order with ``%.6f`` coordinates and a
    ``"<reaction_id> <side>"`` comment line.  Identical input yields
    byte-identical output.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "endpoints.json"
    with atomic_writer(json_path) as handle:
        handle.write((stable_json_dumps(assembly.record) + "\n").encode("utf-8"))
    r_path = target / "R.xyz"
    with atomic_writer(r_path) as handle:
        handle.write(_xyz_bytes(assembly.reaction_id, "R", assembly.reactant_coords, assembly.elements))
    p_path = target / "P.xyz"
    with atomic_writer(p_path) as handle:
        handle.write(_xyz_bytes(assembly.reaction_id, "P", assembly.product_coords, assembly.elements))
    return json_path, r_path, p_path


__all__ = [
    "BASIS_ALL_SHARED_MAPS", "BASIS_ANCHOR_MAPS", "BASIS_CANDIDATE_ANCHOR_FRAME",
    "BASIS_STORED_FRAME_PER_GROUP", "BASIS_TRANSLATION_ONLY", "DEFAULT_ANCHOR_TOLERANCE",
    "DEFAULT_BOND_TOLERANCE", "DEFAULT_COLLISION_MIN_DISTANCE", "DEFAULT_FORMING_MIN_DISTANCE",
    "DEFAULT_FORMING_TARGET_DISTANCE", "ComponentGeometry", "EndpointAssembly", "EndpointError",
    "KabschResult", "MIN_ANCHOR_MAPS", "MULTIPLICITY_BASIS", "Placement", "SideAtom",
    "anchor_maps", "assemble_endpoints", "build_side_atoms", "endpoints_record",
    "kabsch_transform", "place_single", "separate_changed_pairs", "side_bond_pairs",
    "validate_sides", "write_endpoint_files",
]
