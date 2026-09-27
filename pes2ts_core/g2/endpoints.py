"""G2 endpoint construction, part A: map-ordered atom tables and rigid alignment.

This module turns a G1 reaction-change document plus its inventory row into
map-ordered per-side atom tables, validates the cross-side contract, derives
map-space bond-pair sets, and provides the deterministic 1-to-1 placement
primitive used by the multi-component BFS assembly (part B, same file).
Nothing here reads truth data or spawns processes.
"""

# allow: SIZE_OK -- the plan contract fixes every endpoint-construction
# primitive (tables, validation, bond pairs, anchors, Kabsch, placement) in
# this single module; task 3 extends the same file.

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.parse import SideComponents, parse_reaction
from pes2ts_core.utils.hashing import JSONValue

#: Minimum anchor-map count before a Kabsch fit is attempted (plan contract).
MIN_ANCHOR_MAPS: Final[int] = 3
#: Default second-singular-value threshold (plan ``g2.assembly.anchor_tolerance``).
DEFAULT_ANCHOR_TOLERANCE: Final[float] = 1e-3
#: Placement basis: rigid fit on shared maps outside ``reaction_center.with_shell``.
BASIS_ANCHOR_MAPS: Final[str] = "anchor_maps"
#: Placement basis: rigid fit on every shared map (anchor set unusable).
BASIS_ALL_SHARED_MAPS: Final[str] = "all_shared_maps"
#: Placement basis: centroid alignment only (degenerate or too-few shared maps).
BASIS_TRANSLATION_ONLY: Final[str] = "translation_only"
#: Recorded provenance of the accepted charge/multiplicity pair.
MULTIPLICITY_BASIS: Final[str] = "g1_valid_invariant"

Coordinate = tuple[float, float, float]
MapPair = tuple[int, int]
ArrayF64 = NDArray[np.float64]


class EndpointError(Exception):
    """A typed endpoint-construction violation.

    :attr:`code` is always a :class:`RejectionCode` (endpoint-stage code:
    ``G2_ENDPOINT_MISMATCH``) and :attr:`detail` names the violated check.
    """

    def __init__(self, code: RejectionCode, detail: str) -> None:
        self.code = code
        self.detail = detail
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
    covariance = (moving - moving_center).T @ (target - target_center)
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
) -> tuple[dict[int, Coordinate], Placement]:
    """Place *moving* onto *onto* through the basis fallback chain.

    Basis: shared anchor maps (>= ``MIN_ANCHOR_MAPS`` and rank >= 2), else all
    shared maps (same gates), else translation-only centroid alignment.  Every
    moving coordinate is transformed; returns the placed coordinates and the
    :class:`Placement` record.  Raises ``ValueError`` when the components
    share no map (a caller contract violation, never a data rejection).
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
    if len(anchor_shared) >= MIN_ANCHOR_MAPS:
        anchor_rank_ok = _rank_ok(np.array([moving.coords[m] for m in anchor_shared]), anchor_tolerance)
        if anchor_rank_ok:
            basis = BASIS_ANCHOR_MAPS
            result = kabsch_transform(
                np.array([moving.coords[m] for m in anchor_shared]),
                np.array([onto.coords[m] for m in anchor_shared]),
            )
    if result is None and len(shared) >= MIN_ANCHOR_MAPS and _rank_ok(moving_points, anchor_tolerance):
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


__all__ = [
    "BASIS_ALL_SHARED_MAPS", "BASIS_ANCHOR_MAPS", "BASIS_TRANSLATION_ONLY",
    "DEFAULT_ANCHOR_TOLERANCE", "ComponentGeometry", "EndpointError", "KabschResult",
    "MIN_ANCHOR_MAPS", "MULTIPLICITY_BASIS", "Placement", "SideAtom", "anchor_maps",
    "build_side_atoms", "endpoints_record", "kabsch_transform", "place_single",
    "side_bond_pairs", "validate_sides",
]
