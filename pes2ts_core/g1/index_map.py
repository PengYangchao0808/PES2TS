"""Map-to-local-to-global index tables for one matched reaction side.

Coordinates are stored per inventory component in XYZ-row order, while the
component SMILES atom order is independent: the atom carrying local map ``k``
occupies row ``k - 1`` (verified on every stored component; reaction SMILES map
numbers are a separate reaction-global space).  A :class:`ComponentIndex`
therefore exposes, per component, the row list
``{local_index, map, element, global_index}`` where ``local_index`` is the
component-local XYZ row, ``map`` its map label (``local_index + 1`` under the
stored invariant), and ``global_index`` the row's position once the side's
components are concatenated in tag order.

``rows`` stores the first isomorphism enumerated by the matcher (RDKit's
enumeration order is deterministic -- same input, same output) and the full
ambiguity state is preserved: every component records its solution count, its
status, and the first ``max_candidates`` complete ``(map, local_index)``
candidate bijections in query order (tuple position = query atom index, which
the caller composes with the side's per-component map list to recover
reaction-global map numbers).

The element cross-check (the atom carrying map ``k`` must match
``atomic_numbers[k - 1]``) and the bond-geometry check (every stored SMILES
bond's XYZ distance within the Cordero covalent-radius sum plus tolerance) gate
a component before any index table may enter QC.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from rdkit import Chem, RDLogger

from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS

if TYPE_CHECKING:  # pragma: no cover - annotation-only import (no runtime cycle)
    from pes2ts_core.g1.skeleton import ComponentMatch

# RDKit parse failures are surfaced as typed results instead of stderr noise,
# so the C++ logger is muted at import time. (rdkit-stubs omits ``DisableLog``,
# hence the targeted ignore.)
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

#: Cordero-style single-bond covalent radii in Angstrom, indexed by Z = 1..86
#: (Cordero et al., Dalton Trans. 2008, 2832-2838; low-spin values for the
#: first-row transition metals).
COVALENT_RADII: Final[dict[int, float]] = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76,
    21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22,
    31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75,
    41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01,
    61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96, 65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87,
    71: 1.87, 72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32,
    81: 1.45, 82: 1.46, 83: 1.48, 84: 1.40, 85: 1.50, 86: 1.50,
}


@dataclass(frozen=True, slots=True)
class ComponentIndex:
    """The index table and ambiguity record of one matched component.

    ``index_base`` is the component's local-to-global offset: components are
    laid out in tag order and the offset accumulates the preceding components'
    atom counts, so ``global_index = index_base + local_index`` addresses the
    side's concatenated coordinate arrays.  ``status`` is ``"truncated"`` when
    the matcher hit ``match_cap``, ``"symmetric_ambiguous"`` for more than one
    isomorphism, and ``"unique"`` otherwise.  ``element_check`` is false when
    the parsed component SMILES does not carry each map ``k`` on an atom
    matching ``atomic_numbers[k - 1]``; such a component must be rejected
    before its table can enter QC.
    """

    tag: str
    index_base: int
    n_atoms: int
    rows: tuple[dict[str, int | str], ...]
    status: str
    n_candidates: int
    candidates: tuple[tuple[tuple[int, int], ...], ...]
    element_check: bool


def _parse_component(smiles: str) -> Chem.Mol | None:
    """Parse one component SMILES with explicit hydrogens kept."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    return Chem.MolFromSmiles(smiles, params)  # pyright: ignore[reportUnknownMemberType]


def _tag_sort_key(tag: str) -> tuple[str, int, str]:
    """Order tags R0, R1, ..., R10 numerically while keeping prefixes grouped."""
    prefix = tag.rstrip("0123456789")
    suffix = tag[len(prefix):]
    return (prefix, int(suffix) if suffix else -1, tag)


def build_side_indexes(
    matches: Sequence[ComponentMatch],
    inventory_components: Sequence[Mapping[str, Any]],
    *,
    max_candidates: int,
) -> list[ComponentIndex]:
    """Build the per-component index tables of one matched side.

    Components are laid out in tag order (each side owns its own coordinate
    system) and ``index_base`` accumulates the preceding components' atom
    counts.  ``rows`` uses the first isomorphism: under the stored invariant
    the atom at local XYZ row ``i`` carries map ``i + 1`` and element
    ``atomic_numbers[i]``.

    ``candidates`` records the stored isomorphisms as complete
    ``(map, local_index)`` bijections in query order; the tuple position is the
    query atom index, the caller's bridge to the reaction-global map numbers
    (``map_of_row == row + 1`` under the verified invariant).

    Raises
    ------
    ValueError
        When a matched tag is missing from *inventory_components*, its SMILES
        does not parse, or ``max_candidates`` is not positive -- caller
        contract violations, never data rejections.
    """
    if max_candidates < 1:
        raise ValueError(f"max_candidates must be positive, got {max_candidates}")
    by_tag = {str(record["tag"]): record for record in inventory_components}
    indexes: list[ComponentIndex] = []
    index_base = 0
    for match in sorted(matches, key=lambda item: _tag_sort_key(item.tag)):
        record = by_tag.get(match.tag)
        if record is None:
            raise ValueError(
                f"no inventory component record for matched tag {match.tag!r}"
            )
        numbers = [int(number) for number in record["atomic_numbers"]]
        mol = _parse_component(str(record["smiles"]))
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"inventory component {match.tag!r}: SMILES did not parse")
        maps = [
            int(atom.GetAtomMapNum())  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            for atom in mol.GetAtoms()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        ]
        n_atoms = len(numbers)
        element_check = len(maps) == n_atoms and sorted(maps) == list(
            range(1, n_atoms + 1)
        )
        if element_check:
            element_check = all(
                mol.GetAtomWithIdx(i).GetSymbol()
                == ELEMENT_SYMBOLS[numbers[maps[i] - 1] - 1]
                for i in range(n_atoms)
            )
        rows: tuple[dict[str, int | str], ...] = tuple(
            {
                "local_index": i,
                "map": i + 1,
                "element": ELEMENT_SYMBOLS[numbers[i] - 1],
                "global_index": index_base + i,
            }
            for i in range(n_atoms)
        )
        candidates = tuple(
            tuple((local + 1, local) for local in iso)
            for iso in match.isomorphisms[:max_candidates]
        )
        if match.truncated:
            status = "truncated"
        elif match.n_isomorphisms > 1:
            status = "symmetric_ambiguous"
        else:
            status = "unique"
        indexes.append(
            ComponentIndex(
                tag=match.tag,
                index_base=index_base,
                n_atoms=n_atoms,
                rows=rows,
                status=status,
                n_candidates=len(candidates),
                candidates=candidates,
                element_check=element_check,
            )
        )
        index_base += n_atoms
    return indexes


def bond_geometry_check(
    component: Mapping[str, Any], *, tolerance: float
) -> tuple[bool, str]:
    """Verify every stored SMILES bond against the component's coordinates.

    Each bond connects the atoms carrying maps ``(k1, k2)``; the Euclidean
    distance between coordinate rows ``k1 - 1`` and ``k2 - 1`` must not exceed
    ``COVALENT_RADII[z1] + COVALENT_RADII[z2] + tolerance``.  Returns
    ``(True, "")`` when every bond passes, otherwise ``(False, detail)`` where
    *detail* names the worst offending pair (largest excess over the limit).
    A component whose SMILES does not parse, or whose bond map numbers fall
    outside the component, fails with a describing detail.
    """
    numbers = [int(number) for number in component["atomic_numbers"]]
    coordinates = [
        [float(value) for value in row] for row in component["coordinates"]
    ]
    mol = _parse_component(str(component["smiles"]))
    if mol is None or mol.GetNumAtoms() == 0:
        return False, "component SMILES did not parse"
    worst_excess = 0.0
    worst_detail = ""
    for bond in mol.GetBonds():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        first = int(bond.GetBeginAtom().GetAtomMapNum())  # pyright: ignore[reportUnknownMemberType]
        second = int(bond.GetEndAtom().GetAtomMapNum())  # pyright: ignore[reportUnknownMemberType]
        if not (1 <= first <= len(numbers) and 1 <= second <= len(numbers)):
            return (
                False,
                f"bond maps ({first}, {second}) outside the {len(numbers)}-atom component",
            )
        radius_sum = (
            COVALENT_RADII[numbers[first - 1]] + COVALENT_RADII[numbers[second - 1]]
        )
        distance = math.dist(coordinates[first - 1], coordinates[second - 1])
        excess = distance - radius_sum - tolerance
        if excess > worst_excess:
            worst_excess = excess
            worst_detail = (
                f"bond maps ({first}, {second}) distance {distance:.3f} > "
                f"radius sum {radius_sum:.3f} + tolerance {tolerance:.3f}"
            )
    if worst_detail:
        return False, worst_detail
    return True, ""


__all__ = [
    "COVALENT_RADII",
    "ComponentIndex",
    "bond_geometry_check",
    "build_side_indexes",
]
