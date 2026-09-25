"""Map- and direction-invariant reaction identity for the G0 stage.

Two reaction records are the same chemistry when they differ only in atom-map
numbering, in the order of the dot-separated components within a side, or in
which side was written as reactants. This module reduces a reaction SMILES to a
single SHA-256 identity that is invariant under all three transformations (but
still distinguishes stereoisomers), which downstream stages use to group exact
duplicates and reverse reactions.

Canonicalization
----------------
Each dot-separated component of both sides is parsed with RDKit and rewritten
as an isomeric canonical SMILES with the atom-map annotations removed. The
preferred path passes ``ignoreAtomMapNumbers=True`` to
:func:`rdkit.Chem.MolToSmiles` when the installed RDKit accepts that kwarg
(probed once at import; see :data:`CANONICALIZATION_PATH` and
:data:`IGNORE_ATOM_MAP_KWARG`). The kwarg only excludes atom maps from the
canonical *ranking* though -- rdkit 2026.3.6 still emits the map annotations --
so the internal molecule has its map numbers zeroed via
:meth:`rdkit.Chem.Atom.SetAtomMapNum` on **both** paths. ``ClearProp`` is never
used because it cannot remove ``molAtomMapNumber``.

Preimage format (frozen; changing it changes every identity)
------------------------------------------------------------
::

    <side_a>>><side_b>

where each side is the ``"+"``-joined, lexicographically sorted list of its
canonical component SMILES and ``(side_a, side_b)`` is the lexicographically
sorted pair of the two side strings. The identity is the lower-case SHA-256 hex
digest of the UTF-8 encoding of that text. The preimage contains no timestamps,
paths, or other run-dependent values.

Directional comparisons
-----------------------
The identity sorts the two sides, so a reaction and its written reverse share
one hash. :func:`directional_preimage` exposes the same canonicalization
*without* sorting the two sides, so callers can tell a same-direction exact
duplicate from a written reverse inside a single identity group. Both public
functions delegate to the same side-canonicalization code path, so they can
never disagree about what the canonical sides are.

All functions are pure: caller-provided strings are never mutated and no state
is cached across calls, so identities stay cheap to memoize in the caller.
"""

from __future__ import annotations

import logging
from typing import Final

from rdkit import Chem, RDLogger

from pes2ts_core.utils.hashing import sha256_bytes

logger = logging.getLogger(__name__)

#: RDKit reports parse failures on stderr by default; they surface as typed
#: :class:`IdentityError` exceptions instead, so the C++ logger is muted at
#: import time. (rdkit-stubs omits ``DisableLog``, hence the targeted ignore.)
RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

#: Separator between the reactant and product sides of a reaction SMILES.
ARROW: Final[str] = ">>"

#: Separator between components within one side of a reaction SMILES.
COMPONENT_SEPARATOR: Final[str] = "."

#: Joins the canonical component SMILES of one side inside the preimage.
COMPONENT_JOIN: Final[str] = "+"

#: Path label recorded when RDKit accepts the ``ignoreAtomMapNumbers`` kwarg.
KWARG_PATH: Final[str] = "ignore_atom_map_numbers"

#: Path label recorded when only explicit map stripping is available.
STRIP_PATH: Final[str] = "strip_atom_map_numbers"

#: Small mapped molecule used to probe kwarg support at import time.
_PROBE_SMILES: Final[str] = "[CH3:1]O"


class IdentityError(Exception):
    """Raised when a reaction SMILES cannot be canonicalized."""


def _probe_ignore_atom_map_kwarg() -> bool:
    """Return whether :func:`rdkit.Chem.MolToSmiles` accepts the map kwarg."""
    probe = Chem.MolFromSmiles(_PROBE_SMILES)  # pyright: ignore[reportUnknownMemberType]
    try:
        _ = Chem.MolToSmiles(  # pyright: ignore[reportUnknownMemberType]
            probe, isomericSmiles=True, canonical=True, ignoreAtomMapNumbers=True
        )
    except TypeError:
        return False
    return True


#: Whether the pinned RDKit accepts ``ignoreAtomMapNumbers`` (probed at import).
IGNORE_ATOM_MAP_KWARG: Final[bool] = _probe_ignore_atom_map_kwarg()

#: Which canonicalization call form is active (never picked silently: it is
#: probed at import and logged below). Map annotations are stripped from the
#: internal molecule on both paths.
CANONICALIZATION_PATH: Final[str] = (
    KWARG_PATH if IGNORE_ATOM_MAP_KWARG else STRIP_PATH
)

logger.info(
    "reaction identity canonicalization path: %s (ignoreAtomMapNumbers=%s)",
    CANONICALIZATION_PATH,
    IGNORE_ATOM_MAP_KWARG,
)


def _canonical_component(
    component: str, *, side_name: str, reaction_smiles: str
) -> str:
    """Return the map-free isomeric canonical SMILES of one component.

    Raises
    ------
    IdentityError
        When *component* is not parseable by RDKit or parses to an empty
        molecule. The message names the component and the full reaction.
    """
    mol = Chem.MolFromSmiles(component)  # pyright: ignore[reportUnknownMemberType]
    if mol is None or mol.GetNumAtoms() == 0:  # pyright: ignore[reportUnnecessaryComparison]
        msg = (
            f"cannot parse {side_name} component {component!r} of reaction "
            f"SMILES {reaction_smiles!r}"
        )
        raise IdentityError(msg)
    # Atom maps never reach the preimage. The kwarg alone is not enough: it
    # only excludes map numbers from canonical ranking, while the writer still
    # emits the annotations (verified on rdkit 2026.3.6).
    for atom in mol.GetAtoms():  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        atom.SetAtomMapNum(0)  # pyright: ignore[reportUnknownMemberType]
    if IGNORE_ATOM_MAP_KWARG:
        return Chem.MolToSmiles(  # pyright: ignore[reportUnknownMemberType]
            mol, isomericSmiles=True, canonical=True, ignoreAtomMapNumbers=True
        )
    return Chem.MolToSmiles(  # pyright: ignore[reportUnknownMemberType]
        mol, isomericSmiles=True, canonical=True
    )


def _canonical_side(side: str, *, side_name: str, reaction_smiles: str) -> str:
    """Return the sorted, ``"+"``-joined canonical components of one side.

    Raises
    ------
    IdentityError
        For an empty side or an empty dot-separated component.
    """
    if not side.strip():
        msg = f"{side_name} side is empty in reaction SMILES {reaction_smiles!r}"
        raise IdentityError(msg)
    components: list[str] = []
    for raw_component in side.split(COMPONENT_SEPARATOR):
        component = raw_component.strip()
        if not component:
            msg = (
                f"empty {side_name} component in reaction SMILES "
                f"{reaction_smiles!r}"
            )
            raise IdentityError(msg)
        components.append(
            _canonical_component(
                component, side_name=side_name, reaction_smiles=reaction_smiles
            )
        )
    return COMPONENT_JOIN.join(sorted(components))


def _reaction_smiles_text(reaction_smiles: object) -> str:
    """Return *reaction_smiles* when it is a string, else raise.

    The public API is typed ``str``, but rows arriving from CSV/Parquet reads
    can be untyped (for example ``None``); this boundary check turns them into
    a clear :class:`IdentityError` instead of an ``AttributeError`` from deep
    inside the parser.
    """
    if not isinstance(reaction_smiles, str):
        msg = f"reaction SMILES must be a str, got {type(reaction_smiles).__name__}"
        raise IdentityError(msg)
    return reaction_smiles


def directional_preimage(reaction_smiles: str) -> tuple[str, str]:
    """Return the two canonical map-free sides in the WRITTEN order.

    The result is ``(reactant_side, product_side)``; each side is the
    ``"+"``-joined, lexicographically sorted list of its canonical component
    SMILES, exactly as used inside the identity preimage. Unlike
    :func:`canonical_reaction_identity`, which sorts the two sides and is
    therefore direction-agnostic, this function preserves the written
    direction: a reaction and its written reverse return swapped tuples. It
    exists so duplicate detection can separate same-direction exact duplicates
    from reverse pairs inside one identity group, and it is the single side
    canonicalization path both public functions share.

    Raises
    ------
    IdentityError
        For a non-string input, anything other than exactly one ``">>"``,
        empty sides/components, or unparseable components.
    """
    text = _reaction_smiles_text(reaction_smiles).strip()
    arrow_count = text.count(ARROW)
    if arrow_count != 1:
        msg = (
            f"expected exactly one {ARROW!r} separator in reaction SMILES "
            f"{reaction_smiles!r}, found {arrow_count}"
        )
        raise IdentityError(msg)
    reactant_smiles, product_smiles = text.split(ARROW)
    reactant_side = _canonical_side(
        reactant_smiles, side_name="reactant", reaction_smiles=reaction_smiles
    )
    product_side = _canonical_side(
        product_smiles, side_name="product", reaction_smiles=reaction_smiles
    )
    return reactant_side, product_side


def _identity_preimage(reaction_smiles: str) -> str:
    """Return the exact text whose UTF-8 SHA-256 is the reaction identity.

    The format is ``f"{side_a}{ARROW}{side_b}"`` with each side produced by
    :func:`directional_preimage` and the two sides sorted lexicographically, so
    a reaction and its written reverse produce the same preimage.

    Raises
    ------
    IdentityError
        For a non-string input, anything other than exactly one ``">>"``,
        empty sides/components, or unparseable components.
    """
    reactant_side, product_side = directional_preimage(reaction_smiles)
    side_a, side_b = sorted((reactant_side, product_side))
    return f"{side_a}{ARROW}{side_b}"


def canonical_reaction_identity(reaction_smiles: str) -> str:
    """Return the map- and direction-invariant SHA-256 identity of *reaction_smiles*.

    The identity ignores atom-map numbering, component order within each side,
    and reaction direction; it preserves stereochemistry (``isomericSmiles``).
    Surrounding whitespace is tolerated. The exact hash preimage is documented
    in the module docstring and returned verbatim by the private
    :func:`_identity_preimage`.

    Raises
    ------
    IdentityError
        For a non-string input, a reaction SMILES without exactly one ``">>"``,
        empty sides or components, or components RDKit cannot parse.
    """
    return sha256_bytes(_identity_preimage(reaction_smiles).encode("utf-8"))


def reverse_pair_key(reaction_smiles: str) -> str:
    """Return the direction-agnostic key that groups a reaction with its reverse.

    This is intentionally the same value as
    :func:`canonical_reaction_identity`: that identity already sorts the two
    sides, so ``reverse_pair_key(a) == reverse_pair_key(b)`` holds exactly when
    *a* and *b* describe the same chemistry in either direction. The separate
    name exists so deduplication code can state which property it relies on.
    """
    return canonical_reaction_identity(reaction_smiles)


def is_reverse_of(a: str, b: str) -> bool:
    """Return whether *a* and *b* share a direction-agnostic identity.

    ``True`` means *b* is the reverse of *a* -- or, because identical reactions
    canonicalize to the same direction-agnostic identity, that they are the same
    reaction written in the same direction. Use the raw SMILES (or a directional
    comparison) when that distinction matters.

    Raises
    ------
    IdentityError
        When either input cannot be canonicalized.
    """
    return reverse_pair_key(a) == reverse_pair_key(b)


__all__ = [
    "ARROW",
    "CANONICALIZATION_PATH",
    "COMPONENT_JOIN",
    "COMPONENT_SEPARATOR",
    "IGNORE_ATOM_MAP_KWARG",
    "IdentityError",
    "canonical_reaction_identity",
    "directional_preimage",
    "is_reverse_of",
    "reverse_pair_key",
]
