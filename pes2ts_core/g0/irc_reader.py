"""Generic reader for the Reaction-QM IRC trajectory HDF5.

The IRC archive stores one trajectory per reaction: coordinates with shape
``(n_frames, n_atoms, 3)``, optionally accompanied by per-frame energies and
forces.  This module walks that hierarchy generically — a top-level ``RXN_``
group is a reaction, any other top-level group is a bundle whose ``RXN_``
children are reactions, and non-reaction keys are skipped leniently — and
materializes either a shape-only index row or the frames of a single reaction.

The module keeps no path constants and never decides where an IRC file lives:
callers pass an explicit path, so this reader is not by itself an access path
to any quarantined artifact.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import h5py
import numpy as np
from numpy.typing import NDArray

from pes2ts_core.g0.ids import (
    REACTION_ID_PREFIX,
    BadReactionId,
    normalize_reaction_id,
)
from pes2ts_core.g0.reader import H5SchemaError

logger = logging.getLogger(__name__)

#: Trajectory dataset name searched under each reaction group.
COORDINATES_NAME: Final[str] = "coordinates"
#: Optional per-frame energy dataset name.
EHG_NAME: Final[str] = "EHG"
#: Case-insensitive substring identifying a force array.
FORCES_KEY_SUBSTRING: Final[str] = "force"
#: Number of dimensions of a trajectory dataset.
TRAJECTORY_NDIM: Final[int] = 3
#: Width of a single frame (x, y, z).
FRAME_WIDTH: Final[int] = 3


@dataclass(frozen=True, slots=True)
class IrcIndexRow:
    """Shape-only description of one IRC trajectory.

    No frame data is materialized; the index is what downstream stages may
    store, never the trajectory itself.
    """

    reaction_id: str
    n_atoms: int
    n_frames: int
    has_forces: bool


@dataclass(frozen=True, slots=True, eq=False)
class IrcFrames:
    """Frames of one IRC trajectory, materialized for a single reaction.

    ``coordinates`` has shape ``(n_frames, n_atoms, 3)`` and ``ehg`` has shape
    ``(n_frames, 3)`` when the archive stores per-frame energies, else
    ``None``.  Generated equality is disabled because of the numpy fields;
    compare with :func:`numpy.array_equal`.
    """

    reaction_id: str
    n_atoms: int
    n_frames: int
    has_forces: bool
    coordinates: NDArray[np.float64]
    ehg: NDArray[np.float64] | None


def _normalized_or_error(key: str, container: str) -> str:
    """Normalize an ``RXN_`` key, raising :class:`H5SchemaError` when invalid."""
    try:
        return normalize_reaction_id(key)
    except BadReactionId as exc:
        msg = f"{container}: reaction key {key!r} has an invalid reaction id: {exc}"
        raise H5SchemaError(msg) from exc


def iter_irc_reactions(path: str | Path) -> Generator[tuple[str, h5py.Group], None, None]:
    """Yield ``(normalized_reaction_id, reaction_group)`` for every reaction.

    The file is opened read-only and stays open for the lifetime of the
    generator.  A top-level ``RXN_`` key is treated as a reaction directly;
    otherwise every top-level group is a bundle whose sorted ``RXN_`` children
    are reactions.  A reaction ID appearing in two bundles raises
    :class:`H5SchemaError` naming both.
    """
    seen: dict[str, str] = {}
    with h5py.File(path, "r") as handle:
        for top_name, top_node in handle.items():
            if not isinstance(top_node, h5py.Group):
                logger.debug(
                    "Skipping non-group top-level node %r in %s", top_name, path
                )
                continue
            if top_name.startswith(REACTION_ID_PREFIX):
                candidates: tuple[tuple[str, h5py.Group], ...] = ((top_name, top_node),)
            else:
                candidates = tuple(
                    (key, top_node[key])
                    for key in sorted(top_node.keys())
                    if key.startswith(REACTION_ID_PREFIX)
                )
            for key, node in candidates:
                reaction_id = _normalized_or_error(key, f"Bundle {top_name!r}")
                previous = seen.get(reaction_id)
                if previous is not None:
                    msg = (
                        f"Reaction id {reaction_id} appears in bundles "
                        f"{previous!r} and {top_name!r}"
                    )
                    raise H5SchemaError(msg)
                seen[reaction_id] = top_name
                yield reaction_id, node


def _one_level_datasets(group: h5py.Group, name: str) -> list[h5py.Dataset]:
    """Return datasets called *name* on *group* or on one of its children."""
    found: list[h5py.Dataset] = []
    direct = group.get(name)
    if isinstance(direct, h5py.Dataset):
        found.append(direct)
    for child_name in sorted(group.keys()):
        child = group.get(child_name)
        if isinstance(child, h5py.Group):
            nested = child.get(name)
            if isinstance(nested, h5py.Dataset):
                found.append(nested)
    return found


def find_irc_coordinates(group: h5py.Group, *, reaction_id: str) -> h5py.Dataset:
    """Return the 3-D trajectory dataset of one reaction.

    The dataset is searched one level below the reaction group (direct child
    or child-of-child).  Multiple candidates are resolved in discovery order:
    the direct child first, then sorted child groups.

    Raises
    ------
    H5SchemaError
        When no dataset named ``coordinates`` with three dimensions exists.
    """
    for dataset in _one_level_datasets(group, COORDINATES_NAME):
        if dataset.ndim == TRAJECTORY_NDIM:
            return dataset
    msg = (
        f"Reaction {reaction_id}: no 3-D dataset named {COORDINATES_NAME!r} "
        f"under the reaction group (searched one level)"
    )
    raise H5SchemaError(msg)


def _trajectory_shape(dataset: h5py.Dataset, reaction_id: str) -> tuple[int, int]:
    """Validate a trajectory dataset and return ``(n_frames, n_atoms)``."""
    shape = dataset.shape
    if len(shape) != TRAJECTORY_NDIM or shape[-1] != FRAME_WIDTH:
        msg = (
            f"Reaction {reaction_id}: dataset {COORDINATES_NAME!r} has shape "
            f"{shape}; expected (n_frames, n_atoms, {FRAME_WIDTH})"
        )
        raise H5SchemaError(msg)
    n_frames = int(shape[0])
    if n_frames < 1:
        msg = f"Reaction {reaction_id}: dataset {COORDINATES_NAME!r} holds no frames"
        raise H5SchemaError(msg)
    return n_frames, int(shape[1])


def reaction_has_forces(node: h5py.Group) -> bool:
    """Return whether any key under *node* contains ``force`` (case-insensitive)."""
    for key, child in node.items():
        if FORCES_KEY_SUBSTRING in key.lower():
            return True
        if isinstance(child, h5py.Group) and reaction_has_forces(child):
            return True
    return False


def index_irc_reaction(reaction_id: str, group: h5py.Group) -> IrcIndexRow:
    """Describe one IRC trajectory without materializing any frame."""
    dataset = find_irc_coordinates(group, reaction_id=reaction_id)
    n_frames, n_atoms = _trajectory_shape(dataset, reaction_id)
    return IrcIndexRow(
        reaction_id=reaction_id,
        n_atoms=n_atoms,
        n_frames=n_frames,
        has_forces=reaction_has_forces(group),
    )


def read_irc_frames(reaction_id: str, group: h5py.Group) -> IrcFrames:
    """Read the full frame arrays of one IRC trajectory.

    Memory is bounded by a single reaction: ``n_frames × n_atoms × 3`` floats,
    plus ``n_frames × 3`` energies when the archive stores them.
    """
    dataset = find_irc_coordinates(group, reaction_id=reaction_id)
    n_frames, n_atoms = _trajectory_shape(dataset, reaction_id)
    coordinates = np.asarray(dataset[()], dtype=np.float64)
    ehg_dataset: h5py.Dataset | None = None
    for candidate in _one_level_datasets(group, EHG_NAME):
        if candidate.ndim >= 1 and candidate.shape[0] == n_frames:
            ehg_dataset = candidate
            break
    ehg = (
        np.asarray(ehg_dataset[()], dtype=np.float64)
        if ehg_dataset is not None
        else None
    )
    return IrcFrames(
        reaction_id=reaction_id,
        n_atoms=n_atoms,
        n_frames=n_frames,
        has_forces=reaction_has_forces(group),
        coordinates=coordinates,
        ehg=ehg,
    )


def read_irc_frames_at(path: str | Path, reaction_id: str) -> IrcFrames | None:
    """Read the frames of *reaction_id* from the IRC file at *path*.

    Returns ``None`` when the reaction is absent; the file handle is closed
    deterministically before returning either way.
    """
    iterator = iter_irc_reactions(path)
    try:
        for found_id, group in iterator:
            if found_id == reaction_id:
                return read_irc_frames(found_id, group)
    finally:
        iterator.close()
    return None


__all__ = [
    "COORDINATES_NAME",
    "EHG_NAME",
    "FORCES_KEY_SUBSTRING",
    "FRAME_WIDTH",
    "TRAJECTORY_NDIM",
    "IrcFrames",
    "IrcIndexRow",
    "find_irc_coordinates",
    "index_irc_reaction",
    "iter_irc_reactions",
    "reaction_has_forces",
    "read_irc_frames",
    "read_irc_frames_at",
]
