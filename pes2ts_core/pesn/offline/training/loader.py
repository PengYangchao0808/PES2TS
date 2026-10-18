"""Teacher dataset loading with reaction-equal sampling and collation (B3).

Reads the exported shard triples (states / labels / provenance), rebuilds the
runtime contracts (:class:`TaskGraph`, :class:`CoordinateSpec`,
:class:`PESNInput`) from the provenance view, and yields training batches
with reaction-equal weighting: per epoch every selected reaction contributes
up to ``frames_per_reaction`` frames — the TS anchor plus side-stratified
positions (near TS / middle / terminus of each present side) — and up to
``pairs_per_reaction`` same-reaction ranking pairs with ``d_i + margin <
d_j``.  Long trajectories never gain reaction weight.  Only frames with a
valid progress label are eligible; masked tasks stay masked.

Memory scaling: :class:`LazyTeacherProvider` builds a per-reaction index
from the dataset manifest plus the small provenance shards and materializes
reaction arrays on demand through an LRU cache of shard NPZ handles, so
training never holds the full dataset in RAM.  The eager
:func:`load_teacher_reactions` wrapper remains for backward compatibility
and is implemented on top of the same provider.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterator, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.inputs import (
    BondTerm,
    CoordinateDef,
    CoordinateSpec,
    PESNInput,
    TaskEdit,
    TaskGraph,
    build_pesn_input,
)
from pes2ts_core.pesn.features.graph import GraphFeatures, build_features
from pes2ts_core.utils.jsonio import read_json

DEFAULT_FRAMES_PER_REACTION: Final[int] = 12
DEFAULT_PAIRS_PER_REACTION: Final[int] = 8
DEFAULT_PAIR_MARGIN: Final[float] = 0.05
SIDE_STRATA_FRACTIONS: Final[tuple[float, ...]] = (0.10, 0.35, 0.60, 0.85, 1.0)
DEFAULT_SHARD_CACHE_SIZE: Final[int] = 2
ALLOWED_TRAIN_SUBSETS: Final[frozenset[str]] = frozenset({"train_fit", "train_dev"})


class LoaderError(RuntimeError):
    """Typed teacher-dataset loading failure."""


def task_graph_from_doc(doc: dict) -> TaskGraph:
    graph = TaskGraph(
        schema_version="pesn_task_graph_v1",
        atomic_numbers=tuple(int(z) for z in doc["atomic_numbers"]),
        atom_map=tuple(int(m) for m in doc["atom_map"]),
        reactant_bonds=tuple(BondTerm(int(i), int(j), float(o)) for i, j, o in doc["reactant_bonds"]),
        product_bonds=tuple(BondTerm(int(i), int(j), float(o)) for i, j, o in doc["product_bonds"]),
        edits=tuple(
            TaskEdit(str(kind), int(i), int(j), float(ro), float(po))
            for kind, i, j, ro, po in doc["edits"]
        ),
        source=str(doc["source"]),
        provenance=dict(doc.get("provenance", {})),
    )
    graph.validate()
    return graph


def coordinate_spec_from_doc(doc: dict) -> CoordinateSpec:
    spec = CoordinateSpec(
        schema_version="pesn_coordinate_spec_v1",
        coords=tuple(
            CoordinateDef(
                kind=str(c["kind"]),
                atoms=tuple(int(a) for a in c["atoms"]),
                units=str(c["units"]),
                role=str(c["role"]),
                scale_a=float(c["scale_a"]),
                ref_q=float(c["ref_q"]),
                label=str(c["label"]),
            )
            for c in doc["coords"]
        ),
        spec_version=str(doc["spec_version"]),
    )
    return spec


@dataclass(frozen=True, slots=True)
class ReactionData:
    """One reaction's teacher views plus rebuilt runtime contracts."""

    reaction_id: str
    atomic_numbers: np.ndarray
    coords: np.ndarray  # (F, N, 3) float32
    task_graph: TaskGraph
    coordinate_spec: CoordinateSpec
    s: np.ndarray  # (F,)
    d: np.ndarray  # (F,)
    s_mask: np.ndarray  # (F,)
    side: np.ndarray  # (F,)
    d_arc: np.ndarray  # (F,)
    direction: np.ndarray  # (F, C)
    activity: np.ndarray  # (F, C)
    dir_mask: np.ndarray  # (F,)
    act_mask: np.ndarray  # (F,)
    method: str

    @property
    def n_frames(self) -> int:
        return int(self.coords.shape[0])


@dataclass(frozen=True, slots=True)
class ReactionRef:
    """Index row locating one non-excluded reaction inside the shard triples."""

    reaction_id: str
    subset: str  # "" when the manifest declares no subset membership
    shard_tag: str
    row_index: int


class LazyTeacherProvider:
    """Per-split lazy reader over exported teacher shard triples.

    Index construction reads only ``dataset_manifest.json`` plus the small
    per-shard ``provenance_*.json`` files (metadata; no state/label arrays)
    and records, in the canonical eager order (manifest shard order, then
    provenance insertion order), one :class:`ReactionRef` per non-excluded
    reaction.  Subset membership comes from the manifest ``subset_members``
    view; a manifest that declares subsets other than ``train_fit`` /
    ``train_dev`` hard-fails unless ``allow_extra_subsets=True``, and even
    then non-training subsets are never served.  Reaction arrays are
    materialized on demand through :meth:`get_reaction`; at most
    ``shard_cache_size`` shard NPZ handle pairs stay open (LRU).
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        shard_cache_size: int = DEFAULT_SHARD_CACHE_SIZE,
        allow_extra_subsets: bool = False,
    ) -> None:
        self.root = Path(dataset_dir)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self.allow_extra_subsets = bool(allow_extra_subsets)
        manifest_path = self.root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise LoaderError(f"dataset manifest missing: {manifest_path}")
        self.manifest = read_json(manifest_path)
        subset_members = self.manifest.get("subset_members") or {}
        if not isinstance(subset_members, dict):
            raise LoaderError("dataset manifest subset_members must be a mapping")
        extra = set(subset_members) - ALLOWED_TRAIN_SUBSETS
        if extra and not self.allow_extra_subsets:
            raise LoaderError(
                f"dataset declares non-training subsets {sorted(extra)}; "
                "sealed/held-out material must not be loaded "
                "(pass allow_extra_subsets=True only when explicitly intended)"
            )
        self._subset_members = {
            str(name): {str(rid) for rid in members}
            for name, members in subset_members.items()
        }
        self._index: list[ReactionRef] = []
        self._by_id: dict[str, ReactionRef] = {}
        self._shard_tags: list[str] = []
        self._shard_cache: OrderedDict[str, tuple[object, object, dict]] = OrderedDict()
        self._build_index()

    def _build_index(self) -> None:
        member_of: dict[str, str] = {}
        for subset, members in self._subset_members.items():
            for rid in members:
                if rid in member_of:
                    raise LoaderError(f"reaction {rid} declared in multiple subsets")
                member_of[rid] = subset
        for shard in self.manifest.get("shards", []):
            tag = str(shard["shard"])
            provenance_path = self.root / f"provenance_{tag}.json"
            if not provenance_path.is_file():
                raise LoaderError(f"provenance shard missing: {provenance_path}")
            provenance = read_json(provenance_path)
            self._shard_tags.append(tag)
            for row_index, (rid, prov) in enumerate(provenance["reactions"].items()):
                if prov.get("exclusion") is not None:
                    continue
                reaction_id = str(rid)
                if reaction_id in self._by_id:
                    raise LoaderError(f"duplicate reaction id in dataset: {reaction_id}")
                ref = ReactionRef(
                    reaction_id=reaction_id,
                    subset=member_of.get(reaction_id, ""),
                    shard_tag=tag,
                    row_index=row_index,
                )
                self._index.append(ref)
                self._by_id[reaction_id] = ref

    @property
    def open_shard_count(self) -> int:
        """Shard NPZ handle pairs currently held in the LRU cache."""
        return len(self._shard_cache)

    def reaction_ids(self, subset: str | None = None) -> list[str]:
        """Canonical-order reaction ids (all non-excluded, or one subset)."""
        if subset is None:
            return [ref.reaction_id for ref in self._index]
        if subset not in self._subset_members:
            raise LoaderError(f"subset not declared in dataset manifest: {subset!r}")
        if subset not in ALLOWED_TRAIN_SUBSETS:
            raise LoaderError(
                f"subset {subset!r} is not a training subset; "
                "sealed/held-out material is never served"
            )
        return [ref.reaction_id for ref in self._index if ref.subset == subset]

    def valid_label_reaction_ids(
        self,
        reaction_ids: Sequence[str] | None = None,
        *,
        subset: str | None = None,
    ) -> list[str]:
        """Ids (default: one subset) whose stored ``s_mask`` has a valid frame.

        Reads only per-reaction ``s_mask__*`` members of the labels shards —
        never state coordinates or full label stacks — so callers keep the
        eager ``s_mask.any()`` filter exact without materializing reactions.
        Shards holding none of the requested ids are never opened.
        """
        ids = (
            list(reaction_ids)
            if reaction_ids is not None
            else self.reaction_ids(subset)
        )
        by_shard: dict[str, list[str]] = {}
        for rid in ids:
            ref = self._by_id.get(rid)
            if ref is None:
                raise LoaderError(f"reaction not in dataset index: {rid}")
            by_shard.setdefault(ref.shard_tag, []).append(rid)
        valid: set[str] = set()
        for tag in self._shard_tags:
            shard_ids = by_shard.get(tag)
            if not shard_ids:
                continue
            labels = np.load(self.root / f"labels_{tag}.npz")
            try:
                for rid in shard_ids:
                    if bool(np.asarray(labels[f"s_mask__{rid}"]).any()):
                        valid.add(rid)
            finally:
                labels.close()
        return [rid for rid in ids if rid in valid]

    def iter_reactions(self, subset: str | None = None) -> Iterator[ReactionData]:
        """Yield materialized reactions in canonical order (optionally one subset)."""
        for rid in self.reaction_ids(subset):
            yield self.get_reaction(rid)

    def get_reaction(self, reaction_id: str) -> ReactionData:
        """Materialize one reaction from its shard (LRU-cached handles)."""
        ref = self._by_id.get(reaction_id)
        if ref is None:
            raise LoaderError(f"reaction not in dataset index: {reaction_id}")
        states, labels, provenance = self._shard(ref.shard_tag)
        rid = ref.reaction_id
        prov = provenance["reactions"][rid]
        return ReactionData(
            reaction_id=rid,
            atomic_numbers=states[f"atomic_numbers__{rid}"].astype(np.int64),
            coords=states[f"coords__{rid}"].astype(np.float64),
            task_graph=task_graph_from_doc(prov["task_graph"]),
            coordinate_spec=coordinate_spec_from_doc(prov["coordinate_spec"]),
            s=labels[f"s__{rid}"].astype(np.float64),
            d=labels[f"d__{rid}"].astype(np.float64),
            s_mask=labels[f"s_mask__{rid}"].astype(bool),
            side=labels[f"side__{rid}"].astype(np.int64),
            d_arc=labels[f"d_arc__{rid}"].astype(np.float64),
            direction=labels[f"direction__{rid}"].astype(np.float64),
            activity=labels[f"activity__{rid}"].astype(np.float64),
            dir_mask=labels[f"dir_mask__{rid}"].astype(bool),
            act_mask=labels[f"act_mask__{rid}"].astype(bool),
            method=str(prov.get("source_method", "unknown")),
        )

    def _shard(self, tag: str):
        cached = self._shard_cache.pop(tag, None)
        if cached is not None:
            self._shard_cache[tag] = cached
            return cached
        states = np.load(self.root / f"states_{tag}.npz")
        labels = np.load(self.root / f"labels_{tag}.npz")
        provenance = read_json(self.root / f"provenance_{tag}.json")
        entry = (states, labels, provenance)
        self._shard_cache[tag] = entry
        while len(self._shard_cache) > self.shard_cache_size:
            _, evicted = self._shard_cache.popitem(last=False)
            evicted[0].close()
            evicted[1].close()
        return entry


class LazyReactionSequence(Sequence[ReactionData]):
    """Index-addressable lazy view over provider reaction ids.

    Satisfies the ``Sequence[ReactionData]`` contract consumed by the trainer
    so only the reactions of the current batch are materialized.
    """

    def __init__(
        self, provider: LazyTeacherProvider, reaction_ids: Sequence[str]
    ) -> None:
        self.provider = provider
        self._ids = list(reaction_ids)

    @property
    def reaction_ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __bool__(self) -> bool:
        return bool(self._ids)

    def __getitem__(self, index: int) -> ReactionData:
        return self.provider.get_reaction(self._ids[index])

    def __iter__(self) -> Iterator[ReactionData]:
        for position in range(len(self._ids)):
            yield self.provider.get_reaction(self._ids[position])


def load_teacher_reactions(dataset_dir: str | Path) -> list[ReactionData]:
    """Load every non-excluded reaction from the shard triples.

    Full-materialization wrapper over :class:`LazyTeacherProvider`, kept for
    backward compatibility (tests, evaluation entry points); training uses
    the lazy provider directly so memory scales per batch, not per dataset.
    """
    provider = LazyTeacherProvider(dataset_dir)
    return list(provider.iter_reactions())


def select_frames(
    reaction: ReactionData,
    *,
    frames_per_reaction: int,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """TS anchor + side-stratified frames among label-valid frames."""
    valid = np.where(reaction.s_mask)[0]
    if valid.size == 0:
        return []
    ts_frames = valid[reaction.side[valid] == 0]
    anchor = int(ts_frames[0]) if ts_frames.size else None
    budget = frames_per_reaction - (1 if anchor is not None else 0)

    sides_present = [s for s in (-1, 1) if (reaction.side[valid] == s).any()]
    per_side = max(1, budget // len(sides_present)) if sides_present else 0

    chosen: set[int] = {anchor} if anchor is not None else set()
    for side_value in sides_present:
        side_frames = valid[reaction.side[valid] == side_value]
        ordered = side_frames[np.argsort(reaction.d_arc[side_frames])]
        n_take = min(per_side, ordered.size)
        positions = [
            int(round(fraction * (ordered.size - 1)))
            for fraction in SIDE_STRATA_FRACTIONS
        ][:n_take]
        if rng is not None and ordered.size > n_take:
            jittered = set(
                int(np.clip(p + rng.integers(-1, 2), 0, ordered.size - 1))
                for p in positions
            )
            while len(jittered) < n_take:
                jittered.add(int(rng.integers(0, ordered.size)))
            positions = sorted(jittered)[:n_take]
        chosen.update(int(ordered[p]) for p in positions)

    if len(chosen) > frames_per_reaction and anchor is not None:
        chosen.discard(anchor)
    result = sorted(chosen)[:frames_per_reaction]
    if anchor is not None and anchor not in result:
        result = [anchor, *[f for f in result if f != anchor]][:frames_per_reaction]
    return result


def build_pairs(
    reaction: ReactionData,
    frame_indices: Sequence[int],
    *,
    pairs_per_reaction: int,
    margin: float,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """Same-reaction pairs with ``d_i + margin < d_j`` (i closer to TS)."""
    candidates: list[tuple[int, int]] = []
    for a in frame_indices:
        for b in frame_indices:
            if a == b or not (reaction.s_mask[a] and reaction.s_mask[b]):
                continue
            if reaction.d[a] + margin < reaction.d[b]:
                candidates.append((a, b))
    if not candidates:
        return []
    if rng is not None and len(candidates) > pairs_per_reaction:
        pick = rng.choice(len(candidates), size=pairs_per_reaction, replace=False)
        return [candidates[int(k)] for k in sorted(pick)]
    step = max(1, len(candidates) // pairs_per_reaction)
    return candidates[::step][:pairs_per_reaction]


@dataclass(slots=True)
class FrameSample:
    reaction_id: str
    frame_index: int
    features: GraphFeatures
    s: float
    d: float
    valid: bool
    direction: np.ndarray
    dir_valid: bool
    activity: np.ndarray
    act_valid: bool
    pair_role: int = 0


@dataclass(slots=True)
class TrainBatch:
    samples: list[FrameSample]
    pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.samples)


def build_pesn_input_for_frame(
    reaction: ReactionData, frame_index: int
) -> PESNInput:
    return build_pesn_input(
        input_id=f"{reaction.reaction_id}#f{frame_index}",
        task_graph=reaction.task_graph,
        coordinate_spec=reaction.coordinate_spec,
        coordinates=reaction.coords[frame_index],
        charge=None,
        multiplicity=None,
        method=reaction.method,
    )


def make_batch(
    reactions: Sequence[ReactionData],
    *,
    frames_per_reaction: int,
    pairs_per_reaction: int,
    pair_margin: float,
    rng: np.random.Generator | None = None,
) -> TrainBatch:
    """Collate one reaction-equal training batch."""
    samples: list[FrameSample] = []
    pairs: list[tuple[int, int]] = []
    for reaction in reactions:
        frame_indices = select_frames(
            reaction,
            frames_per_reaction=frames_per_reaction,
            rng=rng,
        )
        local_pairs = build_pairs(
            reaction,
            frame_indices,
            pairs_per_reaction=pairs_per_reaction,
            margin=pair_margin,
            rng=rng,
        )
        frame_to_position = {
            frame_index: position for position, frame_index in enumerate(frame_indices)
        }
        base = len(samples)
        for frame_index in frame_indices:
            prediction_input = build_pesn_input_for_frame(reaction, frame_index)
            features = build_features(prediction_input)
            samples.append(
                FrameSample(
                    reaction_id=reaction.reaction_id,
                    frame_index=frame_index,
                    features=features,
                    s=float(reaction.s[frame_index]),
                    d=float(reaction.d[frame_index]),
                    valid=bool(reaction.s_mask[frame_index]),
                    direction=reaction.direction[frame_index],
                    dir_valid=bool(reaction.dir_mask[frame_index]),
                    activity=reaction.activity[frame_index],
                    act_valid=bool(reaction.act_mask[frame_index]),
                )
            )
        pairs.extend(
            (base + frame_to_position[a], base + frame_to_position[b])
            for a, b in local_pairs
            if a in frame_to_position and b in frame_to_position
        )
    return TrainBatch(samples=samples, pairs=pairs)


def epoch_batch_indices(
    n_reactions: int,
    *,
    batch_reactions: int,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> list[list[int]]:
    """Deterministic per-epoch reaction index chunks (reaction-equal weighting).

    Identical RNG semantics to the historical :func:`epoch_batches` shuffle:
    ``np.random.default_rng([seed, epoch])`` over ``np.arange(n_reactions)``.
    """
    order = np.arange(n_reactions)
    generator = np.random.default_rng([seed, epoch])
    if shuffle:
        generator.shuffle(order)
    chunks = [
        order[start : start + batch_reactions].tolist()
        for start in range(0, n_reactions, batch_reactions)
    ]
    return [chunk for chunk in chunks if chunk]


def epoch_batches(
    reactions: Sequence[ReactionData],
    *,
    batch_reactions: int,
    frames_per_reaction: int,
    pairs_per_reaction: int,
    pair_margin: float,
    seed: int,
    epoch: int,
    shuffle: bool = True,
) -> list[list[ReactionData]]:
    """Deterministic per-epoch reaction chunks (reaction-equal weighting)."""
    index_chunks = epoch_batch_indices(
        len(reactions),
        batch_reactions=batch_reactions,
        seed=seed,
        epoch=epoch,
        shuffle=shuffle,
    )
    return [[reactions[int(i)] for i in chunk] for chunk in index_chunks]


__all__ = [
    "ALLOWED_TRAIN_SUBSETS",
    "DEFAULT_FRAMES_PER_REACTION",
    "DEFAULT_PAIRS_PER_REACTION",
    "DEFAULT_PAIR_MARGIN",
    "DEFAULT_SHARD_CACHE_SIZE",
    "FrameSample",
    "LazyReactionSequence",
    "LazyTeacherProvider",
    "LoaderError",
    "ReactionData",
    "ReactionRef",
    "TrainBatch",
    "build_pairs",
    "build_pesn_input_for_frame",
    "epoch_batch_indices",
    "epoch_batches",
    "load_teacher_reactions",
    "make_batch",
    "select_frames",
]
