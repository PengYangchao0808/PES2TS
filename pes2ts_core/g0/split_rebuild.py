"""Seeded Butina rebuild of the whole-inventory split.

The rebuild remediation discards the authors' assignment and re-partitions
EVERY inventory reaction: DRFP bit vectors are clustered with Butina over
Tanimoto distances (threshold ``1 - near_dup.cluster_threshold``) and whole
clusters are then assigned 80/10/10 so no cluster ever straddles splits.

Determinism
-----------
Vectors are fed to Butina in sorted ``reaction_id`` order and
``ClusterData`` runs with ``reordering=False``, so the fixed input order fully
determines the clusters. Clusters are walked in ``(-size, minimum_member_id)``
order and each is assigned to the label with the smallest filled-quota ratio,
breaking ties by the fixed ``train``/``valid``/``test`` label order. The only
nondeterminism left is the caller's ``split.seed``, which is carried into the
manifest.

Scale limitation
----------------
RDKit's Butina implementation materializes the full condensed distance matrix
(O(N^2) floats), so rebuild is practical for fixtures and moderate cohorts
only; at the full 199,890-reaction scale it would need a blocked
implementation. The authors' split is expected to audit clean, in which case
rebuild never runs.

RDKit API note
--------------
The task brief referred to ``DataStructs.ButinaCluster``; that symbol does not
exist in rdkit 2026.3.6. The available (and used) API is
``rdkit.ML.Cluster.Butina.ClusterData``, which consumes the symmetrically
stored condensed distance list built here with blocked
``DataStructs.BulkTanimotoSimilarity`` calls.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from rdkit import DataStructs
from rdkit.DataStructs import ExplicitBitVect
from rdkit.ML.Cluster import Butina

from pes2ts_core.g0.split_sources import SPLIT_LABELS, read_inventory_ids

logger = logging.getLogger(__name__)

#: Target reaction fraction per split label for the greedy cluster walk.
SPLIT_FRACTIONS: Final[dict[str, float]] = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,
}


@dataclass(frozen=True, slots=True)
class RebuildResult:
    """Outcome of one :func:`rebuild_assignment` call."""

    assignment: dict[str, str]
    cluster_cutoff: float
    n_clusters: int
    seed: int


def butina_cluster_indices(
    vectors: Sequence[ExplicitBitVect],
    cutoff: float,
    block_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Cluster *vectors* with Butina at *cutoff* Tanimoto distance.

    The condensed distance list is filled with blocked
    :func:`rdkit.DataStructs.BulkTanimotoSimilarity` calls in row order
    ``(1,0), (2,0), (2,1), ...``, which is the layout
    :func:`rdkit.ML.Cluster.Butina.ClusterData` expects for
    ``isDistData=True``. The returned clusters are index tuples into *vectors*
    (singletons included) and are order-dependent, so callers must fix the
    input order.
    """
    count = len(vectors)
    if count == 0:
        msg = "Cannot cluster zero reaction vectors"
        raise ValueError(msg)
    if block_size < 1:
        msg = f"near_dup.block_size must be >= 1, got {block_size}"
        raise ValueError(msg)
    distances: list[float] = []
    for index in range(1, count):
        start = 0
        while start < index:
            end = min(start + block_size, index)
            similarities = DataStructs.BulkTanimotoSimilarity(
                vectors[index], vectors[start:end]
            )
            distances.extend(1.0 - float(value) for value in similarities)
            start = end
    clusters = Butina.ClusterData(
        distances, count, cutoff, isDistData=True, reordering=False
    )
    return tuple(tuple(int(index) for index in cluster) for cluster in clusters)


def rebuild_assignment(
    config: Mapping[str, Any],
    fingerprints: Sequence[tuple[str, ExplicitBitVect]],
) -> RebuildResult:
    """Re-partition the whole inventory into 80/10/10 cluster-preserving splits.

    Every inventory reaction must carry a fingerprint; a missing one is a hard
    :class:`ValueError` (a rebuild that silently dropped reactions would change
    the evaluation population). Clusters are assigned greedily to the label
    with the smallest ``assigned_count / quota`` ratio, where the quota is the
    configured fraction of the total reaction count.

    Raises
    ------
    FileNotFoundError
        When the inventory Parquet is missing.
    ValueError
        When the inventory or fingerprint table is malformed, when a
        fingerprint is missing, or when the cluster cutoff is outside (0, 1).
    """
    _inventory_path, inventory_ids = read_inventory_ids(config)
    if len(set(inventory_ids)) != len(inventory_ids):
        msg = "Inventory carries duplicate reaction IDs; refusing to rebuild"
        raise ValueError(msg)
    by_id: dict[str, ExplicitBitVect] = {}
    for reaction_id, vector in fingerprints:
        if reaction_id in by_id:
            msg = f"Duplicate reaction ID {reaction_id} in fingerprint input"
            raise ValueError(msg)
        by_id[reaction_id] = vector
    sorted_ids = sorted(inventory_ids)
    missing = [reaction_id for reaction_id in sorted_ids if reaction_id not in by_id]
    if missing:
        msg = (
            f"Fingerprint table lacks {len(missing)} inventory reaction(s), "
            f"e.g. {missing[:10]}"
        )
        raise ValueError(msg)
    cutoff = 1.0 - float(config["near_dup"]["cluster_threshold"])
    if not 0.0 < cutoff < 1.0:
        msg = (
            "near_dup.cluster_threshold must yield a Butina distance cutoff in "
            f"(0, 1), got {cutoff}"
        )
        raise ValueError(msg)

    vectors = [by_id[reaction_id] for reaction_id in sorted_ids]
    block_size = int(config["near_dup"]["block_size"])
    clusters = butina_cluster_indices(vectors, cutoff, block_size)
    groups = sorted(
        (sorted(sorted_ids[index] for index in cluster) for cluster in clusters),
        key=lambda members: (-len(members), members[0]),
    )
    total = len(sorted_ids)
    quotas = {
        label: SPLIT_FRACTIONS[label] * total for label in SPLIT_LABELS
    }
    assigned_counts = dict.fromkeys(SPLIT_LABELS, 0)
    assignment: dict[str, str] = {}
    for members in groups:
        label = min(
            SPLIT_LABELS,
            key=lambda candidate: (
                assigned_counts[candidate] / quotas[candidate],
                SPLIT_LABELS.index(candidate),
            ),
        )
        for member in members:
            assignment[member] = label
        assigned_counts[label] += len(members)
    if len(assignment) != total:
        msg = f"Butina assigned {len(assignment)} of {total} inventory reaction(s)"
        raise ValueError(msg)
    logger.info(
        "Rebuild: %d reaction(s), %d cluster(s), cutoff=%.3f, seed=%s -> "
        "train=%d valid=%d test=%d",
        total,
        len(groups),
        cutoff,
        config["split"]["seed"],
        assigned_counts["train"],
        assigned_counts["valid"],
        assigned_counts["test"],
    )
    return RebuildResult(
        assignment=assignment,
        cluster_cutoff=cutoff,
        n_clusters=len(groups),
        seed=int(config["split"]["seed"]),
    )


__all__ = [
    "SPLIT_FRACTIONS",
    "RebuildResult",
    "butina_cluster_indices",
    "rebuild_assignment",
]
