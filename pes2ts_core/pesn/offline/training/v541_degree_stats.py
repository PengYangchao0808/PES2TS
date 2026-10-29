"""V5.4.1 WP3b — inner-train degree statistics for the E2 PNA scaler
(plan §6.7 E2 / §6.6; config ``feature_recipe.degree_stats``).

Scope discipline — this module is NEW and touches nothing else.  It
computes, EXACTLY ONCE per run, the frozen reference ``mu_log`` that
``pna_chemical.PNAChemicalEmbedding`` consumes::

    mu_log = mean over the inner-train node population of log1p(d)

where ``d`` is the node's in-degree in the legal ``G_chem`` view.  What
is frozen here:

* **inner-train ONLY** — the caller passes the explicit inner-train
  position list; the interface REFUSES any ``split`` other than
  ``"inner-train"`` (dev / inner-val / confirm / dev_confirm are typed
  refusals, plan §6.6 "不能使用 inner-val/dev/confirm");
* **each planned sample once, both intents** — every position is visited
  exactly once (no sampler, no replacement trajectory, plan §352); the
  fixed index set normally contains both directions of a sample;
* **exactly the G_chem rules** — pair qualification / merge replicate
  ``key_atom_graph_v3.build_sample_features_v3`` +
  ``feature_providers.build_chem_graph`` bit-for-bit (same ``/3.0``
  scaling, sticky covalent flag, last-wins bond orders, self-loop and
  range refusals); the fixture tests cross-prove the intent fast path
  against the real provider on every WP3a batch;
* **no self loop, no spatial degree, no label weight** — only real
  chemical-edge in-degree (== incident qualified-chem-pair count, since
  every pair is expanded to both directions);
* **provenance** — the result records SHAs of the split name, the
  canonical position set, the graph recipe and the seed, plus the peak
  RSS and elapsed time of the fit (§6.7 "记录 SHA 和成本"); its
  ``stats_sha256`` binds the payload and is handed verbatim to
  ``PNAChemicalEmbedding.set_degree_stats``;
* **streaming** — one ``(intent, n_atoms)`` is resolved per position at a
  time; the columnar fast path reads the startup intent/curve tables and
  never constructs a record (no feature/label materialisation, I4
  discipline), while the fallback records path exists for v1 loaders and
  tests.

An all-edgeless population is a typed refusal (``mu_log`` would be 0 and
the amplification scaler undefined).
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from typing import Any, Final, Iterator, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.features.key_atom_spec_v3 import BOND_SCALE
from pes2ts_core.pesn.models.v541.feature_providers.common import (
    CHEM_EDGE_FIELDS,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
    _startup_of,
)
from pes2ts_core.pesn.offline.training.v541_identity import (
    positions_sha256,
    sha256_json,
    sha256_text,
)

#: Stats-file schema (config ``feature_recipe.degree_stats.schema``).
DEGREE_STATS_SCHEMA: Final[str] = "pes2ts_v541_degree_stats_v1"

#: The ONLY split this helper may fit on (config ``fit_split``).
ALLOWED_SPLIT: Final[str] = "inner-train"
#: Explicitly refused split names (config ``forbidden_fit_splits``).
FORBIDDEN_SPLITS: Final[tuple[str, ...]] = (
    "dev", "inner-val", "confirm", "dev_confirm")

#: Frozen description of the G_chem recipe the degrees are derived with
#: (hashed into the result so the stats file is bound to the recipe).
GRAPH_RECIPE_V541_CHEM: Final[dict[str, Any]] = {
    "view": "G_chem",
    "provider": ("pes2ts_core.pesn.models.v541.feature_providers."
                 "build_chem_graph"),
    "chem_edge_fields": list(CHEM_EDGE_FIELDS),
    "qualification": "b_src>0 | b_tgt>0 | is_changed>0 | covalent_union>0",
    "bond_scale": float(BOND_SCALE),
    "dedup": "undirected pair, last-wins orders, sticky covalent",
    "directions": "bidirectional expansion, bitwise-identical attrs",
    "degree": ("in-degree of real chem edges (== incident qualified pair); "
               "no self loop, no spatial, no label weight"),
}


class V541DegreeStatsError(RuntimeError):
    """Typed refusal of the inner-train degree-stats fit."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise V541DegreeStatsError(code, detail)


# ---------------------------------------------------------------------------
# G_chem pair set of ONE intent (bit-for-bit the builder + provider rules)
# ---------------------------------------------------------------------------
def qualified_chem_pairs(intent: Any) -> list[tuple[int, int]]:
    """Sorted undirected ``G_chem`` pairs of a single intent.

    Replicates ``build_sample_features_v3``'s pair loop exactly: the
    pair key is normalised to ``i < j`` (the builder's ``_pair``), bond
    orders are scaled by ``BOND_SCALE`` with LAST-occurrence-wins
    semantics and the covalent flag is sticky-OR; qualification is the
    provider predicate ``chem_qualifies`` on the resulting record
    (``b_src>0 | b_tgt>0 | is_changed>0 | covalent>0``).  Self loops and
    out-of-range endpoints are typed refusals (the provider refuses
    ``V541_PROVIDER_SELF_LOOP`` / ``V541_PROVIDER_EDGE_RANGE`` on the
    same input).
    """
    merged: dict[tuple[int, int], list[float]] = {}
    for (i, j), (b_src_raw, b_tgt_raw) in intent.pair_orders().items():
        a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
        _require(a != b, "V541_DEGREE_STATS_TOPOLOGY",
                 f"intent bond ({i},{j}) is a self loop")
        slot = merged.setdefault((a, b), [0.0, 0.0, False])  # type: ignore[list-item]
        if b_src_raw > 0.0 or b_tgt_raw > 0.0:
            slot[2] = True                        # sticky covalent (builder)
        slot[0] = b_src_raw / float(BOND_SCALE)   # last wins (builder)
        slot[1] = b_tgt_raw / float(BOND_SCALE)

    pairs: list[tuple[int, int]] = []
    for (a, b), (bs, bt, covalent) in merged.items():   # type: ignore[misc]
        is_changed = abs(bt - bs) > 0.0
        if bs > 0.0 or bt > 0.0 or is_changed or covalent:
            pairs.append((a, b))
    pairs.sort()
    return pairs


def chem_in_degrees(intent: Any, n_atoms: int) -> np.ndarray:
    """Per-node in-degrees of ``G_chem`` for ONE sample.

    Every qualified pair expands to BOTH message directions, so the
    in-degree of node ``i`` equals the number of qualified pairs incident
    on ``i``.  Endpoints outside ``[0, n_atoms)`` are typed refusals (the
    provider refuses ``V541_PROVIDER_EDGE_RANGE`` on the same input).
    """
    n_atoms = int(n_atoms)
    _require(n_atoms >= 1, "V541_DEGREE_STATS_TOPOLOGY",
             f"n_atoms must be >= 1, got {n_atoms}")
    degrees = np.zeros(n_atoms, dtype=np.int64)
    for a, b in qualified_chem_pairs(intent):
        _require(0 <= a < n_atoms and 0 <= b < n_atoms,
                 "V541_DEGREE_STATS_TOPOLOGY",
                 f"bond ({a},{b}) references an atom outside [0,{n_atoms})")
        degrees[a] += 1
        degrees[b] += 1
    return degrees


# ---------------------------------------------------------------------------
# streaming position → (intent, n_atoms) resolution
# ---------------------------------------------------------------------------
def _resolve_intent_n_atoms(
        samples: Any, startup: Any, position: int,
) -> tuple[Any, int]:
    """``(intent, n_atoms)`` of ONE position.

    Columnar fast path: the startup intent/curve tables give the row
    without constructing a single record (no label/feature access, I4
    discipline).  Fallback: one lazily constructed record (v1 loaders).
    """
    if startup is not None and hasattr(startup, "intent_idx"):
        ds = getattr(startup, "_ds", None)
        intent_by_row = getattr(ds, "_intent_by_row", None)
        curve_atomic = getattr(ds, "_curve_atomic", None)
        if intent_by_row is not None and curve_atomic is not None:
            intent = intent_by_row[int(np.asarray(startup.intent_idx)[position])]
            curve_id = startup.curve_ids[int(np.asarray(startup.curve_idx)[position])]
            return intent, len(curve_atomic[curve_id])
    record = samples[position]
    return record.intent, len(record.atomic_numbers)


def _iter_topologies(
        samples: Any, positions: Sequence[int],
) -> Iterator[tuple[Any, int]]:
    """Stream ``(intent, n_atoms)`` over ``positions`` — one at a time."""
    startup = _startup_of(samples)
    for position in positions:
        yield _resolve_intent_n_atoms(samples, startup, int(position))


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DegreeStatsResult:
    """Frozen output of :func:`compute_degree_stats` (JSON-able)."""

    schema: str
    split: str
    mu_log: float
    stats_sha256: str
    split_sha256: str
    positions_sha256: str
    graph_recipe_sha256: str
    seed: int
    seed_sha256: str
    n_positions: int
    n_unique_graphs: int
    weighted_graph_total: int
    weighted_node_total: int
    weighted_edge_total: int
    weighted_pair_total: int
    degree_histogram: dict[int, int]
    elapsed_seconds: float
    max_rss_kb: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-able mapping (histogram keys stringified)."""
        return {
            "schema": self.schema,
            "split": self.split,
            "mu_log": self.mu_log,
            "stats_sha256": self.stats_sha256,
            "split_sha256": self.split_sha256,
            "positions_sha256": self.positions_sha256,
            "graph_recipe_sha256": self.graph_recipe_sha256,
            "seed": self.seed,
            "seed_sha256": self.seed_sha256,
            "n_positions": self.n_positions,
            "n_unique_graphs": self.n_unique_graphs,
            "weighted_graph_total": self.weighted_graph_total,
            "weighted_node_total": self.weighted_node_total,
            "weighted_edge_total": self.weighted_edge_total,
            "weighted_pair_total": self.weighted_pair_total,
            "degree_histogram": {str(k): v for k, v in
                                 sorted(self.degree_histogram.items())},
            "elapsed_seconds": self.elapsed_seconds,
            "max_rss_kb": self.max_rss_kb,
        }


def compute_degree_stats(
        samples: Any,
        positions: Sequence[int],
        *,
        split: str,
        seed: int,
        graph_recipe: Mapping[str, Any] | None = None,
) -> DegreeStatsResult:
    """Fit ``mu_log`` over the caller's EXPLICIT inner-train positions.

    ``samples`` is the dataset's sample sequence (``dataset.samples`` —
    the columnar lazy sequence or a v1 record tuple).  ``positions`` is
    the fixed inner-train index set; it is canonicalised to a sorted,
    de-duplicated SET so the result depends only on the population (the
    "每个计划样本一次" discipline) and ``split`` MUST be
    ``"inner-train"`` — dev / inner-val / confirm are typed refusals.
    """
    t_start = time.perf_counter()

    _require(str(split) == ALLOWED_SPLIT, "V541_DEGREE_STATS_SPLIT",
             f"degree stats may only be fitted on {ALLOWED_SPLIT!r}; got "
             f"{split!r} (forbidden: {', '.join(FORBIDDEN_SPLITS)})")
    try:
        canonical = sorted({int(p) for p in positions})
    except (TypeError, ValueError) as exc:
        raise V541DegreeStatsError(
            "V541_DEGREE_STATS_POSITION_RANGE",
            f"positions must be integers, got {exc}") from exc
    _require(bool(canonical), "V541_DEGREE_STATS_EMPTY",
             "positions is empty — no inner-train population to fit")
    n_total = len(samples)
    for p in canonical:
        _require(0 <= p < n_total, "V541_DEGREE_STATS_POSITION_RANGE",
                 f"position {p} outside [0,{n_total})")

    recipe = dict(graph_recipe) if graph_recipe is not None \
        else dict(GRAPH_RECIPE_V541_CHEM)

    histogram: dict[int, int] = {}
    unique_topologies: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    sum_log1p = 0.0
    node_total = 0
    pair_total = 0
    edge_total = 0
    for intent, n_atoms in _iter_topologies(samples, canonical):
        pairs = qualified_chem_pairs(intent)
        for a, b in pairs:
            _require(0 <= a < n_atoms and 0 <= b < n_atoms,
                     "V541_DEGREE_STATS_TOPOLOGY",
                     f"bond ({a},{b}) references an atom outside "
                     f"[0,{n_atoms})")
        degrees = np.zeros(n_atoms, dtype=np.int64)
        for a, b in pairs:
            degrees[a] += 1
            degrees[b] += 1
        # weighted per-node histogram (bins include 0 — isolated atoms
        # count in the mu_log denominator)
        for deg in degrees.tolist():
            histogram[int(deg)] = histogram.get(int(deg), 0) + 1
        sum_log1p += float(np.log1p(degrees.astype(np.float64)).sum())
        node_total += n_atoms
        pair_total += len(pairs)
        edge_total += 2 * len(pairs)
        unique_topologies.add((n_atoms, tuple(pairs)))

    _require(edge_total > 0, "V541_DEGREE_STATS_NO_EDGES",
             "the whole inner-train population has no G_chem edges — "
             "mu_log (and the PNA amplification scaler) would be "
             "undefined; refusing instead of emitting a degenerate file")
    mu_log = sum_log1p / float(node_total)

    split_sha = sha256_text(str(split))
    positions_sha = positions_sha256(canonical)
    recipe_sha = sha256_json(recipe)
    seed_sha = sha256_text(str(seed))
    payload = {
        "schema": DEGREE_STATS_SCHEMA,
        "split": str(split),
        "split_sha256": split_sha,
        "positions_sha256": positions_sha,
        "graph_recipe_sha256": recipe_sha,
        "seed": int(seed),
        "seed_sha256": seed_sha,
        "mu_log": mu_log,
        "n_positions": len(canonical),
        "n_unique_graphs": len(unique_topologies),
        "weighted_graph_total": len(canonical),
        "weighted_node_total": node_total,
        "weighted_edge_total": edge_total,
        "weighted_pair_total": pair_total,
        "degree_histogram": {str(k): v for k, v in sorted(histogram.items())},
    }
    stats_sha = sha256_json(payload)

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return DegreeStatsResult(
        schema=DEGREE_STATS_SCHEMA,
        split=str(split),
        mu_log=float(mu_log),
        stats_sha256=stats_sha,
        split_sha256=split_sha,
        positions_sha256=positions_sha,
        graph_recipe_sha256=recipe_sha,
        seed=int(seed),
        seed_sha256=seed_sha,
        n_positions=len(canonical),
        n_unique_graphs=len(unique_topologies),
        weighted_graph_total=len(canonical),
        weighted_node_total=int(node_total),
        weighted_edge_total=int(edge_total),
        weighted_pair_total=int(pair_total),
        degree_histogram=dict(histogram),
        elapsed_seconds=time.perf_counter() - t_start,
        max_rss_kb=int(usage.ru_maxrss),
    )


__all__ = [
    "ALLOWED_SPLIT",
    "DEGREE_STATS_SCHEMA",
    "FORBIDDEN_SPLITS",
    "GRAPH_RECIPE_V541_CHEM",
    "DegreeStatsResult",
    "V541DegreeStatsError",
    "chem_in_degrees",
    "compute_degree_stats",
    "qualified_chem_pairs",
]
