"""V5.2-R D0.1 — molecular-graph automorphisms for label symmetry (numpy-only).

Implements exhaustive graph-automorphism enumeration for the small molecular
graphs of the PD-1 six-table delivery (``reaction_intent.csv`` ``g_src``/
``g_tgt`` payloads: node element labels + edge bond orders, ≤ ~25 heavy atoms
+ hydrogens) and the symmetry-minimized geometric comparators used by the
D0.1 label/data audit (V5.2-R plan §3.1/§3.2):

* :func:`enumerate_automorphisms` — color partition refinement (initial
  classes by element + incident-bond-order multiset, iterated refinement by
  the multiset of ``(bond order, neighbor color)``) followed by a pruned
  backtracking search over the refined cells.  ALL automorphisms are
  enumerated (identity included).  A hard cap
  (:data:`AUTOMORPHISM_CAP`, 100 000) converts a would-be combinatorial
  explosion into a typed :class:`GraphSymmetryError` instead of a hang;
* :func:`k_automorphism_permutations` — the DISTINCT permutations induced on
  a K-atom subset by the automorphisms that map the K set ONTO itself
  (deduplicated; identity always present);
* :func:`min_sym_rmsd` — ``min_π RMSD_K(X, π(X_ref))`` with the FROZEN
  proper-rotation Kabsch of :func:`kabsch_rmsd_v52` (imported, never
  copied);
* :func:`sym_min_cv_deviation` — ``min_π max_pairs |d(X) − d(π(X_ref))|``
  over CV coordinate pairs in the native frame (no alignment; same PD-9
  verified-frame convention as the D0 topology proxy).

Permutation convention (used consistently everywhere): a permutation ``p``
is a tuple with ``p[v] = image of atom v``; applying it to a coordinate
matrix is the gather ``X[list(p)]``.  Minima are invariant under the
``π ↔ π⁻¹`` convention because the enumerated set is the full group.

Read-only, deterministic, numpy-only — ``pes2ts_core/pesn`` must never
import rdkit/sklearn (enforced by test policy; rdkit cross-checks live in
``tests/`` only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import kabsch_rmsd_v52

#: Hard cap on enumerated automorphisms per graph (D0.1 pre-registration:
#: molecules are small; hitting the cap means a bug or a pathological graph,
#: and must fail loudly rather than hang).
AUTOMORPHISM_CAP: Final[int] = 100_000

#: Bond orders are compared/hashed after rounding to this many decimals so
#: float payloads (1.0 vs 1.0000000001) do not fragment color classes.
_BOND_ORDER_DECIMALS: Final[int] = 6


class GraphSymmetryError(RuntimeError):
    """Typed refusal of the graph-symmetry layer."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise GraphSymmetryError(code, detail)


def _round_order(order: float) -> float:
    return round(float(order), _BOND_ORDER_DECIMALS)


@dataclass(frozen=True)
class MolecularGraph:
    """Element-labelled, bond-order-labelled undirected molecular graph.

    ``atoms``: atomic number per row (row space = the shared atom-map order
    of the curve, g1 invariant).  ``bonds``: sorted ``(i, j, order)`` with
    ``i < j`` (aromatic 1.5 allowed).
    """

    atoms: tuple[int, ...]
    bonds: tuple[tuple[int, int, float], ...]

    def __post_init__(self) -> None:
        n = len(self.atoms)
        norm: list[tuple[int, int, float]] = []
        for entry in self.bonds:
            i, j, order = int(entry[0]), int(entry[1]), float(entry[2])
            _require(i != j, "GRAPH_SELFLOOP", f"bond ({i},{j}) is a self loop")
            if i > j:
                i, j = j, i
            _require(0 <= i and j < n, "GRAPH_BOND_RANGE",
                     f"bond ({i},{j}) outside 0..{n - 1}")
            norm.append((i, j, _round_order(order)))
        object.__setattr__(self, "atoms", tuple(int(a) for a in self.atoms))
        object.__setattr__(self, "bonds", tuple(sorted(set(norm))))

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @classmethod
    def from_intent_payload(
        cls,
        atoms: Iterable[int],
        bonds: Iterable[Sequence[float]],
    ) -> "MolecularGraph":
        """Build from an intent ``g_src``/``g_tgt``-style payload."""
        return cls(
            atoms=tuple(int(a) for a in atoms),
            bonds=tuple((int(b[0]), int(b[1]), float(b[2])) for b in bonds))

    @classmethod
    def from_graph_payload(cls, payload: Mapping[str, object]) -> "MolecularGraph":
        """Build from the E1 dict encoding ``{"atoms": [...], "bonds": [...]}``."""
        raw_atoms = payload.get("atoms", ())
        raw_bonds = payload.get("bonds", ())
        return cls.from_intent_payload(raw_atoms, raw_bonds)  # type: ignore[arg-type]


def _adjacency(graph: MolecularGraph) -> np.ndarray:
    """``[N, N]`` float matrix of rounded bond orders (0 = no bond)."""
    n = graph.n_atoms
    adj = np.zeros((n, n), dtype=np.float64)
    for i, j, order in graph.bonds:
        adj[i, j] = order
        adj[j, i] = order
    return adj


def _initial_colors(graph: MolecularGraph, adj: np.ndarray) -> list[int]:
    """Element + sorted multiset of incident bond orders."""
    sigs: list[tuple] = []
    for v in range(graph.n_atoms):
        orders = tuple(sorted(float(o) for o in adj[v] if o != 0.0))
        sigs.append((int(graph.atoms[v]), orders))
    return _compress(sigs)


def _compress(sigs: Sequence[tuple]) -> list[int]:
    """Deterministic signature → dense integer color."""
    table: dict[tuple, int] = {}
    out: list[int] = []
    for s in sigs:
        if s not in table:
            table[s] = len(table)
        out.append(table[s])
    return out


def refine_partition(
    graph: MolecularGraph,
    adj: np.ndarray | None = None,
) -> list[int]:
    """Stable equitable partition (1-dim Weisfeiler-Leman refinement).

    Returns the final integer color per node.  Refinement is monotone (each
    round only splits cells because the new signature embeds the old color),
    so a stable cell COUNT implies a stable partition.
    """
    if adj is None:
        adj = _adjacency(graph)
    colors = _initial_colors(graph, adj)
    n = graph.n_atoms
    for _ in range(max(1, n)):
        sigs: list[tuple] = []
        for v in range(n):
            neigh = tuple(sorted(
                (float(adj[v, u]), int(colors[u]))
                for u in range(n) if adj[v, u] != 0.0))
            sigs.append((int(colors[v]), neigh))
        new_colors = _compress(sigs)
        if len(set(new_colors)) == len(set(colors)):
            return colors
        colors = new_colors
    return colors


def enumerate_automorphisms(
    graph: MolecularGraph,
    *,
    cap: int = AUTOMORPHISM_CAP,
) -> tuple[tuple[int, ...], ...]:
    """ALL automorphisms of ``graph`` (identity included), deterministically sorted.

    A permutation ``p`` maps atom ``v`` to image ``p[v]``; ``p`` is an
    automorphism iff element labels and bond orders are preserved.  Search:
    refine the vertex partition, then backtrack node-by-node over cells,
    admitting only unused same-color candidates that are edge-consistent
    with every already-assigned node.  Raises :class:`GraphSymmetryError`
    (``AUTO_CAP``) when more than ``cap`` automorphisms are found —
    exhaustive enumeration is pre-registered for ≤ ~25-heavy-atom graphs,
    and the cap converts a hang into a typed failure.
    """
    n = graph.n_atoms
    _require(n > 0, "GRAPH_EMPTY", "graph has no atoms")
    _require(cap > 0, "AUTO_CAP_INVALID", f"cap must be positive, got {cap}")
    adj = _adjacency(graph)
    colors = refine_partition(graph, adj)

    # cells: color -> nodes; iterate colors by (size, min node) for determinism
    by_color: dict[int, list[int]] = {}
    for v in range(n):
        by_color.setdefault(colors[v], []).append(v)
    cell_order = sorted(by_color.values(), key=lambda c: (len(c), min(c)))
    node_order: list[int] = [v for cell in cell_order for v in cell]
    cell_of = {v: ci for ci, cell in enumerate(cell_order) for v in cell}
    pools = [sorted(cell) for cell in cell_order]

    perm = [-1] * n
    used = [False] * n
    results: list[tuple[int, ...]] = []

    def _consistent(v: int, cand: int, depth: int) -> bool:
        for j in range(depth):
            u = node_order[j]
            if adj[v, u] != adj[cand, perm[u]]:
                return False
        return True

    def _backtrack(depth: int) -> None:
        if len(results) > cap:
            raise GraphSymmetryError(
                "AUTO_CAP",
                f"graph exceeded the automorphism cap of {cap}; exhaustive "
                "enumeration is pre-registered only for small molecules")
        if depth == n:
            results.append(tuple(perm))
            return
        v = node_order[depth]
        for cand in pools[cell_of[v]]:
            if used[cand]:
                continue
            if not _consistent(v, cand, depth):
                continue
            perm[v] = cand
            used[cand] = True
            _backtrack(depth + 1)
            used[cand] = False
            perm[v] = -1

    _backtrack(0)
    _require(bool(results) and tuple(range(n)) in set(results),
             "IDENTITY_MISSING", "identity is not among enumerated automorphisms")
    return tuple(sorted(results))


def k_automorphism_permutations(
    graph: MolecularGraph,
    k_rows: Iterable[int],
    *,
    cap: int = AUTOMORPHISM_CAP,
) -> tuple[tuple[int, ...], ...]:
    """Distinct permutations induced on ``k_rows`` by K-set-preserving automorphisms.

    Only automorphisms ``π`` with ``π(K) = K`` (as a SET) contribute; the
    induced permutation on the SORTED K positions is deduplicated, so e.g. a
    methyl group with all three H in K contributes 6 induced permutations
    even if the full molecule has additional off-K symmetry.  The identity
    permutation ``(0, 1, …, K−1)`` is always present (asserted).
    """
    k = tuple(sorted({int(a) for a in k_rows}))
    kn = len(k)
    _require(kn > 0, "K_SET_EMPTY", "k_rows is empty")
    _require(all(0 <= a < graph.n_atoms for a in k), "K_SET_RANGE",
             f"k_rows {k} outside 0..{graph.n_atoms - 1}")
    pos = {a: i for i, a in enumerate(k)}
    out: set[tuple[int, ...]] = set()
    for perm in enumerate_automorphisms(graph, cap=cap):
        images = tuple(perm[a] for a in k)
        if set(images) != set(k):
            continue
        induced = tuple(pos[perm[a]] for a in k)
        out.add(induced)
    result = tuple(sorted(out))
    identity = tuple(range(kn))
    _require(identity in set(result), "K_IDENTITY_MISSING",
             "identity is not among the K-restricted permutations")
    return result


def min_sym_rmsd(
    x_k: np.ndarray,
    x_ref_k: np.ndarray,
    perms: Iterable[Sequence[int]],
) -> float:
    """``min_π kabsch_rmsd_v52(X_k, π(X_ref)_k)`` over the given K permutations.

    ``x_k``/``x_ref_k`` are ``[K, 3]``; each permutation reorders the REF
    rows (gather ``X_ref[list(p)]``).  Alignment is the frozen FROZEN
    proper-rotation Kabsch (``kabsch_rmsd_v52``, reflection excluded), so a
    mirrored conformer can never beat the proper-rotation floor.
    """
    mobile = np.asarray(x_k, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(x_ref_k, dtype=np.float64).reshape(-1, 3)
    _require(mobile.shape == ref.shape, "SYM_RMSD_SHAPE",
             f"mobile {mobile.shape} vs ref {ref.shape}")
    best = float("inf")
    found = False
    for p in perms:
        idx = np.asarray(tuple(int(i) for i in p), dtype=np.int64)
        _require(int(idx.shape[0]) == mobile.shape[0], "SYM_RMSD_PERM_LEN",
                 f"perm length {int(idx.shape[0])} != K={mobile.shape[0]}")
        val = kabsch_rmsd_v52(mobile, ref[idx])
        if val < best:
            best = val
        found = True
    _require(found, "SYM_RMSD_NO_PERMS", "perms is empty")
    return float(best)


def sym_min_cv_deviation(
    frame_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    cv_pairs: Iterable[Sequence[int]],
    perms: Iterable[Sequence[int]],
) -> float:
    """``min_π max_pairs | d(X_pair) − d(π(X_ref)_pair) |`` (native frame).

    ``frame_xyz``/``ref_xyz`` are full ``[N, 3]`` frames in the SAME native
    reference frame (PD-9; no alignment — same convention as the D0
    topology-consistency proxy).  ``perms`` are FULL-molecule permutations
    (typically every automorphism of the target-endpoint graph); pair
    distances of the permuted reference are gathered at the permuted rows.
    """
    frame = np.asarray(frame_xyz, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(ref_xyz, dtype=np.float64).reshape(-1, 3)
    _require(frame.shape == ref.shape, "SYM_CV_SHAPE",
             f"frame {frame.shape} vs ref {ref.shape}")
    pairs = [(int(a), int(b)) for a, b in cv_pairs]
    _require(bool(pairs), "SYM_CV_NO_PAIRS", "cv_pairs is empty")
    n = frame.shape[0]
    _require(all(0 <= a < n and 0 <= b < n for a, b in pairs), "SYM_CV_RANGE",
             f"cv pair outside 0..{n - 1}")
    best = float("inf")
    found = False
    for p in perms:
        idx = np.asarray(tuple(int(i) for i in p), dtype=np.int64)
        _require(int(idx.shape[0]) == n, "SYM_CV_PERM_LEN",
                 f"perm length {int(idx.shape[0])} != N={n}")
        ref_p = ref[idx]
        devs = [
            abs(float(np.linalg.norm(frame[a] - frame[b]))
                - float(np.linalg.norm(ref_p[a] - ref_p[b])))
            for a, b in pairs]
        val = max(devs)
        if val < best:
            best = val
        found = True
    _require(found, "SYM_CV_NO_PERMS", "perms is empty")
    return float(best)


__all__ = [
    "AUTOMORPHISM_CAP",
    "GraphSymmetryError",
    "MolecularGraph",
    "refine_partition",
    "enumerate_automorphisms",
    "k_automorphism_permutations",
    "min_sym_rmsd",
    "sym_min_cv_deviation",
]
