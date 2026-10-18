"""Pure builders + collation for V5 key-atom features (ADR-0026 §2.6,
sprint plan §2 / §5.2).

One sample = one (reaction, branch, d0, start-row) record.  All build math is
float64 numpy; storage/consumption is float32.  No torch_geometric — the
PyG-free deviation is REGISTERED in ADR-0026 §2.6: same semantics/shapes via
plain tensors + ``ptr`` offsets (see :class:`~pes2ts_core.pesn.contracts.
key_atom.KeyAtomBatch`).

Graph contract (plan §2, ADR-0026 §2.6):

* ``graph = G_start ∪ G_other ∪ K ∪ {current r < 4.5 Å}``;
* covalent and K edges ALWAYS kept even beyond the radius cutoff;
* spatial edges: ≤32 nearest neighbours per atom, boundary distance ties
  within 1e-6 Å ALL kept (never truncated by atom id — permutation safety);
* each undirected pair stored once, then expanded to BOTH directions with
  identical ``edge_attr``; ``r_ij``/``r0_ij`` flip sign per direction;
* K = the CHANGED pair set of the branch — every unordered ``i < j`` pair
  with ``B_start != B_other`` (Lane A ``KeyAtomTask.K_pairs`` semantics;
  ADR-0026 §1.2 "K 的编辑类型") whose endpoints are BOTH key atoms;
  ``pair_edit = B_other − B_start`` per pair; ``degK_i`` = number of K pairs
  incident on atom ``i``.

``pos`` is whole-molecule-centroid centered; ``pos0`` (causal X0, already
aligned into the X_t frame by the caller) is centered with the SAME
centroid.  Vector features (``r_ij``, ``r0_ij``, ``anchor_vec``) are stored
in raw Å — the 1 Å scale is applied model-side.

The builder consumes the duck-typed surface documented at
:func:`build_sample_features`; it accepts both Lane A's concrete
``KeyAtomTask`` (``key_atom_rows``, Mapping endpoints ``{"bonds": [...]}``)
and test FakeTasks (``key_rows``, object endpoints with ``.bonds``).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.contracts.key_atom import (
    EDGE_ANCHOR_SLOTS,
    EDGE_DIM,
    EDGE_SLOTS,
    KeyAtomBatch,
    NODE_ANCHOR_SLOTS,
    NODE_INV_DIM,
    NODE_INV_SLOTS,
    TASK_DIM,
    TASK_SLOTS,
)
from pes2ts_core.pesn.features.key_atom_spec import (
    ANGLE_NORM_EPS,
    BOND_SCALE,
    CHARGE_SCALE,
    DEGREE_SCALE,
    DELTA_D_CLIP,
    DIST_SCALE,
    D0_SCALE,
    DISP_SCALE,
    HOP_FAR_THRESHOLD,
    HOP_SLOTS,
    METHOD_VOCAB_SIZE,
    MULT_SCALE,
    N_RBF,
    R0_SCALE,
    SCALAR_CLIP,
    SOLVENT_VOCAB_SIZE,
    SPATIAL_CUTOFF,
    SPATIAL_MAX_NEIGHBORS,
    SPATIAL_TIE_TOL,
    gaussian_rbf,
    method_one_hot,
    pauling_proxy,
    solvent_one_hot,
    vocab_index,
)


class KeyAtomGraphError(RuntimeError):
    """Typed refusal of the pure feature builder."""


def _require(cond: bool, code: str, msg: str) -> None:
    if not cond:
        raise KeyAtomGraphError(f"{code}: {msg}")


# ---------------------------------------------------------------------------
# Duck-typed task surface adapters
# ---------------------------------------------------------------------------
# The builder consumes from a task object:
#   task.atomic_numbers  : (N,) int          — IRC row order
#   task.key_rows OR task.key_atom_rows : sequence[int] — key-atom rows (A)
#   task.endpoints       : (graph_start, graph_other) for this branch; each
#                          graph is either a Mapping {"bonds": [[i, j, order],
#                          ...]} (Lane A KeyAtomTask) or an object exposing
#                          .bonds of (i, j, order) triples / bond records
#                          with (.i, .j, .bond_order) attrs; aromatic bonds
#                          carry an explicit truthy ``aromatic`` or order
#                          ≈ 1.5; optional .formal_charges (N,)
#   task.charge, task.multiplicity   — for task19
#   task.isotope_mass_diffs — optional (N,) mass diffs (placeholder: absent
#                             -> isotope_diff slot stays 0; documented)

_AROMATIC_ORDER_TOL = 1e-6


def _bonds_of(graph: Any) -> Iterable[Any]:
    """Bond iterable of either a Mapping endpoint ({"bonds": [...]}) or an
    object endpoint (``.bonds``)."""
    if isinstance(graph, Mapping):
        return graph.get("bonds") or ()
    return getattr(graph, "bonds", ())


def _formal_charges_of(graph: Any) -> np.ndarray | None:
    if isinstance(graph, Mapping):
        return graph.get("formal_charges")
    return getattr(graph, "formal_charges", None)


def _key_rows_of(task: Any) -> tuple[int, ...]:
    rows = getattr(task, "key_rows", None)
    if rows is None:
        rows = getattr(task, "key_atom_rows", ())
    return tuple(int(a) for a in rows)


def _bond_triple(bond: Any) -> tuple[int, int, float, bool]:
    """Normalize one bond record to ``(i, j, order, aromatic)``."""
    if hasattr(bond, "i") and hasattr(bond, "j"):
        i, j = int(bond.i), int(bond.j)
        order = float(getattr(bond, "bond_order", 0.0))
        aromatic = bool(getattr(bond, "aromatic", abs(order - 1.5) < _AROMATIC_ORDER_TOL))
    else:
        seq = tuple(bond)
        i, j = int(seq[0]), int(seq[1])
        order = float(seq[2])
        aromatic = bool(seq[3]) if len(seq) > 3 else abs(order - 1.5) < _AROMATIC_ORDER_TOL
    if i > j:
        i, j = j, i
    return i, j, order, aromatic


class _EndpointView:
    """Immutable per-endpoint lookups (bond orders, degrees, aromaticity,
    formal charges) over the SHARED atom row indexing."""

    def __init__(self, graph: Any, n_atoms: int) -> None:
        self._order: dict[tuple[int, int], float] = {}
        self._aromatic_bond: set[tuple[int, int]] = set()
        for bond in _bonds_of(graph):
            i, j, order, aromatic = _bond_triple(bond)
            _require(0 <= i < n_atoms and 0 <= j < n_atoms and i != j,
                     "BOND_INDEX_INVALID", f"bond ({i},{j}) outside 0..{n_atoms-1}")
            self._order[(i, j)] = order
            if aromatic:
                self._aromatic_bond.add((i, j))
        self._degree = np.zeros(n_atoms, dtype=np.float64)
        for (i, j) in self._order:
            self._degree[i] += 1.0
            self._degree[j] += 1.0
        self._aromatic_atom = np.zeros(n_atoms, dtype=np.float64)
        for (i, j) in self._aromatic_bond:
            self._aromatic_atom[i] = 1.0
            self._aromatic_atom[j] = 1.0
        charges = _formal_charges_of(graph)
        if charges is None:
            self._charges = np.zeros(n_atoms, dtype=np.float64)
        else:
            self._charges = np.asarray(charges, dtype=np.float64).reshape(-1)
            _require(self._charges.shape == (n_atoms,),
                     "CHARGE_SHAPE_INVALID", "formal_charges must be (N,)")
        self.n_atoms = n_atoms

    def order(self, i: int, j: int) -> float:
        a, b = (i, j) if i < j else (j, i)
        return self._order.get((a, b), 0.0)

    def is_covalent(self, i: int, j: int) -> bool:
        return self.order(i, j) > 0.0

    def aromatic_bond(self, i: int, j: int) -> bool:
        a, b = (i, j) if i < j else (j, i)
        return (a, b) in self._aromatic_bond

    def degree(self) -> np.ndarray:
        return self._degree

    def aromatic_atoms(self) -> np.ndarray:
        return self._aromatic_atom

    def charges(self) -> np.ndarray:
        return self._charges


def _endpoint_views(task: Any, branch: int, n_atoms: int) -> tuple[_EndpointView, _EndpointView]:
    endpoints = getattr(task, "endpoints", None)
    _require(endpoints is not None and len(tuple(endpoints)) == 2,
             "ENDPOINT_MISSING", "task.endpoints must expose (start, other) "
             "for the branch (endpoints[branch], endpoints[1-branch])")
    endpoints = tuple(endpoints)
    _require(branch in (0, 1), "BRANCH_INVALID", f"branch {branch} not in {{0,1}}")
    return (_EndpointView(endpoints[branch], n_atoms),
            _EndpointView(endpoints[1 - branch], n_atoms))


def _hop_from_key(key_rows: np.ndarray, bonds: Iterable[tuple[int, int]],
                  n_atoms: int) -> np.ndarray:
    """Hop distance from the key set A in the START covalent graph.

    Returns per-atom hop 0/1/2/3 with 4 meaning ">3 / unreachable" (bin index
    into ``HOP_SLOTS``).
    """
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for (i, j) in bonds:
        adj[i].append(j)
        adj[j].append(i)
    hop = np.full(n_atoms, 4, dtype=np.int64)  # 4 => ">3" bin
    frontier = [int(a) for a in key_rows]
    for a in frontier:
        hop[a] = 0
    level = 0
    # expand only levels 1..HOP_FAR_THRESHOLD; anything farther stays ">3"
    while frontier and level < HOP_FAR_THRESHOLD:
        level += 1
        nxt: list[int] = []
        for i in frontier:
            for j in adj[i]:
                if hop[j] > level:
                    hop[j] = level
                    nxt.append(j)
        frontier = nxt
    return hop


# ---------------------------------------------------------------------------
# Sample builder
# ---------------------------------------------------------------------------
def build_sample_features(
    task: Any,
    pos_t: np.ndarray,
    pos0_aligned: np.ndarray,
    branch: int,
    d0: float,
    anchor_mode: bool,
    method_hash: int,
    solvent_hash: int,
    rho_known: bool,
    solvent_known: bool,
) -> dict[str, np.ndarray]:
    """Build ONE sample's features (anchor-ON geometry; see ``anchor_mode``).

    Parameters
    ----------
    task : duck-typed KeyAtomTask (surface documented at the top of this
        module).  ``task.endpoints[branch]`` is the START endpoint for this
        branch, ``endpoints[1-branch]`` the OTHER one.
    pos_t : (N, 3) float64 — causal current frame X_t (native rows).
    pos0_aligned : (N, 3) float64 — branch start frame X0 ALREADY aligned
        into the X_t frame by the caller (Lane A weighted Kabsch with the
        branch reference block).
    anchor_mode : False applies the ADR-0026 §2.6 / plan §5.2 fixed slot
        mask at the numpy level (zero NODE_ANCHOR_SLOTS, EDGE_ANCHOR_SLOTS,
        ``r0_ij``, ``anchor_vec``, ``pos0``; shapes preserved).  The offline
        dataset always stores anchor-ON and masks at LOAD time — this flag
        exists for direct/eager use and tests only.

    Returns
    -------
    dict of numpy arrays (float32 features, int64 indices) — see the key
    list in the module docstring of :mod:`key_atom_graph` consumers; edge
    and key indices are SAMPLE-LOCAL (collation adds offsets).
    """
    z = np.asarray(getattr(task, "atomic_numbers", ()), dtype=np.int64).reshape(-1)
    n_atoms = int(z.shape[0])
    _require(n_atoms > 0, "ATOM_COUNT_INVALID", "task has no atoms")
    _require(bool(np.all((z >= 1) & (z <= 118))),
             "ATOMIC_NUMBER_INVALID", "atomic numbers must lie in 1..118")

    key_rows = np.asarray(sorted(set(_key_rows_of(task))), dtype=np.int64)
    _require(key_rows.size > 0, "KEY_ROWS_EMPTY", "task key rows are empty")
    _require(bool(np.all((key_rows >= 0) & (key_rows < n_atoms))),
             "KEY_ROWS_INVALID", "key_rows outside 0..N-1")

    x = np.asarray(pos_t, dtype=np.float64).reshape(n_atoms, 3)
    x0 = np.asarray(pos0_aligned, dtype=np.float64).reshape(n_atoms, 3)
    _require(np.all(np.isfinite(x)) and np.all(np.isfinite(x0)),
             "GEOM_NONFINITE", "positions contain non-finite values")

    start_view, other_view = _endpoint_views(task, branch, n_atoms)

    # --- centering: pos by whole-molecule centroid; pos0 with the SAME one
    centroid = x.mean(axis=0)
    xc = x - centroid
    x0c = x0 - centroid
    c_a = xc[key_rows].mean(axis=0)          # key centroid, current frame
    c0_a = x0c[key_rows].mean(axis=0)        # key centroid, X0 frame

    # --- node invariants ---------------------------------------------------
    h = np.zeros((n_atoms, NODE_INV_DIM), dtype=np.float64)
    h[:, NODE_INV_SLOTS["charge_start"]] = start_view.charges() / CHARGE_SCALE
    h[:, NODE_INV_SLOTS["charge_other"]] = other_view.charges() / CHARGE_SCALE
    h[:, NODE_INV_SLOTS["deg_start"]] = start_view.degree() / DEGREE_SCALE
    h[:, NODE_INV_SLOTS["deg_other"]] = other_view.degree() / DEGREE_SCALE
    h[:, NODE_INV_SLOTS["aromatic_start"]] = start_view.aromatic_atoms()
    h[:, NODE_INV_SLOTS["aromatic_other"]] = other_view.aromatic_atoms()

    is_key = np.zeros(n_atoms, dtype=np.float64)
    is_key[key_rows] = 1.0
    h[:, NODE_INV_SLOTS["is_key"]] = is_key

    # K = CHANGED pairs (B_start != B_other) with both endpoints key atoms
    # (Lane A KeyAtomTask.K_pairs semantics; ADR-0026 §1.2 "K 的编辑类型").
    key_set = {int(a) for a in key_rows}
    k_pairs: list[tuple[int, int]] = []
    for (i, j) in set(start_view._order) | set(other_view._order):
        if start_view.order(i, j) != other_view.order(i, j) \
                and i in key_set and j in key_set:
            k_pairs.append((i, j))
    k_pairs.sort()
    deg_k = np.zeros(n_atoms, dtype=np.float64)
    for (i, j) in k_pairs:
        deg_k[i] += 1.0
        deg_k[j] += 1.0
    h[:, NODE_INV_SLOTS["degK"]] = np.minimum(deg_k, 4.0) / 4.0

    hop = _hop_from_key(key_rows, start_view._order.keys(), n_atoms)
    for bin_i, slot in enumerate(HOP_SLOTS):
        h[hop == bin_i, slot] = 1.0

    radial = xc - c_a                       # x_i - c_A
    radial0 = x0c - c0_a                    # x0_i - c0_A
    h[:, NODE_INV_SLOTS["dist_cur"]] = np.linalg.norm(radial, axis=1) / DIST_SCALE
    h[:, NODE_INV_SLOTS["dist_x0"]] = np.linalg.norm(radial0, axis=1) / DIST_SCALE
    anchor_vec = radial - radial0
    h[:, NODE_INV_SLOTS["disp_anchor"]] = (
        np.linalg.norm(anchor_vec, axis=1) / DISP_SCALE)

    n_cur = np.linalg.norm(radial, axis=1)
    n_x0 = np.linalg.norm(radial0, axis=1)
    valid = (n_cur >= ANGLE_NORM_EPS) & (n_x0 >= ANGLE_NORM_EPS)
    cos = np.zeros(n_atoms, dtype=np.float64)
    np.divide((radial * radial0).sum(axis=1), n_cur * n_x0, out=cos, where=valid)
    angle = np.arccos(np.clip(cos, -1.0, 1.0))
    angle[~valid] = 0.0                     # missing -> 0, NEXT slot marks it
    h[:, NODE_INV_SLOTS["radial_angle"]] = angle
    h[:, NODE_INV_SLOTS["anchor_valid"]] = valid.astype(np.float64)

    h[:, NODE_INV_SLOTS["is_h"]] = (z == 1).astype(np.float64)
    iso_diffs = getattr(task, "isotope_mass_diffs", None)
    if iso_diffs is None:
        # Documented placeholder: isotope slot stays 0 until a task supplies
        # per-atom mass differences (plan §2: 同位素质量差/4).
        iso = np.zeros(n_atoms, dtype=np.float64)
    else:
        iso = np.asarray(iso_diffs, dtype=np.float64).reshape(-1) / 4.0
    h[:, NODE_INV_SLOTS["isotope_diff"]] = iso

    np.clip(h, -SCALAR_CLIP, SCALAR_CLIP, out=h)  # scalars only; vectors below stay raw

    # --- edge set ----------------------------------------------------------
    dist = xc[:, None, :] - xc[None, :, :]
    r_mat = np.sqrt((dist * dist).sum(axis=-1))
    dist0 = x0c[:, None, :] - x0c[None, :, :]
    r0_mat = np.sqrt((dist0 * dist0).sum(axis=-1))

    pairs: dict[tuple[int, int], dict[str, float]] = {}

    def _pair(i: int, j: int) -> dict[str, float]:
        a, b = (i, j) if i < j else (j, i)
        return pairs.setdefault((a, b), {
            "covalent_start": 0.0, "covalent_other": 0.0,
            "is_k": 0.0, "is_radius": 0.0,
            "b_start": 0.0, "b_other": 0.0,
            "aromatic_start": 0.0, "aromatic_other": 0.0,
        })

    # G_start ∪ G_other covalent edges (ALWAYS kept, even r > 4.5 Å)
    for (i, j), order in start_view._order.items():
        rec = _pair(i, j)
        rec["covalent_start"] = 1.0
        rec["b_start"] = order / BOND_SCALE
        if start_view.aromatic_bond(i, j):
            rec["aromatic_start"] = 1.0
    for (i, j), order in other_view._order.items():
        rec = _pair(i, j)
        rec["covalent_other"] = 1.0
        rec["b_other"] = order / BOND_SCALE
        if other_view.aromatic_bond(i, j):
            rec["aromatic_other"] = 1.0

    # K edges: changed pairs over key atoms (ALWAYS kept)
    for (i, j) in k_pairs:
        _pair(i, j)["is_k"] = 1.0

    # Spatial graph: per-atom ≤32 nearest with 1e-6 Å tie retention; an
    # undirected pair is kept if EITHER endpoint selected the other.
    for i in range(n_atoms):
        cand = np.concatenate((np.arange(0, i), np.arange(i + 1, n_atoms)))
        d = r_mat[i, cand]
        keep_mask = d < SPATIAL_CUTOFF
        cand, d = cand[keep_mask], d[keep_mask]
        if cand.size == 0:
            continue
        order_idx = np.argsort(d, kind="stable")
        n_take = min(SPATIAL_MAX_NEIGHBORS, cand.size)
        thresh = d[order_idx[n_take - 1]]
        selected = d <= thresh + SPATIAL_TIE_TOL
        for j in cand[selected]:
            _pair(i, int(j))["is_radius"] = 1.0  # provisional; stripped below

    # A pair present ONLY via the spatial rule keeps is_radius=1; pairs that
    # are covalent or K are marked and drop the radius-only flag.
    pair_keys = sorted(pairs)
    e = len(pair_keys)
    edge_index = np.zeros((2, 2 * e), dtype=np.int64)
    edge_attr = np.zeros((2 * e, EDGE_DIM), dtype=np.float64)
    r_ij = np.zeros((2 * e, 3), dtype=np.float64)
    r0_ij = np.zeros((2 * e, 3), dtype=np.float64)

    for p, (i, j) in enumerate(pair_keys):
        rec = pairs[(i, j)]
        forced = max(rec["covalent_start"], rec["covalent_other"], rec["is_k"])
        rec["is_radius"] = 0.0 if forced else rec["is_radius"]
        r = float(r_mat[i, j])
        r0 = float(r0_mat[i, j])
        row = np.zeros(EDGE_DIM, dtype=np.float64)
        row[EDGE_SLOTS["rbf0"]:EDGE_SLOTS["rbf0"] + N_RBF] = gaussian_rbf(np.float64(r))
        row[EDGE_SLOTS["r0_scaled"]] = r0 / R0_SCALE
        row[EDGE_SLOTS["delta_d"]] = float(np.clip(r - r0, -DELTA_D_CLIP, DELTA_D_CLIP))
        row[EDGE_SLOTS["b_start"]] = rec["b_start"]
        row[EDGE_SLOTS["b_other"]] = rec["b_other"]
        row[EDGE_SLOTS["db"]] = rec["b_other"] - rec["b_start"]
        row[EDGE_SLOTS["covalent_start"]] = rec["covalent_start"]
        row[EDGE_SLOTS["covalent_other"]] = rec["covalent_other"]
        row[EDGE_SLOTS["is_k"]] = rec["is_k"]
        row[EDGE_SLOTS["is_radius"]] = rec["is_radius"]
        row[EDGE_SLOTS["aromatic_start"]] = rec["aromatic_start"]
        row[EDGE_SLOTS["aromatic_other"]] = rec["aromatic_other"]
        proxy, mask = pauling_proxy(r, int(z[i]), int(z[j]))
        row[EDGE_SLOTS["pauling_proxy"]] = proxy
        row[EDGE_SLOTS["proxy_mask"]] = mask
        np.clip(row, -SCALAR_CLIP, SCALAR_CLIP, out=row)

        fwd, rev = 2 * p, 2 * p + 1
        edge_index[0, fwd], edge_index[1, fwd] = i, j
        edge_index[0, rev], edge_index[1, rev] = j, i
        edge_attr[fwd] = row
        edge_attr[rev] = row              # identical attrs, both directions
        r_ij[fwd] = xc[j] - xc[i]
        r_ij[rev] = xc[i] - xc[j]
        r0_ij[fwd] = x0c[j] - x0c[i]
        r0_ij[rev] = x0c[i] - x0c[j]

    # --- key pairs (K): changed pairs, both endpoints key, i < j by row ----
    pair_index = np.zeros((2, len(k_pairs)), dtype=np.int64)
    pair_edit = np.zeros(len(k_pairs), dtype=np.float64)
    for p, (i, j) in enumerate(k_pairs):
        pair_index[0, p], pair_index[1, p] = i, j
        pair_edit[p] = (other_view.order(i, j) - start_view.order(i, j))

    sample: dict[str, np.ndarray] = {
        "z_atomic": z.astype(np.int64),
        "h_node_inv": h.astype(np.float32),
        "pos": xc.astype(np.float32),
        "pos0": x0c.astype(np.float32),
        "anchor_vec": anchor_vec.astype(np.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr.astype(np.float32),
        "r_ij": r_ij.astype(np.float32),
        "r0_ij": r0_ij.astype(np.float32),
        "key_index": key_rows.astype(np.int64),
        "pair_index": pair_index,
        "pair_edit": pair_edit.astype(np.float32),
        "degK": deg_k[key_rows].astype(np.float32),
        "branch": np.int64(branch),
        "d0": np.float64(d0),
        "charge": np.float64(float(getattr(task, "charge", 0))),
        "multiplicity": np.float64(float(getattr(task, "multiplicity", 1))),
        "method_hash": np.int64(method_hash),
        "solvent_hash": np.int64(solvent_hash),
        "rho_known": np.bool_(bool(rho_known)),
        "solvent_known": np.bool_(bool(solvent_known)),
    }
    if not anchor_mode:
        _anchor_off_sample(sample)
    return sample


def _anchor_off_sample(sample: dict[str, np.ndarray]) -> None:
    """Numpy twin of ``KeyAtomBatch.anchor_off_`` (shapes preserved)."""
    h = sample["h_node_inv"]
    h[:, list(NODE_ANCHOR_SLOTS)] = 0.0
    ea = sample["edge_attr"]
    ea[:, list(EDGE_ANCHOR_SLOTS)] = 0.0
    sample["r0_ij"] = np.zeros_like(sample["r0_ij"])
    sample["anchor_vec"] = np.zeros_like(sample["anchor_vec"])
    sample["pos0"] = np.zeros_like(sample["pos0"])


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
#: Scalar keys consumed for task19; present in every sample dict.
_TASK_SCALAR_KEYS = ("branch", "charge", "multiplicity", "d0",
                     "method_hash", "solvent_hash", "rho_known", "solvent_known")
#: Optional supervision keys attached at the DATASET level (never built here).
_SUPERVISION_KEYS = ("y_teacher", "q_weight", "near_ts", "terminal", "capped")


def build_task19(
    sample: Mapping[str, np.ndarray],
    method_vocab: Mapping[int, int],
    solvent_vocab: Mapping[int, int],
) -> np.ndarray:
    """task19 row per plan §2: branch 2, method one-hot 8 (UNK → first slot),
    solvent one-hot 4 (UNK → first slot), charge/4, (mult−1)/4, d0/0.08,
    ρ-known, solvent-known."""
    row = np.zeros(TASK_DIM, dtype=np.float64)
    branch = int(sample["branch"])
    _require(branch in (0, 1), "BRANCH_INVALID", f"branch {branch}")
    row[TASK_SLOTS["branch0"] + branch] = 1.0
    m_idx = vocab_index(method_vocab, int(sample["method_hash"]))
    s_idx = vocab_index(solvent_vocab, int(sample["solvent_hash"]))
    _require(m_idx < METHOD_VOCAB_SIZE, "METHOD_VOCAB_INVALID",
             f"method index {m_idx} exceeds one-hot width")
    _require(s_idx < SOLVENT_VOCAB_SIZE, "SOLVENT_VOCAB_INVALID",
             f"solvent index {s_idx} exceeds one-hot width")
    row[TASK_SLOTS["method0"]:TASK_SLOTS["method0"] + METHOD_VOCAB_SIZE] = \
        method_one_hot(m_idx)
    row[TASK_SLOTS["solvent0"]:TASK_SLOTS["solvent0"] + SOLVENT_VOCAB_SIZE] = \
        solvent_one_hot(s_idx)
    row[TASK_SLOTS["charge"]] = float(sample["charge"]) / CHARGE_SCALE
    row[TASK_SLOTS["multiplicity"]] = (float(sample["multiplicity"]) - 1.0) / MULT_SCALE
    row[TASK_SLOTS["d0"]] = float(sample["d0"]) / D0_SCALE
    row[TASK_SLOTS["rho_known"]] = 1.0 if bool(sample["rho_known"]) else 0.0
    row[TASK_SLOTS["solvent_known"]] = 1.0 if bool(sample["solvent_known"]) else 0.0
    np.clip(row, -SCALAR_CLIP, SCALAR_CLIP, out=row)
    return row


def _cat_edges(samples: Sequence[Mapping[str, np.ndarray]],
               node_offsets: np.ndarray) -> np.ndarray:
    """Concatenate per-sample edge_index with batch-global node offsets."""
    parts = [np.asarray(s["edge_index"], dtype=np.int64).reshape(2, -1) + off
             for s, off in zip(samples, node_offsets)]
    return np.concatenate(parts, axis=1)


def collate(
    samples: Sequence[Mapping[str, np.ndarray]],
    device: torch.device | str,
    anchor_mode: bool,
    method_vocab: Mapping[int, int],
    solvent_vocab: Mapping[int, int],
) -> KeyAtomBatch:
    """Assemble per-sample dicts into one :class:`KeyAtomBatch`.

    Indices are offset to the batch-global node axis (``ptr``), key rows to
    the same global axis with ``key_ptr`` segmentation.  With
    ``anchor_mode=False`` the seeded ``KeyAtomBatch.anchor_off_()`` is applied
    AFTER assembly (shapes preserved, plan §5.2).  ``y_teacher`` and the
    diagnostic scalars are attached only when EVERY sample carries them (they
    originate at the dataset level, never in this builder).
    """
    _require(len(samples) > 0, "EMPTY_BATCH", "collate needs ≥1 sample")
    n_nodes = np.array([int(np.asarray(s["h_node_inv"]).shape[0]) for s in samples],
                       dtype=np.int64)
    n_edges = np.array([int(np.asarray(s["edge_index"]).shape[1]) for s in samples],
                       dtype=np.int64)
    n_keys = np.array([int(np.asarray(s["key_index"]).shape[0]) for s in samples],
                      dtype=np.int64)
    ptr = np.concatenate(([0], np.cumsum(n_nodes))).astype(np.int64)
    node_offsets = ptr[:-1]

    z_atomic = np.concatenate([np.asarray(s["z_atomic"], dtype=np.int64) for s in samples])
    h_node_inv = np.concatenate([np.asarray(s["h_node_inv"], dtype=np.float32)
                                 for s in samples], axis=0)
    pos = np.concatenate([np.asarray(s["pos"], dtype=np.float32) for s in samples], axis=0)
    pos0 = np.concatenate([np.asarray(s["pos0"], dtype=np.float32) for s in samples], axis=0)
    anchor_vec = np.concatenate([np.asarray(s["anchor_vec"], dtype=np.float32)
                                 for s in samples], axis=0)

    edge_index = _cat_edges(samples, node_offsets).astype(np.int64)
    edge_attr = np.concatenate([np.asarray(s["edge_attr"], dtype=np.float32)
                                for s in samples], axis=0)
    r_ij = np.concatenate([np.asarray(s["r_ij"], dtype=np.float32) for s in samples], axis=0)
    r0_ij = np.concatenate([np.asarray(s["r0_ij"], dtype=np.float32) for s in samples], axis=0)

    key_ptr = np.concatenate(([0], np.cumsum(n_keys))).astype(np.int64)
    key_index = (np.concatenate([np.asarray(s["key_index"], dtype=np.int64)
                                 for s in samples])
                 + np.repeat(node_offsets, n_keys)).astype(np.int64)

    pair_parts = [np.asarray(s["pair_index"], dtype=np.int64).reshape(2, -1) + off
                  for s, off in zip(samples, node_offsets)]
    pair_index = np.concatenate(pair_parts, axis=1).astype(np.int64)
    pair_edit = np.concatenate([np.asarray(s["pair_edit"], dtype=np.float32)
                                for s in samples]).astype(np.float32)
    deg_k = np.concatenate([np.asarray(s["degK"], dtype=np.float32) for s in samples])

    task19 = np.stack([build_task19(s, method_vocab, solvent_vocab)
                       for s in samples]).astype(np.float32)

    batch_vec = np.repeat(np.arange(len(samples), dtype=np.int64), n_nodes)

    kwargs: dict[str, Any] = {}
    if all("y_teacher" in s for s in samples):
        kwargs["y_teacher"] = np.concatenate(
            [np.asarray(s["y_teacher"], dtype=np.float32) for s in samples], axis=0)
    for key, dtype in (("q_weight", np.float32), ("near_ts", np.float32),
                       ("terminal", np.bool_), ("capped", np.bool_),
                       ("d0", np.float32), ("branch", np.int64)):
        if all(key in s for s in samples):
            kwargs[key] = np.array([s[key] for s in samples], dtype=dtype)
    kwargs["group_ids"] = tuple(str(s["group_id"]) for s in samples) \
        if all("group_id" in s for s in samples) else ()
    kwargs["reaction_ids"] = tuple(str(s["reaction_id"]) for s in samples) \
        if all("reaction_id" in s for s in samples) else ()

    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), device=device)

    batch = KeyAtomBatch(
        z_atomic=_t(z_atomic),
        h_node_inv=_t(h_node_inv),
        pos=_t(pos),
        pos0=_t(pos0),
        anchor_vec=_t(anchor_vec),
        edge_index=_t(edge_index),
        edge_attr=_t(edge_attr),
        r_ij=_t(r_ij),
        r0_ij=_t(r0_ij),
        task19=_t(task19),
        batch=_t(batch_vec),
        ptr=_t(ptr),
        key_index=_t(key_index),
        key_ptr=_t(key_ptr),
        pair_index=_t(pair_index),
        pair_edit=_t(pair_edit),
        degK=_t(deg_k),
        **{k: _t(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()},
    )
    if not anchor_mode:
        batch.anchor_off_()
    return batch


__all__ = [
    "KeyAtomGraphError",
    "build_sample_features",
    "build_task19",
    "collate",
]
