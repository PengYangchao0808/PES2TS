"""V5.2-R D2: chirality/stereo signs + K-atom combination geometry blocks.

Pure-numpy companion to ``pes2ts_core.pesn.features.geometry`` for the D2
input-sufficiency probes (plan ``PES2TS_V52R_输入充分性与化学终点识别优化方案_v1``
§4.2/§4.5).  It adds the mirror-SENSITIVE quantities that pairwise distances
cannot express, plus vectorized K-atom combination builders whose semantics
are IDENTICAL to :func:`evaluate_internal_coordinate`:

* :func:`signed_tetrahedral_volume` — signed volume of an atom quadruple
  ``(a,b,c,d)``: ``(r_b−r_a)·((r_c−r_a)×(r_d−r_a))`` (Å³).  Proper
  rotation/translation invariant; flips sign under an improper mirror —
  the classic tetrahedral chirality indicator.
* :func:`improper_dihedral` — signed torsion of a quadruple computed with the
  SAME numerics as ``evaluate_internal_coordinate(kind="dihedral")`` on
  ``(a,b,c,d)``, returned in ``(−π, π]``.  Rotation invariant; sign flips
  under a mirror (wrapped).
* :func:`cis_trans_flag` — planarity sign of a quadruple: ``sign(x)`` where
  ``x = v·w`` is the in-plane projection of the same torsion decomposition
  (+1 cis-like / |τ| < 90°, −1 trans-like, 0 perpendicular or degenerate).
  Rotation AND mirror invariant (planarity itself is not chiral).

All builders operate on the intent's K key atoms in SORTED (lexicographic)
combination order — the same order as ``np.triu_indices(K, 1)`` for pairs —
so a K=8 row yields C(8,2)=28 distances, C(8,3)=56 angles, C(8,4)=70
dihedrals/quads; every vector zero-pads to the dataset-global width via
:func:`pad_vector` (non-finite values are the caller's to sanitize).

Conventions are frozen in ``outputs/pesn/v52r/d2/probe_spec.json``; this
module is numpy-only (no dataset, no torch, no I/O).
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from math import comb
from typing import Final, Sequence

import numpy as np

#: dataset-wide max |K| (D1 tabular = C(8,2) = 28 pair distances).
MAX_K: Final[int] = 8
#: padded block widths for MAX_K (see plan §4: K up to 8 → 56 angles / 70 dihedrals).
MAX_K_PAIRS: Final[int] = comb(MAX_K, 2)        # 28
MAX_K_ANGLES: Final[int] = comb(MAX_K, 3)       # 56
MAX_K_DIHEDRALS: Final[int] = comb(MAX_K, 4)    # 70
MAX_K_QUADS: Final[int] = comb(MAX_K, 4)        # 70


@lru_cache(maxsize=None)
def _combinations(k: int, r: int) -> np.ndarray:
    arr = np.asarray(list(itertools.combinations(range(int(k), ), int(r))),
                     dtype=np.int64).reshape(-1, int(r))
    arr.setflags(write=False)
    return arr


def sorted_combinations(k: int, r: int) -> np.ndarray:
    """``[C(k,r), r]`` lexicographic index tuples of ``range(k)`` taken r at a time.

    Order matches ``np.triu_indices(k, k=1)`` for r=2 (row-major i<j), so
    distance/angle/dihedral/quad blocks stay aligned across probes.
    """
    k, r = int(k), int(r)
    if k < 0 or r < 0:
        raise ValueError(f"negative arguments: k={k} r={r}")
    return _combinations(k, r)


def k_angle_triples(k: int) -> np.ndarray:
    """``[C(k,3), 3]`` atom index triples (i,j,k) — j is the angle CENTER."""
    return sorted_combinations(k, 3)


def k_dihedral_quads(k: int) -> np.ndarray:
    """``[C(k,4), 4]`` atom index quadruples (i,j,k,l) — used for dihedrals,
    signed volumes, improper dihedrals and cis/trans flags alike."""
    return sorted_combinations(k, 4)


def pad_vector(values: Sequence[float], width: int, *, name: str = "values") -> np.ndarray:
    """Zero-pad (or reject) a 1-D block to the global padded width."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    width = int(width)
    if v.size > width:
        raise ValueError(f"{name}: {v.size} values > padded width {width}")
    out = np.zeros(width, dtype=np.float64)
    out[: v.size] = v
    return out


def _as_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must be [N,3], got {arr.shape}")
    return arr


def k_pair_values(coords: np.ndarray) -> np.ndarray:
    """Pairwise distances among ALL rows of ``coords`` in i<j order.

    Numerically identical expression to ``v52r_d1_probe.k_pair_distances``
    (the D2 script asserts bitwise equality of the padded P0/P4 blocks).
    """
    pts = _as_coords(coords)
    if pts.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    iu = np.triu_indices(pts.shape[0], k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    return np.sqrt((diff * diff).sum(axis=1))


def k_angle_values(coords: np.ndarray) -> np.ndarray:
    """All C(K,3) angles — same math as ``evaluate_internal_coordinate(kind='angle')``."""
    pts = _as_coords(coords)
    tri = k_angle_triples(pts.shape[0])
    if tri.size == 0:
        return np.zeros(0, dtype=np.float64)
    u = pts[tri[:, 0]] - pts[tri[:, 1]]
    v = pts[tri[:, 2]] - pts[tri[:, 1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        cosine = (u * v).sum(axis=-1) / denom
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _torsion_xy(pts: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane projection ``x`` and out-of-plane ``y`` of the (i,j,k,l) torsion.

    Mirrors ``evaluate_internal_coordinate(kind='dihedral')`` vectorized over
    quadruples: b0 = r_i − r_j, b1 = r_k − r_j (normalized), b2 = r_l − r_k;
    v = b0 − (b0·b1)b1, w = b2 − (b2·b1)b1; x = v·w, y = (b1×v)·w.
    """
    if quads.size == 0:
        return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64))
    p = pts[np.asarray(quads, dtype=np.int64).reshape(-1, 4)]
    b0 = p[:, 0] - p[:, 1]
    b1 = p[:, 2] - p[:, 1]
    b2 = p[:, 3] - p[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.linalg.norm(b1, axis=-1)
        b1n = b1 / norm[:, None]
        v = b0 - (b0 * b1n).sum(axis=-1)[:, None] * b1n
        w = b2 - (b2 * b1n).sum(axis=-1)[:, None] * b1n
        x = (v * w).sum(axis=-1)
        y = (np.cross(b1n, v) * w).sum(axis=-1)
    return x, y


def improper_dihedral(coords: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Signed torsion (−π, π] of each quadruple — mirror-SENSITIVE.

    Identical numerics to ``evaluate_internal_coordinate(kind='dihedral')``
    on the quadruple's (i,j,k,l); a proper rotation preserves the value, an
    improper mirror maps τ → −τ (wrapped).
    """
    pts = _as_coords(coords)
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    if quads.size == 0:
        return np.zeros(0, dtype=np.float64)
    x, y = _torsion_xy(pts, quads)
    return np.arctan2(y, x)


def k_dihedral_values(coords: np.ndarray) -> np.ndarray:
    """All C(K,4) torsions — same math as ``evaluate_internal_coordinate(kind='dihedral')``."""
    pts = _as_coords(coords)
    return improper_dihedral(pts, k_dihedral_quads(pts.shape[0]))


def signed_tetrahedral_volume(coords: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Signed tetrahedral volume of each quadruple (mirror-SENSITIVE, Å³).

    ``V = (r_b − r_a)·((r_c − r_a)×(r_d − r_a))`` for quadruple (a,b,c,d):
    invariant under proper rotation/translation, ``V → −V`` under a mirror.
    """
    pts = _as_coords(coords)
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    if quads.size == 0:
        return np.zeros(0, dtype=np.float64)
    p = pts[quads]
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 0]
    v3 = p[:, 3] - p[:, 0]
    return np.einsum("ij,ij->i", v1, np.cross(v2, v3))


def k_signed_volumes(coords: np.ndarray) -> np.ndarray:
    """All C(K,4) signed tetrahedral volumes."""
    pts = _as_coords(coords)
    return signed_tetrahedral_volume(pts, k_dihedral_quads(pts.shape[0]))


def cis_trans_flag(coords: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Planarity sign of each quadruple: ``sign(x)`` of the torsion's x = v·w.

    +1 = cis-like (|τ| < 90°, a and d project to the same side of the
    b–c axis plane), −1 = trans-like, 0 = exactly perpendicular or
    degenerate (non-finite) geometry.  Rotation AND mirror invariant —
    unlike the signed volume/improper dihedral, planarity is not chiral.
    """
    pts = _as_coords(coords)
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    if quads.size == 0:
        return np.zeros(0, dtype=np.float64)
    x, _y = _torsion_xy(pts, quads)
    return np.where(np.isfinite(x), np.sign(x), 0.0).astype(np.float64)


def k_cis_trans_flags(coords: np.ndarray) -> np.ndarray:
    """All C(K,4) cis/trans planarity flags."""
    pts = _as_coords(coords)
    return cis_trans_flag(pts, k_dihedral_quads(pts.shape[0]))


def k_feature_blocks(frame_xyz: np.ndarray, key_rows: Sequence[int],
                     *, max_k: int = MAX_K) -> dict[str, np.ndarray]:
    """Padded current-frame feature blocks over the intent's K key atoms.

    Returns fixed-width float64 vectors (zero-padded by K size, widths for
    ``max_k`` atoms):

    * ``pairs``      — C(max_k,2) K-pair distances (D1-identical numerics);
    * ``angles``     — C(max_k,3) angles;
    * ``dihedrals``  — C(max_k,4) signed torsions;
    * ``volumes``    — C(max_k,4) signed tetrahedral volumes;
    * ``impropers``  — C(max_k,4) improper dihedrals (= torsions);
    * ``cis_trans``  — C(max_k,4) planarity signs.

    The function touches ONLY ``frame_xyz`` + ``key_rows`` (no goal frame —
    the D2 script asserts this via its goal-free frame guard).
    """
    coords = _as_coords(frame_xyz)
    keys = np.asarray(key_rows, dtype=np.int64).reshape(-1)
    if keys.size > int(max_k):
        raise ValueError(f"|K|={keys.size} > max_k={max_k}")
    pts = coords[keys]
    quads = k_dihedral_quads(int(keys.size))
    return {
        "pairs": pad_vector(k_pair_values(pts), comb(int(max_k), 2), name="pairs"),
        "angles": pad_vector(k_angle_values(pts), comb(int(max_k), 3), name="angles"),
        "dihedrals": pad_vector(k_dihedral_values(pts), comb(int(max_k), 4),
                                name="dihedrals"),
        "volumes": pad_vector(signed_tetrahedral_volume(pts, quads),
                              comb(int(max_k), 4), name="volumes"),
        "impropers": pad_vector(improper_dihedral(pts, quads),
                                comb(int(max_k), 4), name="impropers"),
        "cis_trans": pad_vector(cis_trans_flag(pts, quads),
                                comb(int(max_k), 4), name="cis_trans"),
    }


__all__ = [
    "MAX_K",
    "MAX_K_PAIRS",
    "MAX_K_ANGLES",
    "MAX_K_DIHEDRALS",
    "MAX_K_QUADS",
    "sorted_combinations",
    "k_angle_triples",
    "k_dihedral_quads",
    "pad_vector",
    "k_pair_values",
    "k_angle_values",
    "k_dihedral_values",
    "signed_tetrahedral_volume",
    "k_signed_volumes",
    "improper_dihedral",
    "cis_trans_flag",
    "k_cis_trans_flags",
    "k_feature_blocks",
]
