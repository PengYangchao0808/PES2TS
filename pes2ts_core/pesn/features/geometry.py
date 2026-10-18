"""Geometry numerics: distances, invariant basis expansion, internal coords.

Runtime-safe by construction: every quantity here is a function of the
current geometry only, rotation/translation invariant (pairwise distances,
angles) or explicitly masked, and the covalent-radius constant is a cited
physical table (Cordero et al. 2008) — the predictor keeps its own copy so
the runtime import boundary never reaches into G1.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np

#: Cordero covalent radii (Angstrom), Z = 1..86 — Cordero et al.,
#: Dalton Trans. 2008, 2832-2838 (low-spin transition metals).
COVALENT_RADII: Final[dict[int, float]] = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
    9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60,
    23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32,
    30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16,
    37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47,
    44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15, 57: 2.07,
    58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96,
    65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87, 71: 1.87,
    72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36,
    79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48, 84: 1.40, 85: 1.50,
    86: 1.50,
}
RADII_VERSION: Final[str] = "cordero2008"


def covalent_radius(z: int) -> float:
    radius = COVALENT_RADII.get(int(z))
    if radius is None:
        raise ValueError(f"no covalent radius for Z={z}")
    return radius


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Full pairwise distance matrix of an ``(N, 3)`` geometry."""
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def gaussian_distance_basis(
    distances: np.ndarray, *, cutoff: float, n_basis: int
) -> np.ndarray:
    """Gaussian expansion of distances into *n_basis* invariant features.

    Centers are evenly spaced in ``[0, cutoff]`` with width
    ``cutoff / n_basis``; distances beyond the cutoff still expand (the
    spatial-edge selection is what enforces the cutoff, never the basis).
    """
    centers = np.linspace(0.0, cutoff, n_basis)
    width = cutoff / n_basis
    return np.exp(-0.5 * ((distances[..., None] - centers) / width) ** 2)


def bond_contact_threshold(z_i: int, z_j: int, *, tolerance: float = 0.45) -> float:
    """Cordero contact threshold for a bonded pair (Angstrom)."""
    return covalent_radius(z_i) + covalent_radius(z_j) + tolerance


def evaluate_internal_coordinate(
    coords: np.ndarray, atoms: Sequence[int], kind: str
) -> float:
    """One internal coordinate value from the current geometry only."""
    idx = [int(a) for a in atoms]
    if kind == "distance":
        return float(np.linalg.norm(coords[idx[0]] - coords[idx[1]]))
    if kind == "angle":
        u = coords[idx[0]] - coords[idx[1]]
        v = coords[idx[2]] - coords[idx[1]]
        cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    if kind == "dihedral":
        b0 = coords[idx[0]] - coords[idx[1]]
        b1 = coords[idx[2]] - coords[idx[1]]
        b2 = coords[idx[3]] - coords[idx[2]]
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = float(np.dot(v, w))
        y = float(np.dot(np.cross(b1, v), w))
        return float(np.arctan2(y, x))
    raise ValueError(f"unknown coordinate kind {kind!r}")


def evaluate_coordinate_spec(
    coords: np.ndarray, coordinate_spec
) -> np.ndarray:
    """Evaluate every coordinate of a CoordinateSpec at the current geometry."""
    values = [
        evaluate_internal_coordinate(coords, c.atoms, c.kind)
        for c in coordinate_spec.coords
    ]
    return np.asarray(values, dtype=np.float64)


def scaled_coordinates(
    coords: np.ndarray, coordinate_spec
) -> np.ndarray:
    """``q_tilde = (q - ref_q) / scale_a`` per coordinate (inference-safe)."""
    q = evaluate_coordinate_spec(coords, coordinate_spec)
    ref = np.array([c.ref_q for c in coordinate_spec.coords])
    scale = np.array([c.scale_a for c in coordinate_spec.coords])
    return (q - ref) / scale


__all__ = [
    "COVALENT_RADII",
    "RADII_VERSION",
    "bond_contact_threshold",
    "covalent_radius",
    "evaluate_coordinate_spec",
    "evaluate_internal_coordinate",
    "gaussian_distance_basis",
    "pairwise_distances",
    "scaled_coordinates",
]
