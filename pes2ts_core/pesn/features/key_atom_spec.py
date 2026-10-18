"""V5 key-atom feature spec (schema ``pesn_v5_feature_spec_v1``).

Normative sources: ADR-0026 §2.6 and sprint plan §2
(``docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md``).
This module owns:

* re-export of the seeded slot layouts from
  :mod:`pes2ts_core.pesn.contracts.key_atom` (single source of truth — this
  module never redefines them);
* the FROZEN Pauling-proxy single-bond reference table ``r_eq`` (sum of
  Pyykko–Atsumi 2009 single-bond covalent radii) for the sprint element set
  {H, B, C, N, O, F, Si, P, S, Cl, Br, I}; a pair missing from the table gets
  proxy 0.0 + mask 0.0 (plan §2: "缺表置0/mask0");
* Gaussian RBF(16) parameters on the CURRENT distance only
  (centers uniform in [0, 4.5] Å, σ = 0.3 Å);
* train-group-only vocabulary helpers (UNK = index 0) and one-hot helpers
  for the method (8) and solvent (4) task slots;
* the feature-spec JSON writer/loader whose document embeds caller-supplied
  sha256 hashes of the task/native/teacher sources.

Registered interpretation notes (flagged to the orchestrator):
* ``NODE_INV_SLOTS`` in the seeded contract leaves slot 12 unnamed between
  ``hop3`` (11) and ``dist_cur`` (13); per plan §2 the node block contains a
  5-way hop one-hot 0/1/2/3/>3 occupying slots 8..12, so slot 12 is the
  ">3" bin (:data:`HOP_FAR_SLOT`).  A unit test pins the one-hot sum.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Iterable, Mapping

import numpy as np

from pes2ts_core.pesn.contracts.key_atom import (  # noqa: F401  (re-export)
    EDGE_ANCHOR_SLOTS,
    EDGE_DIM,
    EDGE_SLOTS,
    FEATURE_SPEC_SCHEMA,
    NODE_ANCHOR_SLOTS,
    NODE_INV_DIM,
    NODE_INV_SLOTS,
    TASK_DIM,
    TASK_SLOTS,
    KeyAtomBatch,
)

# --- hop one-hot layout -----------------------------------------------------
#: Slot of the hop ">3" bin (unnamed in the seeded contract dict; see module
#: docstring — plan §2 counts a 5-way one-hot at slots 8..12).
HOP_FAR_SLOT: Final[int] = 12
#: The five hop bins (0, 1, 2, 3, >3) in slot order.
HOP_SLOTS: Final[tuple[int, ...]] = (
    NODE_INV_SLOTS["hop0"],
    NODE_INV_SLOTS["hop1"],
    NODE_INV_SLOTS["hop2"],
    NODE_INV_SLOTS["hop3"],
    HOP_FAR_SLOT,
)
#: Hop distance above which the ">3" bin is used.
HOP_FAR_THRESHOLD: Final[int] = 3

# --- fixed scales (plan §2 node/edge/task tables) ---------------------------
DIST_SCALE: Final[float] = 5.0          # ||x_i - c_A|| / 5
DISP_SCALE: Final[float] = 1.0          # ||(x-c_A)-(x0-c0_A)|| / 1
CHARGE_SCALE: Final[float] = 4.0        # formal charge / 4
DEGREE_SCALE: Final[float] = 6.0        # covalent degree / 6
BOND_SCALE: Final[float] = 3.0          # bond order / 3
R0_SCALE: Final[float] = 4.5            # r0 / 4.5
DELTA_D_CLIP: Final[float] = 3.0        # clip(r - r0, -3, 3)
MULT_SCALE: Final[float] = 4.0          # (mult - 1) / 4
D0_SCALE: Final[float] = 0.08           # d0 / 0.08
SCALAR_CLIP: Final[float] = 5.0         # continuous scalars -> [-5, 5]

#: Angular degeneracy threshold: below this norm the radial angle is set to
#: 0 and the next slot (``anchor_valid``) marks the missing value (plan §2).
ANGLE_NORM_EPS: Final[float] = 1e-6

# --- spatial graph (ADR-0026 §2.6) -----------------------------------------
SPATIAL_CUTOFF: Final[float] = 4.5      # Å, current geometry only
SPATIAL_MAX_NEIGHBORS: Final[int] = 32
SPATIAL_TIE_TOL: Final[float] = 1e-6    # Å, boundary ties ALL kept

# --- Gaussian RBF on the current distance (plan §2 edge table) -------------
N_RBF: Final[int] = 16
RBF_CUTOFF: Final[float] = 4.5
RBF_SIGMA: Final[float] = 0.3

# --- Pauling-proxy table ----------------------------------------------------
#: Version tag of the frozen reference table.
PAULING_PROXY_TABLE_VERSION: Final[str] = "pyykko_atsumi2009_single_bond"
#: Single-bond covalent radii (Å) from Pyykko, P.; Atsumi, M. *Chem. Eur. J.*
#: **2009**, *15*, 186–199, Table 1 (molecular single-bond covalent radii
#: r_s(1), elements 1–118) — the sprint element set only.  ``r_eq(i, j)`` is
#: the SUM of the two single-bond radii.  This is a frozen geometric
#: descriptor, NOT a quantum bond order or topological truth (plan §2).
PYKKKO_ATSUMI_2009_SINGLE_BOND_RADII: Final[dict[int, float]] = {
    1: 0.32,   # H
    5: 0.73,   # B
    6: 0.69,   # C (single bond / sp3 entry of the PA2009 table)
    7: 0.65,   # N
    8: 0.63,   # O
    9: 0.64,   # F
    14: 1.11,  # Si
    15: 1.06,  # P
    16: 1.02,  # S
    17: 0.99,  # Cl
    35: 1.14,  # Br
    53: 1.33,  # I
}

# --- vocabulary contract (plan §2: "任务方法编码仅 train 建词表") -----------
#: Reserved UNK index; task one-hot slot 2 (method) / slot 10 (solvent) is
#: the UNK position, i.e. position 0 of the 8 / 4 way one-hot.
UNK_INDEX: Final[int] = 0
METHOD_VOCAB_SIZE: Final[int] = 8   # slots TASK_SLOTS['method0'] .. +7
SOLVENT_VOCAB_SIZE: Final[int] = 4  # slots TASK_SLOTS['solvent0'] .. +3
#: Known (non-UNK) capacity per vocab.
METHOD_MAX_KNOWN: Final[int] = METHOD_VOCAB_SIZE - 1
SOLVENT_MAX_KNOWN: Final[int] = SOLVENT_VOCAB_SIZE - 1


def rbf_centers() -> np.ndarray:
    """The 16 frozen Gaussian centers, uniform in ``[0, 4.5]`` Å (float64)."""
    return np.linspace(0.0, RBF_CUTOFF, N_RBF, dtype=np.float64)


def gaussian_rbf(distances: np.ndarray) -> np.ndarray:
    """Gaussian expansion of CURRENT distances into 16 invariant features.

    ``exp(-0.5 * ((r - c_k) / 0.3)^2)`` with the frozen centers.  Applied to
    the current geometry only; r0 enters the edge vector via ``r0_scaled`` /
    ``delta_d``, never through this basis (plan §2).
    """
    r = np.asarray(distances, dtype=np.float64)
    centers = rbf_centers()
    return np.exp(-0.5 * ((r[..., None] - centers) / RBF_SIGMA) ** 2)


def pair_r_eq(z_i: int, z_j: int) -> float | None:
    """Frozen single-bond reference distance ``r_eq`` (Å) for an element pair.

    Returns ``None`` when either element is outside the frozen table — the
    caller then stores proxy 0.0 + mask 0.0.
    """
    a = PYKKKO_ATSUMI_2009_SINGLE_BOND_RADII.get(int(z_i))
    b = PYKKKO_ATSUMI_2009_SINGLE_BOND_RADII.get(int(z_j))
    if a is None or b is None:
        return None
    return float(a + b)


def pauling_proxy(r: float, z_i: int, z_j: int) -> tuple[float, float]:
    """``(proxy, mask)`` Pauling-proxy edge descriptor (plan §2 edge table).

    ``proxy = exp(clip((r_eq - r) / 0.30, -5, ln 3)) / 3``; missing pair in
    the frozen table → ``(0.0, 0.0)``.
    """
    r_eq = pair_r_eq(z_i, z_j)
    if r_eq is None:
        return 0.0, 0.0
    value = float(np.exp(np.clip((r_eq - float(r)) / 0.30, -5.0, float(np.log(3.0))))) / 3.0
    return value, 1.0


def stable_id_hash(value: str) -> int:
    """Deterministic non-negative int64 hash of a task id string.

    Used for method/solvent vocabulary keys; only the first 8 sha256 bytes
    are consumed and the sign bit is cleared so the value round-trips through
    int64 npz storage.
    """
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def build_vocab(
    ids: Iterable[int], *, max_known: int | None = None,
) -> dict[int, int]:
    """Train-groups-only vocabulary: ``{hash: idx}`` with UNK = 0.

    Known ids are assigned indices ``1, 2, ...`` in ascending hash order
    (deterministic); index 0 is RESERVED for UNK and never assigned to a
    known id — an unseen id at one-hot time maps to position 0, which is
    task slot ``method0`` / ``solvent0`` respectively (contract comment
    "UNK -> slot 2" / "UNK -> slot 10").
    """
    unique = sorted({int(v) for v in ids})
    if max_known is not None and len(unique) > max_known:
        raise ValueError(
            f"vocab overflow: {len(unique)} known ids > capacity {max_known} "
            f"(UNK reserves one slot)")
    return {h: i + 1 for i, h in enumerate(unique)}


def vocab_index(vocab: Mapping[int, int], id_hash: int) -> int:
    """One-hot position for ``id_hash``; unseen → UNK (0)."""
    return int(vocab.get(int(id_hash), UNK_INDEX))


def method_one_hot(index: int) -> np.ndarray:
    """8-way float64 one-hot; index 0 = UNK at the first position."""
    if not 0 <= int(index) < METHOD_VOCAB_SIZE:
        raise ValueError(f"method index {index} outside [0, {METHOD_VOCAB_SIZE})")
    v = np.zeros(METHOD_VOCAB_SIZE, dtype=np.float64)
    v[int(index)] = 1.0
    return v


def solvent_one_hot(index: int) -> np.ndarray:
    """4-way float64 one-hot; index 0 = UNK at the first position."""
    if not 0 <= int(index) < SOLVENT_VOCAB_SIZE:
        raise ValueError(f"solvent index {index} outside [0, {SOLVENT_VOCAB_SIZE})")
    v = np.zeros(SOLVENT_VOCAB_SIZE, dtype=np.float64)
    v[int(index)] = 1.0
    return v


# --- feature-spec JSON ------------------------------------------------------
def build_feature_spec_document(
    source_sha256: Mapping[str, str],
) -> dict:
    """The versioned feature-spec document (schema ``pesn_v5_feature_spec_v1``).

    ``source_sha256`` maps source labels (e.g. ``task`` / ``native`` /
    ``teacher``) to caller-computed sha256 hex digests; this module never
    touches the raw sources itself.
    """
    return {
        "schema_version": FEATURE_SPEC_SCHEMA,
        "node_inv_dim": NODE_INV_DIM,
        "edge_dim": EDGE_DIM,
        "task_dim": TASK_DIM,
        "node_inv_slots": {k: int(v) for k, v in NODE_INV_SLOTS.items()},
        "hop_far_slot": HOP_FAR_SLOT,
        "edge_slots": {k: int(v) for k, v in EDGE_SLOTS.items()},
        "task_slots": {k: int(v) for k, v in TASK_SLOTS.items()},
        "node_anchor_slots": [int(s) for s in NODE_ANCHOR_SLOTS],
        "edge_anchor_slots": [int(s) for s in EDGE_ANCHOR_SLOTS],
        "rbf": {
            "n": N_RBF,
            "center_min": 0.0,
            "center_max": RBF_CUTOFF,
            "sigma": RBF_SIGMA,
            "applies_to": "current_distance_only",
        },
        "spatial_graph": {
            "cutoff_angstrom": SPATIAL_CUTOFF,
            "max_neighbors": SPATIAL_MAX_NEIGHBORS,
            "tie_tolerance_angstrom": SPATIAL_TIE_TOL,
        },
        "scales": {
            "dist": DIST_SCALE,
            "disp": DISP_SCALE,
            "charge": CHARGE_SCALE,
            "degree": DEGREE_SCALE,
            "bond": BOND_SCALE,
            "r0": R0_SCALE,
            "delta_d_clip": DELTA_D_CLIP,
            "multiplicity": MULT_SCALE,
            "d0": D0_SCALE,
            "scalar_clip": SCALAR_CLIP,
        },
        "pauling_proxy": {
            "table": PAULING_PROXY_TABLE_VERSION,
            "radii_angstrom": {
                str(z): r for z, r in sorted(PYKKKO_ATSUMI_2009_SINGLE_BOND_RADII.items())
            },
            "formula": "exp(clip((r_eq - r)/0.30, -5, ln3))/3; missing pair -> 0 + mask 0",
        },
        "vocab": {
            "unk_index": UNK_INDEX,
            "method_size": METHOD_VOCAB_SIZE,
            "solvent_size": SOLVENT_VOCAB_SIZE,
            "scope": "train_groups_only",
        },
        "source_sha256": {str(k): str(v) for k, v in sorted(source_sha256.items())},
    }


def write_feature_spec(path: str | Path, source_sha256: Mapping[str, str]) -> dict:
    """Write the feature-spec JSON atomically-ish; returns the document."""
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = build_feature_spec_document(source_sha256)
    target.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def load_feature_spec(path: str | Path) -> dict:
    """Load and VALIDATE the feature spec against the current constants.

    A mismatch (schema, dims, slot tables, RBF/spatial params) raises — the
    spec on disk is a frozen contract, not documentation.
    """
    import json

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = build_feature_spec_document(doc.get("source_sha256", {}))
    for key in (
        "schema_version", "node_inv_dim", "edge_dim", "task_dim",
        "node_inv_slots", "hop_far_slot", "edge_slots", "task_slots",
        "node_anchor_slots", "edge_anchor_slots", "rbf", "spatial_graph",
        "scales", "pauling_proxy", "vocab",
    ):
        if doc.get(key) != expected[key]:
            raise ValueError(
                f"feature spec drift at key {key!r}: file={doc.get(key)!r} "
                f"code={expected[key]!r}")
    return doc


__all__ = [
    "EDGE_ANCHOR_SLOTS", "EDGE_DIM", "EDGE_SLOTS", "FEATURE_SPEC_SCHEMA",
    "NODE_ANCHOR_SLOTS", "NODE_INV_DIM", "NODE_INV_SLOTS", "TASK_DIM",
    "TASK_SLOTS", "KeyAtomBatch",
    "HOP_FAR_SLOT", "HOP_SLOTS", "HOP_FAR_THRESHOLD",
    "DIST_SCALE", "DISP_SCALE", "CHARGE_SCALE", "DEGREE_SCALE",
    "BOND_SCALE", "R0_SCALE", "DELTA_D_CLIP", "MULT_SCALE", "D0_SCALE",
    "SCALAR_CLIP", "ANGLE_NORM_EPS",
    "SPATIAL_CUTOFF", "SPATIAL_MAX_NEIGHBORS", "SPATIAL_TIE_TOL",
    "N_RBF", "RBF_CUTOFF", "RBF_SIGMA",
    "PAULING_PROXY_TABLE_VERSION", "PYKKKO_ATSUMI_2009_SINGLE_BOND_RADII",
    "UNK_INDEX", "METHOD_VOCAB_SIZE", "SOLVENT_VOCAB_SIZE",
    "METHOD_MAX_KNOWN", "SOLVENT_MAX_KNOWN",
    "rbf_centers", "gaussian_rbf", "pair_r_eq", "pauling_proxy",
    "stable_id_hash", "build_vocab", "vocab_index",
    "method_one_hot", "solvent_one_hot",
    "build_feature_spec_document", "write_feature_spec", "load_feature_spec",
]
