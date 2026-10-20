"""R3 v3 feature spec: F3 superset parameters + level/intent-off masks.

Schema ``pesn_r3_feature_spec_v3``.  Normative sources: campaign yaml
``feature_contract`` (slot_superset F3, f2_content, f3_extra RBF(12),
intent_off_mask PD-6, tolerances PD-2), R3 plan v3 §3.2–§3.4.

Frozen parameters of the v3 spec:

* Gaussian RBF(12) on the CURRENT distance only — centers uniform in
  ``[0, 4.5]`` Å, σ = 0.3 Å (same geometry conventions as the v1 spec, with
  the v3 width 12 mandated by ``f3_extra``);
* spatial graph: cutoff 4.5 Å, ≤32 neighbours per atom, 1e-6 Å boundary-tie
  retention (permutation safety — never truncated by atom id);
* bond-order scale B/3 for the packed/edge scalars;
* d0 fixed at 0.08 Å (no task slot exists in v3 — R3 plan §3.2 Task: 无).

The level masks (:mod:`pes2ts_core.pesn.contracts.key_atom_v3`) are the ONLY
mechanism separating the F0/F1/F2/F3 arms: this module re-exports them and
provides the feature-spec document writer/loader whose validity check is
fail-closed against the code constants (a spec on disk is a frozen
contract, not documentation).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, Mapping

import numpy as np

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: F401  (re-export)
    CV_MAX,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_ABS_DELTA_KEY,
    EDGE_FEATURE_KEYS_V3,
    EDGE_INDEX_KEY,
    EDGE_INPUT_DIM_V3,
    EDGE_IS_CHANGED_KEY,
    EDGE_PACK_DIM_V3,
    EDGE_PACK_SLOTS_V3,
    ENDPOINT_NAMES_KEY,
    FEATURE_LEVELS,
    FEATURE_SPEC_SCHEMA,
    FORWARD_VIEW_KEYS_V3,
    GRAPH_CV_DELTA_KEY,
    GRAPH_SIGNED_INTENT_KEY,
    INTENT_OFF_ALL_V3,
    INTENT_OFF_CLASSES_V3,
    LEVEL_EDGE_KEYS_V3,
    LEVEL_GRAPH_KEYS_V3,
    LEVEL_NODE_KEYS_V3,
    N_RBF_V3,
    NODE_DEGK_KEY,
    NODE_GRAPH_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_IS_KEY_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    PRESERVED_UNORDERED_KEYS_V3,
    RBF_SLOT_SLICE_V3,
    SUPERVISION_KEYS_V3,
    edge_pack_slot_mask_v3,
    intent_off_keys_v3,
)

# --- frozen geometry/scale parameters ---------------------------------------
SPATIAL_CUTOFF: Final[float] = 4.5       # Å, current geometry only
SPATIAL_MAX_NEIGHBORS: Final[int] = 32
SPATIAL_TIE_TOL: Final[float] = 1e-6     # Å, boundary ties ALL kept
BOND_SCALE: Final[float] = 3.0           # bond order / 3
DEGREE_SCALE: Final[float] = 6.0         # undirected degree / 6
SCALAR_CLIP: Final[float] = 5.0          # packed scalars -> [-5, 5]
RBF_SIGMA: Final[float] = 0.3
RBF_CUTOFF: Final[float] = 4.5
AROMATIC_ORDER_TOL: Final[float] = 1e-6  # order ≈ 1.5 marks aromatic bonds


def rbf_centers_v3() -> np.ndarray:
    """The 12 frozen Gaussian centers, uniform in ``[0, 4.5]`` Å (float64)."""
    return np.linspace(0.0, RBF_CUTOFF, N_RBF_V3, dtype=np.float64)


def gaussian_rbf_v3(distances: np.ndarray) -> np.ndarray:
    """Gaussian expansion of CURRENT distances into 12 invariant features.

    ``exp(-0.5 * ((r - c_k) / 0.3)^2)`` over the frozen centers.  Applied to
    the current geometry only — never to a reference/X0 distance (v3 has no
    X0 channel at all: ReactionIntent Contract v1 §5 forbids history).
    """
    r = np.asarray(distances, dtype=np.float64)
    centers = rbf_centers_v3()
    return np.exp(-0.5 * ((r[..., None] - centers) / RBF_SIGMA) ** 2)


def is_aromatic_order(order: float) -> bool:
    """Aromatic bond predicate (order ≈ 1.5 or explicit flag upstream)."""
    return abs(float(order) - 1.5) < AROMATIC_ORDER_TOL


# --- PD-6 intent-off application (numpy level) -------------------------------
def apply_intent_off_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    classes: tuple[str, ...] | list[str] = INTENT_OFF_ALL_V3,
) -> dict[str, np.ndarray]:
    """Return a copy of ``arrays`` with the given PD-6 classes zeroed.

    ``classes=INTENT_OFF_ALL_V3`` is the legal ``F2-intent-off`` ablation arm
    (PD-6 full mask).  Subsets are accepted for leak-detection tests only —
    a partial mask must be observable as a violation by the harness
    ``check_intent_off_mask`` (T1.4c).
    """
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in arrays.items()}
    for key in intent_off_keys_v3(classes):
        if key not in out:
            continue
        value = out[key]
        if isinstance(value, np.ndarray):
            out[key] = np.zeros_like(value)
        else:
            out[key] = value
    return out


# --- feature-level masking (F0/F1/F2 by mask only) ---------------------------
#: Full feature-key universe the level mask may touch (everything else —
#: supervision keys such as ``pair_index``/``y_teacher`` — is untouched).
_FEATURE_KEY_UNIVERSE_V3: frozenset[str] = frozenset(
    {NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
     NODE_IS_KEY_KEY, NODE_DEGK_KEY, NODE_INTENT_ROLE_KEY,
     EDGE_INDEX_KEY, EDGE_ATTR_KEY, "b_src", "b_tgt", "delta_b_intent",
     "abs_delta", "is_changed", "cv_target_delta", "signed_intent"})


def apply_feature_level_arrays(
    arrays: Mapping[str, np.ndarray],
    level: str,
) -> dict[str, np.ndarray]:
    """Zero-mask ``arrays`` down to one feature level (shapes preserved).

    ``node``/``graph`` level keys outside the level are zeroed (the key must
    remain present — the superset shape never changes between arms);
    packed ``edge_attr`` slots are masked via
    :func:`edge_pack_slot_mask_v3`; ``edge_index`` is structural and always
    kept; keys outside the feature universe (supervision) are untouched.
    """
    if level not in FEATURE_LEVELS:
        raise ValueError(f"unknown feature level {level!r}; known {FEATURE_LEVELS}")
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in arrays.items()}
    node_ok = set(LEVEL_NODE_KEYS_V3[level])
    graph_ok = set(LEVEL_GRAPH_KEYS_V3[level])
    edge_ok = set(LEVEL_EDGE_KEYS_V3[level])
    for key, value in out.items():
        if not isinstance(value, np.ndarray):
            continue
        if key not in _FEATURE_KEY_UNIVERSE_V3:
            continue                      # supervision / metadata
        if key in (NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
                   EDGE_INDEX_KEY):
            continue                      # structural, all levels
        if key in node_ok or key in graph_ok or key in edge_ok:
            continue
        out[key] = np.zeros_like(value)
    if EDGE_ATTR_KEY in out and isinstance(out[EDGE_ATTR_KEY], np.ndarray):
        mask = edge_pack_slot_mask_v3(level).astype(out[EDGE_ATTR_KEY].dtype)
        out[EDGE_ATTR_KEY] = out[EDGE_ATTR_KEY] * mask
    return out


# --- feature-spec document ---------------------------------------------------
def build_feature_spec_document_v3(
    source_sha256: Mapping[str, str],
) -> dict:
    """The versioned feature-spec document (schema ``pesn_r3_feature_spec_v3``).

    ``source_sha256`` maps source labels (campaign yaml, intent contract, …)
    to caller-computed sha256 digests; this module never touches the sources.
    """
    return {
        "schema_version": FEATURE_SPEC_SCHEMA,
        "slot_superset": "F3",
        "feature_levels": list(FEATURE_LEVELS),
        "forward_view_keys": list(FORWARD_VIEW_KEYS_V3),
        "supervision_keys": list(SUPERVISION_KEYS_V3),
        "level_node_keys": {lv: list(LEVEL_NODE_KEYS_V3[lv]) for lv in FEATURE_LEVELS},
        "level_edge_keys": {lv: list(LEVEL_EDGE_KEYS_V3[lv]) for lv in FEATURE_LEVELS},
        "level_graph_keys": {lv: list(LEVEL_GRAPH_KEYS_V3[lv]) for lv in FEATURE_LEVELS},
        "edge_pack_slots": {k: int(v) for k, v in EDGE_PACK_SLOTS_V3.items()},
        "edge_pack_dim": EDGE_PACK_DIM_V3,
        "edge_input_dim": EDGE_INPUT_DIM_V3,
        "cv_max": CV_MAX,
        "rbf": {
            "n": N_RBF_V3,
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
            "bond": BOND_SCALE,
            "degree": DEGREE_SCALE,
            "scalar_clip": SCALAR_CLIP,
            "d0_angstrom": 0.08,
        },
        "intent_off_classes": [
            {"class": cls, "keys": list(ks)}
            for cls, ks in INTENT_OFF_CLASSES_V3
        ],
        "preserved_unordered_keys": list(PRESERVED_UNORDERED_KEYS_V3),
        "source_sha256": {str(k): str(v) for k, v in sorted(source_sha256.items())},
    }


def write_feature_spec_v3(path: str | Path, source_sha256: Mapping[str, str]) -> dict:
    """Write the v3 feature-spec JSON; returns the document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = build_feature_spec_document_v3(source_sha256)
    target.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return doc


def load_feature_spec_v3(path: str | Path) -> dict:
    """Load and VALIDATE the v3 feature spec against the code constants.

    A mismatch raises — the spec on disk is a frozen contract (the campaign
    yaml fills ``feature_spec_sha256`` only after this writer runs).
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = build_feature_spec_document_v3(doc.get("source_sha256", {}))
    for key, value in expected.items():
        if doc.get(key) != value:
            raise ValueError(
                f"v3 feature spec drift at key {key!r}: "
                f"file={doc.get(key)!r} code={value!r}")
    return doc


def spec_sha256_v3(source_sha256: Mapping[str, str]) -> str:
    """sha256 of the canonical spec document (for the campaign manifest)."""
    doc = build_feature_spec_document_v3(source_sha256)
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "SPATIAL_CUTOFF", "SPATIAL_MAX_NEIGHBORS", "SPATIAL_TIE_TOL",
    "BOND_SCALE", "DEGREE_SCALE", "SCALAR_CLIP", "RBF_SIGMA", "RBF_CUTOFF",
    "AROMATIC_ORDER_TOL",
    "rbf_centers_v3", "gaussian_rbf_v3", "is_aromatic_order",
    "apply_intent_off_arrays", "apply_feature_level_arrays",
    "build_feature_spec_document_v3", "write_feature_spec_v3",
    "load_feature_spec_v3", "spec_sha256_v3",
    # re-exports (single source of truth: contracts.key_atom_v3)
    "CV_MAX", "EDGE_PACK_DIM_V3", "EDGE_INPUT_DIM_V3", "N_RBF_V3",
    "FEATURE_LEVELS", "FORWARD_VIEW_KEYS_V3", "SUPERVISION_KEYS_V3",
    "INTENT_OFF_ALL_V3", "INTENT_OFF_CLASSES_V3",
    "PRESERVED_UNORDERED_KEYS_V3", "edge_pack_slot_mask_v3",
    "intent_off_keys_v3", "FEATURE_SPEC_SCHEMA",
]
