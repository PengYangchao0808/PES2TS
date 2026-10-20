"""R3 v3 feature builders: Reaction Intent features on the FIXED F3 superset.

One sample = one ``(curve_id, node_index, intent_id, d0)`` training row
(contract §4.5).  All build math is float64 numpy; storage/consumption is
float32.  No torch_geometric; batches are plain dicts of tensors
(harness-native, see ``collate_v3``).

Graph contract (R3 plan v3 §3.2–§3.4 + campaign ``feature_contract``):

* graph = ``G_src ∪ G_tgt ∪ spatial(current r < 4.5 Å, ≤32 neighbours,
  1e-6 Å tie retention)``; covalent edges are ALWAYS kept even beyond the
  radius cutoff; each undirected pair is stored once then expanded to BOTH
  directions with identical attributes;
* ``K`` = the intent-changed pairs (``B_tgt != B_src``) plus CV-spec atoms
  (30h plan §3.1); ``degK`` is the RAW incident changed-pair count (the loss
  needs raw ``1/max(1,degK)`` weights, so it is never normalized here);
* ``intent_role`` [N,3] is the multi-label (breaking, forming, spectator)
  projection of incident ``delta_b_intent`` — breaking and forming may be 1
  on the same atom simultaneously (30h plan §3.1);
* the only 3-D input is the CURRENT ``pos``; v3 has no X0/anchor/history
  channel at all (ReactionIntent Contract v1 §5);
* ``endpoint_names`` are display names carried for the relabel gate only —
  no numeric feature reads them (relabel invariance = inputs bit-identical);
* PD-6 intent-off masks the five classes and preserves the undirected
  chemical topology (``apply_intent_off``).

``level`` (F0/F1/F2/F3) zero-masks features in place — shapes are identical
across arms (slot superset discipline; no parameter may differ between arms).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    CV_MAX,
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_FEATURE_KEYS_V3,
    EDGE_INDEX_KEY,
    EDGE_INPUT_DIM_V3,
    EDGE_IS_CHANGED_KEY,
    EDGE_PACK_DIM_V3,
    EDGE_PACK_SLOTS_V3,
    ENDPOINT_NAMES_KEY,
    FORWARD_VIEW_KEYS_V3,
    GRAPH_CV_DELTA_KEY,
    GRAPH_SIGNED_INTENT_KEY,
    INTENT_OFF_ALL_V3,
    KeyAtomSampleV3,
    KeyAtomV3ContractError,
    NODE_DEGK_KEY,
    NODE_GRAPH_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_IS_KEY_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    SUPERVISION_KEYS_V3,
    _as_bonds,
)
from pes2ts_core.pesn.features.key_atom_spec_v3 import (
    BOND_SCALE,
    RBF_SLOT_SLICE_V3,
    SCALAR_CLIP,
    SPATIAL_CUTOFF,
    SPATIAL_MAX_NEIGHBORS,
    SPATIAL_TIE_TOL,
    apply_feature_level_arrays,
    apply_intent_off_arrays,
    gaussian_rbf_v3,
    is_aromatic_order,
)


class KeyAtomGraphV3Error(RuntimeError):
    """Typed refusal of the pure v3 feature builder."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise KeyAtomGraphV3Error(code, detail)


# ---------------------------------------------------------------------------
# Sample builder
# ---------------------------------------------------------------------------
def build_sample_features_v3(
    sample: KeyAtomSampleV3,
    *,
    level: str = "F3",
    intent_off: bool | Sequence[str] = False,
) -> dict[str, np.ndarray | tuple]:
    """Build ONE v3 sample's features (float32/int64 numpy).

    Parameters
    ----------
    sample : current-frame + directed-intent record (no history channels).
    level : feature arm — F0/F1/F2 by mask only (superset shapes preserved).
    intent_off : ``False`` (intent-ON), ``True`` (PD-6 full mask — the legal
        ``F2-intent-off`` ablation), or an explicit subset of PD-6 classes
        (leak-detection tests only).
    """
    z = np.asarray(sample.atomic_numbers, dtype=np.int64).reshape(-1)
    n_atoms = int(z.shape[0])
    _require(n_atoms > 0, "ATOM_COUNT_INVALID", "sample has no atoms")
    _require(bool(np.all((z >= 1) & (z <= 118))),
             "ATOMIC_NUMBER_INVALID", "atomic numbers must lie in 1..118")
    x = np.asarray(sample.xyz, dtype=np.float64).reshape(n_atoms, 3)
    _require(np.all(np.isfinite(x)), "GEOM_NONFINITE",
             "current coordinates contain non-finite values")

    intent = sample.intent
    pair_orders = intent.pair_orders()
    key_rows = np.asarray(intent.key_rows(), dtype=np.int64)
    _require(key_rows.size > 0, "KEY_ROWS_EMPTY", "intent K set is empty")
    key_set = {int(a) for a in key_rows}

    # --- unsigned / signed bond attributes per union pair -------------------
    changed_pairs: list[tuple[int, int]] = []
    for (i, j), (b_src, b_tgt) in sorted(pair_orders.items()):
        if b_src != b_tgt:
            changed_pairs.append((i, j))

    # intent_role: multi-label over ALL incident signed intent edits
    role_breaking = np.zeros(n_atoms, dtype=np.float64)
    role_forming = np.zeros(n_atoms, dtype=np.float64)
    for (i, j), (b_src, b_tgt) in pair_orders.items():
        delta = b_tgt - b_src
        if delta < 0.0:
            role_breaking[i] = 1.0
            role_breaking[j] = 1.0
        elif delta > 0.0:
            role_forming[i] = 1.0
            role_forming[j] = 1.0
    role_spectator = (
        1.0 - np.maximum(role_breaking, role_forming)).clip(min=0.0)
    intent_role = np.stack(
        [role_breaking, role_forming, role_spectator], axis=1)

    # degK: RAW incident count of intent-changed pairs (both-endpoint-K pairs
    # are the K-pair set below; the node-level degK counts all changed pairs
    # incident on the atom so the loss weight 1/max(1,degK) stays defined).
    deg_k = np.zeros(n_atoms, dtype=np.float64)
    for (i, j) in changed_pairs:
        deg_k[i] += 1.0
        deg_k[j] += 1.0

    is_key = np.zeros(n_atoms, dtype=np.float64)
    is_key[key_rows] = 1.0

    # --- edge set: G_src ∪ G_tgt covalent ∪ current-radius ------------------
    dist = x[:, None, :] - x[None, :, :]
    r_mat = np.sqrt((dist * dist).sum(axis=-1))

    pairs: dict[tuple[int, int], dict[str, float]] = {}

    def _pair(i: int, j: int) -> dict[str, float]:
        a, b = (i, j) if i < j else (j, i)
        return pairs.setdefault((a, b), {
            "covalent": 0.0, "aromatic": 0.0, "is_spatial": 0.0,
            "b_src": 0.0, "b_tgt": 0.0,
        })

    for (i, j), (b_src, b_tgt) in pair_orders.items():
        rec = _pair(i, j)
        if b_src > 0.0 or b_tgt > 0.0:
            rec["covalent"] = 1.0
        if is_aromatic_order(b_src) or is_aromatic_order(b_tgt):
            rec["aromatic"] = 1.0
        rec["b_src"] = b_src / 3.0
        rec["b_tgt"] = b_tgt / 3.0

    # spatial graph: per-atom ≤32 nearest within cutoff, 1e-6 Å tie retention
    for i in range(n_atoms):
        cand = np.concatenate((np.arange(0, i), np.arange(i + 1, n_atoms)))
        d = r_mat[i, cand]
        keep = d < SPATIAL_CUTOFF
        cand, d = cand[keep], d[keep]
        if cand.size == 0:
            continue
        order_idx = np.argsort(d, kind="stable")
        n_take = min(SPATIAL_MAX_NEIGHBORS, cand.size)
        thresh = d[order_idx[n_take - 1]]
        selected = d <= thresh + SPATIAL_TIE_TOL
        for j in cand[selected]:
            rec = _pair(i, int(j))
            rec["is_spatial"] = 0.0 if rec["covalent"] else 1.0

    pair_keys = sorted(pairs)
    n_edges_u = len(pair_keys)
    edge_index = np.zeros((2, 2 * n_edges_u), dtype=np.int64)
    edge_attr = np.zeros((2 * n_edges_u, EDGE_PACK_DIM_V3), dtype=np.float64)
    for p, (i, j) in enumerate(pair_keys):
        rec = pairs[(i, j)]
        b_src, b_tgt = rec["b_src"], rec["b_tgt"]
        row = np.zeros(EDGE_PACK_DIM_V3, dtype=np.float64)
        r = float(r_mat[i, j])
        row[RBF_SLOT_SLICE_V3] = gaussian_rbf_v3(np.float64(r))
        row[EDGE_PACK_SLOTS_V3["is_spatial"]] = rec["is_spatial"]
        row[EDGE_PACK_SLOTS_V3["covalent_union"]] = rec["covalent"]
        row[EDGE_PACK_SLOTS_V3["aromatic_union"]] = rec["aromatic"]
        row[EDGE_PACK_SLOTS_V3["b_min"]] = min(b_src, b_tgt)
        row[EDGE_PACK_SLOTS_V3["b_max"]] = max(b_src, b_tgt)
        np.clip(row, -SCALAR_CLIP, SCALAR_CLIP, out=row)
        fwd, rev = 2 * p, 2 * p + 1
        edge_index[0, fwd], edge_index[1, fwd] = i, j
        edge_index[0, rev], edge_index[1, rev] = j, i
        edge_attr[fwd] = row
        edge_attr[rev] = row              # identical attrs, both directions

    # edge-aligned intent scalars (same directed ordering as edge_index)
    b_src_col = np.zeros(2 * n_edges_u, dtype=np.float64)
    b_tgt_col = np.zeros(2 * n_edges_u, dtype=np.float64)
    delta_col = np.zeros(2 * n_edges_u, dtype=np.float64)
    for p, (i, j) in enumerate(pair_keys):
        rec = pairs[(i, j)]
        b_src, b_tgt = rec["b_src"], rec["b_tgt"]
        for slot in (2 * p, 2 * p + 1):
            b_src_col[slot] = b_src
            b_tgt_col[slot] = b_tgt
            delta_col[slot] = b_tgt - b_src
    abs_delta = np.abs(delta_col)
    is_changed = (abs_delta > 0.0).astype(np.float64)

    # --- K pairs (changed pairs with BOTH endpoints in K) -------------------
    k_pairs = [(i, j) for (i, j) in changed_pairs
               if i in key_set and j in key_set]
    pair_index = np.zeros((2, len(k_pairs)), dtype=np.int64)
    pair_edit = np.zeros(len(k_pairs), dtype=np.float64)
    for p, (i, j) in enumerate(k_pairs):
        pair_index[0, p], pair_index[1, p] = i, j
        b_src, b_tgt = pair_orders[(i, j)]
        pair_edit[p] = b_tgt - b_src

    # --- graph-level directed conditioning ----------------------------------
    cv_delta = np.zeros(CV_MAX, dtype=np.float64)
    n_cv = min(len(intent.cv_deltas), CV_MAX)
    cv_delta[:n_cv] = intent.cv_deltas[:n_cv]
    signed_intent = np.float64(intent.signed_intent_sign())

    sample_dict: dict[str, np.ndarray | tuple] = {
        NODE_Z_KEY: z,
        NODE_POS_KEY: x.astype(np.float32),
        NODE_GRAPH_KEY: np.zeros(n_atoms, dtype=np.int64),
        NODE_KEY_SET_KEY: key_rows,
        NODE_IS_KEY_KEY: is_key,
        NODE_DEGK_KEY: deg_k.astype(np.float32),
        NODE_INTENT_ROLE_KEY: intent_role.astype(np.float32),
        EDGE_INDEX_KEY: edge_index,
        EDGE_ATTR_KEY: edge_attr.astype(np.float32),
        EDGE_B_SRC_KEY: b_src_col.astype(np.float32),
        EDGE_B_TGT_KEY: b_tgt_col.astype(np.float32),
        EDGE_DELTA_B_KEY: delta_col.astype(np.float32),
        EDGE_ABS_DELTA_KEY: abs_delta.astype(np.float32),
        EDGE_IS_CHANGED_KEY: is_changed.astype(np.float32),
        GRAPH_CV_DELTA_KEY: cv_delta.astype(np.float32).reshape(1, CV_MAX),
        GRAPH_SIGNED_INTENT_KEY: signed_intent.astype(np.float32).reshape(1),
        ENDPOINT_NAMES_KEY: tuple(str(n) for n in sample.endpoint_names),
        # --- supervision (never model inputs) ------------------------------
        "pair_index": pair_index,
        "pair_edit": pair_edit.astype(np.float32),
        "y_teacher": (np.asarray(sample.y_teacher, dtype=np.float32)
                      if sample.y_teacher is not None else None),
        "frame_weight": np.float32(sample.frame_weight),
        "d0": np.float64(sample.d0),
        "sample_id": sample.sample_id or f"{sample.curve_id}:{sample.node_index}"
                     f":{intent.intent_id}:{sample.d0}",
        "curve_id": sample.curve_id,
        "intent_id": intent.intent_id,
        "label_status": sample.label_status,
    }

    if intent_off:
        classes = INTENT_OFF_ALL_V3 if intent_off is True else tuple(intent_off)
        arrays = {k: v for k, v in sample_dict.items()
                  if isinstance(v, np.ndarray)}
        masked = apply_intent_off_arrays(arrays, classes=classes)
        sample_dict.update(masked)
    level_arrays = apply_feature_level_arrays(
        {k: v for k, v in sample_dict.items()
         if isinstance(v, np.ndarray)}, level)
    sample_dict.update(level_arrays)
    return sample_dict


def apply_intent_off(
    sample: Mapping[str, np.ndarray | tuple],
    *,
    classes: Sequence[str] = INTENT_OFF_ALL_V3,
) -> dict[str, np.ndarray | tuple]:
    """PD-6 intent-off on a BUILT sample dict (shapes preserved)."""
    arrays = {k: v for k, v in sample.items() if isinstance(v, np.ndarray)}
    masked = apply_intent_off_arrays(arrays, classes=tuple(classes))
    out = dict(sample)
    out.update(masked)
    return out


def forward_view_sample(
    sample: Mapping[str, np.ndarray | tuple],
) -> dict[str, np.ndarray | tuple]:
    """Forward-visible view of one built sample (drops supervision keys)."""
    return {k: sample[k] for k in FORWARD_VIEW_KEYS_V3 if k in sample}


# ---------------------------------------------------------------------------
# Collation (dict batches, global rows)
# ---------------------------------------------------------------------------
def collate_v3(
    samples: Sequence[Mapping[str, np.ndarray | tuple]],
) -> dict[str, Any]:
    """Assemble per-sample dicts into one dict batch (int64 offsets).

    Node rows are concatenated along the batch-global node axis; edge and
    pair indices carry global node offsets; graph-level tensors are stacked
    per sample (one graph per sample).  ``endpoint_names`` becomes a tuple
    of per-graph 2-tuples (relabel-gate metadata).
    """
    _require(len(samples) > 0, "EMPTY_BATCH", "collate needs ≥1 sample")
    n_nodes = [int(np.asarray(s[NODE_POS_KEY]).shape[0]) for s in samples]
    n_edges = [int(np.asarray(s[EDGE_INDEX_KEY]).shape[1]) for s in samples]
    n_keys = [int(np.asarray(s[NODE_KEY_SET_KEY]).shape[0]) for s in samples]
    n_pairs = [int(np.asarray(s["pair_index"]).shape[1]) for s in samples]

    node_offsets = np.concatenate(([0], np.cumsum(n_nodes))).astype(np.int64)

    def _cat(key: str, axis: int = 0) -> np.ndarray:
        return np.concatenate(
            [np.asarray(s[key]) for s in samples], axis=axis)

    edge_index = np.concatenate(
        [np.asarray(s[EDGE_INDEX_KEY], dtype=np.int64) + off
         for s, off in zip(samples, node_offsets)], axis=1)
    key_index = np.concatenate(
        [np.asarray(s[NODE_KEY_SET_KEY], dtype=np.int64) + off
         for s, off in zip(samples, node_offsets)])
    pair_index = np.concatenate(
        [np.asarray(s["pair_index"], dtype=np.int64) + off
         for s, off in zip(samples, node_offsets)], axis=1)

    graph_index = np.concatenate(
        [np.asarray(s[NODE_GRAPH_KEY], dtype=np.int64) + g
         for g, s in enumerate(samples)])

    batch: dict[str, Any] = {
        NODE_Z_KEY: _cat(NODE_Z_KEY).astype(np.int64),
        NODE_POS_KEY: _cat(NODE_POS_KEY).astype(np.float32),
        NODE_GRAPH_KEY: graph_index,
        NODE_KEY_SET_KEY: key_index,
        NODE_IS_KEY_KEY: _cat(NODE_IS_KEY_KEY).astype(np.float32),
        NODE_DEGK_KEY: _cat(NODE_DEGK_KEY).astype(np.float32),
        NODE_INTENT_ROLE_KEY: _cat(NODE_INTENT_ROLE_KEY).astype(np.float32),
        EDGE_INDEX_KEY: edge_index.astype(np.int64),
        EDGE_ATTR_KEY: _cat(EDGE_ATTR_KEY).astype(np.float32),
        EDGE_B_SRC_KEY: _cat(EDGE_B_SRC_KEY).astype(np.float32),
        EDGE_B_TGT_KEY: _cat(EDGE_B_TGT_KEY).astype(np.float32),
        EDGE_DELTA_B_KEY: _cat(EDGE_DELTA_B_KEY).astype(np.float32),
        EDGE_ABS_DELTA_KEY: _cat(EDGE_ABS_DELTA_KEY).astype(np.float32),
        EDGE_IS_CHANGED_KEY: _cat(EDGE_IS_CHANGED_KEY).astype(np.float32),
        GRAPH_CV_DELTA_KEY: np.concatenate(
            [np.asarray(s[GRAPH_CV_DELTA_KEY]).reshape(1, CV_MAX)
             for s in samples], axis=0).astype(np.float32),
        GRAPH_SIGNED_INTENT_KEY: np.concatenate(
            [np.asarray(s[GRAPH_SIGNED_INTENT_KEY]).reshape(1)
             for s in samples], axis=0).astype(np.float32),
        ENDPOINT_NAMES_KEY: tuple(
            tuple(s[ENDPOINT_NAMES_KEY]) for s in samples),
        # --- supervision ---
        "pair_index": pair_index.astype(np.int64),
        "pair_edit": _cat("pair_edit").astype(np.float32),
        "frame_weight": np.array(
            [float(s["frame_weight"]) for s in samples], dtype=np.float32),
        "d0": np.array([float(s["d0"]) for s in samples], dtype=np.float64),
        "sample_ids": tuple(str(s["sample_id"]) for s in samples),
        "curve_ids": tuple(str(s["curve_id"]) for s in samples),
        "intent_ids": tuple(str(s["intent_id"]) for s in samples),
        "ptr": node_offsets,
        "n_nodes": np.array(n_nodes, dtype=np.int64),
        "n_edges": np.array(n_edges, dtype=np.int64),
        "n_keys": np.array(n_keys, dtype=np.int64),
        "n_pairs": np.array(n_pairs, dtype=np.int64),
    }
    if all(s.get("y_teacher") is not None for s in samples):
        batch["y_teacher"] = _cat("y_teacher").astype(np.float32)
    return batch


def to_torch_batch(
    batch: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Convert a collated numpy batch to torch tensors (int64/float32)."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(value)
            out[key] = torch.as_tensor(arr, device=device)
        else:
            out[key] = value
    return out


__all__ = [
    "KeyAtomGraphV3Error",
    "build_sample_features_v3",
    "apply_intent_off",
    "forward_view_sample",
    "collate_v3",
    "to_torch_batch",
]
