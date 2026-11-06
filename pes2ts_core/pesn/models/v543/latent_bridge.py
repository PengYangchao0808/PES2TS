"""V5.4.3 ``LatentBridge`` — the parameter-free bridge from ``CoarsePacket`` to
:class:`DecoderContextV1` (plan §4.1 / §5.3 / §5.4 / §5.5 / §5.6 / §5.7 /
§11.1 / §11.2; WP2).

Contract sources, quoted as implemented
---------------------------------------

* **§4.1 component table (:185)** — ``LatentBridge``: "从 Encoder 输出与
  forward 拓扑构造 ``DecoderContextV1``；只在构造时读一次 ``pos`` 的差值",
  role 冻结, **0 trainable**.  This module therefore owns no parameter: the
  single fixed ``8×48`` sketch matrix is a ``persistent=False`` buffer
  (§5.1 principle 3: "Bridge 不引入可训练参数；固定投影矩阵以
  ``persistent=False`` buffer 存放，不进入 optimizer") and every other
  bridge quantity is a deterministic function of the inputs.
* **§5.1 (:269–276)** — read-only/typed/immutable context, explicit units and
  symmetry, parameter freedom, build-once, cacheable, auditable.
* **§5.2 field table (:277–303) + 形状约束 (:304–309)** — every emitted field
  is one of that table's rows with that row's shape/dtype/unit; the prefix-sum
  identities ``K = key_ptr[-1]``, ``P = pair_ptr[-1]``,
  ``T = token_ptr[-1] = G + K + P`` are asserted on every build, and the
  ``P = 0`` / ``Ek = 0`` empty paths are supported (not special-cased away).
* **§5.3 (:355–375)** — the ``key_v`` 8 channels, verbatim:
  ``ch0 = u0/a_ref`` (:361), ``ch1`` breaking (:362), ``ch2`` forming (:363),
  ``ch3`` unchanged covalent (:364), ``ch4`` spatial ``κ(d)=exp(-((d-3)/1)²)``
  (:365), ``ch5`` critical-pair direction (:366), ``ch6`` distance-weighted
  mean with weight ``d/4`` (:367), ``ch7`` key-centroid direction (:368);
  ``a_ref = 0.08``, ``unit_e = (x[src]-x[dst])/(‖…‖+0.1)``; **every channel
  is zeroed when its denominator is 0** (:372) so no NaN can be produced.
* **§5.4 (:377–393)** — the ``pair_s`` 48-dim layout, segment for segment:
  ``0–5`` bond orders/flags, ``6–7`` aromatic/spatial, ``8–19``
  ``RBF12(d_ij)`` reusing ``gaussian_rbf_v3``'s frozen 12 centres,
  ``20–23`` geometry, ``24–31`` endpoint roles, ``32–39`` graph-level
  broadcast, ``40–47`` the fixed ``W_fixed @ (key_s[i] - key_s[j])`` sketch
  with ``W_fixed`` an ``8×48`` ``±1/√48`` matrix (``persistent=False``).
* **§5.5 (:395–405)** — key-key adjacency (changed pairs bidirectional + up
  to 4 nearest keys within 6.0 Å, deduplicated, undirected, stored both
  ways, ``Ek ≤ 2·(P + 4K)``), the 8-dim ``key_edge_attr`` (``d``,
  ``RBF(d;2,3,4,5)``, ``|ΔB|``, ``is_changed``, ``edge_type``), and the token
  layout ``[graph, keys…, pairs…]`` per graph.
* **§5.6 (:406–434)** — the forbidden-input list, the runtime audit
  (:func:`~pes2ts_core.pesn.models.v543.contracts.assert_no_supervision`) and
  the read-only, offline pair-set source audit
  (:func:`audit_pair_sets` — supervision is compared against, never read by
  the model).
* **§5.7 (:436–457)** — the cache key of six identity components and the
  Packet/context boundary (the context carries no label, no ``sample_id``).
* **F2 (:139)** — the changed K-pair set is RECONSTRUCTED from the forward
  edges (``b_src``/``b_tgt``/``delta_b_intent``/``abs_delta``/``is_changed``);
  direction is the **canonical key-slot order** (small → large) and the sign
  is carried by ``ΔB``; ``pair_index``/``pair_edit`` are supervision and are
  refused at the bridge boundary.
* **F3 (:144) / F7** — key-key adjacency and the shared token definition are
  part of v1 so M3/P3/P4 cannot each invent their own.

Design decisions and their plan basis
-------------------------------------

1. **No supervision can reach the Bridge** (§5.6 :409–415, F2 :139).  Any
   ``SUPERVISION_KEYS_V3`` member in the input mapping is a typed
   ``V543_SUPERVISION_LEAK`` refusal BEFORE any compute; every other
   out-of-contract key is ``V543_BRIDGE_FORWARD_VIEW_LEAK``.  The set
   comparison against ``pair_index`` exists only in :func:`audit_pair_sets`,
   which is called by tests/offline tooling with the supervision tensors
   passed SEPARATELY — the model path never receives them (structurally
   visible in the API: ``build_context(packet, forward_view)``).
2. **Canonical direction, no atom arrow** (F2 :139 "forward 侧只有
   ``b_src/b_tgt/delta_b_intent/abs_delta/is_changed``，**没有**原子级箭头";
   §5.4 :379 "``i<j`` 为 key-slot 规范序").  A pair is stored as
   ``(i, j)`` with ``i < j`` in key-slot order; the chemical arrow is NOT
   invented, and ``ΔB`` is carried unchanged from the forward contract (the
   forward view duplicates the same signed edit on both directed edge slots,
   so the identity of the "source" atom is not recoverable — that loss is
   documented, not papered over).  A forward view that carries two DIFFERENT
   ``ΔB`` values for the same unordered pair is refused
   (``V543_BRIDGE_PAIR_DELTA_CONFLICT``) because the sign reference would be
   ambiguous (§5.6 audit: "若不一致……必须阻断").
3. **`pos` is read only as a difference** (§5.6 :411 "``pos`` 以外的任何几何
   量（且 ``pos`` 只用于差值）").  Every geometric quantity below is built
   from ``x[a] - x[b]``: the channels are translation invariant by
   construction and rotation equivariant by construction, and the assertion
   suite re-proves both.
4. **Keys are re-indexed to graph-contiguous canonical order.**  ``key_ptr``
   is "每图 key 数前缀和" (§5.2 :287), which requires every graph's key rows
   to be contiguous, and §5.4 requires the key-slot order to be canonical.
   The bridge therefore applies ONE stable permutation
   (``argsort(key_graph, stable=True)``) to ``key_index``/``u0`` (and hence
   ``key_s``/``key_v``) so both properties hold; within a graph the relative
   order of ``key_index`` is preserved (it is already ascending in the
   builder, ``key_atom_graph_v3.key_rows`` sorts).  No atom order is invented:
   the permutation is derived from ``graph_index`` only.
5. **Groups/pairs are ordered by (graph, key slot)** so that ``pair_ptr`` is a
   genuine per-graph prefix sum and the pair rows of one graph are contiguous.
   ``torch.unique`` over the composite ``lo*K + hi`` key gives the
   deterministic, permutation-stable canonical order (sorted by key slot), and
   a stable ``argsort`` by graph id then makes the layout graph-contiguous.
6. **Empty sets are first-class** (§5.2 :309–310): ``P = 0`` yields
   ``pair_s [0,48]``, ``intent_pair_index [2,0]``, and empty per-graph pair
   ranges; ``K = 1`` with ``P = 0`` yields ``Ek = 0`` (``key_edge_index
   [2,0]``, ``key_edge_attr [0,8]``).  Every helper is written so that the
   empty case takes the same code path (a zero-width gather, not a branch that
   fabricates values).
7. **Deterministic ties.**  The kNN key edges break equal distances by key
   slot (``sorted((distance, slot))``), never by ``topk``'s internal order, so
   the edge set is reproducible bit-for-bit across torch versions/platforms.
8. **Zero-denominator policy** (§5.3 :372): a channel whose denominator is 0
   (no incident edge, empty κ sum, single-key graph) is the ZERO vector,
   produced by :func:`_safe_div` — never ``0/0``.

Input contract
--------------

The Bridge consumes the **intent-on F2/F3 forward view** (``FORWARD_VIEW_KEYS_V3``
restricted to :data:`REQUIRED_FORWARD_KEYS`): ``pos``, ``graph_index``,
``key_index``, ``intent_role``, ``edge_index``, the packed ``edge_attr``
(``EDGE_PACK_DIM_V3`` = 17) and the five edge-aligned intent scalars of
``EDGE_FEATURE_KEYS_V3``, plus the graph-level ``cv_target_delta`` /
``signed_intent``.  The F0/F1 feature arms carry none of the intent scalars, so
they are refused with ``V543_BRIDGE_MISSING_KEY`` — by contract, not by
accident: §5.4 needs ``ΔB``/``is_changed`` to know what a pair IS.

Plan ambiguities surfaced while implementing (reported, NOT silently invented)
-----------------------------------------------------------------------------

1. **``W_fixed`` sign pattern** (§5.4 :391 "固定的 ``8×48`` ±1/√48 矩阵")
   does not name the pattern.  It is DEFINED here by a fixed integer hash
   (:func:`_sketch_sign`), frozen into :data:`BRIDGE_SPEC_SHA256`, and can be
   swapped for "8 个固定索引的 ``key_s`` 差值" (§5.4 :393) as one spec bump.
2. **``RBF(d;2,3,4,5)`` width** (§5.5 :403) does not name σ; the frozen v3
   width ``RBF_SIGMA = 0.3`` of ``gaussian_rbf_v3`` is reused so that the two
   RBF families cannot drift apart.
3. **``cv_delta[0:6]``** (§5.4 :390) asks for six slots while ``CV_MAX = 4``;
   the six slots are the four real increments plus two zero pads.
4. **``b_src``/``b_tgt`` reference frame** (§5.4 :383 vs F2 :139): the forward
   contract writes the SAME ``b_src``/``b_tgt`` (and the same ``ΔB``) onto both
   directed edge slots of a pair, i.e. they are referenced to the intent's
   stored bond-table order and carry NO atom-level arrow.  They are therefore
   carried unchanged onto the canonically ordered pair; a view that disagrees
   between the two slots is refused (decision 2).  Only ``|ΔB|`` is therefore
   comparable with the supervision ``pair_edit`` in the audit.
5. **``endpoint0_logit``** (§5.2 row) is fed from ``CoarsePacket.e0``, which is
   already the raw endpoint logit (§4.3: no double sigmoid).
6. **``edge_type`` semantics** (§5.5 :403 "0=covalent/changed，1=spatial")
   are read as "the changed-pair edges are the covalent/chemical class; the
   radius-neighbour edges are the spatial class".  A changed pair is always
   ``covalent_union == 1`` in the frozen feature builder (at least one bond
   order is non-zero), so the two readings coincide for class 0; a radius edge
   that happens to join two covalently bonded keys stays class 1 because it
   was added BY DISTANCE, not by chemistry.

Forbidden here (plan §11.3): touching a V5.4.2 / V5.3-MP file, reading
``pair_index``/``pair_edit``/``y_teacher``/``q_true``/endpoint labels/
``pos`` as an absolute quantity, ``torch.compile``, and any trainable
parameter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn as nn

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    CV_MAX,
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_IS_CHANGED_KEY,
    EDGE_PACK_DIM_V3,
    EDGE_PACK_SLOTS_V3,
    FORWARD_VIEW_KEYS_V3,
    GRAPH_CV_DELTA_KEY,
    GRAPH_SIGNED_INTENT_KEY,
    NODE_GRAPH_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    N_RBF_V3,
    SUPERVISION_KEYS_V3,
)
from pes2ts_core.pesn.features.key_atom_spec_v3 import RBF_SIGMA, rbf_centers_v3
from pes2ts_core.pesn.models.v542.identity import canonical_sha256
from pes2ts_core.pesn.models.v543.contracts import (
    A_REF_V543,
    DECODER_CONTEXT_FIELDS,
    DECODER_CONTEXT_SCHEMA_V543,
    FORBIDDEN_CONTEXT_ALL_NAMES,
    KEY_EDGE_ATTR_DIM_V543,
    KEY_S_DIM_V543,
    KEY_V_CHANNELS_V543,
    KEY_V_DIM_V543,
    MOVE_CONTEXT_FIELDS,
    PAIR_S_DIM_V543,
    PRECISION_VERSION_V543,
    PROCESS_CONTEXT_FIELDS,
    CoarsePacket,
    DecoderContextV1,
    V543BridgeAuditError,
    V543BridgeError,
    V543ContractError,
    V543InitError,
    V543SupervisionLeakError,
    assert_no_supervision,
    validate_coarse_packet,
    validate_decoder_context,
)

#: Registered Bridge-spec schema (frozen into ``bridge_spec_sha256``).
BRIDGE_SPEC_SCHEMA: Final[str] = "pesn_v543_latent_bridge_spec_v1"
#: Registered Bridge implementation version (a spec change bumps this).
BRIDGE_VERSION: Final[str] = "pesn_v543_latent_bridge_v1"
#: Record schema of the offline pair-set audit (plan §5.6 :435).
PAIR_AUDIT_SCHEMA: Final[str] = "pesn_v543_pair_set_audit_v1"

# --- §5.3 key_v constants (verbatim from the plan) --------------------------
#: Channel names, in channel order 0..7 (plan §5.3 table :361–368).
KEY_V_CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "u0_over_a_ref",      # ch0  u0[k]/a_ref
    "breaking",           # ch1  Σ|ΔB|·unit_e over ΔB<0 incidents
    "forming",            # ch2  Σ|ΔB|·unit_e over ΔB>0 incidents
    "unchanged_covalent",  # ch3 Σ unit_e over ΔB=0 covalent incidents
    "spatial_kappa",      # ch4  Σ κ(d)·unit_e over spatial incidents
    "critical_pair",      # ch5  Σ sign(ΔB)·(x[other]-x[k])/… over changed pairs
    "distance_weighted",  # ch6  Σ (d/4)·unit_e over all incidents
    "key_centroid",       # ch7  (mean_{j∈K,j≠k} x[j] - x[k])/…
)
#: Denominator "+ε" of ``unit_e`` (plan §5.3 :357 "``||…|| + 0.1``").
UNIT_DENOM_EPS: Final[float] = 0.1
#: ``κ(d) = exp(-((d - 3.0)/1.0)²)`` (plan §5.3 :365).
KEY_V_SPATIAL_MU: Final[float] = 3.0
KEY_V_SPATIAL_SIGMA: Final[float] = 1.0
#: ch6 normalisation ``d/4.0`` (plan §5.3 :367).
KEY_V_DISTANCE_NORM: Final[float] = 4.0
#: Below this magnitude a denominator counts as 0 → channel is zeroed (§5.3
#: :372 "任一分母为 0 时该通道置零向量；不得产生 NaN").
_DENOM_FLOOR: Final[float] = 1e-12

# --- §5.4 pair_s constants --------------------------------------------------
#: The 48-dim layout, segment → ``[start, stop)`` (plan §5.4 :381–391).
PAIR_S_LAYOUT: Final[dict[str, tuple[int, int]]] = {
    "bond_orders": (0, 6),      # b_src, b_tgt, ΔB, |ΔB|, is_changed, is_covalent
    "flags": (6, 8),            # is_aromatic, is_spatial
    "rbf12": (8, 20),           # RBF12(d_ij), gaussian_rbf_v3's 12 centres
    "geometry": (20, 24),       # d_ij, 1/(d_ij+1), ΔZ/10, |ΔZ|/10
    "role_i": (24, 28),         # endpoint i: breaking, forming, spectator, is_key
    "role_j": (28, 32),         # endpoint j: same four
    "graph": (32, 40),          # signed_intent, ‖cv_delta‖, cv_delta[0:6]
    "sketch": (40, 48),         # W_fixed @ (key_s[i] - key_s[j])
}
#: The 12 frozen ``gaussian_rbf_v3`` centres, reused verbatim (§5.4 :385).
PAIR_RBF_CENTERS: Final[tuple[float, ...]] = tuple(
    float(c) for c in rbf_centers_v3())
#: The RBF width of the v3 feature contract (``gaussian_rbf_v3`` sigma).
PAIR_RBF_SIGMA: Final[float] = float(RBF_SIGMA)
#: ``cv_delta[0:6]`` broadcast slots (§5.4 :390); ``CV_MAX = 4`` real slots.
PAIR_CV_BROADCAST_SLOTS: Final[int] = 6
#: Width of the fixed JL-style sketch (§5.4 :391 "8×48 ±1/√48").
PAIR_SKETCH_ROWS: Final[int] = 8

# --- §5.5 key graph constants ----------------------------------------------
#: At most this many nearest keys per key atom (§5.5 :400 "最近的最多 4 个").
KEY_GRAPH_MAX_NEIGHBORS: Final[int] = 4
#: Key-key distance cutoff in Å (§5.5 :400 "距离 < 6.0 Å").
KEY_GRAPH_CUTOFF_ANGSTROM: Final[float] = 6.0
#: ``key_edge_attr`` edge_type values (§5.5 :403).
KEY_EDGE_TYPE_COVALENT: Final[int] = 0
KEY_EDGE_TYPE_SPATIAL: Final[int] = 1
#: The four RBF centres of ``key_edge_attr`` (§5.5 :403 "RBF(d;2,3,4,5)").
KEY_GRAPH_RBF_CENTERS: Final[tuple[float, ...]] = (2.0, 3.0, 4.0, 5.0)

#: The forward-view keys the Bridge reads (a SUBSET of ``FORWARD_VIEW_KEYS_V3``
#: — nothing else is touched, so the §5.6 forbidden list is unreachable).
REQUIRED_FORWARD_KEYS: Final[tuple[str, ...]] = (
    NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
    NODE_INTENT_ROLE_KEY, EDGE_INDEX_KEY, EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY, EDGE_B_TGT_KEY, EDGE_DELTA_B_KEY, EDGE_ABS_DELTA_KEY,
    EDGE_IS_CHANGED_KEY, GRAPH_CV_DELTA_KEY, GRAPH_SIGNED_INTENT_KEY,
)

#: The six cache-key components, in the plan §5.7 order (:443–450).
CACHE_KEY_FIELDS: Final[tuple[str, ...]] = (
    "encoder_ckpt_sha256",
    "bridge_spec_sha256",
    "input_feature_sha256",
    "source_frame_identity",
    "precision_version",
    "code_fingerprint",
)


# ---------------------------------------------------------------------------
# the fixed 8×48 sketch matrix (§5.4 :391) — deterministic, parameter free
# ---------------------------------------------------------------------------
def _sketch_sign(row: int, col: int) -> int:
    """Deterministic ±1 of ``W_fixed[row, col]`` (frozen into the spec SHA).

    The plan freezes the matrix as "固定的 ``8×48`` ±1/√48 矩阵" without
    spelling out the sign pattern, so the pattern is DEFINED here once, by a
    fixed integer hash (Knuth multiplicative, no RNG, no import-time state),
    and then frozen into :data:`BRIDGE_SPEC_SHA256` so that any change to it
    invalidates every cached context (§5.7 identity discipline).
    """
    hashed = ((int(row) * KEY_S_DIM_V543 + int(col)) * 2654435761) & 0xFFFFFFFF
    return 1 if ((hashed >> 13) & 1) == 0 else -1


def fixed_sketch_matrix(
        rows: int = PAIR_SKETCH_ROWS,
        cols: int = KEY_S_DIM_V543,
) -> torch.Tensor:
    """The ``rows×cols`` ``±1/√cols`` sketch matrix (unit-norm rows, §5.4)."""
    scale = 1.0 / math.sqrt(float(cols))
    values = [
        [scale if _sketch_sign(r, c) > 0 else -scale for c in range(cols)]
        for r in range(rows)
    ]
    return torch.tensor(values, dtype=torch.float32)


def _bridge_spec() -> dict[str, Any]:
    """The canonical, JSON-serialisable Bridge specification (frozen by SHA)."""
    return {
        "schema": BRIDGE_SPEC_SCHEMA,
        "version": BRIDGE_VERSION,
        "context_schema": DECODER_CONTEXT_SCHEMA_V543,
        "precision_version": PRECISION_VERSION_V543,
        "context_fields": list(DECODER_CONTEXT_FIELDS),
        "track_fields": {
            "move": list(MOVE_CONTEXT_FIELDS),
            "process": list(PROCESS_CONTEXT_FIELDS),
        },
        "forbidden_names": list(FORBIDDEN_CONTEXT_ALL_NAMES),
        "a_ref": float(A_REF_V543),
        "unit_denom_eps": UNIT_DENOM_EPS,
        "zero_denominator_policy": "channel_is_zero_vector",
        "key_v": {
            "channels": list(KEY_V_CHANNEL_NAMES),
            "spatial_kappa": {
                "mu": KEY_V_SPATIAL_MU,
                "sigma": KEY_V_SPATIAL_SIGMA,
            },
            "distance_norm": KEY_V_DISTANCE_NORM,
        },
        "pair_s": {
            "dim": int(PAIR_S_DIM_V543),
            "layout": {name: list(bounds)
                       for name, bounds in PAIR_S_LAYOUT.items()},
            "rbf_centers": [float(c) for c in PAIR_RBF_CENTERS],
            "rbf_sigma": float(PAIR_RBF_SIGMA),
            "n_rbf": int(N_RBF_V3),
            "cv_broadcast_slots": PAIR_CV_BROADCAST_SLOTS,
            "sketch_rows": PAIR_SKETCH_ROWS,
            "sketch_cols": int(KEY_S_DIM_V543),
            "sketch_scale": 1.0 / math.sqrt(float(KEY_S_DIM_V543)),
            "sketch_signs": [
                [_sketch_sign(r, c) for c in range(KEY_S_DIM_V543)]
                for r in range(PAIR_SKETCH_ROWS)
            ],
        },
        "key_graph": {
            "max_neighbors": KEY_GRAPH_MAX_NEIGHBORS,
            "cutoff_angstrom": KEY_GRAPH_CUTOFF_ANGSTROM,
            "changed_pairs_bidirectional": True,
            "undirected": True,
            "edge_attr": [
                "distance", "rbf_2", "rbf_3", "rbf_4", "rbf_5",
                "abs_delta_b", "is_changed", "edge_type",
            ],
            "edge_type_values": {
                "covalent_or_changed": KEY_EDGE_TYPE_COVALENT,
                "spatial": KEY_EDGE_TYPE_SPATIAL,
            },
            "ek_bound": "2*(P + 4*K)",
        },
        "pair_direction": "canonical_key_slot_ascending",
        "token_layout": ["graph", "keys", "pairs"],
        "key_order": "graph_contiguous_canonical",
        "cache_key_fields": list(CACHE_KEY_FIELDS),
    }


#: Canonical SHA-256 of the whole Bridge spec (plan §5.2 ``bridge_spec_sha256``
#: row / §5.7 cache identity).
BRIDGE_SPEC_SHA256: Final[str] = canonical_sha256(_bridge_spec())


def bridge_spec_sha256() -> str:
    """The frozen Bridge-spec digest (plan §11.2 API ``bridge_spec_sha256()``)."""
    return BRIDGE_SPEC_SHA256


# ---------------------------------------------------------------------------
# small deterministic helpers
# ---------------------------------------------------------------------------
def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """``numerator / denominator`` with the §5.3 zero-denominator policy.

    ``numerator`` is ``[n, 3]`` and ``denominator`` is ``[n]``.  Where the
    denominator's magnitude is below :data:`_DENOM_FLOOR` the result row is
    the ZERO vector (plan §5.3 :372 "任一分母为 0 时该通道置零向量；不得产生
    NaN") — the division itself never happens on a 0/0 pair.
    """
    if numerator.ndim != 2 or denominator.ndim != 1:
        raise V543BridgeError(
            "V543_BRIDGE_INTERNAL",
            f"_safe_div expects [n,3] / [n], got {list(numerator.shape)} / "
            f"{list(denominator.shape)}")
    keep = denominator.abs() > _DENOM_FLOOR
    safe = denominator.abs().clamp_min(_DENOM_FLOOR)
    quotient = numerator / safe.unsqueeze(-1)
    return torch.where(keep.unsqueeze(-1), quotient,
                       torch.zeros_like(numerator))


def _prefix_counts(assignment: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Per-graph counts of ``assignment`` (an int64 [n] graph-id vector)."""
    if int(assignment.numel()) == 0:
        return torch.zeros(n_graphs, dtype=torch.int64,
                           device=assignment.device)
    # ``bincount`` is the integer-safe path (no float accumulation anywhere).
    return torch.bincount(assignment.to(torch.int64), minlength=n_graphs)


def _ptr_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """``[G+1]`` prefix sums starting at 0 (plan §5.2 :287–297)."""
    ptr = torch.zeros(int(counts.numel()) + 1, dtype=torch.int64,
                      device=counts.device)
    if int(counts.numel()):
        ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


# ---------------------------------------------------------------------------
# the reconstructed changed K-pair set (plan F2 / §5.4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChangedPairSet:
    """The changed K-pair set reconstructed from the forward edges (F2).

    All tensors are derived exclusively from the forward view — no supervision
    key participates (see :func:`audit_pair_sets` for the offline comparison).

    * ``key_index`` ``[K]`` — key rows in graph-contiguous canonical order;
    * ``key_graph`` ``[K]`` — graph id of each key row;
    * ``key_permutation`` ``[K]`` — position of each ``key_index`` entry in
      the input order (the stable permutation decision 4 applies);
    * ``node_key_slot`` ``[N]`` — node row → key slot, ``-1`` for non-keys;
    * ``pair_key_slots`` ``[2,P]`` — canonical ``(i, j)`` with ``i < j``;
    * ``pair_node_index`` ``[2,P]`` — the same pairs as global node rows;
    * ``pair_graph`` ``[P]`` — graph id per pair;
    * ``delta_b`` / ``abs_delta`` / ``b_src`` / ``b_tgt`` ``[P]`` — the
      forward contract's signed intent edit, carried unchanged (F2: the sign
      lives in ``ΔB``, the direction is the canonical order);
    * ``is_covalent`` / ``is_aromatic`` / ``is_spatial`` ``[P]`` — the
      ``edge_attr`` flags of the pair's forward edge;
    * ``stats`` — audit counters (never supervision).
    """

    key_index: torch.Tensor
    key_graph: torch.Tensor
    key_permutation: torch.Tensor
    node_key_slot: torch.Tensor
    pair_key_slots: torch.Tensor
    pair_node_index: torch.Tensor
    pair_graph: torch.Tensor
    delta_b: torch.Tensor
    abs_delta: torch.Tensor
    b_src: torch.Tensor
    b_tgt: torch.Tensor
    is_covalent: torch.Tensor
    is_aromatic: torch.Tensor
    is_spatial: torch.Tensor
    stats: dict[str, int]

    @property
    def n_pairs(self) -> int:
        return int(self.delta_b.numel())


# ---------------------------------------------------------------------------
# the Bridge
# ---------------------------------------------------------------------------
class LatentBridge(nn.Module):
    """Parameter-free bridge: ``(CoarsePacket, forward_view) -> context``.

    Args:
        encoder_ckpt_sha256: A0 checkpoint digest recorded in every emitted
            context (plan §5.2 ''encoder_ckpt_sha256'' row).  Empty by default
            (unit tests build local contexts); a non-empty value must be a
            64-hex string.
        strict_delta_consistency: when ``True`` (the default) a forward view
            that carries two different ``ΔB`` values for ONE unordered pair is
            refused — the sign reference would be ambiguous (F2 / §5.6 audit
            "若不一致……必须阻断").

    The module holds NO parameter and NO persistent state: ``w_fixed`` is a
    ``persistent=False`` buffer (plan §5.1 :273), so ``state_dict()`` is empty
    and the optimizer can never see it.
    """

    spec_schema: Final[str] = BRIDGE_SPEC_SCHEMA
    spec_version: Final[str] = BRIDGE_VERSION
    context_schema: Final[str] = DECODER_CONTEXT_SCHEMA_V543

    def __init__(
            self,
            *,
            encoder_ckpt_sha256: str = "",
            strict_delta_consistency: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("w_fixed", fixed_sketch_matrix(), persistent=False)
        self._encoder_ckpt_sha256 = self._check_sha(encoder_ckpt_sha256)
        self._strict_delta_consistency = bool(strict_delta_consistency)
        self._last_stats: dict[str, object] = {}

        n_params = int(sum(int(p.numel()) for p in self.parameters()))
        if n_params:
            raise V543InitError(
                "V543_BRIDGE_PARAMS",
                f"LatentBridge must be parameter free (plan §4.1), found "
                f"{n_params} trainable parameters")
        if "w_fixed" in self.state_dict():
            raise V543InitError(
                "V543_BRIDGE_BUFFER",
                "w_fixed must be a persistent=False buffer (plan §5.1 :273), "
                "but it appears in state_dict()")

    # -- identity -----------------------------------------------------------
    @staticmethod
    def _check_sha(value: str) -> str:
        text = "" if value is None else str(value)
        if text and (len(text) != 64 or any(
                c not in "0123456789abcdef" for c in text.lower())):
            raise V543BridgeError(
                "V543_BRIDGE_CKPT_SHA",
                f"encoder_ckpt_sha256 must be a 64-hex digest or empty, got "
                f"{text!r}")
        return text.lower()

    def bridge_spec_sha256(self) -> str:
        """The frozen Bridge-spec digest (plan §11.2 API)."""
        return BRIDGE_SPEC_SHA256

    @property
    def last_build_stats(self) -> dict[str, object]:
        """Audit counters of the last :meth:`build_context` call."""
        return dict(self._last_stats)

    def describe(self) -> dict[str, Any]:
        """JSON-able description of this bridge (parity/identity input)."""
        return {
            "schema": BRIDGE_SPEC_SCHEMA,
            "version": BRIDGE_VERSION,
            "context_schema": DECODER_CONTEXT_SCHEMA_V543,
            "bridge_spec_sha256": BRIDGE_SPEC_SHA256,
            "parameters": 0,
            "buffers": {"w_fixed": {
                "shape": list(self.w_fixed.shape),
                "persistent": False,
                "dtype": str(self.w_fixed.dtype),
            }},
            "context_fields": list(DECODER_CONTEXT_FIELDS),
            "move_fields": list(MOVE_CONTEXT_FIELDS),
            "process_fields": list(PROCESS_CONTEXT_FIELDS),
            "encoder_ckpt_sha256": self._encoder_ckpt_sha256 or None,
            "writes": {
                "key_v": "plan §5.3",
                "pair_s": "plan §5.4",
                "key_edge_*": "plan §5.5",
                "token_*": "plan §5.5",
            },
        }

    # -- input boundary (plan §5.6) ----------------------------------------
    def _checked_view(
            self, forward_view: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Refuse supervision / non-forward keys, then validate the geometry.

        Refusal codes: ``V543_BRIDGE_VIEW_TYPE``,
        ``V543_SUPERVISION_LEAK`` (any :data:`SUPERVISION_KEYS_V3` member —
        plan §5.6 / F2 "监督键……绝不进 Bridge 输入"),
        ``V543_BRIDGE_FORWARD_VIEW_LEAK`` (any other key outside
        ``FORWARD_VIEW_KEYS_V3``), ``V543_BRIDGE_MISSING_KEY``,
        ``V543_BRIDGE_DTYPE``, ``V543_BRIDGE_SHAPE``.
        """
        if not isinstance(forward_view, Mapping):
            raise V543BridgeError(
                "V543_BRIDGE_VIEW_TYPE",
                f"forward_view must be a mapping, got "
                f"{type(forward_view).__name__}")
        leaks = sorted(key for key in SUPERVISION_KEYS_V3 if key in forward_view)
        if leaks:
            raise V543SupervisionLeakError(
                "V543_SUPERVISION_LEAK",
                f"LatentBridge received supervision keys {leaks}; pair_index/"
                "pair_edit/y_teacher are supervision and may never reach the "
                "Bridge (plan §5.6, F2).  The pair set is reconstructed from "
                "the forward edges instead.")
        extra = sorted(set(forward_view) - set(FORWARD_VIEW_KEYS_V3))
        if extra:
            raise V543BridgeError(
                "V543_BRIDGE_FORWARD_VIEW_LEAK",
                f"forward_view carries non-forward keys {extra}; only "
                "FORWARD_VIEW_KEYS_V3 may cross the Bridge boundary "
                "(plan §5.6)")
        missing = sorted(key for key in REQUIRED_FORWARD_KEYS
                         if key not in forward_view)
        if missing:
            raise V543BridgeError(
                "V543_BRIDGE_MISSING_KEY",
                f"forward_view is missing {missing}; the Bridge needs the "
                "geometry + forward topology of plan §5.3–§5.5")

        view: dict[str, torch.Tensor] = {}
        for key in REQUIRED_FORWARD_KEYS:
            value = forward_view[key]
            if not isinstance(value, torch.Tensor):
                raise V543BridgeError(
                    "V543_BRIDGE_VIEW_TYPE",
                    f"forward_view[{key!r}] must be a torch.Tensor, got "
                    f"{type(value).__name__}")
            view[key] = value

        float_keys = (
            NODE_POS_KEY, EDGE_ATTR_KEY, EDGE_B_SRC_KEY, EDGE_B_TGT_KEY,
            EDGE_DELTA_B_KEY, EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY,
            NODE_INTENT_ROLE_KEY, GRAPH_CV_DELTA_KEY, GRAPH_SIGNED_INTENT_KEY,
        )
        for key in float_keys:
            if view[key].dtype != torch.float32:
                raise V543BridgeError(
                    "V543_BRIDGE_DTYPE",
                    f"forward_view[{key!r}] must be float32 (precision_version "
                    f"{PRECISION_VERSION_V543}), got {view[key].dtype}")
        for key in (NODE_Z_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
                    EDGE_INDEX_KEY):
            if view[key].dtype not in (torch.int64, torch.int32):
                raise V543BridgeError(
                    "V543_BRIDGE_DTYPE",
                    f"forward_view[{key!r}] must be an integer tensor, got "
                    f"{view[key].dtype}")

        pos = view[NODE_POS_KEY]
        if pos.ndim != 2 or int(pos.shape[1]) != 3:
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                f"pos must be [N,3], got {list(pos.shape)}")
        edge_index = view[EDGE_INDEX_KEY]
        if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                f"edge_index must be [2,E], got {list(edge_index.shape)}")
        n_edges = int(edge_index.shape[1])
        if int(view[EDGE_ATTR_KEY].shape[0]) != n_edges:
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                f"edge_attr rows ({int(view[EDGE_ATTR_KEY].shape[0])}) must "
                f"match edge_index columns ({n_edges})")
        if int(view[EDGE_ATTR_KEY].shape[1]) != EDGE_PACK_DIM_V3:
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                f"edge_attr must be the packed F3 superset "
                f"[E,{EDGE_PACK_DIM_V3}], got "
                f"{list(view[EDGE_ATTR_KEY].shape)} — the Bridge reads the "
                "is_spatial / covalent_union / aromatic_union slots "
                "(EDGE_PACK_SLOTS_V3)")
        for key in (EDGE_B_SRC_KEY, EDGE_B_TGT_KEY, EDGE_DELTA_B_KEY,
                    EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY):
            if int(view[key].numel()) != n_edges:
                raise V543BridgeError(
                    "V543_BRIDGE_SHAPE",
                    f"{key} must have one value per edge ({n_edges}), got "
                    f"{int(view[key].numel())}")
        if int(view[NODE_GRAPH_KEY].numel()) != int(pos.shape[0]):
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                "graph_index must have one value per node")
        if view[NODE_INTENT_ROLE_KEY].ndim != 2 or \
                int(view[NODE_INTENT_ROLE_KEY].shape[1]) != 3:
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                "intent_role must be [N,3] (breaking/forming/spectator, "
                "INTENT_ROLE_CHANNELS)")
        if int(view[NODE_INTENT_ROLE_KEY].shape[0]) != int(pos.shape[0]):
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE", "intent_role must have one row per node")
        if view[GRAPH_CV_DELTA_KEY].ndim != 2 or \
                int(view[GRAPH_CV_DELTA_KEY].shape[1]) != CV_MAX:
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                f"cv_target_delta must be [G,{CV_MAX}], got "
                f"{list(view[GRAPH_CV_DELTA_KEY].shape)}")
        n_graphs = int(view[GRAPH_CV_DELTA_KEY].shape[0])
        if int(view[GRAPH_SIGNED_INTENT_KEY].numel()) != n_graphs:
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                f"signed_intent must be [G] with G={n_graphs}, got "
                f"{int(view[GRAPH_SIGNED_INTENT_KEY].numel())}")
        graph_index = view[NODE_GRAPH_KEY]
        if int(pos.shape[0]) and (
                int(graph_index.min()) < 0
                or int(graph_index.max()) >= n_graphs):
            raise V543BridgeError(
                "V543_BRIDGE_SHAPE",
                f"graph_index values must lie in [0,{n_graphs})")
        return view

    # -- changed pair reconstruction (F2 / §5.4) ---------------------------
    def reconstruct_changed_pairs(
            self, forward_view: Mapping[str, Any]) -> ChangedPairSet:
        """Reconstruct the changed K-pair set from the forward edges (F2).

        The set is exactly "every forward edge with ``is_changed == 1`` whose
        two endpoints are key atoms", reduced to undirected canonical pairs
        (key-slot ascending, §5.4 :379).  Supervision is never read; the
        equality with the supervised ``pair_index`` is proved separately by
        :func:`audit_pair_sets` (read-only, offline, plan §5.6 :431–435).
        """
        return self._reconstruct(self._checked_view(forward_view))

    def _reconstruct(
            self, view: Mapping[str, torch.Tensor]) -> ChangedPairSet:
        pos = view[NODE_POS_KEY]
        graph_index = view[NODE_GRAPH_KEY].to(torch.int64)
        key_index_raw = view[NODE_KEY_SET_KEY].to(torch.int64)
        n_nodes = int(pos.shape[0])
        n_keys = int(key_index_raw.numel())
        device = pos.device

        if n_keys == 0:
            raise V543BridgeError(
                "V543_BRIDGE_NO_KEYS",
                "each graph must carry at least 1 key (plan §5.2 :309); the "
                "forward view has none")
        if int(key_index_raw.min()) < 0 or int(key_index_raw.max()) >= n_nodes:
            raise V543BridgeError(
                "V543_BRIDGE_KEY_INDEX",
                f"key_index rows must lie in [0,{n_nodes})")
        if int(torch.unique(key_index_raw).numel()) != n_keys:
            raise V543BridgeError(
                "V543_BRIDGE_KEY_INDEX",
                "key_index must not repeat a node row")

        # decision 4: graph-contiguous canonical key order (key_ptr needs it).
        key_graph_raw = graph_index[key_index_raw]
        key_permutation = torch.argsort(key_graph_raw, stable=True)
        key_index = key_index_raw[key_permutation]
        key_graph = key_graph_raw[key_permutation]
        node_key_slot = torch.full((n_nodes,), -1, dtype=torch.int64,
                                   device=device)
        node_key_slot[key_index] = torch.arange(n_keys, dtype=torch.int64,
                                                device=device)

        edge_index = view[EDGE_INDEX_KEY].to(torch.int64)
        src = edge_index[0]
        dst = edge_index[1]
        src_slot = node_key_slot[src]
        dst_slot = node_key_slot[dst]
        delta_b = view[EDGE_DELTA_B_KEY].to(torch.float32)
        abs_delta = view[EDGE_ABS_DELTA_KEY].to(torch.float32)
        is_changed = view[EDGE_IS_CHANGED_KEY].to(torch.float32)
        edge_attr = view[EDGE_ATTR_KEY].to(torch.float32)
        b_src = view[EDGE_B_SRC_KEY].to(torch.float32)
        b_tgt = view[EDGE_B_TGT_KEY].to(torch.float32)

        both_key = (src_slot >= 0) & (dst_slot >= 0)
        changed = is_changed > 0
        pair_mask = both_key & changed
        n_changed_edges = int(changed.sum())
        n_nonkey_changed_edges = int((changed & ~both_key).sum())
        if int(pair_mask.sum()) and bool((delta_b[pair_mask] == 0).any()):
            raise V543BridgeError(
                "V543_BRIDGE_CHANGED_WITHOUT_DELTA",
                "a forward edge carries is_changed=1 with delta_b_intent=0; "
                "the pair set cannot be reconstructed from an edgeless "
                "change (plan F2)")

        slot_lo = torch.minimum(src_slot[pair_mask], dst_slot[pair_mask])
        slot_hi = torch.maximum(src_slot[pair_mask], dst_slot[pair_mask])
        n_pair_edges = int(slot_lo.numel())
        if n_pair_edges:
            canonical = slot_lo * n_keys + slot_hi
            unique_canonical, inverse = torch.unique(
                canonical, sorted=True, return_inverse=True)
            n_pairs = int(unique_canonical.numel())
            order = torch.argsort(inverse, stable=True)
            sorted_inverse = inverse[order]
            first = torch.ones_like(sorted_inverse, dtype=torch.bool)
            if n_pairs > 1:
                first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
            representative = order[first]

            delta_all = delta_b[pair_mask]
            abs_all = abs_delta[pair_mask]
            b_src_all = b_src[pair_mask]
            b_tgt_all = b_tgt[pair_mask]
            attr_all = edge_attr[pair_mask]

            if self._strict_delta_consistency and n_pair_edges > n_pairs:
                counts = torch.zeros(n_pairs, dtype=torch.float32,
                                     device=device).index_add_(
                    0, inverse, torch.ones_like(delta_all))
                totals = torch.zeros(n_pairs, dtype=torch.float32,
                                     device=device).index_add_(
                    0, inverse, delta_all)
                means = totals / counts.clamp_min(1.0)
                spread = torch.zeros(n_pairs, dtype=torch.float32,
                                     device=device).index_add_(
                    0, inverse, (delta_all - means[inverse]).abs())
                if bool(((counts > 1.0) & (spread > 1e-6)).any()):
                    raise V543BridgeError(
                        "V543_BRIDGE_PAIR_DELTA_CONFLICT",
                        "the forward view carries inconsistent delta_b_intent "
                        "values for one unordered key pair; the sign "
                        "reference of ΔB is ambiguous and the pair set must "
                        "not be built (F2 / plan §5.6 audit)")

            pair_lo = unique_canonical // n_keys
            pair_hi = unique_canonical % n_keys
            pair_graph_unsorted = key_graph[pair_lo]
            if bool((key_graph[pair_hi] != pair_graph_unsorted).any()):
                raise V543BridgeError(
                    "V543_BRIDGE_PAIR_CROSS_GRAPH",
                    "a changed key pair spans two graphs; a pair must belong "
                    "to exactly one graph (plan §5.2 pair_graph row)")

            delta_p = delta_all[representative]
            abs_p = abs_all[representative]
            b_src_p = b_src_all[representative]
            b_tgt_p = b_tgt_all[representative]
            attr_p = attr_all[representative]
            covalent_p = attr_p[:, EDGE_PACK_SLOTS_V3["covalent_union"]]
            aromatic_p = attr_p[:, EDGE_PACK_SLOTS_V3["aromatic_union"]]
            spatial_p = attr_p[:, EDGE_PACK_SLOTS_V3["is_spatial"]]

            # decision 5: pairs are ordered by (graph, canonical key slot).
            graph_order = torch.argsort(pair_graph_unsorted, stable=True)
            pair_key_slots = torch.stack(
                [pair_lo[graph_order], pair_hi[graph_order]], dim=0)
            pair_graph = pair_graph_unsorted[graph_order]
            delta_p = delta_p[graph_order]
            abs_p = abs_p[graph_order]
            b_src_p = b_src_p[graph_order]
            b_tgt_p = b_tgt_p[graph_order]
            covalent_p = covalent_p[graph_order]
            aromatic_p = aromatic_p[graph_order]
            spatial_p = spatial_p[graph_order]
        else:
            n_pairs = 0
            pair_key_slots = torch.zeros((2, 0), dtype=torch.int64,
                                         device=device)
            pair_graph = torch.zeros((0,), dtype=torch.int64, device=device)
            delta_p = torch.zeros((0,), dtype=torch.float32, device=device)
            abs_p = torch.zeros((0,), dtype=torch.float32, device=device)
            b_src_p = torch.zeros((0,), dtype=torch.float32, device=device)
            b_tgt_p = torch.zeros((0,), dtype=torch.float32, device=device)
            covalent_p = torch.zeros((0,), dtype=torch.float32, device=device)
            aromatic_p = torch.zeros((0,), dtype=torch.float32, device=device)
            spatial_p = torch.zeros((0,), dtype=torch.float32, device=device)

        pair_node_index = key_index[pair_key_slots]
        return ChangedPairSet(
            key_index=key_index,
            key_graph=key_graph,
            key_permutation=key_permutation,
            node_key_slot=node_key_slot,
            pair_key_slots=pair_key_slots,
            pair_node_index=pair_node_index,
            pair_graph=pair_graph,
            delta_b=delta_p,
            abs_delta=abs_p,
            b_src=b_src_p,
            b_tgt=b_tgt_p,
            is_covalent=covalent_p,
            is_aromatic=aromatic_p,
            is_spatial=spatial_p,
            stats={
                "n_nodes": n_nodes,
                "n_keys": n_keys,
                "n_edges": int(edge_index.shape[1]),
                "n_changed_edges": n_changed_edges,
                "n_nonkey_changed_edges": n_nonkey_changed_edges,
                "n_pair_edges": n_pair_edges,
                "n_pairs": n_pairs,
            },
        )

    # -- §5.3 key_v ---------------------------------------------------------
    def _edge_weighted_direction(
            self,
            unit: torch.Tensor,
            weights: torch.Tensor,
            dst_slot: torch.Tensor,
            mask: torch.Tensor,
            n_keys: int,
    ) -> torch.Tensor:
        """``Σ_{dst=k, mask} w_e·unit_e / Σ w_e`` (channels 1/2/3/4/6)."""
        keep = mask & (dst_slot >= 0)
        numerator = torch.zeros((n_keys, 3), dtype=unit.dtype,
                                device=unit.device)
        denominator = torch.zeros((n_keys,), dtype=unit.dtype,
                                  device=unit.device)
        if bool(keep.any()):
            index = dst_slot[keep]
            numerator = numerator.index_add_(
                0, index, unit[keep] * weights[keep].unsqueeze(-1))
            denominator = denominator.index_add_(0, index, weights[keep])
        return _safe_div(numerator, denominator)

    def _pair_direction(
            self,
            pos: torch.Tensor,
            key_index: torch.Tensor,
            pairs: ChangedPairSet,
            n_keys: int,
    ) -> torch.Tensor:
        """ch5 — ``Σ_p sign(ΔB_p)·(x[other]-x[k])/(‖…‖+0.1) / count``."""
        device = pos.device
        empty = torch.zeros((n_keys, 3), dtype=torch.float32, device=device)
        if pairs.n_pairs == 0:
            return empty
        key_pos = pos[key_index]
        lo = pairs.pair_key_slots[0]
        hi = pairs.pair_key_slots[1]
        sign = torch.sign(pairs.delta_b)
        vectors = torch.cat([key_pos[hi] - key_pos[lo],
                             key_pos[lo] - key_pos[hi]], dim=0)
        vectors = vectors / (vectors.norm(dim=-1, keepdim=True)
                             + UNIT_DENOM_EPS)
        vectors = vectors * torch.cat([sign, sign], dim=0).unsqueeze(-1)
        index = torch.cat([lo, hi], dim=0)
        numerator = torch.zeros_like(empty).index_add_(0, index, vectors)
        denominator = torch.zeros((n_keys,), dtype=torch.float32,
                                  device=device).index_add_(
            0, index, torch.ones_like(index, dtype=torch.float32))
        return _safe_div(numerator, denominator)

    def _centroid_direction(
            self,
            pos: torch.Tensor,
            key_index: torch.Tensor,
            key_graph: torch.Tensor,
            n_graphs: int,
            n_keys: int,
    ) -> torch.Tensor:
        """ch7 — ``(mean_{j∈K, j≠k} x[j] - x[k]) / (‖…‖+0.1)``."""
        device = pos.device
        key_pos = pos[key_index]
        graph_sum = torch.zeros((n_graphs, 3), dtype=torch.float32,
                                device=device).index_add_(0, key_graph, key_pos)
        counts = torch.zeros((n_graphs,), dtype=torch.float32,
                             device=device).index_add_(
            0, key_graph, torch.ones((n_keys,), dtype=torch.float32,
                                     device=device))
        other_sum = graph_sum[key_graph] - key_pos
        other_count = counts[key_graph] - 1.0
        has_other = other_count > 0.0
        mean_other = other_sum / other_count.clamp_min(1.0).unsqueeze(-1)
        delta = torch.where(has_other.unsqueeze(-1), mean_other - key_pos,
                            torch.zeros_like(key_pos))
        return delta / (delta.norm(dim=-1, keepdim=True) + UNIT_DENOM_EPS)

    def _build_key_v(
            self,
            view: Mapping[str, torch.Tensor],
            u0: torch.Tensor,
            pairs: ChangedPairSet,
            n_graphs: int,
    ) -> torch.Tensor:
        """The ``key_v [K,8,3]`` vector basis (plan §5.3, channel for channel)."""
        pos = view[NODE_POS_KEY]
        key_index = pairs.key_index
        key_graph = pairs.key_graph
        n_keys = int(key_index.numel())
        edge_index = view[EDGE_INDEX_KEY].to(torch.int64)
        edge_attr = view[EDGE_ATTR_KEY]
        delta_b = view[EDGE_DELTA_B_KEY]
        abs_delta = view[EDGE_ABS_DELTA_KEY]

        src = edge_index[0]
        dst = edge_index[1]
        dst_slot = pairs.node_key_slot[dst]
        differences = pos[src] - pos[dst]
        distance = differences.norm(dim=-1)
        unit = differences / (distance.unsqueeze(-1) + UNIT_DENOM_EPS)

        covalent = edge_attr[:, EDGE_PACK_SLOTS_V3["covalent_union"]] > 0.5
        spatial = edge_attr[:, EDGE_PACK_SLOTS_V3["is_spatial"]] > 0.5
        every_edge = torch.ones_like(distance, dtype=torch.bool)

        channels = [
            # ch0: u0[k]/a_ref — inherited MOVE initial guess, made dimensionless.
            u0 / A_REF_V543,
            # ch1: breaking (ΔB<0), weight |ΔB|.
            self._edge_weighted_direction(
                unit, abs_delta, dst_slot, delta_b < 0, n_keys),
            # ch2: forming (ΔB>0), weight |ΔB|.
            self._edge_weighted_direction(
                unit, abs_delta, dst_slot, delta_b > 0, n_keys),
            # ch3: unchanged covalent (ΔB=0 & covalent) — plain mean.
            self._edge_weighted_direction(
                unit, torch.ones_like(distance), dst_slot,
                (delta_b == 0) & covalent, n_keys),
            # ch4: spatial, weight κ(d)=exp(-((d-3)/1)²).
            self._edge_weighted_direction(
                unit,
                torch.exp(-((distance - KEY_V_SPATIAL_MU)
                            / KEY_V_SPATIAL_SIGMA) ** 2),
                dst_slot, spatial, n_keys),
            # ch5: critical-pair directions per changed K-pair.
            self._pair_direction(pos, key_index, pairs, n_keys),
            # ch6: distance-weighted mean over ALL incident edges, weight d/4.
            self._edge_weighted_direction(
                unit, distance / KEY_V_DISTANCE_NORM, dst_slot,
                every_edge, n_keys),
            # ch7: direction to the other keys' centroid (same graph).
            self._centroid_direction(pos, key_index, key_graph, n_graphs,
                                     n_keys),
        ]
        key_v = torch.stack(channels, dim=1)
        if tuple(key_v.shape) != (n_keys, KEY_V_CHANNELS_V543, KEY_V_DIM_V543):
            raise V543BridgeError(
                "V543_BRIDGE_KEY_V_SHAPE",
                f"key_v must be [{n_keys},{KEY_V_CHANNELS_V543},"
                f"{KEY_V_DIM_V543}], got {list(key_v.shape)}")
        return key_v

    # -- §5.4 pair_s --------------------------------------------------------
    def _build_pair_s(
            self,
            view: Mapping[str, torch.Tensor],
            key_s: torch.Tensor,
            pairs: ChangedPairSet,
            n_graphs: int,
    ) -> torch.Tensor:
        """The ``pair_s [P,48]`` layout, segment for segment (plan §5.4)."""
        device = key_s.device
        n_pairs = pairs.n_pairs
        if n_pairs == 0:
            return torch.zeros((0, PAIR_S_DIM_V543), dtype=torch.float32,
                               device=device)
        pos = view[NODE_POS_KEY]
        key_index = pairs.key_index
        key_pos = pos[key_index]
        numbers = view[NODE_Z_KEY].to(torch.float32)[key_index]
        lo = pairs.pair_key_slots[0]
        hi = pairs.pair_key_slots[1]

        difference = key_pos[lo] - key_pos[hi]
        distance = difference.norm(dim=-1)
        centers = torch.tensor(PAIR_RBF_CENTERS, dtype=torch.float32,
                               device=device)
        rbf12 = torch.exp(
            -0.5 * ((distance.unsqueeze(-1) - centers.unsqueeze(0))
                    / PAIR_RBF_SIGMA) ** 2)
        delta_z = (numbers[lo] - numbers[hi]) / 10.0

        role = view[NODE_INTENT_ROLE_KEY]
        role_lo = role[key_index][lo]
        role_hi = role[key_index][hi]
        ones = torch.ones((n_pairs, 1), dtype=torch.float32, device=device)

        cv_delta = view[GRAPH_CV_DELTA_KEY]
        signed_intent = view[GRAPH_SIGNED_INTENT_KEY]
        pair_graph = pairs.pair_graph
        cv_norm = cv_delta.norm(dim=-1)[pair_graph].unsqueeze(-1)
        cv_slots = min(int(cv_delta.shape[1]), PAIR_CV_BROADCAST_SLOTS)
        cv_broadcast = torch.zeros((n_pairs, PAIR_CV_BROADCAST_SLOTS),
                                   dtype=torch.float32, device=device)
        cv_broadcast[:, :cv_slots] = cv_delta[pair_graph, :cv_slots]

        sketch = torch.matmul(key_s[lo] - key_s[hi], self.w_fixed.t())

        segments = (
            ("bond_orders", torch.cat([
                pairs.b_src.unsqueeze(-1),
                pairs.b_tgt.unsqueeze(-1),
                pairs.delta_b.unsqueeze(-1),
                pairs.abs_delta.unsqueeze(-1),
                ones,                          # is_changed (changed by construction)
                pairs.is_covalent.unsqueeze(-1),
            ], dim=-1)),
            ("flags", torch.cat([
                pairs.is_aromatic.unsqueeze(-1),
                pairs.is_spatial.unsqueeze(-1),
            ], dim=-1)),
            ("rbf12", rbf12),
            ("geometry", torch.cat([
                distance.unsqueeze(-1),
                1.0 / (distance.unsqueeze(-1) + 1.0),
                delta_z.unsqueeze(-1),
                delta_z.abs().unsqueeze(-1),
            ], dim=-1)),
            ("role_i", torch.cat([role_lo, ones], dim=-1)),
            ("role_j", torch.cat([role_hi, ones], dim=-1)),
            ("graph", torch.cat([
                signed_intent[pair_graph].unsqueeze(-1),
                cv_norm,
                cv_broadcast,
            ], dim=-1)),
            ("sketch", sketch),
        )
        for name, block in segments:
            start, stop = PAIR_S_LAYOUT[name]
            if int(block.shape[1]) != stop - start:
                raise V543BridgeError(
                    "V543_BRIDGE_PAIR_S_SHAPE",
                    f"pair_s segment {name!r} must be {stop - start} wide, "
                    f"got {int(block.shape[1])} (plan §5.4)")
        pair_s = torch.cat([block for _name, block in segments], dim=-1)
        if tuple(pair_s.shape) != (n_pairs, PAIR_S_DIM_V543):
            raise V543BridgeError(
                "V543_BRIDGE_PAIR_S_SHAPE",
                f"pair_s must be [{n_pairs},{PAIR_S_DIM_V543}], got "
                f"{list(pair_s.shape)}")
        return pair_s

    # -- §5.5 key graph -----------------------------------------------------
    def _build_key_edges(
            self,
            view: Mapping[str, torch.Tensor],
            pairs: ChangedPairSet,
            n_keys: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """The key-key edge set + its 8-dim attributes (plan §5.5 :397–403).

        Edges: (1) every changed K-pair, both directions; (2) every key atom to
        its nearest ≤4 key atoms within 6.0 Å, excluding itself and pairs that
        already exist.  Undirected, deduplicated, stored both ways, with the
        hard bound ``Ek ≤ 2·(P + 4K)``.
        """
        device = view[NODE_POS_KEY].device
        key_index = pairs.key_index
        key_graph = pairs.key_graph
        key_pos = view[NODE_POS_KEY][key_index]
        n_graphs = int(view[GRAPH_CV_DELTA_KEY].shape[0])

        # (lo, hi) -> pair row, or -1 for a pure spatial (kNN) edge.
        undirected: dict[tuple[int, int], int] = {}
        for row in range(pairs.n_pairs):
            lower = int(pairs.pair_key_slots[0, row])
            upper = int(pairs.pair_key_slots[1, row])
            undirected[(lower, upper)] = row
        n_changed_undirected = len(undirected)

        for graph in range(n_graphs):
            slots = torch.nonzero(key_graph == graph, as_tuple=False).squeeze(-1)
            count = int(slots.numel())
            if count < 2:
                continue
            coords = key_pos[slots]
            distance = (coords.unsqueeze(1) - coords.unsqueeze(0)).norm(dim=-1)
            distance = distance.clone()
            distance.fill_diagonal_(float("inf"))
            take = min(KEY_GRAPH_MAX_NEIGHBORS, count - 1)
            values, indices = torch.topk(distance, k=take, largest=False, dim=-1)
            values_list = values.detach().cpu().tolist()
            indices_list = indices.detach().cpu().tolist()
            slots_list = slots.detach().cpu().tolist()
            for row in range(count):
                candidates: list[tuple[float, int, int, int]] = []
                for column in range(take):
                    value = float(values_list[row][column])
                    if not value < KEY_GRAPH_CUTOFF_ANGSTROM:
                        continue
                    other_local = int(indices_list[row][column])
                    a = int(slots_list[row])
                    b = int(slots_list[other_local])
                    lower, upper = (a, b) if a < b else (b, a)
                    candidates.append((value, b, lower, upper))
                # decision 7: deterministic tie-break by (distance, key slot).
                candidates.sort()
                added = 0
                for _value, _other, lower, upper in candidates:
                    if added >= KEY_GRAPH_MAX_NEIGHBORS:
                        break
                    if (lower, upper) in undirected:
                        continue
                    undirected[(lower, upper)] = -1
                    added += 1

        items = sorted(undirected.items())
        n_undirected = len(items)
        n_edges = 2 * n_undirected
        if items:
            lower = torch.tensor([item[0][0] for item in items],
                                 dtype=torch.int64, device=device)
            upper = torch.tensor([item[0][1] for item in items],
                                 dtype=torch.int64, device=device)
            row_of = torch.tensor([item[1] for item in items],
                                  dtype=torch.int64, device=device)
        else:
            lower = torch.zeros((0,), dtype=torch.int64, device=device)
            upper = torch.zeros((0,), dtype=torch.int64, device=device)
            row_of = torch.zeros((0,), dtype=torch.int64, device=device)

        difference = key_pos[lower] - key_pos[upper]
        distance = difference.norm(dim=-1) if n_undirected else \
            torch.zeros((0,), dtype=torch.float32, device=device)
        centers = torch.tensor(KEY_GRAPH_RBF_CENTERS, dtype=torch.float32,
                               device=device)
        if n_undirected:
            rbf = torch.exp(-0.5 * ((distance.unsqueeze(-1)
                                     - centers.unsqueeze(0))
                                    / PAIR_RBF_SIGMA) ** 2)
        else:
            rbf = torch.zeros((0, len(KEY_GRAPH_RBF_CENTERS)),
                              dtype=torch.float32, device=device)
        has_pair = row_of >= 0
        absolute_delta = torch.zeros((n_undirected,), dtype=torch.float32,
                                     device=device)
        if n_undirected and pairs.n_pairs:
            # ``torch.where`` evaluates BOTH branches, so the gather must be
            # guarded: with ``P = 0`` (a legal empty pair set, §5.2 :309)
            # ``pairs.abs_delta`` is a zero-length tensor and ANY index —
            # including the clamped ``0`` of the spatial rows — would raise.
            absolute_delta = torch.where(
                has_pair,
                pairs.abs_delta[row_of.clamp_min(0)],
                torch.zeros((n_undirected,), dtype=torch.float32,
                            device=device))
        is_changed_column = has_pair.to(torch.float32)
        edge_type_column = torch.where(
            has_pair,
            torch.full_like(is_changed_column, float(KEY_EDGE_TYPE_COVALENT)),
            torch.full_like(is_changed_column, float(KEY_EDGE_TYPE_SPATIAL)))

        attributes = torch.cat([
            distance.unsqueeze(-1),
            rbf,
            absolute_delta.unsqueeze(-1),
            is_changed_column.unsqueeze(-1),
            edge_type_column.unsqueeze(-1),
        ], dim=-1) if n_undirected else torch.zeros(
            (0, KEY_EDGE_ATTR_DIM_V543), dtype=torch.float32, device=device)
        if int(attributes.shape[1]) != KEY_EDGE_ATTR_DIM_V543:
            raise V543BridgeError(
                "V543_BRIDGE_KEY_EDGE_ATTR_SHAPE",
                f"key_edge_attr must be {KEY_EDGE_ATTR_DIM_V543} wide, got "
                f"{int(attributes.shape[1])} (plan §5.5)")

        key_edge_index = torch.stack(
            [torch.cat([lower, upper]), torch.cat([upper, lower])], dim=0)
        key_edge_attr = torch.cat([attributes, attributes], dim=0)

        bound = 2 * (pairs.n_pairs + KEY_GRAPH_MAX_NEIGHBORS * n_keys)
        if n_edges > bound:
            raise V543BridgeError(
                "V543_BRIDGE_EK_BOUND",
                f"Ek={n_edges} exceeds 2·(P + 4K) = {bound} (plan §5.5 :401)")
        return key_edge_index, key_edge_attr, {
            "n_key_edges": n_edges,
            "n_changed_key_edges": 2 * n_changed_undirected,
            "n_spatial_key_edges": 2 * (n_undirected - n_changed_undirected),
            "ek_bound": bound,
        }

    # -- §5.5 token layout --------------------------------------------------
    def _build_token_layout(
            self,
            key_graph: torch.Tensor,
            pair_graph: torch.Tensor,
            n_graphs: int,
            n_keys: int,
            n_pairs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """``token_index/type/graph`` + ``key_ptr/pair_ptr/token_ptr`` (§5.5).

        Per graph the token block is ``[graph, keys…, pairs…]``; the graph
        token carries ``token_index = -1`` and ``token_type = 0``; key tokens
        carry their key row and type 1, pair tokens their pair row and type 2.
        """
        device = key_graph.device
        key_counts = _prefix_counts(key_graph, n_graphs)
        pair_counts = _prefix_counts(pair_graph, n_graphs)
        key_ptr = _ptr_from_counts(key_counts)
        pair_ptr = _ptr_from_counts(pair_counts)

        key_ptr_list = key_ptr.detach().cpu().tolist()
        pair_ptr_list = pair_ptr.detach().cpu().tolist()
        index_values: list[int] = []
        type_values: list[int] = []
        graph_values: list[int] = []
        for graph in range(n_graphs):
            index_values.append(-1)
            type_values.append(0)
            graph_values.append(graph)
            for row in range(int(key_ptr_list[graph]),
                             int(key_ptr_list[graph + 1])):
                index_values.append(row)
                type_values.append(1)
                graph_values.append(graph)
            for row in range(int(pair_ptr_list[graph]),
                             int(pair_ptr_list[graph + 1])):
                index_values.append(row)
                type_values.append(2)
                graph_values.append(graph)

        token_index = torch.tensor(index_values, dtype=torch.int64,
                                   device=device)
        token_type = torch.tensor(type_values, dtype=torch.int8, device=device)
        token_graph = torch.tensor(graph_values, dtype=torch.int64,
                                   device=device)
        token_counts = (key_counts + pair_counts
                        + torch.ones(n_graphs, dtype=torch.int64,
                                     device=device))
        token_ptr = _ptr_from_counts(token_counts)
        if int(token_ptr[-1]) != n_graphs + n_keys + n_pairs:
            raise V543BridgeError(
                "V543_BRIDGE_TOKEN_LAYOUT",
                f"T = {int(token_ptr[-1])} must equal G + K + P = "
                f"{n_graphs + n_keys + n_pairs} (plan §5.2 :307)")
        return (token_index, token_type, token_graph, token_ptr,
                key_ptr, pair_ptr)

    # -- the public build ---------------------------------------------------
    def build_context(
            self,
            packet: CoarsePacket,
            forward_view: Mapping[str, Any],
            *,
            encoder_ckpt_sha256: str | None = None,
    ) -> DecoderContextV1:
        """Build one :class:`DecoderContextV1` (plan §11.2 API).

        Order of operations: supervision/forward-view boundary → packet
        validation → changed-pair reconstruction (F2) → ``key_v`` (§5.3) →
        ``pair_s`` (§5.4) → key graph (§5.5) → token layout (§5.5) →
        ``validate_decoder_context`` (§5.2 形状约束) →
        ``assert_no_supervision`` (§5.6).

        The Builder reads ``pos`` only as a difference and never detaches:
        in ``frozen_encoder`` mode the caller wraps the call in
        ``torch.no_grad()`` (§5.8 last bullet), in ``joint_finetune`` mode the
        autograd graph through ``packet.node_h``/``packet.z`` stays live
        (§5.7 first rule: no detached cache).

        Refusals: every code of :meth:`_checked_view` /
        :meth:`reconstruct_changed_pairs`, plus ``V543_BRIDGE_PACKET``,
        ``V543_BRIDGE_KEY_INDEX``, ``V543_BRIDGE_EK_BOUND`` and the
        :func:`validate_decoder_context` family.
        """
        view = self._checked_view(forward_view)
        pairs = self._reconstruct(view)

        n_graphs = int(view[GRAPH_CV_DELTA_KEY].shape[0])
        n_nodes = int(view[NODE_POS_KEY].shape[0])
        n_edges = int(view[EDGE_INDEX_KEY].shape[1])
        n_keys = int(pairs.key_index.numel())
        n_pairs = pairs.n_pairs

        try:
            validate_coarse_packet(
                packet,
                n_graphs=n_graphs,
                n_nodes=n_nodes,
                n_keys=n_keys,
                n_edges=n_edges,
            )
        except V543ContractError as error:
            raise V543BridgeError(
                "V543_BRIDGE_PACKET",
                f"the CoarsePacket does not match the forward view: "
                f"{error.code} ({error.detail})") from error

        sha = (self._encoder_ckpt_sha256
               if encoder_ckpt_sha256 is None
               else self._check_sha(encoder_ckpt_sha256))

        key_s = packet.node_h[pairs.key_index]
        u0 = packet.u0[pairs.key_permutation]

        key_v = self._build_key_v(view, u0, pairs, n_graphs)
        pair_s = self._build_pair_s(view, key_s, pairs, n_graphs)
        key_edge_index, key_edge_attr, edge_stats = self._build_key_edges(
            view, pairs, n_keys)
        (token_index, token_type, token_graph, token_ptr,
         key_ptr, pair_ptr) = self._build_token_layout(
            pairs.key_graph, pairs.pair_graph, n_graphs, n_keys, n_pairs)

        context = DecoderContextV1(
            graph_s=packet.z,
            key_s=key_s,
            key_v=key_v,
            u0=u0,
            key_graph=pairs.key_graph,
            key_ptr=key_ptr,
            key_edge_index=key_edge_index,
            key_edge_attr=key_edge_attr,
            pair_s=pair_s,
            intent_pair_index=pairs.pair_key_slots,
            pair_graph=pairs.pair_graph,
            pair_ptr=pair_ptr,
            token_index=token_index,
            token_type=token_type,
            token_graph=token_graph,
            token_ptr=token_ptr,
            q0_raw=packet.q0_raw,
            endpoint0_logit=packet.e0,
            schema=DECODER_CONTEXT_SCHEMA_V543,
            bridge_spec_sha256=BRIDGE_SPEC_SHA256,
            encoder_ckpt_sha256=sha,
            precision_version=PRECISION_VERSION_V543,
        )
        validate_decoder_context(context)
        assert_no_supervision(context)

        stats: dict[str, object] = dict(pairs.stats)
        stats.update(edge_stats)
        stats.update({
            "n_tokens": int(token_index.numel()),
            "pair_s_dim": int(pair_s.shape[1]),
            "key_v_shape": list(key_v.shape),
            "bridge_spec_sha256": BRIDGE_SPEC_SHA256,
            "encoder_ckpt_sha256": sha,
        })
        self._last_stats = stats
        return context


# ---------------------------------------------------------------------------
# module-level API (plan §11.2: ``build_context`` / ``bridge_spec_sha256``)
# ---------------------------------------------------------------------------
def build_context(
        packet: CoarsePacket,
        forward_view: Mapping[str, Any],
        *,
        encoder_ckpt_sha256: str | None = None,
) -> DecoderContextV1:
    """Build a context with a fresh, stateless :class:`LatentBridge`.

    The Bridge carries no state beyond its fixed buffer, so a fresh instance is
    semantically identical to a reused one; training loops should hold ONE
    instance (its ``w_fixed`` buffer then lives alongside the model) and call
    :meth:`LatentBridge.build_context`.
    """
    return LatentBridge(
        encoder_ckpt_sha256=encoder_ckpt_sha256 or "",
    ).build_context(packet, forward_view)


# ---------------------------------------------------------------------------
# §5.7 cache identity
# ---------------------------------------------------------------------------
def source_frame_identity(
        *,
        curve_id: str,
        node_index: int,
        intent_id: object,
        d0: float,
) -> str:
    """``curve_id + node_index + intent_id + d0`` canonical hash (§5.7 :448).

    This is the ONLY place the Bridge module touches a curve identity, and it
    produces a digest: the context itself never carries ``curve_id`` /
    ``sample_id`` (§5.7 :439 "**不含** ``sample_id``、``curve_id``").
    """
    return canonical_sha256({
        "curve_id": str(curve_id),
        "node_index": int(node_index),
        "intent_id": intent_id,
        "d0": float(d0),
    })


def cache_key(
        *,
        encoder_ckpt_sha256: str,
        bridge_spec_sha256: str = BRIDGE_SPEC_SHA256,
        input_feature_sha256: str,
        source_frame_identity: str,
        precision_version: str = PRECISION_VERSION_V543,
        code_fingerprint: str,
) -> str:
    """The plan §5.7 (:443–450) cache key, in the plan's exact order.

    ``encoder_ckpt_sha256 | bridge_spec_sha256 | input_feature_sha256 |
    source_frame_identity | precision_version | code_fingerprint``

    The key deliberately contains NO label, NO split and NO curve name —
    only identities (§5.7 :439, and the caching rules of :454–458 are the
    cache owner's, not the Bridge's).
    """
    values = {
        "encoder_ckpt_sha256": encoder_ckpt_sha256,
        "bridge_spec_sha256": bridge_spec_sha256,
        "input_feature_sha256": input_feature_sha256,
        "source_frame_identity": source_frame_identity,
        "precision_version": precision_version,
        "code_fingerprint": code_fingerprint,
    }
    parts: list[str] = []
    for name in CACHE_KEY_FIELDS:
        value = values[name]
        if not isinstance(value, str) or not value:
            raise V543BridgeError(
                "V543_BRIDGE_CACHE_KEY",
                f"cache key component {name!r} must be a non-empty string "
                "(plan §5.7 :443–450 — an incomplete identity must rebuild, "
                "never collide)")
        parts.append(value)
    return "|".join(parts)


# ---------------------------------------------------------------------------
# §5.6 pair-set source audit (read-only, offline — supervision is COMPARED
# against, never consumed by the model)
# ---------------------------------------------------------------------------
def audit_pair_sets(
        bridge: LatentBridge,
        forward_view: Mapping[str, Any],
        *,
        supervised_pair_index: torch.Tensor,
        max_examples: int = 8,
) -> dict[str, object]:
    """Compare the reconstructed pair set against supervision (plan §5.6).

    Plan §5.6 (:431–435): "在 10,000 个采样帧上，比较 Bridge 从 forward edges
    重构的 pair 集合与监督 ``pair_index`` 的集合/计数/端点。期望：集合完全
    一致……若不一致，说明 forward 视图缺信息或重构规则错，必须阻断。审计代码
    只读监督字段用于**比对**，不进入 context、不进入模型".

    This function is the ONLY place in the V543 model layer that accepts a
    supervision tensor, and it accepts it as a SEPARATE argument — the Bridge
    call itself (:meth:`LatentBridge.reconstruct_changed_pairs`) never sees
    it, which is exactly the structural separation the plan demands.

    Returns an audit record (``schema`` = :data:`PAIR_AUDIT_SCHEMA`) and raises
    ``V543_BRIDGE_PAIR_AUDIT`` (a :class:`V543BridgeAuditError`) when the
    sets/counts differ.
    """
    if not isinstance(supervised_pair_index, torch.Tensor):
        raise V543BridgeAuditError(
            "V543_BRIDGE_PAIR_AUDIT",
            f"supervised_pair_index must be a tensor, got "
            f"{type(supervised_pair_index).__name__}")
    if supervised_pair_index.ndim != 2 or \
            int(supervised_pair_index.shape[0]) != 2:
        raise V543BridgeAuditError(
            "V543_BRIDGE_PAIR_AUDIT",
            f"supervised_pair_index must be [2,P], got "
            f"{list(supervised_pair_index.shape)}")

    pairs = bridge.reconstruct_changed_pairs(forward_view)
    ours = sorted({
        (int(min(a, b)), int(max(a, b)))
        for a, b in pairs.pair_node_index.t().detach().cpu().tolist()
    })
    theirs = sorted({
        (int(min(a, b)), int(max(a, b)))
        for a, b in supervised_pair_index.t().detach().cpu().tolist()
    })
    ours_set, theirs_set = set(ours), set(theirs)
    missing = sorted(theirs_set - ours_set)
    extra = sorted(ours_set - theirs_set)
    ours_endpoints = {node for pair in ours_set for node in pair}
    theirs_endpoints = {node for pair in theirs_set for node in pair}
    record: dict[str, object] = {
        "schema": PAIR_AUDIT_SCHEMA,
        "n_bridge_pairs": len(ours),
        "n_supervised_pairs": len(theirs),
        "n_shared": len(ours_set & theirs_set),
        "n_missing_from_bridge": len(missing),
        "n_extra_in_bridge": len(extra),
        "endpoints_match": ours_endpoints == theirs_endpoints,
        "missing_examples": missing[:max_examples],
        "extra_examples": extra[:max_examples],
        "bridge_stats": dict(pairs.stats),
        "bridge_spec_sha256": BRIDGE_SPEC_SHA256,
        "supervision_read": "compare_only",
    }
    if missing or extra:
        raise V543BridgeAuditError(
            "V543_BRIDGE_PAIR_AUDIT",
            "the Bridge's reconstructed changed-pair set differs from the "
            f"supervised pair_index: {len(missing)} missing (e.g. "
            f"{missing[:max_examples]}), {len(extra)} extra (e.g. "
            f"{extra[:max_examples]}); the forward view is missing "
            "information or the reconstruction rule is wrong (plan §5.6 :434)")
    return record


__all__ = [
    "BRIDGE_SPEC_SCHEMA",
    "BRIDGE_SPEC_SHA256",
    "BRIDGE_VERSION",
    "CACHE_KEY_FIELDS",
    "ChangedPairSet",
    "KEY_EDGE_TYPE_COVALENT",
    "KEY_EDGE_TYPE_SPATIAL",
    "KEY_GRAPH_CUTOFF_ANGSTROM",
    "KEY_GRAPH_MAX_NEIGHBORS",
    "KEY_GRAPH_RBF_CENTERS",
    "KEY_V_CHANNEL_NAMES",
    "KEY_V_DISTANCE_NORM",
    "KEY_V_SPATIAL_MU",
    "KEY_V_SPATIAL_SIGMA",
    "LatentBridge",
    "PAIR_AUDIT_SCHEMA",
    "PAIR_CV_BROADCAST_SLOTS",
    "PAIR_RBF_CENTERS",
    "PAIR_RBF_SIGMA",
    "PAIR_SKETCH_ROWS",
    "PAIR_S_LAYOUT",
    "REQUIRED_FORWARD_KEYS",
    "UNIT_DENOM_EPS",
    "audit_pair_sets",
    "bridge_spec_sha256",
    "build_context",
    "cache_key",
    "fixed_sketch_matrix",
    "source_frame_identity",
]
