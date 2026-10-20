"""R3 v3 key-atom contracts: reaction-intent fields + FIXED F3 slot superset.

Schema ``pesn_key_atom_v3`` / feature spec ``pesn_r3_feature_spec_v3``.
Normative sources (read-only):

* ``config/campaigns/pes2ts_30h_20261020.yaml`` ``feature_contract`` +
  ``r3_training`` blocks (FROZEN; PD-2/PD-6/PD-9),
* ``docs/contracts/PES2TS_ReactionIntent_Contract_v1.md`` (G_rxn, relabel vs
  reverse, forward-input legality §5),
* ``docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md``
  §3 (feature channels/ladder), §7 (file layout),
* ``docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md``
  §3.3 (Feature Gate), §4.1 (B0-F loss).

Design rules enforced here (never bend these — report a blocker instead):

* **Slot superset discipline**: ONE fixed F3-wide shape.  F0/F1/F2 arms are
  produced by ZERO-MASKING slots (:func:`level_mask_v3`); no parameter, slot
  or width may differ between arms.
* **No branch**: ``branch``/``branch_id`` and any history-side channel (X0,
  anchor, IRC position, TS coords, future frames) are structurally absent —
  v3 features consume the CURRENT frame plus the directed Reaction Intent
  only (ReactionIntent Contract v1 §5).
* **Intent-off (PD-6)** masks exactly the five leak classes and PRESERVES
  legal undirected chemical topology (:data:`INTENT_OFF_CLASSES_V3` /
  :data:`PRESERVED_UNORDERED_KEYS_V3`).
* **Display names are metadata**: ``endpoint_names`` participates only in
  the relabel-invariance gate; no numeric feature may read it.

Batch convention: batches are plain ``dict[str, Tensor]`` (harness-native).
Forward-visible keys are exactly :data:`FORWARD_VIEW_KEYS` (a subset of the
campaign ``feature_contract`` allowlist); supervision keys
(:data:`SUPERVISION_KEYS`) are training-side only and must never reach
``model.forward``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final, Iterable, Mapping, Sequence

SCHEMA_VERSION: Final[str] = "pesn_key_atom_v3"
FEATURE_SPEC_SCHEMA: Final[str] = "pesn_r3_feature_spec_v3"
MODEL_VERSION: Final[str] = "pesn_key_atom_star_egnn_v3"
LOSS_SCHEMA_VERSION: Final[str] = "pesn_key_atom_loss_v3"

#: FROZEN d0 of the main task (campaign ``data_contract.d0_primary_angstrom``).
D0_PRIMARY_ANGSTROM: Final[float] = 0.08


class KeyAtomV3ContractError(ValueError):
    """Typed refusal of the v3 contract layer."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# FIXED F3 slot superset
# ---------------------------------------------------------------------------
# Node-level forward keys (per-atom, shape [N] unless noted).
#: Element identity (model-side ``nn.Embedding(119, 16)`` consumes this).
NODE_Z_KEY: Final[str] = "z"
#: Current-frame coordinates [N,3] (raw Å; the model only ever consumes
#: difference vectors, so translation invariance is by construction).
NODE_POS_KEY: Final[str] = "pos"
NODE_GRAPH_KEY: Final[str] = "graph_index"
NODE_KEY_SET_KEY: Final[str] = "key_index"
#: F1: key-atom membership {0,1}.
NODE_IS_KEY_KEY: Final[str] = "is_key"
#: F1: RAW number of changed (intent) pairs incident on the atom — the loss
#: needs the raw count (``1/max(1,degK)`` frame weights), so features must
#: store it unnormalized.
NODE_DEGK_KEY: Final[str] = "degK"
#: F2 (PD-6 class 3): 3-channel multi-label intent role
#: [breaking, forming, spectator], derived from incident ``delta_b_intent``
#: (both breaking+forming may be 1 simultaneously; 30h plan §3.1).
NODE_INTENT_ROLE_KEY: Final[str] = "intent_role"

#: intent-role channel order (axis 1 of ``intent_role``).
INTENT_ROLE_CHANNELS: Final[tuple[str, ...]] = (
    "breaking", "forming", "spectator")

# Edge-level keys.
EDGE_INDEX_KEY: Final[str] = "edge_index"
#: Packed preserved (undirected) edge scalars — see ``EDGE_PACK_SLOTS_V3``.
EDGE_ATTR_KEY: Final[str] = "edge_attr"
#: PD-6 class 1: bond order in G_src / G_tgt (ordered source/target).
EDGE_B_SRC_KEY: Final[str] = "b_src"
EDGE_B_TGT_KEY: Final[str] = "b_tgt"
#: PD-6 class 2: signed intent edit ΔB_intent = B_tgt − B_src.
EDGE_DELTA_B_KEY: Final[str] = "delta_b_intent"
#: Preserved (PD-6) unsigned twins.
EDGE_ABS_DELTA_KEY: Final[str] = "abs_delta"
EDGE_IS_CHANGED_KEY: Final[str] = "is_changed"

#: Packed preserved edge slots inside ``edge_attr`` (F3 superset width).
#: F0/F1/F2 masks zero whole slots; the width never changes.
EDGE_PACK_SLOTS_V3: Final[dict[str, int]] = {
    "rbf12_0": 0,    # F3: Gaussian RBF bin 0 on the CURRENT distance
    "rbf12_1": 1,
    "rbf12_2": 2,
    "rbf12_3": 3,
    "rbf12_4": 4,
    "rbf12_5": 5,
    "rbf12_6": 6,
    "rbf12_7": 7,
    "rbf12_8": 8,
    "rbf12_9": 9,
    "rbf12_10": 10,
    "rbf12_11": 11,
    "is_spatial": 12,     # F0: radius-only spatial edge flag
    "covalent_union": 13,  # F0: bond present in G_src ∪ G_tgt
    "aromatic_union": 14,  # F0: aromatic bond in either endpoint graph
    "b_min": 15,          # F2 (unordered endpoint context): min(B_src,B_tgt)
    "b_max": 16,          # F2 (unordered endpoint context): max(B_src,B_tgt)
}
EDGE_PACK_DIM_V3: Final[int] = 17
#: RBF(12) occupies the first N_RBF_V3 packed slots (F3 only).
N_RBF_V3: Final[int] = 12
RBF_SLOT_SLICE_V3: Final[slice] = slice(0, N_RBF_V3)
EDGE_PACK_F0_SLOTS_V3: Final[tuple[str, ...]] = (
    "is_spatial", "covalent_union", "aromatic_union")
EDGE_PACK_F2_SLOTS_V3: Final[tuple[str, ...]] = ("b_min", "b_max")
EDGE_PACK_F3_SLOTS_V3: Final[tuple[str, ...]] = tuple(
    f"rbf12_{k}" for k in range(N_RBF_V3))

#: Model-side edge input width = packed + the five edge-aligned intent /
#: preserved scalars consumed as explicit top-level keys.
EDGE_FEATURE_KEYS_V3: Final[tuple[str, ...]] = (
    EDGE_B_SRC_KEY, EDGE_B_TGT_KEY, EDGE_DELTA_B_KEY,
    EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY)
EDGE_INPUT_DIM_V3: Final[int] = EDGE_PACK_DIM_V3 + len(EDGE_FEATURE_KEYS_V3)

# Graph-level (per-intent) forward keys.
#: Directed CV target increments Δξ [G, CV_MAX] (PD-6 class 4); entry
#: ``c`` is the target increment of CV ``c`` under the intent's rules.
GRAPH_CV_DELTA_KEY: Final[str] = "cv_target_delta"
#: Derived direction sign ∈ {-1,0,+1} [G] (PD-6 class 5): the sign of the
#: PRIMARY (first) directed CV target increment.  Deterministic function of
#: G_rxn's (𝒞, ℛ) — NOT an independent direction token (ReactionIntent
#: Contract v1 §3.1 forbids adding one; a derived value cannot conflict).
GRAPH_SIGNED_INTENT_KEY: Final[str] = "signed_intent"
CV_MAX: Final[int] = 4

#: Harness metadata only (relabel gate); no numeric feature reads it.
ENDPOINT_NAMES_KEY: Final[str] = "endpoint_names"

# --- feature ladder masks (plan §3.4; yaml feature_contract.slot_superset) --
LEVEL_F0: Final[str] = "F0"
LEVEL_F1: Final[str] = "F1"
LEVEL_F2: Final[str] = "F2"
LEVEL_F3: Final[str] = "F3"
FEATURE_LEVELS: Final[tuple[str, ...]] = (LEVEL_F0, LEVEL_F1, LEVEL_F2, LEVEL_F3)

# Explicit ladder (cumulative): F0 = elements+pos+basic topology;
# F1 = +is_key+degK; F2 = +intent fields (signed + unsigned); F3 = +RBF(12).
LEVEL_NODE_KEYS_V3: dict[str, tuple[str, ...]] = {
    LEVEL_F0: (NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY),
    LEVEL_F1: (NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
               NODE_IS_KEY_KEY, NODE_DEGK_KEY),
    LEVEL_F2: (NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
               NODE_IS_KEY_KEY, NODE_DEGK_KEY, NODE_INTENT_ROLE_KEY),
    LEVEL_F3: (NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
               NODE_IS_KEY_KEY, NODE_DEGK_KEY, NODE_INTENT_ROLE_KEY),
}
LEVEL_EDGE_KEYS_V3: dict[str, tuple[str, ...]] = {
    LEVEL_F0: (EDGE_INDEX_KEY, EDGE_ATTR_KEY),
    LEVEL_F1: (EDGE_INDEX_KEY, EDGE_ATTR_KEY),
    LEVEL_F2: (EDGE_INDEX_KEY, EDGE_ATTR_KEY, *EDGE_FEATURE_KEYS_V3),
    LEVEL_F3: (EDGE_INDEX_KEY, EDGE_ATTR_KEY, *EDGE_FEATURE_KEYS_V3),
}
LEVEL_GRAPH_KEYS_V3: dict[str, tuple[str, ...]] = {
    LEVEL_F0: (),
    LEVEL_F1: (),
    LEVEL_F2: (GRAPH_CV_DELTA_KEY, GRAPH_SIGNED_INTENT_KEY),
    LEVEL_F3: (GRAPH_CV_DELTA_KEY, GRAPH_SIGNED_INTENT_KEY),
}
#: Packed ``edge_attr`` slots enabled per level (everything else zeroed).
LEVEL_EDGE_PACK_V3: dict[str, tuple[str, ...]] = {
    LEVEL_F0: EDGE_PACK_F0_SLOTS_V3,
    LEVEL_F1: EDGE_PACK_F0_SLOTS_V3,
    LEVEL_F2: (*EDGE_PACK_F0_SLOTS_V3, *EDGE_PACK_F2_SLOTS_V3),
    LEVEL_F3: (*EDGE_PACK_F0_SLOTS_V3, *EDGE_PACK_F2_SLOTS_V3,
               *EDGE_PACK_F3_SLOTS_V3),
}


def edge_pack_slot_mask_v3(level: str):
    """[EDGE_PACK_DIM_V3] 0/1 mask of packed slots enabled at ``level``."""
    import numpy as np

    if level not in FEATURE_LEVELS:
        raise KeyAtomV3ContractError(
            "FEATURE_LEVEL", f"unknown level {level!r}; known {FEATURE_LEVELS}")
    mask = np.zeros(EDGE_PACK_DIM_V3, dtype=np.float64)
    for slot in LEVEL_EDGE_PACK_V3[level]:
        mask[EDGE_PACK_SLOTS_V3[slot]] = 1.0
    return mask


# ---------------------------------------------------------------------------
# PD-6 intent-off mask (five leak classes) + preserved context
# ---------------------------------------------------------------------------
#: The five PD-6 classes mapped onto the exact v3 forward keys they mask.
#: (yaml ``feature_contract.intent_off_mask``; PD-6 ruling 2026-10-20.)
INTENT_OFF_CLASSES_V3: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("ordered_source_target", (EDGE_B_SRC_KEY, EDGE_B_TGT_KEY)),
    ("signed_graph_edits", (EDGE_DELTA_B_KEY,)),
    ("intent_role", (NODE_INTENT_ROLE_KEY,)),
    ("directed_cv_descriptors", (GRAPH_CV_DELTA_KEY,)),
    ("implicit_direction_encodings", (GRAPH_SIGNED_INTENT_KEY,)),
)

#: Legal undirected chemical topology preserved under intent-off
#: (yaml ``feature_contract.preserved_off_context`` = PD-6).
PRESERVED_UNORDERED_KEYS_V3: Final[tuple[str, ...]] = (
    NODE_POS_KEY, NODE_Z_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
    NODE_IS_KEY_KEY, NODE_DEGK_KEY, EDGE_INDEX_KEY, EDGE_ATTR_KEY,
    EDGE_ABS_DELTA_KEY, EDGE_IS_CHANGED_KEY,
)

#: All five classes (the legal ``F2-intent-off`` ablation arm).
INTENT_OFF_ALL_V3: Final[tuple[str, ...]] = tuple(
    cls for cls, _ in INTENT_OFF_CLASSES_V3)


def intent_off_keys_v3(classes: Iterable[str]) -> tuple[str, ...]:
    """Forward keys masked when the given PD-6 classes are turned off."""
    wanted = set(classes)
    known = {cls for cls, _ in INTENT_OFF_CLASSES_V3}
    unknown = wanted - known
    if unknown:
        raise KeyAtomV3ContractError(
            "INTENT_OFF_CLASS", f"unknown class(es) {sorted(unknown)}; "
            f"known {sorted(known)}")
    keys: list[str] = []
    for cls, ks in INTENT_OFF_CLASSES_V3:
        if cls in wanted:
            keys.extend(ks)
    return tuple(keys)


# ---------------------------------------------------------------------------
# Forward view / supervision split (input legality, ReactionIntent §5)
# ---------------------------------------------------------------------------
#: Every key a model forward may ever see.  Must stay a subset of the
#: campaign ``feature_contract`` forward-input allowlist (machine-checked in
#: ``tests/pesn/test_r3_input_legality.py`` and by the feature-gate harness).
FORWARD_VIEW_KEYS_V3: Final[tuple[str, ...]] = (
    NODE_Z_KEY, NODE_POS_KEY, NODE_GRAPH_KEY, NODE_KEY_SET_KEY,
    NODE_IS_KEY_KEY, NODE_DEGK_KEY, NODE_INTENT_ROLE_KEY,
    EDGE_INDEX_KEY, EDGE_ATTR_KEY, *EDGE_FEATURE_KEYS_V3,
    GRAPH_CV_DELTA_KEY, GRAPH_SIGNED_INTENT_KEY, ENDPOINT_NAMES_KEY,
)

#: Training-side keys attached by the dataset/collate; NEVER model inputs.
SUPERVISION_KEYS_V3: Final[tuple[str, ...]] = (
    "y_teacher",          # [ΣK,3] first-contact displacement target (Å)
    "pair_index",         # [2,P] global rows of the key-pair set
    "pair_edit",          # [P] signed ΔB_intent per key pair
    "frame_weight",       # [G] label-quality weight, applied ONCE in the loss
    "frame_mask",         # [G] bool eligibility mask
    "d0",                 # [G] float32 supervision distance
    "sample_ids", "curve_ids", "intent_ids",
)

#: The forbidden input classes of the campaign yaml
#: ``feature_contract.forbidden_model_inputs`` (machine-checked list; used by
#: the input-legality whitelist tests to prove none is reachable).
FORBIDDEN_INPUT_CLASSES_V3: Final[tuple[str, tuple[str, ...]]] = (
    ("future_frames", ("future frames",)),
    ("ts_coords", ("TS coords",)),
    ("true_progress", ("true progress",)),
    ("remaining_arc", ("remaining arc",)),
    ("branch_id", ("branch id",)),
    ("split_family", ("split/family",)),
    ("energy_gradient", ("energy/gradient",)),
    ("near_ts_capped", ("near_ts/capped",)),
    ("endpoint_geometry", ("endpoint geometry",)),
    ("history_hidden_state", ("history/hidden state",)),
)


def forward_view_v3(batch: Mapping[str, object]) -> dict[str, object]:
    """Return only the forward-visible keys (drops supervision/metadata)."""
    return {k: batch[k] for k in FORWARD_VIEW_KEYS_V3 if k in batch}


def check_forward_view_keys_v3(batch: Mapping[str, object]) -> None:
    """Typed refusal when a non-forward key is present in a forward batch."""
    extra = sorted(set(batch) - set(FORWARD_VIEW_KEYS_V3))
    if extra:
        raise KeyAtomV3ContractError(
            "FORWARD_VIEW_LEAK",
            f"batch carries non-forward keys {extra}; supervision keys must "
            f"stay outside model.forward (ReactionIntent Contract v1 §5)")


# ---------------------------------------------------------------------------
# Reaction-intent record (fields of reaction_intent.csv, PD-1 six-table)
# ---------------------------------------------------------------------------
def _as_bonds(payload: object, *, field_name: str) -> tuple[tuple[int, int, float], ...]:
    """Parse a G_src/G_tgt bond payload into sorted ``(i, j, order)`` triples.

    Accepted encodings (E1 ``reaction_intent.csv`` and test fixtures):

    * JSON list ``[[i, j, order], ...]``;
    * JSON dict ``{"atoms": [Z...], "bonds": [[i, j, order], ...]}`` (the
    E1 six-table delivery — ``atoms`` carries per-row atomic numbers);
    * an already-parsed sequence/dict of the above.  Aromatic bond order
    1.5 passes through; pairs are stored with ``i < j``.
    """
    if isinstance(payload, str):
        payload = json.loads(payload) if payload.strip() else []
    if isinstance(payload, Mapping):
        payload = payload.get("bonds", [])
    bonds: list[tuple[int, int, float]] = []
    try:
        for entry in payload:  # type: ignore[union-attr]
            i, j, order = int(entry[0]), int(entry[1]), float(entry[2])
            if i == j:
                raise KeyAtomV3ContractError(
                    "INTENT_BOND_SELFLOOP",
                    f"{field_name}: bond ({i},{j}) is a self loop")
            if i > j:
                i, j = j, i
            bonds.append((i, j, order))
    except (TypeError, ValueError, IndexError) as exc:
        raise KeyAtomV3ContractError(
            "INTENT_BOND_PARSE", f"{field_name}: {exc}") from exc
    return tuple(sorted(bonds))


def _as_graph_atoms(payload: object) -> tuple[int, ...]:
    """Atomic numbers from a G_src/G_tgt dict payload (``atoms`` key)."""
    if isinstance(payload, str):
        payload = json.loads(payload) if payload.strip() else {}
    if isinstance(payload, Mapping):
        atoms = payload.get("atoms")
        if atoms:
            return tuple(int(a) for a in atoms)
    return ()


@dataclass(frozen=True)
class IntentRecordV3:
    """Directed Reaction Intent ``G_rxn = (G_src, G_tgt, K, CV, rules)``.

    ``intent_id`` keys the ``reaction_intent`` table; ``reverse_of_intent_id``
    closes the involution pair (E1 gate T0.2).  Nothing here derives from a
    historical branch or from IRC geometry — only from the reaction task.
    """

    intent_id: str
    curve_id: str
    direction: str                       # "forward" | "reverse"
    reverse_of_intent_id: str
    intent_pair_id: str
    g_src_bonds: tuple[tuple[int, int, float], ...]
    g_tgt_bonds: tuple[tuple[int, int, float], ...]
    cv_deltas: tuple[float, ...]         # directed Δξ, len <= CV_MAX
    intent_version: str = "v3"
    intent_underdetermined: bool = False
    #: per-row atomic numbers when the endpoint graphs carry an ``atoms``
    #: payload (E1 six-table delivery).
    g_atoms: tuple[int, ...] = ()
    #: CV atom-index pairs from ``cv_spec`` (``coordinates`` key).
    cv_coordinates: tuple[tuple[int, int], ...] = ()
    cv_kind: str = ""

    @classmethod
    def from_table_row(cls, row: Mapping[str, object]) -> "IntentRecordV3":
        """Build from one ``reaction_intent.csv`` row (string values).

        ``g_src``/``g_tgt`` accept the E1 dict encoding
        ``{"atoms": [Z...], "bonds": [[i, j, order], ...]}`` as well as a
        plain bond-list payload.  ``cv_spec`` may carry explicit directed
        ``target_deltas``; when it does not (the E1 six-table delivery of
        ``kind="graph_edit_bond_lengths"`` stores only the coordinate
        pairs), the directed CV increments are DERIVED from the intent's
        own signed edits with the contract §3.1 direction rule for
        bond-length CVs — breaking (ΔB<0) ⇒ distance target grows
        (+), forming (ΔB>0) ⇒ distance target shrinks (−), i.e.
        ``Δξ_ij = −ΔB_intent_ij`` for the listed coordinates.  The
        derivation is a deterministic projection of G_rxn itself, so
        relabel invariance and reverse sensitivity both hold.
        """
        g_src_payload = row.get("g_src", "[]")
        g_tgt_payload = row.get("g_tgt", "[]")
        cv_payload = row.get("cv_spec", "[]")
        if isinstance(cv_payload, str):
            cv_doc = json.loads(cv_payload) if cv_payload.strip() else []
        else:
            cv_payload = cv_payload
            cv_doc = cv_payload
        deltas: tuple[float, ...]
        coords: tuple[tuple[int, int], ...] = ()
        cv_kind = ""
        if isinstance(cv_doc, Mapping):
            explicit = cv_doc.get("target_deltas", cv_doc.get("deltas"))
            raw_coords = cv_doc.get("coordinates", ())
            coords = tuple(
                (int(c[0]), int(c[1])) for c in raw_coords)
            cv_kind = str(cv_doc.get("kind", ""))
            if explicit is not None:
                deltas = tuple(float(d) for d in explicit)
            else:
                deltas = ()
        else:
            deltas = tuple(float(d) for d in cv_doc)
        g_src_bonds = _as_bonds(g_src_payload, field_name="g_src")
        g_tgt_bonds = _as_bonds(g_tgt_payload, field_name="g_tgt")
        if not deltas and coords:
            orders: dict[tuple[int, int], tuple[float, float]] = {}
            for i, j, order in g_src_bonds:
                orders[(i, j)] = (order, 0.0)
            for i, j, order in g_tgt_bonds:
                b_src, _ = orders.get((i, j), (0.0, 0.0))
                orders[(i, j)] = (b_src, order)
            deltas_list: list[float] = []
            for i, j in coords[:CV_MAX]:
                a, b = (i, j) if i < j else (j, i)
                b_src, b_tgt = orders.get((a, b), (0.0, 0.0))
                deltas_list.append(-(b_tgt - b_src))   # contract §3.1 rule
            deltas = tuple(deltas_list)
        if len(deltas) > CV_MAX:
            raise KeyAtomV3ContractError(
                "CV_WIDTH", f"cv_spec has {len(deltas)} deltas > CV_MAX={CV_MAX}")
        direction = str(row.get("direction", ""))
        if direction not in ("forward", "reverse"):
            raise KeyAtomV3ContractError(
                "INTENT_DIRECTION", f"direction must be forward|reverse, got {direction!r}")
        g_atoms = _as_graph_atoms(g_src_payload) or _as_graph_atoms(g_tgt_payload)
        return cls(
            intent_id=str(row["intent_id"]),
            curve_id=str(row["curve_id"]),
            direction=direction,
            reverse_of_intent_id=str(row.get("reverse_of_intent_id", "")),
            intent_pair_id=str(row.get("intent_pair_id", "")),
            g_src_bonds=g_src_bonds,
            g_tgt_bonds=g_tgt_bonds,
            cv_deltas=deltas,
            intent_version=str(row.get("intent_version", "v3")),
            intent_underdetermined=str(
                row.get("intent_underdetermined", "false")).lower() in ("true", "1"),
            g_atoms=g_atoms,
            cv_coordinates=coords,
            cv_kind=cv_kind,
        )

    def pair_orders(self) -> dict[tuple[int, int], tuple[float, float]]:
        """``{(i,j): (B_src, B_tgt)}`` over the union of endpoint bonds."""
        out: dict[tuple[int, int], tuple[float, float]] = {}
        for i, j, order in self.g_src_bonds:
            out[(i, j)] = (order, 0.0)
        for i, j, order in self.g_tgt_bonds:
            b_src, _ = out.get((i, j), (0.0, 0.0))
            out[(i, j)] = (b_src, order)
        return out

    def delta_b(self, i: int, j: int) -> float:
        a, b = (i, j) if i < j else (j, i)
        b_src, b_tgt = self.pair_orders().get((a, b), (0.0, 0.0))
        return b_tgt - b_src

    def key_rows(self) -> tuple[int, ...]:
        """Deterministic K set: changed-bond atoms ∪ CV-spec atoms.

        (30h plan §3.1: 合法 K 由成断键/明确 CV 涉及原子得到 — no branch,
        no IRC-geometry recovery.)
        """
        keys: set[int] = set()
        for (i, j), (b_src, b_tgt) in self.pair_orders().items():
            if b_src != b_tgt:
                keys.add(i)
                keys.add(j)
        for idx in self.cv_atom_indices():
            keys.add(idx)
        return tuple(sorted(keys))

    def cv_atom_indices(self) -> tuple[int, ...]:
        """Atoms referenced by the CV coordinate pairs (when supplied)."""
        atoms: set[int] = set()
        for i, j in self.cv_coordinates:
            atoms.add(int(i))
            atoms.add(int(j))
        return tuple(sorted(atoms))

    def signed_intent_sign(self) -> float:
        """PD-6 class-5 carrier: sign of the PRIMARY directed CV increment."""
        if not self.cv_deltas:
            return 0.0
        primary = float(self.cv_deltas[0])
        if primary > 0.0:
            return 1.0
        if primary < 0.0:
            return -1.0
        return 0.0

    def reversed_intent(self) -> "IntentRecordV3":
        """``Reverse(G_rxn)``: swap G_src/G_tgt, negate directed CV deltas.

        The intent id becomes the paired reverse id (label involution
        ``reverse(reverse(x)) == x`` is asserted by the data gate T0.2 and
        at feature level by ``tests/pesn/test_r3_relabel_invariance.py``).
        """
        return IntentRecordV3(
            intent_id=self.reverse_of_intent_id or f"{self.intent_id}::rev",
            curve_id=self.curve_id,
            direction=("reverse" if self.direction == "forward" else "forward"),
            reverse_of_intent_id=self.intent_id,
            intent_pair_id=self.intent_pair_id,
            g_src_bonds=self.g_tgt_bonds,
            g_tgt_bonds=self.g_src_bonds,
            cv_deltas=tuple(-d for d in self.cv_deltas),
            intent_version=self.intent_version,
            intent_underdetermined=self.intent_underdetermined,
            g_atoms=self.g_atoms,
            cv_coordinates=self.cv_coordinates,
            cv_kind=self.cv_kind,
        )


@dataclass(frozen=True)
class KeyAtomSampleV3:
    """One buildable v3 sample: current frame + directed intent + metadata.

    ``xyz`` is the CURRENT geometry only (no X0, no TS, no future frame);
    ``endpoint_names`` are display names for the relabel gate and are never
    read by any numeric feature.
    """

    curve_id: str
    node_index: int
    intent: IntentRecordV3
    atomic_numbers: tuple[int, ...]
    xyz: tuple[tuple[float, float, float], ...]
    d0: float = D0_PRIMARY_ANGSTROM
    endpoint_names: tuple[str, str] = ("A", "B")
    #: supervision (attached by the dataset; never a model input)
    y_teacher: tuple[tuple[float, float, float], ...] | None = None
    frame_weight: float = 1.0
    label_status: str = "ok"
    sample_id: str = ""
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.atomic_numbers)
        if n == 0 or len(self.xyz) != n:
            raise KeyAtomV3ContractError(
                "SAMPLE_SHAPE",
                f"atomic_numbers ({n}) and xyz ({len(self.xyz)}) must align")
        keys = self.intent.key_rows()
        if not keys:
            raise KeyAtomV3ContractError(
                "KEY_SET_EMPTY",
                f"intent {self.intent.intent_id} yields an empty K set")
        if any(not (0 <= k < n) for k in keys):
            raise KeyAtomV3ContractError(
                "KEY_SET_RANGE", "intent key rows outside 0..N-1")


__all__ = [
    "SCHEMA_VERSION", "FEATURE_SPEC_SCHEMA", "MODEL_VERSION",
    "LOSS_SCHEMA_VERSION", "D0_PRIMARY_ANGSTROM", "CV_MAX",
    "KeyAtomV3ContractError", "IntentRecordV3", "KeyAtomSampleV3",
    "NODE_Z_KEY", "NODE_POS_KEY", "NODE_GRAPH_KEY", "NODE_KEY_SET_KEY",
    "NODE_IS_KEY_KEY", "NODE_DEGK_KEY", "NODE_INTENT_ROLE_KEY",
    "INTENT_ROLE_CHANNELS", "EDGE_INDEX_KEY", "EDGE_ATTR_KEY",
    "EDGE_B_SRC_KEY", "EDGE_B_TGT_KEY", "EDGE_DELTA_B_KEY",
    "EDGE_ABS_DELTA_KEY", "EDGE_IS_CHANGED_KEY",
    "EDGE_PACK_SLOTS_V3", "EDGE_PACK_DIM_V3", "N_RBF_V3",
    "RBF_SLOT_SLICE_V3", "EDGE_INPUT_DIM_V3", "EDGE_FEATURE_KEYS_V3",
    "GRAPH_CV_DELTA_KEY", "GRAPH_SIGNED_INTENT_KEY", "ENDPOINT_NAMES_KEY",
    "FEATURE_LEVELS", "LEVEL_F0", "LEVEL_F1", "LEVEL_F2", "LEVEL_F3",
    "LEVEL_NODE_KEYS_V3", "LEVEL_EDGE_KEYS_V3", "LEVEL_GRAPH_KEYS_V3",
    "LEVEL_EDGE_PACK_V3", "edge_pack_slot_mask_v3",
    "INTENT_OFF_CLASSES_V3", "INTENT_OFF_ALL_V3",
    "PRESERVED_UNORDERED_KEYS_V3", "intent_off_keys_v3",
    "FORWARD_VIEW_KEYS_V3", "SUPERVISION_KEYS_V3",
    "FORBIDDEN_INPUT_CLASSES_V3",
    "forward_view_v3", "check_forward_view_keys_v3", "_as_bonds",
]
