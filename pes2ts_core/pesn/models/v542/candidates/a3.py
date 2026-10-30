"""V5.4.2 candidate A3 — A1 + last-layer Reaction Query attention
(ADR-0033 §2.3; config ``backbone_contract.backbone_registry`` A3 row;
Oracle architecture §2.3).

A3 is EXACTLY A1 (same 2 layers, same per-layer zero-init ``W_d``, same
mapping set, same ``z`` recipe) plus one last-layer per-edge reaction
query attention block with **2 heads × 24 dim**:

* graph-level query ``q_g = tanh(Linear(53→48)([mean_pool(h_0),
  cv_target_delta(4), signed_intent(1)]))`` — current state + legal
  directed reaction intent only; no labels, no history;
* per-edge, per-head logits ``logit_ij^h = (W_q^h q_g)·(W_k^h
  m_ij)/sqrt(24)`` (invariants ONLY — scalars, never absolute
  coordinates or vector components), segment softmax over the IN-edges
  of each receiver;
* ``edge_msg_ij = m_ij + W_o(attn_norm(concat_h a_ij^h · (W_v^h m_ij)))``
  — the last-layer node update aggregates this SAME ``edge_msg`` (not
  the raw ``m_ij``), so the query affects ``h``/``z`` AND MOVE;
* ``W_o`` is ZERO-initialised: at init the attention branch contributes
  exactly zero, A3 forward == A1 forward (bitwise in fp32 — proven in
  ``tests/pesn/test_v542_a3.py``), so the R3 warm-start is clean and
  A3−A1 isolates the query operator.

Shapes/params (exact): ``query_g`` Linear(53→48) 2,592;
``w_q``/``w_k``/``w_value`` Linear(48→48) 2,352 each (= 2 heads ×
Linear(48→24) 1,176, the 2×24 split is a reshape of the 48-wide
projection); ``w_o`` Linear(48→48) 2,352 (zero-init); ``attn_norm``
LayerNorm(48) 96 — attention total 12,096 on top of A1 41,232 =
**53,328** active (config estimate, asserted EXACTLY at construction).

``attn_norm`` sits on the concatenated head outputs immediately before
``W_o`` (the Oracle freezes the 96 LayerNorm params but not their seat;
scalars-only LayerNorm keeps E(3) invariance and ``W_o=0`` makes the
whole branch inert at init either way).  Nodes with NO in-edges never
appear as receivers (the edge set is the receiver's row index), so the
segment softmax is always well-defined; an empty edge batch passes
through untouched.  ``q_g``/logits/values/outputs are all invariant
scalars — E(3)-invariant by construction; no coordinate updates, no
dropout/batchnorm, fp32, no torch.compile.

Factory discipline is identical to A1 (:func:`~pes2ts_core.pesn.models.
v542.candidates.a1.build`): frozen construction protocol inside
``construction_scope`` → A1's 21-key R3 mapping (strict audit) →
zero-init ``W_d`` AND ``W_o`` → shared ``w_v`` wiring → exact budget
assertion; the caller runs ``finish_training_seed``."""

from __future__ import annotations

import math
from typing import Final, Mapping

import torch
import torch.nn as nn

from pes2ts_core.pesn.contracts.key_atom_v3 import CV_MAX
from pes2ts_core.pesn.models.v542.decoder import DirectMoveDecoder
from pes2ts_core.pesn.models.v542.interfaces import (
    HIDDEN_V542,
    N_DECODER_PARAMS_V542,
    V542InitError,
)
from pes2ts_core.pesn.models.v542.candidates.a1 import (
    A1Backbone,
    r3_mapping_table,
    _consumed_source_keys,
)
from pes2ts_core.pesn.models.v542.candidates.common import (
    STEP_FRESH_MODULES,
    STEP_R3_MAPPING,
    STEP_SHARED_W_V,
    STEP_ZERO_INIT,
    BuiltCandidate,
    apply_r3_mapping,
    apply_shared_w_v,
    assert_candidate_budget,
    construction_scope,
    load_r3_backbone_state,
    resolve_init_source_id,
    zero_init_,
)

#: Registered A3 backbone version (campaign family
#: ``egnn_reaction_query_2x24``).
A3_BACKBONE_VERSION: Final[str] = "pesn_key_atom_v542_a3_v1"

#: Frozen candidate id.
CANDIDATE_ID: Final[str] = "A3"

#: Exact active-parameter count (config estimate, asserted EXACTLY at
#: construction: A1 41,232 + query attention 12,096).
ESTIMATED_ACTIVE_PARAMS: Final[int] = 53_328

#: The canonical frozen construction steps THIS builder executes (config
#: ``init_policy.construction_order`` 1–4: fresh modules, R3 mapping,
#: W_d/W_o zero-init, shared ``w_v`` from R3).  Steps 5–6 are
#: builder-delegated (:data:`common.BUILDER_DELEGATED_STEPS`); the
#: budget gate is a contract check, NOT a protocol step.
EXECUTED_STEPS: Final[tuple[str, ...]] = (
    STEP_FRESH_MODULES,
    STEP_R3_MAPPING,
    STEP_ZERO_INIT,
    STEP_SHARED_W_V,
)

#: Frozen attention geometry (config ``query: {heads: 2, head_dim: 24}``).
N_HEADS: Final[int] = 2
HEAD_DIM: Final[int] = 24
#: Frozen query input width: mean_pool(h_0) 48 ⊕ cv_target_delta 4 ⊕
#: signed_intent 1.
QUERY_IN_DIM: Final[int] = HIDDEN_V542 + CV_MAX + 1   # 53


def segment_softmax_rows(
        logits: torch.Tensor,
        index: torch.Tensor,
        n_segments: int,
) -> torch.Tensor:
    """Deterministic segment softmax: ``logits [E, H]`` grouped by
    ``index [E]`` (receiver node rows), softmax WITHIN each group.

    Max-subtraction is per segment (``scatter_reduce`` amax with an
    ``-inf`` seed and ``include_self=False``), so untouched segments stay
    ``-inf`` and are never gathered; every receiver of a real edge
    attends to at least one in-edge (its own), hence ``exp`` sums are
    ≥ 1 — no division by zero, no NaN, even with mixed signs.  Empty
    ``logits`` (no edges) pass through unchanged.
    """
    if int(logits.numel()) == 0:
        return logits
    seg_max = torch.full(
        (int(n_segments), int(logits.shape[1])), float("-inf"),
        dtype=logits.dtype, device=logits.device)
    seg_max = seg_max.scatter_reduce(
        0, index.unsqueeze(-1).expand_as(logits), logits,
        reduce="amax", include_self=False)
    weights = torch.exp(logits - seg_max[index])
    denom = torch.zeros(
        n_segments, logits.shape[1], dtype=logits.dtype,
        device=logits.device).index_add(0, index, weights)
    return weights / denom[index]


class A3Backbone(A1Backbone):
    """A1 (2 layers) + last-layer 2×24 reaction query attention.

    Construction consumes the RNG stream AFTER every A1 module, so under
    the same seed the shared A1 slots are byte-identical to
    :class:`A1Backbone`'s; the attention modules
    (``query_g``/``w_q``/``w_k``/``w_value``/``w_o``/``attn_norm``) are
    all fresh (the frozen A3 mapping set is A1's — "A3 query 全 fresh"),
    and ``w_o`` is zero-initialised by the factory.
    """

    #: Registered backbone version.
    v542_version: Final[str] = A3_BACKBONE_VERSION

    def __init__(self, move_decoder: DirectMoveDecoder) -> None:
        super().__init__(move_decoder, layers=2)
        # graph-level reaction query: [mean_pool(h_0) ⊕ graph_cond] → 48
        self.query_g = nn.Linear(QUERY_IN_DIM, self.hidden)
        # 2 heads × 24 = 48-wide projections (per-head Linear(48→24)
        # ≡ one Linear(48→48) reshaped; 2×1,176 = 2,352 params each)
        self.w_q = nn.Linear(self.hidden, self.hidden)
        self.w_k = nn.Linear(self.hidden, self.hidden)
        self.w_value = nn.Linear(self.hidden, self.hidden)
        # ZERO-initialised output projection (config
        # per_candidate_zero_init.A3) — A3 at init == A1.
        self.w_o = nn.Linear(self.hidden, self.hidden)
        self.attn_norm = nn.LayerNorm(self.hidden)

        total = sum(int(p.numel()) for p in self.parameters())
        if total != ESTIMATED_ACTIVE_PARAMS + N_DECODER_PARAMS_V542:
            raise V542InitError(
                "V542_A3_PARAMS",
                f"A3Backbone has {total} params, expected "
                f"{ESTIMATED_ACTIVE_PARAMS} active + "
                f"{N_DECODER_PARAMS_V542} shared w_v")

    # -- last-layer reaction query attention ---------------------------------
    def _final_edge_messages(
            self,
            messages: torch.Tensor,
            *,
            h0: torch.Tensor,
            graph_cond: torch.Tensor,
            graph_index: torch.Tensor,
            dst: torch.Tensor,
            n_nodes: int,
    ) -> torch.Tensor:
        """A1's final message + per-edge 2×24 query attention.

        ``W_o`` is zero-initialised, so the added branch is exactly zero
        at init and this equals ``messages`` (A3 == A1 warm-start).
        """
        n_graphs = int(graph_cond.shape[0])
        # graph-level query from the PRE-layer state h_0 (invariant)
        g_sum = torch.zeros(
            n_graphs, self.hidden, dtype=messages.dtype,
            device=messages.device)
        g_sum = g_sum.index_add(0, graph_index, h0)
        g_count = torch.zeros(
            n_graphs, dtype=messages.dtype, device=messages.device
        ).index_add(0, graph_index,
                    torch.ones_like(graph_index, dtype=messages.dtype))
        g_mean = g_sum / g_count.clamp(min=1.0).unsqueeze(-1)
        q_g = torch.tanh(
            self.query_g(torch.cat([g_mean, graph_cond], dim=-1)))  # [G,48]

        n_edges = int(messages.shape[0])
        if n_edges == 0:
            return messages                     # 空入边安全: no NaN, no ops
        # query of each edge's RECEIVER graph (src/dst share a graph)
        q_e = q_g[graph_index[dst]]                                    # [E,48]
        q = self.w_q(q_e).view(n_edges, N_HEADS, HEAD_DIM)
        k = self.w_k(messages).view(n_edges, N_HEADS, HEAD_DIM)
        v = self.w_value(messages).view(n_edges, N_HEADS, HEAD_DIM)
        logits = (q * k).sum(dim=-1) / math.sqrt(float(HEAD_DIM))      # [E,H]
        attn = segment_softmax_rows(logits, dst, n_nodes)              # [E,H]
        concat = (attn.unsqueeze(-1) * v).reshape(n_edges, N_HEADS * HEAD_DIM)
        return messages + self.w_o(self.attn_norm(concat))


def build_record(
        run_seed: int,
        move_decoder: DirectMoveDecoder,
        *,
        r3_state: Mapping[str, torch.Tensor] | None = None,
) -> BuiltCandidate:
    """Frozen construction protocol for A3; returns the audit-bearing
    :class:`BuiltCandidate` record.

    Identical discipline to A1's factory: fresh modules under
    ``construction_scope(run_seed, "A3")`` → A1's frozen 21-key R3
    mapping with strict audit → zero-init every ``W_d`` AND ``W_o`` →
    :func:`apply_shared_w_v` with the SAME resolved ``source`` the
    mapping used (delegating to
    :func:`~pes2ts_core.pesn.models.v542.candidates.common.apply_r3_w_v`)
    → decoder ``w_v`` ``torch.equal`` check → exact budget assertion
    (53,328) and the 70k hard cap.  Does NOT reset the training seed
    (the caller runs ``finish_training_seed``).
    """
    init_source_id = resolve_init_source_id(CANDIDATE_ID)   # "A3"
    source = (load_r3_backbone_state() if r3_state is None
              else dict(r3_state))

    with construction_scope(run_seed, init_source_id) as seed:
        backbone = A3Backbone(move_decoder)
        mapping = r3_mapping_table(source)
        consumed = _consumed_source_keys(source, mapping)
        audit = apply_r3_mapping(backbone, source, mapping,
                                 consumed_source_keys=consumed)
        for slot in backbone.w_d:                 # A1 per-layer W_d
            zero_init_(slot)
        zero_init_(backbone.w_o)                  # A3: W_o zero-init
        apply_shared_w_v(backbone, move_decoder, source=source)

    # every build ends with a bitwise-equal decoder w_v (B1 / gate 2)
    if not (torch.equal(move_decoder.w_v.weight, source["w_v.weight"])
            and torch.equal(move_decoder.w_v.bias, source["w_v.bias"])):
        raise V542InitError(  # pragma: no cover - copy_ cannot diverge
            "V542_WV_NOT_EQUAL",
            "A3: decoder w_v is not torch.equal to the resolved source "
            "w_v after construction step 4")

    assert_candidate_budget(backbone, expected_active=ESTIMATED_ACTIVE_PARAMS)
    return BuiltCandidate(
        candidate_id=CANDIDATE_ID,
        backbone=backbone,
        move_decoder=move_decoder,
        mapping_audit=audit,
        seed_init=seed,
        init_source_id=init_source_id,
        executed_steps=EXECUTED_STEPS,
    )


def build(
        run_seed: int,
        move_decoder: DirectMoveDecoder,
        *,
        r3_state: Mapping[str, torch.Tensor] | None = None,
) -> A3Backbone:
    """Convenience wrapper over :func:`build_record` returning ONLY the
    backbone (the audit-bearing record stays available for the registry
    / P3 run-identity path)."""
    return build_record(run_seed, move_decoder, r3_state=r3_state).backbone


__all__ = [
    "A3_BACKBONE_VERSION",
    "A3Backbone",
    "CANDIDATE_ID",
    "ESTIMATED_ACTIVE_PARAMS",
    "EXECUTED_STEPS",
    "HEAD_DIM",
    "N_HEADS",
    "QUERY_IN_DIM",
    "build",
    "build_record",
    "segment_softmax_rows",
]
