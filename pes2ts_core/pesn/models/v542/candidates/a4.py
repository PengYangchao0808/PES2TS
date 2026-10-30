"""V5.4.2 candidate A4 — 2-layer sparse geometry transformer (WP2 lane
P2-3; ADR-0033 §2.3; config ``backbone_registry.candidates.A4``; Oracle
architecture §2.4).

Frozen recipe (``family: sparse_geometry_transformer_4x12``)::

    hidden 48 · 2 layers · 4 heads × 12 dim · one reaction token per
    graph · Star-shaped edge readout (118→48→48) · zero-init W_g(5→48)
    graph-conditioning injection · pre-LN blocks · FFN 48→96→48

Node tokens ``h_i^0 = node_proj(element⊕role features) +
W_g(graph_cond)[graph_index]`` (``W_g`` ZERO-initialised so the warm
start keeps the R3 ``h_0``); one reaction token per graph initialised as
``token_init(graph_cond)``.  Each layer runs SPARSE edge-index-driven
multi-head attention — never dense N²: the attention edge set of a batch
is exactly (1) the input message edges ``src→dst`` (a dst node attends
its in-edge sources), (2) every node attending its own graph's token,
and (3) every token attending all nodes of its graph.  Logits are
``q_i·k_j/sqrt(12) + b_h(rbf12_ij)`` with a per-head RBF12 bias network
``b_h: 12→4`` (zero-init) and one learned per-head token-edge bias;
segment softmax is implemented with ``index_add``/``scatter_reduce``
over the query index, so softmax segments NEVER cross a graph boundary
(no dense masks, no leakage).  Empty in-edge sets are structurally
impossible for nodes (every node attends its token) and a graph with no
edges cannot NaN.  Blocks are pre-LN with a ZERO-initialised attention
output projection and a ZERO-initialised FFN output projection, so at
init the whole transformer stack is the identity and the backbone
reduces exactly to ``node_proj h_0 + edge_readout + z_head`` on the R3
warm start.  Positions are NEVER consumed by the math (position-
preserving; geometry enters only through the invariant ``e22`` edge
input and its RBF12 prefix) and coordinates are never updated — E(3)
invariance of ``z``/``edge_msg`` holds by construction.

``edge_msg`` comes from the Star-shaped edge readout
``MLP_e([h_src^L, h_dst^L, e22]): 118→48→48`` evaluated in the INPUT
edge order (row-for-row aligned with ``batch["edge_index"]`` columns),
with a LIVE autograd graph.  ``z = tanh(z_head([graph_mean(h^L),
key_mean(h^L)]))`` — the reaction token is a virtual node that NEVER
enters atom pooling.

Parameter accounting (exact; ``estimated: true`` config row asserts
55,840 after construction)::

    elem_emb     1,904   Embedding(119,16)          ← R3 identity
    node_proj    1,056   Linear(21→48)              ← R3 identity
    z_head       6,208   Linear(96→64)              ← R3 identity
    w_g            288   Linear(5→48)   ZERO-INIT   fresh seeded
    token_init     288   Linear(5→48)                fresh seeded
    layers    2×19,016   qkv Linear(48→144) 7,056   fresh seeded
                          out_proj Linear(48→48) 2,352 ZERO-INIT
                          edge_bias Linear(12→4) 52 ZERO-INIT
                          token_bias [4] 4           fresh seeded
                          ln_attn LayerNorm(48) 96
                          ffn_in Linear(48→96) 4,704 fresh seeded
                          ffn_out Linear(96→48) 4,656 ZERO-INIT
                          ln_ffn LayerNorm(48) 96
    edge_readout 8,064   Sequential 118→48→48       ← R3 message_mlp.2
    ------------------------------------------------------------
    active      55,840   (shared decoder ``w_v`` 49 EXCLUDED; total
                          with the borrowed module = 55,889; joint =
                          55,840 + 49 + 2,178 = 58,067)

R3 semantic mapping (config ``init_policy.init_mapping.
r3_semantic_mapping.A4``): the A1 common set as hosted by A4's module
surface — ``elem_emb`` / ``node_proj`` / ``z_head`` identity — plus
``edge_readout`` ← R3 ``message_mlp.2`` (same-shape Star message MLP,
Oracle §2.6: message_mlp.* maps only onto same-shape same-semantic
Star-shaped MLPs; A4 hosts exactly one, the edge readout).  A4 has NO
``message_mlp``/``node_mlp`` slots, so the remaining R3 payload keys
(``message_mlp.0/1``, ``node_mlp.*``, ``norm_h.*``, ``w_v.*``) are
declared consumed/unmapped in :data:`A4_CONSUMED_SOURCE_KEYS` — the
strict audit of :func:`~pes2ts_core.pesn.models.v542.candidates.common
.apply_r3_mapping` accounts every one of the 37 payload keys exactly
once.  Zero-inits (config ``per_candidate_zero_init.A4`` + Oracle §2.4
layer recipe): attention output projection, FFN output projection,
``b_h`` RBF12 bias, and ``W_g``.

Contract: ``forward(batch) -> BackboneOutput(z [G,64], edge_msg
[E,48])``; forward-view keys only (``FORWARD_VIEW_KEYS_V3`` whitelist);
fp32; deterministic (no dropout / batchnorm / stochastic layers); no
coordinate updates; ``torch.compile`` forbidden (ADR-0033 §2.1) — this
module never calls it.  :func:`build` runs the frozen six-step
construction protocol inside ``common.construction_scope`` (fresh
modules on the ``seed_init`` stream → R3 mapping → special zero-inits →
shared ``w_v``) and asserts the exact 55,840 active-parameter count; it
deliberately does NOT reset the training seed (step 6 stays the
runner's job)."""

from __future__ import annotations

import math
from typing import Final, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_ABS_DELTA_KEY,
    EDGE_ATTR_KEY,
    EDGE_B_SRC_KEY,
    EDGE_B_TGT_KEY,
    EDGE_DELTA_B_KEY,
    EDGE_INDEX_KEY,
    EDGE_INPUT_DIM_V3,
    EDGE_IS_CHANGED_KEY,
    GRAPH_CV_DELTA_KEY,
    GRAPH_SIGNED_INTENT_KEY,
    N_RBF_V3,
    NODE_DEGK_KEY,
    NODE_GRAPH_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_IS_KEY_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    check_forward_view_keys_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import (
    ELEM_DIM,
    GRAPH_COND_DIM_V3,
    KeyAtomV3InvalidInputError,
    NODE_FEATURE_DIM_V3,
    Z_DIM_V3,
)
from pes2ts_core.pesn.models.v542.candidates import (
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
from pes2ts_core.pesn.models.v542.decoder import (
    DirectMoveDecoder,
    assert_decoder_a_ref_frozen,
)
from pes2ts_core.pesn.models.v542.interfaces import (
    HIDDEN_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
    V542InitError,
    V542ParamBudgetError,
    BackboneOutput,
)

#: Registered A4 backbone version (campaign backbone family A4).
A4_BACKBONE_VERSION: Final[str] = "pesn_key_atom_v542_a4_sparse_transformer_v1"

#: Frozen candidate id (config ``backbone_registry.candidates.A4``).
CANDIDATE_ID: Final[str] = "A4"
#: Frozen active-parameter estimate of the config row (exact after build).
ESTIMATED_ACTIVE_PARAMS: Final[int] = 55_840

#: Frozen transformer width schedule (Oracle §2.4).
A4_N_LAYERS: Final[int] = 2
A4_N_HEADS: Final[int] = 4
A4_HEAD_DIM: Final[int] = 12
#: FFN inner width (``FFN 48→96→48``).
A4_FFN_DIM: Final[int] = 96
#: 1/sqrt(head_dim) attention logit scale.
_A4_LOGIT_SCALE: Final[float] = 1.0 / math.sqrt(A4_HEAD_DIM)
#: Per-layer parameter count of the frozen block (design + measured).
A4_LAYER_PARAMS: Final[int] = 19_016

#: R3 semantic mapping (config ``r3_semantic_mapping.A4``): the A1
#: common set hosted by A4 (``elem_emb``/``node_proj``/``z_head``) plus
#: ``edge_readout`` ← R3 ``message_mlp.2`` (same-shape Star MLP).
A4_R3_MAPPING: Final[dict[str, str]] = {
    "elem_emb.weight": "elem_emb.weight",
    "node_proj.weight": "node_proj.weight",
    "node_proj.bias": "node_proj.bias",
    "z_head.weight": "z_head.weight",
    "z_head.bias": "z_head.bias",
    "edge_readout.0.weight": "message_mlp.2.0.weight",
    "edge_readout.0.bias": "message_mlp.2.0.bias",
    "edge_readout.2.weight": "message_mlp.2.2.weight",
    "edge_readout.2.bias": "message_mlp.2.2.bias",
}

#: R3 payload keys deliberately NOT mapped by A4 (no host slots in the
#: transformer surface) plus the decoder-owned ``w_v.*`` consumed by
#: construction step 4 (:func:`apply_shared_w_v` →
#: :func:`~pes2ts_core.pesn.models.v542.candidates.common.apply_r3_w_v`,
#: which runs AFTER the mapping audit on the SAME resolved source and
#: genuinely loads those tensors into the shared decoder).  Together
#: with the 9 mapping values these account for all 37 R3 backbone
#: payload keys exactly once.
A4_CONSUMED_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "message_mlp.0.0.weight", "message_mlp.0.0.bias",
    "message_mlp.0.2.weight", "message_mlp.0.2.bias",
    "message_mlp.1.0.weight", "message_mlp.1.0.bias",
    "message_mlp.1.2.weight", "message_mlp.1.2.bias",
    "node_mlp.0.0.weight", "node_mlp.0.0.bias",
    "node_mlp.0.2.weight", "node_mlp.0.2.bias",
    "node_mlp.1.0.weight", "node_mlp.1.0.bias",
    "node_mlp.1.2.weight", "node_mlp.1.2.bias",
    "node_mlp.2.0.weight", "node_mlp.2.0.bias",
    "node_mlp.2.2.weight", "node_mlp.2.2.bias",
    "norm_h.0.weight", "norm_h.0.bias",
    "norm_h.1.weight", "norm_h.1.bias",
    "norm_h.2.weight", "norm_h.2.bias",
    "w_v.weight", "w_v.bias",
)


# ---------------------------------------------------------------------------
# one pre-LN sparse-attention block (19,016 params; Oracle §2.4)
# ---------------------------------------------------------------------------
class _SparseAttnLayer(nn.Module):
    """One A4 block: sparse MHA (4×12) + FFN, pre-LN, zero-init outputs.

    Zero-initialised at build: ``out_proj`` (attention output
    projection), ``ffn_out`` (FFN output projection), ``edge_bias``
    (``b_h: 12→4`` per-head RBF12 logit bias).  ``token_bias`` (one
    learned logit bias per head for token↔node edges) is fresh-seeded.
    LayerNorms are SHARED between atom rows and the reaction-token row
    (the frozen 19,016-param block budget has exactly two LN(48)).
    """

    def __init__(self) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(HIDDEN_V542)
        # QKV in one Linear(48→144) — 7,056 params.
        self.qkv = nn.Linear(HIDDEN_V542, 3 * HIDDEN_V542)
        self.out_proj = nn.Linear(HIDDEN_V542, HIDDEN_V542)   # ZERO-INIT
        self.edge_bias = nn.Linear(N_RBF_V3, A4_N_HEADS)      # ZERO-INIT
        # one learned bias per head for token edges (fresh seeded)
        self.token_bias = nn.Parameter(torch.randn(A4_N_HEADS) * 0.02)
        self.ln_ffn = nn.LayerNorm(HIDDEN_V542)
        self.ffn_in = nn.Linear(HIDDEN_V542, A4_FFN_DIM)
        self.ffn_out = nn.Linear(A4_FFN_DIM, HIDDEN_V542)     # ZERO-INIT


# ---------------------------------------------------------------------------
# the A4 backbone (Backbone protocol: forward -> BackboneOutput)
# ---------------------------------------------------------------------------
class SparseGeometryTransformerA4(nn.Module):
    """2-layer sparse geometry transformer with a per-graph reaction
    token and a Star-shaped edge readout (see module docstring).

    The constructor freezes the width schedule ``(48, 2, 4, 12)``; any
    other configuration is a typed :class:`V542InitError`
    (``V542_A4_CONFIG``) — widths are part of the interface (ADR-0033
    §2.1) and may never drift per candidate.

    ``w_v`` is NOT owned here: :func:`build` borrows the decoder's
    module object (``backbone.w_v is move_decoder.w_v``) for state-dict
    continuity, and ``active_param_count`` excludes those 49
    decoder-owned params (A0 accounting discipline).
    """

    def __init__(
        self,
        *,
        hidden: int = HIDDEN_V542,
        n_layers: int = A4_N_LAYERS,
        n_heads: int = A4_N_HEADS,
        head_dim: int = A4_HEAD_DIM,
    ) -> None:
        super().__init__()
        if (hidden, n_layers, n_heads, head_dim) != (
                HIDDEN_V542, A4_N_LAYERS, A4_N_HEADS, A4_HEAD_DIM):
            raise V542InitError(
                "V542_A4_CONFIG",
                f"{A4_BACKBONE_VERSION} fixes hidden=48, n_layers=2, "
                f"heads=4, head_dim=12 (got {hidden}/{n_layers}/"
                f"{n_heads}/{head_dim})")
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)

        self.elem_emb = nn.Embedding(119, ELEM_DIM)      # Z=1..118; R3
        self.node_proj = nn.Linear(
            ELEM_DIM + NODE_FEATURE_DIM_V3, self.hidden)  # 21→48; R3
        self.w_g = nn.Linear(GRAPH_COND_DIM_V3, self.hidden)  # ZERO-INIT
        self.token_init = nn.Linear(GRAPH_COND_DIM_V3, self.hidden)
        self.layers = nn.ModuleList(
            [_SparseAttnLayer() for _ in range(self.n_layers)])
        # Star-shaped edge readout (118→48→48) ← R3 message_mlp.2.
        self.edge_readout = nn.Sequential(
            nn.Linear(self.hidden + self.hidden + EDGE_INPUT_DIM_V3,
                      self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden))
        self.z_head = nn.Linear(2 * self.hidden, Z_DIM_V3)
        # NOTE: the shared decoder ``w_v`` is attached by :func:`build`.

    # -- accounting ---------------------------------------------------------
    @property
    def active_param_count(self) -> int:
        """Exact trainable ACTIVE params, excluding the decoder-owned
        shared ``w_v`` (A0 accounting discipline; frozen 55,840)."""
        return sum(
            int(p.numel()) for name, p in self.named_parameters()
            if not name.startswith("w_v."))

    # -- input helpers (legacy v3 taxonomy) ---------------------------------
    @staticmethod
    def _require_finite(name: str, tensor: torch.Tensor) -> None:
        if not bool(torch.isfinite(tensor).all()):
            raise KeyAtomV3InvalidInputError(
                "NONFINITE_INPUT", f"{name} contains non-finite values")

    def _edge_input(self, batch: Mapping[str, torch.Tensor],
                    dtype: torch.dtype) -> torch.Tensor:
        """Concatenate the packed preserved slots with the intent scalars
        (verbatim the legacy ``StarEGNNV3._edge_input``; 22 dims)."""
        parts = [
            batch[EDGE_ATTR_KEY].to(dtype=dtype),
            batch[EDGE_B_SRC_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_B_TGT_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_DELTA_B_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_ABS_DELTA_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_IS_CHANGED_KEY].to(dtype=dtype).unsqueeze(-1),
        ]
        return torch.cat(parts, dim=-1)

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> BackboneOutput:
        """Forward-view batch → ``BackboneOutput(z [G,64], edge_msg
        [E,48])`` (fp32, deterministic, live autograd graph)."""
        dtype = self.node_proj.weight.dtype
        device = self.node_proj.weight.device

        z_atomic = batch[NODE_Z_KEY]
        # Input legality (ReactionIntent Contract v1 §5): only forward-
        # view keys may reach the model — supervision must stay outside.
        check_forward_view_keys_v3(batch)
        if int(z_atomic.numel()) and int(z_atomic.max()) > 118:
            raise KeyAtomV3InvalidInputError(
                "Z_OUT_OF_RANGE", "z > 118 (padding/unknown Z rejected)")
        if int(z_atomic.numel()) and int(z_atomic.min()) < 1:
            raise KeyAtomV3InvalidInputError(
                "Z_PADDING", "z < 1 (padding rows are not valid input)")

        pos = batch[NODE_POS_KEY].to(dtype=dtype)
        edge_index = batch[EDGE_INDEX_KEY].long()
        graph_index = batch[NODE_GRAPH_KEY].long()
        key_index = batch[NODE_KEY_SET_KEY].long()
        is_key_f = batch[NODE_IS_KEY_KEY].to(dtype=dtype)
        degk = batch[NODE_DEGK_KEY].to(dtype=dtype)
        role = batch[NODE_INTENT_ROLE_KEY].to(dtype=dtype)
        edge_in = self._edge_input(batch, dtype)
        cv_delta = batch[GRAPH_CV_DELTA_KEY].to(dtype=dtype)
        signed = batch[GRAPH_SIGNED_INTENT_KEY].to(dtype=dtype)

        # pos is legality-checked like the legacy taxonomy but NEVER
        # consumed by the math (position-preserving; geometry enters only
        # through e22/RBF12 — no absolute coordinates, no coord updates).
        for name, tensor in (
                ("pos", pos), ("edge_attr", edge_in), ("degK", degk),
                ("intent_role", role), ("cv_target_delta", cv_delta),
                ("signed_intent", signed)):
            self._require_finite(name, tensor)

        n_nodes = int(pos.shape[0])
        n_graphs = int(cv_delta.shape[0])
        src, dst = edge_index[0], edge_index[1]

        # node scalar init + zero-init graph-conditioning injection
        degk_norm = degk.clamp(min=0.0, max=4.0) / 4.0
        h = self.node_proj(torch.cat(
            [self.elem_emb(z_atomic.to(device=device)),
             torch.cat([is_key_f.unsqueeze(-1), degk_norm.unsqueeze(-1),
                        role], dim=-1)], dim=-1))
        graph_cond = torch.cat([cv_delta, signed.unsqueeze(-1)], dim=-1)
        h = h + self.w_g(graph_cond)[graph_index]

        # per-graph reaction token (virtual node; never atom-pooled)
        token = self.token_init(graph_cond)               # [G,48]
        hs = torch.cat([h, token], dim=0)                 # [N+G,48]
        n_ext = n_nodes + n_graphs

        # sparse attention edge set (NO dense N², NO cross-graph rows):
        #   rows 0..E-1        : dst attends its in-edge sources (src→dst)
        #   rows E..E+N-1      : node i attends its graph's token
        #   rows E+N..E+2N-1   : token(g) attends every node of graph g
        node_ids = torch.arange(n_nodes, dtype=torch.long, device=device)
        tok_slots = n_nodes + graph_index                 # [N] token row
        att_queries = torch.cat([dst, node_ids, tok_slots])
        att_keys = torch.cat([src, tok_slots, node_ids])
        rbf12 = edge_in[:, :N_RBF_V3]                    # [E,12] invariant

        for layer in self.layers:
            x = layer.ln_attn(hs)                        # pre-LN (shared)
            q, k, v = layer.qkv(x).chunk(3, dim=-1)
            q = q.view(n_ext, self.n_heads, self.head_dim)
            k = k.view(n_ext, self.n_heads, self.head_dim)
            v = v.view(n_ext, self.n_heads, self.head_dim)
            logit_bias = torch.cat(
                [layer.edge_bias(rbf12),
                 layer.token_bias.unsqueeze(0).expand(n_nodes, -1),
                 layer.token_bias.unsqueeze(0).expand(n_nodes, -1)],
                dim=0)                                    # [E+2N,4]
            logits = ((q[att_queries] * k[att_keys]).sum(dim=-1)
                      * _A4_LOGIT_SCALE + logit_bias)     # [E+2N,4]
            # segment softmax over the query index (index_add /
            # scatter_reduce — never a dense mask; segments cannot mix
            # graphs because every (query, key) pair is within one graph)
            seg_max = torch.full(
                (n_ext, self.n_heads), float("-inf"), dtype=dtype,
                device=device)
            seg_max = seg_max.scatter_reduce(
                0, att_queries.unsqueeze(-1).expand_as(logits), logits,
                reduce="amax", include_self=False)
            weights = torch.exp(logits - seg_max[att_queries])
            denom = torch.zeros(n_ext, self.n_heads, dtype=dtype,
                                device=device).index_add(
                                    0, att_queries, weights)
            weights = weights / denom[att_queries].clamp(min=1e-30)
            out = torch.zeros(n_ext, self.n_heads, self.head_dim,
                              dtype=dtype, device=device)
            out = out.index_add(
                0, att_queries, weights.unsqueeze(-1) * v[att_keys])
            # residual + ZERO-init attention output projection (identity
            # at init); residual + ZERO-init FFN output projection
            hs = hs + layer.out_proj(out.reshape(n_ext, self.hidden))
            hs = hs + layer.ffn_out(F.silu(layer.ffn_in(layer.ln_ffn(hs))))

        h_final = hs[:n_nodes]                            # drop token rows

        # Star-shaped edge readout in INPUT edge order (live graph)
        edge_msg = self.edge_readout(
            torch.cat([h_final[src], h_final[dst], edge_in], dim=-1))

        # z head: graph mean ⊕ key mean over ATOM rows only (the token
        # never enters pooling), tanh — identical semantics to R3.
        key_graph = graph_index[key_index]
        k_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        k_sum = k_sum.index_add(0, key_graph, h_final[key_index])
        k_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, key_graph, torch.ones(
                    int(key_index.shape[0]), dtype=dtype, device=device))
        k_mean = k_sum / k_count.clamp(min=1.0).unsqueeze(-1)
        g_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        g_sum = g_sum.index_add(0, graph_index, h_final)
        g_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, graph_index, torch.ones_like(graph_index, dtype=dtype))
        g_mean = g_sum / g_count.clamp(min=1.0).unsqueeze(-1)
        z = torch.tanh(self.z_head(torch.cat([g_mean, k_mean], dim=-1)))
        return BackboneOutput(z=z, edge_msg=edge_msg)


# ---------------------------------------------------------------------------
# accounting helper (construction mismatch reporting)
# ---------------------------------------------------------------------------
def active_param_breakdown(backbone: nn.Module) -> dict[str, int]:
    """Exact per-slot ACTIVE parameter counts (shared ``w_v`` excluded).

    Keys: ``elem_emb``, ``node_proj``, ``w_g``, ``token_init``,
    ``layers`` (sum over both blocks), ``edge_readout``, ``z_head``.
    Used to REPORT the breakdown when the exact-count assertion fires —
    A4 never pads or shrinks architecture to hit the estimate.
    """
    counts: dict[str, int] = {}
    for name, param in backbone.named_parameters():
        if name.startswith("w_v."):
            continue
        slot = name.split(".", 1)[0]
        counts[slot] = counts.get(slot, 0) + int(param.numel())
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# frozen construction protocol (config init_policy.construction_order)
# ---------------------------------------------------------------------------
#: The canonical frozen construction steps THIS builder executes (config
#: ``init_policy.construction_order`` 1–4: fresh modules, R3 mapping,
#: A4 zero-inits, shared ``w_v`` from R3).  Steps 5–6 are
#: builder-delegated (:data:`common.BUILDER_DELEGATED_STEPS`); the
#: budget gate is a contract check, NOT a protocol step.
EXECUTED_STEPS: Final[tuple[str, ...]] = (
    STEP_FRESH_MODULES,
    STEP_R3_MAPPING,
    STEP_ZERO_INIT,
    STEP_SHARED_W_V,
)


def build_record(
        run_seed: int,
        move_decoder: DirectMoveDecoder,
        *,
        r3_state: Mapping[str, torch.Tensor] | None = None,
) -> BuiltCandidate:
    """Build A4 under the frozen construction protocol (steps 1–5);
    returns the audit-bearing :class:`BuiltCandidate` record.

    Inside ``common.construction_scope(run_seed, "A4")`` (the frozen
    ``seed_init`` stream; the caller's global RNG is restored on exit):

    1. fresh ``SparseGeometryTransformerA4`` modules on ``seed_init``;
    2. strict R3 semantic mapping (:data:`A4_R3_MAPPING` + declared
       :data:`A4_CONSUMED_SOURCE_KEYS`; every one of the 37 payload keys
       accounted exactly once, ``torch.equal`` verified per target);
    3. special zero-inits: ``W_g``, per layer the attention output
       projection, the FFN output projection and the ``b_h`` RBF12 bias;
    4. the SHARED ``w_v``: ``backbone.w_v is move_decoder.w_v`` with the
       R3 tensor values — :func:`common.apply_shared_w_v` delegating to
       :func:`common.apply_r3_w_v` with the SAME resolved ``source`` the
       mapping used (the production path ``r3_state=None`` SHA-verifies
       the pinned checkpoint once; an injected ``source`` feeds BOTH the
       backbone mapping and the decoder load);
    5. the decoder ``w_v`` ``torch.equal`` check and the exact
       active-parameter assertion —
       ``common.assert_candidate_budget(..., expected_active=55_840)``;
       a mismatch is a typed STOP (``V542_CANDIDATE_ACTIVE_MISMATCH``)
       with the measured count — never padding/shrinking.

    Step 6 (``finish_training_seed(run_seed)``) is deliberately NOT
    called here: resetting the training seed belongs to the runner after
    the whole model assembly.  ``r3_state`` (optional) injects a
    same-shape 37-key payload for hermetic tests; production callers
    leave it ``None`` so the SHA-256-verified pinned checkpoint is
    re-hashed on every build.
    """
    if not isinstance(move_decoder, DirectMoveDecoder):
        raise V542InitError(
            "V542_DECODER_TYPE",
            "build requires the shared DirectMoveDecoder (got "
            f"{type(move_decoder).__name__})")
    init_source_id = resolve_init_source_id(CANDIDATE_ID)
    with construction_scope(run_seed, init_source_id) as seed:
        # 1. fresh modules on the seed_init stream
        backbone = SparseGeometryTransformerA4()
        source = (load_r3_backbone_state() if r3_state is None
                  else dict(r3_state))
        # 2. strict R3 semantic mapping (typed audit refusals)
        audit = apply_r3_mapping(
            backbone, source, A4_R3_MAPPING,
            consumed_source_keys=list(A4_CONSUMED_SOURCE_KEYS))
        # 3. special zero-inits (config per_candidate_zero_init.A4 +
        #    Oracle §2.4 layer recipe)
        zero_init_(backbone.w_g)
        for layer in backbone.layers:
            zero_init_(layer.out_proj)
            zero_init_(layer.ffn_out)
            zero_init_(layer.edge_bias)
        # 4. shared w_v from the SAME resolved source (module-object
        #    identity + SHA/values via the common loader)
        assert_decoder_a_ref_frozen(move_decoder)
        backbone.w_v = move_decoder.w_v
        apply_shared_w_v(backbone, move_decoder, source=source)

    # every build ends with a bitwise-equal decoder w_v (B1 / gate 2)
    if not (torch.equal(move_decoder.w_v.weight, source["w_v.weight"])
            and torch.equal(move_decoder.w_v.bias, source["w_v.bias"])):
        raise V542InitError(  # pragma: no cover - copy_ cannot diverge
            "V542_WV_NOT_EQUAL",
            "A4: decoder w_v is not torch.equal to the resolved source "
            "w_v after construction step 4")

    # 5. exact active-parameter assertion (STOP on drift)
    assert_candidate_budget(backbone,
                            expected_active=ESTIMATED_ACTIVE_PARAMS)
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
) -> SparseGeometryTransformerA4:
    """Convenience wrapper over :func:`build_record` returning ONLY the
    backbone (the audit-bearing record stays available for the registry
    / P3 run-identity path)."""
    return build_record(run_seed, move_decoder, r3_state=r3_state).backbone


__all__ = [
    "A4_BACKBONE_VERSION",
    "A4_CONSUMED_SOURCE_KEYS",
    "A4_FFN_DIM",
    "A4_HEAD_DIM",
    "A4_LAYER_PARAMS",
    "A4_N_HEADS",
    "A4_N_LAYERS",
    "A4_R3_MAPPING",
    "CANDIDATE_ID",
    "ESTIMATED_ACTIVE_PARAMS",
    "EXECUTED_STEPS",
    "SparseGeometryTransformerA4",
    "active_param_breakdown",
    "build",
    "build_record",
]
