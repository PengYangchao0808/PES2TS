"""V5.4.2 candidate A1 / A1-C — 2L/3L GeoEGNN-lite (ADR-0033 §2.3;
config ``backbone_contract.backbone_registry`` A1/A1-C rows; Oracle
architecture §2.1).

A1 is the Star-shaped 2-layer scalar backbone with a ZERO-initialised
per-layer distance injection ``W_d: RBF12 → 48``::

    m_ij  = message_mlp([h_i, h_j, e22]) + W_d · rbf12_ij     (per layer)
    agg_i = mean over in-edges of m_ij                          (dst-only)
    h_i   = LayerNorm(h_i + node_mlp([h_i, agg_i, graph_cond(5)]))

* ``rbf12`` is the FIRST 12 packed ``edge_attr`` slots (the F3
  current-distance Gaussian RBF) — a deterministic slice, no new
  featurization, no parameters beyond ``W_d`` (Oracle §2.1: rbf12 直接取
  edge_attr 前 12 槽);
* message MLP ``(48+48+22)→48→48`` and node MLP ``(48+48+5)→48→48`` keep
  the EXACT Star shapes/semantics (dst-only mean aggregation, directed
  intent graph conditioning via ``cv_target_delta ⊕ signed_intent``),
  LayerNorm on scalars only; no dropout/batchnorm, fp32, no coordinate
  updates, no torch.compile (ADR-0033 §2.1);
* ``z`` uses the legacy recipe verbatim: graph mean-pool ⊕ key mean-pool
  → ``z_head`` Linear(96→64) → tanh; ``edge_msg`` is the final-layer
  per-edge message tensor in INPUT edge order ``[E, 48]`` with a LIVE
  autograd graph (the unified decoder backprops through it);
* the shared ``w_v`` module OBJECT is adopted from the
  :class:`~pes2ts_core.pesn.models.v542.decoder.DirectMoveDecoder`
  (``backbone.w_v is move_decoder.w_v``) and is EXCLUDED from
  ``active_param_count`` (decoder-owned, 49 params, all candidates
  identical values from R3).

A1-C (``layers=3``) is A1 with a THIRD layer: layers 1–2 are the A1
mapping set (R3-mapped + zeroed ``W_d``), layer 3 is a FRESH draw on the
A1 init stream (``init_source_id="A1"`` via the frozen
``INIT_SOURCE_ID_OVERRIDES``) with its ``W_d`` also zero-initialised.

Parameter accounting (exact; config estimates confirmed on construction):

* common = ``elem_emb`` 1,904 + ``node_proj`` 1,056 + ``z_head`` 6,208
  = 9,168;
* per layer = message MLP 8,064 + node MLP 7,248 + LayerNorm 96 +
  ``W_d`` 624 = 16,032;
* A1 = 9,168 + 2×16,032 = **41,232**; A1-C = 9,168 + 3×16,032 =
  **57,264** (``norm_h`` is fresh-but-constant-init; the frozen A1
  mapping set maps ``elem_emb, node_proj, z_head, message_mlp.0/1,
  node_mlp.0/1`` only — 21 keys).

The factory :func:`build_record` (and its backbone-only wrapper
:func:`build`) runs the frozen construction protocol
(config ``init_policy.construction_order``): fresh modules under
``construction_scope`` → :func:`apply_r3_mapping` (strict audit) →
zero-init ``W_d`` → :func:`apply_shared_w_v` (delegating to
:func:`~pes2ts_core.pesn.models.v542.candidates.common.apply_r3_w_v`
with the SAME resolved source) → decoder ``w_v`` ``torch.equal`` check
→ exact budget assertion.  It does NOT reset the training seed (the
caller runs :func:`finish_training_seed`)."""

from __future__ import annotations

from typing import Final, Mapping

import torch
import torch.nn as nn

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
    RBF_SLOT_SLICE_V3,
    check_forward_view_keys_v3,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import (
    ELEM_DIM,
    GRAPH_COND_DIM_V3,
    KeyAtomV3InvalidInputError,
    NODE_FEATURE_DIM_V3,
    Z_DIM_V3,
)
from pes2ts_core.pesn.models.v542.decoder import (
    DirectMoveDecoder,
    assert_decoder_a_ref_frozen,
)
from pes2ts_core.pesn.models.v542.interfaces import (
    HIDDEN_V542,
    N_DECODER_PARAMS_V542,
    V542InitError,
    BackboneOutput,
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

#: Registered A1 backbone version (campaign family ``geoegnn_lite_2l`` /
#: ``geoegnn_lite_3l`` for A1-C).
A1_BACKBONE_VERSION: Final[str] = "pesn_key_atom_v542_a1_v1"

#: Frozen candidate ids.
CANDIDATE_ID: Final[str] = "A1"
A1C_CANDIDATE_ID: Final[str] = "A1-C"

#: Exact active-parameter counts (config estimates, asserted EXACTLY at
#: construction — 41,232 / 57,264; see the module docstring breakdown).
ESTIMATED_ACTIVE_PARAMS: Final[int] = 41_232
A1C_ESTIMATED_ACTIVE_PARAMS: Final[int] = 57_264

#: Frozen A1/A1-C R3 semantic mapping prefixes (config
#: ``init_policy.init_mapping.r3_semantic_mapping``: A1 映射 elem_emb,
#: node_proj, z_head, message_mlp.0/1, node_mlp.0/1 — R3 同名同形; R3's
#: third-layer keys are deliberately NOT mapped for A1/A1-C's layers 1–2
#: audit; ``w_v.*`` is consumed by construction step 4
#: (:func:`apply_r3_w_v` / :func:`apply_shared_w_v`, which run AFTER the
#: mapping audit on the SAME resolved source — the declaration order is
#: audit-then-load, and the load genuinely consumes those keys).
A1_MAPPING_PREFIXES: Final[tuple[str, ...]] = (
    "elem_emb", "node_proj", "z_head",
    "message_mlp.0", "message_mlp.1",
    "node_mlp.0", "node_mlp.1",
)

#: The canonical frozen construction steps THIS builder executes (config
#: ``init_policy.construction_order`` 1–4: fresh modules, R3 mapping,
#: per-candidate zero-init, shared ``w_v`` from R3).  Steps 5–6 are
#: builder-delegated (:data:`common.BUILDER_DELEGATED_STEPS`); the
#: budget gate is a contract check, NOT a protocol step.
EXECUTED_STEPS: Final[tuple[str, ...]] = (
    STEP_FRESH_MODULES,
    STEP_R3_MAPPING,
    STEP_ZERO_INIT,
    STEP_SHARED_W_V,
)

#: Legal layer counts for the A1 family (2 = A1, 3 = A1-C).
A1_LAYER_COUNTS: Final[tuple[int, ...]] = (2, 3)


def r3_mapping_table(
        source_state: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    """The frozen A1 mapping set: identity over the seven prefixes.

    ``{target_key: source_key}`` with target == source (R3 同名同形);
    exactly the 21 keys of the frozen A1/A3 table — every other payload
    key (R3 ``norm_h.*``/``w_v.*`` and the R3 third-layer keys) is
    declared via ``consumed_source_keys`` by the caller.
    """
    mapping: dict[str, str] = {}
    for key in sorted(source_state):
        if any(key == prefix or key.startswith(prefix + ".")
               for prefix in A1_MAPPING_PREFIXES):
            mapping[key] = key
    return mapping


def _consumed_source_keys(
        source_state: Mapping[str, torch.Tensor],
        mapping: Mapping[str, str],
) -> list[str]:
    """Payload keys accounted OUTSIDE the mapping (strict full audit)."""
    return [key for key in sorted(source_state) if key not in mapping]


class A1Backbone(nn.Module):
    """A1/A1-C GeoEGNN-lite backbone behind the unified interface.

    Args:
        move_decoder: the SHARED decoder whose ``w_v`` module object this
            backbone adopts (``self.w_v is move_decoder.w_v``) — the A0
            wiring invariant, required so ``apply_shared_w_v`` can load
            the R3 ``w_v`` into the single shared slot.  The decoder's
            ``a_ref`` must be the frozen 0.08 (gate-1 B1).
        layers: 2 (A1) or 3 (A1-C).  Any other count is a typed
            :class:`~pes2ts_core.pesn.models.v542.interfaces.V542InitError`
            (``V542_A1_LAYERS``) — the campaign freezes these two.
    """

    #: Registered backbone version.
    v542_version: Final[str] = A1_BACKBONE_VERSION

    def __init__(self, move_decoder: DirectMoveDecoder, *,
                 layers: int = 2) -> None:
        super().__init__()
        if not isinstance(move_decoder, DirectMoveDecoder):
            raise V542InitError(
                "V542_A1_DECODER",
                "A1Backbone requires the shared DirectMoveDecoder "
                f"(got {type(move_decoder).__name__})")
        assert_decoder_a_ref_frozen(move_decoder)
        if int(layers) not in A1_LAYER_COUNTS:
            raise V542InitError(
                "V542_A1_LAYERS",
                f"A1Backbone layers must be one of {A1_LAYER_COUNTS} "
                f"(got {layers!r})")
        self.hidden = HIDDEN_V542
        self.n_layers = int(layers)

        # --- fresh modules (construction order matters for the A1 vs
        # A1-C stream: layers are drawn INTERLEAVED so layers 1–2 see
        # byte-identical RNG draws under the shared A1 init stream) ------
        self.elem_emb = nn.Embedding(119, ELEM_DIM)
        self.node_proj = nn.Linear(
            ELEM_DIM + NODE_FEATURE_DIM_V3, self.hidden)
        msg_in = self.hidden + self.hidden + EDGE_INPUT_DIM_V3   # 118
        node_in = self.hidden + self.hidden + GRAPH_COND_DIM_V3  # 101
        self.message_mlp = nn.ModuleList()
        self.node_mlp = nn.ModuleList()
        self.norm_h = nn.ModuleList()
        self.w_d = nn.ModuleList()
        for _ in range(self.n_layers):
            self.message_mlp.append(nn.Sequential(
                nn.Linear(msg_in, self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, self.hidden)))
            self.node_mlp.append(nn.Sequential(
                nn.Linear(node_in, self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, self.hidden)))
            self.norm_h.append(nn.LayerNorm(self.hidden))
            self.w_d.append(nn.Linear(N_RBF_V3, self.hidden))
        self.z_head = nn.Linear(2 * self.hidden, Z_DIM_V3)

        # Adopt the decoder's w_v module OBJECT (not a copy): no fresh
        # w_v draw, no forked ownership — ``named_parameters()`` dedupes.
        self.w_v = move_decoder.w_v
        assert self.w_v is move_decoder.w_v

        total = sum(int(p.numel()) for p in self.parameters())
        expected_active = (
            ESTIMATED_ACTIVE_PARAMS if self.n_layers == 2
            else A1C_ESTIMATED_ACTIVE_PARAMS)
        if total != expected_active + N_DECODER_PARAMS_V542:
            raise V542InitError(
                "V542_A1_PARAMS",
                f"A1Backbone(layers={self.n_layers}) has {total} params, "
                f"expected {expected_active} active + "
                f"{N_DECODER_PARAMS_V542} shared w_v")

    # -- accounting ---------------------------------------------------------
    @property
    def active_param_count(self) -> int:
        """Exact trainable backbone params EXCLUDING the shared ``w_v``.

        A1: **41,232**; A1-C: **57,264** (config registry estimates,
        asserted EXACTLY in :meth:`__init__` and by the factory budget
        gate).  The 49 ``w_v`` params are counted once, in the decoder.
        """
        return (sum(int(p.numel()) for p in self.parameters())
                - N_DECODER_PARAMS_V542)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _require_finite(name: str, tensor: torch.Tensor) -> None:
        if not bool(torch.isfinite(tensor).all()):
            raise KeyAtomV3InvalidInputError(
                "NONFINITE_INPUT", f"{name} contains non-finite values")

    def _edge_input(self, batch: Mapping[str, torch.Tensor],
                    dtype: torch.dtype) -> torch.Tensor:
        """Concatenate the packed preserved slots with the intent scalars."""
        parts = [
            batch[EDGE_ATTR_KEY].to(dtype=dtype),
            batch[EDGE_B_SRC_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_B_TGT_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_DELTA_B_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_ABS_DELTA_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_IS_CHANGED_KEY].to(dtype=dtype).unsqueeze(-1),
        ]
        return torch.cat(parts, dim=-1)

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
        """Last-layer message transform.  A1: identity (Star-exact).

        A3 overrides this with the last-layer reaction query attention;
        the hook exists so the two candidates share ONE forward loop and
        A3−A1 isolates exactly the query operator.
        """
        return messages

    # -- unified forward ----------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> BackboneOutput:
        """Frozen forward-view batch → ``BackboneOutput(z, edge_msg)``.

        Only ``FORWARD_VIEW_KEYS_V3`` are read (whitelist enforced first);
        non-finite inputs and out-of-range atomic numbers are typed
        refusals (legacy taxonomy, never a fabricated action).  fp32,
        deterministic, no coordinate updates.
        """
        check_forward_view_keys_v3(batch)
        dtype = self.node_proj.weight.dtype
        device = self.node_proj.weight.device

        z_atomic = batch[NODE_Z_KEY]
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
        # RBF12 distance injection source: the first N_RBF_V3 packed
        # edge_attr slots (deterministic current-distance RBF, no params).
        rbf12 = batch[EDGE_ATTR_KEY].to(dtype=dtype)[
            :, RBF_SLOT_SLICE_V3]
        cv_delta = batch[GRAPH_CV_DELTA_KEY].to(dtype=dtype)
        signed = batch[GRAPH_SIGNED_INTENT_KEY].to(dtype=dtype)

        for name, tensor in (
                ("pos", pos), ("edge_attr", edge_in), ("degK", degk),
                ("intent_role", role), ("cv_target_delta", cv_delta),
                ("signed_intent", signed)):
            self._require_finite(name, tensor)

        n_nodes = int(pos.shape[0])
        n_graphs = int(cv_delta.shape[0])
        src, dst = edge_index[0], edge_index[1]

        # node scalar init: element embedding ⊕ v3 node features
        degk_norm = degk.clamp(min=0.0, max=4.0) / 4.0
        h0 = self.node_proj(torch.cat(
            [self.elem_emb(z_atomic.to(device=device)),
             torch.cat([is_key_f.unsqueeze(-1), degk_norm.unsqueeze(-1),
                        role], dim=-1)], dim=-1))
        graph_cond = torch.cat([cv_delta, signed.unsqueeze(-1)], dim=-1)
        cond_nodes = graph_cond[graph_index]      # broadcast per graph

        h = h0
        last_messages: torch.Tensor | None = None
        for layer in range(self.n_layers):
            msg_in = torch.cat([h[src], h[dst], edge_in], dim=-1)
            messages = self.message_mlp[layer](msg_in) \
                + self.w_d[layer](rbf12)          # m_ij + W_d · rbf12
            if layer == self.n_layers - 1:
                # A1: identity; A3: per-edge reaction query attention —
                # the node update then aggregates the SAME edge_msg.
                messages = self._final_edge_messages(
                    messages, h0=h0, graph_cond=graph_cond,
                    graph_index=graph_index, dst=dst, n_nodes=n_nodes)
            agg = torch.zeros_like(h)
            agg = agg.index_add(0, dst, messages)
            counts = torch.zeros(
                n_nodes, dtype=dtype, device=device).index_add(
                    0, dst, torch.ones_like(dst, dtype=dtype))
            agg = agg / counts.clamp(min=1.0).unsqueeze(-1)
            h = self.norm_h[layer](
                h + self.node_mlp[layer](
                    torch.cat([h, agg, cond_nodes], dim=-1)))
            last_messages = messages

        if last_messages is None:  # pragma: no cover - n_layers >= 2
            raise V542InitError(
                "V542_A1_LAYERS",
                f"backbone ran {self.n_layers} layers; no final message")

        # --- z head: graph pool ⊕ key pool -> 64, tanh (legacy recipe) ----
        key_graph = graph_index[key_index]
        n_keys = int(key_index.shape[0])
        k_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        k_sum = k_sum.index_add(0, key_graph, h[key_index])
        k_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, key_graph,
                torch.ones(n_keys, dtype=dtype, device=device))
        k_mean = k_sum / k_count.clamp(min=1.0).unsqueeze(-1)

        g_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        g_sum = g_sum.index_add(0, graph_index, h)
        g_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, graph_index, torch.ones_like(graph_index, dtype=dtype))
        g_mean = g_sum / g_count.clamp(min=1.0).unsqueeze(-1)
        z = torch.tanh(self.z_head(torch.cat([g_mean, k_mean], dim=-1)))
        return BackboneOutput(z=z, edge_msg=last_messages)


def build_record(
        run_seed: int,
        move_decoder: DirectMoveDecoder,
        *,
        r3_state: Mapping[str, torch.Tensor] | None = None,
        layers: int = 2,
) -> BuiltCandidate:
    """Frozen construction protocol for A1 (``layers=2``) / A1-C
    (``layers=3``); returns the audit-bearing :class:`BuiltCandidate`
    record (backbone, shared decoder, the ACTUAL
    :class:`~pes2ts_core.pesn.models.v542.candidates.common.MappingAudit`,
    seed, resolved init source, canonical construction order + executed /
    deferred step lists).

    Steps (config ``init_policy.construction_order`` 1→5, inside
    ``construction_scope``): fresh modules under
    ``seed_init(run_seed, resolve_init_source_id(candidate_id))`` — both
    A1 and A1-C draw on the FROZEN A1 init stream
    (``init_source_id_overrides: {A1-C: A1}``) — → strict
    :func:`apply_r3_mapping` of the 21-key frozen table (the remaining 16
    payload keys — ``norm_h.*``, ``w_v.*`` and the R3 third-layer keys —
    are declared in ``consumed_source_keys``; ``w_v.*`` is then genuinely
    consumed by step 4 from the SAME resolved source) → zero-init every
    ``W_d`` → :func:`apply_shared_w_v` (delegating to
    :func:`common.apply_r3_w_v` with the same ``source``).  After the
    scope, the decoder ``w_v`` is asserted ``torch.equal`` to the source
    and the EXACT active count is asserted against the frozen estimate
    (41,232 / 57,264) and the 70k hard cap.  The factory does NOT reset
    the training seed — the caller MUST run :func:`finish_training_seed`
    before training.
    """
    if int(layers) not in A1_LAYER_COUNTS:
        raise V542InitError(
            "V542_A1_LAYERS",
            f"layers must be one of {A1_LAYER_COUNTS} (got {layers!r})")
    candidate_id = CANDIDATE_ID if int(layers) == 2 else A1C_CANDIDATE_ID
    init_source_id = resolve_init_source_id(candidate_id)   # A1-C → A1
    source = (load_r3_backbone_state() if r3_state is None
              else dict(r3_state))

    with construction_scope(run_seed, init_source_id) as seed:
        backbone = A1Backbone(move_decoder, layers=int(layers))
        mapping = r3_mapping_table(source)
        consumed = _consumed_source_keys(source, mapping)
        audit = apply_r3_mapping(backbone, source, mapping,
                                 consumed_source_keys=consumed)
        for slot in backbone.w_d:                 # per-candidate zero-init
            zero_init_(slot)
        apply_shared_w_v(backbone, move_decoder, source=source)

    # every build ends with a bitwise-equal decoder w_v (B1 / gate 2);
    # apply_shared_w_v already asserted it, this re-checks the post-scope
    if not (torch.equal(move_decoder.w_v.weight, source["w_v.weight"])
            and torch.equal(move_decoder.w_v.bias, source["w_v.bias"])):
        raise V542InitError(  # pragma: no cover - copy_ cannot diverge
            "V542_WV_NOT_EQUAL",
            f"{candidate_id}: decoder w_v is not torch.equal to the "
            "resolved source w_v after construction step 4")

    expected_active = (ESTIMATED_ACTIVE_PARAMS if int(layers) == 2
                       else A1C_ESTIMATED_ACTIVE_PARAMS)
    assert_candidate_budget(backbone, expected_active=expected_active)
    return BuiltCandidate(
        candidate_id=candidate_id,
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
        layers: int = 2,
) -> A1Backbone:
    """Convenience wrapper over :func:`build_record` returning ONLY the
    backbone (the audit-bearing record stays available for the registry
    / P3 run-identity path)."""
    return build_record(
        run_seed, move_decoder, r3_state=r3_state, layers=layers).backbone


__all__ = [
    "A1_BACKBONE_VERSION",
    "A1C_CANDIDATE_ID",
    "A1C_ESTIMATED_ACTIVE_PARAMS",
    "A1_LAYER_COUNTS",
    "A1_MAPPING_PREFIXES",
    "A1Backbone",
    "CANDIDATE_ID",
    "ESTIMATED_ACTIVE_PARAMS",
    "EXECUTED_STEPS",
    "build",
    "build_record",
    "r3_mapping_table",
]
