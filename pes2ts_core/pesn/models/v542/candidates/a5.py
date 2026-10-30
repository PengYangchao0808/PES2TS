"""V5.4.2 candidate A5 / A5-C — shared-GRU GraphGRU backbone
(ADR-0033 §2.1/§2.3; config ``backbone_contract.candidates.A5`` /
``A5-C``; plan §5.1–§5.2).

One module implements BOTH preregistered candidates through the
``cells`` parameter:

* **A5** (``cells=1``, family ``graph_gru_shared_cell``) — node features
  are initialised exactly as R3 (``elem_emb`` ⊕ is_key/degK/role →
  ``node_proj``), then ONE shared ``nn.GRUCell(48, 48)`` is applied
  ``T=4`` times over the per-node MEAN-AGGREGATED messages of the
  Star-shaped shared message MLP (``118→48→48``), each step receiving
  the zero-initialised graph-conditioning injection
  ``W_g: 5→48`` (``cv_target_delta 4 ⊕ signed_intent 1``, broadcast per
  graph).  ``edge_msg`` is the FINAL iteration's per-edge message
  tensor in input ``edge_index`` column order; ``z`` is the R3 z-head
  recipe ``tanh(z_head([graph_mean(h), key_mean(h)]))`` over the final
  node states;
* **A5-C** (``cells=2``, family ``graph_gru_2cell_alternating``,
  diagnostic) — two ALTERNATING recurrent blocks for the same ``T=4``
  (iterations 0/2 use block 0, iterations 1/3 use block 1).  Block 0 IS
  A5's block (built in the identical RNG order, then R3-mapped); block
  1 is fresh under the A5 init stream
  (``init_source_id`` override ``A5-C → A5``, config
  ``init_policy.candidate_fresh_init.init_source_id_overrides``) and
  carries its own fresh message MLP and its own zero-initialised
  ``W_g``.

Parameter arithmetic — the EXACT measured counts equal the frozen
preregistered estimates (config ``backbone_active``; plan §5.2 table;
A5 joint 33,811 = 31,584 + 49 + 2,178, A5-C joint 55,939)::

    elem_emb (119→16)                       1,904
    node_proj (21→48)                       1,056
    z_head (96→64)                          6,208
    message MLP (118→48→48) per cell        8,064
    GRUCell(48,48)      cell 0 (A5 recipe) 14,112
    GRUCell(48,48,bias=False) cell 1       13,824
    W_g (5→48, bias=False) per cell           240

    A5   = base + block 0                    = 31,584
    A5-C = base + block 0 + block 1          = 53,712

Frozen bias resolution: the plan §5.1 prose once writes
``W_g: 5→48, 288`` (a biased Linear), but the frozen ESTIMATES that the
budget gate asserts exactly (31,584 / 53,712, repeated in config, plan
table and ADR) are only reproduced with a BIAS-FREE ``W_g`` (240) and —
for A5-C — a bias-free SECOND cell (13,824) that owns its fresh message
MLP and ``W_g``.  The exact counts are the preregistered invariants;
this module freezes ``W_g = Linear(5, 48, bias=False)`` for every cell,
``cell 0 = nn.GRUCell(48, 48)`` (full biases = the A5 recipe) and
``cell 1 = nn.GRUCell(48, 48, bias=False)``.  The counts are asserted
at construction (:class:`V542InitError` ``V542_A5_PARAMS``) and again
by :func:`~pes2ts_core.pesn.models.v542.candidates.common.
assert_candidate_budget` in :func:`build`; a drift is NEVER padded or
shrunk to fit.

Construction (config ``init_policy.construction_order`` 1→6) happens in
:func:`~pes2ts_core.pesn.models.v542.candidates.common.build_\
construction_scope`-style isolation via :func:`build`:

1. fresh modules under ``seed_init(run_seed, init_source_id)``;
2. frozen R3 semantic mapping (:func:`a5_mapping_table`): the A1
   common set ``elem_emb / node_proj / z_head`` identity + the shared
   message MLP from R3 ``message_mlp.0`` (9 target keys; the remaining
   28 payload keys are declared consumed — ``w_v.*`` is genuinely
   consumed by step 4) — strict-audited copy;
3. per-cell ``W_g`` zero-initialised (:func:`zero_init_`);
4. the shared decoder ``w_v`` is loaded from the SAME resolved source
   the mapping used (:func:`~pes2ts_core.pesn.models.v542.candidates
   .common.apply_r3_w_v`) — the caller supplies the shared
   ``DirectMoveDecoder`` OBJECT (frozen ``a_ref=0.08``, typed refusal
   otherwise); the backbone itself owns NO ``w_v``;
5. the exact active-parameter budget is asserted; the training seed is
   NEVER reset inside.

Contract: ``forward(batch) -> BackboneOutput(z [G,64], edge_msg
[E,48])``; forward-view keys only
(:data:`~pes2ts_core.pesn.contracts.key_atom_v3.FORWARD_VIEW_KEYS_V3`,
legacy typed-invalid taxonomy for non-finite/Z-range violations); fp32;
no dropout; deterministic (two calls on one batch are bitwise equal);
POSITION-PRESERVING — positions are never consumed nor updated (the
backbone math uses only invariant scalars; the vector pathway lives in
the shared decoder); no ``torch.compile`` (ADR-0033 §2.1).  Graph
conditioning enters EVERY iteration through the active cell's
zero-initialised ``W_g`` (the R3 house style conditions every layer;
at init ``W_g=0`` masks the conditioning exactly)."""

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
)
from pes2ts_core.pesn.models.v542.candidates.common import (
    STEP_FRESH_MODULES,
    STEP_R3_MAPPING,
    STEP_SHARED_W_V,
    STEP_ZERO_INIT,
    BuiltCandidate,
    apply_r3_mapping,
    apply_r3_w_v,
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
    EDGE_MSG_DIM_V542,
    HIDDEN_V542,
    NODE_INPUT_DIM_V542,
    V542InitError,
    BackboneOutput,
)

#: Registered A5/A5-C backbone version (campaign families
#: ``graph_gru_shared_cell`` / ``graph_gru_2cell_alternating``).
A5_BACKBONE_VERSION: Final[str] = "pesn_key_atom_v542_a5_graph_gru_v1"

#: Primary candidate id (``cells=1``).
CANDIDATE_ID: Final[str] = "A5"
#: Diagnostic capacity-control id (``cells=2``).
CANDIDATE_ID_2CELL: Final[str] = "A5-C"

#: Frozen recurrent iteration count (config ``candidates.A5.iterations``).
N_ITERATIONS: Final[int] = 4

#: Frozen preregistered ACTIVE parameter estimates (config
#: ``backbone_contract.candidates.*.backbone_active``; plan §5.2 table).
#: These are the EXACT values ``build`` asserts — never padded/shrunk.
ESTIMATED_ACTIVE_PARAMS: Final[dict[str, int]] = {
    CANDIDATE_ID: 31_584,
    CANDIDATE_ID_2CELL: 53_712,
}

#: Frozen R3 semantic mapping (config ``init_policy.init_mapping.
#: r3_semantic_mapping.A5``): the A1 common set identity + the shared
#: message MLP from R3 ``message_mlp.0``.  ``(target, source)`` pairs —
#: explicit, pre-registered, no runtime shape heuristics.
_A5_MAPPED_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("elem_emb.weight", "elem_emb.weight"),
    ("node_proj.weight", "node_proj.weight"),
    ("node_proj.bias", "node_proj.bias"),
    ("z_head.weight", "z_head.weight"),
    ("z_head.bias", "z_head.bias"),
    ("message_mlps.0.0.weight", "message_mlp.0.0.weight"),
    ("message_mlps.0.0.bias", "message_mlp.0.0.bias"),
    ("message_mlps.0.2.weight", "message_mlp.0.2.weight"),
    ("message_mlps.0.2.bias", "message_mlp.0.2.bias"),
)


def candidate_id_for_cells(cells: int) -> str:
    """Frozen ``cells`` → candidate id mapping (1→A5, 2→A5-C).

    Typed :class:`V542InitError` refusal (``V542_A5_CELLS``) for anything
    else — including ``bool`` (``True`` is not a legal ``cells`` value).
    """
    if isinstance(cells, bool) or not isinstance(cells, int) \
            or cells not in (1, 2):
        raise V542InitError(
            "V542_A5_CELLS",
            f"cells must be 1 (A5) or 2 (A5-C), got {cells!r} "
            f"({type(cells).__name__})")
    return CANDIDATE_ID if cells == 1 else CANDIDATE_ID_2CELL


def a5_mapping_table(
        source_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, str], list[str]]:
    """A5's frozen ``{target: source}`` table + declared consumed keys.

    Returns the 9-entry identity-style mapping (``elem_emb`` /
    ``node_proj`` / ``z_head`` + the shared message MLP from R3
    ``message_mlp.0``) and the R3 payload keys NOT consumed by it (the
    remaining layer-2/3 message MLPs, node MLPs, norms and ``w_v`` —
    28 of the frozen 37 backbone keys; ``w_v.*`` is genuinely consumed
    by construction step 4,
    :func:`~pes2ts_core.pesn.models.v542.candidates.common.apply_r3_w_v`,
    which runs AFTER this audit on the SAME resolved source), which
    :func:`~pes2ts_core.pesn.models.v542.candidates.common.apply_r3_
    mapping` requires to be declared exactly once.
    """
    sources = {src for _, src in _A5_MAPPED_PAIRS}
    mapping = {tgt: src for tgt, src in _A5_MAPPED_PAIRS}
    consumed = [k for k in sorted(source_state) if k not in sources]
    return mapping, consumed


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    """Legacy typed-invalid refusal for non-finite inputs."""
    if not bool(torch.isfinite(tensor).all()):
        raise KeyAtomV3InvalidInputError(
            "NONFINITE_INPUT", f"{name} contains non-finite values")


class A5Backbone(nn.Module):
    """A5 / A5-C GraphGRU backbone (see the module docstring for math).

    Args:
        move_decoder: the SHARED :class:`DirectMoveDecoder` whose
            ``w_v`` will consume this backbone's ``edge_msg``.  The
            backbone does NOT own ``w_v`` (MOVE lives entirely in the
            decoder); the argument validates the assembly seam — the
            decoder must be a ``DirectMoveDecoder`` carrying the frozen
            ``a_ref=0.08`` (typed ``V542_A5_DECODER`` /
            ``V542_DECODER_A_REF``).  The decoder module is never
            registered on the backbone, so parameter accounting stays
            backbone-only.
        cells: 1 → A5 (one shared GRUCell), 2 → A5-C (two alternating
            cells).  Anything else is a typed ``V542_A5_CELLS`` refusal.

    The constructor asserts the exact frozen parameter count for the
    chosen candidate (``V542_A5_PARAMS``) — the measured count is never
    padded or shrunk to fit the preregistration.
    """

    #: Registered backbone version.
    v542_version: Final[str] = A5_BACKBONE_VERSION

    def __init__(self, move_decoder: DirectMoveDecoder, *,
                 cells: int = 1) -> None:
        super().__init__()
        if not isinstance(move_decoder, DirectMoveDecoder):
            raise V542InitError(
                "V542_A5_DECODER",
                "A5Backbone requires the shared DirectMoveDecoder "
                f"(got {type(move_decoder).__name__})")
        assert_decoder_a_ref_frozen(move_decoder)
        candidate_id_for_cells(cells)           # typed cells validation
        self.n_cells = int(cells)
        self.n_iterations = N_ITERATIONS
        self.hidden = HIDDEN_V542

        # Build order is part of the init-stream contract: each BLOCK
        # (message MLP + cell + W_g) is drawn BLOCK-BY-BLOCK, so cells=1
        # and cells=2 builds under the same seed_init share ALL of
        # block 0 bit-for-bit (A5-C first cell == A5); block 1 is the
        # fresh second cell.
        self.elem_emb = nn.Embedding(119, ELEM_DIM)
        self.node_proj = nn.Linear(NODE_INPUT_DIM_V542, HIDDEN_V542)
        self.z_head = nn.Linear(2 * HIDDEN_V542, 64)
        msg_in = HIDDEN_V542 + HIDDEN_V542 + EDGE_INPUT_DIM_V3  # 118
        self.message_mlps = nn.ModuleList()
        self.gru_cells = nn.ModuleList()
        self.w_gs = nn.ModuleList()
        for c in range(self.n_cells):
            # cell 0: the A5 recipe — full GRUCell(48, 48) with both
            # bias vectors; cell 1 (A5-C only): bias-free so the
            # measured count matches the frozen 53,712 estimate exactly
            # (module docstring).
            self.message_mlps.append(nn.Sequential(
                nn.Linear(msg_in, HIDDEN_V542), nn.SiLU(),
                nn.Linear(HIDDEN_V542, HIDDEN_V542)))
            self.gru_cells.append(nn.GRUCell(
                HIDDEN_V542, HIDDEN_V542, bias=(c == 0)))
            # zero-initialised graph-conditioning injection (no bias:
            # the frozen estimate counts 240 params per W_g)
            self.w_gs.append(nn.Linear(
                GRAPH_COND_DIM_V3, HIDDEN_V542, bias=False))

        expected = ESTIMATED_ACTIVE_PARAMS[self.candidate_id]
        total = sum(int(p.numel()) for p in self.parameters())
        if total != expected:
            raise V542InitError(
                "V542_A5_PARAMS",
                f"{self.candidate_id} has {total} params, the frozen "
                f"preregistered estimate is {expected} — never pad or "
                "shrink a candidate to fit the budget")

    # -- identity / accounting ---------------------------------------------
    @property
    def candidate_id(self) -> str:
        """``"A5"`` (1 cell) or ``"A5-C"`` (2 alternating cells)."""
        return candidate_id_for_cells(self.n_cells)

    @property
    def active_param_count(self) -> int:
        """Exact trainable backbone params (the backbone owns NO
        decoder parameter — ``w_v`` 49 lives in the shared decoder)."""
        return sum(int(p.numel()) for p in self.parameters())

    # -- helpers -------------------------------------------------------------
    def _edge_input(self, batch: Mapping[str, torch.Tensor],
                    dtype: torch.dtype) -> torch.Tensor:
        """Packed preserved slots ⊕ the five edge-aligned intent scalars."""
        parts = [
            batch[EDGE_ATTR_KEY].to(dtype=dtype),
            batch[EDGE_B_SRC_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_B_TGT_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_DELTA_B_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_ABS_DELTA_KEY].to(dtype=dtype).unsqueeze(-1),
            batch[EDGE_IS_CHANGED_KEY].to(dtype=dtype).unsqueeze(-1),
        ]
        return torch.cat(parts, dim=-1)

    # -- unified forward ----------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> BackboneOutput:
        """``z [G,64]`` ⊕ ``edge_msg [E,48]`` — fp32, deterministic.

        ``edge_msg`` is the FINAL iteration's message tensor, row-for-
        row aligned with the columns of ``batch["edge_index"]`` and
        carrying the live autograd graph (no detach).
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

        # pos is consumed ONLY for typed-invalid validation (the legacy
        # input-legality taxonomy); the math is invariant-scalar only —
        # positions are never read, never updated (position-preserving).
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

        for name, tensor in (
                ("pos", pos), ("edge_attr", edge_in), ("degK", degk),
                ("intent_role", role), ("cv_target_delta", cv_delta),
                ("signed_intent", signed)):
            _require_finite(name, tensor)

        n_nodes = int(pos.shape[0])
        n_graphs = int(cv_delta.shape[0])
        src, dst = edge_index[0], edge_index[1]

        # node scalar init — byte-identical recipe to R3's node input
        degk_norm = degk.clamp(min=0.0, max=4.0) / 4.0
        h = self.node_proj(torch.cat(
            [self.elem_emb(z_atomic.to(device=device)),
             torch.cat([is_key_f.unsqueeze(-1), degk_norm.unsqueeze(-1),
                        role], dim=-1)], dim=-1))
        graph_cond = torch.cat([cv_delta, signed.unsqueeze(-1)], dim=-1)
        cond_nodes = graph_cond[graph_index]      # broadcast per graph

        messages: torch.Tensor | None = None
        for t in range(self.n_iterations):
            c = t % self.n_cells                   # alternating cells
            msg_in = torch.cat([h[src], h[dst], edge_in], dim=-1)
            messages = self.message_mlps[c](msg_in)   # [E,48] input order
            agg = torch.zeros_like(h).index_add(0, dst, messages)
            counts = torch.zeros(
                n_nodes, dtype=dtype, device=device).index_add(
                    0, dst, torch.ones_like(dst, dtype=dtype))
            agg = agg / counts.clamp(min=1.0).unsqueeze(-1)
            h = self.gru_cells[c](
                agg + self.w_gs[c](cond_nodes), h)
        assert messages is not None               # T=4 ≥ 1 by construction

        # --- key-pool + graph-pool scalars (R3 z-head recipe) --------------
        key_graph = graph_index[key_index]
        k_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        k_sum = k_sum.index_add(0, key_graph, h[key_index])
        k_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, key_graph,
                torch.ones(key_index.shape[0], dtype=dtype, device=device))
        k_mean = k_sum / k_count.clamp(min=1.0).unsqueeze(-1)

        g_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        g_sum = g_sum.index_add(0, graph_index, h)
        g_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, graph_index, torch.ones_like(graph_index, dtype=dtype))
        g_mean = g_sum / g_count.clamp(min=1.0).unsqueeze(-1)

        z = torch.tanh(self.z_head(torch.cat([g_mean, k_mean], dim=-1)))
        return BackboneOutput(z=z, edge_msg=messages)


#: The canonical frozen construction steps THIS builder executes (config
#: ``init_policy.construction_order`` 1–4: fresh modules, R3 mapping,
#: per-cell ``W_g`` zero-init, shared ``w_v`` from R3).  Steps 5–6 are
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
        cells: int = 1,
) -> BuiltCandidate:
    """Construct an A5 (``cells=1``) or A5-C (``cells=2``) backbone;
    returns the audit-bearing :class:`BuiltCandidate` record.

    Executes the frozen construction protocol INSIDE
    :func:`~pes2ts_core.pesn.models.v542.candidates.common.
    construction_scope` (seeded ``seed_init(run_seed, init_source_id)``
    with the A5-C→A5 attribution override resolved via
    :func:`~pes2ts_core.pesn.models.v542.candidates.common.
    resolve_init_source_id`):

    1. fresh modules under the init stream;
    2. the frozen R3 semantic mapping applied with the strict audit
       (:func:`a5_mapping_table` → 9 mapped keys, 28 declared consumed);
    3. every cell's ``W_g`` zero-initialised (:func:`zero_init_`);
    4. the SHARED decoder ``w_v`` loaded from the SAME resolved
       ``source`` the mapping used
       (:func:`~pes2ts_core.pesn.models.v542.candidates.common.
       apply_r3_w_v`) — the caller supplies the decoder OBJECT, never
       its values (the backbone owns no ``w_v`` slot);
    5. the decoder ``w_v`` ``torch.equal`` check and the exact
       active-parameter budget asserted
       (:func:`~pes2ts_core.pesn.models.v542.candidates.common.
       assert_candidate_budget` against
       :data:`ESTIMATED_ACTIVE_PARAMS`).

    ``r3_state`` defaults to the SHA-pinned
    :func:`~pes2ts_core.pesn.models.v542.candidates.common.
    load_r3_backbone_state` payload.  The training seed is NEVER reset
    inside (step 6 ``finish_training_seed`` is the caller's job); the
    caller's global torch RNG stream is restored byte-identically by
    ``construction_scope`` on exit.
    """
    candidate_id = candidate_id_for_cells(cells)
    init_source_id = resolve_init_source_id(candidate_id)
    with construction_scope(run_seed, init_source_id) as seed:
        backbone = A5Backbone(move_decoder, cells=cells)
        source = (load_r3_backbone_state() if r3_state is None
                  else dict(r3_state))
        mapping, consumed = a5_mapping_table(source)
        audit = apply_r3_mapping(backbone, source, mapping,
                                 consumed_source_keys=consumed)
        for slot in backbone.w_gs:
            zero_init_(slot)
        apply_r3_w_v(move_decoder, source)

    # every build ends with a bitwise-equal decoder w_v (B1 / gate 2)
    if not (torch.equal(move_decoder.w_v.weight, source["w_v.weight"])
            and torch.equal(move_decoder.w_v.bias, source["w_v.bias"])):
        raise V542InitError(  # pragma: no cover - copy_ cannot diverge
            "V542_WV_NOT_EQUAL",
            f"{candidate_id}: decoder w_v is not torch.equal to the "
            "resolved source w_v after construction step 4")

    assert_candidate_budget(
        backbone, expected_active=ESTIMATED_ACTIVE_PARAMS[candidate_id])
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
        cells: int = 1,
) -> A5Backbone:
    """Convenience wrapper over :func:`build_record` returning ONLY the
    backbone (the audit-bearing record stays available for the registry
    / P3 run-identity path)."""
    return build_record(
        run_seed, move_decoder, r3_state=r3_state, cells=cells).backbone


__all__ = [
    "A5_BACKBONE_VERSION",
    "A5Backbone",
    "CANDIDATE_ID",
    "CANDIDATE_ID_2CELL",
    "ESTIMATED_ACTIVE_PARAMS",
    "EXECUTED_STEPS",
    "N_ITERATIONS",
    "a5_mapping_table",
    "build",
    "build_record",
    "candidate_id_for_cells",
]
