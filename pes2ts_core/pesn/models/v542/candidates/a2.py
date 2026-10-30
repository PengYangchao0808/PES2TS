"""V5.4.2 candidate A2 — single-frame PaiNN-lite v32 (ADR-0033 §2.3/§4).

Frozen recipe (config ``backbone_registry.A2`` family
``painn_lite_single_frame_v32``; Oracle architecture §2.2; ADR-0033 §2.3):
two message-passing layers over scalar width ``hidden=48`` with a polar
vector channel of ``32`` channels per node/edge — scalar state ``s`` and
vector state ``V ∈ [N,3,32]`` — position-preserving (``pos`` is NEVER
updated), fp32, deterministic, forward-view keys only, no dropout /
batchnorm, no ``torch.compile``.

Per layer (Oracle §2.2 steps 1–5, re-derived here verbatim):

1. ``m_ij = phi_s([s_i, s_j, e22])`` — ``phi_s: 118→48→48`` (the R3
   ``message_mlp`` shape, warm-started through the frozen A2 mapping);
2. vector message ``V_msg_i = Σ_j [ V_j ⊙ W_s(m_ij) + W_v(m_ij) ⊙ r̂_ij ]``
   with ``W_s, W_v: 48→32`` (per-channel scalar gates / radial
   coefficients) and ``r̂_ij = r_ij/(‖r_ij‖+1e-6)`` (receiver = ``dst``,
   Star convention);
3. vector gate ``[a_v, b_v] = phi_g([s_i, ‖V_i‖])``, ``phi_g: 80→64``;
   ``V ← sigmoid(a_v) ⊙ V + sigmoid(b_v) ⊙ V_msg`` with the per-channel
   norm over the 3 spatial components
   ``‖V_i‖_c = sqrt(Σ_s V_{c,s}² + 1e-8)`` → ``[N, 32]`` (eps is
   REQUIRED: without it the ``V = 0`` cold start makes the gradient
   NaN);
4. scalar update ``s ← s + phi_u([s_i, ‖V_i‖])``, ``phi_u: 80→48→48``;
5. ``LayerNorm`` on the scalar channel ONLY (never on ``V``).

Initial state: ``s_0 = node_proj(...) + W_g(graph_cond)[graph_index]``
(``W_g: 5→48``, 288 params, ZERO-initialised so the warm start keeps
R3's ``s_0``) and ``V_0 = 0`` (registered zero buffer — the Oracle
estimate has no ``V_0`` parameter term).

``z = tanh(z_head([graph_mean(s^L), key_mean(s^L)]))`` — the frozen
Star z-head recipe, unchanged.

**INTENTIONAL semantic difference from Star (ADR-0033 §4 — registry /
report obligation, NOT a bug, must never be "fixed")**: ``edge_msg`` is
RECOMPUTED from the FINAL scalar state ``s^L`` through the SAME
``phi_s`` (the last layer's module) — ``edge_msg_ij = phi_s([s_i^L,
s_j^L, e22])`` — instead of being the captured last-layer message of
Star.  Because ``s^L`` depends on ``‖V‖`` through ``phi_u``, this is
what makes the vector channel act on MOVE (the shared
``DirectMoveDecoder`` consumes ``edge_msg``); with a Star-style capture
the A2 vector channel would have no path to MOVE at all and the
candidate's hypothesis test would be void.  Returned rows stay in
``batch["edge_index"]`` column order with a live autograd graph.

Frozen initialisation (config ``per_candidate_zero_init.A2``): ``V_0=0``;
``W_s``/``W_v`` default seeded (construction ``fork_rng``); ``phi_u`` /
``phi_g`` final output projections zero-initialised (cold-start gate
closed — the V→s path opens during training); ``W_g`` zero.

Frozen R3 mapping (config ``init_mapping.r3_semantic_mapping.A2``):
the A1 common set (``elem_emb``, ``node_proj``, ``z_head``) identity +
two same-shape ``phi_s`` ← R3 ``message_mlp.0/1`` (13 targets; the
remaining 24 payload keys are declared consumed — A2 has no
``node_mlp``/``norm_h`` slots, and ``w_v.*`` is genuinely consumed by
construction step 4, :func:`~pes2ts_core.pesn.models.v542.candidates
.common.apply_r3_w_v`, which loads the SHA-verified R3 ``w_v`` into the
SHARED ``DirectMoveDecoder`` — the caller supplies the decoder OBJECT,
never its values).  Fresh ``W_s``/``W_v``/``phi_u``/``phi_g``/``norm``/
``W_g`` params come from the candidate ``seed_init`` stream; zero-init
runs AFTER the mapping.

Parameter budget (exact, asserted at construction AND by
:func:`build`): common 9,168 + ``W_g`` 288 + 2×(``phi_s`` 8,064 +
``W_s`` 1,568 + ``W_v`` 1,568 + ``phi_u`` 6,240 + ``phi_g`` 5,184 +
``norm`` 96) = 9,456 + 45,440 = **54,896**.

Discipline: implemented from scratch for this candidate — NO code from
``dual_egnn_painn``/``dual_ts_graph`` (Oracle §2.2: 不得复用 dual-*) is
imported or copied; no cross product (axial vectors would break
reflection covariance); vector components are never concatenated into a
scalar MLP — only ``‖V‖`` enters invariants; no absolute coordinates;
no ``torch.compile`` (ADR-0033 §2.1).
"""

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
    NODE_FEATURE_DIM_V3,
    KeyAtomV3InvalidInputError,
    Z_DIM_V3,
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
    HIDDEN_V542,
    V542InitError,
    BackboneOutput,
)

#: Registered A2 candidate id (config ``backbone_registry.A2``).
CANDIDATE_ID: Final[str] = "A2"
#: Config estimate the measured count must equal EXACTLY (Oracle §2.6
#: table: backbone active 54,896; ``estimated: true`` until asserted).
ESTIMATED_ACTIVE_PARAMS: Final[int] = 54_896
#: Registered A2 backbone version (bumped only with a new ADR).
A2_BACKBONE_VERSION: Final[str] = "pesn_key_atom_v542_a2_painn_lite_v32"

#: Frozen scalar width (interface constant, candidates may not change).
HIDDEN: Final[int] = HIDDEN_V542                       # 48
#: Frozen vector-channel count (config ``vector_channels: 32``).
VECTOR_CHANNELS: Final[int] = 32
#: Frozen layer count (config ``layers: 2``).
N_LAYERS: Final[int] = 2
#: Frozen graph-conditioning width (``cv_target_delta`` 4 ⊕ signed 1).
GRAPH_COND_DIM: Final[int] = GRAPH_COND_DIM_V3          # 5
#: ``phi_u``/``phi_g`` input width: ``[s (48), ‖V‖ (32)]``.
VECTOR_COUPLING_DIM: Final[int] = HIDDEN + VECTOR_CHANNELS  # 80
#: ``phi_g`` output width: two 32-channel gates ``[a_v, b_v]``.
GATE_DIM: Final[int] = 2 * VECTOR_CHANNELS             # 64
#: ``‖V‖`` underflow guard (Oracle §2.2: REQUIRED — ``V=0`` must not
#: produce a NaN gradient).
VECTOR_NORM_EPS: Final[float] = 1e-8
#: ``r̂`` zero-length guard (Oracle §2.2: ``r_ij/(‖r_ij‖+1e-6)``).
R_HAT_EPS: Final[float] = 1e-6
#: Star ``degK`` normalisation constant (node feature width 5).
DEGK_SCALE: Final[float] = 4.0

#: Frozen A2 R3 semantic mapping (config
#: ``init_policy.init_mapping.r3_semantic_mapping.A2``): the A1 common
#: set (``elem_emb``/``node_proj``/``z_head`` identity) + two same-shape
#: ``phi_s`` ← R3 ``message_mlp.0/1``.  13 targets; the other 24 payload
#: keys are declared consumed (no ``node_mlp``/``norm_h`` slots exist on
#: A2; ``w_v.*`` is consumed by construction step 4,
#: :func:`~pes2ts_core.pesn.models.v542.candidates.common.apply_r3_w_v`,
#: which runs AFTER this audit on the SAME resolved source).
A2_R3_MAPPING: Final[dict[str, str]] = {
    "elem_emb.weight": "elem_emb.weight",
    "node_proj.weight": "node_proj.weight",
    "node_proj.bias": "node_proj.bias",
    "z_head.weight": "z_head.weight",
    "z_head.bias": "z_head.bias",
    "phi_s.0.0.weight": "message_mlp.0.0.weight",
    "phi_s.0.0.bias": "message_mlp.0.0.bias",
    "phi_s.0.2.weight": "message_mlp.0.2.weight",
    "phi_s.0.2.bias": "message_mlp.0.2.bias",
    "phi_s.1.0.weight": "message_mlp.1.0.weight",
    "phi_s.1.0.bias": "message_mlp.1.0.bias",
    "phi_s.1.2.weight": "message_mlp.1.2.weight",
    "phi_s.1.2.bias": "message_mlp.1.2.bias",
}


def _scalar_message_mlp() -> nn.Sequential:
    """Star-shaped scalar MLP ``118→48→48`` (SiLU between linears)."""
    msg_in = HIDDEN + HIDDEN + EDGE_INPUT_DIM_V3        # 48+48+22 = 118
    return nn.Sequential(
        nn.Linear(msg_in, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))


class A2Backbone(nn.Module):
    """Single-frame PaiNN-lite v32 backbone behind :class:`BackboneOutput`.

    See the module docstring for the frozen recipe, the exact 54,896
    parameter breakdown, the A2 R3 mapping and the INTENTIONAL
    ``edge_msg``-recompute semantics (ADR-0033 §4).

    Construction discipline: the module is normally built inside
    :func:`~pes2ts_core.pesn.models.v542.candidates.common.construction_scope`
    (fresh ``seed_init`` stream) and then warm-started/zero-initialised
    by :func:`build`; the constructor itself only allocates the fresh
    parameters and asserts the exact budget, so any drift from 54,896 is
    a typed refusal at construction instead of a padded/shrunk model.
    """

    #: Registered candidate version.
    v542_version: Final[str] = A2_BACKBONE_VERSION

    def __init__(self) -> None:
        super().__init__()
        # --- mapped / common blocks (R3 same-shape warm-start) -------------
        self.elem_emb = nn.Embedding(119, ELEM_DIM)      # Z=1..118; 0 padding
        self.node_proj = nn.Linear(
            ELEM_DIM + NODE_FEATURE_DIM_V3, HIDDEN)      # 21 → 48
        self.z_head = nn.Linear(2 * HIDDEN, Z_DIM_V3)    # 96 → 64
        # --- fresh, zero-initialised graph-conditioning injection ---------
        self.W_g = nn.Linear(GRAPH_COND_DIM, HIDDEN)     # 5 → 48 (288)
        # --- fresh per-layer PaiNN-lite cells -----------------------------
        # phi_s: scalar message MLP (mapped from R3 message_mlp.0/1).
        self.phi_s = nn.ModuleList(
            [_scalar_message_mlp() for _ in range(N_LAYERS)])
        # W_s: per-channel scalar gate; W_v: radial vector coefficient.
        self.W_s = nn.ModuleList([
            nn.Linear(HIDDEN, VECTOR_CHANNELS)
            for _ in range(N_LAYERS)])
        self.W_v = nn.ModuleList([
            nn.Linear(HIDDEN, VECTOR_CHANNELS)
            for _ in range(N_LAYERS)])
        # phi_u: scalar update [s, ‖V‖] → s (final output zero-init).
        self.phi_u = nn.ModuleList([
            nn.Sequential(
                nn.Linear(VECTOR_COUPLING_DIM, HIDDEN), nn.SiLU(),
                nn.Linear(HIDDEN, HIDDEN))
            for _ in range(N_LAYERS)])
        # phi_g: vector gates [a_v, b_v] = [s, ‖V‖] → 64 (zero-init).
        # (Oracle §2.2 names this projection φ_uv; the frozen config/ADR
        # zero-init row calls it φ_g — same module.)
        self.phi_g = nn.ModuleList([
            nn.Linear(VECTOR_COUPLING_DIM, GATE_DIM)
            for _ in range(N_LAYERS)])
        # scalar-only LayerNorm (NEVER applied to vector channels).
        self.norm = nn.ModuleList([
            nn.LayerNorm(HIDDEN) for _ in range(N_LAYERS)])
        # V_0 = 0: the initial vector state is a zero BUFFER (the Oracle
        # parameter estimate has no V_0 term; a learnable V_0 would also
        # break O(3) equivariance the moment it left zero).
        self.register_buffer(
            "V_0", torch.zeros(3, VECTOR_CHANNELS), persistent=True)

        measured = sum(int(p.numel()) for p in self.parameters())
        if measured != ESTIMATED_ACTIVE_PARAMS:
            # Exact-assertion discipline (ADR-0033 §4): never pad/shrink
            # silently — refuse construction with the measured count.
            raise V542InitError(
                "V542_A2_PARAMS",
                f"A2Backbone has {measured} parameters, the frozen estimate "
                f"is {ESTIMATED_ACTIVE_PARAMS} — per-component breakdown "
                f"required: {self.param_breakdown()}")

    # -- accounting ---------------------------------------------------------
    @property
    def active_param_count(self) -> int:
        """Active backbone params (A2 owns no decoder ``w_v``)."""
        return sum(int(p.numel()) for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        """Per-top-level-module parameter counts (audit / drift report)."""
        groups: dict[str, int] = {}
        for name, param in self.named_parameters():
            prefix = name.split(".", 1)[0]
            groups[prefix] = groups.get(prefix, 0) + int(param.numel())
        return dict(sorted(groups.items()))

    # -- input helpers ------------------------------------------------------
    @staticmethod
    def _require_finite(name: str, tensor: torch.Tensor) -> None:
        """Legacy typed-invalid (``NONFINITE_INPUT``), never a bare error."""
        if not bool(torch.isfinite(tensor).all()):
            raise KeyAtomV3InvalidInputError(
                "NONFINITE_INPUT", f"{name} contains non-finite values")

    @staticmethod
    def _edge_input(batch: Mapping[str, torch.Tensor],
                    dtype: torch.dtype) -> torch.Tensor:
        """Packed preserved slots ⊕ intent scalars → the 22-dim edge view."""
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
        """Frozen forward-view batch → ``BackboneOutput(z, edge_msg)``.

        fp32, deterministic, no coordinate updates, ``edge_msg`` rows in
        ``batch["edge_index"]`` column order with a live autograd graph.
        """
        dtype = self.node_proj.weight.dtype
        device = self.node_proj.weight.device

        # Input legality (ReactionIntent Contract v1 §5): forward-view
        # whitelist first — supervision keys never reach the backbone.
        check_forward_view_keys_v3(batch)
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
        cv_delta = batch[GRAPH_CV_DELTA_KEY].to(dtype=dtype)
        signed = batch[GRAPH_SIGNED_INTENT_KEY].to(dtype=dtype)

        for name, tensor in (
                ("pos", pos), ("edge_attr", edge_in), ("degK", degk),
                ("intent_role", role), ("cv_target_delta", cv_delta),
                ("signed_intent", signed)):
            self._require_finite(name, tensor)

        n_nodes = int(pos.shape[0])
        n_keys = int(key_index.shape[0])
        n_graphs = int(cv_delta.shape[0])
        src, dst = edge_index[0], edge_index[1]

        # --- s_0: element ⊕ node features, + zero-init graph cond injection
        degk_norm = degk.clamp(min=0.0, max=DEGK_SCALE) / DEGK_SCALE
        s = self.node_proj(torch.cat(
            [self.elem_emb(z_atomic.to(device=device)),
             torch.cat([is_key_f.unsqueeze(-1), degk_norm.unsqueeze(-1),
                        role], dim=-1)], dim=-1))
        graph_cond = torch.cat([cv_delta, signed.unsqueeze(-1)], dim=-1)
        s = s + self.W_g(graph_cond)[graph_index]

        # --- V_0 = 0 (zero buffer broadcast to every node) ------------------
        V = self.V_0.unsqueeze(0).expand(n_nodes, -1, -1)

        # --- deterministic current-frame polar edge directions ---------------
        rel = pos[src] - pos[dst]                       # receiver = dst (Star)
        rhat = rel / (rel.norm(dim=-1, keepdim=True) + R_HAT_EPS)

        # --- 2 PaiNN-lite message-passing layers ----------------------------
        for layer in range(N_LAYERS):
            # 1. scalar message (warm-started Star-shaped MLP)
            m = self.phi_s[layer](
                torch.cat([s[src], s[dst], edge_in], dim=-1))
            # 2. vector message: neighbour transfer + radial message
            gate = self.W_s[layer](m)                   # [E,32] per-channel
            radial = self.W_v[layer](m)                 # [E,32] radial coeff
            v_msg = (V[src] * gate.unsqueeze(1)
                     + radial.unsqueeze(1) * rhat.unsqueeze(2))  # [E,3,32]
            v_agg = torch.zeros(
                n_nodes, 3, VECTOR_CHANNELS, dtype=dtype,
                device=device).index_add(0, dst, v_msg)
            # 3. gated vector update from INVARIANT scalars only;
            #    ‖V_i‖ is the per-channel norm over the 3 SPATIAL
            #    components → [N, 32] (O(3)-invariant), fed to the
            #    80 = 48 + 32 coupling width of phi_g / phi_u.
            vnorm = torch.sqrt((V * V).sum(dim=1) + VECTOR_NORM_EPS)
            a_v, b_v = self.phi_g[layer](
                torch.cat([s, vnorm], dim=-1)).chunk(2, dim=-1)
            V = (torch.sigmoid(a_v).unsqueeze(1) * V
                 + torch.sigmoid(b_v).unsqueeze(1) * v_agg)
            # 4. scalar update through ‖V‖ (the ONLY V → s pathway)
            vnorm = torch.sqrt((V * V).sum(dim=1) + VECTOR_NORM_EPS)
            s = s + self.phi_u[layer](torch.cat([s, vnorm], dim=-1))
            # 5. LayerNorm on the scalar channel ONLY (never on V)
            s = self.norm[layer](s)

        # --- INTENTIONAL A2 semantics (ADR-0033 §4): edge_msg is RECOMPUTED
        # from the final scalar state s^L through the SAME (last-layer)
        # phi_s, so the vector channel — which reached s^L via ‖V‖ — acts
        # on MOVE.  Star-style capture would sever that path; this is a
        # documented difference, NOT a bug, and must not be "fixed".
        edge_msg = self.phi_s[-1](
            torch.cat([s[src], s[dst], edge_in], dim=-1))

        # --- z head: graph mean ⊕ key mean → 64, tanh (frozen recipe) -------
        key_graph = graph_index[key_index]
        k_sum = torch.zeros(n_graphs, HIDDEN, dtype=dtype, device=device)
        k_sum = k_sum.index_add(0, key_graph, s[key_index])
        k_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, key_graph,
                torch.ones(n_keys, dtype=dtype, device=device))
        k_mean = k_sum / k_count.clamp(min=1.0).unsqueeze(-1)

        g_sum = torch.zeros(n_graphs, HIDDEN, dtype=dtype, device=device)
        g_sum = g_sum.index_add(0, graph_index, s)
        g_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, graph_index, torch.ones_like(graph_index, dtype=dtype))
        g_mean = g_sum / g_count.clamp(min=1.0).unsqueeze(-1)

        z = torch.tanh(self.z_head(torch.cat([g_mean, k_mean], dim=-1)))
        return BackboneOutput(z=z, edge_msg=edge_msg)


# ---------------------------------------------------------------------------
# frozen construction protocol (config init_policy.construction_order 1→3)
# ---------------------------------------------------------------------------
def _apply_a2_zero_init(backbone: A2Backbone) -> None:
    """Construction step 3: the frozen ``per_candidate_zero_init.A2`` row.

    ``V_0 = 0`` (buffer), ``W_g`` zero, and the ``phi_u`` / ``phi_g``
    FINAL output projections zero (cold start: the V→s gate is closed;
    ``W_s``/``W_v`` keep their default seeded init).  Runs AFTER the R3
    mapping — none of these slots is a mapping target, so the order is
    auditable either way.
    """
    with torch.no_grad():
        backbone.V_0.zero_()
    zero_init_(backbone.W_g)
    for seq in backbone.phi_u:
        final = seq[-1]
        if not isinstance(final, nn.Linear):  # pragma: no cover - build guard
            raise V542InitError(
                "V542_A2_ZERO_INIT",
                f"phi_u final module is {type(final).__name__}, expected "
                "nn.Linear (frozen zero-init slot)")
        zero_init_(final)
    for gate in backbone.phi_g:
        zero_init_(gate)


#: The canonical frozen construction steps THIS builder executes (config
#: ``init_policy.construction_order`` 1–4: fresh modules, R3 mapping,
#: A2 zero-inits, shared ``w_v`` from R3).  Steps 5–6 are
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
    """Build the A2 backbone under the frozen construction protocol;
    returns the audit-bearing :class:`BuiltCandidate` record.

    Runs ENTIRELY inside :func:`common.construction_scope`
    (``seed_init(run_seed, resolve_init_source_id("A2"))``): fresh
    modules → :func:`common.apply_r3_mapping` (A1 common set + two
    same-shape ``phi_s``) → :func:`_apply_a2_zero_init` →
    :func:`common.apply_r3_w_v` (the SHA-verified R3 ``w_v`` — or the
    SAME injected ``source`` the mapping used — is loaded into the
    SHARED ``DirectMoveDecoder``; every builder does this, the caller
    only supplies the decoder OBJECT) → decoder ``w_v``
    ``torch.equal`` check → :func:`common.assert_candidate_budget`
    (exact 54,896).  The global training seed is NOT reset here
    (construction_scope restores the caller's RNG stream
    byte-identically; ``finish_training_seed`` stays the runner's
    responsibility).

    ``r3_state`` defaults to the SHA-pinned
    :func:`common.load_r3_backbone_state` payload (37 keys); tests may
    inject an already-loaded mapping source.  ``move_decoder`` is the
    shared ``DirectMoveDecoder`` seam (A2 owns no ``w_v`` slot — MOVE is
    the decoder's 49 params, R3-initialised by step 4); a non-decoder is
    a typed refusal, as is a decoder whose frozen ``a_ref`` was tampered
    with (gate-1 B1 / gate-2 B2).
    """
    if not isinstance(move_decoder, DirectMoveDecoder):
        raise V542InitError(
            "V542_A2_DECODER",
            "A2 build requires the shared DirectMoveDecoder seam "
            f"(got {type(move_decoder).__name__})")
    assert_decoder_a_ref_frozen(move_decoder)   # gate-2 B2: same seam as
    # every other candidate — a tampered a_ref is refused before dispatch
    source: Mapping[str, torch.Tensor] = (
        load_r3_backbone_state() if r3_state is None else dict(r3_state))

    with construction_scope(run_seed,
                            resolve_init_source_id(CANDIDATE_ID)) as seed:
        backbone = A2Backbone()
        mapping = dict(A2_R3_MAPPING)
        mapped_sources = set(mapping.values())
        consumed = [key for key in sorted(source)
                    if key not in mapped_sources]
        audit = apply_r3_mapping(backbone, source, mapping,
                                 consumed_source_keys=consumed)
        _apply_a2_zero_init(backbone)
        apply_r3_w_v(move_decoder, source)

    # every build ends with a bitwise-equal decoder w_v (B1 / gate 2)
    if not (torch.equal(move_decoder.w_v.weight, source["w_v.weight"])
            and torch.equal(move_decoder.w_v.bias, source["w_v.bias"])):
        raise V542InitError(  # pragma: no cover - copy_ cannot diverge
            "V542_WV_NOT_EQUAL",
            "A2: decoder w_v is not torch.equal to the resolved source "
            "w_v after construction step 4")

    assert_candidate_budget(backbone,
                            expected_active=ESTIMATED_ACTIVE_PARAMS)
    return BuiltCandidate(
        candidate_id=CANDIDATE_ID,
        backbone=backbone,
        move_decoder=move_decoder,
        mapping_audit=audit,
        seed_init=seed,
        init_source_id=resolve_init_source_id(CANDIDATE_ID),
        executed_steps=EXECUTED_STEPS,
    )


def build(
        run_seed: int,
        move_decoder: DirectMoveDecoder,
        *,
        r3_state: Mapping[str, torch.Tensor] | None = None,
) -> A2Backbone:
    """Convenience wrapper over :func:`build_record` returning ONLY the
    backbone (the audit-bearing record stays available for the registry
    / P3 run-identity path)."""
    return build_record(run_seed, move_decoder, r3_state=r3_state).backbone


__all__ = [
    "A2_BACKBONE_VERSION",
    "A2_R3_MAPPING",
    "A2Backbone",
    "CANDIDATE_ID",
    "ESTIMATED_ACTIVE_PARAMS",
    "EXECUTED_STEPS",
    "GRAPH_COND_DIM",
    "HIDDEN",
    "N_LAYERS",
    "VECTOR_CHANNELS",
    "VECTOR_COUPLING_DIM",
    "build",
    "build_record",
]
