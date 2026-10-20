"""R3 v3 Star EGNN for intent-directed key-atom displacement.

Model version ``pesn_key_atom_star_egnn_v3`` (campaign ``r3_training.model``:
StarEGNN backbone, hidden 48, 3 layers, F3 superset input slots, fp32,
deterministic forward).  Consumes the dict batch produced by
:func:`pes2ts_core.pesn.features.key_atom_graph_v3.collate_v3` (forward keys
only — :data:`~pes2ts_core.pesn.contracts.key_atom_v3.FORWARD_VIEW_KEYS_V3`)
and returns ``{"u": [ΣK,3], "z": [G,64]}`` — key-atom displacements in
``key_index`` order plus a 64-dim invariant graph summary (kept for the
V5.2 progress/stop heads, 30h plan §5.1).

Contract (R3 plan v3 §3.2 + campaign yaml):

* widths: hidden 48, 3 interaction layers.  Node input = element embedding
  16 ⊕ {is_key, degK_norm, intent_role 3} = 21 → linear 21→48; edge input =
  packed preserved slots (17: RBF12 + topology + unordered bond context) ⊕
  {b_src, b_tgt, delta_b_intent, abs_delta, is_changed} = 22 → message MLP
  (48+48+22)→48→48; node residual MLP (48+48+graph_cond 5)→48→48 with the
  directed intent conditioning (cv_target_delta 4 ⊕ signed_intent 1)
  broadcast per graph; LayerNorm on scalar channels only;
* message edges = the batch's directed edge set (covalent ∪ radius ∪ intent
  pairs; no extra model inputs);
* vector head (per key atom i, last-layer messages m_ij over its incident
  message edges)::

      δv_i = a_ref / max(1, degK_i) · Σ_j tanh(w_v·m_ij)·r_ij/(‖r_ij‖+0.1)

  with SIGNED tanh coefficients, a_ref = 0.08 Å, r_ij the current-frame
  polar edge vector.  There is NO anchor/X0 pathway in v3 — the ReactionIntent
  Contract v1 §5 forbids history-side inputs;
* z head: graph mean-pool h ⊕ key mean-pool h → linear → 64, tanh;
* O(3)-equivariance by construction: the vector pathway consumes only polar
  vectors scaled by invariant scalars; no LayerNorm/ReLU/dropout on vector
  channels; no dropout anywhere; positions are NEVER updated;
* forward determinism: no dropout/batchnorm/stochastic layers; two eval-mode
  calls on the same batch are bitwise equal (asserted in
  ``tests/pesn/test_r3_model.py``); fp32 throughout;
* input legality: only :data:`FORWARD_VIEW_KEYS_V3` are read; non-finite
  coordinates/features raise :class:`KeyAtomV3InvalidInputError` (typed
  invalid — never a fabricated action, GeometryPolicyContract v1 §S4.3);
* construction-time budget: ``sum(p.numel()) ≤ 70_000`` (campaign
  ``~59k``-class; exact count in :data:`MEASURED_PARAMS_V3` — v3 drops the
  v1 branch/method/solvent task-19 block and the X0 anchor pathway, both
  illegal here, and adds the directed-CV/intent conditioning)."""

from __future__ import annotations

from typing import Final, Mapping

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
    EDGE_INPUT_DIM_V3,
    EDGE_IS_CHANGED_KEY,
    GRAPH_CV_DELTA_KEY,
    GRAPH_SIGNED_INTENT_KEY,
    MODEL_VERSION,
    NODE_DEGK_KEY,
    NODE_GRAPH_KEY,
    NODE_INTENT_ROLE_KEY,
    NODE_IS_KEY_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    NODE_Z_KEY,
    check_forward_view_keys_v3,
)

STAR_EGNN_V3_VERSION: Final[str] = MODEL_VERSION

#: Element embedding width (plan §3.2: ``nn.Embedding(119, 16)``).
ELEM_DIM: Final[int] = 16
#: Node scalar width: is_key(1) + degK_norm(1) + intent_role(3).
NODE_FEATURE_DIM_V3: Final[int] = 5
#: Graph conditioning width: cv_target_delta(CV_MAX) + signed_intent(1).
GRAPH_COND_DIM_V3: Final[int] = CV_MAX + 1
#: Construction-time parameter budget (campaign "~59k … 少量头扩展"; test
#: asserts the measured count below this and reports the exact value).
PARAM_BUDGET_V3: Final[int] = 70_000

#: Vector-head reference step (Å) — a_ref of the v1 contract (0.08 Å = d0).
A_REF_V3: Final[float] = 0.08

Z_DIM_V3: Final[int] = 64


class KeyAtomV3InvalidInputError(RuntimeError):
    """Typed invalid for out-of-contract model input (NaN/inf/Z range).

    Name and ``code``/``detail`` attributes match the feature-gate harness
    typed-invalid classification (GeometryPolicyContract v1 §S4.3: never
    fabricate an action).
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class StarEGNNV3(nn.Module):
    """R3 v3 Star EGNN (see module docstring for the full contract)."""

    def __init__(
        self,
        *,
        hidden: int = 48,
        n_layers: int = 3,
        a_ref: float = A_REF_V3,
        param_budget: int = PARAM_BUDGET_V3,
    ) -> None:
        super().__init__()
        if hidden != 48 or n_layers != 3:
            # The campaign freezes hidden48/3 layers; other widths would
            # silently leave the registered model version.
            raise RuntimeError(
                f"{STAR_EGNN_V3_VERSION} fixes hidden=48, n_layers=3 "
                f"(got {hidden}/{n_layers})")
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.a_ref = float(a_ref)

        self.elem_emb = nn.Embedding(119, ELEM_DIM)  # Z=1..118; 0 = padding
        self.node_proj = nn.Linear(
            ELEM_DIM + NODE_FEATURE_DIM_V3, self.hidden)
        msg_in = self.hidden + self.hidden + EDGE_INPUT_DIM_V3   # 48+48+22
        node_in = self.hidden + self.hidden + GRAPH_COND_DIM_V3  # 48+48+5
        self.message_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(msg_in, self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, self.hidden))
            for _ in range(self.n_layers)])
        self.node_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_in, self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, self.hidden))
            for _ in range(self.n_layers)])
        self.norm_h = nn.ModuleList([
            nn.LayerNorm(self.hidden) for _ in range(self.n_layers)])
        # signed-tanh vector-head coefficient head (scalar output only)
        self.w_v = nn.Linear(self.hidden, 1)
        # z head: graph pool ⊕ key pool -> 64, tanh (kept for V5.2 heads)
        self.z_head = nn.Linear(2 * self.hidden, Z_DIM_V3)

        n_params = sum(p.numel() for p in self.parameters())
        if n_params > int(param_budget):
            raise RuntimeError(
                f"{STAR_EGNN_V3_VERSION} has {n_params} parameters > budget "
                f"{param_budget}")

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

    # -- forward -------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return ``{"u": [ΣK,3], "z": [G,64]}`` (fp32; deterministic)."""
        dtype = self.node_proj.weight.dtype
        device = self.node_proj.weight.device

        z_atomic = batch[NODE_Z_KEY]
        # Input legality (ReactionIntent Contract v1 §5): only forward-view
        # keys may reach the model — supervision must stay outside.
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

        for name, tensor in (
                ("pos", pos), ("edge_attr", edge_in), ("degK", degk),
                ("intent_role", role), ("cv_target_delta", cv_delta),
                ("signed_intent", signed)):
            self._require_finite(name, tensor)

        n_nodes = pos.shape[0]
        n_keys = key_index.shape[0]
        n_graphs = int(cv_delta.shape[0])
        src, dst = edge_index[0], edge_index[1]

        # node scalar init: element embedding ⊕ v3 node features
        degk_norm = degk.clamp(min=0.0, max=4.0) / 4.0
        h = self.node_proj(torch.cat(
            [self.elem_emb(z_atomic.to(device=device)),
             torch.cat([is_key_f.unsqueeze(-1), degk_norm.unsqueeze(-1),
                        role], dim=-1)], dim=-1))
        graph_cond = torch.cat([cv_delta, signed.unsqueeze(-1)], dim=-1)
        cond_nodes = graph_cond[graph_index]      # broadcast per graph

        m_edge_attr = edge_in
        last_messages: torch.Tensor | None = None
        for layer in range(self.n_layers):
            msg_in = torch.cat([h[src], h[dst], m_edge_attr], dim=-1)
            messages = self.message_mlp[layer](msg_in)
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

        # --- key-pool scalars (z head; pure index_add over key rows) -------
        key_graph = graph_index[key_index]
        k_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        k_sum = k_sum.index_add(0, key_graph, h[key_index])
        k_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, key_graph,
                torch.ones(n_keys, dtype=dtype, device=device))
        k_mean = k_sum / k_count.clamp(min=1.0).unsqueeze(-1)

        # --- vector head (key atoms only; signed tanh; no coord updates) ---
        u = torch.zeros(n_keys, 3, dtype=dtype, device=device)
        if last_messages is not None and last_messages.shape[0] > 0:
            coeff = torch.tanh(self.w_v(last_messages))       # [Em,1] signed
            rel = pos[src] - pos[dst]                          # r_ij, recv=dst
            unit = rel / (rel.norm(dim=-1, keepdim=True) + 0.1)
            contrib = coeff * unit
            key_slot = torch.full(
                (n_nodes,), -1, dtype=torch.long, device=device)
            key_slot[key_index] = torch.arange(
                n_keys, dtype=torch.long, device=device)
            dst_slot = key_slot[dst]
            keep = dst_slot >= 0
            if bool(keep.any()):
                u = u.index_add(0, dst_slot[keep], contrib[keep])
        deg_scale = (self.a_ref / degk[key_index].clamp(min=1.0)).unsqueeze(-1)
        u = deg_scale * u

        # --- z head: graph pool ⊕ key pool -> 64, tanh --------------------
        g_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        g_sum = g_sum.index_add(0, graph_index, h)
        g_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, graph_index, torch.ones_like(graph_index, dtype=dtype))
        g_mean = g_sum / g_count.clamp(min=1.0).unsqueeze(-1)
        z = torch.tanh(self.z_head(torch.cat([g_mean, k_mean], dim=-1)))
        return {"u": u, "z": z}

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def measured_params_v3() -> int:
    """Exact ``sum(p.numel())`` of the default StarEGNNV3 (fresh build)."""
    return StarEGNNV3().n_parameters


#: Recorded count of the default build (exact; asserted by the model test).
MEASURED_PARAMS_V3: Final[int] = 55441


__all__ = [
    "A_REF_V3",
    "ELEM_DIM",
    "GRAPH_COND_DIM_V3",
    "KeyAtomV3InvalidInputError",
    "MEASURED_PARAMS_V3",
    "NODE_FEATURE_DIM_V3",
    "PARAM_BUDGET_V3",
    "STAR_EGNN_V3_VERSION",
    "StarEGNNV3",
    "Z_DIM_V3",
    "measured_params_v3",
]
