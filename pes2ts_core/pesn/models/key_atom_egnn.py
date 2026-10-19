"""M2 Star EGNN for key-atom displacement (ADR-0026 §2.7, plan §3 M2).

Model version ``pesn_key_atom_star_egnn_v1``.  Consumes a
``KeyAtomBatch`` (``pes2ts_core.pesn.contracts.key_atom``) and returns
``(u [Σm,3], z [B,64])`` — displacement vectors for the key atoms only
(gathered via ``key_index``) and a 64-dim invariant scalar summary per
graph.

Contract (ADR-0026 §2.7 / plan §3 M2, verbatim where normative):

* widths: hidden 48, 3 interaction layers; node input = element
  embedding 16 + ``h_node_inv`` 20 = 36 → linear 36→48; message MLP
  (48+48+29)→48→48; node residual MLP (48+48+task19)→48→48;
  LayerNorm on scalar channels only;
* message subgraph: nodes within 2 covalent hops of any key atom ∪
  nodes within 4.5 Å of any key atom; message edges = covalent/K edges
  with both endpoints selected ∪ star spatial edges centred at the key
  atoms (selected node ↔ key atom with ‖r‖ < 4.5 Å).  Both the node
  selection and the star edges are derived at runtime from the batch's
  edge graph (covalent/K flags in ``edge_attr``) and the current
  positions — no extra model inputs;
* vector head (per key atom i, last-layer messages m_ij over its
  incident message edges)::

      δv_i = a_ref / max(1, degK_i) · [ Σ_j tanh(w_v·m_ij)·r_ij/(‖r_ij‖+0.1)
                                        + c_a·tanh(w_a·h_i)·a_i/‖a_i‖·m_soft ]

  with SIGNED tanh coefficients (never sigmoid/softplus-only), a_ref
  = 0.08 Å, r_ij the current-frame polar edge vector, a_i the anchor
  polar vector, and the differentiable soft mask
  m_soft = ‖a_i‖/(‖a_i‖+1e-6) (anchor-off batches — anchor slots zeroed
  upstream — contribute exactly 0 through this gate);
* z head: graph mean-pool h ⊕ key mean-pool h → linear → 64, tanh;
* O(3) equivariance by construction: the vector pathway consumes only
  polar vectors scaled by scalar coefficients; no LayerNorm/ReLU/
  dropout on vector channels (dropout omitted entirely — plan M2 does
  not mandate it and omitting keeps the equivariance audit trivial);
  physical positions are NEVER updated;
* construction-time budget: ``sum(p.numel()) ≤ 90_000`` (plan §3 M2
  ≤90k; measured count in :data:`MEASURED_PARAMS`).

DEVIATION REGISTER: the ADR/plan normative message width is
(48+48+29)→48→48 consuming the raw 29-dim edge attribute, so no separate
29→48 edge embedding layer exists (an unused embedding would only burn
budget); everything else follows ADR-0026 §2.7 verbatim.
"""
from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn

from pes2ts_core.pesn.contracts.key_atom import (
    EDGE_DIM,
    EDGE_SLOTS,
    NODE_INV_DIM,
    TASK_DIM,
    Z_DIM,
    KeyAtomBatch,
)

STAR_EGNN_VERSION: Final[str] = "pesn_key_atom_star_egnn_v1"

#: Element embedding width (plan §2 node36 = elem 16 ⊕ inv 20); the
#: contract's ``Z_DIM`` constant is the 64-dim z-head width instead.
ELEM_DIM: Final[int] = 16

#: Construction-time parameter budget (plan §3 M2: ≤90k).
PARAM_BUDGET: Final[int] = 90_000

#: Measured ``sum(p.numel())`` for the default StarEGNN (hidden48/3 layers),
#: recorded after build per ADR-0026 §2.7 "实测落盘".
MEASURED_PARAMS: Final[int] = 59_235

#: Measured ``sum(p.numel())`` with the V5.1a magnitude head enabled
#: (base 59,235 + 2-layer magnitude MLP 48→48→1 = 2,401).
MEASURED_PARAMS_WITH_HEAD: Final[int] = 61_636

#: V5.1a magnitude head (plan §3.2): direction soft-norm eps (Å).
MAG_DIRECTION_EPS: Final[float] = 1e-4

#: Accepted ``magnitude_mode`` spellings (plan §3.2 fallback ladder).
MAGNITUDE_MODES: Final[tuple[str, ...]] = ("per_atom", "graph")

#: Default star / spatial radius (Å) — plan §3 M2 and §2 graph contract.
DEFAULT_RADIUS: Final[float] = 4.5

#: Vector-head reference step (Å), plan §3 M2 a_ref.
A_REF: Final[float] = 0.08

#: V5.1a magnitude head: EFFECTIVE per-atom magnitude target at init (Å).
#: The forward pass computes ``u = s_eff·v_hat`` with
#: ``s_eff = softplus(head(h)) · deg_scale`` and
#: ``deg_scale = a_ref/max(1, degK)`` — the head's raw softplus output is
#: scaled by degK BEFORE reaching ``u``, so L_mag (which supervises the
#: effective magnitude) must calibrate against s_eff, NOT the raw head
#: output (bug: supervising raw s at degK=2 collapsed ‖u‖ by 4×).
#: Calibration: degK_typ = 2 → deg_scale_typ = a_ref/2 = 0.04 Å →
#: softplus(b) = 0.05/0.04 = 1.25 → b = ln(expm1(1.25)) ≈ 0.9124.
MAG_INIT_TARGET: Final[float] = 0.05         # EFFECTIVE (post deg_scale)
MAG_DEGK_TYP: Final[int] = 2
MAG_DEG_SCALE_TYP: Final[float] = A_REF / MAG_DEGK_TYP   # 0.04 Å
#: Raw softplus output required for the effective init target.
MAG_RAW_INIT_TARGET: Final[float] = MAG_INIT_TARGET / MAG_DEG_SCALE_TYP
MAG_BIAS_INIT: Final[float] = math.log(math.expm1(MAG_RAW_INIT_TARGET))

#: Number of covalent hops from the key-atom set defining the message
#: subgraph (plan §3 M2: "A 的 2 共价 hop").
HOPS: Final[int] = 2

_COV_S = EDGE_SLOTS["covalent_start"]
_COV_O = EDGE_SLOTS["covalent_other"]
_IS_K = EDGE_SLOTS["is_k"]


class KeyAtomModelError(RuntimeError):
    """Typed model refusal (machine-readable code prefix)."""


class StarEGNN(nn.Module):
    """M2 Star EGNN (ADR-0026 §2.7, plan §3): subgraph message passing +
    signed-tanh key-atom vector head.  See module docstring for the full
    contract."""

    def __init__(
        self,
        *,
        hidden: int = 48,
        n_layers: int = 3,
        a_ref: float = A_REF,
        radius: float = DEFAULT_RADIUS,
        hops: int = HOPS,
        param_budget: int = PARAM_BUDGET,
        magnitude_head: bool = False,
        magnitude_mode: str = "per_atom",
    ) -> None:
        super().__init__()
        if hidden != 48 or n_layers != 3:
            # The ADR freezes hidden48/3 layers for this sprint; other
            # widths would silently leave the registered model version.
            raise KeyAtomModelError(
                "STAR_EGNN_FIXED_SHAPE",
                f"{STAR_EGNN_VERSION} fixes hidden=48, n_layers=3 "
                f"(got {hidden}/{n_layers})")
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.a_ref = float(a_ref)
        self.radius = float(radius)
        self.hops = int(hops)
        self.magnitude_head = bool(magnitude_head)
        self.magnitude_mode = str(magnitude_mode)
        if self.magnitude_head and self.magnitude_mode not in MAGNITUDE_MODES:
            raise KeyAtomModelError(
                "STAR_EGNN_MAGNITUDE_MODE",
                f"magnitude_mode must be one of {MAGNITUDE_MODES}, got "
                f"{magnitude_mode!r}")
        #: Per-key-atom EFFECTIVE magnitudes ``s_eff = s·deg_scale`` (Å)
        #: of the LAST forward pass when the head is enabled
        #: (differentiable; None otherwise) — i.e. the per-atom magnitude
        #: that actually reaches ``u`` (``u = s_eff·v_hat``, so
        #: ``s_eff ≥ ‖u‖`` since ``‖v_hat‖ ≤ 1``).  The trainer's L_mag
        #: supervises THIS quantity against ‖y‖ (V5.1a plan §3.3); the
        #: raw head output is the degK-scaled-down pre-factor, NOT the
        #: delivered displacement.
        self.last_magnitudes: torch.Tensor | None = None
        #: Debug stash of the raw softplus head output ``s`` (pre
        #: deg_scale scaling); never supervised directly.
        self.last_raw_magnitudes: torch.Tensor | None = None

        self.elem_emb = nn.Embedding(119, ELEM_DIM)  # Z=1..118 valid, 0 padding
        self.node_proj = nn.Linear(ELEM_DIM + NODE_INV_DIM, self.hidden)
        msg_in = self.hidden + self.hidden + EDGE_DIM  # 48+48+29 = 125
        node_in = self.hidden + self.hidden + TASK_DIM  # 48+48+19 = 115
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
        # signed-tanh vector-head coefficient heads (scalar outputs only)
        self.w_v = nn.Linear(self.hidden, 1)
        self.w_a = nn.Linear(self.hidden, 1)
        self.c_a = nn.Parameter(torch.tensor(1.0))
        # z head: graph pool ⊕ key pool -> 64, tanh (plan §3 M2)
        self.z_head = nn.Linear(2 * self.hidden, Z_DIM)

        # --- V5.1a magnitude head (plan §3.2; CONDITIONAL, head-off
        # state_dict stays bit-compatible with V5 checkpoints) ---
        if self.magnitude_head:
            self.mag_head = nn.Sequential(
                nn.Linear(self.hidden, self.hidden), nn.SiLU(),
                nn.Linear(self.hidden, 1))
            final = self.mag_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, MAG_BIAS_INIT)

        n_params = sum(p.numel() for p in self.parameters())
        if n_params > int(param_budget):
            raise KeyAtomModelError(
                "STAR_EGNN_PARAM_BUDGET",
                f"{STAR_EGNN_VERSION} has {n_params} parameters > budget "
                f"{param_budget}")
        if self.magnitude_head and n_params != MEASURED_PARAMS_WITH_HEAD:
            raise KeyAtomModelError(
                "STAR_EGNN_PARAM_COUNT",
                f"magnitude-head parameter count {n_params} != recorded "
                f"{MEASURED_PARAMS_WITH_HEAD}")

    # -- helpers -------------------------------------------------------------

    def _select_nodes(
        self, batch: KeyAtomBatch, is_key: torch.Tensor
    ) -> torch.Tensor:
        """Boolean node mask: 2 covalent hops from A ∪ within radius of A.

        The covalent hop propagation walks the batch edge graph using the
        covalent flags in ``edge_attr`` (double-directed edges per the
        feature contract); the radial selection uses the current
        positions against key-atom positions inside each graph (``ptr`` /
        ``key_ptr``).
        """
        pos = batch.pos
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        src, dst = edge_index[0], edge_index[1]
        cov = (edge_attr[:, _COV_S] > 0.5) | (edge_attr[:, _COV_O] > 0.5)

        reach = is_key.clone()
        for _ in range(self.hops):
            prop = torch.zeros(
                reach.shape[0], dtype=torch.float32, device=reach.device)
            prop = prop.index_add(0, dst[cov], reach[src[cov]].to(prop.dtype))
            reach = reach | (prop > 0.0)

        near = torch.zeros_like(reach)
        n_graphs = batch.ptr.shape[0] - 1
        for b in range(int(n_graphs)):
            nodes = torch.arange(
                int(batch.ptr[b]), int(batch.ptr[b + 1]), device=pos.device)
            keys = batch.key_index[
                int(batch.key_ptr[b]):int(batch.key_ptr[b + 1])]
            if keys.numel() == 0 or nodes.numel() == 0:
                continue
            d = torch.cdist(pos[nodes], pos[keys])  # [n_nodes, n_keys]
            near[nodes] = d.min(dim=1).values < self.radius
        return reach | near

    def _message_edge_mask(
        self,
        batch: KeyAtomBatch,
        selected: torch.Tensor,
        is_key: torch.Tensor,
    ) -> torch.Tensor:
        """Message edge set: covalent/K edges among selected nodes ∪ star
        spatial edges (selected node ↔ key atom, ‖r‖ < radius)."""
        src, dst = batch.edge_index[0], batch.edge_index[1]
        edge_attr = batch.edge_attr
        both = selected[src] & selected[dst]
        topo = ((edge_attr[:, _COV_S] > 0.5)
                | (edge_attr[:, _COV_O] > 0.5)
                | (edge_attr[:, _IS_K] > 0.5))
        rel = batch.pos[src] - batch.pos[dst]
        dist = rel.norm(dim=-1)
        star = (dist < self.radius) & (is_key[src] | is_key[dst])
        return both & (topo | star)

    # -- forward -------------------------------------------------------------

    def forward(self, batch: KeyAtomBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u [Σm,3], z [B,64])``; fp32 in training, dtype follows
        the module parameters (tests may run float64)."""
        dtype = self.node_proj.weight.dtype
        device = self.node_proj.weight.device
        z_atomic = batch.z_atomic
        if int(z_atomic.numel()) and int(z_atomic.max()) > 118:
            raise KeyAtomModelError(
                "Z_OUT_OF_RANGE", "z_atomic > 118 (padding/unknown Z rejected)")
        if int(z_atomic.numel()) and int(z_atomic.min()) < 1:
            raise KeyAtomModelError(
                "Z_PADDING", "z_atomic < 1 (padding rows are not valid input)")
        h_inv = batch.h_node_inv.to(dtype=dtype)
        edge_attr = batch.edge_attr.to(dtype=dtype)
        pos = batch.pos.to(dtype=dtype)
        anchor = batch.anchor_vec.to(dtype=dtype)
        task = batch.task19.to(dtype=dtype)
        src, dst = batch.edge_index[0], batch.edge_index[1]
        key_index = batch.key_index
        n_nodes = pos.shape[0]
        n_keys = key_index.shape[0]
        n_graphs = batch.ptr.shape[0] - 1

        is_key = torch.zeros(n_nodes, dtype=torch.bool,
                             device=pos.device)
        is_key[key_index.to(pos.device)] = True
        selected = self._select_nodes(batch, is_key)
        msg_mask = self._message_edge_mask(batch, selected, is_key)
        m_src, m_dst = src[msg_mask], dst[msg_mask]
        m_edge_attr = edge_attr[msg_mask]

        # node scalar init: element embedding ⊕ invariant scalars
        h = self.node_proj(torch.cat(
            [self.elem_emb(z_atomic), h_inv], dim=-1))
        task_nodes = task[batch.batch]  # broadcast task19 to nodes

        last_messages: torch.Tensor | None = None
        for layer in range(self.n_layers):
            msg_in = torch.cat([h[m_src], h[m_dst], m_edge_attr], dim=-1)
            messages = self.message_mlp[layer](msg_in)
            agg = torch.zeros_like(h)
            agg = agg.index_add(0, m_dst, messages)
            counts = torch.zeros(
                n_nodes, dtype=dtype, device=device).index_add(
                    0, m_dst, torch.ones_like(m_dst, dtype=dtype))
            agg = agg / counts.clamp(min=1.0).unsqueeze(-1)
            h = self.norm_h[layer](
                h + self.node_mlp[layer](
                    torch.cat([h, agg, task_nodes], dim=-1)))
            last_messages = messages

        # --- key-pool scalars (shared by the z head and the graph-mode
        # magnitude head; pure reorder of the z-head pooling — same
        # index_add on the same inputs, bit-identical results) ---
        key_graph = torch.repeat_interleave(
            torch.arange(n_graphs, device=device),
            (batch.key_ptr[1:] - batch.key_ptr[:-1]).long())
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
            coeff = torch.tanh(self.w_v(last_messages))  # [Em,1] signed
            rel = pos[m_src] - pos[m_dst]  # r_ij, i = receiver (dst)
            unit = rel / (rel.norm(dim=-1, keepdim=True) + 0.1)
            contrib = coeff * unit
            key_slot = torch.full(
                (n_nodes,), -1, dtype=torch.long, device=device)
            key_slot[key_index] = torch.arange(
                n_keys, dtype=torch.long, device=device)
            dst_slot = key_slot[m_dst]
            keep = dst_slot >= 0
            if bool(keep.any()):
                u = u.index_add(0, dst_slot[keep], contrib[keep])

        a_i = anchor[key_index]
        a_norm = a_i.norm(dim=-1, keepdim=True)
        soft_mask = a_norm / (a_norm + 1e-6)  # differentiable 1[‖a‖>1e-6]
        a_unit = a_i / (a_norm + 1e-8)
        a_coeff = self.c_a * torch.tanh(self.w_a(h[key_index]))  # signed
        deg_scale = (self.a_ref / batch.degK.to(dtype=dtype).clamp(min=1.0)
                     ).unsqueeze(-1)
        v_sum = u + a_coeff * soft_mask * a_unit  # summed vector channel
        if self.magnitude_head:
            # V5.1a plan §3.2 direction–magnitude decoupling: the direction
            # is the soft-normalized summed channel (zero vector → zero
            # vector, never a fabricated direction); the magnitude is a
            # positive invariant scalar per key atom (or one graph scalar).
            v_hat = v_sum / torch.sqrt(
                (v_sum * v_sum).sum(dim=-1, keepdim=True)
                + MAG_DIRECTION_EPS ** 2)
            if self.magnitude_mode == "graph":
                s = torch.nn.functional.softplus(
                    self.mag_head(k_mean[key_graph]).squeeze(-1))
            else:
                s = torch.nn.functional.softplus(
                    self.mag_head(h[key_index]).squeeze(-1))
            # EFFECTIVE magnitude: what actually reaches u (u =
            # deg_scale·s·v_hat = s_eff·v_hat).  L_mag supervises s_eff
            # against ‖y‖ — supervising the raw s would shrink the
            # delivered displacement by deg_scale (≈4× at degK=2).
            s_eff = s * deg_scale.squeeze(-1)
            self.last_magnitudes = s_eff
            self.last_raw_magnitudes = s
            u = deg_scale * (s.unsqueeze(-1) * v_hat)
        else:
            self.last_magnitudes = None
            self.last_raw_magnitudes = None
            u = deg_scale * v_sum

        # --- z head: graph pool ⊕ key pool -> 64, tanh ---
        g_sum = torch.zeros(n_graphs, self.hidden, dtype=dtype, device=device)
        g_sum = g_sum.index_add(0, batch.batch, h)
        g_count = torch.zeros(
            n_graphs, dtype=dtype, device=device).index_add(
                0, batch.batch, torch.ones_like(batch.batch, dtype=dtype))
        g_mean = g_sum / g_count.clamp(min=1.0).unsqueeze(-1)
        z = torch.tanh(self.z_head(torch.cat([g_mean, k_mean], dim=-1)))
        return u, z

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = [
    "A_REF",
    "DEFAULT_RADIUS",
    "ELEM_DIM",
    "HOPS",
    "KeyAtomModelError",
    "MAG_DIRECTION_EPS",
    "MAG_BIAS_INIT",
    "MAG_DEGK_TYP",
    "MAG_DEG_SCALE_TYP",
    "MAG_INIT_TARGET",
    "MAG_RAW_INIT_TARGET",
    "MAGNITUDE_MODES",
    "MEASURED_PARAMS",
    "MEASURED_PARAMS_WITH_HEAD",
    "PARAM_BUDGET",
    "STAR_EGNN_VERSION",
    "StarEGNN",
]
