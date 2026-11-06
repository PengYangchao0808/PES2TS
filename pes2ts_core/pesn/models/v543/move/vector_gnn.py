"""``M3`` Vector-GNN MOVE decoder (plan §6.2 :528, §4.5 :243, §6.2 :536,
§5.2 :288–289/:310, §6.4 :558–571, §11.1 ``move/vector_gnn.py`` :1101; WP5).

Normative sources, quoted as implemented
---------------------------------------

* **§6.2 :528 (M3 row)** — "key 图上 2 层标量—向量消息传递；hidden 56；边特征
  8 维（约 34.5K）".  :class:`VectorMessageLayer` is applied exactly twice
  (``n_layers = 2``) on the key graph delivered by ``key_edge_index`` /
  ``key_edge_attr``; scalar hidden width is 56 and the edge feature is the
  8-d ``key_edge_attr`` of §5.2 :289.
* **§6.2 :536 (M3 关键约束)** — "只在 key 图上做 1–2 层局部交互；不得重新运行
  原始几何 Encoder，不得对全部原子做稠密 attention".  The only topology used
  is ``key_edge_index`` (K nodes); there is no attention, no atom-level pass and
  no encoder call — the layer uses ``index_add`` scatter aggregation, i.e. a
  sparse local sum over the delivered key edges.
* **§5.2 :310** — "``Ek=0`` 允许；M3 必须能退化为无消息传递".  With empty
  ``key_edge_index`` every scatter sum is empty, the degree clamp makes the
  normaliser 1, the vector stream is unchanged (``V' = V``) and the scalar
  update receives a zero pooled message; :meth:`VectorMessageLayer.forward`
  is written so this is the exact degenerate path rather than a special case.
* **§6.2 :528 readout** — the residual is read out of the vector channels
  ("向量通道消息聚合"), ``δu = Σ_j c_j V_j``; §4.5 :243 "M3 | 输出 head 最后
  一层" is the zero-initialised :attr:`VectorGNNMoveDecoder.coeff`.
* **§6.4 :558–571** — the widths give :data:`M3_N_TRAINABLE_PARAMS` = 34,448
  measured at construction (band 34,200–37,800, close to the plan's "约
  34.5K"), asserted by :func:`assert_budget` at construction.
* **Edge row order** — the plan does not freeze ``key_edge_index``'s column
  order (§5.2 :288 says only "key-local 行号"), so the layer must not depend on
  it: every edge interaction enters through a sum over edges (``index_add``),
  which is commutative.  Shuffling the edge columns/attributes therefore only
  changes the floating-point summation ORDER; the tests pin that difference with
  an explicit ``atol`` (see ``test_v543_move_decoders.py``).
* **§6.1 :519** — the only input is the read-only :class:`MoveContextV1`.

Deterministic initialisation (WP1 ``fork_rng`` discipline)
----------------------------------------------------------

The build runs inside ``torch.random.fork_rng(devices=[])`` seeded with
:data:`M3_INIT_SEED_V543` (same pattern as ``v542/candidates/common.py::
construction_scope``), so construction is RNG-neutral for the caller and
reproducible for a given ``seed`` (exposed as ``init_seed``).

Discipline: no shared file is modified (package ``__init__`` exports are left
to the integrating agent), constants are ``Final`` and ``torch.compile`` is
never used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.models.v543.contracts import (
    GRAPH_S_DIM_V543,
    KEY_EDGE_ATTR_DIM_V543,
    KEY_S_DIM_V543,
    KEY_V_CHANNELS_V543,
    MoveContextV1,
    V543Error,
)
from pes2ts_core.pesn.models.v543.registry import (
    DecoderSpec,
    TRACK_MOVE,
    V543DecoderProtocolError,
    assert_budget,
    count_trainable_params,
)

#: The campaign id of this candidate (plan §6.2 :523–530).
M3_DECODER_ID: Final[str] = "M3"

#: Registered model version of the WP5 M3 implementation.
M3_MODEL_VERSION: Final[str] = "v543_m3_vector_gnn_v1"

#: Construction seed of the frozen ``fork_rng`` build.
M3_INIT_SEED_V543: Final[int] = 17

#: Scalar input width: ``graph_s`` 64 + ``key_s`` 48 + ``‖key_v‖`` 8 = 120.
M3_INPUT_DIM: Final[int] = (
    GRAPH_S_DIM_V543 + KEY_S_DIM_V543 + KEY_V_CHANNELS_V543)
#: Scalar hidden width (§6.2 :528 "hidden 56").
M3_HIDDEN_DIM: Final[int] = 56
#: Vector channels (the §5.2 :284 ``key_v`` basis).
M3_VECTOR_CHANNELS: Final[int] = KEY_V_CHANNELS_V543
#: Edge feature width (§5.2 :289 ``key_edge_attr [Ek,8]``).
M3_EDGE_DIM: Final[int] = KEY_EDGE_ATTR_DIM_V543
#: Message-passing depth (§6.2 :528 "2 层"; §6.2 :536 "1–2 层局部交互").
M3_N_LAYERS: Final[int] = 2

#: Exact measured ``sum(p.numel())`` of the frozen widths (plan §6.4 :569):
#: embed 120·56+56                        =  6,776
#: layer ×2: edge_in (2·56+8)·56+56        =  6,776
#:           edge_out 56·56+56             =  3,192
#:           vector gate 56·8 (no bias)    =    448
#:           node_out 56·56+56             =  3,192
#:           → 13,608 per layer ×2        = 27,216
#: head 56·8+8                            =    456
#: total                                  = 34,448
M3_N_TRAINABLE_PARAMS: Final[int] = (
    (M3_INPUT_DIM * M3_HIDDEN_DIM + M3_HIDDEN_DIM)
    + M3_N_LAYERS * (
        ((2 * M3_HIDDEN_DIM + M3_EDGE_DIM) * M3_HIDDEN_DIM + M3_HIDDEN_DIM)
        + (M3_HIDDEN_DIM * M3_HIDDEN_DIM + M3_HIDDEN_DIM)
        + (M3_HIDDEN_DIM * M3_VECTOR_CHANNELS)
        + (M3_HIDDEN_DIM * M3_HIDDEN_DIM + M3_HIDDEN_DIM))
    + (M3_HIDDEN_DIM * M3_VECTOR_CHANNELS + M3_VECTOR_CHANNELS))


class M3Error(V543Error):
    """M3 refusal (context shape drift, out-of-band width)."""


def _check_move_context(ctx: object) -> MoveContextV1:
    """Refuse anything that is not the read-only MOVE view (plan §5.2.1)."""
    if not isinstance(ctx, MoveContextV1):
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_TYPE",
            "M3 takes a MoveContextV1 (plan §6.1 :490); got "
            f"{type(ctx).__name__}")
    key_s = ctx.key_s
    key_v = ctx.key_v
    if int(key_s.ndim) != 2 or int(key_s.shape[1]) != KEY_S_DIM_V543:
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_SHAPE",
            f"M3 needs key_s [K,{KEY_S_DIM_V543}], got {list(key_s.shape)}")
    expected = (int(key_s.shape[0]), KEY_V_CHANNELS_V543, 3)
    if tuple(key_v.shape) != expected:
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_SHAPE",
            f"M3 needs key_v {list(expected)}, got {list(key_v.shape)}")
    edge_index = ctx.key_edge_index
    if int(edge_index.ndim) != 2 or int(edge_index.shape[0]) != 2:
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_SHAPE",
            "M3 needs key_edge_index [2,Ek], got "
            f"{list(edge_index.shape)}")
    n_edges = int(edge_index.shape[1])
    if tuple(ctx.key_edge_attr.shape) != (n_edges, M3_EDGE_DIM):
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_SHAPE",
            f"M3 needs key_edge_attr [{n_edges},{M3_EDGE_DIM}], got "
            f"{list(ctx.key_edge_attr.shape)}")
    return ctx


class VectorMessageLayer(torch.nn.Module):
    """One scalar–vector message-passing layer on the key graph (§6.2 :528).

    For every directed key edge ``(src, dst)``::

        m_ij  = SiLU(W2 · SiLU(W1 [h_i, h_j, e_ij] + b1) + b2)   # [E,hidden]
        g_ij  = W_v · m_ij                                      # [E,8]
        ΔV_i  = Σ_{j→i} g_ij ⊙ (V_j − V_i) / deg_i
        V_i'  = V_i + ΔV_i
        h_i'  = h_i + SiLU(W_h · (Σ_{j→i} m_ij / deg_i))

    The vector branch is a scalar-gated difference of POLAR vectors and the
    aggregation is a sum, so the layer is O(3)/polar-covariant and independent
    of the ``key_edge_index`` column order (plan §6.2 :534/:536 discipline: no
    LayerNorm/ReLU on the vector stream, no cross product).
    """

    def __init__(self, hidden: int = M3_HIDDEN_DIM,
                 vector_channels: int = M3_VECTOR_CHANNELS,
                 edge_dim: int = M3_EDGE_DIM) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.vector_channels = int(vector_channels)
        self.edge_dim = int(edge_dim)
        self.edge_in = torch.nn.Linear(2 * self.hidden + self.edge_dim,
                                       self.hidden)
        self.edge_out = torch.nn.Linear(self.hidden, self.hidden)
        self.vector_gate = torch.nn.Linear(self.hidden, self.vector_channels,
                                           bias=False)
        self.node_out = torch.nn.Linear(self.hidden, self.hidden)

    def forward(self, hidden: torch.Tensor, vectors: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``h [K,H]``, ``V [K,8,3]``, ``edge_index [2,E]``, ``e [E,8]``."""
        n_keys = int(hidden.shape[0])
        src = edge_index[0]
        dst = edge_index[1]
        message = torch.cat(
            [hidden[src], hidden[dst], edge_attr], dim=-1)
        message = F.silu(self.edge_in(message))
        message = F.silu(self.edge_out(message))              # [E,H]
        gate = self.vector_gate(message)                      # [E,8]
        # Polar vector message: scalar-gated difference of neighbour vectors.
        delta = gate.unsqueeze(-1) * (vectors[dst] - vectors[src])  # [E,8,3]
        degree = torch.zeros(n_keys, dtype=hidden.dtype,
                             device=hidden.device).index_add_(
            0, dst, torch.ones_like(message[:, 0]))
        # A degree of 0 (isolated key, including Ek=0) is the documented
        # degenerate path (§5.2 :310): every summand is absent, so ΔV = 0 and
        # the pooled message is the zero vector.
        denom = degree.clamp_min(1.0)
        pooled_vec = torch.zeros(
            n_keys, self.vector_channels, 3, dtype=vectors.dtype,
            device=vectors.device).index_add_(0, dst, delta)
        vectors_out = vectors + pooled_vec / denom.view(-1, 1, 1)
        pooled_scalar = torch.zeros(
            n_keys, self.hidden, dtype=hidden.dtype,
            device=hidden.device).index_add_(0, dst, message)
        hidden_out = hidden + F.silu(self.node_out(
            pooled_scalar / denom.view(-1, 1)))
        return hidden_out, vectors_out


class VectorGNNMoveDecoder(torch.nn.Module):
    """``M3``: 2 scalar–vector layers on the key graph, hidden 56.

    34,448 trainable parameters (measured at construction; plan §6.4 band
    34,200–37,800, estimate "约 34.5K" at §6.2 :528).
    """

    decoder_id: Final[str] = M3_DECODER_ID
    track: Final[str] = TRACK_MOVE
    model_version: Final[str] = M3_MODEL_VERSION

    def __init__(self, *, seed: int = M3_INIT_SEED_V543) -> None:
        super().__init__()
        self.init_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.embed = torch.nn.Linear(M3_INPUT_DIM, M3_HIDDEN_DIM)
            self.layers = torch.nn.ModuleList([
                VectorMessageLayer() for _ in range(M3_N_LAYERS)])
            self.coeff = torch.nn.Linear(
                M3_HIDDEN_DIM, M3_VECTOR_CHANNELS)
            # plan §4.5 :243 — the output head is zero-initialised: δu ≡ 0.
            with torch.no_grad():
                self.coeff.weight.zero_()
                self.coeff.bias.zero_()
        measured = count_trainable_params(self)
        if measured != M3_N_TRAINABLE_PARAMS:
            raise M3Error(
                "V543_M3_PARAMS",
                f"M3 (2 layers, hidden {M3_HIDDEN_DIM}, edge "
                f"{M3_EDGE_DIM}) must give exactly {M3_N_TRAINABLE_PARAMS} "
                f"trainable parameters (plan §6.2 :528, §6.4 :569), "
                f"measured {measured}")
        assert_budget(measured, decoder_id=self.decoder_id,
                      track=self.track)

    # -- protocol -----------------------------------------------------------
    def forward(self, ctx: MoveContextV1) -> torch.Tensor:
        """Return ``δu [K,3]`` from the aggregated vector channels."""
        ctx = _check_move_context(ctx)
        hidden = F.silu(self.embed(torch.cat([
            ctx.graph_s[ctx.key_graph],
            ctx.key_s,
            ctx.key_v.norm(dim=-1),
        ], dim=-1)))                                          # [K,56]
        vectors = ctx.key_v                                   # [K,8,3]
        for layer in self.layers:
            hidden, vectors = layer(hidden, vectors,
                                    ctx.key_edge_index, ctx.key_edge_attr)
        coeff = self.coeff(hidden)                            # [K,8]
        return (coeff.unsqueeze(-1) * vectors).sum(dim=1)     # [K,3]

    @property
    def n_trainable_params(self) -> int:
        return count_trainable_params(self)

    # -- audit --------------------------------------------------------------
    def describe(self) -> dict[str, object]:
        return {
            "decoder_id": self.decoder_id,
            "track": self.track,
            "model_version": self.model_version,
            "n_layers": M3_N_LAYERS,
            "hidden_dim": M3_HIDDEN_DIM,
            "vector_channels": M3_VECTOR_CHANNELS,
            "edge_dim": M3_EDGE_DIM,
            "n_trainable_params": self.n_trainable_params,
            "init_seed": self.init_seed,
            "readout": "delta_u = sum_j c_j V_j (plan §6.2 :528)",
            "zero_init": "coeff.weight/bias = 0 (plan §4.5 :243)",
            "topology": "key_edge_index only; index_add local aggregation "
                        "(plan §6.2 :536), Ek=0 degenerate path (§5.2 :310)",
        }


def m3_spec(**kwargs: object) -> DecoderSpec:
    """The §6.1 registry record of M3 (13 fields; plan §6.1 :503–517)."""
    def factory() -> VectorGNNMoveDecoder:
        return VectorGNNMoveDecoder(**kwargs)  # type: ignore[arg-type]

    return DecoderSpec(
        decoder_id=M3_DECODER_ID,
        track=TRACK_MOVE,
        model_version=M3_MODEL_VERSION,
        factory=factory,
        equivariance="polar",
        stochastic=False,
        nfe=1,
        zero_init=True,
        status="proposed",
        acceptance="wp5_implemented_pending_S1_smoke",
    )


#: The process-wide M3 record (registered by the integrating agent / WP7).
M3_SPEC: Final[DecoderSpec] = m3_spec()


def register_m3(registry: object) -> DecoderSpec:
    """Register :data:`M3_SPEC` into ``registry`` (plan §11.2 ``register``)."""
    return registry.register(M3_SPEC)  # type: ignore[attr-defined]


__all__ = [
    "M3_DECODER_ID",
    "M3_EDGE_DIM",
    "M3_HIDDEN_DIM",
    "M3_INIT_SEED_V543",
    "M3_INPUT_DIM",
    "M3_MODEL_VERSION",
    "M3_N_LAYERS",
    "M3_N_TRAINABLE_PARAMS",
    "M3_SPEC",
    "M3_VECTOR_CHANNELS",
    "M3Error",
    "VectorGNNMoveDecoder",
    "VectorMessageLayer",
    "m3_spec",
    "register_m3",
]
