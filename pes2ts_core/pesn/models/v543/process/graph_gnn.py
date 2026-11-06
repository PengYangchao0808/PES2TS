"""V5.4.3 WP6 — ``P3`` PROCESS candidate: Latent Graph-GNN on the frozen
token graph (plan §6.3 :545, §6.4 :557–571, §4.5 :246, §11.1 :1108
``process/graph_gnn.py``).

Normative sources, quoted as implemented
---------------------------------------
* **§6.3 :545** — "``P3`` | Latent Graph-GNN | 同一 token 图上 2 层消息传递；
  hidden 56；边特征 8（约 34.5K）| 36K ±5% | 正式".  Implemented as: the SHARED
  §5.5 token set (same index as P2/P4, §6.3 :552), per-type token projections
  into ``hidden=56``, TWO message-passing layers on the within-graph token
  adjacency, and an 8-dim edge feature.
* **§6.3 :554 "局部图 bias 用 ``key_edge_index``/``token`` 邻接"** — see
  :func:`~pes2ts_core.pesn.models.v543.process.tokens.local_token_edges`: the
  PROCESS view carries **no ``key_edge_index``** (plan §5.2.1 :349), so the
  local structure is the legal half of that sentence — the ``token`` 邻接 of
  the §5.5 layout (pair token ↔ its two endpoint key tokens, graph token ↔ its
  tokens).  The 8 edge scalars are the frozen invariant slice
  ``pair_s[20:28]`` (§5.4 :386–387), not a new geometric quantity.
* **§6.2 :536 M3-style locality rule, applied by analogy** — "只在 key 图上做
  1–2 层局部交互；不得重新运行原始几何 Encoder，不得对全部原子做稠密
  attention": P3 runs exactly 2 local layers over ``T`` tokens, never the
  encoder, never a dense token² attention.
* **§5.2 形状约束 :310** — ``Ek=0`` may be legal on the MOVE side; on the
  PROCESS side every graph has ≥1 key and ≥1 graph token, so the local
  adjacency is never empty, but the code still handles ``E=0`` (an isolated
  message pass is a no-op, not a crash).
* **§4.5 :246** — last ``Linear(·,2)`` zeroed; initial ``δq = δe = 0``.
* **§6.4 :563–571** — measured count asserted in ``__init__``.

Width decision and its basis (plan §6.3 note :550)
--------------------------------------------------
:550 says the table widths are "设计起点，不是实现前的精确断言；构造时必须按
``sum(p.numel())`` 调整宽度，使精确值落入 34,200–37,800".  With a SINGLE shared
``48 → 56`` encoder for key and pair tokens the structure lands at 33,474 —
BELOW the band (−726).  Rather than inflate a width away from the plan's
``hidden 56``, the token encoder is per-type (graph / key / pair), which is also
the semantically honest choice: ``graph_s`` is 64-wide and ``key_s``/``pair_s``
are different 48-dim layouts (§5.2).  That lands at 36,210 with ``hidden = 56``
EXACTLY as §6.3 :545 specifies.  Both numbers are reported in the WP6 report.

Measured parameter count (``sum(p.numel())``, plan §6.4 :569)
------------------------------------------------------------
graph encoder ``64·56+56`` = 3,640; key ``48·56+56`` = 2,744; pair = 2,744;
type embedding ``3·56`` = 168; edge-type embedding ``2·8`` = 16; per layer
``(120·56+56) + (112·56+56) + 2·(2·56)`` = 13,328 → ×2 = 26,656; head
``120·2+2`` = 242 → **36,210** (plan estimate "约 34.5K", +5.0 %).

Discipline: no data reading / loss / checkpoint selection (plan §6.1 :519), no
``torch.compile`` (plan §11.3), every constant ``Final``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Final

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.models.v543.contracts import (
    GRAPH_S_DIM_V543,
    ProcessContextV1,
    ProcessResidual,
)
from pes2ts_core.pesn.models.v543.process.tokens import (
    N_EDGE_TYPES_V543,
    N_TOKEN_TYPES_V543,
    PAIR_EDGE_FEATURE_DIM_V543,
    PROCESS_INIT_SEED_V543,
    PROCESS_TOKEN_DIM_V543,
    assert_candidate_budget,
    assert_process_context,
    deterministic_init,
    local_token_edges,
    measured_params,
    scatter_messages,
    segment_mean,
    token_tensors,
    zero_init_last_layer,
)
from pes2ts_core.pesn.models.v543.registry import TRACK_PROCESS, DecoderSpec

#: ``P3`` id (plan §6.3 :545).
DECODER_ID_P3: Final[str] = "P3"
#: Registered model-layer version of this candidate.
MODEL_VERSION_P3: Final[str] = "v543_p3_latent_graph_gnn_v1"
#: Plan §6.3 :545 "hidden 56".
P3_HIDDEN: Final[int] = 56
#: Plan §6.3 :545 "2 层消息传递".
P3_LAYERS: Final[int] = 2
#: Residual scalars (``δq_raw``, ``δe``).
P3_OUTPUTS: Final[int] = 2
#: Measured parameter count of the frozen structure (§6.4 :569).
P3_EXPECTED_PARAMS: Final[int] = (
    # per-type token projections
    GRAPH_S_DIM_V543 * P3_HIDDEN + P3_HIDDEN
    + 2 * (PROCESS_TOKEN_DIM_V543 * P3_HIDDEN + P3_HIDDEN)
    # token-type and edge-type embeddings
    + N_TOKEN_TYPES_V543 * P3_HIDDEN
    + N_EDGE_TYPES_V543 * PAIR_EDGE_FEATURE_DIM_V543
    # message-passing layers
    + P3_LAYERS * (
        (2 * P3_HIDDEN + PAIR_EDGE_FEATURE_DIM_V543) * P3_HIDDEN + P3_HIDDEN
        + 2 * P3_HIDDEN * P3_HIDDEN + P3_HIDDEN
        + 2 * 2 * P3_HIDDEN)
    # readout head  [mean-pool(hidden) ‖ graph_s] -> 2
    + (P3_HIDDEN + GRAPH_S_DIM_V543) * P3_OUTPUTS + P3_OUTPUTS)


class _MessageLayer(torch.nn.Module):
    """One local message-passing block: ``h ← h + Σ msg`` then ``h ← h + upd``.

    Two pre-norm residuals, mirrored on the P4 block (aggregation ≈ attention,
    update ≈ FFN) so the P3-vs-P4 comparison changes the OPERATOR and not the
    depth or the normalisation placement.
    """

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.message = torch.nn.Linear(2 * hidden + edge_dim, hidden)
        self.update = torch.nn.Linear(2 * hidden, hidden)

    def forward(self, h: torch.Tensor, edges) -> torch.Tensor:  # noqa: ANN001
        pre = self.norm1(h)
        sent = self.message(torch.cat(
            (pre[edges.edge_index[0]], pre[edges.edge_index[1]],
             edges.edge_feature), dim=-1))
        aggregated = scatter_messages(sent, edges.edge_index,
                                      int(h.shape[0]))
        h = h + aggregated
        h = h + self.update(torch.cat((self.norm2(h), aggregated), dim=-1))
        return h


class ProcessLatentGraphGNN(torch.nn.Module):
    """``P3``: 2 local message-passing layers over the §5.5 tokens (§6.3 :545).

    The readout is a per-graph mean of the final token states concatenated with
    the raw ``graph_s`` row, so a graph's prediction depends on the SET of its
    tokens (permutation invariant, §5.2 置换 column) and on the global state.

    Args:
        hidden: token/message width (default 56, the plan's value).
        layers: message-passing depth (default 2, the plan's value).
        seed: initialisation seed (fork_rng discipline, plan §7.2 :667).
    """

    decoder_id: Final[str] = DECODER_ID_P3
    track: Final[str] = TRACK_PROCESS
    model_version: Final[str] = MODEL_VERSION_P3
    equivariance: Final[str] = "invariant"
    stochastic: Final[bool] = False
    nfe: Final[int] = 1
    zero_init: Final[bool] = True

    def __init__(self, *, hidden: int = P3_HIDDEN, layers: int = P3_LAYERS,
                 seed: int = PROCESS_INIT_SEED_V543) -> None:
        super().__init__()
        if int(hidden) < 1 or int(layers) < 1:
            raise ValueError("P3 hidden/layers must be >= 1")
        self.init_seed: Final[int] = int(seed)
        self.hidden: Final[int] = int(hidden)
        self.layers: Final[int] = int(layers)
        edge_dim = PAIR_EDGE_FEATURE_DIM_V543
        with deterministic_init(self.init_seed):
            self.graph_proj = torch.nn.Linear(GRAPH_S_DIM_V543, self.hidden)
            self.key_proj = torch.nn.Linear(PROCESS_TOKEN_DIM_V543,
                                            self.hidden)
            self.pair_proj = torch.nn.Linear(PROCESS_TOKEN_DIM_V543,
                                             self.hidden)
            self.type_bias = torch.nn.Parameter(
                torch.zeros(N_TOKEN_TYPES_V543, self.hidden))
            self.edge_type_embedding = torch.nn.Embedding(
                N_EDGE_TYPES_V543, edge_dim)
            self.blocks = torch.nn.ModuleList(
                [_MessageLayer(self.hidden, edge_dim)
                 for _ in range(self.layers)])
            self.head = torch.nn.Linear(
                self.hidden + GRAPH_S_DIM_V543, P3_OUTPUTS)
        zero_init_last_layer(self.head)
        assert_p3_budget(measured_params(self))

    # -- protocol -----------------------------------------------------------
    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:
        """``token 图`` 2 层消息传递 → ``ProcessResidual`` (§6.1 :496)."""
        context = assert_process_context(ctx, decoder_id=self.decoder_id)
        tokens = token_tensors(context, decoder_id=self.decoder_id)
        hidden = self.type_bias[tokens.token_type]
        if tokens.n_keys:
            hidden = hidden.index_add(
                0, tokens.key_token_rows, self.key_proj(context.key_s))
        if tokens.n_pairs:
            hidden = hidden.index_add(
                0, tokens.pair_token_rows, self.pair_proj(context.pair_s))
        hidden = hidden.index_add(
            0, tokens.graph_token_rows, self.graph_proj(context.graph_s))
        edges = local_token_edges(context, tokens, decoder_id=self.decoder_id)
        edges = replace(
            edges,
            edge_feature=(edges.edge_feature
                          + self.edge_type_embedding(edges.edge_type)))
        for block in self.blocks:
            hidden = block(hidden, edges)
        pooled = segment_mean(hidden, tokens.token_graph, tokens.n_graphs)
        features = torch.cat((pooled, context.graph_s), dim=-1)
        out = self.head(features)
        return ProcessResidual(delta_q_raw=out[:, 0], delta_e=out[:, 1])

    # -- bookkeeping --------------------------------------------------------
    @property
    def n_trainable_params(self) -> int:
        """Measured ``sum(p.numel())`` (plan §6.1 :509 / §6.4 :569)."""
        return measured_params(self)

    def describe(self) -> dict[str, Any]:
        return {
            "decoder_id": self.decoder_id,
            "track": self.track,
            "model_version": self.model_version,
            "structure": ("shared §5.5 tokens -> per-type proj(56) -> 2 local "
                          "message-passing layers (edge feature 8 = "
                          "pair_s[20:28]) -> mean pool ⊕ graph_s -> 2 "
                          "(plan §6.3 :545)"),
            "hidden": int(self.hidden),
            "layers": int(self.layers),
            "edge_feature_dim": PAIR_EDGE_FEATURE_DIM_V543,
            "init_seed": int(self.init_seed),
            "n_trainable_params": self.n_trainable_params,
            "zero_init": True,
        }


def assert_p3_budget(n_trainable_params: int) -> int:
    """Construction-time §6.4 band assertion of ``P3``."""
    return assert_candidate_budget(n_trainable_params, decoder_id=DECODER_ID_P3)


def build_p3(**kwargs: Any) -> ProcessLatentGraphGNN:
    """Registry factory."""
    return ProcessLatentGraphGNN(**kwargs)


#: The §6.1 registry spec of ``P3``.
P3_SPEC: Final[DecoderSpec] = DecoderSpec(
    decoder_id=DECODER_ID_P3,
    track=TRACK_PROCESS,
    model_version=MODEL_VERSION_P3,
    factory=build_p3,
    equivariance="invariant",
    stochastic=False,
    nfe=1,
    zero_init=True,
    status="proposed",
    acceptance=(
        "wp6: constructed, sum(p.numel()) asserted in band, zero-residual "
        "init, 2 local layers on the shared §5.5 token graph, pair/token "
        "permutation invariant; S1 smoke pending"),
)


__all__ = [
    "DECODER_ID_P3",
    "MODEL_VERSION_P3",
    "P3_EXPECTED_PARAMS",
    "P3_HIDDEN",
    "P3_LAYERS",
    "P3_OUTPUTS",
    "P3_SPEC",
    "ProcessLatentGraphGNN",
    "assert_p3_budget",
    "build_p3",
]
