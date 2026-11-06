"""V5.4.3 WP6 — ``P4`` Graph Transformer **and** its parameter-matched
non-attention control ``P4-C`` (plan §6.3 :546–547, §6.4 :558–571, §4.5 :246,
§11.1 :1109 ``process/graph_transformer.py``).

Normative sources, quoted as implemented
---------------------------------------
* **§6.3 :546 ``P4``** — "2 层、d=48、4 heads（head_dim 12）、FFN 80；局部图
  bias；token 级 attention（约 37K）| 36K ±5% | 正式".  Implemented as: the
  shared §5.5 token set, per-type input assembly into ``d = 48`` with a
  ``token_type`` embedding bias, TWO pre-norm transformer blocks with 4 heads of
  ``head_dim = 12`` and an FFN of width 80, and a local-adjacency attention bias.
* **§6.3 :554** — "attention 只在 token 级（T≈G+K+P，通常 <40/图），不做原子级
  稠密 N²；attention logits 不得含绝对坐标；局部图 bias 用
  ``key_edge_index``/``token`` 邻接，不用可学习全局帧."  Three obligations,
  all coded: (a) the score matrix is ``T×T`` within a graph (never atom-level);
  (b) every term of a logit comes from ``graph_s``/``key_s``/``pair_s``/token
  structure — no coordinate enters, and the bias is a LEARNED SCALAR per head on
  a mask, not a learned global frame; (c) the adjacency is the ``token`` 邻接 of
  the §5.5 layout (see ``tokens.local_token_edges``) because the PROCESS view
  carries no ``key_edge_index`` (plan §5.2.1 :349).
* **§6.3 :547 ``P4-C``** — "与 P4 同宽、同 token、同参数 ±2%，MLP/池化替代
  attention | 与 P4 匹配 | 诊断（仅 P4 进 top2 时）".  :class:`ProcessP4Control`
  reuses P4's input assembly, width, block depth, normalisation placement, FFN
  and readout head and replaces ONLY the attention operator with a per-graph
  pooled MLP mix; its measured count is within ±2 % of P4's measured count
  (37,762 vs 37,546 → +0.58 %).
* **§6.4 :563–571** — P4/P4-C use the frozen ±5 % band 34,200–37,800 for the
  candidate itself; **P4-C additionally** uses its own INDEPENDENT frozen band
  «P4 measured ±2 %» (:547 "同参数 ±2%"), which is a different band from the
  36 K ±5 % rule and is never mixed with it (:data:`P4C_BAND_PCT`).
* **§6.1 :514 status vocabulary** — P4-C's status is ``diagnostic`` (:547
  "诊断（仅 P4 进 top2 时）"), not ``formal``.
* **§4.5 :246** — last ``Linear(·,2)`` zeroed on both candidates.

Latent plan ↔ WP4 conflict (reported, not silently worked around)
-----------------------------------------------------------------
Plan §6.3 :547 declares ``P4-C`` as a candidate ID, but WP4's registry declares
the campaign ids as ``P0…P5`` only (``registry.DECODER_IDS_V543``), and both
``DecoderSpec.__post_init__`` and ``assert_decoder_protocol`` refuse any other
id (``V543_REGISTRY_UNKNOWN``).  ``P4-C`` therefore CANNOT be registered through
:class:`~pes2ts_core.pesn.models.v543.registry.DecoderSpec` without editing a
shared WP4 file — which WP6 is forbidden to do.  The honest handling is
:class:`P4ControlSpec`: the same 13 record fields (§6.1 :503–517, in order),
plus the ±2 % band provenance, with no registry mutation and no fabricated id.
Tests assert BOTH the refusal (``P4-C`` is not a declared campaign id) and the
completeness of the 13-field record.

Measured parameter counts (``sum(p.numel())``, plan §6.4 :569)
-------------------------------------------------------------
``P4``: input ``graph_s 64→48`` = 3,120 + type bias ``3·48`` = 144; per block
``qkv 3·48·48`` = 6,912 + ``out 48·48`` = 2,304 + ``FFN 48·80 + 80·48`` = 7,680
+ ``2·LN(48)`` = 192 + ``4`` local-bias scalars = 17,092 → ×2 = 34,184; head
``48·2+2`` = 98 → **37,546** (plan estimate "约 37K", +1.5 %).

``P4-C``: same input 3,264 + per block ``96·64+64`` = 6,208 + ``64·48+48`` =
3,120 + FFN 7,680 + LNs 192 = 17,200 → ×2 = 34,400; head 98 → **37,762**
(+0.58 % vs P4, inside the ±2 % matched band).

Discipline: no data reading / loss / checkpoint selection (plan §6.1 :519), no
``torch.compile`` (plan §11.3), every constant ``Final``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.models.v543.contracts import (
    DECODER_CONTEXT_SCHEMA_V543,
    GRAPH_S_DIM_V543,
    ProcessContextV1,
    ProcessResidual,
    V543ParamBudgetError,
)
from pes2ts_core.pesn.models.v543.process.tokens import (
    N_TOKEN_TYPES_V543,
    PROCESS_INIT_SEED_V543,
    V543ProcessDecoderError,
    assert_candidate_budget,
    assert_process_context,
    deterministic_init,
    local_token_edges,
    measured_params,
    scatter_token_features,
    segment_mean,
    token_tensors,
    zero_init_last_layer,
)
from pes2ts_core.pesn.models.v543.registry import (
    REGISTRY_RECORD_FIELDS,
    STATUS_VALUES_V543,
    TRACK_PROCESS,
    DecoderSpec,
)

# ---------------------------------------------------------------------------
# P4 constants (plan §6.3 :546)
# ---------------------------------------------------------------------------
#: ``P4`` id.
DECODER_ID_P4: Final[str] = "P4"
#: Registered model-layer version of ``P4``.
MODEL_VERSION_P4: Final[str] = "v543_p4_graph_transformer_v1"
#: Plan §6.3 :546 "d=48".
P4_DIM: Final[int] = 48
#: Plan §6.3 :546 "4 heads（head_dim 12）".
P4_HEADS: Final[int] = 4
#: ``head_dim`` implied by d/heads (§6.3 :546).
P4_HEAD_DIM: Final[int] = P4_DIM // P4_HEADS
#: Plan §6.3 :546 "FFN 80".
P4_FFN: Final[int] = 80
#: Plan §6.3 :546 "2 层".
P4_LAYERS: Final[int] = 2
#: Residual scalars (``δq_raw``, ``δe``).
P4_OUTPUTS: Final[int] = 2

# ---------------------------------------------------------------------------
# P4-C constants (plan §6.3 :547)
# ---------------------------------------------------------------------------
#: ``P4-C`` id — the plan's own id, NOT a registry-declared campaign id (:547).
DECODER_ID_P4C: Final[str] = "P4-C"
#: Registered model-layer version of the control.
MODEL_VERSION_P4C: Final[str] = "v543_p4c_param_matched_control_v1"
#: :547 "与 P4 同参数 ±2%" — the INDEPENDENT frozen band half-width, in percent.
P4C_BAND_PCT: Final[float] = 2.0
#: The candidate the control is matched to (:547 "与 P4 匹配").
P4C_REFERENCE_DECODER_ID: Final[str] = DECODER_ID_P4
#: Plan §6.3 :547 "MLP/池化替代 attention": the pooled-mix inner width.
P4C_MIX_HIDDEN: Final[int] = 64
#: P4-C's status in the §6.1 :514 vocabulary (:547 "诊断").
P4C_STATUS: Final[str] = "diagnostic"

assert P4_HEAD_DIM * P4_HEADS == P4_DIM, "head_dim must tile d"
#: Sanity: P4-C is 'diagnostic' in the §6.1 vocabulary (§6.3 :547).
assert P4C_STATUS in STATUS_VALUES_V543

#: Measured parameter count of the frozen P4 structure (§6.4 :569).
P4_EXPECTED_PARAMS: Final[int] = (
    # input assembly: graph token projection + token_type bias
    GRAPH_S_DIM_V543 * P4_DIM + P4_DIM
    + N_TOKEN_TYPES_V543 * P4_DIM
    # transformer blocks
    + P4_LAYERS * (
        3 * P4_DIM * P4_DIM          # qkv (bias-free)
        + P4_DIM * P4_DIM            # out projection (bias-free)
        + 2 * 2 * P4_DIM             # two LayerNorms
        + P4_DIM * P4_FFN            # FFN up (bias-free)
        + P4_FFN * P4_DIM            # FFN down (bias-free)
        + P4_HEADS)                  # one local-graph bias scalar per head
    # readout head: mean-pooled token state -> 2
    + P4_DIM * P4_OUTPUTS + P4_OUTPUTS)

#: Measured parameter count of the frozen P4-C structure (§6.4 :569).
P4C_EXPECTED_PARAMS: Final[int] = (
    GRAPH_S_DIM_V543 * P4_DIM + P4_DIM
    + N_TOKEN_TYPES_V543 * P4_DIM
    + P4_LAYERS * (
        (2 * P4_DIM) * P4C_MIX_HIDDEN + P4C_MIX_HIDDEN
        + P4C_MIX_HIDDEN * P4_DIM + P4_DIM
        + 2 * 2 * P4_DIM
        + P4_DIM * P4_FFN
        + P4_FFN * P4_DIM)
    + P4_DIM * P4_OUTPUTS + P4_OUTPUTS)


# ---------------------------------------------------------------------------
# the shared input assembly + the two block operators
# ---------------------------------------------------------------------------
class _TokenInput(torch.nn.Module):
    """Per-type token assembly into ``d = 48`` with a ``token_type`` bias.

    ``key_s``/``pair_s`` are already 48-wide (§5.2), so only the 64-wide
    ``graph_s`` needs a projection; every token receives a learned type bias, so
    the block can tell a graph token from a key token from a pair token without
    any index-specific parameter (§5.5 "token_type 区分三类").
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.graph_proj = torch.nn.Linear(GRAPH_S_DIM_V543, dim)
        self.type_bias = torch.nn.Parameter(
            torch.zeros(N_TOKEN_TYPES_V543, dim))

    def forward(self, ctx: ProcessContextV1, tokens) -> torch.Tensor:  # noqa: ANN001
        width = int(self.type_bias.shape[-1])
        hidden = self.type_bias[tokens.token_type]
        payload = scatter_token_features(tokens, ctx.key_s, ctx.pair_s)
        if int(payload.shape[-1]) != width:
            raise V543ProcessDecoderError(
                "V543_PROCESS_TOKEN_WIDTH",
                f"token width {width} must equal the frozen key_s/pair_s width "
                f"{int(payload.shape[-1])} (plan §5.2: 48); P4/P4-C reuse the "
                "tokens unchanged and never re-project them")
        hidden = hidden + payload
        hidden = hidden.index_add(
            0, tokens.graph_token_rows, self.graph_proj(ctx.graph_s))
        return hidden


class _TransformerBlock(torch.nn.Module):
    """Pre-norm block: ``h += attn(LN(h))``, ``h += FFN(LN(h))`` (§6.3 :546).

    The attention is limited to the tokens of ONE graph (block-diagonal mask
    from ``token_graph``) and carries a per-head LOCAL bias on the §5.5 token
    adjacency (:554 "局部图 bias ... 不用可学习全局帧").
    """

    def __init__(self, dim: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(dim) // int(heads)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)
        self.ffn_up = torch.nn.Linear(dim, ffn, bias=False)
        self.ffn_down = torch.nn.Linear(ffn, dim, bias=False)
        #: One learned scalar per head: how strongly local adjacency is
        #: preferred.  A mask-modulated scalar cannot encode a frame (:554).
        self.local_bias = torch.nn.Parameter(torch.zeros(self.heads))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        pre = self.norm1(hidden)
        n_tokens = int(pre.shape[0])
        qkv = self.qkv(pre).view(n_tokens, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        query = query.transpose(0, 1)                 # [heads, T, head_dim]
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        scores = query @ key.transpose(-1, -2)
        scores = scores / math.sqrt(float(self.head_dim))
        bias = (self.local_bias.view(1, self.heads, 1, 1)
                * adjacency.to(scores.dtype).view(1, 1, n_tokens, n_tokens))
        scores = scores + bias
        scores = scores.masked_fill(
            ~mask.view(1, 1, n_tokens, n_tokens), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value                     # [heads, T, head_dim]
        attended = attended.transpose(0, 1).reshape(n_tokens, -1)
        hidden = hidden + self.out_proj(attended)
        hidden = hidden + self.ffn_down(F.silu(self.ffn_up(self.norm2(hidden))))
        return hidden


class _PooledMixBlock(torch.nn.Module):
    """P4-C's operator: per-graph MEAN pooling + MLP instead of attention.

    Same depth, same width, same normalisation placement and same FFN as
    :class:`_TransformerBlock`; the only change is that a token sees the MEAN of
    its graph's tokens instead of a learned softmax mixture, so P4-vs-P4-C
    isolates "attention" from "capacity" (plan §6.3 :547, hypothesis H-P2 §3.2
    :167 "P4-C 追平 P4（说明是容量而非注意力）").
    """

    def __init__(self, dim: int, ffn: int, mix_hidden: int) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mix1 = torch.nn.Linear(2 * dim, mix_hidden)
        self.mix2 = torch.nn.Linear(mix_hidden, dim)
        self.ffn_up = torch.nn.Linear(dim, ffn, bias=False)
        self.ffn_down = torch.nn.Linear(ffn, dim, bias=False)

    def forward(self, hidden: torch.Tensor, tokens) -> torch.Tensor:  # noqa: ANN001
        pre = self.norm1(hidden)
        pooled = segment_mean(pre, tokens.token_graph, tokens.n_graphs)
        context = pooled[tokens.token_graph]
        mixed = self.mix2(F.silu(self.mix1(torch.cat((pre, context), dim=-1))))
        hidden = hidden + mixed
        hidden = hidden + self.ffn_down(F.silu(self.ffn_up(self.norm2(hidden))))
        return hidden


def _attention_masks(ctx: ProcessContextV1, tokens,
                     edges) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN001
    """Within-graph attention mask + local-adjacency mask (§6.3 :554).

    Both masks are derived from the context on EVERY forward (never cached), so
    a permuted or a resized context cannot inherit a stale mask.  The
    same-graph mask is what makes the readout a per-graph function and therefore
    keeps the output invariant to the intra-graph token order.
    """
    graph = tokens.token_graph
    same_graph = graph.unsqueeze(0) == graph.unsqueeze(1)
    adjacency = torch.zeros((tokens.n_tokens, tokens.n_tokens), dtype=torch.bool)
    if int(edges.edge_index.shape[1]):
        adjacency[edges.edge_index[0], edges.edge_index[1]] = True
    return same_graph, adjacency


class ProcessGraphTransformer(torch.nn.Module):
    """``P4``: 2-layer token-level Graph Transformer (plan §6.3 :546).

    Args:
        dim: token width (default 48, the plan's value).
        heads: attention heads (default 4; ``dim / heads`` must be integral).
        ffn: FFN width (default 80).
        layers: block depth (default 2).
        seed: initialisation seed (fork_rng discipline, plan §7.2 :667).
    """

    decoder_id: Final[str] = DECODER_ID_P4
    track: Final[str] = TRACK_PROCESS
    model_version: Final[str] = MODEL_VERSION_P4
    equivariance: Final[str] = "invariant"
    stochastic: Final[bool] = False
    nfe: Final[int] = 1
    zero_init: Final[bool] = True

    def __init__(self, *, dim: int = P4_DIM, heads: int = P4_HEADS,
                 ffn: int = P4_FFN, layers: int = P4_LAYERS,
                 seed: int = PROCESS_INIT_SEED_V543) -> None:
        super().__init__()
        if int(dim) < 1 or int(heads) < 1 or int(ffn) < 1 or int(layers) < 1:
            raise ValueError("P4 widths/depth must be >= 1")
        if int(dim) % int(heads):
            raise ValueError(
                f"P4 dim={dim} must be divisible by heads={heads} "
                "(plan §6.3 :546 head_dim 12)")
        self.init_seed: Final[int] = int(seed)
        self.dim: Final[int] = int(dim)
        self.heads: Final[int] = int(heads)
        self.ffn: Final[int] = int(ffn)
        self.layers: Final[int] = int(layers)
        with deterministic_init(self.init_seed):
            self.input = _TokenInput(self.dim)
            self.blocks = torch.nn.ModuleList([
                _TransformerBlock(self.dim, self.heads, self.ffn)
                for _ in range(self.layers)])
            self.head = torch.nn.Linear(self.dim, P4_OUTPUTS)
        zero_init_last_layer(self.head)
        assert_p4_budget(measured_params(self))

    # -- protocol -----------------------------------------------------------
    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:
        """Token-level transformer → ``ProcessResidual`` (plan §6.1 :496)."""
        context = assert_process_context(ctx, decoder_id=self.decoder_id)
        tokens = token_tensors(context, decoder_id=self.decoder_id)
        edges = local_token_edges(context, tokens, decoder_id=self.decoder_id)
        same_graph, adjacency = _attention_masks(context, tokens, edges)
        hidden = self.input(context, tokens)
        for block in self.blocks:
            hidden = block(hidden, same_graph, adjacency)
        pooled = segment_mean(hidden, tokens.token_graph, tokens.n_graphs)
        out = self.head(pooled)
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
            "structure": ("shared §5.5 tokens -> d48 + token_type bias -> 2 "
                          "pre-norm transformer blocks (4 heads, head_dim 12, "
                          "FFN 80, within-graph mask + local token bias) -> "
                          "mean pool -> 2 (plan §6.3 :546)"),
            "dim": int(self.dim),
            "heads": int(self.heads),
            "head_dim": int(self.dim) // int(self.heads),
            "ffn": int(self.ffn),
            "layers": int(self.layers),
            "init_seed": int(self.init_seed),
            "n_trainable_params": self.n_trainable_params,
            "zero_init": True,
        }


class ProcessP4Control(torch.nn.Module):
    """``P4-C``: P4's width/tokens/params with pooling+MLP instead of attention.

    "与 P4 同宽、同 token、同参数 ±2%，MLP/池化替代 attention" (plan §6.3 :547).
    The construction-time assertion is the INDEPENDENT ±2 % band around P4's
    MEASURED count (:func:`assert_p4c_matches_p4`), not the 36 K ±5 % band; the
    candidate happens to sit inside both, and the two bands are still never
    conflated in the record.
    """

    decoder_id: Final[str] = DECODER_ID_P4C
    track: Final[str] = TRACK_PROCESS
    model_version: Final[str] = MODEL_VERSION_P4C
    equivariance: Final[str] = "invariant"
    stochastic: Final[bool] = False
    nfe: Final[int] = 1
    zero_init: Final[bool] = True

    def __init__(self, *, dim: int = P4_DIM, ffn: int = P4_FFN,
                 layers: int = P4_LAYERS,
                 mix_hidden: int = P4C_MIX_HIDDEN,
                 seed: int = PROCESS_INIT_SEED_V543) -> None:
        super().__init__()
        if int(dim) < 1 or int(ffn) < 1 or int(layers) < 1 \
                or int(mix_hidden) < 1:
            raise ValueError("P4-C widths/depth must be >= 1")
        self.init_seed: Final[int] = int(seed)
        self.dim: Final[int] = int(dim)
        self.ffn: Final[int] = int(ffn)
        self.layers: Final[int] = int(layers)
        self.mix_hidden: Final[int] = int(mix_hidden)
        with deterministic_init(self.init_seed):
            self.input = _TokenInput(self.dim)
            self.blocks = torch.nn.ModuleList([
                _PooledMixBlock(self.dim, self.ffn, self.mix_hidden)
                for _ in range(self.layers)])
            self.head = torch.nn.Linear(self.dim, P4_OUTPUTS)
        zero_init_last_layer(self.head)
        # the ±2 % matched band (plan §6.3 :547), asserted at CONSTRUCTION
        assert_p4c_matches_p4(measured_params(self))

    # -- protocol -----------------------------------------------------------
    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:
        """Pooled-mix control → ``ProcessResidual`` (plan §6.1 :496)."""
        context = assert_process_context(ctx, decoder_id=self.decoder_id)
        tokens = token_tensors(context, decoder_id=self.decoder_id)
        hidden = self.input(context, tokens)
        for block in self.blocks:
            hidden = block(hidden, tokens)
        pooled = segment_mean(hidden, tokens.token_graph, tokens.n_graphs)
        out = self.head(pooled)
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
            "structure": ("same d48 token input and readout as P4; per-graph "
                          "mean pooling + 96->64->48 MLP replaces attention; "
                          "same FFN 80 and block depth (plan §6.3 :547)"),
            "dim": int(self.dim),
            "ffn": int(self.ffn),
            "layers": int(self.layers),
            "mix_hidden": int(self.mix_hidden),
            "reference_decoder_id": P4C_REFERENCE_DECODER_ID,
            "reference_params": p4_reference_params(),
            "matched_band_pct": P4C_BAND_PCT,
            "init_seed": int(self.init_seed),
            "n_trainable_params": self.n_trainable_params,
            "zero_init": True,
        }


# ---------------------------------------------------------------------------
# parameter assertions
# ---------------------------------------------------------------------------
def assert_p4_budget(n_trainable_params: int) -> int:
    """Construction-time §6.4 band assertion of ``P4``."""
    return assert_candidate_budget(n_trainable_params, decoder_id=DECODER_ID_P4)


@lru_cache(maxsize=1)
def p4_reference_params() -> int:
    """P4's MEASURED parameter count — the reference of the ±2 % band (:547).

    Built once (the count is seed-independent) and cached; every P4-C
    construction re-checks itself against it, so the matched band can never
    drift silently if P4's width changes.
    """
    return measured_params(ProcessGraphTransformer())


def p4c_band(reference: int | None = None) -> dict[str, Any]:
    """The INDEPENDENT ±2 % band of ``P4-C`` (plan §6.3 :547).

    Deliberately NOT the §6.4 36 K ±5 % band: ``pct`` is 2.0 and ``reference``
    is P4's measured count.  Returned as a plain dict so a record can quote it.
    """
    base = int(p4_reference_params() if reference is None else reference)
    delta = base * P4C_BAND_PCT / 100.0
    return {
        "kind": "matched_to_reference_pct",
        "reference_decoder_id": P4C_REFERENCE_DECODER_ID,
        "reference_params": base,
        "band_pct": float(P4C_BAND_PCT),
        "min": int(math.floor(base - delta)),
        "max": int(math.ceil(base + delta)),
    }


def assert_p4c_matches_p4(n_trainable_params: int, *,
                          reference: int | None = None) -> int:
    """Refuse a P4-C whose measured count leaves «P4 measured ±2 %» (:547).

    Refusal: ``V543_P4C_PARAM_MATCH`` (typed :class:`V543ParamBudgetError` with
    ``code``/``detail``); the detail names both counts, the deviation in percent
    and the band, because a control that is not parameter matched makes the
    P4-vs-P4-C comparison uninterpretable (hypothesis H-P2, plan §3.2 :167).
    """
    base = int(p4_reference_params() if reference is None else reference)
    count = int(n_trainable_params)
    delta = base * P4C_BAND_PCT / 100.0
    low = int(math.floor(base - delta))
    high = int(math.ceil(base + delta))
    if not low <= count <= high:
        deviation = 100.0 * (count - base) / float(base) if base else 0.0
        raise V543ParamBudgetError(
            "V543_P4C_PARAM_MATCH",
            f"{DECODER_ID_P4C}: {count} trainable parameters are "
            f"{deviation:+.2f} % away from {P4C_REFERENCE_DECODER_ID}'s "
            f"measured {base}; the control band is ±{P4C_BAND_PCT} % "
            f"[{low}, {high}] (plan §6.3 :547 '与 P4 同参数 ±2%' — an "
            "independent band, not the §6.4 36K ±5% rule)")
    return count


# ---------------------------------------------------------------------------
# the P4-C record (13 §6.1 fields + the matched-band provenance)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class P4ControlSpec:
    """``P4-C``'s §6.1 record, built without touching the WP4 registry.

    Why a separate record type: plan §6.3 :547 declares ``P4-C`` but
    ``registry.DECODER_IDS_V543`` does not (:539–548 lists P0–P5), and a
    :class:`~pes2ts_core.pesn.models.v543.registry.DecoderSpec` with id
    ``P4-C`` is a ``V543_REGISTRY_UNKNOWN`` refusal.  Editing that shared file
    is outside WP6's write scope, so the record is emitted here with the SAME
    :data:`~pes2ts_core.pesn.models.v543.registry.REGISTRY_RECORD_FIELDS` and
    the band provenance carried explicitly (``band`` in :meth:`record_extra`).
    """

    decoder_id: str = DECODER_ID_P4C
    track: str = TRACK_PROCESS
    model_version: str = MODEL_VERSION_P4C
    contract_version: str = DECODER_CONTEXT_SCHEMA_V543
    equivariance: str = "invariant"
    stochastic: bool = False
    nfe: int = 1
    zero_init: bool = True
    status: str = P4C_STATUS
    checkpoint_sha256: str = ""
    config_sha256: str = ""
    acceptance: str = (
        "wp6: diagnostic control (plan §6.3 :547 — 仅 P4 进 top2 时), "
        "constructed, matched to P4 within ±2 % at construction, "
        "zero-residual init, pair/token permutation invariant; S1 smoke "
        "pending")
    init_seed: int = PROCESS_INIT_SEED_V543
    reference_decoder_id: str = P4C_REFERENCE_DECODER_ID

    def __post_init__(self) -> None:
        if self.decoder_id != DECODER_ID_P4C:
            raise V543ParamBudgetError(
                "V543_P4C_ID",
                f"P4ControlSpec only describes {DECODER_ID_P4C!r}, got "
                f"{self.decoder_id!r}")
        if self.status not in STATUS_VALUES_V543:
            raise V543ParamBudgetError(
                "V543_P4C_STATUS",
                f"status must be one of {list(STATUS_VALUES_V543)}, got "
                f"{self.status!r}")
        if self.contract_version != DECODER_CONTEXT_SCHEMA_V543:
            raise V543ParamBudgetError(
                "V543_P4C_CONTRACT",
                f"contract_version must be {DECODER_CONTEXT_SCHEMA_V543!r}")

    def record(self, n_trainable_params: int) -> dict[str, Any]:
        """Exactly the §6.1 :503–517 field list, in the plan's order."""
        values: dict[str, Any] = {
            "decoder_id": self.decoder_id,
            "track": self.track,
            "model_version": self.model_version,
            "contract_version": self.contract_version,
            "n_trainable_params": int(n_trainable_params),
            "equivariance": self.equivariance,
            "stochastic": bool(self.stochastic),
            "nfe": int(self.nfe),
            "zero_init": bool(self.zero_init),
            "status": self.status,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "acceptance": self.acceptance,
        }
        return {name: values[name] for name in REGISTRY_RECORD_FIELDS}

    def record_extra(self, n_trainable_params: int) -> dict[str, Any]:
        """The 13-field record plus its band/seed provenance (report payload)."""
        payload = self.record(n_trainable_params)
        payload.update({
            "band": p4c_band(),
            "param_band_rule": (
                "independent frozen band: P4 measured ±2 % (plan §6.3 :547); "
                "NOT the §6.4 36,000 ±5 % band"),
            "init_seed": int(self.init_seed),
            "reference_decoder_id": str(self.reference_decoder_id),
            "zero_init_rule": "plan §4.5 :246 (last Linear(·,2) zeroed)",
            "record_fields": list(REGISTRY_RECORD_FIELDS),
        })
        return payload


#: ``P4-C``'s record (frozen).
P4C_SPEC: Final[P4ControlSpec] = P4ControlSpec()


def build_p4(**kwargs: Any) -> ProcessGraphTransformer:
    """Registry factory for ``P4``."""
    return ProcessGraphTransformer(**kwargs)


def build_p4c(**kwargs: Any) -> ProcessP4Control:
    """Factory for ``P4-C`` (not registry-declarable, see :class:`P4ControlSpec`)."""
    return ProcessP4Control(**kwargs)


#: The §6.1 registry spec of ``P4`` (P4-C has no DecoderSpec — see above).
P4_SPEC: Final[DecoderSpec] = DecoderSpec(
    decoder_id=DECODER_ID_P4,
    track=TRACK_PROCESS,
    model_version=MODEL_VERSION_P4,
    factory=build_p4,
    equivariance="invariant",
    stochastic=False,
    nfe=1,
    zero_init=True,
    status="proposed",
    acceptance=(
        "wp6: constructed, sum(p.numel()) asserted in band, zero-residual "
        "init, token-level attention only (within-graph mask, local token "
        "bias, no absolute coordinates), pair/token permutation invariant; "
        "S1 smoke pending"),
)


__all__ = [
    "DECODER_ID_P4",
    "DECODER_ID_P4C",
    "MODEL_VERSION_P4",
    "MODEL_VERSION_P4C",
    "P4C_BAND_PCT",
    "P4C_EXPECTED_PARAMS",
    "P4C_MIX_HIDDEN",
    "P4C_REFERENCE_DECODER_ID",
    "P4C_SPEC",
    "P4C_STATUS",
    "P4_DIM",
    "P4_EXPECTED_PARAMS",
    "P4_FFN",
    "P4_HEAD_DIM",
    "P4_HEADS",
    "P4_LAYERS",
    "P4_OUTPUTS",
    "P4_SPEC",
    "P4ControlSpec",
    "ProcessGraphTransformer",
    "ProcessP4Control",
    "assert_p4_budget",
    "assert_p4c_matches_p4",
    "build_p4",
    "build_p4c",
    "p4_reference_params",
    "p4c_band",
]
