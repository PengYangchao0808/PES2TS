"""V5.4.3 WP6 — the ONE PROCESS token definition, init discipline and
parameter-band helpers shared by the P1–P4 / P4-C candidates.

Why this file exists (and why it is not one of the plan's five candidate files)
-------------------------------------------------------------------------------
Plan §6.3 :552 makes the token definition a CONTRACT, not an implementation
detail:

    "**P2/P3/P4 必须使用同一 token 定义**（§5.5）；否则「局部信息有无收益」
    无法归因。"

If ``deepsets.py``, ``graph_gnn.py`` and ``graph_transformer.py`` each rebuilt
the ``[graph, keys…, pairs…]`` layout, the three candidates would differ in
their INPUT as well as in their inductive bias, and H-P1/H-P2 (plan §3.2
:166–167) would be unattributable.  So the layout, the local token adjacency
and the segment pooling live here ONCE and the candidates only own their
mixing operator.  §11.1 (:1088–1110) sketches one file per candidate; this
extra module is the minimal shared surface that makes §6.3 :552 true, and it
adds no candidate.

Normative sources, quoted as implemented
---------------------------------------
* **§5.5 token layout (:405)** — "每图 token 连续排列为 ``[graph_token,
  key_tokens…, pair_tokens…]``；``token_index`` 对 graph token 填 ``-1``，对
  key/pair 填对应行号；``token_type`` 区分三类."  :func:`token_tensors` reads
  the layout out of the :class:`ProcessContextV1` fields and REFUSES a context
  whose layout disagrees (it never re-derives a competing index).
* **§5.2 field table (:282–299)** — ``graph_s [G,64]`` / ``key_s [K,48]`` /
  ``pair_s [P,48]`` / ``token_index`` / ``token_type`` (0=graph, 1=key,
  2=pair) / ``token_graph`` / ``token_ptr``; ``T = token_ptr[-1] = G+K+P``.
* **§5.2.1 hard isolation (:349)** — the PROCESS view has **no ``u0``, no
  ``key_v``, no ``key_edge_*``**.  §6.3 :554 asks P4 for a "局部图 bias ... 用
  ``key_edge_index``/``token`` 邻接", but ``key_edge_index`` is NOT readable on
  this track.  The local adjacency is therefore built from the only legal
  source: the token structure itself (§5.5) — a pair token is adjacent to its
  two endpoint key tokens (``intent_pair_index``, §5.4) and every token is
  adjacent to its graph token.  This is the "``token`` 邻接" half of :554 and
  it is a documented, reported deviation from the literal ``key_edge_index``
  half (which §5.2.1 forbids).
* **§5.4 pair layout (:380–391)** — the 8 dims ``pair_s[20:28]`` are
  ``d_ij``, ``1/(d_ij+1)``, ``ΔZ/10``, ``|ΔZ|/10`` and the two ``is_aromatic``
  / ``is_spatial`` flags of rows 6–7: exactly the kind of translation- and
  rotation-invariant edge scalar §6.3 :545 asks for ("边特征 8").  P3's edge
  feature is that frozen slice — no new coordinate is introduced.
* **§4.5 zero initialisation (:246)** — "P1–P5 | 最后一层 ``Linear(·,2)`` 的
  weight/bias | ``δq = 0``、``δe = 0``"; :func:`zero_init_last_layer` is that
  rule and every candidate calls it, so the initial model reproduces V542 A0
  bit-for-bit (plan §4.4 boundary 3 :233).
* **§6.4 parameter assertion (:569)** — "构造时 ``sum(p.numel())`` 精确断言";
  :func:`assert_candidate_budget` delegates to
  :func:`~pes2ts_core.pesn.models.v543.registry.assert_budget` (34,200–37,800,
  hard cap 40,000) and every candidate calls it in ``__init__``.

Deterministic initialisation (fork_rng discipline)
--------------------------------------------------
Plan §7.2 :667 freezes the run seeds ``(17, 29, 43)`` and the campaign needs a
reproducible decoder *before* the trainer touches it.  :func:`deterministic_init`
wraps parameter creation in ``torch.random.fork_rng(devices=[])``: the global
RNG state is restored afterwards (so building a candidate cannot perturb the
sampler stream, §7.2 :667) and the parameters are a pure function of the seed.
Every candidate exposes ``init_seed`` so the value reaches the registry record.

Discipline: no candidate reads a batch/``pos``/supervision key (plan §6.1
:519), no ``torch.compile`` (plan §11.3), every constant is ``Final``."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Iterator

import torch

from pes2ts_core.pesn.models.v543.contracts import (
    GRAPH_S_DIM_V543,
    KEY_S_DIM_V543,
    PAIR_S_DIM_V543,
    ProcessContextV1,
    V543Error,
)
from pes2ts_core.pesn.models.v543.registry import (
    TRACK_PROCESS,
    assert_budget,
    count_trainable_params,
)

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------
#: ``token_type`` vocabulary of plan §5.2 (:295) / §5.5 (:405).
TOKEN_TYPE_GRAPH_V543: Final[int] = 0
TOKEN_TYPE_KEY_V543: Final[int] = 1
TOKEN_TYPE_PAIR_V543: Final[int] = 2
N_TOKEN_TYPES_V543: Final[int] = 3

#: Token feature width every candidate consumes — ``key_s``/``pair_s`` are 48
#: (plan §5.2), so a token is a 48-dim vector before any candidate projection.
PROCESS_TOKEN_DIM_V543: Final[int] = KEY_S_DIM_V543

#: Frozen initialisation seed of the campaign (plan §7.2 :667 first seed).
PROCESS_INIT_SEED_V543: Final[int] = 17

#: The 8 frozen ``pair_s`` columns used as the P3 edge feature (§5.4 :386–387:
#: ``d_ij``, ``1/(d_ij+1)``, ``ΔZ/10``, ``|ΔZ|/10``, ``is_aromatic``,
#: ``is_spatial`` plus the two bounds the slice spans).
PAIR_EDGE_FEATURE_SLICE_V543: Final[tuple[int, int]] = (20, 28)
#: Width of that edge feature (§6.3 :545 "边特征 8").
PAIR_EDGE_FEATURE_DIM_V543: Final[int] = (
    PAIR_EDGE_FEATURE_SLICE_V543[1] - PAIR_EDGE_FEATURE_SLICE_V543[0])

#: Local edge-type vocabulary (the two adjacencies of §5.5 that this track can
#: legally construct, see the module docstring).
EDGE_TYPE_GRAPH_TOKEN_V543: Final[int] = 0
EDGE_TYPE_PAIR_KEY_V543: Final[int] = 1
N_EDGE_TYPES_V543: Final[int] = 2


class V543ProcessDecoderError(V543Error):
    """PROCESS candidate refusal (context type, layout, band, readout)."""


# ---------------------------------------------------------------------------
# deterministic initialisation
# ---------------------------------------------------------------------------
@contextmanager
def deterministic_init(seed: int) -> Iterator[None]:
    """Create layers inside a forked RNG so ``seed`` fully determines them.

    ``fork_rng(devices=[])`` never touches CUDA and restores the global CPU
    generator on exit, so constructing a candidate cannot shift the sampler
    stream the trainer draws from (plan §7.2 :667).  The negative-seed guard is
    explicit because ``torch.manual_seed`` silently wraps on some platforms.
    """
    value = int(seed)
    if value < 0:
        raise V543ProcessDecoderError(
            "V543_PROCESS_SEED",
            f"init seed must be non-negative, got {value}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(value)
        yield


def zero_init_last_layer(layer: torch.nn.Linear) -> torch.nn.Linear:
    """Zero the weight AND bias of the residual head (plan §4.5 :246).

    "P1–P5 | 最后一层 ``Linear(·,2)`` 的 weight/bias | ``δq = 0``、``δe = 0``".
    Zeroing both is what makes the initial output EXACTLY zero rather than
    "small"; a zero weight with a live bias would emit a constant non-zero
    residual and break the §5.8 zero-residual parity.
    """
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.zero_()
    return layer


def assert_candidate_budget(n_trainable_params: int, *, decoder_id: str) -> int:
    """Construction-time §6.4 assertion for a PROCESS candidate.

    Delegates to :func:`~pes2ts_core.pesn.models.v543.registry.assert_budget`
    (band 34,200–37,800, hard cap 40,000) so the campaign has ONE band
    definition; the PROCESS-track context is asserted here so a MOVE id cannot
    reach this helper.
    """
    return assert_budget(int(n_trainable_params), decoder_id=str(decoder_id),
                         track=TRACK_PROCESS)


# ---------------------------------------------------------------------------
# the token view of a ProcessContextV1 (§5.5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TokenTensors:
    """Row indices of the frozen ``[graph, keys…, pairs…]`` token layout.

    Read-only, parameter-free projection of a :class:`ProcessContextV1`; it
    holds no tensor of its own beyond index views, so the candidates cannot
    acquire a second copy of the context (plan §5.1 principle 1).

    Fields:
        token_type: ``[T]`` int64 (widened from the context's int8).
        token_graph: ``[T]`` graph id of each token.
        token_ptr: ``[G+1]`` token prefix sums.
        graph_token_rows: ``[G]`` the one graph token of each graph.
        key_token_rows / key_row_of_token: ``[K]`` mutually inverse maps
            (token row ↔ ``key_s`` row).
        pair_token_rows / pair_row_of_token: ``[P]`` the same for ``pair_s``.
    """

    token_type: torch.Tensor
    token_graph: torch.Tensor
    token_ptr: torch.Tensor
    graph_token_rows: torch.Tensor
    key_token_rows: torch.Tensor
    key_row_of_token: torch.Tensor
    pair_token_rows: torch.Tensor
    pair_row_of_token: torch.Tensor
    n_graphs: int
    n_keys: int
    n_pairs: int
    n_tokens: int

    @property
    def payload_mask(self) -> torch.Tensor:
        """``[T]`` bool — True on key/pair tokens, False on graph tokens.

        Pooling must exclude the graph token, otherwise the graph-level vector
        would be counted twice (it is also concatenated raw by P2/P3).
        """
        return self.token_type != TOKEN_TYPE_GRAPH_V543

    def key_mask(self) -> torch.Tensor:
        return self.token_type == TOKEN_TYPE_KEY_V543

    def pair_mask(self) -> torch.Tensor:
        return self.token_type == TOKEN_TYPE_PAIR_V543


def assert_process_context(ctx: object, *, decoder_id: str) -> ProcessContextV1:
    """Typed refusal unless ``ctx`` is a :class:`ProcessContextV1`.

    The one-argument §6.1 protocol means a MOVE context must never be accepted
    here (plan §5.2.1 :348–352); refusing by TYPE is what proves the isolation.
    """
    if not isinstance(ctx, ProcessContextV1):
        raise V543ProcessDecoderError(
            "V543_PROCESS_CONTEXT_TYPE",
            f"{decoder_id} takes a ProcessContextV1 (plan §6.1 :496); got "
            f"{type(ctx).__name__}")
    return ctx


def token_tensors(ctx: ProcessContextV1, *, decoder_id: str = "PROCESS"
                  ) -> TokenTensors:
    """Validate and index the §5.5 token layout of ``ctx``.

    Refusals (typed, ``code``/``detail``):

    * ``V543_PROCESS_TOKEN_LAYOUT`` — ``T != token_ptr[-1]``, a non-contiguous
      per-graph span, a graph token missing at the span start, or a ``token_type``
      value outside ``{0,1,2}``;
    * ``V543_PROCESS_TOKEN_INDEX`` — a key/pair token whose ``token_index`` is
      out of range or does not cover every row exactly once (the layout must be
      a bijection between key/pair rows and their tokens, §5.5 :405).

    The layout is READ, never rebuilt: a candidate that silently re-derived the
    token order would no longer be comparable with P2/P3/P4 (§6.3 :552).
    """
    context = assert_process_context(ctx, decoder_id=decoder_id)
    types = context.token_type.to(torch.int64)
    graph = context.token_graph.to(torch.int64)
    ptr = context.token_ptr.to(torch.int64)
    index = context.token_index.to(torch.int64)
    n_graphs = int(context.graph_s.shape[0])
    n_keys = int(context.key_s.shape[0])
    n_pairs = int(context.pair_s.shape[0])
    n_tokens = int(index.shape[0])
    if int(ptr.shape[0]) != n_graphs + 1 or int(ptr[-1]) != n_tokens:
        raise V543ProcessDecoderError(
            "V543_PROCESS_TOKEN_LAYOUT",
            f"{decoder_id}: token_ptr must be [G+1] with token_ptr[-1] == T "
            f"(plan §5.2 :307); G={n_graphs}, T={n_tokens}, ptr="
            f"{ptr.tolist()}")
    if n_tokens != n_graphs + n_keys + n_pairs:
        raise V543ProcessDecoderError(
            "V543_PROCESS_TOKEN_LAYOUT",
            f"{decoder_id}: T={n_tokens} != G+K+P="
            f"{n_graphs + n_keys + n_pairs} (plan §5.2 :307)")
    if n_tokens and (int(types.min()) < TOKEN_TYPE_GRAPH_V543
                     or int(types.max()) > TOKEN_TYPE_PAIR_V543):
        raise V543ProcessDecoderError(
            "V543_PROCESS_TOKEN_LAYOUT",
            f"{decoder_id}: token_type must be 0=graph/1=key/2=pair "
            f"(plan §5.2 :295), got {sorted(set(types.tolist()))}")
    graph_rows: list[int] = []
    for g in range(n_graphs):
        start, stop = int(ptr[g]), int(ptr[g + 1])
        if stop <= start:
            raise V543ProcessDecoderError(
                "V543_PROCESS_TOKEN_LAYOUT",
                f"{decoder_id}: graph {g} has an empty token span")
        if int(types[start]) != TOKEN_TYPE_GRAPH_V543:
            raise V543ProcessDecoderError(
                "V543_PROCESS_TOKEN_LAYOUT",
                f"{decoder_id}: graph {g} must start with its graph token "
                "(每图 [graph, keys…, pairs…], plan §5.5)")
        graph_rows.append(start)
    key_mask = types == TOKEN_TYPE_KEY_V543
    pair_mask = types == TOKEN_TYPE_PAIR_V543
    key_token_rows = torch.nonzero(key_mask, as_tuple=False).flatten()
    pair_token_rows = torch.nonzero(pair_mask, as_tuple=False).flatten()
    if int(key_token_rows.numel()) != n_keys:
        raise V543ProcessDecoderError(
            "V543_PROCESS_TOKEN_INDEX",
            f"{decoder_id}: {int(key_token_rows.numel())} key tokens for "
            f"K={n_keys} (plan §5.5: one token per key row)")
    if int(pair_token_rows.numel()) != n_pairs:
        raise V543ProcessDecoderError(
            "V543_PROCESS_TOKEN_INDEX",
            f"{decoder_id}: {int(pair_token_rows.numel())} pair tokens for "
            f"P={n_pairs} (plan §5.5: one token per pair row)")
    key_row_of_token = index[key_token_rows]
    pair_row_of_token = index[pair_token_rows]
    _assert_bijection(key_row_of_token, n_keys,
                      decoder_id=decoder_id, what="key")
    _assert_bijection(pair_row_of_token, n_pairs,
                      decoder_id=decoder_id, what="pair")
    return TokenTensors(
        token_type=types,
        token_graph=graph,
        token_ptr=ptr,
        graph_token_rows=torch.tensor(graph_rows, dtype=torch.long),
        key_token_rows=key_token_rows,
        key_row_of_token=key_row_of_token,
        pair_token_rows=pair_token_rows,
        pair_row_of_token=pair_row_of_token,
        n_graphs=n_graphs,
        n_keys=n_keys,
        n_pairs=n_pairs,
        n_tokens=n_tokens,
    )


def _assert_bijection(rows: torch.Tensor, total: int, *, decoder_id: str,
                      what: str) -> None:
    """The token→row map must hit every row of ``what`` exactly once (§5.5)."""
    if total and (bool((rows < 0).any()) or bool((rows >= total).any())):
        raise V543ProcessDecoderError(
            "V543_PROCESS_TOKEN_INDEX",
            f"{decoder_id}: {what} token_index outside [0,{total})")
    if int(rows.numel()) == total and int(torch.unique(rows).numel()) != total:
        raise V543ProcessDecoderError(
            "V543_PROCESS_TOKEN_INDEX",
            f"{decoder_id}: {what} tokens do not cover the {total} "
            f"{what} rows exactly once (plan §5.5 :405)")


# ---------------------------------------------------------------------------
# token feature assembly
# ---------------------------------------------------------------------------
def scatter_token_features(
        tokens: TokenTensors,
        key_values: torch.Tensor,
        pair_values: torch.Tensor,
) -> torch.Tensor:
    """``[T, d]`` token features with ZEROS on the graph-token rows.

    ``key_s``/``pair_s`` are already ``[·,48]`` (plan §5.2), so no projection is
    hidden here: a candidate that wants its own width projects afterwards, and
    a graph token is left at zero for the candidate's own graph embedding.  The
    scatter is ``index_add_`` on a fresh tensor, which keeps autograd intact
    (no in-place write into a leaf).
    """
    n_tokens = int(tokens.n_tokens)
    width = int(key_values.shape[-1]) if key_values.numel() else \
        int(pair_values.shape[-1])
    out = key_values.new_zeros((n_tokens, width))
    if tokens.n_keys:
        out.index_add_(0, tokens.key_token_rows, key_values)
    if tokens.n_pairs:
        out.index_add_(0, tokens.pair_token_rows, pair_values)
    return out


def graph_token_embedding(token_out: torch.Tensor,
                          tokens: TokenTensors) -> torch.Tensor:
    """``[G, d]`` rows that the graph tokens of ``token_out`` must receive."""
    return token_out.index_select(0, tokens.graph_token_rows)


# ---------------------------------------------------------------------------
# segment pooling (per graph)
# ---------------------------------------------------------------------------
def segment_mean(x: torch.Tensor, graph: torch.Tensor, n_graphs: int
                 ) -> torch.Tensor:
    """Per-graph mean of ``x [n, d]`` over the rows carrying each ``graph`` id.

    ``index_add_`` summation (autograd-safe); a graph with no row is refused
    rather than silently zeroed, because §5.2 :309 guarantees at least one key
    per graph and a silent zero would hide a broken context.
    """
    width = int(x.shape[-1])
    sums = x.new_zeros((int(n_graphs), width)).index_add_(0, graph, x)
    counts = x.new_zeros((int(n_graphs),)).index_add_(
        0, graph, x.new_ones((int(x.shape[0]),)))
    if bool((counts <= 0).any()):
        raise V543ProcessDecoderError(
            "V543_PROCESS_POOL_EMPTY",
            "every graph must contribute at least one token to the pooled "
            "readout (plan §5.2 :309: 每个图至少 1 个 key)")
    return sums / counts.unsqueeze(-1)


def segment_max(x: torch.Tensor, graph: torch.Tensor, n_graphs: int
                ) -> torch.Tensor:
    """Per-graph max of ``x [n, d]`` (DeepSets max branch, plan §6.3 :544)."""
    width = int(x.shape[-1])
    out = x.new_empty((int(n_graphs), width))
    for g in range(int(n_graphs)):
        rows = x[graph == g]
        if int(rows.shape[0]) == 0:
            raise V543ProcessDecoderError(
                "V543_PROCESS_POOL_EMPTY",
                f"graph {g} contributes no token to the max-pooling branch "
                "(plan §5.2 :309)")
        out[g] = rows.max(dim=0).values
    return out


# ---------------------------------------------------------------------------
# local token adjacency (§6.3 :554 "token 邻接", §5.2.1-safe)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TokenEdges:
    """Directed local token edges plus their frozen edge features.

    ``edge_index [2,E]`` holds ``(src, dst)`` token rows (both directions are
    present, deduplicated, no self loops); ``edge_feature [E,8]`` is
    ``pair_s[20:28]`` on a pair→key edge and zeros on a graph-token edge;
    ``edge_type [E]`` selects the candidate's learned edge-type embedding.
    """

    edge_index: torch.Tensor
    edge_feature: torch.Tensor
    edge_type: torch.Tensor

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def local_token_edges(ctx: ProcessContextV1, tokens: TokenTensors,
                      *, decoder_id: str = "PROCESS") -> TokenEdges:
    """Build the within-graph local adjacency of the §5.5 token layout.

    The two legal adjacencies (see the module docstring — §5.2.1 :349 removes
    ``key_edge_index`` from this track):

    1. **graph token ↔ every token of its graph** (the §5.5 layout prefix);
    2. **pair token ↔ the two key tokens of ``intent_pair_index``** (§5.4
       canonical ``i<j`` pair, §5.5 "pair token 携带对应行号").

    No absolute coordinate, no distance and no learned global frame enters the
    construction, so the adjacency is invariant under rotation and translation
    and equivariant under the two permutations the plan declares (§5.2 置换
    column: token 置换等变 / 对置换等变).
    """
    key_token_of_row = torch.zeros((tokens.n_keys,), dtype=torch.long)
    key_token_of_row.index_copy_(0, tokens.key_row_of_token,
                                 tokens.key_token_rows)
    pair_token_of_row = torch.zeros((max(tokens.n_pairs, 1),), dtype=torch.long)
    if tokens.n_pairs:
        pair_token_of_row.index_copy_(0, tokens.pair_row_of_token,
                                      tokens.pair_token_rows)
    sources: list[int] = []
    targets: list[int] = []
    pair_edge_rows: list[int] = []      # index into pair_s (edge feature)
    edge_types: list[int] = []
    for g in range(tokens.n_graphs):
        start = int(tokens.token_ptr[g])
        stop = int(tokens.token_ptr[g + 1])
        for row in range(start + 1, stop):
            sources.append(start)
            targets.append(row)
            sources.append(row)
            targets.append(start)
            edge_types.extend((EDGE_TYPE_GRAPH_TOKEN_V543,
                               EDGE_TYPE_GRAPH_TOKEN_V543))
            pair_edge_rows.extend((-1, -1))
    if tokens.n_pairs:
        for pair_row in range(tokens.n_pairs):
            pair_token = int(pair_token_of_row[pair_row])
            endpoints = ctx.intent_pair_index[:, pair_row].to(torch.long)
            for side in range(int(endpoints.shape[0])):
                key_row = int(endpoints[side])
                if not 0 <= key_row < tokens.n_keys:
                    raise V543ProcessDecoderError(
                        "V543_PROCESS_TOKEN_INDEX",
                        f"{decoder_id}: intent_pair_index[{side},{pair_row}]="
                        f"{key_row} outside [0,{tokens.n_keys})")
                key_token = int(key_token_of_row[key_row])
                sources.append(pair_token)
                targets.append(key_token)
                sources.append(key_token)
                targets.append(pair_token)
                edge_types.extend((EDGE_TYPE_PAIR_KEY_V543,
                                   EDGE_TYPE_PAIR_KEY_V543))
                pair_edge_rows.extend((pair_row, pair_row))
    width = PAIR_EDGE_FEATURE_DIM_V543
    if not sources:
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        return TokenEdges(
            edge_index=empty_index,
            edge_feature=ctx.pair_s.new_zeros((0, width)),
            edge_type=torch.zeros((0,), dtype=torch.long))
    # Deduplicate (src, dst) pairs while keeping the first occurrence's feature:
    # double-counting a message would expose how many times a token happens to
    # be adjacent, which is a degree artefact, not information.
    seen: dict[tuple[int, int], int] = {}
    kept_src: list[int] = []
    kept_dst: list[int] = []
    kept_pair: list[int] = []
    kept_type: list[int] = []
    for position, (src, dst) in enumerate(zip(sources, targets)):
        if src == dst:
            continue
        key = (int(src), int(dst))
        if key in seen:
            continue
        seen[key] = position
        kept_src.append(int(src))
        kept_dst.append(int(dst))
        kept_pair.append(int(pair_edge_rows[position]))
        kept_type.append(int(edge_types[position]))
    low, high = PAIR_EDGE_FEATURE_SLICE_V543
    pair_feature = ctx.pair_s[:, low:high]
    feature_rows = [
        (pair_feature[row] if row >= 0
         else ctx.pair_s.new_zeros((width,)))
        for row in kept_pair]
    edge_index = torch.tensor([kept_src, kept_dst], dtype=torch.long)
    edge_feature = (torch.stack(feature_rows) if feature_rows
                    else ctx.pair_s.new_zeros((0, width)))
    return TokenEdges(
        edge_index=edge_index,
        edge_feature=edge_feature.contiguous(),
        edge_type=torch.tensor(kept_type, dtype=torch.long))


def scatter_messages(messages: torch.Tensor, edge_index: torch.Tensor,
                     n_tokens: int) -> torch.Tensor:
    """Sum ``messages [E,d]`` into their DESTINATION token rows."""
    out = messages.new_zeros((int(n_tokens), int(messages.shape[-1])))
    out.index_add_(0, edge_index[1], messages)
    return out


# ---------------------------------------------------------------------------
# measured parameter helper (re-exported for the candidate modules)
# ---------------------------------------------------------------------------
def measured_params(module: torch.nn.Module) -> int:
    """``sum(p.numel())`` of ``module`` (plan §6.4 :569 — measured, never
    estimated)."""
    return count_trainable_params(module)


__all__ = [
    "EDGE_TYPE_GRAPH_TOKEN_V543",
    "EDGE_TYPE_PAIR_KEY_V543",
    "N_EDGE_TYPES_V543",
    "N_TOKEN_TYPES_V543",
    "PAIR_EDGE_FEATURE_DIM_V543",
    "PAIR_EDGE_FEATURE_SLICE_V543",
    "PROCESS_INIT_SEED_V543",
    "PROCESS_TOKEN_DIM_V543",
    "TOKEN_TYPE_GRAPH_V543",
    "TOKEN_TYPE_KEY_V543",
    "TOKEN_TYPE_PAIR_V543",
    "TokenEdges",
    "TokenTensors",
    "V543ProcessDecoderError",
    "assert_candidate_budget",
    "assert_process_context",
    "deterministic_init",
    "graph_token_embedding",
    "local_token_edges",
    "measured_params",
    "scatter_messages",
    "scatter_token_features",
    "segment_max",
    "segment_mean",
    "token_tensors",
    "zero_init_last_layer",
]
