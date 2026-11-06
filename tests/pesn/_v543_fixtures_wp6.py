"""WP6 shared fixtures for the PROCESS-candidate tests (CPU-fast).

Everything here exists ONLY to drive the WP6 candidate tests:

* :func:`real_process_context` runs the REAL frozen encoder + Bridge on the
  frozen collated fixture batch, so every candidate is exercised on a genuine
  ``ProcessContextV1`` (§5.2/§5.5) instead of a hand-made tensor bag;
* :func:`permute_pairs` / :func:`permute_key_tokens` are the two permutation
  probes of the §5.2 置换 column ("对置换等变" / "token 置换等变") that the WP6
  task asks for item by item;
* :func:`drop_all_pairs` builds the ``P = 0`` context plan §5.2 :309 declares
  legal ("允许 ``P=0``… Decoder 必须能处理空 pair");
* :func:`process_assembly` / :func:`process_model` wire a PROCESS candidate into
  the REAL ``V543Assembly`` + ``V543Model`` flow.  They construct the assembly
  through the public :class:`V543Assembly` constructor and public
  :class:`DecoderBinding` record rather than
  :func:`~pes2ts_core.pesn.models.v543.assembly.assemble_v543`, because the
  latter routes through ``assert_decoder_protocol`` → ``track_of_decoder_id``
  which refuses the plan's ``P4-C`` id (see
  ``graph_transformer.py``'s plan ↔ WP4 conflict note).  Using the same public
  assembly object proves the §4.2 flow without touching a shared WP4 file."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
from pes2ts_core.pesn.models.v543.assembly import DecoderBinding, V543Assembly
from pes2ts_core.pesn.models.v543.contracts import (
    DecoderContextV1,
    ProcessContextV1,
)
from pes2ts_core.pesn.models.v543.latent_bridge import LatentBridge
from pes2ts_core.pesn.models.v543.model import V543Model
from pes2ts_core.pesn.models.v543.process.tokens import (
    TOKEN_TYPE_GRAPH_V543,
    TOKEN_TYPE_KEY_V543,
)
from pes2ts_core.pesn.models.v543.registry import (
    TRACK_MOVE,
    TRACK_PROCESS,
    ZeroMoveDecoder,
    count_trainable_params,
)
from pes2ts_core.pesn.models.v543 import CoarseEncoder
from tests.pesn._v543_fixtures_wp4 import batch_with_status


def fixture_batch() -> dict[str, Any]:
    """The frozen collated batch (4 graphs × 3 keys × 2 pairs)."""
    return batch_with_status(("ok",) * 4, jitter_seed=0)


def real_process_context() -> tuple[dict[str, Any], DecoderContextV1,
                                    ProcessContextV1]:
    """``(batch, context, process view)`` of one real encoder+Bridge forward."""
    batch = fixture_batch()
    view = forward_view_v3(dict(batch))
    packet = CoarseEncoder()(view)
    context = LatentBridge().build_context(packet, view)
    return batch, context, context.process_view()


# ---------------------------------------------------------------------------
# permutation probes (§5.2 置换 column)
# ---------------------------------------------------------------------------
def permute_pairs(ctx: ProcessContextV1, *, seed: int = 0
                  ) -> ProcessContextV1:
    """Permute ``pair_s``/``intent_pair_index`` rows WITHIN each graph.

    The pair token of a graph stays at its layout slot (plan §5.5: "pair token
    携带对应行号"), so ``token_index`` is untouched while the pair ROW content
    is reordered — the local index k of a graph now holds a different pair.  A
    graph-level scalar residual must not change (§5.2 置换列: PROCESS 输出对
    对置换不变).
    """
    generator = torch.Generator().manual_seed(int(seed))
    pair_s = torch.zeros_like(ctx.pair_s)
    pair_index = torch.zeros_like(ctx.intent_pair_index)
    for graph in range(int(ctx.pair_ptr.shape[0]) - 1):
        low = int(ctx.pair_ptr[graph])
        high = int(ctx.pair_ptr[graph + 1])
        count = high - low
        if count <= 0:
            continue
        local = torch.randperm(count, generator=generator)
        rows = torch.arange(low, high)
        pair_s[low + local] = ctx.pair_s[rows]
        pair_index[:, low + local] = ctx.intent_pair_index[:, rows]
    return replace(ctx, pair_s=pair_s, intent_pair_index=pair_index)


def permute_key_tokens(ctx: ProcessContextV1, *, seed: int = 0
                       ) -> ProcessContextV1:
    """Permute the KEY rows within each graph (i.e. the key-token order).

    ``key_s`` rows are reordered, ``intent_pair_index`` is remapped through the
    same permutation and the key tokens keep their layout slots but point at
    the new rows.  ``key_ptr``/``token_ptr``/``token_graph`` are unchanged
    because a per-graph permutation preserves every prefix sum.
    """
    generator = torch.Generator().manual_seed(int(seed))
    n_keys = int(ctx.key_s.shape[0])
    key_permutation = torch.arange(n_keys)          # old row -> new row
    key_s = torch.zeros_like(ctx.key_s)
    for graph in range(int(ctx.key_ptr.shape[0]) - 1):
        low = int(ctx.key_ptr[graph])
        high = int(ctx.key_ptr[graph + 1])
        count = high - low
        if count <= 0:
            continue
        local = torch.randperm(count, generator=generator)
        rows = torch.arange(low, high)
        key_permutation[rows] = low + local
        key_s[low + local] = ctx.key_s[rows]
    pair_index = key_permutation[ctx.intent_pair_index]
    types = ctx.token_type.to(torch.int64)
    token_index = ctx.token_index.clone()
    key_tokens = torch.nonzero(types == TOKEN_TYPE_KEY_V543,
                               as_tuple=False).flatten()
    token_index[key_tokens] = key_permutation[ctx.token_index[key_tokens]]
    return replace(ctx, key_s=key_s, intent_pair_index=pair_index,
                   token_index=token_index)


def drop_all_pairs(ctx: ProcessContextV1) -> ProcessContextV1:
    """The legal ``P = 0`` context of plan §5.2 :309 (no changed K-pair).

    Every graph keeps ``[graph_token, key_tokens…]`` only; ``pair_s`` is
    ``[0,48]`` and ``pair_ptr`` is all zeros.
    """
    n_graphs = int(ctx.graph_s.shape[0])
    types: list[int] = []
    row_index: list[int] = []
    token_graph: list[int] = []
    steps: list[int] = []
    for graph in range(n_graphs):
        low = int(ctx.key_ptr[graph])
        high = int(ctx.key_ptr[graph + 1])
        types.append(TOKEN_TYPE_GRAPH_V543)
        row_index.append(-1)
        token_graph.append(graph)
        for row in range(low, high):
            types.append(TOKEN_TYPE_KEY_V543)
            row_index.append(row)
            token_graph.append(graph)
        steps.append(1 + (high - low))
    token_ptr = torch.zeros(n_graphs + 1, dtype=torch.long)
    token_ptr[1:] = torch.cumsum(torch.tensor(steps, dtype=torch.long), dim=0)
    return replace(
        ctx,
        pair_s=ctx.pair_s.new_zeros((0, ctx.pair_s.shape[-1])),
        intent_pair_index=torch.zeros((2, 0), dtype=torch.long),
        pair_graph=torch.zeros((0,), dtype=torch.long),
        pair_ptr=torch.zeros(n_graphs + 1, dtype=torch.long),
        token_index=torch.tensor(row_index, dtype=torch.long),
        token_type=torch.tensor(types, dtype=torch.int8),
        token_graph=torch.tensor(token_graph, dtype=torch.long),
        token_ptr=token_ptr)


# ---------------------------------------------------------------------------
# assembly / model wiring
# ---------------------------------------------------------------------------
def process_assembly(decoder: torch.nn.Module, *, decoder_id: str,
                     encoder: Any | None = None) -> V543Assembly:
    """A REAL §4.2 assembly with ``decoder`` as the active PROCESS refiner."""
    encoder = encoder if encoder is not None else CoarseEncoder()
    move = DecoderBinding(
        decoder_id="M0", track=TRACK_MOVE,
        model_version="v543_m0_zero_residual_v1", role="baseline",
        decoder=ZeroMoveDecoder(), n_trainable_params=0)
    process = DecoderBinding(
        decoder_id=str(decoder_id), track=TRACK_PROCESS,
        model_version=str(getattr(decoder, "model_version", "unversioned")),
        role="active", decoder=decoder,
        n_trainable_params=count_trainable_params(decoder))
    assembly = V543Assembly(encoder, LatentBridge(), move=move,
                            process=process, active_track=TRACK_PROCESS)
    assembly.freeze_encoder()
    for parameter in process.decoder.parameters():
        parameter.requires_grad_(True)
    assembly.assert_param_budget()
    return assembly


def process_model(decoder: torch.nn.Module, *, decoder_id: str) -> V543Model:
    """``V543Model`` over :func:`process_assembly` (the §4.2 flow)."""
    return V543Model(process_assembly(decoder, decoder_id=decoder_id))


__all__ = [
    "drop_all_pairs",
    "fixture_batch",
    "permute_key_tokens",
    "permute_pairs",
    "process_assembly",
    "process_model",
    "real_process_context",
]
