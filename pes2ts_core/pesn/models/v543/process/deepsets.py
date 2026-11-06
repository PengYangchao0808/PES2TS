"""V5.4.3 WP6 — ``P2`` PROCESS candidate: DeepSets over the frozen
key/pair tokens (plan §6.3 :544, §6.4 :557–571, §4.5 :246, §11.1 :1107
``process/deepsets.py``).

Normative sources, quoted as implemented
---------------------------------------
* **§6.3 :544** — "``P2`` | DeepSets | key/pair token 编码 48→88；mean+max
  池化；与 ``graph_s`` 拼接（240）→ 128 → 2（约 35.7K）| 36K ±5% | 正式".
  Implemented literally: ONE shared ``Linear(48, 88)`` token encoder (key and
  pair tokens share it — the plan says "key/pair token 编码", not one per
  type), per-graph ``mean`` AND ``max`` pooling of the encoded KEY+PAIR tokens
  (the graph token is excluded so the graph vector is not counted twice — it is
  concatenated raw), ``64 + 88 + 88 = 240`` concatenation, ``→ 128 → 2``.
* **§5.5 :405 / §6.3 :552** — the tokens are the SHARED definition; they come
  from :func:`~pes2ts_core.pesn.models.v543.process.tokens.token_tensors`, the
  same index P3 and P4 use, so a P2-vs-P4 difference is the mixing operator and
  not the input.
* **§5.2 形状约束 :309** — "允许 ``P=0``（无 changed K-pair）… Decoder 必须能
  处理空 pair": the payload mask can leave only key tokens and the pooling
  still returns a valid ``[88]`` per graph; ``P=0`` is covered by a test.
* **§4.5 :246** — "P1–P5 | 最后一层 ``Linear(·,2)`` 的 weight/bias"; the last
  layer here is the ``128 → 2`` head, zeroed by :func:`zero_init_last_layer`.
* **§6.4 :563–571** — measured, not estimated: 35,418 params, in band, no
  padding.

Measured parameter count (``sum(p.numel())``, plan §6.4 :569)
------------------------------------------------------------
``48·88 + 88`` = 4,312; ``240·128 + 128`` = 30,848; ``128·2 + 2`` = 258
→ **35,418** (plan estimate "约 35.7K", −0.8 %).

Discipline: no data reading / loss / checkpoint selection (plan §6.1 :519), no
``torch.compile`` (plan §11.3), every constant ``Final``."""

from __future__ import annotations

from typing import Any, Final

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.models.v543.contracts import (
    GRAPH_S_DIM_V543,
    PAIR_S_DIM_V543,
    ProcessContextV1,
    ProcessResidual,
)
from pes2ts_core.pesn.models.v543.process.tokens import (
    PROCESS_INIT_SEED_V543,
    PROCESS_TOKEN_DIM_V543,
    assert_candidate_budget,
    assert_process_context,
    deterministic_init,
    measured_params,
    scatter_token_features,
    segment_max,
    segment_mean,
    token_tensors,
    zero_init_last_layer,
)
from pes2ts_core.pesn.models.v543.registry import TRACK_PROCESS, DecoderSpec

#: ``P2`` id (plan §6.3 :544).
DECODER_ID_P2: Final[str] = "P2"
#: Registered model-layer version of this candidate.
MODEL_VERSION_P2: Final[str] = "v543_p2_deepsets_v1"
#: Plan §6.3 :544 "key/pair token 编码 48→88".
P2_TOKEN_HIDDEN: Final[int] = 88
#: Plan §6.3 :544 "（240）→ 128 → 2": the readout width.
P2_READOUT: Final[int] = 128
#: Wide vector fed to the readout: ``graph_s[64] + mean[88] + max[88]``.
P2_FEATURE_DIM: Final[int] = (
    GRAPH_S_DIM_V543 + 2 * P2_TOKEN_HIDDEN)
#: Residual scalars (``δq_raw``, ``δe``).
P2_OUTPUTS: Final[int] = 2
#: Measured parameter count of the frozen structure (§6.4 :569).
P2_EXPECTED_PARAMS: Final[int] = (
    PROCESS_TOKEN_DIM_V543 * P2_TOKEN_HIDDEN + P2_TOKEN_HIDDEN
    + P2_FEATURE_DIM * P2_READOUT + P2_READOUT
    + P2_READOUT * P2_OUTPUTS + P2_OUTPUTS)

#: Sanity: the concatenation the plan spells out is 240 (§6.3 :544).
assert P2_FEATURE_DIM == 240, P2_FEATURE_DIM
#: Sanity: key/pair tokens really are 48-dim (§5.2 field table).
assert PROCESS_TOKEN_DIM_V543 == PAIR_S_DIM_V543 == 48


class ProcessDeepSets(torch.nn.Module):
    """``P2``: permutation-invariant DeepSets readout (plan §6.3 :544).

    The model is invariant to the ORDER of the key and pair tokens by
    construction (mean and max are symmetric), which is exactly the §5.2 置换
    column: the PROCESS output is a graph-level scalar pair, so pair 置换 and
    token 置换 must not change it.

    Args:
        token_hidden: encoded token width (default 88, the plan's start).
        readout: readout width (default 128).
        seed: initialisation seed (fork_rng discipline, plan §7.2 :667).
    """

    decoder_id: Final[str] = DECODER_ID_P2
    track: Final[str] = TRACK_PROCESS
    model_version: Final[str] = MODEL_VERSION_P2
    equivariance: Final[str] = "invariant"
    stochastic: Final[bool] = False
    nfe: Final[int] = 1
    zero_init: Final[bool] = True

    def __init__(self, *, token_hidden: int = P2_TOKEN_HIDDEN,
                 readout: int = P2_READOUT,
                 seed: int = PROCESS_INIT_SEED_V543) -> None:
        super().__init__()
        if int(token_hidden) < 1 or int(readout) < 1:
            raise ValueError("P2 widths must be >= 1")
        self.init_seed: Final[int] = int(seed)
        self.token_hidden: Final[int] = int(token_hidden)
        self.readout: Final[int] = int(readout)
        feature_dim = GRAPH_S_DIM_V543 + 2 * self.token_hidden
        with deterministic_init(self.init_seed):
            self.encode = torch.nn.Linear(PROCESS_TOKEN_DIM_V543,
                                          self.token_hidden)
            self.fc1 = torch.nn.Linear(feature_dim, self.readout)
            self.head = torch.nn.Linear(self.readout, P2_OUTPUTS)
        zero_init_last_layer(self.head)
        assert_p2_budget(measured_params(self))

    # -- protocol -----------------------------------------------------------
    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:
        """``key_s``/``pair_s`` tokens → ``ProcessResidual`` (§6.1 :496)."""
        context = assert_process_context(ctx, decoder_id=self.decoder_id)
        tokens = token_tensors(context, decoder_id=self.decoder_id)
        raw = scatter_token_features(tokens, context.key_s, context.pair_s)
        payload = raw[tokens.payload_mask]
        encoded = F.silu(self.encode(payload))
        graphs = tokens.token_graph[tokens.payload_mask]
        mean = segment_mean(encoded, graphs, tokens.n_graphs)
        peak = segment_max(encoded, graphs, tokens.n_graphs)
        features = torch.cat((context.graph_s, mean, peak), dim=-1)
        hidden = F.silu(self.fc1(features))
        out = self.head(hidden)
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
            "structure": ("tokens key/pair 48->88, mean+max pool, "
                          "cat(graph_s,mean,max)=240 -> 128 -> 2 "
                          "(plan §6.3 :544)"),
            "token_hidden": int(self.token_hidden),
            "readout": int(self.readout),
            "feature_dim": GRAPH_S_DIM_V543 + 2 * int(self.token_hidden),
            "pooling": "mean+max",
            "init_seed": int(self.init_seed),
            "n_trainable_params": self.n_trainable_params,
            "zero_init": True,
        }


def assert_p2_budget(n_trainable_params: int) -> int:
    """Construction-time §6.4 band assertion of ``P2``."""
    return assert_candidate_budget(n_trainable_params, decoder_id=DECODER_ID_P2)


def build_p2(**kwargs: Any) -> ProcessDeepSets:
    """Registry factory."""
    return ProcessDeepSets(**kwargs)


#: The §6.1 registry spec of ``P2``.
P2_SPEC: Final[DecoderSpec] = DecoderSpec(
    decoder_id=DECODER_ID_P2,
    track=TRACK_PROCESS,
    model_version=MODEL_VERSION_P2,
    factory=build_p2,
    equivariance="invariant",
    stochastic=False,
    nfe=1,
    zero_init=True,
    status="proposed",
    acceptance=(
        "wp6: constructed, sum(p.numel()) asserted in band, zero-residual "
        "init, pair/token permutation invariant (mean+max pooling); "
        "S1 smoke pending"),
)


__all__ = [
    "DECODER_ID_P2",
    "MODEL_VERSION_P2",
    "P2_EXPECTED_PARAMS",
    "P2_FEATURE_DIM",
    "P2_OUTPUTS",
    "P2_READOUT",
    "P2_SPEC",
    "P2_TOKEN_HIDDEN",
    "ProcessDeepSets",
    "assert_p2_budget",
    "build_p2",
]
