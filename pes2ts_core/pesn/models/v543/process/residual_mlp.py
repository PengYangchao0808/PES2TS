"""V5.4.3 WP6 — ``P1`` PROCESS candidate: the Residual MLP (plan §6.3 :543,
§6.4 :557–571, §4.5 :246, §11.1 :1105 ``process/residual_mlp.py``).

Normative sources, quoted as implemented
---------------------------------------
* **§6.3 :543** — "``P1`` | Residual MLP | ``graph_s[64]`` → 160 → 160 → 2
  （约 36.5K）| 36K ±5% | 正式".  :class:`ProcessResidualMLP` is exactly that
  stack and nothing else: the INPUT is the whole ``graph_s`` row of one graph
  and there is no aggregation of ``key_s``/``pair_s`` — that is precisely what
  separates P1 (global-only) from P2/P3/P4 (local tokens) in hypothesis H-P1
  (plan §3.2 :166 "局部 key/pair token 比仅全局 ``z`` 更能判断端点与进度").
  Adding a token summary here would destroy the H-P1 contrast.
* **§4.2 :211 / §4.3 :223–224** — the PROCESS residual is ``q = tanh(q0_raw +
  δq)`` and ``e = e0 + δe``: the decoder emits ``δq_raw`` BEFORE the tanh (the
  tanh stays in ``V543Model._decoder_flow``) and an endpoint LOGIT residual
  ("训练时只用 logit，禁止双重 sigmoid").  :class:`ProcessResidual` is that
  pair; this module never applies tanh or sigmoid itself.
* **§4.5 :246** — "P1–P5 | 最后一层 ``Linear(·,2)`` 的 weight/bias | ``δq = 0``、
  ``δe = 0``"; :func:`zero_init_last_layer` zeroes both, so a freshly built P1
  reproduces V542 A0 bit-for-bit (plan §4.4 boundary 3 :233, §5.8).
* **§6.4 :563–571** — target 36,000, band 34,200–37,800, hard cap 40,000,
  "构造时 ``sum(p.numel())`` 精确断言", "禁止用 dead weights / 无用 bias 凑参数".
  :func:`assert_p1_budget` runs in ``__init__`` on the measured count and the
  measured value is 36,482 (plan estimate "约 36.5K"), inside the band with no
  padding layer.
* **§6.1 :495–498** — one argument, ``ProcessContextV1``; a MOVE context is a
  typed refusal (plan §5.2.1 hard isolation).

Measured parameter count (``sum(p.numel())``, plan §6.4 :569)
------------------------------------------------------------
``64·160 + 160`` = 10,400; ``160·160 + 160`` = 25,760; ``160·2 + 2`` = 322
→ **36,482**.  The widths are the plan's own design start (§6.3 note :550) and
land in the frozen band without adjustment.

Discipline: no data reading / loss / checkpoint selection (plan §6.1 :519), no
``torch.compile`` (plan §11.3), every constant ``Final``."""

from __future__ import annotations

from typing import Any, Final

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.models.v543.contracts import (
    GRAPH_S_DIM_V543,
    ProcessContextV1,
    ProcessResidual,
)
from pes2ts_core.pesn.models.v543.process.tokens import (
    PROCESS_INIT_SEED_V543,
    assert_candidate_budget,
    assert_process_context,
    deterministic_init,
    measured_params,
    zero_init_last_layer,
)
from pes2ts_core.pesn.models.v543.registry import (
    TRACK_PROCESS,
    DecoderSpec,
)

#: ``P1`` id (plan §6.3 :543).
DECODER_ID_P1: Final[str] = "P1"
#: Registered model-layer version of this candidate.
MODEL_VERSION_P1: Final[str] = "v543_p1_residual_mlp_v1"
#: Plan §6.3 :543 widths "``graph_s[64]`` → 160 → 160 → 2".
P1_HIDDEN: Final[int] = 160
#: Number of residual scalars: ``δq_raw`` and ``δe`` (§6.3 :543 "→ 2").
P1_OUTPUTS: Final[int] = 2
#: Measured parameter count of the frozen structure (§6.4 :569).
P1_EXPECTED_PARAMS: Final[int] = (
    GRAPH_S_DIM_V543 * P1_HIDDEN + P1_HIDDEN
    + P1_HIDDEN * P1_HIDDEN + P1_HIDDEN
    + P1_HIDDEN * P1_OUTPUTS + P1_OUTPUTS)


class ProcessResidualMLP(torch.nn.Module):
    """``P1``: ``graph_s`` MLP emitting ``(δq_raw, δe)`` (plan §6.3 :543).

    Args:
        hidden: hidden width (default 160, the plan's design start).
        seed: initialisation seed; the parameters are a pure function of it
            (fork_rng discipline, plan §7.2 :667).

    A freshly constructed instance emits EXACTLY zero (§4.5), so
    ``q = tanh(q0_raw)`` and ``endpoint_logit = e0`` before any training step.
    """

    decoder_id: Final[str] = DECODER_ID_P1
    track: Final[str] = TRACK_PROCESS
    model_version: Final[str] = MODEL_VERSION_P1
    equivariance: Final[str] = "invariant"
    stochastic: Final[bool] = False
    nfe: Final[int] = 1
    zero_init: Final[bool] = True

    def __init__(self, *, hidden: int = P1_HIDDEN,
                 seed: int = PROCESS_INIT_SEED_V543) -> None:
        super().__init__()
        if int(hidden) < 1:
            raise ValueError(f"P1 hidden must be >= 1, got {hidden}")
        self.init_seed: Final[int] = int(seed)
        self.hidden: Final[int] = int(hidden)
        with deterministic_init(self.init_seed):
            self.fc1 = torch.nn.Linear(GRAPH_S_DIM_V543, self.hidden)
            self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
            self.head = torch.nn.Linear(self.hidden, P1_OUTPUTS)
        zero_init_last_layer(self.head)
        assert_p1_budget(measured_params(self))

    # -- protocol -----------------------------------------------------------
    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:
        """``graph_s [G,64]`` → ``ProcessResidual`` (plan §6.1 :496)."""
        context = assert_process_context(ctx, decoder_id=self.decoder_id)
        hidden = F.silu(self.fc1(context.graph_s))
        hidden = F.silu(self.fc2(hidden))
        out = self.head(hidden)
        return ProcessResidual(delta_q_raw=out[:, 0], delta_e=out[:, 1])

    # -- bookkeeping --------------------------------------------------------
    @property
    def n_trainable_params(self) -> int:
        """Measured ``sum(p.numel())`` (plan §6.1 :509 / §6.4 :569)."""
        return measured_params(self)

    def describe(self) -> dict[str, Any]:
        """JSON-able description used by the WP6 report / registry record."""
        return {
            "decoder_id": self.decoder_id,
            "track": self.track,
            "model_version": self.model_version,
            "structure": "graph_s[64] -> 160 -> 160 -> 2 (plan §6.3 :543)",
            "hidden": int(self.hidden),
            "init_seed": int(self.init_seed),
            "n_trainable_params": self.n_trainable_params,
            "zero_init": True,
        }


def assert_p1_budget(n_trainable_params: int) -> int:
    """Construction-time §6.4 band assertion of ``P1``."""
    return assert_candidate_budget(n_trainable_params, decoder_id=DECODER_ID_P1)


def build_p1(**kwargs: Any) -> ProcessResidualMLP:
    """Factory used by the registry (no extra argument is a data handle)."""
    return ProcessResidualMLP(**kwargs)


#: The §6.1 registry spec of ``P1`` (13-field record + factory).
P1_SPEC: Final[DecoderSpec] = DecoderSpec(
    decoder_id=DECODER_ID_P1,
    track=TRACK_PROCESS,
    model_version=MODEL_VERSION_P1,
    factory=build_p1,
    equivariance="invariant",
    stochastic=False,
    nfe=1,
    zero_init=True,
    status="proposed",
    acceptance=(
        "wp6: constructed, sum(p.numel()) asserted in band, zero-residual "
        "init, pair/token permutation invariant; S1 smoke pending"),
)


__all__ = [
    "DECODER_ID_P1",
    "MODEL_VERSION_P1",
    "P1_EXPECTED_PARAMS",
    "P1_HIDDEN",
    "P1_OUTPUTS",
    "P1_SPEC",
    "ProcessResidualMLP",
    "assert_p1_budget",
    "build_p1",
]
