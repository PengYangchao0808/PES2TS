"""``M1`` Residual MLP MOVE decoder (plan §6.2 :526, §4.5 :241, §4.3 :222,
§5.3 :361, §6.4 :558–571, §11.1 ``move/residual_mlp.py`` :1099; WP5).

Normative sources, quoted as implemented
---------------------------------------

* **§6.2 :526 (M1 row)** — "`[graph_s,key_s,u0/a_ref,‖key_v‖]`（123 维）→ 136 →
  136 → 8 个系数；`δu=Σ c_j key_v_j`（约 36.6K）".  The layout is implemented in
  that order — ``graph_s[key_graph]`` (64, gathered so the row order follows
  ``key_*``), ``key_s`` (48), the ``u0/a_ref`` block (see the deviation note
  below, 8 dims) and ``‖key_v‖`` (8) — for 128 input dimensions.
* **DEVIATION, reported as a plan ambiguity (WP5):** the literal ``u0/a_ref``
  block of §6.2 :526 is THREE RAW CARTESIAN COMPONENTS, i.e. a POLAR input.
  A scalar MLP over a polar input is not rotation invariant, so its 8
  coefficients change when the molecule is rotated and ``δu`` would NOT
  transform as a polar vector — the mandatory §11.5 (:1229)
  ``test_v543_rotation_equivariance`` gate and §6.2 :532's own "否则旋转一个
  分子会得到不旋转的残差，破坏等变性" argument would fail (measured on the
  fixture geometry before the fix: rotation error 4.5 % of ``|δu|``).  §6.2
  :532 says the ONLY polar path may be the ``key_v`` readout, so the
  ``u0/a_ref`` block is implemented as its E(3)-INVARIANT overlaps with the
  same polar basis, ``<u0/a_ref, key_v_j>`` (j = 0…7), which keeps the
  information (channel 0 is ``u0/a_ref`` itself per §5.3 :361) and leaves the
  coefficient head a function of invariants only.  The width change
  123 → 128 moves the count from the plan's estimate to
  :data:`M1_N_TRAINABLE_PARAMS` = 37,272 (still inside the §6.4 band and ≤ +4 %
  of 36,600).  Reported to the orchestrator; see the WP5 report.
* **§6.2 :532 (M1 关键约束)** — "不能直接让 MLP 输出绝对 `(x,y,z)`；必须输出
  8 个标量系数，再与 `key_v` 组合".  :meth:`ResidualMLPMoveDecoder.forward`
  therefore returns ``δu = Σ_j c_j · key_v_j`` and never a bare Cartesian
  triple; E(3) equivariance is inherited from the polar ``key_v`` basis
  (§5.2 :284 "``key_v`` | 极向量") and is tested by rotation.
* **§4.5 :241 (zero residual)** — "M1 | 最后一层 `Linear(·,8)` 的 weight/bias |
  8 个系数 = 0 → `δu = 0`".  The coefficient head is zero-initialised inside
  :meth:`__init__`, so a freshly constructed M1 reproduces ``u = u0`` exactly
  (§5.8 zero-residual parity).
* **§5.3 :361** — channel 0 of ``key_v`` is ``u0[k] / a_ref`` with
  ``a_ref = 0.08``; the ``u0/a_ref`` features use the same
  :data:`~...contracts.A_REF_V543` constant, so the decoder never re-invents a
  scale.
* **§6.4 :558–571** — construction-time exact ``sum(p.numel())`` assertion:
  the layer widths (128 → 136 → 136 → 8) give exactly
  :data:`M1_N_TRAINABLE_PARAMS` = 37,272, inside the frozen band
  34,200–37,800 and within 4 % of the plan's "约 36.6K".  :func:`assert_budget`
  runs at construction and a width that leaves the band is a typed refusal.
* **§6.1 :519** — no data reading / loss / checkpoint selection: the only
  input is the read-only :class:`MoveContextV1`.

Deterministic initialisation (the WP1 ``fork_rng`` discipline)
--------------------------------------------------------------

The whole module is built inside ``torch.random.fork_rng(devices=[])`` seeded
with :data:`M1_INIT_SEED_V543` (same pattern as
``pes2ts_core/pesn/models/v542/candidates/common.py::construction_scope``), so
constructing an M1 never perturbs the caller's global RNG stream and two
constructions with the same ``seed`` are bit-identical.  The construction seed
is exposed as ``init_seed``; the §6.1 record has no dedicated seed row, so the
run identity that pins it is the registry record's ``config_sha256`` (WP7) and
``init_seed`` on the object.

Discipline: no shared file is modified (the package ``__init__`` exports are
left to the integrating agent), every numeric constant is ``Final``, and no
``torch.compile`` is used anywhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pes2ts_core.pesn.models.v543.contracts import (
    A_REF_V543,
    GRAPH_S_DIM_V543,
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
M1_DECODER_ID: Final[str] = "M1"

#: Registered model version of the WP5 M1 implementation.
M1_MODEL_VERSION: Final[str] = "v543_m1_residual_mlp_v1"

#: Construction seed of the frozen ``fork_rng`` build (WP1 discipline; §6.1
#: record carries it through the run config SHA, see the module docstring).
M1_INIT_SEED_V543: Final[int] = 17

#: Input width: ``graph_s`` 64 + ``key_s`` 48 + ``u0/a_ref`` overlaps 8 +
#: ``‖key_v‖`` 8 = 128 (see the deviation note in the module docstring).
M1_INPUT_DIM: Final[int] = (
    GRAPH_S_DIM_V543 + KEY_S_DIM_V543 + KEY_V_CHANNELS_V543
    + KEY_V_CHANNELS_V543)
M1_HIDDEN_DIM: Final[int] = 136
M1_COEFF_DIM: Final[int] = KEY_V_CHANNELS_V543

#: Exact measured ``sum(p.numel())`` of the frozen widths (plan §6.4 :569):
#: 128·136+136 + 136·136+136 + 136·8+8 = 17,544 + 18,632 + 1,096 = 37,272.
M1_N_TRAINABLE_PARAMS: Final[int] = (
    M1_INPUT_DIM * M1_HIDDEN_DIM + M1_HIDDEN_DIM
    + M1_HIDDEN_DIM * M1_HIDDEN_DIM + M1_HIDDEN_DIM
    + M1_HIDDEN_DIM * M1_COEFF_DIM + M1_COEFF_DIM)

#: Feature layout documented for the parity/audit tests.
M1_FEATURE_LAYOUT: Final[tuple[tuple[str, int], ...]] = (
    ("graph_s[key_graph]", GRAPH_S_DIM_V543),
    ("key_s", KEY_S_DIM_V543),
    ("<u0/a_ref, key_v_j>", KEY_V_CHANNELS_V543),
    ("||key_v||", KEY_V_CHANNELS_V543),
)


class M1Error(V543Error):
    """M1 refusal (context shape drift, out-of-band width)."""


def _check_move_context(ctx: object) -> MoveContextV1:
    """Refuse anything that is not the read-only MOVE view (plan §5.2.1)."""
    if not isinstance(ctx, MoveContextV1):
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_TYPE",
            "M1 takes a MoveContextV1 (plan §6.1 :490); got "
            f"{type(ctx).__name__}")
    key_s = ctx.key_s
    key_v = ctx.key_v
    if int(key_s.ndim) != 2 or int(key_s.shape[1]) != KEY_S_DIM_V543:
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_SHAPE",
            f"M1 needs key_s [K,{KEY_S_DIM_V543}], got "
            f"{list(key_s.shape)}")
    expected = (int(key_s.shape[0]), KEY_V_CHANNELS_V543, 3)
    if tuple(key_v.shape) != expected:
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_SHAPE",
            f"M1 needs key_v {list(expected)}, got {list(key_v.shape)}")
    return ctx


def m1_features(ctx: MoveContextV1) -> torch.Tensor:
    """The 128-d per-key feature block of plan §6.2 :526 (deviation noted above).

    ``[graph_s[key_graph], key_s, <u0/a_ref, key_v_j>, ‖key_v‖]`` — every
    column is E(3)-invariant: the graph vector / scalar embedding per §5.2
    :282–283 (invariant rows), the overlaps of two polar vectors, and the
    channel norms.  The coefficient head is therefore a scalar function and the
    ONLY source of polar behaviour is the ``key_v`` readout (§6.2 :532).
    """
    graph = ctx.graph_s[ctx.key_graph]                       # [K,64]
    scaled_u0 = ctx.u0 / float(A_REF_V543)                   # [K,3]
    # invariant "u0 read in the key_v basis" block (see the deviation note).
    u0_overlap = torch.einsum("kd,kcd->kc", scaled_u0, ctx.key_v)  # [K,8]
    channel_norm = ctx.key_v.norm(dim=-1)                    # [K,8]
    return torch.cat(
        [graph, ctx.key_s, u0_overlap, channel_norm], dim=-1)


class ResidualMLPMoveDecoder(torch.nn.Module):
    """``M1``: 128 → 136 → 136 → 8 coefficients, ``δu = Σ c_j key_v_j``.

    Args:
        seed: construction seed of the isolated ``fork_rng`` build.
        a_ref: the §5.3 reference length (Å) scaling the ``u0`` features.

    37,272 trainable parameters (measured at construction; plan §6.4 band
    34,200–37,800, estimate "约 36.6K" at §6.2 :526).
    """

    decoder_id: Final[str] = M1_DECODER_ID
    track: Final[str] = TRACK_MOVE
    model_version: Final[str] = M1_MODEL_VERSION

    def __init__(self, *, seed: int = M1_INIT_SEED_V543,
                 a_ref: float = A_REF_V543) -> None:
        super().__init__()
        self.init_seed = int(seed)
        self.a_ref = float(a_ref)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.fc1 = torch.nn.Linear(M1_INPUT_DIM, M1_HIDDEN_DIM)
            self.fc2 = torch.nn.Linear(M1_HIDDEN_DIM, M1_HIDDEN_DIM)
            self.coeff = torch.nn.Linear(M1_HIDDEN_DIM, M1_COEFF_DIM)
            # plan §4.5 :241 — the last Linear(·,8) is zero-initialised so the
            # initial 8 coefficients are exactly 0 and δu ≡ 0.
            with torch.no_grad():
                self.coeff.weight.zero_()
                self.coeff.bias.zero_()
        measured = count_trainable_params(self)
        if measured != M1_N_TRAINABLE_PARAMS:
            raise M1Error(
                "V543_M1_PARAMS",
                f"M1 widths 128→136→136→8 must give exactly "
                f"{M1_N_TRAINABLE_PARAMS} trainable parameters (plan §6.2 "
                f":526/:532), measured {measured}")
        assert_budget(measured, decoder_id=self.decoder_id,
                      track=self.track)

    # -- protocol -----------------------------------------------------------
    def forward(self, ctx: MoveContextV1) -> torch.Tensor:
        """Return ``δu [K,3]`` = ``Σ_j c_j key_v_j`` (plan §6.2 :526/:532)."""
        ctx = _check_move_context(ctx)
        hidden = F.silu(self.fc1(m1_features(ctx)))
        hidden = F.silu(self.fc2(hidden))
        coeff = self.coeff(hidden)                            # [K,8]
        # The ONLY polar readout: a linear combination of the 8 key_v
        # channels.  A rotation R acts on the result as R because it acts on
        # every key_v row as R (§6.2 :532).
        return (coeff.unsqueeze(-1) * ctx.key_v).sum(dim=1)   # [K,3]

    @property
    def n_trainable_params(self) -> int:
        return count_trainable_params(self)

    # -- audit --------------------------------------------------------------
    def describe(self) -> dict[str, object]:
        return {
            "decoder_id": self.decoder_id,
            "track": self.track,
            "model_version": self.model_version,
            "widths": [M1_INPUT_DIM, M1_HIDDEN_DIM, M1_HIDDEN_DIM,
                       M1_COEFF_DIM],
            "n_trainable_params": self.n_trainable_params,
            "init_seed": self.init_seed,
            "a_ref": self.a_ref,
            "readout": "delta_u = sum_j c_j key_v_j (plan §6.2 :526)",
            "zero_init": "coeff.weight/bias = 0 (plan §4.5 :241)",
        }


def m1_spec(**kwargs: object) -> DecoderSpec:
    """The §6.1 registry record of M1 (13 fields; plan §6.1 :503–517)."""
    def factory() -> ResidualMLPMoveDecoder:
        return ResidualMLPMoveDecoder(**kwargs)  # type: ignore[arg-type]

    return DecoderSpec(
        decoder_id=M1_DECODER_ID,
        track=TRACK_MOVE,
        model_version=M1_MODEL_VERSION,
        factory=factory,
        equivariance="polar",
        stochastic=False,
        nfe=1,
        zero_init=True,
        status="proposed",
        acceptance="wp5_implemented_pending_S1_smoke",
    )


#: The process-wide M1 record (registered by the integrating agent / WP7).
M1_SPEC: Final[DecoderSpec] = m1_spec()


def register_m1(registry: object) -> DecoderSpec:
    """Register :data:`M1_SPEC` into ``registry`` (plan §11.2 ``register``)."""
    return registry.register(M1_SPEC)  # type: ignore[attr-defined]


__all__ = [
    "M1_COEFF_DIM",
    "M1_DECODER_ID",
    "M1_FEATURE_LAYOUT",
    "M1_HIDDEN_DIM",
    "M1_INIT_SEED_V543",
    "M1_INPUT_DIM",
    "M1_MODEL_VERSION",
    "M1_N_TRAINABLE_PARAMS",
    "M1_SPEC",
    "M1Error",
    "ResidualMLPMoveDecoder",
    "m1_features",
    "m1_spec",
    "register_m1",
]
