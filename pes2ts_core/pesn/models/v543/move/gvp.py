"""``M2`` GVP Refiner MOVE decoder (plan §6.2 :527, §4.5 :242, §6.2 :534,
§5.2 :282–289, §6.4 :558–571, §11.1 ``move/gvp.py`` :1100; WP5).

Normative sources, quoted as implemented
---------------------------------------

* **§6.2 :527 (M2 row)** — "2 个 GVP block；标量 96 / 向量 8 通道；标量—向量
  交互".  :class:`GVPBlock` is the geometric-vector-perceptron block of
  Jing et al. (2020) in its minimal scalar→vector gate form: the scalar stream
  is a ``Linear`` + SiLU, the vector stream is multiplied by a scalar-produced
  ``[K, v_out, v_in]`` gate.  Two blocks: 120/8 → 96/8 → 96/8.
* **§6.2 :534 (M2 关键约束)** — "向量通道禁止 LayerNorm/ReLU；标量可用 SiLU；
  不使用裸 cross product".  The vector path is pure scalar-gated linear mixing
  (no norm/activation on vectors), the scalar path uses SiLU, and no cross
  product is used — so the candidate is genuinely **O(3)/polar**, not merely
  SO(3): a reflection acts on every vector channel as the same orthogonal
  matrix.
* **§4.5 :242 (zero residual)** — "M2 | 最后一个 scalar→vector 的输出投影 +
  外层 gate | `δu = 0`".  :attr:`GVPRefinerMoveDecoder.coeff` (the last
  scalar→vector projection) is zero-initialised, so the initial residual is
  exactly zero for any input.  The outer per-channel gate
  :attr:`GVPRefinerMoveDecoder.out_gate` starts at ONE, not at zero: with both
  factors zero the product is stationary w.r.t. BOTH (``dL/dc ∝ gate = 0`` and
  ``dL/dgate ∝ c = 0``), so the candidate could never leave ``δu = 0`` and the
  §7.2 training gate would never open.  One zero factor is sufficient for the
  zero-residual contract; this is reported as a plan ambiguity (§4.5 :242 lists
  "输出投影 + 外层 gate" without stating that both must be zero).
* **§6.2 :527 readout** — "向量通道直接更新": the residual is read out of the
  UPDATED vector channels, ``δu = Σ_j (g_j · c_j) · V_j`` with ``V`` the
  output of the second GVP block, never from a Cartesian MLP head.
* **§6.4 :558–571** — the widths give :data:`M2_N_TRAINABLE_PARAMS` = 35,600
  measured at construction (band 34,200–37,800, hard cap 40,000); the
  construction assertion is :func:`assert_budget`, so a width change that
  leaves the band is a typed refusal and the registry records the MEASURED
  count only.
* **§6.1 :519** — the only input is the read-only :class:`MoveContextV1`; no
  data, label, loss or checkpoint-selection surface exists.

Deterministic initialisation (WP1 ``fork_rng`` discipline)
----------------------------------------------------------

The build runs inside ``torch.random.fork_rng(devices=[])`` seeded with
:data:`M2_INIT_SEED_V543` (same pattern as ``v542/candidates/common.py::
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
M2_DECODER_ID: Final[str] = "M2"

#: Registered model version of the WP5 M2 implementation.
M2_MODEL_VERSION: Final[str] = "v543_m2_gvp_refiner_v1"

#: Construction seed of the frozen ``fork_rng`` build.
M2_INIT_SEED_V543: Final[int] = 17

#: Scalar input width: ``graph_s`` 64 + ``key_s`` 48 + ``‖key_v‖`` 8 = 120.
M2_SCALAR_IN_DIM: Final[int] = (
    GRAPH_S_DIM_V543 + KEY_S_DIM_V543 + KEY_V_CHANNELS_V543)
#: Scalar width of the two blocks (§6.2 :527 "标量 96").
M2_SCALAR_DIM: Final[int] = 96
#: Vector channels (§6.2 :527 "向量 8 通道").
M2_VECTOR_CHANNELS: Final[int] = KEY_V_CHANNELS_V543
#: Number of GVP blocks (§6.2 :527 "2 个 GVP block").
M2_N_BLOCKS: Final[int] = 2

#: Exact measured ``sum(p.numel())`` of the frozen widths (plan §6.4 :569):
#: block1 120·96+96 + 120·64  = 11,616 + 7,680 = 19,296
#: block2  96·96+96 +  96·64  =  9,312 + 6,144 = 15,456
#: head (96+8)·8+8             =    832 + 8    =    840
#: outer gate per-channel                     =      8
#: total                                       = 35,600
M2_N_TRAINABLE_PARAMS: Final[int] = (
    (M2_SCALAR_IN_DIM * M2_SCALAR_DIM + M2_SCALAR_DIM)
    + (M2_SCALAR_IN_DIM * M2_VECTOR_CHANNELS * M2_VECTOR_CHANNELS)
    + (M2_SCALAR_DIM * M2_SCALAR_DIM + M2_SCALAR_DIM)
    + (M2_SCALAR_DIM * M2_VECTOR_CHANNELS * M2_VECTOR_CHANNELS)
    + (M2_SCALAR_DIM + KEY_V_CHANNELS_V543) * M2_VECTOR_CHANNELS
    + M2_VECTOR_CHANNELS
    + M2_VECTOR_CHANNELS)


class M2Error(V543Error):
    """M2 refusal (context shape drift, out-of-band width)."""


def _check_move_context(ctx: object) -> MoveContextV1:
    """Refuse anything that is not the read-only MOVE view (plan §5.2.1)."""
    if not isinstance(ctx, MoveContextV1):
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_TYPE",
            "M2 takes a MoveContextV1 (plan §6.1 :490); got "
            f"{type(ctx).__name__}")
    key_s = ctx.key_s
    key_v = ctx.key_v
    if int(key_s.ndim) != 2 or int(key_s.shape[1]) != KEY_S_DIM_V543:
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_SHAPE",
            f"M2 needs key_s [K,{KEY_S_DIM_V543}], got {list(key_s.shape)}")
    expected = (int(key_s.shape[0]), KEY_V_CHANNELS_V543, 3)
    if tuple(key_v.shape) != expected:
        raise V543DecoderProtocolError(
            "V543_DECODER_CONTEXT_SHAPE",
            f"M2 needs key_v {list(expected)}, got {list(key_v.shape)}")
    return ctx


class GVPBlock(torch.nn.Module):
    """One GVP block: scalar ``Linear``+SiLU, scalar-gated vector mixing.

    ``h' = SiLU(W_h h + b_h)``; ``V' = (W_v h) · V`` with the gate reshaped to
    ``[v_out, v_in]``.  The vector path carries NO LayerNorm/ReLU and no cross
    product (plan §6.2 :534), so a reflection ``R`` maps ``V`` to ``R V`` and
    ``V'`` to ``R V'`` — the block is O(3)-covariant, not only SO(3).
    """

    def __init__(self, scalar_in: int, scalar_out: int,
                 vector_channels: int = M2_VECTOR_CHANNELS) -> None:
        super().__init__()
        self.scalar_in = int(scalar_in)
        self.scalar_out = int(scalar_out)
        self.vector_channels = int(vector_channels)
        self.scalar = torch.nn.Linear(self.scalar_in, self.scalar_out)
        # v_out*v_in scalars from h; no bias (a bias would add a constant
        # gate, i.e. a fixed linear vector mixing, not a learned one).
        self.vector_gate = torch.nn.Linear(
            self.scalar_in, self.vector_channels * self.vector_channels,
            bias=False)

    def forward(self, hidden: torch.Tensor,
                vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``hidden [K,s_in]``, ``vectors [K,v,3]`` → ``([K,s_out],[K,v,3])``."""
        hidden_out = F.silu(self.scalar(hidden))               # [K,s_out]
        n_keys = int(hidden.shape[0])
        gate = self.vector_gate(hidden).reshape(
            n_keys, self.vector_channels, self.vector_channels)
        # plan §6.2 :534: vectors are mixed by a scalar-produced gate only.
        vectors_out = torch.einsum("kij,kjd->kid", gate, vectors)
        return hidden_out, vectors_out


class GVPRefinerMoveDecoder(torch.nn.Module):
    """``M2``: 2 GVP blocks (scalars 96 / vectors 8) + polar readout.

    35,600 trainable parameters (measured at construction; plan §6.4 band
    34,200–37,800).  The residual is ``δu = Σ_j (g_j c_j) V_j`` with ``V`` the
    second block's vector stream, ``c`` a zero-initialised scalar→vector
    projection and ``g`` an outer per-channel gate initialised to one (§4.5
    :242; see the module docstring for why the gate is not also zero).
    """

    decoder_id: Final[str] = M2_DECODER_ID
    track: Final[str] = TRACK_MOVE
    model_version: Final[str] = M2_MODEL_VERSION

    def __init__(self, *, seed: int = M2_INIT_SEED_V543) -> None:
        super().__init__()
        self.init_seed = int(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.blocks = torch.nn.ModuleList([
                GVPBlock(M2_SCALAR_IN_DIM, M2_SCALAR_DIM),
                GVPBlock(M2_SCALAR_DIM, M2_SCALAR_DIM),
            ])
            self.coeff = torch.nn.Linear(
                M2_SCALAR_DIM + KEY_V_CHANNELS_V543, M2_VECTOR_CHANNELS)
            self.out_gate = torch.nn.Parameter(
                torch.ones(M2_VECTOR_CHANNELS))
            # plan §4.5 :242 — the final scalar→vector projection is zero at
            # construction, so δu ≡ 0 for ANY input.  The outer gate starts at
            # ONE (identity), not at zero: with BOTH factors zero the product
            # has zero gradient w.r.t. both (dL/dc ∝ gate = 0 and
            # dL/dgate ∝ c = 0), i.e. the candidate could never leave the
            # zero-residual point.  One zero factor is enough for δu = 0 and
            # keeps the first update alive — reported as a plan ambiguity (the
            # §4.5 :242 row lists "输出投影 + 外层 gate" without saying that
            # both must be zero).
            with torch.no_grad():
                self.coeff.weight.zero_()
                self.coeff.bias.zero_()
        measured = count_trainable_params(self)
        if measured != M2_N_TRAINABLE_PARAMS:
            raise M2Error(
                "V543_M2_PARAMS",
                f"M2 (2 GVP blocks, scalars {M2_SCALAR_DIM}, vectors "
                f"{M2_VECTOR_CHANNELS}) must give exactly "
                f"{M2_N_TRAINABLE_PARAMS} trainable parameters (plan §6.2 "
                f":527, §6.4 :569), measured {measured}")
        assert_budget(measured, decoder_id=self.decoder_id,
                      track=self.track)

    # -- protocol -----------------------------------------------------------
    def forward(self, ctx: MoveContextV1) -> torch.Tensor:
        """Return ``δu [K,3]`` read out of the updated vector channels."""
        ctx = _check_move_context(ctx)
        # Invariant scalar stream (§5.2 :282–283 invariance rows).
        hidden = torch.cat([
            ctx.graph_s[ctx.key_graph],
            ctx.key_s,
            ctx.key_v.norm(dim=-1),
        ], dim=-1)                                            # [K,120]
        vectors = ctx.key_v                                   # [K,8,3]
        for block in self.blocks:
            hidden, vectors = block(hidden, vectors)
        coeff = self.coeff(torch.cat(
            [hidden, vectors.norm(dim=-1)], dim=-1))          # [K,8]
        gated = coeff * self.out_gate
        # Polar readout: only the (updated) vector channels reach δu, so a
        # rotation/reflection R gives R·δu exactly (plan §6.2 :532 logic).
        return (gated.unsqueeze(-1) * vectors).sum(dim=1)     # [K,3]

    @property
    def n_trainable_params(self) -> int:
        return count_trainable_params(self)

    # -- audit --------------------------------------------------------------
    def describe(self) -> dict[str, object]:
        return {
            "decoder_id": self.decoder_id,
            "track": self.track,
            "model_version": self.model_version,
            "n_blocks": M2_N_BLOCKS,
            "scalar_in_dim": M2_SCALAR_IN_DIM,
            "scalar_dim": M2_SCALAR_DIM,
            "vector_channels": M2_VECTOR_CHANNELS,
            "n_trainable_params": self.n_trainable_params,
            "init_seed": self.init_seed,
            "readout": "delta_u = sum_j (gate_j * c_j) V_j (plan §6.2 :527)",
            "zero_init": "coeff.weight/bias = 0; out_gate = 1 (§4.5 :242)",
            "vector_ops": "scalar-gated linear mixing only (no norm/ReLU/"
                          "cross product, plan §6.2 :534)",
        }


def m2_spec(**kwargs: object) -> DecoderSpec:
    """The §6.1 registry record of M2 (13 fields; plan §6.1 :503–517)."""
    def factory() -> GVPRefinerMoveDecoder:
        return GVPRefinerMoveDecoder(**kwargs)  # type: ignore[arg-type]

    return DecoderSpec(
        decoder_id=M2_DECODER_ID,
        track=TRACK_MOVE,
        model_version=M2_MODEL_VERSION,
        factory=factory,
        equivariance="polar",
        stochastic=False,
        nfe=1,
        zero_init=True,
        status="proposed",
        acceptance="wp5_implemented_pending_S1_smoke",
    )


#: The process-wide M2 record (registered by the integrating agent / WP7).
M2_SPEC: Final[DecoderSpec] = m2_spec()


def register_m2(registry: object) -> DecoderSpec:
    """Register :data:`M2_SPEC` into ``registry`` (plan §11.2 ``register``)."""
    return registry.register(M2_SPEC)  # type: ignore[attr-defined]


__all__ = [
    "GVPBlock",
    "GVPRefinerMoveDecoder",
    "M2_DECODER_ID",
    "M2_INIT_SEED_V543",
    "M2_MODEL_VERSION",
    "M2_N_BLOCKS",
    "M2_N_TRAINABLE_PARAMS",
    "M2_SCALAR_DIM",
    "M2_SCALAR_IN_DIM",
    "M2_SPEC",
    "M2_VECTOR_CHANNELS",
    "M2Error",
    "m2_spec",
    "register_m2",
]
