"""V5.4.3 frozen contracts — WP1 slice: encoder constants, ``CoarsePacket``,
and the V543 typed-refusal taxonomy (plan §4/§5/§11; ADR-0034).

Scope of THIS file in WP1 (WP1 = A0 migration, plan §12.1):

* the V543 NUMERIC / SCHEMA constants the CoarseEncoder and every parity
  assertion are pinned to.  None of them is invented here: each one is
  either re-expressed from the V5.4.2 frozen interface
  (:mod:`pes2ts_core.pesn.models.v542.interfaces`) or read verbatim from
  the plan:

  - ``ENCODER_Z_DIM_V543 = 64`` / ``KEY_S_DIM_V543 = 48``
    (plan §5.2 field table: ``graph_s [G,64]`` / ``key_s [K,48]``);
  - ``A_REF_V543 = 0.08`` (plan §5.3: "记 ``x = pos``, ``a_ref = 0.08``";
    identical to the V5.4.2 frozen ``A_REF_V542``);
  - encoder/decoder/process/joint parameter counts 55,392 / 49 / 2,178 /
    57,619 (plan §6.4 "最终整模（目标）57,619 + 36K + 36K = 129,619", so
    the frozen encoder block IS the V5.4.2 A0 joint block);
  - ``HOOK_FIRES_PER_FORWARD = 1`` (plan §5.8 implementation note:
    "hook 必须在 ``finally`` 中移除，并断言恰好触发一次");
  - ``ENCODER_FORWARD_CALLS_PER_VIEW = 1`` (plan §4.4 冻结边界 1:
    "**Encoder 只运行一次**"; §7.1 S0-A: "Encoder forward 每次当前构型
    恰好 1 次");
  - ``PARITY_ATOL_CPU = 0.0`` / ``PARITY_ATOL_GPU = 1e-6`` (plan §5.8:
    "三 seed、≥8 个真实 inner-val batch、CPU 逐位、GPU ≤1e-6"; §7.1
    S0-A: "CPU bitwise；GPU ≤1e-6" — the tolerance is never loosened,
    "不得用「近似相等」放行");
  - ``DECODER_CONTEXT_SCHEMA_V543 = "decoder_context_v1"`` and
    ``PRECISION_VERSION_V543 = "fp32_v1"`` (plan §5.2 last three table
    rows: the literal schema/precision strings the Bridge must emit);
* :class:`CoarsePacket` — the FROZEN encoder-output contract of WP1.  It
  carries exactly the six quantities plan §11.2 assigns to
  ``coarse_encoder.py`` (``z/h/edge_msg/u0/q0_raw/e0``) and NOTHING else:
  no ``pos``, no supervision key, no ``DecoderContextV1`` field (the
  Bridge/context layer is WP2 and lives in ``latent_bridge.py``).
  ``e0`` is the raw endpoint LOGIT of plan §4.3 ("训练时只用 logit，
  禁止双重 sigmoid") — the packet never sigmoids it;
* the typed refusals (``code`` + ``detail``, repo convention as in
  :class:`~pes2ts_core.pesn.models.v542.interfaces.V542InterfaceError`).

Not in WP1 (explicitly deferred, do not pre-empt):
:class:`~pes2ts_core.pesn.models.v543.latent_bridge.DecoderContextV1`,
``MoveContextV1``/``ProcessContextV1``/``ProcessResidual`` (plan §5.2.1,
§11.2 -> WP2), the refiner registry (WP4) and ``V543Model`` (WP4).  This
module is the WP1 surface those files will import.

> WP2 NOTE (additive, this same file): :class:`DecoderContextV1`,
> :class:`MoveContextV1`, :class:`ProcessContextV1`,
> :class:`ProcessResidual`, :func:`assert_no_supervision` and
> :func:`validate_decoder_context` were APPENDED below the frozen WP1
> surface (see the "WP2 — DecoderContextV1" banner).  No WP1 name above is
> re-interpreted or moved.

Discipline: no V542 / V53MP file is modified (plan §11.3 禁止), no
``torch.compile`` anywhere (plan §11.3), and every constant below is
``Final`` (frozen at import)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import SUPERVISION_KEYS_V3
from pes2ts_core.pesn.models.key_atom_egnn_v3 import Z_DIM_V3
from pes2ts_core.pesn.models.v542.interfaces import (
    A0_ACTIVE_PARAMS_V542,
    A0_JOINT_PARAMS_V542,
    A_REF_V542,
    EDGE_MSG_DIM_V542,
    HIDDEN_V542,
    N_DECODER_PARAMS_V542,
    N_PROCESS_PARAMS_V542,
)

#: Registered V5.4.3 model-layer version (bumped only with a new ADR).
V543_SCHEMA_VERSION: Final[str] = "pesn_key_atom_v543_model_v1"
#: Registered WP1 encoder-adapter version.
V543_ENCODER_VERSION: Final[str] = "pesn_key_atom_v543_a0_coarse_encoder_v1"
#: Schema id of the encoder-output packet (plan §11.2 ``CoarsePacket``).
COARSE_PACKET_SCHEMA: Final[str] = "pesn_v543_coarse_packet_v1"
#: Schema id of the encoder strict-load audit record.
ENCODER_LOAD_AUDIT_SCHEMA: Final[str] = "pesn_v543_encoder_load_audit_v1"

#: A0 checkpoint payload layouts the encoder can strict-load (plan §7.1 S0-A
#: + §11.3).  TWO real forms exist in the campaign, so the loader must
#: discriminate them EXPLICITLY rather than assume one:
#:
#: * ``legacy_v53mp_joint_43`` — the V541/V53-MP ``KeyAtomV53MPModel`` joint
#:   state dict (37 backbone + 6 process-head = 43 keys, only
#:   ``backbone.w_v.*``).  Loaded through the FROZEN ``a0_mapping_table()``
#:   (identity + the declared ``w_v`` dual-write, 43 → 45);
#: * ``v542_a0_joint_45`` — the V5.4.2 ``V542Model`` joint state dict
#:   (45 keys: the dual-write is ALREADY materialized, ``backbone.w_v.*``
#:   AND ``move_decoder.w_v.*`` both present — e.g. the S0-A assets
#:   ``outputs/pesn/v542_backbone_50k/runs/V542_A0_S*/checkpoint_best.pt``,
#:   schema ``pesn_v542_train_v1``).  Loaded through the explicit IDENTITY
#:   mapping 45 → 45, with the FULL frozen strict audit
#:   (missing/unexpected/shape/dtype/``torch.equal``) still in force.
#:
#: Any other key set is a typed ``V543_CKPT_LAYOUT`` refusal — never a
#: guess, never ``strict=False``.
PAYLOAD_LAYOUT_LEGACY_43: Final[str] = "legacy_v53mp_joint_43"
PAYLOAD_LAYOUT_V542_A0_45: Final[str] = "v542_a0_joint_45"
#: Every supported payload layout.
PAYLOAD_LAYOUTS_V543: Final[tuple[str, ...]] = (
    PAYLOAD_LAYOUT_LEGACY_43, PAYLOAD_LAYOUT_V542_A0_45)
#: Load-audit ``mapping_mode`` labels (which mapping table was used).
MAPPING_MODE_FROZEN: Final[str] = "frozen_a0_dual_write"
MAPPING_MODE_IDENTITY: Final[str] = "identity_45"

#: Context schema string the WP2 Bridge must emit (plan §5.2 table row).
DECODER_CONTEXT_SCHEMA_V543: Final[str] = "decoder_context_v1"
#: Precision label of the whole V5.4.3 campaign (plan §5.2 table row).
PRECISION_VERSION_V543: Final[str] = "fp32_v1"

#: ``graph_s`` width — the A0 ``z_head`` output width 64 (plan §5.2).
ENCODER_Z_DIM_V543: Final[int] = Z_DIM_V3
#: Alias of :data:`ENCODER_Z_DIM_V543` under the ``DecoderContextV1`` name.
GRAPH_S_DIM_V543: Final[int] = ENCODER_Z_DIM_V543
#: ``key_s`` width — last-layer node ``h`` width 48 (plan §5.2).
KEY_S_DIM_V543: Final[int] = HIDDEN_V542
#: ``edge_msg`` width — the last-layer scalar message width 48.
EDGE_MSG_DIM_V543: Final[int] = EDGE_MSG_DIM_V542
#: Node-hidden width (same 48; named for the encoder's own docstrings).
HIDDEN_V543: Final[int] = HIDDEN_V542
#: MOVE reference step (Å) — frozen, non-learnable (plan §5.3).
A_REF_V543: Final[float] = A_REF_V542

#: Encoder block parameter counts (frozen; plan §6.4 final-model target).
ENCODER_ACTIVE_PARAMS_V543: Final[int] = A0_ACTIVE_PARAMS_V542  # 55,392
N_DECODER_PARAMS_V543: Final[int] = N_DECODER_PARAMS_V542      # 49
N_PROCESS_PARAMS_V543: Final[int] = N_PROCESS_PARAMS_V542      # 2,178
ENCODER_JOINT_PARAMS_V543: Final[int] = A0_JOINT_PARAMS_V542   # 57,619

#: Number of encoder forwards allowed per current configuration (plan §4.4).
ENCODER_FORWARD_CALLS_PER_VIEW: Final[int] = 1
#: Forward hooks must fire EXACTLY this many times per encoder forward.
HOOK_FIRES_PER_FORWARD: Final[int] = 1

#: Parity tolerances of plan §5.8 (never loosened, never "approximate").
PARITY_ATOL_CPU: Final[float] = 0.0
PARITY_ATOL_GPU: Final[float] = 1e-6

#: The three hook capture points of plan §11.1 (``coarse_encoder.py``:
#: "A0 strict load + 三个 hook + u0/q0_raw/e0") + §11.5
#: ``test_v543_hook_fires_once``.
HOOK_LABEL_EDGE_MSG: Final[str] = "message_mlp_last"
HOOK_LABEL_NODE_H: Final[str] = "norm_h_last"
HOOK_LABEL_Q0_RAW: Final[str] = "progress_fc2"
HOOK_LABELS: Final[tuple[str, ...]] = (
    HOOK_LABEL_EDGE_MSG, HOOK_LABEL_NODE_H, HOOK_LABEL_Q0_RAW)

#: Exact module paths of the three capture points (audit + tests).
HOOK_MODULE_PATHS: Final[tuple[tuple[str, str], ...]] = (
    (HOOK_LABEL_EDGE_MSG, "backbone.message_mlp[-1]"),
    (HOOK_LABEL_NODE_H, "backbone.norm_h[-1]"),
    (HOOK_LABEL_Q0_RAW, "process_head.progress_head.fc2"),
)

#: The six functional fields of :class:`CoarsePacket` (frozen surface).
COARSE_PACKET_KEYS: Final[tuple[str, ...]] = (
    "z", "edge_msg", "node_h", "u0", "q0_raw", "e0")

#: Supervision keys that may NEVER reach the encoder (plan §5.6; the same
#: frozen tuple the legacy contract uses).
SUPERVISION_KEYS_V543: Final[tuple[str, ...]] = SUPERVISION_KEYS_V3


# ---------------------------------------------------------------------------
# error taxonomy (typed refusals; every class carries ``code``/``detail``)
# ---------------------------------------------------------------------------
class V543Error(RuntimeError):
    """Base typed refusal of the V5.4.3 model layer."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class V543ContractError(V543Error):
    """Frozen-contract violation (field drift, schema/shape mismatch)."""


class V543InitError(V543Error):
    """Construction-time wiring violation (shared-module mismatch etc.)."""


class V543HookError(V543Error):
    """A forward hook fired the wrong number of times or leaked."""


class V543LoadError(V543Error):
    """Encoder checkpoint load refusal (beyond the V5.4.2 strict audit)."""


class V543SupervisionLeakError(V543Error):
    """A ``SUPERVISION_KEYS_V3`` member reached the encoder forward."""


class V543ParityError(V543Error):
    """A plan §5.8 parity equality did not hold within its tolerance."""


class V543ParamBudgetError(V543Error):
    """Parameter count violates a frozen V5.4.3 invariant."""


# ---------------------------------------------------------------------------
# the encoder-output contract (plan §11.2 ``CoarsePacket``)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CoarsePacket:
    """Frozen, read-only output of one :class:`CoarseEncoder` forward.

    Fields (plan §5.2 来源 column + §11.2):

    * ``z: [G, 64]`` — the A0 ``z_head`` output (tanh-saturated), graph
      order = ``batch["graph_index"]``.  WP2 exposes it as ``graph_s``;
    * ``edge_msg: [E, 48]`` — last-layer scalar message per DIRECTED input
      edge, row-for-row aligned with ``batch["edge_index"]`` columns,
      autograd graph live (never detached, plan §5.8);
    * ``node_h: [N, 48]`` — the last-layer node state ``h`` captured from
      ``backbone.norm_h[-1]`` (plan §5.8: "增加 ``norm_h[-1]`` 的 forward
      hook 捕获最后一层 ``h``").  WP2 selects ``node_h[key_index]`` as
      ``key_s``; the packet itself keeps ALL node rows so the capture is
      provably the backbone's own tensor;
    * ``u0: [K, 3]`` — the A0 MOVE初猜 in Å, produced by the shared
      ``DirectMoveDecoder`` from the same five tensors as V542
      (plan §5.8: "不得重排 ``edge_index`` 或改变 ``key_index`` 行序");
    * ``q0_raw: [G]`` — the ``progress_head.fc2`` output BEFORE tanh,
      captured by hook (plan §5.2 ``q0_raw`` row / §5.8);
    * ``e0: [G]`` — the endpoint LOGIT ``endpoint_head.fc(z)``; never
      sigmoid'ed here (plan §4.3: "训练时只用 logit，禁止双重 sigmoid").

    ``q0_raw`` and ``e0`` are the two PROCESS scalars; ``tanh(q0_raw)``
    is proven equal to the V542 ``q`` by the encoder's own typed check
    (plan §5.8 equality 4) and re-proven in
    ``tests/pesn/test_v543_q0_raw_tanh_identity.py``.

    The packet deliberately has NO ``pos`` / ``batch`` / supervision key
    and NO ``DecoderContextV1`` field — those arrive with the WP2 Bridge.
    """

    z: torch.Tensor
    edge_msg: torch.Tensor
    node_h: torch.Tensor
    u0: torch.Tensor
    q0_raw: torch.Tensor
    e0: torch.Tensor

    @property
    def n_graphs(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_keys(self) -> int:
        return int(self.u0.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.node_h.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_msg.shape[0])

    @property
    def q0(self) -> torch.Tensor:
        """``tanh(q0_raw)`` — DERIVED, never stored (plan §5.8 equality 4).

        Recomputed on each access (deterministic elementwise ``tanh``), so
        the packet stays a pure capture of the encoder's tensors and can
        never carry a second, diverging copy of the progress scalar.
        """
        return torch.tanh(self.q0_raw)


def validate_coarse_packet(
        packet: CoarsePacket,
        *,
        n_graphs: int,
        n_nodes: int,
        n_keys: int,
        n_edges: int,
        hidden: int = HIDDEN_V543,
        z_dim: int = ENCODER_Z_DIM_V543,
) -> None:
    """Typed shape/dtype/finiteness check of a :class:`CoarsePacket`.

    Refusal codes: ``V543_PACKET_TYPE``, ``V543_PACKET_FIELD_TYPE``,
    ``V543_PACKET_Z_SHAPE``, ``V543_PACKET_EDGE_SHAPE``,
    ``V543_PACKET_NODE_H_SHAPE``, ``V543_PACKET_U0_SHAPE``,
    ``V543_PACKET_Q0_SHAPE``, ``V543_PACKET_E0_SHAPE``,
    ``V543_PACKET_DTYPE``, ``V543_PACKET_NONFINITE``.  Every check is
    exact — a packet is either bit-correct or refused (plan §7.1 S0-A:
    "不得用「近似相等」放行").
    """
    if not isinstance(packet, CoarsePacket):
        raise V543ContractError(
            "V543_PACKET_TYPE",
            f"encoder must return CoarsePacket, got {type(packet).__name__}")
    for name in COARSE_PACKET_KEYS:
        if not isinstance(getattr(packet, name), torch.Tensor):
            raise V543ContractError(
                "V543_PACKET_FIELD_TYPE",
                f"CoarsePacket.{name} must be a torch.Tensor, got "
                f"{type(getattr(packet, name)).__name__}")
    expected = {
        "z": (int(n_graphs), int(z_dim)),
        "edge_msg": (int(n_edges), int(hidden)),
        "node_h": (int(n_nodes), int(hidden)),
        "u0": (int(n_keys), 3),
        "q0_raw": (int(n_graphs),),
        "e0": (int(n_graphs),),
    }
    codes = {
        "z": "V543_PACKET_Z_SHAPE",
        "edge_msg": "V543_PACKET_EDGE_SHAPE",
        "node_h": "V543_PACKET_NODE_H_SHAPE",
        "u0": "V543_PACKET_U0_SHAPE",
        "q0_raw": "V543_PACKET_Q0_SHAPE",
        "e0": "V543_PACKET_E0_SHAPE",
    }
    for name, want in expected.items():
        got = tuple(getattr(packet, name).shape)
        if got != want:
            raise V543ContractError(
                codes[name],
                f"CoarsePacket.{name} must be {list(want)}, got {list(got)}")
    for name in COARSE_PACKET_KEYS:
        tensor = getattr(packet, name)
        if tensor.dtype != torch.float32:
            raise V543ContractError(
                "V543_PACKET_DTYPE",
                f"fp32 required (CoarsePacket.{name} is {tensor.dtype})")
        if not bool(torch.isfinite(tensor).all()):
            raise V543ContractError(
                "V543_PACKET_NONFINITE",
                f"CoarsePacket.{name} contains non-finite values")


# ===========================================================================
# WP2 — ``DecoderContextV1`` + the two mutually-exclusive read-only views
# (plan §5.1/§5.2/§5.2.1/§5.6/§11.2).  Everything BELOW this banner is the
# additive WP2 surface; nothing above is re-interpreted.
#
# Normative sources, quoted as implemented:
#
# * §5.2 field table (:277–303) — field order, shape, dtype, unit and the
#   rotation / translation / permutation behaviour of every field are copied
#   verbatim into :data:`DECODER_CONTEXT_FIELDS` and the dataclass below;
# * §5.2 形状约束 (:304–309) — ``K = key_ptr[-1]``, ``P = pair_ptr[-1]``,
#   ``T = token_ptr[-1] = G + K + P``, monotone prefix sums starting at 0,
#   ``P = 0`` and ``Ek = 0`` explicitly allowed;
# * §5.2.1 (:311–352) — the two views and their hard isolation (MOVE carries
#   no ``q0_raw``/``endpoint0_logit``/``pair_s``/``token_*``, PROCESS carries
#   no ``u0``/``key_v``/``key_edge_*``);
# * §5.6 (:406–434) — :func:`assert_no_supervision` scans EVERY field for the
#   supervision-key tuple and the five forbidden names.
# ===========================================================================
#: Width of one ``key_v`` vector (plan §5.2 row ``key_v`` / §5.3 8 通道).
KEY_V_CHANNELS_V543: Final[int] = 8
#: ``key_v`` field shape tail: ``[K, 8, 3]`` (plan §5.2).
KEY_V_DIM_V543: Final[int] = 3
#: ``pair_s`` width (plan §5.2 ``[P,48]`` / §5.4 48 维布局).
PAIR_S_DIM_V543: Final[int] = KEY_S_DIM_V543
#: ``key_edge_attr`` width (plan §5.5 ``[Ek,8]``: d + 4·RBF + |ΔB| +
#: is_changed + edge_type).
KEY_EDGE_ATTR_DIM_V543: Final[int] = 8

#: Field order of :class:`DecoderContextV1` — plan §5.2 table (:277–303), row
#: for row.  ``__slots__`` contains EXACTLY these names (§5.6: "``__slots__``
#: 只含 §5.2 字段"), so no attribute outside this tuple is ever readable.
DECODER_CONTEXT_FIELDS: Final[tuple[str, ...]] = (
    "graph_s", "key_s", "key_v", "u0",
    "key_graph", "key_ptr",
    "key_edge_index", "key_edge_attr",
    "pair_s", "intent_pair_index", "pair_graph", "pair_ptr",
    "token_index", "token_type", "token_graph", "token_ptr",
    "q0_raw", "endpoint0_logit",
    "schema", "bridge_spec_sha256", "encoder_ckpt_sha256",
    "precision_version",
)

#: The four string-identity fields (plan §5.2 last four rows).
DECODER_CONTEXT_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "schema", "bridge_spec_sha256", "encoder_ckpt_sha256",
    "precision_version")

#: The eighteen tensor fields, in plan §5.2 order.
DECODER_CONTEXT_TENSOR_FIELDS: Final[tuple[str, ...]] = tuple(
    name for name in DECODER_CONTEXT_FIELDS
    if name not in DECODER_CONTEXT_IDENTITY_FIELDS)

#: MOVE view fields — plan §5.2.1 first block (:317–326), same order.
MOVE_CONTEXT_FIELDS: Final[tuple[str, ...]] = (
    "graph_s", "key_s", "key_v", "u0",
    "key_edge_index", "key_edge_attr", "key_graph", "key_ptr")

#: PROCESS view fields — plan §5.2.1 second block (:328–344), same order.
PROCESS_CONTEXT_FIELDS: Final[tuple[str, ...]] = (
    "graph_s", "key_s", "pair_s", "intent_pair_index",
    "key_graph", "pair_graph", "key_ptr", "pair_ptr",
    "token_index", "token_type", "token_graph", "token_ptr",
    "q0_raw", "endpoint0_logit")

#: Context fields a MOVE view must REFUSE (plan §5.2.1 :348 "``MoveContextV1``
#: **不含** ``q0_raw``、``endpoint0_logit``、``pair_s``、``token_*``；MOVE 不能
#: 读 PROCESS 状态").
MOVE_FORBIDDEN_FIELDS: Final[tuple[str, ...]] = tuple(
    name for name in DECODER_CONTEXT_TENSOR_FIELDS
    if name not in MOVE_CONTEXT_FIELDS)

#: Context fields a PROCESS view must REFUSE (plan §5.2.1 :349
#: "``ProcessContextV1`` **不含** ``u0``、``key_v``、``key_edge_*``").
PROCESS_FORBIDDEN_FIELDS: Final[tuple[str, ...]] = tuple(
    name for name in DECODER_CONTEXT_TENSOR_FIELDS
    if name not in PROCESS_CONTEXT_FIELDS)

#: Forbidden attribute names of the §5.6 runtime audit (:423 spelled-out names
#: plus the rest of the §5.6 :409–415 forbidden-input list in its singular
#: form) — on top of every :data:`SUPERVISION_KEYS_V3` member.
FORBIDDEN_CONTEXT_NAMES: Final[tuple[str, ...]] = (
    # :423 the five the plan spells out explicitly
    "pos", "xyz", "pair_edit", "y_teacher", "q_true",
    # :412 supervision labels / weights (singular twins of the plural keys)
    "pair_index", "d0", "frame_weight", "frame_mask",
    # :413 reference/future/history geometry and energy
    "reference_xyz", "future_xyz", "ts_xyz", "irc_history",
    "energy", "gradient",
    # :414 split / branch / identity
    "split_family", "branch_id", "curve_id", "sample_id", "intent_id",
    "frame_index",
)


class V543TrackIsolationError(V543Error, AttributeError):
    """A track view was asked for a field it must never see (plan §5.2.1).

    Inherits ``AttributeError`` as well as :class:`V543Error` so that the
    §5.6/§5.2.1 discipline ("任何 ``ctx.pos`` 访问必须 ``AttributeError``",
    :428) stays true while the refusal still carries ``code``/``detail``.
    """


class V543ContextError(V543ContractError):
    """``DecoderContextV1`` field/shape/identity drift (plan §5.2)."""


class V543BridgeError(V543ContractError):
    """``LatentBridge`` construction/audit refusal (plan §5.4/§5.5/§5.6)."""


class V543BridgeAuditError(V543BridgeError):
    """The pair-set source audit disagreed with supervision (plan §5.6)."""


@dataclass(frozen=True, slots=True)
class DecoderContextV1:
    """The V5.4.3 delivery unit — a frozen, read-only, parameter-free context.

    Fields (plan §5.2 table, :277–303 — shape/dtype/unit/rotation/
    translation/permutation are the table's own columns):

    +---------------------+-----------+---------+--------------------------------+
    | field               | shape     | dtype   | 旋转 / 平移 / 置换             |
    +=====================+===========+=========+================================+
    | ``graph_s``         | [G,64]    | fp32    | 不变 / 不变 / 图置换等变       |
    | ``key_s``           | [K,48]    | fp32    | 不变 / 不变 / 原子置换等变     |
    | ``key_v``           | [K,8,3]   | fp32    | 极向量 / 不变 / 原子置换等变   |
    | ``u0``              | [K,3] Å   | fp32    | 极向量 / 不变 / 原子置换等变   |
    | ``key_graph``       | [K]       | int64   | — / — / 随原子置换重编号       |
    | ``key_ptr``         | [G+1]     | int64   | — / — / 图置换等变             |
    | ``key_edge_index``  | [2,Ek]    | int64   | — / — / 原子置换等变           |
    | ``key_edge_attr``   | [Ek,8]    | fp32    | 不变 / 不变 / 边置换等变       |
    | ``pair_s``          | [P,48]    | fp32    | 不变 / 不变 / 对置换等变       |
    | ``intent_pair_index``| [2,P]    | int64   | — / — / 对置换等变             |
    | ``pair_graph``      | [P]       | int64   | — / — / 图置换等变             |
    | ``pair_ptr``        | [G+1]     | int64   | — / — / 图置换等变             |
    | ``token_index``     | [T]       | int64   | — / — / token 置换等变         |
    | ``token_type``      | [T]       | int8    | 0=graph, 1=key, 2=pair         |
    | ``token_graph``     | [T]       | int64   | — / — / 图置换等变             |
    | ``token_ptr``       | [G+1]     | int64   | — / — / 图置换等变             |
    | ``q0_raw``          | [G]       | fp32    | 不变 / 不变 / 图置换等变       |
    | ``endpoint0_logit`` | [G]       | fp32    | 不变 / 不变 / 图置换等变       |
    | ``schema``          | str       | —       | ``"decoder_context_v1"``       |
    | ``bridge_spec_sha256`` | str    | —       | Bridge 规格 canonical SHA      |
    | ``encoder_ckpt_sha256`` | str   | —       | A0 checkpoint SHA              |
    | ``precision_version`` | str     | —       | ``"fp32_v1"``                  |

    Absolute coordinates, ``batch``, history frames and every supervision key
    are structurally absent: the record cannot carry them (§5.6 :409–415) and
    :func:`assert_no_supervision` proves it on every build.

    The Bridge (WP2, ``latent_bridge.py``) is the ONLY producer; the two
    Refiner tracks never see this object, only :class:`MoveContextV1` /
    :class:`ProcessContextV1` (§5.2.1).
    """

    graph_s: torch.Tensor
    key_s: torch.Tensor
    key_v: torch.Tensor
    u0: torch.Tensor
    key_graph: torch.Tensor
    key_ptr: torch.Tensor
    key_edge_index: torch.Tensor
    key_edge_attr: torch.Tensor
    pair_s: torch.Tensor
    intent_pair_index: torch.Tensor
    pair_graph: torch.Tensor
    pair_ptr: torch.Tensor
    token_index: torch.Tensor
    token_type: torch.Tensor
    token_graph: torch.Tensor
    token_ptr: torch.Tensor
    q0_raw: torch.Tensor
    endpoint0_logit: torch.Tensor
    schema: str = DECODER_CONTEXT_SCHEMA_V543
    bridge_spec_sha256: str = ""
    encoder_ckpt_sha256: str = ""
    precision_version: str = PRECISION_VERSION_V543

    # -- derived counts (never stored; plan §5.2 形状约束 :307) --------------
    @property
    def n_graphs(self) -> int:
        return int(self.key_ptr.shape[0]) - 1

    @property
    def n_keys(self) -> int:
        return int(self.key_s.shape[0])

    @property
    def n_pairs(self) -> int:
        return int(self.pair_s.shape[0])

    @property
    def n_key_edges(self) -> int:
        return int(self.key_edge_attr.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.token_index.shape[0])

    def as_dict(self) -> dict[str, object]:
        """Field-name → value mapping in plan §5.2 order (cache/audit shape)."""
        return {name: getattr(self, name) for name in DECODER_CONTEXT_FIELDS}

    def move_view(self) -> "MoveContextV1":
        """The MOVE-only view (plan §5.2.1)."""
        return MoveContextV1.from_context(self)

    def process_view(self) -> "ProcessContextV1":
        """The PROCESS-only view (plan §5.2.1)."""
        return ProcessContextV1.from_context(self)


@dataclass(frozen=True, slots=True)
class MoveContextV1:
    """MOVE-only read-only view (plan §5.2.1, :317–326) — 8 fields, no more.

    Hard isolation (:348): no ``q0_raw``, no ``endpoint0_logit``, no
    ``pair_s``, no ``token_*``; every such access raises
    :class:`V543TrackIsolationError` (which is also an ``AttributeError``).
    """

    graph_s: torch.Tensor
    key_s: torch.Tensor
    key_v: torch.Tensor
    u0: torch.Tensor
    key_edge_index: torch.Tensor
    key_edge_attr: torch.Tensor
    key_graph: torch.Tensor
    key_ptr: torch.Tensor

    @classmethod
    def from_context(cls, ctx: DecoderContextV1) -> "MoveContextV1":
        """Project a context onto the MOVE field set (no shared new data)."""
        if not isinstance(ctx, DecoderContextV1):
            raise V543ContextError(
                "V543_CONTEXT_TYPE",
                f"MoveContextV1.from_context needs DecoderContextV1, got "
                f"{type(ctx).__name__}")
        return cls(**{name: getattr(ctx, name) for name in MOVE_CONTEXT_FIELDS})

    def __getattr__(self, name: str) -> object:
        _refuse_missing_field("MoveContextV1", name, MOVE_FORBIDDEN_FIELDS)


@dataclass(frozen=True, slots=True)
class ProcessContextV1:
    """PROCESS-only read-only view (plan §5.2.1, :328–344) — 14 fields.

    Hard isolation (:349): no ``u0``, no ``key_v``, no ``key_edge_*``;
    every such access raises :class:`V543TrackIsolationError`.
    """

    graph_s: torch.Tensor
    key_s: torch.Tensor
    pair_s: torch.Tensor
    intent_pair_index: torch.Tensor
    key_graph: torch.Tensor
    pair_graph: torch.Tensor
    key_ptr: torch.Tensor
    pair_ptr: torch.Tensor
    token_index: torch.Tensor
    token_type: torch.Tensor
    token_graph: torch.Tensor
    token_ptr: torch.Tensor
    q0_raw: torch.Tensor
    endpoint0_logit: torch.Tensor

    @classmethod
    def from_context(cls, ctx: DecoderContextV1) -> "ProcessContextV1":
        """Project a context onto the PROCESS field set (no shared new data)."""
        if not isinstance(ctx, DecoderContextV1):
            raise V543ContextError(
                "V543_CONTEXT_TYPE",
                f"ProcessContextV1.from_context needs DecoderContextV1, got "
                f"{type(ctx).__name__}")
        return cls(**{name: getattr(ctx, name)
                      for name in PROCESS_CONTEXT_FIELDS})

    def __getattr__(self, name: str) -> object:
        _refuse_missing_field("ProcessContextV1", name,
                              PROCESS_FORBIDDEN_FIELDS)


def _refuse_missing_field(
        view: str, name: str, forbidden: tuple[str, ...]) -> None:
    """ALWAYS raise the §5.2.1 refusal for an out-of-field-set attribute.

    The raise lives HERE (not at the call site) so a ``__getattr__`` can never
    accidentally *return* the refusal object instead of raising it — that
    failure mode silently turns the hard isolation into a no-op
    (plan §5.2.1 :346–352 "两个视图都不含 …").  A view never forwards an
    unknown name to the context or to the batch (§5.6: "``__getattr__`` 不得
    回退到 batch").
    """
    raise _track_refusal(view, name, forbidden)


def _track_refusal(
        view: str, name: str, forbidden: tuple[str, ...]) -> V543TrackIsolationError:
    """Build the §5.2.1 refusal for a missing/forbidden view attribute.

    A field that lives on the OTHER track gets the isolation code
    ``V543_TRACK_ISOLATION`` (the plan's "硬隔离" wording, :346–352); any
    other name gets ``V543_TRACK_UNKNOWN_FIELD``.  Both are typed
    ``AttributeError``-compatible refusals carrying ``code``/``detail``.
    """
    if name in forbidden:
        return V543TrackIsolationError(
            "V543_TRACK_ISOLATION",
            f"{view} has no field {name!r} by contract: the two track views "
            "are mutually exclusive and may not read each other's state "
            "(plan §5.2.1)")
    return V543TrackIsolationError(
        "V543_TRACK_UNKNOWN_FIELD",
        f"{view} has no field {name!r}; the view carries exactly "
        "the plan §5.2.1 field list (no batch, no pos, no supervision)")


@dataclass(frozen=True, slots=True)
class ProcessResidual:
    """PROCESS-track residual of one forward (plan §6.1 protocol, :497–500).

    ``ProcessDecoderProtocol.forward`` returns "``δq_raw [G]`` and
    ``δe [G]``": pre-tanh progress residual and endpoint-logit residual, both
    per graph.  The zero-residual initialisation of plan §4.5 (P1–P5: the
    last ``Linear(·,2)`` is zero) is exactly ``delta_q_raw == delta_e == 0``,
    so the M0-equivalence of §5.8 holds by construction.
    """

    delta_q_raw: torch.Tensor
    delta_e: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("delta_q_raw", "delta_e"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise V543ContextError(
                    "V543_RESIDUAL_FIELD_TYPE",
                    f"ProcessResidual.{name} must be a torch.Tensor, got "
                    f"{type(value).__name__}")
            if value.ndim != 1:
                raise V543ContextError(
                    "V543_RESIDUAL_SHAPE",
                    f"ProcessResidual.{name} must be [G], got "
                    f"{list(value.shape)}")


def context_field_names(ctx: object) -> tuple[str, ...]:
    """Every readable field name of a context/view/any slotted record."""
    if hasattr(type(ctx), "__dataclass_fields__"):
        return tuple(name for name in type(ctx).__dataclass_fields__)
    slots = getattr(type(ctx), "__slots__", ())
    if isinstance(slots, str):
        return (slots,)
    return tuple(str(name) for name in slots)


#: The full §5.6 forbidden-name set: every supervision key + the five names
#: the plan spells out explicitly (:423).
FORBIDDEN_CONTEXT_ALL_NAMES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys((*SUPERVISION_KEYS_V3, *FORBIDDEN_CONTEXT_NAMES)))


def assert_no_supervision(ctx: object) -> None:
    """Typed runtime audit — no supervision/geometry name is reachable (§5.6).

    Implements :420–425 literally: every ``SUPERVISION_KEYS_V3`` member and
    every name in ``("pos", "xyz", "pair_edit", "y_teacher", "q_true")`` must
    be absent from the record's field set.  The scan covers dataclass fields,
    ``__slots__`` and any ``__dict__`` — a context that gained a private
    attribute cannot hide from it (§5.6: "``__getattr__`` 不得回退到 batch").

    Refusal: ``V543_CONTEXT_SUPERVISION`` (typed, with ``code``/``detail``).
    """
    present = set(context_field_names(ctx))
    extra = getattr(ctx, "__dict__", None)
    if isinstance(extra, dict):
        present.update(str(name) for name in extra)
    leaks = sorted(present.intersection(FORBIDDEN_CONTEXT_ALL_NAMES))
    if leaks:
        raise V543SupervisionLeakError(
            "V543_CONTEXT_SUPERVISION",
            f"{type(ctx).__name__} exposes forbidden name(s) {leaks}; "
            "supervision keys, absolute coordinates and teacher labels may "
            "never reach a DecoderContextV1 or a track view (plan §5.6)")
    # A tensor must never be aliased under a supervision name either.
    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        if name in FORBIDDEN_CONTEXT_ALL_NAMES:  # pragma: no cover - disjoint
            raise V543SupervisionLeakError(
                "V543_CONTEXT_SUPERVISION",
                f"context field {name!r} collides with a forbidden name")


def validate_decoder_context(ctx: DecoderContextV1) -> None:
    """Exact shape/dtype/prefix-sum/index audit of a ``DecoderContextV1``.

    Every rule is plan §5.2 形状约束 (:304–309), quoted as coded:

    * ``K = key_ptr[-1]``, ``P = pair_ptr[-1]``, ``T = token_ptr[-1] = G + K +
      P``;
    * ``key_ptr`` / ``pair_ptr`` / ``token_ptr`` are monotone non-decreasing,
      start at 0 and end at the corresponding total;
    * each graph has at least one key (``key_ptr`` strictly increasing by ≥ 1)
      and ``P = 0`` / ``Ek = 0`` are legal (empty ``pair_s [0,48]``,
      ``key_edge_* [*,0]``);
    * every field has the table's dtype and shape, all floats are finite, and
      ``intent_pair_index`` is canonical (``i < j`` in key-slot order, §5.4).

    Refusal codes: ``V543_CONTEXT_TYPE``, ``V543_CONTEXT_SCHEMA``,
    ``V543_CONTEXT_PRECISION``, ``V543_CONTEXT_FIELD_TYPE``,
    ``V543_CONTEXT_<FIELD>_SHAPE``, ``V543_CONTEXT_DTYPE``,
    ``V543_CONTEXT_PTR``, ``V543_CONTEXT_TOKEN_LAYOUT``,
    ``V543_CONTEXT_INDEX_RANGE``, ``V543_CONTEXT_PAIR_ORDER``,
    ``V543_CONTEXT_NONFINITE``.
    """
    if not isinstance(ctx, DecoderContextV1):
        raise V543ContextError(
            "V543_CONTEXT_TYPE",
            f"expected DecoderContextV1, got {type(ctx).__name__}")
    if ctx.schema != DECODER_CONTEXT_SCHEMA_V543:
        raise V543ContextError(
            "V543_CONTEXT_SCHEMA",
            f"schema must be {DECODER_CONTEXT_SCHEMA_V543!r}, got "
            f"{ctx.schema!r}")
    if ctx.precision_version != PRECISION_VERSION_V543:
        raise V543ContextError(
            "V543_CONTEXT_PRECISION",
            f"precision_version must be {PRECISION_VERSION_V543!r}, got "
            f"{ctx.precision_version!r}")
    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        if not isinstance(getattr(ctx, name), torch.Tensor):
            raise V543ContextError(
                "V543_CONTEXT_FIELD_TYPE",
                f"DecoderContextV1.{name} must be a torch.Tensor, got "
                f"{type(getattr(ctx, name)).__name__}")

    n_graphs = int(ctx.graph_s.shape[0])
    n_keys = int(ctx.key_s.shape[0])
    n_pairs = int(ctx.pair_s.shape[0])
    n_key_edges = int(ctx.key_edge_attr.shape[0])
    n_tokens = int(ctx.token_index.shape[0])

    expected: dict[str, tuple[int, ...]] = {
        "graph_s": (n_graphs, GRAPH_S_DIM_V543),
        "key_s": (n_keys, KEY_S_DIM_V543),
        "key_v": (n_keys, KEY_V_CHANNELS_V543, KEY_V_DIM_V543),
        "u0": (n_keys, 3),
        "key_graph": (n_keys,),
        "key_ptr": (n_graphs + 1,),
        "key_edge_index": (2, n_key_edges),
        "key_edge_attr": (n_key_edges, KEY_EDGE_ATTR_DIM_V543),
        "pair_s": (n_pairs, PAIR_S_DIM_V543),
        "intent_pair_index": (2, n_pairs),
        "pair_graph": (n_pairs,),
        "pair_ptr": (n_graphs + 1,),
        "token_index": (n_tokens,),
        "token_type": (n_tokens,),
        "token_graph": (n_tokens,),
        "token_ptr": (n_graphs + 1,),
        "q0_raw": (n_graphs,),
        "endpoint0_logit": (n_graphs,),
    }
    for name, want in expected.items():
        got = tuple(getattr(ctx, name).shape)
        if got != want:
            raise V543ContextError(
                f"V543_CONTEXT_{name.upper()}_SHAPE",
                f"DecoderContextV1.{name} must be {list(want)}, got "
                f"{list(got)}")

    dtypes: dict[str, torch.dtype] = {
        "graph_s": torch.float32, "key_s": torch.float32,
        "key_v": torch.float32, "u0": torch.float32,
        "key_edge_attr": torch.float32, "pair_s": torch.float32,
        "q0_raw": torch.float32, "endpoint0_logit": torch.float32,
        "key_graph": torch.int64, "key_ptr": torch.int64,
        "key_edge_index": torch.int64, "intent_pair_index": torch.int64,
        "pair_graph": torch.int64, "pair_ptr": torch.int64,
        "token_index": torch.int64, "token_type": torch.int8,
        "token_graph": torch.int64, "token_ptr": torch.int64,
    }
    for name, want_dtype in dtypes.items():
        got_dtype = getattr(ctx, name).dtype
        if got_dtype != want_dtype:
            raise V543ContextError(
                "V543_CONTEXT_DTYPE",
                f"DecoderContextV1.{name} must be {want_dtype}, got "
                f"{got_dtype} (plan §5.2 table)")

    # -- prefix sums (§5.2 形状约束 :307–309) -------------------------------
    for name, total, strict in (("key_ptr", n_keys, True),
                                ("pair_ptr", n_pairs, False),
                                ("token_ptr", n_tokens, False)):
        ptr = getattr(ctx, name)
        if int(ptr[0]) != 0:
            raise V543ContextError(
                "V543_CONTEXT_PTR", f"{name}[0] must be 0, got {int(ptr[0])}")
        steps = ptr[1:] - ptr[:-1]
        if bool((steps < 0).any()):
            raise V543ContextError(
                "V543_CONTEXT_PTR",
                f"{name} must be monotone non-decreasing (got "
                f"{ptr.tolist()})")
        if strict and bool((steps < 1).any()):
            raise V543ContextError(
                "V543_CONTEXT_PTR",
                f"{name} must increase by at least 1 per graph (每个图至少 "
                f"1 个 key, plan §5.2.1); got {ptr.tolist()}")
        if int(ptr[-1]) != int(total):
            raise V543ContextError(
                "V543_CONTEXT_PTR",
                f"{name}[-1] must equal {total}, got {int(ptr[-1])} "
                "(K = key_ptr[-1], P = pair_ptr[-1], T = token_ptr[-1])")
    if n_tokens != n_graphs + n_keys + n_pairs:
        raise V543ContextError(
            "V543_CONTEXT_TOKEN_LAYOUT",
            f"T = token_ptr[-1] = {n_tokens} must equal G + K + P = "
            f"{n_graphs + n_keys + n_pairs} (plan §5.2 :307)")

    # -- index ranges -------------------------------------------------------
    if n_key_edges:
        if int(ctx.key_edge_index.min()) < 0 or \
                int(ctx.key_edge_index.max()) >= n_keys:
            raise V543ContextError(
                "V543_CONTEXT_INDEX_RANGE",
                f"key_edge_index rows must lie in [0,{n_keys})")
    if n_pairs:
        if int(ctx.intent_pair_index.min()) < 0 or \
                int(ctx.intent_pair_index.max()) >= n_keys:
            raise V543ContextError(
                "V543_CONTEXT_INDEX_RANGE",
                f"intent_pair_index rows must lie in [0,{n_keys})")
        if bool((ctx.intent_pair_index[0] >= ctx.intent_pair_index[1]).any()):
            raise V543ContextError(
                "V543_CONTEXT_PAIR_ORDER",
                "intent_pair_index must be canonical (i < j in key-slot "
                "order) — plan §5.4 'i<j 为 key-slot 规范序'")
    types = ctx.token_type.to(torch.int64)
    if int(types.min()) < 0 or int(types.max()) > 2:
        raise V543ContextError(
            "V543_CONTEXT_TOKEN_LAYOUT",
            "token_type must be 0=graph / 1=key / 2=pair (plan §5.2)")
    for graph in range(n_graphs):
        start = int(ctx.token_ptr[graph])
        stop = int(ctx.token_ptr[graph + 1])
        if int(types[start]) != 0:
            raise V543ContextError(
                "V543_CONTEXT_TOKEN_LAYOUT",
                f"token {start} (graph {graph}) must be the graph token "
                "(每图 [graph, keys…, pairs…], plan §5.5)")
        if int(ctx.token_index[start]) != -1:
            raise V543ContextError(
                "V543_CONTEXT_TOKEN_LAYOUT",
                f"graph token {start} must carry token_index -1 (plan §5.5)")
        if int(ctx.token_graph[start]) != graph:
            raise V543ContextError(
                "V543_CONTEXT_TOKEN_LAYOUT",
                f"graph token {start} must carry token_graph {graph}")
        key_end = start + 1 + (int(ctx.key_ptr[graph + 1])
                               - int(ctx.key_ptr[graph]))
        if key_end > stop:
            raise V543ContextError(
                "V543_CONTEXT_TOKEN_LAYOUT",
                f"graph {graph} token span [{start},{stop}) is smaller than "
                "its [graph, keys…] prefix")
        if bool((types[start + 1:key_end] != 1).any()):
            raise V543ContextError(
                "V543_CONTEXT_TOKEN_LAYOUT",
                f"graph {graph} must lay out its key tokens right after the "
                "graph token (plan §5.5)")
        if bool((types[key_end:stop] != 2).any()):
            raise V543ContextError(
                "V543_CONTEXT_TOKEN_LAYOUT",
                f"graph {graph} must lay out its pair tokens after the key "
                "tokens (plan §5.5)")
        for token in range(start + 1, key_end):
            if not 0 <= int(ctx.token_index[token]) < n_keys:
                raise V543ContextError(
                    "V543_CONTEXT_INDEX_RANGE",
                    f"key token {token} carries token_index "
                    f"{int(ctx.token_index[token])} outside [0,{n_keys})")
        for token in range(key_end, stop):
            if not 0 <= int(ctx.token_index[token]) < max(1, n_pairs):
                raise V543ContextError(
                    "V543_CONTEXT_INDEX_RANGE",
                    f"pair token {token} carries token_index "
                    f"{int(ctx.token_index[token])} outside [0,{n_pairs})")

    # -- finiteness --------------------------------------------------------
    for name in DECODER_CONTEXT_TENSOR_FIELDS:
        tensor = getattr(ctx, name)
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise V543ContextError(
                "V543_CONTEXT_NONFINITE",
                f"DecoderContextV1.{name} contains non-finite values")


__all__ = [
    "A_REF_V543",
    "COARSE_PACKET_KEYS",
    "COARSE_PACKET_SCHEMA",
    "DECODER_CONTEXT_SCHEMA_V543",
    "EDGE_MSG_DIM_V543",
    "ENCODER_ACTIVE_PARAMS_V543",
    "ENCODER_FORWARD_CALLS_PER_VIEW",
    "ENCODER_JOINT_PARAMS_V543",
    "ENCODER_LOAD_AUDIT_SCHEMA",
    "ENCODER_Z_DIM_V543",
    "GRAPH_S_DIM_V543",
    "HIDDEN_V543",
    "HOOK_FIRES_PER_FORWARD",
    "HOOK_LABELS",
    "HOOK_LABEL_EDGE_MSG",
    "HOOK_LABEL_NODE_H",
    "HOOK_LABEL_Q0_RAW",
    "HOOK_MODULE_PATHS",
    "KEY_S_DIM_V543",
    "MAPPING_MODE_FROZEN",
    "MAPPING_MODE_IDENTITY",
    "N_DECODER_PARAMS_V543",
    "N_PROCESS_PARAMS_V543",
    "PARITY_ATOL_CPU",
    "PARITY_ATOL_GPU",
    "PAYLOAD_LAYOUTS_V543",
    "PAYLOAD_LAYOUT_LEGACY_43",
    "PAYLOAD_LAYOUT_V542_A0_45",
    "PRECISION_VERSION_V543",
    "SUPERVISION_KEYS_V543",
    "V543_ENCODER_VERSION",
    "V543_SCHEMA_VERSION",
    "CoarsePacket",
    "V543ContractError",
    "V543Error",
    "V543HookError",
    "V543InitError",
    "V543LoadError",
    "V543ParamBudgetError",
    "V543ParityError",
    "V543SupervisionLeakError",
    # --- WP2 additions (plan §5.2 / §5.2.1 / §5.6) ------------------------
    "DECODER_CONTEXT_FIELDS",
    "DECODER_CONTEXT_IDENTITY_FIELDS",
    "DECODER_CONTEXT_TENSOR_FIELDS",
    "FORBIDDEN_CONTEXT_ALL_NAMES",
    "FORBIDDEN_CONTEXT_NAMES",
    "KEY_EDGE_ATTR_DIM_V543",
    "KEY_V_CHANNELS_V543",
    "KEY_V_DIM_V543",
    "MOVE_CONTEXT_FIELDS",
    "MOVE_FORBIDDEN_FIELDS",
    "PAIR_S_DIM_V543",
    "PROCESS_CONTEXT_FIELDS",
    "PROCESS_FORBIDDEN_FIELDS",
    "DecoderContextV1",
    "MoveContextV1",
    "ProcessContextV1",
    "ProcessResidual",
    "V543BridgeAuditError",
    "V543BridgeError",
    "V543ContextError",
    "V543TrackIsolationError",
    "assert_no_supervision",
    "context_field_names",
    "validate_coarse_packet",
    "validate_decoder_context",
]
