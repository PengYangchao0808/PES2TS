"""V5.4.3 Decoder Registry — protocols, the 12-field record, parameter budget
and the M0/P0 zero-parameter baselines (plan §6.1 :485–519, §6.2 :522, §6.3
:539, §6.4 :557–571, §11.2 ``registry.py``; WP4).

Normative sources, quoted as implemented
----------------------------------------

* **§6.1 protocol block (:488–499)** — the two protocols are transcribed
  verbatim as :class:`MoveDecoderProtocol` (``forward(MoveContextV1) ->
  Tensor`` returning ``δu [K,3]``) and :class:`ProcessDecoderProtocol`
  (``forward(ProcessContextV1) -> ProcessResidual`` returning ``δq_raw [G]``
  and ``δe [G]``).  A decoder therefore receives EXACTLY ONE argument — a
  read-only track view — and can reach nothing else.
* **§6.1 record table (:501–517)** — "Registry 每条记录至少包含" 13 rows:
  ``decoder_id`` / ``track`` / ``model_version`` / ``contract_version`` /
  ``n_trainable_params`` / ``equivariance`` / ``stochastic`` / ``nfe`` /
  ``zero_init`` / ``status`` / ``checkpoint_sha256`` / ``config_sha256`` /
  ``acceptance``; :data:`REGISTRY_RECORD_FIELDS` is that list in order and
  :meth:`DecoderSpec.record` emits exactly it.  (The WP4 task text says "12
  字段"; the plan table has 13 rows — the plan wins, see the WP4 report.)
* **§6.1 forbidden block (:519)** — "**禁止**：Decoder 私自实现数据读取、
  损失定义、checkpoint 选择；这些统一由 ``v543_train.py`` 负责."
  :func:`assert_decoder_protocol` enforces this structurally: a decoder that
  exposes a data/label/loss/selection surface, or a ``forward`` that takes
  more than the one context argument, is a typed refusal.
* **§6.2 :522/§6.3 :539** — ``M0`` = 旧 ``D0`` with ``δu = 0`` and 0 params;
  ``P0`` = 旧 q/endpoint head with ``δq = δe = 0`` and 0 params.  "必须（基线）".
* **§6.4 parameter table (:558–571)** — per-track target 36,000 trainable
  params, band ±5 % (34,200–37,800), single-candidate hard cap 40,000, M0/P0
  = 0, global hard cap 150,000, "断言：构造时 ``sum(p.numel())`` 精确断言".
  :func:`assert_budget` is that assertion; it is a typed refusal carrying
  ``code``/``detail`` (never a warning), and out-of-band counts are refused
  by :data:`PARAM_BAND_MIN_V543` / :data:`PARAM_BAND_MAX_V543` /
  :data:`PARAM_HARD_CAP_V543`.

Design decisions and their basis
--------------------------------

1. **No design placeholders.**  WP4 registers only what exists: ``M0`` and
   ``P0`` (zero-parameter baselines, constructible today).  ``M1–M5`` /
   ``P1–P5`` are the WP5/WP6 candidates; their ids are DECLARED in
   :data:`DECODER_IDS_V543` (so a run-plan can name them and a wrong-track
   registration is refused early) but they are NOT registered here — asking
   the registry to build one is a typed ``V543_REGISTRY_UNKNOWN`` refusal
   rather than a silently fabricated stub.  This is the honest WP4 boundary:
   "候选 decoder 本身是 WP5/WP6 的事".
2. **Track ownership is checked on every path.**  A MOVE decoder handed to
   the PROCESS track (or a spec whose ``track`` disagrees with the id's track)
   is ``V543_REGISTRY_TRACK``.  The ids themselves carry the track
   (``M*`` → move, ``P*`` → process), so a typo cannot silently train the
   wrong head.
3. **The budget assertion is the CONSTRUCTION assertion.**  ``register``
   builds the decoder once, counts ``sum(p.numel())`` and refuses an
   out-of-band count immediately; ``build`` re-counts every constructed
   instance (a factory that returns a differently sized model under another
   kwarg is still refused).  ``assert_budget`` is public so WP5/WP6 tests can
   re-assert on a loaded checkpoint.
4. **A decoder is not an nn.Module by requirement.**  The protocol is
   structural (``forward`` only); the zero baselines are ``nn.Module``
   subclasses with no parameter at all, so ``state_dict()`` is empty and an
   optimizer can never see them (mirrors the ``LatentBridge`` parameter
   policy, plan §5.1 :273).

Discipline: this module imports the WP1/WP2 contracts only; it declares no
new numeric contract, keeps every constant ``Final``, and never calls
``torch.compile`` (plan §11.3 禁止)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Mapping, Protocol, runtime_checkable

import torch

from pes2ts_core.pesn.models.v543.contracts import (
    DECODER_CONTEXT_SCHEMA_V543,
    PRECISION_VERSION_V543,
    MoveContextV1,
    ProcessContextV1,
    ProcessResidual,
    V543Error,
    V543ParamBudgetError,
)

#: Schema id of this registry's ``describe`` payloads.
V543_REGISTRY_SCHEMA: Final[str] = "pesn_v543_decoder_registry_v1"

#: Registered model-layer version of the WP4 registry surface.
V543_REGISTRY_VERSION: Final[str] = "pesn_v543_registry_v1"

#: The two tracks (plan §4.2 / §6.1 ``track`` row).
TRACK_MOVE: Final[str] = "move"
TRACK_PROCESS: Final[str] = "process"
TRACKS_V543: Final[tuple[str, ...]] = (TRACK_MOVE, TRACK_PROCESS)

#: Baseline id per track (plan §6.2 :522 M0 / §6.3 :539 P0: "必须（基线）").
BASELINE_DECODER_BY_TRACK: Final[Mapping[str, str]] = {
    TRACK_MOVE: "M0", TRACK_PROCESS: "P0"}

#: Every candidate id of the campaign (plan §6.2 :523–530, §6.3 :541–548).
MOVE_DECODER_IDS_V543: Final[tuple[str, ...]] = (
    "M0", "M1", "M2", "M3", "M4", "M5")
PROCESS_DECODER_IDS_V543: Final[tuple[str, ...]] = (
    "P0", "P1", "P2", "P3", "P4", "P5")
DECODER_IDS_V543: Final[tuple[str, ...]] = (
    MOVE_DECODER_IDS_V543 + PROCESS_DECODER_IDS_V543)

#: Per-track parameter target / band / hard cap (plan §6.4 :562–564).
PARAM_TARGET_V543: Final[int] = 36_000
PARAM_BAND_PCT_V543: Final[float] = 5.0
PARAM_BAND_MIN_V543: Final[int] = 34_200      # 36,000 × 0.95
PARAM_BAND_MAX_V543: Final[int] = 37_800      # 36,000 × 1.05
PARAM_HARD_CAP_V543: Final[int] = 40_000
#: Trainable params of a zero baseline (plan §6.4 :565 "M0/P0 | 0").
PARAM_BASELINE_V543: Final[int] = 0
#: Global model hard cap (plan §6.4 :568).
PARAM_GLOBAL_CAP_V543: Final[int] = 150_000
#: Frozen encoder joint count the assembly adds on top (plan §6.4 :566).
ENCODER_JOINT_PARAMS_V543: Final[int] = 57_619

#: Equivariance vocabulary of the §6.1 record (plan §6.1 :511).
EQUIVARIANCE_VALUES_V543: Final[tuple[str, ...]] = (
    "invariant", "polar", "stochastic_polar")
#: Status vocabulary of the §6.1 record (plan §6.1 :514).
STATUS_VALUES_V543: Final[tuple[str, ...]] = (
    "proposed", "smoke", "diagnostic", "formal", "champion", "rejected")

#: The §6.1 record table, row for row (:503–517), in the plan's order.
REGISTRY_RECORD_FIELDS: Final[tuple[str, ...]] = (
    "decoder_id",
    "track",
    "model_version",
    "contract_version",
    "n_trainable_params",
    "equivariance",
    "stochastic",
    "nfe",
    "zero_init",
    "status",
    "checkpoint_sha256",
    "config_sha256",
    "acceptance",
)

#: Names a decoder may NEVER expose (plan §6.1 :519: no private data reading,
#: no loss definition, no checkpoint selection — those belong to
#: ``v543_train.py``).  Matched against attribute names and submodule names.
FORBIDDEN_DECODER_NAMES: Final[tuple[str, ...]] = (
    # data / batching
    "dataset", "dataloader", "data_loader", "batch", "records", "samples",
    "labels", "labels_v52", "endpoint_labels", "label_status",
    # supervision
    "pos", "xyz", "pair_index", "pair_edit", "y_teacher", "q_true",
    "frame_weight", "frame_mask", "endpoint_mask",
    # loss / optimisation / selection
    "loss", "loss_fn", "criterion", "compute_loss", "loss_on_batch",
    "backward", "step", "optimizer", "select_checkpoint", "checkpoint_score",
    "best_checkpoint", "score",
)

#: Methods a decoder may NEVER define (same §6.1 :519 rule, method form).
FORBIDDEN_DECODER_METHODS: Final[tuple[str, ...]] = (
    "compute_loss", "loss", "backward", "step", "select_checkpoint",
    "load_data", "read_batch", "score_checkpoint",
)


# ---------------------------------------------------------------------------
# typed refusals
# ---------------------------------------------------------------------------
class V543RegistryError(V543Error):
    """Decoder-registry refusal (unknown id, track clash, protocol breach)."""


class V543DecoderProtocolError(V543RegistryError):
    """A decoder object does not satisfy the §6.1 protocol / prohibitions."""


# ---------------------------------------------------------------------------
# §6.1 protocols (verbatim transcription of the plan block :488–499)
# ---------------------------------------------------------------------------
@runtime_checkable
class MoveDecoderProtocol(Protocol):
    """``forward(MoveContextV1) -> δu [K,3]`` (plan §6.1 :489–493).

    "Return δu [K,3] in Å; zero rows for non-key atoms are impossible because
    K rows are key-only."  The single argument is the read-only MOVE view, so
    the decoder provably cannot reach ``q0_raw``/``endpoint0_logit``/``pair_s``
    (plan §5.2.1 hard isolation).
    """

    def forward(self, ctx: MoveContextV1) -> torch.Tensor:  # pragma: no cover
        ...


@runtime_checkable
class ProcessDecoderProtocol(Protocol):
    """``forward(ProcessContextV1) -> ProcessResidual`` (plan §6.1 :495–498).

    "Return δq_raw [G] and δe [G]."  The single argument is the read-only
    PROCESS view, so the decoder provably cannot reach ``u0``/``key_v``/
    ``key_edge_*`` (plan §5.2.1 hard isolation).
    """

    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:  # pragma: no cover
        ...


PROTOCOL_BY_TRACK: Final[Mapping[str, Any]] = {
    TRACK_MOVE: MoveDecoderProtocol, TRACK_PROCESS: ProcessDecoderProtocol}

#: The one and only forward argument of each track (plan §6.1 :490/:496).
CONTEXT_TYPE_BY_TRACK: Final[Mapping[str, Any]] = {
    TRACK_MOVE: MoveContextV1, TRACK_PROCESS: ProcessContextV1}


# ---------------------------------------------------------------------------
# parameter budget (plan §6.4 :558–571)
# ---------------------------------------------------------------------------
def count_trainable_params(module: Any) -> int:
    """Exact ``sum(p.numel())`` over ``module.parameters()`` (§6.4 :569).

    A parameter-free object (the M0/P0 baselines) counts 0 — there is no
    ``sum(())`` shortcut, the empty product IS the answer the plan asks for.
    """
    if module is None:
        return 0
    parameters = getattr(module, "parameters", None)
    if parameters is None:
        return 0
    return int(sum(int(p.numel()) for p in parameters()))


def param_band_tolerance() -> int:
    """Half-width of the ±5 % band in parameters (1,800)."""
    return int(round(PARAM_TARGET_V543 * PARAM_BAND_PCT_V543 / 100.0))


def assert_budget(
        n_trainable_params: int,
        *,
        decoder_id: str,
        track: str,
) -> int:
    """The construction-time parameter assertion of plan §6.4 (:569).

    Refusal codes (all typed, ``code`` + ``detail``):

    * ``V543_REGISTRY_TRACK`` — ``track`` is not one of :data:`TRACKS_V543`;
    * ``V543_REGISTRY_BASELINE_PARAMS`` — M0/P0 must be EXACTLY 0 (§6.4 :565);
    * ``V543_REGISTRY_PARAMS_CAP`` — a single candidate above the 40,000 hard
      cap (§6.4 :564 "单候选硬上限 40,000");
    * ``V543_REGISTRY_PARAMS_BAND`` — a trainable candidate outside
      34,200–37,800 (§6.4 :563 "允许带 ±5 %"); the detail names the exact
      count, the target and the band so a MY5/MY6 tuning step can size the
      layers instead of guessing.

    Returns the asserted count.
    """
    _assert_track(track, decoder_id=decoder_id)
    count = int(n_trainable_params)
    if count < 0:
        raise V543ParamBudgetError(
            "V543_REGISTRY_PARAMS_NEGATIVE",
            f"{decoder_id}: parameter count {count} is negative")
    if decoder_id in BASELINE_DECODER_BY_TRACK.values():
        if count != PARAM_BASELINE_V543:
            raise V543ParamBudgetError(
                "V543_REGISTRY_BASELINE_PARAMS",
                f"{decoder_id} is a zero-parameter baseline (plan §6.4 :565 "
                f"'M0/P0 | 0'); got {count} trainable parameters")
        return count
    if count > PARAM_HARD_CAP_V543:
        raise V543ParamBudgetError(
            "V543_REGISTRY_PARAMS_CAP",
            f"{decoder_id}: {count} trainable parameters exceed the "
            f"single-candidate hard cap {PARAM_HARD_CAP_V543} (plan §6.4 "
            ":564)")
    if not PARAM_BAND_MIN_V543 <= count <= PARAM_BAND_MAX_V543:
        raise V543ParamBudgetError(
            "V543_REGISTRY_PARAMS_BAND",
            f"{decoder_id}: {count} trainable parameters are outside the "
            f"frozen band [{PARAM_BAND_MIN_V543}, {PARAM_BAND_MAX_V543}] "
            f"(target {PARAM_TARGET_V543} ±{PARAM_BAND_PCT_V543} %, plan "
            "§6.4 :563) — resize the layers, never pad with dead weights "
            "(§6.4 :570)")
    return count


def _assert_track(track: str, *, decoder_id: str) -> None:
    if track not in TRACKS_V543:
        raise V543RegistryError(
            "V543_REGISTRY_TRACK",
            f"decoder {decoder_id!r}: track must be one of "
            f"{list(TRACKS_V543)}, got {track!r}")


def track_of_decoder_id(decoder_id: str) -> str:
    """The track an id belongs to (``M*`` → move, ``P*`` → process)."""
    text = str(decoder_id)
    if text in MOVE_DECODER_IDS_V543:
        return TRACK_MOVE
    if text in PROCESS_DECODER_IDS_V543:
        return TRACK_PROCESS
    raise V543RegistryError(
        "V543_REGISTRY_UNKNOWN",
        f"unknown decoder id {text!r}; the campaign declares "
        f"{list(DECODER_IDS_V543)} (plan §6.2/§6.3)")


# ---------------------------------------------------------------------------
# §6.1 protocol enforcement (structure, not intent)
# ---------------------------------------------------------------------------
def _callable_parameters(func: Any) -> tuple[list[str], bool]:
    """Positional parameter names of ``func`` plus a ``*args``/``**kwargs`` flag."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):        # pragma: no cover - builtins
        return [], False
    names: list[str] = []
    variadic = False
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.POSITIONAL_ONLY,
                              parameter.POSITIONAL_OR_KEYWORD):
            names.append(parameter.name)
        elif parameter.kind in (parameter.VAR_POSITIONAL,
                                parameter.VAR_KEYWORD):
            variadic = True
    return names, variadic


def assert_decoder_protocol(
        decoder: Any,
        *,
        decoder_id: str,
        track: str,
) -> Any:
    """Enforce the §6.1 protocol AND its prohibitions on ``decoder``.

    Checks, in this order (each a typed
    :class:`V543DecoderProtocolError` / :class:`V543RegistryError`):

    1. ``track`` is legal and the id's track agrees with it
       (``V543_REGISTRY_TRACK``);
    2. ``forward`` exists, is callable, and takes EXACTLY ONE argument beyond
       ``self`` — the read-only track view (plan §6.1 :490/:496).  ``*args``
       is refused because an unbounded signature can smuggle a second input;
    3. no forbidden data/label/loss/selection method is defined
       (``V543_DECODER_PRIVATE_METHOD``; plan §6.1 :519);
    4. no forbidden attribute or submodule name is present
       (``V543_DECODER_PRIVATE_STATE``; plan §6.1 :519);
    5. a ``torch.nn.Module`` decoder is not a ``torch.compile`` wrapper
       (plan §11.3 禁止).

    Returns the decoder so call sites can write ``decoder = assert_...(...)``.
    """
    _assert_track(track, decoder_id=decoder_id)
    expected_track = track_of_decoder_id(decoder_id)
    if expected_track != track:
        raise V543RegistryError(
            "V543_REGISTRY_TRACK",
            f"decoder {decoder_id!r} belongs to the {expected_track!r} track "
            f"(plan §6.2/§6.3), but was registered for {track!r}")

    forward = getattr(decoder, "forward", None)
    if not callable(forward):
        raise V543DecoderProtocolError(
            "V543_DECODER_PROTOCOL",
            f"decoder {decoder_id!r} ({type(decoder).__name__}) has no "
            "callable forward(ctx); the §6.1 protocol is forward-only")
    bound = getattr(decoder, "forward", None)
    try:
        bound = decoder.forward
        names, variadic = _callable_parameters(bound)
    except Exception:                      # pragma: no cover - defensive
        names, variadic = [], False
    # A bound method already dropped ``self``; an unbound function keeps it.
    if names and names[0] in ("self", "cls") and not isinstance(
            inspect.getattr_static(type(decoder), "forward", None),
            staticmethod):
        names = names[1:]
    if variadic or len(names) != 1:
        raise V543DecoderProtocolError(
            "V543_DECODER_SIGNATURE",
            f"decoder {decoder_id!r}: forward must take exactly one argument "
            f"(the read-only track view), got {names}"
            f"{' plus *args/**kwargs' if variadic else ''} (plan §6.1 "
            ":490/:496 — a decoder that accepts more can read data or "
            "supervision)")

    for name in FORBIDDEN_DECODER_METHODS:
        if callable(getattr(decoder, name, None)):
            raise V543DecoderProtocolError(
                "V543_DECODER_PRIVATE_METHOD",
                f"decoder {decoder_id!r} defines {name}(); data reading, loss "
                "definition and checkpoint selection belong to v543_train.py "
                "(plan §6.1 :519)")

    for name in _decoder_state_names(decoder):
        if name in FORBIDDEN_DECODER_NAMES:
            raise V543DecoderProtocolError(
                "V543_DECODER_PRIVATE_STATE",
                f"decoder {decoder_id!r} exposes {name!r}; a decoder may hold "
                "no data/label/loss/selection state (plan §6.1 :519)")

    if isinstance(decoder, torch.nn.Module):
        from pes2ts_core.pesn.models.v542.interfaces import (
            assert_no_torch_compile,
        )

        assert_no_torch_compile(decoder)
    return decoder


def _decoder_state_names(decoder: Any) -> tuple[str, ...]:
    """Attribute / child-module / buffer names reachable on ``decoder``."""
    names: set[str] = set()
    try:
        names.update(str(key) for key in vars(decoder))
    except TypeError:                      # pragma: no cover - slots-only
        pass
    if isinstance(decoder, torch.nn.Module):
        for name, _module in decoder.named_children():
            names.add(str(name).split(".")[-1])
        for name, _buffer in decoder.named_buffers():
            names.add(str(name).split(".")[-1])
    return tuple(sorted(names))


# ---------------------------------------------------------------------------
# M0 / P0 — the zero-residual baselines (plan §6.2 :522, §6.3 :539)
# ---------------------------------------------------------------------------
class ZeroMoveDecoder(torch.nn.Module):
    """``M0``: the frozen baseline MOVE decoder, ``δu ≡ 0`` (plan §6.2 :522).

    "旧 D0，``δu=0``", 0 trainable params, NFE 1.  ``forward`` returns a
    DETACHED zero tensor with ``u0``'s shape, so the assembled displacement is
    exactly ``u = u0`` (the plan §5.8 zero-residual parity) and a training run
    of M0 exercises the whole pipeline while producing the frozen baseline
    numbers (plan §8.1 T0).
    """

    decoder_id: Final[str] = "M0"
    track: Final[str] = TRACK_MOVE
    model_version: Final[str] = "v543_m0_zero_residual_v1"

    def forward(self, ctx: MoveContextV1) -> torch.Tensor:
        if not isinstance(ctx, MoveContextV1):
            raise V543DecoderProtocolError(
                "V543_DECODER_CONTEXT_TYPE",
                "M0 takes a MoveContextV1 (plan §6.1); got "
                f"{type(ctx).__name__}")
        return torch.zeros_like(ctx.u0)

    @property
    def n_trainable_params(self) -> int:
        return count_trainable_params(self)


class ZeroProcessDecoder(torch.nn.Module):
    """``P0``: the frozen baseline PROCESS decoder, ``δq = δe = 0`` (§6.3 :539).

    "旧 q/endpoint head，``δq=δe=0``", 0 trainable params, NFE 1.  ``q`` and
    ``endpoint_logit`` therefore stay exactly the encoder's ``tanh(q0_raw)`` /
    ``e0`` (§5.8 zero-residual parity).
    """

    decoder_id: Final[str] = "P0"
    track: Final[str] = TRACK_PROCESS
    model_version: Final[str] = "v543_p0_zero_residual_v1"

    def forward(self, ctx: ProcessContextV1) -> ProcessResidual:
        if not isinstance(ctx, ProcessContextV1):
            raise V543DecoderProtocolError(
                "V543_DECODER_CONTEXT_TYPE",
                "P0 takes a ProcessContextV1 (plan §6.1); got "
                f"{type(ctx).__name__}")
        return ProcessResidual(
            delta_q_raw=torch.zeros_like(ctx.q0_raw),
            delta_e=torch.zeros_like(ctx.endpoint0_logit),
        )

    @property
    def n_trainable_params(self) -> int:
        return count_trainable_params(self)


# ---------------------------------------------------------------------------
# the §6.1 record + the registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DecoderSpec:
    """One registry entry — the 12-/13-field §6.1 record plus its factory.

    The record fields are plan §6.1 :503–517 in the plan's own order
    (:data:`REGISTRY_RECORD_FIELDS`); ``factory`` is the WP4 addition that
    makes the entry constructive ("registry.py | ``register``、``build``、
    ``describe``、``assert_budget``", plan §11.2), and is NOT part of the
    serialised record.
    """

    decoder_id: str
    track: str
    model_version: str
    factory: Callable[..., Any] = field(repr=False)
    equivariance: str = "polar"
    stochastic: bool = False
    nfe: int = 1
    zero_init: bool = True
    status: str = "proposed"
    checkpoint_sha256: str = ""
    config_sha256: str = ""
    acceptance: str = "not_evaluated"
    contract_version: str = DECODER_CONTEXT_SCHEMA_V543

    def __post_init__(self) -> None:
        _assert_track(self.track, decoder_id=self.decoder_id)
        if self.decoder_id not in DECODER_IDS_V543:
            raise V543RegistryError(
                "V543_REGISTRY_UNKNOWN",
                f"decoder id {self.decoder_id!r} is not declared by the "
                f"campaign (plan §6.2/§6.3: {list(DECODER_IDS_V543)})")
        if track_of_decoder_id(self.decoder_id) != self.track:
            raise V543RegistryError(
                "V543_REGISTRY_TRACK",
                f"decoder {self.decoder_id!r} is a "
                f"{track_of_decoder_id(self.decoder_id)} id but the spec "
                f"says track={self.track!r}")
        if self.equivariance not in EQUIVARIANCE_VALUES_V543:
            raise V543RegistryError(
                "V543_REGISTRY_EQUIVARIANCE",
                f"{self.decoder_id}: equivariance must be one of "
                f"{list(EQUIVARIANCE_VALUES_V543)}, got "
                f"{self.equivariance!r}")
        if self.status not in STATUS_VALUES_V543:
            raise V543RegistryError(
                "V543_REGISTRY_STATUS",
                f"{self.decoder_id}: status must be one of "
                f"{list(STATUS_VALUES_V543)}, got {self.status!r}")
        if int(self.nfe) < 1:
            raise V543RegistryError(
                "V543_REGISTRY_NFE",
                f"{self.decoder_id}: nfe must be >= 1, got {self.nfe}")
        if self.contract_version != DECODER_CONTEXT_SCHEMA_V543:
            raise V543RegistryError(
                "V543_REGISTRY_CONTRACT",
                f"{self.decoder_id}: contract_version must be "
                f"{DECODER_CONTEXT_SCHEMA_V543!r}, got "
                f"{self.contract_version!r}")
        if not callable(self.factory):
            raise V543RegistryError(
                "V543_REGISTRY_FACTORY",
                f"{self.decoder_id}: factory must be callable")

    def record(self, n_trainable_params: int) -> dict[str, Any]:
        """The §6.1 record payload (exactly :data:`REGISTRY_RECORD_FIELDS`)."""
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


class DecoderRegistry:
    """Register / build / describe the campaign's decoders (plan §11.2)."""

    def __init__(self) -> None:
        self._specs: dict[str, DecoderSpec] = {}
        self._counts: dict[str, int] = {}

    # -- registration -------------------------------------------------------
    def register(self, spec: DecoderSpec) -> DecoderSpec:
        """Register ``spec`` and assert its parameter budget at CONSTRUCTION.

        The spec is registered only after a decoder instance could actually be
        built and counted — a factory that raises, or whose product violates
        plan §6.4, never enters the registry.
        """
        if spec.decoder_id in self._specs:
            raise V543RegistryError(
                "V543_REGISTRY_DUPLICATE",
                f"decoder {spec.decoder_id!r} is already registered "
                f"(model_version {self._specs[spec.decoder_id].model_version!r})")
        decoder = assert_decoder_protocol(
            spec.factory(), decoder_id=spec.decoder_id, track=spec.track)
        count = count_trainable_params(decoder)
        assert_budget(count, decoder_id=spec.decoder_id, track=spec.track)
        self._specs[spec.decoder_id] = spec
        self._counts[spec.decoder_id] = int(count)
        return spec

    # -- lookup -------------------------------------------------------------
    def spec(self, decoder_id: str) -> DecoderSpec:
        """The registered spec of ``decoder_id`` (typed refusal if absent)."""
        spec = self._specs.get(str(decoder_id))
        if spec is None:
            known = sorted(self._specs)
            raise V543RegistryError(
                "V543_REGISTRY_UNKNOWN",
                f"decoder {decoder_id!r} is not registered (registered: "
                f"{known}); WP4 registers only the M0/P0 zero-parameter "
                "baselines — M1–M5/P1–P5 arrive with WP5/WP6")
        return spec

    def contains(self, decoder_id: str) -> bool:
        return str(decoder_id) in self._specs

    def ids(self, *, track: str | None = None) -> tuple[str, ...]:
        """Registered ids, optionally restricted to one track."""
        names = sorted(self._specs)
        if track is None:
            return tuple(names)
        _assert_track(track, decoder_id="<query>")
        return tuple(name for name in names
                     if self._specs[name].track == track)

    # -- construction -------------------------------------------------------
    def build(
            self,
            decoder_id: str,
            *,
            track: str | None = None,
            **kwargs: Any,
    ) -> Any:
        """Construct ``decoder_id``, re-asserting protocol + budget.

        ``track`` (optional) must agree with the registered track — a
        cross-track build is ``V543_REGISTRY_TRACK``.
        """
        spec = self.spec(decoder_id)
        if track is not None:
            _assert_track(track, decoder_id=str(decoder_id))
            if str(track) != spec.track:
                raise V543RegistryError(
                    "V543_REGISTRY_TRACK",
                    f"decoder {decoder_id!r} is registered for the "
                    f"{spec.track!r} track; requested {track!r}")
        decoder = assert_decoder_protocol(
            spec.factory(**kwargs), decoder_id=spec.decoder_id,
            track=spec.track)
        count = count_trainable_params(decoder)
        assert_budget(count, decoder_id=spec.decoder_id, track=spec.track)
        declared = getattr(decoder, "n_trainable_params", None)
        if declared is not None and int(declared) != count:
            raise V543RegistryError(
                "V543_REGISTRY_PARAM_REPORT",
                f"decoder {decoder_id!r} reports n_trainable_params="
                f"{declared} but sum(p.numel())={count}; the record must be "
                "the measured count (plan §6.1 :509)")
        return decoder

    # -- description --------------------------------------------------------
    def describe(self, decoder_id: str, *, build: bool = False) -> dict[str, Any]:
        """The §6.1 record of ``decoder_id`` (JSON-able).

        ``build=True`` constructs a fresh instance to re-measure the count;
        otherwise the count of the registration-time instance is reported
        (``n_trainable_params_source`` says which, so the record never
        pretends a measurement it did not make).
        """
        spec = self.spec(decoder_id)
        if build:
            count = count_trainable_params(
                assert_decoder_protocol(
                    spec.factory(), decoder_id=spec.decoder_id,
                    track=spec.track))
            source = "measured_at_describe"
        else:
            count = int(self._counts[spec.decoder_id])
            source = "measured_at_register"
        record = spec.record(count)
        record.update({
            "schema": V543_REGISTRY_SCHEMA,
            "version": V543_REGISTRY_VERSION,
            "precision_version": PRECISION_VERSION_V543,
            "n_trainable_params_source": source,
            "is_baseline": spec.decoder_id in BASELINE_DECODER_BY_TRACK.values(),
            "param_band": {
                "target": PARAM_TARGET_V543,
                "band_pct": PARAM_BAND_PCT_V543,
                "min": PARAM_BAND_MIN_V543,
                "max": PARAM_BAND_MAX_V543,
                "hard_cap": PARAM_HARD_CAP_V543,
            },
            "record_fields": list(REGISTRY_RECORD_FIELDS),
        })
        return record

    def describe_all(self) -> dict[str, Any]:
        """Registry-wide payload (metrics.json input, plan §7.2 :733)."""
        return {
            "schema": V543_REGISTRY_SCHEMA,
            "version": V543_REGISTRY_VERSION,
            "record_fields": list(REGISTRY_RECORD_FIELDS),
            "declared_ids": list(DECODER_IDS_V543),
            "registered": [self.describe(name) for name in self.ids()],
            "baselines": dict(BASELINE_DECODER_BY_TRACK),
        }


# ---------------------------------------------------------------------------
# the process-wide registry (M0/P0 only — the WP4 boundary)
# ---------------------------------------------------------------------------
def _build_zero_move(**_: Any) -> ZeroMoveDecoder:
    return ZeroMoveDecoder()


def _build_zero_process(**_: Any) -> ZeroProcessDecoder:
    return ZeroProcessDecoder()


M0_SPEC: Final[DecoderSpec] = DecoderSpec(
    decoder_id="M0",
    track=TRACK_MOVE,
    model_version="v543_m0_zero_residual_v1",
    factory=_build_zero_move,
    equivariance="polar",
    stochastic=False,
    nfe=1,
    zero_init=True,
    status="proposed",
    acceptance="baseline_pending_T0",
)
P0_SPEC: Final[DecoderSpec] = DecoderSpec(
    decoder_id="P0",
    track=TRACK_PROCESS,
    model_version="v543_p0_zero_residual_v1",
    factory=_build_zero_process,
    equivariance="invariant",
    stochastic=False,
    nfe=1,
    zero_init=True,
    status="proposed",
    acceptance="baseline_pending_T0",
)


def default_registry() -> DecoderRegistry:
    """A registry holding exactly the two WP4 baselines (fresh instance)."""
    registry = DecoderRegistry()
    registry.register(M0_SPEC)
    registry.register(P0_SPEC)
    return registry


#: The process-wide registry.  Constructed once at import (it only ever builds
#: two parameter-free modules), so ``build_decoder("M0")`` in a trainer is the
#: same registration WP7's ``describe`` reports.
REGISTRY_V543: Final[DecoderRegistry] = default_registry()


def register_decoder(spec: DecoderSpec) -> DecoderSpec:
    """Register one spec into :data:`REGISTRY_V543` (plan §11.2 ``register``)."""
    return REGISTRY_V543.register(spec)


def build_decoder(decoder_id: str, *, track: str | None = None,
                  **kwargs: Any) -> Any:
    """Build a registered decoder (plan §11.2 ``build``)."""
    return REGISTRY_V543.build(decoder_id, track=track, **kwargs)


def describe_decoder(decoder_id: str, *, build: bool = False
                     ) -> dict[str, Any]:
    """Describe a registered decoder (plan §11.2 ``describe``)."""
    return REGISTRY_V543.describe(decoder_id, build=build)


def baseline_decoder_id(track: str) -> str:
    """The zero-parameter baseline id of ``track`` (``M0`` / ``P0``)."""
    _assert_track(track, decoder_id="<baseline>")
    return BASELINE_DECODER_BY_TRACK[track]


def build_baseline(track: str) -> Any:
    """Build the track's zero-residual baseline (plan §4.3 :226)."""
    return build_decoder(baseline_decoder_id(track), track=track)


def assert_parameter_matching_report() -> dict[str, Any]:
    """JSON-able summary of the §6.4 budget rules (report / metrics input)."""
    return {
        "track_target": PARAM_TARGET_V543,
        "band_pct": PARAM_BAND_PCT_V543,
        "band_min": PARAM_BAND_MIN_V543,
        "band_max": PARAM_BAND_MAX_V543,
        "band_half_width": param_band_tolerance(),
        "hard_cap_per_candidate": PARAM_HARD_CAP_V543,
        "baseline_params": PARAM_BASELINE_V543,
        "encoder_joint_params": ENCODER_JOINT_PARAMS_V543,
        "global_hard_cap": PARAM_GLOBAL_CAP_V543,
        "final_target_total": (
            ENCODER_JOINT_PARAMS_V543 + PARAM_TARGET_V543 * 2),
        "final_band_max_total": (
            ENCODER_JOINT_PARAMS_V543 + PARAM_BAND_MAX_V543 * 2),
        "assertion": "sum(p.numel()) at construction (plan §6.4 :569)",
        "forbidden": "dead weights / unused bias padding (plan §6.4 :570)",
    }


__all__ = [
    "BASELINE_DECODER_BY_TRACK",
    "CONTEXT_TYPE_BY_TRACK",
    "DECODER_IDS_V543",
    "ENCODER_JOINT_PARAMS_V543",
    "EQUIVARIANCE_VALUES_V543",
    "FORBIDDEN_DECODER_METHODS",
    "FORBIDDEN_DECODER_NAMES",
    "M0_SPEC",
    "MOVE_DECODER_IDS_V543",
    "MoveDecoderProtocol",
    "P0_SPEC",
    "PARAM_BAND_MAX_V543",
    "PARAM_BAND_MIN_V543",
    "PARAM_BAND_PCT_V543",
    "PARAM_BASELINE_V543",
    "PARAM_GLOBAL_CAP_V543",
    "PARAM_HARD_CAP_V543",
    "PARAM_TARGET_V543",
    "PROCESS_DECODER_IDS_V543",
    "PROTOCOL_BY_TRACK",
    "ProcessDecoderProtocol",
    "REGISTRY_RECORD_FIELDS",
    "REGISTRY_V543",
    "STATUS_VALUES_V543",
    "TRACKS_V543",
    "TRACK_MOVE",
    "TRACK_PROCESS",
    "DecoderRegistry",
    "DecoderSpec",
    "V543DecoderProtocolError",
    "V543_REGISTRY_SCHEMA",
    "V543_REGISTRY_VERSION",
    "V543RegistryError",
    "ZeroMoveDecoder",
    "ZeroProcessDecoder",
    "assert_budget",
    "assert_decoder_protocol",
    "assert_parameter_matching_report",
    "baseline_decoder_id",
    "build_baseline",
    "build_decoder",
    "count_trainable_params",
    "default_registry",
    "describe_decoder",
    "param_band_tolerance",
    "register_decoder",
    "track_of_decoder_id",
]
