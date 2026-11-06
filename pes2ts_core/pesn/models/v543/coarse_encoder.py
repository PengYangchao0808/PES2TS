"""V5.4.3 ``CoarseEncoder`` — the A0 (V5.4.2) encoder adapter of WP1.

Plan §4.1 (component table: ``CoarseEncoder`` = "A0 EGNN + D0 + P0；产生
``z``、``h``、``edge_msg``、``u0``、``q0_raw``、``e0``" / 冻结 / 57,619
params), §5.8 (A0 迁移与零残差 parity), §7.1 S0-A (阻断门), §11.1/§11.2
(``coarse_encoder.py``: "A0 strict load + 三个 hook + ``u0/q0_raw/e0``";
API ``CoarseEncoder.load(checkpoint, seed)`` / ``forward(batch) ->
CoarsePacket``).

Design decisions and their plan basis
-------------------------------------

1. **COMPOSITION, not inheritance.**  This class *contains*
   :class:`~pes2ts_core.pesn.models.v542.backbone_a0.A0LegacyBackbone`
   (``self.backbone``), the shared
   :class:`~pes2ts_core.pesn.models.v542.decoder.DirectMoveDecoder`
   (``self.move_decoder``) and
   :class:`~pes2ts_core.pesn.models.key_atom_heads_v3.ReactionProcessHeadV53MP`
   (``self.process_head``).  Reasons:
   ``A0LegacyBackbone`` is itself a ``StarEGNNV3`` subclass whose forward
   already replaces the legacy one, so inheriting it again would either
   shadow that forward or force a second copy of the hook discipline.  Far
   more important, plan §11.2/§11.3 freeze the A0 state-dict surface: the
   frozen loader mapping (``load_a0_strict``) targets exactly
   ``backbone.*`` / ``move_decoder.w_v.*`` / ``process_head.*`` keys.  A
   composition root yields those three names verbatim, so the encoder is
   strict-loadable by the FROZEN V5.4.2 loader with no mapping override and
   no key renaming — ``strict=False`` is never needed (§11.3 禁止).
   Composition also keeps ``StarEGNNV3``'s parameter tree untouched (no
   inherited ``w_v`` of our own, no polluted submodule names);
2. **Two REAL checkpoint payload layouts, discriminated explicitly.**
   ``V543_M0_Sxx ← V542_A0_Sxx`` (plan §7.1) hands the encoder a
   ``V542Model`` joint state dict whose ``w_v`` dual-write is ALREADY
   materialized — 45 keys, ``backbone.w_v.*`` AND ``move_decoder.w_v.*``
   both present (schema ``pesn_v542_train_v1``).  The FROZEN
   ``a0_mapping_table()`` is built for the legacy 43-key
   ``KeyAtomV53MPModel`` joint dict (``backbone.w_v.*`` as the SOURCE, the
   dual-write as the mapping), so feeding it a 45-key payload is a
   ``V542_LOAD_UNEXPECTED`` on ``move_decoder.w_v.*``.  :meth:`CoarseEncoder.load`
   therefore classifies the key set FIRST
   (:func:`classify_payload_layout`, typed ``V543_CKPT_LAYOUT`` on anything
   else) and picks one of exactly two mapping tables:

   * ``legacy_v53mp_joint_43`` → the FROZEN ``a0_mapping_table()``
     (identity + declared dual-write, 43 → 45) — behaviour unchanged;
   * ``v542_a0_joint_45`` → an explicit 45 → 45 IDENTITY table
     (:func:`identity_mapping_table`; no invented aliasing, the two ``w_v``
     slot families already point at ONE shared tensor object).

   BOTH tables are handed to ``load_a0_strict``, so the complete frozen
   audit — missing / unexpected / shape / dtype / post-load ``torch.equal``
   + the declared dual-write equality — still runs in either mode, and the
   resolved ``payload_layout`` / ``mapping_mode`` / ``mapping_table_sha256``
   are recorded in the load audit.  ``strict=False`` remains unavailable;
3. **Three hooks, exactly once each** (plan §11.1 "三个 hook"; §11.5
   ``test_v543_hook_fires_once``): ``backbone.message_mlp[-1]`` →
   ``edge_msg`` (owned by ``A0LegacyBackbone``; the encoder registers an
   independent counting hook to prove the count), ``backbone.norm_h[-1]``
   → ``node_h`` (NEW in V543, plan §5.8), ``process_head.progress_head.fc2``
   → ``q0_raw`` (plan §5.2/§5.8).  All three are temporary: removed in a
   ``finally`` (so a typed-invalid abort cannot leak them) and asserted to
   have fired EXACTLY once — the ``backbone_a0.py``
   ``V542_A0_HOOK`` discipline, extended to three points;
4. **``q0_raw`` outlet lives HERE, not in its own module**: plan §11.1
   assigns the file list and §11.2 gives ``coarse_encoder.py``
   "``z/h/edge_msg/u0/q0_raw/e0``; hook 计数" — a separate module would
   invent a file the plan does not freeze.  The outlet is
   :func:`single_fire_hook` + :func:`assert_q0_raw_matches_q`, both
   exported so WP2/WP4 reuse the identical capture;
5. **``q`` is still produced by the head**, never by re-tanh'ing
   ``q0_raw`` (plan §5.8: "``q`` 仍由 ``process_head(z)`` 产生；两者逐元素
   验证").  The packet stores the RAW ``q0_raw`` and the encoder refuses
   (``V543_PARITY_Q0_RAW_TANH``) unless ``tanh(q0_raw)`` equals the head's
   ``q`` at the plan §5.8 tolerance (CPU bitwise / GPU ≤1e-6);
6. **No detach anywhere** (plan §5.8 last bullet): all three captured
   tensors keep their autograd graph, so a future ``joint_finetune`` mode
   can backprop through the encoder;
7. **RNG discipline — deliberately STRICTER than ``v542/loader.py:75``.**
   :meth:`CoarseEncoder.load` builds the module inside
   :func:`forked_global_rng`, i.e. inside
   ``torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))``
   on a CUDA host (``devices=[]`` when CUDA is absent) wrapped around an
   explicit ``torch.manual_seed(seed)``.  Construction stays
   seed-deterministic AND every global torch generator (CPU + every
   VISIBLE CUDA device) is restored byte-identically, so an encoder load
   can never perturb a training stream.

   **Why the V5.4.2 pattern is not copied verbatim** (the difference is
   intentional; ``v542/loader.py`` is frozen by plan §11.3 and stays
   untouched): ``loader.py`` uses ``fork_rng(devices=[])`` with the
   rationale "since no CUDA device is touched by the construction, the
   CUDA RNG states too [are restored]".  That rationale holds only while
   the guarded body does not call ``torch.manual_seed``.
   ``torch.manual_seed`` seeds the CPU generator *and* calls
   ``torch.cuda.manual_seed_all``, whereas ``fork_rng(devices=[])``
   snapshots/restores the CPU state ONLY — so the explicit seed LEAKS into
   every CUDA generator state.  Observed on the ``pes2ts-gpu`` host: right
   after a load, ``torch.cuda.get_rng_state_all()`` returned a freshly
   seeded ``[17, 0, 0, …]`` instead of the pre-load states (caught by
   ``tests/pesn/test_v543_encoder_a0.py::test_load_never_perturbs_global_rng``).
   V5.4.3 cannot simply drop the seed — the build must be reproducible per
   ``seed`` for the identity/cache layer (plan §5.2 ``encoder_ckpt_sha256``,
   §5.7 cache keys) — so the fork is WIDENED to every visible CUDA device
   instead.  An explicit device LIST (not ``devices=None``) is used so the
   snapshot set is visible at the call site and the CUDA-absent degradation
   to ``[]`` is locally provable, rather than relying on torch's private
   "``None`` means all devices" default.  The RNG-fork policy is
   independently unit-tested by ``test_forked_global_rng_restores_all_
   generators``.
8. **Supervision cannot enter** (plan §5.6 / §4.4): any
   ``SUPERVISION_KEYS_V3`` member in the batch is refused with
   ``V543_SUPERVISION_LEAK`` BEFORE any compute; every other
   out-of-contract key keeps falling through to the legacy
   ``check_forward_view_keys_v3`` (``FORWARD_VIEW_LEAK``).

Forbidden here (plan §11.3): modifying a V542/V53MP file, copying ``w_v``,
re-training the encoder, ``strict=False``, ``torch.compile``."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn as nn

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    EDGE_INDEX_KEY,
    GRAPH_CV_DELTA_KEY,
    NODE_DEGK_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    check_forward_view_keys_v3,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import ReactionProcessHeadV53MP
from pes2ts_core.pesn.models.v542.backbone_a0 import A0LegacyBackbone
from pes2ts_core.pesn.models.v542.decoder import (
    DirectMoveDecoder,
    assert_decoder_a_ref_frozen,
)
from pes2ts_core.pesn.models.v542.identity import state_dict_sha256
from pes2ts_core.pesn.models.v542.interfaces import (
    assert_a0_param_invariants,
    assert_no_torch_compile,
    validate_backbone_output,
)
from pes2ts_core.pesn.models.v542.loader import a0_mapping_table, load_a0_strict
from pes2ts_core.pesn.models.v543.contracts import (
    COARSE_PACKET_KEYS,
    COARSE_PACKET_SCHEMA,
    ENCODER_ACTIVE_PARAMS_V543,
    ENCODER_JOINT_PARAMS_V543,
    ENCODER_LOAD_AUDIT_SCHEMA,
    HIDDEN_V543,
    HOOK_FIRES_PER_FORWARD,
    HOOK_LABEL_EDGE_MSG,
    HOOK_LABEL_NODE_H,
    HOOK_LABEL_Q0_RAW,
    HOOK_LABELS,
    HOOK_MODULE_PATHS,
    MAPPING_MODE_FROZEN,
    MAPPING_MODE_IDENTITY,
    N_DECODER_PARAMS_V543,
    N_PROCESS_PARAMS_V543,
    PARITY_ATOL_CPU,
    PARITY_ATOL_GPU,
    PAYLOAD_LAYOUT_LEGACY_43,
    PAYLOAD_LAYOUT_V542_A0_45,
    SUPERVISION_KEYS_V543,
    V543_ENCODER_VERSION,
    V543_SCHEMA_VERSION,
    CoarsePacket,
    V543ContractError,
    V543Error,
    V543HookError,
    V543InitError,
    V543LoadError,
    V543ParamBudgetError,
    V543ParityError,
    V543SupervisionLeakError,
    validate_coarse_packet,
)

#: Forward-hook capture label -> module attribute path (audit surface).
HOOK_PATH_BY_LABEL: Final[dict[str, str]] = dict(HOOK_MODULE_PATHS)


# ---------------------------------------------------------------------------
# global-RNG fork (CPU + every VISIBLE CUDA device; see module docstring #7)
# ---------------------------------------------------------------------------
def visible_cuda_devices() -> list[int]:
    """Indices of every CUDA device the process can see (``[]`` without CUDA).

    Used as ``fork_rng(devices=...)``: the set must cover exactly the
    generators ``torch.manual_seed`` also seeds (CPU + all CUDA devices),
    otherwise the explicit construction seed leaks into the training
    stream.  A CUDA stack that is present but unusable degrades to ``[]``
    (CPU-only fork) instead of crashing a load.
    """
    try:
        if torch.cuda.is_available():
            return list(range(int(torch.cuda.device_count())))
    except Exception:  # pragma: no cover - driver / CUDA-init failures
        return []
    return []


@contextlib.contextmanager
def forked_global_rng(*, seed: int | None = None) -> Iterator[list[int]]:
    """Run a body with the GLOBAL torch RNG forked; optionally seed inside.

    Snapshots and restores the CPU generator AND the generator of every
    visible CUDA device (``devices=[]`` on a CPU-only host), so the body —
    including an explicit ``torch.manual_seed(seed)`` — can never move the
    global RNG stream.  Yields the forked device list for audit records.

    This is the V5.4.3 fix for the leak documented in the module docstring
    (#7): ``torch.manual_seed`` seeds ``torch.cuda.manual_seed_all`` too,
    while the V5.4.2 ``fork_rng(devices=[])`` pattern only restores the CPU
    state.
    """
    devices = visible_cuda_devices()
    with torch.random.fork_rng(devices=devices):
        if seed is not None:
            torch.manual_seed(int(seed))
        yield devices


# ---------------------------------------------------------------------------
# checkpoint payload layouts (two REAL forms; plan §7.1 S0-A + §11.3)
# ---------------------------------------------------------------------------
def identity_mapping_table(
        state: Mapping[str, torch.Tensor]) -> dict[str, list[str]]:
    """Explicit 45 → 45 identity mapping (every source key maps to itself).

    Used for a ``v542_a0_joint_45`` payload, whose ``w_v`` dual-write is
    already materialized: the ``backbone.w_v.*`` rows feed
    ``backbone.w_v.*`` and the ``move_decoder.w_v.*`` rows feed the SAME
    shared tensor object, so no aliasing is needed or invented.  The table
    is still handed to :func:`~pes2ts_core.pesn.models.v542.loader.
    load_a0_strict`, i.e. the full frozen audit (missing / unexpected /
    shape / dtype / post-load ``torch.equal``) keeps running — this is not
    a bypass and never becomes ``strict=False``.
    """
    return {str(key): [str(key)] for key in sorted(state)}


def classify_payload_layout(
        state: Mapping[str, torch.Tensor], *, target_keys: set[str]) -> str:
    """Classify a checkpoint state dict into a frozen A0 payload layout.

    Args:
        state: the source state dict (``payload["model"]`` or raw).
        target_keys: the target module's ``state_dict()`` key set (the A0
            joint layout, 45 keys).

    Returns :data:`PAYLOAD_LAYOUT_LEGACY_43` or
    :data:`PAYLOAD_LAYOUT_V542_A0_45`.  Any other key set is a TYPED
    refusal (``V543_CKPT_LAYOUT``) that names the drift against BOTH
    candidates — a layout is never guessed, and a V542 45-key checkpoint is
    never silently fed to the legacy 43-key mapping (which would report
    ``V542_LOAD_UNEXPECTED: ['move_decoder.w_v.*']``).
    """
    keys = set(state)
    legacy_keys = set(a0_mapping_table())
    if keys == legacy_keys:
        return PAYLOAD_LAYOUT_LEGACY_43
    if keys == set(target_keys):
        return PAYLOAD_LAYOUT_V542_A0_45
    raise V543LoadError(
        "V543_CKPT_LAYOUT",
        f"checkpoint with {len(keys)} keys matches neither frozen A0 payload "
        f"layout: vs {PAYLOAD_LAYOUT_LEGACY_43} ({len(legacy_keys)} keys) "
        f"missing={sorted(legacy_keys - keys)} "
        f"extra={sorted(keys - legacy_keys)}; vs "
        f"{PAYLOAD_LAYOUT_V542_A0_45} ({len(set(target_keys))} keys) "
        f"missing={sorted(set(target_keys) - keys)} "
        f"extra={sorted(keys - set(target_keys))}")


def resolve_a0_mapping(
        state: Mapping[str, torch.Tensor], *, target_keys: set[str],
) -> tuple[str, dict[str, list[str]] | None]:
    """``(layout, mapping_table)`` for one checkpoint (typed on ambiguity).

    ``mapping_table is None`` means "use the FROZEN ``a0_mapping_table()``"
    (legacy 43-key dual-write path, unchanged behaviour); otherwise the
    explicit identity table for the 45-key V542 A0 form.

    Exposed so the parity harness can load the SAME payload into the frozen
    ``V542Model`` reference through the same mapping (one checkpoint, one
    mapping, two targets).
    """
    layout = classify_payload_layout(state, target_keys=target_keys)
    if layout == PAYLOAD_LAYOUT_LEGACY_43:
        return layout, None
    return layout, identity_mapping_table(state)


# ---------------------------------------------------------------------------
# one-shot forward-hook capture (the ``q0_raw`` outlet; plan §5.8)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def single_fire_hook(
        module: nn.Module, label: str) -> Iterator[list[torch.Tensor]]:
    """Temporary forward hook that must fire EXACTLY once per ``with`` body.

    Yields the capture list (empty at entry).  The hook is removed in a
    ``finally`` — including when the body raises a typed-invalid abort — and
    the caller asserts ``len(captured) == HOOK_FIRES_PER_FORWARD`` via
    :func:`assert_single_fire` (plan §5.8: "hook 必须在 ``finally`` 中移除，
    并断言恰好触发一次").

    The captured output is NEVER detached: plan §5.8 requires the autograd
    graph to stay live through every hook.
    """
    captured: list[torch.Tensor] = []

    def _capture(_module: nn.Module, _inputs: tuple, output: Any) -> None:
        captured.append(output)

    handle = module.register_forward_hook(_capture)
    try:
        yield captured
    finally:
        handle.remove()


def assert_single_fire(captured: list[torch.Tensor], label: str) -> torch.Tensor:
    """Assert the capture list has exactly one entry; return that tensor.

    Typed refusal ``V543_HOOK_FIRE_COUNT`` (the ``V542_A0_HOOK`` discipline
    of ``backbone_a0.py``, generalized to all three V543 hooks).
    """
    if len(captured) != HOOK_FIRES_PER_FORWARD:
        raise V543HookError(
            "V543_HOOK_FIRE_COUNT",
            f"{label} hook fired {len(captured)} time(s), expected exactly "
            f"{HOOK_FIRES_PER_FORWARD} "
            f"(module {HOOK_PATH_BY_LABEL.get(label, label)})")
    return captured[0]


# ---------------------------------------------------------------------------
# plan §5.8 parity primitives (CPU bitwise / GPU ≤1e-6; never loosened)
# ---------------------------------------------------------------------------
def parity_atol(*tensors: torch.Tensor, atol: float | None = None) -> float:
    """The plan §5.8 tolerance for these tensors (explicit ``atol`` wins).

    CPU comparisons are BITWISE (``atol = 0.0``); any non-CPU tensor means
    the GPU leg, capped at ``1e-6``.  A mixed-device set is a typed refusal
    (plan §5.8: "zero-residual parity 必须在同一 dtype/device 下比较").
    """
    if atol is not None:
        return float(atol)
    devices = {t.device.type for t in tensors}
    if len(devices) > 1:
        raise V543ParityError(
            "V543_PARITY_DEVICE",
            f"parity requires one device (got {sorted(devices)})")
    return PARITY_ATOL_CPU if devices == {"cpu"} else PARITY_ATOL_GPU


def compare_parity(
        name: str,
        actual: torch.Tensor,
        reference: torch.Tensor,
        *,
        atol: float | None = None,
        context: str = "",
) -> dict[str, object]:
    """Compare two tensors at the plan §5.8 tolerance; return an audit row.

    ``bitwise`` is ``torch.equal``; ``ok`` is ``bitwise or max_abs_diff <=
    atol``.  A shape/dtype/device mismatch or an out-of-tolerance element is
    a typed refusal (``V543_PARITY_<NAME>`` / ``V543_PARITY_SHAPE`` /
    ``V543_PARITY_DTYPE``) — plan §7.1 S0-A: "不得用「近似相等」放行".
    """
    code = f"V543_PARITY_{name.upper()}"
    if tuple(actual.shape) != tuple(reference.shape):
        raise V543ParityError(
            "V543_PARITY_SHAPE",
            f"{name}: shape {list(actual.shape)} != {list(reference.shape)}"
            + (f" ({context})" if context else ""))
    if actual.dtype != reference.dtype:
        raise V543ParityError(
            "V543_PARITY_DTYPE",
            f"{name}: dtype {actual.dtype} != {reference.dtype}"
            + (f" ({context})" if context else ""))
    tolerance = parity_atol(actual, reference, atol=atol)
    left, right = actual.detach(), reference.detach()
    bitwise = bool(torch.equal(left, right))
    if int(left.numel()):
        max_abs_diff = float(
            (left.to(torch.float64) - right.to(torch.float64)).abs().max())
    else:  # pragma: no cover - empty tensors compare bitwise
        max_abs_diff = 0.0
    ok = bitwise or max_abs_diff <= tolerance
    if not ok:
        raise V543ParityError(
            code,
            f"{name} parity BROKEN: max_abs_diff={max_abs_diff:.3e} > "
            f"atol={tolerance:.1e} (device={left.device.type}, "
            f"n={int(left.numel())})" + (f" ({context})" if context else ""))
    return {
        "name": name,
        "bitwise": bitwise,
        "max_abs_diff": max_abs_diff,
        "atol": float(tolerance),
        "device": left.device.type,
        "n_elements": int(left.numel()),
        "ok": True,
    }


def assert_q0_raw_matches_q(
        q0_raw: torch.Tensor,
        q: torch.Tensor,
        *,
        atol: float | None = None,
        context: str = "",
) -> dict[str, object]:
    """Plan §5.8 equality 4: ``tanh(q0_raw) == q`` (the head's own ``q``).

    ``q`` is what ``process_head(z)`` produced; this helper only proves the
    hook-captured pre-tanh scalar reproduces it.  Tolerance: CPU bitwise,
    GPU ≤1e-6 (never loosened).
    """
    return compare_parity("q0_raw_tanh", torch.tanh(q0_raw), q,
                          atol=atol, context=context)


# ---------------------------------------------------------------------------
# the encoder
# ---------------------------------------------------------------------------
class CoarseEncoder(nn.Module):
    """A0 (V5.4.2) encoder, frozen, behind the V5.4.3 ``CoarsePacket`` API.

    Args:
        backbone: an EXISTING :class:`A0LegacyBackbone` shared BY REFERENCE
            (same policy as ``V542Model``); ``None`` builds a fresh adapter
            around ``move_decoder``.  When provided it MUST share the
            decoder's ``w_v`` module object (typed refusal
            ``V543_ENCODER_WV_NOT_SHARED``).
        move_decoder: the SHARED :class:`DirectMoveDecoder` (frozen
            ``a_ref = 0.08``, gate-1 B1); ``None`` builds a fresh one.
        process_head: an EXISTING :class:`ReactionProcessHeadV53MP` shared
            by reference; ``None`` builds a fresh one.  This head owns both
            PROCESS scalars: ``q`` (→ ``tanh(q0_raw)`` parity) and
            ``endpoint_logit`` (→ ``e0``).

    The frozen A0 counts are asserted at construction: backbone active
    55,392 / decoder 49 / process 2,178 / joint 57,619 (plan §6.4).  The
    class also REFUSES a ``torch.compile``-wrapped tree (plan §11.3 禁止).
    """

    #: Registered encoder-adapter version (plan §5.2 identity sources).
    v543_version: Final[str] = V543_ENCODER_VERSION
    #: Registered model-layer schema.
    schema: Final[str] = V543_SCHEMA_VERSION
    #: Schema of :meth:`forward`'s return value.
    packet_schema: Final[str] = COARSE_PACKET_SCHEMA

    def __init__(
            self,
            backbone: A0LegacyBackbone | None = None,
            *,
            move_decoder: DirectMoveDecoder | None = None,
            process_head: ReactionProcessHeadV53MP | None = None,
    ) -> None:
        super().__init__()
        decoder = (move_decoder if move_decoder is not None
                   else DirectMoveDecoder())
        if not isinstance(decoder, DirectMoveDecoder):
            raise V543InitError(
                "V543_ENCODER_DECODER_TYPE",
                "move_decoder must be the shared DirectMoveDecoder "
                f"(got {type(decoder).__name__})")
        # Gate-1 B1: the adopted decoder must carry the FROZEN a_ref=0.08.
        assert_decoder_a_ref_frozen(decoder)
        if backbone is not None:
            if not isinstance(backbone, A0LegacyBackbone):
                raise V543InitError(
                    "V543_ENCODER_BACKBONE_TYPE",
                    f"backbone must be an A0LegacyBackbone, got "
                    f"{type(backbone).__name__}")
            if backbone.w_v is not decoder.w_v:
                raise V543InitError(
                    "V543_ENCODER_WV_NOT_SHARED",
                    "backbone.w_v is not the decoder's w_v module object — "
                    "the shared-w_v invariant (backbone.w_v is "
                    "move_decoder.w_v) must hold; pass the decoder that owns "
                    "the backbone")
            self.backbone = backbone
        else:
            self.backbone = A0LegacyBackbone(decoder)
        self.move_decoder = decoder
        self.process_head = (process_head if process_head is not None
                             else ReactionProcessHeadV53MP())
        if not isinstance(self.process_head, ReactionProcessHeadV53MP):
            raise V543InitError(
                "V543_ENCODER_PROCESS_HEAD",
                "process_head must be the frozen ReactionProcessHeadV53MP "
                f"(got {type(self.process_head).__name__})")

        # Structural refusal of torch.compile (plan §11.3; never called).
        assert_no_torch_compile(self)

        # -- frozen A0 invariants at build (plan §6.4) ----------------------
        active = int(self.backbone.active_param_count)
        joint = int(sum(int(p.numel()) for p in self.parameters()))
        if int(self.move_decoder.param_count) != N_DECODER_PARAMS_V543:
            raise V543ParamBudgetError(
                "V543_ENCODER_PARAMS",
                f"decoder has {self.move_decoder.param_count} params, "
                f"expected {N_DECODER_PARAMS_V543}")
        n_process = int(sum(int(p.numel())
                            for p in self.process_head.parameters()))
        if n_process != N_PROCESS_PARAMS_V543:
            raise V543ParamBudgetError(
                "V543_ENCODER_PARAMS",
                f"process_head has {n_process} params, expected "
                f"{N_PROCESS_PARAMS_V543}")
        if active != ENCODER_ACTIVE_PARAMS_V543:
            raise V543ParamBudgetError(
                "V543_ENCODER_PARAMS",
                f"backbone active params {active} != "
                f"{ENCODER_ACTIVE_PARAMS_V543}")
        if joint != ENCODER_JOINT_PARAMS_V543:
            raise V543ParamBudgetError(
                "V543_ENCODER_PARAMS",
                f"encoder joint params {joint} != {ENCODER_JOINT_PARAMS_V543} "
                "(shared w_v counted once)")
        assert_a0_param_invariants(
            active=active, joint=joint)

        # -- audit counters (plain attributes: never params/buffers) -------
        self._forward_count: int = 0
        self._cumulative_hook_fires: dict[str, int] = {
            label: 0 for label in HOOK_LABELS}
        self._last_hook_fires: dict[str, int] = {label: 0 for label in HOOK_LABELS}
        self._load_audit: dict[str, object] | None = None

    # -- accounting ---------------------------------------------------------
    @property
    def active_param_count(self) -> int:
        """Trainable encoder params EXCLUDING the decoder-owned ``w_v``."""
        return int(self.backbone.active_param_count)

    @property
    def forward_count(self) -> int:
        """Number of completed :meth:`forward` calls on this instance.

        Plan §4.4 / §7.1 S0-A: exactly ONE encoder forward per current
        configuration; the harness asserts this counter per forward view.
        """
        return int(self._forward_count)

    @property
    def last_hook_fires(self) -> dict[str, int]:
        """Per-label hook fire counts of the LAST forward (all must be 1)."""
        return dict(self._last_hook_fires)

    @property
    def cumulative_hook_fires(self) -> dict[str, int]:
        """Per-label hook fire counts since construction / last reset."""
        return dict(self._cumulative_hook_fires)

    @property
    def load_audit(self) -> dict[str, object] | None:
        """The strict-load audit record of :meth:`load` (``None`` if fresh)."""
        return None if self._load_audit is None else dict(self._load_audit)

    def reset_counters(self) -> None:
        """Zero the forward/hook counters (per-forward-view audit helper)."""
        self._forward_count = 0
        self._cumulative_hook_fires = {label: 0 for label in HOOK_LABELS}
        self._last_hook_fires = {label: 0 for label in HOOK_LABELS}

    def describe(self) -> dict[str, object]:
        """JSON-able description of this encoder (parity-certificate input)."""
        audit = self._load_audit
        return {
            "schema": V543_SCHEMA_VERSION,
            "version": V543_ENCODER_VERSION,
            "packet_schema": COARSE_PACKET_SCHEMA,
            "packet_fields": list(COARSE_PACKET_KEYS),
            "hooks": [
                {"label": label, "module": path,
                 "fires_per_forward": HOOK_FIRES_PER_FORWARD}
                for label, path in HOOK_MODULE_PATHS],
            "params": {
                "backbone_active": int(self.backbone.active_param_count),
                "decoder": int(self.move_decoder.param_count),
                "process_head": int(sum(
                    p.numel() for p in self.process_head.parameters())),
                "joint": int(sum(
                    p.numel() for p in self.parameters())),
            },
            "encoder_ckpt_sha256": (
                None if audit is None else audit.get("encoder_ckpt_sha256")),
            "payload_layout": (
                None if audit is None else audit.get("payload_layout")),
            "mapping_mode": (
                None if audit is None else audit.get("mapping_mode")),
            "parity_atol": {"cpu": PARITY_ATOL_CPU, "gpu": PARITY_ATOL_GPU},
        }

    # -- strict load (plan §7.1 S0-A; ``strict=False`` is forbidden) --------
    @classmethod
    def load(
            cls,
            checkpoint: object,
            seed: int = 17,
            *,
            process_head: ReactionProcessHeadV53MP | None = None,
    ) -> "CoarseEncoder":
        """Build the encoder and strict-load an A0 joint checkpoint into it.

        Args:
            checkpoint: a checkpoint path (``str``/``Path``) OR an already
                loaded payload (``payload["model"]`` or a raw state dict).
            seed: construction seed; the build runs inside
                :func:`forked_global_rng`, i.e. ``torch.random.fork_rng``
                over the CPU generator AND every visible CUDA device, so the
                GLOBAL torch RNG stream is restored byte-identically (module
                docstring #7 documents why this is wider than the V5.4.2
                ``loader.py`` pattern).
            process_head: optional shared head (defaults to a fresh frozen
                one, exactly like the constructor).

        Returns the encoder; the audit of
        :func:`~pes2ts_core.pesn.models.v542.loader.load_a0_strict` is
        stored on :attr:`load_audit` and extended with
        ``encoder_ckpt_sha256`` (plan §5.2 identity source), the V543
        version/schema block, the resolved ``payload_layout`` /
        ``mapping_mode`` / ``mapping_table_sha256`` and the RNG-fork device
        set.

        Payload layouts (both go through the FULL ``load_a0_strict`` audit —
        missing / unexpected / shape / dtype / post-load ``torch.equal``;
        ``strict=False`` does not exist here):

        * ``legacy_v53mp_joint_43`` (V541/V53-MP joint state dict, 43 keys,
          ``backbone.w_v.*`` only) → the FROZEN ``a0_mapping_table()``
          identity + declared ``w_v`` dual-write (43 → 45);
        * ``v542_a0_joint_45`` (V5.4.2 ``V542Model`` joint state dict, 45
          keys, dual-write already materialized — the S0-A assets) → the
          explicit IDENTITY mapping (45 → 45);
        * anything else → typed ``V543_CKPT_LAYOUT`` refusal.

        Refusals: ``V543_CKPT_MISSING`` (path absent),
        ``V543_CKPT_PAYLOAD`` (unreadable payload), ``V543_CKPT_LAYOUT``
        (unknown key set), plus every typed ``V542_LOAD_*`` refusal of the
        frozen strict audit.
        """
        with forked_global_rng(seed=int(seed)) as rng_devices:
            encoder = cls(process_head=process_head)

        payload = cls._read_payload(checkpoint)
        state = cls._as_state_mapping(payload)
        layout, table = resolve_a0_mapping(
            state, target_keys=set(encoder.state_dict()))
        audit = load_a0_strict(encoder, payload, table)
        record: dict[str, object] = {
            "schema": ENCODER_LOAD_AUDIT_SCHEMA,
            "version": V543_ENCODER_VERSION,
            "ok": True,
            "strict": True,
            "seed": int(seed),
            "payload_layout": layout,
            "mapping_mode": (MAPPING_MODE_FROZEN if table is None
                             else MAPPING_MODE_IDENTITY),
            "encoder_ckpt_sha256": state_dict_sha256(state),
            "n_source_keys": int(len(state)),
            "n_mapped_targets": int(len(audit["loaded_keys"])),
            "torch_equal_verified": int(len(audit["torch_equal_verified"])),
            "mapping_table_sha256": audit["mapping_table_sha256"],
            "rng_fork": {"cpu": True, "cuda_devices": list(rng_devices)},
            "load_audit": audit,
        }
        encoder._load_audit = record
        return encoder

    @staticmethod
    def _read_payload(checkpoint: object) -> object:
        """Resolve a path or an in-memory mapping into a loadable payload."""
        if isinstance(checkpoint, Mapping):
            return checkpoint
        if isinstance(checkpoint, (str, Path)):
            path = Path(checkpoint)
            if not path.is_file():
                raise V543LoadError(
                    "V543_CKPT_MISSING",
                    f"A0 checkpoint not found: {path}")
            try:
                return torch.load(str(path), map_location="cpu",
                                  weights_only=False)
            except Exception as exc:  # pragma: no cover - io level
                raise V543LoadError(
                    "V543_CKPT_PAYLOAD",
                    f"cannot read checkpoint {path}: {exc}") from exc
        raise V543LoadError(
            "V543_CKPT_PAYLOAD",
            "checkpoint must be a path, a raw state dict or a trainer "
            f"payload mapping (got {type(checkpoint).__name__})")

    @staticmethod
    def _as_state_mapping(payload: object) -> Mapping[str, torch.Tensor]:
        """The source state dict of a raw/payload checkpoint (read-only)."""
        if isinstance(payload, Mapping) and isinstance(
                payload.get("model"), Mapping):
            payload = payload["model"]
        if not isinstance(payload, Mapping):
            raise V543LoadError(
                "V543_CKPT_PAYLOAD",
                f"checkpoint payload must be a mapping, got "
                f"{type(payload).__name__}")
        return payload  # type: ignore[return-value]

    # -- forward ------------------------------------------------------------
    def _refuse_supervision(self, batch: Mapping[str, torch.Tensor]) -> None:
        """Typed refusal if a supervision key reached the encoder (plan §5.6)."""
        leaks = sorted(k for k in SUPERVISION_KEYS_V543 if k in batch)
        if leaks:
            raise V543SupervisionLeakError(
                "V543_SUPERVISION_LEAK",
                f"encoder forward received supervision keys {leaks}; "
                "SUPERVISION_KEYS_V3 may never reach the model "
                "(plan §5.6)")

    def _decode_u0(
            self, batch: Mapping[str, torch.Tensor],
            edge_msg: torch.Tensor) -> torch.Tensor:
        """``u0`` from the SAME five tensors as V542 (no reordering)."""
        return self.move_decoder(
            edge_msg,
            batch[NODE_POS_KEY],
            batch[EDGE_INDEX_KEY],
            batch[NODE_KEY_SET_KEY],
            batch[NODE_DEGK_KEY],
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> CoarsePacket:
        """One encoder forward → :class:`CoarsePacket` (frozen six fields).

        Flow (plan §11.2): supervision refusal → three temporary hooks →
        ``backbone`` (``z`` + ``edge_msg``) → ``move_decoder`` (``u0``) →
        ``process_head(z)`` (``q`` + ``endpoint_logit``; ``fc2`` fires) →
        ``finally`` removes EVERY hook → each capture asserted to have
        fired exactly once → ``tanh(q0_raw) == q`` parity → packet
        validation.

        The encoder is NOT re-entrant by design: plan §4.4 forbids a second
        forward per configuration, and :attr:`forward_count` is the audit
        hook for that rule (the model-level enforcement belongs to the WP4
        ``V543Model``).
        """
        self._refuse_supervision(batch)
        # Legacy whitelist for every OTHER out-of-contract key
        # (``FORWARD_VIEW_LEAK``; the backbone re-checks verbatim).
        check_forward_view_keys_v3(batch)

        fires: dict[str, int] = {label: 0 for label in HOOK_LABELS}
        with single_fire_hook(
                self.backbone.message_mlp[-1], HOOK_LABEL_EDGE_MSG) as edge_cap, \
                single_fire_hook(
                    self.backbone.norm_h[-1], HOOK_LABEL_NODE_H) as h_cap, \
                single_fire_hook(
                    self.process_head.progress_head.fc2,
                    HOOK_LABEL_Q0_RAW) as q0_cap:
            out = self.backbone(batch)          # z + edge_msg (its own hook)
            n_graphs = int(batch[GRAPH_CV_DELTA_KEY].shape[0])
            n_edges = int(batch[EDGE_INDEX_KEY].shape[1])
            validate_backbone_output(
                out, n_graphs=n_graphs, n_edges=n_edges)
            u0 = self._decode_u0(batch, out.edge_msg)
            head = self.process_head(out.z)     # ``q`` + endpoint logit
            fires[HOOK_LABEL_EDGE_MSG] = len(edge_cap)
            fires[HOOK_LABEL_NODE_H] = len(h_cap)
            fires[HOOK_LABEL_Q0_RAW] = len(q0_cap)

        # Hooks are gone here — even on the typed-invalid abort path.
        edge_msg = assert_single_fire(edge_cap, HOOK_LABEL_EDGE_MSG)
        node_h = assert_single_fire(h_cap, HOOK_LABEL_NODE_H)
        q0_raw_2d = assert_single_fire(q0_cap, HOOK_LABEL_Q0_RAW)

        if tuple(q0_raw_2d.shape) != (n_graphs, 1):
            raise V543HookError(
                "V543_Q0_RAW_SHAPE",
                f"progress_head.fc2 output must be [{n_graphs},1], got "
                f"{list(q0_raw_2d.shape)}")
        # plan §5.2: ``q0_raw [G]`` — the fc2 [G,1] column squeezed (squeeze,
        # never a reshape/sum: the identity ``tanh(q0_raw) == q`` must stay
        # bit-exact).
        q0_raw = q0_raw_2d.squeeze(-1)

        if tuple(node_h.shape) != (int(batch[NODE_POS_KEY].shape[0]),
                                   HIDDEN_V543):
            raise V543HookError(
                "V543_NODE_H_SHAPE",
                f"norm_h[-1] output must be [N,{HIDDEN_V543}], got "
                f"{list(node_h.shape)}")

        # plan §5.8 equality 4 — proven here, on every forward, not in a test.
        q = head["q"]
        assert_q0_raw_matches_q(
            q0_raw, q, context="encoder forward (plan §5.8)")

        packet = CoarsePacket(
            z=out.z,
            edge_msg=edge_msg,
            node_h=node_h,
            u0=u0,
            q0_raw=q0_raw,
            e0=head["endpoint_logit"],
        )
        validate_coarse_packet(
            packet,
            n_graphs=n_graphs,
            n_nodes=int(node_h.shape[0]),
            n_keys=int(u0.shape[0]),
            n_edges=n_edges,
        )

        self._forward_count += 1
        self._last_hook_fires = dict(fires)
        for label, count in fires.items():
            self._cumulative_hook_fires[label] = (
                int(self._cumulative_hook_fires.get(label, 0)) + int(count))
        return packet


def measured_encoder_params() -> dict[str, int]:
    """Exact, freshly built encoder parameter counts (plan §6.4 invariants)."""
    encoder = CoarseEncoder()
    return {
        "backbone_active": int(encoder.backbone.active_param_count),
        "decoder": int(encoder.move_decoder.param_count),
        "process_head": int(sum(
            p.numel() for p in encoder.process_head.parameters())),
        "joint": int(sum(p.numel() for p in encoder.parameters())),
    }


def assert_encoder_packet_parity(
        packet: CoarsePacket,
        *,
        z: torch.Tensor,
        u0: torch.Tensor,
        q: torch.Tensor,
        endpoint_logit: torch.Tensor,
        edge_msg: torch.Tensor | None = None,
        context: str = "",
) -> dict[str, object]:
    """Plan §5.8 equalities 1/2/3/5 (+ the tanh identity) against a reference.

    Compares the packet against the V5.4.2 reference outputs
    (``z``/``u0``/``q``/``endpoint_logit``) at the CPU-bitwise / GPU-≤1e-6
    tolerance and returns the audit rows keyed by equality name.  Used by
    the WP1 parity certificate and by
    ``tests/pesn/test_v543_encoder_a0.py``.
    """
    rows: dict[str, object] = {
        "u0": compare_parity("u0", packet.u0, u0, context=context),
        "z": compare_parity("z", packet.z, z, context=context),
        "q0": compare_parity("q0", packet.q0, q, context=context),
        "q0_raw_tanh": assert_q0_raw_matches_q(
            packet.q0_raw, q, context=context),
        "endpoint0": compare_parity(
            "endpoint0", packet.e0, endpoint_logit, context=context),
    }
    if edge_msg is not None:
        rows["edge_msg"] = compare_parity(
            "edge_msg", packet.edge_msg, edge_msg, context=context)
    return rows


__all__ = [
    "HOOK_PATH_BY_LABEL",
    "CoarseEncoder",
    "V543Error",
    "V543ContractError",
    "assert_encoder_packet_parity",
    "assert_q0_raw_matches_q",
    "assert_single_fire",
    "classify_payload_layout",
    "compare_parity",
    "forked_global_rng",
    "identity_mapping_table",
    "measured_encoder_params",
    "parity_atol",
    "resolve_a0_mapping",
    "single_fire_hook",
    "visible_cuda_devices",
]
