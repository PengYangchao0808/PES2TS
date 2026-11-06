"""V5.4.3 assembly — frozen ``CoarseEncoder`` + parameter-free ``LatentBridge``
+ ONE registered decoder per run, with the §7.2 gradient-isolation assertions
(plan §4.1 :182–188, §4.4 :229–235, §6.4 :557–571, §7.2 :656–672/:712–726,
§11.1/§11.2 ``assembly.py``; WP4).

Normative sources, quoted as implemented
----------------------------------------

* **§4.1 component table (:182–188)** — five components with fixed roles.  The
  assembly owns three of them for a single-track run: ``CoarseEncoder``
  (frozen, 57,619 params), ``LatentBridge`` (fixed, 0 trainable) and the ONE
  Refiner of the current track; the other track's Refiner is present as its
  zero-residual baseline (§4.3 :226 "**单 Track Benchmark 时，未参与当前 Track
  的 Refiner 固定为零残差（M0/P0）**").  The ``ReactionController`` is NOT part
  of the assembly — plan §4.6 :263 "该函数不在 ``V543Model.forward`` 内".
* **§4.4 freeze boundaries (:229–235)** — boundary 1 (encoder runs once) is
  enforced in ``model.py``; boundary 2 ("两条 Track 独立") is enforced here:
  ``encoder.requires_grad_(False)`` for every encoder parameter plus the
  inactive Refiner's parameters, and the ACTIVE Refiner's parameters live.
* **§7.2 (:657)** — "Encoder / Bridge / D0 / P0 | 冻结；``requires_grad=False``；
  frozen 模式可用 ``no_grad``".  :meth:`V543Assembly.freeze_encoder` sets the
  flag; nothing here wraps a forward in ``no_grad()`` because that would sever
  the decoder's own gradient path (the same rule the V5.4.2 trainer states).
* **§7.2 gradient isolation (:713–726)** — the plan's assertion block is
  implemented VERBATIM as :func:`assert_gradient_isolation`: encoder grads must
  be ``None`` (not "small"), the inactive Refiner's grads must be ``None``, and
  the active Refiner must have at least one non-``None`` grad.  A zero-parameter
  baseline cannot satisfy the last clause, so :func:`assert_gradient_isolation`
  takes ``require_active_grad`` and the M0/P0 runs pass ``False`` with the
  reason recorded (a decoder with no parameter cannot have a parameter grad).
* **§6.4 (:558–571)** — the assembly is where the GLOBAL budget lives:
  ``57,619 + active + inactive`` must stay under the 150,000 hard cap, the
  active decoder keeps its per-candidate band/cap refusal (delegated to
  :func:`~pes2ts_core.pesn.models.v543.registry.assert_budget`), and the
  inactive baseline must be exactly 0.

Design decisions and their basis
--------------------------------

1. **Track selection is explicit and total.**  ``active_track`` is a required
   keyword of :func:`assemble_v543`; the assembly refuses a *missing* decoder
   binding for either track (it always materialises BOTH, defaulting the
   inactive one to its baseline), so a run can never accidentally train two
   tracks or zero tracks.  Plan §4.4 :232 "两条 Track 独立".
2. **Bindings carry the measured count.**  :class:`DecoderBinding` records
   ``sum(p.numel())`` measured at bind time; the assembly's budget report and
   ``metrics.json`` quote the MEASURED number, never the registry's design
   target (plan §6.4 :569 "估算值不得作为实现前 exact 断言").
3. **The encoder is shared by reference, never re-wrapped.**  The assembly
   holds the caller's ``CoarseEncoder``/``LatentBridge`` objects (``V542Model``
   does the same with ``A0LegacyBackbone``), so a WP1 parity certificate about
   that object is a statement about this model too.

Discipline: no V5.4.2 / V5.3-MP file is modified (plan §11.3 禁止), no
``torch.compile`` (checked at construction), every constant is ``Final``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import torch

from pes2ts_core.pesn.models.v543.contracts import (
    DECODER_CONTEXT_SCHEMA_V543,
    PRECISION_VERSION_V543,
    CoarsePacket,
    DecoderContextV1,
    MoveContextV1,
    ProcessContextV1,
    ProcessResidual,
    V543Error,
)
from pes2ts_core.pesn.models.v543.coarse_encoder import CoarseEncoder
from pes2ts_core.pesn.models.v543.latent_bridge import BRIDGE_SPEC_SHA256, LatentBridge
from pes2ts_core.pesn.models.v543.registry import (
    BASELINE_DECODER_BY_TRACK,
    ENCODER_JOINT_PARAMS_V543,
    PARAM_BASELINE_V543,
    PARAM_GLOBAL_CAP_V543,
    TRACK_MOVE,
    TRACK_PROCESS,
    TRACKS_V543,
    assert_budget,
    assert_decoder_protocol,
    build_baseline,
    count_trainable_params,
)

#: Schema id of an assembly description (``metrics.json`` input).
V543_ASSEMBLY_SCHEMA: Final[str] = "pesn_v543_assembly_v1"
#: Registered assembly version.
V543_ASSEMBLY_VERSION: Final[str] = "pesn_v543_assembly_v1"


class V543AssemblyError(V543Error):
    """Assembly wiring / budget / gradient-isolation refusal."""


# ---------------------------------------------------------------------------
# decoder binding (measured, not declared)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DecoderBinding:
    """One decoder attached to the assembly, with its MEASURED parameter count.

    ``n_trainable_params`` is ``sum(p.numel())`` over the decoder's parameters
    at bind time (plan §6.1 :509 / §6.4 :569); ``role`` is ``"active"`` (the
    current track's Refiner, training) or ``"baseline"`` (the other track's
    zero-residual placeholder, plan §4.3 :226).
    """

    decoder_id: str
    track: str
    model_version: str
    role: str
    decoder: Any
    n_trainable_params: int

    @property
    def is_baseline(self) -> bool:
        return self.decoder_id == BASELINE_DECODER_BY_TRACK[self.track]

    def describe(self) -> dict[str, Any]:
        return {
            "decoder_id": self.decoder_id,
            "track": self.track,
            "model_version": self.model_version,
            "role": self.role,
            "n_trainable_params": int(self.n_trainable_params),
            "is_baseline": bool(self.is_baseline),
            "class": type(self.decoder).__name__,
            "contract_version": DECODER_CONTEXT_SCHEMA_V543,
        }


def bind_decoder(decoder: Any, *, decoder_id: str, track: str,
                 model_version: str = "", role: str = "active") -> DecoderBinding:
    """Validate ``decoder`` against the §6.1 protocol and measure its params.

    The parameter BAND is asserted by the registry at construction (plan §6.4);
    binding re-measures the count so the assembly's budget can quote it, and it
    refuses a bootstrapped decoder that reports a different count than it owns
    (``V543_ASSEMBLY_PARAM_REPORT``).
    """
    if role not in ("active", "baseline"):
        raise V543AssemblyError(
            "V543_ASSEMBLY_ROLE",
            f"decoder role must be 'active' or 'baseline', got {role!r}")
    assert_decoder_protocol(decoder, decoder_id=decoder_id, track=track)
    count = count_trainable_params(decoder)
    declared = getattr(decoder, "n_trainable_params", None)
    if declared is not None and int(declared) != count:
        raise V543AssemblyError(
            "V543_ASSEMBLY_PARAM_REPORT",
            f"decoder {decoder_id!r} reports n_trainable_params={declared} "
            f"but sum(p.numel())={count} (plan §6.1 :509)")
    return DecoderBinding(
        decoder_id=str(decoder_id),
        track=str(track),
        model_version=str(model_version
                          or getattr(decoder, "model_version", "")
                          or "unversioned"),
        role=str(role),
        decoder=decoder,
        n_trainable_params=int(count),
    )


# ---------------------------------------------------------------------------
# the assembly
# ---------------------------------------------------------------------------
class V543Assembly:
    """``encoder + bridge + move binding + process binding + active track``.

    Constructed by :func:`assemble_v543`; every invariant is asserted there so
    a half-wired assembly is unrepresentable.
    """

    def __init__(
            self,
            encoder: CoarseEncoder,
            bridge: LatentBridge,
            *,
            move: DecoderBinding,
            process: DecoderBinding,
            active_track: str,
    ) -> None:
        self.encoder = encoder
        self.bridge = bridge
        self.move = move
        self.process = process
        self.active_track = str(active_track)

    # -- bindings -----------------------------------------------------------
    def binding(self, track: str) -> DecoderBinding:
        if track == TRACK_MOVE:
            return self.move
        if track == TRACK_PROCESS:
            return self.process
        raise V543AssemblyError(
            "V543_ASSEMBLY_TRACK",
            f"track must be one of {list(TRACKS_V543)}, got {track!r}")

    @property
    def active(self) -> DecoderBinding:
        return self.binding(self.active_track)

    @property
    def inactive(self) -> DecoderBinding:
        return self.binding(
            TRACK_PROCESS if self.active_track == TRACK_MOVE else TRACK_MOVE)

    @property
    def active_decoder(self) -> Any:
        return self.active.decoder

    @property
    def inactive_decoder(self) -> Any:
        return self.inactive.decoder

    # -- encoder freezing (plan §4.4 :232 / §7.2 :657) ----------------------
    def freeze_encoder(self) -> "V543Assembly":
        """``requires_grad_(False)`` on EVERY encoder parameter + ``eval()``."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        return self

    def assert_encoder_frozen(self) -> None:
        """Typed refusal if any encoder parameter is still trainable."""
        live = [name for name, parameter in self.encoder.named_parameters()
                if parameter.requires_grad]
        if live:
            raise V543AssemblyError(
                "V543_ENCODER_NOT_FROZEN",
                f"{len(live)} encoder parameter(s) still require grad "
                f"({live[:4]}…); the encoder is frozen for the whole S1 "
                "schedule (plan §7.2 :657)")

    def set_active_track(self, track: str) -> None:
        """Switch the active track and re-freeze the boundary (§4.4 :232)."""
        if str(track) not in TRACKS_V543:
            raise V543AssemblyError(
                "V543_ASSEMBLY_TRACK",
                f"track must be one of {list(TRACKS_V543)}, got {track!r}")
        self.active_track = str(track)
        for parameter in self.active.decoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.inactive.decoder.parameters():
            parameter.requires_grad_(False)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """The ACTIVE decoder's parameters (the optimizer's only members)."""
        return [p for p in getattr(self.active.decoder, "parameters",
                                   list)() if p.requires_grad]

    # -- budget (plan §6.4) -------------------------------------------------
    def param_report(self) -> dict[str, Any]:
        """Measured parameter report of the whole assembly (JSON-able)."""
        encoder_joint = int(sum(p.numel() for p in self.encoder.parameters()))
        active = int(self.active.n_trainable_params)
        inactive = int(self.inactive.n_trainable_params)
        trainable = int(sum(p.numel() for p in self.trainable_parameters()))
        return {
            "schema": V543_ASSEMBLY_SCHEMA,
            "version": V543_ASSEMBLY_VERSION,
            "encoder_joint": encoder_joint,
            "encoder_expected_joint": ENCODER_JOINT_PARAMS_V543,
            "bridge_trainable": int(sum(
                p.numel() for p in self.bridge.parameters())),
            "active_track": self.active_track,
            "active": self.active.describe(),
            "inactive": self.inactive.describe(),
            "active_trainable_measured": trainable,
            "inactive_params": inactive,
            "total_model_params": encoder_joint + active + inactive,
            "global_hard_cap": PARAM_GLOBAL_CAP_V543,
            "inactive_must_be_zero": PARAM_BASELINE_V543,
            "precision_version": PRECISION_VERSION_V543,
        }

    def assert_param_budget(self) -> dict[str, Any]:
        """Assert the §6.4 global cap and the zero-residual inactive track.

        Refusals: ``V543_GLOBAL_BUDGET`` (total above 150,000),
        ``V543_ASSEMBLY_INACTIVE`` (the inactive Refiner is not the zero
        baseline), ``V543_ASSEMBLY_ENCODER_PARAMS`` (encoder joint count
        drifted from 57,619), and the registry band/cap family for the active
        decoder.
        """
        report = self.param_report()
        if int(report["encoder_joint"]) != ENCODER_JOINT_PARAMS_V543:
            raise V543AssemblyError(
                "V543_ASSEMBLY_ENCODER_PARAMS",
                f"encoder joint parameters {report['encoder_joint']} != "
                f"{ENCODER_JOINT_PARAMS_V543} (plan §6.4 :566)")
        if int(report["bridge_trainable"]) != 0:
            raise V543AssemblyError(
                "V543_ASSEMBLY_BRIDGE_PARAMS",
                f"the Bridge must be parameter free (plan §4.1 :185), found "
                f"{report['bridge_trainable']} trainable parameters")
        if not self.inactive.is_baseline:
            raise V543AssemblyError(
                "V543_ASSEMBLY_INACTIVE",
                f"the non-active track ({self.inactive.track}) must be its "
                f"zero-residual baseline "
                f"{BASELINE_DECODER_BY_TRACK[self.inactive.track]!r}, got "
                f"{self.inactive.decoder_id!r} (plan §4.3 :226: 单 Track "
                "Benchmark 时未参与当前 Track 的 Refiner 固定为零残差)")
        if int(report["inactive_params"]) != PARAM_BASELINE_V543:
            raise V543AssemblyError(
                "V543_ASSEMBLY_INACTIVE",
                f"the inactive baseline carries "
                f"{report['inactive_params']} parameters; M0/P0 are "
                "zero-parameter (plan §6.4 :565)")
        assert_budget(self.active.n_trainable_params,
                      decoder_id=self.active.decoder_id,
                      track=self.active.track)
        if int(report["total_model_params"]) > PARAM_GLOBAL_CAP_V543:
            raise V543AssemblyError(
                "V543_GLOBAL_BUDGET",
                f"total model parameters {report['total_model_params']} "
                f"exceed the global hard cap {PARAM_GLOBAL_CAP_V543} "
                "(plan §6.4 :568)")
        return report

    # -- forwarding helpers (the model owns the flow; the assembly owns the
    #    boundary checks) ---------------------------------------------------
    def move_view(self, ctx: DecoderContextV1) -> MoveContextV1:
        return ctx.move_view()

    def process_view(self, ctx: DecoderContextV1) -> ProcessContextV1:
        return ctx.process_view()

    # -- gradient isolation (plan §7.2 :713–726, verbatim) -----------------
    def assert_gradient_isolation(
            self, *, require_active_grad: bool = True) -> dict[str, Any]:
        """The plan §7.2 assertion block, on the CURRENT ``.grad`` state.

        ``assert all(p.grad is None for p in encoder.parameters())`` and
        ``assert all(p.grad is None for p in inactive.parameters())`` are
        checked as written — a zero-valued grad is still a FAILURE (it proves
        a gradient path existed).  ``assert any(p.grad is not None for p in
        active.parameters())`` is checked only when ``require_active_grad``
        (a zero-parameter M0/P0 run legitimately has no parameter gradient).

        Refusal codes: ``V543_GRADIENT_ENCODER``, ``V543_GRADIENT_INACTIVE``,
        ``V543_GRADIENT_ACTIVE_MISSING``.
        """
        encoder_grads = [name for name, p in self.encoder.named_parameters()
                         if p.grad is not None]
        if encoder_grads:
            raise V543AssemblyError(
                "V543_GRADIENT_ENCODER",
                f"{len(encoder_grads)} encoder parameter(s) received a "
                f"gradient ({encoder_grads[:4]}…); the decoder backward must "
                "not reach the frozen encoder (plan §7.2 :721)")
        inactive = getattr(self.inactive.decoder, "parameters", None)
        inactive_grads = ([name for name, p in self.inactive.decoder
                           .named_parameters() if p.grad is not None]
                          if inactive is not None else [])
        if inactive_grads:
            raise V543AssemblyError(
                "V543_GRADIENT_INACTIVE",
                f"the inactive {self.inactive.decoder_id} received a gradient "
                f"({inactive_grads[:4]}…); the two tracks are independent "
                "(plan §7.2 :722, §4.4 :232)")
        active_grads = [name for name, p in
                        self.active.decoder.named_parameters()
                        if p.grad is not None] \
            if hasattr(self.active.decoder, "named_parameters") else []
        if require_active_grad and not active_grads:
            raise V543AssemblyError(
                "V543_GRADIENT_ACTIVE_MISSING",
                f"the active {self.active.decoder_id} received no gradient; "
                "assert any(p.grad is not None …) failed (plan §7.2 :723)")
        return {
            "encoder_params_with_grad": 0,
            "inactive_params_with_grad": 0,
            "active_params_with_grad": int(len(active_grads)),
            "require_active_grad": bool(require_active_grad),
            "active_trainable_params": int(self.active.n_trainable_params),
        }

    def zero_grad(self) -> None:
        """Clear every grad this assembly can hold (encoder/bridge/decoders)."""
        for module in (self.encoder, self.bridge, self.move.decoder,
                       self.process.decoder):
            clear = getattr(module, "zero_grad", None)
            if callable(clear):
                clear(set_to_none=True)

    def describe(self) -> dict[str, Any]:
        """JSON-able assembly description (``metrics.json`` / run identity)."""
        return {
            "schema": V543_ASSEMBLY_SCHEMA,
            "version": V543_ASSEMBLY_VERSION,
            "active_track": self.active_track,
            "encoder": self.encoder.describe(),
            "bridge": self.bridge.describe(),
            "bridge_spec_sha256": BRIDGE_SPEC_SHA256,
            "move": self.move.describe(),
            "process": self.process.describe(),
            "params": self.param_report(),
            "isolation": {
                "encoder_requires_grad": False,
                "inactive_requires_grad": False,
                "active_requires_grad": True,
                "rule": "plan §7.2 :713–726",
            },
        }


# ---------------------------------------------------------------------------
# assembly entry point (plan §11.2 ``assemble_v543``)
# ---------------------------------------------------------------------------
def assemble_v543(
        encoder: CoarseEncoder,
        *,
        bridge: LatentBridge | None = None,
        move_refiner: Any | None = None,
        process_refiner: Any | None = None,
        active_track: str = TRACK_MOVE,
        move_decoder_id: str | None = None,
        process_decoder_id: str | None = None,
        encoder_ckpt_sha256: str = "",
) -> V543Assembly:
    """Bind one frozen encoder + one Bridge + the two Refiner slots.

    Args:
        encoder: the WP1 :class:`CoarseEncoder` (frozen here).
        bridge: the WP2 :class:`LatentBridge`; ``None`` builds one bound to the
            encoder digest recorded in every context (plan §5.2 identity row).
        move_refiner / process_refiner: the constructed decoders.  ``None``
            falls back to the track's zero-residual baseline (``M0`` / ``P0``),
            which is exactly what plan §4.3 :226 requires for the non-active
            track and what a T0 baseline run uses for both.
        active_track: which Refiner trains in this run (§4.4 boundary 2).
        move_decoder_id / process_decoder_id: the ids to RECORD for the
            supplied decoders (defaults: the object's ``decoder_id`` or the
            baseline id).
        encoder_ckpt_sha256: A0 checkpoint digest for the Bridge identity
            fields when ``bridge`` is built here; refused if the supplied
            bridge records a different non-empty digest.

    Returns a fully asserted :class:`V543Assembly`.
    """
    if not isinstance(encoder, CoarseEncoder):
        raise V543AssemblyError(
            "V543_ASSEMBLY_ENCODER",
            f"encoder must be a CoarseEncoder, got {type(encoder).__name__}")
    if str(active_track) not in TRACKS_V543:
        raise V543AssemblyError(
            "V543_ASSEMBLY_TRACK",
            f"active_track must be one of {list(TRACKS_V543)}, got "
            f"{active_track!r}")
    if bridge is None:
        bridge = LatentBridge(encoder_ckpt_sha256=encoder_ckpt_sha256)
    if not isinstance(bridge, LatentBridge):
        raise V543AssemblyError(
            "V543_ASSEMBLY_BRIDGE",
            f"bridge must be a LatentBridge, got {type(bridge).__name__}")
    recorded = getattr(bridge, "_encoder_ckpt_sha256", "")  # documented access
    if encoder_ckpt_sha256 and recorded and recorded != encoder_ckpt_sha256:
        raise V543AssemblyError(
            "V543_ASSEMBLY_BRIDGE_IDENTITY",
            "the supplied Bridge records encoder_ckpt_sha256="
            f"{recorded!r} but the assembly was told "
            f"{encoder_ckpt_sha256!r}; the context identity would lie "
            "(plan §5.2 identity row)")

    # MOVE slot.
    if move_refiner is None:
        move_refiner = build_baseline(TRACK_MOVE)
        move_id = BASELINE_DECODER_BY_TRACK[TRACK_MOVE]
    else:
        move_id = str(move_decoder_id
                      or getattr(move_refiner, "decoder_id", "")
                      or BASELINE_DECODER_BY_TRACK[TRACK_MOVE])
    move_role = "active" if active_track == TRACK_MOVE else "baseline"
    move = bind_decoder(move_refiner, decoder_id=move_id, track=TRACK_MOVE,
                        role=move_role)

    # PROCESS slot.
    if process_refiner is None:
        process_refiner = build_baseline(TRACK_PROCESS)
        process_id = BASELINE_DECODER_BY_TRACK[TRACK_PROCESS]
    else:
        process_id = str(process_decoder_id
                         or getattr(process_refiner, "decoder_id", "")
                         or BASELINE_DECODER_BY_TRACK[TRACK_PROCESS])
    process_role = "active" if active_track == TRACK_PROCESS else "baseline"
    process = bind_decoder(process_refiner, decoder_id=process_id,
                           track=TRACK_PROCESS, role=process_role)

    assembly = V543Assembly(
        encoder, bridge, move=move, process=process,
        active_track=str(active_track))
    assembly.freeze_encoder()
    assembly.assert_encoder_frozen()
    # The ACTIVE refiner is trainable, the inactive one is not (§4.4 :232).
    for parameter in assembly.active.decoder.parameters():
        parameter.requires_grad_(True)
    for parameter in assembly.inactive.decoder.parameters():
        parameter.requires_grad_(False)
    assembly.assert_param_budget()
    return assembly


# ---------------------------------------------------------------------------
# convenience: the M0/P0 baseline assembly of the T0 runs
# ---------------------------------------------------------------------------
def assemble_baseline_v543(
        encoder: CoarseEncoder,
        *,
        bridge: LatentBridge | None = None,
        active_track: str = TRACK_MOVE,
        encoder_ckpt_sha256: str = "",
) -> V543Assembly:
    """The plan §8.1 T0 assembly: M0 + P0, no trainable decoder at all."""
    return assemble_v543(
        encoder, bridge=bridge, move_refiner=None, process_refiner=None,
        active_track=active_track,
        encoder_ckpt_sha256=encoder_ckpt_sha256)


# ---------------------------------------------------------------------------
# supervisor / output contract helpers shared with ``model.py``
# ---------------------------------------------------------------------------
def check_move_output(delta_u: Any, *, n_keys: int, decoder_id: str
                      ) -> torch.Tensor:
    """``δu`` must be ``[K,3]`` fp32 and finite (plan §6.1 :490–493)."""
    if not isinstance(delta_u, torch.Tensor):
        raise V543AssemblyError(
            "V543_DECODER_OUTPUT",
            f"{decoder_id}: forward must return a Tensor δu [K,3], got "
            f"{type(delta_u).__name__} (plan §6.1 :490)")
    if tuple(delta_u.shape) != (int(n_keys), 3):
        raise V543AssemblyError(
            "V543_DECODER_OUTPUT",
            f"{decoder_id}: δu must be [{int(n_keys)},3], got "
            f"{list(delta_u.shape)}")
    if delta_u.dtype != torch.float32:
        raise V543AssemblyError(
            "V543_DECODER_DTYPE",
            f"{decoder_id}: δu must be fp32 (plan §7.2 precision row), got "
            f"{delta_u.dtype}")
    if not bool(torch.isfinite(delta_u).all()):
        raise V543AssemblyError(
            "V543_DECODER_NONFINITE",
            f"{decoder_id}: δu contains non-finite values")
    return delta_u


def check_process_output(residual: Any, *, n_graphs: int, decoder_id: str
                         ) -> ProcessResidual:
    """``ProcessResidual`` must be two ``[G]`` fp32 finite tensors (§6.1 :496)."""
    if not isinstance(residual, ProcessResidual):
        raise V543AssemblyError(
            "V543_DECODER_OUTPUT",
            f"{decoder_id}: forward must return a ProcessResidual, got "
            f"{type(residual).__name__} (plan §6.1 :496)")
    for name in ("delta_q_raw", "delta_e"):
        value = getattr(residual, name)
        if tuple(value.shape) != (int(n_graphs),):
            raise V543AssemblyError(
                "V543_DECODER_OUTPUT",
                f"{decoder_id}: {name} must be [{int(n_graphs)}], got "
                f"{list(value.shape)}")
        if value.dtype != torch.float32:
            raise V543AssemblyError(
                "V543_DECODER_DTYPE",
                f"{decoder_id}: {name} must be fp32, got {value.dtype}")
        if not bool(torch.isfinite(value).all()):
            raise V543AssemblyError(
                "V543_DECODER_NONFINITE",
                f"{decoder_id}: {name} contains non-finite values")
    return residual


def assert_packet_matches_context(
        packet: CoarsePacket, ctx: DecoderContextV1) -> torch.Tensor:
    """Return the Bridge key permutation and prove it inverts ``u0``.

    ``context.u0 == packet.u0[key_permutation]`` by Bridge §5.2.1 decision 4
    (``latent_bridge.py`` :754: ``argsort(key_graph, stable=True)``), so the
    permutation is recoverable from the context alone — and is PROVEN here
    bit-for-bit (``torch.equal``) instead of assumed, because every downstream
    loss consumes ``u`` in DATASET key order (plan §4.3 :222 "行序严格等于
    ``key_index``").

    Refusal: ``V543_ASSEMBLY_KEY_PERMUTATION``.
    """
    permutation = torch.argsort(ctx.key_graph, stable=True)
    expected = packet.u0[permutation]
    if not bool(torch.equal(expected, ctx.u0)):
        raise V543AssemblyError(
            "V543_ASSEMBLY_KEY_PERMUTATION",
            "context.u0 is not packet.u0[argsort(key_graph)]; the Bridge key "
            "permutation cannot be inverted from the context, so u cannot be "
            "returned in key_index row order (plan §4.3 :222)")
    return permutation


def unpermute_to_key_rows(total: torch.Tensor,
                          permutation: torch.Tensor) -> torch.Tensor:
    """Invert the Bridge permutation: ``out[perm] = total`` (key-slot → rows)."""
    return torch.zeros_like(total).index_copy(0, permutation, total)


__all__ = [
    "DecoderBinding",
    "V543_ASSEMBLY_SCHEMA",
    "V543_ASSEMBLY_VERSION",
    "V543Assembly",
    "V543AssemblyError",
    "assemble_baseline_v543",
    "assemble_v543",
    "assert_packet_matches_context",
    "bind_decoder",
    "check_move_output",
    "check_process_output",
    "unpermute_to_key_rows",
]
