"""V5.4.3 ``V543Model`` — the ONE forward flow (plan §4.2 :190–218, §4.3
:220–227, §4.4 :229–235, §5.8 :460–480, §11.2 ``model.py``; WP4).

Normative sources, quoted as implemented
----------------------------------------

* **§4.2 the single flow (:192–218)** — ``forward_view`` → ``CoarseEncoder``
  ("只调用一次") → ``LatentBridge`` → the track Refiner → ``u = u0 + δu`` /
  ``q = tanh(q0_raw + δq)`` / ``e = e0 + δe``.  :meth:`V543Model.forward` is
  that flow and nothing else; the ``ReactionController`` is deliberately NOT
  here (§4.6 :263 "该函数不在 ``V543Model.forward`` 内；训练前向不产生
  ``move_flag``").
* **§4.3 output semantics (:222–225)** — ``u = u0 + δu`` in Å with "行序严格
  等于 ``key_index``"; ``q = tanh(q0_raw + δq)`` with "``order_score IS q``"
  (the returned ``order_score`` IS the same tensor object as ``q``, not a
  copy); ``e = e0 + δe`` with "训练时只用 logit，禁止双重 sigmoid";
  "外部 forward 仍只返回五键 ``{u, z, q, endpoint_logit, order_score}``；
  诊断量（``u0``、``δu``、``q0_raw``、``δq``）通过独立 ``forward_debug``
  返回，不进入外部合同".
* **§4.4 boundary 1 (:231)** — "每次当前构型恰好一次 ``CoarseEncoder.forward``";
  :meth:`forward` asserts the encoder counter advanced by EXACTLY
  :data:`ENCODER_FORWARD_CALLS_PER_VIEW` and refuses anything else
  (``V543_MODEL_ENCODER_CALLS``).
* **§4.3 :226** — "未参与当前 Track 的 Refiner 固定为零残差（M0/P0），避免
  另一 Track 的随机初始化污染当前指标".  Both refiners are evaluated on every
  forward; the inactive one is the zero-residual baseline by construction of
  :class:`~pes2ts_core.pesn.models.v543.assembly.V543Assembly`.
* **§5.8 last bullet (:480)** — the frozen mode may run inside ``no_grad()``,
  but the model NEVER wraps its own forward in one: the decoder's gradient path
  must stay live (§7.2 :657 "frozen ≠ no_grad", the V5.4.2 rule).

Design decisions and their basis
--------------------------------

1. **Key-row order is restored, not assumed.**  ``DecoderContextV1.key_*`` is
   in the Bridge's graph-contiguous canonical order, while the frozen MOVE loss
   consumes ``u``/``y_teacher``/``pair_index`` in DATASET row order.  The model
   therefore maps ``δu`` back with the Bridge permutation it PROVES on every
   forward (:func:`~...assembly.assert_packet_matches_context`,
   ``torch.equal``) — a silent mismatch would corrupt every MOVE metric.
2. **``forward_context`` is a first-class entry point.**  The WP4 feature cache
   (§7.3) delivers a context built from cached per-frame contexts; the decoder
   half of the flow is identical in both cases, so it is one method and the
   cache cannot fork the numerics.
3. **Diagnostics are read-only extras.**  ``forward_debug`` returns the five
   contract keys PLUS ``u0``/``δu``/``q0_raw``/``δq``/``endpoint0``/``δe`` and
   the audit counters; it never mutates the model and it is never called by a
   loss (plan §4.3 :225).

Discipline: no V5.4.2/V5.3-MP file modified, no ``torch.compile`` anywhere
(``assert_no_torch_compile`` at construction), ``Final`` constants."""

from __future__ import annotations

from typing import Any, Final, Mapping

import torch

from pes2ts_core.pesn.models.v543.assembly import (
    V543Assembly,
    assert_packet_matches_context,
    check_move_output,
    check_process_output,
    unpermute_to_key_rows,
)
from pes2ts_core.pesn.models.v543.contracts import (
    ENCODER_FORWARD_CALLS_PER_VIEW,
    PRECISION_VERSION_V543,
    CoarsePacket,
    DecoderContextV1,
    V543Error,
)

#: Schema id of the model's external output (plan §4.3 five keys).
V543_MODEL_SCHEMA: Final[str] = "pesn_v543_model_v1"
#: Registered model version.
V543_MODEL_VERSION: Final[str] = "pesn_key_atom_v543_model_v1"

#: The EXTERNAL five-key contract of plan §4.3 :225 — nothing else may appear
#: in ``V543Model.forward``'s return value.
EXTERNAL_OUTPUT_KEYS: Final[tuple[str, ...]] = (
    "u", "z", "q", "endpoint_logit", "order_score")

#: The diagnostic keys ``forward_debug`` adds on top (plan §4.3 :225).
DEBUG_OUTPUT_KEYS: Final[tuple[str, ...]] = (
    "u0", "delta_u", "q0_raw", "delta_q_raw", "endpoint0_logit", "delta_e",
    "key_permutation", "n_graphs", "n_keys", "n_pairs",
    "encoder_forward_count", "active_track", "decoder_ids")


class V543ModelError(V543Error):
    """Model-flow refusal (encoder call count, output contract, wiring)."""


class V543Model(torch.nn.Module):
    """The unique V5.4.3 forward flow for ONE track (plan §4.2/§4.3).

    Args:
        assembly: a fully asserted
            :class:`~pes2ts_core.pesn.models.v543.assembly.V543Assembly`.

    The module owns no parameter of its own: encoder/bridge/decoders live in
    the assembly by reference, so ``state_dict()`` of this model is the union
    of those modules and the trainer's optimizer sees exactly the active
    decoder's parameters (``V543Assembly.trainable_parameters``).
    """

    schema: Final[str] = V543_MODEL_SCHEMA
    model_version: Final[str] = V543_MODEL_VERSION

    def __init__(self, assembly: V543Assembly) -> None:
        super().__init__()
        if not isinstance(assembly, V543Assembly):
            raise V543ModelError(
                "V543_MODEL_ASSEMBLY",
                f"V543Model needs a V543Assembly, got "
                f"{type(assembly).__name__}")
        self.assembly = assembly
        from pes2ts_core.pesn.models.v542.interfaces import (
            assert_no_torch_compile,
        )

        assert_no_torch_compile(self)

    # -- introspection ------------------------------------------------------
    @property
    def encoder_forward_count(self) -> int:
        """Encoder forwards performed through THIS model instance."""
        return int(self.assembly.encoder.forward_count)

    @property
    def active_track(self) -> str:
        return str(self.assembly.active_track)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return self.assembly.trainable_parameters()

    def describe(self) -> dict[str, Any]:
        return {
            "schema": V543_MODEL_SCHEMA,
            "model_version": V543_MODEL_VERSION,
            "external_output_keys": list(EXTERNAL_OUTPUT_KEYS),
            "debug_output_keys": list(DEBUG_OUTPUT_KEYS),
            "precision_version": PRECISION_VERSION_V543,
            "encoder_forward_calls_per_view": ENCODER_FORWARD_CALLS_PER_VIEW,
            "assembly": self.assembly.describe(),
        }

    # -- the flow -----------------------------------------------------------
    def forward(self, forward_view: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """``forward_view`` → the five-key external contract (plan §4.3 :225).

        Supervisory signature: the encoder refuses a ``SUPERVISION_KEYS_V3``
        member and the Bridge refuses any non-forward-view key, so a caller
        cannot smuggle a label into this path (plan §5.6).
        """
        before = int(self.assembly.encoder.forward_count)
        packet = self.assembly.encoder(forward_view)
        after = int(self.assembly.encoder.forward_count)
        calls = after - before
        if calls != ENCODER_FORWARD_CALLS_PER_VIEW:
            raise V543ModelError(
                "V543_MODEL_ENCODER_CALLS",
                f"one forward_view must run the CoarseEncoder exactly "
                f"{ENCODER_FORWARD_CALLS_PER_VIEW} time (plan §4.4 :231), "
                f"observed {calls}")
        return self._flow(packet, forward_view, debug=False)

    def forward_debug(self, forward_view: Mapping[str, Any]
                      ) -> dict[str, Any]:
        """The five contract keys PLUS the plan §4.3 :225 diagnostics."""
        before = int(self.assembly.encoder.forward_count)
        packet = self.assembly.encoder(forward_view)
        calls = int(self.assembly.encoder.forward_count) - before
        if calls != ENCODER_FORWARD_CALLS_PER_VIEW:
            raise V543ModelError(
                "V543_MODEL_ENCODER_CALLS",
                f"one forward_view must run the CoarseEncoder exactly "
                f"{ENCODER_FORWARD_CALLS_PER_VIEW} time (plan §4.4 :231), "
                f"observed {calls}")
        return self._flow(packet, forward_view, debug=True)

    def forward_packet(self, packet: CoarsePacket,
                       forward_view: Mapping[str, Any],
                       *, debug: bool = False) -> dict[str, Any]:
        """Run the flow from an ALREADY built packet (cache/parity paths).

        Used by ``tests`` and by the WP1 parity certificate, which compare a
        packet produced elsewhere; the encoder is NOT called here (the caller
        owns that call and its count).
        """
        return self._flow(packet, forward_view, debug=debug)

    def forward_context(self, ctx: DecoderContextV1, *, debug: bool = False
                        ) -> dict[str, Any]:
        """Run ONLY the decoder half on a prebuilt context (cache path, §7.3).

        The cache stores per-frame contexts; the trainer concatenates them and
        calls this method, so a cache hit skips the encoder but produces the
        identical decoder arithmetic (:meth:`_decoder_flow` is shared).
        """
        if not isinstance(ctx, DecoderContextV1):
            raise V543ModelError(
                "V543_MODEL_CONTEXT",
                f"forward_context needs a DecoderContextV1, got "
                f"{type(ctx).__name__}")
        return self._decoder_flow(ctx, packet=None, debug=debug)

    # -- internals ----------------------------------------------------------
    def _flow(self, packet: CoarsePacket, forward_view: Mapping[str, Any],
              *, debug: bool) -> dict[str, Any]:
        bridge = self.assembly.bridge
        ctx = bridge.build_context(packet, forward_view)
        return self._decoder_flow(ctx, packet=packet, debug=debug)

    def _decoder_flow(self, ctx: DecoderContextV1, *,
                      packet: CoarsePacket | None,
                      debug: bool) -> dict[str, Any]:
        assembly = self.assembly
        n_graphs = int(ctx.n_graphs)
        n_keys = int(ctx.n_keys)

        # MOVE half (always evaluated: the inactive track is the zero
        # baseline, plan §4.3 :226).
        move_delta = check_move_output(
            assembly.move.decoder(ctx.move_view()), n_keys=n_keys,
            decoder_id=assembly.move.decoder_id)
        if packet is not None:
            permutation = assert_packet_matches_context(packet, ctx)
        else:
            permutation = torch.argsort(ctx.key_graph, stable=True)
        total_u = ctx.u0 + move_delta
        u = unpermute_to_key_rows(total_u, permutation)

        # PROCESS half (always evaluated, same reason).
        residual = check_process_output(
            assembly.process.decoder(ctx.process_view()), n_graphs=n_graphs,
            decoder_id=assembly.process.decoder_id)
        q0_raw = ctx.q0_raw
        endpoint0 = ctx.endpoint0_logit
        q = torch.tanh(q0_raw + residual.delta_q_raw)
        endpoint_logit = endpoint0 + residual.delta_e

        out: dict[str, Any] = {
            "u": u,
            "z": ctx.graph_s,
            "q": q,
            "endpoint_logit": endpoint_logit,
            # plan §4.3 :223 "``order_score IS q``" — the SAME tensor.
            "order_score": q,
        }
        if not debug:
            return out
        out.update({
            "u0": ctx.u0,
            "delta_u": total_u - ctx.u0,
            "q0_raw": q0_raw,
            "delta_q_raw": residual.delta_q_raw,
            "endpoint0_logit": endpoint0,
            "delta_e": residual.delta_e,
            "key_permutation": permutation,
            "n_graphs": n_graphs,
            "n_keys": n_keys,
            "n_pairs": int(ctx.n_pairs),
            "encoder_forward_count": int(assembly.encoder.forward_count),
            "active_track": assembly.active_track,
            "decoder_ids": {"move": assembly.move.decoder_id,
                            "process": assembly.process.decoder_id},
        })
        return out

    # -- loss-side convenience ---------------------------------------------
    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Clear the grads of encoder/bridge/decoders (isolation hygiene)."""
        if set_to_none:
            self.assembly.zero_grad()
        else:
            super().zero_grad()

    def assert_gradient_isolation(self, *, require_active_grad: bool = True
                                  ) -> dict[str, Any]:
        """Delegate to :meth:`V543Assembly.assert_gradient_isolation`."""
        return self.assembly.assert_gradient_isolation(
            require_active_grad=require_active_grad)


def check_external_output(out: Mapping[str, Any]) -> None:
    """Assert the §4.3 :225 five-key contract is EXACTLY what was returned."""
    keys = set(out)
    if keys != set(EXTERNAL_OUTPUT_KEYS):
        raise V543ModelError(
            "V543_MODEL_OUTPUT_KEYS",
            f"external forward must return exactly "
            f"{list(EXTERNAL_OUTPUT_KEYS)} (plan §4.3 :225), got "
            f"{sorted(keys)}")
    u = out["u"]
    q = out["q"]
    if not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(q).all()):
        raise V543ModelError(
            "V543_MODEL_NONFINITE",
            "u/q contain non-finite values")


def identity_outputs_v543(out: Mapping[str, Any]) -> dict[str, Any]:
    """The zero-residual identity audit of plan §5.8 (:464–471).

    "零残差 V543(u,q,e) == V542(u,q,e)" and "order_score is q" — both are
    answered from ONE M0/P0 forward (``forward_debug``), so a caller can assert
    the A0 inheritance without re-running the encoder.  ``None`` means the
    quantity was not present (a non-debug output).
    """
    result: dict[str, Any] = {
        "order_score_is_q": out["order_score"] is out["q"],
        "q_equals_tanh_q0_raw": None,
        "u_equals_u0": None,
        "endpoint_equals_endpoint0": None,
        "encoder_forward_count": out.get("encoder_forward_count"),
    }
    if "q0_raw" in out and "delta_q_raw" in out:
        result["q_equals_tanh_q0_raw"] = bool(torch.equal(
            out["q"], torch.tanh(out["q0_raw"] + out["delta_q_raw"])))
    if "u0" in out and "key_permutation" in out:
        expected = unpermute_to_key_rows(out["u0"], out["key_permutation"])
        result["u_equals_u0"] = bool(torch.equal(expected, out["u"]))
    if "endpoint0_logit" in out and "delta_e" in out:
        result["endpoint_equals_endpoint0"] = bool(torch.equal(
            out["endpoint_logit"], out["endpoint0_logit"] + out["delta_e"]))
    return result


__all__ = [
    "DEBUG_OUTPUT_KEYS",
    "EXTERNAL_OUTPUT_KEYS",
    "V543_MODEL_SCHEMA",
    "V543_MODEL_VERSION",
    "V543Model",
    "V543ModelError",
    "check_external_output",
    "identity_outputs_v543",
]
