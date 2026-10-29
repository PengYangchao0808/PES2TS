"""V5.4.2 ``DirectMoveDecoder`` — the single shared MOVE readout (ADR-0033 §2.1).

One module serves MOVE for EVERY candidate (A0–A5): the learnable surface
is exactly ``w_v = nn.Linear(48, 1)`` (**49 params**); ``a_ref = 0.08 Å``
is a FROZEN constant — there is NO constructor argument for it, the
attribute is read-only (property without setter), and ``forward``
re-asserts the value before any math (config
``move_decoder.a_ref_learnable: false``; gate-1 B1: a run identity claiming
``a_ref_angstrom=0.08`` must be structurally unable to compute MOVE with
any other value).

Math — verbatim the legacy v3 vector head
(:mod:`pes2ts_core.pesn.models.key_atom_egnn_v3`, forward lines 252–268),
reproduced BIT-for-bit in fp32 (CPU bitwise parity is the A0 certificate
premise, ADR-0033 §2.2)::

    coeff   = tanh(w_v(edge_msg))                    # [E,1] signed
    rel     = pos[src] - pos[dst]                    # r_ij, receiver = dst
    unit    = rel / (‖rel‖ + 0.1)
    contrib = coeff * unit
    u[key] += Σ over edges with dst ∈ key set       # index_add, key slots
    u       = (a_ref / max(1, degK[key_index]))·u

Interface discipline (config ``move_decoder`` block):

* inputs are exactly five tensors ``edge_msg, pos, edge_index, key_index,
  degK`` — the decoder NEVER receives the batch mapping, so it cannot
  structurally read supervision keys (reads_supervision_keys: false);
* output row order strictly equals ``key_index``; non-key atoms get
  structurally-zero rows (only edges whose ``dst`` is a key atom
  contribute);
* forbidden: absolute coordinates / centroid / orientation / any edge-MLP
  recompute — the decoder consumes the backbone's ``edge_msg`` as-is.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

from pes2ts_core.pesn.models.v542.interfaces import (
    A_REF_V542,
    EDGE_MSG_DIM_V542,
    N_DECODER_PARAMS_V542,
    V542InitError,
    V542ParamBudgetError,
)

#: The ONLY learnable parameter name of the decoder.
W_V_PARAM: Final[str] = "w_v"


def assert_decoder_a_ref_frozen(decoder: "DirectMoveDecoder") -> None:
    """Constructor-time gate-1 B1 check: the decoder's ``a_ref`` is 0.08.

    Called by :class:`~pes2ts_core.pesn.models.v542.backbone_a0.A0LegacyBackbone`
    and :class:`~pes2ts_core.pesn.models.v542.model.V542Model` when a
    caller-supplied decoder is adopted, so a (subclassed) decoder with a
    wrong reference step is a typed refusal at every assembly seam.
    """
    value = getattr(decoder, "a_ref", None)
    if not isinstance(value, (int, float)) or float(value) != A_REF_V542:
        raise V542InitError(
            "V542_DECODER_A_REF",
            f"decoder a_ref must be the frozen constant {A_REF_V542}, got "
            f"{value!r} ({type(decoder).__name__}) — a run identity "
            "claiming a_ref_angstrom=0.08 may not assemble any other value")


class DirectMoveDecoder(nn.Module):
    """Shared MOVE head: ``u = (a_ref/degK)·Σ tanh(w_v·m_ij)·r̂_ij`` (legacy-exact).

    ``a_ref`` is FROZEN at ``A_REF_V542 = 0.08`` (gate-1 B1):

    * there is NO ``a_ref`` constructor argument — ``DirectMoveDecoder
      (a_ref=...)`` is a ``TypeError`` (the constant is not configurable);
    * ``a_ref`` is a read-only property backed by ``_a_ref`` — attribute
      assignment raises ``AttributeError``;
    * ``forward`` re-asserts ``self._a_ref == A_REF_V542`` before any
      math, so even a hand-patched ``_a_ref`` cannot change MOVE under a
      run identity that claims 0.08 (typed ``V542_A_REF_FROZEN``).

    ``named_parameters()`` yields exactly ``w_v.weight``/``w_v.bias``
    (49 params); the constructor asserts that invariant at every build.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_v = nn.Linear(EDGE_MSG_DIM_V542, 1)
        # Plain float attribute — NOT a Parameter/buffer/trainable state.
        # Backs the read-only ``a_ref`` property; never reassigned.
        self._a_ref: float = float(A_REF_V542)

        n_params = sum(int(p.numel()) for p in self.parameters())
        if n_params != N_DECODER_PARAMS_V542:
            # 49 is a frozen invariant; drift means the module surface
            # changed (e.g. a bias was dropped/added) — refuse to build.
            raise V542ParamBudgetError(
                "V542_DECODER_PARAMS",
                f"DirectMoveDecoder has {n_params} params, "
                f"expected exactly {N_DECODER_PARAMS_V542} (w_v = Linear(48,1))")

    # -- accounting ---------------------------------------------------------
    @property
    def param_count(self) -> int:
        """Exact trainable parameter count (49)."""
        return sum(int(p.numel()) for p in self.parameters())

    # -- frozen MOVE reference step (read-only; gate-1 B1) -------------------
    @property
    def a_ref(self) -> float:
        """Frozen MOVE reference step in Å (0.08) — read-only, unlearnable."""
        return self._a_ref

    def _assert_a_ref_frozen(self) -> float:
        """Typed refusal if the frozen constant was tampered with.

        Checks BOTH the backing ``_a_ref`` slot and the ``a_ref``
        property value (so a subclass overriding the property is also
        refused), and returns the validated constant for use in math.
        """
        value = self.__dict__.get("_a_ref", None)
        prop = getattr(self, "a_ref", None)
        if (not isinstance(value, (int, float))
                or float(value) != A_REF_V542
                or not isinstance(prop, (int, float))
                or float(prop) != A_REF_V542):
            raise V542InitError(
                "V542_A_REF_FROZEN",
                f"a_ref must be the frozen constant {A_REF_V542}, got "
                f"_a_ref={value!r} a_ref={prop!r} — MOVE math may not be "
                "changed under a run identity that claims "
                "a_ref_angstrom=0.08")
        return float(A_REF_V542)

    # -- forward (legacy-exact; fp32 bitwise) -------------------------------
    def forward(
            self,
            edge_msg: torch.Tensor,
            pos: torch.Tensor,
            edge_index: torch.Tensor,
            key_index: torch.Tensor,
            degK: torch.Tensor,
    ) -> torch.Tensor:
        """``edge_msg [E,48]`` → ``u [K,3]`` in strict ``key_index`` order.

        ``pos`` is the current-frame node coordinates ``[N,3]``, ``edge_index``
        ``[2,E]`` (row 0 = src, row 1 = dst, matching the backbone's message
        convention), ``key_index`` ``[K]`` node rows of the key set, ``degK``
        ``[N]`` per-node key degree.  All math mirrors the legacy vector
        head op-for-op so CPU fp32 outputs are bitwise identical.
        """
        a_ref = self._assert_a_ref_frozen()   # gate-1 B1: frozen 0.08
        dtype = self.w_v.weight.dtype
        device = self.w_v.weight.device
        pos = pos.to(dtype=dtype)
        degK = degK.to(dtype=dtype)
        edge_index = edge_index.long()
        key_index = key_index.long()

        n_nodes = int(pos.shape[0])
        n_keys = int(key_index.shape[0])
        src, dst = edge_index[0], edge_index[1]

        u = torch.zeros(n_keys, 3, dtype=dtype, device=device)
        if int(edge_msg.shape[0]) > 0:
            coeff = torch.tanh(self.w_v(edge_msg))          # [E,1] signed
            rel = pos[src] - pos[dst]                        # r_ij, recv=dst
            unit = rel / (rel.norm(dim=-1, keepdim=True) + 0.1)
            contrib = coeff * unit
            key_slot = torch.full(
                (n_nodes,), -1, dtype=torch.long, device=device)
            key_slot[key_index] = torch.arange(
                n_keys, dtype=torch.long, device=device)
            dst_slot = key_slot[dst]
            keep = dst_slot >= 0
            if bool(keep.any()):
                u = u.index_add(0, dst_slot[keep], contrib[keep])
        deg_scale = (a_ref / degK[key_index].clamp(min=1.0)).unsqueeze(-1)
        return deg_scale * u


__all__ = [
    "W_V_PARAM",
    "DirectMoveDecoder",
    "assert_decoder_a_ref_frozen",
]
