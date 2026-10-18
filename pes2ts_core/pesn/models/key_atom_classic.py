"""M0 Classic Key-Atom Pair-Driver — 0-parameter analytic baseline
(ADR-0026 §2.7, plan §3 M0).

For each key pair ``l = (i, j)`` (``batch.pair_index`` over key-atom
slots, ``batch.pair_edit`` = signed ΔB = B_other − B_start):

* ``e_l = (x_j − x_i) / d_ij`` from the CURRENT positions;
* ``s_l = sign(ΔB_l)``; accumulate ``v_i += s_l·e_l``, ``v_j −= s_l·e_l``
  (forming bonds point toward each other, breaking bonds apart — the
  start→other direction is already carried by the sign of ΔB);
* per key atom divide by ``max(1, degK_i)``;
* ``U = d0 · v / max(RMS_A(v), 1e−8)`` with ``d0`` from ``batch.d0``;
* then a SINGLE uniform factor per frame so ``max_atom ≤ 0.20 Å``
  (plan §3 M0 / §4: teacher cap);
* degeneracy: if the frame's ``RMS_A(v) < 1e−8`` the frame outputs zeros
  and is flagged (typed degeneracy, plan §3 M0).

Pure torch function over a ``KeyAtomBatch``; no parameters, no learning.
It is a geometric analytic baseline and cannot guarantee that every
actual IRC's local bond distances are monotone (plan §3 M0).
"""
from __future__ import annotations

from typing import Final

import torch

from pes2ts_core.pesn.contracts.key_atom import KeyAtomBatch

CLASSIC_BASELINE_VERSION: Final[str] = "pesn_key_atom_classic_v1"

#: Uniform per-frame max-atom cap (Å), plan §3 M0.
MAX_ATOM_CAP: Final[float] = 0.20

#: Degeneracy threshold on RMS_A(v) (Å-less accumulation).
DEGENERACY_EPS: Final[float] = 1e-8


def classic_pair_driver(
    batch: KeyAtomBatch,
    *,
    max_atom_cap: float = MAX_ATOM_CAP,
) -> tuple[torch.Tensor, dict]:
    """Analytic pair-driver displacement for one batch.

    Returns ``(u [Σm,3] float32, info)`` where ``info`` carries
    ``degenerate_frames`` (int64 CPU tensor, 1 = zeroed frame) and the
    applied ``scale`` per frame (float32 CPU tensor) — the single common
    factor per frame after the cap.  ``d0`` is read per frame from
    ``batch.d0`` (required; frames missing d0 fall back to 0.08 Å).
    """
    torch_fp = torch.float32
    key_index = batch.key_index
    key_ptr = batch.key_ptr
    n_keys = int(key_index.numel())
    n_graphs = int(key_ptr.shape[0]) - 1
    device = batch.pos.device
    out = torch.zeros(n_keys, 3, dtype=torch_fp, device=device)
    degenerate = torch.zeros(n_graphs, dtype=torch.int64)
    scale = torch.ones(n_graphs, dtype=torch_fp)
    if n_keys == 0:
        return out, {"degenerate_frames": degenerate, "scale": scale}

    x_key = batch.pos[key_index]  # [Σm,3] current key-atom positions

    # --- accumulate v from key pairs ----------------------------------------
    v = torch.zeros(n_keys, 3, dtype=batch.pos.dtype, device=device)
    pair_index = batch.pair_index
    if pair_index is not None and pair_index.numel() > 0:
        from pes2ts_core.pesn.contracts.key_atom import (
            global_pair_rows_to_slots,
        )
        pair_index = global_pair_rows_to_slots(
            pair_index, key_index, int(batch.pos.shape[0]))
    if pair_index is not None and pair_index.numel() > 0:
        pi = pair_index[0].long()
        pj = pair_index[1].long()
        d = x_key[pj] - x_key[pi]
        dist = d.norm(dim=-1)
        safe = dist > 1e-12
        e = torch.where(
            safe.unsqueeze(-1), d / dist.clamp(min=1e-12).unsqueeze(-1),
            torch.zeros_like(d))
        s = torch.sign(batch.pair_edit).unsqueeze(-1).to(dtype=e.dtype)
        contrib = s * e
        v = v.index_add(0, pi, contrib)
        v = v.index_add(0, pj, -contrib)

    # per-atom degK normalization: divide by max(1, degK_i)
    deg = batch.degK.to(dtype=v.dtype).clamp(min=1.0).unsqueeze(-1)
    v = v / deg

    # --- per-frame scale: U = d0·v / max(RMS_A(v), eps), then cap ---------
    d0 = batch.d0
    for b in range(n_graphs):
        lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
        if hi <= lo:
            continue
        v_b = v[lo:hi]
        rms = float(torch.sqrt(torch.mean(torch.sum(v_b * v_b, dim=-1))))
        d0_b = float(d0[b]) if d0 is not None else 0.08
        if not (d0_b > 0.0):
            d0_b = 0.08
        if rms < DEGENERACY_EPS:
            degenerate[b] = 1
            continue  # out[lo:hi] stays zero (typed degeneracy)
        u_b = v_b * (d0_b / rms)
        max_atom = float(u_b.norm(dim=-1).max())
        factor = 1.0
        if max_atom > float(max_atom_cap):
            factor = float(max_atom_cap) / max_atom
        out[lo:hi] = u_b.to(torch_fp) * factor
        scale[b] = factor

    return out, {"degenerate_frames": degenerate, "scale": scale}


__all__ = [
    "CLASSIC_BASELINE_VERSION",
    "DEGENERACY_EPS",
    "MAX_ATOM_CAP",
    "classic_pair_driver",
]
