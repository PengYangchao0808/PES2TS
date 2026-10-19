"""V5 sprint key-atom BC losses + trainer + CLI (ADR-0026 §2.8, plan §4).

Schema/version: ``pesn_key_atom_bc_v1`` (loss-schema bump for V5.1a runs:
``pesn_key_atom_bc_loss_v2``, see :data:`LOSS_SCHEMA_VERSION`).

Losses (plan §4, verbatim): radial Huber h_δ(e) = e²/2 (e≤δ) else
δ(e−δ/2), δ=0.25, on the normalized radial error e_i = ‖u_i−y_i‖/0.08;
per-frame degK weights w_i = 1/max(1, degK_i) normalized Σw_i=1;
L_pair = mean_l h((J_l(u)−J_l(y))/0.08) with J_l(u) = e_l·(u_j−u_i) and
e_l recomputed from the CURRENT geometry each batch; L_cos gated on the
teacher only (g_i = 1[‖y_i‖>0.02]) with soft-norm cosine (1e-4 Å);
λ_cos ramps 0 → 0.02 linearly over ``cos_ramp_updates`` after
``cos_start_update``; λ_pair = 0.5.  The batch loss is the plain mean
over frames — the reaction-balanced sampler already applies q weights,
so q is NOT multiplied again; the degK w_i normalization stays inside
each frame term.  Empty cos gate ⇒ L_cos = 0 exactly; all math is in Å;
non-finite values raise a typed abort (status ``aborted_nan``).

:class:`KeyAtomBCTrainer` implements the sprint §4/§5 recipe: fixed
update budget, Lane B pools for batch assembly (8 distinct reaction
groups × 4 frames, 2 per branch where possible, q-weighted draws from
the current cycle pool, cycle advance on exhaustion — INTEGRATION PENDING
on ``pes2ts_core.pesn.offline.data.key_atom_dataset``), paired
initialization/sample order across arms (``torch.manual_seed(seed)``
before model construction AND before sampler creation), AdamW
1e-3/1e-4 with linear warmup 250 then cosine to 1e-4, grad clip 1.0,
fp32, dev-fold DENSE eval every 250 updates with reaction-cluster
median/mean MAE/RMSE, RMSE/zero and RMSE/Classic ratios, effective-atom
cos medians (overall and near-TS), ``eval.jsonl`` /
``checkpoint_last.pt`` / ``checkpoint_final.pt`` / ``metrics.json``
(raw AND bounded via :func:`bounded_projection`) / ``status.json``.
Early stopping exists ONLY in screen mode (inner_val reaction-cluster
median RMSE/zero, patience 8, min_delta 0.005, no stop before update
500); formal mode runs the fixed budget with
``formal_outer_early_stopping=false``.  V5.1a (plan §3.2–§3.4) adds
trainer mode ``v51``: a family/size-stratified 32/8 inner split of the
outer TRAIN clusters (frozen seed 20261019, sha256 recorded), training
on inner-train pools only, selection among inner-val checkpoints
(rmse ≤ R*×1.02 → lowest SRE → highest near-TS cos) with patience 8 /
min_delta 0.005 / floor update 500, ``checkpoint_best.pt`` +
``checkpoint_last.pt`` + ``checkpoint_final.pt``, exact-Δd trio pair
losses + optional magnitude head, and a POST-HOC outer dev_dense eval of
best+final weights (tagged ``dev_dense_best`` / ``dev_dense_final`` in
eval.jsonl) that checkpoint selection never sees.  ``run_record.json``
carries the ADR-0025 ``gv`` block validated by ``validate_gv_run_record``
plus v51 fields (best_update / inner_val_score / inner_val_split_hash /
alpha_median / dead_zone_fraction / ``selection_exposed: false``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.config_loader import load_config_mapping
from pes2ts_core.gv_registry import load_gv_map, validate_gv_run_record
from pes2ts_core.pesn.contracts.key_atom import KeyAtomBatch
from pes2ts_core.pesn.models.key_atom_classic import (
    MAX_ATOM_CAP,
    classic_pair_driver,
)
from pes2ts_core.pesn.models.key_atom_egnn import A_REF, StarEGNN

SCHEMA_VERSION: Final[str] = "pesn_key_atom_bc_v1"
#: V5.1a loss-schema bump (plan §3.3): exact-Δd trio + magnitude
#: supervision terms; emitted ONLY in v51-mode run records so V5 screen/
#: formal artifacts keep their frozen ``pesn_key_atom_bc_v1`` stamp.
LOSS_SCHEMA_VERSION: Final[str] = "pesn_key_atom_bc_loss_v2"
RUN_RECORD_SCHEMA: Final[str] = "pesn_key_atom_bc_run_v1"
CHECKPOINT_SCHEMA: Final[str] = "pesn_key_atom_bc_checkpoint_v1"
CAMPAIGN_ID: Final[str] = "V5_SPRINT"
EXPECTED_CONFIG_SCHEMA: Final[str] = "pesn_v5_sprint_draft2"
#: V5.1a config schema (config/campaigns/v5_1_sprint.yaml, driver lane);
#: the trainer accepts BOTH schemas in every mode — the v51 yaml is a
#: superset of the V5 loss keys and legacy modes default identically.
ACCEPTED_CONFIG_SCHEMAS: Final[tuple[str, ...]] = (
    "pesn_v5_sprint_draft2", "pesn_v5_1_sprint_v1")

#: Status vocabulary (plan §5.7 / sprint contract + V5.1a addition).
STATUS_COMPLETE: Final[str] = "complete"
STATUS_CENSORED_SCREEN: Final[str] = "censored_screen"
STATUS_ABORTED_DATA: Final[str] = "aborted_data"
STATUS_ABORTED_NAN: Final[str] = "aborted_nan"
#: V5.1a (plan §3.4): inner-val R* stalled (patience 8, min_delta 0.005).
#: Successful v51 runs stay STATUS_COMPLETE — the landed V5.1 campaign
#: driver keys run-success off ``status == "complete"``.
STATUS_CENSORED_PATIENCE: Final[str] = "censored_patience"
STATUS_VALUES: Final[tuple[str, ...]] = (
    STATUS_COMPLETE, STATUS_CENSORED_SCREEN,
    STATUS_ABORTED_DATA, STATUS_ABORTED_NAN,
    STATUS_CENSORED_PATIENCE)


class KeyAtomBCTrainerError(RuntimeError):
    """Typed trainer refusal carrying a machine-readable code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class KeyAtomLossError(RuntimeError):
    """Typed loss failure (non-finite math or malformed tensors)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise KeyAtomBCTrainerError(code, detail)


def _require_loss(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise KeyAtomLossError(code, detail)


# ---------------------------------------------------------------------------
# Loss configuration + λ_cos ramp (plan §4)
# ---------------------------------------------------------------------------

#: V5.1a pair-loss arms (plan §3.3): ``legacy_j`` keeps the V5 first-order
#: J term verbatim; ``exact_trio`` swaps in exact-Δd regression + sign
#: margin + dead-zone penalty.
PAIR_LOSS_MODES: Final[tuple[str, ...]] = ("legacy_j", "exact_trio")


@dataclass(frozen=True, slots=True)
class KeyAtomLossConfig:
    """Plan §4 loss hyper-parameters (sprint yaml ``pesn_v5_sprint_draft2``).

    V5.1a (plan §3.3) additions: ``pair_loss`` switch
    (``legacy_j`` | ``exact_trio``), the sign/dead pair terms, log-form
    magnitude supervision and the shared auxiliary ramp schedule.  All
    new fields default to the legacy V5 behavior so an unmodified V5
    yaml trains bit-identically.
    """

    loss_scale_angstrom: float = A_REF      # a_ref = 0.08 Å
    huber_delta: float = 0.25
    lambda_pair: float = 0.5
    lambda_cos_max: float = 0.02
    cos_start_update: int = 250
    cos_ramp_updates: int = 250
    tau_norm_angstrom: float = 0.02
    cos_soft_norm_angstrom: float = 1e-4
    # --- V5.1a (plan §3.3) ---
    pair_loss: str = "legacy_j"             # "legacy_j" | "exact_trio"
    lambda_mag: float = 0.5
    lambda_sign: float = 0.2
    lambda_dead: float = 0.1
    pair_sign_tau_angstrom: float = 0.005   # τ_sign
    pair_dead_tau_angstrom: float = 0.005   # τ_dead
    #: Active-pair gate |Δd_true| ≥ 0.005 Å AND the dead-zone boundary
    #: |Δd_pred| < 0.005 Å (same frozen 0.005 as G53_SIGN_PAIR_THRESHOLD).
    pair_active_threshold_angstrom: float = 0.005
    #: L_mag teacher gate: only key atoms with ‖y‖ > 0.01 Å supervise s.
    mag_supervision_tau_angstrom: float = 0.01
    #: Shared linear ramp for ALL auxiliary terms (cos/mag/sign/dead):
    #: zero for the first ``aux_start_update`` updates, then linear over
    #: ``aux_ramp_updates`` (plan §3.3: 前 250 步只开 L_atom + L_pair_reg).
    aux_start_update: int = 250
    aux_ramp_updates: int = 250

    def __post_init__(self) -> None:
        if self.pair_loss not in PAIR_LOSS_MODES:
            raise KeyAtomLossError(
                "PAIR_LOSS_MODE",
                f"pair_loss must be one of {PAIR_LOSS_MODES}, got "
                f"{self.pair_loss!r}")

    @classmethod
    def from_campaign_config(cls, config: Mapping) -> "KeyAtomLossConfig":
        # V5.1a yaml (driver lane) nests the new knobs under ``v51:``;
        # flat top-level spellings remain accepted so ad-hoc/test configs
        # stay ergonomic.  Precedence: nested v51 block > flat key >
        # legacy default.
        v51 = dict(config.get("v51", {}) or {})
        mag_blk = dict(v51.get("magnitude_head", {}) or {})
        pair_blk = dict(v51.get("pair_exact", {}) or {})
        ramp_blk = dict(v51.get("ramp", {}) or {})
        return cls(
            loss_scale_angstrom=float(config.get("a_ref_angstrom", A_REF)),
            huber_delta=float(config.get("huber_delta_normalized", 0.25)),
            lambda_pair=float(config.get("lambda_pair", 0.5)),
            lambda_cos_max=float(config.get("lambda_cos", 0.02)),
            cos_start_update=int(config.get("cos_start_update", 250)),
            cos_ramp_updates=int(config.get("cos_ramp_updates", 250)),
            tau_norm_angstrom=float(config.get("tau_norm_angstrom", 0.02)),
            cos_soft_norm_angstrom=float(
                config.get("cos_soft_norm_angstrom", 1e-4)),
            pair_loss=str(config.get("pair_loss", "legacy_j")),
            lambda_mag=float(mag_blk.get(
                "lambda_mag", config.get("lambda_mag", 0.5))),
            lambda_sign=float(pair_blk.get(
                "lambda_sign", config.get("lambda_sign", 0.2))),
            lambda_dead=float(pair_blk.get(
                "lambda_dead", config.get("lambda_dead", 0.1))),
            pair_sign_tau_angstrom=float(pair_blk.get(
                "tau_sign_angstrom",
                config.get("pair_sign_tau_angstrom", 0.005))),
            pair_dead_tau_angstrom=float(pair_blk.get(
                "tau_dead_angstrom",
                config.get("pair_dead_tau_angstrom", 0.005))),
            pair_active_threshold_angstrom=float(pair_blk.get(
                "active_threshold_angstrom",
                config.get("pair_active_threshold_angstrom", 0.005))),
            mag_supervision_tau_angstrom=float(
                config.get("mag_supervision_tau_angstrom", 0.01)),
            aux_start_update=int(ramp_blk.get(
                "start_update", config.get("aux_start_update", 250))),
            aux_ramp_updates=int(ramp_blk.get(
                "ramp_updates", config.get("aux_ramp_updates", 250))),
        )


def cos_lambda(update: int, config: KeyAtomLossConfig) -> float:
    """λ_cos at a 1-based update count (plan §4 ramp).

    Exactly 0 before ``cos_start_update``, linear to ``lambda_cos_max``
    over the next ``cos_ramp_updates`` updates, constant afterwards.
    """
    start = int(config.cos_start_update)
    ramp = max(int(config.cos_ramp_updates), 1)
    if update < start:
        return 0.0
    frac = min(1.0, (update - start) / ramp)
    return float(config.lambda_cos_max) * frac


def aux_ramp_fraction(update: int, config: KeyAtomLossConfig) -> float:
    """Shared linear ramp fraction ∈ [0,1] for the V5.1a auxiliary terms
    (mag/sign/dead; plan §3.3: first 250 updates only
    ``L_atom + L_pair_reg``, then 250 updates linear ramp to λ).

    The cos term keeps its own cos_lambda ramp — with the frozen default
    250/250 schedule the two coincide exactly.
    """
    start = int(config.aux_start_update)
    ramp = max(int(config.aux_ramp_updates), 1)
    if update < start:
        return 0.0
    return float(min(1.0, (update - start) / ramp))


def exact_delta_d_torch(
    pos_key: torch.Tensor,
    pair_index: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Differentiable EXACT finite-step Δd per key pair (plan §3.3).

    ``Δd_l(u) = ‖r_l + u_j − u_i‖ − ‖r_l‖`` with ``r_l`` from the CURRENT
    frame key-atom positions ``pos_key`` (slot order) — the torch twin of
    :func:`pes2ts_core.pesn.offline.evaluation.key_atom_protocols.exact_delta_d`.
    ``pair_index`` is [2, P] in KEY-SLOT space.
    """
    if pair_index is None or pair_index.numel() == 0:
        return torch.zeros(0, dtype=u.dtype, device=u.device)
    pi = pair_index[0].long()
    pj = pair_index[1].long()
    r = pos_key[pj] - pos_key[pi]
    return (r + u[pj] - u[pi]).norm(dim=-1) - r.norm(dim=-1)


def _apply_pair_edit_filter(
    pair_index: torch.Tensor | None,
    pair_edit: torch.Tensor | None,
    *,
    mode: str,
) -> torch.Tensor | None:
    """Keep only changed pairs (``pair_edit != 0``) for exact_trio mode.

    The V5 legacy J term is defined over EVERY pair supplied by the
    caller, so filtering is applied only for ``exact_trio`` (same pair
    enumeration as G5.3 / ``_static_diagnostics``).
    """
    if (pair_index is None or pair_index.numel() == 0
            or pair_edit is None or mode != "exact_trio"):
        return pair_index
    keep = pair_edit.to(device=pair_index.device) != 0
    return pair_index[:, keep]


# ---------------------------------------------------------------------------
# Loss building blocks (plan §4 EXACT)
# ---------------------------------------------------------------------------

def key_atom_bc_loss_vectorized(
    u: torch.Tensor,
    y: torch.Tensor,
    degK: torch.Tensor,
    key_ptr: torch.Tensor,
    pos: torch.Tensor,
    key_index: torch.Tensor,
    pair_index: torch.Tensor | None = None,
    *,
    config: KeyAtomLossConfig | None = None,
    lambda_cos: float = 0.0,
    frame_mask: torch.Tensor | None = None,
    pair_edit: torch.Tensor | None = None,
    s: torch.Tensor | None = None,
    lambda_mag: float = 0.0,
    lambda_sign: float = 0.0,
    lambda_dead: float = 0.0,
) -> dict:
    """Vectorized twin of :func:`key_atom_bc_loss` (identical math).

    Segment reductions replace the per-frame Python loop; the loop version
    stays as the reference and the trainer asserts agreement on real
    batches (same formulas, plan §4; no per-frame GPU sync points).

    V5.1a (plan §3.3): with ``config.pair_loss == "exact_trio"`` the pair
    term is the exact-Δd regression over changed pairs (``pair_edit != 0``)
    plus sign-margin / dead-zone terms on teacher-active pairs
    (|Δd_true| ≥ 0.005 Å); ``s`` (per-key-atom EFFECTIVE magnitudes —
    the post-deg_scale contribution to ‖u‖ as stashed by the model in
    ``last_magnitudes``, NOT the raw softplus head output) enables
    log-form L_mag supervision toward ‖y‖.  Zero multipliers keep the
    legacy totals bit-identical.
    """
    cfg = config or KeyAtomLossConfig()
    _require_loss(bool(torch.isfinite(u).all()),
                  "NONFINITE_PREDICTION", "u contains non-finite values")
    _require_loss(bool(torch.isfinite(y).all()),
                  "NONFINITE_TEACHER", "y contains non-finite values")
    from pes2ts_core.pesn.contracts.key_atom import global_pair_rows_to_slots
    if pair_index is not None and pair_index.numel() > 0:
        pair_index = global_pair_rows_to_slots(
            pair_index, key_index, int(pos.shape[0]))
    pair_index = _apply_pair_edit_filter(
        pair_index, pair_edit, mode=cfg.pair_loss)
    if (cfg.pair_loss == "exact_trio" and pair_index is not None
            and pair_index.numel() > 0):
        _require_loss(pair_edit is not None, "EXACT_TRIO_NEEDS_PAIR_EDIT",
                      "exact_trio pair loss requires batch.pair_edit")
    if s is not None:
        _require_loss(s.shape == (u.shape[0],),
                      "MAGNITUDE_SHAPE",
                      f"s (effective magnitudes) must be [{u.shape[0]}], "
                      f"got {tuple(s.shape)}")
    n_graphs = int(key_ptr.shape[0]) - 1
    scale = float(cfg.loss_scale_angstrom)
    device = u.device
    dtype = u.dtype
    lo = key_ptr[:-1]
    hi = key_ptr[1:]
    counts = (hi - lo).clamp(min=1)
    active = (hi > lo)
    if frame_mask is not None:
        active = active & frame_mask.to(device=device, dtype=torch.bool)
    n_used = int(active.sum())
    _require_loss(n_used > 0, "NO_ELIGIBLE_FRAMES",
                  "batch has no unmasked frames with key atoms")

    atom_frame = torch.repeat_interleave(
        torch.arange(n_graphs, device=device), (hi - lo).clamp(min=0))
    # w_i = 1/max(1,degK) normalized per frame (segment softmax-free form)
    w_raw = (1.0 / degK.to(dtype=dtype).clamp(min=1.0))
    w_sum = torch.zeros(n_graphs, dtype=dtype, device=device)
    w_sum.index_add_(0, atom_frame, w_raw)
    w = w_raw / w_sum[atom_frame].clamp(min=1e-12)

    e = (u - y).norm(dim=-1) / scale
    l_atom_atoms = w * radial_huber(e, cfg.huber_delta)
    l_atom = torch.zeros(n_graphs, dtype=dtype, device=device)
    l_atom.index_add_(0, atom_frame, l_atom_atoms)

    l_pair = torch.zeros(n_graphs, dtype=dtype, device=device)
    l_sign = torch.zeros(n_graphs, dtype=dtype, device=device)
    l_dead = torch.zeros(n_graphs, dtype=dtype, device=device)
    n_pairs_active = 0
    if pair_index is not None and pair_index.numel() > 0:
        pi = pair_index[0].long()
        pj = pair_index[1].long()
        pair_frame = atom_frame[pi]
        pair_sum = torch.zeros(n_graphs, dtype=dtype, device=device)
        pair_cnt = torch.zeros(n_graphs, dtype=dtype, device=device)
        if cfg.pair_loss == "exact_trio":
            # plan §3.3: EXACT Δd_l(u) = ‖r_l + u_j − u_i‖ − ‖r_l‖
            x_key = pos[key_index]
            dd_u = exact_delta_d_torch(x_key, pair_index, u)
            dd_y = exact_delta_d_torch(x_key, pair_index, y)
            pair_h = radial_huber(
                ((dd_u - dd_y).abs() / scale).to(dtype=dtype),
                cfg.huber_delta)
            pair_sum.index_add_(0, pair_frame, pair_h)
            pair_cnt.index_add_(
                0, pair_frame, torch.ones_like(pair_h))
            thr = float(cfg.pair_active_threshold_angstrom)
            is_active = dd_y.abs() >= thr
            n_pairs_active += int(is_active.sum())
            if bool(is_active.any()):
                a = is_active
                sign_term = torch.nn.functional.softplus(
                    -torch.sign(dd_y[a]) * dd_u[a]
                    / float(cfg.pair_sign_tau_angstrom))
                dead_term = torch.nn.functional.softplus(
                    (thr - dd_u[a].abs())
                    / float(cfg.pair_dead_tau_angstrom))
                s_sum = torch.zeros(n_graphs, dtype=dtype, device=device)
                s_cnt = torch.zeros(n_graphs, dtype=dtype, device=device)
                s_sum.index_add_(0, pair_frame[a], sign_term.to(dtype))
                s_cnt.index_add_(
                    0, pair_frame[a], torch.ones_like(sign_term))
                l_sign = torch.where(
                    s_cnt > 0, s_sum / s_cnt.clamp(min=1.0),
                    torch.zeros_like(s_sum))
                d_sum = torch.zeros(n_graphs, dtype=dtype, device=device)
                d_cnt = torch.zeros(n_graphs, dtype=dtype, device=device)
                d_sum.index_add_(0, pair_frame[a], dead_term.to(dtype))
                d_cnt.index_add_(
                    0, pair_frame[a], torch.ones_like(dead_term))
                l_dead = torch.where(
                    d_cnt > 0, d_sum / d_cnt.clamp(min=1.0),
                    torch.zeros_like(d_sum))
        else:
            e_l = pair_unit_vectors(pos, key_index, pair_index)
            dj = (e_l * (u[pj] - u[pi])).sum(dim=-1)
            dy = (e_l * (y[pj] - y[pi])).sum(dim=-1)
            pair_h = radial_huber(
                ((dj - dy).abs() / scale).to(dtype=dtype), cfg.huber_delta)
            pair_sum.index_add_(0, pair_frame, pair_h)
            pair_cnt.index_add_(0, pair_frame, torch.ones_like(pair_h))
        l_pair = pair_sum / pair_cnt.clamp(min=1.0)

    g = (y.norm(dim=-1) > float(cfg.tau_norm_angstrom)).to(dtype)
    c = soft_cosine(u, y, cfg.cos_soft_norm_angstrom).to(dtype=dtype)
    wg = w * g
    denom = torch.zeros(n_graphs, dtype=dtype, device=device)
    denom.index_add_(0, atom_frame, wg)
    num = torch.zeros(n_graphs, dtype=dtype, device=device)
    num.index_add_(0, atom_frame, wg * (1.0 - c))
    l_cos = num / denom.clamp(min=1e-8)
    l_cos = torch.where(denom > 0.0, l_cos,
                        torch.zeros_like(l_cos))

    # L_mag (plan §3.3): radial huber of the log magnitude ratio on
    # teacher-gated atoms (‖y‖ > 0.01 Å); zero when the head is off.
    # `s` carries the EFFECTIVE magnitude (post deg_scale contribution
    # to ‖u‖) so the optimum drives ‖u‖ ≈ ‖y‖, not raw-s ≈ ‖y‖.
    l_mag = torch.zeros(n_graphs, dtype=dtype, device=device)
    if s is not None:
        y_norm = y.norm(dim=-1)
        gate_mag = (y_norm > float(cfg.mag_supervision_tau_angstrom)).to(dtype)
        log_r = torch.log(
            (s.to(dtype=dtype) + 1e-6) / (y_norm + 1e-6))
        mag_atom = radial_huber(log_r, cfg.huber_delta) * gate_mag
        mag_sum = torch.zeros(n_graphs, dtype=dtype, device=device)
        mag_sum.index_add_(0, atom_frame, mag_atom)
        mag_cnt = torch.zeros(n_graphs, dtype=dtype, device=device)
        mag_cnt.index_add_(0, atom_frame, gate_mag)
        l_mag = torch.where(
            mag_cnt > 0, mag_sum / mag_cnt.clamp(min=1.0),
            torch.zeros_like(mag_sum))

    frames = (l_atom + float(cfg.lambda_pair) * l_pair
              + float(lambda_cos) * l_cos
              + float(lambda_mag) * l_mag
              + float(lambda_sign) * l_sign
              + float(lambda_dead) * l_dead)
    keep = active.to(dtype)
    total = (frames * keep).sum() / n_used
    _require_loss(bool(torch.isfinite(total)), "NONFINITE_LOSS",
                  "total loss is not finite")
    return {
        "loss": total,
        "l_atom": (l_atom * keep).sum() / n_used,
        "l_pair": (l_pair * keep).sum() / n_used,
        "l_pair_reg": (l_pair * keep).sum() / n_used,
        "l_cos": (l_cos * keep).sum() / n_used,
        "l_mag": (l_mag * keep).sum() / n_used,
        "l_sign": (l_sign * keep).sum() / n_used,
        "l_dead": (l_dead * keep).sum() / n_used,
        "lambda_cos": float(lambda_cos),
        "lambda_mag": float(lambda_mag),
        "lambda_sign": float(lambda_sign),
        "lambda_dead": float(lambda_dead),
        "pair_loss": str(cfg.pair_loss),
        "n_pairs_active": int(n_pairs_active),
        "n_frames": n_used,
    }


def radial_huber(e: torch.Tensor, delta: float = 0.25) -> torch.Tensor:
    """h_δ(e) = e²/2 (e≤δ) else δ(e−δ/2); rotation-invariant by construction
    because it consumes only the radial error e = ‖Δu‖/a_ref."""
    quad = 0.5 * e * e
    linear = delta * (e - 0.5 * delta)
    return torch.where(e <= delta, quad, linear)


def degk_frame_weights(degK: torch.Tensor) -> torch.Tensor:
    """w_i = 1/max(1, degK_i) normalized so Σw_i = 1 (per frame)."""
    w = 1.0 / degK.clamp(min=1.0)
    total = w.sum()
    _require_loss(bool(torch.isfinite(total)) and float(total) > 0.0,
                  "WEIGHTS_DEGENERATE", "per-frame degK weights sum to 0")
    return w / total


def pair_unit_vectors(
    pos: torch.Tensor,
    key_index: torch.Tensor,
    pair_index: torch.Tensor,
) -> torch.Tensor:
    """e_l = (x_j − x_i)/d_ij from the CURRENT geometry (recomputed per
    batch; plan §4).  ``pair_index`` indexes key-atom slots."""
    if pair_index.numel() == 0:
        return torch.zeros(0, 3, dtype=pos.dtype, device=pos.device)
    pi = pair_index[0].long()
    pj = pair_index[1].long()
    x = pos[key_index]
    d = x[pj] - x[pi]
    dist = d.norm(dim=-1)
    safe = dist > 1e-12
    return torch.where(
        safe.unsqueeze(-1), d / dist.clamp(min=1e-12).unsqueeze(-1),
        torch.zeros_like(d))


def pair_projections(
    u: torch.Tensor, pair_index: torch.Tensor, e_l: torch.Tensor
) -> torch.Tensor:
    """J_l(u) = e_l·(u_j − u_i) — linear in u (key-slot pair indices)."""
    if pair_index.numel() == 0:
        return torch.zeros(0, dtype=u.dtype, device=u.device)
    pi = pair_index[0].long()
    pj = pair_index[1].long()
    return (e_l * (u[pj] - u[pi])).sum(dim=-1)


def soft_cosine(
    u: torch.Tensor, y: torch.Tensor, soft: float = 1e-4
) -> torch.Tensor:
    """Soft-norm cosine per atom (plan §4): the 1e-4 Å soft norm keeps the
    zero附近 gradient finite."""
    num = (u * y).sum(dim=-1)
    den = torch.sqrt(u.pow(2).sum(dim=-1) + soft * soft) * torch.sqrt(
        y.pow(2).sum(dim=-1) + soft * soft)
    return num / den.clamp(min=1e-12)


def key_atom_bc_loss(
    u: torch.Tensor,
    y: torch.Tensor,
    degK: torch.Tensor,
    key_ptr: torch.Tensor,
    pos: torch.Tensor,
    key_index: torch.Tensor,
    pair_index: torch.Tensor | None = None,
    *,
    config: KeyAtomLossConfig | None = None,
    lambda_cos: float = 0.0,
    frame_mask: torch.Tensor | None = None,
    pair_edit: torch.Tensor | None = None,
    s: torch.Tensor | None = None,
    lambda_mag: float = 0.0,
    lambda_sign: float = 0.0,
    lambda_dead: float = 0.0,
) -> dict:
    """Total loss + components (plan §4): per frame
    ``L_atom + λ_pair·L_pair + λ_cos·L_cos`` (+ V5.1a mag/sign/dead,
    see :func:`key_atom_bc_loss_vectorized`) with w_i normalized INSIDE
    the frame term; batch loss = mean over unmasked frames (q weights are
    the sampler's job and are NOT re-applied here).

    Raises ``KeyAtomLossError`` on non-finite inputs or results; an empty
    cos gate yields L_cos = 0 exactly; empty active-pair sets yield
    L_sign = L_dead = 0 exactly (never NaN).
    """
    cfg = config or KeyAtomLossConfig()
    _require_loss(bool(torch.isfinite(u).all()),
                  "NONFINITE_PREDICTION", "u contains non-finite values")
    _require_loss(bool(torch.isfinite(y).all()),
                  "NONFINITE_TEACHER", "y contains non-finite values")
    if pair_index is not None and pair_index.numel() > 0:
        from pes2ts_core.pesn.contracts.key_atom import (
            global_pair_rows_to_slots,
        )
        pair_index = global_pair_rows_to_slots(
            pair_index, key_index, int(pos.shape[0]))
    pair_index = _apply_pair_edit_filter(
        pair_index, pair_edit, mode=cfg.pair_loss)
    if (cfg.pair_loss == "exact_trio" and pair_index is not None
            and pair_index.numel() > 0):
        _require_loss(pair_edit is not None, "EXACT_TRIO_NEEDS_PAIR_EDIT",
                      "exact_trio pair loss requires batch.pair_edit")
    if s is not None:
        _require_loss(s.shape == (u.shape[0],),
                      "MAGNITUDE_SHAPE",
                      f"s (effective magnitudes) must be [{u.shape[0]}], "
                      f"got {tuple(s.shape)}")
    n_graphs = int(key_ptr.shape[0]) - 1
    scale = float(cfg.loss_scale_angstrom)
    total = torch.zeros((), dtype=u.dtype, device=u.device)
    n_used = 0
    l_atom_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    l_pair_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    l_cos_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    l_mag_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    l_sign_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    l_dead_acc = torch.zeros((), dtype=u.dtype, device=u.device)
    n_pairs_active = 0
    for b in range(n_graphs):
        if frame_mask is not None and not bool(frame_mask[b]):
            continue
        lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
        u_b, y_b, w_b = u[lo:hi], y[lo:hi], degK[lo:hi]
        m = hi - lo
        if m == 0:
            continue
        w = degk_frame_weights(w_b)
        # L_atom = Σ w_i h(||u−y||/a_ref)
        e = (u_b - y_b).norm(dim=-1) / scale
        l_atom = (w * radial_huber(e, cfg.huber_delta)).sum()
        # L_pair = mean over this frame's key pairs
        l_pair = torch.zeros((), dtype=u.dtype, device=u.device)
        l_sign = torch.zeros((), dtype=u.dtype, device=u.device)
        l_dead = torch.zeros((), dtype=u.dtype, device=u.device)
        if pair_index is not None and pair_index.numel() > 0:
            in_frame = ((pair_index[0] >= lo) & (pair_index[0] < hi)
                        & (pair_index[1] >= lo) & (pair_index[1] < hi))
            if bool(in_frame.any()):
                sub = pair_index[:, in_frame]
                if cfg.pair_loss == "exact_trio":
                    x_key = pos[key_index]
                    dd_u = exact_delta_d_torch(x_key, sub, u)
                    dd_y = exact_delta_d_torch(x_key, sub, y)
                    l_pair = radial_huber(
                        (dd_u - dd_y).abs() / scale,
                        cfg.huber_delta).mean()
                    thr = float(cfg.pair_active_threshold_angstrom)
                    is_active = dd_y.abs() >= thr
                    n_pairs_active += int(is_active.sum())
                    if bool(is_active.any()):
                        l_sign = torch.nn.functional.softplus(
                            -torch.sign(dd_y[is_active]) * dd_u[is_active]
                            / float(cfg.pair_sign_tau_angstrom)).mean()
                        l_dead = torch.nn.functional.softplus(
                            (thr - dd_u[is_active].abs())
                            / float(cfg.pair_dead_tau_angstrom)).mean()
                else:
                    e_l = pair_unit_vectors(pos, key_index, sub)
                    dj = pair_projections(u_b, sub - lo, e_l)
                    dy = pair_projections(y_b, sub - lo, e_l)
                    l_pair = radial_huber(
                        (dj - dy).abs() / scale, cfg.huber_delta).mean()
        # L_cos: teacher-magnitude gate only (never the prediction)
        g = (y_b.norm(dim=-1) > float(cfg.tau_norm_angstrom)).to(u.dtype)
        wg = w * g
        denom = wg.sum()
        if float(denom) > 0.0:
            c = soft_cosine(u_b, y_b, cfg.cos_soft_norm_angstrom)
            l_cos = (wg * (1.0 - c)).sum() / denom.clamp(min=1e-8)
        else:
            l_cos = torch.zeros((), dtype=u.dtype, device=u.device)
        # L_mag (plan §3.3): log-magnitude huber on gated atoms;
        # `s` carries the EFFECTIVE magnitude (see vectorized twin).
        l_mag = torch.zeros((), dtype=u.dtype, device=u.device)
        if s is not None:
            s_b = s[lo:hi]
            y_norm = y_b.norm(dim=-1)
            gate_mag = (y_norm
                        > float(cfg.mag_supervision_tau_angstrom)).to(u.dtype)
            if float(gate_mag.sum()) > 0.0:
                log_r = torch.log((s_b + 1e-6) / (y_norm + 1e-6))
                l_mag = (radial_huber(log_r, cfg.huber_delta)
                         * gate_mag).sum() / gate_mag.sum().clamp(min=1.0)
        frame = (l_atom + float(cfg.lambda_pair) * l_pair
                 + float(lambda_cos) * l_cos
                 + float(lambda_mag) * l_mag
                 + float(lambda_sign) * l_sign
                 + float(lambda_dead) * l_dead)
        total = total + frame
        l_atom_acc = l_atom_acc + l_atom
        l_pair_acc = l_pair_acc + l_pair
        l_cos_acc = l_cos_acc + l_cos
        l_mag_acc = l_mag_acc + l_mag
        l_sign_acc = l_sign_acc + l_sign
        l_dead_acc = l_dead_acc + l_dead
        n_used += 1
    _require_loss(n_used > 0, "NO_ELIGIBLE_FRAMES",
                  "batch has no unmasked frames with key atoms")
    total = total / n_used
    _require_loss(bool(torch.isfinite(total)),
                  "NONFINITE_LOSS", "total loss is not finite")
    return {
        "loss": total,
        "l_atom": l_atom_acc / n_used,
        "l_pair": l_pair_acc / n_used,
        "l_pair_reg": l_pair_acc / n_used,
        "l_cos": l_cos_acc / n_used,
        "l_mag": l_mag_acc / n_used,
        "l_sign": l_sign_acc / n_used,
        "l_dead": l_dead_acc / n_used,
        "lambda_cos": float(lambda_cos),
        "lambda_mag": float(lambda_mag),
        "lambda_sign": float(lambda_sign),
        "lambda_dead": float(lambda_dead),
        "pair_loss": str(cfg.pair_loss),
        "n_pairs_active": int(n_pairs_active),
        "n_frames": n_used,
    }


# ---------------------------------------------------------------------------
# Bounded export (plan §4: raw kept; u_bounded = ONE uniform factor/frame)
# ---------------------------------------------------------------------------

def bounded_projection(
    u: torch.Tensor, d0: torch.Tensor | float, key_ptr: torch.Tensor, *,
    max_atom_cap: float = MAX_ATOM_CAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single common per-frame factor so max_atom ≤ 0.20 Å AND RMS ≤ d0.

    Returns ``(u_bounded [Σm,3], scale [B])`` — the applied factors are
    recorded so nothing is hidden by the truncation.
    """
    out = u.clone()
    n_graphs = int(key_ptr.shape[0]) - 1
    scale = torch.ones(n_graphs, dtype=torch.float32)
    d0_t = (d0 if torch.is_tensor(d0)
            else torch.full((n_graphs,), float(d0), dtype=torch.float32))
    for b in range(n_graphs):
        lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
        if hi <= lo:
            continue
        norms = u[lo:hi].norm(dim=-1)
        max_atom = float(norms.max())
        rms = float(torch.sqrt(torch.mean(norms * norms)))
        f = 1.0
        if max_atom > 1e-12:
            f = min(f, float(max_atom_cap) / max_atom)
        d0_b = float(d0_t[b])
        if rms > 1e-12 and d0_b > 0.0:
            f = min(f, d0_b / rms)
        out[lo:hi] = u[lo:hi] * f
        scale[b] = f
    return out, scale


# ---------------------------------------------------------------------------
# Eval metrics (per reaction cluster + cluster-median aggregation)
# ---------------------------------------------------------------------------

def _median(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _quantile(values: Sequence[float], q: float) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), q))


def compute_cluster_metrics(
    per_sample: Sequence[dict], *,
    soft: float = 1e-4, tau: float = 0.02, near_ts_cut: float = 0.20,
) -> tuple[dict, dict]:
    """Reaction-cluster metrics + cluster-median aggregation.

    Each ``per_sample`` entry: ``group_id``, ``u`` [m,3], ``y`` [m,3],
    ``u_classic`` [m,3] or None, ``near_ts`` float.  Per cluster: median/
    mean MAE & RMSE (Å), RMSE/zero and RMSE/Classic ratios, effective-atom
    (‖y‖>τ) soft-cosine median overall and for near-TS samples, and the
    V5.1a amplitude ratio α = ‖u‖/‖y‖ median over effective atoms (plan
    §5.1 M1).  The aggregate takes the median over clusters for every
    scalar (plus α Q25/Q75).
    """
    clusters: dict[str, list[dict]] = {}
    for s in per_sample:
        clusters.setdefault(str(s["group_id"]), []).append(s)
    per_cluster: dict[str, dict] = {}
    for gid, samples in sorted(clusters.items()):
        errs: list[np.ndarray] = []
        y_norms: list[np.ndarray] = []
        cos_eff: list[float] = []
        cos_near: list[float] = []
        alpha_eff: list[float] = []
        errs_classic: list[np.ndarray] = []
        has_classic = True
        for s in samples:
            u = np.asarray(s["u"], dtype=np.float64)
            y = np.asarray(s["y"], dtype=np.float64)
            e = np.linalg.norm(u - y, axis=1)
            errs.append(e)
            y_norms.append(np.linalg.norm(y, axis=1))
            eff = np.linalg.norm(y, axis=1) > tau
            if bool(eff.any()):
                c = soft_cosine(
                    torch.as_tensor(u[eff], dtype=torch.float64),
                    torch.as_tensor(y[eff], dtype=torch.float64),
                    soft).numpy()
                cos_eff.extend(c.tolist())
                if float(s.get("near_ts", 1.0)) <= near_ts_cut:
                    cos_near.extend(c.tolist())
                alpha_eff.extend(
                    (np.linalg.norm(u[eff], axis=1)
                     / np.maximum(np.linalg.norm(y[eff], axis=1),
                                  1e-12)).tolist())
            uc = s.get("u_classic")
            if uc is None:
                has_classic = False
            else:
                errs_classic.append(
                    np.linalg.norm(
                        np.asarray(uc, dtype=np.float64) - y, axis=1))
        e_all = np.concatenate(errs) if errs else np.zeros(1)
        y_all = np.concatenate(y_norms) if y_norms else np.zeros(1)
        mae = float(e_all.mean())
        rmse = float(np.sqrt(np.mean(e_all ** 2)))
        rmse_zero = float(np.sqrt(np.mean(y_all ** 2)))
        entry = {
            "n_samples": len(samples),
            "mae": mae,
            "rmse": rmse,
            "rmse_over_zero": rmse / max(rmse_zero, 1e-12),
            "cos_eff_median": _median(cos_eff),
            "cos_eff_near_ts_median": _median(cos_near),
            "alpha_median": _median(alpha_eff),
        }
        if has_classic and errs_classic:
            e_cls = np.concatenate(errs_classic)
            rmse_cls = float(np.sqrt(np.mean(e_cls ** 2)))
            entry["rmse_over_classic"] = rmse / max(rmse_cls, 1e-12)
        else:
            entry["rmse_over_classic"] = None
        per_cluster[gid] = entry
    agg_keys = (
        "mae", "rmse", "rmse_over_zero", "rmse_over_classic",
        "cos_eff_median", "cos_eff_near_ts_median", "alpha_median")
    aggregate = {
        k: _median([per_cluster[g][k] for g in per_cluster])
        for k in agg_keys}
    aggregate["alpha_q25"] = _quantile(
        [per_cluster[g]["alpha_median"] for g in per_cluster], 0.25)
    aggregate["alpha_q75"] = _quantile(
        [per_cluster[g]["alpha_median"] for g in per_cluster], 0.75)
    aggregate["n_clusters"] = len(per_cluster)
    return per_cluster, aggregate


# ---------------------------------------------------------------------------
# Sampler (plan §5.2: 8 groups × 4 frames, 2/branch, q-weighted, cycle pool)
# ---------------------------------------------------------------------------

def _entry_fields(entry) -> tuple[str, float]:
    """Tolerant (sample_id, q_weight) access for Lane B pool entries.

    Accepts objects with attributes or dict rows; INTEGRATION PENDING on
    the exact ``key_atom_dataset`` pool row type.
    """
    if isinstance(entry, Mapping):
        sid = entry.get("sample_id", entry.get("id"))
        q = entry.get("q_weight", entry.get("q", 1.0))
    else:
        sid = getattr(entry, "sample_id", getattr(entry, "id", None))
        q = getattr(entry, "q_weight", getattr(entry, "q", 1.0))
    _require(sid is not None, "POOL_ENTRY_INVALID",
             f"pool entry has no sample_id: {entry!r}")
    return str(sid), float(q)


class KeyAtomCycleSampler:
    """Seeded, paired reaction-balanced sampler (plan §5.2 / §4).

    Per batch: 8 distinct groups (uniform without replacement), each with
    4 frames — 2 per branch where the branch pool allows, otherwise as
    many as exist with the remainder from the other branch.  Frames are
    drawn from the group's current cycle pool; within a cycle every frame
    is drawn at most once (q-weighted order), then the cycle advances
    (fresh seeded shuffle of the same branch pool — Lane B rotation hooks
    in via ``pools`` refresh).  The RNG is seeded once per run so the
    sample sequence is identical across arms for the same seed.
    """

    def __init__(
        self,
        pools: Mapping[str, Mapping[int, Sequence]],
        *,
        seed: int,
        reactions_per_batch: int = 8,
        frames_per_reaction: int = 4,
    ) -> None:
        # pools: {group_id: {branch(int): [pool rows]}}
        self.pools: dict[str, dict[int, list]] = {}
        for g, branches in pools.items():
            self.pools[str(g)] = {
                int(b): list(rows) for b, rows in dict(branches).items()}
        if not self.pools:
            raise KeyAtomBCTrainerError(
                "EMPTY_POOLS", "sampler received no reaction groups")
        self.seed = int(seed)
        self.reactions_per_batch = int(reactions_per_batch)
        self.frames_per_reaction = int(frames_per_reaction)
        self.rng = np.random.default_rng(self.seed)
        self._order: dict[tuple[str, int], list[int]] = {}
        self._cycle: dict[tuple[str, int], int] = {}
        self.draws = 0
        self._reset_all_cycles()

    def _frames(self, group: str, branch: int) -> list[tuple[str, float]]:
        entries = self.pools.get(group, {}).get(branch, [])
        return [_entry_fields(e) for e in entries]

    def _reset_cycle(self, group: str, branch: int) -> None:
        frames = self._frames(group, branch)
        n = len(frames)
        # q-weighted without-replacement order: repeated Gumbel-top-k
        if n == 0:
            self._order[(group, branch)] = []
        else:
            q = np.asarray([max(f[1], 0.0) for f in frames], dtype=np.float64)
            if float(q.sum()) <= 0.0:
                q = np.ones(n, dtype=np.float64)
            keys = -np.log(-np.log(self.rng.random(n)) + 1e-12) - np.log(q)
            self._order[(group, branch)] = list(np.argsort(keys))
        self._cycle[(group, branch)] = \
            self._cycle.get((group, branch), 0) + 1

    def _reset_all_cycles(self) -> None:
        for g in sorted(self.pools):
            for b in self.pools[g]:
                self._reset_cycle(g, int(b))

    def _take(self, group: str, branch: int, count: int) -> list[str]:
        frames = self._frames(group, branch)
        out: list[str] = []
        while len(out) < count:
            order = self._order.get((group, branch), [])
            if not order:
                self._reset_cycle(group, branch)
                order = self._order.get((group, branch), [])
                if not order:
                    break  # branch pool empty
            idx = order.pop(0)
            out.append(frames[idx][0])
        return out

    def draw_batch(self) -> list[dict]:
        """One batch of ``reactions_per_batch × frames_per_reaction`` rows:
        ``{"group_id", "branch", "sample_id"}``."""
        groups = sorted(self.pools)
        chosen = self.rng.choice(
            len(groups), size=min(self.reactions_per_batch, len(groups)),
            replace=False)
        per_branch = self.frames_per_reaction // 2
        rows: list[dict] = []
        for gi in chosen:
            group = groups[int(gi)]
            branches = sorted(self.pools[group], key=int)
            taken: dict[int, list[str]] = {}
            for b in branches:
                taken[int(b)] = self._take(group, int(b), per_branch)
            # top up to frames_per_reaction from whichever branch remains
            deficit = self.frames_per_reaction - sum(
                len(v) for v in taken.values())
            for b in branches:
                if deficit <= 0:
                    break
                extra = self._take(group, int(b), deficit + per_branch)
                take_n = min(deficit, len(extra))
                taken[int(b)][:0] = extra[:take_n]
                deficit -= take_n
            for b in sorted(taken):
                for sid in taken[b]:
                    rows.append(
                        {"group_id": group, "branch": b, "sample_id": sid})
        self.draws += len(rows)
        return rows

    def state_dict(self) -> dict:
        return {"seed": self.seed, "draws": int(self.draws),
                "cycles": {f"{g}|{b}": c for (g, b), c in self._cycle.items()}}

    def load_state_dict(self, state: Mapping) -> None:
        if int(state["seed"]) != self.seed:
            raise KeyAtomBCTrainerError(
                "SAMPLER_SEED_MISMATCH", "cannot restore sampler state")
        self.draws = int(state["draws"])


# ---------------------------------------------------------------------------
# LR schedule: linear warmup then cosine to min_lr (plan §4)
# ---------------------------------------------------------------------------

def lr_at_update(
    update: int, *, base_lr: float, warmup: int, total: int, min_lr: float
) -> float:
    """1-based update → learning rate (linear warmup, then cosine)."""
    if warmup > 0 and update < warmup:
        return base_lr * update / warmup
    span = max(total - warmup, 1)
    t = min(1.0, (update - warmup) / span)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Lane B integration (key_atom_dataset runtime adapters)
# ---------------------------------------------------------------------------

_LANE_B_CACHE: dict[tuple[str, str], dict] = {}
_SPLIT_CACHE: dict[tuple, tuple] = {}


def _split_group_reactions(split_path: str | Path, fold: int,
                           config: Mapping | None = None):
    """Read pesn_v5_split_v1 (folds/group_fold + fit50 selection) ->
    (train_groups, dev_groups).  Memoized per (path, mtime, fold, sel)."""
    import json as _json

    path = Path(split_path)
    sel_key = ""
    if config is not None:
        paths = dict((config or {}).get("paths", {}))
        sel_key = str(paths.get(
            "fit50_selection",
            "outputs/pesn/direct_egnn_v1_20261015/splits/"
            "selection_direct_fit50.json"))
    cache_key = (str(path), path.stat().st_mtime_ns if path.exists() else -1,
                 int(fold), sel_key)
    cached = _SPLIT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    doc = _json.loads(path.read_text(encoding="utf-8"))
    folds = doc.get("folds")
    group_fold = doc.get("group_fold")
    if not isinstance(folds, dict) or not isinstance(group_fold, dict):
        raise KeyAtomBCTrainerError(
            "SPLIT_EMPTY", f"no folds/group_fold tables in {split_path}")
    paths = dict((config or {}).get("paths", {}))
    sel_path = Path(paths.get(
        "fit50_selection",
        "outputs/pesn/direct_egnn_v1_20261015/splits/selection_direct_fit50.json"))
    sel = _json.loads(sel_path.read_text(encoding="utf-8"))
    group_rids: dict[str, list[str]] = {}
    for gid, rid in zip(sel.get("group_ids", []), sel.get("reaction_ids", [])):
        group_rids.setdefault(str(gid), []).append(str(rid))
    train: dict[str, list[str]] = {}
    dev: dict[str, list[str]] = {}
    for gid, rids in group_rids.items():
        f = group_fold.get(gid)
        if f is None:
            raise KeyAtomBCTrainerError(
                "SPLIT_EMPTY", f"group {gid} absent from group_fold table")
        bucket = dev if int(f) == int(fold) else train
        bucket[gid] = rids
    if not train and not dev:
        raise KeyAtomBCTrainerError(
            "SPLIT_EMPTY", f"no fit50 rows parsed from {split_path}")
    _SPLIT_CACHE[cache_key] = (train, dev)
    return train, dev


def _lane_b_runtime(config: Mapping, d0: float, split_path: str | Path):
    """Cached shard list + dataset + vocabs for one (feature_root, d0)."""
    from pes2ts_core.pesn.offline.data import key_atom_dataset as lane_b

    paths = dict(config.get("paths", {}))
    feature_root = str(paths.get("feature_root", "outputs/pesn/v5/features"))
    d0_tag = f"{float(d0):.2f}"
    key = (feature_root, d0_tag)
    cached = _LANE_B_CACHE.get(key)
    if cached is None:
        shards = sorted(
            (Path(feature_root) / f"d0_{d0_tag}").glob("*.branch*.npz"))
        if not shards:
            raise KeyAtomBCTrainerError(
                "DATA_LANE_PENDING",
                f"no feature shards under {feature_root}/d0_{d0_tag}; "
                "run stage prepare-audit first")
        dataset = lane_b.V5FeatureDataset([str(p) for p in shards])
        cached = {"shards": shards, "dataset": dataset,
                  "shard_sizes": list(dataset._shard_index[1:]
                                      - dataset._shard_index[:-1]),
                  "sample_cache": {}}
        _LANE_B_CACHE[key] = cached
    return cached


def _sample_id(shard_idx: int, sample_idx: int) -> str:
    return f"s{int(shard_idx)}:i{int(sample_idx)}"


def _parse_sample_id(sid: str, runtime: dict) -> int:
    s, i = str(sid).split(":")
    shard_idx = int(s[1:])
    sample_idx = int(i[1:])
    offset = 0
    for k in range(shard_idx):
        offset += int(runtime["shard_sizes"][k])
    return offset + sample_idx


def load_lane_b_pools(config: Mapping, *, fold: int, d0: float,
                      split_path: str,
                      inner_val: Mapping | None = None):
    """Train cycle pools + dev dense rows (plan §5.2), Lane B API bridge.

    Returns ``{"train": {gid: {branch: [row]}}, "dev_dense": [row]}`` with
    rows ``{"sample_id", "q_weight"}``; ``sample_id`` encodes
    ``(shard_idx, sample_idx)`` over the sorted shard list of the d0 root.

    V5.1a (plan §3.4): pass ``inner_val={"n_val": 8, "seed": 20261019}``
    to additionally receive ``"inner_train"`` (train pools restricted to
    the inner-train clusters), ``"inner_val_dense"`` (dense rows for the
    inner-val clusters) and ``"inner_split"`` (ids/seed/hash provenance).
    The outer dev_dense rows are untouched — v51 training never selects
    on them.
    """
    from pes2ts_core.pesn.offline.data import key_atom_dataset as lane_b

    train_groups, dev_groups = _split_group_reactions(split_path, fold, config)
    paths = dict(config.get("paths", {}))
    feature_root = str(paths.get("feature_root", "outputs/pesn/v5/features"))
    d0_tag = f"{float(d0):.2f}"
    pools = lane_b.build_pools(feature_root, train_groups,
                               max_per_branch_cycle=int(
                                   config.get(
                                       "max_start_frames_per_branch_cycle",
                                       16)),
                               d0_tag=d0_tag)
    train: dict[str, dict[int, list[dict]]] = {}
    for gid, branches in pools.items():
        train[gid] = {}
        for branch, cycles in dict(branches).items():
            rows: list[dict] = []
            for cycle in cycles:
                for shard_idx, sample_idx, q in cycle:
                    rows.append({"sample_id": _sample_id(shard_idx, sample_idx),
                                 "q_weight": float(q)})
            train[gid][int(branch)] = rows
    runtime = _lane_b_runtime(config, d0, split_path)
    dev_rids = {rid for rids in dev_groups.values() for rid in rids}
    dev_rows: list[dict] = _dense_rows_for(runtime, dev_rids)
    if not train:
        raise KeyAtomBCTrainerError(
            "EMPTY_POOLS", "no train pools resolved from split")
    out = {"train": train, "dev_dense": dev_rows}
    if inner_val is not None:
        n_val = int(inner_val.get("n_val", 8))
        seed = int(inner_val.get("seed", V51_INNER_SPLIT_SEED))
        family_of, size_of = _cluster_family_size(split_path)
        train_ids, val_ids, split_hash = inner_val_split(
            sorted(train), family_of=family_of, size_of=size_of,
            n_val=n_val, seed=seed)
        val_set = set(val_ids)
        train_rids = {rid for gid in train_ids
                      for rid in train_groups.get(gid, [])}
        val_rids = {rid for gid in val_ids
                    for rid in train_groups.get(gid, [])}
        out["inner_train"] = {gid: train[gid] for gid in train_ids}
        out["inner_val_dense"] = _dense_rows_for(runtime, val_rids)
        out["inner_split"] = {
            "train_ids": list(train_ids),
            "val_ids": list(val_ids),
            "n_val": n_val,
            "seed": seed,
            "hash": split_hash,
            "train_rids": sorted(train_rids),
            "val_rids": sorted(val_rids),
            "n_val_clusters": len(val_set),
        }
    return out


def _dense_rows_for(runtime: Mapping, rids: set[str]) -> list[dict]:
    """Dense eligible rows (``sample_id``/``q_weight``) for reaction ids."""
    rows: list[dict] = []
    for shard_idx, path in enumerate(runtime["shards"]):
        rid = path.name.split(".branch")[0]
        if rid not in rids:
            continue
        with np.load(path, allow_pickle=False) as data:
            eligible = np.asarray(data["eligible"], dtype=bool)
            q = np.asarray(data["q_weight"], dtype=np.float64)
        for sample_idx in np.nonzero(eligible)[0]:
            rows.append({"sample_id": _sample_id(shard_idx, sample_idx),
                         "q_weight": float(q[sample_idx])})
    return rows


# ---------------------------------------------------------------------------
# V5.1a inner-val split (plan §3.4: 32/8, family+size stratified, hashed)
# ---------------------------------------------------------------------------

#: Frozen inner-split seed (plan §3.4 "seed 固定").
V51_INNER_SPLIT_SEED: Final[int] = 20261019

_FAMILY_SIZE_CACHE: dict[str, tuple[dict, dict]] = {}


def _cluster_family_size(split_path: str | Path) -> tuple[dict, dict]:
    """``({gid: family}, {gid: size})`` from the split manifest components.

    ``family`` is the G1 taxonomy family of the cluster component;
    ``size`` is the reaction count of the cluster (its "size").  Groups
    absent from the components table fall back to ``("unknown", 1)``.
    Memoized per manifest mtime.
    """
    import json as _json

    path = Path(split_path)
    key = f"{path}|{path.stat().st_mtime_ns if path.exists() else -1}"
    cached = _FAMILY_SIZE_CACHE.get(key)
    if cached is not None:
        return cached
    doc = _json.loads(path.read_text(encoding="utf-8"))
    family_of: dict[str, str] = {}
    size_of: dict[str, int] = {}
    for comp in doc.get("components") or ():
        family = str(comp.get("family", "unknown"))
        gids = [str(g) for g in comp.get("group_ids") or ()]
        for gid in gids:
            family_of[gid] = family
            size_of[gid] = len(gids)
    _FAMILY_SIZE_CACHE[key] = (family_of, size_of)
    return family_of, size_of


def inner_val_split(
    cluster_ids: Sequence[str],
    *,
    family_of: Mapping[str, str] | None = None,
    size_of: Mapping[str, int] | None = None,
    n_val: int = 8,
    seed: int = V51_INNER_SPLIT_SEED,
) -> tuple[list[str], list[str], str]:
    """Deterministic 32/8-style inner split of the outer-fold TRAIN
    clusters (plan §3.4: family/size stratified, fixed seed, hash
    recorded).

    Clusters are ordered by ``(family, size, cluster_id)`` with a seeded
    within-stratum shuffle; the inner-val set is a SYSTEMATIC sample of
    that ordering (random start, fixed stride ``n/n_val``), which spreads
    the val slots across every family/size stratum.  Returns
    ``(train_ids, val_ids, split_hash)`` — the hash is sha256 over the
    sorted cluster ids, the sorted val ids and the seed.
    """
    ids = sorted({str(g) for g in cluster_ids})
    n = len(ids)
    _require(0 < int(n_val) < n, "INNER_SPLIT_INVALID",
             f"need 0 < n_val < n_clusters, got n_val={n_val}, n={n}")
    fam_of = {g: str((family_of or {}).get(g, "unknown")) for g in ids}
    size_lookup = {g: int((size_of or {}).get(g, 1)) for g in ids}
    strata: dict[tuple[str, int], list[str]] = {}
    for g in ids:
        strata.setdefault((fam_of[g], size_lookup[g]), []).append(g)
    rng = np.random.default_rng(int(seed))
    ordered: list[str] = []
    for key in sorted(strata):
        members = strata[key]  # already sorted (ids inserted in order)
        perm = rng.permutation(len(members))
        ordered.extend(members[i] for i in perm)
    step = n / float(n_val)
    start = float(rng.uniform(0.0, step))
    val_positions = sorted({int(start + j * step) for j in range(n_val)})
    # step >= 1 guarantees n_distinct == n_val (floor of an arithmetic
    # sequence with stride >= 1); the guard covers degenerate callers.
    pos = 0
    while len(val_positions) < int(n_val):
        candidate = (val_positions[-1] + 1) % n if val_positions else 0
        if candidate not in val_positions:
            val_positions.append(candidate)
        else:
            pos += 1
            candidate = pos % n
            if candidate not in val_positions:
                val_positions.append(candidate)
    val_ids = sorted(ordered[i] for i in val_positions)
    train_ids = [g for g in ids if g not in set(val_ids)]
    payload = json.dumps(
        {"clusters": ids, "inner_val": val_ids, "seed": int(seed)},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    split_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return train_ids, val_ids, split_hash


class _SampleView:
    """Index-mapped view over Lane B's V5FeatureDataset for sample rows.

    Materialized sample dicts are memoized in the SHARED runtime cache:
    pool rotation revisits the same rows across cycles and every batch
    rebuilds a view, so warm batches skip the npz slicing entirely
    (read-only contract downstream — collate copies into fresh tensors)."""

    def __init__(self, dataset, indices, shared_cache: dict):
        self._dataset = dataset
        self._indices = list(int(i) for i in indices)
        self._cache = shared_cache

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, j: int):
        gi = self._indices[j]
        row = self._cache.get(gi)
        if row is None:
            row = self._dataset[gi]
            if len(self._cache) < 200_000:
                self._cache[gi] = row
        return row


def load_lane_b_view(*, config: Mapping, samples: Sequence[dict], fold: int,
                     d0: float, anchor: str, split_path: str, device):
    """Resolve sampler rows to (view, collate) over the shared dataset."""
    from pes2ts_core.pesn.offline.data import key_atom_dataset as lane_b

    runtime = _lane_b_runtime(config, d0, split_path)
    indices = [_parse_sample_id(row["sample_id"], runtime) for row in samples]
    view = _SampleView(runtime["dataset"], indices,
                       runtime["sample_cache"])
    vkey = ("vocabs", str(runtime["shards"][0].parent), f"fold{int(fold)}")
    vocabs = _LANE_B_CACHE.get(vkey)
    if vocabs is None:
        train_groups, _dev = _split_group_reactions(split_path, fold, config)
        train_rids = [rid for rids in train_groups.values() for rid in rids]
        paths = dict(config.get("paths", {}))
        feature_root = str(paths.get("feature_root",
                                     "outputs/pesn/v5/features"))
        vocabs = lane_b.build_vocabs(feature_root, train_rids)
        _LANE_B_CACHE[vkey] = vocabs
    method_vocab, solvent_vocab = vocabs
    collate = runtime["dataset"].get_collate_fn(
        anchor_mode=(str(anchor).lower() == "on"),
        method_vocab=method_vocab, solvent_vocab=solvent_vocab,
        device=device)
    return view, collate


# ---------------------------------------------------------------------------
# Provenance helpers (ADR-0025 gv block)
# ---------------------------------------------------------------------------

def _sha256_or_none(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path | None = None) -> str:
    """HEAD commit via direct ``.git`` reads (no subprocess; truth-guard safe)."""
    try:
        root = Path(repo_root) if repo_root is not None else Path.cwd()
        git = root / ".git"
        if git.is_file():  # worktree pointer
            target = Path(git.read_text(encoding="utf-8").strip().split(
                "gitdir:", 1)[-1].strip())
            if not target.is_absolute():
                target = root / target
            git = target
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = git / head[5:].strip()
            if ref.is_file():
                return ref.read_text(encoding="utf-8").strip()
            return "unknown"
        return head or "unknown"
    except OSError:
        return "unknown"


def build_gv_block(*, config_path: Path, repo_root: Path,
                   campaign_id: str | None = None) -> dict:
    """ADR-0025 ``gv`` provenance block for a sprint run record.

    ``campaign_id`` comes from the campaign config (``sprint.campaign_id``
    — e.g. ``V5_1A_SPRINT`` for v51 runs); when the id is not (yet)
    registered in ``g_v_map`` the block falls back to the registered
    ``V5_SPRINT`` entry's gates so validation still passes — the
    configured id travels in the run record's top-level ``campaign_id``.
    """
    gv_map = load_gv_map()
    requested = str(campaign_id or CAMPAIGN_ID)
    registered = (requested if requested in gv_map.campaigns
                  else CAMPAIGN_ID)
    entry = gv_map.campaigns.get(registered)
    data_manifest = repo_root / "data/manifests/v5/split_manifest.json"
    map_path = Path(gv_map.path)
    return {
        "map_version": gv_map.map_version,
        "campaign_id": registered,
        "campaign_id_configured": requested,
        "primary_g": entry.primary_g if entry is not None else "G3.P.0",
        "supporting_g": (list(entry.supporting_g) if entry is not None
                         else ["G3.P.1", "G3.P.2", "G3.P.3", "G0", "G1"]),
        "gates": (list(entry.gates) if entry is not None
                  else ["V5-G1", "V5-G2", "V5-G3", "V5-G4"]),
        "code_commit": _git_commit(),
        "config_sha256": _sha256_or_none(config_path),
        "data_manifest_sha256": _sha256_or_none(data_manifest),
        "map_sha256": _sha256_or_none(map_path),
    }


def write_run_record(
    run_dir: Path, *, payload: Mapping, gv: Mapping, path: str = "run_record.json"
) -> None:
    """Write (or refresh) the run record; the gv block is validated via the
    ADR-0025 registry before anything hits disk."""
    doc = {**dict(payload), "gv": dict(gv)}
    validate_gv_run_record(doc, load_gv_map())
    (run_dir / path).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class KeyAtomArmSpec:
    """Resolved arm identity: prefix A = anchor on, C = anchor off.

    V5.1a additions (plan §3.2/§3.3): ``magnitude_head`` (direction–
    magnitude decoupling, anchor-on only) and ``pair_loss``
    (``legacy_j`` | ``exact_trio``).
    """

    arm_id: str
    anchor: str          # "on" | "off"
    d0: float
    magnitude_head: bool = False
    pair_loss: str = "legacy_j"

    def to_payload(self) -> dict:
        return {"arm_id": self.arm_id, "anchor": self.anchor,
                "d0": self.d0, "magnitude_head": self.magnitude_head,
                "pair_loss": self.pair_loss}


def _as_bool(value, *, field: str, arm_id: str) -> bool:
    """YAML-1.1 tolerant boolean (bare on/off/yes/no parse as bools)."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0"):
        return False
    raise KeyAtomBCTrainerError(
        "ARM_FIELD_INVALID",
        f"arm {arm_id} {field} must be a boolean, got {value!r}")


def resolve_arm(config: Mapping, arm_id: str) -> KeyAtomArmSpec:
    """Parse the arm id: A→anchor on, C→anchor off; d0 from the config
    arm_table (plan §5.2).  V5.1a arm_table entries may additionally set
    ``magnitude_head`` (bool) and ``pair_loss`` (``legacy_j``|
    ``exact_trio``); both default to the legacy V5 behavior.  The A/C
    prefix rule is enforced only for A/C-prefixed arms; any other prefix
    (B0..B3 ablations) is accepted verbatim.  A magnitude head REQUIRES
    anchor=on (plan §3.2 — the head-on path is anchor-on only).
    """
    table = (config.get("sprint", {}) or {}).get("arm_table", {}) or {}
    _require(arm_id in table, "ARM_UNKNOWN",
             f"arm {arm_id!r} not in config arm_table {sorted(table)}")
    entry = table[arm_id] or {}
    d0 = float(entry.get("d0", config.get("d0_angstrom", 0.08)))
    raw_anchor = entry.get("anchor", "off")
    # YAML 1.1 parses bare on/off as booleans — normalize both spellings
    if isinstance(raw_anchor, bool):
        anchor = "on" if raw_anchor else "off"
    else:
        anchor = str(raw_anchor).strip().lower()
        if anchor in ("true", "yes"):
            anchor = "on"
        elif anchor in ("false", "no"):
            anchor = "off"
    _require(anchor in ("on", "off"), "ARM_ANCHOR_INVALID",
             f"arm {arm_id} anchor must be on/off, got {anchor!r}")
    prefix = arm_id[0].upper()
    if prefix in ("A", "C"):
        _require((prefix == "A") == (anchor == "on"), "ARM_PREFIX_CONFLICT",
                 f"arm {arm_id}: prefix {prefix!r} disagrees with arm_table "
                 f"anchor={anchor!r}")
    magnitude_head = _as_bool(
        entry.get("magnitude_head", False),
        field="magnitude_head", arm_id=arm_id)
    pair_loss = str(entry.get(
        "pair_loss", config.get("pair_loss", "legacy_j"))).strip().lower()
    _require(pair_loss in PAIR_LOSS_MODES, "ARM_PAIR_LOSS_INVALID",
             f"arm {arm_id} pair_loss must be one of {PAIR_LOSS_MODES}, "
             f"got {pair_loss!r}")
    _require(not magnitude_head or anchor == "on", "ARM_HEAD_ANCHOR_REQUIRED",
             f"arm {arm_id}: magnitude_head=true requires anchor=on "
             f"(V5.1 plan §3.2)")
    return KeyAtomArmSpec(arm_id=arm_id, anchor=anchor, d0=d0,
                          magnitude_head=magnitude_head, pair_loss=pair_loss)


# ---------------------------------------------------------------------------
# V5.1a best-checkpoint selection + patience (plan §3.4 — FROZEN rule)
# ---------------------------------------------------------------------------

#: Frozen checkpoint-qualification factor (plan §3.4: rmse ≤ R*×1.02).
V51_QUALIFY_FACTOR: Final[float] = 1.02
#: Frozen patience floor: no stop before this update (plan §3.4).
V51_PATIENCE_FLOOR_UPDATE: Final[int] = 500


@dataclass(frozen=True, slots=True)
class V51EvalPoint:
    """One inner-val evaluation row for the v51 selection rule."""

    update: int
    rmse_over_zero: float | None
    sre_median: float | None
    near_ts_cos: float | None
    alpha_median: float | None = None
    dead_zone_fraction: float | None = None
    sign_accuracy: float | None = None


def _finite_or(value: float | None, fallback: float) -> float:
    if value is None or not math.isfinite(float(value)):
        return fallback
    return float(value)


def select_best_v51(
    points: Sequence[V51EvalPoint],
    *,
    qualify_factor: float = V51_QUALIFY_FACTOR,
) -> V51EvalPoint | None:
    """Pure best-checkpoint selection (plan §3.4, frozen rule):

    maintain the running best rmse R* over ALL checks so far; a checkpoint
    QUALIFIES iff ``rmse ≤ R* × 1.02``; among qualified checkpoints keep
    the one with the lowest SRE median, tie-break highest near-TS cos,
    then earliest update (determinism only — not part of the plan rule).
    Non-finite/None metrics sort worst (SRE → +inf, cos → −inf).
    """
    finite = [p for p in points
              if p.rmse_over_zero is not None
              and math.isfinite(float(p.rmse_over_zero))]
    if not finite:
        return None
    r_star = min(float(p.rmse_over_zero) for p in finite)
    qualified = [p for p in finite
                 if float(p.rmse_over_zero) <= r_star * qualify_factor]

    def _key(p: V51EvalPoint) -> tuple[float, float, int]:
        # lowest SRE first; then highest near-TS cos; then earliest
        # update (determinism only — not part of the plan rule).
        return (_finite_or(p.sre_median, math.inf),
                -_finite_or(p.near_ts_cos, -math.inf),
                int(p.update))

    return min(qualified, key=_key)


class V51EarlyStopper:
    """Patience state for v51 inner-val selection (plan §3.4).

    A check IMPROVES the running best R* by ≥ ``min_delta`` → the bad
    counter resets; otherwise it increments.  Stop fires at
    ``patience`` consecutive bad checks, but never before
    ``floor_update`` (plan §3.4: no stop before update 500).
    """

    def __init__(self, *, patience: int = 8, min_delta: float = 0.005,
                 floor_update: int = V51_PATIENCE_FLOOR_UPDATE) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.floor_update = int(floor_update)
        self.r_star: float | None = None
        self.bad_checks = 0

    def update(self, rmse_over_zero: float | None, *,
               at_update: int) -> bool:
        """Feed one inner-val check; return True → stop early."""
        if rmse_over_zero is None or not math.isfinite(float(rmse_over_zero)):
            return False
        value = float(rmse_over_zero)
        if self.r_star is None or (self.r_star - value) >= self.min_delta:
            self.r_star = value
            self.bad_checks = 0
            return False
        if value < self.r_star:
            self.r_star = value  # new best, but improvement < min_delta
        self.bad_checks += 1
        return (self.bad_checks >= self.patience
                and int(at_update) >= self.floor_update)


class V51BestTracker:
    """Running v51 best-checkpoint selection over inner-val eval points.

    Wraps the frozen plan §3.4 rule (:func:`select_best_v51`): after each
    check the selection is RECOMPUTED over the full history (the
    qualifying band follows the running R*, so an earlier point can drop
    out or re-enter).  ``best_update`` is the update currently selected;
    callers persist model states keyed by update and rewrite
    ``checkpoint_best.pt`` whenever ``best_update`` changes.
    """

    def __init__(self, *, qualify_factor: float = V51_QUALIFY_FACTOR) -> None:
        self.qualify_factor = float(qualify_factor)
        self.points: list[V51EvalPoint] = []
        self.best: V51EvalPoint | None = None
        self.r_star: float | None = None

    def add(self, point: V51EvalPoint) -> V51EvalPoint | None:
        """Record one inner-val check; return the new best (or None)."""
        self.points.append(point)
        rmse = point.rmse_over_zero
        if rmse is not None and math.isfinite(float(rmse)):
            value = float(rmse)
            self.r_star = (value if self.r_star is None
                           else min(self.r_star, value))
        self.best = select_best_v51(
            self.points, qualify_factor=self.qualify_factor)
        return self.best


class KeyAtomBCTrainer:
    """Config-driven V5 sprint trainer (ADR-0026 §2.8, plan §4/§5).

    One trainer drives one (arm, fold, seed, mode) run with a fixed
    update budget.  Screen mode may censored-stop on the inner_val
    reaction-cluster median RMSE/zero (patience/min_delta from config);
    formal mode never early-stops.  V5.1a adds mode ``v51`` (plan §3.4):
    32/8 family/size-stratified inner split of the outer TRAIN clusters,
    eval on inner-val only, best-checkpoint selection (rmse ≤ R*×1.02 →
    lowest SRE → highest near-TS cos), patience 8 / min_delta 0.005 with
    a floor at update 500, and a POST-HOC outer dev_dense eval of
    checkpoint_best AND checkpoint_final that selection never sees.
    """

    def __init__(
        self,
        *,
        config: Mapping,
        config_path: Path,
        arm: KeyAtomArmSpec,
        fold: int,
        seed: int,
        updates: int,
        mode: str,
        run_dir: Path,
        device_override: str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        _require(config.get("schema") in ACCEPTED_CONFIG_SCHEMAS,
                 "CONFIG_SCHEMA",
                 f"expected schema in {ACCEPTED_CONFIG_SCHEMAS}, got "
                 f"{config.get('schema')!r}")
        _require(mode in ("screen", "formal", "v51"), "MODE_UNKNOWN",
                 f"mode must be screen|formal|v51, got {mode!r}")
        _require(int(updates) > 0, "BUDGET_INVALID", "updates must be positive")
        _require(not arm.magnitude_head or arm.anchor == "on",
                 "ARM_HEAD_ANCHOR_REQUIRED",
                 f"arm {arm.arm_id}: magnitude_head=true requires anchor=on "
                 f"(V5.1 plan §3.2)")
        self.config = config
        self.config_path = Path(config_path)
        self.arm = arm
        self.fold = int(fold)
        self.seed = int(seed)
        self.updates = int(updates)
        self.mode = mode
        self.run_dir = Path(run_dir)
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        loss_cfg = KeyAtomLossConfig.from_campaign_config(config)
        # arm-controlled pair-loss switch (plan §3.3); arm wins over config
        self.loss_config = (loss_cfg if loss_cfg.pair_loss == arm.pair_loss
                            else replace(loss_cfg, pair_loss=arm.pair_loss))
        self.eval_every = int(config.get("eval_every_updates", 250))
        self.batch_size = int(config.get("batch_size", 32))
        self.reactions_per_batch = int(
            config.get("reactions_per_batch", 8))
        self.frames_per_reaction = int(config.get("frames_per_reaction", 4))
        self.grad_clip = float(config.get("grad_clip_norm", 1.0))
        self.base_lr = float(config.get("lr", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.warmup = int(config.get("warmup_updates", 250))
        self.min_lr = float(config.get("min_lr", 1e-4))
        self.early_stop_patience = int(
            config.get("early_stop_patience_checks", 8))
        self.early_stop_min_delta = float(
            config.get("early_stop_min_delta", 0.005))
        self.formal_early_stopping = bool(
            config.get("formal_outer_early_stopping", False))
        self.split_path = str(
            (config.get("paths", {}) or {}).get(
                "split_manifest", "data/manifests/v5/split_manifest.json"))
        #: Campaign id from the config (``sprint.campaign_id``; the V5.1
        #: driver yaml sets ``V5_1A_SPRINT``) — echoed in run records.
        self.campaign_id = str(
            (config.get("sprint", {}) or {}).get("campaign_id")
            or CAMPAIGN_ID)
        # --- V5.1a (plan §3.2/§3.4); driver yaml nests these under v51: ---
        v51_cfg = dict(config.get("v51", {}) or {})
        inner_blk = dict(v51_cfg.get("inner_val", {}) or {})
        sel_blk = dict(inner_blk.get("selection", {}) or {})
        mag_blk = dict(v51_cfg.get("magnitude_head", {}) or {})
        self.magnitude_mode = str(
            mag_blk.get("mode", config.get("magnitude_mode", "per_atom")))
        self.inner_val_n = int(inner_blk.get(
            "n_val_clusters", v51_cfg.get("inner_val_clusters", 8)))
        self.inner_val_seed = int(inner_blk.get(
            "seed", v51_cfg.get("inner_split_seed", V51_INNER_SPLIT_SEED)))
        self.v51_qualify_factor = float(sel_blk.get(
            "rmse_tolerance", v51_cfg.get("qualify_rmse_factor",
                                          V51_QUALIFY_FACTOR)))
        self.v51_patience = int(sel_blk.get(
            "patience_checks", self.early_stop_patience))
        self.v51_min_delta = float(sel_blk.get(
            "min_delta", self.early_stop_min_delta))
        self.v51_floor_update = int(sel_blk.get(
            "min_update_floor", V51_PATIENCE_FLOOR_UPDATE))
        self._inner_split: dict | None = None
        self._v51_tracker: V51BestTracker | None = None
        self._v51_stopper: V51EarlyStopper | None = None
        self._v51_states: dict[int, dict] = {}
        self._best_update: int | None = None
        self._best_score: dict | None = None

        # device: config require_cuda honored; --device cpu is smoke-only
        from pes2ts_core.pesn.offline.training.device import resolve_device
        dev_cfg = config.get("device", {}) or {}
        if device_override is not None and device_override.startswith("cpu"):
            self.torch, self.device = resolve_device(
                device=device_override, require_cuda=False)
        else:
            self.torch, self.device = resolve_device(
                device=device_override or str(dev_cfg.get("device", "cuda:0")),
                require_cuda=bool(dev_cfg.get("require_cuda", True)))

        # paired initialization: seed BEFORE model construction
        torch.manual_seed(self.seed)
        if arm.magnitude_head:
            # V5.1a plan §3.2: head-on arms (anchor-on only, asserted
            # above) construct the magnitude head; legacy arms keep the
            # exact V5 construction (identical RNG stream).
            self.model = StarEGNN(
                magnitude_head=True,
                magnitude_mode=self.magnitude_mode).to(
                    torch.float32).to(self.device)
        else:
            self.model = StarEGNN().to(torch.float32).to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.base_lr, weight_decay=self.weight_decay)

        self.sampler = None  # created in fit() after pools load (re-seeded)
        self.updates_run = 0
        self.eval_history: list[dict] = []
        self.bounded_scales: list[dict] = []
        self.status = "running"
        self.status_detail = ""
        self._best_metric: float | None = None
        self._bad_evals = 0

    # -- data (Lane B integration points) -----------------------------------

    def _build_pools(self):
        try:
            inner = ({"n_val": self.inner_val_n, "seed": self.inner_val_seed}
                     if self.mode == "v51" else None)
            return load_lane_b_pools(
                self.config, fold=self.fold, d0=self.arm.d0,
                split_path=self.split_path, inner_val=inner)
        except ImportError as exc:
            raise KeyAtomBCTrainerError(
                "DATA_LANE_PENDING",
                "Lane B module pes2ts_core.pesn.offline.data."
                f"key_atom_dataset not available yet: {exc}") from exc

    def _load_samples(self, sample_rows: Sequence[dict], *, dense: bool):
        try:
            view, collate = load_lane_b_view(
                config=self.config, samples=list(sample_rows), fold=self.fold,
                d0=self.arm.d0, anchor=self.arm.anchor,
                split_path=self.split_path, device=self.device)
        except ImportError as exc:
            raise KeyAtomBCTrainerError(
                "DATA_LANE_PENDING",
                "Lane B module pes2ts_core.pesn.offline.data."
                f"key_atom_dataset not available yet: {exc}") from exc
        batches = []
        for i in range(0, len(view), self.batch_size):
            chunk = [view[j] for j in range(i, min(i + self.batch_size,
                                                    len(view)))]
            if not chunk:
                continue
            batch = collate(chunk)
            batches.append(self._to_device(batch))
        return batches

    def _to_device(self, batch: KeyAtomBatch) -> KeyAtomBatch:
        for name in ("z_atomic", "h_node_inv", "pos", "pos0", "anchor_vec",
                     "edge_index", "edge_attr", "r_ij", "r0_ij", "task19",
                     "batch", "ptr", "key_index", "key_ptr", "pair_index",
                     "pair_edit", "degK", "y_teacher", "q_weight", "near_ts",
                     "terminal", "capped", "d0", "branch"):
            val = getattr(batch, name, None)
            if torch.is_tensor(val):
                if val.dtype in (torch.float64,):
                    val = val.to(torch.float32)
                setattr(batch, name, val.to(self.device))
        return batch

    # -- run record ----------------------------------------------------------

    def _write_run_record(self, *, n_parameters: int) -> None:
        is_v51 = self.mode == "v51"
        payload = {
            "schema_version": RUN_RECORD_SCHEMA,
            "loss_schema_version": (LOSS_SCHEMA_VERSION if is_v51
                                    else SCHEMA_VERSION),
            "run_id": self.run_dir.name,
            "status": "running",
            "campaign_id": self.campaign_id,
            "arm": self.arm.to_payload(),
            "fold": self.fold,
            "seed": self.seed,
            "updates_budget": self.updates,
            "mode": self.mode,
            "formal_outer_early_stopping": (
                False if self.mode == "formal" else self.formal_early_stopping),
            "n_parameters": n_parameters,
            "config_path": str(self.config_path),
            "split_manifest": self.split_path,
        }
        if is_v51:
            # plan §3.4 / §6: checkpoint selection NEVER saw the outer dev
            payload["selection_exposed"] = False
        gv = build_gv_block(
            config_path=self.config_path, repo_root=self.repo_root,
            campaign_id=self.campaign_id)
        write_run_record(self.run_dir, payload=payload, gv=gv)
        self._run_record_payload = payload
        self._run_record_gv = gv

    def _finalize_run_record(self) -> None:
        payload = {**self._run_record_payload, "status": self.status,
                   "updates_run": self.updates_run}
        if self.mode == "v51" and self._best_update is not None:
            # plan §6 run-record additions (best-checkpoint inner-val)
            payload["best_update"] = self._best_update
            payload["inner_val_score"] = self._v51_score_payload()
            payload["alpha_median"] = (self._best_score or {}).get(
                "alpha_median")
            payload["dead_zone_fraction"] = (self._best_score or {}).get(
                "dead_zone_fraction")
            payload["sign_accuracy"] = (self._best_score or {}).get(
                "sign_accuracy")
        write_run_record(self.run_dir, payload=payload, gv=self._run_record_gv)

    # -- checkpointing -------------------------------------------------------

    def _save_checkpoint(self, name: str, *,
                         model_state: Mapping | None = None,
                         update: int | None = None,
                         v51: Mapping | None = None) -> None:
        doc = {
            "schema_version": CHECKPOINT_SCHEMA,
            "update": int(update) if update is not None else self.updates_run,
            "model": (dict(model_state) if model_state is not None
                      else self.model.state_dict()),
            "optimizer": self.optimizer.state_dict(),
            "arm": self.arm.to_payload(),
            "fold": self.fold,
            "seed": self.seed,
            # V5.1a (plan §3.2): head provenance travels with the weights
            "magnitude_head": bool(self.arm.magnitude_head),
            "magnitude_mode": (self.magnitude_mode
                               if self.arm.magnitude_head else None),
            "pair_loss": str(self.arm.pair_loss),
        }
        if v51 is not None:
            doc["v51"] = dict(v51)
        torch.save(doc, self.run_dir / "checkpoints" / name)

    # -- eval ----------------------------------------------------------------

    def evaluate(self, dev_batches: Sequence[KeyAtomBatch], *,
                 include_g53: bool = False,
                 split: str | None = None) -> dict:
        """Dense fold eval: raw + bounded per-cluster metrics.

        ``include_g53`` adds the exact-Δd derived-chemistry block (SRE
        cluster median + dead-zone fraction via the shared
        ``g53_sample_enumeration`` — plan §3.3/§3.4 selection metric ==
        gate metric).  ``split`` labels the eval split in eval.jsonl
        (v51 driver contract: "inner_val" | "dev_dense_best" |
        "dev_dense_final").
        """
        from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
            evaluate_g53, g53_sample_enumeration)

        self.model.train(False)
        per_sample: list[dict] = []
        g53_samples: list[dict] = []
        with torch.no_grad():
            for batch in dev_batches:
                u, _z = self.model(batch)
                y = batch.y_teacher
                if y is None:
                    continue
                u_b, scale = bounded_projection(
                    u.float(), batch.d0 if batch.d0 is not None else A_REF,
                    batch.key_ptr)
                u_cls, cls_info = classic_pair_driver(batch)
                key_ptr = batch.key_ptr.cpu().numpy()
                near = (batch.near_ts.cpu().numpy()
                        if batch.near_ts is not None
                        else np.ones(len(key_ptr) - 1))
                groups = (batch.reaction_ids if batch.reaction_ids
                          else tuple("unknown" for _ in range(len(key_ptr) - 1)))
                u_np = u.float().cpu().numpy()
                y_np = y.cpu().numpy()
                ub_np = u_b.cpu().numpy()
                uc_np = u_cls.cpu().numpy()
                if include_g53:
                    branches = (batch.branch.cpu().numpy()
                                if batch.branch is not None else None)
                    g53_samples.extend(g53_sample_enumeration(
                        pos=batch.pos.cpu().numpy(),
                        ptr=batch.ptr.cpu().numpy(),
                        key_index=batch.key_index.cpu().numpy(),
                        key_ptr=key_ptr,
                        pair_index=batch.pair_index.cpu().numpy(),
                        pair_edit=batch.pair_edit.cpu().numpy(),
                        u=u_np, y=y_np, group_ids=groups,
                        branches=branches))
                for b in range(len(key_ptr) - 1):
                    lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
                    if hi <= lo:
                        continue
                    degenerate = bool(cls_info["degenerate_frames"][b])
                    per_sample.append({
                        "group_id": str(groups[b]),
                        "u": u_np[lo:hi], "y": y_np[lo:hi],
                        "u_classic": None if degenerate else uc_np[lo:hi],
                        "near_ts": float(near[b]),
                        "u_bounded": ub_np[lo:hi],
                        "bounded_scale": float(scale[b]),
                    })
        self.model.train()
        cfg = self.loss_config
        per_cluster_raw, agg_raw = compute_cluster_metrics(
            per_sample, soft=cfg.cos_soft_norm_angstrom,
            tau=cfg.tau_norm_angstrom)
        per_cluster_b, agg_b = compute_cluster_metrics(
            [{**s, "u": s["u_bounded"]} for s in per_sample],
            soft=cfg.cos_soft_norm_angstrom, tau=cfg.tau_norm_angstrom)
        g53 = (evaluate_g53(g53_samples) if g53_samples else None)
        return {
            "update": self.updates_run,
            "wall_s": time.time() - self._t0,
            "split": split,
            "raw": {"per_cluster": per_cluster_raw, "aggregate": agg_raw},
            "bounded": {"per_cluster": per_cluster_b, "aggregate": agg_b},
            "g53": ({"aggregate": g53["aggregate"],
                     "per_cluster_sre": g53["per_cluster_sre"]}
                    if g53 is not None else None),
        }

    def _record_eval(self, record: dict) -> None:
        self.eval_history.append(record)
        raw_agg = dict(record["raw"]["aggregate"])
        g53_agg = (record.get("g53") or {}).get("aggregate")
        if g53_agg is not None:
            # merge the derived-chemistry scalars into the aggregate the
            # v51 driver reads from eval.jsonl / metrics.json (defensive
            # spellings live driver-side; we emit the canonical ones).
            for key in ("sre_cluster_median", "sign_accuracy_cluster_median",
                        "dead_zone_fraction", "n_sign_pairs",
                        "n_sign_pairs_correct", "n_active_pairs",
                        "n_dead_pairs", "n_sre_frames"):
                if key in g53_agg:
                    raw_agg[key] = g53_agg[key]
        line = {
            "update": record["update"],
            "wall_s": record["wall_s"],
            "per_cluster": record["raw"]["per_cluster"],
            "aggregate": raw_agg,
            "per_cluster_bounded": record["bounded"]["per_cluster"],
            "aggregate_bounded": record["bounded"]["aggregate"],
        }
        if record.get("split") is not None:
            line["split"] = record["split"]
        if record.get("checkpoint") is not None:
            line["checkpoint"] = record["checkpoint"]
        if g53_agg is not None:
            line["g53_aggregate"] = g53_agg
        with (self.run_dir / "eval.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _early_stop_check(self, record: dict) -> bool:
        """Screen-mode only: inner_val reaction-cluster median RMSE/zero."""
        if self.mode != "screen":
            return False
        metric = record["raw"]["aggregate"].get("rmse_over_zero")
        if metric is None or not math.isfinite(metric):
            return False
        if self.updates_run < 500:  # no rank-based stop before update 500
            return False
        if self._best_metric is None:
            self._best_metric = metric
            return False
        if self._best_metric - metric > self.early_stop_min_delta:
            self._best_metric = metric
            self._bad_evals = 0
            return False
        self._bad_evals += 1
        return self._bad_evals >= self.early_stop_patience

    # -- main loop -----------------------------------------------------------

    def fit(self) -> str:
        """Run the fixed budget; return the final status string."""
        if self.mode == "v51":
            return self._fit_v51()
        self._t0 = time.time()
        self._write_run_record(n_parameters=self.model.n_parameters)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        try:
            pools = self._build_pools()
            train_pools = pools["train"] if "train" in pools else pools
            dev_samples = (pools.get("dev_dense")
                           if "train" in pools else None)
            # paired sample order: seed BEFORE sampler creation
            torch.manual_seed(self.seed)
            self.sampler = KeyAtomCycleSampler(
                train_pools, seed=self.seed,
                reactions_per_batch=self.reactions_per_batch,
                frames_per_reaction=self.frames_per_reaction)
            dev_batches = self._load_samples(
                dev_samples if dev_samples is not None else [], dense=True)
        except KeyAtomBCTrainerError as exc:
            self._abort(exc.code, exc.detail,
                        STATUS_ABORTED_DATA
                        if exc.code.startswith(("DATA", "EMPTY", "POOL"))
                        else STATUS_ABORTED_NAN)
            return self.status

        self.model.train()
        try:
            for update in range(1, self.updates + 1):
                self.updates_run = update
                lr = lr_at_update(
                    update, base_lr=self.base_lr, warmup=self.warmup,
                    total=self.updates, min_lr=self.min_lr)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                rows = self.sampler.draw_batch()
                batches = self._load_samples(rows, dense=False)
                lam_cos = cos_lambda(update, self.loss_config)
                for batch in batches:
                    u, _z = self.model(batch)
                    if batch.y_teacher is None:
                        continue
                    out = key_atom_bc_loss_vectorized(
                        u.float(), batch.y_teacher.float(),
                        batch.degK.float(), batch.key_ptr, batch.pos,
                        batch.key_index, batch.pair_index,
                        config=self.loss_config, lambda_cos=lam_cos)
                    loss = out["loss"]
                    if not bool(torch.isfinite(loss)):
                        raise KeyAtomBCTrainerError(
                            "ABORT_NAN", f"non-finite loss at update {update}")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                if update % self.eval_every == 0 or update == self.updates:
                    record = self.evaluate(dev_batches)
                    self._record_eval(record)
                    self._save_checkpoint("checkpoint_last.pt")
                    if self._early_stop_check(record):
                        self.status = STATUS_CENSORED_SCREEN
                        self.status_detail = (
                            "screen early stop: inner_val reaction-cluster "
                            "median RMSE/zero stalled (patience "
                            f"{self.early_stop_patience}, min_delta "
                            f"{self.early_stop_min_delta})")
                        break
        except KeyAtomBCTrainerError as exc:
            self._abort(exc.code, exc.detail, STATUS_ABORTED_NAN)
            return self.status
        except KeyAtomLossError as exc:
            self._abort(exc.code, exc.detail, STATUS_ABORTED_NAN)
            return self.status

        if self.status == "running":
            self.status = STATUS_COMPLETE
        self._save_checkpoint("checkpoint_final.pt")
        self._write_metrics()
        (self.run_dir / "status.json").write_text(json.dumps({
            "status": self.status, "detail": self.status_detail,
            "updates_run": self.updates_run, "mode": self.mode,
        }, indent=2) + "\n", encoding="utf-8")
        self._finalize_run_record()
        return self.status

    # -- V5.1a loop (plan §3.3/§3.4) ----------------------------------------

    def _refresh_run_record(self) -> None:
        """Rewrite the run record with the current payload (best-effort)."""
        write_run_record(self.run_dir, payload=self._run_record_payload,
                         gv=self._run_record_gv)

    def _v51_point(self, record: dict) -> "V51EvalPoint":
        agg = record["raw"]["aggregate"]
        g53_agg = (record.get("g53") or {}).get("aggregate") or {}
        return V51EvalPoint(
            update=int(record["update"]),
            rmse_over_zero=agg.get("rmse_over_zero"),
            sre_median=g53_agg.get("sre_cluster_median"),
            near_ts_cos=agg.get("cos_eff_near_ts_median"),
            alpha_median=agg.get("alpha_median"),
            dead_zone_fraction=g53_agg.get("dead_zone_fraction"),
            sign_accuracy=g53_agg.get("sign_accuracy_cluster_median"))

    def _v51_score_payload(self) -> dict:
        """Driver-contract inner_val_score spelling (metrics.json/record)."""
        return {
            "rmse_over_zero": (self._best_score or {}).get("rmse_over_zero"),
            "sre_median": (self._best_score or {}).get("sre_median"),
            "near_ts_cos": (self._best_score or {}).get("near_ts_cos"),
        }

    def _save_v51_best(self) -> None:
        state = self._v51_states.get(int(self._best_update or -1))
        self._save_checkpoint(
            "checkpoint_best.pt", model_state=state,
            update=int(self._best_update),
            v51={"best_update": self._best_update,
                 "inner_val_score": self._v51_score_payload(),
                 "alpha_median": (self._best_score or {}).get("alpha_median"),
                 "dead_zone_fraction": (self._best_score or {}).get(
                     "dead_zone_fraction"),
                 "sign_accuracy": (self._best_score or {}).get("sign_accuracy"),
                 "inner_val_split_hash": (self._inner_split or {}).get("hash"),
                 "selection_exposed": False,
                 "qualify_rmse_factor": self.v51_qualify_factor})

    def _fit_v51(self) -> str:
        """V5.1a loop (plan §3.3/§3.4): inner-train pools only, inner-val
        selection every ``eval_every_updates``, best checkpoint, patience
        stop, POST-HOC outer dev eval that selection never sees."""
        self._t0 = time.time()
        self._write_run_record(n_parameters=self.model.n_parameters)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        try:
            pools = self._build_pools()
            train_pools = pools["inner_train"]
            self._inner_split = dict(pools["inner_split"])
            inner_val_samples = list(pools["inner_val_dense"])
            dev_samples = list(pools.get("dev_dense") or [])
            # paired sample order: seed BEFORE sampler creation
            torch.manual_seed(self.seed)
            self.sampler = KeyAtomCycleSampler(
                train_pools, seed=self.seed,
                reactions_per_batch=self.reactions_per_batch,
                frames_per_reaction=self.frames_per_reaction)
            inner_batches = self._load_samples(inner_val_samples, dense=True)
        except KeyAtomBCTrainerError as exc:
            self._abort(exc.code, exc.detail,
                        STATUS_ABORTED_DATA
                        if exc.code.startswith(
                            ("DATA", "EMPTY", "POOL", "INNER"))
                        else STATUS_ABORTED_NAN)
            return self.status
        # freeze split provenance into the run record before any update
        self._run_record_payload["inner_val_split_hash"] = str(
            self._inner_split["hash"])
        self._run_record_payload["inner_val_split"] = {
            "n_val": self._inner_split["n_val"],
            "seed": self._inner_split["seed"],
            "val_ids": list(self._inner_split["val_ids"]),
        }
        self._refresh_run_record()

        cfg = self.loss_config
        self._v51_tracker = V51BestTracker(
            qualify_factor=self.v51_qualify_factor)
        self._v51_stopper = V51EarlyStopper(
            patience=self.v51_patience, min_delta=self.v51_min_delta,
            floor_update=self.v51_floor_update)

        self.model.train()
        try:
            for update in range(1, self.updates + 1):
                self.updates_run = update
                lr = lr_at_update(
                    update, base_lr=self.base_lr, warmup=self.warmup,
                    total=self.updates, min_lr=self.min_lr)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                rows = self.sampler.draw_batch()
                batches = self._load_samples(rows, dense=False)
                lam_cos = cos_lambda(update, cfg)
                aux = aux_ramp_fraction(update, cfg)
                lam_mag = cfg.lambda_mag * aux
                lam_sign = cfg.lambda_sign * aux
                lam_dead = cfg.lambda_dead * aux
                for batch in batches:
                    u, _z = self.model(batch)
                    if batch.y_teacher is None:
                        continue
                    # last_magnitudes = EFFECTIVE s_eff (post deg_scale);
                    # L_mag supervises it toward ‖y‖ (see model docstring)
                    s = (self.model.last_magnitudes
                         if self.arm.magnitude_head else None)
                    out = key_atom_bc_loss_vectorized(
                        u.float(), batch.y_teacher.float(),
                        batch.degK.float(), batch.key_ptr, batch.pos,
                        batch.key_index, batch.pair_index,
                        config=cfg, lambda_cos=lam_cos,
                        pair_edit=batch.pair_edit, s=s,
                        lambda_mag=lam_mag, lambda_sign=lam_sign,
                        lambda_dead=lam_dead)
                    loss = out["loss"]
                    if not bool(torch.isfinite(loss)):
                        raise KeyAtomBCTrainerError(
                            "ABORT_NAN", f"non-finite loss at update {update}")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                if update % self.eval_every == 0 or update == self.updates:
                    record = self.evaluate(inner_batches, include_g53=True,
                                           split="inner_val")
                    self._record_eval(record)
                    self._save_checkpoint("checkpoint_last.pt")
                    point = self._v51_point(record)
                    self._v51_states[update] = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()}
                    previous_best = self._best_update
                    best = self._v51_tracker.add(point)
                    if best is not None and best.update != previous_best:
                        self._best_update = int(best.update)
                        self._best_score = {
                            "rmse_over_zero": best.rmse_over_zero,
                            "sre_median": best.sre_median,
                            "near_ts_cos": best.near_ts_cos,
                            "alpha_median": best.alpha_median,
                            "dead_zone_fraction": best.dead_zone_fraction,
                            "sign_accuracy": best.sign_accuracy,
                        }
                        self._save_v51_best()
                    if self._v51_stopper.update(
                            point.rmse_over_zero, at_update=update):
                        self.status = STATUS_CENSORED_PATIENCE
                        self.status_detail = (
                            "v51 inner-val early stop: RMSE/zero stalled "
                            f"(patience {self._v51_stopper.patience}, "
                            f"min_delta {self._v51_stopper.min_delta}, "
                            f"floor update {self._v51_stopper.floor_update})")
                        break
        except KeyAtomBCTrainerError as exc:
            self._abort(exc.code, exc.detail, STATUS_ABORTED_NAN)
            return self.status
        except KeyAtomLossError as exc:
            self._abort(exc.code, exc.detail, STATUS_ABORTED_NAN)
            return self.status

        if self.status == "running":
            # successful v51 runs stay STATUS_COMPLETE (the landed V5.1
            # campaign driver keys run-success off status == "complete")
            self.status = STATUS_COMPLETE
            if self._best_update is not None:
                self.status_detail = (
                    f"v51 budget complete; best checkpoint @u"
                    f"{self._best_update} (inner-val "
                    f"rmse/zero={self._v51_score_payload()['rmse_over_zero']})")
        self._save_checkpoint("checkpoint_final.pt")
        # POST-HOC outer dev_dense eval of best AND final weights; these
        # rows are appended to eval.jsonl AFTER selection has ended.
        if dev_samples:
            self._outer_dev_posthoc(dev_samples)
        self._write_metrics()
        status_doc = {
            "status": self.status, "detail": self.status_detail,
            "updates_run": self.updates_run, "mode": self.mode,
        }
        if self._best_update is not None:
            status_doc.update({
                "best_update": self._best_update,
                "alpha_median": (self._best_score or {}).get("alpha_median"),
                "dead_zone_fraction": (self._best_score or {}).get(
                    "dead_zone_fraction"),
                "sign_accuracy": (self._best_score or {}).get("sign_accuracy"),
                "inner_val_score": self._v51_score_payload(),
                "inner_val_split_hash": (self._inner_split or {}).get("hash"),
                "selection_exposed": False,
            })
        (self.run_dir / "status.json").write_text(
            json.dumps(status_doc, indent=2) + "\n", encoding="utf-8")
        self._finalize_run_record()
        return self.status

    def _outer_dev_posthoc(self, dev_samples: Sequence[dict]) -> None:
        """Evaluate checkpoint_best AND checkpoint_final on the OUTER
        dev_dense (plan §3.4: 外折 OOF 只用 best; selection never sees
        these rows — they are appended after training ended)."""
        try:
            dev_batches = self._load_samples(list(dev_samples), dense=True)
        except KeyAtomBCTrainerError:
            return
        if not dev_batches:
            return
        if self._best_update is not None:
            best_state = self._v51_states.get(int(self._best_update))
            if best_state is not None:
                current = {k: v.detach().cpu().clone()
                           for k, v in self.model.state_dict().items()}
                self.model.load_state_dict(best_state)
                rec = self.evaluate(dev_batches, include_g53=True,
                                    split="dev_dense_best")
                rec["checkpoint"] = "checkpoint_best.pt"
                self._record_eval(rec)
                self.model.load_state_dict(current)
        rec = self.evaluate(dev_batches, include_g53=True,
                            split="dev_dense_final")
        rec["checkpoint"] = "checkpoint_final.pt"
        self._record_eval(rec)

    def _abort(self, code: str, detail: str, status: str) -> None:
        self.status = status
        self.status_detail = f"{code}: {detail}"
        (self.run_dir / "status.json").write_text(json.dumps({
            "status": status, "detail": self.status_detail,
            "updates_run": self.updates_run, "mode": self.mode,
        }, indent=2) + "\n", encoding="utf-8")
        try:
            self._finalize_run_record()
        except Exception:  # record refresh is best-effort on abort
            pass

    def _write_metrics(self) -> None:
        raw_rows = []
        bounded_rows = []
        for r in self.eval_history:
            agg = dict(r["raw"]["aggregate"])
            g53_agg = (r.get("g53") or {}).get("aggregate")
            if g53_agg is not None:
                for key in ("sre_cluster_median",
                            "sign_accuracy_cluster_median",
                            "dead_zone_fraction", "n_sign_pairs",
                            "n_sign_pairs_correct", "n_active_pairs",
                            "n_dead_pairs", "n_sre_frames"):
                    if key in g53_agg:
                        agg[key] = g53_agg[key]
            row = {"update": r["update"], "aggregate": agg,
                   "per_cluster": r["raw"]["per_cluster"]}
            if r.get("split") is not None:
                row["split"] = r["split"]
            raw_rows.append(row)
            bounded_rows.append(
                {"update": r["update"],
                 "aggregate": r["bounded"]["aggregate"],
                 "per_cluster": r["bounded"]["per_cluster"]})
        doc = {
            "schema_version": SCHEMA_VERSION,
            "arm": self.arm.to_payload(),
            "fold": self.fold, "seed": self.seed, "mode": self.mode,
            "updates_run": self.updates_run,
            "raw": raw_rows,
            "bounded": bounded_rows,
        }
        if self.mode == "v51":
            # v51 driver contract: top-level best-checkpoint metrics
            doc["best_update"] = self._best_update
            doc["inner_val_score"] = self._v51_score_payload()
            doc["alpha_median"] = (self._best_score or {}).get("alpha_median")
            doc["dead_zone_fraction"] = (self._best_score or {}).get(
                "dead_zone_fraction")
            doc["sign_accuracy"] = (self._best_score or {}).get(
                "sign_accuracy")
            doc["inner_val_split_hash"] = (self._inner_split or {}).get("hash")
            doc["selection_exposed"] = False
            doc["loss_schema_version"] = LOSS_SCHEMA_VERSION
        (self.run_dir / "metrics.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pes2ts_core.pesn.offline.training.key_atom_bc",
        description="V5 sprint key-atom BC single-run trainer (ADR-0026)")
    p.add_argument("--config", default="config/campaigns/v5_sprint.yaml")
    p.add_argument("--arm", required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--mode", choices=("screen", "formal", "v51"),
                   required=True)
    p.add_argument("--out", default=None,
                   help="run dir (default outputs/pesn/v5/sprint/runs/"
                        "<arm>_f<fold>_s<seed>_<mode>)")
    p.add_argument("--device", default=None,
                   help="device override (cpu allowed for smoke only)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config)
    config = load_config_mapping(config_path)
    arm = resolve_arm(config, args.arm)
    default_updates = (1500 if args.mode == "screen" else 3000)
    updates = int(args.updates) if args.updates is not None else default_updates
    run_id = f"{args.arm}_f{args.fold}_s{args.seed}_{args.mode}"
    run_dir = (Path(args.out) if args.out
               else Path("outputs/pesn/v5/sprint/runs") / run_id)
    trainer = KeyAtomBCTrainer(
        config=config, config_path=config_path, arm=arm, fold=args.fold,
        seed=args.seed, updates=updates, mode=args.mode, run_dir=run_dir,
        device_override=args.device)
    status = trainer.fit()
    print(json.dumps({"run_id": run_id, "status": status,
                      "run_dir": str(run_dir)}, ensure_ascii=False))
    return 0 if status in (STATUS_COMPLETE, STATUS_CENSORED_SCREEN,
                           STATUS_CENSORED_PATIENCE) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
