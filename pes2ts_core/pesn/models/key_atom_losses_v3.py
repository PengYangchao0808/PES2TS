"""R3 v3 loss FUNCTIONS (math only — no training loop; W-LOSS owns the loop).

Schema ``pesn_key_atom_loss_v3``.  Normative sources: campaign yaml
``r3_training.loss`` (FROZEN), R3 plan v3 §5, 30h plan §4.1.

Frozen semantics:

* ``L = L_atom + 0.5·L_pair + λ_cos(t)·L_cos + λ_s·L_strong + λ_d·L_dead``
  with the B0-F base identical to
  :func:`pes2ts_core.pesn.offline.training.key_atom_bc.key_atom_bc_loss_vectorized`
  (``pair_mode="legacy_j"``): per frame
  ``L_atom = Σ_i w_i·Huber_0.25(‖û−u*‖/0.08 Å)`` with
  ``w_i ∝ 1/max(1, degK_i)`` normalized inside the frame; ``L_pair`` the
  first-order pair-axis projection Huber over the key pairs; ``L_cos`` the
  teacher-gated (‖u*‖ > 0.02 Å) soft-cosine term; batch reduction = mean
  over eligible frames (optionally re-weighted by ``frame_weight``).
* λ_cos ramp **R0** (30h plan §4.1: the OFFICIAL main schedule): exactly 0
  for the first ``cos_start_update=250`` updates, linear over
  ``cos_ramp_updates=250`` to ``λ_cos_max=0.02`` (:func:`cos_lambda_r0`).
* **Strong bucket** (``|Δd*| ≥ 0.02 Å`` on intent-changed pairs):
  per-frame bucket mean of
  ``Huber_0.25((Δd̂ − Δd*) / max(|Δd*|, 0.01 Å))`` then weighted frame mean
  over frames with a non-empty bucket; **dead bucket**
  (``|Δd*| ≤ 0.005 Å``): per-frame bucket mean of
  ``[max(0, |Δd̂| − 0.005)/0.005]²`` then the same weighted frame mean.
  Empty bucket ⇒ the term is EXACTLY 0 (never NaN).
* ``Δd`` values are the EXACT finite-step pair-distance changes from the
  sample's common coordinates
  (:func:`exact_delta_d_torch`: ``‖r + u_j − u_i‖ − ‖r‖``), so prediction
  and teacher share one frame — no gauge mixing.
* **Aux arms** ``S2/S10/D2/D10`` are SINGLE-bucket arms (never mixed); the
  digits are the target encoder-gradient shares 2%/10% that WP calibration
  fills in (λ ≤ 0.5).  :func:`resolve_aux_arm_v3` maps an arm id to its
  bucket + target share.
* **Weights apply exactly once**: every loss term consumes the same
  ``frame_weight`` (curve-level total weight split over that curve's
  intent/node samples upstream); nothing re-multiplies them.
* **λ=0 bit-identity**: with ``λ_strong=λ_dead=0`` and uniform
  ``frame_weight`` the total and every base component are BIT-identical to
  the B0-F baseline loss.  How this is guaranteed: (1) the frame math is a
  verbatim copy of the baseline's legacy_j path — same helper formulas
  (``radial_huber``/``pair_unit_vectors``/``soft_cosine``/
  ``exact_delta_d_torch``, copied from the frozen ``key_atom_bc`` because
  runtime ``models/**`` may not import ``offline``), same operation order,
  same segmented ``index_add`` reductions in the same slot order; (2) the
  batch reduction multiplies by a 0/1 ``keep`` mask times the weight (all
  ones ⇒ ``w == keep`` bitwise) and divides by ``w.sum()`` which equals the
  baseline's integer frame count exactly; (3) the bucket branches are
  SKIPPED entirely when their λ is 0 (no stray additions).
  ``tests/pesn/test_r3_losses.py`` asserts ``torch.equal`` against the
  baseline function on randomized fixtures.
* Determinism: all reductions are fixed-order sums over fixed index tensors
  (CPU bitwise-reproducible; CUDA ``index_add`` atomics are not asserted
  here — the math test battery runs CPU-only).

Downstream (W-LOSS): ``offline/training/key_atom_bc_v3.py`` imports these
functions read-only and owns the training loop/schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

import torch

# NOTE (product boundary): ``pes2ts_core/pesn/models`` is runtime-reachable
# and may NOT import ``offline`` (tests/pesn/test_product_boundaries.py), so
# the four baseline helpers below are copied VERBATIM from the frozen
# ``pes2ts_core/pesn/offline/training/key_atom_bc.py``.  The copies are kept
# bitwise-identical by ``tests/pesn/test_r3_losses.py``, which asserts
# ``torch.equal`` against the baseline loss on randomized fixtures.

#: Frozen bucket/aux constants (campaign ``r3_training.loss``).
D0_NORMALIZATION_ANGSTROM: Final[float] = 0.08
HUBER_DELTA: Final[float] = 0.25
LAMBDA_PAIR: Final[float] = 0.5
LAMBDA_COS_MAX: Final[float] = 0.02
COS_START_UPDATE_R0: Final[int] = 250
COS_RAMP_UPDATES_R0: Final[int] = 250
COS_ACTIVE_THRESHOLD_ANGSTROM: Final[float] = 0.02
COS_SOFT_NORM: Final[float] = 1e-4
PAIR_MODE: Final[str] = "legacy_j"
STRONG_DELTA_D_STAR_MIN_ANGSTROM: Final[float] = 0.02
DEAD_DELTA_D_STAR_MAX_ANGSTROM: Final[float] = 0.005
STRONG_NORMALIZER_FLOOR_ANGSTROM: Final[float] = 0.01
DEAD_TAU_ANGSTROM: Final[float] = 0.005
STRONG_HUBER_DELTA: Final[float] = 0.25
LAMBDA_MAX_AUX: Final[float] = 0.5

#: Aux arm registry (R3 plan §6 L-arm; 30h plan §4.1).  Values are
#: ``(bucket, target_encoder_gradient_share)`` — λ itself is calibrated on
#: the R3 B0-F checkpoint (WP calibration, ≤ LAMBDA_MAX_AUX) and passed to
#: the loss explicitly.  Arms are never mixed.
AUX_ARMS_V3: Final[dict[str, tuple[str, float]]] = {
    "S2": ("strong", 0.02),
    "S10": ("strong", 0.10),
    "D2": ("dead", 0.02),
    "D10": ("dead", 0.10),
}


class KeyAtomV3LossError(RuntimeError):
    """Typed loss failure (malformed tensors / non-finite math)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise KeyAtomV3LossError(code, detail)


@dataclass(frozen=True, slots=True)
class KeyAtomLossV3Config:
    """Frozen R3 loss hyper-parameters (campaign ``r3_training.loss``)."""

    loss_scale_angstrom: float = D0_NORMALIZATION_ANGSTROM
    huber_delta: float = HUBER_DELTA
    lambda_pair: float = LAMBDA_PAIR
    lambda_cos_max: float = LAMBDA_COS_MAX
    cos_start_update: int = COS_START_UPDATE_R0
    cos_ramp_updates: int = COS_RAMP_UPDATES_R0
    tau_norm_angstrom: float = COS_ACTIVE_THRESHOLD_ANGSTROM
    cos_soft_norm_angstrom: float = COS_SOFT_NORM
    pair_mode: str = PAIR_MODE
    strong_threshold_angstrom: float = STRONG_DELTA_D_STAR_MIN_ANGSTROM
    dead_threshold_angstrom: float = DEAD_DELTA_D_STAR_MAX_ANGSTROM
    strong_normalizer_floor_angstrom: float = STRONG_NORMALIZER_FLOOR_ANGSTROM
    dead_tau_angstrom: float = DEAD_TAU_ANGSTROM
    strong_huber_delta: float = STRONG_HUBER_DELTA

    def __post_init__(self) -> None:
        if self.pair_mode != PAIR_MODE:
            raise KeyAtomV3LossError(
                "PAIR_MODE", "R3 v3 freezes pair_mode='legacy_j' (30h plan "
                f"§4.1); got {self.pair_mode!r}")


def cos_lambda_r0(update: int, config: KeyAtomLossV3Config | None = None) -> float:
    """λ_cos at a 1-based update count under the official R0 schedule.

    Exactly 0 for ``update <= cos_start_update`` (250), then linear to
    ``lambda_cos_max`` over the next ``cos_ramp_updates`` (250) updates,
    constant afterwards.  This is the same ramp formula as the baseline
    ``cos_lambda`` (30h plan §4.1: R0 is the official main schedule).
    """
    cfg = config or KeyAtomLossV3Config()
    start = int(cfg.cos_start_update)
    ramp = max(int(cfg.cos_ramp_updates), 1)
    if update <= start:
        return 0.0
    frac = min(1.0, (update - start) / ramp)
    return float(cfg.lambda_cos_max) * frac


def resolve_aux_arm_v3(
    arm: str | None,
) -> tuple[float, float, str]:
    """``(lambda_strong, lambda_dead, bucket)`` scaffold for an aux arm id.

    The returned lambdas are 0.0 placeholders — the numeric λ must come from
    the WP gradient calibration (target share in the middle tuple slot of
    :data:`AUX_ARMS_V3`, λ ≤ 0.5).  ``arm=None`` is the B0-F baseline (both
    lambdas 0).  Mixing arms is rejected.
    """
    if arm is None:
        return 0.0, 0.0, "none"
    if arm not in AUX_ARMS_V3:
        raise KeyAtomV3LossError(
            "AUX_ARM", f"unknown aux arm {arm!r}; known {sorted(AUX_ARMS_V3)}")
    bucket, _share = AUX_ARMS_V3[arm]
    return (0.0, 0.0, bucket)


def aux_arm_lambdas_v3(arm: str, lambda_arm: float) -> tuple[float, float]:
    """Place ``lambda_arm`` on the arm's bucket only (arms never mixed)."""
    if arm not in AUX_ARMS_V3:
        raise KeyAtomV3LossError(
            "AUX_ARM", f"unknown aux arm {arm!r}; known {sorted(AUX_ARMS_V3)}")
    if not (0.0 <= float(lambda_arm) <= LAMBDA_MAX_AUX):
        raise KeyAtomV3LossError(
            "AUX_LAMBDA", f"lambda must lie in [0, {LAMBDA_MAX_AUX}], got "
            f"{lambda_arm}")
    bucket, _share = AUX_ARMS_V3[arm]
    if bucket == "strong":
        return float(lambda_arm), 0.0
    return 0.0, float(lambda_arm)


# ---------------------------------------------------------------------------
# Loss building blocks (verbatim copies of the frozen baseline helpers)
# ---------------------------------------------------------------------------
def radial_huber(e: torch.Tensor, delta: float = 0.25) -> torch.Tensor:
    """h_δ(e) = e²/2 (e≤δ) else δ(e−δ/2); rotation-invariant radial Huber."""
    quad = 0.5 * e * e
    linear = delta * (e - 0.5 * delta)
    return torch.where(e <= delta, quad, linear)


def soft_cosine(
    u: torch.Tensor, y: torch.Tensor, soft: float = 1e-4
) -> torch.Tensor:
    """Soft-norm cosine per atom (1e-4 Å soft norm keeps zero-norm finite)."""
    num = (u * y).sum(dim=-1)
    den = torch.sqrt(u.pow(2).sum(dim=-1) + soft * soft) * torch.sqrt(
        y.pow(2).sum(dim=-1) + soft * soft)
    return num / den.clamp(min=1e-12)


def pair_unit_vectors(
    pos: torch.Tensor,
    key_index: torch.Tensor,
    pair_index: torch.Tensor,
) -> torch.Tensor:
    """e_l = (x_j − x_i)/d_ij from the CURRENT geometry (key-slot indices)."""
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


def exact_delta_d_torch(
    pos_key: torch.Tensor,
    pair_index: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Differentiable EXACT finite-step Δd per pair:
    ``‖r_l + u_j − u_i‖ − ‖r_l‖`` (key-slot pair indices)."""
    if pair_index is None or pair_index.numel() == 0:
        return torch.zeros(0, dtype=u.dtype, device=u.device)
    pi = pair_index[0].long()
    pj = pair_index[1].long()
    r = pos_key[pj] - pos_key[pi]
    return (r + u[pj] - u[pi]).norm(dim=-1) - r.norm(dim=-1)


# ---------------------------------------------------------------------------
# Bucket terms (R3 semantics — campaign r3_training.loss.strong/dead_bucket)
# ---------------------------------------------------------------------------
def _bucket_frame_mean(
    per_pair_term: torch.Tensor,
    pair_frame: torch.Tensor,
    bucket_mask: torch.Tensor,
    frame_weight: torch.Tensor,
    n_graphs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Weighted frame-mean of per-frame bucket means (plan §5.2).

    ``L = Σ_b w_b·1[|B_b|>0]·L_b / Σ_b w_b·1[|B_b|>0]`` with
    ``L_b = mean_{p∈B_b} term_p``; an all-empty bucket yields exactly 0.
    """
    term = per_pair_term.to(dtype=dtype)
    sel = bucket_mask & (pair_frame >= 0)
    n_pairs = int(sel.sum())
    if n_pairs == 0:
        return torch.zeros((), dtype=dtype, device=device), 0
    frame_sum = torch.zeros(n_graphs, dtype=dtype, device=device)
    frame_cnt = torch.zeros(n_graphs, dtype=dtype, device=device)
    frame_sum.index_add_(0, pair_frame[sel], term[sel])
    frame_cnt.index_add_(
        0, pair_frame[sel], torch.ones_like(term[sel]))
    frame_mean = torch.where(
        frame_cnt > 0, frame_sum / frame_cnt.clamp(min=1.0),
        torch.zeros_like(frame_sum))
    nonempty = frame_cnt > 0
    w = frame_weight.to(dtype=dtype)
    denom = (w * nonempty.to(dtype)).sum()
    if float(denom) == 0.0:
        return torch.zeros((), dtype=dtype, device=device), n_pairs
    num = (w * frame_mean * nonempty.to(dtype)).sum()
    return num / denom, n_pairs


def strong_bucket_loss_v3(
    dd_pred: torch.Tensor,
    dd_true: torch.Tensor,
    pair_frame: torch.Tensor,
    frame_weight: torch.Tensor,
    n_graphs: int,
    *,
    eligible: torch.Tensor | None = None,
    config: KeyAtomLossV3Config | None = None,
) -> tuple[torch.Tensor, int]:
    """L_strong: ``Huber_0.25((Δd̂−Δd*)/max(|Δd*|,0.01))`` on ``|Δd*|≥0.02``."""
    cfg = config or KeyAtomLossV3Config()
    dtype = dd_pred.dtype
    mask = dd_true.abs() >= float(cfg.strong_threshold_angstrom)
    if eligible is not None:
        mask = mask & eligible
    scale = torch.clamp(
        dd_true.abs(), min=float(cfg.strong_normalizer_floor_angstrom))
    err = (dd_pred - dd_true) / scale
    term = radial_huber(err.abs(), cfg.strong_huber_delta)
    return _bucket_frame_mean(
        term, pair_frame, mask, frame_weight, n_graphs, dtype, dd_pred.device)


def dead_bucket_loss_v3(
    dd_pred: torch.Tensor,
    dd_true: torch.Tensor,
    pair_frame: torch.Tensor,
    frame_weight: torch.Tensor,
    n_graphs: int,
    *,
    eligible: torch.Tensor | None = None,
    config: KeyAtomLossV3Config | None = None,
) -> tuple[torch.Tensor, int]:
    """L_dead: ``[max(0,|Δd̂|−0.005)/0.005]²`` on ``|Δd*| ≤ 0.005``."""
    cfg = config or KeyAtomLossV3Config()
    dtype = dd_pred.dtype
    mask = dd_true.abs() <= float(cfg.dead_threshold_angstrom)
    if eligible is not None:
        mask = mask & eligible
    slack = (dd_pred.abs() - float(cfg.dead_tau_angstrom)).clamp(min=0.0)
    term = (slack / float(cfg.dead_tau_angstrom)).pow(2)
    return _bucket_frame_mean(
        term, pair_frame, mask, frame_weight, n_graphs, dtype, dd_pred.device)


# ---------------------------------------------------------------------------
# Total loss
# ---------------------------------------------------------------------------
def key_atom_loss_v3(
    u: torch.Tensor,
    y: torch.Tensor,
    degK: torch.Tensor,
    key_ptr: torch.Tensor,
    pos: torch.Tensor,
    key_index: torch.Tensor,
    pair_index: torch.Tensor | None = None,
    *,
    config: KeyAtomLossV3Config | None = None,
    lambda_cos: float = 0.0,
    frame_mask: torch.Tensor | None = None,
    frame_weight: torch.Tensor | None = None,
    pair_edit: torch.Tensor | None = None,
    lambda_strong: float = 0.0,
    lambda_dead: float = 0.0,
    aux_arm: str | None = None,
) -> dict:
    """R3 v3 total loss + components (functions only; no loop).

    ``pair_index`` is [2,P] in GLOBAL node rows (the dataset/collate
    convention); it is converted to key-slot space internally, exactly like
    the baseline.  ``key_ptr`` [G+1] segments the key rows per frame.
    ``frame_weight`` [G] (label-quality; applied ONCE; default all ones =
    the B0-F plain mean) and ``frame_mask`` [G] gate frame eligibility.

    Aux arms: exactly one of ``aux_arm="S2"|"S10"|"D2"|"D10"`` may be active
    (with its calibrated λ passed via ``lambda_strong``/``lambda_dead``);
    mixing strong and dead in one call is rejected (30h plan §4.1: 先不混合).
    """
    cfg = config or KeyAtomLossV3Config()
    _require(bool(torch.isfinite(u).all()),
             "NONFINITE_PREDICTION", "u contains non-finite values")
    _require(bool(torch.isfinite(y).all()),
             "NONFINITE_TEACHER", "y contains non-finite values")
    if aux_arm is not None:
        if aux_arm not in AUX_ARMS_V3:
            raise KeyAtomV3LossError(
                "AUX_ARM", f"unknown aux arm {aux_arm!r}; "
                f"known {sorted(AUX_ARMS_V3)}")
        bucket, _share = AUX_ARMS_V3[aux_arm]
        if bucket == "strong" and float(lambda_dead) != 0.0:
            raise KeyAtomV3LossError(
                "AUX_ARM_MIXED", "strong arm active but lambda_dead != 0")
        if bucket == "dead" and float(lambda_strong) != 0.0:
            raise KeyAtomV3LossError(
                "AUX_ARM_MIXED", "dead arm active but lambda_strong != 0")
    elif float(lambda_strong) != 0.0 and float(lambda_dead) != 0.0:
        raise KeyAtomV3LossError(
            "AUX_ARM_MIXED", "strong and dead lambdas are both non-zero; "
            "S/D arms are trained separately (30h plan §4.1)")

    from pes2ts_core.pesn.contracts.key_atom import global_pair_rows_to_slots

    if pair_index is not None and pair_index.numel() > 0:
        pair_index = global_pair_rows_to_slots(
            pair_index, key_index, int(pos.shape[0]))

    n_graphs = int(key_ptr.shape[0]) - 1
    scale = float(cfg.loss_scale_angstrom)
    device = u.device
    dtype = u.dtype
    lo = key_ptr[:-1]
    hi = key_ptr[1:]

    # --- eligibility + weights (baseline-identical ops) ---------------------
    active = hi > lo
    if frame_mask is not None:
        active = active & frame_mask.to(device=device, dtype=torch.bool)
    n_used = int(active.sum())
    _require(n_used > 0, "NO_ELIGIBLE_FRAMES",
             "batch has no unmasked frames with key atoms")
    if frame_weight is None:
        weight = None
    else:
        weight = frame_weight.to(device=device, dtype=dtype)
        _require(weight.shape == (n_graphs,),
                 "FRAME_WEIGHT_SHAPE",
                 f"frame_weight must be [{n_graphs}], got {tuple(weight.shape)}")

    keep = active.to(dtype)
    w_batch = keep if weight is None else (keep * weight)

    atom_frame = torch.repeat_interleave(
        torch.arange(n_graphs, device=device), (hi - lo).clamp(min=0))
    # w_i = 1/max(1,degK) normalized per frame (baseline op order)
    w_raw = (1.0 / degK.to(dtype=dtype).clamp(min=1.0))
    w_sum = torch.zeros(n_graphs, dtype=dtype, device=device)
    w_sum.index_add_(0, atom_frame, w_raw)
    w = w_raw / w_sum[atom_frame].clamp(min=1e-12)

    # --- L_atom (verbatim baseline math) ------------------------------------
    e = (u - y).norm(dim=-1) / scale
    l_atom_atoms = w * radial_huber(e, cfg.huber_delta)
    l_atom = torch.zeros(n_graphs, dtype=dtype, device=device)
    l_atom.index_add_(0, atom_frame, l_atom_atoms)

    # --- L_pair (legacy_j verbatim baseline math) ---------------------------
    l_pair = torch.zeros(n_graphs, dtype=dtype, device=device)
    if pair_index is not None and pair_index.numel() > 0:
        pi = pair_index[0].long()
        pj = pair_index[1].long()
        e_l = pair_unit_vectors(pos, key_index, pair_index)
        dj = (e_l * (u[pj] - u[pi])).sum(dim=-1)
        dy = (e_l * (y[pj] - y[pi])).sum(dim=-1)
        pair_h = radial_huber(
            ((dj - dy).abs() / scale).to(dtype=dtype), cfg.huber_delta)
        pair_frame = atom_frame[pi]
        pair_sum = torch.zeros(n_graphs, dtype=dtype, device=device)
        pair_cnt = torch.zeros(n_graphs, dtype=dtype, device=device)
        pair_sum.index_add_(0, pair_frame, pair_h)
        pair_cnt.index_add_(0, pair_frame, torch.ones_like(pair_h))
        l_pair = pair_sum / pair_cnt.clamp(min=1.0)

    # --- L_cos (teacher-magnitude gate only; verbatim baseline math) --------
    g = (y.norm(dim=-1) > float(cfg.tau_norm_angstrom)).to(dtype)
    c = soft_cosine(u, y, cfg.cos_soft_norm_angstrom).to(dtype=dtype)
    wg = w * g
    denom = torch.zeros(n_graphs, dtype=dtype, device=device)
    denom.index_add_(0, atom_frame, wg)
    num = torch.zeros(n_graphs, dtype=dtype, device=device)
    num.index_add_(0, atom_frame, wg * (1.0 - c))
    l_cos = num / denom.clamp(min=1e-8)
    l_cos = torch.where(denom > 0.0, l_cos, torch.zeros_like(l_cos))

    # --- frame total + batch reduction (weights applied exactly once) -------
    frames = l_atom + float(cfg.lambda_pair) * l_pair + float(lambda_cos) * l_cos
    total = (frames * w_batch).sum() / w_batch.sum()

    # --- additive bucket terms (skipped entirely when lambda == 0) ----------
    l_strong = torch.zeros((), dtype=dtype, device=device)
    l_dead = torch.zeros((), dtype=dtype, device=device)
    n_strong_pairs = 0
    n_dead_pairs = 0
    want_strong = float(lambda_strong) != 0.0
    want_dead = float(lambda_dead) != 0.0
    if want_strong or want_dead:
        _require(pair_edit is not None, "BUCKET_NEEDS_PAIR_EDIT",
                 "strong/dead bucket losses require pair_edit")
        _require(pair_index is not None and pair_index.numel() > 0,
                 "BUCKET_NEEDS_PAIRS",
                 "strong/dead bucket losses require pair_index")
        _require(int(pair_edit.numel()) == int(pair_index.shape[1]),
                 "BUCKET_PAIR_EDIT_SHAPE",
                 f"pair_edit ({int(pair_edit.numel())}) must align with "
                 f"pair_index ({int(pair_index.shape[1])}) rows")
        pi_b = pair_index[0].long()
        pair_frame_b = atom_frame[pi_b]
        eligible = (pair_edit.to(device=pair_index.device) != 0) \
            & active[pair_frame_b]
        x_key = pos[key_index]
        dd_u = exact_delta_d_torch(x_key, pair_index, u)
        dd_y = exact_delta_d_torch(x_key, pair_index, y)
        if want_strong:
            l_strong, n_strong_pairs = strong_bucket_loss_v3(
                dd_u, dd_y, pair_frame_b, w_batch, n_graphs,
                eligible=eligible, config=cfg)
            total = total + float(lambda_strong) * l_strong
        if want_dead:
            l_dead, n_dead_pairs = dead_bucket_loss_v3(
                dd_u, dd_y, pair_frame_b, w_batch, n_graphs,
                eligible=eligible, config=cfg)
            total = total + float(lambda_dead) * l_dead

    _require(bool(torch.isfinite(total)), "NONFINITE_LOSS",
             "total loss is not finite")
    return {
        "loss": total,
        "l_atom": (l_atom * w_batch).sum() / w_batch.sum(),
        "l_pair": (l_pair * w_batch).sum() / w_batch.sum(),
        "l_cos": (l_cos * w_batch).sum() / w_batch.sum(),
        "l_strong": l_strong,
        "l_dead": l_dead,
        "lambda_cos": float(lambda_cos),
        "lambda_pair": float(cfg.lambda_pair),
        "lambda_strong": float(lambda_strong),
        "lambda_dead": float(lambda_dead),
        "aux_arm": str(aux_arm) if aux_arm is not None else "none",
        "n_strong_pairs": int(n_strong_pairs),
        "n_dead_pairs": int(n_dead_pairs),
        "n_frames": n_used,
        "schema": "pesn_key_atom_loss_v3",
    }


__all__ = [
    "AUX_ARMS_V3",
    "COS_ACTIVE_THRESHOLD_ANGSTROM",
    "COS_RAMP_UPDATES_R0",
    "COS_START_UPDATE_R0",
    "DEAD_DELTA_D_STAR_MAX_ANGSTROM",
    "DEAD_TAU_ANGSTROM",
    "D0_NORMALIZATION_ANGSTROM",
    "HUBER_DELTA",
    "KeyAtomLossV3Config",
    "KeyAtomV3LossError",
    "LAMBDA_COS_MAX",
    "LAMBDA_MAX_AUX",
    "LAMBDA_PAIR",
    "PAIR_MODE",
    "STRONG_DELTA_D_STAR_MIN_ANGSTROM",
    "STRONG_HUBER_DELTA",
    "STRONG_NORMALIZER_FLOOR_ANGSTROM",
    "aux_arm_lambdas_v3",
    "cos_lambda_r0",
    "dead_bucket_loss_v3",
    "exact_delta_d_torch",
    "key_atom_loss_v3",
    "pair_unit_vectors",
    "radial_huber",
    "resolve_aux_arm_v3",
    "soft_cosine",
    "strong_bucket_loss_v3",
]
