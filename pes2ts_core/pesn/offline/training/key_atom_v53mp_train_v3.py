"""V5.3-MP MOVE+PROCESS trainer (campaign PES2TS_V53MP; plan §5–§7).

Frozen-contract training loop for the V5.3-MP three arms (A/B/C):

* **init**: ``KeyAtomV53MPModel(backbone=StarEGNNV3)`` where ``backbone_init=
  "r3"`` loads the R3 winner checkpoint VERBATIM into ``StarEGNNV3``
  (payload["model"], strict=True — same discipline as V5.2's
  ``load_backbone_from_r3``); ``"scratch"`` builds a fresh seeded
  backbone.  A model carrying any ``legacy_frozen`` head (LEGACY_TS_STOP,
  contract §2/§6) is refused (MoveProcessContract v1 §6).
* **split protocol**: the driver supplies the scale-tier pool via
  :func:`v53mp_pool_split` (curve-level seeded train/inner-val split over
  ``pool_curve_ids``; the 2k/5k/20k pools are nested subsets of the frozen
  20k pool) and the frozen DEV positions from the split manifest.
  Confirm-subset curves (folds 3/4, V5.2 seal discipline) NEVER enter any
  position set — typed refusal (plan §7.1 冻结评估集; contract §7).
* **schedule**: stage A freezes the encoder for ``freeze_encoder_updates``
  (process heads only, lr_head=3e-4), then ``joint_updates`` joint updates
  (encoder lr_encoder=1e-4); AdamW (wd 1e-4, R3 continuity), grad clip
  1.0, fp32.  Arm A has no active head, so the freeze stage is vacuous
  (effective freeze = 0) — the encoder trains from update 1.
* **loss**: ``key_atom_loss_v53mp`` consumed read-only (never
  reimplemented): ``L_move + λr·L_rank + λq·L_q + λe·L_endpoint``;
  λ_cos held at the R0 end value (config ``lambda_cos_fixed``, 0.02).
* **MOVE eligibility**: the dataset is loaded with ALL label statuses; the
  MOVE loss consumes only ``label_status=="ok"`` rows (``frame_mask``;
  contract §4: legality mask affects MOVE only) with ``frame_weight`` =
  label_weight applied exactly once.  PROCESS supervision (q / ordering /
  endpoint) is eligible on ALL loaded rows per its own masks (q_mask from
  the v52 labels; endpoint_mask = endpoint_label ∈ {0,1}; contract §5).
* **ordering pairs**: sampled per batch within (curve, intent direction)
  groups with ``|q_j − q_i| > order_min_gap`` (cross-TS pairs ALLOWED,
  contract §5.2), capped at ``max_order_pairs_per_batch`` with a
  per-update seeded subsample; graph-local ``LongTensor [2, P]``.
  Draws are a PURE function of (seed, update) so resume is bit-identical.
* **endpoint pos_weight**: estimated ONCE from the inner-train rows at run
  start (n_neg / n_pos over unmasked endpoint labels), recorded with
  counts + provenance; never recomputed from dev (structural T3.3-style
  guarantee, plan §6).
* **gradient monitoring**: every ``grad_monitor_every`` updates the
  per-term scalar losses of the SAME forward are differentiated w.r.t. the
  shared encoder (retain_graph); per-term grad norms, pairwise cosines and
  ``encoder_share = ||λ-weighted process grad|| / (||move grad|| + eps)``
  are appended to ``train_log.jsonl`` (plan §6: λ 用 encoder 梯度占比校准).
* **selection + τ_end**: checkpoint_best on inner-val loss improvement;
  after selection, ``τ_end`` is calibrated on the inner-val rows (smallest
  threshold achieving the target precision; fallback: max-F1) and frozen
  into the metrics AND both checkpoint payloads (contract §8/§9).
* **metrics**: Move (RMSE/zero, sign, dead — frozen g52/g53 machinery via
  ``evaluate_checkpoint``) + MOVE GUARD ≤2% vs the matched R3 checkpoint
  zero-shot on the same dev positions (breach ⇒ ``run_outcome =
  "scientific_failure"`` — the run is KEPT, never "invalid"; contract §7),
  q head (per-curve MAE/Pearson vs constant-q baseline), ordering
  (full-path + cross-TS pairwise accuracy from q_pred), endpoint (AUROC /
  AUPRC / prevalence / mis-stop rates at frozen τ_end).  Reported
  SEPARATELY — no composite score (plan §7.1).

The driver (``scripts/campaigns/r3_v53mp_run.py``) owns ledger/lifecycle/
provenance; this module owns the math loop only.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    NODE_DEGK_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    forward_view_v3,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (
    build_sample_features_v3,
    collate_v3,
    to_torch_batch,
)
from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (
    DoubleBufferedPrefetcher,
    FeaturePrefetcher,
    log_prefetch_issue,
    maybe_create_prefetcher,
)
from pes2ts_core.pesn.offline.training.h2d_pinned_v53mp import (
    PinnedBufferPool,
    to_torch_batch_pinned,
)
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
    TrainerError,
    evaluate_checkpoint,
    set_seed,
    sha256_file,
)
from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (
    auc_mann_whitney,
    pearson,
)

__all__ = [
    "CONFIRM_FOLDS_V53MP",
    "PROVENANCE_PLACEHOLDER_KEYS",
    "V53MP_SCHEMA",
    "V53MPTrainerConfig",
    "V53MPTrainerError",
    "apply_cuda_memory_fraction",
    "assert_no_legacy_head",
    "average_precision",
    "calibrate_tau_end",
    "estimate_endpoint_pos_weight",
    "evaluate_v53mp",
    "load_model_v53mp",
    "ordering_pair_accuracy",
    "sample_order_pairs",
    "train_run_v53mp",
    "v53mp_pool_split",
]

V53MP_SCHEMA = "pesn_key_atom_v53mp_train_v1"

#: Confirm-seal folds (V5.2 discipline: dev = folds 0/1/2, confirm = 3/4).
#: Confirm curves NEVER enter train / inner-val / dev position sets.
CONFIRM_FOLDS_V53MP = frozenset({3, 4})

#: Metric provenance keys the run driver fills; None until then.
PROVENANCE_PLACEHOLDER_KEYS = (
    "campaign_run_id",
    "tier",
    "data_manifest_sha256",
    "split_manifest_sha256",
    "teacher_manifest_sha256",
    "endpoint_label_spec_sha256",
)

#: Frozen feature level for on-demand feature building (V5.2 E3 winner
#: config is F2-bidir; the V5.3-MP first round keeps the same inputs).
FEATURE_LEVEL_V53MP = "F2"


class V53MPTrainerError(TrainerError):
    """V5.3-MP trainer input/state violation."""


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise V53MPTrainerError(code, detail)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class V53MPTrainerConfig:
    """One planned V5.3-MP run (arm A/B/C; contract §7 + plan §6–§7)."""

    arm: str                       # "A" | "B" | "C"
    seed: int
    backbone_init: str = "r3"      # "r3" (R3 winner verbatim) | "scratch"
    lambda_rank: float = 0.0
    lambda_q: float = 0.0
    lambda_endpoint: float = 0.0
    rank_margin: float = 0.1
    order_min_gap: float = 0.05    # min |q_j − q_i| for an ordering pair
    max_order_pairs_per_batch: int = 256
    freeze_encoder_updates: int = 400
    joint_updates: int = 1500
    lr_head: float = 3.0e-4
    lr_encoder: float = 1.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    lambda_cos_fixed: float = 0.02
    val_every: int = 100
    grad_monitor_every: int = 100
    batch_curves: int = 8
    frames_per_curve: int = 4
    inner_val_fraction: float = 0.2
    tau_target_precision: float = 0.9
    move_guard_max_degradation_pct: float = 2.0
    smoke_updates: Optional[int] = None   # dev-stability smoke ONLY
    device: str = "cpu"
    feature_cache_capacity: int = 8192    # LRU samples; on-demand features
    #: dev-eval cap (seeded subsample of the frozen dev positions; None =
    #: full dev).  Only the S1 smoke tier uses it (≤1 GPU-h budget);
    #: S2/S3 evaluate the FULL frozen dev set (comparability).
    eval_max_samples: Optional[int] = None
    #: inner-val cap (seeded, FIXED per run): selection + τ_end calibration
    #: use this subset; keeps the per-100-update inner-val sweep cheap at
    #: 5k/20k scale.  None = the full inner-val pool.
    inner_val_max_samples: Optional[int] = None
    #: per-process CUDA memory fraction cap (torch.cuda.
    #: set_per_process_memory_fraction, 0 < f <= 1); applied on CUDA devices
    #: BEFORE the model load / first real allocation.  None = no cap —
    #: zero behaviour change vs the pre-cap trainer.
    cuda_memory_fraction: Optional[float] = None

    def __post_init__(self) -> None:
        if self.arm not in ("A", "B", "C"):
            raise V53MPTrainerError(
                "ARM", "unknown V5.3-MP arm {!r}".format(self.arm))
        if self.backbone_init not in ("r3", "scratch"):
            raise V53MPTrainerError(
                "BACKBONE_INIT",
                "backbone_init must be 'r3' or 'scratch', got {!r}".format(
                    self.backbone_init))
        lambdas = (float(self.lambda_rank), float(self.lambda_q),
                   float(self.lambda_endpoint))
        if any(v < 0.0 for v in lambdas):
            raise V53MPTrainerError(
                "LAMBDA_NEG",
                "lambda_rank/lambda_q/lambda_endpoint must be >= 0, got "
                "{}/{}/{}".format(*lambdas))
        # typed arm refusals (contract §7: L_process = λr·rank + λq·q + λe·end)
        if self.arm == "A" and any(v != 0.0 for v in lambdas):
            raise V53MPTrainerError(
                "LAMBDA_ARM_MISMATCH",
                "arm A is Move-only: all λ must be 0, got "
                "{}/{}/{}".format(*lambdas))
        if self.arm == "B" and not (
                lambdas[0] > 0.0 and lambdas[1] > 0.0 and lambdas[2] == 0.0):
            raise V53MPTrainerError(
                "LAMBDA_ARM_MISMATCH",
                "arm B needs lambda_rank>0 and lambda_q>0 and "
                "lambda_endpoint==0, got {}/{}/{}".format(*lambdas))
        if self.arm == "C" and not all(v > 0.0 for v in lambdas):
            raise V53MPTrainerError(
                "LAMBDA_ARM_MISMATCH",
                "arm C needs all three λ > 0, got {}/{}/{}".format(*lambdas))
        for name, value in zip(
                ("lambda_rank", "lambda_q", "lambda_endpoint"), lambdas):
            if value != 0.0 and not (0.0 < value <= 0.5):
                raise V53MPTrainerError(
                    "LAMBDA_RANGE",
                    "{} must lie in (0, 0.5] when active, got {}".format(
                        name, value))
        if float(self.rank_margin) <= 0.0:
            raise V53MPTrainerError(
                "RANK_MARGIN", "rank_margin must be > 0")
        if float(self.order_min_gap) < 0.0:
            raise V53MPTrainerError(
                "ORDER_MIN_GAP", "order_min_gap must be >= 0")
        if int(self.max_order_pairs_per_batch) < 1:
            raise V53MPTrainerError(
                "ORDER_PAIRS_CAP", "max_order_pairs_per_batch must be >= 1")
        if int(self.freeze_encoder_updates) < 0 or int(self.joint_updates) < 1:
            raise V53MPTrainerError(
                "SCHEDULE",
                "freeze_encoder_updates >= 0 and joint_updates >= 1 required")
        if float(self.lr_head) <= 0.0 or float(self.lr_encoder) <= 0.0:
            raise V53MPTrainerError("LR", "lr_head/lr_encoder must be > 0")
        if float(self.weight_decay) < 0.0:
            raise V53MPTrainerError("WEIGHT_DECAY", "weight_decay must be >= 0")
        if float(self.grad_clip) <= 0.0:
            raise V53MPTrainerError("GRAD_CLIP", "grad_clip must be > 0")
        if int(self.val_every) < 1 or int(self.grad_monitor_every) < 1:
            raise V53MPTrainerError(
                "EVERY", "val_every and grad_monitor_every must be >= 1")
        if int(self.batch_curves) < 1 or int(self.frames_per_curve) < 2:
            raise V53MPTrainerError(
                "BATCH_COMPOSITION",
                "batch_curves >= 1 and frames_per_curve >= 2 required")
        if not (0.0 < float(self.inner_val_fraction) < 1.0):
            raise V53MPTrainerError(
                "INNER_VAL_FRACTION",
                "inner_val_fraction must lie in (0, 1)")
        if self.cuda_memory_fraction is not None and not (
                0.0 < float(self.cuda_memory_fraction) <= 1.0):
            raise V53MPTrainerError(
                "CUDA_MEMORY_FRACTION",
                "cuda_memory_fraction must lie in (0, 1] when set, got "
                "{}".format(self.cuda_memory_fraction))
        if not (0.0 < float(self.tau_target_precision) <= 1.0):
            raise V53MPTrainerError(
                "TAU_TARGET",
                "tau_target_precision must lie in (0, 1]")
        if float(self.move_guard_max_degradation_pct) < 0.0:
            raise V53MPTrainerError(
                "MOVE_GUARD",
                "move_guard_max_degradation_pct must be >= 0")
        if self.smoke_updates is not None and int(self.smoke_updates) < 1:
            raise V53MPTrainerError(
                "SMOKE_UPDATES", "smoke_updates must be >= 1 when set")
        if int(self.feature_cache_capacity) < 1:
            raise V53MPTrainerError(
                "FEATURE_CACHE", "feature_cache_capacity must be >= 1")

    @property
    def train_rank(self) -> bool:
        return self.arm in ("B", "C")

    @property
    def train_q(self) -> bool:
        return self.arm in ("B", "C")

    @property
    def train_endpoint(self) -> bool:
        return self.arm == "C"

    @property
    def effective_freeze_updates(self) -> int:
        """Arm A has no active head — the freeze stage is vacuous."""
        return 0 if self.arm == "A" else int(self.freeze_encoder_updates)

    @property
    def total_updates(self) -> int:
        return int(self.smoke_updates) if self.smoke_updates else (
            int(self.freeze_encoder_updates) + int(self.joint_updates))


# ---------------------------------------------------------------------------
# split protocol (curve-level, confirm firewall)
# ---------------------------------------------------------------------------

#: I4 (plan §7-B2): columnar startup fast paths over the v2 loader's
#: vectorized columns (identical results; per-item paths kept verbatim).
#: ``V53MP_DISABLE_COLUMNAR_FAST=1`` at import forces the per-item path —
#: the startup certificate's reference arm.  Tests toggle the module flag.
_COLUMNAR_FAST = os.environ.get("V53MP_DISABLE_COLUMNAR_FAST", "") != "1"


def _startup_of(samples: Any) -> Any:
    """Vectorized startup columns for ``samples``; None = per-item path.

    Works for the v2 lazy sequence (``samples._ds.startup``); the v1
    tuple / full loader has no ``_ds`` and always gets None (v1 path
    byte-identical by construction).
    """
    if not _COLUMNAR_FAST:
        return None
    ds = getattr(samples, "_ds", None)
    if ds is None:
        return None
    return getattr(ds, "startup", None)


def _curve_id_set_of(samples: Any, positions: Sequence[int]) -> set:
    """Set of curve_ids of ``positions`` (vectorized on the v2 loader)."""
    st = _startup_of(samples)
    if st is not None:
        return st.curve_id_set(positions)
    return {str(samples[int(p)].curve_id) for p in positions}


def _assert_label_coverage(
    samples: Any,
    labels_v52: Any,
    endpoint_labels: Any,
    needed: Sequence[Sequence[int]],
    n_samples: int,
) -> None:
    """Fail-closed label coverage over the FULL used slice.

    Per-item path (v1 / eager tables): every position's sample_id must be
    in both tables (VERBATIM from the original loop).  Columnar fast path
    (I4): the lazy v2 label tables are written in SAMPLE ORDER by the idx
    builder (which asserted full coverage fail-closed), so
    ``covers_all_positions`` (row-count equality) is the structural check;
    a small probe of REAL ``by_sample`` lookups (first/last of each group)
    pins the row-order identity.  Any probe miss falls through to the
    per-item path for that group.
    """
    fast = (_startup_of(samples) is not None
            and bool(getattr(labels_v52, "covers_all_positions", None))
            and bool(getattr(endpoint_labels, "covers_all_positions", None)))
    if fast:
        for tbl, name in ((labels_v52, "v52"),
                          (endpoint_labels, "endpoint")):
            _require(tbl.covers_all_positions(n_samples),
                     "LABEL_COVERAGE" if name == "v52"
                     else "ENDPOINT_LABEL_COVERAGE",
                     "{} table covers {} rows, dataset has {}".format(
                         name, tbl._n_rows, n_samples))
        # probe real lookups at group boundaries (row-order identity)
        for group in needed:
            if not group:
                continue
            for p in (int(group[0]), int(group[-1])):
                sid = str(samples[int(p)].sample_id)
                _require(sid in labels_v52.by_sample, "LABEL_MISSING",
                         "sample {} has no v52 label row".format(sid))
                _require(sid in endpoint_labels.by_sample,
                         "ENDPOINT_LABEL_MISSING",
                         "sample {} has no endpoint label row".format(sid))
        return
    for group in needed:
        for p in group:
            sid = str(samples[int(p)].sample_id)
            _require(sid in labels_v52.by_sample, "LABEL_MISSING",
                     "sample {} has no v52 label row".format(sid))
            _require(sid in endpoint_labels.by_sample,
                     "ENDPOINT_LABEL_MISSING",
                     "sample {} has no endpoint label row".format(sid))


def _curve_fold(samples: Sequence[Any], index: int) -> Optional[int]:
    value = getattr(samples[int(index)], "fold", None)
    return None if value is None else int(value)


def _assert_no_confirm(
    samples: Sequence[Any], positions: Sequence[int], where: str,
) -> None:
    """Confirm-subset folds (3/4) may never enter any position set."""
    st = _startup_of(samples)
    if st is not None and len(positions) > 0:
        # I4 fast path: gather folds (vectorized); -1 = None (never 3/4),
        # so only int values in {3,4} can refuse — identical to the
        # per-item {fold} & CONFIRM_FOLDS intersection below.
        folds = st.fold_array(positions)
        bad = sorted({int(f) for f in folds if int(f) in CONFIRM_FOLDS_V53MP})
        _require(not bad, "CONFIRM_POSITIONS",
                 "{} positions touch confirm folds {}".format(where, bad))
        return
    bad = sorted({_curve_fold(samples, p) for p in positions}
                 & CONFIRM_FOLDS_V53MP)
    _require(not bad, "CONFIRM_POSITIONS",
             "{} positions touch confirm folds {}".format(where, bad))


def v53mp_pool_split(
    samples: Sequence[Any],
    *,
    pool_curve_ids: Sequence[str],
    seed: int,
    inner_val_fraction: float = 0.2,
) -> Tuple[List[int], List[int]]:
    """Curve-level seeded train/inner-val split of the scale-tier pool.

    ``pool_curve_ids`` is the set of curves of the CURRENT scale tier
    (2k/5k/20k nested pools; plan §5).  Returns ``(train_pos,
    inner_val_pos)`` — positions into ``samples`` — disjoint at the curve
    level; inner-val is for checkpoint selection ONLY.  Confirm folds
    (3/4) inside the pool are a typed refusal (plan §7.1 / contract §7:
    最终确认集隔离).
    """
    pool_ids = {str(c) for c in pool_curve_ids}
    _require(bool(pool_ids), "POOL_EMPTY", "pool_curve_ids is empty")
    st = _startup_of(samples)
    if st is not None:
        # I4 fast path (vectorized): EXACTLY the same result.  The seeded
        # rng draws happen at CURVE level over the STRING-sorted curve list
        # (rng.shuffle(shuffled)) — replayed verbatim below, so the draw
        # sequence (and every downstream position) is identical; only the
        # pool/curve MASKING is vectorized.
        pool = st.positions_in_curves(pool_ids)
        _require(pool.size > 0, "POOL_EMPTY",
                 "no samples for pool curves {}".format(sorted(pool_ids)[:5]))
        _assert_no_confirm(samples, pool, "pool")
        curves = st.unique_sorted_curve_ids(pool)
        _require(len(curves) >= 2, "SPLIT_CURVES",
                 "need >=2 pool curves, got {}".format(len(curves)))
        rng = np.random.RandomState(int(seed) ^ 0x5EED53)
        shuffled = list(curves)
        rng.shuffle(shuffled)                      # IDENTICAL draw sequence
        n_val = max(1, int(round(float(inner_val_fraction) * len(shuffled))))
        n_val = min(n_val, len(shuffled) - 1)
        val_curves = set(shuffled[:n_val])
        val_rows = np.asarray(
            sorted(st.curve_row[c] for c in val_curves), dtype=np.int64)
        is_val = np.isin(np.asarray(st.curve_idx)[pool], val_rows)
        val = pool[is_val].tolist()
        train = pool[~is_val].tolist()
        _require(bool(train) and bool(val), "SPLIT_EMPTY",
                 "inner split produced an empty side")
        return train, val
    pool = [i for i, s in enumerate(samples) if str(s.curve_id) in pool_ids]
    _require(bool(pool), "POOL_EMPTY",
             "no samples for pool curves {}".format(sorted(pool_ids)[:5]))
    _assert_no_confirm(samples, pool, "pool")
    curves = sorted({str(samples[i].curve_id) for i in pool})
    _require(len(curves) >= 2, "SPLIT_CURVES",
             "need >=2 pool curves, got {}".format(len(curves)))
    rng = np.random.RandomState(int(seed) ^ 0x5EED53)
    shuffled = list(curves)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(float(inner_val_fraction) * len(shuffled))))
    n_val = min(n_val, len(shuffled) - 1)
    val_curves = set(shuffled[:n_val])
    train = [i for i in pool
             if str(samples[i].curve_id) not in val_curves]
    val = [i for i in pool
           if str(samples[i].curve_id) in val_curves]
    _require(bool(train) and bool(val), "SPLIT_EMPTY",
             "inner split produced an empty side")
    return train, val


# ---------------------------------------------------------------------------
# model init / frozen-phase helpers
# ---------------------------------------------------------------------------


def assert_no_legacy_head(model: torch.nn.Module) -> None:
    """Refuse any module carrying ``legacy_frozen`` (LEGACY_TS_STOP).

    Contract §2/§6: the V5.2 TS-stop head is 弃用 — new training must use
    the V5.3 endpoint head; a model wiring a legacy head into a training
    run is a typed refusal, not a warning.
    """
    for name, module in model.named_modules():
        if bool(getattr(module, "legacy_frozen", False)):
            raise V53MPTrainerError(
                "LEGACY_HEAD",
                "module {} carries legacy_frozen=True (LEGACY_TS_STOP) — "
                "refusing new training (contract §6)".format(
                    name or type(module).__name__))


def _cuda_device_index(device: str) -> int:
    """Device index from a ``"cuda"`` / ``"cuda:N"`` device string (default 0)."""
    text = str(device)
    if ":" in text:
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


def apply_cuda_memory_fraction(config: V53MPTrainerConfig) -> Optional[float]:
    """Cap this process's CUDA allocator BEFORE the first real allocation.

    Active ONLY for CUDA devices with ``cuda_memory_fraction`` set: calls
    ``torch.cuda.init()`` then ``torch.cuda.set_per_process_memory_fraction``
    on the run's device index.  On CPU devices (or when the fraction is
    ``None``) this is a no-op returning ``None`` — zero behaviour change vs
    the pre-cap trainer.  Returns the applied fraction (float) or None.
    """
    if config.cuda_memory_fraction is None:
        return None
    if not str(config.device).startswith("cuda"):
        return None
    fraction = float(config.cuda_memory_fraction)
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(
        fraction, _cuda_device_index(config.device))
    return fraction


def _reset_peak_memory_stats(device: str) -> None:
    """Reset CUDA peak-memory stats (no-op on CPU; never crashes)."""
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def _gpu_memory_metrics(config: V53MPTrainerConfig) -> Dict[str, Any]:
    """Top-level peak-memory report keys for the run metrics.

    CUDA devices report the post-run peaks in BYTES (integers) plus the
    fraction actually applied (float or None).  CPU devices never touch
    torch.cuda: they write 0/0/None (explicit, tested).
    """
    if not str(config.device).startswith("cuda"):
        return {
            "gpu_max_memory_allocated_bytes": 0,
            "gpu_max_memory_reserved_bytes": 0,
            "gpu_memory_fraction_applied": None,
        }
    return {
        "gpu_max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(config.device)),
        "gpu_max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(config.device)),
        "gpu_memory_fraction_applied": (
            None if config.cuda_memory_fraction is None
            else float(config.cuda_memory_fraction)),
    }


def load_model_v53mp(
    config: V53MPTrainerConfig, *, r3_ckpt: Any = None, device: str = "cpu",
) -> torch.nn.Module:
    """Build the run model; ``backbone_init="r3"`` loads R3 VERBATIM."""
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    if config.backbone_init == "r3":
        _require(r3_ckpt is not None, "R3_CHECKPOINT_MISSING",
                 "backbone_init='r3' requires r3_ckpt")
        payload = torch.load(str(r3_ckpt), map_location="cpu",
                             weights_only=False)
        _require(isinstance(payload, Mapping) and "model" in payload,
                 "R3_CHECKPOINT_FORMAT",
                 "{} is not an R3 trainer checkpoint".format(r3_ckpt))
        backbone = StarEGNNV3()
        backbone.load_state_dict(payload["model"], strict=True)
        model = KeyAtomV53MPModel(backbone=backbone)
    else:
        model = KeyAtomV53MPModel()          # fresh seeded backbone + heads
    assert_no_legacy_head(model)
    return model.to(device)


def _set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    for param in model.backbone.parameters():     # shared by reference
        param.requires_grad_(trainable)


def _set_process_heads_trainable(model: torch.nn.Module, active: bool) -> None:
    """Process heads train only for arms B/C (arm A: zero gradient path)."""
    for name, param in model.named_parameters():
        if not name.startswith("backbone."):
            param.requires_grad_(active)


def _process_head_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
            if not name.startswith("backbone.")}


# ---------------------------------------------------------------------------
# on-demand feature LRU (plan §4.4/S0-D: never pre-materialize at 20k)
# ---------------------------------------------------------------------------


class _FeatureLRU:
    """Bounded LRU of built F2 feature dicts, keyed by dataset position."""

    def __init__(self, dataset: Any, *, capacity: int,
                 level: str = FEATURE_LEVEL_V53MP,
                 intent_off: bool = False) -> None:
        self._dataset = dataset
        self._capacity = int(capacity)
        self._level = level
        self._intent_off = intent_off
        self._data: "OrderedDict[int, Mapping[str, Any]]" = OrderedDict()

    def contains(self, position: int) -> bool:
        return int(position) in self._data

    def _finalize(self, rec: Any,
                  built: Mapping[str, Any]) -> Mapping[str, Any]:
        # V5.3-MP loads ALL label statuses; terminal/masked rows carry no
        # teacher (y_teacher=None).  Zero-pad so collated batches always
        # carry ``y_teacher`` — the MOVE loss's frame_mask + zero
        # frame_weight exclude these rows from numerator and denominator,
        # so the padding never contributes a gradient.
        if built.get("y_teacher") is None:
            n_key = len(rec.intent.key_rows())
            built = dict(built)
            built["y_teacher"] = np.zeros((n_key, 3), dtype=np.float32)
        return built

    def _insert(self, key: int, built: Mapping[str, Any]) -> Mapping[str, Any]:
        self._data[key] = built
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)
        return built

    def put_built(self, position: int, rec: Any,
                  built: Mapping[str, Any]) -> Mapping[str, Any]:
        """Insert a PREFETCHED feature dict (same finalize as the serial path).

        Features are pure functions of the record, so a prefetched dict is
        bit-identical to what :meth:`get` would have built; ``put_built``
        only replays the exact post-processing (y_teacher zero-pad) and the
        identical eviction rule, keeping serial and parallel paths
        numerics-equivalent.
        """
        key = int(position)
        hit = self._data.get(key)
        if hit is not None:
            return hit
        return self._insert(key, self._finalize(rec, built))

    def get(self, position: int) -> Mapping[str, Any]:
        key = int(position)
        hit = self._data.pop(key, None)
        if hit is not None:
            self._data[key] = hit
            return hit
        rec = self._dataset.records([key])[0]
        built = build_sample_features_v3(
            self._dataset.to_sample_v3(rec),
            level=self._level, intent_off=self._intent_off)
        return self._insert(key, self._finalize(rec, built))


class _CacheView:
    """``cache[position]`` adapter over an LRU (evaluate_checkpoint API)."""

    def __init__(self, lru: _FeatureLRU) -> None:
        self._lru = lru

    def __getitem__(self, position: int) -> Mapping[str, Any]:
        return self._lru.get(int(position))


# ---------------------------------------------------------------------------
# per-update deterministic draws (resume-identical by construction)
# ---------------------------------------------------------------------------


def _update_rng(seed: int, update: int, salt: int = 0) -> np.random.RandomState:
    """PURE per-update RNG — batch/pair draws depend only on (seed, update)."""
    key = (int(seed) * 1_000_003 + int(update) * 9_176
           + int(salt) * 7_919) % (2 ** 32)
    return np.random.RandomState(key)


class _CurveDirIndex:
    """Prebuilt ``(curve, direction)`` pools + sorted curve list.

    Built ONCE per run over the train pool (one pass, ascending position
    order) so each update's :func:`_curve_batch_positions` draw does not
    rescan the ~1M–5M-entry pool.  Structure is bit-identical to the
    per-update rebuild it replaces: ``pools`` appends positions in
    ``indices`` iteration order; ``curves`` is the sorted unique curve-id
    list — the ``rng.choice`` draws are therefore unchanged.
    """

    __slots__ = ("curves", "pools")

    def __init__(
        self,
        curves: List[str],
        pools: Dict[Tuple[str, str], List[int]],
    ) -> None:
        self.curves = curves
        self.pools = pools

    @classmethod
    def build(
        cls,
        samples: Sequence[Any],
        indices: Sequence[int],
    ) -> "_CurveDirIndex":
        st = _startup_of(samples)
        if st is not None and len(indices) > 0:
            # I4 fast path (vectorized): same pools (each group's positions
            # in ASCENDING order — a stable argsort of the ascending index
            # array preserves it) and the same STRING-sorted curve list.
            # The pools dict INSERTION order differs but only lookups occur
            # (never iteration), so the sampler draws are identical.
            idx = np.asarray(indices, dtype=np.int64)
            ci = np.asarray(st.curve_idx)[idx]
            di = st.direction_int(idx)
            key = ci.astype(np.int64) * 2 + di.astype(np.int64)
            order = np.argsort(key, kind="stable")
            keys_sorted = key[order]
            runs = np.split(order, np.flatnonzero(np.diff(keys_sorted)) + 1)
            pools: Dict[Tuple[str, str], List[int]] = {}
            for run in runs:
                k = int(key[int(run[0])])     # run holds ORIGINAL indices
                cid = st.curve_ids[k // 2]
                direction = "forward" if (k % 2) == 0 else "reverse"
                pools[(cid, direction)] = idx[run].tolist()
            curves = sorted(st.curve_ids[int(c)]
                            for c in np.unique(ci))
            return cls(curves, pools)
        pools: Dict[Tuple[str, str], List[int]] = {}
        seen: set = set()
        for i in indices:
            rec = samples[int(i)]
            cid = str(rec.curve_id)
            seen.add(cid)
            pools.setdefault(
                (cid, str(rec.intent.direction)), []).append(int(i))
        return cls(sorted(seen), pools)


def _curve_batch_positions(
    samples: Sequence[Any],
    indices: Sequence[int],
    *,
    config: V53MPTrainerConfig,
    rng: np.random.RandomState,
    index: Optional[_CurveDirIndex] = None,
) -> List[int]:
    """One curve-balanced batch (dataset positions) from ``indices``.

    ``index`` (optional) supplies the prebuilt pools/curve list (see
    :class:`_CurveDirIndex`); when ``None`` they are rebuilt here, which
    is bit-identical to the prebuilt path (same containers, same order).
    """
    if index is None:
        index = _CurveDirIndex.build(samples, indices)
    by_curve_dir = index.pools
    curves = index.curves
    if not curves:
        raise V53MPTrainerError("SAMPLER_EMPTY", "no train curves to sample")
    chosen = list(rng.choice(
        curves, size=int(config.batch_curves),
        replace=len(curves) < int(config.batch_curves)))
    batch: List[int] = []
    half = int(config.frames_per_curve) // 2
    per_dir = (("forward", half),
               ("reverse", int(config.frames_per_curve) - half))
    for curve in chosen:
        for direction, count in per_dir:
            pool = by_curve_dir.get((curve, direction)) or \
                by_curve_dir.get((curve, "forward")) or \
                by_curve_dir.get((curve, "reverse"))
            if not pool:
                continue
            picks = rng.choice(
                pool, size=count, replace=len(pool) < count)
            batch.extend(int(p) for p in np.atleast_1d(picks))
    if not batch:
        raise V53MPTrainerError("SAMPLER_EMPTY", "no frames drawn")
    return batch


def sample_order_pairs(
    batch: Mapping[str, Any],
    labels_v52: Any,
    *,
    order_min_gap: float,
    max_pairs: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """Directed-ordering pairs for ONE batch (contract §5.2; plan §6).

    Groups the batch's graphs by (curve_id, intent direction); within each
    group every ``(i, j)`` with ``q_true[j] > q_true[i] + order_min_gap``
    (both rows q-masked; CROSS-TS pairs allowed, never restricted to one
    side) is a candidate.  Candidates are subsampled with ``rng`` down to
    ``max_pairs`` and returned as a graph-local ``LongTensor [2, P]``
    (row 0 = earlier ``i``, row 1 = later ``j``).  ``P == 0`` returns an
    EMPTY ``[2, 0]`` tensor — the frozen loss yields exactly 0 for it
    (it types-refuses ``None`` when λ_rank > 0, so the trainer never
    passes None for an active arm).
    """
    sample_ids = [str(s) for s in batch["sample_ids"]]
    groups: Dict[Tuple[str, str], List[int]] = {}
    q_vals: List[float] = []
    q_valid: List[bool] = []
    for g, sid in enumerate(sample_ids):
        if sid not in labels_v52.by_sample:
            raise V53MPTrainerError(
                "LABEL_MISSING", "sample {} has no v52 label row".format(sid))
        lab = labels_v52.by_sample[sid]
        valid = bool(lab.q_mask) and lab.q is not None
        q_valid.append(valid)
        q_vals.append(float(lab.q) if valid else 0.0)
        if valid:
            groups.setdefault(
                (str(lab.curve_id), str(lab.direction)), []).append(g)
    candidates: List[Tuple[int, int]] = []
    gap = float(order_min_gap)
    for key in sorted(groups):
        idxs = groups[key]
        for a, i in enumerate(idxs):
            for j in idxs[a + 1:]:
                qi, qj = q_vals[i], q_vals[j]
                if qj > qi + gap:
                    candidates.append((i, j))
                elif qi > qj + gap:
                    candidates.append((j, i))
    if not candidates:
        return torch.zeros((2, 0), dtype=torch.long)
    if len(candidates) > int(max_pairs):
        pick = rng.choice(len(candidates), size=int(max_pairs), replace=False)
        keep = sorted(int(x) for x in np.atleast_1d(pick))
        candidates = [candidates[k] for k in keep]
    return torch.tensor(
        [[c[0] for c in candidates], [c[1] for c in candidates]],
        dtype=torch.long)


# ---------------------------------------------------------------------------
# batch labels / masks
# ---------------------------------------------------------------------------


def _key_ptr(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    n_keys = batch["n_keys"].long()
    return torch.cat(
        [torch.zeros(1, dtype=torch.long, device=n_keys.device),
         torch.cumsum(n_keys, dim=0)])


def _labels_for_batch(
    sample_ids: Sequence[str], labels_v52: Any, device: str,
) -> Dict[str, torch.Tensor]:
    """Intent-signed q labels ([G], [G]) for the batch's sample_ids."""
    q_true: List[float] = []
    q_mask: List[bool] = []
    for sid in sample_ids:
        sid = str(sid)
        if sid not in labels_v52.by_sample:
            raise V53MPTrainerError(
                "LABEL_MISSING", "sample {} has no v52 label row".format(sid))
        lab = labels_v52.by_sample[sid]
        q_true.append(0.0 if lab.q is None else float(lab.q))
        q_mask.append(bool(lab.q_mask))
    return {
        "q_true": torch.tensor(q_true, dtype=torch.float32, device=device),
        "q_mask": torch.tensor(q_mask, dtype=torch.bool, device=device),
    }


def _endpoint_for_batch(
    sample_ids: Sequence[str], endpoint_labels: Any, device: str,
) -> Dict[str, Any]:
    """Endpoint targets/masks ([G]) + row flags from the endpoint table.

    ``endpoint_mask = (endpoint_label ∈ {0,1})`` — Mask rows (−1 and any
    other value) never enter numerator or denominator (contract §5.3).
    """
    y: List[float] = []
    mask: List[bool] = []
    ts_row: List[bool] = []
    start_row: List[bool] = []
    for sid in sample_ids:
        sid = str(sid)
        if sid not in endpoint_labels.by_sample:
            raise V53MPTrainerError(
                "ENDPOINT_LABEL_MISSING",
                "sample {} has no endpoint label row".format(sid))
        lab = endpoint_labels.by_sample[sid]
        label = int(lab.endpoint_label)
        mask.append(label in (0, 1))
        y.append(1.0 if label == 1 else 0.0)
        ts_row.append(bool(lab.ts_row))
        start_row.append(bool(lab.start_row))
    return {
        "y_endpoint": torch.tensor(y, dtype=torch.float32, device=device),
        "endpoint_mask": torch.tensor(mask, dtype=torch.bool, device=device),
        "ts_row": ts_row,
        "start_row": start_row,
    }


def _frame_mask_for(records: Sequence[Any], device: str) -> torch.Tensor:
    """MOVE eligibility: only label_status=="ok" rows (contract §4)."""
    return torch.tensor(
        [str(getattr(rec, "label_status", "")) == "ok" for rec in records],
        dtype=torch.bool, device=device)


def _prefetch_fill(
    lru: _FeatureLRU,
    dataset: Any,
    positions: Sequence[int],
    prefetcher: Optional[FeaturePrefetcher],
) -> None:
    """Best-effort parallel build of the CACHE MISSES among ``positions``.

    The main process keeps doing ``dataset.records``/``to_sample_v3`` (they
    need the dataset's geometry LRU and are cheap); only
    ``build_sample_features_v3`` runs in the fork workers, over contiguous
    chunks (see ``FeaturePrefetcher.build_many``).  On ANY failure the pool
    logs one stderr line, closes itself, and the untouched misses are built
    by the serial ``lru.get`` path below — a prefetch bug costs performance
    only, never a run.
    """
    if prefetcher is None or not prefetcher.active:
        return
    missing = [int(p) for p in positions if not lru.contains(int(p))]
    if not missing:
        return
    try:
        recs = dataset.records(missing)
        samples = [dataset.to_sample_v3(rec) for rec in recs]
        built_list = prefetcher.build_many(samples)
        if built_list is None or len(built_list) != len(missing):
            # pool disabled/degraded (already logged) — serial path builds
            return
        for pos, rec, built in zip(missing, recs, built_list):
            lru.put_built(pos, rec, built)
    except Exception as exc:
        log_prefetch_issue(
            "lru fill failed ({}: {}); falling back to serial".format(
                type(exc).__name__, exc))
        if prefetcher is not None:
            prefetcher.terminate()


def _prefetch_warm_lru(
    lru: _FeatureLRU,
    dataset: Any,
    positions: Sequence[int],
    prefetcher: Optional[FeaturePrefetcher],
    *,
    chunk: int = 256,
) -> None:
    """Pre-warm an LRU (dev eval cache) in parallel chunks of ``chunk``.

    Best-effort: any mid-way failure stops warming early and the eval's own
    ``cache[position]`` reads build the rest serially — numerics unchanged
    (features are pure functions of the record).
    """
    if prefetcher is None or not prefetcher.active:
        return
    pos = [int(p) for p in positions]
    for start in range(0, len(pos), int(chunk)):
        _prefetch_fill(lru, dataset, pos[start:start + int(chunk)], prefetcher)
        if not prefetcher.active:
            return


def _batch_from_positions(
    dataset: Any, lru: _FeatureLRU, positions: Sequence[int], device: str,
    prefetcher: Optional[FeaturePrefetcher] = None,
    *,
    h2d_pool: Optional[Any] = None,
    dbuf: Optional[Any] = None,
    update: Optional[int] = None,
    dataset_gate: Optional[Any] = None,
) -> Tuple[Dict[str, Any], List[Any]]:
    """One batch (records + collated torch batch) for ``positions``.

    I3 additions (all default-off; defaults reproduce the pre-I3 path
    byte-for-byte):
      * ``h2d_pool`` — pinned-buffer H2D staging (same output contract);
      * ``dbuf``/``update`` — pop a double-buffered build for THIS update
        if the background thread finished it (the LRU put stays on THIS
        thread; the background thread never touches the LRU);
      * ``dataset_gate`` — lock serialising dataset/geometry access with
        the double-buffer thread (only taken when the dbuf is active).
    """
    pos = [int(p) for p in positions]
    if dbuf is not None and update is not None:
        popped = dbuf.pop(int(update))
        if popped is not None and popped["positions"] == pos:
            for p, rec, built in zip(pos, popped["records"], popped["built"]):
                lru.put_built(int(p), rec, built)
    if dataset_gate is not None:
        with dataset_gate:
            records = dataset.records(pos)
            _prefetch_fill(lru, dataset, pos, prefetcher)
            built = [lru.get(int(p)) for p in pos]
    else:
        records = dataset.records(pos)
        _prefetch_fill(lru, dataset, pos, prefetcher)
        built = [lru.get(int(p)) for p in pos]
    collated = collate_v3(built)
    if h2d_pool is not None:
        batch = to_torch_batch_pinned(collated, device, h2d_pool)
    else:
        batch = to_torch_batch(collated, device=device)
    return batch, records


# ---------------------------------------------------------------------------
# loss / optimizer
# ---------------------------------------------------------------------------


def _make_optimizer(model: torch.nn.Module, config: V53MPTrainerConfig):
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone.")]
    groups = []
    if head_params and config.arm != "A":
        groups.append({"params": head_params, "lr": config.lr_head})
    groups.append({"params": list(model.backbone.parameters()),
                   "lr": config.lr_encoder})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _loss_on_batch(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    records: Sequence[Any],
    labels_v52: Any,
    endpoint_labels: Any,
    config: V53MPTrainerConfig,
    *,
    endpoint_pos_weight: float,
    pair_rng: Optional[np.random.RandomState],
) -> Dict[str, Any]:
    """One forward + the frozen V5.3-MP loss (contract §7)."""
    from pes2ts_core.pesn.models.key_atom_loss_v53mp_v3 import (
        key_atom_loss_v53mp,
    )
    from pes2ts_core.pesn.models.key_atom_losses_v3 import (
        KeyAtomLossV3Config,
    )

    out = model(forward_view_v3(batch))
    sample_ids = [str(s) for s in batch["sample_ids"]]
    lab = _labels_for_batch(sample_ids, labels_v52, config.device)
    frame_mask = _frame_mask_for(records, config.device)
    order_pairs = None
    if config.train_rank:
        order_pairs = sample_order_pairs(
            batch, labels_v52,
            order_min_gap=config.order_min_gap,
            max_pairs=config.max_order_pairs_per_batch,
            rng=pair_rng if pair_rng is not None
            else np.random.RandomState(0))
    y_endpoint = endpoint_mask = None
    if config.train_endpoint:
        ep = _endpoint_for_batch(sample_ids, endpoint_labels, config.device)
        y_endpoint, endpoint_mask = ep["y_endpoint"], ep["endpoint_mask"]
    return key_atom_loss_v53mp(
        out["u"], batch["y_teacher"],
        batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
        _key_ptr(batch), batch[NODE_POS_KEY], batch[NODE_KEY_SET_KEY],
        batch["pair_index"],
        q_pred=out["q"] if config.train_q else None,
        q_true=lab["q_true"] if config.train_q else None,
        q_mask=lab["q_mask"] if config.train_q else None,
        order_pairs=order_pairs,
        endpoint_logit=out["endpoint_logit"] if config.train_endpoint else None,
        y_endpoint=y_endpoint,
        endpoint_mask=endpoint_mask,
        endpoint_pos_weight=endpoint_pos_weight,
        lambda_rank=config.lambda_rank,
        lambda_q=config.lambda_q,
        lambda_endpoint=config.lambda_endpoint,
        rank_margin=config.rank_margin,
        config=KeyAtomLossV3Config(),
        lambda_cos=config.lambda_cos_fixed,
        frame_mask=frame_mask,
        frame_weight=batch["frame_weight"],
        pair_edit=batch["pair_edit"],
    )


def _inner_val_loss(
    model: torch.nn.Module,
    dataset: Any,
    lru: _FeatureLRU,
    positions: Sequence[int],
    labels_v52: Any,
    endpoint_labels: Any,
    config: V53MPTrainerConfig,
    *,
    endpoint_pos_weight: float,
    update: int,
    batch_size: int = 64,
    prefetcher: Optional[FeaturePrefetcher] = None,
    dataset_gate: Optional[Any] = None,
) -> float:
    """Weighted mean inner-val loss (same loss; selection uses this ONLY)."""
    model.train(False)
    total = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = list(positions[start:start + batch_size])
            batch, records = _batch_from_positions(
                dataset, lru, chunk, config.device, prefetcher=prefetcher,
                dataset_gate=dataset_gate)
            loss_out = _loss_on_batch(
                model, batch, records, labels_v52, endpoint_labels, config,
                endpoint_pos_weight=endpoint_pos_weight,
                pair_rng=_update_rng(config.seed, update, salt=2))
            w = float(batch["frame_weight"].sum().cpu())
            total += float(loss_out["loss"].cpu()) * w
            weight_sum += w
    model.train(True)
    return total / max(weight_sum, 1e-9)


# ---------------------------------------------------------------------------
# gradient monitoring (plan §6: λ 用 encoder 梯度占比校准)
# ---------------------------------------------------------------------------


def _grad_monitor_row(
    loss_out: Mapping[str, Any],
    encoder_params: Sequence[torch.nn.Parameter],
    config: V53MPTrainerConfig,
) -> Dict[str, Any]:
    """Per-term encoder grad norms, pairwise cosines, encoder_share.

    Only called on monitor steps (guard cost).  ``encoder_share`` uses the
    λ-WEIGHTED process gradient (the term's actual encoder contribution),
    divided by the move gradient norm.
    """
    active: Dict[str, torch.Tensor] = {"move": loss_out["l_move"]}
    lambdas: Dict[str, float] = {"move": 1.0}
    if float(config.lambda_rank) != 0.0:
        active["rank"] = loss_out["l_rank"]
        lambdas["rank"] = float(config.lambda_rank)
    if float(config.lambda_q) != 0.0:
        active["q"] = loss_out["l_q"]
        lambdas["q"] = float(config.lambda_q)
    if float(config.lambda_endpoint) != 0.0:
        active["endpoint"] = loss_out["l_endpoint"]
        lambdas["endpoint"] = float(config.lambda_endpoint)
    row: Dict[str, Any] = {
        "lambdas": dict(lambdas),
        "grad_norms": {},
        "cosines": {},
        "encoder_share": None,
        "encoder_frozen": False,
    }
    params = [p for p in encoder_params if p.requires_grad]
    if not params:
        row["encoder_frozen"] = True
        return row
    grads: Dict[str, torch.Tensor] = {}
    for name, term in active.items():
        if not (torch.is_tensor(term) and bool(term.requires_grad)):
            row["grad_norms"][name] = 0.0
            continue
        g_list = torch.autograd.grad(
            term, params, retain_graph=True, allow_unused=True)
        vec = torch.cat([
            torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
            for p, g in zip(params, g_list)])
        grads[name] = vec
        row["grad_norms"][name] = float(vec.norm())
    names = sorted(grads)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            va, vb = grads[names[a]], grads[names[b]]
            denom = float(va.norm()) * float(vb.norm())
            row["cosines"]["{}_{}".format(names[a], names[b])] = (
                None if denom == 0.0 else float((va @ vb) / denom))
    process_total = None
    for term in ("rank", "q", "endpoint"):
        if term in grads:
            weighted = grads[term] * lambdas[term]
            process_total = (weighted if process_total is None
                             else process_total + weighted)
    if "move" in grads:
        move_norm = float(grads["move"].norm())
        row["encoder_share"] = (
            0.0 if process_total is None
            else float(process_total.norm()) / (move_norm + 1e-12))
    return row


# ---------------------------------------------------------------------------
# endpoint pos_weight (inner-train ONLY, then frozen — T3.3 style)
# ---------------------------------------------------------------------------


def estimate_endpoint_pos_weight(
    samples: Sequence[Any],
    endpoint_labels: Any,
    train_positions: Sequence[int],
    *,
    seed: int,
) -> Dict[str, Any]:
    """n_neg / n_pos over UNMASKED endpoint labels of the inner-train rows.

    Estimated ONCE at run start from the inner-train slice only — dev rows
    are never passed in, so they cannot flow back (structural guarantee).
    Zero positives ⇒ pos_weight 1.0 (flagged via n_pos=0; never fabricate).
    """
    ep_col = getattr(endpoint_labels, "ep_label_col", None)
    st = _startup_of(samples)
    if st is not None and ep_col is not None \
            and len(train_positions) > 0 \
            and endpoint_labels.covers_all_positions(st.n):
        # I4 fast path (vectorized): the lazy endpoint table is written in
        # SAMPLE ORDER, so gathering ep_label at the train positions counts
        # the SAME labels.  Probe row-order identity on the boundaries
        # (real by_sample lookups); any miss falls through per-item.
        pos = np.asarray(train_positions, dtype=np.int64)
        probe_ok = True
        for p in (int(pos[0]), int(pos[-1])):
            sid = str(samples[int(p)].sample_id)
            if int(ep_col[p]) != int(endpoint_labels.by_sample[sid]
                                      .endpoint_label):
                probe_ok = False
                break
        if probe_ok:
            labels = np.asarray(ep_col)[pos]
            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())
            pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
            return {
                "pos_weight": float(pos_weight),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "source": ("inner-train rows of the v53mp pool split "
                           "(seed={}); never recomputed from dev".format(
                               seed)),
            }
    n_pos = 0
    n_neg = 0
    for p in train_positions:
        rec = samples[int(p)]
        lab = endpoint_labels.by_sample[rec.sample_id]
        label = int(lab.endpoint_label)
        if label == 1:
            n_pos += 1
        elif label == 0:
            n_neg += 1
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    return {
        "pos_weight": float(pos_weight),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "source": ("inner-train rows of the v53mp pool split "
                   "(seed={}); never recomputed from dev".format(seed)),
    }


def _train_q_mean(samples: Any, labels_v52: Any,
                  train_positions: Sequence[int]) -> Optional[float]:
    """Mean q over q-masked inner-train rows (constant-q baseline).

    Per-item path (VERBATIM from the original inline block): two by_sample
    lookups per position.  Columnar fast path (I4): the lazy v52 table is
    written in SAMPLE ORDER, so gathering ``q``/``q_mask`` at the train
    positions yields the SAME float64 subsequence in the same order —
    ``np.mean`` over it is bitwise-identical to the original
    ``float(np.mean(list))`` (numpy converts the list to a float64 array
    and pairwise-sums; same values, same order).  Probe row-order identity
    on the boundaries; any miss falls through per-item.
    """
    q_col = getattr(labels_v52, "q_col", None)
    q_mask_col = getattr(labels_v52, "q_mask_col", None)
    st = _startup_of(samples)
    if st is not None and q_col is not None and q_mask_col is not None \
            and len(train_positions) > 0 \
            and labels_v52.covers_all_positions(st.n):
        pos = np.asarray(train_positions, dtype=np.int64)
        probe_ok = True
        for p in (int(pos[0]), int(pos[-1])):
            sid = str(samples[int(p)].sample_id)
            lab = labels_v52.by_sample[sid]
            if bool(q_mask_col[p]) != bool(lab.q_mask) or \
                    (bool(lab.q_mask) and
                     float(q_col[p]) != float(lab.q)):
                probe_ok = False
                break
        if probe_ok:
            q = np.asarray(q_col)[pos].astype(np.float64)
            m = np.asarray(q_mask_col)[pos]
            vals = q[m]
            return float(np.mean(vals)) if vals.size else None
    train_q_vals = [
        float(labels_v52.by_sample[str(samples[p].sample_id)].q)
        for p in train_positions
        if labels_v52.by_sample[str(samples[p].sample_id)].q_mask]
    return float(np.mean(train_q_vals)) if train_q_vals else None


# ---------------------------------------------------------------------------
# math helpers (metrics)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """AUPRC via average precision (ties: mergesort by descending score)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    if n_pos == 0 or y.size == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    precision_at = tp / np.arange(1, ys.size + 1, dtype=np.float64)
    return float(precision_at[ys == 1].mean())


def ordering_pair_accuracy(
    q_pred: Sequence[float],
    q_true: Sequence[float],
    groups: Sequence[Tuple[str, str]],
    *,
    gap: float = 0.0,
    cross_ts_only: bool = False,
) -> Dict[str, Any]:
    """Pairwise ordering accuracy within (curve, direction) groups.

    A pair (i, j) with ``q_true[j] > q_true[i] + gap`` is correct when
    ``q_pred[j] > q_pred[i]`` (ties count 0.5).  ``cross_ts_only`` keeps
    only pairs with strictly opposite q sides (one node before TS, one
    after — the 跨 TS subset of contract §8).
    """
    qp = np.asarray(q_pred, dtype=np.float64)
    qt = np.asarray(q_true, dtype=np.float64)
    by_group: Dict[Tuple[str, str], List[int]] = {}
    for g, key in enumerate(groups):
        by_group.setdefault((str(key[0]), str(key[1])), []).append(g)
    correct = 0.0
    n_pairs = 0
    for idxs in by_group.values():
        for a, i in enumerate(idxs):
            for j in idxs[a + 1:]:
                qi, qj = qt[i], qt[j]
                if qj > qi + gap:
                    earlier, later = i, j
                elif qi > qj + gap:
                    earlier, later = j, i
                else:
                    continue
                if cross_ts_only and not (qt[earlier] * qt[later] < 0.0):
                    continue
                if qp[later] > qp[earlier]:
                    correct += 1.0
                elif qp[later] == qp[earlier]:
                    correct += 0.5
                n_pairs += 1
    return {
        "accuracy": (correct / n_pairs) if n_pairs else None,
        "n_pairs": int(n_pairs),
    }


def calibrate_tau_end(
    logits: Sequence[float],
    y_true: Sequence[int],
    mask: Sequence[bool],
    *,
    target_precision: float,
) -> Dict[str, Any]:
    """Calibrate and freeze τ_end on INNER-VAL rows (contract §8/§9).

    Sweeps ``sigmoid(endpoint_logit)`` thresholds and picks the SMALLEST τ
    achieving ``precision >= target_precision``; fallback: the τ at max F1.
    Never sees dev/confirm rows (the caller passes inner-val only).
    """
    log = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    m = np.asarray(mask, dtype=np.bool_)
    keep = m & np.isin(y, (0, 1))
    n = int(keep.sum())
    out: Dict[str, Any] = {
        "tau_end": None,
        "target_precision": float(target_precision),
        "achieved_precision": None,
        "achieved_recall": None,
        "prevalence": None,
        "n_inner_val_unmasked": n,
    }
    if n == 0:
        return out
    scores = _sigmoid(log[keep])
    yy = y[keep]
    out["prevalence"] = float(yy.mean())
    chosen: Optional[float] = None
    best_f1 = -1.0
    best_f1_tau: Optional[float] = None
    for tau in np.unique(scores):
        pred = scores >= tau
        tp = int((pred & (yy == 1)).sum())
        fp = int((pred & (yy == 0)).sum())
        fn = int(((~pred) & (yy == 1)).sum())
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        if f1 > best_f1 or (f1 == best_f1 and
                            (best_f1_tau is None or float(tau) < best_f1_tau)):
            best_f1 = f1
            best_f1_tau = float(tau)
        if chosen is None and precision >= float(target_precision):
            chosen = float(tau)
            out["achieved_precision"] = float(precision)
            out["achieved_recall"] = float(recall)
    if chosen is None and best_f1_tau is not None:
        chosen = best_f1_tau
        pred = scores >= chosen
        tp = int((pred & (yy == 1)).sum())
        fp = int((pred & (yy == 0)).sum())
        fn = int(((~pred) & (yy == 1)).sum())
        out["achieved_precision"] = (tp / (tp + fp)) if (tp + fp) else None
        out["achieved_recall"] = (tp / (tp + fn)) if (tp + fn) else None
        out["fallback"] = "max_f1"
    else:
        out["fallback"] = None
    out["tau_end"] = chosen
    return out


# ---------------------------------------------------------------------------
# evaluation (contract §8 table; report separately, no composite score)
# ---------------------------------------------------------------------------


def evaluate_v53mp(
    model: torch.nn.Module,
    r3_model: Optional[torch.nn.Module],
    dataset: Any,
    labels_v52: Any,
    endpoint_labels: Any,
    cache: Any,
    positions: Sequence[int],
    config: V53MPTrainerConfig,
    *,
    train_q_mean: Optional[float],
    tau_end: Optional[float],
) -> Dict[str, Any]:
    """Frozen dev metrics: Move guard + q + ordering + endpoint (§8).

    ``cache`` supports ``cache[position] -> built features`` (the trainer
    passes an LRU view).  ``r3_model`` (matched R3 checkpoint, zero-shot)
    enables the ≤2% MOVE guard comparison on the SAME dev positions.
    """
    model.train(False)
    q_preds: List[float] = []
    q_trues: List[float] = []
    q_masks: List[bool] = []
    groups: List[Tuple[str, str]] = []
    ep_logits: List[float] = []
    ep_y: List[int] = []
    ep_mask: List[bool] = []
    ts_rows: List[bool] = []
    start_rows: List[bool] = []
    with torch.no_grad():
        for start in range(0, len(positions), 64):
            chunk = list(positions[start:start + 64])
            built = [cache[int(p)] for p in chunk]
            batch = to_torch_batch(collate_v3(built), device=config.device)
            sample_ids = [str(s) for s in batch["sample_ids"]]
            out = model(forward_view_v3(batch))
            lab = _labels_for_batch(sample_ids, labels_v52, config.device)
            ep = _endpoint_for_batch(sample_ids, endpoint_labels,
                                     config.device)
            q_preds.extend(out["q"].cpu().tolist())
            q_trues.extend(lab["q_true"].cpu().tolist())
            q_masks.extend(lab["q_mask"].cpu().tolist())
            for sid in sample_ids:
                row = labels_v52.by_sample[sid]
                groups.append((str(row.curve_id), str(row.direction)))
            ep_logits.extend(out["endpoint_logit"].cpu().tolist())
            ep_y.extend(
                (ep["y_endpoint"] * ep["endpoint_mask"].float()).cpu()
                .tolist())
            ep_mask.extend(ep["endpoint_mask"].cpu().tolist())
            ts_rows.extend(ep["ts_row"])
            start_rows.extend(ep["start_row"])
    model.train(True)

    qp = np.asarray(q_preds, dtype=np.float64)
    qt = np.asarray(q_trues, dtype=np.float64)
    qm = np.asarray(q_masks, dtype=np.bool_)
    per_curve: Dict[str, List[int]] = {}
    for i, (cid, _direction) in enumerate(groups):
        per_curve.setdefault(cid, []).append(i)
    q_curve_mae: List[float] = []
    q_curve_corr: List[float] = []
    for cid, idxs in sorted(per_curve.items()):
        ok = [i for i in idxs if qm[i]]
        if not ok:
            continue
        q_curve_mae.append(float(np.abs(qp[ok] - qt[ok]).mean()))
        corr = pearson(qp[ok], qt[ok])
        if corr is not None:
            q_curve_corr.append(corr)
    const_mae = (float(np.abs(float(train_q_mean) - qt[qm]).mean())
                 if (qm.any() and train_q_mean is not None) else None)
    q_head = {
        "mae_curve_median": (float(np.median(q_curve_mae))
                             if q_curve_mae else None),
        "mae_curve_mean": (float(np.mean(q_curve_mae))
                           if q_curve_mae else None),
        "corr_curve_median": (float(np.median(q_curve_corr))
                              if q_curve_corr else None),
        "corr_curve_mean": (float(np.mean(q_curve_corr))
                            if q_curve_corr else None),
        "n_eval_curves": len(q_curve_mae),
        "constant_q_baseline_mae": const_mae,
        "constant_q_value": train_q_mean,
        "learnable_above_constant": (
            None if (const_mae is None or not q_curve_mae)
            else float(np.median(q_curve_mae)) < const_mae),
    }

    valid_idx = [i for i in range(len(qp)) if qm[i]]
    full = ordering_pair_accuracy(
        [qp[i] for i in valid_idx], [qt[i] for i in valid_idx],
        [groups[i] for i in valid_idx], gap=0.0, cross_ts_only=False)
    cross = ordering_pair_accuracy(
        [qp[i] for i in valid_idx], [qt[i] for i in valid_idx],
        [groups[i] for i in valid_idx], gap=0.0, cross_ts_only=True)
    ordering = {
        "pairwise_accuracy_full": full["accuracy"],
        "n_pairs_full": full["n_pairs"],
        "pairwise_accuracy_cross_ts": cross["accuracy"],
        "n_pairs_cross_ts": cross["n_pairs"],
        "source": "q_pred pairs within (curve, intent direction) dev groups",
    }

    log = np.asarray(ep_logits, dtype=np.float64)
    y = np.asarray(ep_y, dtype=np.int64)
    m = np.asarray(ep_mask, dtype=np.bool_)
    keep = m & np.isin(y, (0, 1))
    scores = _sigmoid(log[keep])
    labels_keep = y[keep]
    auroc = auc_mann_whitney(scores.tolist(), labels_keep.tolist())
    auprc = average_precision(scores.tolist(), labels_keep.tolist())
    mis_start = None
    mis_ts = None
    n_start = int(np.sum(keep & np.asarray(start_rows, dtype=np.bool_)))
    n_ts = int(np.sum(keep & np.asarray(ts_rows, dtype=np.bool_)))
    if tau_end is not None:
        pred_stop = _sigmoid(log) >= float(tau_end)
        start_arr = np.asarray(start_rows, dtype=np.bool_)
        ts_arr = np.asarray(ts_rows, dtype=np.bool_)
        if n_start:
            mis_start = float(pred_stop[keep & start_arr].mean())
        if n_ts:
            mis_ts = float(pred_stop[keep & ts_arr].mean())
    endpoint = {
        "auroc": auroc,
        "auprc": auprc,
        "prevalence": (float(labels_keep.mean())
                       if int(labels_keep.size) else None),
        "n_unmasked": int(keep.sum()),
        "n_pos": int((labels_keep == 1).sum()),
        "n_neg": int((labels_keep == 0).sum()),
        "tau_end": tau_end,
        "start_misstop_rate_at_tau": mis_start,
        "ts_misstop_rate_at_tau": mis_ts,
        "n_start_rows": n_start,
        "n_ts_rows": n_ts,
    }

    # MOVE metrics are teacher-defined: only ``ok`` rows carry a teacher
    # target (terminal/masked/conflict rows are zero-padded by the feature
    # LRU so batches never crash, but they must not enter the MOVE metrics).
    move_positions = [
        int(p) for p in positions
        if str(getattr(dataset.records([int(p)])[0], "label_status", "ok"))
        == "ok"]
    move = evaluate_checkpoint(
        model, dataset, cache, move_positions, device=config.device)
    move_baseline = None
    if r3_model is not None:
        move_baseline = evaluate_checkpoint(
            r3_model, dataset, cache, move_positions, device=config.device)
    guard: Dict[str, Any] = {
        "policy": ("record + guard (<= {}% RMSE degradation vs matched R3 "
                   "zero-shot; breach = scientific_failure, run KEPT)"
                   .format(config.move_guard_max_degradation_pct)),
        "max_degradation_pct": config.move_guard_max_degradation_pct,
        "baseline": "r3_eval_ckpt zero-shot on the same dev positions",
    }
    run_outcome = "ok"
    if move_baseline is not None:
        m_rmse = move["key_rmse_over_zero"].get("median_rmse")
        b_rmse = move_baseline["key_rmse_over_zero"].get("median_rmse")
        deg = (None if (m_rmse is None or not b_rmse)
               else 100.0 * (float(m_rmse) - float(b_rmse)) / float(b_rmse))
        within = (deg is not None and
                  deg <= config.move_guard_max_degradation_pct)
        if deg is not None and not within:
            run_outcome = "scientific_failure"   # never "invalid" (§7)
        guard.update({
            "model_median_rmse": m_rmse,
            "r3_baseline_median_rmse": b_rmse,
            "rmse_degradation_pct": deg,
            "within_rmse_guard": within if deg is not None else None,
            "model_rmse_over_zero": move["key_rmse_over_zero"].get(
                "median_rmse_over_zero"),
            "baseline_rmse_over_zero": move_baseline["key_rmse_over_zero"].get(
                "median_rmse_over_zero"),
            "model_sign_median": move["sre_and_sign"].get(
                "sign_accuracy_cluster_median"),
            "baseline_sign_median": move_baseline["sre_and_sign"].get(
                "sign_accuracy_cluster_median"),
            "model_dead_zone_fraction": move["sre_and_sign"].get(
                "dead_zone_fraction"),
            "baseline_dead_zone_fraction": move_baseline["sre_and_sign"].get(
                "dead_zone_fraction"),
        })

    return {
        "schema": V53MP_SCHEMA,
        "move": move,
        "move_guard_vs_r3": guard,
        "q_head": q_head,
        "ordering": ordering,
        "endpoint_head": endpoint,
        "run_outcome": run_outcome,
        "n_dev_samples": len(positions),
        "n_dev_curves": len({cid for cid, _ in groups}),
    }


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------


def _save(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(path)


def _ckpt_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    best_val: float,
    best_update: int,
    config: V53MPTrainerConfig,
    seed_map: Mapping[str, int],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": V53MP_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "best_val": float(best_val),
        "best_update": int(best_update),
        "config": asdict(config),
        "seeds": dict(seed_map),
        "provenance": dict(provenance),
    }


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def train_run_v53mp(
    dataset: Any,
    labels_v52: Any,
    endpoint_labels: Any,
    config: V53MPTrainerConfig,
    out_dir: Any,
    *,
    r3_ckpt: Any = None,
    r3_eval_ckpt: Any = None,
    dev_positions: Optional[Sequence[int]] = None,
    train_positions: Optional[Sequence[int]] = None,
    resume: bool = True,
    provenance: Optional[Mapping[str, Any]] = None,
    dev_feature_pack: Any = None,
    double_buffer: bool = False,
    pinned_h2d: bool = False,
    feature_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Train one V5.3-MP run; writes checkpoint{_{best,last}}.pt + metrics.

    ``train_positions`` is the FULL inner-train pool (the union returned
    by :func:`v53mp_pool_split` for the current scale tier) — the trainer
    re-derives the identical curve-level split; passing a curve-INCOMPLETE
    set is a typed refusal.  When omitted, the pool defaults to every
    curve NOT in ``dev_positions``.  ``dev_positions`` is the frozen dev
    subset of the split manifest (driver-owned).  Confirm-subset curves in
    ANY position set are a typed refusal.  Draws are pure functions of
    (seed, update), so ``resume=True`` continues bit-identically.

    ``dev_feature_pack`` (S4 plan §5.3 / lane I2): an optional pack
    directory (or an open :class:`DevFeaturePackView`) holding the frozen
    dev F2 features.  When provided, the final dev eval (model pass AND
    R3-baseline pass) reads features lazily from the pack instead of
    building + pinning them in the dev LRU — numerics are bit-identical
    (the pack stores the FINALIZED ``build_sample_features_v3`` dicts,
    verified by ``r3_v53mp_dev_pack_certificate.py``); when None the code
    path is byte-identical to the pre-pack behaviour.

    I3 (plan §4.3/§6 / lanes B3): ``double_buffer`` (default False — B0–B2
    reproducibility) prefetches update k+1's feature build on a background
    thread while the main thread runs forward on batch k
    (:class:`DoubleBufferedPrefetcher`; fail-open, pure ``(seed, update)``
    sampling so numerics are unchanged); ``pinned_h2d`` (default False)
    stages H2D copies through reusable pinned buffers
    (:func:`to_torch_batch_pinned`; same output contract, fail-open);
    ``feature_workers`` (default None = flag-file gate) overrides the fork
    pool size (driver ``--feature-workers``; the B4 formula
    :func:`workers_for_concurrency`).  All three are recorded in metrics.

    Feature building may be fork-parallelised (opt-in flag file
    ``outputs/pesn/v53mp/reports/feature_workers.on``, default OFF): the
    pool is created HERE — before ``load_model_v53mp`` can initialise CUDA
    — and closed in a ``finally``.  Numerics are bit-identical either way
    (same ``build_sample_features_v3``, same arguments).
    """
    # --- optional feature prefetch pool (gate read at trainer start; fork
    # --- MUST precede any CUDA initialisation, hence the very top) --------
    feature_prefetcher = maybe_create_prefetcher(
        FEATURE_LEVEL_V53MP, False, workers=feature_workers)
    feature_workers = (int(feature_prefetcher.workers)
                       if feature_prefetcher is not None else 0)
    try:
        return _train_run_v53mp_impl(
            dataset, labels_v52, endpoint_labels, config, out_dir,
            r3_ckpt=r3_ckpt, r3_eval_ckpt=r3_eval_ckpt,
            dev_positions=dev_positions, train_positions=train_positions,
            resume=resume, provenance=provenance,
            feature_prefetcher=feature_prefetcher,
            feature_workers=feature_workers,
            dev_feature_pack=dev_feature_pack,
            double_buffer=double_buffer,
            pinned_h2d=pinned_h2d)
    finally:
        if feature_prefetcher is not None:
            feature_prefetcher.close()


def _train_run_v53mp_impl(
    dataset: Any,
    labels_v52: Any,
    endpoint_labels: Any,
    config: V53MPTrainerConfig,
    out_dir: Any,
    *,
    r3_ckpt: Any = None,
    r3_eval_ckpt: Any = None,
    dev_positions: Optional[Sequence[int]] = None,
    train_positions: Optional[Sequence[int]] = None,
    resume: bool = True,
    provenance: Optional[Mapping[str, Any]] = None,
    feature_prefetcher: Optional[FeaturePrefetcher] = None,
    feature_workers: int = 0,
    dev_feature_pack: Any = None,
    double_buffer: bool = False,
    pinned_h2d: bool = False,
) -> Dict[str, Any]:
    """Body of :func:`train_run_v53mp` (pool lifecycle owned by the caller)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov: Dict[str, Any] = {k: None for k in PROVENANCE_PLACEHOLDER_KEYS}
    prov.update(dict(provenance or {}))
    seed_map = set_seed(config.seed)
    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise V53MPTrainerError("CUDA_UNAVAILABLE", "device=cuda but no CUDA")
    # --- CUDA memory suite (cap + peak recording) -------------------------
    # The per-process fraction cap (when set) MUST land before the model
    # load / first real allocator allocation below; peak-memory stats are
    # reset right after CUDA init so the reported peaks cover exactly this
    # run.  On CPU devices both are explicit no-ops (tested).
    apply_cuda_memory_fraction(config)
    _reset_peak_memory_stats(device)

    # --- position sets (fail-closed BEFORE any model/loss lane import) -----
    samples = dataset.samples
    n_samples = len(samples)
    _require(dev_positions is not None and len(list(dev_positions)) > 0,
             "DEV_POSITIONS",
             "dev_positions (frozen dev subset) must be non-empty")
    dev_pos = [int(p) for p in dev_positions]
    for p in dev_pos:
        _require(0 <= p < n_samples, "POSITION_RANGE",
                 "position {} outside 0..{}".format(p, n_samples - 1))
    dev_curve_set = _curve_id_set_of(samples, dev_pos)
    if train_positions is None:
        # default pool = every curve NOT in the frozen dev subset
        st = _startup_of(samples)
        if st is not None:
            dev_rows = np.asarray(
                sorted(st.curve_row[c] for c in dev_curve_set
                       if c in st.curve_row), dtype=np.int64)
            pool = np.flatnonzero(
                ~np.isin(np.asarray(st.curve_idx), dev_rows)
            ).astype(np.int64).tolist()
        else:
            pool = [i for i in range(n_samples)
                    if str(samples[i].curve_id) not in dev_curve_set]
    else:
        pool = [int(p) for p in train_positions]
    for p in pool:
        _require(0 <= p < n_samples, "POSITION_RANGE",
                 "position {} outside 0..{}".format(p, n_samples - 1))
    pool_curves = _curve_id_set_of(samples, pool)
    train_pos, inner_val_pos = v53mp_pool_split(
        samples, pool_curve_ids=sorted(pool_curves), seed=config.seed,
        inner_val_fraction=config.inner_val_fraction)
    if train_positions is not None and \
            set(pool) != set(train_pos) | set(inner_val_pos):
        raise V53MPTrainerError(
            "TRAIN_POOL_NOT_CURVE_COMPLETE",
            "train_positions must be the FULL pool (train + inner-val "
            "curves) returned by v53mp_pool_split; got {} positions, the "
            "curve-level split covers {}".format(
                len(pool), len(set(train_pos) | set(inner_val_pos))))
    _assert_no_confirm(samples, train_pos, "train")
    _assert_no_confirm(samples, inner_val_pos, "inner-val")
    _assert_no_confirm(samples, dev_pos, "dev")

    # --- inner-val cap (seeded, FIXED for the run): selection + τ_end use
    # --- this subset so the per-100-update sweep stays cheap at 5k/20k ------
    if config.inner_val_max_samples is not None and \
            len(inner_val_pos) > int(config.inner_val_max_samples):
        _rng_iv = np.random.RandomState(int(config.seed) ^ 0x1E7A1)
        inner_val_pos = sorted(_rng_iv.choice(
            list(inner_val_pos), size=int(config.inner_val_max_samples),
            replace=False).tolist())
    # --- sampler index: built ONCE over the finalised train pool -----------
    curve_index = _CurveDirIndex.build(samples, train_pos)
    train_curves = _curve_id_set_of(samples, train_pos)
    val_curves = _curve_id_set_of(samples, inner_val_pos)
    dev_curves = _curve_id_set_of(samples, dev_pos)
    _require(not (dev_curves & (train_curves | val_curves)),
             "DEV_TRAIN_OVERLAP",
             "dev curves overlap the train pool: {}".format(
                 sorted(dev_curves & (train_curves | val_curves))))

    # --- label coverage (fail-closed on the FULL used slice) --------------
    needed = [train_pos, inner_val_pos, dev_pos]
    _assert_label_coverage(samples, labels_v52, endpoint_labels,
                           needed, n_samples)

    # --- endpoint pos_weight: ONCE, inner-train only (T3.3) ---------------
    ep_weights = estimate_endpoint_pos_weight(
        samples, endpoint_labels, train_pos, seed=config.seed)
    endpoint_pos_weight = float(ep_weights["pos_weight"])

    # --- model (seeded AFTER the deterministic setup, V5.2 discipline) ----
    set_seed(config.seed)     # re-seed so head init is seed-exact
    model = load_model_v53mp(config, r3_ckpt=r3_ckpt, device=device)
    _set_process_heads_trainable(model, config.arm != "A")
    _set_encoder_trainable(model, False)          # stage A: encoder frozen
    optimizer = _make_optimizer(model, config)

    start_update = 1
    best_val = float("inf")
    best_update = 0
    last_path = out / "checkpoint_last.pt"
    best_path = out / "checkpoint_best.pt"
    if resume and last_path.is_file():
        state = torch.load(str(last_path), map_location=device,
                           weights_only=False)
        cfg0 = state.get("config", {})
        prov0 = state.get("provenance", {})
        if (cfg0.get("arm") != config.arm
                or int(cfg0.get("seed", -1)) != int(config.seed)
                or cfg0.get("backbone_init") != config.backbone_init
                or prov0.get("tier") != prov.get("tier")
                or float(cfg0.get("lambda_rank", -1.0)) != float(config.lambda_rank)
                or float(cfg0.get("lambda_q", -1.0)) != float(config.lambda_q)
                or float(cfg0.get("lambda_endpoint", -1.0)) != float(config.lambda_endpoint)):
            raise V53MPTrainerError(
                "RESUME_IDENTITY_MISMATCH",
                "checkpoint_last belongs to {} seed{} init={} tier={} "
                "λ=({},{},{}); refusing".format(
                    cfg0.get("arm"), cfg0.get("seed"),
                    cfg0.get("backbone_init"), prov0.get("tier"),
                    cfg0.get("lambda_rank"), cfg0.get("lambda_q"),
                    cfg0.get("lambda_endpoint")))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_update = int(state["update"]) + 1
        best_val = float(state["best_val"])
        best_update = int(state["best_update"])
    # stage state is a pure function of start_update (resume-identical)
    _set_encoder_trainable(
        model, start_update > config.effective_freeze_updates)

    lru = _FeatureLRU(
        dataset, capacity=int(config.feature_cache_capacity),
        level=FEATURE_LEVEL_V53MP, intent_off=False)
    # Inner-val gets its OWN cache: the val slice is FIXED for the run, so
    # pinning it (capacity >= len + margin) means each val feature is built
    # exactly once.  Sharing the training LRU would let the 3200 per-eval
    # training inserts evict val entries, forcing ~3200-5000 rebuilds on
    # every eval (~8 min of a 3750-update 5k run).  Features are pure
    # functions of the record, so caching choices cannot change numerics.
    val_lru = _FeatureLRU(
        dataset, capacity=max(len(inner_val_pos), 1) + 64,
        level=FEATURE_LEVEL_V53MP, intent_off=False)
    # Final dev eval gets its OWN pinned cache, shared by BOTH passes inside
    # evaluate_v53mp (the model, then the matched R3 baseline): the frozen
    # dev set (~138k positions) dwarfs the 8192-entry training LRU, so
    # reusing it made the second evaluate_checkpoint pass rebuild every dev
    # feature (~8 min of wasted compute per run).  Capacity is
    # len(dev_positions) + 64 so each dev feature is built exactly once and
    # the baseline pass is a pure hit.  MEMORY GUARD: a pinned full-dev entry
    # set costs ≈0.6–0.7 GB at the 138k tier; only pin while
    # len(dev_positions) <= 200_000, otherwise fall back to the training LRU
    # (thrashing, but bounded memory).  The training loop keeps ``lru`` and
    # the inner-val/tau path keeps ``val_lru``.  Features are pure functions
    # of the record, so caching choices cannot change numerics.
    # I2 (plan §5.3): when a dev feature pack is provided, BOTH eval passes
    # read the frozen features lazily from it — no dev LRU, no warm build,
    # no ~4.7 GB pin.  Verified fail-closed BEFORE the training loop so a
    # stale pack never burns a run.
    dev_pack_view = None
    if dev_feature_pack is not None:
        from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
            DevFeaturePackView,
        )
        if isinstance(dev_feature_pack, DevFeaturePackView):
            dev_pack_view = dev_feature_pack
        else:
            dev_pack_view = DevFeaturePackView.open(
                Path(dev_feature_pack),
                expected_positions=dev_pos,
                feature_level=FEATURE_LEVEL_V53MP)
    dev_lru: Optional[_FeatureLRU] = None
    if dev_pack_view is None and len(dev_pos) <= 200_000:
        dev_lru = _FeatureLRU(
            dataset, capacity=len(dev_pos) + 64,
            level=FEATURE_LEVEL_V53MP, intent_off=False)

    log_path = out / "train_log.jsonl"
    effective = config.total_updates
    # I3 (plan §4.3): optional double buffer — a background thread prefetches
    # update k+1's build while the main thread runs forward on batch k.  The
    # (seed, update) sampler is pure, so the thread's positions are the SAME
    # the main loop will draw; the LRU put stays on the main thread; a shared
    # gate serialises the (cheap) dataset/geometry access.  Any thread error
    # disables the buffer (fail-open → synchronous path, numerics unchanged).
    dbuf = None
    dataset_gate = None
    h2d_pool = PinnedBufferPool() if pinned_h2d else None
    if double_buffer and feature_prefetcher is not None:
        dataset_gate = threading.Lock()

        def _dbuf_compute(u: int) -> List[int]:
            return _curve_batch_positions(
                samples, train_pos, config=config,
                rng=_update_rng(config.seed, int(u), salt=0),
                index=curve_index)

        def _dbuf_materialize(pos: Sequence[int]) -> Dict[str, Any]:
            with dataset_gate:
                recs = dataset.records([int(p) for p in pos])
                return {"records": recs,
                        "samples": [dataset.to_sample_v3(r) for r in recs]}

        dbuf = DoubleBufferedPrefetcher(
            feature_prefetcher, dataset, _dbuf_compute, _dbuf_materialize,
            level=FEATURE_LEVEL_V53MP, intent_off=False)
    elif double_buffer and feature_prefetcher is None:
        log_prefetch_issue(
            "double_buffer requested but the feature pool is off "
            "(flag file / --feature-workers); running without double buffer")
    t0 = time.time()
    last_loss: Optional[float] = None
    for update in range(start_update, effective + 1):
        if update == config.effective_freeze_updates + 1:
            _set_encoder_trainable(model, True)   # stage B: joint
        positions = _curve_batch_positions(
            samples, train_pos, config=config,
            rng=_update_rng(config.seed, update, salt=0),
            index=curve_index)
        # prefetch update+1 on the background thread BEFORE the forward of
        # this batch (submit is non-blocking; falls back when busy/broken)
        if dbuf is not None and update < effective:
            dbuf.submit(update + 1)
        batch, records = _batch_from_positions(
            dataset, lru, positions, device, prefetcher=feature_prefetcher,
            h2d_pool=h2d_pool, dbuf=dbuf, update=update,
            dataset_gate=dataset_gate)
        optimizer.zero_grad(set_to_none=True)
        loss_out = _loss_on_batch(
            model, batch, records, labels_v52, endpoint_labels, config,
            endpoint_pos_weight=endpoint_pos_weight,
            pair_rng=_update_rng(config.seed, update, salt=1))
        monitor_step = (update % int(config.grad_monitor_every) == 0
                        or update == effective)
        grad_row = None
        if monitor_step:
            grad_row = _grad_monitor_row(
                loss_out, list(model.backbone.parameters()), config)
        if bool(loss_out["loss"].requires_grad):
            loss_out["loss"].backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
        last_loss = float(loss_out["loss"].detach().cpu())
        if not bool(loss_out["loss"].requires_grad):
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "update": update,
                    "skipped_no_grad": (
                        "loss has no grad path (encoder frozen and no "
                        "active process rows in this batch)"),
                }, sort_keys=True) + "\n")
        if update % 100 == 0 or update == 1 or update == effective:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "update": update,
                    "phase": ("freeze" if update <= config.effective_freeze_updates
                              else "joint"),
                    "loss": last_loss,
                    "l_move": float(loss_out["l_move"].detach().cpu()),
                    "l_rank": float(loss_out["l_rank"].detach().cpu()),
                    "l_q": float(loss_out["l_q"].detach().cpu()),
                    "l_endpoint": float(loss_out["l_endpoint"].detach().cpu()),
                    "n_rank_pairs": int(loss_out["n_rank_pairs"]),
                    "n_q_rows": int(loss_out["n_q_rows"]),
                    "n_endpoint_rows": int(loss_out["n_endpoint_rows"]),
                    "encoder_trainable": any(
                        p.requires_grad for p in model.backbone.parameters()),
                    "elapsed_s": round(time.time() - t0, 3),
                }, sort_keys=True) + "\n")
        if grad_row is not None:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"update": update, "grad_monitor": grad_row},
                    sort_keys=True) + "\n")
        if update % max(int(config.val_every), 1) == 0 or update == effective:
            val = _inner_val_loss(
                model, dataset, val_lru, inner_val_pos, labels_v52,
                endpoint_labels, config,
                endpoint_pos_weight=endpoint_pos_weight, update=update,
                prefetcher=feature_prefetcher, dataset_gate=dataset_gate)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"update": update, "inner_val_loss": val},
                    sort_keys=True) + "\n")
            if val < best_val:
                best_val = val
                best_update = update
                _save(best_path, _ckpt_payload(
                    model, optimizer, update, best_val, best_update,
                    config, seed_map, prov))
            _save(last_path, _ckpt_payload(
                model, optimizer, update, best_val, best_update,
                config, seed_map, prov))

    # I3: stop the double-buffer thread before selection/eval (no more
    # updates to prefetch); idempotent, joins the thread.
    dbuf_healthy = bool(dbuf is not None and dbuf.active)
    if dbuf is not None:
        dbuf.close()

    # --- selection: inner-val ONLY ----------------------------------------
    if best_path.is_file():
        state = torch.load(str(best_path), map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])
        selected_update = int(state["update"])
    else:  # val_every > updates edge case: last state is the selection
        selected_update = effective
        _save(best_path, _ckpt_payload(
            model, optimizer, selected_update, best_val, selected_update,
            config, seed_map, prov))

    # --- τ_end calibration on inner-val, then frozen into payload+metrics --
    tau_block = _calibrate_tau_on(
        model, dataset, val_lru, inner_val_pos, labels_v52, endpoint_labels,
        config, endpoint_pos_weight=endpoint_pos_weight)
    for path in (best_path, last_path):
        if path.is_file():
            payload = torch.load(str(path), map_location="cpu",
                                 weights_only=False)
            payload["tau_end_calibration"] = tau_block
            _save(path, payload)
    if best_path.is_file():
        state = torch.load(str(best_path), map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])

    # --- constant-q baseline from the inner-train slice only --------------
    train_q_mean = _train_q_mean(samples, labels_v52, train_pos)

    r3_model = None
    if r3_eval_ckpt is not None:
        payload = torch.load(str(r3_eval_ckpt), map_location="cpu",
                             weights_only=False)
        r3_model = StarEGNNV3()
        r3_model.load_state_dict(payload["model"], strict=True)
        r3_model.to(device)

    # dev-eval cap (S1 smoke only): seeded subsample of the frozen dev
    # positions; S2/S3 keep the full dev set (comparability across tiers).
    dev_pos_eval = list(dev_pos)
    if config.eval_max_samples is not None and \
            len(dev_pos_eval) > int(config.eval_max_samples):
        _rng_eval = np.random.RandomState(config.seed ^ 0xE5A1)
        dev_pos_eval = sorted(_rng_eval.choice(
            dev_pos_eval, size=int(config.eval_max_samples),
            replace=False).tolist())

    # Dedicated dev cache when it fits the memory guard (see above); the
    # fallback keeps the old behaviour: one shared training LRU for both the
    # model and the R3 baseline passes.
    # I2 (plan §5.3): the dev pack serves BOTH passes (model + R3 baseline)
    # through the same ``cache[position]`` protocol — no warm, no pin.
    if dev_pack_view is not None:
        dev_cache_obj: Any = dev_pack_view
    else:
        dev_cache = dev_lru if dev_lru is not None else lru
        # Parallel PRE-WARM of the pinned dev cache (chunks of 256): the eval
        # itself then hits the LRU.  Pure cache-warmup — evaluate_v53mp /
        # evaluate_checkpoint are untouched and build any miss serially, so a
        # degraded/absent pool changes only speed, never numerics.
        if dev_lru is not None:
            _prefetch_warm_lru(dev_lru, dataset, dev_pos_eval,
                               feature_prefetcher, chunk=256)
        dev_cache_obj = _CacheView(dev_cache)
    metrics = evaluate_v53mp(
        model, r3_model, dataset, labels_v52, endpoint_labels,
        dev_cache_obj, dev_pos_eval, config,
        train_q_mean=train_q_mean, tau_end=tau_block.get("tau_end"))
    # GPU memory suite: post-run peaks (bytes) + the fraction actually
    # applied (CPU devices: 0/0/None, never touching torch.cuda — tested).
    _gpu_mem = _gpu_memory_metrics(config)
    metrics.update({
        "arm": config.arm,
        "seed": int(config.seed),
        "backbone_init": config.backbone_init,
        "lambda_rank": float(config.lambda_rank),
        "lambda_q": float(config.lambda_q),
        "lambda_endpoint": float(config.lambda_endpoint),
        "rank_margin": float(config.rank_margin),
        "order_min_gap": float(config.order_min_gap),
        "schedule": {
            "freeze_encoder_updates": int(config.freeze_encoder_updates),
            "freeze_encoder_updates_effective":
                config.effective_freeze_updates,
            "joint_updates": int(config.joint_updates),
            "smoke_updates": (None if config.smoke_updates is None
                              else int(config.smoke_updates)),
            "lr_head": config.lr_head,
            "lr_encoder": config.lr_encoder,
            "optimizer": "AdamW",
            "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip,
            "lambda_cos": config.lambda_cos_fixed,
            "batch_curves": int(config.batch_curves),
            "frames_per_curve": int(config.frames_per_curve),
            "val_every": int(config.val_every),
            "grad_monitor_every": int(config.grad_monitor_every),
            "feature_level": FEATURE_LEVEL_V53MP,
        },
        "endpoint_pos_weight": ep_weights,
        "tau_end_calibration": tau_block,
        "selected_update": selected_update,
        "best_inner_val_loss": (None if best_val == float("inf")
                                else float(best_val)),
        "best_update": int(best_update),
        "last_train_loss": last_loss,
        "n_train_samples": len(train_pos),
        "n_inner_val_samples": len(inner_val_pos),
        "inner_val_max_samples": (None if config.inner_val_max_samples is None
                                  else int(config.inner_val_max_samples)),
        "n_dev_samples_total": len(dev_pos),
        "n_dev_samples_evaluated": len(dev_pos_eval),
        "eval_max_samples": (None if config.eval_max_samples is None
                             else int(config.eval_max_samples)),
        "n_pool_curves": len(pool_curves),
        "train_curves": sorted(train_curves),
        "inner_val_curves": sorted(val_curves),
        "dev_curves": sorted(dev_curves),
        "selection_metric": "inner-val loss (scale-tier pool train curves)",
        "seeds": seed_map,
        "device": device,
        "gpu_max_memory_allocated_bytes":
            _gpu_mem["gpu_max_memory_allocated_bytes"],
        "gpu_max_memory_reserved_bytes":
            _gpu_mem["gpu_max_memory_reserved_bytes"],
        "gpu_memory_fraction_applied":
            _gpu_mem["gpu_memory_fraction_applied"],
        "feature_workers": int(feature_workers),
        "double_buffer": bool(dbuf is not None),
        "double_buffer_active": bool(dbuf_healthy),
        "pinned_h2d": bool(h2d_pool is not None),
        "pinned_h2d_stats": (h2d_pool.stats() if h2d_pool is not None
                             else None),
        "dev_feature_pack": (
            {"pack_dir": str(dev_pack_view.pack_dir),
             "n_positions": int(dev_pack_view.n_positions),
             "features_sha256": str(
                 dev_pack_view.binding.get("features_sha256")),
             "positions_sha256": str(
                 dev_pack_view.binding.get("positions_sha256"))}
            if dev_pack_view is not None else None),
        "wall_clock_s": round(time.time() - t0, 3),
        "checkpoint_best_sha256": sha256_file(best_path),
        "checkpoint_best_path": str(best_path),
        "head_params": model.head_parameter_counts(),
        "head_parameter_counts": model.head_parameter_counts(),
        "provenance": prov,
    })
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    metrics["metrics_path"] = str(out / "metrics.json")
    return metrics


def _calibrate_tau_on(
    model: torch.nn.Module,
    dataset: Any,
    lru: _FeatureLRU,
    positions: Sequence[int],
    labels_v52: Any,
    endpoint_labels: Any,
    config: V53MPTrainerConfig,
    *,
    endpoint_pos_weight: float,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Forward the SELECTED model over inner-val rows; calibrate τ_end."""
    model.train(False)
    logits: List[float] = []
    ys: List[int] = []
    masks: List[bool] = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = list(positions[start:start + batch_size])
            built = [lru.get(int(p)) for p in chunk]
            batch = to_torch_batch(collate_v3(built), device=config.device)
            out = model(forward_view_v3(batch))
            sample_ids = [str(s) for s in batch["sample_ids"]]
            ep = _endpoint_for_batch(sample_ids, endpoint_labels,
                                     config.device)
            logits.extend(out["endpoint_logit"].cpu().tolist())
            ys.extend(
                (ep["y_endpoint"] * ep["endpoint_mask"].float()).cpu()
                .tolist())
            masks.extend(ep["endpoint_mask"].cpu().tolist())
    model.train(True)
    return calibrate_tau_end(
        logits, [int(v) for v in ys], masks,
        target_precision=config.tau_target_precision)
