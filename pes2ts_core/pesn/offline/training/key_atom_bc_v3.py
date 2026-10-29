"""R3 v3 behavior-cloning trainer (campaign PES2TS_30H_R3_V52_V53, W-LOSS E3).

Frozen-contract training loop for the Reaction-Intent-conditioned R3 arm
(30h plan §4.1; V5.1-R3 plan v3 §5).  Hard rules enforced here:

* loss math is CONSUMED read-only from
  :func:`pes2ts_core.pesn.models.key_atom_losses_v3.key_atom_loss_v3`
  (never reimplemented/modified here); the model input is exactly
  ``forward_view_v3(batch)`` and supervision (``y_teacher``, ``pair_edit``,
  ``frame_weight``/label_weight, ``d0``) reaches only the loss arguments;
  ``frame_weight`` (label_weight) is applied EXACTLY once, inside the loss.
* frozen hyperparameters (campaign yaml ``r3_training``): AdamW lr=1e-3
  wd=1e-4, warmup=250 updates, cosine schedule, grad clip 1.0, fp32,
  3000 updates, batch 32 = 8 curves x 4 frames (curve-balanced; forward/
  reverse balanced within a curve for bidirectional arms), seeds 17/29/43.
* inner-train/inner-val come ONLY from the run's outer-train fold clusters;
  checkpoint selection uses inner-val ONLY (never outer-val/confirm metrics).
* metrics use the frozen protocol functions
  (:func:`evaluate_g52` -> key_RMSE_over_zero; :func:`evaluate_g53` ->
  SRE/sign/dead-zone) aggregated per independent curve/cluster.  Dev runs
  RECORD guard comparisons vs zero-step and classic CV/Jacobian baselines;
  they do not gate.
* λ_cos follows the official R0 schedule (:func:`cos_lambda_r0`); aux arms
  (S2/S10/D2/D10) place their calibrated λ on their own bucket only
  (:func:`aux_arm_lambdas_v3`; arms are never mixed).
* determinism: every RNG is seeded per run; seeds are recorded; the
  checkpoint sha256 is returned for the ledger record; resume continues
  from ``checkpoint_last.pt`` by run identity.

The driver (``scripts/campaigns/run_v5_1_r3.py``) owns ledger/lifecycle;
this module owns the math loop only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
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
from pes2ts_core.pesn.models.key_atom_losses_v3 import (
    KeyAtomLossV3Config,
    aux_arm_lambdas_v3,
    cos_lambda_r0,
    key_atom_loss_v3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
    KeyAtomDatasetV3,
    SampleRecordV3,
)
from pes2ts_core.pesn.offline.evaluation.pair_ownership_v3 import (
    assign_batch_pairs,
    merge_pair_audits,
)

__all__ = [
    "TRAINER_SCHEMA",
    "ClassicJacobianBaselineError",
    "TrainerConfigV3",
    "TrainerError",
    "build_feature_cache",
    "classic_jacobian_step",
    "curve_batch_sampler",
    "evaluate_checkpoint",
    "make_inner_split",
    "set_seed",
    "sha256_file",
    "train_run",
]

TRAINER_SCHEMA = "pesn_key_atom_bc_v3_train_v1"

#: frozen campaign defaults (yaml r3_training; NOT overridable except via
#: explicit dev-smoke flags recorded in the run metadata)
LR_DEFAULT = 1.0e-3
WEIGHT_DECAY_DEFAULT = 1.0e-4
WARMUP_UPDATES_DEFAULT = 250
GRAD_CLIP_DEFAULT = 1.0
UPDATES_DEFAULT = 3000
BATCH_SIZE_DEFAULT = 32
CURVES_PER_BATCH_DEFAULT = 8
FRAMES_PER_CURVE_DEFAULT = 4
VAL_EVERY_DEFAULT = 250
INNER_VAL_FRACTION = 0.2
D0_METRIC_ANGSTROM = 0.08
GUARD_RMSE_DEGRADATION_MAX_PCT = 2.0
GUARD_SIGN_DROP_MAX = 0.01
GUARD_DEADZONE_RISE_MAX = 0.01


class TrainerError(RuntimeError):
    """Trainer input/state violation (fail-closed; never silent)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__("{}: {}".format(code, detail))
        self.code = code


class ClassicJacobianBaselineError(TrainerError):
    """Classic CV/Jacobian baseline could not be constructed for a frame."""


# ---------------------------------------------------------------------------
# config / determinism
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfigV3:
    """Frozen R3 hyper-parameters + the run's explicit arm switches."""

    arm: str                       # F0/F1/F2/F3/F2-bidir/F2-intent-off/B/S*/D*/...
    fold: int
    seed: int
    level: str = "F3"              # feature ladder level F0..F3
    intent_off: bool = False       # F-ablate (PD-6 full mask)
    bidirectional: bool = False    # F-bidir (fwd+rev share one curve budget)
    aux_arm: Optional[str] = None  # S2/S10/D2/D10 (bucket arms; never mixed)
    lambda_arm: float = 0.0        # calibrated λ for the aux bucket (<= 0.5)
    lr: float = LR_DEFAULT
    weight_decay: float = WEIGHT_DECAY_DEFAULT
    warmup_updates: int = WARMUP_UPDATES_DEFAULT
    grad_clip: float = GRAD_CLIP_DEFAULT
    updates: int = UPDATES_DEFAULT
    batch_size: int = BATCH_SIZE_DEFAULT
    curves_per_batch: int = CURVES_PER_BATCH_DEFAULT
    frames_per_curve: int = FRAMES_PER_CURVE_DEFAULT
    val_every: int = VAL_EVERY_DEFAULT
    device: str = "cpu"
    # dev-throughput smoke ONLY (recorded; formal runs keep updates=3000)
    smoke_updates: Optional[int] = None

    def __post_init__(self) -> None:
        if self.batch_size != self.curves_per_batch * self.frames_per_curve:
            raise TrainerError(
                "BATCH_COMPOSITION",
                "frozen batch = 8 curves x 4 frames; got {} x {}".format(
                    self.curves_per_batch, self.frames_per_curve))
        if self.aux_arm is not None and not (0.0 <= float(self.lambda_arm) <= 0.5):
            raise TrainerError(
                "AUX_LAMBDA_RANGE",
                "aux λ must lie in [0, 0.5] (yaml aux_calibration), got "
                "{}".format(self.lambda_arm))
        if self.aux_arm is None and float(self.lambda_arm) != 0.0:
            raise TrainerError(
                "AUX_LAMBDA_WITHOUT_ARM",
                "lambda_arm without aux_arm is incoherent")

    @property
    def effective_updates(self) -> int:
        return int(self.smoke_updates) if self.smoke_updates else int(self.updates)


def set_seed(seed: int) -> Dict[str, int]:
    """Seed every RNG; returns the recorded seed map."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "python": int(seed),
        "numpy": int(seed) % (2 ** 32),
        "torch": int(seed),
        "torch_cuda": int(seed) if torch.cuda.is_available() else -1,
    }


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# inner split (outer-train clusters ONLY) + curve-balanced sampler
# ---------------------------------------------------------------------------


def make_inner_split(
    samples: Sequence[SampleRecordV3],
    seed: int,
    *,
    val_fraction: float = INNER_VAL_FRACTION,
) -> Tuple[List[int], List[int]]:
    """Deterministic curve-level inner-train/inner-val split.

    Only the indices given (the run's outer-train fold) are considered --
    inner-val NEVER draws from outer-val or confirm clusters.  Curves are
    shuffled with the run seed; ~20% of curves (at least 1 when ≥2 curves)
    become inner-val for checkpoint selection.
    """
    curves = sorted({s.curve_id for s in samples})
    if len(curves) < 2:
        raise TrainerError(
            "INNER_SPLIT_CURVES",
            "need >=2 outer-train curves for an inner split, got {}".format(
                len(curves)))
    rng = np.random.RandomState(int(seed) ^ 0x5EED)
    shuffled = list(curves)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(val_fraction * len(shuffled))))
    n_val = min(n_val, len(shuffled) - 1)
    val_curves = set(shuffled[:n_val])
    train_idx = [i for i, s in enumerate(samples) if s.curve_id not in val_curves]
    val_idx = [i for i, s in enumerate(samples) if s.curve_id in val_curves]
    if not train_idx or not val_idx:
        raise TrainerError(
            "INNER_SPLIT_EMPTY", "inner split produced an empty side")
    return train_idx, val_idx


def curve_batch_sampler(
    samples: Sequence[SampleRecordV3],
    indices: Sequence[int],
    *,
    bidirectional: bool,
    seed: int,
    curves_per_batch: int = CURVES_PER_BATCH_DEFAULT,
    frames_per_curve: int = FRAMES_PER_CURVE_DEFAULT,
):
    """Yield curve-balanced index batches forever (deterministic per seed).

    Composition (frozen yaml ``r3_training.batch``): ``curves_per_batch``
    distinct curves per batch, ``frames_per_curve`` frames per curve, and
    forward/reverse balanced within a curve for bidirectional arms (half
    each when the curve offers both directions).  Forward-only arms draw
    forward-direction frames only.  Yields positions into ``indices``
    (cache positions), not dataset indices.
    """
    by_curve_dir: Dict[Tuple[str, str], List[int]] = {}
    for pos, i in enumerate(indices):
        rec = samples[int(i)]
        by_curve_dir.setdefault((rec.curve_id, rec.intent.direction), []).append(pos)
    curves = sorted({samples[int(i)].curve_id for i in indices})
    if not curves:
        raise TrainerError("SAMPLER_EMPTY", "no inner-train curves to sample")
    directions = ("forward", "reverse") if bidirectional else ("forward",)
    for curve in curves:
        for direction in directions:
            pool = by_curve_dir.get((curve, direction))
            if not pool and direction == "forward":
                # a curve with only reverse intents still trains forward arms
                alt = by_curve_dir.get((curve, "reverse"))
                if alt:
                    by_curve_dir[(curve, direction)] = list(alt)
    rng = np.random.RandomState(int(seed))
    while True:
        # frozen batch = curves_per_batch curve slots x frames_per_curve; a
        # fold with fewer curves repeats curves (fresh frames each slot) so
        # the batch stays 32 and every curve stays equally likely
        chosen = list(rng.choice(
            curves, size=curves_per_batch, replace=len(curves) < curves_per_batch))
        batch: List[int] = []
        for curve in chosen:
            per_dir = []
            if bidirectional:
                half = frames_per_curve // 2
                rest = frames_per_curve - half
                per_dir = [("forward", half), ("reverse", rest)]
            else:
                per_dir = [("forward", frames_per_curve)]
            for direction, count in per_dir:
                pool = by_curve_dir.get((curve, direction)) or \
                    by_curve_dir.get((curve, "forward")) or \
                    by_curve_dir.get((curve, "reverse"))
                if not pool:
                    continue
                replace = len(pool) < count
                picks = rng.choice(pool, size=count, replace=replace)
                batch.extend(int(p) for p in np.atleast_1d(picks))
        if batch:
            yield batch


# ---------------------------------------------------------------------------
# feature cache + batching
# ---------------------------------------------------------------------------


def build_feature_cache(
    dataset: KeyAtomDatasetV3,
    indices: Sequence[int],
    *,
    level: str = "F3",
    intent_off: bool = False,
) -> List[Dict[str, Any]]:
    """Pre-build numpy feature dicts (deterministic; sampled many times)."""
    cache: List[Dict[str, Any]] = []
    for i in indices:
        rec = dataset.samples[int(i)]
        cache.append(
            build_sample_features_v3(
                dataset.to_sample_v3(rec), level=level, intent_off=intent_off))
    return cache


def _batch_from_cache(
    cache: Sequence[Dict[str, Any]],
    positions: Sequence[int],
    device: str,
) -> Dict[str, Any]:
    built = [cache[int(p)] for p in positions]
    return to_torch_batch(collate_v3(built), device=device)


def _key_ptr_of(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    n_keys = batch["n_keys"].long()
    return torch.cat(
        [torch.zeros(1, dtype=torch.long, device=n_keys.device),
         torch.cumsum(n_keys, dim=0)])


# ---------------------------------------------------------------------------
# classic CV/Jacobian baseline (record-only; explicit formula)
# ---------------------------------------------------------------------------


def classic_jacobian_step(
    pos: np.ndarray,
    key_rows: Sequence[int],
    pairs: Sequence[Tuple[int, int]],
    pair_delta_target: np.ndarray,
) -> np.ndarray:
    """Classic CV/Jacobian min-norm step (record-only baseline).

    Solves ``min ||u|| s.t. J u = Δd*`` in the first-order pair projection:
    ``J_l(u) = e_l·(u_j − u_i)`` with ``e_l`` from the CURRENT frame, and
    ``Δd*_l = clip(δB_intent_l, −1, +1) · d0`` (one d0-scale bond move per
    directed CV coordinate).  Solution: ``u = Jᵀ(JJᵀ)⁺Δd*`` (Moore-Penrose
    on the pair space; singular systems use ``lstsq``).  This is a
    diagnostic baseline for guard recording — NOT a model input and NOT a
    gate.
    """
    pos = np.asarray(pos, dtype=np.float64)
    key = np.asarray(key_rows, dtype=np.int64)
    slot = {int(r): k for k, r in enumerate(key)}
    n_key = len(key)
    n_pairs = len(list(pairs))
    jac = np.zeros((n_pairs, n_key * 3), dtype=np.float64)
    for l, (i, j) in enumerate(pairs):
        si, sj = slot[int(i)], slot[int(j)]
        r = pos[int(j)] - pos[int(i)]
        norm = float(np.linalg.norm(r))
        e = r / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
        jac[l, sj * 3:sj * 3 + 3] += e
        jac[l, si * 3:si * 3 + 3] -= e
    target = np.asarray(pair_delta_target, dtype=np.float64)
    try:
        sol, *_ = np.linalg.lstsq(jac, target, rcond=None)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - lstsq is robust
        raise ClassicJacobianBaselineError(str(exc)) from exc
    return sol.reshape(n_key, 3)


# ---------------------------------------------------------------------------
# evaluation (frozen protocol metrics + recorded guards)
# ---------------------------------------------------------------------------


def _predict_batches(
    model: torch.nn.Module,
    cache: Sequence[Dict[str, Any]],
    positions: Sequence[int],
    device: str,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """Run the model over ``positions``; return per-sample blocks."""
    model.train(False)
    out: List[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = list(positions[start:start + batch_size])
            batch = _batch_from_cache(cache, chunk, device)
            u = model(forward_view_v3(batch))["u"].detach().cpu().numpy()
            out.append({"batch": batch, "u": u})
    model.train()
    return out


def evaluate_checkpoint(
    model: torch.nn.Module,
    dataset: KeyAtomDatasetV3,
    cache: Sequence[Dict[str, Any]],
    positions: Sequence[int],
    *,
    device: str,
    d0: float = D0_METRIC_ANGSTROM,
    bootstrap_draws: int = 200,
) -> Dict[str, Any]:
    """Frozen metrics on the given (inner-val) positions.

    Returns key_RMSE_over_zero (``evaluate_g52``), SRE/sign/dead-zone
    (``evaluate_g53``), the zero-step and classic CV/Jacobian baselines and
    the recorded (never gating) guard comparisons, aggregated per
    independent curve/cluster.
    """
    blocks = _predict_batches(model, cache, positions, device)
    # per-sample arrays in dataset order
    u_samples: List[np.ndarray] = []
    y_samples: List[np.ndarray] = []
    curve_ids: List[str] = []
    g53_samples: List[Dict[str, Any]] = []
    pair_audits: List[Dict[str, Any]] = []
    for block in blocks:
        batch = block["batch"]
        u_all = block["u"]
        n_keys = batch["n_keys"].long().cpu().numpy()
        n_nodes = batch["n_nodes"].long().cpu().numpy()
        key_index = batch[NODE_KEY_SET_KEY].long().cpu().numpy()
        y_all = batch["y_teacher"].cpu().numpy()
        pos_all = batch[NODE_POS_KEY].cpu().numpy()
        pair_index = batch["pair_index"].long().cpu().numpy()
        pair_edit = batch["pair_edit"].cpu().numpy()
        # WP1a (V5.4.1 §4.1): pair_index is batch-global NODE rows — pair
        # ownership uses the node-row pointer (n_nodes); the key-row pointer
        # (n_keys) only slices u/y/key_index.  Shared tool, both entry points.
        ownership = assign_batch_pairs(
            n_nodes=n_nodes, n_keys=n_keys, key_index=key_index,
            pair_index=pair_index, pair_edit=pair_edit)
        pair_audits.append(ownership.audit(per_graph=False))
        for g, gs in enumerate(ownership.graphs):
            ks, ke = gs.key_lo, gs.key_hi
            u_g = u_all[ks:ke]
            y_g = y_all[ks:ke]
            rows_g = key_index[ks:ke]
            u_samples.append(u_g.astype(np.float64))
            y_samples.append(y_g.astype(np.float64))
            curve_ids.append(str(batch["curve_ids"][g]))
            g53_samples.append({
                "pos": pos_all,
                "key_rows": [int(r) for r in rows_g],
                "pairs": [(int(a), int(b)) for a, b in gs.pairs],
                "pair_edits": [float(e) for e in gs.pair_edits],
                "u": u_g.astype(np.float64),
                "y": y_g.astype(np.float64),
                "group_id": str(batch["curve_ids"][g]),
            })
    from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
        evaluate_g52,
        evaluate_g53,
    )

    g52 = evaluate_g52(
        u_samples, y_samples, curve_ids, bootstrap_draws=bootstrap_draws)
    g53 = evaluate_g53(g53_samples)
    # zero-step baseline through the SAME frozen machinery
    g52_zero = evaluate_g52(
        [np.zeros_like(y) for y in y_samples], y_samples, curve_ids,
        bootstrap_draws=bootstrap_draws)
    g53_zero = evaluate_g53([
        {**s, "u": np.zeros_like(s["u"])} for s in g53_samples])
    # classic CV/Jacobian baseline (record-only; explicit formula above):
    # Delta_d* = clip(delta_B_intent, -1, +1) * d0 per CV pair
    classic_blocks: List[np.ndarray] = []
    n_classic_failed = 0
    for s in g53_samples:
        if not s["pairs"]:
            classic_blocks.append(np.zeros_like(s["u"]))
            continue
        target = np.clip(np.asarray(s["pair_edits"], dtype=np.float64),
                         -1.0, 1.0) * float(d0)
        try:
            u_c = classic_jacobian_step(
                s["pos"], s["key_rows"], s["pairs"], target)
        except ClassicJacobianBaselineError:
            n_classic_failed += 1
            u_c = np.zeros_like(s["u"])
        classic_blocks.append(u_c)
    g53_classic = evaluate_g53([
        {**s, "u": u_c} for s, u_c in zip(g53_samples, classic_blocks)])

    def _agg(block: Mapping[str, Any]) -> Mapping[str, Any]:
        return block["aggregate"]

    model_rmse_zero_ratio = g52["raw"]["aggregate"].get("median_rmse_over_zero")
    zero_rmse_ratio = g52_zero["raw"]["aggregate"].get("median_rmse_over_zero")
    guards = {
        "policy": "record-only for dev runs (never gates a dev run)",
        "rmse_degradation_max_pct": GUARD_RMSE_DEGRADATION_MAX_PCT,
        "sign_drop_max": GUARD_SIGN_DROP_MAX,
        "deadzone_rise_max": GUARD_DEADZONE_RISE_MAX,
        "model_median_rmse_over_zero": model_rmse_zero_ratio,
        "zero_step_median_rmse_over_zero": zero_rmse_ratio,
        "rmse_degradation_vs_zero_step_pct": (
            None if (model_rmse_zero_ratio is None or not zero_rmse_ratio)
            else 100.0 * (model_rmse_zero_ratio - 1.0)),
        "model_sign_median": _agg(g53).get("sign_accuracy_cluster_median"),
        "zero_step_sign_median": _agg(g53_zero).get("sign_accuracy_cluster_median"),
        "classic_sign_median": _agg(g53_classic).get("sign_accuracy_cluster_median"),
        "sign_drop_vs_zero_step": (
            None if _agg(g53).get("sign_accuracy_cluster_median") is None
            else _agg(g53_zero).get("sign_accuracy_cluster_median", 0.0)
            - _agg(g53).get("sign_accuracy_cluster_median")),
        "model_dead_zone_fraction": _agg(g53).get("dead_zone_fraction"),
        "zero_step_dead_zone_fraction": _agg(g53_zero).get("dead_zone_fraction"),
        "deadzone_rise_vs_zero_step": (
            None if _agg(g53).get("dead_zone_fraction") is None
            else _agg(g53).get("dead_zone_fraction")
            - _agg(g53_zero).get("dead_zone_fraction")),
        "model_sre_median": _agg(g53).get("sre_cluster_median"),
        "classic_sre_median": _agg(g53_classic).get("sre_cluster_median"),
        "classic_jacobian_failed_frames": n_classic_failed,
    }
    return {
        "schema": TRAINER_SCHEMA,
        "aggregate_unit": "independent curve/cluster",
        "key_rmse_over_zero": g52["raw"]["aggregate"],
        "key_rmse_over_zero_bootstrap": g52["raw"].get("bootstrap"),
        "sre_and_sign": g53["aggregate"],
        "baselines": {
            "zero_step": {
                "key_rmse_over_zero": g52_zero["raw"]["aggregate"],
                "sre_and_sign": g53_zero["aggregate"],
            },
            "classic_cv_jacobian": {
                "formula": ("min-norm first-order pair step: u = Jᵀ(JJᵀ)⁺Δd*, "
                            "Δd* = one d0-scale move per CV pair; record-only"),
                "sre_and_sign": g53_classic["aggregate"],
            },
        },
        "guards": guards,
        # WP1a additive audit (V5.4.1 §4.1): pair denominators per eval —
        # totals only, O(1); g53 sample structure above is unchanged.
        "pair_ownership_audit": merge_pair_audits(pair_audits),
        "n_val_samples": len(u_samples),
        "n_val_curves": len(set(curve_ids)),
    }


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def _lr_at(update: int, config: TrainerConfigV3) -> float:
    """Linear warmup to lr, then cosine decay to 0 at ``updates``."""
    warmup = max(int(config.warmup_updates), 1)
    if update <= warmup:
        return config.lr * update / warmup
    progress = (update - warmup) / max(config.effective_updates - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    best_val: float,
    best_update: int,
    config: TrainerConfigV3,
    seed_map: Mapping[str, int],
) -> None:
    payload = {
        "schema": TRAINER_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "best_val": float(best_val),
        "best_update": int(best_update),
        "config": asdict(config),
        "seeds": dict(seed_map),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(path)


def _load_checkpoint(path: Path, map_location: str) -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def train_run(
    dataset: KeyAtomDatasetV3,
    config: TrainerConfigV3,
    out_dir: Any,
    *,
    resume: bool = True,
) -> Dict[str, Any]:
    """Train one planned run; returns the metrics/summary dict.

    Writes to ``out_dir``: ``checkpoint_best.pt`` (inner-val-selected),
    ``checkpoint_last.pt`` (resume), ``metrics.json``.  Selection uses
    inner-val ONLY.  With ``resume=True`` an existing ``checkpoint_last.pt``
    for the same (arm, fold, seed, update) continues the loop (attempt
    counting is the driver's/ledger's job).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_map = set_seed(config.seed)
    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise TrainerError("CUDA_UNAVAILABLE", "device=cuda but no CUDA device")

    indices = list(range(len(dataset.samples)))
    train_idx, val_idx = make_inner_split(dataset.samples, config.seed)
    train_cache = build_feature_cache(
        dataset, train_idx, level=config.level, intent_off=config.intent_off)
    val_cache = build_feature_cache(
        dataset, val_idx, level=config.level, intent_off=config.intent_off)
    # cache is index-aligned with the split lists
    train_positions = list(range(len(train_idx)))
    val_positions = list(range(len(val_idx)))

    set_seed(config.seed)  # re-seed AFTER caching so model init is seed-exact
    model = StarEGNNV3().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_cfg = KeyAtomLossV3Config()
    lambda_strong, lambda_dead = (0.0, 0.0)
    if config.aux_arm is not None:
        lambda_strong, lambda_dead = aux_arm_lambdas_v3(
            config.aux_arm, config.lambda_arm)

    start_update = 1
    best_val = float("inf")
    best_update = 0
    last_path = out / "checkpoint_last.pt"
    best_path = out / "checkpoint_best.pt"
    if resume and last_path.is_file():
        state = _load_checkpoint(last_path, device)
        if state.get("config", {}).get("arm") != config.arm or \
           state.get("config", {}).get("fold") != config.fold or \
           int(state.get("config", {}).get("seed", -1)) != int(config.seed):
            raise TrainerError(
                "RESUME_IDENTITY_MISMATCH",
                "checkpoint_last.pt belongs to {} fold{} seed{}; refusing to "
                "resume into {} fold{} seed{}".format(
                    state.get("config", {}).get("arm"),
                    state.get("config", {}).get("fold"),
                    state.get("config", {}).get("seed"),
                    config.arm, config.fold, config.seed))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_update = int(state["update"]) + 1
        best_val = float(state["best_val"])
        best_update = int(state["best_update"])
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        random.setstate(tuple(state["python_rng"]))
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(
                [s.cpu() for s in state["cuda_rng"]])

    sampler = curve_batch_sampler(
        dataset.samples, train_idx,
        bidirectional=config.bidirectional, seed=config.seed + 1,
        curves_per_batch=config.curves_per_batch,
        frames_per_curve=config.frames_per_curve)

    log_path = out / "train_log.jsonl"
    effective = config.effective_updates
    t0 = time.time()
    last_train_loss = None
    for update in range(start_update, effective + 1):
        positions = next(sampler)
        batch = _batch_from_cache(train_cache, positions, device)
        lr_t = _lr_at(update, config)
        for group in optimizer.param_groups:
            group["lr"] = lr_t
        optimizer.zero_grad(set_to_none=True)
        out_fwd = model(forward_view_v3(batch))
        loss_out = key_atom_loss_v3(
            out_fwd["u"],
            batch["y_teacher"],
            batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
            _key_ptr_of(batch),
            batch[NODE_POS_KEY],
            batch[NODE_KEY_SET_KEY],
            batch["pair_index"],
            config=loss_cfg,
            lambda_cos=cos_lambda_r0(update),
            frame_weight=batch["frame_weight"],   # label_weight: applied ONCE
            pair_edit=batch["pair_edit"],
            lambda_strong=lambda_strong,
            lambda_dead=lambda_dead,
            aux_arm=config.aux_arm,
        )
        loss_out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        last_train_loss = float(loss_out["loss"].detach().cpu())

        if update % 100 == 0 or update == 1 or update == effective:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "update": update,
                    "lr": lr_t,
                    "loss": last_train_loss,
                    "l_atom": float(loss_out["l_atom"].detach().cpu()),
                    "l_pair": float(loss_out["l_pair"].detach().cpu()),
                    "l_cos": float(loss_out["l_cos"].detach().cpu()),
                    "lambda_cos": loss_out["lambda_cos"],
                    "lambda_strong": loss_out["lambda_strong"],
                    "lambda_dead": loss_out["lambda_dead"],
                    "elapsed_s": round(time.time() - t0, 3),
                }, sort_keys=True) + "\n")

        if update % max(int(config.val_every), 1) == 0 or update == effective:
            val_loss = _inner_val_loss(
                model, val_cache, val_positions, device, loss_cfg,
                lambda_strong, lambda_dead, config)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "update": update, "inner_val_loss": val_loss,
                }, sort_keys=True) + "\n")
            if val_loss < best_val:
                best_val = val_loss
                best_update = update
                _save_checkpoint(
                    best_path, model=model, optimizer=optimizer, update=update,
                    best_val=best_val, best_update=best_update,
                    config=config, seed_map=seed_map)
            _save_checkpoint(
                last_path, model=model, optimizer=optimizer, update=update,
                best_val=best_val, best_update=best_update,
                config=config, seed_map=seed_map)

    # final evaluation from the inner-val-selected checkpoint
    if best_path.is_file():
        state = _load_checkpoint(best_path, device)
        model.load_state_dict(state["model"])
        selected_update = int(state["update"])
    else:  # val_every > updates edge case: last state is the selection
        selected_update = effective
        _save_checkpoint(
            best_path, model=model, optimizer=optimizer, update=selected_update,
            best_val=best_val, best_update=selected_update,
            config=config, seed_map=seed_map)
    metrics = evaluate_checkpoint(
        model, dataset, val_cache, val_positions, device=device)
    metrics.update({
        "arm": config.arm,
        "fold": config.fold,
        "seed": config.seed,
        "level": config.level,
        "intent_off": config.intent_off,
        "bidirectional": config.bidirectional,
        "aux_arm": config.aux_arm or "none",
        "lambda_arm": float(config.lambda_arm),
        "updates": effective,
        "selected_update": selected_update,
        "best_inner_val_loss": None if best_val == float("inf") else best_val,
        "best_update": best_update,
        "last_train_loss": last_train_loss,
        "inner_val_curves": sorted({dataset.samples[val_idx[i]].curve_id
                                    for i in val_positions}),
        "inner_train_curves": sorted({dataset.samples[train_idx[i]].curve_id
                                      for i in train_positions}),
        "seeds": seed_map,
        "device": device,
        "hyperparameters": {
            "optimizer": "AdamW", "lr": config.lr,
            "weight_decay": config.weight_decay,
            "warmup_updates": config.warmup_updates,
            "schedule": "cosine", "grad_clip": config.grad_clip,
            "batch_size": config.batch_size,
            "composition": "{} curves x {} frames".format(
                config.curves_per_batch, config.frames_per_curve),
            "lambda_cos_schedule": "R0 (cos_lambda_r0)",
        },
        "wall_clock_s": round(time.time() - t0, 3),
        "checkpoint_best_sha256": sha256_file(best_path),
        "checkpoint_best_path": str(best_path),
        "selection_metric": "inner-val loss (inner-train fold clusters only)",
        "n_train_samples": len(train_idx),
        "n_val_samples_total": len(val_idx),
    })
    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def _inner_val_loss(
    model: torch.nn.Module,
    val_cache: Sequence[Dict[str, Any]],
    positions: Sequence[int],
    device: str,
    loss_cfg: KeyAtomLossV3Config,
    lambda_strong: float,
    lambda_dead: float,
    config: TrainerConfigV3,
    batch_size: int = 64,
) -> float:
    """Weighted mean inner-val loss (same loss, λ_cos frozen at R0 end value)."""
    model.train(False)
    total = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = list(positions[start:start + batch_size])
            batch = _batch_from_cache(val_cache, chunk, device)
            out_fwd = model(forward_view_v3(batch))
            loss_out = key_atom_loss_v3(
                out_fwd["u"],
                batch["y_teacher"],
                batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
                _key_ptr_of(batch),
                batch[NODE_POS_KEY],
                batch[NODE_KEY_SET_KEY],
                batch["pair_index"],
                config=loss_cfg,
                lambda_cos=cos_lambda_r0(config.effective_updates),
                frame_weight=batch["frame_weight"],
                pair_edit=batch["pair_edit"],
                lambda_strong=lambda_strong,
                lambda_dead=lambda_dead,
                aux_arm=config.aux_arm,
            )
            w = float(batch["frame_weight"].sum().cpu())
            total += float(loss_out["loss"].cpu()) * w
            weight_sum += w
    model.train()
    return total / max(weight_sum, 1e-9)
