"""V5.2 joint head trainer (campaign PES2TS_30H_R3_V52_V53, plan §5.5; W-LOSS E5).

Frozen-contract training loop for the Q / T / QT V5.2 arms:

* **init**: ``KeyAtomV52Model(backbone=StarEGNNV3)`` where the backbone is
  loaded VERBATIM from the matched R3 winner checkpoint (fold+seed; the
  E3 winner config is F2-bidir).  Heads are freshly seeded.  The backbone
  is shared by reference — no copy, no in-place edit at construction.
* **split protocol (E4-coherent)**: a fold-F run TRAINS on the dev folds
  ``!= F`` (the E4 label gate's frozen "inner-train = dev folds minus fold
  F" provenance for the stop class weights), selects checkpoints on an
  inner-val split of those training curves (seed-deterministic, never
  fold F), and REPORTS metrics on fold F (outer-val — never used for
  selection).
* **schedule (yaml v52_training.schedule, frozen)**: freeze the encoder
  for 400 updates (heads only, lr_head=3e-4), then 1500 joint updates
  (heads lr 3e-4, encoder lr 1e-4); AdamW (wd 1e-4, R3 continuity),
  grad clip 1.0, fp32.
* **loss**: ``key_atom_loss_v52`` (read-only, never reimplemented):
  ``L_move + λq·L_q + λm·L_TSStop``; λq/λm come from the documented
  gradient-share probe (``r3_v52_lambda_probe.py`` output; never guessed);
  λ_mono=λ_path=0 asserted by the frozen loss; λ_cos held at the R0 end
  value 0.02 (the winner's selection regime).
* **class weights**: the FROZEN per-fold ``StopClassWeightsV52`` from the
  G3a label report (compiled here AND asserted equal to the report when
  present); Mask never enters numerator or denominator (frozen loss).
* **arms**: Q trains the progress head only, T the stop head only, QT
  both; L_move is always active.
* **metrics**: q-head MAE + Pearson per curve (vs the constant-q
  baseline), stop precision/recall/AUC on unmasked rows (Mask excluded),
  and Move non-degradation vs the matched R3 winner re-evaluated on the
  same fold-F curve set (frozen g52/g53 machinery; guard ≤2% RMSE).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, asdict
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
from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV52Model
from pes2ts_core.pesn.models.key_atom_loss_v52_v3 import (
    StopClassWeightsV52,
    key_atom_loss_v52,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
    KeyAtomDatasetV3,
    V52LabelTable,
)
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
    TrainerError,
    build_feature_cache,
    evaluate_checkpoint,
    sha256_file,
)

__all__ = [
    "FROZEN_STOP_WEIGHTS_V52",
    "V52TrainerConfig",
    "V52TrainerError",
    "auc_mann_whitney",
    "load_backbone_from_r3",
    "pearson",
    "train_run_v52",
    "v52_curve_split",
]

V52_SCHEMA = "pesn_key_atom_v52_train_v1"

#: frozen yaml v52_training.schedule
FREEZE_ENCODER_UPDATES = 400
JOINT_UPDATES = 1500
LR_HEAD = 3.0e-4
LR_ENCODER = 1.0e-4
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4          # R3 continuity (yaml v52 block does not restate wd)
LAMBDA_COS_FIXED = 0.02      # R0 end value held constant (winner's regime)
VAL_EVERY = 100
BATCH_SIZE = 32
CURVES_PER_BATCH = 8
FRAMES_PER_CURVE = 4

#: frozen G3a stop class weights (inner-train = dev folds minus fold F);
#: asserted equal to e4_label_report.json when that report is present
FROZEN_STOP_WEIGHTS_V52: Dict[int, Tuple[float, int, int]] = {
    0: (20.3109756097561, 164, 3331),
    1: (16.88888888888889, 171, 2888),
    2: (19.87425149700599, 167, 3319),
}
E4_LABEL_REPORT_SUBPATH = "outputs/pesn/30h_campaign/e4_labels/e4_label_report.json"


class V52TrainerError(TrainerError):
    """V5.2 trainer input/state violation."""


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class V52TrainerConfig:
    """One planned V5.2 run (Q / T / QT)."""

    arm: str                       # "Q" | "T" | "QT"
    fold: int
    seed: int
    lambda_q: float                # calibrated (r3_v52_lambda_probe output)
    lambda_m: float                # calibrated
    backbone_run_id: str           # matched R3 winner checkpoint run_id
    device: str = "cpu"
    freeze_encoder_updates: int = FREEZE_ENCODER_UPDATES
    joint_updates: int = JOINT_UPDATES
    lr_head: float = LR_HEAD
    lr_encoder: float = LR_ENCODER
    weight_decay: float = WEIGHT_DECAY
    grad_clip: float = GRAD_CLIP
    val_every: int = VAL_EVERY
    smoke_updates: Optional[int] = None   # dev-stability smoke ONLY

    def __post_init__(self) -> None:
        if self.arm not in ("Q", "T", "QT"):
            raise V52TrainerError("ARM", "unknown V5.2 arm {!r}".format(self.arm))
        if self.fold not in (0, 1, 2):
            raise V52TrainerError(
                "FOLD", "V5.2 dev arms cover folds0-2, got {}".format(self.fold))
        # ACTIVE terms must carry the calibrated lambda; OFF terms must be 0
        if self.train_progress and not (0.0 < float(self.lambda_q) <= 0.5):
            raise V52TrainerError(
                "LAMBDA_RANGE",
                "lambda_q must come from the calibrated probe in (0, 0.5]; "
                "got {}".format(self.lambda_q))
        if self.train_stop and not (0.0 < float(self.lambda_m) <= 0.5):
            raise V52TrainerError(
                "LAMBDA_RANGE",
                "lambda_m must come from the calibrated probe in (0, 0.5]; "
                "got {}".format(self.lambda_m))
        if not self.train_progress and float(self.lambda_q) != 0.0:
            raise V52TrainerError(
                "LAMBDA_OFF_TERM",
                "arm {} must have lambda_q=0, got {}".format(
                    self.arm, self.lambda_q))
        if not self.train_stop and float(self.lambda_m) != 0.0:
            raise V52TrainerError(
                "LAMBDA_OFF_TERM",
                "arm {} must have lambda_m=0, got {}".format(
                    self.arm, self.lambda_m))

    @property
    def train_progress(self) -> bool:
        return self.arm in ("Q", "QT")

    @property
    def train_stop(self) -> bool:
        return self.arm in ("T", "QT")

    @property
    def total_updates(self) -> int:
        return int(self.smoke_updates) if self.smoke_updates else (
            int(self.freeze_encoder_updates) + int(self.joint_updates))


# ---------------------------------------------------------------------------
# split + labels
# ---------------------------------------------------------------------------


def v52_curve_split(
    samples: Sequence[Any],
    fold: int,
    seed: int,
    *,
    val_fraction: float = 0.2,
) -> Tuple[List[int], List[int], List[int]]:
    """E4-coherent split: train+inner-val on DEV folds != ``fold``; fold-F
    curves are outer-val (metrics only, NEVER selection).  Confirm folds
    3/4 NEVER enter this split (sealed; plan §4.2).  Inner split is a
    seeded curve-level shuffle of the train pool (R3 discipline).
    """
    dev_folds = (0, 1, 2)
    train_pool = [i for i, s in enumerate(samples)
                  if int(s.fold) in dev_folds and int(s.fold) != int(fold)]
    test = [i for i, s in enumerate(samples) if int(s.fold) == int(fold)]
    if not train_pool or not test:
        raise V52TrainerError(
            "SPLIT_EMPTY",
            "fold {}: train pool {} / eval {} samples".format(
                fold, len(train_pool), len(test)))
    used_folds = {int(samples[i].fold) for i in train_pool}
    if not used_folds <= set(dev_folds) - {int(fold)}:
        raise V52TrainerError(
            "SPLIT_SEAL",
            "train pool touches non-dev folds: {}".format(sorted(used_folds)))
    curves = sorted({samples[i].curve_id for i in train_pool})
    if len(curves) < 2:
        raise V52TrainerError(
            "SPLIT_CURVES", "need >=2 train curves, got {}".format(len(curves)))
    rng = np.random.RandomState(int(seed) ^ 0x5EED52)
    shuffled = list(curves)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(val_fraction * len(shuffled))))
    n_val = min(n_val, len(shuffled) - 1)
    val_curves = set(shuffled[:n_val])
    train = [i for i in train_pool if samples[i].curve_id not in val_curves]
    val = [i for i in train_pool if samples[i].curve_id in val_curves]
    return train, val, test


def stop_class_weights_for_fold(
    fold: int, report_path: Optional[Any] = None
) -> StopClassWeightsV52:
    """Frozen per-fold weights; asserted equal to the G3a report when present."""
    if int(fold) not in FROZEN_STOP_WEIGHTS_V52:
        raise V52TrainerError(
            "STOP_WEIGHTS_FOLD",
            "no frozen stop class weight for fold {}".format(fold))
    pw, n_pos, n_neg = FROZEN_STOP_WEIGHTS_V52[int(fold)]
    path = Path(report_path) if report_path is not None else Path(
        E4_LABEL_REPORT_SUBPATH)
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        entries = report.get("stop_labels", {}).get("class_weights_frozen", [])
        for entry in entries:
            if entry.get("fold") == int(fold):
                if not math.isclose(float(entry["pos_weight"]), pw,
                                    rel_tol=0, abs_tol=1e-12):
                    raise V52TrainerError(
                        "STOP_WEIGHTS_DRIFT",
                        "compiled pos_weight {} != report {} for fold {}".format(
                            pw, entry["pos_weight"], fold))
                if int(entry["n_pos"]) != n_pos or int(entry["n_neg"]) != n_neg:
                    raise V52TrainerError(
                        "STOP_WEIGHTS_DRIFT",
                        "compiled counts != report counts for fold {}".format(fold))
                break
        else:
            raise V52TrainerError(
                "STOP_WEIGHTS_MISSING",
                "e4 report {} has no frozen weight for fold {}".format(path, fold))
    return StopClassWeightsV52(
        pos_weight=float(pw), n_pos=int(n_pos), n_neg=int(n_neg),
        source="inner-train = dev folds minus fold {}".format(fold))


# ---------------------------------------------------------------------------
# model init / frozen-phase helpers
# ---------------------------------------------------------------------------


def load_backbone_from_r3(
    r3_ckpt: Any, *, device: str = "cpu"
) -> KeyAtomV52Model:
    """KeyAtomV52Model whose backbone is the R3 checkpoint, heads fresh.

    The R3 payload's ``model`` state maps EXACTLY onto StarEGNNV3 (E3
    checkpoints); heads are left at their (seeded) fresh init.
    """
    payload = torch.load(str(r3_ckpt), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise V52TrainerError(
            "R3_CHECKPOINT_FORMAT", "{} is not an R3 trainer checkpoint".format(
                r3_ckpt))
    backbone = StarEGNNV3()
    backbone.load_state_dict(payload["model"], strict=True)
    model = KeyAtomV52Model(backbone=backbone)
    return model.to(device)


def _set_encoder_trainable(model: KeyAtomV52Model, trainable: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad_(trainable)


def _set_heads_trainable(model: KeyAtomV52Model, arm: str) -> None:
    progress = arm in ("Q", "QT")
    stop = arm in ("T", "QT")
    for param in model.progress_head.parameters():
        param.requires_grad_(progress)
    for param in model.ts_stop_head.parameters():
        param.requires_grad_(stop)


# ---------------------------------------------------------------------------
# math helpers (metrics)
# ---------------------------------------------------------------------------


def pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def auc_mann_whitney(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """AUC via the rank statistic (average ranks for ties); None if degenerate."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + 1 + j + 1)
            ranks[order[i:j + 1]] = avg
        i = j + 1
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def _key_ptr(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    n_keys = batch["n_keys"].long()
    return torch.cat(
        [torch.zeros(1, dtype=torch.long, device=n_keys.device),
         torch.cumsum(n_keys, dim=0)])


def _labels_for_batch(
    batch: Mapping[str, Any], labels: V52LabelTable, device: str
) -> Dict[str, torch.Tensor]:
    """q/stop label tensors ([G]) for the batch's sample_ids (E4 table)."""
    q_true, q_mask, y_stop, stop_mask = [], [], [], []
    for sid in batch["sample_ids"]:
        lab = labels.by_sample.get(str(sid))
        if lab is None:
            raise V52TrainerError(
                "LABEL_MISSING",
                "sample {} has no v52 label row".format(sid))
        q_true.append(0.0 if lab.q is None else float(lab.q))
        q_mask.append(bool(lab.q_mask))
        y_stop.append(1 if lab.stop == 1 else 0)
        stop_mask.append(bool(lab.stop_mask))
    return {
        "q_true": torch.tensor(q_true, dtype=torch.float32, device=device),
        "q_mask": torch.tensor(q_mask, dtype=torch.bool, device=device),
        "y_stop": torch.tensor(y_stop, dtype=torch.float32, device=device),
        "stop_mask": torch.tensor(stop_mask, dtype=torch.bool, device=device),
    }


def _make_optimizer(model: KeyAtomV52Model, config: V52TrainerConfig):
    head_params = list(model.progress_head.parameters()) + \
        list(model.ts_stop_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": config.lr_head},
            {"params": list(model.backbone.parameters()),
             "lr": config.lr_encoder},
        ],
        weight_decay=config.weight_decay,
    )


def _loss_on_batch(
    model: KeyAtomV52Model,
    batch: Mapping[str, Any],
    labels: V52LabelTable,
    config: V52TrainerConfig,
    stop_weights: StopClassWeightsV52,
) -> Dict[str, Any]:
    from pes2ts_core.pesn.models.key_atom_losses_v3 import KeyAtomLossV3Config

    out = model(forward_view_v3(batch))
    lab = _labels_for_batch(batch, labels, config.device)
    return key_atom_loss_v52(
        out["u"], batch["y_teacher"],
        batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
        _key_ptr(batch), batch[NODE_POS_KEY], batch[NODE_KEY_SET_KEY],
        batch["pair_index"],
        q_pred=out["q"] if config.train_progress else None,
        q_true=lab["q_true"], q_mask=lab["q_mask"],
        stop_logit=out["stop_logit"] if config.train_stop else None,
        y_stop=lab["y_stop"], stop_mask=lab["stop_mask"],
        stop_class_weights=stop_weights if config.train_stop else None,
        lambda_q=config.lambda_q if config.train_progress else 0.0,
        lambda_stop=config.lambda_m if config.train_stop else 0.0,
        config=KeyAtomLossV3Config(),
        lambda_cos=LAMBDA_COS_FIXED,
        frame_weight=batch["frame_weight"],
        pair_edit=batch["pair_edit"],
    )


def _curve_batch_positions(
    samples: Sequence[Any], indices: Sequence[int], *, seed: int,
    rng_state: np.random.RandomState,
) -> List[int]:
    """One curve-balanced batch (8 curve slots x 4 frames) from ``indices``."""
    by_curve_dir: Dict[Tuple[str, str], List[int]] = {}
    for pos, i in enumerate(indices):
        rec = samples[int(i)]
        by_curve_dir.setdefault(
            (rec.curve_id, rec.intent.direction), []).append(pos)
    curves = sorted({samples[int(i)].curve_id for i in indices})
    chosen = list(rng_state.choice(
        curves, size=CURVES_PER_BATCH,
        replace=len(curves) < CURVES_PER_BATCH))
    batch: List[int] = []
    for curve in chosen:
        for direction, count in (("forward", FRAMES_PER_CURVE // 2),
                                 ("reverse", FRAMES_PER_CURVE - FRAMES_PER_CURVE // 2)):
            pool = by_curve_dir.get((curve, direction)) or \
                by_curve_dir.get((curve, "forward")) or \
                by_curve_dir.get((curve, "reverse"))
            if not pool:
                continue
            picks = rng_state.choice(
                pool, size=count, replace=len(pool) < count)
            batch.extend(int(p) for p in np.atleast_1d(picks))
    if not batch:
        raise V52TrainerError("SAMPLER_EMPTY", "no frames drawn")
    return batch


def _inner_val_loss(
    model: KeyAtomV52Model,
    cache: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
    labels: V52LabelTable,
    config: V52TrainerConfig,
    stop_weights: StopClassWeightsV52,
    batch_size: int = 64,
) -> float:
    model.train(False)
    total = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = list(positions[start:start + batch_size])
            batch = to_torch_batch(
                collate_v3([cache[int(p)] for p in chunk]),
                device=config.device)
            loss_out = _loss_on_batch(model, batch, labels, config, stop_weights)
            w = float(batch["frame_weight"].sum().cpu())
            total += float(loss_out["loss"].cpu()) * w
            weight_sum += w
    model.train(True)
    return total / max(weight_sum, 1e-9)


def _save(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    tmp.replace(path)


def evaluate_v52(
    model: KeyAtomV52Model,
    r3_model: Optional[torch.nn.Module],
    dataset: KeyAtomDatasetV3,
    labels: V52LabelTable,
    cache: Sequence[Mapping[str, Any]],
    positions: Sequence[int],
    config: V52TrainerConfig,
) -> Dict[str, Any]:
    """Fold-F (outer-val) metrics: q / stop / Move non-degradation."""
    model.train(False)
    q_preds: List[float] = []
    q_trues: List[float] = []
    q_maskFlags: List[bool] = []
    stop_scores: List[float] = []
    stop_labels: List[int] = []
    stop_maskFlags: List[bool] = []
    curve_ids: List[str] = []
    with torch.no_grad():
        for start in range(0, len(positions), 64):
            chunk = list(positions[start:start + 64])
            batch = to_torch_batch(
                collate_v3([cache[int(p)] for p in chunk]),
                device=config.device)
            out = model(forward_view_v3(batch))
            lab = _labels_for_batch(batch, labels, config.device)
            q_preds.extend(out["q"].cpu().tolist())
            q_trues.extend(lab["q_true"].cpu().tolist())
            q_maskFlags.extend(lab["q_mask"].cpu().tolist())
            stop_scores.extend(torch.sigmoid(
                out["stop_logit"]).cpu().tolist())
            stop_labels.extend(
                (lab["y_stop"] * lab["stop_mask"].float()).cpu().tolist())
            stop_maskFlags.extend(lab["stop_mask"].cpu().tolist())
            curve_ids.extend(str(c) for c in batch["curve_ids"])
    model.train(True)
    q_preds_a = np.asarray(q_preds, dtype=np.float64)
    q_trues_a = np.asarray(q_trues, dtype=np.float64)
    q_mask_a = np.asarray(q_maskFlags, dtype=np.bool_)
    stop_mask_a = np.asarray(stop_maskFlags, dtype=np.bool_)
    # constant-q baseline = train-pool q mean (documented trivial baseline)
    train_q_mean = float(np.mean([
        labels.by_sample[str(cache[p]["sample_id"])].q
        for p in range(len(cache))
        if labels.by_sample[str(cache[p]["sample_id"])].q is not None
    ]))
    per_curve_q: Dict[str, List[int]] = {}
    for i, cid in enumerate(curve_ids):
        per_curve_q.setdefault(cid, []).append(i)
    q_curve_mae: List[float] = []
    q_curve_corr: List[float] = []
    for cid, idxs in sorted(per_curve_q.items()):
        idxs_ok = [i for i in idxs if q_mask_a[i]]
        if not idxs_ok:
            continue
        err = np.abs(q_preds_a[idxs_ok] - q_trues_a[idxs_ok])
        q_curve_mae.append(float(err.mean()))
        c = pearson(q_preds_a[idxs_ok], q_trues_a[idxs_ok])
        if c is not None:
            q_curve_corr.append(c)
    const_mae = float(np.abs(train_q_mean - q_trues_a[q_mask_a]).mean()) \
        if q_mask_a.any() else None
    per_curve_q: Dict[str, List[int]] = {}
    scores_m = np.asarray(stop_scores, dtype=np.float64)[stop_mask_a]
    labels_m = np.asarray(stop_labels, dtype=np.int64)[stop_mask_a]
    preds_m = (scores_m >= 0.5).astype(np.int64)
    tp = int(((preds_m == 1) & (labels_m == 1)).sum())
    fp = int(((preds_m == 1) & (labels_m == 0)).sum())
    fn = int(((preds_m == 0) & (labels_m == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    auc = auc_mann_whitney(scores_m.tolist(), labels_m.tolist())

    # Move non-degradation: same fold-F cache/positions, matched R3 winner
    move = evaluate_checkpoint(
        model, dataset, cache, positions, device=config.device)
    move_baseline = None
    if r3_model is not None:
        move_baseline = evaluate_checkpoint(
            r3_model, dataset, cache, positions, device=config.device)
    guard: Dict[str, Any] = {"policy": "record + guard (<=2% RMSE degradation)"}
    if move_baseline is not None:
        m_rmse = move["key_rmse_over_zero"].get("median_rmse")
        b_rmse = move_baseline["key_rmse_over_zero"].get("median_rmse")
        deg = (None if (m_rmse is None or not b_rmse)
               else 100.0 * (float(m_rmse) - float(b_rmse)) / float(b_rmse))
        guard.update({
            "model_median_rmse": m_rmse,
            "r3_baseline_median_rmse": b_rmse,
            "rmse_degradation_pct": deg,
            "within_rmse_guard": (deg is not None and deg <= 2.0),
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
        "schema": V52_SCHEMA,
        "q_head": {
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
        },
        "stop_head": {
            "precision_at_0.5": precision,
            "recall_at_0.5": recall,
            "auc": auc,
            "n_eval_unmasked": int(labels_m.size),
            "n_eval_pos": int(labels_m.sum()),
            "n_eval_neg": int((labels_m == 0).sum()),
            "mask_excluded": True,
            "above_chance_auc": None if auc is None else auc > 0.5,
        },
        "move_guard_vs_r3_winner": guard,
        "n_fold_eval_samples": len(positions),
    }


def train_run_v52(
    dataset: KeyAtomDatasetV3,
    labels: V52LabelTable,
    config: V52TrainerConfig,
    out_dir: Any,
    *,
    r3_ckpt: Any,
    r3_eval_ckpt: Optional[Any] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Train one V5.2 run; writes checkpoint{_{best,last}}.pt + metrics.json."""
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import set_seed

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_map = set_seed(config.seed)
    device = config.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise V52TrainerError("CUDA_UNAVAILABLE", "device=cuda but no CUDA")

    samples = dataset.samples
    train_idx, val_idx, test_idx = v52_curve_split(samples, config.fold, config.seed)
    stop_weights = stop_class_weights_for_fold(config.fold)
    train_cache = build_feature_cache(
        dataset, train_idx, level="F2", intent_off=False)
    val_cache = build_feature_cache(
        dataset, val_idx, level="F2", intent_off=False)
    test_cache = build_feature_cache(
        dataset, test_idx, level="F2", intent_off=False)
    train_pos = list(range(len(train_idx)))
    val_pos = list(range(len(val_idx)))
    test_pos = list(range(len(test_idx)))

    set_seed(config.seed)  # re-seed AFTER caching so head init is seed-exact
    model = load_backbone_from_r3(r3_ckpt, device=device)
    _set_heads_trainable(model, config.arm)
    _set_encoder_trainable(model, False)          # stage A: encoder frozen
    optimizer = _make_optimizer(model, config)

    start_update = 1
    best_val = float("inf")
    best_update = 0
    last_path = out / "checkpoint_last.pt"
    best_path = out / "checkpoint_best.pt"
    if resume and last_path.is_file():
        state = torch.load(str(last_path), map_location=device, weights_only=False)
        cfg0 = state.get("config", {})
        if (cfg0.get("arm") != config.arm or int(cfg0.get("fold", -1)) != config.fold
                or int(cfg0.get("seed", -1)) != int(config.seed)
                or cfg0.get("backbone_run_id") != config.backbone_run_id):
            raise V52TrainerError(
                "RESUME_IDENTITY_MISMATCH",
                "checkpoint_last belongs to {} f{} s{} init={}; refusing".format(
                    cfg0.get("arm"), cfg0.get("fold"), cfg0.get("seed"),
                    cfg0.get("backbone_run_id")))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_update = int(state["update"]) + 1
        best_val = float(state["best_val"])
        best_update = int(state["best_update"])

    rng = np.random.RandomState(config.seed + 52)
    log_path = out / "train_log.jsonl"
    effective = config.total_updates
    t0 = time.time()
    last_loss: Optional[float] = None
    for update in range(start_update, effective + 1):
        if update == config.freeze_encoder_updates + 1:
            _set_encoder_trainable(model, True)   # stage B: joint
        positions = _curve_batch_positions(samples, train_pos, seed=config.seed,
                                           rng_state=rng)
        batch = to_torch_batch(
            collate_v3([train_cache[int(p)] for p in positions]),
            device=device)
        optimizer.zero_grad(set_to_none=True)
        loss_out = _loss_on_batch(model, batch, labels, config, stop_weights)
        loss_out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.grad_clip)
        optimizer.step()
        last_loss = float(loss_out["loss"].detach().cpu())
        if update % 100 == 0 or update == 1 or update == effective:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "update": update,
                    "phase": ("freeze" if update <= config.freeze_encoder_updates
                              else "joint"),
                    "loss": last_loss,
                    "l_move": float(loss_out["l_move"].detach().cpu()),
                    "l_q": float(loss_out["l_q"].detach().cpu()),
                    "l_ts_stop": float(loss_out["l_ts_stop"].detach().cpu()),
                    "encoder_trainable": any(
                        p.requires_grad for p in model.backbone.parameters()),
                    "elapsed_s": round(time.time() - t0, 3),
                }, sort_keys=True) + "\n")
        if update % max(int(config.val_every), 1) == 0 or update == effective:
            val = _inner_val_loss(
                model, val_cache, val_pos, labels, config, stop_weights)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"update": update, "inner_val_loss": val},
                    sort_keys=True) + "\n")
            if val < best_val:
                best_val = val
                best_update = update
                _save(best_path, _ckpt_payload(
                    model, optimizer, update, best_val, best_update,
                    config, seed_map))
            _save(last_path, _ckpt_payload(
                model, optimizer, update, best_val, best_update,
                config, seed_map))

    if best_path.is_file():
        state = torch.load(str(best_path), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        selected_update = int(state["update"])
    else:
        selected_update = effective
        _save(best_path, _ckpt_payload(
            model, optimizer, selected_update, best_val, selected_update,
            config, seed_map))

    r3_model = None
    if r3_eval_ckpt is not None:
        payload = torch.load(str(r3_eval_ckpt), map_location="cpu",
                             weights_only=False)
        r3_model = StarEGNNV3()
        r3_model.load_state_dict(payload["model"], strict=True)
        r3_model.to(device)
    metrics = evaluate_v52(
        model, r3_model, dataset, labels, test_cache, test_pos, config)
    metrics.update({
        "arm": config.arm,
        "fold": config.fold,
        "seed": config.seed,
        "backbone_run_id": config.backbone_run_id,
        "lambda_q": config.lambda_q,
        "lambda_m": config.lambda_m,
        "schedule": {
            "freeze_encoder_updates": config.freeze_encoder_updates,
            "joint_updates": config.joint_updates,
            "lr_head": config.lr_head, "lr_encoder": config.lr_encoder,
            "optimizer": "AdamW", "weight_decay": config.weight_decay,
            "grad_clip": config.grad_clip, "lambda_cos": LAMBDA_COS_FIXED,
        },
        "stop_class_weights": stop_weights.as_dict(),
        "selected_update": selected_update,
        "best_inner_val_loss": None if best_val == float("inf") else best_val,
        "best_update": best_update,
        "last_train_loss": last_loss,
        "n_train_samples": len(train_idx),
        "n_inner_val_samples": len(val_idx),
        "n_fold_eval_samples_total": len(test_idx),
        "train_folds": sorted({int(samples[i].fold) for i in train_idx}),
        "eval_fold": int(config.fold),
        "selection_metric": "inner-val loss (train folds != eval fold)",
        "seeds": seed_map,
        "device": device,
        "wall_clock_s": round(time.time() - t0, 3),
        "checkpoint_best_sha256": sha256_file(best_path),
        "checkpoint_best_path": str(best_path),
        "head_params": model.head_parameter_counts(),
    })
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    metrics["metrics_path"] = str(out / "metrics.json")
    return metrics


def _ckpt_payload(model, optimizer, update, best_val, best_update,
                  config: V52TrainerConfig, seed_map) -> Dict[str, Any]:
    return {
        "schema": V52_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "best_val": float(best_val),
        "best_update": int(best_update),
        "config": asdict(config),
        "seeds": dict(seed_map),
    }
