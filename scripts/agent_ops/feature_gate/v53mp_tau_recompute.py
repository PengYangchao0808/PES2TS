#!/usr/bin/env python3
"""τ_end offline recalibration gate (plan §9.3; S4 infra proposal 2026-10-23).

PRE-REGISTERED decision-layer change, evaluated OFFLINE on the frozen S3
checkpoints — no training, no model/loss/trainer edits.  For each S3
``v53mp_20k_{A,B,C}_s{17,29,43}`` checkpoint_best:

1. SANITY ANCHOR (fail-closed): rebuild the run's capped inner-val
   population EXACTLY as the trainer derived it (``v53mp_pool_split`` +
   the ``seed ^ 0x1E7A1`` inner-val cap from ``metrics.json``), forward
   the checkpoint, and recompute τ with the OLD formula
   (``calibrate_tau_end``, verbatim trainer import).  The recomputed τ
   must reproduce ``metrics.json endpoint_head.tau_end`` within
   ``--anchor-tol`` (default 1e-6).  ANY anchor failure ⇒ we do NOT
   publish new-τ numbers for ANY run (the harness is wrong, not the
   checkpoint) — anchor diagnostics are written and the process exits 2.
2. NEW FORMULA (plan §9.3, no absolute 0.50 floor):
       τ_prec = existing logic (smallest τ with precision ≥ target;
                max-F1 fallback preserved and reported)
       τ_fpr  = q``--fpr-quantile`` (0.99) of sigmoid(endpoint_logit)
                over the NEGATIVE population (endpoint_mask & y==0) —
                the same population ``calibrate_tau_end`` scores
                against; FPR ≤ 1% by construction
       τ_new  = max(τ_prec, τ_fpr)
       support(τ_new) = #{positive rows with score ≥ τ_new}
       positive curves = distinct curve_ids among positive rows
       status = "insufficient_support" (No-Move disabled) when
                support < 30 OR positive curves < 10, else "ok"
3. MIS-STOP RE-EVALUATION: forward the frozen dev positions (135,926
   for the 20k tier, ascending dataset order — the driver's
   ``dev_positions``) and report stop-rate / start-misstop / ts-misstop
   under τ_old vs τ_new with the EXACT formulas of ``evaluate_v53mp``
   (key_atom_v53mp_train_v3.py:1269-1290: ``pred_stop = sigmoid(log) >=
   tau`` over ``keep = endpoint_mask & y ∈ {0,1}`` — NOTE: the endpoint
   mis-stop block carries NO label_status filter; that filter belongs to
   the MOVE metrics only).  Plan target: the s29 0.63 mis-stop must drop
   to ≤1%.

Feature path mirrors ``_calibrate_tau_on``: one bounded ``_FeatureLRU``
(capacity ~4096) shared by the calibration and dev passes, warmed with
the fail-open fork prefetcher (``maybe_create_prefetcher``, flag file
``outputs/pesn/v53mp/reports/feature_workers.on``) in chunks of 256,
then consumed in batches of 64.  Checkpoints are loaded via
``load_v53mp_checkpoint`` (rollout runner, strict state_dict) and freed
after each run.

Outputs ``<out>.json`` (per-run rows + summary + config echo + code
commit + dataset/labels/checkpoint sha provenance) and an ``.md``
sibling (operator-facing table, framed as a pre-registered
decision-layer change proposal per plan §9.3: does NOT touch training
or scaling comparability; requires sign-off before S4).

Usage::

    python3 scripts/agent_ops/feature_gate/v53mp_tau_recompute.py \
        --tables-dir outputs/pesn/v53mp/data \
        --split outputs/pesn/v53mp/data/split_manifest_v53mp.json \
        --runs-root outputs/pesn/v53mp --tier 20k \
        --arms A,B,C --seeds 17,29,43 --device cpu \
        --out outputs/pesn/v53mp/reports/tau_end_recompute_20261023.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3  # noqa: E402
from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    collate_v3,
    to_torch_batch,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    LABEL_STATUSES,
    ShardedKeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_label_cache_v53mp import (  # noqa: E402
    LabelCacheError,
    load_label_cache,
    save_label_cache,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    build_endpoint_labels_spec_v1,
)
from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (  # noqa: E402
    maybe_create_prefetcher,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    FEATURE_LEVEL_V53MP,
    _FeatureLRU,
    _endpoint_for_batch,
    _prefetch_fill,
    _sigmoid,
    calibrate_tau_end,
    v53mp_pool_split,
)
from pes2ts_core.utils.hashing import sha256_file  # noqa: E402
from scripts.agent_ops.feature_gate.v53mp_rollout_exec import (  # noqa: E402
    load_v53mp_checkpoint,
)
from scripts.campaigns.r3_v53mp_run import (  # noqa: E402
    GEOMETRY_CACHE_CURVES,
    _subset_curve_ids,
    _tier_pool_curve_ids,
)

SCHEMA_V53MP: str = "pesn_v53mp_tau_recompute_v1"
GATE_ID: str = "V53MP-tau-end-recalibration"

#: Anchor tolerance on |τ_recomputed − τ_metrics| (task spec: 1e-6).
DEFAULT_ANCHOR_TOL: float = 1e-6
#: FPR quantile for τ_fpr (plan §9.3: FPR ≤ 1% ⇒ q99).
DEFAULT_FPR_QUANTILE: float = 0.99
#: Support gates (plan §9.3): support < 30 or positive curves < 10 ⇒
#: status="insufficient_support" (No-Move disabled downstream).
DEFAULT_MIN_SUPPORT: int = 30
DEFAULT_MIN_POS_CURVES: int = 10
#: Batch / cache discipline: mirror _calibrate_tau_on (64) and keep the
#: single feature LRU bounded (task spec: ~4096).
BATCH_SIZE: int = 64
FEATURE_LRU_CAPACITY: int = 4096
PREFETCH_CHUNK: int = 256


def log(message: str) -> None:
    print("[{}] {}".format(time.strftime("%H:%M:%S"), message), flush=True)


# ---------------------------------------------------------------------------
# pure math (unit-tested)
# ---------------------------------------------------------------------------


def sigmoid_scores(logits: Sequence[float]) -> np.ndarray:
    """Sigmoid over float64 logits — bitwise the trainer's ``_sigmoid``."""
    return _sigmoid(np.asarray(logits, dtype=np.float64))


def calibrate_tau_old(
    logits: Sequence[float],
    y_true: Sequence[int],
    mask: Sequence[bool],
    *,
    target_precision: float,
) -> Dict[str, Any]:
    """OLD τ formula: verbatim trainer ``calibrate_tau_end`` (imported)."""
    return calibrate_tau_end(
        logits, y_true, mask, target_precision=target_precision)


def tau_recalibrate(
    logits: Sequence[float],
    y_true: Sequence[int],
    mask: Sequence[bool],
    *,
    target_precision: float = 0.9,
    fpr_quantile: float = DEFAULT_FPR_QUANTILE,
    min_support: int = DEFAULT_MIN_SUPPORT,
    min_pos_curves: int = DEFAULT_MIN_POS_CURVES,
    curve_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """NEW τ formula (plan §9.3): τ_new = max(τ_prec, τ_fpr), no 0.5 floor.

    ``τ_prec`` reuses the existing calibration logic (smallest τ achieving
    ``precision >= target_precision`` on the unmasked {0,1} rows, max-F1
    fallback preserved and reported).  ``τ_fpr`` is the
    ``fpr_quantile``-quantile of ``sigmoid(logit)`` over the NEGATIVE
    population (endpoint_mask & y==0 — the same population
    ``calibrate_tau_end`` scores against).  ``curve_ids`` (aligned with
    the logits) feeds the positive-curves support gate; omitting it skips
    that gate component (reported as null).
    """
    log_a = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    m = np.asarray(mask, dtype=np.bool_)
    keep = m & np.isin(y, (0, 1))
    neg = keep & (y == 0)
    pos = keep & (y == 1)
    scores = sigmoid_scores(log_a)

    old = calibrate_tau_end(
        log_a.tolist(), y.tolist(), m.tolist(),
        target_precision=target_precision)
    tau_prec: Optional[float] = old.get("tau_end")
    tau_fpr: Optional[float] = (
        float(np.quantile(scores[neg], fpr_quantile)) if neg.any() else None)

    candidates = [t for t in (tau_prec, tau_fpr) if t is not None]
    tau_new: Optional[float] = max(candidates) if candidates else None

    binding: Optional[str] = None
    if tau_new is not None:
        if tau_prec is not None and tau_fpr is not None:
            binding = ("both" if tau_prec == tau_fpr
                       else ("tau_prec" if tau_prec > tau_fpr else "tau_fpr"))
        elif tau_prec is not None:
            binding = "tau_prec"
        else:
            binding = "tau_fpr"

    out: Dict[str, Any] = {
        "target_precision": float(target_precision),
        "fpr_quantile": float(fpr_quantile),
        "tau_prec": tau_prec,
        "tau_prec_fallback": old.get("fallback"),
        "tau_prec_achieved_precision": old.get("achieved_precision"),
        "tau_prec_achieved_recall": old.get("achieved_recall"),
        "tau_fpr": tau_fpr,
        "tau_new": tau_new,
        "binding": binding,
        "n_unmasked": int(keep.sum()),
        "n_pos": int(pos.sum()),
        "n_neg": int(neg.sum()),
        "prevalence": (float(y[keep].mean()) if keep.any() else None),
        "support_pos_at_tau_new": None,
        "precision_at_tau_new": None,
        "fpr_at_tau_new": None,
        "n_pos_curves": None,
        "status": "insufficient_support",
    }
    if tau_new is None:
        out["status_reason"] = "no_threshold_available"
        return out

    pred = scores >= float(tau_new)
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    out["support_pos_at_tau_new"] = tp
    out["precision_at_tau_new"] = (tp / (tp + fp)) if (tp + fp) else None
    out["fpr_at_tau_new"] = (fp / int(neg.sum())) if neg.any() else None

    enough_support = tp >= int(min_support)
    enough_curves: Optional[bool] = None
    if curve_ids is not None:
        cids = np.asarray([str(c) for c in curve_ids], dtype=object)
        if cids.shape[0] != log_a.shape[0]:
            raise ValueError(
                "curve_ids length {} != logits length {}".format(
                    cids.shape[0], log_a.shape[0]))
        out["n_pos_curves"] = int(len(set(cids[pos].tolist())))
        enough_curves = out["n_pos_curves"] >= int(min_pos_curves)
    ok = enough_support and (enough_curves is None or enough_curves)
    out["status"] = "ok" if ok else "insufficient_support"
    reasons = []
    if not enough_support:
        reasons.append(
            "support({}) < {}".format(tp, int(min_support)))
    if enough_curves is False:
        reasons.append(
            "positive_curves({}) < {}".format(
                out["n_pos_curves"], int(min_pos_curves)))
    out["status_reason"] = "; ".join(reasons) if reasons else None
    return out


def anchor_check(
    tau_recomputed: Optional[float],
    tau_expected: Optional[float],
    *,
    tol: float = DEFAULT_ANCHOR_TOL,
) -> Dict[str, Any]:
    """Reproduction check of the OLD-formula τ vs the frozen metrics τ."""
    if tau_recomputed is None or tau_expected is None:
        return {
            "reproduced": False,
            "abs_diff": None,
            "tolerance": float(tol),
            "reason": ("recomputed tau is None" if tau_recomputed is None
                       else "expected tau is None"),
        }
    diff = abs(float(tau_recomputed) - float(tau_expected))
    return {
        "reproduced": bool(diff <= float(tol)),
        "abs_diff": float(diff),
        "tolerance": float(tol),
        "reason": None,
    }


def misstop_metrics(
    logits: Sequence[float],
    y_true: Sequence[int],
    mask: Sequence[bool],
    start_rows: Sequence[bool],
    ts_rows: Sequence[bool],
    *,
    tau: Optional[float],
) -> Dict[str, Any]:
    """Dev mis-stop at ``tau`` — EXACT evaluate_v53mp formulas (:1269-1290).

    ``pred_stop = sigmoid(logit) >= tau`` over all rows; rates are means
    over ``keep & start_rows`` / ``keep & ts_rows`` where ``keep =
    endpoint_mask & y ∈ {0,1}``.  The endpoint block has NO
    label_status=="ok" filter (that filter is MOVE-only).
    """
    log_a = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    m = np.asarray(mask, dtype=np.bool_)
    keep = m & np.isin(y, (0, 1))
    start_arr = np.asarray(start_rows, dtype=np.bool_)
    ts_arr = np.asarray(ts_rows, dtype=np.bool_)
    n_start = int(np.sum(keep & start_arr))
    n_ts = int(np.sum(keep & ts_arr))
    out: Dict[str, Any] = {
        "tau_end": (None if tau is None else float(tau)),
        "n_unmasked": int(keep.sum()),
        "n_pos": int((keep & (y == 1)).sum()),
        "n_neg": int((keep & (y == 0)).sum()),
        "n_start_rows": n_start,
        "n_ts_rows": n_ts,
        "start_misstop_rate_at_tau": None,
        "ts_misstop_rate_at_tau": None,
        "stop_rate_overall_at_tau": None,
    }
    if tau is None:
        return out
    pred_stop = sigmoid_scores(log_a) >= float(tau)
    if n_start:
        out["start_misstop_rate_at_tau"] = float(
            pred_stop[keep & start_arr].mean())
    if n_ts:
        out["ts_misstop_rate_at_tau"] = float(
            pred_stop[keep & ts_arr].mean())
    if keep.any():
        out["stop_rate_overall_at_tau"] = float(pred_stop[keep].mean())
    return out


# ---------------------------------------------------------------------------
# logits collection (mirror _calibrate_tau_on :1823-1857)
# ---------------------------------------------------------------------------


def collect_endpoint_logits(
    model: torch.nn.Module,
    dataset: Any,
    lru: Any,
    positions: Sequence[int],
    endpoint_labels: Any,
    *,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
    prefetcher: Any = None,
    want_rows: bool = False,
    label: str = "",
) -> Dict[str, Any]:
    """Forward ``model`` over ``positions``; return endpoint logits/labels.

    Same feature path as training: bounded ``_FeatureLRU`` (fail-open
    prefetch warm per 256 positions), ``collate_v3`` + ``to_torch_batch``
    + ``forward_view_v3`` + ``_endpoint_for_batch``.  When
    ``want_rows`` the per-row ``start_row``/``ts_row`` flags are collected
    too (needed for the mis-stop block).
    """
    model.train(False)
    pos = [int(p) for p in positions]
    logits: List[float] = []
    ys: List[int] = []
    masks: List[bool] = []
    start_rows: List[bool] = []
    ts_rows: List[bool] = []
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for cstart in range(0, len(pos), PREFETCH_CHUNK):
            chunk = pos[cstart:cstart + PREFETCH_CHUNK]
            _prefetch_fill(lru, dataset, chunk, prefetcher)
            for start in range(0, len(chunk), int(batch_size)):
                built = [lru.get(int(p)) for p in chunk[start:start + batch_size]]
                batch = to_torch_batch(collate_v3(built), device=device)
                out = model(forward_view_v3(batch))
                sample_ids = [str(s) for s in batch["sample_ids"]]
                ep = _endpoint_for_batch(sample_ids, endpoint_labels, device)
                logits.extend(out["endpoint_logit"].detach().cpu().tolist())
                ys.extend(
                    (ep["y_endpoint"] * ep["endpoint_mask"].float())
                    .cpu().tolist())
                masks.extend(ep["endpoint_mask"].cpu().tolist())
                if want_rows:
                    start_rows.extend(bool(v) for v in ep["start_row"])
                    ts_rows.extend(bool(v) for v in ep["ts_row"])
                done += len(built)
            if label and done and (cstart + len(chunk) >= len(pos)
                                   or (cstart // PREFETCH_CHUNK) % 20 == 19):
                log("  {}: {}/{} positions in {:.1f}s".format(
                    label, done, len(pos), time.time() - t0))
    model.train(True)
    result: Dict[str, Any] = {
        "logits": logits,
        "y": [int(v) for v in ys],
        "masks": [bool(v) for v in masks],
    }
    if want_rows:
        result["start_rows"] = start_rows
        result["ts_rows"] = ts_rows
    return result


# ---------------------------------------------------------------------------
# run-artifact loading (mirrors v53mp_rollout_exec._load_run_artifact)
# ---------------------------------------------------------------------------


def load_run_artifact(
    runs_root: Path, tier: str, arm: str, seed: int
) -> Dict[str, Any]:
    """checkpoint_best.pt + metrics.json for one run (fail-closed)."""
    run_id = "v53mp_{}_{}_s{}".format(tier, arm, seed)
    run_dir = runs_root / run_id
    ckpt = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    missing = [str(p) for p in (ckpt, metrics_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "{}: missing {} (run not trained yet?)".format(run_id, missing))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ckpt_sha = sha256_file(str(ckpt))
    metrics_sha = metrics.get("checkpoint_best_sha256")
    if metrics_sha and str(metrics_sha) != ckpt_sha:
        raise RuntimeError(
            "{}: metrics checkpoint_best_sha256 {} != actual file sha256 "
            "{}".format(run_id, metrics_sha, ckpt_sha))
    tau_old = (metrics.get("endpoint_head") or {}).get("tau_end")
    tau_block = metrics.get("tau_end_calibration") or {}
    if tau_old is None:
        tau_old = tau_block.get("tau_end")
    return {
        "run_id": run_id,
        "arm": str(arm),
        "seed": int(seed),
        "ckpt_path": str(ckpt),
        "ckpt_sha256": ckpt_sha,
        "metrics_path": str(metrics_path),
        "tau_old": (None if tau_old is None else float(tau_old)),
        "tau_target_precision": float(
            tau_block.get("target_precision") or 0.9),
        "inner_val_max_samples": (
            None if metrics.get("inner_val_max_samples") is None
            else int(metrics["inner_val_max_samples"])),
        "metrics_n_inner_val_unmasked": tau_block.get("n_inner_val_unmasked"),
        "metrics_fallback": tau_block.get("fallback"),
        "metrics_misstop": {
            "start_misstop_rate_at_tau": (metrics.get("endpoint_head")
                                          or {}).get(
                "start_misstop_rate_at_tau"),
            "ts_misstop_rate_at_tau": (metrics.get("endpoint_head")
                                       or {}).get("ts_misstop_rate_at_tau"),
        },
        "metrics_n_dev_samples_total": metrics.get("n_dev_samples_total"),
        "provenance": metrics.get("provenance") or {},
    }


def capped_inner_val_positions(
    samples: Sequence[Any],
    pool_curve_ids: Sequence[str],
    seed: int,
    cap: Optional[int],
) -> List[int]:
    """Capped inner-val positions — mirror trainer :1489-1511 EXACTLY.

    ``v53mp_pool_split`` (default inner_val_fraction=0.2, as both the
    driver and the trainer config use), then the seeded cap
    ``np.random.RandomState(seed ^ 0x1E7A1).choice(list(...), size=cap)``
    sorted — the population the trainer actually calibrated τ_end on.
    """
    _, inner_val_pos = v53mp_pool_split(
        samples, pool_curve_ids=list(pool_curve_ids), seed=int(seed))
    if cap is not None and len(inner_val_pos) > int(cap):
        rng = np.random.RandomState(int(seed) ^ 0x1E7A1)
        inner_val_pos = sorted(rng.choice(
            list(inner_val_pos), size=int(cap), replace=False).tolist())
    return [int(p) for p in inner_val_pos]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(REPO_ROOT), capture_output=True,
            text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# report writers
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "nan"
        return "{:.{d}f}".format(float(value), d=digits)
    return str(value)


def render_markdown(doc: Mapping[str, Any]) -> str:
    """Operator-facing report — PRE-REGISTERED decision-layer proposal."""
    cfg = doc["config"]
    summary = doc["summary"]
    lines: List[str] = []
    lines.append("# τ_end 离线重校准复算（S3 九检查点）— 预注册决策层变更提案")
    lines.append("")
    lines.append("> Gate: `{}` · Schema: `{}` · 生成时间: `{}`".format(
        GATE_ID, doc["schema"], doc["created_at"]))
    lines.append("> 状态: **离线复算完成，待签字** —— 按 plan §9.3，"
                 "本变更**不触碰训练与 scaling 可比性**（模型/损失/"
                 "trainer 代码零改动，仅改部署时的 τ 选择公式）；"
                 "**S4 启动前需要单独签字**。")
    lines.append("")
    lines.append("## 1. 公式（plan §9.3，无绝对 0.50 下界）")
    lines.append("")
    lines.append("```")
    lines.append("τ_prec = 满足 precision ≥ {:.2f} 的最小阈值（现有逻辑，"
                 "含 max-F1 回退）".format(cfg["target_precision"]))
    lines.append("τ_fpr  = 负样本（endpoint_mask & y==0）sigmoid(logit) 的 "
                 "{} 分位数（FPR ≤ 1%）".format(cfg["fpr_quantile"]))
    lines.append("τ_new  = max(τ_prec, τ_fpr)")
    lines.append("support(τ_new) < {} 或 正样本曲线 < {} → "
                 "status = insufficient_support（禁用 No-Move）".format(
                     cfg["min_support"], cfg["min_pos_curves"]))
    lines.append("```")
    lines.append("")
    lines.append("## 2. 逐 run 结果（{} tier）".format(cfg["tier"]))
    lines.append("")
    header = ("| run | τ_old | τ_new | Δ | prec@τ_new | FPR@τ_new | "
              "support | 正样本曲线 | status | start误停 old→new | "
              "ts误停 old→new |")
    lines.append(header)
    lines.append("|" + "---|" * 11)
    for row in doc["runs"]:
        if row.get("status") == "anchor_failed":
            lines.append(
                "| {} | {} | **ANCHOR FAILED** | — | — | — | — | — | "
                "anchor_failed | — | — |".format(
                    row["run_id"], _fmt(row["tau_old_metrics"])))
            continue
        rec = row["recalibration"]
        mis_old = row["misstop"]["at_tau_old"]
        mis_new = row["misstop"]["at_tau_new"]
        delta = None
        if rec["tau_new"] is not None and row["tau_old_metrics"] is not None:
            delta = rec["tau_new"] - row["tau_old_metrics"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | "
                     "{} → {} | {} → {} |".format(
                         row["run_id"],
                         _fmt(row["tau_old_metrics"]),
                         _fmt(rec["tau_new"]),
                         _fmt(delta),
                         _fmt(rec["precision_at_tau_new"]),
                         _fmt(rec["fpr_at_tau_new"]),
                         _fmt(rec["support_pos_at_tau_new"]),
                         _fmt(rec["n_pos_curves"]),
                         rec["status"],
                         _fmt(mis_old["start_misstop_rate_at_tau"], 4),
                         _fmt(mis_new["start_misstop_rate_at_tau"], 4),
                         _fmt(mis_old["ts_misstop_rate_at_tau"], 4),
                         _fmt(mis_new["ts_misstop_rate_at_tau"], 4)))
    lines.append("")
    lines.append("（insufficient_support 行：No-Move 下游禁用，"
                 "τ_new 数字仅供参考，不构成部署建议。）")
    lines.append("")
    lines.append("## 3. Sanity anchor（旧公式复现 τ_old，容差 {}）".format(
        cfg["anchor_tol"]))
    lines.append("")
    lines.append("| run | τ_old (metrics) | τ_old (复算) | |Δ| | "
                 "n_inner_val_unmasked 复算/metrics | 判定 |")
    lines.append("|" + "---|" * 6)
    for row in doc["runs"]:
        anc = row["anchor"]
        lines.append("| {} | {} | {} | {} | {}/{} | {} |".format(
            row["run_id"],
            _fmt(row["tau_old_metrics"]),
            _fmt(row.get("tau_old_recomputed")),
            _fmt(anc.get("abs_diff")),
            _fmt(row.get("anchor_n_unmasked_recomputed")),
            _fmt(row.get("anchor_n_unmasked_metrics")),
            "PASS" if anc.get("reproduced") else "**FAIL**"))
    lines.append("")
    anchor = summary["sanity_anchor"]
    if anchor["all_reproduced"]:
        lines.append("**Anchor 结论: 全部 {} 个 run 复现 τ_old"
                     "（max |Δ| = {}）。复算 harness 通过，"
                     "第 2 节数字有效。**".format(
                         anchor["n_runs"], _fmt(anchor["max_abs_diff"])))
    else:
        lines.append("**Anchor 结论: {}/{} 个 run 未能复现 τ_old —— "
                     "**复算 harness 存在人口/数值偏差，"
                     "第 2 节的新 τ 数字不予采信**（harness 的问题，"
                     "不是 checkpoint 的问题）。不得据此推进 S4 签字。"
                     "**".format(
                         anchor["n_failed"], anchor["n_runs"]))
    lines.append("")
    lines.append("## 4. plan §9.3 目标核对（s29 误停 0.63 → ≤1%）")
    lines.append("")
    target = summary.get("plan_target_s29")
    if target is None:
        lines.append("—（s29 C 臂 run 未包含在本次运行中）")
    else:
        lines.append("- run: `{}`".format(target["run_id"]))
        lines.append("- start 误停 @τ_old: {} → @τ_new: {}（目标 ≤ 1%）"
                     "⇒ {}".format(
                         _fmt(target["start_misstop_old"], 4),
                         _fmt(target["start_misstop_new"], 4),
                         "**达成**" if target["start_met"] else "**未达成**"))
        lines.append("- ts 误停 @τ_old: {} → @τ_new: {}（目标 ≤ 1%）"
                     "⇒ {}".format(
                         _fmt(target["ts_misstop_old"], 4),
                         _fmt(target["ts_misstop_new"], 4),
                         "**达成**" if target["ts_met"] else "**未达成**"))
        lines.append("- status: {}".format(target["status"]))
    lines.append("")
    lines.append("## 5. 治理与出处")
    lines.append("")
    lines.append("- 决策层变更：仅离线复算与部署阈值建议；"
                 "**不改模型/损失/trainer/features_sha256 绑定**，"
                 "训练与 20k↔50k scaling 比较不受影响。")
    lines.append("- **S4 启动前需要对本提案单独签字**；"
                 "insufficient_support 的 run 禁用 No-Move 在线停止。")
    lines.append("- code commit: `{}`{}".format(
        cfg["code_commit"][:12] if cfg["code_commit"] else "unknown",
        " (dirty)" if cfg["code_commit_dirty"] else ""))
    lines.append("- dataset manifest sha256: `{}`".format(
        cfg["dataset_manifest_sha256"]))
    lines.append("- split manifest sha256: `{}`".format(
        cfg["split_manifest_sha256"]))
    lines.append("- label cache: json `{}` / npz `{}`".format(
        cfg["label_cache_json_sha256"][:12],
        cfg["label_cache_npz_sha256"][:12]))
    lines.append("- checkpoint shas: 见 JSON `runs[].ckpt_sha256`"
                 "（已逐 run 与 metrics.json 绑定 sha 校验）")
    lines.append("- 复算人口: 每 seed 的 capped inner-val"
                 "（trainer 实际校准人口，含 `seed ^ 0x1E7A1` cap）；"
                 "dev = 冻结 split 的 {} 个位置（{} tier）".format(
                     cfg["n_dev_positions"], cfg["tier"]))
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v53mp_tau_recompute",
        description="τ_end offline recalibration gate (plan §9.3): "
                    "recompute τ on frozen S3 checkpoints, OLD-formula "
                    "sanity anchor, NEW-formula proposal, dev mis-stop "
                    "re-evaluation. Decision-layer only.")
    parser.add_argument("--tables-dir", default="outputs/pesn/v53mp/data")
    parser.add_argument("--split",
                        default="outputs/pesn/v53mp/data/"
                                "split_manifest_v53mp.json")
    parser.add_argument("--runs-root", default="outputs/pesn/v53mp")
    parser.add_argument("--tier", default="20k")
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out",
                        default="outputs/pesn/v53mp/reports/"
                                "tau_end_recompute_20261023.json")
    parser.add_argument("--anchor-tol", type=float,
                        default=DEFAULT_ANCHOR_TOL)
    parser.add_argument("--fpr-quantile", type=float,
                        default=DEFAULT_FPR_QUANTILE)
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--min-pos-curves", type=int,
                        default=DEFAULT_MIN_POS_CURVES)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    tables_dir = Path(args.tables_dir)
    split_path = Path(args.split)
    runs_root = Path(args.runs_root)

    t_start = time.time()
    log("stage: reading run artifacts (metrics + checkpoint sha binding)")
    artifacts: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            art = load_run_artifact(runs_root, args.tier, arm, seed)
            artifacts.append(art)
            log("  {} τ_old={} ckpt_sha={}..".format(
                art["run_id"], art["tau_old"], art["ckpt_sha256"][:12]))
    # τ population contract must be uniform across the campaign slice.
    caps = {a["inner_val_max_samples"] for a in artifacts}
    targets = {a["tau_target_precision"] for a in artifacts}
    if len(caps) != 1 or len(targets) != 1:
        log("ERROR: heterogeneous inner_val cap / target precision across "
            "runs: caps={} targets={}".format(caps, targets))
        return 1
    inner_val_cap = next(iter(caps))
    target_precision = next(iter(targets))
    log("population contract: inner_val_cap={} target_precision={}".format(
        inner_val_cap, target_precision))

    log("stage: loading dataset+labels (light index; expect ~12 min / "
        "~12 GB anon on first load) — {}".format(tables_dir))
    t0 = time.time()
    dataset = ShardedKeyAtomDatasetV3(
        str(tables_dir),
        split_manifest=str(split_path),
        label_statuses=LABEL_STATUSES,   # ok/masked/terminal/conflict
        light_cache=True,
        cache_curves=int(GEOMETRY_CACHE_CURVES.get(args.tier, 26000)),
    )
    samples = dataset.samples
    log("  dataset ready: {} samples in {:.1f}s (manifest sha {})".format(
        len(samples), time.time() - t0, dataset.manifest_sha256[:12]))
    t0 = time.time()
    try:
        labels_v52, endpoint_labels = load_label_cache(
            tables_dir, dataset_manifest_sha256=dataset.manifest_sha256)
        log("  label cache hit in {:.1f}s".format(time.time() - t0))
    except LabelCacheError as exc:
        log("  label cache unusable ({}); recomputing from frozen "
            "tables...".format(exc))
        labels_v52 = dataset.v52_labels()
        endpoint_labels = build_endpoint_labels_spec_v1(dataset, labels_v52)
        save_label_cache(
            tables_dir, labels_v52, endpoint_labels,
            dataset_manifest_sha256=dataset.manifest_sha256)
    label_json = tables_dir / "labels_cache_v53mp.json"
    label_npz = tables_dir / "labels_cache_v53mp.npz"

    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    pool_curve_ids = _tier_pool_curve_ids(split_doc, args.tier)
    dev_curve_ids = set(_subset_curve_ids(split_doc, "dev"))
    t0 = time.time()
    dev_positions = [
        i for i, s in enumerate(samples) if str(s.curve_id) in dev_curve_ids]
    log("stage: positions — pool {} curves, dev {} curves / {} positions "
        "({:.1f}s)".format(
            len(pool_curve_ids), len(dev_curve_ids), len(dev_positions),
            time.time() - t0))
    if not dev_positions:
        log("ERROR: frozen dev subset has no loaded samples")
        return 1

    # per-seed capped inner-val calibration population (shared by arms)
    calib_positions: Dict[int, List[int]] = {}
    for seed in seeds:
        pos = capped_inner_val_positions(
            samples, pool_curve_ids, seed, inner_val_cap)
        calib_positions[seed] = pos
        log("  seed {}: capped inner-val {} positions (cap={})".format(
            seed, len(pos), inner_val_cap))

    # One bounded feature LRU, shared by calibration and dev passes.
    lru = _FeatureLRU(
        dataset, capacity=FEATURE_LRU_CAPACITY,
        level=FEATURE_LEVEL_V53MP, intent_off=False)
    prefetcher = maybe_create_prefetcher(FEATURE_LEVEL_V53MP, False)
    log("feature prefetcher: {}".format(
        "{} workers".format(prefetcher.workers)
        if (prefetcher is not None and prefetcher.active) else "serial"))

    rows: List[Dict[str, Any]] = []
    anchor_failed = False
    try:
        for art in artifacts:
            run_id = art["run_id"]
            seed = art["seed"]
            log("stage: {} — load ckpt + calibration forward".format(run_id))
            t0 = time.time()
            model = load_v53mp_checkpoint(
                Path(art["ckpt_path"]), device=args.device)
            cal_pos = calib_positions[seed]
            calib = collect_endpoint_logits(
                model, dataset, lru, cal_pos, endpoint_labels,
                device=args.device, prefetcher=prefetcher,
                label=run_id + "/calib")
            curve_ids = [str(samples[int(p)].curve_id) for p in cal_pos]

            # --- 1. SANITY ANCHOR: OLD formula must reproduce τ_old ------
            old = calibrate_tau_old(
                calib["logits"], calib["y"], calib["masks"],
                target_precision=target_precision)
            anchor = anchor_check(
                old.get("tau_end"), art["tau_old"], tol=args.anchor_tol)
            log("  anchor: τ_old metrics={} recomputed={} |Δ|={} "
                "n_unmasked={} (metrics {}) → {}".format(
                    art["tau_old"], old.get("tau_end"),
                    anchor["abs_diff"], old.get("n_inner_val_unmasked"),
                    art["metrics_n_inner_val_unmasked"],
                    "PASS" if anchor["reproduced"] else "FAIL"))
            population_ok = (
                old.get("n_inner_val_unmasked")
                == art["metrics_n_inner_val_unmasked"])
            row: Dict[str, Any] = {
                "run_id": run_id,
                "arm": art["arm"],
                "seed": seed,
                "ckpt_path": art["ckpt_path"],
                "ckpt_sha256": art["ckpt_sha256"],
                "tau_old_metrics": art["tau_old"],
                "tau_old_recomputed": old.get("tau_end"),
                "tau_old_fallback_metrics": art["metrics_fallback"],
                "tau_old_fallback_recomputed": old.get("fallback"),
                "anchor": anchor,
                "anchor_population_ok": bool(population_ok),
                "anchor_n_unmasked_recomputed": old.get(
                    "n_inner_val_unmasked"),
                "anchor_n_unmasked_metrics": art[
                    "metrics_n_inner_val_unmasked"],
                "anchor_prevalence_recomputed": old.get("prevalence"),
                "calibration_timings_s": round(time.time() - t0, 3),
            }
            if not anchor["reproduced"] or not population_ok:
                # FAIL-CLOSED: the harness is wrong, not the checkpoint.
                # No new-τ numbers are computed or published for this run;
                # if ANY run fails the anchor we refuse the whole batch.
                anchor_failed = True
                row["status"] = "anchor_failed"
                rows.append(row)
                del model
                gc.collect()
                log("  ANCHOR FAILED — skipping new-formula evaluation for "
                    "{} (batch is fail-closed)".format(run_id))
                continue

            # --- 2. NEW FORMULA ------------------------------------------
            rec = tau_recalibrate(
                calib["logits"], calib["y"], calib["masks"],
                target_precision=target_precision,
                fpr_quantile=args.fpr_quantile,
                min_support=args.min_support,
                min_pos_curves=args.min_pos_curves,
                curve_ids=curve_ids)
            log("  new formula: τ_prec={} τ_fpr={} τ_new={} binding={} "
                "prec@τ_new={} FPR@τ_new={} support={} curves={} status="
                "{}".format(
                    rec["tau_prec"], rec["tau_fpr"], rec["tau_new"],
                    rec["binding"], rec["precision_at_tau_new"],
                    rec["fpr_at_tau_new"], rec["support_pos_at_tau_new"],
                    rec["n_pos_curves"], rec["status"]))
            row["recalibration"] = rec
            row["status"] = rec["status"]

            # --- 3. MIS-STOP RE-EVALUATION on dev ------------------------
            t0 = time.time()
            dev = collect_endpoint_logits(
                model, dataset, lru, dev_positions, endpoint_labels,
                device=args.device, prefetcher=prefetcher,
                want_rows=True, label=run_id + "/dev")
            mis_old = misstop_metrics(
                dev["logits"], dev["y"], dev["masks"],
                dev["start_rows"], dev["ts_rows"], tau=art["tau_old"])
            mis_new = misstop_metrics(
                dev["logits"], dev["y"], dev["masks"],
                dev["start_rows"], dev["ts_rows"], tau=rec["tau_new"])
            row["misstop"] = {
                "at_tau_old": mis_old,
                "at_tau_new": mis_new,
                "dev_timings_s": round(time.time() - t0, 3),
                "n_dev_positions": len(dev_positions),
                "no_move_enabled_at_tau_new": rec["status"] == "ok",
            }
            log("  dev mis-stop ({} positions, {:.1f}s): start {} → {} | "
                "ts {} → {}".format(
                    len(dev_positions), time.time() - t0,
                    mis_old["start_misstop_rate_at_tau"],
                    mis_new["start_misstop_rate_at_tau"],
                    mis_old["ts_misstop_rate_at_tau"],
                    mis_new["ts_misstop_rate_at_tau"]))
            rows.append(row)
            del model, calib, dev
            gc.collect()
    finally:
        if prefetcher is not None:
            prefetcher.close()

    # --- summary ---------------------------------------------------------
    anchors = [r["anchor"] for r in rows]
    diffs = [a["abs_diff"] for a in anchors
             if a.get("abs_diff") is not None]
    summary: Dict[str, Any] = {
        "sanity_anchor": {
            "all_reproduced": bool(all(a["reproduced"] for a in anchors)),
            "n_runs": len(anchors),
            "n_failed": int(sum(1 for a in anchors if not a["reproduced"])),
            "max_abs_diff": (max(diffs) if diffs else None),
            "tolerance": float(args.anchor_tol),
        },
        "status_counts": {},
    }
    for row in rows:
        key = row.get("status", "unknown")
        summary["status_counts"][key] = summary["status_counts"].get(key, 0) + 1
    target_row = next(
        (r for r in rows
         if r["arm"] == "C" and r["seed"] == 29 and "misstop" in r), None)
    if target_row is not None:
        mo = target_row["misstop"]["at_tau_old"]
        mn = target_row["misstop"]["at_tau_new"]
        summary["plan_target_s29"] = {
            "run_id": target_row["run_id"],
            "start_misstop_old": mo["start_misstop_rate_at_tau"],
            "start_misstop_new": mn["start_misstop_rate_at_tau"],
            "ts_misstop_old": mo["ts_misstop_rate_at_tau"],
            "ts_misstop_new": mn["ts_misstop_rate_at_tau"],
            "target": 0.01,
            "start_met": (mn["start_misstop_rate_at_tau"] is not None
                          and mn["start_misstop_rate_at_tau"] <= 0.01),
            "ts_met": (mn["ts_misstop_rate_at_tau"] is not None
                       and mn["ts_misstop_rate_at_tau"] <= 0.01),
            "status": target_row.get("status"),
        }

    doc: Dict[str, Any] = {
        "schema": SCHEMA_V53MP,
        "gate_id": GATE_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": round(time.time() - t_start, 3),
        "config": {
            "tables_dir": str(tables_dir),
            "split": str(split_path),
            "runs_root": str(runs_root),
            "tier": args.tier,
            "arms": arms,
            "seeds": seeds,
            "device": args.device,
            "out": str(out_path),
            "anchor_tol": float(args.anchor_tol),
            "target_precision": float(target_precision),
            "fpr_quantile": float(args.fpr_quantile),
            "min_support": int(args.min_support),
            "min_pos_curves": int(args.min_pos_curves),
            "inner_val_max_samples": inner_val_cap,
            "batch_size": BATCH_SIZE,
            "feature_lru_capacity": FEATURE_LRU_CAPACITY,
            "n_dev_positions": len(dev_positions),
            "n_pool_curves": len(pool_curve_ids),
            "code_commit": _git("rev-parse", "HEAD"),
            "code_commit_dirty": bool(_git("status", "--porcelain")),
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "split_manifest_sha256": sha256_file(str(split_path)),
            "label_cache_json_sha256": (
                sha256_file(str(label_json)) if label_json.is_file()
                else None),
            "label_cache_npz_sha256": (
                sha256_file(str(label_npz)) if label_npz.is_file()
                else None),
        },
        "runs": rows,
        "summary": summary,
        "claim_scope": (
            "PRE-REGISTERED decision-layer proposal (plan §9.3); "
            "no training/scaling comparability impact; requires sign-off "
            "before S4"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(doc), encoding="utf-8")
    log("wrote {} and {} ({:.1f}s total)".format(
        out_path, md_path, time.time() - t_start))
    log("summary: anchor={} statuses={}".format(
        summary["sanity_anchor"], summary["status_counts"]))
    if anchor_failed or not summary["sanity_anchor"]["all_reproduced"]:
        log("EXIT 2: sanity anchor failed — new-τ numbers are NOT valid; "
            "fix the harness before any S4 sign-off.")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
