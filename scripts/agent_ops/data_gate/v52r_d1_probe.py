#!/usr/bin/env python3
"""V5.2-R stage D1: frozen-encoder Endpoint Head + directed ordering probes.

Implements the D1 section of
``docs/plans/PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md``
against contract ``docs/contracts/PES2TS_FullPathNavigationContract_v1.md``
§5/§8/§9.  Writes ONLY under ``--out`` (default
``outputs/pesn/v52r/d1``): embedding caches, ``evidence_pack.json`` and
``D1_PROBE.md``.  No training, no architecture change, no sealed-fold use —
folds 3/4 are COUNTED only, never encoded/probed (dev folds 0–2 only).

Pipeline (CPU-only, deterministic seed 0):

1. **Embedding extraction** — for each eval fold F ∈ {0,1,2} and seed
   s ∈ {17,29,43}, the SEED-MATCHED R3 winner checkpoint (``checkpoint_best.pt``
   of the fold-matched run; encoder trained on the OTHER two dev folds, so it
   never saw fold F's curves) is loaded into ``StarEGNNV3`` (strict=True) and
   every dev row is forwarded once.  Per row we cache ``z`` (64-d, model
   output) and the key-pooled ``h`` (48-d = key half of the ``z_head`` input
   ``graph-pool ⊕ key-pool``, captured with a forward hook — contract §8
   head-input choice z vs key-pooled h).  Cache files:
   ``embeddings_{fold}_{seed}.npz``.
2. **Tabular features** — pairwise distances among the intent's K atoms
   (audit-20261021 §5 method) padded to the global max |K|, plus the intent
   direction one-hot (2) → ``tabular.npz``.
3. **Endpoint probes per eval fold** (train = OTHER two dev folds, grouped by
   curve; 3 folds × 3 seeds = 9 units, aggregated median + range):
   (a) constant prevalence baseline; (b) logistic on tabular; (c)
   HistGradientBoosting on tabular (class_weight=balanced); (d) logistic on z
   with C ∈ {0.01,0.1,1,10} by inner group-split; (e) MLP(32) on z; (f)
   logistic on key-pooled h.  Train/eval rows carry endpoint_stop_label ∈
   {0,1} (Mask −1 excluded); degenerate-flagged rows (contract §5.3 rule 4)
   are excluded from train AND eval and reported separately.  Threshold is
   PRE-REGISTERED: smallest threshold achieving precision ≥ 0.80 on an inner
   curve-group split of TRAIN; if no threshold reaches it, no positives are
   predicted (thr=∅) — never tuned on eval.
4. **Directed ordering probe** — Ridge z→q (v5.2 q labels; reverse = −q),
   pairwise accuracy within same curve+intent over ALL eval pairs, the
   cross-TS subset (pair straddles the TS node), per-curve distribution, and
   pooled Spearman(q_pred, q_true).
5. **Window sensitivity** for the best z-probe: labels rebuilt at
   0.08/0.10/0.12/0.15 (w_neg 0.30); pre-registered rule = endpoint recall at
   precision ≥ 0.8 AND start FP ≤ 2% AND TS FP ≤ 2% → recommended window
   (table always reported; no AUPRC-based window shopping).
6. **Statistics** — cluster bootstrap by curve (1000 resamples, rng seed 0)
   for the best z-probe's aggregated AUPRC → 95% CI; prevalence reported.
7. **Gate V5.2R-D1-ENDPOINT-HEAD** (honest negative results are valid
   output; everything is written even on FAIL):
   - aggregated best-z-probe AUPRC CI lower bound > prevalence;
   - start false-stop ≤ 2% and TS false-stop ≤ 5% at the pre-registered
     threshold (primary window; TS-FP excludes the 2 degenerate curves and
     reports the with-degenerate caliber separately);
   - directed ordering mean accuracy > 0.65 AND cross-TS > 0.6.

``--strict`` exits 1 when the gate verdict is FAIL.  Torch is imported only
in the extraction phase; ``--extract-only`` / ``--probe-only`` plus
``--embeddings-dir`` decouple extraction (torch) from probing (sklearn).

Usage::

    python scripts/agent_ops/data_gate/v52r_d1_probe.py [--strict]
        [--tables-dir outputs/pesn/30h_campaign/data]
        [--out outputs/pesn/v52r/d1] [--embeddings-dir DIR]
        [--extract-only | --probe-only] [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    DATA_MANIFEST_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    DEGENERATE_TS_CURVES,
    ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM,
    EndpointStopLabelTable,
    build_endpoint_stop_labels,
)

SCHEMA_ID = "pes2ts_v52r_d1_evidence_v1"
GATE_ID = "V5.2R-D1-ENDPOINT-HEAD"
DEFAULT_TABLES_DIR = REPO_ROOT / "outputs/pesn/30h_campaign/data"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/pesn/v52r/d1"
R3_RUNS_DIR = REPO_ROOT / "outputs/pesn/30h_campaign/r3_runs"
CAMPAIGN_CONFIG = REPO_ROOT / "config/campaigns/pes2ts_v52r_20261021.yaml"

DEV_FOLDS = (0, 1, 2)
SEALED_FOLDS = (3, 4)
SEEDS = (17, 29, 43)
PROBE_SEED = 0
Z_DIM = 64
H_KEY_DIM = 48
W_NEG = ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM      # 0.30 (frozen)
PRIMARY_W_POS = 0.10
WINDOW_GRID = (0.08, 0.10, 0.12, 0.15)
PRECISION_TARGET = 0.80
C_GRID = (0.01, 0.1, 1.0, 10.0)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
START_FP_LIMIT = 0.02
TS_FP_LIMIT_GATE = 0.05
TS_FP_LIMIT_WINDOW = 0.02
ORDERING_MEAN_LIMIT = 0.65
ORDERING_CROSS_TS_LIMIT = 0.60
RIDGE_ALPHA = 1.0
MLP_MAX_ITER = 300
MLP_TIME_BUDGET_S = 600.0

#: eval fold → seed → R3 winner run id (arm F2-bidir fold-matched checkpoints)
FOLD_SEED_RUNS: dict[int, dict[int, str]] = {
    0: {17: "5372d9b12807c3f0", 29: "1e01499b933d4ad1", 43: "9e2ef3ef468db0af"},
    1: {17: "cc9340b86592b099", 29: "0629b5979df6a2df", 43: "c09e9cd601bafb88"},
    2: {17: "0ebd4c92c85f5601", 29: "dedd62d588f3b6cd", 43: "b3941e53f44b962b"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _r6(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def _median(vals: Iterable[float]) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    return None if not vals else float(np.median(vals))


def _minmax(vals: Iterable[float]) -> dict[str, float | None]:
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"median": None, "min": None, "max": None}
    return {"median": _r6(np.median(vals)), "min": _r6(min(vals)),
            "max": _r6(max(vals))}


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def pick_precision_threshold(
    scores: np.ndarray, labels: np.ndarray, precision_target: float
) -> float | None:
    """PRE-REGISTERED threshold rule: smallest score threshold whose
    precision (score ≥ thr) on the given (inner-val) data is ≥ target.

    Returns None when NO threshold attains the target — the caller must then
    predict nothing positive (never silently relax the target).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.size == 0 or int(labels.sum()) == 0:
        return None
    for thr in np.unique(scores):                 # ascending
        pred = scores >= thr
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        if tp + fp > 0 and tp / (tp + fp) >= float(precision_target):
            return float(thr)
    return None


def pairwise_ordering_accuracy(
    q_true: np.ndarray, q_pred: np.ndarray, nodes: np.ndarray,
    ts_index: int,
) -> dict[str, Any]:
    """Directed pairwise ordering within ONE (curve, intent) row group.

    All i<j pairs with a strict q_true difference count; a pair is correct
    when ``sign(q_pred_j − q_pred_i) == sign(q_true_j − q_true_i)``.  The
    cross-TS subset contains pairs whose node indices straddle ``ts_index``
    (``(node_i − ts)·(node_j − ts) < 0``; the TS node itself is excluded).
    """
    q_true = np.asarray(q_true, dtype=np.float64)
    q_pred = np.asarray(q_pred, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.int64)
    n = q_true.size
    if n < 2:
        return {"n_pairs": 0, "n_correct": 0, "accuracy": None,
                "n_pairs_cross_ts": 0, "n_correct_cross_ts": 0,
                "cross_ts_accuracy": None}
    iu = np.triu_indices(n, k=1)
    dq_t = q_true[iu[1]] - q_true[iu[0]]
    dq_p = q_pred[iu[1]] - q_pred[iu[0]]
    strict = dq_t != 0.0
    correct = (dq_p * dq_t) > 0.0
    straddle = ((nodes[iu[0]] - int(ts_index))
                * (nodes[iu[1]] - int(ts_index))) < 0
    strict_straddle = strict & straddle
    n_pairs = int(strict.sum())
    n_correct = int((strict & correct).sum())
    n_cs = int(strict_straddle.sum())
    n_cs_correct = int((strict_straddle & correct).sum())
    return {
        "n_pairs": n_pairs,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_pairs) if n_pairs else None,
        "n_pairs_cross_ts": n_cs,
        "n_correct_cross_ts": n_cs_correct,
        "cross_ts_accuracy": (n_cs_correct / n_cs) if n_cs else None,
    }


def pad_tabular_features(
    per_row_pair_dists: Sequence[Sequence[float]],
    directions: Sequence[str],
    max_pairs: int,
) -> np.ndarray:
    """[N, max_pairs+2] tabular matrix: padded K-pair distances ⊕ one-hot."""
    out = np.zeros((len(per_row_pair_dists), int(max_pairs) + 2),
                   dtype=np.float64)
    for i, (dists, direction) in enumerate(
            zip(per_row_pair_dists, directions)):
        d = np.asarray(dists, dtype=np.float64).reshape(-1)
        if d.size > max_pairs:
            raise ValueError(
                f"row {i} has {d.size} pair distances > max_pairs {max_pairs}")
        out[i, :d.size] = d
        if direction == "forward":
            out[i, max_pairs] = 1.0
        elif direction == "reverse":
            out[i, max_pairs + 1] = 1.0
        else:
            raise ValueError(f"row {i}: unknown direction {direction!r}")
    return out


def k_pair_distances(frame_xyz: np.ndarray, key_rows: Sequence[int]) -> np.ndarray:
    """Pairwise distances among the intent K atoms, K-order i<j (audit §5)."""
    keys = np.asarray(key_rows, dtype=np.int64).reshape(-1)
    pts = np.asarray(frame_xyz, dtype=np.float64)[keys]
    if pts.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    iu = np.triu_indices(keys.size, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    return np.sqrt((diff * diff).sum(axis=1))


def curve_cluster_bootstrap_auprc(
    scores: np.ndarray, labels: np.ndarray, curve_ids: Sequence[str],
    *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Cluster bootstrap by curve for pooled AUPRC (rng seed fixed)."""
    from sklearn.metrics import average_precision_score

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    curve_ids = np.asarray(curve_ids, dtype=object)
    curves = np.array(sorted(set(curve_ids.tolist())))
    point = float(average_precision_score(labels, scores)) \
        if labels.size else None
    prevalence = float(labels.mean()) if labels.size else None
    rng = np.random.RandomState(int(seed))
    stats: list[float] = []
    for _ in range(int(n_resamples)):
        picked = rng.choice(curves, size=curves.size, replace=True)
        parts = [np.flatnonzero(curve_ids == c) for c in picked]
        idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        if idx.size == 0:
            continue
        y = labels[idx]
        if y.min() == y.max():          # AP undefined for one-class resample
            continue
        stats.append(float(average_precision_score(y, scores[idx])))
    if not stats:
        ci = {"lower": None, "upper": None}
    else:
        ci = {"lower": _r6(np.percentile(stats, 2.5)),
              "upper": _r6(np.percentile(stats, 97.5))}
    return {
        "point": _r6(point),
        "prevalence": _r6(prevalence),
        "ci95": ci,
        "n_resamples": int(n_resamples),
        "n_valid_resamples": len(stats),
        "n_curves": int(curves.size),
        "seed": int(seed),
    }


def spearman_corr(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation (average ranks for ties); None if degenerate."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 2:
        return None

    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(v.size, dtype=np.float64)
        ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
        sv = v[order]
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_gate(
    bootstrap: dict[str, Any],
    false_stop: dict[str, Any],
    ordering: dict[str, Any],
) -> dict[str, Any]:
    """Gate V5.2R-D1-ENDPOINT-HEAD (plan §3 D1 / §5 acceptance table)."""
    ci_low = bootstrap.get("ci95", {}).get("lower")
    prevalence = bootstrap.get("prevalence")
    checks = [
        {
            "id": "auprc_ci_lower_above_prevalence",
            "expect": "best z-probe aggregated AUPRC 95% CI lower bound "
                      "> prevalence",
            "actual": f"ci_low={ci_low} prevalence={prevalence}",
            "pass": (ci_low is not None and prevalence is not None
                     and float(ci_low) > float(prevalence)),
        },
        {
            "id": "start_false_stop_le_2pct",
            "expect": f"start false-stop ≤ {START_FP_LIMIT:.0%} at "
                      "pre-registered threshold",
            "actual": f"pooled={false_stop.get('start_pooled_rate')} "
                      f"(median unit={false_stop.get('start_median_rate')})",
            "pass": (false_stop.get("start_pooled_rate") is not None
                     and float(false_stop["start_pooled_rate"])
                     <= START_FP_LIMIT),
        },
        {
            "id": "ts_false_stop_le_5pct",
            "expect": f"TS false-stop ≤ {TS_FP_LIMIT_GATE:.0%} (excl. "
                      "degenerate curves) at pre-registered threshold",
            "actual": f"pooled={false_stop.get('ts_pooled_rate')} "
                      f"(median unit={false_stop.get('ts_median_rate')})",
            "pass": (false_stop.get("ts_pooled_rate") is not None
                     and float(false_stop["ts_pooled_rate"])
                     <= TS_FP_LIMIT_GATE),
        },
        {
            "id": "ordering_mean_above_065",
            "expect": f"directed ordering mean accuracy > {ORDERING_MEAN_LIMIT}",
            "actual": f"mean={ordering.get('accuracy_mean')} "
                      f"(min={ordering.get('accuracy_min')})",
            "pass": (ordering.get("accuracy_mean") is not None
                     and float(ordering["accuracy_mean"])
                     > ORDERING_MEAN_LIMIT),
        },
        {
            "id": "ordering_cross_ts_above_060",
            "expect": f"cross-TS ordering accuracy > {ORDERING_CROSS_TS_LIMIT}",
            "actual": f"mean={ordering.get('cross_ts_accuracy_mean')} "
                      f"(min={ordering.get('cross_ts_accuracy_min')})",
            "pass": (ordering.get("cross_ts_accuracy_mean") is not None
                     and float(ordering["cross_ts_accuracy_mean"])
                     > ORDERING_CROSS_TS_LIMIT),
        },
    ]
    return {"gate_id": GATE_ID, "checks": checks,
            "verdict": "PASS" if all(c["pass"] for c in checks) else "FAIL"}


# ---------------------------------------------------------------------------
# phase 1: embedding extraction (torch)
# ---------------------------------------------------------------------------


def extract_embeddings(
    dataset: KeyAtomDatasetV3,
    dev_positions: Sequence[int],
    out_dir: Path,
    runs_dir: Path,
) -> list[dict[str, Any]]:
    """Forward every dev row through each fold-matched encoder; cache npz."""
    import torch
    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        collate_v3, to_torch_batch,
    )
    from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
        build_feature_cache, set_seed,
    )

    torch.set_num_threads(max(1, torch.get_num_threads()))
    set_seed(PROBE_SEED)
    records: list[dict[str, Any]] = []
    cache = build_feature_cache(
        dataset, list(dev_positions), level="F2", intent_off=False)
    sample_ids = np.array(
        [dataset.samples[int(i)].sample_id for i in dev_positions],
        dtype=object)
    curve_ids = np.array(
        [dataset.samples[int(i)].curve_id for i in dev_positions],
        dtype=object)
    folds = np.array(
        [int(dataset.samples[int(i)].fold) for i in dev_positions],
        dtype=np.int64)

    for fold in DEV_FOLDS:
        for seed in SEEDS:
            run_id = FOLD_SEED_RUNS[fold][seed]
            ckpt = runs_dir / run_id / "checkpoint_best.pt"
            if not ckpt.is_file():
                raise FileNotFoundError(f"missing checkpoint {ckpt}")
            payload = torch.load(str(ckpt), map_location="cpu",
                                 weights_only=False)
            if "model" not in payload:
                raise RuntimeError(f"{ckpt} has no 'model' state dict")
            cfg = payload.get("config", {})
            if cfg.get("fold") is not None and int(cfg["fold"]) != fold:
                raise RuntimeError(
                    f"{ckpt}: checkpoint fold {cfg['fold']} != expected {fold}")
            if cfg.get("seed") is not None and int(cfg["seed"]) != seed:
                raise RuntimeError(
                    f"{ckpt}: checkpoint seed {cfg['seed']} != expected {seed}")
            model = StarEGNNV3()
            model.load_state_dict(payload["model"], strict=True)
            model.train(False)

            captured: dict[str, np.ndarray] = {}

            def _hook(_module, inputs, _output):     # z_head input = g⊕k pool
                captured["h"] = inputs[0].detach().cpu().numpy()

            handle = model.z_head.register_forward_hook(_hook)
            zs: list[np.ndarray] = []
            hs: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(cache), 64):
                    batch = to_torch_batch(
                        collate_v3(cache[start:start + 64]), device="cpu")
                    out = model(forward_view_v3(batch))
                    zs.append(out["z"].cpu().numpy())
                    pooled = captured["h"]      # [G, 96] = g-pool ⊕ k-pool
                    hs.append(pooled[:, H_KEY_DIM:].copy())   # key-pooled h
            handle.remove()
            z = np.concatenate(zs, axis=0).astype(np.float32)
            h = np.concatenate(hs, axis=0).astype(np.float32)
            if z.shape != (len(cache), Z_DIM) or h.shape != (len(cache), H_KEY_DIM):
                raise RuntimeError(
                    f"fold {fold} seed {seed}: bad embedding shapes "
                    f"z={z.shape} h={h.shape}")
            out_path = out_dir / f"embeddings_{fold}_{seed}.npz"
            np.savez_compressed(
                out_path, z=z, h=h, sample_ids=sample_ids,
                curve_ids=curve_ids, folds=folds)
            records.append({
                "fold": fold, "seed": seed, "run_id": run_id,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": sha256_file(ckpt),
                "checkpoint_config_fold": cfg.get("fold"),
                "checkpoint_config_seed": cfg.get("seed"),
                "checkpoint_config_arm": cfg.get("arm"),
                "embeddings_file": str(out_path),
                "embeddings_sha256": sha256_file(out_path),
                "n_rows": int(z.shape[0]),
            })
    return records


def build_tabular(
    dataset: KeyAtomDatasetV3, dev_positions: Sequence[int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Global tabular matrix over dev rows (K-pair distances + one-hot)."""
    per_row: list[np.ndarray] = []
    directions: list[str] = []
    max_pairs = 0
    for i in dev_positions:
        rec = dataset.samples[int(i)]
        keys = rec.intent.key_rows()
        frame = np.asarray(dataset.frame_xyz(rec.curve_id, rec.node_index),
                           dtype=np.float64)
        d = k_pair_distances(frame, keys)
        per_row.append(d)
        directions.append(rec.intent.direction)
        max_pairs = max(max_pairs, int(d.size))
    matrix = pad_tabular_features(per_row, directions, max_pairs)
    info = {
        "max_pairs": max_pairs,
        "n_features": int(matrix.shape[1]),
        "composition": f"{max_pairs} K-pair distances + 2 direction one-hot",
    }
    return matrix, info


# ---------------------------------------------------------------------------
# phase 2: probing (sklearn)
# ---------------------------------------------------------------------------


def _fit_probe(name: str, x_train: np.ndarray, y_train: np.ndarray,
               seed: int = PROBE_SEED):
    """Fit one endpoint probe; returns (proba_fn(X)->P(y=1), extra)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, penalty="l2", max_iter=2000,
                               random_state=seed))
        model.fit(x_train, y_train)
        return (lambda x: model.predict_proba(x)[:, 1]), {"C": 1.0}
    if name == "hgb":
        model = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed)
        model.fit(x_train, y_train)
        return (lambda x: model.predict_proba(x)[:, 1]), {}
    if name == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True,
                          max_iter=MLP_MAX_ITER, random_state=seed))
        model.fit(x_train, y_train)
        return (lambda x: model.predict_proba(x)[:, 1]), {}
    raise ValueError(f"unknown probe {name!r}")


def _select_c(x_train: np.ndarray, y_train: np.ndarray,
              groups: np.ndarray) -> float:
    """C by inner curve-group split (max inner AUPRC; ties → smaller C)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                            random_state=PROBE_SEED)
    tr, va = next(gss.split(x_train, y_train, groups=groups))
    if np.unique(y_train[va]).size < 2:
        return C_GRID[0]
    best_c, best_ap = C_GRID[0], -np.inf
    for c in C_GRID:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(c), penalty="l2", max_iter=2000,
                               random_state=PROBE_SEED))
        model.fit(x_train[tr], y_train[tr])
        ap = average_precision_score(
            y_train[va], model.predict_proba(x_train[va])[:, 1])
        if ap > best_ap + 1e-12:
            best_c, best_ap = float(c), ap
    return float(best_c)


def _inner_split_threshold(
    x_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray,
    probe: str, c_value: float | None,
) -> tuple[float | None, dict[str, Any]]:
    """Pre-registered threshold from an inner curve-group split of TRAIN."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                            random_state=PROBE_SEED)
    tr, va = next(gss.split(x_train, y_train, groups=groups))
    info: dict[str, Any] = {
        "inner_val_rows": int(va.size),
        "inner_val_pos": int(y_train[va].sum()),
        "inner_val_curves": int(np.unique(groups[va]).size),
    }
    if np.unique(y_train[va]).size < 2 or int(y_train[va].sum()) == 0:
        info["threshold"] = None
        info["reason"] = "inner-val lacks positives"
        return None, info
    if probe == "logistic_z":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(c_value), penalty="l2", max_iter=2000,
                               random_state=PROBE_SEED))
        model.fit(x_train[tr], y_train[tr])
        scores = model.predict_proba(x_train[va])[:, 1]
    else:
        fit_name = {"mlp_z": "mlp", "hgb_tab": "hgb",
                    "logistic_tab": "logistic", "logistic_h": "logistic",
                    "hgb": "hgb", "mlp": "mlp",
                    "logistic": "logistic"}[probe]
        score_fn, _ = _fit_probe(fit_name, x_train[tr], y_train[tr])
        scores = score_fn(x_train[va])
    thr = pick_precision_threshold(scores, y_train[va], PRECISION_TARGET)
    info["threshold"] = thr
    if thr is None:
        info["reason"] = (
            f"no threshold reaches precision ≥ {PRECISION_TARGET} on "
            "inner-val; predict-nothing policy")
    return thr, info


def _classification_metrics(
    scores: np.ndarray, labels: np.ndarray, thr: float | None
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    auroc = float(roc_auc_score(labels, scores)) if n_pos and n_neg else None
    auprc = float(average_precision_score(labels, scores)) if n_pos else None
    precision = recall = None
    if thr is not None:
        pred = scores >= thr
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        precision = (tp / (tp + fp)) if (tp + fp) else None
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    return {
        "auroc": _r6(auroc),
        "auprc": _r6(auprc),
        "prevalence": _r6(labels.mean()) if labels.size else None,
        "n_eval": int(labels.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": _r6(thr),
        "precision_at_threshold": _r6(precision),
        "recall_at_threshold": _r6(recall),
    }


def _false_stop_rates(
    eval_rows: dict[str, np.ndarray], scores: np.ndarray, thr: float | None,
    lab_by_sample: dict[str, Any], start_samples: Sequence[str],
    ts_samples: Sequence[str], ts_samples_degenerate: Sequence[str],
) -> dict[str, Any]:
    """Start / TS false-stop rates at the pre-registered threshold."""
    sids = list(eval_rows["sample_ids"])
    index = {sid: i for i, sid in enumerate(sids)}

    def _rate(samples: Sequence[str]) -> tuple[int, int, float | None]:
        rows = [index[s] for s in samples if s in index]
        if thr is None or not rows:
            return 0, len(rows), (0.0 if rows else None)
        flagged = int((scores[np.asarray(rows)] >= thr).sum())
        return flagged, len(rows), flagged / len(rows)

    s_flag, s_n, s_rate = _rate(start_samples)
    t_flag, t_n, t_rate = _rate(ts_samples)
    d_flag, d_n, d_rate = _rate(ts_samples_degenerate)
    return {
        "threshold": _r6(thr),
        "start": {"flagged": s_flag, "n_rows": s_n, "rate": _r6(s_rate)},
        "ts": {"flagged": t_flag, "n_rows": t_n, "rate": _r6(t_rate)},
        "ts_degenerate_curves": {
            "flagged": d_flag, "n_rows": d_n, "rate": _r6(d_rate),
            "curves": list(DEGENERATE_TS_CURVES)},
    }


def run_endpoint_probes(
    features: dict[str, np.ndarray],
    labels: EndpointStopLabelTable,
    meta: dict[str, np.ndarray],
    probe_names: Sequence[str] = ("constant", "logistic_tab", "hgb_tab",
                                  "logistic_z", "mlp_z", "logistic_h"),
    mlp_time_state: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Run the 9 (fold × seed) probe units; returns flat unit records."""
    if mlp_time_state is None:
        mlp_time_state = {"spent": 0.0}
    z_by_unit = features["z"]                  # keyed by (fold, seed)
    h_by_unit = features["h"]
    units: list[dict[str, Any]] = []
    folds = meta["folds"]
    sample_ids = meta["sample_ids"]
    curve_ids = meta["curve_ids"]
    tab = features["tabular"]

    lab_vec = np.array([
        labels.by_sample[sid].endpoint_stop_label
        for sid in sample_ids], dtype=np.int64)
    deg_vec = np.array([
        bool(labels.by_sample[sid].degenerate_flag)
        for sid in sample_ids], dtype=np.bool_)
    usable = (lab_vec >= 0) & (~deg_vec)

    for fold in DEV_FOLDS:
        train_mask = (folds != fold)
        eval_mask = (folds == fold)
        # start / TS row bookkeeping over ALL eval-fold rows (false-stop is a
        # score-threshold event, independent of the label mask)
        eval_sids = [str(s) for s in sample_ids[eval_mask]]
        start_samples: list[str] = []
        ts_samples: list[str] = []
        ts_deg: list[str] = []
        for sid in eval_sids:
            lab = labels.by_sample[sid]
            csum = labels.curves[lab.curve_id]
            node = int(lab.node_index)
            if ((lab.direction == "forward" and node == csum.endpoint_a_node)
                    or (lab.direction == "reverse"
                        and node == csum.endpoint_b_node)):
                start_samples.append(sid)
            if node == csum.ts_index:
                if lab.curve_id in DEGENERATE_TS_CURVES:
                    ts_deg.append(sid)
                else:
                    ts_samples.append(sid)

        for seed in SEEDS:
            z = z_by_unit[(fold, seed)]
            h = h_by_unit[(fold, seed)]
            tr = train_mask & usable
            ev = eval_mask & usable
            y_tr = lab_vec[tr]
            y_ev = lab_vec[ev]
            groups_tr = curve_ids[tr]

            record: dict[str, Any] = {
                "fold": fold, "seed": seed,
                "n_train": int(tr.sum()), "n_eval": int(ev.sum()),
                "train_prevalence": _r6(y_tr.mean()) if y_tr.size else None,
                "eval_prevalence": _r6(y_ev.mean()) if y_ev.size else None,
                "probes": {},
            }

            c_value = None
            if "logistic_z" in probe_names:
                c_value = _select_c(z[tr], y_tr, groups_tr)
                record["logistic_z_best_C"] = c_value

            for probe in probe_names:
                if probe == "constant":
                    prev = float(y_tr.mean()) if y_tr.size else 0.0
                    score_all = np.full(int(eval_mask.sum()), prev)
                    thr = None
                    thr_info = {"threshold": None,
                                "reason": "constant baseline (no threshold)"}
                elif probe == "logistic_z":
                    score_fn, _ = _fit_probe("logistic", z[tr], y_tr)
                    thr, thr_info = _inner_split_threshold(
                        z[tr], y_tr, groups_tr, "logistic_z", c_value)
                    score_all = np.asarray(score_fn(z[eval_mask]))
                elif probe == "mlp_z":
                    if mlp_time_state["spent"] > MLP_TIME_BUDGET_S:
                        record["probes"]["mlp_z"] = {
                            "skipped": True,
                            "reason": f"MLP time budget {MLP_TIME_BUDGET_S}s "
                                      "exhausted"}
                        continue
                    t0 = time.time()
                    score_fn, _ = _fit_probe("mlp", z[tr], y_tr)
                    thr, thr_info = _inner_split_threshold(
                        z[tr], y_tr, groups_tr, "mlp_z", None)
                    mlp_time_state["spent"] += time.time() - t0
                    score_all = np.asarray(score_fn(z[eval_mask]))
                elif probe == "logistic_h":
                    score_fn, _ = _fit_probe("logistic", h[tr], y_tr)
                    thr, thr_info = _inner_split_threshold(
                        h[tr], y_tr, groups_tr, "logistic", None)
                    score_all = np.asarray(score_fn(h[eval_mask]))
                elif probe == "logistic_tab":
                    score_fn, _ = _fit_probe("logistic", tab[tr], y_tr)
                    thr, thr_info = _inner_split_threshold(
                        tab[tr], y_tr, groups_tr, "logistic", None)
                    score_all = np.asarray(score_fn(tab[eval_mask]))
                elif probe == "hgb_tab":
                    score_fn, _ = _fit_probe("hgb", tab[tr], y_tr)
                    thr, thr_info = _inner_split_threshold(
                        tab[tr], y_tr, groups_tr, "hgb", None)
                    score_all = np.asarray(score_fn(tab[eval_mask]))
                else:
                    raise ValueError(probe)

                score_all = np.asarray(score_all, dtype=np.float64)
                eval_pos = np.flatnonzero(eval_mask)
                # classification metrics on the usable (labelled, non-degenerate)
                # subset of the eval fold; false-stop uses ALL eval rows
                usable_local = usable[eval_pos]
                scores = score_all[usable_local]
                metrics = _classification_metrics(scores, y_ev, thr)
                metrics["threshold_selection"] = thr_info
                if thr is not None:
                    metrics["eval_scores_at_thr"] = {
                        "tp": int(((scores >= thr) & (y_ev == 1)).sum()),
                        "fp": int(((scores >= thr) & (y_ev == 0)).sum()),
                    }
                fs = _false_stop_rates(
                    {"sample_ids": eval_sids}, score_all, thr, labels,
                    start_samples, ts_samples, ts_deg)
                metrics["false_stop"] = fs
                record["probes"][probe] = metrics
            units.append(record)
    return units


def aggregate_units(units: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Median + range over the 9 units, per probe."""
    probe_names = sorted({p for u in units for p in u["probes"]})
    out: dict[str, Any] = {}
    for probe in probe_names:
        rows = [u["probes"][probe] for u in units
                if probe in u["probes"] and not u["probes"][probe].get("skipped")]
        if not rows:
            out[probe] = {"skipped": True}
            continue
        out[probe] = {
            "n_units": len(rows),
            "auroc": _minmax(r["auroc"] for r in rows),
            "auprc": _minmax(r["auprc"] for r in rows),
            "eval_prevalence": _minmax(r["prevalence"] for r in rows),
            "precision_at_threshold": _minmax(
                r["precision_at_threshold"] for r in rows),
            "recall_at_threshold": _minmax(
                r["recall_at_threshold"] for r in rows),
            "threshold": _minmax(r["threshold"] for r in rows),
            "start_fp_rate": _minmax(
                r["false_stop"]["start"]["rate"] for r in rows),
            "ts_fp_rate": _minmax(
                r["false_stop"]["ts"]["rate"] for r in rows),
            "units_missing_threshold": sum(
                1 for r in rows if r["threshold"] is None),
        }
    return out


def run_ordering_probe(
    features: dict[str, Any],
    q: np.ndarray, q_mask: np.ndarray, meta: dict[str, np.ndarray],
    degenerate: np.ndarray,
) -> list[dict[str, Any]]:
    """Ridge z→q per (fold, seed); pairwise ordering on the eval fold."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z_by_unit = features["z"]
    folds = meta["folds"]
    curve_ids = meta["curve_ids"]
    intent_ids = meta["intent_ids"]
    nodes = meta["node_index"]
    ts_index_of = meta["ts_index"]

    units: list[dict[str, Any]] = []
    for fold in DEV_FOLDS:
        usable = q_mask & (~degenerate)
        tr = (folds != fold) & usable
        ev = (fold == folds) & usable
        for seed in SEEDS:
            z = z_by_unit[(fold, seed)]
            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=RIDGE_ALPHA))
            model.fit(z[tr], q[tr])
            q_pred = model.predict(z[ev])
            groups: dict[tuple[str, str], list[int]] = defaultdict(list)
            ev_pos = np.flatnonzero(ev)
            for j, pos in enumerate(ev_pos):
                groups[(curve_ids[pos], intent_ids[pos])].append(j)
            per_curve: dict[str, list[float]] = defaultdict(list)
            n_pairs = n_correct = n_cs = n_cs_correct = 0
            for (cid, _intent), idxs in sorted(groups.items()):
                idxs_a = np.asarray(idxs, dtype=np.int64)
                res = pairwise_ordering_accuracy(
                    q[ev][idxs_a], q_pred[idxs_a], nodes[ev][idxs_a],
                    int(ts_index_of[ev][idxs_a][0]))
                n_pairs += res["n_pairs"]
                n_correct += res["n_correct"]
                n_cs += res["n_pairs_cross_ts"]
                n_cs_correct += res["n_correct_cross_ts"]
                if res["accuracy"] is not None:
                    per_curve[cid].append(res["accuracy"])
            curve_accs = [a for v in per_curve.values() for a in v]
            units.append({
                "fold": fold, "seed": seed,
                "n_train": int(tr.sum()), "n_eval": int(ev.sum()),
                "accuracy": _r6(n_correct / n_pairs) if n_pairs else None,
                "n_pairs": n_pairs,
                "cross_ts_accuracy": _r6(n_cs_correct / n_cs) if n_cs else None,
                "n_pairs_cross_ts": n_cs,
                "spearman_pooled": _r6(spearman_corr(q[ev], q_pred)),
                "per_curve_accuracy": {
                    "median": _r6(np.median(curve_accs)) if curve_accs else None,
                    "min": _r6(min(curve_accs)) if curve_accs else None,
                    "max": _r6(max(curve_accs)) if curve_accs else None,
                    "n_curves": len(per_curve),
                },
            })
    return units


def aggregate_ordering(units: Sequence[dict[str, Any]]) -> dict[str, Any]:
    accs = [u["accuracy"] for u in units if u["accuracy"] is not None]
    cross = [u["cross_ts_accuracy"] for u in units
             if u["cross_ts_accuracy"] is not None]
    sp = [u["spearman_pooled"] for u in units if u["spearman_pooled"] is not None]
    return {
        "n_units": len(units),
        "accuracy_mean": _r6(np.mean(accs)) if accs else None,
        "accuracy_min": _r6(min(accs)) if accs else None,
        "accuracy_max": _r6(max(accs)) if accs else None,
        "cross_ts_accuracy_mean": _r6(np.mean(cross)) if cross else None,
        "cross_ts_accuracy_min": _r6(min(cross)) if cross else None,
        "cross_ts_accuracy_max": _r6(max(cross)) if cross else None,
        "spearman_pooled_median": _r6(np.median(sp)) if sp else None,
        "n_pairs_total": int(sum(u["n_pairs"] for u in units)),
        "n_pairs_cross_ts_total": int(
            sum(u["n_pairs_cross_ts"] for u in units)),
    }


def window_sensitivity(
    best_probe: str,
    dataset: KeyAtomDatasetV3,
    dev_positions: Sequence[int],
    base_features: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    v52_labels: Any,
    c_by_fold: dict[int, float],
) -> dict[str, Any]:
    """Rebuild endpoint labels per w_pos; retrain best z-probe per unit."""
    rows: list[dict[str, Any]] = []
    for w_pos in WINDOW_GRID:
        table = build_endpoint_stop_labels(
            dataset, v52_labels, positive_window=w_pos, negative_window=W_NEG)
        units = run_endpoint_probes(
            base_features, table, meta,
            probe_names=(best_probe,))
        # per-unit precision/recall at pre-registered threshold + FP rates
        recalls = [u["probes"][best_probe]["recall_at_threshold"]
                   for u in units if best_probe in u["probes"]]
        precisions = [u["probes"][best_probe]["precision_at_threshold"]
                      for u in units if best_probe in u["probes"]]
        thr_missing = sum(
            1 for u in units if best_probe in u["probes"]
            and u["probes"][best_probe]["threshold"] is None)
        start_rates = [u["probes"][best_probe]["false_stop"]["start"]["rate"]
                       for u in units if best_probe in u["probes"]]
        ts_rates = [u["probes"][best_probe]["false_stop"]["ts"]["rate"]
                    for u in units if best_probe in u["probes"]]
        start_pool = _pooled_false_stop(units, best_probe, "start")
        ts_pool = _pooled_false_stop(units, best_probe, "ts")
        precision_mean = _median(precisions)
        # pre-registered rule: the threshold CALIBRATION must achieve
        # precision ≥ 0.8 on inner-val (else the window fails the rule), and
        # on eval the start/TS false-stop rates must stay within limits;
        # achieved eval precision/recall at that threshold are reported.
        rule_hit = (
            thr_missing == 0
            and start_pool is not None and start_pool <= START_FP_LIMIT
            and ts_pool is not None and ts_pool <= TS_FP_LIMIT_WINDOW)
        rows.append({
            "w_pos": w_pos, "w_neg": W_NEG,
            "n_positive": table.n_positive,
            "n_negative": table.n_negative,
            "n_mask": table.n_mask,
            "prevalence": _r6(table.n_positive / max(1, table.n_rows)),
            "precision_median": _r6(precision_mean),
            "recall_median": _r6(_median(recalls)),
            "threshold_missing_units": thr_missing,
            "start_fp_pooled": _r6(start_pool),
            "ts_fp_pooled": _r6(ts_pool),
            "start_fp_median": _r6(_median(start_rates)),
            "ts_fp_median": _r6(_median(ts_rates)),
            "rule_hit": bool(rule_hit),
        })
    hits = [r for r in rows if r["rule_hit"]]
    if hits:
        # pre-registered: among rule-hitting windows pick the largest recall;
        # ties → smaller w_pos (stricter window, conservative).
        best = sorted(hits, key=lambda r: (-(r["recall_median"] or 0.0),
                                           r["w_pos"]))[0]
        recommended: float | None = float(best["w_pos"])
        note = ("recommended window = rule-hit window with highest median "
                "endpoint recall @ precision ≥ 0.8 (ties → smaller w_pos)")
    else:
        recommended = None
        note = ("NO window satisfies the pre-registered rule "
                "(recall @ precision ≥ 0.8 with start FP ≤ 2% and TS FP ≤ 2%); "
                "no window recommended — do not pick outside the rule")
    return {"probe": best_probe, "w_neg": W_NEG, "rows": rows,
            "recommended_w_pos": recommended, "rule_note": note}


def _pooled_false_stop(
    units: Sequence[dict[str, Any]], probe: str, kind: str
) -> float | None:
    flagged = total = 0
    seen = False
    for u in units:
        if probe not in u["probes"]:
            continue
        fs = u["probes"][probe]["false_stop"][kind]
        flagged += int(fs["flagged"])
        total += int(fs["n_rows"])
        seen = True
    if not seen or total == 0:
        return None
    return flagged / total


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def render_markdown(pack: dict[str, Any]) -> str:
    probes = pack["endpoint_probes"]
    agg = probes["aggregate"]
    ordering = pack["ordering"]
    win = pack["window_sensitivity"]
    boot = pack["bootstrap"]
    gate = pack["gate"]
    prov = pack["provenance"]

    def mark(ok: bool) -> str:
        return "PASS" if ok else "**FAIL**"

    lines: list[str] = []
    a = lines.append
    a("# V5.2-R D1 报告 — 冻结编码器 Endpoint Head + Directed Ordering 探针")
    a("")
    a(f"> 生成: `scripts/agent_ops/data_gate/v52r_d1_probe.py` · schema "
      f"`{pack['schema']}` · gate `{gate['gate_id']}`")
    a(f"> commit `{prov['code_commit'][:12]}` · 主窗口 "
      f"w_pos={PRIMARY_W_POS:.2f} / w_neg={W_NEG:.2f} Å（敏感性见 §5）")
    a(f"> 编码器: R3 winner checkpoint_best.pt（fold-matched，eval 折编码器"
      f"未见过 eval 折曲线）· dev folds 0–2，sealed folds 3/4 仅计数")
    a("")
    a("## 0. 数据与编码器")
    a("")
    a(f"- dev 行数 {pack['dataset']['n_dev_rows']} / 总行数 "
      f"{pack['dataset']['n_rows']}；sealed 行 "
      f"{pack['dataset']['n_sealed_rows']}（曲线 "
      f"{pack['dataset']['n_sealed_curves']}，未编码/未探针）")
    a(f"- 退化 flag 行（排除 train/eval，单独报告）: "
      f"{pack['dataset']['n_degenerate_rows']} 行 / 曲线 "
      f"{list(DEGENERATE_TS_CURVES)}")
    a(f"- 主窗口标签: 正例 {pack['dataset']['primary_window']['n_positive']} / "
      f"负例 {pack['dataset']['primary_window']['n_negative']} / Mask "
      f"{pack['dataset']['primary_window']['n_mask']}（prevalence "
      f"{pack['dataset']['primary_window']['prevalence']}）")
    a(f"- 9 个编码器: fold-matched R3 winner `checkpoint_best.pt`（arm "
      f"F2-bidir），checkpoint/embedding sha256 见 evidence_pack.json "
      f"`encoders` 字段")
    a(f"- tabular 特征: {pack['tabular_features']['composition']}")
    a("")
    a("## 1. 每折 Endpoint 探针（z-logistic vs HGB vs constant）")
    a("")
    a("| fold | seed | probe | AUROC | AUPRC | prevalence | prec@thr | "
      "rec@thr | start FP | TS FP |")
    a("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for u in probes["units"]:
        for probe in ("constant", "logistic_tab", "hgb_tab", "logistic_z",
                      "mlp_z", "logistic_h"):
            m = u["probes"].get(probe)
            if m is None:
                continue
            if m.get("skipped"):
                a(f"| {u['fold']} | {u['seed']} | {probe} | — | — | — | — | "
                  "— | — | — |")
                continue
            fs = m["false_stop"]
            a(f"| {u['fold']} | {u['seed']} | {probe} | {m['auroc']} | "
              f"{m['auprc']} | {m['prevalence']} | "
              f"{m['precision_at_threshold']} | {m['recall_at_threshold']} | "
              f"{fs['start']['rate']} | {fs['ts']['rate']} |")
    a("")
    a("### 聚合（9 单元 median [min, max]）")
    a("")
    a("| probe | AUROC | AUPRC | prec@thr | rec@thr | start FP | TS FP | "
      "缺阈值单元 |")
    a("|---|---|---|---|---|---|---|---:|")
    for probe, s in agg.items():
        if s.get("skipped"):
            a(f"| {probe} | — | — | — | — | — | — | skipped |")
            continue

        def fmt(d: dict) -> str:
            if d["median"] is None:
                return "—"
            return f"{d['median']} [{d['min']}, {d['max']}]"
        a(f"| {probe} | {fmt(s['auroc'])} | {fmt(s['auprc'])} | "
          f"{fmt(s['precision_at_threshold'])} | "
          f"{fmt(s['recall_at_threshold'])} | {fmt(s['start_fp_rate'])} | "
          f"{fmt(s['ts_fp_rate'])} | {s['units_missing_threshold']} |")
    a("")
    a(f"- 阈值规则（预注册）: 训练集内层曲线分组 split 上 precision ≥ "
      f"{PRECISION_TARGET:.2f} 的最小阈值；达不到则 thr=∅（预测零正例）。")
    a(f"- 退化 flag 行已排除 train/eval（单独计数，见 §0）。")
    a("")
    a("## 2. Directed Ordering 探针（Ridge z→q）")
    a("")
    a("| fold | seed | pairs | accuracy | cross-TS pairs | cross-TS acc | "
      "Spearman(池化) | 曲线内 acc med/min/max |")
    a("|---|---|---:|---:|---:|---:|---:|---|")
    for u in ordering["units"]:
        pc = u["per_curve_accuracy"]
        a(f"| {u['fold']} | {u['seed']} | {u['n_pairs']} | {u['accuracy']} | "
          f"{u['n_pairs_cross_ts']} | {u['cross_ts_accuracy']} | "
          f"{u['spearman_pooled']} | {pc['median']} / {pc['min']} / "
          f"{pc['max']} |")
    a("")
    oa = ordering["aggregate"]
    a(f"- 聚合: mean accuracy **{oa['accuracy_mean']}** "
      f"(min {oa['accuracy_min']} / max {oa['accuracy_max']})；cross-TS mean "
      f"**{oa['cross_ts_accuracy_mean']}** (min {oa['cross_ts_accuracy_min']})；"
      f"Spearman 池化中位 {oa['spearman_pooled_median']}")
    a(f"- 总 pairs {oa['n_pairs_total']}（跨 TS {oa['n_pairs_cross_ts_total']}）；"
      "q 符号: reverse = −q（v5.2 冻结）")
    a("")
    a("## 3. 窗口敏感性（最佳 z-probe，w_neg=0.30 冻结）")
    a("")
    a(f"- 探针: `{win['probe']}`；预注册规则 = endpoint recall @ precision ≥ "
      f"{PRECISION_TARGET:.2f} 且 start FP ≤ {START_FP_LIMIT:.0%} 且 TS FP ≤ "
      f"{TS_FP_LIMIT_WINDOW:.0%}")
    a("")
    a("| w_pos (Å) | 正例 | prevalence | prec med | rec med | start FP(池化) | "
      "TS FP(池化) | rule hit |")
    a("|---|---:|---:|---:|---:|---:|---:|:--:|")
    for r in win["rows"]:
        a(f"| {r['w_pos']:.2f} | {r['n_positive']} | {r['prevalence']} | "
          f"{r['precision_median']} | {r['recall_median']} | "
          f"{r['start_fp_pooled']} | {r['ts_fp_pooled']} | "
          f"{'✓' if r['rule_hit'] else '✗'} |")
    a("")
    if win["recommended_w_pos"] is not None:
        a(f"- **建议冻结窗口: w_pos = {win['recommended_w_pos']:.2f} Å / "
          f"w_neg = {W_NEG:.2f} Å**（{win['rule_note']}）")
    else:
        a(f"- **无窗口满足预注册规则**（{win['rule_note']}）")
    a("")
    a("## 4. 统计（cluster bootstrap by curve）")
    a("")
    a(f"- 最佳 z-probe `{boot['probe']}` 池化 AUPRC = **{boot['point']}**，"
      f"95% CI **[{boot['ci95']['lower']}, {boot['ci95']['upper']}]**，"
      f"prevalence **{boot['prevalence']}**")
    a(f"- {boot['n_resamples']} 次重采样（有效 {boot['n_valid_resamples']}，"
      f"rng seed {boot['seed']}，{boot['n_curves']} 条 dev 曲线）")
    a("")
    a("## 5. Gate 判定（V5.2R-D1-ENDPOINT-HEAD）")
    a("")
    a("| 检查 | 期望 | 实测 | 判定 |")
    a("|---|---|---|:--:|")
    for c in gate["checks"]:
        a(f"| {c['id']} | {c['expect']} | {c['actual']} | {mark(c['pass'])} |")
    a("")
    a(f"**Overall: {gate['verdict']}**")
    a("")
    a(f"> 运行时: extraction {pack['runtime_s'].get('extraction')} s · "
      f"probing+stats {pack['runtime_s'].get('probing_and_stats')} s · "
      f"总计 {pack['runtime_s'].get('total')} s（预算 < 1800 s）")
    a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def collect_provenance(tables_dir: Path) -> dict[str, Any]:
    data_manifest = tables_dir / DATA_MANIFEST_FILENAME
    teacher_manifest = tables_dir / "teacher_manifest.json"
    split_manifest = tables_dir / SPLIT_MANIFEST_FILENAME
    dataset_sha = ""
    if data_manifest.is_file():
        doc = json.loads(data_manifest.read_text(encoding="utf-8"))
        dataset_sha = str(doc.get("dataset_sha256", ""))
    prov = {
        "code_commit": git_head(),
        "data_manifest_sha256": sha256_file(data_manifest)
        if data_manifest.is_file() else "",
        "teacher_manifest_sha256": sha256_file(teacher_manifest)
        if teacher_manifest.is_file() else "",
        "split_sha256": sha256_file(split_manifest)
        if split_manifest.is_file() else "",
        "label_schema": "pes2ts_fullpath_nav_labels_v1",
        "config_sha256": sha256_file(CAMPAIGN_CONFIG)
        if CAMPAIGN_CONFIG.is_file() else "",
        "dataset_sha256": dataset_sha,
        "tables_dir": str(tables_dir),
    }
    required = ("code_commit", "data_manifest_sha256",
                "teacher_manifest_sha256", "split_sha256", "label_schema",
                "config_sha256")
    prov["required_fields_present"] = all(prov.get(k, "") for k in required)
    prov["missing_required_fields"] = [k for k in required if not prov.get(k)]
    return prov


def load_meta(dataset: KeyAtomDatasetV3,
              dev_positions: Sequence[int]) -> dict[str, np.ndarray]:
    samples = dataset.samples
    return {
        "sample_ids": np.array(
            [samples[int(i)].sample_id for i in dev_positions], dtype=object),
        "curve_ids": np.array(
            [samples[int(i)].curve_id for i in dev_positions], dtype=object),
        "intent_ids": np.array(
            [samples[int(i)].intent.intent_id for i in dev_positions],
            dtype=object),
        "folds": np.array(
            [int(samples[int(i)].fold) for i in dev_positions],
            dtype=np.int64),
        "node_index": np.array(
            [int(samples[int(i)].node_index) for i in dev_positions],
            dtype=np.int64),
        "direction": np.array(
            [samples[int(i)].intent.direction for i in dev_positions],
            dtype=object),
        "ts_index": np.array(
            [dataset.curve_ts_index(samples[int(i)].curve_id)
             for i in dev_positions], dtype=np.int64),
    }


def run(
    tables_dir: Path,
    out_dir: Path,
    embeddings_dir: Path | None = None,
    *,
    extract_only: bool = False,
    probe_only: bool = False,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(embeddings_dir) if embeddings_dir else out_dir
    emb_dir.mkdir(parents=True, exist_ok=True)

    dataset = KeyAtomDatasetV3(
        tables_dir,
        label_statuses=("ok", "terminal", "masked", "conflict"))
    samples = dataset.samples
    dev_positions = [i for i, s in enumerate(samples) if s.fold in DEV_FOLDS]
    sealed_positions = [i for i, s in enumerate(samples)
                        if s.fold in SEALED_FOLDS]

    v52_labels = dataset.v52_labels()
    primary = build_endpoint_stop_labels(dataset, v52_labels)
    meta = load_meta(dataset, dev_positions)

    t_extract: float | None = None
    extract_records: list[dict[str, Any]] = []
    if probe_only:
        for fold in DEV_FOLDS:
            for seed in SEEDS:
                p = emb_dir / f"embeddings_{fold}_{seed}.npz"
                if not p.is_file():
                    raise FileNotFoundError(
                        f"--probe-only requires {p}; run extraction first")
    else:
        t_ex0 = time.time()
        extract_records = extract_embeddings(
            dataset, dev_positions, emb_dir, R3_RUNS_DIR)
        t_extract = _r6(time.time() - t_ex0)
    if extract_only:
        return {"schema": SCHEMA_ID, "phase": "extraction-only",
                "embeddings_dir": str(emb_dir)}

    t0 = time.time()
    # one encoder per (eval fold, seed): its embeddings cover ALL dev rows
    # (train rows = other dev folds, eval rows = fold F) — keyed by unit
    z_by_unit: dict[tuple[int, int], np.ndarray] = {}
    h_by_unit: dict[tuple[int, int], np.ndarray] = {}
    ckpt_by_unit = {(r["fold"], r["seed"]): r for r in extract_records}
    encoder_files: list[dict[str, Any]] = []
    for fold in DEV_FOLDS:
        for seed in SEEDS:
            path = emb_dir / f"embeddings_{fold}_{seed}.npz"
            data = np.load(path, allow_pickle=True)
            if list(data["sample_ids"]) != list(meta["sample_ids"]):
                raise RuntimeError(
                    f"{path}: sample_ids misaligned with dataset order")
            z_by_unit[(fold, seed)] = data["z"].astype(np.float64)
            h_by_unit[(fold, seed)] = data["h"].astype(np.float64)
            entry = {
                "fold": fold, "seed": seed,
                "run_id": FOLD_SEED_RUNS[fold][seed],
                "path": str(path), "sha256": sha256_file(path),
                "n_rows": int(data["z"].shape[0]),
            }
            ck = ckpt_by_unit.get((fold, seed))
            if ck is not None:
                entry["checkpoint"] = ck["checkpoint"]
                entry["checkpoint_sha256"] = ck["checkpoint_sha256"]
                entry["checkpoint_config_arm"] = ck["checkpoint_config_arm"]
            encoder_files.append(entry)

    tab, tab_info = build_tabular(dataset, dev_positions)
    degenerate = np.array([
        bool(primary.by_sample[sid].degenerate_flag)
        for sid in meta["sample_ids"]], dtype=np.bool_)
    base_features = {"z": z_by_unit, "h": h_by_unit, "tabular": tab}
    t_probe = time.time()

    units = run_endpoint_probes(base_features, primary, meta)
    agg = aggregate_units(units)

    # best z-probe = highest aggregated median AUPRC among z-input probes
    z_probes = [p for p in ("logistic_z", "mlp_z") if p in agg
                and not agg[p].get("skipped")]
    if not z_probes:
        raise RuntimeError("no z-input probe produced results")
    best_probe = max(
        z_probes, key=lambda p: (agg[p]["auprc"]["median"] or -np.inf,
                                 -z_probes.index(p)))

    c_by_fold: dict[int, float] = {}
    for u in units:
        if "logistic_z_best_C" in u:
            c_by_fold[u["fold"]] = float(u["logistic_z_best_C"])

    ordering_units = run_ordering_probe(
        base_features,
        np.array([v52_labels.by_sample[sid].q if v52_labels.by_sample[sid].q
                  is not None else np.nan for sid in meta["sample_ids"]],
                 dtype=np.float64),
        np.array([bool(v52_labels.by_sample[sid].q_mask)
                  for sid in meta["sample_ids"]], dtype=np.bool_),
        meta, degenerate)
    ordering_agg = aggregate_ordering(ordering_units)

    win = window_sensitivity(
        best_probe, dataset, dev_positions, base_features, meta,
        v52_labels, c_by_fold)

    # bootstrap: best probe, pooled scores (row = mean over 3 seeds)
    scores_pooled, labels_pooled, curves_pooled = _pool_eval_scores(
        best_probe, base_features, meta, primary)
    boot = curve_cluster_bootstrap_auprc(
        scores_pooled, labels_pooled, curves_pooled, n_resamples=n_bootstrap)
    boot["probe"] = best_probe
    boot["pooling"] = "per-row mean of 3 fold-matched-seed probe scores"

    # false-stop aggregation for the gate (best probe, primary window)
    false_stop_agg = {
        "probe": best_probe,
        "start_pooled_rate": _r6(_pooled_false_stop(units, best_probe, "start")),
        "ts_pooled_rate": _r6(_pooled_false_stop(units, best_probe, "ts")),
        "start_median_rate": agg.get(best_probe, {}).get(
            "start_fp_rate", {}).get("median"),
        "ts_median_rate": agg.get(best_probe, {}).get(
            "ts_fp_rate", {}).get("median"),
        "ts_degenerate_curves": list(DEGENERATE_TS_CURVES),
        "note": "pooled over 9 units at the pre-registered threshold",
    }

    gate = evaluate_gate(boot, false_stop_agg, ordering_agg)

    degenerate_report = {
        "curves": list(DEGENERATE_TS_CURVES),
        "n_flagged_rows_dev": int(degenerate.sum()),
        "flagged_row_sample_ids": [
            sid for sid, d in zip(meta["sample_ids"], degenerate) if d],
        "note": "excluded from probe train AND eval; reported separately "
                "(contract §5.3 rule 4)",
    }

    pack: dict[str, Any] = {        "schema": SCHEMA_ID,
        "label_schema": "pes2ts_fullpath_nav_labels_v1",
        "gate": gate,
        "dataset": {
            "n_rows": len(samples),
            "n_dev_rows": len(dev_positions),
            "n_sealed_rows": len(sealed_positions),
            "n_sealed_curves": len({samples[i].curve_id
                                    for i in sealed_positions}),
            "sealed_folds": list(SEALED_FOLDS),
            "sealed_use": "counted only; never encoded/probed",
            "n_dev_curves": len(set(meta["curve_ids"].tolist())),
            "n_degenerate_rows": int(degenerate.sum()),
            "primary_window": {
                "w_pos": PRIMARY_W_POS, "w_neg": W_NEG,
                "n_positive": primary.n_positive,
                "n_negative": primary.n_negative,
                "n_mask": primary.n_mask,
                "prevalence": _r6(primary.n_positive / max(1, primary.n_rows)),
            },
        },
        "degenerate_report": degenerate_report,
        "encoders": encoder_files,
        "tabular_features": tab_info,
        "endpoint_probes": {
            "primary_window": [PRIMARY_W_POS, W_NEG],
            "precision_target": PRECISION_TARGET,
            "threshold_rule": "smallest threshold with precision ≥ "
                              f"{PRECISION_TARGET} on inner curve-group "
                              "split of TRAIN; else predict-nothing",
            "best_z_probe": best_probe,
            "units": units,
            "aggregate": agg,
        },
        "ordering": {"units": ordering_units, "aggregate": ordering_agg,
                     "ridge_alpha": RIDGE_ALPHA},
        "window_sensitivity": win,
        "bootstrap": boot,
        "false_stop_gate": false_stop_agg,
        "provenance": collect_provenance(tables_dir),
        "runtime_s": {
            "extraction": t_extract,
            "label_and_feature_build": _r6(t_probe - t0),
            "probing_and_stats": _r6(time.time() - t_probe),
            "total": _r6(time.time() - t_start),
        },
    }
    (out_dir / "evidence_pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "D1_PROBE.md").write_text(
        render_markdown(pack), encoding="utf-8")
    return pack


def _pool_eval_scores(
    probe: str,
    features: dict[str, np.ndarray],
    meta: dict[str, np.ndarray],
    labels: EndpointStopLabelTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refit ``probe`` per (fold, seed); pooled per-row eval scores.

    Each usable dev row is scored by the 3 fold-matched-seed probes of its
    fold; the pooled score is the mean over those seeds.  Labels/curves are
    aligned by construction (rows grouped per fold in dev order).
    """
    z_by_unit = features["z"]
    h_by_unit = features["h"]
    tab = features["tabular"]
    folds = meta["folds"]
    sample_ids = meta["sample_ids"]
    lab_vec = np.array([
        labels.by_sample[sid].endpoint_stop_label for sid in sample_ids],
        dtype=np.int64)
    deg_vec = np.array([
        bool(labels.by_sample[sid].degenerate_flag) for sid in sample_ids],
        dtype=np.bool_)
    usable = (lab_vec >= 0) & (~deg_vec)
    scores_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    curves_parts: list[np.ndarray] = []
    for fold in DEV_FOLDS:
        tr = (folds != fold) & usable
        ev = (fold == folds) & usable
        if not ev.any():
            continue
        acc = np.zeros(int(ev.sum()), dtype=np.float64)
        for seed in SEEDS:
            x = {"logistic_z": z_by_unit[(fold, seed)],
                 "mlp_z": z_by_unit[(fold, seed)],
                 "logistic_h": h_by_unit[(fold, seed)], "hgb_tab": tab,
                 "logistic_tab": tab}[probe]
            if probe in ("logistic_z", "logistic_h", "logistic_tab"):
                score_fn, _ = _fit_probe("logistic", x[tr], lab_vec[tr])
            elif probe == "hgb_tab":
                score_fn, _ = _fit_probe("hgb", x[tr], lab_vec[tr])
            else:
                score_fn, _ = _fit_probe("mlp", x[tr], lab_vec[tr])
            acc += np.asarray(score_fn(x[ev]), dtype=np.float64)
        scores_parts.append(acc / len(SEEDS))
        labels_parts.append(lab_vec[ev])
        curves_parts.append(meta["curve_ids"][ev])
    if not scores_parts:
        return (np.zeros(0), np.zeros(0, dtype=np.int64), np.array([], dtype=object))
    return (np.concatenate(scores_parts), np.concatenate(labels_parts),
            np.concatenate(curves_parts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--embeddings-dir", type=Path, default=None,
                    help="embedding cache dir (default: --out)")
    ap.add_argument("--extract-only", action="store_true",
                    help="run embedding extraction only (torch)")
    ap.add_argument("--probe-only", action="store_true",
                    help="skip extraction; reuse cached embeddings (sklearn)")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES,
                    help="cluster-bootstrap resamples (default 1000)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the gate verdict is FAIL")
    args = ap.parse_args(argv)
    if args.extract_only and args.probe_only:
        ap.error("--extract-only and --probe-only are mutually exclusive")
    pack = run(args.tables_dir, args.out, args.embeddings_dir,
               extract_only=args.extract_only, probe_only=args.probe_only,
               n_bootstrap=args.bootstrap)
    if args.extract_only:
        print(f"extraction complete → {args.embeddings_dir or args.out}")
        return 0
    gate = pack["gate"]
    print(f"[{gate['gate_id']}] verdict: {gate['verdict']}")
    for c in gate["checks"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['id']}: {c['actual']}")
    print(f"best z-probe: {pack['endpoint_probes']['best_z_probe']}  "
          f"bootstrap AUPRC {pack['bootstrap']['point']} "
          f"CI [{pack['bootstrap']['ci95']['lower']}, "
          f"{pack['bootstrap']['ci95']['upper']}] vs prevalence "
          f"{pack['bootstrap']['prevalence']}")
    print(f"recommended window: "
          f"{pack['window_sensitivity']['recommended_w_pos']}")
    print(f"runtime total: {pack['runtime_s']['total']} s "
          f"(budget 1800 s)")
    print(f"evidence: {args.out / 'evidence_pack.json'}")
    print(f"report:   {args.out / 'D1_PROBE.md'}")
    if args.strict and gate["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
