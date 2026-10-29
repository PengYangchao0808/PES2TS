"""V5.3-MP scale-ladder paired CI: curve-level bootstrap for 20k vs 50k.

WHY THIS EXISTS
---------------
``metrics.json`` only keeps the AGGREGATE Move/q/endpoint numbers; the
per-curve/per-cluster dicts are dropped inside ``evaluate_checkpoint``.  A
3-seed paired test on seeds cannot reach significance (min two-sided
p = 0.25 for n=3), so the honest unit for an interval is the frozen dev
CURVE (589 curves, shared by the 20k and 50k cohorts).

This CLI re-runs the frozen outer eval on existing checkpoints (NO
retraining), keeps the per-curve decomposition, and then does a PAIRED
curve-level bootstrap of the 50k - 20k difference.

FAITHFULNESS GATE (fail-closed)
-------------------------------
Every re-eval must reproduce the run's own ``metrics.json`` aggregates:
  * Move   : ``move.guards.model_median_rmse_over_zero``
  * sign   : ``move.sre_and_sign.sign_accuracy_cluster_median``
  * q MAE  : ``q_head.mae_curve_mean``
  * AUPRC  : ``endpoint_head.auprc`` / ``auroc``
A mismatch > TOLERANCE refuses to write the artifact (the per-curve
decomposition would be untrustworthy).  Because the re-eval runs on the
loader-v2 columnar index while the 20k runs were TRAINED through loader v1,
an exact match is also an eval-path v1<->v2 equivalence certificate on the
full 589-curve dev set (it does NOT certify the training path).

Subcommands
-----------
``predict`` : one tier -> ``per_curve_<tier>_<arm>_s<seed>.json`` (+ endpoint rows .npz)
``stats``   : read both tiers -> paired bootstrap -> JSON + Markdown

Exit codes: 0 ok · 1 refusal (gate/corpus mismatch) · 2 usage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

# WP1a (V5.4.1 §4.1): ONE shared pair-ownership tool for every eval entry
# point (this CLI and key_atom_bc_v3.evaluate_checkpoint) — numpy-only, so
# it is safe to import at module level.  Never re-implement the pair-window
# filter locally (the key-ptr/n_nodes mirror bug this replaces).
from pes2ts_core.pesn.offline.evaluation.pair_ownership_v3 import (  # noqa: E402
    assign_batch_pairs,
    merge_pair_audits,
)

#: tier -> (default runs root, default tables dir, geometry LRU bound)
TIER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "5k": {
        "runs_root": "outputs/pesn/v53mp",
        "tables_dir": "/pes2ts_data/v53mp_data",
        # re-eval only touches the frozen dev curves, so a small geometry
        # LRU is enough (the training tiers' 6500..65000 bounds are for the
        # sampler's whole-pool access pattern, not this one)
        "cache_curves": 2000,
    },
    "20k": {
        "runs_root": "outputs/pesn/v53mp",
        "tables_dir": "/pes2ts_data/v53mp_data",
        "cache_curves": 2000,
    },
    "50k": {
        "runs_root": "outputs/pesn/v53mp/s4_50k",
        "tables_dir": "/pes2ts_data/v53mp_50k_data",
        "cache_curves": 2000,
    },
}

ARMS = ("A", "B", "C")
SEEDS = (17, 29, 43)
#: float64 aggregates reproduce to ~1e-12 when the eval path is identical;
#: anything above this means the re-eval is not the same computation.
TOLERANCE = 1e-9
#: endpoint ranking metrics are tie-sensitive: a 1e-8 float32 drift in the
#: logits can reorder near-tied rows and move AUPRC by ~1e-5 (measured).
TOLERANCES = {
    # 1e-6 is still a hard gate (float32 forward noise is ~3e-9 measured);
    # sign/dead-zone are discrete comparisons, so they must match exactly.
    "median_rmse_over_zero": 1e-6,
    "sign_accuracy_cluster_median": 1e-9,
    "q_mae_curve_mean": 1e-6,
    "endpoint_auprc": 1e-3,
    "endpoint_auroc": 1e-6,
}


class PairedCIError(RuntimeError):
    """Typed fail-closed refusal."""


# --------------------------------------------------------------------------
# tier corpus
# --------------------------------------------------------------------------


def _build_tier(
    tier: str, tables_dir: Path, split_path: Path, device: str,
    cache_curves: int, dev_pack: str,
) -> Tuple[Any, Any, Any, List[int], Any, Dict[str, Any]]:
    """Loader-v2 dataset + lazy labels + frozen dev positions + feature cache."""
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import LABEL_STATUSES
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
        DevFeaturePackView,
        dev_positions_from_split,
    )
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        FEATURE_LEVEL_V53MP,
        _CacheView,
        _FeatureLRU,
    )
    from scripts.campaigns.r3_v53mp_run import _subset_curve_ids

    idx_dir = tables_dir / "idx_v2"
    if not idx_dir.is_dir():
        raise PairedCIError("IDX2_MISSING: {}".format(idx_dir))
    dataset = ColumnarKeyAtomDatasetV3(
        tables_dir,
        split_manifest=str(split_path),
        label_statuses=LABEL_STATUSES,
        cache_curves=int(cache_curves),
    )
    labels, endpoint = dataset.label_tables()
    samples = dataset.samples
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    dev_curve_ids = set(_subset_curve_ids(split_doc, "dev"))
    dev_pos = dev_positions_from_split(samples, dev_curve_ids)
    if not dev_pos:
        raise PairedCIError("DEV_EMPTY: no dev positions for tier {}".format(tier))

    cache: Any
    pack_info: Dict[str, Any] = {"used": False}
    if dev_pack == "auto":
        fsha = None
        try:
            from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
                features_sha256_of,
                positions_sha256,
            )
            fsha = features_sha256_of()
            psha = positions_sha256(dev_pos)
            candidate = tables_dir / "dev_packs" / fsha / psha
            if candidate.is_dir():
                cache = DevFeaturePackView.open(
                    candidate,
                    expected_positions=dev_pos,
                    feature_level=FEATURE_LEVEL_V53MP,
                )
                pack_info = {"used": True, "path": str(candidate)}
        except Exception as exc:  # pack is an optimization only
            pack_info = {"used": False, "note": "pack unavailable: {}".format(exc)}
    if not pack_info.get("used"):
        cache = _CacheView(_FeatureLRU(
            dataset, capacity=len(dev_pos) + 64,
            level=FEATURE_LEVEL_V53MP, intent_off=False))
    return dataset, labels, endpoint, dev_pos, cache, pack_info


def _trainer_config(state: Mapping[str, Any], device: str) -> Any:
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        V53MPTrainerConfig,
    )

    stored = state.get("config") or {}
    names = {f.name for f in dc_fields(V53MPTrainerConfig)}
    kwargs = {k: v for k, v in stored.items() if k in names}
    kwargs["device"] = device
    kwargs["cuda_memory_fraction"] = None
    kwargs["eval_vram_gate_mib"] = None
    kwargs["eval_vram_gate_timeout_s"] = 0
    return V53MPTrainerConfig(**kwargs)


# --------------------------------------------------------------------------
# one checkpoint -> per-curve decomposition
# --------------------------------------------------------------------------


def _collect(
    model: Any, cache: Any, dev_pos: Sequence[int], labels: Any,
    endpoint: Any, device: str, batch_size: int = 64,
) -> Dict[str, Any]:
    """Mirror of ``evaluate_checkpoint`` + ``evaluate_v53mp`` collection.

    Same batching (``_batch_from_cache`` @ 64), same label readers, same
    batch views — only the OUTPUT differs: per-curve arrays are kept.
    """
    import torch

    from pes2ts_core.pesn.contracts.key_atom_v3 import (
        NODE_KEY_SET_KEY,
        NODE_POS_KEY,
        forward_view_v3,
    )
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import _batch_from_cache
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _endpoint_for_batch,
        _labels_for_batch,
    )

    u_samples: List[np.ndarray] = []
    y_samples: List[np.ndarray] = []
    curve_ids: List[str] = []
    g53_samples: List[Dict[str, Any]] = []
    pair_audits: List[Dict[str, Any]] = []
    q_preds: List[float] = []
    q_trues: List[float] = []
    q_masks: List[bool] = []
    q_curves: List[str] = []
    ep_logits: List[float] = []
    ep_y: List[int] = []
    ep_keep: List[bool] = []
    ep_curves: List[str] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(dev_pos), batch_size):
            chunk = list(dev_pos[start:start + batch_size])
            batch = _batch_from_cache(cache, chunk, device)
            out = model(forward_view_v3(batch))
            sample_ids = [str(s) for s in batch["sample_ids"]]
            n_keys = batch["n_keys"].long().cpu().numpy()
            n_nodes = batch["n_nodes"].long().cpu().numpy()
            key_index = batch[NODE_KEY_SET_KEY].long().cpu().numpy()
            y_all = batch["y_teacher"].cpu().numpy()
            pos_all = batch[NODE_POS_KEY].cpu().numpy()
            pair_index = batch["pair_index"].long().cpu().numpy()
            pair_edit = batch["pair_edit"].cpu().numpy()
            u_all = out["u"].cpu().numpy()
            # WP1a (V5.4.1 §4.1): same shared tool as evaluate_checkpoint —
            # pair_index is batch-global NODE rows (node-row ptr owns pair
            # windows); key-row ptr only slices u/y/key_index.
            ownership = assign_batch_pairs(
                n_nodes=n_nodes, n_keys=n_keys, key_index=key_index,
                pair_index=pair_index, pair_edit=pair_edit)
            pair_audits.append(ownership.audit(per_graph=False))
            for g, gs in enumerate(ownership.graphs):
                ks, ke = gs.key_lo, gs.key_hi
                u_g = u_all[ks:ke]
                y_g = y_all[ks:ke]
                rows_g = key_index[ks:ke]
                cid = str(batch["curve_ids"][g])
                u_samples.append(u_g.astype(np.float64))
                y_samples.append(y_g.astype(np.float64))
                curve_ids.append(cid)
                g53_samples.append({
                    "pos": pos_all,
                    "key_rows": [int(r) for r in rows_g],
                    "pairs": [(int(a), int(b)) for a, b in gs.pairs],
                    "pair_edits": [float(e) for e in gs.pair_edits],
                    "u": u_g.astype(np.float64),
                    "y": y_g.astype(np.float64),
                    "group_id": cid,
                })
            lab = _labels_for_batch(sample_ids, labels, device)
            ep = _endpoint_for_batch(sample_ids, endpoint, device)
            q_preds.extend(out["q"].cpu().tolist())
            q_trues.extend(lab["q_true"].cpu().tolist())
            q_masks.extend(lab["q_mask"].cpu().tolist())
            for sid in sample_ids:
                q_curves.append(str(labels.by_sample[sid].curve_id))
            ep_logits.extend(out["endpoint_logit"].cpu().tolist())
            ep_mask = ep["endpoint_mask"].cpu().numpy().astype(bool)
            ys = (ep["y_endpoint"] * ep["endpoint_mask"].float()).cpu().numpy()
            ep_y.extend(ys.astype(int).tolist())
            ep_keep.extend(ep_mask.tolist())
            ep_curves.extend(str(labels.by_sample[sid].curve_id)
                             for sid in sample_ids)

    return {
        "u_samples": u_samples,
        "y_samples": y_samples,
        "curve_ids": curve_ids,
        "g53_samples": g53_samples,
        "pair_ownership_audit": merge_pair_audits(pair_audits),
        "q_preds": np.asarray(q_preds, dtype=np.float64),
        "q_trues": np.asarray(q_trues, dtype=np.float64),
        "q_masks": np.asarray(q_masks, dtype=bool),
        "q_curves": np.asarray(q_curves, dtype=object),
        "ep_logits": np.asarray(ep_logits, dtype=np.float64),
        "ep_y": np.asarray(ep_y, dtype=np.int64),
        "ep_keep": np.asarray(ep_keep, dtype=bool),
        "ep_curves": np.asarray(ep_curves, dtype=object),
    }


def _ok_mask(dataset: Any, positions: Sequence[int]) -> np.ndarray:
    """``label_status == 'ok'`` per position (the Move-metric denominator).

    ``evaluate_v53mp`` filters dev positions to teacher-defined ``ok`` rows
    before calling ``evaluate_checkpoint``; terminal/masked/conflict rows are
    zero-padded and must never enter the MOVE metrics.
    """
    flags: List[bool] = []
    chunk = 4096
    for start in range(0, len(positions), chunk):
        recs = dataset.records([int(p) for p in positions[start:start + chunk]])
        flags.extend(str(getattr(r, "label_status", "ok")) == "ok" for r in recs)
    return np.asarray(flags, dtype=bool)


def _per_curve(col_move: Mapping[str, Any],
               col_all: Mapping[str, Any]) -> Dict[str, Any]:
    """Official g52/g53 aggregates + the per-curve decomposition.

    ``col_move`` is the eval over the ``label_status == 'ok'`` positions (the
    Move denominator, exactly what ``evaluate_v53mp`` passes to
    ``evaluate_checkpoint``); ``col_all`` covers every dev position (the q and
    endpoint denominators, 589 curves / 113,040 unmasked rows).
    """
    from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
        evaluate_g52,
        evaluate_g53,
    )
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _sigmoid,
        auc_mann_whitney,
        average_precision,
    )

    g52 = evaluate_g52(
        col_move["u_samples"], col_move["y_samples"],
        col_move["curve_ids"], bootstrap_draws=0)
    g53 = evaluate_g53(col_move["g53_samples"])
    collected = col_all
    per_curve_rmse0 = {
        cid: entry["rmse_over_zero"]
        for cid, entry in g52["raw"]["per_cluster"].items()
    }
    per_curve_sign = dict(g53["per_cluster_sign_accuracy"])

    # q per curve (identical grouping to evaluate_v53mp: curve only)
    qp, qt, qm = collected["q_preds"], collected["q_trues"], collected["q_masks"]
    per_curve_q: Dict[str, List[int]] = {}
    for i, cid in enumerate(collected["q_curves"].tolist()):
        per_curve_q.setdefault(cid, []).append(i)
    q_mae: Dict[str, float] = {}
    q_corr: Dict[str, float] = {}
    for cid, idxs in per_curve_q.items():
        ok = [i for i in idxs if qm[i]]
        if not ok:
            continue
        q_mae[cid] = float(np.abs(qp[ok] - qt[ok]).mean())
        if len(ok) > 1 and np.std(qp[ok]) > 0 and np.std(qt[ok]) > 0:
            q_corr[cid] = float(np.corrcoef(qp[ok], qt[ok])[0, 1])

    # endpoint rows (kept for the curve-level AUPRC/AUROC bootstrap)
    keep = collected["ep_keep"] & np.isin(collected["ep_y"], (0, 1))
    scores = _sigmoid(collected["ep_logits"][keep])
    labels_keep = collected["ep_y"][keep]
    ep_curves = collected["ep_curves"][keep]
    auprc = average_precision(scores.tolist(), labels_keep.tolist())
    auroc = auc_mann_whitney(scores.tolist(), labels_keep.tolist())

    return {
        "per_curve_rmse_over_zero": per_curve_rmse0,
        "per_curve_sign": per_curve_sign,
        "per_curve_q_mae": q_mae,
        "per_curve_q_corr": q_corr,
        "endpoint_rows": {
            "scores": scores.tolist(),
            "y": labels_keep.tolist(),
            "curves": ep_curves.tolist(),
        },
        "recomputed": {
            "median_rmse_over_zero": g52["raw"]["aggregate"].get(
                "median_rmse_over_zero"),
            "sign_accuracy_cluster_median": g53["aggregate"].get(
                "sign_accuracy_cluster_median"),
            "dead_zone_fraction": g53["aggregate"].get("dead_zone_fraction"),
            "n_active_pairs": g53["aggregate"].get("n_active_pairs"),
            "n_dead_pairs": g53["aggregate"].get("n_dead_pairs"),
            "q_mae_curve_mean": (float(np.mean(list(q_mae.values())))
                                 if q_mae else None),
            "q_corr_curve_mean": (float(np.mean(list(q_corr.values())))
                                  if q_corr else None),
            "endpoint_auprc": auprc,
            "endpoint_auroc": auroc,
            "prevalence": (float(labels_keep.mean())
                           if labels_keep.size else None),
        },
    }


def _assert_matches(name: str, got: Any, want: Any, tol: float = TOLERANCE) -> None:
    if got is None or want is None:
        raise PairedCIError(
            "{}: missing value (got={!r} want={!r})".format(name, got, want))
    if not (math.isfinite(float(got)) and math.isfinite(float(want))):
        raise PairedCIError("{}: non-finite ({} vs {})".format(name, got, want))
    if abs(float(got) - float(want)) > tol:
        raise PairedCIError(
            "{}: re-eval {} != metrics.json {} (|d|={:.3e} > {:.0e})".format(
                name, got, want, abs(float(got) - float(want)), tol))


def cmd_predict(args: argparse.Namespace) -> int:
    import torch

    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        load_model_v53mp,
    )
    from scripts.campaigns.r3_v53mp_run import R3_WINNER_CKPT

    tier = args.tier
    runs_root = Path(args.runs_root or TIER_DEFAULTS[tier]["runs_root"])
    tables_dir = Path(args.tables_dir or TIER_DEFAULTS[tier]["tables_dir"])
    split_path = (Path(args.split) if args.split
                  else tables_dir / "split_manifest_v53mp.json")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    dataset, labels, endpoint, dev_pos, cache, pack_info = _build_tier(
        tier, tables_dir, split_path, args.device,
        int(args.cache_curves or TIER_DEFAULTS[tier]["cache_curves"]),
        args.dev_pack)
    print(json.dumps({
        "tier": tier, "dev_positions": len(dev_pos),
        "dev_pack": pack_info, "build_s": round(time.time() - t0, 1),
    }, sort_keys=True), flush=True)
    t_mask = time.time()
    ok_mask = _ok_mask(dataset, dev_pos)
    move_pos = [int(p) for p, keep in zip(dev_pos, ok_mask) if keep]
    print(json.dumps({
        "tier": tier, "ok_positions": len(move_pos),
        "mask_s": round(time.time() - t_mask, 1),
    }, sort_keys=True), flush=True)

    wanted = [a for a in ARMS if a in args.arms]
    for arm in wanted:
        for seed in SEEDS:
            run_id = "v53mp_{}_{}_s{}".format(tier, arm, seed)
            run_dir = runs_root / run_id
            ckpt = run_dir / "checkpoint_best.pt"
            if not ckpt.is_file():
                raise PairedCIError("CHECKPOINT_MISSING: {}".format(ckpt))
            metrics = json.loads((run_dir / "metrics.json").read_text())
            t_run = time.time()
            state = torch.load(str(ckpt), map_location=args.device,
                               weights_only=False)
            cfg = _trainer_config(state, args.device)
            model = load_model_v53mp(cfg, r3_ckpt=R3_WINNER_CKPT,
                                     device=args.device)
            model.load_state_dict(state["model"], strict=True)
            collected = _collect(model, cache, dev_pos, labels, endpoint,
                                 args.device)
            move_collected = _collect(model, cache, move_pos, labels, endpoint,
                                      args.device)
            del model
            torch.cuda.empty_cache()
            per_curve = _per_curve(move_collected, collected)

            want = {
                "median_rmse_over_zero":
                    metrics["move"]["guards"]["model_median_rmse_over_zero"],
                "sign_accuracy_cluster_median":
                    metrics["move"]["sre_and_sign"]["sign_accuracy_cluster_median"],
                "q_mae_curve_mean": metrics["q_head"]["mae_curve_mean"],
                "endpoint_auprc": metrics["endpoint_head"]["auprc"],
                "endpoint_auroc": metrics["endpoint_head"]["auroc"],
            }
            gates = {}
            for key, want_v in want.items():
                _assert_matches("{}:{}".format(run_id, key),
                                per_curve["recomputed"][key], want_v,
                                TOLERANCES[key])
                gates[key] = {"recomputed": per_curve["recomputed"][key],
                              "metrics_json": want_v,
                              "abs_diff": abs(float(per_curve["recomputed"][key])
                                              - float(want_v)),
                              "tolerance": TOLERANCES[key]}

            payload = {
                "schema": "v53mp_scale_paired_per_curve_v1",
                "tier": tier, "arm": arm, "seed": seed, "run_id": run_id,
                "checkpoint_sha256": metrics.get("checkpoint_best_sha256"),
                "n_dev_positions": len(dev_pos),
                "n_dev_curves": len(per_curve["per_curve_rmse_over_zero"]),
                "dev_pack": pack_info,
                "metrics_path": str(run_dir / "metrics.json"),
                "gates_vs_metrics_json": gates,
                "recomputed": per_curve["recomputed"],
                "per_curve_rmse_over_zero":
                    per_curve["per_curve_rmse_over_zero"],
                "per_curve_sign": per_curve["per_curve_sign"],
                "per_curve_q_mae": per_curve["per_curve_q_mae"],
                "per_curve_q_corr": per_curve["per_curve_q_corr"],
            }
            target = out_dir / "per_curve_{}_{}_s{}.json".format(tier, arm, seed)
            target.write_text(json.dumps(payload, indent=1, sort_keys=True),
                              encoding="utf-8")
            ep = per_curve["endpoint_rows"]
            np.savez_compressed(
                out_dir / "endpoint_rows_{}_{}_s{}.npz".format(tier, arm, seed),
                scores=np.asarray(ep["scores"], dtype=np.float32),
                y=np.asarray(ep["y"], dtype=np.int8),
                curves=np.asarray(ep["curves"], dtype="U32"))
            print(json.dumps({
                "run_id": run_id, "ok": True,
                "rmse0": per_curve["recomputed"]["median_rmse_over_zero"],
                "sign": per_curve["recomputed"]["sign_accuracy_cluster_median"],
                "auprc": per_curve["recomputed"]["endpoint_auprc"],
                "wall_s": round(time.time() - t_run, 1),
            }, sort_keys=True), flush=True)
    print(json.dumps({"tier": tier, "done": True,
                      "wall_s": round(time.time() - t0, 1)}, sort_keys=True))
    return 0


# --------------------------------------------------------------------------
# paired curve-level bootstrap
# --------------------------------------------------------------------------


def _median(x: np.ndarray) -> float:
    return float(np.median(x))


def _load_tier(out_dir: Path, tier: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    runs: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = out_dir / "per_curve_{}_{}_s{}.json".format(tier, arm, seed)
            if not path.is_file():
                continue
            runs[(arm, seed)] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def _paired_ci_diff_of_medians(
    a: np.ndarray, b: np.ndarray, *, draws: int, seed: int,
) -> Dict[str, Any]:
    """Paired curve bootstrap of ``median(b) - median(a)``."""
    rng = np.random.default_rng(seed)
    n = a.size
    samples = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        samples[i] = np.median(b[idx]) - np.median(a[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"delta": float(np.median(b) - np.median(a)),
            "ci95": [float(lo), float(hi)],
            "frac_draws_opposite_sign": float(min((samples <= 0).mean(),
                                                 (samples >= 0).mean())),
            "draws": int(draws), "seed": int(seed), "unit": "dev_curve"}


def _paired_ci_mean_diff(
    a: np.ndarray, b: np.ndarray, *, draws: int, seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = a.size
    samples = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(b[idx]) - np.mean(a[idx]))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"delta": float(np.mean(b) - np.mean(a)),
            "ci95": [float(lo), float(hi)],
            "frac_draws_opposite_sign": float(min((samples <= 0).mean(),
                                                 (samples >= 0).mean())),
            "draws": int(draws), "seed": int(seed), "unit": "dev_curve"}


def _load_endpoint(out_dir: Path, tier: str, arm: str, seed: int) -> Dict[str, Any]:
    path = out_dir / "endpoint_rows_{}_{}_s{}.npz".format(tier, arm, seed)
    if not path.is_file():
        raise PairedCIError("ENDPOINT_ROWS_MISSING: {}".format(path))
    doc = np.load(path, allow_pickle=False)
    return {"scores": doc["scores"], "y": doc["y"].astype(np.int8),
            "curves": doc["curves"].astype(str)}


def _paired_ci_auprc(
    a: Mapping[str, Any], b: Mapping[str, Any], *, draws: int, seed: int,
) -> Dict[str, Any]:
    """Paired curve bootstrap of the POOLED AUPRC difference."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    curves = sorted(set(a["curves"]) & set(b["curves"]))
    if not curves:
        raise PairedCIError("ENDPOINT_CURVES_EMPTY: no shared curves")
    idx: Dict[str, Dict[str, np.ndarray]] = {}
    for tag, block in (("a", a), ("b", b)):
        buckets: Dict[str, List[int]] = {c: [] for c in curves}
        for i, c in enumerate(block["curves"]):
            if c in buckets:
                buckets[c].append(i)
        idx[tag] = {c: np.asarray(v, dtype=np.int64) for c, v in buckets.items()}

    def _score(block: Mapping[str, Any], which: str, pick: np.ndarray) -> Tuple[float, float]:
        rows = np.concatenate([idx[which][c] for c in pick]) if len(pick) else \
            np.zeros(0, dtype=np.int64)
        y = block["y"][rows]
        s = block["scores"][rows]
        if y.size == 0 or y.min() == y.max():
            return float("nan"), float("nan")
        return (float(average_precision_score(y, s)),
                float(roc_auc_score(y, s)))

    a_ap, a_auc = _score(a, "a", np.asarray(curves, dtype=object))
    b_ap, b_auc = _score(b, "b", np.asarray(curves, dtype=object))
    rng = np.random.default_rng(seed)
    n = len(curves)
    arr = np.asarray(curves, dtype=object)
    dap = np.full(draws, np.nan)
    dauc = np.full(draws, np.nan)
    for i in range(draws):
        pick = arr[rng.integers(0, n, size=n)]
        ap_a, auc_a = _score(a, "a", pick)
        ap_b, auc_b = _score(b, "b", pick)
        dap[i] = ap_b - ap_a
        dauc[i] = auc_b - auc_a
    def _ci(x: np.ndarray) -> List[float]:
        x = x[np.isfinite(x)]
        return [float(v) for v in np.percentile(x, [2.5, 97.5])] if x.size else [None, None]
    return {
        "delta_auprc": b_ap - a_ap, "ci95_auprc": _ci(dap),
        "delta_auroc": b_auc - a_auc, "ci95_auroc": _ci(dauc),
        "draws": int(draws), "seed": int(seed), "unit": "dev_curve",
        "auprc_a": a_ap, "auprc_b": b_ap,
        "auroc_a": a_auc, "auroc_b": b_auc,
        "metric": "pooled rows of resampled curves (sklearn AP/AUROC)",
    }


def cmd_stats(args: argparse.Namespace) -> int:
    out_dir = Path(args.dir)
    tier_a, tier_b = args.a, args.b
    runs_a = _load_tier(out_dir, tier_a)
    runs_b = _load_tier(out_dir, tier_b)
    if not runs_a or not runs_b:
        raise PairedCIError(
            "ARTIFACTS_MISSING: {} / {} per-curve files under {}".format(
                tier_a, tier_b, out_dir))

    report: Dict[str, Any] = {
        "schema": "v53mp_scale_paired_ci_v1",
        "tier_a": tier_a, "tier_b": tier_b,
        "draws": int(args.draws),
        "note": ("curve-level paired bootstrap over the frozen dev curves; "
                 "the 3 seeds are NOT a significance unit (min p=0.25)"),
        "arms": {},
    }

    for arm in ARMS:
        seeds = [s for s in SEEDS
                 if (arm, s) in runs_a and (arm, s) in runs_b]
        if not seeds:
            continue
        arm_block: Dict[str, Any] = {"seeds": seeds, "per_seed": {}}
        for seed in seeds:
            ra, rb = runs_a[(arm, seed)], runs_b[(arm, seed)]

            def vec(key: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
                shared = sorted(set(ra[key]) & set(rb[key]))
                a = np.asarray([ra[key][c] for c in shared], dtype=np.float64)
                b = np.asarray([rb[key][c] for c in shared], dtype=np.float64)
                return shared, a, b

            shared_r, a_r, b_r = vec("per_curve_rmse_over_zero")
            shared_s, a_s, b_s = vec("per_curve_sign")
            shared_q, a_q, b_q = vec("per_curve_q_mae")
            if len(shared_r) < 100:
                raise PairedCIError(
                    "CURVE_SET_TOO_SMALL: {} s{} shared={}".format(
                        arm, seed, len(shared_r)))
            seed_block = {
                "n_curves": len(shared_r),
                "n_curves_sign": len(shared_s),
                "n_curves_q": len(shared_q),
                "move_rmse_over_zero": _paired_ci_diff_of_medians(
                    a_r, b_r, draws=args.draws, seed=seed),
                "move_sign": _paired_ci_diff_of_medians(
                    a_s, b_s, draws=args.draws, seed=seed),
                "q_mae": _paired_ci_mean_diff(
                    a_q, b_q, draws=args.draws, seed=seed),
                "same_curve_set_as_other_tier":
                    len(shared_r) == len(ra["per_curve_rmse_over_zero"]) ==
                    len(rb["per_curve_rmse_over_zero"]),
            }
            if arm == "C" and args.endpoint_draws > 0:
                seed_block["endpoint"] = _paired_ci_auprc(
                    _load_endpoint(out_dir, tier_a, arm, seed),
                    _load_endpoint(out_dir, tier_b, arm, seed),
                    draws=args.endpoint_draws, seed=seed)
            arm_block["per_seed"][str(seed)] = seed_block

        # seed-level summary of the point estimates (NO p-value: n=3)
        for metric in ("move_rmse_over_zero", "move_sign", "q_mae"):
            deltas = [arm_block["per_seed"][str(s)][metric]["delta"]
                      for s in seeds]
            arm_block.setdefault("seed_summary", {})[metric] = {
                "deltas": deltas,
                "median": float(np.median(deltas)),
                "all_same_sign": all(d > 0 for d in deltas) or
                                 all(d < 0 for d in deltas),
            }
        if arm == "C":
            ep = [arm_block["per_seed"][str(s)]["endpoint"]["delta_auprc"]
                  for s in seeds
                  if "endpoint" in arm_block["per_seed"][str(s)]]
            arm_block["seed_summary"]["endpoint_auprc"] = {
                "deltas": ep,
                "median": float(np.median(ep)) if ep else None,
                "all_same_sign": (all(d > 0 for d in ep) or
                                  all(d < 0 for d in ep)) if ep else None,
            }
        report["arms"][arm] = arm_block

    # corpus identity: the two tiers must decompose the SAME dev curves
    a_curves = set(next(iter(runs_a.values()))["per_curve_rmse_over_zero"])
    b_curves = set(next(iter(runs_b.values()))["per_curve_rmse_over_zero"])
    report["corpus"] = {
        "curves_a": len(a_curves), "curves_b": len(b_curves),
        "shared": len(a_curves & b_curves),
        "identical": a_curves == b_curves,
    }
    if not report["corpus"]["identical"]:
        raise PairedCIError(
            "CORPUS_MISMATCH: {} dev curves vs {} (shared {})".format(
                len(a_curves), len(b_curves), len(a_curves & b_curves)))

    target = out_dir / "scale_paired_ci_{}_vs_{}.json".format(tier_a, tier_b)
    target.write_text(json.dumps(report, indent=1, sort_keys=True),
                      encoding="utf-8")
    md = _render_markdown(report)
    (out_dir / "scale_paired_ci_{}_vs_{}.md".format(tier_a, tier_b)).write_text(
        md, encoding="utf-8")
    print(md)
    print("wrote: {}".format(target))
    return 0


def _fmt_ci(delta: float, ci: Sequence[float]) -> str:
    lo, hi = ci
    if lo is None or hi is None:
        return "{:+.4f} [--, --]".format(delta)
    flag = "" if lo <= 0.0 <= hi else " *"
    return "{:+.4f} [{:+.4f}, {:+.4f}]{}".format(delta, lo, hi, flag)


def _render_markdown(report: Mapping[str, Any]) -> str:
    a, b = report["tier_a"], report["tier_b"]
    lines = [
        "# V5.3-MP scale-ladder paired CI: {} vs {}".format(a, b),
        "",
        "> 单位 = 冻结 dev curve（{} 条，两档完全一致：{}）。".format(
            report["corpus"]["shared"], report["corpus"]["identical"]),
        "> 3 seed 不是显著性单位（n=3 最小双侧 p=0.25）；区间来自 curve-level "
        "paired bootstrap（{} draws）。".format(report["draws"]),
        "> `*` = 95% CI 不跨 0。",
        "",
    ]
    for arm, block in report["arms"].items():
        lines.append("## Arm {}".format(arm))
        lines.append("")
        lines.append("| seed | n curves | Move RMSE/0 Δ | sign Δ | q MAE Δ |")
        lines.append("|---|---:|---|---|---|")
        for seed, sb in block["per_seed"].items():
            lines.append("| {} | {} | {} | {} | {} |".format(
                seed, sb["n_curves"],
                _fmt_ci(sb["move_rmse_over_zero"]["delta"],
                        sb["move_rmse_over_zero"]["ci95"]),
                _fmt_ci(sb["move_sign"]["delta"], sb["move_sign"]["ci95"]),
                _fmt_ci(sb["q_mae"]["delta"], sb["q_mae"]["ci95"])))
        lines.append("")
        if arm == "C":
            lines.append("| seed | AUPRC {} | Δ AUPRC | Δ AUROC |".format(a))
            lines.append("|---|---|---|---|")
            for seed, sb in block["per_seed"].items():
                ep = sb.get("endpoint")
                if not ep:
                    continue
                lines.append("| {} | {:.4f} → {:.4f} | {} | {} |".format(
                    seed, ep["auprc_a"], ep["auprc_b"],
                    _fmt_ci(ep["delta_auprc"], ep["ci95_auprc"]),
                    _fmt_ci(ep["delta_auroc"], ep["ci95_auroc"])))
            lines.append("")
        summary = block.get("seed_summary", {})
        if summary:
            lines.append("Seed-level point estimates ({}): ".format(
                ", ".join(str(s) for s in block["seeds"])) + "; ".join(
                "{} {:.4f} (same sign: {})".format(
                    k, v["median"], v["all_same_sign"])
                for k, v in summary.items() if v.get("median") is not None))
            lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_scale_paired_ci",
        description="V5.3-MP scale-ladder per-curve re-eval + paired bootstrap",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pred = sub.add_parser("predict", help="re-eval one tier into per-curve files")
    pred.add_argument("--tier", required=True, choices=sorted(TIER_DEFAULTS))
    pred.add_argument("--runs-root", default=None)
    pred.add_argument("--tables-dir", default=None)
    pred.add_argument("--split", default=None)
    pred.add_argument("--out-dir", required=True)
    pred.add_argument("--arms", default="ABC")
    pred.add_argument("--dev-pack", default="auto", choices=("auto", "off"))
    pred.add_argument("--cache-curves", type=int, default=None)
    pred.add_argument("--device", default="cuda")
    pred.set_defaults(func=cmd_predict)

    stats = sub.add_parser("stats", help="paired curve bootstrap over both tiers")
    stats.add_argument("--dir", required=True)
    stats.add_argument("--a", default="20k")
    stats.add_argument("--b", default="50k")
    stats.add_argument("--draws", type=int, default=10000)
    stats.add_argument("--endpoint-draws", type=int, default=1000)
    stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PairedCIError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except FileNotFoundError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
