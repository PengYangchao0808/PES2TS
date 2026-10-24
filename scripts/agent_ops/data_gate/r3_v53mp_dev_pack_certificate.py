#!/usr/bin/env python3
"""Dev-feature-pack bitwise certificate (S4 plan §5.3 / lane I2 gate).

Two-level gate on one real checkpoint:

1. **FEATURES bitwise** (primary; the pack's actual output) — for a
   seeded sample of frozen dev positions, the pack-served feature dicts
   are compared array-by-array (``np.array_equal`` + dtype) + value-exact
   against a LIVE ``build_sample_features_v3`` build through the same
   ``to_sample_v3`` path.  This is deterministic on every device.
2. **METRICS bitwise** — the FINAL dev eval is run twice via the
   trainer's OWN functions (``evaluate_v53mp`` + ``_FeatureLRU``/
   ``_CacheView``/``DevFeaturePackView``, imported, never copied): once
   with features built live into a warmed dev LRU (the pre-pack path),
   once served from the pack.  Both the model pass AND the matched R3-
   baseline second pass go through the same cache object in each run,
   exactly like ``_train_run_v53mp_impl``.  Every metric field must be
   bit-identical.

DEVIATION note (measured, reported): the EGNN forward on CUDA is NOT
bit-deterministic (atomic-add ordering in message passing — even the SAME
path run twice on GPU differs by ~1e-8).  The metrics check therefore
runs on CPU by default (measured deterministic: 0 run-vs-run diffs);
``--device cuda`` is available but its metric diff must be interpreted
against that noise floor (the FEATURES check above is unaffected).

Inputs (defaults target the S3 C_s17 run):
  --run-dir       outputs/pesn/v53mp/v53mp_20k_C_s17 (ckpt + metrics.json)
  --r3-ckpt       the matched R3 baseline (driver: r3_eval_ckpt = r3_ckpt)
  --tables-dir    the v2 tables (idx_v2 + dev_packs required)
  --pack          pack dir (default: resolved from the live features_sha +
                  the run's dev positions via the driver rule)
  --max-dev N     REDUCED certificate: evaluate a seed-17 subsample of the
                  frozen dev positions instead of the full set (both sides
                  equal — the gate is pack-vs-build equality, not coverage)
  --device        cpu (default, deterministic) or cuda (see note)
  --feature-probes N   positions for the FEATURES bitwise check
                       (default 2048; 0 disables)

Writes ``outputs/pesn/v53mp/reports/dev_pack_certificate_<sha12>.json``
(+ ``.md``); exit code 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    LABEL_STATUSES,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    ColumnarKeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (  # noqa: E402
    DevFeaturePackView,
    dev_positions_from_split,
    features_sha256_of,
    positions_sha256,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    FEATURE_LEVEL_V53MP,
    V53MPTrainerConfig,
    _CacheView,
    _FeatureLRU,
    _prefetch_warm_lru,
    evaluate_v53mp,
)
from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (  # noqa: E402
    FeaturePrefetcher,
)

DEFAULT_TABLES_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/data"
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/v53mp_20k_A_s17"
REPORT_DIR = PROJECT_ROOT / "outputs/pesn/v53mp/reports"
SPLIT_DEFAULT = "split_manifest_v53mp.json"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _rss_mib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024


def _diff_metrics(a: Any, b: Any, path: str = "") -> list[dict]:
    """Exact recursive diff of two metric trees (floats bitwise)."""
    out: list[dict] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append({"field": "{}.{}".format(path, key),
                            "v1": a.get(key), "v2": b.get(key)})
                continue
            out.extend(_diff_metrics(a[key], b[key],
                                     "{}.{}".format(path, key)))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append({"field": path, "v1_len": len(a), "v2_len": len(b)})
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out.extend(_diff_metrics(x, y, "{}[{}]".format(path, i)))
    else:
        same = (a == b)
        if not same and isinstance(a, float) and isinstance(b, float):
            same = (np.float64(a).tobytes() == np.float64(b).tobytes())
        if not same:
            out.append({"field": path, "built": a, "pack": b})
    return out


def run_certificate(tables_dir: Path, run_dir: Path, r3_ckpt: Path,
                    pack_dir: Path | None, report_dir: Path, *,
                    max_dev: int | None, device: str,
                    workers: int,
                    feature_probes: int = 2048) -> dict[str, Any]:
    import torch

    from pes2ts_core.pesn.features.key_atom_graph_v3 import (
        build_sample_features_v3,
    )
    from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        load_model_v53mp,
    )

    tables_dir = Path(tables_dir)
    run_dir = Path(run_dir)
    timings: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    rss_start = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024

    # --- checkpoint + config + frozen eval inputs ---------------------------
    ckpt_path = run_dir / "checkpoint_best.pt"
    _require_file(ckpt_path)
    metrics_path = run_dir / "metrics.json"
    _require_file(metrics_path)
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload = torch.load(str(ckpt_path), map_location="cpu",
                         weights_only=False)
    config = V53MPTrainerConfig(**dict(payload["config"]))
    config.device = str(device)
    tau_end = (payload.get("tau_end_calibration") or {}).get("tau_end")
    train_q_mean = (run_metrics.get("q_head") or {}).get("constant_q_value")

    _log("[1/6] reconstructing model + R3 baseline …")
    t0 = time.perf_counter()
    model = load_model_v53mp(config, r3_ckpt=str(r3_ckpt), device=device)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    r3_model = None
    if r3_ckpt is not None and Path(r3_ckpt).is_file():
        r3_payload = torch.load(str(r3_ckpt), map_location="cpu",
                                weights_only=False)
        r3_model = StarEGNNV3()
        r3_model.load_state_dict(r3_payload["model"], strict=True)
        r3_model.to(device)
    timings["model_load_s"] = round(time.perf_counter() - t0, 1)

    _log("[2/6] opening v2 dataset + frozen dev positions …")
    t0 = time.perf_counter()
    ds = ColumnarKeyAtomDatasetV3(
        tables_dir, label_statuses=LABEL_STATUSES,
        verify_manifest=False, use_geometry_pack=False)
    labels_v52, endpoint_labels = ds.label_tables()
    split = tables_dir / SPLIT_DEFAULT
    split_doc = json.loads(split.read_text(encoding="utf-8"))
    dev_curve_ids = split_doc.get("subsets", {}).get("dev") or []
    dev_pos = dev_positions_from_split(ds.samples, dev_curve_ids)
    _require(bool(dev_pos), "DEV_EMPTY", "no frozen dev positions")
    full_n = len(dev_pos)
    if max_dev is not None and len(dev_pos) > int(max_dev):
        rng = np.random.RandomState(17)
        keep = sorted(int(x) for x in rng.choice(
            len(dev_pos), size=int(max_dev), replace=False))
        dev_pos = [dev_pos[i] for i in keep]
    timings["dataset_open_s"] = round(time.perf_counter() - t0, 1)
    _log(f"    {len(dev_pos)} dev positions "
         f"(full frozen set: {full_n})")

    # --- resolve the pack -----------------------------------------------------
    fsha = features_sha256_of()
    psha_full = positions_sha256(dev_positions_from_split(
        ds.samples, dev_curve_ids))
    if pack_dir is None:
        pack_dir = tables_dir / "dev_packs" / fsha / psha_full
    _log(f"[3/6] opening pack {pack_dir} …")
    t0 = time.perf_counter()
    view = DevFeaturePackView.open(
        Path(pack_dir), expected_positions=dev_pos,
        feature_level=FEATURE_LEVEL_V53MP, verify_files=True)
    timings["pack_open_s"] = round(time.perf_counter() - t0, 1)
    _log(f"    binding ok (features {fsha[:12]}, positions {psha_full[:12]})")

    # --- FEATURES bitwise (primary gate: the pack's actual output) ----------
    feat_diffs: list[dict] = []
    n_feat_checked = 0
    if feature_probes and feature_probes > 0:
        _log("[3a/6] FEATURES bitwise: pack vs live build …")
        t0 = time.perf_counter()
        rng = np.random.RandomState(17)
        n_probe = min(int(feature_probes), len(dev_pos))
        probe_idx = sorted(int(x) for x in rng.choice(
            len(dev_pos), size=n_probe, replace=False))
        for i in probe_idx:
            pos = dev_pos[i]
            got = view[pos]
            rec = ds.records([pos])[0]
            want = build_sample_features_v3(
                ds.to_sample_v3(rec), level=FEATURE_LEVEL_V53MP,
                intent_off=False)
            # the pack stores FINALIZED dicts (_FeatureLRU._finalize:
            # y_teacher zero-pad) — apply the identical step here
            if want.get("y_teacher") is None:
                want = dict(want)
                want["y_teacher"] = np.zeros(
                    (len(rec.intent.key_rows()), 3), dtype=np.float32)
            n_feat_checked += 1
            if set(got) != set(want):
                feat_diffs.append({"pos": pos, "kind": "keys",
                                   "got": sorted(got), "want": sorted(want)})
                continue
            for key in sorted(want):
                va, vb = want[key], got[key]
                if isinstance(va, np.ndarray) != isinstance(vb, np.ndarray):
                    feat_diffs.append({"pos": pos, "key": key,
                                       "kind": "type",
                                       "want": type(va).__name__,
                                       "got": type(vb).__name__})
                elif isinstance(va, np.ndarray):
                    if (va.dtype != vb.dtype or va.shape != vb.shape
                            or va.tobytes() != vb.tobytes()):
                        feat_diffs.append({"pos": pos, "key": key,
                                           "kind": "array"})
                elif isinstance(va, tuple):
                    if va != vb:
                        feat_diffs.append({"pos": pos, "key": key,
                                           "kind": "tuple"})
                elif isinstance(va, (np.float32, np.float64)):
                    if va.tobytes() != vb.tobytes():
                        feat_diffs.append({"pos": pos, "key": key,
                                           "kind": "np_scalar"})
                elif not (va == vb):
                    feat_diffs.append({"pos": pos, "key": key,
                                       "kind": "value",
                                       "want": repr(va), "got": repr(vb)})
        checks["features_bitwise_equal"] = {
            "passed": not feat_diffs,
            "n_checked": n_feat_checked,
            "n_diffs": len(feat_diffs),
            "diffs": feat_diffs[:20],
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        _log(f"    {n_feat_checked} features checked, "
             f"{len(feat_diffs)} diffs")

    # --- run A: built features (pre-pack path, trainer's own machinery) ------
    _log("[4/6] eval pass A: live-built features (warmed dev LRU) …")
    t0 = time.perf_counter()
    dev_lru = _FeatureLRU(ds, capacity=len(dev_pos) + 64,
                          level=FEATURE_LEVEL_V53MP, intent_off=False)
    prefetcher = None
    if workers > 0:
        try:
            prefetcher = FeaturePrefetcher(
                min(workers, 12), FEATURE_LEVEL_V53MP, False)
        except Exception as exc:  # noqa: BLE001 - fall back to serial warm
            _log("    prefetcher unavailable ({}: {}); serial warm".format(
                type(exc).__name__, exc))
    try:
        _prefetch_warm_lru(dev_lru, ds, dev_pos, prefetcher, chunk=256)
    finally:
        if prefetcher is not None:
            prefetcher.close()
    metrics_a = evaluate_v53mp(
        model, r3_model, ds, labels_v52, endpoint_labels,
        _CacheView(dev_lru), dev_pos, config,
        train_q_mean=train_q_mean, tau_end=tau_end)
    timings["eval_built_s"] = round(time.perf_counter() - t0, 1)
    _log(f"    done in {timings['eval_built_s']}s")
    del dev_lru
    import gc
    gc.collect()

    # --- run B: pack-served features ----------------------------------------
    _log("[5/6] eval pass B: pack-served features …")
    t0 = time.perf_counter()
    metrics_b = evaluate_v53mp(
        model, r3_model, ds, labels_v52, endpoint_labels,
        view, dev_pos, config,
        train_q_mean=train_q_mean, tau_end=tau_end)
    timings["eval_pack_s"] = round(time.perf_counter() - t0, 1)
    _log(f"    done in {timings['eval_pack_s']}s")

    # --- diff ----------------------------------------------------------------
    _log("[6/6] diffing metrics …")
    diffs = _diff_metrics(metrics_a, metrics_b)
    metrics_passed = not diffs
    note = ""
    if not metrics_passed and str(device).startswith("cuda"):
        # CUDA EGNN forward is not bit-deterministic (atomic-add ordering);
        # the same path run twice on GPU also differs by ~1e-8.  Report the
        # noise floor so the diff is interpretable; FEATURES check above is
        # the device-independent gate.
        note = ("CUDA forward is not bit-deterministic (measured: the SAME "
                "pack path run twice on GPU differs by ~1e-8); interpret "
                "GPU diffs against that noise floor — re-run with "
                "--device cpu for the exact-equal gate")
        _log("    NOTE: " + note)
    checks["metrics_bitwise_equal"] = {
        "passed": metrics_passed,
        "n_diffs": len(diffs),
        "diffs": diffs[:50],
        "device": str(device),
        "note": note,
        "n_fields_note": "full recursive compare of evaluate_v53mp output",
        "n_dev_positions": len(dev_pos),
        "n_dev_positions_full": full_n,
        "reduced": len(dev_pos) != full_n,
    }
    rss_end = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) // 1024
    passed = all(c["passed"] for c in checks.values())
    verdict = {
        "schema": "v53mp_dev_pack_certificate_v1",
        "passed": bool(passed),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "r3_ckpt": str(r3_ckpt),
        "tables_dir": str(tables_dir),
        "pack_dir": str(pack_dir),
        "pack_binding": {
            "features_sha256": str(view.binding.get("features_sha256")),
            "positions_sha256": str(view.binding.get("positions_sha256")),
            "n_positions": int(view.binding.get("n_positions", 0)),
            "feature_level": str(view.binding.get("feature_level")),
        },
        "device": str(device),
        "checks": checks,
        "timings": timings,
        "rss_start_mib": rss_start,
        "rss_end_mib": rss_end,
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    short = str(view.binding.get("positions_sha256"))[:12]
    report_path = report_dir / "dev_pack_certificate_{}.json".format(short)
    report_path.write_text(
        json.dumps(verdict, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    md = ["# Dev pack certificate {}".format(verdict["created_at"]), "",
          "- passed: **{}**".format(verdict["passed"]),
          "- run: `{}`".format(run_dir),
          "- pack: `{}`".format(pack_dir),
          "- features_sha256: `{}`".format(
              verdict["pack_binding"]["features_sha256"]),
          "- positions_sha256: `{}`".format(
              verdict["pack_binding"]["positions_sha256"]),
          "- dev positions: {} (full frozen set {})".format(
              len(dev_pos), full_n),
          "- device: {}".format(device), "",
          "| check | passed | detail |", "|---|---|---|"]
    if "features_bitwise_equal" in checks:
        md.append("| features bitwise equal | {} | {} positions, {} "
                  "diffs |".format(
                      checks["features_bitwise_equal"]["passed"],
                      checks["features_bitwise_equal"]["n_checked"],
                      checks["features_bitwise_equal"]["n_diffs"]))
    md.append("| metrics bitwise equal | {} | {} field diffs |".format(
        checks["metrics_bitwise_equal"]["passed"],
        checks["metrics_bitwise_equal"]["n_diffs"]))
    if checks["metrics_bitwise_equal"].get("note"):
        md.append("")
        md.append("> note: {}".format(
            checks["metrics_bitwise_equal"]["note"]))
    md += ["",
           "## Timings", "",
           "```json", json.dumps(timings, indent=1, sort_keys=True), "```",
           ""]
    report_path_md = report_path.with_suffix(".md")
    report_path_md.write_text("\n".join(md), encoding="utf-8")
    verdict["report_path"] = str(report_path)
    verdict["report_md_path"] = str(report_path_md)
    return verdict


def _require_file(path: Path) -> None:
    if not Path(path).is_file():
        raise SystemExit("error: FILE_MISSING: {}".format(path))


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise SystemExit("error: {}: {}".format(code, detail))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_dev_pack_certificate",
        description="Bitwise A/B certificate: pack-served vs live-built "
                    "final dev eval (plan §5.3)")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--r3-ckpt", default=None,
                        help="matched R3 baseline ckpt "
                             "(default: <run-dir>/../.. is not assumed — "
                             "defaults to the R3 winner used by all runs)")
    parser.add_argument("--pack", default=None,
                        help="pack dir (default: resolved from live "
                             "features_sha + frozen dev positions)")
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--max-dev", type=int, default=None,
                        help="REDUCED cert: seed-17 subsample of the frozen "
                             "dev set (both sides equal; full preferred)")
    parser.add_argument("--device", default="cpu",
                        help="cpu (default; deterministic — the exact-equal "
                             "gate) or cuda (EGNN forward is not bit-"
                             "deterministic there; see the report note)")
    parser.add_argument("--feature-probes", type=int, default=2048,
                        help="positions for the FEATURES bitwise check "
                             "(0 disables)")
    parser.add_argument("--workers", type=int, default=8,
                        help="fork workers for the built-side dev warm "
                             "(0 = serial)")
    args = parser.parse_args(argv)
    r3_ckpt = Path(args.r3_ckpt) if args.r3_ckpt else (
        PROJECT_ROOT / "outputs/pesn/30h_campaign/r3_runs"
        / "5372d9b12807c3f0" / "checkpoint_best.pt")
    _require_file(r3_ckpt)
    verdict = run_certificate(
        Path(args.tables_dir), Path(args.run_dir), r3_ckpt,
        Path(args.pack) if args.pack else None,
        Path(args.report_dir),
        max_dev=args.max_dev, device=args.device, workers=args.workers,
        feature_probes=args.feature_probes)
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
