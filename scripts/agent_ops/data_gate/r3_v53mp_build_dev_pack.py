#!/usr/bin/env python3
"""Build the V5.3-MP dev feature pack (S4 plan §5.3 / lane I2).

Builds the F2 features for the FROZEN dev positions ONCE from the v2
columnar loader and persists them pickle-free as per-sample blobs (see
``key_atom_dev_pack_v53mp`` for the codec + binding) under::

    <tables_dir>/dev_packs/<features_sha256>/<positions_sha256>/

Positions follow the DRIVER's rule (shared single implementation
``dev_positions_from_split``: split manifest ``subsets.dev`` curve ids →
dataset enumerate order), so the pack can never diverge from the run
driver's ``dev_positions``.  Feature building reuses the read-only
``FeaturePrefetcher`` fork pool (same ``build_sample_features_v3``, same
level/intent_off — numerics bit-identical to the serial path); workers
are explicit here (the trainer's flag-file gate does not apply to a
one-shot builder).  ``y_teacher`` None-rows are zero-padded exactly like
``_FeatureLRU._finalize`` so the pack serves the FINALIZED dicts the dev
LRU would have served (the bitwise gate depends on it).

Idempotent: an existing pack with a matching binding is kept (logged)
unless ``--overwrite``.  Progress on stderr, summary JSON on stdout,
typed refusals (exit 1) on any invariant violation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    LABEL_STATUSES,
    KeyAtomDatasetV3Error,
    _require,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (  # noqa: E402
    ColumnarKeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (  # noqa: E402
    DEV_PACKS_DIRNAME,
    DevPackError,
    blob_encode,
    dev_positions_from_split,
    features_sha256_of,
    positions_sha256,
    write_dev_feature_pack,
)
from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (  # noqa: E402
    FEATURE_WORKERS_MAX,
    FeaturePrefetcher,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (  # noqa: E402
    FEATURE_LEVEL_V53MP,
)

SPLIT_DEFAULT = "split_manifest_v53mp.json"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, check=True)
        commit = proc.stdout.strip()
        if len(commit) == 40:
            return commit
    except (OSError, subprocess.CalledProcessError):
        pass
    return ""


def _finalize(rec: Any, built: dict) -> dict:
    """Mirror ``_FeatureLRU._finalize`` (y_teacher zero-pad) — bitwise gate."""
    if built.get("y_teacher") is None:
        n_key = len(rec.intent.key_rows())
        built = dict(built)
        built["y_teacher"] = np.zeros((n_key, 3), dtype=np.float32)
    return built


def build_dev_pack(tables_dir: Path, *, workers: int = 8,
                   overwrite: bool = False, limit: int | None = None,
                   split_path: Path | None = None) -> dict:
    """Build the pack for the frozen dev set; returns the summary dict."""
    t_start = time.perf_counter()
    tables = Path(tables_dir)
    _require(tables.is_dir(), "TABLES_DIR", f"{tables} is not a directory")
    split = Path(split_path) if split_path else tables / SPLIT_DEFAULT
    _require(split.is_file(), "SOURCE_MISSING", f"{split} not found")
    workers = max(1, min(int(workers), FEATURE_WORKERS_MAX))

    _log("[1/5] opening v2 columnar dataset …")
    t0 = time.perf_counter()
    ds = ColumnarKeyAtomDatasetV3(
        tables, label_statuses=LABEL_STATUSES, verify_manifest=False)
    _log(f"    {len(ds)} samples, init {time.perf_counter() - t0:.1f}s")

    _log("[2/5] deriving frozen dev positions (driver rule) …")
    split_doc = json.loads(split.read_text(encoding="utf-8"))
    dev_curve_ids = split_doc.get("subsets", {}).get("dev")
    _require(isinstance(dev_curve_ids, (list, tuple)) and dev_curve_ids,
             "SPLIT_SCHEMA", "split manifest has no subsets.dev")
    positions = dev_positions_from_split(ds.samples, dev_curve_ids)
    _require(bool(positions), "DEV_EMPTY", "no dev positions in dataset")
    if limit is not None:
        positions = positions[: int(limit)]
    fsha = features_sha256_of()
    psha = positions_sha256(positions)
    pack_dir = tables / DEV_PACKS_DIRNAME / fsha / psha
    _log(f"    {len(positions)} positions; pack -> {pack_dir}")

    binding_path = pack_dir / f"{psha}.json"
    if binding_path.is_file() and not overwrite:
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        _log("    pack already present with matching binding — keeping "
             "(pass --overwrite to rebuild); "
             "n={}".format(existing.get("n_positions")))
        return {"schema": existing.get("schema"), "pack_dir": str(pack_dir),
                "n_positions": int(existing.get("n_positions", 0)),
                "reused": True, "elapsed_s": 0.0}

    _log(f"[3/5] building features ({workers} fork workers) …")
    t0 = time.perf_counter()
    prefetcher = FeaturePrefetcher(workers, FEATURE_LEVEL_V53MP, False)
    dataset_binding = {
        "data_manifest_sha256": ds.manifest_sha256,
        "idx_curves_sha256": _idx_curves_sha(tables),
        "split_manifest_sha256": _split_sha(split),
        "n_samples": len(ds),
        "d0": float(ds.d0),
        "label_statuses": sorted(str(s) for s in ds.label_statuses),
    }
    n = len(positions)
    built_count = 0

    def _blobs():
        nonlocal built_count
        chunk: list[int] = []
        chunk_size = max(64, workers * 8)
        for pos in positions:
            chunk.append(int(pos))
            if len(chunk) >= chunk_size:
                yield from _encode_chunk(ds, prefetcher, chunk)
                built_count += len(chunk)
                if built_count % 2048 == 0 or built_count == n:
                    rate = built_count / max(time.perf_counter() - t0, 1e-9)
                    _log(f"    {built_count}/{n} "
                         f"({rate:.0f} feat/s, ETA "
                         f"{(n - built_count) / max(rate, 1e-9):.0f}s) …")
                chunk = []
        if chunk:
            yield from _encode_chunk(ds, prefetcher, chunk)
            built_count += len(chunk)
    try:
        binding = write_dev_feature_pack(
            pack_dir, positions, _blobs(),
            feature_level=FEATURE_LEVEL_V53MP, intent_off=False,
            dataset_binding=dataset_binding, git_commit=_git_commit())
    finally:
        prefetcher.close()
    _log(f"    wrote {n} blobs "
         f"({binding['files']['blobs.npy']['bytes'] / 1e6:.0f} MB) "
         f"in {time.perf_counter() - t0:.1f}s")

    _log("[4/5] verifying roundtrip on a sample …")
    from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
        open_dev_feature_pack,
    )
    view = open_dev_feature_pack(pack_dir, feature_level=FEATURE_LEVEL_V53MP)
    rng = np.random.RandomState(17)
    probe = rng.choice(len(positions),
                       size=min(32, len(positions)), replace=False)
    for k in probe:
        pos = positions[int(k)]
        got = view[pos]
        rec = ds.records([pos])[0]
        want = _finalize(rec, build_sample_features_v3(
            ds.to_sample_v3(rec), level=FEATURE_LEVEL_V53MP,
            intent_off=False))
        _require(set(got) == set(want), "VERIFY_KEYS",
                 f"position {pos}: key sets differ")
        for key in want:
            va, vb = want[key], got[key]
            if isinstance(va, np.ndarray):
                _require(va.dtype == vb.dtype
                         and np.array_equal(va, vb), "VERIFY_ARRAY",
                         f"position {pos} key {key}: array mismatch")
            else:
                _require(va == vb, "VERIFY_VALUE",
                         f"position {pos} key {key}: {va!r} != {vb!r}")
    _log("[5/5] done")
    return {
        "schema": binding["schema"],
        "pack_dir": str(pack_dir),
        "n_positions": n,
        "features_sha256": fsha,
        "positions_sha256": psha,
        "feature_level": FEATURE_LEVEL_V53MP,
        "blob_bytes": int(binding["files"]["blobs.npy"]["bytes"]),
        "workers": workers,
        "reused": False,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "created_at": binding["created_at"],
        "builder_git_commit": binding["builder_git_commit"],
    }


def _encode_chunk(ds: ColumnarKeyAtomDatasetV3, prefetcher: FeaturePrefetcher,
                  positions: list[int]):
    """Build one chunk (fork pool, serial fallback) + finalize + encode."""
    recs = ds.records(positions)
    samples = [ds.to_sample_v3(r) for r in recs]
    built_list = prefetcher.build_many(samples)
    if built_list is None or len(built_list) != len(recs):
        built_list = [build_sample_features_v3(
            s, level=FEATURE_LEVEL_V53MP, intent_off=False)
            for s in samples]
    for rec, built in zip(recs, built_list):
        yield blob_encode(_finalize(rec, built))


def _idx_curves_sha(tables: Path) -> str:
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import _sha256
    path = tables / "idx_v2" / "curves.json"
    return _sha256(path) if path.is_file() else ""


def _split_sha(split: Path) -> str:
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import _sha256
    return _sha256(split)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r3_v53mp_build_dev_pack",
        description="Build the frozen-dev F2 feature pack (plan §5.3)")
    parser.add_argument(
        "--tables-dir",
        default=str(PROJECT_ROOT / "outputs/pesn/v53mp/data"))
    parser.add_argument("--split", default=None,
                        help="default: <tables-dir>/split_manifest_v53mp.json")
    parser.add_argument("--workers", type=int, default=8,
                        help="fork-pool size for feature building (1-12)")
    parser.add_argument("--limit", type=int, default=None,
                        help="build only the first N dev positions "
                             "(certificate/smoke only; NEVER for a run pack)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = build_dev_pack(
            Path(args.tables_dir), workers=args.workers,
            overwrite=args.overwrite, limit=args.limit,
            split_path=Path(args.split) if args.split else None)
    except (KeyAtomDatasetV3Error, DevPackError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
