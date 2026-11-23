#!/usr/bin/env python
"""Write the PES2TS datastore top-level registry (version lock).

Scans the datastore root, hashes build artifacts and manifests, and writes
`01_master_index/datastore_registry.json` plus the read-only vault pointer
`00_vault_readonly/vault_manifest.json` (the truth HDF5 files stay in the repo
`data/ground_truth/` so frozen truth_reader paths are preserved; the vault entry
only records their sha256 and access policy).

Usage:
    python scripts/datastore/register_datastore.py --root /pes2ts_datastore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_WIN = Path(__file__).resolve().parents[2]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_WIN), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/pes2ts_datastore")
    ap.add_argument("--archives", default=str(REPO_WIN / "data" / "archives_summary"))
    args = ap.parse_args()

    root = Path(args.root)
    archives = Path(args.archives)
    idx_dir = root / "01_master_index"
    idx_dir.mkdir(parents=True, exist_ok=True)

    def comp(rel: str, required: bool) -> dict:
        p = root / rel
        if not p.exists():
            return {"path": rel, "exists": False, "required": required}
        return {
            "path": rel,
            "exists": True,
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p) if p.is_file() else None,
        }

    master = read_json(idx_dir / "master_split_manifest.json")
    cross = read_json(idx_dir / "frame_crosswalk_manifest.json")
    xtb_registry = read_json(REPO_WIN / "data/manifests/xtb_singlepoints_train_fit_v1.json")
    camps_index = read_json(archives / "campaigns_index.json")

    # 02_core_tensors: hash every channel artifact (source-of-truth row counts come from manifests)
    core = {}
    ct_root = root / "02_core_tensors"
    if ct_root.is_dir():
        for p in sorted(ct_root.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            rel = str(p.relative_to(root))
            core[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    g1_meta = read_json(ct_root / "g1_metadata" / "g1_summary_manifest.json")
    xtb_meta = read_json(ct_root / "xtb_channel" / "xtb_channel_manifest.json")
    geo_meta = read_json(ct_root / "geometry" / "geometry_manifest.json")
    dft_meta = read_json(ct_root / "dft_channel" / "dft_channel_manifest.json")

    def parquet_rows(p: Path):
        if not p.is_file():
            return None
        try:
            import pyarrow.parquet as pq  # local import: registry may run without pyarrow
            return pq.ParquetFile(str(p)).metadata.num_rows
        except Exception:
            return None

    def npy_shape(p: Path):
        if not p.is_file():
            return None
        try:
            import numpy as np
            return list(np.lib.format.read_magic(open(p, "rb"))) and list(np.load(p, mmap_mode="r").shape)
        except Exception:
            return None

    row_counts = {
        "master_split": parquet_rows(idx_dir / "master_split.parquet"),
        "frame_crosswalk": parquet_rows(idx_dir / "frame_crosswalk.parquet"),
        "xtb_row_ledger": parquet_rows(idx_dir / "xtb_row_ledger.parquet"),
        "xtb_points": parquet_rows(ct_root / "xtb_channel" / "xtb_points.parquet"),
        "g1_reaction_summary": parquet_rows(ct_root / "g1_metadata" / "g1_reaction_summary.parquet"),
        "dft_energy_shape": npy_shape(ct_root / "dft_channel" / "dft_energy.npy"),
        "geometry_coords_shape": npy_shape(ct_root / "geometry" / "coords_f64.npy"),
        "expected": {
            "master_split": "66,195 = 64,195 built curves + 2,000 sealed confirm ids (64,195 with --strict-curve-only)",
            "frame_crosswalk": "7,342,581 = sum(curves.json n_nodes)",
            "xtb_row_ledger": "2,877,830 xTB rows",
            "xtb_points": "2,877,830 = 2,877,433 success + 397 scf failures",
            "g1_reaction_summary": "199,217 = inventory rows",
            "dft_energy_shape": "7,342,581",
            "geometry_coords_shape": "123,443,835 atoms x 3",
        },
    }

    # vault: truth files stay at frozen repo paths; record hashes + policy
    source_manifest = read_json(REPO_WIN / "data/manifests/source_manifest.json") or {}
    vault_files = []
    for f in source_manifest.get("files", []):
        vault_files.append({
            "filename": f.get("filename"),
            "sha256": f.get("sha256"),
            "size_bytes": f.get("size_bytes"),
            "location": f"data/ground_truth/{f.get('filename')}",
            "access_policy": "read_only_truth_reader_only",
        })
    vault = {
        "schema": "pes2ts_vault_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": ("真值 HDF5 保留在仓库冻结路径 data/ground_truth/ 下（不复制进 datastore，"
                 "以免破坏 G0/G1 的 truth_reader 冻结路径与隔离审计）；datastore 侧只记录哈希与访问策略。"),
        "truth_access": {
            "single_reader": "pes2ts_core/g0/truth/truth_reader.py",
            "audit_log": "data/manifests/truth_access_log.jsonl",
            "allowlist_guard": "pes2ts_core/utils/truth_guard.py",
        },
        "files": vault_files,
        "derived_truth_files": [
            {"filename": "ts.parquet", "location": "data/ground_truth/ts.parquet"},
            {"filename": "irc_index.parquet", "location": "data/ground_truth/irc_index.parquet"},
        ],
    }
    (root / "00_vault_readonly").mkdir(parents=True, exist_ok=True)
    (root / "00_vault_readonly" / "vault_manifest.json").write_text(
        json.dumps(vault, ensure_ascii=False, indent=1), encoding="utf-8")

    registry = {
        "schema": "pes2ts_datastore_registry_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "root": str(root),
        "capacity_policy": {
            "total_target_gib": 50,
            "note": "Windows 侧只保留 vault 指针/归档/清单；热数据在 WSL 原生 ext4，冷归档在 Windows data/archives_summary。",
        },
        "components": {
            "vault_manifest": {
                "path": "00_vault_readonly/vault_manifest.json",
                "exists": (root / "00_vault_readonly" / "vault_manifest.json").exists(),
                "note": "指向 data/ground_truth 只读真值",
            },
            "master_split": comp("01_master_index/master_split.parquet", required=True),
            "master_split_manifest": comp("01_master_index/master_split_manifest.json", required=True),
            "frame_crosswalk": comp("01_master_index/frame_crosswalk.parquet", required=False),
            "frame_crosswalk_manifest": comp("01_master_index/frame_crosswalk_manifest.json", required=False),
            "excluded_frames": comp("01_master_index/excluded_frames.parquet", required=False),
            "excluded_ledger": comp("01_master_index/excluded_ledger.parquet", required=False),
            "xtb_row_ledger": comp("01_master_index/xtb_row_ledger.parquet", required=False),
            "core_tensors": core,
            "channel_row_counts": row_counts,
            "archives_index": {
                "path": str(archives / "campaigns_index.json"),
                "exists": (archives / "campaigns_index.json").exists(),
                "campaigns": (camps_index or {}).get("campaigns", []),
                "total_bytes": (camps_index or {}).get("total_bytes"),
            },
            "xtb_registry": {
                "path": "data/manifests/xtb_singlepoints_train_fit_v1.json",
                "sha256": xtb_registry and sha256_file(REPO_WIN / "data/manifests/xtb_singlepoints_train_fit_v1.json"),
            },
        },
        "dataset_versions": {
            "idx_v2": {
                "schema": "v53mp_idx_v2",
                "data_manifest_sha256": "d7d47aff26573a8c4df1584155c5993fc37638a9363c2bb96dd33a5ba0251f0e",
                "split_manifest_sha256": "c0d2cb2aabde403277d413a113f57924877cc370784865e65951678690b7d7f3",
                "n_curves": 64195, "n_intents": 128390, "n_frames": 7342581, "n_samples": 14685162,
            },
            "xtb": {
                "dataset_id": "pes2ts_xtb_singlepoints_train_fit_v1",
                "run_id": "train_fit_irc_dense12_v1_native671_802f6243649b",
                "n_reactions": 120222,
                "n_branches": 240444,
                "n_points": 2877830,
                "n_success": 2877433,
                "n_failure": 397,
                "nominal_points": 2912712,
                "points_accounting": 34486722,
                "offline_only": True,
                "truth_assisted": True,
            },
            "split_table": (master or {}).get("counts"),
        },
        "decisions": {
            "point_count": "2,877,830 single points = 2,877,433 success + 397 scf failures; 2,912,712 nominal cap; 34,486,722 is Σ n_points accounting value",
            "cross_channel_energy_diff": "forbidden",
            "sealed_policy": "confirm_2000 and official test are sealed_never_read",
            "xtb_boundary": "offline_only=true, truth_assisted=true",
        },
        "open_questions": ((cross or {}).get("open_questions", [])
                           + (master or {}).get("open_questions", [])),
        "status": ("ACTIVE"
                   if row_counts.get("frame_crosswalk") == 7342581
                   and row_counts.get("xtb_points") == 2877830
                   and row_counts.get("master_split") in (64195, 66195)
                   and row_counts.get("g1_reaction_summary") == 199217
                   else "ACTIVE-BUILD"),
        "stage3_progress": {
            "deleted": [
                "data/raw/pes2ts_xtb_singlepoints_train_fit_v1.tar.zst (16.85 GiB, sha256 verified)",
                "outputs historical campaigns (34 dirs, 70.19 GiB, tombstoned)",
                "/pes2ts_data/v53mp_50k_data/fragments (51 GiB build intermediates)",
                "xTB run/logs (2,877,830 files, 13.9 GiB, scf_log_sha256 preserved in xtb channel)",
                "xTB manifest/shards/*.jsonl + run/results/*.jsonl (converted to parquet)",
            ],
            "kept_for_live_readers": [
                "/pes2ts_data/v53mp_50k_data/*.csv (six tables; read by pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py)",
                "/pes2ts_data/v53mp_50k_data/.light_index_v53mp.pkl + labels_cache_v53mp.npz (read by key_atom_stream_view / label cache)",
                "/pes2ts_data/v53mp_data (20k; subset of 50k but referenced by scripts/campaigns/r3_v53mp_scale_paired_ci.py)",
            ],
        },
    }
    (idx_dir / "datastore_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"registry -> {idx_dir / 'datastore_registry.json'}")
    print(f"vault    -> {root / '00_vault_readonly' / 'vault_manifest.json'}")
    print(f"crosswalk present: {(idx_dir / 'frame_crosswalk.parquet').exists()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
