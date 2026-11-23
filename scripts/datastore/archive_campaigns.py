#!/usr/bin/env python
"""Archive historical PES2TS campaign directories into single-file tombstones.

Strategy ("single-file trace"): before deleting a historical run directory, extract
a lightweight JSON tombstone that keeps provenance (git commit, hashes of key
artifacts, config snapshot, metrics, conclusion pointers, inventory summary) plus
copies of the few small decisive artifacts (summary/metrics/report/model_selection).

Usage (dry-run first):
    python scripts/datastore/archive_campaigns.py --dry-run
    python scripts/datastore/archive_campaigns.py --apply

Safety:
- never deletes anything itself (deletion is a separate, explicit step);
- skips tree entries whose total size is below --min-bytes (small evidence dirs kept);
- writes only under --archives-dir;
- unreadable entries (broken symlinks/reparse points) are skipped, never fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat as statmod
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = REPO / "outputs"
DEFAULT_ARCHIVES = REPO / "data" / "archives_summary"

# directories kept in place even if large: latest V5.6 smoke evidence + ops logs
KEEP = {"v5", "ops"}

KEY_PATTERNS = (
    "run_manifest", "summary", "metrics", "report", "leaderboard",
    "model_selection", "audit", "final", "registry", "conclusion",
)
KEY_EXTS = {".json", ".md", ".csv", ".yaml", ".yml", ".txt", ".jsonl"}
COPY_MAX_BYTES = 2 * 1024 * 1024
COPY_TOTAL_BUDGET = 24 * 1024 * 1024
HASH_MAX_BYTES = 8 * 1024 * 1024
MAX_KEY_FILES = 25
MAX_METRIC_LEAVES = 120


def sha256_file(p: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        remaining = limit
        while True:
            n = 1 << 20 if remaining is None else min(1 << 20, remaining)
            if n <= 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if remaining is not None:
                remaining -= len(b)
    return h.hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:
        return None


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def stat_regular(p: Path):
    """Return stat for a regular file, None for symlinks/broken/unreadable entries."""
    try:
        st = os.stat(p, follow_symlinks=False)
    except OSError:
        return None
    if not statmod.S_ISREG(st.st_mode):
        return None
    return st


def walk_regular(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda e: None):
        for name in filenames:
            p = Path(dirpath) / name
            st = stat_regular(p)
            if st is not None:
                yield p, st


def dir_size(root: Path) -> int:
    total = 0
    for _p, st in walk_regular(root):
        total += st.st_size
    return total


def flatten_metrics(obj, prefix="", out=None, depth=0):
    if out is None:
        out = {}
    if depth > 4 or len(out) >= MAX_METRIC_LEAVES:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_metrics(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
            if len(out) >= MAX_METRIC_LEAVES:
                break
    elif isinstance(obj, list):
        out[prefix or "<list>"] = f"list[{len(obj)}]"
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        s = obj if not isinstance(obj, str) else (obj[:160] + ("…" if len(obj) > 160 else ""))
        out[prefix or "<value>"] = s
    return out


def inspect_campaign(campaign: Path, arch_root: Path, apply: bool) -> dict:
    t0 = time.time()
    files = 0
    total = 0
    ext_counter: Counter[str] = Counter()
    ext_bytes: Counter[str] = Counter()
    largest: list[tuple[int, str]] = []
    candidates: list[tuple[int, int, Path, int]] = []  # (path_depth, -size, path, size)
    config_candidates: list[Path] = []

    for p, st in walk_regular(campaign):
        files += 1
        total += st.st_size
        ext = p.suffix.lower() or "<none>"
        ext_counter[ext] += 1
        ext_bytes[ext] += st.st_size
        largest.append((st.st_size, str(p.relative_to(campaign))))
        low = p.name.lower()
        if ext in KEY_EXTS and any(pat in low for pat in KEY_PATTERNS) and st.st_size <= HASH_MAX_BYTES:
            candidates.append((len(p.relative_to(campaign).parts), -st.st_size, p, st.st_size))
        if ("config" in low) and ext in {".json", ".yaml", ".yml"} and st.st_size <= 512 * 1024:
            config_candidates.append(p)

    largest.sort(reverse=True)
    candidates.sort()
    candidates = candidates[:MAX_KEY_FILES]

    artifacts = []
    metrics: dict = {}
    configs = []
    copied: list[dict] = []
    copy_budget = COPY_TOTAL_BUDGET
    camp_copy_dir = arch_root / "artifacts" / campaign.name

    for _depth, _negsize, p, size in candidates:
        rel = p.relative_to(campaign)
        try:
            digest = sha256_file(p)
        except OSError:
            continue
        artifacts.append({"path": str(rel), "bytes": size, "sha256": digest})
        if apply and size <= COPY_MAX_BYTES and copy_budget >= size:
            try:
                dst = camp_copy_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes())
                copied.append({"path": str(rel), "bytes": size})
                copy_budget -= size
            except OSError:
                pass
        if (len(metrics) < MAX_METRIC_LEAVES and p.suffix.lower() == ".json"
                and size <= 2 * 1024 * 1024):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                for k, v in flatten_metrics(obj, prefix=str(rel)).items():
                    metrics.setdefault(k, v)
                    if len(metrics) >= MAX_METRIC_LEAVES:
                        break
            except Exception:
                pass

    for p in config_candidates[:8]:
        rel = p.relative_to(campaign)
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            text = "<unreadable>"
        try:
            digest = sha256_file(p)
        except OSError:
            continue
        configs.append({
            "path": str(rel),
            "bytes": p.stat().st_size,
            "sha256": digest,
            "content_head": text[:4000],
        })
        if apply:
            try:
                dst = camp_copy_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(text, encoding="utf-8")
            except OSError:
                pass

    report_hits = []
    docs_reports = REPO / "docs" / "reports"
    tokens = [t for t in campaign.name.replace("-", "_").split("_") if len(t) >= 3]
    if docs_reports.is_dir():
        for rp in docs_reports.glob("*.md"):
            if any(tok.lower() in rp.name.lower() for tok in tokens):
                report_hits.append(str(rp.relative_to(REPO)))

    tomb = {
        "schema": "pes2ts_campaign_tombstone_v1",
        "campaign_id": campaign.name,
        "source_path": str(campaign.relative_to(REPO)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_repo_head": git_head(REPO),
        "inventory": {
            "files": files,
            "bytes": total,
            "human": human(total),
            "ext_top": [{"ext": e, "files": c, "bytes": ext_bytes[e]}
                        for e, c in ext_counter.most_common(15)],
            "largest_files": [{"path": p, "bytes": s} for s, p in largest[:15]],
        },
        "key_artifacts": artifacts,
        "config_snapshot": configs,
        "metrics_extracted": metrics,
        "conclusion_pointers": report_hits,
        "conclusion_summary": None,
        "deprecation_reason": ("历史战役运行中间态；按 datastore 单文件留痕策略清盘，"
                               "结论以 conclusion_pointers 中的报告与 key_artifacts 为准"),
        "copied_artifacts": copied,
        "policy": {
            "kept": "key_artifacts + config_snapshot + copied_artifacts + 本 JSON",
            "deleted": "全部其余文件（epoch checkpoint / rollout 帧 / 临时日志 / 重复图件）",
        },
    }
    if apply:
        out = arch_root / "campaigns" / f"{campaign.name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tomb, ensure_ascii=False, indent=1), encoding="utf-8")
        tomb["_tombstone_path"] = str(out.relative_to(REPO))
    tomb["_elapsed_s"] = round(time.time() - t0, 1)
    return tomb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUTS))
    ap.add_argument("--archives-dir", default=str(DEFAULT_ARCHIVES))
    ap.add_argument("--min-bytes", type=int, default=int(0.05 * 1024 ** 3))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reindex", action="store_true",
                    help="only rebuild campaigns_index.json from existing tombstones")
    args = ap.parse_args()

    if not (args.dry_run or args.apply or args.reindex):
        print("refuse: pass --dry-run, --apply or --reindex")
        return 2

    outputs = Path(args.outputs_dir)
    arch = Path(args.archives_dir)
    arch.mkdir(parents=True, exist_ok=True)

    if args.reindex:
        index = rebuild_index(arch)
        print(f"[REINDEX] {len(index['campaigns'])} campaigns, {human(index['total_bytes'])}")
        return 0

    selected = []
    skipped_keep = []
    skipped_small = []
    for d in sorted(p for p in outputs.iterdir() if p.is_dir()):
        if args.only and d.name not in args.only:
            continue
        if d.name in KEEP:
            skipped_keep.append(d.name)
            continue
        size = dir_size(d)
        if args.only is None and size < args.min_bytes:
            skipped_small.append((d.name, size))
            continue
        selected.append((d, size))

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] outputs={outputs} archives={arch}")
    print(f"[{mode}] selected {len(selected)} campaign dir(s):")
    tombstones = []
    for d, size in sorted(selected, key=lambda x: -x[1]):
        print(f"  - {d.name:48s} {human(size)}")
        tomb = inspect_campaign(d, arch, apply=args.apply)
        tombstones.append(tomb)
        if args.apply:
            print(f"      tombstone -> {tomb.get('_tombstone_path')} "
                  f"(artifacts={len(tomb['key_artifacts'])}, metrics={len(tomb['metrics_extracted'])}, "
                  f"configs={len(tomb['config_snapshot'])})")
        else:
            print(f"      would write {len(tomb['key_artifacts'])} key artifacts, "
                  f"{len(tomb['metrics_extracted'])} metrics, "
                  f"{len(tomb['config_snapshot'])} configs")

    total = sum(s for _, s in selected)
    print(f"[{mode}] total selected: {human(total)}")
    if skipped_keep:
        print(f"[{mode}] kept in place: {', '.join(skipped_keep)}")
    if skipped_small:
        print(f"[{mode}] skipped <{human(args.min_bytes)}: {len(skipped_small)} dir(s)")

    if args.apply:
        index = rebuild_index(arch)
        print(f"[APPLY] index -> {arch / 'campaigns_index.json'} "
              f"({len(index['campaigns'])} campaigns, {human(index['total_bytes'])})")
    return 0


def rebuild_index(arch: Path) -> dict:
    """Rebuild the archive index from every tombstone present on disk (idempotent)."""
    campaigns = []
    total = 0
    camp_dir = arch / "campaigns"
    if camp_dir.is_dir():
        for fp in sorted(camp_dir.glob("*.json")):
            try:
                tomb = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            nbytes = int(tomb.get("inventory", {}).get("bytes", 0) or 0)
            total += nbytes
            campaigns.append({
                "campaign_id": tomb.get("campaign_id", fp.stem),
                "bytes": nbytes,
                "files": int(tomb.get("inventory", {}).get("files", 0) or 0),
                "tombstone": str(fp.relative_to(REPO)),
            })
    index = {
        "schema": "pes2ts_campaign_archive_index_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archives_dir": str(arch.relative_to(REPO)),
        "campaigns": campaigns,
        "total_bytes": total,
    }
    (arch / "campaigns_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return index


if __name__ == "__main__":
    sys.exit(main())
