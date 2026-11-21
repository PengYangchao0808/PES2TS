"""Lead 重算 LINK_FIX：裁决后删除集变化（ADR 全留），需按最终 KEEP/删除集重建
"KEEP 文档 → 待删文档" 的 markdown 链接修正清单。

输出：覆盖 CLEANUP_LINKFIX 前先备份原文件；输入为更新的 CLEANUP_CANDIDATES.csv。
只读仓库；只写 .slim/deepwork/20261121_cleanup/cleanup/。
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CAND = HERE / "CLEANUP_CANDIDATES.csv"
LINKFIX = HERE / "LINK_FIX.csv"

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SCAN_ROOTS = [Path("README.md"), Path("AGENT.md")]
SKIP_PREFIX = ("docs/archive/", "docs/infra/doc_audit/", "docs/infra/doc_cleanup/")


def norm_target(referrer: str, target: str) -> str | None:
    t = target.strip()
    if t.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if re.match(r"^[A-Za-z]:[\\/]", t) or t.startswith(("/", "\\")):
        return None
    t = t.split("#", 1)[0].split("?", 1)[0]
    if not t:
        return None
    base = Path(referrer).parent
    try:
        rel = (base / t).as_posix()
    except Exception:
        return None
    parts: list[str] = []
    for seg in rel.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts)


def main() -> int:
    rows = list(csv.DictReader(io.open(CAND, encoding="utf-8-sig")))
    keep = {r["path"] for r in rows if r["decision"] == "KEEP"}
    deleted = {r["path"]: r for r in rows if r["decision"] != "KEEP"}

    md_files: list[str] = []
    for p in SCAN_ROOTS:
        if (ROOT / p).is_file():
            md_files.append(p.as_posix())
    for p in sorted((ROOT / "docs").rglob("*.md")):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(SKIP_PREFIX):
            continue
        md_files.append(rel)

    out_rows = []
    seen = set()
    for referrer in md_files:
        if referrer not in keep:
            continue  # b2：LINK_FIX 只针对 KEEP 集活跃扫描面
        text = (ROOT / referrer).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in LINK_RE.finditer(line):
                target = norm_target(referrer, m.group(1))
                if not target or target not in deleted:
                    continue
                key = (referrer, lineno, target, m.group(1))
                if key in seen:
                    continue
                seen.add(key)
                info = deleted[target]
                out_rows.append({
                    "referrer_path": referrer,
                    "line_no": str(lineno),
                    "context": line.strip()[:160],
                    "old_target": m.group(1),
                    "new_target": "删除该链接/改指档案树条目",
                    "action": "改为档案树指针或删行",
                    "batch": info["phase"].replace("D0", "D1"),
                })

    bak = LINKFIX.with_suffix(".csv.bak")
    if not bak.exists() and LINKFIX.exists():
        bak.write_bytes(LINKFIX.read_bytes())
    with io.open(LINKFIX, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["referrer_path", "line_no", "context",
                                           "old_target", "new_target", "action", "batch"])
        w.writeheader()
        w.writerows(out_rows)

    print(f"kept={len(keep)} deleted={len(deleted)} linkfix_rows={len(out_rows)} "
          f"referrers={len({r['referrer_path'] for r in out_rows})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
