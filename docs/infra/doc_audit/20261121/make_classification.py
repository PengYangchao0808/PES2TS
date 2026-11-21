# -*- coding: utf-8 -*-
"""生成交付物：分类桶映射 CSV（覆盖 task-1 清单 253 条）+ 统计摘要。

输入：.slim/deepwork/20261121_docaudit/inventory/doc_inventory.csv（task-1 清单，253 条）
      .slim/deepwork/20261121_docaudit/disposition/doc_inventory.csv（本任务分类结果，306 条）
输出：classification_mapping.csv（path,class_bucket,suggested_action + 证据列）
      classification_stats.json
只读仓库；只写本目录。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docscan_lib as L  # noqa: E402

AUDIT = Path(__file__).resolve().parents[1]
INV_LIST = AUDIT / "inventory" / "doc_inventory.csv"
MINE = L.OUT_DIR / "doc_inventory.csv"
OUT = L.OUT_DIR / "classification_mapping.csv"

# 分类桶词表（与 task-3 描述一致）
BUCKETS = ["治理规范", "活跃计划", "冻结计划", "已取代计划", "历史提案", "设计", "ADR",
           "合同", "证据报告", "运营记录", "快照", "产品文档", "外部说明", "其它", "未分类-需裁决"]


def load_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    mine = {r["path"]: r for r in load_csv(MINE)}
    inv = load_csv(INV_LIST)
    rows = []
    matched = unmatched = 0
    for r in inv:
        path = r["path"]
        m = mine.get(path)
        if m:
            matched += 1
            rows.append({
                "path": path,
                "class_bucket": m["class_bucket"],
                "suggested_action": m["suggested_action"],
                "gate_locked": m["gate_locked"],
                "inbound_refs": m["inbound_refs"],
                "inventory_doc_type": r.get("doc_type", ""),
                "inventory_declared_status": (r.get("declared_status", "") or "")[:80],
                "evidence_note": m["bucket_reason"],
            })
        else:
            unmatched += 1
            rows.append({
                "path": path, "class_bucket": "未分类-需裁决",
                "suggested_action": "需所有者裁决",
                "gate_locked": "", "inbound_refs": r.get("ref_count", ""),
                "inventory_doc_type": r.get("doc_type", ""),
                "inventory_declared_status": (r.get("declared_status", "") or "")[:80],
                "evidence_note": "task-1 清单有记录但本任务扫描域未覆盖（可能为 docs 外文件），需人工确认",
            })
    with OUT.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bucket_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for r in rows:
        bucket_counts[r["class_bucket"]] = bucket_counts.get(r["class_bucket"], 0) + 1
        action_counts[r["suggested_action"]] = action_counts.get(r["suggested_action"], 0) + 1
    # 候选集合（与 RECOMMENDATIONS.md 对齐）
    candidates = {
        "立即归位_0引用且自声明且非门禁": [r["path"] for r in rows
                                            if "可迁入" in r["suggested_action"]],
        "需加历史横幅": [r["path"] for r in rows if r["suggested_action"] == "仅加历史横幅"],
        "需所有者裁决": [r["path"] for r in rows if r["suggested_action"] == "需所有者裁决"],
        "降级为历史提案": [r["path"] for r in rows if "降级" in r["suggested_action"]],
        "门禁钉死原地保留": [r["path"] for r in rows if "门禁钉死" in r["suggested_action"]],
        "只读证据原地保留": [r["path"] for r in rows if "只读证据" in r["suggested_action"]],
        "快照原地保留": [r["path"] for r in rows if "快照" in r["suggested_action"]],
        "已归档_superseded": [r["path"] for r in rows if "已归档" in r["suggested_action"]],
    }
    stats = {"inventory_rows": len(inv), "matched": matched, "unmatched": unmatched,
             "bucket_counts": {k: bucket_counts.get(k, 0) for k in BUCKETS if bucket_counts.get(k)},
             "action_counts": action_counts,
             "candidates": candidates}
    (L.OUT_DIR / "classification_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("inventory_rows", "matched", "unmatched")}, ensure_ascii=False))
    print(json.dumps(stats["bucket_counts"], ensure_ascii=False))
    print(json.dumps(action_counts, ensure_ascii=False))
    print("candidates:", {k: len(v) for k, v in candidates.items()})


if __name__ == "__main__":
    main()
