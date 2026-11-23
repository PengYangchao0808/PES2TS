"""构建全量文档处置映射 DISPOSITION.csv（2026-11-21 审计）。

输入：classification_mapping.csv（253 条分类结论，来自 doc-classifier）
规则：开发文档规范化归档方案 v2.0 §5.3（状态归并）、§7.2（动作矩阵）、§9（批次）
输出：DISPOSITION.csv（253 行）、LINK_FIX.csv（条件迁移的引用修正清单）

设计要点：本脚本只做"规则化判定"，不做内容判断；所有需要所有者裁决的项一律
batch=B3、blocker=需所有者裁决，绝不擅自 move。默认 action 为 keep 或
retain-readonly；banner 只加状态行；move 仅在 B3（待批准）出现。
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # docs/infra/doc_audit/20261121 -> repo root
AUDIT = Path(__file__).resolve().parent
SRC = AUDIT / "classification_mapping.csv"

# --- 状态归并（方案 §5.3） -------------------------------------------------
STATE_RULES = [
    ("Superseded", ("superseded", "已取代")),
    ("Historical", ("historical", "历史", "前版记录", "参考（已并入")),
    ("Executed-Frozen", ("executed-frozen", "执行完毕", "已实现", "implemented", "executed")),
    ("Frozen-Preregistered", ("frozen-preregistered",)),
    ("Active-Normative", ("active-normative",)),
    ("Accepted", ("accepted", "已接受")),
    ("Active-Proposed", ("active-proposed", "proposed", "执行中", "待实施", "执行方案",
                         "批次方案", "设计稿", "当前")),
    ("Canonical", ("权威", "canonical", "ssot")),
]

# 证据类：状态固定为 Evidence-ReadOnly / Snapshot
EVIDENCE_PREFIXES = ("docs/reports/",)
SNAPSHOT_PREFIXES = ("docs/archive/",)
OPS_PREFIXES = ("docs/agent_runs/",)


# g_v_map / render-stage 类钉死（不在 docs_consistency_config 名单内，但移动会断
# render_gv_map --check 或使 evidence 失效）：来自 GOVERNANCE_CONSTRAINTS.md §3.5
GVM_PINNED = {
    "docs/reports/PES2TS_实现报告_20261018.md",                                  # V4.2–V4.7 evidence
    "docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md",          # V5.4.1 evidence
    "docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md",           # V5.4.2 evidence
    "V53_ROLLOUT_REPORT.md",                                                    # V5.3-ENG evidence
    "docs/design/decisions/ADR-0026-V5冲刺key-atom位移合同与OOF语义.md",            # V5_SPRINT evidence
    "docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md",                # V5_1A_SPRINT evidence
    "docs/plans/PES2TS_阶段定义_G0-G3.md",                                       # stage_doc
    "docs/design/PES2TS_G_V_MAPPING_v1_20261018.md",                             # render 目标（生成物）
}


def canonical_status(path: str, doc_type: str, declared: str) -> str:
    if path.startswith(SNAPSHOT_PREFIXES):
        return "Snapshot"
    if path.startswith(EVIDENCE_PREFIXES) or doc_type == "报告":
        return "Evidence-ReadOnly"
    if path.startswith("artifacts/") and path.endswith("model_card.md"):
        return "Evidence-ReadOnly"  # 模型卡=只读产品证据（§4.2-L3）
    low = declared.lower()
    for canonical, keys in STATE_RULES:
        if any(k in low for k in keys):
            return canonical
    if path.startswith(OPS_PREFIXES) or doc_type == "运营记录":
        # L4 运营记录：非状态标记，不参与 G5 状态校验（方案 §5.1 注）
        return "运营记录（非状态标记）"
    if doc_type == "合同":
        return "Active-Normative"
    if doc_type == "ADR":
        return "Accepted"
    return "未登记"


# --- 动作矩阵（方案 §7.2 / §9） --------------------------------------------
def disposition(row: dict, p0_paths: set[str], p1_paths: set[str]) -> dict:
    path = row["path"]
    suggested = (row["suggested_action"] or "").strip()
    gate = (row["gate_locked"] or "").strip()
    refs = int(row["inbound_refs"] or 0)
    doc_type = row["inventory_doc_type"] or ""
    declared = row["inventory_declared_status"] or ""

    action, blocker, batch = "keep", "无", "B0"
    reason = suggested
    if gate:
        # 门禁钉死优先：gate_locked 非空即不得移动（与 config 精确一致）
        blocker, batch = "门禁钉死", "B0"
        if "只读证据" in suggested:
            action, blocker = "retain-readonly", "报告只读"
        elif "快照" in suggested:
            action, blocker = "retain-readonly", "快照只读"
    elif "门禁钉死" in suggested:
        blocker, batch = "门禁钉死", "B0"
    elif "SSOT" in suggested or "索引" in suggested:
        blocker, batch = "SSOT/索引", "B0"
    elif "只读证据" in suggested or "快照" in suggested:
        action, blocker, batch = "retain-readonly", "报告只读" if "证据" in suggested else "快照只读", "B0"
    elif "已归档" in suggested:
        blocker, batch = "已归档（只读）", "B0"
    elif "仅加历史横幅" in suggested or "降级为历史提案" in suggested:
        action, batch = "banner", "B0"
    elif "需所有者裁决" in suggested:
        blocker, batch = "需所有者裁决", "B3"
    elif "未分类" in suggested:
        if path.startswith("docs/infra/doc_audit/"):
            action, blocker, batch = "retain-readonly", "报告只读", "B0"
        elif path in ("LICENSE",) or path.startswith(("requirements", "pytest.ini")):
            action, blocker, batch = "out-of-scope", "无", "B0"
        else:
            action, blocker, batch = "keep", "需所有者裁决", "B3"
    else:
        blocker, batch = "需所有者裁决", "B3"

    # F19：自身已声明被取代（前版记录）但仍有引用 → 条件迁移，需批准
    f19 = ("EGNN_BC_IQL_SAC下一步完整训练方案_v2" in path
           or "EGNN_BC_IQL_SAC实施与训练计划_v3" in path
           or "IRC_BC_IQL_SAC直接位移研究方案_v1" in path)
    target = ""
    if f19:
        action, batch, blocker = "move", "B3", "需所有者裁决"
        target = "docs/superseded/" + Path(path).name
        reason = f"自身已声明被 v4 取代；引用 {refs} 处需先修链；迁 superseded 需所有者批准"

    priority = "P2"
    if path in p0_paths:
        priority = "P0"
    elif path in p1_paths:
        priority = "P1"
    elif batch == "B3":
        priority = "P1" if f19 else "P2"
    elif action == "banner":
        priority = "P1"

    # 不可移动来源（pinned_by）：显式记录"为什么不能动"，不改变 blocker 语义
    if gate:
        pinned_by = "docs_consistency"
    elif path in GVM_PINNED:
        pinned_by = "g_v_map/render"
    elif path in ("R3_RESULT_SUMMARY.md",):
        pinned_by = "软登记(门禁配置待同步)"
    else:
        pinned_by = "无"

    if batch == "B0":
        verification = ("python infra/governance/check_docs_links.py → 0 error；"
                        "python infra/governance/docs_consistency.py --strict → CLEAN")
    elif action == "move":
        verification = ("git grep 引用清零 → git mv → 修链 → links 0 error "
                        "→ superseded/README 索引 → 台账登记")
    else:
        verification = "改状态行后复跑 4 门（links/consistency/gv_map/registry tests）"
    if action == "retain-readonly":
        verification = "不修改内容；仅索引与指针可加"

    return {
        "path": path,
        "zone": zone_of(path),
        "doc_type": doc_type,
        "declared_status": declared,
        "canonical_status": canonical_status(path, doc_type, declared),
        "target_zone": "docs/superseded" if target else zone_of(path),
        "action": action,
        "target_path": target,
        "priority": priority,
        "batch": batch,
        "blocker": blocker,
        "reason": reason[:120],
        "evidence": (row.get("evidence_note") or "")[:240],
        "verification": verification,
        "inbound_refs": str(refs),
        "gate_locked": gate,
        "pinned_by": pinned_by,
    }


def posix_relpath(from_dir: Path, target: Path) -> str:
    """从 from_dir（仓库相对）到 target（仓库相对）的 posix 相对链接。"""
    import posixpath
    base = from_dir.as_posix()
    if base in ("", "."):
        return target.as_posix()
    return posixpath.relpath(target.as_posix(), base)


def zone_of(path: str) -> str:
    p = path.replace("\\", "/")
    if "/" not in p:
        return "root"
    if p.startswith("docs/plans/"):
        return "docs/plans"
    if p.startswith("docs/design/decisions/"):
        return "docs/design/decisions"
    if p.startswith("docs/design/"):
        return "docs/design"
    if p.startswith("docs/contracts/"):
        return "docs/contracts"
    if p.startswith("docs/reports/"):
        return "docs/reports"
    if p.startswith("docs/archive/"):
        return "docs/archive"
    if p.startswith("docs/superseded/"):
        return "docs/superseded"
    if p.startswith("docs/agent_runs/"):
        return "docs/agent_runs"
    if p.startswith("docs/infra/"):
        return "docs/infra"
    if p.startswith("docs/demo/"):
        return "docs/demo"
    if p.startswith("docs/"):
        return "docs/root"
    if p.startswith("examples/"):
        return "examples"
    if p.startswith("artifacts/"):
        return "artifacts"
    if p.startswith("scripts/"):
        return "scripts"
    if p.startswith(("data/", "outputs/")):
        return "data-outputs"
    return "other"


def main() -> int:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    if not rows:
        print("FAIL: empty classification_mapping.csv")
        return 2

    # P0/P1：来自 ANOMALIES.md 的精编结论（A01/A02 为 P0；A03/A04/A05 等为 P1）
    p0_paths = {
        "docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md",
        "docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md",
    }
    p1_paths = {
        "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md",
        "docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md",
        "docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md",
        "docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md",
        "docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md",
    }

    out = []
    seen: Counter[str] = Counter()
    for row in rows:
        rec = disposition(row, p0_paths, p1_paths)
        out.append(rec)
        seen[rec["path"]] += 1

    fields = ["path", "zone", "doc_type", "declared_status", "canonical_status",
              "target_zone", "action", "target_path", "priority", "batch", "blocker",
              "reason", "evidence", "verification", "inbound_refs", "gate_locked",
              "pinned_by"]
    with (AUDIT / "DISPOSITION.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    # LINK_FIX.csv：条件迁移（F19 三份）的引用修正清单，action=pending-approval/no-change
    link_rows = []
    f19_names = {Path(r["path"]).name for r in out if r["action"] == "move"}
    for name in f19_names:
        for (referrer, line_no), line in _grep_referrers(name):
            target_repo = f"docs/superseded/{name}"
            referrer_name = Path(referrer).name
            if referrer_name in f19_names:
                # 同迁兄弟文件：两者同目录迁移，相对链接无需改动
                markdown_link, action = name, "no-change"
            else:
                markdown_link = posix_relpath(Path(referrer).parent, Path(target_repo))
                action = "pending-approval"
            link_rows.append({
                "referrer_path": referrer,
                "line_no": str(line_no),
                "context": line.strip()[:160],
                "old_target": name,
                "new_target": target_repo,
                "markdown_link": markdown_link,
                "action": action,
            })
    with (AUDIT / "LINK_FIX.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["referrer_path", "line_no", "context",
                                          "old_target", "new_target", "markdown_link",
                                          "action"])
        w.writeheader()
        w.writerows(link_rows)

    # 校验
    actions = Counter(r["action"] for r in out)
    batches = Counter(r["batch"] for r in out)
    prios = Counter(r["priority"] for r in out)
    dup = [p for p, c in seen.items() if c > 1]
    missing = [r["path"] for r in out if not r["action"]]
    print(f"rows={len(out)} unique={len(seen)} dup={len(dup)} missing_action={len(missing)}")
    print("actions:", dict(actions))
    print("batches:", dict(batches))
    print("priorities:", dict(prios))
    print("link_fix_rows:", len(link_rows))
    ok = not dup and not missing and len(out) == len(rows) and len(out) == 253
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _grep_referrers(name: str):
    """返回 (referrer_path, line_no, line) 列表；用 git grep 保证与仓库一致。"""
    import subprocess
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", "grep", "-n", "--no-color", "--", name,
         ":!docs/archive", ":!docs/superseded"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            yield (parts[0], int(parts[1])), parts[2]


if __name__ == "__main__":
    sys.exit(main())
