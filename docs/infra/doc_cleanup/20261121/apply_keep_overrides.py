"""Lead 裁决覆盖：在 task-7 生成的清理候选上应用"ADR 全留"（34 份）与
"核心集 5 份改判 KEEP"（已由 task-7 的 OVERRIDE_KEEP 落地）。

输出：更新后的 CLEANUP_CANDIDATES.csv / KEEP_SET.csv + CLEANUP_OVERRIDES.md。
只读仓库；只写 .slim/deepwork/20261121_cleanup/cleanup/。
"""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / "CLEANUP_CANDIDATES.csv"
KEEP = HERE / "KEEP_SET.csv"
INV = Path(".slim/deepwork/20261121_docaudit/inventory/doc_inventory.csv")


def main() -> int:
    rows = list(csv.DictReader(io.open(CAND, encoding="utf-8-sig")))
    fields = list(rows[0].keys())

    keep_block = list(csv.DictReader(io.open(KEEP, encoding="utf-8-sig")))
    keep_fields = list(keep_block[0].keys())
    keep_paths = {r["path"] for r in keep_block}

    # ---------- 裁决 1：ADR 全留（34 份，档案树决策链的锚点） ----------
    adr_paths = [r["path"] for r in rows if r["path"].startswith("docs/design/decisions/ADR-")]
    changed = 0
    for r in rows:
        if r["path"].startswith("docs/design/decisions/ADR-") and r["decision"] != "KEEP":
            r["decision"] = "KEEP"
            r["intensity"] = "-"
            r["phase"] = "D0"
            r["reason"] = ("Lead 裁决：ADR 全留（34 份）——档案树「关键决策」字段与"
                           "负结果追溯的唯一锚点；冻结后不可改写，编号=决策指纹")
            changed += 1
            if r["path"] not in keep_paths:
                keep_block.append({
                    "path": r["path"],
                    "layer": "L1",
                    "duty": "架构/接口/科学路线决策留痕（决策+依据+被放弃方案）",
                    "maintenance": "状态头与编号连续性；活跃 6 份双周，其余季度抽 1/3",
                    "owner_role": "科学负责人",
                    "in_archive_tree": "是（X0 / decision_index）",
                })
                keep_paths.add(r["path"])

    # ---------- 统计 ----------
    tiers = {t: sum(1 for r in rows if r["intensity"] == t) for t in ("safe", "standard", "aggressive")}
    keep_n = sum(1 for r in rows if r["decision"] == "KEEP")
    decision = Counter(r["decision"] for r in rows)
    extract_n = decision["ARCHIVE_EXTRACT_THEN_DELETE"] + decision["GATE_CONFIG_MUST_CHANGE"]
    phase_dec = Counter((r["phase"], r["decision"]) for r in rows)
    gcm = [r["path"] for r in rows if r["decision"] == "GATE_CONFIG_MUST_CHANGE"]

    # ---------- 写回 ----------
    with io.open(CAND, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with io.open(KEEP, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keep_fields)
        w.writeheader()
        w.writerows(keep_block)

    # ---------- 覆盖说明 ----------
    lines = [
        "# CLEANUP_OVERRIDES — Lead 最终裁决覆盖（2026-11-21）",
        "",
        "> 本文件是对 `CLEANUP_PLAN.md` 与 `CLEANUP_CANDIDATES.csv` 的**最终裁决覆盖件**：",
        "> 在 task-7 机器判定之上，Lead 追加两项人工裁决并重算三档数字。**以本文件为准**。",
        "",
        "## 1. 覆盖项",
        "",
        "| # | 裁决 | 对象 | 影响 |",
        "|---|---|---|---|",
        "| O1 | **ADR 全留** | `docs/design/decisions/ADR-0001…0034`（34 份） | 25 份从删除集移入 KEEP；其中 safe 档 4、standard 档 9、aggressive 档 12（含原 GCM 3） |",
        "| O2 | **核心集 5 份改判 KEEP** | `PES2TS_contracts_v1.md`、`PES2TS_ACP字段映射_v1.md`、`PES2TS_数据规范与ACP对接优化方案.md`、`PES2TS_CODE_MAP_v2_20261018.md`、`PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md` | 已在 task-7 `OVERRIDE_KEEP` 落地 |",
        "| O3 | **保持退役 3 份** | G2T 字段字典、PESN_IRC 教师合同、PES-C→PES-G 架构迁移 | 先抽结论入档案树 E04/E05/E11 再退役 |",
        "",
        "**裁决理由（ADR 全留）**：① ADR 是“为什么否掉别的方案”的唯一载体，档案树每条目的「关键决策」字段必须引用 ADR 编号；② 成本极低（只读记录，无正文维护义务）；③ 编号=决策指纹，删除/合并会让现行文档 100+ 处 `ADR-00xx` 引用失锚；④ 门禁 `required_files` 含 ADR-0030…0034、`historical` 含 0021/0028/0029。",
        "",
        "## 2. 覆盖后最终数字（权威）",
        "",
        "| 项 | 覆盖前（task-7 初稿） | **覆盖后（最终）** |",
        "|---|---:|---:|",
        f"| KEEP（不删） | 76 | **{keep_n}** |",
        f"| safe 档删除 | 67 | **{tiers['safe']}** |",
        f"| standard 档累计删除 | 129 | **{tiers['safe'] + tiers['standard']}** |",
        f"| aggressive 档累计删除 | 177 | **{tiers['safe'] + tiers['standard'] + tiers['aggressive']}** |",
        f"| 需先抽结论（ARCHIVE + GCM） | 109 | **{extract_n}** |",
        f"| GATE_CONFIG_MUST_CHANGE | 18 | **{len(gcm)}** |",
        "",
        "**决策分布**：" + "、".join(f"`{k}` {v}" for k, v in sorted(decision.items())),
        "",
        "**按 Phase**：" + "；".join(f"{ph}/{d} {v}" for (ph, d), v in sorted(phase_dec.items())),
        "",
        f"**保留集构成**（{keep_n} 份）：核心受管单元 44 组（≈98 份，含 34 ADR + 8 模型卡）+ `docs/archive/**` manifest/README 8 份 + 其余门禁证据/索引。",
        "",
        "## 3. 对 CLEANUP_PLAN.md 的修正点（执行时以本表为准）",
        "",
        "| 原计划位置 | 原数字 | 修正后 |",
        "|---|---|---|",
        "| §0 / §0.1 三档删除数 | 67 / 129 / 177 | **" + f"{tiers['safe']} / {tiers['safe']+tiers['standard']} / {tiers['safe']+tiers['standard']+tiers['aggressive']}" + "** |",
        "| §3.1 D1 | 67 条 | **" + str(tiers["safe"]) + "** 条 |",
        "| §3.2 D2 | +65 → 累计 132 | **+" + str(tiers["standard"]) + " → 累计 " + str(tiers["safe"]+tiers["standard"]) + "** |",
        "| §3.3 D3 | +48 → 累计 182（含 18 GCM） | **+" + str(tiers["aggressive"]) + " → 累计 " + str(tiers["safe"]+tiers["standard"]+tiers["aggressive"]) + "（含 " + str(len(gcm)) + " GCM）** |",
        "| §3.3 D3a/D3b | 32 + 18 | **" + str(decision["ARCHIVE_EXTRACT_THEN_DELETE"] and tiers["aggressive"] - len(gcm)) + " + " + str(len(gcm)) + "** |",
        "| §6 绝不删除清单 | 76 条 KEEP | **" + str(keep_n) + "** 条 KEEP（含 34 ADR） |",
        "| §7 抽取覆盖 | 109 条 | **" + str(extract_n) + "** 条 |",
        "",
        "## 4. 执行前必须重跑的校验",
        "",
        "```powershell",
        "python .slim/deepwork/20261121_cleanup/cleanup/check_deletion.py",
        "# 期望：全部 PASS；三档 = " + f"{tiers['safe']} / {tiers['safe']+tiers['standard']} / {tiers['safe']+tiers['standard']+tiers['aggressive']}" + "；KEEP = " + str(keep_n),
        "```",
        "",
        "> 备注：`check_deletion.py` 中针对 GCM 的提示文案（“18 条”）为初稿口径，实际以本文件的 " + str(len(gcm)) + " 条为准；断言逻辑按 CSV 动态计算，不受影响。",
        "",
    ]
    (HERE / "CLEANUP_OVERRIDES.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"ADR overridden: {changed}; KEEP={keep_n}; tiers safe/standard/aggressive="
          f"{tiers['safe']}/{tiers['safe']+tiers['standard']}/{tiers['safe']+tiers['standard']+tiers['aggressive']}; "
          f"extract={extract_n}; GCM={len(gcm)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
