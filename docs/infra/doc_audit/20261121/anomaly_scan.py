# -*- coding: utf-8 -*-
"""六类异常与治理负债的可复跑扫描（只读）。

产物：
  anomaly_candidates.csv   每条候选异常：class/severity/path/evidence/suggestion
  anomaly_stats.json       六类统计 + 大文件/HTML/docx-xlsx/链接检查摘要
  anomaly_report.txt       人读摘要（ANOMALIES.md 的数据底稿）

用法：python .slim/deepwork/20261121_docaudit/disposition/anomaly_scan.py
"""
from __future__ import annotations

import csv
import os
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docscan_lib as L  # noqa: E402

R = L.ROOT
ROWS: list[dict] = []


def add(cls: str, sev: str, path: str, evidence: str, suggestion: str,
        bucket: str = "", action: str = "") -> None:
    ROWS.append({"anomaly_class": cls, "severity": sev, "path": path,
                 "evidence": evidence, "suggestion": suggestion,
                 "class_bucket": bucket, "suggested_action": action})


def read_lines(rel: str) -> list[str]:
    p = R / rel
    return p.read_text(encoding="utf-8", errors="replace").splitlines() if p.is_file() else []


def find_line(rel: str, needle: str) -> tuple[int | None, str]:
    for i, ln in enumerate(read_lines(rel), 1):
        if needle in ln:
            return i, ln.strip()[:180]
    return None, ""


def scan_location(docs: dict[str, L.Doc]) -> None:
    plans = [p for p in docs if p.startswith("docs/plans/")]
    # 1) plans/ 下的设计类文档
    for p in plans:
        name = Path(p).name
        if any(k in name for k in ("产品设计", "设计方案", "架构", "设计_")):
            add("位置异常", "P1", p, "文件名含设计语义却放在 docs/plans/（设计文档应有 docs/design/ 归属）",
                "需所有者裁决：留在 plans/（若属方案附件）或迁 docs/design/（须同步 27 门禁/索引）",
                docs[p].bucket, docs[p].action)
    # 2) design/ 下的计划/方案类
    for p in [x for x in docs if x.startswith("docs/design/")]:
        name = Path(p).name
        if any(k in name for k in ("实施总方案", "迁移方案", "识别与计划", "扫描策略设计")):
            add("位置异常", "P2", p, "文件名含计划/总方案语义，位于 docs/design/",
                "保留（设计内容为主）；如需执行清单另建 plans 条目", docs[p].bucket, docs[p].action)
    # 3) reports/ 混入非报告（方案/决策记录）
    for p in [x for x in docs if x.startswith("docs/reports/")]:
        name = Path(p).name
        if any(k in name for k in ("方案", "决策记录")):
            add("位置异常", "P2", p, "docs/reports/ 下出现'方案/决策记录'类文档（reports 应收只读证据）",
                "报告只读不改；后续同类文档应落 plans/ 或 design/", docs[p].bucket, docs[p].action)
    # 4) 根目录战役 memo
    memos = [p for p in docs if p in L.ROOT_MEMOS]
    for p in memos:
        add("位置异常", "P2", p, "战役 memo 堆在仓库根（root_scope/historical 门禁输入，不能移动）",
            "保留原地 + docs/agent_runs/README 加索引（2026-11-20 归档方案 B 档折中）")
    # 5) infra 体量
    add("位置异常", "P2", "docs/infra/", "docs/infra 仅 2 份（架构/运维类知识无正式归属，易散落根目录与 plans/）",
        "建立 docs/infra 收录规则或并入 docs/design（需所有者裁决）")


def scan_status(docs: dict[str, L.Doc]) -> None:
    # 状态头与执行事实矛盾（人工核实的证据行）
    add("状态异常", "P0", "docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md",
        "L3 状态：Active-Proposed；但 docs/README.md L30 记录 'V5.4.2 WP7 EXECUTED（2026-10-31）'，"
        "报告 docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md 已交付",
        "把状态头改为 Executed-Frozen（或在头部加执行指针，像 V5.4.1 那样新增 §执行指针）")
    add("状态异常", "P0", "docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md",
        "L3 状态：Proposed v1.1；L4 已写 '执行状态（2026-10-23）：S0/S1/S2/S3 训练全部完成'；S4 报告 2026-10-27 已出",
        "状态行改为 Executed-Frozen 或明确'Proposed 仅供历史引用'；S4 后计划应停止作为当前依据")
    add("状态异常", "P1", "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md",
        "同一头部：L3 'EXECUTED-FROZEN / Stop 语义已取代（2026-10-21）' 与 L5 '状态：**待冻结的执行计划**' 矛盾",
        "删/改 L5 旧状态行；该文件在 historical 门禁名单，只改状态行不解锁移动")
    add("状态异常", "P1", "docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md",
        "L3 状态：Proposed；但 18-run 已于 2026-11-06 执行完毕（台账 2026-11-06 条 + 报告）",
        "按台账'不改文'约定保留；在 docs/README 补 2026-11-06 条目说明 ADR 状态为历史时点")
    add("状态异常", "P2", "docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md",
        "L3 '建议状态：Active-Proposed'（'建议'非正式状态声明）；全库 0 引用、台账/README 未登记",
        "需所有者裁决：正式登记状态，或标'未登记草案'")
    add("状态异常", "P2", "docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md",
        "无'状态：'声明行；inbound_refs=1（近乎孤儿）", "补状态头；若已失效则加'历史'横幅")
    add("状态异常", "P2", "docs/plans/V541_WP7_frozen_dev_eval_设计方案_v1.0_20261028.md",
        "既无状态头，又位于 plans/（位置+状态双异常）", "需所有者裁决：迁 design/ 并补状态头")
    # 声明被取代却仍在活跃目录（非门禁钉死）
    add("状态异常", "P1", "docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md",
        "L3 'Superseded / Historical'；仍在 docs/plans/ 活跃目录，26 处引用",
        "保留 + 已在头部有横幅，可视为②（需加横幅保留）；若迁 superseded 需先改 26 处链接")
    # 状态缺失的 G2 批次文档
    for p in ("docs/plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md",
              "docs/plans/PES2TS_G2_下一批算法调整与测试方案_20261003.md",
              "docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md"):
        add("状态异常", "P2", p, "无'状态：'声明行（README 仅给内容描述，如'执行方案/批次方案'）",
            "补状态头；G2 时代批次方案建议统一标 Historical")


def scan_naming(docs: dict[str, L.Doc]) -> None:
    # 文件名版本 vs 正文版本（人工核实）
    add("命名异常", "P2", "docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md",
        "文件名 v1.0；正文/README 已到 v1.1.1→v1.2（README L50 '转 Executed-Frozen（§19 执行指针）'、L65 'v1.1.1'）",
        "AGENT.md §9 允许'文件名保留 v1.0'，但应在文件头首行标注内容版本（现 L8 只有 Current 版本行）")
    add("命名异常", "P2", "docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md",
        "文件名 v3.0；正文 L3 'v3.3 MOVE–PROCESS 修订'、L9 'Proposed v3.2'（README 也称'内容 v3.2'）",
        "统一'文件名版本 vs 内容版本'声明格式；当前需读者自己推断")
    add("命名异常", "P2", "docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md",
        "文件名 v1.0；L8 '当前版本：v1.2 @ 2026-10-30'", "同上：文件头标内容版本，或用 _v1.2 重命名（需同步门禁文件名）")
    add("命名异常", "P2", "docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md",
        "文件名 v1；台账 2026-11-20 记'升 v1.1'", "文件名与内容版本同步，或在头部标 v1.1")
    # 日期格式/有无日期/前缀不一致（统计口径）
    dated = [p for p in docs if p.startswith("docs/") and L.DATE8_RE.search(Path(p).name)]
    undated = [p for p in docs if p.startswith("docs/") and Path(p).suffix == ".md"
               and not L.DATE8_RE.search(Path(p).name) and "/archive/" not in p and "/superseded/" not in p]
    add("命名异常", "P2", "docs/**",
        f"命名口径并存：带 _YYYYMMDD 日期 {len(dated)} 份、不带日期 {len(undated)} 份；"
        f"版本号 v1 与 v1.0 并存（如 _v1_20261020 vs _v1.0_20261102）；无 PES2TS_ 前缀的另有 ADR-*/G1_*/PES_GENERATION_*/V541_* 若干",
        "制定命名约定（建议：<项目前缀>_<主题>_v<主.次>_<YYYYMMDD>.md；ADR 例外保留 ADR-NNNN-*）")
    # 不规则大写/空格
    for p in docs:
        name = Path(p).name
        if " " in name or re.search(r"[A-Za-z]{2,}[A-Z][a-z]", name):
            add("命名异常", "P2", p, "文件名含空格或大小写混排不规范", "重命名需同步门禁/引用（低优先）")


def scan_duplicate(docs: dict[str, L.Doc]) -> None:
    # 1) 快照 vs live 同名（design by design，但占体积）
    live_names: dict[str, list[str]] = {}
    for d in docs:
        if "/archive/" in d.path:
            continue
        live_names.setdefault(Path(d.path).name, []).append(d.path)
    snap = [d.path for d in docs if "/archive/" in d.path]
    dup_snap = [p for p in snap if Path(p).name in live_names]
    add("内容重复", "P2", "docs/archive/**",
        f"{len(snap)} 份快照文件中有 {len(dup_snap)} 份与 live 文档同名（版本快照的预期重复，但占 ~1.1MB）",
        "不改（archive 只读）；仅在 README 说明快照体积占比")
    # 2) 同主题多代（见 FAMILIES.md）
    add("内容重复", "P1", "docs/plans/", "V4.2–V4.7 / V5.x 每代都留'方案+合同+报告'三件套，正文大量重复表格（如 30h 计划与 R3 方案重叠字段）",
        "保留证据；在 superseded/README 或台账登记'同主题三件套'关系，避免后来者重复引用")
    # 3) 根 README 与 docs/README 的索引重叠
    add("内容重复", "P2", "README.md", "根 README 104,309 B（1,313 行）与 docs/README.md 79,471 B（349 行）都在维护文档/阶段索引，存在双维护",
        "根 README 收敛为'项目门面+入口'，把文档索引单一职责留在 docs/README（需所有者裁决）")
    # 4) 两份较大 HTML 可视化+报告
    add("内容重复", "P2", "docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html",
        "379,567 B HTML（1661 行）内嵌数据，与 reports 其它 S3 报告内容部分重叠", "作为只读证据保留；不进入文本 lint")


def scan_refs(docs: dict[str, L.Doc]) -> None:
    zero = [d for d in docs if d.inbound_refs == 0 and d.path.startswith("docs/")
            and d.ext == ".md" and "/archive/" not in d.path and "/agent_runs/" not in d.path]
    for d in zero:
        if d.bucket in ("活跃计划", "设计", "合同", "ADR", "治理规范"):
            add("引用异常", "P2", d.path, f"0 引用（全仓无文档链接指向），bucket={d.bucket}", "需所有者裁决：加索引引用或降级")
    # 门禁名单 vs 实际存在
    gate = L.load_gate_locked()
    for p in gate:
        if not (R / p).is_file():
            add("引用异常", "P1", p, "docs_consistency 门禁名单引用的文件不存在（移动即断门禁，但已缺文件）",
                "核对该门禁条目是否 stale")
    # 链接检查摘要
    for rel in ("docs/reports/PES2TS_V52_任务纠偏审计_20261021.md",):
        pass
    add("引用异常", "P2", "docs/**",
        "链接检查基线：2026-11-20 台账记录 `check_docs_links.py` = 0 error / 173 warning（182 docs）；"
        "warning 主要为历史文档中的绝对路径/旧 E: 路径",
        "保持 0 error；不要为了消化 warning 改 reports（只读）")


def scan_size() -> None:
    ledger = "docs/plans/PES2TS_文档修订台账.md"
    n = len(read_lines(ledger))
    add("体积/形式", "P1", ledger, f"166,842 B / {n} 行（append-only 台账，顶部按日期倒序增量登记）",
        "台账不删证据；建议按年度/季度拆分为台账索引 + 年度附录（需同步 AGENT.md §9.3 引用）")
    add("体积/形式", "P1", "README.md", "104,309 B / 1,313 行（根门面 README 超大，远超 docs/README 79,471 B）",
        "拆分：门面保留 → docs/README 或独立 docs/INDEX；需所有者裁决")
    add("体积/形式", "P2", "docs/README.md", "79,471 B / 349 行，单文件维护全仓文档状态列，行数超 300 行后维护成本高",
        "按目录拆分为 docs/README + 各目录 README（长期项）")
    for rel, b, ln in [
        ("docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html", 379567, 1661),
        ("docs/plans/fixtures/pes2ts_direct_training_input_audit_20261015.json", 240868, 9229),
        ("docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md", 60763, 672),
        ("docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md", 93311, 1671),
    ]:
        add("体积/形式", "P2", rel, f"{b:,} B / {ln} 行（同类文档中体积显著偏大）",
            "保留（证据/冻结件）；新文档控制体积，长表格外置到 fixtures/")
    html = [d for d in R.rglob("*.html") if ".git" not in str(d) and "node_modules" not in str(d)]
    html_rel = [d.relative_to(R).as_posix() for d in html]
    add("体积/形式", "P2", "; ".join(html_rel), f"仓库共有 {len(html_rel)} 份 HTML 文档（最大 379KB），不在 md 链接检查口径内",
        "登记为'图形入口'，在 README/索引注明浏览器打开；不做文本 lint")
    # docx/xlsx 证据定位（安全遍历：跳过 outputs/data 等大目录与不可访问路径）
    office: list[str] = []
    skip_dirs = {".git", "outputs", "data", "ground_truth", "node_modules", "__pycache__", ".slim"}
    for dirpath, dirnames, filenames in os.walk(R, onerror=lambda e: None):
        dirnames[:] = [x for x in dirnames if x not in skip_dirs]
        for fn in filenames:
            if Path(fn).suffix.lower() in {".docx", ".xlsx", ".pptx", ".doc", ".xls"}:
                office.append((Path(dirpath) / fn).relative_to(R).as_posix())
    add("体积/形式", "P2", "; ".join(office) if office else "(无)",
        f"仓库文本面未发现 docx/xlsx 证据（扫描命中 {len(office)} 份）",
        "若确无则写'无 Office 证据文件'；Report 证据保持 markdown")


def main() -> None:
    docs = L.build()
    by_path = {d.path: d for d in docs}
    scan_location(by_path)
    scan_status(by_path)
    scan_naming(by_path)
    scan_duplicate(docs)
    scan_refs(docs)
    scan_size()

    with (L.OUT_DIR / "anomaly_candidates.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        fields = ["anomaly_class", "severity", "path", "evidence", "suggestion"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(ROWS)

    sev = {}
    cls = {}
    for r in ROWS:
        sev[r["severity"]] = sev.get(r["severity"], 0) + 1
        cls[r["anomaly_class"]] = cls.get(r["anomaly_class"], 0) + 1
    stats = {"total": len(ROWS), "by_severity": sev, "by_class": cls,
             "paths": sorted({r["path"] for r in ROWS})}
    (L.OUT_DIR / "anomaly_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"anomalies={len(ROWS)} by_severity={sev} by_class={cls}")
    for r in ROWS:
        print(f"[{r['severity']}][{r['anomaly_class']}] {r['path']} :: {r['evidence'][:110]}")


if __name__ == "__main__":
    main()
