# -*- coding: utf-8 -*-
"""PES2TS 文档诊断共用扫描库（只读仓库）。

用途：为 FAMILIES / ANOMALIES / RECOMMENDATIONS 提供可复跑的证据口径。
本模块只读取仓库文件，不写任何既有文件；调用方脚本把结果写到本目录。

证据优先级（与任务要求一致）：
  A. 文档内状态头/横幅（头部 15 行内的 status 行、`状态：` 行、`**Superseded**` 横幅）
  B. docs/README.md 的人工状态列（表格行内 `**状态**` 或三列表第 2 列）
  C. docs/plans/PES2TS_文档修订台账.md 的增量登记（由 families_scan 按文件取证）
  D. docs_consistency_config.json 的 historical / required_files / root_scope（门禁钉死）
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = Path(__file__).resolve().parent

EXCLUDE_DIRS = {".git", ".slim", "node_modules", "__pycache__", "data", "outputs",
                "ground_truth", "bin", ".venv", "venv"}
TEXT_EXTS = {".md", ".html", ".txt", ".json", ".yaml", ".yml", ".py", ".sh", ".ps1", ".csv", ".ini"}

EN_STATUS = [
    "Frozen-Preregistered", "Active-Normative", "Evidence-Historical", "Executed-Frozen",
    "Active-Proposed", "Historical-Proposed", "Historical-Design", "Superseded", "SUPERSEDED",
    "Historical", "HISTORICAL", "LEGACY", "Executed", "Proposed", "Frozen", "Closed",
    "Deprecated", "Canonical", "Accepted", "Operational", "Descriptive", "Draft", "Active",
]
CN_STATUS = ["已取代", "被取代", "已被取代", "取代", "历史", "冻结", "预注册", "活跃", "当前", "权威", "规范根",
             "已关闭", "关闭", "执行完毕", "已执行", "未执行", "提案", "拟议", "草案",
             "只读", "非规范", "操作性", "说明性"]
STATUS_ALL = EN_STATUS + CN_STATUS
STATUS_SUPERSEDED = {"Superseded", "SUPERSEDED", "已取代", "被取代", "已被取代"}
STATUS_HISTORICAL = {"Historical", "HISTORICAL", "历史", "LEGACY", "Evidence-Historical",
                     "Historical-Proposed", "Historical-Design"}
STATUS_FROZEN = {"Frozen", "Frozen-Preregistered", "Executed-Frozen", "Executed", "冻结", "执行完毕", "已执行"}
STATUS_ACTIVE = {"Active", "Active-Proposed", "Active-Normative", "Proposed", "Accepted", "活跃", "当前", "提案", "拟议"}
STATUS_AUTHORITY = {"Canonical", "权威", "规范根"}

VERSION_RE = re.compile(r"_v(\d+(?:\.\d+)*)(?:_|\.md|\.html|$)")
CAMPAIGN_VER_RE = re.compile(r"(?<![A-Za-z0-9])([Vv]\d+(?:\.\d+)*(?:-R\d+|-MP|-ENG)?)")
DATE8_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")
DATE_ISO_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
MD_LINK_RE = re.compile(r"\]\(([^()\s]+?\.(?:md|html))(?::\d+)?(?:#[^()\s]*)?\)")
MD_NAME_RE = re.compile(r"([\w.\-（）()\u4e00-\u9fff]+\.md)")
HTML_LINK_RE = re.compile(r"href=[\"']([^\"']+\.html)(?:#[^\"']*)?[\"']")
HEAD_LINES = 80


def _read_text(path: Path, max_bytes: int = 400_000) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _find_status_words(text: str) -> list[str]:
    found = []
    for w in EN_STATUS:
        if re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", text):
            found.append(w)
    for w in CN_STATUS:
        if w in text:
            found.append(w)
    return found


def strict_status(full_text: str) -> tuple[list[str], bool]:
    """只从"状态声明"抽取状态词，避免正文中的'历史/取代'误报。

    识别：`状态：X` / `Status: X` / `| 状态 | X |` / 首行状态词 /
    `> **Superseded**` 类短横幅 / `Superseded by` 语义行。
    """
    lines = full_text.splitlines()
    words: list[str] = []
    banner = False
    for i, line in enumerate(lines[:HEAD_LINES]):
        s = line.strip()
        value = None
        m = re.search(r"(?:建议)?(?:状态|Status|Staus)\s*[:：]\s*(.{0,120})", s)
        if m:
            value = m.group(1)
        else:
            m = re.match(r"^\|\s*(?:状态|Status)\s*\|\s*([^|]{0,120})", s)
            if m:
                value = m.group(1)
        if value is None and i <= 3:
            m = re.match(r"^(?:>\s*)?\*{0,2}([A-Z][A-Za-z\-]{2,40})\*{0,2}\s*$", s)
            if m and _find_status_words(m.group(1)):
                value = m.group(1)
            elif re.match(r"^(SUPERSEDED|HISTORICAL|Historical|Active|Frozen|Executed|Proposed|Closed|Accepted)\b", s):
                value = s[:120]
        if value is not None:
            words.extend(_find_status_words(value))
        if i < 15 and re.search(r"(Superseded by|已被[^。]{0,40}取代|取代者[为:])", s):
            words.extend(_find_status_words(s))
            banner = True
    seen, res = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            res.append(w)
    return res[:8], banner


def parse_readme_status() -> dict[str, str]:
    """从 docs/README.md 的表格解析 path -> 状态串（人工状态列，AGENT.md §9.3 强制同步）。"""
    readme = ROOT / "docs" / "README.md"
    out: dict[str, str] = {}
    if not readme.is_file():
        return out
    text = _read_text(readme, max_bytes=400_000)
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        m = MD_LINK_RE.search(cells[0])
        if not m:
            continue
        raw = m.group(1).split("#", 1)[0]
        tgt = (readme.parent / raw)
        try:
            rel = tgt.resolve().relative_to(ROOT.resolve()).as_posix()
        except (ValueError, OSError):
            continue
        # 状态：优先第 2 列的粗体前缀；否则在第 2 列中找状态词
        status_cell = cells[1] if len(cells) >= 2 else ""
        bold = re.search(r"\*\*([^*]{2,120})\*\*", status_cell)
        candidate = bold.group(1) if bold else status_cell
        ws = _find_status_words(candidate)
        if ws:
            out.setdefault(rel, "**" + candidate.strip()[:120] + "**" if bold else candidate.strip()[:120])
    return out


@dataclass
class Doc:
    path: str
    abs_path: Path
    kind: str
    ext: str
    size: int
    lines: int
    head: str = field(repr=False, default="")
    status: list = field(default_factory=list)
    readme_status: str = ""
    self_declared_superseded: bool = False
    dates: list = field(default_factory=list)
    file_version: str = ""
    campaign_versions: list = field(default_factory=list)
    inbound_refs: int = 0
    ref_sources: list = field(default_factory=list, repr=False)
    bucket: str = ""
    bucket_reason: str = ""
    action: str = ""
    gate_locked: list = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "path": self.path,
            "class_bucket": self.bucket,
            "suggested_action": self.action,
            "gate_locked": ";".join(self.gate_locked),
            "inbound_refs": self.inbound_refs,
            "self_declared_superseded": int(self.self_declared_superseded),
            "head_status": ";".join(self.status),
            "readme_status": self.readme_status,
            "file_version": self.file_version,
            "campaign_versions": ";".join(self.campaign_versions),
            "date_in_name": ";".join(self.dates),
            "bytes": self.size,
            "lines": self.lines,
            "bucket_reason": self.bucket_reason,
        }


def collect_docs() -> list[Doc]:
    docs: list[Doc] = []
    jobs: list[tuple[Path, str]] = []
    for p in sorted((ROOT / "docs").rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_EXTS:
            jobs.append((p, "docs"))
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".ini", ".yaml", ".yml"}:
            jobs.append((p, "root"))
    for p in sorted((ROOT / "artifacts").rglob("model_card.md")):
        jobs.append((p, "artifact"))
    for p in sorted((ROOT / "examples").rglob("README.md")):
        jobs.append((p, "example"))
    p = ROOT / "scripts" / "README.md"
    if p.is_file():
        jobs.append((p, "script-readme"))

    readme_status = parse_readme_status()
    for abs_path, kind in jobs:
        rel = abs_path.relative_to(ROOT).as_posix()
        text = _read_text(abs_path)
        head = "\n".join(text.splitlines()[:HEAD_LINES])
        statuses, banner = strict_status(text)
        dates = extract_dates(head)
        name = abs_path.name
        fv = ""
        m = VERSION_RE.search(name)
        if m:
            fv = m.group(1)
        else:
            m2 = re.search(r"(?<![A-Za-z0-9])v(\d+(?:\.\d+)*)(?:_|\.md|\.html|$)", name)
            if m2:
                fv = m2.group(1)
        camp = sorted(set(CAMPAIGN_VER_RE.findall(name)))
        docs.append(Doc(
            path=rel, abs_path=abs_path, kind=kind, ext=abs_path.suffix.lower(),
            size=abs_path.stat().st_size, lines=text.count("\n") + (1 if text else 0),
            head=head, status=statuses, readme_status=readme_status.get(rel, ""),
            self_declared_superseded=banner, dates=dates, file_version=fv,
            campaign_versions=camp,
        ))
    return docs


def extract_dates(head: str) -> list[str]:
    out = []
    for m in DATE_ISO_RE.finditer(head):
        out.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    for m in DATE8_RE.finditer(head):
        out.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    seen, res = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            res.append(d)
    return res[:6]


def compute_inbound(docs: list[Doc]) -> None:
    by_rel = {d.path: d for d in docs}
    basename_map: dict[str, list[Doc]] = {}
    for d in docs:
        basename_map.setdefault(Path(d.path).name, []).append(d)
    sources = [d for d in docs if d.ext in {".md", ".html"}]
    for src in sources:
        text = _read_text(src.abs_path, max_bytes=800_000)
        targets: set[str] = set()
        for m in MD_LINK_RE.finditer(text):
            raw = m.group(1).split("#", 1)[0]
            if not raw or raw.startswith(("http://", "https://", "mailto:")):
                continue
            cand = raw.lstrip("/")
            for base in (src.abs_path.parent, ROOT):
                try:
                    tgt = (base / cand).resolve()
                except OSError:
                    continue
                if tgt.is_file():
                    try:
                        rel = tgt.relative_to(ROOT.resolve()).as_posix()
                    except ValueError:
                        continue
                    if rel in by_rel and rel != src.path:
                        targets.add(rel)
                    break
        for m in MD_NAME_RE.finditer(text):
            name = m.group(1)
            if name == src.abs_path.name:
                continue
            cands = basename_map.get(name, [])
            if len(cands) == 1 and cands[0].path != src.path:
                targets.add(cands[0].path)
        for t in targets:
            by_rel[t].inbound_refs += 1
            if len(by_rel[t].ref_sources) < 8:
                by_rel[t].ref_sources.append(src.path)


def load_gate_locked() -> dict[str, list[str]]:
    cfg_path = ROOT / "scripts" / "agent_ops" / "docs_consistency_config.json"
    gate: dict[str, list[str]] = {}
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return gate
    for p in cfg.get("historical", []):
        gate.setdefault(p, []).append("historical")
    for p in cfg.get("required_files", {}):
        gate.setdefault(p, []).append("required_files")
    for p in cfg.get("root_scope", []):
        gate.setdefault(p, []).append("root_scope")
    return gate


BUCKET_ORDER = ["治理规范", "活跃计划", "冻结计划", "已取代计划", "历史提案", "设计", "ADR",
                "合同", "证据报告", "运营记录", "快照", "产品文档", "外部说明", "其它"]
ACTION_ORDER = ["原地保留(门禁钉死)", "原地保留(SSOT/索引)", "原地保留(只读证据)",
                "原地保留(快照)", "原地保留(已归档)", "仅加历史横幅",
                "可迁入docs/superseded(0引用且自声明)", "降级为历史提案(不移动)", "需所有者裁决"]


def _has(status: list[str], words: set[str]) -> bool:
    return bool(set(status) & words)


GOVERNANCE_EXACT = {
    "docs/PES2TS_开发宪法_v1.md": "宪法 SSOT（AGENT.md 头部与 §9 指定规范根）",
    "docs/plans/PES2TS_文档修订台账.md": "修订台账，AGENT.md §9.3 强制同步的治理件",
    "docs/plans/PES2TS_阶段定义_G0-G3.md": "阶段定义唯一来源（AGENT.md 头部权威来源）",
    "docs/README.md": "人工文档索引（AGENT.md §9.3 强制同步）",
    "AGENT.md": "宪法执行投影（根 13 份之一）",
    "docs/superseded/README.md": "superseded 收录规则",
    "docs/archive/README.md": "archive 快照规则",
    "docs/infra/wsl_memory_governance.md": "基础设施治理规范",
    "docs/plans/PES2TS_完整开发流方案_整合版_v1.md": "流程总纲（AGENT.md 头部指定）",
}
ROOT_MEMOS = {"V52_PROGRESS_STOP_REPORT.md", "NEXT_DAY_DECISION_MEMO.md",
              "FAILURE_AND_REMAINING_WORK.md", "24H_FINAL_REPORT.md",
              "24H_EXECUTION_TIMELINE.md", "R3_RESULT_SUMMARY.md",
              "V53_ROLLOUT_REPORT.md", "V53MP_RESUME_20261022.md"}


# 人工覆盖：仅用于自动规则无法判定或已人工核实存在状态错位的案例；依据见每条注释。
OVERRIDES = {
    "docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md":
        ("活跃计划", "需所有者裁决", "头部写'建议状态：Active-Proposed'而非正式状态声明；0 引用，需所有者确认是否正式登记"),
    "docs/plans/V541_WP7_frozen_dev_eval_设计方案_v1.0_20261028.md":
        ("设计", "需所有者裁决", "位置异常：设计方案类文档放在 plans/；无状态头，被 V541 文档引用 1 次"),
    "docs/plans/PES2TS_实施路线图.html":
        ("设计", "原地保留(只读证据)", "HTML 路线图（26KB）：形式异常但为图形入口，docs/README plans 表标'参考'"),
    "docs/plans/PES2TS_V5.3-MP_文档同步清单_v1_20261021.md":
        ("冻结计划", "原地保留(只读证据)", "属已执行 V5.3-MP 战役配套清单（docs/README 2026-10-21 条），随战役冻结"),
    "docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md":
        ("设计", "原地保留(SSOT/索引)", "说明性汇总（Descriptive 非规范），docs/README 2026-11-20 条登记为'对齐用'"),
    "docs/plans/PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md":
        ("冻结计划", "原地保留(只读证据)", "docs/README 状态列：Closed — D2-G STOP，负结果归档"),
    "docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md":
        ("冻结计划", "原地保留(只读证据)", "V52R 配套方案，随 D2-G STOP 关闭（台账 2026-10-21 条目）"),
    "docs/plans/PES2TS_扫描规划谱系合并方案_20260930.md":
        ("历史提案", "降级为历史提案(不移动)", "docs/README plans 表状态'参考'（已并入 G2.4/X）；12 处引用，不移动"),
    "docs/plans/PES2TS_整体开发与发布方案_v2.md":
        ("活跃计划", "原地保留(SSOT/索引)", "docs/README '权威文档'表：项目级开发与发布主方案（内容 v3.5）"),
    "docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md":
        ("活跃计划", "原地保留(SSOT/索引)", "docs/README 状态列 Active-Proposed；取代 v1/v2（已入 superseded）"),
    "docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md":
        ("活跃计划", "原地保留(SSOT/索引)", "docs/README 状态列 Active-Proposed；头部状态行'Proposed v3.2'"),
    "docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md":
        ("活跃计划", "原地保留(SSOT/索引)", "docs/README 状态列 Active-Proposed；取代 R3 端点对称 v1/v2"),
    "docs/plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md":
        ("活跃计划", "原地保留(SSOT/索引)", "docs/README 状态列 Active-Proposed（执行细化）；superseded/README 明确未收录原因"),
    "docs/plans/PES2TS_双项目开发与Demo方案.md":
        ("设计", "原地保留(只读证据)", "docs/README plans 表状态'参考'：定位/合同/指标说明，非执行计划"),
    "docs/plans/PES2TS_逐步实施与验证方案.md":
        ("历史提案", "降级为历史提案(不移动)", "docs/README plans 表状态'历史'（早期分步方案）"),
    "docs/plans/PES2TS_统一开发顺序与阶段验收.md":
        ("历史提案", "降级为历史提案(不移动)", "docs/README plans 表状态'历史'（S0–S8 已并入 G0–G4）"),
    "docs/plans/PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md":
        ("活跃计划", "原地保留(SSOT/索引)", "docs/README plans 表状态'参考'：R 工作流范围与验收来源；13 引用"),
    "docs/plans/PES2TS_G1_G2边界审计与补齐任务_20261003.md":
        ("历史提案", "降级为历史提案(不移动)", "docs/README 状态列'抽样审计/合同草案'；G1/G2 边界任务已被后续承接"),
}

def classify(d: Doc, gate: dict[str, list[str]]) -> tuple[str, str, str]:
    p, name, st = d.path, Path(d.path).name, d.status
    d.gate_locked = gate.get(p, [])
    locked = d.gate_locked
    if p in OVERRIDES:
        return OVERRIDES[p]
    readme = d.readme_status
    combined = st + _find_status_words(readme) if readme else st

    if p.startswith("docs/archive/"):
        return "快照", "原地保留(快照)", "docs/archive 版本快照，archive/README 规则2 明文禁止搬入/移动"
    if p.startswith("docs/superseded/"):
        return "已取代计划", "原地保留(已归档)", "已在 docs/superseded/ 归档区，superseded/README 正文不改"
    if p in GOVERNANCE_EXACT:
        reason = GOVERNANCE_EXACT[p]
        if locked:
            return "治理规范", "原地保留(门禁钉死)", reason + "；门禁名单(" + ",".join(locked) + ")"
        return "治理规范", "原地保留(SSOT/索引)", reason
    if p.startswith("docs/infra/"):
        return "治理规范", "原地保留(SSOT/索引)", "基础设施治理/归档操作规则来源"
    if p.startswith("docs/design/decisions/ADR-"):
        if _has(combined, STATUS_SUPERSEDED) or _has(combined, STATUS_HISTORICAL):
            act = "原地保留(门禁钉死)" if locked else "仅加历史横幅"
            return "ADR", act, "ADR 历史/被取代（README 状态列或头部）" + ("；门禁名单" + ",".join(locked) if locked else "；ADR 不原地改写结论")
        return "ADR", "原地保留(SSOT/索引)", "决策记录，一文档一职责（AGENT.md §9.1）"
    if p.startswith("docs/contracts/"):
        act = "原地保留(门禁钉死)" if locked else ("仅加历史横幅" if _has(combined, STATUS_SUPERSEDED | STATUS_HISTORICAL) else "原地保留(SSOT/索引)")
        return "合同", act, "合同/字段字典为接口依据；AGENT.md §9 单一主干编号" + ("；门禁名单" + ",".join(locked) if locked else "")
    if p.startswith("docs/design/"):
        act = "仅加历史横幅" if _has(combined, STATUS_HISTORICAL | STATUS_SUPERSEDED) and not locked else "原地保留(SSOT/索引)"
        return "设计", act, "阶段设计文档（非 ADR 决策记录）"
    if p.startswith("docs/agent_runs/"):
        return "运营记录", "原地保留(只读证据)", "agent_runs 运营编排记录；docs_consistency token_exempt_prefix"
    if d.kind == "root" and name in ROOT_MEMOS:
        return "运营记录", "原地保留(门禁钉死)", "根目录战役 memo（historical/root_scope 门禁输入，移动即红）"
    if p.startswith("docs/plans/fixtures/"):
        return "证据报告", "原地保留(只读证据)", "plans/fixtures 审计/环境证据夹具（被 plans 文档引用的登记数据）"
    if p.startswith("docs/infra/"):
        return "运营记录", "原地保留(只读证据)", "基础设施运维/操作记录"
    if d.kind == "root" and name == "README.md":
        return "外部说明", "原地保留(SSOT/索引)", "仓库门面 README（root_scope 门禁输入；104KB 超大，建议拆分子目录索引）"
    if d.kind == "root" and name in {"CONTRIBUTING.md", "LICENSE"}:
        return "外部说明", "原地保留(SSOT/索引)", "仓库对外说明"
    if p.startswith("docs/reports/"):
        return "证据报告", "原地保留(只读证据)", "docs/reports/** 内容永不改（AGENT.md §9.4）"
    if p.startswith("docs/demo/"):
        if _has(combined, STATUS_HISTORICAL):
            return "设计", "仅加历史横幅", "Demo 设计/审核文档，README 标历史"
        return "设计", "原地保留(只读证据)", "Demo 流程设计/指标/审核文档"
    if p.startswith("docs/plans/"):
        if _has(combined, STATUS_SUPERSEDED):
            lock_reason = "；在 docs_consistency 门禁名单(" + ",".join(locked) + ")，不得移动" if locked else ""
            if locked:
                return "已取代计划", "原地保留(门禁钉死)", "生命周期=已被取代，但门禁钉死原地保留" + lock_reason
            if d.self_declared_superseded and d.inbound_refs == 0:
                return "已取代计划", "可迁入docs/superseded(0引用且自声明)", "自身状态头已声明被取代 + 0 引用 + 不在门禁名单"
            if d.self_declared_superseded:
                return "已取代计划", "需所有者裁决", "自身已声明被取代，但有 %d 处引用，迁入需先改链" % d.inbound_refs
            return "已取代计划", "仅加历史横幅", "README 状态列声明被取代，正文头部缺横幅；先补横幅再评估迁入"
        if _has(combined, STATUS_HISTORICAL):
            if locked:
                return "历史提案", "原地保留(门禁钉死)", "生命周期=历史，且门禁钉死原地保留(" + ",".join(locked) + ")"
            return "历史提案", "降级为历史提案(不移动)", "README/头部标注历史或已被后续方案承接"
        if _has(combined, STATUS_FROZEN):
            if locked:
                return "冻结计划", "原地保留(门禁钉死)", "在 docs_consistency 门禁名单(" + ",".join(locked) + ")，移动断门禁"
            return "冻结计划", "原地保留(只读证据)", "已执行/冻结（Executed-Frozen/Frozen-Preregistered），保留冻结证据"
        if _has(combined, STATUS_AUTHORITY):
            return "活跃计划", "原地保留(SSOT/索引)", "README 标注权威的计划（项目级主方案）"
        if _has(combined, STATUS_ACTIVE):
            return "活跃计划", "原地保留(SSOT/索引)", "README/头部为 Active/Proposed 且未声明被取代"
        if {"Descriptive", "Operational", "说明性", "操作性", "非规范"} & set(combined):
            return "设计", "原地保留(SSOT/索引)", "说明性/操作性非规范文档，规划性质弱于方案"
        return "活跃计划", "需所有者裁决", "计划目录但状态头缺失/自相矛盾，需所有者确认"
    if d.kind == "artifact":
        return "产品文档", "原地保留(只读证据)", "artifacts model_card 属模型产物证据，原地只读"
    if d.kind in {"example", "script-readme"}:
        return "外部说明", "原地保留(SSOT/索引)", "示例/脚本使用说明，面向外部读者"
    if p.startswith("docs/"):
        return "设计", "需所有者裁决", "docs 根/其它子目录文件，需人工确认归属"
    return "其它", "原地保留(只读证据)", "非文档的仓库配置/构建文件（.ini/requirements 等）"


def build(verbose: bool = False) -> list[Doc]:
    docs = collect_docs()
    compute_inbound(docs)
    gate = load_gate_locked()
    for d in docs:
        d.bucket, d.action, d.bucket_reason = classify(d, gate)
    if verbose:
        print(f"collected {len(docs)} docs")
    return docs


def write_csv(docs: list[Doc], out: Path) -> None:
    rows = [d.to_row() for d in docs]
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def bucket_counts(docs: list[Doc]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in docs:
        out[d.bucket] = out.get(d.bucket, 0) + 1
    return {k: out.get(k, 0) for k in BUCKET_ORDER if out.get(k)}


def action_counts(docs: list[Doc]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in docs:
        out[d.action] = out.get(d.action, 0) + 1
    return {k: out.get(k, 0) for k in ACTION_ORDER if out.get(k)}


def print_summary(docs: list[Doc]) -> None:
    print(f"[scope] docs={sum(d.kind=='docs' for d in docs)} root={sum(d.kind=='root' for d in docs)} "
          f"artifact={sum(d.kind=='artifact' for d in docs)} example={sum(d.kind=='example' for d in docs)} "
          f"scripts-readme={sum(d.kind=='script-readme' for d in docs)} total={len(docs)}")
    print("[bucket]", json.dumps(bucket_counts(docs), ensure_ascii=False))
    print("[action]", json.dumps(action_counts(docs), ensure_ascii=False))


if __name__ == "__main__":
    print_summary(build(verbose=True))
