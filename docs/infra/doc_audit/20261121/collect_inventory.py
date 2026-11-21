#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PES2TS 开发文档全量清单采集器 (collect_inventory.py)

用途
----
对 PES2TS 仓库做一次只读的"文档盘点"，产出:
  - doc_inventory.csv   全量清单 (16 字段, UTF-8 无 BOM)
  - doc_inventory.json  同数据 + stats 汇总
  - inventory_stats.md  人读统计报告

用法
----
  python collect_inventory.py                       # 默认: 自动定位仓库根, 输出到本脚本目录
  python collect_inventory.py --root E:\\path\\repo --out DIR
  python collect_inventory.py --debug               # 额外写 _debug_fields.txt 供人工抽检

设计约束
--------
  * 只读仓库: 不修改/移动/删除任何既有文件; 只写 --out 目录。
  * Windows 控制台 GBK: 屏幕只打印 ASCII 摘要, 中文一律写 UTF-8 文件。
  * 不递归 data/ output/ (12GB/1.3GB), 只在白名单目录扫描。
  * git 元数据: git -c core.quotepath=false log --follow --format=%ad --date=short -- <path>
    显式 encoding='utf-8', errors='replace'。

口径 (CALIBER) 详见 inventory_stats.md 顶部。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path

try:  # 保证控制台不因中文/emoji 崩溃
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 常量 / 口径
# ---------------------------------------------------------------------------

ROOT_DIRNAMES = ("docs", "examples", "artifacts", "scripts")      # 递归扫描
TOPLEVEL_DIRNAMES = ("data", "outputs")                          # 只取顶层说明文件
CONFIG_DIRNAME = "config"                                        # 只取 .md/.txt/.rst 说明文档
SKIP_DIRNAMES = {
    ".git", ".slim", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "node_modules", ".venv", "venv", ".idea", ".vscode",
    ".ipynb_checkpoints", ".cache",
}

# 文档类扩展名（决定"什么是文档"）
DOC_EXTS_RECURSE = {".md", ".html"}                 # docs/ examples/ artifacts/ scripts/
DOC_EXTS_TOPDIR = {".md"}                           # data/ outputs/ 顶层
ROOT_TXT_EXTS = {".md", ".txt"}                     # 仓库根: *.md + *.txt
ROOT_EXTRA_NAMES = {"LICENSE"}                      # 根目录法务/说明无扩展文件

# ref_count 语料范围
REF_CORPUS_EXTS = {".md", ".py", ".json", ".yaml", ".yml"}
REF_EXCLUDE_DIRS = {".git", ".slim", "data", "outputs", "__pycache__",
                    ".pytest_cache", ".ruff_cache", ".mypy_cache",
                    "node_modules", ".venv", "venv"}
REF_EXCLUDE_PREFIXES = ("docs/archive/",)          # 归档快照不计引用

ZONE_BY_DOCS_SUBDIR = {
    "contracts": "docs-contracts",
    "design": "docs-design",
    "plans": "docs-plans",
    "reports": "docs-reports",
    "infra": "docs-infra",
    "demo": "docs-demo",
    "superseded": "docs-superseded",
    "archive": "docs-archive",
    "agent_runs": "docs-agent-runs",
}

DOC_TYPES = ["治理规范", "阶段计划", "设计", "ADR", "合同", "报告", "索引",
             "运营记录", "快照", "产品文档", "外部说明", "其它"]

# 状态块解析
# 1) 显式字段: "> 状态：..." / "> **状态**：..." / "Status: ..." (行首)
STATUS_FIELD_RE = re.compile(
    r"^\s*(?:>+\s*)?(?:[-*+]\s+)?\|?\s*(?:\*\*)?(?:状态|狀態|Status|status)(?:\*\*)?\s*(?:[:：]|\|)\s*(.*)$"
)
# 2) 同块内联字段: "> 日期：... · 状态：..." / "> 建议状态：..." (行内 search, 需分隔符或"建议"前缀)
STATUS_INLINE_RE = re.compile(
    r"(?:^|[。·|｜；;、]\s*|建议)(?:\*\*)?(?:状态|Status|status)(?:\*\*)?\s*[:：]\s*(.+)$"
)
# 3) 表格形态: "| 状态 | 值 |"
STATUS_TABLE_RE = re.compile(
    r"^\s*\|\s*(?:\*\*)?(?:状态|Status|status)(?:\*\*)?\s*\|\s*([^|]*?)\s*\|"
)
# 4) 英文状态横幅行首标记 (常用于取代/冻结横幅)
BANNER_TOKENS = ("Superseded", "Executed-Frozen", "Historical", "Active-Proposed",
                 "Frozen-Preregistered", "Evidence", "SUPERSEDED", "HISTORICAL",
                 "EXECUTED-FROZEN", "ACTIVE-PROPOSED", "FROZEN-PREREGISTERED")
BANNER_RE = re.compile(
    r"^(?:>+\s*)?(?:\*\*)?(Superseded|Executed-Frozen|Historical|Active-Proposed|"
    r"Frozen-Preregistered|Evidence|SUPERSEDED|HISTORICAL|EXECUTED-FROZEN|"
    r"ACTIVE-PROPOSED|FROZEN-PREREGISTERED)\b\s*(.*)$"
)
# 5) 中文状态词行首 (参考/历史/权威), 如 "> **历史资料。**"
CN_FALLBACK_RE = re.compile(r"^(?:>+\s*)?(?:\*\*)?(参考|历史|权威|现行)(.*)$")
VERSION_LABEL_RE = re.compile(
    r"(?:当前版本|内容版本|版本号|版本)\s*[:：]?\s*[^A-Za-z0-9\n]{0,14}?(v\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
VERSION_IN_TITLE_RE = re.compile(r"(?<![A-Za-z0-9])v(\d+(?:\.\d+)+)(?![A-Za-z0-9])")
DATE_LABEL_RE = re.compile(
    r"(?:日期|生成|签发|冻结日期|date|Date)\s*[:：]\s*(\d{4})[-/年]?(\d{1,2})[-/月]?(\d{1,2})"
)
DATE_AT_RE = re.compile(r"@\s*(\d{4})-(\d{2})-(\d{2})")
DATE_FILENAME_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
HTML_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")

CSV_FIELDS = ["path", "ext", "size_bytes", "lines", "sha256_16", "mtime",
              "git_first_add_date", "git_last_commit_date", "title", "zone",
              "doc_type", "declared_status", "declared_version", "declared_date",
              "ref_count", "on_disk_flag"]

STATUS_CATEGORY_ORDER = [
    "权威/SSOT", "Accepted/已接受", "Superseded/已取代", "Historical/历史快照",
    "Executed-Frozen", "Frozen-Preregistered", "Proposed/Active-Proposed",
    "冻结/Frozen", "已实现", "执行中/待实施", "设计稿", "其它", "(空)",
]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """从脚本位置向上找 .git，找不到则退回当前工作目录。"""
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    cwd = Path.cwd().resolve()
    if (cwd / ".git").exists():
        return cwd
    raise SystemExit("ERROR: cannot locate repository root (.git); pass --root")


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def count_lines(text: str) -> int:
    if not text:
        return 0
    n = text.count("\n")
    if not text.endswith("\n"):
        n += 1
    return n


def sha256_16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def mtime_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def diff_days(a_iso_date: str, mtime_iso_str: str) -> int | None:
    try:
        d1 = date.fromisoformat(a_iso_date)
        d2 = datetime.fromisoformat(mtime_iso_str).date()
        return abs((d2 - d1).days)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------

def walk(base: Path):
    out = []
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d not in SKIP_DIRNAMES]
        for fn in fns:
            out.append(Path(dp) / fn)
    return out


def discover_documents(root: Path) -> list[Path]:
    """按白名单返回文档文件绝对路径列表（去重、排序）。"""
    found: list[Path] = []

    # 1) 仓库根: 顶层 *.md / *.txt + LICENSE
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in ROOT_TXT_EXTS or p.name in ROOT_EXTRA_NAMES:
            found.append(p)

    # 2) docs / examples / artifacts / scripts 递归取 .md + .html
    for d in ROOT_DIRNAMES:
        base = root / d
        if not base.is_dir():
            continue
        for p in walk(base):
            if p.suffix.lower() in DOC_EXTS_RECURSE:
                found.append(p)

    # 3) data / outputs 只取顶层说明文件 (*.md)
    for d in TOPLEVEL_DIRNAMES:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            if p.is_file() and p.suffix.lower() in DOC_EXTS_TOPDIR:
                found.append(p)

    # 4) config: 只取说明类文档
    base = root / CONFIG_DIRNAME
    if base.is_dir():
        for p in walk(base):
            if p.suffix.lower() in {".md", ".txt", ".rst"}:
                found.append(p)

    # 去重 + 排序
    uniq = sorted({p.resolve() for p in found}, key=lambda x: rel_posix(x, root))
    return uniq


def excluded_docs_report(root: Path) -> dict:
    """统计 docs/ 下"非文档类"被排除的文件，写进口径说明。"""
    docs = root / "docs"
    if not docs.is_dir():
        return {}
    by_ext: dict[str, int] = {}
    examples: list[str] = []
    for p in walk(docs):
        ext = p.suffix.lower()
        if ext in DOC_EXTS_RECURSE:
            continue
        key = ext or "<none>"
        by_ext[key] = by_ext.get(key, 0) + 1
        if len(examples) < 12:
            examples.append(rel_posix(p, root))
    return {"by_ext": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])),
            "count": sum(by_ext.values()), "examples": examples}


# ---------------------------------------------------------------------------
# git 元数据
# ---------------------------------------------------------------------------

def git_tracked_set(root: Path) -> set[str]:
    try:
        r = subprocess.run(
            ["git", "-c", "core.quotepath=false", "ls-files", "-z"],
            cwd=str(root), capture_output=True, encoding="utf-8", errors="replace",
            timeout=180,
        )
        if r.returncode != 0:
            return set()
        return {s for s in r.stdout.split("\x00") if s}
    except Exception:
        return set()


def git_dates_for(root: Path, rel: str) -> tuple[str, str]:
    """返回 (first_add_date, last_commit_date)，YYYY-MM-DD；失败返回 ('','')。"""
    cmd = ["git", "-c", "core.quotepath=false", "log", "--follow",
           "--format=%ad", "--date=short", "--", rel]
    try:
        r = subprocess.run(cmd, cwd=str(root), capture_output=True,
                           encoding="utf-8", errors="replace", timeout=120)
        if r.returncode != 0:
            return "", ""
        dates = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if not dates:
            return "", ""
        return dates[-1], dates[0]        # log 新→旧
    except Exception:
        return "", ""


# ---------------------------------------------------------------------------
# 内容解析
# ---------------------------------------------------------------------------

def clean_inline(s: str, limit: int = 140) -> str:
    s = s.replace("**", "").replace("`", "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" >|*")
    return s[:limit].strip()


MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
TRAIL_LABEL_RE = re.compile(r"\s*[·•|]\s*(日期|编制|评审人|生成|层级|方向|作者)\s*$")


def clean_status(s: str, limit: int = 120) -> str:
    """状态值清洗: 保留 markdown 链接文字、去粗体/反引号、去尾部 ' · 日期' 碎片、≤limit。"""
    s = MD_LINK_RE.sub(r"\1", s)
    s = s.replace("**", "").replace("`", "")
    s = re.sub(r"\s+", " ", s).strip(" >|*")
    s = TRAIL_LABEL_RE.sub("", s).strip()
    if len(s) > limit:
        s = s[:limit] + "…"
    return s


def extract_title(path: Path, text: str, head_lines: list[str]) -> str:
    ext = path.suffix.lower()
    if ext == ".html":
        m = HTML_TITLE_RE.search(text) or HTML_H1_RE.search(text)
        if m:
            t = clean_inline(TAG_RE.sub(" ", m.group(1)), 160)
            if t:
                return t
        return ""
    if ext in (".md",):
        for ln in head_lines[:12]:
            m = re.match(r"^#\s+(.+?)\s*$", ln)
            if m:
                return clean_inline(m.group(1), 160)
        for ln in head_lines[:5]:
            if ln.strip():
                return clean_inline(ln, 160)
        return ""
    # .txt / LICENSE
    for ln in head_lines[:5]:
        if ln.strip():
            return clean_inline(ln, 160)
    return ""


def cut_top_level(s: str, limit: int = 120) -> str:
    """在括号深度 0 处按 。；：截断, 避免把括号里的分句截一半。"""
    depth = 0
    for i, ch in enumerate(s):
        if ch in "（(":
            depth += 1
        elif ch in "）)":
            depth = max(0, depth - 1)
        elif depth == 0 and ch in "。；：":
            return s[:i]
    return s


def strip_block_prefix(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^>+\s*", "", s)
    s = re.sub(r"^(?:[-*+]\s+)", "", s)
    return s


def _cn_join(token: str, rest: str) -> str:
    if not rest:
        return token
    if token in ("参考", "历史", "权威", "现行"):
        return token + rest
    return token + " " + rest


def extract_declared_status(head_lines: list[str]) -> str:
    """优先级: ① 英文状态横幅(行首) → ② 显式 状态:/Status: 字段 →
    ③ 中文状态词行首(参考/历史/权威) → ④ 表格行。取第一句, ≤120 字符。"""
    cleaned = [strip_block_prefix(ln) for ln in head_lines]

    # ① 横幅 (Superseded / Executed-Frozen / Historical / ...)
    for ln in cleaned:
        m = BANNER_RE.match(ln)
        if m:
            token = m.group(1)
            rest = m.group(2).lstrip("*").strip()
            value = cut_top_level(_cn_join(token, rest))
            out = clean_status(value)
            if out:
                return out

    # ② 显式字段（行首）
    for ln in cleaned:
        m = STATUS_FIELD_RE.match(ln)
        if m:
            val = m.group(1).strip()
            if not val or val in ("|", ":", "：", "**"):
                continue
            out = clean_status(cut_top_level(val))
            if out:
                return out
    # ②b 行内字段（"> 日期：... · 状态：xxx" / "> 建议状态：xxx"）
    for ln in cleaned:
        if ln.startswith("状"):
            continue  # 已由 ② 处理
        m = STATUS_INLINE_RE.search(ln)
        if m:
            val = m.group(1).strip()
            if not val or val in ("|", ":", "：", "**"):
                continue
            out = clean_status(cut_top_level(val))
            if out:
                return out

    # ③ 中文状态词行首
    for ln in cleaned:
        m = CN_FALLBACK_RE.match(ln)
        if m:
            token, rest = m.group(1), m.group(2).lstrip("*").strip()
            value = cut_top_level(_cn_join(token, rest))
            out = clean_status(value)
            if out and len(out) >= 2:
                return out

    # ④ 表格行 | 状态 | 值 |
    for ln in cleaned:
        m = STATUS_TABLE_RE.match(ln)
        if m:
            val = clean_status(m.group(1))
            if val:
                return val
    return ""



def extract_declared_version(head_lines: list[str], title: str) -> str:
    for ln in head_lines:
        m = VERSION_LABEL_RE.search(ln)
        if m:
            return m.group(1)
    m = VERSION_IN_TITLE_RE.search(title or "")
    if m:
        return "v" + m.group(1)
    return ""


def extract_declared_date(head_lines: list[str]) -> str:
    for ln in head_lines:
        m = DATE_LABEL_RE.search(ln)
        if m:
            return _ymd(m.groups())
    for ln in head_lines:
        m = DATE_AT_RE.search(ln)
        if m:
            return _ymd(m.groups())
    return ""


def _ymd(groups) -> str:
    y, mo, d = groups
    try:
        return date(int(y), int(mo), int(d)).strftime("%Y%m%d")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 归类
# ---------------------------------------------------------------------------

def zone_of(rel: str) -> str:
    parts = rel.split("/")
    top = parts[0]
    if len(parts) == 1:
        return "root"
    if top == "docs":
        if len(parts) == 2:
            return "docs-root"
        sub = parts[1]
        if sub == "design" and len(parts) >= 4 and parts[2] == "decisions":
            return "docs-design-decisions"
        return ZONE_BY_DOCS_SUBDIR.get(sub, "docs-" + sub.replace("_", "-"))
    if top in ("examples", "artifacts", "scripts"):
        return top
    if top in ("data", "outputs"):
        return "data-outputs"
    if top == CONFIG_DIRNAME:
        return "config"
    return top


def doc_type_of(rel: str, name: str, zone: str) -> str:
    low = name.lower()
    if zone == "docs-archive":
        return "快照"
    if re.match(r"^ADR-\d+", name):
        return "ADR"
    if rel in ("docs/PES2TS_开发宪法_v1.md", "AGENT.md"):
        return "治理规范"
    # 目录 README / 索引类
    if name == "README.md" and zone not in ("root", "docs-archive", "examples", "artifacts"):
        return "索引"
    if zone == "docs-contracts":
        return "合同"
    if zone in ("docs-design", "docs-design-decisions"):
        return "设计"
    if zone == "docs-reports":
        return "报告"
    if zone == "docs-plans":
        if "台账" in name:
            return "运营记录"
        if "报告" in name or "复盘" in name:
            return "报告"
        return "阶段计划"
    if zone == "docs-infra":
        return "运营记录"
    if zone == "docs-demo":
        if "审核" in name or "解读" in name:
            return "报告"
        return "设计"
    if zone == "docs-superseded":
        if "产品设计" in name:
            return "设计"
        return "阶段计划"
    if zone == "docs-agent-runs":
        if "REPORT" in name.upper() or "REVIEW" in name.upper():
            return "报告"
        if "PLAN" in name.upper():
            return "阶段计划"
        return "运营记录"
    if zone == "docs-root":
        return "索引" if "README" in name.upper() else "治理规范"
    if zone == "root":
        if name == "README.md":
            return "产品文档"
        if name == "AGENT.md":
            return "治理规范"
        if name in ("CONTRIBUTING.md", "LICENSE"):
            return "外部说明"
        if low.startswith("requirements"):
            return "其它"
        if "REPORT" in name.upper() or "RESULT_SUMMARY" in name.upper():
            return "报告"
        if any(k in name.upper() for k in ("MEMO", "REMAINING", "TIMELINE", "RESUME")):
            return "运营记录"
        return "报告"
    if zone == "examples":
        return "产品文档"
    if zone == "artifacts":
        return "产品文档"
    if zone in ("scripts", "data-outputs"):
        return "索引"
    if zone == "config":
        return "外部说明"
    return "其它"


def status_category(val: str) -> str:
    if not val:
        return "(空)"
    s = val.strip()
    low = s.lower()
    # 强 token 优先(横幅式), 避免 "已取代" 等词误判
    if "executed-frozen" in low:
        return "Executed-Frozen"
    if "superseded" in low or "已被取代" in s or ("已由" in s and "取代" in s):
        return "Superseded/已取代"
    if "historical" in low or "历史" in s or "只读" in s:
        return "Historical/历史快照"
    if "frozen-preregistered" in low:
        return "Frozen-Preregistered"
    if "active-proposed" in low or "proposed" in low or "建议状态" in s:
        return "Proposed/Active-Proposed"
    if "权威" in s or "ssot" in low:
        return "权威/SSOT"
    if "accepted" in low or "已接受" in s:
        return "Accepted/已接受"
    if "冻结" in s:
        return "冻结/Frozen"
    if "已实现" in s or "implemented" in low:
        return "已实现"
    if any(k in s for k in ("执行方案", "待实施", "待冻结", "待创建", "计划")):
        return "执行中/待实施"
    if "设计稿" in s or "draft" in low:
        return "设计稿"
    return "其它"


# ---------------------------------------------------------------------------
# ref_count
# ---------------------------------------------------------------------------

def build_ref_corpus(root: Path) -> list[tuple[str, str]]:
    """返回 [(relpath, text)]，范围: 全库 .md/.py/.json/.yaml/.yml，
    排除 data/ outputs/ .git/ .slim/ 缓存目录与 docs/archive/ 快照。"""
    corpus = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in REF_EXCLUDE_DIRS]
        for fn in fns:
            p = Path(dp) / fn
            if p.suffix.lower() not in REF_CORPUS_EXTS:
                continue
            try:
                rel = rel_posix(p, root)
            except ValueError:
                continue
            if any(rel.startswith(pref) for pref in REF_EXCLUDE_PREFIXES):
                continue
            try:
                corpus.append((rel, decode_text(read_bytes(p))))
            except Exception:
                continue
    return corpus


def count_refs(basenames: set[str], corpus: list[tuple[str, str]]) -> dict[str, int]:
    counts = {b: 0 for b in basenames}
    for _rel, text in corpus:
        for b in basenames:
            if b in text:
                counts[b] += text.count(b)
    return counts


# ---------------------------------------------------------------------------
# 采集主流程
# ---------------------------------------------------------------------------

def collect(root: Path, debug: bool = False) -> dict:
    docs = discover_documents(root)
    tracked = git_tracked_set(root)

    # 一次读完文件内容
    raw: dict[str, dict] = {}
    for p in docs:
        rel = rel_posix(p, root)
        data = read_bytes(p)
        text = decode_text(data)
        head_lines = [ln.rstrip("\r\n") for ln in text.splitlines()[:20]]
        st = p.stat()
        raw[rel] = {
            "path": rel,
            "ext": p.suffix.lower(),
            "size_bytes": st.st_size,
            "lines": count_lines(text),
            "sha256_16": sha256_16(data),
            "mtime": mtime_iso(st.st_mtime),
            "title": extract_title(p, text, head_lines),
            "zone": zone_of(rel),
            "doc_type": "",  # 下面补
            "declared_status": extract_declared_status(head_lines),
            "declared_version": "",
            "declared_date": extract_declared_date(head_lines),
            "ref_count": 0,
            "on_disk_flag": 1 if rel in tracked else 0,
        }
        raw[rel]["doc_type"] = doc_type_of(rel, p.name, raw[rel]["zone"])
        raw[rel]["declared_version"] = extract_declared_version(
            head_lines, raw[rel]["title"])

    # git 日期（并发）
    rels = list(raw.keys())
    def _git(rel: str):
        return rel, git_dates_for(root, rel)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rel, (first, last) in ex.map(_git, rels):
            raw[rel]["git_first_add_date"] = first
            raw[rel]["git_last_commit_date"] = last

    # ref_count
    corpus = build_ref_corpus(root)
    basenames = {os.path.basename(r) for r in rels}
    refcounts = count_refs(basenames, corpus)
    for rel in rels:
        raw[rel]["ref_count"] = refcounts.get(os.path.basename(rel), 0)

    records = [raw[r] for r in sorted(raw.keys())]

    if debug:
        lines = []
        for r in records:
            lines.append("\t".join([
                r["path"], r["zone"], r["doc_type"],
                r["declared_status"] or "<EMPTY>",
                r["declared_version"] or "-", r["declared_date"] or "-",
                str(r["ref_count"]), r["title"][:70],
            ]))
        return {"records": records, "_debug": "\n".join(lines),
                "_corpus": {"files": len(corpus),
                            "chars": sum(len(t) for _, t in corpus),
                            "excluded": excluded_docs_report(root)}}

    return {"records": records,
            "_corpus": {"files": len(corpus),
                        "chars": sum(len(t) for _, t in corpus),
                        "excluded": excluded_docs_report(root)}}


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

def build_stats(records: list[dict], corpus_meta: dict, root: Path) -> dict:
    def count_by(key, transform=None):
        d: dict[str, int] = {}
        for r in records:
            v = r[key]
            if transform:
                v = transform(v)
            d[v] = d.get(v, 0) + 1
        return dict(sorted(d.items(), key=lambda kv: (-kv[1], str(kv[0]))))

    total_size = sum(r["size_bytes"] for r in records)
    total_lines = sum(r["lines"] for r in records)

    no_status = [r["path"] for r in records if not r["declared_status"]]
    untracked = [r["path"] for r in records if not r["on_disk_flag"]]

    stale = []
    for r in records:
        d = diff_days(r["git_last_commit_date"], r["mtime"]) if r["git_last_commit_date"] else None
        if d is not None and d > 7:
            stale.append({"path": r["path"], "mtime": r["mtime"],
                          "git_last_commit_date": r["git_last_commit_date"],
                          "diff_days": d})
    stale.sort(key=lambda x: -x["diff_days"])

    largest = sorted(records, key=lambda r: -r["size_bytes"])[:20]
    largest = [{"path": r["path"], "size_bytes": r["size_bytes"],
                "lines": r["lines"], "zone": r["zone"]} for r in largest]

    dated_no_version = []
    for r in records:
        name = os.path.basename(r["path"])
        if DATE_FILENAME_RE.search(name) and not r["declared_version"]:
            dated_no_version.append(r["path"])

    with_status = len(records) - len(no_status)
    cats = count_by("declared_status", status_category)
    cats = {k: cats[k] for k in STATUS_CATEGORY_ORDER if k in cats} | {
        k: v for k, v in cats.items() if k not in STATUS_CATEGORY_ORDER}

    top_refs = sorted(records, key=lambda r: -r["ref_count"])[:20]
    top_refs = [{"path": r["path"], "ref_count": r["ref_count"]} for r in top_refs]

    return {
        "total_records": len(records),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / 1024 / 1024, 3),
        "total_lines": total_lines,
        "with_status_block": with_status,
        "no_status_block_count": len(no_status),
        "no_status_block_paths": no_status,
        "untracked_count": len(untracked),
        "untracked_paths": untracked,
        "tracked_count": len(records) - len(untracked),
        "by_zone": count_by("zone"),
        "by_doc_type": count_by("doc_type"),
        "by_ext": count_by("ext"),
        "by_declared_status_category": cats,
        "by_declared_status_raw": count_by("declared_status"),
        "by_declared_version": count_by("declared_version"),
        "largest_20": largest,
        "top_ref_count_20": top_refs,
        "mtime_vs_git_last_commit_gt_7d": stale,
        "mtime_vs_git_last_commit_gt_7d_count": len(stale),
        "filename_has_date_but_no_version": dated_no_version,
        "filename_has_date_but_no_version_count": len(dated_no_version),
        "records_without_git_dates": [r["path"] for r in records
                                      if not r["git_last_commit_date"]],
        "git_first_add_min": min((r["git_first_add_date"] for r in records
                                  if r["git_first_add_date"]), default=""),
        "git_first_add_max": max((r["git_first_add_date"] for r in records
                                  if r["git_first_add_date"]), default=""),
        "git_last_commit_max": max((r["git_last_commit_date"] for r in records
                                    if r["git_last_commit_date"]), default=""),
        "ref_corpus": {"files": corpus_meta.get("files"),
                       "chars": corpus_meta.get("chars")},
        "excluded_from_scan": corpus_meta.get("excluded", {}),
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_csv(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\r\n")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def write_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "0.0%"


def write_stats_md(path: Path, records: list[dict], stats: dict, root: Path,
                   generated_at: str):
    L: list[str] = []
    a = L.append

    a("# PES2TS 开发文档盘点统计报告 (inventory_stats.md)")
    a("")
    a(f"- 采集时点 (快照): {generated_at} (本清单为**该时点**仓库状态; 此后新增/修改的文档不在本清单内)")
    a(f"- 仓库根: `{root.as_posix()}`")
    a(f"- 采集脚本: `collect_inventory.py` (同目录, 可复跑: `python collect_inventory.py`)")
    a(f"- 机器可读清单: `doc_inventory.csv` / `doc_inventory.json`")
    a(f"- **记录总数: {stats['total_records']}** 条; 总体积 **{stats['total_size_bytes']:,} B** "
      f"({stats['total_size_mb']} MB); 总行数 {stats['total_lines']:,} 行")
    a("")

    # ---- 口径 ----
    a("## 0. 采集口径 (CALIBER)")
    a("")
    a("**扫描范围 (白名单)**:")
    a("")
    a("| 位置 | 规则 | 命中 | 说明 |")
    a("|---|---|---:|---|")
    zones = stats["by_zone"]
    a(f"| 仓库根 | `*.md` + `*.txt` + `LICENSE` | {zones.get('root', 0)} | 项目入口/宪法投影/战役 memo/依赖清单/许可证 |")
    a("| `docs/` | 递归 `*.md` + `*.html` | %d | 不含 data/state/fixture 类文件(见下) |" %
      sum(v for k, v in zones.items() if k.startswith("docs-")))
    a(f"| `examples/` | 递归 `*.md` + `*.html` | {zones.get('examples', 0)} | 示例说明 + 可视化 html |")
    a(f"| `artifacts/` | 递归 `*.md` | {zones.get('artifacts', 0)} | 仅 8 份 `model_card.md` |")
    a(f"| `scripts/` | 递归 `*.md` | {zones.get('scripts', 0)} | 脚本目录 README |")
    a(f"| `data/` + `outputs/` | 仅顶层 `*.md` | {zones.get('data-outputs', 0)} | 只取两个 README,不递归(12GB/1.3GB 大数据目录) |")
    a(f"| `config/` | 递归 `*.md`/`*.txt`/`*.rst` | {zones.get('config', 0)} | 实测 config 下无说明文档,只有 yaml/ini |")
    a("")
    a("**排除项**: `.git/`、`.slim/`(工作区)、`data/` 数据文件、`outputs/` 大文件、"
      "`__pycache__`/`.pytest_cache`/`.ruff_cache`/`node_modules`/`.venv` 缓存目录、"
      "以及 `docs/` 下非文档类文件。")
    a("")
    ex = stats.get("excluded_from_scan") or {}
    if ex.get("count"):
        a(f"`docs/` 下按口径排除的非文档文件共 **{ex['count']}** 个: " +
          "、".join(f"`{k}` {v}" for k, v in ex["by_ext"].items()) +
          "(状态 JSON、归档快照 `.snapshot.txt` 字节副本、`.lock` 等;属数据/状态/字节快照,不是文档)。")
        a("")
    by_ext = stats["by_ext"]
    n_md = by_ext.get(".md", 0)
    n_html = by_ext.get(".html", 0)
    n_txt = by_ext.get(".txt", 0)
    n_lic = by_ext.get("", 0)
    a(f"**计划分母对照 (任务预期 ~245 条)**: 实际 **{stats['total_records']}** 条 = `.md` {n_md} + "
      f"`.html` {n_html} + `.txt` {n_txt} + `LICENSE` {n_lic}。静态预期 245 的构成: docs 216 `.md` + "
      "4 `.html`(3 docs + 1 examples) + 根 11 `.md` + 3 `requirements*.txt` + `LICENSE` + "
      "8 artifacts `model_card.md` + 2 examples `.md` + 2 scripts `.md` + 2 data/outputs README。"
      "差额来自: ① html/txt/LICENSE 是否计为文档的口径(本报告计入, 见上表); "
      "② 盘点期间团队仍在写入新文档(`docs/infra/**`, 见 §6.1 未跟踪清单), 清单以采集时点为准。")
    a("")
    a("**doc_type 判定**: 按路径与文件名规则判定(不用目录名硬猜内容),规则链依次为: "
      "`docs/archive/**`→快照; `ADR-*`→ADR; 宪法/AGENT→治理规范; "
      "`docs/contracts/**`→合同; `docs/design/**`→设计; `docs/plans/**`→阶段计划(含“台账”转为运营记录、"
      "含“报告/复盘”转为报告); `docs/reports/**`→报告; `docs/infra/**`→运营记录; "
      "`docs/demo/**`→设计(含“审核/解读”转为报告); `docs/superseded/**`→阶段计划/设计; "
      "`docs/agent_runs/**`→运营记录(含 REPORT/REVIEW 转报告、PLAN 转阶段计划); "
      "根 README→产品文档、CONTRIBUTING/LICENSE→外部说明、requirements→其它、其余 memo/报告按名判定; "
      "examples/artifacts→产品文档; README 类索引→索引; 其余→其它。")
    a("")
    a("**declared_status**: 按优先级解析文档顶部 20 行: ① 行首英文状态横幅 "
      "(`> **SUPERSEDED（...）**`/`Executed-Frozen`/`Historical`/`Active-Proposed`/"
      "`Frozen-Preregistered`/`Evidence`); ② 显式 `状态:`/`Status:` 字段(含 `> **状态**：` "
      "与 `日期：… · 状态：…` 行内形态); ③ 行首中文状态词(`参考`/`历史`/`权威`/`现行`); "
      "④ 表格行 `| 状态 | … |`。取到后在括号深度 0 处按 `。`/`：` 截断、去 markdown 标记, "
      "≤120 字符。取不到一律留空,**不做推断**。")
    a("")
    a("**declared_version**: 顶部 20 行内 `当前版本:`/`版本:` 等标签后的 `vX.Y[.Z]`; "
      "无标签时回退取标题中的 `vX.Y`。**declared_date**: 顶部 20 行内 `日期:`/`生成:` 等标签或 "
      "`@ YYYY-MM-DD` 形态,统一归一为 `YYYYMMDD`。")
    a("")
    a("**ref_count**: 以文档基本名为单位,在全库 `.md/.py/.json/.yaml/.yml` 文本中出现的次数; "
      f"语料 {stats['ref_corpus'].get('files')} 个文件 / "
      f"{stats['ref_corpus'].get('chars', 0):,} 字符; 排除 `.git/`、`.slim/`、`data/`、`outputs/`、"
      "缓存目录与 `docs/archive/` 归档快照。**注意**: 同名文件(如多个 `README.md`、8 份 "
      "`model_card.md`)共享同一基本名计数,故该字段对同名通用文件名只表示“该名字被引用次数”,"
      "不区分具体实例; 计数含文档自身出现。")
    a("")
    a("**on_disk_flag**: `1` = 已被 git 跟踪(`git ls-files`); `0` = 磁盘存在但未跟踪/被忽略。"
      "时间戳: `mtime` 为本地时区 `YYYY-MM-DDTHH:MM:SS`; `git_*_date` 为 git 提交日期 `YYYY-MM-DD`"
      "(`git log --follow`,首次=最早提交,末次=最新提交)。")
    a("")

    # ---- 1 zone ----
    a("## 1. 按 zone 计数")
    a("")
    a("| zone | 数量 | 占比 |")
    a("|---|---:|---:|")
    for k, v in stats["by_zone"].items():
        a(f"| `{k}` | {v} | {pct(v, stats['total_records'])} |")
    a(f"| **合计** | **{stats['total_records']}** | 100% |")
    a("")

    # ---- 2 doc_type ----
    a("## 2. doc_type 分布")
    a("")
    a("| doc_type | 数量 | 占比 |")
    a("|---|---:|---:|")
    for k, v in stats["by_doc_type"].items():
        a(f"| {k} | {v} | {pct(v, stats['total_records'])} |")
    a("")
    a("交叉表 (zone × doc_type, 只列非零):")
    a("")
    zones_sorted = list(stats["by_zone"].keys())
    a("| zone | " + " | ".join(dt for dt in DOC_TYPES) + " | 合计 |")
    a("|---|" + "---:|" * len(DOC_TYPES) + "---:|")
    cross: dict[str, dict[str, int]] = {}
    for r in records:
        cross.setdefault(r["zone"], {}).setdefault(r["doc_type"], 0)
        cross[r["zone"]][r["doc_type"]] += 1
    for z in zones_sorted:
        row = [str(cross.get(z, {}).get(dt, 0) or "") for dt in DOC_TYPES]
        a(f"| `{z}` | " + " | ".join(row) + f" | {stats['by_zone'][z]} |")
    a("")

    # ---- 3 status ----
    a("## 3. 按 declared_status 计数")
    a("")
    a(f"- 有状态块声明: **{stats['with_status_block']}** / {stats['total_records']} "
      f"({pct(stats['with_status_block'], stats['total_records'])})")
    a(f"- **无状态块: {stats['no_status_block_count']}** 条 (详见 §7)")
    a("")
    a("### 3.1 归一化状态类别")
    a("")
    a("| 状态类别 | 数量 |")
    a("|---|---:|")
    for k, v in stats["by_declared_status_category"].items():
        a(f"| {k} | {v} |")
    a("")
    a("### 3.2 原始声明值 (Top 30)")
    a("")
    a("| 原始 declared_status | 数量 |")
    a("|---|---:|")
    for i, (k, v) in enumerate(stats["by_declared_status_raw"].items()):
        if i >= 30:
            break
        k2 = k if k else "(空)"
        k2 = k2.replace("|", "\\|")
        if len(k2) > 110:
            k2 = k2[:110] + "…"
        a(f"| {k2} | {v} |")
    a("")
    if len(stats["by_declared_status_raw"]) > 30:
        a(f"(另 {len(stats['by_declared_status_raw']) - 30} 个长尾状态值, 见 doc_inventory.json "
          "`stats.by_declared_status_raw`)")
        a("")

    # ---- 4 size ----
    a("## 4. 文档总量与最大 20 个文档")
    a("")
    a(f"- 总体积: **{stats['total_size_bytes']:,} B** = {stats['total_size_mb']} MB")
    a(f"- 总行数: {stats['total_lines']:,} 行")
    a(f"- 平均: {stats['total_size_bytes'] / max(stats['total_records'], 1):,.0f} B/篇, "
      f"{stats['total_lines'] / max(stats['total_records'], 1):,.0f} 行/篇")
    a("")
    a("| # | path | size (B) | 行数 | zone |")
    a("|---:|---|---:|---:|---|")
    for i, r in enumerate(stats["largest_20"], 1):
        a(f"| {i} | `{r['path']}` | {r['size_bytes']:,} | {r['lines']:,} | `{r['zone']}` |")
    a("")

    # ---- 5 top ref ----
    a("## 5. ref_count 最高的 20 个文档")
    a("")
    a("| # | path | ref_count |")
    a("|---:|---|---:|")
    for i, r in enumerate(stats["top_ref_count_20"], 1):
        a(f"| {i} | `{r['path']}` | {r['ref_count']} |")
    a("")

    # ---- 6 git ----
    a("## 6. git 跟踪与时间戳")
    a("")
    a(f"- 已被 git 跟踪: **{stats['tracked_count']}** 条")
    a(f"- **未被 git 跟踪: {stats['untracked_count']}** 条")
    a(f"- git 首次提交日期范围: `{stats['git_first_add_min']}` → `{stats['git_first_add_max']}`")
    a(f"- git 最新提交日期(全清单最大值): `{stats['git_last_commit_max']}`")
    a(f"- 取不到 git 日期的记录: {len(stats['records_without_git_dates'])} 条")
    a("")
    a("### 6.1 未被 git 跟踪的文档")
    a("")
    if stats["untracked_paths"]:
        for p in stats["untracked_paths"]:
            a(f"- `{p}`")
    else:
        a("无 (本次盘点范围内全部文档均已被 git 跟踪; 工作树干净, 无未跟踪/被忽略的文档)。")
    a("")
    if stats["records_without_git_dates"]:
        a("### 6.2 取不到 git 日期的文档")
        a("")
        for p in stats["records_without_git_dates"]:
            a(f"- `{p}`")
        a("")

    # ---- 7 no status ----
    a("## 7. 无状态块文档清单")
    a("")
    a(f"共 **{stats['no_status_block_count']}** 条 (定义: 顶部 20 行内既无 `状态:`/`Status:` 字段, "
      "也无 Superseded/Historical/Executed-Frozen/Active-Proposed/Frozen-Preregistered/Evidence/"
      "参考/历史/权威 行)。")
    a("")
    if stats["no_status_block_paths"]:
        by_zone: dict[str, list[str]] = {}
        for p in stats["no_status_block_paths"]:
            by_zone.setdefault(p.split("/")[0] if "/" in p else "(root)", []).append(p)
        for z in sorted(by_zone):
            a(f"### {z}/ ({len(by_zone[z])} 条)")
            a("")
            for p in by_zone[z]:
                a(f"- `{p}`")
            a("")
    else:
        a("无")
        a("")

    # ---- 8 stale ----
    a("## 8. mtime 与 git_last_commit 相差 > 7 天的文档")
    a("")
    a(f"共 **{stats['mtime_vs_git_last_commit_gt_7d_count']}** 条。含义: 最后一次提交之后工作区仍被本地"
      "修改过(或 checkout 时间不同), 内容可能与 git 记录不一致, 需人工确认。")
    a("")
    if stats["mtime_vs_git_last_commit_gt_7d"]:
        a("| # | path | mtime | 最后提交 | 差(天) |")
        a("|---:|---|---|---|---:|")
        for i, r in enumerate(stats["mtime_vs_git_last_commit_gt_7d"], 1):
            a(f"| {i} | `{r['path']}` | {r['mtime']} | {r['git_last_commit_date']} | {r['diff_days']} |")
    else:
        a("无")
    a("")

    # ---- 9 dated but no version ----
    a("## 9. 文件名含日期但内容无版本信息的文档")
    a("")
    a(f"共 **{stats['filename_has_date_but_no_version_count']}** 条 (文件名含 `YYYYMMDD`/`YYYY-MM-DD`, "
      "但顶部 20 行与标题均未声明版本号)。")
    a("")
    if stats["filename_has_date_but_no_version"]:
        for p in stats["filename_has_date_but_no_version"]:
            a(f"- `{p}`")
    else:
        a("无")
    a("")

    # ---- 10 declared_version ----
    a("## 10. declared_version 分布 (非空)")
    a("")
    a("| declared_version | 数量 |")
    a("|---|---:|")
    ver = {k: v for k, v in stats["by_declared_version"].items() if k}
    if ver:
        for k, v in list(ver.items())[:40]:
            a(f"| `{k}` | {v} |")
    else:
        a("| (全部为空) | 0 |")
    a(f"| (空) | {stats['by_declared_version'].get('', 0)} |")
    a("")

    # ---- 附: 全量明细 ----
    a("## 11. 附: 全量清单 (按 zone, path 排序)")
    a("")
    a("| # | path | ext | size(B) | 行 | zone | doc_type | declared_status | ver | date | ref | tracked |")
    a("|---:|---|---|---:|---:|---|---|---|---|---|---:|:-:|")
    for i, r in enumerate(sorted(records, key=lambda x: (x["zone"], x["path"])), 1):
        st = (r["declared_status"] or "").replace("|", "\\|")
        if len(st) > 60:
            st = st[:60] + "…"
        a(f"| {i} | `{r['path']}` | {r['ext'] or '-'} | {r['size_bytes']:,} | {r['lines']:,} | "
          f"`{r['zone']}` | {r['doc_type']} | {st} | {r['declared_version'] or '-'} | "
          f"{r['declared_date'] or '-'} | {r['ref_count']} | {r['on_disk_flag']} |")
    a("")
    a("---")
    a(f"*本报告由 `collect_inventory.py` 自动生成, 记录总数 {stats['total_records']}。"
      "数据以 `doc_inventory.csv` / `doc_inventory.json` 为准。*")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="PES2TS 文档盘点采集器")
    ap.add_argument("--root", default=None, help="仓库根 (默认从脚本位置向上找 .git)")
    ap.add_argument("--out", default=None, help="输出目录 (默认脚本所在目录)")
    ap.add_argument("--debug", action="store_true", help="额外输出 _debug_fields.txt")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else find_repo_root(script_dir)
    out_dir = Path(args.out).resolve() if args.out else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[collect] root = {root}")
    print(f"[collect] out  = {out_dir}")

    result = collect(root, debug=args.debug)
    records = result["records"]
    corpus_meta = result["_corpus"]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = build_stats(records, corpus_meta, root)
    stats["generated_at"] = generated_at
    stats["root"] = root.as_posix()
    stats["script"] = "collect_inventory.py"

    payload = {
        "generated_at": generated_at,
        "root": root.as_posix(),
        "caliber": {
            "scan_roots": ["<repo-root>/*.md,*.txt,LICENSE",
                           "docs/**/*.md,*.html", "examples/**/*.md,*.html",
                           "artifacts/**/*.md", "scripts/**/*.md",
                           "data/*.md (top-level only)", "outputs/*.md (top-level only)",
                           "config/**/*.md,*.txt,*.rst"],
            "excluded": ["data/ data files", "outputs/ large files", ".git/", ".slim/",
                         "docs/ non-document files (state json / snapshot txt / lock)"],
            "ref_corpus": "all .md/.py/.json/.yaml/.yml minus .git, .slim, data, outputs, "
                          "caches, docs/archive/",
            "on_disk_flag": "1 = git-tracked (git ls-files), 0 = untracked",
        },
        "stats": stats,
        "records": records,
    }

    csv_path = out_dir / "doc_inventory.csv"
    json_path = out_dir / "doc_inventory.json"
    md_path = out_dir / "inventory_stats.md"

    write_csv(csv_path, records)
    write_json(json_path, payload)
    write_stats_md(md_path, records, stats, root, generated_at)

    if args.debug and "_debug" in result:
        (out_dir / "_debug_fields.txt").write_text(result["_debug"], encoding="utf-8")
        print(f"[collect] debug -> {out_dir / '_debug_fields.txt'}")

    print(f"[collect] records = {len(records)}")
    print(f"[collect] total_size = {stats['total_size_bytes']} bytes")
    print(f"[collect] no_status = {stats['no_status_block_count']}")
    print(f"[collect] untracked = {stats['untracked_count']}")
    print(f"[collect] ref_corpus_files = {corpus_meta.get('files')}")
    print(f"[OK] wrote:\n  {csv_path}\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
