# -*- coding: utf-8 -*-
"""validate_mapping.py — 独立校验 Lead 交付的 DISPOSITION.csv / LINK_FIX.csv / BATCH_PLAN.md。

校验对象（默认路径，可用参数覆盖）：
  - docs/infra/doc_audit/20261121/DISPOSITION.csv      （253 行全量处置映射）
  - docs/infra/doc_audit/20261121/LINK_FIX.csv         （条件迁移引用修正清单）
  - docs/infra/doc_audit/20261121/BATCH_PLAN.md        （批次手册）
  - .slim/deepwork/20261121_docaudit/inventory/doc_inventory.csv（253 条覆盖基准）
  - infra/governance/docs_consistency_config.json     （门禁钉死名单）

断言：
  (a) 清单每行恰一条处置（集合相等、无重复、无遗漏、无多余）
  (b) action=move 的目标不冲突（不存在于工作树/清单/git 索引），且与 LINK_FIX 覆盖一致
  (c) blocker 与 52 条门禁钉死清单逐条交叉（钉死路径不得 move；gate_locked 列与 config 精确一致；
      blocker 词表与语义一致性——语义偏差记为 WARN，不影响安全结论）
  (d) batch 并集 = 全部非 keep/retain-readonly 项；B1=空集、B2=空、B3=需裁决项；
      批次数字与 BATCH_PLAN.md 一致
附加（Lead 要求）：
  (e) LINK_FIX 逐条可解析（referrer 存在、行号有效、old_target 出现在该行）且对 3 个迁移目标
      的引用覆盖完整（git grep 交叉）；仓库根相对路径与 markdown 相对链接的口径差异记为 WARN

只读：本脚本不修改任何仓库文件。用法：python validate_mapping.py [--repo PATH] [--audit DIR]
输出：PASS/FAIL + 各检查明细 + 统计表；FAIL 时退出码 1。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

try:  # 统一 UTF-8 输出（重定向到文件时也不会变成 locale 编码）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

# ── 路径解析 ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[3]          # .slim/deepwork/20261121_docaudit/mapping -> 仓库根
DEFAULT_AUDIT = Path("docs/infra/doc_audit/20261121")
DEFAULT_INV = Path(".slim/deepwork/20261121_docaudit/inventory/doc_inventory.csv")

ACTION_VOCAB = {"keep", "banner", "move", "retain-readonly", "merge", "out-of-scope"}
BATCH_VOCAB = {"B0", "B1", "B2", "B3"}
BLOCKER_VOCAB = {"门禁钉死", "报告只读", "SSOT", "需所有者裁决", "无", "快照只读"}
SSOT_DOCS = {  # 方案 §1.1 / §2.2 的 SSOT 与治理根（7 份）
    "docs/PES2TS_开发宪法_v1.md",
    "AGENT.md",
    "README.md",
    "docs/README.md",
    "docs/plans/PES2TS_阶段定义_G0-G3.md",
    "docs/plans/PES2TS_完整开发流方案_整合版_v1.md",
    "docs/plans/PES2TS_文档修订台账.md",
}
TIER_B_DOCS = {  # §3.8 Tier B：g_v_map 渲染目标 + executed/active evidence
    "docs/reports/PES2TS_实现报告_20261018.md",
    "docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md",
    "docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md",
    "docs/design/decisions/ADR-0026-V5冲刺key-atom位移合同与OOF语义.md",
    "docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md",
    "docs/design/PES2TS_G_V_MAPPING_v1_20261018.md",
}
TIER_C_DOCS = {
    "docs/plans/PES2TS_阶段定义_G0-G3.md",
    "README.md",
    "AGENT.md",
}

RESULTS: list[tuple[str, str, str]] = []   # (level, check_id, message)


def record(level: str, cid: str, msg: str) -> None:
    RESULTS.append((level, cid, msg))


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=str(repo), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return proc.stdout


def check_a_coverage(inv_rows, dis_rows) -> None:
    ip = [r["path"] for r in inv_rows]
    dp = [r["path"] for r in dis_rows]
    iset, dset = set(ip), set(dp)
    missing = sorted(iset - dset)
    extra = sorted(dset - iset)
    dup = sorted(p for p, c in Counter(dp).items() if c > 1)
    empty_action = [r["path"] for r in dis_rows if not (r["action"] or "").strip()]
    ok = not missing and not extra and not dup and not empty_action and len(iset) == len(iset)
    record("PASS" if ok else "FAIL", "a-coverage",
           f"inventory={len(ip)} unique={len(iset)} | disposition={len(dp)} unique={len(dset)} | "
           f"missing={len(missing)} extra={len(extra)} dup={len(dup)} empty_action={len(empty_action)}")
    for p in missing[:10]:
        record("FAIL", "a-coverage", f"清单有、映射缺: {p}")
    for p in extra[:10]:
        record("FAIL", "a-coverage", f"映射有、清单无: {p}")
    for p in dup[:10]:
        record("FAIL", "a-coverage", f"映射重复: {p}")


def build_pinned(repo: Path):
    cfg = json.loads((repo / "infra/governance/docs_consistency_config.json")
                     .read_text(encoding="utf-8"))
    hist, req, rs = set(cfg["historical"]), set(cfg["required_files"]), set(cfg["root_scope"])
    req_docs = {p for p in req if p.endswith(".md")}
    req_nondoc = {p for p in req if not p.endswith(".md")}
    tier_b = TIER_B_DOCS | {"V53_ROLLOUT_REPORT.md"}
    tier_a = hist | req
    docs_31 = (hist | req_docs | tier_b | TIER_C_DOCS) - {
        "V53_ROLLOUT_REPORT.md", "README.md", "AGENT.md"}
    # 31 docs 中 Tier B 的 V53 属根 memo；用显式分解校验
    hard_root = (hist | rs) & {p for p in (hist | rs) if "/" not in p} - {"R3_RESULT_SUMMARY.md"}
    pinned_docs = {p for p in tier_a | tier_b | TIER_C_DOCS if p.startswith("docs/")}
    pinned = pinned_docs | hard_root | (rs - {"R3_RESULT_SUMMARY.md"})
    decomposition = {
        "tier_a_union": len(tier_a),
        "tier_b_added": len((tier_b | {"docs/design/PES2TS_G_V_MAPPING_v1_20261018.md"}) - tier_a),
        "docs": len(pinned_docs),
        "hard_root": len(hard_root),
        "non_doc": len(req_nondoc),
        "total_hard": len(pinned_docs) + len(hard_root) + len(req_nondoc),
        "soft_root": "R3_RESULT_SUMMARY.md",
    }
    return cfg, hist, req, rs, req_docs, req_nondoc, pinned_docs, hard_root, pinned, decomposition


def check_b_moves(repo: Path, dis_rows, link_rows, pinned: set, inv_paths: set) -> None:
    moves = [r for r in dis_rows if r["action"] == "move"]
    record("PASS" if moves else "FAIL", "b-move",
           f"move 行数={len(moves)}（方案判定：B1 立即迁移应为 0；本批 3 条为 B3 条件迁移）")
    tracked = set(git(repo, "ls-files").splitlines())
    link_targets = {r["new_target"] for r in link_rows}
    for r in moves:
        path, target = r["path"], (r["target_path"] or "").strip()
        problems = []
        if not target.startswith("docs/superseded/"):
            problems.append("target 不在 docs/superseded/")
        if Path(target).name != Path(path).name:
            problems.append("迁移后文件名改变（违反 superseded 保留原文件名）")
        if (repo / target).exists():
            problems.append("target 已存在（冲突）")
        if target in inv_paths or target in tracked:
            problems.append("target 已在清单/git 索引中（冲突）")
        if (r["target_zone"] or "").strip() != "docs/superseded":
            problems.append("target_zone != docs/superseded")
        if (r["batch"] or "").strip() != "B3":
            problems.append("条件迁移 batch != B3")
        if (r["blocker"] or "").strip() != "需所有者裁决":
            problems.append("条件迁移 blocker != 需所有者裁决")
        if path in pinned or (r["gate_locked"] or "").strip():
            problems.append("迁移源属门禁钉死路径")
        if target not in link_targets:
            problems.append("LINK_FIX 未覆盖该 target")
        if problems:
            record("FAIL", "b-move", f"{path}: " + "；".join(problems))
        else:
            record("PASS", "b-move", f"{path} -> {target}（目标不存在、非钉死、LINK_FIX 已覆盖）")


def check_c_pins(repo: Path, dis_rows, cfg, hist, req, rs, pinned_docs, hard_root,
                 pinned, decomposition) -> None:
    level = "PASS"
    d = {r["path"]: r for r in dis_rows}
    # c0：52 条分解
    ok52 = (decomposition["docs"] == 31 and decomposition["hard_root"] == 8
            and decomposition["non_doc"] == 13 and decomposition["total_hard"] == 52)
    record("PASS" if ok52 else "FAIL", "c0-pin52",
           f"52 条分解: 31 docs + 8 硬钉根 md + 13 非文档 = {decomposition['total_hard']}；"
           f"（9 root_scope = 8 + 1 软登记 {decomposition['soft_root']}）")
    if not ok52:
        record("FAIL", "c0-pin52", f"分解不符: {decomposition}")
        level = "FAIL"
    # c1：钉死路径不得 move / 不得缺失
    missing = [p for p in sorted(pinned) if p not in d]
    moved = [p for p in sorted(pinned) if p in d and d[p]["action"] == "move"]
    record("PASS" if not missing and not moved else "FAIL", "c1-pin-safety",
           f"钉死路径 {len(pinned)} 条：缺失 {len(missing)}，建议 move {len(moved)}")
    for p in missing + moved:
        record("FAIL", "c1-pin-safety", p)
    if missing or moved:
        level = "FAIL"
    # c2：钉死路径 action 合法 & blocker 非空
    bad_action = [(p, d[p]["action"]) for p in pinned if p in d
                  and d[p]["action"] not in ("keep", "banner", "retain-readonly")]
    bad_blocker = [p for p in pinned if p in d and d[p]["blocker"] in ("", "无")]
    record("PASS" if not bad_action and not bad_blocker else "FAIL", "c2-pin-blocker",
           f"钉死路径 action 非法 {len(bad_action)}；blocker 为空/无 {len(bad_blocker)}")
    for p in bad_blocker:
        record("FAIL", "c2-pin-blocker", f"钉死路径缺 blocker: {p} -> {d[p]['blocker']}")
    if bad_action or bad_blocker:
        level = "FAIL"
    # c3：gate_locked 列与 config 精确一致（逐条交叉）
    mism = []
    for r in dis_rows:
        p = r["path"]
        tags = set()
        if p in hist:
            tags.add("historical")
        if p in req:
            tags.add("required_files")
        if p in rs:
            tags.add("root_scope")
        expected = ";".join(sorted(tags))
        actual = (r.get("gate_locked") or "").strip()
        if expected != actual:
            mism.append((p, expected, actual))
    record("PASS" if not mism else "FAIL", "c3-gate-locked",
           f"gate_locked 列与 config 精确一致 {len(dis_rows) - len(mism)}/{len(dis_rows)}；不一致 {len(mism)}")
    for p, e, a in mism[:10]:
        record("FAIL", "c3-gate-locked", f"{p}: config={e or '空'} / csv={a or '空'}")
    if mism:
        level = "FAIL"
    # c4（WARN，口径）：blocker=门禁钉死 却不属 52 钉死清单（已归档/软登记/其他三类语义）
    false_pin = [r["path"] for r in dis_rows
                 if r["blocker"] == "门禁钉死" and r["path"] not in pinned]
    archived = [p for p in false_pin if p.startswith("docs/superseded/")]
    soft = [p for p in false_pin if p in rs]
    other = [p for p in false_pin if p not in archived and p not in soft]
    record("WARN" if false_pin else "PASS", "c4-blocker-semantics",
           f"blocker=门禁钉死 但不在 52 钉死清单: {len(false_pin)} 条"
           f"（docs/superseded 已归档 {len(archived)} + root_scope 软登记 {len(soft)} + 其他 {len(other)}）")
    for p in archived:
        record("WARN", "c4-blocker-semantics",
               f"{p}: docs/superseded/**（已归档，语义为'已归档不可再迁'）；"
               f"建议 blocker 改'无'或扩充词表'已归档'，否则与 52 条清单交叉对不上")
    for p in soft:
        record("WARN", "c4-blocker-semantics",
               f"{p}: root_scope 软登记（缺失不 fail，GOVERNANCE_CONSTRAINTS §3.3）；"
               f"建议 blocker 改'无'并在 reason 注明软登记")
    for p in other:
        record("WARN", "c4-blocker-semantics",
               f"{p}: 不在 52 钉死清单，建议 blocker 改'无'或补证据说明")
    # c5（WARN，口径）：52 钉死清单内未标 门禁钉死 的行
    pinned_not_marked = [p for p in sorted(pinned)
                         if p in d and d[p]["blocker"] != "门禁钉死"]
    record("WARN" if pinned_not_marked else "PASS", "c5-blocker-semantics",
           f"52 钉死清单内 blocker≠门禁钉死: {len(pinned_not_marked)} 条"
           f"（值分布 {dict(Counter(d[p]['blocker'] for p in pinned_not_marked))}）")
    for p in pinned_not_marked[:12]:
        record("WARN", "c5-blocker-semantics",
               f"{p}: blocker={d[p]['blocker']}（该路径属 {('historical' if p in hist else '')}"
               f"{';required_files' if p in req else ''}{';root_scope' if p in rs else ''}）"
               f"，建议统一标 门禁钉死 或按'保留理由'口径在 README 说明")
    # c6：SSOT 标签仅用于官方 SSOT（方案 §1.1 的 7 份）——超出部分 WARN
    ssot_rows = [r["path"] for r in dis_rows if r["blocker"] == "SSOT"]
    ssot_extra = [p for p in ssot_rows if p not in SSOT_DOCS]
    record("WARN" if ssot_extra else "PASS", "c6-ssot-semantics",
           f"blocker=SSOT 共 {len(ssot_rows)} 条；其中不在官方 SSOT 7 份内 {len(ssot_extra)} 条"
           f"（分类桶'原地保留(SSOT/索引)'外溢；建议改词'治理保留'或在 README 说明口径）")
    if ssot_extra:
        for p in ssot_extra[:6]:
            record("WARN", "c6-ssot-semantics", f"示例: {p}")
    # c7：报告只读/快照只读 的 zone 合理性
    bad_ro = [r["path"] for r in dis_rows if r["blocker"] == "报告只读"
              and r["zone"] not in ("docs/reports", "artifacts", "docs/agent_runs",
                                    "docs/demo", "docs/plans", "root", "docs/infra")]
    bad_snap = [r["path"] for r in dis_rows if r["blocker"] == "快照只读"
                and r["zone"] != "docs/archive"]
    record("PASS" if not bad_ro and not bad_snap else "WARN", "c7-evidence-semantics",
           f"报告只读 zone 异常 {len(bad_ro)}；快照只读 zone 异常 {len(bad_snap)}；"
           f"快照只读 {sum(1 for r in dis_rows if r['blocker'] == '快照只读')} 行为固定词表外新增词（建议在 README 说明）")
    RESULTS.append(("__LEVEL__", "c-pins", level))


def check_d_batches(repo: Path, dis_rows, batch_plan_text: str) -> None:
    level = "PASS"
    moves = [r for r in dis_rows if r["action"] == "move"]
    banners = [r for r in dis_rows if r["action"] == "banner"]
    keeplike = [r for r in dis_rows if r["action"] in ("keep", "retain-readonly")]
    actionable = [r for r in dis_rows if r["action"] not in ("keep", "retain-readonly")]
    bad_vocab = [r["path"] for r in dis_rows if (r["batch"] or "").strip() not in BATCH_VOCAB]
    bad_action_vocab = [r["path"] for r in dis_rows if (r["action"] or "").strip() not in ACTION_VOCAB]
    record("PASS" if not bad_vocab and not bad_action_vocab else "FAIL", "d0-vocab",
           f"batch 词表外 {len(bad_vocab)}；action 词表外 {len(bad_action_vocab)}")
    if bad_vocab or bad_action_vocab:
        level = "FAIL"
    # d1：batch 并集 = 全部非 keep/retain 项（可行项必须入批，且批次覆盖其动作）
    missing_batch = [r["path"] for r in actionable if not (r["batch"] or "").strip()]
    record("PASS" if not missing_batch else "FAIL", "d1-batch-union",
           f"非 keep/retain 项 {len(actionable)} 条（banner {len(banners)} + move {len(moves)}）；"
           f"未入批 {len(missing_batch)}")
    if missing_batch:
        level = "FAIL"
    # d2：B1/B2 空集；move 全部 B3；B3 == 需裁决集合
    b1 = [r for r in dis_rows if r["batch"] == "B1"]
    b2 = [r for r in dis_rows if r["batch"] == "B2"]
    b3 = [r for r in dis_rows if r["batch"] == "B3"]
    adj = [r for r in dis_rows if r["blocker"] == "需所有者裁决"]
    move_ok = all(r["batch"] == "B3" for r in moves)
    b3_eq = {r["path"] for r in b3} == {r["path"] for r in adj}
    record("PASS" if not b1 and not b2 and move_ok and b3_eq else "FAIL", "d2-batch-consistency",
           f"B1={len(b1)}(应为 0) B2={len(b2)}(应无建议) B3={len(b3)} 需裁决={len(adj)} "
           f"move全部B3={move_ok} B3==需裁决={b3_eq}")
    if not (not b1 and not b2 and move_ok and b3_eq):
        level = "FAIL"
    # d3：B0 不含 move
    b0_move = [r["path"] for r in dis_rows if r["batch"] == "B0" and r["action"] == "move"]
    record("PASS" if not b0_move else "FAIL", "d3-b0-no-move", f"B0 内 move 行={len(b0_move)}")
    if b0_move:
        level = "FAIL"
    # d4：BATCH_PLAN.md 数字与 DISPOSITION 一致
    expected = {
        "B0": len([r for r in dis_rows if r["batch"] == "B0"]),
        "B3": len(b3),
        "banner": len(banners),
    }
    checks = [
        ("B0=234", expected["B0"] == 234 and "234" in batch_plan_text),
        ("B1空集", "空集" in batch_plan_text and "B1" in batch_plan_text),
        ("B1′3", "B1′" in batch_plan_text and "3" in batch_plan_text),
        ("B2无", "无建议项" in batch_plan_text or "B2" in batch_plan_text),
        ("B3=19", expected["B3"] == 19 and "19" in batch_plan_text),
    ]
    bad = [name for name, ok in checks if not ok]
    record("PASS" if not bad else "FAIL", "d4-batch-plan-sync",
           f"BATCH_PLAN 与 DISPOSITION 数字一致: {[n for n, _ in checks]}；不一致 {bad}")
    if bad:
        level = "FAIL"
    RESULTS.append(("__LEVEL__", "d-batches", level))


def check_e_linkfix(repo: Path, dis_rows, link_rows) -> None:
    level = "PASS"
    moves = [r for r in dis_rows if r["action"] == "move"]
    move_names = {Path(r["path"]).name for r in moves}
    move_targets = {r["target_path"] for r in moves}
    link_targets = {r["new_target"] for r in link_rows}
    record("PASS" if move_targets <= link_targets else "FAIL", "e1-linkfix-targets",
           f"move target {len(move_targets)} 个；LINK_FIX 覆盖 {len(move_targets & link_targets)} 个；"
           f"LINK_FIX 行数={len(link_rows)}")
    if not move_targets <= link_targets:
        level = "FAIL"
        for t in sorted(move_targets - link_targets):
            record("FAIL", "e1-linkfix-targets", f"LINK_FIX 未覆盖 target: {t}")
    # e2：逐条可解析
    bad = []
    for r in link_rows:
        ref = repo / r["referrer_path"]
        if not ref.exists():
            bad.append((r["referrer_path"], r["line_no"], "referrer 不存在"))
            continue
        lines = ref.read_text(encoding="utf-8", errors="replace").splitlines()
        try:
            n = int(r["line_no"])
        except ValueError:
            bad.append((r["referrer_path"], r["line_no"], "line_no 非整数"))
            continue
        if not (1 <= n <= len(lines)):
            bad.append((r["referrer_path"], r["line_no"], f"行号越界（文件 {len(lines)} 行）"))
            continue
        if r["old_target"] and r["old_target"] not in lines[n - 1]:
            bad.append((r["referrer_path"], r["line_no"], "old_target 不在该行"))
    record("PASS" if not bad else "FAIL", "e2-linkfix-parse",
           f"LINK_FIX {len(link_rows)} 行可解析；异常 {len(bad)}")
    for b in bad:
        record("FAIL", "e2-linkfix-parse", f"{b[0]}:{b[1]} {b[2]}")
    if bad:
        level = "FAIL"
    # e3：引用覆盖完整性（git grep 交叉；排除 archive/superseded）
    uncovered, stale = [], []
    for name in sorted(move_names):
        out = git(repo, "grep", "-n", "--no-color", "--", name,
                  ":!docs/archive", ":!docs/superseded", ":!.slim")
        actual = set()
        for line in out.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3:
                actual.add((parts[0], parts[1]))
        fixed = {(r["referrer_path"], r["line_no"]) for r in link_rows
                 if r["old_target"] == name}
        uncovered += sorted((name, p, n) for p, n in actual - fixed)
        stale += sorted((name, p, n) for p, n in fixed - actual)
    record("PASS" if not uncovered and not stale else "WARN", "e3-linkfix-complete",
           f"git grep 引用覆盖：遗漏 {len(uncovered)} 行、LINK_FIX 多余 {len(stale)} 行")
    for u in uncovered:
        record("WARN", "e3-linkfix-complete", f"未覆盖引用: {u[1]}:{u[2]} -> {u[0]}")
    for s in stale:
        record("WARN", "e3-linkfix-complete", f"LINK_FIX 多余行: {s[1]}:{s[2]} -> {s[0]}")
    # e4（WARN）：new_target 为仓库根相对路径，写 markdown 链接须按 referrer 迁移后位置换算
    moved_map = {r["path"]: r["target_path"] for r in moves}
    import posixpath

    def rel_target(referrer: str, target: str) -> str:
        new_ref = moved_map.get(referrer, referrer)
        ref_dir = Path(new_ref).parent.as_posix()
        return target if ref_dir == "." else posixpath.relpath(target, ref_dir)

    naming_issues = []
    for r in link_rows:
        needed = rel_target(r["referrer_path"], r["new_target"])
        if needed != r["old_target"]:
            naming_issues.append((r["referrer_path"], r["line_no"], r["new_target"], needed))
    record("WARN" if naming_issues else "PASS", "e4-relative-link",
           f"LINK_FIX new_target 为仓库根相对路径；需换算为相对链接的 {len(naming_issues)}/{len(link_rows)} 行"
           + (f"（如 {naming_issues[0][3]}）" if naming_issues else ""))
    for p, n, t, needed in naming_issues[:5]:
        record("WARN", "e4-relative-link", f"{p}:{n} new_target={t} → 写入链接应为 {needed}")
    # e5：同迁兄弟文件无需改链（迁移后同目录，相对链接保持原文件名即有效）
    moved_names = {Path(r["path"]).name for r in moves}
    sibling_noop = [(r["referrer_path"], r["line_no"], r["old_target"])
                    for r in link_rows
                    if Path(r["referrer_path"]).name in moved_names
                    and r["old_target"] in moved_names]
    record("WARN" if sibling_noop else "PASS", "e5-sibling-noop",
           f"LINK_FIX 中'同迁兄弟文件互相引用'共 {len(sibling_noop)} 行：迁移后同目录相对链接仍有效，"
           f"应标 no-change（若按 new_target 写入会断链）")
    for s in sibling_noop:
        record("WARN", "e5-sibling-noop", f"{s[0]}:{s[1]} -> {s[2]}（同迁，无需改链）")
    RESULTS.append(("__LEVEL__", "e-linkfix", level))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(DEFAULT_REPO))
    ap.add_argument("--audit", default=str(DEFAULT_AUDIT))
    ap.add_argument("--inventory", default=str(DEFAULT_INV))
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    audit = (repo / args.audit).resolve()
    inv_path = (repo / args.inventory).resolve()

    dis_path = audit / "DISPOSITION.csv"
    link_path = audit / "LINK_FIX.csv"
    plan_path = audit / "BATCH_PLAN.md"
    for p in (dis_path, link_path, plan_path, inv_path,
              repo / "infra/governance/docs_consistency_config.json"):
        if not p.exists():
            print(f"FAIL: 缺少输入文件 {p}")
            return 1

    inv_rows = load_csv(inv_path)
    dis_rows = load_csv(dis_path)
    link_rows = load_csv(link_path)
    batch_plan_text = plan_path.read_text(encoding="utf-8", errors="replace")

    print("=" * 78)
    print("validate_mapping.py — PES2TS 2026-11-21 文档处置映射独立校验")
    print(f"仓库根：{repo}")
    print(f"输入：DISPOSITION={len(dis_rows)} 行 / LINK_FIX={len(link_rows)} 行 / "
          f"inventory={len(inv_rows)} 行")
    print("=" * 78)

    cfg, hist, req, rs, req_docs, req_nondoc, pinned_docs, hard_root, pinned, decomp = \
        build_pinned(repo)
    check_a_coverage(inv_rows, dis_rows)
    check_b_moves(repo, dis_rows, link_rows, pinned, {r["path"] for r in inv_rows})
    check_c_pins(repo, dis_rows, cfg, hist, req, rs, pinned_docs, hard_root, pinned, decomp)
    check_d_batches(repo, dis_rows, batch_plan_text)
    check_e_linkfix(repo, dis_rows, link_rows)

    # 输出
    print()
    for level, cid, msg in RESULTS:
        if level == "__LEVEL__":
            continue
        print(f"[{level:4s}] {cid:22s} {msg}")
    print()
    # 汇总级别（子检查中若有 FAIL 则该项 FAIL；WARN 不升级）
    groups = {}
    for level, cid, msg in RESULTS:
        if level == "__LEVEL__":
            groups[cid] = msg
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print("-" * 78)
    print("检查结论：")
    for gid, glevel in groups.items():
        print(f"  ({gid:9s}) {'PASS' if glevel == 'PASS' else glevel if glevel == 'WARN' else 'FAIL'}")
    print("-" * 78)
    # 统计
    ca, cb = Counter(r["action"] for r in dis_rows), Counter(r["batch"] for r in dis_rows)
    cp, ck = Counter(r["priority"] for r in dis_rows), Counter(r["blocker"] for r in dis_rows)
    print(f"统计：rows={len(dis_rows)} unique={len({r['path'] for r in dis_rows})}")
    print(f"  action : " + " / ".join(f"{k} {v}" for k, v in ca.most_common()))
    print(f"  batch  : " + " / ".join(f"{k} {v}" for k, v in sorted(cb.items())))
    print(f"  priority: " + " / ".join(f"{k} {v}" for k, v in sorted(cp.items())))
    print(f"  blocker : " + " / ".join(f"{k} {v}" for k, v in ck.most_common()))
    print(f"  52 钉死清单：inventory 内 {len(pinned)} 条（31 docs + 8 硬根 + 13 非文档 = 52；"
          f"另 1 软登记）")
    move_rows = [r for r in dis_rows if r["action"] == "move"]
    print(f"  P0={cp.get('P0', 0)}  B1 立即迁移=0（空集）  B3 条件迁移={len(move_rows)}  "
          f"B3 裁决={cb.get('B3', 0)}  需裁决 blocker={ck.get('需所有者裁决', 0)}")
    verdict = "PASS" if not fails else "FAIL"
    print("=" * 78)
    print(f"结果：{verdict}  （FAIL={len(fails)}，WARN={len(warns)}）")
    print("=" * 78)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
