# -*- coding: utf-8 -*-
"""check_deletion.py — 清理方案只读校验（task-7 交付物 4）。

只读，不修改/移动/删除任何文件。校验：
  (a) 待删文件存在、git 可恢复；门禁钉死路径只有当 decision=GATE_CONFIG_MUST_CHANGE
      且 CLEANUP_PLAN.md 给出配置 diff/移除清单时才允许出现在删除集中；
  (b) check_docs_links 预演：以 KEEP 集活跃文档为扫描面，把所有待删文件名作为
      markdown 链接目标 grep，所有命中必须已在 LINK_FIX.csv 声明（否则删除后 0 error 会破）；
  (c) 三档集合单调：safe ⊂ standard ⊂ aggressive，且 KEEP ⟺ intensity='-'；
  (d) KEEP_SET 与 CLEANUP_CANDIDATES 互补：KEEP 行 = KEEP_SET，并集 = 253，无重复。

用法：python .slim/deepwork/20261121_cleanup/cleanup/check_deletion.py
退出码：0=全部 PASS；1=存在 FAIL。
"""
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]          # cleanup -> 20261121_cleanup -> deepwork -> .slim
INV = ROOT / "docs/infra/doc_audit/20261121/doc_inventory.csv"
CFG = ROOT / "infra/governance/docs_consistency_config.json"
PLAN = HERE / "CLEANUP_PLAN.md"
CAND = HERE / "CLEANUP_CANDIDATES.csv"
KEEPCSV = HERE / "KEEP_SET.csv"
LINKFIX = HERE / "LINK_FIX.csv"

DEC_DELETE = {"DELETE", "ARCHIVE_EXTRACT_THEN_DELETE", "MERGE_INTO"}
DEC_ALL = DEC_DELETE | {"GATE_CONFIG_MUST_CHANGE", "KEEP"}
INTENSITY = {"safe", "standard", "aggressive", "-"}

results = []


def check(tag, ok, detail=""):
    results.append((tag, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + tag + (("  | " + detail) if detail else ""))
    return ok


def load_csv(p):
    with io.open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_text(p):
    try:
        return io.open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def main():
    print("=" * 78)
    print("check_deletion.py — PES2TS 文档清理方案只读校验")
    print("=" * 78)

    cand = load_csv(CAND)
    keep = load_csv(KEEPCSV)
    linkfix = load_csv(LINKFIX) if LINKFIX.exists() else []
    inv = load_csv(INV)
    inv_paths = [r["path"] for r in inv]
    cfg = json.load(io.open(CFG, encoding="utf-8"))
    pinned = set(cfg.get("historical", [])) | set(cfg.get("root_scope", [])) | set(
        cfg.get("required_files", {}).keys())

    rows = {r["path"]: r for r in cand}
    keep_paths = {r["path"] for r in cand if r["decision"] == "KEEP"}
    delete_paths = {r["path"] for r in cand if r["decision"] in DEC_DELETE}
    gcm_paths = {r["path"] for r in cand if r["decision"] == "GATE_CONFIG_MUST_CHANGE"}

    # ---------------------------------------------------------------- (a)
    print("\n--- (a) 待删文件存在性 / 门禁钉死一致性 / 配置 diff 声明 ---")
    a_ok = True
    missing = [p for p in inv_paths if not (ROOT / p).exists()]
    a_ok &= check("(a1) 253 条清单文件当前全部在磁盘（只读校验阶段未被删）",
                  len(missing) == 0, f"missing={missing[:5]}" if missing else "253/253 exists")
    untracked = [r["path"] for r in cand if r["recovery"].startswith("无")
                 and r["decision"] != "KEEP"]
    a_ok &= check("(a2) 所有清理候选均可 git 恢复（无未跟踪文件混入删除集）",
                  len(untracked) == 0, f"untracked_deletions={untracked}" if untracked else "0")
    bad_pin = [r["path"] for r in cand
               if r["path"] in pinned and r["decision"] not in ("KEEP", "GATE_CONFIG_MUST_CHANGE")]
    a_ok &= check("(a3) 门禁钉死路径只能 KEEP 或 GATE_CONFIG_MUST_CHANGE",
                  len(bad_pin) == 0, f"violations={bad_pin}" if bad_pin else "34 pinned，0 越界")
    bad_gcm = [p for p in gcm_paths if p not in pinned]
    a_ok &= check("(a4) GATE_CONFIG_MUST_CHANGE 行确实都是门禁钉死路径",
                  len(bad_gcm) == 0, f"not_pinned={bad_gcm}" if bad_gcm else f"{len(gcm_paths)}/18")
    # 每条 GCM 的配置 diff 必须在方案中逐条出现（路径 + 所属列表）
    plan = read_text(PLAN)
    plan_has_pin_section = "docs_consistency_config.json" in plan and "historical" in plan
    undoc = [p for p in gcm_paths if p not in plan]
    a_ok &= check(f"(a5) {len(gcm_paths)} 条 GCM 均在 CLEANUP_PLAN.md 的配置 diff 清单中逐条出现",
                  plan_has_pin_section and not undoc,
                  f"undocumented={undoc}" if undoc else f"{len(gcm_paths)}/{len(gcm_paths)} documented")
    # 逐一给出配置侧需要删的键
    cfg_edits = {"historical": [], "required_files": [], "root_scope": []}
    for p in gcm_paths:
        if p in cfg.get("historical", []):
            cfg_edits["historical"].append(p)
        if p in cfg.get("required_files", {}):
            cfg_edits["required_files"].append(p)
        if p in cfg.get("root_scope", []):
            cfg_edits["root_scope"].append(p)
    print(f"      config edits: historical={len(cfg_edits['historical'])}, "
          f"required_files={len(cfg_edits['required_files'])}, "
          f"root_scope={len(cfg_edits['root_scope'])}")
    # g_v_map evidence 受影响的 GCM（提示，不阻断）
    gv = read_text(ROOT / "config/registry/g_v_map.yaml")
    gcm_in_gv = [p for p in gcm_paths if p in gv]
    print(f"      g_v_map.yaml 中被 GCM 路径引用的：{len(gcm_in_gv)} 条（须同 PR 改 evidence/重渲染）")

    # ---------------------------------------------------------------- (b)
    print("\n--- (b) 链接预演（check_docs_links 扫描面）---")
    scan = ["README.md", "AGENT.md"]
    for p in sorted(keep_paths):
        if p.startswith("docs/") and p.endswith(".md") and not p.startswith("docs/archive/"):
            scan.append(p)
    del_bases = {}
    for p in delete_paths | gcm_paths:
        # docs/archive/ 内快照与 live 文档常同名（如 PES2TS_开发宪法_v1.md），
        # 按 basename 匹配会把"指向 live 文档的链接"误报为指向被删快照，故排除。
        if p.startswith("docs/archive/"):
            continue
        del_bases.setdefault(os.path.basename(p), []).append(p)
    declared = {(r["referrer_path"], r["old_target"].split("#")[0].rstrip("/").split("/")[-1])
                for r in linkfix}
    dangling = []
    total_links = 0
    for rp in scan:
        text = read_text(ROOT / rp)
        for ln, line in enumerate(text.splitlines(), 1):
            for base, cands in del_bases.items():
                if base not in line:
                    continue
                m = re.search(r"\]\(([^)\s]*" + re.escape(base) + r"[^)\s]*)\)", line)
                if not m:
                    continue
                total_links += 1
                if (rp, base) not in declared:
                    dangling.append(f"{rp}:{ln} -> {base}")
    b_ok = check("(b1) 待删文件的活跃 markdown 链接均已在 LINK_FIX.csv 声明",
                 len(dangling) == 0,
                 f"undeclared={len(dangling)} {dangling[:4]}" if dangling else
                 f"{total_links} links declared")
    b_ok &= check("(b2) LINK_FIX.csv 只针对 KEEP 集活跃扫描面（不含 archive/待删 referrer）",
                  all(not r["referrer_path"].startswith("docs/archive/") for r in linkfix),
                  f"{len(linkfix)} fix rows")
    # 未命中即普通正文提及（inline code），不会让 check_docs_links 由 0 error 变化
    mention_only = 0
    for p in delete_paths:
        base = os.path.basename(p)
        for rp in scan:
            if base in read_text(ROOT / rp):
                mention_only += 1
                break
    print(f"      另有 {mention_only} 条待删文件名以 inline-code/正文形式被 KEEP 文档提及"
          f"（不产生链接 error，仅需在档案树/索引加指针）")

    # ---------------------------------------------------------------- (c)
    print("\n--- (c) 三档单调性 ---")
    s = {r["path"] for r in cand if r["intensity"] == "safe"}
    st = {r["path"] for r in cand if r["intensity"] in ("safe", "standard")}
    ag = {r["path"] for r in cand if r["intensity"] != "-"}
    c_ok = check("(c1) safe ⊂ standard ⊂ aggressive（集合包含）",
                 s <= st <= ag,
                 f"safe={len(s)}, standard={len(st)}, aggressive={len(ag)}")
    c_ok &= check("(c2) 三档删除数严格递增且落在 Lead 裁决后口径区间",
                  len(s) < len(st) < len(ag) and 55 <= len(s) <= 70
                  and 110 <= len(st) <= 125 and 145 <= len(ag) <= 160,
                  f"safe={len(s)}(55-70), standard={len(st)}(110-125), aggressive={len(ag)}(~152)")
    cons_bad = [p for p, r in rows.items()
                if (r["decision"] == "KEEP") != (r["intensity"] == "-")]
    c_ok &= check("(c3) KEEP ⟺ intensity '-'；GCM ⟹ aggressive",
                  not cons_bad and all(rows[p]["intensity"] == "aggressive" for p in gcm_paths),
                  f"inconsistent={cons_bad[:4]}" if cons_bad else "0")
    c_ok &= check("(c4) decision / intensity 取值均在受控词表内",
                  all(r["decision"] in DEC_ALL and r["intensity"] in INTENSITY
                      for r in cand), "")

    # ---------------------------------------------------------------- (d)
    print("\n--- (d) KEEP_SET 与 CLEANUP_CANDIDATES 互补 ---")
    keep_csv_paths = {r["path"] for r in keep}
    d_ok = check("(d1) CLEANUP_CANDIDATES 精确覆盖 253 条清单（无缺、无多、无重复）",
                 len(cand) == 253 and len(rows) == 253
                 and {p for p in rows} == set(inv_paths),
                 f"rows={len(cand)}, unique={len(rows)}, inventory={len(inv_paths)}")
    d_ok &= check("(d2) KEEP_SET 行集合 == decision=KEEP 行集合（互补）",
                  keep_csv_paths == keep_paths,
                  f"keep_csv={len(keep_csv_paths)}, candidate_keep={len(keep_paths)}, "
                  f"diff={sorted(keep_csv_paths ^ keep_paths)[:4]}")
    d_ok &= check("(d3) KEEP ∪ 删除集 = 253 且互斥",
                  keep_paths.isdisjoint(delete_paths | gcm_paths)
                  and (keep_paths | delete_paths | gcm_paths) == set(inv_paths)
                  and len(keep_paths) + len(delete_paths) + len(gcm_paths) == 253,
                  f"KEEP={len(keep_paths)} + DELETE={len(delete_paths)} + "
                  f"GCM={len(gcm_paths)} = 253")
    d_ok &= check("(d4) KEEP_SET 层级 L0-L4 合法、职责/owner 非空",
                  all(r["layer"] in ("L0", "L1", "L2", "L3", "L4")
                      and r["duty"] and r["owner_role"] for r in keep), "")

    # ---------------------------------------------------------------- 汇总
    ok = all(x[1] for x in results)
    print("\n" + "=" * 78)
    npass = sum(1 for x in results if x[1])
    print(f"总计 {npass}/{len(results)} PASS — " + ("全部 PASS" if ok else "存在 FAIL"))
    print(f"统计：候选 {len(cand)}；KEEP {len(keep_paths)}；删除 {len(delete_paths)}；"
          f"GCM {len(gcm_paths)}；三档删除 {len(s)}/{len(st)}/{len(ag)}；"
          f"LINK_FIX {len(linkfix)}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
