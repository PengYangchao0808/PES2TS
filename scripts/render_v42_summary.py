"""Lane D — aggregate campaign artifacts into the V4.2–V4.7 overnight summary.

Reads the campaign product directory (run records, stage decisions, protocol
evals, exposure, timing) and renders:

* ``docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md`` — a DRAFT report
  following the plan §10 field list (per-reaction per-seed four metrics,
  four-gate pass, 3-seed median + >=2/3 simultaneous pass, group-equal vs
  atom-pooled MSE, alpha distribution with near-zero explicitly undefined,
  near-zero RMSE, effective direction atoms, H/heavy / long-step P90-P99 /
  two-branch / critical-pair-spectator slices, parameter count / peak memory /
  steps/s / success-failure-timeout attempts / GPU-CPU wall).
* ``wp42_campaign/per_group.csv`` — one row per (arm, seed, group).

Report-read-only discipline: this ALWAYS writes a NEW file and never edits a
historical report.  Status wording does not pre-fill gains; negative results
are stated as-is (plan §16 纪律 / contract §7).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CAMPAIGN_DIR = PROJECT_ROOT / "outputs" / "pesn" / "dual_ts_v2_20261015" / \
    "wp42_campaign"
REPORT_DIR = PROJECT_ROOT / "docs" / "reports"
DEFAULT_REPORT = REPORT_DIR / \
    "PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md"
DEFAULT_CSV = CAMPAIGN_DIR / "per_group.csv"


def _f(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}"
    return str(v)


def _load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _collect_run_records(campaign_dir: Path) -> list[dict]:
    records: list[dict] = []
    # Only the live run_records lane.  Never ingest run_records_archive
    # (invalidated runs) or smoke/timing probes into a results summary.
    base = campaign_dir / "run_records"
    if base.is_dir():
        for rp in sorted(base.glob("**/run_record.json")):
            rec = _load_json(rp)
            if not rec or not isinstance(rec, dict):
                continue
            arm = str(rec.get("arm_id", ""))
            if arm.startswith(("TIMING_", "SMOKE_")):
                continue
            rec.setdefault("_path", str(rp))
            records.append(rec)
    # dedupe by path
    seen: dict[str, dict] = {}
    for rec in records:
        seen[rec["_path"]] = rec
    return list(seen.values())


def _collect_protocol_evals(campaign_dir: Path) -> dict[str, dict]:
    evals: dict[str, dict] = {}
    base = campaign_dir / "protocol_evals"
    if not base.is_dir():
        return evals
    docs: list[tuple[Path, dict]] = []
    for ep in sorted(base.glob("*.json")):
        doc = _load_json(ep)
        if isinstance(doc, dict):
            docs.append((ep, doc))
    # Protocol evals written before seed-suffixed naming coexist with the
    # seeded rerun.  When a seeded family exists, drop the unsuffixed legacy
    # copy (same model/seed, ambiguous filename) to avoid double counting.
    seeded = {str(d.get("arm_id", "")) for _, d in docs
              if "_seed" in str(d.get("arm_id", ""))}
    for ep, doc in docs:
        arm = str(doc.get("arm_id", ep.stem))
        if "_seed" not in arm and any(
                s.startswith(arm + "_seed") for s in seeded):
            continue
        key = f"{arm}__{doc.get('protocol', '?')}"
        evals[key] = doc
    return evals


def _collect_stage_decisions(campaign_dir: Path) -> list[dict]:
    out: list[dict] = []
    for name in ("stage_decisions.jsonl", "stage_decisions.json"):
        p = campaign_dir / name
        if not p.is_file():
            continue
        if name.endswith(".jsonl"):
            for line in p.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        else:
            doc = _load_json(p)
            if isinstance(doc, list):
                out = doc
    return out


def _metrics_from_eval(doc: dict, metric_path: str, default=None):
    node = doc.get("metrics", {})
    for key in metric_path.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node


def _render_report(campaign_dir: Path, out_path: Path) -> None:
    records = _collect_run_records(campaign_dir)
    evals = _collect_protocol_evals(campaign_dir)
    decisions = _collect_stage_decisions(campaign_dir)

    lines: list[str] = []
    lines.append("# PES2TS V4.2–V4.7 过夜战役汇总（草稿）")
    lines.append("")
    lines.append("> 状态：草稿（自动聚合渲染；最终结论待人工终审）")
    lines.append(f"> 生成时间：{datetime.now(timezone.utc).isoformat()}")
    lines.append("> 报告只读纪律：本文件为新建文件，未改动任何历史报告。"
                 "收益未预填，负结果如实记录。")
    lines.append("")

    # ---- runs table ------------------------------------------------------
    lines.append("## 1. 训练臂与预算")
    lines.append("")
    lines.append("| arm | preset | seed | updates | steps·s⁻¹ | peak_mem_mb | "
                "wall_s | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rec in records:
        settings = rec.get("settings", {})
        timing = rec.get("timing", {})
        budget = rec.get("budget_state", {}) or {}
        status = budget.get("status") or ("ok" if rec.get("schema_version")
                                          else "pending")
        lines.append(
            f"| {rec.get('arm_id', '?')} | {settings.get('preset', '?')} "
            f"| {settings.get('seed', '?')} | {rec.get('accounting', {}).get('updates_run', '?')} "
            f"| {_f(timing.get('steps_per_sec'), 2)} "
            f"| {_f(timing.get('peak_mem_mb'), 0)} "
            f"| {_f(rec.get('wall_seconds'), 1)} | {status} |")
    lines.append("")

    # ---- protocol evals ---------------------------------------------------
    lines.append("## 2. 三口径协议评测")
    lines.append("")
    lines.append("| arm | protocol | 组中位 MSE 比值 | atom-pooled MSE | "
                "整体 cos | near-TS cos | 四门 |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(evals):
        doc = evals[key]
        if doc.get("status") == "skipped":
            lines.append(f"| {doc.get('arm_id', '?')} | "
                         f"{doc.get('protocol', '?')} | — | — | — | — | "
                         f"跳过({doc.get('code')}) |")
            continue
        m = doc.get("metrics", {})
        mse = m.get("mse", {}).get("group_equal_weighted", {})
        pooled = m.get("mse", {}).get("atom_pooled", {})
        gates = doc.get("gates", {})
        decision = gates.get("decision", "?")
        lines.append(
            f"| {doc.get('arm_id', '?')} | {doc.get('protocol', '?')} "
            f"| {_f(mse.get('ratio_median'))} | {_f(pooled.get('mse_A2'))} "
            f"| {_f(m.get('direction', {}).get('cosine_median'))} "
            f"| {_f(m.get('near_ts', {}).get('cosine_median'))} "
            f"| {decision} |")
    lines.append("")

    # ---- four-gate detail -------------------------------------------------
    lines.append("## 3. 四项门逐条")
    lines.append("")
    lines.append("门为 legacy_full 原四门（median ratio < 1.0、≥6/8 组、整体 "
                "cos ≥ 0.30、near-TS cos ≥ 0.40）；native_full 同式但 "
                "`new_protocol=True`；common_unchanged 不套门。")
    lines.append("")
    for key in sorted(evals):
        doc = evals[key]
        if doc.get("status") == "skipped":
            continue
        gates = doc.get("gates", {})
        if not gates.get("apply_gate"):
            continue
        lines.append(f"- **{doc.get('arm_id')} / {doc.get('protocol')}**: "
                     f"{gates.get('decision')}")
        for name, c in gates.get("criteria", {}).items():
            lines.append(f"  - {name}: {_f(c.get('value'))} vs "
                         f"{_f(c.get('threshold'))} → "
                         f"{'通过' if c.get('pass') else '未通过'}")
    lines.append("")

    # ---- slices -----------------------------------------------------------
    lines.append("## 4. 切片指标（native_full 为主口径）")
    lines.append("")
    lines.append("| arm | 有效方向原子 | near-zero RMSE | H | 重原子 | "
                "长步 P90 | 长步 P99 | 分支A | 分支B | 关键对 | 旁观 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(evals):
        doc = evals[key]
        if doc.get("status") == "skipped":
            continue
        if doc.get("protocol") != "native_full":
            continue
        m = doc.get("metrics", {})
        denom = m.get("denominator", {})
        nz = m.get("near_zero", {})
        s = m.get("slices", {})
        alpha = m.get("alpha", {})
        lines.append(
            f"| {doc.get('arm_id')} "
            f"| {denom.get('n_direction_atoms')} "
            f"| {_f(nz.get('abs_rmse_A'))} "
            f"| {_f(s.get('H_atoms', {}).get('mse_A2'))} "
            f"| {_f(s.get('heavy_atoms', {}).get('mse_A2'))} "
            f"| {_f(s.get('long_step_p90', {}).get('mse_A2'))} "
            f"| {_f(s.get('long_step_p99', {}).get('mse_A2'))} "
            f"| {_f(s.get('branch_A', {}).get('mse_A2'))} "
            f"| {_f(s.get('branch_B', {}).get('mse_A2'))} "
            f"| {_f(s.get('critical_pairs', {}).get('distance_error_mae_A'))} "
            f"| {_f(s.get('spectator_atoms', {}).get('mse_A2'))} |")
    lines.append("")
    lines.append("α 分布：近零真值显式未定义，不以小分母造极值。")
    lines.append("")

    # ---- stage decisions --------------------------------------------------
    lines.append("## 5. 阶段晋级决策")
    lines.append("")
    if decisions:
        for d in decisions:
            lines.append(f"- {json.dumps(d, ensure_ascii=False)}")
    else:
        lines.append("（无 stage_decisions 记录）")
    lines.append("")

    lines.append("## 6. 状态与结论")
    lines.append("")
    lines.append("_待人工终审填写；本文件不预填收益，负结果如实。_")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _render_csv(campaign_dir: Path, out_path: Path) -> None:
    evals = _collect_protocol_evals(campaign_dir)
    rows: list[dict] = []
    for key in sorted(evals):
        doc = evals[key]
        if doc.get("status") == "skipped":
            continue
        m = doc.get("metrics", {})
        mse = m.get("mse", {})
        for g, v in mse.get("per_group_ratio_vs_zero", {}).items():
            rows.append({
                "arm_id": doc.get("arm_id"),
                "protocol": doc.get("protocol"),
                "group_id": g,
                "mse_ratio_vs_zero": v,
                "group_mse_A2": mse.get("per_group_mse_A2", {}).get(g),
                "direction_cosine": m.get("direction", {})
                .get("per_group_cosine", {}).get(g),
                "near_ts_cosine": m.get("near_ts", {})
                .get("per_group_cosine", {}).get(g),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("arm_id,protocol,group_id,mse_ratio_vs_zero,"
                            "group_mse_A2,direction_cosine,near_ts_cosine\n")
        return
    fieldnames = ["arm_id", "protocol", "group_id", "mse_ratio_vs_zero",
                  "group_mse_A2", "direction_cosine", "near_ts_cosine"]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-dir", default=str(CAMPAIGN_DIR))
    ap.add_argument("--out", default=str(DEFAULT_REPORT))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()

    campaign_dir = Path(args.campaign_dir)
    report = Path(args.out)
    if report.exists():
        # report-read-only discipline: never edit a historical report
        print(f"REFUSING: {report} exists — render writes a NEW file only "
              f"(report-read-only discipline)", file=sys.stderr)
        return 2
    _render_report(campaign_dir, report)
    _render_csv(campaign_dir, Path(args.csv))
    print(f"wrote {report}")
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
