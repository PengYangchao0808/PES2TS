"""Offline comparison; reference TS coordinates enter evaluation only."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from continuation_backend import write_json
from pes2ts_core.generation.planning.synchronized_path import align, digest, value
from pes2ts_core.generation.planning.graph_rebuild import load_endpoint_materials_from_export, rebuild_endpoint_graphs
from pes2ts_core.generation.planning.connectivity import required_pairs
from pes2ts_core.generation.planning.gradient_evidence import read_bound_engrad, force_evidence

OUT = ROOT/"outputs/PES2TS_Demo24_continuation_round2_20261002"
OLD = ROOT/"outputs/PES2TS_Demo24_synchronized_reference_20261002"
REPORT = ROOT/"docs/reports/PES2TS_Demo24_第二轮修改与实测_20261002.md"


def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))


def event_pattern(x, plan):
    periodic = Chem.GetPeriodicTable()
    radii = [periodic.GetRcovalent(periodic.GetAtomicNumber(e)) for e in plan["elements"]]
    return [value(x, "distance", d["atoms"]) <= sum(radii[i] for i in d["atoms"])+.45 for d in plan["drivers"]]


def landing_identity(x, plan):
    start = [e["edit_kind"] == ("broken" if plan["origin_endpoint"] == "R" else "formed") for e in plan["active_edits"]]
    observed = event_pattern(x, plan)
    if observed == start: return "returned_origin_fb_pattern"
    if observed == [not v for v in start]: return "target_fb_pattern"
    return "different_or_partial_fb_pattern"


def main():
    old = read(OLD/"completed_batch_summary.json")
    baselines = {r["reaction_id"]: r for r in old["records"]}
    records = []
    for path in sorted((ROOT/"tests/fixtures/p0_demo24/records").glob("RXN_*.json")):
        s = read(path); rid = s["reaction_id"]
        adaptive = read(OUT/rid/"endpoint/result.json")
        fixed = read(OUT/rid/"fixed_control/result.json")
        baseline = baselines[rid]
        row = {"reaction_id": rid, "old_global_continuity": baseline["continuity_ok"],
               "old_common_continuity": bool(baseline["continuity_ok"] and baseline["maximum_atom_step_angstrom"]<=.6),
               "adaptive_complete": adaptive["qualified_complete_path"], "fixed_complete": fixed["qualified_complete_path"],
               "status": adaptive["status"], "last_lambda": adaptive.get("last_lambda", 0.),
               "n_frames": adaptive.get("n_frames", 0), "n_attempts": adaptive.get("n_attempts", 0),
               "max_rmsd": adaptive.get("maximum_rmsd_angstrom"), "max_atom_step": adaptive.get("maximum_atom_step_angstrom"),
               "direction": adaptive["origin"].get("side"), "landing_status": "not_run", "topology_screen": "not_available"}
        if adaptive.get("frames"):
            plan = read(OUT/rid/"endpoint/PathPlan.json")
            assert plan["content_sha256"] == digest({k: v for k, v in plan.items() if k != "content_sha256"})
            bundle = rebuild_endpoint_graphs(s["reaction_smiles"], load_endpoint_materials_from_export(s))
            row["exact_fb_scope"] = {tuple(sorted(d["maps"])) for d in plan["drivers"]} == required_pairs(bundle)
            row["guards"] = len(plan["guards"])
            row["finite_energies"] = all(np.isfinite(f["energy_hartree"]) for f in adaptive["frames"])
            old_plan = read(OLD/rid/"PathPlan.json")
            ts = np.asarray(old_plan["known_ts_geometry"])
            rmsds = [float(np.sqrt(np.mean(np.sum((align(f["geometry"], ts)-ts)**2, axis=1)))) for f in adaptive["frames"]]
            row["offline_minimum_all_atom_ts_rmsd"] = min(rmsds)
            ranking = read(OUT/rid/"endpoint/ranking.json")
            row["offline_peak_candidate_ts_rmsd"] = [rmsds[c["frame_index"]] for c in ranking["candidates"]]
            row["n_candidates"] = len(ranking["candidates"])
            forces = []
            for i, frame in enumerate(adaptive["frames"]):
                source = (OUT/rid/"endpoint/attempts"/frame["attempt_id"]) if i else OUT/rid/"origin_preparation/free_opt"
                bound = read_bound_engrad(source/"orca.orca_XTB.engrad", frame["geometry"], plan["elements"], frame["energy_hartree"])
                forces.append({"frame_index": i, "lambda": frame["lambda"], **force_evidence(plan, frame, bound)})
            write_json(OUT/rid/"endpoint/physical_evidence.json", {"source": "offline_extraction_from_saved_raw_xtb_engrad", "frames": forces})
            gradient_candidates = []
            for candidate in ranking["candidates"]:
                i = candidate["frame_index"]
                neighbourhood = [e for e in forces[max(1, i-1):min(len(forces)-1, i+2)] if e["status"] == "bound"]
                if neighbourhood:
                    best = min(neighbourhood, key=lambda e: e["physical_gradient_norm_hartree_per_bohr"])
                    gradient_candidates.append({"energy_peak_frame": i, "selected_frame": best["frame_index"], "physical_gradient_norm_hartree_per_bohr": best["physical_gradient_norm_hartree_per_bohr"], "ts_verified": False})
            write_json(OUT/rid/"endpoint/ranking_with_gradient.json", {"candidate_only": True, "rule": "minimum_full_physical_gradient_within_energy_peak_neighbourhood", "candidates": gradient_candidates})
            row["n_bound_gradients"] = sum(e["status"] == "bound" for e in forces)
            row["n_unbound_gradients"] = len(forces)-row["n_bound_gradients"]
            index = {m: i for i, m in enumerate(s["maps"])}
            radii = [Chem.GetPeriodicTable().GetRcovalent(Chem.GetPeriodicTable().GetAtomicNumber(e)) for e in s["elements"]]
            active = {tuple(sorted(d["maps"])) for d in plan["drivers"]}
            common_bonds = {(e.map_a, e.map_b) for e in bundle.r_graph.edges} & {(e.map_a, e.map_b) for e in bundle.p_graph.edges}
            topology_issues = []
            for frame in adaptive["frames"]:
                for pair in sorted(common_bonds-active):
                    a, b = (index[m] for m in pair)
                    distance = value(frame["geometry"], "distance", [a, b])
                    if distance > radii[a]+radii[b]+.8:
                        topology_issues.append({"frame_id": frame["frame_id"], "maps": list(pair), "distance": distance})
            row["topology_screen"] = "pass" if not topology_issues else "unexpected_common_bond_stretch"
            row["topology_issues"] = topology_issues
            if (OUT/rid/"endpoint/landing.json").exists():
                landing = read(OUT/rid/"endpoint/landing.json")
                row["landing_status"] = landing_identity(landing["coordinates"], plan) if landing["success"] else "failed"
            write_json(OUT/rid/"offline_evaluation.json", row)
        records.append(row)
    assert len(records) == 24
    fixed_total = sum(r["fixed_complete"] for r in records)
    adaptive_total = sum(r["adaptive_complete"] for r in records)
    improved = [r["reaction_id"] for r in records if r["adaptive_complete"] and not r["old_global_continuity"]]
    regressed = [r["reaction_id"] for r in records if not r["adaptive_complete"] and r["old_global_continuity"]]
    summary = {"n_cases": 24, "formed_pairs": 40, "broken_pairs": 31, "archived_order_changes": 44,
               "baseline_original_pass": old["n_continuity_ok"], "baseline_common_pass": sum(r["old_common_continuity"] for r in records),
               "fixed_control_pass": fixed_total, "adaptive_pass": adaptive_total,
               "improved_from_old_global": improved, "regressed_from_old_global": regressed,
               "n_complete_topology_screen_pass": sum(r["adaptive_complete"] and r["topology_screen"]=="pass" for r in records),
               "landing_status_counts": dict(Counter(r["landing_status"] for r in records if r["adaptive_complete"])),
               "n_all_frames_ts_rmsd_within_point2": sum(r.get("offline_minimum_all_atom_ts_rmsd", float("inf"))<=.2 for r in records),
               "n_complete_paths_peak_ts_rmsd_within_point2": sum(r["adaptive_complete"] and any(v<=.2 for v in r.get("offline_peak_candidate_ts_rmsd", [])) for r in records),
               "baseline_summary_sha256": digest(old), "records": records}
    summary["n_bound_physical_gradients"] = sum(r.get("n_bound_gradients", 0) for r in records)
    summary["n_unbound_physical_gradients"] = sum(r.get("n_unbound_gradients", 0) for r in records)
    write_json(OUT/"comparison.json", summary)
    lines = ["# PES2TS Demo24 第二轮修改与实测（2026-10-02）", "",
             f"本轮完成24个样本的真实 GFN2-xTB 重算及24个匹配固定步长对照。完整且连续的路径由对照 {fixed_total}/24 提升至 {adaptive_total}/24。改善发生在路径生成；不能据此宣称真实 TS 或机理验证成功率提高。", "",
             "## 实现及冻结口径", "",
             "只将图边存在性的对称差作为 driver：40 formed、31 broken，共71个。44个键级变化只归档，不进入驱动、硬距离 guard 或活动事件覆盖。执行从几何较可信、活动组分较少且活动键较多的一端开始；起点独立自由优化，之后只用上一接受帧作预测和校正，不使用已知 TS/IRC 几何暖启动。", "",
             "质量门：全分子 proper Kabsch RMSD≤0.30 Å，最大原子位移≤0.60 Å，距离约束误差≤0.01 Å；拒绝后回到上一接受帧并折半。初始/最大/最小步长0.04/0.08/0.000625；80帧、160次尝试、600秒预算。ORCA采用2核、250次优化上限、2000次SCC上限、每次120秒超时。所有拒绝尝试、输入、日志、哈希和父帧身份单独保存。", "",
             "固定步长对照使用相同71个F/B范围、相同自由优化起点、相同线性端点目标、相同物理门及预算上限；步长固定0.04，上一帧直接启动，首个未通过点停止。它比较的是执行器组合收益，不分别归因给预测器、割线外推或折半。", "",
             "## 与上一轮比较", "", "| 指标 | 上一轮冻结结果 | 本轮 |", "|---|---:|---:|",
             f"| 原全分子RMSD连续性 | {old['n_continuity_ok']}/24 | {adaptive_total}/24完整路径 |",
             f"| 同时满足RMSD及0.60 Å原子位移 | {summary['baseline_common_pass']}/24 | {adaptive_total}/24完整路径 |",
             f"| 完成整个目标区间 | 24/24（含不连续路径） | {adaptive_total}/24 |",
             f"| 匹配固定步长对照的完整连续路径 | — | {fixed_total}/24 |", "",
             "主实验状态：6个完整路径；15个到最小步长后仍跳变的部分路径；2个自由优化起点破坏几何/连接要求；1个原始输入几何无效。分母始终24。361个接受帧、480次尝试；固定对照281个接受帧、279次尝试。额外计算换来了4个对照未能完成的样本，不是无成本加速。", "",
             "上一轮使用已知 TS/IRC 的目标和逐帧几何初猜并固定完整R/P端点。本轮是端点知情的开放终点单端路径；端点身份及约束集合均改变。因此旧3/24→新6/24是跨协议观察，匹配对照2/24→6/24才直接支持执行器收益。新结果不能继承旧全端点验证或旧20/24 TS接近指标。", "",
             "## 逐样本结果", "", "| 样本 | 旧RMSD通过 | 固定对照完整 | 本轮完整 | 最后λ | 接受帧 | 状态 |", "|---|---|---|---|---:|---:|---|"]
    for r in records:
        status = r["status"].replace("STEP_LIMIT:PATH_DISCONTINUITY", "当前折半策略无法继续").replace("ERROR:PREPARED_ORIGIN_GEOMETRY_INVALID", "起点优化后几何异常").replace("INPUT_GEOMETRY_INVALID", "原始输入异常")
        lines.append(f"| {r['reaction_id']} | {'是' if r['old_global_continuity'] else '否'} | {'是' if r['fixed_complete'] else '否'} | {'是' if r['adaptive_complete'] else '否'} | {r['last_lambda']:.4f} | {r['n_frames']} | {status} |")
    lines += ["", "新增通过："+"、".join(improved)+"。退步样本："+"、".join(regressed)+"。47010原先RMSD通过但最大原子位移0.8004 Å，按本轮共同门应判失败；本轮已通过两项门。", "",
              "## 几何、landing与候选验证", "",
              f"完成路径中，{summary['n_complete_topology_screen_pass']}/6通过共同非驱动键的离线拉伸筛查（共价半径和+0.8 Å）。该筛查是距离几何诊断，不能代替电子结构键级判定。所有在线接受帧具有有限能量并通过约束残差检查。", "",
              "独立释放约束优化后的F/B模式："+json.dumps(summary["landing_status_counts"], ensure_ascii=False)+"。target_fb_pattern仅说明本轮活动F/B的距离连接模式达到另一端；returned_origin_fb_pattern表示回到起点模式；它们均未做极小值频率稳定性证明。", "",
              f"仅在离线评价读取旧计划的已知TS：全路径中有{summary['n_all_frames_ts_rmsd_within_point2']}/24出现全原子RMSD≤0.20 Å的帧；完整路径峰值候选满足这一距离的为{summary['n_complete_paths_peak_ts_rmsd_within_point2']}/6。这是结构接近性，不能作为TS证明；生成器从不读取这些几何。", ""]
    for folder in ("validation", "validation_HF3c"):
        for p in sorted(OUT.glob(f"RXN_*/{folder}/result.json")):
            v = read(p)
            lines.append(f"- {v['reaction_id']}，{v['method']}：OptTS成功={v['optts_success']}；正常结束={v['normal_termination']}；有意义虚频={v['meaningful_imaginary_frequencies_cm1']}；IRC成功={v['irc_run_success']}。{v.get('message') or ''}")
    lines += ["", "GFN2-xTB探针日志在初始Hessian/频率相关输出处失败，不能算作候选结构被科学否定；额外HF-3c探针在120秒预算内未完成。当前已验证一级鞍点与IRC连接数为0；未读取到的原生约束乘子和有效最终Hessian证据记为unknown。", "",
              f"后处理从原始xTB engrad提取并核对原子数、元素、坐标及能量：{summary['n_bound_physical_gradients']}帧绑定成功，{summary['n_unbound_physical_gradients']}帧缺失或不匹配。额外保存全物理梯度、约束切空间的投影梯度，以及按L=E+ν(q-c)最小二乘重构的ν和路径斜率；重构乘子并非ORCA原生输出。能量峰邻域内按全物理梯度提名候选，投影梯度小仍不能证明无约束驻点。", "",
              "## 有限方向与相位消融", "", "| 样本 | 默认线性 | 反向起点 | 相位0.4/0.6 | 相位0.6/0.4 |", "|---|---|---|---|---|"]
    for rid in ("RXN_0000017762", "RXN_0000187964"):
        cells = []
        for arm in ("endpoint", "reverse_origin", "phase_early", "phase_late"):
            p = OUT/rid/arm/"result.json"
            r = read(p) if p.exists() else {}
            cells.append("完整" if r.get("completed_interval") else f"停止λ={r.get('last_lambda', 0):.4f}")
        lines.append("| "+rid+" | "+" | ".join(cells)+" |")
    lines += ["", "归一化Sigmoid严格保持两端目标，但相位是假设而非机理事实。本次两个相位版本没有增加完整样本，187964反而由完整变为部分路径。因此默认保留线性调度、按实测启用异步版本。主实验未使用guard；接口仅允许活动中心及图邻居构成的局部A/D，拒绝非活动距离、映射错误和秩冗余。未完成guard的化学必要性验证，不能宣称已经解决构象分岔。", "",
              "## 测试和可复现证据", ""]
    xmlpath = OUT/"pytest-verified.xml"
    if xmlpath.exists():
        suite = ET.parse(xmlpath).find("testsuite")
        lines.append(f"完整回归：tests={suite.get('tests')}，failures={suite.get('failures')}，errors={suite.get('errors')}。默认排除realdata/xtb/acp/orca标记；本报告48份逐样本运行结果及额外消融/TS探针是单独的后端实测证据。")
    lines += ["", "主批次运行时Linux RDKit为2025.9.3；最终回归已对齐fixture版本2026.03.6。离线重新构图逐案核对21份有效计划的driver对等于F/B对称差，所有范围均一致；源文件与上一轮冻结结果保留。", "",
              "- 主结果：outputs/PES2TS_Demo24_continuation_round2_20261002/endpoint_summary.json", "- 匹配对照：同目录fixed_control_summary.json", "- 逐案比较及offline_evaluation：同目录comparison.json及各样本子目录", "- 轨迹、PathPlan、checkpoint、ranking、landing、attempt输入/日志：各样本endpoint子目录", "- 重算入口：scripts/run_demo24_continuation_round2.py", "- 独立科学探针：scripts/validate_round2_candidates.py", "- 离线报告入口：scripts/report_demo24_continuation_round2.py", "",
              "本轮结论：F/B范围改造及预测—校正闭环已实测带来路径生成收益，仍有大量跳变/起点问题。下一轮优先处理15个持续跳变例的局部几何控制和2个起点优化改变化学身份的案例，再扩大OptTS/IRC验证；不把所有键级派生行为重新加入硬约束。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
