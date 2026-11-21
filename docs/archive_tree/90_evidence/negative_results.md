# 跨期负结果总表（negative_results）

> **状态**：Evidence-ReadOnly
> **层级**：L3 证据
> **取代关系**：无
> **上游依赖**：各期条目与原始报告
> **下游消费者**：维护者、论文/评审
> **当前版本**：v1.0 @ 2026-11-21

| # | 期/条目 | 负结果 | 判定口径 | 原始证据 |
|---|---|---|---|---|
| N01 | P1/E01 | G1 含 `truth_assisted_p1` 来源，不能称端到端 truth-free | 阶段定义原文 | `docs/plans/PES2TS_阶段定义_G0-G3.md` §2 G1 |
| N02 | P1/E02 | G2/G2.T 24 例 validated=0；S4 24/24 blocked_on_human_review | 四类终态分账 | 根 `README.md` 2026-10-05 段 |
| N03 | P2/E03 | ADR-0007/0008/0009 三条价值模型路线被 ADR-0010 取代 | ADR 状态行 | `docs/design/decisions/ADR-0007~0009` |
| N04 | P2/E04 | AB3 原生 24 例仍 validated=0；原生引擎未成生产默认 | 四类终态 + 发布门 | `docs/plans/PES2TS_G2_AB3...20261006.md`；根 `README.md` 10-07 段 |
| N05 | P2/E04 | AB4 ORCA-xTB 四点试点仅设计，真实试点未实施 | 状态头 | `docs/plans/PES2TS_G2_AB4...20261007.md` |
| N06 | P2/E06 | 真实 QC 采集/训练/冻结重执行/策略收益均未运行 | 根 README 原文 | 根 `README.md` 旧段 |
| N07 | P3/E07 | v4.1 三臂全未过四门（A 1.0817/2-of-8；B 1.0984/1-of-8；C 1.0555/3-of-8） | 冻结四门 | `docs/reports/PES2TS_v4.1阶段训练...最终报告_20261017.md` |
| N08 | P3/E08 | V4.7 FINAL 三种子四门全负（1.132 / 0-of-8 / 0.180 / 0.248）；全部筛选臂 BELOW_GATE | 冻结四门 + 三口径 | `docs/reports/PES2TS_实现报告_20261018.md` |
| N09 | P3/E08 | fit45 内 342 对近似相同输入却教师方向冲突 → 输入可辨识性缺陷 | 诊断统计 | 同上 §摘要 |
| N10 | P4/E10 | V5.1a 5-fold 门 rejected，V5.1a-R1 无合格候选，V5.1b 关闭 | rejected_gate | 根 `README.md` 10-20 状态段 |
| N11 | P4/E10 | V5.2 q 不可学习（0.507 vs 常数 0.476）、stop 随机（AUC 0.491–0.527）、Move 退化 +5.5%（4/6 破护栏） | 预注册 G3 门 | `24H_FINAL_REPORT.md` §3 |
| N12 | P4/E10 | classic 臂 rollout 崩塌（215 碰撞步）；16/36 物理失败中 15 例溯源自碰撞种子 | rollout 记录 | `24H_FINAL_REPORT.md` §4 |
| N13 | P4/E11 | V5.2 的 Stop 任务定义错误被撤回（92/200 终点行被 terminal 排除；旧正例 d_key_TS 中位 0.021Å） | 只读审计重算 | `docs/reports/PES2TS_V52_任务纠偏审计_20261021.md` |
| N14 | P4/E11 | V5.2-R D1 FAIL / D2 P4P5 FAIL → D2-G STOP、T1 不授权 | 预注册门 | `docs/plans/PES2TS_V52R_...v1_20261021.md` |
| N15 | P5/E12 | Move 规模收益 20k 饱和、50k sign 降 1.7–2.2pp、dead-zone 2.1%→6.0% | 四档同口径 dev 589 | `docs/reports/PES2TS_V53MP_规模训练收益报告_20261027.md` §0 |
| N16 | P5/E12 | endpoint 头 seed 不稳定（C_s29 误停率 0.63）；训练资格 ≠ 部署资格 | inner-val τ 校准 | `docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md` §2 |
| N17 | P6/E13 | V5.4.1 四候选全败：E1/E2 H2 sign、E3 sign+dead、E4 成本门 3.6455>2.0 | H2 硬门/成本门 | `docs/reports/PES2TS_V541_..._20261029.md` §3 |
| N18 | P6/E14 | V5.4.2 `no_backbone_winner`：A2/A3 seed43 H2 breach、A1 未进 top2、A4 被 5% 筛淘汰 | H2/H3 冻结规则 | `docs/reports/PES2TS_V542_..._20261031.md` §0/§4 |
| N19 | P6/E15 | V5.4.3 M3 被 H2 排除、M2 seed43 复现失败；M4/M5/P5 未触发；No-Move 结构性未开放 | H2 非劣/条件激活 | `docs/reports/PES2TS_V543_..._20261106.md` §1/§4 |
| N20 | P7/E16 | 40 份索引双缺、176 warning 容忍债务、7 份根 memo 不可移动 | 审计统计 | `docs/infra/doc_audit/20261121/README.md`；`ANOMALIES.md` A30/C-11 |
| N21 | P7/E17 | `docs_consistency` 未进 CI，12 项冲突在 CLEAN 下并存；"可立即归位"实际 0 | C-09/C-08；FAMILIES §四 | `docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md` |

---
