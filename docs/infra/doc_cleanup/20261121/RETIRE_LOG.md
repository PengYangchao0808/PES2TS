# 退役流水（RETIRE_LOG）— 2026-11-21/22 文档清理

> **状态**：Evidence-ReadOnly（append-only；不得回改历史行）
> **层级**：L3 证据 / L4 运营（退役执行流水）
> **取代关系**：无
> **上游依赖**：ADR-0035、`docs/infra/PES2TS_开发档案树与文档清理方案_v1_20261121.md` §5、`CLEANUP_CANDIDATES.csv`
> **下游消费者**：仓库所有者、审计、恢复操作
> **当前版本**：v1.0 @ 2026-11-22（tag `pre-doc-purge-20261121`；bundle/tar 备份在仓库父目录）

## 执行摘要

| 项 | 值 |
|---|---|
| 退役文件数 | **107**（D1 56 + D2 51） |
| 退役方式 | `git rm`（工作树删除；git 历史保留） |
| 恢复锚点 | tag `pre-doc-purge-20261121`（commit `76ab46c` 之后）、`PES2TS-pre-doc-purge-20261121.bundle`、`PES2TS-docs-20261121.tar.gz` |
| 链接修正 | 143 处（38 个文件），0 error |
| 门禁 | consistency 0 violation；links 0 error（warning 172→132）；G–V current；registry 26 passed |
| 暂缓 | `MERGE_INTO` 8 份（需先并入目标文档）；`GATE_CONFIG_MUST_CHANGE` 15 份（aggressive 档，需同 PR 改门禁配置）；`docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md`（被 `g_v_map` 引用，保留） |

## 恢复命令

```powershell
# 单文件恢复（不改历史）
git show pre-doc-purge-20261121:<path> > <path>
# 或整文件回到工作树（含暂存区）
git restore --source=pre-doc-purge-20261121 -- <path>
# 整批回滚（未 push 时）
git reset --hard pre-doc-purge-20261121
# 最坏情况：从 bundle 重建仓库副本
git clone ..\PES2TS-pre-doc-purge-20261121.bundle PES2TS-restored
```

## 退役清单

| # | 批次 | 原路径 | 档案条目 | 恢复锚点 |
|---:|---|---|---|---|
| 1 | D1 | `V53MP_RESUME_20261022.md` | E12 | `pre-doc-purge-20261121` |
| 2 | D1 | `docs/agent_runs/E0_SIGNOFF_REPORT.md` | E10 | `pre-doc-purge-20261121` |
| 3 | D1 | `docs/agent_runs/handoffs/W-DATA_E7_stats.md` | E10 | `pre-doc-purge-20261121` |
| 4 | D1 | `docs/agent_runs/handoffs/W-FEAT_E4_v52labels.md` | E10 | `pre-doc-purge-20261121` |
| 5 | D1 | `docs/agent_runs/handoffs/W-FEAT_E6_v53exec.md` | E10 | `pre-doc-purge-20261121` |
| 6 | D1 | `docs/agent_runs/handoffs/W-FEAT_E6_v53prep.md` | E10 | `pre-doc-purge-20261121` |
| 7 | D1 | `docs/archive/20261007_pre_light_policy/AGENT.md` | E06 | `pre-doc-purge-20261121` |
| 8 | D1 | `docs/archive/20261007_pre_light_policy/docs/PES2TS_开发宪法_v1.md` | E06 | `pre-doc-purge-20261121` |
| 9 | D1 | `docs/archive/20261007_pre_light_policy/docs/README.md` | E06 | `pre-doc-purge-20261121` |
| 10 | D1 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md` | E06 | `pre-doc-purge-20261121` |
| 11 | D1 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md` | E06 | `pre-doc-purge-20261121` |
| 12 | D1 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_完整开发流方案_整合版_v1.md` | E06 | `pre-doc-purge-20261121` |
| 13 | D1 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_整体开发与发布方案_v2.md` | E06 | `pre-doc-purge-20261121` |
| 14 | D1 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_文档修订台账.md` | E06 | `pre-doc-purge-20261121` |
| 15 | D1 | `docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_阶段定义_G0-G3.md` | E06 | `pre-doc-purge-20261121` |
| 16 | D1 | `docs/archive/20261007_pre_light_policy/source_README.md` | E06 | `pre-doc-purge-20261121` |
| 17 | D1 | `docs/archive/20261007_pre_orca_xtb_pilot/files/AGENT.md` | E04 | `pre-doc-purge-20261121` |
| 18 | D1 | `docs/archive/20261007_pre_orca_xtb_pilot/files/README.md` | E04 | `pre-doc-purge-20261121` |
| 19 | D1 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/PES2TS_开发宪法_v1.md` | E04 | `pre-doc-purge-20261121` |
| 20 | D1 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/README.md` | E04 | `pre-doc-purge-20261121` |
| 21 | D1 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/plans/PES2TS_文档修订台账.md` | E04 | `pre-doc-purge-20261121` |
| 22 | D1 | `docs/archive/20261007_pre_orca_xtb_pilot/files/docs/plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md` | E04 | `pre-doc-purge-20261121` |
| 23 | D1 | `docs/archive/20261007_pre_unified_residual_rl/AGENT.md` | E06 | `pre-doc-purge-20261121` |
| 24 | D1 | `docs/archive/20261007_pre_unified_residual_rl/docs__PES2TS_开发宪法_v1.md` | E06 | `pre-doc-purge-20261121` |
| 25 | D1 | `docs/archive/20261007_pre_unified_residual_rl/docs__README.md` | E06 | `pre-doc-purge-20261121` |
| 26 | D1 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md` | E06 | `pre-doc-purge-20261121` |
| 27 | D1 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_PES-N未来开发计划_v2_20261005.md` | E06 | `pre-doc-purge-20261121` |
| 28 | D1 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_文档修订台账.md` | E06 | `pre-doc-purge-20261121` |
| 29 | D1 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_轻量控制与RL衔接计划_v1_20261007.md` | E06 | `pre-doc-purge-20261121` |
| 30 | D1 | `docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_阶段定义_G0-G3.md` | E06 | `pre-doc-purge-20261121` |
| 31 | D1 | `docs/archive/20261008_pre_scale20k_rl2k/AGENT.md` | E06 | `pre-doc-purge-20261121` |
| 32 | D1 | `docs/archive/20261008_pre_scale20k_rl2k/docs/README.md` | E06 | `pre-doc-purge-20261121` |
| 33 | D1 | `docs/archive/20261008_pre_scale20k_rl2k/docs/plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md` | E06 | `pre-doc-purge-20261121` |
| 34 | D1 | `docs/archive/20261008_pre_scale20k_rl2k/docs/plans/PES2TS_文档修订台账.md` | E06 | `pre-doc-purge-20261121` |
| 35 | D1 | `docs/demo/PES2TS_Demo实现审核与架构解读_20260930.md` | E02 | `pre-doc-purge-20261121` |
| 36 | D1 | `docs/demo/PES2TS_demo24_反应挑选与复核方案.md` | E02 | `pre-doc-purge-20261121` |
| 37 | D1 | `docs/plans/PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md` | E07 | `pre-doc-purge-20261121` |
| 38 | D1 | `docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md` | E07 | `pre-doc-purge-20261121` |
| 39 | D1 | `docs/plans/PES2TS_G0-G4整体开发框架修改方案_v1_20261005.md` | E03 | `pre-doc-purge-20261121` |
| 40 | D1 | `docs/plans/PES2TS_G1_G2边界审计与补齐任务_20261003.md` | E02 | `pre-doc-purge-20261121` |
| 41 | D1 | `docs/plans/PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md` | E07 | `pre-doc-purge-20261121` |
| 42 | D1 | `docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md` | E03 | `pre-doc-purge-20261121` |
| 43 | D1 | `docs/plans/PES2TS_双端拓扑条件训练与项目进度统一方案_v4_20261015.md` | E07 | `pre-doc-purge-20261121` |
| 44 | D1 | `docs/plans/PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md` | E02 | `pre-doc-purge-20261121` |
| 45 | D1 | `docs/plans/PES2TS_扫描规划谱系合并方案_20260930.md` | E02 | `pre-doc-purge-20261121` |
| 46 | D1 | `docs/plans/PES2TS_拓扑约束TS接近度与位移联合学习方案_v2_20261004.md` | E03 | `pre-doc-purge-20261121` |
| 47 | D1 | `docs/plans/PES2TS_统一价值模型与搜索排序一体化发展方案_v1_20261004.md` | E03 | `pre-doc-purge-20261121` |
| 48 | D1 | `docs/plans/PES2TS_统一开发顺序与阶段验收.md` | E02 | `pre-doc-purge-20261121` |
| 49 | D1 | `docs/plans/PES2TS_逐步实施与验证方案.md` | E01 | `pre-doc-purge-20261121` |
| 50 | D1 | `docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md` | E09 | `pre-doc-purge-20261121` |
| 51 | D1 | `docs/reports/PES2TS_对话与现有任务完整总结报告_20261002.md` | E02 | `pre-doc-purge-20261121` |
| 52 | D1 | `docs/reports/PES2TS_开发阶段性报告_20260930.md` | E01 | `pre-doc-purge-20261121` |
| 53 | D1 | `docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v1_20261020.md` | E11 | `pre-doc-purge-20261121` |
| 54 | D1 | `docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v2_20261020.md` | E11 | `pre-doc-purge-20261121` |
| 55 | D1 | `docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v1_20261020.md` | E11 | `pre-doc-purge-20261121` |
| 56 | D1 | `docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v2_20261020.md` | E11 | `pre-doc-purge-20261121` |
| 57 | D2 | `docs/agent_runs/BOOTSTRAP_REVIEW.md` | E10 | `pre-doc-purge-20261121` |
| 58 | D2 | `docs/agent_runs/handoffs/W-DATA_E1_sixtables.md` | E10 | `pre-doc-purge-20261121` |
| 59 | D2 | `docs/agent_runs/handoffs/W-FEAT_E2_v3_stack.md` | E10 | `pre-doc-purge-20261121` |
| 60 | D2 | `docs/agent_runs/handoffs/W-LOSS_E3_BVL_arms.md` | E10 | `pre-doc-purge-20261121` |
| 61 | D2 | `docs/agent_runs/handoffs/W-LOSS_E3_trainer_and_runs.md` | E10 | `pre-doc-purge-20261121` |
| 62 | D2 | `docs/agent_runs/handoffs/W-LOSS_E5_v52.md` | E10 | `pre-doc-purge-20261121` |
| 63 | D2 | `docs/agent_runs/handoffs/W-LOSS_WP-L3_ledger_hardening.md` | E10 | `pre-doc-purge-20261121` |
| 64 | D2 | `docs/contracts/PES2TS_G2T_战役合同字段字典_v1.md` | E02 | `pre-doc-purge-20261121` |
| 65 | D2 | `docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md` | E08 | `pre-doc-purge-20261121` |
| 66 | D2 | `docs/design/G1_TS_IRC_映射与反应归簇实现方案.md` | E01 | `pre-doc-purge-20261121` |
| 67 | D2 | `docs/design/G1_v2_L0到G2扫描策略设计.md` | E02 | `pre-doc-purge-20261121` |
| 68 | D2 | `docs/design/G1_v2_ORCA扫描模式识别与计划.md` | E02 | `pre-doc-purge-20261121` |
| 69 | D2 | `docs/design/G1_v2_一维优先与xTB二维成本策略.md` | E02 | `pre-doc-purge-20261121` |
| 70 | D2 | `docs/design/G1_v2_产物成键逆向扫描优先策略.md` | E02 | `pre-doc-purge-20261121` |
| 71 | D2 | `docs/design/G1_v2_多成键与少成键方向对比.md` | E02 | `pre-doc-purge-20261121` |
| 72 | D2 | `docs/design/G1_v2_成断键扫描方向选择可视化.html` | E02 | `pre-doc-purge-20261121` |
| 73 | D2 | `docs/design/G1_v2_方向无关成键锚点与双向扫描策略.md` | E02 | `pre-doc-purge-20261121` |
| 74 | D2 | `docs/design/G1_v2_补全实施总方案.md` | E02 | `pre-doc-purge-20261121` |
| 75 | D2 | `docs/design/PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md` | E03 | `pre-doc-purge-20261121` |
| 76 | D2 | `docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md` | E02 | `pre-doc-purge-20261121` |
| 77 | D2 | `docs/infra/wsl_memory_governance.md` | E12 | `pre-doc-purge-20261121` |
| 78 | D2 | `docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md` | E02 | `pre-doc-purge-20261121` |
| 79 | D2 | `docs/plans/PES2TS_G2T真实战役执行方案_20261004.md` | E04 | `pre-doc-purge-20261121` |
| 80 | D2 | `docs/plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md` | E04 | `pre-doc-purge-20261121` |
| 81 | D2 | `docs/plans/PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md` | E04 | `pre-doc-purge-20261121` |
| 82 | D2 | `docs/plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md` | E02 | `pre-doc-purge-20261121` |
| 83 | D2 | `docs/plans/PES2TS_G2_PESG收紧算法迭代方案_20261003.md` | E02 | `pre-doc-purge-20261121` |
| 84 | D2 | `docs/plans/PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md` | E04 | `pre-doc-purge-20261121` |
| 85 | D2 | `docs/plans/PES2TS_G2_下一批算法调整与测试方案_20261003.md` | E02 | `pre-doc-purge-20261121` |
| 86 | D2 | `docs/plans/PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md` | E02 | `pre-doc-purge-20261121` |
| 87 | D2 | `docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md` | E04 | `pre-doc-purge-20261121` |
| 88 | D2 | `docs/plans/PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md` | E06 | `pre-doc-purge-20261121` |
| 89 | D2 | `docs/plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md` | E06 | `pre-doc-purge-20261121` |
| 90 | D2 | `docs/plans/PES2TS_RL自适应路径生长与IRC验证研究方案_v1_20261009.md` | E06 | `pre-doc-purge-20261121` |
| 91 | D2 | `docs/reports/PES2TS_Demo24_ACP首批计算_20261001.md` | E02 | `pre-doc-purge-20261121` |
| 92 | D2 | `docs/reports/PES2TS_Demo24_TS_IRC算法诊断_20261001.md` | E02 | `pre-doc-purge-20261121` |
| 93 | D2 | `docs/reports/PES2TS_Demo24_多坐标全批计算结果_20261002.md` | E02 | `pre-doc-purge-20261121` |
| 94 | D2 | `docs/reports/PES2TS_Demo24_端点与TS几何核查_20261001.md` | E02 | `pre-doc-purge-20261121` |
| 95 | D2 | `docs/reports/PES2TS_Demo24_第二轮修改与实测_20261002.md` | E02 | `pre-doc-purge-20261121` |
| 96 | D2 | `docs/reports/PES2TS_Direct_fit50训练表征分析_20261015.md` | E07 | `pre-doc-purge-20261121` |
| 97 | D2 | `docs/reports/PES2TS_Direct批次1-6实施记录_20261015.md` | E07 | `pre-doc-purge-20261121` |
| 98 | D2 | `docs/reports/PES2TS_GPU环境与磁盘预算执行记录_20261015.md` | E07 | `pre-doc-purge-20261121` |
| 99 | D2 | `docs/reports/PES2TS_PES-C-G架构审核与创新性可行性分析_20261004.md` | E03 | `pre-doc-purge-20261121` |
| 100 | D2 | `docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md` | E08 | `pre-doc-purge-20261121` |
| 101 | D2 | `docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md` | E12 | `pre-doc-purge-20261121` |
| 102 | D2 | `docs/reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md` | E12 | `pre-doc-purge-20261121` |
| 103 | D2 | `docs/reports/PES2TS_V53MP_S2_5k规模报告_20261022.md` | E12 | `pre-doc-purge-20261121` |
| 104 | D2 | `docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md` | E12 | `pre-doc-purge-20261121` |
| 105 | D2 | `docs/reports/PES2TS_V53MP_S3_闭环评估报告_20261023.md` | E12 | `pre-doc-purge-20261121` |
| 106 | D2 | `docs/reports/PES2TS_V53MP_S4_20k_vs_50k_配对统计与C臂导出_20261027.md` | E12 | `pre-doc-purge-20261121` |
| 107 | D2 | `docs/reports/PES2TS_V53MP_S4_50k_单路正式训练报告_20261027.md` | E12 | `pre-doc-purge-20261121` |
