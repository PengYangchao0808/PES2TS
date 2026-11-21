# -*- coding: utf-8 -*-
"""生成 CLEANUP_CANDIDATES.csv 与 KEEP_SET.csv（253 条全覆盖）。

只读输入：
  docs/infra/doc_audit/20261121/{doc_inventory.csv,DISPOSITION.csv,classification_mapping.csv}
  .slim/deepwork/20261121_cleanup/archive_tree/ARCHIVE_MAP.csv
只写本目录（cleanup/）。
三档强度定义（intensity = 最早删除档，删除档单调）：
  safe       -> safe 起即删（safe ⊂ standard ⊂ aggressive 都删）
  standard   -> standard / aggressive 删（safe 保留）
  aggressive -> 仅 aggressive 删
  -          -> KEEP（三档都保留）
"""
import csv
import io
import collections
import sys

BASE = "docs/infra/doc_audit/20261121/"
AMAP = ".slim/deepwork/20261121_cleanup/archive_tree/ARCHIVE_MAP.csv"
OUT = ".slim/deepwork/20261121_cleanup/cleanup/"

ZONE_LABEL = {
    "root": "根目录", "docs-root": "docs/根", "docs-plans": "docs/plans",
    "docs-design": "docs/design", "docs-design-decisions": "docs/design/decisions",
    "docs-contracts": "docs/contracts", "docs-reports": "docs/reports",
    "docs-archive": "docs/archive", "docs-superseded": "docs/superseded",
    "docs-agent-runs": "docs/agent_runs", "docs-infra": "docs/infra",
    "docs-demo": "docs/demo", "artifacts": "artifacts", "data-outputs": "data/outputs",
    "examples": "examples", "scripts": "scripts",
}

# ---------------------------------------------------------------- KEEP 集（71）
KEEP = set("""
README.md
AGENT.md
CONTRIBUTING.md
LICENSE
requirements.txt
requirements-pesn-runtime.txt
requirements-pesn-training.txt
V53_ROLLOUT_REPORT.md
data/README.md
outputs/README.md
docs/PES2TS_开发宪法_v1.md
docs/README.md
docs/archive/README.md
docs/superseded/README.md
docs/agent_runs/README.md
docs/contracts/PES2TS_MoveProcessContract_v1.md
docs/contracts/PES2TS_GeometryPolicyContract_v2.md
docs/contracts/PES2TS_ReactionIntent_Contract_v2.md
docs/design/PES2TS_G_V_MAPPING_v1_20261018.md
docs/design/PES2TS_三模块与可视化产品架构_v1_20261005.md
docs/design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md
docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md
docs/infra/doc_audit/20261121/README.md
docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md
docs/infra/doc_audit/20261121/MIGRATION_LOG.md
examples/contracts_v1/README.md
examples/contracts_v1/path_viewer.html
examples/m1_first_plan/README.md
scripts/README.md
scripts/archive/README.md
""".split())
# 8 份 artifacts 模型卡
KEEP |= {
    "artifacts/pesn/pesn-control-demo24-bc-v0.1.0/model_card.md",
    "artifacts/pesn/pesn-control-demo24-bc-v0.1.1-rlrepair1/model_card.md",
    "artifacts/pesn/pesn-control-demo24-cql-v0.1.1-rlrepair1/model_card.md",
    "artifacts/pesn/pesn-control-demo24-v0.1.0/model_card.md",
    "artifacts/pesn/pesn-control-scale-bc-v0.1.0-rl50/model_card.md",
    "artifacts/pesn/pesn-control-scale-cql-v0.1.0-rl50/model_card.md",
    "artifacts/pesn/pesn-irc-scale5k-v0.1.0-rc.1/model_card.md",
    "artifacts/pesn/pesn-irc-v0.1.0-rc.1/model_card.md",
}
# 9 条现行/必留 ADR
KEEP |= {
    "docs/design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md",
    "docs/design/decisions/ADR-0011-三模块产品架构与统一可视化.md",
    "docs/design/decisions/ADR-0025-G-V双轴映射与战役注册.md",
    "docs/design/decisions/ADR-0026-V5冲刺key-atom位移合同与OOF语义.md",
    "docs/design/decisions/ADR-0030-V5.2全路径导航与端点停止.md",
    "docs/design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md",
    "docs/design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md",
    "docs/design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md",
    "docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md",
}
# 12 条现行计划/治理计划
KEEP |= {
    "docs/plans/PES2TS_文档修订台账.md",
    "docs/plans/PES2TS_阶段定义_G0-G3.md",
    "docs/plans/PES2TS_完整开发流方案_整合版_v1.md",
    "docs/plans/PES2TS_整体开发与发布方案_v2.md",
    "docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md",
    "docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md",
    "docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md",
    "docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md",
    "docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md",
    "docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md",
    "docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md",
    "docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md",
}
# 5 条必须留下的证据报告（g_v_map executed evidence / 最新终报）
KEEP |= {
    "docs/reports/PES2TS_实现报告_20261018.md",
    "docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md",
    "docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md",
    "docs/reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md",
    "docs/reports/PES2TS_V52_任务纠偏审计_20261021.md",
}
# 8 份 archive manifest（快照 README，绝不做"墓地清空"）
KEEP |= {
    "docs/archive/20261005_pre_pesn/README.md",
    "docs/archive/20261006_pre_g3p/README.md",
    "docs/archive/20261007_pre_light_policy/README.md",
    "docs/archive/20261007_pre_orca_xtb_pilot/README.md",
    "docs/archive/20261007_pre_predictor_scope/README.md",
    "docs/archive/20261007_pre_unified_residual_rl/README.md",
    "docs/archive/20261008_pre_scale20k_rl2k/README.md",
}

# ---------------------------------------------------------------- SAFE 集（67）
SAFE = set()
# 1) 28 份 archive 快照载荷（payload 是 live 文档/旧计划的逐字节副本；README manifest 保留）
SAFE |= {
    "docs/archive/20261007_pre_light_policy/AGENT.md",
    "docs/archive/20261007_pre_light_policy/docs/PES2TS_开发宪法_v1.md",
    "docs/archive/20261007_pre_light_policy/docs/README.md",
    "docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md",
    "docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md",
    "docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_完整开发流方案_整合版_v1.md",
    "docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_整体开发与发布方案_v2.md",
    "docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_文档修订台账.md",
    "docs/archive/20261007_pre_light_policy/docs/plans/PES2TS_阶段定义_G0-G3.md",
    "docs/archive/20261007_pre_light_policy/source_README.md",
    "docs/archive/20261007_pre_orca_xtb_pilot/files/AGENT.md",
    "docs/archive/20261007_pre_orca_xtb_pilot/files/README.md",
    "docs/archive/20261007_pre_orca_xtb_pilot/files/docs/PES2TS_开发宪法_v1.md",
    "docs/archive/20261007_pre_orca_xtb_pilot/files/docs/README.md",
    "docs/archive/20261007_pre_orca_xtb_pilot/files/docs/plans/PES2TS_文档修订台账.md",
    "docs/archive/20261007_pre_orca_xtb_pilot/files/docs/plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md",
    "docs/archive/20261007_pre_unified_residual_rl/AGENT.md",
    "docs/archive/20261007_pre_unified_residual_rl/docs__PES2TS_开发宪法_v1.md",
    "docs/archive/20261007_pre_unified_residual_rl/docs__README.md",
    "docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md",
    "docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_PES-N未来开发计划_v2_20261005.md",
    "docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_文档修订台账.md",
    "docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_轻量控制与RL衔接计划_v1_20261007.md",
    "docs/archive/20261007_pre_unified_residual_rl/docs__plans__PES2TS_阶段定义_G0-G3.md",
    "docs/archive/20261008_pre_scale20k_rl2k/AGENT.md",
    "docs/archive/20261008_pre_scale20k_rl2k/docs/README.md",
    "docs/archive/20261008_pre_scale20k_rl2k/docs/plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md",
    "docs/archive/20261008_pre_scale20k_rl2k/docs/plans/PES2TS_文档修订台账.md",
}
# 2) 已取代区 4 份（2026-11-20 迁入；superseded/README 索引保留取代者）
SAFE |= {
    "docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v1_20261020.md",
    "docs/superseded/PES2TS_V5.1-R3_端点对称特征与损失重训方案_v2_20261020.md",
    "docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v1_20261020.md",
    "docs/superseded/PES2TS_单端点IRC生成控制器_产品设计_v2_20261020.md",
}
# 3) 已被取代的旧版本计划（18）：F19 三代 + 10 份 banner 历史提案 + 5 份一次性清单/旧运营稿
SAFE |= {
    "docs/plans/PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md",
    "docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md",
    "docs/plans/PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md",
    "docs/plans/PES2TS_G0-G4整体开发框架修改方案_v1_20261005.md",
    "docs/plans/PES2TS_G1_G2边界审计与补齐任务_20261003.md",
    "docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md",
    "docs/plans/PES2TS_双端拓扑条件训练与项目进度统一方案_v4_20261015.md",
    "docs/plans/PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md",
    "docs/plans/PES2TS_扫描规划谱系合并方案_20260930.md",
    "docs/plans/PES2TS_拓扑约束TS接近度与位移联合学习方案_v2_20261004.md",
    "docs/plans/PES2TS_统一价值模型与搜索排序一体化发展方案_v1_20261004.md",
    "docs/plans/PES2TS_统一开发顺序与阶段验收.md",
    "docs/plans/PES2TS_逐步实施与验证方案.md",
    "docs/plans/PES2TS_V5.3-MP_文档同步清单_v1_20261021.md",
    "docs/plans/PES2TS_实施路线图.html",
    "docs/plans/PES2TS_双项目开发与Demo方案.md",
    "docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md",
    "docs/plans/PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md",
}
# 4) 4 份历史提案 ADR（自身已声明被 ADR-0010 系列取代）
SAFE |= {
    "docs/design/decisions/ADR-0007-统一价值模型与搜索内生选点.md",
    "docs/design/decisions/ADR-0008-拓扑约束下的距离与位移联合预测.md",
    "docs/design/decisions/ADR-0009-G阶段下的统一模型开发框架.md",
    "docs/design/decisions/ADR-0020-EGNN直接位移与BC-IQL-SAC研究主线.md",
}
# 5) 5 份零引用过程记录 + 1 份一次性 resume memo
SAFE |= {
    "docs/agent_runs/E0_SIGNOFF_REPORT.md",
    "docs/agent_runs/handoffs/W-DATA_E7_stats.md",
    "docs/agent_runs/handoffs/W-FEAT_E4_v52labels.md",
    "docs/agent_runs/handoffs/W-FEAT_E6_v53exec.md",
    "docs/agent_runs/handoffs/W-FEAT_E6_v53prep.md",
    "V53MP_RESUME_20261022.md",
}
# 6) 1 份被 v2 方案取代的旧清理方案
SAFE.add("docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md")
# 7) 3 份 old demo 设计（Demo24 结论已被 E02/E08 报告与 archive tree 收敛）
SAFE |= {
    "docs/demo/PES2TS_Demo实现审核与架构解读_20260930.md",
    "docs/demo/PES2TS_Demo评估指标_v1.md",
    "docs/demo/PES2TS_demo24_反应挑选与复核方案.md",
}
# 8) 3 份纯过程报告（结论已在后续终报/阶段定义中重复）
SAFE |= {
    "docs/reports/PES2TS_开发阶段性报告_20260930.md",
    "docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md",
    "docs/reports/PES2TS_对话与现有任务完整总结报告_20261002.md",
}

# ---------------------------------------------------------------- STANDARD-only（65）
STD = set()
STD |= {
    "docs/agent_runs/BOOTSTRAP_REVIEW.md",
    "docs/agent_runs/PARALLEL_EXECUTION_PLAN.md",
    "docs/agent_runs/handoffs/W-DATA_E1_sixtables.md",
    "docs/agent_runs/handoffs/W-FEAT_E2_v3_stack.md",
    "docs/agent_runs/handoffs/W-LOSS_E3_BVL_arms.md",
    "docs/agent_runs/handoffs/W-LOSS_E3_trainer_and_runs.md",
    "docs/agent_runs/handoffs/W-LOSS_E5_v52.md",
    "docs/agent_runs/handoffs/W-LOSS_WP-L3_ledger_hardening.md",
}
STD |= {
    "docs/contracts/PES2TS_G2T_战役合同字段字典_v1.md",
    "docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md",
    "docs/contracts/PES2TS_contracts_v1.md",
    "docs/contracts/PES2TS_数据规范与ACP对接优化方案.md",
}
STD |= {
    "docs/design/G1_TS_IRC_映射与反应归簇实现方案.md",
    "docs/design/G1_v2_L0到G2扫描策略设计.md",
    "docs/design/G1_v2_ORCA扫描模式识别与计划.md",
    "docs/design/G1_v2_一维优先与xTB二维成本策略.md",
    "docs/design/G1_v2_产物成键逆向扫描优先策略.md",
    "docs/design/G1_v2_多成键与少成键方向对比.md",
    "docs/design/G1_v2_成断键扫描方向选择可视化.html",
    "docs/design/G1_v2_方向无关成键锚点与双向扫描策略.md",
    "docs/design/G1_v2_补全实施总方案.md",
    "docs/design/PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md",
    "docs/design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md",
    "docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md",
}
STD |= {
    "docs/design/decisions/ADR-0003-G2-AB1接口登记.md",
    "docs/design/decisions/ADR-0004-G2-AB2接口登记.md",
    "docs/design/decisions/ADR-0005-G2-AB2执行前判定器与恢复修补.md",
    "docs/design/decisions/ADR-0018-RL纠错与动作支持研究计划.md",
    "docs/design/decisions/ADR-0019-IRC20k与RL2000分阶段扩量.md",
    "docs/design/decisions/ADR-0022-v4.1因果输入合同与特征v3.md",
    "docs/design/decisions/ADR-0023-V4.2原生重连数据与内部开发划分.md",
    "docs/design/decisions/ADR-0024-V4.2无界位移模型与战役训练协议.md",
    "docs/design/decisions/ADR-0027-V5.1参考架fallback与双分母评估.md",
}
STD.add("docs/infra/wsl_memory_governance.md")
STD |= {
    "docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md",
    "docs/plans/PES2TS_G2T真实战役执行方案_20261004.md",
    "docs/plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md",
    "docs/plans/PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md",
    "docs/plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md",
    "docs/plans/PES2TS_G2_PESG收紧算法迭代方案_20261003.md",
    "docs/plans/PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md",
    "docs/plans/PES2TS_G2_下一批算法调整与测试方案_20261003.md",
    "docs/plans/PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md",
    "docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md",
    "docs/plans/PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md",
    "docs/plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md",
    "docs/plans/PES2TS_RL自适应路径生长与IRC验证研究方案_v1_20261009.md",
}
STD |= {
    "docs/reports/PES2TS_Demo24_ACP首批计算_20261001.md",
    "docs/reports/PES2TS_Demo24_TS_IRC算法诊断_20261001.md",
    "docs/reports/PES2TS_Demo24_多坐标全批计算结果_20261002.md",
    "docs/reports/PES2TS_Demo24_端点与TS几何核查_20261001.md",
    "docs/reports/PES2TS_Demo24_第二轮修改与实测_20261002.md",
    "docs/reports/PES2TS_Direct_fit50训练表征分析_20261015.md",
    "docs/reports/PES2TS_Direct批次1-6实施记录_20261015.md",
    "docs/reports/PES2TS_GPU环境与磁盘预算执行记录_20261015.md",
    "docs/reports/PES2TS_PES-C-G架构审核与创新性可行性分析_20261004.md",
    "docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md",
    "docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md",
    "docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md",
    "docs/reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md",
    "docs/reports/PES2TS_V53MP_S2_5k规模报告_20261022.md",
    "docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md",
    "docs/reports/PES2TS_V53MP_S3_闭环评估报告_20261023.md",
    "docs/reports/PES2TS_V53MP_S4_20k_vs_50k_配对统计与C臂导出_20261027.md",
    "docs/reports/PES2TS_V53MP_S4_50k_单路正式训练报告_20261027.md",
}

# ---------------------------------------------------------------- AGGRESSIVE-only（50）
AGG = set()
# GATE_CONFIG_MUST_CHANGE（18，仅在 aggressive 且同 PR 改门禁配置+重渲染后删）
GCM = {
    "24H_EXECUTION_TIMELINE.md",
    "24H_FINAL_REPORT.md",
    "FAILURE_AND_REMAINING_WORK.md",
    "NEXT_DAY_DECISION_MEMO.md",
    "V52_PROGRESS_STOP_REPORT.md",
    "R3_RESULT_SUMMARY.md",
    "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md",
    "docs/plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md",
    "docs/plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md",
    "docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md",
    "docs/plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md",
    "docs/contracts/PES2TS_FullPathNavigationContract_v1.md",
    "docs/contracts/PES2TS_ReactionIntent_Contract_v1.md",
    "docs/contracts/PES2TS_GeometryPolicyContract_v1.md",
    "docs/contracts/PES2TS_双端TS教师与相容转移合同_v2.md",
    "docs/design/decisions/ADR-0021-无序双端拓扑与完整双侧TS教师主线.md",
    "docs/design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md",
    "docs/design/decisions/ADR-0029-V5.2+双门评估与封存边界.md",
}
AGG |= GCM
# 非 GCM 的 aggressive-only（32）
AGG |= {
    "docs/contracts/PES2TS_ACP字段映射_v1.md",
    "docs/contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md",
}
AGG.add("docs/design/PES2TS_CODE_MAP_v2_20261018.md")
AGG |= {
    "docs/design/decisions/ADR-0001-统一执行后端接缝.md",
    "docs/design/decisions/ADR-0002-计算后端统一经ACP执行.md",
    "docs/design/decisions/ADR-0006-ORCA原生约束优化与控制策略分离.md",
    "docs/design/decisions/ADR-0012-Classic图论驱动与xTB原生约束扫描.md",
    "docs/design/decisions/ADR-0013-G3并行IRC预训练与模型版本体系.md",
    "docs/design/decisions/ADR-0014-预测器产品归属与G3代码范围.md",
    "docs/design/decisions/ADR-0015-G2稳定底座与轻量RL研究入口.md",
    "docs/design/decisions/ADR-0016-ORCA驱动xTB硬约束优化试点.md",
    "docs/design/decisions/ADR-0017-统一残差决策与G2有限回退.md",
}
AGG |= {
    "docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md",
    "docs/plans/PES2TS_V5.1-R3_数据库特征损失三位一体修复方案_v1_20261020.md",
    "docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md",
    "docs/plans/PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md",
    "docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md",
    "docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md",
    "docs/plans/PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md",
    "docs/plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md",
    "docs/plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md",
    "docs/plans/V541_WP7_frozen_dev_eval_设计方案_v1.0_20261028.md",
}
AGG |= {
    "docs/reports/PES2TS_V53MP_规模训练收益报告_2k_to_50k_20261027.md",
    "docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html",
    "docs/reports/PES2TS_generation关键问题与根因分析_20261002.md",
    "docs/reports/PES2TS_v4.1阶段训练_修复输入语义与最小实验矩阵_最终报告_20261017.md",
    "docs/reports/PES2TS_v4阶段训练_最终报告与问题复核_20261017.md",
    "docs/reports/PES2TS_多坐标PES逐步实施方案_20261001.md",
    "docs/reports/PES2TS_多坐标同步第一轮实现与实测_20261002.md",
    "docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md",
    "docs/reports/PES_GENERATION_单端预测校正延续可行性分析_20261002.md",
    "docs/reports/PES_generation_多坐标约束学习_ORCA能力与数学方案_20261001.md",
}

# ---------------------------------------------------------------- MERGE_INTO（8，显式覆盖 DELETE/EXTRACT 判定）
MERGE = {
    "docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md":
        "docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md",
    "docs/plans/PES2TS_V5.3-MP_文档同步清单_v1_20261021.md":
        "docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md",
    "docs/plans/PES2TS_实施路线图.html":
        "docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md",
    "docs/plans/PES2TS_双项目开发与Demo方案.md":
        "docs/design/PES2TS_三模块与可视化产品架构_v1_20261005.md",
    "docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md":
        "docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md",
    "docs/plans/PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md":
        "docs/plans/PES2TS_G2T真实战役执行方案_20261004.md",
    "docs/agent_runs/PARALLEL_EXECUTION_PLAN.md":
        "docs/plans/PES2TS_完整开发流方案_整合版_v1.md",
    "docs/demo/PES2TS_Demo评估指标_v1.md":
        "docs/reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md",
}

# ---------------------------------------------------------------- Lead 裁决覆盖（CORE_SET 47 受管单元冲突）
# 2026-11-21 Lead 裁决：以下 5 份属核心集（CORE_SET C4/C5/C8/D1/D5），改判 KEEP，从各档删除集中移除。
OVERRIDE_KEEP = {
    "docs/contracts/PES2TS_contracts_v1.md",                 # CORE_SET C4 schema 总表
    "docs/contracts/PES2TS_ACP字段映射_v1.md",                # CORE_SET C5 ACP 接口映射
    "docs/contracts/PES2TS_数据规范与ACP对接优化方案.md",      # CORE_SET C8
    "docs/design/PES2TS_CODE_MAP_v2_20261018.md",            # CORE_SET D1 当前代码导航
    "docs/design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md",  # CORE_SET D5 + 宪法 :345 链接
}
assert OVERRIDE_KEEP.isdisjoint(KEEP)
assert all(_p in (SAFE | STD | AGG) for _p in OVERRIDE_KEEP), "override 必须命中删除集"
KEEP |= OVERRIDE_KEEP
for _s in (SAFE, STD, AGG):
    _s -= OVERRIDE_KEEP

QVERB = {
    "报告": "中", "运营记录": "中", "阶段计划": "中", "设计": "中",
    "合同": "高", "ADR": "高", "治理规范": "高", "快照": "无",
    "索引": "中", "产品文档": "中", "外部说明": "低", "其它": "低",
}
EPOCH_TARGET = {
    "KEEP": "", "DELETE": "",
}


def main():
    inv = list(csv.DictReader(io.open(BASE + "doc_inventory.csv", encoding="utf-8-sig", newline="")))
    disp = {r["path"]: r for r in csv.DictReader(
        io.open(BASE + "DISPOSITION.csv", encoding="utf-8-sig", newline=""))}
    cls = {r["path"]: r for r in csv.DictReader(
        io.open(BASE + "classification_mapping.csv", encoding="utf-8-sig", newline=""))}
    amap = {r["path"]: r for r in csv.DictReader(
        io.open(AMAP, encoding="utf-8-sig", newline=""))}
    tracked = {r["path"]: r["on_disk_flag"] for r in inv}

    all_paths = [r["path"] for r in inv]
    assert len(all_paths) == len(set(all_paths)) == 253

    # 一致性：集合互斥、单调、并集
    assert KEEP.isdisjoint(SAFE | STD | AGG)
    assert SAFE.isdisjoint(STD | AGG) and STD.isdisjoint(AGG)
    union = KEEP | SAFE | STD | AGG
    missing = [p for p in all_paths if p not in union]
    extra = [p for p in union if p not in all_paths]
    assert not missing, f"未覆盖: {missing}"
    assert not extra, f"多余: {extra}"
    assert len(KEEP | SAFE | STD | AGG) == 253

    rows = []
    for r in inv:
        p = r["path"]
        d = disp.get(p, {})
        c = cls.get(p, {})
        a = amap.get(p, {})
        gate = c.get("gate_locked", "").strip()
        if p in KEEP:
            decision, intensity, phase = "KEEP", "-", "D0"
            target = "常青层" if a.get("archive_entry") == "X0" else "档案树节点"
            reason = "核心保留：" + (d.get("reason", "现行/治理/证据必需")[:46])
            uv = "高"
        elif p in GCM:
            decision, intensity, phase = "GATE_CONFIG_MUST_CHANGE", "aggressive", "D3"
            target = f"{a.get('archive_entry','E??')} + 门禁配置 diff"
            reason = "门禁钉死；仅在 D3 同 PR 改配置并重渲染后删除，否则保留"
            uv = "高"
        elif p in SAFE:
            intensity, phase = "safe", "D1"
            decision = "MERGE_INTO" if p in MERGE else "DELETE"
            target = MERGE.get(p, (a.get("archive_entry", "") + "（重复副本，结论已存在于现行件）"))
            reason = d.get("reason", "") or "已被取代/逐字节副本，无独有结论"
            uv = QVERB.get(r["doc_type"], "低")
            if r["doc_type"] == "快照":
                uv = "无"
        elif p in STD:
            intensity, phase = "standard", "D2"
            decision = "MERGE_INTO" if p in MERGE else "ARCHIVE_EXTRACT_THEN_DELETE"
            target = MERGE.get(p, a.get("archive_entry", "E??"))
            reason = "过程性报告/一次性计划；结论抽入档案树后删除"
            uv = "中"
        else:  # AGG
            intensity, phase = "aggressive", "D3"
            decision = "MERGE_INTO" if p in MERGE else "ARCHIVE_EXTRACT_THEN_DELETE"
            target = MERGE.get(p, a.get("archive_entry", "E??"))
            reason = "仅在 aggressive 删；先抽结论入档案树"
            uv = QVERB.get(r["doc_type"], "中")
        # 抽取需求：非 KEEP 且非纯快照/纯零价值
        extract = "N"
        if decision in ("ARCHIVE_EXTRACT_THEN_DELETE", "GATE_CONFIG_MUST_CHANGE"):
            extract = "Y"
        rows.append({
            "path": p,
            "zone": ZONE_LABEL.get(r["zone"], r["zone"]),
            "doc_type": r["doc_type"],
            "current_action": d.get("action", ""),
            "decision": decision,
            "intensity": intensity,
            "reason": reason[:80],
            "unique_value": uv,
            "gate_locked": gate,
            "referrers": r["ref_count"],
            "recovery": "git blob" if tracked.get(p) == "1" else "无(未跟踪,需先归档)",
            "target": target,
            "phase": phase,
        })

    # 写入 CLEANUP_CANDIDATES.csv（字段顺序固定）
    fn = ["path", "zone", "doc_type", "current_action", "decision", "intensity",
          "reason", "unique_value", "gate_locked", "referrers", "recovery", "target", "phase"]
    with io.open(OUT + "CLEANUP_CANDIDATES.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(rows)

    # 写入 KEEP_SET.csv
    LAYER = {}

    def layer_of(p):
        if p in ("README.md", "AGENT.md", "docs/PES2TS_开发宪法_v1.md", "docs/README.md",
                 "docs/plans/PES2TS_文档修订台账.md", "docs/plans/PES2TS_阶段定义_G0-G3.md",
                 "docs/plans/PES2TS_完整开发流方案_整合版_v1.md",
                 "docs/plans/PES2TS_整体开发与发布方案_v2.md",
                 "docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md",
                 "docs/infra/doc_audit/20261121/README.md"):
            return "L0"
        if p.startswith("docs/contracts/") or p.startswith("docs/design/decisions/") \
                or p.startswith("docs/plans/PES2TS_V5") or p.startswith("docs/plans/PES2TS_V5-V7"):
            return "L1"
        if p.startswith("docs/reports/") or p.startswith("artifacts/"):
            return "L2"
        if p.startswith("docs/archive/") or p.endswith("README.md") and p.startswith("docs/"):
            return "L3"
        if p.startswith("docs/plans/") or p.startswith("docs/design/") or p.startswith("docs/infra/"):
            return "L1"
        return "L4"

    def duty_of(p):
        if p in ("README.md", "AGENT.md"):
            return "仓库入口/执行投影"
        if "宪法" in p:
            return "唯一规范来源 SSOT"
        if "台账" in p:
            return "文档修订台账（增量登记）"
        if "阶段定义" in p:
            return "G0–G4 阶段与边界唯一来源"
        if "完整开发流" in p:
            return "流程总纲"
        if "整体开发与发布" in p:
            return "项目级开发与发布纲领"
        if "G_V_MAPPING" in p:
            return "G–V 映射渲染件（禁止手改）"
        if p.endswith("docs/README.md"):
            return "文档索引"
        if p.startswith("docs/contracts/"):
            return "现行合同/接口"
        if p.startswith("docs/design/decisions/"):
            return "现行 ADR 决策记录"
        if p.startswith("docs/plans/PES2TS_V5.4"):
            return "当前战役预注册/执行方案"
        if p.startswith("docs/plans/PES2TS_V5.3-MP"):
            return "V5.3-MP 联合训练计划"
        if p.startswith("docs/plans/PES2TS_V5"):
            return "活跃研究计划"
        if p.startswith("docs/plans/PES2TS_训练测试集"):
            return "训练测试集划分说明"
        if p.startswith("docs/reports/"):
            return "只读执行证据（g_v_map evidence/终报）"
        if p.startswith("artifacts/"):
            return "模型卡（只读证据）"
        if p.startswith("docs/archive/"):
            return "快照 manifest（只读索引）"
        if "superseded/README" in p:
            return "已取代区索引"
        if "agent_runs/README" in p:
            return "运营记录区索引"
        if p.startswith("docs/infra/doc_audit/"):
            return "本次审计证据（只读）"
        if "规范化归档方案" in p:
            return "现行文档治理方案 v2"
        if p in ("data/README.md", "outputs/README.md", "scripts/README.md",
                 "scripts/archive/README.md", "examples/contracts_v1/README.md",
                 "examples/m1_first_plan/README.md", "examples/contracts_v1/path_viewer.html"):
            return "外部/目录说明"
        if p.startswith("requirements"):
            return "运行时依赖清单"
        if p == "CONTRIBUTING.md":
            return "贡献指南"
        if p == "LICENSE":
            return "许可"
        return "保留件"

    def freq_of(p, lay):
        if lay == "L0":
            return "事件驱动（规范变更即更新）"
        if lay == "L1":
            return "里程碑/战役结束"
        if lay == "L2":
            return "只读不改（新证据另立）"
        if lay == "L3":
            return "季度（快照新增时）"
        return "季度"

    def owner_of(p, lay):
        if "宪法" in p or p in ("README.md", "AGENT.md"):
            return "仓库所有者"
        if p.startswith("docs/reports/") or p.startswith("artifacts/"):
            return "科学负责人"
        if p.startswith("docs/contracts/") or p.startswith("docs/design/decisions/"):
            return "科学负责人+工程负责人"
        if p.startswith("docs/archive/"):
            return "档案管理员（Agent 会话）"
        if p.startswith("docs/infra/doc_audit/") or "规范化归档方案" in p:
            return "文档审计员/Agent 会话"
        if p.startswith("docs/plans/") or p.startswith("docs/design/"):
            return "工程负责人"
        return "Agent 会话"

    keep_rows = []
    for p in all_paths:
        if p not in KEEP:
            continue
        lay = layer_of(p)
        keep_rows.append({
            "path": p,
            "layer": lay,
            "duty": duty_of(p),
            "maintenance": freq_of(p, lay),
            "owner_role": owner_of(p, lay),
            "in_archive_tree": "是（节点/manifest）" if p.startswith("docs/archive/") else
                              ("是（常青层指针）" if lay == "L0" else "否（现行件）"),
        })
    with io.open(OUT + "KEEP_SET.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "layer", "duty", "maintenance",
                                          "owner_role", "in_archive_tree"])
        w.writeheader()
        w.writerows(keep_rows)

    # ---------------- LINK_FIX.csv：删除后必须同步修正的活跃引用 ----------------
    import os
    import re
    del_by_base = {}
    for r in rows:
        if r["decision"] in ("KEEP",):
            continue
        base = r["path"].split("/")[-1]
        del_by_base.setdefault(base, []).append(r)
    scan = ["README.md", "AGENT.md"] + [
        p for p in all_paths
        if p.startswith("docs/") and p.endswith(".md")
        and not p.startswith("docs/archive/") and p in KEEP
    ]
    linkfix = []
    for rp in scan:
        try:
            text = io.open(rp, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for ln, line in enumerate(text.splitlines(), 1):
            for base, cands in del_by_base.items():
                if base not in line:
                    continue
                # 只统计 markdown 硬链接目标（check_docs_links 会 error 的面）
                hard = re.search(r"\]\(([^)\s]*" + re.escape(base) + r"[^)\s]*)\)", line)
                if not hard:
                    continue
                r0 = cands[0]
                linkfix.append({
                    "referrer_path": rp,
                    "line_no": ln,
                    "context": line.strip()[:100],
                    "old_target": hard.group(1)[:160],
                    "new_target": r0["target"] if r0["target"] and not r0["target"].startswith("E") else "删除该行/改指档案树条目",
                    "action": "删除索引行" if rp == "docs/README.md" else
                              "改为档案树指针或删行",
                    "batch": r0["phase"],
                })
    with io.open(OUT + "LINK_FIX.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["referrer_path", "line_no", "context",
                                          "old_target", "new_target", "action", "batch"])
        w.writeheader()
        w.writerows(linkfix)

    # ---- 统计（供 CLEANUP_PLAN.md 使用）----
    dec = collections.Counter(r["decision"] for r in rows)
    inten = collections.Counter(r["intensity"] for r in rows)
    safe_n = inten["safe"]
    std_n = inten["safe"] + inten["standard"]
    agg_n = inten["safe"] + inten["standard"] + inten["aggressive"]
    print("总数", len(rows))
    print("decision:", dict(dec))
    print("intensity:", dict(inten))
    print("safe删", safe_n, "standard删", std_n, "aggressive删", agg_n, "KEEP", len(KEEP))
    print("GCM", len(GCM), "MERGE", len(MERGE), "抽取Y", sum(1 for r in rows if
          r["decision"] in ("ARCHIVE_EXTRACT_THEN_DELETE", "GATE_CONFIG_MUST_CHANGE")))
    print("KEEP层级:", dict(collections.Counter(r["layer"] for r in keep_rows)))
    print("phase:", dict(collections.Counter(r["phase"] for r in rows)))
    print("gate_locked in candidates:", sum(1 for r in rows if r["gate_locked"]))
    print("未跟踪待删:", [r["path"] for r in rows if r["recovery"].startswith("无") and r["decision"] != "KEEP"])
    print("单调校验: safe⊂standard⊂aggressive =", safe_n <= std_n <= agg_n)


if __name__ == "__main__":
    main()
