# PES2TS 文档索引

> **状态**：Canonical（人工导航短索引；详细历史见档案树）
> **层级**：L0 治理 / 导航
> **取代关系**：取代 2026-11-21 之前的长索引（历史条目见 `docs/archive_tree/_INDEX.md`）
> **上游依赖**：ADR-0035、归档方案 v2.0、档案树
> **下游消费者**：全体读者、AI 会话
> **当前版本**：v2.0 @ 2026-11-22（短索引）

## 四个入口

1. **想了解项目怎么走到今天** → `docs/archive_tree/README.md`（7 期 / 17 条目，含每期核心思路、关键决策、成果与负结果）
2. **想知道当前该按什么执行** → 治理与现行文档（下表）
3. **想查历史证据/负结果** → `docs/archive_tree/90_evidence/negative_results.md` 与 `docs/archive_tree/_INDEX.md`
4. **想知道文档怎么维护** → `docs/infra/doc_cleanup/20261121/MAINTENANCE.md` + `ADR-0035`

## 治理与索引

| 文档 | 说明 |
|---|---|
| `docs/PES2TS_开发宪法_v1.md` | 规范根（SSOT，内容 v2.9） |
| `AGENT.md` | 宪法执行投影 |
| `docs/plans/PES2TS_阶段定义_G0-G3.md` | 阶段与边界唯一来源 |
| `docs/plans/PES2TS_完整开发流方案_整合版_v1.md` | 流程总纲 |
| `docs/plans/PES2TS_文档修订台账.md` | 变更流水（含退役/档案条目登记） |
| `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md` | 状态机/命名/元数据/归档规则 |
| `docs/infra/PES2TS_开发档案树与文档清理方案_v1_20261121.md` | 本次清理方案与执行记录 |
| `docs/design/decisions/ADR-0035-文档生命周期与退役政策.md` | 退役政策 |

## 现行合同

| 文档 | 说明 |
|---|---|
| `docs/contracts/PES2TS_MoveProcessContract_v1.md` | 唯一任务合同（MOVE/PROCESS/No-Move/TS） |
| `docs/contracts/PES2TS_ReactionIntent_Contract_v2.md` | 反应意图输入合同 |
| `docs/contracts/PES2TS_GeometryPolicyContract_v2.md` | 几何策略合同 |
| `docs/contracts/PES2TS_contracts_v1.md` | 数据/接口契约总表 |
| `docs/contracts/PES2TS_ACP字段映射_v1.md` | ACP 字段映射 |
| `docs/contracts/PES2TS_数据规范与ACP对接优化方案.md` | 数据规范与对接 |

## 现行计划与设计

| 文档 | 说明 |
|---|---|
| `docs/plans/PES2TS_V5-V7_全阶段研发与Benchmark方案_v2.0_20261121.md` | **路线重规划（Active-Proposed）**：V5 离线架构 Benchmark / V6 真实闭环校准 / V7 RL 成本优化；12 里程碑 + Go/No-Go（V5 章已由下一行 v2.1 细化取代） |
| `docs/plans/PES2TS_V5_深度模型架构Benchmark方案_v2.1_20261122.md` | **V5 详细实验方案（Active-Proposed）**：14 架构/五大范式、V5.5 轨迹级数据冻结、四测试集 OOD、三层公平对照、122 次训练预算 |
| `docs/plans/PES2TS_V5后续阶段训练大纲_v1_20261123.md` | **V5 训练执行大纲（Active）**：工作盘点、V5.5 数据地基→V5.8 训练矩阵、122 run 预算、里程碑 Go/No-Go、WSL 稳定性前置与文档同步关系 |
| `docs/plans/PES2TS_本地数据资产整合与数据工程方案_v1_20261123.md` | **本地数据资产与数据工程（Active-Proposed）**：五类资产账本（Windows 139.5 GB + WSL 182 GB）、五层划分与能量/梯度通道登记、xTB 单点数据集三层接入工程（reader/index + 帧对齐视图 + G1 消费者）、V5/V6 落位矩阵、M1–M11 裁决与 D0–D5 执行路线 |
| `docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md` | V5 反应意图条件化主线（V5.5–V5.8 规划由 v2.0/v2.1 修订） |
| `docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md` | 最近已执行战役方案（状态见文件头） |
| `docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md` | 最近战役 WP 清单 |
| `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | V5.3-MP 战役（已执行） |
| `docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md` | V5.4.2 战役（已执行） |
| `docs/plans/PES2TS_整体开发与发布方案_v2.md` | 项目级发布主方案 |
| `docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md` | 数据划分与来源 |
| `docs/design/PES2TS_CODE_MAP_v3_20261123.md` | **代码地图 v3（Active-Reference，新会话先读）**：五功能区（`pes2ts_core`/`pes2ts_data`/`pes2ts_v5`/`infra`/`archive`）、唯一训练引擎与 runner CLI、任务→入口表、`pes2ts_core/pesn/**` FROZEN、v2 planned 缺口收敛为 NOT_IMPLEMENTED |
| `docs/design/PES2TS_CODE_MAP_v2_20261018.md` | 代码地图 v2（历史版；现行见 v3，v2 的 planned 区块已由 v3 收敛登记） |
| `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md` | G–V 映射（生成物，勿手改） |
| `docs/design/decisions/` | ADR-0001…0036（全部保留；0036 = 五功能区与单一训练引擎） |

## 证据与运维

| 文档 | 说明 |
|---|---|
| `docs/archive_tree/` | 7 期 / 17 条目开发档案（历史结论入口） |
| `docs/agent_runs/README.md` | 战役编排/交接说明 |
| `docs/infra/PES2TS_WSL稳定运行与内存治理方案_v1_20261122.md` | **WSL 稳定运行与内存治理（Active）**：四层防线、阈值、SOP、preflight/回收/受限运行脚本 |
| `docs/infra/doc_audit/20261121/README.md` | 2026-11-21 全量文档审计证据 |
| `docs/reports/PES2TS_代码情况报告_v2_20261123.md` | **当前代码情况报告 v2（Active-Reference，重组后口径）**：五区账本（1,685 文件/1,027 `.py`/452,175 行）、四层契约实现度、健康度（lint/长函数/未接线/测试分层/门禁）、R1–R6 复审、owner 决策与工程待办、调整指南（改什么→动哪里→跑什么）、复现附录；取代 v1 |
| `docs/reports/PES2TS_代码探查报告_v1_20261123.md` | 代码清点与归档依据 v1（历史：重组前 991 文件/443,792 行清点、40 个疑似遗留模块分类、R1–R6 审计、四类归档建议）；现行口径见 v2 |
| `docs/reports/PES2TS_代码分区重构执行记录_v1_20261123.md` | **五功能区重构执行记录（2026-11-23，Evidence-ReadOnly）**：提交 `ccd31f2` 的 575 重命名/0 删除、逐树迁移计数、四个合同与五道硬门、门禁实跑、legacy POSIX/依赖限制、G–V 语义待决与 GPU/WSL 全量 pytest 待补 |
| `docs/infra/doc_cleanup/20261121/README.md` | 本次清理证据与维护机制 |

## 历史

- 档案树索引：`docs/archive_tree/_INDEX.md`
- 已退役文档：从 `pre-doc-purge-20261121` tag 或 `git log --diff-filter=D` 恢复；退役流水见 `docs/infra/doc_cleanup/20261121/RETIRE_LOG.md`
