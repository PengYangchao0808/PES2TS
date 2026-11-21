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
| `docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md` | 当前研究计划（V5/V6/V7） |
| `docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md` | 最近已执行战役方案（状态见文件头） |
| `docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md` | 最近战役 WP 清单 |
| `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | V5.3-MP 战役（已执行） |
| `docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md` | V5.4.2 战役（已执行） |
| `docs/plans/PES2TS_整体开发与发布方案_v2.md` | 项目级发布主方案 |
| `docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md` | 数据划分与来源 |
| `docs/design/PES2TS_CODE_MAP_v2_20261018.md` | 代码地图（新会话先读） |
| `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md` | G–V 映射（生成物，勿手改） |
| `docs/design/decisions/` | ADR-0001…0035（全部保留） |

## 证据与运维

| 文档 | 说明 |
|---|---|
| `docs/archive_tree/` | 7 期 / 17 条目开发档案（历史结论入口） |
| `docs/agent_runs/README.md` | 战役编排/交接说明 |
| `docs/infra/wsl_memory_governance.md` | WSL 内存治理 |
| `docs/infra/doc_audit/20261121/README.md` | 2026-11-21 全量文档审计证据 |
| `docs/infra/doc_cleanup/20261121/README.md` | 本次清理证据与维护机制 |

## 历史

- 档案树索引：`docs/archive_tree/_INDEX.md`
- 已退役文档：从 `pre-doc-purge-20261121` tag 或 `git log --diff-filter=D` 恢复；退役流水见 `docs/infra/doc_cleanup/20261121/RETIRE_LOG.md`
