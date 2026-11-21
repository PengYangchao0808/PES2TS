# E03 PES-N 立项与 SSOT 重构（宪法/阶段定义/PES-N 内核）

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P2
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E03　**所属期**：P2（2026-10-05→10-07）　**关联 ADR**：ADR-0010　**负结果**：N03

---
> 条目标号：E03　时间范围：2026-10-05 → 2026-10-07　状态：Archived
> 阶段：PES-N 内核（规划） × G2/G3 阶段重定义　关联 ADR：ADR-0010（Accepted，2026-10-05）、ADR-0007/0008/0009（Historical，被 ADR-0010 取代）
> 数据/代码/模型版本指针：`docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md`（内容 v2.5，Superseded 后被 V5–V7 取代）；宪法内容 v2.x（快照 `20261007_pre_light_policy` 为 v2.3）
> 证据根：`docs/PES2TS_开发宪法_v1.md`、`docs/plans/PES2TS_阶段定义_G0-G3.md`、`docs/archive/20261005_pre_pesn/`、`docs/README.md` 2026-10-05 节

## 目标
把项目从"G 阶段线性工程"重定义为"PES-N 反应状态价值内核（ranking + navigation）+ Classic 兜底"，并确立 SSOT 文档治理。

## 核心思路
- 内核抽象：`ranking` 评估是否值得 `STOP_OPTTS`；`navigation` 比较 STOP / 图定义 MOVE / 验证过的 fallback。
- 单一规范来源：宪法（SSOT）→ AGENT（执行投影）→ 阶段定义（唯一阶段来源）→ 方案/合同（只引用）。
- 三层效力：红线 R1–R6 不可违背 / 工程政策 P1–P5 可时限豁免 / 接口规范 I-* 只以新版本变更。
- G3.P 教师预训练可与 G2 并行（不等待 G2 调教/出口）。
- 研究能力四级状态：implemented / fixture_passed / smoke_passed / scientifically_validated（后续 24H 战役正式成词）。

## 关键决策
- ADR-0010（Accepted，2026-10-05）：PES-N 反应状态价值与 STOP/MOVE 统一决策；**取代 ADR-0007（统一价值模型）、ADR-0008（拓扑约束距离/位移联合）、ADR-0009（G 阶段统一模型框架）三条历史提案**。
- 宪法 v2.x 建立（`docs/PES2TS_开发宪法_v1.md`，当前 v2.9 @ 10-21）；阶段定义 v4.2（快照 20261007）确立 G0–G4 主干 + C/R/X 工作流。

## 主要成果与量化指标
- 10-05 单日 49 次提交（全项目单日峰值），PES-N 方案、ADR-0010、阶段定义、开发宪法、产品架构同日成体系落盘。
- `docs/archive/20261005_pre_pesn/` 快照：PES-N 产品归属/ADR-0014 前的入口状态，含 manifest 与 revision_check。
- 阶段定义明确"S0–S8/demo 阶段表降为历史映射"，G0–G4 成为唯一阶段口径。

## 关键负结果
- **三条历史提案被否**：ADR-0007/0008/0009 价值模型方向作废（状态头 `历史提案，当前方向已由 ADR-0010/PES-N 计划取代`）。这是研究方向的负结果，不是文档错误。
- PES-N v2 内核"仍待构建与评估"；方案中 STOP/MOVE/fallback 语义在 10-21 被 ADR-0030/0031 两代修订（当时不可视为最终）。
- 根 README 明确："这是规划修订，不是实现或 QC 验证声明"。

## 限制与未解问题
- G2 出口未闭合时 PES-N 无法验收（依赖链写明 G3.0 需 G2 闭环）。
- 三模块产品架构（ADR-0011）与预测器归属（ADR-0014）在本条目仅为并行产出，详见 E05。

## 证据指针
- `docs/design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md`（状态行 L3）
- `docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md`（状态头 `Superseded / Historical`）
- `docs/archive/20261005_pre_pesn/README.md`
- `docs/PES2TS_开发宪法_v1.md` L3–L16（效力层级与增量登记）

## 后续影响
- ADR-0010 的 STOP/MOVE 语义演化为：ADR-0021（无序双端教师，E07）→ ADR-0030（撤回 TS 邻域 Stop，E11）→ ADR-0031（MOVE–PROCESS，E12）。读 E03 时必须按"当时判定"理解，不得直接套用 10-21 后语义。
