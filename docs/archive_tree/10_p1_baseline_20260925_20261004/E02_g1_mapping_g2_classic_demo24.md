# E02 G1 映射 + G2 Classic/Demo24 与 24 例战役准备

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P1
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E02　**所属期**：P1（2026-09-30→10-04）　**关联 ADR**：ADR-0002~0005　**负结果**：N02

---
> 条目标号：E02　时间范围：2026-09-30 → 2026-10-04　状态：Archived
> 关联 ADR：ADR-0002（计算后端统一经 ACP，Accepted）、ADR-0003/0004/0005（G2-AB1/AB2 接口登记/判定器/恢复修补，Accepted）
> 数据/代码/模型版本指针：G2.T 合同字段字典 `docs/contracts/PES2TS_G2T_战役合同字段字典_v1.md`；扫描策略选择器设计 `docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`
> 证据根：`docs/reports/PES2TS_Demo24_*.md`（6 份，10-01/10-02）、`docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md`、`docs/plans/PES2TS_G2T真实战役执行方案_20261004.md`、`docs/reports/PES2TS_开发阶段性报告_20260930.md`

## 目标
让 Classic 主链第一次可审计地跑通 24 例 Demo：多坐标全批计算 → 四类终态（rejected_typed / partial_prefix / complete_path / validated）→ 失败与成本账本。

## 核心思路
- 24 例 Demo 作为"可审计终态"的最小可信战场，而不是成功率战场。
- ACP 统一执行：所有后端（XTB_PATH / ORCA scan/constraints/NEB / continuation）经 ACP，本地执行接缝退役（ADR-0002）。
- 类型化拒绝（typed rejection）与质量分层（完整路径/部分前缀/拒绝）分开记账。
- 多坐标 PES 学习路线作为生成能力候选，先出数学方案再谈实现。

## 关键决策
- ADR-0002（Accepted，2026-10-03）：统一 ACP 后端，ADR-0001 的执行本地化条款被部分取代；实施按"X1′–X5′"（`docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md`）推进。
- ADR-0003/0004/0005：G2-AB1/AB2 接口登记、执行前判定器 v2 与三项恢复/适配修补在 S2 真实执行前登记并锁定。

## 主要成果与量化指标
- Demo24 系列 9 份报告落盘：首批 ACP 计算、端点与 TS 几何核查、TS/IRC 算法诊断、多坐标全批计算、第二轮修改与实测、多键同步一维扫描问题报告、generation 根因分析、对话/任务总结。
- G2.T 战役合同字段字典 v1（10-05 前）：四类终态、成本账本字段、typed rejection 词表。
- 24 例真实战役方案（`docs/plans/PES2TS_G2T真实战役执行方案_20261004.md`）冻结 S1–S5 与 D1–D5 交付口径（**执行结果归 E04**）。

## 关键负结果
- 多键同步一维扫描存在缺陷清单与根因（10-01 报告）：24 例在第二轮修改后仍需第三轮，说明"一维扫描 + 单坐标"不足。
- Demo24 的 9 份报告中，端到端 validated=0 是当时已知事实（G2 退出条件要求 ≥1 条非参考种子严格目标 TS，**当期未达成**）。
- 本地 xTB runner 删除与全后端 ACP 化未完成（ADR-0002 实施注记，10-03 时点）。

## 限制与未解问题
- G2/G2.T 仍是"进行中（round0–round3）"；G2 出口四项条件未闭合。
- 多坐标 PES 学习仅数学方案（`docs/reports/PES_generation_多坐标约束学习_ORCA能力与数学方案_20261001.md`），未实现模型。

## 证据指针
- `docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md`
- `docs/reports/PES2TS_generation关键问题与根因分析_20261002.md`
- `docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`
- `docs/contracts/PES2TS_G2T_战役合同字段字典_v1.md`

## 后续影响
- 24 例真实战役于 2026-10-05 执行 → E04；四类终态词汇沿用至 E08/E13/E14/E15 的門判定。

---

### 第 2 期 · PES-N 立项期（2026-10-05 → 2026-10-14）

一句话目标：把项目从"G 阶段工程流水线"重定义为"PES-N 反应状态价值内核 + Classic 兜底"，并在同一周完成 G2 收口、G3.P 教师预训练立项、产品架构划分与轻量 RL 研究入口。

核心思路：①先定内核与决策语义（STOP/MOVE/fallback），再排训练；②教师预训练（G3.P）可与 G2 并行，不等 G2 调教；③产品架构三模块（共享内核 / Classic 控制 / 预测器），代码归属写进 ADR；④轻量路线：先在当前反应规模接通真实 RL，再扩量；⑤SSOT 治理：宪法为规范根、AGENT 为执行投影、阶段定义为唯一阶段来源。

关键技术决策：ADR-0010（PES-N 反应状态价值与 STOP-MOVE 统一决策，Accepted，10-05）、ADR-0011（三模块产品架构，Accepted）、ADR-0012（Classic 图论驱动与 xTB 原生约束扫描，Accepted）、ADR-0013（G3 并行 IRC 预训练与模型版本体系，Accepted）、ADR-0014（预测器产品归属与 G3 代码范围，Accepted）、ADR-0015（G2 稳定底座与轻量 RL 入口，Accepted）、ADR-0016（ORCA 驱动 xTB 硬约束优化，Accepted）、ADR-0017（统一残差决策与 G2 有限回退，Accepted）、ADR-0018（RL 纠错与动作支持研究计划，Proposed）、ADR-0019（IRC20k 与 RL2000 分阶段扩量，Proposed）。ADR-0007/0008/0009 三条价值模型路线被 ADR-0010 取代（历史提案）。

主要成果与指标：G2.T 24 例真实战役执行（2026-10-05，S1–S5 端到端）：24 个 typed 终态、443 条成本账本、N_gradient 总计 26,676；四类分账 2 rejected_typed / 19 partial_prefix / 3 complete_path / 0 validated，S4 全员 24/24 `blocked_on_human_review`（不是"跳过"）。AB3 原生约束扫描 24 例对比（2026-10-07）：13 complete_path / 9 partial_prefix / 2 rejected_typed（旧链 3/19/2），中位配对 scan 阶段加速约 61×；validated 仍为 0，生产默认未切换。G3.P/P1 立项：研究模型 `pesn-irc-v0.1.0`、教师合同 v1、B0–B7 模块划分。7 个 archive 快照时点（10-05/06/07/08）全部落在本期。

关键负结果：G2 四类终态中 validated=0、S4 无任何案例通过人工化学复核（24/24 blocked）；AB3 原生引擎只是显式分支，**不改变生产默认、不构成 TS 验证声明**；AB4 ORCA-xTB 四点试点设计完成但真实试点"尚未实施"（ADR-0016 状态）；G3.P 第一批的 STOP 概率、成本、MOVE-Q、生产控制全部 disabled（不可宣称能力）；PES-N 内核与 G3/G4 学习能力"仍待构建与评估"（根 README 原文）。

数据/代码/模型指针：`docs/contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md`；`artifacts/pesn/pesn-irc-v0.1.0-rc.1/model_card.md`；`config/registry/g_v_map.yaml` 前身记录在 E09 才系统化。当期原始文档 git 范围：10-05（49 提交）+ 10-06…10-14 文档（于 10-18 批量入库）；`docs/archive/20261005_pre_pesn`、`20261006_pre_g3p`、`20261007_pre_light_policy`、`20261007_pre_orca_xtb_pilot`、`20261007_pre_predictor_scope`、`20261007_pre_unified_residual_rl`、`20261008_pre_scale20k_rl2k`。
