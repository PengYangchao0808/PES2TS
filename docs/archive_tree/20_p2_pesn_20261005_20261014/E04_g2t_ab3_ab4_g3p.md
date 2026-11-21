# E04 G2.T 24 例真实战役 / AB3 / AB4 / G3.P 立项

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P2
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E04　**所属期**：P2（2026-10-05→10-08）　**关联 ADR**：ADR-0012/13/16　**负结果**：N04,N05

---
> 条目标号：E04　时间范围：2026-10-05 → 2026-10-08　状态：Archived
> 关联 ADR：ADR-0006（ORCA 原生约束优化，Proposed）、ADR-0012（Classic 图论驱动与 xTB 原生约束扫描，Accepted）、ADR-0013（G3 并行 IRC 预训练与模型版本体系，Accepted）、ADR-0016（ORCA 驱动 xTB 硬约束优化试点，Accepted）
> 数据/代码/模型版本指针：`docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md`（内容 v1.2）；`artifacts/pesn/pesn-irc-v0.1.0-rc.1/model_card.md`
> 证据根：`docs/plans/PES2TS_G2T真实战役执行方案_20261004.md`、`docs/plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md`、`docs/plans/PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md`、根 `README.md` 2026-10-05/10-07 段

## 目标
在 24 例上第一次取得可审计终态与成本账本（G2.T 第一轮），并用原生约束扫描（AB3）验证"图论定义约束 + xTB 原生优化"是否优于旧链。

## 核心思路
- 24 例 Demo 的核心产出是**失败分布与成本反馈**，不是成功率。
- 扫描加速从"减少扫描点数、原生约束优化"入手，而不是加算力。
- G3.P 教师预训练独立立项（不等 G2），教师/标签/分裂先于训练。
- 每个新能力必须有独立证据验收，不能因"训练完成"自动成为生产默认。

## 关键决策
- ADR-0012：Classic 图论驱动 + xTB 原生约束扫描（Accepted，10-06）。
- ADR-0013：G3 并行 IRC 预训练与模型版本体系（Accepted，10-06）。
- ADR-0016：ORCA 驱动 xTB 硬约束优化试点（Accepted，10-07；实施/真实能力待验收）。
- ADR-0006：ORCA 原生约束优化与控制策略分离（Proposed，10-04/10-05）。

## 主要成果与量化指标
- **G2.T 第一轮（2026-10-05，S1–S5 端到端）**：24 个 typed 终态；443 条成本账本；N_gradient 合计 26,676；四类分账 **2 rejected_typed / 19 partial_prefix / 3 complete_path / 0 validated**；S4 **24/24 `blocked_on_human_review`**（设计内门，不是跳过）。
- **AB3 原生 24 例（2026-10-07）**：13 complete_path / 9 partial_prefix / 2 rejected_typed（旧链 3/19/2）；中位配对 **scan 阶段加速约 61×**；成本/报告缺陷在对比前修复（AB2 独立复核 `outputs/reviews/G2_real_campaign_review_20261006.md`）。
- G3.P/P1 立项文档齐全：教师数据与模型包合同 v1、`pesn-irc-v0.1.0` 版本体系、B0–B7 模块划分（代码归属 E05）。

## 关键负结果
- **validated=0**：两轮 24 例都没有任何案例通过严格 TS 验证；AB3 的"13 complete_path"不等于化学成功（终态=自动流程闭环）。
- **原生引擎未成为生产默认**：AB3 引擎是显式 `g2t-native-run` 分支，生产默认与 TS 验证声明均未改变（发布门未过）。
- AB4 四点试点仅设计完成：工作流、回退与真实试点"尚未实施"（方案状态头）。
- G3.P 首包 STOP 概率/成本/MOVE-Q/生产控制 disabled；数据/权重/硬件/评测均未验收。

## 限制与未解问题
- 24 例的人工化学复核门（S4）在整期中都未通过，直接导致 validated 长期为 0。
- AB3 成本对比只覆盖 scan 阶段加速，不含下游精修/验证成本，不能外推为总成本收益。

## 证据指针
- 根 `README.md` 2026-10-05 段（24 例数字）、2026-10-07 段（AB3 对比与 61×）
- `docs/plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md`（状态头：24 例终态）
- `docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md`（状态头）
- `docs/contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md`

## 后续影响
- validated=0 与 S4 阻塞延续到 E07/E08 的评测协议设计；G3.P 进入 E05/E06/E07 的教师与训练主线。
