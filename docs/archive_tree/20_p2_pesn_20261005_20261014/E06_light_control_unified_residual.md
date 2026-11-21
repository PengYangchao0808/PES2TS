# E06 轻量控制 / 统一残差 / IRC20k-RL2000 扩量入口

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P2
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E06　**所属期**：P2（2026-10-07→10-14）　**关联 ADR**：ADR-0015/17/18/19　**负结果**：N06

---
> 条目标号：E06　时间范围：2026-10-07 → 2026-10-14　状态：Archived
> 关联 ADR：ADR-0015（G2 稳定底座与轻量 RL 研究入口，Accepted）、ADR-0017（统一残差决策与 G2 有限回退，Accepted）、ADR-0018（RL 纠错与动作支持研究计划，Proposed）、ADR-0019（IRC20k 与 RL2000 分阶段扩量，Proposed）
> 数据/代码/模型版本指针：`artifacts/pesn/pesn-control-*/model_card.md`（6 份控制模型卡）；`docs/archive/20261007_pre_light_policy/`、`20261007_pre_unified_residual_rl/`、`20261008_pre_scale20k_rl2k/`
> 证据根：`docs/plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md`、`docs/plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md`、`docs/plans/PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md`、根 `README.md` 旧段（离线 fake-ACP 包 A/B/C）

## 目标
在 G2 底座稳定的前提下，把 RL 研究接入现实规模：统一残差决策 + 动作条件 Q；同时冻结扩量路线（IRC20k / RL2000）。

## 核心思路
- 先接通真实 RL 再扩量；不做"小数据上炫技、大规模崩塌"的路线（ADR-0017 取代"步长分类优先"）。
- 一个共享编码器 + 坐标残差头 + 动作条件 Q 头；预算化 constrained-MDP 近似。
- 候选效率代理奖励与严格 TS 成功分离，不把代理指标当成功。
- 环境统一：G2 共用同一执行/质量/恢复/成本接缝；AB4 四点门固定在策略动作空间之外。

## 主要成果与量化指标
- 控制侧离线包 A/B/C 实现并 fixture-passed（六控制合同、G2 单步提取 + E_native 宏步环境、候选/合法性/选择策略、统一残差 + Q 头、离线 conservative-Q 训练器、`g2t-control-round` CLI；见根 README 旧段）。
- Demo24 真实 RL 闭环 round0 完成一轮（`docs/plans/PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md` 状态头）。
- 扩量路线文档：IRC20k 与 RL2000 下一阶段详细方案（10-08）、RL 下一轮代码修复训练与调整（10-08）、RL 自适应路径生长（10-09）。

## 关键负结果
- 真实 QC 采集、训练、冻结重执行、AB4 适配器（包 D）与任何策略收益**均未运行**（根 README 原文：`Real QC collection, training, frozen re-execution ... remain not run`）。
- 闭环只有 fixture 与 round0 级别证据；`implemented/fixture_passed` 不构成能力验收。
- 扩量方案均为 `Proposed`，IRC20k/RL2000 实际未启动。

## 证据指针
- `docs/plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md`（状态头 `当前实施方案`）
- `docs/design/decisions/ADR-0017-统一残差决策与G2有限回退.md`
- `artifacts/pesn/pesn-control-demo24-v0.1.0/model_card.md`（仅模型卡，非性能声明）
- `docs/archive/20261008_pre_scale20k_rl2k/README.md`

## 后续影响
- 统一残差/Q 头与 G2 接缝成为 E09 V5 冲刺与 E10 24H 战役的代码底座；真实 QC 闭环在 E12 的 S1–S3 才第一次大规模执行。

---

### 第 3 期 · V4.x 训练实验期（2026-10-15 → 2026-10-18）

一句话目标：在"无序双端拓扑 + 完整双侧 TS 教师"的新任务上，用最小实验矩阵找出 V4.0/V4.1 的真实基线，再用 10 小时窗口做 V4.2–V4.7 决定性战役——结果是一个必须原样保留的负结果。

核心思路：①先修输入语义（去掉标签派生输入）再谈模型改进；②三口径冻结评测（legacy_full / native_full / common_unchanged）在看任何预测前冻结；③四门冻结（MSE/zero <1.0、≥6/8 组优于零步、整体 cos ≥0.30、near-TS cos ≥0.40）不可事后调整；④区分"训练内选择集"与"历史 monitor8"，confirm/test/family holdout 完全不读；⑤失败原因优先查输入可辨识性与数据覆盖，而不是堆容量。

关键技术决策：ADR-0021（无序双端拓扑与完整双侧 TS 教师主线，Accepted）、ADR-0020（EGNN→BC→IQL→SAC 研究主线，后转 Historical）、ADR-0022（v4.1 因果输入合同与特征 v3）、ADR-0023（V4.2 原生重连数据与内部开发划分）、ADR-0024（V4.2 无界位移模型与战役训练协议）。V4.2–V4.7 实施合同与战役冻结 v1（Frozen，训练开始后不可改）。

主要成果与指标：v4/v4.1 终报（10-17）与 V4.2–V4.7 十小时战役（10-18）共 **23 个正式 run 全部 completed**；5 项新工程能力：原生重连数据（剔除 399 个插入节点、重建 50 组/100 分支/5,568 个原生动作）、无界 Cartesian 位移头（取消 0.1·tanh 上限）、`campaign_v1` 训练协议、三口径冻结评测、可恢复战役驱动器。V4.1 纯因果基线：A 臂 MSE/zero 1.0817（2/8 组优于零步）、B 1.0984（1/8）、C 1.0555（3/8），cos 0.1186/0.1910/0.1466，near-TS 0.2393/0.2713/0.3050——**四门全部未过**。V4.7 FINAL（fit50、三种子、native_full 主口径）三种子中位：MSE/zero **1.132**、优于零步 **0/8**、整体 cos **0.180**、near-TS cos **0.248**，四个指标全部低于冻结门，三种子无一人过门。

关键负结果：V4.7 判负，且全部筛选臂（D0–D3、H2、E1/E2、V1、W96/W128）均未过门；H1 因物理先验 `Δs_fit=0` 被预注册门判 `prior_not_supported`；V2 因接触边实现仅 0.43 steps/s 被按预注册顺序裁剪；40k 延长门判 `STAY_20K`；E2 出现 `MODEL_LOAD_FAILED`（过夜汇总跳过）。诊断结论：fit 侧 MSE 持续下降（E1 0.706、W128 0.681、V1 0.516）而 dev 侧升到 1.26–1.49、优于零步组数降为 0 → **跨反应泛化缺口**；fit45 内存在 **342 对近似相同输入但教师方向冲突**的样本 → **输入可辨识性缺陷**，单纯加大模型/延长训练无证据支持。

数据/代码/模型指针：`docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md`（Frozen）；native reconnect 审计、梯度基线审计、协议冻结、run records 均在 `outputs/`（不入库，报告内有路径）。当期原始文档 git 范围：10-15…10-17 文档（10-18 批量入库）；10-18 起 17 次提交（V4.2–V4.7 战役执行与报告）。
