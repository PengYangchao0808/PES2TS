# E14 V5.4.2 Backbone-50K（no_backbone_winner）

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P6
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E14　**所属期**：P6（2026-10-29→10-31）　**关联 ADR**：ADR-0033　**负结果**：N18

---
> 条目标号：E14　时间范围：2026-10-29 → 2026-10-31　状态：Archived（Executed-Frozen；计划状态头滞后 A01/P0）
> 关联 ADR：ADR-0033（Accepted，v1.3.3）；合同 MoveProcessContract v1
> 数据/代码/模型版本指针：`config/campaigns/pes2ts_v542_backbone_50k.yaml`（`0.1.2-wp0`，schema `pes2ts_v542_campaign_v1`）；registry `V5.4.2`（status executed，`plan_revision 2026-10-30.3`）
> 证据根：`docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md`（内容 v1.2）、`docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md`、`docs/design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md`

## 目标
在统一单帧空间 Backbone 接口（`z[G,64]` + `edge_msg[E,48]`）下，比较 A1–A5 候选与 matched A0，判断是否有候选能在三个 seed 上同时不劣化 MOVE 并带来 PROCESS 收益。

## 核心思路
- 接口冻结：唯一共享 `DirectMoveDecoder`（仅 `w_v` 49 参数），PROCESS 头原样复用（2,178 参数），A0 是 legacy Star 子类 adapter + hook + 显式映射审计。
- A0 默认重训 matched ×3（不实现 shadow-run/reuse 分支）；V541 E0×3 只作 parity/历史参照。
- 预算用同环境 40+40 profile 实测签认，core ratio ≤2.0。
- H2 逐 seed 对照 matched A0 与 R3；H3 只在 H2 无 breach 的候选上做 Holm 校正的 10,000 draws 配对 bootstrap。
- 无 H3 合格候选就走 `zero_h3_qualified_candidates_a0_retained`，不产生"最低分胜出"。

## 关键决策
- ADR-0033（Accepted）：Backbone 接口/迁移不变量/parity 证书/预注册记录模板全部冻结；默认重训 matched A0×3。
- WP4 diagnostic ruling：A0=`control_not_screened`、A4=`eliminated/gate_skipped`（3.849742% < 5.0%），阈值与预算数值不变（协议修正记录 `V542_WP4_DIAG_20261030`）。

## 主要成果与量化指标
- 执行账本全部实跑：smoke 200（6/6 done）、profile 8 backbones、diagnostic 2000、A0 sanity band PASS（seed 均值 A0 AUPRC 0.29670 vs E0 0.28858，+0.00812 ≤0.01；q_corr −0.00045；RMSE/zero −0.199%）、T0/T1/T3/T2 全部 done、**N_core=17**。
- 内层 inner-val：A2_S17 0.3275 / A2_S29 0.3213 / A2_S43 0.3209（显著优于 A0 ~0.67）；A3_S17 0.6476 / S29 0.6506 / S43 0.6713；A5_S17 0.7079；A1_S17 0.6924。
- parity 证书 6/6 资产：CPU bitwise 0.0、GPU 最大 9.5367e-7、frozen evaluator 重算 delta 0.0。

## 关键负结果
- **战役结论 `no_backbone_winner`：H3 eligible = []，A0 留任，不产生 Backbone\*。** 该结论不得写成"A0 胜出"。
- **A2/A3 的 seed 43 触发 per-seed H2 breach**（matched_a0.sign_abs_drop、fixed_coverage_direction_rel_degradation）→ 记 `scientific_failure`，不晋级、不得以 PROCESS 收益豁免。
- A1 未进 top2（seed17 screen top2=[A2,A3]）；A4 被冻结的 5% 学习筛淘汰；A5 完成 full-dose 但 seed17 未过 screen，未获补种。
- 计划状态头滞后：L3 `Active-Proposed`（WP1–WP7 待执行）与 README/report 的 "WP7 EXECUTED（2026-10-31）" 直接冲突（A01/P0）——**不得据计划状态头认为未执行**。
- `V5.4 剩余架构研究工作方案 v1.0` 全库 0 引用、未正式登记（A13/P2），本期不纳入执行。

## 限制与未解问题
- 候选侧 MOVE sign/direction 的 seed 敏感性说明 50K 数据下 MOVE 结构搜索的统计功效有限；PROCESS 收益真实但不足以豁免 MOVE 门。
- confirm 未读、QC=0、dev 未访问；无生产资格声明。

## 证据指针
- `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md`（§0 结论冻结写法、§2 账本、§3 指标表、§4 H2/H3 判定）
- `docs/design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md`
- `docs/README.md` 2026-10-29 节（WP7 执行事实）

## 后续影响
- E15 因此改问"不换 backbone、换 decoder（MOVE/PROCESS 残差轨道）能否同时提升"——并首次采用双轨冠军 + BMP 集成验证。
