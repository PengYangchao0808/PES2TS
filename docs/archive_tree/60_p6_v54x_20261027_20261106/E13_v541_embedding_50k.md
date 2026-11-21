# E13 V5.4.1 Embedding-50K 执行（E0 留任）

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P6
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E13　**所属期**：P6（2026-10-27→10-29）　**关联 ADR**：ADR-0032　**负结果**：N17

---
> 条目标号：E13　时间范围：2026-10-27 → 2026-10-29　状态：Archived（Executed-Frozen）
> 关联 ADR：ADR-0032（Accepted，10-27）；任务合同 MoveProcessContract v1
> 数据/代码/模型版本指针：`config/campaigns/pes2ts_v541_embed_50k.yaml`（schema `pes2ts_v541_campaign_v1`）；冻结计划 `plan/runplan.pin.json`（code_commit `1d8b7972…`、common_protocol `2b9ce1ed…`、config `464c1f6f…`）；证据根 `outputs/pesn/v541_embed_50k/`
> 证据根：`docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md`（内容 v1.2）、`docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md`、`docs/design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md`

## 目标
固定 50K 数据、37,900 updates、MOVE+PROCESS 任务与三层 StarEGNNV3 core，只替换 embedding 残差接口，判断是否有 embedding 候选能在 MOVE 上确证优于 matched E0。

## 核心思路
- 单变量替换：任务语义、合同、评估口径、剂量全部冻结，只动 embedding。
- 预注册治理先行：独立 schema/validator（12 项检查）+ 阈值签认 + 预算签认 + 三类 run（nominal/conditional/diagnostic）分离。
- 先修 pair evaluator 再重评（T0-R matched E0×3 作为同 seed 对照基线）。
- H2 硬门：PROCESS 增益不得豁免 MOVE 坍缩；无候选时不补 seed、不编造 CI。

## 关键决策
- ADR-0032（Accepted）：只扩展 Embedding 残差接口，objective 仅 `unified_move_process`，不暴露三臂矩阵；未知 embedding/非法 forward key/confirm 授权均 typed rejection。
- 2026-10-27 用户指令签认 §11.2 阈值为 frozen（hard_ratio 2.0、GPU parity 1e-6/1e-5、CI 10,000 次、rollout 96 曲线）。

## 主要成果与量化指标
- WP0–WP7 全部完成；6 次正式训练（T0-R E0×3 + E1/E2/E3 seed17），合计 GPU 墙钟 27,943s ≈ 7.76h。
- E0 三 seed 冻结 dev：endpoint AUPRC **0.290702**、q corr **0.397494**、MOVE RMSE/zero **0.910039**、dead 0.095771、ordering full 0.751 / cross-TS 0.856。
- 候选（单 seed inner-val，对照 matched E0）：E1 ΔAUPRC +0.07874/Δq_corr +0.10188；E2 +0.05162/+0.10544；E3 +0.46004/+0.45043。
- no-physics rollout 1,152 episodes：reference/autonomous 双口径分开，endpoint arrival 两口径均 0.005208、TS candidate 0.055556、QC=0、非关键原子位移 0。
- 全仓测试 2026-10-29：pytest 4,954 passed / 1 skipped / 11 deselected；`docs_consistency --strict` CLEAN。

## 关键负结果
- **E1/E2 因 H2 MOVE sign guard 失败（sign_drop_abs 0.03333 > 0.01）、E3 因 sign+dead 双双失败 → 判 `scientific_failure`**；三者 inner-val `provisional_H3=true` 但不得豁免硬门。
- **E4 因 core-time ratio 3.6455 > 2.0 被 `gate_skipped`**（成本门，不是资源失败）；E5 记 `parent_E4_cost_gate_failed`；T4 记 `no_new_embedding_gain`。
- `n_selected=0` → T3 的 10 个条件 run 全部未激活、0 次补 seed；无候选三 seed CI（`bootstrap.executed=false`，不编造 p 值）。
- 容量/深度混杂未闭合（T4 skip 时已声明）；**确认结论只能写"这批 embedding 未带来可确证的 MOVE 增益"，不能写"embedding 无用"**。

## 限制与未解问题
- E0 留任是"没有更好候选"，不是"E0 最优"；E0 三 seed 配对基线已冻结供后续复用。
- dev veto 不适用（E*=E0，无升级可否决）；tau 未重算；confirm 从未读取。

## 证据指针
- `docs/reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md`（§0 结论先行、§3 单 seed 初筛、§4 E0 留任与 dev）
- `docs/design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md`
- `docs/README.md` 2026-10-29 节

## 后续影响
- E14 换 backbone（不换任务/评估）、E15 换 decoder 均沿用 E13 的 matched-A0/预注册/H2 硬门范式；E13 的 E0 三 seed 成为后续 parity 资产。
