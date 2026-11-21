# E12 V5.3-MP 规模联合训练与 2k→50k 收益结论

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P5
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E12　**所属期**：P5（2026-10-21→10-27）　**关联 ADR**：ADR-0031　**负结果**：N15,N16

---
> 条目标号：E12　时间范围：2026-10-21 → 2026-10-27　状态：Archived（报告 write-once / 计划状态头待裁决 A02）
> 关联 ADR：ADR-0031（Accepted，10-21）；合同 `docs/contracts/PES2TS_MoveProcessContract_v1.md`（Active-Normative）
> 数据/代码/模型版本指针：`split_manifest_v53mp.json`、`v53mp_split_v1`、dev pack（589 曲线/135,926 positions）；evidence 根 `outputs/pesn/v53mp/`；registry `config/registry/g_v_map.yaml` 的 V5.3-MP 条目
> 证据根：`docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md`、`docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md`、S1/S2/S3 报告、`docs/reports/PES2TS_V53MP_规模训练收益报告_2k_to_50k_20261027.md`

## 目标
在 MOVE–PROCESS 统一合同下，用真实 20k/50k 数据回答：加数据对 MOVE 还有没有用？对 PROCESS 头有没有用？代价是什么？

## 核心思路
- S0 是**真实数据建库**（当前六表只有 50 曲线/12,034 行），不是配置开关；分片惰性加载是硬交付。
- 嵌套池 + 冻结 dev + confirm 仅列名封存（数据纪律先于训练）。
- 三臂分别隔离监督信号（A=Move-only、B=+ranking/q、C=+endpoint），同评估集对比。
- Move guard `rmse_degradation_pct ≤ 2.0`（相对 R3 F2-bidir zero-shot）作为负退化护栏。
- 规模结论分头报告，禁止把 Move 与 Process 的收益合并成一句"加数据有效"。

## 关键决策
- ADR-0031：一个共享 EGNN、两个头；No-Move = p_end ≥ τ_end 且目标化学状态一致；q 仅辅助（`never_stop_criterion`）。
- S3 前冻结 `V53MP-S3-formal` 门（9/9 done、guard、无 NaN、checkpoint 三方一致、评估集冻结、attempt 账目）；S3 后单独报告闭环门。

## 主要成果与量化指标
- S0：20,000 反应冻结池 + dev 589 + confirm 2,000（仅列名）；S0-B 提取 26,038 曲线/5,958,548 样本；endpoint 正例覆盖 99.62%；就绪 7/7、表校验 10/10。
- S1：三臂 ok、Move guard 全过、GPU 1,829s（≤3,600s 上限）。
- S2：9/9 done、guard 全过、零 NaN、GPU 合计 13,048s。
- S3（20k 主结果）：9/9 done、guard 全为负退化 −9.9%～−12.0%（优于 R3 zero-shot）；ordering full 0.709 / cross-TS **0.80**；q corr 0.21–0.25；endpoint AUPRC 0.244、AUROC 0.597；闭环门通过（reference-normalized 与 autonomous 分开报告，声明限 TS_candidate/endpoint_reached）。
- 规模收益（四档同口径、589 条 dev curve/135,926 positions）：Move RMSE/zero 5k 0.9244→20k 0.8827→50k 0.8795；Process q corr 0.05→0.30→0.40、AUPRC 0.2096→0.2442→**0.3003**（1.72× prevalence）、AUROC 0.6609；**若目标是 process 头，50k 值得；若目标是 Move，不值（+2.5× 数据换 <1%）**。

## 关键负结果
- **Move 规模收益 20k 基本饱和**；50k 的 Move 变好一部分来自"弃权变多"（sign 降 1.7–2.2pp、dead-zone 2.1%→6.0%）。
- **endpoint 头 seed 间不稳定**：C_s29 起点/TS 误停率 0.63；A/B 未训练头误停率在 {≈0,≈1} 摆动 → No-Move 在线控制权仍待独立验证（τ_end 未校准，No-Move 结构性未开放）。
- **成本不可跨档比较**：代码在档间演进（I/O 与缓存层），20k GPU 小时与 5k 几乎相同是缓存优化而非规模效率。
- 2k 档 1 seed/100 updates 仅链路验证；λ 探针记录 rank 项为弱项（0.48% encoder 梯度占比），虽在 20k 转化为显著排序能力，但机制仍弱。
- 计划状态头矛盾：L3 `Proposed v1.1` vs 同头部"执行状态（2026-10-23）：S0/S1/S2/S3 训练全部完成"（A02/P0，禁新训练再引用 Proposed 版）。

## 限制与未解问题
- confirm 2,000 全程未触碰；dev 589 是唯一评测集，泛化边界未展开。
- 50k 六表 128 GB 无发布渠道（11-20 才发布划分名单，见 E16）；本战役证据在本地 `outputs/`，克隆不可复现训练。
- 下一档 100k 仅建议在 process 目标下做，且须先登记 endpoint 验收线。

## 证据指针
- `docs/reports/PES2TS_V53MP_规模训练收益报告_2k_to_50k_20261027.md`（§0 一页结论、§1 阶梯与口径、§2 主表）
- `docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md`（状态头与 §1–§2）
- `docs/reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md`（状态头）
- `docs/contracts/PES2TS_MoveProcessContract_v1.md`

## 后续影响
- PROCESS 头在 50k 起量的判断直接决定 E13（Embedding 50K）与 E15（双 Decoder 50K）沿用 50K 数据与 PROCESS 评估合同；Move 饱和结论促成 E14 改为"backbone 结构搜索"而非"继续加数据"。

---

### 第 6 期 · V5.4.x 三阶段期（2026-10-27 → 2026-11-06）

一句话目标：在 MoveProcessContract v1 不变的前提下，连续三个独立预注册战役（V5.4.1 Embedding / V5.4.2 Backbone / V5.4.3 双 Decoder）分别回答"换嵌入有没有用""换 backbone 有没有用""换 decoder 有没有用"。

核心思路：①每次只改一个受控组件，任务语义、合同、评估口径不动；②预注册 + D1–D8 签认 + 预算签认 + parity 证书先行；③H2 非劣护栏优先于 PROCESS 增益（增益不得豁免 MOVE 坍缩）；④confirm 永封、dev 一次性访问；⑤无候选合格就诚实写 `no_winner`，不产生"最低分胜出"。

关键技术决策：ADR-0032（V5.4.1 Embedding 接口扩展与预注册治理，Accepted）、ADR-0033（V5.4.2 Backbone 接口与预注册治理，Accepted v1.3.3）、ADR-0034（V5.4.3 双 Decoder 与 DecoderContextV1，Proposed 但 D1–D8 已签认）。

主要成果与指标：V5.4.1（10-29）：WP0–WP7 完成、6 次正式训练（37,900 updates）、**E*=E0 留任**（dev AUPRC 0.290702 / q_corr 0.397494 / RMSE-zero 0.910039）；E1/E2 H2 sign guard 失败、E3 sign+dead 失败 → `scientific_failure`；E4 core ratio 3.6455>2.0 → `gate_skipped`；E5/T3/T4 `gate_skipped`；无三 seed CI（不编造）；rollout 1,152 episodes、endpoint arrival 0.005208、TS candidate 0.055556、QC=0。V5.4.2（10-31）：**`no_backbone_winner`，A0 留任**——H3 eligible=[] → `zero_h3_qualified_candidates_a0_retained`；N_core=17；A0 sanity band PASS（AUPRC 0.29670 vs E0 0.28858）。V5.4.3（11-06）：18-run 矩阵全部 done（682,200 updates）；**MOVE 冠军 M1 Residual MLP（37,272 参）：rmse_zero +4.77% 相对改善（95% CI [+4.60%, +4.95%]，p=1/10001），dev −6.1%/dead −27%/sign +3.9pp**；**PROCESS 冠军 P3 Latent Graph-GNN（36,210 参）：AUPRC +0.283、q MAE +0.351，dev AUPRC 0.437→0.901、q MAE 0.614→0.233、ordering 0.804→0.993、Brier −68%、ECE −80%**；BMP 双冠军集成为两轨精确并集、零干扰。

关键负结果：V5.4.1 的 4 个 embedding 候选**全部失败**（E1/E2 sign、E3 sign+dead、E4 成本门）→ 判定"这批 embedding 没有带来可确证的 MOVE 增益"，容量/深度混杂未闭合已声明；V5.4.2 的 A2/A3 inner-val 与 PROCESS endpoint 明显优于 A0，但 **seed 43 触发 H2 breach（MOVE sign 与 fixed-coverage direction 相对同 seed A0 退化）→ `scientific_failure`，不得以 PROCESS 收益豁免**，A1 未进 top2、A4 被 5% 学习筛淘汰（3.849742%<5.0%）、A5 完成 full-dose 但 seed17 未过 screen；V5.4.3 的 M3（rmse 改善最大 −14.1%）被 H2 非劣门双 seed 排除（direction/dead/sign 劣化）、M2（screen 第 1）在 seed43 复现失败 → **多 seed 复现纪律拦截了单 seed 假冠军**；V5.4.3 的 M4 Flow/M5 Diffusion/P5 GraphGRU 条件未触发、P4-C 未入正式矩阵、S2 私有 Adapter 未启动（MOVE 信息上界突破未做）。

数据/代码/模型指针：`config/registry/g_v_map.yaml` 的 V5.4.1/V5.4.2/V5.4.3 条目（status executed / plan_revision）；V5.4.3 终版 bundle `PES2TSBundleV543_20261106T095436Z`（208 文件 verify ok）；V5.4.1 E0 冠军 bundle `outputs/pesn/v541_embed_50k/exports/embedding_champion_bundle/`（8 文件 white-list）。当期原始文档 git 范围：10-27…11-06 共 4+1+6+5+6+6 次提交（10-27、10-29、10-30、10-31、11-06）。
