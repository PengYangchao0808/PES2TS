# E11 V5.2 任务纠偏与 ADR-0030/0031 两次语义修订

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P4
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E11　**所属期**：P4（2026-10-21）　**关联 ADR**：ADR-0030/0031　**负结果**：N13,N14

---
> 条目标号：E11　时间范围：2026-10-21（单日两次语义修订）　状态：Archived
> 关联 ADR：ADR-0030（Accepted，10-21）、ADR-0031（Accepted，10-21）；取代链：ADR-0021/0028/0029 相关语义、FullPathNavigationContract v1
> 证据根：`docs/reports/PES2TS_V52_任务纠偏审计_20261021.md`、`docs/plans/PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md`、`docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md`、`docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md`

## 目标
回答"V5.2 为什么 q 不可学习、stop 随机"：是训练问题还是任务定义错误？

## 核心思路
- 先做只读审计（不新训练）定位标签-任务错配，再用 ADR 撤回旧语义，而不是调参。
- 任务定义必须能在数据结构里被验证（终点样本是否进入训练集）。
- 撤回与替代同日完成：旧语义冻结为历史，新语义另立合同。

## 关键决策
- **ADR-0030（Accepted）**：V5.2 的"TS 邻域 Stop"属任务定义错误，撤回；改为 Full-Path Navigation + Endpoint Stop（Move 沿完整 IRC 推进并跨 TS；No-Move=目标端点到达，d_end 窗口 0.10/0.30 Å）；TS 是途经点。
- **ADR-0031（Accepted）**：V5.3-MP 的任务定为 MOVE + PROCESS 双头（一个共享 EGNN）；No-Move = p_end ≥ τ_end 且目标化学状态一致；q 仅辅助/诊断、不得单独触发停止；旧 FullPathNavigationContract v1 只有历史价值。

## 主要成果与量化指标
- 审计实测：到达目标端点行 92/200，其中 90 行 stop=0、1 行偶然 1、1 行 Mask；`terminal` 行 1,123（距最近端点中位 6 / 最大 47 节点）；旧 Stop 正例 405 行 / 196 个 (curve,node)、d_key_TS 中位 **0.021 Å**、端点行正例仅 2/200。
- V5.2-R 判定链：D0 PASS / D1 FAIL / D0.1 PASS（22 条异常全部 T→Mask）/ D2 P1 PASS + P4/P5 FAIL → **D2-G STOP，T1 不授权**；V5.2-R 关闭并由 V5.3-MP 接续。
- 合同同步：`docs/contracts/PES2TS_ReactionIntent_Contract_v2.md`、`docs/contracts/PES2TS_GeometryPolicyContract_v2.md` 取代 v1 的 Stop/q 条款；V5–V7 v3.0（内容 v3.2）重定义三阶段。

## 关键负结果
- **V5.2 的 Stop/Progress 任务整体判负并撤回**（不是 bug 修复，是任务定义错误）。
- V5.2-R D1 FAIL、D2 P4/P5 FAIL → T1 不授权；旧 `FullPathNavigationContract v1` 的 d_end 单一语义被 ADR-0031 取代。
- V5–V7 v2.0 被 v3.0 取代；旧 branch/旧 V5.2 语义"不得作为当前训练依据"。

## 证据指针
- `docs/reports/PES2TS_V52_任务纠偏审计_20261021.md`（§0–§3）
- `docs/design/decisions/ADR-0030-V5.2全路径导航与端点停止.md`、`ADR-0031-MOVE-PROCESS统一学习合同.md`
- `docs/README.md` 2026-10-21 两节（V5.2-R 与 V5.3-MP）

## 后续影响
- ADR-0031 成为 E12 及之后所有 V5.x 战役的**唯一任务合同**，直至 11-21 仍有效。

---

### 第 5 期 · V5.3-MP 规模联合训练期（2026-10-21 → 2026-10-27）

一句话目标：在 MOVE–PROCESS 合同下，用嵌套规模阶梯（2k→5k→20k→50k）独立回答"MOVE 与 PROCESS 各自的规模收益在哪里饱和"。

核心思路：①S0 先建真实 20k 数据池（不是配置开关：六表只有 50 曲线/12,034 行，约 200k 反应库尚未转换）；②嵌套池 2k⊂5k⊂20k⊂50k + 冻结 dev 589 + confirm 2,000 仅列名封存；③三臂 A=Move-only / B=+rank+q / C=+endpoint，同剂同评估集；④Move guard ≤2% 负退化护栏；⑤规模收益分头报告（Move 与 Process 不合并结论）。

关键技术决策：ADR-0031 + `docs/contracts/PES2TS_MoveProcessContract_v1.md`（Active-Normative，10-21）；V5.3-MP 计划 v1.1（10-21）冻结 S0–S4 与三臂（状态头 `Proposed v1.1` 与已执行事实冲突，A02/P0）。

主要成果与指标：S0（10-22）PASS——20,000 条合格独立反应冻结池（2k⊂5k⊂20k 嵌套）+dev 589+confirm 2,000 仅列名；资格 142,221/141,483；审计提取 26,038 曲线 / 5,958,548 样本；endpoint_label_spec_v1 有效正例覆盖 99.62%、TS/起点 0% 正例；S0-D 分片惰性加载过资源门；S0-F 就绪 7/7 + 表校验 10/10。S1（10-22）三臂 ok、Move guard 全过、GPU 1,829s ≤ 3,600s。S2（10-22）9/9 done、guard 全过、零 NaN、GPU 13,048s。S3（10-23）9/9 done、guard 全为负退化（−9.9% ~ −12.0%）、零 NaN、GPU 12,699s（≈3.5 GPU-h，断点续跑后实际量级 ≈14,000–14,200s）；ordering cross-TS 从 5k 的 0.57 升至 **0.80**、q corr 中位 0.21–0.25、C 臂 endpoint AUROC 0.597–0.602；闭环门 `V53MP-S3-closed-loop` 通过。S4/规模收益（10-27）：Move 5k→20k 改善 2.9–4.5%，20k→50k 仅 0.4–1.0%；50k Move sign 降 1.7–2.2pp、dead-zone 从 2.1% 涨到 6.0%；Process 头 q corr 0.05→0.30→0.40、endpoint AUPRC 0.210→0.244→0.300（=prevalence 1.72×）、AUROC 0.6609。

关键负结果：**Move 的规模收益在 20k 基本吃完**（50k 的 Move 数字变好有一部分是"弃权变多"而非判得更准）；**过程头收益不能迁移为 No-Move 在线控制权**——C_s29 的起点/TS 误停率 0.63（seed 间不稳定），A/B 未训练头误停率在 {≈0,≈1} 摆动，直接证明"训练资格 ≠ 部署资格"；跨档代码在演进 → 精度可比、**成本不可比**（20k 的 GPU 小时与 5k 几乎相同是 loader 缓存优化，不是规模效率）；2k 档只有 1 seed/100 updates，仅链路验证不参与收益结论；lambda 探针曾记录 rank 项弱（λ_rank=0.5 上限仅 0.48% encoder 梯度占比）。

数据/代码/模型版本指针：`split_manifest_v53mp.json`（`v53mp_split_v1`：pool tiers 2k/5k/20k/50k、dev 589、confirm 2,000、curve_folds 66,195）；50K 六表在 WSL `pes2ts_data/v53mp_50k_data`（128 GB，不入库，见 E16 划分名单发布）；`docs/infra/wsl_memory_governance.md`（S4 infra 工具链）。当期原始文档 git 范围：10-21…10-27 共 37+19+4+3+4+1+4 次提交（10-21 后段、10-22…10-27）。
