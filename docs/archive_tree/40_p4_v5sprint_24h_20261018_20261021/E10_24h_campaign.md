# E10 24H 自主战役 R3→V5.2→V5.3-ENG（完整示范成稿）

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P4
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E10　**所属期**：P4（2026-10-20→10-21）　**关联 ADR**：ADR-0027/28/29　**负结果**：N10,N11,N12

---
> 条目标号：E10　时间范围：2026-10-20 → 2026-10-21　状态：Archived
> 关联 ADR：ADR-0027（Proposed）、ADR-0028（Proposed）、ADR-0029（Proposed）；上位计划 `docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md`（状态头自相矛盾，A03/P1）
> 数据/代码/模型版本指针：`outputs/pesn/30h_campaign/data/`（六表；`dataset_sha256=4635c330…`，50 曲线 / 12,034 行，见 E11 审计）；gate 证据 `docs/agent_runs/state/gates/*.json`；E0 签认 `docs/agent_runs/E0_SIGNOFF_REPORT.md`
> 证据根：`24H_FINAL_REPORT.md`、`FAILURE_AND_REMAINING_WORK.md`、`R3_RESULT_SUMMARY.md`、`V52_PROGRESS_STOP_REPORT.md`、`V53_ROLLOUT_REPORT.md`、`docs/agent_runs/README.md`

## 目标
在 24h 自主授权窗口内一次跑完 R3（反应意图条件化步进）→ V5.2（进度/停止头）→ V5.3-ENG（rollout 工程闭环），并按 fail-closed 纪律给出明确的"成立/不成立"判定。

## 核心思路
- 三 lane 并行（DATA/FEAT/LOSS）+ supervisor 编排；每个 work package 有独立门与终态记录。
- 失败即归档：no-gain 后关闭余量，不靠调参复活。
- 训练/物理/工程成本分开记账（GPU 秒、xtb 秒、rollout CPU 秒）。
- 用四级完成度区分"实现/夹具/冒烟/科学有效"，禁止层级混用。

## 关键决策
- 用户 2026-10-20 授权 24h 自主执行（窗口 13:45Z→10-21T13:45Z），fail-closed 纪律全程无越权。
- E5 判 **scientifically_validated NEGATIVE**：q 不可学习、stop 随机、Move 退化 → 按设计关闭后续（G3 FAIL，E-NO-GAIN 诚实保存）。
- E6 V5.3-ENG 只声明 engineering-validated，不声明 TS 科学主张。

## 主要成果与量化指标
- 11 个工作包全部终态；六门全部终态记录；93 次 launch attempts（R3 61/66、V5.2 32/33；6 个从未 launch 的为 BLOCKED 臂）。
- GPU 训练 **0.93 GPU-h**（70 runs × ~48s，RTX 4060 Ti 8GB）；物理 xtb **49s**（36 次受约束松弛）；墙钟 **7.75h**。
- **R3 冠军 F2-bidir（fold0）：RMSE/zero 0.910 [0.884–0.949]、sign 0.888、6/6 优于零步、三种子一致**；B = F2-bidir 跨折 0.9735 [0.949–0.998]、10/12、衰减 ~6.4pp。
- V5.3-ENG rollout：216 episodes + 36 次真实 xtb，rollout sign-agree 0.88 / 弧长饱和 0.62 / contact 0.08；G4 PASS（仅工程闭环）。
- E1 六表数据三确定性重建、10/10 门；WP-L3 账本强化（Oracle 三篡改路径实拒）；E7 cluster 级 CI + 成本账本。

## 关键负结果
- **V5.2 三个假设全部不成立**：q MAE 0.507 vs 常数 0.476（劣于常数，corr≈0）；stop 头 AUC 0.491–0.527、recall@0.5=0；Move 相对 R3 退化 **+5.5% 中位**（4/6 破 ≤2% 护栏）→ 联合训练损伤。
- R3 非冠军臂 F0/F1/F2/F2-intent-off/F3 forward：RMSE/zero 1.107–1.225、sign 0.31–0.52、≈0/6 → 负结果保留。
- classic CV/Jacobian 对照在 rollout 中崩塌（sign 0.24、215 碰撞步），不可用作对照。
- rollout 漂移：16/36 物理失败中 15 例溯源于碰撞种子几何（因果链完整记录）；弧长饱和 0.62 → 幅度不足。
- 过早停止：R3 的 43/48 run inner-val 选择 ≤250/3000 updates（冻结日程过长的证据）。
- 已修复 8 项（含 λ-bug 15 attempts 烧毁、确认折泄漏 8 runs 自判无效、truth_guard 误报、worktree 工件漂移）；按设计阻塞 4 类（confirm 授权、R1/R2 spec gap、V5.2 余量、folds3/4 封存）。

## 限制与未解问题
- V5.2 的负结果在次日（E11）被证明是**任务定义错配**的必然结果，不是"Stop 难学"——但本条目的当时判定原样保留。
- 封存折 3/4 未揭示；confirm 臂未授权；E6 无任何 TS 科学主张。

## 证据指针
- `24H_FINAL_REPORT.md`（§1 完成度、§2 成本、§3 指标表、§4 四项显式回答）
- `FAILURE_AND_REMAINING_WORK.md`（§1 修复、§3 负结果）
- `R3_RESULT_SUMMARY.md`（状态头 `R3 阶段关闭（dev-positive / confirm-blocked）`）
- `V53_ROLLOUT_REPORT.md`（状态头 `历史命名（V5.3-ENG）`）
- `docs/agent_runs/README.md`（三 lane 运营纪律）

## 后续影响
- E11（10-21 同日）：任务纠偏审计 → ADR-0030（撤回 TS 邻域 Stop）→ ADR-0031（MOVE–PROCESS）→ E12 V5.3-MP 用独立预注册战役重做"跨 TS 的 Move + PROCESS 头"。
