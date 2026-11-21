# E15 V5.4.3 双 Decoder-50K（双轨冠军；完整示范成稿）

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P6
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E15　**所属期**：P6（2026-11-02→11-06）　**关联 ADR**：ADR-0034　**负结果**：N19

---
> 条目标号：E15　时间范围：2026-11-02 → 2026-11-06　状态：Archived（Executed-Frozen）
> 关联 ADR：ADR-0034（Proposed，但 D1–D8 于 11-02 经仓库所有者签认）；方案与 WP 清单保持 `Frozen-Preregistered v1.0`（按台账明示"执行完毕不改文"）
> 数据/代码/模型版本指针：`config/campaigns/pes2ts_v543_dual_decoder_50k.yaml`；终版 bundle `PES2TSBundleV543_20261106T095436Z`（208 文件 verify ok）；dev pack `4f6c1e95…`（589 曲线/135,926 positions）
> 证据根：`docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md`、`docs/plans/PES2TS_V5.4.3_WP执行清单_v1.0_20261102.md`、`docs/reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md`、`docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md`

## 目标
固定 A0 Encoder（57,619 冻结）+ 参数自由 `DecoderContextV1`，用 MOVE/PROCESS 两条独立残差 Track 的 7 个候选回答：只改 decoder 能否同时确证提升两轨？

## 核心思路
- 轨道隔离：MOVE 与 PROCESS 各自独立 decoder 与损失，共享只读 encoder；用 BMP 集成验证"两轨改善是精确并集、零干扰"。
- 预注册 18-run 上限 + 37,900 flat + 400 warmup + 同 Track 参数匹配（≤36K，整模硬上限 150,000）。
- 新指标 `fixed_coverage_direction@3`、q MAE/ordering、FPR-constrained recall、Brier/ECE；H2/H3 用 V543 M0/P0 重算基线。
- confirm 永封、dev 仅 T7 一次性访问、`strict=False` 禁用、禁 `torch.compile`。

## 关键决策
- ADR-0034（D1–D8 签认，2026-11-02 经 DSH 编排会话，签认人=仓库所有者）：五组件冻结架构（CoarseEncoder 57,619 / LatentBridge 0 trainable / MoveRefiner ≤36K / ProcessRefiner ≤36K / ReactionController 非学习）；`DecoderContextV1` 字段/单位/对称性/双只读视图/禁读清单/缓存键/A0 零残差 parity；提案 F1–F8 八处修正。
- 三条运行授权：WP3 go/no-go 内部核验后自主通过；每 Track smoke 通过后直接排队正式 run；dev 仅 T7 一次性、confirm 永封。

## 主要成果与量化指标
- 18-run 矩阵全部 `done`（682,200 updates；T3 seed17×7 + T4 seed29×7 + T5 seed43×4；每 run 37,900+400）；CPU 执行，单 run ~2.4h，合计 ~43h（frozen trainer 不迁 device，same-env 分母纪律优先）。
- **MOVE 冠军 M1 Residual MLP（37,272 参）**：inner-val rmse_zero **+4.77%** 相对改善（95% CI [+4.60%, +4.95%]，p=1/10001）；direction@3 +0.47pp（p=0.0008）；dev：rmse_zero −6.1%、dead −27%、sign +3.9pp、direction +2.4pp。
- **PROCESS 冠军 P3 Latent Graph-GNN（36,210 参）**：endpoint AUPRC **+0.283**、q MAE 相对 **+0.351**（双端点 p=1/10001）；dev：AUPRC 0.437→**0.901**、q MAE 0.614→**0.233**、ordering 0.804→**0.993**、Brier −68%、ECE −80%。
- BMP 双冠军集成为两轨改善的**精确并集、零干扰**（轨道隔离合同最终实证）；T7 dev 一次性访问恰好 1 行 ledger，misstop=pass、success=1.000、no_move=0；终版 bundle 208 文件 verify ok；`docs_consistency --strict` CLEAN。

## 关键负结果
- **M3（rmse 改善最大 −14.1%）被 H2 非劣门双 seed 排除**（direction/dead/sign 劣化）——单指标最优不等于合格。
- **M2（screen 第 1 名）在 seed43 复现失败（scientific_failure）**——多 seed 复现纪律拦截单 seed 假冠军。
- M4 Flow / M5 Diffusion / P5 GraphGRU 激活条件未触发；P4-C 对照未入正式矩阵（直连装配路径已验证）；S2 私有 Adapter/部分解冻未启动 → **MOVE 信息上界突破未做**。
- `no_move=0`、τ_end 未校准 → No-Move 在 dev 上结构性不可能，不能据此宣称 No-Move 能力。
- CPU 单 run ~2.4h 与方案 §8.4 的 43–55 min GPU 假设不符（~43h 总耗时）——成本口径偏差已入档。

## 限制与未解问题
- ADR-0034 状态仍为 `Proposed`（台账明示"不改冻结原文"，属设计内）；`docs/README.md` 缺 2026-11-06 条目（A04/A08 索引滞后）。
- 主方案 §2 F6 的 A0 q_corr 0.9988 与 V542 执行口径 0.99682 存在文档不一致（以 V543 evaluator 重算值为准）。
- 下一阶段（V5.4.4+/V6 外层物理松弛、chemical_state_ok 核验）只在报告交接段列为计划，未开始。

## 证据指针
- `docs/reports/PES2TS_V543_双Decoder_50K_实施与训练报告_20261106.md`（§1 执行摘要、§3 矩阵、§4 统计、§6 关键事实）
- `docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md`
- `docs/README.md` 2026-11-02 节

## 后续影响
- 双轨冠军 BMP 成为后续外层控制的基座（报告 §8 交接）；E16/E17 的治理动作不改动本战役证据。

---

### 第 7 期 · 治理归档与公开化期（2026-11-20 → 2026-11-21）

一句话目标：在"研究暂停、准备公开"的窗口里，把 253 份文档的史前史整理成可审计的资产：补 LICENSE、发布划分名单、建立 superseded 区、做全量审计并给出规范化归档方案——**本档案树就是这次治理的产物之一**。

核心思路：①只增不删：证据一份不删，移动必须过门禁（52 条硬钉死路径）；②先体检后方案：253 条清单 → 32 条异常 → 21 家族取代链 → 处置映射；③状态必须成为契约（状态头/索引/单权威要进 CI）；④可复现优先：划分名单逐字节入库 + sha256 链校验；⑤"归档"与"取代"分区（`docs/archive/` 快照 vs `docs/superseded/` 取代）。

关键技术决策：`docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md`（Active-Proposed，规范性方案；经所有者签认后转 Canonical 并纳入 G5 门禁）；`docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md`（Operational，非规范）；`docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md`（Descriptive v1.1）；`docs/superseded/README.md`（取代区规则）。

主要成果与指标：LICENSE（MIT，2026-11-20）补齐公开前最后缺口；5 份划分名单 15.2 MB 逐字节入库（索引 `data/manifests/split_registry_v1.json`，sha256 前缀 07bdbedc…/802f6243…/83e72e9a…/b65fa4c9…/c0d2cb2a…，链条校验均等于 L1 `802f6243…`）；`docs/superseded/` 建立并迁入 4 份自身已声明 Superseded 的文档；5 个零引用 V4 脚本归档 `scripts/archive/v4/`；CI 红灯修复（`gv_registry.py` 区分仓库内外路径，26 tests 由 4 failed 转全过）；门禁 `check_docs_links.py` 0 error / 173 warning（182 docs）、`docs_consistency --strict` CLEAN。全量审计（11-21）：253 条 / 4.97 MB / 55,452 行；`docs/**/*.md` 216（活跃 180 / archive 36）；状态块 167/216；状态原始值 137 种 → 17 个归并桶；索引双缺 40；21 家族/53 成员（6 组状态自相矛盾）；52 条硬钉死路径；32 条异常（P0 2 / P1 10 / P2 20）。

关键负结果：**40 份文档既不在 `docs/README.md` 也不在台账**（C-08）且索引自述版本滞后 5 处（C-03）；**`docs_consistency` 从未进入 CI**，12 项冲突在门禁 CLEAN 基线下长期并存（C-09）；176 条 warning 是"容忍债务"不是 0 债务（C-11，台账 11-20 记 173/182，口径差 1 属不同时点）；7 份根目录战役 memo 因门禁钉死不移动、只加索引（A30）；`docs/archive/**` 28 份与 live 文档同名（A22，快照只读不改）；归档家族"①可立即归位"实际为 **0** 份（F19 三份因有引用不满足），说明"零成本归位"在现实中几乎不存在。

数据/代码/模型指针：`docs/infra/doc_audit/20261121/`（doc_inventory.csv / DISPOSITION.csv / FAMILIES.md / ANOMALIES.md / GOVERNANCE_CONSTRAINTS.md / BATCH_PLAN.md / VERIFICATION_REPORT.md / baseline_gate_outputs.txt）；`data/manifests/split_registry_v1.json`；HEAD `2f1d3ba`。当期原始文档 git 范围：11-20 的 6 次提交（归档 4 份 + 5 脚本、LICENSE、CI 修复、划分名单 ×2、ids -text 固定）与 11-21 的审计/方案文档（未提交或本轮提交，`[未跟踪]` 状态以台账口径为准）。
