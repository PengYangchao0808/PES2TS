# E17 文档规范化归档方案 v2.0 与 253 条全量审计

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P7
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E17　**所属期**：P7（2026-11-21）　**关联 ADR**：无新 ADR　**负结果**：N21

---
> 条目标号：E17　时间范围：2026-11-21　状态：Archived（方案本身 Active-Proposed，待所有者签认转 Canonical）
> 关联 ADR：无新 ADR；治理依据 `docs/PES2TS_开发宪法_v1.md` §0/§12/§14、`AGENT.md` §9、`docs/archive/README.md`、`docs/superseded/README.md`
> 数据/代码/模型版本指针：审计目录 `docs/infra/doc_audit/20261121/`（doc_inventory.csv 253 条 / DISPOSITION.csv 253 条 / FAMILIES.md 21 家族 / ANOMALIES.md 32 条 / GOVERNANCE_CONSTRAINTS.md 30 条约束 + 52 条钉死路径 / BATCH_PLAN.md / VERIFICATION_REPORT.md / baseline_gate_outputs.txt）；HEAD `2f1d3ba`
> 证据根：`docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md`、`docs/infra/doc_audit/20261121/README.md`、`docs/infra/doc_audit/20261121/FAMILIES.md`、`docs/infra/doc_audit/20261121/ANOMALIES.md`、`docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md`

## 目标
把 253 份文档的治理现状一次性盘清，并给出可分批执行、可回滚、可进 CI 的规范化归档方案。

## 核心思路
- 五层分类法（L0 治理 / L1 规范 / L2 执行 / L3 证据 / L4 运营）+ 11 状态封闭词表与转换规则。
- 元数据头 v2（8 字段，兼容既有 `> 状态：`）+ 命名与目录规范 + 归档三问判定法 + 动作矩阵。
- 全量处置映射（253 条逐条 keep/banner/move/retain-readonly 与 B0–B3 批次）+ 可复制迁移命令 + 门禁复跑与回滚。
- 索引/台账/清单三件套 + **G5 门禁规格**（`doc_manifest --check`）与 CI 接入计划，状态从"文档里的字"变成"机器可查的契约"。
- **本档案树（task-6）即该方案的"档案化"部分**：把历史压成条目，与"处置映射（搬家/横幅）"互补。

## 关键决策
- 方案自身状态 `Active-Proposed`（规范性方案），经仓库所有者签认后转 `Canonical` 并纳入文档门禁 G5。
- 逐条声明不违反 D-01…D-22 禁止破坏项；**默认不移动/删除任何文件**；B1/B2 执行需所有者逐批批准。
- 状态矛盾与治理选择列入方案 §14（12 个议题）与 `BATCH_PLAN.md` §5（19 条文档级裁决）待裁决。

## 主要成果与量化指标
- 全量：**253 条文档**（`docs/**/*.md` 216 份，活跃 180 / archive 36）；4.97 MB / 55,452 行；249 已跟踪 + 4 条本轮新增未跟踪。
- 状态：顶部状态块 167/216（活跃面缺 42）；状态原始值 137 种 vs 官方 5 种；状态头书写 5 种变体。
- 索引：`docs/README.md` 覆盖 133 / 台账 100 / **双缺 40**（reports 19、agent_runs 12、plans 4、superseded 4、infra 1）。
- 家族：**21 家族 / 53 成员**；6 组状态自相矛盾；"①可立即归位"实际 **0**。
- 门禁：**52 条硬钉死路径**（historical 17 / required_files 27 / root_scope 9 / g_v_map 43 引用去重合并，另 1 条软登记）；`docs_consistency --strict` CLEAN 基线下并存 **12 项冲突**（C-01…C-12）。
- 处置：keep 125 / retain-readonly 109 / banner 16 / move 3；B0 234 / B3 19。
- 本任务产出：档案树 7 期 17 条目 + 1 横切层 + 253 条映射（`extract_needed=Y` 67 条）+ 拟建文件树草案。

## 关键负结果
- **`docs_consistency` 未进 CI**：宪法 §12.4 要求的状态头/单权威/索引校验从未机器化，因此 12 项冲突长期在 CLEAN 基线下并存（C-09）。
- **索引双缺 40 份**且 README 自述版本滞后 5 处（C-03、C-08）——索引本身不是可信来源。
- 状态词表 17 个归并桶/40+ 原始值 vs 官方 5 值、状态头 5 种书写变体、状态头版本与文件名/正文三方不一致 2 例（C-01/C-02/C-05/C-07）。
- 门禁钉死路径只增不改（A09），归档动作空间被极大压缩：真正"零成本归档"的文档为 0。
- 176 warning 是容忍债务（C-11），不是 0 债务。
- 归档方案的风险：B1/B2 一旦执行会动 52 条钉死路径中的部分文件，需同步门禁配置；本轮**未执行任何移动**，方案与审计均声明"不构成移动授权"。

## 限制与未解问题
- 19 条文档级裁决（BATCH_PLAN §5）与 12 个治理议题（方案 §14）待所有者裁决；G5 门禁未接入 CI（M4a/M4b 两步计划）。
- 审计口径说明：审计目录自身后续新增的证据 md 不纳入处置映射（口径见方案 §3.1）。
- `docs/infra/doc_audit/20261121/` 内 7 份证据 md 与本档案树目录均属"本轮新增"，在 253 快照之外。

## 证据指针
- `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md`（§3 五层分类法、§7/§9 处置、§14 议题、G5 规格）
- `docs/infra/doc_audit/20261121/README.md`（审计口径与 253 条统计）
- `docs/infra/doc_audit/20261121/FAMILIES.md`（21 家族表）、`ANOMALIES.md`（32 条）、`GOVERNANCE_CONSTRAINTS.md`（C-01…C-12、52 条钉死）
- `docs/plans/PES2TS_文档修订台账.md` 2026-11-21 增量

## 后续影响
- 下一动作不在研究侧而在治理侧：所有者裁决 19+12 项 → 接入 G5 门禁 → 分批执行 B1/B2（每批门禁复跑）。档案树条目在治理执行期间保持冻结，仅追加"后续指针"。

---
