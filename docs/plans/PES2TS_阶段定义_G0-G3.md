# PES2TS 阶段定义 G0–G3（权威）

> 状态：权威（阶段与边界定义的唯一来源）
> 层级：主干 G0/G1/G2/G3；工作流 C/R/X
> 取代关系：取代《统一开发顺序与阶段验收》中的 S0–S8/demo 阶段表；S0–S8 内容降为历史映射（见 §5）
> 上游依赖：无
> 下游消费者：整体开发与发布方案、G2 生成攻坚计划、文档修订台账
> 当前版本：v2 @ 2026-10-03（Classic / Rank / Gen、严格验证与训练边界修订）

---

## 1. 阶段总览

```
G0  数据底座
    download → inventory → quarantine(真值隔离) → dedup/audit → split/freeze → cohorts

G1  反应图审计与成键方式统计
    endpoint graph → edit graph ΔG(F/B/O/H) → event coupling → P1 mapping → P2 classify
    → v2 mutually-exclusive edits + 3D audit → scan-ready whitelist export
    ▸ exit: G1→G2 boundary contract frozen + 24-case human review closed

G2  PES generation（当前战场）
    2.0 endpoint assembly
    2.1 method selection & plan freeze
    2.2 execution (xTB PATH / ORCA Scan·Constraints·single-point gradient)
    2.3 per-frame recovery + quality + rule candidates + validation evidence
    2.4 execution unification (Track A/B)
    2.T small-batch test (24 cases)   ← 原 "demo"，仅为 G2 的测试
    ▸ exit: 24 cases with audited terminal outcomes under ACP + ≥1 strictly verified generated target TS

G3  Scale-out & first-generation RANKING
    3.0 frozen 1000-reaction cohort + complete funnel + labeled subset
    3.1 learned RANKING on fixed paths + cost metrics
    3.2 independent validation sampling
```

**核心裁决**：`demo` 不是阶段，`Demo24` 更名为 **G2.T 小批测试集**。物理验证（OptTS/频率/双向 IRC）不是独立阶段，而是 **G2/G3 每批的验收门**。

**算法路线映射（2026-10-03）**：Classic = G2 的物理基线与 G3.0 规模化；Rank = G3.1–3.2 固定 generation 后学习排序；Gen = Rank 验收后的 G3 研究扩展。G2 必须已有能量峰/启发式候选提取以完成验证链，推迟到 G3 的是学习型排名。详见 [G2 后续开发方案](PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md) 与 [ACP 配套计划](PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md)。

---

## 2. 各阶段定义

### G0 · 数据底座
- **定义**：把 Reaction-QM 原始数据变成干净、防泄漏、可复现的数据集，并把答案（TS/IRC）物理隔离。
- **交付**：`inventory.parquet`、`split_manifest.json`、`cohort_{trial,stratified}.json`、`data/ground_truth/*`。
- **当前状态**：✅ 完成（199,217 行；划分冻结）。
- **退出条件**：已达成。

### G1 · 反应图审计与成键方式统计
- **定义**：对每条反应从无真值端点算出键变化图，做互斥语义审计与统计，并产出交给 G2 的白名单导出。
- **子层**：
  - 端点图 / 编辑图 ΔG（formed/broken/order_changed/H 迁移）
  - 事件耦合图 H（把编辑聚成反应事件）
  - P1 真值映射（`--allow-truth`，仅供诊断/分类，不入生成输入）
  - P2 反应分类（方向敏感 + 方向不变）
  - G1 v2 审计修复（互斥编辑 + 三维审计 + 白名单导出）
- **交付**：`g1_change_v1`、`g1_p1_truth_v1`、`g1_p2_class_v1`、`g1_v2_edits_v1`、`g1_v2_class_v1`、`g1_v2_export_v1`、`g1_v2_gate.json`、`g2_scan_ready.json`。
- **当前状态**：✅ 图/审计/统计完成（v1+v2，导出 183,460）；⚠️ 边界合同未显式化；24 条人工复核仍 pending。
- **退出条件**：G1→G2 边界合同冻结（§3）+ 24 条人工复核放行。

### G2 · PES generation
- **定义**：把反应变成可审计的路径（选方法 → 编译 → 执行 → 逐帧回收 → 质量 → 验证）。
- **子阶段**：
  | 子阶段 | 内容 | 现状 |
  |---|---|---|
  | 2.0 端点装配 | 多组分刚体装配 + 碰撞分离 + 打分 | ✅（i X 共享服务） |
  | 2.1 方法选择与计划冻结 | 图论选择器 → StrategyProposal → GenerationPlanV2 | 🔶 合同完整，未真正接线全量 |
  | 2.2 执行 | Track A xTB PATH；Track B ORCA Scan/Constraints/单点梯度 | 🔶 A 可全量；B 受阻 |
  | 2.3 回收、质量与验证证据 | 逐帧 → 部分/完整路径资格 → 规则候选 → ACP 验证 → 图身份/成本 | 🔶 有模块，严格闭环未完成 |
  | 2.4 执行统一 | ExecutionBackend + ACP 唯一计算后端 | 🔶 接缝已建；迁移与能力补齐见 ADR-0002 |
  | 2.T 小批测试(24) | 24 条端到端 | 🔶 进行中（round0–round3） |
- **当前状态**：🔶 进行中。第二轮 6/24 连续、尚无已确认目标 TS；第三轮 strict 四例记录为 1 完整、2 LOCALITY_LIMIT、1 缓存身份错误，参考 TS 校准记录仍 IRC 失败。局部校正/梯度原型已有，正式 ACP 全入口收敛与科学验证尚未完成。证据入口见 G2 后续方案 §2。
- **退出条件**：冻结的 24 条均经 ACP 统一体系取得可审计终态与完整失败/成本账本；**≥1 条非参考种子生成的严格目标 TS**；配方/回放证书和真实能力冒烟通过。拒绝或部分路径算已处理，不能算完整路径/科学成功，三类数量分列。
- 此处“跑通”明确指自动流程闭环；不意味着 24/24 化学成功。≥1 是工程最低门，不构成普适可靠性结论。完整连续路径率、目标成功率和适用范围必须随发布披露。

> **G2.4 执行统一注记（2026-10-03）：** 子阶段 2.4「方法统一」的裁决现见 [ADR-0002 计算后端统一经 ACP 执行](../design/decisions/ADR-0002-计算后端统一经ACP执行.md)（**Accepted**）：所有计算后端（XTB_PATH、ORCA scan/constraints/NEB、continuation）统一经外部 ACP 执行；ADR-0001 的执行本地化条款与 X1/X2 退出条件被**部分取代**，接缝/计划/记录/词表仍有效。实施按《[PES2TS_ACP执行统一与旧代码清理方案_20261003](PES2TS_ACP执行统一与旧代码清理方案_20261003.md)》X1′–X5′ 推进（**planned / ADR-0002**；本地 xTB runner 删除与全后端 ACP 化尚未完成）。G-R 模块布局迁移已完成：`scan_strategy/` → `generation/planning/`，`g2/` → `generation/execution/xtb_path/`，`endpoints` → `generation/assembly/`。

### G3 · 规模化与第一代学习型 RANKING
- **定义**：把 Classic 扩到冻结的 1000 反应输入队列，建立验证标签子集，再在固定 generation 上学习排序。
- **子阶段**：
  - 3.0：1000 条输入的完整漏斗；有效路径、候选数和已验证标签数分别统计。1000 是规模化目标，不通过剔除失败凑“1000 成功”。
  - 3.1：固定路径上比较能量峰、事件峰、启发式与学习型 RANKING；输出 SeedProposal + 目标成功概率/成本指标。
  - 3.2：独立验证抽样（OptTS/频率/IRC）。
- **当前状态**：⬜ 学习型排名未开始；已有 v1 规则 ranker，先在 G2 接通真实验证。
- **退出条件**：1000 输入队列终态/成本完整；有效路径与真实标签覆盖明确；同预算独立排序/验证指标可复现。此定义替代旧“1000 条有效路径”作为唯一门，防止只保留成功样本。
- **后续扩展**：Rank 通过后再学习 generation 和在线停止，不新增并列主干阶段。帧数不等于标签数，训练启动还需标签质量和独立划分门。

**标签边界**：ValidationResult 不得反馈当前推理或 valid/test 调参；允许由隔离构建器导出 train 分区标签用于离线训练，再冻结模型评测。Demo24 已使用真值开发，整集作为开发回归；保留历史划分记录并增加污染标记，新的独立队列承担泛化评估。

---

## 3. G1→G2 边界合同（解决割裂的核心）

> 本章是 G1 放行 G2 的唯一契约。它取代"G2 自行假设 G1 已处理"的隐式约定。

### 3.1 G1 必须向 G2 保证（否则不放行）
1. **白名单导出完整**：map 升序原子序、端点坐标、电荷/自旋、**互斥** F/B/O、氢伙伴变化、芳香区、反应中心、分类。
2. **活动编辑视图**：明确哪些编辑是 `scan-active`（完全成/断键）、哪些 `archived`（键级变化）——G2 **不自行推断**。
3. **电子态一致**：R/P 电荷与多重度确定且一致。
4. **方法一致性标注**：端点是/不是**工作方法**（如 GFN2-xTB）的稳定极小点，供 G2 起点处理与 `METHOD_INCONSISTENT_ORIGIN` 判定。
5. **真值无关 + 可复现身份**：content hash 绑定，零真值派生字段。

### 3.2 G2 不得假设（必须在 G2 内处理）
- 端点几何是工作方法的稳定极小点；
- 活动编辑集能唯一确定反应通道/构象分支；
- 线性 λ 调度有效；
- 单一固定扫描方法适用于所有反应。

### 3.3 放行门（强化版 gate）
放行产物必须同时给出 `{method, plan_ref}` 与 §3.1 的边界字段；`gate_pass=false` 时不得暴露"看似可用"的导出。

---

## 4. 工作流（与主干正交）

| 工作流 | 阶段 | 目标 | 归属 |
|---|---|---|---|
| **C · 清理** | C0/C1/C2 | 收口未提交实验层、去重、归档脚本、文档收敛 | 前置于多数阶段 |
| **R · generation 核心研究** | R0–R4 | 量化非驻点→驻点接受门→分支态+信任域→软 guard/NEB→起点一致性+验证接线 | 阻塞 G2 质量 |
| **X · 执行收敛** | X0 + X1′–X5′ | 接缝→ACP xTB→证书/投影→全 QC 迁 ACP→可视化与清理 | G2.4 |

详见《PES2TS_整体开发与发布方案_v2》§5 与 ADR-0001；G2.4 执行统一的最新裁决与实施计划见 [ADR-0002](../design/decisions/ADR-0002-计算后端统一经ACP执行.md) 与《[PES2TS_ACP执行统一与旧代码清理方案_20261003](PES2TS_ACP执行统一与旧代码清理方案_20261003.md)》（X1′–X5′）。

---

## 5. 旧阶梯 → G0–G3 映射（历史兼容）

| 旧阶梯 | 出处 | 归入 |
|---|---|---|
| S0–S8 | 统一开发顺序与阶段验收 | 主干 G0–G3 |
| 层0–7 | 双项目开发与Demo方案 | = 主干 |
| G1-0…G1-7 | G1_v2 补全实施总方案 | = L0–L4 |
| L0–L4 / P0–P3 | 扫描规划谱系合并方案 | L0=C 收口；其余=X 工作流 |
| 第二轮 S0–S3 / 立体 v2 M0–M4 | 两份 20261002 方案 | = R0–R4 |
| S5 物理验证 | 统一开发顺序 | G2/G3 验收门（非独立阶段） |
| S7 "24 Demo" | 统一开发顺序 | **G2.T 小批测试** |
| S8 试点 200–1000 | 统一开发顺序 | G3.0 |

---

## 6. 术语

| 术语 | 含义 |
|---|---|
| F/B/O | formed / broken / order_changed 键编辑 |
| scan-active / archived | 作驱动 / 仅归档 的编辑 |
| G1→G2 boundary | G1 保证 + G2 不得假设 的契约（§3） |
| G2.T | G2 的 24 条小批测试（原 demo） |
| 严格验证 TS | 无生成约束的 OptTS 收敛、相关单虚频、双向 IRC、端点极小值及完整 R/P 化学身份通过 |
| C/R/X | 清理 / 生成研究 / 执行收敛 三条工作流 |
