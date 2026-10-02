# PES2TS 阶段定义 G0–G3（权威）

> 状态：权威（阶段与边界定义的唯一来源）
> 层级：主干 G0/G1/G2/G3；工作流 C/R/X
> 取代关系：取代《统一开发顺序与阶段验收》中的 S0–S8/demo 阶段表；S0–S8 内容降为历史映射（见 §5）
> 上游依赖：无
> 下游消费者：整体开发与发布方案、G2 生成攻坚计划、文档修订台账
> 当前版本：v1 @ 2026-10-02

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
    2.3 per-frame recovery + quality
    2.4 execution unification (Track A/B)
    2.T small-batch test (24 cases)   ← 原 "demo"，仅为 G2 的测试
    ▸ exit: 24 cases run under the unified system + ≥1 strictly verified TS

G3  Scale-out & first-generation RANKING
    3.0 1000-reaction PES end-to-end runthrough
    3.1 first-generation RANKING + metrics
    3.2 independent validation sampling
```

**核心裁决**：`demo` 不是阶段，`Demo24` 更名为 **G2.T 小批测试集**。物理验证（OptTS/频率/双向 IRC）不是独立阶段，而是 **G2/G3 每批的验收门**。

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
  | 2.3 逐帧回收+质量 | multicoord → frame_recovery → target_path 四层 | 🔶 已落地未接线 |
  | 2.4 方法统一 | ExecutionBackend 协议 | ⬜ 见 ADR-0001 / X 工作流 |
  | 2.T 小批测试(24) | 24 条端到端 | 🔶 进行中（round0–round3） |
- **当前状态**：🔶 进行中。Track A 可跑；Track B 延续不稳定（24 条 6/24 连续、0 验证 TS）；两轨未统一；实验层未提交。
- **退出条件**：24 条在统一体系下跑通 + **≥1 条严格验证 TS**。

### G3 · 规模化与第一代 RANKING
- **定义**：把 G2 方法规模化到 1000 条，并在其上写第一代排序。
- **子阶段**：
  - 3.0：1000 条 PES 端到端跑通（完整漏斗入账）。
  - 3.1：第一代 RANKING（路径可用性/拒绝 → 帧排序 → SeedProposal）+ 指标。
  - 3.2：独立验证抽样（OptTS/频率/IRC）。
- **当前状态**：⬜ 未开始（仅 v1 合成演示 ranker）。
- **退出条件**：1000 条有效路径 + 排序与验证指标可复现。

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
| **X · 执行收敛** | X0–X4 | 接缝协议→xtb 后端正名→统一门→结果投影→纳入 ORCA/continuation+按需选方法 | G2.4 |

详见《PES2TS_整体开发与发布方案_v2》§5 与 ADR-0001。

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
| 严格验证 TS | OptTS + 单虚频 + 双向 IRC 达标 |
| C/R/X | 清理 / 生成研究 / 执行收敛 三条工作流 |
