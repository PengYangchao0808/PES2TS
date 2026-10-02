# PES2TS 整体开发与发布方案 v2（合并原始假设 + 最新修订）

> 状态：权威（项目级主方案）
> 层级：主干 G0–G3
> 取代关系：整合并取代《统一开发顺序与阶段验收》的阶段表与 demo 里程碑；其余定位/指标沿用《双项目开发与Demo方案》
> 上游依赖：PES2TS_阶段定义_G0-G3.md、ADR-0001
> 下游消费者：G2 生成攻坚计划(R)、G1-G2 边界合同、文档修订台账
> 当前版本：v2 @ 2026-10-02

---

## 1. 定位与不变原则（原始假设，全部保留）

**定位。** PES2TS = **PES generation**（从 R/P 端点生成可审计的廉价近似路径与能量剖面）+ **PES ranking**（在已计算路径中筛选、排序可提交 OptTS 的真实几何帧）。两者通过 `PathBundle` 合同连接，可分别开发/替换/评价。

**严格成功终点（不变）。** OptTS 收敛 + 频率确认目标模态的一阶鞍点 + 双向 IRC 连回目标 R/P。几何接近参考 TS、能量峰、单虚频**都只是中间证据**。

**执行边界（不变）。** ACP 负责任务执行/预算/日志/溯源；RPH 消费初猜并承担 QC 验证链；PES2TS 不把执行基础设施、数据真值、下游 QC 引擎内化。

**四条红线（不变）。** 真值隔离 / 合同密封 / 确定性 / 诚实拒绝（typed rejection，能力未冒烟=拒绝）。

**冻结纪律（不变）。** split 冻结；选择/阈值/参数在读取 valid/test 前冻结；已发布版本不可覆盖。

---

## 2. 原始假设 → 最新修订

| 主题 | 原始假设 | 最新修订 |
|---|---|---|
| Demo24 地位 | S7 里程碑 | **G2.T 小批测试**（非阶段） |
| 阶段编号 | S0–S8 等并行阶梯 | **统一 G0/G1/G2/G3** + C/R/X |
| 物理验证 | S5 独立阶段 | **每批验收门** |
| RANKING | S4 | **G3（先有 1000 条路径）** |
| G2 内部 | 隐含两轨 | **Track A/B 统一**（X 工作流） |
| G1 完成定义 | 图审计 + 成键统计 | 增加 **G1→G2 边界合同** |
| 规模 | S8 试点 200–1000 | **G3 = 1000 条 + 第一代 RANKING** |

---

## 3. 阶段与验收（摘要，详见《阶段定义_G0-G3》）

| 阶段 | 交付 | 退出条件 |
|---|---|---|
| G0 数据底座 | inventory/split/cohorts/真值隔离 | ✅ 已达成 |
| G1 图审计成键统计 | g1_change_v1 / P1 / P2 / v2 编辑 / 白名单导出 | 边界合同冻结 + 24 条复核放行 |
| G2 PES generation | 统一执行体系 + 计划 + 质量层 | 24 条跑通 + ≥1 严格验证 TS |
| G3 规模化 + RANKING | 1000 条路径 + ranking v1 + 指标 | 路径有效 + 指标可复现 |

统一漏斗：`候选 → 人工接受 → 输入可构造 → 路径完成 → 化学有效 → 有候选/拒绝 → OptTS → 频率 → 双向 IRC 目标匹配`。

---

## 4. 开发序列

| 序 | 阶段 | 内容 | 依赖 | 退出条件 |
|---|---|---|---|---|
| P0 | C 收口 | 提交/归档实验层、去重、归档脚本、清临时产物 | — | HEAD==worktree；套件绿 |
| P1 | G1 收口 | 边界合同 + 24 条复核放行 | P0 | 合同冻结；复核闭环 |
| P2 | G2 接缝 | X0–X2（协议 + xtb 后端正名 + 统一门） | P0 | 等价证书绿；xTB 人群零变化 |
| P3 | G2 攻坚 | R0–R4 | P0 | 同预算质量改善；≥1 验证 TS |
| P4 | G2 收敛 | X3–X4 | P2,P3 | 两法同 fixture 判定；能力诚实拒绝 |
| P5 | G2.T | 24 条小批测试 | P4 | 24 条跑通 + ≥1 验证 TS → G2 退出 |
| P6 | G3.0 | 1000 条端到端跑通 | P5 | 1000 条有效路径入账 |
| P7 | G3.1 | 第一代 RANKING + 指标 | P6 | 排序指标可复现 |
| P8 | G3.2 | 独立验证抽样 | P7 | 报告闭环 |

依赖图：

```
C0→C1→C2
X0→X1→X2→X3→X4
R0→R1→R2→R3→R4
G2 exit ⇐ R(≥1 TS) + X(统一协议)
G3 ⇐ G2 exit
```

---

## 5. 工作流定义

### C · 清理
- **C0 收口**：提交或归档未提交实验层；`_is_genuine_ring_event` 去重为单一来源；一次性脚本归档 `scripts/archive/`；删除临时产物；孤儿模块逐个标注 `landed-not-wired` / `experiment` / `dead`。
- **C1 去重/归档**：禁键家族集中登记 + 派生一致性测试；`planning.py`/v1 `ScanPlan` 标 deprecated。
- **C2 文档与配置**：文档治理（见 §10）；config `TODO: calibrate` 进标定台账。

### R · generation 核心研究（阻塞 G2 质量）
- **R0** 对 round2 每个接受帧跑一次 EnGrad，统计投影自由梯度直方图（判定"非驻点"是否全群）。
- **R1** 接受门加 `require_physical_gradient=True`（先不改校正器）。
- **R2** 每帧先收敛到驻点再作参考 + 质量加权信任域 + 特征值下限正则化 BFGS。
- **R3** 仅对仍 `LOCALITY_LIMIT` 的通道：投影 Hessian 最软特征向量选软 guard（soft harmonic）；仍失败则 CI-NEB/string（IDPP 初始化）。
- **R4** 起点连通性保持优化 + `METHOD_INCONSISTENT_ORIGIN` 检测；梯度排名接入验证；部分路径候选独立 OptTS/IRC。

### X · 执行收敛（把 Track A/B 合成一个体系；详见 ADR-0001）
- **X0** 冻结 `ExecutionBackend` 协议 + `TrajectoryRecord` + method 词表。
- **X1** `xtb_path` 后端包住现有 G2，对照子集**逐字节一致**等价证书。
- **X2** 追加 `unified_gate.json`（超集投影，G2 读旧列表不变）。
- **X3** `g2_path_v1` + Track B 逐帧 → `TrajectoryRecord` → `PathBundle`；`target_path` 同判两种方法。
- **X4** 纳入 ORCA/NEB/continuation；规划面按反应选方法（`run_if/pass_if/fallback_ids`）。

**明确非目标**：不中途迁移 19.1 万条；不重写 `runner.py`/`adapter.py`/`planning.py`；不合并 `g2_eligible`/`g2_scan_ready`；不合并两个合同模块；不靠复制 `assemble_endpoints` 统一；不削弱真值守卫白名单；不把 ORCA 的 `compiled` 强加给 xTB。

---

## 6. 发布方案

### 6.1 发布单元

| 发布物 | 内容 | 依赖 | 稳定标志 |
|---|---|---|---|
| `pes2ts-data-v1` | G0/G1 数据底座 + 图审计 + 白名单导出 | — | gate+hash+拒绝账本闭合 |
| `pes2ts-generation-v1` | G2 生成（统一后端+计划+质量层） | data-v1 | 24 条跑通 + ≥1 验证 TS；等价证书 |
| `pes2ts-ranking-v1` | G3 第一代排序（只依赖 contracts+PathBundle） | generation-v1 | 同路径集 Top-k 指标可复现 |
| `pes2ts-eval-v1` | 隔离评估区（真值/OptTS/频率/IRC/分母/成本） | — | 只读、不回流生成/推理 |
| `pes2ts-acp-bridge-v1` | ACP/RPH 集成与端到端清单 | data/generation | 真实冒烟 never-skip |

### 6.2 版本与冻结纪律
- **不可变产物**：冻结清单 + hash 收据；已发布版本不可覆盖，重算走新任务/新版本。
- **版本维度**：`dataset_version`、`schema_version`、`plan_version`、`content_sha256`。
- **最小命令合同**：`plan → generate → rank → validate → report`；每步读前一步版本化清单、写新清单+失败账本，可按 reaction ID 重跑不覆盖。
- **发布门**：每级 verify/freeze；`gate_pass=false` 不暴露可用导出。

### 6.3 发布边界
代码按接口隔离：`contracts/`、`g2/`（generation）、`ranking/`、`evaluation/`（隔离真值）、`integration/`（ACP/RPH）。合同稳定后 generation 与 ranking **可独立发布**；排序器只依赖 contracts。

---

## 7. 合同与接口

| 平面 | 合同 | 生产者→消费者 | 最小不变量 |
|---|---|---|---|
| 结果/执行 | `ReactionCase` | G0/G1→generation | 端点映射/几何/电荷/自旋/编辑；真值禁止 |
| 结果/执行 | `ScanPlan`(v1, deprecated) | G1→G2 | 被 `GenerationPlanV2` 取代 |
| 规划 | `StrategyProposal`/`GenerationPlanV2`/`BackendCapability` | G1→G2 | 候选携带 method；能力诚实拒绝 |
| 结果 | `PathBundle` | generation→ranking | 同原子序与能量口径；逐帧；异常可辨认 |
| 结果 | `SeedProposal` | ranking→验证器/RPH | 真实帧 ID/XYZ/排名/依据/拒绝 |
| 结果 | `ValidationResult` | 验证器→隔离评估区 | 各阶段状态；**不得回流** |
| 结果 | `ReviewRecord` | 复核→放行 | 双人独立 + 裁定 |

---

## 8. 验收与指标

- **Generation**：`输入可构造/计划`、`完成/可构造`、`化学有效/完成`、`化学有效/全部计划`；按 A–H、方法、方向、多组分展开。
- **Ranking**：**同一批冻结有效路径**、同 Top-k 与 QC 预算下比较 Top-1/3、MRR、每正确 TS 的核时。
- **集成**：`正确 TS+双向 IRC / 测试反应数`；`QC 核时 / 正确完成`；到首个正确 TS 的壁钟。
- **失败与成本**：失败/回退/超时照记；未知值显式 null；禁止只对成功案例算平均成本。

**G2.T 分层**：A–H，16 train + 8 valid；A–E 15 条核心，F–H 9 条特殊/回退。valid 用锁定规则与同预算；不做显著性或泛化结论。

**错误归因决策树**：无有效路径→改 G1/G2 计划与端点；有可用帧但规则选错→改 ranking；候选合理但 OptTS/IRC 不过→查验证协议/能量级别/候选几何。

---

## 9. 风险登记册

| 风险 | 影响 | 缓解 |
|---|---|---|
| G1/G2 边界不清 | 进入 G2 反复返工 G1 | G1→G2 边界合同 + 强化 gate |
| Track A/B 双维护 | 割裂、第三面 | X：接口先冻结 + 一个真实后端证明 |
| 生成延续不稳定 | G2 无法达标 | R：驻点接受门 + 分支态 + 正则化信任域 |
| 实验层未提交 | HEAD 与工作区不一致 | C0 强制收口 |
| 真值泄漏 | 科学有效性受损 | 静态守卫 + 白名单 + 评估区隔离 |
| 确定性破坏 | 不可复现 | 加 method 字段须新版本；绝不重封已冻结计划 |
| 能力不诚实 | 虚假 pass | `unknown` 必须拒绝；xTB 也需探针登记 |
| 动态执行白名单 | 挪动子进程削弱守卫 | 新后端文件显式登记 |
| 双人复核未完成 | 无法放行 valid | G1 收口（P1）前置 |

---

## 10. 文档治理

**唯一主干编号 = G0/G1/G2/G3；工作流 = C/R/X。** 每份文档加状态头。新增权威文档：本方案、阶段定义、G1→G2 边界合同（阶段定义 §3）、ADR 决策日志、标定台账。修订动作见《PES2TS_文档修订台账》。

---

## 11. 里程碑

| 里程碑 | 组成 | 判据 |
|---|---|---|
| M-A 收口 | P0+P1 | 工作区干净；边界合同冻结；24 条复核闭环 |
| M-B 统一生成 | P2+P4 | 一个执行体系；等价证书绿 |
| M-C G2 退出 | P3+P5 | 24 条跑通 + ≥1 严格验证 TS |
| M-D 规模化 | P6 | 1000 条有效路径 |
| M-E 第一代排序 | P7+P8 | 排序指标 + 独立验证可复现 |

---

## 12. 立即行动

1. **C0 收口**：提交/归档实验层、去重、归档脚本、清临时产物。
2. **G1→G2 边界合同** + 24 条人工复核放行。
3. **冻结执行接缝**（协议 + xtb 后端 + 等价证书）。
4. **启动 R0/R1**（量化非驻点、加驻点接受门）。
