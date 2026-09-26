# G1 补全技术方案：TS/IRC 原子映射、成断键分析与反应归簇

> 状态：**已实现**（Batches A–E 全部落地并通过 355 项测试；Batch F 全量运行经
> `g1 resolve-map --allow-truth` 执行。实现模块：`pes2ts_core/g1/truth_schema.py`、
> `truth_join.py`、`truth_alignment.py`、`irc_bond_evidence.py`、`graph_payload.py`、
> `p1_truth.py`（唯一 truth 白名单模块）、`reaction_class.py`、`p2_build.py`、
> `p1_verify.py`、`p2_verify.py`、`gate.py`；CLI：`g1 join-audit / resolve-map /
> classify / gate`、`g1 verify --stage {build,p1,p2}`；exit 23 =
> `EXIT_TRUTH_FLAG_REQUIRED`。实测确认的数据事实：TS/IRC 原子行严格按 map 序
> （逐反应核验而非假设）、IRC 布局为 [TS, 分支A, 分支B] 且拼接点由步长离群值
> 定位、少数轨迹相对 ts.parquet 存在刚体旋转（帧识别须 Kabsch 对齐）、IRC 档案
> 无逐帧 EHG、分支-侧方向必须由键模式判定——全分子 RMSD 会因构象噪声系统性误标
> （中途 QA 实测 R_first/P_first 事件误判率 40.4%/1.2% 不对称，改用末端键模式后
> 0%，符合第 3.2 节图结构优先原则）。本文件保留为验收规格与设计依据。）
> 适用项目：PES2TS / Reaction-QM B3LYP-D3/TZVP 数据  
> 目标顺序：**P1 原子映射与成断键 → P2 反应类型分类与归簇 → G1 门控 → G2 路径生成**

## 1. 目标与范围

当前 G1 已能从映射反应 SMILES 和 R/P 几何构建反应变化文档、反应中心、索引表和统计报告。下一版 G1 要补上 TS/IRC 参照标注，使每条可用反应拥有：

1. 可追溯且唯一的**规范路径原子映射**：R/P 映射号、R/P 几何原子行、TS/IRC 原子行之间有明确的对应表。
2. 映射空间内的成键、断键、键级变化及氢迁移记录，并由 IRC 上的几何变化核验。
3. 基于已核验的反应中心和局部环境生成稳定、可复现的分层反应类别和簇编号。
4. 对所有无法判定的记录保留失败类型和证据，不用任意候选填充结果。

本方案把 TS/IRC 用于建立参考映射和标签。**G2 的普通路径生成输入仍不得包含 TS/IRC 坐标、能量或由其直接导出的路径帧特征。**P1 输出的原子映射、反应编辑和分类标签可以作为反应元数据；若未来要研究“已知 TS/IRC 条件下”的路径生成，应单独标记为 oracle 实验，不能与端点驱动任务混报。

## 2. 当前状态与需要继承的项目事实

项目中的真值资产已隔离：`data/ground_truth/ts.parquet` 保存 TS 原子序数、坐标、E/H/G、总电荷、自旋多重度和映射反应 SMILES；`data/ground_truth/irc_index.parquet` 是形状索引，完整轨迹仍在隔离的 IRC HDF5 中，由审计读取器按反应读取。当前 manifest 记录 TS 199,890 行、IRC 索引 199,890 行、无 TS 缺失；inventory 有 199,217 行。因此必须用反应 ID 做交集核对，不能仅凭总行数声称每一条 G1 记录都能匹配真值。

现有 G1 全量产物记录 199,217 条，其中 198,503 条通过旧版结构检查、714 条被拒绝。索引候选统计为 unique 5,442、ambiguous 192,638、truncated 1,137。这里的 ambiguous 是索引匹配候选状态，不应直接当作错误键变化数。新版 P1 要将这些候选进一步按“路径可区分”“对称等价”“反应中心不一致”“候选被截断”等原因细分。

现有设计明确禁止 G1 读取隔离真值；新增 P1 会改变这一边界，须一并更新 README、实现方案、静态 truth guard 和 CLI 合同。当前 IRC 索引构建把 `ts_index` 写为 0；新版必须先用 TS 坐标、IRC 坐标和能量验证其含义，不能把该常量当作数据事实。

本方案中的行数来自项目现有 manifest 与 G1 报告，实际编码前应重新读取 manifest、校验 SHA-256，并按 ID 复核。Reaction-QM 数据论文说明其 B3LYP TS 经振动分析与 IRC 验证，且提供完整 IRC 轨迹；这些数据适合作为参考标注，但不能替代本项目的逐条一致性检查。

## 3. 核心术语与判定约定

### 3.1 “唯一映射”

验收目标定义为：**每条进入 G2 的反应有且只有一份确定、可重现的规范路径映射。**这不等于化学上所有原子都天然可区分。对于严格对称等价原子，如果候选置换不改变反应编辑、路径几何或反应中心，按固定 canonical rule 选择一个代表，并保存其 symmetry-equivalence 信息。若候选导致不同反应中心或不同成断键事件，IRC 也不能消歧，则状态为 unresolved，不进入 G2。

### 3.2 离散键变化与路径证据

- R/P 映射反应图是离散键编辑的定义来源：formed、broken、order_changed、hydrogen_migration。
- TS/IRC 提供映射候选的路径级证据：反应中心原子对的距离变化、端点连通性、TS 邻近结构及路径连续性。
- TS 单帧上的一个距离不直接决定键级。IRC 的距离曲线用于支持或质疑端点图给出的事件，不用一个全元素统一阈值取代化学图。
- “成键/断键分析”输出结构事件；“同步/异步、路径方向、反应阶段”等是独立的路径描述标签。

### 3.3 真值和预测的边界

P1 生成的是参考标注层。任何使用 TS/IRC 坐标或能量的判断都必须带 `truth_assisted=true` 和来源摘要。G2 端点驱动任务只能读取允许的映射/键编辑元数据，不得读取 oracle 几何、IRC 帧、由其拟合出的距离阈值或路径阶段标签作为模型输入。

## 4. 目标数据流

```text
inventory + reaction SMILES
          │
          ├── 旧 G1 R/P 图解析与候选原子索引
          │
TS geometry + IRC trajectories ── audited truth accessor
          │
          ▼
P1.0 ID / 原子 / 电荷 / 自旋覆盖审计
          ▼
P1.1 R/P ↔ TS/IRC 原子顺序候选对齐
          ▼
P1.2 对称等价归并、规范映射、未解决状态
          ▼
P1.3 映射键编辑 + IRC 路径核验
          ▼
P1 产物冻结与覆盖报告
          ▼
P2.1 规范反应编辑签名
          ▼
P2.2 反应中心/邻域层级分类与归簇
          ▼
P2.3 IRC 路径特征侧标签 + 簇验证
          ▼
G1 release gate → G2 endpoint-only PATH
```

建议在现有 `pes2ts_core/g1/` 基础上新增独立模块，例如：

- `g1/truth_join.py`：核对 inventory、TS、IRC 索引的 reaction_id 和基本字段。
- `g1/truth_alignment.py`：候选 R/P—TS/IRC 原子顺序对齐与 canonical mapping。
- `g1/irc_bond_evidence.py`：按映射计算成断键距离曲线、路径一致性和质量标记。
- `g1/reaction_class.py`：构造 canonical edit signature、层级类别和簇编号。
- `g1/p1_verify.py`、`g1/p2_verify.py`：分别验证 P1、P2 产物与门槛。

这些是建议的代码分层，不要求一次性按文件名实现。旧版 G1 产物应保留作基线；新 schema 使用新版本号和独立路径，避免静默覆盖旧结果。

## 5. P1：原子映射和成断键分析

### P1.0 建立 ID 覆盖清单

以 `reaction_id` 为唯一连接键，逐条连接：

1. `inventory.parquet`；
2. `ts.parquet`；
3. `irc_index.parquet`；
4. 按需读取的单反应 IRC HDF5 轨迹；
5. G1 反应变化文档及源 reaction SMILES。

检查项：

- 每张表的 ID 唯一性、缺失 ID、额外 ID、重复 ID。
- TS `atomic_numbers` 长度、坐标形状、IRC `n_atoms`、R/P 原子总数是否一致。
- TS、R/P 结构和映射 SMILES 的元素、总电荷、多重度是否相容。
- TS 行中的 `reaction_smiles` 与 inventory/CSV 的反应定义是否一致。
- 记录所有 join 的源文件 SHA-256、dataset version 和 schema version。

输出 `p1_join_audit.json`，至少包括 `n_inventory`、`n_ts`、`n_irc_index`、`n_joined_all`、各类缺失数、重复数、原子数冲突数、元素冲突数和拒绝原因示例。join 不完整的反应不得进入后续“已解决”计数。

### P1.1 验证 TS 帧和 IRC 轨迹

IRC reader 当前可以逐条读出 `(n_frames, n_atoms, 3)` 坐标及可选的逐帧 EHG；运行时必须单反应处理，不把全库约 2,300 万帧载入内存。

实施顺序：

1. 对一个分层样本，逐帧比较 IRC 坐标与 TS 几何；对齐刚体旋转/平移后计算 RMSD，结合能量识别 IRC 中的 TS 对应位置。
2. 明确 IRC 帧的顺序、TS 帧索引、两侧端点帧和方向约定；禁止默认固定 `ts_index=0`。
3. 用端点元素组成、分子图和坐标对齐确认两个 IRC 分支分别对应 R 侧和 P 侧；允许反向轨迹并显式记录 `irc_orientation`。
4. 验证 IRC 全程原子数和元素顺序不变；遇到非有限坐标、能量缺失、断帧或无法匹配的端点，写入 typed status。

大规模处理采用流式/单反应方式，只持久化汇总特征和必要的质量摘要，不复制整套真值帧到普通 interim 目录。

### P1.2 求解 R/P 到 TS/IRC 的映射

现有 G1 已有映射 SMILES 解析、R/P 组分候选匹配和本地 XYZ 索引表。新版应复用这些候选，不把“首个 graph match”直接作为最终路径映射。

对每个 reaction_id：

1. 从映射 R/P 图确定每个 reaction map number、分子组分、图原子、组件局部 XYZ 行及全局 R/P 行。
2. 为 R/P 组分匹配枚举候选，硬约束至少包括元素/原子序数、原子数、组分结构、映射号合法性、可用的电荷/同位素/立体信息。
3. 联合匹配两侧原子与 TS/IRC 原子行，不独立逐个原子贪心匹配。多组分交换、同元素重复、氢原子和反应物/产物端点方向都要纳入候选问题。
4. 用刚体对齐后的 endpoint geometry residual、IRC 端点几何、映射后反应中心的距离演化和元素序列一致性筛选候选。几何评分只在硬约束匹配后使用；不能让低 RMSD 覆盖图结构冲突。
5. 将候选按映射后的 edit signature、反应中心和 IRC 距离轨迹等价性分组。完全相同或仅相差严格对称置换的候选可折叠为一个等价类。
6. 对每个 resolved 等价类用稳定 canonicalization 规则选择代表：输入 reaction_id、源行顺序和 RDKit 枚举次序不得改变结果。保存原始候选数、等价类数、选择理由和规则版本。
7. 仍有多个非等价候选时输出 unresolved；不得用 `candidates[0]`、任意 map 排序或最近距离单独强行消歧。

推荐映射状态：

| 状态 | 定义 | 是否进入 G2 |
|---|---|---:|
| `resolved_unique` | 只有一个满足约束的映射 | 是 |
| `resolved_symmetry_collapsed` | 多个候选属于同一对称等价类，已规范化 | 是，保留等价标志 |
| `unresolved_reactive_center` | 非等价候选导致反应中心/事件不同 | 否 |
| `unresolved_truncated` | 候选枚举达到上限，无法证明完整 | 否 |
| `ts_irc_endpoint_mismatch` | TS/IRC 端点无法对应到给定 R/P | 否 |
| `atom_or_element_mismatch` | 原子总数或元素序列不守恒 | 否 |
| `missing_truth_join` | inventory 无相应 TS/IRC 记录 | 否 |
| `source_structure_mismatch` | 源结构与 reaction SMILES 不能一致匹配 | 否，保留原证据 |

候选匹配必须能证明搜索完备，或明确标记 `truncated`。可优先对图 automorphism/orbit 做归并，避免为大量对称置换全部物化。

### P1.3 生成键事件并用 IRC 核验

对选定映射，将 R/P 侧的键表转成 reaction-map 空间的键集合，计算：

- `formed`: R 无键、P 有键；
- `broken`: R 有键、P 无键；
- `order_changed`: 两侧同一原子对存在键但键级不同；
- `hydrogen_migration`: 映射氢的成键伙伴变化；
- `reaction_center`: 上述事件关联原子及可选邻域。

对每个变化原子对 `(i,j)` 计算 IRC 距离曲线 `d_ij(frame)`，并记录 R 端、TS 邻近帧、P 端的距离摘要、曲线方向、分支一致性及数据质量。R/P 键变化是事件定义；IRC 曲线是独立核验信号。

建议核验输出之一：

- formed：R 侧不存在/距离较远，沿 IRC 向 P 端缩短并形成合理 endpoint 键；
- broken：R 端存在键，沿对应路径增大并在 P 端断开；
- order change：端点键级变化有映射图支持，距离轨迹与该变化不矛盾；不能仅凭距离断定单双键或芳香键级；
- H migration：氢从 `from_map` 指向 `to_map`，两条相关距离曲线应显示伙伴交换趋势；
- 方向不确定、曲线异常、末端未到达 endpoint 等情况均记录为 `weak` 或 `mismatch`，不可伪装成通过。

P1 记录既保存 P1 计算的键事件，也保存旧 G1/预览 counter 的交叉核对结果。若 TS/IRC 支持的几何证据与映射 SMILES 编辑矛盾，输出冲突状态和可定位证据，不静默覆盖输入。

### P1.4 P1 输出 schema 建议

建议写入新的、版本化目录，例如：

```text
data/interim/g1_truth/p1_mapping/<shard>/<reaction_id>.json
data/interim/g1_truth/g1_p1_summary.parquet
data/manifests/g1_p1_manifest.json
data/manifests/g1_p1_coverage.json
data/manifests/g1_p1_join_audit.json
```

每反应 JSON 至少包含：

```json
{
  "schema_version": "g1_p1_truth_v1",
  "reaction_id": "RXN_...",
  "sources": {
    "dataset_version": "...",
    "inventory_digest": "...",
    "truth_manifest_digest": "...",
    "mapping_algorithm": "...",
    "mapping_config_digest": "..."
  },
  "status": "resolved_unique | resolved_symmetry_collapsed | ...",
  "mapping": {
    "map_to_atoms": [
      {"map": 1, "element": "C", "component": "R0", "component_local_index": 0,
       "rp_global_index": 0, "ts_irc_index": 0}
    ],
    "n_candidates": 1,
    "n_equivalence_classes": 1,
    "selected_class": 0,
    "resolution_reason": "..."
  },
  "bond_events": {
    "formed": [], "broken": [], "order_changed": [], "hydrogen_migration": []
  },
  "irc_validation": {
    "ts_frame_index": 0,
    "orientation": "R_to_P | P_to_R | unresolved",
    "endpoint_match": "pass | weak | fail",
    "event_support": [],
    "quality_flags": []
  },
  "reaction_center": {"core_maps": [], "shell1_maps": []},
  "validation": {"status": "pass | unresolved | rejected", "reasons": []}
}
```

上述字段是接口建议，具体序列化方式可用现有项目的稳定 JSON/Parquet 规范。不要把完整 TS/IRC 帧数组放进此 JSON。

## 6. P2：反应类型分类与归簇

### P2.1 分类输入

P2 只读取 P1 状态为 `resolved_unique` 或 `resolved_symmetry_collapsed` 的规范映射和键事件。类别 signature 不得包含原始 reaction map number、reaction_id、文件行号或 RDKit 候选枚举次序；否则同一种化学反应仅因编号差异就被拆成多个簇。

### P2.2 建议的分层类别

| 层 | 分类对象 | 用途 |
|---|---|---|
| `L0 edit_family` | 成键/断键/键级变化/H 转移的事件组合、事件数 | 快速分层统计和抽样 |
| `L1 center_template` | reaction center 的标记图；原子元素、电荷/价态及成断键类型 | 精确反应中心归簇 |
| `L2 context_r1` | L1 加反应中心邻域一层原子/键环境 | 区分局部化学环境 |
| `L3 context_r2` | L1 加两层邻域；必要时细分立体化学/芳香性 | 高特异性转化模板 |
| `family_labels` | 加成、取代、消除、重排、环化/开环、碎裂、氢转移、其他/未知 | 面向人的可读标签，可多标签 |
| `pathway_labels` | IRC 展示的路径方向、同步/异步程度、阶段/异常 | 描述已知路径，不代替结构类别 |

精确簇使用 canonical labeled graph/signature 的等价性分组，簇 ID 由 `taxonomy_version + level + canonical_signature_hash` 确定。更宽的邻域层级是父子关系，而不是彼此独立的一组无含义编号。

SynTemp 展示了从 ITS/反应中心到扩展邻域的层级模板聚类路线；其规则数量和覆盖率是特定数据集结果，不作为本项目 KPI。DRFP 可作为额外的 reaction-similarity 指纹，用于近邻检索或探索性分析；它不能代替 P1 映射、精确反应编辑或可解释类别。

### P2.3 分类验证与产物

建议产物：

```text
data/interim/g1_truth/reaction_classes/<shard>/<reaction_id>.json
data/interim/g1_truth/g1_reaction_class_summary.parquet
data/manifests/g1_p2_manifest.json
data/manifests/g1_p2_coverage.json
data/manifests/g1_cluster_report.json
```

报告至少包括：

- 每层簇数、每簇大小、singleton 比例、覆盖率；
- 各 edit family / 可读标签的数量和占比；
- P1 状态按簇的交叉统计；
- 形成/断裂/键级变化事件的类别分布；
- 各簇中的 IRC 证据质量及人工抽查结果；
- 大簇、稀有簇、未分类簇的代表反应；
- 同一稳定 signature 是否因 map relabeling、分子组分顺序或反应方向变化意外拆簇。

G0 已冻结的数据划分不得因 P2 静默重排。可以报告反应簇跨 train/valid/test 的分布，供后续拆分泄漏评估和分层抽样使用；如需按簇重新划分，另立决策并重新做重复/泄漏审计。

## 7. 真值隔离、CLI 和审计改动

现有项目仅允许 `pes2ts_core.g0.truth.truth_reader` 访问真值，且 G1 的 truth guard 禁止读取 TS/IRC。新方案应采用最小范围的例外：

1. 通过现有 audited accessor 读取 TS/IRC，禁止新模块自行拼接 HDF5 路径或直接打开隔离 HDF5。
2. 将读取真值的逻辑限制在独立 P1 标注模块；在 truth guard 中只对这个明确模块开放白名单，其他 `g1`、`g2` 和 CLI 代码仍受保护。
3. 每次成功读取记录 reaction_id、调用模块、函数和时间；manifest 写入真值文件摘要与算法/配置版本。
4. P1/P2 普通产物仅保存映射、事件、审计摘要和簇标签，不复制原始真值坐标、完整 IRC 或原始能量轨迹。
5. CLI 可扩充为显式子命令，例如 `g1 resolve-map`、`g1 classify`、`g1 verify --stage p1|p2`；具体命名遵守当前 argparse 模式。无 `--allow-truth`/明确标注配置时，不进行 truth-assisted 计算。
6. 静态 guard 与测试要验证：G2 路径生成包不能导入 truth reader、不能访问 `ground_truth/sources`，G2 输入 schema 不含 TS/IRC 字段。

## 8. 编码顺序与检查清单

按以下顺序实现，先小样本验证接口，再全量构建：

### Batch A：schema、数据连接与状态枚举

- 定义 P1/P2 schema version、错误码、状态枚举、稳定排序和 source digest 字段。
- 实现 join audit；覆盖缺 ID、重复 ID、原子数冲突、缺 TS、缺 IRC 等 fixture。
- 验收：任何分母差异都能在 manifest 中按原因解释。

### Batch B：单反应 TS/IRC 对齐原型

- 选纯成键、纯断键、键级变化、H 转移、多组分、对称候选、反向 IRC 等代表样本。
- 核对 TS frame index、IRC 方向、端点匹配、原子顺序守恒。
- 验收：每条样例生成可人工检查的映射表与轨迹距离图/摘要；不得只报告“匹配成功”。

### Batch C：规范映射求解器

- 枚举或按 graph automorphism orbit 折叠候选；加入元素/图/几何/路径约束。
- 对称映射 canonicalization 与非等价反应中心歧义分别处理。
- 验收：候选输入顺序打乱不改变规范输出；超限候选必为 `unresolved_truncated`。

### Batch D：键事件与 IRC 核验

- 在 reaction-map 空间生成 formed/broken/order_changed/H migration。
- 计算逐事件 IRC 距离曲线摘要、端点一致性和反例状态。
- 验收：既有 G1 counter、映射反应图和 P1 结果可对账；冲突有原因及证据。

### Batch E：层级分类与归簇

- 实现 map-number-independent canonical signature、层级上下文模板和可读规则标签。
- 另算可选 DRFP 相似度；明确它只用于相似性，不替代结构簇。
- 验收：原子 map relabeling、组分序重排、候选顺序变化不改变类别；改变 reaction center 或指定上下文会按规则改变层级簇。

### Batch F：全量运行、报告与 G1 门控

- 先 dry-run join audit，再对全量 inventory 流式解析 TS/IRC；保存确定性 manifest、覆盖报告、失败 ledger 和访问日志。
- 按每个主要簇、罕见簇、映射歧义状态抽样人工复核。
- 全量重复运行同输入/配置，验证产物稳定；真值读取日志可审计。
- 只有通过第 9 节门槛后才标记 G1 完成。

## 9. G1→G2 验收门槛

### P1 必须通过

- inventory 分母中的每条记录都落入一个明确状态；TS/IRC join 数量和所有缺失均可解释。
- 所有 G2 eligible 反应都有一份规范路径映射；每个映射覆盖所有反应原子，并通过原子数/元素核对。
- 不存在静默使用首个候选、静默截断、或未记录的反应中心歧义。
- 成断键结果由映射反应图计算，IRC 证据完成逐条或按明确策略核验；冲突和弱证据单独列示。
- P1 样本覆盖对称原子、多组分、氢迁移、不同键编辑类别和所有失败状态，并通过人工核查。

### P2 必须通过

- 每条 P1 eligible 反应都有可重现的 taxonomy version、层级类别 ID 和状态；不可分类项有明确标签。
- 类别对 map-number relabeling、输入顺序和运行重复保持稳定；簇大小和覆盖率完整报告。
- 每个主要簇及稀有类别通过抽样审阅；结构类别和 IRC 路径标签没有混为一列。
- G0 split 不被静默改变；跨 split 簇分布和近重复情况有报告。

### G2 准入

只有 `resolved_unique` 和 `resolved_symmetry_collapsed` 且 P1/P2 验收通过的 reaction_id 进入 endpoint-only G2。其余记录进入排除/待修复清单并保留总分母。报告应同时列出 eligible 数量与占全量比例，不以“已有完整报告”替代实际门控结果。

## 10. 建议测试用例

最少建立以下 fixture 类别：

- 一个成键、一个断键、成断并存、一个键级变化、氢迁移；
- 两个相同组分可互换的多组分反应；
- 对称原子交换但 reaction center 不变；
- 对称候选导致不同 reaction center，IRC 可消歧与不可消歧两种情况；
- map 标签整体重编号后 edit signature 不变；
- IRC 反向排序、TS 帧不是 0、端点不匹配、候选搜索被截断；
- R/P/TS/IRC 原子数、元素、电荷或 multiplicity 冲突；
- 键变化距离趋势与端点图矛盾、非有限坐标或能量缺失；
- split manifest 不变、truth access audit 有记录、G2 不能读取原始真值。

上述是后续编码的测试计划，不代表当前已新增或已运行这些测试。

## 11. 参考方法与依据

- Reaction-QM 论文：B3LYP 反应、TS、完整 IRC 及其数据结构；[Scientific Data 论文](https://doi.org/10.1038/s41597-026-07325-w)。
- SynTemp：反应中心与扩展邻域的分层反应模板聚类；可借鉴算法形态，论文结果不可直接视为 PES2TS 的预期覆盖率；[SynTemp 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11938280/)。
- DRFP：由两侧分子局部子结构差异构造反应指纹，适合相似性检索/辅助分群，不替代精确 atom-mapped edit template；[DRFP 论文](https://doi.org/10.1039/D1DD00006C)。
- EC-BLAST：按 bond change、reaction center、reaction structure 多层比较反应相似度；其 enzyme-specific 分类范围不直接等同于 Reaction-QM，但指纹分层思路可参考；[EC-BLAST 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4122987/)。

## 12. 相关项目文件

- 当前总体路线与 G1/G2 边界：`PES2TS_逐步实施与验证方案.md`
- 当前 G1 行为、schema 与残余歧义：`README.md` 的 G1 章节
- 当前真值 manifest：`data/manifests/truth_manifest.json`
- 当前 inventory manifest：`data/manifests/inventory_manifest.json`
- 当前 G1 coverage：`data/manifests/g1_coverage.json`
- 审计真值 accessor：`pes2ts_core/g0/truth/truth_reader.py`
- IRC 读取器：`pes2ts_core/g0/irc_reader.py`
- 现有 G1 图解析、索引及键变化：`pes2ts_core/g1/parse.py`、`index_map.py`、`bond_changes.py`、`document.py`、`build.py`
- truth access 静态保护：`pes2ts_core/utils/truth_guard.py`

