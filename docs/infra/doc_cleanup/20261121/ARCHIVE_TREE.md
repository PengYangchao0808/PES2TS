# PES2TS 开发档案树（Development Archive Tree）

> 生成：2026-11-21；角色：task-6（档案架构设计）；只读仓库，未修改/移动/删除任何既有文件。
> 输入证据：`docs/infra/doc_audit/20261121/doc_inventory.csv`（253 条）、`DISPOSITION.csv`（253 条）、`FAMILIES.md`（21 家族/53 成员）、`ANOMALIES.md`（32 条：P0 2 / P1 10 / P2 20）、`classification_mapping.csv`、`docs/README.md`、`docs/plans/PES2TS_文档修订台账.md`、`docs/archive/README.md`、各战役终报、根 `README.md`、`git log --date=short`（331 提交，HEAD `2f1d3ba`，2026-11-20 21:08 +0800）。
> 产物三件：本文件（总纲+分期+模板+17 份成稿）、`ARCHIVE_MAP.csv`（253→条目映射）、`ARCHIVE_TREE_OUTLINE.txt`（拟建 `docs/archive_tree/` 文件树）。
> 红线：不虚构成果/指标；负结果一律保留且不美化；每条目必须有路径级证据指针；本档案树是**内容压缩层**，不是目录搬家层。

---

## 0. 总纲

### 0.1 定位

PES2TS 在 2026-09-25 至 2026-11-21 的 58 天里产生了 253 条入库文档、331 次提交、21 个版本家族、6 组状态自相矛盾的家族、32 条治理异常。**信息量不是问题，问题是"为什么这么做"只存在于当时的上下文里**：方案写的"要做什么"、报告写的"做成了什么"、而"为什么否掉另一条路/为什么这个指标算失败"散落在 30 余份战役终报与台账增量里。

开发档案树要解决的就是这件事：把全部开发历史压缩为**按时间分期的档案条目**，每一条回答五个问题——

1. 这一阶段的目标是什么（一句话）；
2. 核心思路是什么（3–5 条，可复用的思维方式）；
3. 关键决策是什么（关联 ADR 编号与生效日期）；
4. 做成了什么（量化指标，只写有落盘证据的数）；
5. **没做成什么、为什么（负结果与限制，这是本项目最有价值的资产）**。

目标读者与用法：

| 读者 | 路径 | 收益 |
|---|---|---|
| 未来的作者（同一团队） | 先读 `INDEX.md` 时间轴 → 最近一期的条目 → 需要细节再跳原报告 | 避免重复踩已判定负结果的坑（V4.7、V5.1a、V5.2-R、V5.4.2 等） |
| 新加入者 | 按第 1→7 期顺序读条目正文（每期 1–4 条，共 17 条） | 半天建立完整项目心智模型，不必读 253 份文档 |
| 外部评审/合作方 | `README.md` + 每期「发表级结论」+ `90_evidence/negative_results.md` | 看到诚实的失败记录与证据链，而不是只有成功叙事 |

### 0.2 与既有四个目录的边界（本档案树不搬家）

| 目录 | 定位 | 档案树与它的关系 |
|---|---|---|
| `docs/archive/` | 版本快照（7 个时点，只读） | 档案条目引用其**时点**作为分期证据，绝不搬入 |
| `docs/superseded/` | 已取代文档存放区（2026-11-20 建立，4 份） | 档案条目引用其**取代链**，不重复收录 |
| `docs/reports/` | 只读证据（内容永不改） | 档案条目引用其**量化结论**，不复制正文 |
| `docs/archive_tree/`（拟建） | **本任务新增**：压缩后的时间线叙事层 | 只新增条目文件，不改任何既有文件 |

处置原则（与 `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md` 的 B0 一致）：**档案化 ≠ 搬迁**。本树只写"从哪些文档中抽取了什么"，原文档全部原地保留（`ARCHIVE_MAP.csv` 的 `current_action` 列沿用 `DISPOSITION.csv` 口径：keep 125 / retain-readonly 109 / banner 16 / move 3）。

### 0.3 使用方式

- **按时间读**：`INDEX.md` 给出 7 期 17 条目 + 时点、主导 ADR、关键负结果索引。
- **按主题读**：`INDEX.md` 的主题交叉索引（数据/教师/控制/评估/治理 5 条线）指向分散在不同期的条目段。
- **按证据读**：每条目 §证据指针 给出**原路径**（必要时给报告内的 § 号）；不要复制原文，链接用反引号 inline code（后续会复制进 `docs/` 受链接门禁扫描）。
- **写新条目**：只用 `99_meta/entry_template.md`；新条目必须登记 `CHANGELOG.md`，且不得修改已冻结条目（只可追加"后续指针"）。

### 0.4 存放位置建议与命名

建议落位 `docs/archive_tree/`（与 `docs/archive/`、`docs/superseded/` 并列，不进入 `docs/reports/` 的只读证据区）：

```
docs/archive_tree/
├── README.md                 # 从本文件 §0–§3 收敛（总纲+模板+用法）
├── INDEX.md                  # 分期×条目总索引 + 主题交叉索引
├── CHANGELOG.md              # 档案树自身变更记录（append-only）
├── 00_crosscut/              # X0 横切常青层指针
├── 10_p1_.../ ... 70_p7_.../ # 7 个分期目录，各含 E## 条目文件
├── 90_evidence/              # 证据指针层（索引 csv、负结果总表、指标账本）
└── 99_meta/                  # 模板与映射表副本
```

条目文件名：`E##_<期号>_<主题>_<YYYYMMDD>.md`（期号 p1–p7；主题 ≤12 字）；条目编号 `E01`–`E17` 一经分配不复用；横切层编号 `X0`。**不按版本号命名目录**（版本号会随取代链漂移，时间不会）。

### 0.5 证据纪律（写条目时必须遵守）

1. **只写文档事实**：指标必须能在原报告的状态行/结论段/主表找到；找不到就写"未留档，不引用"。
2. **负结果不豁免**：判定失败的实验、被否的方案、被裁剪的 run、被阻塞的门，必须进入条目「负结果」段，且不得用"待优化"等模糊词替代。
3. **状态与时间双轴**：条目的"当时状态"按文档状态头抄写；若事后被取代（如 V5.2 的 Stop 语义被 ADR-0030 撤回），用"**当时判定 → 事后修正**"两行并列，不覆盖当时结论。
4. **区分完成层级**：`implemented / fixture_passed / smoke_passed / scientifically_validated` 四级（24H 战役确立的词汇），不得把 fixture 通过写成能力验收。
5. **证据指针格式**：`docs/reports/...md`（+§号）或 `git log -- <path>`；不写 commit hash 除非该 hash 已在文档正文出现（如 V5.4.2 报告的提交链 `d94ad73`…）。

---

## 1. 分期方案（7 期 / 17 条目 / 1 横切层）

分期依据三种独立证据的交集：**文档声明日期分布**（`doc_inventory.csv` 的 `declared_date`/`git_first_add_date`）、**git 提交日分布**（331 提交）、**里程碑状态头**（`Executed-Frozen`/`Superseded`/ADR 接受日期）。三者一致的断点才作为期边界。

**git 提交日分布（决定性证据）**：2026-09-25 18、09-26 21、09-27 2、09-28 25、09-30 9、10-01 31、10-03 6、10-05 49（单日峰值）→ **10-06…10-17 无提交** → 10-18 17、10-19 14、10-20 43、10-21 37、10-22 19、10-23 4、10-24 3、10-25 4、10-26 1、10-27 4、10-29 1、10-30 6、10-31 5、11-06 6 → **11-07…11-19 无提交** → 11-20 6。
两段空档说明：**文档并非按提交时序落盘**——10-06…10-17 的文档在 10-18 批量入库（`git_first_add_date=2026-10-18`），11-07…11-19 无产出。因此分期以**里程碑/声明日期**为准，git 提交日只作"工程活动强度"旁证，绝不用 `git_first_add_date` 反推创作日期。

| 期 | 名称 | 时间范围 | 条目 | 边界证据（可复核） |
|---|---|---|---|---|
| P1 | 基础期 G0–G2 | 2026-09-25 → 2026-10-04 | E01、E02 | 首提交 `a7c4ed7`（09-25）；G1 实现方案声明 09-26；`20261005_pre_pesn` 快照时点；10-04 后无 G2.0 新增方案 |
| P2 | PES-N 立项期 | 2026-10-05 → 2026-10-14 | E03、E04、E05、E06 | 10-05 单日 49 提交 + ADR-0010 Accepted；快照 20261006/07/08；10-15 起 v4 主线 |
| P3 | V4.x 训练实验期 | 2026-10-15 → 2026-10-18 | E07、E08 | v4 方案 10-15（ADR-0021）；v4/v4.1 终报 10-17；V4.7 三种子 10-18 判定负结果 |
| P4 | V5 冲刺与 24H 战役期 | 2026-10-18 → 2026-10-21 | E09、E10、E11 | V5 冲刺方案 10-18；24H 授权 10-20、终报 10-21；ADR-0030/0031 Accepted 10-21 |
| P5 | V5.3-MP 规模联合训练期 | 2026-10-21 → 2026-10-27 | E12 | ADR-0031 任务合同生效 10-21；S0 10-22；S3 10-23；S4/规模收益 10-27 |
| P6 | V5.4.x 三阶段期 | 2026-10-27 → 2026-11-06 | E13、E14、E15 | V5.4.1 WP0 10-27 / 执行 10-29；V5.4.2 执行 10-31；V5.4.3 执行 11-06（全部 `Executed-Frozen`） |
| P7 | 治理归档与公开化期 | 2026-11-20 → 2026-11-21 | E16、E17 | 11-20 6 提交（LICENSE/名单/归档）；11-21 归档方案 v2.0 + 253 条审计 |
| — | 横切常青层 | 全期有效 | X0 | 宪法、AGENT、阶段定义、G–V 映射、CODE_MAP、README、台账（随期更新，不随期归档） |

> 关于"当前"：截至 2026-11-21，**没有活跃训练战役**。最后一战力 `V5.4.3` 于 11-06 冻结，11-20/21 全部动作是仓库治理。因此 P7 不是"新研究期"，而是**工程收口期**；下一研究期（V5.4.4+/V6 外层物理松弛）在 `V5.4.3` 报告的交接段列为计划，**尚未开始**（无方案文档、无 registry executed 条目）。

---

## 2. 档案条目模板

复制以下模板新建条目（字段不得增删；确需扩展的写在"附注"）：

```
# E## <标题>（<期号> <期名>）

> 条目标号：E##          时间范围：YYYY-MM-DD → YYYY-MM-DD
> 阶段：G0/G1/G2/G2.T/G3.P/G4.x（代码能力） × V4.x/V5.x（训练战役） × R/X/C 工作流
> 状态：Archived（已冻结）/ Living（横切常青，仅 X0 允许）
> 关联 ADR：ADR-00xx（Accepted/Proposed，日期）；无 ADR 的写"无（见方案 §x）"
> 数据/代码/模型版本指针：commit 或 config/registry 条目或数据集 SHA（只写文档已声明者）
> 证据根：docs/...（报告/方案/合同；必要时 §号）、outputs/...（只写指针，不引用大文件）

## 目标（一句话）

## 核心思路（3–5 条，可复用）

## 关键决策（决策 → 日期 → 证据）

## 主要成果与量化指标（只写落盘证据；区分 implemented/fixture/smoke/scientifically_validated）

## 关键负结果（不得美化；写明"判定口径 + 证据"）

## 限制与未解问题（当时未做/被裁剪/被阻塞）

## 证据指针

## 后续影响（指向下一期条目/取代者；冻结后只可追加）
```

填写规则：①指标带口径（哪个 dev 集、哪个 seed、哪个门）；②负结果写"判定规则 + 数值 + 结论"，不写"表现欠佳"；③跨条目引用用 `E##`；④**禁用 markdown 链接**，文件路径用反引号 inline code。

---

## 3. 分期详解与成稿（7 期 / 17 条目）

### 第 1 期 · 基础期 G0–G2（2026-09-25 → 2026-10-04）

一句话目标：把 Reaction-QM 原始数据变成防泄漏、可复现的数据底座，并让"图论选点 → 编译 → 真实 QC 执行 → 回收验证"这条 Classic 主链第一次端到端跑起来。

核心思路：①先立契约后写代码（哈希/IO/统一 rejection/manifest 三件套先于业务）；②真值物理隔离（teacher 标签/TS/IRC 只能经 audited bridge 读取）；③扫描策略按"成键/断键方向"分层选择，而不是穷举；④多坐标 PES 学习优于单坐标逐步实施（方案层面结论，实现未验收）；⑤每个动作留执行身份与成本账本。

关键技术决策：ADR-0001（统一执行后端接缝，Accepted）、ADR-0002（计算后端统一经 ACP 执行，Accepted）、ADR-0003/0004/0005（G2-AB1/AB2 接口登记、执行前判定器与恢复修补，Accepted）。当期无 G 阶段出口被宣布通过。

主要成果与指标：G0 数据与分裂冻结（Zenodo 采集/校验/manifest 幂等，`feat(g0)` 首日提交）；G1 TS/IRC 映射与反应归簇 Batches A–E 落地并通过 355 项测试；`contracts_v1` 合同字典、ACP 字段映射、数据规范草案落盘；Demo24 首批 ACP 计算与 TS/IRC 算法诊断、多坐标全批计算、多键同步一维扫描问题报告形成（10-01/10-02）；G2.T 真实战役方案与四类终态字段字典成型（10-04，执行归 E02/E04）。

关键负结果：**G1 仍含 `truth_assisted_p1` 来源，不能称端到端 truth-free**（`docs/plans/PES2TS_阶段定义_G0-G3.md` G1 节）；G1 的 Batch F 未列入"已实现"范围（实现方案状态头在 Batch F 处截断，档案只录"未完成"，不补写原因）；多键同步一维扫描存在成体系缺陷（`docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md`）；generation 阶段存在未定位根因（`docs/reports/PES2TS_generation关键问题与根因分析_20261002.md`）。

数据/代码/模型指针：`requirements.txt`（09-25 起）；`data/manifests/`（G0 分裂）；`docs/contracts/PES2TS_contracts_v1.md`。当期原始文档 git 范围：2026-09-25…10-05 的 161 次提交（09-25/26/28/30、10-01/03/05）与 `docs/archive/20261005_pre_pesn/` 快照。

#### 示范成稿 · E01 数据底座与契约 v1（P1 基础期）（完整字段样板）

> 条目标号：E01　时间范围：2026-09-25 → 2026-09-30
> 阶段：G0/G1 代码能力 + X/C 工作流　状态：Archived
> 关联 ADR：ADR-0001（统一执行后端接缝，Accepted）
> 数据/代码/模型版本指针：首提交 `a7c4ed7`（2026-09-25 `chore(scaffold)`）；`requirements.txt`；G0 源清单 `data/manifests/source_manifest.json`（由 `g0 fetch` 生成）
> 证据根：`docs/design/G1_TS_IRC_映射与反应归簇实现方案.md`、`docs/contracts/PES2TS_contracts_v1.md`、`docs/contracts/PES2TS_ACP字段映射_v1.md`、`docs/reports/PES2TS_开发阶段性报告_20260930.md`、`docs/archive/20261005_pre_pesn/`

## 目标
把 Reaction-QM 原始数据变成干净、防泄漏、可复现的数据集，并把 TS/IRC 答案物理隔离；同时把 G0→G1 的接口与拒绝语义固化成合同。

## 核心思路
- 主链顺序：先数据契约与真值隔离，再谈生成能力（"没有合格数据就没有可宣称的能力"）。
- 一切执行留身份：哈希/IO/统一 rejection/manifest 先于业务代码（首日三个提交即此顺序）。
- 图论先验驱动扫描方向（成键/断键、一维优先 vs xTB 二维），而非全枚举。
- ACP 作为唯一外部计算出口（后续 ADR-0002 固化）。

## 关键决策
- ADR-0001 统一执行后端接缝（Accepted，文档声明 2026-10-02）：执行本地化条款与 X1/X2 退出条件后续被 ADR-0002 部分取代。
- G0 分裂冻结 + 真值隔离（`data/ground_truth/*`）写入宪法级红线（R1 真值隔离）。

## 主要成果与量化指标
- G0：Zenodo 六文件采集 → MD5 → `source_manifest.json` 全通过才落盘（`g0 fetch` 语义，10-18 前已可复跑）。
- G1：TS/IRC 映射与反应归簇 **Batches A–E 全部落地，355 项测试通过**（实现方案状态头）。
- 合同 v1 字典 + ACP 字段映射 + 数据规范与 ACP 对接优化方案（09-30）：接口与字段口径第一次形成可引用文本。

## 关键负结果
- **G1 含 `truth_assisted_p1` 来源，不能声称端到端 truth-free**（`docs/plans/PES2TS_阶段定义_G0-G3.md` §2 G1）。
- G1 Batch F 未进入已实现范围（状态头截断于 "Batch F 全量运行经…"）：档案只记"未完成"，不补写结论。

## 限制与未解问题
- G0 完成状态只覆盖历史数据底座；跨来源状态/相容转移资格"尚未由历史 G0 完成状态覆盖"（阶段定义原文）。
- 355 项测试是单元/回归证据，**不等于** G1 出口条件（边界合同 + 24 例复核）闭合。

## 证据指针
- `docs/design/G1_TS_IRC_映射与反应归簇实现方案.md`（状态头）
- `docs/contracts/PES2TS_contracts_v1.md`、`docs/contracts/PES2TS_ACP字段映射_v1.md`
- `git log --oneline a7c4ed7.. -- data/ docs/contracts/`（2026-09-25…09-30 共 18+21+2+25+9 次提交）

## 后续影响
- G1 边界合同/24 例复核未闭合 → 转入 E02（G2.T 战役）与 E03（PES-N 阶段重定义）；G0 数据的"历史完成状态"在 E03 被明确限定适用范围。

#### E02 G1 映射与 G2 Classic / Demo24 / 24 例战役准备（P1 基础期）

> 条目标号：E02　时间范围：2026-09-30 → 2026-10-04　状态：Archived
> 关联 ADR：ADR-0002（计算后端统一经 ACP，Accepted）、ADR-0003/0004/0005（G2-AB1/AB2 接口登记/判定器/恢复修补，Accepted）
> 数据/代码/模型版本指针：G2.T 合同字段字典 `docs/contracts/PES2TS_G2T_战役合同字段字典_v1.md`；扫描策略选择器设计 `docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`
> 证据根：`docs/reports/PES2TS_Demo24_*.md`（6 份，10-01/10-02）、`docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md`、`docs/plans/PES2TS_G2T真实战役执行方案_20261004.md`、`docs/reports/PES2TS_开发阶段性报告_20260930.md`

## 目标
让 Classic 主链第一次可审计地跑通 24 例 Demo：多坐标全批计算 → 四类终态（rejected_typed / partial_prefix / complete_path / validated）→ 失败与成本账本。

## 核心思路
- 24 例 Demo 作为"可审计终态"的最小可信战场，而不是成功率战场。
- ACP 统一执行：所有后端（XTB_PATH / ORCA scan/constraints/NEB / continuation）经 ACP，本地执行接缝退役（ADR-0002）。
- 类型化拒绝（typed rejection）与质量分层（完整路径/部分前缀/拒绝）分开记账。
- 多坐标 PES 学习路线作为生成能力候选，先出数学方案再谈实现。

## 关键决策
- ADR-0002（Accepted，2026-10-03）：统一 ACP 后端，ADR-0001 的执行本地化条款被部分取代；实施按"X1′–X5′"（`docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md`）推进。
- ADR-0003/0004/0005：G2-AB1/AB2 接口登记、执行前判定器 v2 与三项恢复/适配修补在 S2 真实执行前登记并锁定。

## 主要成果与量化指标
- Demo24 系列 9 份报告落盘：首批 ACP 计算、端点与 TS 几何核查、TS/IRC 算法诊断、多坐标全批计算、第二轮修改与实测、多键同步一维扫描问题报告、generation 根因分析、对话/任务总结。
- G2.T 战役合同字段字典 v1（10-05 前）：四类终态、成本账本字段、typed rejection 词表。
- 24 例真实战役方案（`docs/plans/PES2TS_G2T真实战役执行方案_20261004.md`）冻结 S1–S5 与 D1–D5 交付口径（**执行结果归 E04**）。

## 关键负结果
- 多键同步一维扫描存在缺陷清单与根因（10-01 报告）：24 例在第二轮修改后仍需第三轮，说明"一维扫描 + 单坐标"不足。
- Demo24 的 9 份报告中，端到端 validated=0 是当时已知事实（G2 退出条件要求 ≥1 条非参考种子严格目标 TS，**当期未达成**）。
- 本地 xTB runner 删除与全后端 ACP 化未完成（ADR-0002 实施注记，10-03 时点）。

## 限制与未解问题
- G2/G2.T 仍是"进行中（round0–round3）"；G2 出口四项条件未闭合。
- 多坐标 PES 学习仅数学方案（`docs/reports/PES_generation_多坐标约束学习_ORCA能力与数学方案_20261001.md`），未实现模型。

## 证据指针
- `docs/reports/PES2TS_多键同步一维扫描全面问题报告_20261001.md`
- `docs/reports/PES2TS_generation关键问题与根因分析_20261002.md`
- `docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`
- `docs/contracts/PES2TS_G2T_战役合同字段字典_v1.md`

## 后续影响
- 24 例真实战役于 2026-10-05 执行 → E04；四类终态词汇沿用至 E08/E13/E14/E15 的門判定。

---

### 第 2 期 · PES-N 立项期（2026-10-05 → 2026-10-14）

一句话目标：把项目从"G 阶段工程流水线"重定义为"PES-N 反应状态价值内核 + Classic 兜底"，并在同一周完成 G2 收口、G3.P 教师预训练立项、产品架构划分与轻量 RL 研究入口。

核心思路：①先定内核与决策语义（STOP/MOVE/fallback），再排训练；②教师预训练（G3.P）可与 G2 并行，不等 G2 调教；③产品架构三模块（共享内核 / Classic 控制 / 预测器），代码归属写进 ADR；④轻量路线：先在当前反应规模接通真实 RL，再扩量；⑤SSOT 治理：宪法为规范根、AGENT 为执行投影、阶段定义为唯一阶段来源。

关键技术决策：ADR-0010（PES-N 反应状态价值与 STOP-MOVE 统一决策，Accepted，10-05）、ADR-0011（三模块产品架构，Accepted）、ADR-0012（Classic 图论驱动与 xTB 原生约束扫描，Accepted）、ADR-0013（G3 并行 IRC 预训练与模型版本体系，Accepted）、ADR-0014（预测器产品归属与 G3 代码范围，Accepted）、ADR-0015（G2 稳定底座与轻量 RL 入口，Accepted）、ADR-0016（ORCA 驱动 xTB 硬约束优化，Accepted）、ADR-0017（统一残差决策与 G2 有限回退，Accepted）、ADR-0018（RL 纠错与动作支持研究计划，Proposed）、ADR-0019（IRC20k 与 RL2000 分阶段扩量，Proposed）。ADR-0007/0008/0009 三条价值模型路线被 ADR-0010 取代（历史提案）。

主要成果与指标：G2.T 24 例真实战役执行（2026-10-05，S1–S5 端到端）：24 个 typed 终态、443 条成本账本、N_gradient 总计 26,676；四类分账 2 rejected_typed / 19 partial_prefix / 3 complete_path / 0 validated，S4 全员 24/24 `blocked_on_human_review`（不是"跳过"）。AB3 原生约束扫描 24 例对比（2026-10-07）：13 complete_path / 9 partial_prefix / 2 rejected_typed（旧链 3/19/2），中位配对 scan 阶段加速约 61×；validated 仍为 0，生产默认未切换。G3.P/P1 立项：研究模型 `pesn-irc-v0.1.0`、教师合同 v1、B0–B7 模块划分。7 个 archive 快照时点（10-05/06/07/08）全部落在本期。

关键负结果：G2 四类终态中 validated=0、S4 无任何案例通过人工化学复核（24/24 blocked）；AB3 原生引擎只是显式分支，**不改变生产默认、不构成 TS 验证声明**；AB4 ORCA-xTB 四点试点设计完成但真实试点"尚未实施"（ADR-0016 状态）；G3.P 第一批的 STOP 概率、成本、MOVE-Q、生产控制全部 disabled（不可宣称能力）；PES-N 内核与 G3/G4 学习能力"仍待构建与评估"（根 README 原文）。

数据/代码/模型指针：`docs/contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md`；`artifacts/pesn/pesn-irc-v0.1.0-rc.1/model_card.md`；`config/registry/g_v_map.yaml` 前身记录在 E09 才系统化。当期原始文档 git 范围：10-05（49 提交）+ 10-06…10-14 文档（于 10-18 批量入库）；`docs/archive/20261005_pre_pesn`、`20261006_pre_g3p`、`20261007_pre_light_policy`、`20261007_pre_orca_xtb_pilot`、`20261007_pre_predictor_scope`、`20261007_pre_unified_residual_rl`、`20261008_pre_scale20k_rl2k`。

#### 示范成稿 · E03 PES-N 立项与 SSOT 重构（P2 PES-N 立项期）

> 条目标号：E03　时间范围：2026-10-05 → 2026-10-07　状态：Archived
> 阶段：PES-N 内核（规划） × G2/G3 阶段重定义　关联 ADR：ADR-0010（Accepted，2026-10-05）、ADR-0007/0008/0009（Historical，被 ADR-0010 取代）
> 数据/代码/模型版本指针：`docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md`（内容 v2.5，Superseded 后被 V5–V7 取代）；宪法内容 v2.x（快照 `20261007_pre_light_policy` 为 v2.3）
> 证据根：`docs/PES2TS_开发宪法_v1.md`、`docs/plans/PES2TS_阶段定义_G0-G3.md`、`docs/archive/20261005_pre_pesn/`、`docs/README.md` 2026-10-05 节

## 目标
把项目从"G 阶段线性工程"重定义为"PES-N 反应状态价值内核（ranking + navigation）+ Classic 兜底"，并确立 SSOT 文档治理。

## 核心思路
- 内核抽象：`ranking` 评估是否值得 `STOP_OPTTS`；`navigation` 比较 STOP / 图定义 MOVE / 验证过的 fallback。
- 单一规范来源：宪法（SSOT）→ AGENT（执行投影）→ 阶段定义（唯一阶段来源）→ 方案/合同（只引用）。
- 三层效力：红线 R1–R6 不可违背 / 工程政策 P1–P5 可时限豁免 / 接口规范 I-* 只以新版本变更。
- G3.P 教师预训练可与 G2 并行（不等待 G2 调教/出口）。
- 研究能力四级状态：implemented / fixture_passed / smoke_passed / scientifically_validated（后续 24H 战役正式成词）。

## 关键决策
- ADR-0010（Accepted，2026-10-05）：PES-N 反应状态价值与 STOP/MOVE 统一决策；**取代 ADR-0007（统一价值模型）、ADR-0008（拓扑约束距离/位移联合）、ADR-0009（G 阶段统一模型框架）三条历史提案**。
- 宪法 v2.x 建立（`docs/PES2TS_开发宪法_v1.md`，当前 v2.9 @ 10-21）；阶段定义 v4.2（快照 20261007）确立 G0–G4 主干 + C/R/X 工作流。

## 主要成果与量化指标
- 10-05 单日 49 次提交（全项目单日峰值），PES-N 方案、ADR-0010、阶段定义、开发宪法、产品架构同日成体系落盘。
- `docs/archive/20261005_pre_pesn/` 快照：PES-N 产品归属/ADR-0014 前的入口状态，含 manifest 与 revision_check。
- 阶段定义明确"S0–S8/demo 阶段表降为历史映射"，G0–G4 成为唯一阶段口径。

## 关键负结果
- **三条历史提案被否**：ADR-0007/0008/0009 价值模型方向作废（状态头 `历史提案，当前方向已由 ADR-0010/PES-N 计划取代`）。这是研究方向的负结果，不是文档错误。
- PES-N v2 内核"仍待构建与评估"；方案中 STOP/MOVE/fallback 语义在 10-21 被 ADR-0030/0031 两代修订（当时不可视为最终）。
- 根 README 明确："这是规划修订，不是实现或 QC 验证声明"。

## 限制与未解问题
- G2 出口未闭合时 PES-N 无法验收（依赖链写明 G3.0 需 G2 闭环）。
- 三模块产品架构（ADR-0011）与预测器归属（ADR-0014）在本条目仅为并行产出，详见 E05。

## 证据指针
- `docs/design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md`（状态行 L3）
- `docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md`（状态头 `Superseded / Historical`）
- `docs/archive/20261005_pre_pesn/README.md`
- `docs/PES2TS_开发宪法_v1.md` L3–L16（效力层级与增量登记）

## 后续影响
- ADR-0010 的 STOP/MOVE 语义演化为：ADR-0021（无序双端教师，E07）→ ADR-0030（撤回 TS 邻域 Stop，E11）→ ADR-0031（MOVE–PROCESS，E12）。读 E03 时必须按"当时判定"理解，不得直接套用 10-21 后语义。

#### E04 G2.T / AB3 / AB4 与 G3.P 立项（P2 PES-N 立项期）

> 条目标号：E04　时间范围：2026-10-05 → 2026-10-08　状态：Archived
> 关联 ADR：ADR-0006（ORCA 原生约束优化，Proposed）、ADR-0012（Classic 图论驱动与 xTB 原生约束扫描，Accepted）、ADR-0013（G3 并行 IRC 预训练与模型版本体系，Accepted）、ADR-0016（ORCA 驱动 xTB 硬约束优化试点，Accepted）
> 数据/代码/模型版本指针：`docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md`（内容 v1.2）；`artifacts/pesn/pesn-irc-v0.1.0-rc.1/model_card.md`
> 证据根：`docs/plans/PES2TS_G2T真实战役执行方案_20261004.md`、`docs/plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md`、`docs/plans/PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md`、根 `README.md` 2026-10-05/10-07 段

## 目标
在 24 例上第一次取得可审计终态与成本账本（G2.T 第一轮），并用原生约束扫描（AB3）验证"图论定义约束 + xTB 原生优化"是否优于旧链。

## 核心思路
- 24 例 Demo 的核心产出是**失败分布与成本反馈**，不是成功率。
- 扫描加速从"减少扫描点数、原生约束优化"入手，而不是加算力。
- G3.P 教师预训练独立立项（不等 G2），教师/标签/分裂先于训练。
- 每个新能力必须有独立证据验收，不能因"训练完成"自动成为生产默认。

## 关键决策
- ADR-0012：Classic 图论驱动 + xTB 原生约束扫描（Accepted，10-06）。
- ADR-0013：G3 并行 IRC 预训练与模型版本体系（Accepted，10-06）。
- ADR-0016：ORCA 驱动 xTB 硬约束优化试点（Accepted，10-07；实施/真实能力待验收）。
- ADR-0006：ORCA 原生约束优化与控制策略分离（Proposed，10-04/10-05）。

## 主要成果与量化指标
- **G2.T 第一轮（2026-10-05，S1–S5 端到端）**：24 个 typed 终态；443 条成本账本；N_gradient 合计 26,676；四类分账 **2 rejected_typed / 19 partial_prefix / 3 complete_path / 0 validated**；S4 **24/24 `blocked_on_human_review`**（设计内门，不是跳过）。
- **AB3 原生 24 例（2026-10-07）**：13 complete_path / 9 partial_prefix / 2 rejected_typed（旧链 3/19/2）；中位配对 **scan 阶段加速约 61×**；成本/报告缺陷在对比前修复（AB2 独立复核 `outputs/reviews/G2_real_campaign_review_20261006.md`）。
- G3.P/P1 立项文档齐全：教师数据与模型包合同 v1、`pesn-irc-v0.1.0` 版本体系、B0–B7 模块划分（代码归属 E05）。

## 关键负结果
- **validated=0**：两轮 24 例都没有任何案例通过严格 TS 验证；AB3 的"13 complete_path"不等于化学成功（终态=自动流程闭环）。
- **原生引擎未成为生产默认**：AB3 引擎是显式 `g2t-native-run` 分支，生产默认与 TS 验证声明均未改变（发布门未过）。
- AB4 四点试点仅设计完成：工作流、回退与真实试点"尚未实施"（方案状态头）。
- G3.P 首包 STOP 概率/成本/MOVE-Q/生产控制 disabled；数据/权重/硬件/评测均未验收。

## 限制与未解问题
- 24 例的人工化学复核门（S4）在整期中都未通过，直接导致 validated 长期为 0。
- AB3 成本对比只覆盖 scan 阶段加速，不含下游精修/验证成本，不能外推为总成本收益。

## 证据指针
- 根 `README.md` 2026-10-05 段（24 例数字）、2026-10-07 段（AB3 对比与 61×）
- `docs/plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md`（状态头：24 例终态）
- `docs/plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md`（状态头）
- `docs/contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md`

## 后续影响
- validated=0 与 S4 阻塞延续到 E07/E08 的评测协议设计；G3.P 进入 E05/E06/E07 的教师与训练主线。

#### E05 三模块产品架构与预测器归属（P2 PES-N 立项期）

> 条目标号：E05　时间范围：2026-10-05 → 2026-10-07　状态：Archived
> 关联 ADR：ADR-0011（三模块产品架构与统一可视化，Accepted）、ADR-0014（预测器产品归属与 G3 代码范围，Accepted）
> 证据根：`docs/design/PES2TS_三模块与可视化产品架构_v1_20261005.md`、`docs/design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md`、`docs/archive/20261007_pre_predictor_scope/`

## 目标
把"一个研究原型"拆成可交付产品边界：共享 PES-N 内核、Classic 控制/兜底、预测器（G3）三条产品线，并明确新代码归属。

## 核心思路
- 共享内核 + 上层产品：G 与 R 使用同一内核；静态排序可先交付，导航后接。
- 新业务代码统一 `pes2ts_core/pesn/`；现有拓扑/控制/ACP/严格验证产品只读复用（ADR-0014）。
- 代码范围与能力状态解耦：文件级范围（B0–B7）可先定，能力仍按证据验收。

## 关键决策
- ADR-0011：三模块产品架构与统一可视化（Accepted，10-05）。
- ADR-0014：预测器产品归属与 G3 代码范围（Accepted，10-07）。

## 主要成果与量化指标
- 产品架构 v1 + 文件级模块划分 v1 + 快照 `20261007_pre_predictor_scope`（13 份文档快照 + manifest + verification）。
- 运行时 API、隔离的离线训练/评测/导出、可选依赖、导入/导出验收检查被写入 B0–B7 → 成为后续 V4/V5 训练代码的实际目录结构。

## 关键负结果
- 架构文档明确"统一代码模块、接口 schema、UI 和真实能力另验收"——**架构成立 ≠ 能力成立**，截至入口冻结 0 能力验收。
- 后来的 G–V 双轴（ADR-0025，E09）重排了"V5–V7 不是新 G 阶段"的口径，说明本期的产品分层并非最终治理结构。

## 证据指针
- `docs/design/PES2TS_三模块与可视化产品架构_v1_20261005.md`（状态头 `当前产品架构设计`，内容 v1.1）
- `docs/design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md`
- `docs/archive/20261007_pre_predictor_scope/README.md`

## 后续影响
- 目录结构与 ADR-0014 的代码归属被 E07/E08 的 V4 训练代码直接继承。

#### E06 轻量控制 / 统一残差 / 扩量入口（P2 PES-N 立项期）

> 条目标号：E06　时间范围：2026-10-07 → 2026-10-14　状态：Archived
> 关联 ADR：ADR-0015（G2 稳定底座与轻量 RL 研究入口，Accepted）、ADR-0017（统一残差决策与 G2 有限回退，Accepted）、ADR-0018（RL 纠错与动作支持研究计划，Proposed）、ADR-0019（IRC20k 与 RL2000 分阶段扩量，Proposed）
> 数据/代码/模型版本指针：`artifacts/pesn/pesn-control-*/model_card.md`（6 份控制模型卡）；`docs/archive/20261007_pre_light_policy/`、`20261007_pre_unified_residual_rl/`、`20261008_pre_scale20k_rl2k/`
> 证据根：`docs/plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md`、`docs/plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md`、`docs/plans/PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md`、根 `README.md` 旧段（离线 fake-ACP 包 A/B/C）

## 目标
在 G2 底座稳定的前提下，把 RL 研究接入现实规模：统一残差决策 + 动作条件 Q；同时冻结扩量路线（IRC20k / RL2000）。

## 核心思路
- 先接通真实 RL 再扩量；不做"小数据上炫技、大规模崩塌"的路线（ADR-0017 取代"步长分类优先"）。
- 一个共享编码器 + 坐标残差头 + 动作条件 Q 头；预算化 constrained-MDP 近似。
- 候选效率代理奖励与严格 TS 成功分离，不把代理指标当成功。
- 环境统一：G2 共用同一执行/质量/恢复/成本接缝；AB4 四点门固定在策略动作空间之外。

## 主要成果与量化指标
- 控制侧离线包 A/B/C 实现并 fixture-passed（六控制合同、G2 单步提取 + E_native 宏步环境、候选/合法性/选择策略、统一残差 + Q 头、离线 conservative-Q 训练器、`g2t-control-round` CLI；见根 README 旧段）。
- Demo24 真实 RL 闭环 round0 完成一轮（`docs/plans/PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md` 状态头）。
- 扩量路线文档：IRC20k 与 RL2000 下一阶段详细方案（10-08）、RL 下一轮代码修复训练与调整（10-08）、RL 自适应路径生长（10-09）。

## 关键负结果
- 真实 QC 采集、训练、冻结重执行、AB4 适配器（包 D）与任何策略收益**均未运行**（根 README 原文：`Real QC collection, training, frozen re-execution ... remain not run`）。
- 闭环只有 fixture 与 round0 级别证据；`implemented/fixture_passed` 不构成能力验收。
- 扩量方案均为 `Proposed`，IRC20k/RL2000 实际未启动。

## 证据指针
- `docs/plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md`（状态头 `当前实施方案`）
- `docs/design/decisions/ADR-0017-统一残差决策与G2有限回退.md`
- `artifacts/pesn/pesn-control-demo24-v0.1.0/model_card.md`（仅模型卡，非性能声明）
- `docs/archive/20261008_pre_scale20k_rl2k/README.md`

## 后续影响
- 统一残差/Q 头与 G2 接缝成为 E09 V5 冲刺与 E10 24H 战役的代码底座；真实 QC 闭环在 E12 的 S1–S3 才第一次大规模执行。

---

### 第 3 期 · V4.x 训练实验期（2026-10-15 → 2026-10-18）

一句话目标：在"无序双端拓扑 + 完整双侧 TS 教师"的新任务上，用最小实验矩阵找出 V4.0/V4.1 的真实基线，再用 10 小时窗口做 V4.2–V4.7 决定性战役——结果是一个必须原样保留的负结果。

核心思路：①先修输入语义（去掉标签派生输入）再谈模型改进；②三口径冻结评测（legacy_full / native_full / common_unchanged）在看任何预测前冻结；③四门冻结（MSE/zero <1.0、≥6/8 组优于零步、整体 cos ≥0.30、near-TS cos ≥0.40）不可事后调整；④区分"训练内选择集"与"历史 monitor8"，confirm/test/family holdout 完全不读；⑤失败原因优先查输入可辨识性与数据覆盖，而不是堆容量。

关键技术决策：ADR-0021（无序双端拓扑与完整双侧 TS 教师主线，Accepted）、ADR-0020（EGNN→BC→IQL→SAC 研究主线，后转 Historical）、ADR-0022（v4.1 因果输入合同与特征 v3）、ADR-0023（V4.2 原生重连数据与内部开发划分）、ADR-0024（V4.2 无界位移模型与战役训练协议）。V4.2–V4.7 实施合同与战役冻结 v1（Frozen，训练开始后不可改）。

主要成果与指标：v4/v4.1 终报（10-17）与 V4.2–V4.7 十小时战役（10-18）共 **23 个正式 run 全部 completed**；5 项新工程能力：原生重连数据（剔除 399 个插入节点、重建 50 组/100 分支/5,568 个原生动作）、无界 Cartesian 位移头（取消 0.1·tanh 上限）、`campaign_v1` 训练协议、三口径冻结评测、可恢复战役驱动器。V4.1 纯因果基线：A 臂 MSE/zero 1.0817（2/8 组优于零步）、B 1.0984（1/8）、C 1.0555（3/8），cos 0.1186/0.1910/0.1466，near-TS 0.2393/0.2713/0.3050——**四门全部未过**。V4.7 FINAL（fit50、三种子、native_full 主口径）三种子中位：MSE/zero **1.132**、优于零步 **0/8**、整体 cos **0.180**、near-TS cos **0.248**，四个指标全部低于冻结门，三种子无一人过门。

关键负结果：V4.7 判负，且全部筛选臂（D0–D3、H2、E1/E2、V1、W96/W128）均未过门；H1 因物理先验 `Δs_fit=0` 被预注册门判 `prior_not_supported`；V2 因接触边实现仅 0.43 steps/s 被按预注册顺序裁剪；40k 延长门判 `STAY_20K`；E2 出现 `MODEL_LOAD_FAILED`（过夜汇总跳过）。诊断结论：fit 侧 MSE 持续下降（E1 0.706、W128 0.681、V1 0.516）而 dev 侧升到 1.26–1.49、优于零步组数降为 0 → **跨反应泛化缺口**；fit45 内存在 **342 对近似相同输入但教师方向冲突**的样本 → **输入可辨识性缺陷**，单纯加大模型/延长训练无证据支持。

数据/代码/模型指针：`docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md`（Frozen）；native reconnect 审计、梯度基线审计、协议冻结、run records 均在 `outputs/`（不入库，报告内有路径）。当期原始文档 git 范围：10-15…10-17 文档（10-18 批量入库）；10-18 起 17 次提交（V4.2–V4.7 战役执行与报告）。

#### 示范成稿 · E07 V4.0/V4.1 输入语义修复与双端 v4 主线（P3 V4.x 训练实验期）

> 条目标号：E07　时间范围：2026-10-15 → 2026-10-17　状态：Archived
> 关联 ADR：ADR-0021（Accepted，10-15）、ADR-0020（Historical）、ADR-0022（v4.1 因果输入合同，Accepted）
> 数据/代码/模型版本指针：`config/registry/g_v_map.yaml` 尚未建立；证据在 `outputs/pesn/dual_ts_v2_20261015/`（只读指针）；`docs/contracts/PES2TS_双端TS教师与相容转移合同_v2.md`（HISTORICAL/LEGACY 2026-10-21）
> 证据根：`docs/reports/PES2TS_v4阶段训练_最终报告与问题复核_20261017.md`、`docs/reports/PES2TS_v4.1阶段训练_修复输入语义与最小实验矩阵_最终报告_20261017.md`、`docs/reports/PES2TS_Direct_fit50训练表征分析_20261015.md`、`docs/reports/PES2TS_Direct批次1-6实施记录_20261015.md`

## 目标
在无序双端拓扑任务上建立可信基线：修复 v4 的输入语义问题，用最小实验矩阵（纯因果输入三臂）测出真实水平。

## 核心思路
- 输入可辨识性优先：v4.0 使用"标签派生输入"因此不能作为可部署基线（历史对照仅）。
- 三臂对照：纯因果（A）/ 方向损失（B）/ 有符号消息（C），其余变量冻结。
- 只用内部开发集做选择；历史 monitor8 不参与晋级；confirm8/test/family holdout 不读。
- 失败先归因（幅度收缩/方向拟合/跨反应泛化），再决定是否扩容。

## 关键决策
- ADR-0021（10-15）：无序双端点 + 完整双侧 TS 教师主线；之后被 ADR-0030（10-21）部分取代（Stop/TS 语义）。
- ADR-0022：v4.1 因果输入合同与特征 v3。

## 主要成果与量化指标
- v4.0 单种子 BC-NLL 基线：MSE/zero 0.9443（5/8 组优于零步）、整体 cos 0.1578、near-TS 0.2614（**含标签派生输入，仅历史对照**）。
- v4.1 三臂（三种子中位）：A 1.0817 / 2-of-8 / 0.1186 / 0.2393；B 1.0984 / 1-of-8 / 0.1910 / 0.2713；C 1.0555 / 3-of-8 / 0.1466 / 0.3050。
- 诊断：B 的方向改善在 fit 上 cos 0.365–0.367，但未转化为 monitor 误差改善；幅度比 α B≈0.217、C≈0.144/0.181 → 不能把失败解释成"只是步长小"。
- Direct fit50 表征分析与 Direct 批次 1–6 实施记录（10-15）作为 EGNN→BC→IQL→SAC 路线的执行证据。

## 关键负结果
- **v4.1 三臂全部未过冻结四门**（<1.0 / ≥6-of-8 / ≥0.30 / ≥0.40；四门需同一候选同时满足）。
- C 臂整体 cos 三种子都改善，但 MSE 只一粒种子改善，pooled 口径会反转 → 单指标改善不是晋级依据。
- 交叉版本差值存在输入与采样器混杂，不可拼成"逐版增益曲线"。
- A45 未提供"扩数据必然有效"的证据；同一旧配方延长训练缺乏支持（方案 §2 明示）。

## 限制与未解问题
- v4.1 是单种子还是三种子中位在矩阵部分为三种子；但 fit50 规模（50 组）本身小，结论限筛选。
- 双端拓扑任务无 schema、无能力升级、无生产默认变更（ADR-0021 声明）。

## 证据指针
- `docs/reports/PES2TS_v4.1阶段训练_修复输入语义与最小实验矩阵_最终报告_20261017.md`（指标表）
- `docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md` §2（V4.0/V4.1 汇总表，v1.1）
- `docs/plans/PES2TS_双端拓扑条件训练与项目进度统一方案_v4_20261015.md`

## 后续影响
- v4.1 三臂失败直接引出 E08 的"原生重连 + 无界位移 + 三口径评测"五项工程能力与 V4.7 定稿战役。

#### E08 V4.2–V4.7 十小时战役（P3 V4.x 训练实验期）

> 条目标号：E08　时间范围：2026-10-17 → 2026-10-18　状态：Archived（Executed-Frozen）
> 关联 ADR：ADR-0022（v4.1 因果输入合同）、ADR-0023（V4.2 原生重连数据）、ADR-0024（V4.2 无界位移模型与战役训练协议）；合同：`docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md`（Frozen）
> 数据/代码/模型版本指针：`campaign_manifest.json`、`native_reconnect_audit.json`、`gradient_baseline_audit.json`、`protocol_freeze.json`、`run_records/*/run_record.json`、`protocol_evals/*.json`（均在 `outputs/`，见报告证据边界）
> 证据根：`docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md`（v1.1）、`docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md`、`docs/reports/PES2TS_实现报告_20261018.md`

## 目标
在一个单 GPU 10 小时窗口内，用 V4.2–V4.6 五组对照 + V4.7 三种子定稿，判断"原生重连数据 + 无界 Cartesian 位移 + campaign_v1 协议"能否把双端 IRC 教师学习推到四门之上。

## 核心思路
- 先审计（原生标签/坐标、解析梯度基线、测速），再训练；每项收益预设判据。
- 数据版本升级：剔除 399 个插入节点，重建 50 组/100 分支/5,568 个原生动作。
- 模型侧：取消 `0.1·tanh` 动作上限，直接回归 Å 位移；旧 radial-tanh 头保留为 D0/D1 对照。
- 评测侧：三口径协议（legacy_full / native_full / common_unchanged）在看预测前冻结；四门原样保留。
- 工程侧：可恢复战役驱动器（预检、测速、阶段晋级、预注册裁剪、预算删失、失效 run 归档）。

## 关键决策
- ADR-0023/0024（V4.2 数据与模型协议）；实施合同 Frozen（训练开始后不得修改）。
- V4.7 配方先冻结（最终配置、步数、选权重规则），再在完整 fit50 从头训三种子（20261017/18/19）；筛选与定稿分开命名与统计。

## 主要成果与量化指标
- **23 个正式 run 全部 completed**（20000 updates/run；峰值显存 65–103 MB；吞吐约 28–49 steps/s）。臂：D0/D1（legacy_v41 头）、D2/D3/E1/E2（campaign_v1）、FINAL×3、H2、V1、W96、W128。
- 5 项新工程能力：原生重连数据、无界位移头、`campaign_v1`（AdamW + warmup/cosine + EMA + u5k/u10k/u20k 检查点 + 曝光账本 + 断点恢复）、三口径冻结评测、可恢复战役驱动器。
- **V4.7 FINAL 三种子中位（native_full 主口径）：MSE/zero 1.132、优于零步 0/8、整体 cos 0.180、near-TS cos 0.248**；legacy_full 与 common_unchanged 同方向结论。

## 关键负结果
- **V4.7 判定负结果**：三项种子四项门全部未通过；不晋级、不部署、不扩量，保留为冻结证据（方案状态头原文）。
- 全部筛选臂（D0–D3、H2、E1/E2、V1、W96/W128）均 `BELOW_GATE`，没有任何一条通过四门。
- H1 因物理先验 `Δs_fit=0` 被预注册门判 `prior_not_supported`（不是训练失败，是假设不成立）。
- V2 因接触边实现仅 **0.43 steps/s** 按预注册顺序裁剪；40k 延长门判 `STAY_20K`；E2 出现 `MODEL_LOAD_FAILED`（三口径行跳过）。
- 诊断结论：fit 侧 MSE 下降（E1 0.706、W128 0.681、V1 0.516）而 dev 侧 MSE 升到 1.26–1.49、优于零步组数 0 → 当前 50 组数据 + 单帧 5 通道 + 全原子位移目标 + 20k 预算的组合存在**严重跨反应泛化缺口**；fit45 内 **342 对近似相同输入但教师方向冲突** → 优先解决输入可辨识性/训练时长/数据覆盖，而不是继续堆容量。

## 限制与未解问题
- 10 小时窗口只覆盖训练与离线评估；真实 rollout、新 QC、500/2k 扩量、IQL/SAC 不计入本夜。
- monitor8 是历史回归集不是封存测试集；confirm8/test/family_holdout 未触碰（不能声称泛化）。
- 三口径协议虽冻结，但协议间差异较大（见过夜汇总表），单一口径不能代表全貌。

## 证据指针
- `docs/plans/PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md`（状态头 `Executed-Frozen`；§2 基线表）
- `docs/reports/PES2TS_V4.2-V4.7_过夜战役汇总_20261018.md`（§1 run 表、§2 三口径评测）
- `docs/reports/PES2TS_实现报告_20261018.md`（摘要：负结果与 342 冲突对）
- `docs/contracts/PES2TS_V4.2-V4.7_实施合同与战役冻结_v1.md`

## 后续影响
- 负结果 + "输入可辨识性/泛化缺口"判断直接促成 E09 的 V5.0–V5.2 冲刺方案与 E11 的任务定义纠偏讨论。

---

### 第 4 期 · V5 冲刺与 24H 战役期（2026-10-18 → 2026-10-21）

一句话目标：在 V4 负结果之后，用两个短周期战役（V5.0–V5.2 十小时冲刺、24H 自主战役 R3→V5.2→V5.3）回答"任务定义是不是错了"——答案是：R3 方向成立而 V5.2 的 Stop 任务定义错了，于是 10-21 当天完成两次任务语义修订。

核心思路：①把"路径导航"拆成 Move / Progress / Stop 三个可分别证伪的假设；②训练前冻结门与预注册，失败不调参复活；③多 lane 并行（DATA/FEAT/LOSS）+ fail-closed 授权；④负结果按"已修复/负结果/按设计阻塞"三类分账；⑤任务定义错误要写进 ADR 撤回，而不是悄悄改标签。

关键技术决策：ADR-0026（V5 冲刺 key-atom 位移合同与 OOF 语义，Accepted）、ADR-0027（V5.1 参考架 fallback 与双分母评估，Proposed）、ADR-0028（关键原子几何策略合同与 V5–V7 任务拆分，Proposed）、ADR-0029（V5.2+ 双门评估与封存边界，Proposed）、ADR-0030（全路径导航与端点停止，Accepted，10-21）、ADR-0031（MOVE–PROCESS 统一学习合同，Accepted，10-21）。

主要成果与指标：V5 冲刺 `complete_oof_1seed`（prepare-audit / label-screen / recipe-recheck / crossfit 完成，report not_executed，封存评测 REFUSED，复核稳定配方 A05）。24H 战役：11 个工作包全部终态、六门终态；93 次 launch attempts（R3 61/66、V5.2 32/33）；GPU 训练 0.93 GPU-h（70 runs × ~48s）；物理 xtb 成本 49s（36 次受约束松弛）；墙钟 7.75h。R3 冠军 **F2-bidir：RMSE/zero 0.910 [0.884–0.949]、sign 0.888、6/6 优于零步、三种子一致**；跨折泛化 0.9735 [0.949–0.998]、10/12、衰减约 6.4pp。V5.3-ENG rollout：216 episodes + 36 次真实 xtb，工程闭环 PASS。

关键负结果：V5.1a 在 5-fold 门被 rejected，V5.1a-R1 无合格候选，V5.1b 保持关闭；V5.2 **q 不可学习**（MAE 0.507 vs 常数 0.476，corr≈0）、**stop 头随机**（AUC 0.491–0.527、recall@0.5=0）、**Move 退化 +5.5% 中位（4/6 破 ≤2% 护栏）**；classic 臂严重漂移（215 碰撞步），16/36 物理失败中 15 例溯源于碰撞种子几何；R3 43/48 run 的 inner-val 选择 ≤250/3000 updates（冻结日程过长的证据，非 bug）；V5.2 的 15 次 λ-bug 烧毁 attempts 计入总账（失败修复记录）。任务层根因（E11 审计）：V5.2 的 Stop 任务本身定义错——"应该 Stop"的终点样本被 `terminal` 排除在训练外。

数据/代码/模型指针：`docs/agent_runs/state/gates/*.json`（门禁证据，不删）；7 份根 memo（`24H_EXECUTION_TIMELINE.md`、`24H_FINAL_REPORT.md`、`FAILURE_AND_REMAINING_WORK.md`、`NEXT_DAY_DECISION_MEMO.md`、`R3_RESULT_SUMMARY.md`、`V52_PROGRESS_STOP_REPORT.md`、`V53_ROLLOUT_REPORT.md`）；`V53MP_RESUME_20261022.md`（转 E12）。当期原始文档 git 范围：10-18…10-21 共 111 次提交（17+14+43+37）。

#### 示范成稿 · E10 24H 自主战役 R3→V5.2→V5.3-ENG（P4 V5 冲刺与 24H 战役期）

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

#### E09 V5.0–V5.2 十小时冲刺与 R3 首轮（P4 V5 冲刺与 24H 战役期）

> 条目标号：E09　时间范围：2026-10-18 → 2026-10-19　状态：Archived
> 关联 ADR：ADR-0026（V5 冲刺 key-atom 位移合同与 OOF 语义，Accepted）
> 证据根：`docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md`、`docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md`、`docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md`

## 目标
V4 负结果后 24 小时内启动 V5 冲刺：先做 OOF 交叉拟合验证配方，再决定是否进入修复冲刺与扩量。

## 主要成果与量化指标
- sprint status `complete_oof_1seed`：prepare-audit / label-screen / recipe-recheck / crossfit 均 complete，report 未执行；G5.1 fixtures pass；Oracle static ceiling `ORACLE_STATIC_OK`；封存评测 `REFUSED (sealed_confirm_not_opened)`；复核稳定配方 **A05**。
- V5.1 修复冲刺与扩量准入方案落盘（10-19，无状态头，A14/P2 异常）。

## 关键负结果
- 报告未执行、封存评测被明确拒绝 → 只有单 seed OOF 证据，不能作正式结论。
- V5.1 修复冲刺方案无状态声明（治理异常），其"扩量准入"未经独立签认。

## 证据指针
- `docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md`（阶段状态表）
- `docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md`（ANOMALIES A14）

## 后续影响
- 冲刺余量并入 E10 的 24H 战役；V5.1a 在 5-fold 门被 rejected（见 E10 上位计划与根 README 10-20 状态段）。

#### E11 V5.2 任务纠偏与 ADR-0030/0031（P4 V5 冲刺与 24H 战役期）

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

#### 示范成稿 · E12 V5.3-MP 规模联合训练与 2k→50k 收益结论（P5 V5.3-MP 规模联合训练期）

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

#### 示范成稿 · E13 V5.4.1 Embedding-50K 执行（E0 留任）（P6 V5.4.x 三阶段期）

> 条目标号：E13　时间范围：2026-10-27 → 2026-10-29　状态：Archived（Executed-Frozen）
> 关联 ADR：ADR-0032（Accepted，10-27）；任务合同 MoveProcessContract v1
> 数据/代码/模型版本指针：`archive/v5x_configs/campaigns/pes2ts_v541_embed_50k.yaml`（schema `pes2ts_v541_campaign_v1`）；冻结计划 `plan/runplan.pin.json`（code_commit `1d8b7972…`、common_protocol `2b9ce1ed…`、config `464c1f6f…`）；证据根 `outputs/pesn/v541_embed_50k/`
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

#### E14 V5.4.2 Backbone-50K 执行（`no_backbone_winner`）（P6 V5.4.x 三阶段期）

> 条目标号：E14　时间范围：2026-10-29 → 2026-10-31　状态：Archived（Executed-Frozen；计划状态头滞后 A01/P0）
> 关联 ADR：ADR-0033（Accepted，v1.3.3）；合同 MoveProcessContract v1
> 数据/代码/模型版本指针：`archive/v5x_configs/campaigns/pes2ts_v542_backbone_50k.yaml`（`0.1.2-wp0`，schema `pes2ts_v542_campaign_v1`）；registry `V5.4.2`（status executed，`plan_revision 2026-10-30.3`）
> 证据根：`docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md`（内容 v1.2）、`docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md`、`docs/design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md`

## 目标
在统一单帧空间 Backbone 接口（`z[G,64]` + `edge_msg[E,48]`）下，比较 A1–A5 候选与 matched A0，判断是否有候选能在三个 seed 上同时不劣化 MOVE 并带来 PROCESS 收益。

## 核心思路
- 接口冻结：唯一共享 `DirectMoveDecoder`（仅 `w_v` 49 参数），PROCESS 头原样复用（2,178 参数），A0 是 legacy Star 子类 adapter + hook + 显式映射审计。
- A0 默认重训 matched ×3（不实现 shadow-run/reuse 分支）；V541 E0×3 只作 parity/历史参照。
- 预算用同环境 40+40 profile 实测签认，core ratio ≤2.0。
- H2 逐 seed 对照 matched A0 与 R3；H3 只在 H2 无 breach 的候选上做 Holm 校正的 10,000 draws 配对 bootstrap。
- 无 H3 合格候选就走 `zero_h3_qualified_candidates_a0_retained`，不产生"最低分胜出"。

## 关键决策
- ADR-0033（Accepted）：Backbone 接口/迁移不变量/parity 证书/预注册记录模板全部冻结；默认重训 matched A0×3。
- WP4 diagnostic ruling：A0=`control_not_screened`、A4=`eliminated/gate_skipped`（3.849742% < 5.0%），阈值与预算数值不变（协议修正记录 `V542_WP4_DIAG_20261030`）。

## 主要成果与量化指标
- 执行账本全部实跑：smoke 200（6/6 done）、profile 8 backbones、diagnostic 2000、A0 sanity band PASS（seed 均值 A0 AUPRC 0.29670 vs E0 0.28858，+0.00812 ≤0.01；q_corr −0.00045；RMSE/zero −0.199%）、T0/T1/T3/T2 全部 done、**N_core=17**。
- 内层 inner-val：A2_S17 0.3275 / A2_S29 0.3213 / A2_S43 0.3209（显著优于 A0 ~0.67）；A3_S17 0.6476 / S29 0.6506 / S43 0.6713；A5_S17 0.7079；A1_S17 0.6924。
- parity 证书 6/6 资产：CPU bitwise 0.0、GPU 最大 9.5367e-7、frozen evaluator 重算 delta 0.0。

## 关键负结果
- **战役结论 `no_backbone_winner`：H3 eligible = []，A0 留任，不产生 Backbone\*。** 该结论不得写成"A0 胜出"。
- **A2/A3 的 seed 43 触发 per-seed H2 breach**（matched_a0.sign_abs_drop、fixed_coverage_direction_rel_degradation）→ 记 `scientific_failure`，不晋级、不得以 PROCESS 收益豁免。
- A1 未进 top2（seed17 screen top2=[A2,A3]）；A4 被冻结的 5% 学习筛淘汰；A5 完成 full-dose 但 seed17 未过 screen，未获补种。
- 计划状态头滞后：L3 `Active-Proposed`（WP1–WP7 待执行）与 README/report 的 "WP7 EXECUTED（2026-10-31）" 直接冲突（A01/P0）——**不得据计划状态头认为未执行**。
- `V5.4 剩余架构研究工作方案 v1.0` 全库 0 引用、未正式登记（A13/P2），本期不纳入执行。

## 限制与未解问题
- 候选侧 MOVE sign/direction 的 seed 敏感性说明 50K 数据下 MOVE 结构搜索的统计功效有限；PROCESS 收益真实但不足以豁免 MOVE 门。
- confirm 未读、QC=0、dev 未访问；无生产资格声明。

## 证据指针
- `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md`（§0 结论冻结写法、§2 账本、§3 指标表、§4 H2/H3 判定）
- `docs/design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md`
- `docs/README.md` 2026-10-29 节（WP7 执行事实）

## 后续影响
- E15 因此改问"不换 backbone、换 decoder（MOVE/PROCESS 残差轨道）能否同时提升"——并首次采用双轨冠军 + BMP 集成验证。

#### E15 V5.4.3 双 Decoder-50K 执行（双轨冠军）（P6 V5.4.x 三阶段期）

> 条目标号：E15　时间范围：2026-11-02 → 2026-11-06　状态：Archived（Executed-Frozen）
> 关联 ADR：ADR-0034（Proposed，但 D1–D8 于 11-02 经仓库所有者签认）；方案与 WP 清单保持 `Frozen-Preregistered v1.0`（按台账明示"执行完毕不改文"）
> 数据/代码/模型版本指针：`archive/v5x_configs/campaigns/pes2ts_v543_dual_decoder_50k.yaml`；终版 bundle `PES2TSBundleV543_20261106T095436Z`（208 文件 verify ok）；dev pack `4f6c1e95…`（589 曲线/135,926 positions）
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

主要成果与指标：LICENSE（MIT，2026-11-20）补齐公开前最后缺口；5 份划分名单 15.2 MB 逐字节入库（索引 `data/manifests/split_registry_v1.json`，sha256 前缀 07bdbedc…/802f6243…/83e72e9a…/b65fa4c9…/c0d2cb2a…，链条校验均等于 L1 `802f6243…`）；`docs/superseded/` 建立并迁入 4 份自身已声明 Superseded 的文档；5 个零引用 V4 脚本归档 `archive/v4/v4/`；CI 红灯修复（`gv_registry.py` 区分仓库内外路径，26 tests 由 4 failed 转全过）；门禁 `check_docs_links.py` 0 error / 173 warning（182 docs）、`docs_consistency --strict` CLEAN。全量审计（11-21）：253 条 / 4.97 MB / 55,452 行；`docs/**/*.md` 216（活跃 180 / archive 36）；状态块 167/216；状态原始值 137 种 → 17 个归并桶；索引双缺 40；21 家族/53 成员（6 组状态自相矛盾）；52 条硬钉死路径；32 条异常（P0 2 / P1 10 / P2 20）。

关键负结果：**40 份文档既不在 `docs/README.md` 也不在台账**（C-08）且索引自述版本滞后 5 处（C-03）；**`docs_consistency` 从未进入 CI**，12 项冲突在门禁 CLEAN 基线下长期并存（C-09）；176 条 warning 是"容忍债务"不是 0 债务（C-11，台账 11-20 记 173/182，口径差 1 属不同时点）；7 份根目录战役 memo 因门禁钉死不移动、只加索引（A30）；`docs/archive/**` 28 份与 live 文档同名（A22，快照只读不改）；归档家族"①可立即归位"实际为 **0** 份（F19 三份因有引用不满足），说明"零成本归位"在现实中几乎不存在。

数据/代码/模型指针：`docs/infra/doc_audit/20261121/`（doc_inventory.csv / DISPOSITION.csv / FAMILIES.md / ANOMALIES.md / GOVERNANCE_CONSTRAINTS.md / BATCH_PLAN.md / VERIFICATION_REPORT.md / baseline_gate_outputs.txt）；`data/manifests/split_registry_v1.json`；HEAD `2f1d3ba`。当期原始文档 git 范围：11-20 的 6 次提交（归档 4 份 + 5 脚本、LICENSE、CI 修复、划分名单 ×2、ids -text 固定）与 11-21 的审计/方案文档（未提交或本轮提交，`[未跟踪]` 状态以台账口径为准）。

#### 示范成稿 · E16 归档治理与公开化（P7 治理归档与公开化期）

> 条目标号：E16　时间范围：2026-11-20　状态：Archived
> 关联 ADR：无新 ADR（治理动作；规范化方案 v2.0 的升格需经 ADR 与宪法登记——见 `docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md` 边界段）
> 数据/代码/模型版本指针：`LICENSE`（MIT）；远端 `git@github.com:PengYangchao0808/PES2TS.git`；分支 `master` + `sync/20261018`；`data/manifests/split_registry_v1.json`
> 证据根：`docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md`、`docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md`、`docs/superseded/README.md`、`data/README.md`、`docs/plans/PES2TS_文档修订台账.md` 2026-11-20 四条增量

## 目标
在推 GitHub 前做一次"只增不删"的静态体检与最小执行：补许可证、发布不可重建的划分名单、建立取代区、修 CI 红灯。

## 核心思路
- 大块头已被 `.gitignore` 挡住（`data/` 12.15 GB、`outputs/` GB 级）——真正要治的是入库面的历史沉淀，不是删数据。
- `docs/archive/` 是版本快照、**禁止当"过时文档墓地"**；取代文档另立 `docs/superseded/`。
- `docs/reports/**` 内容永不改；门禁钉死路径移动即红 → 移动必须同 PR 改配置。
- 划分名单"只做逐字节副本 + registry SHA"，因为该划分无法从仓库其它文件重建。
- 每条动作都要有回滚路径与门禁复跑结果。

## 关键决策
- 仓库归档与清理方案 v1（Operational，非规范）：静态体检 + A/B/C 三档清单 + P0–P5 五步执行与回滚 + 推 GitHub 10 项 checklist。
- 训练/测试集划分与数据来源方案 v1（Descriptive，非规范；当晚升 v1.1）：L0–L4 五层继承链首次成文。
- `docs/superseded/` 收录规则：只收"自身顶部已声明 Superseded 且全库 0 引用"的文档，正文不改、只修跨目录链接。

## 主要成果与量化指标
- **LICENSE**：新增 MIT（Copyright (c) 2026 PengYangchao0808）；此前仓库无任何许可文件 = 公开后等同保留所有权利。
- **划分名单发布**：5 份、15.2 MB、逐字节副本；`split_ids_g0_v1.csv` 199,168 行（train 159,355 / valid 19,904 / test 19,909）、`pesn_irc_split_ids_v1.json`（train_fit 121,363 / train_dev 15,161 / train_calibration 15,159 + family_holdout 9,573）、v53mp cohort 20k/50k、`v53mp_split_50k_ids_v1.json`（pool tiers + dev 589 + confirm 2,000 仅列名 + curve_folds 66,195）；链条校验：两个 cohort 的父哈希与 50K manifest 的 `parents.g3p1_split` 均等于 L1 `802f6243…`。
- 归档执行：`docs/superseded/` 建立 + 迁入 4 份（单端点 IRC 产品设计 v1/v2、V5.1-R3 端点对称 v1/v2）；5 个零引用 V4 脚本迁 `archive/v4/v4/`（该目录 .py 16→21）；7 份根 memo 按建议不移动只留索引。
- CI 修复：`gv_registry.py` 新增 `_is_external_path()`（仓库外路径只作 provenance、不强制存在；仓库内相对路径仍严格校验）；`pytest tests/test_gv_registry.py tests/test_config_env.py` → 26 passed（修复前 4 failed / 20 passed）。
- 门禁：`check_docs_links.py` 0 error / 173 warning（182 docs）、`docs_consistency --strict` CLEAN；`render_gv_map --check` 的失败源于既有环境的 WSL 路径问题（同日第二条增量已修）。

## 关键负结果
- 公开前体检暴露 40 份文档索引双缺、索引版本滞后 5 处、176 warning 容忍债务——**治理负债在公开时点仍在**。
- 7 份根目录战役 memo 与 5 份多版本残留因门禁/引用钉死**不能移动**，只能加索引/横幅（"低成本归位"被证否）。
- `render_gv_map --check` 在修复前持续失败（自 2026-10-31 起 `master` 每次 CI run 都挂），原因是数据 manifest 记录 WSL 绝对路径；说明"外部路径强校验"是过度约束。
- 未删除任何证据、未改 `docs/reports/**`、未改 `docs_consistency` 门禁名单（边界声明，不是缺陷）。

## 限制与未解问题
- `data/` 六表本体与 dev pack/idx 仍未入库（128 GB 无发布渠道）；克隆者只能"重生成 + 对数字"。
- 远端为 SSH、需确认 key 或切 HTTPS+PAT；`master` 与 `sync/20261018` 两条分支是否都推未决。
- 划分方案仍是 Descriptive，无规范位；升格须走 ADR。

## 证据指针
- `docs/infra/PES2TS_仓库归档与清理方案_v1_20261120.md`（§静态体检/三档清单/checklist）
- `docs/README.md` 2026-11-20 三节；`docs/plans/PES2TS_文档修订台账.md` 2026-11-20 四条增量
- `data/manifests/split_registry_v1.json`、`data/README.md`
- `git log --oneline --since=2026-11-19 --until=2026-11-21`（6 次提交）

## 后续影响
- 体检出的 253 条/32 异常/21 家族直接成为 E17 全量审计与归档方案的输入；`docs/superseded/` 分区规则被 E17 的方案继承。

#### E17 文档规范化归档方案 v2.0 与全量审计（P7 治理归档与公开化期）

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

## 4. 跨期负结果总表（红线：不得删除、不得美化）

| # | 期/条目 | 负结果 | 判定口径 | 原始证据 |
|---|---|---|---|---|
| N01 | P1/E01 | G1 含 `truth_assisted_p1` 来源，不能称端到端 truth-free | 阶段定义原文 | `docs/plans/PES2TS_阶段定义_G0-G3.md` §2 G1 |
| N02 | P1/E02 | G2/G2.T 24 例 validated=0；S4 24/24 blocked_on_human_review | 四类终态分账 | 根 `README.md` 2026-10-05 段 |
| N03 | P2/E03 | ADR-0007/0008/0009 三条价值模型路线被 ADR-0010 取代 | ADR 状态行 | `docs/design/decisions/ADR-0007~0009` |
| N04 | P2/E04 | AB3 原生 24 例仍 validated=0；原生引擎未成生产默认 | 四类终态 + 发布门 | `docs/plans/PES2TS_G2_AB3...20261006.md`；根 `README.md` 10-07 段 |
| N05 | P2/E04 | AB4 ORCA-xTB 四点试点仅设计，真实试点未实施 | 状态头 | `docs/plans/PES2TS_G2_AB4...20261007.md` |
| N06 | P2/E06 | 真实 QC 采集/训练/冻结重执行/策略收益均未运行 | 根 README 原文 | 根 `README.md` 旧段 |
| N07 | P3/E07 | v4.1 三臂全未过四门（A 1.0817/2-of-8；B 1.0984/1-of-8；C 1.0555/3-of-8） | 冻结四门 | `docs/reports/PES2TS_v4.1阶段训练...最终报告_20261017.md` |
| N08 | P3/E08 | V4.7 FINAL 三种子四门全负（1.132 / 0-of-8 / 0.180 / 0.248）；全部筛选臂 BELOW_GATE | 冻结四门 + 三口径 | `docs/reports/PES2TS_实现报告_20261018.md` |
| N09 | P3/E08 | fit45 内 342 对近似相同输入却教师方向冲突 → 输入可辨识性缺陷 | 诊断统计 | 同上 §摘要 |
| N10 | P4/E10 | V5.1a 5-fold 门 rejected，V5.1a-R1 无合格候选，V5.1b 关闭 | rejected_gate | 根 `README.md` 10-20 状态段 |
| N11 | P4/E10 | V5.2 q 不可学习（0.507 vs 常数 0.476）、stop 随机（AUC 0.491–0.527）、Move 退化 +5.5%（4/6 破护栏） | 预注册 G3 门 | `24H_FINAL_REPORT.md` §3 |
| N12 | P4/E10 | classic 臂 rollout 崩塌（215 碰撞步）；16/36 物理失败中 15 例溯源自碰撞种子 | rollout 记录 | `24H_FINAL_REPORT.md` §4 |
| N13 | P4/E11 | V5.2 的 Stop 任务定义错误被撤回（92/200 终点行被 terminal 排除；旧正例 d_key_TS 中位 0.021Å） | 只读审计重算 | `docs/reports/PES2TS_V52_任务纠偏审计_20261021.md` |
| N14 | P4/E11 | V5.2-R D1 FAIL / D2 P4P5 FAIL → D2-G STOP、T1 不授权 | 预注册门 | `docs/plans/PES2TS_V52R_...v1_20261021.md` |
| N15 | P5/E12 | Move 规模收益 20k 饱和、50k sign 降 1.7–2.2pp、dead-zone 2.1%→6.0% | 四档同口径 dev 589 | `docs/reports/PES2TS_V53MP_规模训练收益报告_20261027.md` §0 |
| N16 | P5/E12 | endpoint 头 seed 不稳定（C_s29 误停率 0.63）；训练资格 ≠ 部署资格 | inner-val τ 校准 | `docs/reports/PES2TS_V53MP_S3_20k主结果报告_20261023.md` §2 |
| N17 | P6/E13 | V5.4.1 四候选全败：E1/E2 H2 sign、E3 sign+dead、E4 成本门 3.6455>2.0 | H2 硬门/成本门 | `docs/reports/PES2TS_V541_..._20261029.md` §3 |
| N18 | P6/E14 | V5.4.2 `no_backbone_winner`：A2/A3 seed43 H2 breach、A1 未进 top2、A4 被 5% 筛淘汰 | H2/H3 冻结规则 | `docs/reports/PES2TS_V542_..._20261031.md` §0/§4 |
| N19 | P6/E15 | V5.4.3 M3 被 H2 排除、M2 seed43 复现失败；M4/M5/P5 未触发；No-Move 结构性未开放 | H2 非劣/条件激活 | `docs/reports/PES2TS_V543_..._20261106.md` §1/§4 |
| N20 | P7/E16 | 40 份索引双缺、176 warning 容忍债务、7 份根 memo 不可移动 | 审计统计 | `docs/infra/doc_audit/20261121/README.md`；`ANOMALIES.md` A30/C-11 |
| N21 | P7/E17 | `docs_consistency` 未进 CI，12 项冲突在 CLEAN 下并存；"可立即归位"实际 0 | C-09/C-08；FAMILIES §四 | `docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md` |

---

## 5. 现存文档 → 档案条目映射说明（`ARCHIVE_MAP.csv`）

字段定义与统计：

| 字段 | 取值来源 | 说明 |
|---|---|---|
| `path` | `doc_inventory.csv` | 仓库相对路径（253 条全量，1:1 覆盖） |
| `current_action` | `DISPOSITION.csv` | keep 125 / retain-readonly 109 / banner 16 / move 3（**沿用处置口径，本档案树不执行**） |
| `archive_entry` | 本设计 | E01–E17 / X0（横切常青层） |
| `extract_needed` | 本设计 | Y=需抽取结论/负结果到条目正文（67 条）/ N=只作指针（186 条） |
| `extract_content` | 本设计 | 需抽取的具体结论（≤120 字；全部可在原文定位，不虚构） |
| `keep_reason` | 本设计 | 该文档在档案树中的角色（证据/指针/快照/横切） |

覆盖与分布：253/253 全覆盖；按条目 E01 11、E02 40、E03 11、E04 21、E05 5、E06 41、E07 12、E08 7、E09 4、E10 29、E11 17、E12 15、E13 4、E14 4、E15 4、E16 6、E17 4、X0 18。（E02/E06 偏大主要因 2026-10-01~10-04 的 22 份 Demo24/多坐标报告与 7 个快照目录的 36 份快照文件；这些文件多数 `extract_needed=N`，只作证据指针，不增加条目正文负担。）

**snapshot 与 archive 的处理**：`docs/archive/**` 36 条全部映射到其**所快照的历史期**（20261005→E03、20261006→E04、20261007 五目录→E04/E05/E06、20261008→E06），`action=retain-readonly`、`extract_needed=N`；`docs/archive/README.md` 归 X0。理由：快照是时点证据，字段"当期数据/代码/模型版本指针"引用它们即可，条目正文不复述快照内容。

---

## 6. 后续执行清单（档案树落位后）

1. **确认分期与条目编号**：所有者对 7 期/17 条目/边界证据（本文件 §1）签认；编号冻结，不复用。
2. **按 `ARCHIVE_MAP.csv` 的 Y 清单抽取正文**：67 份文档 → 17 条目（每份的 `extract_content` 即抽取提纲）；抽取时只读原文档，不改原文。
3. **负结果先行**：先填 §4 的 N01–N21 到各自条目「关键负结果」段，再补成果段——防止叙事美化。
4. **证据指针校验**：每条目的 `证据指针` 逐条 `Test-Path` 校验；不得引用不存在的路径（本文件已按现存文档写）。
5. **门禁适配**：若落位 `docs/archive_tree/`，同 PR 更新 `docs/README.md`、`docs/plans/PES2TS_文档修订台账.md`；若接入 `docs_consistency`/G5，按方案 §G5 规格增量登记（不要一次性改 52 条钉死路径）。
6. **不做的事**：不移动/删除任何既有文档；不改 `docs/reports/**`；不改 `docs/consistency` 门禁名单；不把本档案树写成"新规范"（它只是叙述层，规范仍以宪法为准）。

---

## 附录 A · 期边界证据清单

| 边界 | 文档证据 | git 证据 | 快照/里程碑 |
|---|---|---|---|
| P1 起 | `requirements.txt` 09-25；`G1_TS_IRC_映射...` 09-26 | 首提交 `a7c4ed7` 09-25；09-25…10-05 共 161 提交 | — |
| P1→P2 | 10-04 最后一批 G2 方案；ADR-0010 于 10-05 Accepted | 10-05 单日 49 提交（峰值）；10-06…10-17 无提交 | `20261005_pre_pesn`（10-05） |
| P2 内 | G3.P/AB3/AB4/预测器/轻量 RL 方案 10-06…10-09；7 快照 | 文档于 10-18 批量入库（`git_first_add_date`） | `20261006`/`20261007`×5/`20261008` |
| P2→P3 | v4 方案 10-15；v4/v4.1 终报 10-17 | 10-18 起 17 提交（V4 战役执行） | — |
| P3→P4 | V4.7 判负 10-18；V5 冲刺方案/report 10-18 | 10-18…10-21 共 111 提交 | — |
| P4→P5 | ADR-0030/0031 同日 Accepted 10-21 | 10-21 共 37 提交 | — |
| P5 内 | S0/S1/S2 10-22；S3 10-23；S4/规模收益 10-27 | 10-22 19 提交 | — |
| P5→P6 | V5.4.1 WP0 10-27 / 执行 10-29 | 10-27 4 提交、10-29 1 提交 | — |
| P6→P7 | V5.4.3 Executed-Frozen 11-06 | 11-06 6 提交；11-07…11-19 无提交 | — |
| P7 内 | 归档方案 v1/License 11-20；归档方案 v2/审计 11-21 | 11-20 6 提交；HEAD `2f1d3ba` | — |

## 附录 B · 横切常青层 X0 清单（18 条）

`README.md`、`AGENT.md`、`docs/PES2TS_开发宪法_v1.md`、`docs/README.md`、`docs/plans/PES2TS_阶段定义_G0-G3.md`、`docs/plans/PES2TS_文档修订台账.md`、`docs/plans/PES2TS_整体开发与发布方案_v2.md`、`docs/plans/PES2TS_完整开发流方案_整合版_v1.md`、`docs/design/PES2TS_G_V_MAPPING_v1_20261018.md`、`docs/design/PES2TS_CODE_MAP_v2_20261018.md`、`docs/design/decisions/ADR-0025-G-V双轴映射与战役注册.md`、`docs/archive/README.md`、`docs/superseded/README.md`、`archive/v5x_scripts/README_scripts_convention.md`、`outputs/README.md`、`requirements.txt`、`requirements-pesn-runtime.txt`、`requirements-pesn-training.txt`。

> 横切层规则：**只更新、不归档**。它们的价值在"始终是当前事实"，历史版本走 `docs/archive/` 快照或台账增量记录；档案条目只引用其"当时版本"（如宪法 v2.9 @ 10-21）。

---

*本文件为 task-6 交付；`ARCHIVE_MAP.csv`（253 条映射）与 `ARCHIVE_TREE_OUTLINE.txt`（拟建文件树）同目录。只读仓库，未修改任何既有文件。*
