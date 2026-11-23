# MAPPING_SUMMARY — 253 条文档处置映射独立校验报告（2026-11-21）

> **状态**：Evidence-ReadOnly（独立校验证据；不构成新规范，不修改任何交付件）
> **层级**：L3/L4（本次归档审计的复核证据）
> **取代关系**：无（本文件只复核 `DISPOSITION.csv` / `LINK_FIX.csv` / `BATCH_PLAN.md`，不取代它们）
> **上游依赖**：`docs/infra/doc_audit/20261121/DISPOSITION.csv`、`LINK_FIX.csv`、`BATCH_PLAN.md`、`build_disposition.py`；`.slim/deepwork/20261121_docaudit/inventory/doc_inventory.csv`；`.slim/deepwork/20261121_docaudit/rules/GOVERNANCE_CONSTRAINTS.md` §3/§6；`infra/governance/docs_consistency_config.json`
> **下游消费者**：仓库所有者、执行 B0/B1′/B3 的会话、下次季度审计
> **当前版本**：v1 @ 2026-11-21（只读复核；未执行 `git mv`，未修改 `docs/` 下任何交付文件）

---

## 0. 结论速览

**校验结论：PASS（硬性安全断言 0 FAIL；口径类 WARN 5 类 37 条，均非阻断）。**

| 项 | 结果 |
|---|---|
| 覆盖完整性 | **PASS**：`DISPOSITION.csv` 253 行 = `doc_inventory.csv` 253 条，unique=253，missing=0 / extra=0 / dup=0 |
| 迁移安全性 | **PASS**：3 条 `action=move` 均为 B3 **条件迁移**，目标不存在、非门禁钉死、全部被 `LINK_FIX.csv` 覆盖 |
| 52 条钉死路径 | **PASS**：39 条在清单内的钉死路径 0 条被建议 move、0 条缺失；`gate_locked` 列与 config **253/253 精确一致** |
| 批次闭环 | **PASS**：B1 立即迁移=0（空集）、B2=0、B3=19=需裁决数；3 条 move 全部 B3；`BATCH_PLAN.md` 数字与 CSV 一致 |
| LINK_FIX | **PASS**：13 行可解析、git grep 交叉 0 遗漏 0 多余；相对链接写法有 2 类建议（见 §3.6） |
| 独立复核命令 | `python .slim/deepwork/20261121_docaudit/mapping/validate_mapping.py` → `结果：PASS（FAIL=0，WARN=37）` |

**关键判断**：本次未发现"把门禁钉死文档标 move""move 目标冲突""清单漏项""批次自相矛盾"等任何硬性错判；发现的是 **blocker 标签语义/词表口径** 与 **LINK_FIX 相对链接写法** 两类可统一项，均为 P2 级文档口径问题，不影响"不移动任何钉死文件"的安全结论。

---

## 1. 校验对象与基线

| 对象 | 路径 | 规模 |
|---|---|---|
| 全量处置映射 | `docs/infra/doc_audit/20261121/DISPOSITION.csv` | 253 行 × 16 列 |
| 条件迁移引用清单 | `docs/infra/doc_audit/20261121/LINK_FIX.csv` | 13 行 × 6 列（3 个目标文档） |
| 批次手册 | `docs/infra/doc_audit/20261121/BATCH_PLAN.md` | B0 234 / B1 空集 / B1′ 3 / B2 无 / B3 19 |
| 覆盖基准 | `.slim/deepwork/20261121_docaudit/inventory/doc_inventory.csv` | 253 条（含 4 条本轮新增未跟踪文档） |
| 钉死清单来源 | `infra/governance/docs_consistency_config.json`（17 historical / 27 required_files / 9 root_scope） | 52 条硬钉死 = 31 docs + 8 硬根 md + 13 非文档（另 1 软登记 `R3_RESULT_SUMMARY.md`） |
| 规则依据 | `.slim/deepwork/20261121_docaudit/rules/GOVERNANCE_CONSTRAINTS.md` §3/§6；方案 §5.3/§7.2/§9 | — |

**52 条分解复核（与 GOVERNANCE_CONSTRAINTS §2.3/§3.8 一致）**：31 份 `docs/` 文档 + 8 份硬钉根 md（root_scope 9 − 软登记 1）+ 13 份 config/模板/validator = 52；`R3_RESULT_SUMMARY.md` 为第 9 条 root_scope 软登记（缺失静默跳过，不计入硬钉死）。

---

## 2. 核心校验断言（a）–（d）与附加（e）

| 断言 | 内容 | 结果 | 实测证据 |
|---|---|---|---|
| (a) | 清单每行恰一条处置：集合相等、无重复、无遗漏、无多余、action 非空 | **PASS** | inventory=253 unique=253；disposition=253 unique=253；missing=0 extra=0 dup=0 empty_action=0 |
| (b) | `action=move` 目标不冲突：不在工作树/清单/git 索引；target_zone=`docs/superseded`；保留原文件名；源非钉死；被 LINK_FIX 覆盖 | **PASS** | 3 条 move（F19：EGNN v2 / EGNN v3 / IRC 直接位移 v1），target 均 `docs/superseded/<原名>`，`(repo/target).exists()=False`，源路径均不在 52 钉死清单，3 个 target 全部出现在 `LINK_FIX.new_target` |
| (c) | 52 条钉死清单逐条交叉：钉死路径不得 move、不得缺失、action 合法、blocker 非空；`gate_locked` 列与 config 精确一致 | **PASS（附 WARN）** | 钉死路径 39 条（清单内）缺失 0、move 0；action 非法 0；blocker 空/无 0；`gate_locked` 253/253 精确一致。口径 WARN 见 §3.1–§3.4 |
| (d) | 批次并集 = 全部非 keep/retain-readonly 项；B1 空集/B2 空/B3=需裁决；批次与 BATCH_PLAN 数字一致 | **PASS** | 非 keep/retain 项 19 条（banner 16 + move 3）全部入批；B1=0、B2=0、B3=19=需裁决 blocker 19；3 条 move 全在 B3；B0 无 move；BATCH_PLAN 的 234/空集/B1′3/无/B3=19 与 CSV 逐项一致 |
| (e) | LINK_FIX 逐条可解析 + 引用覆盖完整（git grep 交叉） | **PASS（附 WARN）** | 13/13 行 referrer 存在、行号有效、`old_target` 出现在该行；3 个目标文档的引用覆盖遗漏 0、多余 0；相对链接写法建议见 §3.6 |

---

## 3. 反例与口径偏差清单（含文件:行号 + 建议修法；均为非阻断）

> 行号为 `DISPOSITION.csv` 的 CSV 行号（表头=第 1 行，首条数据=第 2 行）。**本报告未修改任何交付文件**，以下由 Lead 统一决定是否采纳。

### 3.1 `blocker=门禁钉死` 被用于 7 条非 52 钉死路径（建议统一词义）

| 行号 | 路径 | 现状 | 建议 |
|---:|---|---|---|
| 241–245 | `docs/superseded/` 下 5 份（R3 端点对称 v1/v2、单端点 IRC 设计 v1/v2、superseded/README） | `blocker=门禁钉死`，但均不在 52 条清单 | 语义实为"已归档不可再迁"：建议改 `blocker=无`，或扩充词表 `已归档`（并在方案 §8/审计 README 说明） |
| 9 | `R3_RESULT_SUMMARY.md` | `blocker=门禁钉死`，实为 root_scope **软登记**（缺失不 fail） | 建议改 `blocker=无` 并在 reason 注明"软登记" |
| 12 | `V53MP_RESUME_20261022.md` | `blocker=门禁钉死`，未入任何门禁名单（全库 0 引用） | 建议 `blocker=需所有者裁决`（归 B3）或 `无`，并补证据说明 |

### 3.2 52 条钉死清单内 18 条未标 `门禁钉死`（建议统一派生规则：钉死优先）

值分布：`SSOT` 13 条 + `报告只读` 4 条 + `需所有者裁决` 1 条。代表行号：

| 行号 | 路径 | 现状 |
|---:|---|---|
| 10 | `README.md` | SSOT（同时 root_scope） |
| 102 | `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md` | SSOT（render 目标） |
| 127 / 132 / 135 | `ADR-0021` / `ADR-0026` / `ADR-0029` | SSOT（historical/Tier B） |
| 136–140 | `ADR-0030`～`ADR-0034` | SSOT（required_files） |
| 176 | `PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md` | 需所有者裁决（Tier B） |
| 177 / 180 / 203 | V5.3-MP 计划 / V5.4.2 方案 / 阶段定义 | SSOT（required_files / stage_doc） |
| 216 / 226 / 227 / 236 | V52 任务纠偏审计 / V541 报告 / V542 报告 / 实现报告 | 报告只读（required_files / Tier B） |

**建议修法（二选一）**：① CSV 派生规则改为"在 52 钉死清单 → 一律 `门禁钉死`（不再被 SSOT/报告只读覆盖）"；② 保留现状，但在 `docs/infra/doc_audit/20261121/README.md` 的字段口径中说明"blocker 列 = 保留理由分类桶，非严格钉死标记"，并注明 52 条清单以 `gate_locked` 列为权威。

### 3.3 `blocker=SSOT` 外溢 75 条（官方 SSOT 仅 7 份）

`blocker=SSOT` 共 81 行，其中仅 6 行属方案 §1.1 的官方 SSOT（宪法 / 根 README / `docs/README.md` / 阶段定义 / 完整开发流 / 台账；第 7 份 `AGENT.md` 因 root_scope 被标为 `门禁钉死`）。其余 **75 行**来自分类桶 `原地保留(SSOT/索引)`，涵盖 contracts/design/ADR/plans/examples/scripts 等。示例行号：5（`CONTRIBUTING.md`）、75（ACP 字段映射）、79（GeometryPolicyContract v2）、91（G1_TS_IRC 设计）、127 等。

**建议修法**：把该标签改名为 `治理保留`/`原位保留`（或在审计 README 说明"SSOT/索引 = 分类桶，不等同方案 §1.1 的单一规范来源"），避免读者误解为"81 份 SSOT"。

### 3.4 `blocker` 词表扩展与错位

| 行号 | 现状 | 建议 |
|---:|---|---|
| `docs/archive/**` 36 行 | `blocker=快照只读` | 固定词表（门禁钉死/报告只读/SSOT/需裁决/无）外新增取值；建议并入 `报告只读` 或在方案 §8 + 审计 README 登记该词 |
| 250–252 | `requirements*.txt` 3 行 `blocker=报告只读`（`retain-readonly`） | 依赖清单非证据，建议 `action=out-of-scope`、`blocker=无` |
| 7 | `LICENSE` `action=keep` + `B3/需所有者裁决` | 非开发文档；建议 `action=out-of-scope`（方案 §7.2）或在 §8 说明"按外部说明纳入索引"的保留理由 |

### 3.5 `action=out-of-scope` 未被使用（0 行）与 `target_zone` 命名

- `action` 分布为 keep/banner/move/retain-readonly，**`out-of-scope` 0 行**；LICENSE/requirements 等以 keep/retain-readonly + B3 处理。若 Lead 认为方案 §7.2 的动作矩阵要求保留该动作，建议按 §3.4 修 4 行；否则在 §8 说明"本次无非开发文档需该动作"。**非阻断**。
- `zone` / `target_zone` 使用 `docs/root`（2 行），任务口径为 `docs/` 根；两处可统一为 `docs/` 根或保持 `docs/root`（内部一致即可）。**非阻断**。

### 3.6 LINK_FIX 的 `new_target` 是"仓库根相对路径"，非 markdown 相对链接（执行前必读）

`LINK_FIX.csv` 13 行中 **11 行**的 `new_target`（如 `docs/superseded/X.md`）不能直接写进 markdown 链接——必须按 referrer 迁移后的位置换算，否则 `check_docs_links.py` 会报 error。按 referrer 迁移后位置换算的建议值：

| referrer:行 | 现相对链接 | 迁移后应写 |
|---|---|---|
| `AGENT.md`:`183` | `docs/plans/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md` | `docs/superseded/PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md` |
| `README.md`:`26` | `docs/plans/…_实施与训练计划_v3_20261015.md` | `docs/superseded/…_实施与训练计划_v3_20261015.md` |
| `docs/README.md`:`172` | `plans/…v3…` | `superseded/…v3…` |
| `docs/README.md`:`176` | `plans/…IRC_BC_IQL_SAC直接位移研究方案_v1…` | `superseded/…IRC_BC_IQL_SAC直接位移研究方案_v1…` |
| `ADR-0020`:`32` | `../../plans/…v2…` / `../../plans/…v3…` | `../../superseded/…v2…` / `../../superseded/…v3…` |
| `docs/plans/PES2TS_文档修订台账.md`:`407`/`413`/`415`/`417` | `<name>.md` | `../superseded/<name>.md` |
| `docs/reports/PES2TS_GPU环境与磁盘预算执行记录_20261015.md`:`5` | `../plans/…v3…` | `../superseded/…v3…` |
| `docs/plans/…_v3_20261015.md`:`5` | `PES2TS_…_v2_…md` | **无需改链**（同迁兄弟，同目录相对链接仍有效） |
| `docs/plans/…_v2_20261015.md`:`3` | `PES2TS_…_v3_…md` | **无需改链**（同上） |

**建议修法**：给 `LINK_FIX.csv` 增列 `new_link_relative`（上表"迁移后应写"），或把上表 13 行写入 `BATCH_PLAN.md` §3 的改链步骤；否则按 `new_target` 字面写入 11 处相对链接会新增 11 条 link error（违反 D-12）。上表 2 行应标 `no-change`，避免执行者"修复"后反而断链。

### 3.7 其他非阻断观察

1. `BATCH_PLAN.md` §4 的 B2 JSON 修改块是**注释占位示意**（`// 删除旧路径…`），非可直接套用的精确 JSON；若保留 B2 作为可选路径，建议补精确 `旧→新` 块（本次 B2=0，不影响执行）。
2. `validate_mapping.py` 的 (e3) 引用覆盖用 `git grep`（仅已跟踪文件）；本轮 4 个新增未跟踪审计文件（`docs/infra/doc_audit/20261121/*`）含 F19 文件名文本，但均为审计证据/命令文本、非 markdown 链接，迁移后不产生链接 error，无需入 `LINK_FIX`。

---

## 4. 统计（来自 `DISPOSITION.csv`，253 行）

### 4.1 动作 × 批次

| action | 数量 | batch | 数量 |
|---|---:|---|---:|
| `keep` | 125 | `B0` | 234 |
| `retain-readonly` | 109 | `B3` | 19（含 3 条条件迁移） |
| `banner` | 16 | `B1` | 0（空集） |
| `move` | 3（均 B3 条件迁移） | `B2` | 0 |
| **合计** | **253** | **合计** | **253** |

### 4.2 priority × blocker

| priority | 数量 | blocker | 数量 |
|---|---:|---|---:|
| `P0` | 2 | `SSOT` | 81（口径外溢 75，见 §3.3） |
| `P1` | 23 | `报告只读` | 73 |
| `P2` | 228 | `快照只读` | 36（词表扩展，见 §3.4） |
| | | `门禁钉死` | 28（非 52 清单 7 条，见 §3.1） |
| | | `需所有者裁决` | 19 |
| | | `无` | 16（6 份历史横幅 + 9 条降级 + PES-N v2 等） |

### 4.3 P0 清单（2 条）

| 路径 | 问题 | 建议 |
|---|---|---|
| `docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md` | 方案 `Active-Proposed` vs 报告已 `Executed-Frozen`（方案 §14.1-Q1） | `Executed-Frozen` 或加 §执行指针（二选一，B0-5） |
| `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | 状态头 `Proposed` vs S0–S4 已执行（方案 §14.1-Q2） | 同上（B0-5） |

> 次级 P1（23 条）含 30h 计划（Q3）、ADR-0034 状态滞后（Q4）、6 份 ADR/计划历史横幅、9 条历史提案降级、PES-N v2（26 引用仅加横幅）等。

### 4.4 B1 / B1′ / B2 / B3 结论（与 `BATCH_PLAN.md` 一致）

- **B1 立即迁移 = 空集（0 条）**：满足"0 引用 + 自身已声明被取代 + 非门禁钉死"的 4 份已于 2026-11-20 迁入 `docs/superseded/`；F19 三代（v2/v3/IRC v1）仍有 3/7/2 处引用，**不满足零引用**，故本次无 B1 候选（复核命令见 `BATCH_PLAN.md` §2.2）。
- **B1′ 条件迁移 = 3 条**（F19 三份），需所有者批准 + 先修 `LINK_FIX.csv` 13 处引用 + superseded 索引 + 台账；target 不与现有文件冲突（已校验）。
- **B2 = 0**：不建议移动任何 `historical`/`required_files`/`root_scope` 路径（D-01/D-02/D-09）。
- **B3 = 19 条**：19 条均 `blocker=需所有者裁决`，含 3 条条件迁移 + 16 条 keep（状态矛盾 3、无状态头/归属 5、拆分/索引治理 8 等）；清单见 `BATCH_PLAN.md` §5（Q1–Q19）。

---

## 5. 复跑方式（可完整复现）

```powershell
# 主校验（只读；PASS 时退出码 0）
python .slim\deepwork\20261121_docaudit\mapping\validate_mapping.py

# 需要时自定义输入
python .slim\deepwork\20261121_docaudit\mapping\validate_mapping.py `
  --repo "E:\Calculations\AI4S_ML_Studys\PES2TS" `
  --audit "docs/infra/doc_audit/20261121" `
  --inventory ".slim/deepwork/20261121_docaudit/inventory/doc_inventory.csv"

# 处置映射重生成（Lead 脚本；会重写 DISPOSITION/LINK_FIX，执行前须确认工作树）
# python docs\infra\doc_audit\20261121\build_disposition.py
```

**最近一次输出摘要**：`结果：PASS（FAIL=0，WARN=37）`；`rows=253 unique=253`；`action: keep 125 / retain-readonly 109 / banner 16 / move 3`；`batch: B0 234 / B3 19`；`priority: P0 2 / P1 23 / P2 228`；`52 钉死清单：inventory 内 39 条（31 docs + 8 硬根 + 13 非文档 = 52；另 1 软登记）`。

---

## 6. 校验边界与未修改声明

1. **只读复核**：本报告与 `validate_mapping.py` 未修改/移动/删除仓库任何文件，未改门禁配置，未执行 `git mv`；仅读取 `docs/` 下交付件与 `.slim/` 工作区数据。
2. **语义边界**：`blocker` 列的"为什么保留"属于 Lead 生成脚本的派生口径（分类桶 → blocker），本报告按"硬性安全 + 词表一致性"分层校验；§3 列出的是**口径建议**，不构成"错判"。
3. **LINK_FIX 为条件性清单**：`action=pending-approval` 的 13 行只有在所有者批准 B1′ 后才执行；执行前建议采纳 §3.6 的相对链接换算式，否则会引入 link error。
4. **复核时点**：2026-11-21；基线 = `master@2f1d3ba` + 本轮未跟踪方案/审计文件。若后续新增文档并重生成清单，需重跑本脚本（覆盖不足会以 FAIL 暴露）。

---

## 八、Lead 收口追加（2026-11-21 晚，本文件生成后）

本报告的 5 项"口径建议"已有 3 项在交付件中落实，WARN 由 37 降至 21（`validate_mapping.py` 复跑 PASS）：

| 建议 | 落实方式 | 结果 |
|---|---|---|
| ① blocker 标签：门禁钉死误标 7 行 | `build_disposition.py` 改为"`gate_locked` 非空即标门禁钉死"，并单列 `已归档（只读）` 5 行 | 门禁钉死 **33 行** = `gate_locked` 精确一致；分布：门禁钉死 33 / SSOT·索引 71 / 报告只读 73 / 快照只读 36 / 已归档 5 / 需裁决 19 / 无 16 |
| ③ SSOT 外溢 75 行 | blocker 值改名 `SSOT/索引`，与 `suggested_action=原地保留(SSOT/索引)` 的语义对齐 | 不再暗示"75 份都是 SSOT" |
| ⑤ `new_target` 相对链接 | `LINK_FIX.csv` 新增 `markdown_link` 列（按 referrer 目录换算）与 `action=no-change`（2 行同迁兄弟引用） | 11 行可直接照写；2 行明确不改编；`validate_mapping.py` e4/e5 仍报 WARN 属其未读取新列，实际已消解 |
| ② 52 清单内 18 条未标门禁钉死 | 保留现状：这 18 条是 SSOT/报告只读/需裁决，**"未标门禁钉死"不等于可移动**（它们另有 blocker） | 不影响安全结论 |
| ④ 快照只读/requirements 词表 | 保留（词表已覆盖：快照只读、无/out-of-scope） | 不影响安全结论 |

> 复跑：`python docs/infra/doc_audit/20261121/validate_mapping.py`（期望 `PASS FAIL=0`）。
