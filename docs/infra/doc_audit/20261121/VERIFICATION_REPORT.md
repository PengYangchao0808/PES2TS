# VERIFICATION_REPORT — 开发文档规范化归档方案 v2.0 独立复核报告（2026-11-21）

> **状态**：Evidence-ReadOnly（独立复核证据；不修改任何交付件，结论供仓库所有者决定是否发布）
> **层级**：L3 证据 / L4 运营（复核面，不涉及 G0–G4 能力与 V 战役语义）
> **取代关系**：无（首次独立复核；不取代方案、CSV、BATCH_PLAN 与 MAPPING_SUMMARY）
> **上游依赖**：[开发文档规范化归档方案 v2.0](../../PES2TS_开发文档规范化归档方案_v2_20261121.md)、[DISPOSITION.csv](DISPOSITION.csv)、[GOVERNANCE_CONSTRAINTS.md](GOVERNANCE_CONSTRAINTS.md)、[MAPPING_SUMMARY.md](MAPPING_SUMMARY.md)、`scripts/agent_ops/docs_consistency_config.json`、`config/registry/g_v_map.yaml`
> **下游消费者**：仓库所有者、方案作者（修复必修项）、下次季度审计
> **当前版本**：v1 @ 2026-11-21（对抗式独立复核；基线 master@2f1d3ba + 未提交交付物；复核时点 2026-11-21T22:37–22:56 +08:00）

---

## 0. 结论

**结论：有条件通过（CONDITIONAL PASS）。**

- **硬合规面全部通过**：253 条清单与处置一一对应、0 重复 0 缺失；52 条硬钉死路径**无一条被建议移动**；`docs/reports/**` 36/36 全部 `retain-readonly`；`docs/archive/**`、`state/gates`、冻结件均未被触碰；本方案与审计**未修改门禁配置/脚本/CI、未改报告正文、未移动任何文件**（`git status` 仅方案 + 审计目录 + README + 台账，见 §3.6）。
- **4 门实测最终全绿**：`docs_consistency --strict` = **scanned 169 / 0 violation / CLEAN**；`check_docs_links` = **checked 193 docs / 0 error / 172 warning**（本报告落盘后由 2 error 归零，见 §3.2）；`render_gv_map --check` = **current**；pytest = **26 passed**。
- **但存在 2 项 P1 可执行性缺陷 + 若干 P2 自洽性/覆盖缺口**（见 §4），其中回滚命令实测无效、PowerShell heredoc 实测报错属"执行手册可照做"（方案 §13.1-DoD#7）的阻断项；清单快照未覆盖审计目录最终新增 md 属 DoD#1 的交付范围缺口。**这 2 项 P1 修好前不建议按方案原样执行 B1′/B2 或发布为可执行手册**；方案作为"提案 + 证据包"发布已具备条件。
- **发布前必修项：P1 共 3 条、P2 共 11 条**（最小清单见 §5）。本报告**不代修**。

**门禁 PASS/FAIL 总览**：完整性 PASS；合规性 PASS（标记完整性有 P2 缺口）；自洽性 **FAIL**（3 处数字/引用不自洽，均为 P2 级，均易修）；可执行性 **FAIL**（2 项 P1）；门禁实测 PASS（4/4 最终全绿）；缺口项 **部分 PASS**（5 项关闭 / 4 项仍缺）。

---

## 1. 复核对象、基线与"审计后变更"记录

### 1.1 复核对象（全部实读，非转述）

| 对象 | 规模/位置 | 复核方式 |
|---|---|---|
| 方案主文档 | 858 行 / 70,397 B | 全文分段读，重点 §2/§5/§6/§7/§8/§9/§10/§13/§14 |
| 审计证据目录 | 20 文件（含 8 份 md + 2 份复核脚本） | 逐文件抽读 + 程序化交叉校验 |
| `doc_inventory.csv` / `DISPOSITION.csv` | 253 / 253 行 | 集合、重复、状态词表、动作/批次/优先级/blocker 计数 |
| `GOVERNANCE_CONSTRAINTS.md` | 535 行 | §3 钉死清单逐条、§6 D-01…D-22 逐条 |
| `BATCH_PLAN.md` / `LINK_FIX.csv` / `MAPPING_SUMMARY.md` | 184 行 / 13 行 / 210 行 | 命令语法、路径存在性、行号与目标、覆盖交叉 |
| `docs/README.md` / `PES2TS_文档修订台账.md` | 新增 2026-11-21 节 | 登记、链接、字段、数字一致性 |
| `.slim/deepwork/20261121_docaudit/**` | inventory/rules/disposition/mapping | 与交付副本一致性抽查 |

### 1.2 复核基线

- `master@2f1d3ba`，工作树复核开始时仅预期未提交变更（`M docs/README.md`、`M 台账`、`?? 方案`、`?? doc_audit/`）。
- 复核期间**共享工作区出现并发变更**，以下 3 项"审计后修复/新增"已纳入本报告（Lead 2026-11-21 消息 + 文件 mtime 实证）：
  1. **GOVERNANCE_CONSTRAINTS.md:251 修复**（22:37:48）：原"禁用词登记"行含 6 个禁用语原文且无同行豁免标记，导致 `docs_consistency --strict` 报 6 violation（复核首测实测）；Lead 在该行追加"禁止"豁免标记后 CLEAN。**本报告记录：复核首测 FAIL（6 violation）→ 修复后 PASS（0 violation）**；修复属审计证据行，非门禁配置/脚本改动，不违反 D-19。
  2. **新增独立校验证据**：`MAPPING_SUMMARY.md`、`validate_mapping.py`、`build_disposition.py`；`LINK_FIX.csv` 增 `markdown_link`/`action` 列（11 行 `pending-approval` + 2 行 `no-change`）；`DISPOSITION.blocker` 精化为：门禁钉死 33 / SSOT·索引 71 / 报告只读 73 / 快照只读 36 / 已归档 5 / 需裁决 19 / 无 16（合计 253）；方案 §8.1 已同步。
  3. **本报告自身创建**消解 `check_docs_links` 的 2 条 error（均指向 `VERIFICATION_REPORT.md` 尚未落盘）。
- **复核未修改任何交付件**；工作笔记与原始输出在 `.slim/deepwork/20261121_docaudit/verify/`。

---

## 2. 必做复核项：逐项 PASS/FAIL

### 2.1 完整性

| 检查点 | 结果 | 证据 |
|---|---|---|
| inventory 253 条 path ↔ DISPOSITION 253 条 path 集合相等 | **PASS** | 集合差：inventory-only=0、disposition-only=0；两侧 unique=253、dup=0、空 path=0 |
| 每份文档有且仅有一条处置 | **PASS** | 同上；`validate_mapping.py` 亦输出 PASS（FAIL=0、WARN=21） |
| `canonical_status` 是否有"未登记"以外的非法值 | **FAIL（P2）** | 出现第 12 个取值 `运营记录(不要求状态头)` **22 行**，不在 §5.1 的 11 状态封闭词表内；其中 10 行不在 `docs/agent_runs/`（docs/infra 5、root 4、docs/plans 台账 1），且**方案自身行** `canonical_status=运营记录(不要求状态头)` 与其元数据头 `Active-Proposed` 直接矛盾（方案 §8.2 又把该取值列为分布桶，形成"词表封闭"与"实际取值"自相矛盾）。修法见 §5-P2-1 |

### 2.2 合规性

| 检查点 | 结果 | 证据 |
|---|---|---|
| 52 条硬钉死路径在 DISPOSITION 中不可移动 | **PASS** | 程序化重建钉死集：52 = `historical`(17) ∪ `required_files`(27) ∪ `root_scope`(9) + `g_v_map` 6 条新增钉死 − 交集；其中 **40 条 docs/根 md** 全部在 DISPOSITION（0 缺失），`action` 全部 ∈ {keep 31, retain-readonly 7, banner 2}，**0 条 move**；13 条非文档（config/模板/validator）按 §1.1 口径不纳入文档清单（非缺口） |
| 上述钉死路径"blocker 含门禁钉死" | **部分 FAIL（P2）** | `action` 全合规，但 **8 条钉死路径** blocker 非"门禁钉死"（`报告只读` 4、`SSOT/索引` 3、`需所有者裁决` 1）；其中 **6 条** 连 `gate_locked` 列也为空（`PES2TS_实现报告_20261018`、`V541 报告`、`G_V_MAPPING`、`阶段定义_G0-G3`、`ADR-0026`、`V5.1_修复冲刺`）。8 条均为 g_v_map/render-stage 类钉死，属"未标来源"而非"建议移动"；MAPPING_SUMMARY §3.2 已自报同类 WARN（18 条），Lead 精化后仍余 8/6。修法见 §5-P2-2 |
| `docs/reports/**` 全部 retain-readonly | **PASS** | 36/36 行 `action=retain-readonly`、`canonical_status=Evidence-ReadOnly`、blocker=`报告只读` |
| D-01…D-22 无违反 | **PASS**（含 1 项修复确认） | D-01/D-02/D-04/D-05/D-09：钉死/根 memo 0 建议移动；D-03：G–V 渲染目标 0 手改（`render --check` current）；D-06：archive 36 条 Snapshot 只读；D-07：`docs/reports/**` 0 修改（git status）；D-08：0 条 reports/archive/historical 建议迁 superseded；D-10：无 ignore 路径入库；**D-11：复核首测 FAIL（6 violation @ GOVERNANCE:251）→ 已修复 → PASS**；D-12：0 error（报告落盘后）；D-13/D-14：方案 6 必填字段齐全、版本四方一致（文件名 v2 / 状态 v2.0 / README v2.0 / 台账 v2.0）；D-15：README + 台账均登记；D-16：0 改名；D-17：未新增第三索引（G5 为规格未实施）；D-18：G/V 轴未混淆；D-19：门禁配置/脚本/CI 0 改动；D-20：R1–R6 未触碰；D-21：未把写计划当能力升级；D-22：4 门复跑（见 §3） |
| 52 条"引用条数对得上" | **FAIL（P2）** | `52 条（31 docs + 9 根 md + 13 非文档）` 三项相加 = **53 ≠ 52**（方案 §2.3、GOVERNANCE §3.0/附录 B、`docs/README.md`、台账同源错误）。正确分解为 **31 docs + 8 硬钉根 md + 13 非文档 = 52**（`R3_RESULT_SUMMARY.md` 为软登记不计）；`MAPPING_SUMMARY.md` 与 `validate_mapping.py` 已按 31+8+13 输出，四份上游文档需同步。修法见 §5-P2-3 |

### 2.3 自洽性

| 检查点 | 结果 | 证据 |
|---|---|---|
| 状态机 11 状态均有"定义 + 允许目录 + 门禁判定" | **PASS** | §5.1 表 11 行三列齐全（Canonical/Active-Normative/Accepted/Active-Proposed/Frozen-Preregistered/Executed-Frozen/Superseded/Historical/Evidence-ReadOnly/Snapshot/Deprecated）；§5.2 转换事件 10 行、不可逆规则明确；`Deprecated` 定义存在、当前 0 条使用（合法） |
| §5.3 旧→新映射覆盖主要旧值 | **部分 FAIL（P2）** | §5.3 覆盖 `权威/Canonical/Accepted/已接受/Proposed*/设计稿/执行方案/批次方案/Frozen*/Executed-Frozen/Superseded/历史提案/前版记录/历史/Historical/Evidence-Historical/参考/只读报告/Evidence/IMPLEMENTED*/implemented/fixture_passed/Review/Draft` 等主流值；但 DISPOSITION 中 **16 行有可解析 declared_status 却落 `未登记`**（例：`参考（非规范）`、`Active`、`Closed`、`设计完成`、`已实施`、`规划完成`、`修订设计 v1`、`设计提案`、`方案设计完成`、`配套开发设计，待实现与真实冒烟`、`建议状态：...`），说明 §5.3 未覆盖"过程性/长尾描述"这一整类；§8.2 却称 50 条未登记即"无状态头、无法归入"（不精确：其中 16 条有状态头）。修法见 §5-P2-4 |
| 动作/批次/优先级计数与 DISPOSITION 一致 | **PASS（动作/优先级）** | action：keep 125 / retain-readonly 109 / banner 16 / move 3 = 253（与 §8.1、BATCH_PLAN 数值一致）；batch：B0 234 / B3 19 = 253；priority：P0 2 / P1 23 / P2 228 = 253；blocker 精化后 33+71+73+36+5+19+16=253 |
| 批次数字自洽 | **FAIL（P2）** | ① `BATCH_PLAN.md §0` 写 "B0（234）= `keep` 125 + `retain-readonly` 109 + `banner` 16"，三项和 **250 ≠ 234**；实际 B0 仅 `keep` **109**（另 16 keep 属 B3）。② 方案 §8.1 批次表并列 `B1′ = 3` 与 `B3 = 19`，但 DISPOSITION 中 3 条 move 的 `batch` 均为 **B3** → 234+3+19=256≠253（B1′ 是 B3 的子集未注明）。修法见 §5-P2-5 |
| 方案自身 8 字段元数据头合规 + README/台账登记 | **PASS** | 方案头部：状态/层级/取代关系/上游依赖/下游消费者/当前版本 6 必填齐全（`归档批次`/`索引键` 2 可选未填，符合 §6.1"后 2 个可选"）；`docs/README.md:3-8` 新增 2026-11-21 节、台账 `:3-10` 新增增量登记，两处链接均可达（本报告落盘后 0 error） |
| 数字引用一致性（§8.6 / §14） | **FAIL（P2）** | §8.6 称"真正需要决策的是 **19 项（§14）**"，但 §14.1 实际只有 **11 项**（Q1–Q11）；19 项在 `BATCH_PLAN.md §5`（Q1–Q19）。交叉引用错位（且两处 Q 编号体系不同，读者会误配）。修法见 §5-P2-6 |
| 审计证据目录路径一致性 | **FAIL（P2）** | §6.3 目录表与附录 B 写 `docs/doc_audit/YYYYMMDD/`，实际交付为 `docs/infra/doc_audit/20261121/`（方案自身链接、§6.4 目录树、§9.3 迁移记录路径又指向 infra 下）；目录规范与实际布局不一致。修法见 §5-P2-7 |

### 2.4 可执行性

| 检查点 | 结果 | 证据 |
|---|---|---|
| `git mv` 源路径真实存在 / 目标目录存在 | **PASS** | 3 条 move 源文件均存在；`docs/superseded/` 与其中 `README.md` 均存在；`target_path` 均为合法新路径 |
| 回滚命令正确 | **FAIL（P1）** | 实测（临时 git 仓库）：`git mv docs/a.md docs/superseded/a.md` 后执行 `git checkout -- .`，状态仍为 `R  docs/a.md -> docs/superseded/a.md`，**原路径未恢复**（staged rename 不受 worktree checkout 影响）；`git clean -fd docs/superseded` 因文件已入索引也无法清理 → **B1/B1′ 按现命令无法回滚**。正确最小修法：`git reset --hard HEAD`（本批未提交场景）或 `git restore --staged --worktree .`（实测可恢复原位，附 `git clean -fd docs/superseded` 清未跟踪件）。修正位置：方案 §9.2/§9.5、`BATCH_PLAN.md §3/§6`、`MIGRATION_LOG.md `:26-27`（共 6 处）。修法见 §5-P1-1 |
| PowerShell 语法正确（全命令） | **FAIL（P1）** | 方案 §9.2 前置检查用 **bash heredoc** `python - <<'PY' ... PY`，PowerShell 7 实测报 `Missing file specification after redirection operator`（`<<` 为保留重定向符），该段不可直接照做。修法：改为 `python -c "..."` 或 `.slim/.../check_pins.py` 独立脚本。其余 `git mv`/`git grep`/`Select-String`/`python scripts/...` 命令语法正确。修法见 §5-P1-2 |
| 门禁命令都存在且当前可运行 | **PASS** | 4 条命令脚本均存在，实测 exit 0/0/0/0（见 §3）；`validate_mapping.py` 可运行并 PASS（FAIL=0/WARN=21）；`build_disposition.py` 可复跑（审计目录内） |
| `D:\` 与中文引号安全 | **PASS** | 全文未出现 `D:\` 硬编码；路径使用仓库相对路径；中文文件名在 PowerShell 中均以引号包裹（实测 `git mv "中文路径"` 可行） |

### 2.5 门禁实测（只读复跑，未改任何文件）

| # | 门禁 | 期望 | 实测 | 判定 |
|---|---|---|---|---|
| 1 | `python scripts/agent_ops/docs_consistency.py --strict` | 0 violation / CLEAN | 首测（22:37）：`scanned 167; 6 violation(s) [forbidden_token=6]; VIOLATIONS`；**修复后终测（本报告落盘后，22:55）：`scanned 169 file(s); 0 violation(s); CLEAN`** | **PASS（经审计后修复）** |
| 2 | `python scripts/check_docs_links.py` | 0 error；warning ≤172 | 本报告落盘前：`checked 192 docs; 2 error(s), 172 warning(s)`；**本报告落盘后：`checked 193 docs; 0 error(s), 172 warning(s)`** | **PASS（本报告消解 2 error）** |
| 3 | `python scripts/render_gv_map.py --check` | current | `G–V mapping doc is current` | **PASS** |
| 4 | `python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py` | 26 passed | `26 passed in 1.12s` | **PASS** |
| 附 | `python docs/infra/doc_audit/20261121/validate_mapping.py` | PASS | `结果：PASS （FAIL=0，WARN=21）` | **PASS** |

**warning 口径**：172 与基线完全一致（101 `E:/` + 71 local-only），新增方案/审计文档**未新增 warning** → 满足 D-12/C-11"不得静默增长"。

### 2.6 缺口

| 检查点 | 结果 | 证据/说明 |
|---|---|---|
| doc_type 覆盖（HTML 证据、model_card、examples、scripts/data/outputs README、LICENSE/requirements） | **PASS（分类层）/ 部分缺（处置层）** | 253 清单已纳入：`.html` 4（3 docs + 1 examples）、`model_card.md` 8、`examples/**` 3、`scripts/README` 2、`data/README`+`outputs/README` 2、`LICENSE` 1、`requirements*.txt` 3；方案 §1.1 纳入表与 §7.5 均覆盖 HTML/二进制/大文件策略；但 **8 份 model_card 被标 `未登记`**（按不变式 1/§4.2-L3 应为 `Evidence-ReadOnly`），且 examples/scripts/data-outputs/root 产品文档的"是否需状态头"未在 §6.1 明确（§6.1 只强制 plan/design/contract/infra）。见 §5-P2-8 |
| "未登记 50 条"后续动作明确 | **部分 FAIL（P2）** | §8.2 称 50 条是 "B0-6/B0-8 直接工作清单"，但 B0-6（12 份 contracts/demo/design）+ B0-8（5 份）= **17 份**，其余 33 份（root 6、examples 3、scripts 2、data-outputs 2、artifacts 8、docs/root 1、superseded 1、plans/design 其余）**无逐条后续动作**；且 B0-6 声称的 "contracts/demo/design 12 份" 与 DISPOSITION 实际未登记分布（plans 13 / design 11 / demo 2 / contracts **0**）不符。见 §5-P2-9 |
| G5 规格可实现且不与 docs_consistency 重复 | **PASS（规格）/ 需澄清 2 点（P2）** | §10.3 模式/输入/6 校验项/生成物/豁免机制齐全，且显式声明"与既有 docs_consistency 去重，G5 只引用不重复实现"；可复用 `render_gv_map --check` 先例。需澄清：①"非证据类文档"无机器可判据（reports/archive 排除规则已给，但 `docs/infra/doc_audit/` 证据目录未列入豁免默认集，G5 上线会要求其状态头）；②G5 校验项 3"目录相容"与现状冲突：DISPOSITION 现状 17 行会越 §5.1 允许目录集（如 root 4 份 `Evidence-ReadOnly`、`docs/design` 的 `Canonical`/`Executed-Frozen`、`design/decisions` 的 `Active-Proposed`），G5 需先给存量豁免白名单或修订目录相容规则。见 §5-P2-10 |
| §14 覆盖 ANOMALIES 的 P0/P1 | **部分 FAIL（P2）** | P0 2/2 已覆盖（Q1=V5.4.2、Q2=V5.3-MP）；P1 10 条中 §14 直接列 3 条（A03→Q3、A06→Q8 附近、A10→§14.2-1），其余（A04/A05/A07/A08/A09/A11/A12）散落 §8.4/§10/§14.2/BATCH_PLAN §5，未在 §14 建立"ANOMALIES→裁决项"映射表；结合 §8.6"19 项（§14）"错位，读者无法从 §14 完整回收 P1。见 §5-P2-6 |
| 清单快照覆盖最终交付集 | **FAIL（P1）** | `doc_inventory.csv` 采集时点 22:32，仅含 4 条本轮新增（方案 + 审计 README/GOVERNANCE/MIGRATION_LOG 中的 3 条）；最终交付的审计目录还有 `ANOMALIES/FAMILIES/RECOMMENDATIONS/BATCH_PLAN/inventory_stats/MAPPING_SUMMARY` 6 份 md（+本报告共 7 份）**不在 253 条内** → DoD#1"全量清单 100% 覆盖"对最终交付集不成立；`validate_mapping.py` 只校验 CSV↔CSV，发现不了这一层。修法见 §5-P1-3 |

---

## 3. 门禁实测原文（原始输出摘录）

> 环境：Windows 11 / PowerShell 7 / Python 3.12.7；仓库根 `E:\Calculations\AI4S_ML_Studys\PES2TS`；工作区含未提交交付物（§3.6）。

### 3.1 `docs_consistency --strict`（终测，本报告落盘后，22:55）

```text
docs_consistency: scanned 169 file(s); 0 violation(s); CLEAN
```

**首测（22:37，修复前）原文**（保留以存证；行尾"禁止"为同行豁免标记说明）：

```text
VIOLATION docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md:251:TSStop   ← 禁用词登记行，禁止用于活跃文档（行内已有豁免标记）
VIOLATION docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md:251:TS Stop   ← 同上（禁止）
VIOLATION docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md:251:TS-neighborhood   ← 同上（禁止）
VIOLATION docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md:251:approach-side   ← 同上（禁止）
VIOLATION docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md:251:TS-directed   ← 同上（禁止）
VIOLATION docs/infra/doc_audit/20261121/GOVERNANCE_CONSTRAINTS.md:251:No-Move candidate   ← 同上（禁止）
docs_consistency: scanned 167 file(s); 6 violation(s) [forbidden_token=6]; VIOLATIONS
```

### 3.2 `check_docs_links`（终测，本报告落盘后，22:54）

```text
checked 193 docs; 0 error(s), 172 warning(s)
```

**落盘前（22:53）原文**：

```text
ERROR docs\infra\doc_audit\20261121\README.md: missing link target VERIFICATION_REPORT.md
ERROR docs\README.md: missing link target infra/doc_audit/20261121/VERIFICATION_REPORT.md
checked 192 docs; 2 error(s), 172 warning(s)
```

> 说明：2 条 error 均由"预登记本报告但报告尚未落盘"造成；本文件创建后归零，**非方案/清单缺陷**。warning 保持基线 172，无新增。

### 3.3 `render_gv_map.py --check`

```text
G–V mapping doc is current
```

### 3.4 pytest

```text
..........................                                               [100%]
26 passed in 1.12s
```

### 3.5 `validate_mapping.py`（审计目录新证据，辅助门禁）

```text
52 钉死清单：inventory 内 39 条（31 docs + 8 硬根 + 13 非文档 = 52；另 1 软登记）
P0=2  B1 立即迁移=0（空集）  B3 条件迁移=3  B3 裁决=19  需裁决 blocker=19
==============================================================================
结果：PASS  （FAIL=0，WARN=21）
==============================================================================
```

### 3.6 `git status --short`（复核时点，未提交变更清单）

```text
 M docs/README.md
 M "docs/plans/PES2TS_文档修订台账.md"
?? "docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md"
?? docs/infra/doc_audit/
```

> 判定：变更面**严格限于**方案 + 审计目录 + README + 台账；**无 `scripts/**`、`config/**`、`docs/reports/**`、`pes2ts_core/**`、`tests/**`、`.github/**` 改动** —— 符合 D-07/D-19 与"证据只读"。审计目录内 `.py` 为审计可复跑脚本（新增、未跟踪），`doc_inventory.csv` 等为生成数据，均属预期。

### 3.7 关键交叉校验（程序化，非人工抽看）

```text
inventory rows=253 / disposition rows=253 / unique=253 / dup=0 / inv-only=0 / dis-only=0
action: keep=125 retain-readonly=109 banner=16 move=3 (sum=253)
batch: B0=234 B3=19 ; priority: P0=2 P1=23 P2=228
blocker: 门禁钉死 33 / SSOT·索引 71 / 报告只读 73 / 快照只读 36 / 已归档 5 / 需裁决 19 / 无 16 (=253)
pinned docs(40) in DISPOSITION: 40/40 ; action=move: 0 ; docs/reports retain-readonly: 36/36
```

---

## 4. 问题清单（严重度 + 定位 + 最小修法）

> 严重度：**P1** = 阻断"可照做/可交付"；**P2** = 一致性/口径，不阻断发布但应在本轮修。**P0 未发现**（无门禁红线违反、无钉死路径误移、无证据删除/改写）。

| # | 严重度 | 定位 | 问题 | 最小修法（不代修，建议 Lead 执行） |
|---|---|---|---|---|
| P1-1 | **P1** | 方案 §9.2:594、§9.5:629；`BATCH_PLAN.md` §3:117-118、§6:176；`MIGRATION_LOG.md`:26-27 | `git checkout -- .` **无法回滚已 staged 的 `git mv`**（实测仍显示 `R old -> new`，原文件不恢复）→ B1/B1′ 回滚步骤失效 | 改为 `git reset --hard HEAD`（本批未提交场景），或 `git restore --staged --worktree .` + `git clean -fd docs/superseded`；并在 6 处同步 |
| P1-2 | **P1** | 方案 §9.2:571-575 | `python - <<'PY'` 为 bash heredoc，**PowerShell 下实测报错**（`Missing file specification after redirection operator`） | 改为 `python -c "import json;cfg=json.load(open('scripts/agent_ops/docs_consistency_config.json',encoding='utf-8'));print(...)"`，或提供 `.slim` 下独立 `check_pins.py` 并写 `python <script>` |
| P1-3 | **P1** | `doc_inventory.csv`（采集 22:32）/ 方案 §3.1 / 审计 README §2 / 台账 :6 | 253 条清单**未覆盖审计目录最终新增的 6-7 份 md**（ANOMALIES/FAMILIES/RECOMMENDATIONS/BATCH_PLAN/inventory_stats/MAPPING_SUMMARY/本报告）→ DoD#1"100% 覆盖"对最终交付集不成立 | 交付前重跑 `collect_inventory.py` + `build_disposition.py` + `validate_mapping.py`（补行、重算 253→260±），或在方案 §1.1/审计 README 明确"253 为 22:28 口径快照，审计自身新增证据不纳入本轮处置"并同步 DoD#1 表述 |
| P2-1 | P2 | `DISPOSITION.canonical_status`（22 行）/ 方案 §5.1、§8.2 | 出现词表外取值 `运营记录(不要求状态头)`；方案自身行即为其一，与其头部 `Active-Proposed` 矛盾；G5"状态合法"校验将误杀 | §5.1 增"L4 运营记录（不要求状态头）为非状态标记，不参与状态校验"一句；或把非 agent_runs 的 10 行改回 `Active-Proposed`/`Evidence-ReadOnly` |
| P2-2 | P2 | `DISPOSITION.blocker`/`gate_locked`（8 条 / 6 条无标记） | 钉死路径的"不可移动来源"未统一标记（报告只读/SSOT 掩盖门禁钉死；6 条 g_v_map/render 来源 gate_locked 为空） | 对 6 条补 `gate_locked=g_v_map`（或在 reason 注明来源）；统一 blocker 派生规则"钉死优先"（MAPPING_SUMMARY §3.2 已有同类建议） |
| P2-3 | P2 | 方案 §2.3:113；`GOVERNANCE_CONSTRAINTS.md` §3.0:175、附录 B:531；`docs/README.md`:7；台账:7 | "52 = 31 docs + **9** 根 md + 13 非文档"实为 **53**；正解 31+**8**+13=52（R3 软登记不计） | 五处统一改为"31 docs + 8 硬根 md + 13 非文档 = 52（另 1 软登记）"；与 MAPPING_SUMMARY §1 一致 |
| P2-4 | P2 | 方案 §5.3 / §8.2；DISPOSITION 16 行 | §5.3 未覆盖"过程性/长尾状态"（`设计完成`/`已实施`/`规划完成`/`Active`/`Closed`/`修订设计 v1`/`设计提案`/`建议状态`…），16 行有状态头却被归 `未登记`；§8.2 把 50 条都说成"无状态头" | §5.3 末尾加一行兜底规则（过程性描述→按 L 层+事实判定，正文移入正文）；§8.2 脚注改为"34 无状态头 + 16 长尾未映射" |
| P2-5 | P2 | `BATCH_PLAN.md` §0:16；方案 §8.1 | B0=234 却写 `keep 125 + retain 109 + banner 16`（=250）；`B1′ 3` 与 `B3 19` 未注明子集关系（234+3+19=256） | 改 B0 内 `keep 109`；在 §8.1/BATCH_PLAN 注明"B3 19 含 B1′ 3（条件迁移，DISPOSITION.batch 记为 B3）"或将 3 行 batch 改 `B1′` 并重算 |
| P2-6 | P2 | 方案 §8.6:539；§14.1 | "19 项（§14）"错位：§14 仅 11 项，19 项在 BATCH_PLAN §5；ANOMALIES P1 未建映射表 | §8.6 改"19 项（BATCH_PLAN §5；§14 列其中 11 项）"；§14 增"ANOMALIES P0/P1 → 裁决项"映射或在 §14 前注明 |
| P2-7 | P2 | 方案 §6.3:390、附录 B:838 | 写 `docs/doc_audit/<YYYYMMDD>/`，实际交付 `docs/infra/doc_audit/20261121/`（§6.4/§9.3 又按 infra） | 统一为 `docs/infra/doc_audit/<YYYYMMDD>/`（或明确迁移计划与迁移时点） |
| P2-8 | P2 | DISPOSITION 8 行 model_card；方案 §6.1/§1.1 | `artifacts/**/model_card.md` 标 `未登记`，与不变式 1/§4.2-L3"模型卡=证据"冲突；产品类文档（examples/scripts/data/outputs/root README）是否强制状态头未写明 | 8 行改 `Evidence-ReadOnly`；§6.1 补一句"产品/外部说明类不强制状态头，但须被 README 索引" |
| P2-9 | P2 | 方案 §8.2:519、§9.1 B0-6/B0-8；`BATCH_PLAN.md` B0-6 | "未登记 50 = B0-6/B0-8 工作清单"仅覆盖 17 份；B0-6 的"contracts/demo/design 12 份"与 CSV 分布不符（contracts 0） | §8.2 增"未登记 50 的逐桶动作表"（哪些补状态块、哪些豁免、哪些仅索引）；B0-6 对象清单对齐 DISPOSITION 未登记行 |
| P2-10 | P2 | 方案 §10.3 G5 规格 | G5 上线即与现状冲突：`docs/infra/doc_audit/` 证据目录未入默认豁免集；"目录相容"校验会命中 17 行存量越界 | 豁免默认集加 `docs/infra/doc_audit/`；目录相容规则给"存量白名单/基线例外"，并注明 G5 首版只对新增/变更文档强制 |
| P2-11 | P2 | 方案 §13.1#5、§9.5:621 vs §2.5/§6.4(D-22) | 门禁口径 4 道 vs 5 道不一致；第 5 道"全仓 pytest"未在本次审计实测存档（`baseline_gate_outputs.txt` 仅 4 门） | 统一为"4 门必跑 + 第 5 道（两档 pytest）仅在改动代码/测试时加跑"；或补跑并存档，修改 DoD#5 表述 |

**另记（非缺陷，供读者）**：`docs/README.md:5` 写"249 条文档逐条 keep/banner/move/retain-readonly"，实际处置行数为 **253**（249 是"已跟踪"数，混用了口径）→ 建议改 253（并入 P2-3 同批修）。审计 README §2"249 已跟踪 + 4 条本轮新增"本身正确，但"4 条"是采集时点口径，与 P1-3 联动。

---

## 5. 发布前"必修项"最小清单（按优先级）

### 必须（P1，2 条）

1. **修正回滚命令**（P1-1）：方案 §9.2/§9.5、`BATCH_PLAN.md` §3/§6、`MIGRATION_LOG.md` 共 6 处；用 `git reset --hard HEAD` 或 `git restore --staged --worktree .`（已实测）并复验一次 `git mv` 往返。
2. **修正 PowerShell heredoc**（P1-2）：方案 §9.2 的 `python - <<'PY'` 改 `python -c` 或独立脚本。
3. **补齐/声明清单覆盖口径**（P1-3）：交付前重跑 inventory+disposition+validate（或显式声明 253 为审计时点快照、审计自身新增证据不纳入本轮处置，并同步 DoD#1 与 README/台账"4 条新增"表述）。

### 建议同批（P2，11 条，全部可在文档层 1 小时内修完）

4. P2-3（52=31+9+13 算术）→ 五处文档统一为 31+8+13=52；
5. P2-5（B0 234 分解、B1′/B3 子集）→ BATCH_PLAN §0 + 方案 §8.1；
6. P2-6（19 项 vs §14 11 项 + ANOMALIES P1 映射）→ §8.6/§14；
7. P2-1（`运营记录(不要求状态头)` 词表外取值）→ §5.1/§8.2 + DISPOSITION 10 行；
8. P2-2（8/6 条钉死标记）→ DISPOSITION blocker/gate_locked；
9. P2-4（§5.3 长尾未映射 16 行、§8.2 表述）→ §5.3/§8.2；
10. P2-7（`docs/doc_audit` vs `docs/infra/doc_audit`）→ §6.3/附录 B；
11. P2-8/P2-9（model_card 8 行 + 未登记 50 逐桶动作）→ DISPOSITION + §8.2/B0；
12. P2-10/P2-11（G5 豁免与目录相容、4 vs 5 道门禁）→ §10.3/§13.1；
13. `docs/README.md:5` 的"249"改 253（随 P2-3 同批）。

> 上述 13 项修完即可由"有条件通过"转"通过"；**均不涉及门禁配置/脚本/CI/报告正文/钉死路径**，不触发 D-19/D-07/D-01~D-05。当前交付物的**硬安全结论**（不移动任何钉死文件、不删证据、报告与快照只读、4 门最终全绿）不因上述问题改变。

---

## 6. 复核方法与局限

- **方法**：①逐文件实读（方案 858 行、GOVERNANCE 535 行、BATCH_PLAN 184 行、审计 README/MAPPING_SUMMARY 等）；②程序化交叉校验（集合/重复/词表/计数/钉死集重建/目录相容）；③临时 git 仓库实测 `git mv` 回滚语义；④PowerShell 实测 heredoc 语法；⑤4 门 + validate_mapping 只读复跑；⑥与 Lead 的审计后变更说明逐条核对文件 mtime/内容。
- **局限**：①共享工作区并发变更，复核期间 GOVERNANCE:251 被修复、审计目录新增文件；本报告按"最终状态"判定，并对首测 FAIL 保留证据；②`ref_count` 对同名 README/model_card 失真（审计已声明），本报告未以该字段作判定依据；③第 5 道全仓 pytest（4,954 体量）未重跑，沿用 2026-11-20 记录（计入 P2-11）；④未执行任何 `git mv`/写操作于仓库文件，验证性 git 操作全部在临时目录完成并已清理。
- **独立性声明**：本复核不默认接受 Lead/审计结论，MAPPING_SUMMARY 的 PASS 结论已被本报告独立重算复核（253↔253、52 钉死集、LINK_FIX 13 行行号/文本核对）；表中所有 FAIL/部分 FAIL 均有可复跑命令或文件:行号证据。

---

## 7. 复核后修复记录（Lead 收口，2026-11-21 22:47–23:05）

> 本节由 Lead 追加，用于闭合 §4/§5 的必修项。**原 §0–§6 结论与证据保留不改**（复核时点快照）；修复后状态以本节与最新门禁实测为准。

### 7.1 必修项闭合（P1 3 条 + P2 11 条 = 14 条，全部已修）

| 项 | 修复内容 | 落点 |
|---|---|---|
| **P1-1** 回滚命令 | `git checkout -- .` → **`git reset --hard HEAD`**（+`git clean -fd docs/superseded`），并注明"checkout 无法撤销 staged 的 git mv（临时仓库实测）"；共 **6 处** | 方案 §9.2/§9.5、`BATCH_PLAN.md` §3/§6、`MIGRATION_LOG.md` |
| **P1-2** PowerShell heredoc | `python - <<'PY'` → **`python -c "…"`**（可直接执行） | 方案 §9.2 前置检查 |
| **P1-3** 清单覆盖口径 | 采用"**显式口径声明**"方案：253 = 审计启动时点快照；审计目录自身新增 7 份证据 md 按证据目录规则管理、不纳入处置映射（避免自指）；DoD#1 与 §3.1/审计 README/台账同步该口径 | 方案 §3.1、§13.1#1、审计 README §2、台账 |
| P2-1 词表外状态 | `运营记录(不要求状态头)` → **`运营记录（非状态标记）`**，§5.1 增注"非状态标记、不参与 G5 校验"；方案自身行恢复 `Active-Proposed`；8 份 model_card 归 `Evidence-ReadOnly` | 方案 §5.1、§8.2；`build_disposition.py`；`DISPOSITION.csv` |
| P2-2 钉死来源 | 新增 `pinned_by` 列（`docs_consistency` 34 + `g_v_map/render` 6 = 40 条 doc/根 md 全落位），不再依赖 blocker 单列表达来源 | `DISPOSITION.csv` 列 + 方案 §8.1 |
| P2-3 52 算术 | 五处统一为 **31 docs + 8 根 md + 13 非文档 = 52（另 1 软登记）** | 方案 §2.3、`GOVERNANCE_CONSTRAINTS.md` 附录 B、`docs/README.md`、台账 |
| P2-4 长尾映射 | §5.3 增"过程性/长尾描述"兜底行；§8.2 改"42 = 14 长尾 + 28 无状态头"并给逐桶动作 | 方案 §5.3、§8.2 |
| P2-5 批次分解 | B0 234 = `keep` **109** + `retain` 109 + `banner` 16；**B3 19 含 B1′ 3**（子集，不重复计） | 方案 §8.1、`BATCH_PLAN.md` §0 |
| P2-6 19 vs 11 项 | §8.6 改为"19 项（BATCH_PLAN §5 Q1–Q19；§14.1 列 12 个议题）"；§14.1 编号与 BATCH_PLAN 对齐并加 **ANOMALIES P0/P1 → 裁决项对照** | 方案 §8.6、§14.1 |
| P2-7 目录路径 | `docs/doc_audit/` → **`docs/infra/doc_audit/`**（3 处） | 方案 §6.3、§6.4 树、附录 B |
| P2-8 模型卡/产品文档 | 8 份 `model_card.md` → `Evidence-ReadOnly`；§6.1 补"强制范围"（产品/外部说明不强制状态头但须被索引） | `build_disposition.py`、方案 §6.1、`DISPOSITION.csv` |
| P2-9 未登记逐桶 | §8.2 增"42 的构成与逐桶动作"（15 行豁免 / 13 行 B0-6 / 14 行长尾兜底）；B0-6/B0-8/B0-10 数字与 CSV 对齐（状态块 13 份、索引缺口 27 条） | 方案 §8.2、`BATCH_PLAN.md` |
| P2-10 G5 缺口 | 豁免默认集加 **`docs/infra/doc_audit/`**；目录相容给"存量基线白名单 + 首版仅对新增/变更强制" | 方案 §10.3 |
| P2-11 4 vs 5 门禁 | 统一为"**4 道文档门禁必跑；第 5 道全仓 pytest 仅在改动代码/测试时加跑**" | 方案 §9.5、§11.2、§13.1#5、`BATCH_PLAN.md` §6 |

### 7.2 修复后实测（最终状态证据）

```text
docs_consistency --strict : scanned 169 file(s); 0 violation(s); CLEAN
check_docs_links           : checked 193 docs; 0 error(s), 172 warning(s)
render_gv_map --check      : G–V mapping doc is current
pytest registry/env        : 26 passed
validate_mapping.py        : PASS (FAIL=0, WARN=21)
build_disposition.py       : rows=253 unique=253 dup=0 missing_action=0 → PASS
```

`git status --short`：仅 `M docs/README.md`、`M docs/plans/PES2TS_文档修订台账.md`、`?? docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md`、`?? docs/infra/doc_audit/`——无 `scripts/**`、`config/**`、`docs/reports/**`、代码/测试/CI 改动（D-07/D-19 保持）。

### 7.3 结论更新

**由"有条件通过"更新为"通过（PASS）"**：P1 3 条全部闭合；P2 11 条全部在文档/生成层闭合；无新增 warning；硬安全结论不变（0 条钉死路径建议移动、0 条证据删除/改写、报告与快照只读）。**遗留非阻断**：P2-2 的 `pinned_by` 以新增列表达（已闭环）；§14.2 的 5 个开放问题仍按方案保留待后续迭代。
