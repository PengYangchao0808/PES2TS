# RECOMMENDATIONS — PES2TS 文档分类结论与归档候选

> 任务：task-3；生成日期：2026-11-21；只读仓库，未修改/移动任何既有文件。
> 数据底稿：`classification_mapping.csv`（覆盖 task-1 清单 **253 条**：244 条已分类 + 9 条清单外/新增标 `未分类-需裁决`）、
> `classification_stats.json`、`doc_inventory.csv`（扫描域 306 条，含 docs 280 文件，其中 .md 216）；复跑 `python make_classification.py`。
> 判据一律可回溯：`class_bucket` 依据桶定义（见 §1）+ 文件路径/状态头/README 状态列/门禁名单，逐条证据列在 CSV 的 `evidence_note`。

## 一、建议采用的分类维度（4 轴 + 1 标志）

单靠目录分类不够（`docs/plans` 里既有 Active 又有 Superseded，`docs/reports` 里混入方案），建议四轴定位：

| 轴 | 取值 | 用途 |
|---|---|---|
| **A. 生命周期状态** | 治理规范 / 活跃计划 / 冻结计划（已执行）/ 已取代计划 / 历史提案 / 快照 | 决定"能否作为当前执行依据"（对应 AGENT.md §9 状态头） |
| **B. 职责类型** | 设计 / ADR / 合同 / 证据报告 / 运营记录 / 产品文档 / 外部说明 / 其它 | 决定"一文档一职责"归属与目录期望 |
| **C. 权威级别** | SSOT（宪法/阶段定义/台账/索引）/ 规范性（合同/ADR）/ 说明性（方案/设计）/ 操作性（infra/清理）/ 证据（报告/快照） | 决定冲突时以谁为准 |
| **D. 可变性** | 只读（reports/archive/model_card）/ 仅横幅与链接可改（superseded）/ 可修订（plans/design） | 决定处置动作合法边界（AGENT.md §9.4、archive/superseded README） |
| **＋ 门禁标志** | `gate_locked ∈ {historical, required_files, root_scope}` | 该路径被 `docs_consistency_config.json` 钉死：**任何移动/改名都会断门禁** |

CSV 输出列已按此设计：`path, class_bucket, suggested_action, gate_locked, inbound_refs, self_declared_superseded, head_status, readme_status, file_version, campaign_versions, date_in_name, bytes, lines, evidence_note`。

## 二、分类桶计数（253 条清单口径）

| 桶 | 数量 | 主要组成 |
|---|---:|---|
| 治理规范 | 8 | 宪法、台账、阶段定义、docs/README、AGENT、archive/superseded README、完整开发流总纲、infra 治理 |
| 活跃计划 | 23 | V5–V7 v3.0、R3 系列、V5.4.3（Frozen 但未执行完前按活跃）、V5.4.2（状态滞后）、G2 AB3/AB4、G3-P1 等 |
| 冻结计划 | 11 | V5.4.1、V5.4.3 方案/清单、30h 计划、V5.3-MP、V4.2-V4.7、V52R、G2T 等已执行/已关闭 |
| 已取代计划 | 12 | superseded/ 4 份 + PES-N v2/v1.2、V5-V7 v2.0、V5.1-R2、EGNN 三代、30h 等（部分兼门禁钉死） |
| 历史提案 | 10 | G0–G4 框架、G2 早期批次、拓扑约束 v2、统一价值模型 v1、双端 v4 等 |
| 设计 | 23 | docs/design 16 + docs/demo 3 + 误置/说明性 4 |
| ADR | 34 | ADR-0001…ADR-0034 |
| 合同 | 13 | contracts/ 13（Active 5 + 历史/部分取代 4 + 其它 4） |
| 证据报告 | 36 | reports 36（含 fixtures 证据与 HTML 可视化） |
| 运营记录 | 22 | agent_runs（含 handoffs 9）、根目录战役 memo 8 |
| 快照 | 36 | docs/archive 清单内文件（全量 archive 为 79 文件） |
| 产品文档 | 8 | artifacts/*/model_card.md |
| 外部说明 | 5 | 根 README/CONTRIBUTING/LICENSE、examples/README、scripts/README |
| 其它 | 3 | pytest.ini/requirements*.txt 等非文档配置 |
| 未分类-需裁决 | 9 | 清单外/本次新增：`docs/infra/doc_audit/20261121/*`（4）、LICENSE、data/README.md、outputs/README.md、scripts/archive/README.md、examples/contracts_v1/path_viewer.html |
| **合计** | **253** | |

处置动作计数：`原地保留(门禁钉死)` **23**、`原地保留(SSOT/索引)` **81**、`原地保留(只读证据)` **73**、`原地保留(快照)` **36**、
`原地保留(已归档)` **5**、`仅加历史横幅` **7**、`降级为历史提案(不移动)` **9**、`需所有者裁决` **19**。

## 三、三类处置结论（任务要求的硬口径）

### ① 可立即归位（0 引用 + 自身已声明被取代 + 不在门禁名单）— **0 条**
- 满足该三条件的 4 份文档已由 2026-11-20 归档执行迁入 `docs/superseded/`（单端点 v1/v2、R3 端点对称 v1/v2），复查其 inbound_refs=1（仅 superseded/README 索引计数），归位已完成。
- **F19 EGNN 三代**（`EGNN_BC_IQL_SAC下一步完整训练方案_v2`、`EGNN_BC_IQL_SAC实施与训练计划_v3`、`IRC_BC_IQL_SAC直接位移研究方案_v1`）自身已声明被取代、不在门禁名单，但引用数 3/7/2 → **不满足①**，只能走③。

### ② 需加横幅原地保留 — **7 条**
`ADR-0007`、`ADR-0008`、`ADR-0009`、`ADR-0020`、`ADR-0023`、`ADR-0024`（README 已标历史/被取代，建议头部补统一横幅）、
`docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md`（已声明 Superseded 但 26 引用，不迁）。
理由：ADR 不原地改写结论（AGENT.md §7.2）；PES-N v2 迁移动链成本高、非必要。

### ③ 需所有者裁决 — **19 条**
- 状态矛盾/滞后（执行完毕但状态 Active）：V5.4.2 方案（A01/P0）、V5.3-MP 计划（A02/P0）、30h 计划（A03/P1）、ADR-0034（A04，台账已明示不改文）。
- 迁移前需决策：F19 三份（需先改 2–7 处链接）、F03 三位一体 v1 是否并入 R3 家族、无正式状态头的 5 份文档（V5.4 剩余方案 / V5.1 修复冲刺 / V541 WP7 设计 / G2-ACP 补齐 / G2 下一批算法，见 A13–A15）。
- 拆分/改名级：台账 166KB 拆分（A10）、根 README 与 docs/README 双索引治理（A11）、docs/infra 收录规则（A31）。
- 清单外 9 条（含本次审计新增 `docs/infra/doc_audit/20261121/`）由 owner 确认归属。

## 四、建议归档/降级候选集合（带判据与风险）

| 集合 | 成员 | 判据 | 风险/前置 |
|---|---|---|---|
| **A. 迁入 `docs/superseded/`（需先改链）** | F19 三份（EGNN v2/v3、IRC 直接位移 v1） | 头部已声明被取代；不在门禁名单；引用 3/7/2 | 须先修引用处链接 + superseded/README 索引 + 台账登记；`check_docs_links` 保持 0 error |
| **B. 降级为历史提案（不移动，仅状态/横幅）** | 9 条：G0-G4 框架修改 v1、G1_G2 边界审计、双端拓扑 v4、图论立体先验 v2、扫描规划谱系合并、拓扑约束TS v2、统一价值模型 v1、统一开发顺序、逐步实施 | README 状态列=历史/参考；已被后续 G0–G4/V5 系列承接 | 只改状态不改正文；报告类不动 |
| **C. 仅加历史横幅** | 7 条（见③②） | 已被取代/历史但引用多或属 ADR | 不改结论不改位置 |
| **D. 明确不归档** | 见 §五 | 门禁钉死 / 只读证据 / SSOT / 快照 | — |

## 五、必须原地保留清单（不可动）

1. **门禁钉死 23 条**（`historical` 17 + `required_files` 27 + `root_scope` 9 的并集路径，见 `classification_mapping.csv` 的 `gate_locked` 列）：移动/改名即断 `docs_consistency`、`render_gv_map --check`、registry tests 四道 CI 门。
   含：宪法、AGENT.md、台账、阶段定义、docs/README、根 7+1 份 memo、5 份 plans（30h、PES-N v1.2、V5-V7 v2.0、V5.0-V5.2 冲刺、V5.1-R2）、V5.4.1/2/3 方案与清单、4 份合同/ADR+MoveProcess 等。
2. **只读证据**：`docs/reports/**` 36 份（AGENT.md §9.4 永不改内容，只能加顶部指针）；`artifacts/*/model_card.md` 8 份。
3. **SSOT/索引**：`docs/PES2TS_开发宪法_v1.md`、`docs/plans/PES2TS_文档修订台账.md`、`docs/plans/PES2TS_阶段定义_G0-G3.md`、`docs/README.md`、`AGENT.md`、`docs/plans/PES2TS_完整开发流方案_整合版_v1.md`。
4. **快照**：`docs/archive/**` 79 份（archive/README 规则 2 禁止当"过时文档墓地"）。
5. **当前有效合同/ADR**：MoveProcessContract v1、GeometryPolicyContract v2、ReactionIntent Contract v2、ADR-0030/0031/0032/0033 等 Active/Accepted 件。
6. **运营记录**：`docs/agent_runs/**`（handoffs 9 + state 门禁 JSON；`token_exempt_prefixes` 已豁免）。

## 六、执行顺序建议（风险从低到高）

1. **零风险（不碰位置/门禁）**：补 7 条历史横幅（②）、修 3 处状态滞后头（A01/A02/A03-L5）、补 docs/README 的 2026-11-06 条目（A04）。
2. **低风险**：把 F19 三份加入 `docs/superseded/README.md` 的"待迁清单"并在各引用处加指针（不改文件位置）。
3. **中风险（需 owner 批准）**：F19 批量迁移（三件套动作：改链 / superseded 索引 / 台账登记）；台账拆分与双索引治理（A10/A11）。
4. **禁止**：移动 `historical`/`required_files`/`root_scope` 路径；修改 `docs/reports/**`；删除任何证据；把过时文档搬入 `docs/archive/`。

## 七、与治理约束的对应（引用 task-2 已核结论）

- C-01/C-02/C-03（版本与索引滞后）→ 本文件 §三②③ 与 ANOMALIES A17–A20、A08。
- C-05/C-06/C-07（状态词表/状态头缺失/书写变体）→ 分类轴 A 需先收敛状态词表，否则 CSV 的 head_status 仅作证据不作判定。
- C-08（索引覆盖缺口 40 份）→ `classification_mapping.csv` 可作补索引输入。
- C-10/C-11（g_v_map 校验盲区 / 172–173 warning）→ 本文件所有迁移动作的前置校验。
- C-12（归档后双登记负担）→ 建议 superseded 迁移只登记"原路径→取代者→日期"三列，避免第 4 份索引。

> 边界：本文件是**建议**，不构成规范；所有涉及移动/拆分/改名的动作须按 AGENT.md §7.2/§9 走 owner 批准，并在同一 PR 同步台账、docs/README 与门禁配置（若必须动门禁路径）。
