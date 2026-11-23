# ANOMALIES — PES2TS 文档异常与治理负债清单

> 任务：task-3（内容-关系层面诊断）；生成日期：2026-11-21；只读仓库，未修改/移动任何既有文件。
> 证据口径：文档内状态头/横幅（头部行）、`docs/README.md` 状态列、`docs/plans/PES2TS_文档修订台账.md` 登记行、
> `infra/governance/docs_consistency_config.json` 门禁名单、git 文件名与体积统计。
> 数据底稿：本目录 `anomaly_candidates.csv`（机械候选 60 条）、`anomaly_stats.json`；复跑 `python anomaly_scan.py`。
> 精编后 **32 条**（P0 2 / P1 10 / P2 20）；机械候选中的逐文件展开（根 memo 8 份、HTML 8 份、命名混排 8 份）
> 已合并为单条，避免"一条异常刷屏"。C-xx 编号引用 `rules/GOVERNANCE_CONSTRAINTS.md` 已核事实，不重复验证。
> 严重度：**P0**=会误导执行/训练依据；**P1**=治理门禁或索引失真；**P2**=维护成本/一致性债务。

## 一、P0（2 条）

| 编号 | 类别 | 文件 | 证据（文件:行号 / 统计） | 建议动作 |
|---|---|---|---|---|
| A01 | 状态异常 | `docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md` | L3 `状态：**Active-Proposed**`（WP1–WP7 待实现）；但 `docs/README.md` L30 记 **"V5.4.2 WP7 EXECUTED（2026-10-31）"**，报告 `docs/reports/PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md` 已交付 | 状态头改 `Executed-Frozen`，或照 V5.4.1 做法加"§执行指针"（不改 §0–§18 规划历史）；同 PR 更新 docs/README 状态列 |
| A02 | 状态异常 | `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | L3 `状态：**Proposed v1.1**`；同一头部 L4 已写 **"执行状态（2026-10-23）：S0/S1/S2/S3 训练全部完成"**；S4 报告（2026-10-27）已出 | 状态行改 `Executed-Frozen`（或标"计划文本冻结，执行状态见 L4"）；禁止新训练再引用 Proposed 版 |

## 二、P1（10 条）

| 编号 | 类别 | 文件 | 证据（文件:行号 / 统计） | 建议动作 |
|---|---|---|---|---|
| A03 | 状态异常 | `docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md` | 同一头部自相矛盾：L3 `EXECUTED-FROZEN / Stop 语义已取代（2026-10-21）`；L5 `状态：**待冻结的执行计划**` | 删除/修正 L5 旧状态行；该文件在 `historical` 门禁名单（`docs_consistency_config.json`），只改状态不改位置 |
| A04 | 状态异常 | `docs/design/decisions/ADR-0034-V5.4.3-双Decoder与DecoderContextV1.md` | L3 `状态：**Proposed**`；但 18-run 已于 2026-11-06 执行完毕（台账 2026-11-06 条："V5.4.3 全阶段执行完毕（Executed-Frozen；双轨冠军确证）"） | 台账已明示"不改冻结原文"，故**保留 ADR 原文**；必须在 `docs/README.md` 补 2026-11-06 条目，注明 ADR 状态为历史时点（交叉 A20） |
| A05 | 状态异常 | `docs/plans/PES2TS_PES-N未来开发计划_v2_20261005.md` | L3 `状态：**Superseded / Historical**`；仍留在 `docs/plans/` 活跃目录，且被 **26 处**引用（`doc_inventory.csv` inbound_refs=26） | 保留原地 + 已有横幅（②类）；不要迁 superseded（会断 26 链） |
| A06 | 位置异常 | `docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md`、`PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md`、`V541_WP7_frozen_dev_eval_设计方案_v1.0_20261028.md` | 3 份文件文件名含"产品设计/设计方案/架构"（设计语义）却位于 `docs/plans/`；V541 WP7 那份**既无状态头又位置异常** | 需所有者裁决：留 plans（若属方案附件）或迁 `docs/design/`（须同步 docs/README 链接与门禁）；V541 WP7 至少补状态头 |
| A07 | 位置异常 | `docs/reports/PES2TS_多坐标PES逐步实施方案_20261001.md`、`docs/reports/PES_generation_多坐标约束学习_ORCA能力与数学方案_20261001.md`、`docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md` | `docs/reports/` 应收只读证据，但混入"方案/决策记录"类文本（3 份） | `docs/reports/**` 内容永不改（AGENT.md §9.4）；仅在 docs/README/reports 索引注明"历史方案误置、只读保留"，新文档不得再入 reports |
| A08 | 引用异常 | 索引/台账覆盖缺口 | `rules/GOVERNANCE_CONSTRAINTS.md` **C-08**：40 份文档既不在 `docs/README.md` 也不在台账（README-only 40、台账-only 7）；**C-03**：索引自述版本滞后 5 处（门禁 CLEAN 也检测不到） | 治理负债，交 task-2/owner：以 C-08 清单为输入补双索引；索引版本号纳入 CI 校验 |
| A09 | 引用异常 | 门禁名单 stale 风险 | `infra/governance/docs_consistency_config.json`：`historical` 17 条 / `required_files` 27 条；其中 `docs/reports/PES2TS_V52_任务纠偏审计_20261021.md` 等被 required_files 钉死但属 reports（只读）；另 **C-10** 指出 `g_v_map` 路径校验分级存在"看起来钉死、实际不校验"盲区 | 门禁名单只增不改；新归档动作前先跑 `check_docs_links.py` + `docs_consistency --strict`（基线：0 error / ~172–173 warning，见 C-11） |
| A10 | 体积/形式 | `docs/plans/PES2TS_文档修订台账.md` | **166,842 B / 885 行**，append-only 顶部倒序增量登记；单文件已承载 2026-09 至 2026-11 全部修订 | 不删证据；建议拆"台账索引 + 年度附录"（拆分会动 AGENT.md §9.3/GOVERNANCE 引用，需 owner 批准 + 同步门禁） |
| A11 | 体积/形式 | `README.md`（根） | **104,309 B / 1,313 行**，与 `docs/README.md`（79,471 B / 349 行）双维护文档/阶段索引 | 需 owner：根 README 收敛为门面+入口，索引单一职责归 docs/README；root_scope 门禁输入，移动需同 PR 改配置 |
| A12 | 内容重复 | `docs/plans/` 同主题三件套 | V4.2–V4.7 / V5.x 每代都留"方案+合同/清单+报告"，正文大量重复表格（30h 计划与 R3 方案重叠字段；V5.4.3 方案 93 KB/1671 行） | 保留证据；在 `docs/superseded/README.md` 或台账登记"同主题三件套"关系，防止后来者重复引用旧版 |

## 三、P2（20 条）

| 编号 | 类别 | 文件/对象 | 证据（文件:行号 / 统计） | 建议动作 |
|---|---|---|---|---|
| A13 | 状态异常 | `docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md` | L3 写 **"建议状态：Active-Proposed"**（"建议"非正式状态声明）；全库 **0 引用**、台账/README 均未登记 | 需 owner 裁决：正式登记状态，或标"未登记草案"并补索引 |
| A14 | 状态异常 | `docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md` | 无 `状态：` 声明行；inbound_refs=**1**（近乎孤儿） | 补状态头；若已失效则加"历史"横幅（参照 C-06：42/161 live 文档无状态块） |
| A15 | 状态异常 | G2 时代批次方案 3 份：`PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md`、`PES2TS_G2_下一批算法调整与测试方案_20261003.md`、`PES2TS_ACP执行统一与旧代码清理方案_20261003.md` | 无 `状态：` 声明行；README 仅给内容描述（"执行方案/批次方案"）；G2 阶段已被 PES-N/V5 系列跨代 | 补 `Historical` 状态头；不移动（引用 12–21 处，改链成本高） |
| A16 | 状态异常 | 状态词表未收敛 | `rules/GOVERNANCE_CONSTRAINTS.md` **C-05**：17 个归并桶 vs 官方 5 值、台账 4 值；**C-07**：状态头书写 5 种变体（`状态：`/`> 状态：`/`**状态**`/`建议状态：`/缺失）；**C-02**：状态头版本与文件名/正文三方不一致 2 例 | 治理规范层统一状态词表 + 状态头模板（task-2/owner）；本任务扫描已兼容 5 变体（`docscan_lib.strict_status`） |
| A17 | 命名异常 | `docs/plans/PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md` | 文件名 v1.0；正文内容已 v1.1.1→v1.2（`docs/README.md` L50/L65；文件头部有"当前版本"行）；对应 C-01 4 例之一 | AGENT.md §9 允许"文件名保留 v1.0"，但应在文件头首行标注内容版本（现需翻 L8） |
| A18 | 命名异常 | `docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md` | 文件名 v3.0；正文 L3 "v3.3 MOVE–PROCESS 修订"、L9 "Proposed v3.2"（README 也称"内容 v3.2"） | 统一"文件名版本 vs 内容版本"声明格式；C-01/C-02 同源 |
| A19 | 命名异常 | `docs/plans/PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md`、`docs/plans/PES2TS_训练测试集划分标准_v1_20261120.md` | 前者 L8 "当前版本：v1.2 @ 2026-10-30"；后者文件名 v1、台账 2026-11-20 记"升 v1.1" | 同上；短期在头部补"内容版本"行 |
| A20 | 命名异常 | docs 全局命名口径 | 带 `_YYYYMMDD` 日期 **117** 份 / 不带日期 **82** 份；版本号 `_v1_` 与 `_v1.0_` 并存（如 `_v1_20261020` vs `_v1.0_20261102`）；无 `PES2TS_` 前缀另有 ADR-*（34）、G1_*（10）、PES_GENERATION_*、V541_* | 制定命名约定（`<项目前缀>_<主题>_v<主.次>_<YYYYMMDD>.md`，ADR 例外）；重命名须同步门禁/引用，仅对新文档强制 |
| A21 | 命名异常 | `docs/contracts/` 4 份 + ADR-0034 | 文件名含英文单词连写（`PES2TS_FullPathNavigationContract_v1.md`、`GeometryPolicyContract_v1/v2`、`MoveProcessContract_v1`、`ReactionIntent_Contract_v1/v2`、`ADR-0034-…-DecoderContextV1.md`）与中文命名并存 | 不重命名（高频引用 + 门禁）；仅在命名规范中登记"合同类英文驼峰"为例外形式 |
| A22 | 内容重复 | `docs/archive/**` | 79 份快照文件中 **28 份**与 live 文档同名；`docs/README.md` L158 注：早期报告含 `E:/...` 绝对路径（仅证据文本） | 快照只读不改；在 archive/README 注明"28 份同名快照体积占比"，链接检查继续排除 archive |
| A23 | 内容重复 | 根 README vs docs/README | 两份都在维护文档索引（见 A11）；另 `docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html`（379,567 B）与 S3 报告正文部分重叠 | 见 A11；HTML 作为只读证据保留，不纳入文本 lint |
| A24 | 引用异常 | 0 引用治理产物 | `docs/infra/PES2TS_开发文档规范化归档方案_v2_20261121.md`（0 引用，bucket=治理规范）；`docs/plans/PES2TS_V5.4_剩余架构研究工作方案_v1.0_20261029.md`（0 引用）；`docs/agent_runs/**` 由 token_exempt_prefix 豁免引用检查 | 新增治理文档必须同步 docs/README/台账（AGENT.md §9.3），否则 0 引用；agent_runs 豁免属既定规则 |
| A25 | 引用异常 | 链接检查容忍债务 | `rules/GOVERNANCE_CONSTRAINTS.md` **C-11**：172 warning 是"容忍债务"非 0 债务（台账 2026-11-20 记 173 warning/182 docs；两处口径差 1 属不同时点） | 保持 0 error；warning 清理只针对 live 文档，`docs/reports/**` 与 archive 不动 |
| A26 | 体积/形式 | HTML 文档 8 份 | 仓库 8 份 HTML（docs 内 3：`docs/design/G1_v2_成断键扫描方向选择可视化.html` 30,773 B、`docs/plans/PES2TS_实施路线图.html` 26,753 B、`docs/reports/PES2TS_V53MP_训练过程可视化报告_20261023.html` 379,567 B；其余 5 份在 data/examples/outputs） | docs 内 3 份登记为"图形入口"（浏览器打开），不纳入 md 链接检查；报告内 HTML 只读 |
| A27 | 体积/形式 | `docs/plans/fixtures/pes2ts_direct_training_input_audit_20261015.json` | **240,868 B / 9,229 行** 的证据夹具，与摘要文档重复承载审计数据 | 保留（复核证据）；新审计产物外置到 fixtures 并在正文只放摘要+哈希 |
| A28 | 体积/形式 | `docs/design/decisions/ADR-0034-…md`（60,763 B / 672 行）、`docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0_20261102.md`（93,311 B / 1,671 行） | 为历史最大 ADR/方案；含完整训练/评估/预注册合同，长表格直接内嵌 | 保留（冻结证据）；新 ADR 建议拆"决策正文 + 附录/模板引用"，控制单文件 ≤40 KB |
| A29 | 体积/形式 | Office 证据 | 全仓文本面扫描 **0 份** docx/xlsx/pptx（跳过 data/outputs 后 `os.walk` 结果） | 现状良好；若将来有 Office 证据，应落 `outputs/` 并只在 docs 留指针，不入库大二进制 |
| A30 | 位置异常 | 根目录战役 memo 8 份 | `24H_EXECUTION_TIMELINE.md`/`24H_FINAL_REPORT.md`/`FAILURE_AND_REMAINING_WORK.md`/`NEXT_DAY_DECISION_MEMO.md`/`R3_RESULT_SUMMARY.md`/`V52_PROGRESS_STOP_REPORT.md`/`V53_ROLLOUT_REPORT.md`/`V53MP_RESUME_20261022.md` 全在仓库根 | **不移动**（historical/root_scope 门禁输入，移动即红）；按下文 RECOMMENDATIONS 仅在 `docs/agent_runs/README.md` 加索引 |
| A31 | 位置异常 | `docs/infra/` 仅 2 份（扫描时点） | 基础设施/运维类知识无正式归属区，易散落根目录与 plans/；本次任务期间 `docs/infra/doc_audit/20261121/` 已新增审计产物 | 需 owner：明确 docs/infra 收录规则（或并入 docs/design），否则继续散落 |
| A32 | 位置异常 | `docs/design/` 混入计划类 4 份 | `G1_v2_补全实施总方案.md`、`G1_v2_ORCA扫描模式识别与计划.md`、`G1_v2_L0到G2扫描策略设计.md`、`PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md` 位于 design/ 但语义偏"计划/总方案" | 保留（设计内容为主）；如需执行清单另建 plans 条目，不搬动 |

## 四、汇总与处置优先级

| 指标 | 数值 | 说明 |
|---|---|---|
| 精编异常 | **32** 条 | P0 2 / P1 10 / P2 20 |
| 机械候选（脚本） | 60 条 | `anomaly_scan.py` 输出；逐文件展开已在上表合并 |
| 立即可修（不影响门禁） | A01、A02、A03(L5)、A14、A17–A19、A13/A15（补状态头） | 均为"改状态行/加横幅"，不动文件位置与门禁名单 |
| 需 owner 裁决 | A06、A10、A11、A13、A16、A31 与所有"迁移/拆分"动作 | 涉及门禁路径、索引职责或治理规范 |
| 明确不动 | A07、A09、A22、A25、A30（只读证据/门禁钉死/快照） | 与 archive/superseded README 及 AGENT.md §9.4 一致 |

> 备注：`rules/GOVERNANCE_CONSTRAINTS.md` 的 C-01…C-12 是 task-2 的独立核验结论，本清单在其已核事实上只做"文档-关系"层面的归类引用；
> 其中 C-04（AGENT.md 未随宪法 v2.9 更新）、C-09（门禁实现滞后 §12.4）、C-12（归档后双登记负担）属治理机制项，不在本清单重复立条。

---

## 附注（2026-11-21 晚，审计收口时点）

- **A24**（本方案 0 引用）与 **A31**（docs/infra 仅 2 份）为扫描时点状态；审计收口时本方案与审计目录已在 `docs/README.md` 与修订台账登记，`docs/infra/` 变为 6+ 份，上述两条已消解，保留原文以存证。
- 本文件为**精编 32 条**（P0 2 / P1 10 / P2 20）；机械候选 **60 条**逐条数据见 [anomalies.csv](anomalies.csv)，复跑脚本 [anomaly_scan.py](anomaly_scan.py)。
- 严重度与处置以 [开发文档规范化归档方案 v2.0](../../PES2TS_开发文档规范化归档方案_v2_20261121.md) §7/§9 为准；本文件不构成移动授权。
