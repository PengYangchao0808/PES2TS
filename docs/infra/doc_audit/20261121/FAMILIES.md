# FAMILIES — PES2TS 版本家族与取代链

> 任务：task-3；生成日期：2026-11-21；只读仓库。
> 机器底稿：`families.csv`（21 家族 × 53 成员）、`families_evidence_check.json`（58 条证据全部命中，0 失败）、
> `families_report.txt`；复跑 `python families_scan.py`。
> 角色：`current` 当前有效 / `superseded` 已被取代 / `partial` 部分条款被取代 / `historical` 已冻结为历史 / `contradictory` 状态自相矛盾 / `unknown` 状态缺失。
> 三类处置：**①可立即归位**（0 引用 + 自身已声明被取代 + 不在门禁名单）；**②需加横幅原地保留**；**③需所有者裁决**。

## 一、家族总表（21 个）

| ID | 家族 | 成员（角色） | 当前有效者 | 被取代者 | 取代/状态证据 | 迁移建议 | 裁决 |
|---|---|---|---|---|---|---|---|
| F01 | 单端点 IRC 生成控制器·产品设计 | v1(S)、v2(S)、v3(C) | v3 | v1、v2（已入 `docs/superseded/`） | `docs/superseded/README.md` 索引表登记原路径与取代者=v3；v3 头部 L4 `状态：**Proposed v3**（取代 v1/v2）`；`docs/README.md` L9 记录"迁入 4 份" | **①已完成归位**；v3 留 plans/（Active-Proposed） | 否 |
| F02 | V5.1-R3 端点对称特征与损失重训 | v1(S)、v2(S)、R3 v3(C) | R3 v3 | 端点对称 v1、v2（已入 superseded） | `docs/superseded/README.md` 取代者=`[R3 v3]`；`docs/superseded/README.md` 明确"未收录…三位一体 v1"另案 | **①已完成**；R3 v3 留 plans/ | 否 |
| F03 | V5.1-R3 三位一体修复（执行细化） | v1(C) | v1（执行细化） | 无（未声明被取代） | 头部 L5 `状态：**Proposed v1.1**`；`docs/README.md` 状态列 `Active-Proposed（执行细化）`；`docs/superseded/README.md` 明示"自身未声明被取代 → 留 plans/" | **②留原地** | 文件名 `_v1` 与 R3 v3 并列，是否需要并入 R3 家族 → ③确认 |
| F04 | V5–V7 三阶段研究方案（三代） | PES-N v1.2(S)、三阶段 v2.0(S)、反应意图 v3.0(C) | v3.0（内容 v3.2/v3.3） | v1.2、v2.0 | README L124 "Superseded（2026-10-20）"、L225 同；v2.0 `Superseded by v3.0`（README L103/L244）；v3.0 头部 L9 `Proposed v3.2（取代 v2.0）` | **②/①不可迁**：v1.2、v2.0 在 `historical` 门禁 17 条名单，移动即红 → 仅加/保留横幅 | 否（门禁钉死） |
| F05 | PES-N 未来开发计划 → V5–V7 三代 | PES-N v2(S)、PES-N v1.2(S)、反应意图 v3.0(C) | v3.0 | PES-N v2、v1.2 | PES-N v2 头部 L3 `状态：**Superseded / Historical**`；README L189 `历史（已被 V5–V7 v2.0 取代；内容 v2.5）`；台账 L295/L329 登记 V5-V7 架构调整取代关系 | **②PES-N v2 留原地+横幅**（26 引用；不在门禁名单，迁 superseded 需改 26 链） | 否（②类） |
| F06 | GeometryPolicyContract v1/v2 | v1(P)、v2(C) | v2 | v1 的 §4.2/§5.2 | v1 头部 L3 `**SUPERSEDED（2026-10-21）**`；README L283 `Superseded（§4.2/§5.2，由 v2 取代；ADR-0030）` | **②v1 原地+横幅**（`historical` 门禁钉死）；v2 当前 Active | 否 |
| F07 | ReactionIntent Contract v1/v2 | v1(P)、v2(C) | v2 | v1 的 §4/§6.2 | v1 头部 L3 `**SUPERSEDED（2026-10-21）**`；README L239 `Superseded（§4 Stop/q 条款由 v2 取代；ADR-0030）` | **②v1 原地+横幅**（`historical` 门禁钉死）；v2 当前 Active | 否 |
| F08 | FullPathNavigation v1 → MoveProcessContract v1 | v1(S)、MoveProcess v1(C) | MoveProcessContract v1 | FullPathNavigationContract v1 | 前者头部 L3 `**SUPERSEDED（2026-10-21，ADR-0031 / MoveProcessContract v1）**`；README L86 与 L315 状态列 | **②v1 门禁钉死+横幅**（`historical`+`required_files` 双重钉死）；MoveProcess v1 权威 | 否 |
| F09 | V4.2–V4.7 十小时战役（方案+合同+报告） | 方案(H)、合同(H)、实现报告(H)、过夜汇总(H) | 无（已执行、负结果冻结） | — | README L227/L228 `Executed-Frozen`；L231 实现报告 `Evidence 只读`；L130 实现报告结论"V4.7 否定/负结果" | **②全部原地保留**（报告只读 + 冻结证据） | 否 |
| F10 | V5.4.1 Embedding-50K（方案+ADR+报告） | 方案(C,Executed-Frozen)、ADR-0032(C)、报告(C) | 报告 20261029 + E0 冠军 bundle | 无 | 方案头部 L3 `状态：**Executed-Frozen**（2026-10-29…）`；README L50 "转 Executed-Frozen（新增 §19 执行指针）"；L51 `docs_consistency --strict CLEAN` | **②原地保留**（方案在 `required_files` 门禁）；文件名 v1.0 vs 内容 v1.2 → A17 命名异常 | 否 |
| F11 | V5.4.2 Backbone-50K（方案+ADR+报告） | 方案(contradictory)、ADR-0033(C)、报告(C) | 报告 20261031（A0 留任，no_backbone_winner） | 无 | 方案 L3 `状态：**Active-Proposed**` vs README L30 **"WP7 EXECUTED（2026-10-31）"**；报告含 `no_backbone_winner` | **③需 owner 裁决**：方案状态滞后（A01/P0）；报告只读，ADR 不动 | **是** |
| F12 | V5.4.3 双 Decoder-50K（方案+清单+ADR+报告） | 方案(C,Frozen-Preregistered)、WP清单(C)、ADR-0034(C,Proposed)、报告(C) | 报告 20261106（双轨冠军确证） | 无 | 方案 L5 `状态：**Frozen-Preregistered v1.0**`；ADR-0034 L3 `状态：**Proposed**`；台账 2026-11-06 "全阶段执行完毕（Executed-Frozen）"+"保持 Frozen-Preregistered v1.0"（明示不改文） | **②原地保留**（明示冻结原文不改）；docs/README 缺 2026-11-06 条目 → A04/A08 索引滞后 | 否（索引补条目） |
| F13 | V5.4 剩余架构工作方案 | v1(unknown) | 无（未正式登记） | 无 | 头部 L3 `**建议状态**：Active-Proposed`（非正式状态声明）；全库 0 引用、台账/README 未登记 | **③需 owner 裁决**（正式登记或标未登记草案；0 引用可低成本处置） | **是** |
| F14 | V5.3-MP 大规模 MOVE–PROCESS | 计划(contradictory)、同步清单(H)、S0/S1/S3/S4 报告(H) | 报告集（S0–S4 已完成） | 无 | 计划 L3 `状态：**Proposed v1.1**` vs L4 `执行状态（2026-10-23）：S0/S1/S2/S3 训练全部完成`；S4 报告 2026-10-27 | **③需 owner 裁决**：状态头须改 Executed-Frozen（A02/P0）；报告只读 | **是** |
| F15 | 30h R3→V5.2→V5.3 分层训练与验收 | v1.1(contradictory) | R3 语义仍有效；Stop 章节仅历史 | Stop 语义（§5.3/§5.6/T3.2）被 ADR-0030 取代 | 头部 L3 `EXECUTED-FROZEN / Stop 语义已取代（2026-10-21）` 与 L5 `状态：**待冻结的执行计划**` 自相矛盾；README L243 "Executed-Frozen / Stop 语义已被 ADR-0030 取代" | **③需 owner 裁决**（同文件双状态，A03/P1）；`historical` 门禁钉死不可移动 | **是** |
| F16 | V5.0–V5.2 十小时冲刺 | 方案(H)、冲刺执行报告(H) | 已执行（V5.1a rejected_gate） | 无 | README L226 `Executed-Frozen（V5.1a）/ Historical`；L122-L126 记录"V5.1a 已执行并 rejected_gate" | **②原地保留**（`historical` 门禁钉死；报告只读） | 否 |
| F17 | V5.1 修复冲刺与扩量准入 / V5.1-R2 | 修复冲刺 v1(unknown)、R2(S) | 无（R2 被 R3 取代） | V5.1-R2 | README L245 `Superseded by R3`；`docs_consistency_config.json` `historical` 含 R2 | **②R2 原地+横幅**（门禁钉死）；**③修复冲刺需裁决**（无状态头、1 引用） | **是** |
| F18 | V52R 全路径导航与端点停止修复 | 修复方案 v1(H)、输入充分性优化 v1(H)、决策记录报告(H) | 无（D2-G STOP 关闭，负结果归档） | 无 | README L87 `Closed — D2-G STOP`；AGENT.md `V5.2 / V5.2-R 已关闭（D2-G STOP，T1 不授权，负结果归档）` | **②原地保留**（负结果证据；报告只读） | 否 |
| F19 | EGNN→BC→IQL→SAC 训练方案三代 | v2(S)、v3(S)、IRC 直接位移 v1(S) | 无（均自声明被取代） | v2/v3/IRC v1 互相及后续方案 | 三份头部均含"被取代/已被取代"横幅；均**不在**门禁名单；引用数 3/7/2 | **③需 owner 裁决**：可迁 `docs/superseded/`，但需先改 2–7 处链接（当前①不成立：有引用） | **是** |
| F20 | 训练/测试集划分与数据来源 | 同文件名 v1→内容 v1.1(C) | v1.1（README 2026-11-20 已登记） | 无 | README L17 `v1.1（2026-11-20 晚）`；台账 "升 **v1.1**" | **②原地保留**；文件名 v1 vs 内容 v1.1 → A19 命名异常 | 否 |
| F21 | 仓库归档与清理方案 | v1(C,Operational) | v1（归档治理来源） | 无 | README L5 顶部条目；台账 2026-11-20 "新增 [PES2TS 仓库归档与清理方案 v1]" | **②原地保留**（docs/infra 操作规则来源） | 否 |

## 二、重点家族说明（需 owner 优先看）

### F11 / F14：状态滞后但已执行（P0）
- **V5.4.2**：方案 L3 `Active-Proposed`（"WP1–WP7 待实现与执行"）与 README L30 `WP7 EXECUTED（2026-10-31）` + 报告 20261031 直接冲突。家族内**当前有效者应是执行报告**，方案应转 `Executed-Frozen` 或加执行指针。ADR-0033 已 Accepted（v1.3.3），不受影响。
- **V5.3-MP**：方案 L3 `Proposed v1.1` 但同头部 L4 已自述 S0–S3 全部完成、且 S4 报告已出。状态头应改 `Executed-Frozen`；报告中 S3 结论"Move guard 全为负退化"是负结果证据，只读。

### F12 / F10：已执行但"不改冻结原文"是明示规则
- V5.4.3 台账 2026-11-06 明示"主方案与 WP 清单保持 `Frozen-Preregistered v1.0`（按预注册执行完毕，**不改文**）；ADR-0034 保持 Proposed（v1.1 补记在册）"。因此这**不是**文档错误，而是"执行证据在报告、状态停在与预注册时点"的设计；缺的是 `docs/README.md` 的 2026-11-06 条目（索引滞后，A04/A08）。
- V5.4.1 的 Executed-Frozen 写法可作为模板（§19 执行指针，不改 §0–§18）。

### F19：唯一"可迁 superseded 但暂不满足①"的家族
- 三份文件自身已声明被取代（横幅在头部），但 `doc_inventory.csv` 统计引用数为 3/7/2，不满足任务定义的"0 引用"硬条件；均不在 `historical`/`required_files` 门禁名单。
- 建议：先在各引用处补"（已取代，仅历史参考）"指针，再批量迁入 `docs/superseded/` 并同步 `docs/superseded/README.md` 索引与台账（一条增量登记）。

### F03 / F20：文件名带 `_v1` 但**未**声明被取代（人工裁决项）
- F03 三位一体修复 v1：状态 `Proposed v1.1`、README 记 `Active-Proposed（执行细化）`、superseded/README 明确"自身未声明被取代 → 留 plans/"。**不得**因文件名 `_v1` 视作旧版。
- F20 训练测试集划分标准 v1：文件名 v1，内容已 v1.1；属命名异常而非版本家族换代。

## 三、未被 FAMILIES 覆盖的多版本文件名组（families_scan 自动发现 11 组）
脚本 `autodetect_uncovered()` 按"去日期、去 `_vX` 后同基名"发现 11 组多版本/多日期文件，经人工核对主要是：
`V52R_输入充分性*`（已并入 F18）、`PES2TS_V5.4.3_*`（已并入 F12）、
以及 `_执行报告/报告` 与源方案**不同日期同主题**的报告族（报告只读，不构成取代链，故未单列家族）。
完整清单见 `families_evidence_check.json` 的 `uncovered_multi_version_groups`。

## 四、家族结论统计

| 项 | 数量 |
|---|---|
| 识别家族 | **21** |
| 家族成员文档 | **53** |
| 证据条（全部命中） | **58 / 58**（失败 0） |
| ①可立即归位（0 引用+自声明+非门禁） | **0**（4 份已在 superseded 完成归位；F19 3 份因有引用不满足） |
| ②需加横幅/原地保留 | **8 个家族**（F03、F05、F06、F07、F08、F09、F16、F17、F18 中的横幅项） |
| ③需所有者裁决 | **6 个案例**（F03 归属、F11、F13、F14、F15、F17 修复冲刺、F19 迁移） |
