# PES2TS V5.3-MP 文档同步清单 v1

> 状态：**Active**（执行清单；每项完成后勾选）
> 日期：2026-10-21
> 依据：[ADR-0031](../design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md)、[MoveProcessContract v1](../contracts/PES2TS_MoveProcessContract_v1.md)、[V5.3-MP 计划](PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md)
> 强制门：`python scripts/agent_ops/docs_consistency.py --strict` 必须 CLEAN（0 violation）

## 0. 同步原则

1. **一切以最新为准**：宪法 v2.9 → ADR-0031 → MoveProcessContract v1 → V5.3-MP 计划；旧文档只引用，不重定义。
2. **旧文档不改内容**：只加状态横幅/指针；报告只读，只加“当前阶段/作废”指针。
3. **命名区分**：`V5.3-ENG` = 已执行的 30h rollout/relax 工程闭环；`V5.3-MP` = 新大规模 MOVE–PROCESS 战役。
4. **禁止词**：`TSStop` / `TS Stop` / `TS-neighborhood` / `approach-side` / `TS-directed` / `No-Move candidate` 只允许出现在带豁免标记（弃用/撤回/禁止/LEGACY/HISTORICAL/SUPERSEDED 等）的同一行。

## 0.1 执行状态（2026-10-21）

- **已完成**：1.1–1.6、2.1–2.4、3.1–3.4、4.1–4.3、5.1–5.7、6.1；registry 已重渲染，docs gate `--strict` **CLEAN（134 files）**，165 项相关测试通过。
- **待完成（S1 前）**：5.8（V5.1 扩量方案 pointer）、5.9（V5.1-R3 v1/v2 pointer）、6.2（CONTRIBUTING campaign 示例）。
- **不适用**：5.10（历史报告已有 POST-HOC 横幅，无需再加）。

---

## 1. 规范核心（P0，必须先完成）

| # | 文件 | 操作 | 状态 |
|---|---|---|---|
| 1.1 | `docs/design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md` | 新建（Accepted） | ✅ 已建 |
| 1.2 | `docs/contracts/PES2TS_MoveProcessContract_v1.md` | 新建（Active-Normative） | ✅ 已建 |
| 1.3 | `docs/plans/PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | 新建（Proposed） | ✅ 已建 |
| 1.4 | `docs/PES2TS_开发宪法_v1.md` | v2.8 → **v2.9**；新增 v2.9 增量段；§6.7 注记加 ADR-0031；§14 SSOT 表加 MoveProcessContract | ⬜ |
| 1.5 | `docs/contracts/PES2TS_FullPathNavigationContract_v1.md` | 顶部加 **SUPERSEDED by ADR-0031 / MoveProcessContract v1** 横幅；保留正文与 ADR-0030 证据链；加入 `historical` 列表前先确认横幅含 `ADR-0030` | ⬜ |
| 1.6 | `docs/design/decisions/ADR-0030-V5.2全路径导航与端点停止.md` | 顶部加 **Partially superseded by ADR-0031**（仅 Endpoint Stop 单一 d_end 语义被 PROCESS 取代；任务纠偏结论继续有效） | ⬜ |

---

## 2. SSOT 索引与治理（P1）

| # | 文件 | 操作 | 状态 |
|---|---|---|---|
| 2.1 | `README.md` | `## Current project direction` 插入 2026-10-21 V5.3-MP 段：MOVE+PROCESS、V5.2 关闭、S0 启动；旧 2026-10-21 task correction 段降为历史指针 | ⬜ |
| 2.2 | `AGENT.md` | §9 SSOT 行更新；§10 当前进度指针置顶 V5.3-MP（S0）；旧 V5.2-R 指针降为历史 | ⬜ |
| 2.3 | `docs/README.md` | 新增“2026-10-21 V5.3-MP（当前）”段；ADR-0031/Contract/计划入表；ADR-0030/FullPathNavigation 标 Superseded/Partially | ⬜ |
| 2.4 | `docs/plans/PES2TS_文档修订台账.md` | 新增 2026-10-21 条目；版本号 v3.19 → v3.20 | ⬜ |

---

## 3. 机器注册表与生成文档（P2）

| # | 文件 | 操作 | 状态 |
|---|---|---|---|
| 3.1 | `config/registry/g_v_map.yaml` | `plan_revision` 升 2026-10-21.2；V5.2-R → `status: closed`；旧 V5.3 → `V5.3-ENG`（executed, historical）；新增 `V5.3-MP`（proposed；gates `V53MP-S0-data`/`S1-smoke`/`S2-scale`/`S3-formal`/`S3-closed-loop`；config/scripts/code_scope） | ⬜ |
| 3.2 | `docs/design/PES2TS_G_V_MAPPING_v1_20261018.md` | 由 `python scripts/render_gv_map.py --write` 重渲染；`--check` 通过 | ⬜ |
| 3.3 | `docs/design/PES2TS_CODE_MAP_v2_20261018.md` | 新增 V5.3-MP 模块/脚本/配置路径；标 `ReactionProcessHead`、`key_atom_v53mp_*`、`r3_v53mp_run.py`；旧 `TSStopHeadV3` 标 LEGACY | ⬜ |
| 3.4 | `docs/plans/PES2TS_阶段定义_G0-G3.md` | 版本 v4.7 → v4.8；上游宪法 v2.7 → v2.9；G3.P 注记加 V5.3-MP；G3.P.3 的 STOP/MOVE-Q 说明改为 PROCESS 语义 | ⬜ |

---

## 4. 下游合同（P3）

| # | 文件 | 操作 | 状态 |
|---|---|---|---|
| 4.1 | `docs/contracts/PES2TS_GeometryPolicyContract_v2.md` | §4.2 `No-Move（Endpoint Stop）` 与 §5.2 stop labels 加“由 MoveProcessContract v1 取代”指针；窗口 0.10/0.30 标为诊断量 | ⬜ |
| 4.2 | `docs/contracts/PES2TS_ReactionIntent_Contract_v2.md` | §4.3 Stop（Endpoint Stop）与术语迁移表加指针；q 语义改为 PROCESS 辅助量 | ⬜ |
| 4.3 | `docs/contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md` | 顶部增补：2026-10-21 ADR-0031 增量；教师/包 schema 的新字段指向 MoveProcessContract | ⬜ |

---

## 5. 旧计划与报告（P4/P5）

| # | 文件 | 操作 | 状态 |
|---|---|---|---|
| 5.1 | `docs/plans/PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md` | 状态改为 **Closed（D2-G STOP）**；T1 段保留为“未授权历史文本”；顶部指向 V5.3-MP | ⬜ |
| 5.2 | `docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md` | 状态改为 **Executed/Closed**；§7 T1 条件协议标历史；§6 矩阵结果归档 | ⬜ |
| 5.3 | `docs/reports/PES2TS_V52R_输入充分性决策记录_20261021.md` | 加 **Addendum 2026-10-21b**（不重写原记录）：①“不可学”→“未获可学证据”；② P5 TS 为 Mask 带、无正例污染；③ P3 标 DEGENERATE_TRIGGER；④ sealed 14 行 QC 例外；⑤ sklearn pin | ⬜ |
| 5.4 | `docs/plans/PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md` | 顶部 v3.2 → **v3.3** 横幅；§3.3 V5.3 改为 V5.3-MP 指针；术语迁移表加 PROCESS | ⬜ |
| 5.5 | `docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md` | 横幅补 ADR-0031；旧 V5.3 rollout 标 **V5.3-ENG** | ⬜ |
| 5.6 | `V53_ROLLOUT_REPORT.md` | 顶部加横幅：V5.3-ENG（历史工程闭环）；V5.3-MP 为新战役；不改变原报告内容 | ⬜ |
| 5.7 | `docs/plans/PES2TS_单端点IRC生成控制器_产品设计_v1/v2/v3_20261020.md` | v1/v2 加 ADR-0031 指针；v3 的 Stop Head 图注改为 PROCESS | ⬜ |
| 5.8 | `docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md` | 加“V5.3-MP 取代其扩量准入语义”指针 | ⬜ |
| 5.9 | `docs/plans/PES2TS_V5.1-R3_*`（v1/v2/v3） | pointer 文案：Stop/Progress 归 PROCESS；不重写正文 | ⬜ |
| 5.10 | 根目录已 POST-HOC 的历史报告（24H/FAILURE/NEXT_DAY） | 无需改内容；如 token 扫描需要，保持现有横幅 | ⬜ |

---

## 6. 机器门配置（P6）

| # | 文件 | 操作 | 状态 |
|---|---|---|---|
| 6.1 | `scripts/agent_ops/docs_consistency_config.json` | `required_files` 增加 ADR-0031 / MoveProcessContract v1 / V5.3-MP 计划；`historical` 增加 FullPathNavigationContract v1（需先加含 ADR-0030 的 SUPERSEDED 横幅）；`root_scope` 增加 `V53_ROLLOUT_REPORT.md` | ⬜ |
| 6.2 | `CONTRIBUTING.md` | 如 campaign/gate ID 变化，更新 V/gv 示例为 V5.3-MP | ⬜ |

---

## 7. 执行顺序与验收

```text
1) 宪法 v2.9 + ADR-0031 + MoveProcessContract v1          ← 规范根
2) 旧合同/ADR 的 SUPERSEDED / Partially superseded 横幅
3) README + AGENT + docs/README + 台账
4) config/registry/g_v_map.yaml → render_gv_map.py --write
5) CODE MAP + 阶段定义
6) 旧计划/报告指针（5.1–5.10）
7) docs_consistency_config.json（required/historical/root_scope）
8) python scripts/agent_ops/docs_consistency.py --strict  → 0 violation
9) 相关测试（155 项 + 新增 V5.3-MP 测试）全绿
```

**完成判据**：`--strict` CLEAN；`git status` 只含预期文件；V5.2 决策记录 Addendum 已加；注册表与 G–V map 一致；README/AGENT 的“当前”指针唯一指向 V5.3-MP。

---

## 8. v1.1 增补（2026-10-21 用户评审后）

| # | 文件 | 操作 | 状态 |
|---|---|---|---|
| 8.1 | `PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md` | 升 v1.1：S0-A/B 真值访问、20k 目标、分片惰性加载、Endpoint 标签算法、三臂评估、S2→S3 放宽、闭环双预算、实施就绪门 | ✅ |
| 8.2 | `PES2TS_MoveProcessContract_v1.md` | 增补 §10.1：合法真值访问、训练/部署资格分离、`endpoint_label_spec_v1`、双预算、规模主变量 | ✅ |
| 8.3 | `config/registry/g_v_map.yaml` | V5.3-MP gate_note 增补 S0-A/B、20k、readiness、分片加载、Endpoint spec | ✅ |
| 8.4 | `README.md` / `docs/README.md` / 台账 | 计划版本引用更新为 v1.1；S0 目标表述为 20k | ⬜ |
| 8.5 | 实施文件 | `key_atom_loss_v53mp_v3.py`、`key_atom_v53mp_train_v3.py`、`r3_v53mp_run.py`、`pes2ts_v53mp_20261021.yaml`、分片加载器、Endpoint 标签生成器、bundle 导出 | ❌ 未实现（S0-F 7/7 前不得 S1） |

- **实施就绪门**：S0-F 的 7 项必须实际存在并通过测试；docs gate 通过不等于代码就绪。
- **真值访问门**：S0-A 允许空日志；S0-B 合法访问必须逐次记录，硬门是无未授权访问。

## 9. 变更记录

| 版本 | 日期 | 内容 |
|---|---|---|
| v1.0 | 2026-10-21 | 初版：P0–P6 同步清单；命名 V5.3-ENG / V5.3-MP；强制 docs gate |
