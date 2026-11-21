# 2026-11-21 开发文档全量审计证据（doc_audit/20261121）

> **状态**：Evidence-ReadOnly（只读证据；结论与规则以 [开发文档规范化归档方案 v2.0](../../PES2TS_开发文档规范化归档方案_v2_20261121.md) 为准，本目录只提供可复跑的原始数据）
> **层级**：L3 证据 / L4 运营（工程治理面，不涉及 G0–G4 能力与 V 战役语义）
> **取代关系**：无（本目录为一次性审计证据，不取代任何文档）
> **上游依赖**：[开发宪法](../../../PES2TS_开发宪法_v1.md)、[AGENT.md](../../../../AGENT.md) §9、[文档修订台账](../../../plans/PES2TS_文档修订台账.md)、[docs/README.md](../../../README.md)、`scripts/agent_ops/docs_consistency_config.json`、`scripts/check_docs_links.py`、`config/registry/g_v_map.yaml`
> **下游消费者**：执行归档批次的所有者、方案复核者、下次季度审计（复用本模板）
> **当前版本**：v1 @ 2026-11-21（采集时点 2026-11-21T22:28–22:45 +08:00，基线 master@2f1d3ba）

---

## 1. 本次审计回答什么

1. 仓库里**到底有哪些开发文档**（全量 253 条、可机读、含 git 元数据）→ `doc_inventory.csv`
2. 每份文档**现在是什么状态、应该怎么处置**（保留 / 加横幅 / 迁入 superseded / 只读证据 / 待裁决）→ `DISPOSITION.csv`
3. 现行治理**允许与禁止什么**（规则、门禁、52 条钉死路径、22 条禁止破坏项）→ `GOVERNANCE_CONSTRAINTS.md`
4. 现状**有哪些异常与治理负债** → `ANOMALIES.md`
5. 若要执行，**按什么批次、什么命令、怎么回滚** → `BATCH_PLAN.md`
6. 上述结论**是否经得起独立复核** → `VERIFICATION_REPORT.md`

## 2. 采集口径（CALIBER）

**白名单扫描范围**（不递归扫 `data/`、`outputs/`——两者合计 12 GB+，且已被 `.gitignore` 隔离）：仓库根（`*.md`+`*.txt`+`LICENSE`）、`docs/`（递归 `*.md`+`*.html`）、`examples/`、`artifacts/`（仅 `model_card.md`）、`scripts/` 与 `data|outputs/` 的 README。**最终口径与逐目录命中数见 [inventory_stats.md](inventory_stats.md) §0**（本表只给总量）。

**总量**：253 条（`.md` 245 / `.html` 4 / `.txt` 3 / `LICENSE` 1）、4,972,194 B、55,452 行；249 条已跟踪 + 4 条本轮新增未跟踪。

> **口径声明（2026-11-21 复核后）**：253 条是**审计启动时点（22:32）的快照**。审计目录自身随后新增的 7 份证据 md（`ANOMALIES/FAMILIES/RECOMMENDATIONS/BATCH_PLAN/inventory_stats/MAPPING_SUMMARY/VERIFICATION_REPORT`）作为**审计证据**统一按方案 §6.3/§10.3 管理（`Evidence-ReadOnly`、G5 豁免、不纳入本轮处置映射），**不是清单遗漏**——"审计清单包含审计报告自身"会形成自指循环。复跑 `collect_inventory.py` 会得到更大的数字，属预期；复跑前请先明确口径。

**排除项**：`.git/`、`.slim/`（本地工作区）、`data/` 数据文件、`outputs/` 大文件、缓存目录；`docs/` 下被排除的非文档文件（状态 JSON、归档字节副本 `.snapshot.txt`、`.lock`、fixture 等，约 70 个；精确清单见 `inventory_stats.md` §0）。

**字段口径**：
- `declared_status`：解析文件顶部 20 行内的 `状态:`/`Status:` 字段（兼容 `> **状态**：`、表格行形态）；取不到时回退匹配 `Superseded/Historical/Executed-Frozen/Active-Proposed/Frozen-Preregistered/Evidence/参考/历史/权威` 开头行；**不推断**。
- `ref_count`：以"文件名基本名"为单位的全库文本出现次数（语料 1,305 文件 / 21,468,562 字符，排除 archive 快照）。**注意**：11 份 `README.md`、8 份 `model_card.md` 共享基本名，`ref_count` 不区分实例；判断具体文件是否被引用须 `git grep "<唯一文件名或完整相对路径>"` 复核。
- `on_disk_flag`：`1` = 被 git 跟踪（`git ls-files`）；`0` = 磁盘存在但未跟踪。

## 3. 文件索引

| 文件 | 类型 | 内容 |
|---|---|---|
| [doc_inventory.csv](doc_inventory.csv) | 生成数据 | 全量清单（253 条）：path/ext/size/lines/sha256/mtime/git 首末提交/title/zone/doc_type/declared_status/declared_version/declared_date/ref_count/on_disk |
| [inventory_stats.md](inventory_stats.md) | 统计报告 | 采集口径、各维度计数、全量明细、无状态块全列、未跟踪列、stale>7d 列 |
| [doc_inventory.json](doc_inventory.json) | 生成数据 | 同数据 + `stats` 汇总块（供程序读取） |
| [DISPOSITION.csv](DISPOSITION.csv) | 决策数据 | 全量处置映射：canonical_status、target_zone、action、priority、batch、blocker、reason、evidence、verification（253 行，自校验 PASS） |
| [LINK_FIX.csv](LINK_FIX.csv) | 决策数据 | 条件迁移（F19 三份）的 13 处引用修正清单：referrer/line/context/old_target/new_target（仓库根相对）/markdown_link（应写入的相对链接）/action（11 处 pending-approval + 2 处 no-change） |
| [MAPPING_SUMMARY.md](MAPPING_SUMMARY.md) | 证据 | 处置映射的**独立校验报告**：覆盖/钉死/move 目标/LINK_FIX/批次数字逐项 PASS，附口径建议与 Lead 收口追加 |
| [validate_mapping.py](validate_mapping.py) | 可复跑脚本 | 独立校验 DISPOSITION/LINK_FIX/BATCH_PLAN（断言 a–d + e1–e5；当前 PASS，FAIL=0） |
| [build_disposition.py](build_disposition.py) | 可复跑脚本 | 由 `classification_mapping.csv` + 门禁规则生成 DISPOSITION.csv 与 LINK_FIX.csv（含 253 行完整性自校验） |
| [classification_mapping.csv](classification_mapping.csv) | 生成数据 | 253 条分类桶映射：class_bucket、suggested_action、gate_locked、inbound_refs、evidence_note（分类结论底稿） |
| [families.csv](families.csv) | 生成数据 | 21 个版本家族/53 个成员及其取代链初判 |
| [FAMILIES.md](FAMILIES.md) | 证据 | 21 家族总表 + 重点家族说明（F11/F14 状态滞后、F19 可迁候选、F03/F20 待裁决） |
| [GOVERNANCE_CONSTRAINTS.md](GOVERNANCE_CONSTRAINTS.md) | 证据+规则 | 30 条现行规则（G-xx）、4 类门禁实现事实与实测基线、52 条钉死路径逐条、状态词表统计、12 项冲突（C-01…C-12）、22 条禁止破坏项（D-01…D-22） |
| [ANOMALIES.md](ANOMALIES.md) | 证据 | 32 条精编异常（P0 2 / P1 10 / P2 20），带文件:行号证据；机械候选 60 条见 [anomalies.csv](anomalies.csv) |
| [RECOMMENDATIONS.md](RECOMMENDATIONS.md) | 证据 | 4 轴分类维度 + 桶计数 + 三类处置结论（可立即归位 0 / 加横幅 7 / 需裁决 19）+ 必须原地保留清单 |
| [BATCH_PLAN.md](BATCH_PLAN.md) | 执行手册 | B0–B3 批次：前置检查、变更清单、`git mv` 命令、链接修正、门禁复跑、回滚、验收判据 |
| [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md) | 证据 | 独立复核结论（完整性/合规性/自洽性/可执行性/门禁实测/缺口） |
| [MIGRATION_LOG.md](MIGRATION_LOG.md) | 流水模板 | 若执行 B1/B2，逐条记录 旧路径→新路径、引用修正数、门禁结果、执行人 |
| [baseline_gate_outputs.txt](baseline_gate_outputs.txt) | 原始输出 | 4 类门禁命令/时间/退出码与完整 stdout |
| [pinned_paths.txt](pinned_paths.txt) | 生成数据 | 17 historical + 27 required_files + 9 root_scope + `g_v_map` 43 条文档引用的逐条明细 |
| [status_vocabulary_stats.txt](status_vocabulary_stats.txt) | 生成数据 | 状态取值 Top 20、17 个归并桶、5 种格式变体、42 份无状态块 live 文档 |
| [index_coverage_stats.txt](index_coverage_stats.txt) | 生成数据 | 双索引覆盖：README 133 / 台账 100 / 双有 93 / 双缺 40 及逐条清单 |
| [anomalies.csv](anomalies.csv) | 生成数据 | 机械候选 60 条（类别/严重度/路径/证据/建议） |
| [collect_inventory.py](collect_inventory.py) | 可复跑脚本 | 全量清单采集（幂等，默认仓库根，UTF-8 输出） |
| [anomaly_scan.py](anomaly_scan.py) / [families_scan.py](families_scan.py) / [make_classification.py](make_classification.py) / [docscan_lib.py](docscan_lib.py) | 可复跑脚本 | 异常扫描 / 家族识别 / 分类桶生成 / 共用扫描库 |

## 4. 可复跑命令

```powershell
# 门禁基线（4 门；期望 0 violation / 0 error / current / 26 passed）
python scripts/agent_ops/docs_consistency.py --strict
python scripts/check_docs_links.py
python scripts/render_gv_map.py --check
python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py

# 全量清单重生成（覆盖 doc_inventory.csv；口径见 §2）
python docs/infra/doc_audit/20261121/collect_inventory.py
```

**采集时点提醒**：清单是"某一时刻的快照"。新增/迁移文档后必须重跑并同步 `DISPOSITION.csv`，否则 `DISPOSITION` 的完整性校验（`validate_mapping.py`）会失败——这正是方案 §10.3 G5 要自动化的事情。

## 5. 本次审计的边界与已知局限

1. **只读**：未移动/删除任何文件、未改门禁配置、未改 `docs/reports/**`、未触碰冻结件与预注册配置；两张 CSV 是"建议"，执行权在仓库所有者。
2. `ref_count` 对通用文件名（README/model_card）失真（见 §2），处置判定已用 `git grep` 逐条复核关键项。
3. 异常判定以"可复现的文件事实"为准，**不使用"看起来旧"**作为依据；状态矛盾的裁决权在文档所有者（方案 §14 列 11 项待裁决）。
4. `GOVERNANCE_CONSTRAINTS.md` 中的门禁基线为 2026-11-21T22:27 实测；本轮新增方案与审计目录后 `scanned`/`checked` 数会增大，violation/error 必须仍为 0（复核见 `VERIFICATION_REPORT.md`）。

## 6. 与方案的关系

- 本目录是 [开发文档规范化归档方案 v2.0](../../PES2TS_开发文档规范化归档方案_v2_20261121.md) 的**证据附件**：方案正文给结论与规则，本目录给可复跑数据与执行手册。
- 方案 §9-B0 中的"审计证据落盘"即本目录；下次季度审计（方案 §13.2-M5）直接复用本目录结构与脚本，新建 `doc_audit/<YYYYMMDD>/`。
