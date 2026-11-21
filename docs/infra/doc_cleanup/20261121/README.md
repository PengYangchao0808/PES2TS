# 开发档案树与文档清理 · 证据与配套材料（doc_cleanup/20261121）

> **状态**：Evidence-ReadOnly（本目录为 [开发档案树与文档清理方案 v1.0](../../PES2TS_开发档案树与文档清理方案_v1_20261121.md) 的配套证据与执行材料；**方案结论以主文档为准**）
> **层级**：L3 证据 / L4 运营（工程治理面，不涉及 G0–G4 能力与 V 战役语义）
> **取代关系**：无（本目录为一次性设计产出；不取代任何文档）
> **上游依赖**：[开发档案树与文档清理方案 v1.0](../../PES2TS_开发档案树与文档清理方案_v1_20261121.md)、[开发文档规范化归档方案 v2.0](../../PES2TS_开发文档规范化归档方案_v2_20261121.md)、[全量文档审计](../../doc_audit/20261121/README.md)、`AGENT.md` §9、`docs/PES2TS_开发宪法_v1.md`
> **下游消费者**：仓库所有者（审批与执行）、AI 执行会话、下次季度审计
> **当前版本**：v1 @ 2026-11-21（**未删除/未移动任何文件**；删除执行需 Phase 0 治理修订 + 所有者批准）

---

## 1. 本目录回答什么

1. **清理后文档怎么读** → `ARCHIVE_TREE.md`（7 期 / 17 条目档案树，781 行完整成稿）
2. **哪些文档删、哪些留、哪些先抽结论** → `CLEANUP_CANDIDATES.csv`（253 条逐条判定）
3. **删除安全吗、怎么分批删** → `CLEANUP_PLAN.md`（Phase 0 治理修订 + D1/D2/D3 命令与配置 diff）+ `CLEANUP_OVERRIDES.md`（**最终裁决数字，以此为准**）
4. **删除会不会断链** → `LINK_FIX.csv`（342 条待改链接）+ `check_deletion.py`（15/15 PASS 只读校验）
5. **留下的核心文档长期怎么维护** → `CORE_SET.md`（44 个受管单元）+ `MAINTENANCE.md`（双周/owner/五类落点）+ `DOC_OWNERS.csv` + `DISCUSSION_TEMPLATE.md`

## 2. 最终数字（Lead 裁决后，权威）

| 项 | 数值 | 说明 |
|---|---:|---|
| 文档总量 | 253 | 2026-11-21 审计清单口径 |
| **KEEP（不删）** | **101** | 含 34 ADR（全留）+ 8 模型卡 + 治理/门禁证据 + 产品/外部说明 |
| **safe 档删除** | **63** | 已被取代且无独有结论（56 `DELETE` + 7 合并） |
| **standard 档（推荐）累计删除** | **116** | 再删 53（52 抽结论后删 + 1 合并）→ 保留 137 |
| **aggressive 档累计删除** | **152** | 再删 36（21 抽结论 + 15 门禁配置受审变更）→ 保留 101 |
| 需先抽结论 | **88** | 73 过程文档 + 15 门禁钉死路径；结论先入档案树条目 |
| 门禁配置必须同 PR 改 | **15** | 根 memo 6 + 旧合同 4 + 历史计划 5（ADR 已全留，不再动） |
| 删除后待改链接 | **342** | 见 `LINK_FIX.csv`，逐条含 referrer/行号/batch |
| 档案树规模 | **32 份** | 7 期 17 条目 + 横切层/索引/证据/元数据层 |
| 核心受管单元 | **44** | 治理 11 + 合同 6 + 设计计划 12 + 档案树入口 7 + 产品外部 8 |

## 3. 文件索引

| 文件 | 类型 | 内容 |
|---|---|---|
| [ARCHIVE_TREE.md](ARCHIVE_TREE.md) | 设计+成稿 | 总纲/与四目录边界/7 期分期（边界证据）/条目模板/17 份完整成稿/跨期负结果表 N01–N21/期边界附录（105 KB） |
| [ARCHIVE_MAP.csv](ARCHIVE_MAP.csv) | 生成数据 | 253 条文档 → 17 条目 / X0 映射 + extract_needed + extract_content |
| [ARCHIVE_TREE_OUTLINE.txt](ARCHIVE_TREE_OUTLINE.txt) | 设计 | 拟建 `docs/archive_tree/` 文件树、17 条目×期号×ADR×负结果索引、落位门禁注意 |
| [CLEANUP_CANDIDATES.csv](CLEANUP_CANDIDATES.csv) | 决策数据 | 253 条：decision / intensity / unique_value / phase / gate_locked / referrers / recovery 等 |
| [CLEANUP_PLAN.md](CLEANUP_PLAN.md) | 执行手册 | Phase 0（ADR-0035 + 条款修订 + 审批点）/ tag+bundle+tar 快照/ D1–D3 命令与配置 diff / 回滚 / 风险表 / 绝不删除清单 |
| [CLEANUP_OVERRIDES.md](CLEANUP_OVERRIDES.md) | 裁决覆盖 | **ADR 全留等最终裁决与覆盖后数字；与 CLEANUP_PLAN 冲突时以本文件为准** |
| [KEEP_SET.csv](KEEP_SET.csv) | 决策数据 | 101 条保留集：层级 L0–L4 / 职责 / 维护频率 / owner 角色 / 是否入档案树 |
| [LINK_FIX.csv](LINK_FIX.csv) | 决策数据 | 342 条：删除后需同步修正的活跃 markdown 链接（referrer/行号/批次/处理建议） |
| [CORE_SET.md](CORE_SET.md) | 设计 | 44 个核心受管单元（含 ADR 全留论证、旧版合同处置、逐单元维护要求） |
| [MAINTENANCE.md](MAINTENANCE.md) | 设计 | 双周 30min 同步/里程碑冻结/季度档案刷新 + 五类落点判定树 + owner 交接 + L1–L4 审批 + 4 指标 |
| [DOC_OWNERS.csv](DOC_OWNERS.csv) | 设计 | 65 行：核心单元 + 目录巡检的 owner 角色/频率/审查日 |
| [DISCUSSION_TEMPLATE.md](DISCUSSION_TEMPLATE.md) | 模板 | 讨论记录模板 + FAQ 模板 + V5.4.3 状态裁决完整示例与落档检查 |
| [check_deletion.py](check_deletion.py) | 可复跑脚本 | 只读校验：门禁钉死/可恢复/链接预演/三档单调/互补（当前 15/15 PASS） |
| [gen_candidates.py](gen_candidates.py) / [apply_keep_overrides.py](apply_keep_overrides.py) / [regen_linkfix.py](regen_linkfix.py) | 可复跑脚本 | 候选生成 / Lead 裁决覆盖 / 链接清单重算 |

## 4. 可复跑命令

```powershell
# 只读校验（删除前必须全 PASS）
python docs/infra/doc_cleanup/20261121/check_deletion.py

# 重新生成候选（改动判定规则后）
python docs/infra/doc_cleanup/20261121/gen_candidates.py
python docs/infra/doc_cleanup/20261121/apply_keep_overrides.py   # ADR 全留等裁决
python docs/infra/doc_cleanup/20261121/regen_linkfix.py

# 门禁（任何文档改动后）
python scripts/agent_ops/docs_consistency.py --strict
python scripts/check_docs_links.py
python scripts/render_gv_map.py --check
python -m pytest -q tests/test_gv_registry.py tests/test_config_env.py
```

## 5. 边界与已知限制

1. **未执行任何 `git rm` / `git mv`**：本目录全部为"设计 + 清单 + 校验"，删除必须经 Phase 0 治理修订（ADR-0035 + `AGENT.md` §9.4 例外）与所有者批准。
2. `CLEANUP_PLAN.md` 是 task-7 机器初稿（67/129/177 口径）；**最终数字（63/116/152、KEEP 101）以 `CLEANUP_OVERRIDES.md` 为准**，执行时按覆盖件调整批次清单。
3. `ARCHIVE_TREE.md` 的 17 份成稿是"草稿级完整"：指标均可在原文状态行/结论段找到，但 Phase 2 落盘时需所有者抽审 3 期（建议 E03/E10/E14）确认口径。
4. `check_deletion.py` 的 b1 预演按 basename 匹配，已排除 `docs/archive/` 快照同名文件（否则会把"指向 live 文档的链接"误报为指向被删快照）。
5. 删除的兜底是 git 历史 + `pre-doc-purge-20261121` tag + bundle；任何批次失败可 `git reset --hard` 回滚。
