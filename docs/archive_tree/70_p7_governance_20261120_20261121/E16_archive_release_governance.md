# E16 归档治理与公开化（LICENSE/划分名单/superseded 区）

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P7
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E16　**所属期**：P7（2026-11-20）　**关联 ADR**：无新 ADR　**负结果**：N20

---
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
