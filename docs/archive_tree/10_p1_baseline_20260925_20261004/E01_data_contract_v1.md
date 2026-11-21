# E01 数据底座与契约 v1（G0/G1 首批可复现证据）

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P1
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E01　**所属期**：P1（2026-09-25→09-30）　**关联 ADR**：ADR-0001　**负结果**：N01

---
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
