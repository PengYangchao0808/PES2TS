# E09 V5.0–V5.2 十小时冲刺与 R3 首轮

> **状态**：Archived（档案条目，冻结后只可追加“后续指针”）
> **层级**：L3 证据 / 开发档案树 · P4
> **取代关系**：无（档案条目；原文已被退役或作为指针保留）
> **上游依赖**：ADR-0035、`docs/infra/doc_cleanup/20261121/ARCHIVE_TREE.md`
> **下游消费者**：维护者、季度审计、AI 会话
> **当前版本**：v1.0 @ 2026-11-21
> **条目编号**：E09　**所属期**：P4（2026-10-18→10-19）　**关联 ADR**：ADR-0026　**负结果**：—

---
> 条目标号：E09　时间范围：2026-10-18 → 2026-10-19　状态：Archived
> 关联 ADR：ADR-0026（V5 冲刺 key-atom 位移合同与 OOF 语义，Accepted）
> 证据根：`docs/plans/PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md`、`docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md`、`docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md`

## 目标
V4 负结果后 24 小时内启动 V5 冲刺：先做 OOF 交叉拟合验证配方，再决定是否进入修复冲刺与扩量。

## 主要成果与量化指标
- sprint status `complete_oof_1seed`：prepare-audit / label-screen / recipe-recheck / crossfit 均 complete，report 未执行；G5.1 fixtures pass；Oracle static ceiling `ORACLE_STATIC_OK`；封存评测 `REFUSED (sealed_confirm_not_opened)`；复核稳定配方 **A05**。
- V5.1 修复冲刺与扩量准入方案落盘（10-19，无状态头，A14/P2 异常）。

## 关键负结果
- 报告未执行、封存评测被明确拒绝 → 只有单 seed OOF 证据，不能作正式结论。
- V5.1 修复冲刺方案无状态声明（治理异常），其"扩量准入"未经独立签认。

## 证据指针
- `docs/reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md`（阶段状态表）
- `docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md`（ANOMALIES A14）

## 后续影响
- 冲刺余量并入 E10 的 24H 战役；V5.1a 在 5-fold 门被 rejected（见 E10 上位计划与根 README 10-20 状态段）。
