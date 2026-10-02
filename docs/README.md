# PES2TS 文档索引

2026-10-02 更新。代码与数据说明见仓库根 [README](../README.md)；本目录收集计划、设计、合同与报告。

> 文档治理见 [PES2TS_文档修订台账](plans/PES2TS_文档修订台账.md)：唯一主干编号为 **G0–G3**，工作流为 **C（清理）/ R（生成研究）/ X（执行收敛）**。

## 权威文档（当前，只看这几份了解现状）

| 文档 | 内容 |
| --- | --- |
| [PES2TS_阶段定义_G0-G3.md](plans/PES2TS_阶段定义_G0-G3.md) | **阶段与边界定义唯一来源**：G0–G3、G1→G2 边界合同、退出条件 |
| [PES2TS_整体开发与发布方案_v2.md](plans/PES2TS_整体开发与发布方案_v2.md) | **项目级开发与发布主方案**（原始假设 + 最新修订） |
| [ADR-0001-统一执行后端接缝.md](design/decisions/ADR-0001-统一执行后端接缝.md) | 执行层统一决策（Track A/B 收口） |
| [PES2TS_文档修订台账.md](plans/PES2TS_文档修订台账.md) | 文档状态与取代关系 |

## plans/ — 项目级计划与验收

| 文档 | 状态 | 内容 |
| --- | --- | --- |
| [PES2TS_阶段定义_G0-G3.md](plans/PES2TS_阶段定义_G0-G3.md) | **权威** | 阶段与边界定义 |
| [PES2TS_整体开发与发布方案_v2.md](plans/PES2TS_整体开发与发布方案_v2.md) | **权威** | 开发与发布主方案 |
| [PES2TS_文档修订台账.md](plans/PES2TS_文档修订台账.md) | **权威** | 文档治理与取代关系 |
| [PES2TS_双项目开发与Demo方案.md](plans/PES2TS_双项目开发与Demo方案.md) | 参考 | 定位（generation / ranking）、合同、指标 |
| [PES2TS_统一开发顺序与阶段验收.md](plans/PES2TS_统一开发顺序与阶段验收.md) | 历史 | S0–S8 阶段表（已并入 G0–G3） |
| [PES2TS_逐步实施与验证方案.md](plans/PES2TS_逐步实施与验证方案.md) | 历史 | 早期分步方案 |
| [PES2TS_实施路线图.html](plans/PES2TS_实施路线图.html) | 参考 | 实施路线图（浏览器打开） |
| [PES2TS_扫描规划谱系合并方案_20260930.md](plans/PES2TS_扫描规划谱系合并方案_20260930.md) | 参考 | 规划两代谱系归一裁决（已并入 G2.4/X） |
| [PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md](plans/PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md) | 参考 | R 工作流范围与验收来源 |
| [PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md](plans/PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md) | 参考 | R 工作流 M0–M4 来源 |

## design/ — 阶段设计文档

| 文档 | 内容 |
| --- | --- |
| [G1_TS_IRC_映射与反应归簇实现方案.md](design/G1_TS_IRC_映射与反应归簇实现方案.md) | G1 真值层（P1 映射 / P2 分类）设计 |
| [G1_v2_补全实施总方案.md](design/G1_v2_补全实施总方案.md) | G1 v2 完成定义、F/B/O 语义、G1-0…G1-7 关卡（外部工作流已并入图论设计 P0–P3） |
| [G1_v2_L0到G2扫描策略设计.md](design/G1_v2_L0到G2扫描策略设计.md) | L0 群体与扫描路由分析 |
| [G1_v2_ORCA扫描模式识别与计划.md](design/G1_v2_ORCA扫描模式识别与计划.md) | ORCA 扫描模式识别与计划 |
| [G1_v2_一维优先与xTB二维成本策略.md](design/G1_v2_一维优先与xTB二维成本策略.md) | 一维优先与 xTB 二维网格成本分析 |
| [G1_v2_产物成键逆向扫描优先策略.md](design/G1_v2_产物成键逆向扫描优先策略.md) | 产物侧逆向扫描与 H 转移捷径分析 |
| [G1_v2_方向无关成键锚点与双向扫描策略.md](design/G1_v2_方向无关成键锚点与双向扫描策略.md) | 方向无关锚点基线 |
| [G1_v2_多成键与少成键方向对比.md](design/G1_v2_多成键与少成键方向对比.md) | 混合选边（多/少成键）对比 |
| [G1_v2_成断键扫描方向选择可视化.html](design/G1_v2_成断键扫描方向选择可视化.html) | 交互式 F/B 选择可视化（浏览器打开） |
| [PES_GENERATION_图论扫描策略选择器设计_v1.md](design/PES_GENERATION_图论扫描策略选择器设计_v1.md) | 图论选择器设计 v1：三层图、11 策略、四角色坐标、P0–P3（实施并轨记录见头部） |

## contracts/ — 数据合同与对接规范

| 文档 | 内容 |
| --- | --- |
| [PES2TS_contracts_v1.md](contracts/PES2TS_contracts_v1.md) | 七类核心对象 + 附属对象的字段字典 |
| [PES2TS_ACP字段映射_v1.md](contracts/PES2TS_ACP字段映射_v1.md) | PES2TS 合同 ↔ ACP v2 字段映射 |
| [PES2TS_数据规范与ACP对接优化方案.md](contracts/PES2TS_数据规范与ACP对接优化方案.md) | 数据规范与 ACP 对接优化（设计建议） |

## demo/ — Demo24 流程

| 文档 | 内容 |
| --- | --- |
| [PES2TS_demo24_反应挑选与复核方案.md](demo/PES2TS_demo24_反应挑选与复核方案.md) | 24 条候选筛选硬条件、A–H 分层、双人复核与替补规则 |
| [PES2TS_Demo评估指标_v1.md](demo/PES2TS_Demo评估指标_v1.md) | Demo 指标定义（generation / ranking / 端到端 / 成本） |
| [PES2TS_Demo实现审核与架构解读_20260930.md](demo/PES2TS_Demo实现审核与架构解读_20260930.md) | Demo 实现审核（F1–F6 与阶段判断） |

## reports/ — 阶段性报告

| 文档 | 内容 |
| --- | --- |
| [PES2TS_开发阶段性报告_20260930.md](reports/PES2TS_开发阶段性报告_20260930.md) | 全阶段（G0/G1/G1 v2/G2/集成/Demo24）进展、风险与路线图 |
