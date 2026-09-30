# PES2TS Demo 评估指标 v1

本文件定义 Demo24 的 generation、ranking、端到端和计算成本指标。当前只有合成运行样例，24 条真实运行结果和人工评价标签尚未形成；不设目标阈值，也不把合成结果当成实测基线。

## 统计单位与输入

- **统计单位：**一个 reaction_id 对应的一份冻结 `ReactionCase`。v1 拒绝同一反应的多个 case snapshot 同时进入 cohort，避免快照或候选数量抬高分母。分母来自显式传入的 `cohort_cases` 冻结队列；某个案例缺少运行记录时仍计入分母并显示为未开始。候选/帧/重试是该反应内的明细，不能各自增加样本分母。
- **每行输入：**`case`、可空的 `plan`、`execution`、`path`、`proposals` 和 `validation`。S2–S5 的身份关系与内容摘要必须匹配。一个 Demo24 版本每个 `case_id` 只有一行；多候选研究需先冻结候选选择策略，不得把多候选展开成多条反应。
- **排名标签：**作为独立的 `ranking_labels` 参数提供，使用 `case_id` 和该 PathBundle 中可接受的 `acceptable_frame_ids`。不得写入生产 `ReactionCase`、ScanPlan 或执行输入。标签需在生成与排序方案冻结后再用于 valid 评价。
- **标签来源：**每行还必须包含 `label_source`、标签方法 `method` 和来源文件 SHA256 `source_sha256`，以便追溯标签构造。
- **分组：**按 ReactionCase 的 `split` 分开报告 train/valid/test/unassigned。ranking 默认只测 `valid`，避免把训练集调参分数当作泛化效果。

## 指标定义

| 阶段 | 指标 | 分子 / 分母 | 解释 |
| --- | --- | --- | --- |
| Generation | Ready plan rate | `ScanPlan.status=ready` 的案例数 / 案例总数 | 计划覆盖；显式拒绝保留在分母 |
| Generation | Completed execution rate | 至少有一条已绑定 `ExecutionRecord.status=completed` 的案例数 / 案例总数 | 计算任务完成，不等于路径可用 |
| Generation | Usable path coverage | `PathBundle.status=usable` 的案例数 / 案例总数 | 技术路径覆盖，不等于 TS 物理验证通过 |
| Ranking | Proposal coverage | 有非空、来自指定规则的排序建议案例数 / 有 ranking label 且路径 usable 的案例数 | 无建议的合格路径按未覆盖报告 |
| Ranking | Recall@k | Top-k 命中至少一个可接受帧的案例数 / 有 ranking label 且路径 usable 的案例数 | 逐案例命中率；Top-k 默认 1、3 |
| Ranking | MRR | 每案例第一个可接受帧名次倒数的均值；无命中记 0 | 反映候选位置，不按帧数加权 |
| End-to-end | Validation pass rate | `ValidationResult.status=passed` 案例数 / 案例总数 | 通过需满足 S5 物理验证合同；missing/not_run 不算通过 |
| Cost | CPU seconds | ExecutionRecord 与 ValidationResult 阶段成本之和 | 缺成本时完整总成本为 null，另报已知成本下界及成本完整案例数 |

排名只在有效标签且存在技术可用 PathBundle 的案例上计算，同时报告有标签但没有 usable path 的案例数。这把 generation 覆盖率与“给定可用路径时”的 ranking 能力分开；不以缺失路径伪装排序失误，也不丢弃分母而不披露覆盖损失。无 valid 标签时 ranking `status=not_measured`，指标为 null。

## 失败与成本报告

每个 split 同时报告 plan 拒绝原因直方图、ExecutionRecord 失败码、PathBundle 状态计数，以及 ValidationResult 的 `not_run`/`incomplete`/`failed`/`passed`/missing 计数。执行重试保留在单个 ExecutionRecord 内；failure code 按失败 attempt 计数，案例覆盖率仍按 ReactionCase 计数。

每个案例的已知 CPU 秒数可作为下界相加。只要任一已发生阶段的成本不完整，全 Demo `total_cpu_seconds` 和平均每案例总成本记为 null；不将缺项当 0。未运行的验证阶段成本按 0 计。此版本不估算 wall time、美元成本或 GPU 成本。

## 实现

`pes2ts_core.demo_metrics.evaluate_demo_runs` 实施以上统计。它验证合同、split、plan/execution/path/proposal/validation ID 绑定，拒绝重复 `case_id`、过期 proposal 和来自非 ranking 的候选；输出 `pes2ts_demo_metrics_v1` JSON-compatible 摘要。输入标签不会被回写到生产合同。

## 运行包与命令

`pes2ts_demo_run_bundle_v1` 清单显式引用完整冻结 cohort 和可用运行对象；cohort 是 generation 分母权威来源。全部路径相对清单目录，不能越出该目录。可空对象用 JSON `null`，proposal 使用路径数组；ranking 标签文件单独放置：

```json
{
  "schema_version": "pes2ts_demo_run_bundle_v1",
  "cohort_cases": ["cases/RXN_0001.json", "cases/RXN_0002.json"],
  "ranking_labels": "evaluation/valid_labels.json",
  "runs": [
    {
      "case": "cases/RXN_0001.json",
      "plan": "plans/RXN_0001.json",
      "execution": "executions/RXN_0001.json",
      "path": "paths/RXN_0001.json",
      "proposals": ["proposals/RXN_0001_highest_energy.json"],
      "validation": "validations/RXN_0001.json"
    }
  ]
}
```

生成报告：`python scripts/evaluate_demo_runs.py run_bundle.json --output metrics.json --evaluation-split valid --ranking-rule highest_scan_energy`。标签文件缺省或无 valid 标签时 ranking 会显示 `not_measured`，不会猜标签或产生 0 分。

当前没有经人工复核并冻结的 24 条运行包，因此实际 generation/ranking/端到端效果、失败率和成本仍待计算后填写；本定义本身不构成效果证据或达标声明。
