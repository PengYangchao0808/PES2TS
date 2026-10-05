# PES2TS G2.T 战役合同字段字典 v1（G2-AB2）

> 规范根：[PES2TS 开发宪法_v1](../PES2TS_开发宪法_v1.md)（P1 存在性+校验、§9.6 接口登记）。
> 登记决策：[ADR-0004](../design/decisions/ADR-0004-G2-AB2接口登记.md)。
> 实现落点：`pes2ts_core/generation/campaign.py`、`campaign_report.py`、`planning/cost_capture.py`、`planning/replay_report.py`、`planning/branch_calibration.py`、`integration/acp/origin_backend.py`。
> 状态：离线 V0 已验证（默认测试套件）；真实引擎执行（S2 试点/S3 全量/S4 验证）为 gated，需 ACP/ORCA 环境与人工复核。

本字典按宪法 P1 只固定**存在性与校验要求**，不冻结字段集合；语义破坏性变更须新版本 + 新 ADR。

## `pes2ts_campaign_manifest_v1`（战役清单）

| 字段 | 存在性/校验 |
| --- | --- |
| `cases[]` | ≥1 条；`reaction_id` 唯一；`snapshot_path` 为文件路径；`snapshot_sha256` 为小写 64 位十六进制（快照字节摘要，恢复时重验）。顺序按 reaction_id 排序冻结 |
| `pilot_cases[]` | ⊆ cases 的 reaction_id 集合（可空） |
| `frozen_parameters.continuation_policy` | 可构造 `ContinuationPolicy`（非法值拒绝） |
| `frozen_parameters.local_policy` / `branch_policy` | 可构造 `LocalCorrectorPolicy` / `BranchPolicy` |
| `frozen_parameters.origin_preparation` | `{method, timeout_seconds>0, capability_probe}` |
| `frozen_parameters.candidate_top_k` | 整数 ∈ [1,3] |
| `budget` | `max_frames_per_path`/`max_attempts_per_path`/`max_wall_seconds_per_path` **必须等于**冻结 continuation policy 的 `max_frames`/`max_attempts`/`max_seconds`（单一事实来源；违反 = `BUDGET_POLICY_MISMATCH`）；`max_gradient_calls_per_path`/`max_total_gradient_calls` ≥1；`max_session_wall_seconds` ≥0（0=不限） |
| `acp_wiring` | `{root, python, config_path, resolve_from_environment}`；null 表示经 `PES2TS_ACP_*` 环境变量解析 |
| `review_status` | 逐例映射，键集合 = cases 全集；值 ∈ {needs_review, accepted, rejected} |
| `reference_failures` | 键 ⊆ cases；值含 `last_accepted_lambda`（round2/round3 已知失败参照） |
| `content_sha256` | 战役身份摘要（除自身与 `generated_at` 外全量）；同一输出根目录拒绝不同身份（`CAMPAIGN_ROOT_IDENTITY_MISMATCH`） |

## `pes2ts_g2t_terminal_state_v1`（逐例终态）

| 字段 | 存在性/校验 |
| --- | --- |
| `terminal_class` | ∈ {rejected_typed, partial_prefix, complete_path, validated_passed, validated_failed, validated_incomplete, censored_budget, blocked}。前三类 + `validated_*` = 报告四类分母；`censored_budget`/`blocked` 为显式非成功审计态，**永不折入成功分母** |
| `label_state` | 宪法 §9.2 七态；verified 态必须携带证据引用（R2） |
| `stage_reached` | ∈ {input, origin_preparation, plan, continuation, candidate_extraction} |
| `failure_code` / `blocked_reason` + `blocked_needs` | typed（P2）；blocked 必带需求清单；`blocked_on_human_review` 不自动生成、不降格为跳过 |
| `identity_conflict_count` | 整数；`IDENTITY_CONFLICT` 计入 B1-1 对照 |
| `n_accepted_frames_beyond_origin` / `last_accepted_lambda` / `completed_interval` | 生成轨道判定输入 |
| `origin_status` / `origin_failure_code` | `endpoint_method_evidence_v1` 的透传；断键失败须为 `ORIGIN_CONNECTION_LOST` + 原子对/前后距离/源 hash |
| `candidates` | `{n, evidence_classes, completed_interval, ref}`（B1-4：部分前缀候选不被静默丢弃） |
| `costs` | `{n_gradient, cpu_seconds, ref→costs.json}`；无账本不入报告（R5） |
| `validation` | 默认 `{status: blocked_on_human_review, needs[...]}`；`attach_validation_result` 折入 typed `ValidationResult` 后升 `validated_*`，化学失败为合法终态（R2） |

幂等/恢复纪律：终态存在 + 输入 hash 未变 + 非 blocked → 跳过；blocked 可重试；恢复运行不得改参数/预算。

## `pes2ts_case_costs_v1`（逐例成本账本）

attempt 级 `flywheel.cost_ledger` 列表 + `aggregate`（父子经 `includes_cost_ids` 去重）。计数规则：`N_energy=N_gradient`（EnGrad 同次）；缓存命中 = 真实零 + `cost_source="cache_hit:*"`；`cpu_seconds` 优先 receipt `wall_seconds`，缺失 null+reason（禁写 0）；曲率探针独立账本（内部梯度只计一次）；blocked 案例写空账本（零尝试可审计记录）。

## `pes2ts_branch_calibration_v1`（分支校准报告）

存在性：`grid`、`criteria`（branch_selection_ok / biased_only_honest / cumulative_gate_discriminates / probe_budget_respected）、`conservative_ordering`、`chosen_cell`（最保守通过格）、`calibration_set_sha256`、`holdout_touched=false`、`holdout_tuning_forbidden=true`。选中值与 `config/defaults.yaml`（`g2.continuation.*`）由 `CHOSEN_BRANCH_POLICY`/`CHOSEN_CONTINUATION_KNOBS` 测试互锁（R6）。

## `pes2ts_pilot_replay_report_v1`（试点判定）

纯函数输出：逐例 `{verdict ∈ passed|failed|blocked, criteria{identity_conflicts, progressed_beyond_reference, origin_failure_typed}, evidence}` + 汇总。判据冻结于判定器（不可经 CLI 放宽）；blocked ≠ failed；passed 只证明工程可用性，不证明化学正确性（§11.5）。

## `pes2ts_g2t_feedback_report_v1`（第一份反馈报告）

内容清单 1–9 冻结（漏斗/typed 失败直方图/B1 对照/分支证据统计/候选与验证/成本总表/最小 J_G/AB3 改进清单/能力状态词表）。冻结校验：`m0_through_m5_claimed=false`；无 success_rate 字段；四类分母由终态机械计算，`rejected/censored/blocked` 不入成功分母；最小 `J_G` 无 verified 数据时显式 no-data 并注明完整口径属 G3.0 前置批次（§11.4）。

## `ORCAOriginPreparation`（起点自由优化 ACP 适配器）

生产 `evaluate_free`：ACP `BatchOptimize` `--profile opt`（role `minimum`）；fail-closed 能力探测失败 = typed `ORIGIN_BACKEND_UNAVAILABLE` + 需求清单（无本地回退，R3）；评估身份 `origin-<NNNN>/eval-<content16>` 内容寻址，同址异 binding = `IDENTITY_CONFLICT`；`landing_free_opt` 为独立 operation；产出 `endpoint_method_evidence_v1`（`minimum_status` 恒 unknown，未算曲率不得 verified）。
