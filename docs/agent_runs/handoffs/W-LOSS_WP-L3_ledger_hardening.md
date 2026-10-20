# W-LOSS · WP-L3 — Ledger Hardening（E0 残余关闭）

> 状态：**IMPLEMENTED + FIXTURE_PASSED** · 日期：2026-10-20 · 分支 `ops/wp-l3` · lane W-LOSS
> 关闭对象：E0_SIGNOFF_REPORT §5 前三项残余（formal_training_authorized 前置条件）
> 授权状态不变：`formal_training_authorized=false`（本 WP 只硬化账本，不开授权）

---

## 1. 变更摘要

| # | 残余 | 修复 | 位置 |
|---|---|---|---|
| 1 | ledger.append 不校验钉定 run_plan sha+mode（Oracle 探明：confirm flag 翻转 / partial 改 full 头 / 任意字节改动的 plan 副本仍可 append+launch） | **PLAN-SHA+MODE PINNING**：绑定 plan 的 ledger 在**每次** append/confirm_unseal/metric_unseal 前重算 plan sha256 + 重读 header campaign_mode，对 E0 钉定值（task_state.e0_freeze.run_plan_pinned / protocol_30h.json；编译常量 DEFAULT_PLAN_PIN = `d2a63599…3163` / `full`）比对；任何不符 → `PlanNotFrozenError`（`PLAN_SHA_MISMATCH` / `PLAN_MODE_MISMATCH`），fail-closed 零写入 | `scripts/agent_ops/ledger/run_ledger.py` |
| 2 | metric_unseal token 存在但消费未接线（B folds3/4 指标揭示是装饰性的） | **METRIC_UNSEAL WIRING**：plan 行 `substage=="B" && fold∈{3,4}` 的 `done` 转移**消费**一次性 token（batch=plan 的 B folds3/4 run 集，幂等 batch 语义同 confirm token），消费凭据写入 run record `metric_unseal` 字段；token 未配置 / 异 batch / 二次不同 batch → `UnsealExceeded`，零写入 | 同上 |
| 3 | supervisor/monitor 遇 PlanNotFrozenError 会崩 | **MONITOR CAPTURE**：新增 `supervisor/monitor.py`——`attempt_launch` 捕获全部 PlanNotFrozen 条件（pin 失效 / PENDING plan / stage 未绑定），改为在 run record 上标注 `blocked_with_reason`（状态不变）+ 审计事件 `blocked`，返回结构化 blocked 报告（exit 3）；非 plan 类账本错误（budget/retry/transition）**不吞**，照常抛出；`mark_blocked` 幂等，成功转移自动清除标注 | `scripts/agent_ops/supervisor/monitor.py` + ledger |
| 4 | 复用 driver 默认输出根指向 sealed 目录 | **assert_campaign_root 集成**：新增 `scripts/campaigns/run_v5_1_r3.py`（默认根 `outputs/pesn/30h_campaign/r3_runs`）与 `r3_launch.py`（E3 发射入口：campaign-root 断言 → plan pin → 可选 `--stage g0` 五 sha 绑定 → monitor 发射）；sealed 前缀 `v5_1/v5_1_r1` 与 campaign 外路径在打开任何输出文件前拒绝 | `scripts/campaigns/{run_v5_1_r3,r3_launch}.py` |
| 5 | stage-g0 五 sha 绑定无 supervisor 校验路径 | **STAGE-2 BINDING HELPER**：新增 `supervisor/stage_bind.py --stage g0|launch`——plan 行五 sha（config/data/teacher/features/split）必须全为真实 64-hex（拒绝 PENDING/zero/畸形），`StageBindingError` 是 `PlanNotFrozenError` 子类（monitor 自动捕获）；**未绑定任何真实 E1 SHA**（E1 落地后由 supervisor 重验） | `scripts/agent_ops/supervisor/stage_bind.py` |

**语义要点（fail-closed 冻结契约，未弱化任何拒绝）**

- **默认即钉定**：`RunLedger(plan_path=…)` 不显式传 pin → 默认绑定 campaign pin（task_state 派生）。fixture 计划必须显式 `plan_pin=None`（有 `PLAN_PIN_NO_PLAN`/`PLAN_PIN_INVALID` 构造期校验）。CLI `--plan-pin auto|none` + `--plan-pin-file`。
- pin 检查在 **append 路径最前部**（`_file_lock` 内、任何账本写入前），且**每次** append 都重验（不是只在注册时）。
- 监控捕获**仅限** PlanNotFrozenError 族；budget-cap 等其余 LedgerError 继续向上抛（回归测试证明）。
- 两阶段绑定语义不变：e0 冻结计划（PENDING token）注册合法、launch 仍被 `PLAN_NOT_FROZEN` 拒绝——pin 检查与 frozen 检查是**两道独立门**。

---

## 2. TEST MATRIX（path × positive/rejection × result）

解释器：`/tmp/opencode/venv/bin/python`（torch 环境）；全套 `tests/agent_ops/`：**405 passed / 0 failed**（合并前基线 349 + 本 WP 新增 56）。

| # | 路径（Oracle 探针回放 / 正向） | 类型 | 结果 | 测试 |
|---|---|---|---|---|
| 1a | **冻结 plan 副本翻转 confirm flag → append/launch**（Oracle 残余路径） | rejection | PASS `PLAN_SHA_MISMATCH`，零写入（ledger/events 均空） | `test_ledger_plan_pin.py::test_oracle_a_confirm_flag_flip_rejected` |
| 1b | **partial-mode plan 改 header 为 full → append**（Oracle 残余路径） | rejection | PASS `PLAN_SHA_MISMATCH`；sha 相符但 mode 谎报 → `PLAN_MODE_MISMATCH`（纵深防御） | `::test_oracle_b_partial_plan_reheadered_as_full_rejected` |
| 1c | **任意字节改动**（尾部加换行 / 单字符 decision_note 篡改） | rejection | PASS `PLAN_SHA_MISMATCH` | `::test_oracle_c_any_byte_change_rejected` |
| 1d | 注册后、launch 前篡改 plan → 下一次 append | rejection | PASS 重验捕获 `PLAN_SHA_MISMATCH`（每次 append 都重验） | `::test_oracle_tamper_between_appends_is_recaught` |
| 1e | **原始钉定 plan → append 成功**（正向）；launch 仍被两阶段 PENDING 门拒（`PLAN_NOT_FROZEN`，非 pin 错） | positive | PASS created+noop；launch 拒因正确归类 | `::test_pristine_pinned_plan_append_succeeds` |
| 1f | fixture plan 自 pin → launch 成功（正向全链路） | positive | PASS transitioned | `::test_synthetic_pinned_plan_launch_succeeds` |
| 1g | 默认 pin 绑定（不传 pin=campaign pin 拒 fixture plan）；显式 `plan_pin=None` 放行 fixture；无 plan 配 pin → 构造期拒 | rejection+positive | PASS | `::test_default_pin_binding_for_bound_plan` |
| 1h | confirm/metric unseal 路径也先验 pin（篡改后 unseal 拒绝且不碰 token） | rejection | PASS | `::test_confirm_and_metric_unseal_verify_pin` |
| 1i | CLI：默认 auto-pin 拒篡改 plan（exit 2 零写入）；pristine append 成功；`--plan-pin-file` 正确；`--plan-pin none` fixture 放行；坏 pin 文件拒 | rejection+positive | PASS | `::test_cli_*`（3 项） |
| 1j | 编译常量 == task_state.e0_freeze.run_plan_pinned == protocol_30h.json（防 pin 源漂移） | positive | PASS | `::test_pinned_constants_match_task_state_and_protocol` |
| 1k | pin 源损坏/缺失/畸形 → 拒绝 | rejection | PASS | `::test_load_campaign_pin_corrupt_refused`、`::test_verify_plan_pin_input_refusals` |
| 2a | **B fold3 `done` → 消费 metric_unseal**（count=1，batch=plan B folds3/4 六 run 集，凭据写入 run record `metric_unseal` 字段 + token 文件） | positive | PASS | `test_ledger_metric_unseal.py::test_first_b_folds34_done_consumes_token_and_records_it` |
| 2b | batch 内第二个 B fold4 `done` → 幂等（count 保持 1） | positive | PASS | `::test_batch_member_finalizes_idempotently` |
| 2c | 非 B-folds3/4（B fold1 / F-main）`done` → 不碰 token | positive | PASS | `::test_non_b_folds34_done_never_touches_token` |
| 2d | token 未配置 → B folds3/4 `done` 拒（`METRIC_UNSEAL_NOT_CONFIGURED`），无 done 行写入 | rejection | PASS | `::test_unconfigured_token_refuses_done_and_writes_nothing` |
| 2e | **二次不同 batch 消费** → `UNSEAL_BATCH_MISMATCH`，零 done 写入，token count 保持 1 | rejection | PASS | `::test_second_different_batch_refused` |
| 2f | token 文件损坏 → `UNSEAL_CORRUPT`，run 保持 running | rejection | PASS | `::test_corrupt_token_refuses_done` |
| 2g | pin 违规 vs metric 接线次序：篡改 plan 后 `done` → pin 错优先，**token 未被烧** | rejection | PASS | `::test_pin_violation_beats_metric_wiring` |
| 2h | `ledger.metric_unseal()` 显式路径同样默认 plan batch；批内幂等；批外拒 | positive+rejection | PASS | `::test_ledger_method_metric_unseal_uses_plan_batch` |
| 3a | **monitor 发射遇篡改 plan → 不崩**，`blocked_with_reason` 写入 run record（status 不变）+ `blocked` 审计事件 | rejection | PASS | `test_supervisor_monitor.py::test_attempt_launch_tampered_plan_blocked_not_crash` |
| 3b | monitor 发射遇 PENDING（e0）plan → blocked `PLAN_NOT_FROZEN`，未注册 run → `blocked_unregistered`（仍不崩） | rejection | PASS | `::test_attempt_launch_pending_plan_blocked`、`::test_attempt_launch_unregistered_run_blocked_unregistered` |
| 3c | 恢复 pristine plan 后再发射 → 成功且**清除** blocked 标注 | positive | PASS | `::test_attempt_launch_success_clears_annotation` |
| 3d | `--stage g0`：PENDING plan → blocked `STAGE_G0_NOT_BOUND`；全实 sha plan → running | rejection+positive | PASS | `::test_attempt_launch_stage_g0_binding` |
| 3e | **非 plan 类错误不吞**：budget-cap 照常抛 `BudgetCapExceeded`（监控捕获范围=PlanNotFrozen 族，契约回归） | rejection | PASS | `::test_non_plan_errors_still_propagate` |
| 3f | `monitor_scan` 只读报告 plan 漂移（plan_frozen False + 原因），不产生新账本行 | positive | PASS | `::test_monitor_scan_reports_freeze_and_blocks` |
| 3g | `mark_blocked` 幂等/换因重标/非法理由拒/未知 run 拒/状态不变 | rejection+positive | PASS | `::test_mark_blocked_idempotent_and_validation` |
| 3h | monitor CLI：blocked → **exit 3**（非 traceback），scan 报告漂移 | rejection | PASS | `::test_monitor_cli_blocked_exit_and_scan` |
| 4a | `run_v5_1_r3` 默认根 = `outputs/pesn/30h_campaign/r3_runs`；显式 campaign 路径（相对/绝对/./ 前缀）接受 | positive | PASS | `test_supervisor_drivers.py::test_default_output_root_is_campaign`、`::test_explicit_campaign_roots_accepted` |
| 4b | **sealed 前缀拒绝**：`outputs/pesn/v5_1/**`、`outputs/pesn/v5_1_r1/**`（含深层路径） | rejection | PASS（5 param） | `::test_sealed_roots_refused` |
| 4c | campaign 外 / 非法根（空、`..` 穿越到 sealed）拒绝 | rejection | PASS | `::test_outside_campaign_roots_refused`、`::test_invalid_roots_refused` |
| 4d | driver CLI：`--check` exit 0 打印根；sealed 根 exit 1 `SEALED_ROOT_REFUSED` | positive+rejection | PASS | `::test_cli_exit_codes` |
| 4e | `r3_launch`：sealed ledger 拒 / campaign 外 plan 拒 / 缺 record 参数 exit 2 | rejection | PASS | `::test_r3_launch_refuses_sealed_ledger` 等 3 项 |
| 4f | `r3_launch` 非钉定 plan → blocked `PLAN_SHA_MISMATCH` **exit 3**（非崩溃），`ledger_root` 已断言 | rejection | PASS | `::test_r3_launch_blocked_exit_code` |
| 5a | stage-g0：**全实 sha fixture plan → PASS**（g0 与 launch 两 stage，99 行全查） | positive | PASS | `test_supervisor_stage_bind.py::test_all_real_shas_pass_g0_and_launch` |
| 5b | e0 PENDING plan → `STAGE_G0_NOT_BOUND`（含 PENDING token 证据） | rejection | PASS | `::test_e0_pending_plan_refused_at_g0` |
| 5c | `--stage e0` 不能作发射门 → `STAGE_NOT_LAUNCHABLE` | rejection | PASS | `::test_stage_e0_is_not_a_launch_stage` |
| 5d | zero sha / 畸形 digest → 拒（任意 stage） | rejection | PASS | `::test_zero_sha_refused`、`::test_malformed_sha_refused` |
| 5e | 未知 run_id / 缺失 plan / 空 plan / 损坏 plan → 拒 | rejection | PASS | `::test_unknown_run_id_and_missing_plan_refused`、`::test_corrupt_plan_refused` |
| 5f | stage_bind CLI exit 0/1 + `--run-id` 单查 | positive+rejection | PASS | `::test_cli_exit_codes` |

**未绑定真实 E1 SHA**（任务约束）：所有五-sha fixture 使用合成 digest（`a1*32` 等）；E1 落地后 supervisor 用真实值重建 plan 并以 `--stage g0` 重验（届时 pin 需按冻结流程重新钉定——`--force` 重建 + 重新冻结 + 用户签认，代码内常量与 task_state 同步更新）。

---

## 3. 接口变化（下游注意）

1. **`RunLedger.__init__` 新参数 `plan_pin`**（默认 `CAMPAIGN_PIN` 哨兵=campaign pin）：既有 `RunLedger(plan_path=…)` 调用点若绑 fixture plan 必须显式 `plan_pin=None`（本 lane 测试已全部更新）。`plan_pin` 与 `plan_path` 不一致（有 pin 无 plan）→ 构造期 `PLAN_PIN_WITHOUT_PLAN`。
2. **B folds3/4 的 `done` 现在要求 `metric_unseal_path` 已配置**且 token 可消费——E3 正式训练前 supervisor 须放置 `state/gates/metric_unseal.json`（批次预登记同 confirm token 规则）。
3. **新 CLI 面**：`run_ledger --plan-pin auto|none --plan-pin-file F`；`python -m scripts.agent_ops.supervisor.monitor {scan,launch}`（launch blocked = exit 3）；`python -m scripts.agent_ops.supervisor.stage_bind --stage g0`；`scripts/campaigns/r3_launch.py`（E3 发射入口）、`run_v5_1_r3.py --check`。
4. run record 新字段：`metric_unseal{kind,batch_id,count,consumed_at}`（done 时）、`blocked_with_reason/blocked_source/blocked_at`（监控标注，成功转移清除）；events.jsonl 新事件 `blocked`。
5. `read_run_plan` 返回新增 `metric_run_ids`（B folds3/4 run 集）。

## 4. 下游依赖（supervisor 合并时注意）

- **E3 lane**：发射一律走 `r3_launch.py` / `monitor.attempt_launch`（不要再直接 `ledger.append(status=running)` 绕过 stage 门）；E1 后重建 plan → 重新冻结 pin（task_state + 协议 + 代码常量三处同步）→ `stage_bind --stage g0` 全 PASS 后才可发射。
- **supervisor**：B folds3/4 揭示前放置 metric_unseal token；monitor_scan 可入周期性巡检（exit 语义：scan 恒 0，launch blocked=3）。
- **授权链**：本 WP 不触碰任何 authorization flag；`formal_training_authorized` 仍待用户独立显式授权。

## 5. 验证记录

- `tests/agent_ops/` 全套：**405 passed**（`/tmp/opencode/venv/bin/python -m pytest tests/agent_ops/ -q`，exit 0，22s）。
- 真实钉定 plan 冒烟：`verify_plan_pin(outputs/pesn/30h_campaign/run_plan.jsonl, DEFAULT_PLAN_PIN)` PASS；`load_campaign_pin(task_state)` == 编译常量；pristine append=created，尾字节篡改=`PLAN_SHA_MISMATCH`（见 §2 1e/1j）。
- 范围核验：`git diff --name-only master..HEAD` ⊆ e0_write_allowlist lanes.W-LOSS（见 commit）。
