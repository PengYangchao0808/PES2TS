# PES2TS 文档修订台账（协同性）

## 2026-10-30 增量登记：V5.4.2 WP4 diagnostic ruling 收口（ADR-0033 v1.3.3；minimal path，阈值/预算数值不变）

- 背景：Oracle WP4 评审（`.slim/deepwork/v542-backbone-20261029/oracle-wp4-diagnostic.md`，绑定 HEAD `3acecc4` 证据）裁定 seed17 diagnostic 2,000 六行处置：**A0 是 matched 同剂量对照、不是候选学习筛对象**（harness「A0 included」为相对冻结 run matrix 的实现越权）；**A4 在冻结规则下 3.849742% < 5.0% = binding 淘汰**；5% 阈值与 comparator 冻结不变（改值即 post-hoc）。minimal path：只修 A0 control 角色 + 记录 A4 binding + 声明 initialization-proxy 混杂为设计局限，不重跑、不改阈值/预算/run 矩阵。
- **protocol amendment**：`outputs/pesn/v542_backbone_50k/plan/protocol_amendment_wp4_diagnostic.json`（`protocol_amendment_id=V542_WP4_DIAG_20261030`；**config SHA `a414edd0fa77…` → `6294a986c9cf…`**；`thresholds_changed=false`、`budget_values_changed=false`、`diagnostic_screen_rule_changed=false`、`a4_gate_skipped=true`、`a0_control_correction=true`；`signed_by=main_session_owner`、`signed_at=2026-10-30`；canonical `record_sha256` = identity `canonical_bytes` 配方）。config 本身不引用 record 路径（validator spec 无此要求）。
- **更正诊断报告（append-only）**：`outputs/pesn/v542_backbone_50k/diagnostics/diagnostic_report.v2.jsonl`（逐行 `supersedes_sha256` = 对应 v1 行原始字节（LF 除外）的 SHA-256 + `protocol_amendment_id`；A0 = `control_not_screened`/`a0_matched_control_not_screened`/`screen_role=control`/`screen_applied=false`/`failing_screens=[]`/learning-screen 数值仅 informational/`next_stage=T0_formal_mandatory`；A4 = `eliminated`/`gate_skipped`/`diag_best_inner_val_drop_lt_5pct_vs_initial`/`failing_screens=["best_inner_val_total_loss_drop"]`/`next_stage=none`；A1/A2/A3/A5 = `advance_full_dose`，next_stage T1/T1/T1/T2_conditional；六行 `run_outcome=done`、`rank_excluded=true`、`scientific_result=false`）+ `diagnostic_report_current.json`（命名 v1/v2 SHA 与 amendment id）。**不可变 v1 `diagnostic_report.jsonl` 与 per-run `run_records.jsonl`/`metrics.json` 逐字节未动**（`outputs/*` 在 .gitignore 内，`git status` 不显示；以 SHA-256 核验）。
- **config（additive）**：`diagnostic:` 增 `screen_subjects=[A1..A5]`/`a0_role=matched_dose_reference`/`a0_screen_role=control`/`a0_disposition=control_not_screened`/`a0_reason_code=a0_matched_control_not_screened`/`screen_applies_to_a0=false`/`control_validity_screens`/`disposition_vocabulary=[advance_full_dose, eliminated, control_not_screened]`/`disposition_note`；`thresholds.diagnostic_loss_drop_min_pct` 保持 5.0、28 签认阈值、budget 表、elimination_rules、26-run 矩阵全部未动。
- **validator**：新增 `diagnostic_disposition` check（**prereg checks 27 → 28**；**typed codes 保持 22** —— 复用 `V542_RUN_MATRIX`）：精确断言 screen_subjects（A0 出现即拒）、a0_role/a0_screen_role/a0_disposition/a0_reason_code、`screen_applies_to_a0` 严格布尔 false（`"false"`/`0` near-miss 拒绝）、control_validity_screens、disposition_vocabulary、elimination_rules 五规则逐项、threshold 5.0（漂移 typed 拒绝）、A0 诊断行 `role=matched_dose_reference`。
- **模板**：runplan_pin 增 `wp4_diagnostic_disposition`（`diagnostic_control`：`V542_DIAG_A0_S17` → `control_not_screened`/`a0_matched_control_not_screened`/`T0_formal_mandatory`；`diagnostic_gate_skipped`：`V542_A4_S17` → `diag_best_inner_val_drop_lt_5pct_vs_initial`、`new_training=0`、S29/S43 不激活；26 run_ids 不动）；preregistration_signoff 增 `wp4_diagnostic_disposition` + amendment note（`thresholds_changed=false`/`budget_values_changed=false`/`a4_gate_skipped=true`；28 签认阈值不动）；a0_reuse_decision 增 `wp4_diagnostic_control_note`（`decision=retrain_a0_x3` 不变）。
- **文档/registry**：ADR-0033 **v1.3.3**（§2.5/§2.6 逐字裁定句 + 状态行 + 变更记录行）；plan §7 diagnostic 条追加同一句 + §8.1 T-DIAG 行注记 `A0=control_not_screened`、`A4=gate_skipped`；registry `V5.4.2` gate_note 逐字追加 WP4 句 + stale tail「budget stays null in the committed config」→「budget signed 2026-10-30 from the same-env 40+40 profile (values unchanged by this ruling)」；`plan_revision: 2026-10-30.2`（`plan_revision_note` 前缀记录 amendment；G–V 生成文档重渲染，仅 plan_revision 行变化）；README 增 WP4 ruling bullet；`docs_consistency_config.json` 增 required tokens（ADR/plan/config/三模板：`control_not_screened`/`a0_matched_control_not_screened`/`screen_subjects`/`disposition_vocabulary`/`diagnostic_control`/`diagnostic_gate_skipped`/`wp4_diagnostic_disposition`/`wp4_diagnostic_control_note`/`thresholds_changed` 等）。
- **测试**：`tests/agent_ops/test_v542_config_schema.py` 新增 22 例（冻结字段正例；A0 `eliminated` 企图/`screen_applies_to_a0` 漂移与 near-miss/screen_subjects 漂移/reason code/vocabulary/elimination_rules/阈值 5.0→3.0 typed 拒绝；A0 行 control role；v2 报告 A0 control-not-screened 与 A4 gate_skipped 路径（3.849742% < 5.0）；A1/A2/A3/A5 advance 不变；v1 不可变 + 逐行 supersedes SHA；current.json 双 SHA；amendment 字段 + canonical digest；三模板镜像 + 26 run ids/28 阈值不动）；`tests/test_gv_registry.py` 1 例断言更新（`plan_revision == 2026-10-30.2`）。**schema tests 583 → 605，+20 registry = 625 total**。
- 验证：`v542_config_schema.py` prereg checks=**28** 绿（t1_blocked 保持 2 条 record instance 缺席）；`--formal` typed-block **`V542_RECORD_MISSING`**（不变）；`pytest tests/agent_ops/test_v542_config_schema.py tests/test_gv_registry.py -q` = **625 passed**；`docs_consistency.py --strict` CLEAN；`render_gv_map.py --check` current；`git diff --check` clean。
- 边界：未动 `pes2ts_core/**`、`scripts/agent_ops/v542_campaign.py`、`v542_identity.py`、`tests/pesn/**`；未动阈值/预算数值/26-run 矩阵/不可变 v1 诊断报告与 run 记录；无训练、无 GPU、无 QC；confirm/sealed 恒 false；未 commit。

## 2026-10-30 增量登记：V5.4.2 WP4 预算签认（同环境 40+40 profile）

- 背景：WP3 提交（dbae3f4）后按 gate-3 A4 纪律执行同环境 profile（cuda，40 W0 + 40 W1，20 update 预热剔除）。A0–A5 主表与 A1-C/A5-C conditional 两轮共 8 行写入 `outputs/pesn/v542_backbone_50k/profile/profile_report.jsonl`；`rank_excluded=true`、`scientific_result=false`。
- 实测 core-seconds（80 update window）：A0 1.919 / A1 1.953 / A2 2.041 / A3 1.840 / A4 1.896 / A5 1.763 / A1-C 1.337 / A5-C 1.303；相对 A0 的 core 比 1.018 / 1.064 / 0.959 / 0.988 / 0.919 / 0.697 / 0.679 —— 全部 ≤ `max_core_time_ratio=2.0`，无候选触发 `gate_skipped_cost_ratio_exceeds_2`。
- config `budget.per_candidate_core_time` 回填实测值（覆盖全部 8 个 backbone，含两个 full-dose 诊断变体）、`per_candidate_status=signed_wp4_profile`、新增 `budget.profile_signoff`（signed_by=main_session_owner、signed_at=2026-10-30、device=cuda、窗口、cost_ratio_vs_a0、max_core_time_ratio、all_within_ratio=true、分母=V542 A0 实测）。
- 状态变化：`--formal` 的 typed-block 由 `V542_BUDGET_NULL` 转为 `V542_RECORD_MISSING`（两条 governance record 实例仍未签认，按设计在 T1 前生成）；prereg `t1_blocked` 从 3 条降为 2 条（仅 record 实例）。gate-0 A1（budget formal gate）关闭点达成（值由 profile 签认，不再有默认值/占位值）。
- 测试同步：`tests/agent_ops/test_v542_config_schema.py` 7 条断言从「committed config 预算为 null」更新为「预算已签认 + 仅 record 阻断」；null 预算路径保留在内存副本测试。603 tests 全绿（583 schema + 20 registry）。
- 边界：仅预算/文档/测试状态同步；未改接口/阈值/计数/run 矩阵；未启动 smoke/formal/QC；confirm/sealed 恒 false；`new_QC_calls=0`；单路、swap=0。

## 2026-10-30 增量登记：V5.4.2 WP3 提交（runner/identity/resume + WP4 harness）

## 2026-10-30 增量登记：V5.4.2 P3 gate-3 remediation lane R2（B2/C2 record gate 镜像+SHA 核验 + C4/A1 budget formal gate 硬化）

- 背景：Oracle gate 3 = **REVISE**（`.slim/deepwork/v542-backbone-20261029/oracle-gate3.md`）——**B2**（A5 record gate 不镜像 `config_path`/`signed_by_roles`：`config_path="config/campaigns/WRONG.yaml"` 与 `signed_by_roles={}` 均通过 `--formal`）+ **C2**（`sha_rule: sha256_of_canonical_file_bytes` 声明了但从未计算/核验 record digest）+ **C4/A1**（budget formal gate：`per_candidate_core_time=[]`/全零/全负/NaN 通过 `--formal`）。本 lane 只改 config/validator/schema-tests/templates/docs/registry gate_note，**未动** `pes2ts_core/**`、`scripts/agent_ops/v542_campaign.py`、`v542_identity.py`、`tests/pesn/**`（并行车道 owns）、阈值（数值不变）、预算（保持 null）、`outputs/`；未 commit。
- **B2/C2 关闭**：`_check_governance_records` 强制两条机器核验 record 镜像冻结 config —— `config_path` = `config/campaigns/pes2ts_v542_backbone_50k.yaml`（新常量 `CAMPAIGN_CONFIG_REL`）、`campaign_id`/`gv_campaign_id`/`schema`（= 模板 schema id `pes2ts_v542_a0_reuse_decision_v1`/`pes2ts_v542_preregistration_signoff_v1`）匹配、A0 record `decision=retrain_a0_x3` + `runs` 镜像、signoff `signed_by_roles` 为精确冻结角色键集（`SIGNOFF_ROLE_KEYS` 10 键；`SIGNOFF_REQUIRED_ROLES` 绑定 statistics_owner/evaluation_signoff = `main_session_owner`、oracle_reviewer = `oracle_gate0_reviewer`；空 mapping/缺键/多键/绑定漂移/Oracle 出现任一 signatory 角色 = `V542_RECORD_MISMATCH`）、signoff thresholds 块与 config thresholds **canonical-JSON 字节相等**（逐键 `!=` 之外新增 byte-equality：4 vs 4.0 编码漂移也拒绝）。**SHA 核验（C2）**：每条 record 必填 `record_sha256`（入 `required_fields`）= sha256(canonical_bytes(实例去掉 `record_sha256` 字段))；配方**从 `pes2ts_core.pesn.models.v542.identity.canonical_bytes` 只读导入**（`_canonical_json_bytes`/`_record_canonical_sha256`，不复制配方，fail-closed on import error）；缺失 digest = `V542_RECORD_MISSING`，错误/stale digest = `V542_RECORD_MISMATCH`。**runplan_pin_record 澄清**：仍由 WP5 runner 在 runplan 冻结时核验，WP0 validator `--formal` 不读 pin 实例（validator verified_by + config records 注释 + 三模板 note 显式声明）。
- **C4/A1 关闭**：新增 `_check_budget_table` —— null/空（`null`/`{}`/`[]`）保持 `V542_BUDGET_NULL`；list/string/bool/非 mapping 表、部分键覆盖、多余候选键、每候选条目非有限正数（null/零/负/NaN/inf/bool）= 新增 typed **`V542_BUDGET_INVALID`**；prereg 不要求数值（畸形表 prereg 只记状态，formal 才 typed reject——数值检查 formal-only）。**A1 advisory 关闭**。
- **文档/模板同步**：config `governance.records` 块增 `schema`/`record_sha256` 入 required_fields + expected 镜像（config_path/campaign_id/schema）+ `signoff_roles: true` + `record_sha256_note`；`t1_start_requires_note` 与 budget 注释更新（`V542_BUDGET_INVALID` + runplan_pin WP5 runner 澄清）；a0_reuse_decision 与 preregistration_signoff 模板增 `record_sha256` 字段 + gate-3 note（canonical_bytes 配方、镜像要求）；runplan_pin 模板 note 增「WP0 validator 不读 pin 实例」澄清；signoff 模板 `t1_record_requirements` 增 config_mirror 说明；ADR-0033 **v1.3.2**（§2.4 增 `V542_BUDGET_INVALID` 行 + `V542_RECORD_MISSING`/`V542_RECORD_MISMATCH` 描述扩展 + §2.6 增 gate-3 收口 bullet + 状态行 + 变更记录行）；registry `V5.4.2` gate_note 追加 gate-3 B2/C2/C4 收口记录（`plan_revision` 保持 `2026-10-30.1`；G–V 生成文档不含 gate_note）；docs/README 增 P3 gate-3 lane R2 bullet；`docs_consistency_config.json` 增 required tokens（config：`V542_RECORD_MISMATCH`/`V542_BUDGET_INVALID`/`record_sha256`；ADR：`V542_BUDGET_INVALID`/`record_sha256`；两模板：`record_sha256`/`canonical_bytes`）。
- **测试**：`tests/agent_ops/test_v542_config_schema.py` 更新 8 例（`_write_valid_records` 产带镜像+`record_sha256` 的合法实例、`_mutate_record` 按需 re-seal、attribution budget 行改 `V542_BUDGET_INVALID`、code count 21 → 22）+ 新增 40 例（wrong config_path ×2 record、empty roles、wrong role set、wrong role binding、Oracle-as-signatory、schema drift ×2、missing/incorrect record SHA、tamper-after-signing、threshold encoding drift 4 vs 4.0、prereg blocker-not-rejection ×2、runplan_pin runner-verified、identity canonical recipe 一致、budget `[]`/`{}`/string/number/list/bool、all-null、zero/negative、NaN/inf、bool/string/null entries、partial、extra keys、bare-number accepted、prereg shape-not-required、correct table formal pass，parametrize 展开计）；**schema tests 543 → 583，+20 registry = 603 total**。
- 验证：`v542_config_schema.py` prereg checks=27 绿（t1_blocked = budget null + 两条 record instance 缺席，仅 `t1_blocked` 原因）；`--formal` typed-block `V542_BUDGET_NULL`（budget 在提交 config 中保持 null，为首道 formal 拒绝；budget 填充后 records typed-reject 由测试覆盖）；`pytest tests/agent_ops/test_v542_config_schema.py tests/test_gv_registry.py -q` = 603 passed；`docs_consistency.py --strict` CLEAN；`render_gv_map.py --check` current；`git diff --check` clean。

## 2026-10-30 增量登记：V5.4.2 P3 lane D（governance）— Oracle gate-0 A5/A2 收口（A5 T1 record gates 机器化 + A2 token-only 转 exact/structured；A1/A3 仍 open）

- 背景：Oracle gate-0 final（`.slim/deepwork/v542-backbone-20261029/oracle-gate0-final.md`）advisories **A5**（T1 record gates 未机器核验）与 **A2**（selection/statistics/H2/core-loop 仅 token-presence 检查，token-preserving 近邻可通过）+ gate-1 §2.2 复核（两条均未关闭）。本 lane 为 governance 收口：只改 config/validator/schema-tests/templates/docs/registry gate_note，**未动** `pes2ts_core/**`、`scripts/agent_ops/v542_campaign.py`、`v542_identity.py`、`tests/pesn/**`（并行车道 owns）、阈值（数值不变）、预算（保持 null）、`outputs/`；未 commit。
- **A5 关闭**：`governance.authorization.t1_start_requires` 增两条 record requirement —— `a0_reuse_decision_record` + `preregistration_signoff_record`（= 9 canonical gate IDs + 3 records；record requirement 非 gate alias，registry gates 不变）；新增 `governance.records` 块冻结三条 record 的 template/instance（campaign plan 目录下）/required_fields/SHA 规则（`sha256_of_canonical_file_bytes`）。validator 新增 `governance_records` check（checks 26 → 27）：config 镜像漂移 = `V542_RECORD_MISMATCH`；机器读取 A0 决策与 preregistration signoff 两个 instance —— prereg 缺失/不可读/None/内容漂移只记 `t1_blocked` reasons（WP0 缺席为设计内，prereg 保持绿：2 条 record + budget null 共 3 条 blocker）；`--formal` typed-reject —— 缺失/不可读/必填字段 None → **`V542_RECORD_MISSING`**，内容漂移（instance decision ≠ `retrain_a0_x3`、signoff 28 阈值与 config 漂移、runs 不镜像）→ **`V542_RECORD_MISMATCH`**（新 code 2 个：19 → 21）。runplan_pin_record 保持 runner 在 WP5 runplan freeze 时核验（records 块记录其 spec + verified_by）。
- **A2 关闭**：`final_selection_rule` 增 `eligible_condition=H3_QUALIFIED_ONLY`/`single_candidate_rule_id`/`no_candidate_rule_id` 结构化字段（exact）且 eligible/single/no-candidate prose 冻结为精确串；`statistics.h3_eligible_candidates` 自由字符串改结构化块（`criteria=[completed_three_seeds, per_seed_h2_no_breach, winner_eligible]` + `a5_rule=a5_only_when_three_seeds_complete` + `excluded=[A0_matched_baseline, diagnostic_variants_A1-C_A5-C]`）；`champion_rule` 改结构化块（`requires`/`selection_ref=evaluation_contract.final_selection_rule`/`report=uncorrected_and_holm_adjusted`）；`seed17_screen_pass_predicate.h2_no_breach` prose 精确冻结 + 新增 `h2_no_breach_thresholds` 结构化数值（sign 0.01 / dead 0.01 / rmse_zero 0.02 / fixed_coverage 0.02，镜像 thresholds.*；数值未改）；`a0_migration.core_loop` 改精确枚举 `super().forward_verbatim`（`"not verbatim"` 拒绝）。Oracle 探针表（gate-0-final §1 / gate-1 §2.2）的全部 token-preserving 近邻均新增对抗性测试并 typed 拒绝。
- **文档/模板同步**：config `t1_start_requires_note` + `records` 块 + A2 结构化字段；signoff 模板增 `t1_record_requirements` 镜像 + `t1_config_ready_requires` 两条 instance 要求；a0_reuse/runplan_pin 模板 note 记录 record 身份；runplan_pin `final_selection_rule` 镜像三个结构化字段；ADR-0033 **v1.3.1**（§2.4 增两 code 行 + §2.6 授权位清单与 record-gates/A2 收口段 + 变更记录行）；registry `V5.4.2` gate_note 追加同一记录（`plan_revision` 保持 `2026-10-30.1`；G–V 生成文档不含 gate_note）；docs/README 增 P3 lane D bullet；`docs_consistency_config.json` 增 required tokens（config：`a0_reuse_decision_record`/`preregistration_signoff_record`/`H3_QUALIFIED_ONLY`/`h2_no_breach_thresholds`/`super().forward_verbatim`/`V542_RECORD_MISSING`；ADR：+`V542_RECORD_MISMATCH`；模板相应增 token）。
- **测试**：`tests/agent_ops/test_v542_config_schema.py` 更新 18 例 + 新增 29 例（records spec/prereg-blocker/formal-typed/mismatch 对抗 + A2 六类近邻 + 结构化字段冻结断言）；**schema tests 496 → 543，+20 registry = 563 total**。
- **仍 open（不变）**：gate-0 advisory **A1**（budget formal gate；关闭点 P4 formal 前，`--formal` 在 records 之前仍 typed-block `V542_BUDGET_NULL`）与 **A3**（identity/SHA canonicalization recipes）；gate-0 A5/A2 自本 lane 起关闭。
- 验证：`v542_config_schema.py` prereg checks=27 绿（t1_blocked = budget null + 两条 record instance 缺席）；`--formal` typed-block `V542_BUDGET_NULL`（A1 保持首道 formal 拒绝；budget 填充后 records typed-reject 由测试覆盖）；`pytest tests/agent_ops/test_v542_config_schema.py tests/test_gv_registry.py -q` = 563 passed；`docs_consistency.py --strict` CLEAN；`render_gv_map.py --check` current；`git diff --check` clean。

## 2026-10-30 增量登记：V5.4.2 P2 lane（governance）— Oracle gate-2 Ruling 2 W_g/GRU bias 逐候选记录（A1/A2/A3/A5 gate-0 advisories 仍 open）

- 背景：Oracle gate 2 = **REVISE**（`.slim/deepwork/v542-backbone-20261029/oracle-gate2.md`）——Ruling 2 裁定 A2/A4 与 A5/A5-C 的 `W_g` bias 差异可保留但必须显式记录；plan §5.1 旧泛写「`W_g: 5→48`，288」对 A5/A5-C 不实。本 lane 为 governance 记录：只改 config/validator/schema-tests/docs/registry gate_note，**未动** `pes2ts_core/**`、`tests/pesn/**`（实现并行车道 owns）、阈值、预算（保持 null）、`outputs/`；未 commit。
- **事实（只读核实 `a2.py`/`a4.py`/`a5.py` 与 Oracle gate-2 实测）**：A2/A4 = biased `nn.Linear(5,48)` → 288 params；A5/A5-C = `nn.Linear(5,48,bias=False)` → 240 params/cell；A5 唯一 `GRUCell(48,48)` 带 bias（14,112）；A5-C 第一 cell 同 A5（14,112）、第二 `GRUCell(48,48,bias=False)` 无 bias（13,824）。冻结总量 A2 54,896 / A4 55,840 / A5 31,584 / A5-C 53,712 = 预注册不变量，精确复现，**未改任何计数**。
- **config**：`backbone_contract.candidates` 行增 `w_g_bias`（A2/A4 `true`；A5/A5-C `false`）与 `gru_cell_bias`（A5 `[true]`；A5-C `[true, false]`），backbone_contract 级增 `w_g_gru_bias_note` 汇总；现有字段全部保留。
- **validator**（`_check_backbone_registry` 内扩展现有 `backbone_registry` check，**无新 check 名 ⇒ checks 保持 26**、typed codes 保持 19）：`w_g_bias`/`gru_cell_bias` exact-match —— 缺失 → `V542_MISSING_FIELD`；值漂移（含 `"true"`/`"True"`/`1`/`0`/`1.0` 等 token-preserving near-miss、长度/顺序错位）→ `V542_PARAM_BUDGET`；无 `W_g`/无 GRU cell 的候选（A0/A1/A1-C/A3）携带字段 → `V542_PARAM_BUDGET`。
- **文档**：plan §5.1 泛写修正为逐候选 truth（A2/A4 288 biased；A5/A5-C 240 bias-free + GRUCell 14,112/13,824 + 冻结总量不变 + warm-start 一致）；ADR-0033 **v1.3**（§2.3 记录 bullet + §2.4 `V542_PARAM_BUDGET` 描述扩至 bias 漂移 + 变更记录行）；registry `V5.4.2` gate_note 追加同一记录（`plan_revision` 保持 `2026-10-30.1`；G–V 生成文档不含 gate_note，无需重渲染）；docs/README 增 P2 lane bullet；`docs_consistency_config.json` 增 required tokens（config：`w_g_bias`/`gru_cell_bias`；ADR：`w_g_bias`/`gru_cell_bias`/`Ruling 2`；plan：`Ruling 2`/`bias=False`）。
- **测试**：`tests/agent_ops/test_v542_config_schema.py` 新增 24 例（冻结值断言 + 逐候选漂移 + token near-miss 对抗 + 缺字段 + 无 W_g/无 GRU cell 候选携带字段 + A5 单 cell/A5-C 双 cell 顺序）；**472 → 496 schema tests，+20 registry = 516 total**。
- **仍 open（不变）**：gate-0 advisories **A1**（budget formal gate；关闭点 P4 formal 前）、**A2**（token-only selection/statistics/H2/core-loop；关闭点 P3/WP6 前）、**A3**（identity/SHA recipes）、**A5**（T1 record gates）——本 lane 不关闭任何一条；gate-2 BLOCKER B1/B2、conditions C1/C3/C4 与 gate-2 advisories A1–A5 归实现并行车道处理。
- 验证：`v542_config_schema.py` prereg checks=26 绿；`--formal` 仍 typed-block `V542_BUDGET_NULL`；`pytest tests/agent_ops/test_v542_config_schema.py tests/test_gv_registry.py -q` = 516 passed；`docs_consistency.py --strict` CLEAN；`render_gv_map.py --check` current；`git diff --check` clean。

## 2026-10-30 增量登记：V5.4.2 P1 lane A — Oracle gate-1 REVISE 修复（B1 + C1–C5 收口；A1/A2/A3/A5 仍 open）

- 背景：Oracle gate 1 = **REVISE**（`.slim/deepwork/v542-backbone-20261029/oracle-gate1.md`）：BLOCKER **B1**（`DirectMoveDecoder.a_ref` 可构造覆写 + 可事后变异 — 同一 run identity 下 MOVE 可被改 25%）+ 条件 **C1–C5**（证书未携带 strict-load 审计、部分运行可报 PASS、真实 batch 覆盖未机器验证、loader 消耗训练 RNG、A1/A2 关闭声明失实）。本轮为 WP1 有界修复：未改 legacy 模型源、`outputs/`、V5.4/V5.4.1 registry 条目、阈值、预算（保持 null）；未 commit。
- **B1 关闭**：`DirectMoveDecoder` 移除公共 `a_ref` 构造参数（传入即 `TypeError`）；`a_ref` 改只读 property（无 setter，赋值 `AttributeError`）；`forward` 前置断言 `_a_ref == 0.08`（typed `V542_A_REF_FROZEN`，property 覆盖/槽位篡改均拒绝）；`A0LegacyBackbone`/`V542Model` 构造时校验 decoder `a_ref == 0.08`（typed `V542_DECODER_A_REF`/`V542_DECODER_TYPE`）。测试覆盖：构造覆写、事后变异、wrong-a_ref decoder 组装拒绝、默认 `forward_move` 不变。
- **C1 关闭**：parity harness `build_pair()` 返回 loader 审计；`run_asset()` 逐资产落盘 `source_path`/`source_sha256`（实际核验 digest）/`mapping_table_sha256`（`LOCAL_PLACEHOLDER:` 本地摘要——canonical recipe 仍归 advisory A3）/`n_source_keys=43`/`n_mapped_targets=45`/`dual_write_aliases`/`target_key_set`/`torch_equal_verified` 计数/空的 missing/unexpected/shape/dtype；报告 schema `pes2ts_v542_a0_parity_report_v1` 与证书模板 `pes2ts_v542_parity_certificate_v1` 的关系在报告 `report_schema_note` 与模板 `load_audit.report_schema_note` 双向显式声明（逐字段映射，不静默分歧）；新增报告-模板一致性测试。
- **C2 关闭**：报告新增 `certificate_complete`/`only`/`skip_gpu`/`gpu_disposition`（pass/fail/inconclusive/skipped_by_flag）；证书 PASS 仅当 6/6 必需资产全跑 + CPU bitwise + evaluator pass + GPU disposition ∈ {pass, inconclusive}；`--asset`/`--skip-gpu` 一律打印 `DIAGNOSTIC (not a certificate)` 并退出非零。
- **C3 关闭（混合 a+b）**：8 个类别（start/pre_TS/post_TS/endpoint/forward/reverse/small_graph≤8 atoms/large_graph≥24 atoms）由 harness 确定性选择实际覆盖（冻结 8-chunk base + 按 sorted inner-val 顺序 first-hit 追加缺失类别；缺失 typed `V542_PARITY_COVERAGE` 失败），逐资产记录 per-category 计数+布尔；`isolated_atom`/`zero_in_degree_node` 经探针核实 50K inner-val 切片不存在（4.5 Å 空间截断下分子全连通）——从模板 coverage 声明中移出为 `measured_categories`（逐资产实测计数+布尔，可为零），模板 coverage_note 记录该裁决。
- **C4 关闭**：`a0_mapping_table()` 的 legacy 构造包进 `torch.random.fork_rng(devices=[])`；文档明示 `load_a0_strict`/`a0_mapping_table` 不得扰动训练 RNG 流；测试断言 torch CPU（+CUDA 若可用）全局 RNG 状态前后逐字节一致。
- **C5 审计修正**：**A1（budget formal gate）与 A2（token-only selection/statistics/H2/core-loop）未关闭**——保持 gate-0 open advisory：A1 关闭点 = P4 formal 前（`v542_config_schema.py` budget 表校验）；A2 关闭点 = P3/WP6 前（final-selection/h3_eligible/champion_rule/H2 sign/core_loop 转精确串）。修正三处 validator docstring（`_check_interface`/`_check_statistics`/`_check_seed17_screen`）不再宣称不存在的 exactness。**A3**（identity/SHA recipes）、**A5**（T1 record gates）同样仍 open，关闭点不变。
- **A6 措辞修正**：GPU 观测最大为 9.5367e-7（V541_E0_S43 endpoint_logit，Oracle 实测；仍在 atol 1e-6 硬上限内），此前车道消息「≤7.9e-7」措辞不准确——更正 claim 而非容差。
- **A7 关闭**：evaluator leg 新增 no-op 防护——要求双侧 `EVALUATOR_TOP_KEYS` 顶层键齐全且 `n_float_leaves > 0`，否则 typed fail（`evaluator NO-OP guard (gate-1 A7)`）。
- 验证：`pytest tests/pesn/test_v542_model.py tests/pesn/test_v542_a0_parity.py -q`；`pytest tests/agent_ops/test_v542_config_schema.py tests/test_gv_registry.py -q`；六资产 parity CLI 全量 6/6；单资产 `--skip-gpu` DIAGNOSTIC 非零退出；`v542_config_schema.py` prereg/`--formal`；`docs_consistency.py --strict`；`render_gv_map.py --check`；`git diff --check`。

## 2026-10-30 增量登记：V5.4.2 P1 lane B — Oracle gate-0 A6/A8 收口（provenance 双字段 + 身份层按缺陷类别 bump + w_v 审计例外）

- **A6 parity 溯源消歧**：config/parity 模板资产行新增 `run_updates`（37,900 V541 E0 / 15,400 V53MP C）与 `checkpoint_update`（payload 实际 best update：37000/36400/37800；15400/15300/15400），`training_updates` 保留为 **`= run_updates` 的单义别名**（P1 parity harness 仍读它）；schema 强制两字段为正整数、`run_updates` = tier 剂量、`checkpoint_update <= run_updates`；模板 source 措辞改为「best inner-val checkpoint from a 37,900-update run (checkpoint update …)」；新增只读 payload 测试（文件在则断言 `payload["update"] == checkpoint_update`，缺失则 clean skip）。
- **A8 审计/breach 措辞机器化**：`init_policy.mapping_audit` 冻结精确串（声明 `w_v` 双写例外，不再误判 double consumption）；`sanity_band_breach_procedure` step 3 与 `sanity_band_note` 冻结为**按缺陷类别 bump 对应身份层**（代码→`source_tree_fingerprint`、数据→`data_manifest_sha256`、初始化→`init_mapping_sha256`、anchor 牵连→`process_head_anchor_sha256`；旧 A0×3 记 `invalid_run`，禁止 exception）——ADR-0033 §2.2/§4、方案 §4.5/§7/§14、signoff/a0_reuse 模板、registry gate_note 同步为同一规则；validator 对这些串做 exact-match（非 token-presence）+ adversarial near-miss 测试。
- **映射键数澄清**：37 backbone 键 identity（含 `w_v` 双写例外）+ 6 process-head 键 identity = 联合 checkpoint **43 joint 键**（与 `pes2ts_core/pesn/models/v542/loader.py` 实际映射全部 43 键一致，只读核实）；backbone-only（R3 warm-start）映射保持 37 键。写入 config `init_mapping.note`/A0 行、ADR-0033 §2.2/§2.3、方案 §4.4/§5.2。
- 验证：prereg checks=26 绿；`--formal` 仍 typed-block `V542_BUDGET_NULL`（budget 保持 null）；schema+registry 测试全绿；docs gate CLEAN；`render_gv_map --check` current；`git diff --check` clean。边界：未动 `pes2ts_core/pesn/models/v542/**`、`v542_a0_parity.py`、`tests/pesn/test_v542_*`（另一车道）；未动阈值/预算/旧 V5.4/V5.4.1 条目/`outputs/`；无训练/GPU/QC；未 commit。

## 2026-10-30 增量登记：V5.4.2 WP0 fix-5 条件收口（Oracle PASS_WITH_CONDITIONS；ADR-0033 v1.2）

- 背景：Oracle gate 0 re-review **PASS_WITH_CONDITIONS**（`.slim/deepwork/v542-backbone-20261029/oracle-gate0-rereview.md`）——4 个 BLOCKER 已在 v1.1 解决，余 C1–C7 与 advisories。本轮为 **WP0-only 有界语义/模板/registry/docs 收口**（fix-5）：未改模型/训练器/runner，未改 `v542_config_schema.py`/tests（断言增量归 fix-6），未启动训练/GPU/QC，未访问 dev/confirm。
- **C1 最终 Backbone\* 选择冻结（`final_selection_rule`）**：候选资格=H3-qualified only（A0/diagnostic 不参加）；恰 1 个选它；0 个 A0 留任；多个按冻结字典序（MOVE-qualified 优先 PROCESS-only → qualifying endpoint 的 Holm-adjusted p 最小 → effect margin 最大（MOVE direction=min(rel/0.03,abs/0.01)、MOVE RMSE=rel/0.03、AUPRC=delta/0.02、q=delta/0.03）→ min core_time_ratio → min params → min inner-val loss → min candidate_id）；不用 dev；报全 qualified；完整 rule block 进 statistics/final-selection SHA 与 run identity；A5-C 仅用被选 candidate。落地：config `evaluation_contract.final_selection_rule` + plan §9/§10 + ADR-0033 §2.5 + runplan_pin + signoff 模板。
- **C2 route precedence 括号化（`route_logic`）**：MOVE pass = `(direction_pass AND rmse_non_inferior) OR (rmse_pass AND direction_non_inferior)`；PROCESS pass = `(auprc_pass AND q_non_inferior) OR (q_pass AND auprc_non_inferior)`；`seed17_screen_pass` 显式复用同一括号逻辑；消除「或」后条件悬空。落地：config `evaluation_contract.route_logic` + gates.H3.rule/seed17 point_effect 重写 + plan §8.1/§9 + ADR + signoff 模板。
- **C3 parity 资产分层修正**：V541 E0 S17/S29/S43 = `runs/` 37,900-update final **主证据**；V53MP C S17/S29/S43 = `outputs/pesn/v53mp/` S3 formal 20k-tier **15,400-update 辅助证据**（非 smoke、非 37,900、不在 `runs/`）；六个 required=true、精确 path+SHA 不变；config/parity 模板逐行增 `source_tier`/`training_updates`；删除「必须 runs/ 37,900 final」绝对措辞；smoke/非钉死禁止；GPU 1e-6/1e-5 硬上限、floor>cap ⇒ GPU parity inconclusive、CPU bitwise 决定、不放宽容差。schema error 文本由 fix-6 修。
- **C4 signoff 角色修正**：`statistics_owner`/`evaluation_signoff` → `main_session_owner`（非 Oracle）；新增 `oracle_reviewer: oracle_gate0_reviewer`（reviewer 非签认人）；parity/training/resource 保持 pending；`_template_note` 改「WP0 authoring complete; formal signoff pending owner re-affirm; reviewer not signatory」；「all 24 thresholds」→ 28。
- **C5 A0 sanity 顺序与硬门**：新 registry canonical gate **`V542-A0-sanity-band`** 入 `gv.gates`/`t1_start_requires`（+原 canonical gates/record）与 registry；`top2_freeze_requires` = A0_S17 done + A0×3 sanity band pass；band = A0×3 seed 均值 vs V541 E0×3 seed 均值；breach 严格阻断 T1 + 迁移审计（含 code/data/init）+ 缺陷修复后 bump `source_tree_fingerprint` 重跑 A0×3，**不得签 exception 继续**（移除旧 exception-continue 路径）；plan WP5 顺序重排为 T0 A0×3+sanity → T1/T2 formal → T3；smoke/profile/diagnostic 可先行。
- **C7/advisories**：README/台账 stale 清理（机器硬化最终实际 **426 schema tests + 20 registry tests = 446 total**，prereg checks=26，删除「166 例仅部分断言」等 stale 措辞）；ADR v1.2 / plan v1.2 / config `0.1.2-wp0`（日期 2026-10-30，旧历史不改）；`per_candidate_zero_init` 补 A1-C 第 3 层 W_d 与 A5-C 第二 cell init；smoke 增 `conditional_backbones:[A1-C,A5-C]`；runplan_pin 增 `bootstrap_seed_root:20261029`；registry `plan_revision: 2026-10-30.1` + note 更新（V5.4/V5.4.1 旧条目语义不变）+ `render_gv_map --write/--check`；A0 decision instance 仍按设计缺席（T1 前必须生成签认）。
- 验证 owner=主会话；`docs_consistency_config.json` required tokens 同步（`final_selection_rule`、`route_logic`、`V542-A0-sanity-band`、`source_tier`、`oracle_gate0_reviewer`、`main_session_owner` 等）。~~预期失败~~ **已由 fix-6 关闭（2026-10-30）**：`tests/test_gv_registry.py` 的 `plan_revision` 断言已更新为 `2026-10-30.1`，`test_v542_config_schema.py` 的 `T1_START_REQUIRES` 精确元组断言已同步；426 schema + 20 registry = 446 total 全绿。
- 边界：未改模型/训练器/runner/validator/tests/旧冻结文件/`outputs/` 历史记录；未启动训练/QC；未访问 dev/confirm；不宣称训练完成或候选科学收益。

## 2026-10-30 增量登记：V5.4.2 WP0 Oracle gate 0 REVISE 有界语义修复（ADR-0033 v1.1）

- 背景：Oracle gate 0 首轮 **REVISE**（`.slim/deepwork/v542-backbone-20261029/oracle-gate0.md`）——接口/A0 数学无异议，4 个 BLOCKER（A5 T3/max-4/N_core 算术；未定义 seed17 谓词；H3 Holm/p 值/bootstrap seed；A1-C init seed）+ conditions/advisories。本轮为 **WP0-only 有界治理语义修复**：未改模型/训练器/runner 代码，未启动训练/GPU/QC，未访问 dev/confirm。
- **[ADR-0033](../design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md) v1.1**：§2.2 parity 6 checkpoint 全 required（path+SHA 钉死，smoke 路径禁止）；§2.3 新增 seed recipe（sha256 → int mod 2147483647）、construction order、per-candidate R3 语义映射表、A1-C/A5-C `init_source_id` 覆盖、A2 V0/φ 零初始化、A4/A5 当前帧确定性几何量允许；§2.5 H2 恢复「退化」方向词 + H3 统计合同冻结（`bootstrap_seed_root=20261029`、确定性子 seed 公式、one-sided improvement p 值、Holm family 显式冻结、CI 与 p 同 draws）；§2.6 top2/A5 supplement 完全拆分（`top2_max_runs=4`、`a5_supplement_max_runs=2`、`t3_max_runs=6`；`seed17_screen_pass` predicate 冻结；A5-C 激活收窄为 `a5_selected_backbone`；`t1_start_requires` 改 canonical gate IDs + `runplan_pin_record`；identity 增列 source_manifest/bootstrap/anchor/mapping/plan/oracle SHA、`code_commit` 降为可选）；变更记录 v1.1。
- **[方案](PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md) 同步**：§0/§4.5/§5/§6/§7/§8/§9/§12/§14 逐节对齐（`seed17_screen_pass` 完整谓词；N_core 显式公式 + smoke/profile 不进 N_core；H3 统计合同全文；A0 sanity band 冻结审计程序；parity 六资产 path+SHA；`a0_reuse_decision` 实例为 T1 前置）。
- **[config](../../config/campaigns/pes2ts_v542_backbone_50k.yaml)**：`identity_contract` 增 `run_identity_required/optional`；`oracle_design_source_sha256=4f13bfbd…ccccb`；`t1_start_requires` canonical 化；`init_policy` 增 seed recipe/construction order/init_mapping/per-candidate zero-init；A1-C/A5-C `init_source_id`；parity assets 六项 path+SHA；`top2_preregistration` 收窄为 A1–A4（listed_runs 8）+ 新增 `a5_supplement` 块；A5 S17 `launch_requires=[diagnostic_gate,cost_gate,H0,H1]`；A5 T3 行条件/父门改 `a5_full_dose_complete_and_seed17_screen_pass`；A1-C/A5-C 行增 `activation_requires`/`skip_reason`；`statistics` 块冻结 seed root/子 seed 公式/p 值/Holm family/eligible 集；`seed_aggregation.min_seeds_same_direction` 改 numeric `{2, total_seeds: 3}`；`seed17_screen_pass_predicate` 块新增；H2/H3 gate 措辞修复；`a0_baseline` 区分 template 与 instance + sanity band 冻结程序；`thresholds`/signoff 增 `bootstrap_seed_root/top2_max_runs/a5_supplement_max_runs/t3_max_runs`。
- **五模板同步**：`parity_certificate`（6/6 required + 精确 path/sha + smoke 禁止）；`runplan_pin`（top2/A5 supplement 拆分、N_core 显式公式、statistics contract、`_t1_block_rule` 改 `runplan_sha256 + source_manifest_sha256`、nominal `counts_toward_winner` 改 per-run 判定）；`budget_policy`（`top2_max_runs/a5_supplement_max_runs/t3_max_runs`、A1-C/A5-C null 条目）；`preregistration_signoff`（统计合同、H2 退化、`seed17_screen_pass_rule`、A5-C rule、signoff roles 填充（authorization=`user_directive_20261029`、governance/statistics owner 记录；formal launch 仍需 owner re-affirm）、manifest SHA 层增列）；`a0_reuse_decision`（instance path + `instance_required_before_formal` + breach procedure）。
- **[registry](../../config/registry/g_v_map.yaml) V5.4.2**：`data_manifest` 改实际 `/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json`；`gate_note` 全量同步 v1.1 语义；`render_gv_map.py --write` 重渲染 G–V 文档。
- **docs gate**：`docs_consistency_config.json` 增补新 required 子串（`top2_max_runs`、`a5_supplement_max_runs`、`seed17_screen_pass`、`bootstrap_seed_root`、`a5_full_dose_complete_and_seed17_screen_pass`、parity 资产 SHA 等）；本台账与 docs/README 同步（「17 类 typed rejection」「满额 4」等 stale 措辞清除，计数改指 validator）。
- ~~已知暂态~~ **fix-4/fix-6 机器硬化车道已完成（2026-10-30）**：`v542_config_schema.py`/`test_v542_config_schema.py` 已对齐 v1.1+v1.2 精确合同（**426 schema tests + 20 registry tests = 446 total** 全绿；prereg checks=26；19 typed codes；28 thresholds；26-run 集精确）；v1.1/v1.2 语义字段（`top2_max_runs`、A5 supplement 父门、A5-C 条件、A1–A4 top2 集、`final_selection_rule`、括号化 `route_logic`、`V542-A0-sanity-band`）均已机器冻结；验证 owner=主会话。
- 边界：未改模型/训练器/runner/validator/tests/旧冻结文件/`outputs/` 历史记录；未启动训练/QC；未访问 dev/confirm；不宣称训练完成或候选科学收益。

## 2026-10-29 增量登记：V5.4.2 WP0 治理/预注册落地（ADR-0033）

- 新增 [ADR-0033 V5.4.2 Backbone 接口与预注册治理](../design/decisions/ADR-0033-V5.4.2-Backbone接口与预注册治理.md)（**Accepted**）：统一 `BackboneOutput(z[G,64], edge_msg[E,48])` + 唯一共享 `DirectMoveDecoder`（仅 `w_v` 49 params）+ 原样 `ReactionProcessHeadV53MP`（2,178）；A0 legacy 子类 adapter（hook 捕获 + 共享 `w_v` 同一 tensor + 显式映射 strict audit + CPU bitwise/GPU 1e-6/1e-5 parity + `q==order_score` + evaluator parity；active 55,392/joint 57,619 迁移不变量；禁 `torch.compile`）；记录用户 2026-10-29「请完成5.4.2阶段代码修改与训练工作」= implementation + gated formal training 授权及 Oracle 建议阈值签认为 frozen；明确授权**不免除** H0/H1/smoke/profile/runplan/预算门，confirm/sealed/新增 QC 恒 false/0，单路、无 swap；**默认且实际选择 `retrain_a0_x3`**（不实现 shadow-run/reuse 分支；V541 E0×3 仅 parity/历史参照，不作 H2/H3 对照）。
- 新增战役配置 [pes2ts_v542_backbone_50k.yaml](../../config/campaigns/pes2ts_v542_backbone_50k.yaml)（schema `pes2ts_v542_campaign_v1`，primary_g=G3.P.2，root `outputs/pesn/v542_backbone_50k`；v1.1 语义修复见下一小节）：候选 A1–A5（hidden48/z64/edge_msg48/node21；active ≤70k；R3 冻结 checkpoint 显式语义映射 warm-start + process-head `fork_rng` anchor + 候选 fresh seeded `fork_rng` + 训练前 reset seed + param-group 按 id）；A1-C/A5-C 仅诊断；smoke 200 ×A0–A5；同环境 profile ×A0–A5（分母 V542 A0 实测，禁用 V541 存档表）；seed17 diagnostic 2,000=400+1,600（A0–A5 全做，A0@2000 同剂量对照，只淘汰 H0/H1/NaN/不学习/超预算，不套 full-dose H3）；formal 37,900；A0 seeds17/29/43 必跑；top2 预注册条件 run（`activated_only_by_inner_val_top2`；v1.1 后 top2 只属 A1–A4、A5 行独立 supplement 分支，`N_core` 公式计数）；A5 full-dose 仅 A1–A4 无任何 H3 合格候选时自动激活；A1-C 激活条件 `a1_in_top2_supplement_set_OR_a1_selected_as_backbone`（**不是仅 top-ranked**）；H3 对照 matched A0、MOVE 两 exit 各 α=0.025 各自对 eligible candidates Holm、PROCESS 单一 family Holm（v1.1 统计合同细化）、seed-averaged curve-level paired bootstrap 10,000；outcome 六词表与旧 ledger 状态机分开、append-only。
- 新增独立 validator `scripts/agent_ops/v542_config_schema.py`（prereg/formal 双模式 + CLI）：schema/必填字段/objective、候选注册表、forward-view 白名单、接口契约（BackboneOutput 字段/decoder 49/PROCESS 2178/外部 5 键/torch.compile/GPU 容差）、参数不变量（A0 55,392/57,619、70k 上限、estimated 标记）、R3-only warm-start（禁 V541 final、SHA 逐字节核验）、`retrain_a0_x3` 决策钉死、confirm/sealed/QC/单路/swap、sampler 与 live `v541_strata` 镜像、smoke/profile/diagnostic 剂量块、`V542_GV_MISMATCH`、run matrix/`TOP2` 占位/激活分支（A1-C 非 top1-only、A5 诊断无条件）、outcome 词表/旧 ledger 不改、阈值 null/signoff 镜像——typed rejection 计数以 validator 模块为准；**`budget.per_candidate_core_time=null` 时 prereg 通过并记 `t1_blocked`，`--formal` typed block `V542_BUDGET_NULL`**（profile 前的合法 WP0 状态，runner 不得填默认值）。
- 新增 WP0 五模板 `config/campaigns/v542_templates/{preregistration_signoff,budget_policy,runplan_pin,parity_certificate,a0_reuse_decision}.template.json`：预算全 null 待同环境 profile；`retrain_a0_x3` 决策记录（reuse 分支未实现）；6-checkpoint parity 证书（CPU bitwise + 真实 batch + GPU 地板 + evaluator parity，明示不构成 E0 复用/训练可比证据）；run 三类分离 pin（diagnostic 不计主赢家）。
- 注册表 `config/registry/g_v_map.yaml` 新增 `V5.4.2`（status **`proposed`**——未写 executed/smoke_passed；primary G3.P.2；supporting G3.P.1/G3.P.3/G0/G1；12 个 `V542-` 门），**旧 `V5.4` 与 `V5.4.1` 条目不动**；`python scripts/render_gv_map.py --write` 重渲染 G–V 文档。
- 测试与门：新增 `tests/agent_ops/test_v542_config_schema.py`（初版 166 例，含两条冻结覆盖规则：A1-C 拒绝 top1-only 条件、A5 诊断必须无条件；**经 fix-4 + fix-6 机器硬化扩至 426 schema tests + 20 registry tests = 446 total**）；`docs_consistency_config.json` 增补 ADR-0033/方案/配置的 required-file 登记；docs/README 与本台账增量登记。
- 边界：本轮仅 WP0 治理产物；未改模型/训练器/评估器/旧 V5.3/V5.4/V5.4.1 冻结文件/`outputs/` 历史记录，未启动训练/QC，未访问 dev/confirm，不宣称训练完成或候选科学收益。

## 2026-10-29 增量登记：V5.4.1 执行完成收尾（只读最终报告 + 状态同步）

- 新增只读最终报告 [PES2TS_V541_Embedding_50K_实施与训练报告_20261029](../reports/PES2TS_V541_Embedding_50K_实施与训练报告_20261029.md)（Executed-Frozen）：WP0–WP7 全部完成；正式 run 与成本、失败/删失全账、dev/rollout/bundle 路径与证据边界逐项列出。核心事实：T0-R E0×3（17/29/43）与 E1/E2/E3 seed17 均 37,900 updates done；E1/E2 `scientific_failure`（H2 sign guard）、E3 `scientific_failure`（H2 sign+dead）；E4 `gate_skipped`（core ratio 3.6455>2.0 成本门，非资源失败）；E5（父门）/T3（`n_selected=0`，0 补 seed）/T4（`no_new_embedding_gain`，容量/深度混杂未闭合）均 `gate_skipped`；**E\*=E0 留任**（dev mean AUPRC 0.290702 / q_corr 0.397494 / RMSE-zero 0.910039；dev veto 不适用、无调参、tau 未重算）；rollout 1,152 episodes reference/autonomous 分开（endpoint arrival 均 0.005208、TS candidate 0.055556、QC=0、非关键原子位移 0）；E0 bundle 8 文件白名单+SHA+strict-load/read-only 审计通过；confirm 从未读取、`new_QC_calls=0`、无生产 No-Move/verified TS/部署资格声明。
- 保留的失败与修复记录（不删除、不改写）：loader WP5-CERT 单次 `EVIDENCE_MISSING`（§8.1 一轮补证不重试）→ `e0_reuse_decision.json` 判 `retrain_e0_x3`（sampler 不等价裁决保留）；rollout attempt-1 v1 light-cache loader SIGTERM 资源失败留账 + loader v2 实现修复重跑（名单/checkpoint/tau/预算不变）；formal 首次启动前 resource 配置 falsy-default 缺陷拒绝记录保留。
- [实施方案 v1.2](PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md)：状态 Active-Proposed → **Executed-Frozen**，新增 §19 执行指针（阶段状态映射、§16.1 出口逐项状态、与规划假设的偏离登记、V5.4.2 交接）；§0–§18 规划正文未改写。
- [ADR-0032](../design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md) v1.2：新增 §6 执行附录（决策项执行结果、§4 风险逐项裁决、E*=E0 最终裁决）；§4 原文保留并加 2026-10-29 指针注记。
- 注册表：`V5.4.1` `proposed` → `executed`，evidence 指向最终报告，code_scope 回填实际实现路径（`pes2ts_core/pesn/models/v541/**`、`offline/training/v541_*.py`、`pair_ownership_v3.py`、`key_atom_stream_view.py`、`scripts/campaigns/v541_*.py`），`plan_revision: 2026-10-29.1`；**旧 `V5.4` 条目未动**；G–V 映射文档已 `render_gv_map.py --write` 重渲染。同步 `tests/test_gv_registry.py` 的 V5.4.1 状态/plan_revision 断言（仅测试，不改科学内容）。
- 索引同步：`docs/README.md` 顶部新增执行完成小节（历史小节保留）；`AGENT.md` §10 与根 `README.md` 增加当前方向指针。
- 验证（2026-10-29）：全仓 pytest **4,954 passed / 1 skipped / 11 deselected**；`python scripts/agent_ops/docs_consistency.py --strict` CLEAN（145 文件 0 violation）；`python scripts/render_gv_map.py --check` current；`validate_gv_map(load_gv_map())` 在 executed 强制路径存在下通过。

## 2026-10-27 增量登记：V5.4.1 WP0 治理/预注册落地（ADR-0032）

- 新增 [ADR-0032 V5.4.1 Embedding 接口扩展与预注册治理](../design/decisions/ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md)（Accepted）：MoveProcessContract v1 任务语义不变，只扩展 Embedding 残差接口；记录用户 2026-10-27「完成5.4.1的代码生成和训练工作」指令 = implementation + formal training 按既有方案授权，同时将方案 §11.2 建议阈值签认为 frozen；明确该授权**不免除** H0/H1/单路 smoke/预算签认/formal SHA 冻结门，confirm/sealed/新增 QC 恒 false/0，单路、无 swap。
- 新增战役配置 [pes2ts_v541_embed_50k.yaml](../../config/campaigns/pes2ts_v541_embed_50k.yaml)（schema `pes2ts_v541_campaign_v1`，objective 仅 `unified_move_process`，不暴露三臂矩阵；embedding 冻结注册表 E0–E6、forward-view 白名单 17 键、core 只消费 legacy `edge_scalar[E,22]` 且 `node_vector_consumed/edge_adapter=false`）。
- 新增独立 validator `scripts/agent_ops/v541_config_schema.py`（prereg/formal 双模式 + CLI）：缺字段、非统一 objective、未知 embedding、非法 forward key、EdgeAdapter/vector 消费、confirm/sealed/QC 授权、单路/swap、`V541_GV_MISMATCH`、run matrix/命名空间越界均 typed rejection；`--formal` 下任一 §11.2 阈值为 null 或 signoff 与 config 不镜像即 `V541_THRESHOLD_NULL` 阻断 T1（prereg 模式记入 `t1_blocked`，runner 不填默认值）。
- 新增 WP0 模板 `config/campaigns/v541_templates/{preregistration_signoff,budget_policy,runplan_pin}.template.json`：run 三类分离（nominal/conditional/diagnostic，diagnostic 不计入主赢家）、37,900 固定剂量、W0/W1 分项测速、hard_ratio 成本规则、24⊂96⊂dev rollout 名单规则。
- 注册表 `config/registry/g_v_map.yaml` 新增 `V5.4.1`（primary `G3.P.2`；supporting `G3.P.1/G3.P.3/G0/G1`；`plan_revision: 2026-10-27.1`，`map_version` 保持 2026-10-18.1），**旧 `V5.4` from-any-point 条目保持不动**；`python scripts/render_gv_map.py --write` 重渲染 G–V 文档。run outcome 六词表（`done/invalid_run/scientific_failure/resource_failure/budget_censored/gate_skipped`）与旧 ledger 状态机分开登记，**未修改** `run_ledger.py` 状态机。
- 测试与门：新增 `tests/agent_ops/test_v541_config_schema.py`（54 例）、扩展 `tests/test_gv_registry.py`（V5.4.1 登记/V5.4 不变/typed rejection）；`docs_consistency_config.json` 增补 ADR-0032、V5.4.1 方案与配置的 required-file 登记。
- 边界：本轮仅 WP0 治理产物，未改模型/训练器/评估器/旧 V5.3 文件/`outputs/` 历史记录，未启动训练/QC，未访问 dev/confirm，不宣称训练完成或候选科学收益。

## 2026-10-27 增量登记：V5.4.1 WP0 run_matrix T3 预注册修正（复核响应）

- 删除 `V541_T3_TOP2_S29/S43` 两个 `embedding=null` 占位（无法表达 §9.1 的「两个冻结候选各补 seeds 29/43、满额 4 次」，且 run_id 无确定 embedding 身份）。
- 改为预注册全部可能入选的 T3 条件 run：**E1–E5 × seeds {29,43} = 10 个稳定 `V541_E#_S##`**，每个 `class=conditional`、`winner_eligible=true`、`condition=activated_only_by_inner_val_top2`、`new_training_if_activated=1`、`unactivated_status=gate_skipped`；仅 inner-val top-2 冻结后激活，**实际激活满额仅 4 个**；`N_core` 按 §9.1 公式计数，不按 listed runs 求和（config `run_matrix.t3_preregistration` + pin/signoff 模板同步声明 `n_core_counting: formula_only`）。
- **E5 门叠加**：E5 的 T3 行在选择门之上必须带 `parent_gate=parents_E1_E4_H0_H1_H2_pass_and_budget_approved`（两门皆过才激活）；E1–E4 行禁止携带 `parent_gate`，validator 均 typed rejection。
- validator 新增 `_check_t3_preregistration`（预注册集必须恰为 10 个 E1–E5×{29,43}、`listed_runs=10`、`full_activation_count=4`、`activation/unactivated_status/n_core_counting` 固定值），并在通用 run 检查中拒绝 `TOP2` 占位 token 与任何 `embedding=null` 的 run。
- **WP5 待裁决事实（已核实）记入 ADR-0032 §4**：旧 V53MP `_curve_batch_positions`（`key_atom_v53mp_train_v3.py:883-926`，v52 同源 `key_atom_v52_train_v3.py:424`）实际按 `(curve_id, direction)` 分池 8×4 采样，**无** `start/pre_TS/post_TS/endpoint` 分层实现，而 V53MP 与 V541 冻结 config 均声明该分层。不得静默把旧采样当等价；V541 runner 必须实现/证明配置所声明的共同 sampler，WP5 确认不成立则触发 §8.1 E0×3 重训分支。**未改** config `strata`、方案文件、训练剂量、阈值。
- 同步 `runplan_pin/preregistration_signoff` 模板、ADR-0032（v1.1 变更记录）、`docs_consistency_config.json` required 子串（`activated_only_by_inner_val_top2`、`full_activation_count: 4`、`_curve_batch_positions`）、docs/README 索引与本台账。

## 2026-10-27 增量登记：V5.4.1 零新增 QC 边界澄清 v1.1.1

- 明确 V5.4.1 的 50K 训练、E0 重评/条件重训、冒烟、离线评估与 rollout 均为既有数据/模型上的研究计算，`new_QC_calls=0`；不调用 ACP/xTB/ORCA/DFT，不新增能量/梯度、OptTS、频率或 IRC。
- `reference_normalized` 与 `autonomous` rollout 都固定为 no-physics/pure-geometry；autonomous 仅指固定预算下模型自主展开，不代表真实物理闭环。真实 OptTS/Freq/IRC 属后续单独授权范围，不是 V5.4.1 出口。
- 仅做边界澄清，主候选、训练剂量、run 矩阵和授权状态不变；docs 索引同步，未启动训练/QC、未读取 confirm。

## 2026-10-27 增量登记：V5.4.1 分子 GNN 调研细化 v1.1

- 按用户要求检索原论文与官方实现，将选型依据、F2/RBF mask 纠偏、E1–E5 最小规格、非零 gate 验收、梯度可达性、缓存及统计分组规则写回原方案；内容 v1.1，文件名保持 v1.0，docs 索引同步。
- 主矩阵/训练剂量/任务合同/授权状态保持；T4 建议优先 MLP 与深度对照，E5 结构提前至 T1 前预注册。未启动训练、未修改模型代码或读取 confirm；保留工作区原有改动。
- 验证：文档一致性检查扫描 144 文件、0 violation；方案章节/代码围栏/行尾空白检查通过。

## 2026-10-27 增量登记：V5.4.1 Embedding 50K 详细实施方案 v1.0

- 新增 [V5.4.1 Embedding 50K 详细阶段实施方案 v1.0](PES2TS_V5.4.1_Embedding_50K_详细阶段实施方案_v1.0_20261027.md)（Active-Proposed）：依据用户提供的 V5.4.1 单臂双功能头设计冻结候选 v1.1（来源文件 SHA256 `c2c1e5e5588ecd7109037095cd2035474722b366f953f1811cfcc0da56cfd24c`），细化 WP0–WP7、T0–T5、E0 可比性审计、单路 smoke/profiling、三 seed 配对评估、容量诊断和 Embedding 冻结交接。
- 归属 V5.4 / G3.P.2，评估与 bundle 交付延伸至 G3.P.3；不新增 G 阶段，不等待旧 V5.4 from-any-point 入口作为前置，后续 ADR/registry/config/schema 实施登记列入 WP0。
- 冲突裁决：历史 C×3 的修复后重评与是否重训分开；评估器修复只触发 re-evaluation，训练可比性失败或证据缺失才触发 matched E0×3。旧绝对 sign 不作正式基线，confirm 继续封存。
- 当前仓库核查：无 V5.4.1 模型/config/runner/tests；V5.3-MP 训练栈可复用，pair_index 评估缺陷未修，loader-v2 50K 训练路径位级证书仍为 SKIP，故 E0 复用状态为 `unknown`。
- 同步 `docs/README.md`；经独立读者复核后补明 dev 密封访问、R3 同口径参照、E0 成本口径、run 数公式和有界补证规则。`python scripts/agent_ops/docs_consistency.py --strict` 扫描 144 文件、0 violation；`git diff --check` 通过。
- 本轮仅新增规划与索引，不修改业务代码、历史报告、模型/数据、生产默认或授权位，不启动训练/QC，不访问 confirm。

## 2026-10-22 增量登记：V5.3-MP S1 2k 冒烟完成（S1 PASS）

- 新增报告：[V5.3-MP S1 2k 冒烟报告 20261022](../reports/PES2TS_V53MP_S1_2k冒烟报告_20261022.md)（write-once）——三臂 ok、Move guard 全过、GPU 1,829s≤3,600s 上限；ordering/endpoint 处冒烟冻结段的随机水平（与 V5.2-R D1 负结果一致）；λ 探针 endpoint calibrated / q target_unreachable / rank scale_unreachable（上限运行并记录）。
- 新增脚本：`r3_v53mp_orchestrate.py`（单 GPU 串行编排 + 锁等待 + 诚实失败记录）；`r3_v53mp_lambda_probe.py`（gradient-share λ 校准）。
- 修复（诚实记录）：driver 按臂清零 λ（arm A 曾因此失败一次，S1 attempt cap 3→4 按消耗数上调并注明）；`_FeatureLRU` 对 terminal/masked 行零填充 `y_teacher`（修混合批次丢键）；eval 的 MOVE 指标只在 ok 行上评估；`s1_eval_max_samples` 接线。
- 配置：`pes2ts_v53mp_20261021.yaml` 回填 λ 值 + verdicts；`s1_authorized`/`s2_authorized` 翻转（S0 全过 + S1 健康）；`joint_updates_scaling: density_preserving_per_tier`（2k/5k/20k = 1500/3750/15000 joint）。
- docs gate `--strict` CLEAN。

## 2026-10-22 增量登记：V5.3-MP S0 数据建库与资格（S0 全部门通过）

- 新增报告：[V5.3-MP S0 数据建库与资格报告 20261022](../reports/PES2TS_V53MP_S0_数据建库与资格报告_20261022.md)（write-once）——S0-A 元数据资格（142,221 合格 / 141,483 组）→ S0-B 授权提取建库（26,038 曲线 / 5,958,548 样本，经审计 truth bridge，逐反应记录）→ S0-C/E 标签（endpoint_label_spec_v1，有效正例覆盖 99.62%，TS/起点 0% 正例）→ S0-D 分片惰性加载（启动 695.8s / RSS 9.71GiB / 分片↔CSV 逐比特等价）→ S0-F 就绪 7/7 + 表校验 10/10。
- 20k 冻结池：`split_manifest_v53mp.json`（pool 20,000 嵌套 2k/5k/20k + dev 589 + confirm 2,000 仅列名封存）；cohort `v53mp_cohort_v1.json`。
- 新代码（本轮提交）：`r3_v53mp_s0a_qualify.py` / `r3_v53mp_freeze_cohort.py` / `r3_v53mp_build_tables_h5.py` / `r3_v53mp_finalize.py`；`irc_reader` 增 `energies`/`atomic_numbers` 别名读取；`validate_r3_tables.py` 增 `--split-manifest`；`key_atom_label_cache_v53mp.py` 标签缓存（绑定 data manifest sha）。
- 已知缺陷修复：`EndpointLabelTableV1.report()` 由 O(曲线×行) 改为单遍 O(行)（20k 规模下原形为病态二次）；`build_endpoint_labels_spec_v1` 加逐节点/逐 intent 缓存。
- 注册更新：`config/registry/g_v_map.yaml` 中 V5.3-MP `code_scope` 增补 S0 数据链四个脚本 + λ 探针；`status` 保持 `proposed`（S1 未授权）；S1 前置：λ gradient-share 探针回填 config → 翻转 `s1_authorized` → plan-init。
- docs gate `--strict` CLEAN；本轮含真实数据建库与真值访问（经审计），未启动 GPU 训练。

## 2026-10-21 增量登记：ADR-0031 MOVE–PROCESS 统一学习（V5.3-MP）

- 新增规范件：[ADR-0031 MOVE–PROCESS 统一学习合同](../design/decisions/ADR-0031-MOVE-PROCESS统一学习合同.md)（**Accepted**）、[MoveProcessContract v1](../contracts/PES2TS_MoveProcessContract_v1.md)（**Active-Normative**，MOVE/PROCESS/No-Move/TS 唯一任务合同）、[V5.3-MP 大规模联合训练计划 v1](PES2TS_V5.3-MP_大规模MOVE-PROCESS联合训练计划_v1_20261021.md)、[V5.3-MP 文档同步清单 v1](PES2TS_V5.3-MP_文档同步清单_v1_20261021.md)。
- **V5.2 / V5.2-R 关闭**：D2-G STOP（T1 不授权，P5 标签不冻结，GPU 消耗 0）；D0/D0.1/D1/D2 证据保留，旧负结果对旧任务继续有效。
- 取代关系：[FullPathNavigationContract v1](../contracts/PES2TS_FullPathNavigationContract_v1.md) 标 **SUPERSEDED by ADR-0031 / MoveProcessContract v1**；[ADR-0030](../design/decisions/ADR-0030-V5.2全路径导航与端点停止.md) 标 **Partially superseded**（任务纠偏结论继续有效，单一 `d_end` Endpoint Stop 语义降级为诊断量）。
- 宪法升 **v2.9**：新增 MOVE–PROCESS 增量；§6.7 加 v2.9 注记；术语表新增 MOVE/PROCESS/No-Move/MoveProcessContract v1。
- 命名区分：**V5.3-ENG** = 已执行的 30h rollout/relax 工程闭环（历史）；**V5.3-MP** = 本大规模 MOVE–PROCESS 战役。
- 注册更新：`config/registry/g_v_map.yaml` 中 `V5.2-R` → `archived`（D2-G STOP），旧 `V5.3` → `V5.3-ENG` `executed`，新增 `V5.3-MP` `proposed`（gates `V53MP-S0-data`/`S1-smoke`/`S2-scale`/`S3-formal`/`S3-closed-loop`）；`plan_revision: 2026-10-21.2`；G–V 映射文档由 `scripts/render_gv_map.py --write` 重新生成。
- 索引同步：根 README 新增 V5.3-MP 当前段；AGENT §10 指针置顶 V5.3-MP；docs 索引新增当前段并把 V5.2-R 标历史；合同/ADR/计划状态同步。
- 数据事实登记：当前六表仅 50 曲线/12,034 行；~200k 反应库尚未转 V5 六表；S0 是真实建库而非配置开关；S0 通过前 GPU=0。
- 本轮仅文档/注册/规范修订；未改业务训练代码、未启动训练/QC、未触 sealed（folds 3/4 继续封存）。docs gate `--strict` CLEAN。

## 2026-10-21 增量登记：ADR-0030 全路径导航与端点停止（V5.2-R）

- 新增规范件：[ADR-0030 V5.2 全路径导航与端点停止](../design/decisions/ADR-0030-V5.2全路径导航与端点停止.md)（**Accepted**）、[FullPathNavigationContract v1](../contracts/PES2TS_FullPathNavigationContract_v1.md)（**Active-Normative**，Move/Stop/TS/q 唯一任务合同）、[V5.2-R 全路径导航与端点停止修复方案 v1](PES2TS_V52R_全路径导航与端点停止修复方案_v1_20261021.md)、[V5.2 任务纠偏审计 20261021](../reports/PES2TS_V52_任务纠偏审计_20261021.md)（证据基础）。
- 合同升级：[ReactionIntent Contract v2](../contracts/PES2TS_ReactionIntent_Contract_v2.md)、[GeometryPolicyContract v2](../contracts/PES2TS_GeometryPolicyContract_v2.md)，条款级取代 v1 的 Stop/q 条款（v1 §4、§4.2/§5.2）。
- ADR-0021/0028/0029 加条款级取代横幅（Superseded / Partially superseded by ADR-0030）；[双端 TS 教师合同 v2](../contracts/PES2TS_双端TS教师与相容转移合同_v2.md) 标 **HISTORICAL**（branch 不跨 TS 语义不适用于 V5.2-R）。
- [30h R3→V5.2→V5.3 计划 v1.1](PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md) 标 **Executed-Frozen**（Stop 语义被 ADR-0030 取代，TSStop 章节仅历史，R3 语义仍有效）；[V5–V7 v3.0](PES2TS_V5-V7_反应意图条件化与低维生长方案_v3.0_20261020.md) 内容升 **v3.2（2026-10-21 任务纠偏修订）**。
- 根目录 5 份战役报告加 **POST-HOC 纠偏横幅**（证据不改写，仅加指针）。
- 宪法升 **v2.8**（2026-10-21 增量）：登记 ADR-0030/FullPathNavigationContract v1 为任务语义单一真相；§6.7"完整 A→TS 与 B→TS、不跨 TS"降为 V5.1a/V4 历史教师隔离语义。
- 历史方案补横幅：V5.0–V5.2 冲刺方案、V5.1-R2 方案、PES-N V5–V7 v1.2；[单端点 IRC 生成控制器 v3](PES2TS_单端点IRC生成控制器_产品设计_v3_20261020.md) 加任务语义修订注记（TS-only Growth/q≈0 停止不再是 V5 策略语义）。
- 索引同步：根 README（2026-10-21 task correction 段）、docs 索引（新 2026-10-21 顶节 + 合同/ADR/计划行状态）、AGENT §10 进度指针、[代码地图](../design/PES2TS_CODE_MAP_v2_20261018.md)（v2.1 注记 + v52r planned 模块）。
- 注册更新：`config/registry/g_v_map.yaml` 中 `V5.2` → `executed`（旧 TS-stop 任务负结果、冻结证据、evidence 保留），新增 `V5.2-R` proposed（gates：V5.2R-D0-labels / D1-endpoint-head / T1-joint / rollout），`V5.3` gate_note 补端点停止与长度感知预算 `B=max(40,ceil(1.5*L_key/0.08))`；`plan_revision: 2026-10-21.1`；G–V 映射文档由 `scripts/render_gv_map.py --write` 重新生成。
- 新增文档一致性门 `scripts/agent_ops/docs_consistency.py`（`--strict`，训练前强制）+ 测试。
- 本轮仅文档/注册/工具修订；未改业务训练代码、未启动训练/QC、未触 sealed（folds 3/4 继续封存）。

## 2026-10-20 增量登记：V5–V7 v2.0 架构修订与 V5.1-R2

- 新增 [V5–V7 三阶段研究与架构修订方案 v2.0](../plans/PES2TS_V5-V7_三阶段研究与架构修订方案_v2.0_20261020.md)：V5=Geometry Policy（仅关键原子位移 + 曲线停止位置）、V6=xTB 物理响应 + MLP/CNN 修正、V7=cost-aware RL。**取代** [v1.2](../plans/PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md) 的 V6 progress-decoder 路线；v1.2 保留为历史设计。
- 新增 [V5.1-R2 损失尺度与数据完整性修复方案](../plans/PES2TS_V5.1-R2_损失尺度与数据完整性修复方案_v1_20261020.md)：旧 V5.1 任务上的无量纲 strong Huber、τ 归一化 dead hinge、梯度占比 2%/10%、ramp/checkpoint 修复、S1 数据质量审计；验收是可靠结论，不要求晋级。
- 新增 [GeometryPolicyContract v1](../contracts/PES2TS_GeometryPolicyContract_v1.md)：冻结 `π(X_t,G) → (ΔX_K,m)`、K 定义、No-Move 曲线停止语义、在线禁止项与验证点。
- 新增 [ADR-0027](../design/decisions/ADR-0027-V5.1参考架fallback与双分母评估.md)（gauge fallback 研究门）、[ADR-0028](../design/decisions/ADR-0028-关键原子几何策略合同与V5-V7任务拆分.md)（合同/任务拆分）、[ADR-0029](../design/decisions/ADR-0029-V5.2+双门评估与封存边界.md)（监督门/物理搜索门，旧 G5.2–G5.4 保留）。
- 注册更新：`V5.1-R2`、`V5.2`、`V5.3`、`V5.4` proposed；`V5_SPRINT`、`V5_1A_SPRINT` executed；旧 `V6.0–V6.2`、`V7-A/B/C` archived。V5.1a-R1 的证据登记在 `outputs/pesn/v5_1_r1/` 与本台账，**暂不新增 executed 战役条目**（其代码/脚本尚未提交；提交并打 tag 后再注册）。G–V 映射文档已由 `scripts/render_gv_map.py --write` 同步；`map_version` 保持 2026-10-18.1（schema 不变），新增 `plan_revision: 2026-10-20.1`。
- 关键边界：V5.1a 已 `rejected_gate`；V5.1a-R1 无候选准入；**V5.1b 继续关闭**；不启动正式 OOF、Confirm/OOD/sealed、Fit150+。R2 不得自我批准 V5.1b；新 V5.2 不得靠改名绕过旧封存协议。
- 文档影响：根 README、AGENT、docs 索引、阶段定义、代码地图 v2.1、V5.1 方案、ADR-0026 均加修订/取代指针；历史报告、V5.1a/R1 冻结产物不改写。
- 本轮仅文档/注册修订；未修改业务代码、未启动训练/QC、未访问新真值。

## 2026-10-19 增量登记：V5 第一阶段冲刺执行完成（complete_oof_1seed）

- 按 [V5.0–V5.2 十小时方案 v1.1](PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md) 与 [ADR-0026](../design/decisions/ADR-0026-V5冲刺key-atom位移合同与OOF语义.md) 完成五阶段执行：prepare-audit（split 冻结/三跨度教师/冲突审计/特征/测速/G5.1）→ label-screen（六臂×1500 更新）→ recipe-recheck（Top-2×双折×三种子，A05/A12 双双稳定，胜者 A05=d0 0.05+anchor on）→ crossfit（测速阶梯降档 5fold×1seed 正式 OOF，3000 更新/折，`move_proposal_v5_draft1.parquet` 5088 行无删失，selection-exposed）→ report（封存 REFUSED 正确登记）。
- 关键实测：G5.1 pass；Oracle 静态上限 ORACLE_STATIC_OK（8/8）；A05 formal 开发级 G5.2 全部 8 项阈值通过（RMSE/zero 0.742、/classic 0.663、cos 0.841、nearTS 0.662、88.9% 优于零），verdict 保守 fail_or_insufficient 仅为 9 簇<20 簇样本量门；X0 起点记忆为主导因子（A 系全胜 C 系）；与 V4.7 负结果（MSE/zero 1.132、cos 0.180）形成对照。
- 执行中发现并修复的实现缺陷（均已提交+测试）：端点边集匹配需键级签名（仅键级变化反应）；参考块分级回退 + `native_verified_v1` 帧连续性分层（预注册阈值，实证 IRC 存档帧连续，eligible 11.7%→91.4%）；G5.4 pair 轴分段错误（key 轴混用致 Oracle 假阴性）；YAML on/off 布尔解析（报告复验 anchor 错配）；向量化损失（与循环参照 42 项等值测试）。
- 执行报告：[PES2TS_V5第一阶段冲刺执行报告_20261018](../reports/PES2TS_V5第一阶段冲刺执行报告_20261018.md)；机器产物 `outputs/pesn/v5/sprint/`（gate_report、label_recipe_matrix、recipe_recheck、ablation_table、timing_probe、OOF parquet、sealed_eval 等 20 件，含 sha256 链）。
- 未完成/边界（如实）：三种子 OOF 未做（测速降档）；Confirm/OOD 封存未开启；G5.2/G5.3 正式门不发 pass（样本量门）；扩量/粗到细仅计划（scale_curriculum_next_run.json）；sprint status = `complete_oof_1seed`，不宣称 V5.2 完成。

## 2026-10-19 增量登记：ADR-0026 V5 冲刺实现与战役注册

- 新增 [ADR-0026 V5 冲刺 key-atom 位移合同与 OOF 语义](../design/decisions/ADR-0026-V5冲刺key-atom位移合同与OOF语义.md)：落实冲刺方案 §0"实现前版本化"要求；key-atom 任务/帧合同/三跨度教师/fit50 内部冲突审计/5-fold split/特征 36-29-19/M2 Star EGNN/径向 Huber 损失/OOF 语义正式登记。
- `config/registry/g_v_map.yaml` 新增战役 `V5_SPRINT`（status proposed，Stage0 冻结 split 后升 active）；渲染文档已同步（`scripts/render_gv_map.py --write`）。新增 `config/campaigns/v5_sprint.yaml`（冲刺 §4 冻结块 + 编排段）。
- 登记偏差：无 torch_geometric/TensorDict 依赖，以纯 torch 张量 + ptr 实现同语义合同；元素 embedding 16 维在模型侧实现（节点总宽仍 36）；历史 342 冲突对不在本轮重扫（fit50 内部检索为准）；M1/M3/M4、pre_relax、xtb_freeze 系列留后续窗口。
- 本轮开始实现代码与运行五阶段（prepare-audit/label-screen/recipe-recheck/crossfit/report，600 分钟硬截止）；实现时间不计入训练窗口；能力状态按四层纪律单列，完成前不升级。

## 2026-10-18 增量登记：十小时冲刺内容v1.1，标签优先与粗到细

- 按用户确认修订 [V5.0–V5.2十小时方案](PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md)，保留原文件名与G–V映射；内容版本升v1.1。六输入/标签组合、短更新预算、稀疏轮换起点与密集验证替代本轮默认四架构赛马；Top-2为配方，跨划分/三seed复核，正式OOF为单胜出配方条件任务。
- 600分钟五阶段仍为90+120+150+150+90；原30run与额外架构/损失消融移至后续实验池。新增50→200→500扩量门、请求步距条件化及粗到细对照（同总预算、同学习率日程、全预训练成本计入）、独立复盘与结果资格。
- 仅修改计划/索引/台账，不改业务代码、机器可读G–V注册、现行生产默认或权限，不启动训练/QC，不发布新运行时接口。新增CLI/YAML字段仍是待实现草案，能力状态不升级。

## 2026-10-18 增量登记：G–V 双轴映射与 V5–V7 v1.2

- 新增 [ADR-0025 G–V 双轴映射与战役注册](../design/decisions/ADR-0025-G-V双轴映射与战役注册.md)：G = 代码/物理能力，V = 训练/测试战役；V5–V7 不是新 G 阶段。
- 新增 [G–V 映射表](../design/PES2TS_G_V_MAPPING_v1_20261018.md)（由 `config/registry/g_v_map.yaml` 生成）与 [代码地图 v2](../design/PES2TS_CODE_MAP_v2_20261018.md)。机器可读源为唯一真相，禁止手改生成文档。
- 登记 [V5–V7 架构计划 v1.2](PES2TS_PES-N_V5-V7架构调整与实施计划_v1.2_20261018.md)：Proposed；V5=G3.P.0–.3、V6=G3.2、V7-A/B=G4.1、V7-C=G4.0→G4.1、M0.5/M6.5=G2.T；G4.2 发布门不变。
- 登记 V5 冲刺与 v1.2 的差异：模型别名建议 `B0–B4`、工作包保留 `M0–M9`；`d0=0.08 Å` 覆盖 v1.2 的 `0.06 Å`；`λ_cos=0.02` 为 V5.0–V5.2 专项值；`cap=0.20 Å`，legacy `0.12 Å` 仅诊断。
- V4.2–V4.7 状态统一为 `executed / negative / frozen evidence`（10-18 实现报告）；V5 主结论不得用 V4 checkpoint 初始化。

## 2026-10-18 增量登记：V5–V7 终审与 V5.0–V5.2 冲刺提案

- 新增 [V5.0–V5.2 十小时冲刺训练与验证执行方案 v1.0](PES2TS_V5.0-V5.2_十小时冲刺训练与验证执行方案_v1_20261018.md)，归属 G3.P，数据资格关联 G0/G1；V7 真实执行仍归 G2/G2.T。
- 状态 Proposed：七模块设计与终审，不覆盖 V5–V7 v1.2、现行宪法、历史报告/数据；不发布可执行 schema，不修改业务代码、guard、默认或能力状态。本轮未运行训练/QC。
- 关键提案：当前输入因果处理与标签配准分离、退化参考隔离、O(3)极向量与局部架反射处理、strict-copy主门、选择暴露与OOF权重拟合隔离、跨折隐空间限制、600分钟测速降级。工程阈值为待预注册假设。
- 索引已同步 docs/README.md；具体新接口及适用范围在实现前按宪法§12进行ADR/版本化登记，本条不使草案生效为生产规范。

> 状态：权威（文档治理）
> 层级：工作流 C2 / 主干全局
> 用途：登记每份计划/设计/报告文档的权威归属、状态与修订动作，防止并行阶梯与"第三面"再生
> 取代关系：本文持续登记；2026-10-05 当前学习主线按 ADR-0010 更新，旧日期记录保留
> 上游依赖：[开发宪法](../PES2TS_开发宪法_v1.md)、[阶段定义](PES2TS_阶段定义_G0-G3.md)
> 下游消费者：文档索引、执行投影、各阶段开发与审计
> 当前版本：v3.25 @ 2026-10-29（V5.4.1 执行完成收尾：只读最终报告 + 方案 v1.2 Executed-Frozen + ADR-0032 §6 执行附录 + registry executed；v3.24 WP0 run_matrix T3 修正；旧 V5.3-MP 记录保留）

---

## 2026-10-17 增量登记：V4.2–V4.7 实验提案 v1.1 修订

修订 [V4.2–V4.7 方案](PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md)，依据用户建议及本地 `dual_egnn`、`dual_ts_policy`、预处理/特征/训练器源码与既有 run_record 复核：模型已有梯度向量种子、质量加权链式 Kabsch、梯度单位转换和曝光记录；现存每 5k 评估为 fit 诊断、检查点保存仍固定终点文件名。新增解析梯度基线/原生标签/来源及坐标硬门，V4.3 改为 Direct / 冻结物理 skip+残差 / 优胜者辅助监督，极坐标退出首夜；补充独立中间 checkpoint、重复曝光与饱和诊断。600min 重分配，V4.7 默认 20k，仅内部曲线与预算同时支持时升 40k；保留三种评测口径与旧门。同步 docs 索引；仅文档修订、状态 Proposed，未重跑历史训练、未产生新架构性能证据、不改代码/接口/宪法/报告/生产默认，不启动 QC。

## 2026-10-17 增量登记：V4.2–V4.7 约十小时实验提案

新增 [V4.2–V4.7 约十小时训练实验方案](PES2TS_V4.2-V4.7_十小时训练实验方案_20261017.md)，按用户要求结合 V4/V4.1 成果与 GNN 原始文献细化五组实验及 V4.7 三种子定稿。记录原生直连与取消头内 0.1 Å 上限的迁移对照、内部反应组开发划分、目标/稀疏关键对条件/向量传播/容量实验、基于实测速度的 600min 封套、新旧评测标签对齐和失败预算。纠正已有 RBF/changed_flag/残差被当作新增模块、插值毒性过度归因、叉积反射问题、对端构象资格及独立 seed 权重 soup 建议。状态 Proposed；本次仅新增方案并同步 docs 索引，不发布新 schema、不修改宪法/旧协议/阈值/报告/业务代码，不训练或新增 QC；实现批次另登记数据/动作/特征/bundle ADR 与验收。

## 2026-10-17 增量登记：v4.1 WP1 因果输入合同与特征 v3

新增[ADR-0022](../design/decisions/ADR-0022-v4.1因果输入合同与特征v3.md)：依据 v4 最终报告 P01，把教师标签派生的 `is_teacher_endpoint`/`near_zero_action_mask` 移出网络输入（node_scalar 7→5 观测-only），教师标签仅保留在监督目标/损失掩码/离线评测。特征数据合同升级 `pesn_dual_ts_features_v3`（旧 `pesn_dual_ts_features_v2` 保留注册以继续读旧产物），shard 重建到 `wp3_features_v41`/`wp4_monitor/features_v41`（旧目录只读不动），训练/自由运行/评测链路统一调用同一观测-only builder。新增两条验收测试（教师标签干预、离线/在线一致性）并跑通全 pesn 套件。不声明旧模型曾依赖通道 5/6、不重训、不重跑 monitor8、不改阈值（0.01 Å、monitor8 门、`dir_mask = valid & ~near_zero` 均不变）。docs/README 增补 ADR-0022 入口；不修改宪法/方案/合同正文或历史报告。

## 2026-10-17 增量登记：v4阶段最终报告与问题复核

新增[阶段最终报告](../reports/PES2TS_v4阶段训练_最终报告与问题复核_20261017.md)，按用户要求汇总WP0–WP5-prep及WP6/WP7状态、五臂冻结门数值、14类问题、成本范围和建议验收顺序。保留历史G1–G5裁决与模型/阈值原件；本次不启动训练、QC、扩量或新计划。

补充独立只读复核确认：教师下一步动作的近零掩码进入node_scalars并送入模型，自由运行改用上一步动作；全fit50 5967非终端态中4149态至少一原子掩码不同，证据 artifacts/v4_final_report/evidence_audit.json。因此报告收窄“数据/泄漏问题已排除”的旧结论，并登记错误NLL已修复、BC/IQL采样仍未闭合、多样性诊断非因果证明、主实验7318与额外成功冒烟1次及早期失败尝试需分账。docs/README新增事实入口；不修改宪法/方案/合同或历史报告。

## 2026-10-17 增量登记：v4.1 阶段最终报告与下一阶段决策包

新增[阶段最终报告](../reports/PES2TS_v4.1阶段训练_修复输入语义与最小实验矩阵_最终报告_20261017.md)：汇总 v4.1 输入语义修复（观测-only 5 通道、ADR-0022、两条验收测试）与可信实验基建（统一采样 60/60、协议链 r0→r1→r2、曝光账目、多种子门控、arm 元组绑定），并按冻结协议完成 A/B/C × 3 种子矩阵。注册门结果 REGISTER_ABLATION（0/3 种子×3 臂四项全不满足，非边缘差距）；B 臂 NOT EFFECTIVE（中位规则）、D 跳过；C 臂增益仅中位层面成立（逐种子仅 |a|-加权 cos 一致、pooled MSE 反转）；A-vs-v4 包回归成立但归因未定（输入×采样器未分离）；v4.1 checkpoint 实测幅度收缩持续且随臂异（A 0.074/0.154、B 0.217/0.217、C 0.144/0.181）；A45 种子散布内无实质变化。附三选项下一阶段决策包（包归因 2×2 → 扩数据 → 架构增强），明确不能区分「数据不足 vs 表达不足」最终归因。同步 docs/README 入口与 AGENT §10 进度指针；不改写任何历史门审、模型、阈值、prereg、协议或旧报告。

## 2026-10-15 增量登记：双端TS训练与整体项目进度统一

用户确认最终规格后，新增[统一方案v4](PES2TS_双端拓扑条件训练与项目进度统一方案_v4_20261015.md)、[ADR-0021](../design/decisions/ADR-0021-无序双端拓扑与完整双侧TS教师主线.md)、[合同v2草案](../contracts/PES2TS_双端TS教师与相容转移合同_v2.md)。明确无序完整双图、完整两侧向TS教师、逐点E/g、fit-only规范化、BC三配方、相容IQL、按门500/2k及三seed；SAC后置。建立旧证据/新任务矩阵、WP与G阶段映射、文件归属/依赖、资格/终止、重定价及历史资产规则，用户15节规格原文保留。

同步宪法v2.7 §6.7/§9.6（研究直接动作和有条件转移资格，R1–R6保持）、AGENT v3.15、阶段定义v4.5、整体发布v3.5、流程v2.4、两个README与旧训练/未来计划的取代指针。旧ADR-0020和训练v3追加当前取代说明；不改写其历史结论或报告。新接口只是设计登记，可执行validator/schema与模型能力仍待实现。文档整合完成不表示整个WP0通过；未动业务代码、运行配置、guard、数据/权重/报告，未启动训练或QC。

## 2026-10-15 增量登记：fit50 训练表征分析（BC/静态RL）

[表征分析报告](../reports/PES2TS_Direct_fit50训练表征分析_20261015.md)。BC 修复前运行发现 **σ 头硬钳位饱和缺陷**（994/994 节点超上限、梯度精确为 0、模型退化为 N(0,I) 先验平台 42.23）→ 修复为 `σ=σ_min+softplus(raw)` 并加回归测试；修复后 50/150 epoch NLL −66.7/−78.4 单调下降（每原子 log-density +4.99 vs 先验 +2.72、oracle +7.10）。**方向余弦 0.163（移动原子 0.129）**：NLL 在该动作尺度（u 半径均值 0.098）下方向可学信号仅 0.2%，为计划 §7.1 预登记的辅助方向损失版本提供依据（不静默改配方）。IQL 真实数据 3,000 更新：损失收敛、Q/V 有限、**AWR ESS≈均匀（255.9/256）→ 重加权无信号**、V-MC MAE 1.37 如实报告；训练器特征缓存修复 ~7× 提速。新增 `evaluate_direct_bc.py`、`diagnose_direct_bc.py`、`train_direct_iql.py`。全部为 fit50 表征（非正式配方/三 seed）；无新增 QC/真值访问；宪法/guard/生产默认未改。

## 2026-10-15 增量登记：Direct 批次 1–6 实施

实施计划v3.4 §11.11、[批次 1–6 实施记录](../reports/PES2TS_Direct批次1-6实施记录_20261015.md)。真实 RL 前业务代码落地：WP0 `pesn/contracts/direct.py`+`offline/splits/direct_freeze.py`（真实 split 冻结：fit 4,964 组/隔离 29/14/3/6 selection）；WP5a `generation/planning/direct_mdp.py`+`direct_environment.py`（奖励手算表/状态充分性）；WP1 `integration/acp/direct_observation.py`+`offline/data/direct_preprocess.py`+`direct_dataset.py`+`scripts/prepare_direct_data.py`（**fit50 50/50 完成：3,612 转移、3,662 QC、0 失败、3.51h 真实执行、验收 pass，无进展失败终端 shaping 语义修复并以零 QC refresh 重放**）；WP2 `features/direct_graph.py`+`models/{egnn,direct_policy,direct_critics}.py`（概率合同 float64 Jacobian 1e-6/等变/逆变换）；WP-INFRA1 `offline/training/{device,direct_loader,performance}.py`；WP3 `direct_bc.py`（完整 fit50 GPU 3-epoch 冒烟 NLL 42.2）；WP7a `generation/planning/direct_candidates.py`；WP4 `direct_iql.py`。全仓 2051 测试通过；truth guard 全绿（未扩 allowlist）。**校准信号**：goal 12/50、budget 21、no-progress 17；a_max=0.2 投影几何可达 54% 触发 §6.1 校准决策（待评审重建）；500 组定价 70.2h 串行/35.1h 二路/17.5h 四路（p90）。未完成：a_max 决策与重建、fit500/2k 正式数据、三 seed 正式训练、严格链、SAC。所有新能力按四层纪律标注；宪法/guard/历史产物未改。

## 2026-10-15 增量登记：GPU环境建立与P0/P1执行

实施计划v3.3 §11.10、[GPU环境与磁盘预算执行记录](../reports/PES2TS_GPU环境与磁盘预算执行记录_20261015.md)。按§11.9顺序执行：磁盘预算GO（E:26G可用，稳态余~18.5G/峰值~15G≥10G保留线；实测新环境5.8G、E:余量不变——VHD存在~25G已分配块复用）；冻结CPU环境依赖清单（SHA256落盘）。新建`pes2ts-gpu`（Python 3.12.15，torch **2.14.1+cu130**官方wheel，与CPU环境同核心版本；numpy/pandas/pyarrow/pyyaml对齐；`--no-cache-dir`）。P1验证通过：CUDA可用、4060Ti capability(8,9)、BF16支持、小张量matmul/autograd/optimizer与合成变长图批次前反向/优化步全在cuda:0且梯度有限；跨环境同seed冒烟数值一致。WP-INFRA0交付：`scripts/probe_direct_training_env.py`（只读探针+显式GPU代表批次子命令，缺CUDA明确失败）、`config/pesn/direct_infra_v1.yaml`（python_path/锁hash回填，严格loader属WP-INFRA1）、[GPU探针JSON](fixtures/pes2ts_gpu_env_probe_20261015.json)与双环境freeze。

未迁移磁盘/清理pip缓存/改`.wslconfig`/动CPU与ACP环境/改业务代码/运行训练QC；P2 device全链路依赖WP5a→WP1→WP2；加速倍数与并发对照未测量。图批次smoke吞吐不作为加速声明。

## 2026-10-15 增量登记：WSL GPU运行环境实查

实施计划v3.2 §11.9、ADR-0020 v1.3；AGENT v3.12及两个README同步。用户明确WSL主运行环境，查得Ubuntu24.04.2/WSL2、32线程、memory56GB/swap0；历史训练解释器torch2.14.1+cpu，但CUDA驱动cuInit成功且发现一张4060Ti。新增`scripts/probe_wsl_gpu.py`与[只读WSL证据](fixtures/pes2ts_wsl_gpu_probe_20261015.json)。E盘约25.7GiB空闲并承载WSL VHD，计划显式防止误读guest空闲；登记独立GPU环境、业务device迁移、9p缓存、主客内存及分阶段性能测量。未安装依赖、执行GPU模型计算/训练/QC、修改系统配置或迁移/清理文件。

## 2026-10-15 增量登记：多核/GPU基础设施规划

[实施训练计划v3.1](PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md)新增§11.3–11.8，[ADR-0020](../design/decisions/ADR-0020-EGNN直接位移与BC-IQL-SAC研究主线.md) v1.2。只读核查本机7950X 16物理核/32线程、4060Ti 8188MiB、当前可用内存约16.7GiB；两Windows解释器未发现torch，WSL访问与磁盘核查失败。[资源快照](fixtures/pes2ts_infra_probe_20261015.json)区分瞬时负载与未测历史训练。

新增单GPU learner、CPU loader/独立episode与ACP全局QC资源分工；safe_start 2×2线程、条件并发对照、轮屏障/幂等/恢复、数值与AMP验证、峰值内存/显存/磁盘及吞吐测量。追加WP-INFRA0–3和Gate-INFRA，回填预算后放行扩量。同步根README、AGENT v3.11和docs索引。本次仅改计划/指针和只读证据，未安装依赖、更改系统或执行配置、停止进程、修改业务训练代码/guard/宪法、访问真值或运行训练/QC；不声明已获得加速。

## 2026-10-15 增量登记：BC/IQL/SAC 直接位移研究方案

本日审核后进一步形成[实施与训练计划v3](PES2TS_EGNN_BC_IQL_SAC实施与训练计划_v3_20261015.md)，取代v2的下一步实施安排。明确2k独立组主训练、50/500小试、fit12双初始化SAC；登记文件级代码修改、前置共享推进器/严格候选桥、逻辑预算与删失、固定失败基础回报、无量纲每维熵、训练超参/实验矩阵/预算和验收。新增[候选元数据审计](fixtures/pes2ts_direct_training_input_audit_20261015.json)及`scripts/audit_direct_training_inputs.py`：旧5k为4,994组、17组历史开发替补交叉，剩4,977候选；14个组多族标签/3个成员缺有效族标签仍待正式闭合。核查仅涉及名单/分类元数据，未读IRC/TS几何、未训练/QC。ADR-0020更新为v1.1 Proposed，根README/AGENT/docs索引同步；宪法、guard、执行配置与历史报告保持。

本日后续整合为[EGNN完整训练方案v2](PES2TS_EGNN_BC_IQL_SAC下一步完整训练方案_v2_20261015.md)，登记[ADR-0020草案](../design/decisions/ADR-0020-EGNN直接位移与BC-IQL-SAC研究主线.md)。用户确认EGNN主模型；v2承接v1并细化概率动作、主线数据规模、fit/monitor/dev/sealed用途、奖励、训练配置、critic校准、真实费用、小试/对照与WP0–7。更新根README、AGENT和索引的下一步研究指针；旧报告/模型/配置保持。全部新增WP仍planned，不改变宪法、真值guard、生产默认或已验收能力。

新增 [IRC→BC→IQL→SAC直接位移研究方案](PES2TS_IRC_BC_IQL_SAC直接位移研究方案_v1_20261015.md)，状态 Proposed，承接用户明确的 MDP、IRC 邻帧、直接原子位移与三阶段学习方向。设计第一版纯位移/单点观测环境，使专家与真实交互共享动作/转移语义；横向松弛留作独立环境。细化专家制备、奖励与终态、等变概率密度、IQL expectile/优势加权、SAC soft critic 校准、replay及冻结 iid/family严格评测。WP0 在实现前登记新接口/真值用途边界；本次仅新增计划并同步索引，未变规范、默认、能力、业务代码或启动训练/QC。

## 2026-10-09 增量登记：自适应路径生长研究提案

新增 [RL 自适应路径生长与 IRC 验证研究方案 v1](PES2TS_RL自适应路径生长与IRC验证研究方案_v1_20261009.md)，状态 Proposed，归属现有 G2.T/X 与 G3.2/G4。依据首批 47 个训练 episode、开发 46 组三组对照与 156 配对分支，提出解除固定 lambda/driver 同步推进限制及新 on-policy 探索入口，保留物理/身份/预算与真值隔离边界。

提案区分自主轨迹、候选路径与严格 TS/IRC；包含真实梯度横向松弛、fit 配对监督、四组消融、22小时24分/11040请求的小包建议以及额外严格验证/训练全成本。新接口与默认变化在实施前沿既有架构流程登记。同步 docs 索引；本次未修改业务代码、历史报告/模型或启动计算，不将方案记为能力验收。

## 1. 治理原则

1. **单一主干编号**：程序进度只用 `G0/G1/G2/G3/G4`；工作流用 `C/R/X`。禁止再发明新字母阶梯。（G4 于 2026-10-03 新增，见 §9。）
2. **一文档一职责**：每份文档只对一个领域权威；其余引用它，不重复。
3. **状态头强制**：每份 plan/design 顶部加状态块（模板见 §4）。
4. **取代登记**：被取代的文档不删，加"已并入 X"注记。
5. **报告只读**：报告是历史证据，标"当前/作废"，永不改内容。
6. **一切新努力先登记主干**：新工作必须声明属于 G# 或 C/R/X，否则不立项。

---

## 2. 权威文档（2026-10-08 同步RL纠错与下一轮实施指针）

| 文档 | 权威领域 |
|---|---|
| `PES2TS_开发宪法_v1.md`（docs 根） | **规范根（SSOT）**：身份、红线 R1–R6、政策 P1–P5、接口注册表、阶段/命名、评测与修订；冲突以本文件为准 |
| `plans/PES2TS_阶段定义_G0-G3.md` | 阶段与边界定义（唯一来源） |
| `plans/PES2TS_整体开发与发布方案_v2.md` | 项目级开发与发布主方案 |
| `plans/PES2TS_PES-N未来开发计划_v2_20261005.md` | 当前未来路线的算法/数据/模型实施细化；规范服从宪法，阶段服从阶段定义 |
| `plans/PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md` | 统一设计与A/B/C fixture基线：共享模型/残差/Q、真实RL优先、G2单步与AB4独立门；真实round0及对照资格限制见Demo24与下一轮纠错方案，修复/收益待实施 |
| `plans/PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md` | **当前扩量执行细化v1.0，planned**：审核补修、独立反应计数、IRC20k/新RL2000嵌套批次、跨反应支持、全成本预算和封存评测 |
| `design/decisions/ADR-0019-IRC20k与RL2000分阶段扩量.md` | Proposed规划：承接ADR-0018并局部更新规模/预算；新接口、能力和训练另验收 |
| `plans/PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md` | **上一轮纠错算法细化v1.1**：第一批BC模式修复与纠错模型证据成立；四项审核缺口未关闭；扩量由新方案承接 |
| `plans/PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md` | round0已运行；258转移/51探索/369模型非零响应证据保留。2026-10-08审查发现BC实际CQL、同权重与Q标签门限制；不能作纯BC/CQL或非零Q收益证据；新一轮由纠错方案承接 |
| `design/decisions/ADR-0018-RL纠错与动作支持研究计划.md` | Proposed规划登记：完整状态/动作支持、成对监督与新合同草案；正式schema注册和能力验收待实施 |
| `design/decisions/ADR-0017-统一残差决策与G2有限回退.md` | Accepted规划：局部取代ADR-0015近期顺序，承接ADR-0016回退门；新增合同仅草案 |
| `plans/PES2TS_轻量控制与RL衔接计划_v1_20261007.md` | 研究入口和历史v1合同保留；近期实施顺序及合同扩展服从ADR-0017/统一方案，正式门引用权威根 |
| `plans/PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md` | 当前G2优化器增量：4点成对试验、ACP硬约束/物理单点与共享预算有限回退；设计完成，未实施 |
| `design/decisions/ADR-0016-ORCA驱动xTB硬约束优化试点.md` | Accepted：保留原生基线，外部xTB/ORCA备选为新配方；请求/收据/混合结果草案，未升级能力 |
| `design/decisions/ADR-0015-G2稳定底座与轻量RL研究入口.md` | Accepted：研究原型入口与正式能力门分开，沿已有执行器/产品归属；R1–R6和正式出口保持 |
| `plans/PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md` | 当前第一期训练实施细化：G3.P/G3-P1、版本、数据/标签/模型/科研门；阶段与规范分别引用权威根 |
| `design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md` | 当前G3-P1产品/文件/接口/测试实施范围：pesn单一业务根与runtime/offline隔离 |
| `design/decisions/ADR-0014-预测器产品归属与G3代码范围.md` | Accepted：部分取代ADR-0013目录安排，保持训练版本与正式门；模块/schema/guard待实现 |
| `contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md` | 已登记教师/run/bundle接口草案；正式schema/guard/validator待实现验收 |
| `design/decisions/ADR-0013-G3并行IRC预训练与模型版本体系.md` | Accepted：G3.P与G2并行、版本/隔离边界，保留正式决策/闭环门 |
| `design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md` | Accepted：PES-N 方向及规范同步，未声明实现或生产切换 |
| `design/decisions/ADR-0011-三模块产品架构与统一可视化.md` | Accepted：三模块与横向可视化产品方向，schema/UI/能力另验收 |
| `design/PES2TS_三模块与可视化产品架构_v1_20261005.md` | 当前产品架构细化：组件复用、弱输入、控制/执行/UI责任与分期验收；规范服从宪法 |
| `design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md` | 当前Classic算法设计：关键原子初猜、ACP/xTB逐点优化、合同与科学边界；实现待验收 |
| `plans/PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md` | 当前G2-AB3实施设计：功能整合、跨仓缺口与文件级工作包/验收 |
| `design/decisions/ADR-0012-Classic图论驱动与xTB原生约束扫描.md` | Accepted设计方向；具体schema为草案，生产能力和默认不升级 |
| `design/decisions/ADR-0001-统一执行后端接缝.md` | 执行层统一决策（部分条款被 ADR-0002 取代） |
| `design/decisions/ADR-0002-计算后端统一经ACP执行.md` | 计算后端统一经 ACP 执行（部分取代 ADR-0001） |
| `plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md` | G2.4 执行统一实施计划（X1′–X5′） |
| `plans/PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md` | 当前 G2 工作包与实验设计；阶段定义仍以阶段权威文档为准 |
| `plans/PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md` | ACP 跨仓能力缺口、请求/产品草案与联合验收 |
| `plans/PES2TS_G2_PESG收紧算法迭代方案_20261003.md` | G2 数学与算法细化；不替代阶段定义 |
| `plans/PES2TS_G1_G2边界审计与补齐任务_20261003.md` | 本次边界抽样审计与补齐任务；实际边界定义仍在阶段文档 §3 |
| `plans/PES2TS_G2_下一批算法调整与测试方案_20261003.md` | 下一批 G2 算法核心 + 测试批次方案（批次号 G2-AB1：分支稳定与验证接通）；实施状态以能力证据为准 |
| `plans/PES2TS_完整开发流方案_整合版_v1.md` | 流程总纲（G0–G4 阶段流、依赖图、里程碑、文档地图）；指向各阶段方案 |
| `AGENT.md`（仓库根） | **宪法执行投影**：红线 ID 摘要、操作性规则、架构变更审查协议（非独立规范；分歧以宪法为准） |
| `plans/PES2TS_文档修订台账.md` | 本文档治理 |
| `plans/PES2TS_计数与标定台账.md` | config `TODO: calibrate`（待建） |

---

## 3. 文档修订动作表

### 3.1 plans/

| 文档 | 新状态 | 修订动作 |
|---|---|---|
| [IRC20k与RL2000详细方案](PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md) | 当前扩量v1.0；planned | 2026-10-08新增：独立组计数、审核补修/回放/跨反应支持、两条扩量线、配方/迁移/指标/完整预算/封存评测；新增规划JSON与文档校验 |
| [PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md](PES2TS_RL下一轮代码修复训练与调整方案_v1_20261008.md) | 上一轮算法细化v1.1；部分实施，审核未全通过 | 2026-10-08新增第一批证据/四项缺口与新扩量承接说明；原规划正文和历史产物保留 |
| [PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md](PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md) | round0历史实施与纠错指针 | 2026-10-08增加当前审查说明；旧产物/报告不改写，纯BC对照声明按新证据限定 |
| [PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md](PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md) | 当前第一期训练方案v1.1；设计完成 | 2026-10-07细化预测器B0–B7文件/接口/依赖/验收；G3.P及模型身份保持；未训练 |
| [PES2TS_PES-N未来开发计划_v2_20261005.md](PES2TS_PES-N未来开发计划_v2_20261005.md) | 当前未来开发计划 | 2026-10-05 同步规范；STOP监督→残差→离线Q→冻结闭环；新增合同草案/系统验收，代码不变 |
| [PES2TS_G0-G4整体开发框架修改方案_v1_20261005.md](PES2TS_G0-G4整体开发框架修改方案_v1_20261005.md) | 历史提案，已被 PES-N 取代 | 保留原文；当前模型主线及阶段安排改由 PES-N 计划与 ADR-0010 定义 |
| [PES2TS_拓扑约束TS接近度与位移联合学习方案_v2_20261004.md](PES2TS_拓扑约束TS接近度与位移联合学习方案_v2_20261004.md) | 历史提案，已被 PES-N 取代 | 距离/直接位移改为辅助任务；保留原数学/标签论证与取代指针 |
| [PES2TS_统一价值模型与搜索排序一体化发展方案_v1_20261004.md](PES2TS_统一价值模型与搜索排序一体化发展方案_v1_20261004.md) | 历史提案，最新由 PES-N 承接 | 当前以协议条件化 p/C/Q 和 STOP/MOVE 统一；旧提案链保留 |
| `../PES2TS_开发宪法_v1.md` | 权威（SSOT，内容 v2） | 2026-10-05 同步 PES-N；R1–R6 原文保留，模型/阶段/接口/评测更新 |
| `PES2TS_阶段定义_G0-G3.md` | 权威（内容 v4） | 2026-10-05 G1 化学编译、G2 底座、G3 STOP/残差、G4 离线 Q/闭环；历史状态保留 |
| `PES2TS_整体开发与发布方案_v2.md` | 权威（内容 v3） | 2026-10-05 共享内核/宿主、合同兼容与当前发布门；旧版精确归档 |
| `PES2TS_G2后续开发方案_Classic到Rank到Gen_20261003.md` | G2 细化，后续学习被取代 | G2 工作保留，新增 PES-N 当前范围指针 |
| `PES2TS_G2_ACP能力补齐与联合验收计划_20261003.md` | 配套开发方案 | 新增；能力盘点不等于已冒烟，8 个联合任务仅是任务 ID |
| `PES2TS_统一开发顺序与阶段验收.md` | 历史 | 顶部加"已并入 G0–G4"；S7 处注"demo 降级为 G2.T" |
| `PES2TS_双项目开发与Demo方案.md` | 参考 | 保留定位/指标；阶段表指向主干 |
| `PES2TS_扫描规划谱系合并方案_20260930.md` | 参考(ADR) | 顶部加"已并入 G2.4 / X 工作流" |
| `PES2TS_第二轮修改最终方案_仅完全成断键_20261002.md` | 参考 | 顶部加"已并入 R 工作流（R0–R4）" |
| `PES2TS_图论立体先验与单端协同延续优化方案_v2_20261002.md` | 参考 | 顶部加"已并入 R 工作流（R0–R4）" |
| `PES2TS_逐步实施与验证方案.md` | 历史 | 顶部加"历史参考" |
| `PES2TS_ACP执行统一与旧代码清理方案_20261003.md` | 执行方案 | 新增登记；实现 ADR-0002，部分取代 ADR-0001 的 X1/X2 机制（X1′–X5′，planned） |
| `PES2TS_完整开发流方案_整合版_v1.md` | 权威（流程总纲，内容 v2） | 2026-10-05 共享内核与两入口、依赖、G2 关键路径与 ACP 优先级 |
| `../../AGENT.md`（仓库根） | 执行投影 v3 | 2026-10-05 同步宪法内容 v2/阶段 v4；非独立规范 |
| `PES2TS_G2_下一批算法调整与测试方案_20261003.md` | 批次方案（G2-AB1） | 新增；把 PES-G 的 A/B/G/H 包与根因报告 §10 P0/P1 细化为可实施/可测工作包（WP-1…WP-6），范围锁定 PES2TS 侧算法核心 + 测试 |

### 3.2 design/

| 文档 | 新状态 | 修订动作 |
|---|---|---|
| [decisions/ADR-0019-IRC20k与RL2000分阶段扩量.md](../design/decisions/ADR-0019-IRC20k与RL2000分阶段扩量.md) | Proposed（详细规划已登记） | 2026-10-08新增；反应计数/分批、跨任务用途和支持适用、全成本预算；不变更正式阶段/权限/能力 |
| [decisions/ADR-0018-RL纠错与动作支持研究计划.md](../design/decisions/ADR-0018-RL纠错与动作支持研究计划.md) | Proposed（规划已登记） | 2026-10-08新增；接口、支持域、回放与成对监督的条件/边界登记；实现前正式schema及规范同步，当前不升级能力 |
| [decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md](../design/decisions/ADR-0010-PES-N反应状态价值与STOP-MOVE统一决策.md) | Accepted（规划） | PES-N 共享反应状态/动作价值；同步规范，能力验收另行执行 |
| [decisions/ADR-0009-G阶段下的统一模型开发框架.md](../design/decisions/ADR-0009-G阶段下的统一模型开发框架.md) | 历史 Proposed，已由 ADR-0010 取代 | 保留原正文与当时状态；新增当前取代指针，模型与阶段主线按 PES-N |
| [decisions/ADR-0008-拓扑约束下的距离与位移联合预测.md](../design/decisions/ADR-0008-拓扑约束下的距离与位移联合预测.md) | 历史 Proposed，已由 ADR-0010 取代 | 保留原正文与当时状态；新增当前取代指针，模型与阶段主线按 PES-N |
| [decisions/ADR-0007-统一价值模型与搜索内生选点.md](../design/decisions/ADR-0007-统一价值模型与搜索内生选点.md) | 历史 Proposed，已由 ADR-0010 取代 | 保留原正文与当时状态；新增当前取代指针，模型与阶段主线按 PES-N |
| `PES2TS_PES-C到PES-G架构与迁移方案_v1_20261004.md` | 原生优化实施参考 | 控制/原生优化分工由 ADR-0010 承接；学习顺序被 PES-N 取代，生产迁移仍待验收，正文保留 |
| `PES2TS_PES-C-G架构审核与创新性可行性分析_20261004.md` | 审核报告（只读） | 2026-10-04 核查当前代码与公开一手文献；提出验证/标签、逐步几何反馈、数学定义、对照和创新性修改建议；不取代原方案/ADR，不使规范修订生效，不改变生产默认或能力状态 |
| `decisions/ADR-0006-ORCA原生约束优化与控制策略分离.md` | 迁移实现待验收 | 2026-10-05 方向由 ADR-0010 承接；正式合同/默认切换仍需证据；旧记录保留 |
| `G1_v2_补全实施总方案.md` | 参考 | 加"已并入 L0–L4 / X" |
| `G1_v2_L0到G2扫描策略设计.md` | 参考 | 保留 |
| `G1_v2_ORCA扫描模式识别与计划.md` | 参考 | 保留 |
| `G1_v2_一维优先与xTB二维成本策略.md` | 参考 | 保留 |
| `G1_v2_产物成键逆向扫描优先策略.md` | 参考 | 保留 |
| `G1_v2_方向无关成键锚点与双向扫描策略.md` | 参考 | 保留 |
| `G1_v2_多成键与少成键方向对比.md` | 参考 | 保留 |
| `G1_v2_成断键扫描方向选择可视化.html` | 参考 | 保留 |
| `G1_TS_IRC_映射与反应归簇实现方案.md` | 参考 | 保留（P1/P2 设计） |
| `PES_GENERATION_图论扫描策略选择器设计_v1.md` | 参考 | 保留（规划面权威设计） |
| `decisions/ADR-0001-*.md` | 权威 | 头部已注：执行本地化条款 + X1/X2 退出条件被 ADR-0002 部分取代；接缝/计划/记录/词表仍有效 |
| `decisions/ADR-0002-*.md` | 权威 | 新增登记；计算后端统一经 ACP 执行（部分取代 ADR-0001） |

### 3.3 contracts/

| 文档 | 新状态 | 修订动作 |
|---|---|---|
| `PES2TS_contracts_v1.md` | 参考 | 结果平面字段字典；补注 v2 规划平面另见 ADR/阶段定义 |
| `PES2TS_ACP字段映射_v1.md` | 参考 | 保留；补注：`XtbPathSearch` + `pes2ts_xtb_path_request_v1` 将扩展本映射（见 ACP 执行统一方案附录 A） |
| `PES2TS_数据规范与ACP对接优化方案.md` | 参考 | 保留 |

### 3.4 demo/

| 文档 | 新状态 | 修订动作 |
|---|---|---|
| `PES2TS_demo24_反应挑选与复核方案.md` | 参考 | 概念改名"G2.T 小批测试集规范" |
| `PES2TS_Demo评估指标_v1.md` | 参考 | 概念改名"G2.T 小批测试指标" |
| `PES2TS_Demo实现审核与架构解读_20260930.md` | 历史 | 加注 |

### 3.5 reports/（只读历史）

- 全部保留，**永不修改内容**；顶部加"当前阶段=G2"指针。
- `outputs/.../replay_summary.json`（17:27 批）及引用它的报告 → 标**作废**，以 strict 重跑为准。
- `PES_generation_多坐标约束学习_ORCA能力与数学方案_20261001.md` → 实为**设计方案误置于 reports/**；2026-10-04 加**顶部指针**（不改正文，符合"报告只读"）声明其"学一条目标曲线/联合路径"目标**非当前 PES-G 路线**；待 C2 归位到 `design/archive/`。

---

## 4. 状态头规范

```markdown
> 状态：权威 | 参考 | 已取代 | 历史
> 层级：主干 G# / 工作流 C/R/X
> 取代关系：被 X 取代 / 已并入 Y
> 上游依赖：...
> 下游消费者：...
> 当前版本：vN @ YYYY-MM-DD
```

---

## 5. 阶梯映射（防复活第三面）

见《PES2TS_阶段定义_G0-G3.md》§5。任何旧阶梯编号出现时，必须同时给出其对应的 G#/工作流。

---

## 6. 执行状态

- [x] 建立权威文档（阶段定义 / 主方案 / ADR-0001 / 本台账）
- [x] 为关键计划文档加状态头（统一开发顺序 / 谱系合并 / 两份 20261002）
- [x] 更新 `docs/README.md` 索引
- [x] 登记 ADR-0002 与《PES2TS_ACP执行统一与旧代码清理方案_20261003》（2026-10-03，含取代关系）
- [x] `docs/README.md` 索引核验：已含 ADR-0002 与新方案链接（2026-10-03）
- [x] 新增 G2 Classic→Rank→Gen 细化方案及 ACP 能力配套计划；同步阶段/主方案/执行方案与索引（2026-10-03）
- [ ] 建立《计数与标定台账》（C2 后续）
- [x] 修订 PES-G 数学方案、G1 边界审计与 ACP 增量；按 HEAD 3a5c875 区分已实现客户端和未验证能力；同步阶段/主方案/Classic 与索引（2026-10-03）
- [ ] reports 顶部"当前阶段指针"（批量，C2 后续）
- [x] 新增《完整开发流方案（整合版）》与仓库根 `AGENT.md`；阶段定义升 v3 并新增 G4；同步索引（2026-10-03，见 §9）
- [x] 新增《G2 下一批算法调整与测试方案（G2-AB1）》；同步台账与索引（2026-10-03，见 §10）
- [x] 建立《PES2TS 开发宪法 v1》为规范根（SSOT）；`AGENT.md` 降级为执行投影；完成全盘归一化裁决（2026-10-04，见 §11）
- [ ] 提交 CI 文档权威 lint（状态头 / 单权威 / AGENT↔宪法对齐）（C2 后续，见 §11.4）
- [ ] 阶段定义文件重命名 `_G0-G3` → `_G0-G4` 并同步全部入链（独立受审改动，见 §9）
- [ ] 多坐标学习方案归位 `design/archive/`（C2 后续）

## 7. 本轮设计修订裁决（2026-10-03）

- 阶段编号仍为 G0–G3、工作流仍为 C/R/X；Classic/Rank/Gen 是算法路线，ACP-G2-01…08 是联合任务 ID。**（阶段编号与 Gen 位置已由 §9 升级为 G0–G4；本行其余内容继续有效。）**
- G2 必须完成规则候选→严格验证与留数；G3 再训练 ranking，Gen 排在 Rank 之后。
- 24 条“跑通”细化为全输入可审计终态，成功、拒绝、部分路径分列；≥1 严格生成目标 TS 保留为工程最低门。
- G3 规模门从“1000 有效路径”改为“1000 冻结输入完整漏斗 + 实际有效路径/标签覆盖”，防止选择性排除失败。
- ValidationResult 不回流当前推理/valid/test 调参；允许隔离导出 train 标签。Demo24 保留旧 split 记录并增加开发污染标记，不再承担独立泛化结论。
- 活动事件 F/B、order_changed 仅归档、ACP 唯一执行后端、旧产物只读等既有约束继续有效。
- 本轮仅设计文档更新；没有把 ACP 部分接线、真实冒烟、TS 科学验证或模型训练标为完成。

## 8. PES-G 数学与边界修订（2026-10-03）

- 新增 [PES-G 方案](PES2TS_G2_PESG收紧算法迭代方案_20261003.md)：取代粘贴草案中质量加权防 H 滑移、测地距离保拓扑、谱 barrier/同构投影保证、强制单 irrep、VRI 泛化及“RMSD-PP/NT2 是带优化”等表述。采用有条件的 KKT 分支理论与可测试假说，不宣称全球收敛。
- 新增 [G1→G2 审计](PES2TS_G1_G2边界审计与补齐任务_20261003.md)：历史 gate/白名单对账 + 五个确定性样本，不冒称全量复审。现有映射与队列筛选都有真值来源，替换“端到端真值独立”结论。
- 阶段文档 §3 改为：G1 交方法来源和证据状态，unknown 合法；G2 选方法后经 ACP 起点准备。边界 v2/旁车待实现，旧 export v1、旧 gate 和数据不覆盖。
- ACP 计划 §2.1/§9 复用已存在的 OrcaGradient 客户端、SVD、曲率及候选模块；原生键级产品提前为 P1，键级导数仍为独立可选能力；ACP 部署/真实冒烟本次未核查。
- 约束分支、受驱动轨迹、NEB/string 三类方法按各自资格门生产帧，共享严格精修验证；历史直调实验脚本保持归档。
- 同步 Classic/主方案/索引；只新增设计和只读审计/代数证据，没有修改生产算法、启动 QC 或改变科学成功状态。

---

## 9. 完整开发流整合与 G4 新增（2026-10-03）

- 新增 [完整开发流方案（整合版）](PES2TS_完整开发流方案_整合版_v1.md)：G0–G4 阶段流总纲、双轨/三工作流、依赖图、里程碑、文档地图。
- 新增仓库根 [`AGENT.md`](../../AGENT.md)：受保护不变量（ACP 唯一后端、真值隔离、活动事件范围、三类路径方法分开、生成/排序分离、旧产物只读等）、数学工具证据边界、**架构变更审查协议**。
- **阶段定义升 v3**：新增 **G4 机理驱动定向生长（Topologically Constrained Path Growth）**；原 `Gen`（排在 Rank 之后）升为独立阶段 G4。§7 中"阶段编号仍为 G0–G3"自本版起改为 **G0–G4**。
- 文件名 `PES2TS_阶段定义_G0-G3.md` 保留历史名以兼容既有链接，内容已扩展至 G0–G4；**重命名至 `_G0-G4` 作为独立的受审改动另行推进**。
- 本版仍为设计/治理层：未实施新算法、未启动 QC、未改变科学成功状态。

---

## 10. G2-AB1 下一批算法调整与测试（2026-10-03；2026-10-04 宪法对齐修订 v2）

- 新增 [G2 下一批算法调整与测试方案](PES2TS_G2_下一批算法调整与测试方案_20261003.md)：批次号 **G2-AB1**，范围锁定 **PES2TS 侧算法核心 + 测试**，不新增主干阶段。
- 工作包：WP-1 执行身份命名空间与内容寻址梯度复用（对应 strict `CACHED_GRADIENT_INPUT_MISMATCH`）；WP-2 相对上一接受帧的分支约束/释放/分级局部性（对应校正阶段大位移）；WP-3 起点准备证据化与结构化失败；WP-4 梯度排名接入验证入口并允许部分路径候选；WP-5 严格 landing 身份分级；WP-6 代数反例固化为仓库测试；WP-7（v2 新增）R5 成本账本 / R6 分裂防火墙 / 数据飞轮接口固化。
- **v2 宪法对齐（2026-10-04）**：全文改为引用宪法 SSOT（R1–R6 / P1–P5 / §5–§11），不新增规范；命名改用 `G0–G4` + 学习级 `L-G*/L-R*` + M0–M5 声明投影；WP-2 增加 `A(H)`/分支证据；WP-4 分离 basin-entry 与 verified TS；WP-7 落地 R5/R6；新增 §5 Preparation Layer 诚实声明、§7 决策层回放（P3，非逐位）与 `J_G = P_basin − λ·C_xTB/C_ref` / `R_oracle(H)` 报告口径；§8 架构审查按宪法 §12 重写。
- 与 PES-G 的 A/B/G/H 包、根因报告 §10 的 P0/P1 条目对齐；`evaluation_id` 改名对齐数据链身份（AGENT §5 / 宪法 P1），冲突拒绝保留。
- **离线实施（2026-10-04，V0 门通过）**：WP-1…WP-7 全部落地（实现 + 测试一体；改动文件与新测试清单见 `.omo/evidence/task-10-pes2ts-g2-ab1.txt`）。新增接口按 P1 只固定存在性与校验并登记宪法 §9.6 + [ADR-0003](../design/decisions/ADR-0003-G2-AB1接口登记.md)：`endpoint_method_evidence_v1`（拟定）、`pes2ts_execution_identity_v1`（评估身份命名空间）、`pes2ts_gradient_seed_proposals_v1`（验证入口唯一候选来源）、`pes2ts.path_provenance.v1`（completed_interval 报告字段 + `preparation_layer_status=absent`）、成本账本/`label_state` 七态/四层 provenance/分裂防火墙守卫（`pes2ts_core/flywheel.py`）。四层状态：离线新能力至多 `fixture_passed`；V2/V3 真实引擎与科学评测未运行，保持 `implemented`（无环境时 `blocked`）。
- 四层状态纪律不变：本批完成至多把离线部分升为 `fixture_passed`；真实 QC 冒烟（V2/V3）为 gated，未就绪记 `blocked`。**写入本方案不改变任何能力状态。**

---

## 11. 开发宪法归一化裁决（2026-10-04）

**背景**：对全部计划/设计/合同/报告（约 25 份）与新冻结的长期战略定位做冲突审计，建立规范根并归一。

### 11.1 新增与降级
- **新增规范根**：`docs/PES2TS_开发宪法_v1.md`（SSOT）。
- **降级**：仓库根 `AGENT.md` → 宪法执行投影（只含红线 ID 摘要 + 操作性规则）。
- 规范冲突一律以宪法为准。

### 11.2 冲突裁决表

| # | 冲突点 | 裁决（写入宪法位置） |
|---|---|---|
| D1 | PES-G "学一条目标曲线/联合路径"（多坐标方案） | 宪法 §6：PES-G = learned predictor + QM corrector，目标为进 TS basin 而非 MEP；该方案降级为 teacher/oracle 参考（§3.5 指针） |
| D2 | 学习型生成无阶段归宿 | 宪法 §6.5：新增**学习级**命名空间 `L-G0…L-G4`（非工程阶段）；映射 M3/M4 → G4.0/G4.1 |
| D3 | 开发顺序语义 | 宪法 §10.3：dev = Classic→Rank→Gen；paper = Gen→Rank→DFT |
| D4 | 家族泛化非强制 | 宪法 R6 / §11.2：**强制 reaction-family holdout** |
| D5 | 缺 preparation layer | 宪法 §8：xTB→DFT 之间必须插 constrained DFT preparation；当前未实现须标注 |
| D6 | 论文顺序缺失 | 宪法 §10.3 |
| D7 | 定位术语缺失 | 宪法 §1.1/§1.2 正式定义 |
| D8 | 缺 H=(R,P,ΔG,E,O,S,Q,M) | 宪法 §5.1（版本化接口） |
| D9 | 缺 A(H)/action | 宪法 §6.2 + R4 |
| D10 | 缺模型阶梯 | 宪法 §6.6：rule→GBDT/MLP→MPNN→E(3)-GNN |
| D11 | 缺经典 LTR | 宪法 §7.4：LambdaMART/LightGBM/XGBoost 先行 |
| D12 | 排序标签未定义 | 宪法 R2 / §7.3：label := OptTS→Freq→IRC 成功 |
| D13 | 缺数据飞轮合同 | 宪法 §9.1：(s_t,a_t,s_{t+1})、(X_i,OptTS,C_i) |
| D14 | 里程碑多套 | 宪法 §10.1/§10.2：G0–G4 为唯一交付轴；M0–M5 为声明投影；§10.4 命名纪律 |
| D15 | benchmark 未命名 | 宪法 §11.1：Generation / Ranking 两套，解耦 |
| D16 | "方向/调度"归 PES2TS | 宪法 §2.3 注记：专指扫描方向/路径调度；作业调度归 ACP；同步修 `G2_ACP能力补齐` §1 |
| D17 | RPH 同词异义 | 宪法 §2.4：ReactionProfileHunter（code）vs RPH-Agent（文献推理） |
| D18 | CCCP 命名缺失 | 宪法 §2.4：正式命名计算接口层 CCCP |
| D19 | RPH bridge 发布单元前置 | 宪法 §2.3：RPH 为第二层；`pes2ts-acp-bridge-v1` 降级为后置集成 |
| D20 | IRC 距离地位 | 宪法 §6.3：轨迹近 IRC 仅辅助诊断 |

### 11.3 内部矛盾裁决

| # | 矛盾 | 裁决 |
|---|---|---|
| I1 | 主干 G0–G3 vs G0–G4 | 全域统一 **G0–G4**；已修 `整体方案v2` L4/L24/L35/L182、`G2后续` L19/L21/L224、本台账 §7 |
| I2 | Gen = G4 vs G3 后续 | 统一 **Gen = G4**；已修 `G2后续` |
| I3 | "原始解析"归属 ACP vs PES2TS | 裁决：ACP 提供**原始输出/原始产物**；PES2TS 保留 **xtb_output 解析 + inspect 判定**（与 ADR-0002 一致）；同步修 `G2_ACP能力补齐` §1 |
| I4 | 阶段文件名 `_G0-G3` vs 内容 G0–G4 | 保留历史名以兼容链接；内容 G0–G4；重命名列为独立受审改动（§9） |
| I5/I6 | 旧里程碑多套（M1–M3 / M-A–M-F 等） | 宪法 §10.4 命名纪律；旧阶梯仅作历史映射（阶段定义 §5） |

### 11.4 遗留待办（C2）
- 提交 **CI 文档权威 lint**：所有 `docs/**/*.md` 带状态头；每主题恰一 `AUTHORITATIVE`；`AGENT.md` §2 与宪法 §3 对齐；已取代文档不入检索索引。
- 阶段文件重命名 `_G0-G3` → `_G0-G4` 并更新全部入链（README / AGENT / 台账）。
- 多坐标学习方案归位 `design/archive/`。
- `docs/README.md` 与根 `README.md` 增加宪法入口（本次已同步）。

### 11.5 效力
本裁决自 2026-10-04 起生效；与宪法不一致者以宪法为准。本轮只做**治理与归一**：未修改生产算法、未启动 QC、未改变任何能力状态。

---

## 12. G2-AB2 下一批实现方案：真引擎接通与验收（2026-10-04）

- 新增 [G2 下一批实现方案：真引擎接通与验收](PES2TS_G2_下一批实现方案_真引擎接通与验收_20261004.md)：批次号 **G2-AB2**，细化 G2-AB1 遗留的 V2/V3 gated 验收、Classic→Rank→Gen §10 工作包 3–5 与阶段定义 G2 退出门前置项①③。
- **v2 裁决（2026-10-04）**：以 **G2.T 24 例首轮端到端战役 + 第一份真实反馈** 为唯一主轴重组（取代 v1 并行目标结构）——四例 strict 重放降为战役试点阶段（S2，=V2 门），参数校准改试点驱动，`J_G` 完整口径降为条件项，第一份反馈报告升为第一交付物。
- 工作包/阶段：S1 WP-1 战役基建（编排器 `pes2ts g2t-run`/幂等恢复/预算强制/四类终态 + 成本采集）；S2 WP-2 试点 4 例 + 参数冻结；WP-3 起点准备 ACP 生产适配器（fail-closed）；S3 WP-4 全量 24 例（四类终态分列 + 完整账本 = G2 退出门前置项①）；S4 WP-5 验证链执行（硬依赖人工复核，双轨不阻塞生成轨道）；S5 WP-6 第一份反馈报告（漏斗/失败直方图/B1 对照/成本/AB3 改进清单）+ WP-7 收口冻结。
- 交付物 D1–D5（战役基建、试点通过、24/24 可审计终态、≥1 次验证链执行、首份反馈报告）。**不承诺** ≥1 严格目标 TS 与 M0 声明（G2 退出门）。
- 新接口拟定：`pes2ts_campaign_manifest_v1`、`pes2ts_g2t_terminal_state_v1`、`pes2ts_g2t_feedback_report_v1`、`pes2ts_branch_calibration_v1`（实施时按宪法 §12 登记 §9.6）。
- 状态纪律：本方案为设计层，未实施、未启动 QC、不改变任何能力状态（宪法 §11.5）。
- **离线实施（2026-10-04，V0 门通过）**：S1 离线交付全部落地——WP-1 战役编排器（`pes2ts_core/generation/campaign.py`：manifest 冻结/幂等恢复/预算强制/typed 四类终态）+ 成本采集（`planning/cost_capture.py`）+ CLI `g2t-manifest`/`g2t-run`/`g2t-status`/`g2t-report`；WP-2-A 离线解析网格 + 判定器（`planning/branch_calibration.py`、`planning/replay_report.py`），最保守通过格冻结入 `config/defaults.yaml`（`g2.continuation.branch_policy` 等，测试互锁）；WP-3 `ORCAOriginPreparation`（`integration/acp/origin_backend.py`，fail-closed，加入 `SUBPROCESS_IMPORT_ALLOWLIST`）；WP-6 报告器（`generation/campaign_report.py`）；WP-7 `scripts/collect_g2t_terminal_states.py`。新接口登记 §9.6 + [ADR-0004](../design/decisions/ADR-0004-G2-AB2接口登记.md) + [战役合同字段字典](../contracts/PES2TS_G2T_战役合同字段字典_v1.md)。无 ACP 环境实测：`g2t-run` 正确产出 typed `blocked`（`ORIGIN_BACKEND_UNAVAILABLE` + 需求清单，不静默回退）。改动文件与测试清单见 `.omo/evidence/task-11-pes2ts-g2-ab2.txt`。**S2 试点/S3 全量/S4 验证为 gated**（真实引擎 + 人工复核），相关能力保持 `fixture_passed`/`implemented`，不升 `smoke_passed`（§11.5）。

---

## 13. G2.T 真实战役执行方案（2026-10-04）

- 新增 [G2.T 真实战役执行方案](PES2TS_G2T真实战役执行方案_20261004.md)：嵌套归属 = G2-AB2 批次 S2-S5 执行 ⊂ G2.T 子阶段（阶段定义 :46）⊂ G2 ⊂ G0-G4 主干；细化不取代 G2-AB2 v2，判据修订以 ADR-0005 为准。
- **Wave 0 四项执行前修补（2026-10-04 已实施，先修订后执行）**：①campaign.py 恢复顺序（已完成终态先于预算检查）②corrector 缓存复用证据透传（acp.reused → cache_hit 零成本账本）③origin BatchOptimize profile opt→opt_only + 能量读取改轨迹 JSON 末周期 energy_hartree ④试点判定器语义 v2（schema `pes2ts_pilot_replay_report_v2`）。全部经 [ADR-0005](../design/decisions/ADR-0005-G2-AB2执行前判定器与恢复修补.md) 登记（已接受）；离线回归门 1720 passed > 基线 1709，证据 `outputs/g2t_campaign_logs/T2..T6`。
- S2-S5 阶段概要：S2 试点 4 例（7762 双跑独立账本，λ 绑定锚点复现）→ S3 全量 24 例（单 manifest `--case` 子集分批）→ D3 验收（24/24 可审计终态 + 完整账本 = G2 退出门前置项①）→ S4 验证链条件（硬依赖人工复核导入，双轨）→ 状态冻结与 D5 第一份反馈报告。
- 状态：设计登记，不改变能力状态（宪法 §11.5）；S2 及之后阶段待真实执行。本方案为执行细化，不新增规范条文、不承诺 ≥1 严格 TS / M0-M5。
- **S2–S5 执行结果（2026-10-05 战役收口同步；嵌套归属不变：G2-AB2 批次 S2-S5 执行 ⊂ G2.T 子阶段 ⊂ G2 ⊂ G0-G4；判据修订仍以 ADR-0005 为准）**：
  - D1–D5 全交付。D3 双真：collector exit 0 + 断言 a–g 全过（`.omo/evidence/T18b-05-collector.log`、`T18b-06-assertions.log`、`T18b-07-summary.log`）；S3 根 `outputs/g2t_campaign_s3_20261004/`，24/24 typed 终态，443 ledgers，N_gradient 总 26676。
  - 四类分母（分列，不合并）：rejected_typed 2 / partial_prefix 19 / complete_path 3 / validated 0；S4 gated：24/24 `blocked_on_human_review`（`.omo/evidence/T19-01-gated.log`），非 skip。
  - 试点三轮校准 GO 4/4：T15（local_corrector/continuation 120/600 s）0/4 → T15b（480/2400 s）3/4 → T15c（480/4000 s）4/4（`outputs/g2t_campaign_logs/T15c-08-judge.log`、`T15c-09-summary.log`）；判定器 v2（ADR-0005），身份冲突全程 0。
  - 执行期缺陷两例已修：T17b curvature probe 契约缺陷（先红后绿，证据 `outputs/g2t_campaign_logs/T17b-01..05`）；T18b RXN_0000155302 零 attempt 账本补写（先红后绿，证据 `.omo/evidence/T18b-01..07`）；T18 首轮 FAIL 记录保留不改写。
  - D5 报告：`outputs/g2t_campaign_s3_20261004/feedback_report_final_v1.json`（schema `pes2ts_g2t_feedback_report_v1`）；`claims.success_rate_claimed=false`、`m0_through_m5_claimed=false`；能力状态在 T20 冻结后由该报告嵌入，本条目不重述层级。

## 14. PES-N 未来计划与规范同步（2026-10-05）

用户明确要求按最新开发哲学修改未来计划，本次据此使开发方向及规范投影同步生效；未把文档生效解释为代码、模型或 QC 能力完成。

- 新增 PES-N 计划和 ADR-0010；同步宪法内容 v2、阶段 v4、AGENT v3、项目发布 v3、流程总纲 v2、根 README 及索引。
- 采用共享 p_seed/精修成本/Q 内核；Ranking=评价 STOP，Navigation=比较 STOP/MOVE；距离与直接位移降为有证据的辅助/残差任务。
- 路线改为 Classic 物理底座→STOP 监督→模仿/残差→保守离线 Q→冻结模型的真实闭环；小模型、风险校准、跨来源和整反应族留出纳入发布门。
- 修正数学及数据边界：局部松弛分支、乘子梯度/力符号、有效曲率条件；失败惩罚与预算终端；梯度残差不是流形距离；跨来源帧不等于共同动力学转移；付费真实前瞻不自动带来 MPC 稳定保证。
- 原生优化迁移与 G2 工作包继续承接；原双输出/独立学习提案标历史，正文保留。旧报告和已冻结物理产物不改写。
- 原六份入口文件的精确快照及哈希保存在 [改版归档](../archive/20261005_pre_pesn/README.md)。本轮校验限文档一致性、引用、快照/业务文件哈希；不以此替代真实模型或 QC 验收。

十五项第一性原则逐项工程落点见 PES-N 计划 §10；当前未来工作清单见 §11。没有启动计算或训练，没有修改业务代码/配置/生产默认。

---

## 15. 三模块产品架构与统一可视化（2026-10-05）

用户要求把现有G1/G2组件整合为拓扑约束、控制器、计算器三个根本模块，并增加便于输入与理解结果的可视化层。本次落实设计与规范投影，具体模块实现、schema、UI及真实能力另验收。

| 文档 | 归属与本次动作 |
|---|---|
| [三模块产品架构设计](../design/PES2TS_三模块与可视化产品架构_v1_20261005.md) | 新增当前设计v1：职责、闭环、多输入、组件映射、数据与UI接缝、分期验收 |
| [ADR-0011](../design/decisions/ADR-0011-三模块产品架构与统一可视化.md) | Accepted开发方向：细化ADR-0010与ACP展示边界，不取代阶段/旧接口 |
| [开发宪法](../PES2TS_开发宪法_v1.md) | 内容v2.1：新增§2.5产品视图及§5.5不同完整度输入与评价资格；R1–R6原文保留 |
| [AGENT执行投影](../../AGENT.md) | v3.1：同步宪法版本与产品设计指针；受保护目录和阶段出口保持 |
| [PES-N未来计划](PES2TS_PES-N未来开发计划_v2_20261005.md) | v2.1：§8.3承接产品整合，STOP→模仿/残差→离线Q→冻结闭环顺序保持 |
| [项目开发与发布方案](PES2TS_整体开发与发布方案_v2.md) | v3.1：一个用户产品，工程发布单元不等于多套用户产品 |
| [文档索引](../README.md) | 增补设计/ADR入口及当前版本 |

接口沿用ADR-0010已登记存在性的ChemicalControlSpec、ReactiveState、Action/Transition、DecisionEstimate等合同族；本轮未新增正式schema，也未扩大已有F/B、电子态、坐标和后端放行范围。新输入不能通过伪造空P进入旧ready合同；目标不充分时目标STOP评价不可用。

拓扑规则服务独立于模型，控制器调用规则并管理接受状态；计算器实际执行与任务/帧展示复用ACP。UI与CLI共享应用入口和版本化记录，参考真值留在隔离侧。本轮证据限源文件/接口定位、文档一致性与引用检查；没有训练、计算、业务代码/配置变化、目录迁移或能力状态升级。G1/G2整合及可视化属C/R/X，模型接入仍归G3/G4，G2退出门继续以阶段定义为准。

---

## 16. Classic 图论驱动与 xTB 原生扫描设计（2026-10-06）

用户指定“化学图论约束→每次只移动关键成键原子→xTB限制性扫描→重复覆盖范围”，要求现有代码功能整合及优化器改造方案。本轮交付设计；未改运行代码、正式schema、预算、默认方法或真实能力。

- 新增[Classic设计](../design/PES2TS_Classic图论驱动与xTB原生约束扫描设计_v1_20261006.md)、[G2-AB3实施方案](PES2TS_G2_AB3功能整合与原生扫描改造方案_20261006.md)、[ADR-0012](../design/decisions/ADR-0012-Classic图论驱动与xTB原生约束扫描.md)，归属G1/G2及C/R/X。
- 三模块以薄门面整合、保留目录：拓扑编译与独立规则；Classic目标/关键原子初猜/接受状态；ACP原生优化与收据接缝。旧投影BFGS保留显式对照，新方法无静默回退。
- 初猜只改变F/B driver所涉K原子；默认其余自由度由xTB松弛，全冻结另作协议。扫描完成指冻结低维区间覆盖，不宣称完整高维PES、MEP或TS。
- ACP底层已有xTB constrained_optimize/relaxed_scan，但当前success仅依赖结构存在，批量回调无化学接受握手。公开单点入口、收敛/实际残差、物理与约束能量、费用/恢复需先验收；PES2TS不绕过ACP。
- 登记待实施草案：`pes2ts_chemical_control_spec_v1`、`pes2ts_native_scan_plan_v1`、`pes2ts_native_constraint_request_v1`、`pes2ts_native_constraint_outcome_v1`、`pes2ts_native_scan_trace_v1`、`pes2ts_native_scan_seed_proposals_v1`。字段及父hash检查见设计§7；正式schema/fixture在实现批次另登记，旧链/历史报告不改义。
- 工作包WP-0…10：先修审计报告/费用，再ACP合同、图规则与稀疏预测、原生控制器、候选/兼容；离线→真实train/calibration4例→冻结24例→严格验证→发布裁决。不能因加速或计划勾选宣称G2科学出口。
- 同步宪法§9.6草案指针（规范内容仍v2.1）、AGENT当前指针、根README、本台账v2.2与docs索引。未改G3/G4路线；未来PES-N消费有资格的原生轨迹及观测掩码。

本轮证据为当前源码定位、已有AB2审核与xTB官方资料、文档状态/链接/变更范围检查。没有启动量化计算或训练、Git提交、目录迁移，未把新能力标smoke_passed。实施时按全分母、真实收据与严格目标验证验收。

---

## 17. G3.P 并行 IRC 教师预训练与版本立项（2026-10-06）

用户明确授权调整现有阶段规划并制定第一期训练整体版本/计划。按 ADR-0013 增加 G3 内 G3.P 支线；这部分提前启动，正式 G3.0–G4 的物理标签/接入与科学出口继续保留。

| 文档 | 当前版本 / 本次动作 |
|---|---|
| [ADR-0013](../design/decisions/ADR-0013-G3并行IRC预训练与模型版本体系.md) | v1 Accepted：部分取代 ADR-0010 全串行启动安排；教师与生产状态/转移资格分开 |
| [G3-P1 训练方案](PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md) | v1 新增；来源/分裂→标签/基线→编码器→研究冻结；WP-0…7、配置/扩量/指标及迁移 |
| [教师合同](../contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md) | v1 新增设计草案；dataset/labels/split/features/run/bundle 各版本与父哈希；正式schema未发布 |
| [开发宪法](../PES2TS_开发宪法_v1.md) / [执行投影](../../AGENT.md) | v2.2 / v3.2；登记隔离教师和并行入口，R1–R6原文不改，未扩guard权限 |
| [阶段定义](PES2TS_阶段定义_G0-G3.md) | v4.1；G3.P.0–.3及G3-P1，入口无G2调教依赖；正式1000漏斗/STOP/残差/Q/闭环门保留 |
| [PES-N计划](PES2TS_PES-N未来开发计划_v2_20261005.md) / [发布方案](PES2TS_整体开发与发布方案_v2.md) / [流程总纲](PES2TS_完整开发流方案_整合版_v1.md) | v2.2 / v3.2 / v2.1；双线汇合、研究包和真实决策能力分开发布、初始化对照/版本兼容 |
| README / docs索引 / 本台账 | 当前入口和取代关系同步；本台账v2.3 |

首个模型目标 `pesn-irc-v0.1.0`；后续 0.2 STOP、0.3 residual、0.4 offline Q、0.5 closed loop、1.0 core 是待验收的规划身份，软件 `0.1.0` 不 bump。教师数据发布目标0.1.0和六个v1合同草案分别登记，不覆盖旧split/数据/报告。

源方法/规模先核验：当前真值清单是B3LYPD3_TZVP TS/IRC，不自动把“20万IRC/TS/xTB”理解为20万xTB IRC或逐帧匹配。第一期教师标签包括分支质量弧长进度、路径接近度、尺度化坐标朝TS方向/协同；不当精修成功/成本或真实执行命令，不用于Bellman转移。scope访问与正式schema在实现阶段验收。

状态：设计/版本/阶段已登记；数据构建/训练程序/权重/资源测量/独立评测尚未启动，正式决策能力未升级。未改业务代码、软件版本、默认配置、truth allowlist、冻结清单或旧物理结果；现有未提交工作在当前内容上增量保留。

改版前九份入口精确快照及376份源码/测试/配置/脚本/清单hash见[归档](../archive/20261006_pre_g3p/README.md)。本轮检查是文档链接/版本/边界一致性、R1–R6与原出口保留及变更范围核验，不用作模型或QC冒烟。真实收益、迁移和科学里程碑在对应阶段另提供证据。

## 18. 预测器产品归属与G3代码范围细化（2026-10-07）

用户要求当前代码开发按产品内容细分，并统一在预测器模块下。源码对照发现pesn目录尚未存在、ranking仍为规则排序、通用evaluation仅为标记；原方案覆盖训练过程但没有完整的产品/文件/导入/分发范围。本轮据现状完善设计，未执行代码开发。

| 文档 | 当前版本 / 本次动作 |
|---|---|
| [ADR-0014](../design/decisions/ADR-0014-预测器产品归属与G3代码范围.md) | v1 Accepted：部分取代ADR-0013目录建议；所有新增业务代码统一pesn，runtime/offline隔离 |
| [预测器模块划分](../design/PES2TS_预测器产品与G3代码模块划分_v1_20261007.md) | v1新增：现有资产/产品矩阵、目标文件树、B0–B7、接口/配置/依赖、导入/真值/导出及测试验收 |
| [G3-P1](PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md) / [教师合同](../contracts/PES2TS_PESN_IRC教师数据与模型包合同_v1.md) | v1.1 / 草案内容v1.1；实施范围及PESNInput/TeacherPrediction草案；payload schema身份保持v1 |
| [宪法](../PES2TS_开发宪法_v1.md) / [执行投影](../../AGENT.md) | 内容v2.3 / v3.3；§2.6产品归属及依赖边界；R1–R6原文保留 |
| [阶段](PES2TS_阶段定义_G0-G3.md) / [未来计划](PES2TS_PES-N未来开发计划_v2_20261005.md) | v4.2 / v2.3；G3.P代码归属，原G2出口与正式G3/G4门保持 |
| [发布](PES2TS_整体开发与发布方案_v2.md) / [流程](PES2TS_完整开发流方案_整合版_v1.md) / [产品架构](../design/PES2TS_三模块与可视化产品架构_v1_20261005.md) | v3.3 / v2.2 / v1.1；预测器在线/离线分发、预测与显式选择、教师评测与系统评测分工 |
| ADR-0013 / 根README / docs索引 / 本台账 | 前ADR新增局部取代指针，历史决策保留；入口同步；本台账v2.4 |

本期业务文件唯一根`pes2ts_core/pesn/`；教师数据/标签/分裂/训练/模型评测/导出归`pesn/offline/`，取消未实施的`evaluation/pesn_irc/`及顶层`pesn/training/`布局。在线特征只计算拓扑authority给定描述，不重新定义许可；控制器/ACP接入及全系统严格验证是其他产品工作。

测试、产品配置与独立可选依赖是配套范围；B2计划仅登记`offline/data/truth_source.py`窄桥访问G0唯一审计reader，guard当前未修改。runtime不得传递导入offline/来源/QC/生成链；导出白名单禁止教师答案、原始数据和optimizer。独立sealed评测无fit路径，不改变冻结父split。

软件`0.1.0`、模型目标`pesn-irc-v0.1.0`、数据目标0.1.0、G3-P1批次及科学出口保持；代码、依赖安装、数据构建、训练、权重和QC均未实施。本轮保存[十三份精确文档快照及376个受保护文件hash](../archive/20261007_pre_predictor_scope/README.md)，最终文档校验与范围核验记录到同目录verification.json，不作为模型能力验收。


## 19. 稳定G2与轻量RL衔接（2026-10-07）

用户要求保持简单快速、继续正在实施的G3，并通过学习控制推进G2；本次规划沿用现有原生执行器和G3范围。

| 文档 | 当前版本 / 本次动作 |
|---|---|
| [轻量控制与RL衔接计划](PES2TS_轻量控制与RL衔接计划_v1_20261007.md) / [ADR-0015](../design/decisions/ADR-0015-G2稳定底座与轻量RL研究入口.md) | v1 / v1；有限步长与候选提交、共用执行循环、规则/BC/小Q、小批同预算对照；复杂物理后置 |
| [宪法](../PES2TS_开发宪法_v1.md) / [执行投影](../../AGENT.md) | 内容v2.4 / v3.4；P5/§6.5研究入口与正式默认门分开，R1–R6原文保持 |
| [阶段](PES2TS_阶段定义_G0-G3.md) / [未来计划](PES2TS_PES-N未来开发计划_v2_20261005.md) | v4.3 / v2.4；研究原型可并行，AB3/G3指针更新，正式阶段出口保持 |
| [发布](PES2TS_整体开发与发布方案_v2.md) / [流程](PES2TS_完整开发流方案_整合版_v1.md) | v3.4 / v2.3；固定G2、续做G3、4例试点；研究能力不自动发布为默认 |
| [G3-P1](PES2TS_G3_P1_IRC教师预训练与模型版本计划_v1_20261006.md) | v1.2；沿已有B0–B7和首轮模型目标，仅补后续可选复用，不重复实施 |
| 根README / docs索引 / 本台账 | 当前入口/状态/版本同步；台账v2.5；历史条目按当时版本保留 |

接口草案登记：`pes2ts_control_action_v1`、`pes2ts_control_transition_v1`、`pes2ts_policy_reward_spec_v1`，归既有Action/Transition族；模型沿PESNModelBundle扩研究能力。正式schema/validator待实现；旧网格和新变步长完整性分开。奖励代理不作严格Y，人工gate/未知终态不自动作负例；所有失败/尝试和费用保留。

证据范围：已有原生24例13/9/2且validated=0；用户确认G3实施进行中、工作区目录存在；本次只改计划/规范/指针，没有修改业务代码、guard、依赖、权重、报告或默认，没有启动QC/训练。改版前10份文档原字节和hash见[快照](../archive/20261007_pre_light_policy/README.md)。文档与链接核验单独留存，不代表策略已验收或科学门通过。


## 20. ORCA驱动xTB硬约束优化试点（2026-10-07）

用户授权针对几何未收敛尝试ORCA驱动xTB，并要求优化方案。本批归G2-AB4，聚焦一个失败机制，沿用原生基线、父帧/网格/化学检查和ACP唯一执行；G3-P1任务及RL有限动作保持。

| 文档 | 当前版本 / 本次动作 |
|---|---|
| [G2-AB4方案](PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md) / [ADR-0016](../design/decisions/ADR-0016-ORCA驱动xTB硬约束优化试点.md) | v1 / v1；公开薄工作流、硬约束与物理能量、4点成对门、共享预算有限回退及文件级实施范围 |
| [4点选择清单](fixtures/g2_ab4_orca_xtb_pilot_selection_v1.json) | 文档选择manifest v1；3历史失败点+1正常点及16项源产物hash；未执行，方法/电子态/能力资格待核验 |
| [宪法](../PES2TS_开发宪法_v1.md) / [执行投影](../../AGENT.md) | 内容v2.5 / v3.5；仅接口/设计登记与实施指针，R1–R6和工程政策原文保持 |
| [轻量计划](PES2TS_轻量控制与RL衔接计划_v1_20261007.md) | v1.1；对已有失败机制启动有界优化器试点，不增加学习动作或复制G3 |
| 根README / docs索引 / 本台账 | 入口和状态同步；本台账v2.6，历史记录保持当时口径 |

待实施接口草案：`pes2ts_orca_xtb_constrained_opt_request_v1`、`pes2ts_orca_xtb_constrained_opt_evidence_v1`、混合`pes2ts_native_scan_result_v2`及trace v2。硬约束E_effective为不适用，候选只读裸xTB单点E_physical；缓存绑定优化器/约束/方法，旧v1只读。实际schema/validator、能力和默认另验收。

证据为395份AB3优化收据核验（350收敛/45未收敛，后者7例且未检出SCC失败）、现有代码/ACP注册与官方手册、冻结4点源文件摘要；不将此证据当ORCA收益或严格TS验证。旧source trace中Linux pilot路径保留，选择清单只以workspace_path定位复制产物。

本轮只改文档和选择清单，未改ACP/PES2TS业务代码、guard、依赖、权重、历史运行报告或默认，未启动QC/训练。改版前6份文档原字节及hash见[快照](../archive/20261007_pre_orca_xtb_pilot/README.md)；正式阶段/G3范围保持，文档核验不升级算法能力。

## 21. 统一残差决策、MDP研究闭环与G2有限回退（2026-10-07）

用户要求先接通RL再扩数据、一个统一模型、明确决策与注意力，并在此轮统一G2-AB4改造。本次规划局部取代ADR-0015的步长分类优先/独立小模型近期顺序，承接ADR-0016的原四点、方法、60秒、正常点合格/救回≥2/3及每目标一次/每例两次/900秒共享预算门。

| 文档 | 当前版本 / 本次动作 |
|---|---|
| [统一实施方案](PES2TS_统一残差决策与G2有限回退实施方案_v1_20261007.md) / [ADR-0017](../design/decisions/ADR-0017-统一残差决策与G2有限回退.md) | v1 / v1：现有规模真实RL、共享网络残差/Q头、MDP近似及坐标注意力、公共G2单步/宏步/费用，六包交付和边界 |
| [轻量计划](PES2TS_轻量控制与RL衔接计划_v1_20261007.md) / [AB4](PES2TS_G2_AB4_ORCA驱动xTB约束优化试点方案_20261007.md) | v1.2 / v1.1：当前入口统一，旧步长为可选对照；AB4救援门/原清单不变，稳定逻辑目标延续配额，策略不选优化器 |
| [宪法](../PES2TS_开发宪法_v1.md) / [执行投影](../../AGENT.md) | 内容v2.6 / v3.6：近期研究顺序、接口及状态指针；R1–R6原文保持 |
| [阶段](PES2TS_阶段定义_G0-G3.md) / [未来计划](PES2TS_PES-N未来开发计划_v2_20261005.md) | v4.4 / v2.5：当前研究入口，正式阶段/退出门及长期路线保持 |
| 根README / docs索引 / 本台账 | 入口/版本/局部取代关系同步；台账v2.7；历史条目保持当时口径 |

接口存在性草案：pesn_control_input/prediction_v1、control_action/transition_v2、policy_reward_spec_v2、model_bundle_v2、学习native_scan_result/trace_v3。实施schema/validator待冻结；旧教师v1、固定网格原生v1/AB4混合v2与历史产物不改义。不扩大真值/执行allowlist，不解锁正式p/C/Q_MOVE或生产控制。

规则/Q比较采用同一冻结环境；E_native和E_hybrid数据不无条件混训。恢复余量/pending目标/预算入状态，宏步/内部尝试费用不重复；IRC不冒充命令或真实转移。四点/四路径/四RL反应资格分别核验，Demo开发暴露不自动成为train。

本次新增计划/ADR并同步9份文档，未改业务代码、配置、测试、依赖、权重、AB4选点清单或旧报告，未运行QC/训练/读取真值。改版前精确快照及386个业务配套哈希见[归档](../archive/20261007_pre_unified_residual_rl/README.md)，规划检查记录到verification.json，不代表能力验收。

## 22. Demo24真实RL闭环与冻结策略对照方案（2026-10-07）

用户要求下一阶段在之前遴选的24例中跑通RL。本次新增[Demo24实施方案](PES2TS_Demo24_真实RL闭环与对照实施方案_v1_20261007.md) v1.0，作为ADR-0017/统一方案包C/E的执行细化。原24例为16 train与8 valid；仅合格train采集/拟合，全部24例在模型、规则与预算冻结后参加工程对照，不重新划分官方split或将已暴露valid作为独立holdout。

已核对原候选CSV与遴选文档的16/8划分、历史native报告13 complete/9 partial/2 rejected，以及两例起点缺口：RXN_0000155302为INPUT_GEOMETRY_INVALID，RXN_0000171675为ORIGIN_BUDGET_EXHAUSTED。原始化学复核记录仍有pending/awaiting状态，训练/执行资格与实际rollout数量另核验；原24名单不因失败换题。

计划顺序为24资格与接线预检、入口/环境/数据/TD/恢复费用修补、1→4 train真实冒烟、train16采集、BC/CQL离线训练与冻结、24×3同环境同预算对照及逐例报告。主扫描预算包络为88 episode×900秒=22小时串行上限（4线程预留88 core-hours）；制备/独立冒烟/训练/AB4/严格验证费用另列，不代表实测费用或完成时间。

本次仅新增方案并同步docs索引与本台账（v2.9）。未修改业务代码、配置、权重、受保护guard或既有运行报告，未启动QC/训练、未访问隔离TS/IRC真值，未生成可执行训练授权清单或升级研究/生产能力。接口字段的后续实现仍按既有版本化与审查流程登记；AB4门、正式阶段出口与默认保持原定义。

## 23. IRC20k与RL2000分阶段详细方案（2026-10-08）

用户在第一批结果审核后提出IRC2万个、RL2000个规模目标，并要求制定下一阶段详细方案。本次按独立canonical反应组解释IRC20000fit与RL新2000主episode，失败计尝试、未运行/重复分支/参数更新不代替数量。若单位另有意图，修改计划和预算，不混计。

新增[详细方案](PES2TS_IRC20k与RL2000下一阶段详细方案_v1_20261008.md)、[ADR-0019](../design/decisions/ADR-0019-IRC20k与RL2000分阶段扩量.md)及[不可执行规划清单](fixtures/irc20k_rl2000_campaign_plan_v1_20261008.json)。先补F1–F4、多episode/回放与教师加载资源缺口，再按IRC2k/5k/10k/20k、RL50/200/500/1000/2000分批；同反应支持不证明跨反应Q适用，迁移门和阈值只用fit内校准。

第一批BC模式纠错/权重有真实证据、CQL复现旧权重；独立定向测试27通过/1跳过，未重跑全量1966。context联合匹配、TD白名单、registry贯穿与审计缺源反例未关闭，因此不认可WP-2全部验收。更新上一轮计划v1.1、AGENTv3.8、docs索引和本台账v3.1，旧运行报告/模型不覆盖。

主QC预算为3877episode+156诊断宏步，串行上限982小时15分/468360请求；仅RL主采集是500小时/240000请求。离线训练、制备、严格验证及条件消融实测后另冻结并计入总成本。最先真实小包上限64小时15分/27720请求，不能把计划总包络当已启动任务或实际费用。

本次只修改规划与入口、完成文档/算术核验，未启动训练/QC、访问新增真值或修业务代码。宪法内容v2.6、正式阶段/能力/默认和权限保持；新schema实施前另按既有架构流程登记。改版前四份入口原字节与hash见[快照](../archive/20261008_pre_scale20k_rl2k/README.md)。

## 24. PES2TS-30H 并行执行套件（2026-10-20）

30h 战役（PES2TS_30H_R3_V52_V53）升级为并行研究执行模式。新增执行套件目录 [docs/agent_runs/](../agent_runs/)（非规范层）：[PARALLEL_EXECUTION_PLAN.md](../agent_runs/PARALLEL_EXECUTION_PLAN.md)（3-lane 并发/DAG/11+1 WP/串行 gate/合规映射/待决策清单）、[BOOTSTRAP_REVIEW.md](../agent_runs/BOOTSTRAP_REVIEW.md)（环境核查、3-lane dry-run 复核、Oracle ORA-1 预审处置）、state/parallel_config.json、state/task_state.json。配套脚手架 `scripts/agent_ops/{data_gate,feature_gate,ledger,gpu_queue}` 与 `tests/agent_ops`（70 项新测试）。

并行 dry-run（3 独立 worktree lane）已过；Oracle 预审 3 项启动阻塞已处置或列入 WP-L2 预启动硬化。未启动训练/QC；未触碰 sealed 目录与 data/ground_truth；未修改科研指标/阈值/损失/数据语义；旧 53-run 证据（outputs/pesn/v5_1 + v5_1_r1）只读。正式启动被 PD-1/2/4/6（宪法阻塞决策）+ WP-L2 + E0 冻结签认阻挡，`formal_training_authorized=false`。
