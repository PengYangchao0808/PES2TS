# learnings

## 2026-09-30 session start
- env: conda pes2ts, python 3.12.14, rdkit 2026.03.6
- data tree present: data/manifests/g1_v2_gate.json gate_pass=true (n_denominator 199217)
- evidence naming: .omo/evidence/task-<N>-pes-generation-graph-scan-strategy.txt

## 2026-09-30 todo 1 L0 closure verification — FAILED (stale export tree)
- v2-verify exit 22: every export doc in `data/interim/g1_v2/export/` carries `status_summary.orientation` + `status_summary.endpoint_match` (forbidden keys). Handler printed 10 capped errors, all `forbidden truth-derived key 'orientation'`.
- Root cause: the on-disk `export/` tree is stale — written by an older v2_gate.py that embedded IRC-derived `orientation`/`endpoint_match` inside `status_summary`. Current `pes2ts_core/g1/v2_gate.py` code is clean: `status_summary` holds only `p1_status` + `audit_status` (comment at lines 164-166 explicitly excludes orientation).
- `data/interim/g1_v2/export_contracts_v1/` (sanitized tree, the plan's sample target) is CLEAN: 100-file recursive-key sample (seed 42, shard-dir per-reaction exports only) → 0 violations of `endpoint_match`/`orientation`/`irc_evidence`.
- Sanitize manifest confirms: `removed_truth_key_occurrences: {endpoint_match: 183460, orientation: 183460}` — the sanitizer stripped both keys from all 183,460 exports.
- QA injection check PASSED: copying a real export from shard 00000, injecting `"orientation":"R_first"` → recursive-key check flags it (hits=['orientation']); original export had 0 forbidden hits.
- Current `data/manifests/g1_v2_gate.json`: gate_pass=true, generated 2026-09-27, 0 refusals, 183460 scan_ready — stale relative to the dirty export/ tree now on disk.
- v2-verify runtime: ~1h50m on /mnt/e/ (WSL9p mount), ~2.2GB read, ~1.1M read syscalls. Use nohup + direct conda python (`/root/miniforge3/envs/pes2ts/bin/python`) to survive tool timeouts; `conda run` heredoc swallows stdin.
- Per failure-stop rule: did NOT write success evidence file, did NOT commit. Re-running `g1 v2-gate` would regenerate `export/` with the current clean code, then a second verify/gate pass should clear — but that is a follow-up decision, not done here.

## 2026-10-01 todo 1 resumed — PASSED after stale-tree remediation
- Orchestrator confirmed diagnosis; remediation sequence executed (no code changes):
  1. `g1 v2-gate --assume-verified` → regenerated `export/` (rmtree + rewrite 183,460 docs), exit 0, ~1h38m
  2. `g1 v2-verify` → `problems=0`, exit 0, ~1h41m (total=199217 clean=183460 issues=7688 excluded=8069)
  3. `g1 v2-gate` (full) → exit 0, gate_pass=true, refusals=0, ~3h22m
- Gate manifest: `data/manifests/g1_v2_gate.json` sha256 `9d991a624d5c5e4d6800af6b4068676d9d887e695623fdd5efbb12d1c9af8be5`, generated_at 2026-09-30T20:43:21+00:00
- v2-verify writes NO manifest artifact (cli.py handler prints summary only) — receipt records sha256 as N/A for verify
- Dual-tree 100-file recursive sample (seed 42, shard-dir exports only): `export/` 0 violations, `export_contracts_v1/` 0 violations, TOTAL 200/0
- QA injection re-check: real export copied → inject orientation → recursive checker flags it (hits=['orientation'])
- Evidence: `.omo/evidence/task-1-pes-generation-graph-scan-strategy.txt`; committed as `chore(g1): record L0 closure verification receipt`
- Timing note: on this /mnt/e WSL 9p mount, verify ≈ 1h40m / 2.2GB read / ~1.1M syscalls; gate regen ≈ 1h40m for 183k writes; full gate (verify+rewrite) ≈ 3h20m. Always nohup + direct conda python; poll via /proc/<pid>/io + export file count.
- Lesson: `export/` (gate output) and `export_contracts_v1/` (sanitized copy) are separate trees; the sanitize manifest at export_contracts_v1 root records removal counts using the forbidden key names — exclude root-level meta files when sampling per-reaction exports.

## 2026-10-01 todo 2 — 接口承诺 + 谱系登记 + 配置/pytest 标记基础 — PASSED
- config/defaults.yaml: new top-level `scan_strategy` section inserted between `g1_v2` and `g2`; existing keys byte-identical (additions only). Thresholds from design doc §6/§8/§9: context_radius=2, max_scan_coordinates=3, point_limits{baseline:9,max:101}, schedule_budget=3, direction_budget=2, assembly_candidate_budget=2, max_total_candidates{scan:6,path_neb:1}, max_step_by_kind{distance:0.2,angle:10.0,dihedral:10.0}, jacobian_condition_number_max=1.0e6, element_pair_bond_thresholds.tolerance=0.45, special_domain_policy=unsupported. All marked `# TODO: calibrate` (待校准).
- pytest.ini: appended `acp` (PES2TS_ACP_ROOT + ACP python) and `orca` (PES2TS_ORCA_EXECUTABLE) markers; realdata/xtb lines untouched; addopts now `-m "not realdata and not xtb and not acp and not orca"`. pytest --markers prints them as `@pytest.mark.acp:` / `@pytest.mark.orca:` (grep for `^acp:` FAILS — use `pytest.mark.acp` pattern).
- 7 old G1_v2 design docs: one-line merge note `> 已并入图谱选择器 P0–P3 (见 ...)` inserted as line 3 (first blockquote after H1), rest unaltered.
- v2_gate.py docstring L18-19: "G2 may choose its own scan method" → "G1 freezes the scan method and candidate plan per reaction; G2 executes it..." (English, consistent with file style). grep for old phrase returns empty.
- collect-only: 666/670 collected, 4 deselected (unchanged gated realdata/xtb families), exit 0. No tests use acp/orca yet.
- Evidence: `.omo/evidence/task-2-pes-generation-graph-scan-strategy.txt` (force-add required, .omo/ gitignored).
- Commit: `chore(scan-strategy): add interface commitments, policy section, and gated markers` (single commit per plan todo-2 acceptance + plan note "接口承诺表（todo 2）为单独提交").
- Lesson: pytest 9.1 `--markers` output uses `@pytest.mark.<name>:` format, not bare `<name>:` — adjust grep patterns in future marker checks.

## 2026-10-01 todo 3 — contracts v2 模块（提案/计划/能力 + 身份密封）— PASSED
- 新建 `pes2ts_core/scan_strategy/{__init__,contracts_v2}.py` + `tests/test_scan_strategy_contracts_v2.py`（21 tests 全绿）。contracts.py 逐字节未动。
- 身份密封直接 `import` `contracts.seal_document`（volatile 语义 v1/v2 完全一致：created_at/generated_at/producer 排除，content_sha256 稳定）；`stable_json_dumps` 来自 `utils/hashing.py`。测试：同输入两次哈希相等、改 drivers 哈希变、改 created_at 哈希不变。
- FORBIDDEN 单源：`contracts_v2.FORBIDDEN_KEYS = lower(FORBIDDEN_TRUTH_KEYS) ∪ lower(v2_verify.FORBIDDEN_EXPORT_KEYS)`——import 无环（g1 不 import scan_strategy），truth_guard 不报（无 ground_truth/truth_sources 字符串）。validate 对 v2 文档**始终**检查真值键（比 v1 的 production_input 门更严）。
- 判别联合：plan.candidates 按 `candidate_kind` ∈ {ScanCandidateV2, PathCandidateV1}；跨种污染键被拒（scan-only: mode/drivers/lambda_values/schedule_id/schedule_kind/assembly_id；path-only: endpoint_geometries/image_chain/method_kind）。SINGLE_1D 恰 1 driver；COUPLED/SCHEDULED 2..3（MAX_SCAN_DRIVERS=3 对齐 config）。PathCandidateV1 最小形：n_atoms + 双端 N×3 几何（行数==n_atoms）+ image_chain.n_images>=1；todo 26 再展开。
- Proposal：execution_eligible 强制 False（传 True 抛 ContractError）；needs_review 允许（reasons 必填）；rejected 需 blocking_reasons；候选级 failure_reasons 与全案 blocking_reasons 分开存；epistemic_status 固定 endpoint_hypothesis。提案阶段候选仅 scan 模式（path 提案随 todo 26）。
- BackendCapability = orca_capabilities_v1.json 形：engine/adapter_version、supported_modes⊆{SINGLE_1D,COUPLED_1D,SCHEDULED_1D,PATH_NEB}、coordinate_kinds⊆{B,A,D}、point_limits{baseline<=max}、constraint_support 三布尔、method_element_coverage、probe_receipts。effective_capability 留给 todo 16 计算（不在合同内）。
- 全量套件：700 passed, 4 deselected（666 基线 + 本 todo 21 + 并行 todo5 worker 的 test_endpoint_graph.py 13 个——其文件未入本提交）。lsp error 级清零（_is_int 等改 TypeGuard）；reportAny 警告与 contracts.py 同型，仓库无 basedpyright CI，门是 pytest。
- 教训：`.omo/notepads/.../learnings.md` 已被 git 跟踪（早前 force-add），更新需随提交；evidence 仍需 `git add -f`。contracts_v2.py ~850 纯 LOC 超 250 上限——SIZE_OK：任务显式单文件合同 + contracts.py（559 纯 LOC）先例。
