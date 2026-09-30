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

## 2026-10-01 todo 5 — EndpointGraphBundle + 表示归一化 — PASSED
- 新建 `pes2ts_core/g1/endpoint_graph.py`（EndpointGraphBundle frozen dataclass + build_endpoint_graph_bundle 纯函数）与 `pes2ts_core/g1/endpoint_materials.py`（白名单材料 schema 信任边界）+ `tests/test_endpoint_graph.py`（13 tests 全绿）。现有 g1 文件零改动（只增）。
- 双侧同一代码路径：复用 `g1/parse.py::parse_reaction`（removeHs=False），芳香键保持 1.5 不 kekulize，docstring 固化 `rdkit_default_unkekulized` + rdkit.__version__（本环境 2026.03.6）；禁用互变异构标准化（根本不调 MolStandardize/AddHs）。
- 材料 schema：顶层键仅 {r,p}；每原子键 {map,element,coordinates}；dict 按 map 键 或 list 带显式 "map" 字段。`MAP_INVALID`（非双射：extra/missing/duplicate map、materials 元素与 SMILES 不符）；`MAP_AMBIGUOUS`（order_based_binding_forbidden——禁止顺序绑定）；`MATERIALS_SCHEMA_INVALID`（未知键/坏坐标/缺 side）。
- 守恒：R→P 逐 map 元素比较 → `ELEMENT_IMBALANCE`；同位素 → `ISOTOPE_IMBALANCE`；显式 H 清点缺口（SMILES 侧或 materials 侧丢 H map）→ `H_INVENTORY_MISMATCH`（typed，绝不静默）。注意：`[CH3:1]` 括号 H 是隐式计数、不是图原子——`explicit_H_neighbors` 只数图上显式 H 邻居；测试 fixture 必须写 `[H:n]` 才有 H 清点语义。
- content_sha256：节点按 map 排序、边按 (min_map,max_map,type,order) 排序、stable_json_dumps；几何绑定 = SHA256 over centroid-centered 坐标（round 8 位）→ 刚体平移不变；材料按 map 键 → SMILES 原子顺序置换不变（测试：反转 SMILES 原子序哈希相等；平移 (5,-2,1.5) 哈希相等）。
- 真值防火墙：`to_doc()` 键扫描 vs `contracts.FORBIDDEN_TRUTH_KEYS ∪ v2_verify.FORBIDDEN_EXPORT_KEYS` = 0；数据类字段名同样扫描。
- SIZE：endpoint_materials.py 159 纯 LOC 达标；endpoint_graph.py 355 纯 LOC → `# noqa: SIZE_OK`（设计 §13.1 单模块图合同：冻结 schema+builder+身份；材料边界已拆出）。
- 全量套件：`python -m pytest -q` → **700 passed, 4 deselected**（666 基线 + todo3 21 + 本 todo 13；0 失败）。`conda run -n pes2ts python -m pytest -q tests/test_endpoint_graph.py` → 13 passed。
- Evidence: `.omo/evidence/task-5-pes-generation-graph-scan-strategy.txt`；Commit: `feat(g1): add EndpointGraphBundle and representation normalization`。
- 教训（给 todo 6-9）：下游只从 `g1.endpoint_graph` import（re-export 全部 codes/EndpointGraphError，与 materials 同一类对象）；材料适配（demo24 CSV + export 快照 → map 键 {element,coordinates}）是后续 todo 的职责；bundle 不存原始坐标（只存 geometry_ref SHA256）。

## 2026-10-01 todo 6 — R/P 完整键图重建（白名单输入）— PASSED
- 新建 `pes2ts_core/scan_strategy/graph_rebuild.py`（284 纯 LOC，SIZE_OK）+ `tests/test_graph_rebuild.py`（21 tests 全绿，纯合成 fixture 零读 data/）。现有模块零改动（含 scan_strategy/__init__.py —— MUST NOT 改既有模块，下游直接 import 子模块）。
- API：`rebuild_endpoint_graphs(smiles, materials) -> RebuiltEndpointGraphBundle`（is-a EndpointGraphBundle）——materials 经 `endpoint_materials.normalize_side_materials` 校验归一后委托 `endpoint_graph.build_endpoint_graph_bundle`（todo 5）；provenance 块绑定三哈希：`reaction_smiles_sha256`（字面串）、`endpoint_materials_sha256`（canonical sorted (side,map,element,coords) 行）、`graph_payload_sha256`（`graph_payload.py` map-space payload 的 string-key 投影）——graph_payload/parse 为 import 复用非拷贝。
- 教训（slots dataclass 子类）：`@dataclass(frozen=True, slots=True)` 子类里零参 `super().to_doc()` 报 `TypeError: obj must be an instance or subtype of type`——decorator 重建类对象、孤儿化方法的 `__class__` cell；必须显式 `EndpointGraphBundle.to_doc(self)` 并留注释。
- 语义分层（测试钉死）：`content_sha256` = 化学身份（模板原子序/刚体平移不变）；provenance 材料哈希 = 字面源绑定（平移后改变）；SMILES 哈希 = 字面串（shuffled 模板后改变）。
- Loader：`load_endpoint_materials_from_export(doc)` schema 通用（maps + r/p_coordinates + atom_rows|elements 回退），只读白名单字段、忽略 doc 其余键；类型化 `EXPORT_SCHEMA_INVALID`。真实 doc 冒烟：export_contracts_v1/00000/RXN_0000000001.json（12 maps）loader 通过（手工终端冒烟，非测试——测试禁读 data/）。
- 零读保证：模块源静态断言无内部数据树引用字符串；测试 monkeypatch `Path.open`+`builtins.open`（路径含内部数据树段则抛）跑全量重建 + 注入探针证伪守卫已武装。
- 纯度：递归扫描 to_doc()+dataclass 字段名 vs FORBIDDEN_TRUTH_KEYS∪FORBIDDEN_EXPORT_KEYS 及具名 {endpoint_match,orientation,irc_evidence} = 0 命中。
- 全量套件：`python -m pytest -q` → **730 passed, 4 deselected**（700 基线 + 本 todo 21 + 并行 todo4 worker 的 9 个 cli 测试——cli.py/scan_strategy/cli.py/test_scan_strategy_cli.py/config/defaults.yaml 属 todo 4，未入本提交）。
- Evidence: `.omo/evidence/task-6-pes-generation-graph-scan-strategy.txt`；Commit: `feat(scan-strategy): rebuild full R/P bond graph from whitelisted SMILES`。
- 教训（给 todo 7-9）：直接 `from pes2ts_core.scan_strategy.graph_rebuild import rebuild_endpoint_graphs`（不改 __init__）；demo24 全量冒烟推迟到 todo 10 golden（API 已 batch-capable，测试 21 证明）；下游消费 bundle.r_graph/p_graph/conservation/content_sha256，provenance 只在 to_doc() 文档层。

