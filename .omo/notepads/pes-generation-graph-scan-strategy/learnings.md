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

## 2026-10-01 todo 4 — CLI + 产物树骨架（v2-scan-plan/verify/freeze）— PASSED
- 新建 `pes2ts_core/scan_strategy/cli.py`（scan_plan_proposals / verify_scan_proposals / freeze_scan_plans + 4 frozen dataclass）+ `tests/test_scan_strategy_cli.py`（9 tests 全绿）；`config/defaults.yaml` scan_strategy 段追加 7 个产物路径键（additive，既有键逐字节不变）；`pes2ts_core/cli.py` 按 5 处注册模式接入（G1_SUBCOMMANDS 尾部追加三个、3 个惰性 import handler 返回 int、G1_SUBCOMMAND_HANDLERS + g1_help 各 3 条）。既有 v2-* handler 零改动；bin/pes2ts 零改动。
- 产物路径 idiom：config 键为相对路径——proposals_dir/plans_dir/proposal_summary/plan_summary 挂 `paths.interim`，三个 manifest 挂 `paths.manifests`（与 g1_v2 的 `interim/g1_v2/<subdir>` 同构）。默认值：`g1_v2/scan_proposals`、`g1_v2/scan_plans`、`g1_v2_scan_{proposal,plan}_summary.parquet`、`g1_v2_scan_{proposal,plan}_manifest.json`、`g1_v2_scan_freeze.json`。
- 空输入 plan：写 proposals 空目录骨架 + 空 summary parquet（固定列集，write_parquet 按列名排序）+ manifest（n_total=0/n_files=0/summary_sha256）。确定性：write_json=stable_json_dumps，两次运行 manifest 去 generated_at 后哈希相等、summary parquet 字节相等（实测 True/True）。非空输入走 `NotImplementedError`（todo 17 选择器接线），绝不静默写空壳冒充提案。
- verify：rglob 排序遍历 proposals 树 → 每文档 `validate_v2_document` + `seal_document` digest 复核 → manifest n_total/n_files/summary_sha256 对账。干净 exit 0 `problems=0`；脏树 exit 22（EXIT_G1_BUILD_FAILED）列出问题（handler 截前 10 条）。注入非法文档实测 problems=3（文档校验 + n_total 不符 + n_files 不符——一次注入会同时打破三个对账项，测试断言要按合并计数写）。
- freeze 门控纪律（关键设计决定）：先跑 verify；gate = verification 干净 **且** ≥1 个 execution_eligible 提案。**contracts_v2 强制所有提案 execution_eligible=False**（validate 会拒绝 True），因此当前 gate 永远不过——空输入冻结路径稳定落到 `plan_gate_pass=false` + typed reason `NO_ELIGIBLE_PROPOSALS`（脏树再加 `VERIFICATION_FAILED`）。这是正确行为非 stub：冻结导出（g1_generation_plan_v2 文档 + plan manifest + plan summary）留给 todo 23；gate-pass 分支显式 `NotImplementedError` 而非写假导出。拒绝时只写 freeze manifest + plans_dir 空目录骨架，零 json 文件、零 plan manifest（`gate_pass=false` 不暴露可消费导出）。
- CLI handler 输出走 print（与既有 v2-* handler 一致），错误走 logging；handler 内不 import scan_strategy 模块级依赖（惰性 import 在函数体）。
- SIZE：scan_strategy/cli.py 267 纯 LOC → `# allow: SIZE_OK` 标记（plan 点名的单一 CLI-stage 模块，g2/pipeline.py 同型先例；todo 23 在本模块内扩展 freeze 而非拆分）。
- 测试模式：tmp_path re-root 仿 `test_g2_cli.py::_yaml_config`（load_config() → 改 paths.interim/manifests → yaml.safe_dump → main(["--config",...])）。`--help` 测试用 `pytest.raises(SystemExit)`（argparse help 走 SystemExit(0)，不经过 load_config）。
- 全量套件：**730 passed, 4 deselected**（基线 700 + 本 todo 9 + 并行 todo6 worker 的 21；0 失败）。lsp error 级清零。
- Evidence: `.omo/evidence/task-4-pes-generation-graph-scan-strategy.txt`；Commit: `feat(cli): add g1 v2-scan-plan/verify/freeze and artifact trees`。
- 教训（给 todo 11/17/23）：(1) `_proposal_inputs()` 是 todo 17 的接线点，当前恒返回 []；(2) freeze gate 的"eligible"判定读提案文档 `execution_eligible is True`——todo 17 若需要 gate 通过，必须先解决合同层 execution_eligible 强制 False 的矛盾（合同演进或引入独立的 release-gate 字段，见设计 §5.1 释放门语义）；(3) 注入一个脏文档会同时打爆 verify 的多个对账项，测试断言 problems 计数时要合并计算。

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


## 2026-10-01 todo 8 — 端点上下文图 — PASSED
- 新建 `pes2ts_core/g1/endpoint_context.py`（876 纯 LOC → `# noqa: SIZE_OK`，plan 点名的 §4.2 单模块）+ `tests/test_context_graph.py`（23 tests 全绿）。现有模块零改动；无新依赖（requirements 无 networkx，union-find/BFS/DFS low-link 全手写 stdlib）。
- **EditGraph 接口假设（给 todo 9/10 对齐）**：todo 7 的 `reaction_edit_graph.py` 在本 todo 执行时**尚未落地**。模块以 duck-typing 消费：`edit_graph.edits` 序列，每条记录提供 `.pair`（或 mapping 键 `pair`）、`.edit_kind`（`"formed"|"broken"|"order_changed"` 字符串）、`.r_bond_order`/`.p_bond_order`（float|None）；`edit_graph.atom_events` 原样透传到 `EndpointContext.atom_events`。模块级 `try: from ...reaction_edit_graph import EditGraph except ImportError: EditGraph=None`，`__all__` 导出 `EditGraph`/`EditGraphProtocol`/`EditRecordProtocol`。若 todo 7 落地后字段名/枚举不同（如 kind 用枚举而非 str），在 todo 9 接线时于 `_rec_field`/kind 分支处适配，并回写本 notepad。
- 语义移植（build_evidence.py 逐字）：`shortest_support_path` = BFS 路径队列 + `sorted(adj)` 邻居扩展、路径含两端、不可达返回 None；formed → `support_path_R`（R 图），broken → `support_path_P`（P 图），order_changed 两者皆无（evidence key presence 语义由 `evidence_edit_fields` 复刻：formed 只出 `support_path_R` 键（跨组分时值为 null），broken 只出 `support_path_P`，distance 仅在提供坐标时出现）。`cross_component_R/P` = 该侧 component_id 不等（与 evidence atom_rows 比较等价——partition 不变量）。距离 `round(math.dist(...),6)`。
- **几何坐标来源**：todo 5 的 bundle 只存 `geometry_ref.coordinates_sha256`，不存坐标本体。`build_endpoint_context(..., r_coordinates=, p_coordinates=)` 由调用方显式传 map→xyz（review case 的 `reactant/product.geometry` 按 atoms 顺序对齐 atom_map_id）；charge/multiplicity 经 `EndpointElectronic` 传入。证据交叉验证 6 条 Demo24 记录（7104/161724/77619/26256/102998/100736）support_path/cross_component/distances/endpoints 七键逐条一致（RXN id 为 `RXN_`+10 位零填充，如 `RXN_0000077619`——11 位填充查 CSV 会误报 missing）。
- 环证据分组（两根不相交新键 → 同一环化）：formed 用 P 图、broken 用 R 图；边在该侧为桥 → region=None；否则 region=含该边的 2-edge-cc 原子集。并查集 union：region 相交（basis `shared_ring_region`）或**同 kind** 且双方 support-path 原子集相交（basis `support_path_overlap`）。跨 kind 不因共享原子并组（H 转移不误并成 RING）。order_changed 不参与分组。合成 fixture：两 disjoint formed 闭环丁烷 → 1 组；两个独立闭环 → 2 组；链上两条新键支持路径交叠 → 1 组；真实 102998（formed 2-7/4-8 + broken 2-4）证据路径逐条复现。
- 拓扑：环尺寸 = BFS 生成树基本环多重集（确定性、**基依赖**——绝不作唯一 SSSR 分类，配套 `in_cycle`（属 ≥3 元素块）、双连通块（桥以 {u,v} 块出现，标准 Hopcroft–Tarjan 分解）、桥、`cycle_rank=|E|-|V|+C`（分量级+总级）、2-edge-cc。`bridge_edges_R/P` = 任务定义（一侧边存在、对侧 component 结构不同）；`graph bridges` 存于 `SideTopology.bridges`。芳香区域=芳香原子连通簇；boundary=区域外相邻原子；periphery=区域边缘芳香原子。旁观组分=无编辑原子的 bundle 组件（map 元组）。`context_radius` 默认 2（`config/defaults.yaml scan_strategy.context_radius`），`context_radius_from_config` 非法值回退 2；radius 1 vs 3 不改变 support/cross/endpoints/ring_groups/bridges（测试钉死），仅 shell 增大。
- 教训（给 todo 9/10）：(1) `_spectators` 返回 tuple-of-tuple（早期返回 tuple-of-list 被测试抓出）；(2) 双环夹桥 fixture 的双连通块含桥块 (6,7)——标准分解三块，勿期望仅两块；(3) shell 测试需稀疏 seeds（长链跨组分 fixture），密集 seeds 时 radius1 已覆盖全图；(4) bundle 组件编号按最小 map 排序，`component_of` 直接读 bundle.components 不重算，避免编号漂移。
- 全量套件：`python -m pytest -q` → **753 passed, 4 deselected**（730 基线 + 本 todo 23；todo 7 模块/测试执行时仍未落地，全套件无 reaction_edit_graph import）。lsp error 级清零；truth_guard 全绿（模块无真值字符串）。
- Evidence: `.omo/evidence/task-8-pes-generation-graph-scan-strategy.txt`；Commit: `feat(g1): add endpoint context graph`。

## 2026-10-01 todo 7 — 原子级反应编辑图 ΔG — PASSED
- 新建 `pes2ts_core/g1/reaction_edit_graph.py`（288 纯 LOC → SIZE_OK，先例 endpoint_graph.py/graph_rebuild.py）+ `tests/test_reaction_edit_graph.py`（15 tests 全绿）。现有模块零改动（只增）。
- API：`build_reaction_edit_graph(bundle, aromatic_regions=None) -> ReactionEditGraph`；输入是 todo 5 的 `EndpointGraphBundle`（r_graph/p_graph 边为排序 map 对+bond_order+aromatic 标志）。F/B/O 互斥归属与 v2 合同一致：升键级只记一次 O，绝不同时计 F/B。
- 芳香 1.5 语义：`1.5 if edge.aromatic else bond_order`（与 build_evidence.py graph()、v2_edits AROMATIC_ORDER 逐字一致）。H 伙伴变化自然以 broken/formed 出现在 (H,X)/(H,Y) 对上——无独立 H 路径。
- `aromatic_region` 只透传调用方提供的区域映射（导出形 `{region_id: [[a,b],...]}` 或 pair 形 `{(a,b): id}`），未提供则 null；区域分组本身是 todo 9/v2_edits 领地，本模块绝不重算。`getattr(bundle, "aromatic_regions", None)` 回退允许扩展 bundle 自带区域。
- 证据字节等价：`edit_counts` 用 `dict(Counter(kind))`（缺省 kind 不出现键，与证据 JSON 一致——如 RXN_0000109608 只有 formed/order_changed）；`atom_attribute_changes` 逐字段 `{atom_map_id, R/P {formal_charge, radical_electrons, aromatic, chiral_tag}}`（bundle 的 stereo 字段映射为证据键名 chiral_tag）。元素/同位素变化在有效 bundle 中不可能（上游守恒抛错），故属性集与 build_evidence.py 完全一致。
- 分量算法：忠实移植 build_evidence.py `components()`——从最小未见顶点 DFS、邻居排序展开、分量按 min map 排序。`edit_cycle_rank = |E|-|V|+C` 只对 F/B/O 编辑图；`connectivity_edit_components` 只含 F/B 边（O 桥不计入——测试钉死：F(1,2)+O(2,3)+B(3,4) → edit_cc=[[1,2,3,4]] 但 fb_cc=[[1,2],[3,4]]）。
- todo 8 接缝：`ReactionEdit.to_record()` 正好暴露 6 个金标字段的普通 dict，todo 8 复制后追加 distance_R_A/cross_component_*/support_path_* 即可；本模块刻意不算这些上下文字段。数据类设计允许后续加性扩展。
- Demo24 交叉验证（数据树存在 → 实跑非 skip）：候选 CSV SMILES + export_contracts_v1 导出经 `graph_rebuild` 重建 bundle，`edit_counts`/`edit_components`/`connectivity_edit_components`/`edit_cycle_rank`/`atom_attribute_changes` + 全部 edits[] 六字段逐条字节等价（24 条全比对，门槛 ≥3）。证据 export_sha256 与 export_contracts_v1 树匹配。
- R/P 交换性质：手-built 图 F/B 对调 + SMILES 反向书写双路验证——order_changed 方向不变（r/p 订单互换），F↔B 镜像，分量/cycle_rank 不变。注意测试断言方向：R 侧独有的键是 broken，P 侧独有是 formed（首版测试曾写反，已修正）。
- 确定性：同输入两次 `stable_json_dumps(to_doc())` 字节相等；纯度：to_doc 键扫描 vs FORBIDDEN ∪ = 0。
- 全量套件：**768 passed, 4 deselected**（730 基线 + 本 todo 15 + 并行 todo8 worker 的 23 个 context 测试——endpoint_context.py/test_context_graph.py 已由该 worker 提交）。lsp error 级清零。
- 教训（给 todo 9/10）：(1) edit_counts 是 Counter 语义——缺省 kind 无键，golden 比对用 dict 相等即可，不要补零；(2) 芳香区域 id 从导出透传，todo 9 的区域分组必须产出同形 `{region_id: [[a,b],...]}` 才能无损对接；(3) todo 10 golden 可直接复用本模块的 to_doc()['edits'] 投影与 features 四字段；(4) 手-built bundle 测试 fixture 用 `_hand_bundle(r_specs, p_specs, ...)` 形态（边列表+可选 per-side 属性覆盖），比 SMILES 更精确控制分量拓扑。

## 2026-10-01 todo 9 — 事件耦合图 H — PASSED
- 新建 `pes2ts_core/g1/event_coupling.py`（864 纯 LOC → `# noqa: SIZE_OK`，plan 点名的 §4.3 单模块，先例 endpoint_context/contracts_v2）+ `tests/test_event_coupling.py`（12 tests 全绿）。todo 7/8 模块零改动（只读 import；无适配器需求——`ReactionEditGraph`/`EndpointContext` 字段直接消费）。
- **H API（给 todo 11 路由接线）**：`build_event_coupling_graph(bundle, edit_graph, context) -> EventCouplingGraph`；关键消费面：(1) `events_by_type(EVENT_H_TRANSFER|EVENT_H2|...)` 按类型取事件节点（motif 匹配输入）；(2) `strong_components` —— 每个强分量是一个路由/坐标生成单元（union-find 仅在 strong 链接上：`shared_edit_membership` + `shared_center`；CONTEXT_NEAR 弱联系绝不合并，远端两编辑不同步）；(3) `hydrogen_events` 金标投影（{from,from_state,h,kind,to,to_state}，驱动 H_TRANSFER/H2 策略）；(4) `aromatic_regions` 金标投影（{region_id: [[a,b],...]} 或 {}，AROMATIC_COUPLED 一区域一事件）；(5) `membership_map()` edit pair→event ids（todo 12 event_coverage 的分母）；(6) `weak_links` 仅作监测提示，禁止进入驱动选择。
- 七类事件/联系（§4.3）：H_TRANSFER（每 H 伙伴变化一事件，kind∈{transfer,release,capture}）、H2_EVENT（HH_EVENT_KINDS 五种，按 H–H 无序对簇去重，绝不标成 H 转移）、CONNECTIVITY_EXCHANGE（成+断共享非 H 中心原子，union-find 分组）、RING_REORGANIZATION（todo 8 ring_groups 一组合一事件）、DELOCALIZED_REGION（v2 花香区域一区域一事件；孤立 order_changed 单例同型退化成员）、SHARED_CENTER（遗留编辑按共享中心原子分组 + 事件间强联系规则）、CONTEXT_NEAR（仅弱联系）。每条规则返回 `rule_id/support_atom_maps/support_edges/evidence_level`。
- **金标端口**：`hydrogen_events_from_bundle` = bundle 图 orders（aromatic 1.5）+ elements → `v2_edits.hydrogen_partner_changes`；`aromatic_regions_from_bundle` = 同输入 → `v2_edits.exclusive_edits()['aromatic_regions']`（region id = sha256(repr(members))[:12]）。24/24 Demo24 记录与证据 JSON `features.hydrogen_events`/`features.aromatic_regions` 逐条相等（数据树在场实跑非 skip）。
- 成员注册表：每 edit pair 恰一主归属（优先级 H > EXCHANGE > RING > DELOCALIZED > 遗留单例）或显式共享（额外归属带类型化 justification：`aromatic_region_membership`/`ring_group_membership`/`h_partner_change`/`shared_center`）。多归属 edit 的共属事件间自动发 `shared_edit_membership` 强联系 → 同强分量。
- 遗留编辑定型（Demo24 未覆盖的全是 order_changed）：≥2 个共享中心原子 → SHARED_CENTER 事件；单例 order_changed → DELOCALIZED_REGION（rule_id `order_change_singleton`，设计 §4.3 "O 或电荷重排" 类；aromatic_region=None 时 todo 11 应走无 F/B 分支而非 AROMATIC_COUPLED）；单例 formed/broken → CONNECTIVITY_EXCHANGE（rule_id `connectivity_single`）。
- 测试锁：两分离 H 转移 → 2 强分量零耦合记录；近邻不相交 H 转移（radius 2 壳交叠）→ CONTEXT_NEAR 弱联系存在但分量不合并；H2 形成（to_hh）→ H2_EVENT 非 H_TRANSFER；苯 6 个 O 编辑 → 1 个 DELOCALIZED 事件；共享中心 1 → SHARED_CENTER 联系 + 同分量 + 多归属 justification；注册表全覆盖；确定性 stable_json_dumps 字节相等；to_doc 零真值键。
- 全量套件：**780 passed, 4 deselected**（768 基线 + 本 todo 12）。lsp error 级清零。
- Evidence: `.omo/evidence/task-9-pes-generation-graph-scan-strategy.txt`；Commit: `feat(g1): add typed event coupling graph`。
- 教训（给 todo 10/11/12）：(1) todo 8 的 ring grouping 会给每个有 support path 的 H 键编辑建 singleton ring group——多归属合法（ring_group_membership justification），测试断言"2 events"时按 H_TRANSFER 计数/强分量数断言，勿断言事件节点总数；(2) v2 花香区域 id 确定性依赖 repr(members) 排序——bundle 路径必须用 sorted pairs 才能复现金标；(3) HAND-built bundle 测试 fixture 必须填充 `components`（build_endpoint_context 的 `component_of` 会 KeyError），照抄 todo 7 的 `_hand_bundle` 并加 union-find 组件计算；(4) 弱联系支持原子用 R∪P 并集骨架 BFS 壳交集，radius 取 `context.context_radius`。

## 2026-10-01 todo 10 — P0 金标验收（Demo24 复现 + fixture 冻结）— PASSED
- 新建 `tests/test_p0_demo24_golden.py`（7 tests 全绿）+ 冻结 fixture `tests/fixtures/p0_demo24/`（228K：CSV 字节拷贝 + 24 份最小快照 + manifest.json + README.md）。生产模块零改动。
- CSV 冻结前 sha256 校验：`63ac74ee...` == 证据 JSON `source_csv_sha256`（`data/manifests/*.csv` 被 gitignore，fixture 拷贝是入库输入）。manifest 另记 `source_manifest_sha256`（demo24_reaction_case_manifest_v1.json，本身被 `!data/manifests/*.json` 例外跟踪）。
- 最小快照字段（仅 P0 白名单）：maps/elements/r_coordinates/p_coordinates/atom_rows/components/aromatic_regions/hydrogen_partner_changes + `endpoint_electronic`（case 的 charge/multiplicity，fixture 构建时与 export totals 交叉断言）+ `export_sha256`/`reaction_smiles`/`mapping_provenance`。不提交完整 export/case 文档。
- 金标管线（仅 fixture + git 跟踪的 evidence JSON，零读 data/）：`load_endpoint_materials_from_export` → `rebuild_endpoint_graphs` → `aromatic_regions_from_bundle` → `build_reaction_edit_graph` → `build_endpoint_context`(coords + EndpointElectronic) → `build_event_coupling_graph`。features 八字段逐记录 EXACT；edits[] 按 pair 索引逐字段比对（support_path 键存在语义 + distance round-6）；hydrogen_events/aromatic_regions 三重绑定（recomputed == snapshot 副本 == evidence）。
- **membership 语义教训（给 todo 11/12）**：`MembershipRecord.event_ids = tuple(sorted({primary, *extras}))`——**按字母序，primary 不是 [0]**；justifications 只覆盖 extras（每条带 typed rule_id ∈ {aromatic_region_membership, ring_group_membership, shared_center, h_partner_change}）。正确断言：共享行 `len(justified)==len(event_ids)-1` 且 `justified ⊊ event_ids`，无 justification 的唯一 id 即 primary。首版误断言 `event_ids[1:] ⊆ justified` 在 RXN_0000047010 (5,6) 失败（event_ids=['DELOCALIZED...', 'RING...'] 但 primary=_RING_）。
- 纯度：bundle/edit_graph/context/coupling 的 to_doc() 递归键扫描 vs FORBIDDEN_TRUTH_KEYS ∪ FORBIDDEN_EXPORT_KEYS ∪ {endpoint_match,orientation,irc_evidence} = 0。绑定：snapshot.export_sha256 == manifest 映射 == evidence；篡改测试（内存 + tmp_path 拷贝，不碰 fixture）断言 binding.export_sha256 失败。离线保证：monkeypatch open/Path.read_text/read_bytes 拒绝任何 data/ 路径跑满 24 条管线。
- RDKit pin：fixture manifest `rdkit_version=2026.03.6`（仓库参考 env），`evidence_rdkit_version=2026.03.3`（证据构建环境）；运行 rdkit ≠ pin → pytest.fail（never-skip）。两 pin 间图特征稳定（todo 7/9 交叉验证 + 本金标在 2026.03.6 下复现）。
- **PROPOSALS 纪律**：evidence JSON 的 records[].proposal 与 build_evidence.py 的 PROPOSALS 表是 2026-09-30 设计期人工假设，**不是选择器输出**（选择器 todo 17 才到）。金标测试不读、不断言 proposal 字段；fixture README 写明此边界。
- 全量套件：**787 passed, 4 deselected**（780 基线 + 本 todo 7）。golden 单文件 `pytest -q tests/test_p0_demo24_golden.py` → 7 passed（24/24 记录覆盖在 golden 测试内）。
- Evidence: `.omo/evidence/task-10-pes-generation-graph-scan-strategy.txt`；Commit: `test(p0): golden Demo24 evidence reproduction and frozen fixture`。
- 教训（给 todo 11+）：(1) 金标离线重算路径用 bundle 重算的 aromatic_regions/H 事件（event_coupling 端口），不直接抄 export 字段——三重断言同时锁住重算与冻结副本；(2) fixture 再生脚本放 /tmp 不入库，README 描述再生步骤即可；(3) evidence JSON 在 outputs/pes_generation_strategy_design_v1/ 且被 git 跟踪（`!outputs/pes_generation_strategy_design_v1/` 例外），金标可直接读。
