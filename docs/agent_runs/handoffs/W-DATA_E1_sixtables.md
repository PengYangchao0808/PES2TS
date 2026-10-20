# W-DATA handoff — E1 R3 six-table dataset (G0 evidence)

- task_id: E1-DATA · lane: W-DATA · branch `ops/e1-data` · campaign PES2TS_30H_R3_V52_V53
- Deliverable: R3 six-table dataset built at **main tree** `outputs/pesn/30h_campaign/data/` (grant: e1_execution_grant); builder at `scripts/agent_ops/data_gate/r3_build_tables.py` (plan-doc path scripts/campaigns/r3_build_data.py stays W-LOSS per frozen allowlist; relocation already recorded in task_state).
- Changed files (this branch): `scripts/agent_ops/data_gate/r3_build_tables.py` (new), `scripts/agent_ops/data_gate/__init__.py` (docstring), `tests/agent_ops/test_data_r3_build.py` (new); data outputs under the grant dir only.

## Manifest SHAs (real, deterministic; supervisor binds at G0)
- `data_manifest.json` sha256 = `c21ff84b3790282b4c60764e6a356b8cb3c46e5f3b76f2bf785eeb8e70a56cbc`
- `teacher_manifest.json` sha256 = `bc02a2bd6f6496dff32d04533ca233a1c8175779410295e2b136d17ee89ce26a`
- `split_manifest_r3.json` sha256 = `7cc1f98e5d28077ae202eb8bf04554c53647f9c9467f0877c34365850b055879`
- dataset content digest (`dataset_sha256` inside data_manifest) = `4635c330e94f59ad80f8da51c996923e1e1f9d236b5a8d8def293bd4af2dfd84`
- run_plan binding suggestion: `data_sha256`←data_manifest file sha, `teacher_sha256`←teacher_manifest file sha, `split_sha256`←split_manifest file sha (protocol `pending_artifacts_two_stage` convention). Rebuild reproduces all three shas byte-identically (tested).

## Lineage (SHA-bound in build_ledger_r3.json)
- Trajectory: `outputs/pesn/dual_ts_v2_20261015/wp2_fit50/RXN_*.{task,teacher}.json` — coverage-audited dual-branch IRC export (chained mass-weighted Kabsch drift rule, a_max=0.1 Å insertions, TS frames exactly shared, node count == coverage_summary 6017). fit50 set == selection_direct_fit50 (50).
- K + R/P sides: `data/interim/g1_truth/p1_mapping/**` via frozen `build_key_atom_task` (bond_events key atoms; endpoint side matching with recorded evidence). Row space verified = wp2 identity order.
- Split: inherited frozen `data/manifests/v5/split_manifest.json` (50 groups, 10/10/10/10/10, singleton components; duplicate-ledger has no identity group with ≥2 fit50 members).
- Teacher: `intent_directed_first_contact_v1`, d0=0.08 Å, frozen `solve_teacher` in native/raw layer (no Kabsch; reverse intent re-reads reversed path ⇒ exact edge negations + involution). Prior V5 native/teacher shards used only for cohort dispositions/denominators (never label values). Note: V5 shards use different truth-bridge frames (divergence recorded).

## Gate evidence (validate_r3_tables.py, schema r3_6table_v2)
- `python3 scripts/agent_ops/data_gate/validate_r3_tables.py --tables-dir outputs/pesn/30h_campaign/data --report outputs/pesn/30h_campaign/data/data_gate_report_r3.json --truth-log outputs/pesn/30h_campaign/data/truth_access_log.jsonl` → **pass=10 fail=0 blocked=0 not_run=0** (T0.1a/b/c, T0.2a/b, T0.3a/b, T0.4a/b, T0.5). Truth log empty (zero ground-truth reads).
- pytest (system python3): `tests/agent_ops/test_data_r3_build.py` **14 passed**; existing `test_data_schema_validator.py` + `test_data_gate_t02.py` **52 passed**.

## Coverage (original denominators kept; ledger `prior_denominators`)
- R3: 50 curves / 6017 nodes / 11934 edges / 100 intents (0 underdetermined) / 12034 samples (d0=0.08 main only).
- Statuses: ok 9655 (train_eligible 9640 + near-conflict downweighted 0.25 ×15), terminal 1123 (masked, no fabricated action), conflict 68 (masked), masked 1188 = gauge 1054 + quarantine 134. mask=true total 2379. Usable curves: 48.
- Prior denominators (V5 shards): rows 5568; eligible 5088; gauge_unresolved 480 (8 branches / 7 reactions); terminal 137@0.05 & 282@0.08; contact 4951@0.05; effective clusters 49.

## Cohort dispositions
- RXN_0000001248: `excluded_gauge_unresolved` (prior 0/100 eligible @0.05+0.08) — all 242 samples masked; dual evidence recorded (R3 wp2-frame continuity passes but ruled exclusion kept).
- Prior gauge cohort (8 branch-sides): node-level mask (gauge_unresolved_excluded, 812 samples) — main denominator not silently expanded.
- Mapping-overflow 5 (238/1089/1174/1504/2226): pair-edit vs R/P geometry check → 4 consistent; **RXN_0000001174 quarantined** (edit (0,8) forming but Δlen=−6e-6 Å).
- Historical 342 pairs: pair-list artifact NOT in repo (ADR-0026 boundary; R2 DQ2 open) → intersection not computable; R3_scope_not_affected NOT claimed; closed instead by full R3 re-audit: 15372 near-input pairs (<0.05 Å), 26593 label comparisons → 89 conflict pairs masked (curves 990/1094/1174/2226), 12 near downweighted 0.25.
- RXN_0000002022: rechecked @0.08 → ok 229 / terminal 15 (R3) vs prior shard @0.08 terminal 5 — 0.05 "all contact" superseded.

## Open issues / downstream deps
- Supervisor: bind the three manifest SHAs at G0 (stage g0 verify); do NOT treat masked/terminal rows as training data (filter label_status=ok; near-conflict rows carry label_weight 0.25).
- W-FEAT: `irc_node.coords_flat`/`sample_view.coords_flat` extra columns carry full [N,3] geometry (x,y,z = centroid scalar slots); feature_sha covers legal forward inputs only.
- Historical-342 classification remains an open research item beyond E1 (artifact missing); R3 re-audit results supersede for R3 semantics only.
- GPU: none used (CPU-only build, ~12 s full build).
