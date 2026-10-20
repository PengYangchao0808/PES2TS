# W-FEAT / E6-V53EXEC handoff — V5.3 rollout execution + xTB constrained relax (G4)

- **task_id**: E6-V53EXEC
- **lane**: W-FEAT wave 4
- **branch**: `ops/e6-exec` (from master `2029264`)
- **mode**: H20 (G3-V52Supervision FAILED → V5.3 = upper-bound controllers only; v52 arm = DIAGNOSTIC-ONLY everywhere)
- **device**: CPU only (`pes2ts-gpu` env, torch 2.14.1+cu130 but CUDA unused — full 216-episode matrix ran in ~8 s wall, far below GPU-lock acquisition overhead; GPU left free, documented)

## 1. No-physics rollouts (supervisor-sanctioned → MAIN tree `v53/rollouts/`)

Matrix: 12 frozen curves × {forward, reverse} × steps {5,10,20} × arms {classic, r3, v52} = **216 episodes**.
Arms: `classic` = CV/Jacobian min-norm; `r3` = F2-bidir fold0 seed 17 (`5372d9b12807c3f0`, sha `a2e46b6a…`, the R3 winner); `v52` = V5.2 QT (`9ef757ff0e04a770`, sha `3802c4dd…`) labeled `diagnostic_only=true` + `diagnostic_label="v52_supervision_failed_no_claim"` in every record (machine-enforced).
Determinism: torch/np seed 17 per episode; two episodes (classic/fwd/s5, r3/fwd/s20) re-run **bitwise-equal**.
Budget parity: `check_budget_parity_v3` over all 216 — 72 (curve,direction,steps) groups, identical arm sets + identical step-0 fingerprints. PASS.

### Aggregate (per arm × steps; mean over 24 episodes/group)

| arm | steps | sign-agree | arc-prog | tsRMSD(Å) | collisions | diverged | contact(≤0.15Å/CV) | disp(Å) |
|-----|-------|-----------|----------|-----------|-----------|----------|--------------------|---------|
| classic | 5  | 0.219 | 0.07 | 0.68 | 0   | 0 | 0.04 | 1.42 |
| classic | 10 | 0.219 | 0.07 | 0.78 | 17  | 0 | 0.04 | 2.88 |
| classic | 20 | 0.240 | 0.07 | 1.02 | 215 | 0 | 0.04 | 6.37 |
| r3      | 5  | 0.889 | 0.29 | 0.60 | 0   | 0 | 0.08 | 0.89 |
| r3      | 10 | 0.879 | 0.40 | 0.61 | 0   | 0 | 0.08 | 1.76 |
| r3      | 20 | 0.876 | 0.62 | 0.66 | 14  | 1 | 0.08 | 3.40 |
| v52*    | 5  | 0.989 | 0.36 | 0.59 | 0   | 0 | 0.08 | 0.62 |
| v52*    | 10 | 0.989 | 0.50 | 0.58 | 0   | 0 | 0.08 | 1.23 |
| v52*    | 20 | 0.991 | 0.76 | 0.59 | 4   | 0 | 0.08 | 2.40 |

\* diagnostic-only, `v52_supervision_failed_no_claim` — NOT a validated controller.

Honest drift patterns: classic collapses (sign-agree ~0.22, 215 collision steps @20, tsRMSD grows 0.68→1.02); r3 holds sign agreement but arc progress saturates ~0.6 and tsRMSD does not improve past s5; v52 looks smoothest BUT its stop readouts fire 40×@20 with contact 0.08 — chance-level, consistent with G3. Contact-made ≤0.08 for every arm/budget: **no arm demonstrates TS-approach competence** here.

## 2. Constrained relax (real xTB, supervisor G4 clearance → MAIN tree `v53/relax/`)

Seed = forward-direction 20-step rollout endpoint per curve × arm (classic/r3 mandatory + v52 included) = 36 runs.
Recipe frozen & identical across arms (`RELAX_RECIPE_V3`): GFN-2, k=50 eV/Å², normal opt, maxcycle 500 (`$opt` block), 4 threads, 300 s timeout, no fixed atoms. Constraints = **all** intent CV-pair distances driven to the TRUE IRC TS-node distances (ts_index_record_only, diagnostics tier; targets geometry-independent ⇒ bit-identical across arms per curve — `check_relax_budget_parity_v3` PASS).
Request payloads built with frozen `build_constraint_request` (ACP `pes2ts_xtb_constrained_opt_request_v1`, archived per run in `runs/<curve>__<arm>/request.json`); executed by direct xtb 6.7.1 CLI (`$constrain` control translated from the same request; ACP CLI transport not used — timebox, disclosed). Per-run stdout/stderr/xtbopt.xyz/result.json archived.

### Relax table (status per curve × arm)

| arm | converged | not_converged | physics_failed | wall (s) |
|-----|-----------|---------------|----------------|----------|
| classic | 0 | 0 | 12 | 1.3 |
| r3      | 3 | 6 | 3  | ~18 |
| v52*    | 2 | 9 | 1  | ~29 |

Failure attribution (honest, in each result record): 15/16 `physics_failed` have clashed seed geometries (`seed_min_interatomic` < 0.7 Å → xtb optimizer abort |grad|>500), i.e. the open-loop rollout drifted into unphysical territory, not an xtb backend fault. 5/5 converged runs hit CV targets to ≤6e-4 Å. **No verified-TS claims** — single-imaginary-frequency verification not attempted (OptTS territory).

## 3. G4 evidence pack → `v53/g4_evidence_pack.json`

9/9 engineering-closure checklist PASS: determinism ✓, rollout budget parity ✓, relax constraint parity ✓, frozen curve list immutable ✓, v52 diagnostic labeling enforced ✓, claim-scope audit (positive verified-TS language = 0; 16 mentions all negated/forbidden contexts) ✓, aggregate recompute == manifest ✓, physics failures recorded not hidden ✓, TS-frequency verification not attempted ✓. Cost ledger inside (rollout ~8 s, relax 48.8 s total, GPU unused).

## 4. Files

- `pes2ts_core/pesn/models/rollout_v3.py` — extended: `V52_DIAGNOSTIC_LABEL`, `EPISODE_SCHEMA_V3`, per-step `disp_l1_sum`/`max_excursion_from_start`, summary `n_clip_events`/`total_displacement`/`max_atom_excursion`/`divergence_flag`, `enforce_diagnostic_labeling_v3`, `check_budget_parity_v3`, `aggregate_episode_records_v3`, `PhysicsOutcomeV3`, `execute_relax_episode_v3` (mockable runner), `RELAX_RECIPE_V3`, `check_relax_budget_parity_v3`.
- `scripts/agent_ops/feature_gate/v53_rollout_exec.py` — full-matrix runner (determinism + parity + aggregate in-manifest).
- `scripts/agent_ops/feature_gate/v53_relax_exec.py` — xTB constrained-relax executor (per-run archives, honest failure classes).
- `scripts/agent_ops/feature_gate/v53_g4_evidence.py` — G4 pack builder (recomputes checks, claim-scope audit).
- `tests/pesn/test_r3_rollout_v3.py` — +8 tests (aggregation math, parity pass/violations, labeling enforcement both directions, displacement/divergence summary, relax schema with mock executor, runner-exception→physics_failed, v52 forced-diagnostic, relax parity violations).
- MAIN-tree artifacts: `outputs/pesn/30h_campaign/v53/{rollouts/episodes.jsonl, rollouts/rollouts_manifest.json, relax/relax_manifest.json, relax/relax_results.jsonl, relax/runs/* (36 dirs), g4_evidence_pack.json}` (worktree copies committed only for small manifests per campaign convention; episodes.jsonl/runs/ are synced-to-main only, not versioned).

## 5. Tests

- `tests/pesn/test_r3_rollout_v3.py`: 29 passed.
- `tests/pesn`: **876 passed, 22 skipped** (`pes2ts` env; the `pes2ts-gpu` env lacks rdkit → use `pes2ts` for the full suite).
- `tests/agent_ops`: **471 passed** (feature/v53 subset 127 passed).

## 6. Downstream notes

- v52 arm numbers must never be quoted without the `v52_supervision_failed_no_claim` label (enforced in code + outputs).
- Relax endpoints are TS_candidate seeds only; any TS claim requires OptTS + single-imaginary mode + IRC endpoint connection (out of V5.3 scope).
- The classic arm is an upper-bound-free analytic baseline that drifts badly; future controller work should compare against r3, not classic.
