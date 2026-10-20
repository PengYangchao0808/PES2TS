# W-FEAT E6 V5.3 Prep Handoff (E6-V53PREP)

Branch: `ops/e6-v53` (from master @ fda8343, E3 trainer + E4 V5.2 labels).
CPU-only lane; no GPU touched; **no physics executed** anywhere in this task.

## 1. Rollout harness API

Module: **`pes2ts_core/pesn/models/rollout_v3.py`** (schema
`pes2ts_v53_rollout_v3`; runtime-side, passes B0/B6 product-boundary scan).

```python
from pes2ts_core.pesn.models.rollout_v3 import (
    RolloutStartV3, run_rollout_v3,          # start node + main loop
    load_r3_checkpoint_v3, load_v52_checkpoint_v3,  # checkpoint loaders (CPU ok)
    NO_PHYSICS_STEPS_V3, ARMS_V3,            # frozen (5,10,20) / ("classic","r3","v52")
    CLAIM_SCOPE_V3,                          # "TS_candidate_only" (verified_ts forbidden)
)

start = RolloutStartV3(curve_id, node_index, intent, atomic_numbers, xyz, d0=0.08)
result = run_rollout_v3(arm="r3", start=start, n_steps=5,
                        model=load_r3_checkpoint_v3(path), level="F2")
result.to_dict()   # JSON-safe: steps[] diagnostics + final_xyz + summary
```

Semantics (30h plan §6.1): per step `predict u → trust_region_clip (0.20 Å
uniform factor, same for all arms) → apply to K atoms only → re-embed from
MOVED geometry with the SAME fixed intent`. No history/future/TS/progress can
reach `model.forward` (forward-view filter + test-asserted). Every re-embed
goes through `build_sample_features_v3 → collate_v3 → to_torch_batch →
forward_view_v3`. `n_steps` must be in the frozen `{5,10,20}` budget
(else typed `STEP_BUDGET` refusal). Non-finite model `u` = typed refusal
(`NONFINITE_ACTION`), never a fabricated action.

**Per-step diagnostics** (`RolloutStepV3`): `u_max_atom`, `u_rms`,
`clip_factor`, `d_key` (intent CV-pair distances after the step),
`d_key_delta`, `cv_sign_agree` (per-CV sign match vs intent Δξ),
`min_interatomic`, `collision` (<0.5 Å), `all_finite`, and for v52:
`q_pred`, `stop_prob`, `ts_candidate_event` (record-only — never terminates
a full-curve episode).

### Controller arms
- `classic` — CV/Jacobian min-norm step `u = Jᵀ(JJᵀ)⁺Δd*`,
  `Δd* = clip(δB,−1,+1)·d0`; formula mirrored from the W-LOSS
  `key_atom_bc_v3.classic_jacobian_step` (runtime modules may not import
  `offline/*`; **bitwise parity asserted** in
  `test_classic_arm_formula_mirror_parity`). 0 params, no model needed.
- `r3` — `StarEGNNV3` `u` from an R3 checkpoint (`level` MUST match the
  checkpoint arm, e.g. F2-bidir → `"F2"`).
- `v52` — `KeyAtomV52Model` `u` + q/stop readouts. **Slot ready**:
  `load_v52_checkpoint_v3` loads the E5 joint checkpoint when it exists;
  until then an untrained `KeyAtomV52Model` can smoke wiring only (never
  for claims).

## 2. Curve pre-registration (frozen)

Artifact: **`outputs/pesn/30h_campaign/v53/rollout_curves.json`**
(schema `pes2ts_v53_rollout_curves_v1`, `frozen: true`,
`curve_list_sha256` recorded; generator
`scripts/agent_ops/feature_gate/v53_rollout_prep.py select-curves`).

Selection rule (deterministic, no RNG beyond the frozen seed-17 hash
tiebreak): pool = curves with ≥1 `label_status=ok` sample AND fold ∈ {0,1,2}
(Development only; 28 of the 48 usable curves) → strata = fold × K-quartile
(rank-based over (K, curve_id); K = |key_rows| of the forward intent) →
per-stratum winner = `min sha256("v53-rollout|17|<curve_id>")` → 12 curves
(3 folds × 4 quartiles, all non-empty). Both intent directions + start node
(endpoint A) recorded per curve. Test re-runs the selection and asserts
byte-equality with the frozen artifact, plus fold/K/quartile span.

Frozen set (12): RXN_0000000238(f2,K6,q1), 0000000406(f1,K6,q1),
0000000582(f2,K4,q0), 0000000908(f1,K7,q3), 0000001309(f0,K6,q2),
0000001423(f1,K4,q0), 0000001675(f0,K4,q0), 0000001715(f2,K6,q2),
0000001745(f1,K6,q2), 0000001847(f2,K6,q3), 0000002062(f0,K5,q1),
0000002090(f0,K6,q3).

## 3. CPU smoke (5 steps, R3 F2-bidir fold0 seed17 ckpt 5372d9b12807c3f0)

Artifact: `outputs/pesn/30h_campaign/v53/smoke_rollout_r3.json`
(rerun bitwise-equal → harness+data path is deterministic end-to-end).
Two out-of-fold frozen curves, forward intents, level F2, CPU:

| curve | fold | K | mean u_max (Å) | mean clip | d_key first → last (Å) | CV sign agree | collisions | nonfinite |
|---|---|---|---|---|---|---|---|---|
| RXN_0000000582 | 2 | 4 | 0.0243 | 1.00 | [1.868, 1.941, 1.948, 3.591] → [1.847, 2.085, 2.106, 3.668] | 0.75 (3/4 CVs; CV#4 wrong-signed every step) | 0 | 0 |
| RXN_0000001423 | 1 | 4 | 0.0513 | 1.00 | [2.893, 1.804, 1.792, 1.646] → [3.025, 1.981, 2.021, 1.628] | 0.75 (CV#1 wrong-signed; the ~2.89 Å pair grows instead of shrinking) | 0 | 0 |

Honest read: displacements finite and smooth (~0.02–0.06 Å/step, no clip
engagement, no collisions, min interatomic 0.97–1.09 Å); d_key evolves
monotonically per CV; **one CV per curve moves persistently against its
intent sign** — real diagnostic signal (consistent with the R3 SRE/sign
metrics: sign accuracy well below 1 on dev), not a harness fault. No TS
claims of any kind: `claim_scope = TS_candidate_only`.

## 4. Constrained-relax bridge verdict

Artifact: `outputs/pesn/30h_campaign/v53/relax_preflight.json`
(generator `v53_rollout_prep.py preflight-relax`).

- ACP wiring: **present** (root `/mnt/e/.../ACP_V1_20260811`, `src/acp/cli.py`
  exists, rph4 python imports `acp.cli`, config defaults.yaml present).
- xtb binary: **MISSING** (not on PATH, not in /opt/xtb, /usr/local/bin,
  or any miniforge env).
- Verdict: `wiring_present_xtb_missing`, `executor_runnable: false`,
  **`physics_missing: "physics_not_run"`, `physics_executed: false`**
  (fail-safe defaults per frozen yaml).
- Dry-run request: a real `pes2ts_xtb_constrained_opt_request_v1` payload WAS
  built for frozen curve RXN_0000000238 (driver = first K-pair distance,
  target = one-d0 directed move along Δξ; request_sha256 recorded) via
  `build_constraint_request` — **request built, never executed**.

Next stage (NOT this task): install xtb + supervisor-approved queue
clearance, then a constrained-relax executor can consume the frozen curve
list with same-budget/same-constraints across the three arms (30h §6.2).

## 5. Tests

`tests/pesn/test_r3_rollout_v3.py` — 21 tests, all green (CPU):
determinism (bitwise), step-count exactness (5/10/20) + out-of-budget
refusal, arm parity at step 0 (identical step-0 fingerprint across
classic/r3/v52), classic-formula mirror parity vs offline
`classic_jacobian_step` (bitwise), no-forbidden-inputs in the loop
(forward-view-only batches; prefix-consistent re-embed geometry; fixed
intent), non-finite typed refusal, TS_candidate_only claim scope, frozen
curve list (determinism + fold/K/quartile span), bridge preflight
fail-safe (incl. bad-wiring monkeypatch).

Full suites: `tests/pesn` 867 passed / 22 skipped /
1 failed = **pre-existing** `test_truth_guard_clean_with_pesn_bridge_registered`
(2 `.eval()` findings in W-LOSS-owned `offline/training/key_atom_bc_v3.py`;
reproduces on clean master with my files stashed — not mine, not fixed here).
`tests/agent_ops` feature-gate subset: 127 passed.

## 6. Downstream deps / notes

- E5 (V5.2 training) should write checkpoints loadable by
  `load_v52_checkpoint_v3` (joint state dict incl. `progress_head.*` /
  `ts_stop_head.*`); the `v52` arm then runs the frozen curve list unchanged.
- The G4-Rollout gate (yaml `gates.order`) can consume
  `smoke_rollout_*.json` + the frozen curve list; hard-stop H20 (V5.2 not
  passed → V5.3 classic-only controllers) is enforced by the arm choice at
  call time.
- `outputs/pesn/30h_campaign/protocol_30h.json` + `run_plan.jsonl` show as
  modified in the worktree — those are pre-existing local campaign-artifact
  sync edits, NOT part of this branch's commit.
