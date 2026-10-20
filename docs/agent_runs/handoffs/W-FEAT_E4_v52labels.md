# W-FEAT handoff — E4 V5.2 labels/heads (G3a evidence; prepares E5)

- task_id: E4-V52HEADS · lane: W-FEAT (wave 2) · branch `ops/e4-v52` · worktree `/tmp/opencode/pes2ts-wt-b`
- campaign PES2TS_30H_R3_V52_V53 · base master `6553d50` (G0+G1 PASS, formal training authorized) · **no GPU used (CPU-only, as tasked)**
- G3a label report (REAL data, supervisor-sanctioned dir):
  **`outputs/pesn/30h_campaign/e4_labels/e4_label_report.json`** (main tree; `outputs/*` is gitignored — the file is a runtime artifact, regenerable byte-stably except `generated_at`)
  regen: `python3 scripts/agent_ops/feature_gate/e4_label_gate.py --tables-dir outputs/pesn/30h_campaign/data --report outputs/pesn/30h_campaign/e4_labels/e4_label_report.json`

## 1. What was built

| File | Purpose | Status |
|---|---|---|
| `pes2ts_core/pesn/models/key_atom_heads_v3.py` | V5.2 heads on the SAME StarEGNNV3: `ProgressHeadV3` = `tanh(Linear(64,32)→SiLU→Linear(32,1))`, `TSStopHeadV3` = raw scalar logit `Linear(64,1)` (BCEWithLogits; no pre-sigmoid), `KeyAtomV52Model(backbone=…)` wrapper sharing an existing R3 backbone BY REFERENCE | 61-test battery green |
| `pes2ts_core/pesn/models/key_atom_loss_v52_v3.py` | `L_v52 = L_move + λq·L_q + λm·L_TSStop`; `L_q`=weighted-mean Huber_0.1 over q-valid rows; `L_TSStop`=weighted-mean BCEWithLogits, **Mask never in numerator or denominator**; `StopClassWeightsV52` frozen dataclass + `estimate_stop_class_weights_v52` (inner-train-only structurally); λ_mono=λ_path=0 asserted | green |
| `pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py` (extension, read-compatible) | E4 label math: `ATOMIC_MASS_TABLE_V52` (amu Z=1..118), `mass_weighted_arc_lengths_v52`, `progress_q_from_arcs_v52`, `kabsch_rmsd_v52` (proper rotation, no reflection), `stop_label_from_distance_v52`; `KeyAtomDatasetV3.frame_xyz/curve_ts_index/curve_endpoints/v52_labels` → `V52LabelTable` | all prior r3 tests + new green |
| `scripts/agent_ops/feature_gate/e4_label_gate.py` | G3a label-gate report generator (fixture via `--fixture` + real); includes head wiring (bitwise) + Move non-degradation **design** (not run) | green |
| `tests/pesn/test_r3_v52_{heads,labels,losses}.py`, `tests/agent_ops/test_feature_e4_label_gate.py` | math/gate battery (61 tests) | 61 passed |

## 2. Head parameter counts (exact; reported separately from backbone)

- backbone `StarEGNNV3` = **55,441** (`MEASURED_PARAMS_V3`, E2-frozen, unchanged)
- `ProgressHeadV3` = **2,113** = (64·32+32) + (32+1)
- `TSStopHeadV3` = **65** = (64+1)
- joint `KeyAtomV52Model` = **57,619** (budget 70k)

**Backbone-unchanged guarantee (tested explicitly, fixture + real batch):** attaching heads and running `forward`/`forward_move` leaves `u` and `z` `torch.equal`-bitwise to the bare backbone, and backbone `state_dict` bytes are untouched (no copy, no in-place edit; heads read `z` only, never write `u`/node states). `model.forward` still refuses any non-forward-view key (supervision, incl. q/stop labels, can never reach any module).

Frozen-arch note: yaml freezes stop only as "logit scalar" → minimal legal wiring `Linear(64,1)` implemented; any richer stop trunk would be improvising — route through architecture review if desired later.

## 3. E4 label definitions (derived ONLY from six-table teacher columns; 0 truth reads)

- **q**: mass-weighted arc-length along the canonical A→B IRC path — `ds_i = sqrt(Σ_a m_a‖Δx_a‖²/Σ_a m_a)` (amu table frozen; per-curve mass sum cancels in the q normalization), no inter-frame Kabsch (E1 stores one verified frame per curve; teacher is Kabsch-free by design). `q=−(s_TS−s_i)/(s_TS−s_A)` (i≤TS), `(s_i−s_TS)/(s_B−s_TS)` (i>TS); A=−1/TS=0/B=+1; **reverse intent q=−q exactly**; zero-arc segments masked (eps 1e-8 Å), never divided.
- **stop**: `d_key_TS` = Kabsch-aligned (proper rotation, det=+1, **no reflection**) RMSD over the intent K atoms between current frame and the trusted TS frame (`irc_curve.ts_index`); **positive d≤0.04 Å, negative d≥0.12 Å (both inclusive), middle=Mask**; windows NOT widened anywhere.
- Labels attach per `sample_id` (`V52LabelTable`), fwd/rev share the geometry labels; E5 consumes via `dataset.v52_labels()` then maps onto batch `sample_ids`.

## 4. REAL-data E4 label-gate evidence (from e4_label_report.json; n=12,034 samples / 50 curves / ok=9,655)

**Per-fold stop coverage (label_status=ok rows; full-denominator in report):**

| fold | ok n | pos (≤0.04) | neg (≥0.12) | mask | pos_rate | pos-bearing curves |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1705 | 87 | 1438 | 180 | 5.10% | 9 |
| 1 | 2157 | 80 | 1881 | 196 | 3.71% | 10 |
| 2 | 1706 | 84 | 1450 | 172 | 4.92% | 9 |
| 3 | 1960 | 72 | 1740 | 148 | 3.67% | 10 |
| 4 | 2127 | 82 | 1857 | 188 | 3.86% | 10 |

- **No `insufficient_stop_positives` flag** on any dev fold or the confirm estimate under the DECLARED engineering precheck (min 20 inner-train positives AND ≥5 positive-bearing curves; constants reported in the JSON, explicitly `frozen: false` — plan §5.3 freezes no number, supervisor may re-rule; windows were NOT touched to achieve this).
- **Frozen stop class weights (inner-train ONLY; T3.3)**: fold0 `pos_weight=20.31` (164 pos/3331 neg, folds{1,2}); fold1 `16.89` (171/2888, folds{0,2}); fold2 `19.87` (167/3319, folds{0,1}); confirm-runs estimate `19.00` (251/4769, folds{0,1,2}). E5 must consume these verbatim (or re-derive with the same inner-train rule and assert equality).
- **q labels**: valid on all 9,655 ok rows (zero-arc list EMPTY); per-fold min=−1.0, max 0.92–0.97 (+1.0 rows are terminal-status end frames, correctly not in ok), mean ≈ −0.09…−0.13, std ≈ 0.53–0.55.
- **reverse q=−q check: exact** — max |q_fwd+q_rev| = 0.0 over 6,017 (curve,node) pairs (6,017 = every node of every curve).
- Integrity checks (anchors A/TS/B exact, q∈[−1,1], stop domain, coverage sums) all pass; gate `proposed_state = pass` (supervisor binds G3a formally).

## 5. G3a report + Move non-degradation hook (design only — NOT run)

- Report path: `outputs/pesn/30h_campaign/e4_labels/e4_label_report.json` (binds the six table shas + three manifest shas; `truth_reads: 0`).
- Move non-degradation design is embedded as `move_non_degradation` (status `not_run`): paired (V5.2 QT joint_best) vs matched R3-only checkpoint on same (fold, seed, curve set, budget); per-curve RMSE/zero primary, SRE/cos/sign/dead guards, cluster aggregation, **≤2% RMSE/zero degradation** pass condition (campaign `guards`), failure → joint_fail (reduce λ or keep encoder frozen; never tune on confirm). Executes in G3a/G3 once R3 G2 checkpoints exist (H12 respected: no R3 baseline ⇒ no V5.2 validation claims).

## 6. Test matrix (interpreter `/tmp/opencode/venv/bin/python`, CPU)

| Command | Result |
|---|---|
| `pytest tests/pesn/test_r3_v52_{heads,labels,losses}.py tests/agent_ops/test_feature_e4_label_gate.py -q` | **61 passed** (~6 s; includes REAL-data report tests — located via git common dir / `PES2TS_E1_TABLES`, skip-if-absent) |
| `pytest tests/pesn tests/agent_ops -q` | **1273 passed, 22 skipped, 1 failed, 8 errors** — the 9 are `tests/agent_ops/test_ledger_plan_pin.py` only, **reproduced identically on CLEAN master `6553d50` (git stash)** = pre-existing campaign-state/environment issue (main-tree `run_plan.jsonl` has live uncommitted appends by the R3 lane; those files are never-write for this lane), NOT caused by E4 |

## 7. Open issues / downstream deps

1. **W-LOSS (E5)**: consume `KeyAtomDatasetV3.v52_labels()` + the frozen class weights above; train via `KeyAtomV52Model(backbone=r3_ckpt)` with Stage A freeze-encoder 400 updates then joint 1500 (yaml schedule); loss entrypoint `key_atom_loss_v52(...)` (move kwargs pass through to frozen R3 loss verbatim; pass `forward_view_v3(batch)` to the model).
2. λq/λm values are NOT set here (exploratory 2–10% gradient-share calibration is E5/WP work per plan §5.4; loss defaults λq=λm=0 = move-only).
3. `insufficient_stop_positives` precheck constants (20 pos / 5 curves) are declared engineering values, `frozen:false` in the report — supervisor may re-rule; the windows themselves are frozen.
4. Stop class weights apply through the weighted MEAN (denominator = Σ effective weights over unmasked rows, plan §5.4 `weighted_mean`): with a single-class batch the pos_weight cancels — a documented property, asserted by test; E5 should keep batches fwd/rev-mixed as composed.
5. q=+1/B-terminal rows are `terminal` status in E1 (masked for Move) — they also carry q labels in the table; whether E5 trains q on terminal rows is a W-LOSS decision (Move teacher is masked there; q itself is well-defined).
6. `tests/agent_ops/test_ledger_plan_pin.py` failure is pre-existing (see §6) — infrastructure/supervisor item, not W-FEAT.
