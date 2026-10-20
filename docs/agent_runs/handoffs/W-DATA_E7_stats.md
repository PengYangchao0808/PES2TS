# W-DATA → E7 统计复核 toolkit handoff

**task**: E7-STATSPREP · **lane**: W-DATA · **campaign**: PES2TS_30H_R3_V52_V53
**branch/commit**: `ops/e7-stats` (see final commit below)
**scope**: analysis-only statistics/reporting toolkit; no model/data/protocol changes.

## What was built

### 1. `scripts/agent_ops/data_gate/r3_stats.py`

Deterministic E7 statistics CLI (plan §7 + §4.3). Reads (read-only) `r3_runs/*/metrics.json`,
`run_records.jsonl`, `data/curve_quality.csv`, and `docs/agent_runs/state/gates/*.json`;
writes **only** `<campaign-root>/stats/r3_stats_report_<label>.{json,md}`.

```bash
python scripts/agent_ops/data_gate/r3_stats.py \
    --campaign-root /mnt/e/Calculations/AI4S_ML_Studys/PES2TS/outputs/pesn/30h_campaign \
    --gates-dir docs/agent_runs/state/gates \
    [--out-dir <campaign-root>/stats] [--label v1] \
    [--worst-n 5] [--bootstrap-draws 2000] [--bootstrap-seed 20261021] [--no-write]
```

Key behaviours:

- **Aggregate unit = independent curve/cluster.** Every arm metric is a
  run-level cluster-median (from metrics.json); `n_clusters`,
  `cluster_evaluations_total`, and unique val-curve counts are reported in
  every table. No frame-level statistics anywhere.
- **Per-run cluster bootstrap CIs are REUSED** from
  `metrics.json.key_rmse_over_zero_bootstrap` (unit=`reaction_cluster`,
  seed=20261018, draws=200) — surfaced under
  `arms.<arm>.per_run_cluster_bootstrap_ci95` and on every error card.
- **Arm-level CI** = seeded percentile bootstrap of the mean over run-level
  cluster-medians (resampling unit: run = fold×seed). Explicitly labelled as
  descriptive (runs within an arm share fold val curves) in
  `discipline_notes` — do not present these as independent-cluster CIs.
- **Sealed-fold discipline**: folds 3/4 are excluded from all dev aggregates;
  any encountered sealed run is listed under `sealed_runs_flagged` with
  `action=excluded_from_dev_aggregates`. Fold ≠ {0,1,2,3,4} runs are counted
  separately (`n_other_fold_runs`) and never aggregated.
- **Error cluster cards**: worst-N dev evaluations per arm with val-curve
  identifiers (the cluster units), n_clusters, RMSE/zero + reused CI95,
  sign/SRE, and per-curve `quality_status` from `data/curve_quality.csv`.
  metrics.json has no per-cluster decomposition; the card says so explicitly
  (no invented per-cluster numbers).
- **Cost ledger** from run_records.jsonl (latest record per run_id):
  per (phase, arm) run counts, status counts, total attempts, retried runs,
  wall-clock (metrics `wall_clock_s` preferred; ledger
  created_at→updated_at span as fallback for runs without metrics yet),
  GPU-hours estimate = cuda-device wall-clock / 3600, blocked list with
  reasons/sources, retried list. Orchestration/queue time is NOT included
  (stated in the report).
- **Gate snapshot** → one table (gate / status / evaluated_at) from
  `docs/agent_runs/state/gates/*.json`.
- **Incremental arrival tolerance**: unreadable/partial metrics.json are
  skipped and listed (`scan.skipped_metrics`); ledger runs without metrics
  yet are listed (`scan.pending_ledger_runs_without_metrics`); malformed
  ledger lines are tolerated. Re-running the tool as new arms land simply
  picks them up — regenerate `r3_stats_report_v1.{json,md}` in place.
- **Determinism**: fixed bootstrap seed, sorted keys/lists, no wall-clock
  timestamps in the payload; an `input_fingerprint` (sha256 over sorted
  metrics hashes + ledger hash + gate file hashes) is embedded instead, so
  any input change (new runs) is visible in the fingerprint. Verified
  byte-identical on rerun (tests + manual double-run with diff).

### 2. Tests — `tests/agent_ops/test_data_r3_stats.py` (12 passed)

Synthetic fixtures only (tmp_path), no dependence on live artifacts:
aggregation math (mean/std/median/cluster accounting/fraction-better),
seed-stability ≥2/3 accounting, bootstrap determinism (same seed →
identical CI) + edge cases (empty/single/NaN), end-to-end byte-identical
JSON+MD reruns, sealed-fold exclusion + flagging (+ fold-7 never
aggregated), cost arithmetic (attempts/retries/metrics-vs-ledger
wall-clock/GPU-hours/blocked), latest-ledger-record-wins, missing/corrupt/
incremental run tolerance, missing ledger+gates dirs, write-scope (only
the two report files under out_dir).

### 3. Current F-stage report (regenerated on 54 metrics files)

Artifacts (MAIN tree, only dir written):
`outputs/pesn/30h_campaign/stats/r3_stats_report_v1.json` + `.md`

Scan: 54 metrics files = 48 dev (folds 0–2) + 6 sealed (B f3/f4, flagged &
excluded); 18 ledger runs pending metrics (C/QT confirm + R1/R2 blocked);
1 retry (F0); 18 blocked (C/QT `confirm_awaiting_dev_positive`, R1/R2
`spec_gap_ramp_undefined_in_frozen_protocol`).

Cross-arm dev summary (run-level cluster-medians; RMSE/zero mean ± std
[arm CI95]; n=3 runs/arm except B=6, V-top1/2=6):

| arm | folds | RMSE/zero mean ± std [CI95] | sign med | SRE med | frac better zero |
|---|---|---|---:|---:|---|
| B | 1–2 | 0.9735 ± 0.0339 [0.9485, 0.9975] | 0.847 | 0.559 | 10/12 |
| D10 | 0 | 0.9200 ± 0.0359 [0.8986, 0.9614] | 0.888 | 0.412 | 6/6 |
| D2 | 0 | 0.9117 ± 0.0342 [0.8879, 0.9508] | 0.888 | 0.412 | 6/6 |
| F0 | 0 | 1.2045 ± 0.1593 [1.0296, 1.3412] | 0.313 | 0.485 | 0/6 |
| F1 | 0 | 1.2253 ± 0.0586 [1.1814, 1.2919] | 0.518 | 0.566 | 0/6 |
| F2 | 0 | 1.1535 ± 0.0484 [1.1175, 1.2085] | 0.462 | 0.577 | 1/6 |
| F2-bidir | 0 | 0.9098 ± 0.0349 [0.8839, 0.9494] | 0.888 | 0.412 | 6/6 |
| F2-intent-off | 0 | 1.1417 ± 0.0449 [1.1062, 1.1922] | 0.313 | 0.496 | 0/6 |
| F3 | 0 | 1.1070 ± 0.0546 [1.0750, 1.1701] | 0.417 | 0.501 | 1/6 |
| S10 | 0 | 0.9114 ± 0.0444 [0.8738, 0.9604] | 0.888 | 0.414 | 6/6 |
| S2 | 0 | 0.9078 ± 0.0341 [0.8815, 0.9463] | 0.888 | 0.412 | 6/6 |
| V-top1 | 1–2 | 0.9735 ± 0.0339 [0.9493, 0.9977] | 0.847 | 0.559 | 10/12 |
| V-top2 | 1–2 | 1.0392 ± 0.0397 [1.0084, 1.0676] | 0.624 | 0.573 | 3/12 |

Reading notes (for whoever consumes this — NOT decisions):
F2-bidir/S2/S10/D2/D10 beat zero-step on every dev cluster evaluated;
F0/F1/F2/F2-intent-off/F3 do not (RMSE/zero > 1, CI excludes <1 for most);
V-top2 ≈ parity-to-slightly-worse; B and V-top1 ≈ slight gain (10/12
clusters better). Remember fold-0 arms evaluate only 3 disjoint val pairs
(6 curves) — n is tiny; per-run cluster CIs in the JSON are wide. The plan
requires ≥2/3 seeds per fold for stability claims — `seed_stability`
table has the per-fold/per-seed numbers; nothing here is a gate decision,
G2-Move remains `not_run`.

Cost snapshot: 2837.9 s measured wall-clock across 48 done dev runs
(~0.79 GPU-h, cuda); 6 B-arm f3/f4 runs interrupted (metrics written);
18 planned confirm/V5.2 runs blocked pending dev-positive.

Gates: G0-Data pass, G1-Feature pass, G2-Move / G3 / G3a / G4 all not_run.

## How to regenerate later (V5.2/V5.3 arms land)

Same command; optionally `--label v2` to keep v1. New arms/folds appear
automatically; sealed folds keep being excluded; blocked/attempt counters
update from the ledger. Nothing else needs editing.

## Downstream dependencies

- Supervisor/Report-Gate: consume `stats/r3_stats_report_v1.json`
  (schema `pes2ts_r3_stats_report_v1`) as E7 evidence for the Report Gate;
  G2-Move evaluation is still the supervisor's to run.
- W-LOSS / trainer lane: nothing required; if metrics.json schema changes,
  extend `ARM_METRICS` paths in r3_stats.py (guarded — missing fields are
  tolerated, but re-run tests).
