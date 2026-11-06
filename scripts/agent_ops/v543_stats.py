#!/usr/bin/env python3
"""V5.4.3 WP6 statistics lane — extract / H0 / H1 / H2 / screen / H3 / select.

One CLI, seven subcommands, over the plan's FROZEN pre-registered contract.
Every threshold, formula, family and rule used here is quoted from the plan;
none of it may be tuned after the fact.  The plan lines are cited next to the
constant that implements them (see the ``PLAN_*`` tags in the constant blocks
below), so a reviewer can diff this file against the plan text.

The contract, clause by clause
------------------------------
§9.3 :916–927   H0 legality/equivalence: ten bullets ("任一失败 = invalid_run").
                This lane expands the two multi-part bullets (E(3); "有限梯度、
                有限 loss") into atomic checks and reports BOTH counts — the
                plan's ten bullets and the fourteen atomic checks — because the
                task brief says "12 项" and the plan text does not contain a
                twelve-item list (see ``H0_ITEMS`` / ``ambiguities()``).
§9.3 :929–935   H1 resources: single lane / no swap / no OOM; host ≤12 GB; VRAM
                inside its limit; core-time ratio ≤2.0 with the denominator
                being the SAME-environment M0/P0 (§9.5 :1037 forbids a V542
                archive denominator); p95 latency ≤2× A0; parameters inside the
                §6.4 budget band.  Per the brief the measured inputs may be
                absent (WP7 profile has not landed) — the INTERFACE (input
                schema + per-check detail) is frozen here and every absent
                measurement is reported as ``pending``, never as a pass.
§9.3 :937–960   H2 per-seed non-inferiority: MOVE four items against the same
                seed's V543 M0, PROCESS five items against the same seed's V543
                P0 (values from 附录 B :1589–1600); per seed independently, then
                a three-seed summary; ANY breach ⇒ ``scientific_failure``
                (:958) and no cross-track compensation.
§9.3 :960        Zero-baseline rule: when a *relative* threshold's denominator
                is zero or near zero it degrades to an absolute threshold
                (≤0.01 ratio-class, ≤0.005 Å distance-class).  The "near zero"
                epsilon is NOT fixed by the plan; this lane fixes it at
                :data:`ZERO_BASELINE_EPS` and reports that choice.
§8.2  :811–829   Screen score and tie-break, used ONLY to pick the ≤2 seed43
                (T5) top-ups per track; §8.2 :815/:817 forbid using two-seed
                numbers as a scientific verdict and forbid seed43 selection by
                any other rule.
§9.3 :962–1012   H3: the parenthesized bracket logic (:966–992), the
                significance rules (:995–1003: ≥2/3 seeds same direction, no
                per-seed H2 breach, 10,000-draw seed-averaged curve-level paired
                bootstrap with CI lower bound > 0, one-sided
                p = (1+#{improvement ≤ 0}) / 10001, MOVE two exits at α=0.025
                each with its own Holm family, PROCESS one family of
                (2 endpoints × eligible candidates) at α=0.025), and the effect
                margins (:1005–1012) that are tie-break material only.
§9.4 :1014–1033  Statistical contract: bootstrap unit = curve; shared resample
                indices; percentile 95 %; seed-average per draw first; missing
                clusters dropped on both sides; FRAME-LEVEL BOOTSTRAP FORBIDDEN
                (:1025) — a frame-level metric handed to the bootstrap is a
                typed refusal here; q validity (:1029–1033): a curve needs ≥3
                valid q points and ``frac_curves_ge3 < 0.70`` demotes
                ``q.corr_curve_median`` to diagnostic-only.
§9.5 :1035–1041  Baseline identity: the V543-recomputed M0/P0 document is the
                ONLY denominator; V542 archive numbers may never be one.
附录 B :1585–1609 All nineteen frozen thresholds + draws/seed root, mirrored
                into :data:`FROZEN_THRESHOLDS` with their plan line numbers.
附录 C :1611–1633 The 18-run matrix; T5 run ids are
                ``V543_<decoder>_S43`` (this lane emits them from the screen).

Discipline
----------
* append-only: the metrics table is a JSON document whose ``entries`` list only
  ever grows and whose records are never rewritten (an already-present
  ``(run_id, seed, identity_sha256)`` triple is skipped idempotently);
* no torch dependency in the pure paths (the evaluator's bootstrap is imported
  lazily and is the §9.4 implementation itself — it is CONSUMED, never copied);
* the CLI exposes NO threshold/dose/seed knobs; ``draws`` exists only as a
  program-level test seam and always defaults to the frozen 10,000;
* no dev read, no confirm read, no QC call, no ledger write: this lane reads
  evaluation documents and writes reports, nothing else;
* every report carries the full input SHA chain (evaluator code SHA + spec SHA
  + bridge spec SHA + per-seed checkpoint SHA) so "same instrument, same
  denominator" is auditable.

CLI::

    python scripts/agent_ops/v543_stats.py extract [--baseline PATH]
                                                   [--candidate PATH ...]
                                                   [--check-instrument]
    python scripts/agent_ops/v543_stats.py h0 [--evidence PATH]
    python scripts/agent_ops/v543_stats.py h1 [--input PATH]
    python scripts/agent_ops/v543_stats.py h2
    python scripts/agent_ops/v543_stats.py screen [--self-comparison]
    python scripts/agent_ops/v543_stats.py h3 [--self-comparison]
    python scripts/agent_ops/v543_stats.py select
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Final, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# locations
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Final[Path] = PROJECT_ROOT / (
    "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml")
DEFAULT_CAMPAIGN_ROOT: Final[Path] = (
    PROJECT_ROOT / "outputs/pesn/v543_dual_decoder_50k")
#: §9.5 :1037 — the frozen M0/P0 re-computation under the V543 evaluator.  This
#: document (and nothing else) is the H2/H3 denominator.
DEFAULT_BASELINE_REL: Final[str] = "audit/m0_p0_baseline_v543_eval.json"

REPORTS_DIR_REL: Final[str] = "reports"
METRICS_TABLE_REL: Final[str] = "reports/metrics_table.json"
H0_REPORT_REL: Final[str] = "reports/h0_legality.json"
H1_REPORT_REL: Final[str] = "reports/h1_resource.json"
H2_REPORT_REL: Final[str] = "reports/h2_report.json"
SCREEN_REPORT_REL: Final[str] = "reports/seed43_screen.json"
H3_REPORT_REL: Final[str] = "reports/h3_report.json"
SELECT_REPORT_REL: Final[str] = "reports/track_champions.json"

# ---------------------------------------------------------------------------
# schema ids (``pesn_v543_stats_*_v1`` family, mirroring the evaluator's
# ``pesn_v543_eval_report_v1`` naming)
# ---------------------------------------------------------------------------
METRICS_TABLE_SCHEMA: Final[str] = "pesn_v543_stats_metrics_table_v1"
METRICS_ENTRY_SCHEMA: Final[str] = "pesn_v543_stats_metrics_entry_v1"
CANDIDATE_BUNDLE_SCHEMA: Final[str] = "pesn_v543_candidate_eval_bundle_v1"
#: Integration alias (2026-11-05): the formal runner
#: (scripts/agent_ops/v543_formal.py) emits its candidate bundles under the
#: ``pes2ts_v543_candidate_eval_bundle_v1`` spelling — same document shape,
#: produced artefacts must not be rewritten, so both spellings are accepted.
CANDIDATE_BUNDLE_SCHEMAS: Final[frozenset] = frozenset({
    CANDIDATE_BUNDLE_SCHEMA, "pes2ts_v543_candidate_eval_bundle_v1"})
H0_EVIDENCE_SCHEMA: Final[str] = "pesn_v543_h0_evidence_v1"
H0_REPORT_SCHEMA: Final[str] = "pesn_v543_stats_h0_report_v1"
H1_INPUT_SCHEMA: Final[str] = "pesn_v543_h1_input_v1"
H1_REPORT_SCHEMA: Final[str] = "pesn_v543_stats_h1_report_v1"
H2_REPORT_SCHEMA: Final[str] = "pesn_v543_stats_h2_report_v1"
SCREEN_REPORT_SCHEMA: Final[str] = "pesn_v543_stats_screen_report_v1"
H3_REPORT_SCHEMA: Final[str] = "pesn_v543_stats_h3_report_v1"
SELECT_SCHEMA: Final[str] = "pesn_v543_stats_select_report_v1"
#: The frozen evaluator's report schema (§9.1; ``v543_eval.REPORT_SCHEMA``).
EVAL_REPORT_SCHEMA: Final[str] = "pesn_v543_eval_report_v1"
#: The M0/P0 baseline document schema (``scripts/agent_ops/v543_eval.py``).
BASELINE_SCHEMA: Final[str] = "pesn_v543_m0p0_baseline_v1"

#: Plan §8.1 :798–809 / 附录 C :1611–1633 — the three formal seeds.
FROZEN_SEEDS: Final[tuple[int, ...]] = (17, 29, 43)
#: T3/T4 seeds only (§8.2 :813) — the screen sees exactly these two.
SCREEN_SEEDS: Final[tuple[int, ...]] = (17, 29)
#: T5 seed (§8.2 :814).
SEED43: Final[int] = 43
TRACKS: Final[tuple[str, ...]] = ("move", "process")
#: 附录 C :1615–1628 — the seven formal T3/T4 candidate ids.  Only these may
#: ever produce a formal ``V543_<id>_S43`` T5 run id (:1629–1632).
FROZEN_CANDIDATE_IDS: Final[tuple[str, ...]] = ("M1", "M2", "M3",
                                                "P1", "P2", "P3", "P4")
#: 附录 C :1577–1578 / :302–303 — the T3/T4 candidate ids per track.
TRACK_CANDIDATES: Final[Mapping[str, tuple[str, ...]]] = {
    "move": ("M1", "M2", "M3", "P1", "P2", "P3", "P4"),
    "process": ("M1", "M2", "M3", "P1", "P2", "P3", "P4"),
}
#: §9.3 :939/:948 — the per-track baselines (M0 / P0).
BASELINE_ID_BY_TRACK: Final[Mapping[str, str]] = {"move": "M0",
                                                 "process": "P0"}
#: The combined id of the zero-residual row in the §9.5 baseline document: one
#: ``evaluate_baseline`` report IS M0 (u = u0) and P0 (q = tanh(q0_raw),
#: e = e0) at the same time (§4.3 :226).
BASELINE_ROW_ID: Final[str] = "M0/P0"

# ---------------------------------------------------------------------------
# frozen thresholds — 附录 B :1585–1609 verbatim (value + plan line)
# ---------------------------------------------------------------------------
#: ``move_rmse_regression_rel_max_pct`` — 附录 B :1589, plan §9.3 :943.
MOVE_RMSE_REGRESSION_REL_MAX: Final[float] = 2.0 / 100.0
#: ``move_sign_drop_abs_max`` — 附录 B :1590, plan §9.3 :944.
MOVE_SIGN_DROP_ABS_MAX: Final[float] = 0.01
#: ``move_dead_increase_abs_max`` — 附录 B :1591, plan §9.3 :945.
MOVE_DEAD_INCREASE_ABS_MAX: Final[float] = 0.01
#: ``move_fixed_coverage_regression_rel_max_pct`` — 附录 B :1592, :946.
MOVE_FIXED_COVERAGE_REGRESSION_REL_MAX: Final[float] = 2.0 / 100.0
#: ``move_h3_rmse_rel_min_pct`` — 附录 B :1593, plan §9.3 :971.
MOVE_H3_RMSE_REL_MIN: Final[float] = 3.0 / 100.0
#: ``move_h3_direction_rel_min_pct`` — 附录 B :1594, plan §9.3 :972.
MOVE_H3_DIRECTION_REL_MIN: Final[float] = 3.0 / 100.0
#: ``move_h3_direction_abs_min`` — 附录 B :1595, plan §9.3 :973.
MOVE_H3_DIRECTION_ABS_MIN: Final[float] = 0.01
#: ``process_auprc_drop_abs_max`` — 附录 B :1596, plan §9.3 :952/:988.
PROCESS_AUPRC_DROP_ABS_MAX: Final[float] = 0.01
#: ``process_recall_at_fpr_drop_abs_max`` — 附录 B :1597, plan §9.3 :953/:990.
PROCESS_RECALL_DROP_ABS_MAX: Final[float] = 0.02
#: ``process_q_mae_regression_rel_max_pct`` — 附录 B :1598, :954/:989.
PROCESS_Q_MAE_REGRESSION_REL_MAX: Final[float] = 5.0 / 100.0
#: ``process_ordering_drop_abs_max`` — 附录 B :1599, plan §9.3 :955/:992.
PROCESS_ORDERING_DROP_ABS_MAX: Final[float] = 0.02
#: ``process_ece_increase_abs_max`` — 附录 B :1600, plan §9.3 :956/:991.
PROCESS_ECE_INCREASE_ABS_MAX: Final[float] = 0.02
#: ``process_h3_auprc_abs_min`` — 附录 B :1601, plan §9.3 :985.
PROCESS_H3_AUPRC_ABS_MIN: Final[float] = 0.02
#: ``process_h3_q_mae_rel_min_pct`` — 附录 B :1602, plan §9.3 :986.
PROCESS_H3_Q_MAE_REL_MIN: Final[float] = 5.0 / 100.0
#: ``process_fpr_constraint`` — 附录 B :1603 (the evaluator's operating point).
PROCESS_FPR_CONSTRAINT: Final[float] = 0.05
#: ``move_exit_alpha_each`` — 附录 B :1604, plan §9.3 :1001.
MOVE_EXIT_ALPHA_EACH: Final[float] = 0.025
#: ``process_family_alpha`` — 附录 B :1605, plan §9.3 :1002.
PROCESS_FAMILY_ALPHA: Final[float] = 0.025
#: ``max_core_time_ratio`` — 附录 B :1606, plan §9.3 :933 / §8.5 :865.
MAX_CORE_TIME_RATIO: Final[float] = 2.0
#: ``max_latency_ratio_p95`` — 附录 B :1607, plan §9.3 :934 / §8.3 :838.
MAX_LATENCY_RATIO_P95: Final[float] = 2.0
#: ``bootstrap_draws`` — 附录 B :1608, plan §9.4 :1020.
BOOTSTRAP_DRAWS: Final[int] = 10_000
#: ``bootstrap_seed_root`` — 附录 B :1609, plan §9.4.
BOOTSTRAP_SEED_ROOT: Final[int] = 20261102
#: §9.3 :997 — "三 seed 中 ≥2 个同向".
H3_MIN_SEEDS_SAME_DIRECTION: Final[int] = 2
H3_TOTAL_SEEDS: Final[int] = 3
#: §9.4 :1021 — percentile 95 % (2.5/97.5); the evaluator owns the constants,
#: they are repeated here only for the report text.
CI_LOW_PERCENTILE: Final[float] = 2.5
CI_HIGH_PERCENTILE: Final[float] = 97.5
#: §9.3 :960 — "在对照基线为 0 或接近 0 时退化为绝对阈值：退化 ≤0.01（比例类）
#: 或 ≤0.005 Å（距离类）".  The plan fixes the ABSOLUTE limits but not the
#: "near zero" epsilon; this lane fixes it at 1e-6 and reports the choice.
ZERO_BASELINE_EPS: Final[float] = 1.0e-6
ZERO_BASELINE_ABS_MAX_RATIO_CLASS: Final[float] = 0.01
ZERO_BASELINE_ABS_MAX_DISTANCE_CLASS: Final[float] = 0.005
#: §9.4 :1029–1033 — q validity rule (mirrors the evaluator's constants).
Q_CORR_MIN_POINTS: Final[int] = 3
Q_FRAC_GE3_MIN: Final[float] = 0.70
#: Floating-point comparison slack for a "≤" gate: it absorbs binary
#: representation noise at an EXACT boundary and is far below any scientific
#: resolution (it never relaxes a threshold in a meaningful way).
LIMIT_TOLERANCE: Final[float] = 1.0e-12

#: The frozen threshold table as DOCUMENTED data (附录 B :1587–1609).  Every
#: report carries this block so a reader sees the values THIS lane applied and
#: the plan line each one comes from.
FROZEN_THRESHOLDS: Final[Mapping[str, Mapping[str, Any]]] = {
    "move_rmse_regression_rel_max_pct": {
        "value": 2.0, "plan_line": "附录 B :1589 / §9.3 :943"},
    "move_sign_drop_abs_max": {
        "value": 0.01, "plan_line": "附录 B :1590 / §9.3 :944"},
    "move_dead_increase_abs_max": {
        "value": 0.01, "plan_line": "附录 B :1591 / §9.3 :945"},
    "move_fixed_coverage_regression_rel_max_pct": {
        "value": 2.0, "plan_line": "附录 B :1592 / §9.3 :946"},
    "move_h3_rmse_rel_min_pct": {
        "value": 3.0, "plan_line": "附录 B :1593 / §9.3 :971"},
    "move_h3_direction_rel_min_pct": {
        "value": 3.0, "plan_line": "附录 B :1594 / §9.3 :972"},
    "move_h3_direction_abs_min": {
        "value": 0.01, "plan_line": "附录 B :1595 / §9.3 :973"},
    "process_auprc_drop_abs_max": {
        "value": 0.01, "plan_line": "附录 B :1596 / §9.3 :952"},
    "process_recall_at_fpr_drop_abs_max": {
        "value": 0.02, "plan_line": "附录 B :1597 / §9.3 :953"},
    "process_q_mae_regression_rel_max_pct": {
        "value": 5.0, "plan_line": "附录 B :1598 / §9.3 :954"},
    "process_ordering_drop_abs_max": {
        "value": 0.02, "plan_line": "附录 B :1599 / §9.3 :955"},
    "process_ece_increase_abs_max": {
        "value": 0.02, "plan_line": "附录 B :1600 / §9.3 :956"},
    "process_h3_auprc_abs_min": {
        "value": 0.02, "plan_line": "附录 B :1601 / §9.3 :985"},
    "process_h3_q_mae_rel_min_pct": {
        "value": 5.0, "plan_line": "附录 B :1602 / §9.3 :986"},
    "process_fpr_constraint": {
        "value": 0.05, "plan_line": "附录 B :1603"},
    "move_exit_alpha_each": {
        "value": 0.025, "plan_line": "附录 B :1604 / §9.3 :1001"},
    "process_family_alpha": {
        "value": 0.025, "plan_line": "附录 B :1605 / §9.3 :1002"},
    "max_core_time_ratio": {
        "value": 2.0, "plan_line": "附录 B :1606 / §9.3 :933"},
    "max_latency_ratio_p95": {
        "value": 2.0, "plan_line": "附录 B :1607 / §9.3 :934"},
}
#: 附录 B :1608–1609 — frozen bootstrap identity (config ``evaluation`` block).
FROZEN_BOOTSTRAP: Final[Mapping[str, Any]] = {
    "bootstrap_draws": BOOTSTRAP_DRAWS,
    "bootstrap_seed_root": BOOTSTRAP_SEED_ROOT,
    "plan_line": "附录 B :1608-1609 / §9.4 :1020-1023",
}

# ---------------------------------------------------------------------------
# metric ids — the evaluator's own bootstrap metric ids, so the per-curve
# tables line up 1:1 with ``bootstrap_inputs`` (§9.4)
# ---------------------------------------------------------------------------
#: The nine H2 leaves (§9.3 :941–956).  Ids match
#: ``scripts/agent_ops/v543_eval.BOOTSTRAP_METRICS`` where they exist.
METRIC_IDS: Final[tuple[str, ...]] = (
    "move.rmse_zero",
    "move.sign_accuracy",
    "move.dead_fraction",
    "move.fixed_coverage.direction_at_3",
    "process.endpoint.auprc",
    "process.endpoint.recall_at_fpr_0p05",
    "process.endpoint.ece",
    "process.q.mae_curve_median",
    "process.ordering.pairwise_accuracy_full",
)
#: Diagnostics recorded alongside (never gate material except where §9.3 says).
DIAGNOSTIC_METRIC_IDS: Final[tuple[str, ...]] = (
    "process.endpoint.auroc",
    "process.endpoint.brier",
    "process.q.corr_curve_median",
    "move.fixed_coverage.topk_overlap_at_3",
    "move.atom_dead_fraction",
    "move.pair_delta_d_mae",
    "process.ordering.pairwise_accuracy_cross_ts",
)
#: Where each pooled leaf lives inside a ``pesn_v543_eval_report_v1`` document.
POOLED_LEAVES: Final[Mapping[str, tuple[str, ...]]] = {
    "move.rmse_zero": ("pooled", "move", "rmse_zero"),
    "move.sign_accuracy": ("pooled", "move", "sign_accuracy_cluster_median"),
    "move.dead_fraction": ("pooled", "move", "dead_fraction"),
    "move.fixed_coverage.direction_at_3":
        ("pooled", "move", "fixed_coverage_direction_at_3"),
    "process.endpoint.auprc": ("pooled", "process", "endpoint", "auprc"),
    "process.endpoint.recall_at_fpr_0p05":
        ("pooled", "process", "endpoint", "recall_at_fpr_0p05"),
    "process.endpoint.ece": ("pooled", "process", "endpoint", "ece"),
    "process.q.mae_curve_median":
        ("pooled", "process", "q", "mae_curve_median"),
    "process.ordering.pairwise_accuracy_full":
        ("pooled", "process", "ordering", "pairwise_accuracy_full"),
    # -- diagnostics --------------------------------------------------------
    "process.endpoint.auroc": ("pooled", "process", "endpoint", "auroc"),
    "process.endpoint.brier": ("pooled", "process", "endpoint", "brier"),
    "process.q.corr_curve_median":
        ("pooled", "process", "q", "corr_curve_median"),
    "move.fixed_coverage.topk_overlap_at_3":
        ("pooled", "move", "fixed_coverage_topk_overlap_at_3"),
    "move.atom_dead_fraction": ("pooled", "move", "atom_dead_fraction"),
    "move.pair_delta_d_mae": ("pooled", "move", "pair_delta_d_mae"),
    "process.ordering.pairwise_accuracy_cross_ts":
        ("pooled", "process", "ordering", "pairwise_accuracy_cross_ts"),
}
#: Where each per-curve value lives inside a ``per_curve`` row of the report
#: (identical to ``BOOTSTRAP_METRICS`` for the ten bootstrap leaves).
CURVE_LEAVES: Final[Mapping[str, tuple[str, str]]] = {
    "move.rmse_zero": ("move", "rmse_zero"),
    "move.sign_accuracy": ("move", "sign_accuracy"),
    "move.dead_fraction": ("move", "dead_zone_fraction"),
    "move.fixed_coverage.direction_at_3":
        ("move", "fixed_coverage_direction_at_3"),
    "move.fixed_coverage.topk_overlap_at_3":
        ("move", "fixed_coverage_topk_overlap_at_3"),
    "process.q.mae_curve_median": ("q", "mae"),
    "process.ordering.pairwise_accuracy_full":
        ("ordering", "pairwise_accuracy_full"),
    "process.endpoint.auprc": ("endpoint", "auprc"),
    "process.endpoint.brier": ("endpoint", "brier"),
    "process.endpoint.ece": ("endpoint", "ece"),
    # -- extras kept for the validity / diagnostic read --------------------
    "process.ordering.pairwise_accuracy_cross_ts":
        ("ordering", "pairwise_accuracy_cross_ts"),
    "process.q.n_points": ("q", "n_points"),
    "process.q.corr_curve_median": ("q", "corr"),
    "move.atom_dead_fraction": ("move", "atom_dead_fraction"),
}
#: §9.4 :1018 — the bootstrap unit.  Only CURVE-level metrics may be resampled;
#: the table below is the frozen higher-is-better flag of every such metric
#: (taken from the evaluator's ``BOOTSTRAP_METRICS``).
CURVE_LEVEL_HIGHER_IS_BETTER: Final[Mapping[str, bool]] = {
    "move.rmse_zero": False,
    "move.sign_accuracy": True,
    "move.dead_fraction": False,
    "move.fixed_coverage.direction_at_3": True,
    "move.fixed_coverage.topk_overlap_at_3": True,
    "process.q.mae_curve_median": False,
    "process.ordering.pairwise_accuracy_full": True,
    "process.endpoint.auprc": True,
    "process.endpoint.brier": False,
    "process.endpoint.ece": False,
}
#: §9.4 :1025 — "frame-level bootstrap 禁止".  These metrics are pooled
#: frame-level quantities (an operating point over all frames, a frame-mean
#: stratum value); handing one to the bootstrap is a typed refusal.
FRAME_LEVEL_METRICS: Final[Mapping[str, str]] = {
    "process.endpoint.recall_at_fpr_0p05":
        "pooled frame-level operating point (the threshold maximises recall "
        "over ALL frames, §9.1 :884); a per-curve value would be a different "
        "quantity",
    "process.endpoint.fpr_at_threshold":
        "pooled frame-level operating point",
    "process.endpoint.prevalence": "pooled frame-level prevalence",
    "move.fixed_coverage.direction_at_3_frame_mean":
        "frame-mean of the per-frame stratum metric (§9.2 :910 reports it as "
        "a breakdown, never as a bootstrap cluster value)",
    "process.q.total_points": "frame-level count, not a per-curve statistic",
}

# ---------------------------------------------------------------------------
# H2 gates — 附录 B values, plan §9.3 :939–956 rows
# ---------------------------------------------------------------------------
#: ``kind``:
#:   relative_degradation — degradation relative to the baseline (see below)
#:   absolute_drop        — degradation measured in metric units
#:   absolute_increase    — degradation measured in metric units
#: ``higher_is_better`` fixes which direction is a degradation (§9.1 :876-891:
#: ``rmse_zero``/``q.mae``/``ece``/``dead_fraction`` are lower-is-better,
#: ``sign_accuracy``/``direction_at_3``/``auprc``/``recall``/``ordering`` are
#: higher-is-better), so one uniform ``degradation`` quantity is used:
#: ``baseline − candidate`` for higher-is-better, ``candidate − baseline``
#: otherwise.  A positive degradation is bad; every gate is
#: ``degradation ≤ limit``.
H2_SPECS: Final[tuple[Mapping[str, Any], ...]] = (
    {
        "id": "move_rmse_zero_rel_degradation",
        "block": "move",
        "baseline_id": "M0",
        "metric": "move.rmse_zero",
        "higher_is_better": False,
        "kind": "relative_degradation",
        "limit": MOVE_RMSE_REGRESSION_REL_MAX,
        "threshold_key": "move_rmse_regression_rel_max_pct",
        "plan_line": "§9.3 :943 / 附录 B :1589",
        "table_row": "rmse_zero | ≤2.0% 相对",
    },
    {
        "id": "move_sign_accuracy_abs_drop",
        "block": "move",
        "baseline_id": "M0",
        "metric": "move.sign_accuracy",
        "higher_is_better": True,
        "kind": "absolute_drop",
        "limit": MOVE_SIGN_DROP_ABS_MAX,
        "threshold_key": "move_sign_drop_abs_max",
        "plan_line": "§9.3 :944 / 附录 B :1590",
        "table_row": "sign_accuracy | ≤0.01 绝对",
    },
    {
        "id": "move_dead_fraction_abs_increase",
        "block": "move",
        "baseline_id": "M0",
        "metric": "move.dead_fraction",
        "higher_is_better": False,
        "kind": "absolute_increase",
        "limit": MOVE_DEAD_INCREASE_ABS_MAX,
        "threshold_key": "move_dead_increase_abs_max",
        "plan_line": "§9.3 :945 / 附录 B :1591",
        "table_row": "dead_fraction | ≤0.01 绝对增加",
    },
    {
        "id": "move_fixed_coverage_direction_rel_degradation",
        "block": "move",
        "baseline_id": "M0",
        "metric": "move.fixed_coverage.direction_at_3",
        "higher_is_better": True,
        "kind": "relative_degradation",
        "limit": MOVE_FIXED_COVERAGE_REGRESSION_REL_MAX,
        "threshold_key": "move_fixed_coverage_regression_rel_max_pct",
        "plan_line": "§9.3 :946 / 附录 B :1592",
        "table_row": "fixed_coverage.direction_at_3 | ≤2.0% 相对",
    },
    {
        "id": "process_auprc_abs_drop",
        "block": "process",
        "baseline_id": "P0",
        "metric": "process.endpoint.auprc",
        "higher_is_better": True,
        "kind": "absolute_drop",
        "limit": PROCESS_AUPRC_DROP_ABS_MAX,
        "threshold_key": "process_auprc_drop_abs_max",
        "plan_line": "§9.3 :952 / 附录 B :1596",
        "table_row": "endpoint.auprc | ≤0.01 绝对",
    },
    {
        "id": "process_recall_at_fpr_abs_drop",
        "block": "process",
        "baseline_id": "P0",
        "metric": "process.endpoint.recall_at_fpr_0p05",
        "higher_is_better": True,
        "kind": "absolute_drop",
        "limit": PROCESS_RECALL_DROP_ABS_MAX,
        "threshold_key": "process_recall_at_fpr_drop_abs_max",
        "plan_line": "§9.3 :953 / 附录 B :1597",
        "table_row": "endpoint.recall_at_fpr_0p05 | ≤0.02 绝对",
    },
    {
        "id": "process_q_mae_rel_degradation",
        "block": "process",
        "baseline_id": "P0",
        "metric": "process.q.mae_curve_median",
        "higher_is_better": False,
        "kind": "relative_degradation",
        "limit": PROCESS_Q_MAE_REGRESSION_REL_MAX,
        "threshold_key": "process_q_mae_regression_rel_max_pct",
        "plan_line": "§9.3 :954 / 附录 B :1598",
        "table_row": "q.mae_curve_median | ≤5.0% 相对",
    },
    {
        "id": "process_ordering_abs_drop",
        "block": "process",
        "baseline_id": "P0",
        "metric": "process.ordering.pairwise_accuracy_full",
        "higher_is_better": True,
        "kind": "absolute_drop",
        "limit": PROCESS_ORDERING_DROP_ABS_MAX,
        "threshold_key": "process_ordering_drop_abs_max",
        "plan_line": "§9.3 :955 / 附录 B :1599",
        "table_row": "ordering.pairwise_accuracy_full | ≤0.02 绝对",
    },
    {
        "id": "process_ece_abs_increase",
        "block": "process",
        "baseline_id": "P0",
        "metric": "process.endpoint.ece",
        "higher_is_better": False,
        "kind": "absolute_increase",
        "limit": PROCESS_ECE_INCREASE_ABS_MAX,
        "threshold_key": "process_ece_increase_abs_max",
        "plan_line": "§9.3 :956 / 附录 B :1600",
        "table_row": "endpoint.ece | ≤0.02 绝对增加",
    },
)
#: The ratio-class metrics of §9.3 :960 (zero-baseline degradation); the
#: distance-class 0.005 Å branch is declared but no H2/H3 metric is a distance
#: — ``pair_delta_d.mae`` is the only Å-valued leaf and it is diagnostic.
ZERO_BASELINE_CLASS: Final[Mapping[str, str]] = {
    "move.rmse_zero": "ratio",
    "move.fixed_coverage.direction_at_3": "ratio",
    "process.q.mae_curve_median": "ratio",
    "move.pair_delta_d_mae": "distance",
}

# ---------------------------------------------------------------------------
# H3 endpoints — the four confirmatory quantities of §9.3 :966–992
# ---------------------------------------------------------------------------
H3_ENDPOINTS: Final[tuple[Mapping[str, Any], ...]] = (
    {
        "endpoint": "move_rmse",
        "track": "move",
        "metric": "move.rmse_zero",
        "higher_is_better": False,
        "family": "move_rmse_exit",
        "alpha": MOVE_EXIT_ALPHA_EACH,
        "plan_line": "§9.3 :966-976 (MOVE 括号) / :1001 (两出口各 α=0.025)",
    },
    {
        "endpoint": "move_direction",
        "track": "move",
        "metric": "move.fixed_coverage.direction_at_3",
        "higher_is_better": True,
        "family": "move_direction_exit",
        "alpha": MOVE_EXIT_ALPHA_EACH,
        "plan_line": "§9.3 :966-976 / :1001",
    },
    {
        "endpoint": "process_auprc",
        "track": "process",
        "metric": "process.endpoint.auprc",
        "higher_is_better": True,
        "family": "process_route",
        "alpha": PROCESS_FAMILY_ALPHA,
        "plan_line": "§9.3 :981-992 / :1002 (PROCESS 单 family)",
    },
    {
        "endpoint": "process_q",
        "track": "process",
        "metric": "process.q.mae_curve_median",
        "higher_is_better": False,
        "family": "process_route",
        "alpha": PROCESS_FAMILY_ALPHA,
        "plan_line": "§9.3 :981-992 / :1002",
    },
)
ENDPOINT_IDS: Final[tuple[str, ...]] = tuple(
    str(spec["endpoint"]) for spec in H3_ENDPOINTS)

#: §8.2 :819–827 — the pre-registered screen-score normalizers.  The MOVE
#: formula divides by 0.03 twice; PROCESS by 0.02 / 0.05 / 0.02.
SCREEN_NORMALIZERS: Final[Mapping[str, float]] = {
    "move": 0.03,
    "move_rmse_rel_improve": MOVE_H3_RMSE_REL_MIN,
    "move_fixed_coverage_direction_rel": MOVE_H3_DIRECTION_REL_MIN,
    "process_auprc_delta": PROCESS_H3_AUPRC_ABS_MIN,
    "process_q_mae_rel_improve": PROCESS_H3_Q_MAE_REL_MIN,
    "process_ordering_abs_improve": PROCESS_ORDERING_DROP_ABS_MAX,
}

# ---------------------------------------------------------------------------
# H0 checklist — §9.3 :916–927, one bullet per entry, verbatim quote
# ---------------------------------------------------------------------------
H0_ITEMS: Final[tuple[Mapping[str, Any], ...]] = (
    {
        "id": "h0.schema_forward_view_whitelist",
        "bullet": 1,
        "quote": "schema/forward-view 白名单",
        "plan_line": "§9.3 :918",
        "check": ("the DecoderContextV1 schema id and the forward-view field "
                  "whitelist are enforced; an unknown/extra field is refused"),
        "related_test": "test_v543_track_view_isolation / "
                        "test_v543_decoder_no_xyz",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.no_supervision_leak",
        "bullet": 2,
        "quote": "无监督泄漏（哨兵测试）",
        "plan_line": "§9.3 :919",
        "check": ("sentinel supervision tensors never reach the DecoderContext "
                  "(§5.6 :420-425); the sentinel test must be present"),
        "related_test": "test_v543_no_supervision_leak",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.e3_translation_invariance",
        "bullet": 3,
        "quote": "E(3)：平移不变、旋转/反射等变、原子置换等变、边置换等变",
        "plan_line": "§9.3 :920",
        "check": "translating the input leaves the prediction unchanged",
        "related_test": "test_v543_translation_invariance",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.e3_rotation_reflection_equivariance",
        "bullet": 3,
        "quote": "E(3)：平移不变、旋转/反射等变、原子置换等变、边置换等变",
        "plan_line": "§9.3 :920",
        "check": ("a rotation/reflection R maps key_v/u0/du to R·key_v/R·u0/"
                  "R·du"),
        "related_test": "test_v543_rotation_equivariance",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.e3_atom_permutation_equivariance",
        "bullet": 3,
        "quote": "E(3)：平移不变、旋转/反射等变、原子置换等变、边置换等变",
        "plan_line": "§9.3 :920",
        "check": "atom permutation equivariance (key rows follow the slots)",
        "related_test": "test_v543_permutation_equivariance",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.e3_edge_permutation_equivariance",
        "bullet": 3,
        "quote": "E(3)：平移不变、旋转/反射等变、原子置换等变、边置换等变",
        "plan_line": "§9.3 :920",
        "check": "edge/pair permutation equivariance",
        "related_test": "test_v543_permutation_equivariance",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.reverse_intent_contract",
        "bullet": 4,
        "quote": "reverse intent 合同",
        "plan_line": "§9.3 :921",
        "check": ("the reverse-intent contract holds (pair direction is the "
                  "canonical key-slot order with signed ΔB, not a chemistry "
                  "arrow)"),
        "related_test": "test_v543_reverse_intent",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.non_critical_atoms_zero_displacement",
        "bullet": 5,
        "quote": "非关键原子零位移",
        "plan_line": "§9.3 :922",
        "check": ("non-key atoms receive exactly zero displacement (the MOVE "
                  "head can only move key slots)"),
        "related_test": "test_v543_no_move_contract (supporting)",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.typed_invalid_input_rejection",
        "bullet": 6,
        "quote": "typed-invalid 输入拒绝",
        "plan_line": "§9.3 :923",
        "check": ("malformed identity/shape/units raise a typed refusal "
                  "instead of producing a number"),
        "related_test": "test_v543_bundle_strict_load (supporting)",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.restore_resume",
        "bullet": 7,
        "quote": "restore/resume",
        "plan_line": "§9.3 :924",
        "check": "exact resume/restore equivalence",
        "related_test": "test_v543_exact_resume",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.finite_gradients",
        "bullet": 8,
        "quote": "有限梯度、有限 loss",
        "plan_line": "§9.3 :925",
        "check": "every gradient of a training step is finite",
        "related_test": "test_v543_gradient_isolation (supporting)",
        "evidence_kind": "run_ledger_record",
    },
    {
        "id": "h0.finite_loss",
        "bullet": 8,
        "quote": "有限梯度、有限 loss",
        "plan_line": "§9.3 :925",
        "check": "the loss is finite at every logged step (non-finite = "
                 "invalid_run, §8.5 :863)",
        "related_test": "test_v543_gradient_isolation (supporting)",
        "evidence_kind": "run_ledger_record",
    },
    {
        "id": "h0.encoder_forward_exactly_once",
        "bullet": 9,
        "quote": "Encoder 每次 forward 恰好 1 次",
        "plan_line": "§9.3 :926",
        "check": "exactly one Encoder forward per model forward (§8.3 :839)",
        "related_test": "test_v543_encoder_forward_once",
        "evidence_kind": "test_or_audit_record",
    },
    {
        "id": "h0.zero_residual_parity",
        "bullet": 10,
        "quote": "零残差 parity",
        "plan_line": "§9.3 :927",
        "check": ("zero residual reproduces M0/P0 exactly (u = u0, "
                  "q = tanh(q0_raw), e = e0; §4.3 :226)"),
        "related_test": "test_v543_zero_residual_parity / "
                        "test_v543_q0_raw_tanh_identity",
        "evidence_kind": "test_or_audit_record",
    },
)
H0_ITEM_IDS: Final[tuple[str, ...]] = tuple(
    str(item["id"]) for item in H0_ITEMS)


# ---------------------------------------------------------------------------
# typed refusal + small helpers
# ---------------------------------------------------------------------------
class V543StatsError(RuntimeError):
    """Typed refusal of the V5.4.3 statistics lane (``code`` + ``detail``)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else str(code))
        self.code = str(code)
        self.detail = str(detail)


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise V543StatsError(code, detail)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(obj: Any) -> str:
    """The ONE canonical digest recipe (identity ``canonical_sha256``).

    Imported lazily so the pure paths stay light; the recipe is never copied
    into this module.
    """
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256

    return canonical_sha256(obj)


def _load_json(path: Any, code: str, label: str) -> Dict[str, Any]:
    p = Path(path)
    _require(p.is_file(), code, f"{label} not found: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V543StatsError(code, f"{label} unreadable: {p}: {exc}") from exc
    _require(isinstance(doc, Mapping), code,
             f"{label} is not a JSON object: {p}")
    return dict(doc)


def _write_report(campaign_root: Any, rel: str,
                  doc: Mapping[str, Any]) -> Path:
    path = Path(campaign_root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(doc), indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return path


def _dig(obj: Any, path: Sequence[str]) -> Any:
    node: Any = obj
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _eval_module() -> Any:
    """The frozen evaluator (its bootstrap is the §9.4 implementation).

    A single retry after ``importlib.invalidate_caches()`` absorbs the
    stale-``__pycache__`` flake seen on network/9p mounted checkouts; a real
    import failure still surfaces as a typed refusal.
    """
    try:
        from pes2ts_core.pesn.offline.evaluation import v543_eval
        return v543_eval
    except Exception as first:  # noqa: BLE001 - import failure is a refusal
        try:
            import importlib

            importlib.invalidate_caches()
            from pes2ts_core.pesn.offline.evaluation import v543_eval
            return v543_eval
        except Exception as exc:  # noqa: BLE001
            raise V543StatsError(
                "V543_STATS_EVALUATOR_UNAVAILABLE",
                f"the frozen V543 evaluator could not be imported ({exc}; "
                f"first attempt: {first}); the statistics lane never "
                "re-implements §9.4") from exc


def _registry_band() -> Dict[str, Any]:
    """The §6.4 budget band from the registry (the frozen single source)."""
    try:
        from pes2ts_core.pesn.models.v543 import registry
    except Exception as exc:  # noqa: BLE001
        raise V543StatsError(
            "V543_STATS_REGISTRY_UNAVAILABLE",
            f"the V543 registry (parameter band, §6.4) is unavailable: {exc}"
        ) from exc
    return {
        "target": int(registry.PARAM_TARGET_V543),
        "band_pct": float(registry.PARAM_BAND_PCT_V543),
        "band_min": int(registry.PARAM_BAND_MIN_V543),
        "band_max": int(registry.PARAM_BAND_MAX_V543),
        "hard_cap": int(registry.PARAM_HARD_CAP_V543),
        "global_cap": int(registry.PARAM_GLOBAL_CAP_V543),
        "source": "pes2ts_core/pesn/models/v543/registry.py (§6.4 :558-571)",
    }


def _limit_ok(value: float, limit: float) -> bool:
    """``value <= limit`` with representation slack (:data:`LIMIT_TOLERANCE`)."""
    return float(value) <= float(limit) + LIMIT_TOLERANCE


# ===========================================================================
# 0. baseline identity — the §9.5 SHA chain
# ===========================================================================
def bootstrap_contract(document: Mapping[str, Any], *,
                       seeds: Sequence[int] = ()
                       ) -> Dict[str, Any]:
    """The producer's §9.4 bootstrap contract, read from ``bootstrap_inputs``.

    The refreshed evaluator declares, in the report itself, WHICH bootstrap is
    confirmatory (``modes.per_seed_independent`` →
    ``paired_curve_bootstrap_multiseed``) and which is audit/compatibility only
    (``modes.shared_intersection`` → ``paired_curve_bootstrap``), together with
    ``seed_order``, ``draws``, ``seed_root`` and the per-metric
    ``higher_is_better`` flag.  This lane reads those values instead of
    hard-coding them, so an instrument bump cannot silently desynchronise the
    statistics lane from the instrument.

    Everything here is OPTIONAL: a document without ``bootstrap_inputs``
    (a synthetic candidate bundle) yields the frozen plan defaults of 附录 B,
    and ``source`` records that fact.
    """
    blocks = document.get("bootstrap_inputs")
    block = blocks if isinstance(blocks, Mapping) else {}
    raw_order = block.get("seed_order") or list(seeds) or list(FROZEN_SEEDS)
    seed_order = [int(s) for s in raw_order]
    metrics = block.get("metrics") if isinstance(block.get("metrics"),
                                                 Mapping) else {}
    higher: Dict[str, bool] = {}
    shared: Dict[str, Any] = {}
    per_seed_counts: Dict[str, Any] = {}
    for metric_id, entry in dict(metrics or {}).items():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("higher_is_better") is not None:
            higher[str(metric_id)] = bool(entry["higher_is_better"])
        if entry.get("n_curves_shared_across_all_seeds") is not None:
            shared[str(metric_id)] = entry["n_curves_shared_across_all_seeds"]
        if entry.get("n_curves_per_seed") is not None:
            per_seed_counts[str(metric_id)] = entry["n_curves_per_seed"]
    modes = block.get("modes") if isinstance(block.get("modes"), Mapping) else {}
    return {
        "seed_order": seed_order,
        "draws": (int(block["draws"]) if _finite(block.get("draws"))
                  else BOOTSTRAP_DRAWS),
        "seed_root": (int(block["seed_root"]) if _finite(block.get("seed_root"))
                      else BOOTSTRAP_SEED_ROOT),
        "unit": str(block.get("unit") or "curve (reaction cluster)"),
        "modes": {str(k): dict(v) for k, v in dict(modes).items()
                  if isinstance(v, Mapping)},
        "confirmatory_mode": "per_seed_independent",
        "confirmatory_api": "paired_curve_bootstrap_multiseed",
        "audit_mode": "shared_intersection",
        "audit_api": "paired_curve_bootstrap",
        "metrics_higher_is_better": higher,
        "n_curves_shared_across_all_seeds": shared,
        "n_curves_per_seed": per_seed_counts,
        "consumption": str(block.get("consumption") or ""),
        "source": ("report.bootstrap_inputs (§9.4 producer contract)"
                   if block else
                   "frozen plan defaults (附录 B :1608-1609) — the document "
                   "carries no bootstrap_inputs block"),
        "rule": ("§9.3 :995-1003 / §9.4 :1014-1033 — the multi-seed "
                 "per-seed-independent bootstrap is CONFIRMATORY; the "
                 "all-seed shared-intersection bootstrap is retained as an "
                 "AUDIT output (its intersection collapses because each formal "
                 "seed evaluates its own inner-val sample, §9.5 :1037)"),
    }


def baseline_identity(baseline: Mapping[str, Any], *,
                      source_path: Any = "") -> Dict[str, Any]:
    """Extract and validate the frozen identity block of the §9.5 baseline.

    Refusal codes: ``V543_STATS_BASELINE_SCHEMA`` (not the M0/P0 document),
    ``V543_STATS_BASELINE_IDENTITY`` (code/spec/bridge/checkpoint SHA missing).
    """
    _require(str(baseline.get("schema")) == BASELINE_SCHEMA,
             "V543_STATS_BASELINE_SCHEMA",
             f"the H2/H3 denominator must be a {BASELINE_SCHEMA} document "
             f"(§9.5 :1037); got schema={baseline.get('schema')!r}")
    evaluator = baseline.get("evaluator")
    _require(isinstance(evaluator, Mapping),
             "V543_STATS_BASELINE_IDENTITY",
             "the baseline document carries no 'evaluator' identity block")
    code_sha = str(evaluator.get("code_sha256") or "")
    spec_sha = str(evaluator.get("spec_sha256") or "")
    bridge_sha = str(baseline.get("bridge_spec_sha256") or "")
    for name, value in (("evaluator.code_sha256", code_sha),
                        ("evaluator.spec_sha256", spec_sha),
                        ("bridge_spec_sha256", bridge_sha)):
        _require(bool(value), "V543_STATS_BASELINE_IDENTITY",
                 f"the baseline identity field {name} is empty — the SHA "
                 "chain of §9.5 :1037 is incomplete")
    checkpoints = baseline.get("checkpoints")
    _require(isinstance(checkpoints, Mapping) and bool(checkpoints),
             "V543_STATS_BASELINE_IDENTITY",
             "the baseline document carries no per-seed 'checkpoints' SHA map")
    ckpt = {}
    for seed, entry in dict(checkpoints).items():
        sha = str((entry or {}).get("sha256") or "")
        _require(bool(sha), "V543_STATS_BASELINE_IDENTITY",
                 f"baseline checkpoint sha for seed {seed} is empty")
        ckpt[str(seed)] = {"sha256": sha,
                           "path": str((entry or {}).get("path") or "")}
    seeds = [int(s) for s in (baseline.get("seeds") or [])]
    _require(bool(seeds), "V543_STATS_BASELINE_IDENTITY",
             "the baseline document declares no seeds")
    # every per-seed report must carry the same spec sha (§9.5 freeze)
    per_seed = baseline.get("per_seed")
    _require(isinstance(per_seed, Mapping) and bool(per_seed),
             "V543_STATS_BASELINE_IDENTITY",
             "the baseline document carries no 'per_seed' reports")
    for seed, report in dict(per_seed).items():
        got = str((report or {}).get("evaluator_spec_sha256") or "")
        _require(got == spec_sha, "V543_STATS_INSTRUMENT_DRIFT",
                 f"baseline per_seed[{seed}].evaluator_spec_sha256={got!r} "
                 f"differs from evaluator.spec_sha256={spec_sha!r} — the "
                 "document was produced by more than one spec")
        _require(str((report or {}).get("schema")) == EVAL_REPORT_SCHEMA,
                 "V543_STATS_BASELINE_SCHEMA",
                 f"baseline per_seed[{seed}].schema is "
                 f"{(report or {}).get('schema')!r}, expected "
                 f"{EVAL_REPORT_SCHEMA!r}")
    return {
        "schema": BASELINE_SCHEMA,
        "source_path": str(source_path),
        "campaign_id": str(baseline.get("campaign_id") or ""),
        "seeds": seeds,
        "evaluator": {
            "module": str(evaluator.get("module") or ""),
            "code_sha256": code_sha,
            "spec_sha256": spec_sha,
            "report_schema": str(evaluator.get("report_schema") or ""),
        },
        "bootstrap_contract": bootstrap_contract(baseline, seeds=seeds),
        "bridge_spec_sha256": bridge_sha,
        "checkpoints": ckpt,
        "per_seed_document_sha256": dict(
            baseline.get("per_seed_document_sha256") or {}),
        "decision_record": str(baseline.get("decision_record") or ""),
        "baseline_kind": str(_dig(baseline, ("baseline", "kind")) or
                             "M0P0_zero_residual"),
        "rule": ("plan §9.5 :1037 — the M0/P0 re-computation under the V543 "
                 "evaluator is the ONLY denominator; V542 archive numbers may "
                 "never be used (§1.6 :110-118)"),
    }


def identity_chain(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """The four SHAs a candidate MUST share with the baseline (§9.5)."""
    evaluator = identity.get("evaluator") or {}
    return {
        "evaluator_code_sha256": str(evaluator.get("code_sha256") or ""),
        "evaluator_spec_sha256": str(evaluator.get("spec_sha256") or ""),
        "bridge_spec_sha256": str(identity.get("bridge_spec_sha256") or ""),
        "checkpoints": {str(k): str((v or {}).get("sha256") or "")
                        for k, v in dict(identity.get("checkpoints")
                                         or {}).items()},
    }


def verify_identity_chain(*, candidate: Mapping[str, Any],
                          baseline: Mapping[str, Any],
                          run_id: str) -> Dict[str, Any]:
    """Compare a candidate's identity with the frozen baseline (§9.5 :1037).

    Any drift is a typed refusal ``V543_STATS_IDENTITY_MISMATCH``: a metric
    produced by a different instrument/Encoder is not comparable, so it may
    never enter an H2/H3 denominator or numerator.
    """
    base = identity_chain(baseline)
    seed = str(candidate.get("seed"))
    cand = {
        "evaluator_code_sha256":
            str((candidate.get("evaluator") or {}).get("code_sha256") or ""),
        "evaluator_spec_sha256":
            str((candidate.get("evaluator") or {}).get("spec_sha256") or ""),
        "bridge_spec_sha256": str(candidate.get("bridge_spec_sha256") or ""),
        "checkpoint_sha256": str((candidate.get("checkpoint") or {})
                                 .get("sha256") or ""),
    }
    problems: List[str] = []
    for key in ("evaluator_code_sha256", "evaluator_spec_sha256",
                "bridge_spec_sha256"):
        if cand[key] != base[key]:
            problems.append(f"{key}: candidate={cand[key]!r} "
                            f"baseline={base[key]!r}")
    base_ckpt = str(base["checkpoints"].get(seed) or "")
    if not cand["checkpoint_sha256"]:
        problems.append(f"checkpoint sha for seed {seed} is empty")
    elif base_ckpt and cand["checkpoint_sha256"] != base_ckpt:
        problems.append(f"checkpoint_sha256[seed {seed}]: "
                        f"candidate={cand['checkpoint_sha256']!r} "
                        f"baseline={base_ckpt!r}")
    if problems:
        raise V543StatsError(
            "V543_STATS_IDENTITY_MISMATCH",
            f"{run_id}: the candidate's evaluation identity differs from the "
            "frozen §9.5 baseline — refusing to compare different "
            "instruments/Encoders: " + "; ".join(problems))
    return {
        "run_id": run_id,
        "seed": seed,
        "chain": cand,
        "matches_baseline": True,
        "rule": ("§9.5 :1037 — evaluator code SHA + spec SHA + Bridge spec "
                 "SHA + per-seed checkpoint SHA must equal the baseline's"),
    }


def instrument_check(baseline: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify the on-disk evaluator still hashes to the baseline's claim.

    §9.5 :1037 freezes the instrument by SHA; a checkout whose evaluator file
    or spec document drifted cannot reproduce the frozen baseline, so this is
    reported (and, with ``--check-instrument``, enforced as a refusal).
    """
    identity = baseline_identity(baseline)
    module_path = PROJECT_ROOT / str(identity["evaluator"]["module"])
    out: Dict[str, Any] = {
        "module": str(identity["evaluator"]["module"]),
        "module_present": module_path.is_file(),
        "claimed_code_sha256": identity["evaluator"]["code_sha256"],
        "claimed_spec_sha256": identity["evaluator"]["spec_sha256"],
        "rule": "§9.5 :1037 / §9.4 — the instrument is frozen by SHA",
    }
    if not module_path.is_file():
        out.update({"checked": False, "matches_baseline": None,
                    "note": "evaluator module not present on this checkout"})
        return out
    actual_code = _sha256_file(module_path)
    try:
        actual_spec = str(_eval_module().evaluator_spec_sha256())
        spec_error = None
    except V543StatsError as exc:
        actual_spec = ""
        spec_error = f"{exc.code}: {exc.detail}"
    out.update({
        "checked": True,
        "actual_code_sha256": actual_code,
        "actual_spec_sha256": actual_spec,
        "code_matches": actual_code == identity["evaluator"]["code_sha256"],
        "spec_matches": actual_spec == identity["evaluator"]["spec_sha256"],
    })
    if spec_error:
        out["spec_error"] = spec_error
    out["matches_baseline"] = bool(out["code_matches"] and out["spec_matches"])
    return out


# ===========================================================================
# 1. extract — the per-run metric table with per-curve passthrough
# ===========================================================================
def pooled_metrics(report: Mapping[str, Any]) -> Dict[str, Any]:
    """The nine H2 leaves + the diagnostic leaves out of one eval report."""
    out: Dict[str, Any] = {}
    for metric_id, leaf in POOLED_LEAVES.items():
        value = _dig(report, leaf)
        out[metric_id] = (None if value is None else float(value))
    return out


def q_validity(report: Mapping[str, Any]) -> Dict[str, Any]:
    """§9.4 :1029–1033 — the q validity block, verbatim from the evaluator."""
    validity = _dig(report, ("validity", "q")) or {}
    frac = validity.get("frac_curves_ge3")
    out = {
        "n_eval_curves": validity.get("n_eval_curves"),
        "median_points_per_curve": validity.get("median_points_per_curve"),
        "frac_curves_ge3": frac,
        "frac_curves_ge5": validity.get("frac_curves_ge5"),
        "corr_min_points": int(validity.get("corr_min_points")
                               or Q_CORR_MIN_POINTS),
        "corr_diagnostic_only": bool(validity.get("corr_diagnostic_only")),
    }
    # Re-derive the demotion rule ourselves (§9.4 :1032) so the report can
    # never silently inherit a wrong flag from a producer.
    derived = (None if not _finite(frac)
               else bool(float(frac) < Q_FRAC_GE3_MIN))
    out["frac_ge3_threshold"] = Q_FRAC_GE3_MIN
    out["q_corr_diagnostic_only_derived"] = derived
    out["q_corr_role"] = ("diagnostic_only" if derived else "reported")
    out["q_corr_is_conclusion"] = False
    out["rule"] = ("§9.4 :1029-1033 — q.corr_curve_median is NEVER a "
                   "conclusion (q MAE + ordering are primary); a curve needs "
                   ">=3 valid q points; frac_curves_ge3 < 0.70 forces "
                   "diagnostic-only")
    if derived is not None and derived != bool(
            validity.get("corr_diagnostic_only")):
        out["flag_disagreement"] = (
            "the producer's corr_diagnostic_only flag disagrees with the "
            "§9.4 :1032 rule recomputed from frac_curves_ge3; the rule wins")
    return out


def curve_values(report: Mapping[str, Any], metric_id: str) -> Dict[str, float]:
    """Per-curve ``{curve_id: value}`` for one metric of one eval report."""
    leaf = CURVE_LEAVES.get(metric_id)
    _require(leaf is not None, "V543_STATS_METRIC_UNKNOWN",
             f"{metric_id!r} has no per-curve leaf in the V543 report")
    block, key = leaf
    values: Dict[str, float] = {}
    for row in report.get("per_curve") or []:
        value = (row.get(block) or {}).get(key)
        if value is not None and _finite(value):
            values[str(row.get("curve_id"))] = float(value)
    return values


def _curve_values_from_bootstrap_inputs(
        bootstrap_inputs: Mapping[str, Any], seed: str,
        metric_id: str) -> Optional[Dict[str, float]]:
    """The evaluator's own ``bootstrap_inputs`` table for one metric/seed."""
    metrics = (bootstrap_inputs or {}).get("metrics") or {}
    entry = metrics.get(metric_id)
    if not isinstance(entry, Mapping):
        return None
    per_seed = entry.get("per_seed_curve_values") or {}
    table = per_seed.get(str(seed))
    if not isinstance(table, Mapping):
        return None
    return {str(k): float(v) for k, v in table.items() if _finite(v)}


def curve_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The reduced per-curve passthrough rows (bootstrap cluster unit).

    Only the values a downstream bootstrap or validity read can consume are
    kept (``CURVE_LEAVES``); the full evaluator report stays in ``audit/``.
    """
    rows: List[Dict[str, Any]] = []
    for row in report.get("per_curve") or []:
        out: Dict[str, Any] = {"curve_id": str(row.get("curve_id"))}
        for metric_id, (block, key) in CURVE_LEAVES.items():
            value = (row.get(block) or {}).get(key)
            out[metric_id] = (None if value is None else float(value))
        out["n_frames"] = row.get("n_frames")
        out["endpoint.n_pos"] = (row.get("endpoint") or {}).get("n_pos")
        out["endpoint.n_neg"] = (row.get("endpoint") or {}).get("n_neg")
        out["ordering.n_pairs_full"] = (row.get("ordering") or {}).get(
            "n_pairs_full")
        rows.append(out)
    return rows


def seed_curve_values(report: Mapping[str, Any], seed: str,
                      bootstrap_inputs: Optional[Mapping[str, Any]] = None
                      ) -> Dict[str, Any]:
    """Per-metric curve tables of one seed + the source of each table.

    ``bootstrap_inputs`` (the evaluator's own §9.4 table) is preferred when
    present — it is the producer's frozen reading; the per-curve rows are the
    fallback.  When both exist they must agree exactly, otherwise the document
    is internally inconsistent (typed refusal).
    """
    out: Dict[str, Dict[str, float]] = {}
    sources: Dict[str, str] = {}
    for metric_id in tuple(CURVE_LEAVES):
        derived = curve_values(report, metric_id)
        declared = (None if bootstrap_inputs is None
                    else _curve_values_from_bootstrap_inputs(
                        bootstrap_inputs, str(seed), metric_id))
        if declared is not None:
            for curve in sorted(set(declared) | set(derived)):
                a, b = declared.get(curve), derived.get(curve)
                if a is None or b is None:
                    _require(False, "V543_STATS_BOOTSTRAP_INPUT_DRIFT",
                             f"seed {seed} metric {metric_id}: curve {curve} "
                             f"is present on only one side "
                             f"(bootstrap_inputs={a!r}, per_curve={b!r})")
                _require(abs(float(a) - float(b)) <= 1e-12,
                         "V543_STATS_BOOTSTRAP_INPUT_DRIFT",
                         f"seed {seed} metric {metric_id}: curve {curve} "
                         f"differs between bootstrap_inputs ({a!r}) and "
                         f"per_curve rows ({b!r})")
            out[metric_id] = declared
            sources[metric_id] = "report.bootstrap_inputs (§9.4)"
        else:
            out[metric_id] = derived
            sources[metric_id] = "report.per_curve rows (§9.4)"
    return {"values": out, "sources": sources}


def metrics_entry(*, run_id: str, candidate_id: str, track: str, seed: int,
                  stage: str, role: str, report: Mapping[str, Any],
                  identity: Mapping[str, Any],
                  bootstrap_inputs: Optional[Mapping[str, Any]],
                  source: Mapping[str, Any],
                  run_info: Optional[Mapping[str, Any]] = None
                  ) -> Dict[str, Any]:
    """One metrics-table entry: pooled leaves + per-curve passthrough + SHAs."""
    _require(str(report.get("schema")) == EVAL_REPORT_SCHEMA,
             "V543_STATS_EVAL_REPORT_SCHEMA",
             f"{run_id}: expected a {EVAL_REPORT_SCHEMA} document, got "
             f"{report.get('schema')!r}")
    curves = seed_curve_values(report, str(seed), bootstrap_inputs)
    per_curve = curve_rows(report)
    pooled = pooled_metrics(report)
    for metric_id in METRIC_IDS:
        _require(pooled.get(metric_id) is not None
                 or metric_id == "process.endpoint.recall_at_fpr_0p05",
                 "V543_STATS_METRIC_MISSING",
                 f"{run_id}: the H2 leaf {metric_id} is absent from the "
                 "evaluator report — the frozen gate cannot be evaluated")
    core = {
        "schema": METRICS_ENTRY_SCHEMA,
        "run_id": str(run_id),
        "candidate_id": str(candidate_id),
        "track": str(track),
        "seed": int(seed),
        "stage": str(stage),
        "role": str(role),
        "outcome": str((run_info or {}).get("outcome") or "done"),
        "evaluator": dict(identity.get("evaluator") or {}),
        "bridge_spec_sha256": str(identity.get("bridge_spec_sha256") or ""),
        "checkpoint": dict(identity.get("checkpoint") or {}),
        "metrics": pooled,
        "diagnostics": {k: pooled.get(k) for k in DIAGNOSTIC_METRIC_IDS},
        "q_validity": q_validity(report),
        "per_curve": per_curve,
        "curve_values": curves["values"],
        "curve_values_source": curves["sources"],
        "curve_values_higher_is_better": dict(CURVE_LEVEL_HIGHER_IS_BETTER),
        "n_curves": len(per_curve),
        "run": dict(run_info or {}),
        "source": dict(source),
        "frozen_bootstrap": dict(FROZEN_BOOTSTRAP),
    }
    digest_payload = {
        "run_id": core["run_id"],
        "candidate_id": core["candidate_id"],
        "track": core["track"],
        "seed": core["seed"],
        "evaluator": core["evaluator"],
        "bridge_spec_sha256": core["bridge_spec_sha256"],
        "checkpoint": core["checkpoint"],
        "metrics": core["metrics"],
        "per_curve": core["per_curve"],
        "source_sha256": core["source"].get("document_sha256"),
    }
    core["identity_sha256"] = _canonical_sha256(digest_payload)
    return core


def candidate_entries(bundle: Mapping[str, Any], *,
                      baseline: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalise one candidate bundle into (usually one) metrics entry."""
    _require(str(bundle.get("schema")) in CANDIDATE_BUNDLE_SCHEMAS,
             "V543_STATS_CANDIDATE_SCHEMA",
             f"a candidate document must be a {CANDIDATE_BUNDLE_SCHEMA} "
             f"bundle, got {bundle.get('schema')!r}")
    run_id = str(bundle.get("run_id") or "")
    _require(bool(run_id), "V543_STATS_CANDIDATE_IDENTITY",
             "the candidate bundle carries no run_id")
    identity = {
        "evaluator": dict(bundle.get("evaluator") or {}),
        "bridge_spec_sha256": str(bundle.get("bridge_spec_sha256") or ""),
        "checkpoint": dict(bundle.get("checkpoint") or {}),
    }
    seed = int(bundle.get("seed"))
    verify_identity_chain(candidate={**identity, "seed": seed},
                          baseline=baseline, run_id=run_id)
    report = bundle.get("eval_report")
    _require(isinstance(report, Mapping),
             "V543_STATS_CANDIDATE_SCHEMA",
             f"{run_id}: the bundle carries no 'eval_report' object")
    return [metrics_entry(
        run_id=run_id,
        candidate_id=str(bundle.get("candidate_id") or ""),
        track=str(bundle.get("track") or ""),
        seed=seed,
        stage=str(bundle.get("stage") or ""),
        role="candidate",
        report=report,
        identity=identity,
        bootstrap_inputs=bundle.get("bootstrap_inputs"),
        source={"path": str(bundle.get("path") or ""),
                "document_sha256": _canonical_sha256(dict(bundle))},
        run_info=dict(bundle.get("run") or {}),
    )]


def baseline_entries(baseline: Mapping[str, Any], *,
                     source_path: Any) -> List[Dict[str, Any]]:
    """One entry per seed of the frozen M0/P0 document (§9.5 :1037)."""
    identity = baseline_identity(baseline, source_path=source_path)
    bootstrap_inputs = baseline.get("bootstrap_inputs")
    entries: List[Dict[str, Any]] = []
    document_sha = dict(baseline.get("per_seed_document_sha256") or {})
    for seed_str, report in sorted(
            (str(k), v) for k, v in dict(baseline.get("per_seed") or {}).items()):
        checkpoint = dict((baseline.get("checkpoints") or {}).get(
            seed_str) or {})
        entries.append(metrics_entry(
            run_id=f"V543_M0P0_S{seed_str}",
            candidate_id=BASELINE_ROW_ID,
            track="both",
            seed=int(seed_str),
            stage="T0",
            role="baseline",
            report=report,
            identity={
                "evaluator": {"code_sha256":
                              identity["evaluator"]["code_sha256"],
                              "spec_sha256":
                              identity["evaluator"]["spec_sha256"],
                              "report_schema": EVAL_REPORT_SCHEMA},
                "bridge_spec_sha256": identity["bridge_spec_sha256"],
                "checkpoint": {
                    "path": str(checkpoint.get("path") or ""),
                    "sha256": str(checkpoint.get("sha256") or ""),
                },
            },
            bootstrap_inputs=(bootstrap_inputs
                              if isinstance(bootstrap_inputs, Mapping)
                              else None),
            source={"path": str(source_path),
                    "document_sha256": str(document_sha.get(seed_str) or "")},
            run_info={"outcome": "done", "baseline": True},
        ))
    return entries


def load_table(path: Any) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"schema": METRICS_TABLE_SCHEMA, "kind": "metrics_table",
                "created_utc": _now(), "entries": [], "append_log": []}
    doc = _load_json(p, "V543_STATS_TABLE_UNREADABLE", "the metrics table")
    _require(str(doc.get("schema")) == METRICS_TABLE_SCHEMA,
             "V543_STATS_TABLE_SCHEMA",
             f"{p} is not a {METRICS_TABLE_SCHEMA} document")
    _require(isinstance(doc.get("entries"), list),
             "V543_STATS_TABLE_SCHEMA",
             f"{p}: 'entries' must be a list")
    return doc


def append_entries(table: Mapping[str, Any],
                   entries: Sequence[Mapping[str, Any]], *,
                   note: str = "") -> Dict[str, Any]:
    """Append-only merge: an existing identity triple is never rewritten.

    A row whose (run_id, seed) already exists under a DIFFERENT instrument SHA
    is a typed refusal (``V543_STATS_TABLE_IDENTITY_MIX``): §9.5 :1037 freezes
    the instrument, so one table may never mix measurements from two
    evaluator generations.  A re-instrumented baseline needs a fresh table.
    """
    out = dict(table)
    out["entries"] = [dict(e) for e in (table.get("entries") or [])]
    out.setdefault("append_log", [])
    out["append_log"] = [dict(x) for x in (table.get("append_log") or [])]
    existing = {(str(e.get("run_id")), int(e.get("seed", -1)),
                 str(e.get("identity_sha256")))
                for e in out["entries"]}
    existing_identity = {(str(e.get("run_id")), int(e.get("seed", -1))):
                         (str((e.get("evaluator") or {}).get("code_sha256")),
                          str((e.get("evaluator") or {}).get("spec_sha256")))
                         for e in out["entries"]}
    added: List[str] = []
    skipped: List[str] = []
    for entry in entries:
        key = (str(entry.get("run_id")), int(entry.get("seed", -1)),
               str(entry.get("identity_sha256")))
        slot = key[:2]
        if slot in existing_identity:
            prior = existing_identity[slot]
            now = (str((entry.get("evaluator") or {}).get("code_sha256")),
                   str((entry.get("evaluator") or {}).get("spec_sha256")))
            _require(prior == now, "V543_STATS_TABLE_IDENTITY_MIX",
                     f"{key[0]}/S{key[1]} is already in the table under "
                     f"evaluator code/spec {prior} but this extraction was "
                     f"produced under {now} — §9.5 :1037 freezes the "
                     "instrument, so one metrics table may not mix two "
                     "evaluator generations. Use a fresh campaign root (or "
                     "archive the old table) after an instrument bump.")
        if key in existing:
            skipped.append(f"{key[0]}/S{key[1]}")
            continue
        out["entries"].append(dict(entry))
        existing.add(key)
        existing_identity[slot] = (
            str((entry.get("evaluator") or {}).get("code_sha256")),
            str((entry.get("evaluator") or {}).get("spec_sha256")))
        added.append(f"{key[0]}/S{key[1]}")
    out["append_log"].append({
        "at": _now(), "added": added, "skipped_idempotent": skipped,
        "note": note,
    })
    out["schema"] = METRICS_TABLE_SCHEMA
    out["kind"] = "metrics_table"
    out["n_entries"] = len(out["entries"])
    out["discipline"] = ("append-only: records are never rewritten; an "
                         "already-present (run_id, seed, identity_sha256) "
                         "triple is skipped idempotently; a changed "
                         "instrument SHA for an existing (run_id, seed) is a "
                         "typed refusal")
    return out


def latest_by_run(table: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Latest record per run (last record wins, the ledger discipline)."""
    latest: Dict[str, Dict[str, Any]] = {}
    for entry in table.get("entries") or []:
        latest[str(entry.get("run_id"))] = dict(entry)
    return latest


def rows_by_seed(table: Mapping[str, Any],
                 candidate_id: str) -> Dict[int, Dict[str, Any]]:
    """Latest entry per seed of one candidate id."""
    out: Dict[int, Dict[str, Any]] = {}
    for entry in table.get("entries") or []:
        if str(entry.get("candidate_id")) != str(candidate_id):
            continue
        out[int(entry.get("seed"))] = dict(entry)
    return out


def baseline_row(table: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """The same-seed M0/P0 row of the §9.5 baseline (the only denominator).

    Last record wins (the ledger discipline): if a table were ever to hold two
    generations of a row, the newest one is the denominator and the identity-mix
    guard in :func:`append_entries` would already have refused the append.
    """
    found: Optional[Dict[str, Any]] = None
    for entry in table.get("entries") or []:
        if (str(entry.get("role")) == "baseline"
                and int(entry.get("seed", -1)) == int(seed)):
            found = dict(entry)
    if found is not None:
        return found
    raise V543StatsError(
        "V543_STATS_BASELINE_MISSING",
        f"no baseline (M0/P0) entry for seed {seed} in the metrics table — "
        "run `extract` with the §9.5 baseline document first")


def table_bootstrap_contract(table: Mapping[str, Any]) -> Dict[str, Any]:
    """The bootstrap contract of the table (producer values, frozen defaults)."""
    stored = table.get("bootstrap_contract")
    if isinstance(stored, Mapping) and stored.get("seed_order"):
        return dict(stored)
    identity = dict(table.get("baseline_identity") or {})
    if isinstance(identity.get("bootstrap_contract"), Mapping):
        return dict(identity["bootstrap_contract"])
    return bootstrap_contract({}, seeds=FROZEN_SEEDS)


def candidate_ids(table: Mapping[str, Any], *,
                  include_baseline: bool = False) -> List[str]:
    ids = {str(e.get("candidate_id")) for e in table.get("entries") or []
           if str(e.get("role")) == "candidate"}
    if include_baseline:
        ids |= {str(e.get("candidate_id")) for e in table.get("entries") or []
                if str(e.get("role")) == "baseline"}
    return sorted(i for i in ids if i)


def run_extract(cfg: Mapping[str, Any], *, campaign_root: Any,
                baseline_path: Any = None,
                candidates: Sequence[Any] = (),
                check_instrument: bool = False) -> Dict[str, Any]:
    """Build/extend ``reports/metrics_table.json`` (append-only)."""
    root = Path(campaign_root)
    baseline_file = (Path(baseline_path) if baseline_path is not None
                     else root / DEFAULT_BASELINE_REL)
    baseline = _load_json(baseline_file, "V543_STATS_BASELINE_MISSING",
                          "the §9.5 M0/P0 baseline document")
    identity = baseline_identity(baseline, source_path=baseline_file)
    instrument = (instrument_check(baseline) if check_instrument else
                  {"checked": False, "requested": False,
                   "rule": "§9.5 :1037 — pass --check-instrument to enforce"})
    if check_instrument:
        _require(bool(instrument.get("matches_baseline")),
                 "V543_STATS_INSTRUMENT_DRIFT",
                 "the on-disk V543 evaluator no longer hashes to the frozen "
                 f"baseline claim ({instrument}) — a metric produced now is "
                 "not comparable with the frozen denominator (§9.5 :1037)")
    entries = baseline_entries(baseline, source_path=baseline_file)
    for path in candidates:
        bundle = _load_json(path, "V543_STATS_CANDIDATE_MISSING",
                            f"candidate bundle {path}")
        bundle = dict(bundle)
        bundle.setdefault("path", str(path))
        entries.extend(candidate_entries(bundle, baseline=baseline))
    table = append_entries(
        load_table(root / METRICS_TABLE_REL), entries,
        note="extract: §9.5 baseline rows + candidate bundles")
    table["baseline_identity"] = identity
    table["bootstrap_contract"] = dict(identity["bootstrap_contract"])
    table["instrument_check"] = instrument
    table["plan_section"] = ("§9.1-§9.5 / 附录 B :1585-1609 — extract is the "
                             "only producer of the metrics table; the "
                             "denominator is the §9.5 M0/P0 re-computation")
    table["thresholds"] = {k: dict(v) for k, v in FROZEN_THRESHOLDS.items()}
    path = _write_report(root, METRICS_TABLE_REL, table)
    out = {
        "schema": METRICS_TABLE_SCHEMA,
        "kind": "metrics_table",
        "table_path": str(path),
        "baseline_identity": identity,
        "instrument_check": instrument,
        "n_entries": len(table["entries"]),
        "append_log": table["append_log"][-1],
        "runs": [{"run_id": e["run_id"], "candidate_id": e["candidate_id"],
                  "track": e["track"], "seed": e["seed"],
                  "role": e["role"],
                  "metrics": e["metrics"],
                  "q_validity": e["q_validity"],
                  "n_curves": e["n_curves"],
                  "identity_sha256": e["identity_sha256"]}
                 for e in table["entries"]],
        "config_consistency": config_consistency(cfg),
        "written_at": _now(),
    }
    return out


# ===========================================================================
# config vs 附录 B — report a disagreement, never "fix" it
# ===========================================================================
def config_consistency(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare the campaign config's frozen numbers with 附录 B.

    The plan is authoritative (the brief: "以附录 B 数值为准"); a config that
    disagrees is REPORTED here and never silently corrected.
    """
    thresholds = dict(cfg.get("thresholds") or {})
    evaluation = dict(cfg.get("evaluation") or {})
    checks: Dict[str, Any] = {}
    for key, spec in FROZEN_THRESHOLDS.items():
        have = thresholds.get(key)
        if have is None and key in ("bootstrap_draws", "bootstrap_seed_root"):
            have = evaluation.get(key)
        checks[key] = {
            "plan_value": spec["value"],
            "config_value": have,
            "plan_line": spec["plan_line"],
            "matches": (have is not None and _finite(have) and
                        abs(float(have) - float(spec["value"])) <= 0.0),
        }
    for key in ("bootstrap_draws", "bootstrap_seed_root"):
        have = evaluation.get(key, thresholds.get(key))
        checks[key] = {
            "plan_value": FROZEN_BOOTSTRAP[key],
            "config_value": have,
            "plan_line": FROZEN_BOOTSTRAP["plan_line"],
            "matches": (have is not None and
                        int(have) == int(FROZEN_BOOTSTRAP[key])),
            "block": "evaluation (附录 B lists it under thresholds — "
                     "structural difference, value agrees)",
        }
    mismatched = sorted(k for k, v in checks.items() if not v["matches"])
    return {
        "checks": checks,
        "mismatched": mismatched,
        "n_checked": len(checks),
        "rule": ("附录 B :1585-1609 is authoritative; every disagreement is "
                 "reported here and NEVER patched by this lane"),
        "side_note": ("附录 B :1608-1609 lists bootstrap_draws/"
                      "bootstrap_seed_root inside the thresholds table while "
                      "the campaign config carries them under `evaluation` — "
                      "a structural, not a numeric, difference"),
    }


# ===========================================================================
# 2. h0 — legality / equivalence checklist (§9.3 :916–927)
# ===========================================================================
def h0_checklist(evidence: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The frozen H0 checklist, evaluated against supplied evidence.

    No evidence ⇒ every item is ``pending`` and the verdict is
    ``incomplete`` (never ``pass``): this lane can record H0, it cannot
    conjure the tests that prove it.  ``fail`` on any item ⇒ ``invalid_run``
    (:916).
    """
    items = dict((evidence or {}).get("items") or {})
    unknown = sorted(set(items) - set(H0_ITEM_IDS))
    _require(not unknown, "V543_STATS_H0_ITEM_UNKNOWN",
             f"the H0 evidence names unknown item ids {unknown}; the frozen "
             f"checklist is {list(H0_ITEM_IDS)}")
    rows: List[Dict[str, Any]] = []
    for spec in H0_ITEMS:
        entry = dict(items.get(spec["id"]) or {})
        status = str(entry.get("status") or "pending")
        _require(status in ("pass", "fail", "pending", "waived"),
                 "V543_STATS_H0_STATUS",
                 f"{spec['id']}: status must be pass/fail/pending/waived, "
                 f"got {status!r}")
        rows.append({
            "id": spec["id"],
            "bullet": spec["bullet"],
            "quote": spec["quote"],
            "plan_line": spec["plan_line"],
            "check": spec["check"],
            "related_test": spec["related_test"],
            "evidence_kind": spec["evidence_kind"],
            "status": status,
            "evidence": entry.get("evidence"),
            "evidence_sha256": (entry.get("evidence_sha256")
                                or (None if not entry.get("evidence")
                                    else hashlib.sha256(str(
                                        entry["evidence"]).encode("utf-8")
                                    ).hexdigest())),
            "reported_at": entry.get("reported_at"),
        })
    n_pass = sum(1 for r in rows if r["status"] == "pass")
    n_fail = sum(1 for r in rows if r["status"] == "fail")
    n_pending = sum(1 for r in rows if r["status"] == "pending")
    n_waived = sum(1 for r in rows if r["status"] == "waived")
    if n_fail:
        verdict = "invalid_run"
    elif n_pending:
        verdict = "incomplete"
    else:
        verdict = "pass"
    n_bullets = len({int(spec["bullet"]) for spec in H0_ITEMS})
    return {
        "schema": H0_REPORT_SCHEMA,
        "kind": "h0_legality",
        "gate": "H0",
        "rule": ("§9.3 :916 — 合法性与等价性；任一失败 = invalid_run. A "
                 "missing measurement is `incomplete`, never a pass."),
        "plan_lines": "§9.3 :916-927",
        "items": rows,
        "verdict": verdict,
        "counts": {
            "n_plan_bullets": n_bullets,
            "n_atomic_checks": len(rows),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_pending": n_pending,
            "n_waived": n_waived,
            "brief_expected_count": 12,
            "counting_note": (
                "§9.3 :917-938 lists 10 bullets; two of them are compound "
                "(E(3) names four symmetries; '有限梯度、有限 loss' names two "
                "checkables), which this lane expands into 14 atomic checks. "
                "The task brief says '12 项' — that number matches NEITHER "
                "reading of the plan text, so both counts are reported and "
                "the plan text wins (see ambiguities)."),
        },
        "run_id": (evidence or {}).get("run_id"),
        "candidate_id": (evidence or {}).get("candidate_id"),
        "evidence_document_sha256": (None if evidence is None
                                     else _canonical_sha256(dict(evidence))),
        "written_at": _now(),
    }


def run_h0(cfg: Mapping[str, Any], *, campaign_root: Any,
           evidence_path: Any = None) -> Dict[str, Any]:
    evidence = None
    if evidence_path:
        evidence = _load_json(evidence_path, "V543_STATS_H0_EVIDENCE_MISSING",
                              "the H0 evidence document")
        _require(str(evidence.get("schema")) == H0_EVIDENCE_SCHEMA,
                 "V543_STATS_H0_EVIDENCE_SCHEMA",
                 f"the H0 evidence must be a {H0_EVIDENCE_SCHEMA} document, "
                 f"got {evidence.get('schema')!r}")
    report = h0_checklist(evidence)
    report["config_consistency"] = config_consistency(cfg)
    path = _write_report(campaign_root, H0_REPORT_REL, report)
    report["report_path"] = str(path)
    return report


# ===========================================================================
# 3. h1 — resource gate (§9.3 :929–935); inputs may be pending
# ===========================================================================
H1_FIELD_DOC: Final[Mapping[str, str]] = {
    "run_id": "the formal run this measurement belongs to",
    "track": "move | process",
    "decoder_id": "M1..M3 / P1..P4 (registry id)",
    "single_lane": "§9.3 :931 — true iff the run used one GPU lane",
    "swap_events": "§9.3 :931 — count of swap events (must be 0)",
    "oom_events": "§9.3 :931 — count of OOM kills (must be 0)",
    "host_peak_gib": "§9.3 :932 — host memory peak (≤12 GB)",
    "vram_peak_gib": "§9.3 :932 — VRAM peak",
    "vram_limit_gib": "§9.3 :932 — the card's limit the peak is compared to",
    "core_time_candidate_s": "§9.3 :933 — candidate core time (s)",
    "core_time_denominator_s": "§9.3 :933/:1037 — SAME-environment M0/P0 core "
                               "time (s); a V542 archive denominator is a "
                               "typed refusal",
    "core_time_denominator_source": "must be 'same_env_m0p0_v543'",
    "p95_latency_candidate_s": "§9.3 :934 / 附录 B :1607 — candidate p95 "
                               "latency (s)",
    "p95_latency_a0_s": "§9.3 :934 — A0 p95 latency in the same environment",
    "n_trainable_params": "§9.3 :935 — measured trainable parameter count",
}


def evaluate_h1(input_doc: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Evaluate the H1 resource gate for every candidate the input names.

    Every field is OPTIONAL: an absent measurement yields ``pending`` (with
    the field named) and the gate verdict can never be ``pass``.  The input
    interface is frozen here so WP7's profile can be dropped in unchanged.
    """
    doc = dict(input_doc or {})
    if doc:
        _require(str(doc.get("schema")) == H1_INPUT_SCHEMA,
                 "V543_STATS_H1_SCHEMA",
                 f"the H1 input must be a {H1_INPUT_SCHEMA} document, got "
                 f"{doc.get('schema')!r}")
    per_candidate = dict(doc.get("per_candidate") or {})
    band: Optional[Dict[str, Any]] = None
    rows: Dict[str, Any] = {}
    for decoder_id in sorted(per_candidate):
        raw = dict(per_candidate[decoder_id] or {})
        checks: Dict[str, Any] = {}

        def _pending(reason: str) -> Dict[str, Any]:
            return {"status": "pending", "pass": None, "reason": reason,
                    "plan_line": ""}

        # -- §9.3 :931 single lane / no swap / no OOM -----------------------
        core_inputs = {k: raw.get(k) for k in
                       ("single_lane", "swap_events", "oom_events")}
        if all(v is not None for v in core_inputs.values()):
            ok = (bool(core_inputs["single_lane"])
                  and int(core_inputs["swap_events"]) == 0
                  and int(core_inputs["oom_events"]) == 0)
            checks["single_lane_no_swap_no_oom"] = {
                "status": "pass" if ok else "fail",
                "pass": bool(ok),
                "observed": core_inputs,
                "rule": "single lane, swap = 0, OOM = 0",
                "plan_line": "§9.3 :931 / §8.3 :840 / §8.5 :862",
            }
        else:
            checks["single_lane_no_swap_no_oom"] = _pending(
                "single_lane/swap_events/oom_events not measured")
            checks["single_lane_no_swap_no_oom"]["plan_line"] = "§9.3 :931"

        # -- §9.3 :932 host ≤ 12 GB ----------------------------------------
        host = raw.get("host_peak_gib")
        host_limit = float(raw.get("host_limit_gib") or 12.0)
        if _finite(host):
            ok = _limit_ok(float(host), host_limit)
            checks["host_peak_within_12gib"] = {
                "status": "pass" if ok else "fail",
                "pass": bool(ok),
                "observed_gib": float(host),
                "limit_gib": host_limit,
                "plan_line": "§9.3 :932 / §8.3 :840 (host ≤12 GB)",
            }
        else:
            checks["host_peak_within_12gib"] = _pending(
                "host_peak_gib not measured")
            checks["host_peak_within_12gib"]["plan_line"] = "§9.3 :932"

        # -- §9.3 :932 VRAM inside the card's limit ------------------------
        vram, vram_limit = raw.get("vram_peak_gib"), raw.get("vram_limit_gib")
        if _finite(vram) and _finite(vram_limit):
            ok = _limit_ok(float(vram), float(vram_limit))
            checks["vram_peak_within_limit"] = {
                "status": "pass" if ok else "fail",
                "pass": bool(ok),
                "observed_gib": float(vram),
                "limit_gib": float(vram_limit),
                "plan_line": "§9.3 :932 (VRAM 不超线)",
            }
        else:
            checks["vram_peak_within_limit"] = _pending(
                "vram_peak_gib/vram_limit_gib not measured")
            checks["vram_peak_within_limit"]["plan_line"] = "§9.3 :932"

        # -- §9.3 :933 core-time ratio ≤ 2.0 -------------------------------
        cand_t = raw.get("core_time_candidate_s")
        base_t = raw.get("core_time_denominator_s")
        denom_source = str(raw.get("core_time_denominator_source") or "")
        if denom_source and denom_source != "same_env_m0p0_v543":
            raise V543StatsError(
                "V543_STATS_COST_DENOMINATOR_SOURCE",
                f"{decoder_id}: core_time_denominator_source="
                f"{denom_source!r}; §9.3 :933 requires the SAME-environment "
                "M0/P0 denominator and §9.5 :1037 forbids a V542 archive "
                "number as a denominator")
        if _finite(cand_t) and _finite(base_t) and float(base_t) > 0.0:
            ratio = float(cand_t) / float(base_t)
            ok = _limit_ok(ratio, MAX_CORE_TIME_RATIO)
            checks["core_time_ratio_le_2"] = {
                "status": "pass" if ok else "fail",
                "pass": bool(ok),
                "ratio": float(ratio),
                "limit": MAX_CORE_TIME_RATIO,
                "candidate_s": float(cand_t),
                "denominator_s": float(base_t),
                "denominator_source": denom_source or "same_env_m0p0_v543",
                "on_fail": ("§8.5 :865 — gate_skipped_cost_ratio_exceeds_2: "
                            "the scientific result is RETAINED, the candidate "
                            "does not advance"),
                "plan_line": "§9.3 :933 / 附录 B :1606",
            }
        else:
            checks["core_time_ratio_le_2"] = _pending(
                "core_time_candidate_s/core_time_denominator_s not measured")
            checks["core_time_ratio_le_2"]["plan_line"] = "§9.3 :933"

        # -- §9.3 :934 p95 latency ≤ 2× A0 ---------------------------------
        p_cand, p_a0 = (raw.get("p95_latency_candidate_s"),
                        raw.get("p95_latency_a0_s"))
        if _finite(p_cand) and _finite(p_a0) and float(p_a0) > 0.0:
            ratio = float(p_cand) / float(p_a0)
            ok = _limit_ok(ratio, MAX_LATENCY_RATIO_P95)
            checks["p95_latency_ratio_le_2"] = {
                "status": "pass" if ok else "fail",
                "pass": bool(ok),
                "ratio": float(ratio),
                "limit": MAX_LATENCY_RATIO_P95,
                "candidate_s": float(p_cand),
                "a0_s": float(p_a0),
                "plan_line": "§9.3 :934 / 附录 B :1607 / §8.3 :838",
            }
        else:
            checks["p95_latency_ratio_le_2"] = _pending(
                "p95_latency_candidate_s/p95_latency_a0_s not measured")
            checks["p95_latency_ratio_le_2"]["plan_line"] = "§9.3 :934"

        # -- §9.3 :935 parameters inside the §6.4 band ---------------------
        params = raw.get("n_trainable_params")
        if params is not None:
            if band is None:
                band = _registry_band()
            value = int(params)
            registry_agrees = None
            if raw.get("decoder_id"):
                try:
                    from pes2ts_core.pesn.models.v543.registry import (
                        describe_decoder)
                    described = describe_decoder(str(raw["decoder_id"]),
                                                 build=False)
                    registry_agrees = (
                        int(described.get("n_trainable_params", -1)) == value)
                except Exception:  # noqa: BLE001 - registry check is optional
                    registry_agrees = None
            ok = bool(band["band_min"] <= value <= band["band_max"])
            checks["params_in_budget_band"] = {
                "status": "pass" if ok else "fail",
                "pass": ok,
                "n_trainable_params": value,
                "band": [band["band_min"], band["band_max"]],
                "target": band["target"],
                "band_pct": band["band_pct"],
                "hard_cap": band["hard_cap"],
                "registry_agrees": registry_agrees,
                "plan_line": "§9.3 :935 / §6.4 :558-571 / §8.3 :835",
            }
        else:
            checks["params_in_budget_band"] = _pending(
                "n_trainable_params not measured")
            checks["params_in_budget_band"]["plan_line"] = "§9.3 :935"

        failed = sorted(k for k, v in checks.items() if v["status"] == "fail")
        pending = sorted(k for k, v in checks.items()
                         if v["status"] == "pending")
        cost_failed = "core_time_ratio_le_2" in failed
        verdict = ("pass" if not failed and not pending
                   else "resource_failure" if failed
                   else "pending_wp7_profiling")
        rows[decoder_id] = {
            "decoder_id": decoder_id,
            "run_id": raw.get("run_id"),
            "track": raw.get("track"),
            "checks": checks,
            "failed": failed,
            "pending": pending,
            "verdict": verdict,
            "disposition": ("gate_skipped_cost_ratio_exceeds_2"
                            if cost_failed else
                            "resource_failure" if failed else
                            "not_evaluated" if pending else "advance"),
            "on_breach": ["resource_failure", "budget_censored"],
        }
    return {
        "schema": H1_REPORT_SCHEMA,
        "kind": "h1_resource",
        "gate": "H1",
        "rule": ("§9.3 :929-935 — single lane / no swap / no OOM; host ≤12 GB; "
                 "VRAM 不超线; core-time ratio ≤2.0 versus the same-env M0/P0; "
                 "p95 latency ≤2× A0; parameters inside the §6.4 band. Any "
                 "failure = resource_failure / budget_censored."),
        "input_schema": H1_INPUT_SCHEMA,
        "input_field_doc": dict(H1_FIELD_DOC),
        "thresholds": {
            "max_core_time_ratio": MAX_CORE_TIME_RATIO,
            "max_latency_ratio_p95": MAX_LATENCY_RATIO_P95,
            "host_limit_gib": 12.0,
            "plan_line": "附录 B :1606-1607 / §9.3 :931-935",
        },
        "param_band": band,
        "per_candidate": rows,
        "n_candidates": len(rows),
        "data_status": ("no_input_measured" if not rows else
                        "pending_wp7_profiling" if any(
                            r["pending"] for r in rows.values()) else "complete"),
        "written_at": _now(),
    }


def run_h1(cfg: Mapping[str, Any], *, campaign_root: Any,
           input_path: Any = None) -> Dict[str, Any]:
    doc = None
    if input_path:
        doc = _load_json(input_path, "V543_STATS_H1_INPUT_MISSING",
                         "the H1 measurement input")
    report = evaluate_h1(doc)
    report["config_consistency"] = config_consistency(cfg)
    path = _write_report(campaign_root, H1_REPORT_REL, report)
    report["report_path"] = str(path)
    return report


# ===========================================================================
# 4. h2 — per-seed non-inferiority (§9.3 :937–960)
# ===========================================================================
def h2_check(spec: Mapping[str, Any], *, candidate: float,
             baseline: float) -> Dict[str, Any]:
    """One H2 guard on one seed (``degradation <= limit``).

    ``degradation`` is direction-aware (``baseline − candidate`` for
    higher-is-better metrics, ``candidate − baseline`` otherwise), so a
    positive number is always bad and every gate is the same ``≤`` test.
    """
    _require(_finite(candidate) and _finite(baseline),
             "V543_STATS_METRIC_MISSING",
             f"H2 {spec['id']}: candidate={candidate!r} "
             f"baseline={baseline!r} is not finite — the frozen gate cannot "
             "be evaluated")
    cand, base = float(candidate), float(baseline)
    higher = bool(spec.get("higher_is_better"))
    worse = (base - cand) if higher else (cand - base)
    kind, limit = str(spec["kind"]), float(spec["limit"])
    mode = kind
    if kind == "relative_degradation":
        cls = ZERO_BASELINE_CLASS.get(str(spec["metric"]), "ratio")
        abs_limit = (ZERO_BASELINE_ABS_MAX_RATIO_CLASS if cls == "ratio"
                     else ZERO_BASELINE_ABS_MAX_DISTANCE_CLASS)
        if abs(base) <= ZERO_BASELINE_EPS:
            mode = f"absolute_fallback_zero_baseline_{cls}"
            passed = _limit_ok(worse, abs_limit)
            return {
                "id": spec["id"], "block": spec["block"],
                "metric": spec["metric"], "kind": kind,
                "higher_is_better": higher, "mode": mode,
                "candidate": cand, "baseline": base,
                "observed": float(worse),
                "degradation": float(worse),
                "limit": float(abs_limit),
                "pass": bool(passed), "status": "pass" if passed else "fail",
                "threshold_key": spec["threshold_key"],
                "plan_line": spec["plan_line"] + " / §9.3 :960",
                "rule": ("§9.3 :960 — a relative threshold degenerates to an "
                         f"absolute one (≤{abs_limit}) when the baseline is "
                         "zero or near zero; never pick the more favourable "
                         "reading after the fact"),
                "zero_baseline_eps": ZERO_BASELINE_EPS,
            }
        observed = worse / base
        passed = _limit_ok(observed, limit)
    elif kind in ("absolute_drop", "absolute_increase"):
        observed = worse
        passed = _limit_ok(observed, limit)
    else:  # pragma: no cover - frozen table only
        raise V543StatsError("V543_STATS_H2_KIND",
                             f"unknown H2 kind {kind!r} in {spec['id']}")
    return {
        "id": spec["id"], "block": spec["block"], "metric": spec["metric"],
        "kind": kind, "higher_is_better": higher, "mode": mode,
        "candidate": cand, "baseline": base,
        "observed": float(observed),
        "degradation": float(observed),
        "limit": float(limit),
        "pass": bool(passed), "status": "pass" if passed else "fail",
        "threshold_key": spec["threshold_key"],
        "plan_line": spec["plan_line"],
        "rule": ("degradation ≤ limit, where degradation = baseline − "
                 "candidate for a higher-is-better metric and candidate − "
                 "baseline otherwise; `<=` — a value exactly at the limit is "
                 "non-inferior"),
    }


def h2_seed_entry(candidate_row: Mapping[str, Any],
                  baseline: Mapping[str, Any]) -> Dict[str, Any]:
    """All nine H2 guards for one candidate seed."""
    seed = int(candidate_row.get("seed"))
    metrics = dict(candidate_row.get("metrics") or {})
    base_metrics = dict(baseline.get("metrics") or {})
    checks: Dict[str, Any] = {}
    for spec in H2_SPECS:
        checks[spec["id"]] = h2_check(
            spec,
            candidate=metrics.get(spec["metric"]),
            baseline=base_metrics.get(spec["metric"]))
    breached = sorted(k for k, v in checks.items() if not v["pass"])
    return {
        "seed": seed,
        "candidate_id": str(candidate_row.get("candidate_id")),
        "track": str(candidate_row.get("track")),
        "baseline_run_id": str(baseline.get("run_id")),
        "identity_sha256": candidate_row.get("identity_sha256"),
        "baseline_identity_sha256": baseline.get("identity_sha256"),
        "source_sha256": (candidate_row.get("source") or {}).get(
            "document_sha256"),
        "baseline_source_sha256": (baseline.get("source") or {}).get(
            "document_sha256"),
        "n_checks": len(checks),
        "checks": checks,
        "breached_guards": breached,
        "h2_no_breach": not breached,
        "verdict": "pass" if not breached else "scientific_failure",
        "rule": ("§9.3 :937-958 — MOVE four items vs same-seed M0, PROCESS "
                 "five items vs same-seed P0; ANY breach on ANY formal seed "
                 "⇒ scientific_failure with the result retained and no "
                 "cross-track compensation"),
    }


def build_h2_report(*, table: Mapping[str, Any],
                    self_comparison: bool = False) -> Dict[str, Any]:
    """The H2 report over every candidate in the metrics table.

    ``self_comparison`` additionally evaluates the baseline rows against
    themselves — the denominator sanity check (Δ = 0 everywhere, so every
    guard must pass by construction).
    """
    runs: Dict[str, Any] = {}
    for candidate_id in candidate_ids(table, include_baseline=self_comparison):
        seed_rows = rows_by_seed(table, candidate_id)
        if not seed_rows:
            continue
        per_seed: Dict[str, Any] = {}
        for seed in sorted(seed_rows):
            entry = h2_seed_entry(seed_rows[seed],
                                  baseline_row(table, seed))
            per_seed[str(seed)] = entry
            runs[f"{candidate_id}/S{seed}"] = entry
        breached = [f"S{s}" for s, e in per_seed.items()
                    if not e["h2_no_breach"]]
        missing = [s for s in FROZEN_SEEDS if str(s) not in per_seed]
        runs[f"{candidate_id}"] = {
            "candidate_id": candidate_id,
            "track": str((seed_rows[sorted(seed_rows)[0]]).get("track")),
            "per_seed": per_seed,
            "seeds_present": sorted(int(s) for s in per_seed),
            "seeds_missing": missing,
            "breached_seeds": breached,
            "h2_no_breach": not breached,
            "verdict": ("pass" if not breached
                        else "scientific_failure"),
            "three_seed_complete": not missing,
            "kind": "candidate_summary",
            "rule": ("§9.3 :958 — every formal seed must be free of H2 "
                     "breach; the candidate's verdict is the AND over seeds"),
        }
    n_candidates = len({k for k, v in runs.items()
                        if v.get("kind") == "candidate_summary"})
    return {
        "schema": H2_REPORT_SCHEMA,
        "kind": "h2",
        "gate": "H2",
        "plan_lines": "§9.3 :937-960 / 附录 B :1589-1600",
        "rule": ("§9.3 :937-956 — per seed, MOVE 4 items vs the same-seed V543 "
                 "M0 and PROCESS 5 items vs the same-seed V543 P0; values from "
                 "附录 B. Per-seed verdicts and a three-seed summary are both "
                 "reported; on_fail = scientific_failure (the result is "
                 "retained, the run ledger is never rewritten)."),
        "thresholds": {k: dict(v) for k, v in FROZEN_THRESHOLDS.items()},
        "zero_baseline_rule": {
            "eps": ZERO_BASELINE_EPS,
            "ratio_class_abs_max": ZERO_BASELINE_ABS_MAX_RATIO_CLASS,
            "distance_class_abs_max_angstrom":
                ZERO_BASELINE_ABS_MAX_DISTANCE_CLASS,
            "plan_line": "§9.3 :960",
            "note": ("the plan fixes the absolute limits but NOT the 'near "
                     "zero' epsilon; this lane fixes 1e-6 and reports it as an "
                     "ambiguity (the rule itself must be written in WP0)"),
        },
        "seeds": list(FROZEN_SEEDS),
        "n_candidates": n_candidates,
        "runs": runs,
        "denominator": {
            "kind": "V543_recomputed_M0P0",
            "plan_line": "§9.5 :1037",
            "table_baseline_identity": dict(
                table.get("baseline_identity") or {}),
            "v542_archive_used": False,
        },
        "written_at": _now(),
    }


def run_h2(cfg: Mapping[str, Any], *, campaign_root: Any,
           self_comparison: bool = False) -> Dict[str, Any]:
    table = load_table(Path(campaign_root) / METRICS_TABLE_REL)
    _require(bool(table.get("entries")), "V543_STATS_TABLE_EMPTY",
             "the metrics table is empty — run `extract` first")
    report = build_h2_report(table=table, self_comparison=self_comparison)
    report["self_comparison"] = bool(self_comparison)
    report["scientific_claim"] = False
    report["config_consistency"] = config_consistency(cfg)
    path = _write_report(campaign_root, H2_REPORT_REL, report)
    report["report_path"] = str(path)
    return report


# ===========================================================================
# 5. screen — §8.2 screen score + tie-break, T5 seed43 top-ups only
# ===========================================================================
def _rel_improve(candidate: float, baseline: float, *, metric: str,
                 higher_is_better: bool) -> float:
    """Relative improvement (positive = better) of one metric.

    ``(candidate − baseline)/baseline`` for a higher-is-better metric,
    ``(baseline − candidate)/baseline`` for a lower-is-better one — the §9.1
    :876–891 direction of the leaf decides, never the caller's convenience.
    """
    _require(_finite(candidate) and _finite(baseline),
             "V543_STATS_METRIC_MISSING",
             f"{metric}: candidate={candidate!r} baseline={baseline!r}")
    _require(abs(float(baseline)) > ZERO_BASELINE_EPS,
             "V543_STATS_REFERENCE_ZERO",
             f"{metric}: the baseline is {baseline!r}; a relative screen term "
             "is undefined at a zero denominator (§9.3 :960 only prescribes "
             "the H2/H3 fallback, so the screen refuses instead of inventing "
             "one)")
    if higher_is_better:
        return (float(candidate) - float(baseline)) / float(baseline)
    return (float(baseline) - float(candidate)) / float(baseline)


def screen_terms(*, candidate: Mapping[str, Any], baseline: Mapping[str, Any],
                 track: str) -> Dict[str, Any]:
    """The five §8.2 quantities of one (candidate, baseline) pair."""
    cand = dict(candidate or {})
    base = dict(baseline or {})

    def _m(table: Mapping[str, Any], key: str) -> float:
        value = table.get(key)
        _require(_finite(value), "V543_STATS_METRIC_MISSING",
                 f"screen needs metric {key!r} (got {value!r})")
        return float(value)

    rmse_rel = _rel_improve(_m(cand, "move.rmse_zero"),
                            _m(base, "move.rmse_zero"),
                            metric="move.rmse_zero",
                            higher_is_better=False)
    direction_rel = _rel_improve(
        _m(cand, "move.fixed_coverage.direction_at_3"),
        _m(base, "move.fixed_coverage.direction_at_3"),
        metric="move.fixed_coverage.direction_at_3",
        higher_is_better=True)
    auprc_delta = (_m(cand, "process.endpoint.auprc")
                   - _m(base, "process.endpoint.auprc"))
    q_rel = _rel_improve(_m(cand, "process.q.mae_curve_median"),
                         _m(base, "process.q.mae_curve_median"),
                         metric="process.q.mae_curve_median",
                         higher_is_better=False)
    ordering_delta = (_m(cand, "process.ordering.pairwise_accuracy_full")
                      - _m(base, "process.ordering.pairwise_accuracy_full"))
    terms = {
        "move_rmse_rel_improve": rmse_rel,
        "move_fixed_coverage_direction_rel": direction_rel,
        "process_auprc_delta": auprc_delta,
        "process_q_mae_rel_improve": q_rel,
        "process_ordering_abs_improve": ordering_delta,
    }
    normalizers = {
        "move_rmse_rel_improve": SCREEN_NORMALIZERS["move_rmse_rel_improve"],
        "move_fixed_coverage_direction_rel":
            SCREEN_NORMALIZERS["move_fixed_coverage_direction_rel"],
        "process_auprc_delta": SCREEN_NORMALIZERS["process_auprc_delta"],
        "process_q_mae_rel_improve":
            SCREEN_NORMALIZERS["process_q_mae_rel_improve"],
        "process_ordering_abs_improve":
            SCREEN_NORMALIZERS["process_ordering_abs_improve"],
    }
    used = (("move_rmse_rel_improve", "move_fixed_coverage_direction_rel")
            if track == "move" else
            ("process_auprc_delta", "process_q_mae_rel_improve",
             "process_ordering_abs_improve"))
    scored = {name: terms[name] / normalizers[name] for name in used}
    return {
        "track": track,
        "terms": terms,
        "normalizers": normalizers,
        "scored_terms": scored,
        "score": float(max(scored.values())),
        "formula": (
            "MOVE: max(rmse_rel_improve/0.03, fixed_coverage_direction_rel/"
            "0.03)" if track == "move" else
            "PROCESS: max(delta_auprc/0.02, q_mae_rel_improve/0.05, "
            "ordering_abs_improve/0.02)"),
        "plan_line": "§8.2 :819-827",
    }


def build_screen_report(*, table: Mapping[str, Any],
                        self_comparison: bool = False,
                        diagnostic_h3_forbidden: bool = True
                        ) -> Dict[str, Any]:
    """Rank each track's T3/T4 candidates and name the ≤2 seed43 top-ups."""
    candidate_pool: List[str] = []
    for entry in table.get("entries") or []:
        role = str(entry.get("role"))
        if role == "candidate" or (self_comparison and role == "baseline"):
            candidate_pool.append(str(entry.get("candidate_id")))
    tracks_out: Dict[str, Any] = {}
    for track in TRACKS:
        screened: List[Dict[str, Any]] = []
        for candidate_id in sorted(set(candidate_pool)):
            seed_rows = {s: r for s, r in rows_by_seed(table, candidate_id).items()
                         if s in SCREEN_SEEDS}
            if not seed_rows:
                continue
            row_track = str((seed_rows[sorted(seed_rows)[0]]).get("track"))
            if row_track not in (track, "both"):
                continue
            per_seed_scores: Dict[str, Any] = {}
            per_seed_h2: Dict[str, bool] = {}
            seed_metric_tables: Dict[int, Dict[str, Any]] = {}
            reasons: List[str] = []
            for seed in SCREEN_SEEDS:
                row = seed_rows.get(seed)
                if row is None:
                    reasons.append(f"T3T4_incomplete:S{seed}_missing")
                    continue
                base = baseline_row(table, seed)
                seed_metric_tables[seed] = {
                    "candidate": dict(row.get("metrics") or {}),
                    "baseline": dict(base.get("metrics") or {}),
                }
                h2 = h2_seed_entry(row, base)
                per_seed_h2[str(seed)] = bool(h2["h2_no_breach"])
                if not h2["h2_no_breach"]:
                    reasons.append(f"H2_breach:S{seed}:" +
                                   ",".join(h2["breached_guards"]))
                if row_track == "both":
                    continue
                per_seed_scores[str(seed)] = screen_terms(
                    candidate=row.get("metrics") or {},
                    baseline=base.get("metrics") or {}, track=track)
            if len(seed_metric_tables) != len(SCREEN_SEEDS):
                screened.append({
                    "candidate_id": candidate_id,
                    "eligible": False,
                    "ineligible_reasons": reasons,
                    "per_seed_h2_no_breach": per_seed_h2,
                    "per_seed_scores": per_seed_scores,
                })
                continue
            # §8.2 fixes the score formula but not the two-seed aggregation;
            # this lane seed-averages the point metrics first (mirroring the
            # §9.4 :1023 "seed-average first" ordering) and reports the
            # per-seed scores beside it.
            averaged = {
                side: {key: float(np.mean([tables[side][key]
                                           for tables in
                                           seed_metric_tables.values()]))
                       for key in (seed_metric_tables[SCREEN_SEEDS[0]][side])}
                for side in ("candidate", "baseline")
            }
            score = screen_terms(candidate=averaged["candidate"],
                                 baseline=averaged["baseline"], track=track)
            run_info = dict((seed_rows[SCREEN_SEEDS[0]]).get("run") or {})
            eligible = not reasons
            screened.append({
                "candidate_id": candidate_id,
                "track": track,
                "eligible": bool(eligible),
                "ineligible_reasons": reasons,
                "per_seed_h2_no_breach": per_seed_h2,
                "per_seed_scores": per_seed_scores,
                "seed_averaged_point_metrics": averaged,
                "screen_score": score,
                "inner_val_total_loss": run_info.get("inner_val_total_loss"),
                "core_time_s": run_info.get("core_time_s"),
                "n_trainable_params": run_info.get("n_trainable_params"),
                "t3_run_id": str((seed_rows.get(SCREEN_SEEDS[0]) or {}).get(
                    "run_id")),
                "t4_run_id": str((seed_rows.get(SCREEN_SEEDS[1]) or {}).get(
                    "run_id")),
                "identity_sha256_by_seed": {
                    str(s): (seed_rows.get(s) or {}).get("identity_sha256")
                    for s in SCREEN_SEEDS},
                "baseline_identity_sha256_by_seed": {
                    str(s): baseline_row(table, s).get("identity_sha256")
                    for s in SCREEN_SEEDS},
            })
        ranked = _rank_top2(screened)
        screened_by_id = {str(c["candidate_id"]): c for c in screened}
        for rank, cid in enumerate(ranked["ranked_eligible"], start=1):
            screened_by_id[cid]["rank"] = rank
        top2 = ranked["top2"]
        tracks_out[track] = {
            "track": track,
            "baseline_retained_id": BASELINE_ID_BY_TRACK[track],
            "screened": screened,
            "ranked_eligible": ranked["ranked_eligible"],
            "top2": top2,
            "top2_count": len(top2),
            "top2_max_candidates": 2,
            "no_substitutes": True,
            "not_selected": ranked["not_selected"],
            "ties_broken": ranked["ties_broken"],
            "t5_run_ids": [f"V543_{cid}_S{SEED43}" for cid in top2
                           if cid in FROZEN_CANDIDATE_IDS],
            "t5_run_ids_not_formal": {
                cid: f"V543_{cid}_S{SEED43}" for cid in top2
                if cid not in FROZEN_CANDIDATE_IDS},
            "tie_break": ["inner_val_point_loss", "core_time_s",
                          "n_trainable_params_ascending",
                          "decoder_id_lexicographic"],
        }
    total_t5 = sum(len(t["t5_run_ids"]) for t in tracks_out.values())
    return {
        "schema": SCREEN_REPORT_SCHEMA,
        "kind": "screen",
        "gate": "screen (seed43 top-up selection)",
        "plan_lines": "§8.2 :811-829 / 附录 C :1629-1632",
        "rule": ("§8.2 :813-817 — every formal candidate first completes "
                 "seed17 and seed29; each track then promotes AT MOST 2 "
                 "candidates by the pre-registered screen score for the seed43 "
                 "(T5) top-up. Two-seed numbers rank candidates, they never "
                 "declare a scientific champion (:815)."),
        "screen_score_usage": {
            "used_for": "seed43_top_up_selection_only",
            "forbidden_for": "H3_selection (§8.2 :815/:817)",
            "diagnostic_h3_selection_forbidden": bool(diagnostic_h3_forbidden),
            "config_key": "screening.diagnostic_h3_selection_forbidden",
        },
        "formulas": {
            "move": "max(rmse_rel_improve/0.03, "
                    "fixed_coverage_direction_rel/0.03)",
            "process": "max(delta_auprc/0.02, q_mae_rel_improve/0.05, "
                       "ordering_abs_improve/0.02)",
            "plan_line": "§8.2 :822-826",
        },
        "seed_aggregation": {
            "used": "seed_average_of_point_metrics(T3,T4)",
            "also_reported": "per_seed_scores",
            "note": ("§8.2 fixes the score formula but NOT how the two seeds "
                     "are combined; this lane seed-averages the point metrics "
                     "before applying the formula (mirroring §9.4 :1023's "
                     "'seed-average first') and reports the per-seed scores "
                     "beside it — recorded as an ambiguity"),
        },
        "eligibility_rule": ("both T3(seed17) and T4(seed29) records present; "
                             "metrics finite; no per-seed H2 breach (§9.3 :958 "
                             "— a breached candidate can never be promoted)"),
        "tracks": tracks_out,
        "t5_activation": {
            "max_runs": 4,
            "emitted": total_t5,
            "rule": ("附录 C :1634 — the T5 rows activate only for as many "
                     "qualified candidates as the track actually has"),
        },
        "self_comparison": bool(self_comparison),
        "scientific_claim": False,
        "denominator": {
            "kind": "V543_recomputed_M0P0", "plan_line": "§9.5 :1037",
            "v542_archive_used": False,
        },
        "written_at": _now(),
    }


def _rank_top2(screened: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Rank by screen score then the §8.2 :829 tie-break; take ≤2."""
    eligible = [dict(c) for c in screened if c.get("eligible")]

    def _key(c: Mapping[str, Any]) -> Any:
        score = float(((c.get("screen_score") or {}).get("score"))
                      if _finite((c.get("screen_score") or {}).get("score"))
                      else float("-inf"))
        loss = c.get("inner_val_total_loss")
        core = c.get("core_time_s")
        params = c.get("n_trainable_params")
        return (-score,
                float(loss) if _finite(loss) else float("inf"),
                float(core) if _finite(core) else float("inf"),
                int(params) if _finite(params) else 2 ** 31,
                str(c.get("candidate_id")))

    ranked = sorted(eligible, key=_key)
    # ties = adjacent pairs sharing the whole key except the id
    ties: List[Dict[str, Any]] = []
    for left, right in zip(ranked, ranked[1:]):
        if _key(left)[:4] == _key(right)[:4]:
            ties.append({"candidates": [left["candidate_id"],
                                        right["candidate_id"]],
                         "resolved_by": "decoder_id_lexicographic"})
    return {
        "ranked_eligible": [str(c["candidate_id"]) for c in ranked],
        "top2": [str(c["candidate_id"]) for c in ranked[:2]],
        "not_selected": [str(c["candidate_id"]) for c in ranked[2:]],
        "ties_broken": ties,
    }


def run_screen(cfg: Mapping[str, Any], *, campaign_root: Any,
               self_comparison: bool = False) -> Dict[str, Any]:
    table = load_table(Path(campaign_root) / METRICS_TABLE_REL)
    _require(bool(table.get("entries")), "V543_STATS_TABLE_EMPTY",
             "the metrics table is empty — run `extract` first")
    report = build_screen_report(
        table=table, self_comparison=self_comparison,
        diagnostic_h3_forbidden=bool(
            (dict(cfg.get("screening") or {})).get(
                "diagnostic_h3_selection_forbidden", True)))
    report["config_consistency"] = config_consistency(cfg)
    report["source_config"] = {"screening": dict(cfg.get("screening") or {})}
    path = _write_report(campaign_root, SCREEN_REPORT_REL, report)
    report["report_path"] = str(path)
    return report


# ===========================================================================
# 6. h3 — bracket logic + significance (§9.3 :962–1012, §9.4 :1014–1033)
# ===========================================================================
def holm_adjust(p_values: Sequence[float], alpha: float) -> Dict[str, Any]:
    """Holm step-down over ONE frozen family (§9.3 :1001–1003).

    Adjusted p = running max of ``(m − rank) · p`` over the ascending order
    (capped at 1); rejection is the step-down rule at ``alpha``.
    """
    values = [float(v) for v in p_values]
    _require(bool(values), "V543_STATS_HOLM_EMPTY",
             "a Holm family must carry at least one p-value")
    _require(all(0.0 <= v <= 1.0 for v in values),
             "V543_STATS_HOLM_P_RANGE",
             f"p-values must be in [0, 1] (got {values!r})")
    m = len(values)
    order = sorted(range(m), key=lambda i: (values[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * values[i])
        adjusted[i] = min(1.0, running)
    rejected = [False] * m
    stopped = False
    for i in order:
        if stopped:
            break
        if adjusted[i] < alpha:
            rejected[i] = True
        else:
            stopped = True
    return {"alpha": float(alpha), "family_size": m,
            "adjusted": adjusted, "rejected": rejected,
            "order": list(order),
            "rule": ("§9.3 :1001-1003 — Holm step-down; MOVE's two exits are "
                     "two families of their own at α=0.025 each, PROCESS is "
                     "one family of (2 endpoints × eligible candidates) at "
                     "α=0.025")}


def require_curve_level(metric_id: str) -> None:
    """§9.4 :1025 — a frame-level metric may never enter the bootstrap."""
    _require(metric_id not in FRAME_LEVEL_METRICS,
             "V543_STATS_FRAME_LEVEL_BOOTSTRAP",
             f"{metric_id!r} is a frame-level quantity "
             f"({FRAME_LEVEL_METRICS.get(metric_id)}) and §9.4 :1025 forbids "
             "a frame-level bootstrap; only curve (reaction cluster) values "
             "may be resampled")
    _require(metric_id in CURVE_LEVEL_HIGHER_IS_BETTER,
             "V543_STATS_METRIC_UNKNOWN",
             f"{metric_id!r} is not a frozen curve-level bootstrap metric "
             f"({sorted(CURVE_LEVEL_HIGHER_IS_BETTER)})")


def _bootstrap_call(function: str, *,
                    candidate_tables: Sequence[Mapping[str, float]],
                    baseline_tables: Sequence[Mapping[str, float]],
                    metric_id: str, higher_is_better: bool,
                    draws: int, seed_root: int,
                    **kwargs: Any) -> Dict[str, Any]:
    """Call one of the frozen evaluator's two §9.4 entry points.

    The statistic is never re-implemented here: both
    ``paired_curve_bootstrap_multiseed`` (confirmatory, per-seed cluster sets)
    and ``paired_curve_bootstrap`` (audit, all-seed shared intersection) live in
    ``pes2ts_core.pesn.offline.evaluation.v543_eval`` and are consumed as-is.
    """
    require_curve_level(metric_id)
    evaluator = _eval_module()
    entry = getattr(evaluator, function, None)
    _require(entry is not None, "V543_STATS_EVALUATOR_API_MISSING",
             f"the frozen evaluator exposes no {function!r}; the statistics "
             "lane refuses rather than substituting its own implementation")
    try:
        out = entry(list(candidate_tables), list(baseline_tables),
                    higher_is_better=bool(higher_is_better), metric=metric_id,
                    draws=int(draws), seed_root=int(seed_root), **kwargs)
    except Exception as exc:  # noqa: BLE001 - typed pass-through
        code = getattr(exc, "code", "V543_STATS_BOOTSTRAP_FAILED")
        detail = getattr(exc, "detail", str(exc))
        raise V543StatsError(str(code), f"{metric_id}: {detail}") from exc
    out["delegated_to"] = (
        f"pes2ts_core.pesn.offline.evaluation.v543_eval.{function} "
        "(§9.4 :1014-1033)")
    out["higher_is_better"] = bool(higher_is_better)
    return out


def paired_bootstrap_multiseed(
        *, candidate_tables: Sequence[Mapping[str, float]],
        baseline_tables: Sequence[Mapping[str, float]],
        metric_id: str, higher_is_better: Optional[bool] = None,
        draws: int = BOOTSTRAP_DRAWS,
        seed_root: int = BOOTSTRAP_SEED_ROOT) -> Dict[str, Any]:
    """CONFIRMATORY §9.4 bootstrap — per-seed cluster sets, seed-average per draw.

    the candidate and baseline tables are given in the producer's ``seed_order``
    (``bootstrap_inputs.seed_order``, the campaign seed order [17, 29, 43]);
    ``seed_index`` inside the frozen function is positional, so the order is
    part of the statistic.
    """
    require_curve_level(metric_id)
    higher = (bool(CURVE_LEVEL_HIGHER_IS_BETTER[metric_id])
              if higher_is_better is None else bool(higher_is_better))
    out = _bootstrap_call(
        "paired_curve_bootstrap_multiseed",
        candidate_tables=candidate_tables, baseline_tables=baseline_tables,
        metric_id=metric_id, higher_is_better=higher, draws=draws,
        seed_root=seed_root, keep_draws=True)
    draws_values = out.pop("draw_values", None)
    out["draw_values_sha256"] = (
        None if draws_values is None
        else hashlib.sha256(
            np.asarray(draws_values, dtype=np.float64).astype("<f8").tobytes()
        ).hexdigest())
    out["draw_values_retained"] = False
    out["draw_values_note"] = (
        "the per-draw values are reproducible from draws + seed_root + "
        "seed_derivation; only their digest is kept in the report")
    out["mode_role"] = "CONFIRMATORY (§9.3 :995-1003 / §9.4 :1023)"
    return out


def paired_bootstrap(*, candidate_tables: Sequence[Mapping[str, float]],
                     baseline_tables: Sequence[Mapping[str, float]],
                     metric_id: str, higher_is_better: Optional[bool] = None,
                     draws: int = BOOTSTRAP_DRAWS,
                     seed_root: int = BOOTSTRAP_SEED_ROOT) -> Dict[str, Any]:
    """AUDIT-ONLY §9.4 bootstrap — all-seed shared intersection.

    Kept so an independent reader can reproduce the older reading and see how
    far the intersection collapses; it is never a confirmatory input on this
    campaign (see :func:`bootstrap_contract` and the baseline's
    ``bootstrap_inputs.modes``).
    """
    require_curve_level(metric_id)
    higher = (bool(CURVE_LEVEL_HIGHER_IS_BETTER[metric_id])
              if higher_is_better is None else bool(higher_is_better))
    out = _bootstrap_call(
        "paired_curve_bootstrap",
        candidate_tables=candidate_tables, baseline_tables=baseline_tables,
        metric_id=metric_id, higher_is_better=higher, draws=draws,
        seed_root=seed_root)
    out["mode_role"] = "AUDIT_ONLY (shared all-seed intersection)"
    return out


def point_effects(*, candidate: Mapping[str, Any],
                  baseline: Mapping[str, Any]) -> Dict[str, Any]:
    """Seed-averaged point effects of the four §9.3 endpoints."""
    out: Dict[str, Any] = {}
    for spec in H3_ENDPOINTS:
        metric = str(spec["metric"])
        cand = _require_finite(candidate.get(metric), metric)
        base = _require_finite(baseline.get(metric), metric)
        if bool(spec["higher_is_better"]):
            absolute = cand - base
            relative = (None if abs(base) <= ZERO_BASELINE_EPS
                        else absolute / base)
        else:
            absolute = base - cand
            relative = (None if abs(base) <= ZERO_BASELINE_EPS
                        else absolute / base)
        out[str(spec["endpoint"])] = {
            "metric": metric,
            "candidate": cand, "baseline": base,
            "absolute_improvement": float(absolute),
            "relative_improvement": (None if relative is None
                                     else float(relative)),
            "higher_is_better": bool(spec["higher_is_better"]),
            "plan_line": spec["plan_line"],
        }
    return out


def _require_finite(value: Any, label: str) -> float:
    _require(_finite(value), "V543_STATS_METRIC_MISSING",
             f"metric {label} is missing/non-finite (got {value!r})")
    return float(value)


def move_route(effects: Mapping[str, Any]) -> Dict[str, Any]:
    """§9.3 :966–976 MOVE bracket (each branch closes its own bracket)."""
    rmse = effects["move_rmse"]
    direction = effects["move_direction"]
    rmse_rel = rmse["relative_improvement"]
    direction_rel = direction["relative_improvement"]
    _require(rmse_rel is not None and direction_rel is not None,
             "V543_STATS_REFERENCE_ZERO",
             "the MOVE relative effects need a non-zero M0 baseline (§9.3 :960)")
    rmse_pass = _limit_ok(MOVE_H3_RMSE_REL_MIN, float(rmse_rel))
    direction_pass = (_limit_ok(MOVE_H3_DIRECTION_REL_MIN,
                                float(direction_rel))
                      and _limit_ok(MOVE_H3_DIRECTION_ABS_MIN,
                                    float(direction["absolute_improvement"])))
    rmse_non_inferior = -float(rmse_rel) <= (
        MOVE_RMSE_REGRESSION_REL_MAX + LIMIT_TOLERANCE)
    direction_non_inferior = -float(direction_rel) <= (
        MOVE_FIXED_COVERAGE_REGRESSION_REL_MAX + LIMIT_TOLERANCE)
    return {
        "pass_logic": ("(rmse_pass AND direction_non_inferior) OR "
                       "(direction_pass AND rmse_non_inferior)"),
        "components": {
            "rmse_pass": bool(rmse_pass),
            "direction_pass": bool(direction_pass),
            "rmse_non_inferior": bool(rmse_non_inferior),
            "direction_non_inferior": bool(direction_non_inferior),
        },
        "thresholds": {
            "move_h3_rmse_rel_min": MOVE_H3_RMSE_REL_MIN,
            "move_h3_direction_rel_min": MOVE_H3_DIRECTION_REL_MIN,
            "move_h3_direction_abs_min": MOVE_H3_DIRECTION_ABS_MIN,
            "move_rmse_regression_rel_max": MOVE_RMSE_REGRESSION_REL_MAX,
            "move_fixed_coverage_regression_rel_max":
                MOVE_FIXED_COVERAGE_REGRESSION_REL_MAX,
        },
        "plan_line": "§9.3 :966-976 / 附录 B :1593-1595",
    }


def process_route(effects: Mapping[str, Any],
                  metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """§9.3 :981–992 PROCESS bracket (five non-inferiority conditions)."""
    cand = dict(metrics.get("candidate") or {})
    base = dict(metrics.get("baseline") or {})
    auprc_delta = effects["process_auprc"]["absolute_improvement"]
    q_rel = effects["process_q"]["relative_improvement"]
    _require(q_rel is not None, "V543_STATS_REFERENCE_ZERO",
             "the PROCESS q relative effect needs a non-zero P0 q MAE")
    recall_drop = (float(_require_finite(base.get(
        "process.endpoint.recall_at_fpr_0p05"),
        "process.endpoint.recall_at_fpr_0p05"))
        - float(_require_finite(cand.get("process.endpoint.recall_at_fpr_0p05"),
                                "process.endpoint.recall_at_fpr_0p05")))
    ece_increase = (float(_require_finite(cand.get("process.endpoint.ece"),
                                          "process.endpoint.ece"))
                    - float(_require_finite(base.get("process.endpoint.ece"),
                                            "process.endpoint.ece")))
    ordering_drop = (float(_require_finite(
        base.get("process.ordering.pairwise_accuracy_full"),
        "process.ordering.pairwise_accuracy_full"))
        - float(_require_finite(
            cand.get("process.ordering.pairwise_accuracy_full"),
            "process.ordering.pairwise_accuracy_full")))
    auprc_pass = _limit_ok(PROCESS_H3_AUPRC_ABS_MIN, float(auprc_delta))
    q_pass = (_limit_ok(PROCESS_H3_Q_MAE_REL_MIN, float(q_rel))
              and _limit_ok(ordering_drop, PROCESS_ORDERING_DROP_ABS_MAX))
    auprc_non_inferior = (-float(auprc_delta)
                          <= PROCESS_AUPRC_DROP_ABS_MAX + LIMIT_TOLERANCE)
    q_non_inferior = (-float(q_rel)
                      <= PROCESS_Q_MAE_REGRESSION_REL_MAX + LIMIT_TOLERANCE)
    recall_non_inferior = (recall_drop
                           <= PROCESS_RECALL_DROP_ABS_MAX + LIMIT_TOLERANCE)
    calibration_non_inferior = (ece_increase
                                <= PROCESS_ECE_INCREASE_ABS_MAX
                                + LIMIT_TOLERANCE)
    ordering_non_inferior = (ordering_drop
                             <= PROCESS_ORDERING_DROP_ABS_MAX
                             + LIMIT_TOLERANCE)
    return {
        "pass_logic": ("(auprc_pass AND q_non_inferior AND recall_non_inferior "
                       "AND calibration_non_inferior AND ordering_non_inferior) "
                       "OR (q_pass AND auprc_non_inferior AND "
                       "recall_non_inferior AND calibration_non_inferior)"),
        "components": {
            "auprc_pass": bool(auprc_pass),
            "q_pass": bool(q_pass),
            "auprc_non_inferior": bool(auprc_non_inferior),
            "q_non_inferior": bool(q_non_inferior),
            "recall_non_inferior": bool(recall_non_inferior),
            "calibration_non_inferior": bool(calibration_non_inferior),
            "ordering_non_inferior": bool(ordering_non_inferior),
        },
        "observed": {
            "auprc_delta": float(auprc_delta),
            "q_mae_relative_improvement": float(q_rel),
            "recall_at_fpr_drop": float(recall_drop),
            "ece_increase": float(ece_increase),
            "ordering_drop": float(ordering_drop),
        },
        "thresholds": {
            "process_h3_auprc_abs_min": PROCESS_H3_AUPRC_ABS_MIN,
            "process_h3_q_mae_rel_min": PROCESS_H3_Q_MAE_REL_MIN,
            "process_auprc_drop_abs_max": PROCESS_AUPRC_DROP_ABS_MAX,
            "process_q_mae_regression_rel_max":
                PROCESS_Q_MAE_REGRESSION_REL_MAX,
            "process_recall_at_fpr_drop_abs_max":
                PROCESS_RECALL_DROP_ABS_MAX,
            "process_ece_increase_abs_max": PROCESS_ECE_INCREASE_ABS_MAX,
            "process_ordering_drop_abs_max": PROCESS_ORDERING_DROP_ABS_MAX,
            "process_fpr_constraint": PROCESS_FPR_CONSTRAINT,
        },
        "plan_line": "§9.3 :981-992 / 附录 B :1596-1603",
    }


def effect_margin(endpoint: str, effects: Mapping[str, Any]) -> Optional[float]:
    """§9.3 :1005–1012 — the frozen effect margin (tie-break only)."""
    entry = effects.get(endpoint)
    _require(entry is not None, "V543_STATS_ENDPOINT_UNKNOWN",
             f"effect margin for unknown endpoint {endpoint!r}")
    rel = entry.get("relative_improvement")
    absolute = entry.get("absolute_improvement")
    if endpoint == "move_rmse":
        if rel is None:
            return None
        return float(rel / MOVE_H3_RMSE_REL_MIN)
    if endpoint == "move_direction":
        if rel is None:
            return None
        return float(min(rel / MOVE_H3_DIRECTION_REL_MIN,
                         absolute / MOVE_H3_DIRECTION_ABS_MIN))
    if endpoint == "process_auprc":
        return float(absolute / PROCESS_H3_AUPRC_ABS_MIN)
    if endpoint == "process_q":
        if rel is None:
            return None
        return float(rel / PROCESS_H3_Q_MAE_REL_MIN)
    raise V543StatsError("V543_STATS_ENDPOINT_UNKNOWN",
                         f"no frozen margin definition for {endpoint!r}")


def _seed_mean_metrics(seed_rows: Mapping[int, Mapping[str, Any]]
                       ) -> Dict[str, float]:
    """Three-seed mean of every metric (the bracket's point estimate)."""
    ids = sorted(seed_rows)
    _require(len(ids) == H3_TOTAL_SEEDS, "V543_STATS_H3_SEEDS_INCOMPLETE",
             f"the bracket point estimate needs all {H3_TOTAL_SEEDS} seeds "
             f"(got {ids})")
    keys = set()
    for seed in ids:
        keys |= set((seed_rows[seed].get("metrics") or {}))
    out: Dict[str, float] = {}
    for key in sorted(keys):
        values = [(seed_rows[seed].get("metrics") or {}).get(key)
                  for seed in ids]
        if all(_finite(v) for v in values):
            out[key] = float(np.mean([float(v) for v in values]))
    return out


def compute_h3(*, table: Mapping[str, Any],
               draws: Optional[int] = None,
               seed_root: Optional[int] = None,
               self_comparison: bool = False,
               only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The H3 report: brackets, both bootstrap modes, Holm, effect margins.

    Wiring (per the §9.4 producer contract in ``bootstrap_inputs``):

    * CONFIRMATORY — ``paired_curve_bootstrap_multiseed``: every draw resamples
      each seed's OWN inner-val cluster set (candidate and baseline paired
      inside a seed), then the per-seed statistics are seed-averaged per draw
      (§9.3 :995–1003 / §9.4 :1023).  ``seed_order`` comes from
      ``bootstrap_inputs.seed_order``; ``draws``/``seed_root``/the per-metric
      ``higher_is_better`` flag are read from the same contract.
    * AUDIT — ``paired_curve_bootstrap`` (all-seed shared intersection) is run
      alongside and reported with ``audit_only: true`` +
      ``n_curves_shared`` so the historical collapse stays on record.  It never
      enters a verdict.
    """
    contract = table_bootstrap_contract(table)
    seed_order = [int(s) for s in (contract.get("seed_order") or FROZEN_SEEDS)]
    _require(sorted(seed_order) == sorted(FROZEN_SEEDS),
             "V543_STATS_SEED_ORDER",
             f"the bootstrap contract's seed_order {seed_order} is not the "
             f"campaign seed set {list(FROZEN_SEEDS)} (§8.1 :798-809)")
    n_draws = (int(contract.get("draws") or BOOTSTRAP_DRAWS)
               if draws is None else int(draws))
    root = (int(contract.get("seed_root") or BOOTSTRAP_SEED_ROOT)
            if seed_root is None else int(seed_root))
    declared_higher = dict(contract.get("metrics_higher_is_better") or {})
    baseline_identity_block = dict(table.get("baseline_identity") or {})
    pool = candidate_ids(table, include_baseline=self_comparison)
    if only:
        missing = sorted(set(only) - set(pool))
        _require(not missing, "V543_STATS_CANDIDATE_UNKNOWN",
                 f"requested candidates {missing} are not in the metrics "
                 f"table ({pool})")
        pool = list(only)
    excluded: Dict[str, str] = {}
    eligible_ids: List[str] = []
    for candidate_id in pool:
        seed_rows = rows_by_seed(table, candidate_id)
        if not seed_rows:
            excluded[candidate_id] = "no_rows"
            continue
        missing = [s for s in FROZEN_SEEDS if s not in seed_rows]
        if missing:
            excluded[candidate_id] = f"three_seeds_incomplete(missing {missing})"
            continue
        breached: List[int] = []
        for seed in FROZEN_SEEDS:
            h2 = h2_seed_entry(seed_rows[seed], baseline_row(table, seed))
            if not h2["h2_no_breach"]:
                breached.append(seed)
        if breached:
            excluded[candidate_id] = f"per_seed_h2_breach(S{breached})"
            continue
        eligible_ids.append(candidate_id)
    eligible_ids.sort()

    boot: Dict[tuple[str, str], Dict[str, Any]] = {}
    boot_audit: Dict[tuple[str, str], Dict[str, Any]] = {}
    boot_per_seed: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    pairing: Dict[tuple[str, str], Dict[str, Any]] = {}
    effects: Dict[str, Dict[str, Any]] = {}
    routes: Dict[str, Any] = {}
    candidates: Dict[str, Any] = {}
    for candidate_id in eligible_ids:
        seed_rows = rows_by_seed(table, candidate_id)
        cand_metrics = _seed_mean_metrics(seed_rows)
        base_rows = {seed: baseline_row(table, seed) for seed in seed_order}
        base_metrics = _seed_mean_metrics(base_rows)
        effects[candidate_id] = point_effects(candidate=cand_metrics,
                                             baseline=base_metrics)
        move = move_route(effects[candidate_id])
        process = process_route(effects[candidate_id],
                                {"candidate": cand_metrics,
                                 "baseline": base_metrics})
        routes[candidate_id] = {"move": move, "process": process}
        for spec in H3_ENDPOINTS:
            endpoint = str(spec["endpoint"])
            metric = str(spec["metric"])
            require_curve_level(metric)
            # the producer's bootstrap_inputs flag wins; the frozen endpoint
            # spec is the cross-check (a contradiction is a typed refusal).
            declared_flag = declared_higher.get(metric)
            higher = bool(spec["higher_is_better"])
            _require(declared_flag is None or bool(declared_flag) == higher,
                     "V543_STATS_BOOTSTRAP_DIRECTION",
                     f"{candidate_id}/{metric}: bootstrap_inputs declares "
                     f"higher_is_better={declared_flag!r} but the frozen "
                     f"endpoint spec says {higher!r} (§9.1 :876-891)")
            cand_tables: List[Mapping[str, float]] = []
            base_tables: List[Mapping[str, float]] = []
            per_seed_pairing: Dict[str, Any] = {}
            for seed in seed_order:
                cand_entry = seed_rows[seed]
                base_entry = base_rows[seed]
                cand_table = dict((cand_entry.get("curve_values") or {}).get(
                    metric) or {})
                base_table = dict((base_entry.get("curve_values") or {}).get(
                    metric) or {})
                _require(bool(cand_table) and bool(base_table),
                         "V543_STATS_BOOTSTRAP_EMPTY",
                         f"{candidate_id}/S{seed}/{metric}: no per-curve "
                         "values — the §9.4 bootstrap needs curve tables")
                entry_flag = ((cand_entry.get(
                    "curve_values_higher_is_better") or {}).get(metric))
                _require(entry_flag is None or bool(entry_flag) == higher,
                         "V543_STATS_BOOTSTRAP_DIRECTION",
                         f"{candidate_id}/S{seed}/{metric}: the entry declares "
                         f"higher_is_better={entry_flag!r} but the contract / "
                         f"frozen spec says {higher!r}")
                shared = set(cand_table) & set(base_table)
                per_seed_pairing[str(seed)] = {
                    "n_candidate_curves": len(cand_table),
                    "n_baseline_curves": len(base_table),
                    "n_shared_curves": len(shared),
                    "n_candidate_only": len(set(cand_table) - shared),
                    "n_baseline_only": len(set(base_table) - shared),
                }
                cand_tables.append(cand_table)
                base_tables.append(base_table)
                # per-seed audit read of the same frozen implementations
                boot_per_seed[(candidate_id, endpoint, seed)] = paired_bootstrap(
                    candidate_tables=[cand_table],
                    baseline_tables=[base_table],
                    metric_id=metric, higher_is_better=higher,
                    draws=n_draws, seed_root=root)
            # ---- CONFIRMATORY: per-seed cluster sets, seed-averaged per draw
            boot[(candidate_id, endpoint)] = paired_bootstrap_multiseed(
                candidate_tables=cand_tables, baseline_tables=base_tables,
                metric_id=metric, higher_is_better=higher,
                draws=n_draws, seed_root=root)
            # ---- AUDIT: all-seed shared intersection (may have no overlap)
            try:
                audit = paired_bootstrap(
                    candidate_tables=cand_tables,
                    baseline_tables=base_tables, metric_id=metric,
                    higher_is_better=higher, draws=n_draws, seed_root=root)
                boot_audit[(candidate_id, endpoint)] = audit
            except V543StatsError as exc:
                boot_audit[(candidate_id, endpoint)] = {
                    "audit_only": True,
                    "status": "unavailable",
                    "refusal_code": exc.code,
                    "detail": exc.detail,
                    "mode_role": "AUDIT_ONLY (shared all-seed intersection)",
                }
            cross_seed = len(set.intersection(
                *(set(t) for t in (*cand_tables, *base_tables))))
            pairing[(candidate_id, endpoint)] = {
                "seed_order": seed_order,
                "per_seed": per_seed_pairing,
                "n_curves_shared_across_all_seeds": int(cross_seed),
                "n_curves_confirmatory": int(
                    boot[(candidate_id, endpoint)]["n_curves_total"]),
                "n_curves_per_seed_confirmatory": list(
                    boot[(candidate_id, endpoint)]["n_curves_per_seed"]),
                "producer_shared_across_all_seeds": (
                    contract.get("n_curves_shared_across_all_seeds") or {}
                ).get(metric),
                "note": (
                    "§9.4 :1019 governs the PAIRING inside a comparison "
                    "(candidate and baseline of one seed share resample "
                    "indices); :1023 seed-averages per draw. Each formal seed "
                    "evaluates its own inner-val sample (§9.5 :1037), so the "
                    "confirmatory bootstrap resamples per seed and the "
                    "all-seed shared intersection is reported as AUDIT only "
                    "(it collapses to tens of curves on this campaign)."),
            }

    # -- Holm families -----------------------------------------------------
    families: Dict[str, Any] = {}
    adjusted: Dict[tuple[str, str], float] = {}
    rejected: Dict[tuple[str, str], bool] = {}
    for exit_name, endpoint in (("move_rmse_exit", "move_rmse"),
                                ("move_direction_exit", "move_direction")):
        if not eligible_ids:
            continue
        raw = [boot[(cid, endpoint)]["p_one_sided"] for cid in eligible_ids]
        result = holm_adjust(raw, MOVE_EXIT_ALPHA_EACH)
        families[exit_name] = {
            "endpoint": endpoint,
            "alpha": MOVE_EXIT_ALPHA_EACH,
            "scope": "eligible_candidate_set_within_exit",
            "p_raw": {cid: raw[i] for i, cid in enumerate(eligible_ids)},
            "p_adjusted": {cid: result["adjusted"][i]
                           for i, cid in enumerate(eligible_ids)},
            "rejected": {cid: result["rejected"][i]
                         for i, cid in enumerate(eligible_ids)},
            "plan_line": "§9.3 :1001 / 附录 B :1604",
        }
        for i, cid in enumerate(eligible_ids):
            adjusted[(cid, endpoint)] = float(result["adjusted"][i])
            rejected[(cid, endpoint)] = bool(result["rejected"][i])
    process_keys: List[tuple[str, str]] = []
    raw_process: List[float] = []
    for cid in eligible_ids:
        for endpoint in ("process_auprc", "process_q"):
            process_keys.append((cid, endpoint))
            raw_process.append(boot[(cid, endpoint)]["p_one_sided"])
    if process_keys:
        result = holm_adjust(raw_process, PROCESS_FAMILY_ALPHA)
        families["process_route"] = {
            "alpha": PROCESS_FAMILY_ALPHA,
            "scope": "2_endpoints_x_eligible_candidates",
            "family": "single_process_family",
            "p_raw": {f"{cid}:{ep}": raw_process[i]
                      for i, (cid, ep) in enumerate(process_keys)},
            "p_adjusted": {f"{cid}:{ep}": result["adjusted"][i]
                           for i, (cid, ep) in enumerate(process_keys)},
            "rejected": {f"{cid}:{ep}": result["rejected"][i]
                         for i, (cid, ep) in enumerate(process_keys)},
            "plan_line": "§9.3 :1002 / 附录 B :1605",
        }
        for i, key in enumerate(process_keys):
            adjusted[key] = float(result["adjusted"][i])
            rejected[key] = bool(result["rejected"][i])

    # -- per-candidate endpoints, brackets, verdicts -----------------------
    for candidate_id in eligible_ids:
        seed_rows = rows_by_seed(table, candidate_id)
        per_seed_improvement: Dict[str, Dict[str, float]] = {}
        for spec in H3_ENDPOINTS:
            endpoint = str(spec["endpoint"])
            metric = str(spec["metric"])
            values: Dict[str, float] = {}
            for seed in FROZEN_SEEDS:
                cand = _require_finite(
                    (seed_rows[seed].get("metrics") or {}).get(metric), metric)
                base = _require_finite(
                    (baseline_row(table, seed).get("metrics") or {}).get(metric),
                    metric)
                values[str(seed)] = ((cand - base) if spec["higher_is_better"]
                                     else (base - cand))
            per_seed_improvement[endpoint] = values
        endpoints: Dict[str, Any] = {}
        for spec in H3_ENDPOINTS:
            endpoint = str(spec["endpoint"])
            metric = str(spec["metric"])
            b = boot[(candidate_id, endpoint)]
            e = effects[candidate_id][endpoint]
            same_dir = sum(1 for v in per_seed_improvement[endpoint].values()
                           if v > 0.0)
            adj = adjusted.get((candidate_id, endpoint))
            alpha = float(spec["alpha"])
            ci_ok = float(b["ci95"][0]) > 0.0
            adj_ok = bool(adj is not None and adj < alpha)
            endpoints[endpoint] = {
                "metric": metric,
                "family": spec["family"],
                "plan_line": spec["plan_line"],
                "higher_is_better": bool(spec["higher_is_better"]),
                "point_effect": {
                    "candidate_seed_mean": e["candidate"],
                    "baseline_seed_mean": e["baseline"],
                    "absolute_improvement": e["absolute_improvement"],
                    "relative_improvement": e["relative_improvement"],
                },
                "per_seed_improvement": per_seed_improvement[endpoint],
                "same_direction_seeds": int(same_dir),
                "same_direction_ok": bool(same_dir
                                          >= H3_MIN_SEEDS_SAME_DIRECTION),
                "ci95": [float(b["ci95"][0]), float(b["ci95"][1])],
                "ci_lower_gt_zero": bool(ci_ok),
                "p_one_sided": float(b["p_one_sided"]),
                "p_adjusted_holm": (None if adj is None else float(adj)),
                "alpha": alpha,
                "holm_rejected": bool(rejected.get((candidate_id, endpoint),
                                                   False)),
                "n_curves": int(b["n_curves"]),
                "n_curves_per_seed": list(b.get("n_curves_per_seed") or []),
                "n_curves_total": int(b.get("n_curves_total")
                                      or b["n_curves"]),
                "n_seeds": int(b["n_seeds"]),
                "draws": int(b["draws"]),
                "seed_root": int(b["seed_root"]),
                "bootstrap_mode": str(b.get("mode") or ""),
                "bootstrap_mode_role": str(b.get("mode_role") or ""),
                "draw_values_sha256": b.get("draw_values_sha256"),
                "draw_values_retained": bool(b.get("draw_values_retained",
                                                   False)),
                "dropped_curves": dict(b.get("dropped_curves") or {}),
                "effect_margin": effect_margin(endpoint, effects[candidate_id]),
                "curve_pairing": pairing[(candidate_id, endpoint)],
                "audit_shared_intersection": {
                    **{k: v for k, v in boot_audit[(candidate_id, endpoint)]
                       .items()
                       if k in ("mode", "mode_role", "delegated_to",
                                "status", "refusal_code", "detail",
                                "n_curves", "n_seeds", "draws", "seed_root",
                                "point_estimate", "ci95", "p_one_sided",
                                "n_draws_nonpositive", "dropped_curves")},
                    "audit_only": True,
                    "status": (boot_audit[(candidate_id, endpoint)].get(
                        "status") or "ok"),
                    "n_curves_shared": pairing[(candidate_id, endpoint)][
                        "n_curves_shared_across_all_seeds"],
                    "used_for_verdict": False,
                    "note": ("AUDIT ONLY — the all-seed shared-intersection "
                             "bootstrap (`paired_curve_bootstrap`). Retained so "
                             "the historical collapse (1,024 → "
                             f"{pairing[(candidate_id, endpoint)]['n_curves_shared_across_all_seeds']} "
                             "curves) stays on record; the CONFIRMATORY result "
                             "is `bootstrap_mode` above "
                             "(`paired_curve_bootstrap_multiseed`)."),
                },
                "per_seed_bootstrap": {
                    str(seed): {
                        "n_curves": int(boot_per_seed[
                            (candidate_id, endpoint, seed)]["n_curves"]),
                        "p_one_sided": float(boot_per_seed[
                            (candidate_id, endpoint, seed)]["p_one_sided"]),
                        "ci95": [float(x) for x in boot_per_seed[
                            (candidate_id, endpoint, seed)]["ci95"]],
                        "point_estimate": float(boot_per_seed[
                            (candidate_id, endpoint, seed)]["point_estimate"]),
                        "audit_only": True,
                        "note": ("AUDIT ONLY — one seed on its own with the "
                                 "shared-intersection function; the "
                                 "confirmatory number is the per-seed-"
                                 "independent multi-seed call above (§9.4 "
                                 ":1023 seed-average inside every draw)"),
                    } for seed in seed_order},
                "confirmed": bool(endpoints_confirmable(
                    spec, same_dir, ci_ok, adj_ok,
                    routes[candidate_id])),
            }
        move_qualified, move_components = _bracket_qualified(
            routes[candidate_id]["move"]["components"],
            endpoints["move_direction"]["confirmed"],
            endpoints["move_rmse"]["confirmed"])
        process_qualified, process_components = _bracket_qualified(
            routes[candidate_id]["process"]["components"],
            endpoints["process_auprc"]["confirmed"],
            endpoints["process_q"]["confirmed"])
        qualifying = [ep for ep in ENDPOINT_IDS
                      if endpoints[ep]["confirmed"]
                      and ((ep.startswith("move_") and move_qualified)
                           or (ep.startswith("process_") and process_qualified))]
        run_info = dict((seed_rows[FROZEN_SEEDS[0]].get("run") or {}))
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "track": str(seed_rows[FROZEN_SEEDS[0]].get("track")),
            "move_route": {**routes[candidate_id]["move"],
                           "confirmed_components": move_components,
                           "qualified": bool(move_qualified)},
            "process_route": {**routes[candidate_id]["process"],
                              "confirmed_components": process_components,
                              "qualified": bool(process_qualified)},
            "move_qualified": bool(move_qualified),
            "process_qualified": bool(process_qualified),
            "h3_qualified": bool(move_qualified or process_qualified),
            "qualifying_confirmatory_endpoints": qualifying,
            "min_holm_adjusted_p_on_qualifying_endpoint": (
                min(adjusted[(candidate_id, ep)] for ep in qualifying)
                if qualifying else None),
            "max_effect_margin_on_qualifying_endpoint": (
                max(endpoints[ep]["effect_margin"] for ep in qualifying)
                if qualifying else None),
            "endpoints": endpoints,
            "per_seed_h2_no_breach": {
                str(s): bool(h2_seed_entry(
                    seed_rows[s], baseline_row(table, s))["h2_no_breach"])
                for s in FROZEN_SEEDS},
            "identity_sha256_by_seed": {
                str(s): seed_rows[s].get("identity_sha256")
                for s in FROZEN_SEEDS},
            "baseline_identity_sha256_by_seed": {
                str(s): baseline_row(table, s).get("identity_sha256")
                for s in FROZEN_SEEDS},
            "run": run_info,
            "core_time_ratio": run_info.get("core_time_ratio"),
            "n_trainable_params": run_info.get("n_trainable_params"),
            "inner_val_total_loss": run_info.get("inner_val_total_loss"),
        }
        if process_qualified and not move_qualified:
            candidates[candidate_id]["process_only_note"] = (
                "PROCESS-only promotion — the report must state explicitly "
                "that MOVE did not improve (§9.3 :958)")

    return {
        "schema": H3_REPORT_SCHEMA,
        "kind": "h3",
        "gate": "H3",
        "plan_lines": "§9.3 :962-1012 / §9.4 :1014-1033",
        "rule": ("§9.3 :995-1003 — ≥2/3 seeds same direction, no per-seed H2 "
                 "breach, 10,000-draw seed-averaged curve-level paired "
                 "bootstrap with CI lower > 0, one-sided p = (1+#{Δ≤0})/10001; "
                 "MOVE's two exits are two Holm families at α=0.025 each, "
                 "PROCESS is one family of (2 endpoints × eligible candidates) "
                 "at α=0.025; unadjusted and adjusted results are both "
                 "reported."),
        "statistics_contract": {
            "bootstrap_unit": "curve (reaction cluster)",
            "pairing": ("within a seed, candidate and baseline share the "
                        "cluster indices (§9.4 :1019); across seeds each seed "
                        "resamples its own inner-val cluster set"),
            "draws": int(n_draws), "seed_root": int(root),
            "seed_order": seed_order,
            "ci": "percentile 95% (2.5/97.5)",
            "seed_aggregation": "per draw seed-average FIRST, then compare",
            "p_rule": "(1 + #{seed-averaged paired improvement <= 0}) / "
                      "(draws + 1)",
            "multiple_comparison": "Holm; family definition per §9.3",
            "frame_level_bootstrap": "forbidden (§9.4 :1025) — enforced by "
                                     "require_curve_level()",
            "confirmatory_mode": "per_seed_independent (seeds resample their "
                                 "OWN cluster sets; seed-average per draw)",
            "audit_mode": "shared_intersection (all-seed intersection; "
                          "reported, never used for a verdict)",
            "contract_source": contract.get("source"),
            "plan_line": "§9.4 :1016-1027 / 附录 B :1608-1609",
            "delegated_implementations": [
                "pes2ts_core.pesn.offline.evaluation.v543_eval."
                "paired_curve_bootstrap_multiseed (CONFIRMATORY)",
                "pes2ts_core.pesn.offline.evaluation.v543_eval."
                "paired_curve_bootstrap (AUDIT)",
            ],
            "producer_modes": dict(contract.get("modes") or {}),
        },
        "alphas": {"move_exits_each": MOVE_EXIT_ALPHA_EACH,
                   "process_family": PROCESS_FAMILY_ALPHA},
        "min_seeds_same_direction": H3_MIN_SEEDS_SAME_DIRECTION,
        "total_seeds": H3_TOTAL_SEEDS,
        "effect_margin_rule": {
            "move_rmse": "relative_improvement / 0.03",
            "move_direction": "min(relative_improvement / 0.03, "
                              "absolute_improvement / 0.01)",
            "process_auprc": "delta_auprc / 0.02",
            "process_q": "relative_q_mae_improvement / 0.05",
            "usage": ("tie-break ONLY — it never replaces significance "
                      "(§9.3 :1005)"),
            "plan_line": "§9.3 :1005-1012",
        },
        "holm_families": families,
        "eligible_candidates": eligible_ids,
        "excluded_candidates": excluded,
        "candidates": candidates,
        "bootstrap_by_endpoint": {
            f"{cid}:{ep}": {k: v for k, v in b.items()
                            if k not in ("_draws",)}
            for (cid, ep), b in sorted(boot.items())},
        "bootstrap_audit_shared_intersection": {
            f"{cid}:{ep}": {**dict(b), "audit_only": True,
                            "used_for_verdict": False}
            for (cid, ep), b in sorted(boot_audit.items())},
        "bootstrap_modes": {
            "confirmatory": {
                "api": "paired_curve_bootstrap_multiseed",
                "role": "CONFIRMATORY — per-seed cluster sets, seed-average "
                        "per draw (§9.3 :995-1003 / §9.4 :1023)",
                "declared_by": "baseline report bootstrap_inputs.modes."
                               "per_seed_independent",
                "used_for_verdict": True,
            },
            "audit": {
                "api": "paired_curve_bootstrap",
                "role": "AUDIT ONLY — all-seed shared intersection; kept so "
                        "the historical collapse stays reproducible",
                "declared_by": "baseline report bootstrap_inputs.modes."
                               "shared_intersection",
                "used_for_verdict": False,
                "n_curves_shared": {
                    f"{cid}:{ep}": pairing[(cid, ep)][
                        "n_curves_shared_across_all_seeds"]
                    for (cid, ep) in sorted(pairing)},
            },
        },
        "curve_pairing_note": {
            "rule": ("§9.4 :1019 governs the PAIRING inside a comparison; "
                     ":1023 seed-averages per draw; :1024 drops missing "
                     "clusters on both sides of that seed's comparison and "
                     "reports the count"),
            "seed_order": seed_order,
            "observed_n_curves_shared_across_all_seeds": {
                f"{cid}:{ep}": pairing[(cid, ep)][
                    "n_curves_shared_across_all_seeds"]
                for (cid, ep) in sorted(pairing)},
            "observed_n_curves_confirmatory": {
                f"{cid}:{ep}": pairing[(cid, ep)]["n_curves_confirmatory"]
                for (cid, ep) in sorted(pairing)},
            "flag": ("the shared-intersection number is an AUDIT figure only. "
                     "The confirmatory bootstrap uses each seed's own cluster "
                     "set, so the earlier 1,024 → 37 collapse no longer "
                     "constrains the H3 test; the evaluator owner added "
                     "paired_curve_bootstrap_multiseed for exactly this reason"),
        },
        "q_validity": {
            "rule": ("§9.4 :1029-1033 — q.corr_curve_median is diagnostic "
                     "only; every candidate's q validity block is recorded in "
                     "the metrics table and echoed here"),
            "by_candidate": {
                cid: {
                    "q_corr_role": (
                        rows_by_seed(table, cid)[FROZEN_SEEDS[0]]
                        .get("q_validity", {}) or {}).get("q_corr_role"),
                    "frac_curves_ge3": (
                        rows_by_seed(table, cid)[FROZEN_SEEDS[0]]
                        .get("q_validity", {}) or {}).get("frac_curves_ge3"),
                } for cid in eligible_ids},
            "q_corr_used_as_conclusion": False,
        },
        "denominator": {
            "kind": "V543_recomputed_M0P0",
            "plan_line": "§9.5 :1037",
            "baseline_identity": baseline_identity_block,
            "v542_archive_used": False,
        },
        "self_comparison": bool(self_comparison),
        "scientific_claim": False,
        "written_at": _now(),
    }


def endpoints_confirmable(spec: Mapping[str, Any], same_dir: int,
                          ci_ok: bool, adj_ok: bool,
                          routes: Mapping[str, Any]) -> bool:
    """A pass-component is confirmatory only with all three statistical legs."""
    pass_component = bool((routes.get(spec["track"]) or {})
                          .get("components", {}).get(
                              "direction_pass" if spec["endpoint"]
                              == "move_direction" else
                              "rmse_pass" if spec["endpoint"] == "move_rmse"
                              else "auprc_pass" if spec["endpoint"]
                              == "process_auprc" else "q_pass", False))
    return bool(pass_component
                and same_dir >= H3_MIN_SEEDS_SAME_DIRECTION
                and ci_ok and adj_ok)


def _bracket_qualified(components: Mapping[str, Any],
                       confirmed_a: bool, confirmed_b: bool
                       ) -> tuple[bool, Dict[str, bool]]:
    """Apply the bracket with only the CONFIRMED pass-side components.

    For MOVE, ``confirmed_a`` = ``direction_pass`` and ``confirmed_b`` =
    ``rmse_pass``; for PROCESS, ``confirmed_a`` = ``auprc_pass`` and
    ``confirmed_b`` = ``q_pass``.  The non-inferiority conditions are
    point-estimate constraints (§9.3 :966–992) and carry no separate test.
    Each branch closes its own parentheses — no dangling OR.
    """
    if "rmse_pass" in components:
        left = confirmed_a and bool(components["rmse_non_inferior"])
        right = confirmed_b and bool(components["direction_non_inferior"])
        ok = bool(left or right)
        return ok, {
            "direction_pass_confirmed": bool(confirmed_a),
            "rmse_pass_confirmed": bool(confirmed_b),
            "branch_direction_pass": bool(left),
            "branch_rmse_pass": bool(right),
        }
    left = (confirmed_a and bool(components["q_non_inferior"])
            and bool(components["recall_non_inferior"])
            and bool(components["calibration_non_inferior"])
            and bool(components["ordering_non_inferior"]))
    right = (confirmed_b and bool(components["auprc_non_inferior"])
             and bool(components["recall_non_inferior"])
             and bool(components["calibration_non_inferior"]))
    return bool(left or right), {
        "auprc_pass_confirmed": bool(confirmed_a),
        "q_pass_confirmed": bool(confirmed_b),
        "branch_auprc_pass": bool(left),
        "branch_q_pass": bool(right),
    }


def run_h3(cfg: Mapping[str, Any], *, campaign_root: Any,
           draws: Optional[int] = None,
           self_comparison: bool = False) -> Dict[str, Any]:
    table = load_table(Path(campaign_root) / METRICS_TABLE_REL)
    _require(bool(table.get("entries")), "V543_STATS_TABLE_EMPTY",
             "the metrics table is empty — run `extract` first")
    report = compute_h3(
        table=table,
        draws=(None if draws is None else int(draws)),
        self_comparison=self_comparison)
    report["config_consistency"] = config_consistency(cfg)
    path = _write_report(campaign_root, H3_REPORT_REL, report)
    report["report_path"] = str(path)
    return report


# ===========================================================================
# 7. select — per-track champion (§9.3 :994–1012, §8.2 :816)
# ===========================================================================
def select_champions(h3_report: Mapping[str, Any]) -> Dict[str, Any]:
    """One champion per track from the H3-qualified set, else retain the base.

    Eligibility: H3-qualified ONLY.  The frozen effect margin (§9.3 :1005) is
    tie-break material; the remaining order (adjusted p, core time, parameter
    count, decoder id) is stated explicitly because §9.3 fixes only the margin.
    The screen score is NEVER consulted (§8.2 :815/:817).
    """
    candidates = dict(h3_report.get("candidates") or {})
    tracks_out: Dict[str, Any] = {}
    for track in TRACKS:
        qualified: List[Dict[str, Any]] = []
        for cid, entry in sorted(candidates.items()):
            route_key = f"{track}_qualified"
            if not entry.get(route_key):
                continue
            qualified.append({
                "candidate_id": cid,
                "track": track,
                "move_qualified": bool(entry.get("move_qualified")),
                "process_qualified": bool(entry.get("process_qualified")),
                "qualifying_confirmatory_endpoints": list(
                    entry.get("qualifying_confirmatory_endpoints") or []),
                "min_holm_adjusted_p_on_qualifying_endpoint": entry.get(
                    "min_holm_adjusted_p_on_qualifying_endpoint"),
                "max_effect_margin_on_qualifying_endpoint": entry.get(
                    "max_effect_margin_on_qualifying_endpoint"),
                "core_time_ratio": entry.get("core_time_ratio"),
                "n_trainable_params": entry.get("n_trainable_params"),
                "inner_val_total_loss": entry.get("inner_val_total_loss"),
                "process_only_note": entry.get("process_only_note"),
            })

        def _key(c: Mapping[str, Any]) -> Any:
            margin = c["max_effect_margin_on_qualifying_endpoint"]
            adj = c["min_holm_adjusted_p_on_qualifying_endpoint"]
            return (
                -float(margin) if _finite(margin) else float("inf"),
                float(adj) if _finite(adj) else float("inf"),
                float(c["core_time_ratio"]) if _finite(c.get("core_time_ratio"))
                else float("inf"),
                int(c["n_trainable_params"])
                if c.get("n_trainable_params") is not None else 2 ** 31,
                float(c["inner_val_total_loss"])
                if _finite(c.get("inner_val_total_loss")) else float("inf"),
                str(c["candidate_id"]),
            )

        qualified.sort(key=_key)
        for rank, c in enumerate(qualified, start=1):
            c["tie_break_rank"] = rank
        champion = qualified[0]["candidate_id"] if qualified else None
        tracks_out[track] = {
            "track": track,
            "n_qualified": len(qualified),
            "qualified_candidates": qualified,
            "champion": champion,
            "retained_baseline": (None if champion
                                  else BASELINE_ID_BY_TRACK[track]),
            "outcome": ("champion_selected" if champion
                        else "no_qualified_candidate_retain_baseline"),
            "note": (None if champion else
                     "§8.2 :816 — a track without a qualified candidate "
                     "keeps its M0/P0 baseline; this is a normal negative "
                     "result, not a failure of the campaign"),
        }
    eliminations = {
        cid: {
            "move_qualified": bool(e.get("move_qualified")),
            "process_qualified": bool(e.get("process_qualified")),
        } for cid, e in sorted(candidates.items())
        if not e.get("h3_qualified")}
    return {
        "schema": SELECT_SCHEMA,
        "kind": "select",
        "gate": "T6 track champions",
        "plan_lines": "§9.3 :994-1012 / §8.2 :816 / 附录 B :1601-1605",
        "rule": ("§9.3 :994-1012 — a track's champion must be H3-qualified "
                 "for THAT track; the frozen effect margin (:1005) is "
                 "tie-break material only and never replaces significance. A "
                 "track with no qualified candidate retains M0/P0 (§8.2 :816)."),
        "tie_break_chain": ["max_effect_margin_desc", "min_holm_adjusted_p",
                            "core_time_ratio_asc",
                            "n_trainable_params_asc",
                            "inner_val_total_loss_asc", "decoder_id_lex"],
        "tie_break_note": ("§9.3 fixes ONLY the effect margin as tie-break; "
                           "the remaining order is stated explicitly here and "
                           "recorded as an ambiguity (it never changes a "
                           "single-candidate outcome)"),
        "screen_score_used": False,
        "screen_score_discipline": ("§8.2 :815/:817 — the screen score ranks "
                                    "the seed43 top-up only and is NEVER an "
                                    "H3/selection input"),
        "tracks": tracks_out,
        "eliminated_candidates": eliminations,
        "n_eligible": sum(t["n_qualified"] for t in tracks_out.values()),
        "dev_used": False,
        "confirm_read": False,
        "written_at": _now(),
    }


def run_select(cfg: Mapping[str, Any], *, campaign_root: Any) -> Dict[str, Any]:
    h3_doc = _load_json(Path(campaign_root) / H3_REPORT_REL,
                        "V543_STATS_H3_REPORT_MISSING",
                        "the H3 report")
    _require(str(h3_doc.get("schema")) == H3_REPORT_SCHEMA,
             "V543_STATS_H3_REPORT_SCHEMA",
             f"the H3 report must be a {H3_REPORT_SCHEMA} document")
    record = select_champions(h3_doc)
    record["config_consistency"] = config_consistency(cfg)
    record["source_h3_sha256"] = _canonical_sha256(
        {k: v for k, v in h3_doc.items() if k != "report_path"})
    path = _write_report(campaign_root, SELECT_REPORT_REL, record)
    record["report_path"] = str(path)
    return record


# ===========================================================================
# plan ambiguities this lane had to resolve (reported, never hidden)
# ===========================================================================
def ambiguities() -> List[Dict[str, Any]]:
    """Every place the plan text left a choice, and what this lane chose."""
    return [
        {
            "id": "h0_item_count",
            "plan_lines": "§9.3 :916-927",
            "issue": ("§9.3 lists TEN bullets; the task brief says '12 项'. "
                      "Two bullets are compound (E(3) names four symmetries; "
                      "'有限梯度、有限 loss' names two checkables), so the "
                      "atomic expansion is 14."),
            "resolution": ("both counts are reported (n_plan_bullets=10, "
                           "n_atomic_checks=14); the plan text is "
                           "authoritative and no twelve-item list exists in "
                           "the plan"),
            "impact": "report labelling only — the gate semantics are unchanged",
        },
        {
            "id": "zero_baseline_epsilon",
            "plan_lines": "§9.3 :960",
            "issue": ("the plan fixes the degraded ABSOLUTE limits (0.01 "
                      "ratio-class, 0.005 Å distance-class) but not the 'near "
                      "zero' threshold that triggers the degradation, and "
                      "requires the rule to be written into WP0 — it is not "
                      "in the campaign config."),
            "resolution": ("eps = 1e-6, reported in every H2 document and in "
                           "zero_baseline_rule"),
            "impact": ("only reachable when a baseline leaf is essentially "
                       "zero; the frozen baseline's leaves are not"),
        },
        {
            "id": "screen_two_seed_aggregation",
            "plan_lines": "§8.2 :811-829",
            "issue": ("the screen-score formula is frozen but the plan does "
                      "not say how the seed17 and seed29 numbers are "
                      "combined into one ranking key."),
            "resolution": ("the point metrics are seed-averaged first and the "
                           "frozen formula is applied to the average "
                           "(mirroring §9.4 :1023 '每 draw 先 seed-average，再"
                           "比较'); per-seed scores are reported beside it"),
            "impact": "changes the ranking only if the two seeds disagree in sign",
        },
        {
            "id": "screen_eligibility_vs_score_only",
            "plan_lines": "§8.2 :814-817 / §9.3 :958",
            "issue": ("§8.2 says 'rank by score and take ≤2'; it does not say "
                      "whether an H2-breached candidate may still be ranked."),
            "resolution": ("a candidate with any per-seed H2 breach is "
                           "ineligible — §9.3 :958 already forbids promoting a "
                           "breached candidate; the reason is recorded per "
                           "candidate"),
            "impact": "can only shrink the top-up set, never enlarge it",
        },
        {
            "id": "h2_block_scope",
            "plan_lines": "§9.3 :937-958",
            "issue": ("the plan assigns MOVE items to M0 and PROCESS items to "
                      "P0, but a single candidate evaluation report contains "
                      "both blocks (the V543 evaluator is one instrument)."),
            "resolution": ("all nine guards are evaluated per seed and "
                           "reported; nothing is skipped and no cross-track "
                           "compensation is allowed"),
            "impact": ("stricter than a track-local reading and catches a "
                       "cross-track regression that the plan forbids"),
        },
        {
            "id": "selection_tie_break_beyond_margin",
            "plan_lines": "§9.3 :1005-1012",
            "issue": ("the plan fixes the effect margin as the tie-break but "
                      "not the order after it."),
            "resolution": ("adjusted p → core time → parameter count → inner-"
                           "val loss → id, all recorded in tie_break_chain"),
            "impact": "only ever breaks a tie between already-qualified candidates",
        },
        {
            "id": "endpoint_list_for_H3_holm",
            "plan_lines": "§9.3 :1001-1002",
            "issue": ("'MOVE 两个出口' names rmse_zero and the direction "
                      "metric; the PROCESS family names '2 endpoints' — AUPRC "
                      "and q MAE.  Recall/ECE/ordering appear only as "
                      "non-inferiority conditions, never as test endpoints."),
            "resolution": ("four confirmatory endpoints (move_rmse, "
                           "move_direction, process_auprc, process_q); the "
                           "remaining PROCESS conditions stay point-estimate "
                           "constraints of the bracket"),
            "impact": ("the family sizes are 1 (MOVE rmse), 1 (MOVE "
                       "direction), 2×N (PROCESS)"),
        },
        {
            "id": "cross_seed_cluster_intersection",
            "plan_lines": "§9.4 :1019 / :1023 / :1024 / §9.5 :1037",
            "issue": ("§9.4 requires pairing on 'same cluster, same frame set' "
                      "and seed-averaging INSIDE every draw, but does not say "
                      "whether the three seeds must share one cluster set. "
                      "Each formal seed evaluates its own inner-val sample, so "
                      "an all-seed × both-sides intersection can be far smaller "
                      "than a single seed's curve count (measured on the frozen "
                      "M0/P0 baseline: 1,024 curves per seed, 37 shared by all "
                      "three)."),
            "resolution": ("RESOLVED BY THE EVALUATOR OWNER (2026-11-03): the "
                           "producer now ships paired_curve_bootstrap_multiseed "
                           "and declares it CONFIRMATORY in "
                           "bootstrap_inputs.modes.per_seed_independent, while "
                           "the shared-intersection function stays "
                           "audit/compatibility only (modes."
                           "shared_intersection). This lane consumes both: the "
                           "multi-seed call drives every H3 verdict, the "
                           "shared-intersection call is reported with "
                           "audit_only=true and its n_curves_shared (37) so the "
                           "old reading stays reproducible. seed_order / draws "
                           "/ seed_root / higher_is_better are read from the "
                           "same bootstrap_inputs block (no hard-coded "
                           "instrument values in this lane)."),
            "impact": ("the confirmatory H3 test now runs on every seed's own "
                       "~1,020 curves instead of the 37-curve intersection; the "
                       "audit figure is retained for continuity with any "
                       "earlier reading"),
        },
    ]


def _console_view(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop the bulky per-curve tables before printing to stdout."""
    out = {k: v for k, v in dict(doc).items() if k != "per_curve"}
    if isinstance(out.get("runs"), list):
        out["runs"] = [{k: v for k, v in row.items() if k != "per_curve"}
                       for row in out["runs"]]
    out.setdefault("_printed", "per-curve tables omitted from stdout; they "
                               "are in the written report/table")
    return out


def _print(doc: Mapping[str, Any]) -> None:
    print(json.dumps(_console_view(doc), indent=1, sort_keys=True,
                     default=str))


def _load_cfg(config_path: Any) -> Dict[str, Any]:
    import yaml

    path = Path(config_path)
    _require(path.is_file(), "V543_STATS_CONFIG_MISSING",
             f"campaign config not found: {path}")
    try:
        return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    except (OSError, ValueError) as exc:
        raise V543StatsError("V543_STATS_CONFIG_UNREADABLE",
                             f"{path}: {exc}") from exc


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v543_stats.py",
        description="V5.4.3 WP6 statistics lane (extract / h0 / h1 / h2 / "
                    "screen / h3 / select) over the frozen pre-registered "
                    "contract; no dev read, no confirm read, no QC call, no "
                    "threshold knobs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser(
        "extract", help="evaluation documents → append-only metrics table "
                        "(per-curve passthrough + §9.5 SHA chain)")
    p_ext.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_ext.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_ext.add_argument("--baseline", default=None,
                       help="the §9.5 M0/P0 baseline document (default: "
                            "<campaign-root>/" + DEFAULT_BASELINE_REL + ")")
    p_ext.add_argument("--candidate", dest="candidates", nargs="*", default=[],
                       help="candidate bundle JSON documents")
    p_ext.add_argument("--check-instrument", action="store_true",
                       help="also verify the on-disk evaluator still hashes "
                            "to the baseline's §9.5 claim")

    p_h0 = sub.add_parser("h0", help="H0 legality checklist (§9.3 :916-927)")
    p_h0.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_h0.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_h0.add_argument("--evidence", default=None)

    p_h1 = sub.add_parser("h1", help="H1 resource gate (§9.3 :929-935)")
    p_h1.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_h1.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_h1.add_argument("--input", default=None,
                      help="the pesn_v543_h1_input_v1 measurement document")

    p_h2 = sub.add_parser("h2", help="H2 per-seed non-inferiority (§9.3 :937-960)")
    p_h2.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_h2.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_h2.add_argument("--self-comparison", action="store_true",
                      help="also run the baseline rows against themselves "
                           "(the denominator sanity check: every guard must "
                           "pass with Δ = 0)")

    p_scr = sub.add_parser("screen",
                           help="§8.2 screen score → ≤2 seed43 top-ups per track")
    p_scr.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_scr.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_scr.add_argument("--self-comparison", action="store_true",
                       help="allow the baseline rows themselves as placeholder "
                            "candidates (formula plumbing check only — the "
                            "report is marked scientific_claim=false)")

    p_h3 = sub.add_parser("h3", help="H3 brackets + paired bootstrap + Holm")
    p_h3.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_h3.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_h3.add_argument("--draws", type=int, default=None,
                      help="test seam only; defaults to the producer's "
                           "bootstrap_inputs.draws (frozen 10000)")
    p_h3.add_argument("--self-comparison", action="store_true",
                      help="M0/P0 versus itself (plumbing check; "
                           "scientific_claim=false)")

    p_sel = sub.add_parser("select", help="per-track champion (§9.3 :994-1012)")
    p_sel.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_sel.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))

    args = parser.parse_args(argv)
    try:
        cfg = _load_cfg(args.config)
        root = args.campaign_root
        if args.command == "extract":
            report = run_extract(cfg, campaign_root=root,
                                 baseline_path=args.baseline,
                                 candidates=args.candidates,
                                 check_instrument=bool(args.check_instrument))
        elif args.command == "h0":
            report = run_h0(cfg, campaign_root=root,
                            evidence_path=args.evidence)
        elif args.command == "h1":
            report = run_h1(cfg, campaign_root=root, input_path=args.input)
        elif args.command == "h2":
            report = run_h2(cfg, campaign_root=root,
                            self_comparison=bool(args.self_comparison))
        elif args.command == "screen":
            report = run_screen(cfg, campaign_root=root,
                                self_comparison=bool(args.self_comparison))
        elif args.command == "h3":
            report = run_h3(cfg, campaign_root=root, draws=args.draws,
                            self_comparison=bool(args.self_comparison))
        else:
            report = run_select(cfg, campaign_root=root)
        report.setdefault("ambiguities", ambiguities())
        _print(report)
        return 0
    except V543StatsError as exc:
        print(f"V543 statistics lane refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
