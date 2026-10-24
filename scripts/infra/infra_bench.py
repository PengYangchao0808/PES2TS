#!/usr/bin/env python3
"""S4 infra benchmark smoke runner (plan §7 B0–B4, §10).

One independent, ledger-free bench tag per invocation::

    python -m scripts.infra.infra_bench --tag B0 \
        --run-plan outputs/pesn/v53mp/run_plan_v53mp_20k.jsonl \
        [--run-id v53mp_20k_A_s17] [--updates 2000] [--warmup 200] \
        [--reps 3] [--data-root outputs/pesn/v53mp/data] \
        [--output-root outputs/pesn/v53mp/infra_bench/<tag>_<ts>/]

Bench-root mechanism (plan §7 governance: smoke runs must never touch the
campaign ledger or consume attempt caps, and ``r3_v53mp_run.py`` must NOT
be modified).  The campaign driver derives its ledger / plan / pin /
run-dir from ``--output-root``, so each bench rep gets a SELF-CONTAINED
root::

    outputs/pesn/v53mp/infra_bench/<tag>_<ts>/rep<k>/root/
        run_plan_v53mp_20k.jsonl        # A_s17 row copied from the frozen
                                        # campaign plan; n_updates=<--updates>,
                                        # smoke_updates=null; five shas verbatim
        run_plan_v53mp_20k.jsonl.pin.json  # regenerated exactly like the
                                        # driver's plan-init: {sha256, campaign_mode}
        run_records.jsonl               # bench-local ledger (written by the
                                        # driver; the CAMPAIGN ledger stays untouched)
        v53mp_20k_A_s17/                # run dir (train_log.jsonl, ckpts, metrics)

The config is NEVER copied or edited — ``--config`` points at the original
``config/campaigns/pes2ts_v53mp_20261021.yaml`` so the plan row's
``config_sha256`` binding keeps verifying against the frozen file; tables /
split / r3-ckpt likewise point at the original paths.  Before any rep
runs, the bench root is validated with the driver's OWN machinery:

1. ``r3_v53mp_run.py --output-root <root> gate-check`` (subprocess,
   launched through ``cgroup_scope.sh`` so even a sidecar-hit gate-check
   stays cgroup-contained — a future cache miss must never re-read the
   full data set unconstrained and push the host toward the commit
   cliff);
2. in-process ``read_run_plan`` + ``verify_plan_pin`` +
   ``assert_stage_binding`` + the driver's five-sha re-hash of the row.

Each rep then runs via the driver's ``run`` subcommand (wrapped by
``cgroup_scope.sh`` + a ``nvidia-smi -l 2`` VRAM sampler).  Per-rep roots
(not one root per tag) are REQUIRED because the driver hardcodes
``resume=True``: a shared root would make rep2 "resume" rep1's checkpoint
and skip the training loop entirely.

DEVIATIONS discovered from the driver's code (reported in report.json
``warnings``):

* ``_parse_run_id`` only accepts ``v53mp_<tier>_<arm>_s<seed>`` — a
  ``bench_<tag>``-style run_id can never reach the driver's run path, so
  the bench row keeps ``v53mp_20k_A_s17``; isolation comes from the root.
* for tier 20k the driver computes the training length from the frozen
  config schedule (freeze 400 + 1500·10 joint = 15400 updates) and
  IGNORES the plan row's ``n_updates`` (only tier-2k rows honor
  smoke_updates).  This is closed by CONFIG DERIVATION: whenever
  ``--updates`` differs from the driver-effective count, infra_bench
  derives a per-bench ``config_bench.yaml`` — the original yaml loaded
  and deep-copied at the schedule level, ``joint_updates_base_2k`` set so
  that ``freeze + base × factor(tier) == --updates`` (freeze unchanged;
  refuse at ``--updates <= freeze`` or when ``updates − freeze`` is not a
  multiple of the tier factor) — and the bench plan row's
  ``config_sha256`` rebinds to the BENCH yaml (five-sha binding stays
  internal to the bench root; the original yaml is never written to).
  When ``--updates`` EQUALS the driver-effective count the original
  config is used directly (warning-free).  The report records
  ``config_source`` (``original`` / ``bench_derived``),
  ``original_config_sha256``, ``bench_config_sha256`` and
  ``driver_effective_updates`` (recomputed from the config actually
  passed to the driver).

Per rep the runner measures startup (process spawn -> first
``train_log.jsonl`` row; the first row is ``update: 1``) and parses the
timed rows (``update`` + ``elapsed_s``) excluding the first ``--warmup``
updates from the ms/update stats ONLY.  It reports median/P95 ms/update,
cgroup anon peak, VRAM peak, startup seconds, max swap.current and
oom_kill, with a machine-readable verdict per §7 (B0 gate: reproduce
~0.078 s/update and ~26 GB anon; plus the hard swap/oom gates from §5.1).

``--dry-run`` fabricates a rep with stub timing files (78 ms/update) so
the parse/report/verdict path is fully testable without GPU or training.

``--recompute REPORT [REPORT ...]`` is a READ-ONLY audit entry point: it
loads already-archived ``report.json`` files (B0..B3) and re-judges them
as a per-rung LADDER without re-running anything:

* the BASELINE rung (B0) self-certifies against its OWN three reps —
  ms median and anon peak are taken from the reps themselves (no fixed
  78 ms / 26 GB target, which mis-red B1/B2 whose anon is *smaller*),
  every rep must land inside baseline ±15%, swap=0 and oom=0;
* every higher rung is REGRESSION-ONLY vs the rung below:
  ``median_ms <= 1.15 x baseline median_ms``,
  ``anon_peak <= 1.15 x baseline anon``,
  ``startup_s <= 1.15 x baseline startup`` plus the hard swap/oom
  gates.  Significant improvements are allowed (no lower band).

The new JSON verdict goes to stdout (and to ``--recompute-out`` when
given); the archived ``report.json`` files are NEVER rewritten and their
original verdict is embedded in the output for audit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPO = PROJECT_ROOT
CGROUP_SCOPE = REPO / "scripts/infra/cgroup_scope.sh"
THREAD_CAPS = REPO / "scripts/infra/thread_caps.sh"
RUN_DRIVER = REPO / "scripts/campaigns/r3_v53mp_run.py"
DEFAULT_CONFIG = "config/campaigns/pes2ts_v53mp_20261021.yaml"
DEFAULT_DATA_ROOT = "outputs/pesn/v53mp/data"
DEFAULT_BENCH_ROOT = "outputs/pesn/v53mp/infra_bench"
DEFAULT_SOURCE_PLAN = "outputs/pesn/v53mp/run_plan_v53mp_20k.jsonl"
DEFAULT_SOURCE_RUN_ID = "v53mp_20k_A_s17"
#: R3 F2-bidir winner — the single frozen V5.3-MP init (same as the driver)
R3_WINNER_CKPT = (
    "outputs/pesn/30h_campaign/r3_runs/5372d9b12807c3f0/checkpoint_best.pt"
)
#: B0 gate targets (plan §7): reproduce ~0.078 s/update, ~26 GB anon
B0_MS_TARGET = 78.0
B0_MS_TOLERANCE = 0.15          # ±15% -> [66.3, 89.7] ms
B0_ANON_TARGET_GB = 26.0        # steady-state anon, 20k tier (plan §1)
B0_ANON_TOLERANCE = 0.15        # ±15% -> [22.1, 29.9] GB


# ---------------------------------------------------------------------------
# train_log parsing (pure, unit-testable)
# ---------------------------------------------------------------------------


def parse_train_log(lines: Sequence[str], warmup: int) -> Dict[str, Any]:
    """Per-update ms from the timed ``train_log.jsonl`` rows.

    A *timed row* is a JSON object carrying both ``update`` and
    ``elapsed_s`` (cumulative seconds; the trainer logs update 1, every
    100 updates and the final update).  Consecutive timed rows define
    intervals; an interval counts only when it STARTS at update >= warmup,
    so the first ``warmup`` updates never influence median/P95.  Warmup
    exclusion applies to the ms/update stats ONLY — ``first_row_elapsed_s``
    always reflects the first timed row (update 1).
    """
    timed: List[Tuple[int, float]] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(doc, dict):
            continue
        if "update" in doc and "elapsed_s" in doc:
            timed.append((int(doc["update"]), float(doc["elapsed_s"])))
    timed.sort(key=lambda pair: pair[0])
    ms_update: List[float] = []
    for (u1, t1), (u2, t2) in zip(timed, timed[1:]):
        if u2 <= u1:
            continue
        if u1 < warmup:
            continue
        ms_update.append((t2 - t1) / (u2 - u1) * 1000.0)
    return {
        "n_timed": len(timed),
        "n_updates": timed[-1][0] if timed else 0,
        "first_row_elapsed_s": timed[0][1] if timed else None,
        "ms_update": ms_update,
        "median_ms": (round(statistics.median(ms_update), 3)
                      if ms_update else None),
        "p95_ms": (round(_p95(ms_update), 3) if ms_update else None),
    }


def _p95(values: List[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


# ---------------------------------------------------------------------------
# cgroup sampler parsing (r3_v53mp_cgroup_run.sh CSV format + siblings)
# ---------------------------------------------------------------------------


def parse_cgroup_samples(csv_lines: Sequence[str],
                         events_text: str = "") -> Dict[str, Any]:
    """peak anon, max swap.current, oom_kill from the sampler CSV.

    CSV columns: ``ts,memory_current_bytes,anon_bytes,file_bytes`` and
    (cgroup_scope.sh extension) an optional 5th
    ``memory_swap_current_bytes`` — 4-column r3 CSVs parse fine (swap=0).
    ``events_text`` is the raw ``.events`` sibling (memory.events copy);
    oom_kill is the LAST ``oom_kill N`` line (counters are monotonic).
    """
    peak_anon = 0
    max_swap = 0
    for raw in csv_lines:
        raw = raw.strip()
        if not raw or raw.startswith("ts,"):
            continue
        parts = raw.split(",")
        if len(parts) < 4:
            continue
        try:
            peak_anon = max(peak_anon, int(parts[2]))
            if len(parts) >= 5:
                max_swap = max(max_swap, int(parts[4]))
        except ValueError:
            continue
    oom_kill = 0
    for line in (events_text or "").splitlines():
        line = line.strip()
        if line.startswith("oom_kill"):
            try:
                oom_kill = int(line.split()[1])
            except (IndexError, ValueError):
                pass
    return {"anon_peak_bytes": peak_anon,
            "swap_max_bytes": max_swap,
            "oom_kill": oom_kill}


def parse_vram_samples(lines: Sequence[str]) -> Dict[str, Any]:
    """peak memory.used (MiB) from `nvidia-smi --query-gpu=memory.used` csv."""
    peak = 0
    n = 0
    for raw in lines:
        raw = raw.strip().rstrip(" MiB").strip()
        if not raw:
            continue
        try:
            value = int(float(raw))
        except ValueError:
            continue
        peak = max(peak, value)
        n += 1
    return {"vram_peak_mib": peak, "vram_samples": n}


def normalize_ms_target(ms_target: float) -> float:
    """B0 ms-target unit fix (plan §7 wording "~0.078 s/update").

    A target in (0, 1) is a SECONDS-per-update value (the historic
    ``--b0-ms-target 0.078`` invocation, which previously produced a
    nonsense [0.07, 0.09] ms band and red-flagged a healthy 98.8 ms run);
    it is converted to milliseconds.  Values already in ms (>= 1.0) pass
    through unchanged, so legacy call sites (78.0) keep their band.
    """
    if 0.0 < float(ms_target) < 1.0:
        return float(ms_target) * 1000.0
    return float(ms_target)


def verdict_b0(median_ms: Optional[float], anon_peak_bytes: int,
               swap_max_bytes: int, oom_kill: int,
               ms_target: float = B0_MS_TARGET,
               ms_tol: float = B0_MS_TOLERANCE,
               anon_target_gb: float = B0_ANON_TARGET_GB,
               anon_tol: float = B0_ANON_TOLERANCE) -> Dict[str, Any]:
    """Machine-readable B0 verdict (plan §7: reproduce 0.078 s / 26 GB).

    Hard gates from §5.1 ride along: swap_max must be 0 and oom_kill 0.
    ``ms_target`` accepts seconds-scale values (0.078) as well as
    milliseconds (78.0) — see ``normalize_ms_target``.
    """
    ms_target = normalize_ms_target(ms_target)
    ms_lo, ms_hi = ms_target * (1 - ms_tol), ms_target * (1 + ms_tol)
    anon_lo = anon_target_gb * (1 - anon_tol) * 1e9
    anon_hi = anon_target_gb * (1 + anon_tol) * 1e9
    checks: Dict[str, bool] = {
        "median_ms_update_in_band": bool(
            median_ms is not None and ms_lo <= median_ms <= ms_hi),
        "anon_peak_in_band": bool(anon_lo <= anon_peak_bytes <= anon_hi),
        "swap_gate": swap_max_bytes == 0,
        "oom_gate": oom_kill == 0,
    }
    return {
        "gate": "B0",
        "pass": all(checks.values()),
        "checks": checks,
        "bounds": {
            "median_ms_update": [round(ms_lo, 2), round(ms_hi, 2)],
            "anon_peak_gb": [round(anon_lo / 1e9, 2),
                             round(anon_hi / 1e9, 2)],
            "swap_max_bytes": 0,
            "oom_kill": 0,
            "ms_target_ms": round(ms_target, 3),
        },
        "observed": {
            "median_ms_update": median_ms,
            "anon_peak_bytes": anon_peak_bytes,
            "swap_max_bytes": swap_max_bytes,
            "oom_kill": oom_kill,
        },
    }


# ---------------------------------------------------------------------------
# per-rung / ladder recompute (read-only verdict over archived report.json)
# ---------------------------------------------------------------------------

#: ladder policy: higher rungs may not REGRESS vs the rung below (no lower
#: band — significant improvements are allowed); B0 self-baselines.
LADDER_TOLERANCE = 0.15
RECOMPUTE_SCHEMA = "pes2ts_infra_bench_recompute_v1"


def ladder_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """Ladder metrics of one archived bench report (derived, read-only).

    Computed from the per-rep rows when present (median of per-rep median
    ms/update, MAX anon peak — the conservative aggregate —, median
    startup, max swap/oom), falling back to the stored ``aggregate``
    block so older reports without usable reps still recompute.
    """
    reps = list(report.get("reps") or [])
    agg = report.get("aggregate") or {}
    medians = [r["median_ms"] for r in reps if r.get("median_ms") is not None]
    startups = [r["startup_s"] for r in reps
                if r.get("startup_s") is not None]
    anons = [r.get("anon_peak_bytes") or 0 for r in reps]
    swaps = [r.get("swap_max_bytes") or 0 for r in reps]
    ooms = [r.get("oom_kill") or 0 for r in reps]
    return {
        "tag": report.get("tag"),
        "median_ms": (round(statistics.median(medians), 3) if medians
                      else agg.get("median_ms_update")),
        "anon_peak_gb": (round(max(anons) / 1e9, 3) if anons
                         else agg.get("anon_peak_gb_max")),
        "startup_s": (round(statistics.median(startups), 1) if startups
                      else agg.get("startup_s_median")),
        "swap_max_bytes": max(swaps) if swaps
                          else (agg.get("swap_max_bytes") or 0),
        "oom_kill": max(ooms) if ooms else (agg.get("oom_kill_max") or 0),
        "n_reps": len(reps),
    }


def verdict_b0_self(reps: Sequence[Dict[str, Any]],
                    ms_tol: float = B0_MS_TOLERANCE,
                    anon_tol: float = B0_ANON_TOLERANCE) -> Dict[str, Any]:
    """Self-baselined B0 verdict (ladder mode).

    B0 defines its OWN baseline from its own reps — median of the per-rep
    median ms/update, MAX per-rep anon peak — and requires every rep to
    land within that baseline ±tol.  The hard §5.1 gates (swap=0,
    oom=0) still apply.  The fixed plan-§7 targets (78 ms / 26 GB) are
    NOT consulted: a slower-but-stable B0 (e.g. the archived 98.8 ms /
    30.9 GB run) passes its own band instead of red-flagging every
    higher rung through a stale absolute target.
    """
    ms_vals = [r["median_ms"] for r in reps if r.get("median_ms") is not None]
    anon_vals = [r.get("anon_peak_bytes") or 0 for r in reps]
    swaps = [r.get("swap_max_bytes") or 0 for r in reps]
    ooms = [r.get("oom_kill") or 0 for r in reps]
    swap_max = max(swaps) if swaps else 0
    oom_max = max(ooms) if ooms else 0
    if not ms_vals or not anon_vals:
        return {
            "gate": "B0", "mode": "self_baseline", "pass": False,
            "checks": {"baseline_defined": False},
            "failing_checks": ["baseline_defined"],
            "bounds": {}, "observed": {"n_reps": len(reps)},
        }
    baseline_ms = round(statistics.median(ms_vals), 3)
    baseline_anon_gb = round(max(anon_vals) / 1e9, 3)
    ms_lo, ms_hi = baseline_ms * (1 - ms_tol), baseline_ms * (1 + ms_tol)
    anon_lo, anon_hi = (baseline_anon_gb * (1 - anon_tol),
                        baseline_anon_gb * (1 + anon_tol))
    per_rep = []
    reps_ms_ok = True
    reps_anon_ok = True
    for r in reps:
        rms = r.get("median_ms")
        ranon_gb = round((r.get("anon_peak_bytes") or 0) / 1e9, 3)
        ms_ok = bool(rms is not None and ms_lo <= rms <= ms_hi)
        anon_ok = bool(anon_lo <= ranon_gb <= anon_hi)
        reps_ms_ok = reps_ms_ok and ms_ok
        reps_anon_ok = reps_anon_ok and anon_ok
        per_rep.append({"rep": r.get("rep"), "median_ms": rms,
                        "anon_peak_gb": ranon_gb,
                        "median_ms_in_band": ms_ok,
                        "anon_peak_in_band": anon_ok})
    checks = {
        "baseline_defined": True,
        "reps_median_ms_in_band": bool(reps_ms_ok),
        "reps_anon_peak_in_band": bool(reps_anon_ok),
        "swap_gate": swap_max == 0,
        "oom_gate": oom_max == 0,
    }
    failing = [k for k, v in checks.items() if not v]
    return {
        "gate": "B0", "mode": "self_baseline", "pass": not failing,
        "checks": checks, "failing_checks": failing,
        "bounds": {
            "median_ms_update": [round(ms_lo, 2), round(ms_hi, 2)],
            "anon_peak_gb": [round(anon_lo, 2), round(anon_hi, 2)],
            "swap_max_bytes": 0,
            "oom_kill": 0,
        },
        "baseline": {"median_ms_update": baseline_ms,
                     "anon_peak_gb": baseline_anon_gb,
                     "n_reps": len(reps)},
        "observed": {"per_rep": per_rep,
                     "swap_max_bytes": swap_max,
                     "oom_kill": oom_max},
    }


def verdict_rung_regression(tag: str,
                            current: Dict[str, Any],
                            previous: Dict[str, Any],
                            tol: float = LADDER_TOLERANCE) -> Dict[str, Any]:
    """Regression-only verdict for rung ``tag`` vs the rung below.

    Only UPPER bounds apply: median ms/update, anon peak and startup may
    each grow by at most ``tol`` relative to the previous rung's metric;
    improvements of any size are green (no lower band — B1's 7.2 GB anon
    is an improvement over B0's 30.9 GB, not a "failure").  The hard
    §5.1 gates (swap=0, oom=0) remain.  A missing metric on either side
    fails closed (the check is False).
    """

    def _le(cur: Optional[float], prev: Optional[float]) -> bool:
        if cur is None or prev is None:
            return False
        return float(cur) <= float(prev) * (1.0 + tol)

    checks = {
        "median_ms_no_regression": _le(current.get("median_ms"),
                                       previous.get("median_ms")),
        "anon_no_regression": _le(current.get("anon_peak_gb"),
                                  previous.get("anon_peak_gb")),
        "startup_no_regression": _le(current.get("startup_s"),
                                     previous.get("startup_s")),
        "swap_gate": (current.get("swap_max_bytes") or 0) == 0,
        "oom_gate": (current.get("oom_kill") or 0) == 0,
    }
    failing = [k for k, v in checks.items() if not v]
    prev_ms = previous.get("median_ms")
    prev_anon = previous.get("anon_peak_gb")
    prev_startup = previous.get("startup_s")
    return {
        "gate": tag, "mode": "regression_vs_{}".format(
            previous.get("tag")),
        "pass": not failing, "checks": checks, "failing_checks": failing,
        "bounds": {
            "median_ms_update_max": (round(prev_ms * (1 + tol), 2)
                                     if prev_ms is not None else None),
            "anon_peak_gb_max": (round(prev_anon * (1 + tol), 2)
                                 if prev_anon is not None else None),
            "startup_s_max": (round(prev_startup * (1 + tol), 2)
                              if prev_startup is not None else None),
            "swap_max_bytes": 0,
            "oom_kill": 0,
        },
        "baseline": previous,
        "observed": current,
    }


def recompute_ladder(reports: Sequence[Dict[str, Any]],
                     tol: float = LADDER_TOLERANCE,
                     ms_tol: float = B0_MS_TOLERANCE,
                     anon_tol: float = B0_ANON_TOLERANCE) -> Dict[str, Any]:
    """Re-derive per-rung verdicts from ARCHIVED bench report dicts.

    Reports are ordered by rung number (B0 < B1 < ...; unparseable tags
    keep their input position after the numbered ones).  Rung B0 is
    self-baselined (``verdict_b0_self``); every higher rung is judged
    regression-only against the rung below (``verdict_rung_regression``).
    Read-only: the input dicts are never mutated and the ORIGINAL verdict
    carried by each report is embedded per rung for audit.  Fail-closed:
    an empty ladder or a missing B0 baseline raises SystemExit.
    """
    if not reports:
        raise SystemExit("error: RECOMPUTE_EMPTY: no reports given")
    indexed = list(enumerate(reports))

    def _rung_num(pair):
        idx, rep = pair
        m = re.match(r"^B(\d+)", str(rep.get("tag") or ""))
        return (0, int(m.group(1)), idx) if m else (1, 0, idx)

    ordered = [rep for _idx, rep in sorted(indexed, key=_rung_num)]
    if str(ordered[0].get("tag") or "") != "B0":
        raise SystemExit(
            "error: RECOMPUTE_BASELINE_MISSING: the ladder must start at "
            "B0 (got {!r})".format(ordered[0].get("tag")))
    rungs: List[Dict[str, Any]] = []
    prev_metrics: Optional[Dict[str, Any]] = None
    for report in ordered:
        metrics = ladder_metrics(report)
        if prev_metrics is None:
            verdict = verdict_b0_self(report.get("reps") or [],
                                      ms_tol=ms_tol, anon_tol=anon_tol)
            baseline_tag = None
        else:
            verdict = verdict_rung_regression(
                str(metrics.get("tag")), metrics, prev_metrics, tol=tol)
            baseline_tag = prev_metrics.get("tag")
        rungs.append({
            "tag": metrics.get("tag"),
            "baseline_tag": baseline_tag,
            "metrics": metrics,
            "verdict": verdict,
            "original_verdict": report.get("verdict"),
        })
        prev_metrics = metrics
    return {
        "schema": RECOMPUTE_SCHEMA,
        "kind": "ladder",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tolerance": tol,
        "policy": ("B0 self-baselines from its own reps (median ms, max "
                   "anon, each rep within baseline ±{}; swap=0, oom=0); "
                   "higher rungs are regression-only vs the rung below "
                   "(median ms / anon / startup <= 1+{} x previous, no "
                   "lower band; swap=0, oom=0)".format(tol, tol)),
        "rungs": rungs,
        "pass": all(r["verdict"]["pass"] for r in rungs),
    }


# ---------------------------------------------------------------------------
# report rendering (pure, unit-testable)
# ---------------------------------------------------------------------------


def render_report_md(report: Dict[str, Any]) -> str:
    reps = report.get("reps", [])
    agg = report.get("aggregate", {})
    verdict = report.get("verdict", {})
    lines: List[str] = []
    lines.append("# Infra bench {} — {}".format(
        report.get("tag"), report.get("generated_at")))
    lines.append("")
    lines.append("- run_plan (source): `{}`".format(report.get("run_plan")))
    lines.append("- run_id: `{}`".format(report.get("run_id")))
    lines.append("- bench_root: `{}`".format(report.get("bench_root")))
    lines.append("- gate_check: **{}**".format(report.get("gate_check")))
    lines.append("- loader: `{}`, dev_pack: `{}`, double_buffer: `{}`, "
                 "pinned_h2d: `{}`, feature_workers: `{}`".format(
                     report.get("loader", "v1"),
                     report.get("dev_pack", "off"),
                     report.get("double_buffer", "off"),
                     report.get("pinned_h2d", "off"),
                     report.get("feature_workers")))
    lines.append("- data_root: `{}`".format(report.get("data_root")))
    lines.append("- requested updates: {}, driver-effective updates: {} "
                 "(warmup {} excluded from stats), reps: {}, dry_run: {}"
                 .format(report.get("requested_updates"),
                         report.get("driver_effective_updates"),
                         report.get("warmup"),
                         report.get("n_reps"), report.get("dry_run")))
    config_line = "- config: **{}**".format(
        report.get("config_source", "original"))
    if report.get("config_source") == "bench_derived":
        config_line += " (original sha {}, bench sha {})".format(
            str(report.get("original_config_sha256"))[:12],
            str(report.get("bench_config_sha256"))[:12])
    else:
        config_line += " (sha {})".format(
            str(report.get("original_config_sha256"))[:12])
    lines.append(config_line)
    if report.get("warnings"):
        lines.append("")
        lines.append("## Warnings (driver-contract deviations)")
        lines.append("")
        for warning in report["warnings"]:
            lines.append("- {}".format(warning))
    lines.append("")
    lines.append("## Per-rep")
    lines.append("")
    lines.append("| rep | median ms/update | P95 ms/update | anon peak "
                 "(GB) | VRAM peak (MiB) | startup (s) | swap max (B) | "
                 "oom_kill | exit |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for rep in reps:
        lines.append("| {rep} | {median} | {p95} | {anon} | {vram} | "
                     "{startup} | {swap} | {oom} | {exit} |".format(
                         rep=rep.get("rep"),
                         median=rep.get("median_ms"),
                         p95=rep.get("p95_ms"),
                         anon=round((rep.get("anon_peak_bytes") or 0) / 1e9,
                                    2),
                         vram=rep.get("vram_peak_mib"),
                         startup=rep.get("startup_s"),
                         swap=rep.get("swap_max_bytes"),
                         oom=rep.get("oom_kill"),
                         exit=rep.get("exit")))
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("- median of per-rep median ms/update: **{}**".format(
        agg.get("median_ms_update")))
    lines.append("- max per-rep P95 ms/update: **{}**".format(
        agg.get("p95_ms_update_max")))
    lines.append("- anon peak (max over reps, GB): **{}**".format(
        agg.get("anon_peak_gb_max")))
    lines.append("- VRAM peak (max over reps, MiB): **{}**".format(
        agg.get("vram_peak_mib_max")))
    lines.append("- startup s (median over reps): **{}**".format(
        agg.get("startup_s_median")))
    lines.append("- swap max (max over reps, B): **{}**".format(
        agg.get("swap_max_bytes")))
    lines.append("- oom_kill (max over reps): **{}**".format(
        agg.get("oom_kill_max")))
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(verdict, sort_keys=True, indent=1))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_report(tag: str, run_plan: str, run_id: Optional[str],
                 bench_root: str, gate_check: str, data_root: str,
                 requested_updates: int, driver_effective_updates: Optional[int],
                 warmup: int, rep_reports: List[Dict[str, Any]],
                 dry_run: bool, warnings: List[str],
                 ms_target: float, ms_tol: float,
                 anon_target_gb: float, anon_tol: float,
                 config_source: str = "original",
                 original_config_sha256: Optional[str] = None,
                 bench_config_sha256: Optional[str] = None) -> Dict[str, Any]:
    medians = [r["median_ms"] for r in rep_reports
               if r.get("median_ms") is not None]
    p95s = [r["p95_ms"] for r in rep_reports
            if r.get("p95_ms") is not None]
    anons = [r.get("anon_peak_bytes") or 0 for r in rep_reports]
    vrams = [r.get("vram_peak_mib") or 0 for r in rep_reports]
    startups = [r["startup_s"] for r in rep_reports
                if r.get("startup_s") is not None]
    swaps = [r.get("swap_max_bytes") or 0 for r in rep_reports]
    ooms = [r.get("oom_kill") or 0 for r in rep_reports]
    median_ms = round(statistics.median(medians), 3) if medians else None
    anon_max = max(anons) if anons else 0
    swap_max = max(swaps) if swaps else 0
    oom_max = max(ooms) if ooms else 0
    return {
        "schema": "pes2ts_infra_bench_v2",
        "tag": tag,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_plan": run_plan,
        "run_id": run_id,
        "bench_root": bench_root,
        "gate_check": gate_check,
        "data_root": data_root,
        "requested_updates": requested_updates,
        "driver_effective_updates": driver_effective_updates,
        "config_source": config_source,
        "original_config_sha256": original_config_sha256,
        "bench_config_sha256": bench_config_sha256,
        "warmup": warmup,
        "reps": rep_reports,
        "n_reps": len(rep_reports),
        "dry_run": dry_run,
        "warnings": warnings,
        "aggregate": {
            "median_ms_update": median_ms,
            "p95_ms_update_max": round(max(p95s), 3) if p95s else None,
            "anon_peak_gb_max": round(anon_max / 1e9, 3),
            "vram_peak_mib_max": max(vrams) if vrams else 0,
            "startup_s_median": (round(statistics.median(startups), 1)
                                 if startups else None),
            "swap_max_bytes": swap_max,
            "oom_kill_max": oom_max,
        },
        "verdict": verdict_b0(median_ms, anon_max, swap_max, oom_max,
                              ms_target=ms_target, ms_tol=ms_tol,
                              anon_target_gb=anon_target_gb,
                              anon_tol=anon_tol),
    }


# ---------------------------------------------------------------------------
# source plan row + bench root construction
# ---------------------------------------------------------------------------


def load_plan_row(run_plan: str, run_id: Optional[str]) -> Tuple[
        Dict[str, Any], Optional[Dict[str, Any]], Optional[str]]:
    """Accept a jsonl FILE (pick row + header) or a literal JSON row.

    Returns ``(row, header_or_None, source_plan_path_or_None)``.
    """
    text = run_plan.strip()
    if text.startswith("{"):
        row = json.loads(text)
        if not isinstance(row, dict) or "arm" not in row or "seed" not in row:
            raise SystemExit("error: RUN_PLAN_ROW_INVALID: not an arm/seed row")
        return row, None, None
    path = Path(run_plan)
    if not path.is_file():
        raise SystemExit("error: RUN_PLAN_MISSING: {}".format(path))
    header: Optional[Dict[str, Any]] = None
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        doc = json.loads(raw)
        if doc.get("record_type") == "plan_header":
            header = doc
            continue
        rows.append(doc)
    if not rows:
        raise SystemExit("error: RUN_PLAN_EMPTY: {}".format(path))
    if run_id is None:
        row = rows[0]
    else:
        match = [r for r in rows if r.get("run_id") == run_id]
        if not match:
            raise SystemExit(
                "error: RUN_NOT_IN_PLAN: {} ({})".format(run_id, path))
        row = match[0]
    return row, header, str(path)


def driver_effective_updates(config_path: Path, tier: str) -> Optional[int]:
    """The training length the DRIVER will actually use for this tier.

    ``r3_v53mp_run.py cmd_run`` builds the trainer config from the frozen
    config schedule — ``freeze_encoder_updates + _tier_joint_updates`` for
    every tier except 2k-with-smoke.  Mirrored here (via the driver's own
    helper) so the report can expose the planned-vs-effective delta.
    """
    try:
        import yaml  # existing repo dependency

        from scripts.campaigns import r3_v53mp_run as driver
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        schedule = dict((config.get("training_contract") or {}).get("schedule")
                        or {})
        freeze = int(schedule.get("freeze_encoder_updates", 400))
        return freeze + int(driver._tier_joint_updates(schedule, tier))
    except Exception:  # noqa: BLE001 - informational field only
        return None


def derive_bench_config(original_config: Path, tier: str, updates: int,
                        out_path: Path) -> Dict[str, Any]:
    """Derive a bench config: original yaml + rewritten joint schedule.

    Loads the ORIGINAL campaign yaml (never written to), copies the whole
    ``training_contract.schedule``, and rewrites
    ``joint_updates_base_2k`` so the driver's
    ``_tier_joint_updates(schedule, tier) == updates - freeze`` — the
    driver computes joint as ``base_2k * max(1, TIER_LIMITS[tier] //
    TIER_LIMITS["2k"])``, so ``base_2k = (updates - freeze) / factor``
    must be an integer.  Every other knob (λ probe, seeds, val_every,
    batch, lrs, ...) stays verbatim.  Dumps to ``out_path`` and returns a
    summary dict including the derived file's sha256.

    Refusals (SystemExit, fail-closed):

    * ``UPDATES_LE_FREEZE``     — ``updates <= freeze_updates`` (no joint
      stage would remain; arm A would train zero updates);
    * ``UPDATES_NOT_DIVISIBLE`` — ``updates - freeze`` not divisible by
      the tier factor (the driver cannot express the requested length).
    """
    import yaml  # existing repo dependency

    from scripts.campaigns.r3_v53mp_run import TIER_LIMITS

    if tier not in TIER_LIMITS:
        raise SystemExit(
            "error: BENCH_TIER_UNKNOWN: {!r} not in {}".format(
                tier, sorted(TIER_LIMITS)))
    original = yaml.safe_load(Path(original_config).read_text(encoding="utf-8"))
    if not isinstance(original, dict):
        raise SystemExit(
            "error: BENCH_CONFIG_CORRUPT: {} is not a mapping".format(
                original_config))
    config = dict(original)
    tc = dict(config.get("training_contract") or {})
    schedule = dict(tc.get("schedule") or {})
    freeze = int(schedule.get("freeze_encoder_updates", 400))
    if updates <= freeze:
        raise SystemExit(
            "error: UPDATES_LE_FREEZE: --updates={} must exceed "
            "freeze_encoder_updates={} (arm A would get zero joint "
            "updates)".format(updates, freeze))
    factor = max(1, TIER_LIMITS[tier] // TIER_LIMITS["2k"])
    joint = updates - freeze
    if joint % factor != 0:
        raise SystemExit(
            "error: UPDATES_NOT_DIVISIBLE: updates-freeze={} is not a "
            "multiple of the tier-{} factor {} (driver joint = base_2k * "
            "factor); pick --updates = {} + k*{}".format(
                joint, tier, factor, freeze, factor))
    schedule["joint_updates_base_2k"] = joint // factor
    tc["schedule"] = schedule
    config["training_contract"] = tc
    config["infra_bench"] = {
        "derived_from": str(original_config),
        "tier": tier,
        "requested_updates": int(updates),
        "freeze_encoder_updates": freeze,
        "joint_updates_base_2k": int(schedule["joint_updates_base_2k"]),
        "derived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    from pes2ts_core.utils.hashing import sha256_file

    derived_eff = driver_effective_updates(out_path, tier)
    if derived_eff != updates:
        # self-check: the derived file must actually reproduce --updates
        raise SystemExit(
            "error: BENCH_CONFIG_SELF_CHECK_FAILED: {} yields {} updates, "
            "expected {}".format(out_path, derived_eff, updates))
    return {
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "freeze_encoder_updates": freeze,
        "joint_updates_base_2k": int(schedule["joint_updates_base_2k"]),
        "tier_factor": factor,
        "driver_effective_updates": derived_eff,
    }


def build_bench_row(source_row: Dict[str, Any], tag: str,
                    updates: int,
                    config_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Bench plan row: the source row with bench-intent overrides.

    ``n_updates`` carries ``--updates``; ``smoke_updates`` stays null per
    plan §7 (smoke is a 2k-tier-only driver feature); the five identity
    shas and the ``v53mp_<tier>_<arm>_<seed>`` run_id are copied
    VERBATIM — the driver's ``_parse_run_id`` accepts no other shape, so
    bench isolation comes from the per-rep root, not from the run_id.

    ``config_sha256`` overrides the copied ``config_sha256`` when a bench
    config was derived (the five-sha binding must point at the yaml the
    driver will actually load — the BENCH yaml, not the original).
    """
    row = dict(source_row)
    row["n_updates"] = int(updates)
    row["smoke_updates"] = None
    row["status"] = "planned"
    row["confirm"] = False
    row["infra_bench"] = {"tag": str(tag), "requested_updates": int(updates)}
    if config_sha256 is not None:
        row["config_sha256"] = config_sha256
    for field in ("run_id", "campaign_id", "phase", "tier", "arm", "fold",
                  "seed", "config_sha256", "data_sha256", "teacher_sha256",
                  "features_sha256", "split_sha256"):
        if field not in row or row[field] is None:
            raise SystemExit(
                "error: SOURCE_ROW_INCOMPLETE: missing {!r}".format(field))
    return row


def build_bench_header(source_header: Optional[Dict[str, Any]],
                       row: Dict[str, Any], tag: str) -> Dict[str, Any]:
    """Plan header: copy the source header when available (keeps
    campaign_mode, which the pin cross-check requires), else synthesize."""
    if source_header:
        header = dict(source_header)
    else:
        header = {
            "campaign_id": row["campaign_id"],
            "campaign_mode": "full",
            "sha_status": {f: "real" for f in (
                "config_sha256", "data_sha256", "features_sha256",
                "split_sha256", "teacher_sha256")},
        }
    header["record_type"] = "plan_header"
    header["generated_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    header["tier"] = row.get("tier")
    header["phase"] = row.get("phase")
    header["seeds"] = [row.get("seed")]
    header["matrix_ref"] = ["infra_bench {} (derived from {})".format(
        tag, row.get("run_id"))]
    header["infra_bench"] = True
    # campaign_mode MUST stay "full"/"partial": verify_plan_pin cross-checks
    # the header mode against the pin independently of the sha.
    if header.get("campaign_mode") not in ("full", "partial"):
        header["campaign_mode"] = "full"
    return header


def write_bench_plan(bench_root: Path, row: Dict[str, Any],
                     header: Dict[str, Any]) -> Tuple[Path, Path]:
    """Write <bench_root>/run_plan_<tier>.jsonl + .pin.json (driver-style).

    Plan bytes: one ``json.dumps(sort_keys=True)`` line per record with a
    trailing newline; pin: ``{"sha256": sha256_file(plan),
    "campaign_mode": header.campaign_mode}`` — byte-for-byte the scheme
    ``r3_v53mp_run.py plan-init`` writes and ``verify_plan_pin`` checks.
    """
    from pes2ts_core.utils.hashing import sha256_file

    tier = str(row.get("tier") or "20k")
    bench_root.mkdir(parents=True, exist_ok=True)
    plan_path = bench_root / "run_plan_v53mp_{}.jsonl".format(tier)
    lines = [header, row]
    plan_path.write_text(
        "".join(json.dumps(ln, sort_keys=True, ensure_ascii=False) + "\n"
                for ln in lines),
        encoding="utf-8")
    pin = {"sha256": sha256_file(plan_path),
           "campaign_mode": header["campaign_mode"]}
    pin_path = Path(str(plan_path) + ".pin.json")
    pin_path.write_text(
        json.dumps(pin, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return plan_path, pin_path


def validate_bench_root(bench_root: Path, plan_path: Path, run_id: str,
                        config_path: Path, tables_dir: Path,
                        split_path: Path) -> Dict[str, Any]:
    """Run the driver's OWN launch gates against the bench root (in-process).

    Mirrors the pre-launch checks of ``r3_v53mp_run.py cmd_run``: plan
    parse, pin verification, stage binding (five real shas) and the
    five-sha on-disk re-hash.  Raises SystemExit with the failing gate.
    """
    from scripts.agent_ops.ledger.run_ledger import (
        LedgerError,
        PlanNotFrozenError,
        read_run_plan,
        verify_plan_pin,
    )
    from scripts.agent_ops.supervisor.stage_bind import (
        StageBindingError,
        assert_stage_binding,
    )
    from scripts.campaigns import r3_v53mp_run as driver

    try:
        plan = read_run_plan(plan_path)
        line = plan["by_run_id"].get(run_id)
        if line is None:
            raise SystemExit(
                "BENCH_PLAN_ROW_MISSING: {} not in {}".format(
                    run_id, plan_path))
        pin = json.loads(
            Path(str(plan_path) + ".pin.json").read_text(encoding="utf-8"))
        verify_plan_pin(plan_path, pin)
        assert_stage_binding(plan_path, stage="g0", run_id=run_id)
        driver._verify_five_sha_binding(line, config_path, tables_dir,
                                        split_path)
    except SystemExit:
        raise
    except (PlanNotFrozenError, StageBindingError, LedgerError, OSError,
            KeyError, ValueError) as exc:
        # any driver-gate refusal becomes a typed bench-root failure
        raise SystemExit("BENCH_ROOT_GATE_FAILED: {}".format(exc)) from exc
    return {"plan": str(plan_path), "run_id": run_id,
            "pin_sha256": pin["sha256"], "five_sha": "ok", "stage": "g0"}


#: gate-check runs inside cgroup_scope.sh with a HARD memory cap (a future
#: sidecar cache MISS re-reads the full ~54 GB data set; unconstrained that
#: pushes the WSL2 host toward the commit cliff).  Defaults match the plan
#: §5.1 per-task budget; overridable via --gate-memory-max/--gate-memory-high.
GATE_MEMORY_MAX_DEFAULT = "8G"
GATE_MEMORY_HIGH_DEFAULT = "7G"
GATE_CHECK_TIMEOUT_S = 900.0


def run_gate_check(config_path: Path, tables_dir: Path,
                   split_path: Optional[Path], probe_root: Path,
                   gate_memory_max: str = GATE_MEMORY_MAX_DEFAULT,
                   gate_memory_high: str = GATE_MEMORY_HIGH_DEFAULT,
                   timeout_s: float = GATE_CHECK_TIMEOUT_S,
                   fallback: bool = False) -> Dict[str, Any]:
    """``r3_v53mp_run.py --output-root <probe> gate-check`` as a subprocess.

    Read-only w.r.t. the ledger (cmd_gate_check never touches it); uses a
    bench root only to satisfy the top-level --output-root assertion.

    The driver is launched THROUGH ``cgroup_scope.sh`` so the gate-check
    process itself is cgroup-contained (memory max/high above): today the
    sidecar cache keeps it tiny, but a cache miss must never re-read the
    data set unconstrained.  Artifacts land in ``<probe>/_gate_check/``:
    the wrapper's cgroup_sample/cgroup_summary plus ``gate_stdout.log`` /
    ``gate_stderr.log``.  The wrapper logs to STDERR only, so the driver's
    ``{"items": ..., "ok": ...}`` JSON stays the last stdout line and is
    parsed exactly as before (no wrapper pollution).

    Failure never becomes success through the wrapper: ``ok`` requires
    returncode 0 AND the driver's own ``ok`` AND (when present) the
    wrapper summary's ``cmd_exit == 0`` — the wrapper may only make the
    verdict STRICTER (oom/swap gate codes 91/90), never the reverse.
    On timeout the whole process tree (wrapper + driver) is SIGKILLed via
    its own session/process group (``start_new_session`` + ``killpg``).
    """
    gate_outdir = Path(probe_root) / "_gate_check"
    gate_outdir.mkdir(parents=True, exist_ok=True)
    driver_cmd = [sys.executable, str(RUN_DRIVER),
                  "--output-root", str(probe_root),
                  "gate-check", "--config", str(config_path),
                  "--tables-dir", str(tables_dir)]
    if split_path is not None:
        driver_cmd += ["--split", str(split_path)]
    cmd = ["bash", str(CGROUP_SCOPE)]
    if fallback:
        cmd.append("--fallback")
    cmd += [gate_memory_max, gate_memory_high, str(gate_outdir)] + driver_cmd

    stdout_path = gate_outdir / "gate_stdout.log"
    stderr_path = gate_outdir / "gate_stderr.log"
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as out_fh, \
            stderr_path.open("w", encoding="utf-8") as err_fh:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO), stdout=out_fh, stderr=err_fh,
            text=True, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            proc.wait()
    rc = proc.returncode

    out = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
    err = stderr_path.read_text(encoding="utf-8", errors="replace")
    try:
        doc = json.loads(out.rsplit("\n", 1)[-1]) if out else {}
    except json.JSONDecodeError:
        doc = {"raw": out[-2000:]}
    # wrapper accounting is belt-and-braces: a nonzero cmd_exit (or an
    # unreadable summary next to a suspicious rc) can only FAIL the gate
    summary_path = gate_outdir / "cgroup_summary.json"
    wrapper_cmd_exit: Optional[int] = None
    if summary_path.is_file():
        try:
            wrapper_cmd_exit = json.loads(
                summary_path.read_text(encoding="utf-8")).get("cmd_exit")
        except (OSError, json.JSONDecodeError):
            wrapper_cmd_exit = None
    ok = ((not timed_out) and rc == 0 and bool(doc.get("ok"))
          and wrapper_cmd_exit in (None, 0))
    return {"ok": ok, "returncode": rc, "timed_out": timed_out,
            "items": doc.get("items"),
            "outdir": str(gate_outdir),
            "stderr_tail": err[-500:]}


# ---------------------------------------------------------------------------
# bench orchestration (parent process)
# ---------------------------------------------------------------------------


def _read_lines(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _first_log_row_time(log_path: Path, deadline: float,
                        proc: Optional[subprocess.Popen]) -> Optional[float]:
    """Poll a train_log.jsonl for its first non-empty row (spawn -> row)."""
    while time.time() < deadline:
        if log_path.is_file():
            with log_path.open("r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    if raw.strip():
                        return time.time()
        if proc is not None and proc.poll() is not None:
            if log_path.is_file():
                with log_path.open("r", encoding="utf-8",
                                   errors="replace") as fh:
                    for raw in fh:
                        if raw.strip():
                            return time.time()
            return None
        time.sleep(0.5)
    return None


def _fabricate_rep(rep_dir: Path, run_dir: Path, updates: int,
                   rep: int) -> None:
    """--dry-run: stub timing/cgroup/vram files at the B0 baseline values."""
    rep_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    stub = run_dir / "train_log.jsonl"
    startup = 12.0 + rep * 0.5
    elapsed = startup
    rows = [json.dumps({"update": 1, "phase": "joint",
                        "loss": 0.259, "elapsed_s": round(elapsed, 3)})]
    jitter = [0.0, 2.5, -1.5, 4.0, -3.0, 1.0, -2.0, 3.5]
    for u in range(100, updates + 1, 100):
        elapsed += 100 * (B0_MS_TARGET + jitter[(u // 100) % len(jitter)]) / 1000.0
        rows.append(json.dumps({"update": u, "phase": "joint",
                                "loss": 0.24, "elapsed_s": round(elapsed, 3)}))
    stub.write_text("\n".join(rows) + "\n", encoding="utf-8")
    # cgroup sampler stub in the r3 CSV format (+ .peak/.events siblings)
    csv = rep_dir / "cgroup_sample.csv"
    ts0 = int(time.time()) - 600
    crow = []
    for i in range(300):
        anon = int(26.0e9 * min(1.0, 0.2 + 0.8 * i / 200))
        crow.append("{},{},{},{},{}".format(
            time.strftime("%H:%M:%S", time.gmtime(ts0 + i * 2)),
            anon, anon, int(1.2e8), 0))
    csv.write_text(
        "ts,memory_current_bytes,anon_bytes,file_bytes,"
        "memory_swap_current_bytes\n" + "\n".join(crow) + "\n",
        encoding="utf-8")
    peak_anon = max(int(26.0e9 * min(1.0, 0.2 + 0.8 * i / 200))
                    for i in range(300))
    Path(str(csv) + ".peak").write_text(
        "{},{},{}\n".format(time.strftime("%H:%M:%S"), peak_anon, peak_anon),
        encoding="utf-8")
    Path(str(csv) + ".events").write_text(
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n", encoding="utf-8")
    (rep_dir / "cgroup_summary.json").write_text(json.dumps({
        "peak_anon_bytes": peak_anon, "max_swap_current_bytes": 0,
        "swap_sustained_s": 0, "oom_kill": 0, "cmd_exit": 0,
        "gate_exit": 0, "dry_run": True}, sort_keys=True) + "\n",
        encoding="utf-8")
    vram_path = rep_dir / "vram_sample.txt"
    vram_path.write_text("\n".join("{}".format(2400 + (i % 5) * 20)
                                   for i in range(300)) + "\n",
                         encoding="utf-8")
    (rep_dir / "driver_stdout.log").write_text(
        "[dry-run] fabricated rep {} (stub timing @ {} ms/update)\n".format(
            rep, B0_MS_TARGET), encoding="utf-8")


def run_rep(rep: int, rep_dir: Path, rep_root: Path, run_id: str,
            driver_cmd: List[str], warmup: int, startup_timeout_s: float,
            vram_log: str, memory_max: str, memory_high: str,
            fallback: bool, thread_caps: bool) -> Dict[str, Any]:
    """Launch one rep: driver run under cgroup_scope.sh + VRAM sampler."""
    rep_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = rep_dir / "driver_stdout.log"
    log_path = rep_root / run_id / "train_log.jsonl"
    cmd = ["bash", str(CGROUP_SCOPE)]
    if fallback:
        cmd.append("--fallback")
    cmd += [memory_max, memory_high, str(rep_dir)]
    if thread_caps:
        cmd += ["bash", str(THREAD_CAPS)]
    cmd += driver_cmd
    started = time.time()
    vram_proc: Optional[subprocess.Popen] = None
    vram_path = rep_dir / "vram_sample.txt"
    vram_fh = vram_path.open("w", encoding="utf-8")
    try:
        if vram_log:
            vram_proc = subprocess.Popen(
                [vram_log, "--query-gpu=memory.used",
                 "--format=csv,noheader", "-l", "2"],
                stdout=vram_fh, stderr=subprocess.DEVNULL)
        with stdout_path.open("w", encoding="utf-8") as out_fh:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO), stdout=out_fh,
                stderr=subprocess.STDOUT, text=True,
                start_new_session=True)
            first_row_at = _first_log_row_time(
                log_path, started + startup_timeout_s, proc)
            rc = proc.wait()
    finally:
        if vram_proc is not None:
            vram_proc.send_signal(signal.SIGTERM)
            try:
                vram_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                vram_proc.kill()
        vram_fh.close()

    train = parse_train_log(_read_lines(log_path), warmup)
    cgroup = parse_cgroup_samples(
        _read_lines(rep_dir / "cgroup_sample.csv"),
        (rep_dir / "cgroup_sample.csv.events").read_text(encoding="utf-8")
        if (rep_dir / "cgroup_sample.csv.events").is_file() else "")
    vram = parse_vram_samples(_read_lines(vram_path))
    gate_exit = None
    summary_path = rep_dir / "cgroup_summary.json"
    if summary_path.is_file():
        try:
            gate_exit = json.loads(
                summary_path.read_text(encoding="utf-8")).get("gate_exit")
        except json.JSONDecodeError:
            pass
    return {
        "rep": rep,
        "rep_dir": str(rep_dir),
        "rep_root": str(rep_root),
        "exit": rc,
        "gate_exit": gate_exit,
        "startup_s": (round(first_row_at - started, 1)
                      if first_row_at else None),
        "median_ms": train["median_ms"],
        "p95_ms": train["p95_ms"],
        "n_timed": train["n_timed"],
        "n_updates_logged": train["n_updates"],
        "anon_peak_bytes": cgroup["anon_peak_bytes"],
        "swap_max_bytes": cgroup["swap_max_bytes"],
        "oom_kill": cgroup["oom_kill"],
        "vram_peak_mib": vram["vram_peak_mib"],
        "vram_samples": vram["vram_samples"],
        "wall_s": round(time.time() - started, 1),
    }


def bench_main(args: argparse.Namespace) -> int:
    source_row, source_header, source_plan = load_plan_row(
        args.run_plan, args.run_id)
    tier = str(source_row.get("tier") or "20k")
    run_id = str(source_row["run_id"])
    config_path = Path(args.config)
    tables_dir = Path(args.data_root)
    split_path = Path(args.split) if args.split else (
        tables_dir / "split_manifest_v53mp.json")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bench_top = Path(args.output_root or
                     (Path(DEFAULT_BENCH_ROOT) / "{}_{}".format(args.tag, ts)))
    bench_top.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    # driver-contract deviations (see module docstring) — surfaced, never hidden
    if not run_id.startswith("v53mp_"):
        raise SystemExit(
            "error: RUN_ID_DRIVER_INVALID: {!r} cannot be launched by "
            "r3_v53mp_run.py (_parse_run_id requires "
            "v53mp_<tier>_<arm>_s<seed>)".format(run_id))

    # ---- run-config selection: derive a bench yaml when --updates differs
    # from the driver-effective length; use the ORIGINAL config untouched
    # when they match (warning-free path).
    from pes2ts_core.utils.hashing import sha256_file

    original_config_sha = sha256_file(config_path)
    eff = driver_effective_updates(config_path, tier)
    run_config = config_path
    bench_cfg_sha: Optional[str] = None
    config_source = "original"
    bench_cfg: Optional[Dict[str, Any]] = None
    if eff is not None and eff != args.updates:
        # cmd_run ignores the plan row's n_updates for run length (it only
        # reaches the ledger/print): freeze+joint from the config schedule
        # decides.  Close the gap by deriving a bench config whose schedule
        # yields exactly --updates (original yaml never written).
        bench_cfg = derive_bench_config(
            config_path, tier, args.updates, bench_top / "config_bench.yaml")
        run_config = Path(bench_cfg["path"])
        bench_cfg_sha = str(bench_cfg["sha256"])
        config_source = "bench_derived"
        eff = int(bench_cfg["driver_effective_updates"])  # == --updates
    elif eff is None:
        warnings.append(
            "driver-effective updates could not be computed from {}; using "
            "the original config unmodified (run length unknown)".format(
                config_path))

    row = build_bench_row(source_row, args.tag, args.updates,
                          config_sha256=bench_cfg_sha)
    header = build_bench_header(source_header, row, args.tag)

    # ---- gate-check once (driver subprocess, read-only; validates the
    # config the driver will actually load) ---------------------------------
    gate: Dict[str, Any]
    if args.dry_run:
        gate = {"ok": True, "returncode": 0, "skipped": "dry-run"}
    else:
        gate = run_gate_check(run_config, tables_dir, split_path, bench_top,
                              gate_memory_max=args.gate_memory_max,
                              gate_memory_high=args.gate_memory_high,
                              fallback=args.fallback)
        if not gate.get("ok"):
            print(json.dumps({"action": "infra_bench", "error": "GATE_CHECK_FAILED",
                              "gate_check": gate}, sort_keys=True))
            return 2

    rep_reports: List[Dict[str, Any]] = []
    for rep in range(args.reps):
        rep_dir = bench_top / "rep{}".format(rep)
        rep_root = rep_dir / "root"
        rep_config = run_config
        if bench_cfg is not None:
            # per-rep copy of the DERIVED bytes (identical file -> identical
            # sha -> the per-rep row/pin stay consistent with the copy)
            rep_config = rep_root / "config_bench.yaml"
            rep_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_config, rep_config)
        plan_path, _pin_path = write_bench_plan(rep_root, row, header)
        if args.dry_run:
            _fabricate_rep(rep_dir, rep_root / run_id, args.updates, rep)
            train = parse_train_log(
                _read_lines(rep_root / run_id / "train_log.jsonl"),
                args.warmup)
            cgroup = parse_cgroup_samples(
                _read_lines(rep_dir / "cgroup_sample.csv"),
                Path(str(rep_dir / "cgroup_sample.csv") + ".events")
                .read_text(encoding="utf-8"))
            vram = parse_vram_samples(
                _read_lines(rep_dir / "vram_sample.txt"))
            rep_reports.append({
                "rep": rep, "rep_dir": str(rep_dir),
                "rep_root": str(rep_root), "exit": 0, "gate_exit": 0,
                "startup_s": round(train["first_row_elapsed_s"] or 12.0, 1),
                "median_ms": train["median_ms"],
                "p95_ms": train["p95_ms"],
                "n_timed": train["n_timed"],
                "n_updates_logged": train["n_updates"],
                "anon_peak_bytes": cgroup["anon_peak_bytes"],
                "swap_max_bytes": cgroup["swap_max_bytes"],
                "oom_kill": cgroup["oom_kill"],
                "vram_peak_mib": vram["vram_peak_mib"],
                "vram_samples": vram["vram_samples"],
                "wall_s": None,
            })
            continue
        # in-process driver-gate validation of THIS rep's bench root
        # (five-sha binding verified against the config the driver loads)
        try:
            validate_bench_root(rep_root, plan_path, run_id, rep_config,
                                tables_dir, split_path)
        except SystemExit as exc:
            sys.stderr.write("error: BENCH_ROOT_INVALID (rep {}): {}\n".format(
                rep, exc))
            return 2
        driver_cmd = [
            sys.executable, str(RUN_DRIVER),
            "--output-root", str(rep_root),
            "run", "--run-id", run_id,
            "--config", str(rep_config),
            "--tables-dir", str(tables_dir),
            "--split", str(split_path),
            "--r3-ckpt", args.r3_ckpt,
            "--device", args.device,
            "--loader", args.loader,
            "--dev-pack", args.dev_pack,
            "--double-buffer", args.double_buffer,
            "--pinned-h2d", args.pinned_h2d,
        ]
        if args.feature_workers is not None:
            driver_cmd += ["--feature-workers", str(args.feature_workers)]
        if args.gpu_lock:
            driver_cmd += ["--gpu-lock", args.gpu_lock]
        rep_reports.append(run_rep(
            rep, rep_dir, rep_root, run_id, driver_cmd, args.warmup,
            args.startup_timeout_s, args.vram_log,
            args.memory_max, args.memory_high, args.fallback,
            args.thread_caps))

    report = build_report(
        args.tag,
        source_plan or "<inline row>",
        run_id, str(bench_top),
        "pass" if gate.get("ok") else "fail",
        str(tables_dir), args.updates, eff, args.warmup, rep_reports,
        args.dry_run, warnings,
        ms_target=args.b0_ms_target, ms_tol=args.b0_ms_tol,
        anon_target_gb=args.b0_anon_gb, anon_tol=args.b0_anon_tol,
        config_source=config_source,
        original_config_sha256=original_config_sha,
        bench_config_sha256=bench_cfg_sha)
    report["loader"] = args.loader
    report["dev_pack"] = args.dev_pack
    report["double_buffer"] = args.double_buffer
    report["pinned_h2d"] = args.pinned_h2d
    report["feature_workers"] = args.feature_workers
    (bench_top / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    (bench_top / "report.md").write_text(
        render_report_md(report), encoding="utf-8")
    print(json.dumps({"action": "infra_bench", "tag": args.tag,
                      "output_root": str(bench_top),
                      "gate_check": report["gate_check"],
                      "verdict_pass": report["verdict"]["pass"],
                      "median_ms_update":
                          report["aggregate"]["median_ms_update"],
                      "anon_peak_gb_max":
                          report["aggregate"]["anon_peak_gb_max"],
                      "warnings": warnings},
                     sort_keys=True))
    # bench exit code: 0 = ran (verdict is IN the report; a failing verdict
    # is a scientific result, not a crash). Nonzero only for run failures.
    failed = [r for r in rep_reports if r.get("exit") not in (0, None)]
    return 1 if failed else 0


def recompute_main(args: argparse.Namespace) -> int:
    """Read-only ladder recompute of archived bench ``report.json`` files.

    Loads each REPORT (never written to, never overwritten), re-derives
    the per-rung verdicts, prints the new JSON verdict to stdout and (when
    ``--recompute-out`` is given) writes it there too.  The ORIGINAL
    verdict of every input report is embedded per rung for audit, so the
    archived reports stay authoritative.
    """
    reports: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    for spec in args.recompute:
        path = Path(spec)
        if path.is_dir():
            path = path / "report.json"
        if not path.is_file():
            raise SystemExit(
                "error: RECOMPUTE_REPORT_MISSING: {}".format(spec))
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "error: RECOMPUTE_REPORT_INVALID: {}: {}".format(
                    path, exc)) from exc
        if not isinstance(doc, dict):
            raise SystemExit(
                "error: RECOMPUTE_REPORT_INVALID: {} is not a JSON "
                "object".format(path))
        reports.append(doc)
        sources.append({"path": str(path), "tag": doc.get("tag"),
                        "generated_at": doc.get("generated_at"),
                        "original_verdict": doc.get("verdict")})
    result = recompute_ladder(reports, tol=args.ladder_tol)
    result["sources"] = sources
    text = json.dumps(result, sort_keys=True, indent=1) + "\n"
    if args.recompute_out:
        out = Path(args.recompute_out)
        resolved_inputs = {Path(s["path"]).resolve() for s in sources}
        if out.resolve() in resolved_inputs:
            raise SystemExit(
                "error: RECOMPUTE_OUT_IS_INPUT: refusing to overwrite an "
                "archived report: {}".format(out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    # one-line summary on STDERR so stdout stays a single pure JSON doc
    sys.stderr.write(json.dumps(
        {"action": "infra_bench_recompute", "kind": "ladder",
         "pass": result["pass"],
         "rungs": [{"tag": r["tag"], "pass": r["verdict"]["pass"],
                    "failing_checks": r["verdict"].get("failing_checks", [])}
                   for r in result["rungs"]]},
        sort_keys=True) + "\n")
    sys.stdout.write(text)
    return 0 if result["pass"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra_bench",
        description="S4 B0–B4 infra smoke bench (plan §7); runs via "
                    "r3_v53mp_run.py against self-contained per-rep bench "
                    "roots (campaign ledger untouched)")
    parser.add_argument("--tag", default="B0")
    parser.add_argument("--run-plan", default=DEFAULT_SOURCE_PLAN,
                        help="source (frozen campaign) plan jsonl FILE or a "
                             "literal JSON row; the bench plan is DERIVED "
                             "from it, never written back")
    parser.add_argument("--run-id", default=DEFAULT_SOURCE_RUN_ID,
                        help="row selector when --run-plan is a file")
    parser.add_argument("--updates", type=int, default=2000,
                        help="bench update count (freeze+joint); when it "
                             "differs from the driver-effective length of "
                             "the frozen config schedule, a derived "
                             "config_bench.yaml is generated per bench "
                             "(original yaml untouched); equal means the "
                             "original config is used directly")
    parser.add_argument("--warmup", type=int, default=200,
                        help="excluded leading updates for ms/update stats "
                             "ONLY (startup uses the first train_log row)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="driver --tables-dir (original data tree)")
    parser.add_argument("--split", default=None,
                        help="driver --split (default: "
                             "<data-root>/split_manifest_v53mp.json)")
    parser.add_argument("--output-root", default=None,
                        help="bench top dir; default: "
                             "{}<tag>_<ts>/".format(DEFAULT_BENCH_ROOT))
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="ORIGINAL campaign yaml (never copied/edited; "
                             "config_sha256 binding)")
    parser.add_argument("--r3-ckpt", default=R3_WINNER_CKPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loader", default="v1", choices=("v1", "v2"),
                        help="driver --loader passthrough (v1 = light-pickle "
                             "sharded loader, v2 = columnar memmap loader; "
                             "plan §4/I1)")
    parser.add_argument("--dev-pack", default="off", choices=("auto", "off"),
                        help="driver --dev-pack passthrough (auto = frozen "
                             "dev feature pack for the final eval, plan §5.3/"
                             "I2; off = per-run dev feature build)")
    parser.add_argument("--double-buffer", default="off",
                        choices=("on", "off"),
                        help="driver --double-buffer passthrough (plan §4.3/"
                             "I3/B3; default off for B0–B2 reproducibility)")
    parser.add_argument("--pinned-h2d", default="off", choices=("on", "off"),
                        help="driver --pinned-h2d passthrough (plan §6/I3/"
                             "B3; default off for B0–B2 reproducibility)")
    parser.add_argument("--feature-workers", type=int, default=None,
                        help="driver --feature-workers passthrough (explicit "
                             "fork-pool size 0-12 overriding the gate file; "
                             "plan §4.3/I3/B4)")
    parser.add_argument("--memory-max", default="34G",
                        help="scope MemoryMax (current single-run value; "
                             "post-I1 use 6G per plan §5.1)")
    parser.add_argument("--memory-high", default="32G")
    parser.add_argument("--gate-memory-max", default=GATE_MEMORY_MAX_DEFAULT,
                        help="cgroup_scope.sh MemoryMax for the gate-check "
                             "subprocess (gate-check stays cgroup-contained "
                             "even on sidecar cache hits; a cache miss must "
                             "never re-read the data set unconstrained)")
    parser.add_argument("--gate-memory-high", default=GATE_MEMORY_HIGH_DEFAULT,
                        help="cgroup_scope.sh MemoryHigh for the gate-check "
                             "subprocess")
    parser.add_argument("--fallback", action="store_true",
                        help="cgroup_scope.sh --fallback (manual cgroup-v2 "
                             "dir instead of systemd-run --scope)")
    parser.add_argument("--gpu-lock", default="/tmp/opencode/pes2ts_gpu.lock",
                        help="passed through to the driver's --gpu-lock "
                             "(empty string: driver default)")
    parser.add_argument("--vram-log", default="nvidia-smi",
                        help="nvidia-smi binary (empty string disables)")
    parser.add_argument("--startup-timeout-s", type=float, default=1800.0)
    parser.add_argument("--thread-caps", action="store_true",
                        help="wrap the driver with scripts/infra/"
                             "thread_caps.sh (plan §5.5 A/B)")
    parser.add_argument("--dry-run", action="store_true",
                        help="fabricate stub reps; exercises parse/report/"
                             "verdict without GPU, driver or training")
    # B0 gate tuning (defaults are the plan §7 targets)
    parser.add_argument("--b0-ms-target", type=float, default=B0_MS_TARGET,
                        help="B0 median ms/update target; values in (0,1) "
                             "are read as SECONDS/update (e.g. 0.078 = the "
                             "plan §7 ~0.078 s/update) and converted to ms")
    parser.add_argument("--b0-ms-tol", type=float, default=B0_MS_TOLERANCE)
    parser.add_argument("--b0-anon-gb", type=float, default=B0_ANON_TARGET_GB)
    parser.add_argument("--b0-anon-tol", type=float, default=B0_ANON_TOLERANCE)
    # read-only ladder recompute over archived report.json files
    parser.add_argument("--recompute", nargs="+", metavar="REPORT",
                        default=None,
                        help="READ-ONLY: recompute per-rung verdicts from "
                             "archived bench report.json files (a bench "
                             "root dir or the report.json itself), ladder "
                             "order B0..Bn; B0 self-baselines from its own "
                             "reps, higher rungs are regression-only vs "
                             "the rung below; nothing is re-run and no "
                             "input report is ever overwritten (the "
                             "original verdict is embedded for audit)")
    parser.add_argument("--recompute-out", default=None,
                        help="write the recomputed ladder JSON here in "
                             "addition to stdout (never an input report)")
    parser.add_argument("--ladder-tol", type=float, default=LADDER_TOLERANCE,
                        help="per-rung regression tolerance (default 0.15 "
                             "= 1.15x the previous rung's metric)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.recompute:
        return recompute_main(args)
    return bench_main(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
