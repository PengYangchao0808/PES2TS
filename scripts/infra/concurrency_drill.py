#!/usr/bin/env python3
"""S4 B4 concurrency drill (plan §7 / lane I3).

Runs the plan §7-B4 drill: N in {1,3,4} CONCURRENT short runs (2000
updates) against the frozen campaign plan, each in its own bench root
(self-contained plan+pin+derived config — the ``infra_bench`` mechanism,
imported and REUSED, never copied), each with:

* ``--loader v2 --dev-pack auto`` (the I1/I2 data path),
* a per-run cgroup scope (``cgroup_scope.sh 6G 5.4G <slot-outdir>`` — one
  runaway run kills only its own scope, plan §5.1),
* per-slot feature workers ``workers_for_concurrency(N)`` (plan §4.3),
* distinct arms/seeds per slot taken from the frozen plan rows in order
  (A_s17, B_s29, C_s43, ... — never two slots on one (arm, seed)),
* a per-slot gpu lock (the driver's exclusive flock), plus ONE
  drill-level lock the drill holds for its whole duration (the drill
  owns the GPU window; a concurrent campaign/orchestrator run is refused
  up front rather than deadlocking at the flock).

Every phase also runs ONE whole-card VRAM sampler for its full duration
(``nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 2`` →
``<phase_root>/vram_samples.txt``, reliably terminate/wait-stopped in a
``finally`` even when the phase raises), parsed with
``infra_bench.parse_vram_samples`` (REUSED, never copied).

Measured per N: per-run wall time, per-run anon peak (from the cgroup
sampler CSVs), total anon sum, oom_kill count, whole-card VRAM peak.
Verdict (plan §7-B4):

* ``oom_kill == 0`` (every N),
* per-run wall ``<= 1.5 x`` the N=1 baseline (N > 1),
* N=3 total anon ``< 24 GB`` (the plan's ``3 x anon < 24 GB`` gate; the
  N=4 total is reported informationally — the plan gates 3-way only),
* N=3 whole-card VRAM peak ``<= 6800 MiB``
  (``whole_card_vram_le_6800_mib`` — ENFORCED at N=3 only; N=1 records
  it as a reference and N=4 reports it, neither gates.  A missing /
  empty sample file parses to 0 and is FAIL-CLOSED for N=3).

``--cuda-memory-fraction f`` (0 < f <= 1) is forwarded to every driver
argv as ``--cuda-memory-fraction f``; default off (flag absent).
``--eval-vram-gate-mib m`` (``>= 1``) is forwarded to every driver argv
as ``--eval-vram-gate-mib m --eval-vram-gate-timeout-s t`` (3-way
staggered evals; ``--eval-vram-gate-timeout-s`` defaults to 600 and is
forwarded only when the gate is set); default off (flags absent).

Phases run SEQUENTIALLY (N=1 first — it is the baseline — then 3, then
4); the N runs inside one phase are concurrent.  ``--dry-run`` builds
and validates the FULL plan (bench roots, derived configs, plans + pins,
five-sha binding, cgroup wrapper contract, worker counts, slot locks)
without launching anything and without touching the GPU.

Report: ``<out-root>/report.json`` + ``report.md``.  Exit code 0 iff
every phase ran and every gate verdict passed; nonzero otherwise.

GPU discipline: the drill refuses to start (exit 2) when another process
holds the drill lock — the orchestrator runs the drill when the campaign
window is free.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.infra import infra_bench as ib  # noqa: E402  (REUSE, not copy)
from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (  # noqa: E402
    workers_for_concurrency,
)

DRILL_LOCK = "/tmp/opencode/pes2ts_gpu.drill.lock"
SLOT_LOCK_TEMPLATE = "/tmp/opencode/pes2ts_gpu.drillslot{k}.lock"
CGROUP_SCOPE = PROJECT_ROOT / "scripts/infra/cgroup_scope.sh"
THREAD_CAPS = PROJECT_ROOT / "scripts/infra/thread_caps.sh"
DEFAULT_N_VALUES = (1, 3, 4)
DEFAULT_UPDATES = 2000
DEFAULT_MEMORY_MAX = "6G"
#: B4 gate (plan §7): 3 concurrent runs' combined anon < 24 GB.
B3_ANON_SUM_GATE_BYTES = 24_000_000_000
#: B4 gate: per-run wall <= 1.5x the N=1 baseline.
B4_WALL_RATIO_GATE = 1.5
#: B4 gate (plan §7): whole-card VRAM peak <= 6800 MiB at N=3
#: (ENFORCED at N=3 only; N=1/N=4 record it informationally).
WHOLE_CARD_VRAM_GATE_MIB = 6800
#: per-phase sampler output (phase_root/<this file>).
VRAM_SAMPLE_FILENAME = "vram_samples.txt"
VRAM_SAMPLE_CMD = ["nvidia-smi", "--query-gpu=memory.used",
                   "--format=csv,noheader", "-l", "2"]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _cuda_memory_fraction_arg(value: Any) -> float:
    """argparse type: 0 < f <= 1 (fail-closed on junk / 0 / negatives)."""
    try:
        frac = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "cuda-memory-fraction must be a number in (0, 1], got {!r}"
            .format(value))
    if not (0.0 < frac <= 1.0):
        raise argparse.ArgumentTypeError(
            "cuda-memory-fraction must satisfy 0 < f <= 1, got {!r}"
            .format(value))
    return frac


def _eval_vram_gate_mib_arg(value: Any) -> int:
    """argparse type: whole-card eval VRAM gate threshold in MiB (>= 1)."""
    try:
        mib = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "eval-vram-gate-mib must be an integer >= 1, got {!r}".format(
                value))
    if mib < 1:
        raise argparse.ArgumentTypeError(
            "eval-vram-gate-mib must be >= 1, got {!r}".format(value))
    return mib


def _eval_vram_gate_timeout_arg(value: Any) -> int:
    """argparse type: eval VRAM gate wait timeout in seconds (>= 0)."""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "eval-vram-gate-timeout-s must be an integer >= 0, got {!r}"
            .format(value))
    if seconds < 0:
        raise argparse.ArgumentTypeError(
            "eval-vram-gate-timeout-s must be >= 0, got {!r}".format(value))
    return seconds


# ---------------------------------------------------------------------------
# whole-card VRAM sampler (one per N phase; infra_bench mechanism, REUSED)
# ---------------------------------------------------------------------------
def _start_vram_sampler(phase_root: Path
                         ) -> Optional[Tuple[subprocess.Popen, IO[str]]]:
    """Start the whole-card ``nvidia-smi -l 2`` sampler for one phase.

    Output goes to ``phase_root/vram_samples.txt``, which is always
    created (an empty/absent log parses to peak 0 → the N=3 whole-card
    gate fails CLOSED).  Returns ``(proc, fh)`` or ``None`` when the
    sampler cannot start (nvidia-smi missing/unusable) — the caller
    still stops/cleans up via :func:`_stop_vram_sampler`.
    """
    phase_root.mkdir(parents=True, exist_ok=True)
    vram_path = phase_root / VRAM_SAMPLE_FILENAME
    fh = vram_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(VRAM_SAMPLE_CMD, stdout=fh,
                                stderr=subprocess.DEVNULL)
    except OSError as exc:
        fh.close()                            # file left in place, EMPTY
        _log("[vram] sampler unavailable ({}): {} stays empty — the N=3 "
             "whole-card gate fails closed".format(exc, vram_path))
        return None
    return proc, fh


def _stop_vram_sampler(handle: Optional[Tuple[subprocess.Popen, IO[str]]]
                        ) -> None:
    """Reliable stop: terminate → wait (kill on timeout) → always close."""
    if handle is None:
        return
    proc, fh = handle
    try:
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:     # pragma: no cover
            try:
                proc.kill()
                proc.wait(timeout=5)
            except OSError:
                pass
    finally:
        try:
            fh.close()
        except OSError:
            pass


def _parse_memory_high(memory_max: str) -> str:
    """MemoryHigh = 0.9 x MemoryMax (systemd size suffix preserved)."""
    s = str(memory_max).strip()
    num, suffix = s[:-1], s[-1:]
    value = float(num)
    high = value * 0.9
    if abs(high - round(high)) < 1e-9:
        high_s = str(int(round(high)))
    else:
        high_s = ("{:.2f}".format(high)).rstrip("0").rstrip(".")
    return high_s + suffix


# ---------------------------------------------------------------------------
# phase plan construction (pure; validated in --dry-run)
# ---------------------------------------------------------------------------
def _select_rows(plan_rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """First N plan rows with DISTINCT (arm, seed) — one per drill slot."""
    seen: set = set()
    picked: List[Dict[str, Any]] = []
    for row in plan_rows:
        key = (str(row.get("arm")), int(row.get("seed", -1)))
        if key in seen:
            continue
        seen.add(key)
        picked.append(row)
        if len(picked) == n:
            return picked
    raise SystemExit(
        "error: PLAN_ROWS_EXHAUSTED: need {} distinct (arm, seed) rows, "
        "found {}".format(n, len(picked)))


def build_phase_plan(
    source_row: Dict[str, Any], tag: str, updates: int,
    n_slots: int, phase_root: Path, tables_dir: Path, split_path: Path,
    config_path: Path, r3_ckpt: Path, device: str,
    memory_max: str, feature_workers: int, slot_index: int = 0,
    thread_caps: bool = False,
    cuda_memory_fraction: Optional[float] = None,
    eval_vram_gate_mib: Optional[int] = None,
    eval_vram_gate_timeout_s: int = 600,
) -> Dict[str, Any]:
    """Build + validate one phase's N slot roots (plan/config/pin/binding).

    Returns a per-slot descriptor list.  Fail-closed: any driver-gate
    refusal (SystemExit from ``validate_bench_root``) propagates, as does
    an invalid ``cuda_memory_fraction`` (must satisfy 0 < f <= 1) or
    ``eval_vram_gate_mib`` (must be >= 1 when set).
    """
    high = _parse_memory_high(memory_max)
    if cuda_memory_fraction is not None:
        try:
            cuda_memory_fraction = _cuda_memory_fraction_arg(
                cuda_memory_fraction)
        except argparse.ArgumentTypeError as exc:
            raise SystemExit(
                "error: BAD_CUDA_MEMORY_FRACTION: {}".format(exc))
    if eval_vram_gate_mib is not None:
        try:
            eval_vram_gate_mib = _eval_vram_gate_mib_arg(eval_vram_gate_mib)
            eval_vram_gate_timeout_s = _eval_vram_gate_timeout_arg(
                eval_vram_gate_timeout_s)
        except argparse.ArgumentTypeError as exc:
            raise SystemExit(
                "error: BAD_EVAL_VRAM_GATE: {}".format(exc))
    slots: List[Dict[str, Any]] = []
    from pes2ts_core.utils.hashing import sha256_file

    bench_cfg = ib.derive_bench_config(
        config_path, str(source_row.get("tier") or "20k"), updates,
        phase_root / "config_bench.yaml")
    cfg_path = Path(bench_cfg["path"])
    for slot in range(n_slots):
        if n_slots == 1:
            # caller builds ONE slot root per call; the loop counter is
            # meaningless — use the real slot id (distinct gpu lock + unit)
            slot = int(slot_index)
        # distinct arm/seed per slot is the CALLER's job (_select_rows);
        # here source_row is already the slot's row
        slot_root = phase_root / "root"
        slot_config = slot_root / "config_bench.yaml"
        slot_config.parent.mkdir(parents=True, exist_ok=True)
        slot_config.write_bytes(cfg_path.read_bytes())   # identical bytes
        row = ib.build_bench_row(source_row, tag, updates,
                                 config_sha256=sha256_file(slot_config))
        header = ib.build_bench_header(None, row, tag)
        plan_path, _pin = ib.write_bench_plan(slot_root, row, header)
        ib.validate_bench_root(slot_root, plan_path, str(row["run_id"]),
                               slot_config, tables_dir, split_path)
        lock_path = Path(SLOT_LOCK_TEMPLATE.format(k=slot))
        cgroup_outdir = phase_root
        argv: List[str] = []
        if thread_caps:
            argv += ["bash", str(THREAD_CAPS)]
        argv += [
            "bash", str(CGROUP_SCOPE), memory_max, high, str(cgroup_outdir),
            sys.executable, str(ib.RUN_DRIVER),
            "--output-root", str(slot_root),
            "run", "--run-id", str(row["run_id"]),
            "--config", str(slot_config),
            "--tables-dir", str(tables_dir),
            "--split", str(split_path),
            "--r3-ckpt", str(r3_ckpt),
            "--device", device,
            "--gpu-lock", str(lock_path),
            "--loader", "v2",
            "--dev-pack", "auto",
            "--feature-workers", str(int(feature_workers)),
        ]
        if cuda_memory_fraction is not None:
            # forwarded to the driver (NOT re-implemented here): every slot
            # of this phase gets the same cap, argv evidence in dry-run
            argv += ["--cuda-memory-fraction", "{}".format(cuda_memory_fraction)]
        if eval_vram_gate_mib is not None:
            # forwarded to the driver (NOT re-implemented here): every slot
            # of this phase gets the same staggered-eval gate, argv evidence
            # in dry-run
            argv += ["--eval-vram-gate-mib", str(int(eval_vram_gate_mib)),
                     "--eval-vram-gate-timeout-s",
                     str(int(eval_vram_gate_timeout_s))]
        slots.append({
            "slot": slot,
            "run_id": str(row["run_id"]),
            "arm": str(row.get("arm")),
            "seed": int(row.get("seed", -1)),
            "root": str(slot_root),
            "plan": str(plan_path),
            "config": str(slot_config),
            "gpu_lock": str(lock_path),
            "cgroup_outdir": str(cgroup_outdir),
            "memory_max": memory_max,
            "memory_high": high,
            "feature_workers": int(feature_workers),
            "cuda_memory_fraction": cuda_memory_fraction,
            "eval_vram_gate_mib": eval_vram_gate_mib,
            "eval_vram_gate_timeout_s": (
                int(eval_vram_gate_timeout_s)
                if eval_vram_gate_mib is not None else None),
            "argv": argv,
        })
    return {"phase_root": str(phase_root), "slots": slots,
            "n_slots": n_slots, "updates": updates}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def _parse_slot_result(slot: Dict[str, Any], wall_s: float,
                       returncode: int) -> Dict[str, Any]:
    outdir = Path(slot["cgroup_outdir"])
    csv_path = outdir / "cgroup_sample.csv"
    csv_lines = (csv_path.read_text(encoding="utf-8").splitlines()
                 if csv_path.is_file() else [])
    events = ""
    ev_path = Path(str(csv_path) + ".events")
    if ev_path.is_file():
        events = ev_path.read_text(encoding="utf-8")
    cgroup = ib.parse_cgroup_samples(csv_lines, events)
    summary: Dict[str, Any] = {}
    summary_path = outdir / "cgroup_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    train = ib.parse_train_log(
        ib._read_lines(Path(slot["root"]) / slot["run_id"] /
                       "train_log.jsonl"),
        warmup=200)
    return {
        "slot": slot["slot"],
        "run_id": slot["run_id"],
        "arm": slot["arm"],
        "seed": slot["seed"],
        "returncode": int(returncode),
        "wall_s": round(float(wall_s), 1),
        "anon_peak_bytes": cgroup["anon_peak_bytes"],
        "max_swap_bytes": cgroup["swap_max_bytes"],
        "oom_kill": cgroup["oom_kill"],
        "gate_exit": summary.get("gate_exit"),
        "median_ms": train["median_ms"],
        "p95_ms": train["p95_ms"],
    }


def verdict_phase(phase_n: int, slot_results: List[Dict[str, Any]],
                  baseline_wall_s: Optional[float],
                  vram_peak_mib: Optional[int] = None,
                  vram_samples: int = 0) -> Dict[str, Any]:
    """Plan §7-B4 gates for one phase (pure; unit-testable).

    ``vram_peak_mib``/``vram_samples`` come from the per-phase whole-card
    ``nvidia-smi -l 2`` sampler (``infra_bench.parse_vram_samples``).  The
    ``whole_card_vram_le_6800_mib`` gate is ENFORCED at N=3 only; N=1 and
    N=4 record it informationally.  Missing samples (0) FAIL CLOSED at
    N=3.
    """
    walls = [r["wall_s"] for r in slot_results]
    anons = [r["anon_peak_bytes"] for r in slot_results]
    oom_total = sum(int(r["oom_kill"]) for r in slot_results)
    anon_sum = sum(anons)
    peak_mib = int(vram_peak_mib or 0)
    n_vram = int(vram_samples or 0)
    # fail-closed: no samples at all (peak 0) can never pass
    vram_ok = n_vram > 0 and 0 < peak_mib <= WHOLE_CARD_VRAM_GATE_MIB
    slot_ok = all(
        int(r.get("returncode", 1)) == 0 and r.get("median_ms") is not None
        for r in slot_results)
    checks: Dict[str, bool] = {
        "oom_kill_zero": oom_total == 0,
        "all_slots_completed": slot_ok,
    }
    if phase_n > 1 and baseline_wall_s:
        worst_ratio = max(walls) / max(baseline_wall_s, 1e-9)
        checks["per_run_wall_le_1_5x_baseline"] = (
            worst_ratio <= B4_WALL_RATIO_GATE)
    else:
        worst_ratio = None
    if phase_n == 3:
        checks["total_anon_lt_24gb"] = anon_sum < B3_ANON_SUM_GATE_BYTES
        checks["whole_card_vram_le_6800_mib"] = vram_ok
    return {
        "n_ways": phase_n,
        "pass": all(checks.values()),
        "checks": checks,
        "observed": {
            "walls_s": walls,
            "worst_wall_ratio_vs_baseline": (
                round(worst_ratio, 3) if worst_ratio is not None else None),
            "anon_peaks_bytes": anons,
            "total_anon_bytes": anon_sum,
            "total_anon_gb": round(anon_sum / 1e9, 3),
            "oom_kill_total": oom_total,
            "vram_peak_mib": peak_mib,
            "vram_samples": n_vram,
            # informational everywhere; gating only via checks at N=3
            "whole_card_vram_le_6800_mib": vram_ok,
            "failed_slots": [
                r.get("run_id") for r in slot_results
                if int(r.get("returncode", 1)) != 0
                or r.get("median_ms") is None],
        },
        "gates": {
            "total_anon_lt_24gb": "< 24 GB (N=3 only, plan §7-B4)",
            "per_run_wall_le_1_5x_baseline":
                "<= 1.5 x the N=1 phase wall (N>1)",
            "oom_kill_zero": "== 0 (every N)",
            "whole_card_vram_le_6800_mib":
                "<= 6800 MiB whole-card nvidia-smi peak (N=3 ENFORCED; "
                "N=1/N=4 informational; missing samples = fail)",
        },
    }


# ---------------------------------------------------------------------------
# drill orchestration
# ---------------------------------------------------------------------------
def _acquire_drill_lock() -> Optional[int]:
    """Non-blocking drill-level GPU-window lock; None = refused (busy)."""
    Path(DRILL_LOCK).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(DRILL_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd                                  # caller owns + releases it


def run_drill(args: argparse.Namespace) -> int:
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.plan)
    if not plan_path.is_file():
        raise SystemExit("error: PLAN_MISSING: {}".format(plan_path))
    plan_rows: List[Dict[str, Any]] = []
    for raw in plan_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        doc = json.loads(raw)
        if doc.get("record_type") == "plan_header":
            continue
        plan_rows.append(doc)
    if not plan_rows:
        raise SystemExit("error: PLAN_EMPTY: {}".format(plan_path))
    tables_dir = Path(args.tables_dir)
    split_path = Path(args.split) if args.split else (
        tables_dir / "split_manifest_v53mp.json")
    config_path = Path(args.config)
    r3_ckpt = Path(args.r3_ckpt)
    n_values = [int(x) for x in str(args.ns).split(",") if x.strip()]
    if not n_values or any(n < 1 for n in n_values):
        raise SystemExit("error: BAD_NS: {!r}".format(args.ns))
    if not CGROUP_SCOPE.is_file():
        raise SystemExit(
            "error: CGROUP_SCOPE_MISSING: {} (the drill is per-run "
            "cgroup-scoped)".format(CGROUP_SCOPE))
    feature_workers = workers_for_concurrency(max(n_values))
    cuda_fraction: Optional[float] = getattr(
        args, "cuda_memory_fraction", None)
    if cuda_fraction is not None:             # fail-closed, even off-CLI
        try:
            cuda_fraction = _cuda_memory_fraction_arg(cuda_fraction)
        except argparse.ArgumentTypeError as exc:
            raise SystemExit(
                "error: BAD_CUDA_MEMORY_FRACTION: {}".format(exc))
    eval_gate_mib: Optional[int] = getattr(
        args, "eval_vram_gate_mib", None)
    eval_gate_timeout_s: int = int(getattr(
        args, "eval_vram_gate_timeout_s", 600) or 600)
    if eval_gate_mib is not None:             # fail-closed, even off-CLI
        try:
            eval_gate_mib = _eval_vram_gate_mib_arg(eval_gate_mib)
            eval_gate_timeout_s = _eval_vram_gate_timeout_arg(
                eval_gate_timeout_s)
        except argparse.ArgumentTypeError as exc:
            raise SystemExit(
                "error: BAD_EVAL_VRAM_GATE: {}".format(exc))

    # --- plan: build + validate every phase/slot WITHOUT launching --------
    phases: List[Dict[str, Any]] = []
    for n in n_values:
        picked = _select_rows(plan_rows, n)
        phase_root = out_root / "n{}".format(n)
        slots_meta = []
        for slot, row in enumerate(picked):
            one_root = phase_root / "slot{}".format(slot)
            phase = build_phase_plan(
                row, tag="B4_n{}".format(n), updates=args.updates,
                n_slots=1, phase_root=one_root, tables_dir=tables_dir,
                split_path=split_path, config_path=config_path,
                r3_ckpt=r3_ckpt, device=args.device,
                memory_max=args.memory_max,
                feature_workers=workers_for_concurrency(n),
                slot_index=slot,
                thread_caps=bool(args.thread_caps),
                cuda_memory_fraction=cuda_fraction,
                eval_vram_gate_mib=eval_gate_mib,
                eval_vram_gate_timeout_s=eval_gate_timeout_s)
            slots_meta.append(phase["slots"][0])
        phases.append({"n": n, "phase_root": str(phase_root),
                       "slots": slots_meta,
                       "feature_workers": workers_for_concurrency(n)})
    _log("[plan] validated {} phases, {} slots total".format(
        len(phases), sum(len(p["slots"]) for p in phases)))

    if args.dry_run:
        report = {
            "schema": "v53mp_concurrency_drill_v1",
            "created_at": _utc_now(),
            "dry_run": True,
            "plan": str(plan_path),
            "updates": args.updates,
            "memory_max": args.memory_max,
            "feature_workers_max": feature_workers,
            "cuda_memory_fraction": cuda_fraction,
            "eval_vram_gate_mib": eval_gate_mib,
            "eval_vram_gate_timeout_s": eval_gate_timeout_s,
            "phases": phases,               # includes per-slot argv (evidence)
            "drill_lock": DRILL_LOCK,
        }
        (out_root / "report.json").write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps({"action": "concurrency_drill", "dry_run": True,
                          "phases": [p["n"] for p in phases],
                          "output_root": str(out_root)}, sort_keys=True))
        return 0

    # --- execute: ONE drill-level lock owns the GPU window ------------------
    lock_fd = _acquire_drill_lock()
    if lock_fd is None:
        raise SystemExit(
            "error: DRILL_LOCK_BUSY: {} (another drill/campaign owns the "
            "GPU window)".format(DRILL_LOCK))
    try:
        baseline_wall: Optional[float] = None
        phase_verdicts: List[Dict[str, Any]] = []
        phase_results: List[Dict[str, Any]] = []
        for phase in phases:
            n = phase["n"]
            phase_root = Path(phase["phase_root"])
            vram_path = phase_root / VRAM_SAMPLE_FILENAME
            _log("[phase] N={} launching {} concurrent runs "
                 "(whole-card VRAM sampler → {}) …".format(
                     n, len(phase["slots"]), vram_path))
            sampler = _start_vram_sampler(phase_root)
            try:
                procs = []
                t0 = time.perf_counter()
                for slot in phase["slots"]:
                    env = dict(os.environ)
                    env["CGROUP_SCOPE_UNIT"] = "v53mp-drill-n{}-slot{}".format(
                        n, slot["slot"])
                    log_path = Path(slot["cgroup_outdir"]).parent / "stdout.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    fh = log_path.open("w", encoding="utf-8")
                    proc = subprocess.Popen(
                        slot["argv"], stdout=fh, stderr=subprocess.STDOUT,
                        text=True, env=env)
                    procs.append((proc, fh, slot, time.perf_counter()))
                results: List[Dict[str, Any]] = []
                for proc, fh, slot, t_start in procs:
                    rc = proc.wait()
                    fh.close()
                    results.append(_parse_slot_result(
                        slot, time.perf_counter() - t_start, rc))
                phase_wall = time.perf_counter() - t0
            finally:
                # reliable stop even when the phase raised mid-flight
                _stop_vram_sampler(sampler)
            results.sort(key=lambda r: r["slot"])
            vram = ib.parse_vram_samples(ib._read_lines(vram_path))
            verdict = verdict_phase(
                n, results, baseline_wall if n > 1 else None,
                vram_peak_mib=vram["vram_peak_mib"],
                vram_samples=vram["vram_samples"])
            if n == 1 and results:
                baseline_wall = results[0]["wall_s"]
            phase_results.append({
                "n": n, "phase_wall_s": round(phase_wall, 1),
                "vram_sample_file": str(vram_path),
                "vram_peak_mib": vram["vram_peak_mib"],
                "vram_samples": vram["vram_samples"],
                "results": results})
            phase_verdicts.append(verdict)
            _log("[phase] N={} done: walls={} total_anon_gb={} "
                 "oom={} vram_peak_mib={} pass={}".format(
                     n, [r["wall_s"] for r in results],
                     verdict["observed"]["total_anon_gb"],
                     verdict["observed"]["oom_kill_total"],
                     verdict["observed"]["vram_peak_mib"],
                     verdict["pass"]))
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(lock_fd)

    passed = all(v["pass"] for v in phase_verdicts)
    report = {
        "schema": "v53mp_concurrency_drill_v1",
        "passed": bool(passed),
        "created_at": _utc_now(),
        "plan": str(plan_path),
        "updates": args.updates,
        "memory_max": args.memory_max,
        "feature_workers_max": feature_workers,
        "cuda_memory_fraction": cuda_fraction,
        "eval_vram_gate_mib": eval_gate_mib,
        "eval_vram_gate_timeout_s": eval_gate_timeout_s,
        "baseline_wall_s": baseline_wall,
        "phases": phase_results,
        "verdicts": phase_verdicts,
    }
    (out_root / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")
    md = ["# B4 concurrency drill {}".format(report["created_at"]), "",
          "- passed: **{}**".format(report["passed"]),
          "- plan: `{}`".format(plan_path),
          "- updates: {}, memory_max: {}, feature_workers(max): {}".format(
              args.updates, args.memory_max, feature_workers),
          "- baseline (N=1) wall: {} s".format(baseline_wall),
          "- cuda_memory_fraction: {}".format(cuda_fraction),
          "- eval_vram_gate_mib: {} (timeout {} s)".format(
              eval_gate_mib, eval_gate_timeout_s), "",
          "| N | walls (s) | total anon (GB) | oom_kill | vram peak (MiB) | pass |",
          "|---|---|---|---|---|---|"]
    for verdict, pres in zip(phase_verdicts, phase_results):
        md.append("| {} | {} | {} | {} | {} | {} |".format(
            verdict["n_ways"],
            verdict["observed"]["walls_s"],
            verdict["observed"]["total_anon_gb"],
            verdict["observed"]["oom_kill_total"],
            verdict["observed"]["vram_peak_mib"],
            verdict["pass"]))
    md += ["", "## Verdicts", "", "```json",
           json.dumps(phase_verdicts, indent=1, sort_keys=True), "```", ""]
    (out_root / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"action": "concurrency_drill",
                      "output_root": str(out_root),
                      "passed": passed,
                      "baseline_wall_s": baseline_wall}, sort_keys=True))
    return 0 if passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="concurrency_drill",
        description="S4 B4 concurrency drill (plan §7); N={1,3,4} "
                    "concurrent 2000-update bench runs under per-run "
                    "cgroup scopes")
    parser.add_argument("--plan", default="outputs/pesn/v53mp/"
                        "run_plan_v53mp_20k.jsonl")
    parser.add_argument("--output-root",
                        default="outputs/pesn/v53mp/concurrency_drill")
    parser.add_argument("--ns", default="1,3,4",
                        help="comma-separated concurrency values "
                             "(default 1,3,4; N=1 is the wall baseline)")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--tables-dir", default=ib.DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default=None)
    parser.add_argument("--config", default=ib.DEFAULT_CONFIG)
    parser.add_argument("--r3-ckpt", default=ib.R3_WINNER_CKPT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-max", default=DEFAULT_MEMORY_MAX,

                        help="per-run cgroup MemoryMax (plan §5.1: 6G)")
    parser.add_argument("--thread-caps", action="store_true",
                        help="wrap each slot in thread_caps.sh (OMP/BLAS=1; "
                             "plan §5.5 — measured numerics-safe for the "
                             "feature builder)")
    parser.add_argument("--cuda-memory-fraction",
                        type=_cuda_memory_fraction_arg, default=None,
                        help="0 < f <= 1: forwarded to every slot's driver "
                             "argv as '--cuda-memory-fraction f' "
                             "(default: off — flag absent)")
    parser.add_argument("--eval-vram-gate-mib",
                        type=_eval_vram_gate_mib_arg, default=None,
                        help="whole-card MiB threshold forwarded to every "
                             "slot's driver argv as '--eval-vram-gate-mib "
                             "mib --eval-vram-gate-timeout-s s' (3-way "
                             "staggered evals; default: off — flags absent)")
    parser.add_argument("--eval-vram-gate-timeout-s",
                        type=_eval_vram_gate_timeout_arg, default=600,
                        help="max seconds each slot's trainer waits per "
                             "gated eval (fail-open; default 600; forwarded "
                             "only when --eval-vram-gate-mib is set)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build + validate the full plan (roots, "
                             "configs, wrappers, worker counts) without "
                             "launching or touching the GPU")
    args = parser.parse_args(argv)
    return run_drill(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
