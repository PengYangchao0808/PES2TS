#!/usr/bin/env python
"""V5 phase-1 ten-hour sprint campaign driver (Lane D; plan §5, ADR-0026).

Single-entry orchestration of the frozen five-stage sprint:

* ``prepare-audit`` (90 min) — split freeze → teacher/audit build → feature
  precompute + Oracle/Classic baseline table + G5.1 pytest evidence → real
  GPU timing probe (``timing_probe.json``) → ``start_frame_sampling_manifest``
  → activate the ``V5_SPRINT`` G–V registry entry (proposed→active,
  idempotent) and re-render the mapping doc;
* ``label-screen`` (120 min) — feasibility-first unified budget
  (6×t_run(1500) ≤ 108 min else ALL SIX arms downgrade 1500→1000→500,
  500 = exploratory only), then the six screen arms as sequential
  SUBPROCESS trainer runs; ranking per plan §5.3 (cluster-median RMSE/zero,
  gap<0.02 → near-TS cos → static violation rate; never raw MSE across
  d0 spans); writes ``label_recipe_matrix.json`` + ``ablation_table.parquet``;
* ``recipe-recheck`` (150 min) — top-2 recipes × folds{0,1} × seeds{17,29,43}
  (12 cells, 2 reused), stability verdict per plan §5.3; writes
  ``recipe_recheck.parquet`` + decision;
* ``crossfit`` (150 min, CONDITIONAL) — winning recipe formal 5-fold
  (3 seeds if the feasibility gate allows, else 1 seed, else archive-only
  OOF deferred) from-scratch 3000-update runs + OOF assembly to
  ``move_proposal_v5_draft1.parquet`` (plan §1.4 columns; missing rows are
  explicitly censored, never zero-filled; selection-exposed recorded);
* ``report`` (90 min) — completeness + dense dev re-validation, sealed-eval
  request (typed REFUSED with registration echoed — this sprint NEVER
  unseals, plan §5.5/ADR-0026 §4), collision/follower diagnostics, grouped
  bootstrap intervals, ``scale_curriculum_next_run.json`` (planned-only),
  ``gate_report.md/json`` and the dated sprint report under ``docs/reports/``.

House rules (plan §5.1 / v42 driver conventions): dry-run lists actions and
freezes NOTHING (no T0, no files); every formal run is a subprocess
(``python -m pes2ts_core.pesn.offline.training.key_atom_bc``) so a crashed
arm cannot take down the driver; crash-restart safe via
``campaign_manifest.json`` + ``stage_decisions.jsonl`` + per-run
``status.json`` (completed runs are skipped); T0 is frozen once on the first
real (non-dry) execution; parent hashes (config / split manifest) are
re-verified at every stage start (``--force`` downgrades a mismatch to a
recorded override); T+window−5min launches no new subprocess and T+window
saves a censored state and exits typed.  NO QC/ACP is ever launched.

Status vocabulary (plan §5.7, EXACT): complete_label_screen /
complete_recipe_recheck / complete_oof_3seed / complete_oof_1seed /
censored_budget / rejected_data / rejected_gate / insufficient_evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.config_loader import load_config_mapping  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (  # noqa: E402
    evaluate_g52,
    evaluate_g54,
    follower_counterfactual_diagnostics,
    oracle_static_ceiling,
    sealed_eval_request,
    write_gate_report,
)
from pes2ts_core.utils.jsonio import read_json, write_json  # noqa: E402

#: Python executable for the single-GPU training environment.
PYTHON: Final[str] = os.environ.get("PES2TS_GPU_PYTHON", sys.executable)

#: Campaign id (ADR-0026 §3; g_v_map ``V5_SPRINT``).
CAMPAIGN_ID: Final[str] = "V5_SPRINT"
MANIFEST_SCHEMA: Final[str] = "pesn_v5_sprint_manifest_v1"
DECISION_SCHEMA: Final[str] = "pesn_v5_sprint_decision_v1"

#: Plan §5.4 five-stage schedule — cumulative minute offsets from T0.
STAGE_ORDER: Final[tuple[str, ...]] = (
    "prepare-audit", "label-screen", "recipe-recheck", "crossfit", "report")
STAGE_OFFSETS_MIN: Final[dict[str, tuple[int, int]]] = {
    "prepare-audit": (0, 90),
    "label-screen": (90, 210),
    "recipe-recheck": (210, 360),
    "crossfit": (360, 510),
    "report": (510, 600),
}
#: Per-stage launch guard: no new subprocess in the final 5 minutes.
LAUNCH_GUARD_MINUTES: Final[int] = 5

#: Plan §5.7 status vocabulary — EXACT, no extensions.
STAGE_STATUS_VALUES: Final[tuple[str, ...]] = (
    "complete_label_screen",
    "complete_recipe_recheck",
    "complete_oof_3seed",
    "complete_oof_1seed",
    "censored_budget",
    "rejected_data",
    "rejected_gate",
    "insufficient_evidence",
)

#: Tier-1 six arms (plan §5.2 table, config ``sprint.arm_table`` order).
SCREEN_ARMS: Final[tuple[str, ...]] = ("C05", "C08", "C12", "A05", "A08", "A12")

#: Exit codes.
EXIT_OK: Final[int] = 0
EXIT_REFUSED: Final[int] = 2
EXIT_CENSORED: Final[int] = 3

#: G5.1 evidence files (plan §5.4 stage 0, 65–80 min checkpoint).
G51_TEST_FILES: Final[tuple[str, ...]] = (
    "tests/pesn/test_key_atom_model.py",
    "tests/pesn/test_key_atom_losses.py",
    "tests/pesn/test_key_atom_features.py",
    "tests/pesn/test_key_atom_teacher.py",
)

#: Near-TS stability floor for the recipe-recheck verdict (pre-registered
#: G5.2 near-TS cosine gate value; plan §5.3 "near-TS not degraded").
RECHECK_NEAR_TS_FLOOR: Final[float] = 0.50


class SprintRefused(RuntimeError):
    """Typed campaign precheck refusal (machine-readable code)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"REFUSED[{code}]: {detail}")
        self.code = code
        self.detail = detail


class StageCensored(RuntimeError):
    """Raised inside a stage when the wall-clock envelope is exhausted."""


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise SprintRefused(code, detail)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pure planning helpers (hermetic; unit-tested)
# ---------------------------------------------------------------------------

def t_run_seconds(updates: int, t_step_p90_s: float, t_eval_s: float,
                  t_start_save_s: float) -> float:
    """Plan §5.4 timing model: ``t_run(U)=U*t_step_P90+ceil(U/250)*t_eval
    +t_start_save`` (eval interval fixed at 250 updates)."""
    u = int(updates)
    _require(u > 0, "TIMING_UPDATES", f"updates must be positive, got {u}")
    return (u * float(t_step_p90_s)
            + math.ceil(u / 250) * float(t_eval_s)
            + float(t_start_save_s))


def resolve_screen_updates(
    timing: Mapping[str, float],
    *,
    ladder: Sequence[int] = (1500, 1000, 500),
    n_runs: int = 6,
    budget_minutes: float = 108.0,
) -> dict:
    """Feasibility-first unified screen budget (plan §5.4 stage 1).

    Predicted ``n_runs × t_run(largest)`` must fit ``budget_minutes``;
    otherwise ALL arms downgrade together along the ladder
    [1500, 1000, 500].  500 is EXPLORATORY ONLY and is recorded as such.
    Returns ``{updates, exploratory_only, per_run_minutes,
    total_minutes, ladder, feasible}``; ``updates=None`` when even the
    smallest rung does not fit (caller must refuse, never start).
    """
    step = float(timing["t_step_p90_s"])
    ev = float(timing["t_eval_s"])
    save = float(timing["t_start_save_s"])
    for updates in ladder:
        per_run = t_run_seconds(updates, step, ev, save) / 60.0
        total = per_run * int(n_runs)
        if total <= float(budget_minutes):
            return {
                "updates": int(updates),
                "exploratory_only": int(updates) < 1000,
                "per_run_minutes": per_run,
                "total_minutes": total,
                "ladder": [int(v) for v in ladder],
                "budget_minutes": float(budget_minutes),
                "n_runs": int(n_runs),
                "feasible": True,
            }
    smallest = int(ladder[-1])
    per_run = t_run_seconds(smallest, step, ev, save) / 60.0
    return {
        "updates": None,
        "exploratory_only": True,
        "per_run_minutes": per_run,
        "total_minutes": per_run * int(n_runs),
        "ladder": [int(v) for v in ladder],
        "budget_minutes": float(budget_minutes),
        "n_runs": int(n_runs),
        "feasible": False,
    }


def resolve_recheck_updates(
    timing: Mapping[str, float], *, window_minutes: float = 150.0,
    n_new_runs: int = 10, per_run_cap_minutes: float = 12.0,
    freeze_minutes: float = 30.0, ladder: Sequence[int] = (1500, 1000, 500),
) -> dict:
    """Tier-2 budget (plan §5.4 stage 2): 10 new runs ≤12 min each inside
    ``window_minutes − freeze_minutes``; downgrade along the ladder when
    infeasible.  ``stable_winner_allowed`` is False when the resolved
    budget is <1000 updates — a sub-1000 recheck cannot claim stability
    (plan §5.4: censored_budget, no stability winner)."""
    run_budget = float(window_minutes) - float(freeze_minutes)
    step = float(timing["t_step_p90_s"])
    ev = float(timing["t_eval_s"])
    save = float(timing["t_start_save_s"])
    for updates in ladder:
        per_run = t_run_seconds(updates, step, ev, save) / 60.0
        if per_run <= float(per_run_cap_minutes) \
                and per_run * int(n_new_runs) <= run_budget:
            return {
                "updates": int(updates),
                "per_run_minutes": per_run,
                "total_minutes": per_run * int(n_new_runs),
                "per_run_cap_minutes": float(per_run_cap_minutes),
                "run_budget_minutes": run_budget,
                "stable_winner_allowed": int(updates) >= 1000,
                "feasible": True,
            }
    smallest = int(ladder[-1])
    per_run = t_run_seconds(smallest, step, ev, save) / 60.0
    return {
        "updates": None,
        "per_run_minutes": per_run,
        "total_minutes": per_run * int(n_new_runs),
        "per_run_cap_minutes": float(per_run_cap_minutes),
        "run_budget_minutes": run_budget,
        "stable_winner_allowed": False,
        "feasible": False,
    }


def resolve_crossfit_mode(
    timing: Mapping[str, float], *, window_minutes: float = 150.0,
    oof_assembly_minutes: float = 15.0, single_seed_assembly_minutes: float = 15.0,
    formal_updates: int = 3000,
) -> dict:
    """Plan §5.4 stage 3 feasibility ladder.

    ``15×t_run(3000)+30 ≤ window`` → full 5-fold × 3 seeds
    (``complete_oof_3seed``); else ``5×t_run(3000)+15 ≤ window`` →
    5-fold × 1 seed (``complete_oof_1seed``); else archive-only with the
    OOF explicitly deferred (never faked).
    """
    step = float(timing["t_step_p90_s"])
    ev = float(timing["t_eval_s"])
    save = float(timing["t_start_save_s"])
    per_run_min = t_run_seconds(formal_updates, step, ev, save) / 60.0
    full = per_run_min * 15.0 + 30.0
    if full <= float(window_minutes):
        return {"mode": "full_15", "n_runs": 15, "n_seeds": 3,
                "per_run_minutes": per_run_min, "predicted_minutes": full,
                "status_on_completion": "complete_oof_3seed",
                "feasible": True}
    single = per_run_min * 5.0 + float(single_seed_assembly_minutes)
    if single <= float(window_minutes):
        return {"mode": "single_seed_5", "n_runs": 5, "n_seeds": 1,
                "per_run_minutes": per_run_min, "predicted_minutes": single,
                "status_on_completion": "complete_oof_1seed",
                "feasible": True}
    return {"mode": "archive_only", "n_runs": 0, "n_seeds": 0,
            "per_run_minutes": per_run_min,
            "predicted_minutes": None,
            "status_on_completion": "insufficient_evidence",
            "feasible": False}


def is_eval_curve_stable(series: Sequence[float], *,
                         tolerance: float = 0.05) -> bool:
    """Screen-curve stability flag (plan §5.3 "inconclusive").

    Unstable when any aggregate is non-finite OR the final value regressed
    more than ``tolerance`` above the best value (a stalled/oscillating
    short run must not be reported as a final capacity verdict).
    """
    values = [float(v) for v in series]
    if not values:
        return False
    if any(not math.isfinite(v) for v in values):
        return False
    return values[-1] <= min(values) + tolerance


def rank_screen_arms(rows: Sequence[Mapping]) -> list[dict]:
    """Plan §5.3 screen ranking with the pre-registered tie-break chain.

    Primary: per-span reaction-cluster-median ``RMSE/zero`` (lower better;
    each span is normalized by ITS OWN teacher/zero baseline — raw MSE is
    never compared across d0 spans).  When the primary gap is <0.02 the
    contending set is re-ordered by near-TS cosine (higher better), then
    by lower static violation rate, then deterministically by
    (primary, arm_id).  Censored / undefined rows are ranked last.
    Returns copies with ``rank`` (1-based, None when unranked) and
    ``tie_break`` provenance.
    """
    prepared: list[dict] = []
    for row in rows:
        item = dict(row)
        rmse = item.get("rmse_over_zero")
        item["_ranked"] = (
            item.get("censored") is not True
            and rmse is not None and math.isfinite(float(rmse)))
        item["rmse_over_zero"] = float(rmse) if item["_ranked"] else None
        prepared.append(item)
    ranked = [r for r in prepared if r["_ranked"]]
    unranked = [r for r in prepared if not r["_ranked"]]
    if not ranked:
        for item in prepared:
            item["rank"] = None
            item["tie_break"] = "unranked"
        return prepared
    primary = sorted(ranked, key=lambda r: (r["rmse_over_zero"],
                                            str(r.get("arm_id", ""))))
    best = primary[0]["rmse_over_zero"]
    contending = [r for r in primary if r["rmse_over_zero"] <= best + 0.02]
    contending.sort(key=lambda r: (
        -(float(r["near_ts_cos"])
          if r.get("near_ts_cos") is not None else -1e9),
        float(r["static_violation_rate"])
        if r.get("static_violation_rate") is not None else 1e9,
        r["rmse_over_zero"],
        str(r.get("arm_id", ""))))
    tie_applied = len(contending) > 1
    ordered = contending + primary[len(contending):]
    for index, item in enumerate(ordered):
        item["rank"] = index + 1
        item["tie_break"] = (
            "near_ts_cos+static_violation" if (tie_applied and index
                                               < len(contending))
            else "primary_rmse_over_zero")
    for item in unranked:
        item["rank"] = None
        item["tie_break"] = "censored_or_undefined"
    return ordered + unranked


def validate_status(status: str) -> str:
    """Plan §5.7: only the exact eight statuses are admissible."""
    if status not in STAGE_STATUS_VALUES:
        raise ValueError(
            f"status {status!r} not in plan §5.7 vocabulary "
            f"{list(STAGE_STATUS_VALUES)}")
    return status


class StageWindow:
    """Wall-clock envelope of one sprint stage (plan §5.4/§5.7).

    ``may_launch()`` is False in the final ``LAUNCH_GUARD_MINUTES`` (no new
    subprocess); ``expired()`` at T+window (save censored state, exit
    typed).  The clock is injectable for hermetic tests.
    """

    def __init__(self, t0: float, stage: str, *,
                 clock: Callable[[], float] = time.time) -> None:
        _require(stage in STAGE_OFFSETS_MIN, "STAGE_UNKNOWN",
                 f"unknown stage {stage!r}")
        self.t0 = float(t0)
        self.stage = stage
        self.start_min, self.end_min = STAGE_OFFSETS_MIN[stage]
        self._clock = clock

    def now(self) -> float:
        return float(self._clock())

    def deadline(self) -> float:
        return self.t0 + self.end_min * 60.0

    def launch_deadline(self) -> float:
        return self.t0 + (self.end_min - LAUNCH_GUARD_MINUTES) * 60.0

    def remaining_minutes(self) -> float:
        return (self.deadline() - self.now()) / 60.0

    def may_launch(self) -> bool:
        return self.now() < self.launch_deadline()

    def expired(self) -> bool:
        return self.now() >= self.deadline()

    def ensure_launch(self) -> None:
        """Raise :class:`StageCensored` when no further run may START."""
        if self.expired():
            raise StageCensored(
                f"stage {self.stage} hit T+{self.end_min}min window end")
        if not self.may_launch():
            raise StageCensored(
                f"stage {self.stage} within final {LAUNCH_GUARD_MINUTES}min "
                "launch guard (T+window-5min: no new subprocess)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class SprintDriver:
    """Crash-restart-safe orchestrator over the five sprint stages."""

    def __init__(
        self,
        *,
        config_path: Path,
        out_root: Path | None = None,
        allow_truth: bool = False,
        unseal: bool = False,
        force: bool = False,
        dry_run: bool = False,
        device: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_config_mapping(self.config_path)
        _require(self.config.get("schema") == "pesn_v5_sprint_draft2",
                 "CONFIG_SCHEMA",
                 f"expected schema pesn_v5_sprint_draft2, got "
                 f"{self.config.get('schema')!r}")
        self.paths = dict(self.config.get("paths") or {})
        self.sprint = dict(self.config.get("sprint") or {})
        self.allow_truth = bool(allow_truth)
        self.unseal = bool(unseal)
        self.force = bool(force)
        self.dry_run = bool(dry_run)
        self.device = device
        self._clock = clock
        self.out_root = Path(out_root) if out_root is not None else Path(
            self.paths.get("output_root", "outputs/pesn/v5/sprint"))
        self.manifest_path = self.out_root / "campaign_manifest.json"
        self.decisions_path = self.out_root / "stage_decisions.jsonl"
        self.manifest: dict | None = None
        self._load_manifest()

    # -- paths ---------------------------------------------------------------

    def _p(self, key: str, default: str) -> Path:
        return Path(self.paths.get(key, default))

    def _run_dir(self, run_id: str) -> Path:
        return self.out_root / "runs" / run_id

    @staticmethod
    def _run_id(arm: str, fold: int, seed: int, mode: str) -> str:
        return f"{arm}_f{int(fold)}_s{int(seed)}_{mode}"

    # -- manifest / decisions (append-only; dry-run writes nothing) ----------

    def _load_manifest(self) -> None:
        if self.manifest_path.is_file():
            self.manifest = read_json(self.manifest_path)  # type: ignore[assignment]

    def ensure_manifest(self) -> dict:
        """Freeze T0 on the first REAL execution (never in dry-run)."""
        if self.manifest is None:
            _require(not self.dry_run, "DRY_RUN_WRITE",
                     "dry-run must never freeze T0 or write the manifest")
            self.out_root.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "T0_unix": float(self._clock()),
                "T0_iso": _utc_now(),
                "config_path": str(self.config_path),
                "config_sha256": _sha256_file(self.config_path),
                "arms": list(self.sprint.get("arms") or SCREEN_ARMS),
                "stage_offsets_minutes": {
                    k: list(v) for k, v in STAGE_OFFSETS_MIN.items()},
                "status_vocabulary": list(STAGE_STATUS_VALUES),
                "stages": {},
                "parents": {},
                "qc": "QC not scheduled",
            }
            write_json(self.manifest_path, self.manifest)
        return self.manifest

    def append_decision(self, decision: Mapping) -> None:
        if self.dry_run:
            return
        self.out_root.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": DECISION_SCHEMA,
                   "at_unix": float(self._clock()), **dict(decision)}
        with self.decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True,
                                ensure_ascii=False) + "\n")

    def _record_stage(self, stage: str, status: str, *, detail: str = "",
                      extra: Mapping | None = None) -> str:
        validate_status(status)
        entry = {"status": status, "detail": detail,
                 "at_unix": float(self._clock()), "at_iso": _utc_now()}
        if extra:
            entry.update(dict(extra))
        if not self.dry_run:
            manifest = self.ensure_manifest()
            manifest.setdefault("stages", {})[stage] = entry
            write_json(self.manifest_path, manifest)
            self.append_decision({"stage": stage, **entry})
        return status

    # -- parent hash re-verification (common) --------------------------------

    def _verify_parents(self, stage: str) -> dict:
        """Re-verify config + split-manifest hashes recorded at stage 0."""
        manifest = self.ensure_manifest()
        parents = manifest.setdefault("parents", {})
        current = {
            "config_sha256": _sha256_file(self.config_path),
            "split_manifest_sha256": _sha256_file(
                self._p("split_manifest",
                        "data/manifests/v5/split_manifest.json")),
        }
        if not parents:
            parents.update(current)
            write_json(self.manifest_path, manifest)
        else:
            mismatched = [k for k, v in current.items()
                          if v is not None and parents.get(k) not in (None, v)]
            if mismatched and not self.force:
                raise SprintRefused(
                    "PARENT_HASH_MISMATCH",
                    f"stage {stage}: parent hashes changed since first "
                    f"stage ({', '.join(mismatched)}); re-freeze or --force")
            if mismatched:
                self.append_decision({
                    "stage": stage, "decision": "parent_hash_override",
                    "forced": True, "mismatched": mismatched,
                    "recorded": {k: parents.get(k) for k in mismatched},
                    "current": {k: current[k] for k in mismatched}})
        return current

    # -- stage prechecks (typed REFUSED) ------------------------------------

    def _precheck(self, stage: str) -> None:
        if stage in ("label-screen", "recipe-recheck", "crossfit", "report"):
            _require(self._p("split_manifest",
                             "data/manifests/v5/split_manifest.json").is_file(),
                     "SPLIT_MANIFEST_MISSING",
                     "run --stage prepare-audit first")
        if stage in ("label-screen", "recipe-recheck", "crossfit", "report"):
            _require((self.out_root / "timing_probe.json").is_file(),
                     "TIMING_PROBE_MISSING",
                     "run --stage prepare-audit first (timing probe)")
        if stage == "recipe-recheck":
            _require((self.out_root / "label_recipe_matrix.json").is_file(),
                     "LABEL_MATRIX_MISSING",
                     "run --stage label-screen first")
        if stage == "crossfit":
            decision = self._recheck_decision()
            _require(decision is not None, "RECHECK_DECISION_MISSING",
                     "run --stage recipe-recheck first")
            _require(bool(decision.get("stable_recipe")),
                     "NO_STABLE_WINNER",
                     "crossfit is conditional on a Stage-2 stable winner "
                     "(plan §5.3); archive dev results only")
            audit = self._prepare_audit_summary()
            _require(audit.get("g51_pass") is True, "G51_NOT_GREEN",
                     "crossfit requires G5.1 fixtures green (plan §5.3)")
            oracle = audit.get("oracle_static")
            _require(oracle is not None, "ORACLE_STATIC_MISSING",
                     "crossfit requires the Oracle static ceiling from "
                     "prepare-audit")
            _require(str(oracle.get("status")) == "ORACLE_STATIC_OK",
                     "ORACLE_STATIC_FAILED",
                     "Oracle teacher fails G5.4 under the same protocol — "
                     "the ACTION REPRESENTATION is flagged, not the network "
                     "(plan §6 G5.4); revise d0/contract before crossfit")

    # -- subprocess plumbing (ONLY here; never inside pes2ts_core/pesn) -----

    def _spawn(self, argv: Sequence[str], *, stage: str,
               window: StageWindow) -> subprocess.CompletedProcess:
        window.ensure_launch()
        started = time.time()
        proc = subprocess.run(list(argv), cwd=str(PROJECT_ROOT))
        self.append_decision({
            "stage": stage, "decision": "subprocess",
            "argv": list(argv), "returncode": proc.returncode,
            "wall_s": time.time() - started,
            "gpu_memory_mib": self._gpu_memory_snapshot(),
        })
        return proc

    @staticmethod
    def _gpu_memory_snapshot() -> int | None:
        """Best-effort used-VRAM reading (logged per run; None if absent)."""
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30)
            if proc.returncode == 0 and proc.stdout.strip():
                return int(proc.stdout.strip().splitlines()[0].strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        return None

    def _trainer_argv(self, *, arm: str, fold: int, seed: int, updates: int,
                      mode: str, run_dir: Path) -> list[str]:
        argv = [PYTHON, "-m",
                "pes2ts_core.pesn.offline.training.key_atom_bc",
                "--config", str(self.config_path),
                "--arm", str(arm), "--fold", str(fold),
                "--seed", str(seed), "--updates", str(int(updates)),
                "--mode", mode, "--out", str(run_dir)]
        if self.device:
            argv += ["--device", str(self.device)]
        return argv

    def _run_status(self, run_dir: Path) -> dict | None:
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            return None
        try:
            return read_json(status_path)  # type: ignore[return-value]
        except (OSError, json.JSONDecodeError):
            return None

    def _spawn_or_skip(self, *, arm: str, fold: int, seed: int, updates: int,
                       mode: str, stage: str, window: StageWindow) -> dict:
        run_dir = self._run_dir(self._run_id(arm, fold, seed, mode))
        status = self._run_status(run_dir)
        if (not self.force and status is not None
                and str(status.get("status")) == "complete"):
            self.append_decision({
                "stage": stage, "decision": "skip_complete",
                "arm": arm, "fold": fold, "seed": seed, "mode": mode,
                "run_dir": str(run_dir)})
            return {"run_dir": str(run_dir), "skipped": True, **status}
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        proc = self._spawn(
            self._trainer_argv(arm=arm, fold=fold, seed=seed, updates=updates,
                               mode=mode, run_dir=run_dir),
            stage=stage, window=window)
        final = self._run_status(run_dir) or {}
        return {
            "run_dir": str(run_dir), "skipped": False,
            "returncode": proc.returncode,
            "wall_s": time.time() - started,
            "gpu_memory_mib": self._gpu_memory_snapshot(),
            "arm": arm, "fold": fold, "seed": seed, "mode": mode,
            "updates": int(updates),
            "status": str(final.get("status", "unknown")),
        }

    # -- run metrics readers -------------------------------------------------

    def _final_eval_aggregate(self, run_dir: Path) -> dict | None:
        """Last raw + bounded aggregate from ``eval.jsonl`` (or None)."""
        eval_path = Path(run_dir) / "eval.jsonl"
        if not eval_path.is_file():
            return None
        last: dict | None = None
        for line in eval_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
        if last is None:
            return None
        return {
            "update": last.get("update"),
            "raw": last.get("aggregate") or {},
            "bounded": last.get("aggregate_bounded") or {},
            "per_cluster": last.get("per_cluster") or {},
            "per_cluster_bounded": last.get("per_cluster_bounded") or {},
        }

    @staticmethod
    def _arm_static_violation_rate(run_dir: Path) -> float | None:
        """G5.4-style static violation summary recorded by the report stage.

        Screen runs do not carry static diagnostics; the report stage
        computes them from checkpoints (``static_diagnostics.json``).  The
        ranking tie-break consumes the recorded value when present.
        """
        path = Path(run_dir) / "static_diagnostics.json"
        if not path.is_file():
            return None
        try:
            doc = read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        rates = (doc.get("g54") or {}).get("rates") or {}
        clash = rates.get("new_clash_rate")
        prot = rates.get("new_protected_rate")
        if clash is None and prot is None:
            return None
        return float((clash or 0.0) + (prot or 0.0))

    # -- timing probe --------------------------------------------------------

    def _timing_probe(self) -> dict:
        """Real GPU probe: 30 warmup + 200 timed train steps + 3 dense evals.

        Uses the Lane A/B adapters (``load_lane_b_pools`` /
        ``load_lane_b_view``), forward/backward/optimizer on batch 32 with
        the campaign loss, and records the RAW step/eval/save distributions
        plus the P50/P90 summaries feeding the §5.4 t_run model.
        """
        import numpy as np
        import torch

        from pes2ts_core.pesn.offline.training.device import resolve_device
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            KeyAtomBCTrainerError, KeyAtomLossConfig, KeyAtomCycleSampler,
            cos_lambda, key_atom_bc_loss_vectorized as key_atom_bc_loss,
            load_lane_b_pools, load_lane_b_view)
        from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN

        dev_cfg = dict(self.config.get("device") or {})
        try:
            _torch, device = resolve_device(
                device=self.device or str(dev_cfg.get("device", "cuda:0")),
                require_cuda=bool(dev_cfg.get("require_cuda", True)))
        except Exception as exc:  # typed device refusal -> rejected_data
            raise SprintRefused("PROBE_DEVICE", str(exc)) from exc

        probe_cfg = dict(self.sprint.get("timing_probe") or {})
        warmup = int(probe_cfg.get("warmup_steps", 30))
        train_steps = int(probe_cfg.get("train_steps", 200))
        dense_evals = int(probe_cfg.get("dense_evals", 3))
        split_path = str(self._p(
            "split_manifest", "data/manifests/v5/split_manifest.json"))
        main_d0 = float(self.config.get("d0_angstrom", 0.08))
        batch_size = int(self.config.get("batch_size", 32))

        try:
            pools = load_lane_b_pools(self.config, fold=0, d0=main_d0,
                                      split_path=split_path)
            torch.manual_seed(int((self.config.get("seeds") or [17])[0]))
            model = StarEGNN().to(torch.float32).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(self.config.get("lr", 1e-3)),
                weight_decay=float(self.config.get("weight_decay", 1e-4)))
            sampler = KeyAtomCycleSampler(
                pools["train"], seed=17,
                reactions_per_batch=int(
                    self.config.get("reactions_per_batch", 8)),
                frames_per_reaction=int(
                    self.config.get("frames_per_reaction", 4)))
            dev_view, dev_collate = load_lane_b_view(
                config=self.config, samples=list(pools["dev_dense"]),
                fold=0, d0=main_d0, anchor="on", split_path=split_path,
                device=device)
            dev_batches = []
            for i in range(0, len(dev_view), batch_size):
                chunk = [dev_view[j] for j in
                         range(i, min(i + batch_size, len(dev_view)))]
                if chunk:
                    dev_batches.append(dev_collate(chunk))
        except (KeyAtomBCTrainerError, ImportError, FileNotFoundError) as exc:
            raise SprintRefused("PROBE_DATA", str(exc)) from exc

        loss_cfg = KeyAtomLossConfig.from_campaign_config(self.config)

        def _one_step() -> None:
            rows = sampler.draw_batch()
            view, collate = load_lane_b_view(
                config=self.config, samples=rows, fold=0, d0=main_d0,
                anchor="on", split_path=split_path, device=device)
            for i in range(0, len(view), batch_size):
                chunk = [view[j] for j in
                         range(i, min(i + batch_size, len(view)))]
                if not chunk:
                    continue
                batch = collate(chunk)
                u, _z = model(batch)
                out = key_atom_bc_loss(
                    u.float(), batch.y_teacher.float(), batch.degK.float(),
                    batch.key_ptr, batch.pos, batch.key_index,
                    batch.pair_index, config=loss_cfg,
                    lambda_cos=cos_lambda(1, loss_cfg))
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(self.config.get("grad_clip_norm", 1.0)))
                optimizer.step()

        model.train()
        for _ in range(warmup):
            _one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times: list[float] = []
        for _ in range(train_steps):
            t_start = time.time()
            _one_step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.time() - t_start)

        eval_times: list[float] = []
        model.train(False)
        with torch.no_grad():
            for _ in range(max(1, dense_evals)):
                t_start = time.time()
                for batch in dev_batches:
                    _u, _z = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                eval_times.append(time.time() - t_start)
        model.train()

        t_start = time.time()
        tmp_ckpt = self.out_root / ".timing_probe_ckpt.pt"
        self.out_root.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, tmp_ckpt)
        start_save_s = time.time() - t_start
        tmp_ckpt.unlink(missing_ok=True)

        steps = np.asarray(step_times, dtype=np.float64)
        evals = np.asarray(eval_times, dtype=np.float64)
        probe = {
            "schema_version": "pesn_v5_timing_probe_v1",
            "device": str(device),
            "gpu_name": (torch.cuda.get_device_name(0)
                         if device.type == "cuda" else "cpu"),
            "warmup_steps": warmup,
            "train_steps": train_steps,
            "dense_evals": dense_evals,
            "batch_size": batch_size,
            "t_step_p50_s": float(np.quantile(steps, 0.50)),
            "t_step_p90_s": float(np.quantile(steps, 0.90)),
            "t_step_mean_s": float(steps.mean()),
            "t_eval_s": float(evals.mean()),
            "t_start_save_s": float(start_save_s),
            "raw": {
                "step_seconds": [float(v) for v in steps],
                "eval_seconds": [float(v) for v in evals],
                "start_save_seconds": float(start_save_s),
            },
            "t_run_seconds": {
                str(u): t_run_seconds(
                    u, float(np.quantile(steps, 0.90)), float(evals.mean()),
                    start_save_s)
                for u in (500, 1000, 1500, 3000)},
            "measured_at": _utc_now(),
        }
        write_json(self.out_root / "timing_probe.json", probe)
        return probe

    # -- start-frame sampling manifest --------------------------------------

    def _write_start_frame_sampling_manifest(self) -> dict:
        """Plan §5.2/§8: cycles, sample ids and sampler seeds for fold 0."""
        from pes2ts_core.pesn.offline.data import key_atom_dataset as lane_b
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            _sample_id, _split_group_reactions)

        split_path = self._p("split_manifest",
                             "data/manifests/v5/split_manifest.json")
        feature_root = self._p("feature_root", "outputs/pesn/v5/features")
        main_d0 = float(self.config.get("d0_angstrom", 0.08))
        d0_tag = f"{main_d0:.2f}"
        train_groups, _dev = _split_group_reactions(split_path, 0)
        pools = lane_b.build_pools(
            str(feature_root), train_groups,
            max_per_branch_cycle=int(
                self.config.get("max_start_frames_per_branch_cycle", 16)),
            d0_tag=d0_tag)
        groups_out: dict = {}
        n_entries = 0
        for gid, branches in pools.items():
            entry: dict = {}
            for branch, cycles in dict(branches).items():
                entry[str(branch)] = [
                    [{"sample_id": _sample_id(si, li), "q_weight": float(q)}
                     for (si, li, q) in cycle]
                    for cycle in cycles]
                n_entries += sum(len(c) for c in cycles)
            groups_out[str(gid)] = entry
        doc = {
            "schema_version": "pesn_v5_start_frame_sampling_v1",
            "fold": 0,
            "d0": main_d0,
            "max_start_frames_per_branch_cycle": int(
                self.config.get("max_start_frames_per_branch_cycle", 16)),
            "sampler_seed_rule": ("KeyAtomCycleSampler seeded per run with "
                                  "the run seed; cycles rotate by "
                                  "deterministic per-(group,branch) offsets "
                                  "(plan §5.2)"),
            "seeds": list(self.config.get("seeds") or [17, 29, 43]),
            "n_pool_entries": n_entries,
            "groups": groups_out,
        }
        write_json(self.out_root / "start_frame_sampling_manifest.json", doc)
        return doc

    # -- Oracle/Classic baseline table (prepare-audit, 65–80 min) -----------

    def _baseline_table(self) -> dict:
        """Classic (M0) + zero baseline on the fold-0 dev dense set."""
        import numpy as np
        import torch

        from pes2ts_core.pesn.models.key_atom_classic import classic_pair_driver
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            load_lane_b_pools, load_lane_b_view)

        split_path = str(self._p(
            "split_manifest", "data/manifests/v5/split_manifest.json"))
        main_d0 = float(self.config.get("d0_angstrom", 0.08))
        pools = load_lane_b_pools(self.config, fold=0, d0=main_d0,
                                  split_path=split_path)
        view, collate = load_lane_b_view(
            config=self.config, samples=list(pools["dev_dense"]), fold=0,
            d0=main_d0, anchor="on", split_path=split_path, device="cpu")
        per_sample: list[dict] = []
        batch_size = int(self.config.get("batch_size", 32))
        for i in range(0, len(view), batch_size):
            chunk = [view[j] for j in
                     range(i, min(i + batch_size, len(view)))]
            if not chunk:
                continue
            batch = collate(chunk)
            y = batch.y_teacher
            u_cls, info = classic_pair_driver(batch)
            key_ptr = batch.key_ptr.cpu().numpy()
            near = (batch.near_ts.cpu().numpy()
                    if batch.near_ts is not None
                    else np.ones(len(key_ptr) - 1))
            groups = (batch.reaction_ids if batch.reaction_ids
                      else tuple("unknown" for _ in range(len(key_ptr) - 1)))
            y_np = y.cpu().numpy()
            uc_np = u_cls.cpu().numpy()
            for b in range(len(key_ptr) - 1):
                lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
                if hi <= lo:
                    continue
                per_sample.append({
                    "group_id": str(groups[b]),
                    "u": uc_np[lo:hi], "y": y_np[lo:hi],
                    "u_classic": uc_np[lo:hi],
                    "degenerate": bool(info["degenerate_frames"][b]),
                    "near_ts": float(near[b]),
                })
        g52 = evaluate_g52(
            predictions={"raw": [s["u"] for s in per_sample],
                         "bounded": [s["u"] for s in per_sample]},
            teacher=[s["y"] for s in per_sample],
            groups=[s["group_id"] for s in per_sample],
            classic=[s["u_classic"] for s in per_sample],
            near_ts=[s["near_ts"] for s in per_sample],
            bootstrap_draws=200, bootstrap_seed=20261018)
        table = {
            "schema_version": "pesn_v5_baseline_table_v1",
            "fold": 0,
            "d0": main_d0,
            "baseline": "classic_pair_driver (M0, 0-param) vs zero action",
            "n_dev_dense_samples": len(per_sample),
            "classic": {
                "aggregate": g52["raw"]["aggregate"],
                "per_cluster": g52["raw"]["per_cluster"],
                "undefined": g52["raw"]["undefined"],
            },
            "measured_at": _utc_now(),
        }
        write_json(self.out_root / "baseline_table.json", table)
        return table

    # -- Oracle static ceiling (prepare-audit; crossfit precondition) -------

    def _oracle_static_ceiling(self) -> dict:
        """G5.4 oracle ceiling on the fold-0 dev dense set (main d0)."""
        import numpy as np
        import torch  # noqa: F401  (batch tensors are torch)

        from pes2ts_core.pesn.contracts.key_atom import load_key_atom_task
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            load_lane_b_pools, load_lane_b_view)

        split_path = str(self._p(
            "split_manifest", "data/manifests/v5/split_manifest.json"))
        main_d0 = float(self.config.get("d0_angstrom", 0.08))
        task_root = self._p("task_root", "outputs/pesn/v5/tasks")
        pools = load_lane_b_pools(self.config, fold=0, d0=main_d0,
                                  split_path=split_path)
        view, collate = load_lane_b_view(
            config=self.config, samples=list(pools["dev_dense"]), fold=0,
            d0=main_d0, anchor="on", split_path=split_path, device="cpu")
        tasks: dict = {}
        samples: list[dict] = []
        teachers: list[np.ndarray] = []
        batch_size = int(self.config.get("batch_size", 32))
        for i in range(0, len(view), batch_size):
            chunk = [view[j] for j in
                     range(i, min(i + batch_size, len(view)))]
            if not chunk:
                continue
            batch = collate(chunk)
            ptr = batch.ptr.cpu().numpy()
            key_ptr = batch.key_ptr.cpu().numpy()
            y_np = batch.y_teacher.cpu().numpy()
            pos_np = batch.pos.cpu().numpy()
            pos0_np = batch.pos0.cpu().numpy()
            z_np = batch.z_atomic.cpu().numpy()
            key_index = batch.key_index.cpu().numpy()
            pair_index = batch.pair_index.cpu().numpy()
            pair_edit = batch.pair_edit.cpu().numpy()
            branches = (batch.branch.cpu().numpy()
                        if batch.branch is not None
                        else np.zeros(len(key_ptr) - 1, dtype=np.int64))
            for b in range(len(key_ptr) - 1):
                rid = str(batch.reaction_ids[b])
                branch = int(branches[b])
                if rid not in tasks:
                    tasks[rid] = load_key_atom_task(
                        task_root / f"{rid}.task.json")
                task = tasks[rid]
                n_lo, n_hi = int(ptr[b]), int(ptr[b + 1])
                k_lo, k_hi = int(key_ptr[b]), int(key_ptr[b + 1])
                key_rows = [int(v) - n_lo for v in key_index[k_lo:k_hi]]
                k_pairs: list[tuple[int, int]] = []
                for col in range(pair_index.shape[1]):
                    a, c = int(pair_index[0, col]), int(pair_index[1, col])
                    # pair rows live on the NODE axis: segment by the
                    # molecule's node span [n_lo, n_hi) — NEVER the key-axis
                    # span (k_lo/k_hi index the packed key axis and would
                    # misassign pairs across samples)
                    if n_lo <= a < n_hi and n_lo <= c < n_hi and \
                            float(pair_edit[col]) != 0.0:
                        k_pairs.append((a - n_lo, c - n_lo))
                samples.append({
                    "group_id": rid,
                    "pos": pos_np[n_lo:n_hi],
                    "pos_start": pos0_np[n_lo:n_hi],
                    "atomic_numbers": [int(v) for v in z_np[n_lo:n_hi]],
                    "key_rows": key_rows,
                    "k_pairs": k_pairs,
                    "start_bonds": [tuple(map(float, bnd))
                                    for bnd in task.start_bonds(branch)],
                })
                teachers.append(y_np[k_lo:k_hi])
        result = oracle_static_ceiling(samples, teachers)
        write_json(self.out_root / "oracle_static_ceiling.json", result)
        return result

    # -- Stage 0: prepare-audit ---------------------------------------------

    def _stage_prepare_audit(self, window: StageWindow) -> str:
        checkpoints: list[dict] = []

        def _checkpoint(name: str, t_start: float, **extra) -> None:
            checkpoints.append({
                "checkpoint": name,
                "at_unix": float(self._clock()),
                "elapsed_s": time.time() - t_start, **extra})

        # 0–15: split freeze (parent hash asserts; REFUSED when sources
        # are missing)
        t_stage = time.time()
        selection = self._p(
            "fit50_selection",
            "outputs/pesn/direct_egnn_v1_20261015/splits/"
            "selection_direct_fit50.json")
        lineage = PROJECT_ROOT / (
            "outputs/pesn/dual_ts_v2_20261015/splits/"
            "dual_ts_split_lineage_v1.json")
        _require(selection.is_file(), "FIT50_SELECTION_MISSING",
                 f"parent selection not found: {selection}")
        _require(lineage.is_file(), "DUAL_TS_LINEAGE_MISSING",
                 f"parent lineage not found: {lineage}")
        split_out = self._p("split_manifest",
                            "data/manifests/v5/split_manifest.json")
        if self.force or not split_out.is_file():
            proc = self._spawn(
                [PYTHON, "-m", "pes2ts_core.pesn.offline.splits.v5_fold",
                 "--fit50-selection", str(selection),
                 "--lineage", str(lineage),
                 "--output", str(split_out), "--write"],
                stage="prepare-audit", window=window)
            _require(proc.returncode == 0, "SPLIT_FREEZE_FAILED",
                     f"v5_fold exited {proc.returncode}")
            _require(split_out.is_file(), "SPLIT_FREEZE_MISSING",
                     "v5_fold --write produced no manifest")
        _checkpoint("split_freeze", t_stage,
                    split_sha256=_sha256_file(split_out))

        self.ensure_manifest()
        self._verify_parents("prepare-audit")

        # 15–45: registration/teacher/conflict audit (guarded, resumable)
        t_mark = time.time()
        from pes2ts_core.pesn.offline.data.key_atom_teacher import (
            KeyAtomTeacherError, prepare_audit)
        try:
            progress = prepare_audit(self.config, allow_truth=self.allow_truth)
        except KeyAtomTeacherError as exc:
            self._record_stage("prepare-audit", "rejected_data",
                               detail=f"{exc.code}: {exc.detail}")
            raise SprintRefused(exc.code, exc.detail) from exc
        _checkpoint("prepare_audit", t_mark,
                    n_reactions=len(progress.get("reactions") or {}))

        # 45–65: teacher completion check (3 d0 × shards; resumable)
        t_mark = time.time()
        teacher_root = self._p("teacher_root", "outputs/pesn/v5/teacher")
        selection_doc = read_json(selection)
        reaction_ids = [str(r) for r in
                        (selection_doc.get("reaction_ids") or [])]
        d0_values = tuple(self.config.get("d0_ablation") or (0.05, 0.08, 0.12))
        teacher_counts: dict[str, int] = {}
        missing_shards: list[str] = []
        for d0 in d0_values:
            tag = f"{float(d0):g}"
            n_found = 0
            for rid in reaction_ids:
                for branch in (0, 1):
                    shard = (teacher_root / f"d0_{tag}"
                             / f"{rid}.branch{branch}.npz")
                    if shard.is_file():
                        n_found += 1
                    else:
                        missing_shards.append(str(shard))
            teacher_counts[tag] = n_found
        expected = len(reaction_ids) * 2
        teacher_complete = all(v == expected for v in teacher_counts.values())
        _checkpoint("teacher_completion", t_mark,
                    counts=teacher_counts, expected_per_d0=expected,
                    complete=teacher_complete)
        if not teacher_complete:
            self._record_stage(
                "prepare-audit", "rejected_data",
                detail=f"teacher shards incomplete: {teacher_counts} "
                       f"(expected {expected} per d0)",
                extra={"checkpoints": checkpoints,
                       "missing_shards": missing_shards[:20]})
            raise SprintRefused(
                "TEACHER_INCOMPLETE",
                f"teacher shards incomplete: {teacher_counts}")

        # 65–80: feature precompute (all 3 d0) + Oracle/Classic baseline
        # table + G5.1 evidence (pytest subprocess; fail -> rejected_gate)
        t_mark = time.time()
        from pes2ts_core.pesn.offline.data.key_atom_dataset import (
            KeyAtomDatasetError, precompute_features)
        conflict_weights_path = self._p(
            "teacher_root", "outputs/pesn/v5/teacher").parent \
            / "conflict_weights.json"
        conflict_map: dict[str, float] = {}
        if conflict_weights_path.is_file():
            raw_weights = (read_json(conflict_weights_path)
                           or {}).get("weights") or {}
            for rid, branches in dict(raw_weights).items():
                branch_values = [float(w) for w in dict(branches).values()]
                if branch_values and min(branch_values) == 0.0:
                    # a zero-weight branch must not silently vanish: keep
                    # the reaction at its minimum (0) so the loader applies
                    # the conflict exclusion
                    conflict_map[str(rid)] = 0.0
                elif branch_values and min(branch_values) < 1.0:
                    conflict_map[str(rid)] = min(branch_values)
        pc_config = {
            "root": str(PROJECT_ROOT),
            "native_root": self.paths.get("native_root",
                                          "outputs/pesn/v5/native"),
            "teacher_root": self.paths.get("teacher_root",
                                           "outputs/pesn/v5/teacher"),
            "task_root": self.paths.get("task_root",
                                        "outputs/pesn/v5/tasks"),
            "feature_root": self.paths.get("feature_root",
                                           "outputs/pesn/v5/features"),
            "d0_values": [float(d) for d in d0_values],
            "conflict_weights": conflict_map,
        }
        try:
            precompute = precompute_features(pc_config)
        except KeyAtomDatasetError as exc:
            self._record_stage("prepare-audit", "rejected_data",
                               detail=str(exc))
            raise SprintRefused("PRECOMPUTE_FAILED", str(exc)) from exc
        baseline = self._baseline_table()
        oracle = self._oracle_static_ceiling()
        g51_proc = subprocess.run(
            [PYTHON, "-m", "pytest", *G51_TEST_FILES, "-q"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        g51_pass = g51_proc.returncode == 0
        g51_summary = {
            "returncode": g51_proc.returncode,
            "stdout_tail": g51_proc.stdout.strip().splitlines()[-1:]
            if g51_proc.stdout.strip() else [],
            "files": list(G51_TEST_FILES),
        }
        _checkpoint("features_and_evidence", t_mark,
                    n_shards=len(precompute.shard_paths),
                    n_samples=precompute.n_samples,
                    n_eligible=precompute.n_eligible,
                    feature_spec=precompute.feature_spec_path,
                    g51_pass=g51_pass)
        if not g51_pass:
            self._record_stage(
                "prepare-audit", "rejected_gate",
                detail="G5.1 fixture suite failed",
                extra={"checkpoints": checkpoints, "g51": g51_summary})
            raise SprintRefused("G51_FAILED",
                                "G5.1 math-gate fixture suite failed")

        # 80–90: real timing probe + start-frame sampling manifest
        t_mark = time.time()
        try:
            probe = self._timing_probe()
        except SprintRefused as exc:
            self._record_stage("prepare-audit", "rejected_data",
                               detail=str(exc),
                               extra={"checkpoints": checkpoints})
            raise
        sampling = self._write_start_frame_sampling_manifest()
        _checkpoint("timing_probe", t_mark,
                    t_step_p90_s=probe["t_step_p90_s"],
                    t_eval_s=probe["t_eval_s"],
                    t_start_save_s=probe["t_start_save_s"],
                    n_pool_entries=sampling["n_pool_entries"])

        # g_v_map activation (idempotent) + re-render
        activation = self._activate_gv_campaign()
        _checkpoint("gv_activation", t_stage, **activation)

        # Stage-0 has no dedicated "complete_*" vocabulary value; its
        # outcome is recorded as a checkpoint summary + a decision row, and
        # the sprint status stays unset until label-screen produces one.
        summary = {
            "checkpoints": checkpoints,
            "teacher_counts": teacher_counts,
            "baseline_table": str(self.out_root / "baseline_table.json"),
            "oracle_static": {"status": oracle.get("status"),
                              "action_representation_flagged":
                                  oracle.get("action_representation_flagged")},
            "g51_pass": g51_pass,
            "g51": g51_summary,
            "timing_probe": str(self.out_root / "timing_probe.json"),
            "gv_activation": activation,
        }
        write_json(self.out_root / "prepare_audit_summary.json", summary)
        self.append_decision({
            "stage": "prepare-audit",
            "decision": "prepare_audit_complete",
            "g51_pass": g51_pass,
            "oracle_static_status": oracle.get("status"),
            "t_step_p90_s": probe["t_step_p90_s"],
            "t_eval_s": probe["t_eval_s"],
            "t_start_save_s": probe["t_start_save_s"],
            "wall_s": time.time() - t_stage,
        })
        if str(oracle.get("status")) == "ORACLE_STATIC_FAILED":
            self._record_stage(
                "prepare-audit", "rejected_gate",
                detail="Oracle static ceiling failed G5.4 — action "
                       "representation flagged (plan §6 G5.4)")
            raise SprintRefused(
                "ORACLE_STATIC_FAILED",
                "oracle teacher fails the G5.4 static protocol; revise "
                "the action contract before training (plan §6 G5.4)")
        print(f"[prepare-audit] complete: {len(reaction_ids)} reactions, "
              f"{precompute.n_samples} samples, t_step_p90="
              f"{probe['t_step_p90_s']:.3f}s")
        return "prepare-audit:complete"

    def _prepare_audit_summary(self) -> dict:
        path = self.out_root / "prepare_audit_summary.json"
        return read_json(path) if path.is_file() else {}

    def _activate_gv_campaign(self) -> dict:
        """Flip g_v_map V5_SPRINT proposed→active (idempotent) + re-render."""
        map_path = PROJECT_ROOT / "config" / "registry" / "g_v_map.yaml"
        text = map_path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        in_block = False
        changed = False
        already_active = False
        for line in lines:
            if line.startswith("  V5_SPRINT:"):
                in_block = True
                out.append(line)
                continue
            if in_block and line.startswith("  ") and not line.startswith("   "):
                in_block = False  # next campaign key
            if in_block and line.strip() == "status: active":
                already_active = True
            if in_block and line.strip() == "status: proposed":
                out.append(line.replace("status: proposed", "status: active"))
                changed = True
                continue
            out.append(line)
        if changed:
            map_path.write_text("".join(out), encoding="utf-8")
        if changed or already_active:
            proc = subprocess.run(
                [PYTHON, str(PROJECT_ROOT / "scripts" / "render_gv_map.py"),
                 "--write"],
                cwd=str(PROJECT_ROOT))
            _require(proc.returncode == 0, "GV_RENDER_FAILED",
                     f"render_gv_map.py exited {proc.returncode}")
        return {"changed": changed, "already_active": already_active,
                "map_path": str(map_path)}

    # -- Stage 1: label-screen ----------------------------------------------

    def _stage_label_screen(self, window: StageWindow) -> str:
        self._verify_parents("label-screen")
        probe = read_json(self.out_root / "timing_probe.json")
        ladder = [int(v) for v in
                  (self.sprint.get("downgrade_ladder") or [1500, 1000, 500])]
        tier1 = dict(self.sprint.get("tier1") or {})
        budget = resolve_screen_updates(
            probe, ladder=ladder, n_runs=len(SCREEN_ARMS),
            budget_minutes=float(tier1.get("max_total_minutes", 108)))
        self.append_decision({
            "stage": "label-screen", "decision": "screen_budget",
            **budget})
        if not budget["feasible"]:
            self._record_stage(
                "label-screen", "censored_budget",
                detail="even the smallest ladder rung exceeds the 108-min "
                       "six-arm budget; no screen run started",
                extra={"budget": budget})
            return "censored_budget"
        updates = int(budget["updates"])
        fold = int(tier1.get("fold", 0))
        seed = int(tier1.get("seed", 17))
        runs: list[dict] = []
        for arm in SCREEN_ARMS:
            try:
                runs.append(self._spawn_or_skip(
                    arm=arm, fold=fold, seed=seed, updates=updates,
                    mode="screen", stage="label-screen", window=window))
            except StageCensored as exc:
                self._record_stage(
                    "label-screen", "censored_budget",
                    detail=str(exc),
                    extra={"budget": budget, "runs": runs})
                return "censored_budget"

        rows: list[dict] = []
        for record in runs:
            run_dir = Path(record["run_dir"])
            arm = str(record["arm"])
            arm_cfg = (dict((self.sprint.get("arm_table") or {}).get(arm)
                            or {}))
            agg = self._final_eval_aggregate(run_dir) or {}
            raw = agg.get("raw") or {}
            run_status = str(record.get("status", "unknown"))
            censored = run_status not in ("complete",)
            curve = [entry.get("aggregate", {}).get("rmse_over_zero")
                     for entry in self._eval_series(run_dir)]
            stable = is_eval_curve_stable(
                [v for v in curve if v is not None]) if curve else False
            rows.append({
                "arm_id": arm,
                "d0": float(arm_cfg.get("d0", 0.08)),
                "anchor": str(arm_cfg.get("anchor", "off")),
                "fold": fold, "seed": seed,
                "updates": updates,
                "status": run_status,
                "censored": censored,
                "rmse_over_zero": raw.get("rmse_over_zero"),
                "rmse_over_classic": raw.get("rmse_over_classic"),
                "mae": raw.get("mae"),
                "rmse": raw.get("rmse"),
                "cos_eff_median": raw.get("cos_eff_median"),
                "near_ts_cos": raw.get("cos_eff_near_ts_median"),
                "bounded_rmse_over_zero": (agg.get("bounded") or {}).get(
                    "rmse_over_zero"),
                "bounded_cos_eff_near_ts_median":
                    (agg.get("bounded") or {}).get(
                        "cos_eff_near_ts_median"),
                "static_violation_rate": self._arm_static_violation_rate(
                    run_dir),
                "curve_stable": stable,
                "inconclusive": (not stable) or censored,
                "wall_s": record.get("wall_s"),
                "gpu_memory_mib": record.get("gpu_memory_mib"),
                "run_dir": str(run_dir),
            })
        ranked = rank_screen_arms(rows)
        for item in ranked:
            item.pop("_ranked", None)
        exploratory = bool(budget["exploratory_only"])
        matrix = {
            "schema_version": "pesn_v5_label_recipe_matrix_v1",
            "budget": budget,
            "ranking_rule": ("per-span cluster-median RMSE/zero (lower); "
                             "gap<0.02 -> near-TS cos (higher) -> static "
                             "violation rate (lower); raw MSE never "
                             "compared across d0 spans (plan §5.3)"),
            "rows": ranked,
            "top2": [r["arm_id"] for r in ranked
                     if r.get("rank") is not None][:2],
            "exploratory_only": exploratory,
            "measured_at": _utc_now(),
        }
        write_json(self.out_root / "label_recipe_matrix.json", matrix)
        self._write_ablation_table(ranked, updates=updates,
                                   stage="label-screen")
        all_terminal = all(r.get("status") in ("complete", "censored_screen")
                           for r in rows)
        if not all_terminal:
            status = "censored_budget"
        elif exploratory:
            status = "insufficient_evidence"
        else:
            status = "complete_label_screen"
        self._record_stage(
            "label-screen", status,
            detail=(f"screen complete ({len(rows)} arms, {updates} updates)"
                    if status == "complete_label_screen" else
                    "screen incomplete or exploratory-only budget"),
            extra={"budget": budget, "top2": matrix["top2"]})
        return status

    @staticmethod
    def _eval_series(run_dir: Path) -> list[dict]:
        eval_path = Path(run_dir) / "eval.jsonl"
        if not eval_path.is_file():
            return []
        out: list[dict] = []
        for line in eval_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _write_ablation_table(self, rows: Sequence[Mapping], *,
                              updates: int, stage: str) -> None:
        import pandas as pd

        frame = pd.DataFrame([dict(r) for r in rows])
        frame["updates_budget"] = int(updates)
        frame["stage"] = stage
        frame["censored"] = frame.get("censored", pd.Series(
            [False] * len(frame))).astype(bool)
        frame.to_parquet(self.out_root / "ablation_table.parquet",
                         index=False)

    # -- Stage 2: recipe-recheck --------------------------------------------

    def _recheck_decision(self) -> dict | None:
        path = self.out_root / "recipe_recheck_decision.json"
        return read_json(path) if path.is_file() else None

    def _stage_recipe_recheck(self, window: StageWindow) -> str:
        self._verify_parents("recipe-recheck")
        matrix = read_json(self.out_root / "label_recipe_matrix.json")
        top2 = [str(a) for a in (matrix.get("top2") or [])]
        _require(len(top2) == 2, "TOP2_MISSING",
                 f"label-screen produced top2={top2!r}; cannot recheck")
        probe = read_json(self.out_root / "timing_probe.json")
        tier2 = dict(self.sprint.get("tier2") or {})
        ladder = [int(v) for v in
                  (self.sprint.get("downgrade_ladder") or [1500, 1000, 500])]
        budget = resolve_recheck_updates(
            probe, window_minutes=150.0, n_new_runs=10,
            ladder=ladder)
        self.append_decision({
            "stage": "recipe-recheck", "decision": "recheck_budget", **budget})
        if not budget["feasible"]:
            self._record_stage(
                "recipe-recheck", "censored_budget",
                detail="10 new runs cannot fit the 150-min window even at "
                       "the smallest ladder rung; no stable winner declared",
                extra={"budget": budget})
            return "censored_budget"
        updates = int(budget["updates"])
        folds = [int(v) for v in (tier2.get("folds") or [0, 1])]
        seeds = [int(v) for v in (tier2.get("seeds") or [17, 29, 43])]
        cells: list[dict] = []
        for arm in top2:
            for fold in folds:
                for seed in seeds:
                    try:
                        result = self._spawn_or_skip(
                            arm=arm, fold=fold, seed=seed, updates=updates,
                            mode="screen", stage="recipe-recheck",
                            window=window)
                    except StageCensored as exc:
                        self._record_stage(
                            "recipe-recheck", "censored_budget",
                            detail=str(exc),
                            extra={"budget": budget, "cells": cells})
                        return "censored_budget"
                    run_dir = Path(result["run_dir"])
                    agg = self._final_eval_aggregate(run_dir) or {}
                    raw = agg.get("raw") or {}
                    cells.append({
                        "arm_id": arm, "fold": fold, "seed": seed,
                        "updates": updates,
                        "status": str(result.get("status", "unknown")),
                        "rmse_over_zero": raw.get("rmse_over_zero"),
                        "rmse_over_classic": raw.get("rmse_over_classic"),
                        "near_ts_cos": raw.get("cos_eff_near_ts_median"),
                        "static_violation_rate":
                            self._arm_static_violation_rate(run_dir),
                        "run_dir": str(run_dir),
                    })
        verdict = self._stability_verdict(cells, top2, updates,
                                           budget["stable_winner_allowed"])
        import pandas as pd

        pd.DataFrame(cells).to_parquet(
            self.out_root / "recipe_recheck.parquet", index=False)
        write_json(self.out_root / "recipe_recheck_decision.json", verdict)
        if not budget["stable_winner_allowed"]:
            status = "censored_budget"
            detail = (f"recheck budget downgraded to {updates} updates "
                      "(<1000): stability claim censored, no winner")
        elif all(str(c["status"]) not in ("complete", "censored_screen")
                 for c in cells):
            status = "censored_budget"
            detail = "no recheck cell completed"
        elif verdict["stable_recipe"] is not None:
            status = "complete_recipe_recheck"
            detail = (f"stable recipe {verdict['stable_recipe']} "
                      "(beats zero AND Classic on both folds in >=2/3 "
                      "seeds, near-TS not degraded)")
        else:
            status = "insufficient_evidence"
            detail = ("12 cells complete but no recipe is stable under the "
                      "plan §5.3 rule; scale-up stopped with attribution")
        self._record_stage("recipe-recheck", status, detail=detail,
                           extra={"budget": budget, "verdict": verdict})
        return status

    def _stability_verdict(self, cells: Sequence[Mapping],
                           top2: Sequence[str], updates: int,
                           stable_allowed: bool) -> dict:
        """Plan §5.3: stable iff beats zero AND Classic on BOTH folds in
        >=2/3 seeds AND near-TS not degraded (>= the pre-registered 0.50
        cosine floor)."""
        per_recipe: dict[str, dict] = {}
        for arm in top2:
            seeds_ok = 0
            per_seed: dict[str, dict] = {}
            for seed in sorted({int(c["seed"]) for c in cells}):
                seed_cells = [c for c in cells if c["arm_id"] == arm
                              and int(c["seed"]) == seed]
                folds_ok = 0
                for cell in seed_cells:
                    rz = cell.get("rmse_over_zero")
                    rc = cell.get("rmse_over_classic")
                    nt = cell.get("near_ts_cos")
                    beats = (rz is not None and rz < 1.0
                             and rc is not None and rc < 1.0
                             and nt is not None and nt >= RECHECK_NEAR_TS_FLOOR)
                    folds_ok += int(bool(beats))
                ok = folds_ok == len(seed_cells) and len(seed_cells) > 0
                seeds_ok += int(ok)
                per_seed[str(seed)] = {"folds_ok": folds_ok,
                                       "n_cells": len(seed_cells),
                                       "seed_ok": ok}
            stable = stable_allowed and seeds_ok >= 2
            per_recipe[arm] = {"seeds_ok": seeds_ok, "n_seeds":
                               len(per_seed), "per_seed": per_seed,
                               "stable": stable}
        stable_recipes = [arm for arm in top2
                          if per_recipe[arm]["stable"]]
        winner = None
        if stable_recipes:
            # deterministic tie: more seeds_ok, then better median
            # rmse_over_zero across that recipe's cells
            def _median_rz(arm: str) -> float:
                vals = sorted(float(c["rmse_over_zero"]) for c in cells
                              if c["arm_id"] == arm
                              and c.get("rmse_over_zero") is not None)
                return vals[len(vals) // 2] if vals else 1e9
            winner = sorted(stable_recipes,
                            key=lambda a: (-per_recipe[a]["seeds_ok"],
                                           _median_rz(a), a))[0]
        return {
            "schema_version": "pesn_v5_recipe_recheck_decision_v1",
            "rule": ("stable iff beats zero AND Classic on both folds in "
                     ">=2/3 seeds and near-TS cosine >= "
                     f"{RECHECK_NEAR_TS_FLOOR} (plan §5.3)"),
            "updates": int(updates),
            "stability_allowed": bool(stable_allowed),
            "top2": list(top2),
            "per_recipe": per_recipe,
            "stable_recipe": winner,
            "selection_exposed": True,
            "selection_note": ("fold-0 screen selection is exposed; OOF is "
                               "selection-exposed, not nested CV (plan §1.4)"),
            "decided_at": _utc_now(),
        }

    # -- Stage 3: crossfit ---------------------------------------------------

    def _stage_crossfit(self, window: StageWindow) -> str:
        self._verify_parents("crossfit")
        verdict = self._recheck_decision() or {}
        recipe = str(verdict.get("stable_recipe"))
        probe = read_json(self.out_root / "timing_probe.json")
        tier3 = dict(self.sprint.get("tier3") or {})
        formal_updates = int(tier3.get("updates", 3000))
        mode_plan = resolve_crossfit_mode(
            probe, window_minutes=float(tier3.get("window_minutes", 150.0)),
            formal_updates=formal_updates)
        self.append_decision({
            "stage": "crossfit", "decision": "crossfit_feasibility",
            **mode_plan, "recipe": recipe})
        if not mode_plan["feasible"]:
            self._record_stage(
                "crossfit", "insufficient_evidence",
                detail="formal OOF does not fit the 150-min window even at "
                       "5 folds x 1 seed; development results archived, "
                       "OOF explicitly deferred (plan §5.4)",
                extra={"mode_plan": mode_plan})
            return "insufficient_evidence"
        arm_cfg = dict((self.sprint.get("arm_table") or {}).get(recipe)
                       or {})
        d0 = float(arm_cfg.get("d0", 0.08))
        anchor = str(arm_cfg.get("anchor", "on")).lower()
        seeds = [int(v) for v in (tier3.get("seeds") or [17, 29, 43])]
        seeds = seeds[:int(mode_plan["n_seeds"])]
        n_folds = int(tier3.get("folds", 5))
        runs: list[dict] = []
        for fold in range(n_folds):
            for seed in seeds:
                try:
                    runs.append(self._spawn_or_skip(
                        arm=recipe, fold=fold, seed=seed,
                        updates=formal_updates, mode="formal",
                        stage="crossfit", window=window))
                except StageCensored as exc:
                    self._record_stage(
                        "crossfit", "censored_budget",
                        detail=str(exc),
                        extra={"mode_plan": mode_plan, "runs": runs})
                    return "censored_budget"
        complete_runs = [r for r in runs
                         if str(r.get("status")) == "complete"]
        if len(complete_runs) < len(runs):
            self._record_stage(
                "crossfit", "censored_budget",
                detail=f"{len(complete_runs)}/{len(runs)} formal runs "
                       "completed before the window closed",
                extra={"mode_plan": mode_plan, "runs": runs})
            return "censored_budget"
        export = self._export_oof(
            recipe=recipe, d0=d0, anchor=anchor, seeds=seeds,
            n_folds=n_folds, formal_updates=formal_updates)
        status = validate_status(str(mode_plan["status_on_completion"]))
        self._record_stage(
            "crossfit", status,
            detail=(f"formal crossfit complete: {len(runs)} runs, OOF "
                    f"exported to {export['path']} "
                    f"({export['n_rows']} rows, "
                    f"{export['n_censored']} censored)"),
            extra={"mode_plan": mode_plan, "oof": export,
                   "selection_exposed": True})
        return status

    def _export_oof(self, *, recipe: str, d0: float, anchor: str,
                    seeds: Sequence[int], n_folds: int,
                    formal_updates: int) -> dict:
        """Assemble ``move_proposal_v5_draft1.parquet`` (plan §1.4).

        Every eligible dense sample gets exactly ONE row per (recipe, seed)
        from the final checkpoint of its OUTER-fold run; a missing row is
        an explicit censored entry (eligible=False + reason), NEVER
        zero-filled.  z is archived float32[64] and not part of the V6
        interface (plan §1.4/§5.5).
        """
        import numpy as np
        import pandas as pd
        import torch

        from pes2ts_core.pesn.contracts.key_atom import load_key_atom_task
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            bounded_projection, load_lane_b_pools, load_lane_b_view)
        from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN

        split_path = self._p("split_manifest",
                             "data/manifests/v5/split_manifest.json")
        split_doc = read_json(split_path)
        group_family: dict[str, str] = {}
        for comp in (split_doc.get("components") or []):
            family = str(comp.get("family", "unknown"))
            for gid in (comp.get("group_ids") or []):
                group_family[str(gid)] = family
        task_root = self._p("task_root", "outputs/pesn/v5/tasks")
        feature_root = self._p("feature_root", "outputs/pesn/v5/features")
        frame_contract_sha = _sha256_file(self._p(
            "frame_contract", "data/manifests/v5/frame_contract.json"))
        config_sha = _sha256_file(self.config_path)
        split_sha = _sha256_file(split_path)
        selection_scope_hash = hashlib.sha256(
            json.dumps({"recipe": recipe, "seeds": list(seeds),
                        "folds": n_folds,
                        "selection_source": "fold-0 label screen + recheck"},
                       sort_keys=True).encode()).hexdigest()

        rows: list[dict] = []
        tasks: dict = {}
        for fold in range(n_folds):
            pools = load_lane_b_pools(self.config, fold=fold, d0=d0,
                                      split_path=str(split_path))
            dev_rows = list(pools["dev_dense"])
            train_groups, _dev = self._split_groups(fold)
            training_group_hash = hashlib.sha256(
                json.dumps(sorted(train_groups), sort_keys=True).encode()
            ).hexdigest()
            for seed in seeds:
                run_dir = self._run_dir(
                    self._run_id(recipe, fold, seed, "formal"))
                ckpt_path = run_dir / "checkpoints" / "checkpoint_final.pt"
                model_hash = _sha256_file(ckpt_path)
                expected_ids = {str(r["sample_id"]) for r in dev_rows}
                produced: set[str] = set()
                if ckpt_path.is_file():
                    device = torch.device(
                        "cuda:0" if torch.cuda.is_available() else "cpu")
                    model = StarEGNN().to(torch.float32).to(device)
                    doc = torch.load(ckpt_path, map_location="cpu")
                    model.load_state_dict(doc["model"])
                    model.train(False)
                    view, collate = load_lane_b_view(
                        config=self.config, samples=dev_rows, fold=fold,
                        d0=d0, anchor=anchor, split_path=str(split_path),
                        device=device)
                    batch_size = int(self.config.get("batch_size", 32))
                    with torch.no_grad():
                        for i in range(0, len(view), batch_size):
                            indices = list(range(
                                i, min(i + batch_size, len(view))))
                            chunk = [view[j] for j in indices]
                            if not chunk:
                                continue
                            batch = collate(chunk)
                            u, z = model(batch)
                            u_b, _scale = bounded_projection(
                                u.float(),
                                batch.d0 if batch.d0 is not None else d0,
                                batch.key_ptr)
                            key_ptr = batch.key_ptr.cpu().numpy()
                            ptr = batch.ptr.cpu().numpy()
                            pos_np = batch.pos.cpu().numpy()
                            pos0_np = batch.pos0.cpu().numpy()
                            y_np = (batch.y_teacher.cpu().numpy()
                                    if batch.y_teacher is not None else None)
                            key_index = batch.key_index.cpu().numpy()
                            z_np = z.cpu().numpy()
                            u_np = u.float().cpu().numpy()
                            ub_np = u_b.cpu().numpy()
                            near = (batch.near_ts.cpu().numpy()
                                    if batch.near_ts is not None
                                    else np.ones(len(key_ptr) - 1))
                            for b in range(len(key_ptr) - 1):
                                row_index = indices[b]
                                sample_row = dev_rows[row_index]
                                sid = str(sample_row["sample_id"])
                                rid = str(batch.reaction_ids[b])
                                gid = str(batch.group_ids[b]) if batch.group_ids \
                                    else rid
                                if rid not in tasks:
                                    task_path = task_root / f"{rid}.task.json"
                                    tasks[rid] = (
                                        load_key_atom_task(task_path),
                                        _sha256_file(task_path))
                                task, task_sha = tasks[rid]
                                k_lo, k_hi = int(key_ptr[b]), int(key_ptr[b + 1])
                                n_lo, n_hi = int(ptr[b]), int(ptr[b + 1])
                                key_rows = [int(v) for v in
                                            key_index[k_lo:k_hi]]
                                u_raw_b = u_np[k_lo:k_hi].astype(np.float32)
                                u_bound_b = ub_np[k_lo:k_hi].astype(
                                    np.float32)
                                pos_b = pos_np[n_lo:n_hi].astype(np.float32)
                                pos0_b = pos0_np[n_lo:n_hi].astype(np.float32)
                                rows.append({
                                    "sample_id": sid,
                                    "reaction_id": rid,
                                    "group_id": gid,
                                    "family": group_family.get(gid, "unknown"),
                                    "branch": int(sample_row.get("branch", 0)),
                                    "fold": int(fold),
                                    "recipe": recipe,
                                    "seed": int(seed),
                                    "d0": float(d0),
                                    "model_version":
                                        "pesn_key_atom_star_egnn_v1",
                                    "config_sha256": config_sha,
                                    "checkpoint_sha256": model_hash,
                                    "split_sha256": split_sha,
                                    "training_group_sha256":
                                        training_group_hash,
                                    "task_sha256": task_sha,
                                    "atom_map_hash": hashlib.sha256(
                                        json.dumps(list(task.atom_map_ids),
                                                   sort_keys=True).encode()
                                    ).hexdigest(),
                                    "input_geometry_hash": hashlib.sha256(
                                        pos_b.tobytes()).hexdigest(),
                                    "x0_hash": hashlib.sha256(
                                        pos0_b.tobytes()).hexdigest(),
                                    "frame_contract_sha256":
                                        frame_contract_sha,
                                    "key_atom_ids": [int(v) for v in key_rows],
                                    "u_raw": [row.tolist() for row in u_raw_b],
                                    "u_bounded": [row.tolist()
                                                  for row in u_bound_b],
                                    "z": z_np[b].astype(np.float32).tolist(),
                                    "near_ts": float(near[b]),
                                    "eligible": True,
                                    "reason": "ok",
                                    "selection_exposed": True,
                                    "selection_scope_hash":
                                        selection_scope_hash,
                                    "formal_updates": int(formal_updates),
                                })
                                produced.add(sid)
                # explicit censored rows: never zero-filled (plan §1.4)
                for sample_row in dev_rows:
                    sid = str(sample_row["sample_id"])
                    if sid in produced:
                        continue
                    rid = str(sample_row.get("reaction_id", ""))
                    gid = str(sample_row.get("group_id", rid))
                    rows.append({
                        "sample_id": sid,
                        "reaction_id": rid,
                        "group_id": gid,
                        "family": group_family.get(gid, "unknown"),
                        "branch": int(sample_row.get("branch", 0)),
                        "fold": int(fold),
                        "recipe": recipe,
                        "seed": int(seed),
                        "d0": float(d0),
                        "model_version": "pesn_key_atom_star_egnn_v1",
                        "config_sha256": config_sha,
                        "checkpoint_sha256": model_hash,
                        "split_sha256": split_sha,
                        "training_group_sha256": training_group_hash,
                        "task_sha256": None,
                        "atom_map_hash": None,
                        "input_geometry_hash": None,
                        "x0_hash": None,
                        "frame_contract_sha256": frame_contract_sha,
                        "key_atom_ids": None,
                        "u_raw": None,
                        "u_bounded": None,
                        "z": None,
                        "near_ts": float(sample_row.get("q_weight", 1.0)),
                        "eligible": False,
                        "reason": "censored_missing_prediction",
                        "selection_exposed": True,
                        "selection_scope_hash": selection_scope_hash,
                        "formal_updates": int(formal_updates),
                    })
                expected_ids -= produced
        frame = pd.DataFrame(rows)
        path = self.out_root / "move_proposal_v5_draft1.parquet"
        frame.to_parquet(path, index=False)
        meta = {
            "schema_version": "pesn_v5_oof_export_v1",
            "path": str(path),
            "n_rows": int(len(frame)),
            "n_censored": int((~frame["eligible"].astype(bool)).sum()),
            "recipes": [recipe],
            "seeds": [int(s) for s in seeds],
            "folds": int(n_folds),
            "selection_exposed": True,
            "selection_scope_hash": selection_scope_hash,
            "z_policy": ("z float32[64] archived only; not part of the V6 "
                         "interface (plan §1.4/§5.5)"),
            "sha256": _sha256_file(path),
            "exported_at": _utc_now(),
        }
        write_json(self.out_root / "move_proposal_v5_draft1.meta.json", meta)
        return meta

    def _split_groups(self, fold: int) -> tuple[dict, dict]:
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            _split_group_reactions)
        return _split_group_reactions(
            self._p("split_manifest", "data/manifests/v5/split_manifest.json"),
            fold)

    # -- Stage 4: report -----------------------------------------------------

    def _stage_report(self, window: StageWindow) -> str:
        self._verify_parents("report")
        stages = (self.manifest or {}).get("stages") or {}
        measured: dict[str, str] = {
            k: str((v or {}).get("status", "unknown"))
            for k, v in stages.items()}
        sprint_status = self._overall_status(measured)

        # 20–45: sealed-eval request (typed REFUSED; registration echoed).
        # This sprint NEVER unseals (allow_unseal=False): even a complete
        # 15-run crossfit keeps the confirm set closed (plan §5.5).
        crossfit_complete = (
            str((stages.get("crossfit") or {}).get("status", ""))
            in ("complete_oof_3seed", "complete_oof_1seed"))
        crossfit_runs = 15 if (
            crossfit_complete and str((stages.get("crossfit")
                                       ).get("status")) == "complete_oof_3seed"
        ) else (5 if crossfit_complete else 0)
        sealed = sealed_eval_request(
            {"campaign_id": CAMPAIGN_ID,
             "config_sha256": _sha256_file(self.config_path),
             "split_manifest_sha256": _sha256_file(self._p(
                 "split_manifest", "data/manifests/v5/split_manifest.json")),
             "stage_statuses": measured,
             "sprint_policy": "v5_sprint_phase1_never_unseals"},
            crossfit_complete_runs=crossfit_runs,
            model_hash=None, config_hash=_sha256_file(self.config_path),
            protocol_hash=_sha256_file(self.config_path),
            unseal=self.unseal, allow_unseal=False)
        write_json(self.out_root / "sealed_eval.json", sealed)
        _require(sealed["status"] == "REFUSED", "SEALED_UNEXPECTED",
                 "sealed evaluation must stay REFUSED for this sprint "
                 "(plan §5.5 / ADR-0026 §4)")
        print(f"[report] sealed eval: {sealed['status']} "
              f"({sealed['code']}); reasons={sealed['reasons']}")

        # 45–60: collision / follower diagnostics + 60–80: grouped
        # bootstrap intervals from the winner's checkpoints (best-effort:
        # absent checkpoints degrade the sections to not_executed)
        gates: dict[str, dict] = {}
        artifacts: list[dict] = []
        g52_payload = self._g52_from_final_checkpoints()
        gates["G5.2"] = ({
            "status": "measured", "payload": g52_payload,
            "verdict": ("pass" if (g52_payload or {}).get("gate", {}).get(
                "pass") else "fail_or_insufficient"),
            "note": ("dev-fold dense re-validation from final checkpoints; "
                     "the 10-cluster dev fold cannot issue a formal G5.2 "
                     "pass (plan §5.3)")}
            if g52_payload else {
                "status": "not_executed", "verdict": "—",
                "note": "no final checkpoints available for re-validation"})
        static_diag = self._static_diagnostics()
        gates["G5.4"] = ({
            "status": "measured",
            "payload": {"g54": static_diag.get("g54"),
                        "follower_counterfactual":
                            static_diag.get("follower_counterfactual")},
            "verdict": ("pass" if static_diag.get("g54", {}).get(
                "gate", {}).get("pass") else "fail"),
            "note": ("strict-copy static protocol incl. bonded pairs; "
                     "follower branch is contract_counterfactual and NOT "
                     "counted in the pass (plan §6 G5.4)")}
            if static_diag else {
                "status": "not_executed", "verdict": "—",
                "note": "no checkpoints available for static diagnostics"})
        audit_summary = self._prepare_audit_summary()
        gates["G5.1"] = ({
            "status": "measured",
            "verdict": "pass" if audit_summary.get("g51_pass") else "fail",
            "payload": audit_summary.get("g51"),
            "note": "fixture suite run at prepare-audit (65–80 min)"}
            if audit_summary else {
                "status": "not_executed", "verdict": "—",
                "note": "prepare-audit summary missing"})
        oracle_doc = self._read_oracle_static()
        gates["oracle_static"] = ({
            "status": "measured",
            "verdict": oracle_doc.get("status"),
            "payload": oracle_doc,
            "note": ("teacher ceiling under the identical protocol; a "
                     "failure flags the action representation, not the "
                     "network (plan §6 G5.4)")}
            if oracle_doc else {
                "status": "not_executed", "verdict": "—",
                "note": "oracle static ceiling not computed"})
        gates["sealed_eval"] = {
            "status": "measured", "verdict": sealed["status"],
            "payload": sealed,
            "note": "typed REFUSED with registration echoed (plan §5.5)"}
        recheck_verdict = self._recheck_decision()
        gates["G5.3"] = ({
            "status": "measured",
            "payload": (static_diag or {}).get("g53"),
            "verdict": ("pass" if (static_diag or {}).get("g53", {}).get(
                "gate", {}).get("pass") else "fail_or_insufficient"),
            "note": ("exact finite-step Δd sign accuracy + SRE on dev "
                     "dense predictions; first-order J vs exact Δd "
                     "deviation reported (plan §6 G5.3)")}
            if static_diag and static_diag.get("g53") else {
                "status": "not_executed", "verdict": "—",
                "note": "no checkpoints available for G5.3 re-validation"})
        if recheck_verdict:
            artifacts.append({
                "name": "recipe_recheck_decision",
                "schema": "pesn_v5_recipe_recheck_decision_v1",
                "path": str(self.out_root / "recipe_recheck_decision.json"),
                "config_sha256": _sha256_file(self.config_path),
                "source": "stage recipe-recheck",
                "status": "measured"})
        matrix_path = self.out_root / "label_recipe_matrix.json"
        if matrix_path.is_file():
            artifacts.append({
                "name": "label_recipe_matrix",
                "schema": "pesn_v5_label_recipe_matrix_v1",
                "path": str(matrix_path),
                "config_sha256": _sha256_file(self.config_path),
                "source": "stage label-screen", "status": "measured"})
        for name, schema, key in (
                ("timing_probe", "pesn_v5_timing_probe_v1",
                 "timing_probe.json"),
                ("ablation_table", None, "ablation_table.parquet"),
                ("recipe_recheck", None, "recipe_recheck.parquet"),
                ("start_frame_sampling_manifest",
                 "pesn_v5_start_frame_sampling_v1",
                 "start_frame_sampling_manifest.json"),
                ("move_proposal_v5_draft1", "pesn_v5_oof_export_v1",
                 "move_proposal_v5_draft1.parquet"),
                ("scale_curriculum_next_run",
                 "pesn_v5_scale_curriculum_plan_v1",
                 "scale_curriculum_next_run.json")):
            path = self.out_root / key
            if path.is_file():
                artifacts.append({
                    "name": name, "schema": schema, "path": str(path),
                    "config_sha256": _sha256_file(self.config_path),
                    "source": "V5_SPRINT campaign",
                    "status": "measured"})

        # 60–80: scale/curriculum plan for the NEXT run (planned-only)
        scale_plan = {
            "schema_version": "pesn_v5_scale_curriculum_plan_v1",
            "status": "planned",
            "scales": [50, 200, 500],
            "plan": ("freeze Confirm/OOD + nested Fit50 ⊂ Fit200 ⊂ Fit500 "
                     "with a fixed extra dev-validation set; fixed winning "
                     "recipe; 16 start frames per branch; scale x 3 seeds; "
                     "compare validation-error vs updates AND vs wall "
                     "clock before any 2,000-cluster step (plan §5.6)"),
            "curriculum": ("only after a larger set shows a stable signal; "
                           "requested_step_v1 loss scale; A/B paired 6000-"
                           "update trials (plan §5.6)"),
            "note": ("planned-only — this sprint executes NO scale-up or "
                     "curriculum run (ADR-0026 §4)"),
        }
        if not self.dry_run:
            write_json(self.out_root / "scale_curriculum_next_run.json",
                       scale_plan)
            artifacts.append({
                "name": "scale_curriculum_next_run",
                "schema": "pesn_v5_scale_curriculum_plan_v1",
                "path": str(self.out_root / "scale_curriculum_next_run.json"),
                "config_sha256": _sha256_file(self.config_path),
                "source": "stage report", "status": "planned"})

        report_paths = write_gate_report(
            self.out_root,
            gates=gates, artifacts=artifacts,
            stage_statuses=measured, campaign_id=CAMPAIGN_ID,
            qc="QC not scheduled",
            config_sha256=_sha256_file(self.config_path),
            repo_root=PROJECT_ROOT)
        self._write_sprint_report_doc(measured, sealed, audit_summary,
                                      recheck_verdict, sprint_status)
        self.append_decision({
            "stage": "report", "decision": "report_written",
            "sprint_status": sprint_status,
            "sealed_status": sealed["status"],
            "artifacts": {k: v["path"] for k, v in report_paths.items()}})
        print(f"[report] sprint status: {sprint_status}; gate report at "
              f"{report_paths['gate_report_json']['path']}")
        return sprint_status

    @staticmethod
    def _overall_status(measured: Mapping[str, str]) -> str:
        """Highest actually-completed tier (plan §5.7 vocabulary)."""
        for status in ("complete_oof_3seed", "complete_oof_1seed",
                       "complete_recipe_recheck", "complete_label_screen"):
            if status in measured.values():
                return status
        for status in ("rejected_gate", "rejected_data", "censored_budget",
                       "insufficient_evidence"):
            if status in measured.values():
                return status
        return "insufficient_evidence"

    def _read_oracle_static(self) -> dict | None:
        path = self.out_root / "oracle_static_ceiling.json"
        return read_json(path) if path.is_file() else None

    def _g52_from_final_checkpoints(self) -> dict | None:
        """Dense dev re-validation (plan §5.4 report, 0–20 min) for the
        best available recipe across completed runs."""
        import numpy as np
        import torch

        from pes2ts_core.pesn.models.key_atom_classic import classic_pair_driver
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            bounded_projection, load_lane_b_pools, load_lane_b_view)
        from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN

        candidates = self._completed_recipe_folds()
        if not candidates:
            return None
        split_path = str(self._p(
            "split_manifest", "data/manifests/v5/split_manifest.json"))
        best: dict | None = None
        for recipe, fold, seed, mode, run_dir in candidates:
            arm_cfg = dict((self.sprint.get("arm_table") or {}).get(recipe)
                           or {})
            d0 = float(arm_cfg.get("d0", 0.08))
            anchor = str(arm_cfg.get("anchor", "on")).lower()
            ckpt = Path(run_dir) / "checkpoints" / "checkpoint_final.pt"
            if not ckpt.is_file():
                continue
            try:
                pools = load_lane_b_pools(self.config, fold=fold, d0=d0,
                                          split_path=split_path)
                view, collate = load_lane_b_view(
                    config=self.config, samples=list(pools["dev_dense"]),
                    fold=fold, d0=d0, anchor=anchor,
                    split_path=split_path, device="cpu")
                device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")
                model = StarEGNN().to(torch.float32).to(device)
                doc = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(doc["model"])
                model.train(False)
            except Exception:
                continue
            per_sample: list[dict] = []
            batch_size = int(self.config.get("batch_size", 32))
            with torch.no_grad():
                for i in range(0, len(view), batch_size):
                    chunk = [view[j] for j in
                             range(i, min(i + batch_size, len(view)))]
                    if not chunk:
                        continue
                    batch = collate(chunk)
                    u, _z = model(batch)
                    u_b, _s = bounded_projection(
                        u.float(),
                        batch.d0 if batch.d0 is not None else d0,
                        batch.key_ptr)
                    u_cls, info = classic_pair_driver(batch)
                    key_ptr = batch.key_ptr.cpu().numpy()
                    near = (batch.near_ts.cpu().numpy()
                            if batch.near_ts is not None
                            else np.ones(len(key_ptr) - 1))
                    groups = (batch.reaction_ids if batch.reaction_ids
                              else tuple("unknown"
                                         for _ in range(len(key_ptr) - 1)))
                    y_np = batch.y_teacher.cpu().numpy()
                    u_np = u.float().cpu().numpy()
                    ub_np = u_b.cpu().numpy()
                    uc_np = u_cls.cpu().numpy()
                    for b in range(len(key_ptr) - 1):
                        lo, hi = int(key_ptr[b]), int(key_ptr[b + 1])
                        if hi <= lo:
                            continue
                        per_sample.append({
                            "group_id": str(groups[b]),
                            "u": u_np[lo:hi], "y": y_np[lo:hi],
                            "u_bounded": ub_np[lo:hi],
                            "u_classic": uc_np[lo:hi],
                            "near_ts": float(near[b]),
                        })
            if not per_sample:
                continue
            g52 = evaluate_g52(
                predictions={
                    "raw": [s["u"] for s in per_sample],
                    "bounded": [s["u_bounded"] for s in per_sample]},
                teacher=[s["y"] for s in per_sample],
                groups=[s["group_id"] for s in per_sample],
                classic=[s["u_classic"] for s in per_sample],
                near_ts=[s["near_ts"] for s in per_sample])
            best = {
                "recipe": recipe, "fold": fold, "seed": seed, "mode": mode,
                "run_dir": str(run_dir),
                "gate": g52["raw"]["gate"],
                "raw": {"aggregate": g52["raw"]["aggregate"],
                        "bootstrap": g52["raw"]["bootstrap"],
                        "undefined": g52["raw"]["undefined"]},
                "bounded": {"aggregate": g52["bounded"]["aggregate"]},
                "per_cluster": g52["raw"]["per_cluster"],
            }
            break
        return best

    def _completed_recipe_folds(self) -> list[tuple]:
        """(recipe, fold, seed, mode, run_dir) over completed runs, winner
        first (from the recheck decision when available)."""
        verdict = self._recheck_decision() or {}
        order = [verdict.get("stable_recipe")] if verdict.get(
            "stable_recipe") else []
        order += [r for r in (verdict.get("top2") or []) if r not in order]
        out: list[tuple] = []
        runs_root = self.out_root / "runs"
        if not runs_root.is_dir():
            return out
        for recipe in order:
            for run_dir in sorted(runs_root.glob(f"{recipe}_f*_s*_*")):
                status = self._run_status(run_dir)
                if not status or str(status.get("status")) != "complete":
                    continue
                name = run_dir.name
                try:
                    fold = int(name.split("_f")[1].split("_")[0])
                    seed = int(name.split("_s")[1].split("_")[0])
                    mode = name.rsplit("_", 1)[-1]
                except (IndexError, ValueError):
                    continue
                out.append((str(recipe), fold, seed, mode, str(run_dir)))
        return out

    def _static_diagnostics(self) -> dict | None:
        """G5.3 + G5.4 + follower diagnostics on dev dense predictions;
        also drops ``static_diagnostics.json`` next to the run dir."""
        import numpy as np
        import torch

        from pes2ts_core.pesn.contracts.key_atom import load_key_atom_task
        from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
            evaluate_g53)
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            bounded_projection, load_lane_b_pools, load_lane_b_view)
        from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN

        candidates = self._completed_recipe_folds()
        if not candidates:
            return None
        split_path = str(self._p(
            "split_manifest", "data/manifests/v5/split_manifest.json"))
        task_root = self._p("task_root", "outputs/pesn/v5/tasks")
        recipe, fold, seed, mode, run_dir = candidates[0]
        arm_cfg = dict((self.sprint.get("arm_table") or {}).get(recipe) or {})
        d0 = float(arm_cfg.get("d0", 0.08))
        anchor = str(arm_cfg.get("anchor", "on")).lower()
        ckpt = Path(run_dir) / "checkpoints" / "checkpoint_final.pt"
        if not ckpt.is_file():
            return None
        try:
            pools = load_lane_b_pools(self.config, fold=fold, d0=d0,
                                      split_path=split_path)
            view, collate = load_lane_b_view(
                config=self.config, samples=list(pools["dev_dense"]),
                fold=fold, d0=d0, anchor=anchor, split_path=split_path,
                device="cpu")
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            model = StarEGNN().to(torch.float32).to(device)
            doc = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(doc["model"])
            model.train(False)
        except Exception:
            return None
        tasks: dict = {}
        samples: list[dict] = []
        bounded_preds: list[np.ndarray] = []
        g53_samples: list[dict] = []
        batch_size = int(self.config.get("batch_size", 32))
        with torch.no_grad():
            for i in range(0, len(view), batch_size):
                chunk = [view[j] for j in
                         range(i, min(i + batch_size, len(view)))]
                if not chunk:
                    continue
                batch = collate(chunk)
                u, _z = model(batch)
                u_b, _s = bounded_projection(
                    u.float(), batch.d0 if batch.d0 is not None else d0,
                    batch.key_ptr)
                ptr = batch.ptr.cpu().numpy()
                key_ptr = batch.key_ptr.cpu().numpy()
                pos_np = batch.pos.cpu().numpy()
                pos0_np = batch.pos0.cpu().numpy()
                z_np = batch.z_atomic.cpu().numpy()
                key_index = batch.key_index.cpu().numpy()
                pair_index = batch.pair_index.cpu().numpy()
                pair_edit = batch.pair_edit.cpu().numpy()
                ub_np = u_b.cpu().numpy()
                u_np = u.float().cpu().numpy()
                y_np = (batch.y_teacher.cpu().numpy()
                        if batch.y_teacher is not None
                        else np.zeros_like(u_np))
                branches = (batch.branch.cpu().numpy()
                            if batch.branch is not None
                            else np.zeros(len(key_ptr) - 1, dtype=np.int64))
                for b in range(len(key_ptr) - 1):
                    rid = str(batch.reaction_ids[b])
                    branch = int(branches[b])
                    if rid not in tasks:
                        tasks[rid] = load_key_atom_task(
                            task_root / f"{rid}.task.json")
                    task = tasks[rid]
                    n_lo, n_hi = int(ptr[b]), int(ptr[b + 1])
                    k_lo, k_hi = int(key_ptr[b]), int(key_ptr[b + 1])
                    key_rows = [int(v) - n_lo for v in key_index[k_lo:k_hi]]
                    k_pairs: list[tuple[int, int]] = []
                    for col in range(pair_index.shape[1]):
                        a, c = int(pair_index[0, col]), int(pair_index[1, col])
                        # node-axis segmentation (see _oracle_static_ceiling)
                        if n_lo <= a < n_hi and n_lo <= c < n_hi and \
                                float(pair_edit[col]) != 0.0:
                            k_pairs.append((a - n_lo, c - n_lo))
                    samples.append({
                        "group_id": rid,
                        "pos": pos_np[n_lo:n_hi],
                        "pos_start": pos0_np[n_lo:n_hi],
                        "atomic_numbers": [int(v) for v in z_np[n_lo:n_hi]],
                        "key_rows": key_rows,
                        "k_pairs": k_pairs,
                        "start_bonds": [tuple(map(float, bnd))
                                        for bnd in task.start_bonds(branch)],
                    })
                    bounded_preds.append(ub_np[k_lo:k_hi])
                    g53_samples.append({
                        "group_id": rid,
                        "reaction_id": rid,
                        "branch": branch,
                        "pos": pos_np[n_lo:n_hi],
                        "key_rows": key_rows,
                        "pairs": k_pairs,
                        "u": u_np[k_lo:k_hi],
                        "y": y_np[k_lo:k_hi],
                    })
        if not samples:
            return None
        g54 = evaluate_g54(samples, bounded_preds)
        follower = follower_counterfactual_diagnostics(samples, bounded_preds)
        g53 = evaluate_g53(g53_samples) if g53_samples else None
        doc = {
            "schema_version": "pesn_v5_static_diagnostics_v1",
            "recipe": recipe, "fold": fold, "seed": seed, "mode": mode,
            "g53": g53,
            "g54": g54,
            "follower_counterfactual": follower,
            "measured_at": _utc_now(),
        }
        write_json(Path(run_dir) / "static_diagnostics.json", doc)
        return doc

    def _write_sprint_report_doc(self, measured: Mapping[str, str],
                                 sealed: Mapping, audit: Mapping,
                                 recheck: Mapping | None,
                                 sprint_status: str) -> None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = PROJECT_ROOT / "docs" / "reports" / (
            f"PES2TS_V5第一阶段冲刺执行报告_{day}.md")
        lines = [
            "# PES2TS V5 第一阶段冲刺执行报告",
            "",
            f"> 生成：{_utc_now()}（campaign `{CAMPAIGN_ID}`）",
            f"> sprint status：`{validate_status(sprint_status)}`",
            "> 本报告只记录实际完成层级；封存评测未开启（REFUSED 已回显登记）。",
            "",
            "## 阶段状态（方案 §5.7 词汇）",
            "",
            "| stage | status |",
            "|---|---|",
        ]
        for stage in STAGE_ORDER:
            lines.append(f"| {stage} | {measured.get(stage, 'not_executed')} |")
        lines += [
            "",
            "## 关键实测",
            "",
            f"- G5.1 fixtures：{'pass' if audit.get('g51_pass') else 'not executed'}",
            f"- Oracle static ceiling：{((audit.get('oracle_static') or {}).get('status') or 'not executed')}",
            f"- 封存评测：{sealed.get('status')} (`{sealed.get('code')}`)",
        ]
        if recheck:
            lines.append(f"- 复核稳定配方：{recheck.get('stable_recipe') or '无（insufficient_evidence）'}")
        lines += [
            "",
            "详见 `outputs/pesn/v5/sprint/gate_report.md`（measured / planned / "
            "not_executed / censored 分区；无 QC 计划处标注 `QC not scheduled`）。",
            "",
        ]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        self.append_decision({
            "stage": "report", "decision": "sprint_report_written",
            "path": str(out)})

    # -- dispatch ------------------------------------------------------------

    STAGE_HANDLERS: Final[dict[str, str]] = {
        "prepare-audit": "_stage_prepare_audit",
        "label-screen": "_stage_label_screen",
        "recipe-recheck": "_stage_recipe_recheck",
        "crossfit": "_stage_crossfit",
        "report": "_stage_report",
    }

    def run_stage(self, stage: str) -> str | int:
        _require(stage in STAGE_ORDER, "STAGE_UNKNOWN",
                 f"unknown stage {stage!r}; known: {list(STAGE_ORDER)}")
        if self.dry_run:
            return self._dry_run_stage(stage)
        window = StageWindow(self.ensure_manifest()["T0_unix"], stage,
                             clock=self._clock)
        try:
            self._precheck(stage)
            handler = getattr(self, self.STAGE_HANDLERS[stage])
            return handler(window)
        except StageCensored as exc:
            return self._record_stage(stage, "censored_budget", detail=str(exc))
        except SprintRefused as exc:
            if exc.code in ("G51_FAILED", "ORACLE_STATIC_FAILED",
                            "GV_RENDER_FAILED"):
                status = "rejected_gate"
            elif stage == "prepare-audit":
                status = "rejected_data"
            else:
                raise
            return self._record_stage(stage, status, detail=str(exc))

    def _dry_run_stage(self, stage: str) -> str:
        """List the planned actions; freeze nothing; write NOTHING."""
        print(f"[dry-run] stage={stage}: planned actions follow "
              f"(T0 NOT frozen, no files written)")
        plans = {
            "prepare-audit": (
                "freeze split (v5_fold --write); prepare_audit(allow_truth="
                f"{self.allow_truth}); teacher completion check (3 d0); "
                "precompute_features; Oracle/Classic baseline table; G5.1 "
                "pytest evidence; GPU timing probe (30 warmup + 200 steps "
                "+ 3 dense evals); start_frame_sampling_manifest.json; "
                "activate g_v_map V5_SPRINT + render_gv_map --write"),
            "label-screen": (
                "feasibility 6×t_run(1500) ≤ 108 min else downgrade arms "
                f"{list(self.sprint.get('downgrade_ladder') or [1500,1000,500])}"
                "; run six screen arms C05/C08/C12/A05/A08/A12 (fold 0, "
                "seed 17) as trainer subprocesses; rank per plan §5.3; "
                "write label_recipe_matrix.json + ablation_table.parquet"),
            "recipe-recheck": (
                "top-2 recipes × folds{0,1} × seeds{17,29,43} = 12 cells "
                "(reuse 2 Tier-1 cells; 10 new runs); stability verdict; "
                "write recipe_recheck.parquet + decision"),
            "crossfit": (
                "CONDITIONAL on stable winner + G5.1 green + Oracle static "
                "pass; feasibility 15×t_run(3000)+30 ≤ 150 else 5×1seed "
                "else archive-only; formal 3000-update runs from scratch; "
                "export move_proposal_v5_draft1.parquet (missing rows "
                "censored, never zero-filled)"),
            "report": (
                "completeness + dense dev re-validation; sealed_eval_request "
                "(REFUSED, registration echoed — never unseals); "
                "collision/follower diagnostics; grouped bootstrap; "
                "scale_curriculum_next_run.json (planned-only); "
                "gate_report.md/json + docs/reports sprint report"),
        }
        print(f"  - {plans[stage]}")
        return "dry_run"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/campaigns/run_v5_sprint.py",
        description="V5 phase-1 ten-hour sprint campaign driver (ADR-0026)")
    parser.add_argument("--config", default="config/campaigns/v5_sprint.yaml")
    parser.add_argument("--stage", required=True, choices=STAGE_ORDER)
    parser.add_argument("--dry-run", action="store_true",
                        help="list planned actions; freeze nothing; write "
                             "nothing")
    parser.add_argument("--allow-truth", action="store_true",
                        help="audited IRC access gate for prepare-audit")
    parser.add_argument("--unseal", action="store_true",
                        help="recorded request only — the sealed confirm "
                             "set stays REFUSED all sprint (plan §5.5)")
    parser.add_argument("--force", action="store_true",
                        help="re-run completed work; downgrade parent-hash "
                             "mismatch to a recorded override")
    parser.add_argument("--device", default=None,
                        help="trainer device override (e.g. cpu for smoke)")
    parser.add_argument("--out", default=None,
                        help="campaign output root (default from config "
                             "paths.output_root)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.is_file():
        print(f"REFUSED[CONFIG_MISSING]: {config_path}", file=sys.stderr)
        return EXIT_REFUSED
    out_root = (Path(args.out) if args.out else None)
    if out_root is None:
        cfg = load_config_mapping(config_path)
        out_root = Path((cfg.get("paths") or {}).get(
            "output_root", "outputs/pesn/v5/sprint"))
    driver = SprintDriver(
        config_path=config_path, out_root=out_root,
        allow_truth=args.allow_truth, unseal=args.unseal,
        force=args.force, dry_run=args.dry_run, device=args.device)
    try:
        result = driver.run_stage(args.stage)
    except SprintRefused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED
    if result == "censored_budget":
        return EXIT_CENSORED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
