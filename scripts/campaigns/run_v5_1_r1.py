#!/usr/bin/env python
"""V5.1a-R1 resumable run-queue driver (campaign V5_1A_R1_REPAIR).

Standalone sibling of ``scripts/campaigns/run_v5_1_sprint.py`` — same
house rules (dry-run writes NOTHING, subprocess-per-run, crash-restart
safe via manifest + decisions + per-run ``status.json``), adapted to the
R1 new-loss lane frozen in ``outputs/pesn/v5_1_r1/protocol_r1.json``:

* ``--stage {baseline_fill, screen, dev, confirm}`` + optional ``--plan``
  JSON (validated: no fold outside the stage plan, no seed outside
  {17,29,43}, mode must stay ``v51`` — formal OOF is REFUSED);
* ``baseline_fill`` — the A0 matrix: the 7 PRE-EXISTING V5.1a B0 runs
  (folds 0 × seeds 17/29/43 + folds 1–4 × seed 17) are REUSED when
  delivered (status + checkpoint_best), recorded with their config/data
  hashes; the 8 missing cells train B0 with the ORIGINAL
  ``config/campaigns/v5_1_sprint.yaml`` (identical effective config sha
  to the V5.1a runs);
* ``screen`` — runs ``scripts/campaigns/r1_weight_screen.py`` (CPU);
* ``dev`` — arms {A2, A3} × folds {0,1,2} × seeds {17,29,43} with
  ``config/campaigns/v5_1_r1.yaml``; ``confirm`` — ``--arm`` × folds
  {3,4} × seeds {17,29,43};
* per-run subprocess stdout/stderr CAPTURED to ``run_dir/console.log``,
  argv/returncode/wall_s/gpu-mem appended to
  ``outputs/pesn/v5_1_r1/sprint/r1_decisions.jsonl``
  (schema ``pesn_v5_1_r1_decision_v1``);
* budget guard ``--max-wall-minutes`` (default 240): no NEW run is
  launched once elapsed ≥ limit − 5 min; the queue finishes gracefully
  and writes a partial stage state (``censored_budget``);
* HARD REFUSALS (nonzero exit): folds/seeds/arms outside the stage plan,
  any plan mode other than ``v51`` (formal OOF), and there is
  deliberately NO ``--unseal`` flag of any kind — argparse rejects it;
* manifest ``outputs/pesn/v5_1_r1/sprint/r1_manifest.json`` (config
  sha256 of BOTH campaign configs, split-manifest sha, stage statuses,
  T0) and a per-stage summary CSV ``outputs/pesn/v5_1_r1/<stage>_summary.csv``.

Planner units live in :func:`plan_stage` (pure, hermetic — covered by
``tests/pesn/test_r1_driver_plan.py``).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.config_loader import load_config_mapping  # noqa: E402
from pes2ts_core.utils.jsonio import read_json, write_json  # noqa: E402

# ---------------------------------------------------------------------------
# Vocabulary / constants
# ---------------------------------------------------------------------------

EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 2
EXIT_REFUSED: Final[int] = 3
EXIT_CENSORED: Final[int] = 4

STAGES: Final[tuple[str, ...]] = (
    "baseline_fill", "screen", "dev", "confirm")

MANIFEST_SCHEMA: Final[str] = "pesn_v5_1_r1_manifest_v1"
DECISION_SCHEMA: Final[str] = "pesn_v5_1_r1_decision_v1"
CAMPAIGN_ID: Final[str] = "V5_1A_R1_REPAIR"
R1_CONFIG_SCHEMA: Final[str] = "pesn_v5_1_r1_v1"
SPRINT_CONFIG_SCHEMA: Final[str] = "pesn_v5_1_sprint_v1"

DEFAULT_SPRINT_CONFIG: Final[str] = "config/campaigns/v5_1_sprint.yaml"
DEFAULT_R1_CONFIG: Final[str] = "config/campaigns/v5_1_r1.yaml"
DEFAULT_LEGACY_RUNS_ROOT: Final[str] = "outputs/pesn/v5_1/runs"
DEFAULT_R1_ROOT: Final[str] = "outputs/pesn/v5_1_r1"
DEFAULT_R1_RUNS_ROOT: Final[str] = "outputs/pesn/v5_1_r1/runs"
DEFAULT_UPDATES: Final[int] = 3000

#: A run counts as delivered iff status.json exists with one of these
#: statuses AND ``checkpoints/checkpoint_best.pt`` exists (mirror of
#: ``run_v5_1_sprint._run_delivered`` + the best-checkpoint requirement
#: for every delivered status).
DELIVERED_STATUSES: Final[tuple[str, ...]] = (
    "complete", "censored_patience", "censored_screen")

#: Preregistered A0 cells reused from the V5.1a campaign (never re-run
#: when delivered): fold0 × {17,29,43} + folds 1–4 × seed 17.
A0_REUSE_CELLS: Final[tuple[tuple[int, int], ...]] = (
    (0, 17), (0, 29), (0, 43), (1, 17), (2, 17), (3, 17), (4, 17))

#: The 8 missing A0 baselines trained by ``baseline_fill`` with the
#: ORIGINAL v5_1_sprint.yaml (protocol_r1.json baseline_fill_config).
BASELINE_FILL_CELLS: Final[tuple[tuple[int, int], ...]] = (
    (1, 29), (1, 43), (2, 29), (2, 43),
    (3, 29), (3, 43), (4, 29), (4, 43))

ALL_FOLDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
SEEDS: Final[tuple[int, ...]] = (17, 29, 43)
DEV_FOLDS: Final[tuple[int, ...]] = (0, 1, 2)
CONFIRM_FOLDS: Final[tuple[int, ...]] = (3, 4)
DEV_ARMS: Final[tuple[str, ...]] = ("A2", "A3")
CONFIRM_ARMS: Final[tuple[str, ...]] = ("A2", "A3", "A4")

STAGE_STATUS_VALUES: Final[tuple[str, ...]] = (
    "complete", "censored_budget", "failed_runs")

#: No new subprocess inside the final minutes of the wall budget.
LAUNCH_GUARD_MINUTES: Final[int] = 5
DEFAULT_MAX_WALL_MINUTES: Final[int] = 240

#: Keys accepted in a ``--plan`` JSON document (anything else → refusal).
PLAN_KEYS: Final[frozenset] = frozenset({
    "arms", "folds", "seeds", "updates", "mode",
    "lambda_strong", "lambda_dead2"})

TRAINER_MODULE: Final[str] = "pes2ts_core.pesn.offline.training.key_atom_bc"
WEIGHT_SCREEN_SCRIPT: Final[str] = "scripts/campaigns/r1_weight_screen.py"


class DriverRefused(RuntimeError):
    """Typed refusal (nonzero exit + machine-readable code)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise DriverRefused(code, detail)


def _utc_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    import hashlib
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pure planning (hermetic; unit-tested)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RunCell:
    """One planned queue entry (arm × fold × seed, one stage)."""

    stage: str
    run_id: str
    arm: str
    fold: int
    seed: int
    mode: str
    updates: int
    config_path: Path
    out_dir: Path
    reuse_from: Path | None = None

    def to_payload(self) -> dict:
        return {
            "stage": self.stage, "run_id": self.run_id, "arm": self.arm,
            "fold": self.fold, "seed": self.seed, "mode": self.mode,
            "updates": self.updates, "config_path": str(self.config_path),
            "out_dir": str(self.out_dir),
            "reuse_from": (str(self.reuse_from)
                           if self.reuse_from is not None else None),
        }


def _run_id(arm: str, fold: int, seed: int, mode: str = "v51") -> str:
    return f"{arm}_f{int(fold)}_s{int(seed)}_{mode}"


def _as_int_list(value, *, field: str) -> list[int]:
    _require(isinstance(value, (list, tuple)) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in value),
        "PLAN_INVALID", f"plan.{field} must be a list of ints, got {value!r}")
    return [int(v) for v in value]


def _as_str_list(value, *, field: str) -> list[str]:
    _require(isinstance(value, (list, tuple)) and all(
        isinstance(v, str) for v in value),
        "PLAN_INVALID", f"plan.{field} must be a list of strings, "
        f"got {value!r}")
    return [str(v) for v in value]


def validate_plan(stage: str, plan: Mapping | None) -> dict:
    """Validate a ``--plan`` document against the STAGE plan.

    Refuses (typed :class:`DriverRefused`) on: unknown keys, any mode
    other than ``v51`` (formal OOF), folds/seeds/arms outside the stage
    plan, non-positive updates and r1-lambda overrides outside the
    training stages.
    """
    plan = dict(plan or {})
    unknown = sorted(set(plan) - PLAN_KEYS)
    _require(not unknown, "PLAN_KEY_UNKNOWN",
             f"plan keys {unknown} not in {sorted(PLAN_KEYS)}")
    if stage == "screen":
        _require(not (set(plan) - {"mode"}), "PLAN_KEY_NOT_APPLICABLE",
                 "the screen stage has no run cells (weight screen only); "
                 f"got plan keys {sorted(plan)}")
    mode = str(plan.get("mode", "v51"))
    _require(mode == "v51", "FORMAL_OOF_REFUSED",
             f"only mode v51 is admissible for {CAMPAIGN_ID} — formal OOF "
             f"mode {mode!r} is refused (protocol_r1.json: NO formal OOF)")
    plan["mode"] = mode
    if "updates" in plan:
        updates = plan["updates"]
        _require(isinstance(updates, int) and not isinstance(updates, bool)
                 and updates > 0, "PLAN_INVALID",
                 f"plan.updates must be a positive int, got {updates!r}")
    for key in ("lambda_strong", "lambda_dead2"):
        if key in plan:
            _require(stage in ("dev", "confirm"), "PLAN_KEY_NOT_APPLICABLE",
                     f"plan.{key} only applies to dev/confirm, not {stage}")
            value = plan[key]
            _require(isinstance(value, (int, float))
                     and not isinstance(value, bool) and float(value) > 0,
                     "PLAN_INVALID",
                     f"plan.{key} must be a positive number, got {value!r}")
    if "folds" in plan:
        folds = set(_as_int_list(plan["folds"], field="folds"))
        allowed = _allowed_folds(stage)
        _require(bool(folds) and folds <= set(allowed), "FOLD_NOT_IN_PLAN",
                 f"plan folds {sorted(folds)} outside the {stage} plan "
                 f"{list(allowed)} — adding folds is refused")
        plan["folds"] = sorted(folds)
    if "seeds" in plan:
        seeds = set(_as_int_list(plan["seeds"], field="seeds"))
        _require(bool(seeds) and seeds <= set(SEEDS), "SEED_NOT_IN_PLAN",
                 f"plan seeds {sorted(seeds)} outside {list(SEEDS)}")
        plan["seeds"] = sorted(seeds)
    if "arms" in plan:
        arms = _as_str_list(plan["arms"], field="arms")
        allowed = _allowed_arms(stage)
        _require(bool(arms) and set(arms) <= set(allowed),
                 "ARM_NOT_IN_PLAN",
                 f"plan arms {arms} outside the {stage} plan "
                 f"{list(allowed)}")
        plan["arms"] = arms
    return plan


def _allowed_folds(stage: str) -> tuple[int, ...]:
    if stage == "baseline_fill":
        return ALL_FOLDS
    if stage == "dev":
        return DEV_FOLDS
    if stage == "confirm":
        return CONFIRM_FOLDS
    return ()


def _allowed_arms(stage: str) -> tuple[str, ...]:
    if stage == "baseline_fill":
        return ("B0",)
    if stage == "dev":
        return DEV_ARMS
    if stage == "confirm":
        return CONFIRM_ARMS
    return ()


def plan_stage(
    stage: str,
    *,
    plan: Mapping | None = None,
    arm: str | None = None,
    r1_config_path: Path,
    sprint_config_path: Path,
    runs_root: Path,
    legacy_runs_root: Path,
    default_updates: int = DEFAULT_UPDATES,
) -> list[RunCell]:
    """Build the deterministic queue for one stage (pure; no I/O).

    ``baseline_fill`` plans the full A0 matrix (5 folds × 3 seeds):
    preregistered V5.1a cells carry ``reuse_from`` (reused when
    delivered), the 8 missing cells run B0 with the ORIGINAL
    ``v5_1_sprint.yaml``.  ``screen`` plans no cells (the weight screen
    runs as its own subprocess).
    """
    _require(stage in STAGES, "STAGE_UNKNOWN",
             f"unknown stage {stage!r}; known: {list(STAGES)}")
    cfg = validate_plan(stage, plan)
    mode = "v51"  # formal OOF refused in validate_plan; hardwired here
    updates = int(cfg.get("updates", default_updates))

    if stage == "screen":
        return []

    if stage == "baseline_fill":
        arms = cfg.get("arms", ["B0"])
        folds = cfg.get("folds", list(ALL_FOLDS))
        seeds = cfg.get("seeds", list(SEEDS))
        cells: list[RunCell] = []
        for arm in arms:
            for fold in folds:
                for seed in seeds:
                    rid = _run_id(arm, fold, seed, mode)
                    if (fold, seed) in A0_REUSE_CELLS:
                        cells.append(RunCell(
                            stage=stage, run_id=rid, arm=arm, fold=fold,
                            seed=seed, mode=mode, updates=updates,
                            config_path=sprint_config_path,
                            out_dir=legacy_runs_root / rid,
                            reuse_from=legacy_runs_root / rid))
                    else:
                        cells.append(RunCell(
                            stage=stage, run_id=rid, arm=arm, fold=fold,
                            seed=seed, mode=mode, updates=updates,
                            config_path=sprint_config_path,
                            out_dir=runs_root / rid))
        return cells

    if stage == "dev":
        arms = cfg.get("arms", list(DEV_ARMS))
        folds = cfg.get("folds", list(DEV_FOLDS))
        seeds = cfg.get("seeds", list(SEEDS))
    else:  # confirm
        _require(arm is not None, "ARM_REQUIRED",
                 "confirm requires --arm "
                 f"({'|'.join(CONFIRM_ARMS)})")
        _require(arm in CONFIRM_ARMS, "ARM_NOT_IN_PLAN",
                 f"confirm arm {arm!r} outside {list(CONFIRM_ARMS)}")
        if "arms" in cfg:
            _require(cfg["arms"] == [arm], "ARM_NOT_IN_PLAN",
                     f"plan arms {cfg['arms']} must be exactly [{arm!r}] "
                     "for the chosen --arm")
        arms = [arm]
        folds = cfg.get("folds", list(CONFIRM_FOLDS))
        seeds = cfg.get("seeds", list(SEEDS))

    cells = []
    for a in arms:
        for fold in folds:
            for seed in seeds:
                rid = _run_id(a, fold, seed, mode)
                cells.append(RunCell(
                    stage=stage, run_id=rid, arm=a, fold=fold, seed=seed,
                    mode=mode, updates=updates, config_path=r1_config_path,
                    out_dir=runs_root / rid))
    return cells


def run_delivered(status: Mapping | None, run_dir: Path | str | None) -> bool:
    """Delivered ⇔ status.json ∈ DELIVERED_STATUSES AND best ckpt exists."""
    if not status:
        return False
    if str(status.get("status")) not in DELIVERED_STATUSES:
        return False
    if run_dir is None:
        return True
    return (Path(run_dir) / "checkpoints" / "checkpoint_best.pt").is_file()


class BudgetGuard:
    """Wall-clock budget for NEW launches (injectable clock)."""

    def __init__(self, t0: float, *, limit_minutes: float,
                 launch_guard_minutes: float = LAUNCH_GUARD_MINUTES,
                 clock: Callable[[], float] = time.time) -> None:
        self.t0 = float(t0)
        self.limit_minutes = float(limit_minutes)
        self.launch_guard_minutes = float(launch_guard_minutes)
        self._clock = clock

    def elapsed_minutes(self) -> float:
        return (float(self._clock()) - self.t0) / 60.0

    def expired(self) -> bool:
        return self.elapsed_minutes() >= self.limit_minutes

    def may_launch(self) -> bool:
        """True while a NEW run may start (guard window respected)."""
        return self.elapsed_minutes() < (
            self.limit_minutes - self.launch_guard_minutes)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class R1QueueDriver:
    """Crash-restart-safe sequential queue over the R1 stages."""

    def __init__(
        self,
        *,
        sprint_config_path: Path,
        r1_config_path: Path,
        root: Path,
        runs_root: Path,
        legacy_runs_root: Path,
        dry_run: bool = False,
        device: str | None = None,
        max_wall_minutes: float = DEFAULT_MAX_WALL_MINUTES,
        arm: str | None = None,
        plan: Mapping | None = None,
        lambda_strong: float | None = None,
        lambda_dead2: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.sprint_config_path = Path(sprint_config_path)
        self.r1_config_path = Path(r1_config_path)
        self.sprint_config = load_config_mapping(self.sprint_config_path)
        self.r1_config = load_config_mapping(self.r1_config_path)
        _require(self.sprint_config.get("schema") == SPRINT_CONFIG_SCHEMA,
                 "CONFIG_SCHEMA",
                 f"expected {SPRINT_CONFIG_SCHEMA} in "
                 f"{self.sprint_config_path}, got "
                 f"{self.sprint_config.get('schema')!r}")
        _require(self.r1_config.get("schema") == R1_CONFIG_SCHEMA,
                 "CONFIG_SCHEMA",
                 f"expected {R1_CONFIG_SCHEMA} in {self.r1_config_path}, "
                 f"got {self.r1_config.get('schema')!r}")
        self.root = Path(root)
        self.sprint_dir = self.root / "sprint"
        self.runs_root = Path(runs_root)
        self.legacy_runs_root = Path(legacy_runs_root)
        self.dry_run = bool(dry_run)
        self.device = device
        self.max_wall_minutes = float(max_wall_minutes)
        self.arm = arm
        self.plan = dict(plan or {})
        self.lambda_strong = (
            None if lambda_strong is None else float(lambda_strong))
        self.lambda_dead2 = (
            None if lambda_dead2 is None else float(lambda_dead2))
        self._clock = clock
        self._inv_t0 = float(clock())
        self.manifest_path = self.sprint_dir / "r1_manifest.json"
        self.decisions_path = self.sprint_dir / "r1_decisions.jsonl"
        self.manifest: dict | None = None
        self._patched_config_path: Path | None = None
        self._load_manifest()

    # -- manifest / decisions (dry-run writes NOTHING) ----------------------

    def _load_manifest(self) -> None:
        if self.manifest_path.is_file():
            self.manifest = read_json(self.manifest_path)  # type: ignore

    def ensure_manifest(self) -> dict:
        if self.manifest is None:
            _require(not self.dry_run, "DRY_RUN_WRITE",
                     "dry-run must never freeze T0 or write the manifest")
            self.sprint_dir.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "campaign_id": CAMPAIGN_ID,
                "T0_unix": float(self._clock()),
                "T0_iso": _utc_now(),
                "configs": {
                    "v5_1_sprint": {
                        "path": str(self.sprint_config_path),
                        "sha256": _sha256_file(self.sprint_config_path)},
                    "v5_1_r1": {
                        "path": str(self.r1_config_path),
                        "sha256": _sha256_file(self.r1_config_path)},
                },
                "split_manifest": {
                    "path": str(self._split_manifest_path()),
                    "sha256": _sha256_file(self._split_manifest_path())},
                "legacy_runs_root": str(self.legacy_runs_root),
                "runs_root": str(self.runs_root),
                "arms": list(self.r1_config.get("sprint", {}).get("arms")
                             or ()),
                "status_vocabulary": list(STAGE_STATUS_VALUES),
                "delivered_statuses": list(DELIVERED_STATUSES),
                "oof_policy": (f"{CAMPAIGN_ID}: NO formal OOF — mode is "
                               "hardwired to v51; any other mode is "
                               "refused (protocol_r1.json)"),
                "unseal_policy": ("no --unseal flag exists in this driver "
                                  "(refused by argparse)"),
                "max_wall_minutes_default": DEFAULT_MAX_WALL_MINUTES,
                "stages": {},
            }
            write_json(self.manifest_path, self.manifest)
        return self.manifest

    def _split_manifest_path(self) -> Path:
        paths = dict(self.r1_config.get("paths") or {})
        return Path(paths.get(
            "split_manifest", "data/manifests/v5/split_manifest.json"))

    def append_decision(self, decision: Mapping) -> None:
        if self.dry_run:
            return
        self.sprint_dir.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": DECISION_SCHEMA,
                   "at_unix": float(self._clock()), **dict(decision)}
        with self.decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True,
                                ensure_ascii=False) + "\n")

    def _record_stage(self, stage: str, status: str, *, detail: str = "",
                      extra: Mapping | None = None) -> str:
        _require(status in STAGE_STATUS_VALUES, "STATUS_UNKNOWN",
                 f"status {status!r} not in {list(STAGE_STATUS_VALUES)}")
        entry = {"status": status, "detail": detail,
                 "at_unix": float(self._clock()), "at_iso": _utc_now()}
        if extra:
            entry.update(dict(extra))
        if not self.dry_run:
            manifest = self.ensure_manifest()
            manifest.setdefault("stages", {})[stage] = entry
            write_json(self.manifest_path, manifest)
            self.append_decision({"stage": stage, "decision": "stage_status",
                                  **entry})
        return status

    def _verify_parents(self, stage: str) -> None:
        """Re-verify config + split hashes frozen at T0 (typed refusal)."""
        manifest = self.ensure_manifest()
        current = {
            "v5_1_sprint": _sha256_file(self.sprint_config_path),
            "v5_1_r1": _sha256_file(self.r1_config_path),
            "split_manifest": _sha256_file(self._split_manifest_path()),
        }
        frozen = {
            "v5_1_sprint": (manifest.get("configs", {}).get(
                "v5_1_sprint", {}) or {}).get("sha256"),
            "v5_1_r1": (manifest.get("configs", {}).get(
                "v5_1_r1", {}) or {}).get("sha256"),
            "split_manifest": (manifest.get("split_manifest", {})
                               or {}).get("sha256"),
        }
        mismatched = [k for k, v in current.items()
                      if v is not None and frozen.get(k) not in (None, v)]
        _require(not mismatched, "PARENT_HASH_MISMATCH",
                 f"stage {stage}: frozen hashes changed for "
                 f"{mismatched} since T0 — R1 configs/split are frozen "
                 "(protocol_r1.json)")

    # -- run readers ---------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            doc = read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        return doc if isinstance(doc, dict) else None

    def _status_of(self, run_dir: Path) -> dict | None:
        return self._read_json(run_dir / "status.json")

    def _delivered(self, run_dir: Path) -> bool:
        return run_delivered(self._status_of(run_dir), run_dir)

    # -- subprocess plumbing (captured console.log + gpu sampling) -----------

    @staticmethod
    def _gpu_snapshot() -> int | None:
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

    def _spawn_captured(self, argv: Sequence[str], *, console_log: Path,
                        stage: str, sample_every_s: float = 10.0
                        ) -> tuple[int, float, int | None]:
        """Run a subprocess with stdout/stderr CAPTURED to console.log.

        Returns ``(returncode, wall_s, peak_gpu_mib)``; peak GPU mem is
        sampled every ``sample_every_s`` (None when nvidia-smi is absent
        or idle — CPU runs).
        """
        console_log.parent.mkdir(parents=True, exist_ok=True)
        peak: dict = {"mib": None}
        stop = threading.Event()

        def _monitor() -> None:
            while not stop.wait(sample_every_s):
                mib = self._gpu_snapshot()
                if mib is not None:
                    prev = peak["mib"]
                    peak["mib"] = mib if prev is None else max(prev, mib)

        started = time.time()
        monitor = threading.Thread(target=_monitor, daemon=True)
        monitor.start()
        try:
            with console_log.open("w", encoding="utf-8") as fh:
                fh.write(f"# argv: {json.dumps(list(argv))}\n")
                fh.write(f"# started_utc: {_utc_now()}\n")
                fh.write(f"# stage: {stage}\n\n")
                fh.flush()
                proc = subprocess.run(
                    list(argv), cwd=str(PROJECT_ROOT),
                    stdout=fh, stderr=subprocess.STDOUT)
                returncode = proc.returncode
        finally:
            stop.set()
            monitor.join(timeout=2.0)
        wall_s = time.time() - started
        return returncode, wall_s, peak["mib"]

    def _trainer_argv(self, cell: RunCell, config_path: Path) -> list[str]:
        argv = [sys.executable, "-m", TRAINER_MODULE,
                "--config", str(config_path),
                "--arm", str(cell.arm),
                "--fold", str(cell.fold),
                "--seed", str(cell.seed),
                "--updates", str(int(cell.updates)),
                # mode is HARDWIRDED to v51 — formal OOF does not exist
                # in this campaign (protocol_r1.json).
                "--mode", cell.mode,
                "--out", str(cell.out_dir)]
        if self.device:
            argv += ["--device", str(self.device)]
        return argv

    # -- config patch (screened r1 lambdas override) -------------------------

    def resolve_config_path(self, cell: RunCell) -> Path:
        """R1-arm runs may carry CLI-passed r1-lambda overrides, applied
        via a PATCHED copy of v5_1_r1.yaml (both shas recorded)."""
        if cell.config_path != self.r1_config_path:
            return cell.config_path  # baseline_fill: original sprint yaml
        ls = self.lambda_strong
        ld = self.lambda_dead2
        if ls is None and ld is None:
            return cell.config_path
        if self._patched_config_path is None:
            import yaml
            doc = yaml.safe_load(
                self.r1_config_path.read_text(encoding="utf-8"))
            r1_blk = dict((doc.get("v51") or {}).get("r1") or {})
            if ls is not None:
                r1_blk["lambda_strong"] = float(ls)
            if ld is not None:
                r1_blk["lambda_dead2"] = float(ld)
            doc.setdefault("v51", {})["r1"] = r1_blk
            name = (f"r1_config_ls{float(r1_blk['lambda_strong']):g}"
                    f"_ld{float(r1_blk['lambda_dead2']):g}.yaml")
            patched = self.sprint_dir / name
            if not self.dry_run:
                self.sprint_dir.mkdir(parents=True, exist_ok=True)
                patched.write_text(
                    yaml.safe_dump(doc, sort_keys=False,
                                   allow_unicode=True),
                    encoding="utf-8")
                self.append_decision({
                    "stage": cell.stage, "decision": "config_patch",
                    "base_path": str(self.r1_config_path),
                    "base_sha256": _sha256_file(self.r1_config_path),
                    "patched_path": str(patched),
                    "patched_sha256": _sha256_file(patched),
                    "lambda_strong": float(r1_blk["lambda_strong"]),
                    "lambda_dead2": float(r1_blk["lambda_dead2"])})
            self._patched_config_path = patched
        return self._patched_config_path

    # -- summary CSV ---------------------------------------------------------

    @staticmethod
    def _summary_row(cell: RunCell, run_dir: Path) -> dict:
        status_doc = R1QueueDriver._read_json(
            run_dir / "status.json") or {}
        metrics = R1QueueDriver._read_json(run_dir / "metrics.json") or {}
        inner = dict(metrics.get("inner_val_score") or {})

        def _fmt(value):
            if value is None:
                return ""
            if isinstance(value, float):
                return repr(value)
            return value

        return {
            "run_id": cell.run_id,
            "arm": cell.arm,
            "fold": cell.fold,
            "seed": cell.seed,
            "status": str(status_doc.get("status", "missing")),
            "best_update": _fmt(metrics.get("best_update")),
            "inner_rmse_over_zero": _fmt(inner.get("rmse_over_zero")),
            "inner_sre_median": _fmt(inner.get("sre_median")),
            "inner_near_ts_cos": _fmt(inner.get("near_ts_cos")),
            "inner_sign_accuracy": _fmt(metrics.get("sign_accuracy")),
            "alpha_median": _fmt(metrics.get("alpha_median")),
            "dead_zone_fraction": _fmt(metrics.get("dead_zone_fraction")),
        }

    SUMMARY_FIELDS: Final[tuple[str, ...]] = (
        "run_id", "arm", "fold", "seed", "status", "best_update",
        "inner_rmse_over_zero", "inner_sre_median", "inner_near_ts_cos",
        "inner_sign_accuracy", "alpha_median", "dead_zone_fraction")

    def _write_summary(self, stage: str, cells: Sequence[RunCell],
                       *, rows_by_run: Mapping[str, Mapping]) -> Path:
        path = self.root / f"{stage}_summary.csv"
        if self.dry_run:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.SUMMARY_FIELDS))
            writer.writeheader()
            for cell in cells:
                row = dict(rows_by_run.get(cell.run_id)
                           or self._summary_row(
                               cell,
                               cell.reuse_from or cell.out_dir))
                writer.writerow({k: row.get(k, "")
                                 for k in self.SUMMARY_FIELDS})
        return path

    # -- stage runners -------------------------------------------------------

    def _stage_screen(self) -> tuple[str, str, int]:
        """Run the preregistered weight screen (CPU) as a subprocess."""
        out_json = self.root / "screen" / "weight_screen.json"
        console = self.root / "screen" / "screen_console.log"
        argv = [sys.executable, WEIGHT_SCREEN_SCRIPT]
        if not self.dry_run:
            argv += ["--out", str(out_json)]
            if self.device:
                argv += ["--device", str(self.device)]
            returncode, wall_s, gpu = self._spawn_captured(
                argv, console_log=console, stage="screen")
            self.append_decision({
                "stage": "screen", "decision": "weight_screen",
                "argv": argv, "returncode": returncode,
                "wall_s": wall_s, "gpu_memory_mib": gpu,
                "out": str(out_json),
                "delivered": out_json.is_file()})
            if returncode != 0 or not out_json.is_file():
                return ("failed_runs",
                        f"weight screen failed (rc={returncode}, "
                        f"out={out_json.is_file()})", EXIT_FAILED)
            return ("complete", f"weight screen -> {out_json}", EXIT_OK)
        return ("complete", "dry-run: would run "
                f"{WEIGHT_SCREEN_SCRIPT}", EXIT_OK)

    def run_stage(self, stage: str) -> int:
        cells = plan_stage(
            stage, plan=self.plan, arm=self.arm,
            r1_config_path=self.r1_config_path,
            sprint_config_path=self.sprint_config_path,
            runs_root=self.runs_root,
            legacy_runs_root=self.legacy_runs_root)
        _require(
            stage in ("dev", "confirm")
            or (self.lambda_strong is None and self.lambda_dead2 is None),
            "PLAN_KEY_NOT_APPLICABLE",
            f"--lambda-strong/--lambda-dead2 overrides only apply to the "
            f"dev/confirm stages, not {stage}")
        if self.dry_run:
            payload = {
                "stage": stage, "dry_run": True,
                "plan": [c.to_payload() for c in cells],
                "screen": (None if stage != "screen" else
                           {"script": WEIGHT_SCREEN_SCRIPT}),
                "lambda_strong_override": self.lambda_strong,
                "lambda_dead2_override": self.lambda_dead2,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return EXIT_OK

        manifest = self.ensure_manifest()
        self._verify_parents(stage)
        guard = BudgetGuard(self._inv_t0,
                            limit_minutes=self.max_wall_minutes,
                            clock=self._clock)

        rows_by_run: dict[str, dict] = {}
        budget_stop = False
        any_failed = False
        detail = ""

        if stage == "screen":
            status, detail, code = self._stage_screen()
            self._record_stage(stage, status, detail=detail)
            self._write_summary(stage, cells, rows_by_run={})
            return code

        for idx, cell in enumerate(cells):
            # 1) reuse preregistered A0 cells from the V5.1a campaign
            if cell.reuse_from is not None and self._delivered(
                    cell.reuse_from):
                status_doc = self._status_of(cell.reuse_from) or {}
                record = self._read_json(
                    cell.reuse_from / "run_record.json") or {}
                gv = dict(record.get("gv") or {})
                self.append_decision({
                    "stage": stage, "decision": "reuse_a0",
                    "run_id": cell.run_id, "arm": cell.arm,
                    "fold": cell.fold, "seed": cell.seed,
                    "mode": cell.mode, "run_dir": str(cell.reuse_from),
                    "status": str(status_doc.get("status")),
                    "config_sha256": gv.get("config_sha256"),
                    "data_manifest_sha256": gv.get("data_manifest_sha256"),
                    "loss_schema_version": record.get(
                        "loss_schema_version"),
                })
                rows_by_run[cell.run_id] = self._summary_row(
                    cell, cell.reuse_from)
                continue
            # 2) reuse candidates that vanished → train the cell fresh
            run_dir = cell.out_dir
            if cell.reuse_from is not None and not self._delivered(
                    cell.reuse_from):
                run_dir = self.runs_root / cell.run_id
                cell = RunCell(
                    stage=cell.stage, run_id=cell.run_id, arm=cell.arm,
                    fold=cell.fold, seed=cell.seed, mode=cell.mode,
                    updates=cell.updates, config_path=cell.config_path,
                    out_dir=run_dir, reuse_from=None)
            # 3) already delivered in THIS campaign → skip (resumable)
            if self._delivered(run_dir):
                self.append_decision({
                    "stage": stage, "decision": "skip_complete",
                    "run_id": cell.run_id, "arm": cell.arm,
                    "fold": cell.fold, "seed": cell.seed,
                    "mode": cell.mode, "run_dir": str(run_dir),
                    "status": str((self._status_of(run_dir) or {}).get(
                        "status"))})
                rows_by_run[cell.run_id] = self._summary_row(cell, run_dir)
                continue
            # 4) budget guard — refuse NEW launches, finish gracefully
            if not guard.may_launch():
                budget_stop = True
                detail = (
                    f"budget guard: elapsed "
                    f"{guard.elapsed_minutes():.1f} min >= "
                    f"{self.max_wall_minutes} - {LAUNCH_GUARD_MINUTES} "
                    f"min; {len(cells) - idx} queue "
                    "entries left unlaunched")
                self.append_decision({
                    "stage": stage, "decision": "budget_stop",
                    "elapsed_minutes": guard.elapsed_minutes(),
                    "max_wall_minutes": self.max_wall_minutes,
                    "run_id": cell.run_id})
                break
            # 5) launch the trainer subprocess (captured console.log)
            config_path = self.resolve_config_path(cell)
            run_dir.mkdir(parents=True, exist_ok=True)
            argv = self._trainer_argv(cell, config_path)
            returncode, wall_s, gpu = self._spawn_captured(
                argv, console_log=run_dir / "console.log", stage=stage)
            status_doc = self._status_of(run_dir) or {}
            delivered = self._delivered(run_dir)
            if not delivered:
                any_failed = True
            self.append_decision({
                "stage": stage, "decision": "launch",
                "run_id": cell.run_id, "arm": cell.arm,
                "fold": cell.fold, "seed": cell.seed, "mode": cell.mode,
                "updates": int(cell.updates),
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "out_dir": str(run_dir),
                "argv": argv, "returncode": returncode,
                "wall_s": wall_s, "gpu_memory_mib": gpu,
                "status": str(status_doc.get("status", "unknown")),
                "delivered": delivered,
                "console_log": str(run_dir / "console.log")})
            rows_by_run[cell.run_id] = self._summary_row(cell, run_dir)

        summary_path = self._write_summary(stage, cells,
                                           rows_by_run=rows_by_run)
        n_cells = len(cells)
        n_delivered = sum(
            1 for c in cells
            if (rows_by_run.get(c.run_id) or {}).get("status")
            in DELIVERED_STATUSES)
        if budget_stop:
            status, code = "censored_budget", EXIT_CENSORED
        elif any_failed:
            status, code = "failed_runs", EXIT_FAILED
        else:
            status, code = "complete", EXIT_OK
        extra = {"cells": n_cells, "delivered": n_delivered,
                 "summary_csv": str(summary_path),
                 "elapsed_minutes": guard.elapsed_minutes()}
        self._record_stage(stage, status, detail=detail, extra=extra)
        print(json.dumps({"stage": stage, "status": status,
                          "cells": n_cells, "delivered": n_delivered,
                          "summary_csv": str(summary_path),
                          "detail": detail},
                         ensure_ascii=False))
        return code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/campaigns/run_v5_1_r1.py",
        description=("V5.1a-R1 resumable run-queue driver "
                     "(NO formal OOF, NO unseal)"))
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--plan", default=None,
                   help="optional plan JSON (arms/folds/seeds/updates/"
                        "mode/lambda_strong/lambda_dead2) — validated "
                        "against the frozen stage plan")
    p.add_argument("--arm", default=None,
                   help="confirm-stage arm (A2|A3|A4)")
    p.add_argument("--config", default=DEFAULT_R1_CONFIG,
                   help="R1 campaign yaml (default config/campaigns/"
                        "v5_1_r1.yaml)")
    p.add_argument("--sprint-config", default=DEFAULT_SPRINT_CONFIG,
                   help="original V5.1a yaml used by baseline_fill B0")
    p.add_argument("--root", default=None,
                   help="R1 output root (default: config paths.report_root "
                        "= outputs/pesn/v5_1_r1)")
    p.add_argument("--runs-root", default=None,
                   help="R1 runs root (default: config paths.runs_root)")
    p.add_argument("--legacy-runs-root", default=DEFAULT_LEGACY_RUNS_ROOT,
                   help="pre-existing V5.1a B0 runs (A0 reuse pool)")
    p.add_argument("--device", default=None,
                   help="trainer device override (e.g. cuda:0 / cpu)")
    p.add_argument("--max-wall-minutes", type=float,
                   default=DEFAULT_MAX_WALL_MINUTES,
                   help="budget guard for NEW launches (default 240)")
    p.add_argument("--lambda-strong", type=float, default=None,
                   help="override v51.r1.lambda_strong via a patched "
                        "config copy (screened weight)")
    p.add_argument("--lambda-dead2", type=float, default=None,
                   help="override v51.r1.lambda_dead2 via a patched "
                        "config copy (screened weight)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and write NOTHING")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    plan_doc: dict | None = None
    if args.plan:
        import json as _json
        path = Path(args.plan)
        if not path.is_file():
            print(f"PLAN_NOT_FOUND: {path}", file=sys.stderr)
            return EXIT_REFUSED
        try:
            plan_doc = _json.loads(path.read_text(encoding="utf-8"))
        except _json.JSONDecodeError as exc:
            print(f"PLAN_INVALID: {path}: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        if not isinstance(plan_doc, dict):
            print("PLAN_INVALID: plan JSON must be an object",
                  file=sys.stderr)
            return EXIT_REFUSED
    r1_config_path = Path(args.config)
    sprint_config_path = Path(args.sprint_config)
    try:
        probe = load_config_mapping(r1_config_path)
        paths = dict(probe.get("paths") or {})
        root = Path(args.root) if args.root else Path(paths.get(
            "report_root", DEFAULT_R1_ROOT))
        runs_root = Path(args.runs_root) if args.runs_root else Path(
            paths.get("runs_root", DEFAULT_R1_RUNS_ROOT))
        driver = R1QueueDriver(
            sprint_config_path=sprint_config_path,
            r1_config_path=r1_config_path,
            root=root, runs_root=runs_root,
            legacy_runs_root=Path(args.legacy_runs_root),
            dry_run=args.dry_run, device=args.device,
            max_wall_minutes=args.max_wall_minutes,
            arm=args.arm, plan=plan_doc,
            lambda_strong=args.lambda_strong,
            lambda_dead2=args.lambda_dead2)
        return driver.run_stage(args.stage)
    except DriverRefused as exc:
        print(f"REFUSED {exc.code}: {exc.detail}", file=sys.stderr)
        return EXIT_REFUSED
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
