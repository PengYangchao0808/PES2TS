#!/usr/bin/env python
"""V5.1a repair-sprint campaign driver (plan: docs/plans/PES2TS_V5.1_修复冲刺与扩量准入方案_v1_20261019.md §3–§8).

V5.1a is a REPAIR sprint, not a scale-up sprint and NOT a formal-OOF
sprint (plan §0.1/§2).  Hard constraints encoded below:

* **NO formal OOF** — the status vocabulary of this campaign never contains
  ``complete_oof_*``; every fold0/inner-val number is labelled
  ``development evidence only``;
* **NO Confirm/OOD unseal** — ``sealed_eval_request`` is always called with
  ``allow_unseal=False`` and policy ``v5_1a_never_unseals`` (typed REFUSED);
* **thresholds frozen BEFORE training** — ``outputs/pesn/v5_1/protocol_v51.json``
  is written during ``prepare`` and every later stage refuses
  (``PROTOCOL_NOT_FROZEN`` / ``PROTOCOL_HASH_MISMATCH``) when it is missing or
  the config sha256 no longer matches the frozen record.

Five stages (``STAGE_ORDER``): ``prepare`` → ``diagnose`` → ``ablate`` →
``confirm`` → ``report``.

* ``prepare`` — (a) G5.1 fixture pytest over the four classic test files plus
  ``tests/pesn/test_key_atom_trainer_v51.py`` when it exists (REFUSED
  ``G51_NOT_GREEN`` when the v51 trainer code is present but its tests are
  missing, or when pytest fails); (b) re-run the GPU timing probe with the
  parametrized eval interval 50 (``t_run_seconds(..., eval_every=50)``) →
  ``timing_probe.json``; (c) FREEZE ``protocol_v51.json`` — gate thresholds
  imported from ``key_atom_protocols.py`` constants (never retyped), the
  G5.2-v51 full-OOF protocol, M1/S1 gates, stretch targets, the D1 decision
  tree, frozen λs, the inner-val rule, the arm matrix and the frozen
  best-config selection rule;
* ``diagnose`` — D1 checkpoint-effect measurement: B0 arm (legacy losses,
  magnitude head off, pair_loss legacy_j), fold0 seed17, 3000 updates,
  ``mode v51`` → ``outputs/pesn/v5_1/runs/B0_f0_s17_d1/``; post-hoc
  α_best/α_final on the outer dev set with strata (d_eff≈d0 / terminal /
  capped / nearTS / per-cluster) → ``diagnostics/d1_best_checkpoint_alpha.json``
  with the frozen decision-tree verdict;
* ``ablate`` — 2×2 matrix B0–B3 × seeds {17,29,43} × fold0, 3000 updates,
  ``mode v51`` (12 sequential runs, skip-if-complete) →
  ``ablation_v51.parquet`` + ``.json``.  When D1 says "checkpoint problem
  only" (α_best ≥ 0.85) all four arms still run, but the magnitude-head arms
  are marked confirmatory-only per the frozen decision tree;
* ``confirm`` — best config per the frozen selection rule (fold0 3-seed
  median inner-val SRE, subject to rmse_over_zero ≤ B0 median ×1.05, tie-break
  |α−1| then sign accuracy) → 5 folds × seed17 plus B0 reference runs for
  folds 1–4 (marked reference); no fold may regress vs its B0-baseline fold
  rmse; writes ``frozen_recipe_v51.json`` + confirm table;
* ``report`` — ``gate_report_v51.md/json`` (G5.1 evidence, M1 amplitude
  diagnostics, inner-val curves, ablation + confirm tables, all labelled
  development-evidence-only), ``run_records.jsonl`` merger,
  ``scale_readiness_v51.json`` (S0 pending — formal OOF is V5.1b) and the
  sealed-eval request that must stay REFUSED.

House rules inherited from ``run_v5_sprint`` (imported, never forked):
dry-run writes NOTHING; every training run is a subprocess of
``python -m pes2ts_core.pesn.offline.training.key_atom_bc`` with
``--mode v51``; crash-restart safe via ``campaign_manifest.json`` +
``stage_decisions.jsonl`` + per-run ``status.json``; parent hashes
(config / split manifest) re-verified at every stage start; no subprocess in
the final 5 minutes of the wall-clock envelope (``wall_minutes``, default
240 — one campaign-wide window starting at T0).

Trainer argv contract coded against (core v51 lane)::

    python -m pes2ts_core.pesn.offline.training.key_atom_bc \
        --config <yaml> --arm B0..B3 --fold <n> --seed <n> \
        --updates 3000 --mode v51 --out <run_dir> [--device cuda:0]

Config keys consumed by the core lane: ``v51.inner_val.*``,
``v51.magnitude_head.*``, ``v51.pair_exact.*``, ``v51.ramp.*``,
``sprint.arm_table.<arm>.magnitude_head`` / ``.pair_loss``,
``eval_every_updates`` and ``mode v51``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Final, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.config_loader import load_config_mapping  # noqa: E402
from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EFFECTIVE_ATOM_TAU,
    G52_BETTER_THAN_ZERO_FRACTION_MIN,
    G52_CLASSIC_IMPROVEMENT_LOWER_MIN,
    G52_COS_NEAR_TS_MIN,
    G52_COS_OVERALL_MIN,
    G52_MAE_OVER_ZERO_MAX,
    G52_MIN_CLUSTERS,
    G52_MIN_NEAR_TS_CLUSTERS,
    G52_Q75_RMSE_OVER_ZERO_MAX,
    G52_RMSE_OVER_CLASSIC_MAX,
    G52_RMSE_OVER_ZERO_MAX,
    G53_SIGN_MEDIAN_MIN,
    G53_SIGN_Q25_MIN,
    G53_SRE_MEDIAN_MAX,
    G53_SRE_Q75_MAX,
    G54_CLUSTER_MAX,
    G54_NEW_CLASH_MAX,
    G54_NEW_PROTECTED_MAX,
    G54_REFUSAL_MAX,
    G54_UPPER_BOUND_MAX,
    sealed_eval_request,
    write_gate_report,
)
from pes2ts_core.utils.jsonio import read_json, write_json  # noqa: E402

# Reuse the V5 sprint driver mechanics (window, refusal typing, subprocess
# plumbing, manifest/decision readers) — imported, NEVER fork-copied.
from scripts.campaigns.run_v5_sprint import (  # noqa: E402
    EXIT_CENSORED,
    EXIT_OK,
    EXIT_REFUSED,
    G51_TEST_FILES,
    PYTHON,
    SprintDriver,
    SprintRefused,
    StageCensored,
    StageWindow,
    _require,
    _resolve_anchor_mode,
    _sha256_file,
    _utc_now,
    t_run_seconds,
)

# ---------------------------------------------------------------------------
# Campaign identity / vocabulary (V5.1a — no complete_oof_* statuses, ever)
# ---------------------------------------------------------------------------

#: Config schema gate (read from config ``schema``; never the V5 draft2 one).
V51_SCHEMA: Final[str] = "pesn_v5_1_sprint_v1"
DEFAULT_CAMPAIGN_ID: Final[str] = "V5_1A_SPRINT"

V51_STAGE_ORDER: Final[tuple[str, ...]] = (
    "prepare", "diagnose", "ablate", "confirm", "report")

V51_MANIFEST_SCHEMA: Final[str] = "pesn_v5_1_sprint_manifest_v1"
V51_DECISION_SCHEMA: Final[str] = "pesn_v5_1_sprint_decision_v1"
V51_PROTOCOL_SCHEMA: Final[str] = "pesn_v5_1_protocol_v1"

#: V5.1a status vocabulary — DELIBERATELY has no ``complete_oof_*`` value:
#: V5.1a does no formal OOF (plan §0.1); those statuses belong to V5.1b/V5.2.
V51_STATUS_VALUES: Final[tuple[str, ...]] = (
    "complete_prepare_v51",
    "complete_diagnose_v51",
    "complete_ablate_v51",
    "complete_confirm_v51",
    "complete_report_v51",
    "censored_budget",
    "rejected_data",
    "rejected_gate",
    "insufficient_evidence",
)

#: Refusal codes that map to the ``rejected_gate`` stage status.
GATE_REFUSAL_CODES: Final[frozenset] = frozenset({
    "G51_NOT_GREEN",
    "G51_FAILED",
    "PROTOCOL_NOT_FROZEN",
    "PROTOCOL_HASH_MISMATCH",
})

#: Extra G5.1 file — required whenever the core v51 trainer code exists.
G51_V51_TEST_FILE: Final[str] = "tests/pesn/test_key_atom_trainer_v51.py"
V51_TRAINER_SOURCE: Final[str] = (
    "pes2ts_core/pesn/offline/training/key_atom_bc.py")

ABLATION_ARMS: Final[tuple[str, ...]] = ("B0", "B1", "B2", "B3")
ABLATION_SEEDS: Final[tuple[int, ...]] = (17, 29, 43)
CONFIRM_SEED: Final[int] = 17
#: Arms carrying the magnitude head (2×2: B1/B3) — confirmatory-only when
#: the D1 verdict is "checkpoint problem only".
MAGNITUDE_ARMS: Final[frozenset] = frozenset({"B1", "B3"})

SEALED_POLICY_V51: Final[str] = "v5_1a_never_unseals"

DEVELOPMENT_EVIDENCE_NOTE: Final[str] = (
    "V5.1a does NO formal OOF: every fold0 / inner-val number is "
    "development evidence only; formal G5.2/G5.3 claims belong to "
    "V5.1b/V5.2 (plan §0.1/§3.5)")

#: G5.3 dead-zone abstention cap — plan §5.1 table.  NOTE: unlike the other
#: gate thresholds this one has NO constant in key_atom_protocols.py
#: (verified), so it is recorded here with an explicit source annotation.
DEAD_ZONE_ABSTENTION_MAX: Final[float] = 0.10
DEAD_ZONE_ABSTENTION_SOURCE: Final[str] = (
    "plan §5.1 gate table (no constant exists in key_atom_protocols.py)")


def validate_status_v51(status: str) -> str:
    """V5.1a: only the exact nine statuses are admissible (never
    ``complete_oof_*``)."""
    if status not in V51_STATUS_VALUES:
        raise ValueError(
            f"status {status!r} not in V5.1a vocabulary "
            f"{list(V51_STATUS_VALUES)}")
    return status


class V51StageWindow(StageWindow):
    """Wall-clock envelope of a V5.1a stage.

    Every stage shares ONE campaign-wide envelope ``[0, wall_minutes]``
    measured from T0 (config ``wall_minutes``, default 240): the budget is
    cumulative, so a stage launched late inherits the remaining wall clock.
    The base class gate on V5 ``STAGE_OFFSETS_MIN`` is bypassed on purpose
    (this campaign has its own stage names); the launch-guard mechanics
    (``LAUNCH_GUARD_MINUTES``) are inherited unchanged.
    """

    def __init__(self, t0: float, stage: str, *,
                 offsets: Mapping[str, tuple[int, int]],
                 clock: Callable[[], float] = time.time) -> None:
        _require(stage in offsets, "STAGE_UNKNOWN",
                 f"unknown stage {stage!r}; known: {list(offsets)}")
        self.t0 = float(t0)
        self.stage = stage
        self.start_min, self.end_min = offsets[stage]
        self._clock = clock


# ---------------------------------------------------------------------------
# Pure planning helpers (hermetic; unit-tested)
# ---------------------------------------------------------------------------

def _finite_median(values: Sequence) -> float | None:
    """Median over finite, non-None values; ``None`` when nothing survives."""
    vals = sorted(float(v) for v in values
                  if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _first_present(mapping: Mapping, keys: Sequence[str]):
    for key in keys:
        if mapping.get(key) is not None:
            return mapping.get(key)
    return None


def d1_decision_tree(alpha_best: float | None) -> str:
    """Frozen D1 decision tree (plan §0.3 / §3.1).

    ``α_best ≥ 0.85`` → checkpoint problem only; ``0.70 ≤ α_best < 0.85`` →
    post-hoc scale calibration first; ``α_best < 0.70`` → magnitude head
    justified.  ``None`` (unmeasurable) → ``insufficient_evidence``.
    """
    if alpha_best is None or not math.isfinite(float(alpha_best)):
        return "insufficient_evidence"
    value = float(alpha_best)
    if value >= 0.85:
        return "checkpoint_problem_only"
    if value >= 0.70:
        return "post_hoc_calibration_first"
    return "magnitude_head_justified"


V51_DELIVERED_STATUSES = ("complete", "censored_patience")


def _run_delivered(status: Mapping | None,
                   run_dir: Path | str | None = None) -> bool:
    """A v51 run counts as delivered when the frozen protocol finished.

    ``censored_patience`` is the DESIGNED outcome of the inner-val patience
    rule (plan §3.4): training stops early, ``checkpoint_best.pt`` is
    retained and OOF-facing artifacts are final — a successful run, not a
    budget censoring.  A patience-stopped run without its best checkpoint
    is NOT delivered.
    """
    if not status:
        return False
    state = str(status.get("status"))
    if state == "complete":
        return True
    if state != "censored_patience":
        return False
    if run_dir is None:
        return True
    return (Path(run_dir) / "checkpoints" / "checkpoint_best.pt").is_file()


def select_best_arm(rows: Sequence[Mapping]) -> dict:
    """Frozen best-config selection rule for the 5-fold confirm run.

    Pre-registered in ``protocol_v51.json`` (frozen before training):

    1. primary: fold0 3-seed median inner-val SRE (lower better);
    2. subject to: rmse_over_zero ≤ B0 median × 1.05;
    3. tie-break: |α − 1| (closer to 1 first), then sign accuracy (higher).

    Arms with missing primary metrics are listed but never selected — a
    refusal to rank is honest; a fabricated rank is not.
    """
    arms: dict[str, list[dict]] = {}
    for row in rows:
        arms.setdefault(str(row.get("arm")), []).append(dict(row))
    per_arm: dict[str, dict] = {}
    for arm, items in arms.items():
        per_arm[arm] = {
            "n_seeds": len(items),
            "inner_val_sre_median": _finite_median(
                [i.get("inner_val_sre") for i in items]),
            "rmse_over_zero_median": _finite_median(
                [i.get("inner_val_rmse_over_zero") for i in items]),
            "alpha_median": _finite_median(
                [i.get("alpha_median") for i in items]),
            "sign_accuracy_median": _finite_median(
                [i.get("inner_val_sign_accuracy") for i in items]),
        }
    b0_rmse = (per_arm.get("B0") or {}).get("rmse_over_zero_median")
    rmse_limit = None if b0_rmse is None else float(b0_rmse) * 1.05
    eligible: list[str] = []
    for arm, metrics in per_arm.items():
        reasons: list[str] = []
        if metrics["inner_val_sre_median"] is None:
            reasons.append("inner_val_sre median missing")
        if rmse_limit is None or metrics["rmse_over_zero_median"] is None:
            reasons.append("rmse_over_zero vs B0 baseline undefined")
        elif metrics["rmse_over_zero_median"] > rmse_limit:
            reasons.append("rmse_over_zero median exceeds B0 median x1.05")
        metrics["eligible"] = not reasons
        metrics["ineligible_reasons"] = reasons
        if metrics["eligible"]:
            eligible.append(arm)

    def _sort_key(arm: str):
        metrics = per_arm[arm]
        alpha = metrics["alpha_median"]
        alpha_gap = abs(float(alpha) - 1.0) if alpha is not None else float("inf")
        sign = metrics["sign_accuracy_median"]
        sign_key = -float(sign) if sign is not None else float("inf")
        return (metrics["inner_val_sre_median"], alpha_gap, sign_key, arm)

    ranked = sorted(eligible, key=_sort_key)
    return {
        "schema_version": "pesn_v51_best_config_selection_v1",
        "rule": ("rank by fold0 3-seed median inner-val SRE (lower better), "
                 "subject to rmse_over_zero <= B0 median x1.05, tie-break "
                 "|alpha-1| then sign accuracy; frozen pre-training "
                 "(protocol_v51.json)"),
        "rmse_limit_over_b0": rmse_limit,
        "per_arm": per_arm,
        "ranking": ranked,
        "best_arm": ranked[0] if ranked else None,
        "selection_exposed": True,
        "development_evidence_only": True,
        "decided_at": _utc_now(),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class V51SprintDriver(SprintDriver):
    """Crash-restart-safe orchestrator over the five V5.1a stages.

    Subclasses the V5 ``SprintDriver`` and reuses its plumbing (manifest /
    decisions / parent-hash verification / subprocess spawn-or-skip / timing
    probe mechanics); everything V5.1-specific — stage set, status
    vocabulary, protocol freeze, D1/ablate/confirm/report logic — is
    overridden or added here.  The base ``__init__`` is NOT called because
    it hard-gates the V5 ``pesn_v5_sprint_draft2`` schema.
    """

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
        _require(self.config.get("schema") == V51_SCHEMA,
                 "CONFIG_SCHEMA",
                 f"expected schema {V51_SCHEMA}, got "
                 f"{self.config.get('schema')!r}")
        self.paths = dict(self.config.get("paths") or {})
        self.sprint = dict(self.config.get("sprint") or {})
        self.v51 = dict(self.config.get("v51") or {})
        self.campaign_id = str(
            self.sprint.get("campaign_id") or DEFAULT_CAMPAIGN_ID)
        self.allow_truth = bool(allow_truth)
        self.unseal = bool(unseal)
        self.force = bool(force)
        self.dry_run = bool(dry_run)
        self.device = device
        self._clock = clock
        self.out_root = Path(out_root) if out_root is not None else self._p(
            "output_root", "outputs/pesn/v5_1/sprint")
        self.manifest_path = self.out_root / "campaign_manifest.json"
        self.decisions_path = self.out_root / "stage_decisions.jsonl"
        self.manifest: dict | None = None
        self._load_manifest()

    # -- derived roots / budgets --------------------------------------------

    @property
    def runs_root(self) -> Path:
        return self._p("runs_root", "outputs/pesn/v5_1/runs")

    @property
    def protocol_path(self) -> Path:
        return self._p("protocol", "outputs/pesn/v5_1/protocol_v51.json")

    @property
    def diagnostics_root(self) -> Path:
        return self._p("diagnostics_root", "outputs/pesn/v5_1/diagnostics")

    @property
    def report_root(self) -> Path:
        return self._p("report_root", "outputs/pesn/v5_1")

    @property
    def eval_every(self) -> int:
        return int(self.config.get("eval_every_updates", 50))

    @property
    def formal_updates(self) -> int:
        return int(self.config.get("formal_updates", 3000))

    @property
    def wall_minutes(self) -> int:
        return int(self.config.get("wall_minutes", 240))

    def _stage_offsets(self) -> dict[str, tuple[int, int]]:
        """One campaign-wide envelope: every stage ends at T+wall_minutes."""
        return {stage: (0, self.wall_minutes) for stage in V51_STAGE_ORDER}

    def _run_dir(self, run_id: str) -> Path:
        # V5.1 runs live under outputs/pesn/v5_1/runs (NOT out_root/runs).
        return self.runs_root / run_id

    # -- manifest / decisions (V5.1 schemas; dry-run writes nothing) --------

    def ensure_manifest(self) -> dict:
        """Freeze T0 on the first REAL execution (never in dry-run)."""
        if self.manifest is None:
            _require(not self.dry_run, "DRY_RUN_WRITE",
                     "dry-run must never freeze T0 or write the manifest")
            self.out_root.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "schema_version": V51_MANIFEST_SCHEMA,
                "campaign_id": self.campaign_id,
                "T0_unix": float(self._clock()),
                "T0_iso": _utc_now(),
                "config_path": str(self.config_path),
                "config_sha256": _sha256_file(self.config_path),
                "arms": list(self.sprint.get("arms") or ABLATION_ARMS),
                "stage_offsets_minutes": {
                    k: list(v) for k, v in self._stage_offsets().items()},
                "status_vocabulary": list(V51_STATUS_VALUES),
                "protocol_path": str(self.protocol_path),
                "oof_policy": ("V5.1a does NO formal OOF; the campaign "
                               "status vocabulary contains no formal-OOF "
                               "completion status (plan §0.1)"),
                "unseal_policy": (f"sealed eval policy {SEALED_POLICY_V51!r} "
                                  "— never unseals (allow_unseal=False)"),
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
        payload = {"schema_version": V51_DECISION_SCHEMA,
                   "at_unix": float(self._clock()), **dict(decision)}
        with self.decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True,
                                ensure_ascii=False) + "\n")

    def _record_stage(self, stage: str, status: str, *, detail: str = "",
                      extra: Mapping | None = None) -> str:
        validate_status_v51(status)
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

    # -- protocol freeze (plan §2: thresholds frozen before training) -------

    @staticmethod
    def _code_commit() -> str | None:
        """HEAD commit of the repo (best-effort; ``None`` when unavailable)."""
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=15)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return None

    def _protocol_payload(self) -> dict:
        """Exact frozen protocol content (plan §5.1/§5.2/§3.3/§3.4).

        Gate thresholds come from the ``key_atom_protocols`` constants —
        imported, never retyped.  The one cap with no module constant
        (dead-zone abstention) carries an explicit source annotation.
        """
        inner_val = dict(self.v51.get("inner_val") or {})
        selection = dict(inner_val.get("selection") or {})
        magnitude_head = dict(self.v51.get("magnitude_head") or {})
        pair_exact = dict(self.v51.get("pair_exact") or {})
        ramp = dict(self.v51.get("ramp") or {})
        arm_table = {
            str(arm): dict(entry or {})
            for arm, entry in dict(self.sprint.get("arm_table") or {}).items()}
        return {
            "schema_version": V51_PROTOCOL_SCHEMA,
            "campaign_id": self.campaign_id,
            "source_plan": ("docs/plans/PES2TS_V5.1_修复冲刺与扩量准入"
                            "方案_v1_20261019.md"),
            "created_at": _utc_now(),
            "frozen_before_training": True,
            "statement": ("thresholds frozen pre-training, no post-hoc "
                          "relaxation"),
            "gate_thresholds": {
                "source": ("pes2ts_core/pesn/offline/evaluation/"
                           "key_atom_protocols.py constants — imported, "
                           "never retyped"),
                "G5.2": {
                    "mae_over_zero_max": G52_MAE_OVER_ZERO_MAX,
                    "rmse_over_zero_max": G52_RMSE_OVER_ZERO_MAX,
                    "rmse_over_classic_max": G52_RMSE_OVER_CLASSIC_MAX,
                    "q75_rmse_over_zero_max": G52_Q75_RMSE_OVER_ZERO_MAX,
                    "frac_better_than_zero_min":
                        G52_BETTER_THAN_ZERO_FRACTION_MIN,
                    "cos_overall_min": G52_COS_OVERALL_MIN,
                    "cos_near_ts_min": G52_COS_NEAR_TS_MIN,
                    "min_clusters": G52_MIN_CLUSTERS,
                    "min_near_ts_clusters": G52_MIN_NEAR_TS_CLUSTERS,
                },
                "G5.3": {
                    "sign_median_min": G53_SIGN_MEDIAN_MIN,
                    "sign_q25_min": G53_SIGN_Q25_MIN,
                    "sre_median_max": G53_SRE_MEDIAN_MAX,
                    "sre_q75_max": G53_SRE_Q75_MAX,
                    "dead_zone_abstention_max": DEAD_ZONE_ABSTENTION_MAX,
                    "dead_zone_abstention_source":
                        DEAD_ZONE_ABSTENTION_SOURCE,
                },
                "G5.4": {
                    "new_clash_max": G54_NEW_CLASH_MAX,
                    "new_protected_max": G54_NEW_PROTECTED_MAX,
                    "bootstrap_upper95_max": G54_UPPER_BOUND_MAX,
                    "per_cluster_max": G54_CLUSTER_MAX,
                    "refusal_max": G54_REFUSAL_MAX,
                },
            },
            "g52_v51_full_oof_protocol": {
                "aggregation": "49-cluster cluster-median over full OOF",
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "classic_improvement_lower95_gt":
                    G52_CLASSIC_IMPROVEMENT_LOWER_MIN,
                "folds_point_estimate_pass_min": "4/5",
                "fold_rmse_over_zero_median_max": 1.0,
                "v5_9_cluster_gate": (
                    "demoted to secondary/historical — the V5 formal gate is "
                    "9-cluster and must never be swapped post-hoc"),
                "effective_atom_norm_angstrom": EFFECTIVE_ATOM_TAU,
                "status": ("frozen for V5.1b/V5.2 — V5.1a runs NO formal "
                           "OOF (plan §0.1/§5.2)"),
            },
            "m1_amplitude_gate": {
                "alpha_median_pass_range": [0.85, 1.05],
                "alpha_q25_min": 0.70,
                "alpha_q75_max": 1.30,
                "effective_atom_norm_angstrom": EFFECTIVE_ATOM_TAU,
                "alpha_definition": (
                    "per-atom ratio ||u_pred|| / ||y_teacher|| on effective "
                    "atoms (||y|| > EFFECTIVE_ATOM_TAU); per-cluster median "
                    "over effective atoms; aggregate = median / q25 / q75 "
                    "over cluster medians. FROZEN convention for D1 verdict, "
                    "M1 gate and all V5.1 reports."),
                "v5_baseline_reconciliation": (
                    "V5 final-checkpoint OOF baseline under this convention: "
                    "alpha_median = 0.6757 (d0=0.05, 5 formal runs seed 17, "
                    "outputs/pesn/v5_1/diagnostics/baseline_oof_v5.json). "
                    "The plan-cited 0.52 is NOT reproducible under any tested "
                    "convention (all-atom 0.8104, rms-ratio 0.7208, "
                    "frame-mean-norm 0.6765); citation convention was "
                    "unrecorded. Decision-tree thresholds (0.85 / 0.70) are "
                    "kept as frozen in plan §0.3 and applied to THIS "
                    "convention against the 0.6757 baseline."),
                "source": "plan §5.1 gate table (M1) + diagnostics "
                          "reconciliation 2026-10-19",
            },
            "s1_stability_gate": {
                "seeds_pass_min": "2/3",
                "ensemble_required": True,
                "source": "plan §5.1 gate table (S1)",
            },
            "stretch_targets": {
                "alpha_median_pass_range": [0.90, 1.05],
                "sign_median_min": 0.92,
                "sign_q25_min": 0.85,
                "sre_median_max": 0.12,
                "sre_q75_max": 0.20,
                "dead_zone_abstention_max": 0.05,
                "source": "plan §5.1 Stretch column",
            },
            "d1_decision_tree": {
                "alpha_best_ge_0.85": "checkpoint_problem_only",
                "0.70_le_alpha_best_lt_0.85": "post_hoc_calibration_first",
                "alpha_best_lt_0.70": "magnitude_head_justified",
                "source": "plan §0.3 / §3.1 (frozen before D1 runs)",
            },
            "frozen_lambdas": {
                "lambda_cos": float(self.config.get("lambda_cos", 0.02)),
                "lambda_mag": float(magnitude_head.get("lambda_mag", 0.5)),
                "lambda_sign": float(pair_exact.get("lambda_sign", 0.2)),
                "lambda_dead": float(pair_exact.get("lambda_dead", 0.1)),
                "lambda_pair_reg": float(pair_exact.get("lambda_pair_reg",
                                                        0.5)),
                "ramp": {
                    "start_update": int(ramp.get("start_update", 250)),
                    "ramp_updates": int(ramp.get("ramp_updates", 250)),
                },
            },
            "inner_val_rule": {
                "n_train_clusters": 32,
                "n_val_clusters": int(inner_val.get("n_val_clusters", 8)),
                "stratify": list(inner_val.get("stratify") or
                                 ["family", "size"]),
                "seed": int(inner_val.get("seed", 20261019)),
                "eval_every_updates": self.eval_every,
                "primary_metric": ("inner-val cluster-median RMSE/zero with "
                                   "SRE / nearTS cos tie-breaks"),
                "selection": ("rmse <= best x tolerance -> min SRE -> max "
                              "nearTS cos"),
                "rmse_tolerance": float(selection.get("rmse_tolerance", 1.02)),
                "patience_checks": int(selection.get("patience_checks", 8)),
                "min_delta": float(selection.get("min_delta", 0.005)),
                "min_update_floor": int(selection.get("min_update_floor",
                                                      500)),
                "checkpoint_policy": ("save checkpoint_best.pt + "
                                      "checkpoint_last.pt; outer dev NEVER "
                                      "selects checkpoints (plan §3.4)"),
            },
            "arm_matrix": {
                "arms": list(self.sprint.get("arms") or ABLATION_ARMS),
                "arm_table": arm_table,
                "design": ("2×2: magnitude head {off,on} × pair loss "
                           "{legacy_j, exact_trio} on fold0 x seeds "
                           "{17,29,43} (plan §4)"),
            },
            "best_config_selection_rule": {
                "primary": "fold0 3-seed median inner-val SRE (lower better)",
                "constraint": ("rmse_over_zero <= B0 median x 1.05"),
                "tie_break": ["|alpha - 1|", "sign accuracy (higher first)"],
                "note": ("selection-exposed development evidence; the frozen "
                         "winner feeds the 5-fold confirm, not a formal "
                         "OOF claim"),
            },
            "hashes": {
                "code_commit": self._code_commit(),
                "config_sha256": _sha256_file(self.config_path),
                "split_manifest_sha256": _sha256_file(self._p(
                    "split_manifest",
                    "data/manifests/v5/split_manifest.json")),
            },
            "thresholds_frozen_pre_training": True,
        }

    def _freeze_protocol(self) -> dict:
        """Write ``protocol_v51.json`` (idempotent; --force re-freezes).

        Called ONLY from the ``prepare`` stage — never by a training stage.
        """
        path = self.protocol_path
        if path.is_file() and not self.force:
            existing = read_json(path) or {}
            recorded = ((existing.get("hashes") or {}).get("config_sha256"))
            current = _sha256_file(self.config_path)
            _require(recorded is not None and recorded == current,
                     "PROTOCOL_HASH_MISMATCH",
                     f"config changed after the protocol was frozen "
                     f"({path}); thresholds may not drift — re-freeze via "
                     "--force --stage prepare with an explicit decision")
            self.append_decision({
                "stage": "prepare", "decision": "protocol_already_frozen",
                "protocol_path": str(path),
                "protocol_sha256": _sha256_file(path)})
            return existing
        payload = self._protocol_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
        self.append_decision({
            "stage": "prepare", "decision": "protocol_frozen",
            "protocol_path": str(path),
            "protocol_sha256": _sha256_file(path),
            "config_sha256": payload["hashes"]["config_sha256"],
            "statement": payload["statement"]})
        return payload

    def _require_protocol(self) -> dict:
        """Gate every post-prepare stage on the frozen protocol."""
        path = self.protocol_path
        _require(path.is_file(), "PROTOCOL_NOT_FROZEN",
                 f"protocol not found: {path} — run --stage prepare first "
                 "(thresholds must freeze BEFORE any V5.1 training)")
        doc = read_json(path) or {}
        recorded = ((doc.get("hashes") or {}).get("config_sha256"))
        current = _sha256_file(self.config_path)
        _require(recorded is not None and recorded == current,
                 "PROTOCOL_HASH_MISMATCH",
                 f"frozen protocol records config_sha256={recorded!r} but "
                 f"the current config hashes to {current!r} — thresholds "
                 "are frozen; fix the config or re-freeze explicitly")
        return doc

    # -- summaries -----------------------------------------------------------

    def _prepare_summary(self) -> dict:
        path = self.out_root / "prepare_summary_v51.json"
        if not path.is_file():
            return {}
        try:
            return read_json(path) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _read_d1_diagnostics(self) -> dict:
        path = self.diagnostics_root / "d1_best_checkpoint_alpha.json"
        if not path.is_file():
            return {}
        try:
            return read_json(path) or {}
        except (OSError, json.JSONDecodeError):
            return {}

    # -- G5.1 evidence (stage prepare, part a) -------------------------------

    @staticmethod
    def _v51_trainer_code_present() -> bool:
        """Heuristic: the core trainer source mentions the ``v51`` mode."""
        src = PROJECT_ROOT / V51_TRAINER_SOURCE
        if not src.is_file():
            return False
        try:
            text = src.read_text(encoding="utf-8")
        except OSError:
            return False
        return '"v51"' in text or "'v51'" in text

    def _g51_evidence(self) -> dict:
        """Run the G5.1 fixture suite (+ v51 trainer tests once they exist).

        REFUSED ``G51_NOT_GREEN`` when pytest fails OR when the v51 trainer
        code is present but ``tests/pesn/test_key_atom_trainer_v51.py`` is
        missing (a new loss/head without fixtures may never train).
        """
        files = list(G51_TEST_FILES)
        v51_test = PROJECT_ROOT / G51_V51_TEST_FILE
        code_present = self._v51_trainer_code_present()
        v51_test_included = v51_test.is_file()
        if v51_test_included:
            files.append(G51_V51_TEST_FILE)
        proc = subprocess.run(
            [PYTHON, "-m", "pytest", *files, "-q"],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        summary = {
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout.strip().splitlines()[-1:]
                            if proc.stdout.strip() else []),
            "files": files,
            "v51_trainer_test": (
                "included" if v51_test_included else
                "absent — skipped (v51 trainer code not present yet)"
                if not code_present else "absent"),
            "v51_trainer_code_present": code_present,
        }
        if proc.returncode != 0:
            raise SprintRefused(
                "G51_NOT_GREEN",
                f"G5.1 fixture suite failed (pytest exit {proc.returncode}); "
                "V5.1a may not train on red fixtures")
        if code_present and not v51_test_included:
            raise SprintRefused(
                "G51_NOT_GREEN",
                f"v51 trainer code is present but {G51_V51_TEST_FILE} is "
                "missing — G5.1 evidence incomplete (plan §2: every new "
                "loss/head must pass fixtures first)")
        return summary

    # -- timing probe (part b): eval interval parametrized to 50 -----------

    def _timing_probe(self) -> dict:
        """Re-run the V5 GPU probe, then re-model t_run at eval interval 50.

        Writes ``outputs/pesn/v5_1/sprint/timing_probe.json`` (out_root)
        with ``t_run_seconds`` computed via the parametrized
        ``t_run_seconds(..., eval_every=50)``.
        """
        probe = super()._timing_probe()
        every = self.eval_every
        step = float(probe["t_step_p90_s"])
        ev = float(probe["t_eval_s"])
        save = float(probe["t_start_save_s"])
        probe["schema_version"] = "pesn_v51_timing_probe_v1"
        probe["eval_every_updates"] = int(every)
        probe["t_run_seconds"] = {
            str(u): t_run_seconds(u, step, ev, save, eval_every=every)
            for u in (500, 1000, 1500, 3000)}
        probe["t_run_formula"] = (
            "t_run(U) = U*t_step_P90 + ceil(U/eval_every)*t_eval + "
            f"t_start_save with eval_every={every} (V5.1 inner-val cadence)")
        write_json(self.out_root / "timing_probe.json", probe)
        return probe

    # -- run metric readers (defensive against the core v51 lane) -----------

    def _run_metrics(self, run_dir: Path) -> dict:
        """Merge whatever inner-val metrics a run produced (never faked).

        Reads ``status.json`` / ``metrics.json`` / ``eval.jsonl`` and, when
        present, the best-checkpoint update.  Missing fields stay ``None``
        (the exact key spellings tolerated are listed per field).
        """
        out: dict = {}
        status = self._run_status(run_dir) or {}
        for key in ("best_update", "alpha_median", "dead_zone_fraction"):
            if status.get(key) is not None:
                out[key] = status[key]
        aggregate: dict = {}
        metrics_path = Path(run_dir) / "metrics.json"
        if metrics_path.is_file():
            try:
                metrics_doc = read_json(metrics_path) or {}
            except (OSError, json.JSONDecodeError):
                metrics_doc = {}
            raw_series = list(metrics_doc.get("raw") or [])
            if raw_series:
                aggregate = dict(
                    (raw_series[-1] or {}).get("aggregate") or {})
            for key in ("best_update", "alpha_median", "dead_zone_fraction"):
                if out.get(key) is None and metrics_doc.get(key) is not None:
                    out[key] = metrics_doc[key]
        series = self._eval_series(Path(run_dir))
        inner_row = None
        if series:
            inner_rows = [r for r in series
                          if str(r.get("split") or "") == "inner_val"]
            if inner_rows:
                # v51 leak guard (plan §2/§3.4): arm selection and ablation
                # tables must consume the INNER-VAL row at best_update — the
                # same numbers that drove checkpoint selection.  Post-hoc
                # dev_dense rows are report-only and never selected on.
                best_update = out.get("best_update")
                if best_update is None:
                    best_ckpt = (Path(run_dir) / "checkpoints"
                                 / "checkpoint_best.pt")
                    if best_ckpt.is_file():
                        best_update = self._checkpoint_update(best_ckpt)
                by_update = {r.get("update"): r for r in inner_rows
                             if r.get("update") is not None}
                inner_row = (by_update.get(best_update)
                             or inner_rows[-1] or None)
            last = series[-1] or {}
            if not inner_row and last.get("update") is not None:
                out.setdefault("final_update", last.get("update"))
            if not inner_row and last.get("wall_s") is not None:
                out.setdefault("wall_time_s", last.get("wall_s"))
            source = inner_row or (series[-1] or {})
            aggregate = {**aggregate, **(source.get("aggregate") or {})}
            out["metrics_source_split"] = str(source.get("split") or "")
            if source.get("update") is not None:
                out["metrics_source_update"] = source.get("update")
        out["rmse_over_zero"] = aggregate.get("rmse_over_zero")
        out["near_ts_cos"] = aggregate.get("cos_eff_near_ts_median")
        out["sre"] = _first_present(aggregate, (
            "sre", "sre_median", "sre_cluster_median"))
        out["sign_accuracy"] = _first_present(aggregate, (
            "sign_accuracy", "sign_accuracy_median",
            "sign_accuracy_cluster_median"))
        if out.get("alpha_median") is None:
            out["alpha_median"] = _first_present(aggregate, (
                "alpha_median", "alpha"))
        if out.get("dead_zone_fraction") is None:
            out["dead_zone_fraction"] = _first_present(aggregate, (
                "dead_zone_fraction", "dead_zone"))
        if out.get("best_update") is None:
            best_ckpt = Path(run_dir) / "checkpoints" / "checkpoint_best.pt"
            if best_ckpt.is_file():
                out["best_update"] = self._checkpoint_update(best_ckpt)
        return out

    @staticmethod
    def _checkpoint_update(path: Path) -> int | None:
        """Best-effort ``update`` index stored inside a checkpoint file."""
        try:
            import torch

            doc = torch.load(path, map_location="cpu")
            if isinstance(doc, dict) and doc.get("update") is not None:
                return int(doc["update"])
        except Exception:  # torch absent / unreadable checkpoint → None
            pass
        return None

    def _spawn_or_skip(self, *, arm: str, fold: int, seed: int, updates: int,
                       mode: str, stage: str, window: StageWindow,
                       run_id: str | None = None,
                       extra: Mapping | None = None) -> dict:
        """Base skip-if-complete semantics with an explicit ``run_id`` tag.

        ``run_id`` decouples the on-disk dir from the trainer mode (D1 uses
        ``B0_f0_s17_d1`` while the trainer still gets ``--mode v51``).
        """
        rid = run_id or self._run_id(arm, fold, seed, mode)
        run_dir = self._run_dir(rid)
        status = self._run_status(run_dir)
        if (not self.force and _run_delivered(status, run_dir)):
            self.append_decision({
                "stage": stage, "decision": "skip_complete",
                "arm": arm, "fold": fold, "seed": seed, "mode": mode,
                "run_dir": str(run_dir), **dict(extra or {})})
            return {"run_dir": str(run_dir), "skipped": True,
                    **dict(extra or {}), **status}
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
            **dict(extra or {}),
        }

    # -- precheck / dispatch -------------------------------------------------

    def _precheck(self, stage: str) -> None:
        _require(self._p("split_manifest",
                         "data/manifests/v5/split_manifest.json").is_file(),
                 "SPLIT_MANIFEST_MISSING",
                 "V5.1a reuses the frozen V5 5-fold split verbatim "
                 "(split hash unchanged, plan §2) — run the V5 sprint "
                 "prepare-audit first")
        if stage == "prepare":
            return
        # Every post-prepare stage is gated on the frozen protocol first:
        self._require_protocol()
        if stage in ("ablate", "confirm"):
            _require((self.out_root / "timing_probe.json").is_file(),
                     "TIMING_PROBE_MISSING",
                     "run --stage prepare first (timing probe @ "
                     "eval_every=50)")
            summary = self._prepare_summary()
            _require(bool(summary), "PREPARE_NOT_RUN",
                     "run --stage prepare first")
            _require(summary.get("g51_pass") is True, "G51_NOT_GREEN",
                     "G5.1 fixtures must be green before any V5.1 training "
                     "stage (plan §2)")
        if stage == "confirm":
            _require((self.out_root / "ablation_v51.json").is_file(),
                     "ABLATION_MISSING",
                     "run --stage ablate first (best-config selection needs "
                     "the 12-run ablation table)")

    def run_stage(self, stage: str) -> str | int:
        _require(stage in V51_STAGE_ORDER, "STAGE_UNKNOWN",
                 f"unknown stage {stage!r}; known: {list(V51_STAGE_ORDER)}")
        if self.dry_run:
            return self._dry_run_stage(stage)
        window = V51StageWindow(
            self.ensure_manifest()["T0_unix"], stage,
            offsets=self._stage_offsets(), clock=self._clock)
        try:
            self._precheck(stage)
            handler = getattr(self, self.STAGE_HANDLERS[stage])
            return handler(window)
        except StageCensored as exc:
            return self._record_stage(stage, "censored_budget",
                                      detail=str(exc))
        except SprintRefused as exc:
            status = ("rejected_gate" if exc.code in GATE_REFUSAL_CODES
                      else "rejected_data")
            self._record_stage(stage, status, detail=str(exc))
            raise

    STAGE_HANDLERS: Final[dict[str, str]] = {
        "prepare": "_stage_prepare",
        "diagnose": "_stage_diagnose",
        "ablate": "_stage_ablate",
        "confirm": "_stage_confirm",
        "report": "_stage_report",
    }

    # -- Stage 1: prepare ----------------------------------------------------

    def _stage_prepare(self, window: StageWindow) -> str:
        window.ensure_launch()
        t_stage = time.time()
        checkpoints: list[dict] = []

        def _checkpoint(name: str, t_mark: float, **extra) -> None:
            checkpoints.append({
                "checkpoint": name,
                "at_unix": float(self._clock()),
                "elapsed_s": time.time() - t_mark, **extra})

        self.ensure_manifest()
        self._verify_parents("prepare")

        # (a) G5.1 fixture evidence — REFUSED G51_NOT_GREEN on red fixtures
        #     or a v51 trainer without its test file.
        t_mark = time.time()
        g51 = self._g51_evidence()
        _checkpoint("g51_evidence", t_mark, g51_pass=True,
                    files=g51.get("files"))

        # (b) timing probe re-run with eval interval 50 (parametrized t_run)
        t_mark = time.time()
        try:
            probe = self._timing_probe()
        except SprintRefused as exc:
            self._record_stage("prepare", "rejected_data", detail=str(exc))
            raise
        _checkpoint("timing_probe", t_mark,
                    t_step_p90_s=probe["t_step_p90_s"],
                    t_eval_s=probe["t_eval_s"],
                    eval_every_updates=int(probe.get("eval_every_updates",
                                                     self.eval_every)))

        # (c) FREEZE the protocol BEFORE any training stage may run.
        t_mark = time.time()
        protocol = self._freeze_protocol()
        _checkpoint("protocol_frozen", t_mark,
                    protocol_path=str(self.protocol_path),
                    protocol_sha256=_sha256_file(self.protocol_path),
                    statement=protocol.get("statement"))

        summary = {
            "schema_version": "pesn_v51_prepare_summary_v1",
            "campaign_id": self.campaign_id,
            "checkpoints": checkpoints,
            "g51_pass": True,
            "g51": g51,
            "timing_probe": str(self.out_root / "timing_probe.json"),
            "eval_every_updates": self.eval_every,
            "protocol": str(self.protocol_path),
            "protocol_sha256": _sha256_file(self.protocol_path),
            "no_formal_oof": True,
            "measured_at": _utc_now(),
        }
        write_json(self.out_root / "prepare_summary_v51.json", summary)
        self.append_decision({
            "stage": "prepare", "decision": "prepare_complete",
            "g51_pass": True,
            "protocol_sha256": summary["protocol_sha256"],
            "t_step_p90_s": probe["t_step_p90_s"],
            "wall_s": time.time() - t_stage})
        self._record_stage(
            "prepare", "complete_prepare_v51",
            detail=("G5.1 green, timing probe @ eval_every="
                    f"{self.eval_every}, protocol frozen BEFORE training"),
            extra={"protocol_sha256": summary["protocol_sha256"],
                   "g51_files": g51.get("files")})
        print(f"[prepare] complete: G5.1 green, protocol frozen at "
              f"{self.protocol_path} (t_step_p90={probe['t_step_p90_s']:.3f}s,"
              f" eval_every={self.eval_every})")
        return "complete_prepare_v51"

    # -- Stage 2: diagnose (D1) ---------------------------------------------

    def _stage_diagnose(self, window: StageWindow) -> str:
        self._verify_parents("diagnose")
        plan = dict(self.sprint.get("diagnose") or {})
        arm = str(plan.get("arm", "B0"))
        fold = int(plan.get("fold", 0))
        seed = int(plan.get("seed", 17))
        updates = int(plan.get("updates", self.formal_updates))
        run_id = str(plan.get("run_id", f"{arm}_f{fold}_s{seed}_d1"))
        entry = dict((self.sprint.get("arm_table") or {}).get(arm) or {})
        _require(not bool(entry.get("magnitude_head", False))
                 and str(entry.get("pair_loss", "legacy_j")) == "legacy_j",
                 "D1_ARM_NOT_LEGACY",
                 "D1 must be the legacy checkpoint-effect measurement arm "
                 "(magnitude head off, pair_loss legacy_j) — plan §3.1")
        try:
            record = self._spawn_or_skip(
                arm=arm, fold=fold, seed=seed, updates=updates, mode="v51",
                stage="diagnose", window=window, run_id=run_id,
                extra={"tag": "d1",
                       "purpose": "checkpoint-effect measurement (plan §3.1 D1)"})
        except StageCensored as exc:
            return self._record_stage("diagnose", "censored_budget",
                                      detail=str(exc))
        run_dir = Path(record["run_dir"])
        diag = self._alpha_diagnostics(run_dir=run_dir)
        diag_path = self.diagnostics_root / "d1_best_checkpoint_alpha.json"
        run_ok = str(record.get("status")) == "complete"
        if diag is None:
            doc = {
                "schema_version": "pesn_v51_d1_alpha_v1",
                "status": "not_measured",
                "run_dir": str(run_dir),
                "arm": arm, "fold": fold, "seed": seed, "tag": "d1",
                "reason": ("run did not produce a checkpoint"
                           if not run_ok else "no checkpoint on disk"),
                "decision_tree_verdict": None,
                "development_evidence_only": True,
                "measured_at": _utc_now(),
            }
            status = "insufficient_evidence"
            detail = "D1 produced no checkpoint; α diagnostic not measured"
        else:
            doc = diag
            status = "complete_diagnose_v51" if run_ok else "insufficient_evidence"
            detail = (f"D1 measured: α_best="
                      f"{doc.get('alpha_best_median')} → "
                      f"{doc.get('decision_tree_verdict')}")
        self.diagnostics_root.mkdir(parents=True, exist_ok=True)
        write_json(diag_path, doc)
        self._record_stage(
            "diagnose", status, detail=detail,
            extra={"diagnostics": str(diag_path), "run_dir": str(run_dir),
                   "decision_tree_verdict": doc.get("decision_tree_verdict"),
                   "checkpoint_policy": doc.get("checkpoint_policy")})
        print(f"[diagnose] {status}: {detail}")
        return status

    def _alpha_diagnostics(self, *, run_dir: Path) -> dict | None:
        """Post-hoc α = ‖u_pred‖/‖y_teacher‖ on the OUTER dev fold (D1).

        Measures BOTH checkpoints under the frozen policy: α_best from
        ``checkpoint_best.pt`` (falls back to ``checkpoint_final.pt`` and
        says so) and α_final from the endpoint ``checkpoint_final.pt``.
        Strata: d_eff≈d0 / terminal / capped / nearTS / per-cluster.
        Strictly post-hoc: it never selects checkpoints and never feeds a
        gate claim (the decision tree consumes α_best only).
        """
        import numpy as np
        import torch

        from pes2ts_core.pesn.contracts.key_atom import load_key_atom_task
        from pes2ts_core.pesn.models.key_atom_egnn import StarEGNN
        from pes2ts_core.pesn.offline.training.device import resolve_device
        from pes2ts_core.pesn.offline.training.key_atom_bc import (
            load_lane_b_pools, load_lane_b_view)

        best_path = Path(run_dir) / "checkpoints" / "checkpoint_best.pt"
        final_path = Path(run_dir) / "checkpoints" / "checkpoint_final.pt"
        if best_path.is_file():
            best_ckpt, best_policy = best_path, "checkpoint_best.pt"
        elif final_path.is_file():
            best_ckpt, best_policy = final_path, (
                "checkpoint_final.pt fallback (no checkpoint_best.pt on "
                "disk — V5.1 best-checkpoint contract not produced)")
        else:
            return None

        dev_cfg = dict(self.config.get("device") or {})
        try:
            _torch, device = resolve_device(
                device=self.device or str(dev_cfg.get("device", "cuda:0")),
                require_cuda=bool(dev_cfg.get("require_cuda", True)))
        except Exception as exc:
            raise SprintRefused("ALPHA_DIAG_DEVICE", str(exc)) from exc

        split_path = str(self._p("split_manifest",
                                 "data/manifests/v5/split_manifest.json"))
        task_root = self._p("task_root", "outputs/pesn/v5/tasks")
        arm = str((self.sprint.get("diagnose") or {}).get("arm", "B0"))
        arm_cfg = dict((self.sprint.get("arm_table") or {}).get(arm) or {})
        d0 = float(arm_cfg.get("d0", 0.05))
        anchor = _resolve_anchor_mode(arm_cfg.get("anchor", "on"))
        cap = float(self.config.get("teacher_atom_cap_angstrom", 0.20))
        try:
            pools = load_lane_b_pools(self.config, fold=0, d0=d0,
                                      split_path=split_path)
            dev_rows = list(pools["dev_dense"])
            view, collate = load_lane_b_view(
                config=self.config, samples=dev_rows, fold=0, d0=d0,
                anchor=anchor, split_path=split_path, device=device)
        except SprintRefused:
            raise
        except Exception as exc:
            raise SprintRefused("ALPHA_DIAG_DATA", str(exc)) from exc

        batch_size = int(self.config.get("batch_size", 32))
        tasks: dict = {}

        def _measure_checkpoint(ckpt_path: Path) -> dict:
            """One forward pass over the outer dev fold → stratum medians."""
            model = StarEGNN().to(torch.float32).to(device)
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state["model"])
            model.train(False)
            rows: list[dict] = []
            with torch.no_grad():
                for i in range(0, len(view), batch_size):
                    chunk = [view[j] for j in
                             range(i, min(i + batch_size, len(view)))]
                    if not chunk:
                        continue
                    batch = collate(chunk)
                    u, _z = model(batch)
                    u_np = u.float().cpu().numpy()
                    y_np = (batch.y_teacher.cpu().numpy()
                            if batch.y_teacher is not None
                            else np.zeros_like(u_np))
                    key_ptr = batch.key_ptr.cpu().numpy()
                    ptr = batch.ptr.cpu().numpy()
                    z_np = batch.z_atomic.cpu().numpy()
                    key_index = batch.key_index.cpu().numpy()
                    near = (batch.near_ts.cpu().numpy()
                            if batch.near_ts is not None
                            else np.ones(len(key_ptr) - 1))
                    branches = (batch.branch.cpu().numpy()
                                if batch.branch is not None
                                else np.zeros(len(key_ptr) - 1,
                                              dtype=np.int64))
                    for b in range(len(key_ptr) - 1):
                        rid = str(batch.reaction_ids[b])
                        branch = int(branches[b])
                        if rid not in tasks:
                            tasks[rid] = load_key_atom_task(
                                task_root / f"{rid}.task.json")
                        task = tasks[rid]
                        n_lo, n_hi = int(ptr[b]), int(ptr[b + 1])
                        k_lo, k_hi = int(key_ptr[b]), int(key_ptr[b + 1])
                        # start-graph heavy degree → terminal stratum
                        neighbours: dict[int, list[int]] = {}
                        for (a, c, _order) in task.start_bonds(branch):
                            neighbours.setdefault(int(a), []).append(int(c))
                            neighbours.setdefault(int(c), []).append(int(a))
                        for k in range(k_lo, k_hi):
                            node = int(key_index[k]) - n_lo
                            d_eff = float(np.linalg.norm(y_np[k]))
                            alpha = (float(np.linalg.norm(u_np[k]))
                                     / (d_eff + 1e-8))
                            heavy_degree = sum(
                                1 for nb in neighbours.get(node, [])
                                if int(z_np[n_lo + nb]) > 1)
                            rows.append({
                                "cluster": rid,
                                "alpha": alpha,
                                "d_eff": d_eff,
                                "near_ts": float(near[b]) >= 0.5,
                                "terminal": heavy_degree <= 1,
                                "capped": d_eff >= 0.9 * cap,
                                "d_eff_approx_d0":
                                    abs(d_eff - d0) <= 0.25 * d0,
                            })
            effective = [r for r in rows
                         if r["d_eff"] > float(EFFECTIVE_ATOM_TAU)]
            clusters = sorted({r["cluster"] for r in effective})
            return {
                "checkpoint": str(ckpt_path),
                "n_effective_atoms": len(effective),
                "alpha_median": _finite_median(
                    [r["alpha"] for r in effective]),
                "strata": {
                    "overall_effective": {
                        "n_atoms": len(effective),
                        "alpha_median": _finite_median(
                            [r["alpha"] for r in effective])},
                    "d_eff_approx_d0": {
                        "n_atoms": sum(1 for r in effective
                                       if r["d_eff_approx_d0"]),
                        "alpha_median": _finite_median(
                            [r["alpha"] for r in effective
                             if r["d_eff_approx_d0"]])},
                    "terminal": {
                        "n_atoms": sum(1 for r in effective
                                       if r["terminal"]),
                        "alpha_median": _finite_median(
                            [r["alpha"] for r in effective
                             if r["terminal"]])},
                    "capped": {
                        "n_atoms": sum(1 for r in effective if r["capped"]),
                        "alpha_median": _finite_median(
                            [r["alpha"] for r in effective
                             if r["capped"]])},
                    "near_ts": {
                        "n_atoms": sum(1 for r in effective
                                       if r["near_ts"]),
                        "alpha_median": _finite_median(
                            [r["alpha"] for r in effective
                             if r["near_ts"]])},
                    "far_ts": {
                        "n_atoms": sum(1 for r in effective
                                       if not r["near_ts"]),
                        "alpha_median": _finite_median(
                            [r["alpha"] for r in effective
                             if not r["near_ts"]])},
                    "per_cluster": {
                        cid: _finite_median(
                            [r["alpha"] for r in effective
                             if r["cluster"] == cid])
                        for cid in clusters},
                },
            }

        try:
            best_meas = _measure_checkpoint(best_ckpt)
        except SprintRefused:
            raise
        except Exception as exc:
            raise SprintRefused("ALPHA_DIAG_DATA", str(exc)) from exc
        if final_path.is_file() and final_path != best_ckpt:
            try:
                final_meas = _measure_checkpoint(final_path)
            except Exception as exc:  # endpoint is secondary evidence
                final_meas = {"checkpoint": str(final_path),
                              "error": f"endpoint measurement failed: {exc}"}
        else:
            final_meas = {
                "checkpoint": str(final_path) if final_path.is_file()
                else None,
                "note": ("no distinct endpoint checkpoint — α_final equals "
                         "α_best under the fallback policy"),
                "alpha_median": best_meas["alpha_median"],
            }

        alpha_best_median = best_meas["alpha_median"]
        return {
            "schema_version": "pesn_v51_d1_alpha_v1",
            "status": "measured",
            "arm": arm, "fold": 0,
            "seed": int((self.sprint.get("diagnose") or {}).get("seed", 17)),
            "tag": "d1", "mode": "v51",
            "run_dir": str(run_dir),
            "checkpoint_policy": best_policy,
            "best_checkpoint": str(best_ckpt),
            "alpha_definition": ("alpha = ||u_pred|| / ||y_teacher|| per key "
                                 "atom, restricted to effective atoms "
                                 f"(||y|| > {EFFECTIVE_ATOM_TAU} A)"),
            "strata_definitions": {
                "overall_effective": "all effective key atoms",
                "d_eff_approx_d0": f"||y|| within 25% of d0={d0} A",
                "terminal": "key atom with <=1 heavy neighbour in the "
                            "start-frame bond graph",
                "capped": f"||y|| >= 0.9 * teacher cap ({cap} A)",
                "near_ts": "sample-level near-TS flag >= 0.5",
                "far_ts": "sample-level near-TS flag < 0.5",
                "per_cluster": "reaction-cluster medians over effective atoms",
            },
            "alpha_best": best_meas["strata"],
            "alpha_best_median": alpha_best_median,
            "alpha_final": final_meas,
            "decision_tree_verdict": d1_decision_tree(alpha_best_median),
            "decision_tree_source": (
                "protocol_v51.json d1_decision_tree (frozen pre-training; "
                "consumes alpha_best ONLY)"),
            "measured_at": _utc_now(),
            "post_hoc_only": True,
            "development_evidence_only": True,
        }

    # -- Stage 3: ablate (2×2 × 3 seeds = 12 runs) --------------------------

    def _ablation_row(self, *, arm: str, seed: int, record: Mapping,
                      updates: int, d1_verdict: str | None) -> dict:
        run_dir = Path(record["run_dir"])
        metrics = self._run_metrics(run_dir)
        arm_cfg = dict((self.sprint.get("arm_table") or {}).get(arm) or {})
        magnitude_on = bool(arm_cfg.get("magnitude_head", False))
        confirmatory_only = (magnitude_on
                             and d1_verdict == "checkpoint_problem_only")
        return {
            "arm": arm, "fold": 0, "seed": int(seed),
            "updates": int(updates),
            "d0": float(arm_cfg.get("d0", 0.05)),
            "anchor": _resolve_anchor_mode(arm_cfg.get("anchor", "on")),
            "magnitude_head": magnitude_on,
            "pair_loss": str(arm_cfg.get("pair_loss", "legacy_j")),
            "status": str(record.get("status", "unknown")),
            "best_update": metrics.get("best_update"),
            "inner_val_rmse_over_zero": metrics.get("rmse_over_zero"),
            "inner_val_sre": metrics.get("sre"),
            "inner_val_near_ts_cos": metrics.get("near_ts_cos"),
            "inner_val_sign_accuracy": metrics.get("sign_accuracy"),
            "alpha_median": metrics.get("alpha_median"),
            "dead_zone_fraction": metrics.get("dead_zone_fraction"),
            "wall_time_s": record.get("wall_s"),
            "run_dir": str(run_dir),
            "decision_tree_verdict": d1_verdict,
            "confirmatory_only": confirmatory_only,
            "confirmatory_reason": (
                "D1 verdict checkpoint_problem_only (α_best ≥ 0.85): "
                "magnitude-head arms still run all seeds but are "
                "confirmatory-only per the frozen decision tree"
                if confirmatory_only else None),
            "selection_exposed": True,
            "development_evidence_only": True,
        }

    def _stage_ablate(self, window: StageWindow) -> str:
        self._verify_parents("ablate")
        arms = list(self.sprint.get("arms") or ABLATION_ARMS)
        ablation_plan = dict(self.sprint.get("ablation") or {})
        fold = int(ablation_plan.get("fold", 0))
        seeds = [int(s) for s in (ablation_plan.get("seeds")
                                  or ABLATION_SEEDS)]
        updates = int(ablation_plan.get("updates", self.formal_updates))
        d1 = self._read_d1_diagnostics()
        d1_verdict = d1.get("decision_tree_verdict")
        rows: list[dict] = []
        for arm in arms:
            for seed in seeds:
                try:
                    record = self._spawn_or_skip(
                        arm=arm, fold=fold, seed=seed, updates=updates,
                        mode="v51", stage="ablate", window=window)
                except StageCensored as exc:
                    return self._record_stage(
                        "ablate", "censored_budget", detail=str(exc),
                        extra={"runs_so_far": len(rows)})
                rows.append(self._ablation_row(
                    arm=arm, seed=seed, record=record, updates=updates,
                    d1_verdict=d1_verdict))
        doc = {
            "schema_version": "pesn_v51_ablation_v1",
            "campaign_id": self.campaign_id,
            "design": ("2×2: B0/B1/B2/B3 × seeds {17,29,43} × fold0, "
                       f"{updates} updates, mode v51 (plan §4)"),
            "decision_tree_verdict": d1_verdict,
            "n_runs": len(rows),
            "arms": arms, "seeds": seeds, "fold": fold,
            "label": ("development evidence only — fold0/inner-val, NO "
                      "formal OOF claim (V5.1a)"),
            "rows": rows,
            "measured_at": _utc_now(),
        }
        import pandas as pd

        pd.DataFrame(rows).to_parquet(
            self.out_root / "ablation_v51.parquet", index=False)
        write_json(self.out_root / "ablation_v51.json", doc)
        complete = [r for r in rows
                    if str(r["status"]) in V51_DELIVERED_STATUSES]
        if len(complete) == len(rows) and rows:
            status = "complete_ablate_v51"
            detail = (f"12-run ablation complete "
                      f"(d1 verdict: {d1_verdict})")
        else:
            status = "insufficient_evidence"
            detail = (f"{len(complete)}/{len(rows)} ablation runs completed")
        self._record_stage(
            "ablate", status, detail=detail,
            extra={"n_runs": len(rows),
                   "decision_tree_verdict": d1_verdict,
                   "ablation_table": str(self.out_root / "ablation_v51.parquet")})
        print(f"[ablate] {status}: {detail}")
        return status

    # -- Stage 4: confirm (best config × 5 folds + B0 references) -----------

    def _stage_confirm(self, window: StageWindow) -> str:
        self._verify_parents("confirm")
        ablation_doc = read_json(self.out_root / "ablation_v51.json") or {}
        selection = select_best_arm(list(ablation_doc.get("rows") or []))
        _require(bool(selection.get("best_arm")), "NO_ELIGIBLE_ARM",
                 "frozen selection rule found no eligible arm (metrics "
                 "missing or every arm exceeds the B0×1.05 rmse constraint); "
                 "return to ablate rather than guessing a recipe")
        recipe = str(selection["best_arm"])
        confirm_plan = dict(self.sprint.get("confirm") or {})
        n_folds = int(confirm_plan.get("folds", 5))
        seed = int(confirm_plan.get("seed", CONFIRM_SEED))
        updates = int(confirm_plan.get("updates", self.formal_updates))
        reference_folds = [int(f) for f in
                           (confirm_plan.get("b0_reference_folds")
                            or list(range(1, n_folds)))]

        frozen = self._frozen_recipe(recipe=recipe, selection=selection,
                                     updates=updates, seed=seed,
                                     n_folds=n_folds)
        write_json(self.out_root / "frozen_recipe_v51.json", frozen)
        self.append_decision({
            "stage": "confirm", "decision": "recipe_frozen",
            "arm": recipe, "selection": selection,
            "frozen_recipe": str(self.out_root / "frozen_recipe_v51.json")})

        # B0 reference runs for folds 1–4 (fold0 baseline comes from the
        # ablation table) — marked reference runs.
        references: dict[int, dict] = {}
        for fold in reference_folds:
            try:
                references[fold] = self._spawn_or_skip(
                    arm="B0", fold=fold, seed=seed, updates=updates,
                    mode="v51", stage="confirm", window=window,
                    extra={"reference": True,
                           "reference_role": "B0 baseline for the "
                                             "non-regression check"})
            except StageCensored as exc:
                return self._record_stage("confirm", "censored_budget",
                                          detail=str(exc),
                                          extra={"recipe": recipe})
            self.append_decision({
                "stage": "confirm", "decision": "reference_run",
                "arm": "B0", "fold": fold, "seed": seed, "updates": updates,
                "mode": "v51", "reference": True,
                "reference_role": "B0 baseline for the non-regression check",
                "run_dir": references[fold]["run_dir"]})
        confirm_runs: dict[int, dict] = {}
        for fold in range(n_folds):
            try:
                confirm_runs[fold] = self._spawn_or_skip(
                    arm=recipe, fold=fold, seed=seed, updates=updates,
                    mode="v51", stage="confirm", window=window,
                    extra={"reference": False, "role": "confirm"})
            except StageCensored as exc:
                return self._record_stage("confirm", "censored_budget",
                                          detail=str(exc),
                                          extra={"recipe": recipe})

        # -- non-regression check vs each fold's B0 baseline ----------------
        ablation_rows = list(ablation_doc.get("rows") or [])
        fold0_b0 = next((r for r in ablation_rows
                         if r.get("arm") == "B0"
                         and int(r.get("fold", 0)) == 0
                         and int(r.get("seed", seed)) == seed), None)
        baselines: dict[int, float | None] = {
            0: (fold0_b0 or {}).get("inner_val_rmse_over_zero")}
        for fold, record in references.items():
            baselines[fold] = self._run_metrics(
                Path(record["run_dir"])).get("rmse_over_zero")
        rows: list[dict] = []
        regressed: list[int] = []
        unverified: list[int] = []
        for fold in range(n_folds):
            record = confirm_runs[fold]
            metrics = self._run_metrics(Path(record["run_dir"]))
            baseline = baselines.get(fold)
            rmse = metrics.get("rmse_over_zero")
            if baseline is None or rmse is None:
                is_regressed: bool | None = None
                unverified.append(fold)
            else:
                is_regressed = bool(float(rmse) > float(baseline))
                if is_regressed:
                    regressed.append(fold)
            rows.append({
                "role": "confirm", "fold": fold, "arm": recipe, "seed": seed,
                "updates": updates, "status": str(record.get("status",
                                                             "unknown")),
                "reference": False,
                "inner_val_rmse_over_zero": rmse,
                "b0_baseline_rmse_over_zero": baseline,
                "regressed": is_regressed,
                "best_update": metrics.get("best_update"),
                "inner_val_sre": metrics.get("sre"),
                "inner_val_near_ts_cos": metrics.get("near_ts_cos"),
                "inner_val_sign_accuracy": metrics.get("sign_accuracy"),
                "alpha_median": metrics.get("alpha_median"),
                "dead_zone_fraction": metrics.get("dead_zone_fraction"),
                "wall_time_s": record.get("wall_s"),
                "run_dir": record["run_dir"],
                "selection_exposed": True,
                "development_evidence_only": True,
            })
        for fold in sorted(references):
            record = references[fold]
            metrics = self._run_metrics(Path(record["run_dir"]))
            rows.append({
                "role": "b0_reference", "fold": fold, "arm": "B0",
                "seed": seed, "updates": updates,
                "status": str(record.get("status", "unknown")),
                "reference": True,
                "inner_val_rmse_over_zero": metrics.get("rmse_over_zero"),
                "b0_baseline_rmse_over_zero": None,
                "regressed": None,
                "best_update": metrics.get("best_update"),
                "inner_val_sre": metrics.get("sre"),
                "inner_val_near_ts_cos": metrics.get("near_ts_cos"),
                "inner_val_sign_accuracy": metrics.get("sign_accuracy"),
                "alpha_median": metrics.get("alpha_median"),
                "dead_zone_fraction": metrics.get("dead_zone_fraction"),
                "wall_time_s": record.get("wall_s"),
                "run_dir": record["run_dir"],
                "selection_exposed": True,
                "development_evidence_only": True,
            })
        import pandas as pd

        pd.DataFrame(rows).to_parquet(
            self.out_root / "confirm_v51.parquet", index=False)
        confirm_doc = {
            "schema_version": "pesn_v51_confirm_v1",
            "campaign_id": self.campaign_id,
            "recipe": recipe,
            "selection": selection,
            "n_confirm_runs": n_folds, "n_reference_runs": len(references),
            "regressed_folds": regressed,
            "unverified_folds": unverified,
            "no_fold_regressed": (not regressed and not unverified),
            "label": ("development evidence only — 5-fold confirmation on "
                      "inner-val metrics, NOT a formal OOF claim (V5.1a)"),
            "rows": rows,
            "measured_at": _utc_now(),
        }
        write_json(self.out_root / "confirm_v51.json", confirm_doc)

        # refresh the frozen recipe with the confirmation outcome
        frozen["confirmation"] = {
            "no_fold_regressed": confirm_doc["no_fold_regressed"],
            "regressed_folds": regressed,
            "unverified_folds": unverified,
            "approved_for_v5_1b": confirm_doc["no_fold_regressed"],
            "note": ("plan §4: 5-fold confirmation failing 'no fold "
                     "regresses' → return to the 2×2, do not start the "
                     "15-run OOF"),
        }
        write_json(self.out_root / "frozen_recipe_v51.json", frozen)

        all_complete = all(
            str(r.get("status")) in V51_DELIVERED_STATUSES
            for r in [*confirm_runs.values(), *references.values()])
        if not all_complete or not rows:
            status = "insufficient_evidence"
            detail = "not every confirm/reference run completed"
        elif regressed:
            status = "rejected_gate"
            detail = (f"fold(s) {regressed} regress vs their B0 baseline — "
                      "plan §4: return to the 2×2; recipe NOT approved for "
                      "the 15-run OOF")
        elif unverified:
            status = "insufficient_evidence"
            detail = (f"non-regression unverifiable for fold(s) "
                      f"{unverified} (missing inner-val metrics)")
        else:
            status = "complete_confirm_v51"
            detail = (f"recipe {recipe}: {n_folds} confirm + "
                      f"{len(references)} B0 reference runs; no fold "
                      "regressed vs its B0 baseline")
        self._record_stage(
            "confirm", status, detail=detail,
            extra={"recipe": recipe, "regressed_folds": regressed,
                   "confirm_table": str(self.out_root / "confirm_v51.parquet")})
        print(f"[confirm] {status}: {detail}")
        return status

    def _frozen_recipe(self, *, recipe: str, selection: Mapping,
                       updates: int, seed: int, n_folds: int) -> dict:
        """Frozen-recipe record: arm + all λs + inner-val rule + hashes."""
        arm_cfg = dict((self.sprint.get("arm_table") or {}).get(recipe) or {})
        ramp = dict((self.v51.get("ramp") or {}))
        pair_exact = dict((self.v51.get("pair_exact") or {}))
        return {
            "schema_version": "pesn_v51_frozen_recipe_v1",
            "campaign_id": self.campaign_id,
            "arm": recipe,
            "arm_table_entry": arm_cfg,
            "lambdas": {
                "lambda_cos": float(self.config.get("lambda_cos", 0.02)),
                "lambda_pair": float(self.config.get("lambda_pair", 0.5)),
                "lambda_pair_reg": float(pair_exact.get("lambda_pair_reg",
                                                        0.5)),
                "lambda_mag": float(
                    (self.v51.get("magnitude_head") or {}).get(
                        "lambda_mag", 0.5)),
                "lambda_sign": float(pair_exact.get("lambda_sign", 0.2)),
                "lambda_dead": float(pair_exact.get("lambda_dead", 0.1)),
            },
            "ramp": {"start_update": int(ramp.get("start_update", 250)),
                     "ramp_updates": int(ramp.get("ramp_updates", 250))},
            "inner_val_rule": dict(
                (self._protocol_payload().get("inner_val_rule") or {})),
            "v51_block": {key: dict(self.v51.get(key) or {}) for key in
                          ("inner_val", "magnitude_head", "pair_exact",
                           "ramp")},
            "updates": int(updates), "seed": int(seed),
            "folds": int(n_folds),
            "eval_every_updates": self.eval_every,
            "mode": "v51",
            "hashes": {
                "config_sha256": _sha256_file(self.config_path),
                "split_manifest_sha256": _sha256_file(self._p(
                    "split_manifest",
                    "data/manifests/v5/split_manifest.json")),
                "protocol_sha256": _sha256_file(self.protocol_path),
                "code_commit": self._code_commit(),
            },
            "selection": dict(selection),
            "frozen_at": _utc_now(),
            "selection_exposed": True,
            "thresholds_frozen_pre_training": True,
            "note": (DEVELOPMENT_EVIDENCE_NOTE),
        }

    # -- Stage 5: report -----------------------------------------------------

    def _overall_status51(self, measured: Mapping[str, str]) -> str:
        for status in ("complete_report_v51", "complete_confirm_v51",
                       "complete_ablate_v51", "complete_diagnose_v51",
                       "complete_prepare_v51"):
            if status in measured.values():
                return status
        for status in ("rejected_gate", "rejected_data", "censored_budget",
                       "insufficient_evidence"):
            if status in measured.values():
                return status
        return "insufficient_evidence"

    def _inner_val_curves_summary(self) -> dict:
        """Per-run inner-val curve summary (development evidence)."""
        summary: dict = {}
        if not self.runs_root.is_dir():
            return summary
        for run_dir in sorted(p for p in self.runs_root.iterdir()
                              if p.is_dir()):
            series = self._eval_series(run_dir)
            if not series:
                continue
            points = [(entry.get("update"),
                       (entry.get("aggregate") or {}).get("rmse_over_zero"))
                      for entry in series]
            points = [(u, v) for (u, v) in points
                      if u is not None and v is not None
                      and math.isfinite(float(v))]
            if not points:
                continue
            best_u, best_v = min(points, key=lambda p: float(p[1]))
            summary[run_dir.name] = {
                "n_evals": len(points),
                "eval_every_updates": self.eval_every,
                "best_rmse_over_zero": float(best_v),
                "best_update": int(best_u),
                "final_rmse_over_zero": float(points[-1][1]),
                "final_update": int(points[-1][0]),
            }
        return summary

    def _merge_run_records(self) -> list[dict]:
        """Merge every v51 run into ``outputs/pesn/v5_1/run_records.jsonl``."""
        records: list[dict] = []
        pattern = re.compile(
            r"^(?P<arm>[A-Za-z]+\d+)_f(?P<fold>\d+)_s(?P<seed>\d+)_"
            r"(?P<tag>.+)$")
        config_sha = _sha256_file(self.config_path)
        split_sha = _sha256_file(self._p(
            "split_manifest", "data/manifests/v5/split_manifest.json"))
        protocol_sha = (_sha256_file(self.protocol_path)
                        if self.protocol_path.is_file() else None)
        if self.runs_root.is_dir():
            for run_dir in sorted(p for p in self.runs_root.iterdir()
                                  if p.is_dir()):
                match = pattern.match(run_dir.name)
                if not match:
                    continue
                status = self._run_status(run_dir) or {}
                metrics = self._run_metrics(run_dir)
                best = (run_dir / "checkpoints" / "checkpoint_best.pt")
                final = (run_dir / "checkpoints" / "checkpoint_final.pt")
                ckpt = best if best.is_file() else (
                    final if final.is_file() else None)
                updates_run = status.get("updates_run")
                records.append({
                    "run_id": run_dir.name,
                    "arm": match.group("arm"),
                    "fold": int(match.group("fold")),
                    "seed": int(match.group("seed")),
                    "tag": match.group("tag"),
                    "mode": "v51",
                    "updates": int(updates_run) if updates_run is not None
                    else self.formal_updates,
                    "status": str(status.get("status", "unknown")),
                    "best_update": metrics.get("best_update"),
                    "inner_val_score": {
                        "rmse_over_zero": metrics.get("rmse_over_zero"),
                        "sre": metrics.get("sre"),
                        "near_ts_cos": metrics.get("near_ts_cos"),
                        "sign_accuracy": metrics.get("sign_accuracy"),
                    },
                    "hashes": {
                        "config_sha256": config_sha,
                        "split_manifest_sha256": split_sha,
                        "protocol_sha256": protocol_sha,
                        "checkpoint_sha256": _sha256_file(ckpt)
                        if ckpt is not None else None,
                    },
                    "alpha_median": metrics.get("alpha_median"),
                    "dead_zone_fraction": metrics.get("dead_zone_fraction"),
                    "selection_exposed": True,
                    "development_evidence_only": True,
                })
        self.report_root.mkdir(parents=True, exist_ok=True)
        path = self.report_root / "run_records.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True,
                                    ensure_ascii=False) + "\n")
        return records

    def _write_scale_readiness(self, *, g51_pass: bool,
                               d1: Mapping, confirm_doc: Mapping) -> dict:
        """`scale_readiness_v51.json` — S0 pending; V5.1a cannot pass S0."""
        satisfied: list[str] = ["thresholds frozen pre-training "
                                "(protocol_v51.json)"]
        if g51_pass:
            satisfied.append("G5.1 fixtures green (incl. v51 trainer tests "
                             "when present)")
        if d1.get("status") == "measured":
            satisfied.append(f"D1 checkpoint-effect measured "
                             f"(verdict: {d1.get('decision_tree_verdict')})")
        if confirm_doc.get("no_fold_regressed"):
            satisfied.append("5-fold confirm: no fold regressed vs its B0 "
                             "baseline")
        doc = {
            "schema_version": "pesn_v51_scale_readiness_v1",
            "campaign_id": self.campaign_id,
            "generated_at": _utc_now(),
            "expansion_allowed": False,
            "gates": {
                "S0_model_repair": {
                    "status": "pending",
                    "can_pass_in_v5_1a": False,
                    "criteria": {
                        "g51_green": {
                            "met": bool(g51_pass),
                            "status": "satisfied" if g51_pass else "pending",
                            "evidence": "prepare stage fixture suite"},
                        "g52_original_thresholds_formal_oof": {
                            "met": False, "status": "pending",
                            "note": "formal 49-cluster OOF is V5.1b/V5.2"},
                        "m1_amplitude_0p85_1p05_formal": {
                            "met": False, "status": "pending",
                            "note": "fold0/inner-val alpha is development "
                                    "evidence only (M1 needs formal OOF)"},
                        "g53_original_thresholds_formal_oof": {
                            "met": False, "status": "pending"},
                        "s1_stability_2of3_seed_plus_ensemble": {
                            "met": False, "status": "pending",
                            "note": "V5.1a runs a single seed per cell"},
                        "g54_true_amplitude_formal": {
                            "met": False, "status": "pending"},
                    },
                    "satisfied_now": satisfied,
                    "note": ("V5.1a cannot itself pass S0: S0 requires the "
                             "formal G5.1–G5.4 + M1 + 3-seed evidence from "
                             "V5.1b/V5.2 (plan §7.2)"),
                },
                "S1_data_quality": {
                    "status": "not_started", "met": False,
                    "criteria": {"gauge_unresolved_below_2pct": False,
                                 "conflicts_342_classified": False,
                                 "fit50_label_audit": False,
                                 "no_open_p0": False}},
                "S2_dynamic_microloop": {
                    "status": "not_started", "met": False},
                "S3_fit150_learning_curve": {
                    "status": "not_started", "met": False},
                "S4_confirm_ood_one_shot_unseal": {
                    "status": "not_started", "met": False,
                    "unsealed_in_v5_1a": False,
                    "note": (f"policy {SEALED_POLICY_V51!r} — Confirm/OOD "
                             "stay sealed during V5.1a")},
            },
            "next_step": ("V5.1b/V5.2: frozen recipe × 3 seeds × 5 folds "
                          "formal OOF, then the full gate (plan §7.3)"),
            "development_evidence_only": True,
        }
        path = self.report_root / "scale_readiness_v51.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, doc)
        return doc

    def _artifact(self, name: str, path: Path, *, schema: str | None,
                  source: str, status: str = "measured",
                  note: str | None = None) -> dict:
        return {"name": name, "schema": schema, "path": str(path),
                "config_sha256": _sha256_file(self.config_path),
                "source": source, "status": status, "note": note}

    def _stage_report(self, window: StageWindow) -> str:
        self._verify_parents("report")
        stages = (self.manifest or {}).get("stages") or {}
        measured: dict[str, str] = {
            k: str((v or {}).get("status", "unknown"))
            for k, v in stages.items()}

        # 1) Sealed eval — typed REFUSED, ALWAYS (policy v5_1a_never_unseals).
        sealed = sealed_eval_request(
            {"campaign_id": self.campaign_id,
             "config_sha256": _sha256_file(self.config_path),
             "split_manifest_sha256": _sha256_file(self._p(
                 "split_manifest", "data/manifests/v5/split_manifest.json")),
             "protocol_sha256": (_sha256_file(self.protocol_path)
                                 if self.protocol_path.is_file() else None),
             "stage_statuses": measured,
             "formal_oof": "none — V5.1a does no formal OOF (plan §0.1)",
             "sprint_policy": SEALED_POLICY_V51},
            crossfit_complete_runs=0,
            model_hash=None, config_hash=_sha256_file(self.config_path),
            protocol_hash=(_sha256_file(self.protocol_path)
                           if self.protocol_path.is_file() else None),
            unseal=self.unseal, allow_unseal=False,
            sprint_policy=SEALED_POLICY_V51)
        write_json(self.out_root / "sealed_eval.json", sealed)
        _require(sealed["status"] == "REFUSED", "SEALED_UNEXPECTED",
                 "sealed evaluation must stay REFUSED for V5.1a "
                 f"(policy {SEALED_POLICY_V51!r})")
        print(f"[report] sealed eval: {sealed['status']} "
              f"({sealed['code']}); policy={SEALED_POLICY_V51}")

        # 2) Evidence aggregation (all development-only).
        g51_summary = self._prepare_summary()
        d1 = self._read_d1_diagnostics()
        ablation_doc = {}
        if (self.out_root / "ablation_v51.json").is_file():
            ablation_doc = read_json(self.out_root / "ablation_v51.json") or {}
        confirm_doc = {}
        if (self.out_root / "confirm_v51.json").is_file():
            confirm_doc = read_json(self.out_root / "confirm_v51.json") or {}
        curves = self._inner_val_curves_summary()
        records = self._merge_run_records()
        readiness = self._write_scale_readiness(
            g51_pass=bool(g51_summary.get("g51_pass")),
            d1=d1, confirm_doc=confirm_doc)

        g51_pass = bool(g51_summary.get("g51_pass"))
        deferred_note = ("NO formal OOF in V5.1a — fold0/inner-val numbers "
                         "are development evidence only; formal gate "
                         "deferred to V5.1b/V5.2 (plan §0.1/§5.2)")
        gates: dict[str, dict] = {
            "G5.1": ({
                "status": "measured",
                "verdict": "pass" if g51_pass else "fail",
                "payload": g51_summary.get("g51"),
                "note": "fixture suite run at prepare (incl. "
                        "test_key_atom_trainer_v51.py when present)"}
                if g51_summary else {
                    "status": "not_executed", "verdict": "—",
                    "note": "prepare summary missing"}),
            "G5.2": {
                "status": "not_executed", "verdict": "deferred_to_v5_1b",
                "payload": None, "note": deferred_note},
            "G5.3": {
                "status": "not_executed", "verdict": "deferred_to_v5_1b",
                "payload": None, "note": deferred_note},
            "G5.4": {
                "status": "not_executed", "verdict": "deferred_to_v5_1b",
                "payload": None,
                "note": "true-amplitude static re-test needs the formal "
                        "V5.1b OOF: " + deferred_note},
            "oracle_static": {
                "status": "not_executed", "verdict": "—",
                "payload": None,
                "note": "V5 oracle ceiling is historical evidence; V5.1 "
                        "re-measurement deferred to V5.1b"},
            "sealed_eval": {
                "status": "measured", "verdict": sealed["status"],
                "payload": sealed,
                "note": f"typed REFUSED under policy {SEALED_POLICY_V51!r} "
                        "(never unseals, allow_unseal=False)"},
            "M1": ({
                "status": "measured",
                "verdict": str(d1.get("decision_tree_verdict") or "—"),
                "payload": {
                    "alpha_best_median": d1.get("alpha_best_median"),
                    "alpha_best_strata": d1.get("alpha_best"),
                    "decision_tree": d1.get("decision_tree_verdict"),
                    "ablation_alpha_medians": {
                        str(row.get("arm")): row.get("alpha_median")
                        for row in ablation_doc.get("rows", [])},
                    "checkpoint_policy": d1.get("checkpoint_policy"),
                },
                "note": "post-hoc α diagnostics on outer dev — "
                        "development evidence only, never a checkpoint "
                        "selection input"}
                if d1 else {
                    "status": "not_executed", "verdict": "—",
                    "note": "D1 diagnostics missing (run --stage diagnose)"}),
            "inner_val": ({
                "status": "measured", "verdict": "development_evidence",
                "payload": curves,
                "note": "inner-val curves @ eval_every="
                        f"{self.eval_every} — {DEVELOPMENT_EVIDENCE_NOTE}"}
                if curves else {
                    "status": "not_executed", "verdict": "—",
                    "note": "no v51 runs with eval curves yet"}),
            "ablation": ({
                "status": "measured", "verdict": "development_evidence",
                "payload": {"n_runs": ablation_doc.get("n_runs"),
                            "design": ablation_doc.get("design"),
                            "decision_tree_verdict":
                                ablation_doc.get("decision_tree_verdict"),
                            "path": str(self.out_root / "ablation_v51.parquet")},
                "note": DEVELOPMENT_EVIDENCE_NOTE}
                if ablation_doc else {
                    "status": "not_executed", "verdict": "—",
                    "note": "ablation table missing (run --stage ablate)"}),
            "confirm": ({
                "status": "measured",
                "verdict": ("no_fold_regressed"
                            if confirm_doc.get("no_fold_regressed")
                            else "regressed_or_incomplete"),
                "payload": {"recipe": confirm_doc.get("recipe"),
                            "regressed_folds":
                                confirm_doc.get("regressed_folds"),
                            "no_fold_regressed":
                                confirm_doc.get("no_fold_regressed"),
                            "path": str(self.out_root / "confirm_v51.parquet")},
                "note": DEVELOPMENT_EVIDENCE_NOTE}
                if confirm_doc else {
                    "status": "not_executed", "verdict": "—",
                    "note": "confirm table missing (run --stage confirm)"}),
        }

        artifacts = [
            self._artifact("protocol_v51", self.protocol_path,
                           schema=V51_PROTOCOL_SCHEMA,
                           source="stage prepare (frozen pre-training)"),
            self._artifact("timing_probe", self.out_root / "timing_probe.json",
                           schema="pesn_v51_timing_probe_v1",
                           source="stage prepare"),
            self._artifact("prepare_summary_v51",
                           self.out_root / "prepare_summary_v51.json",
                           schema="pesn_v51_prepare_summary_v1",
                           source="stage prepare"),
            self._artifact("d1_best_checkpoint_alpha",
                           self.diagnostics_root /
                           "d1_best_checkpoint_alpha.json",
                           schema="pesn_v51_d1_alpha_v1",
                           source="stage diagnose"),
            self._artifact("ablation_v51_parquet",
                           self.out_root / "ablation_v51.parquet",
                           schema="pesn_v51_ablation_v1",
                           source="stage ablate"),
            self._artifact("ablation_v51_json",
                           self.out_root / "ablation_v51.json",
                           schema="pesn_v51_ablation_v1",
                           source="stage ablate"),
            self._artifact("frozen_recipe_v51",
                           self.out_root / "frozen_recipe_v51.json",
                           schema="pesn_v51_frozen_recipe_v1",
                           source="stage confirm"),
            self._artifact("confirm_v51_parquet",
                           self.out_root / "confirm_v51.parquet",
                           schema="pesn_v51_confirm_v1",
                           source="stage confirm"),
            self._artifact("confirm_v51_json",
                           self.out_root / "confirm_v51.json",
                           schema="pesn_v51_confirm_v1",
                           source="stage confirm"),
            self._artifact("sealed_eval", self.out_root / "sealed_eval.json",
                           schema="pesn_v5_sealed_eval_v1",
                           source="stage report"),
            self._artifact("run_records", self.report_root /
                           "run_records.jsonl",
                           schema="pesn_v51_run_records_v1",
                           source="stage report (merger of all v51 runs)"),
            self._artifact("scale_readiness_v51",
                           self.report_root / "scale_readiness_v51.json",
                           schema="pesn_v51_scale_readiness_v1",
                           source="stage report"),
        ]

        report_paths = self._write_gate_report(
            gates=gates, artifacts=artifacts, measured=measured)
        self._append_development_evidence_md(
            d1=d1, ablation_doc=ablation_doc, confirm_doc=confirm_doc,
            curves=curves, records=records, readiness=readiness,
            selection=(confirm_doc.get("selection") or {}))
        overall = self._overall_status51(measured)
        self.append_decision({
            "stage": "report", "decision": "report_written",
            "campaign_status": overall,
            "sealed_status": sealed["status"],
            "oof_claims": "none (development evidence only)",
            "artifacts": {k: v["path"] for k, v in report_paths.items()}})
        self._record_stage(
            "report", "complete_report_v51",
            detail=(f"gate_report_v51 written; sealed eval "
                    f"{sealed['status']}; no formal OOF claims "
                    "(development evidence only)"),
            extra={"sealed_status": sealed["status"],
                   "campaign_status": overall})
        print(f"[report] complete: {report_paths['gate_report_json']['path']} "
              f"(sealed={sealed['status']}, campaign={overall})")
        return "complete_report_v51"

    def _write_gate_report(self, *, gates: Mapping[str, Mapping],
                           artifacts: Sequence[Mapping],
                           measured: Mapping[str, str]) -> dict:
        """Write ``gate_report_v51.json`` + ``.md`` via the shared writer.

        ``write_gate_report`` renders fixed names into a staging dir; the
        files are then renamed to the V5.1 deliverable names (the V5
        ``gate_report.*`` names belong to the V5 campaign).
        """
        staging = self.report_root / ".gate_report_v51_staging"
        paths = write_gate_report(
            staging, gates=gates, artifacts=artifacts,
            stage_statuses=dict(measured), campaign_id=self.campaign_id,
            qc="QC not scheduled",
            config_sha256=_sha256_file(self.config_path),
            extra={"development_evidence": DEVELOPMENT_EVIDENCE_NOTE,
                   "oof_claims": "none — the campaign status vocabulary "
                                 "contains no formal-OOF completion status",
                   "protocol_path": str(self.protocol_path),
                   "sealed_policy": SEALED_POLICY_V51},
            repo_root=PROJECT_ROOT)
        self.report_root.mkdir(parents=True, exist_ok=True)
        json_dst = self.report_root / "gate_report_v51.json"
        md_dst = self.report_root / "gate_report_v51.md"
        os.replace(staging / "gate_report.json", json_dst)
        os.replace(staging / "gate_report.md", md_dst)
        try:
            staging.rmdir()
        except OSError:
            pass
        return {"gate_report_json": {"path": str(json_dst)},
                "gate_report_md": {"path": str(md_dst)}}

    def _append_development_evidence_md(self, *, d1: Mapping,
                                        ablation_doc: Mapping,
                                        confirm_doc: Mapping,
                                        curves: Mapping, records: Sequence,
                                        readiness: Mapping,
                                        selection: Mapping) -> None:
        """Append the V5.1a evidence sections to ``gate_report_v51.md``."""
        md_path = self.report_root / "gate_report_v51.md"
        if not md_path.is_file():
            return
        lines = [
            "",
            "## 4. V5.1a 开发证据（development evidence only）",
            "",
            "> **fold0 / inner-val 全部为 development evidence**：V5.1a 不做"
            "正式 OOF，不产生任何 formal G5.2/G5.3 结论；正式 3-seed OOF 门"
            "属于 V5.1b/V5.2（plan §0.1）。",
            f"> thresholds frozen pre-training, no post-hoc relaxation "
            f"（`{self.protocol_path}`）。",
            f"> sealed eval policy `{SEALED_POLICY_V51}` — never unseals。",
            "",
            "### 4.1 M1 / D1 幅度诊断（post-hoc）",
            "",
            f"- checkpoint policy：`{d1.get('checkpoint_policy') or '—'}`",
            f"- α_best（effective atoms median）："
            f"`{d1.get('alpha_best_median')}`",
            f"- decision-tree verdict：`{d1.get('decision_tree_verdict') or '—'}`",
            "",
            "### 4.2 消融 / 确认表",
            "",
            f"- ablation：`{ablation_doc.get('n_runs', 0)}` runs "
            f"（{ablation_doc.get('design') or 'not executed'}）",
            f"- confirm recipe：`{confirm_doc.get('recipe') or '—'}`；"
            f"no_fold_regressed=`{confirm_doc.get('no_fold_regressed')}`；"
            f"regressed={confirm_doc.get('regressed_folds')}",
            "",
            "### 4.3 inner-val 曲线摘要（eval_every=50）",
            "",
        ]
        if curves:
            for run_id, entry in sorted(curves.items()):
                lines.append(f"- `{run_id}`: n_evals={entry['n_evals']}, "
                             f"best={entry['best_rmse_over_zero']:.4f} "
                             f"@u{entry['best_update']}, "
                             f"final={entry['final_rmse_over_zero']:.4f}")
        else:
            lines.append("- （无 v51 run 曲线）")
        lines += [
            "",
            "### 4.4 run records / scale readiness",
            "",
            f"- run_records.jsonl：{len(records)} runs"
            "（arm/fold/seed/updates/best_update/inner_val_score/hashes/"
            "alpha_median/dead_zone_fraction/selection_exposed）",
            f"- scale readiness：S0="
            f"`{(readiness.get('gates') or {}).get('S0_model_repair', {}).get('status')}`，"
            f"expansion_allowed=`{readiness.get('expansion_allowed')}`"
            "（V5.1a 无法自证 S0 —— 正式 OOF 在 V5.1b）",
            "",
        ]
        md_path.write_text(md_path.read_text(encoding="utf-8")
                           + "\n".join(lines), encoding="utf-8")

    # -- dry run -------------------------------------------------------------

    def _dry_run_stage(self, stage: str) -> str:
        """List the planned actions; freeze nothing; write NOTHING."""
        print(f"[dry-run] stage={stage}: planned actions follow "
              f"(T0 NOT frozen, no files written)")
        plans = {
            "prepare": (
                "G5.1 pytest evidence (4 fixture files + "
                f"{G51_V51_TEST_FILE} when present; REFUSED G51_NOT_GREEN "
                "if v51 trainer code present without tests); re-run timing "
                f"probe with eval interval {self.eval_every} "
                "(t_run_seconds eval_every=50) → timing_probe.json; FREEZE "
                "protocol_v51.json (thresholds imported from "
                "key_atom_protocols constants) BEFORE any training"),
            "diagnose": (
                "spawn D1 = B0 (legacy losses, magnitude head off, "
                "pair_loss legacy_j) fold0 seed17 3000 updates --mode v51 → "
                "runs/B0_f0_s17_d1/; post-hoc α_best/α_final strata on outer "
                "dev → diagnostics/d1_best_checkpoint_alpha.json; frozen D1 "
                "decision tree"),
            "ablate": (
                "2×2 matrix B0/B1/B2/B3 × seeds {17,29,43} × fold0 = 12 "
                "runs (--mode v51, 3000 updates, sequential GPU, "
                "skip-if-complete); ablation_v51.parquet/.json with "
                "confirmatory-only marks for magnitude-head arms when D1 "
                "says checkpoint problem only"),
            "confirm": (
                "frozen selection rule (fold0 3-seed median inner-val SRE, "
                "≤ B0 median ×1.05, tie |α−1| then sign) → freeze "
                "frozen_recipe_v51.json; 5 folds × seed17 confirm runs + "
                "B0 reference runs for folds 1–4; no fold may regress vs "
                "its B0-baseline fold rmse"),
            "report": (
                "gate_report_v51.md/json (G5.1 evidence, M1 amplitude "
                "diagnostics, inner-val curves, ablation + confirm tables — "
                "ALL development evidence only, no formal G5.2/G5.3 OOF "
                "claims); run_records.jsonl merger; scale_readiness_v51.json "
                "(S0 pending); sealed_eval_request(allow_unseal=False, "
                f"policy {SEALED_POLICY_V51}) → REFUSED"),
        }
        print(f"  - {plans[stage]}")
        return "dry_run"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/campaigns/run_v5_1_sprint.py",
        description="V5.1a repair-sprint campaign driver "
                    "(no formal OOF, never unseals)")
    parser.add_argument("--config",
                        default="config/campaigns/v5_1_sprint.yaml")
    parser.add_argument("--stage", required=True, choices=V51_STAGE_ORDER)
    parser.add_argument("--dry-run", action="store_true",
                        help="list planned actions; freeze nothing; write "
                             "nothing")
    parser.add_argument("--unseal", action="store_true",
                        help="recorded request only — the sealed confirm "
                             "set stays REFUSED all campaign "
                             "(policy v5_1a_never_unseals)")
    parser.add_argument("--force", action="store_true",
                        help="re-run completed work; downgrade parent-hash "
                             "mismatch to a recorded override; re-freeze the "
                             "protocol (explicit decision recorded)")
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
            "output_root", "outputs/pesn/v5_1/sprint"))
    driver = V51SprintDriver(
        config_path=config_path, out_root=out_root,
        unseal=args.unseal, force=args.force, dry_run=args.dry_run,
        device=args.device)
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
