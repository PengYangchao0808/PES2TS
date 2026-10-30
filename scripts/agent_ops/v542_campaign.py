#!/usr/bin/env python3
"""V5.4.2 WP4 campaign technical-measurement harness (P3 lane C; plan §7).

One CLI, three TECHNICAL-MEASUREMENT subcommands — smoke / profile /
diagnostic are measurements, never scientific results and never formal
runs (plan §7: "smoke/profile/diagnostic 可在 T0 前先行，属技术测量"):

* ``profile`` — same-environment measurement: per backbone, ``w0`` update
  steps in W0 then ``w1`` in W1 with a ``warmup``-update warm-up on the
  FROZEN data view (config ``data_contract``/``split_contract``).  The
  report carries, per candidate: ``core_seconds`` / ``wall_seconds`` over
  the measured (post-warm-up) updates, ``peak_vram_mb`` (CUDA only),
  ``host_rss_mb``, ``edges_per_second`` (edges actually processed per
  measured core-second — a real per-edge throughput proxy, never a V541
  archived cost number), device, torch/CUDA versions and the exact
  config/source identity digests.  The report is APPEND-only JSONL and
  this harness NEVER writes ``budget.per_candidate_core_time`` — the
  owner signs the budget after review (typed refusal
  ``V542_CAMPAIGN_BUDGET_WRITE_FORBIDDEN`` guards the seam).  A
  diagnostic variant (A1-C/A5-C) in the candidate set is refused unless
  ``--conditional`` is given (typed
  ``V542_CAMPAIGN_CONDITIONAL_REQUIRED``); conditional variants are
  recorded in their OWN report file.

* ``smoke`` — 200 updates, seed 17, single lane, A0–A5 (the main smoke
  matrix), with ONE kill-and-resume cycle proving crash recovery (a
  bounded first leg to ``kill_at``, then a resumed leg to the full dose
  — the V541 ``--stop-after`` drill pattern, identity-checked resume via
  the frozen ``checkpoint_last.pt`` machinery).  ``rank_excluded=true``:
  a smoke outcome can NEVER be reported as a scientific result (typed
  refusal ``V542_CAMPAIGN_SCIENCE_LEAK``).  A1-C/A5-C only via
  ``--conditional`` and recorded separately from the main matrix.

* ``diagnostic`` — 2,000 updates (400 W0 + 1,600 W1) for A0–A5 with the
  A0@2000 same-dose control (``V542_DIAG_A0_S17``; A0 ALWAYS runs first
  so the control exists before any candidate is judged).  ONLY the
  frozen ENGINEERING screens of config ``diagnostic.elimination_rules``
  are applied (plan §7):

  1. non-finite loss / gradient;
  2. best inner-val total-loss improvement < 5% vs the initial
     diagnostic evaluation (``thresholds.diagnostic_loss_drop_min_pct``);
  3. dead fraction > A0@2000 + 0.20 (``diagnostic_dead_fraction_margin_vs_a0``);
  4. RMSE/zero degradation > 10% vs A0@2000 (``diagnostic_rmse_zero_rel_max_pct``).

  Everything else advances to full dose.  The harness does NOT apply
  H3's 3% effect thresholds (typed refusal
  ``V542_DIAG_H3_GATES_FORBIDDEN``) and does NOT rank candidates —
  decisions are per-candidate ``advance_full_dose`` / ``eliminated``
  rows with explicit reason codes.

Common discipline (asserted from the frozen config on every run):

* single lane (``single_lane=true``), no swap (``swap_max=0``), confirm
  NEVER read (``confirm_authorized=false``), ``new_qc_calls=0`` — typed
  refusals ``V542_CAMPAIGN_GUARD`` on any drift;
* APPEND-only JSONL/JSON outputs under the campaign root, every row
  carrying the run identity block (+ digest), the guard block and an
  explicit ``reason_code``;
* ``--dry-run`` prints the planned steps WITHOUT training (no writes);
* ``--device cpu`` is fully supported (tests exercise real short runs);
* ``--formal`` DELEGATES to the frozen config validator
  (``v542_config_schema``) and stays typed-blocked while
  ``budget.per_candidate_core_time`` is null (``V542_BUDGET_NULL``) —
  this harness never launches a formal run either way;
* no ``torch.compile`` anywhere (the assembled model asserts it too).

Frozen trainer math is REUSED, never forked: runs go through
``V542Trainer`` / ``train_run_v542`` (profile additionally replays the
SAME per-update sampler/loss/optimizer sequence the trainer loop uses —
imported calls, never re-implemented math).

CLI::

    python scripts/agent_ops/v542_campaign.py profile [--dry-run] [--device cpu]
    python scripts/agent_ops/v542_campaign.py smoke   [--kill-at N] ...
    python scripts/agent_ops/v542_campaign.py diagnostic [--evaluator-metrics DIR]
                                               [--allow-unevaluated-grad-screen] ...
    # --formal on ANY subcommand: typed-blocked by the validator (V542_BUDGET_NULL)
    # --seed must be in {17,29,43} (V542_CAMPAIGN_SEED_FROZEN); --device cuda
    # requires CUDA availability (V542_CAMPAIGN_DEVICE_UNAVAILABLE) — typed
    # refusals, never tracebacks (oracle gate-3 C5).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # PyYAML: existing repo dependency

# ---------------------------------------------------------------------------
# frozen constants (mirror the campaign config; nothing is invented here)
# ---------------------------------------------------------------------------
#: Campaign-ops record schema id.
CAMPAIGN_SCHEMA: str = "pes2ts_v542_campaign_harness_v1"

DEFAULT_CONFIG = PROJECT_ROOT / (
    "config/campaigns/pes2ts_v542_backbone_50k.yaml")
DEFAULT_RUNPLAN = PROJECT_ROOT / (
    "config/campaigns/v542_templates/runplan_pin.template.json")
DEFAULT_BUDGET = PROJECT_ROOT / (
    "config/campaigns/v542_templates/budget_policy.template.json")
#: config ``plan_outputs.root``.
DEFAULT_CAMPAIGN_ROOT = PROJECT_ROOT / "outputs/pesn/v542_backbone_50k"

#: The main measurement matrix (config ``*.backbones``).
MAIN_BACKBONES: tuple = ("A0", "A1", "A2", "A3", "A4", "A5")
#: Full-dose diagnostic variants — conditional activation only
#: (config ``*.conditional_backbones``).
DIAGNOSTIC_VARIANTS: tuple = ("A1-C", "A5-C")
ALL_BACKBONES: tuple = MAIN_BACKBONES + DIAGNOSTIC_VARIANTS

#: Technical-measurement seeds (config ``smoke.smoke_seed`` /
#: ``diagnostic.seed``; profile is same-environment at the anchor seed).
SMOKE_SEED: int = 17
PROFILE_SEED: int = 17
DIAGNOSTIC_SEED: int = 17

#: Frozen profile window (config ``profile.window`` / ``warmup_updates``).
PROFILE_W0_DEFAULT: int = 40
PROFILE_W1_DEFAULT: int = 40
PROFILE_WARMUP_DEFAULT: int = 20
#: Frozen smoke dose (config ``smoke.updates``).
SMOKE_UPDATES_DEFAULT: int = 200
#: Frozen diagnostic dose (config ``diagnostic.updates`` / ``w0_updates`` /
#: ``w1_updates``).
DIAGNOSTIC_UPDATES_DEFAULT: int = 2000
DIAGNOSTIC_W0_DEFAULT: int = 400
DIAGNOSTIC_W1_DEFAULT: int = 1600

#: Frozen engineering-screen thresholds (config ``thresholds.diagnostic_*``,
#: signed-frozen 2026-10-29) — the fallbacks mirror the config exactly.
DIAGNOSTIC_LOSS_DROP_MIN_PCT: float = 5.0
DIAGNOSTIC_DEAD_FRACTION_MARGIN_VS_A0: float = 0.20
DIAGNOSTIC_RMSE_ZERO_REL_MAX_PCT: float = 10.0

#: H3 effect-gate keys (config ``thresholds.h3_*``) that must NEVER gate a
#: 2,000-update diagnostic (plan §7: "不得套用 H3 的 3% 门").
H3_EFFECT_THRESHOLD_KEYS: tuple = (
    "h3_move_direction_rel_min_pct",
    "h3_move_direction_abs_min",
    "h3_move_rmse_rel_min_pct",
    "h3_process_delta_auprc_abs_min",
    "h3_process_delta_q_corr_abs_min",
)

#: Reason codes — every gate/refusal decision carries one explicitly.
REASON_PROFILE_OK: str = "profile_ok"
REASON_SMOKE_OK: str = "smoke_kill_resume_proven"
REASON_DIAG_ALL_PASS: str = "diag_all_engineering_screens_pass_advance_full_dose"
REASON_DIAG_NON_FINITE: str = "diag_non_finite_loss_or_gradient"
REASON_DIAG_LOSS_DROP: str = "diag_best_inner_val_drop_lt_5pct_vs_initial"
REASON_DIAG_DEAD_FRACTION: str = "diag_dead_fraction_gt_a0_at_2000_plus_0_20"
REASON_DIAG_RMSE_ZERO: str = "diag_rmse_zero_rel_degradation_gt_10pct_vs_a0_at_2000"
REASON_DIAG_NOT_EVALUATED_NOTE: str = "screen_not_evaluated_metrics_absent"
#: The gradient half of the non-finite screen is unevaluated when the
#: trainer run records carry NO ``grad_finite_all_finite`` field (older
#: records / a trainer without the finite-gradient check) — an explicit
#: reason code, never a silent pass (oracle gate-3 C3).
REASON_DIAG_GRAD_UNAVAILABLE: str = (
    "grad_screen_unavailable_trainer_records_lack_field")

#: The trainer-side gradient-finiteness record fields (the parallel
#: trainer lane writes ``grad_finite_all_finite: bool`` — plus
#: ``grad_nonfinite_params: [...]`` when false — to every trainer run
#: record; the diagnostic screen CONSUMES them and never depends on the
#: optional evaluator metrics for the gradient half).
TRAINER_GRAD_FINITE_FIELD: str = "grad_finite_all_finite"
TRAINER_GRAD_NONFINITE_PARAMS_FIELD: str = "grad_nonfinite_params"

#: The frozen technical-measurement seeds (config
#: ``init_policy.process_head_anchor.seeds`` / ``smoke.smoke_seed`` /
#: ``diagnostic.seed``).
FROZEN_RUN_SEEDS: tuple = (17, 29, 43)

#: Per-screen stable ids (frozen order — the decision reason is the FIRST
#: failing screen in this order).
SCREEN_NON_FINITE: str = "non_finite_loss_or_gradient"
SCREEN_LOSS_DROP: str = "best_inner_val_total_loss_drop"
SCREEN_DEAD_FRACTION: str = "dead_fraction_vs_a0_at_2000"
SCREEN_RMSE_ZERO: str = "rmse_zero_vs_a0_at_2000"
SCREEN_ORDER: tuple = (
    SCREEN_NON_FINITE, SCREEN_LOSS_DROP, SCREEN_DEAD_FRACTION, SCREEN_RMSE_ZERO)

#: Run-id namespace (config ``plan_outputs.run_id_namespace``).
RUN_ID_NAMESPACE: str = "V542_"

#: Config knobs a caller may override — run-length / wall-clock ONLY.
#: Anything else is a typed refusal: the harness can never be used to
#: smuggle a forbidden workaround (config ``budget.forbidden_workarounds``)
#: past the frozen trainer config.
_TEST_ONLY_OVERRIDES: frozenset = frozenset({
    "val_every", "batch_curves", "frames_per_curve",
    "inner_val_max_samples", "feature_cache_capacity", "grad_monitor_every",
})


class V542CampaignError(RuntimeError):
    """Typed refusal of the campaign harness (``code`` + ``detail``)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}" if detail else str(code))
        self.code = str(code)
        self.detail = str(detail)


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise V542CampaignError(code, detail)


# ---------------------------------------------------------------------------
# config + operational guards (single lane / no swap / confirm / QC)
# ---------------------------------------------------------------------------
def load_campaign_config(path: Optional[Any] = None) -> Dict[str, Any]:
    """Load the frozen V5.4.2 campaign config YAML (read-only)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    _require(cfg_path.is_file(), "V542_CAMPAIGN_CONFIG_MISSING",
             f"campaign config not found: {cfg_path}")
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - typed conversion
        raise V542CampaignError(
            "V542_CAMPAIGN_CONFIG_INVALID", f"config unreadable: {exc}") from exc
    _require(isinstance(cfg, Mapping), "V542_CAMPAIGN_CONFIG_INVALID",
             "config is not a mapping")
    return dict(cfg)


def assert_operational_guards(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail-closed operational guards, asserted on EVERY subcommand.

    The frozen authorization/budget blocks (config ``governance.
    authorization`` / ``budget``) MUST keep: single lane, no swap, no
    OOM kills, confirm+sealed never authorized, zero new QC calls.  Any
    drift is a typed ``V542_CAMPAIGN_GUARD`` refusal BEFORE anything is
    trained or written.  Returns the guard record every output row
    carries verbatim.
    """
    auth = (cfg.get("governance") or {}).get("authorization") or {}
    budget = cfg.get("budget") or {}
    _require(auth.get("single_lane") is True, "V542_CAMPAIGN_GUARD",
             "governance.authorization.single_lane must be true (single-lane "
             f"campaign; got {auth.get('single_lane')!r})")
    _require(auth.get("swap_allowed") is False, "V542_CAMPAIGN_GUARD",
             "governance.authorization.swap_allowed must be false (swap=0; "
             f"got {auth.get('swap_allowed')!r})")
    _require(auth.get("confirm_authorized") is False, "V542_CAMPAIGN_GUARD",
             "governance.authorization.confirm_authorized must be false — "
             "confirm is sealed forever (got "
             f"{auth.get('confirm_authorized')!r})")
    _require(auth.get("sealed_authorized") is False, "V542_CAMPAIGN_GUARD",
             "governance.authorization.sealed_authorized must be false (got "
             f"{auth.get('sealed_authorized')!r})")
    _require(int(auth.get("new_qc_calls", 0) or 0) == 0,
             "V542_CAMPAIGN_GUARD",
             "governance.authorization.new_qc_calls must be 0 (got "
             f"{auth.get('new_qc_calls')!r})")
    _require(budget.get("single_lane") is True, "V542_CAMPAIGN_GUARD",
             "budget.single_lane must be true (got "
             f"{budget.get('single_lane')!r})")
    _require(int(budget.get("swap_max", -1)) == 0, "V542_CAMPAIGN_GUARD",
             f"budget.swap_max must be 0 (got {budget.get('swap_max')!r})")
    _require(int(budget.get("oom_kill_max", -1)) == 0, "V542_CAMPAIGN_GUARD",
             "budget.oom_kill_max must be 0 (got "
             f"{budget.get('oom_kill_max')!r})")
    return {
        "single_lane": True,
        "swap_max": 0,
        "oom_kill_max": 0,
        "confirm_read": False,
        "sealed_authorized": False,
        "new_qc_calls": 0,
    }


def resolve_candidates(requested: Optional[Sequence[str]], *,
                       conditional: bool) -> List[str]:
    """Validate the requested candidate set; refuse un-flagged variants.

    A diagnostic variant (A1-C/A5-C) may only be measured with the
    explicit conditional flag (config ``profile.conditional_backbones`` /
    ``smoke.conditional_backbones``): typed
    ``V542_CAMPAIGN_CONDITIONAL_REQUIRED`` otherwise.
    """
    cands = [str(c) for c in (requested or MAIN_BACKBONES)]
    _require(bool(cands), "V542_CAMPAIGN_CANDIDATES",
             "candidate set is empty")
    unknown = [c for c in cands if c not in ALL_BACKBONES]
    _require(not unknown, "V542_CAMPAIGN_CANDIDATE_UNKNOWN",
             f"unknown candidate id(s) {unknown}; known: {list(ALL_BACKBONES)}")
    variants = [c for c in cands if c in DIAGNOSTIC_VARIANTS]
    _require(not variants or conditional,
             "V542_CAMPAIGN_CONDITIONAL_REQUIRED",
             "diagnostic variant(s) {} require the explicit conditional flag "
             "(A1-C/A5-C are full-dose diagnostics: conditional smoke + "
             "profile + cost gate + budget record before activation; config "
             "*.conditional_backbones)".format(variants))
    return cands


def assert_technical_result(row: Mapping[str, Any]) -> None:
    """Typed refusal if a technical measurement would be reported as a
    scientific result (``rank_excluded`` MUST be true and
    ``scientific_result`` MUST be false on every smoke/profile/
    diagnostic record — plan §7: 技术测量不进排名、不是科学结论)."""
    _require(row.get("rank_excluded") is True,
             "V542_CAMPAIGN_SCIENCE_LEAK",
             "a technical measurement must carry rank_excluded=true (got "
             f"{row.get('rank_excluded')!r}) — smoke/profile/diagnostic "
             "outcomes are never ranked and never scientific results")
    _require(row.get("scientific_result") is False,
             "V542_CAMPAIGN_SCIENCE_LEAK",
             "a technical measurement must carry scientific_result=false "
             f"(got {row.get('scientific_result')!r})")


# ---------------------------------------------------------------------------
# identity (run identity block attached to EVERY output row)
# ---------------------------------------------------------------------------
def build_identity_record(config_path: Any, candidate_id: str, run_seed: int, *,
                          root: Path = PROJECT_ROOT,
                          runplan_path: Any = DEFAULT_RUNPLAN,
                          budget_path: Any = DEFAULT_BUDGET) -> Dict[str, Any]:
    """The frozen 14-field run identity for one (candidate, seed).

    Delegates to ``pes2ts_core.pesn.models.v542.identity.
    assemble_run_identity`` with the WP0 templates (runplan pin + the
    PRE-PROFILE budget record whose null budget values are the frozen,
    signed-over state — this harness never rewrites them).
    """
    from pes2ts_core.pesn.models.v542.identity import assemble_run_identity

    cfg = load_campaign_config(config_path)
    for path in (Path(runplan_path), Path(budget_path)):
        _require(path.is_file(), "V542_CAMPAIGN_IDENTITY_INPUT",
                 f"identity input record not found: {path}")
    runplan_record = json.loads(
        Path(runplan_path).read_text(encoding="utf-8"))
    budget_block = json.loads(Path(budget_path).read_text(encoding="utf-8"))
    return assemble_run_identity(
        cfg, candidate_id=str(candidate_id), run_seed=int(run_seed),
        runplan_record=runplan_record, budget_block=budget_block,
        root=root, config_path=Path(config_path))


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    # gate-3 A1: ONE canonical recipe (strict; non-JSON values are a typed
    # refusal instead of silent str() coercion).
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256
    return canonical_sha256(dict(identity))


def _identity_for(identity_by_candidate: Optional[Mapping[str, Any]],
                  candidate_id: str) -> Dict[str, Any]:
    block = (identity_by_candidate or {}).get(candidate_id)
    _require(isinstance(block, Mapping) and bool(block),
             "V542_CAMPAIGN_IDENTITY_MISSING",
             f"no run-identity block provided for candidate {candidate_id!r}")
    return dict(block)


# ---------------------------------------------------------------------------
# append-only outputs
# ---------------------------------------------------------------------------
def append_jsonl(path: Any, row: Mapping[str, Any]) -> None:
    """APPEND one JSON row (never rewrite; the file is an audit log)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def read_jsonl(path: Any) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# frozen trainer-config construction (config-driven; typed overrides)
# ---------------------------------------------------------------------------
def _trainer_config(cfg_mapping: Mapping[str, Any], *,
                    candidate_id: str, seed: int, freeze: int, joint: int,
                    smoke_updates: Optional[int], device: str,
                    overrides: Optional[Mapping[str, Any]] = None
                    ) -> Any:
    """One planned :class:`V542TrainerConfig` from the frozen config.

    λ / lrs / backbone_init / objective / eval_scope come from the frozen
    trainer config defaults (= the config's signed values); schedule,
    batch, selection and split knobs are read from the campaign config so
    the trainer config identity stays bound to the config file.  Test-only
    overrides are restricted to run-length / wall-clock knobs — anything
    else is a typed ``V542_CAMPAIGN_OVERRIDE_FORBIDDEN`` refusal (the
    harness can never become a forbidden-workaround vector).
    """
    from pes2ts_core.pesn.offline.training.v542_train import V542TrainerConfig

    tc = cfg_mapping.get("training_contract") or {}
    schedule = dict(tc.get("schedule") or {})
    batch = dict(tc.get("batch") or {})
    lambdas = dict(tc.get("lambdas") or {})
    selection = dict(tc.get("selection") or {})
    split = cfg_mapping.get("split_contract") or {}
    base: Dict[str, Any] = dict(
        seed=int(seed),
        candidate_id=str(candidate_id),
        freeze_encoder_updates=int(freeze),
        joint_updates=int(joint),
        smoke_updates=(None if smoke_updates is None else int(smoke_updates)),
        lambda_rank=float(lambdas.get("lambda_rank", 0.5)),
        lambda_q=float(lambdas.get("lambda_q", 0.5)),
        lambda_endpoint=float(lambdas.get(
            "lambda_endpoint", 0.012385049119773344)),
        weight_decay=float(schedule.get("weight_decay", 1.0e-4)),
        grad_clip=float(schedule.get("grad_clip", 1.0)),
        val_every=int(selection.get("val_every", 100)),
        batch_curves=int(batch.get("curves", 8)),
        frames_per_curve=int(batch.get("frames", 4)),
        inner_val_max_samples=(
            int(split["inner_val_max_samples"])
            if split.get("inner_val_max_samples") else None),
        device=str(device),
    )
    for key, value in (overrides or {}).items():
        _require(key in _TEST_ONLY_OVERRIDES,
                 "V542_CAMPAIGN_OVERRIDE_FORBIDDEN",
                 f"config override {key!r} is not a run-length/wall-clock "
                 "knob; allowed: {}".format(sorted(_TEST_ONLY_OVERRIDES)))
        base[key] = value
    return V542TrainerConfig(**base)


def _schedule_windows(cfg_mapping: Mapping[str, Any]) -> Dict[str, int]:
    schedule = (cfg_mapping.get("training_contract") or {}).get("schedule") or {}
    return {
        "w0_updates": int(schedule.get("w0_updates", 400)),
        "w1_updates": int(schedule.get("w1_updates", 37500)),
    }


def _common_row(*, kind: str, run_id: str, candidate_id: str, seed: int,
                device: str, guards: Mapping[str, Any],
                identity: Mapping[str, Any], config: Any) -> Dict[str, Any]:
    """The identity/guard preamble every campaign row carries."""
    from pes2ts_core.pesn.offline.training.v542_train import (
        config_full_sha256, config_identity_sha256)

    return {
        "schema": CAMPAIGN_SCHEMA,
        "kind": str(kind),
        "run_id": str(run_id),
        "run_id_namespace": RUN_ID_NAMESPACE,
        "objective": "unified_move_process",
        "candidate_id": str(candidate_id),
        "seed": int(seed),
        "device": str(device),
        "ts": _now(),
        "guards": dict(guards),
        "single_lane": True,
        "confirm_read": False,
        "new_qc_calls": 0,
        "torch_compile": False,
        "rank_excluded": True,
        "scientific_result": False,
        "identity": dict(identity),
        "identity_sha256": _identity_sha256(identity),
        "config_identity_sha256": config_identity_sha256(config),
        "config_full_sha256": config_full_sha256(config),
    }


def _environment_record(device: str) -> Dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_name = None
    if str(device).startswith("cuda") and cuda_available:
        try:
            device_name = torch.cuda.get_device_name(device)
        except RuntimeError:  # pragma: no cover - driver edge
            device_name = "<unavailable>"
    return {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device_name": device_name,
    }


# ---------------------------------------------------------------------------
# PROFILE — same-environment 40 W0 + 40 W1 measurement (20 warm-up)
# ---------------------------------------------------------------------------
def _profile_update_rows(run_dir: Any, total: int) -> List[Dict[str, Any]]:
    """Per-update core/wall deltas from the trainer's append-only log.

    ``train_log.jsonl`` rows carry CUMULATIVE ``core_s``/``wall_s``; the
    per-update cost is the consecutive difference (the first update is
    differenced against 0 — the core timer starts at 0 in ``_prepare``).
    """
    rows = [r for r in read_jsonl(Path(run_dir) / "train_log.jsonl")
            if r.get("event") == "train"]
    by_update = {int(r["update"]): r for r in rows}
    out: List[Dict[str, Any]] = []
    prev_core, prev_wall = 0.0, 0.0
    for update in range(1, int(total) + 1):
        row = by_update.get(update)
        _require(row is not None, "V542_CAMPAIGN_PROFILE_LOG",
                 f"train_log.jsonl has no row for update {update}")
        core = float(row["core_s"])
        wall = float(row["wall_s"])
        out.append({
            "update": update,
            "phase": str(row.get("phase")),
            "core_s": core - prev_core,
            "wall_s": wall - prev_wall,
            "batch_positions": [int(p) for p in row.get("batch_positions", [])],
        })
        prev_core, prev_wall = core, wall
    return out


def _count_edges(lru: Any, positions: Sequence[int]) -> int:
    """Edges actually processed for ``positions`` — counted from the SAME
    built feature dicts the training loop consumed (the LRU holds pure
    functions of the records; a miss rebuilds the identical dict)."""
    import numpy as np

    total = 0
    for position in positions:
        built = lru.get(int(position))
        total += int(np.asarray(built["edge_index"]).shape[1])
    return total


def profile_candidate(env: Any, candidate_id: str, *,
                      config: Mapping[str, Any], run_dir: Any,
                      identity: Mapping[str, Any], device: str = "cpu",
                      seed: int = PROFILE_SEED,
                      w0: int = PROFILE_W0_DEFAULT,
                      w1: int = PROFILE_W1_DEFAULT,
                      warmup: int = PROFILE_WARMUP_DEFAULT,
                      overrides: Optional[Mapping[str, Any]] = None
                      ) -> Dict[str, Any]:
    """Measure ONE backbone: ``w0`` W0 updates + ``w1`` W1 updates with a
    ``warmup``-update warm-up, on the frozen data view.

    The updates run through the FROZEN trainer core (:class:
    `V542Trainer` deterministic prepare + the exact per-update sampler/
    loss/backward/clip/AdamW sequence of ``V542Trainer._train_loop``,
    imported — never re-implemented).  Inner-val selection is not part of
    a profile run: the ONLY eval is the trainer's mandatory final sweep.
    """
    import torch

    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _batch_from_positions,
        _loss_on_batch,
        _update_rng,
    )
    from pes2ts_core.pesn.offline.training.v541_monitor import (
        host_peak_rss_bytes,
    )
    from pes2ts_core.pesn.offline.training.v541_strata import (
        batch_rng_v541,
        curve_batch_positions_v541,
    )
    from pes2ts_core.pesn.offline.training.v542_train import (
        V542Trainer,
        set_training_stage_v542,
    )

    w0, w1, warmup = int(w0), int(w1), int(warmup)
    total = w0 + w1
    _require(total >= 1, "V542_CAMPAIGN_PROFILE_WINDOW",
             "profile window must contain at least one update")
    _require(0 <= warmup < total, "V542_CAMPAIGN_PROFILE_WINDOW",
             f"warmup must satisfy 0 <= warmup < total ({total}); got {warmup}")

    cfg = _trainer_config(config, candidate_id=candidate_id, seed=seed,
                          freeze=w0, joint=w1, smoke_updates=None,
                          device=device, overrides=overrides)
    trainer = V542Trainer(
        config=cfg, out_dir=run_dir, dataset=env.ds, labels_v52=env.labels,
        endpoint_labels=env.endpoint, dev_positions=env.dev_pos,
        identity=identity, require_identity=True, resume=False)
    metrics = trainer.run()

    per_update = _profile_update_rows(run_dir, total)
    measured = [row for row in per_update if row["update"] > warmup]
    st = trainer.st
    _require(st is not None, "V542_CAMPAIGN_PROFILE_STATE",
             "trainer prepared no state (cannot measure)")

    edges = 0
    for row in measured:
        edges += _count_edges(st.lru, row["batch_positions"])

    def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        return round(sum(float(r[key]) for r in rows), 6)

    core_seconds = _sum(measured, "core_s")
    wall_seconds = _sum(measured, "wall_s")
    w0_measured = [r for r in measured if r["update"] <= w0]
    w1_measured = [r for r in measured if r["update"] > w0]

    peak_vram_mb = None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        peak_vram_mb = round(
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 3)
    host_rss_mb = round(host_peak_rss_bytes() / (1024.0 ** 2), 3)
    edges_per_second = (round(edges / core_seconds, 3)
                        if core_seconds > 0 else None)

    row = _common_row(
        kind="profile",
        run_id="{}PROFILE_{}_S{}".format(
            RUN_ID_NAMESPACE, str(candidate_id).replace("-", ""), int(seed)),
        candidate_id=candidate_id, seed=seed, device=device,
        guards=assert_operational_guards(config), identity=identity,
        config=cfg)
    row.update(_environment_record(device))
    row.update({
        "window": {
            "w0_updates": w0,
            "w1_updates": w1,
            "warmup_updates": warmup,
            "total_updates": total,
            "measured_updates": [r["update"] for r in measured],
            "w0_measured_updates": len(w0_measured),
            "w1_measured_updates": len(w1_measured),
        },
        "core_seconds": core_seconds,
        "wall_seconds": wall_seconds,
        "w0_core_seconds": _sum(w0_measured, "core_s"),
        "w0_wall_seconds": _sum(w0_measured, "wall_s"),
        "w1_core_seconds": _sum(w1_measured, "core_s"),
        "w1_wall_seconds": _sum(w1_measured, "wall_s"),
        "peak_vram_mb": peak_vram_mb,
        "host_rss_mb": host_rss_mb,
        "edges_measured": int(edges),
        "edges_per_second": edges_per_second,
        "edges_per_second_note": (
            "edges actually processed in the measured window / measured "
            "core-seconds (real per-edge throughput proxy; NEVER a V541 "
            "archived cost number — config profile.forbidden_denominator)"),
        "loop_wall_s": metrics.get("timing", {}).get("loop_wall_s"),
        "run_outcome": metrics.get("run_outcome"),
        "is_formal_schedule": bool(metrics.get("is_formal_schedule")),
        "run_dir": str(run_dir),
        # budget seam: this harness NEVER writes the budget table —
        # the owner signs budget.per_candidate_core_time after review.
        "budget_written": False,
        "budget_note": (
            "measurement only: budget.per_candidate_core_time is signed by "
            "the owner after review; this harness never writes it"),
        "reason_code": REASON_PROFILE_OK,
    })
    _require("per_candidate_core_time" not in row,
             "V542_CAMPAIGN_BUDGET_WRITE_FORBIDDEN",
             "profile rows must never carry budget.per_candidate_core_time")
    assert_technical_result(row)
    return row


def run_profile(env: Any, *,
                config: Mapping[str, Any],
                campaign_root: Any,
                candidates: Optional[Sequence[str]] = None,
                conditional: bool = False,
                identity_by_candidate: Optional[Mapping[str, Any]] = None,
                device: str = "cpu",
                seed: int = PROFILE_SEED,
                w0: int = PROFILE_W0_DEFAULT,
                w1: int = PROFILE_W1_DEFAULT,
                warmup: int = PROFILE_WARMUP_DEFAULT,
                overrides: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Profile every requested backbone; APPEND each row to the report."""
    guards = assert_operational_guards(config)
    resolved = resolve_candidates(candidates, conditional=conditional)
    root = Path(campaign_root)
    report_path = root / "profile" / "profile_report.jsonl"
    rows: List[Dict[str, Any]] = []
    for candidate_id in resolved:
        identity = _identity_for(identity_by_candidate, candidate_id)
        run_dir = root / "profile" / "{}PROFILE_{}_S{}".format(
            RUN_ID_NAMESPACE, candidate_id.replace("-", ""), int(seed))
        row = profile_candidate(
            env, candidate_id, config=config, run_dir=run_dir,
            identity=identity, device=device, seed=seed, w0=w0, w1=w1,
            warmup=warmup, overrides=overrides)
        row["guards"] = guards
        append_jsonl(report_path, row)
        rows.append(row)
    return rows


def plan_profile(*, config: Mapping[str, Any], campaign_root: Any,
                 candidates: Optional[Sequence[str]] = None,
                 conditional: bool = False, device: str = "cpu",
                 seed: int = PROFILE_SEED, w0: int = PROFILE_W0_DEFAULT,
                 w1: int = PROFILE_W1_DEFAULT,
                 warmup: int = PROFILE_WARMUP_DEFAULT) -> Dict[str, Any]:
    """The planned profile steps (--dry-run; no training, no writes)."""
    guards = assert_operational_guards(config)
    resolved = resolve_candidates(candidates, conditional=conditional)
    root = Path(campaign_root)
    return {
        "kind": "profile",
        "dry_run": True,
        "training_performed": False,
        "device": str(device),
        "seed": int(seed),
        "window": {"w0_updates": int(w0), "w1_updates": int(w1),
                   "warmup_updates": int(warmup),
                   "total_updates": int(w0) + int(w1)},
        "steps": [{
            "step": index,
            "run_id": "{}PROFILE_{}_S{}".format(
                RUN_ID_NAMESPACE, c.replace("-", ""), int(seed)),
            "candidate_id": c,
            "conditional": c in DIAGNOSTIC_VARIANTS,
            "run_dir": str(root / "profile" / "{}PROFILE_{}_S{}".format(
                RUN_ID_NAMESPACE, c.replace("-", ""), int(seed))),
        } for index, c in enumerate(resolved, start=1)],
        "report_path": str(root / "profile" / "profile_report.jsonl"),
        "guards": guards,
        "budget_written": False,
    }


# ---------------------------------------------------------------------------
# SMOKE — 200 updates, seed 17, single lane, kill-resume once, rank_excluded
# ---------------------------------------------------------------------------
def smoke_candidate(env: Any, candidate_id: str, *,
                    config: Mapping[str, Any], run_dir: Any,
                    identity: Mapping[str, Any], device: str = "cpu",
                    seed: int = SMOKE_SEED,
                    updates: int = SMOKE_UPDATES_DEFAULT,
                    kill_at: Optional[int] = None,
                    overrides: Optional[Mapping[str, Any]] = None
                    ) -> Dict[str, Any]:
    """ONE smoke run with a single kill-and-resume cycle.

    Leg 1 runs a BOUNDED dose (``kill_at`` updates — the V541
    ``--stop-after`` crash drill: the leg stops with ``checkpoint_last.pt``
    on disk, exactly as a killed process would leave it); leg 2 resumes
    from that checkpoint (identity-checked by the frozen trainer) and
    completes the full smoke dose.  ``smoke_updates`` is a run-length
    knob excluded from the resume identity, so the two legs share ONE
    identity digest by construction.
    """
    from pes2ts_core.pesn.offline.training.v542_train import train_run_v542

    updates = int(updates)
    kill_at = int(kill_at) if kill_at is not None else max(1, updates // 2)
    _require(1 <= kill_at < updates, "V542_CAMPAIGN_KILL_AT",
             f"kill_at must satisfy 1 <= kill_at < updates ({updates}); "
             f"got {kill_at}")

    windows = _schedule_windows(config)
    common = dict(freeze=windows["w0_updates"], joint=windows["w1_updates"],
                  device=device, overrides=overrides)

    cfg_killed = _trainer_config(
        config, candidate_id=candidate_id, seed=seed,
        smoke_updates=kill_at, **common)
    leg1 = train_run_v542(
        env.ds, env.labels, env.endpoint, cfg_killed, run_dir,
        dev_positions=env.dev_pos, identity=identity, require_identity=True,
        resume=False)

    cfg_full = _trainer_config(
        config, candidate_id=candidate_id, seed=seed,
        smoke_updates=updates, **common)
    leg2 = train_run_v542(
        env.ds, env.labels, env.endpoint, cfg_full, run_dir,
        dev_positions=env.dev_pos, identity=identity, require_identity=True,
        resume=True)

    _require(leg1.get("identity_sha256") == leg2.get("identity_sha256"),
             "V542_CAMPAIGN_KILL_RESUME",
             "kill/resume legs disagree on the run-identity digest — the "
             "resume was not an identity-checked continuation")
    manifest = read_jsonl(Path(run_dir) / "manifest.jsonl")
    resume_events = [r for r in manifest if r.get("event") == "resume"]
    _require(len(resume_events) >= 1, "V542_CAMPAIGN_KILL_RESUME",
             "the resumed leg left no resume event in manifest.jsonl — "
             "crash recovery was not proven")
    return {
        "kill_at": kill_at,
        "updates": updates,
        "leg1": {
            "dose": kill_at,
            "run_outcome": leg1.get("run_outcome"),
            "updates_run": leg1.get("updates_run"),
            "selected_update": leg1.get("selected_update"),
        },
        "leg2": {
            "dose": updates,
            "run_outcome": leg2.get("run_outcome"),
            "updates_run": leg2.get("updates_run"),
            "selected_update": leg2.get("selected_update"),
            "resumed": True,
        },
        "resume_events": len(resume_events),
        "identity_sha256": leg2.get("identity_sha256"),
        "manifest_path": str(Path(run_dir) / "manifest.jsonl"),
        "checkpoint_last_path": leg2.get("checkpoint_last_path"),
    }


def run_smoke(env: Any, *,
              config: Mapping[str, Any],
              campaign_root: Any,
              candidates: Optional[Sequence[str]] = None,
              conditional: bool = False,
              identity_by_candidate: Optional[Mapping[str, Any]] = None,
              device: str = "cpu",
              seed: int = SMOKE_SEED,
              updates: int = SMOKE_UPDATES_DEFAULT,
              kill_at: Optional[int] = None,
              overrides: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Smoke every requested backbone; main matrix and conditional
    variants are recorded in SEPARATE append-only reports."""
    guards = assert_operational_guards(config)
    resolved = resolve_candidates(candidates, conditional=conditional)
    root = Path(campaign_root) / "smoke"
    rows: List[Dict[str, Any]] = []
    for candidate_id in resolved:
        identity = _identity_for(identity_by_candidate, candidate_id)
        run_id = "{}SMOKE_{}_S{}".format(
            RUN_ID_NAMESPACE, candidate_id.replace("-", ""), int(seed))
        run_dir = root / run_id
        evidence = smoke_candidate(
            env, candidate_id, config=config, run_dir=run_dir,
            identity=identity, device=device, seed=seed, updates=updates,
            kill_at=kill_at, overrides=overrides)
        row = _common_row(
            kind="smoke", run_id=run_id, candidate_id=candidate_id,
            seed=seed, device=device, guards=guards, identity=identity,
            config=_trainer_config(
                config, candidate_id=candidate_id, seed=seed,
                freeze=_schedule_windows(config)["w0_updates"],
                joint=_schedule_windows(config)["w1_updates"],
                smoke_updates=updates, device=device, overrides=overrides))
        row.update(_environment_record(device))
        row.update({
            "matrix": ("conditional" if candidate_id in DIAGNOSTIC_VARIANTS
                       else "main"),
            "dose": int(updates),
            "kill_resume": evidence,
            "run_outcome": evidence["leg2"]["run_outcome"],
            "run_dir": str(run_dir),
            "reason_code": REASON_SMOKE_OK,
        })
        assert_technical_result(row)
        report = root / (
            "smoke_conditional_report.jsonl"
            if row["matrix"] == "conditional" else "smoke_report.jsonl")
        append_jsonl(report, row)
        rows.append(row)
    return rows


def plan_smoke(*, config: Mapping[str, Any], campaign_root: Any,
               candidates: Optional[Sequence[str]] = None,
               conditional: bool = False, device: str = "cpu",
               seed: int = SMOKE_SEED, updates: int = SMOKE_UPDATES_DEFAULT,
               kill_at: Optional[int] = None) -> Dict[str, Any]:
    """The planned smoke steps (--dry-run; no training, no writes)."""
    guards = assert_operational_guards(config)
    resolved = resolve_candidates(candidates, conditional=conditional)
    resolved_kill = int(kill_at) if kill_at is not None else max(1, int(updates) // 2)
    _require(1 <= resolved_kill < int(updates), "V542_CAMPAIGN_KILL_AT",
             f"kill_at must satisfy 1 <= kill_at < updates ({updates}); "
             f"got {resolved_kill}")
    root = Path(campaign_root) / "smoke"
    return {
        "kind": "smoke",
        "dry_run": True,
        "training_performed": False,
        "device": str(device),
        "seed": int(seed),
        "dose": int(updates),
        "kill_at": resolved_kill,
        "steps": [{
            "step": index,
            "run_id": "{}SMOKE_{}_S{}".format(
                RUN_ID_NAMESPACE, c.replace("-", ""), int(seed)),
            "candidate_id": c,
            "matrix": ("conditional" if c in DIAGNOSTIC_VARIANTS else "main"),
            "legs": [{"leg": 1, "dose": resolved_kill, "resume": False},
                     {"leg": 2, "dose": int(updates), "resume": True}],
            "run_dir": str(root / "{}SMOKE_{}_S{}".format(
                RUN_ID_NAMESPACE, c.replace("-", ""), int(seed))),
        } for index, c in enumerate(resolved, start=1)],
        "report_paths": {
            "main": str(root / "smoke_report.jsonl"),
            "conditional": str(root / "smoke_conditional_report.jsonl"),
        },
        "rank_excluded": True,
        "guards": guards,
    }


# ---------------------------------------------------------------------------
# DIAGNOSTIC — 2,000 = 400 + 1,600, A0@2000 control, engineering screens ONLY
# ---------------------------------------------------------------------------
def diagnostic_thresholds(cfg_mapping: Mapping[str, Any]) -> Dict[str, float]:
    """The THREE frozen engineering-screen thresholds (config
    ``thresholds.diagnostic_*``) — and NOTHING else.  H3 effect keys are
    structurally impossible to smuggle in: the payload is an allowlist."""
    th = dict(cfg_mapping.get("thresholds") or {})
    return {
        "loss_drop_min_pct": float(th.get(
            "diagnostic_loss_drop_min_pct", DIAGNOSTIC_LOSS_DROP_MIN_PCT)),
        "dead_fraction_margin_vs_a0": float(th.get(
            "diagnostic_dead_fraction_margin_vs_a0",
            DIAGNOSTIC_DEAD_FRACTION_MARGIN_VS_A0)),
        "rmse_zero_rel_max_pct": float(th.get(
            "diagnostic_rmse_zero_rel_max_pct",
            DIAGNOSTIC_RMSE_ZERO_REL_MAX_PCT)),
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def apply_diagnostic_screens(
        metrics: Mapping[str, Any],
        control_metrics: Optional[Mapping[str, Any]],
        thresholds: Optional[Mapping[str, Any]] = None, *,
        h3_effect_thresholds: Optional[Mapping[str, Any]] = None,
        allow_unevaluated_grad_screen: bool = False
) -> Dict[str, Any]:
    """Apply ONLY the frozen engineering screens (plan §7 / config
    ``diagnostic.elimination_rules``) to ONE candidate's diagnostic
    metrics against the A0@2000 same-dose control.

    Screens (in frozen order; the decision reason is the FIRST failure):

    1. ``non_finite_loss_or_gradient`` — any non-finite train loss or
       gradient.  The gradient half is read from the TRAINER run-record
       field ``grad_finite_all_finite`` (primary; the optional evaluator
       ``grad_finite`` remains a legacy fallback).  ``False`` =
       eliminated (frozen reason code); a records set WITHOUT the field
       (older trainer records) marks the screen ``evaluated=false`` /
       ``gradient_screen_unavailable`` — it can never eliminate, and it
       also REFUSES to advance the candidate to full dose
       (typed ``V542_DIAG_GRAD_SCREEN_UNEVALUATED``) unless the caller
       passes ``allow_unevaluated_grad_screen=True`` (the explicit
       ``--allow-unevaluated-grad-screen`` override, recorded on the
       decision row and refused in formal mode);
    2. ``best_inner_val_total_loss_drop`` — best inner-val total loss
       improved < ``loss_drop_min_pct``% over the INITIAL diagnostic
       evaluation ⇒ eliminated (not learning);
    3. ``dead_fraction_vs_a0_at_2000`` — dead fraction > control +
       ``dead_fraction_margin_vs_a0`` (absolute);
    4. ``rmse_zero_vs_a0_at_2000`` — RMSE/zero relative degradation vs
       the control > ``rmse_zero_rel_max_pct``%.

    Everything else ADVANCES to full dose.  The function does not rank
    (no scores, no orderings) and REFUSES H3-style effect gating: any
    ``h3_effect_thresholds`` payload is a typed
    ``V542_DIAG_H3_GATES_FORBIDDEN`` refusal (the 2,000-update diagnostic
    must never be gated by H3's 3% effect sizes).
    """
    _require(h3_effect_thresholds is None, "V542_DIAG_H3_GATES_FORBIDDEN",
             "H3 effect thresholds (3% route gates) must never gate the "
             "2,000-update diagnostic (plan §7: 不得套用 H3 的 3% 门); got "
             f"{dict(h3_effect_thresholds or {})}")
    if thresholds:
        unknown = sorted(set(thresholds) - {
            "loss_drop_min_pct", "dead_fraction_margin_vs_a0",
            "rmse_zero_rel_max_pct"})
        _require(not unknown, "V542_DIAG_H3_GATE_FORBIDDEN_THRESHOLD",
                 "only the three frozen engineering-screen thresholds are "
                 f"allowed; refusing H3/effect-gate key(s): {unknown}")
    th = {
        "loss_drop_min_pct": DIAGNOSTIC_LOSS_DROP_MIN_PCT,
        "dead_fraction_margin_vs_a0": DIAGNOSTIC_DEAD_FRACTION_MARGIN_VS_A0,
        "rmse_zero_rel_max_pct": DIAGNOSTIC_RMSE_ZERO_REL_MAX_PCT,
    }
    th.update({k: float(v) for k, v in (thresholds or {}).items()})

    screens: List[Dict[str, Any]] = []

    def _screen(name: str, *, passed: bool, reason: str,
                evaluated: bool = True, **evidence: Any) -> None:
        entry: Dict[str, Any] = {
            "screen": name,
            "status": ("pass" if passed else "fail") if evaluated
            else REASON_DIAG_NOT_EVALUATED_NOTE,
            "evaluated": bool(evaluated),
            "passed": bool(passed),
            "reason_code": reason,
        }
        entry.update(evidence)
        screens.append(entry)

    # -- 1. non-finite loss / gradient -------------------------------------
    # The gradient half CONSUMES the trainer run-record flag
    # ``grad_finite_all_finite`` (primary; surfaced by
    # ``diagnostic_metrics_from_run``), with the optional evaluator
    # ``grad_finite`` as a legacy fallback — never evaluator-only.
    loss_finite = metrics.get("loss_finite")
    grad_flag = metrics.get(TRAINER_GRAD_FINITE_FIELD)
    grad_source = TRAINER_GRAD_FINITE_FIELD
    if grad_flag is None:
        grad_flag = metrics.get("grad_finite")
        grad_source = ("evaluator_grad_finite_fallback"
                       if grad_flag is not None else None)
    grad_nonfinite_params = metrics.get(TRAINER_GRAD_NONFINITE_PARAMS_FIELD)
    loss_bad = loss_finite is False
    grad_bad = grad_flag is False
    if loss_bad or grad_bad:
        _screen(SCREEN_NON_FINITE, passed=False,
                reason=REASON_DIAG_NON_FINITE,
                loss_finite=loss_finite,
                grad_finite_all_finite=(
                    grad_flag if grad_source == TRAINER_GRAD_FINITE_FIELD
                    else None),
                grad_finite=metrics.get("grad_finite"),
                grad_finite_source=grad_source,
                grad_nonfinite_params=grad_nonfinite_params)
    elif grad_flag is None or loss_finite is None:
        # unevaluated: the trainer records lack the gradient-finiteness
        # field (and/or the loss half is unmeasured) — cannot eliminate,
        # but the decision row is explicitly flagged and advancing is
        # refused unless the caller opted in.
        _screen(SCREEN_NON_FINITE, passed=True, evaluated=False,
                reason=(REASON_DIAG_GRAD_UNAVAILABLE
                        if grad_flag is None
                        else REASON_DIAG_NOT_EVALUATED_NOTE),
                loss_finite=loss_finite,
                grad_finite_all_finite=None,
                grad_finite=metrics.get("grad_finite"),
                grad_finite_source=grad_source,
                grad_nonfinite_params=grad_nonfinite_params,
                note=("gradient finiteness is not present in the trainer "
                      "run records (older records lack "
                      f"{TRAINER_GRAD_FINITE_FIELD!r}); the screen is "
                      "recorded as not-evaluated with an explicit reason "
                      "code, can never eliminate, and REFUSES to advance "
                      "the candidate to full dose unless "
                      "--allow-unevaluated-grad-screen is given"))
    else:
        _screen(SCREEN_NON_FINITE, passed=True,
                reason="diag_loss_and_gradient_finite",
                loss_finite=loss_finite,
                grad_finite_all_finite=(
                    grad_flag if grad_source == TRAINER_GRAD_FINITE_FIELD
                    else None),
                grad_finite=metrics.get("grad_finite"),
                grad_finite_source=grad_source,
                grad_nonfinite_params=grad_nonfinite_params)

    # -- 2. best inner-val total-loss drop vs the initial diagnostic eval --
    initial = metrics.get("initial_inner_val_total_loss")
    best = metrics.get("best_inner_val_total_loss")
    _require(_finite(initial) and _finite(best), "V542_DIAG_METRICS_INVALID",
             "initial/best inner-val total loss must be finite numbers, got "
             f"initial={initial!r} best={best!r}")
    initial = float(initial)
    best = float(best)
    _require(initial > 0.0, "V542_DIAG_METRICS_INVALID",
             f"initial diagnostic evaluation must be > 0, got {initial!r}")
    drop_pct = (initial - best) / initial * 100.0
    drop_ok = drop_pct >= float(th["loss_drop_min_pct"])
    _screen(SCREEN_LOSS_DROP, passed=drop_ok,
            reason=(REASON_DIAG_LOSS_DROP if not drop_ok
                    else "diag_loss_drop_ge_5pct"),
            initial_inner_val_total_loss=initial,
            best_inner_val_total_loss=best,
            loss_drop_pct=round(drop_pct, 6),
            loss_drop_min_pct=float(th["loss_drop_min_pct"]))

    control = dict(control_metrics or {})

    # -- 3. dead fraction vs A0@2000 ---------------------------------------
    dead = metrics.get("dead_fraction")
    control_dead = control.get("dead_fraction")
    if dead is None or control_dead is None:
        _screen(SCREEN_DEAD_FRACTION, passed=True, evaluated=False,
                reason=REASON_DIAG_NOT_EVALUATED_NOTE,
                dead_fraction=dead, control_dead_fraction=control_dead,
                note=("absent evaluator metrics cannot eliminate; the "
                      "screen is recorded as not-evaluated (reason code) "
                      "and the candidate advances under the frozen "
                      "'everything else advances' rule"))
    else:
        margin = float(dead) - float(control_dead)
        dead_ok = margin <= float(th["dead_fraction_margin_vs_a0"])
        _screen(SCREEN_DEAD_FRACTION, passed=dead_ok,
                reason=(REASON_DIAG_DEAD_FRACTION if not dead_ok
                        else "diag_dead_fraction_within_a0_margin"),
                dead_fraction=float(dead),
                control_dead_fraction=float(control_dead),
                dead_fraction_margin=round(margin, 6),
                dead_fraction_margin_max=float(
                    th["dead_fraction_margin_vs_a0"]))

    # -- 4. RMSE/zero relative degradation vs A0@2000 ----------------------
    rmse = metrics.get("rmse_zero")
    control_rmse = control.get("rmse_zero")
    if rmse is None or control_rmse is None:
        _screen(SCREEN_RMSE_ZERO, passed=True, evaluated=False,
                reason=REASON_DIAG_NOT_EVALUATED_NOTE,
                rmse_zero=rmse, control_rmse_zero=control_rmse,
                note="absent evaluator metrics cannot eliminate")
    else:
        control_rmse = float(control_rmse)
        _require(control_rmse != 0.0, "V542_DIAG_CONTROL_INVALID",
                 "A0@2000 control rmse_zero must be non-zero to form a "
                 f"relative degradation (got {control_rmse!r})")
        rel = (float(rmse) - control_rmse) / control_rmse * 100.0
        rel_ok = rel <= float(th["rmse_zero_rel_max_pct"])
        _screen(SCREEN_RMSE_ZERO, passed=rel_ok,
                reason=(REASON_DIAG_RMSE_ZERO if not rel_ok
                        else "diag_rmse_zero_within_10pct_of_a0"),
                rmse_zero=float(rmse), control_rmse_zero=control_rmse,
                rmse_zero_rel_degradation_pct=round(rel, 6),
                rmse_zero_rel_max_pct=float(th["rmse_zero_rel_max_pct"]))

    failures = [s for s in screens if s["evaluated"] and not s["passed"]]
    decision = "eliminated" if failures else "advance_full_dose"
    reason = failures[0]["reason_code"] if failures else REASON_DIAG_ALL_PASS
    grad_screen = next(s for s in screens if s["screen"] == SCREEN_NON_FINITE)
    gradient_screen_unavailable = not grad_screen["evaluated"]
    override_applied = False
    if (decision == "advance_full_dose" and gradient_screen_unavailable
            and not allow_unevaluated_grad_screen):
        raise V542CampaignError(
            "V542_DIAG_GRAD_SCREEN_UNEVALUATED",
            "the non-finite gradient screen is NOT evaluated (the trainer "
            f"run records lack {TRAINER_GRAD_FINITE_FIELD!r}); refusing to "
            "advance this candidate to full dose on an unevaluated "
            "gradient screen — pass --allow-unevaluated-grad-screen to "
            "advance with the override recorded (refused in formal mode)")
    if (decision == "advance_full_dose" and gradient_screen_unavailable):
        override_applied = True
    return {
        "decision": decision,
        "reason_code": reason,
        "failing_screens": [s["screen"] for s in failures],
        "screens": screens,
        "thresholds": dict(th),
        "h3_effect_thresholds_applied": False,
        "h3_gates_forbidden": list(H3_EFFECT_THRESHOLD_KEYS),
        "rank_excluded": True,
        "ranking_applied": False,
        "elimination_only": True,
        "advance_default": "everything that does not trigger a frozen "
                           "engineering screen advances to full dose",
        "gradient_screen_unavailable": gradient_screen_unavailable,
        "unevaluated_grad_screen_override": override_applied,
    }


def _grad_finite_from_run_records(
        run_dir: Any) -> Tuple[Optional[bool], Optional[List[str]]]:
    """Read the trainer-recorded gradient-finiteness flag from the
    append-only run artifacts (``run_records.jsonl`` / ``train_log.jsonl``).

    The parallel trainer lane writes ``grad_finite_all_finite: bool``
    (plus ``grad_nonfinite_params: [...]`` when false) to every trainer
    run record.  Returns ``(flag, nonfinite_params)``: ``flag`` is False
    as soon as ANY record reports a non-finite gradient (one bad update
    poisons the run), True when records exist and all report finite,
    and ``None`` when NO record carries the field (older records → the
    gradient screen is unavailable, never a silent pass).
    """
    rows = (read_jsonl(Path(run_dir) / "run_records.jsonl")
            + read_jsonl(Path(run_dir) / "train_log.jsonl"))
    flag: Optional[bool] = None
    params: List[str] = []
    for row in rows:
        if TRAINER_GRAD_FINITE_FIELD not in row:
            continue
        value = bool(row[TRAINER_GRAD_FINITE_FIELD])
        flag = value if flag is None else (flag and value)
        if not value:
            listed = row.get(TRAINER_GRAD_NONFINITE_PARAMS_FIELD)
            if isinstance(listed, (list, tuple)):
                for name in listed:
                    if str(name) not in params:
                        params.append(str(name))
    return flag, (params or None)


def diagnostic_metrics_from_run(
        run_dir: Any, *,
        evaluator_metrics: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Assemble one candidate's diagnostic metrics from its run artifacts.

    Training-derived inputs (always available): initial/best inner-val
    total loss from the append-only ``run_records.jsonl`` (the initial
    diagnostic evaluation = the FIRST inner-val sweep), loss
    finiteness from ``train_log.jsonl`` AND the gradient-finiteness
    screen input ``grad_finite_all_finite`` (+ ``grad_nonfinite_params``
    when false) read from the trainer run records — the gradient half
    NEVER depends on the optional evaluator metrics (oracle gate-3 C3).
    Evaluator-derived inputs (dead fraction / RMSE/zero, plus the legacy
    evaluator ``grad_finite`` fallback) come from the optional evaluator
    metrics mapping and stay ``None`` (not-measured) when absent.
    """
    run_dir = Path(run_dir)
    records = [r for r in read_jsonl(run_dir / "run_records.jsonl")
               if r.get("event") == "inner_val"]
    _require(bool(records), "V542_DIAG_NO_INNER_VAL",
             f"no inner-val records under {run_dir} — cannot form the "
             "diagnostic screens")
    initial = float(records[0]["inner_val_total_loss"])
    best = min(float(r["inner_val_total_loss"]) for r in records)
    trains = [r for r in read_jsonl(run_dir / "train_log.jsonl")
              if r.get("event") == "train"]
    loss_finite = all(_finite(r.get("loss")) for r in trains) if trains \
        else False
    grad_flag, grad_nonfinite_params = _grad_finite_from_run_records(run_dir)
    evaluator = dict(evaluator_metrics or {})
    return {
        "initial_inner_val_total_loss": initial,
        "best_inner_val_total_loss": best,
        "loss_finite": bool(loss_finite),
        TRAINER_GRAD_FINITE_FIELD: grad_flag,
        TRAINER_GRAD_NONFINITE_PARAMS_FIELD: grad_nonfinite_params,
        "grad_finite": evaluator.get("grad_finite"),
        "dead_fraction": evaluator.get("dead_fraction"),
        "rmse_zero": evaluator.get("rmse_zero"),
        "inner_val_sweeps": len(records),
        "updates_logged": len(trains),
        "evaluator_metrics_provided": bool(evaluator),
    }


def run_diagnostic(env: Any, *,
                   config: Mapping[str, Any],
                   campaign_root: Any,
                   candidates: Optional[Sequence[str]] = None,
                   identity_by_candidate: Optional[Mapping[str, Any]] = None,
                   device: str = "cpu",
                   seed: int = DIAGNOSTIC_SEED,
                   updates: int = DIAGNOSTIC_UPDATES_DEFAULT,
                   w0: int = DIAGNOSTIC_W0_DEFAULT,
                   w1: int = DIAGNOSTIC_W1_DEFAULT,
                   evaluator_metrics: Optional[Mapping[str, Mapping[str, Any]]] = None,
                   allow_unevaluated_grad_screen: bool = False,
                   overrides: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Run the seed-17 diagnostic dose for A0–A5 and screen ENGINEERING-only.

    A0 ALWAYS runs first (the ``V542_DIAG_A0_S17`` same-dose control must
    exist before any candidate is judged); every candidate — A0 included —
    is screened and recorded with an explicit reason code.  No ranking,
    no H3 effect gates.  A candidate whose trainer run records lack the
    gradient-finiteness field cannot ADVANCE to full dose unless
    ``allow_unevaluated_grad_screen`` (the explicit CLI override
    ``--allow-unevaluated-grad-screen``) is given; the override is
    recorded on the decision row.
    """
    from pes2ts_core.pesn.offline.training.v542_train import train_run_v542

    guards = assert_operational_guards(config)
    resolved = resolve_candidates(candidates, conditional=False)
    _require("A0" in resolved, "V542_CAMPAIGN_CANDIDATES",
             "the diagnostic MUST include A0 — V542_DIAG_A0_S17 is the "
             "same-dose control (without it the screens systematically "
             "lean toward eliminating everything)")
    updates, w0, w1 = int(updates), int(w0), int(w1)
    _require(w0 + w1 == updates, "V542_CAMPAIGN_DIAGNOSTIC_DOSE",
             f"diagnostic dose must split exactly: w0 ({w0}) + w1 ({w1}) "
             f"== total ({updates})")
    thresholds = diagnostic_thresholds(config)
    root = Path(campaign_root) / "diagnostics"
    order = ["A0"] + [c for c in resolved if c != "A0"]

    control_metrics: Optional[Dict[str, Any]] = None
    control_run_id = "{}DIAG_A0_S{}".format(RUN_ID_NAMESPACE, int(seed))
    rows: List[Dict[str, Any]] = []
    for candidate_id in order:
        identity = _identity_for(identity_by_candidate, candidate_id)
        run_id = "{}DIAG_{}_S{}".format(
            RUN_ID_NAMESPACE, candidate_id.replace("-", ""), int(seed))
        run_dir = root / run_id
        cfg = _trainer_config(
            config, candidate_id=candidate_id, seed=seed, freeze=w0,
            joint=w1, smoke_updates=None, device=device, overrides=overrides)
        metrics = train_run_v542(
            env.ds, env.labels, env.endpoint, cfg, run_dir,
            dev_positions=env.dev_pos, identity=identity,
            require_identity=True, resume=True)
        diag = diagnostic_metrics_from_run(
            run_dir,
            evaluator_metrics=(evaluator_metrics or {}).get(candidate_id))
        if candidate_id == "A0":
            control_metrics = diag
        decision = apply_diagnostic_screens(
            diag, control_metrics, thresholds,
            allow_unevaluated_grad_screen=allow_unevaluated_grad_screen)
        row = _common_row(
            kind="diagnostic", run_id=run_id, candidate_id=candidate_id,
            seed=seed, device=device, guards=guards, identity=identity,
            config=cfg)
        row.update(_environment_record(device))
        row.update({
            "dose": {"w0_updates": w0, "w1_updates": w1,
                     "total_updates": updates},
            "is_control": candidate_id == "A0",
            "control_run_id": control_run_id,
            "diag_metrics": diag,
            "decision": decision,
            "reason_code": decision["reason_code"],
            "run_outcome": metrics.get("run_outcome"),
            "run_dir": str(run_dir),
            "evaluator_metrics_supplied": bool(
                (evaluator_metrics or {}).get(candidate_id)),
        })
        assert_technical_result(row)
        append_jsonl(root / "diagnostic_report.jsonl", row)
        rows.append(row)
    return rows


def plan_diagnostic(*, config: Mapping[str, Any], campaign_root: Any,
                    candidates: Optional[Sequence[str]] = None,
                    device: str = "cpu", seed: int = DIAGNOSTIC_SEED,
                    updates: int = DIAGNOSTIC_UPDATES_DEFAULT,
                    w0: int = DIAGNOSTIC_W0_DEFAULT,
                    w1: int = DIAGNOSTIC_W1_DEFAULT) -> Dict[str, Any]:
    """The planned diagnostic steps (--dry-run; no training, no writes)."""
    guards = assert_operational_guards(config)
    resolved = resolve_candidates(candidates, conditional=False)
    _require("A0" in resolved, "V542_CAMPAIGN_CANDIDATES",
             "the diagnostic MUST include A0 — V542_DIAG_A0_S17 is the "
             "same-dose control")
    root = Path(campaign_root) / "diagnostics"
    control_run_id = "{}DIAG_A0_S{}".format(RUN_ID_NAMESPACE, int(seed))
    return {
        "kind": "diagnostic",
        "dry_run": True,
        "training_performed": False,
        "device": str(device),
        "seed": int(seed),
        "dose": {"w0_updates": int(w0), "w1_updates": int(w1),
                 "total_updates": int(updates)},
        "control_run_id": control_run_id,
        "thresholds": diagnostic_thresholds(config),
        "screens": list(SCREEN_ORDER),
        "h3_effect_gates_applied": False,
        "ranking_applied": False,
        "steps": [{
            "step": index,
            "run_id": "{}DIAG_{}_S{}".format(
                RUN_ID_NAMESPACE, c.replace("-", ""), int(seed)),
            "candidate_id": c,
            "is_control": c == "A0",
            "control_run_id": control_run_id,
            "run_dir": str(root / "{}DIAG_{}_S{}".format(
                RUN_ID_NAMESPACE, c.replace("-", ""), int(seed))),
        } for index, c in enumerate(["A0"] + [x for x in resolved if x != "A0"],
                                    start=1)],
        "report_path": str(root / "diagnostic_report.jsonl"),
        "rank_excluded": True,
        "guards": guards,
    }


# ---------------------------------------------------------------------------
# formal gate (typed-blocked by the frozen validator while budget is null)
# ---------------------------------------------------------------------------
def formal_gate(config_path: Any) -> None:
    """``--formal`` DELEGATES to the frozen config validator.

    While ``budget.per_candidate_core_time`` is null (the legal
    pre-profile WP0 state) the validator refuses with the typed
    ``V542_BUDGET_NULL`` block, which this harness surfaces verbatim —
    a formal launch can NEVER happen from here.  Even if the validator
    one day passes, this technical-measurement harness still refuses to
    execute formal runs (they belong to the formal campaign lane).
    """
    from scripts.agent_ops.v542_config_schema import (
        V542SchemaError,
        load_and_validate,
    )

    try:
        load_and_validate(config_path, formal=True)
    except V542SchemaError as exc:
        raise V542CampaignError(
            exc.code,
            "formal launch is typed-blocked by the frozen config validator: "
            + exc.detail) from exc
    raise V542CampaignError(
        "V542_CAMPAIGN_FORMAL_DELEGATED",
        "the validator passed the formal gate; formal runs are executed by "
        "the formal campaign lane, never by this technical-measurement "
        "harness (profile/smoke/diagnostic only)")


# ---------------------------------------------------------------------------
# frozen data view (CLI runs; tests inject their own env)
# ---------------------------------------------------------------------------
def _subset_curve_ids(split_doc: Mapping[str, Any], subset: str) -> List[str]:
    """curve_ids of one frozen split subset (``dev`` only — confirm is
    NEVER read; minimal local reader, no heavy imports)."""
    subsets = split_doc.get("subsets")
    _require(isinstance(subsets, Mapping) and subset in subsets,
             "V542_CAMPAIGN_SPLIT", f"split manifest has no subsets.{subset}")
    entries = subsets[subset]
    _require(isinstance(entries, (list, tuple)), "V542_CAMPAIGN_SPLIT",
             f"subsets.{subset} is not a list")
    ids: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, Mapping) and "curve_id" in entry:
            ids.append(str(entry["curve_id"]))
        else:  # pragma: no cover - schema-guarded input
            raise V542CampaignError(
                "V542_CAMPAIGN_SPLIT",
                f"subsets.{subset} entry is not a curve_id/record: {entry!r}")
    return ids


def load_campaign_env(cfg: Mapping[str, Any]) -> Any:
    """Open the FROZEN 50K data view READ-ONLY (the existing columnar
    loader — no new pipeline; dev positions are curve-EXCLUSION only and
    confirm is never read)."""
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
        dev_positions_from_split,
    )

    tables = Path((cfg.get("data_contract") or {}).get("tables_dir")
                  or "/pes2ts_data/v53mp_50k_data")
    split = Path((cfg.get("split_contract") or {}).get("split_manifest")
                 or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
    for path in (tables / "data_manifest.json",
                 tables / "teacher_manifest.json", split):
        _require(path.is_file(), "V542_CAMPAIGN_DATA_MISSING",
                 f"frozen data view artifact missing: {path}")
    dataset = ColumnarKeyAtomDatasetV3(
        str(tables), split_manifest=str(split),
        label_statuses=LABEL_STATUSES, cache_curves=65000)
    labels, endpoint = dataset.label_tables()
    split_doc = json.loads(split.read_text(encoding="utf-8"))
    dev_ids = _subset_curve_ids(split_doc, "dev")
    dev_pos = list(dev_positions_from_split(dataset.samples, set(dev_ids)))
    _require(bool(dev_pos), "V542_CAMPAIGN_DEV_POSITIONS",
             "frozen dev subset has no loaded samples (exclusion impossible)")
    return SimpleNamespace(ds=dataset, labels=labels, endpoint=endpoint,
                           dev_pos=dev_pos)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def validate_cli_seed(seed: Any) -> int:
    """Typed validation of a CLI ``--seed`` value (frozen seeds only).

    An invalid seed (e.g. 99) is a typed ``V542_CAMPAIGN_SEED_FROZEN``
    refusal — never a downstream trainer traceback (oracle gate-3 C5).
    """
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise V542CampaignError(
            "V542_CAMPAIGN_SEED_FROZEN",
            f"--seed must be an integer in {list(FROZEN_RUN_SEEDS)}, "
            f"got {seed!r}") from exc
    _require(value in FROZEN_RUN_SEEDS, "V542_CAMPAIGN_SEED_FROZEN",
             f"--seed must be one of the frozen seeds "
             f"{list(FROZEN_RUN_SEEDS)} (config "
             f"init_policy.process_head_anchor.seeds), got {value!r}")
    return value


def validate_cli_device(device: Any) -> str:
    """Typed validation of a CLI ``--device`` value.

    ``cpu`` is always allowed (fully supported by this harness); a
    ``cuda``/``cuda:N`` device must be AVAILABLE on this host, else a
    typed ``V542_CAMPAIGN_DEVICE_UNAVAILABLE`` refusal — never a
    downstream trainer traceback (oracle gate-3 C5).  Any other device
    string is a typed ``V542_CAMPAIGN_DEVICE_UNKNOWN`` refusal.
    """
    dev = str(device)
    if dev == "cpu":
        return dev
    if dev.startswith("cuda"):
        import torch

        _require(bool(torch.cuda.is_available()),
                 "V542_CAMPAIGN_DEVICE_UNAVAILABLE",
                 f"device={dev!r} requested but CUDA is not available on "
                 "this host (cpu is fully supported; cuda must be "
                 "available to be requested)")
        if dev != "cuda":
            try:
                index = int(dev.split(":", 1)[1])
            except (IndexError, ValueError) as exc:
                raise V542CampaignError(
                    "V542_CAMPAIGN_DEVICE_UNKNOWN",
                    f"--device must be 'cpu', 'cuda' or 'cuda:N' "
                    f"(N a non-negative integer), got {dev!r}") from exc
            _require(0 <= index < torch.cuda.device_count(),
                     "V542_CAMPAIGN_DEVICE_UNAVAILABLE",
                     f"device={dev!r} requested but this host has "
                     f"{torch.cuda.device_count()} CUDA device(s)")
        return dev
    raise V542CampaignError(
        "V542_CAMPAIGN_DEVICE_UNKNOWN",
        f"--device must be 'cpu' or 'cuda[:N]', got {dev!r}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    parser.add_argument("--device", default="cpu",
                        help="torch device (cpu fully supported; cuda must be "
                        "requested explicitly)")
    parser.add_argument("--seed", type=int, default=None,
                        help="measurement seed (default: the frozen "
                        "per-subcommand seed, always in {17,29,43})")
    parser.add_argument("--conditional", action="store_true",
                        help="allow the A1-C/A5-C diagnostic variants "
                        "(recorded separately from the main matrix)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned steps WITHOUT training "
                        "(no outputs are written)")
    parser.add_argument("--formal", action="store_true",
                        help="attempt a formal launch: DELEGATES to the "
                        "frozen config validator (typed-blocked with "
                        "V542_BUDGET_NULL while budget.per_candidate_core_"
                        "time is null; this harness never runs formal)")


def _campaign_refusal_from_exception(
        exc: BaseException) -> Optional[V542CampaignError]:
    """Convert an EXPECTED failure class into a typed campaign refusal.

    The frozen trainer (``V542TrainerError``), the frozen config
    validator (``V542SchemaError``) and a missing frozen data-view
    artifact (``FileNotFoundError``) are expected failure classes: they
    surface as a typed ``V542_CAMPAIGN_*`` refusal with a non-zero exit,
    never an uncaught traceback (oracle gate-3 C5).  Anything else
    returns ``None`` — the caller MUST re-raise unexpected exceptions
    (they are never swallowed).
    """
    try:
        from pes2ts_core.pesn.offline.training.v542_train import (
            V542TrainerError,
        )
        trainer_refusal = isinstance(exc, V542TrainerError)
    except Exception:  # noqa: BLE001 - trainer import unavailable
        trainer_refusal = False
    if trainer_refusal:
        return V542CampaignError(
            "V542_CAMPAIGN_TRAINER_REFUSAL",
            f"frozen trainer refused (code "
            f"{getattr(exc, 'code', None)!r}): {exc}")
    try:
        from scripts.agent_ops.v542_config_schema import V542SchemaError
        if isinstance(exc, V542SchemaError):
            return V542CampaignError(
                getattr(exc, "code", "V542_CAMPAIGN_VALIDATOR_REFUSAL"),
                f"frozen config validator refused: {exc}")
    except Exception:  # noqa: BLE001 - validator import unavailable
        pass
    if isinstance(exc, FileNotFoundError):
        return V542CampaignError(
            "V542_CAMPAIGN_DATA_MISSING",
            f"frozen data-view artifact missing: {exc}")
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v542_campaign.py",
        description="V5.4.2 WP4 campaign technical-measurement harness "
                    "(smoke / profile / diagnostic; never formal, never "
                    "scientific results)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_profile = sub.add_parser(
        "profile", help="same-environment 40 W0 + 40 W1 profile (20 warm-up)")
    _add_common_args(p_profile)
    p_profile.add_argument("--candidates", nargs="*", default=None)
    p_profile.add_argument("--w0", type=int, default=PROFILE_W0_DEFAULT)
    p_profile.add_argument("--w1", type=int, default=PROFILE_W1_DEFAULT)
    p_profile.add_argument("--warmup", type=int, default=PROFILE_WARMUP_DEFAULT)

    p_smoke = sub.add_parser("smoke", help="200-update single-lane smoke "
                                           "with one kill-resume cycle")
    _add_common_args(p_smoke)
    p_smoke.add_argument("--candidates", nargs="*", default=None)
    p_smoke.add_argument("--updates", type=int, default=SMOKE_UPDATES_DEFAULT)
    p_smoke.add_argument("--kill-at", type=int, default=None,
                         help="kill point of the bounded first leg "
                         "(default: updates // 2)")

    p_diag = sub.add_parser(
        "diagnostic",
        help="2,000 = 400 + 1,600 engineering-screen diagnostic with the "
             "A0@2000 same-dose control")
    _add_common_args(p_diag)
    p_diag.add_argument("--candidates", nargs="*", default=None)
    p_diag.add_argument("--updates", type=int,
                        default=DIAGNOSTIC_UPDATES_DEFAULT)
    p_diag.add_argument("--w0", type=int, default=DIAGNOSTIC_W0_DEFAULT)
    p_diag.add_argument("--w1", type=int, default=DIAGNOSTIC_W1_DEFAULT)
    p_diag.add_argument(
        "--evaluator-metrics", default=None,
        help="optional JSON file {candidate: {dead_fraction, rmse_zero, "
             "grad_finite}} with the evaluator-side diagnostic metrics")
    p_diag.add_argument(
        "--allow-unevaluated-grad-screen", action="store_true",
        help="allow advancing a candidate to full dose even when the "
             "trainer run records lack grad_finite_all_finite (the "
             "override is recorded on the decision row; refused in "
             "formal mode)")

    args = parser.parse_args(argv)
    try:
        if getattr(args, "formal", False):
            # the unevaluated-gradient-screen override is a technical-
            # measurement escape hatch; it can never ride along a formal
            # launch (oracle gate-3 C3).
            _require(not getattr(args, "allow_unevaluated_grad_screen",
                                 False),
                     "V542_CAMPAIGN_GRAD_OVERRIDE_FORMAL_FORBIDDEN",
                     "--allow-unevaluated-grad-screen is refused in formal "
                     "mode: a formal launch may never advance a candidate "
                     "on an unevaluated gradient screen")
            formal_gate(args.config)
            return 2  # pragma: no cover - formal_gate always raises

        cfg = load_campaign_config(args.config)
        seed_default = {"profile": PROFILE_SEED, "smoke": SMOKE_SEED,
                        "diagnostic": DIAGNOSTIC_SEED}[args.command]
        # typed validation FIRST — an invalid --seed / unavailable
        # --device is a typed refusal, never a trainer traceback
        # (oracle gate-3 C5).
        seed = validate_cli_seed(
            args.seed if args.seed is not None else seed_default)
        device = validate_cli_device(args.device)

        if args.dry_run:
            if args.command == "profile":
                plan = plan_profile(
                    config=cfg, campaign_root=args.campaign_root,
                    candidates=args.candidates, conditional=args.conditional,
                    device=device, seed=seed, w0=args.w0, w1=args.w1,
                    warmup=args.warmup)
            elif args.command == "smoke":
                plan = plan_smoke(
                    config=cfg, campaign_root=args.campaign_root,
                    candidates=args.candidates, conditional=args.conditional,
                    device=device, seed=seed, updates=args.updates,
                    kill_at=args.kill_at)
            else:
                plan = plan_diagnostic(
                    config=cfg, campaign_root=args.campaign_root,
                    candidates=args.candidates, device=device,
                    seed=seed, updates=args.updates, w0=args.w0, w1=args.w1)
            print(json.dumps(plan, indent=2, sort_keys=True, default=str))
            return 0

        env = load_campaign_env(cfg)
        if args.command == "profile":
            candidates = resolve_candidates(args.candidates,
                                            conditional=args.conditional)
            identities = {c: build_identity_record(
                args.config, c, seed) for c in candidates}
            rows = run_profile(
                env, config=cfg, campaign_root=args.campaign_root,
                candidates=candidates, conditional=args.conditional,
                identity_by_candidate=identities, device=device,
                seed=seed, w0=args.w0, w1=args.w1, warmup=args.warmup)
        elif args.command == "smoke":
            candidates = resolve_candidates(args.candidates,
                                            conditional=args.conditional)
            identities = {c: build_identity_record(
                args.config, c, seed) for c in candidates}
            rows = run_smoke(
                env, config=cfg, campaign_root=args.campaign_root,
                candidates=candidates, conditional=args.conditional,
                identity_by_candidate=identities, device=device,
                seed=seed, updates=args.updates, kill_at=args.kill_at)
        else:
            candidates = resolve_candidates(args.candidates, conditional=False)
            identities = {c: build_identity_record(
                args.config, c, seed) for c in candidates}
            evaluator = None
            if args.evaluator_metrics:
                try:
                    evaluator = json.loads(
                        Path(args.evaluator_metrics).read_text(
                            encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise V542CampaignError(
                        "V542_CAMPAIGN_EVALUATOR_METRICS_INVALID",
                        f"--evaluator-metrics file unreadable/invalid: "
                        f"{args.evaluator_metrics} ({exc})") from exc
            rows = run_diagnostic(
                env, config=cfg, campaign_root=args.campaign_root,
                candidates=candidates, identity_by_candidate=identities,
                device=device, seed=seed, updates=args.updates,
                w0=args.w0, w1=args.w1, evaluator_metrics=evaluator,
                allow_unevaluated_grad_screen=bool(
                    getattr(args, "allow_unevaluated_grad_screen", False)))
        summary = {
            "kind": args.command,
            "rows": len(rows),
            "reason_codes": [r.get("reason_code") for r in rows],
            "run_ids": [r.get("run_id") for r in rows],
            "rank_excluded": True,
            "scientific_result": False,
        }
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0
    except V542CampaignError as exc:
        print(f"V542 campaign refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - typed conversion, re-raise below
        # expected failure classes from the frozen trainer/config stack
        # become a typed campaign refusal; UNEXPECTED exceptions are
        # re-raised (never swallowed).
        typed = _campaign_refusal_from_exception(exc)
        if typed is None:
            raise
        print(f"V542 campaign refused: {typed}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
