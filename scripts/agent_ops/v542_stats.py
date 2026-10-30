#!/usr/bin/env python3
"""V5.4.2 WP6 statistics + selection machinery (ADR-0033 §2.5/§2.6,
config ``evaluation_contract``, plan §8.2/§9).

One CLI, six subcommands over the FROZEN pre-registered contracts —
every threshold, formula, family and rule block used here is the frozen
value (mirrored from the frozen config / ``v542_config_schema``); none
of it may be tuned after the fact:

* ``extract`` — load each finished formal run's ``checkpoint_best.pt``
  through the FROZEN evaluator
  ``key_atom_v53mp_train_v3.evaluate_v53mp`` over the SAME seed inner-val
  positions and the SAME config as the ``V542-A0-sanity-band`` leg
  (``v542_formal`` position derivation is reused, never forked) and
  APPEND one record per run/seed to
  ``<campaign_root>/reports/metrics_table.json`` — five metric leaves
  (``rmse_zero``, ``direction`` [fixed-coverage direction accuracy IS
  the H2 sign accuracy in the frozen V5.3-MP evaluator — see
  ``v542_a0_parity``], ``dead_fraction``, ``endpoint_auprc``,
  ``q_corr``) plus inner-val total loss and core time from the run
  ledger, with per-run identity digests and full provenance.  The table
  is append-only: an already-present (run_id, identity, entry digest)
  triple is skipped idempotently; records are never rewritten;
  last-record-per-run wins on load (same discipline as the ledger).

* ``h2`` — the frozen H2 guards for every formal candidate run against
  BOTH the same-seed matched A0 AND the R3 reference (R3 is LOCATED
  from ``plan/r3_reference_metrics.json``, never fabricated): RMSE/zero
  relative degradation ≤ 2%; direction (sign) absolute drop ≤ 0.01;
  dead-fraction absolute increase ≤ 0.01; fixed-coverage direction
  relative degradation ≤ 2% — each guard on each of the two sides.
  A breach marks the run ``scientific_failure`` IN THE H2 REPORT ONLY
  — the run ledger is never rewritten.  All values must be finite and
  the R3 side is mandatory (``r3_guard_block``: no silent downgrade to
  a matched-A0-only comparison).

* ``screen`` — the frozen ``seed17_screen_pass`` predicate for A1–A4:
  run done, all pre-registered metrics finite, H0/H1/diagnostic/cost≤2
  gates pass, matched A0_S17 done, seed17 H2 all-guards-clear, and at
  least one route passing under the frozen parenthesized
  ``route_logic`` (dangling-OR forbidden — each branch closes its own
  bracket).  ``screen_score = max(direction_rel/0.03,
  rmse_rel_improve/0.03, delta_auprc/0.02, delta_q/0.03)`` (tie: inner-
  val total loss, then core time, then candidate id).  A4 is always
  reported ineligible (``gate_skipped`` with the binding WP4 diagnostic
  reason).  The top2 freeze precondition (A0_S17 done + ``V542-A0-
  sanity-band`` pass) is asserted; the top2 set is ≤ 2 candidates with
  NO substitutes.

* ``h3`` — the frozen paired cluster bootstrap: seed-averaged
  curve-level paired bootstrap, 10,000 final draws,
  ``bootstrap_seed_root=20261029``, deterministic per-endpoint sub-seed
  ``int(sha256(f"v542|{root}|{seed}|{endpoint}").hexdigest()[:16], 16)
  % 2147483647``; CI and p come from the SAME draws; one-sided
  improvement ``p = (1 + #{improvement ≤ 0}) / 10001`` (RMSE improvement
  = A0−candidate; direction/AUPRC/q = candidate−A0).  Holm: MOVE
  direction and MOVE RMSE are two exits each at α=0.025, each Holm over
  the eligible candidate set; PROCESS is one family over 2 endpoints ×
  eligible candidates at family α=0.025.  Both unadjusted and adjusted
  p, effect sizes, CIs and each candidate's MOVE/PROCESS route verdict
  (frozen ``route_logic`` with confirmed pass-components) are reported.

* ``select`` — the frozen ``final_selection_rule``: H3-qualified only,
  A0/diagnostics excluded, 0 ⇒ ``no_backbone_winner`` (A0 retained),
  1 ⇒ that candidate, many ⇒ exact 7-step lexicographic order;
  ``dev_used=false``; every qualified candidate is reported; the rule
  block and its canonical SHA go into the selection record.

* ``activate`` — the frozen activation decisions: A5 full-dose =
  ``no_A1_A4_h3_qualified_candidate_after_top2_seeds`` (an A4 that
  never ran is a CONSEQUENCE of that condition, never a blocker); A5
  supplement = ``a5_full_dose_complete_and_seed17_screen_pass``; A1-C =
  ``a1_in_top2_supplement_set_OR_a1_selected_as_backbone``; A5-C =
  ``a5_selected_backbone`` only.  Every skip carries the frozen reason
  code and zero new training.

Discipline: no ``torch.compile``; no dev read (dev_used=false — dev is
evaluated once after the backbone freeze and NEVER participates in
selection); no QC/confirm access; the CLI exposes NO threshold/dose
knobs (draw counts exist only as program-level test seams and always
default to the frozen 10,000); frozen thresholds/rule wording are never
modified; the running run ledger is read-only for this lane.

CLI::

    python scripts/agent_ops/v542_stats.py extract [--device cpu]
                                                   [--run RUN_ID ...]
    python scripts/agent_ops/v542_stats.py h2 [--device cpu]
                                             [--r3-metrics PATH]
    python scripts/agent_ops/v542_stats.py screen
    python scripts/agent_ops/v542_stats.py h3 [--device cpu]
    python scripts/agent_ops/v542_stats.py select
    python scripts/agent_ops/v542_stats.py activate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Frozen constants mirrored from the schema module — imported, never
# copied-and-edited (the schema module IS the frozen single source).
from scripts.agent_ops.v542_config_schema import (  # noqa: E402
    FINAL_SELECTION_TIE_BREAK,
    H2_NO_BREACH_THRESHOLDS,
    MOVE_PASS_LOGIC,
    PROCESS_PASS_LOGIC,
)
# WP5 formal runner: ledger/band readers, position derivation, evaluator
# identity and the frozen reason codes are REUSED, never forked.
from scripts.agent_ops.v542_formal import (  # noqa: E402
    A0_SANITY_BAND_GATE,
    A5_FROZEN_CONDITION,
    A5_SUPPLEMENT_CONDITION,
    EVALUATOR_IDENTITY,
    REASON_A1C_SKIP,
    REASON_A4_GATE_SKIPPED,
    REASON_A5C_SKIP,
    REASON_A5_NOT_ACTIVATED,
    REASON_A5_SUPPLEMENT,
    cost_gate_ratio,
    diagnostic_disposition,
    latest_ledger_state,
    read_band_record,
    read_ledger,
)

DEFAULT_CONFIG = PROJECT_ROOT / (
    "config/campaigns/pes2ts_v542_backbone_50k.yaml")
#: config ``plan_outputs.root``.
DEFAULT_CAMPAIGN_ROOT = PROJECT_ROOT / "outputs/pesn/v542_backbone_50k"

# ---------------------------------------------------------------------------
# report locations (all under <campaign_root>; append-only discipline)
# ---------------------------------------------------------------------------
REPORTS_DIR_REL = "reports"
METRICS_TABLE_REL = "reports/metrics_table.json"
H2_REPORT_REL = "reports/h2_report.json"
SCREEN_REPORT_REL = "reports/seed17_screen_report.json"
H3_REPORT_REL = "reports/h3_report.json"
SELECTION_RECORD_REL = "reports/final_selection.json"
ACTIVATION_REPORT_REL = "reports/activation_decisions.json"
#: Located (never fabricated) R3 reference metrics for the H2 lane.
R3_REFERENCE_REL = "plan/r3_reference_metrics.json"

#: Report schema ids.
METRICS_TABLE_SCHEMA = "pes2ts_v542_metrics_table_v1"
METRICS_ENTRY_SCHEMA = "pes2ts_v542_metrics_entry_v1"
H2_REPORT_SCHEMA = "pes2ts_v542_h2_report_v1"
SCREEN_REPORT_SCHEMA = "pes2ts_v542_seed17_screen_report_v1"
H3_REPORT_SCHEMA = "pes2ts_v542_h3_report_v1"
SELECTION_SCHEMA = "pes2ts_v542_final_selection_v1"
ACTIVATION_SCHEMA = "pes2ts_v542_activation_decisions_v1"
R3_REFERENCE_SCHEMA = "pes2ts_v542_r3_reference_metrics_v1"

FROZEN_SEEDS: tuple = (17, 29, 43)
#: Stages whose finished runs enter the metrics table (formal only).
EXTRACT_STAGES: tuple = ("T0", "T1", "T2", "T3")
TOP2_SEEDS: tuple = (29, 43)
TOP2_CANDIDATES: tuple = ("A1", "A2", "A3", "A4")

# ---------------------------------------------------------------------------
# the five frozen metric leaves (extract provenance map)
# ---------------------------------------------------------------------------
METRIC_KEYS: tuple = (
    "rmse_zero", "direction", "dead_fraction", "endpoint_auprc", "q_corr")
#: logical metric -> leaf path inside an ``evaluate_v53mp`` document.
#: ``direction`` IS the H2 sign accuracy: the frozen V5.3-MP evaluator
#: has no separate ``fixed_coverage_direction`` leaf (v542_a0_parity
#: certificate note) — the frozen evaluator's direction metric IS
#: ``sre_and_sign.sign_accuracy_cluster_median``.
METRIC_LEAVES: Dict[str, str] = {
    "rmse_zero": "move.key_rmse_over_zero.median_rmse_over_zero",
    "direction": "move.sre_and_sign.sign_accuracy_cluster_median",
    "dead_fraction": "move.sre_and_sign.dead_zone_fraction",
    "endpoint_auprc": "endpoint_head.auprc",
    "q_corr": "q_head.corr_curve_median",
}

# ---------------------------------------------------------------------------
# frozen H3 / route / screen numeric thresholds (mirror the config
# thresholds block — values are pre-registered, never tuned here)
# ---------------------------------------------------------------------------
DIRECTION_REL_MIN = 0.03          # thresholds.h3_move_direction_rel_min_pct
DIRECTION_ABS_MIN = 0.01          # thresholds.h3_move_direction_abs_min
RMSE_REL_MIN = 0.03               # thresholds.h3_move_rmse_rel_min_pct
AUPRC_DELTA_MIN = 0.02            # thresholds.h3_process_delta_auprc_abs_min
Q_DELTA_MIN = 0.03                # thresholds.h3_process_delta_q_corr_abs_min
AUPRC_DROP_MAX = 0.01             # thresholds.other_endpoint_auprc_drop_abs_max
Q_DROP_MAX = 0.02                 # thresholds.other_q_corr_drop_abs_max
MOVE_EXIT_ALPHA = 0.025           # thresholds.move_exit_alpha_each
PROCESS_FAMILY_ALPHA = 0.025      # statistics.holm_family.process_route.alpha
MAX_CORE_TIME_RATIO = 2.0         # thresholds.max_core_time_ratio

#: statistics.p_value_rule.improvement_direction (frozen).
IMPROVEMENT_DIRECTION: Dict[str, str] = {
    "move_direction": "candidate_minus_a0",
    "move_rmse": "a0_minus_candidate",
    "process_auprc": "candidate_minus_a0",
    "process_q": "candidate_minus_a0",
}
ENDPOINTS: tuple = (
    "move_direction", "move_rmse", "process_auprc", "process_q")
#: endpoint -> metric key in the metrics table.
ENDPOINT_TO_METRIC: Dict[str, str] = {
    "move_direction": "direction",
    "move_rmse": "rmse_zero",
    "process_auprc": "endpoint_auprc",
    "process_q": "q_corr",
}
#: statistics (frozen bootstrap contract).
BOOTSTRAP_SEED_ROOT = 20261029    # thresholds.bootstrap_seed_root
FINAL_DRAWS = 10000               # thresholds.final_bootstrap_draws
CI_LOW_PERCENTILE = 2.5
CI_HIGH_PERCENTILE = 97.5
H3_MIN_SEEDS_SAME_DIRECTION = 2   # seed_aggregation.min_seeds_same_direction
H3_TOTAL_SEEDS = 3                # seed_aggregation.total_seeds

#: evaluation_contract.selection_score (frozen screen score).
SCREEN_SCORE_NORMALIZERS: Dict[str, float] = {
    "direction_rel": DIRECTION_REL_MIN,
    "rmse_rel_improve": RMSE_REL_MIN,
    "delta_auprc": AUPRC_DELTA_MIN,
    "delta_q": Q_DELTA_MIN,
}
#: structured rule ids (schema-module single source).
from scripts.agent_ops.v542_config_schema import (  # noqa: E402
    FINAL_SELECTION_ELIGIBLE_CONDITION,
    FINAL_SELECTION_NO_CANDIDATE_RULE_ID,
    FINAL_SELECTION_SINGLE_RULE_ID,
)

#: Frozen skip/disposition reason codes (WP4 ruling + runner constants).
A4_DIAG_REASON = REASON_A4_GATE_SKIPPED
A1C_SKIP_REASON = REASON_A1C_SKIP
A5C_SKIP_REASON = REASON_A5C_SKIP
A5_SKIP_REASON = REASON_A5_NOT_ACTIVATED
A5_SUPPLEMENT_SKIP_REASON = REASON_A5_SUPPLEMENT
CAPACITY_CAVEAT_UNRESOLVED = "capacity_caveat_unresolved"


class V542StatsError(RuntimeError):
    """Typed refusal of the WP6 statistics lane (``code`` + ``detail``)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}" if detail else str(code))
        self.code = str(code)
        self.detail = str(detail)


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise V542StatsError(code, detail)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _canonical_sha256(obj: Any) -> str:
    """The ONE canonical digest recipe (identity ``canonical_sha256``).

    Imported lazily so ``--help`` and pure-logic paths stay torch-free;
    the recipe itself is NEVER copied into this module.
    """
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256

    return canonical_sha256(obj)


def _write_report(campaign_root: Any, rel: str,
                  doc: Mapping[str, Any]) -> Path:
    path = Path(campaign_root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(doc), indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return path


def _load_json(path: Any, code: str, label: str) -> Dict[str, Any]:
    p = Path(path)
    _require(p.is_file(), code, f"{label} not found: {p}")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V542StatsError(code, f"{label} unreadable: {p}: {exc}") from exc
    _require(isinstance(doc, Mapping), code,
             f"{label} is not a JSON object: {p}")
    return dict(doc)


# ===========================================================================
# 1. extract — frozen-evaluator metric table (append-only)
# ===========================================================================
def metric_values_from_eval(report: Mapping[str, Any]) -> Dict[str, float]:
    """The five frozen metric leaves out of an ``evaluate_v53mp`` doc."""
    out: Dict[str, float] = {}
    for key, leaf in METRIC_LEAVES.items():
        node: Any = report
        for part in leaf.split("."):
            node = (node or {}).get(part) if isinstance(node, Mapping) else None
        _require(_finite(node), "V542_STATS_METRIC_NON_FINITE",
                 f"frozen evaluator output misses a finite {key!r} leaf "
                 f"({leaf}); got {node!r} — values are never fabricated")
        out[key] = float(node)
    return out


def build_metrics_record(*, run_id: str, candidate_id: str, seed: int,
                         metrics: Mapping[str, Any],
                         inner_val_total_loss: Optional[float],
                         core_time_s: Optional[float],
                         outcome: str,
                         identity_sha256: Optional[str],
                         provenance: Mapping[str, Any]) -> Dict[str, Any]:
    """One append-only metrics-table record (identity + provenance)."""
    _require(run_id and candidate_id, "V542_STATS_RECORD_INVALID",
             "run_id/candidate_id must be non-empty")
    vals: Dict[str, float] = {}
    for key in METRIC_KEYS:
        value = (metrics or {}).get(key)
        _require(_finite(value), "V542_STATS_METRIC_NON_FINITE",
                 f"metric {key!r} of {run_id} is missing/non-finite "
                 f"(got {value!r})")
        vals[key] = float(value)
    _require(inner_val_total_loss is None or _finite(inner_val_total_loss),
             "V542_STATS_METRIC_NON_FINITE",
             f"inner_val_total_loss of {run_id} is non-finite "
             f"(got {inner_val_total_loss!r})")
    _require(core_time_s is None or _finite(core_time_s),
             "V542_STATS_METRIC_NON_FINITE",
             f"core_time_s of {run_id} is non-finite (got {core_time_s!r})")
    record: Dict[str, Any] = {
        "schema": METRICS_ENTRY_SCHEMA,
        "run_id": str(run_id),
        "candidate_id": str(candidate_id),
        "seed": int(seed),
        "outcome": str(outcome),
        "metrics": vals,
        "inner_val_total_loss": (None if inner_val_total_loss is None
                                 else float(inner_val_total_loss)),
        "core_time_s": (None if core_time_s is None
                        else float(core_time_s)),
        "identity_sha256": identity_sha256,
        "provenance": dict(provenance),
        "torch_compile": False,
        "written_at": _now(),
    }
    record["entry_sha256"] = _canonical_sha256(
        {k: v for k, v in record.items() if k != "entry_sha256"})
    return record


def load_metrics_table(path: Any) -> List[Dict[str, Any]]:
    """All metrics-table records (missing file ⇒ ``[]``)."""
    p = Path(path)
    if not p.is_file():
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V542StatsError(
            "V542_STATS_TABLE_INVALID",
            f"metrics table unreadable: {p}: {exc}") from exc
    _require(isinstance(doc, Mapping)
             and str(doc.get("schema")) == METRICS_TABLE_SCHEMA
             and isinstance(doc.get("records"), list),
             "V542_STATS_TABLE_INVALID",
             f"metrics table at {p} is not a {METRICS_TABLE_SCHEMA} "
             "document with a records list")
    return [dict(r) for r in doc["records"] if isinstance(r, Mapping)]


def metrics_record_fingerprint(record: Mapping[str, Any]) -> str:
    """Content fingerprint of a metrics record (identity + payload).

    ``written_at``/``entry_sha256`` are bookkeeping — the fingerprint
    covers the run identity AND the evaluated content, so a re-extract
    of the same checkpoint is idempotent while a genuinely new identity
    or a new evaluation payload appends (last wins on load).
    """
    payload = {k: v for k, v in dict(record).items()
               if k not in ("written_at", "entry_sha256")}
    return _canonical_sha256(payload)


def append_metrics_records(path: Any,
                           records: Sequence[Mapping[str, Any]]
                           ) -> Dict[str, Any]:
    """APPEND metrics records (append-only; idempotent; never rewrite).

    A record whose content fingerprint (run identity + evaluated
    payload) is already present is skipped — re-running ``extract``
    over unchanged checkpoints never duplicates rows.  A record for a
    run with a NEW identity (e.g. a clean rerun after an identity bump)
    or a genuinely new payload is appended and supersedes by load order
    — exactly the ledger's append-only last-wins discipline.  Existing
    records are never modified or removed.
    """
    p = Path(path)
    existing = load_metrics_table(p)
    seen = {metrics_record_fingerprint(r) for r in existing}
    appended: List[str] = []
    skipped: List[str] = []
    for raw in records:
        _require(isinstance(raw, Mapping), "V542_STATS_RECORD_INVALID",
                 "metrics records must be mappings")
        record = dict(raw)
        _require(str(record.get("schema")) == METRICS_ENTRY_SCHEMA
                 and record.get("entry_sha256"),
                 "V542_STATS_RECORD_INVALID",
                 "metrics records must be build_metrics_record outputs "
                 "(schema + entry_sha256)")
        fingerprint = metrics_record_fingerprint(record)
        if fingerprint in seen:
            skipped.append(str(record.get("run_id")))
            continue
        existing.append(record)
        seen.add(fingerprint)
        appended.append(str(record.get("run_id")))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({
            "schema": METRICS_TABLE_SCHEMA,
            "append_only": True,
            "last_wins_on_load": True,
            "records": existing,
        }, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return {"appended": appended, "skipped_existing": skipped,
            "total_records": len(existing), "path": str(p)}


def latest_metrics_by_run(records: Sequence[Mapping[str, Any]]
                          ) -> Dict[str, Dict[str, Any]]:
    """Latest record per run_id (append-only table; last wins)."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in records:
        run_id = str(row.get("run_id"))
        if run_id:
            out[run_id] = dict(row)
    return out


def evaluate_run_endpoints(cfg: Mapping[str, Any], campaign_root: Any,
                           run_id: str, *, device: str = "cpu"
                           ) -> Dict[str, Any]:
    """Evaluate ONE finished formal checkpoint — the sanity-band way.

    Identical machinery to ``v542_formal.evaluate_a0_band_endpoints``,
    generalized to any formal candidate: ``runs/<run_id>/checkpoint_
    best.pt`` strict-loaded into the freshly assembled model (frozen
    construction order), evaluated by the FROZEN ``evaluate_v53mp`` over
    the seed's inner-val positions with ``tau_end=None``.  No torch.compile.
    """
    import torch

    from pes2ts_core.pesn.models.v542.assembly import assemble_v542_model
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        V53MPTrainerConfig,
        _CacheView,
        _FeatureLRU,
        _train_q_mean,
        evaluate_v53mp,
    )
    from pes2ts_core.pesn.offline.training.v542_train import V542_TRAIN_SCHEMA
    from scripts.agent_ops.v542_campaign import load_campaign_env
    from scripts.agent_ops.v542_formal import (
        _band_inner_val_positions,
        _band_train_positions,
    )

    state_rows = latest_ledger_state(read_ledger(campaign_root))
    row = state_rows.get(str(run_id)) or {}
    _require(str(row.get("outcome")) == "done",
             "V542_STATS_RUN_INCOMPLETE",
             f"{run_id} ledger outcome is {row.get('outcome')!r}, not "
             "'done' — only finished runs are extracted")
    candidate_id = str(row.get("backbone"))
    seed = int(row.get("seed"))
    run_dir = Path(campaign_root) / "runs" / run_id
    ckpt = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    _require(ckpt.is_file() and metrics_path.is_file(),
             "V542_STATS_RUN_INCOMPLETE",
             f"{run_id} has no complete artifacts under {run_dir} "
             "(need checkpoint_best.pt + metrics.json)")
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _require(run_metrics.get("run_outcome") == "done",
             "V542_STATS_RUN_INCOMPLETE",
             f"{run_id} metrics.json outcome is "
             f"{run_metrics.get('run_outcome')!r}, not 'done'")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    _require(isinstance(state, Mapping)
             and str(state.get("schema")) == V542_TRAIN_SCHEMA
             and str(state.get("candidate_id")) == candidate_id
             and int((state.get("identity") or {}).get("seed", -1)) == seed,
             "V542_STATS_RUN_INCOMPLETE",
             f"{ckpt} is not the seed-{seed} {candidate_id} formal "
             f"checkpoint (schema={state.get('schema')!r} "
             f"candidate={state.get('candidate_id')!r})")

    env = load_campaign_env(cfg)
    assembled = assemble_v542_model(
        candidate_id, seed, device=str(device), require_identity=False)
    assembled.model.load_state_dict(state["model"], strict=True)
    assembled.model.train(False)

    positions = _band_inner_val_positions(cfg, env.ds, seed)
    lru = _FeatureLRU(env.ds, capacity=len(positions) + 64)
    cache = _CacheView(lru)
    train_pos, _val = _band_train_positions(cfg, env.ds, seed)
    eval_cfg = V53MPTrainerConfig(arm="A", seed=seed, device=str(device))
    report = evaluate_v53mp(
        assembled.model, None, env.ds, env.labels, env.endpoint, cache,
        positions, eval_cfg,
        train_q_mean=_train_q_mean(env.ds.samples, env.labels, train_pos),
        tau_end=None)
    metrics = metric_values_from_eval(report)
    checkpoint_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    provenance = {
        "evaluator": dict(EVALUATOR_IDENTITY),
        "metric_leaves": dict(METRIC_LEAVES),
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": checkpoint_sha,
        "n_inner_val_positions": len(positions),
        "tau_end": None,
        "ledger_seq": row.get("seq"),
        "ledger_identity_sha256": row.get("identity_sha256"),
        "inner_val_total_loss": row.get("final_inner_val_loss"),
        "core_time_s": row.get("core_s"),
        "eval_scope": "inner_val_only",
    }
    return {"metrics": metrics, "provenance": provenance,
            "candidate_id": candidate_id, "seed": seed,
            "outcome": str(row.get("outcome")),
            "identity_sha256": row.get("identity_sha256"),
            "inner_val_total_loss": row.get("final_inner_val_loss"),
            "core_time_s": row.get("core_s")}


def run_extract(cfg: Mapping[str, Any], *, campaign_root: Any,
                device: str = "cpu",
                only: Optional[Sequence[str]] = None,
                evaluator: Any = None,
                ledger_rows: Optional[Sequence[Mapping[str, Any]]] = None,
                table_path: Any = None) -> Dict[str, Any]:
    """Extract every finished formal run into the append-only table.

    ``evaluator`` is a TEST seam: ``evaluator(run_id) -> {"metrics",
    "provenance", ...}``; the production path is
    :func:`evaluate_run_endpoints` (frozen evaluator).  Ledger and table
    I/O are real in both cases (tests point them at a tmp campaign root).
    """
    rows = (list(ledger_rows) if ledger_rows is not None
            else read_ledger(campaign_root))
    state = latest_ledger_state(rows)
    path = Path(table_path) if table_path is not None else (
        Path(campaign_root) / METRICS_TABLE_REL)
    wanted = [str(r) for r in only] if only else None
    planned = [run_id for run_id, row in sorted(state.items())
               if str(row.get("stage")) in EXTRACT_STAGES
               and str(row.get("outcome")) == "done"
               and (wanted is None or run_id in wanted)]
    if wanted is not None:
        unknown = [r for r in wanted if r not in state]
        _require(not unknown, "V542_STATS_RUN_UNKNOWN",
                 f"run id(s) {unknown} are not rows of the ledger")
    records: List[Dict[str, Any]] = []
    for run_id in planned:
        got = (evaluate_run_endpoints(cfg, campaign_root, run_id,
                                      device=device)
               if evaluator is None else evaluator(run_id))
        records.append(build_metrics_record(
            run_id=run_id, candidate_id=str(got["candidate_id"]),
            seed=int(got["seed"]), metrics=got["metrics"],
            inner_val_total_loss=got.get("inner_val_total_loss"),
            core_time_s=got.get("core_time_s"),
            outcome=str(got.get("outcome", "done")),
            identity_sha256=got.get("identity_sha256"),
            provenance=got.get("provenance") or {}))
    result = append_metrics_records(path, records)
    return {"schema": METRICS_TABLE_SCHEMA, "kind": "extract",
            "training_performed": False,
            "torch_compile": False,
            "planned_runs": planned, **result,
            "records": records}


# ===========================================================================
# 2. h2 — frozen guards vs matched A0 AND R3 (report-only disposition)
# ===========================================================================
def h2_side_guards(run_metrics: Mapping[str, Any],
                   ref_metrics: Mapping[str, Any],
                   thresholds: Optional[Mapping[str, Any]] = None
                   ) -> Dict[str, Dict[str, Any]]:
    """The four frozen H2 guards on ONE reference side.

    RMSE/zero relative degradation ≤ 2%; direction (sign) absolute drop
    ≤ 0.01; dead-fraction absolute increase ≤ 0.01; fixed-coverage
    direction relative degradation ≤ 2%.  All comparisons are ``<=``
    (a value exactly at the limit PASSES — non-inferiority semantics).
    """
    th = dict(H2_NO_BREACH_THRESHOLDS if thresholds is None else thresholds)

    def _val(key: str, side: str) -> float:
        value = (run_metrics if side == "run" else ref_metrics).get(key)
        _require(_finite(value), "V542_STATS_METRIC_NON_FINITE",
                 f"H2 {key!r} is missing/non-finite on the {side} side "
                 f"(got {value!r})")
        return float(value)

    out: Dict[str, Dict[str, Any]] = {}

    run_rmse, ref_rmse = _val("rmse_zero", "run"), _val("rmse_zero", "ref")
    _require(ref_rmse > 0.0, "V542_STATS_REFERENCE_INVALID",
             f"reference rmse_zero must be > 0 for the relative guard "
             f"(got {ref_rmse!r})")
    rmse_deg = (run_rmse - ref_rmse) / ref_rmse
    limit_rmse = float(th["rmse_zero_rel_degradation_max"])
    out["rmse_zero_rel_degradation"] = {
        "run": run_rmse, "reference": ref_rmse,
        "degradation_relative": rmse_deg, "limit": limit_rmse,
        "pass": bool(rmse_deg <= limit_rmse),
        "rule": "rmse_zero degradation <= 2% relative (degradation word "
                "is frozen — H2 is a non-inferiority guard)"}

    run_sign, ref_sign = _val("direction", "run"), _val("direction", "ref")
    sign_drop = ref_sign - run_sign
    limit_sign = float(th["sign_abs_drop_max"])
    out["sign_abs_drop"] = {
        "run": run_sign, "reference": ref_sign,
        "absolute_drop": sign_drop, "limit": limit_sign,
        "pass": bool(sign_drop <= limit_sign),
        "rule": "direction (H2 sign accuracy) absolute drop <= 0.01"}

    run_dead, ref_dead = _val("dead_fraction", "run"), _val(
        "dead_fraction", "ref")
    dead_inc = run_dead - ref_dead
    limit_dead = float(th["dead_abs_increase_max"])
    out["dead_fraction_abs_increase"] = {
        "run": run_dead, "reference": ref_dead,
        "absolute_increase": dead_inc, "limit": limit_dead,
        "pass": bool(dead_inc <= limit_dead),
        "rule": "dead fraction absolute increase <= 0.01"}

    _require(ref_sign > 0.0, "V542_STATS_REFERENCE_INVALID",
             f"reference direction must be > 0 for the relative guard "
             f"(got {ref_sign!r})")
    dir_deg = (ref_sign - run_sign) / ref_sign
    limit_dir = float(th["fixed_coverage_rel_degradation_max"])
    out["fixed_coverage_direction_rel_degradation"] = {
        "run": run_sign, "reference": ref_sign,
        "degradation_relative": dir_deg, "limit": limit_dir,
        "pass": bool(dir_deg <= limit_dir),
        "rule": "fixed-coverage direction (the frozen evaluator's sign "
                "accuracy leaf) degradation <= 2% relative"}
    return out


def compute_h2(run_metrics: Mapping[str, Any],
               a0_metrics: Mapping[str, Any],
               r3_metrics: Mapping[str, Any],
               thresholds: Optional[Mapping[str, Any]] = None
               ) -> Dict[str, Any]:
    """Frozen H2 for ONE formal run: guards vs matched A0 AND vs R3.

    Both reference sides are MANDATORY (config ``r3_guard_block``: a
    missing R3 side is a typed refusal — never a silent downgrade to a
    matched-A0-only comparison).  The run passes iff EVERY guard on
    BOTH sides passes; any breach maps the run to
    ``scientific_failure`` (the report's disposition — the run ledger
    is never rewritten by this lane).
    """
    _require(a0_metrics is not None, "V542_STATS_MATCHED_A0_MISSING",
             "the same-seed matched A0 metrics are required for H2")
    _require(r3_metrics is not None, "V542_STATS_R3_REFERENCE_MISSING",
             "the R3 reference metrics are required for H2 "
             "(r3_guard_block: T1 stays blocked rather than degrading to "
             "a matched-A0-only comparison)")
    sides = {
        "matched_a0": h2_side_guards(run_metrics, a0_metrics, thresholds),
        "r3": h2_side_guards(run_metrics, r3_metrics, thresholds),
    }
    breached: List[str] = []
    for side_name, guards in sides.items():
        for guard_name, guard in guards.items():
            if not guard["pass"]:
                breached.append(f"{side_name}.{guard_name}")
    return {"verdict": "pass" if not breached else "scientific_failure",
            "h2_no_breach": not breached,
            "breached_guards": breached,
            "sides": sides}


def build_h2_report(table_by_run: Mapping[str, Mapping[str, Any]],
                    r3_by_seed: Mapping[Any, Mapping[str, Any]],
                    *,
                    thresholds: Optional[Mapping[str, Any]] = None,
                    r3_provenance: Optional[Mapping[str, Any]] = None
                    ) -> Dict[str, Any]:
    """The H2 report over every finished formal candidate run.

    ``table_by_run`` is the latest-record-per-run metrics table (see
    :func:`latest_metrics_by_run`); ``r3_by_seed`` maps seed -> the five
    R3 reference metrics (located, never fabricated).  A0 rows are
    included as the matched control (their matched-A0 side is trivially
    zero — recorded explicitly, never hidden).
    """
    th = dict(H2_NO_BREACH_THRESHOLDS if thresholds is None else thresholds)
    runs: Dict[str, Dict[str, Any]] = {}
    for run_id in sorted(table_by_run):
        entry = table_by_run[run_id]
        candidate_id = str(entry.get("candidate_id"))
        seed = int(entry.get("seed"))
        if candidate_id not in ("A0", "A1", "A2", "A3", "A4", "A5"):
            continue                            # diagnostic variants: not here
        if str(entry.get("outcome")) != "done":
            continue
        a0_entry = table_by_run.get(f"V542_A0_S{seed}")
        _require(a0_entry is not None,
                 "V542_STATS_MATCHED_A0_MISSING",
                 f"matched A0_S{seed} metrics are missing from the table "
                 f"(required before H2 of {run_id})")
        r3_entry = r3_by_seed.get(seed) or r3_by_seed.get(str(seed))
        _require(r3_entry is not None, "V542_STATS_R3_REFERENCE_MISSING",
                 f"R3 reference metrics for seed {seed} are missing "
                 "(located from plan/r3_reference_metrics.json)")
        result = compute_h2(entry["metrics"], a0_entry["metrics"],
                            r3_entry, thresholds=th)
        runs[run_id] = {
            "candidate_id": candidate_id,
            "seed": seed,
            "metrics": dict(entry["metrics"]),
            "matched_a0_run": f"V542_A0_S{seed}",
            "sides": result["sides"],
            "breached_guards": result["breached_guards"],
            "verdict": result["verdict"],
            "h2_no_breach": result["h2_no_breach"],
            "ledger_unchanged": True,
            "identity_sha256": entry.get("identity_sha256"),
        }
    return {
        "schema": H2_REPORT_SCHEMA,
        "kind": "h2",
        "gate": "H2",
        "rule": ("each formal seed: RMSE/zero degradation <=2% relative "
                 "R3 AND <=2% relative the same-seed matched A0; sign "
                 "absolute drop <=0.01; dead absolute increase <=0.01; "
                 "fixed-coverage direction degradation <=2%; on_fail = "
                 "scientific_failure (report disposition only — the run "
                 "ledger is NEVER rewritten by this lane)"),
        "thresholds": th,
        "evaluator_identity": dict(EVALUATOR_IDENTITY),
        "r3_reference_provenance": dict(r3_provenance or {}),
        "runs": runs,
        "written_at": _now(),
    }


def run_h2(cfg: Mapping[str, Any], *, campaign_root: Any,
           r3_metrics: Optional[Mapping[str, Any]] = None,
           table_path: Any = None) -> Dict[str, Any]:
    """Locate the R3 reference + the metrics table, compute the report."""
    path = Path(table_path) if table_path is not None else (
        Path(campaign_root) / METRICS_TABLE_REL)
    table = latest_metrics_by_run(load_metrics_table(path))
    if r3_metrics is None:
        doc = _load_json(Path(campaign_root) / R3_REFERENCE_REL,
                         "V542_STATS_R3_REFERENCE_MISSING",
                         "the R3 reference metrics record")
        _require(str(doc.get("schema")) == R3_REFERENCE_SCHEMA
                 and isinstance(doc.get("by_seed"), Mapping),
                 "V542_STATS_R3_REFERENCE_INVALID",
                 f"{R3_REFERENCE_REL} is not a {R3_REFERENCE_SCHEMA} "
                 "document with a by_seed mapping")
        r3_metrics = doc
    by_seed = {int(k): v for k, v in dict(
        r3_metrics.get("by_seed") or r3_metrics).items()}
    report = build_h2_report(
        table, by_seed,
        r3_provenance=r3_metrics.get("provenance") or {})
    report_path = _write_report(campaign_root, H2_REPORT_REL, report)
    report["report_path"] = str(report_path)
    return report


# ===========================================================================
# 3. route_logic (frozen parenthesized logic) + seed17 screen
# ===========================================================================
def _route_metrics(cand: Mapping[str, Any],
                   a0: Mapping[str, Any]) -> Dict[str, float]:
    """Point-effect quantities of one seed (or seed-mean) comparison."""

    def _m(table: Mapping[str, Any], key: str) -> float:
        value = table.get(key)
        _require(_finite(value), "V542_STATS_METRIC_NON_FINITE",
                 f"metric {key!r} is missing/non-finite (got {value!r})")
        return float(value)

    d_dir = _m(cand, "direction") - _m(a0, "direction")
    d_rmse = _m(cand, "rmse_zero") - _m(a0, "rmse_zero")
    a0_dir = _m(a0, "direction")
    a0_rmse = _m(a0, "rmse_zero")
    _require(a0_dir > 0.0, "V542_STATS_REFERENCE_INVALID",
             f"A0 direction must be > 0 for relative effects "
             f"(got {a0_dir!r})")
    _require(a0_rmse > 0.0, "V542_STATS_REFERENCE_INVALID",
             f"A0 rmse_zero must be > 0 for relative effects "
             f"(got {a0_rmse!r})")
    return {
        "direction_rel": d_dir / a0_dir,
        "direction_abs": d_dir,
        "rmse_rel_improve": -d_rmse / a0_rmse,
        "delta_auprc": _m(cand, "endpoint_auprc") - _m(a0, "endpoint_auprc"),
        "delta_q": _m(cand, "q_corr") - _m(a0, "q_corr"),
    }


def evaluate_move_route(cand: Mapping[str, Any],
                        a0: Mapping[str, Any]) -> Dict[str, Any]:
    """Frozen MOVE route: ``(direction_pass AND rmse_non_inferior) OR
    (rmse_pass AND direction_non_inferior)``.

    The non-inferiority conditions constrain ONLY their own branch —
    both branches close their own bracket (dangling-OR forbidden).
    """
    q = _route_metrics(cand, a0)
    direction_pass = (q["direction_rel"] >= DIRECTION_REL_MIN
                      and q["direction_abs"] >= DIRECTION_ABS_MIN)
    rmse_pass = q["rmse_rel_improve"] >= RMSE_REL_MIN
    rmse_non_inferior = q["rmse_rel_improve"] >= -0.02
    direction_non_inferior = q["direction_rel"] >= -0.02
    passed = ((direction_pass and rmse_non_inferior)
              or (rmse_pass and direction_non_inferior))
    return {"pass_logic": MOVE_PASS_LOGIC, "pass": bool(passed),
            "components": {
                "direction_pass": bool(direction_pass),
                "rmse_pass": bool(rmse_pass),
                "rmse_non_inferior": bool(rmse_non_inferior),
                "direction_non_inferior": bool(direction_non_inferior),
            },
            "quantities": q}


def evaluate_process_route(cand: Mapping[str, Any],
                           a0: Mapping[str, Any]) -> Dict[str, Any]:
    """Frozen PROCESS route: ``(auprc_pass AND q_non_inferior) OR
    (q_pass AND auprc_non_inferior)`` — no dangling OR."""
    q = _route_metrics(cand, a0)
    auprc_pass = q["delta_auprc"] >= AUPRC_DELTA_MIN
    q_pass = q["delta_q"] >= Q_DELTA_MIN
    auprc_non_inferior = q["delta_auprc"] >= -AUPRC_DROP_MAX
    q_non_inferior = q["delta_q"] >= -Q_DROP_MAX
    passed = ((auprc_pass and q_non_inferior)
              or (q_pass and auprc_non_inferior))
    return {"pass_logic": PROCESS_PASS_LOGIC, "pass": bool(passed),
            "components": {
                "auprc_pass": bool(auprc_pass),
                "q_pass": bool(q_pass),
                "auprc_non_inferior": bool(auprc_non_inferior),
                "q_non_inferior": bool(q_non_inferior),
            },
            "quantities": q}


def compute_screen_score(cand: Mapping[str, Any],
                         a0: Mapping[str, Any]) -> Dict[str, Any]:
    """The frozen screen score (selection only, never reporting).

    ``max(direction_rel/0.03, rmse_rel_improve/0.03, delta_auprc/0.02,
    delta_q/0.03)``.
    """
    q = _route_metrics(cand, a0)
    terms = {name: q[name] / norm
             for name, norm in SCREEN_SCORE_NORMALIZERS.items()}
    return {"score": float(max(terms.values())), "terms": terms,
            "formula": ("max(direction_rel/0.03, rmse_rel_improve/0.03, "
                        "delta_auprc/0.02, delta_q/0.03)")}


def screen_candidate(candidate_id: str, *,
                     run_metrics: Optional[Mapping[str, Any]],
                     a0_metrics: Optional[Mapping[str, Any]],
                     run_done: bool,
                     matched_a0_done: bool,
                     h2_no_breach: bool,
                     gates: Mapping[str, Any],
                     inner_val_total_loss: Optional[float],
                     core_time_s: Optional[float]) -> Dict[str, Any]:
    """The frozen ``seed17_screen_pass`` predicate for ONE A1–A4 run.

    This is the seed17 SELECTION screen — NOT a three-seed H3 claim.
    """
    reasons: List[str] = []
    if not run_done:
        reasons.append("run_not_done")
    metrics_finite = bool(run_metrics) and all(
        _finite((run_metrics or {}).get(k)) for k in METRIC_KEYS)
    if not metrics_finite:
        reasons.append("metrics_non_finite")
    if not a0_metrics or not all(_finite(a0_metrics.get(k))
                                 for k in METRIC_KEYS):
        reasons.append("matched_a0_metrics_non_finite")
        matched_a0_done = False
    if not matched_a0_done:
        reasons.append("matched_a0_s17_incomplete")
    for gate_name in ("H0_legality", "H1_resource", "diagnostic_gate",
                      "cost_gate"):
        if not bool((gates or {}).get(gate_name)):
            reasons.append(f"gate_failed:{gate_name}")
    if not h2_no_breach:
        reasons.append("seed17_h2_breach")
    move = process = None
    score: Optional[Dict[str, Any]] = None
    if metrics_finite and matched_a0_done:
        move = evaluate_move_route(run_metrics, a0_metrics)
        process = evaluate_process_route(run_metrics, a0_metrics)
        score = compute_screen_score(run_metrics, a0_metrics)
        if not (move["pass"] or process["pass"]):
            reasons.append("no_route_point_effect")
    else:
        reasons.append("no_route_point_effect")
    eligible = not reasons
    return {
        "candidate_id": str(candidate_id),
        "eligible": bool(eligible),
        "ineligible_reasons": reasons,
        "run_done": bool(run_done),
        "metrics_finite": bool(metrics_finite),
        "matched_a0_done": bool(matched_a0_done),
        "h2_no_breach": bool(h2_no_breach),
        "gates": {k: bool((gates or {}).get(k)) for k in
                  ("H0_legality", "H1_resource", "diagnostic_gate",
                   "cost_gate")},
        "gate_detail": {k: v for k, v in (gates or {}).items()
                        if k not in ("H0_legality", "H1_resource",
                                     "diagnostic_gate", "cost_gate")},
        "routes": {"move": move, "process": process},
        "screen_score": score,
        "inner_val_total_loss": (None if inner_val_total_loss is None
                                 else float(inner_val_total_loss)),
        "core_time_s": (None if core_time_s is None
                        else float(core_time_s)),
        "scope": ("seed17 selection screen only — NOT a three-seed H3 "
                  "declaration"),
    }


def rank_and_select_top2(screened: Sequence[Mapping[str, Any]]
                         ) -> Dict[str, Any]:
    """Rank eligible candidates by frozen screen score; take ≤ 2.

    Tie-break (frozen): inner-val total loss, then core time, then
    candidate id lexicographic.  The top2 set is NEVER padded — no
    substitutes, fewer than 2 is a legal outcome.
    """
    eligible = [dict(c) for c in screened if c.get("eligible")]
    def _key(c: Mapping[str, Any]) -> Any:
        loss = c.get("inner_val_total_loss")
        core = c.get("core_time_s")
        return (-float(c["screen_score"]["score"]),
                float(loss) if _finite(loss) else float("inf"),
                float(core) if _finite(core) else float("inf"),
                str(c["candidate_id"]))
    ranked = sorted(eligible, key=_key)
    top2 = ranked[:2]
    return {
        "ranked_eligible": [str(c["candidate_id"]) for c in ranked],
        "top2": [str(c["candidate_id"]) for c in top2],
        "top2_count": len(top2),
        "top2_max_candidates": 2,
        "no_substitutes": True,
        "not_selected": [str(c["candidate_id"]) for c in ranked[2:]],
        "tie_break": ["inner_val_total_loss", "core_time",
                      "candidate_id_lexicographic"],
    }


def require_top2_freeze(a0_s17_done: bool, band_verdict: Optional[str]
                        ) -> None:
    """The frozen top2 freeze precondition (asserted, fail-closed)."""
    _require(bool(a0_s17_done), "V542_STATS_TOP2_FREEZE_A0_INCOMPLETE",
             "top2 freeze requires V542_A0_S17 complete (done) — the "
             "matched H2/H3 control must exist first")
    _require(band_verdict == "pass", "V542_STATS_TOP2_FREEZE_BAND",
             f"top2 freeze requires the {A0_SANITY_BAND_GATE} record "
             f"with verdict=pass (got {band_verdict!r})")


def run_screen(cfg: Mapping[str, Any], *, campaign_root: Any,
               table_path: Any = None,
               h2_report: Optional[Mapping[str, Any]] = None,
               band: Optional[Mapping[str, Any]] = None,
               gate_detail_by_candidate: Optional[Mapping[str, Any]] = None
               ) -> Dict[str, Any]:
    """The seed17 screen + top2 selection over A1–A4 (no training).

    Production wiring: ledger (run done), metrics table (finite
    metrics), the H2 report (seed17 guards), the WP4 diagnostic
    disposition, the signed cost gate and the sanity-band record.
    ``gate_detail_by_candidate`` is a test seam overriding the H0/H1/
    diagnostic/cost gate booleans.
    """
    path = Path(table_path) if table_path is not None else (
        Path(campaign_root) / METRICS_TABLE_REL)
    table = latest_metrics_by_run(load_metrics_table(path))
    state = latest_ledger_state(read_ledger(campaign_root))
    if h2_report is None:
        h2_report = _load_json(Path(campaign_root) / H2_REPORT_REL,
                               "V542_STATS_H2_REPORT_MISSING",
                               "the H2 report")
    if band is None:
        band = read_band_record(campaign_root)
    a0_s17_row = state.get("V542_A0_S17") or {}
    require_top2_freeze(str(a0_s17_row.get("outcome")) == "done",
                        None if band is None else band.get("verdict"))

    a0_entry = table.get("V542_A0_S17")
    _require(a0_entry is not None, "V542_STATS_TABLE_A0_S17_MISSING",
             "V542_A0_S17 has no metrics-table entry — run `extract` "
             "after T0 completes")
    screened: List[Dict[str, Any]] = []
    a4_report: Optional[Dict[str, Any]] = None
    for candidate_id in TOP2_CANDIDATES:
        run_id = f"V542_{candidate_id}_S17"
        entry = table.get(run_id)
        row = state.get(run_id) or {}
        h2_entry = ((h2_report or {}).get("runs") or {}).get(run_id) or {}
        if candidate_id == "A4":
            # WP4 binding ruling: A4 never runs full-dose; ALWAYS
            # reported ineligible with the frozen diagnostic reason.
            a4_report = {
                "candidate_id": "A4",
                "run_id": run_id,
                "eligible": False,
                "status": "gate_skipped",
                "reason_code": A4_DIAG_REASON,
                "ruling": ("ADR-0033 v1.3.3 WP4 diagnostic ruling "
                           "(3.849742% < 5.0% frozen threshold, binding); "
                           "S29/S43 not activated; no substitutes"),
                "frozen_threshold_pct": 5.0,
                "measured_drop_pct": 3.849742,
                "ineligible_reasons": ["gate_failed:diagnostic_gate"],
            }
            screened.append(a4_report)
            continue
        gates: Dict[str, Any] = {
            # a done formal run implies H0/H1 passed (launch gates);
            # failures surface as invalid_run / resource_failure, never done.
            "H0_legality": str(row.get("outcome")) == "done",
            "H1_resource": str(row.get("outcome")) == "done",
        }
        diag_reason: Optional[str] = None
        if str(row.get("outcome")) == "done":
            diag = diagnostic_disposition(campaign_root, candidate_id)
            gates["diagnostic_gate"] = (
                diag["disposition"] == "advance_full_dose")
            diag_reason = (diag["reason_code"] or diag["disposition"])
            cost = cost_gate_ratio(cfg, candidate_id)
            gates["cost_gate"] = bool(cost["within"])
            gates["core_time_ratio"] = cost["ratio"]
        else:
            gates["diagnostic_gate"] = False
            gates["cost_gate"] = False
        if gate_detail_by_candidate and candidate_id in gate_detail_by_candidate:
            gates.update(dict(gate_detail_by_candidate[candidate_id]))
        screened.append(screen_candidate(
            candidate_id,
            run_metrics=(entry or {}).get("metrics"),
            a0_metrics=a0_entry["metrics"],
            run_done=str(row.get("outcome")) == "done",
            matched_a0_done=True,
            h2_no_breach=bool(h2_entry.get("h2_no_breach")),
            gates=gates,
            inner_val_total_loss=(entry or {}).get("inner_val_total_loss"),
            core_time_s=(entry or {}).get("core_time_s")))
        if diag_reason and not gates.get("diagnostic_gate"):
            screened[-1]["diagnostic_reason_code"] = diag_reason

    ranking = rank_and_select_top2(screened)
    top2_runs = [f"V542_{cid}_S{seed}" for cid in ranking["top2"]
                 for seed in TOP2_SEEDS]
    report = {
        "schema": SCREEN_REPORT_SCHEMA,
        "kind": "seed17_screen",
        "predicate": "evaluation_contract.seed17_screen_pass_predicate",
        "predicate_scope": ("top2 eligibility + A5 supplement activation "
                            "only; NOT a three-seed H3 declaration"),
        "route_logic": {"move": MOVE_PASS_LOGIC,
                        "process": PROCESS_PASS_LOGIC,
                        "seed17_screen_reuses_same_parenthesized_logic":
                            True,
                        "dangling_or_condition_forbidden": True},
        "screen_score_formula": (
            "max(direction_rel/0.03, rmse_rel_improve/0.03, "
            "delta_auprc/0.02, delta_q/0.03)"),
        "top2_freeze_precondition": {
            "a0_s17_done": True,
            "sanity_band_gate": A0_SANITY_BAND_GATE,
            "sanity_band_verdict": (band or {}).get("verdict"),
            "asserted": True,
        },
        "screened": screened,
        **ranking,
        "top2_runs": top2_runs,
        "a4_ineligible": a4_report,
        "training_performed": False,
        "written_at": _now(),
    }
    report_path = _write_report(campaign_root, SCREEN_REPORT_REL, report)
    report["report_path"] = str(report_path)
    return report


# ===========================================================================
# 4. h3 — frozen paired cluster bootstrap + Holm + route verdicts
# ===========================================================================
def bootstrap_subseed(root: int, seed: int, endpoint: str) -> int:
    """The frozen deterministic per-(seed, endpoint) sub-seed."""
    _require(endpoint in ENDPOINTS, "V542_STATS_ENDPOINT_UNKNOWN",
             f"endpoint must be one of {list(ENDPOINTS)} "
             f"(got {endpoint!r})")
    digest = hashlib.sha256(
        f"v542|{int(root)}|{int(seed)}|{endpoint}".encode("utf-8")
    ).hexdigest()[:16]
    return int(digest, 16) % 2147483647


def one_sided_improvement_p(improvements: Any) -> float:
    """``p = (1 + #{improvement <= 0}) / (n_draws + 1)`` (frozen).

    The denominator is ``1 + final_draws`` — 10001 for the frozen
    10,000-draw final analysis.
    """
    values = [float(v) for v in improvements]
    _require(bool(values), "V542_STATS_BOOTSTRAP_EMPTY",
             "improvements must be a non-empty sequence")
    nonpositive = sum(1 for v in values if v <= 0.0)
    return (1.0 + nonpositive) / (len(values) + 1.0)


def paired_cluster_bootstrap(per_seed_pairs: Mapping[Any, Mapping[str, Any]],
                             *, endpoint: str,
                             root: int = BOOTSTRAP_SEED_ROOT,
                             draws: int = FINAL_DRAWS) -> Dict[str, Any]:
    """Seed-averaged curve-level PAIRED cluster bootstrap (frozen).

    ``per_seed_pairs[seed] = {"candidate": [...], "a0": [...]}`` —
    per-CLUSTER (curve) metric values on the SAME clusters for both
    models (pairing: same cluster, same frame set; both models share the
    resample indices).  Frame-level resampling is forbidden.  Clusters
    where either side is undefined are dropped on BOTH sides
    (undefined_cluster_rule: report the dropped count, never impute).

    CI (percentile 95%) and the one-sided p come from the SAME draws.
    """
    _require(endpoint in ENDPOINTS, "V542_STATS_ENDPOINT_UNKNOWN",
             f"endpoint must be one of {list(ENDPOINTS)} "
             f"(got {endpoint!r})")
    _require(int(draws) > 0, "V542_STATS_BOOTSTRAP_DRAWS",
             f"draws must be positive (got {draws!r})")
    n_draws = int(draws)
    sign = (1.0 if IMPROVEMENT_DIRECTION[endpoint] == "candidate_minus_a0"
            else -1.0)
    seeds = sorted(int(s) for s in per_seed_pairs)
    _require(bool(seeds), "V542_STATS_BOOTSTRAP_EMPTY",
             "per_seed_pairs must carry at least one seed")
    arrays: Dict[int, Any] = {}
    subseeds: Dict[int, int] = {}
    dropped = 0
    for seed in seeds:
        pair = per_seed_pairs.get(seed) or per_seed_pairs.get(str(seed)) or {}
        cand, a0 = list(pair.get("candidate") or []), list(pair.get("a0") or [])
        _require(len(cand) == len(a0) and len(cand) > 0,
                 "V542_STATS_BOOTSTRAP_PAIRING",
                 f"seed {seed}: candidate/a0 cluster arrays must be "
                 f"non-empty and equal length (got {len(cand)} vs "
                 f"{len(a0)})")
        keep = [i for i in range(len(cand))
                if _finite(cand[i]) and _finite(a0[i])]
        dropped += len(cand) - len(keep)
        _require(bool(keep), "V542_STATS_BOOTSTRAP_EMPTY",
                 f"seed {seed}: no finite paired clusters remain")
        subseeds[seed] = bootstrap_subseed(root, seed, endpoint)
        arrays[seed] = (np.asarray([float(cand[i]) for i in keep],
                                   dtype=np.float64),
                        np.asarray([float(a0[i]) for i in keep],
                                   dtype=np.float64))
    rngs = {seed: np.random.RandomState(subseeds[seed])
            for seed in seeds}
    improvements = np.empty(n_draws, dtype=np.float64)
    for d in range(n_draws):
        per_seed: List[float] = []
        for seed in seeds:
            c, a = arrays[seed]
            idx = rngs[seed].randint(0, len(c), size=len(c))
            per_seed.append(sign * (float(c[idx].mean())
                                    - float(a[idx].mean())))
        improvements[d] = float(np.mean(per_seed))
    low, high = np.percentile(improvements,
                              [CI_LOW_PERCENTILE, CI_HIGH_PERCENTILE])
    digest = hashlib.sha256(
        improvements.astype("<f8").tobytes()).hexdigest()
    return {
        "endpoint": endpoint,
        "improvement_direction": IMPROVEMENT_DIRECTION[endpoint],
        "bootstrap_seed_root": int(root),
        "draws": n_draws,
        "subseed_formula": (
            'int(sha256(f"v542|{root}|{seed}|{endpoint}").hexdigest()[:16], '
            "16) % 2147483647"),
        "subseed_by_seed": {str(s): subseeds[s] for s in seeds},
        "n_clusters_by_seed": {str(s): int(len(arrays[s][0]))
                               for s in seeds},
        "dropped_clusters": int(dropped),
        "ci_percentiles": [CI_LOW_PERCENTILE, CI_HIGH_PERCENTILE],
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": one_sided_improvement_p(improvements),
        "improvements_sha256": digest,
        # kept for tests/audits; the report writer digests + drops it.
        "improvements": improvements.tolist(),
        "ci_and_p_same_draws": True,
    }


def holm_adjust(p_values: Sequence[float], alpha: float) -> Dict[str, Any]:
    """Holm step-down adjustment over one frozen family.

    Adjusted p-values are the running maximum of ``(m - rank) * p`` over
    the ascending order (capped at 1) — monotone and always ≥ the raw
    p.  Rejection is the step-down rule at ``alpha`` (stop at the first
    non-rejection).
    """
    values = [float(v) for v in p_values]
    _require(bool(values), "V542_STATS_HOLM_EMPTY",
             "holm family must carry at least one p-value")
    _require(all(0.0 <= v <= 1.0 for v in values),
             "V542_STATS_HOLM_P_RANGE",
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
            "order": list(order)}


def _endpoint_point_effects(candidate: Mapping[str, Any]
                            ) -> Dict[str, Dict[str, float]]:
    """Seed-averaged point effects + per-seed improvements (4 endpoints)."""
    point = candidate.get("point_metrics") or {}
    seeds = [s for s in FROZEN_SEEDS if str(s) in {str(k) for k in point}]
    _require(len(seeds) == H3_TOTAL_SEEDS, "V542_STATS_H3_SEEDS_INCOMPLETE",
             f"point metrics for all three frozen seeds are required "
             f"(got {sorted(str(k) for k in point)})")
    norm = {str(k): point[k] for k in point}
    out: Dict[str, Dict[str, float]] = {}
    for endpoint, metric in ENDPOINT_TO_METRIC.items():
        cand_mean = float(np.mean([
            float(norm[str(s)]["candidate"][metric]) for s in seeds]))
        a0_mean = float(np.mean([
            float(norm[str(s)]["a0"][metric]) for s in seeds]))
        sign = (1.0 if IMPROVEMENT_DIRECTION[endpoint]
                == "candidate_minus_a0" else -1.0)
        per_seed = {str(s): sign * (
            float(norm[str(s)]["candidate"][metric])
            - float(norm[str(s)]["a0"][metric])) for s in seeds}
        if endpoint == "move_direction":
            _require(a0_mean > 0.0, "V542_STATS_REFERENCE_INVALID",
                     "A0 direction seed-mean must be > 0")
            rel = (cand_mean - a0_mean) / a0_mean
            absolute = cand_mean - a0_mean
        elif endpoint == "move_rmse":
            _require(a0_mean > 0.0, "V542_STATS_REFERENCE_INVALID",
                     "A0 rmse_zero seed-mean must be > 0")
            rel = (a0_mean - cand_mean) / a0_mean
            absolute = a0_mean - cand_mean
        else:
            rel = cand_mean - a0_mean
            absolute = cand_mean - a0_mean
        out[endpoint] = {
            "candidate_seed_mean": cand_mean,
            "a0_seed_mean": a0_mean,
            "relative_effect": float(rel),
            "absolute_effect": float(absolute),
            "per_seed_improvement": per_seed,
            "same_direction_seeds": sum(
                1 for v in per_seed.values() if v > 0.0),
        }
    return out


def _effect_margin(endpoint: str, effects: Mapping[str, Any]) -> float:
    """The frozen effect-margin definition per endpoint."""
    if endpoint == "move_direction":
        return float(min(effects["relative_effect"] / DIRECTION_REL_MIN,
                         effects["absolute_effect"] / DIRECTION_ABS_MIN))
    if endpoint == "move_rmse":
        return float(effects["relative_effect"] / RMSE_REL_MIN)
    if endpoint == "process_auprc":
        return float(effects["absolute_effect"] / AUPRC_DELTA_MIN)
    return float(effects["absolute_effect"] / Q_DELTA_MIN)


def compute_h3(candidate_inputs: Mapping[str, Mapping[str, Any]],
               *, root: int = BOOTSTRAP_SEED_ROOT,
               draws: int = FINAL_DRAWS,
               move_alpha: float = MOVE_EXIT_ALPHA,
               process_alpha: float = PROCESS_FAMILY_ALPHA) -> Dict[str, Any]:
    """The frozen H3 statistics over the eligible candidate set.

    ``candidate_inputs[cid]``::

        {
          "winner_eligible": bool,
          "h2_no_breach_by_seed": {17: bool, 29: bool, 43: bool},
          "point_metrics": {seed: {"candidate": {metric: v},
                                   "a0": {metric: v}}},
          "paired": {endpoint: {seed: {"candidate": [v|None],
                                       "a0": [v|None]}}},
          "core_time_ratio": float,
          "active_params": int,
          "inner_val_total_loss": float,
        }

    Eligibility (frozen): completed three seeds AND per-seed H2 no
    breach AND winner-eligible; A0 and diagnostic variants never enter.
    Holm families (frozen): MOVE direction and MOVE RMSE are two exits
    each at α=0.025, each Holm over the eligible set; PROCESS is ONE
    family over 2 endpoints × eligible candidates at family α=0.025.
    """
    excluded: Dict[str, str] = {}
    eligible_ids: List[str] = []
    for cid, data in candidate_inputs.items():
        _require(cid not in ("A0", "A1-C", "A5-C"),
                 "V542_STATS_H3_EXCLUDED_CANDIDATE",
                 f"{cid} is A0/diagnostic — never H3-eligible")
        if not bool(data.get("winner_eligible")):
            excluded[cid] = "not_winner_eligible"
            continue
        seeds_done = sorted(str(k) for k in (data.get("point_metrics") or {}))
        if seeds_done != [str(s) for s in FROZEN_SEEDS]:
            excluded[cid] = "three_seeds_incomplete"
            continue
        h2_by_seed = data.get("h2_no_breach_by_seed") or {}
        if not all(bool(h2_by_seed.get(s)) for s in FROZEN_SEEDS):
            excluded[cid] = "per_seed_h2_breach"
            continue
        eligible_ids.append(cid)
    eligible_ids.sort()

    # -- point effects ----------------------------------------------------
    effects = {cid: _endpoint_point_effects(candidate_inputs[cid])
               for cid in eligible_ids}

    # -- bootstrap (CI and p from the SAME draws) -------------------------
    boot: Dict[str, Dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        for cid in eligible_ids:
            pairs = ((candidate_inputs[cid].get("paired") or {})
                     .get(endpoint) or {})
            _require(bool(pairs), "V542_STATS_H3_PAIRED_MISSING",
                     f"{cid}/{endpoint}: per-cluster paired values are "
                     "required for the frozen bootstrap")
            boot[(cid, endpoint)] = paired_cluster_bootstrap(
                pairs, endpoint=endpoint, root=root, draws=draws)

    # -- Holm families over the eligible set ------------------------------
    # (an EMPTY eligible set is legal: no family to correct, families = {})
    families: Dict[str, Any] = {}
    adjusted: Dict[str, float] = {}
    for exit_name, endpoint in (("move_direction_exit", "move_direction"),
                                ("move_rmse_exit", "move_rmse")):
        if not eligible_ids:
            continue
        raw = [boot[(cid, endpoint)]["p_value"] for cid in eligible_ids]
        result = holm_adjust(raw, move_alpha)
        families[exit_name] = {
            "endpoint": endpoint, "alpha": float(move_alpha),
            "scope": "eligible_candidate_set_within_exit",
            "p_raw": {cid: raw[i] for i, cid in enumerate(eligible_ids)},
            "p_adjusted": {cid: result["adjusted"][i]
                           for i, cid in enumerate(eligible_ids)},
            "rejected": {cid: result["rejected"][i]
                         for i, cid in enumerate(eligible_ids)},
        }
        for i, cid in enumerate(eligible_ids):
            adjusted[(cid, endpoint)] = result["adjusted"][i]
    raw_process: List[float] = []
    process_keys: List[Any] = []
    for cid in eligible_ids:
        for endpoint in ("process_auprc", "process_q"):
            raw_process.append(boot[(cid, endpoint)]["p_value"])
            process_keys.append((cid, endpoint))
    if eligible_ids:
        result = holm_adjust(raw_process, process_alpha)
        families["process_route"] = {
            "alpha": float(process_alpha),
            "scope": "2_endpoints_x_eligible_candidates",
            "family": "single_process_family",
            "p_raw": {f"{cid}:{ep}": raw_process[i]
                      for i, (cid, ep) in enumerate(process_keys)},
            "p_adjusted": {f"{cid}:{ep}": result["adjusted"][i]
                           for i, (cid, ep) in enumerate(process_keys)},
            "rejected": {f"{cid}:{ep}": result["rejected"][i]
                         for i, (cid, ep) in enumerate(process_keys)},
        }
        for i, (cid, ep) in enumerate(process_keys):
            adjusted[(cid, ep)] = result["adjusted"][i]

    # -- per-candidate endpoint confirmation + frozen route verdicts ------
    candidates: Dict[str, Dict[str, Any]] = {}
    for cid in eligible_ids:
        data = candidate_inputs[cid]
        eff = effects[cid]
        endpoints: Dict[str, Dict[str, Any]] = {}
        for endpoint in ENDPOINTS:
            b = boot[(cid, endpoint)]
            e = eff[endpoint]
            adj_p = adjusted[(cid, endpoint)]
            alpha = (move_alpha if endpoint.startswith("move_")
                     else process_alpha)
            same_dir_ok = (e["same_direction_seeds"]
                           >= H3_MIN_SEEDS_SAME_DIRECTION)
            ci_ok = b["ci_low"] > 0.0
            adj_ok = adj_p < alpha
            endpoints[endpoint] = {
                "point_effect": {k: v for k, v in e.items()
                                 if k != "per_seed_improvement"},
                "per_seed_improvement": e["per_seed_improvement"],
                "same_direction_seeds": e["same_direction_seeds"],
                "same_direction_ok": bool(same_dir_ok),
                "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                "ci_lower_gt_zero": bool(ci_ok),
                "p_value": b["p_value"],
                "p_adjusted": adj_p,
                "alpha": float(alpha),
                "holm_adjusted_below_alpha": bool(adj_ok),
                "improvements_sha256": b["improvements_sha256"],
                "n_draws": b["draws"],
                "effect_margin": _effect_margin(endpoint, e),
            }
        move_c = evaluate_move_route(
            _seed_mean_metrics(data["point_metrics"], "candidate"),
            _seed_mean_metrics(data["point_metrics"], "a0"))
        proc_c = evaluate_process_route(
            _seed_mean_metrics(data["point_metrics"], "candidate"),
            _seed_mean_metrics(data["point_metrics"], "a0"))

        move_pass_components = {
            "move_direction": move_c["components"]["direction_pass"],
            "move_rmse": move_c["components"]["rmse_pass"],
        }
        process_pass_components = {
            "process_auprc": proc_c["components"]["auprc_pass"],
            "process_q": proc_c["components"]["q_pass"],
        }

        def _confirmed(endpoint: str) -> bool:
            ep = endpoints[endpoint]
            pass_component = (
                move_pass_components if endpoint.startswith("move_")
                else process_pass_components)[endpoint]
            return bool(pass_component and ep["same_direction_ok"]
                        and ep["ci_lower_gt_zero"]
                        and ep["holm_adjusted_below_alpha"])

        for endpoint in ENDPOINTS:
            endpoints[endpoint]["pass_component"] = bool(
                (move_pass_components if endpoint.startswith("move_")
                 else process_pass_components)[endpoint])
            endpoints[endpoint]["confirmed"] = _confirmed(endpoint)

        move_qualified = bool(
            (endpoints["move_direction"]["confirmed"]
             and move_c["components"]["rmse_non_inferior"])
            or (endpoints["move_rmse"]["confirmed"]
                and move_c["components"]["direction_non_inferior"]))
        process_qualified = bool(
            (endpoints["process_auprc"]["confirmed"]
             and proc_c["components"]["q_non_inferior"])
            or (endpoints["process_q"]["confirmed"]
                and proc_c["components"]["auprc_non_inferior"]))
        qualifying = [ep for ep in
                      ("move_direction", "move_rmse",
                       "process_auprc", "process_q")
                      if endpoints[ep]["confirmed"] and (
                          (ep.startswith("move_") and move_qualified)
                          or (not ep.startswith("move_")
                              and process_qualified))]
        candidates[cid] = {
            "candidate_id": cid,
            "move_route": {**move_c, "qualified": move_qualified},
            "process_route": {**proc_c, "qualified": process_qualified},
            "move_qualified": move_qualified,
            "process_qualified": process_qualified,
            "h3_qualified": bool(move_qualified or process_qualified),
            "qualifying_confirmatory_endpoints": qualifying,
            "min_holm_adjusted_p_on_qualifying_endpoint": (
                min(adjusted[(cid, ep)] for ep in qualifying)
                if qualifying else None),
            "max_effect_margin_on_qualifying_endpoint": (
                max(endpoints[ep]["effect_margin"] for ep in qualifying)
                if qualifying else None),
            "endpoints": endpoints,
            "core_time_ratio": float(data.get("core_time_ratio")),
            "active_params": int(data.get("active_params")),
            "inner_val_total_loss": float(data.get(
                "inner_val_total_loss")),
            "h2_no_breach_by_seed": {
                str(s): bool((data.get("h2_no_breach_by_seed") or {}).get(s))
                for s in FROZEN_SEEDS},
        }
        if (move_qualified or process_qualified) and not move_qualified:
            candidates[cid]["process_only_note"] = (
                "PROCESS-only promotion — the report must state "
                "explicitly that MOVE did not improve")

    return {
        "schema": H3_REPORT_SCHEMA,
        "kind": "h3",
        "gate": "H3",
        "bootstrap_seed_root": int(root),
        "final_draws": int(draws),
        "ci_and_p_same_draws": True,
        "p_value_rule": {
            "type": "one_sided_improvement_paired_cluster_bootstrap",
            "formula": ("p = (1 + #{seed-averaged paired bootstrap "
                        "improvement <= 0}) / (draws + 1)"),
            "improvement_direction": dict(IMPROVEMENT_DIRECTION),
        },
        "holm_families": families,
        "route_logic": {"move": MOVE_PASS_LOGIC,
                        "process": PROCESS_PASS_LOGIC},
        "alphas": {"move_exits_each": float(move_alpha),
                   "process_family": float(process_alpha)},
        "eligible_candidates": eligible_ids,
        "excluded_candidates": excluded,
        "candidates": candidates,
        "bootstrap_by_endpoint": {
            f"{cid}:{ep}": {k: v for k, v in b.items()
                            if k != "improvements"}
            for (cid, ep), b in sorted(boot.items())},
        "written_at": _now(),
    }


def _seed_mean_metrics(point_metrics: Mapping[str, Any], side: str
                       ) -> Dict[str, float]:
    """Three-seed mean of one side's five metrics (threshold_on: seed
    averaged point estimate)."""
    norm = {str(k): point_metrics[k] for k in point_metrics}
    out: Dict[str, float] = {}
    for metric in METRIC_KEYS:
        out[metric] = float(np.mean([
            float(norm[str(s)][side][metric]) for s in FROZEN_SEEDS]))
    return out


def _load_formal_model(cfg: Mapping[str, Any], campaign_root: Any,
                       run_id: str, *, device: str) -> Any:
    """Strict-load ONE finished formal checkpoint (extract-leg recipe)."""
    import torch

    from pes2ts_core.pesn.models.v542.assembly import assemble_v542_model
    from pes2ts_core.pesn.offline.training.v542_train import V542_TRAIN_SCHEMA

    state_rows = latest_ledger_state(read_ledger(campaign_root))
    row = state_rows.get(str(run_id)) or {}
    _require(str(row.get("outcome")) == "done",
             "V542_STATS_RUN_INCOMPLETE",
             f"{run_id} ledger outcome is {row.get('outcome')!r}, not "
             "'done'")
    candidate_id = str(row.get("backbone"))
    seed = int(row.get("seed"))
    ckpt = Path(campaign_root) / "runs" / run_id / "checkpoint_best.pt"
    _require(ckpt.is_file(), "V542_STATS_RUN_INCOMPLETE",
             f"{run_id} has no checkpoint_best.pt at {ckpt}")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    _require(isinstance(state, Mapping)
             and str(state.get("schema")) == V542_TRAIN_SCHEMA
             and str(state.get("candidate_id")) == candidate_id
             and int((state.get("identity") or {}).get("seed", -1)) == seed,
             "V542_STATS_RUN_INCOMPLETE",
             f"{ckpt} is not the seed-{seed} {candidate_id} formal "
             "checkpoint")
    assembled = assemble_v542_model(
        candidate_id, seed, device=str(device), require_identity=False)
    assembled.model.load_state_dict(state["model"], strict=True)
    assembled.model.train(False)
    return assembled.model


def paired_cluster_values(cfg: Mapping[str, Any], campaign_root: Any,
                          candidate_id: str, *, device: str = "cpu",
                          seeds: Sequence[int] = FROZEN_SEEDS
                          ) -> Dict[str, Any]:
    """Production per-cluster paired values (curve = cluster).

    Reuses the FROZEN evaluator exactly as the sanity-band/extract legs
    (same positions derivation, same config, ``tau_end=None``): each
    inner-val curve is evaluated on its own frame subset for the
    candidate and its matched A0, giving per-cluster metric values on
    identical frame sets for both models (pairing).  Undefined endpoint
    values are ``None`` and are dropped pairwise by the bootstrap.
    """
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        V53MPTrainerConfig,
        _CacheView,
        _FeatureLRU,
        _train_q_mean,
        evaluate_v53mp,
    )
    from scripts.agent_ops.v542_campaign import load_campaign_env
    from scripts.agent_ops.v542_formal import (
        _band_inner_val_positions,
        _band_train_positions,
    )

    env = load_campaign_env(cfg)
    out: Dict[str, Any] = {endpoint: {} for endpoint in ENDPOINTS}
    for seed in seeds:
        positions = _band_inner_val_positions(cfg, env.ds, int(seed))
        by_curve: Dict[str, List[int]] = {}
        for p in positions:
            sid = str(env.ds.samples[int(p)])
            curve = str(env.labels.by_sample[sid].curve_id)
            by_curve.setdefault(curve, []).append(int(p))
        train_pos, _val = _band_train_positions(cfg, env.ds, int(seed))
        q_mean = _train_q_mean(env.ds.samples, env.labels, train_pos)
        eval_cfg = V53MPTrainerConfig(
            arm="A", seed=int(seed), device=str(device))
        lru = _FeatureLRU(env.ds, capacity=len(positions) + 64)
        cache = _CacheView(lru)
        models = {
            "candidate": _load_formal_model(
                cfg, campaign_root, f"V542_{candidate_id}_S{seed}",
                device=device),
            "a0": _load_formal_model(
                cfg, campaign_root, f"V542_A0_S{seed}", device=device),
        }
        per_side: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
            side: {endpoint: {} for endpoint in ENDPOINTS}
            for side in ("candidate", "a0")}
        for curve in sorted(by_curve):
            sub = sorted(by_curve[curve])
            for side, model in models.items():
                report = evaluate_v53mp(
                    model, None, env.ds, env.labels, env.endpoint, cache,
                    sub, eval_cfg, train_q_mean=q_mean, tau_end=None)
                values: Dict[str, Optional[float]] = {}
                for key, leaf in METRIC_LEAVES.items():
                    cur: Any = report
                    for part in leaf.split("."):
                        cur = (cur or {}).get(
                            part) if isinstance(cur, Mapping) else None
                    values[key] = float(cur) if _finite(cur) else None
                for endpoint, metric in ENDPOINT_TO_METRIC.items():
                    per_side[side][endpoint][curve] = values[metric]
        for endpoint in ENDPOINTS:
            out[endpoint][int(seed)] = {
                "candidate": [per_side["candidate"][endpoint][c]
                              for c in sorted(by_curve)],
                "a0": [per_side["a0"][endpoint][c]
                       for c in sorted(by_curve)],
            }
    return out


def run_h3(cfg: Mapping[str, Any], *, campaign_root: Any,
           device: str = "cpu",
           candidate_inputs: Optional[Mapping[str, Mapping[str, Any]]] = None
           ) -> Dict[str, Any]:
    """The full H3 computation over the eligible candidate set.

    ``candidate_inputs`` is a TEST seam (synthetic dicts); the
    production path assembles them from the metrics table + the H2
    report + the run matrix + the frozen per-cluster evaluator leg.
    """
    if candidate_inputs is None:
        path = Path(campaign_root) / METRICS_TABLE_REL
        table = latest_metrics_by_run(load_metrics_table(path))
        h2_doc = _load_json(Path(campaign_root) / H2_REPORT_REL,
                            "V542_STATS_H2_REPORT_MISSING",
                            "the H2 report")
        runs_matrix = {
            str(r.get("backbone")): bool(r.get("winner_eligible"))
            for r in ((cfg.get("run_matrix") or {}).get("runs") or [])
            if isinstance(r, Mapping)}
        ledger = latest_ledger_state(read_ledger(campaign_root))
        candidate_inputs = {}
        seen_backbones = sorted({
            str(e.get("candidate_id")) for e in table.values()
            if str(e.get("candidate_id")) in ("A1", "A2", "A3", "A4", "A5")
            and str(e.get("outcome")) == "done"})
        for backbone in seen_backbones:
            point: Dict[str, Any] = {}
            h2_by_seed: Dict[int, bool] = {}
            losses: List[float] = []
            for seed in FROZEN_SEEDS:
                cand_entry = table.get(f"V542_{backbone}_S{seed}")
                a0_entry = table.get(f"V542_A0_S{seed}")
                h2_entry = ((h2_doc.get("runs") or {})
                            .get(f"V542_{backbone}_S{seed}")) or {}
                if cand_entry is None or a0_entry is None:
                    continue
                point[str(seed)] = {"candidate": dict(cand_entry["metrics"]),
                                    "a0": dict(a0_entry["metrics"])}
                h2_by_seed[seed] = bool(h2_entry.get("h2_no_breach"))
                loss = (ledger.get(f"V542_{backbone}_S{seed}")
                        or {}).get("final_inner_val_loss")
                if _finite(loss):
                    losses.append(float(loss))
            if len(point) < H3_TOTAL_SEEDS and not point:
                # incomplete run: keep an empty shell so eligibility
                # records "three_seeds_incomplete" honestly.
                point = {}
            paired = paired_cluster_values(
                cfg, campaign_root, backbone, device=device)
            candidate_inputs[backbone] = {
                "winner_eligible": bool(runs_matrix.get(backbone)),
                "h2_no_breach_by_seed": h2_by_seed,
                "point_metrics": point,
                "paired": paired,
                "core_time_ratio": float(cost_gate_ratio(
                    cfg, backbone)["ratio"]),
                "active_params": int(
                    ((cfg.get("backbone_contract") or {}).get("candidates")
                     or {}).get(backbone, {}).get("backbone_active", 0)),
                "inner_val_total_loss": (
                    float(np.mean(losses)) if losses else float("nan")),
            }
    report = compute_h3(candidate_inputs)
    report_path = _write_report(campaign_root, H3_REPORT_REL, report)
    report["report_path"] = str(report_path)
    return report


# ===========================================================================
# 5. select — frozen final_selection_rule
# ===========================================================================
def select_backbone(h3_report: Mapping[str, Any], *,
                    rule_block: Mapping[str, Any],
                    statistics_contract_sha256: Optional[str] = None
                    ) -> Dict[str, Any]:
    """The frozen ``final_selection_rule`` (7-step lexicographic tuple).

    Eligibility is H3-qualified ONLY (A0 and the diagnostic variants
    A1-C/A5-C never participate); 0 qualified ⇒ ``no_backbone_winner``
    (A0 retained); 1 ⇒ that candidate; many ⇒ the exact frozen
    tie-break order.  ``dev_used=false`` always.  Every H3-qualified
    candidate is reported; the rule block and its canonical SHA go into
    the selection record.
    """
    _require(bool(rule_block), "V542_STATS_RULE_BLOCK_MISSING",
             "the frozen final_selection_rule block is required")
    _require(str(rule_block.get("eligible_condition", ""))
             == FINAL_SELECTION_ELIGIBLE_CONDITION,
             "V542_STATS_RULE_BLOCK_INVALID",
             "the rule block must carry eligible_condition="
             f"{FINAL_SELECTION_ELIGIBLE_CONDITION!r} (got "
             f"{rule_block.get('eligible_condition')!r})")
    qualified: List[Dict[str, Any]] = []
    for cid, entry in sorted((h3_report.get("candidates") or {}).items()):
        if not entry.get("h3_qualified"):
            continue
        margin = entry.get("max_effect_margin_on_qualifying_endpoint")
        qualified.append({
            "candidate_id": cid,
            "move_qualified": bool(entry.get("move_qualified")),
            "process_qualified": bool(entry.get("process_qualified")),
            "route_priority": (0 if entry.get("move_qualified") else 1),
            "qualifying_confirmatory_endpoints": list(
                entry.get("qualifying_confirmatory_endpoints") or []),
            "min_holm_adjusted_p_on_qualifying_endpoint": entry.get(
                "min_holm_adjusted_p_on_qualifying_endpoint"),
            "max_effect_margin_on_qualifying_endpoint": margin,
            "core_time_ratio": entry.get("core_time_ratio"),
            "active_params": entry.get("active_params"),
            "inner_val_total_loss": entry.get("inner_val_total_loss"),
            "process_only_note": entry.get("process_only_note"),
        })

    def _sort_key(c: Mapping[str, Any]) -> Any:
        adj_p = c["min_holm_adjusted_p_on_qualifying_endpoint"]
        margin = c["max_effect_margin_on_qualifying_endpoint"]
        return (
            int(c["route_priority"]),
            float(adj_p) if _finite(adj_p) else float("inf"),
            -float(margin) if _finite(margin) else float("inf"),
            float(c["core_time_ratio"]) if _finite(c.get("core_time_ratio"))
            else float("inf"),
            int(c["active_params"]) if c.get("active_params") is not None
            else 2 ** 31,
            float(c["inner_val_total_loss"])
            if _finite(c.get("inner_val_total_loss")) else float("inf"),
            str(c["candidate_id"]),
        )

    n = len(qualified)
    if n == 0:
        branch = FINAL_SELECTION_NO_CANDIDATE_RULE_ID
        outcome, selected = "no_backbone_winner", None
    elif n == 1:
        branch = FINAL_SELECTION_SINGLE_RULE_ID
        outcome, selected = "backbone_selected", str(
            qualified[0]["candidate_id"])
    else:
        branch = "multi_candidate_frozen_lexicographic_tie_break"
        qualified.sort(key=_sort_key)
        outcome, selected = "backbone_selected", str(
            qualified[0]["candidate_id"])
    if n > 1:
        for rank, c in enumerate(qualified, start=1):
            c["tie_break_rank"] = rank
    return {
        "schema": SELECTION_SCHEMA,
        "kind": "final_selection",
        "eligible_condition": FINAL_SELECTION_ELIGIBLE_CONDITION,
        "selection_rule_branch": branch,
        "outcome": outcome,
        "selected_backbone": selected,
        "a0_retained": bool(selected is None),
        "n_qualified": n,
        "qualified_candidates": qualified,
        "excluded_from_selection": ["A0_matched_baseline",
                                    "diagnostic_variants_A1-C_A5-C"],
        "dev_used": False,
        "dev_policy": ("frozen_final_models_only — dev is evaluated once "
                       "AFTER the backbone freeze and NEVER participates "
                       "in selection"),
        "report_all_qualified": True,
        "tie_break_order": list(
            rule_block.get("tie_break_order") or FINAL_SELECTION_TIE_BREAK),
        "effect_margin_definition": dict(
            rule_block.get("effect_margin_definition") or {}),
        "rule_block": dict(rule_block),
        "rule_block_sha256": _canonical_sha256(dict(rule_block)),
        "statistics_contract_sha256": statistics_contract_sha256,
        "rule_block_in_statistics_sha": True,
        "written_at": _now(),
    }


def run_select(cfg: Mapping[str, Any], *, campaign_root: Any
               ) -> Dict[str, Any]:
    """Select Backbone* from the H3 report via the frozen rule."""
    h3_doc = _load_json(Path(campaign_root) / H3_REPORT_REL,
                        "V542_STATS_H3_REPORT_MISSING", "the H3 report")
    rule_block = dict(((cfg.get("evaluation_contract") or {}).get(
        "final_selection_rule")) or {})
    _require(bool(rule_block), "V542_STATS_RULE_BLOCK_MISSING",
             "config evaluation_contract.final_selection_rule is missing")
    stats_sha: Optional[str] = None
    try:
        from pes2ts_core.pesn.models.v542.identity import (
            statistics_contract_sha256)
        stats_sha, _payload = statistics_contract_sha256(cfg)
    except Exception:  # noqa: BLE001 - identity/torch unavailable
        stats_sha = None
    record = select_backbone(h3_doc, rule_block=rule_block,
                             statistics_contract_sha256=stats_sha)
    path = _write_report(campaign_root, SELECTION_RECORD_REL, record)
    record["selection_record_path"] = str(path)
    return record


# ===========================================================================
# 6. activate — frozen activation decisions (skips carry frozen codes)
# ===========================================================================
def decide_activations(*, h3_report: Mapping[str, Any],
                       top2: Sequence[str],
                       selected_backbone: Optional[str],
                       a5_screen_pass: bool = False,
                       a5_full_dose_done: bool = False,
                       a5_launch_gates_ok: bool = True) -> Dict[str, Any]:
    """The frozen activation decisions (no training; skips = 0 runs).

    * A5 full-dose (``V542_A5_S17``) — condition
      ``no_A1_A4_h3_qualified_candidate_after_top2_seeds``: activates
      only when the A1–A4 top2 three-seed final H3 has NO qualified
      candidate.  An A4 that never ran (the binding WP4 diagnostic
      ruling) is a CONSEQUENCE of that condition — never a blocker.
    * A5 supplement (``V542_A5_S29/S43``) — condition
      ``a5_full_dose_complete_and_seed17_screen_pass``.
    * A1-C — ``a1_in_top2_supplement_set_OR_a1_selected_as_backbone``
      (in the top2 set OR selected Backbone* — not merely top-ranked).
    * A5-C — ``a5_selected_backbone`` ONLY.
    """
    qualified = [cid for cid, entry in sorted(
        (h3_report.get("candidates") or {}).items())
        if entry.get("h3_qualified")]
    qualified_a1_a4 = [cid for cid in qualified
                       if cid in ("A1", "A2", "A3", "A4")]
    a4_qualified = "A4" in qualified
    no_qualified = not qualified_a1_a4

    a5_activated = bool(no_qualified and a5_launch_gates_ok)
    a5_decision = {
        "run_id": "V542_A5_S17",
        "backbone": "A5",
        "frozen_condition": A5_FROZEN_CONDITION,
        "activated": a5_activated,
        "status": "launchable" if a5_activated else "gate_skipped",
        "reason": None if a5_activated else A5_SKIP_REASON,
        "new_training": 1 if a5_activated else 0,
        "qualified_a1_a4_candidates": qualified_a1_a4,
        "a4_qualified": a4_qualified,
        "a4_never_ran_is_consequence_not_blocker": (
            "A4 full-dose was never run by the binding WP4 diagnostic "
            "ruling (gate_skipped); its absence only contributes to the "
            "frozen 'no qualified A1-A4 candidate' condition and NEVER "
            "blocks or waives it"),
        "launch_requires_ok": bool(a5_launch_gates_ok),
        "no_discretion": True,
    }

    supplement_condition = bool(a5_full_dose_done and a5_screen_pass)
    a5_runs = ["V542_A5_S29", "V542_A5_S43"]
    supplement_decision = {
        "run_ids": a5_runs,
        "backbone": "A5",
        "frozen_condition": A5_SUPPLEMENT_CONDITION,
        "activated": bool(supplement_condition),
        "status": "launchable" if supplement_condition else "gate_skipped",
        "reason": (None if supplement_condition
                   else A5_SUPPLEMENT_SKIP_REASON),
        "new_training": len(a5_runs) if supplement_condition else 0,
        "a5_full_dose_complete": bool(a5_full_dose_done),
        "a5_seed17_screen_pass": bool(a5_screen_pass),
        "a5_supplement_max_runs": 2,
        "not_in_top2": True,
    }

    a1c_activated = bool(("A1" in top2) or selected_backbone == "A1")
    a1c_decision = {
        "run_id": "V542_A1C_S17",
        "backbone": "A1-C",
        "frozen_condition": ("a1_in_top2_supplement_set_OR_a1_selected_as"
                             "_backbone"),
        "activated": a1c_activated,
        "status": "launchable" if a1c_activated else "gate_skipped",
        "reason": None if a1c_activated else A1C_SKIP_REASON,
        "new_training": 1 if a1c_activated else 0,
        "winner_eligible": False,
        "not_only_top_ranked": True,
    }

    a5c_activated = selected_backbone == "A5"
    a5c_decision = {
        "run_id": "V542_A5C_S17",
        "backbone": "A5-C",
        "frozen_condition": "a5_selected_backbone",
        "activated": a5c_activated,
        "status": "launchable" if a5c_activated else "gate_skipped",
        "reason": None if a5c_activated else A5C_SKIP_REASON,
        "new_training": 1 if a5c_activated else 0,
        "winner_eligible": False,
        "capacity_caveat_unresolved": bool(
            "A5" in qualified and not a5c_activated),
    }
    if a5c_decision["capacity_caveat_unresolved"]:
        a5c_decision["capacity_caveat_note"] = (
            "A5 is H3-qualified but was NOT selected Backbone* — the "
            f"capacity caveat stays {CAPACITY_CAVEAT_UNRESOLVED}; A5-C "
            "is not run afterwards (no post-hoc rerun)")

    total_new = (a5_decision["new_training"]
                 + supplement_decision["new_training"]
                 + a1c_decision["new_training"]
                 + a5c_decision["new_training"])
    return {
        "schema": ACTIVATION_SCHEMA,
        "kind": "activation_decisions",
        "a5_full_dose": a5_decision,
        "a5_supplement": supplement_decision,
        "a1c": a1c_decision,
        "a5c": a5c_decision,
        "h3_qualified_candidates": qualified,
        "selected_backbone": selected_backbone,
        "top2": list(top2),
        "total_new_training_runs": total_new,
        "skips_are_zero_new_training": all(
            d["new_training"] == 0 for d in
            (a5_decision, supplement_decision, a1c_decision, a5c_decision)
            if not d["activated"]),
        "training_performed": False,
        "written_at": _now(),
    }


def run_activate(cfg: Mapping[str, Any], *, campaign_root: Any
                 ) -> Dict[str, Any]:
    """Emit the frozen activation decisions from the WP6 reports."""
    h3_doc = _load_json(Path(campaign_root) / H3_REPORT_REL,
                        "V542_STATS_H3_REPORT_MISSING", "the H3 report")
    screen_doc = _load_json(Path(campaign_root) / SCREEN_REPORT_REL,
                            "V542_STATS_SCREEN_REPORT_MISSING",
                            "the seed17 screen report")
    selection_path = Path(campaign_root) / SELECTION_RECORD_REL
    selected: Optional[str] = None
    if selection_path.is_file():
        selected = _load_json(
            selection_path, "V542_STATS_SELECTION_INVALID",
            "the selection record").get("selected_backbone")
    top2 = list(screen_doc.get("top2") or [])
    a5_screen = screen_doc.get("a5_seed17_screen_pass")
    state = latest_ledger_state(read_ledger(campaign_root))
    a5_row = state.get("V542_A5_S17") or {}
    report = decide_activations(
        h3_report=h3_doc, top2=top2, selected_backbone=selected,
        a5_screen_pass=bool(a5_screen),
        a5_full_dose_done=str(a5_row.get("outcome")) == "done")
    path = _write_report(campaign_root, ACTIVATION_REPORT_REL, report)
    report["activation_report_path"] = str(path)
    return report


# ===========================================================================
# CLI
# ===========================================================================
def _load_cfg(config_path: Any) -> Dict[str, Any]:
    from scripts.agent_ops.v542_formal import load_formal_config

    return load_formal_config(config_path)


def _print(doc: Mapping[str, Any]) -> None:
    print(json.dumps(dict(doc), indent=1, sort_keys=True, default=str))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v542_stats.py",
        description="V5.4.2 WP6 statistics + selection machinery "
                    "(extract / h2 / screen / h3 / select / activate — "
                    "frozen pre-registered contracts only; no torch."
                    "compile; no dev read; no QC/confirm; the CLI "
                    "exposes NO threshold/dose knobs)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser(
        "extract", help="evaluate finished formal runs into the "
                        "append-only metrics table (frozen evaluator)")
    p_ext.add_argument("--device", default="cpu")
    p_ext.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_ext.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_ext.add_argument("--run", dest="runs", nargs="*", default=None,
                       help="restrict to these run ids")

    p_h2 = sub.add_parser(
        "h2", help="frozen H2 guards vs matched A0 AND R3 "
                   "(report-only scientific_failure disposition)")
    p_h2.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_h2.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_h2.add_argument("--r3-metrics", default=None,
                      help="located R3 reference record (default: "
                           "<campaign-root>/plan/r3_reference_metrics.json)")

    p_scr = sub.add_parser(
        "screen", help="frozen seed17_screen_pass + top2 selection "
                       "(A0_S17 + sanity-band freeze asserted)")
    p_scr.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_scr.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))

    p_h3 = sub.add_parser(
        "h3", help="frozen paired cluster bootstrap + Holm + route "
                   "verdicts (10,000 draws, root 20261029)")
    p_h3.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_h3.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_h3.add_argument("--device", default="cpu")

    p_sel = sub.add_parser(
        "select", help="frozen final_selection_rule over the H3 report")
    p_sel.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_sel.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))

    p_act = sub.add_parser(
        "activate", help="frozen A5/A1-C/A5-C activation decisions "
                         "(skips carry the frozen reason codes)")
    p_act.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_act.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))

    args = parser.parse_args(argv)
    try:
        cfg = _load_cfg(args.config)
        if args.command == "extract":
            report = run_extract(cfg, campaign_root=args.campaign_root,
                                 device=args.device, only=args.runs)
        elif args.command == "h2":
            r3 = None
            if args.r3_metrics:
                r3 = _load_json(args.r3_metrics,
                                "V542_STATS_R3_REFERENCE_MISSING",
                                "the --r3-metrics record")
            report = run_h2(cfg, campaign_root=args.campaign_root,
                            r3_metrics=r3)
        elif args.command == "screen":
            report = run_screen(cfg, campaign_root=args.campaign_root)
        elif args.command == "h3":
            report = run_h3(cfg, campaign_root=args.campaign_root,
                            device=args.device)
        elif args.command == "select":
            report = run_select(cfg, campaign_root=args.campaign_root)
        else:
            report = run_activate(cfg, campaign_root=args.campaign_root)
        _print(report)
        return 0
    except V542StatsError as exc:
        print(f"V542 stats lane refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
