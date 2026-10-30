#!/usr/bin/env python3
"""V5.4.2 WP5 FORMAL runner + ``V542-A0-sanity-band`` evaluator (plan §8,
ADR-0033 §2.2/§2.6, config ``run_matrix``).

One CLI, three subcommands — this is the FORMAL lane (unlike the WP4
technical-measurement harness ``v542_campaign.py`` which never runs
formal):

* ``run --stage T0|T1|T2|T3|T-ATTR`` — validate the frozen config in
  FORMAL mode first (every validator rejection is surfaced as a typed
  refusal with the VERBATIM validator code, e.g. ``V542_RECORD_MISSING``),
  then decide + execute the eligible rows of that stage.  Ordering and
  gating are enforced in code with explicit reason codes:

  - **T0** = ``V542_A0_S17/S29/S43`` at the frozen formal dose
    (400 W0 + 37,500 W1, ``val_every=100``); ``winner_eligible=false``;
    NO diagnostic gate (WP4 ruling: A0 is the matched control,
    ``control_not_screened``); must ALL be ``done`` before any T1;
  - **T1** = ``V542_A1/A2/A3_S17`` only.  ``V542_A4_S17`` NEVER runs:
    the WP4 diagnostic ruling (3.849742% < 5.0%, binding) records it
    ``gate_skipped(diag_best_inner_val_drop_lt_5pct_vs_initial)`` with
    ``new_training=0`` — enforced even under ``--only``.  T1 additionally
    requires the ``V542-A0-sanity-band`` PASS record (absent or breach =
    typed refusal, no T1 launch), the per-candidate diagnostic disposition
    (effective v2 report) and the signed cost gate (ratio ≤ 2.0);
  - **T2** = ``V542_A5_S17``: the frozen activation condition
    (``no_A1_A4_h3_qualified_candidate_after_top2_seeds``) is not
    evaluable before WP6 ⇒ recorded ``gate_skipped`` with the frozen
    ``a5_not_activated_*`` reason, never launched;
  - **T3**: top-2 rows (A1–A4 × S29/S43) are ``gate_skipped
    (top2_not_frozen)`` (A4 rows keep the binding A4 reason); A5
    supplement rows are ``gate_skipped(a5_supplement_parent_gate_unmet)``
    — nothing launches before WP6 freezes the predicates;
  - **T-ATTR**: ``V542_A1C_S17`` / ``V542_A5C_S17`` stay
    ``gate_skipped`` with the frozen skip reasons
    (``a1_not_in_top2_and_not_selected`` /
    ``a5_not_selected_backbone``); this runner launches them only if the
    frozen activation predicate is satisfied — which, before the WP7
    selection record exists, it never is.

* ``sanity-band`` — the ``V542-A0-sanity-band`` hard gate (fix-5 C5):
  evaluate the three T0 A0 formal checkpoints with the FROZEN evaluator
  ``key_atom_v53mp_train_v3.evaluate_v53mp`` over the same seed's inner-
  val positions, locate the V541 E0×3 reference in the V541 run
  artifacts (archived ``metrics.json`` — provenance recorded, never
  fabricated), then compute the band on PER-ENDPOINT SEED MEANS (not
  per-seed checks): RMSE/zero relative ≤ 2% AND AUPRC/q absolute ≤ 0.01.
  Writes ``<campaign_root>/plan/a0_sanity_band.json`` with the verdict
  (``pass``/``breach``), the numbers, the evaluator identity and the
  source run identities.  On breach it writes verdict=breach (never a
  pass), lists the breached endpoints and states that T1 stays blocked
  with the frozen migration-audit categories (code/data/init + the
  corresponding identity-layer bumps).  ``--dry-run`` prints the plan
  without evaluating or writing.

* ``status`` — print the run-ledger state (per run: outcome,
  new_training, identity digest, stage).

Per-run execution reuses the FROZEN machinery, never forks it: model
assembly via ``v542.assembly.assemble_v542_model`` (through the campaign
identity seam), training via ``v542_train.train_run_v542`` (resume from
``checkpoint_last.pt``; an identity mismatch is the trainer's typed
``invalid_run`` refusal raised BEFORE anything is written — the
checkpoint stays byte-for-byte untouched).  The runner appends exactly
one row per run decision to ``<campaign_root>/run_ledger.jsonl``
(append-only, six-word outcome vocabulary) and writes a
``run_manifest.json`` per executed run.

Discipline: no ``torch.compile``; single-lane / no-swap / confirm-never-
read / zero-QC guards asserted on the frozen config on every subcommand;
no QC/dev/confirm access; the CLI exposes NO dose/threshold knobs (test
doses exist only as explicit program-level test seams and are recorded
honestly as ``test_dose`` with ``is_formal_schedule=false``).

CLI::

    python scripts/agent_ops/v542_formal.py run --stage T0 [--only RUN_ID ...]
                                            [--device cpu] [--dry-run]
    python scripts/agent_ops/v542_formal.py sanity-band [--device cpu] [--dry-run]
    python scripts/agent_ops/v542_formal.py status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# frozen constants (mirror the frozen config / ADR-0033; nothing invented)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = PROJECT_ROOT / (
    "config/campaigns/pes2ts_v542_backbone_50k.yaml")
#: config ``plan_outputs.root``.
DEFAULT_CAMPAIGN_ROOT = PROJECT_ROOT / "outputs/pesn/v542_backbone_50k"
#: Frozen runplan pin template used when the CLI assembles run identities.
DEFAULT_RUNPLAN = PROJECT_ROOT / (
    "config/campaigns/v542_templates/runplan_pin.template.json")
#: Default location of the V541 E0 archived runs (parity asset paths in
#: config ``backbone_contract.a0_migration.parity.assets``).
DEFAULT_V541_RUNS_ROOT = PROJECT_ROOT / "outputs/pesn/v541_embed_50k/runs"

#: Ledger / manifest / band record schema ids.
FORMAL_SCHEMA: str = "pes2ts_v542_formal_runner_v1"
LEDGER_SCHEMA: str = "pes2ts_v542_run_ledger_v1"
MANIFEST_SCHEMA: str = "pes2ts_v542_run_manifest_v1"
BAND_SCHEMA: str = "pes2ts_v542_a0_sanity_band_v1"

#: Append-only ledger filename under the campaign root.
LEDGER_FILENAME = "run_ledger.jsonl"
#: The registry canonical gate this runner implements.
A0_SANITY_BAND_GATE = "V542-A0-sanity-band"
#: Band record path (relative to the campaign root).
BAND_RECORD_REL = "plan/a0_sanity_band.json"

#: Formal stages this CLI accepts (T-DIAG belongs to the WP4 harness).
FORMAL_STAGES: tuple = ("T0", "T1", "T2", "T3", "T-ATTR")

#: The three T0 A0 runs (config ``a0_baseline.runs``).
T0_RUN_IDS: tuple = ("V542_A0_S17", "V542_A0_S29", "V542_A0_S43")
#: The frozen seeds (config ``split_contract.seeds``).
FROZEN_SEEDS: tuple = (17, 29, 43)

#: Launch whitelist: the ONLY (stage, backbone) pairs this runner may
#: ever train.  A4 is NEVER launchable (WP4 binding ruling); every
#: conditional/diagnostic row is decided by its frozen predicate.
LAUNCHABLE: Dict[str, tuple] = {
    "T0": ("A0",),
    "T1": ("A1", "A2", "A3"),
    # WP6 top-2 rows: only A1-A3 can ever appear here; membership in the
    # frozen top2 set is enforced by ``decide_stage_runs`` (A4/A5/A1-C/
    # A5-C stay structurally forbidden and are recorded gate_skipped).
    "T3": ("A1", "A2", "A3"),
}

# -- frozen reason codes ----------------------------------------------------
REASON_FORMAL_DONE: str = "formal_run_complete"
#: WP4 diagnostic ruling 2026-10-30 (ADR-0033 v1.3.3) — A4 binding.
REASON_A4_GATE_SKIPPED: str = "diag_best_inner_val_drop_lt_5pct_vs_initial"
#: T2 A5 frozen activation condition is not evaluable before WP6.
REASON_A5_NOT_ACTIVATED: str = (
    "a5_not_activated_no_A1_A4_h3_qualified_candidate_after_top2_seeds")
A5_FROZEN_CONDITION: str = (
    "no_A1_A4_h3_qualified_candidate_after_top2_seeds")
#: T3 top-2 rows: top-2 freeze (A0_S17 complete + band pass + seed17
#: screens) happens in WP6 — before that the rows stay gate_skipped.
REASON_TOP2_NOT_FROZEN: str = "top2_not_frozen"
#: T3 top-2 rows whose candidate is NOT in the frozen top2 set.
REASON_TOP2_NOT_SELECTED: str = "top2_not_selected_inner_val_screen"
#: WP6 seed17 screen report (written by ``v542_stats.py screen``).
TOP2_SCREEN_REL: str = "reports/seed17_screen_report.json"
#: T3 A5 supplement rows: parent gate cannot hold while T2 never ran.
REASON_A5_SUPPLEMENT: str = "a5_supplement_parent_gate_unmet"
A5_SUPPLEMENT_CONDITION: str = (
    "a5_full_dose_complete_and_seed17_screen_pass")
#: T-ATTR frozen skip reasons (config ``a1c_activation`` /
#: ``a5c_activation.skip_reason``).
REASON_A1C_SKIP: str = "a1_not_in_top2_and_not_selected"
REASON_A5C_SKIP: str = "a5_not_selected_backbone"
#: Cost gate exceed (plan §12: NOT a resource_failure).
REASON_COST_EXCEEDS: str = "gate_skipped_cost_ratio_exceeds_2"
#: Diagnostic-eliminated T1 candidate carries the report's own reason.
REASON_DIAG_ADVANCE: str = (
    "diag_all_engineering_screens_pass_advance_full_dose")

# -- typed refusal codes ----------------------------------------------------
#: Frozen E0 reference run-id pattern.
_E0_RUN_PATTERN = "V541_T0R_E0_S{seed}"

#: The frozen evaluator identity recorded in every band record.
EVALUATOR_IDENTITY: Dict[str, Any] = {
    "evaluator": ("pes2ts_core.pesn.offline.training."
                  "key_atom_v53mp_train_v3.evaluate_v53mp (frozen)"),
    "endpoint_keys": {
        "rmse_zero": "move.key_rmse_over_zero.median_rmse_over_zero",
        "endpoint_auprc": "endpoint_head.auprc",
        "q_corr": "q_head.corr_curve_median",
    },
    "statistic": ("per-endpoint THREE-SEED MEAN (A0x3 formal vs V541 E0x3 "
                  "archive) — seed-mean vs seed-mean, not per-seed checks"),
    "tau_end": None,
    "tau_note": ("tau_end=None on the live A0 leg — tau only affects "
                 "misstop metrics, never the three band endpoints"),
    "v541_side": ("archived V541 E0 metrics.json (frozen-evaluator outputs "
                  "recorded by the V541 trainer's inner-val evaluation); "
                  "cross-evaluator bridge = the V542-A0 parity certificate "
                  "(evaluator parity bitwise-or-<=1e-9)"),
}


class V542FormalError(RuntimeError):
    """Typed refusal of the formal runner (``code`` + ``detail``).

    ``outcome`` (when set) is the six-word outcome this refusal maps to
    (e.g. ``invalid_run`` on a resume-identity mismatch) so operational
    failures stay structurally distinct from scientific results.
    """

    def __init__(self, code: str, detail: str,
                 outcome: Optional[str] = None) -> None:
        super().__init__(f"{code}: {detail}" if detail else str(code))
        self.code = str(code)
        self.detail = str(detail)
        self.outcome = outcome


def _require(cond: bool, code: str, detail: str,
             outcome: Optional[str] = None) -> None:
    if not cond:
        raise V542FormalError(code, detail, outcome=outcome)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_digest(identity: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The ONE canonical digest recipe (identity ``canonical_sha256``)."""
    if identity is None:
        return None
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256

    return canonical_sha256(dict(identity))


# ---------------------------------------------------------------------------
# config + operational guards + formal validation gate
# ---------------------------------------------------------------------------
def load_formal_config(path: Optional[Any] = None) -> Dict[str, Any]:
    """Load the frozen campaign config (read-only)."""
    from scripts.agent_ops.v542_campaign import (
        load_campaign_config,
        assert_operational_guards,
    )

    cfg = load_campaign_config(path)
    assert_operational_guards(cfg)     # single-lane / no-swap / confirm / QC
    return cfg


def formal_gate(config_path: Any, *,
                record_root: Optional[Path] = None) -> Dict[str, Any]:
    """``--formal`` DELEGATES to the frozen config validator.

    Any typed validator rejection (:class:`V542SchemaError`) is surfaced
    VERBATIM (same code, same detail prefix) as a typed formal refusal —
    the runner can never launch on a config the frozen validator rejects.
    ``record_root`` is a test seam (mirrors ``validate_config``).
    """
    from scripts.agent_ops.v542_config_schema import (
        V542SchemaError,
        load_config,
        validate_config,
    )

    try:
        return validate_config(load_config(config_path), formal=True,
                               record_root=record_root)
    except V542SchemaError as exc:
        raise V542FormalError(
            exc.code,
            "formal launch is typed-blocked by the frozen config "
            "validator: " + exc.detail) from exc


# ---------------------------------------------------------------------------
# run matrix projection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunRow:
    """One frozen ``run_matrix.runs`` row, projected."""

    run_id: str
    stage: str
    backbone: str
    seed: int
    run_class: str
    winner_eligible: bool
    condition: Optional[str] = None
    skip_reason: Optional[str] = None


def stage_rows(cfg: Mapping[str, Any], stage: str) -> List[RunRow]:
    """The frozen run rows of one formal stage (config order)."""
    _require(stage in FORMAL_STAGES, "V542_FORMAL_STAGE_UNKNOWN",
             f"stage must be one of {list(FORMAL_STAGES)} (T-DIAG belongs "
             f"to the WP4 technical-measurement harness), got {stage!r}")
    runs = (cfg.get("run_matrix") or {}).get("runs")
    _require(isinstance(runs, Sequence) and bool(runs),
             "V542_FORMAL_CONFIG_INVALID",
             "config run_matrix.runs is missing/empty")
    out: List[RunRow] = []
    for raw in runs:
        if not isinstance(raw, Mapping) or raw.get("stage") != stage:
            continue
        out.append(RunRow(
            run_id=str(raw["run_id"]),
            stage=str(raw["stage"]),
            backbone=str(raw["backbone"]),
            seed=int(raw["seed"]),
            run_class=str(raw.get("class", "")),
            winner_eligible=bool(raw.get("winner_eligible", False)),
            condition=(None if raw.get("condition") is None
                       else str(raw["condition"])),
            skip_reason=(None if raw.get("skip_reason") is None
                         else str(raw["skip_reason"])),
        ))
    _require(bool(out), "V542_FORMAL_STAGE_UNKNOWN",
             f"stage {stage!r} has no rows in the frozen run matrix")
    return out


def filter_only(rows: Sequence[RunRow],
                only: Optional[Sequence[str]]) -> List[RunRow]:
    """Restrict to ``--only`` run ids (typed refusal for unknown ids)."""
    if not only:
        return list(rows)
    wanted = [str(r) for r in only]
    known = {row.run_id for row in rows}
    unknown = [r for r in wanted if r not in known]
    _require(not unknown, "V542_FORMAL_RUN_UNKNOWN",
             f"run id(s) {unknown} are not rows of this stage "
             f"(stage rows: {sorted(known)})")
    keep = set(wanted)
    return [row for row in rows if row.run_id in keep]


# ---------------------------------------------------------------------------
# ledger (append-only)
# ---------------------------------------------------------------------------
def ledger_path(campaign_root: Any) -> Path:
    return Path(campaign_root) / LEDGER_FILENAME


def read_ledger(campaign_root: Any) -> List[Dict[str, Any]]:
    """All ledger rows (append-only JSONL; missing file ⇒ ``[]``)."""
    from scripts.agent_ops.v542_campaign import read_jsonl

    return read_jsonl(ledger_path(campaign_root))


def latest_ledger_state(rows: Sequence[Mapping[str, Any]]
                        ) -> Dict[str, Dict[str, Any]]:
    """Latest row per run_id (the ledger is append-only; last wins)."""
    state: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        if run_id:
            state[run_id] = dict(row)
    return state


def append_ledger_row(campaign_root: Any,
                      row: Mapping[str, Any]) -> Dict[str, Any]:
    """APPEND exactly one ledger row (``seq`` = append order, monotone)."""
    from scripts.agent_ops.v542_campaign import append_jsonl

    existing = read_ledger(campaign_root)
    payload = dict(row)
    payload["schema"] = LEDGER_SCHEMA
    payload["seq"] = len(existing) + 1
    payload.setdefault("ts", _now())
    payload.setdefault("torch_compile", False)
    append_jsonl(ledger_path(campaign_root), payload)
    return payload


def _ledger_row(*, run_id: str, stage: str, backbone: str, seed: int,
                outcome: str, new_training: int, reason: str,
                identity: Optional[Mapping[str, Any]],
                identity_sha256: Optional[str],
                wall_s: Optional[float], core_s: Optional[float],
                final_inner_val_loss: Optional[float],
                is_formal_schedule: Optional[bool],
                device: str, run_dir: Optional[str] = None,
                manifest: Optional[str] = None,
                extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_id": str(run_id),
        "stage": str(stage),
        "backbone": str(backbone),
        "seed": int(seed),
        "outcome": str(outcome),
        "new_training": int(new_training),
        "reason": str(reason),
        "identity": (None if identity is None else dict(identity)),
        "identity_sha256": identity_sha256,
        "wall_s": wall_s,
        "core_s": core_s,
        "final_inner_val_loss": final_inner_val_loss,
        "is_formal_schedule": is_formal_schedule,
        "device": str(device),
        "run_dir": run_dir,
        "manifest": manifest,
    }
    if extra:
        row.update(dict(extra))
    return row


# ---------------------------------------------------------------------------
# gates: T0 completeness, sanity band, diagnostic, cost
# ---------------------------------------------------------------------------
def _t0_missing(state: Mapping[str, Mapping[str, Any]]) -> List[str]:
    return [run_id for run_id in T0_RUN_IDS
            if str((state.get(run_id) or {}).get("outcome")) != "done"]


def require_t0_complete(state: Mapping[str, Mapping[str, Any]]) -> None:
    """T0 A0x3 formal MUST all be ``done`` before any T1 (fix-5 C5)."""
    missing = _t0_missing(state)
    _require(not missing, "V542_FORMAL_T0_INCOMPLETE",
             "T0 A0x3 formal must ALL be done before any T1 formal run "
             f"(fix-5 C5 frozen order); not done: {missing}")


def read_band_record(campaign_root: Any) -> Optional[Dict[str, Any]]:
    """The ``V542-A0-sanity-band`` record written by ``sanity-band``."""
    path = Path(campaign_root) / BAND_RECORD_REL
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, Mapping) else None


def require_sanity_band_pass(band: Optional[Mapping[str, Any]]) -> None:
    """T1 requires the ``V542-A0-sanity-band`` PASS record (fix-5 C5)."""
    _require(band is not None, "V542_SANITY_BAND_ABSENT",
             f"the {A0_SANITY_BAND_GATE} record is absent — run "
             "`v542_formal.py sanity-band` after T0 A0x3 completes; no "
             "T1 formal launch without a band PASS")
    _require(str(band.get("verdict")) == "pass",
             "V542_SANITY_BAND_BREACH",
             f"{A0_SANITY_BAND_GATE} verdict is "
             f"{band.get('verdict')!r} — a breach STRICTLY blocks all T1 "
             "formal candidate launches (no exception may be signed); "
             f"breached endpoints: "
             f"{list(band.get('breached_endpoints') or [])}; T1 stays "
             "blocked pending the frozen migration audit "
             "(parity_certificate / process_head_anchor / r3_mapping / "
             "sampler / evaluator / code / data / initialization) and, if "
             "a defect is found, a per-defect-class identity bump "
             "(code=>source_tree_fingerprint, data=>data_manifest_sha256, "
             "init=>init_mapping_sha256, anchor=>"
             "process_head_anchor_sha256) + clean A0x3 rerun")


def _resolve_record_path(value: str, campaign_root: Any) -> Path:
    """Repo-relative first, then campaign-root-relative."""
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    repo = PROJECT_ROOT / candidate
    if repo.is_file():
        return repo
    return Path(campaign_root) / candidate


def diagnostic_disposition(campaign_root: Any, backbone: str
                           ) -> Dict[str, Any]:
    """Effective WP4 diagnostic disposition of one T1 candidate.

    Reads the CURRENT (v2) diagnostic report via the
    ``diagnostic_report_current.json`` pointer under the campaign root —
    the append-only corrected ledger of Oracle WP4 diagnostic ruling
    2026-10-30.  Missing pointer/report/row is a typed
    ``V542_FORMAL_DIAGNOSTIC_GATE_UNVERIFIED`` refusal (fail-closed:
    a T1 candidate can never advance on an unverified diagnostic gate).
    """
    pointer = Path(campaign_root) / "diagnostics" / (
        "diagnostic_report_current.json")
    _require(pointer.is_file(), "V542_FORMAL_DIAGNOSTIC_GATE_UNVERIFIED",
             f"diagnostic gate cannot be verified: {pointer} is missing "
             "(the effective WP4 diagnostic disposition record is required "
             "before any T1 candidate launch)")
    doc = json.loads(pointer.read_text(encoding="utf-8"))
    effective = _resolve_record_path(
        (doc or {}).get("effective_report", ""), campaign_root)
    _require(effective.is_file(), "V542_FORMAL_DIAGNOSTIC_GATE_UNVERIFIED",
             f"diagnostic gate cannot be verified: effective report "
             f"{effective} is missing")
    want_run = f"V542_DIAG_{backbone}_S17"
    row = None
    for raw in effective.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        entry = json.loads(raw)
        if str(entry.get("run_id")) == want_run:
            row = entry
    _require(row is not None, "V542_FORMAL_DIAGNOSTIC_GATE_UNVERIFIED",
             f"diagnostic gate cannot be verified: {want_run} has no row "
             f"in {effective}")
    disposition = str(row.get("gate_disposition")
                      or row.get("decision") or "")
    return {"disposition": disposition,
            "reason_code": str(row.get("reason_code") or ""),
            "report": str(effective)}


def cost_gate_ratio(cfg: Mapping[str, Any], backbone: str
                    ) -> Dict[str, Any]:
    """Signed cost gate for one candidate (denominator = V542 A0 profile).

    Ratio comes from the owner-signed ``budget.profile_signoff`` block
    (fallback: the signed ``budget.per_candidate_core_time`` table);
    a ratio that cannot be verified is a typed refusal (fail-closed) and
    a ratio > ``thresholds.max_core_time_ratio`` maps to
    ``gate_skipped(gate_skipped_cost_ratio_exceeds_2)`` — science kept,
    never a ``resource_failure``.
    """
    th = dict(cfg.get("thresholds") or {})
    max_ratio = float(th.get("max_core_time_ratio", 2.0))
    budget = dict(cfg.get("budget") or {})
    signoff = dict(budget.get("profile_signoff") or {})
    ratios = dict(signoff.get("cost_ratio_vs_a0") or {})
    ratio = ratios.get(backbone)
    if ratio is None:
        table = dict(budget.get("per_candidate_core_time") or {})
        cand, a0 = table.get(backbone), table.get("A0")

        def _num(value: Any) -> Optional[float]:
            if isinstance(value, Mapping):
                value = value.get("core_seconds")
            return (float(value)
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool) else None)

        cand_n, a0_n = _num(cand), _num(a0)
        _require(cand_n is not None and a0_n is not None and a0_n > 0.0,
                 "V542_FORMAL_COST_GATE_UNVERIFIED",
                 f"cost gate for {backbone} cannot be verified: signed "
                 f"budget carries neither profile_signoff.cost_ratio_vs_a0"
                 f"[{backbone!r}] nor a usable per_candidate_core_time "
                 f"entry (got {backbone!r}={cand!r}, A0={a0!r})")
        ratio = cand_n / a0_n
    return {"ratio": float(ratio), "max_core_time_ratio": max_ratio,
            "within": float(ratio) <= max_ratio}


# ---------------------------------------------------------------------------
# stage decisions
# ---------------------------------------------------------------------------
@dataclass
class RunDecision:
    """One per-run scheduling decision (launch vs gate_skipped)."""

    run_id: str
    stage: str
    backbone: str
    seed: int
    action: str                       # "launch" | "gate_skipped" | "skip"
    reason: str
    new_training: int
    winner_eligible: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def _gate_skipped(row: RunRow, reason: str,
                  **extra: Any) -> RunDecision:
    return RunDecision(row.run_id, row.stage, row.backbone, row.seed,
                       "gate_skipped", reason, 0,
                       winner_eligible=row.winner_eligible,
                       extra=dict(extra))


def _launch(row: RunRow, reason: str = REASON_FORMAL_DONE) -> RunDecision:
    return RunDecision(row.run_id, row.stage, row.backbone, row.seed,
                       "launch", reason, 1,
                       winner_eligible=row.winner_eligible)


def load_top2_freeze(campaign_root: Any) -> Dict[str, Any]:
    """Read the WP6 seed17 screen report and return the frozen top2 set.

    A T3 top-2 row may launch only when (a) the screen report exists, (b)
    its ``top2_freeze_precondition`` is asserted (A0_S17 complete + the
    ``V542-A0-sanity-band`` pass), and (c) the candidate is IN that set.
    Anything else is a typed refusal — never a silent fallback.
    """
    path = Path(campaign_root) / TOP2_SCREEN_REL
    _require(path.is_file(), "V542_TOP2_SCREEN_ABSENT",
             f"the WP6 seed17 screen report is missing ({path}); "
             "T3 top-2 rows cannot launch before the screen is frozen")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V542FormalError(
            "V542_TOP2_SCREEN_ABSENT",
            f"the WP6 seed17 screen report is unreadable ({path}): {exc}"
        ) from exc
    pre = doc.get("top2_freeze_precondition") or {}
    _require(pre.get("asserted") is True
             and str(pre.get("sanity_band_verdict")) == "pass"
             and pre.get("a0_s17_done") is True,
             "V542_TOP2_SCREEN_ABSENT",
             f"{path} does not assert the frozen top2 freeze "
             f"precondition (a0_s17_done + sanity-band pass)")
    top2 = list(doc.get("top2") or [])
    _require(len(top2) <= 2,
             "V542_TOP2_SCREEN_ABSENT",
             f"{path} carries {len(top2)} top2 candidates; the frozen "
             "maximum is 2")
    return {"top2": top2,
            "report_path": str(path),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def decide_stage_runs(stage: str, rows: Sequence[RunRow], *,
                      state: Mapping[str, Mapping[str, Any]],
                      band: Optional[Mapping[str, Any]],
                      cfg: Mapping[str, Any],
                      campaign_root: Any) -> List[RunDecision]:
    """Frozen per-run decisions for one stage (no training here).

    Raises the typed ordering/gating refusals (``V542_FORMAL_T0_INCOMPLETE``
    / ``V542_SANITY_BAND_ABSENT`` / ``V542_SANITY_BAND_BREACH``) BEFORE
    any decision is returned, so a refused T1 stage touches nothing.
    """
    decisions: List[RunDecision] = []
    if stage == "T1":
        require_t0_complete(state)
        require_sanity_band_pass(band)
    for row in rows:
        if stage == "T0":
            # A0 matched-control rows: mandatory, no diagnostic gate.
            decisions.append(_launch(row))
        elif stage == "T1":
            if row.backbone == "A4":
                # WP4 binding ruling: A4 NEVER runs at full dose.
                decisions.append(_gate_skipped(
                    row, REASON_A4_GATE_SKIPPED,
                    ruling="ADR-0033 v1.3.3 WP4 diagnostic ruling "
                           "(3.849742% < 5.0% frozen threshold, binding)",
                    frozen_threshold_pct=5.0, measured_drop_pct=3.849742))
                continue
            diag = diagnostic_disposition(campaign_root, row.backbone)
            if diag["disposition"] != "advance_full_dose":
                decisions.append(_gate_skipped(
                    row, diag["reason_code"] or diag["disposition"],
                    diagnostic_disposition=diag["disposition"],
                    diagnostic_report=diag["report"]))
                continue
            cost = cost_gate_ratio(cfg, row.backbone)
            if not cost["within"]:
                decisions.append(_gate_skipped(
                    row, REASON_COST_EXCEEDS,
                    core_time_ratio=cost["ratio"],
                    max_core_time_ratio=cost["max_core_time_ratio"]))
                continue
            decisions.append(_launch(row))
        elif stage == "T2":
            decisions.append(_gate_skipped(
                row, REASON_A5_NOT_ACTIVATED,
                frozen_condition=A5_FROZEN_CONDITION,
                not_evaluable_before="WP6 (top2 seeds + final H3)"))
        elif stage == "T3":
            if row.backbone == "A4":
                decisions.append(_gate_skipped(
                    row, REASON_A4_GATE_SKIPPED,
                    ruling="ADR-0033 v1.3.3: A4 S29/S43 not activated "
                           "(same binding diagnostic reason)"))
            elif row.backbone == "A5":
                decisions.append(_gate_skipped(
                    row, REASON_A5_SUPPLEMENT,
                    frozen_condition=A5_SUPPLEMENT_CONDITION,
                    parent_gate_unmet_since=("T2 V542_A5_S17 is "
                                             "gate_skipped")))
            else:
                frozen_top2 = load_top2_freeze(campaign_root)
                if row.backbone not in frozen_top2["top2"]:
                    decisions.append(_gate_skipped(
                        row, REASON_TOP2_NOT_SELECTED,
                        frozen_condition="activated_only_by_inner_val_top2",
                        top2=frozen_top2["top2"],
                        screen_report=frozen_top2["report_path"]))
                    continue
                cost = cost_gate_ratio(cfg, row.backbone)
                if not cost["within"]:
                    decisions.append(_gate_skipped(
                        row, REASON_COST_EXCEEDS,
                        core_time_ratio=cost["ratio"],
                        max_core_time_ratio=cost["max_core_time_ratio"]))
                    continue
                decisions.append(_launch(row))
        else:  # T-ATTR
            reason = (REASON_A1C_SKIP if row.backbone == "A1-C"
                      else REASON_A5C_SKIP)
            condition = (row.condition or "")
            decisions.append(_gate_skipped(
                row, reason, frozen_condition=condition,
                activated_only_by=("the frozen activation predicate "
                                   "(post-WP7 selection record) — never "
                                   "satisfied before WP7")))
    return decisions


# ---------------------------------------------------------------------------
# per-run execution (frozen trainer, frozen dose, resume-aware)
# ---------------------------------------------------------------------------
def _assert_launch_allowed(row: RunRow) -> None:
    """Structural guard: A4 / conditional rows can never be launched."""
    allowed = LAUNCHABLE.get(row.stage, ())
    _require(row.backbone in allowed, "V542_FORMAL_LAUNCH_FORBIDDEN",
             f"this runner never launches {row.run_id} "
             f"(stage={row.stage}, backbone={row.backbone}); launchable "
             f"rows are exactly {[f'{s}:{c}' for s, cs in LAUNCHABLE.items()
                                  for c in cs]} — A4 is gate_skipped by "
             "the binding WP4 ruling and conditional rows stay "
             "gate_skipped until their frozen predicates are evaluable")


def _trainer_config_for(row: RunRow, cfg_mapping: Mapping[str, Any], *,
                        device: str,
                        test_dose: Optional[Sequence[int]] = None,
                        test_smoke_updates: Optional[int] = None) -> Any:
    """The frozen-surface :class:`V542TrainerConfig` of one formal row.

    ``test_dose``/``test_smoke_updates`` are TEST-ONLY seams (never
    reachable from the CLI) recorded honestly on the ledger row; the
    frozen production dose is always 400 + 37,500 from the config.
    """
    from scripts.agent_ops.v542_campaign import (
        _schedule_windows,
        _trainer_config,
    )

    windows = _schedule_windows(cfg_mapping)
    w0, w1 = windows["w0_updates"], windows["w1_updates"]
    if test_dose is not None:
        w0, w1 = int(test_dose[0]), int(test_dose[1])
    return _trainer_config(
        cfg_mapping, candidate_id=row.backbone, seed=row.seed,
        freeze=w0, joint=w1, smoke_updates=test_smoke_updates,
        device=device)


def execute_formal_run(env: Any, row: RunRow, *,
                       cfg_mapping: Mapping[str, Any],
                       config_path: Any,
                       campaign_root: Any,
                       device: str,
                       identity: Mapping[str, Any],
                       test_dose: Optional[Sequence[int]] = None,
                       test_smoke_updates: Optional[int] = None
                       ) -> Dict[str, Any]:
    """Train ONE formal row at the frozen dose via the frozen trainer.

    Checkpoints land under ``<campaign_root>/runs/<run_id>/``; resume
    continues from ``checkpoint_last.pt`` (identity-checked by the
    trainer).  A resume-identity mismatch surfaces as the trainer's typed
    ``invalid_run`` refusal — recorded in the ledger with
    ``new_training=0`` and re-raised as a typed formal refusal (the
    checkpoint is untouched: the trainer refuses BEFORE any write).
    """
    from pes2ts_core.pesn.offline.training.v542_train import (
        RUN_OUTCOMES_V542,
        V542TrainerError,
        train_run_v542,
    )

    _assert_launch_allowed(row)
    run_dir = Path(campaign_root) / "runs" / row.run_id
    config = _trainer_config_for(
        row, cfg_mapping, device=device, test_dose=test_dose,
        test_smoke_updates=test_smoke_updates)
    try:
        metrics = train_run_v542(
            env.ds, env.labels, env.endpoint, config, run_dir,
            dev_positions=env.dev_pos, identity=dict(identity),
            require_identity=True, resume=True)
    except V542TrainerError as exc:
        outcome = getattr(exc, "outcome", "invalid_run")
        _require(outcome in RUN_OUTCOMES_V542, "V542_FORMAL_RUN_FAILED",
                 f"frozen trainer refusal with unknown outcome "
                 f"{outcome!r}: {exc}")
        manifest_path = run_dir / "run_manifest.json"
        append_ledger_row(campaign_root, _ledger_row(
            run_id=row.run_id, stage=row.stage, backbone=row.backbone,
            seed=row.seed, outcome=outcome, new_training=0,
            reason=f"{getattr(exc, 'code', 'TRAINER_ERROR')}: {exc}",
            identity=identity,
            identity_sha256=_identity_digest(identity), wall_s=None,
            core_s=None, final_inner_val_loss=None,
            is_formal_schedule=(test_dose is None
                                and test_smoke_updates is None),
            device=device, run_dir=str(run_dir),
            manifest=str(manifest_path),
            extra={"trainer_code": str(getattr(exc, "code", ""))}))
        raise V542FormalError(
            str(getattr(exc, "code", "V542_FORMAL_RUN_FAILED")),
            f"frozen trainer refused run {row.run_id} "
            f"(outcome={outcome}, checkpoint untouched): {exc}",
            outcome=outcome) from exc

    outcome = str(metrics.get("run_outcome"))
    _require(outcome in RUN_OUTCOMES_V542, "V542_FORMAL_RUN_FAILED",
             f"trainer returned outcome {outcome!r} outside the frozen "
             f"six-word vocabulary {list(RUN_OUTCOMES_V542)}")
    phases = (metrics.get("timing") or {}).get("phases") or {}
    core_s = round(sum(float(p.get("core_s", 0.0) or 0.0)
                       for p in phases.values()), 3)
    wall_s = (metrics.get("timing") or {}).get("wall_clock_s")
    identity_sha = metrics.get("identity_sha256")
    is_formal = bool(metrics.get("is_formal_schedule"))
    manifest_path = run_dir / "run_manifest.json"
    row_extra = {
        "is_formal_schedule": is_formal,
        "dose": {"w0_updates": int(config.effective_freeze_updates),
                 "w1_updates": int(config.joint_updates),
                 "total_updates": int(config.total_updates),
                 "val_every": int(config.val_every)},
        "test_dose": (None if test_dose is None
                      else [int(test_dose[0]), int(test_dose[1])]),
        "test_smoke_updates": (None if test_smoke_updates is None
                               else int(test_smoke_updates)),
        "selected_update": metrics.get("selected_update"),
        "best_update": metrics.get("best_update"),
    }
    ledger_row = append_ledger_row(campaign_root, _ledger_row(
        run_id=row.run_id, stage=row.stage, backbone=row.backbone,
        seed=row.seed, outcome=outcome,
        new_training=1 if outcome in ("done", "scientific_failure") else 0,
        reason=REASON_FORMAL_DONE if outcome == "done" else outcome,
        identity=metrics.get("identity"), identity_sha256=identity_sha,
        wall_s=wall_s, core_s=core_s,
        final_inner_val_loss=metrics.get("best_inner_val_loss"),
        is_formal_schedule=is_formal, device=device,
        run_dir=str(run_dir), manifest=str(manifest_path),
        extra=row_extra))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "run_id": row.run_id,
        "stage": row.stage,
        "backbone": row.backbone,
        "seed": row.seed,
        "winner_eligible": row.winner_eligible,
        "outcome": outcome,
        "reason": ledger_row["reason"],
        "identity": metrics.get("identity"),
        "identity_sha256": identity_sha,
        "config_path": str(config_path),
        "config_identity_sha256": metrics.get("config_sha256"),
        "config_full_sha256": metrics.get("config_full_sha256"),
        "dose": row_extra["dose"],
        "is_formal_schedule": is_formal,
        "updates_run": metrics.get("updates_run"),
        "best_inner_val_loss": metrics.get("best_inner_val_loss"),
        "best_update": metrics.get("best_update"),
        "checkpoint_best_path": metrics.get("checkpoint_best_path"),
        "checkpoint_best_sha256": metrics.get("checkpoint_best_sha256"),
        "checkpoint_last_path": metrics.get("checkpoint_last_path"),
        "selection_metric": "inner-val total loss",
        "h3_metric_selection_forbidden": True,
        "wall_s": wall_s,
        "core_s": core_s,
        "ledger_seq": ledger_row["seq"],
        "torch_compile": False,
        "qc_calls": 0,
        "confirm_read": False,
        "written_at": _now(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return {"ledger_row": ledger_row, "manifest": manifest,
            "metrics": metrics}


# ---------------------------------------------------------------------------
# run subcommand orchestration
# ---------------------------------------------------------------------------
def _assemble_identity(cfg: Mapping[str, Any], *, config_path: Any,
                       candidate_id: str, run_seed: int,
                       runplan_path: Any = None,
                       budget_path: Any = None,
                       require_clean: bool = False) -> Dict[str, Any]:
    """Assemble the frozen 14-field run identity for one formal run.

    The CLI must never launch a formal run without an identity block
    (``V542_FORMAL_IDENTITY_MISSING``); the identity comes from
    :func:`pes2ts_core.pesn.models.v542.identity.assemble_run_identity`
    with the frozen runplan/budget records, so two launches of the same
    run seed produce the same digest.
    """
    from pes2ts_core.pesn.models.v542.identity import assemble_run_identity

    runplan = _load_json_record(
        runplan_path or DEFAULT_RUNPLAN, "runplan")
    budget = (_load_json_record(budget_path, "budget")
              if budget_path is not None else cfg.get("budget"))
    return assemble_run_identity(
        cfg,
        candidate_id=str(candidate_id),
        run_seed=int(run_seed),
        runplan_record=runplan,
        budget_block=budget,
        root=PROJECT_ROOT,
        config_path=config_path,
        require_clean=bool(require_clean),
    )


def _load_json_record(path: Any, label: str) -> Mapping[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise V542FormalError(
            "V542_FORMAL_IDENTITY_MISSING",
            f"{label} record not found: {p}")
    try:
        record = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V542FormalError(
            "V542_FORMAL_IDENTITY_MISSING",
            f"{label} record unreadable: {p}: {exc}") from exc
    if not isinstance(record, Mapping):
        raise V542FormalError(
            "V542_FORMAL_IDENTITY_MISSING",
            f"{label} record is not a JSON object: {p}")
    return record

def run_stage(stage: str, *, cfg: Mapping[str, Any],
              config_path: Any = DEFAULT_CONFIG,
              runplan_path: Any = None,
              budget_path: Any = None,
              require_clean: bool = False,
              campaign_root: Any = DEFAULT_CAMPAIGN_ROOT,
              device: str = "cpu",
              only: Optional[Sequence[str]] = None,
              dry_run: bool = False,
              env: Any = None,
              identity_by_run: Optional[Mapping[str, Any]] = None,
              test_dose: Optional[Sequence[int]] = None,
              test_smoke_updates: Optional[int] = None,
              formal_validated: bool = False) -> Dict[str, Any]:
    """Decide (+ execute) every eligible row of one formal stage.

    The formal validation gate runs FIRST (unless the caller already ran
    it): any frozen-validator rejection is a typed refusal with the
    verbatim code.  ``--dry-run`` prints the decisions WITHOUT training
    and WITHOUT writing (no ledger, no run dirs).
    """
    if not formal_validated:
        formal_gate(config_path)
    rows = filter_only(stage_rows(cfg, stage), only)
    state = latest_ledger_state(read_ledger(campaign_root))
    band = read_band_record(campaign_root)
    decisions = decide_stage_runs(
        stage, rows, state=state, band=band, cfg=cfg,
        campaign_root=campaign_root)

    plan = {
        "schema": FORMAL_SCHEMA,
        "kind": "formal_run",
        "stage": stage,
        "dry_run": bool(dry_run),
        "training_performed": False,
        "device": str(device),
        "campaign_root": str(campaign_root),
        "ledger_path": str(ledger_path(campaign_root)),
        "sanity_band": (None if band is None
                        else {"verdict": band.get("verdict"),
                              "record": BAND_RECORD_REL}),
        "steps": [{
            "step": index,
            "run_id": d.run_id,
            "backbone": d.backbone,
            "seed": d.seed,
            "action": d.action,
            "reason": d.reason,
            "new_training": d.new_training,
            "winner_eligible": d.winner_eligible,
            "run_dir": str(Path(campaign_root) / "runs" / d.run_id),
            **({"detail": d.extra} if d.extra else {}),
        } for index, d in enumerate(decisions, start=1)],
        "torch_compile": False,
        "qc_calls": 0,
        "confirm_read": False,
    }
    if dry_run:
        return plan

    executed: List[Dict[str, Any]] = []
    recorded: List[str] = []
    for decision, row in zip(decisions, rows):
        if decision.action == "launch":
            identity = (identity_by_run or {}).get(decision.run_id)
            if identity is None:
                # CLI path: assemble the frozen 14-field identity.  A run
                # must never launch without it.
                identity = _assemble_identity(
                    cfg, config_path=config_path,
                    candidate_id=decision.backbone,
                    run_seed=decision.seed,
                    runplan_path=runplan_path,
                    budget_path=budget_path,
                    require_clean=require_clean)
            _require(identity is not None, "V542_FORMAL_IDENTITY_MISSING",
                     f"no run-identity block provided for "
                     f"{decision.run_id}")
            if env is None:
                from scripts.agent_ops.v542_campaign import load_campaign_env

                env = load_campaign_env(cfg)
            latest = (latest_ledger_state(
                read_ledger(campaign_root)).get(decision.run_id) or {})
            if str(latest.get("outcome")) == "done":
                plan.setdefault("skipped_already_done", []).append(
                    decision.run_id)
                continue
            executed.append(execute_formal_run(
                env, row, cfg_mapping=cfg, config_path=config_path,
                campaign_root=campaign_root, device=device,
                identity=identity, test_dose=test_dose,
                test_smoke_updates=test_smoke_updates))
        else:
            latest = (latest_ledger_state(
                read_ledger(campaign_root)).get(decision.run_id) or {})
            if (str(latest.get("outcome")) == "gate_skipped"
                    and str(latest.get("reason")) == decision.reason):
                plan.setdefault("already_recorded", []).append(
                    decision.run_id)
                continue
            append_ledger_row(campaign_root, _ledger_row(
                run_id=decision.run_id, stage=decision.stage,
                backbone=decision.backbone, seed=decision.seed,
                outcome="gate_skipped", new_training=0,
                reason=decision.reason, identity=None,
                identity_sha256=None, wall_s=None, core_s=None,
                final_inner_val_loss=None, is_formal_schedule=False,
                device=device, extra=decision.extra))
            recorded.append(decision.run_id)
    plan["training_performed"] = bool(executed)
    plan["executed"] = [e["ledger_row"]["run_id"] for e in executed]
    plan["gate_skipped_recorded"] = recorded
    return plan


# ---------------------------------------------------------------------------
# sanity band: evaluation legs + the frozen band rule
# ---------------------------------------------------------------------------
def _band_endpoints_from_eval(ev: Mapping[str, Any]) -> Dict[str, float]:
    """The three band endpoints out of a frozen ``evaluate_v53mp`` doc."""
    values = {
        "rmse_zero": (((ev.get("move") or {}).get(
            "key_rmse_over_zero") or {}).get("median_rmse_over_zero")),
        "endpoint_auprc": (ev.get("endpoint_head") or {}).get("auprc"),
        "q_corr": (ev.get("q_head") or {}).get("corr_curve_median"),
    }
    for key, value in values.items():
        _require(isinstance(value, (int, float))
                 and not isinstance(value, bool)
                 and float(value) == float(value),
                 "V542_BAND_EVALUATION_INVALID",
                 f"frozen evaluator output misses a finite band endpoint "
                 f"{key!r} (got {value!r})")
    return {key: float(value) for key, value in values.items()}


def compute_sanity_band(
        a0_by_seed: Mapping[int, Mapping[str, float]],
        v541_by_seed: Mapping[int, Mapping[str, float]], *,
        rmse_zero_rel_max_pct: float = 2.0,
        auprc_abs_max: float = 0.01,
        q_corr_abs_max: float = 0.01) -> Dict[str, Any]:
    """The frozen band rule on PER-ENDPOINT SEED MEANS (never per-seed).

    * ``rmse_zero`` — relative difference of the two three-seed MEANS
      must be ``<= rmse_zero_rel_max_pct`` (2%);
    * ``endpoint_auprc`` / ``q_corr`` — absolute difference of the two
      three-seed MEANS must be ``<= 0.01`` each.

    Verdict ``pass`` iff EVERY endpoint is within band; otherwise
    ``breach`` with the breached endpoint list.  All six per-seed values
    must be present and finite (missing/non-finite = typed refusal —
    values are never fabricated).
    """
    import math

    def _finite(value: Any) -> bool:
        return (isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)))

    endpoints = ("rmse_zero", "endpoint_auprc", "q_corr")

    def _normalize(table: Mapping[Any, Mapping[str, float]],
                   side: str) -> Dict[int, Mapping[str, float]]:
        out: Dict[int, Mapping[str, float]] = {}
        for key, value in dict(table).items():
            try:
                out[int(key)] = value            # accept "17" as well
            except (TypeError, ValueError):  # pragma: no cover - guarded
                raise V542FormalError(
                    "V542_BAND_SEEDS_INCOMPLETE",
                    f"{side} side carries a non-integer seed key {key!r}")
        return out

    a0_by_seed = _normalize(a0_by_seed, "a0")
    v541_by_seed = _normalize(v541_by_seed, "v541_e0")
    for side, table in (("a0", a0_by_seed), ("v541_e0", v541_by_seed)):
        for seed in FROZEN_SEEDS:
            entry = table.get(seed) or table.get(str(seed))
            _require(isinstance(entry, Mapping),
                     "V542_BAND_SEEDS_INCOMPLETE",
                     f"{side} side is missing the seed-{seed} endpoints "
                     f"(all three frozen seeds {list(FROZEN_SEEDS)} are "
                     "required for the seed-mean band)")
            for key in endpoints:
                _require(_finite(entry.get(key)),
                         "V542_BAND_SEEDS_INCOMPLETE",
                         f"{side} seed {seed} endpoint {key!r} is "
                         f"missing/non-finite (got {entry.get(key)!r})")

    def _mean(side: Mapping[Any, Mapping[str, float]],
              key: str) -> float:
        return sum(float(side[seed][key]) for seed in FROZEN_SEEDS) / 3.0

    a0_means = {k: _mean(a0_by_seed, k) for k in endpoints}
    e0_means = {k: _mean(v541_by_seed, k) for k in endpoints}

    results: Dict[str, Dict[str, Any]] = {}
    breached: List[str] = []

    # rmse_zero: RELATIVE band on the seed means (E0 mean is the base).
    _require(e0_means["rmse_zero"] > 0.0, "V542_BAND_REFERENCE_INVALID",
             "V541 E0 rmse_zero seed-mean must be > 0 to form the "
             f"relative band (got {e0_means['rmse_zero']!r})")
    rel_pct = (a0_means["rmse_zero"] - e0_means["rmse_zero"]) \
        / e0_means["rmse_zero"] * 100.0
    rmse_ok = abs(rel_pct) <= float(rmse_zero_rel_max_pct)
    results["rmse_zero"] = {
        "a0_by_seed": {str(s): float(a0_by_seed[s]["rmse_zero"])
                       for s in FROZEN_SEEDS},
        "v541_e0_by_seed": {str(s): float(v541_by_seed[s]["rmse_zero"])
                            for s in FROZEN_SEEDS},
        "a0_seed_mean": a0_means["rmse_zero"],
        "v541_e0_seed_mean": e0_means["rmse_zero"],
        "relative_diff_pct": rel_pct,
        "rule": "relative seed-mean difference <= {}%".format(
            rmse_zero_rel_max_pct),
        "max_pct": float(rmse_zero_rel_max_pct),
        "within_band": bool(rmse_ok),
    }
    if not rmse_ok:
        breached.append("rmse_zero")

    for key, limit in (("endpoint_auprc", auprc_abs_max),
                       ("q_corr", q_corr_abs_max)):
        diff = a0_means[key] - e0_means[key]
        ok = abs(diff) <= float(limit)
        results[key] = {
            "a0_by_seed": {str(s): float(a0_by_seed[s][key])
                           for s in FROZEN_SEEDS},
            "v541_e0_by_seed": {str(s): float(v541_by_seed[s][key])
                                for s in FROZEN_SEEDS},
            "a0_seed_mean": a0_means[key],
            "v541_e0_seed_mean": e0_means[key],
            "absolute_diff": diff,
            "rule": "absolute seed-mean difference <= {}".format(limit),
            "max_abs": float(limit),
            "within_band": bool(ok),
        }
        if not ok:
            breached.append(key)

    return {
        "verdict": "pass" if not breached else "breach",
        "breached_endpoints": breached,
        "endpoints": results,
        "rule": {
            "rmse_zero_rel_max_pct": float(rmse_zero_rel_max_pct),
            "auprc_abs_max": float(auprc_abs_max),
            "q_corr_abs_max": float(q_corr_abs_max),
            "statistic": EVALUATOR_IDENTITY["statistic"],
        },
    }


def _band_inner_val_positions(cfg_mapping: Mapping[str, Any],
                              dataset: Any, seed: int) -> List[int]:
    """The seed's inner-val slice — SAME derivation as the trainer."""
    import numpy as np

    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        v53mp_pool_split,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    split_path = str((cfg_mapping.get("split_contract") or {}).get(
        "split_manifest")
        or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
    split_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    _train_pos, val_pos = v53mp_pool_split(
        dataset.samples, pool_curve_ids=pool_ids, seed=int(seed))
    positions = [int(p) for p in val_pos]
    cap = (cfg_mapping.get("split_contract") or {}).get(
        "inner_val_max_samples")
    if cap and len(positions) > int(cap):
        rng = np.random.RandomState(int(seed) ^ 0x1E7A1)
        positions = sorted(int(p) for p in rng.choice(
            list(positions), size=int(cap), replace=False).tolist())
    return positions


def evaluate_a0_band_endpoints(cfg_mapping: Mapping[str, Any],
                               campaign_root: Any, seed: int, *,
                               device: str = "cpu") -> Dict[str, float]:
    """Evaluate ONE T0 A0 formal checkpoint with the FROZEN evaluator.

    Loads ``<campaign_root>/runs/V542_A0_S<seed>/checkpoint_best.pt``,
    strict-loads it into a freshly assembled A0 (the frozen construction
    order), and runs ``evaluate_v53mp`` over the seed's inner-val
    positions.  ``tau_end=None`` (tau never touches the band endpoints).
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
    from pes2ts_core.pesn.offline.training.v542_train import (
        V542_TRAIN_SCHEMA,
    )
    from scripts.agent_ops.v542_campaign import load_campaign_env

    run_dir = Path(campaign_root) / "runs" / f"V542_A0_S{seed}"
    ckpt = run_dir / "checkpoint_best.pt"
    metrics_path = run_dir / "metrics.json"
    _require(ckpt.is_file() and metrics_path.is_file(),
             "V542_BAND_A0_RUN_INCOMPLETE",
             f"T0 A0 run seed {seed} has no complete artifacts under "
             f"{run_dir} (need checkpoint_best.pt + metrics.json)")
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _require(run_metrics.get("run_outcome") == "done",
             "V542_BAND_A0_RUN_INCOMPLETE",
             f"T0 A0 run seed {seed} outcome is "
             f"{run_metrics.get('run_outcome')!r}, not 'done'")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    _require(isinstance(state, Mapping)
             and str(state.get("schema")) == V542_TRAIN_SCHEMA
             and str(state.get("candidate_id")) == "A0"
             and int((state.get("identity") or {}).get("seed", -1))
             == int(seed),
             "V542_BAND_A0_RUN_INCOMPLETE",
             f"{ckpt} is not the seed-{seed} A0 formal checkpoint "
             f"(schema={state.get('schema')!r} "
             f"candidate={state.get('candidate_id')!r})")

    env = load_campaign_env(cfg_mapping)
    assembled = assemble_v542_model(
        "A0", int(seed), device=str(device), require_identity=False)
    assembled.model.load_state_dict(state["model"], strict=True)
    assembled.model.train(False)

    positions = _band_inner_val_positions(cfg_mapping, env.ds, seed)
    lru = _FeatureLRU(env.ds, capacity=len(positions) + 64)
    cache = _CacheView(lru)
    train_pos, _val = _band_train_positions(cfg_mapping, env.ds, seed)
    eval_cfg = V53MPTrainerConfig(
        arm="A", seed=int(seed), device=str(device))
    report = evaluate_v53mp(
        assembled.model, None, env.ds, env.labels, env.endpoint, cache,
        positions, eval_cfg,
        train_q_mean=_train_q_mean(env.ds.samples, env.labels, train_pos),
        tau_end=None)
    endpoints = _band_endpoints_from_eval(report)
    endpoints["_n_inner_val_positions"] = float(len(positions))
    return endpoints


def _band_train_positions(cfg_mapping: Mapping[str, Any],
                          dataset: Any, seed: int) -> Any:
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        v53mp_pool_split,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    split_path = str((cfg_mapping.get("split_contract") or {}).get(
        "split_manifest")
        or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
    split_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    train_pos, val_pos = v53mp_pool_split(
        dataset.samples, pool_curve_ids=pool_ids, seed=int(seed))
    return train_pos, val_pos


def locate_v541_e0_reference(cfg: Mapping[str, Any], *,
                             v541_runs_root: Any = DEFAULT_V541_RUNS_ROOT
                             ) -> Dict[int, Dict[str, Any]]:
    """The V541 E0×3 archived reference — located, never fabricated.

    The run dirs come from the config's PINNED parity assets
    (``V541_E0_S17/S29/S43``); the three band endpoints are extracted
    from each run's archived ``metrics.json`` (the same frozen-evaluator
    metric leaves), and the provenance (metrics path + SHA-256, checkpoint
    pin cross-check) is recorded.  The archived checkpoint SHA must equal
    the config-pinned asset SHA, and the archived run outcome must be
    ``done`` — anything else is a typed refusal.
    """
    import hashlib

    assets = ((cfg.get("backbone_contract") or {}).get("a0_migration")
              or {}).get("parity") or {}
    pinned = {str(a.get("id")): a for a in (assets.get("assets") or [])
              if isinstance(a, Mapping)}
    root = Path(v541_runs_root)
    out: Dict[int, Dict[str, Any]] = {}
    for seed in FROZEN_SEEDS:
        asset_id = f"V541_E0_S{seed}"
        asset = pinned.get(asset_id)
        _require(asset is not None, "V542_BAND_REFERENCE_INVALID",
                 f"config parity assets miss the pinned {asset_id} row "
                 "(the V541 E0 reference must come from the pinned "
                 "runs/ assets)")
        run_dir = root / _E0_RUN_PATTERN.format(seed=seed)
        metrics_path = run_dir / "metrics.json"
        _require(metrics_path.is_file(), "V542_BAND_REFERENCE_INVALID",
                 f"V541 E0 seed {seed} archived metrics not found at "
                 f"{metrics_path} — the reference is LOCATED from the "
                 "V541 run artifacts, never fabricated")
        raw = metrics_path.read_bytes()
        doc = json.loads(raw.decode("utf-8"))
        _require(isinstance(doc, Mapping)
                 and doc.get("run_outcome") == "done",
                 "V542_BAND_REFERENCE_INVALID",
                 f"V541 E0 seed {seed} archived run outcome is "
                 f"{doc.get('run_outcome')!r}, not 'done'")
        ckpt_pin = str(asset.get("sha256") or "")
        archived_ckpt_sha = doc.get("checkpoint_best_sha256")
        _require(archived_ckpt_sha == ckpt_pin,
                 "V542_BAND_REFERENCE_INVALID",
                 f"V541 E0 seed {seed} archived metrics name checkpoint "
                 f"SHA {archived_ckpt_sha!r} which != the config-pinned "
                 f"asset SHA {ckpt_pin!r}")
        endpoints = _band_endpoints_from_eval(doc)
        out[int(seed)] = {
            "endpoints": endpoints,
            "provenance": {
                "metrics_path": str(metrics_path),
                "metrics_sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint_path": str(asset.get("path")),
                "checkpoint_sha256_pinned": ckpt_pin,
                "run_id": _E0_RUN_PATTERN.format(seed=seed),
                "source_tier": str(asset.get("source_tier")),
                "run_updates": asset.get("run_updates"),
                "checkpoint_update": asset.get("checkpoint_update"),
            },
        }
    return out


def build_band_record(a0_by_seed: Mapping[int, Mapping[str, float]],
                      v541_reference: Mapping[int, Mapping[str, Any]],
                      band: Mapping[str, Any], *,
                      cfg: Mapping[str, Any],
                      a0_identities: Optional[Mapping[str, Any]] = None
                      ) -> Dict[str, Any]:
    """The ``a0_sanity_band.json`` document (verdict + numbers + IDs)."""
    a0_sources = [{
        "run_id": f"V542_A0_S{seed}",
        "seed": int(seed),
        "identity_sha256": (a0_identities or {}).get(
            f"V542_A0_S{seed}"),
        "checkpoint_path": str(
            Path("runs") / f"V542_A0_S{seed}" / "checkpoint_best.pt"),
        "n_inner_val_positions": a0_by_seed[seed].get(
            "_n_inner_val_positions"),
    } for seed in FROZEN_SEEDS]
    record: Dict[str, Any] = {
        "schema": BAND_SCHEMA,
        "gate": A0_SANITY_BAND_GATE,
        "campaign_id": str(cfg.get("campaign_id")),
        "verdict": str(band["verdict"]),
        "breached_endpoints": list(band["breached_endpoints"]),
        "rule": dict(band["rule"]),
        "endpoints": dict(band["endpoints"]),
        "evaluator_identity": dict(EVALUATOR_IDENTITY),
        "a0_source_runs": a0_sources,
        "v541_e0_reference": [
            {"seed": int(seed),
             **dict(v541_reference[seed]["provenance"])}
            for seed in FROZEN_SEEDS],
        "written_at": _now(),
    }
    if band["verdict"] == "pass":
        record["t1_status"] = "unblocked"
        record["note"] = (
            f"{A0_SANITY_BAND_GATE} PASS on clean T0 A0x3 — T1 A1-A3 "
            "formal launches are unblocked by this gate (diagnostic/cost "
            "gates still apply per run)")
    else:
        record["t1_status"] = "blocked_sanity_band_breach"
        record["breach_note"] = (
            "T1 stays STRICTLY blocked: no candidate formal launch may "
            "proceed under this breach and no sanity exception may be "
            "signed.  Frozen migration-audit categories: parity "
            "certificate + process-head anchor + R3 mapping + sampler + "
            "evaluator + code + data + initialization.  If a defect is "
            "found, fix it and bump the matching identity layer "
            "(code defect => source_tree_fingerprint; data defect => "
            "data_manifest_sha256; init defect => init_mapping_sha256; "
            "anchor implicated => process_head_anchor_sha256), then "
            "RERUN the T0 A0x3 formal (old A0x3 => invalid_run, never "
            "mixed); T1 resumes only after a fresh PASS on a clean A0x3 "
            "(ADR-0033 §2.2 / config a0_baseline."
            "sanity_band_breach_procedure).")
    return record


def write_band_record(campaign_root: Any,
                      record: Mapping[str, Any]) -> Path:
    path = Path(campaign_root) / BAND_RECORD_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(record), indent=1, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")
    return path


def run_sanity_band(cfg: Mapping[str, Any], *,
                    campaign_root: Any = DEFAULT_CAMPAIGN_ROOT,
                    device: str = "cpu",
                    v541_runs_root: Any = DEFAULT_V541_RUNS_ROOT,
                    a0_endpoints: Optional[Mapping[int, Mapping[str, float]]] = None,
                    v541_reference: Optional[Mapping[int, Mapping[str, Any]]] = None,
                    a0_identities: Optional[Mapping[str, Any]] = None,
                    dry_run: bool = False) -> Dict[str, Any]:
    """The full ``V542-A0-sanity-band`` computation + record write.

    ``a0_endpoints`` / ``v541_reference`` are TEST seams (synthetic
    metrics); the production path evaluates the three T0 A0 checkpoints
    with the frozen evaluator and LOCATES the V541 E0 archive.
    """
    state = latest_ledger_state(read_ledger(campaign_root))
    missing = _t0_missing(state)

    band_cfg = dict(cfg.get("a0_baseline") or {}).get(
        "sanity_band_vs_v541_e0") or {}
    plan = {
        "schema": FORMAL_SCHEMA,
        "kind": "sanity_band",
        "dry_run": bool(dry_run),
        "gate": A0_SANITY_BAND_GATE,
        "device": str(device),
        "campaign_root": str(campaign_root),
        "a0_runs": [f"V542_A0_S{s}" for s in FROZEN_SEEDS],
        "v541_reference_root": str(v541_runs_root),
        "rule": {
            "rmse_zero_rel_max_pct": float(
                band_cfg.get("rmse_zero_rel_max_pct", 2.0)),
            "auprc_abs_max": float(band_cfg.get("auprc_abs_max", 0.01)),
            "q_corr_abs_max": float(band_cfg.get("q_corr_abs_max", 0.01)),
            "statistic": EVALUATOR_IDENTITY["statistic"],
        },
        "record_path": BAND_RECORD_REL,
        "evaluator_identity": EVALUATOR_IDENTITY,
        "training_performed": False,
        "t0_runs_not_done": missing,
    }
    if dry_run:
        return plan

    _require(not missing, "V542_SANITY_BAND_T0_INCOMPLETE",
             f"the {A0_SANITY_BAND_GATE} requires all three T0 A0 formal "
             f"runs to be done first; not done: {missing}")

    if a0_endpoints is None:
        a0_endpoints = {
            seed: evaluate_a0_band_endpoints(
                cfg, campaign_root, seed, device=device)
            for seed in FROZEN_SEEDS}
    if v541_reference is None:
        v541_reference = locate_v541_e0_reference(
            cfg, v541_runs_root=v541_runs_root)
    v541_by_seed = {seed: dict(entry["endpoints"])
                    for seed, entry in v541_reference.items()}
    band = compute_sanity_band(
        a0_endpoints, v541_by_seed,
        rmse_zero_rel_max_pct=plan["rule"]["rmse_zero_rel_max_pct"],
        auprc_abs_max=plan["rule"]["auprc_abs_max"],
        q_corr_abs_max=plan["rule"]["q_corr_abs_max"])
    record = build_band_record(
        a0_endpoints, v541_reference, band, cfg=cfg,
        a0_identities=a0_identities)
    path = write_band_record(campaign_root, record)
    plan["verdict"] = record["verdict"]
    plan["breached_endpoints"] = record["breached_endpoints"]
    plan["t1_status"] = record["t1_status"]
    plan["record_written"] = str(path)
    plan["record"] = record
    return plan


# ---------------------------------------------------------------------------
# status subcommand
# ---------------------------------------------------------------------------
def status_report(cfg: Mapping[str, Any], *,
                  campaign_root: Any = DEFAULT_CAMPAIGN_ROOT
                  ) -> Dict[str, Any]:
    """Per-run ledger state over the WHOLE frozen 26-row matrix."""
    state = latest_ledger_state(read_ledger(campaign_root))
    band = read_band_record(campaign_root)
    runs: List[Dict[str, Any]] = []
    for raw in (cfg.get("run_matrix") or {}).get("runs") or []:
        if not isinstance(raw, Mapping):
            continue
        run_id = str(raw.get("run_id"))
        latest = state.get(run_id) or {}
        runs.append({
            "run_id": run_id,
            "stage": str(raw.get("stage")),
            "backbone": str(raw.get("backbone")),
            "seed": raw.get("seed"),
            "outcome": latest.get("outcome"),
            "new_training": latest.get("new_training"),
            "identity_sha256": latest.get("identity_sha256"),
            "reason": latest.get("reason"),
            "seq": latest.get("seq"),
        })
    return {
        "schema": FORMAL_SCHEMA,
        "kind": "status",
        "ledger_path": str(ledger_path(campaign_root)),
        "ledger_rows": len(read_ledger(campaign_root)),
        "sanity_band": (None if band is None else {
            "gate": A0_SANITY_BAND_GATE,
            "verdict": band.get("verdict"),
            "t1_status": band.get("t1_status"),
            "breached_endpoints": band.get("breached_endpoints"),
            "written_at": band.get("written_at"),
        }),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _validate_cli_device(device: Any) -> str:
    from scripts.agent_ops.v542_campaign import validate_cli_device

    return validate_cli_device(device)


def _formal_refusal_from_exception(
        exc: BaseException) -> Optional[V542FormalError]:
    """Expected failure classes ⇒ typed refusal; UNEXPECTED re-raise."""
    if isinstance(exc, V542FormalError):
        return exc
    try:
        from pes2ts_core.pesn.offline.training.v542_train import (
            V542TrainerError,
        )
        if isinstance(exc, V542TrainerError):
            return V542FormalError(
                str(getattr(exc, "code", "V542_FORMAL_TRAINER_REFUSAL")),
                f"frozen trainer refused: {exc}",
                outcome=str(getattr(exc, "outcome", "invalid_run")))
    except Exception:  # noqa: BLE001 - trainer import unavailable
        pass
    try:
        from scripts.agent_ops.v542_config_schema import V542SchemaError
        if isinstance(exc, V542SchemaError):
            return V542FormalError(
                str(exc.code), f"frozen config validator refused: {exc}")
    except Exception:  # noqa: BLE001 - validator import unavailable
        pass
    if isinstance(exc, FileNotFoundError):
        return V542FormalError(
            "V542_FORMAL_DATA_MISSING",
            f"frozen data-view artifact missing: {exc}")
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v542_formal.py",
        description="V5.4.2 WP5 FORMAL runner + A0 sanity-band evaluator "
                    "(T0/T1/T2/T3/T-ATTR with frozen ordering gates; "
                    "never torch.compile; never QC/dev/confirm)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="execute (or gate-skip) the rows of one formal stage")
    p_run.add_argument("--stage", required=True,
                       choices=list(FORMAL_STAGES))
    p_run.add_argument("--only", nargs="*", default=None,
                       help="restrict to these run ids of the stage")
    p_run.add_argument("--device", default="cpu")
    p_run.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_run.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_run.add_argument("--dry-run", action="store_true",
                       help="print the stage plan WITHOUT training or "
                            "writing")

    p_band = sub.add_parser(
        "sanity-band",
        help="compute + record the V542-A0-sanity-band verdict")
    p_band.add_argument("--device", default="cpu")
    p_band.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_band.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    p_band.add_argument("--v541-runs-root", default=str(
        DEFAULT_V541_RUNS_ROOT))
    p_band.add_argument("--dry-run", action="store_true",
                        help="print the band plan WITHOUT evaluating or "
                             "writing")

    p_status = sub.add_parser(
        "status", help="print the run-ledger state of the whole matrix")
    p_status.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_status.add_argument("--campaign-root", default=str(
        DEFAULT_CAMPAIGN_ROOT))

    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            cfg = load_formal_config(args.config)
            report = status_report(cfg, campaign_root=args.campaign_root)
            for entry in report["runs"]:
                print("{:<16} {:<7} {:<5} {:>2}  outcome={:<14} "
                      "new_training={}  identity={}".format(
                          entry["run_id"], entry["stage"],
                          entry["backbone"], entry["seed"],
                          entry["outcome"] or "-",
                          entry["new_training"]
                          if entry["new_training"] is not None else "-",
                          (entry["identity_sha256"] or "-")[:12]))
            band = report["sanity_band"]
            print("sanity_band: {}".format(
                "none" if band is None
                else f"{band['verdict']} (t1_status={band['t1_status']})"))
            return 0

        if args.command == "run":
            device = _validate_cli_device(args.device)
            cfg = load_formal_config(args.config)
            formal_gate(args.config)          # formal gate FIRST
            plan = run_stage(
                args.stage, cfg=cfg, config_path=args.config,
                campaign_root=args.campaign_root, device=device,
                only=args.only, dry_run=args.dry_run,
                runplan_path=getattr(args, "runplan", None),
                budget_path=getattr(args, "budget", None),
                require_clean=bool(getattr(args, "require_clean", False)),
                formal_validated=True)
            print(json.dumps(plan, indent=2, sort_keys=True, default=str))
            return 0

        # sanity-band
        device = _validate_cli_device(args.device)
        cfg = load_formal_config(args.config)
        formal_gate(args.config)
        plan = run_sanity_band(
            cfg, campaign_root=args.campaign_root, device=device,
            v541_runs_root=args.v541_runs_root, dry_run=args.dry_run)
        record = plan.get("record") or {}
        printable = {k: v for k, v in plan.items() if k != "record"}
        print(json.dumps(printable, indent=2, sort_keys=True, default=str))
        if args.dry_run:
            return 0
        if plan.get("verdict") == "breach":
            print(f"{A0_SANITY_BAND_GATE} BREACH — T1 stays blocked; "
                  f"breached endpoints: "
                  f"{plan.get('breached_endpoints')}", file=sys.stderr)
            print(json.dumps({k: record.get(k) for k in
                              ("breached_endpoints", "breach_note",
                               "t1_status")},
                             indent=2, sort_keys=True, default=str),
                  file=sys.stderr)
            return 1
        return 0
    except V542FormalError as exc:
        print(f"V542 formal runner refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - typed conversion, re-raise
        typed = _formal_refusal_from_exception(exc)
        if typed is None:
            raise
        print(f"V542 formal runner refused: {typed}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
