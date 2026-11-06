#!/usr/bin/env python3
"""V5.4.3 WP8 FORMAL runner — runplan freeze + stage orchestration (plan §8
run matrix, §7.4 resume identity, §9 evaluation, §11.2 ``v543_formal.py`` row,
§12 WP8 completion standard; ADR-0034).

This is the FORMAL lane (unlike ``v543_campaign.py``, which is the WP7
technical-measurement harness and never runs formal work).

Subcommands
-----------
``runplan``
    Project the campaign YAML's ``run_matrix`` into the concrete 18-row matrix:
    T3 = the seven formal decoders at seed 17, T4 = the same seven at seed 29,
    T5 = the frozen finalists at seed 43 (at most 2 per track / 4 total, taken
    from the SCREEN product — a pinned finalist-list JSON; the screen
    computation belongs to the statistics lane and this runner only CONSUMES
    the pinned list).  ``--dry-run``/``runplan`` print the document; nothing is
    written.

``freeze-runplan``
    Write ``<campaign_root>/plan/runplan.json`` and pin it in
    ``<campaign_root>/plan/runplan.pin.json``.  The pin carries the frozen
    record fields (``campaign_id``, ``config_path``, ``schema_id``,
    ``seeds``, ``runplan_sha256``, ``frozen_on``, ``frozen_by``) with
    ``runplan_sha256 = sha256(canonical file bytes)`` — the
    ``governance.records.runplan_pin_record`` sha rule.  Any later edit of
    ``runplan.json`` is the typed ``V543_FORMAL_RUNPLAN_TAMPERED`` refusal.

``run-stage T0|T3|T4|T5 [--only RUN_ID ...]``
    The formal gate runs FIRST and cannot be bypassed: the frozen validator is
    invoked in ``--formal`` mode, so while ``budget.per_candidate_core_time``
    is null the runner refuses with the validator's own code
    ``V543_BUDGET_NULL`` (threshold/record problems surface as
    ``V543_THRESHOLD`` / ``V543_RECORD_MISSING`` / ``V543_RECORD_MISMATCH``).
    Only then are the pinned rows executed, each at the frozen 37,900-update
    dose (plan §7.2 :659 "37,900 flat"; the dose is never reduced and
    ``stop_after`` is NOT exposed by this CLI):

    * train through the frozen trainer, resuming from ``checkpoint_last.pt``
      only when the §7.4 identity matches (a mismatch is the trainer's own
      ``invalid_run`` refusal → ``V543_FORMAL_RESUME_IDENTITY``; the runner
      records ``new_training=0`` and never writes over the checkpoint);
    * evaluate the trained decoder with the FROZEN V543 evaluator
      (``pes2ts_core.pesn.offline.evaluation.v543_eval.evaluate_v543``) on the
      SAME inner-val frame set the M0/P0 baseline used
      (``scripts.agent_ops.v543_eval.inner_val_positions``), and write the
      candidate bundle ``pes2ts_v543_candidate_eval_bundle_v1`` whose
      ``eval_report`` is a ``pesn_v543_eval_report_v1`` document — the exact
      per-seed report schema the frozen baseline carries, so the statistics
      lane can consume candidate and baseline with one reader;
    * archive everything under ``runs/<RUN_ID>/``: ``checkpoint_best.pt`` /
      ``checkpoint_last.pt`` / ``train_log.jsonl`` / ``run_records.jsonl`` /
      ``metrics.json`` (trainer), ``identity.json`` (the §7.4 block + every
      SHA), ``eval_report.json`` (the candidate bundle) and
      ``run_manifest.json`` (the run record).

``status`` / ``list``
    Print the ledger state / the pinned matrix.

Governance and discipline
-------------------------
* the seven formal decoders are built through a FRESH registry
  (``register_m1/m2/m3`` + ``register_process_candidates``); M0/P0 are the
  S0-A parity baselines and never train here (plan §8.1 T0 = 0 trainings);
* single lane only: the same ``v543_campaign.LaneLock`` serialises every
  formal run, so two runs can never contend for the machine;
* the 18-run ceiling is enforced structurally (a 19th row is
  ``V543_FORMAL_RUN_MATRIX``), and every ledger row carries
  ``new_training`` ∈ {0,1} so ``N_core`` is counted, never summed by hand;
* no ``torch.compile``, no QC/dev/confirm access, no dose/threshold knobs.

CLI::

    python scripts/agent_ops/v543_formal.py runplan [--t5-finalists FILE]
    python scripts/agent_ops/v543_formal.py freeze-runplan [--t5-finalists FILE]
    python scripts/agent_ops/v543_formal.py run-stage T3 [--only V543_M1_S17]
    python scripts/agent_ops/v543_formal.py status
    python scripts/agent_ops/v543_formal.py list
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops import v543_campaign as campaign  # noqa: E402

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------
#: Schema ids of the formal artifacts.
FORMAL_SCHEMA: Final[str] = "pes2ts_v543_formal_runner_v1"
RUNPLAN_SCHEMA: Final[str] = "pes2ts_v543_runplan_v1"
PIN_SCHEMA: Final[str] = "pes2ts_v543_runplan_pin_v1"
LEDGER_SCHEMA: Final[str] = "pes2ts_v543_run_ledger_v1"
MANIFEST_SCHEMA: Final[str] = "pes2ts_v543_run_manifest_v1"
IDENTITY_SCHEMA: Final[str] = "pes2ts_v543_run_identity_v1"
#: The statistics lane's candidate-bundle contract (``v543_stats.py``).
CANDIDATE_BUNDLE_SCHEMA: Final[str] = "pes2ts_v543_candidate_eval_bundle_v1"
#: The frozen evaluator report schema (what ``evaluate_v543`` returns).
EVAL_REPORT_SCHEMA: Final[str] = "pesn_v543_eval_report_v1"
#: The campaign config schema id recorded on the pin.
CONFIG_SCHEMA_ID: Final[str] = "pes2ts_v543_dual_decoder_campaign_v1"

DEFAULT_CONFIG: Final[Path] = campaign.DEFAULT_CONFIG
DEFAULT_CAMPAIGN_ROOT: Final[Path] = campaign.DEFAULT_CAMPAIGN_ROOT
DEFAULT_DATA_CONFIG: Final[Path] = campaign.DEFAULT_DATA_CONFIG
#: Frozen runplan document + pin (``governance.records`` instances).
RUNPLAN_REL: Final[str] = "plan/runplan.json"
RUNPLAN_PIN_REL: Final[str] = "plan/runplan.pin.json"
#: Append-only run ledger (one row per launch attempt).
LEDGER_FILENAME: Final[str] = "run_ledger.jsonl"
#: The M0/P0 baseline document the candidate evaluation must stay comparable to.
BASELINE_REL: Final[str] = "audit/m0_p0_baseline_v543_eval.json"
#: The frozen evaluator module (its file SHA is the instrument identity).
EVALUATOR_MODULE_REL: Final[str] = (
    "pes2ts_core/pesn/offline/evaluation/v543_eval.py")

#: Formal stages §8.1 accepts.  T0 has ZERO trainings (M0/P0 parity only).
FORMAL_STAGES: Final[tuple[str, ...]] = ("T0", "T3", "T4", "T5")
#: The stages that train (T0 is a zero-training verification stage).
LAUNCHABLE_STAGES: Final[tuple[str, ...]] = ("T3", "T4", "T5")
STAGE_SEED: Final[Mapping[str, int]] = {"T3": 17, "T4": 29, "T5": 43}
FORMAL_RUNS_MAX: Final[int] = 18
T5_MAX_RUNS: Final[int] = 4
T5_PER_TRACK_MAX: Final[int] = 2
DOSE: Final[int] = campaign.FORMAL_DOSE
WARMUP: Final[int] = 400
#: The 18-run matrix total (18 × 37,900 = 682,200 updates; plan §8.1).
TOTAL_FORMAL_UPDATES: Final[int] = FORMAL_RUNS_MAX * DOSE

#: Frozen reason codes.
REASON_FORMAL_DONE: Final[str] = "formal_run_complete"
REASON_T0_VERIFIED: Final[str] = "t0_m0_p0_parity_verified_no_training"
REASON_T5_NOT_PINNED: Final[str] = "t5_finalists_not_pinned"
REASON_ALREADY_DONE: Final[str] = "already_done_in_ledger"

# -- typed refusal codes (explicit, for the report/ledger) ------------------
CODE_RUNPLAN_MISSING: Final[str] = "V543_FORMAL_RUNPLAN_MISSING"
CODE_RUNPLAN_TAMPERED: Final[str] = "V543_FORMAL_RUNPLAN_TAMPERED"
CODE_RUNPLAN_DRIFT: Final[str] = "V543_FORMAL_RUNPLAN_DRIFT"
CODE_RUN_MATRIX: Final[str] = "V543_FORMAL_RUN_MATRIX"
CODE_T5_FINALISTS: Final[str] = "V543_FORMAL_T5_FINALISTS"
CODE_RUN_UNKNOWN: Final[str] = "V543_FORMAL_RUN_UNKNOWN"
CODE_LAUNCH_FORBIDDEN: Final[str] = "V543_FORMAL_LAUNCH_FORBIDDEN"
CODE_IDENTITY_MISSING: Final[str] = "V543_FORMAL_IDENTITY_MISSING"
CODE_RESUME_IDENTITY: Final[str] = "V543_FORMAL_RESUME_IDENTITY"
CODE_RUN_FAILED: Final[str] = "V543_FORMAL_RUN_FAILED"
CODE_EVAL_FAILED: Final[str] = "V543_FORMAL_EVAL_FAILED"
CODE_STAGE_UNKNOWN: Final[str] = "V543_FORMAL_STAGE_UNKNOWN"
CODE_BASELINE_MISSING: Final[str] = "V543_FORMAL_BASELINE_MISSING"


class V543FormalError(RuntimeError):
    """Typed refusal of the formal runner (``code`` + ``detail``).

    ``outcome`` is the frozen six-word run outcome the refusal maps to
    (plan §7.4), so an operational failure can never be mistaken for a
    scientific result.
    """

    def __init__(self, code: str, detail: str = "",
                 outcome: str | None = None) -> None:
        super().__init__(f"{code}: {detail}" if detail else str(code))
        self.code = str(code)
        self.detail = str(detail)
        self.outcome = outcome


def _require(cond: bool, code: str, detail: str,
             outcome: str | None = None) -> None:
    if not cond:
        raise V543FormalError(code, detail, outcome=outcome)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# config + the formal gate (never bypassed)
# ---------------------------------------------------------------------------
def load_formal_config(path: Any | None = None) -> dict[str, Any]:
    """Load the frozen campaign config and assert the operational guards."""
    cfg = campaign.load_campaign_config(path)
    campaign.assert_operational_guards(cfg)   # single lane / no swap / QC
    return cfg


def formal_gate(config_path: Any = DEFAULT_CONFIG, *,
                record_root: Any = None) -> dict[str, Any]:
    """Run the FROZEN validator in ``--formal`` mode.

    Every validator rejection is surfaced VERBATIM (same code, same detail
    prefix): a null budget is ``V543_BUDGET_NULL``, an unsigned threshold is
    ``V543_THRESHOLD``, a missing/drifted signoff record is
    ``V543_RECORD_MISSING`` / ``V543_RECORD_MISMATCH``.  The runner can never
    launch on a config the frozen validator rejects.
    """
    from scripts.agent_ops.v543_config_schema import (
        V543SchemaError,
        load_config,
        validate_config,
    )

    try:
        return validate_config(load_config(config_path), formal=True,
                               record_root=record_root)
    except V543SchemaError as exc:
        raise V543FormalError(
            exc.code,
            "formal launch is typed-blocked by the frozen config validator: "
            + exc.detail) from exc


# ---------------------------------------------------------------------------
# runplan: project run_matrix (+ the pinned T5 finalist list) into rows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunRow:
    """One pinned formal run of the 18-row matrix."""

    run_id: str
    stage: str
    decoder_id: str
    track: str
    seed: int
    dose: int
    ordinal: int

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "decoder_id": self.decoder_id,
            "track": self.track,
            "seed": int(self.seed),
            "dose": int(self.dose),
            "ordinal": int(self.ordinal),
            "is_formal_schedule": True,
            "counts_toward_n_core": True,
        }


def load_t5_finalists(path: Any) -> dict[str, list[str]]:
    """Load the pinned T5 finalist list (the SCREEN product, read-only).

    Accepted shapes::

        {"campaign_id": "...", "move": ["M2", "M1"], "process": ["P4", "P1"]}
        {"campaign_id": "...", "finalists": {"move": [...], "process": [...]}}
        {"move": [...], "process": [...]}

    The screen COMPUTATION belongs to the statistics lane; this loader only
    validates what was pinned (≤2 per track, ≤4 total, formal decoders only,
    track-consistent ids).
    """
    p = Path(path)
    _require(p.is_file(), CODE_T5_FINALISTS,
             f"T5 finalist list not found: {p} (produce it with the "
             "statistics lane's screen, then pass --t5-finalists)")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - typed conversion
        raise V543FormalError(CODE_T5_FINALISTS,
                              f"T5 finalist list unreadable: {p}: {exc}") from exc
    _require(isinstance(doc, Mapping), CODE_T5_FINALISTS,
             f"T5 finalist list is not a JSON object: {p}")
    block = doc.get("finalists") if isinstance(doc.get("finalists"), Mapping) \
        else doc
    out: dict[str, list[str]] = {}
    for track in ("move", "process"):
        raw = block.get(track) or []
        _require(isinstance(raw, (list, tuple)), CODE_T5_FINALISTS,
                 f"T5 finalists.{track} must be a list, got {type(raw).__name__}")
        picks = [str(item) for item in raw]
        _require(len(picks) <= T5_PER_TRACK_MAX, CODE_T5_FINALISTS,
                 f"T5 takes at most {T5_PER_TRACK_MAX} finalists per track "
                 f"(plan §8.2); {track} carries {len(picks)}: {picks}")
        for decoder_id in picks:
            _require(decoder_id in campaign.FORMAL_DECODERS, CODE_T5_FINALISTS,
                     f"T5 finalist {decoder_id!r} is not one of the seven "
                     f"formal decoders {list(campaign.FORMAL_DECODERS)}")
            _require(campaign.track_of(decoder_id) == track, CODE_T5_FINALISTS,
                     f"T5 finalist {decoder_id!r} belongs to the "
                     f"{campaign.track_of(decoder_id)!r} track, not {track!r}")
        seen: dict[str, None] = {}
        for decoder_id in picks:
            _require(decoder_id not in seen, CODE_T5_FINALISTS,
                     f"T5 finalist {decoder_id!r} is listed twice in {track}")
            seen[decoder_id] = None
        out[track] = picks
    _require(sum(len(v) for v in out.values()) <= T5_MAX_RUNS, CODE_T5_FINALISTS,
             f"T5 takes at most {T5_MAX_RUNS} runs in total (plan §8.1/§8.2); "
             f"the pinned list carries {sum(len(v) for v in out.values())}")
    return out


def build_runplan(cfg: Mapping[str, Any], *,
                  t5_finalists: Mapping[str, Sequence[str]] | None = None,
                  t5_source: Mapping[str, Any] | None = None,
                  config_path: Any = DEFAULT_CONFIG) -> dict[str, Any]:
    """Project the frozen ``run_matrix`` into the concrete pinned rows.

    T3/T4 come straight from ``run_matrix.t3_seed17`` / ``t4_seed29`` (order
    is part of the freeze); T5 comes from the pinned finalist list.  Without a
    finalist list the plan still describes the nominal 14 rows and records
    ``t5_finalists_pinned=false`` — a T5 launch is then impossible (typed
    ``V543_FORMAL_T5_FINALISTS``), never silently re-derived from seed-17
    results (plan §8.2 rule 5).
    """
    matrix = dict(cfg.get("run_matrix") or {})
    rows: list[RunRow] = []
    ordinal = 0

    def _add(stage: str, decoder_id: str) -> None:
        nonlocal ordinal
        decoder_id = str(decoder_id)
        _require(decoder_id in campaign.FORMAL_DECODERS, CODE_RUN_MATRIX,
                 f"{stage}: {decoder_id!r} is not a formal decoder "
                 f"({list(campaign.FORMAL_DECODERS)}); M0/P0 never train and "
                 "M4/M5/P4-C/P5 are outside the 18 rows (plan §8.1)")
        seed = int(STAGE_SEED[stage])
        ordinal += 1
        rows.append(RunRow(
            run_id=f"V543_{decoder_id}_S{seed}", stage=stage,
            decoder_id=decoder_id, track=campaign.track_of(decoder_id),
            seed=seed, dose=DOSE, ordinal=ordinal))

    for stage, key in (("T3", "t3_seed17"), ("T4", "t4_seed29")):
        declared = matrix.get(key)
        _require(isinstance(declared, (list, tuple)) and bool(declared),
                 CODE_RUN_MATRIX,
                 f"run_matrix.{key} is missing/empty in the frozen config")
        expected = list(campaign.FORMAL_DECODERS)
        _require([str(x) for x in declared] == expected, CODE_RUN_MATRIX,
                 f"run_matrix.{key} must be exactly {expected} at seed "
                 f"{STAGE_SEED[stage]} (plan §8.1/附录 C), got "
                 f"{list(declared)}")
        for decoder_id in declared:
            _add(stage, decoder_id)

    t5_pinned = bool(t5_finalists)
    if t5_pinned:
        for track in ("move", "process"):
            for decoder_id in list(t5_finalists.get(track) or []):
                _add("T5", decoder_id)

    _require(len(rows) <= FORMAL_RUNS_MAX, CODE_RUN_MATRIX,
             f"the matrix holds {len(rows)} rows but the first-round ceiling is "
             f"{FORMAL_RUNS_MAX} (plan §8.1: 7 + 7 + <=4)")
    seen: dict[str, None] = {}
    for row in rows:
        _require(row.run_id not in seen, CODE_RUN_MATRIX,
                 f"duplicate run id {row.run_id!r} in the matrix")
        seen[row.run_id] = None

    return {
        "schema": RUNPLAN_SCHEMA,
        "campaign_id": str(cfg.get("campaign_id", "")),
        "schema_id": CONFIG_SCHEMA_ID,
        "config_path": str(config_path),
        "config_sha256": campaign.sha256_file(config_path),
        "seeds": [int(s) for s in STAGE_SEED.values()],
        "dose": {"updates": DOSE, "warmup_updates": WARMUP,
                 "flat_note": ("37,900 flat + 400 warmup (plan §7.2 :659); the "
                               "V542 400+37,500 two-phase split is retired")},
        "run_matrix": {
            "formal_runs_max": FORMAL_RUNS_MAX,
            "nominal_rows": 14,
            "rows_total": len(rows),
            "t5_rows": sum(1 for row in rows if row.stage == "T5"),
            "t5_finalists_pinned": t5_pinned,
            "total_formal_updates_ceiling": int(TOTAL_FORMAL_UPDATES),
        },
        "t5_source": (None if not t5_source else dict(t5_source)),
        "identity": {
            "sampler_spec_sha256": str(
                (cfg.get("training") or {}).get("sampler_spec_sha256", "")),
            "bridge_spec_sha256": _bridge_spec_sha256(),
            "evaluator": _evaluator_identity(),
            "plan_section": "plan §8.1 / 附录 C; §7.4 identity",
        },
        "rows": [row.describe() for row in rows],
        "created_utc": _now(),
        "runner_schema": FORMAL_SCHEMA,
    }


def _bridge_spec_sha256() -> str:
    from pes2ts_core.pesn.models.v543.latent_bridge import bridge_spec_sha256

    return str(bridge_spec_sha256())


def _evaluator_identity() -> dict[str, Any]:
    """The frozen evaluator identity (code SHA + spec SHA + schema)."""
    module = PROJECT_ROOT / EVALUATOR_MODULE_REL
    spec_sha: str | None = None
    try:
        from pes2ts_core.pesn.offline.evaluation.v543_eval import (
            evaluator_spec_sha256,
        )

        spec_sha = str(evaluator_spec_sha256())
    except Exception:  # noqa: BLE001 - recorded, never fabricated
        spec_sha = None
    return {
        "module": EVALUATOR_MODULE_REL,
        "code_sha256": (campaign.sha256_file(module)
                        if module.is_file() else None),
        "spec_sha256": spec_sha,
        "report_schema": EVAL_REPORT_SCHEMA,
    }


def runplan_rows(plan: Mapping[str, Any]) -> list[RunRow]:
    """The typed rows of a runplan document."""
    raw_rows = plan.get("rows")
    _require(isinstance(raw_rows, (list, tuple)), CODE_RUNPLAN_DRIFT,
             "the runplan document carries no 'rows' list")
    out: list[RunRow] = []
    for index, raw in enumerate(raw_rows, start=1):
        _require(isinstance(raw, Mapping), CODE_RUNPLAN_DRIFT,
                 f"runplan row #{index} is not an object")
        out.append(RunRow(
            run_id=str(raw["run_id"]), stage=str(raw["stage"]),
            decoder_id=str(raw["decoder_id"]), track=str(raw["track"]),
            seed=int(raw["seed"]), dose=int(raw["dose"]),
            ordinal=int(raw.get("ordinal", index))))
    return out


def stage_rows(plan: Mapping[str, Any], stage: str) -> list[RunRow]:
    """The pinned rows of one formal stage (frozen order)."""
    _require(stage in FORMAL_STAGES, CODE_STAGE_UNKNOWN,
             f"stage must be one of {list(FORMAL_STAGES)} (T1/T2 smoke and "
             "diagnostic belong to the WP7 harness v543_campaign.py), got "
             f"{stage!r}")
    return [row for row in runplan_rows(plan) if row.stage == stage]


def filter_only(rows: Sequence[RunRow], only: Sequence[str] | None
                ) -> list[RunRow]:
    """Restrict to ``--only`` run ids (unknown ids are typed refusals)."""
    if not only:
        return list(rows)
    wanted = [str(value) for value in only]
    known = {row.run_id for row in rows}
    unknown = [value for value in wanted if value not in known]
    _require(not unknown, CODE_RUN_UNKNOWN,
             f"run id(s) {unknown} are not pinned rows of this stage "
             f"(stage rows: {sorted(known)})")
    keep = set(wanted)
    return [row for row in rows if row.run_id in keep]


# ---------------------------------------------------------------------------
# runplan freeze / pin verification
# ---------------------------------------------------------------------------
def runplan_path(campaign_root: Any) -> Path:
    return Path(campaign_root) / RUNPLAN_REL


def pin_path(campaign_root: Any) -> Path:
    return Path(campaign_root) / RUNPLAN_PIN_REL


def freeze_runplan(plan: Mapping[str, Any], *, campaign_root: Any,
                   frozen_by: str = "repository_owner_via_dsh_orchestration",
                   require_formal_gate: bool = False,
                   config_path: Any = DEFAULT_CONFIG,
                   record_root: Any = None) -> dict[str, Any]:
    """Write the runplan document and its pin (plan §12 WP5 record requirement).

    ``runplan_sha256`` is ``sha256(canonical file bytes)`` — the frozen
    ``governance.records.runplan_pin_record.sha_rule``.  A pin whose digest no
    longer matches the document is ``V543_FORMAL_RUNPLAN_TAMPERED``.
    """
    if require_formal_gate:
        formal_gate(config_path, record_root=record_root)
    root = Path(campaign_root)
    document = dict(plan)
    document["frozen_on"] = _now()
    document["frozen_by"] = str(frozen_by)
    doc_path = runplan_path(root)
    campaign.write_json(doc_path, document)
    runplan_sha = campaign.sha256_file(doc_path)
    pin = {
        "schema": PIN_SCHEMA,
        # -- the frozen records-spec required fields ------------------------
        "campaign_id": str(document.get("campaign_id", "")),
        "config_path": str(document.get("config_path", "")),
        "schema_id": str(document.get("schema_id", CONFIG_SCHEMA_ID)),
        "seeds": [int(seed) for seed in document.get("seeds") or []],
        "runplan_sha256": runplan_sha,
        "frozen_on": document["frozen_on"],
        "frozen_by": document["frozen_by"],
        # -- provenance / audit ---------------------------------------------
        "gv_campaign_id": "V5.4.3",
        "runplan_path": str(doc_path.relative_to(PROJECT_ROOT)
                            if doc_path.is_relative_to(PROJECT_ROOT)
                            else doc_path),
        "config_sha256": str(document.get("config_sha256", "")),
        "formal_runs_max": FORMAL_RUNS_MAX,
        "rows_total": len(document.get("rows") or []),
        "t5_rows": int((document.get("run_matrix") or {}).get("t5_rows", 0)),
        "t5_finalists_pinned": bool(
            (document.get("run_matrix") or {}).get("t5_finalists_pinned")),
        "t5_note": (
            "T5 rows are projected ONLY from the pinned screen finalist list "
            "(--t5-finalists); with t5_finalists_pinned=false this pin holds "
            "the 14 nominal rows and a T5 launch is impossible until the "
            "screen emits finalists and the runplan is re-frozen (plan §8.2 "
            "rule 5: the list is never re-derived from seed-17/29 results)"),
        "refreeze_rule": (
            "re-freeze AFTER the WP7 budget fill: filling "
            "budget.per_candidate_core_time changes config_sha256 and the pin "
            "must follow the config (plan §8 change rule: any field change = "
            "config_sha256 change + runplan re-freeze)"),
        "total_formal_updates_ceiling": int(TOTAL_FORMAL_UPDATES),
        "dose": dict(document.get("dose") or {}),
        "sha_rule": "sha256_of_canonical_file_bytes",
        "frozen_sequence": [
            "T0 M0/P0 parity (no training)",
            "T3 seven formal decoders @ seed 17",
            "T4 seven formal decoders @ seed 29",
            "T5 pinned finalists @ seed 43 (<=4)",
        ],
        "recompute": ("runplan_sha256 = sha256 of the bytes of "
                      f"{RUNPLAN_REL}; any edit of that file invalidates the "
                      "pin (V543_FORMAL_RUNPLAN_TAMPERED)"),
    }
    campaign.write_json(pin_path(root), pin)
    return pin


def load_runplan(campaign_root: Any) -> dict[str, Any]:
    path = runplan_path(campaign_root)
    _require(path.is_file(), CODE_RUNPLAN_MISSING,
             f"no frozen runplan at {path} — run 'freeze-runplan' first (the "
             "formal lane never invents matrix rows)")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_runplan_pin(campaign_root: Any, *,
                       config_path: Any = DEFAULT_CONFIG) -> dict[str, Any]:
    """Verify the pin against the document AND the frozen config.

    Three checks: the pin exists, its ``runplan_sha256`` matches the document's
    current bytes, and the document still matches the campaign config's
    ``run_matrix`` (a drifted config is ``V543_FORMAL_RUNPLAN_DRIFT``).
    """
    pin_file = pin_path(campaign_root)
    _require(pin_file.is_file(), CODE_RUNPLAN_MISSING,
             f"no runplan pin at {pin_file} — 'freeze-runplan' writes it")
    pin = json.loads(pin_file.read_text(encoding="utf-8"))
    document = load_runplan(campaign_root)
    actual = campaign.sha256_file(runplan_path(campaign_root))
    _require(str(pin.get("runplan_sha256")) == actual, CODE_RUNPLAN_TAMPERED,
             "the runplan document no longer matches its pin: pinned "
             f"runplan_sha256={pin.get('runplan_sha256')!r} but the file "
             f"hashes to {actual!r} — re-freeze the runplan (a silent edit of "
             "the matrix is forbidden)")
    config_sha = campaign.sha256_file(config_path)
    if pin.get("config_sha256"):
        _require(str(pin["config_sha256"]) == config_sha,
                 CODE_RUNPLAN_DRIFT,
                 "the campaign config changed after the runplan was frozen "
                 f"(pinned {pin['config_sha256']!r}, now {config_sha!r}) — "
                 "re-freeze the runplan (plan §8: any matrix/seed change needs "
                 "a new pin)")
    cfg = load_formal_config(config_path)
    expected = build_runplan(
        cfg, t5_finalists=None, config_path=config_path)
    nominal = [row for row in runplan_rows(document) if row.stage != "T5"]
    expected_rows = {(row.run_id, row.stage, row.decoder_id, row.seed)
                     for row in runplan_rows(expected)}
    for row in nominal:
        _require((row.run_id, row.stage, row.decoder_id, row.seed)
                 in expected_rows, CODE_RUNPLAN_DRIFT,
                 f"pinned row {row.run_id} is not in the config's run_matrix "
                 "(the pin and the frozen config disagree)")
    return {"pin": pin, "document": document, "runplan_sha256": actual,
            "config_sha256": config_sha, "rows": len(runplan_rows(document))}


# ---------------------------------------------------------------------------
# ledger (append-only)
# ---------------------------------------------------------------------------
def ledger_path(campaign_root: Any) -> Path:
    return Path(campaign_root) / LEDGER_FILENAME


def read_ledger(campaign_root: Any) -> list[dict[str, Any]]:
    return campaign.read_jsonl(ledger_path(campaign_root))


def latest_ledger_state(rows: Sequence[Mapping[str, Any]]
                        ) -> dict[str, dict[str, Any]]:
    """Latest row per run id (append-only ledger: last record wins)."""
    state: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        if run_id:
            state[run_id] = dict(row)
    return state


def append_ledger_row(campaign_root: Any, row: Mapping[str, Any]
                      ) -> dict[str, Any]:
    """APPEND exactly one ledger row (``seq`` = append order, monotone)."""
    existing = read_ledger(campaign_root)
    payload = dict(row)
    payload["schema"] = LEDGER_SCHEMA
    payload["seq"] = len(existing) + 1
    payload.setdefault("ts", _now())
    payload.setdefault("torch_compile", False)
    payload.setdefault("qc_calls", 0)
    payload.setdefault("confirm_read", False)
    campaign.append_jsonl(ledger_path(campaign_root), payload)
    return payload


def new_training_count(state: Mapping[str, Mapping[str, Any]]) -> int:
    """``N_core`` counted from the ledger (never summed from run rows)."""
    return sum(int((row or {}).get("new_training", 0) or 0)
               for row in state.values())


# ---------------------------------------------------------------------------
# per-run execution (frozen trainer + frozen evaluator; seams for CPU tests)
# ---------------------------------------------------------------------------
def run_dir_for(campaign_root: Any, run_id: str) -> Path:
    return Path(campaign_root) / "runs" / str(run_id)


def run_identity(*, row: RunRow, cfg: Mapping[str, Any],
                 config_path: Any, campaign_root: Any,
                 encoder_checkpoint: Mapping[str, Any],
                 pin: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The archived §7.4 identity of one formal run (all SHAs recorded)."""
    identity = {
        "schema": IDENTITY_SCHEMA,
        "run_id": row.run_id,
        "stage": row.stage,
        "decoder_id": row.decoder_id,
        "track": row.track,
        "seed": int(row.seed),
        "dose": {"updates": int(row.dose), "warmup_updates": WARMUP,
                 "is_formal_schedule": True},
        "campaign": {
            "campaign_id": str(cfg.get("campaign_id", "")),
            "config_path": str(config_path),
            "config_sha256": campaign.sha256_file(config_path),
            "schema_id": CONFIG_SCHEMA_ID,
        },
        "runplan": {
            "runplan_path": RUNPLAN_REL,
            "runplan_sha256": (None if pin is None
                               else str(pin.get("runplan_sha256"))),
            "pin_path": RUNPLAN_PIN_REL,
            "pin_sha256": (None if pin is None
                           else campaign.sha256_file(pin_path(campaign_root))
                           if pin_path(campaign_root).is_file() else None),
            "rows_total": (None if pin is None else int(pin.get("rows_total", 0))),
        },
        "encoder": {
            "path": str(encoder_checkpoint.get("path", "")),
            "sha256": str(encoder_checkpoint.get("sha256", "")),
        },
        "bridge_spec_sha256": _bridge_spec_sha256(),
        "evaluator": _evaluator_identity(),
        "sampler_spec_sha256": str(
            (cfg.get("training") or {}).get("sampler_spec_sha256", "")),
        "created_utc": _now(),
    }
    identity["identity_sha256"] = campaign.canonical_sha256(identity)
    return identity


def build_formal_context(*, cfg: Mapping[str, Any], env: Any,
                         decoder_id: str, seed: int, device: str = "cpu",
                         config_path: Any = DEFAULT_CONFIG) -> Any:
    """Build the frozen encoder + candidate model + data source for one run."""
    return campaign.build_candidate_context(
        cfg=cfg, env=env, decoder_id=decoder_id, seed=int(seed),
        device=device, campaign_config_path=config_path)


def train_formal_run(ctx: Any, *, run_dir: Any, device: str = "cpu"
                     ) -> dict[str, Any]:
    """Train ONE formal row at the FULL frozen dose (resume-aware).

    ``stop_after`` is never passed: a formal run always executes the frozen
    37,900 updates (plan §7.2 :659, §8.5 "预注册剂量用尽即停").
    """
    from pes2ts_core.pesn.offline.training.v543_train import V543TrainerError

    checkpoint_last = Path(run_dir) / "checkpoint_last.pt"
    resume = checkpoint_last.is_file()
    try:
        # 2026-11-03 fix: stop_after must be None for a formal run — the
        # frozen dose comes from updates=37,900 in the trainer config, and a
        # non-None stop_after flips the trainer's is_formal_schedule property
        # to False (v543_train.py:490-493), mislabelling the ledger.
        # identity_dict excludes stop_after, so this is identity-neutral and
        # behaviour-identical (same 37,900 updates).
        metrics = campaign.run_leg(
            ctx, out_dir=run_dir, stop_after=None, device=device,
            resume=resume)
    except campaign.V543CampaignError as exc:
        # The campaign seam maps the trainer's identity refusal; map it onto
        # the formal lane's own typed refusal (the run is an invalid_run).
        if exc.code == "V543_CAMPAIGN_RESUME_IDENTITY":
            raise V543FormalError(CODE_RESUME_IDENTITY, exc.detail,
                                  outcome="invalid_run") from exc
        if exc.code == "V543_CAMPAIGN_RESUME_MISSING":
            raise V543FormalError(CODE_RESUME_IDENTITY, exc.detail,
                                  outcome="invalid_run") from exc
        raise V543FormalError(
            str(getattr(exc, "code", CODE_RUN_FAILED)), exc.detail,
            outcome=getattr(exc, "outcome", None)) from exc
    except V543TrainerError as exc:  # pragma: no cover - seam maps it already
        raise V543FormalError(
            str(getattr(exc, "code", CODE_RUN_FAILED)), str(exc),
            outcome=getattr(exc, "outcome", "invalid_run")) from exc
    metrics = dict(metrics)
    metrics["resumed"] = bool(resume)
    return metrics


def evaluate_formal_run(ctx: Any, *, run_id: str, seed: int, stage: str,
                        decoder_id: str, track: str, env: Any,
                        cfg: Mapping[str, Any], encoder_checkpoint: Mapping[str, Any],
                        device: str = "cpu",
                        eval_curves: int | None = None,
                        frames_per_curve: int | None = None) -> dict[str, Any]:
    """Evaluate the trained decoder with the FROZEN V543 evaluator (§9).

    The frame set is the SAME frozen inner-val slice the M0/P0 baseline used
    (``scripts.agent_ops.v543_eval.inner_val_positions`` over
    ``v53mp_pool_split(seed)``); dev and confirm are never read.  The emitted
    document is the statistics lane's candidate bundle, whose ``eval_report``
    carries the ``pesn_v543_eval_report_v1`` schema the baseline per-seed
    reports carry.
    """
    import numpy as np
    import torch

    from pes2ts_core.pesn.contracts.key_atom_v3 import (
        NODE_KEY_SET_KEY,
        NODE_POS_KEY,
        forward_view_v3,
    )
    from pes2ts_core.pesn.offline.evaluation.v543_eval import (
        EvalFrame,
        evaluate_v543,
    )
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _batch_from_positions,
        _train_q_mean,
        _FeatureLRU,
        v53mp_pool_split,
    )
    from scripts.agent_ops.v543_eval import (
        DEFAULT_EVAL_CURVES,
        DEFAULT_FRAMES_PER_CURVE,
        frame_intent_class,
        inner_val_positions,
        label_status_of,
    )
    from scripts.agent_ops.v543_s0_probe import positions_meta

    eval_curves = int(DEFAULT_EVAL_CURVES if eval_curves is None
                      else eval_curves)
    frames_per_curve = int(DEFAULT_FRAMES_PER_CURVE if frames_per_curve is None
                           else frames_per_curve)
    train_pos, val_pos = v53mp_pool_split(
        env.dataset.samples, pool_curve_ids=env.pool_ids, seed=int(seed))
    positions, rule = inner_val_positions(
        val_pos, env.dataset, eval_curves=eval_curves,
        frames_per_curve=frames_per_curve)
    meta = positions_meta(env.dataset, positions)

    encoder = ctx.encoder
    bridge = ctx.model.assembly.bridge
    model = ctx.model
    checkpoint_sha = str(encoder_checkpoint.get("sha256", ""))
    lru = _FeatureLRU(env.dataset, capacity=8_192)
    frames: list[Any] = []
    status_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    for start in range(0, len(positions), 32):
        chunk = [int(p) for p in positions[start:start + 32]]
        batch, records = _batch_from_positions(env.dataset, lru, chunk, "cpu")
        view = forward_view_v3(batch)
        with torch.no_grad():
            packet = encoder(view)
            context = bridge.build_context(
                packet, view, encoder_ckpt_sha256=checkpoint_sha)
            out = model.forward_context(context, debug=True)
        key_ptr = context.key_ptr.detach().cpu().numpy().astype(np.int64)
        pair_ptr = context.pair_ptr.detach().cpu().numpy().astype(np.int64)
        intent_pair_index = context.intent_pair_index.detach().cpu().numpy(
        ).astype(np.int64)
        permutation = np.argsort(
            context.key_graph.detach().cpu().numpy().astype(np.int64),
            kind="stable")
        key_index_raw = batch[NODE_KEY_SET_KEY].detach().cpu().numpy().astype(
            np.int64)
        canonical_key_rows = key_index_raw[permutation]
        u_rows = out["u"].detach().cpu().numpy().astype(np.float64)
        q_rows = out["q"].detach().cpu().numpy().astype(np.float64)
        e_rows = out["endpoint_logit"].detach().cpu().numpy().astype(
            np.float64)
        u0_canonical = context.u0.detach().cpu().numpy()
        u0_rows = np.empty_like(u0_canonical)
        u0_rows[permutation] = u0_canonical
        key_v = context.key_v.detach().cpu().numpy()
        y_teacher = batch["y_teacher"].detach().cpu().numpy()
        pos = batch[NODE_POS_KEY].detach().cpu().numpy().astype(np.float64)
        pair_index = batch["pair_index"].detach().cpu().numpy().astype(np.int64)
        pair_edit = batch["pair_edit"].detach().cpu().numpy().astype(
            np.float64)
        ptr = batch["ptr"].detach().cpu().numpy().astype(np.int64)
        sample_ids = [str(s) for s in batch["sample_ids"]]
        statuses = label_status_of(records)
        n_graphs = int(key_ptr.shape[0]) - 1
        _require(n_graphs == len(chunk), CODE_EVAL_FAILED,
                 f"{run_id}: {n_graphs} graphs in the context for "
                 f"{len(chunk)} positions")
        for g in range(n_graphs):
            k_lo, k_hi = int(key_ptr[g]), int(key_ptr[g + 1])
            p_lo, p_hi = int(pair_ptr[g]), int(pair_ptr[g + 1])
            sample_id = sample_ids[g]
            label_row = env.labels.by_sample[sample_id]
            endpoint_row = env.endpoint.by_sample[sample_id]
            slots_global = intent_pair_index[:, p_lo:p_hi]
            if int(slots_global.size):
                pair_offsets = (
                    pos[canonical_key_rows[slots_global[1]]]
                    - pos[canonical_key_rows[slots_global[0]]])
                slots_local = (slots_global - k_lo).T
            else:
                pair_offsets = np.zeros((0, 3), dtype=np.float64)
                slots_local = np.zeros((0, 2), dtype=np.int64)
            status = statuses[g]
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            intent = frame_intent_class(int(ptr[g]), int(ptr[g + 1]),
                                        pair_index, pair_edit)
            intent_counts[intent] = int(intent_counts.get(intent, 0)) + 1
            endpoint_label = int(endpoint_row.endpoint_label)
            q_value = float(q_rows[g])
            frames.append(EvalFrame(
                curve_id=str(label_row.curve_id),
                u=np.asarray(u_rows[k_lo:k_hi], dtype=np.float64),
                y=np.asarray(y_teacher[k_lo:k_hi], dtype=np.float64),
                pair_slots=np.asarray(slots_local, dtype=np.int64),
                pair_offsets=np.asarray(pair_offsets, dtype=np.float64),
                direction=str(label_row.direction),
                stratum=str(meta[start + g]["stratum"]),
                intent_class=intent,
                q_pred=q_value,
                q_true=(0.0 if label_row.q is None else float(label_row.q)),
                q_mask=bool(label_row.q_mask),
                endpoint_logit=float(e_rows[g]),
                endpoint_label=endpoint_label,
                endpoint_mask=endpoint_label in (0, 1),
                is_start_row=bool(endpoint_row.start_row),
                is_ts_row=bool(endpoint_row.ts_row),
                move_eligible=(status == "ok"),
                u0=np.asarray(u0_rows[k_lo:k_hi], dtype=np.float64),
                q0_raw=float(context.q0_raw.detach().cpu().numpy()[g]),
                endpoint0_logit=float(
                    context.endpoint0_logit.detach().cpu().numpy()[g]),
                key_v=(np.asarray(key_v[k_lo:k_hi], dtype=np.float64)),
            ))

    train_q_mean = _train_q_mean(env.dataset.samples, env.labels,
                                 [int(p) for p in train_pos])
    report = evaluate_v543(frames, train_q_mean=train_q_mean, tau_end=None)
    _require(str(report.get("schema")) == EVAL_REPORT_SCHEMA, CODE_EVAL_FAILED,
             f"{run_id}: the evaluator returned schema "
             f"{report.get('schema')!r}, expected {EVAL_REPORT_SCHEMA!r}")
    evaluator = _evaluator_identity()
    return {
        "schema": CANDIDATE_BUNDLE_SCHEMA,
        "run_id": str(run_id),
        "candidate_id": str(decoder_id),
        "track": str(track),
        "seed": int(seed),
        "stage": str(stage),
        "role": "formal_candidate",
        "evaluator": {
            "module": evaluator["module"],
            "code_sha256": evaluator["code_sha256"],
            "spec_sha256": evaluator["spec_sha256"],
            "report_schema": EVAL_REPORT_SCHEMA,
        },
        "bridge_spec_sha256": _bridge_spec_sha256(),
        # §9.5 :1037 identity chain: the checkpoint slot is the seed's A0
        # encoder checkpoint (the same value the baseline document carries), so
        # candidate and baseline stay comparable under one reader.
        "checkpoint": {
            "path": str(encoder_checkpoint.get("path", "")),
            "sha256": str(encoder_checkpoint.get("sha256", "")),
        },
        "eval_report": report,
        "bootstrap_inputs": None,
        "bootstrap_note": (
            "single-seed candidate bundle: the §9.4 curve values are the "
            "evaluator's own per_curve rows (the statistics lane derives them); "
            "no duplicate curve table is declared, so the two can never "
            "disagree"),
        "positions": {
            "rule": rule,
            "n_positions": len(positions),
            "positions_sha256": campaign.canonical_sha256(positions),
            "train_q_mean": train_q_mean,
            "eval_curves": eval_curves,
            "frames_per_curve": frames_per_curve,
        },
        "label_status_evidence": {
            "n_frames": len(frames),
            "by_label_status": dict(sorted(status_counts.items())),
            "by_intent_class": dict(sorted(intent_counts.items())),
            "move_eligibility_rule": (
                "only label_status == 'ok' frames enter a MOVE denominator"),
        },
        "data": {
            "split_manifest": str(env.dataset.tables_dir),
            "pool_tier": "50k",
            "dev_accessed": False,
            "confirm_accessed": False,
        },
        "created_utc": _now(),
    }


def _write_run_artifacts(*, run_dir: Path, record: Mapping[str, Any],
                         identity: Mapping[str, Any],
                         eval_bundle: Mapping[str, Any] | None) -> None:
    """Archive the run record + identity (+ eval bundle) under the run dir."""
    campaign.write_json(Path(run_dir) / "identity.json", identity)
    campaign.write_json(Path(run_dir) / "run_manifest.json", record)
    if eval_bundle is not None:
        campaign.write_json(Path(run_dir) / "eval_report.json", eval_bundle)
    campaign.write_json(Path(run_dir) / "run_record.json", record)


def execute_run(env: Any, row: RunRow, *, cfg: Mapping[str, Any],
                config_path: Any, campaign_root: Any, device: str = "cpu",
                pin: Mapping[str, Any] | None = None,
                context_builder: Any = None,
                trainer: Any = None,
                evaluator: Any = None,
                evaluate: bool = True) -> dict[str, Any]:
    """Train (+ evaluate) ONE pinned formal row and archive everything.

    A resume-identity mismatch is the trainer's own ``invalid_run`` refusal:
    the ledger records ``new_training=0`` and the checkpoint is untouched
    (the trainer refuses BEFORE writing, plan §7.4 :771).
    """
    _require(row.stage in LAUNCHABLE_STAGES, CODE_LAUNCH_FORBIDDEN,
             f"{row.run_id}: stage {row.stage!r} never trains (T0 is the "
             "zero-training M0/P0 parity stage)")
    _require(row.decoder_id in campaign.FORMAL_DECODERS, CODE_LAUNCH_FORBIDDEN,
             f"{row.run_id}: {row.decoder_id!r} is not a formal decoder")
    _require(int(row.dose) == DOSE, CODE_RUN_MATRIX,
             f"{row.run_id}: dose {row.dose} != the frozen {DOSE} (plan §7.2 "
             ":659; the dose is never reduced)")
    builder = context_builder or build_formal_context
    trainer_fn = trainer or train_formal_run
    evaluator_fn = evaluator or evaluate_formal_run

    run_dir = run_dir_for(campaign_root, row.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = builder(cfg=cfg, env=env, decoder_id=row.decoder_id,
                  seed=int(row.seed), device=device, config_path=config_path)
    encoder_checkpoint = dict(ctx.checkpoint)
    identity = run_identity(
        row=row, cfg=cfg, config_path=config_path,
        campaign_root=campaign_root, encoder_checkpoint=encoder_checkpoint,
        pin=pin)

    try:
        metrics = trainer_fn(ctx, run_dir=run_dir, device=device)
    except (V543FormalError, campaign.V543CampaignError) as exc:
        code = str(getattr(exc, "code", CODE_RUN_FAILED))
        outcome = (getattr(exc, "outcome", None) or "invalid_run")
        if isinstance(exc, campaign.V543CampaignError) and \
                code == "V543_CAMPAIGN_RESUME_IDENTITY":  # pragma: no cover
            code = CODE_RESUME_IDENTITY
        detail = str(getattr(exc, "detail", "") or exc)  # pragma: no cover
        record = _record(row=row, cfg=cfg, identity=identity,
                         outcome=outcome,
                         reason=f"{code}: {detail}",
                         metrics=None, eval_bundle=None, run_dir=run_dir,
                         config_path=config_path)
        _write_run_artifacts(run_dir=run_dir, record=record, identity=identity,
                             eval_bundle=None)
        append_ledger_row(campaign_root, _ledger_row(record, new_training=0))
        raise V543FormalError(code, detail, outcome=outcome) from exc

    run_metrics = dict(metrics.get("metrics") or {})
    trainer_identity = dict(metrics.get("identity") or {})
    trainer_identity_sha = metrics.get("identity_sha256")
    eval_bundle = None
    eval_error: str | None = None
    if evaluate:
        try:
            eval_bundle = evaluator_fn(
                ctx, run_id=row.run_id, seed=int(row.seed), stage=row.stage,
                decoder_id=row.decoder_id, track=row.track, env=env,
                cfg=cfg, encoder_checkpoint=encoder_checkpoint, device=device)
        except Exception as exc:  # noqa: BLE001 - recorded, then typed
            eval_error = f"{type(exc).__name__}: {exc}"
            eval_bundle = None

    outcome = str(run_metrics.get("outcome", "done"))
    reason = REASON_FORMAL_DONE if eval_error is None else (
        f"{CODE_EVAL_FAILED}: {eval_error}")
    record = _record(row=row, cfg=cfg, identity=identity, outcome=outcome,
                     reason=reason, metrics=run_metrics, eval_bundle=eval_bundle,
                     run_dir=run_dir, config_path=config_path,
                     trainer_identity=trainer_identity,
                     trainer_identity_sha256=trainer_identity_sha,
                     eval_error=eval_error)
    _write_run_artifacts(run_dir=run_dir, record=record, identity=identity,
                         eval_bundle=eval_bundle)
    append_ledger_row(campaign_root, _ledger_row(
        record,
        new_training=1 if outcome in ("done", "scientific_failure") else 0))
    if eval_error is not None:
        raise V543FormalError(
            CODE_EVAL_FAILED,
            f"{row.run_id}: training finished but the frozen evaluator failed "
            f"({eval_error}); the run artifacts and ledger row are archived",
            outcome="invalid_run")
    return record


def _record(*, row: RunRow, cfg: Mapping[str, Any],
            identity: Mapping[str, Any], outcome: str, reason: str,
            metrics: Mapping[str, Any] | None,
            eval_bundle: Mapping[str, Any] | None, run_dir: Path,
            config_path: Any,
            trainer_identity: Mapping[str, Any] | None = None,
            trainer_identity_sha256: str | None = None,
            eval_error: str | None = None) -> dict[str, Any]:
    """The archived per-run record (``run_manifest.json`` / ``run_record.json``)."""
    metrics = dict(metrics or {})
    identity_file = Path(run_dir) / "identity.json"
    eval_report = Path(run_dir) / "eval_report.json"
    return {
        "schema": MANIFEST_SCHEMA,
        "runner_schema": FORMAL_SCHEMA,
        "kind": "formal_run",
        "run_id": row.run_id,
        "stage": row.stage,
        "decoder_id": row.decoder_id,
        "track": row.track,
        "seed": int(row.seed),
        "dose": {"updates": int(row.dose), "warmup_updates": WARMUP,
                 "is_formal_schedule": bool(
                     metrics.get("is_formal_schedule", True))},
        "outcome": str(outcome),
        "reason": str(reason),
        "new_training": 1 if outcome in ("done", "scientific_failure") else 0,
        "winner_eligible": True,
        "h3_metric_selection_forbidden": True,
        "identity": dict(identity),
        "identity_sha256": identity.get("identity_sha256"),
        "trainer_identity": dict(trainer_identity or {}),
        "trainer_identity_sha256": trainer_identity_sha256,
        "config_path": str(config_path),
        "config_sha256": campaign.sha256_file(config_path),
        "best_inner_val_loss": metrics.get("best_val_loss"),
        "best_update": metrics.get("best_update"),
        "updates_run": metrics.get("updates"),
        "wall_seconds": metrics.get("wall_seconds"),
        "seconds_per_update": metrics.get("seconds_per_update"),
        "encoder_forward_count": metrics.get("encoder_forward_count"),
        "cache": metrics.get("cache"),
        "params": metrics.get("params"),
        "loss_coefficients": metrics.get("loss_coefficients"),
        "resumed": bool(metrics.get("resumed", False)),
        "eval": {
            "evaluated": eval_bundle is not None,
            "bundle_schema": (None if eval_bundle is None
                              else eval_bundle.get("schema")),
            "eval_report_path": (str(eval_report)
                                 if eval_bundle is not None else None),
            "evaluator": (None if eval_bundle is None
                          else dict(eval_bundle.get("evaluator") or {})),
            "report_schema": (None if eval_bundle is None
                              else (eval_bundle.get("eval_report") or {})
                              .get("schema")),
            "error": eval_error,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "checkpoint_best": (str(Path(run_dir) / "checkpoint_best.pt")
                                if (Path(run_dir) / "checkpoint_best.pt")
                                .is_file() else None),
            "checkpoint_last": (str(Path(run_dir) / "checkpoint_last.pt")
                                if (Path(run_dir) / "checkpoint_last.pt")
                                .is_file() else None),
            "train_log": (str(Path(run_dir) / "train_log.jsonl")
                          if (Path(run_dir) / "train_log.jsonl").is_file()
                          else None),
            "run_records": (str(Path(run_dir) / "run_records.jsonl")
                            if (Path(run_dir) / "run_records.jsonl").is_file()
                            else None),
            "metrics": (str(Path(run_dir) / "metrics.json")
                        if (Path(run_dir) / "metrics.json").is_file() else None),
            "identity": str(identity_file),
            "eval_report": (str(eval_report) if eval_bundle is not None
                            else None),
        },
        "torch_compile": False,
        "qc_calls": 0,
        "confirm_read": False,
        "dev_read": False,
        "created_utc": _now(),
    }


def _ledger_row(record: Mapping[str, Any], *, new_training: int
                ) -> dict[str, Any]:
    """The append-only ledger projection of one run record."""
    return {
        "run_id": str(record.get("run_id")),
        "stage": str(record.get("stage")),
        "decoder_id": str(record.get("decoder_id")),
        "track": str(record.get("track")),
        "seed": int(record.get("seed", -1)),
        "outcome": str(record.get("outcome")),
        "new_training": int(new_training),
        "reason": str(record.get("reason")),
        "identity_sha256": record.get("identity_sha256"),
        "trainer_identity_sha256": record.get("trainer_identity_sha256"),
        "best_inner_val_loss": record.get("best_inner_val_loss"),
        "wall_seconds": record.get("wall_seconds"),
        "is_formal_schedule": bool(
            (record.get("dose") or {}).get("is_formal_schedule", True)),
        "eval_report": (record.get("eval") or {}).get("eval_report_path"),
        "run_dir": (record.get("artifacts") or {}).get("run_dir"),
    }


# ---------------------------------------------------------------------------
# stage orchestration
# ---------------------------------------------------------------------------
def decide_stage(*, stage: str, rows: Sequence[RunRow],
                 state: Mapping[str, Mapping[str, Any]],
                 campaign_root: Any) -> list[dict[str, Any]]:
    """Decide, per pinned row, whether to launch, skip or refuse."""
    decisions: list[dict[str, Any]] = []
    for row in rows:
        prior = state.get(row.run_id) or {}
        prior_outcome = str(prior.get("outcome") or "")
        action = "launch"
        reason = REASON_FORMAL_DONE
        if prior_outcome == "done":
            action = "skip"
            reason = REASON_ALREADY_DONE
        decisions.append({
            "run_id": row.run_id,
            "stage": row.stage,
            "decoder_id": row.decoder_id,
            "track": row.track,
            "seed": int(row.seed),
            "dose": int(row.dose),
            "action": action,
            "reason": reason,
            "prior_outcome": (prior_outcome or None),
            "run_dir": str(run_dir_for(campaign_root, row.run_id)),
        })
    return decisions


def run_stage(stage: str, *, cfg: Mapping[str, Any],
              config_path: Any = DEFAULT_CONFIG,
              campaign_root: Any = DEFAULT_CAMPAIGN_ROOT,
              device: str = "cpu", only: Sequence[str] | None = None,
              dry_run: bool = False, env: Any = None,
              formal_validated: bool = False,
              record_root: Any = None,
              verify_pin: bool = True,
              context_builder: Any = None, trainer: Any = None,
              evaluator: Any = None, evaluate: bool = True,
              data_config: Any = DEFAULT_DATA_CONFIG) -> dict[str, Any]:
    """Decide (+ execute) every pinned row of one formal stage.

    The formal gate runs FIRST (unless the caller already ran it): a null
    budget, an unsigned threshold or a missing/drifted signoff record is a
    typed refusal with the validator's VERBATIM code — no training starts.
    ``--dry-run`` prints the decisions without training and without writing.
    """
    _require(stage in FORMAL_STAGES, CODE_STAGE_UNKNOWN,
             f"stage must be one of {list(FORMAL_STAGES)}, got {stage!r}")
    if not formal_validated:
        formal_gate(config_path, record_root=record_root)

    pin: Mapping[str, Any] | None = None
    if verify_pin:
        verified = verify_runplan_pin(campaign_root, config_path=config_path)
        pin = verified["pin"]
        plan = verified["document"]
    else:  # pragma: no cover - test seam
        plan = load_runplan(campaign_root)
        pin = json.loads(pin_path(campaign_root).read_text(encoding="utf-8")) \
            if pin_path(campaign_root).is_file() else None

    rows = filter_only(stage_rows(plan, stage),
                       None if only is None else list(only))
    state = latest_ledger_state(read_ledger(campaign_root))
    decisions = decide_stage(stage=stage, rows=rows, state=state,
                             campaign_root=campaign_root)
    baseline_doc = Path(campaign_root) / BASELINE_REL
    plan_doc = {
        "schema": FORMAL_SCHEMA,
        "kind": "formal_run_stage",
        "stage": stage,
        "dry_run": bool(dry_run),
        "training_performed": False,
        "device": str(device),
        "execution_device": "cpu",
        "campaign_root": str(campaign_root),
        "ledger_path": str(ledger_path(campaign_root)),
        "runplan": {"path": str(runplan_path(campaign_root)),
                    "pin": str(pin_path(campaign_root)),
                    "runplan_sha256": (None if pin is None
                                       else pin.get("runplan_sha256")),
                    "rows_total": (None if pin is None
                                   else pin.get("rows_total"))},
        "baseline_document": {
            "path": str(baseline_doc),
            "present": baseline_doc.is_file(),
            "role": ("§9.5 M0/P0 recomputed baseline — the only denominator a "
                     "candidate eval may be compared against"),
        },
        "n_core_so_far": new_training_count(state),
        "formal_runs_max": FORMAL_RUNS_MAX,
        "steps": decisions,
        "torch_compile": False,
        "qc_calls": 0,
        "confirm_read": False,
    }
    if stage == "T0":
        plan_doc["t0_note"] = (
            "T0 is the zero-training M0/P0 parity anchor (plan §8.1: 0 "
            "trainings); the frozen parity certificate and the audit/m0_p0_"
            "baseline_v543_eval.json document ARE that stage. Nothing trains.")
        if not dry_run:
            plan_doc["reason"] = REASON_T0_VERIFIED
            plan_doc["verified"] = bool(baseline_doc.is_file())
            _require(baseline_doc.is_file(), CODE_BASELINE_MISSING,
                     "T0 needs the frozen M0/P0 baseline document at "
                     f"{baseline_doc} (§9.5 :1037) — it is the denominator of "
                     "every later comparison")
        return plan_doc
    if dry_run:
        return plan_doc

    _require(new_training_count(state) + sum(
        1 for decision in decisions if decision["action"] == "launch")
        <= FORMAL_RUNS_MAX, CODE_RUN_MATRIX,
        f"launching {sum(1 for d in decisions if d['action'] == 'launch')} "
        f"row(s) would exceed the {FORMAL_RUNS_MAX}-run ceiling (plan §8.1; "
        f"the ledger already counts {new_training_count(state)})")

    if env is None:
        env = campaign.load_campaign_env(cfg, data_config=data_config)
    executed: list[dict[str, Any]] = []
    with campaign.LaneLock(campaign_root):
        for decision, row in zip(decisions, rows):
            if decision["action"] != "launch":
                continue
            record = execute_run(
                env, row, cfg=cfg, config_path=config_path,
                campaign_root=campaign_root, device=device, pin=pin,
                context_builder=context_builder, trainer=trainer,
                evaluator=evaluator, evaluate=evaluate)
            executed.append(record)
    plan_doc["training_performed"] = bool(executed)
    plan_doc["executed"] = [
        {"run_id": record["run_id"], "outcome": record["outcome"],
         "reason": record["reason"],
         "new_training": record["new_training"],
         "best_inner_val_loss": record["best_inner_val_loss"],
         "eval_report": (record["eval"] or {}).get("eval_report_path")}
        for record in executed]
    plan_doc["n_core_after"] = new_training_count(
        latest_ledger_state(read_ledger(campaign_root)))
    return plan_doc


# ---------------------------------------------------------------------------
# status / list
# ---------------------------------------------------------------------------
def status_report(*, campaign_root: Any = DEFAULT_CAMPAIGN_ROOT,
                  config_path: Any = DEFAULT_CONFIG,
                  check_gate: bool = True) -> dict[str, Any]:
    """The ledger state: per run outcome, ``new_training`` and ``N_core``."""
    ledger = read_ledger(campaign_root)
    state = latest_ledger_state(ledger)
    plan_present = runplan_path(campaign_root).is_file()
    pin_present = pin_path(campaign_root).is_file()
    plan_rows: list[RunRow] = []
    if plan_present:
        try:
            plan_rows = runplan_rows(load_runplan(campaign_root))
        except V543FormalError:
            plan_rows = []
    done = [run_id for run_id, row in state.items()
            if str(row.get("outcome")) == "done"]
    pin_state: dict[str, Any] = {"checked": False}
    if pin_present and plan_present:
        try:
            verified = verify_runplan_pin(campaign_root, config_path=config_path)
            pin_state = {"checked": True, "verified": True,
                         "runplan_sha256": verified["runplan_sha256"],
                         "config_sha256": verified["config_sha256"]}
        except V543FormalError as exc:
            pin_state = {"checked": True, "verified": False,
                         "code": exc.code, "detail": exc.detail}
    gate: dict[str, Any] = {"checked": False}
    if check_gate:
        try:
            gate = {"checked": True, "passed": True,
                    **{key: value for key, value in
                       formal_gate(config_path).items()
                       if key in ("mode", "schema", "campaign_id", "status")}}
        except V543FormalError as exc:
            gate = {"checked": True, "passed": False, "code": exc.code,
                    "detail": exc.detail}
    return {
        "schema": FORMAL_SCHEMA,
        "kind": "formal_status",
        "campaign_root": str(campaign_root),
        "ledger_path": str(ledger_path(campaign_root)),
        "runplan": {"present": plan_present, "pin_present": pin_present,
                    "rows": len(plan_rows),
                    "t5_rows": sum(1 for row in plan_rows
                                   if row.stage == "T5")},
        "runplan_pin": pin_state,
        "gate": gate,
        "n_core": new_training_count(state),
        "formal_runs_max": FORMAL_RUNS_MAX,
        "pending_new_trainings": max(
            0, FORMAL_RUNS_MAX - new_training_count(state)),
        "per_run": {
            run_id: {
                "stage": row.get("stage"),
                "decoder_id": row.get("decoder_id"),
                "seed": row.get("seed"),
                "outcome": row.get("outcome"),
                "new_training": row.get("new_training"),
                "reason": row.get("reason"),
                "eval_report": row.get("eval_report"),
                "seq": row.get("seq"),
            } for run_id, row in sorted(state.items())},
        "done": sorted(done),
        "ledger_rows": len(ledger),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def verify_pin_if_frozen(campaign_root: Any, config_path: Any
                         ) -> Mapping[str, Any] | None:
    """Verify an existing pin (no-op when nothing is frozen yet)."""
    if not (runplan_path(campaign_root).is_file()
            and pin_path(campaign_root).is_file()):
        return None
    return verify_runplan_pin(campaign_root, config_path=config_path)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT))
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t5-finalists", default=None,
                        help="pinned T5 finalist list (the screen product; "
                             "at most 2 per track / 4 total)")
    parser.add_argument("--frozen-by", default=None,
                        help="identity recorded in the runplan pin")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan only: no training, no writes")


def _campaign_refusal_from_exception(
        exc: BaseException) -> V543FormalError | None:
    """Convert expected failure classes into typed formal refusals."""
    if isinstance(exc, campaign.V543CampaignError):
        return V543FormalError(exc.code, exc.detail, outcome=exc.outcome)
    try:
        from pes2ts_core.pesn.offline.training.v543_train import (
            V543TrainerError,
        )

        if isinstance(exc, V543TrainerError):
            return V543FormalError(
                "V543_FORMAL_TRAINER_REFUSAL",
                f"frozen trainer refused (code "
                f"{getattr(exc, 'code', None)!r}, outcome "
                f"{getattr(exc, 'outcome', None)!r}): {exc}",
                outcome=getattr(exc, "outcome", None))
    except Exception:  # noqa: BLE001 - trainer import unavailable
        pass
    try:
        from scripts.agent_ops.v543_config_schema import V543SchemaError

        if isinstance(exc, V543SchemaError):
            return V543FormalError(
                getattr(exc, "code", "V543_FORMAL_VALIDATOR_REFUSAL"),
                f"frozen config validator refused: {exc}")
    except Exception:  # noqa: BLE001 - validator unavailable
        pass
    if isinstance(exc, FileNotFoundError):
        return V543FormalError(CODE_BASELINE_MISSING,
                               f"frozen artifact missing: {exc}")
    return None


def _build_plan_document(args: argparse.Namespace, cfg: Mapping[str, Any]
                         ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    finalists = None
    t5_source = None
    if args.t5_finalists:
        finalists = load_t5_finalists(args.t5_finalists)
        t5_source = {
            "path": str(args.t5_finalists),
            "sha256": campaign.sha256_file(args.t5_finalists),
            "finalists": finalists,
            "rule": ("plan §8.2 rules 3/5: the screen product is pinned and "
                     "consumed; it is never re-derived from seed-17/29 "
                     "eyeballing in this runner"),
        }
    plan = build_runplan(cfg, t5_finalists=finalists, t5_source=t5_source,
                         config_path=args.config)
    return plan, finalists


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="v543_formal.py",
        description="V5.4.3 WP8 formal runner (runplan freeze + stage "
                    "orchestration; the formal gate cannot be bypassed)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("runplan", help="print the projected matrix")
    _add_common_args(p_plan)

    p_freeze = sub.add_parser("freeze-runplan",
                              help="write plan/runplan.json + the pin")
    _add_common_args(p_freeze)
    p_freeze.add_argument("--require-formal-gate", action="store_true",
                          help="also require the frozen --formal gate to pass")

    p_run = sub.add_parser("run-stage", help="execute one formal stage")
    _add_common_args(p_run)
    p_run.add_argument("stage", choices=list(FORMAL_STAGES))
    p_run.add_argument("--only", nargs="*", default=None,
                       help="restrict to pinned run ids")
    p_run.add_argument("--skip-pin-verify", action="store_true",
                       help="test seam: trust the on-disk runplan")
    p_run.add_argument("--no-eval", action="store_true",
                       help="skip the evaluator (never used in production)")

    p_status = sub.add_parser("status", help="ledger state")
    _add_common_args(p_status)
    p_status.add_argument("--no-gate", action="store_true")

    p_list = sub.add_parser("list", help="the pinned matrix rows")
    _add_common_args(p_list)

    args = parser.parse_args(argv)
    try:
        cfg = load_formal_config(args.config)
        if args.command in ("runplan", "list", "freeze-runplan"):
            plan, _finalists = _build_plan_document(args, cfg)
            if args.command == "runplan":
                # A pinned matrix that no longer matches its pin is refused
                # even in the planning command (the edit is the refusal).
                if verify_pin_if_frozen(args.campaign_root, args.config):
                    pass
                plan["dry_run"] = True
                print(json.dumps(plan, indent=1, sort_keys=True, default=str))
                return 0
            if args.command == "list":
                verify_pin_if_frozen(args.campaign_root, args.config)
                print(json.dumps({
                    "schema": FORMAL_SCHEMA,
                    "kind": "formal_runplan_list",
                    "rows_total": len(plan["rows"]),
                    "t5_pinned": plan["run_matrix"]["t5_finalists_pinned"],
                    "rows": plan["rows"],
                }, indent=1, sort_keys=True, default=str))
                return 0
            pin = freeze_runplan(
                plan, campaign_root=args.campaign_root,
                frozen_by=(args.frozen_by
                           or "repository_owner_via_dsh_orchestration"),
                require_formal_gate=bool(args.require_formal_gate),
                config_path=args.config)
            print(json.dumps({
                "schema": FORMAL_SCHEMA,
                "kind": "formal_runplan_frozen",
                "runplan": str(runplan_path(args.campaign_root)),
                "pin": str(pin_path(args.campaign_root)),
                "runplan_sha256": pin["runplan_sha256"],
                "rows_total": pin["rows_total"],
                "t5_rows": pin["t5_rows"],
                "t5_finalists_pinned": pin["t5_finalists_pinned"],
            }, indent=1, sort_keys=True, default=str))
            return 0

        if args.command == "status":
            report = status_report(campaign_root=args.campaign_root,
                                   config_path=args.config,
                                   check_gate=not args.no_gate)
            print(json.dumps(report, indent=1, sort_keys=True, default=str))
            return 0

        plan_doc = run_stage(
            args.stage, cfg=cfg, config_path=args.config,
            campaign_root=args.campaign_root, device=args.device,
            only=args.only, dry_run=args.dry_run,
            verify_pin=not args.skip_pin_verify,
            evaluate=not args.no_eval, data_config=args.data_config)
        print(json.dumps(plan_doc, indent=1, sort_keys=True, default=str))
        return 0
    except V543FormalError as exc:
        print(f"V543 formal refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - typed conversion, re-raise below
        typed = _campaign_refusal_from_exception(exc)
        if typed is None:
            raise
        print(f"V543 formal refused: {typed}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
