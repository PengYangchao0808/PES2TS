"""V5.4.1 WP4c — runplan instantiation, pinning and tamper verification.

Instantiates ``outputs/pesn/v541_embed_50k/plan/`` from the frozen
campaign config + the WP0 templates (plan §3.2 / §12.1):

==============================  =========================================
file                            content
==============================  =========================================
``runplan.jsonl``               header + ONE line per preregistered run
                                (objective/seed/status/attempt + every
                                provenance SHA the run must carry)
``runplan.pin.json``            byte-pin of ``runplan.jsonl`` (+ class
                                freeze so diagnostics can never enter the
                                winner count, §9.1)
``preregistration_signoff.json`` thresholds / authorization / T3 block
                                mirrored from the config, SHAs filled
``budget_policy.json``          execution policy + FIXED 37,900 dose;
                                per-candidate budgets stay ``null`` until
                                W0/W1 profiling (the runner never fills
                                defaults)
==============================  =========================================

Rules (fail-closed):

* an EXISTING plan or pin is never overwritten without ``--force``;
* ``--force`` additionally requires that NO run has started (no attempt in
  ``run_records.jsonl``, no run/smoke/diagnostics directory);
* the pin is verified with the repo's own
  :func:`run_ledger.verify_plan_pin` (read-only reuse) — any byte change
  after freeze is ``V541_PLAN_PIN_MISMATCH``;
* embedding/feature SHAs are computed PER CANDIDATE RECIPE, the common
  SHA once per numeric protocol — never one shared digest for all
  candidates.

Typed refusal: :class:`V541RunplanError` (``code`` + ``detail``).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.campaigns.v541_provenance import (  # noqa: E402
    V541ProvenanceError,
    artifact_shas,
    code_sha256,
    common_protocol_sha256,
    embedding_spec_sha256,
    feature_recipe_sha256,
    sha256_file,
    source_manifest,
)

__all__ = [
    "ATTEMPT_EVENTS",
    "FORMAL_MODES",
    "FORMAL_SUBDIRS",
    "PLAN_HEADER_TYPE",
    "PLAN_SCHEMA",
    "RUN_RECORDS_NAME",
    "RUN_SUBDIRS",
    "T3_FULL_ACTIVATION",
    "TECHNICAL_MODES",
    "TECHNICAL_SUBDIRS",
    "V541RunplanError",
    "build_runplan_lines",
    "classify_started_runs",
    "plan_paths",
    "read_plan",
    "started_run_ids",
    "verify_plan_pin_v541",
    "write_plan",
]

PLAN_SCHEMA = "pes2ts_v541_runplan_v1"
PLAN_HEADER_TYPE = "plan_header"
RUN_LINE_TYPE = "v541_run"
T3_FULL_ACTIVATION = 4

TEMPLATE_DIR = PROJECT_ROOT / "config" / "campaigns" / "v541_templates"
TEMPLATES = {
    "pin": "runplan_pin.template.json",
    "signoff": "preregistration_signoff.template.json",
    "budget": "budget_policy.template.json",
}

#: Plan §8.2: a smoke/profile run is NON-formal TECHNICAL evidence.  It
#: survives a plan re-freeze (WP5 back-fills the budget afterwards and the
#: plan is re-frozen with the new formal SHAs) — it never blocks
#: ``plan-init --force``.
TECHNICAL_SUBDIRS: Tuple[str, ...] = ("smoke", "profile")
#: A FORMAL run (``runs/``) or a T4 diagnostic (``diagnostics/``) DID
#: consume a real training slot: their presence keeps refusing a rebuild.
FORMAL_SUBDIRS: Tuple[str, ...] = ("runs", "diagnostics")
#: Every output subdir that can hold run artifacts.
RUN_SUBDIRS: Tuple[str, ...] = FORMAL_SUBDIRS + TECHNICAL_SUBDIRS
#: Record events that mean "a run actually started".
ATTEMPT_EVENTS: Tuple[str, ...] = ("attempt_start", "attempt_end",
                                   "run_complete", "resume_start")
#: Modes that count as formal evidence (blocking).  Anything else that is
#: not smoke/profile — including a record written before ``mode`` existed —
#: is CONSERVATIVELY treated as blocking.
FORMAL_MODES: Tuple[str, ...] = ("formal", "diagnostic")
TECHNICAL_MODES: Tuple[str, ...] = ("smoke", "profile")
#: The V541-only append-only ledger (never the legacy run ledger).
RUN_RECORDS_NAME = "run_records.jsonl"


class V541RunplanError(RuntimeError):
    """Typed refusal of the V541 plan machinery."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)
        self.detail = str(detail)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def plan_paths(output_root: Any) -> Dict[str, Path]:
    """The four plan artifacts (+ the declared source manifest)."""
    root = Path(output_root) / "plan"
    return {
        "plan": root / "runplan.jsonl",
        "pin": root / "runplan.pin.json",
        "signoff": root / "preregistration_signoff.json",
        "budget": root / "budget_policy.json",
        "source_manifest": root / "source_manifest.json",
        "dir": root,
    }


def load_template(name: str) -> Dict[str, Any]:
    """Read one WP0 template (fail-closed: a missing template refuses)."""
    filename = TEMPLATES.get(str(name))
    if filename is None:
        raise V541RunplanError(
            "V541_TEMPLATE_UNKNOWN", "unknown template {!r}".format(name))
    path = TEMPLATE_DIR / filename
    if not path.is_file():
        raise V541RunplanError(
            "V541_TEMPLATE_MISSING", "template missing: {}".format(path))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V541RunplanError(
            "V541_TEMPLATE_CORRUPT", "{}: {}".format(path, exc)) from exc
    if not isinstance(doc, dict):
        raise V541RunplanError(
            "V541_TEMPLATE_CORRUPT", "{} is not a JSON object".format(path))
    return doc


# ---------------------------------------------------------------------------
# runplan lines
# ---------------------------------------------------------------------------
def build_runplan_lines(
    cfg: Mapping[str, Any],
    *,
    config_path: Any,
    tables_dir: Any,
    split_path: Any,
    output_root: Any,
    v53mp_config_path: Any = None,
) -> List[Dict[str, Any]]:
    """Header + one fully-SHA'd line per preregistered run (config matrix)."""
    identity = cfg.get("identity_contract") or {}
    code_contract = identity.get("code") or {}
    manifest = source_manifest(code_contract, root=PROJECT_ROOT)
    shas = artifact_shas(
        cfg,
        config_path=config_path,
        tables_dir=tables_dir,
        split_path=split_path,
        code_sha=str(manifest["code_sha256"]),
        v53mp_config_path=v53mp_config_path,
    )
    common_sha = common_protocol_sha256(
        cfg,
        code_sha=str(manifest["code_sha256"]),
        data_manifest_sha256=shas["data_manifest_sha256"],
        teacher_manifest_sha256=shas["teacher_manifest_sha256"],
        split_sha256=shas["split_sha256"],
        label_spec_sha256=shas["label_spec_sha256"],
    )
    budget_path = plan_paths(output_root)["budget"]
    if not budget_path.is_file():
        raise V541RunplanError(
            "V541_BUDGET_MISSING",
            "budget_policy.json must be instantiated before runplan.jsonl "
            "({})".format(budget_path),
        )
    budget_sha = sha256_file(budget_path)

    runs = (cfg.get("run_matrix") or {}).get("runs")
    if not isinstance(runs, Sequence) or not runs:
        raise V541RunplanError(
            "V541_RUN_MATRIX", "config run_matrix.runs missing/empty")

    header: Dict[str, Any] = {
        "record_type": PLAN_HEADER_TYPE,
        "schema": PLAN_SCHEMA,
        "campaign_id": cfg.get("campaign_id"),
        "campaign_mode": "full",
        "objective": cfg.get("objective"),
        "generated_at": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": shas["config_sha256"],
        "code_sha256": shas["code_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "git_commit": shas["git_commit"],
        "environment_sha256": shas["environment_sha256"],
        "common_protocol_sha256": common_sha,
        "budget_sha256": budget_sha,
        "r3_init_sha256": shas["r3_init_sha256"],
        "data_manifest_sha256": shas["data_manifest_sha256"],
        "teacher_manifest_sha256": shas["teacher_manifest_sha256"],
        "split_sha256": shas["split_sha256"],
        "label_spec_sha256": shas["label_spec_sha256"],
        "endpoint_label_spec_sha256": shas["endpoint_label_spec_sha256"],
        "run_count": len(runs),
        "t3_full_activation": T3_FULL_ACTIVATION,
        "new_qc_calls": 0,
        "confirm": False,
        "matrix_ref": [
            "config/campaigns/pes2ts_v541_embed_50k.yaml#run_matrix",
            "plan §9.1 / §12.1",
        ],
    }
    lines: List[Dict[str, Any]] = [header]
    for entry in runs:
        if not isinstance(entry, Mapping):
            raise V541RunplanError(
                "V541_RUN_MATRIX", "run entry not a mapping: {!r}".format(entry))
        embedding_id = str(entry.get("embedding"))
        seed = entry.get("seed")
        if not isinstance(seed, int):
            raise V541RunplanError(
                "V541_RUN_MATRIX",
                "{}: seed must be an int, got {!r}".format(
                    entry.get("run_id"), seed),
            )
        line: Dict[str, Any] = {
            "record_type": RUN_LINE_TYPE,
            "schema": PLAN_SCHEMA,
            "run_id": str(entry.get("run_id")),
            "campaign_id": cfg.get("campaign_id"),
            "objective": cfg.get("objective"),
            "stage": entry.get("stage"),
            "class": entry.get("class"),
            "embedding_id": embedding_id,
            "seed": int(seed),
            "status": "planned",
            "attempt": 0,
            "confirm": False,
            "new_qc_calls": 0,
            "single_lane": True,
            "common_protocol_sha256": common_sha,
            "embedding_spec_sha256": embedding_spec_sha256(cfg, embedding_id),
            "feature_recipe_sha256": feature_recipe_sha256(cfg, embedding_id),
            "budget_sha256": budget_sha,
        }
        line.update(shas)
        # per-candidate rows keep the condition/gate provenance verbatim
        for key in ("condition", "parent_gate", "unactivated_status",
                    "winner_eligible", "new_training",
                    "new_training_if_activated", "control"):
            if key in entry:
                line[key] = entry[key]
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# write / read / verify
# ---------------------------------------------------------------------------
def classify_started_runs(output_root: Any) -> Dict[str, Dict[str, Any]]:
    """Split started runs into ``blocking`` (formal/diagnostic) vs ``technical``.

    plan §8.2 / §17: a smoke/profile run is NON-formal technical evidence —
    WP5 measures the budget on it and then RE-FREEZES the plan with the new
    formal code/common/plan SHAs.  Those technical artifacts must therefore
    never lock ``plan-init --force``: they are preserved verbatim while the
    plan is rebuilt.  A formal run (``runs/``) or a T4 diagnostic
    (``diagnostics/``) DID consume a real training slot, so both the
    directory and the matching ledger record keep refusing the rebuild —
    and nothing is ever deleted or overwritten to get around that.

    Ledger records are classified by their ``mode`` field; a record without
    one (legacy / unknown) is CONSERVATIVELY blocking, and an unreadable
    ledger line blocks too — an unreadable ledger is never read as "no runs".
    """
    root = Path(output_root)
    blocking: Dict[str, Dict[str, Any]] = {}
    technical: Dict[str, Dict[str, Any]] = {}

    def _add(bucket: Dict[str, Dict[str, Any]], run_id: str, source: str,
             mode: str) -> None:
        bucket[str(run_id)] = {"source": str(source), "mode": str(mode)}

    records = root / RUN_RECORDS_NAME
    if records.is_file():
        for lineno, raw in enumerate(
                records.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                _add(blocking, "<unreadable-record-{}>".format(lineno),
                     "run_records", "unknown")
                continue
            if not isinstance(rec, Mapping):
                _add(blocking, "<unreadable-record-{}>".format(lineno),
                     "run_records", "unknown")
                continue
            if str(rec.get("event")) not in ATTEMPT_EVENTS:
                continue                      # e.g. a refusal: no attempt
            run_id = rec.get("run_id")
            if not run_id:
                _add(blocking, "<unlabelled-attempt-{}>".format(lineno),
                     "run_records", "unknown")
                continue
            mode = str(rec.get("mode") or "").lower()
            if mode in TECHNICAL_MODES:
                _add(technical, str(run_id), "run_records", mode)
            elif mode in FORMAL_MODES:
                _add(blocking, str(run_id), "run_records", mode)
            else:
                # unknown / pre-mode record -> conservatively blocking
                _add(blocking, str(run_id), "run_records", mode or "unknown")

    for sub in FORMAL_SUBDIRS:
        base = root / sub
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and any(child.iterdir()):
                _add(blocking, child.name, sub,
                     "formal" if sub == "runs" else "diagnostic")
    for sub in TECHNICAL_SUBDIRS:
        base = root / sub
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and any(child.iterdir()):
                _add(technical, child.name, sub, sub)
    return {"blocking": blocking, "technical": technical}


def started_run_ids(output_root: Any) -> List[str]:
    """Every run that has actually started (blocking AND technical)."""
    classified = classify_started_runs(output_root)
    return sorted(set(classified["blocking"]) | set(classified["technical"]))


def write_plan(
    output_root: Any,
    cfg: Mapping[str, Any],
    *,
    config_path: Any,
    tables_dir: Any,
    split_path: Any,
    force: bool = False,
    v53mp_config_path: Any = None,
    frozen_by: str = "v541_run.plan-init",
) -> Dict[str, Any]:
    """Instantiate all four plan artifacts (never over a live plan/pin).

    ``--force`` re-freeze rules (plan §8.2 / §17):

    * only TECHNICAL artifacts (``smoke/`` / ``profile/`` + their
      ``run_records.jsonl`` attempts) may coexist with a rebuild — they are
      PRESERVED VERBATIM and the new pin binds the fresh formal
      code / common-protocol / plan SHAs;
    * ANY formal (``runs/``) or diagnostic (``diagnostics/``) attempt still
      refuses (``V541_PLAN_STARTED_RUNS``) and no evidence is deleted or
      overwritten to get around it;
    * ``--force`` remains explicit, and the budget/threshold signoff rules
      are untouched (per-candidate budgets stay ``null`` until W0/W1
      profiling — this function never fills a default).
    """
    paths = plan_paths(output_root)
    existing = [name for name, path in paths.items()
                if name != "dir" and path.exists()]
    preserved_technical: List[str] = []
    if existing:
        if "pin" in existing and not force:
            raise V541RunplanError(
                "V541_PLAN_PIN_EXISTS",
                "runplan.pin.json already exists — an existing pin is never "
                "overwritten without an explicit --force re-freeze "
                "({})".format(paths["pin"]),
            )
        if not force:
            raise V541RunplanError(
                "V541_PLAN_EXISTS",
                "plan artifact(s) {} already exist; rebuild requires "
                "--force + re-signoff".format(sorted(existing)),
            )
        started = classify_started_runs(output_root)
        blocking = sorted(started["blocking"])
        if blocking:
            raise V541RunplanError(
                "V541_PLAN_STARTED_RUNS",
                "--force refused: formal/diagnostic run(s) already started "
                "({}); a frozen plan with real attempts can never be "
                "rebuilt in place and their evidence is never deleted or "
                "overwritten".format(blocking),
            )
        preserved_technical = sorted(started["technical"])

    paths["dir"].mkdir(parents=True, exist_ok=True)

    # 1) budget policy (per-candidate budgets stay NULL — no defaults)
    budget = load_template("budget")
    budget["config_path"] = str(config_path)
    budget["config_sha256"] = sha256_file(config_path)
    budget["instantiated_at"] = _utc_now()
    _write_json(paths["budget"], budget)
    budget_sha = sha256_file(paths["budget"])

    # 2) runplan.jsonl (header + one line per run)
    lines = build_runplan_lines(
        cfg, config_path=config_path, tables_dir=tables_dir,
        split_path=split_path, output_root=output_root,
        v53mp_config_path=v53mp_config_path)
    _write_jsonl(paths["plan"], lines)
    plan_sha = sha256_file(paths["plan"])

    # 3) pin (byte-pin of the plan + class freeze)
    pin = load_template("pin")
    pin["config_path"] = str(config_path)
    pin["config_sha256"] = sha256_file(config_path)
    pin["sha256"] = plan_sha
    pin["campaign_mode"] = "full"
    pin["runplan_sha256"] = plan_sha
    pin["budget_sha256"] = budget_sha
    pin["budget_policy_sha256"] = budget_sha
    pin["code_sha256"] = lines[0]["code_sha256"]
    pin["source_manifest_sha256"] = lines[0]["source_manifest_sha256"]
    pin["code_commit"] = lines[0]["git_commit"]
    pin["environment_sha256"] = lines[0]["environment_sha256"]
    pin["common_protocol_sha256"] = lines[0]["common_protocol_sha256"]
    pin["frozen_on"] = _utc_now()
    pin["frozen_by"] = frozen_by
    _write_json(paths["pin"], pin)

    # 4) preregistration signoff (thresholds stay byte-identical to config)
    signoff = load_template("signoff")
    signoff["config_path"] = str(config_path)
    signoff["config_sha256"] = sha256_file(config_path)
    signoff["runplan_sha256"] = plan_sha
    signoff["budget_policy_sha256"] = budget_sha
    signoff["code_commit"] = lines[0]["git_commit"]
    signoff["code_sha256"] = lines[0]["code_sha256"]
    signoff["environment_sha256"] = lines[0]["environment_sha256"]
    signoff["common_protocol_sha256"] = lines[0]["common_protocol_sha256"]
    signoff["instantiated_at"] = _utc_now()
    _write_json(paths["signoff"], signoff)

    # 5) declared source manifest (code identity, plan §12.2 provenance)
    manifest = source_manifest(
        (cfg.get("identity_contract") or {}).get("code") or {},
        root=PROJECT_ROOT)
    _write_json(paths["source_manifest"], manifest)

    return {
        "action": "plan_init",
        "plan": str(paths["plan"]),
        "pin": str(paths["pin"]),
        "signoff": str(paths["signoff"]),
        "budget": str(paths["budget"]),
        "source_manifest": str(paths["source_manifest"]),
        "run_count": len(lines) - 1,
        "run_ids": [ln["run_id"] for ln in lines[1:]],
        "plan_sha256": plan_sha,
        "budget_sha256": budget_sha,
        "code_sha256": lines[0]["code_sha256"],
        "environment_sha256": lines[0]["environment_sha256"],
        "common_protocol_sha256": lines[0]["common_protocol_sha256"],
        "rebuilt": bool(existing),
        # plan §8.2 / §17: a re-freeze may coexist with (preserved)
        # technical smoke/profile evidence — the new pin binds the FORMAL
        # SHAs below, and budget/threshold signoff rules are unchanged.
        "refreeze": {
            "forced": bool(force),
            "rebuilt": bool(existing),
            "preserved_technical_runs": preserved_technical,
            "technical_evidence_policy": (
                "smoke/profile artifacts and their run_records attempts are "
                "preserved verbatim (plan §8.2 non-formal evidence); only a "
                "formal or diagnostic attempt blocks a re-freeze"),
            "new_formal_code_sha256": lines[0]["code_sha256"],
            "new_formal_common_protocol_sha256":
                lines[0]["common_protocol_sha256"],
            "new_formal_plan_sha256": plan_sha,
            "new_formal_budget_sha256": budget_sha,
            "new_formal_environment_sha256":
                lines[0]["environment_sha256"],
        },
    }


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(doc), ensure_ascii=False, indent=2, sort_keys=True,
                   default=str) + "\n",
        encoding="utf-8")


def _write_jsonl(path: Path, lines: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(ln), ensure_ascii=False, sort_keys=True,
                           default=str) + "\n" for ln in lines),
        encoding="utf-8")


def read_plan(plan_path: Any) -> Dict[str, Any]:
    """Parse ``runplan.jsonl`` → header / by_run_id / run_ids (fail-closed)."""
    path = Path(plan_path)
    if not path.is_file():
        raise V541RunplanError(
            "V541_PLAN_MISSING", "runplan.jsonl not found: {}".format(path))
    lines: List[Dict[str, Any]] = []
    for lineno, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise V541RunplanError(
                "V541_PLAN_CORRUPT",
                "{} line {}: {}".format(path, lineno, exc)) from exc
        if not isinstance(rec, dict):
            raise V541RunplanError(
                "V541_PLAN_CORRUPT",
                "{} line {} is not a JSON object".format(path, lineno))
        lines.append(rec)
    if not lines:
        raise V541RunplanError(
            "V541_PLAN_CORRUPT", "{} is empty".format(path))
    header = lines[0]
    if header.get("record_type") != PLAN_HEADER_TYPE:
        raise V541RunplanError(
            "V541_PLAN_CORRUPT", "first line is not a plan_header")
    by_run_id: Dict[str, Dict[str, Any]] = {}
    for rec in lines[1:]:
        run_id = rec.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise V541RunplanError(
                "V541_PLAN_CORRUPT", "run line without run_id")
        if run_id in by_run_id:
            raise V541RunplanError(
                "V541_PLAN_CORRUPT", "duplicate run_id {}".format(run_id))
        by_run_id[run_id] = rec
    return {
        "header": header,
        "by_run_id": by_run_id,
        "run_ids": sorted(by_run_id),
        "mode": header.get("campaign_mode", "full"),
        "lines": lines,
    }


def verify_plan_pin_v541(plan_path: Any, pin_path: Any = None) -> Dict[str, Any]:
    """Byte-verify ``runplan.jsonl`` against its pin (read-only reuse).

    Delegates the digest/mode comparison to the repo's own
    ``run_ledger.verify_plan_pin`` so the tamper semantics can never drift
    from the legacy WP-L3 discipline, then additionally re-checks the
    V541-only identity fields (code / common / environment SHAs).
    """
    plan = Path(plan_path)
    pin_file = Path(pin_path) if pin_path is not None else Path(
        str(plan) + ".pin.json")
    if not pin_file.is_file():
        raise V541RunplanError(
            "V541_PLAN_PIN_MISSING",
            "runplan.pin.json not found: {} (an un-pinned plan is never "
            "launchable)".format(pin_file))
    try:
        pin = json.loads(pin_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V541RunplanError(
            "V541_PLAN_PIN_CORRUPT", "{}: {}".format(pin_file, exc)) from exc
    if not isinstance(pin, dict):
        raise V541RunplanError(
            "V541_PLAN_PIN_CORRUPT", "{} is not a JSON object".format(pin_file))

    from scripts.agent_ops.ledger.run_ledger import verify_plan_pin

    try:
        verified = verify_plan_pin(plan, pin)
    except Exception as exc:  # noqa: BLE001 - legacy typed errors -> V541 code
        raise V541RunplanError(
            "V541_PLAN_PIN_MISMATCH", str(exc)) from exc

    parsed = read_plan(plan)
    header = parsed["header"]
    for field in ("code_sha256", "environment_sha256",
                  "common_protocol_sha256", "budget_sha256",
                  "config_sha256"):
        if pin.get(field) is None:
            raise V541RunplanError(
                "V541_PLAN_PIN_INCOMPLETE",
                "pin is missing {} — a plan frozen without the full "
                "provenance block is not launchable".format(field))
        if header.get(field) != pin.get(field):
            raise V541RunplanError(
                "V541_PLAN_PIN_MISMATCH",
                "header {} = {} but pin carries {}".format(
                    field, header.get(field), pin.get(field)),
            )
    return {
        "pin": dict(verified),
        "plan": str(plan),
        "pin_path": str(pin_file),
        "run_ids": parsed["run_ids"],
        "code_sha256": header.get("code_sha256"),
        "environment_sha256": header.get("environment_sha256"),
        "common_protocol_sha256": header.get("common_protocol_sha256"),
    }


def current_code_sha256(cfg: Mapping[str, Any]) -> str:
    """Recompute the code fingerprint from disk (drift detection)."""
    try:
        return code_sha256((cfg.get("identity_contract") or {}).get("code") or {},
                           root=PROJECT_ROOT)
    except V541ProvenanceError:
        raise
