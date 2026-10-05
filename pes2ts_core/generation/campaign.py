"""G2.T campaign orchestration: 24-case frozen-manifest end-to-end runner.

G2-AB2 WP-1/WP-4.  One campaign = one frozen manifest
(``pes2ts_campaign_manifest_v1``: 24 input snapshots + hashes, frozen
parameters/budgets, ACP wiring, review status, round2/round3 references)
executed case by case into typed terminal states
(``pes2ts_g2t_terminal_state_v1``) with an R5 cost ledger per case.

Disciplines frozen here (constitution R5/P2/P3 + plan §3):

- **Idempotent resume** — a case whose terminal state exists, whose input
  snapshot hash still matches the manifest, and whose terminal class is not
  ``blocked`` is skipped; blocked cases (environment/review) are re-attempted.
  Resuming never changes parameters or budget: a manifest edit is a NEW
  campaign identity (content digest) and refuses to reuse the same root.
- **Four-class generation terminals** — ``rejected_typed`` / ``partial_prefix``
  / ``complete_path`` (+ ``validated_*`` once a typed ValidationResult is
  attached, WP-5).  ``blocked`` (environment / human review / campaign budget)
  and ``censored_budget`` (per-path budget exhausted, data kept) are explicit
  auditable states, never folded into a success denominator.
- **Never-skip** — a missing ACP environment produces ``blocked`` terminals
  with a typed needs list; failures keep their partial artifacts, logs and
  costs (失败也留数).  Nothing is silently skipped or retried past budget.
- **Cost before conclusion** — every terminal references a ``costs.json``
  derived through :mod:`cost_capture` (R5); no ledger, no report.

The continuation stack itself (plan building, corrector caching, candidate
ranking) is untouched: the campaign only orchestrates the AB1 pieces behind
injectable per-case backends, so offline tests drive synthetic correctors and
production wires :class:`ORCALocalCorrector` + :class:`ORCAOriginPreparation`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from pes2ts_core.flywheel import LABEL_STATES
from pes2ts_core.generation.planning.continuation import (
    ContinuationPolicy,
    run_continuation,
)
from pes2ts_core.generation.planning.local_corrector import (
    BranchPolicy,
    LocalCorrectorPolicy,
)
from pes2ts_core.generation.planning.cost_capture import (
    capture_continuation_costs,
    capture_origin_cost,
    total_gradient_calls,
)
from pes2ts_core.generation.planning.origin_preparation import (
    ENDPOINT_METHOD_EVIDENCE_SCHEMA,
    ORIGIN_FAILURE_CODES,
    prepare_origin,
)
from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.integration.acp.origin_backend import OriginBackendUnavailable
from pes2ts_core.utils.jsonio import read_json, write_json

#: Registered interfaces (constitution §9.6 via ADR-0004).
CAMPAIGN_MANIFEST_SCHEMA = "pes2ts_campaign_manifest_v1"
TERMINAL_STATE_SCHEMA = "pes2ts_g2t_terminal_state_v1"

#: Terminal-class vocabulary.  The first three + ``validated_*`` are the
#: report's frozen four-class denominators; ``blocked`` and ``censored_budget``
#: are explicit non-success audit states that must never be counted into a
#: success fraction (§11.4).
GENERATION_TERMINAL_CLASSES = ("rejected_typed", "partial_prefix", "complete_path")
VALIDATED_TERMINAL_CLASSES = ("validated_passed", "validated_failed",
                              "validated_incomplete")
AUDIT_TERMINAL_CLASSES = ("blocked", "censored_budget")
TERMINAL_CLASSES = GENERATION_TERMINAL_CLASSES + VALIDATED_TERMINAL_CLASSES \
    + AUDIT_TERMINAL_CLASSES

#: Stage order of the generation track (轨道 A).
CASE_STAGES = ("input", "origin_preparation", "plan", "continuation",
               "candidate_extraction")

#: Typed blocked reasons (P2: a blocked case carries a needs list).
BLOCKED_ENVIRONMENT = "acp_environment_unavailable"
BLOCKED_HUMAN_REVIEW = "blocked_on_human_review"
BLOCKED_CAMPAIGN_BUDGET = "campaign_budget_exhausted"

_REVIEW_STATES = ("needs_review", "accepted", "rejected")


class CampaignError(ValueError):
    """Typed campaign infrastructure error (manifest/root discipline)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_campaign_manifest(case_snapshots: Sequence[Path | str], *,
                            output_path: Path | str | None = None,
                            pilot_cases: Sequence[str] = (),
                            continuation_policy: ContinuationPolicy | None = None,
                            local_policy: LocalCorrectorPolicy | None = None,
                            branch_policy: BranchPolicy | None = None,
                            origin_preparation: dict[str, Any] | None = None,
                            budget: dict[str, Any] | None = None,
                            acp_wiring: dict[str, Any] | None = None,
                            review_status: dict[str, str] | None = None,
                            reference_failures: dict[str, Any] | None = None,
                            campaign_label: str = "g2t") -> dict[str, Any]:
    """Freeze one campaign manifest from concrete input snapshots.

    Snapshot order is frozen sorted-by-reaction-id; every case records the
    snapshot's sha256 so a later resume can prove the inputs never moved.
    ``review_status`` defaults to ``needs_review`` for every case (the Demo24
    truth today); ``reference_failures`` carries the round2/round3 known-fail
    λ per case for B1-2 clearance bookkeeping.
    """
    cases = []
    for raw in sorted(case_snapshots, key=lambda p: Path(p).stem):
        path = Path(raw)
        payload = path.read_text(encoding="utf-8")
        import hashlib
        cases.append({"reaction_id": path.stem,
                      "snapshot_path": str(path),
                      "snapshot_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()})
    if not cases:
        raise CampaignError("EMPTY_CAMPAIGN")
    ids = [case["reaction_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise CampaignError("DUPLICATE_CASE_ID")
    pilot = list(pilot_cases)
    unknown = [rid for rid in pilot if rid not in set(ids)]
    if unknown:
        raise CampaignError(f"PILOT_CASE_NOT_IN_CAMPAIGN:{unknown[0]}")
    manifest = {
        "schema_version": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_label": campaign_label,
        "cases": cases,
        "pilot_cases": pilot,
        "frozen_parameters": {
            "continuation_policy": dict(sorted(asdict(
                continuation_policy or ContinuationPolicy()).items())),
            "local_policy": dict(sorted(asdict(
                local_policy or LocalCorrectorPolicy()).items())),
            "branch_policy": dict(sorted(asdict(
                branch_policy or BranchPolicy()).items())),
            "origin_preparation": dict(origin_preparation or
                                       {"method": "GFN2-xTB", "timeout_seconds": 600.0,
                                        "capability_probe": True}),
            "candidate_top_k": 3,
        },
        "budget": _default_budget(budget),
        "acp_wiring": {
            "root": (acp_wiring or {}).get("root"),
            "python": (acp_wiring or {}).get("python"),
            "config_path": (acp_wiring or {}).get("config_path"),
            "resolve_from_environment": True,
        },
        "review_status": {rid: (review_status or {}).get(rid, "needs_review")
                          for rid in ids},
        "reference_failures": dict(reference_failures or {}),
        "generated_at": _now(),
    }
    validate_campaign_manifest(manifest)
    manifest["content_sha256"] = digest(
        {k: v for k, v in manifest.items() if k != "content_sha256"})
    if output_path is not None:
        write_json(Path(output_path), manifest)
    return manifest


def _default_budget(overrides: dict[str, Any] | None) -> dict[str, Any]:
    budget = {"max_frames_per_path": ContinuationPolicy().max_frames,
              "max_attempts_per_path": ContinuationPolicy().max_attempts,
              "max_wall_seconds_per_path": ContinuationPolicy().max_seconds,
              "max_gradient_calls_per_path": 2000,
              "max_total_gradient_calls": 60000,
              "max_session_wall_seconds": 0.0}  # 0 = no session wall budget
    budget.update(overrides or {})
    return budget


def validate_campaign_manifest(manifest: dict[str, Any]) -> None:
    """Structural + policy validation of a frozen campaign manifest (P1/P3)."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != CAMPAIGN_MANIFEST_SCHEMA:
        raise CampaignError("INVALID_CAMPAIGN_MANIFEST_SCHEMA")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CampaignError("EMPTY_CAMPAIGN")
    ids: list[str] = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("reaction_id"), str):
            raise CampaignError("INVALID_CASE_RECORD")
        if not isinstance(case.get("snapshot_path"), str):
            raise CampaignError("INVALID_CASE_SNAPSHOT_PATH")
        sha = case.get("snapshot_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise CampaignError("INVALID_CASE_SNAPSHOT_HASH")
        ids.append(case["reaction_id"])
    if len(set(ids)) != len(ids):
        raise CampaignError("DUPLICATE_CASE_ID")
    frozen = manifest.get("frozen_parameters")
    if not isinstance(frozen, dict):
        raise CampaignError("INVALID_FROZEN_PARAMETERS")
    try:
        ContinuationPolicy(**frozen["continuation_policy"])
        LocalCorrectorPolicy(**frozen["local_policy"])
        BranchPolicy(**frozen["branch_policy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"INVALID_FROZEN_POLICY:{exc}") from exc
    top_k = frozen.get("candidate_top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 3:
        raise CampaignError("INVALID_CANDIDATE_TOP_K")
    budget = manifest.get("budget")
    if not isinstance(budget, dict):
        raise CampaignError("INVALID_BUDGET")
    for key in ("max_frames_per_path", "max_attempts_per_path",
                "max_gradient_calls_per_path", "max_total_gradient_calls"):
        value = budget.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise CampaignError(f"INVALID_BUDGET:{key}")
    for key in ("max_wall_seconds_per_path", "max_session_wall_seconds"):
        value = budget.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise CampaignError(f"INVALID_BUDGET:{key}")
    # Single source of truth: the per-path attempt/frame/wall ceilings are the
    # frozen continuation policy itself; a diverging budget block would create
    # two truths for the same knob (P3).
    policy = ContinuationPolicy(**frozen["continuation_policy"])
    for budget_key, policy_key in (("max_frames_per_path", "max_frames"),
                                   ("max_attempts_per_path", "max_attempts"),
                                   ("max_wall_seconds_per_path", "max_seconds")):
        if budget.get(budget_key) != getattr(policy, policy_key):
            raise CampaignError(f"BUDGET_POLICY_MISMATCH:{budget_key}")
    wiring = manifest.get("acp_wiring")
    if not isinstance(wiring, dict):
        raise CampaignError("INVALID_ACP_WIRING")
    review = manifest.get("review_status")
    if not isinstance(review, dict) or set(review) != set(ids):
        raise CampaignError("REVIEW_STATUS_MUST_COVER_EXACTLY_THE_CASES")
    for rid, state in review.items():
        if state not in _REVIEW_STATES:
            raise CampaignError(f"INVALID_REVIEW_STATUS:{rid}")
    pilot = manifest.get("pilot_cases")
    if not isinstance(pilot, list) or any(rid not in set(ids) for rid in pilot):
        raise CampaignError("INVALID_PILOT_CASES")
    references = manifest.get("reference_failures")
    if not isinstance(references, dict) or any(rid not in set(ids) for rid in references):
        raise CampaignError("INVALID_REFERENCE_FAILURES")
    if "content_sha256" in manifest:
        expected = digest({k: v for k, v in manifest.items() if k != "content_sha256"})
        if manifest["content_sha256"] != expected:
            raise CampaignError("MANIFEST_CONTENT_HASH_MISMATCH")


def load_campaign_manifest(path: Path | str) -> dict[str, Any]:
    manifest = read_json(Path(path))
    validate_campaign_manifest(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

@dataclass
class CampaignBackends:
    """Injectable per-case production seams.

    ``prepare_origin(geometry, elements, bond_indices, charge, multiplicity,
    case_root)`` returns an ``endpoint_method_evidence_v1`` dict (or raises
    :class:`OriginBackendUnavailable`).  ``make_corrector(plan, case_root)``
    returns the :func:`run_continuation` corrector (production:
    ``ORCALocalCorrector``).
    """

    prepare_origin: Callable[..., dict[str, Any]]
    make_corrector: Callable[[dict[str, Any], Path], Any]


def production_backends(*, acp_wiring: dict[str, Any] | None = None,
                        origin_config: dict[str, Any] | None = None,
                        local_policy: LocalCorrectorPolicy | None = None,
                        branch_policy: BranchPolicy | None = None) -> CampaignBackends:
    """Wire the ACP production origin preparation and local corrector."""
    from pes2ts_core.integration.acp.gradient_backend import ORCALocalCorrector
    from pes2ts_core.integration.acp.origin_backend import ORCAOriginPreparation

    config = dict(origin_config or {})
    wiring = dict(acp_wiring or {})

    def prepare_origin_backend(geometry, elements, bond_indices, charge, multiplicity,
                               case_root: Path):
        backend = ORCAOriginPreparation(
            elements=list(elements), charge=charge, multiplicity=multiplicity,
            folder=case_root / "origin_backend", method=config.get("method", "GFN2-xTB"),
            timeout_seconds=float(config.get("timeout_seconds", 600.0)),
            nproc=int(config.get("nproc", 2)), acp_wiring=wiring,
            capability_probe=bool(config.get("capability_probe", True)))
        return prepare_origin(
            geometry, elements, bond_indices, backend.evaluate_free,
            method_context={"method": backend.method, "charge": charge,
                            "multiplicity": multiplicity, "elements": list(elements)},
            budget_seconds=float(config.get("timeout_seconds", 600.0)))

    def make_corrector(plan: dict[str, Any], case_root: Path):
        return ORCALocalCorrector(
            plan, case_root / "attempts", policy=local_policy,
            branch_policy=branch_policy, acp_wiring=wiring)

    return CampaignBackends(prepare_origin=prepare_origin_backend,
                            make_corrector=make_corrector)


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------

def _terminal_doc(reaction_id: str, manifest: dict[str, Any], *, terminal_class: str,
                  stage: str, case_sha256: str,
                  failure_code: str | None = None, failure_detail: Any = None,
                  blocked_reason: str | None = None, blocked_needs: Sequence[str] = (),
                  identity_conflicts: int = 0, result: dict[str, Any] | None = None,
                  candidates: dict[str, Any] | None = None,
                  origin_evidence: dict[str, Any] | None = None,
                  costs: dict[str, Any] | None = None,
                  evidence_refs: dict[str, str] | None = None,
                  label_state: str = "unattempted") -> dict[str, Any]:
    if terminal_class not in TERMINAL_CLASSES:
        raise CampaignError(f"INVALID_TERMINAL_CLASS:{terminal_class}")
    if label_state not in LABEL_STATES:
        raise CampaignError(f"INVALID_LABEL_STATE:{label_state}")
    n_frames = len((result or {}).get("frames", []))
    last_lambda = (result or {}).get("last_lambda")
    completed = bool((result or {}).get("completed_interval", False))
    doc: dict[str, Any] = {
        "schema_version": TERMINAL_STATE_SCHEMA,
        "reaction_id": reaction_id,
        "campaign_sha256": manifest["content_sha256"],
        "case_input_sha256": case_sha256,
        "terminal_class": terminal_class,
        "label_state": label_state,
        "stage_reached": stage,
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "blocked_reason": blocked_reason,
        "blocked_needs": list(blocked_needs),
        "identity_conflict_count": int(identity_conflicts),
        "n_accepted_frames_beyond_origin": max(0, n_frames - 1),
        "last_accepted_lambda": last_lambda,
        "completed_interval": completed,
        "origin_status": (origin_evidence or {}).get("status", "not_reached"),
        "origin_failure_code": (origin_evidence or {}).get("failure_code"),
        "candidates": ({"n": len(candidates.get("candidates", [])),
                        "evidence_classes": [row.get("evidence_class")
                                              for row in candidates.get("candidates", [])],
                        "completed_interval": candidates.get("completed_interval"),
                        "ref": "candidates.json"}
                       if candidates is not None else None),
        "costs": ({"n_gradient": int(((costs or {}).get("aggregate") or {}).get("N_gradient") or 0),
                   "cpu_seconds": (costs or {}).get("aggregate", {}).get("cpu_seconds") if (costs or {}).get("aggregate") else None,
                   "ref": "costs.json"} if costs is not None else None),
        "validation": {"status": BLOCKED_HUMAN_REVIEW,
                       "needs": ["accepted ReviewRecord for this case "
                                 "(two-reviewer workbook + adjudication + import)"]},
        "evidence_refs": evidence_refs or {},
        "generated_at": _now(),
    }
    return doc


def _generation_terminal_class(result: dict[str, Any]) -> tuple[str, str, Any]:
    """Map one continuation result onto the frozen four-class vocabulary."""
    completed = bool(result.get("completed_interval"))
    n_beyond = max(0, len(result.get("frames", [])) - 1)
    if completed:
        return "complete_path", None, None
    if n_beyond >= 1:
        return "partial_prefix", (result.get("status") or "incomplete"), {
            "last_lambda": result.get("last_lambda"),
            "n_accepted_frames_beyond_origin": n_beyond}
    status = result.get("status") or "NO_FRAMES"
    code = status.split(":", 1)[0] if ":" in status else status
    return "rejected_typed", code, {"continuation_status": status}


def _classify_budget(costs: dict[str, Any] | None, *, budget: dict[str, Any],
                     doc: dict[str, Any]) -> dict[str, Any]:
    """Re-classify a terminal as ``censored_budget`` when the path budget died."""
    if costs is None:
        return doc
    used = int(costs.get("aggregate", {}).get("N_gradient") or 0)
    if used > int(budget["max_gradient_calls_per_path"]):
        doc = dict(doc)
        doc["terminal_class"] = "censored_budget"
        doc["label_state"] = "censored_budget"
        doc["failure_code"] = "PATH_GRADIENT_BUDGET_EXCEEDED"
        doc["failure_detail"] = {"n_gradient": used,
                                 "max_gradient_calls_per_path":
                                     budget["max_gradient_calls_per_path"]}
    return doc


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class CampaignRunSummary:
    manifest_path: str
    root: str
    n_cases: int = 0
    n_executed: int = 0
    n_skipped_idempotent: int = 0
    n_blocked: int = 0
    terminal_counts: dict[str, int] = field(default_factory=dict)
    session_wall_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"manifest_path": self.manifest_path, "root": self.root,
                "n_cases": self.n_cases, "n_executed": self.n_executed,
                "n_skipped_idempotent": self.n_skipped_idempotent,
                "n_blocked": self.n_blocked,
                "terminal_counts": dict(sorted(self.terminal_counts.items())),
                "session_wall_seconds": round(self.session_wall_seconds, 3)}


def run_campaign(manifest_path: Path | str, root: Path | str, *,
                 case_filter: Sequence[str] | None = None,
                 resume: bool = True,
                 backends: CampaignBackends | None = None,
                 runner: Callable[[str, dict[str, Any]], None] | None = None) -> CampaignRunSummary:
    """Execute (or resume) one campaign; every case ends in a typed terminal.

    ``runner`` is an optional progress callback ``runner(reaction_id,
    terminal_doc)`` invoked after each case terminal is written.
    """
    manifest = load_campaign_manifest(manifest_path)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _assert_root_belongs_to_campaign(root, manifest)
    by_id = {case["reaction_id"]: case for case in manifest["cases"]}
    selected = [case["reaction_id"] for case in manifest["cases"]
                if case_filter is None or case["reaction_id"] in set(case_filter)]
    unknown = [rid for rid in (case_filter or []) if rid not in by_id]
    if unknown:
        raise CampaignError(f"CASE_NOT_IN_CAMPAIGN:{unknown[0]}")
    if backends is None:
        backends = production_backends(
            acp_wiring=manifest["acp_wiring"],
            origin_config=manifest["frozen_parameters"]["origin_preparation"],
            local_policy=LocalCorrectorPolicy(**manifest["frozen_parameters"]["local_policy"]),
            branch_policy=BranchPolicy(**manifest["frozen_parameters"]["branch_policy"]))
    policy = ContinuationPolicy(**manifest["frozen_parameters"]["continuation_policy"])
    budget = manifest["budget"]
    session_budget = (float(budget["max_session_wall_seconds"]) or None)
    started = time.monotonic()
    summary = CampaignRunSummary(manifest_path=str(manifest_path), root=str(root),
                                 n_cases=len(selected))
    used_gradients = _campaign_gradient_total(root, manifest)
    for reaction_id in selected:
        case_root = root / reaction_id
        existing = _load_existing_terminal(case_root, manifest)
        if existing is not None and resume:
            # A completed case is never re-blocked by a later budget state.
            summary.n_skipped_idempotent += 1
            _bump(summary, existing)
            continue
        if session_budget is not None and time.monotonic()-started > session_budget \
                or used_gradients >= int(budget["max_total_gradient_calls"]):
            if existing is None and (case_root / "terminal_state.json").is_file():
                # Existing blocked terminal: the budget cannot re-attempt it,
                # so keep the artifact (and its needs) untouched.
                existing = read_json(case_root / "terminal_state.json")
                summary.n_blocked += 1
                _bump(summary, existing)
                if runner:
                    runner(reaction_id, existing)
                continue
            doc = _terminal_doc(reaction_id, manifest, terminal_class="blocked",
                                stage="input", case_sha256=by_id[reaction_id]["snapshot_sha256"],
                                blocked_reason=BLOCKED_CAMPAIGN_BUDGET,
                                blocked_needs=[
                                    f"campaign budget exhausted: N_gradient={used_gradients} "
                                    f"(max {budget['max_total_gradient_calls']})",
                                    "split the remaining cases into a NEW campaign manifest"],
                                label_state="censored_budget")
            _write_case_artifacts(case_root, terminal=doc,
                                  costs=_merge_case_costs([], reaction_id))
            summary.n_blocked += 1
            _bump(summary, doc)
            if runner:
                runner(reaction_id, doc)
            continue
        doc = _execute_case(reaction_id, by_id[reaction_id], case_root,
                            manifest, policy, backends)
        _bump(summary, doc)
        summary.n_executed += 1
        if doc["terminal_class"] == "blocked":
            summary.n_blocked += 1
        costs_doc = read_json(case_root / "costs.json") \
            if (case_root / "costs.json").is_file() else None
        used_gradients += int(((costs_doc or {}).get("aggregate") or {}).get("N_gradient") or 0)
        if runner:
            runner(reaction_id, doc)
    summary.session_wall_seconds = time.monotonic()-started
    write_json(root / "campaign_run_summary.json", {**summary.as_dict(),
                                                           "generated_at": _now()})
    return summary


def _bump(summary: CampaignRunSummary, doc: dict[str, Any]) -> None:
    summary.terminal_counts[doc["terminal_class"]] = \
        summary.terminal_counts.get(doc["terminal_class"], 0) + 1


def _assert_root_belongs_to_campaign(root: Path, manifest: dict[str, Any]) -> None:
    """A root is bound to one campaign identity; edits mean a new root (P3)."""
    marker = root / "campaign_identity.json"
    identity = {"campaign_sha256": manifest["content_sha256"],
                "campaign_label": manifest.get("campaign_label"),
                "generated_at": _now()}
    if marker.is_file():
        stored = read_json(marker)
        if stored.get("campaign_sha256") != identity["campaign_sha256"]:
            raise CampaignError(
                "CAMPAIGN_ROOT_IDENTITY_MISMATCH: manifest content changed; "
                "use a new output root (parameters/budget edits are a new campaign)")
        return
    write_json(marker, identity)


def _campaign_gradient_total(root: Path, manifest: dict[str, Any]) -> int:
    """Cumulative gradient cost of every completed case in this root."""
    bundles = []
    for case in manifest["cases"]:
        path = root / case["reaction_id"] / "costs.json"
        if path.is_file():
            bundles.append(read_json(path))
    return total_gradient_calls(*bundles)


def _load_existing_terminal(case_root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    path = case_root / "terminal_state.json"
    if not path.is_file():
        return None
    doc = read_json(path)
    if doc.get("schema_version") != TERMINAL_STATE_SCHEMA:
        raise CampaignError(f"INVALID_TERMINAL_STATE:{case_root.name}")
    if doc.get("campaign_sha256") != manifest["content_sha256"]:
        raise CampaignError("CAMPAIGN_ROOT_IDENTITY_MISMATCH: terminal from another campaign")
    if doc.get("terminal_class") == "blocked":
        return None  # blocked cases are re-attempted (typed needs may now be met)
    case = next(c for c in manifest["cases"] if c["reaction_id"] == case_root.name)
    if doc.get("case_input_sha256") != case["snapshot_sha256"]:
        raise CampaignError("CASE_INPUT_HASH_CHANGED")
    return doc


def _write_case_artifacts(case_root: Path, *, terminal: dict[str, Any],
                          origin: dict[str, Any] | None = None,
                          plan: dict[str, Any] | None = None,
                          result: dict[str, Any] | None = None,
                          candidates: dict[str, Any] | None = None,
                          costs: dict[str, Any] | None = None) -> None:
    """Persist every stage artifact; a failure keeps its partial data."""
    case_root.mkdir(parents=True, exist_ok=True)
    if origin is not None:
        write_json(case_root / "origin_evidence.json", origin)
    if plan is not None:
        write_json(case_root / "plan.json", plan)
    if result is not None:
        write_json(case_root / "continuation_result.json", result)
    if candidates is not None:
        write_json(case_root / "candidates.json", candidates)
    if costs is not None:
        write_json(case_root / "costs.json", costs)
    write_json(case_root / "terminal_state.json", terminal)


def _execute_case(reaction_id: str, case: dict[str, Any], case_root: Path,
                  manifest: dict[str, Any], policy: ContinuationPolicy,
                  backends: CampaignBackends) -> dict[str, Any]:
    """Run the full generation track for one case; ALWAYS yields a terminal."""
    sha = case["snapshot_sha256"]
    import hashlib
    payload = Path(case["snapshot_path"]).read_text(encoding="utf-8")
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != sha:
        raise CampaignError(f"SNAPSHOT_HASH_MISMATCH:{reaction_id}")
    snapshot = json.loads(payload)
    stage = "input"
    origin_evidence: dict[str, Any] | None = None
    costs_parts: list[dict[str, Any]] = []
    identity_conflicts = 0
    from pes2ts_core.generation.planning.connectivity_plan import (
        build_plan, choose_origin,
    )
    from pes2ts_core.generation.planning.graph_rebuild import (
        load_endpoint_materials_from_export, rebuild_endpoint_graphs,
    )
    from pes2ts_core.generation.planning.structure_checks import add_online_checks
    from pes2ts_core.generation.planning.candidate_selection import (
        rank_continuation_candidates,
    )
    try:
        bundle = rebuild_endpoint_graphs(snapshot["reaction_smiles"],
                                         load_endpoint_materials_from_export(snapshot))
        choice = choose_origin(snapshot, bundle)
        if choice.get("status") != "ready":
            doc = _terminal_doc(reaction_id, manifest, terminal_class="rejected_typed",
                                stage=stage, case_sha256=sha,
                                failure_code=choice.get("status"),
                                failure_detail={"choose_origin": choice},
                                label_state="execution_failed")
            _write_case_artifacts(case_root, terminal=doc)
            return doc
        side_geometry = np.asarray(
            snapshot["r_coordinates" if choice["side"] == "R" else "p_coordinates"], float)
        stage = "origin_preparation"
        try:
            origin_evidence = backends.prepare_origin(
                side_geometry, snapshot["elements"], choice["bond_indices"],
                choice["charge"], choice["multiplicity"], case_root)
        except OriginBackendUnavailable as exc:
            costs = _merge_case_costs([], reaction_id)
            doc = _terminal_doc(reaction_id, manifest, terminal_class="blocked",
                                stage=stage, case_sha256=sha,
                                failure_code="ORIGIN_BACKEND_UNAVAILABLE",
                                failure_detail=exc.evidence,
                                blocked_reason=BLOCKED_ENVIRONMENT,
                                blocked_needs=exc.needs, costs=costs,
                                label_state="censored_budget")
            _write_case_artifacts(case_root, terminal=doc, costs=costs)
            return doc
        if origin_evidence.get("schema_version") != ENDPOINT_METHOD_EVIDENCE_SCHEMA:
            raise CampaignError("INVALID_ORIGIN_EVIDENCE_SCHEMA")
        origin_cost = capture_origin_cost(origin_evidence, reaction_id=reaction_id)
        if origin_cost is not None:
            costs_parts.append(origin_cost)
        if origin_evidence.get("status") != "prepared":
            costs = _merge_case_costs(costs_parts, reaction_id)
            doc = _terminal_doc(reaction_id, manifest, terminal_class="rejected_typed",
                                stage=stage, case_sha256=sha,
                                failure_code=origin_evidence.get("failure_code"),
                                failure_detail=origin_evidence.get("failure_detail"),
                                origin_evidence=origin_evidence, costs=costs,
                                label_state="execution_failed")
            _write_case_artifacts(case_root, terminal=doc, origin=origin_evidence,
                                  costs=costs)
            return doc
        stage = "plan"
        plan = build_plan(snapshot, bundle, choice,
                          np.asarray(origin_evidence["coordinates"], float), policy=policy)
        plan = add_online_checks(plan, snapshot, bundle)
        write_json(case_root / "plan.json", plan)
        stage = "continuation"
        corrector = backends.make_corrector(plan, case_root)
        initial = {"success": True, "converged": True,
                   "coordinates": origin_evidence["coordinates"],
                   "energy": origin_evidence.get("energy"),
                   "duration_seconds": origin_evidence.get("duration_seconds", 0.0)}
        result = run_continuation(
            plan, initial, corrector,
            save=lambda snapshot_doc: write_json(
                case_root / "continuation_result.json", snapshot_doc))
        stage = "candidate_extraction"
        candidates = rank_continuation_candidates(
            result, plan, top_k=manifest["frozen_parameters"]["candidate_top_k"],
            allow_edge_candidates=True)
    except ValueError as exc:
        # Typed lower-stack rejection (plan validation, identity conflict, …).
        text = str(exc)
        if "IDENTITY_CONFLICT" in text:
            identity_conflicts += 1
        failure_code = "IDENTITY_CONFLICT" if identity_conflicts else (
            text.split(":", 1)[0] if text else "UNTYPED_VALUE_ERROR")
        doc = _terminal_doc(reaction_id, manifest, terminal_class="rejected_typed",
                            stage=stage, case_sha256=sha, failure_code=failure_code,
                            failure_detail={"error": text[:2000]},
                            origin_evidence=origin_evidence,
                            identity_conflicts=identity_conflicts,
                            label_state="execution_failed")
        costs = _merge_case_costs(costs_parts, reaction_id)
        _write_case_artifacts(case_root, terminal=doc, origin=origin_evidence,
                              costs=costs)
        return doc
    costs = _merge_case_costs(
        costs_parts + [capture_continuation_costs(result, reaction_id=reaction_id)],
        reaction_id)
    terminal_class, failure_code, detail = _generation_terminal_class(result)
    label_state = {"complete_path": "unattempted",
                   "partial_prefix": "unattempted",
                   "rejected_typed": "optimizer_failed"}[terminal_class]
    doc = _terminal_doc(reaction_id, manifest, terminal_class=terminal_class,
                        stage=stage, case_sha256=sha, failure_code=failure_code,
                        failure_detail=detail, result=result, candidates=candidates,
                        origin_evidence=origin_evidence, costs=costs,
                        evidence_refs={"plan": "plan.json",
                                       "continuation_result": "continuation_result.json",
                                       "candidates": "candidates.json",
                                       "costs": "costs.json",
                                       "origin_evidence": "origin_evidence.json"},
                        label_state=label_state)
    doc = _classify_budget(costs, budget=manifest["budget"], doc=doc)
    _write_case_artifacts(case_root, terminal=doc, origin=origin_evidence,
                          result=result, candidates=candidates, costs=costs)
    return doc


def _merge_case_costs(parts: list[dict[str, Any]], reaction_id: str) -> dict[str, Any]:
    ledgers: list[dict[str, Any]] = []
    for part in parts:
        if "ledgers" in part:          # a cost bundle (capture_continuation_costs)
            ledgers.extend(part["ledgers"] or [])
        elif "cost_id" in part:        # a single ledger (origin / probe)
            ledgers.append(part)
    from pes2ts_core.flywheel import aggregate_costs
    return {"schema_version": "pes2ts_case_costs_v1", "reaction_id": reaction_id,
            "ledgers": ledgers,
            "aggregate": aggregate_costs(ledgers) if ledgers else None}


def attach_validation_result(terminal: dict[str, Any], validation_result: dict[str, Any],
                             *, preparation_layer_status: str = "absent") -> dict[str, Any]:
    """WP-5 bridge: fold one typed ValidationResult into a terminal state.

    Chemical failure is a legal terminal: ``failed``/``incomplete`` map to
    their own ``validated_*`` classes and never to an engineering failure.
    """
    if terminal.get("schema_version") != TERMINAL_STATE_SCHEMA:
        raise CampaignError("INVALID_TERMINAL_STATE")
    status = validation_result.get("status")
    if status not in {"passed", "failed", "incomplete"}:
        raise CampaignError(f"INVALID_VALIDATION_STATUS:{status}")
    doc = dict(terminal)
    doc["terminal_class"] = f"validated_{status}"
    doc["label_state"] = ("verified_target" if status == "passed" else
                          "verified_non_target" if status == "failed"
                          else "validation_incomplete")
    doc["validation"] = {"status": status,
                         "preparation_layer_status": preparation_layer_status,
                         "result_ref": validation_result.get("object_id")}
    if status == "passed":
        doc["validation"]["evidence_refs"] = {
            stage: validation_result.get(stage, {}).get("attempt_id")
            for stage in ("optts", "frequency", "irc_forward", "irc_reverse")}
    doc["generated_at"] = _now()
    return doc


# ---------------------------------------------------------------------------
# Read-only status
# ---------------------------------------------------------------------------

def campaign_status(root: Path | str) -> dict[str, Any]:
    """Read-only progress/terminal/cost summary of a campaign root."""
    root = Path(root)
    marker = root / "campaign_identity.json"
    if not marker.is_file():
        raise CampaignError("NOT_A_CAMPAIGN_ROOT")
    terminals = []
    for path in sorted(root.glob("RXN_*/terminal_state.json")):
        terminals.append(read_json(path))
    counts: dict[str, int] = {}
    for doc in terminals:
        counts[doc["terminal_class"]] = counts.get(doc["terminal_class"], 0) + 1
    gradient_total = 0
    cpu_values: list[float] = []
    for path in sorted(root.glob("RXN_*/costs.json")):
        costs = read_json(path)
        aggregate = costs.get("aggregate") or {}
        gradient_total += int(aggregate.get("N_gradient") or 0)
        cpu = aggregate.get("cpu_seconds")
        if isinstance(cpu, (int, float)):
            cpu_values.append(float(cpu))
    blocked_needs = sorted({need for doc in terminals
                            for need in doc.get("blocked_needs", [])})
    return {"campaign_sha256": read_json(marker).get("campaign_sha256"),
            "n_cases_with_terminal": len(terminals),
            "terminal_counts": dict(sorted(counts.items())),
            "four_class_denominators": {
                "rejected_typed": counts.get("rejected_typed", 0)
                + counts.get("censored_budget", 0),
                "partial_prefix": counts.get("partial_prefix", 0),
                "complete_path": counts.get("complete_path", 0),
                "validated": sum(counts.get(name, 0)
                                 for name in VALIDATED_TERMINAL_CLASSES),
                "blocked": counts.get("blocked", 0)},
            "cost_totals": {"N_gradient": gradient_total,
                            "cpu_seconds_known_cases": len(cpu_values),
                            "cpu_seconds_sum_known": round(sum(cpu_values), 3)},
            "blocked_needs": blocked_needs,
            "generated_at": _now()}


__all__ = [
    "AUDIT_TERMINAL_CLASSES", "BLOCKED_CAMPAIGN_BUDGET", "BLOCKED_ENVIRONMENT",
    "BLOCKED_HUMAN_REVIEW", "CAMPAIGN_MANIFEST_SCHEMA", "CampaignBackends",
    "CampaignError", "CampaignRunSummary", "GENERATION_TERMINAL_CLASSES",
    "ORIGIN_FAILURE_CODES", "OriginBackendUnavailable", "TERMINAL_CLASSES",
    "TERMINAL_STATE_SCHEMA", "VALIDATED_TERMINAL_CLASSES",
    "attach_validation_result", "build_campaign_manifest", "campaign_status",
    "load_campaign_manifest", "production_backends", "run_campaign",
    "validate_campaign_manifest",
]
