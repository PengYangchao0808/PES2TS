"""Append-only JSONL run ledger for the PES2TS 30h campaign (plan section 7.2).

Design rules enforced here (30h plan ``docs/plans/PES2TS_30h_R3_V52_V53_..._v1.1``):

* Record schema follows plan section 7.2 ``run_record.json``:
  campaign_id, phase, arm, fold, seed, n_updates, data_sha256, teacher_sha256,
  features_sha256, split_sha256, config_sha256, checkpoint_update,
  checkpoint_sha256, status, metrics_path (plus optional ``variant``).
* ``run_id`` is deterministic: ``campaign_id:phase:arm:f{fold}:s{seed}[:variant]``.
  Re-appending the exact same record (same run_id, same content) is an
  idempotent no-op (safe resubmission after interruption); re-appending a
  *different* record under the same run_id raises ``RUN_ID_CONFLICT``.
* ``validate()`` rejects ``REQUIRED_AT_RUNTIME`` placeholder hashes, missing
  ``*_sha256`` fields and invalid hex digests -- placeholders are never valid
  evidence.
* Status machine: planned -> running -> (done | failed | killed);
  planned -> skipped; running -> interrupted (reconcile only);
  failed/interrupted -> running is a retry.  ``retry_count`` is maintained by
  the ledger: retries 1 and 2 are free, a 3rd retry attempt (retry_count == 3)
  requires ``supervisor_override=true`` in the incoming record.
* Every mutation writes the new ledger line and an event in ``events.jsonl``
  (sibling of the ledger file) while holding a sidecar flock.  Files are
  rewritten via tmp + ``os.replace`` so readers only ever observe a complete
  file; the ledger content itself is strictly append-only (existing lines are
  never modified or removed).
* ``reconcile`` marks ``running`` records whose holder pid is dead as
  ``interrupted`` -- it never deletes anything.

WP-L2 hardening (fail-closed; Oracle ORA-1):

* Formal records carry ``gate_id`` (non-empty str), ``admissibility``
  (``admissible`` | ``not_admissible_until_gate_pass``), ``evidence_sha``
  (sha256 of the gate report; required real hex once admissible), ``attempt``
  (int >= 1) and ``parent_run_id`` (nullable str).  A record carrying any of
  the formal keys is a *formal* record: all five keys are then mandatory and
  ``run_id`` switches to the hashed scheme
  ``sha256(campaign_id|phase|arm|fold|seed|variant|config|data|teacher|features|split)[:16]``
  so changed-config-same-name becomes a **different run** instead of silent
  reuse.  Legacy records (none of the formal keys) keep the historical
  ``campaign:phase:arm:fN:sN`` id -- pre-WP-L2 ledgers stay valid and the
  state machine (``RunIdConflict`` on identity drift) is unchanged for them.
* Budget caps (task_state ``budget.run_caps``): any transition into
  ``running`` counts one launch attempt; attempts include retries and are
  cross-checked against the bound ``run_plan.jsonl`` (``plan_path``): the
  campaign+phase plan count is the cap (R3=66, V5.2=33), launching a run not
  EXACTLY matching a planned run_id (identity = full hashed run_id incl.
  variant), or a phase absent from a bound plan, raises
  :class:`BudgetCapExceeded`; a plan line whose five identity shas are still
  ``PENDING_*``/zero at launch raises :class:`PlanNotFrozenError`.
* ``confirm_unseal`` (ORA-2/3): the token is consumed FOR A BATCH of
  pre-registered ``confirm: true`` run_ids (C-stage + QT-confirm lines of the
  bound plan): first launch opens the batch for all of its runs (idempotent
  per run), runs outside the batch and second/different batches raise
  :class:`UnsealExceeded`.  ``metric_unseal`` is a separate one-shot token
  for B-stage folds3/4 sealed-baseline metric reveal (30h §4.2).
* Partial (PD-4, 21-run) plans force ``admissibility =
  not_admissible_exploratory`` at launch and refuse ``scale_controlled``/
  ``V5.1b`` keywords in any finalize (:class:`ExploratoryOnlyError`).

WP-L3 hardening (fail-closed; E0 residuals #1/#2):

* **Pinned-plan verification on EVERY mutation** (residual: Oracle showed a
  tampered copy of the frozen ``run_plan.jsonl`` -- confirm flags flipped, or
  a partial-mode plan re-headered as ``full`` -- still appended because only
  ``runplan verify`` checked identity while ``append`` never did).  Whenever
  a ledger is bound to a ``plan_path``, every ``append`` /
  ``confirm_unseal`` / ``metric_unseal`` call first recomputes the plan's
  sha256 and re-reads ``campaign_mode`` from its header and compares both
  against the E0-frozen pin (``task_state.e0_freeze.run_plan_pinned`` /
  ``protocol_30h.json`` frozen_artifacts; compiled here as
  :data:`DEFAULT_PLAN_PIN`).  ANY byte change -- confirm flags, header mode,
  whitespace -- is ``PlanNotFrozenError`` ``PLAN_SHA_MISMATCH`` /
  ``PLAN_MODE_MISMATCH`` and nothing is written.  Re-pinning requires a
  ``--force`` plan rebuild + re-freeze + human re-signoff (never in-band).
* **B folds3/4 metric reveal is consumed, not decorative** (residual #2): a
  ``done`` transition for a plan line with ``substage == "B"`` and ``fold in
  (3, 4)`` consumes the one-shot ``metric_unseal`` token for the plan's B
  folds3/4 batch (idempotent inside the batch; :class:`UnsealExceeded`
  ``UNSEAL_OUTSIDE_BATCH`` / ``UNSEAL_BATCH_MISMATCH`` /
  ``METRIC_UNSEAL_NOT_CONFIGURED`` outside) and records the consumption in
  the run record (``metric_unseal`` field) + an audit event.  A sealed-metric
  run can never reach ``done`` without burning the token.
* ``mark_blocked`` lets the supervisor monitor annotate a run with
  ``blocked_with_reason`` (e.g. a :class:`PlanNotFrozenError`) without
  crashing and without a status transition; the annotation clears on the
  next real transition.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

__all__ = [
    "AppendOutcome",
    "BudgetCapExceeded",
    "CAMPAIGN_PIN",
    "DEFAULT_CAPS",
    "DEFAULT_PLAN_PIN",
    "DEFAULT_TASK_STATE_SUBPATH",
    "ExploratoryOnlyError",
    "IllegalTransition",
    "LedgerError",
    "MISSING_PLACEHOLDER",
    "PINNED_CAMPAIGN_MODE",
    "PINNED_RUN_PLAN_SHA256",
    "PLACEHOLDER",
    "PlanNotFrozenError",
    "RetryCapExceeded",
    "RunIdConflict",
    "RunLedger",
    "RunNotFound",
    "UnsealExceeded",
    "compute_run_id",
    "confirm_unseal",
    "is_confirm_run",
    "is_formal_record",
    "is_frozen_sha",
    "load_campaign_pin",
    "main",
    "metric_unseal",
    "pid_alive",
    "read_run_plan",
    "sha256_file",
    "validate",
    "verify_plan_pin",
]

# --------------------------------------------------------------------------
# constants / schema (plan section 7.2)
# --------------------------------------------------------------------------

PLACEHOLDER = "REQUIRED_AT_RUNTIME"
MISSING_PLACEHOLDER = PLACEHOLDER
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_HASH_FIELDS = (
    "data_sha256",
    "teacher_sha256",
    "features_sha256",
    "split_sha256",
    "config_sha256",
)
OPTIONAL_HASH_FIELDS = ("checkpoint_sha256",)
ALL_HASH_FIELDS = REQUIRED_HASH_FIELDS + OPTIONAL_HASH_FIELDS

#: keys that must exist in every record (values may be None only where noted)
REQUIRED_KEYS = (
    "campaign_id",
    "phase",
    "arm",
    "fold",
    "seed",
    "n_updates",
    "data_sha256",
    "teacher_sha256",
    "features_sha256",
    "split_sha256",
    "config_sha256",
    "checkpoint_update",
    "checkpoint_sha256",
    "status",
    "metrics_path",
)

#: identity of a run -- may never change under one run_id
IDENTITY_FIELDS = (
    "campaign_id",
    "phase",
    "arm",
    "fold",
    "seed",
    "variant",
    "n_updates",
    "data_sha256",
    "teacher_sha256",
    "features_sha256",
    "split_sha256",
    "config_sha256",
)

#: fields compared for idempotent-duplicate detection (volatile fields excluded)
CONTENT_FIELDS = IDENTITY_FIELDS + (
    "status",
    "checkpoint_update",
    "checkpoint_sha256",
    "metrics_path",
    # WP-L2 formal metadata: drift under the same status is a conflict, never
    # a silent overwrite (identity fields above still never change at all).
    "gate_id",
    "admissibility",
    "evidence_sha",
    "attempt",
    "parent_run_id",
    # explicit confirm-run flag (mirrors run_plan ``confirm: true``; ORA-2/2
    # forbids heuristic arm/variant detection)
    "confirm",
)

KNOWN_STATUSES = ("planned", "running", "done", "failed", "killed", "skipped", "interrupted")
INITIAL_STATUSES = ("planned",)
ALLOWED_TRANSITIONS: Dict[str, frozenset] = {
    "planned": frozenset({"running", "skipped"}),
    "running": frozenset({"done", "failed", "killed", "interrupted"}),
    "failed": frozenset({"running"}),
    "interrupted": frozenset({"running"}),
    "done": frozenset(),
    "killed": frozenset(),
    "skipped": frozenset(),
}
#: leaving one of these states by re-entering ``running`` counts as a retry
RETRY_SOURCES = frozenset({"failed", "interrupted"})
MAX_RETRIES = 2  # retry_count <= 2 free of charge; the 3rd retry needs override

#: -- WP-L2 formal-record schema ------------------------------------------
FORMAL_FIELDS = ("gate_id", "admissibility", "evidence_sha", "attempt", "parent_run_id")
ADMISSIBILITY_VALUES = (
    "admissible",
    "not_admissible_until_gate_pass",
    # PD-4 partial (21-run exploratory) campaigns can NEVER be admissible:
    "not_admissible_exploratory",
)

#: statuses that mean "this run has been launched at least once"
LAUNCHED_STATES = frozenset({"running", "done", "failed", "killed", "interrupted"})

#: campaign budget caps (task_state budget.run_caps; counts ALL launch attempts)
DEFAULT_CAPS: Dict[str, int] = {"R3": 66, "V5.2": 33}

#: default unseal token locations (relative to the campaign root)
DEFAULT_UNSEAL_SUBPATH = "docs/agent_runs/state/gates/unseal.json"
DEFAULT_METRIC_UNSEAL_SUBPATH = "docs/agent_runs/state/gates/metric_unseal.json"

#: E0-frozen pin of the campaign run_plan.jsonl (task_state.e0_freeze.
#: run_plan_pinned / protocol_30h.json frozen_artifacts.run_plan -- WP-L3).
#: Re-pinning requires a --force plan rebuild + re-freeze + human re-signoff;
#: it is NEVER changed in-band by ledger/tooling code.
PINNED_RUN_PLAN_SHA256 = (
    "a8972a1958385ed2373bba00af677125992239b8af0e10745257983f2b7cf051"
)
PINNED_CAMPAIGN_MODE = "full"
DEFAULT_PLAN_PIN = {"sha256": PINNED_RUN_PLAN_SHA256, "campaign_mode": PINNED_CAMPAIGN_MODE}
#: task_state location carrying e0_freeze.run_plan_pinned (pin source of truth)
DEFAULT_TASK_STATE_SUBPATH = "docs/agent_runs/state/task_state.json"

#: keywords forbidden in any finalize of a partial (PD-4) campaign plan
PARTIAL_FORBIDDEN_FINALIZE_KEYWORDS = ("scale_controlled", "v5.1b")

DEFAULT_LEDGER_PATH = "outputs/pesn/30h_campaign/run_records.jsonl"
EVENTS_FILENAME = "events.jsonl"


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class LedgerError(Exception):
    """Base class for ledger violations."""


class RunIdConflict(LedgerError):
    """Same run_id resubmitted with different content."""


class PlaceholderHashError(LedgerError):
    """Placeholder (REQUIRED_AT_RUNTIME) / missing / invalid sha256 evidence."""


class IllegalTransition(LedgerError):
    """Status transition outside the plan state machine."""


class RetryCapExceeded(LedgerError):
    """Retry budget exhausted without supervisor_override=true."""


class BudgetCapExceeded(LedgerError):
    """Campaign run cap (R3=66 / V5.2=33, all attempts incl. retries) exceeded,
    or a launch not backed by the bound ``run_plan.jsonl``."""


class PlanNotFrozenError(LedgerError):
    """Plan line for this run still carries PENDING/zero identity shas, or the
    launched run_id does not exactly match the planned run_id (ORA-2/1,2)."""


class ExploratoryOnlyError(LedgerError):
    """A partial (PD-4) plan can never yield an admissible record."""


class UnsealExceeded(LedgerError):
    """fold3/4 confirm / metric unseal token missing/corrupt/already consumed
    for a different batch."""


class RunNotFound(LedgerError):
    """Unknown run_id."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pid_alive(pid: Any) -> bool:
    """Best-effort liveness check; unknown/invalid pids count as dead."""
    try:
        pid_int = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else
    except OverflowError:
        return False
    return True


def compute_run_id(record: Dict[str, Any], scheme: Optional[str] = None) -> str:
    """Deterministic run_id for a record.

    ``scheme`` is one of ``"legacy"`` (``campaign:phase:arm:fN:sN[:variant]``),
    ``"hashed"`` (``sha256(campaign|phase|arm|fold|seed|variant|config_sha|
    data_sha|teacher_sha|features_sha|split_sha)[:16]``) or ``"auto"`` /
    ``None``: formal records (carrying any formal key -- ``gate_id``,
    ``admissibility``, ``attempt``, ...) get the hashed scheme so a changed
    config under the same name is a *different* run; legacy records keep the
    historical id (backward compatible with pre-WP-L2 ledgers).
    """
    resolved = scheme or "auto"
    if resolved == "auto":
        resolved = "hashed" if is_formal_record(record) else "legacy"
    if resolved == "legacy":
        parts = [
            str(record["campaign_id"]),
            str(record["phase"]),
            str(record["arm"]),
            "f{}".format(record["fold"]),
            "s{}".format(record["seed"]),
        ]
        variant = record.get("variant")
        if variant:
            parts.append(str(variant))
        return ":".join(parts)
    if resolved != "hashed":
        raise LedgerError("INVALID_SCHEME: {!r} not in auto/legacy/hashed".format(scheme))
    joined = "|".join(
        [
            str(record["campaign_id"]),
            str(record["phase"]),
            str(record["arm"]),
            str(record["fold"]),
            str(record["seed"]),
            str(record.get("variant") or ""),
            str(record.get("config_sha256")),
            str(record.get("data_sha256")),
            str(record.get("teacher_sha256")),
            str(record.get("features_sha256")),
            str(record.get("split_sha256")),
        ]
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def is_formal_record(record: Dict[str, Any]) -> bool:
    """True when the record carries any WP-L2 formal-run key."""
    return any(key in record for key in ("gate_id", "admissibility", "attempt"))


def is_confirm_run(record: Dict[str, Any]) -> bool:
    """Confirm runs are declared EXPLICITLY (``confirm: true``), never inferred.

    The run_plan builder emits ``confirm: true`` only for fold3/4 confirm
    lines (R3 arm-C folds3/4 and V5.2 QT-confirm); ORA-2 blocker 2 forbids
    arm/variant heuristics.
    """
    return record.get("confirm") is True


def _identity(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: record.get(k) for k in IDENTITY_FIELDS}


def _content(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: record.get(k) for k in CONTENT_FIELDS}


def validate(record: Any) -> Dict[str, Any]:
    """Validate a section 7.2 record.  Returns the record on success.

    Rejects:
      * non-dict records / missing schema keys
      * any ``*_sha256 == "REQUIRED_AT_RUNTIME"`` placeholder
      * missing (None) required frozen hashes or non-64-hex digests
      * unknown/illegal status values, bad fold/seed/n_updates types
      * ``done`` records without checkpoint + metrics evidence
    """
    if not isinstance(record, dict):
        raise LedgerError("RECORD_NOT_OBJECT: record must be a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in record]
    if missing:
        raise PlaceholderHashError(
            "MISSING_FIELD: record is missing required field(s): {}".format(
                ", ".join(missing)
            )
        )
    for key in ALL_HASH_FIELDS:
        if key not in record:
            raise PlaceholderHashError("MISSING_FIELD: record is missing {}".format(key))
    for key, value in record.items():
        if not str(key).endswith("_sha256"):
            continue
        if value == PLACEHOLDER:
            raise PlaceholderHashError(
                "PLACEHOLDER_HASH: {} == {!r} is not valid evidence".format(key, PLACEHOLDER)
            )
    for key in REQUIRED_HASH_FIELDS:
        value = record.get(key)
        if value is None:
            raise PlaceholderHashError(
                "MISSING_HASH: {} is required and must be a real sha256".format(key)
            )
        if not (isinstance(value, str) and HASH_RE.match(value)):
            raise PlaceholderHashError(
                "INVALID_HASH: {} must be a lowercase 64-char sha256 hex digest".format(key)
            )
    ckpt = record.get("checkpoint_sha256")
    if ckpt is not None and not (isinstance(ckpt, str) and HASH_RE.match(ckpt)):
        raise PlaceholderHashError(
            "INVALID_HASH: checkpoint_sha256 must be null or a 64-char sha256 hex digest"
        )
    status = record.get("status")
    if status not in KNOWN_STATUSES:
        raise LedgerError(
            "INVALID_STATUS: {!r} not in {}".format(status, ", ".join(KNOWN_STATUSES))
        )
    for key in ("fold", "seed", "n_updates"):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise LedgerError("INVALID_FIELD: {} must be an int, got {!r}".format(key, value))
    if record["fold"] < 0:
        raise LedgerError("INVALID_FIELD: fold must be >= 0")
    if record["n_updates"] < 0:
        raise LedgerError("INVALID_FIELD: n_updates must be >= 0")
    for key in ("campaign_id", "phase", "arm"):
        value = record.get(key)
        if not (isinstance(value, str) and value):
            raise LedgerError("INVALID_FIELD: {} must be a non-empty string".format(key))
    if "supervisor_override" in record and not isinstance(record["supervisor_override"], bool):
        raise LedgerError("INVALID_FIELD: supervisor_override must be a bool")
    if "confirm" in record and not isinstance(record.get("confirm"), bool):
        raise LedgerError(
            "INVALID_CONFIRM: confirm must be a bool (explicit plan flag), got {!r}".format(
                record.get("confirm")
            )
        )
    # -- WP-L2 formal record schema (fail-closed: all five keys, exact types)
    formal_keys_present = [k for k in FORMAL_FIELDS if k in record]
    if formal_keys_present and len(formal_keys_present) != len(FORMAL_FIELDS):
        missing_formal = [k for k in FORMAL_FIELDS if k not in record]
        raise LedgerError(
            "INCOMPLETE_FORMAL_FIELDS: record mixes formal/legacy schema; missing {}".format(
                ", ".join(missing_formal)
            )
        )
    if formal_keys_present:
        gate_id = record.get("gate_id")
        if not (isinstance(gate_id, str) and gate_id):
            raise LedgerError(
                "INVALID_GATE_ID: gate_id must be a non-empty string, got {!r}".format(
                    gate_id
                )
            )
        adm = record.get("admissibility")
        if adm not in ADMISSIBILITY_VALUES:
            raise LedgerError(
                "INVALID_ADMISSIBILITY: {!r} not in {}".format(
                    adm, ", ".join(ADMISSIBILITY_VALUES)
                )
            )
        attempt = record.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise LedgerError(
                "INVALID_ATTEMPT: attempt must be an int >= 1, got {!r}".format(attempt)
            )
        parent = record.get("parent_run_id")
        if parent is not None and not (isinstance(parent, str) and parent):
            raise LedgerError(
                "INVALID_PARENT_RUN_ID: parent_run_id must be null or a non-empty string"
            )
        evidence = record.get("evidence_sha")
        if evidence is not None and not (isinstance(evidence, str) and HASH_RE.match(evidence)):
            raise LedgerError(
                "INVALID_HASH: evidence_sha must be null or a lowercase 64-char sha256"
            )
        if adm == "admissible" and evidence is None:
            raise PlaceholderHashError(
                "MISSING_HASH: admissibility=admissible requires evidence_sha "
                "(sha256 of the gate report)"
            )
    if status == "done":
        # a finished run must never ship placeholder/absent evidence
        if record.get("checkpoint_sha256") is None:
            raise PlaceholderHashError(
                "MISSING_HASH: status=done requires a real checkpoint_sha256"
            )
        if record.get("checkpoint_update") is None:
            raise LedgerError("INCOMPLETE_EVIDENCE: status=done requires checkpoint_update")
        if not record.get("metrics_path"):
            raise LedgerError("INCOMPLETE_EVIDENCE: status=done requires metrics_path")
    return record



def _check_transition(from_status: str, to_status: str) -> None:
    allowed = ALLOWED_TRANSITIONS.get(from_status, frozenset())
    if to_status not in allowed:
        raise IllegalTransition(
            "ILLEGAL_TRANSITION: {} -> {} not allowed (allowed from {}: {})".format(
                from_status,
                to_status,
                from_status,
                ", ".join(sorted(allowed)) or "<terminal>",
            )
        )


# --------------------------------------------------------------------------
# file primitives (flock + tmp/os.replace)
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        dfd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _atomic_rewrite(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via tmp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path)


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


# --------------------------------------------------------------------------
# run_plan.jsonl cross-check helpers (WP-L2 budget caps)
# --------------------------------------------------------------------------


def read_run_plan(plan_path: Any) -> Dict[str, Any]:
    """Parse a ``run_plan.jsonl`` for append-time budget/identity cross-checks.

    Returns::

        {
          "counts":   {(campaign_id, phase): planned_runs},
          "keys":     {(campaign_id, phase, arm, fold, seed, variant)},
          "run_ids":  {run_id, ...},
          "by_run_id": {run_id: line},
          "mode":     "full" | "partial",
          "confirm_run_ids": [run_id, ...],   # lines with explicit confirm=true
          "metric_run_ids":  [run_id, ...],   # B-stage folds3/4 lines (sealed metrics)
          "header":   plan_header dict,
        }

    Plan identity (ORA-2 blocker 2): membership is checked against the final
    ``run_id`` (and the 6-tuple including ``variant`` for diagnostics), not the
    5-tuple alone -- changed-config-same-name must never match a planned run.
    Header lines (``record_type == "plan_header"``) are skipped for membership;
    a missing or corrupt plan raises :class:`LedgerError` -- a bound plan is
    fail-closed, never optional.
    """
    path = Path(plan_path)
    if not path.is_file():
        raise LedgerError("PLAN_MISSING: bound run_plan.jsonl not found: {}".format(path))
    lines = [ln for ln in _read_lines(path) if ln.strip()]
    if not lines:
        raise LedgerError("PLAN_CORRUPT: bound run_plan.jsonl is empty: {}".format(path))
    counts: Dict[tuple, int] = {}
    keys = set()
    by_key: Dict[tuple, str] = {}
    run_ids = set()
    by_run_id: Dict[str, Dict[str, Any]] = {}
    confirm_run_ids: List[str] = []
    metric_run_ids: List[str] = []
    header: Dict[str, Any] = {}
    seen_header = False
    for lineno, line in enumerate(lines, start=1):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(
                "PLAN_CORRUPT: {} line {}: {}".format(path, lineno, exc)
            ) from exc
        if not isinstance(rec, dict):
            raise LedgerError(
                "PLAN_CORRUPT: {} line {} is not a JSON object".format(path, lineno)
            )
        if rec.get("record_type") == "plan_header":
            if seen_header:
                raise LedgerError(
                    "PLAN_CORRUPT: {} has more than one plan_header".format(path)
                )
            seen_header = True
            header = rec
            continue
        for field_name in ("campaign_id", "phase", "arm"):
            if not (isinstance(rec.get(field_name), str) and rec.get(field_name)):
                raise LedgerError(
                    "PLAN_CORRUPT: {} line {} missing/invalid {!r}".format(
                        path, lineno, field_name
                    )
                )
        for field_name in ("fold", "seed"):
            value = rec.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise LedgerError(
                    "PLAN_CORRUPT: {} line {} invalid {!r}".format(
                        path, lineno, field_name
                    )
                )
        run_id = rec.get("run_id")
        if not (isinstance(run_id, str) and run_id):
            raise LedgerError(
                "PLAN_CORRUPT: {} line {} missing run_id".format(path, lineno)
            )
        if run_id in by_run_id:
            raise LedgerError(
                "PLAN_CORRUPT: {} duplicate run_id {}".format(path, run_id)
            )
        campaign_phase = (rec["campaign_id"], rec["phase"])
        counts[campaign_phase] = counts.get(campaign_phase, 0) + 1
        key6 = (
            rec["campaign_id"],
            rec["phase"],
            rec["arm"],
            rec["fold"],
            rec["seed"],
            rec.get("variant"),
        )
        keys.add(key6)
        if key6 in by_key and by_key[key6] != run_id:
            raise LedgerError(
                "PLAN_CORRUPT: {} duplicate (campaign,phase,arm,fold,seed,variant) "
                "key {} under run_ids {} and {}".format(path, key6, by_key[key6], run_id)
            )
        by_key[key6] = run_id
        run_ids.add(run_id)
        by_run_id[run_id] = rec
        if rec.get("confirm") is True:
            confirm_run_ids.append(run_id)
        if rec.get("substage") == "B" and rec.get("fold") in (3, 4):
            # B-stage folds3/4: sealed-baseline metrics, one-shot metric_unseal
            metric_run_ids.append(run_id)
    mode = header.get("campaign_mode", "full")
    if mode not in ("full", "partial"):
        raise LedgerError("PLAN_CORRUPT: campaign_mode {!r} not in full/partial".format(mode))
    return {
        "counts": counts,
        "keys": keys,
        "by_key": by_key,
        "run_ids": run_ids,
        "by_run_id": by_run_id,
        "mode": mode,
        "confirm_run_ids": sorted(confirm_run_ids),
        "metric_run_ids": sorted(metric_run_ids),
        "header": header,
    }


def is_frozen_sha(value: Any) -> bool:
    """A real, bound, non-placeholder 64-hex digest (not zero, not PENDING_*)."""
    return bool(
        isinstance(value, str)
        and HASH_RE.match(value)
        and value != "0" * 64
    )


def sha256_file(path: Any) -> str:
    """Streaming sha256 of a file (WP-L3 plan-pin recomputation)."""
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# pinned-plan verification (WP-L3 / E0 residual #1)
# --------------------------------------------------------------------------


def _validate_pin(pin: Any, where: str) -> Dict[str, str]:
    """Fail-closed validation of a plan pin dict ``{sha256, campaign_mode}``."""
    if not isinstance(pin, dict):
        raise PlanNotFrozenError(
            "PLAN_PIN_INVALID: {} pin must be a dict with sha256/campaign_mode, "
            "got {!r}".format(where, pin)
        )
    sha = pin.get("sha256")
    mode = pin.get("campaign_mode")
    if not (isinstance(sha, str) and HASH_RE.match(sha) and sha != "0" * 64):
        raise PlanNotFrozenError(
            "PLAN_PIN_INVALID: {} pin sha256 must be a non-zero 64-hex digest, "
            "got {!r}".format(where, sha)
        )
    if mode not in ("full", "partial"):
        raise PlanNotFrozenError(
            "PLAN_PIN_INVALID: {} pin campaign_mode must be full/partial, got "
            "{!r}".format(where, mode)
        )
    return {"sha256": sha, "campaign_mode": mode}


def load_campaign_pin(task_state_path: Any = None) -> Dict[str, str]:
    """Read the E0-frozen plan pin from ``task_state.e0_freeze.run_plan_pinned``.

    This is the human-signed source of truth (E0 sign-off; re-pinning requires
    a --force plan rebuild + re-freeze + user re-approval).  Missing/corrupt
    state or a malformed pin always raises :class:`PlanNotFrozenError` --
    fail-closed, never a default fallback to "unpinned".
    """
    path = Path(task_state_path or DEFAULT_TASK_STATE_SUBPATH)
    if not path.is_file():
        raise PlanNotFrozenError(
            "PLAN_PIN_SOURCE_MISSING: {} not found (cannot resolve the frozen "
            "run_plan pin)".format(path)
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanNotFrozenError(
            "PLAN_PIN_SOURCE_CORRUPT: {}: {}".format(path, exc)
        ) from exc
    if not isinstance(state, dict):
        raise PlanNotFrozenError(
            "PLAN_PIN_SOURCE_CORRUPT: {} is not a JSON object".format(path)
        )
    pinned = state.get("e0_freeze", {}).get("run_plan_pinned") if isinstance(
        state.get("e0_freeze"), dict
    ) else None
    if not isinstance(pinned, dict):
        raise PlanNotFrozenError(
            "PLAN_PIN_SOURCE_CORRUPT: {} has no e0_freeze.run_plan_pinned object"
            .format(path)
        )
    return _validate_pin(
        {"sha256": pinned.get("sha256"), "campaign_mode": pinned.get("campaign_mode")},
        str(path),
    )


def verify_plan_pin(plan_path: Any, pin: Any) -> Dict[str, Any]:
    """Recompute the plan's sha256 + header campaign_mode and compare to ``pin``.

    WP-L3 / E0 residual #1: Oracle showed a *tampered copy* of the frozen
    ``run_plan.jsonl`` (confirm flags flipped, or a PD-4 partial plan
    re-headered as ``full``, or any other byte change) still passed every
    append-time check because only ``runplan verify`` hashed the plan while
    ``ledger.append`` never did.  This function closes that path: ANY byte
    difference is a :class:`PlanNotFrozenError` (``PLAN_SHA_MISMATCH``), and
    the header mode is cross-checked independently (``PLAN_MODE_MISMATCH``)
    as defense in depth.  Returns the verified pin on success.
    """
    verified = _validate_pin(pin, "bound")
    path = Path(plan_path)
    if not path.is_file():
        raise PlanNotFrozenError(
            "PLAN_PIN_CHECK_FAILED: bound run_plan.jsonl not found: {}".format(path)
        )
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise PlanNotFrozenError(
            "PLAN_PIN_CHECK_FAILED: cannot hash {}: {}".format(path, exc)
        ) from exc
    if actual != verified["sha256"]:
        raise PlanNotFrozenError(
            "PLAN_SHA_MISMATCH: {} hashes to {} but the frozen pin is {} -- the "
            "plan was modified after E0 freeze (confirm-flag/mode/byte tamper "
            "or an in-band rebuild); restore the frozen file or re-freeze with "
            "--force + human re-signoff".format(path, actual, verified["sha256"])
        )
    # independent header-mode check (defense in depth: a mode flip is also a
    # byte change, but report the semantic mismatch explicitly)
    header: Optional[Dict[str, Any]] = None
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    header = parsed
                break
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanNotFrozenError(
            "PLAN_PIN_CHECK_FAILED: cannot parse the header of {}: {}".format(
                path, exc
            )
        ) from exc
    if not isinstance(header, dict) or header.get("record_type") != "plan_header":
        raise PlanNotFrozenError(
            "PLAN_PIN_CHECK_FAILED: {} does not start with a plan_header line"
            .format(path)
        )
    mode = header.get("campaign_mode")
    if mode != verified["campaign_mode"]:
        raise PlanNotFrozenError(
            "PLAN_MODE_MISMATCH: {} header campaign_mode={!r} but the frozen "
            "pin is {!r} (e.g. a PD-4 partial plan re-headered as full is "
            "refused)".format(path, mode, verified["campaign_mode"])
        )
    return dict(verified)


# --------------------------------------------------------------------------
# one-shot unseal tokens: BATCH semantics (30h plan 7.3 / ORA-2 blocker 3)
# --------------------------------------------------------------------------


def _consume_batch_token(
    path: Any,
    *,
    batch_id: Any,
    allowed_run_ids: Any,
    run_id: Optional[str],
    kind: str,
    missing_code: str,
) -> Dict[str, Any]:
    """Consume a one-shot token FOR A BATCH of pre-registered run_ids.

    Token file schema: ``{kind, batch_id, allowed_run_ids[], count,
    consumed_at, run_id}``.  First use writes ``count=1`` covering the whole
    batch; every run inside ``allowed_run_ids`` is then admissible
    (idempotent, never re-increments); a run outside the registered batch, or
    a *second, different* batch (different run_id set), raises
    :class:`UnsealExceeded`.  Corrupt/missing structure always refuses.
    Written via flock + tmp + ``os.replace``.
    """
    if path is None:
        raise UnsealExceeded(
            "{}: {} token path not configured".format(missing_code, kind)
        )
    if not (isinstance(batch_id, str) and batch_id):
        raise UnsealExceeded(
            "UNSEAL_CORRUPT: batch_id must be a non-empty string, got {!r}".format(batch_id)
        )
    if not isinstance(allowed_run_ids, (list, tuple)) or not allowed_run_ids:
        raise UnsealExceeded(
            "UNSEAL_CORRUPT: allowed_run_ids must be a non-empty list of run_ids"
        )
    normalized = sorted({str(r) for r in allowed_run_ids if str(r)})
    if not normalized or len(normalized) != len(set(str(r) for r in allowed_run_ids)):
        raise UnsealExceeded(
            "UNSEAL_CORRUPT: allowed_run_ids must be unique non-empty strings"
        )
    if run_id is not None and run_id not in normalized:
        # the batch is PRE-REGISTERED: an unregistered run can never open it
        raise UnsealExceeded(
            "UNSEAL_OUTSIDE_BATCH: run {!r} is not in the pre-registered batch "
            "{!r} ({})".format(run_id, batch_id, normalized)
        )

    token_path = Path(path)
    lock = Path(str(token_path) + ".lock")
    with _file_lock(lock):
        payload: Dict[str, Any]
        if token_path.exists():
            try:
                raw = token_path.read_text(encoding="utf-8")
                payload = json.loads(raw)
            except (OSError, json.JSONDecodeError) as exc:
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: {} is unreadable/invalid: {}".format(token_path, exc)
                ) from exc
            if not isinstance(payload, dict):
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: {} must contain a JSON object".format(token_path)
                )
            count = payload.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: count must be an int >= 0, got {!r}".format(count)
                )
            stored_ids = payload.get("allowed_run_ids")
            if not isinstance(stored_ids, list) or not all(
                isinstance(r, str) and r for r in stored_ids
            ):
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: stored allowed_run_ids invalid in {}".format(token_path)
                )
            if count >= 1:
                if sorted(stored_ids) != normalized:
                    # a *different* pre-registered batch after consumption: refuse
                    raise UnsealExceeded(
                        "UNSEAL_BATCH_MISMATCH: {} already consumed for batch {!r} "
                        "({} run_ids); second different batch refused".format(
                            token_path, payload.get("batch_id"), len(stored_ids)
                        )
                    )
                if run_id is not None and run_id in stored_ids:
                    return dict(payload)  # in-batch run (incl. same-run retry): idempotent
                raise UnsealExceeded(
                    "UNSEAL_OUTSIDE_BATCH: run {!r} is not in the consumed batch "
                    "{!r} ({})".format(run_id, payload.get("batch_id"), token_path)
                )
        else:
            payload = {}
        payload["kind"] = kind
        payload["batch_id"] = batch_id
        payload["allowed_run_ids"] = normalized
        payload["count"] = int(payload.get("count", 0) or 0) + 1
        payload["consumed_at"] = _utc_now_iso()
        payload["run_id"] = run_id
        _atomic_rewrite(token_path, json.dumps(payload, sort_keys=True, ensure_ascii=False))
        return dict(payload)


def confirm_unseal(
    unseal_path: Any,
    run_id: Optional[str] = None,
    *,
    batch_id: Optional[str] = None,
    allowed_run_ids: Any = None,
) -> Dict[str, Any]:
    """Consume the one-shot fold3/4 confirm-unseal token for a registered batch.

    ``allowed_run_ids`` is the pre-registered confirm run set (from
    ``run_plan.jsonl`` lines with ``confirm: true``; defaults to the single
    ``run_id`` for ad-hoc use).  See :func:`_consume_batch_token` for the
    batch semantics.
    """
    if allowed_run_ids is None:
        if run_id is None:
            raise UnsealExceeded(
                "UNSEAL_CORRUPT: allowed_run_ids or run_id required to register a batch"
            )
        allowed_run_ids = [run_id]
    if batch_id is None:
        batch_id = "confirm-batch"
    return _consume_batch_token(
        unseal_path,
        batch_id=batch_id,
        allowed_run_ids=allowed_run_ids,
        run_id=run_id,
        kind="confirm",
        missing_code="UNSEAL_NOT_CONFIGURED",
    )


def metric_unseal(
    metric_unseal_path: Any,
    run_id: Optional[str] = None,
    *,
    batch_id: Optional[str] = None,
    allowed_run_ids: Any = None,
) -> Dict[str, Any]:
    """One-shot reveal of B-stage folds3/4 sealed-baseline metrics (30h §4.2).

    Same batch semantics as :func:`confirm_unseal` but a *separate* token
    (``state/gates/metric_unseal.json``): consuming it never consumes the
    confirm token, and vice versa.
    """
    if allowed_run_ids is None:
        if run_id is None:
            raise UnsealExceeded(
                "UNSEAL_CORRUPT: allowed_run_ids or run_id required to register a batch"
            )
        allowed_run_ids = [run_id]
    if batch_id is None:
        batch_id = "metric-batch"
    return _consume_batch_token(
        metric_unseal_path,
        batch_id=batch_id,
        allowed_run_ids=allowed_run_ids,
        run_id=run_id,
        kind="metric",
        missing_code="METRIC_UNSEAL_NOT_CONFIGURED",
    )


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppendOutcome:
    """Result of :meth:`RunLedger.append` / :meth:`RunLedger.mark_blocked`."""

    action: str  # "created" | "noop" | "transitioned" | "blocked"
    entry: Dict[str, Any]


#: sentinel: "bind the E0-frozen campaign pin" (vs ``None`` = explicitly
#: unpinned, ONLY legitimate for synthetic test-fixture plans).
CAMPAIGN_PIN = object()


class RunLedger:
    """Append-only run ledger with run_id idempotency and audit events."""

    def __init__(
        self,
        path: Any,
        *,
        run_id_scheme: str = "auto",
        plan_path: Optional[Any] = None,
        caps: Optional[Dict[str, int]] = None,
        unseal_path: Optional[Any] = None,
        metric_unseal_path: Optional[Any] = None,
        plan_pin: Any = CAMPAIGN_PIN,
    ) -> None:
        if run_id_scheme not in ("auto", "legacy", "hashed"):
            raise LedgerError(
                "INVALID_SCHEME: run_id_scheme must be auto/legacy/hashed, got {!r}".format(
                    run_id_scheme
                )
            )
        self.path = Path(path)
        self.events_path = self.path.parent / EVENTS_FILENAME
        self.lock_path = Path(str(self.path) + ".lock")
        self.run_id_scheme = run_id_scheme
        #: bound run_plan.jsonl (E0 planning artifact); None = no plan cross-check
        self.plan_path = Path(plan_path) if plan_path is not None else None
        #: WP-L3 plan pin: {sha256, campaign_mode} verified against the bound
        #: plan file on EVERY append / unseal.  Default = the E0-frozen
        #: campaign pin (task_state.e0_freeze.run_plan_pinned; compiled as
        #: DEFAULT_PLAN_PIN).  ``None`` disables the check and is ONLY
        #: legitimate for synthetic test-fixture plans -- binding a plan
        #: without overriding the pin means "this IS the frozen campaign
        #: plan" and it must match (fail-closed, WP-L3 residual #1).
        if plan_pin is CAMPAIGN_PIN:
            self.plan_pin: Optional[Dict[str, str]] = (
                dict(DEFAULT_PLAN_PIN) if self.plan_path is not None else None
            )
        elif plan_pin is None:
            self.plan_pin = None
        else:
            self.plan_pin = _validate_pin(plan_pin, "plan_pin")
            if self.plan_path is None:
                raise LedgerError(
                    "PLAN_PIN_WITHOUT_PLAN: an explicit plan_pin was configured "
                    "but no plan_path is bound (pinning a non-existent plan is "
                    "incoherent)"
                )
        self.caps: Dict[str, int] = dict(DEFAULT_CAPS)
        if caps is not None:
            self.caps.update(caps)
        #: one-shot fold3/4 confirm BATCH unseal token (None = confirm refuses)
        self.unseal_path = Path(unseal_path) if unseal_path is not None else None
        #: separate one-shot B-stage folds3/4 metric-reveal token
        self.metric_unseal_path = (
            Path(metric_unseal_path) if metric_unseal_path is not None else None
        )

    # -- lifecycle ---------------------------------------------------------

    def init(self) -> Dict[str, Any]:
        """Create empty ledger/events files if missing (never truncates)."""
        with _file_lock(self.lock_path):
            created = []
            for target in (self.path, self.events_path):
                if not target.exists():
                    _atomic_rewrite(target, "")
                    created.append(str(target))
            return {
                "ledger": str(self.path),
                "events": str(self.events_path),
                "created": created,
                "lines": len(_read_lines(self.path)),
            }

    # -- reads -------------------------------------------------------------

    def _latest_locked(self) -> Dict[str, Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for lineno, line in enumerate(_read_lines(self.path), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerError(
                    "CORRUPT_LEDGER: {} line {}: {}".format(self.path, lineno, exc)
                ) from exc
            run_id = rec.get("run_id")
            if not run_id:
                raise LedgerError("CORRUPT_LEDGER: {} line {} has no run_id".format(self.path, lineno))
            latest[run_id] = rec
        return latest

    def read_all(self) -> Dict[str, Dict[str, Any]]:
        with _file_lock(self.lock_path):
            return self._latest_locked()

    def status(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self.read_all().get(run_id)

    def entries(
        self,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        arm: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        out = []
        for entry in self.read_all().values():
            if status is not None and entry.get("status") != status:
                continue
            if phase is not None and entry.get("phase") != phase:
                continue
            if arm is not None and entry.get("arm") != arm:
                continue
            if campaign_id is not None and entry.get("campaign_id") != campaign_id:
                continue
            out.append(entry)
        out.sort(key=lambda e: e.get("run_id", ""))
        return out

    # alias kept for readability at call sites
    def list(self, **filters: Any) -> List[Dict[str, Any]]:  # noqa: A003
        return self.entries(**filters)

    # -- writes ------------------------------------------------------------

    def _emit_locked(
        self, entry: Dict[str, Any], event_kind: str, from_status: Optional[str]
    ) -> None:
        """Append one ledger line + one audit event (caller holds the lock)."""
        ledger_lines = [ln for ln in _read_lines(self.path) if ln.strip()]
        event_lines = [ln for ln in _read_lines(self.events_path) if ln.strip()]
        ledger_lines.append(json.dumps(entry, sort_keys=True, ensure_ascii=False))
        event = {
            "ts": _utc_now_iso(),
            "event": event_kind,
            "run_id": entry.get("run_id"),
            "from_status": from_status,
            "to_status": entry.get("status"),
            "retry_count": entry.get("retry_count", 0),
        }
        event_lines.append(json.dumps(event, sort_keys=True, ensure_ascii=False))
        # tmp + os.replace for both files, while still holding the flock: a
        # reader sees either the pre- or post-mutation file, never a torn one.
        _atomic_rewrite(self.path, "".join(ln + "\n" for ln in ledger_lines))
        _atomic_rewrite(self.events_path, "".join(ln + "\n" for ln in event_lines))

    def append(self, record: Dict[str, Any]) -> AppendOutcome:
        """Idempotent submit / status transition for one run.

        * first record for a run_id must have status ``planned``;
        * identical resubmission -> ``noop`` (no new line, no event);
        * changed identity under same run_id, or same-status content drift ->
          :class:`RunIdConflict`;
        * status change -> state-machine + retry-cap validation, then append;
        * every transition into ``running`` is a *launch attempt*: it must pass
          the campaign budget cap cross-check against the bound
          ``run_plan.jsonl`` (:class:`BudgetCapExceeded`) and, for fold3/4
          confirm runs, consume the one-shot unseal token
          (:class:`UnsealExceeded`).
        * **WP-L3**: when a plan is bound and pinned, the plan file's sha256 +
          header ``campaign_mode`` are re-verified against the pin BEFORE any
          ledger mutation (Oracle residual: tampered plan copies used to
          append).  Any mismatch -> :class:`PlanNotFrozenError`, nothing
          written.
        * **WP-L3**: a ``done`` transition for a B folds3/4 plan line consumes
          the one-shot ``metric_unseal`` token (batch = the plan's B
          folds3/4 run set) and records the consumption on the run record
          (``metric_unseal`` field).  Missing token path, a second/different
          batch or a non-batch run -> :class:`UnsealExceeded`, nothing
          written.
        * ``blocked_with_reason`` annotations (monitor) are cleared by any
          successful transition.
        """
        rec = dict(record)
        validate(rec)
        run_id = compute_run_id(rec, self.run_id_scheme)
        rec["run_id"] = run_id
        now = _utc_now_iso()
        with _file_lock(self.lock_path):
            # WP-L3 fail-closed plan pin: verify BEFORE any ledger mutation.
            if self.plan_path is not None and self.plan_pin is not None:
                verify_plan_pin(self.plan_path, self.plan_pin)
            all_latest = self._latest_locked()
            latest = all_latest.get(run_id)
            if latest is None:
                if rec["status"] not in INITIAL_STATUSES:
                    raise IllegalTransition(
                        "ILLEGAL_TRANSITION: first record for {} must be one of {}, got {!r}".format(
                            run_id, ", ".join(INITIAL_STATUSES), rec["status"]
                        )
                    )
                entry = dict(rec)
                entry["run_id"] = run_id
                entry["created_at"] = now
                entry["updated_at"] = now
                entry["retry_count"] = 0
                entry["supervisor_override"] = bool(rec.get("supervisor_override", False))
                self._emit_locked(entry, "append", None)
                return AppendOutcome("created", entry)

            if _identity(rec) != _identity(latest):
                raise RunIdConflict(
                    "RUN_ID_CONFLICT: {} already registered with a different identity; "
                    "identity fields {} may never change under one run_id".format(
                        run_id, ", ".join(IDENTITY_FIELDS)
                    )
                )
            if _content(rec) == _content(latest):
                # same terminal content -> idempotent no-op (resubmission safe)
                return AppendOutcome("noop", dict(latest))

            from_status = str(latest.get("status"))
            to_status = str(rec.get("status"))
            if from_status == to_status:
                raise RunIdConflict(
                    "RUN_ID_CONFLICT: {} resubmitted with the same status {!r} but "
                    "different content".format(run_id, to_status)
                )
            _check_transition(from_status, to_status)

            retry_count = int(latest.get("retry_count", 0) or 0)
            if to_status == "running" and from_status in RETRY_SOURCES:
                retry_count += 1
                if retry_count > MAX_RETRIES and not bool(rec.get("supervisor_override", False)):
                    raise RetryCapExceeded(
                        "RETRY_CAP_EXCEEDED: {} attempt {} exceeds the retry budget of {}; "
                        "a further retry requires supervisor_override=true".format(
                            run_id, retry_count, MAX_RETRIES
                        )
                    )

            if to_status == "running":
                # WP-L2/ORA-2: every launch attempt is plan-bound (exact run_id
                # membership + frozen identity shas), budget-checked (all
                # attempts incl. retries), partial-plan-forced exploratory, and
                # confirm runs burn the one-shot BATCH unseal token.
                previously_launched = from_status in LAUNCHED_STATES
                plan = (
                    read_run_plan(self.plan_path)
                    if self.plan_path is not None
                    else None
                )
                plan_line: Optional[Dict[str, Any]] = None
                if plan is not None:
                    plan_line = plan["by_run_id"].get(run_id)
                    if plan["mode"] == "partial":
                        adm = rec.get("admissibility")
                        if adm != "not_admissible_exploratory":
                            raise ExploratoryOnlyError(
                                "NOT_ADMISSIBLE_EXPLORATORY: partial (PD-4) plan "
                                "{!r} forbids admissibility={!r}; records must be "
                                "not_admissible_exploratory".format(
                                    str(self.plan_path), adm
                                )
                            )
                self._budget_check_locked(
                    rec, run_id, all_latest, retry_count, previously_launched, plan
                )
                effective_confirm = is_confirm_run(rec)
                if plan_line is not None:
                    plan_confirm = plan_line.get("confirm") is True
                    if "confirm" in rec and bool(rec["confirm"]) != plan_confirm:
                        raise LedgerError(
                            "CONFIRM_FLAG_MISMATCH: record confirm={!r} but plan "
                            "line {} says confirm={!r} (plan is authoritative)".format(
                                rec.get("confirm"), run_id, plan_confirm
                            )
                        )
                    effective_confirm = plan_confirm
                if effective_confirm:
                    if plan is not None:
                        allowed = plan["confirm_run_ids"]
                        batch_id = "plan-confirm-batch"
                    else:
                        allowed = [run_id]
                        batch_id = "solo-confirm-{}".format(run_id)
                    confirm_unseal(
                        self.unseal_path,
                        run_id,
                        batch_id=batch_id,
                        allowed_run_ids=allowed,
                    )

            metric_token: Optional[Dict[str, Any]] = None
            if to_status == "done":
                # PD-4: partial-campaign finalizes may never claim these
                if self.plan_path is not None:
                    plan = read_run_plan(self.plan_path)
                    if plan["mode"] == "partial":
                        for field_name, value in rec.items():
                            if not isinstance(value, str):
                                continue
                            lowered = value.lower()
                            for keyword in PARTIAL_FORBIDDEN_FINALIZE_KEYWORDS:
                                if keyword in lowered:
                                    raise ExploratoryOnlyError(
                                        "PARTIAL_FINALIZE_FORBIDDEN: keyword {!r} in "
                                        "{} of a partial-plan finalize (PD-4: no "
                                        "scale_controlled/V5.1b from 21-run "
                                        "exploratory sets)".format(keyword, field_name)
                                    )
                    # WP-L3 residual #2: B folds3/4 sealed-baseline metrics are
                    # revealed exactly once per campaign via the one-shot
                    # metric_unseal token -- finalize CONSUMES it (batch = the
                    # plan's B folds3/4 run set) and records the consumption.
                    plan_line_done = plan["by_run_id"].get(run_id)
                    if (
                        plan_line_done is not None
                        and plan_line_done.get("substage") == "B"
                        and plan_line_done.get("fold") in (3, 4)
                    ):
                        if self.metric_unseal_path is None:
                            raise UnsealExceeded(
                                "METRIC_UNSEAL_NOT_CONFIGURED: B folds3/4 run {} "
                                "cannot finalize without the one-shot metric "
                                "token (expected {})".format(
                                    run_id, DEFAULT_METRIC_UNSEAL_SUBPATH
                                )
                            )
                        metric_token = metric_unseal(
                            self.metric_unseal_path,
                            run_id,
                            batch_id="plan-metric-batch",
                            allowed_run_ids=plan["metric_run_ids"],
                        )

            entry = dict(latest)
            entry.update(rec)
            entry["run_id"] = run_id
            entry["created_at"] = latest.get("created_at", now)
            entry["updated_at"] = now
            entry["retry_count"] = retry_count
            entry["supervisor_override"] = bool(rec.get("supervisor_override", False))
            if metric_token is not None:
                # record the consumption on the run record (auditable reveal)
                entry["metric_unseal"] = {
                    "kind": metric_token.get("kind"),
                    "batch_id": metric_token.get("batch_id"),
                    "count": metric_token.get("count"),
                    "consumed_at": metric_token.get("consumed_at"),
                }
            # a successful transition supersedes any monitor block annotation
            for _block_key in ("blocked_with_reason", "blocked_source", "blocked_at"):
                entry.pop(_block_key, None)
            self._emit_locked(entry, "transition", from_status)
            return AppendOutcome("transitioned", entry)

    def _budget_check_locked(
        self,
        rec: Dict[str, Any],
        run_id: str,
        all_latest: Dict[str, Dict[str, Any]],
        retry_count: int,
        previously_launched: bool,
        plan: Optional[Dict[str, Any]],
    ) -> None:
        """Fail-closed launch budget/identity check (caller holds the lock).

        With a bound plan (ORA-2 blockers 1+2):
          * the launched ``run_id`` must EQUAL a planned run_id
            (:class:`BudgetCapExceeded` ``NOT_IN_PLAN`` otherwise; a matching
            6-tuple with a different run_id is ``PlanNotFrozenError``
            ``RUN_ID_MISMATCH`` -- the plan was built with other shas);
          * the plan line's five identity shas must be REAL (no PENDING/zero)
            at launch stage, else :class:`PlanNotFrozenError`
            ``PLAN_NOT_FROZEN``;
          * the plan count for the phase is the effective cap.
        Without a plan the phase ``DEFAULT_CAPS`` (R3=66 / V5.2=33) applies.
        """
        campaign = rec.get("campaign_id")
        phase = rec.get("phase")
        cap = self.caps.get(phase)
        plan_count: Optional[int] = None
        if plan is not None:
            plan_count = plan["counts"].get((campaign, phase))
            if plan_count is None:
                if cap is not None:
                    raise BudgetCapExceeded(
                        "NOT_PLANNED: phase {!r} carries cap {} but has no planned "
                        "runs in {}".format(phase, cap, self.plan_path)
                    )
            else:
                line = plan["by_run_id"].get(run_id)
                if line is None:
                    key6 = (
                        campaign,
                        phase,
                        rec.get("arm"),
                        rec.get("fold"),
                        rec.get("seed"),
                        rec.get("variant"),
                    )
                    planned_for_key = plan["by_key"].get(key6)
                    if planned_for_key is not None:
                        # same planned run, different run_id: the plan line's
                        # shas decide WHICH refusal this is (blocker 1)
                        planned_line = plan["by_run_id"][planned_for_key]
                        for sha_field in REQUIRED_HASH_FIELDS:
                            if not is_frozen_sha(planned_line.get(sha_field)):
                                raise PlanNotFrozenError(
                                    "PLAN_NOT_FROZEN: planned run {} still carries "
                                    "{}={!r} (stage e0 pending/zero); bind all five "
                                    "identity shas (stage g0/launch) and rebuild "
                                    "the plan before launching".format(
                                        planned_for_key,
                                        sha_field,
                                        planned_line.get(sha_field),
                                    )
                                )
                        raise PlanNotFrozenError(
                            "RUN_ID_MISMATCH: planned run_id {} for (campaign={}, "
                            "phase={}, arm={}, fold={}, seed={}, variant={!r}) but "
                            "launched {} -- rebuild the plan with the frozen shas "
                            "or fix the record identity".format(
                                planned_for_key, *key6, run_id
                            )
                        )
                    raise BudgetCapExceeded(
                        "NOT_IN_PLAN: run_id {} (campaign={}, phase={}, arm={}, "
                        "fold={}, seed={}, variant={!r}) is not in {}".format(
                            run_id, *key6, self.plan_path
                        )
                    )
                for sha_field in REQUIRED_HASH_FIELDS:
                    if not is_frozen_sha(line.get(sha_field)):
                        raise PlanNotFrozenError(
                            "PLAN_NOT_FROZEN: planned run {} still carries "
                            "{}={!r}; bind all five identity shas (stage "
                            "g0/launch) before launching".format(
                                run_id, sha_field, line.get(sha_field)
                            )
                        )
                cap = plan_count if cap is None else min(cap, plan_count)
        if cap is None:
            return  # phase has neither a cap nor planned runs: not budget-governed

        attempts = 0
        for rid, entry in all_latest.items():
            if rid == run_id:
                continue
            if entry.get("campaign_id") != campaign or entry.get("phase") != phase:
                continue
            if entry.get("status") not in LAUNCHED_STATES:
                continue
            attempts += 1 + int(entry.get("retry_count", 0) or 0)
        if previously_launched:
            this_run_attempts = 1 + retry_count  # retry_count already incremented
        else:
            this_run_attempts = 1
        candidate = attempts + this_run_attempts
        if candidate > cap:
            raise BudgetCapExceeded(
                "BUDGET_CAP_EXCEEDED: campaign={} phase={} attempt {} of run {} would "
                "exceed the cap of {} (all launch attempts incl. retries are counted; "
                "cap source: {})".format(
                    campaign,
                    phase,
                    this_run_attempts,
                    run_id,
                    cap,
                    "run_plan.jsonl"
                    if plan is not None and plan_count is not None
                    else "run_caps",
                )
            )

    def confirm_unseal(
        self,
        run_id: Optional[str] = None,
        *,
        batch_id: Optional[str] = None,
        allowed_run_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Consume the one-shot fold3/4 confirm-unseal token for a registered batch.

        When the ledger has a bound plan and no explicit batch is given, the
        batch is the plan's ``confirm: true`` run set (all C-stage + QT-confirm
        run_ids); otherwise the batch defaults to the single ``run_id``.
        The bound plan is pin-verified first (WP-L3).
        """
        if self.unseal_path is None:
            raise UnsealExceeded(
                "UNSEAL_NOT_CONFIGURED: ledger has no unseal_path (expected {})".format(
                    DEFAULT_UNSEAL_SUBPATH
                )
            )
        if self.plan_path is not None and self.plan_pin is not None:
            verify_plan_pin(self.plan_path, self.plan_pin)
        if allowed_run_ids is None and self.plan_path is not None:
            plan = read_run_plan(self.plan_path)
            allowed_run_ids = list(plan["confirm_run_ids"])
            batch_id = batch_id or "plan-confirm-batch"
            if not allowed_run_ids:
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: bound plan {} registers no confirm:true runs"
                    .format(self.plan_path)
                )
        if batch_id is None:
            if run_id is None:
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: run_id required to register a solo batch"
                )
            batch_id = "solo-confirm-{}".format(run_id)
        return confirm_unseal(
            self.unseal_path,
            run_id,
            batch_id=batch_id,
            allowed_run_ids=allowed_run_ids,
        )

    def metric_unseal(
        self,
        run_id: Optional[str] = None,
        *,
        batch_id: Optional[str] = None,
        allowed_run_ids: Any = None,
    ) -> Dict[str, Any]:
        """Consume the separate one-shot B-stage folds3/4 metric-reveal token.

        With a bound plan and no explicit batch, the batch is the plan's
        B folds3/4 run set (``metric_run_ids``) -- the same batch the
        ``done``-transition wiring consumes (WP-L3 residual #2).  The bound
        plan is pin-verified first.
        """
        if getattr(self, "metric_unseal_path", None) is None:
            raise UnsealExceeded(
                "METRIC_UNSEAL_NOT_CONFIGURED: ledger has no metric_unseal_path "
                "(expected {})".format(DEFAULT_METRIC_UNSEAL_SUBPATH)
            )
        if self.plan_path is not None and self.plan_pin is not None:
            verify_plan_pin(self.plan_path, self.plan_pin)
        if allowed_run_ids is None and self.plan_path is not None:
            plan = read_run_plan(self.plan_path)
            allowed_run_ids = list(plan["metric_run_ids"])
            batch_id = batch_id or "plan-metric-batch"
            if not allowed_run_ids:
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: bound plan {} registers no B folds3/4 runs"
                    .format(self.plan_path)
                )
        if batch_id is None:
            batch_id = "metric-batch"
        return metric_unseal(
            self.metric_unseal_path,
            run_id,
            batch_id=batch_id,
            allowed_run_ids=allowed_run_ids,
        )

    def mark_blocked(
        self,
        run_id: str,
        reason: Any,
        *,
        source: str = "supervisor.monitor",
    ) -> AppendOutcome:
        """Annotate a run record with ``blocked_with_reason`` (no status change).

        WP-L3 residual #3: the supervisor monitor must record a blocking
        condition (typically a :class:`PlanNotFrozenError`) on the affected
        run record instead of crashing.  This is an *annotation* append --
        the status machine is untouched, the reason is auditable in
        ``events.jsonl`` (event ``blocked``), and any subsequent successful
        transition clears the annotation.  Idempotent: re-marking the same
        reason is a no-op.  Unknown run -> :class:`RunNotFound`.
        """
        if not (isinstance(reason, str) and reason.strip()):
            raise LedgerError(
                "INVALID_BLOCK_REASON: blocked_with_reason must be a non-empty "
                "string, got {!r}".format(reason)
            )
        with _file_lock(self.lock_path):
            latest = self._latest_locked().get(run_id)
            if latest is None:
                raise RunNotFound(
                    "RUN_NOT_FOUND: cannot mark blocked; {} is not in {}".format(
                        run_id, self.path
                    )
                )
            if latest.get("blocked_with_reason") == reason:
                return AppendOutcome("noop", dict(latest))
            updated = dict(latest)
            updated["blocked_with_reason"] = reason
            updated["blocked_source"] = source
            updated["blocked_at"] = _utc_now_iso()
            updated["updated_at"] = updated["blocked_at"]
            self._emit_locked(updated, "blocked", latest.get("status"))
            return AppendOutcome("blocked", updated)

    def reconcile(
        self, pid_alive_fn: Optional[Callable[[Any], bool]] = None
    ) -> Dict[str, Any]:
        """Mark ``running`` records with a dead holder pid as ``interrupted``.

        Never deletes records.  Records without a pid are reported as
        ``unknown`` and left untouched (liveness cannot be established).
        """
        alive = pid_alive_fn or pid_alive
        summary: Dict[str, Any] = {"interrupted": [], "alive": [], "unknown": []}
        with _file_lock(self.lock_path):
            for run_id, entry in sorted(self._latest_locked().items()):
                if entry.get("status") != "running":
                    continue
                holder_pid = entry.get("pid")
                if holder_pid is None:
                    summary["unknown"].append(run_id)
                    continue
                if alive(holder_pid):
                    summary["alive"].append(run_id)
                    continue
                updated = dict(entry)
                updated["status"] = "interrupted"
                updated["interrupted_reason"] = "dead_pid:{}".format(holder_pid)
                updated["updated_at"] = _utc_now_iso()
                updated["supervisor_override"] = bool(entry.get("supervisor_override", False))
                self._emit_locked(updated, "reconcile_interrupt", "running")
                summary["interrupted"].append(run_id)
        return summary

    # convenience wrappers -------------------------------------------------

    def validate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return validate(record)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ledger", description="PES2TS 30h append-only run ledger"
    )
    parser.add_argument(
        "--ledger",
        default=DEFAULT_LEDGER_PATH,
        help="path to run_records.jsonl (events.jsonl is written alongside)",
    )
    parser.add_argument(
        "--plan",
        default=None,
        help="bind a run_plan.jsonl for append-time budget cross-checks "
        "(R3=66 / V5.2=33, plan membership enforced)",
    )
    parser.add_argument(
        "--plan-pin",
        choices=("auto", "none"),
        default="auto",
        help="WP-L3 plan pin: 'auto' (default) verifies the bound plan's "
        "sha256+campaign_mode against the E0-frozen pin from task_state "
        "(e0_freeze.run_plan_pinned); 'none' disables the check and is ONLY "
        "legitimate for synthetic fixture plans",
    )
    parser.add_argument(
        "--plan-pin-file",
        default=None,
        help="explicit pin JSON ({sha256, campaign_mode}) instead of the "
        "task_state-derived campaign pin",
    )
    parser.add_argument(
        "--unseal",
        default=None,
        help="path of the one-shot fold3/4 confirm BATCH unseal token "
        "(default rule: state/gates/unseal.json under the campaign root)",
    )
    parser.add_argument(
        "--metric-unseal",
        default=None,
        help="path of the separate one-shot B-stage folds3/4 metric-reveal token "
        "(default rule: state/gates/metric_unseal.json)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the ledger and events files if missing")

    ap = sub.add_parser("append", help="idempotent record submission / transition")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", help="path to a run_record JSON file")
    group.add_argument("--record-json", help="inline run_record JSON")

    sp = sub.add_parser("status", help="show the latest entry for a run_id")
    sp.add_argument("--run-id", required=True)

    lp = sub.add_parser("list", help="list current run entries")
    lp.add_argument("--status")
    lp.add_argument("--phase")
    lp.add_argument("--arm")
    lp.add_argument("--campaign-id")

    sub.add_parser("reconcile", help="mark dead-pid running records as interrupted")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    plan_pin: Any
    if args.plan is None:
        plan_pin = None  # no plan bound -> no pin (RunLedger allows this)
    elif args.plan_pin == "none":
        plan_pin = None  # explicit fixture opt-out
    elif args.plan_pin_file is not None:
        try:
            plan_pin = json.loads(
                Path(args.plan_pin_file).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(
                "error: PLAN_PIN_SOURCE_CORRUPT: {}: {}\n".format(
                    args.plan_pin_file, exc
                )
            )
            return 2
    else:
        # E0-frozen pin from task_state (fail-closed; compiled constant is the
        # same value and is asserted equal in the WP-L3 tests)
        try:
            plan_pin = load_campaign_pin()
        except PlanNotFrozenError as exc:
            sys.stderr.write("error: {}\n".format(exc))
            return 2
    ledger = RunLedger(
        args.ledger,
        plan_path=args.plan,
        plan_pin=plan_pin,
        unseal_path=args.unseal,
        metric_unseal_path=args.metric_unseal,
    )
    try:
        if args.cmd == "init":
            print(json.dumps(ledger.init(), sort_keys=True))
            return 0
        if args.cmd == "append":
            if args.record:
                raw = Path(args.record).read_text(encoding="utf-8")
            else:
                raw = args.record_json
            record = json.loads(raw)
            outcome = ledger.append(record)
            print(
                json.dumps(
                    {"action": outcome.action, "entry": outcome.entry}, sort_keys=True
                )
            )
            return 0
        if args.cmd == "status":
            entry = ledger.status(args.run_id)
            if entry is None:
                sys.stderr.write("error: RUN_NOT_FOUND: {}\n".format(args.run_id))
                return 1
            print(json.dumps(entry, sort_keys=True))
            return 0
        if args.cmd == "list":
            entries = ledger.entries(
                status=args.status,
                phase=args.phase,
                arm=args.arm,
                campaign_id=args.campaign_id,
            )
            print(json.dumps(entries, sort_keys=True))
            return 0
        if args.cmd == "reconcile":
            print(json.dumps(ledger.reconcile(), sort_keys=True))
            return 0
        return 2
    except LedgerError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    except (json.JSONDecodeError, OSError) as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
