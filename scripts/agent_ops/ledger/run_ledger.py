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
  listed in the plan, or a phase absent from a bound plan, raises
  :class:`BudgetCapExceeded`.  Bound-plan problems (missing/corrupt) raise,
  never warn.
* ``confirm_unseal``: launching a fold3/4 **confirm** run consumes the single
  token in ``state/gates/unseal.json`` (count<=1); a second use, a corrupt
  file or an unconfigured unseal path raises :class:`UnsealExceeded`.
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
    "DEFAULT_CAPS",
    "IllegalTransition",
    "LedgerError",
    "MISSING_PLACEHOLDER",
    "PLACEHOLDER",
    "RetryCapExceeded",
    "RunIdConflict",
    "RunLedger",
    "RunNotFound",
    "UnsealExceeded",
    "compute_run_id",
    "confirm_unseal",
    "is_confirm_run",
    "is_formal_record",
    "main",
    "pid_alive",
    "validate",
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

# -- WP-L2 formal-record schema ------------------------------------------
FORMAL_FIELDS = ("gate_id", "admissibility", "evidence_sha", "attempt", "parent_run_id")
ADMISSIBILITY_VALUES = ("admissible", "not_admissible_until_gate_pass")

#: statuses that mean "this run has been launched at least once"
LAUNCHED_STATES = frozenset({"running", "done", "failed", "killed", "interrupted"})

#: campaign budget caps (task_state budget.run_caps; counts ALL launch attempts)
DEFAULT_CAPS: Dict[str, int] = {"R3": 66, "V5.2": 33}

#: default unseal token location (relative to the campaign root)
DEFAULT_UNSEAL_SUBPATH = "docs/agent_runs/state/gates/unseal.json"

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


class UnsealExceeded(LedgerError):
    """fold3/4 confirm unseal token missing/corrupt/already consumed."""


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
    """fold3/4 confirm runs consume the one-shot unseal token (30h plan 7.3)."""
    fold = record.get("fold")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 3:
        return False
    arm = str(record.get("arm") or "")
    variant = str(record.get("variant") or "")
    lowered = "{} {}".format(arm, variant).lower()
    return arm == "C" or "confirm" in lowered


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
    """Parse a ``run_plan.jsonl`` for append-time budget cross-checks.

    Returns ``{"counts": {(campaign_id, phase): planned_runs},
    "keys": {(campaign_id, phase, arm, fold, seed)}}``.  Header lines
    (``record_type == "plan_header"``) are skipped.  A missing or corrupt plan
    raises :class:`LedgerError` -- a bound plan is fail-closed, never optional.
    """
    path = Path(plan_path)
    if not path.is_file():
        raise LedgerError("PLAN_MISSING: bound run_plan.jsonl not found: {}".format(path))
    lines = [ln for ln in _read_lines(path) if ln.strip()]
    if not lines:
        raise LedgerError("PLAN_CORRUPT: bound run_plan.jsonl is empty: {}".format(path))
    counts: Dict[tuple, int] = {}
    keys = set()
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
        campaign_phase = (rec["campaign_id"], rec["phase"])
        counts[campaign_phase] = counts.get(campaign_phase, 0) + 1
        keys.add(
            (rec["campaign_id"], rec["phase"], rec["arm"], rec["fold"], rec["seed"])
        )
    return {"counts": counts, "keys": keys}


# --------------------------------------------------------------------------
# one-shot fold3/4 unseal token (30h plan section 7.3 / task_state)
# --------------------------------------------------------------------------


def confirm_unseal(unseal_path: Any, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Consume the single fold3/4 confirm-unseal token.

    ``state/gates/unseal.json`` must satisfy ``count <= 0`` to be consumable:
    the first use writes ``count=1``; any *second* run (or a tampered file with
    ``count >= 1`` naming another run) raises :class:`UnsealExceeded`.  A
    re-entry of the *same* run_id (retry of an already-opened confirm run) is
    an idempotent no-op -- it never increments the counter.  Corrupt/unreadable
    content always refuses.  Written via flock + tmp + ``os.replace``.
    """
    if unseal_path is None:
        raise UnsealExceeded(
            "UNSEAL_NOT_CONFIGURED: fold3/4 confirm runs require an unseal path "
            "({} default)".format(DEFAULT_UNSEAL_SUBPATH)
        )
    path = Path(unseal_path)
    lock = Path(str(path) + ".lock")
    with _file_lock(lock):
        payload: Dict[str, Any]
        if path.exists():
            try:
                raw = path.read_text(encoding="utf-8")
                payload = json.loads(raw)
            except (OSError, json.JSONDecodeError) as exc:
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: {} is unreadable/invalid: {}".format(path, exc)
                ) from exc
            if not isinstance(payload, dict):
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: {} must contain a JSON object".format(path)
                )
            count = payload.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise UnsealExceeded(
                    "UNSEAL_CORRUPT: count must be an int >= 0, got {!r}".format(count)
                )
            if count >= 1:
                owner = payload.get("run_id")
                if run_id is not None and owner == run_id:
                    # same run retry: token already consumed by us, no-op
                    return dict(payload)
                raise UnsealExceeded(
                    "UNSEAL_ALREADY_USED: {} already consumed (count={}, run_id={!r}); "
                    "the fold3/4 confirm unseal is one-shot".format(
                        path, count, owner
                    )
                )
        else:
            payload = {}
        payload["count"] = int(payload.get("count", 0) or 0) + 1
        payload["used_at"] = _utc_now_iso()
        payload["run_id"] = run_id
        _atomic_rewrite(path, json.dumps(payload, sort_keys=True, ensure_ascii=False))
        return dict(payload)


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppendOutcome:
    """Result of :meth:`RunLedger.append`."""

    action: str  # "created" | "noop" | "transitioned"
    entry: Dict[str, Any]


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
        self.caps: Dict[str, int] = dict(DEFAULT_CAPS)
        if caps is not None:
            self.caps.update(caps)
        #: one-shot fold3/4 unseal token path (None = confirm launches refuse)
        self.unseal_path = Path(unseal_path) if unseal_path is not None else None

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
        """
        rec = dict(record)
        validate(rec)
        run_id = compute_run_id(rec, self.run_id_scheme)
        rec["run_id"] = run_id
        now = _utc_now_iso()
        with _file_lock(self.lock_path):
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
                # WP-L2: every launch attempt is budget-checked (all attempts,
                # incl. retries) and confirm runs burn the one-shot unseal.
                previously_launched = from_status in LAUNCHED_STATES
                self._budget_check_locked(
                    rec, run_id, all_latest, retry_count, previously_launched
                )
                if is_confirm_run(rec):
                    confirm_unseal(self.unseal_path, run_id=run_id)

            entry = dict(latest)
            entry.update(rec)
            entry["run_id"] = run_id
            entry["created_at"] = latest.get("created_at", now)
            entry["updated_at"] = now
            entry["retry_count"] = retry_count
            entry["supervisor_override"] = bool(rec.get("supervisor_override", False))
            self._emit_locked(entry, "transition", from_status)
            return AppendOutcome("transitioned", entry)

    def _budget_check_locked(
        self,
        rec: Dict[str, Any],
        run_id: str,
        all_latest: Dict[str, Dict[str, Any]],
        retry_count: int,
        previously_launched: bool,
    ) -> None:
        """Fail-closed launch budget check (caller holds the ledger lock).

        Raises :class:`BudgetCapExceeded` when this launch attempt would take
        the campaign+phase past its cap (plan count and/or ``DEFAULT_CAPS``),
        when a cap-bearing phase has no entry in a bound plan, or when the run
        itself is not listed in a bound plan.
        """
        campaign = rec.get("campaign_id")
        phase = rec.get("phase")
        cap = self.caps.get(phase)
        plan_count: Optional[int] = None
        if self.plan_path is not None:
            plan = read_run_plan(self.plan_path)
            plan_count = plan["counts"].get((campaign, phase))
            if plan_count is None:
                if cap is not None:
                    raise BudgetCapExceeded(
                        "NOT_PLANNED: phase {!r} carries cap {} but has no planned "
                        "runs in {}".format(phase, cap, self.plan_path)
                    )
            else:
                plan_key = (campaign, phase, rec.get("arm"), rec.get("fold"), rec.get("seed"))
                if plan_key not in plan["keys"]:
                    raise BudgetCapExceeded(
                        "NOT_IN_PLAN: run (campaign={}, phase={}, arm={}, fold={}, "
                        "seed={}) is not in {}".format(*plan_key, self.plan_path)
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
                    if self.plan_path is not None and plan_count is not None
                    else "run_caps",
                )
            )

    def confirm_unseal(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Consume the one-shot fold3/4 confirm-unseal token for this ledger."""
        if self.unseal_path is None:
            raise UnsealExceeded(
                "UNSEAL_NOT_CONFIGURED: ledger has no unseal_path (expected {})".format(
                    DEFAULT_UNSEAL_SUBPATH
                )
            )
        return confirm_unseal(self.unseal_path, run_id=run_id)

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
        "--unseal",
        default=None,
        help="path of the one-shot fold3/4 confirm unseal token "
        "(default rule: state/gates/unseal.json under the campaign root)",
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
    ledger = RunLedger(args.ledger, plan_path=args.plan, unseal_path=args.unseal)
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
