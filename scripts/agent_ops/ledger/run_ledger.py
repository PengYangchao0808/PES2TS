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
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
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
    "IllegalTransition",
    "LedgerError",
    "MISSING_PLACEHOLDER",
    "PLACEHOLDER",
    "RetryCapExceeded",
    "RunIdConflict",
    "RunLedger",
    "RunNotFound",
    "compute_run_id",
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

DEFAULT_LEDGER_PATH = "outputs/pes2ts/30h_campaign/run_records.jsonl"
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


def compute_run_id(record: Dict[str, Any]) -> str:
    """Deterministic run_id from (campaign_id, phase, arm, fold, seed[, variant])."""
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
# ledger
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AppendOutcome:
    """Result of :meth:`RunLedger.append`."""

    action: str  # "created" | "noop" | "transitioned"
    entry: Dict[str, Any]


class RunLedger:
    """Append-only run ledger with run_id idempotency and audit events."""

    def __init__(self, path: Any) -> None:
        self.path = Path(path)
        self.events_path = self.path.parent / EVENTS_FILENAME
        self.lock_path = Path(str(self.path) + ".lock")

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
        * status change -> state-machine + retry-cap validation, then append.
        """
        rec = dict(record)
        validate(rec)
        run_id = compute_run_id(rec)
        rec["run_id"] = run_id
        now = _utc_now_iso()
        with _file_lock(self.lock_path):
            latest = self._latest_locked().get(run_id)
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

            entry = dict(latest)
            entry.update(rec)
            entry["run_id"] = run_id
            entry["created_at"] = latest.get("created_at", now)
            entry["updated_at"] = now
            entry["retry_count"] = retry_count
            entry["supervisor_override"] = bool(rec.get("supervisor_override", False))
            self._emit_locked(entry, "transition", from_status)
            return AppendOutcome("transitioned", entry)

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
    ledger = RunLedger(args.ledger)
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
