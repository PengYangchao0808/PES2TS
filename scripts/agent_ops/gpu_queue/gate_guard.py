"""Fail-closed gate enforcement for formal (non-fixture) campaign execution.

Machine-enforced counterpart of the supervisor gate rule in
``docs/agent_runs/state/parallel_config.json`` (``serial_gates`` /
``gate_rule``): a formal run may only start when its gate record
``docs/agent_runs/state/gates/<gate_id>.json`` exists, says ``status == "pass"``,
and pins the *current* campaign protocol file by sha256.

Design rules (all fail-closed -- missing/invalid/corrupt input is always a
refusal, never a warning-and-continue):

* :class:`GateNotPassedError` is raised when anything at all is wrong:
  unknown/unsafe gate id, missing campaign root, missing gate file,
  unparseable/non-object gate JSON, status other than ``"pass"``,
  missing/malformed ``protocol_sha256``, missing protocol file or
  ``protocol_sha256`` mismatch against the frozen protocol file
  (default ``outputs/pesn/30h_campaign/protocol_30h.json``, 30h plan 7.1;
  overridable via the gate record's ``protocol_file`` key or the
  ``protocol_file`` argument -- the override must stay inside campaign_root).
* The gate record's own ``gate_id`` field, when present, must match the
  gate being asserted (renamed/repurposed gate files are refused).
* :func:`assert_runnable` never mutates anything: it is a pure read-and-check
  guard intended to run *before* any lock is taken or any job is dispatched.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = [
    "DEFAULT_GATES_SUBPATH",
    "DEFAULT_PROTOCOL_SUBPATH",
    "GateGuardError",
    "GateNotPassedError",
    "assert_runnable",
    "gate_path",
    "sha256_file",
]

#: gate records live under the agent_runs state tree (see docs/agent_runs/README.md)
DEFAULT_GATES_SUBPATH = "docs/agent_runs/state/gates"
#: frozen campaign protocol file (30h plan section 7.1)
DEFAULT_PROTOCOL_SUBPATH = "outputs/pesn/30h_campaign/protocol_30h.json"

VALID_GATE_STATUS = "pass"


class GateGuardError(Exception):
    """Base class for gate-guard refusals."""


class GateNotPassedError(GateGuardError):
    """Formal execution refused: gate missing / not passed / protocol mismatch."""


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_gate_id(gate_id: Any) -> str:
    if not isinstance(gate_id, str) or not gate_id.strip():
        raise GateNotPassedError(
            "INVALID_GATE_ID: gate_id must be a non-empty string, got {!r}".format(gate_id)
        )
    if any(sep in gate_id for sep in ("/", "\\", "\x00")) or gate_id in (".", ".."):
        raise GateNotPassedError(
            "INVALID_GATE_ID: {!r} must be a bare gate name (no path separators)".format(
                gate_id
            )
        )
    return gate_id


def gate_path(gate_id: Any, campaign_root: Union[str, Path]) -> Path:
    """Absolute path of the gate record (validating the id first)."""
    safe_id = _safe_gate_id(gate_id)
    root = Path(campaign_root)
    return root / DEFAULT_GATES_SUBPATH / "{}.json".format(safe_id)


def _resolve_protocol_file(
    campaign_root: Path, gate_record: Dict[str, Any], protocol_file: Optional[str]
) -> Path:
    """Pick the protocol file path and refuse anything escaping campaign_root."""
    raw = gate_record.get("protocol_file") or protocol_file or DEFAULT_PROTOCOL_SUBPATH
    if not isinstance(raw, str) or not raw.strip():
        raise GateNotPassedError(
            "PROTOCOL_FILE_INVALID: protocol file reference {!r} is not a path".format(raw)
        )
    candidate = Path(raw)
    if candidate.is_absolute():
        raise GateNotPassedError(
            "PROTOCOL_FILE_INVALID: protocol file must be relative to campaign_root, "
            "got absolute {!r}".format(raw)
        )
    resolved_root = root_resolved = campaign_root.resolve()
    resolved = (campaign_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise GateNotPassedError(
            "PROTOCOL_FILE_INVALID: {!r} escapes campaign_root {}".format(
                raw, root_resolved
            )
        ) from exc
    return resolved


def assert_runnable(
    gate_id: Any,
    campaign_root: Union[str, Path],
    protocol_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Refuse unless ``gate_id`` is a passing, protocol-pinned gate record.

    Returns the parsed gate record on success.  Raises
    :class:`GateNotPassedError` on *any* problem (missing, non-pass, corrupt,
    protocol sha mismatch) -- there is deliberately no warn-and-continue path.
    """
    safe_id = _safe_gate_id(gate_id)
    root = Path(campaign_root)
    if not root.is_dir():
        raise GateNotPassedError(
            "CAMPAIGN_ROOT_MISSING: {} is not a directory".format(campaign_root)
        )

    path = gate_path(safe_id, root)
    if not path.is_file():
        raise GateNotPassedError(
            "GATE_FILE_MISSING: expected gate record at {}".format(path)
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GateNotPassedError(
            "GATE_FILE_UNREADABLE: {}: {}".format(path, exc)
        ) from exc
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GateNotPassedError(
            "GATE_FILE_CORRUPT: {} is not valid JSON: {}".format(path, exc)
        ) from exc
    if not isinstance(record, dict):
        raise GateNotPassedError(
            "GATE_FILE_CORRUPT: {} must contain a JSON object".format(path)
        )

    recorded_id = record.get("gate_id")
    if recorded_id is not None and recorded_id != safe_id:
        raise GateNotPassedError(
            "GATE_ID_MISMATCH: file for {!r} records gate_id={!r}".format(
                safe_id, recorded_id
            )
        )

    status = record.get("status")
    if status != VALID_GATE_STATUS:
        raise GateNotPassedError(
            "GATE_NOT_PASSED: gate {!r} status is {!r} (required {!r})".format(
                safe_id, status, VALID_GATE_STATUS
            )
        )

    recorded_sha = record.get("protocol_sha256")
    if not (isinstance(recorded_sha, str) and len(recorded_sha) == 64 and all(
        c in "0123456789abcdef" for c in recorded_sha
    )):
        raise GateNotPassedError(
            "PROTOCOL_SHA_MISSING: gate {!r} has no valid protocol_sha256".format(safe_id)
        )

    protocol_path = _resolve_protocol_file(root, record, protocol_file)
    if not protocol_path.is_file():
        raise GateNotPassedError(
            "PROTOCOL_FILE_MISSING: {} (gate {!r} pins it)".format(protocol_path, safe_id)
        )
    actual_sha = sha256_file(protocol_path)
    if actual_sha != recorded_sha:
        raise GateNotPassedError(
            "PROTOCOL_SHA_MISMATCH: gate {!r} pins {} but {} hashes to {}".format(
                safe_id, recorded_sha, protocol_path, actual_sha
            )
        )

    out = dict(record)
    out["gate_id"] = safe_id
    out["protocol_file_resolved"] = str(protocol_path)
    out["protocol_sha256_verified"] = actual_sha
    return out
