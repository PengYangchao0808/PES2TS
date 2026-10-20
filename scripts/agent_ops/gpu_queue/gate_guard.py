"""Fail-closed gate enforcement for formal (non-fixture) campaign execution.

Machine-enforced counterpart of the supervisor gate rule in
``docs/agent_runs/state/parallel_config.json`` (``serial_gates`` /
``gate_rule``): a formal run may only start when its gate record
``docs/agent_runs/state/gates/<gate_id>.json`` exists, says ``status == "pass"``,
pins the campaign protocol (sha256 of the frozen campaign yaml), sits behind a
fully passing **upstream gate chain**, and -- for formal runs -- when
``task_state.json`` grants ``authorization.formal_training_authorized``.

Design rules (all fail-closed -- missing/invalid/corrupt input is always a
refusal, never a warning-and-continue):

* :class:`GateNotPassedError` is raised when anything at all is wrong:
  gate id outside the serial chain, missing campaign root, missing gate file,
  unparseable/non-object gate JSON, status other than ``"pass"``, missing/
  malformed ``protocol_sha256``, protocol sha mismatch against the campaign
  yaml (``config/campaigns/pes2ts_30h_20261020.yaml``, ORA-2 blocker 4c),
  ANY upstream gate in G0->G1->G2->G3a->G3->G4 up to ``gate_id`` missing or
  not passed (blocker 4a), or (when ``formal=True``) task_state authorization
  false/missing/corrupt (blocker 4b -- the flags are intentionally false
  until the user signs off, so formal acquisitions MUST refuse).
* The gate record's own ``gate_id`` field, when present, must match the
  gate being asserted; an optional ``protocol_file`` key in the record must
  stay inside campaign_root and hash to the same pinned digest.
* :func:`assert_runnable` never mutates anything: it is a pure read-and-check
  guard intended to run *before* any lock is taken or any job is dispatched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = [
    "DEFAULT_GATES_SUBPATH",
    "DEFAULT_PROTOCOL_SUBPATH",
    "DEFAULT_TASK_STATE_SUBPATH",
    "GATE_ORDER",
    "GateGuardError",
    "GateNotPassedError",
    "assert_runnable",
    "gate_path",
    "sha256_file",
]

#: gate records live under the agent_runs state tree (see docs/agent_runs/README.md)
DEFAULT_GATES_SUBPATH = "docs/agent_runs/state/gates"
#: frozen campaign protocol = the campaign yaml (ORA-2 blocker 4c)
DEFAULT_PROTOCOL_SUBPATH = "config/campaigns/pes2ts_30h_20261020.yaml"
#: authorization flags live here (ORA-2 blocker 4b)
DEFAULT_TASK_STATE_SUBPATH = "docs/agent_runs/state/task_state.json"

#: strict serial chain (campaign yaml ``gates.order`` + parallel_config serial_gates)
GATE_ORDER = (
    "G0-Data",
    "G1-Feature",
    "G2-Move",
    "G3a-Label",
    "G3-V52Supervision",
    "G4-Rollout",
)

VALID_GATE_STATUS = "pass"


class GateGuardError(Exception):
    """Base class for gate-guard refusals."""


class GateNotPassedError(GateGuardError):
    """Formal execution refused: gate/chain/authorization/protocol failure."""


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
    if gate_id not in GATE_ORDER:
        raise GateNotPassedError(
            "INVALID_GATE_ID: {!r} is not in the serial gate chain {}".format(
                gate_id, ", ".join(GATE_ORDER)
            )
        )
    return gate_id


def gate_path(gate_id: Any, campaign_root: Union[str, Path]) -> Path:
    """Absolute path of the gate record (validating the id first)."""
    safe_id = _safe_gate_id(gate_id)
    root = Path(campaign_root)
    return root / DEFAULT_GATES_SUBPATH / "{}.json".format(safe_id)


def _read_gate_record(path: Path, gate_id: str, upstream: bool) -> Dict[str, Any]:
    if not path.is_file():
        code = "CHAIN_GATE_MISSING" if upstream else "GATE_FILE_MISSING"
        raise GateNotPassedError(
            "{}: gate {!r} expects record at {}".format(code, gate_id, path)
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GateNotPassedError("GATE_FILE_UNREADABLE: {}: {}".format(path, exc)) from exc
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
    if recorded_id is not None and recorded_id != gate_id:
        raise GateNotPassedError(
            "GATE_ID_MISMATCH: file for {!r} records gate_id={!r}".format(
                gate_id, recorded_id
            )
        )
    status = record.get("status")
    if status != VALID_GATE_STATUS:
        code = "CHAIN_GATE_NOT_PASSED" if upstream else "GATE_NOT_PASSED"
        raise GateNotPassedError(
            "{}: gate {!r} status is {!r} (required {!r})".format(
                code, gate_id, status, VALID_GATE_STATUS
            )
        )
    return record


def _check_protocol_pin(
    gate_id: str, record: Dict[str, Any], root: Path, require: bool
) -> Optional[str]:
    """protocol_sha256 must equal sha256(campaign yaml); optional record pin too."""
    recorded_sha = record.get("protocol_sha256")
    if recorded_sha is None and not require:
        return None  # upstream gate: no pin recorded -> status check was enough
    if not (
        isinstance(recorded_sha, str)
        and len(recorded_sha) == 64
        and all(c in "0123456789abcdef" for c in recorded_sha)
    ):
        raise GateNotPassedError(
            "PROTOCOL_SHA_MISSING: gate {!r} has no valid protocol_sha256".format(gate_id)
        )

    yaml_path = root / DEFAULT_PROTOCOL_SUBPATH
    if not yaml_path.is_file():
        raise GateNotPassedError(
            "PROTOCOL_FILE_MISSING: {} (gate {!r} must pin the campaign yaml "
            "per ORA-2/4c)".format(yaml_path, gate_id)
        )
    yaml_sha = sha256_file(yaml_path)
    if yaml_sha != recorded_sha:
        raise GateNotPassedError(
            "PROTOCOL_SHA_MISMATCH: gate {!r} pins {} but campaign yaml {} hashes "
            "to {}".format(gate_id, recorded_sha, yaml_path, yaml_sha)
        )

    # optional additional pin declared by the record itself (must hash the same)
    extra = record.get("protocol_file")
    if extra is not None:
        if not isinstance(extra, str) or not extra.strip():
            raise GateNotPassedError(
                "PROTOCOL_FILE_INVALID: protocol file reference {!r} is not a path".format(
                    extra
                )
            )
        candidate = Path(extra)
        if candidate.is_absolute():
            raise GateNotPassedError(
                "PROTOCOL_FILE_INVALID: protocol file must be relative to "
                "campaign_root, got absolute {!r}".format(extra)
            )
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise GateNotPassedError(
                "PROTOCOL_FILE_INVALID: {!r} escapes campaign_root".format(extra)
            ) from exc
        if not resolved.is_file():
            raise GateNotPassedError(
                "PROTOCOL_FILE_MISSING: {} (declared by gate {!r})".format(
                    resolved, gate_id
                )
            )
        extra_sha = sha256_file(resolved)
        if extra_sha != recorded_sha:
            raise GateNotPassedError(
                "PROTOCOL_SHA_MISMATCH: gate {!r} pins {} but {} hashes to {}".format(
                    gate_id, recorded_sha, resolved, extra_sha
                )
            )
    return recorded_sha


def _check_authorization(root: Path) -> Dict[str, Any]:
    """Formal runs require task_state authorization.formal_training_authorized=true."""
    state_path = root / DEFAULT_TASK_STATE_SUBPATH
    if not state_path.is_file():
        raise GateNotPassedError(
            "AUTHORIZATION_MISSING: task_state.json not found at {}".format(state_path)
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateNotPassedError(
            "AUTHORIZATION_CORRUPT: {}: {}".format(state_path, exc)
        ) from exc
    if not isinstance(state, dict):
        raise GateNotPassedError(
            "AUTHORIZATION_CORRUPT: {} is not a JSON object".format(state_path)
        )
    auth = state.get("authorization")
    if not isinstance(auth, dict):
        raise GateNotPassedError(
            "AUTHORIZATION_MISSING: {} has no authorization object".format(state_path)
        )
    if auth.get("formal_training_authorized") is not True:
        raise GateNotPassedError(
            "AUTHORIZATION_REQUIRED: formal_training_authorized is {!r} (must be "
            "true; user sign-off required before any formal run)".format(
                auth.get("formal_training_authorized")
            )
        )
    return auth


def assert_runnable(
    gate_id: Any,
    campaign_root: Union[str, Path],
    protocol_file: Optional[str] = None,
    formal: bool = False,
) -> Dict[str, Any]:
    """Refuse unless ``gate_id`` is a passing, protocol-pinned, chain-covered gate.

    Checks, fail-closed, in order:
      1. ``gate_id`` is a bare name inside the serial chain (G0->G4);
      2. campaign root exists;
      3. **upstream chain**: every gate from G0 up to and including ``gate_id``
         has a record with ``status == "pass"`` (missing = refuse);
      4. the target gate record pins the campaign yaml sha256 (plus any
         record-declared ``protocol_file``, still inside campaign_root);
      5. when ``formal=True``: task_state authorization
         ``formal_training_authorized == true`` (ORA-2/4b).

    Returns the parsed target gate record on success.  Raises
    :class:`GateNotPassedError` on *any* problem.
    """
    safe_id = _safe_gate_id(gate_id)
    root = Path(campaign_root)
    if not root.is_dir():
        raise GateNotPassedError(
            "CAMPAIGN_ROOT_MISSING: {} is not a directory".format(campaign_root)
        )

    chain = GATE_ORDER[: GATE_ORDER.index(safe_id) + 1]
    target_record: Optional[Dict[str, Any]] = None
    for upstream_gate in chain:
        is_target = upstream_gate == safe_id
        record = _read_gate_record(
            gate_path(upstream_gate, root), upstream_gate, upstream=not is_target
        )
        # upstream gates: status must be pass; if they pin a protocol sha it
        # must match the campaign yaml too (fail-closed, no pin => status-only)
        _check_protocol_pin(
            upstream_gate, record, root, require=is_target
        )
        if is_target:
            target_record = record
    assert target_record is not None

    if protocol_file is not None:
        # caller-declared extra pin: same escape rules, must hash to pinned sha
        merged = dict(target_record)
        merged["protocol_file"] = protocol_file
        _check_protocol_pin(safe_id, merged, root, require=True)

    if formal:
        _check_authorization(root)

    out = dict(target_record)
    out["gate_id"] = safe_id
    out["protocol_file_resolved"] = str(root / DEFAULT_PROTOCOL_SUBPATH)
    out["protocol_sha256_verified"] = sha256_file(root / DEFAULT_PROTOCOL_SUBPATH)
    out["upstream_chain_verified"] = list(chain)
    out["formal_authorization_checked"] = bool(formal)
    return out
