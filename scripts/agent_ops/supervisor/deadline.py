"""Absolute campaign time window + phase deadline enforcement (WP-L2).

Implements task_state ``time_control`` (E0 sign-off pins absolute start/stop):

* :func:`set_campaign_window` validates the ISO-8601 instants (start <
  hard_stop), then rewrites ``task_state.json`` via tmp + ``os.replace``
  preserving every other top-level field.  Missing/corrupt state or an
  inverted window always raises -- fail-closed.
* :func:`check` requires a pinned window, computes
  ``remaining = hard_stop - now`` and reports every phase carrying
  ``deadline_utc`` as ``phase:blocked`` when overdue (never mutates state).
* :func:`guard_dispatch` refuses a new dispatch when the remaining budget is
  less than the task's ``eta_h`` (raises :class:`DispatchRefused`), or when
  the window/state is unusable (:class:`TimeControlError`) -- refusing to
  prove safety is refusing to run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "DEFAULT_TASK_STATE_PATH",
    "DispatchRefused",
    "TimeControlError",
    "check",
    "guard_dispatch",
    "main",
    "set_campaign_window",
]

DEFAULT_TASK_STATE_PATH = "docs/agent_runs/state/task_state.json"


class TimeControlError(Exception):
    """Unusable/missing time control input (always a refusal)."""


class DispatchRefused(TimeControlError):
    """Remaining wall-clock budget is insufficient for the task ETA."""


def _parse_instant(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise TimeControlError(
            "INVALID_WINDOW: {} must be a non-empty ISO-8601 string, got {!r}".format(
                field, value
            )
        )
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TimeControlError(
            "INVALID_WINDOW: {} {!r} is not ISO-8601: {}".format(field, value, exc)
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _read_state(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise TimeControlError("TASK_STATE_MISSING: {}".format(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TimeControlError("TASK_STATE_CORRUPT: {}: {}".format(path, exc)) from exc
    if not isinstance(data, dict):
        raise TimeControlError("TASK_STATE_CORRUPT: {} is not a JSON object".format(path))
    return data


def _write_state(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def set_campaign_window(
    start_utc: Any,
    hard_stop_utc: Any,
    task_state_path: Any = DEFAULT_TASK_STATE_PATH,
) -> Dict[str, Any]:
    """Pin the absolute campaign window into ``task_state.time_control``.

    Preserves every other field of the state file (temp + rename write).
    """
    start = _parse_instant(start_utc, "start_utc")
    stop = _parse_instant(hard_stop_utc, "hard_stop_utc")
    if stop <= start:
        raise TimeControlError(
            "INVALID_WINDOW: hard_stop_utc ({}) must be after start_utc ({})".format(
                _iso(stop), _iso(start)
            )
        )
    path = Path(task_state_path)
    state = _read_state(path)
    control = state.get("time_control")
    if control is None:
        control = {}
    if not isinstance(control, dict):
        raise TimeControlError(
            "TASK_STATE_CORRUPT: time_control must be an object, got {!r}".format(control)
        )
    control = dict(control)
    control["campaign_started_at_utc"] = _iso(start)
    control["hard_stop_at_utc"] = _iso(stop)
    state["time_control"] = control
    _write_state(path, state)
    return dict(control)


def _window(path: Path) -> tuple:
    state = _read_state(path)
    control = state.get("time_control")
    if not isinstance(control, dict):
        raise TimeControlError(
            "TIME_CONTROL_MISSING: {} has no time_control object (pin the window "
            "at E0 sign-off first)".format(path)
        )
    start_raw = control.get("campaign_started_at_utc")
    stop_raw = control.get("hard_stop_at_utc")
    if start_raw is None or stop_raw is None:
        raise TimeControlError(
            "TIME_CONTROL_UNPINNED: campaign_started_at_utc/hard_stop_at_utc are "
            "not set in {}".format(path)
        )
    start = _parse_instant(start_raw, "campaign_started_at_utc")
    stop = _parse_instant(stop_raw, "hard_stop_at_utc")
    if stop <= start:
        raise TimeControlError(
            "TIME_CONTROL_INVALID: hard_stop_at_utc ({}) is not after start ({})".format(
                _iso(stop), _iso(start)
            )
        )
    return state, start, stop


def check(
    task_state_path: Any = DEFAULT_TASK_STATE_PATH, now: Optional[datetime] = None
) -> Dict[str, Any]:
    """Report overdue phases and the remaining wall-clock budget.

    Returns ``{"blocked_phases": [...], "phases": [{"id", "report", ...}],
    "remaining_h": float, "hard_stop_at_utc": ..., "campaign_report": ...}``.
    A phase with ``deadline_utc`` in the past is reported ``phase:blocked``.
    Missing/corrupt state or an unpinned window raises :class:`TimeControlError`.
    """
    path = Path(task_state_path)
    state, start, stop = _window(path)
    current = now if now is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)

    remaining_s = (stop - current).total_seconds()
    blocked: List[str] = []
    reports: List[Dict[str, Any]] = []
    phases = state.get("phases")
    if phases is None:
        phases = []
    if not isinstance(phases, list):
        raise TimeControlError("TASK_STATE_CORRUPT: phases must be a list")
    for phase in phases:
        if not isinstance(phase, dict):
            raise TimeControlError("TASK_STATE_CORRUPT: phase entries must be objects")
        deadline_raw = phase.get("deadline_utc")
        if deadline_raw is None:
            continue  # no absolute deadline pinned for this phase
        phase_id = str(phase.get("id", "<unknown>"))
        deadline = _parse_instant(deadline_raw, "deadline_utc of {}".format(phase_id))
        overdue = current > deadline
        report = "phase:blocked" if overdue else "phase:ok"
        entry = {
            "id": phase_id,
            "report": report,
            "deadline_utc": _iso(deadline),
            "hours_to_deadline": round((deadline - current).total_seconds() / 3600.0, 3),
        }
        reports.append(entry)
        if overdue:
            blocked.append(phase_id)

    campaign_report = "campaign:ok" if remaining_s > 0 else "campaign:hard_stop_reached"
    return {
        "campaign_report": campaign_report,
        "blocked_phases": blocked,
        "phases": reports,
        "remaining_h": round(remaining_s / 3600.0, 6),
        "started_at_utc": _iso(start),
        "hard_stop_at_utc": _iso(stop),
    }


def guard_dispatch(
    eta_h: Any,
    task_state_path: Any = DEFAULT_TASK_STATE_PATH,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Refuse a new dispatch whose ETA exceeds the remaining wall clock.

    Raises :class:`DispatchRefused` when ``remaining < eta_h`` (or the window
    has already elapsed); returns the ``check()`` report when dispatch is
    allowed.
    """
    if isinstance(eta_h, bool) or not isinstance(eta_h, (int, float)) or eta_h < 0:
        raise TimeControlError(
            "INVALID_ETA: eta_h must be a non-negative number, got {!r}".format(eta_h)
        )
    report = check(task_state_path, now=now)
    if report["remaining_h"] <= 0:
        raise DispatchRefused(
            "DISPATCH_REFUSED: hard stop reached (remaining_h={})".format(
                report["remaining_h"]
            )
        )
    if report["remaining_h"] < float(eta_h):
        raise DispatchRefused(
            "DISPATCH_REFUSED: remaining {}h < task eta {}h "
            "(hard stop {})".format(report["remaining_h"], eta_h, report["hard_stop_at_utc"])
        )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deadline", description="Campaign window + phase deadline control (fail-closed)"
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_TASK_STATE_PATH,
        help="path to task_state.json",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    set_cmd = sub.add_parser("set", help="pin the absolute campaign window")
    set_cmd.add_argument("--start", required=True, help="campaign start UTC (ISO-8601)")
    set_cmd.add_argument("--hard-stop", required=True, help="hard stop UTC (ISO-8601)")

    sub.add_parser("check", help="report blocked phases + remaining hours")

    guard = sub.add_parser("guard-dispatch", help="refuse dispatch if remaining < eta")
    guard.add_argument("--eta-h", type=float, required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "set":
            control = set_campaign_window(args.start, args.hard_stop, args.state)
            print(json.dumps(control, sort_keys=True))
            return 0
        if args.cmd == "check":
            print(json.dumps(check(args.state), sort_keys=True))
            return 0
        if args.cmd == "guard-dispatch":
            print(json.dumps(guard_dispatch(args.eta_h, args.state), sort_keys=True))
            return 0
        return 2
    except DispatchRefused as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    except TimeControlError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    except OSError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
