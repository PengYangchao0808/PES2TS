"""flock-based exclusive GPU mutex for the PES2TS 30h campaign.

Exactly one GPU job at a time is enforced with ``fcntl.flock`` on a lock file
(default ``/tmp/opencode/pes2ts_gpu.lock``).  The flock is the source of truth
for mutual exclusion; a JSON holder record (``<lock>.holder.json``) written
alongside records who holds it: pid, lane, run_id, acquired_at, cmdline.

Stale detection: if the holder pid no longer exists the status probe reports
``STALE`` with full holder details.  The lock is **never** auto-released or
unlinked -- releasing a stale lock is a supervisor decision.

No GPU hardware access is required to use this module.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "DEFAULT_LOCK_PATH",
    "LockHandle",
    "acquire",
    "holder_path",
    "main",
    "pid_alive",
    "probe_status",
    "read_holder",
]

DEFAULT_LOCK_PATH = "/tmp/opencode/pes2ts_gpu.lock"
HOLDER_SUFFIX = ".holder.json"


def holder_path(lock_path: Any) -> Path:
    return Path(str(lock_path) + HOLDER_SUFFIX)


def pid_alive(pid: Any) -> bool:
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
        return True
    except OverflowError:
        return False
    return True


def _cmdline() -> str:
    try:
        raw = Path("/proc/self/cmdline").read_bytes()
        text = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        if text:
            return text
    except OSError:
        pass
    return " ".join(sys.argv)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """tmp + os.replace so readers never see a half-written holder record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_holder(lock_path: Any) -> Optional[Dict[str, Any]]:
    """Return the holder record, ``None`` if absent, or a marker if corrupt."""
    path = holder_path(lock_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"pid": None, "corrupt": True}
    if not isinstance(data, dict):
        return {"pid": None, "corrupt": True}
    return data


@dataclass
class LockHandle:
    """Owns the flock; the lock dies with the process if release() is skipped."""

    path: Path
    fd: int
    holder: Dict[str, Any] = field(default_factory=dict)
    _released: bool = False

    # metadata passthrough
    @property
    def pid(self) -> Any:
        return self.holder.get("pid")

    @property
    def lane(self) -> Any:
        return self.holder.get("lane")

    @property
    def run_id(self) -> Any:
        return self.holder.get("run_id")

    @property
    def acquired_at(self) -> Any:
        return self.holder.get("acquired_at")

    @property
    def cmdline(self) -> Any:
        return self.holder.get("cmdline")

    def release(self) -> None:
        """Release the flock and drop our own holder record (only if ours)."""
        if self._released:
            return
        hp = holder_path(self.path)
        current = read_holder(self.path)
        if current is not None and current.get("pid") == os.getpid():
            with contextlib.suppress(OSError):
                hp.unlink()
        with contextlib.suppress(OSError):
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
        self._released = True

    def __enter__(self) -> "LockHandle":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


def acquire(
    lock_path: Any = DEFAULT_LOCK_PATH,
    *,
    blocking: bool = False,
    lane: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Optional[LockHandle]:
    """Try to take the GPU mutex.

    Returns a :class:`LockHandle` on success, or ``None`` when the lock is
    held by someone else and ``blocking=False``.
    """
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(fd, flags)
    except OSError as exc:
        os.close(fd)
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return None
        raise
    holder: Dict[str, Any] = {
        "pid": os.getpid(),
        "lane": lane,
        "run_id": run_id,
        "acquired_at": time.time(),
        "acquired_at_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cmdline": _cmdline(),
        "lock_path": str(path),
    }
    try:
        _atomic_write_json(holder_path(path), holder)
    except OSError:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        raise
    return LockHandle(path=path, fd=fd, holder=holder)


def probe_status(lock_path: Any = DEFAULT_LOCK_PATH) -> Dict[str, Any]:
    """Classify the lock as FREE / HELD / STALE (see module docstring).

    * holder pid dead (or holder record unreadable) -> ``STALE``: reported
      with full details, never auto-released, never unlinked;
    * lock acquirable -> ``FREE``;
    * otherwise -> ``HELD`` by the recorded (alive) holder, or by an
      unknown holder if the metadata is missing.
    """
    path = Path(lock_path)
    meta = read_holder(path)
    if meta is not None:
        holder_pid = meta.get("pid")
        if not pid_alive(holder_pid):
            return {
                "state": "STALE",
                "stale": True,
                "lock_path": str(path),
                "holder": meta,
                "detail": (
                    "holder pid {!r} no longer exists; lock NOT auto-released "
                    "(supervisor decision)".format(holder_pid)
                ),
            }
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            held = True
        else:
            held = False
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    if not held:
        return {
            "state": "FREE",
            "stale": False,
            "lock_path": str(path),
            "holder": meta,
            "detail": (
                "lock is free; leftover holder metadata present"
                if meta is not None
                else "lock is free"
            ),
        }
    return {
        "state": "HELD",
        "stale": False,
        "lock_path": str(path),
        "holder": meta,
        "detail": (
            "held by pid {} (lane={!r}, run_id={!r})".format(
                meta.get("pid"), meta.get("lane"), meta.get("run_id")
            )
            if meta is not None
            else "held by unknown holder (no metadata record)"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu_lock", description="PES2TS single-GPU mutex status probe"
    )
    parser.add_argument("--lock", default=DEFAULT_LOCK_PATH, help="lock file path")
    parser.add_argument(
        "--status",
        action="store_true",
        help="print FREE/HELD/STALE status as JSON (always exit 0 when the probe works)",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.status:
        _build_parser().print_help()
        return 2
    try:
        status = probe_status(args.lock)
    except OSError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 2
    print(json.dumps(status, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
