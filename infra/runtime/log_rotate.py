#!/usr/bin/env python3
"""Size-based rotation for the R1 governance audit log (stdlib only).

Why this exists
---------------
``data/manifests/truth_access_log.jsonl`` is the append-only audit trail of every
ground-truth access (Constitution R1: truth is read only through the audited
reader, and every access leaves a record).  The log grows forever, so it needs a
retention policy that **never loses an access record**: rotation renames the
live file to ``<path>.1`` and shifts the older files to ``.2`` ... ``.N``; nothing
is truncated in place and nothing is ever unlinked by this tool.  The rotated
files stay on disk as ``.1`` ... ``.N`` so an auditor can still reconstruct the
whole access history.

How it is invoked
-----------------
Cron / Windows Task Scheduler only, e.g.::

    # every day 03:10, keep three 64 MiB backups (cron, UTC host)
    10 3 * * * cd /repo && python -m infra.runtime.log_rotate \
        --path data/manifests/truth_access_log.jsonl --max-bytes 64MiB --backups 3

    # dry run first (prints planned renames, touches nothing)
    python -m infra.runtime.log_rotate --path data/manifests/truth_access_log.jsonl \
        --max-bytes 64MiB --dry-run

This module is deliberately **not imported by library/business code**: it is an
operational tool in ``infra.runtime`` (Zone 4).  Active zones must not import
``infra`` business logic and the audit path is never written by trainers.
``python -c "from infra.runtime.log_rotate import rotate"`` is the documented
import check only.

Contract
--------
``rotate(path, max_bytes=64*1024*1024, backups=3, *, dry_run=False) -> dict``

* refuses a missing path (``PATH_MISSING``) or a non-regular file
  (``PATH_NOT_FILE``) instead of silently creating/truncating anything;
* no-op (``rotated=False, reason="below_threshold"``) while
  ``size(path) <= max_bytes``;
* when rotation is due: ``path -> path.1``, ``path.1 -> path.2`` ... in one pass,
  highest slot first, using ``os.replace`` (atomic on the same filesystem);
* refuses (``CHAIN_FULL``) when ``path.<backups>`` already exists: the chain is
  bounded at ``backups`` entries and this tool never deletes a file, so the
  operator must archive/remove the oldest backup deliberately;
* ``dry_run=True`` returns the same report with the planned actions but performs
  no filesystem change;
* failures raise :class:`RotationError` with a stable ``.code`` (P2 style).

The JSON report is machine-readable and is what the CLI prints.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

__all__ = [
    "DEFAULT_BACKUPS",
    "DEFAULT_MAX_BYTES",
    "RotationError",
    "main",
    "rotate",
]

#: Default cap for one audit file: 64 MiB.
DEFAULT_MAX_BYTES = 64 * 1024 * 1024

#: Default number of rotated files retained (``path.1`` .. ``path.N``).
DEFAULT_BACKUPS = 3

#: Accepted size suffixes for the CLI (case-insensitive).
_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
}


class RotationError(RuntimeError):
    """Typed rotation refusal (stable ``.code``, never a bare OSError).

    Codes: ``PATH_MISSING``, ``PATH_NOT_FILE``, ``BAD_MAX_BYTES``,
    ``BAD_BACKUPS``, ``CHAIN_FULL``, ``RENAME_FAILED``.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.message = str(message)
        super().__init__(f"[{self.code}] {self.message}")


def _parse_size(text: str) -> int:
    """Parse ``"64MiB"`` / ``"64M"`` / ``"1048576"`` into a positive byte count."""
    raw = str(text).strip().lower().replace(" ", "")
    if not raw:
        raise argparse.ArgumentTypeError("empty size")
    number = raw
    unit = ""
    for suffix in sorted(_SIZE_UNITS, key=len, reverse=True):
        if suffix and raw.endswith(suffix):
            number, unit = raw[: -len(suffix)], suffix
            break
    try:
        value = float(number)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid size: {text!r}") from exc
    result = int(value * _SIZE_UNITS[unit])
    if result <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive: {text!r}")
    return result


def _slot(path: Path, index: int) -> Path:
    """Return the rotated slot path (``path.1`` .. ``path.N``)."""
    return Path(str(path) + f".{index}")


def _existing_chain(path: Path, backups: int) -> list[dict]:
    """Describe the current rotation chain (path.1 .. path.N)."""
    out = []
    for index in range(1, backups + 1):
        slot = _slot(path, index)
        out.append(
            {
                "index": index,
                "path": str(slot),
                "exists": slot.exists(),
                "bytes": slot.stat().st_size if slot.is_file() else None,
            }
        )
    return out


def _fsync_parent(path: Path) -> None:
    """Best-effort directory fsync so a rename survives a crash (POSIX only)."""
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def rotate(
    path: str | Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backups: int = DEFAULT_BACKUPS,
    *,
    dry_run: bool = False,
) -> dict:
    """Rotate one audit log if it exceeds ``max_bytes``; return a JSON-ready report.

    Parameters
    ----------
    path:
        Audit log path (must already exist as a regular file).
    max_bytes:
        Rotate only when ``size(path) > max_bytes``.
    backups:
        Number of rotated generations retained (``path.1`` .. ``path.<backups>``).
        This tool never deletes beyond that chain; if the oldest slot already
        exists the rotation is refused with ``CHAIN_FULL``.
    dry_run:
        When true, report the planned renames without touching the filesystem.

    Returns
    -------
    dict
        Keys: ``schema``, ``path``, ``rotated``, ``dry_run``, ``reason``,
        ``size_bytes``, ``max_bytes``, ``backups``, ``actions``,
        ``chain_before``, ``chain_after``.

    Raises
    ------
    RotationError
        Stable codes; see :class:`RotationError`.
    """
    target = Path(path)
    if not target.exists():
        raise RotationError("PATH_MISSING", f"audit log not found: {target}")
    if not target.is_file():
        raise RotationError(
            "PATH_NOT_FILE", f"audit log is not a regular file: {target}"
        )
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise RotationError(
            "BAD_MAX_BYTES", f"max_bytes must be a positive int, got {max_bytes!r}"
        )
    if isinstance(backups, bool) or not isinstance(backups, int) or backups < 1:
        raise RotationError(
            "BAD_BACKUPS", f"backups must be a positive int, got {backups!r}"
        )

    size = target.stat().st_size
    report: dict = {
        "schema": "pes2ts_log_rotation_v1",
        "path": str(target),
        "rotated": False,
        "dry_run": bool(dry_run),
        "reason": "",
        "size_bytes": size,
        "max_bytes": max_bytes,
        "backups": backups,
        "actions": [],
        "chain_before": _existing_chain(target, backups),
        "chain_after": [],
    }

    if size <= max_bytes:
        report["reason"] = "below_threshold"
        report["chain_after"] = report["chain_before"]
        return report

    oldest = _slot(target, backups)
    if oldest.exists():
        raise RotationError(
            "CHAIN_FULL",
            f"rotation chain is full ({oldest} exists); this tool never deletes "
            f"audit records -- archive or remove the oldest backup, then retry",
        )

    # Plan highest slot first so an existing .2 is moved to .3 before .1 -> .2.
    planned: list[tuple[Path, Path]] = []
    for index in range(backups, 1, -1):
        src = _slot(target, index - 1)
        dst = _slot(target, index)
        if src.exists():
            planned.append((src, dst))
    planned.append((target, _slot(target, 1)))
    report["actions"] = [
        {"op": "rename", "src": str(src), "dst": str(dst)} for src, dst in planned
    ]

    if dry_run:
        report["reason"] = "would_rotate"
        report["chain_after"] = [
            {"index": index, "path": str(_slot(target, index)), "exists": False, "bytes": None}
            for index in range(1, backups + 1)
        ]
        return report

    for src, dst in planned:
        try:
            os.replace(src, dst)
        except OSError as exc:  # pragma: no cover - filesystem/permission failure
            raise RotationError(
                "RENAME_FAILED", f"cannot rename {src} -> {dst}: {exc}"
            ) from exc
    _fsync_parent(target)

    report["rotated"] = True
    report["reason"] = "rotated"
    report["size_bytes"] = 0
    report["chain_after"] = _existing_chain(target, backups)
    return report


def _print_json(payload: dict) -> None:
    """Write one JSON document to stdout, ASCII-escaped for console safety.

    ``ensure_ascii=True`` keeps the output byte-identical across Windows code
    pages (GBK/UTF-8) and POSIX UTF-8 terminals; the JSON stays machine-readable
    and non-ASCII paths are escaped rather than mojibake'd.
    """
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    """CLI: rotate one audit log and print the JSON report (exit 0/1)."""
    parser = argparse.ArgumentParser(
        description="Size-based rotation for the R1 truth-access audit log "
        "(retains rotated files as .1..N; never truncates or deletes).",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="audit log path, e.g. data/manifests/truth_access_log.jsonl",
    )
    parser.add_argument(
        "--max-bytes",
        type=_parse_size,
        default=DEFAULT_MAX_BYTES,
        help="rotate only when the file is larger than this (default 64MiB)",
    )
    parser.add_argument(
        "--backups",
        type=int,
        default=DEFAULT_BACKUPS,
        help="number of rotated generations to retain (default 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned renames without touching the filesystem",
    )
    args = parser.parse_args(argv)

    try:
        report = rotate(
            args.path,
            max_bytes=args.max_bytes,
            backups=args.backups,
            dry_run=args.dry_run,
        )
    except RotationError as exc:
        _print_json({"status": "error", "code": exc.code, "message": str(exc)})
        return 1

    _print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
