"""Atomic file-writing helpers for PES2TS.

``atomic_writer`` is the single atomic-write primitive used across the
pipeline (JSON manifests, JSONL ledgers, Parquet tables): data goes to a
sibling ``<name>.tmp-<pid>`` file that is flushed, fsynced, and moved over the
target with :func:`os.replace`.  On failure the temp file is removed and any
pre-existing target is left untouched.
"""

from __future__ import annotations

import errno
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps


def _ensure_parent_dir(target: Path) -> None:
    """Create *target*'s parent directory unless it already exists.

    ``mkdir(parents=True, exist_ok=True)`` raises ``FileExistsError`` when the
    parent path is an existing regular file; skipping the call in that case
    lets the subsequent open raise the accurate ``NotADirectoryError``.
    """
    parent = target.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def atomic_writer(path: str | Path) -> Iterator[BinaryIO]:
    """Yield a binary temp-file stream that is atomically published to *path*.

    On a clean exit the stream is flushed and fsynced, then ``os.replace``
    moves the temp file over *path*.  On any failure the temp file is deleted
    and *path* is left untouched.
    """
    target = Path(path)
    _ensure_parent_dir(target)
    tmp = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    try:
        try:
            with tmp.open("wb") as handle:
                yield handle
                handle.flush()
                os.fsync(handle.fileno())
        except FileNotFoundError as exc:
            # Windows reports ERROR_PATH_NOT_FOUND when a parent component is
            # a regular file; POSIX reports ENOTDIR for the same condition.
            if target.parent.is_file():
                raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(tmp)) from exc
            raise
        os.replace(tmp, target)
    finally:
        # A successful os.replace already consumed tmp.  tmp.exists() is False
        # when the parent is a regular file, which also skips an unlink that
        # would raise NotADirectoryError and mask the real error.
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_json(path: str | Path, obj: JSONValue) -> None:
    """Atomically write *obj* as canonical UTF-8 JSON to *path*.

    The parent directory is created when missing.  Serialization is delegated
    to :func:`stable_json_dumps`, so key order in *obj* never changes the file
    contents.
    """
    logger = logging.getLogger(__name__)
    payload = stable_json_dumps(obj).encode("utf-8")
    with atomic_writer(path) as handle:
        handle.write(payload)
    logger.debug("Wrote JSON %s (%d bytes)", path, len(payload))


def read_json(path: str | Path) -> JSONValue:
    """Read and parse the JSON document at *path*."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
