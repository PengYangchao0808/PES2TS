"""Hashing and canonical-serialization helpers for PES2TS.

File digests are computed with a streaming loop over fixed-size chunks so that
multi-gigabyte artifacts (for example the 9.9 GB ``B3LYPD3_TZVP_IRC.h5``) can be
hashed in constant memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final, Protocol

#: Read size used by the streaming file digests (1 MiB).
HASH_CHUNK_SIZE: Final[int] = 1024 * 1024

#: Any scalar accepted in JSON.
JSONScalar = str | int | float | bool | None
#: Any value accepted by, or produced from, JSON (recursive).
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class _Hasher(Protocol):
    """Minimal hashlib interface needed for streaming digests."""

    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


def _hash_file(path: str | Path, hasher: _Hasher) -> str:
    """Stream the file at *path* through *hasher* and return its hex digest."""
    with Path(path).open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*.

    The file is read in 1 MiB chunks, so peak memory stays constant regardless
    of file size.
    """
    return _hash_file(path, hashlib.sha256())


def md5_file(path: str | Path) -> str:
    """Return the MD5 hex digest of the file at *path*.

    MD5 is required to match the Zenodo-recorded source checksums; it is used
    for integrity verification against a trusted manifest, never as a security
    primitive.
    """
    return _hash_file(path, hashlib.md5(usedforsecurity=False))


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of the bytes *data*."""
    return hashlib.sha256(data).hexdigest()


def stable_json_dumps(obj: JSONValue) -> str:
    """Serialize *obj* to key-order-insensitive, UTF-8-safe JSON text.

    Dict keys are sorted, non-ASCII characters stay literal, and separators are
    compact, so two logically equal objects always serialize to identical text.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
