"""Content-addressed sha256 sidecar cache (plan §5.5).

Maps ``(path, size, mtime_ns, inode) -> sha256``.  The digest lives in a
single JSON sidecar next to the target file::

    <target>.sha256sidecar.json

One JSON per target (not one shared index) so the cache is mmap-safe and
parallel-friendly: several workers may hash/discover different files at the
same time without ever contending on a shared index, and a sidecar write is
a single small-file create — never a rewrite of a global table.

Invalidation is silent: ANY drift in ``size`` / ``mtime_ns`` / ``inode``
causes the sidecar to be ignored and the file to be rehashed (then the
sidecar is refreshed in place).  ``reverify=True`` / ``--reverify`` forces
a full rehash even when the stat tuple still matches — use it after
suspicious same-size/same-mtime edits or as a periodic audit.

CLI (exit 1 on any mismatch / missing sidecar)::

    python -m scripts.infra.sha_sidecar verify path1 path2 ...
    python -m scripts.infra.sha_sidecar verify --reverify data/shard_*.npz
    python -m scripts.infra.sha_sidecar update  path1 path2 ...   # refresh sidecars

Importable API::

    from scripts.infra.sha_sidecar import sidecar_sha256
    digest = sidecar_sha256("outputs/pesn/v53mp/data/shard_00000.npz")
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SIDECAR_SUFFIX = ".sha256sidecar.json"
#: read granularity for streaming hashes (1 MiB)
_CHUNK = 1 << 20


class SidecarError(RuntimeError):
    """Sidecar cache unusable (missing target, unreadable/corrupt sidecar)."""


def sidecar_path(target: Any) -> Path:
    """``<target>.sha256sidecar.json`` for a target path."""
    return Path(str(target) + SIDECAR_SUFFIX)


def _stat_tuple(path: Path) -> Tuple[int, int, int]:
    st = os.stat(path)  # follows symlinks: the sidecar binds the CONTENT
    return int(st.st_size), int(st.st_mtime_ns), int(st.st_ino)


def sha256_file(path: Any) -> str:
    """Streamed sha256 of a file (1 MiB chunks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a sidecar JSON; ``None`` when absent or corrupt (fail-open:
    a corrupt sidecar is treated as a cache miss and simply rehashed)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _write_sidecar(target: Path, digest: str,
                   stat_tuple: Tuple[int, int, int]) -> Path:
    size, mtime_ns, inode = stat_tuple
    side = sidecar_path(target)
    payload: Dict[str, Any] = {
        "algorithm": "sha256",
        "sha256": digest,
        "path": str(target),
        "size": size,
        "mtime_ns": mtime_ns,
        "inode": inode,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = side.with_name(side.name + ".tmp.{}".format(os.getpid()))
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, side)  # atomic: readers see old or new, never a torn file
    return side


def _sidecar_matches(doc: Optional[Dict[str, Any]],
                     stat_tuple: Tuple[int, int, int]) -> bool:
    if doc is None:
        return False
    size, mtime_ns, inode = stat_tuple
    try:
        return (
            doc.get("algorithm") == "sha256"
            and int(doc["size"]) == size
            and int(doc["mtime_ns"]) == mtime_ns
            and int(doc["inode"]) == inode
            and isinstance(doc.get("sha256"), str)
            and len(doc["sha256"]) == 64
        )
    except (TypeError, ValueError, KeyError):
        return False


def sidecar_sha256(path: Any, reverify: bool = False) -> str:
    """Return the sha256 of ``path`` via its sidecar cache.

    * stat tuple matches sidecar and ``reverify`` is False -> cached digest,
      **no hashing** (cheap path for repeated verification of big tables);
    * stat drift (size / mtime_ns / inode) or ``reverify`` -> full rehash,
      sidecar refreshed in place, digest returned.

    Raises :class:`SidecarError` when the target does not exist.
    """
    target = Path(path)
    try:
        stat_tuple = _stat_tuple(target)
    except FileNotFoundError as exc:
        raise SidecarError("TARGET_MISSING: {}".format(target)) from exc
    side = sidecar_path(target)
    doc = None if reverify else _read_sidecar(side)
    if _sidecar_matches(doc, stat_tuple):
        return str(doc["sha256"])
    digest = sha256_file(target)
    _write_sidecar(target, digest, stat_tuple)
    return digest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _verify_one(target: Path, reverify: bool) -> Tuple[str, str]:
    """(status, detail) for one target.

    status in ``ok`` | ``missing`` | ``corrupt`` | ``stale`` — the last
    three are FAILs (exit 1).  ``verify`` never silently repairs a FAIL's
    verdict; on a content mismatch the sidecar is refreshed so the NEXT
    verify of an intentionally-replaced file passes, but this run still
    exits 1.
    """
    try:
        stat_tuple = _stat_tuple(target)
    except FileNotFoundError:
        return "missing", "target does not exist"
    side = sidecar_path(target)
    if not side.is_file():
        return "missing", ("no sidecar (run: python -m scripts.infra.sha_sidecar "
                          "update -- <path>)")
    doc = _read_sidecar(side)
    if doc is None:
        return "corrupt", "sidecar unreadable/invalid JSON"
    if not reverify and _sidecar_matches(doc, stat_tuple):
        return "ok", "stat tuple matches sidecar (no hash needed)"
    # stat drift or --reverify: full hash, compare against the recorded digest
    recorded = str(doc.get("sha256", ""))
    actual = sha256_file(target)
    _write_sidecar(target, actual, stat_tuple)  # refresh stat tuple either way
    if actual != recorded:
        return "stale", "sha256 mismatch (recorded {} != actual {})".format(
            recorded[:12], actual[:12])
    return "ok", "full hash matches recorded digest"


def cmd_verify(paths: Iterable[str], reverify: bool) -> int:
    rc = 0
    for raw in paths:
        target = Path(raw)
        status, detail = _verify_one(target, reverify)
        tag = "OK" if status == "ok" else "FAIL"
        if tag == "FAIL":
            rc = 1
        print("{} {} {} {}".format(
            tag, status, target, detail), file=sys.stdout, flush=True)
    return rc


def cmd_update(paths: Iterable[str]) -> int:
    """Force (re)hash + sidecar write for each path (``--reverify`` semantics)."""
    rc = 0
    for raw in paths:
        try:
            digest = sidecar_sha256(raw, reverify=True)
        except SidecarError as exc:
            print("FAIL {} {}".format(raw, exc), flush=True)
            rc = 1
            continue
        print("OK {} {}".format(digest[:16], raw), flush=True)
    return rc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sha_sidecar",
        description="Content-addressed sha256 sidecar cache (plan §5.5)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    verify = sub.add_parser(
        "verify", help="verify sidecars; exit 1 on any mismatch/missing")
    verify.add_argument("--reverify", action="store_true",
                        help="force a full rehash of every path")
    verify.add_argument("paths", nargs="+")

    update = sub.add_parser(
        "update", help="force rehash + rewrite the sidecar for every path")
    update.add_argument("paths", nargs="+")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "verify":
        return cmd_verify(args.paths, args.reverify)
    if args.cmd == "update":
        return cmd_update(args.paths)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
