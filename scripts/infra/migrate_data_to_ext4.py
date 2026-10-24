#!/usr/bin/env python3
"""Migrate a data tree to an ext4 mount, two-phase, sha-verified (plan §5.4).

Why: the V5.3-MP data tree lives on ``/mnt/e`` (drvfs/9p — slow random
access, unreliable large mmaps).  Plan §5.4 requires moving
``outputs/pesn/v53mp/data`` onto a Linux-native ext4 mount, keeping a
manifest + symlinks behind in the repo.

Phases (each is an explicit CLI flag; default is the DRY-RUN plan)::

    # 1. plan (default, read-only): files, bytes, src/dst fs type, free check
    python -m scripts.infra.migrate_data_to_ext4

    # 2. copy (resumable; skips dst files whose sha already matches)
    python -m scripts.infra.migrate_data_to_ext4 --copy

    # 3. verify: hash EVERY dst file against the src sidecar sha
    python -m scripts.infra.migrate_data_to_ext4 --verify

    # 4. commit: manifest to BOTH sides, then src files -> symlinks
    python -m scripts.infra.migrate_data_to_ext4 --commit

    # 5. post-commit audit: full sha256 of every manifest entry on dst
    #    (sidecars are NOT trusted; missing/size/sha drift all fail)
    python -m scripts.infra.migrate_data_to_ext4 --reverify-data --target ...

Durability / memory (WSL: 40 GB RAM, 54 GB tree):

* every copied file is ``fsync``ed BEFORE the tmp -> dst rename, so an
  interrupted run can never leave a 0-byte dst behind a valid sidecar;
* after each file is copied (and after each dst file is re-hashed during
  ``--verify``) its page cache is dropped via ``posix_fadvise(DONTNEED)``
  so the guest page cache stays flat instead of growing to the full tree
  size.  ``--no-drop-cache`` restores the old keep-everything behaviour.

Guards (fail-closed, checked before any byte moves):

* destination must be **ext4** (``statfs`` f_type == 0xEF53);
* destination free space must be >= **1.2x** the tree size;
* ``--target`` is REQUIRED for copy/verify/commit (the detector only
  *proposes* a target in the plan).

``--commit`` re-runs ``--verify`` semantics itself unless the same
invocation already passed ``--verify``: symlinks are only ever created
after a full, in-process verification pass.

Post-commit integrity guards (fail-closed; the committed tree is all
symlinks, so a naive re-run would operate on an empty payload scan):

* a repeated ``--verify`` would scan 0 regular payload files and report
  ``ok=true`` — it now fails with ``PAYLOAD_EMPTY`` when committed
  symlinks AND a migration manifest are present (audit with
  ``--reverify-data`` instead);
* a repeated ``--commit`` would overwrite the only complete manifest on
  both sides with an ``n_files: 0`` one — it now refuses with
  ``MANIFEST_SHRINK_REFUSED`` when the dst manifest already records more
  files than the current scan finds;
* ``--reverify-data`` reads the migration manifest (src reports dir or
  the dst root, newest wins) and force-re-hashes every recorded
  ``{path, sha256, size}`` on dst with a full sha256 (sidecar cache NOT
  trusted), dropping each file's page cache via fadvise afterwards.

Layout after a full run (plan §5.4)::

    /pes2ts_data/v53mp/...                      # real files (ext4)
    outputs/pesn/v53mp/data/<file>              # symlink -> ext4 copy
    outputs/pesn/v53mp/reports/data_migration_manifest_<date>.json
    /pes2ts_data/v53mp/data_migration_manifest_<date>.json
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import shutil
import stat as stat_mod
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.infra.sha_sidecar import (  # noqa: E402
    SIDECAR_SUFFIX,
    SidecarError,
    sha256_file,
    sidecar_path,
    sidecar_sha256,
)

DEFAULT_SRC = "outputs/pesn/v53mp/data"
#: where the src-side manifest lands (repo reports dir, plan §5.4)
DEFAULT_MANIFEST_DIR = "outputs/pesn/v53mp/reports"
#: migration manifest file pattern (both sides, plan §5.4)
MANIFEST_GLOB = "data_migration_manifest_*.json"
#: propose <largest-ext4-mount>/pes2ts_data/v53mp
PROPOSED_REL = "pes2ts_data/v53mp"

EXT4_SUPER_MAGIC = 0xEF53
#: refusal threshold: dst free must cover 1.2x the tree (headroom for the
#: manifest copies + avoid filling the fs to the brim)
FREE_SPACE_FACTOR = 1.2
#: per-file page-cache eviction is ON by default (CLI ``--no-drop-cache``
#: flips it); a 54 GB verify otherwise pins the whole tree in guest memory
DEFAULT_DROP_CACHE = True

ProgressFn = Callable[[str, int, int], None]


class MigrationError(RuntimeError):
    """Refusal (non-ext4 dst, insufficient space, tree drift, ...)."""


# ---------------------------------------------------------------------------
# fs probing
# ---------------------------------------------------------------------------


def fs_stat(path: Path) -> Tuple[int, int]:
    """(f_type, f_bavail * f_bsize) for the fs holding ``path``.

    ``path`` need not exist: we statfs its nearest existing ancestor, so a
    *proposed* target dir on an existing mount is probeable.
    """
    lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    class StatFS(ctypes.Structure):
        _fields_ = [
            ("f_type", ctypes.c_long),
            ("f_bsize", ctypes.c_ulong),
            ("f_blocks", ctypes.c_ulong),
            ("f_bfree", ctypes.c_ulong),
            ("f_bavail", ctypes.c_ulong),
            ("f_files", ctypes.c_ulong),
            ("f_ffree", ctypes.c_ulong),
            ("f_fsid", ctypes.c_byte * 8),
            ("f_flag", ctypes.c_ulong),
            ("f_namemax", ctypes.c_ulong),
            ("f_spare", ctypes.c_int * 6),
        ]

    lib.statfs.argtypes = [ctypes.c_char_p, ctypes.POINTER(StatFS)]
    lib.statfs.restype = ctypes.c_int
    probe = Path(path)
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    buf = StatFS()
    rc = lib.statfs(os.fsencode(str(probe)), ctypes.byref(buf))
    if rc != 0:
        err = os.strerror(ctypes.get_errno())
        raise MigrationError("STATFS_FAILED: {}: {}".format(path, err))
    free_bytes = int(buf.f_bavail) * int(buf.f_bsize)
    return int(buf.f_type) & 0xFFFFFFFF, free_bytes


def fs_type_name(f_type: int) -> str:
    if f_type == EXT4_SUPER_MAGIC:
        return "ext4"
    return "0x{:x}".format(f_type)


def _is_mount_point(path: Path) -> bool:
    """True for real mount points (`/` and any dir whose st_dev differs
    from its parent). Plain dirs like `/mnt` (a dir ON the root fs) are
    NOT mount points even when statfs would report the parent's fs."""
    try:
        if path.resolve() == Path("/"):
            return True
        return os.stat(path).st_dev != os.stat(path.parent).st_dev
    except OSError:
        return False


def detect_ext4_target(tree_size: int) -> Optional[Path]:
    """Largest-free ext4 mount under /mnt or / -> proposed target dir.

    Only actual mount points are considered (plan §5.4: a Linux-native
    filesystem — not a drvfs subdir that merely lives on an ext4 parent).
    Returns ``None`` when no ext4 mount is visible (e.g. pure-9p setups).
    """
    candidates: List[Tuple[int, Path]] = []
    seen: set = set()
    roots = [Path("/")]
    if Path("/mnt").is_dir():
        for entry in sorted(Path("/mnt").iterdir()):
            if entry.is_dir():
                roots.append(entry)
    for root in roots:
        if not _is_mount_point(root):
            continue
        try:
            f_type, free = fs_stat(root)
        except MigrationError:
            continue
        mount = root.resolve()
        if f_type != EXT4_SUPER_MAGIC or mount in seen:
            continue
        seen.add(mount)
        candidates.append((free, mount))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    biggest_free, mount = candidates[0]
    if tree_size and biggest_free < FREE_SPACE_FACTOR * tree_size:
        return None  # detector never proposes an fs that would refuse anyway
    return mount / PROPOSED_REL


# ---------------------------------------------------------------------------
# tree scan + phases
# ---------------------------------------------------------------------------


def _is_sidecar_artifact(name: str) -> bool:
    """True for a sidecar cache file or an interrupted sidecar write.

    ``sha_sidecar`` writes ``<file>.sha256sidecar.json.tmp.<pid>`` and then
    atomically renames it; a crash can leave the tmp behind (observed as a
    stray file on the real dst tree).  Both forms are derived cache, never
    payload.
    """
    if name.endswith(SIDECAR_SUFFIX):
        return True
    marker = SIDECAR_SUFFIX + ".tmp."
    idx = name.find(marker)
    if idx < 0:
        return False
    return name[idx + len(marker):].isdigit()


def scan_tree(src: Path) -> Tuple[List[Path], int]:
    """(sorted relative file paths, total bytes).

    Symlinks and ``*.sha256sidecar.json`` cache artifacts (including
    interrupted ``*.sha256sidecar.json.tmp.<pid>`` sidecar writes) are
    skipped: the sidecar is derived data (regenerated on demand on either
    side), never part of the migrated payload or the manifest.
    """
    files: List[Path] = []
    total = 0
    for dirpath, _dirnames, filenames in os.walk(src):
        for name in sorted(filenames):
            if _is_sidecar_artifact(name):
                continue
            full = Path(dirpath) / name
            st = os.lstat(full)
            if stat_mod.S_ISLNK(st.st_mode):
                continue
            if not stat_mod.S_ISREG(st.st_mode):
                continue
            files.append(full.relative_to(src))
            total += st.st_size
    files.sort()
    return files, total


def _count_symlinks(tree: Path) -> int:
    """Number of symlink entries under ``tree``.

    ``os.walk`` lists symlink-to-file entries in ``filenames`` (it does not
    follow dir symlinks by default), so this counts the committed links a
    post-``--commit`` src tree consists of.
    """
    n = 0
    for dirpath, _dirnames, filenames in os.walk(tree):
        for name in filenames:
            if os.path.islink(os.path.join(dirpath, name)):
                n += 1
    return n


def _find_migration_manifest(
        search_dirs: Iterable[Optional[Path]]) -> Optional[Path]:
    """Newest ``data_migration_manifest_*.json`` under any of ``search_dirs``.

    ``None`` dirs and missing/unreadable dirs are skipped.  Candidates are
    ordered by ``(name, mtime_ns)`` — the name embeds the YYYYMMDD stamp, so
    the newest manifest wins; ties (same stamp, different dirs) fall back to
    mtime.  This is the completeness record that the post-commit guards and
    ``--reverify-data`` hinge on.
    """
    candidates: List[Tuple[Tuple[str, int], Path]] = []
    for directory in search_dirs:
        if directory is None:
            continue
        try:
            found = sorted(Path(directory).glob(MANIFEST_GLOB))
        except OSError:
            continue
        for path in found:
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                continue
            candidates.append(((path.name, mtime_ns), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _require_ext4(path: Path, what: str) -> None:
    f_type, _free = fs_stat(path)
    if f_type != EXT4_SUPER_MAGIC:
        raise MigrationError(
            "NOT_EXT4: {} is on {} (f_type 0x{:x}); plan §5.4 requires an "
            "ext4 destination".format(what, fs_type_name(f_type), f_type))


def _require_free_space(dst: Path, tree_size: int) -> None:
    _f_type, free = fs_stat(dst)
    needed = int(FREE_SPACE_FACTOR * tree_size)
    if free < needed:
        raise MigrationError(
            "INSUFFICIENT_FREE_SPACE: dst has {:.1f} GB free but "
            "{:.1f} GB required (1.2x tree of {:.1f} GB)".format(
                free / 1e9, needed / 1e9, tree_size / 1e9))


def build_plan(src: Path, dst: Optional[Path]) -> Dict[str, Any]:
    """Dry-run plan dict; never raises on guard failures — records them."""
    if not src.is_dir():
        raise MigrationError("SRC_MISSING: {}".format(src))
    files, total = scan_tree(src)
    src_f_type, _src_free = fs_stat(src)
    plan: Dict[str, Any] = {
        "action": "plan",
        "src": str(src),
        "n_files": len(files),
        "total_bytes": total,
        "total_gb": round(total / 1e9, 3),
        "src_fs_type": fs_type_name(src_f_type),
        "files": [str(p) for p in files[:50]],
        "files_truncated": max(0, len(files) - 50),
    }
    if dst is None:
        proposal = detect_ext4_target(total)
        plan["target"] = None
        plan["proposed_target"] = str(proposal) if proposal else None
        plan["dst_fs_type"] = None
        plan["dst_free_bytes"] = None
        plan["refusal"] = ("no --target given (detector proposed {}; pass "
                           "--target to proceed)".format(proposal))
        return plan
    try:
        dst_f_type, free = fs_stat(dst)
    except MigrationError as exc:
        plan["target"] = str(dst)
        plan["refusal"] = str(exc)
        return plan
    plan["target"] = str(dst)
    plan["dst_fs_type"] = fs_type_name(dst_f_type)
    plan["dst_free_bytes"] = free
    plan["dst_free_gb"] = round(free / 1e9, 3)
    plan["required_bytes"] = int(FREE_SPACE_FACTOR * total)
    refusals: List[str] = []
    if dst_f_type != EXT4_SUPER_MAGIC:
        refusals.append("NOT_EXT4: dst is {}".format(fs_type_name(dst_f_type)))
    if free < int(FREE_SPACE_FACTOR * total):
        refusals.append(
            "INSUFFICIENT_FREE_SPACE: {:.1f} GB free < 1.2x tree "
            "{:.1f} GB".format(free / 1e9, FREE_SPACE_FACTOR * total / 1e9))
    if refusals:
        plan["refusal"] = "; ".join(refusals)
    return plan


def _default_progress(rel: str, done: int, total_files: int) -> None:
    if done == 1 or done == total_files or done % 25 == 0:
        print("[copy] {}/{} {}".format(done, total_files, rel), flush=True)


def _fsync_file(path: Path) -> None:
    """Flush ``path``'s data to stable storage (best effort).

    Called on the ``.partial`` tmp right before ``os.replace``: the rename
    keeps the inode, so this fsync also covers the final dst name.  Without
    it, a crash mid-run can leave a 0-byte dst that the next resume then
    has to detect and recopy.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return  # nothing to flush
    try:
        os.fsync(fd)
    except OSError:
        pass  # best-effort: --verify remains the backstop
    finally:
        os.close(fd)


def _drop_file_cache(path: Path) -> None:
    """Best-effort eviction of ``path``'s page cache via fadvise(2).

    ``POSIX_FADV_DONTNEED`` releases the clean pages of one file, so a full
    54 GB copy/verify keeps the guest page cache flat (WSL: vmmemWSL used
    to grow to ~40 GB otherwise).  Non-Linux (no ``os.posix_fadvise`` /
    ``POSIX_FADV_DONTNEED``), a vanished file, or any kernel refusal is a
    silent no-op — cache pressure is never a reason to fail a migration.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return  # non-POSIX / non-Linux: no-op
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return  # file vanished/unreadable: nothing to evict
    try:
        fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        pass
    finally:
        os.close(fd)


def phase_copy(src: Path, dst: Path, progress: Optional[ProgressFn] = None,
               drop_cache: Optional[bool] = None
               ) -> Dict[str, Any]:
    """Resumable copy: dst files whose sha matches the src sidecar are skipped.

    Each copied file gets its sha sidecar on BOTH sides (src sidecar is
    created on demand; dst sidecar after the copy), so ``--verify`` and any
    later re-run are cheap.  Every tmp is fsynced before the rename (no
    0-byte dst after a crash) and, unless ``drop_cache`` is False (default:
    :data:`DEFAULT_DROP_CACHE`, CLI ``--no-drop-cache``), src+dst page
    cache is dropped per file so a 54 GB run stays flat in guest memory.
    """
    progress = progress or _default_progress
    if drop_cache is None:
        drop_cache = DEFAULT_DROP_CACHE
    _require_ext4(dst, "dst")
    files, total = scan_tree(src)
    _require_free_space(dst, total)
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    bytes_copied = 0
    started = time.time()
    for rel in files:
        src_file = src / rel
        dst_file = dst / rel
        try:
            src_sha = sidecar_sha256(src_file)
        except SidecarError as exc:
            raise MigrationError(str(exc)) from exc
        if dst_file.is_file():
            try:
                if sidecar_sha256(dst_file) == src_sha:
                    skipped += 1
                    progress(str(rel), copied + skipped, len(files))
                    continue
            except SidecarError:
                pass  # dst sidecar missing/stale -> recopy
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_file.with_name(dst_file.name + ".partial")
        shutil.copy2(src_file, tmp)  # keeps src mtime
        _fsync_file(tmp)  # data hits disk BEFORE the rename (no 0-byte dst)
        os.replace(tmp, dst_file)
        # bind the dst sidecar to the just-written file
        if sidecar_sha256(dst_file) != src_sha:
            raise MigrationError(
                "COPY_SHA_MISMATCH: {} (src {} dst {})".format(
                    rel, src_sha[:12], sidecar_sha256(dst_file)[:12]))
        if drop_cache:
            # file is fully written + hashed: release its pages (and the src
            # pages read while copying) so the cache does not grow to tree size
            _drop_file_cache(src_file)
            _drop_file_cache(dst_file)
        copied += 1
        bytes_copied += src_file.stat().st_size
        progress(str(rel), copied + skipped, len(files))
    return {
        "action": "copy",
        "src": str(src),
        "dst": str(dst),
        "n_files": len(files),
        "copied": copied,
        "skipped": skipped,
        "bytes_copied": bytes_copied,
        "elapsed_s": round(time.time() - started, 1),
    }


def phase_verify(src: Path, dst: Path,
                 drop_cache: Optional[bool] = None,
                 manifest_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Hash EVERY dst file (forced reverify) against the src sidecar sha.

    ``drop_cache`` (default :data:`DEFAULT_DROP_CACHE`) evicts each dst
    file's page cache after it was re-hashed, so verifying a 54 GB tree
    does not pin 54 GB in the guest page cache.

    Post-commit drift guard: after ``--commit`` the src tree is ALL
    symlinks, so this scan finds 0 regular payload files.  Reporting
    ``ok=true`` for such an empty scan would let a repeated ``--verify``
    "pass" on nothing and a repeated ``--commit`` would overwrite the only
    complete manifest with an ``n_files: 0`` one.  When committed symlinks
    AND a migration manifest (src reports dir / dst root) are present, the
    result is a ``PAYLOAD_EMPTY`` failure instead; audit the migrated data
    with ``--reverify-data``.
    """
    if drop_cache is None:
        drop_cache = DEFAULT_DROP_CACHE
    files, _total = scan_tree(src)
    mismatches: List[str] = []
    for rel in files:
        src_file = src / rel
        dst_file = dst / rel
        if not dst_file.is_file():
            mismatches.append("{}: missing on dst".format(rel))
            continue
        try:
            src_sha = sidecar_sha256(src_file)
        except SidecarError as exc:
            mismatches.append("{}: {}".format(rel, exc))
            continue
        if sidecar_sha256(dst_file, reverify=True) != src_sha:
            mismatches.append("{}: sha256 mismatch".format(rel))
        if drop_cache:
            _drop_file_cache(dst_file)
    if not files:
        n_links = _count_symlinks(src)
        manifest = _find_migration_manifest([manifest_dir, dst])
        if n_links and manifest is not None:
            return {
                "action": "verify",
                "src": str(src),
                "dst": str(dst),
                "n_files": 0,
                "error": "PAYLOAD_EMPTY",
                "mismatches": [
                    "PAYLOAD_EMPTY: src holds {} committed symlinks and no "
                    "regular payload (migration manifest {}); refusing to "
                    "report ok for an empty scan — audit with "
                    "--reverify-data".format(n_links, manifest)],
                "extra_dst_files": [],
                "ok": False,
            }
    # extras on dst that no longer exist on src are NOT fatal, just reported
    # (dst sidecars are derived cache, not payload)
    extra: List[str] = []
    for dirpath, _dns, filenames in os.walk(dst):
        for name in filenames:
            if _is_sidecar_artifact(name):
                continue
            full = Path(dirpath) / name
            rel = full.relative_to(dst)
            if not (src / rel).is_file():
                extra.append(str(rel))
    return {
        "action": "verify",
        "src": str(src),
        "dst": str(dst),
        "n_files": len(files),
        "mismatches": mismatches,
        "extra_dst_files": sorted(extra),
        "ok": not mismatches,
    }


def phase_commit(src: Path, dst: Path,
                 manifest_dir: Path, verified: bool) -> Dict[str, Any]:
    """Manifest to BOTH sides, then src regular files -> symlinks to dst.

    ``verified`` must be True (the CLI sets it only after a full
    ``phase_verify`` in this process or the passed ``--verify``).

    Fail-closed shrink guard: if the dst side already holds a migration
    manifest recording ``n_files > 0`` and this scan finds fewer payload
    files (including the post-commit all-symlink scan of 0), the commit is
    refused with ``MANIFEST_SHRINK_REFUSED`` BEFORE any manifest byte is
    written — a repeated ``--commit`` must never clobber the only complete
    manifest with an ``n_files: 0`` one.
    """
    if not verified:
        raise MigrationError(
            "COMMIT_WITHOUT_VERIFY: run --verify first (or pass --commit "
            "in the same invocation; it re-verifies before switching)")
    files, total = scan_tree(src)
    existing_manifest = _find_migration_manifest([dst])
    if existing_manifest is not None:
        try:
            old_n = int(json.loads(existing_manifest.read_text(
                encoding="utf-8")).get("n_files", 0))
        except (OSError, ValueError, TypeError, AttributeError):
            old_n = 0
        if old_n > 0 and len(files) < old_n:
            raise MigrationError(
                "MANIFEST_SHRINK_REFUSED: {} records n_files={} but this "
                "scan found only {} payload files under {}; refusing to "
                "overwrite the manifest (audit with --reverify-data)".format(
                    existing_manifest, old_n, len(files), src))
    stamp = time.strftime("%Y%m%d")
    manifest: Dict[str, Any] = {
        "schema": "pes2ts_data_migration_v1",
        "src": str(src.resolve()),
        "dst": str(dst.resolve()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(files),
        "total_bytes": total,
        "files": [],
    }
    for rel in files:
        manifest["files"].append({
            "path": str(rel),
            "sha256": sidecar_sha256(src / rel),
            "size": (src / rel).stat().st_size,
        })
    manifest_dir.mkdir(parents=True, exist_ok=True)
    src_manifest = manifest_dir / "data_migration_manifest_{}.json".format(
        stamp)
    dst_manifest = dst / "data_migration_manifest_{}.json".format(stamp)
    payload = json.dumps(manifest, sort_keys=True, indent=1) + "\n"
    src_manifest.write_text(payload, encoding="utf-8")
    dst_manifest.write_text(payload, encoding="utf-8")
    # switch src files -> symlinks, one by one (a crash mid-way leaves a
    # mixed tree; the manifest written above is the complete record and
    # --reverify-data audits the dst payload — a re-run --commit on the
    # shrunk regular-file scan is refused by the MANIFEST_SHRINK_REFUSED
    # guard above, so the manifest can never be clobbered)
    linked = 0
    for rel in files:
        src_file = src / rel
        target = (dst / rel).resolve()
        tmp_link = src_file.with_name(src_file.name + ".symlink.tmp")
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        os.symlink(str(target), tmp_link)
        os.replace(tmp_link, src_file)
        linked += 1
    return {
        "action": "commit",
        "src_manifest": str(src_manifest),
        "dst_manifest": str(dst_manifest),
        "n_files": len(files),
        "symlinked": linked,
        "ok": True,
    }


def phase_reverify_data(dst: Path, manifest_dir: Optional[Path] = None,
                        drop_cache: Optional[bool] = None) -> Dict[str, Any]:
    """Post-commit audit: full sha256 of every manifest entry on dst.

    Reads the migration manifest (src reports dir if given, else/also the
    dst root — newest ``data_migration_manifest_*.json`` wins), then re-
    hashes each recorded ``{path, sha256, size}`` against the dst file.
    The sidecar cache is NOT trusted: every file gets a fresh streamed
    sha256.  Missing file, size drift and sha drift are all mismatches
    (fail).  Unless ``drop_cache`` is False (default:
    :data:`DEFAULT_DROP_CACHE`), each dst file's page cache is released via
    fadvise after hashing, so auditing 54 GB stays flat in guest memory.
    """
    if drop_cache is None:
        drop_cache = DEFAULT_DROP_CACHE
    manifest_path = _find_migration_manifest([manifest_dir, dst])
    if manifest_path is None:
        raise MigrationError(
            "MANIFEST_MISSING: no {} under {} or dst {}".format(
                MANIFEST_GLOB, manifest_dir, dst))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = list(manifest["files"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise MigrationError(
            "MANIFEST_UNREADABLE: {}: {}".format(manifest_path, exc)) from exc
    mismatches: List[str] = []
    for entry in entries:
        try:
            rel = str(entry["path"])
            expected_sha = str(entry["sha256"])
            expected_size = int(entry["size"])
        except (TypeError, ValueError, KeyError):
            mismatches.append("malformed manifest entry: {!r}".format(entry))
            continue
        dst_file = dst / rel
        try:
            st = os.stat(dst_file)
        except OSError:
            mismatches.append("{}: missing on dst".format(rel))
            continue
        if st.st_size != expected_size:
            mismatches.append(
                "{}: size mismatch (manifest {} != dst {})".format(
                    rel, expected_size, st.st_size))
        try:
            actual = sha256_file(dst_file)  # full rehash, no sidecar cache
        except OSError as exc:
            mismatches.append("{}: unreadable on dst ({})".format(rel, exc))
            continue
        if actual != expected_sha:
            mismatches.append("{}: sha256 mismatch".format(rel))
        if drop_cache:
            _drop_file_cache(dst_file)
    return {
        "action": "verify_data",
        "dst": str(dst),
        "manifest": str(manifest_path),
        "n_files": len(entries),
        "mismatches": mismatches,
        "ok": not mismatches,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrate_data_to_ext4",
        description="Migrate a data tree to ext4 (plan §5.4); default = plan",
    )
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--target", default=None,
                        help="ext4 destination dir (REQUIRED for "
                             "--copy/--verify/--commit)")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--reverify-data", action="store_true",
                        help="post-commit audit: full sha256 of every file "
                             "recorded in the migration manifest, checked on "
                             "dst (sidecar cache NOT trusted; missing/size/"
                             "sha drift all fail)")
    parser.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR,
                        help="src-side manifest directory")
    parser.add_argument("--no-drop-cache", action="store_true",
                        help="keep each file's page cache after copy/verify "
                             "(default: drop it via fadvise(DONTNEED) so a "
                             "54 GB run stays flat in guest memory)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    drop_cache = not args.no_drop_cache

    src = Path(args.src)
    dst = Path(args.target) if args.target else None
    mutating = args.copy or args.verify or args.commit or args.reverify_data
    if mutating and dst is None:
        sys.stderr.write(
            "error: TARGET_REQUIRED: --target is required for "
            "--copy/--verify/--commit/--reverify-data (the default phase is "
            "a dry-run plan that only proposes a target)\n")
        return 2

    try:
        if not mutating:
            plan = build_plan(src, dst)
            print(json.dumps(plan, sort_keys=True, indent=1))
            return 1 if plan.get("refusal") else 0

        assert dst is not None
        manifest_dir = Path(args.manifest_dir)
        results: List[Dict[str, Any]] = []
        if args.copy:
            results.append(phase_copy(src, dst, drop_cache=drop_cache))
            print(json.dumps(results[-1], sort_keys=True))
        verified = False
        if args.verify:
            vres = phase_verify(src, dst, drop_cache=drop_cache,
                                manifest_dir=manifest_dir)
            results.append(vres)
            print(json.dumps(vres, sort_keys=True))
            verified = bool(vres["ok"])
        if args.commit:
            # --commit re-verifies in-process unless --verify in the SAME
            # invocation already passed; symlinks never precede verify
            if not verified:
                vres = phase_verify(src, dst, drop_cache=drop_cache,
                                    manifest_dir=manifest_dir)
                results.append(vres)
                print(json.dumps(vres, sort_keys=True))
                verified = bool(vres["ok"])
            cres = phase_commit(src, dst, manifest_dir, verified)
            results.append(cres)
            print(json.dumps(cres, sort_keys=True))
        if args.reverify_data:
            rres = phase_reverify_data(dst, manifest_dir,
                                       drop_cache=drop_cache)
            results.append(rres)
            print(json.dumps(rres, sort_keys=True))
        failed = any(
            (r.get("ok") is False) or r.get("mismatches") for r in results)
        return 1 if failed else 0
    except MigrationError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
