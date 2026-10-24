"""ext4 migration acceptance (plan §5.4, scripts/infra/migrate_data_to_ext4.py).

Covers: dry-run plan fields, copy/verify/commit happy path on tmp dirs
(fs-type check monkeypatched to ext4), resumable copy skip, the
fail-closed refusal conditions (non-ext4 dst, insufficient free space),
and the durability/memory fixes: per-file fsync before the tmp->dst
rename, per-file page-cache drop (fadvise DONTNEED) in copy/verify,
``--no-drop-cache`` wiring, and recopy of a 0-byte dst left by an
interrupted run even when its sidecar records the empty content.

Plus the post-commit integrity guards: ``PAYLOAD_EMPTY`` on a repeated
``--verify`` of the all-symlink committed tree, ``MANIFEST_SHRINK_REFUSED``
on a repeated ``--commit`` (manifest bytes must stay untouched), the
``--reverify-data`` post-commit audit (ok / missing / size drift /
same-size tamper behind a lying sidecar), and scan_tree skipping stray
``*.sha256sidecar.json.tmp.<pid>`` sidecar writes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.infra import migrate_data_to_ext4 as mig  # noqa: E402
from scripts.infra.sha_sidecar import sidecar_path, sidecar_sha256  # noqa: E402

RELS = ("a.csv", "manifest.json", "shards/shard_00000.npz")


@pytest.fixture
def src_tree(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "shards").mkdir(parents=True)
    (src / "a.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")
    (src / "shards" / "shard_00000.npz").write_bytes(b"\x00\x01\x02" * 100)
    (src / "manifest.json").write_text('{"schema": "x"}\n', encoding="utf-8")
    return src


@pytest.fixture
def ext4_fs(monkeypatch: pytest.MonkeyPatch) -> None:
    """fs_stat -> (ext4, plenty of free). tmp dirs stand in for mounts."""

    def fake_fs_stat(path: Path) -> Tuple[int, int]:
        return mig.EXT4_SUPER_MAGIC, 500 * 10**9

    monkeypatch.setattr(mig, "fs_stat", fake_fs_stat)


def test_plan_dry_run_fields(src_tree: Path, ext4_fs: None,
                             tmp_path: Path) -> None:
    dst = tmp_path / "dst_ext4" / "v53mp"
    plan = mig.build_plan(src_tree, dst)
    assert plan["action"] == "plan"
    assert plan["n_files"] == 3
    assert plan["total_bytes"] > 0
    assert plan["src_fs_type"] == "ext4"
    assert plan["dst_fs_type"] == "ext4"
    assert plan["dst_free_bytes"] == 500 * 10**9
    assert plan["required_bytes"] == int(1.2 * plan["total_bytes"])
    assert "refusal" not in plan


def test_plan_without_target_proposes(src_tree: Path,
                                      ext4_fs: None) -> None:
    plan = mig.build_plan(src_tree, None)
    assert plan["target"] is None
    assert "refusal" in plan  # explicit --target required to mutate
    # detector ran without crashing (proposed_target may be None on
    # non-ext4 hosts, but on this WSL box / is ext4)
    assert "proposed_target" in plan


def test_refuse_non_ext4_dst(src_tree: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    # dst lives on 9p (drvfs /mnt/e): refuse before any copy
    monkeypatch.setattr(
        mig, "fs_stat",
        lambda path: (0x01021997, 500 * 10**9))  # 9p magic
    rc = mig.main(["--src", str(src_tree), "--copy",
                    "--target", "/mnt/e/should_not_be_written"])
    assert rc == 1


def test_refuse_insufficient_free_space(src_tree: Path,
                                        monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Path) -> None:
    # dst fs has only 10 bytes free: 1.2x-tree check must refuse
    monkeypatch.setattr(
        mig, "fs_stat",
        lambda path: (mig.EXT4_SUPER_MAGIC, 10))
    rc = mig.main(["--src", str(src_tree), "--copy",
                    "--target", str(tmp_path / "tiny")])
    assert rc == 1


def test_refuse_copy_without_target(src_tree: Path) -> None:
    rc = mig.main(["--src", str(src_tree), "--copy"])
    assert rc == 2


def test_copy_verify_commit_roundtrip(src_tree: Path, ext4_fs: None,
                                      tmp_path: Path) -> None:
    dst = tmp_path / "dst" / "v53mp"
    manifest_dir = tmp_path / "reports"

    # copy
    cres = mig.phase_copy(src_tree, dst, progress=lambda *_: None)
    assert cres["copied"] == 3
    assert cres["skipped"] == 0
    assert (dst / "shards" / "shard_00000.npz").is_file()

    # resumable: second copy skips everything
    cres2 = mig.phase_copy(src_tree, dst, progress=lambda *_: None)
    assert cres2["copied"] == 0
    assert cres2["skipped"] == 3

    # verify passes
    vres = mig.phase_verify(src_tree, dst)
    assert vres["ok"] is True
    assert vres["mismatches"] == []

    # commit: manifests on both sides, src files become symlinks to dst
    cres3 = mig.phase_commit(src_tree, dst, manifest_dir, verified=True)
    assert cres3["symlinked"] == 3
    manifests = list(manifest_dir.glob("data_migration_manifest_*.json"))
    assert len(manifests) == 1
    dst_manifests = list(dst.glob("data_migration_manifest_*.json"))
    assert len(dst_manifests) == 1
    assert manifests[0].read_text(encoding="utf-8") == \
        dst_manifests[0].read_text(encoding="utf-8")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["schema"] == "pes2ts_data_migration_v1"
    assert manifest["n_files"] == 3
    assert {f["path"] for f in manifest["files"]} == {
        "a.csv", "shards/shard_00000.npz", "manifest.json"}
    for entry in manifest["files"]:
        assert len(entry["sha256"]) == 64

    # src side: each original file is now a symlink resolving into dst
    for rel in ("a.csv", "shards/shard_00000.npz", "manifest.json"):
        link = src_tree / rel
        assert link.is_symlink(), rel
        assert dst in link.resolve().parents or link.resolve() == (dst / rel).resolve()
        # content survives the switch
        assert link.read_bytes() == (dst / rel).read_bytes()

    # a post-commit sidecar still validates through the symlink (stat
    # follows to the dst inode; sidecar was rebound at commit-time hashing)
    assert sidecar_path(src_tree / "a.csv").is_file()


def test_commit_refused_without_verify(src_tree: Path, ext4_fs: None,
                                       tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    with pytest.raises(mig.MigrationError, match="COMMIT_WITHOUT_VERIFY"):
        mig.phase_commit(src_tree, dst, tmp_path / "reports", verified=False)


def test_verify_detects_corruption(src_tree: Path, ext4_fs: None,
                                   tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    mig.phase_copy(src_tree, dst, progress=lambda *_: None)
    (dst / "a.csv").write_text("corrupted\n", encoding="utf-8")
    vres = mig.phase_verify(src_tree, dst)
    assert vres["ok"] is False
    assert any("a.csv" in m for m in vres["mismatches"])


# ---------------------------------------------------------------------------
# durability + page-cache drop (the WSL memory / 0-byte-file fix)
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_trace(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str]]:
    """Record ("fsync"|"fadvise", path) events for every os.fsync /
    os.posix_fadvise call, resolving the fd back to its path."""
    events: List[Tuple[str, str]] = []
    fd_paths: dict = {}
    real_open, real_fsync, real_close = os.open, os.fsync, os.close
    real_fadvise = getattr(os, "posix_fadvise", None)

    def t_open(path: object, flags: int, *a: object, **kw: object) -> int:
        fd = real_open(path, flags, *a, **kw)  # type: ignore[arg-type]
        fd_paths[fd] = str(path)
        return fd

    def t_fsync(fd: int) -> None:
        events.append(("fsync", fd_paths.get(fd, "fd:{}".format(fd))))
        return real_fsync(fd)

    def t_fadvise(fd: int, offset: int, length: int, adv: int) -> None:
        assert (offset, length) == (0, 0), "drop-cache must target whole file"
        assert adv == os.POSIX_FADV_DONTNEED
        events.append(("fadvise", fd_paths.get(fd, "fd:{}".format(fd))))
        assert real_fadvise is not None
        return real_fadvise(fd, offset, length, adv)

    def t_close(fd: int) -> None:
        fd_paths.pop(fd, None)
        return real_close(fd)

    monkeypatch.setattr(os, "open", t_open)
    monkeypatch.setattr(os, "fsync", t_fsync)
    monkeypatch.setattr(os, "close", t_close)
    if real_fadvise is not None:
        monkeypatch.setattr(os, "posix_fadvise", t_fadvise)
    return events


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="needs POSIX")
def test_copy_fsyncs_tmp_and_drops_cache_per_file(src_tree: Path,
                                                  ext4_fs: None,
                                                  tmp_path: Path,
                                                  fs_trace: List) -> None:
    dst = tmp_path / "dst"
    cres = mig.phase_copy(src_tree, dst, progress=lambda *_: None)
    assert cres["copied"] == 3
    # exactly one fsync per copied file, on the .partial tmp BEFORE rename
    fsynced = sorted(p for op, p in fs_trace if op == "fsync")
    expected_tmp = sorted(
        str((dst / rel).with_name((dst / rel).name + ".partial"))
        for rel in RELS)
    assert fsynced == expected_tmp
    # after each file: page cache dropped for BOTH src and dst
    fadvised = sorted(p for op, p in fs_trace if op == "fadvise")
    assert fadvised == sorted(
        [str(src_tree / rel) for rel in RELS]
        + [str(dst / rel) for rel in RELS])
    # copy2 mtime semantics unchanged: dst mtime == src mtime
    for rel in RELS:
        st_src = os.stat(src_tree / rel)
        st_dst = os.stat(dst / rel)
        assert st_dst.st_mtime_ns == st_src.st_mtime_ns


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="needs POSIX")
def test_copy_no_drop_cache_keeps_fadvise_off(src_tree: Path, ext4_fs: None,
                                              tmp_path: Path,
                                              fs_trace: List) -> None:
    dst = tmp_path / "dst"
    cres = mig.phase_copy(src_tree, dst, progress=lambda *_: None,
                          drop_cache=False)
    assert cres["copied"] == 3
    assert [e for e in fs_trace if e[0] == "fadvise"] == []
    # durability (fsync) stays on even with the cache kept
    assert len([e for e in fs_trace if e[0] == "fsync"]) == 3


@pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="needs POSIX")
def test_verify_drops_dst_cache_per_file(src_tree: Path, ext4_fs: None,
                                         tmp_path: Path,
                                         fs_trace: List) -> None:
    dst = tmp_path / "dst"
    mig.phase_copy(src_tree, dst, progress=lambda *_: None, drop_cache=False)
    fs_trace.clear()
    vres = mig.phase_verify(src_tree, dst)
    assert vres["ok"] is True
    # one fadvise(DONTNEED) per dst file right after its reverify hash
    fadvised = sorted(p for op, p in fs_trace if op == "fadvise")
    assert fadvised == sorted(str(dst / rel) for rel in RELS)
    # verify never writes -> no fsync at all
    assert not any(op == "fsync" for op, _ in fs_trace)


def test_drop_file_cache_tolerates_errors(tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch
                                          ) -> None:
    target = tmp_path / "f.bin"
    target.write_bytes(b"x" * 10)
    calls: list = []
    monkeypatch.setattr(os, "posix_fadvise",
                        lambda *a: calls.append(a), raising=False)
    mig._drop_file_cache(target)
    assert len(calls) == 1
    fd, offset, length, adv = calls[0]
    assert (offset, length, adv) == (0, 0, os.POSIX_FADV_DONTNEED)

    def boom(*_a: object) -> None:
        raise OSError(5, "EIO")

    monkeypatch.setattr(os, "posix_fadvise", boom)
    mig._drop_file_cache(target)  # OSError swallowed
    # non-Linux: no posix_fadvise attribute at all -> silent no-op
    monkeypatch.delattr(os, "posix_fadvise", raising=False)
    mig._drop_file_cache(target)
    # missing file -> no-op, never raises
    monkeypatch.setattr(os, "posix_fadvise", lambda *_: calls.append(()),
                        raising=False)
    mig._drop_file_cache(tmp_path / "missing.bin")


def test_copy_recopies_zero_byte_dst_with_matching_sidecar(
        src_tree: Path, ext4_fs: None, tmp_path: Path) -> None:
    """Regression: a 0-byte dst whose sidecar already records the empty
    content (the interrupted-run artefact on the real tree) must be
    recopied, not skipped."""
    dst = tmp_path / "dst"
    cres = mig.phase_copy(src_tree, dst, progress=lambda *_: None)
    assert cres["copied"] == 3
    victim = dst / "a.csv"
    original = victim.read_bytes()
    assert original  # sanity
    # truncate to 0 bytes and refresh the sidecar to record the empty file
    victim.write_bytes(b"")
    empty_sha = sidecar_sha256(victim, reverify=True)
    assert victim.stat().st_size == 0
    assert sidecar_path(victim).is_file()
    assert empty_sha != sidecar_sha256(src_tree / "a.csv")
    # resume: only the truncated file is recopied, content restored
    cres2 = mig.phase_copy(src_tree, dst, progress=lambda *_: None)
    assert cres2["copied"] == 1
    assert cres2["skipped"] == 2
    assert victim.read_bytes() == original
    vres = mig.phase_verify(src_tree, dst)
    assert vres["ok"] is True


def test_cli_no_drop_cache_parsing() -> None:
    parser = mig._build_parser()
    # default: drop cache ON (flag unset)
    assert parser.parse_args([]).no_drop_cache is False
    args = parser.parse_args(
        ["--src", "s", "--copy", "--target", "/tmp/d", "--no-drop-cache"])
    assert args.no_drop_cache is True
    assert args.copy is True


def test_main_passes_drop_cache_flag(src_tree: Path,
                                     monkeypatch: pytest.MonkeyPatch,
                                     tmp_path: Path) -> None:
    seen: List[bool] = []

    def fake_copy(src: Path, dst: Path, progress: object = None,
                  drop_cache: bool = True) -> dict:
        seen.append(bool(drop_cache))
        return {"action": "copy", "copied": 1}

    monkeypatch.setattr(mig, "phase_copy", fake_copy)
    target = str(tmp_path / "d")
    rc = mig.main(["--src", str(src_tree), "--copy", "--target", target])
    assert rc == 0
    rc = mig.main(["--src", str(src_tree), "--copy", "--target", target,
                   "--no-drop-cache"])
    assert rc == 0
    assert seen == [True, False]  # default ON, --no-drop-cache OFF


# ---------------------------------------------------------------------------
# post-commit integrity: PAYLOAD_EMPTY / MANIFEST_SHRINK_REFUSED /
# --reverify-data audit / stray tmp sidecars
# ---------------------------------------------------------------------------


def _commit_tree(src_tree: Path, tmp_path: Path
                 ) -> Tuple[Path, Path, Path, Path]:
    """copy + commit; returns (dst, manifest_dir, src_manifest, dst_manifest)."""
    dst = tmp_path / "dst"
    manifest_dir = tmp_path / "reports"
    cres = mig.phase_copy(src_tree, dst, progress=lambda *_: None)
    assert cres["copied"] == 3
    cres2 = mig.phase_commit(src_tree, dst, manifest_dir, verified=True)
    assert cres2["symlinked"] == 3
    return dst, manifest_dir, Path(cres2["src_manifest"]), \
        Path(cres2["dst_manifest"])


def test_post_commit_verify_reports_payload_empty(src_tree: Path,
                                                  ext4_fs: None,
                                                  tmp_path: Path) -> None:
    """Regression (Oracle): after --commit the src tree is all symlinks, so
    scan_tree finds 0 payload files; a repeated --verify must NOT report
    ok=true for that empty scan."""
    dst, manifest_dir, _src_m, _dst_m = _commit_tree(src_tree, tmp_path)
    # sanity: committed tree really is all symlinks, scan finds nothing
    files, _total = mig.scan_tree(src_tree)
    assert files == []
    vres = mig.phase_verify(src_tree, dst, manifest_dir=manifest_dir)
    assert vres["ok"] is False
    assert vres["n_files"] == 0
    assert vres["error"] == "PAYLOAD_EMPTY"
    assert any("PAYLOAD_EMPTY" in m for m in vres["mismatches"])
    # same via CLI: exit 1, not 0
    rc = mig.main(["--src", str(src_tree), "--target", str(dst),
                   "--verify", "--manifest-dir", str(manifest_dir)])
    assert rc == 1
    # pre-commit behaviour unchanged: a clean verify of the real payload
    # tree before commit is still ok (covered by roundtrip test above)


def test_empty_commit_refused_manifest_bytes_unchanged(src_tree: Path,
                                                       ext4_fs: None,
                                                       tmp_path: Path) -> None:
    """Regression (Oracle): re-running --commit after a successful commit
    (here: the real tree had 1649 files) must fail-closed with
    MANIFEST_SHRINK_REFUSED and leave BOTH manifests byte-identical —
    they are the only complete record of the migration."""
    dst, manifest_dir, src_manifest, dst_manifest = _commit_tree(
        src_tree, tmp_path)
    before_src = src_manifest.read_bytes()
    before_dst = dst_manifest.read_bytes()
    assert json.loads(before_src.decode("utf-8"))["n_files"] == 3
    with pytest.raises(mig.MigrationError, match="MANIFEST_SHRINK_REFUSED"):
        mig.phase_commit(src_tree, dst, manifest_dir, verified=True)
    assert src_manifest.read_bytes() == before_src
    assert dst_manifest.read_bytes() == before_dst
    # manifest stays parseable and complete (not a torn/empty overwrite)
    doc = json.loads(src_manifest.read_text(encoding="utf-8"))
    assert doc["n_files"] == 3
    assert len(doc["files"]) == 3
    # CLI path: --commit re-verifies in-process, hits PAYLOAD_EMPTY, then
    # refuses (verified=False -> COMMIT_WITHOUT_VERIFY); rc must be 1 and
    # the manifests untouched either way
    rc = mig.main(["--src", str(src_tree), "--target", str(dst),
                   "--commit", "--manifest-dir", str(manifest_dir)])
    assert rc == 1
    assert src_manifest.read_bytes() == before_src
    assert dst_manifest.read_bytes() == before_dst


def test_reverify_data_ok(src_tree: Path, ext4_fs: None,
                          tmp_path: Path) -> None:
    dst, manifest_dir, _src_m, dst_manifest = _commit_tree(
        src_tree, tmp_path)
    # stray tmp sidecar on dst (the Oracle stray) must not break the audit
    (dst / "a.csv.sha256sidecar.json.tmp.12345").write_text(
        "{}\n", encoding="utf-8")
    rres = mig.phase_reverify_data(dst, manifest_dir)
    assert rres["action"] == "verify_data"
    assert rres["n_files"] == 3
    assert rres["mismatches"] == []
    assert rres["ok"] is True
    assert rres["manifest"] == str(dst_manifest)
    # manifest discovered from the dst root when the reports dir is unknown
    rres2 = mig.phase_reverify_data(dst, tmp_path / "no_such_reports")
    assert rres2["ok"] is True
    assert rres2["manifest"] == str(dst_manifest)
    # CLI wiring: exit 0 on a clean audit
    rc = mig.main(["--src", str(src_tree), "--target", str(dst),
                   "--reverify-data", "--manifest-dir", str(manifest_dir)])
    assert rc == 0
    # no manifest anywhere -> hard refusal
    with pytest.raises(mig.MigrationError, match="MANIFEST_MISSING"):
        mig.phase_reverify_data(tmp_path / "nowhere", tmp_path / "no_reports")


def test_reverify_data_detects_missing_and_size_drift(src_tree: Path,
                                                      ext4_fs: None,
                                                      tmp_path: Path) -> None:
    dst, manifest_dir, _src_m, _dst_m = _commit_tree(src_tree, tmp_path)
    (dst / "shards" / "shard_00000.npz").unlink()
    (dst / "a.csv").write_text("much longer content than before\n",
                               encoding="utf-8")
    rres = mig.phase_reverify_data(dst, manifest_dir)
    assert rres["ok"] is False
    assert any("shard_00000.npz" in m and "missing" in m
               for m in rres["mismatches"])
    assert any("a.csv" in m and "size mismatch" in m
               for m in rres["mismatches"])
    rc = mig.main(["--src", str(src_tree), "--target", str(dst),
                   "--reverify-data", "--manifest-dir", str(manifest_dir)])
    assert rc == 1


def test_reverify_data_detects_tamper_despite_sidecar(src_tree: Path,
                                                      ext4_fs: None,
                                                      tmp_path: Path) -> None:
    """Same-size tamper behind a *refreshed* (lying) sidecar: the audit must
    full-rehash from the manifest, never trust the sidecar cache."""
    dst, manifest_dir, _src_m, _dst_m = _commit_tree(src_tree, tmp_path)
    victim = dst / "a.csv"
    tampered = b"col1,col2\n9,9\n"  # same length as the original content
    assert len(tampered) == victim.stat().st_size
    victim.write_bytes(tampered)
    # refresh the sidecar so it now records the tampered content — the
    # sidecar cache is "valid" for the corrupted file
    sidecar_sha256(victim, reverify=True)
    rres = mig.phase_reverify_data(dst, manifest_dir)
    assert rres["ok"] is False
    assert any("a.csv" in m and "sha256 mismatch" in m
               for m in rres["mismatches"])
    assert len(rres["mismatches"]) == 1  # only the tampered file


def test_scan_tree_skips_stray_tmp_sidecar(tmp_path: Path) -> None:
    """A crash during a sidecar write leaves ``*.sha256sidecar.json.tmp.<pid>``
    behind (seen on the real dst tree); it is derived cache, never payload."""
    src = tmp_path / "tree"
    (src / "shards").mkdir(parents=True)
    (src / "a.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")
    (src / "a.csv.sha256sidecar.json").write_text(
        '{"algorithm": "sha256"}\n', encoding="utf-8")
    (src / "shards" / "shard_00000.npz.sha256sidecar.json.tmp.4242").write_text(
        '{"algorithm": "sha256"', encoding="utf-8")  # torn write, no close
    files, total = mig.scan_tree(src)
    assert [str(p) for p in files] == ["a.csv"]
    assert total == (src / "a.csv").stat().st_size
    # and it is not even counted as an "extra" on dst during verify
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "a.csv").write_text("col1,col2\n1,2\n", encoding="utf-8")
    (dst / "a.csv.sha256sidecar.json.tmp.99").write_text(
        "{}\n", encoding="utf-8")
    vres = mig.phase_verify(src, dst)
    assert vres["extra_dst_files"] == []


def test_cli_reverify_data_parsing_and_target_required() -> None:
    parser = mig._build_parser()
    args = parser.parse_args(
        ["--src", "s", "--reverify-data", "--target", "/tmp/d"])
    assert args.reverify_data is True
    assert parser.parse_args([]).reverify_data is False
    # --reverify-data is a mutating phase: --target is REQUIRED
    assert mig.main(["--src", "s", "--reverify-data"]) == 2
