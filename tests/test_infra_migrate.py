"""ext4 migration acceptance (plan §5.4, scripts/infra/migrate_data_to_ext4.py).

Covers: dry-run plan fields, copy/verify/commit happy path on tmp dirs
(fs-type check monkeypatched to ext4), resumable copy skip, and the
fail-closed refusal conditions (non-ext4 dst, insufficient free space).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.infra import migrate_data_to_ext4 as mig  # noqa: E402
from scripts.infra.sha_sidecar import sidecar_path  # noqa: E402


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
