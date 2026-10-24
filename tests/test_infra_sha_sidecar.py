"""Sidecar sha256 cache acceptance (plan §5.5, scripts/infra/sha_sidecar.py).

Covers: fresh hash, stat-tuple reuse (no rehash), invalidation on mtime and
size change, ``--reverify`` forcing a full hash, and the verify CLI's
exit-1-on-mismatch/missing contract.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.infra import sha_sidecar  # noqa: E402
from scripts.infra.sha_sidecar import (  # noqa: E402
    SidecarError,
    sidecar_path,
    sidecar_sha256,
)


def _touch(path: Path, data: bytes = b"hello pes2ts") -> Path:
    path.write_bytes(data)
    return path


def test_fresh_hash_creates_sidecar(tmp_path: Path) -> None:
    target = _touch(tmp_path / "data.bin")
    digest = sidecar_sha256(target)
    assert len(digest) == 64
    side = sidecar_path(target)
    assert side.is_file()
    doc = json.loads(side.read_text(encoding="utf-8"))
    assert doc["sha256"] == digest
    assert doc["size"] == target.stat().st_size
    assert doc["mtime_ns"] == target.stat().st_mtime_ns
    assert doc["inode"] == target.stat().st_ino
    assert doc["algorithm"] == "sha256"


def test_reuse_does_not_rehash(tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
    target = _touch(tmp_path / "data.bin")
    first = sidecar_sha256(target)

    calls = {"n": 0}
    real_sha256_file = sha_sidecar.sha256_file

    def counting_sha256_file(path):
        calls["n"] += 1
        return real_sha256_file(path)

    monkeypatch.setattr(sha_sidecar, "sha256_file", counting_sha256_file)
    second = sidecar_sha256(target)
    assert second == first
    assert calls["n"] == 0, "stat tuple unchanged -> no rehash allowed"


def test_invalidation_on_mtime_change(tmp_path: Path,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    target = _touch(tmp_path / "data.bin")
    sidecar_sha256(target)

    calls = {"n": 0}
    real_sha256_file = sha_sidecar.sha256_file

    def counting_sha256_file(path):
        calls["n"] += 1
        return real_sha256_file(path)

    monkeypatch.setattr(sha_sidecar, "sha256_file", counting_sha256_file)
    # bump mtime only (content identical): stat tuple drifts -> rehash
    st = target.stat()
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    digest = sidecar_sha256(target)
    assert calls["n"] == 1
    doc = json.loads(sidecar_path(target).read_text(encoding="utf-8"))
    assert doc["sha256"] == digest
    assert doc["mtime_ns"] == target.stat().st_mtime_ns


def test_invalidation_on_size_change(tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    target = _touch(tmp_path / "data.bin")
    first = sidecar_sha256(target)

    calls = {"n": 0}
    real_sha256_file = sha_sidecar.sha256_file

    def counting_sha256_file(path):
        calls["n"] += 1
        return real_sha256_file(path)

    monkeypatch.setattr(sha_sidecar, "sha256_file", counting_sha256_file)
    target.write_bytes(b"totally different, longer payload")
    second = sidecar_sha256(target)
    assert calls["n"] == 1
    assert second != first
    doc = json.loads(sidecar_path(target).read_text(encoding="utf-8"))
    assert doc["sha256"] == second
    assert doc["size"] == target.stat().st_size


def test_reverify_forces_full_hash(tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    target = _touch(tmp_path / "data.bin")
    sidecar_sha256(target)

    calls = {"n": 0}
    real_sha256_file = sha_sidecar.sha256_file

    def counting_sha256_file(path):
        calls["n"] += 1
        return real_sha256_file(path)

    monkeypatch.setattr(sha_sidecar, "sha256_file", counting_sha256_file)
    digest = sidecar_sha256(target, reverify=True)
    assert calls["n"] == 1
    assert len(digest) == 64


def test_missing_target_raises(tmp_path: Path) -> None:
    with pytest.raises(SidecarError):
        sidecar_sha256(tmp_path / "nope.bin")


def test_verify_cli_exit_codes(tmp_path: Path) -> None:
    good = _touch(tmp_path / "good.bin")
    sidecar_sha256(good)
    assert sha_sidecar.main(["verify", str(good)]) == 0

    # missing sidecar -> exit 1
    bare = _touch(tmp_path / "bare.bin")
    assert sha_sidecar.main(["verify", str(bare)]) == 1

    # content changed behind the sidecar's back (size+mtime restored):
    # cheap stat check cannot see it, --reverify must catch it
    sneaky = _touch(tmp_path / "sneaky.bin", b"AAAA")
    sidecar_sha256(sneaky)
    st = sneaky.stat()
    sneaky.write_bytes(b"BBBB")               # same length
    os.utime(sneaky, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert sha_sidecar.main(["verify", str(sneaky)]) == 0  # stat still matches
    assert sha_sidecar.main(["verify", "--reverify", str(sneaky)]) == 1


def test_verify_detects_deleted_target(tmp_path: Path) -> None:
    target = _touch(tmp_path / "gone.bin")
    sidecar_sha256(target)
    target.unlink()
    assert sha_sidecar.main(["verify", str(target)]) == 1
