"""Read-only ACP RESULT manifest integrity checks."""
from __future__ import annotations

import hashlib

import pytest

from pes2ts_core.integration.acp.adapter import ACPMappingError, verify_acp_result_manifest_files


def _manifest(file_bytes: bytes, *, path: str = "trajectory/path.xyz") -> dict:
    return {
        "version": 2,
        "task_id": "acp-task-1",
        "workflow": "PESsearch",
        "status": "completed",
        "products": [{
            "id": "trajectory-1",
            "label": "trajectory",
            "path": path,
            "kind": "trajectory",
            "metadata": {"sha256": hashlib.sha256(file_bytes).hexdigest(),
                         "size_bytes": len(file_bytes)},
        }],
    }


def test_verifies_result_product_hash_and_returns_portable_summary(tmp_path) -> None:
    payload = b"2\nframe\nH 0 0 0\nH 0 0 0.7\n"
    target = tmp_path / "trajectory" / "path.xyz"
    target.parent.mkdir()
    target.write_bytes(payload)

    result = verify_acp_result_manifest_files(_manifest(payload), tmp_path)

    assert result["products_verified"] == 1
    assert result["products"][0]["path"] == "trajectory/path.xyz"
    assert result["products"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert "tmp_path" not in str(result)


@pytest.mark.parametrize("path", ["../outside.xyz", "a/../../outside.xyz", "/absolute.xyz", "C:/outside.xyz", "a//b.xyz", "a/./b.xyz"])
def test_rejects_unsafe_result_relative_paths(tmp_path, path: str) -> None:
    with pytest.raises(ACPMappingError, match="unsafe RESULT path"):
        verify_acp_result_manifest_files(_manifest(b"x", path=path), tmp_path)


def test_rejects_missing_or_tampered_product_files(tmp_path) -> None:
    target = tmp_path / "trajectory" / "path.xyz"
    target.parent.mkdir()
    target.write_bytes(b"tampered")
    with pytest.raises(ACPMappingError, match="SHA256 does not match"):
        verify_acp_result_manifest_files(_manifest(b"expected"), tmp_path)

    target.unlink()
    with pytest.raises(ACPMappingError, match="missing or inaccessible"):
        verify_acp_result_manifest_files(_manifest(b"expected"), tmp_path)


def test_rejects_product_without_digest_metadata(tmp_path) -> None:
    target = tmp_path / "result.dat"
    target.write_bytes(b"data")
    manifest = _manifest(b"data", path="result.dat")
    manifest["products"][0]["metadata"].pop("sha256")
    with pytest.raises(ACPMappingError, match="requires metadata.sha256"):
        verify_acp_result_manifest_files(manifest, tmp_path)


def test_rejects_duplicate_product_paths(tmp_path) -> None:
    target = tmp_path / "result.dat"
    target.write_bytes(b"data")
    manifest = _manifest(b"data", path="result.dat")
    manifest["products"].append({**manifest["products"][0], "id": "duplicate-id"})
    with pytest.raises(ACPMappingError, match="paths must be unique"):
        verify_acp_result_manifest_files(manifest, tmp_path)
