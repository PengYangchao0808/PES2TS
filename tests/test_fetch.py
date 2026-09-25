"""Offline and real-data tests for verified Zenodo acquisition.

The default (non-``realdata``) tests never touch the network: a local
``http.server`` on an ephemeral port serves tiny fixture payloads to the
download code under test.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import os
import threading
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, override

import pytest
import yaml

from pes2ts_core.cli import main
from pes2ts_core.config_loader import load_config
from pes2ts_core.g0.fetch import (
    EXIT_CHECKSUM_MISMATCH,
    SOURCE_MANIFEST_FILENAME,
    ChecksumMismatch,
    fetch_sources,
    source_manifest_path,
    verify_sources,
    write_source_manifest,
)

SOURCE_NAMES: tuple[str, ...] = (
    "B3LYPD3_TZVP.h5",
    "B3LYPD3_TZVP_IRC.h5",
    "B3LYPD3_TZVP_reaction_info.csv",
    "B3LYP-RXN_train.csv",
    "B3LYP-RXN_valid.csv",
    "B3LYP-RXN_test.csv",
)


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` that stays silent in the pytest log."""

    @override
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        """Suppress per-request logging."""


@pytest.fixture
def serve_dir(tmp_path: Path) -> Path:
    """Directory served by the local HTTP fixture."""
    directory = tmp_path / "served"
    directory.mkdir()
    return directory


@pytest.fixture
def base_url(serve_dir: Path) -> Iterator[str]:
    """Serve *serve_dir* on an ephemeral localhost port and yield its base URL."""
    handler = functools.partial(_QuietHandler, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _source_payload(name: str) -> bytes:
    """Return deterministic fixture bytes for source file *name*."""
    return f"pes2ts fixture payload for {name}\n".encode("utf-8") * 64


def _publish(serve_dir: Path, name: str, payload: bytes) -> str:
    """Write *payload* into *serve_dir* and return its MD5."""
    (serve_dir / name).write_bytes(payload)
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def _config(
    tmp_path: Path,
    base_url: str,
    digests: Mapping[str, str],
    *,
    wrong_md5_for: str | None = None,
) -> dict[str, Any]:
    """Build a minimal merged-config-shaped dict pointing at the local server."""
    files = {
        name: {
            "filename": name,
            "url": f"{base_url}/{name}",
            "md5": "0" * 32 if name == wrong_md5_for else md5,
        }
        for name, md5 in digests.items()
    }
    return {
        "source": {
            "zenodo_record": 18551029,
            "zenodo_doi": "10.5281/zenodo.18551029",
            "zenodo_revision": "1",
            "zenodo_modified": "2025-09-01",
            "files": files,
        },
        "paths": {
            "raw": str(tmp_path / "raw" / "reaction_qm"),
            "manifests": str(tmp_path / "manifests"),
        },
    }


def _write_user_config(tmp_path: Path, config: Mapping[str, Any]) -> Path:
    """Write *config*'s source/paths sections as a user YAML override file."""
    path = tmp_path / "user_config.yaml"
    path.write_text(
        yaml.safe_dump({"source": config["source"], "paths": config["paths"]}),
        encoding="utf-8",
    )
    return path


def test_fetch_sources_downloads_and_writes_manifest(
    tmp_path: Path, serve_dir: Path, base_url: str
) -> None:
    payload = _source_payload("sample.bin")
    md5 = _publish(serve_dir, "sample.bin", payload)
    config = _config(tmp_path, base_url, {"sample.bin": md5})
    sha256 = hashlib.sha256(payload).hexdigest()

    fetched = fetch_sources(config)
    digests = verify_sources(config)
    manifest_path = write_source_manifest(config, fetched, digests)

    target = tmp_path / "raw" / "reaction_qm" / "sample.bin"
    assert target.read_bytes() == payload
    assert digests == {"sample.bin": sha256}
    assert manifest_path == tmp_path / "manifests" / SOURCE_MANIFEST_FILENAME

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "g0_manifest_v1"
    assert manifest["dataset_version"] == "zenodo-18551029-rev1"
    assert manifest["zenodo_doi"] == "10.5281/zenodo.18551029"
    assert manifest["zenodo_revision"] == "1"
    assert manifest["zenodo_modified"] == "2025-09-01"

    assert len(manifest["files"]) == 1
    record = manifest["files"][0]
    assert record == {
        "filename": "sample.bin",
        "url": f"{base_url}/sample.bin",
        "zenodo_md5": md5,
        "sha256": sha256,
        "size_bytes": len(payload),
        "downloaded_at": record["downloaded_at"],
    }
    assert fetched["sample.bin"].downloaded_at is not None
    assert datetime.fromisoformat(record["downloaded_at"]).tzinfo is not None
    assert list((tmp_path / "raw" / "reaction_qm").glob("*.part-*")) == []


def test_fetch_sources_skips_verified_file_until_forced(
    tmp_path: Path, serve_dir: Path, base_url: str
) -> None:
    payload = _source_payload("skip.bin")
    md5 = _publish(serve_dir, "skip.bin", payload)
    config = _config(tmp_path, base_url, {"skip.bin": md5})
    target = tmp_path / "raw" / "reaction_qm" / "skip.bin"

    first = fetch_sources(config)
    assert first["skip.bin"].downloaded_at is not None

    frozen_mtime = 1_000_000_000
    os.utime(target, ns=(frozen_mtime, frozen_mtime))

    skipped = fetch_sources(config)
    assert skipped["skip.bin"].downloaded_at is None
    assert target.stat().st_mtime_ns == frozen_mtime

    forced = fetch_sources(config, force=True)
    assert forced["skip.bin"].downloaded_at is not None
    assert target.stat().st_mtime_ns > frozen_mtime
    assert target.read_bytes() == payload


def test_fetch_sources_aborts_on_md5_mismatch_without_manifest(
    tmp_path: Path, serve_dir: Path, base_url: str
) -> None:
    payload = _source_payload("bad.bin")
    actual_md5 = _publish(serve_dir, "bad.bin", payload)
    config = _config(tmp_path, base_url, {"bad.bin": actual_md5}, wrong_md5_for="bad.bin")

    with pytest.raises(ChecksumMismatch) as excinfo:
        fetch_sources(config)

    assert excinfo.value.filename == "bad.bin"
    assert excinfo.value.expected == "0" * 32
    assert excinfo.value.actual == actual_md5
    assert "bad.bin" in str(excinfo.value)
    assert not (tmp_path / "raw" / "reaction_qm" / "bad.bin").exists()
    assert not source_manifest_path(config).exists()
    assert list((tmp_path / "raw" / "reaction_qm").glob("*.part-*")) == []


def test_verify_sources_reports_every_missing_file(tmp_path: Path, base_url: str) -> None:
    config = _config(tmp_path, base_url, {"one.bin": "1" * 32, "two.bin": "2" * 32})

    with pytest.raises(ChecksumMismatch) as excinfo:
        verify_sources(config)

    assert {entry[0] for entry in excinfo.value.mismatches} == {"one.bin", "two.bin"}
    assert all(entry[2] is None for entry in excinfo.value.mismatches)
    assert "one.bin" in str(excinfo.value)
    assert "two.bin" in str(excinfo.value)
    assert "file missing" in str(excinfo.value)


def test_cli_fetch_writes_manifest_for_all_configured_files(
    tmp_path: Path, serve_dir: Path, base_url: str
) -> None:
    digests = {
        name: _publish(serve_dir, name, _source_payload(name)) for name in SOURCE_NAMES
    }
    config = _config(tmp_path, base_url, digests)
    user_config = _write_user_config(tmp_path, config)

    exit_code = main(["g0", "fetch", "--config", str(user_config)])

    assert exit_code == 0
    manifest = json.loads(source_manifest_path(config).read_text(encoding="utf-8"))
    assert [record["filename"] for record in manifest["files"]] == list(SOURCE_NAMES)
    assert all(record["downloaded_at"] is not None for record in manifest["files"])
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])
    raw_dir = tmp_path / "raw" / "reaction_qm"
    assert sorted(path.name for path in raw_dir.iterdir()) == sorted(SOURCE_NAMES)
    assert list(raw_dir.glob("*.part-*")) == []


def test_cli_fetch_exits_three_and_keeps_no_manifest_on_mismatch(
    tmp_path: Path, serve_dir: Path, base_url: str
) -> None:
    digests = {
        name: _publish(serve_dir, name, _source_payload(name)) for name in SOURCE_NAMES
    }
    config = _config(tmp_path, base_url, digests, wrong_md5_for="B3LYP-RXN_valid.csv")
    user_config = _write_user_config(tmp_path, config)

    exit_code = main(["g0", "fetch", "--config", str(user_config)])

    assert exit_code == EXIT_CHECKSUM_MISMATCH
    assert exit_code == 3
    assert not source_manifest_path(config).exists()
    assert not (tmp_path / "raw" / "reaction_qm" / "B3LYP-RXN_valid.csv").exists()
    assert list((tmp_path / "raw" / "reaction_qm").glob("*.part-*")) == []


@pytest.mark.realdata
def test_realdata_manifest_and_no_redownload() -> None:
    config = load_config()
    raw_dir = Path(config["paths"]["raw"])
    if not raw_dir.exists():
        pytest.skip(f"real Reaction-QM data not present: {raw_dir}")

    fetched = fetch_sources(config)
    digests = verify_sources(config)
    manifest_path = write_source_manifest(config, fetched, digests)

    configured = config["source"]["files"]
    assert set(fetched) == set(configured)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["files"]
    assert len(records) == 6
    for record in records:
        name = record["filename"]
        assert record["zenodo_md5"] == configured[name]["md5"]
        assert len(record["sha256"]) == 64
        assert all(char in "0123456789abcdef" for char in record["sha256"])
        assert record["size_bytes"] == (raw_dir / name).stat().st_size

    def _snapshot() -> dict[str, tuple[int, int]]:
        return {
            path.name: (path.stat().st_mtime_ns, path.stat().st_size)
            for path in raw_dir.iterdir()
            if path.is_file()
        }

    before = _snapshot()
    second = fetch_sources(config)
    assert all(entry.downloaded_at is None for entry in second.values())
    assert _snapshot() == before
