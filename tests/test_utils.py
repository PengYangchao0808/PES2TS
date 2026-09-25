"""Unit tests for the shared hashing, JSON, and Parquet helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pes2ts_core.utils.hashing import (
    HASH_CHUNK_SIZE,
    md5_file,
    sha256_bytes,
    sha256_file,
    stable_json_dumps,
)
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sha256_file_of_requirements_is_64_hex_chars() -> None:
    path = PROJECT_ROOT / "requirements.txt"
    digest = sha256_file(path)
    assert len(digest) == 64
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sha256_file_streams_across_chunk_boundaries(tmp_path: Path) -> None:
    payload = b"a" * (HASH_CHUNK_SIZE + 12345)
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_md5_and_sha256_bytes_match_hashlib(tmp_path: Path) -> None:
    payload = b"pes2ts known small payload"
    path = tmp_path / "small.bin"
    path.write_bytes(payload)
    assert md5_file(path) == hashlib.md5(payload, usedforsecurity=False).hexdigest()
    assert len(md5_file(path)) == 32
    assert sha256_bytes(payload) == hashlib.sha256(payload).hexdigest()


def test_stable_json_dumps_is_key_order_insensitive() -> None:
    assert stable_json_dumps({"b": 1, "a": 2}) == stable_json_dumps({"a": 2, "b": 1})
    assert stable_json_dumps({"outer": {"z": 1, "y": [{"b": 2, "a": 1}]}}) == stable_json_dumps(
        {"outer": {"y": [{"a": 1, "b": 2}], "z": 1}}
    )


def test_stable_json_dumps_keeps_unicode_literal() -> None:
    text = stable_json_dumps({"molecule": "苯", "note": "réaction"})
    assert "苯" in text
    assert "réaction" in text
    assert json.loads(text) == {"molecule": "苯", "note": "réaction"}


def test_write_json_read_json_round_trip(tmp_path: Path) -> None:
    payload = {
        "reaction_id": "RXN_0000000001",
        "nested": {"δ": [1, 2.5, None, True], "empty": {}},
        "labels": ["反应", "product"],
    }
    path = tmp_path / "nested" / "record.json"
    write_json(path, payload)
    assert read_json(path) == payload
    assert not list(path.parent.glob("*.tmp*"))


def test_write_parquet_read_parquet_sorts_columns(tmp_path: Path) -> None:
    path = tmp_path / "table.parquet"
    write_parquet(path, {"zeta": [1, 2], "alpha": ["a", "b"], "middle": [0.5, 1.5]})
    table = read_parquet(path)
    assert table.column_names == ["alpha", "middle", "zeta"]
    assert table.to_pydict() == {"alpha": ["a", "b"], "middle": [0.5, 1.5], "zeta": [1, 2]}
    assert not list(tmp_path.glob("*.tmp*"))


def test_write_parquet_into_file_parent_raises_and_leaves_no_tmp(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("regular file, not a directory", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        write_parquet(blocker / "out.parquet", {"a": [1]})
    assert blocker.is_file()
    assert list(tmp_path.glob("**/*.tmp*")) == []
