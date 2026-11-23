"""Tests for the datastore registry contract (``pes2ts_data.registry``)."""

from __future__ import annotations

import json

import pytest

from pes2ts_data.registry import (
    REGISTRY_SCHEMA_ID,
    DatastoreEntry,
    DatastoreRegistry,
    RegistryError,
    iter_artifacts,
    load_registry,
    validate_registry,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def valid_registry(**overrides) -> dict:
    doc = {
        "schema": REGISTRY_SCHEMA_ID,
        "generated_at": "2026-11-23T00:00:00+00:00",
        "git_commit": "deadbeef",
        "root": "/pes2ts_datastore",
        "status": "ACTIVE",
        "components": {
            "master_split": {
                "path": "01_master_index/master_split.parquet",
                "exists": True,
                "bytes": 123,
                "sha256": SHA_A,
                "required": True,
            },
            "frame_crosswalk": {
                "path": "01_master_index/frame_crosswalk.parquet",
                "exists": False,
                "required": True,
            },
            "core_tensors": {
                "02_core_tensors/geometry/coords_f64.npy": {
                    "bytes": 10,
                    "sha256": SHA_B,
                },
            },
            "channel_row_counts": {
                "master_split": 66195,
                "frame_crosswalk": 7342581,
                "expected": {"master_split": "64,195 + 2,000 sealed"},
            },
        },
    }
    doc.update(overrides)
    return doc


def write_registry(tmp_path, doc) -> "object":
    path = tmp_path / "datastore_registry.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_validate_registry_summary():
    report = validate_registry(valid_registry())
    assert report["schema"] == REGISTRY_SCHEMA_ID
    assert report["status"] == "ACTIVE"
    assert report["n_artifacts"] == 3
    assert report["n_present"] == 2
    # frame_crosswalk is required but absent -> reported, not silently dropped.
    assert report["n_missing_required"] == ["frame_crosswalk"]
    assert report["row_counts"] == {
        "master_split": 66195, "frame_crosswalk": 7342581}
    assert report["artifacts"]["master_split"]["sha256"] == SHA_A


def test_load_registry_model(tmp_path):
    path = write_registry(tmp_path, valid_registry())
    registry = load_registry(path)
    assert isinstance(registry, DatastoreRegistry)
    assert registry.status == "ACTIVE"
    assert registry.root == "/pes2ts_datastore"
    artifacts = registry.artifacts()
    assert "master_split" in artifacts
    assert "core_tensors.02_core_tensors/geometry/coords_f64.npy" in artifacts
    assert artifacts["master_split"].is_verified()
    assert registry.row_counts()["frame_crosswalk"] == 7342581


def test_validate_registry_accepts_model_input():
    registry = DatastoreRegistry.model_validate(valid_registry())
    assert validate_registry(registry)["status"] == "ACTIVE"


def test_registry_entry_sha256_validation():
    entry = DatastoreEntry(path="x", exists=True, sha256=SHA_A)
    assert entry.is_verified()
    with pytest.raises(Exception):
        DatastoreEntry(path="x", exists=True, sha256="not-a-sha")


def test_iter_artifacts_collects_nested_paths():
    artifacts = iter_artifacts(valid_registry()["components"])
    assert set(artifacts) == {
        "master_split",
        "frame_crosswalk",
        "core_tensors.02_core_tensors/geometry/coords_f64.npy",
    }


# ---------------------------------------------------------------------------
# Typed rejections
# ---------------------------------------------------------------------------

def test_registry_schema_and_status_are_enforced():
    with pytest.raises(RegistryError) as excinfo:
        validate_registry(valid_registry(schema="pes2ts_datastore_registry_v0"))
    assert excinfo.value.code == "REGISTRY_SCHEMA_UNSUPPORTED"

    with pytest.raises(RegistryError) as excinfo:
        validate_registry(valid_registry(status="ACTIVE-BUILD"))
    assert excinfo.value.code == "REGISTRY_BAD_STATUS"

    with pytest.raises(RegistryError) as excinfo:
        validate_registry(valid_registry(status="BROKEN"))
    assert excinfo.value.code == "REGISTRY_BAD_STATUS"

    # FROZEN is the other legal lifecycle state.
    assert validate_registry(valid_registry(status="FROZEN"))["status"] == "FROZEN"


def test_registry_rejects_malformed_sha256():
    doc = valid_registry()
    doc["components"]["master_split"]["sha256"] = "abc123"
    with pytest.raises(RegistryError) as excinfo:
        validate_registry(doc)
    assert excinfo.value.code in {"REGISTRY_BAD_SHA256", "REGISTRY_MALFORMED"}

    doc = valid_registry()
    doc["components"]["core_tensors"][
        "02_core_tensors/geometry/coords_f64.npy"]["sha256"] = "z" * 64
    with pytest.raises(RegistryError) as excinfo:
        validate_registry(doc)
    assert excinfo.value.code in {"REGISTRY_BAD_SHA256", "REGISTRY_MALFORMED"}


def test_registry_rejects_non_positive_row_counts():
    doc = valid_registry()
    doc["components"]["channel_row_counts"]["master_split"] = 0
    with pytest.raises(RegistryError) as excinfo:
        validate_registry(doc)
    assert excinfo.value.code == "REGISTRY_BAD_ROW_COUNT"

    doc = valid_registry()
    doc["components"]["channel_row_counts"]["frame_crosswalk"] = -5
    with pytest.raises(RegistryError) as excinfo:
        validate_registry(doc)
    assert excinfo.value.code == "REGISTRY_BAD_ROW_COUNT"


def test_registry_loader_typed_errors(tmp_path):
    with pytest.raises(RegistryError) as excinfo:
        load_registry(tmp_path / "missing.json")
    assert excinfo.value.code == "REGISTRY_FILE_MISSING"

    broken = tmp_path / "broken.json"
    broken.write_text("{not-json", encoding="utf-8")
    with pytest.raises(RegistryError) as excinfo:
        load_registry(broken)
    assert excinfo.value.code == "REGISTRY_PARSE_ERROR"

    with pytest.raises(RegistryError) as excinfo:
        validate_registry(["not", "a", "mapping"])
    assert excinfo.value.code == "REGISTRY_BAD_ARGUMENT"


def test_registry_requires_schema_and_status_keys(tmp_path):
    doc = valid_registry()
    doc.pop("schema")
    with pytest.raises(RegistryError) as excinfo:
        validate_registry(doc)
    assert excinfo.value.code == "REGISTRY_SCHEMA_UNSUPPORTED"

    doc = valid_registry()
    doc.pop("status")
    with pytest.raises(RegistryError) as excinfo:
        validate_registry(doc)
    assert excinfo.value.code == "REGISTRY_BAD_STATUS"
