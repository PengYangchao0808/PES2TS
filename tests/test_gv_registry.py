"""G–V campaign registry: structural and path-existence contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pes2ts_core.gv_registry import (
    DEFAULT_MAP_PATH,
    GVMapError,
    load_gv_map,
    resolve_campaign,
    validate_gv_map,
    validate_gv_run_record,
)


def _write_map(tmp_path: Path, *, status: str = "proposed") -> Path:
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    document = {
        "schema": "pes2ts_gv_map_v1",
        "map_version": "test.1",
        "stages": {
            "G0": {
                "title": "data",
                "stage_doc": "docs/plans/PES2TS_阶段定义_G0-G3.md",
                "code_globs": ["code.py"],
            },
            "G3.P.2": {
                "title": "models",
                "stage_doc": "docs/plans/PES2TS_阶段定义_G0-G3.md",
                "code_globs": ["code.py"],
            },
        },
        "campaigns": {
            "V5.1": {
                "title": "test campaign",
                "status": status,
                "primary_g": "G3.P.2",
                "supporting_g": ["G0"],
                "data_dependencies": ["G0"],
                "code_scope": ["code.py", "missing_planned.py"],
                "config": "missing_config.yaml",
                "data_manifest": "missing_manifest.json",
                "gates": ["V5-G1"],
                "evidence": "missing_evidence.md",
                "comparison_only": [],
            }
        },
    }
    path = tmp_path / "gv.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_default_map_loads_and_validates():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    validate_gv_map(gv_map)
    assert gv_map.map_version == "2026-10-18.1"
    assert resolve_campaign(gv_map, "V5.1").primary_g == "G3.P.2"
    assert resolve_campaign(gv_map, "V7-C").primary_g == "G4.0"
    assert "V4.7" in gv_map.campaigns


def test_proposed_campaign_may_reference_planned_paths(tmp_path):
    path = _write_map(tmp_path, status="proposed")
    gv_map = load_gv_map(path)
    # proposed: missing code/config/data/evidence are allowed
    validate_gv_map(gv_map, repo_root=tmp_path, require_existing=None)


def test_active_campaign_requires_paths(tmp_path):
    path = _write_map(tmp_path, status="active")
    gv_map = load_gv_map(path)
    with pytest.raises(GVMapError, match="missing_planned.py"):
        validate_gv_map(gv_map, repo_root=tmp_path)


def test_primary_must_be_a_stage(tmp_path):
    path = _write_map(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["campaigns"]["V5.1"]["primary_g"] = "G9"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    gv_map = load_gv_map(path)
    with pytest.raises(GVMapError, match="not a stage"):
        validate_gv_map(gv_map, repo_root=tmp_path)


def test_supporting_must_not_repeat_primary(tmp_path):
    path = _write_map(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["campaigns"]["V5.1"]["supporting_g"] = ["G0", "G3.P.2"]
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    gv_map = load_gv_map(path)
    with pytest.raises(GVMapError, match="must not repeat primary_g"):
        validate_gv_map(gv_map, repo_root=tmp_path)


def test_unknown_campaign_is_typed():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    with pytest.raises(GVMapError, match="unknown V campaign"):
        resolve_campaign(gv_map, "V9.9")


def _record(**overrides):
    base = {
        "gv": {
            "map_version": "2026-10-18.1",
            "campaign_id": "V5.1",
            "primary_g": "G3.P.2",
            "supporting_g": ["G0", "G1"],
            "gates": ["V5-G1", "V5-G2"],
            "code_commit": "0" * 40,
            "config_sha256": "a" * 64,
            "data_manifest_sha256": "b" * 64,
            "map_sha256": "c" * 64,
        }
    }
    base["gv"].update(overrides)
    return base


def test_run_record_accepts_declared_campaign():
    validate_gv_run_record(_record(), load_gv_map(DEFAULT_MAP_PATH))


def test_run_record_rejects_wrong_primary():
    with pytest.raises(GVMapError, match="primary_g"):
        validate_gv_run_record(
            _record(primary_g="G4.1"), load_gv_map(DEFAULT_MAP_PATH)
        )


def test_run_record_rejects_undeclared_gate():
    with pytest.raises(GVMapError, match="gates not declared"):
        validate_gv_run_record(
            _record(gates=["V7-RL"]), load_gv_map(DEFAULT_MAP_PATH)
        )


def test_run_record_rejects_bad_digest():
    with pytest.raises(GVMapError, match="64-character"):
        validate_gv_run_record(
            _record(config_sha256="not-a-digest"), load_gv_map(DEFAULT_MAP_PATH)
        )


def test_run_record_requires_gv_block():
    with pytest.raises(GVMapError, match="missing a 'gv'"):
        validate_gv_run_record({}, load_gv_map(DEFAULT_MAP_PATH))
