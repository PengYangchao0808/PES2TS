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


# ---------------------------------------------------------------------------
# V5.4.1 (ADR-0032) — new campaign must not disturb the old V5.4 entry
# ---------------------------------------------------------------------------
def test_v541_registered_with_required_g_support():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    validate_gv_map(gv_map)
    entry = resolve_campaign(gv_map, "V5.4.1")
    assert entry.status == "executed"
    assert entry.primary_g == "G3.P.2"
    assert set(entry.supporting_g) == {"G3.P.1", "G3.P.3", "G0", "G1"}
    assert "G3.P.2" not in entry.supporting_g
    assert set(entry.data_dependencies) == {"G0", "G1"}
    assert entry.config == "config/campaigns/pes2ts_v541_embed_50k.yaml"
    assert entry.evidence and "V541" in entry.evidence
    # executed: evidence must be the read-only final report
    assert "实施与训练报告_20261029" in entry.evidence
    assert "E0" in (entry.gate_note or "") and "E*=E0" in (entry.gate_note or "")


def test_v541_config_file_exists():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    entry = resolve_campaign(gv_map, "V5.4.1")
    assert (gv_map.path.parents[2] / entry.config).is_file()


def test_v54_untouched_by_v541_registration():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    old = resolve_campaign(gv_map, "V5.4")
    assert old.status == "proposed"
    assert old.primary_g == "G3.P.2"
    assert old.config == "config/campaigns/v5_4_rollout.yaml"
    assert "from-any-point" in (old.gate_note or "")
    # both campaigns coexist and stay distinct
    assert "V5.4" in gv_map.campaigns and "V5.4.1" in gv_map.campaigns


def test_v541_gates_are_v541_namespaced():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    entry = resolve_campaign(gv_map, "V5.4.1")
    assert all(g.startswith("V541-") for g in entry.gates)
    assert len(set(entry.gates)) == len(entry.gates)


def test_v541_plan_revision_recorded():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    # 2026-10-31 WP7 execution: registry plan_revision moved to
    # 2026-10-30.3 (V5.4.2 executed, no_backbone_winner; thresholds/budget
    # unchanged); the V5.4.1 historical note must still be present.
    assert gv_map.plan_revision == "2026-10-30.3"
    import yaml as _yaml

    doc = _yaml.safe_load(DEFAULT_MAP_PATH.read_text(encoding="utf-8"))
    note = doc.get("plan_revision_note", "")
    # V541 historical entries survive the revision (provenance unchanged)
    assert "V5.4.1" in note
    assert "ADR-0032" in note
    assert "EXECUTED" in note
    # the current revision is the V5.4.2 fix-5/fix-6 closure
    assert "V5.4.2" in note
    assert "final_selection_rule" in note
    assert "route_logic" in note
    assert "V542-A0-sanity-band" in note


def test_v542_registered_as_proposed_with_fix6_gates():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    validate_gv_map(gv_map)
    entry = resolve_campaign(gv_map, "V5.4.2")
    # WP7 execution closed the campaign: status is now ``executed`` with the
    # frozen no_backbone_winner verdict (A0 matched baseline retained).
    assert entry.status == "executed"
    assert "no_backbone_winner" in entry.title
    assert entry.primary_g == "G3.P.2"
    assert set(entry.supporting_g) == {"G3.P.1", "G3.P.3", "G0", "G1"}
    # the A0 sanity band is a canonical registry gate (fix-5 C5 / fix-6)
    assert "V542-A0-sanity-band" in entry.gates
    assert entry.gates.index("V542-A0-sanity-band") < entry.gates.index(
        "V542-H0-legality")
    assert len(set(entry.gates)) == len(entry.gates)
    # fix-6 hardening: gate_note carries the machine-frozen contracts
    gate_note = entry.gate_note or ""
    for token in (
        "final_selection_rule",
        "route_logic",
        "V542-A0-sanity-band",
        "15,400",
        "37,900",
        "main_session_owner",
        "oracle_reviewer",
    ):
        assert token in gate_note, token
    assert "status proposed" in gate_note or "status: proposed" in gate_note
    # config + schema + tests + templates are the declared code scope
    for pattern in (
        "config/campaigns/pes2ts_v542_backbone_50k.yaml",
        "scripts/agent_ops/v542_config_schema.py",
        "tests/agent_ops/test_v542_config_schema.py",
    ):
        assert pattern in entry.code_scope, pattern
    assert entry.config == "config/campaigns/pes2ts_v542_backbone_50k.yaml"
    # WP7: evidence points at the frozen execution report (V542 naming)
    assert entry.evidence.endswith(
        "PES2TS_V542_Backbone_50K_实施与训练报告_20261031.md")


def test_v541_run_record_validates_against_registry():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    entry = resolve_campaign(gv_map, "V5.4.1")
    record = {
        "gv": {
            "map_version": gv_map.map_version,
            "campaign_id": "V5.4.1",
            "primary_g": entry.primary_g,
            "supporting_g": list(entry.supporting_g),
            "gates": list(entry.gates[:2]),
            "code_commit": "0" * 40,
            "config_sha256": "a" * 64,
        }
    }
    validate_gv_run_record(record, gv_map)


def test_v541_run_record_rejects_undeclared_gate():
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    with pytest.raises(GVMapError, match="gates not declared"):
        validate_gv_run_record(
            {
                "gv": {
                    "map_version": gv_map.map_version,
                    "campaign_id": "V5.4.1",
                    "primary_g": "G3.P.2",
                    "supporting_g": ["G0"],
                    "gates": ["V53MP-S1-smoke"],
                }
            },
            gv_map,
        )


def test_v541_code_scope_paths_exist_for_executed():
    """executed status enforces code_scope existence; every glob must resolve."""
    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    entry = resolve_campaign(gv_map, "V5.4.1")
    assert entry.status == "executed"
    # default validate_gv_map (require_existing=None) enforces executed paths
    validate_gv_map(gv_map)
    root = gv_map.path.parents[2]
    assert (root / entry.config).is_file()
    assert (root / "scripts/agent_ops/v541_config_schema.py").is_file()
    assert (root / "tests/agent_ops/test_v541_config_schema.py").is_file()
    # every declared code_scope pattern must match at least one file
    for pattern in entry.code_scope:
        assert any(root.glob(pattern)), f"code_scope pattern matches nothing: {pattern}"
    # evidence + data manifest exist
    assert (root / entry.evidence).is_file()
    assert (root / entry.data_manifest).is_file()


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
