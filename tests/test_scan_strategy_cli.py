"""CLI contract tests for ``g1 v2-scan-plan|verify|freeze`` (plan task 4).

Covers the three additive g1 subcommands wired in :mod:`pes2ts_core.cli`:
``--help`` exits 0 for each; ``g1 --help`` lists them alongside the existing
v2-* commands; an empty-input ``v2-scan-plan`` run writes the proposals-tree
skeleton plus manifest deterministically (hash-stable across two runs after
removing the volatile ``generated_at``); ``v2-scan-verify`` on that skeleton
exits 0 with ``problems=0`` and a dirty tree exits 22; ``v2-scan-freeze``
refuses with exit 22, writes ``plan_gate_pass=false`` with the typed reason
``NO_ELIGIBLE_PROPOSALS``, and creates no consumable files under the plans
tree.  All fixtures are synthetic and written under ``tmp_path`` roots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pes2ts_core.cli import EXIT_G1_BUILD_FAILED, main
from pes2ts_core.config_loader import load_config
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json


def _config_path(tmp_path: Path) -> Path:
    """Re-root every configured path under ``tmp_path`` and dump the YAML."""
    config = load_config()
    config["paths"]["interim"] = str(tmp_path / "interim")
    config["paths"]["manifests"] = str(tmp_path / "manifests")
    path = tmp_path / "fixture.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _cli(config_path: Path, *argv: str) -> int:
    return main(["--config", str(config_path), *argv])


def _manifest_without_volatile(path: Path) -> dict[str, Any]:
    document = dict(read_json(path))
    document.pop("generated_at", None)
    return document


@pytest.mark.parametrize("command", ["v2-scan-plan", "v2-scan-verify", "v2-scan-freeze"])
def test_help_exits_zero(command: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["g1", command, "--help"])
    assert excinfo.value.code == 0


def test_g1_help_lists_the_three_scan_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["g1", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "v2-scan-plan" in out
    assert "v2-scan-verify" in out
    assert "v2-scan-freeze" in out
    assert "v2-gate" in out
    assert "v2-sanitize-exports" in out


def test_empty_plan_run_writes_skeleton_and_is_deterministic(tmp_path: Path) -> None:
    config_path = _config_path(tmp_path)
    interim = tmp_path / "interim"
    manifests = tmp_path / "manifests"
    proposals_dir = interim / "g1_v2" / "scan_proposals"
    proposal_manifest = manifests / "g1_v2_scan_proposal_manifest.json"
    proposal_summary = interim / "g1_v2_scan_proposal_summary.parquet"

    assert _cli(config_path, "g1", "v2-scan-plan") == 0
    assert proposals_dir.is_dir()
    assert proposal_manifest.is_file()
    assert proposal_summary.is_file()
    first_manifest = _manifest_without_volatile(proposal_manifest)
    assert first_manifest["n_total"] == 0
    assert first_manifest["n_files"] == 0
    first_summary_sha = sha256_file(proposal_summary)
    assert list(proposals_dir.iterdir()) == []

    assert _cli(config_path, "g1", "v2-scan-plan") == 0
    second_manifest = _manifest_without_volatile(proposal_manifest)
    assert second_manifest == first_manifest
    assert sha256_file(proposal_summary) == first_summary_sha


def test_verify_on_skeleton_exits_zero_with_no_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _config_path(tmp_path)
    assert _cli(config_path, "g1", "v2-scan-plan") == 0
    capsys.readouterr()
    assert _cli(config_path, "g1", "v2-scan-verify") == 0
    out = capsys.readouterr().out
    assert "problems=0" in out
    assert "total=0" in out
    assert "clean=0" in out


def test_verify_on_dirty_tree_exits_22_and_lists_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _config_path(tmp_path)
    assert _cli(config_path, "g1", "v2-scan-plan") == 0
    proposals_dir = tmp_path / "interim" / "g1_v2" / "scan_proposals" / "00000"
    proposals_dir.mkdir(parents=True)
    (proposals_dir / "RXN_0000000001.json").write_text(
        json.dumps({"schema_name": "StrategyProposal", "status": "proposed"}),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert _cli(config_path, "g1", "v2-scan-verify") == EXIT_G1_BUILD_FAILED
    out = capsys.readouterr().out
    # Invalid document + manifest n_total/n_files mismatches.
    assert "problems=3" in out


def test_freeze_refuses_without_eligible_proposals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _config_path(tmp_path)
    assert _cli(config_path, "g1", "v2-scan-plan") == 0
    manifests = tmp_path / "manifests"
    plans_dir = tmp_path / "interim" / "g1_v2" / "scan_plans"
    freeze_manifest = manifests / "g1_v2_scan_freeze.json"
    plan_manifest = manifests / "g1_v2_scan_plan_manifest.json"
    capsys.readouterr()

    assert _cli(config_path, "g1", "v2-scan-freeze") == EXIT_G1_BUILD_FAILED
    out = capsys.readouterr().out
    assert "plan_gate_pass=false" in out

    manifest = read_json(freeze_manifest)
    assert manifest["plan_gate_pass"] is False
    assert "NO_ELIGIBLE_PROPOSALS" in manifest["reasons"]
    assert manifest["n_plans"] == 0
    assert manifest["n_proposals"] == 0

    assert plans_dir.is_dir()
    assert list(plans_dir.rglob("*.json")) == []
    assert not plan_manifest.exists()


def test_freeze_refuses_when_verification_is_dirty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _config_path(tmp_path)
    assert _cli(config_path, "g1", "v2-scan-plan") == 0
    proposals_dir = tmp_path / "interim" / "g1_v2" / "scan_proposals" / "00000"
    proposals_dir.mkdir(parents=True)
    (proposals_dir / "RXN_0000000001.json").write_text("{not json", encoding="utf-8")
    capsys.readouterr()

    assert _cli(config_path, "g1", "v2-scan-freeze") == EXIT_G1_BUILD_FAILED
    manifest = read_json(tmp_path / "manifests" / "g1_v2_scan_freeze.json")
    assert manifest["plan_gate_pass"] is False
    assert "VERIFICATION_FAILED" in manifest["reasons"]
    assert "NO_ELIGIBLE_PROPOSALS" in manifest["reasons"]
