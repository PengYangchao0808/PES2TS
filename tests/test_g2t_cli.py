"""G2.T campaign CLI contract tests (G2-AB2 WP-1, offline).

``g2t-manifest`` freezes a manifest from the config-driven policies;
``g2t-run`` without ACP wiring produces typed ``blocked`` terminals with a
needs list (never a silent skip, exit 0 = batch completed); ``g2t-status``
is read-only; ``g2t-report`` writes ``pes2ts_g2t_feedback_report_v1``.
``--help`` exits 0 for every level.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pes2ts_core.cli import main

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "p0_demo24" / "records"


def test_help_exits_zero_for_every_g2t_command(capsys):
    for argv in (["g2t-manifest"], ["g2t-run"], ["g2t-status"], ["g2t-report"]):
        with pytest.raises(SystemExit) as excinfo:
            main([*argv, "--help"])
        assert excinfo.value.code == 0
        assert "g2t" in capsys.readouterr().out


def test_manifest_run_status_report_chain_without_acp(tmp_path):
    manifest = tmp_path / "manifest.json"
    assert main(["g2t-manifest", "--snapshots", str(FIXTURES),
                 "--output", str(manifest),
                 "--pilot", "RXN_0000017762,RXN_0000079731"]) == 0
    frozen = json.loads(manifest.read_text())
    assert len(frozen["cases"]) == 24
    assert frozen["pilot_cases"] == ["RXN_0000017762", "RXN_0000079731"]
    assert frozen["frozen_parameters"]["branch_policy"]["bias_kappa"] == 0.5
    root = tmp_path / "root"
    assert main(["g2t-run", "--manifest", str(manifest), "--output-root", str(root),
                 "--case", "RXN_0000017762", "--case", "RXN_0000079731"]) == 0
    terminal = json.loads((root / "RXN_0000017762" / "terminal_state.json").read_text())
    assert terminal["terminal_class"] == "blocked"
    assert terminal["blocked_reason"] == "acp_environment_unavailable"
    assert terminal["blocked_needs"]
    import yaml
    status = tmp_path / "status.json"
    assert main(["g2t-run", "--manifest", str(manifest), "--output-root", str(root),
                 "--case", "RXN_0000017762"]) == 0  # blocked case re-attempted, still 0
    report = tmp_path / "report.json"
    assert main(["g2t-report", "--root", str(root), "--output", str(report),
                 "--manifest", str(manifest)]) == 0
    payload = json.loads(report.read_text())
    assert payload["funnel"]["four_class_denominators"]["blocked"] == 2
    assert payload["typed_failure_histogram"][0]["failure_code"] == \
        "ORIGIN_BACKEND_UNAVAILABLE"


def test_manifest_requires_snapshots_with_cases(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["g2t-manifest", "--snapshots", str(empty),
                 "--output", str(tmp_path / "m.json")]) == 24
    assert "NO_CASE_SNAPSHOTS" in capsys.readouterr().err


def test_run_refuses_edited_manifest_in_same_root(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    assert main(["g2t-manifest", "--snapshots", str(FIXTURES),
                 "--output", str(manifest)]) == 0
    root = tmp_path / "root"
    assert main(["g2t-run", "--manifest", str(manifest), "--output-root", str(root),
                 "--case", "RXN_0000017762"]) == 0
    tampered = json.loads(manifest.read_text())
    tampered["frozen_parameters"]["candidate_top_k"] = 1
    del tampered["content_sha256"]
    from pes2ts_core.generation.planning.synchronized_path import digest
    tampered["content_sha256"] = digest(tampered)
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(tampered))
    assert main(["g2t-run", "--manifest", str(edited), "--output-root", str(root)]) == 24
    assert "CAMPAIGN_ROOT_IDENTITY_MISMATCH" in capsys.readouterr().err


def test_status_refuses_non_campaign_roots(tmp_path, capsys):
    assert main(["g2t-status", str(tmp_path)]) == 24
    assert "NOT_A_CAMPAIGN_ROOT" in capsys.readouterr().err


def test_report_refuses_roots_without_terminals(tmp_path, capsys):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "campaign_identity.json").write_text("{}", encoding="utf-8")
    assert main(["g2t-report", "--root", str(root),
                 "--output", str(tmp_path / "r.json")]) == 24
    assert "NO_TERMINAL_STATES" in capsys.readouterr().err


def test_policies_absorb_budget_keys_from_config():
    """T15b/T15c: only the budget keys are absorbed; absent keys keep
    dataclass defaults (continuation 600 / local 120).  Calibration chain:
    T15 local 120->480 / continuation 600->2400 (evaluation-cap-first
    semantics: 80 local evaluations x ~2.9 s/grad ~ 232 s < 480);
    T15c continuation 2400->4000 (T15b replay extrapolation: ~9.7e-5 λ/s
    -> ~3070 s to reference lambda 0.298125; 30% margin)."""
    from pes2ts_core.cli import _g2t_policies_from_config

    policy, local_policy, _branch = _g2t_policies_from_config({
        "g2": {
            "continuation": {"max_seconds": 4000},
            "local_corrector": {"max_seconds": 480},
        }})
    assert policy.max_seconds == 4000
    assert local_policy.max_seconds == 480

    default_policy, default_local, _ = _g2t_policies_from_config({})
    assert default_policy.max_seconds == 600
    assert default_local.max_seconds == 120


def test_calibrated_defaults_keep_budget_policy_interlock():
    """T15b/T15c: config/defaults.yaml carries the calibrated budget and
    keeps the campaign/continuation single-truth interlock (campaign.py
    BUDGET_POLICY_MISMATCH: max_wall_seconds_per_path == policy.max_seconds).
    T15c: continuation.max_seconds 4000, local_corrector 480 unchanged."""
    from pes2ts_core.cli import _g2t_policies_from_config
    from pes2ts_core.config_loader import load_config

    config = load_config()
    policy, local_policy, _ = _g2t_policies_from_config(config)
    campaign = config["g2"]["campaign"]
    assert policy.max_seconds == 4000
    assert local_policy.max_seconds == 480
    assert campaign["max_wall_seconds_per_path"] == policy.max_seconds
    assert campaign["max_frames_per_path"] == policy.max_frames
    assert campaign["max_attempts_per_path"] == policy.max_attempts
