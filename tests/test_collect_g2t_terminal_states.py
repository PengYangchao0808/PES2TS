"""WP-7 collector tests: 24/24 typed terminals + complete ledgers (G2-AB2).

Locks the G2 exit-precondition collector contract: complete campaigns exit 0
with per-case ledger rows; missing terminals/ledgers are NAMED and the script
exits 3 without rerunning anything; terminals from another campaign and
changed input hashes are refused; a blocked case's empty ledger is the
auditable zero-attempt record.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.generation.campaign import (
    CampaignBackends,
    OriginBackendUnavailable,
    build_campaign_manifest,
    run_campaign,
)
from pes2ts_core.generation.planning.continuation import (
    ContinuationPolicy,
    project_geometry,
)
from pes2ts_core.generation.planning.origin_preparation import prepare_origin
from scripts.collect_g2t_terminal_states import main as collect

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "p0_demo24" / "records"
SNAPSHOTS = [FIXTURES / "RXN_0000017762.json", FIXTURES / "RXN_0000079731.json"]


def synthetic_policy() -> ContinuationPolicy:
    return ContinuationPolicy(max_frames=14, max_attempts=80, max_seconds=600.,
                              cumulative_rmsd_limit=10.0)


def identity_origin(geometry, elements, bond_indices, charge, multiplicity, case_root):
    def evaluate_free(x, evaluation_id):
        return {"success": True, "failure_class": None,
                "coordinates": np.asarray(x, float).tolist(), "energy": -1.0,
                "duration_seconds": 0.01}
    return prepare_origin(geometry, elements, bond_indices, evaluate_free,
                          method_context={"method": "synthetic"})


class SyntheticCorrector:
    def __init__(self, plan):
        self.plan = plan
        self.coordinates = plan["drivers"] + plan.get("guards", [])

    def set_branch_reference(self, geometry, frame_id):
        pass

    def on_accept(self, result):
        pass

    def __call__(self, guess, targets, attempt_id):
        x, _info = project_geometry(np.asarray(guess, float), self.coordinates,
                                    targets, self.plan["masses"], ContinuationPolicy())
        if x is None:
            return {"success": False, "converged": False, "failure_class": "OPT_FAILED",
                    "duration_seconds": 0.001, "n_gradient_evaluations": 1,
                    "acp": {"reused": False}}
        return {"success": True, "converged": True, "coordinates": x.tolist(),
                "energy": 1e-6*float(np.sum(x**2)), "duration_seconds": 0.002,
                "physical_gradient_status": "not_collected",
                "n_gradient_evaluations": 2, "acp": {"reused": False,
                                                     "wall_seconds": 0.002}}


def synthetic_backends():
    return CampaignBackends(prepare_origin=identity_origin,
                             make_corrector=lambda plan, root: SyntheticCorrector(plan))


def blocked_backends():
    def blocked_origin(geometry, elements, bond_indices, charge, multiplicity,
                       case_root):
        raise OriginBackendUnavailable(["PES2TS_ACP_ROOT must point at an ACP checkout"])
    return CampaignBackends(prepare_origin=blocked_origin,
                             make_corrector=lambda plan, root: SyntheticCorrector(plan))


def build_manifest(tmp_path: Path, name: str = "manifest.json",
                   pilot: tuple[str, ...] = ()) -> Path:
    path = tmp_path / name
    build_campaign_manifest(SNAPSHOTS, output_path=path,
                            pilot_cases=list(pilot),
                            continuation_policy=synthetic_policy(),
                            budget={"max_frames_per_path": 14,
                                    "max_attempts_per_path": 80,
                                    "max_wall_seconds_per_path": 600.0})
    return path


def test_complete_campaign_collects_all_terminals_and_ledgers(tmp_path, capsys):
    manifest = build_manifest(tmp_path)
    root = tmp_path / "root"
    run_campaign(manifest, root, backends=synthetic_backends())
    assert collect([str(root), "--manifest", str(manifest)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_terminals"] == summary["n_cases"] == 2
    assert not summary["problems"]
    assert sum(summary["four_class_denominators"].values()) == 2
    for row in summary["cases"]:
        assert isinstance(row["n_gradient"], int) and row["n_gradient"] > 0
        assert row["label_state"]


def test_missing_cases_are_named_and_never_rerun(tmp_path, capsys):
    manifest = build_manifest(tmp_path)
    root = tmp_path / "partial"
    run_campaign(manifest, root, case_filter=["RXN_0000017762"],
                 backends=synthetic_backends())
    assert collect([str(root), "--manifest", str(manifest)]) == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["n_terminals"] == 1
    assert "missing terminal_state: RXN_0000079731" in summary["problems"]
    # Nothing was rerun: the missing case still has no directory at all.
    assert not (root / "RXN_0000079731").exists()


def test_missing_ledger_is_named_even_with_a_terminal(tmp_path, capsys):
    manifest = build_manifest(tmp_path)
    root = tmp_path / "gutted"
    run_campaign(manifest, root, case_filter=["RXN_0000017762"],
                 backends=synthetic_backends())
    (root / "RXN_0000017762" / "costs.json").unlink()
    assert collect([str(root), "--manifest", str(manifest)]) == 3
    summary = json.loads(capsys.readouterr().out)
    assert "missing cost ledger: RXN_0000017762" in summary["problems"]


def test_terminal_from_another_campaign_is_refused(tmp_path, capsys):
    first = build_manifest(tmp_path, "first.json")
    second = build_manifest(tmp_path, "second.json",
                            pilot=("RXN_0000017762",))
    assert first.read_text() != second.read_text()  # pilot differs → new identity
    root = tmp_path / "root"
    run_campaign(first, root, backends=synthetic_backends())
    assert collect([str(root), "--manifest", str(second)]) == 3
    summary = json.loads(capsys.readouterr().out)
    assert any("terminal from another campaign" in problem
               for problem in summary["problems"])


def test_blocked_cases_with_empty_ledgers_are_auditable_zero_attempts(tmp_path, capsys):
    manifest = build_manifest(tmp_path)
    root = tmp_path / "blocked"
    run_campaign(manifest, root, backends=blocked_backends())
    assert collect([str(root), "--manifest", str(manifest)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["four_class_denominators"]["blocked"] == 2
    for row in summary["cases"]:
        assert row["terminal_class"] == "blocked"
        assert row["n_gradient"] == 0
