"""Tests for ``infra/governance/arch_guard.py`` -- the five hard gates of
ADR-0036 (five-zone layout, one-way imports, single training loop).

Structure:

1. repo-level gate tests: each of G1..G5 reports zero *unbaselined* findings
   against the real repository (G5 is skipped gracefully when ruff is absent,
   because ``requirements.txt`` does not pin ruff and the torch-free CI tier
   may not have it);
2. firing tests: every gate is exercised on a synthetic ``tmp_path`` tree so a
   regression in the scanner itself (not just in the repo) turns red;
3. direction tests: active zones must not import ``archive``; ``archive`` may
   import ``infra.runtime`` (the documented sink exception) but nothing else
   from the active zones.

No torch/numpy needed: the guard is stdlib + AST (+ subprocess for git/ruff).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.governance import arch_guard as ag  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_tree(root: Path, files: Dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _gate(report: Dict[str, object], gate: str) -> List[Dict[str, object]]:
    violations = report["violations"]  # type: ignore[index]
    return [finding for finding in violations if finding["gate"] == gate]


def _ruff_available() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


@pytest.fixture(scope="module")
def repo_report() -> Dict[str, object]:
    """Run all five gates once against the real repository (module scope)."""
    return ag.run_guard(REPO_ROOT)


@pytest.fixture(scope="module")
def repo_raw_g3() -> List[ag.Finding]:
    """Raw, unbaselined G3 findings on the real repository (module scope)."""
    return ag.scan_g3(REPO_ROOT)


# ---------------------------------------------------------------------------
# repo-level gates: zero unbaselined findings
# ---------------------------------------------------------------------------


def test_g1_single_training_loop(repo_report: Dict[str, object]) -> None:
    assert _gate(repo_report, "G1") == []


def test_g2_no_campaign_versioned_code(repo_report: Dict[str, object]) -> None:
    assert _gate(repo_report, "G2") == []


def test_g3_one_way_imports(repo_report: Dict[str, object]) -> None:
    assert _gate(repo_report, "G3") == []


def test_g4_frozen_zone_advisory(repo_report: Dict[str, object]) -> None:
    assert _gate(repo_report, "G4") == []
    frozen = repo_report["frozen"]  # type: ignore[index]
    assert frozen["enforced"] is False  # advisory by default


def test_g5_ruff_clean_against_baseline(repo_report: Dict[str, object]) -> None:
    ruff = repo_report["ruff"]  # type: ignore[index]
    if not ruff["available"]:  # type: ignore[index]
        pytest.skip("ruff is not installed in this interpreter (G5 skipped)")
    assert _gate(repo_report, "G5") == []


def test_scan_api_is_clean_on_a_clean_tree(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {"pes2ts_v5/training/ok.py": "def noop() -> None:\n    return None\n"},
    )
    assert ag.scan(tmp_path, ruff=False) == []


def test_scan_api_returns_gate_tagged_findings(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"pes2ts_v5/evil.py": "from archive.v4 import helper\n"})
    findings = ag.scan(tmp_path, ruff=False)
    assert findings, "scan() must report the synthetic violation"
    assert all(isinstance(finding, ag.Finding) for finding in findings)
    assert {finding.gate for finding in findings} == {"G3"}


def test_baseline_file_is_documented_and_bounded(repo_report: Dict[str, object]) -> None:
    entries = ag.load_baseline()
    assert entries, "arch_guard_baseline.json must exist and justify tolerances"
    assert len(entries) <= 40, "keep the tolerance baseline small and honest"
    for entry in entries:
        assert entry["gate"] in ("G3", "G5"), entry
        assert int(entry["count"]) >= 1, entry
        assert str(entry["reason"]).strip(), entry
    baseline = repo_report["baseline"]  # type: ignore[index]
    assert baseline["stale"] == 0  # type: ignore[index]
    assert repo_report["stale_baseline"] == []


# ---------------------------------------------------------------------------
# synthetic firing tests: each gate must actually catch a violation
# ---------------------------------------------------------------------------


def test_g1_fires_on_backward_and_step_outside_training(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/evil.py": (
                "def step(loss, optimizer):\n"
                "    loss.backward()\n"
                "    optimizer.step()\n"
            ),
            "pes2ts_v5/training/ok.py": (
                "def step(loss, optimizer):\n"
                "    loss.backward()\n"
                "    optimizer.step()\n"
            ),
            "pes2ts_v5/training/callbacks.py": (
                "# this comment mentions loss.backward() but never calls it\n"
            ),
            "pes2ts_core/pesn/frozen.py": (
                "def step(loss, optimizer):\n"
                "    loss.backward()\n"
                "    optimizer.step()\n"
            ),
        },
    )
    findings = ag.scan_g1(tmp_path)
    paths = [finding.path for finding in findings]
    assert paths.count("pes2ts_v5/evil.py") == 2, findings
    assert "pes2ts_v5/training/ok.py" not in paths
    assert "pes2ts_core/pesn/frozen.py" not in paths  # frozen legacy is exempt
    assert "pes2ts_v5/training/callbacks.py" not in paths  # AST, not comments


def test_g2_fires_on_campaign_versioned_names(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/v54_trainer.py": "",
            "pes2ts_v5/v563_trainer.py": "",
            "pes2ts_data/v6_core.py": "",
            "infra/v7_runner.py": "",
            "pes2ts_core/g1/v59_adapter.py": "",
            "pes2ts_v5/models/v5_model.py": "",
            "pes2ts_v5/runner.py": "",
            "pes2ts_core/pesn/models/v54_legacy.py": "",
        },
    )
    flagged = {finding.path for finding in ag.scan_g2(tmp_path)}
    assert flagged == {
        "pes2ts_v5/v54_trainer.py",
        "pes2ts_v5/v563_trainer.py",
        "pes2ts_data/v6_core.py",
        "infra/v7_runner.py",
        "pes2ts_core/g1/v59_adapter.py",
    }


def test_g3_fires_on_cross_zone_imports(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/evil.py": (
                "from archive.v4 import helper\n"
                "import pes2ts_core.pesn\n"
                "from tests.legacy import old\n"
            ),
            "pes2ts_data/bad.py": "import archive\n",
            "pes2ts_core/utils/bad.py": "import torch\n",
            "pes2ts_core/g1/bad.py": "from pes2ts_v5 import runner\n",
            "infra/bad.py": "import pes2ts_data\n",
            "tests/kernel/test_bad.py": "from archive.v4 import thing\n",
            "tests/legacy/test_ok.py": "from archive.v4 import thing\n",
            "pes2ts_core/pesn/ok.py": "import torch\n",
            "pes2ts_v5/good.py": "from pes2ts_core.utils.hashing import sha256_file\n",
        },
    )
    findings = ag.scan_g3(tmp_path)
    by_path: Dict[str, List[str]] = {}
    for finding in findings:
        by_path.setdefault(finding.path, []).append(finding.code)
    assert sorted(by_path["pes2ts_v5/evil.py"]) == [
        "import:archive",
        "import:pes2ts_core.pesn",
        "import:tests.legacy",
    ]
    assert by_path["pes2ts_data/bad.py"] == ["import:archive"]
    assert by_path["pes2ts_core/utils/bad.py"] == ["import:torch"]
    assert by_path["pes2ts_core/g1/bad.py"] == ["import:pes2ts_v5"]
    assert by_path["infra/bad.py"] == ["import:pes2ts_data"]
    assert by_path["tests/kernel/test_bad.py"] == ["import:archive"]
    # tests/legacy, the frozen pesn tree and intra-zone pes2ts_core imports
    # are deliberately outside the gate.
    assert "tests/legacy/test_ok.py" not in by_path
    assert "pes2ts_core/pesn/ok.py" not in by_path
    assert "pes2ts_v5/good.py" not in by_path


def test_g3_archive_is_a_frozen_sink(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "archive/v4/ok.py": (
                "from infra.runtime import sha_sidecar\n"
                "from infra.runtime.sha_sidecar import sha256_file\n"
                "from pes2ts_core.utils import hashing\n"
            ),
            "archive/v4/bad.py": (
                "from pes2ts_v5 import runner\n"
                "from infra.governance import arch_guard\n"
                "from pes2ts_data import contracts\n"
            ),
        },
    )
    findings = ag.scan_g3(tmp_path)
    by_path: Dict[str, List[str]] = {}
    for finding in findings:
        by_path.setdefault(finding.path, []).append(finding.code)
    assert "archive/v4/ok.py" not in by_path, by_path
    assert sorted(by_path["archive/v4/bad.py"]) == [
        "import:infra-non-runtime",
        "import:pes2ts_data",
        "import:pes2ts_v5",
    ]


def test_g4_fires_and_unlock_waiver_suppresses(tmp_path: Path) -> None:
    diff = [
        "archive/v4/old_script.py",
        "pes2ts_core/pesn/models/v54_model.py",
        "pes2ts_v5/runner.py",
    ]
    strict = ag.scan_g4(
        tmp_path, strict_frozen=True, diff_paths=diff, commit_message="normal commit"
    )
    assert {finding.path for finding in strict} == {
        "archive/v4/old_script.py",
        "pes2ts_core/pesn/models/v54_model.py",
    }
    # advisory by default: a dirty working tree never fails the daily gate
    assert (
        ag.scan_g4(tmp_path, diff_paths=diff, commit_message="normal commit") == []
    )
    waived = ag.scan_g4(
        tmp_path,
        strict_frozen=True,
        diff_paths=diff,
        commit_message="owner-approved legacy unlock [unlock-legacy]",
    )
    assert waived == []


def test_g5_fires_with_injected_ruff_findings(tmp_path: Path) -> None:
    def fake_ruff(root: Path) -> List[ag.Finding]:
        return [ag.Finding("pes2ts_v5/evil.py", 3, "G5", "F401 unused import", "F401")]

    findings = ag.scan_g5(tmp_path, ruff_runner=fake_ruff)
    assert [(f.path, f.line, f.gate, f.code) for f in findings] == [
        ("pes2ts_v5/evil.py", 3, "G5", "F401")
    ]


def test_g5_missing_ruff_is_skipped_not_a_violation(tmp_path: Path) -> None:
    """CI base tier may not install ruff: G5 must degrade to SKIPPED, not fail."""

    def unavailable_ruff(root: Path) -> List[ag.Finding]:
        raise ag.RuffUnavailable("ruff is not installed in this interpreter")

    report = ag.run_guard(
        tmp_path,
        use_baseline=False,
        ruff_runner=unavailable_ruff,
        diff_paths=[],
        commit_message="synthetic",
    )
    assert report["ruff"]["available"] is False
    assert any("G5 skipped" in warning for warning in report["warnings"])
    assert [finding for finding in report["violations"] if finding["gate"] == "G5"] == []


@pytest.mark.skipif(not _ruff_available(), reason="ruff is not installed")
def test_g5_real_ruff_fires_on_synthetic_tree(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/evil.py": "import os\n",
            "archive/v4/skip_me.py": "import os\n",  # not an active zone
        },
    )
    findings = ag.scan_g5(tmp_path)
    paths = {finding.path for finding in findings}
    assert "pes2ts_v5/evil.py" in paths
    assert "archive/v4/skip_me.py" not in paths


def test_guard_reports_violations_on_synthetic_tree(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/v54_trainer.py": (
                "from archive.v4 import helper\n"
                "def step(loss, optimizer):\n"
                "    loss.backward()\n"
            ),
        },
    )
    report = ag.run_guard(
        tmp_path,
        ruff=False,
        use_baseline=False,
        diff_paths=[],
        commit_message="synthetic",
    )
    assert report["clean"] is False
    gates = {finding["gate"] for finding in report["violations"]}  # type: ignore[index]
    assert {"G1", "G2", "G3"} <= gates


# ---------------------------------------------------------------------------
# direction semantics: archive is a sink, never a dependency of active code
# ---------------------------------------------------------------------------


def test_active_zones_do_not_import_archive(repo_raw_g3: List[ag.Finding]) -> None:
    offenders = [
        finding
        for finding in repo_raw_g3
        if finding.code == "import:archive" and not finding.path.startswith("infra/")
    ]
    assert offenders == [], offenders


def test_archive_only_reaches_infra_runtime(repo_raw_g3: List[ag.Finding]) -> None:
    archive_findings = [
        finding for finding in repo_raw_g3 if finding.path.startswith("archive/")
    ]
    assert archive_findings == [], archive_findings
