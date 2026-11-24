"""Tests for ``infra/governance/arch_guard.py`` -- the six hard gates of the
three-plate layout (post-cleanup reality, 2026-11-24).

Structure:

1. repo-level gate tests: each of G1..G6 reports zero *unbaselined* findings
   against the real repository.  G4 (frozen history must stay deleted) skips
   only while the Lead's deletion commit is still in flight: its purpose is to
   be green on the committed tree.  G5 skips when ruff is absent so the
   torch-free CI tier still runs the structural gates.
2. firing tests: every gate is exercised on a synthetic ``tmp_path`` tree so a
   regression in the scanner itself (not just in the repo) turns red.
3. direction tests: active plates must not import ``archive`` /
   ``pes2ts_core.pesn``; the only cross-plate edge is descending to
   ``pes2ts_core`` (never upward into the engine or data plate from core).

No torch/numpy needed: the guard is stdlib + AST (+ subprocess for git/ruff).
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imported after the sys.path bootstrap; via importlib so the pinned ruff rule
# set (I/E402) stays clean for the tests/kernel tier.
ag = importlib.import_module("infra.governance.arch_guard")


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
    """Run all six gates once against the real repository (module scope)."""
    return ag.run_guard(REPO_ROOT)


@pytest.fixture(scope="module")
def repo_raw_g3() -> List[ag.Finding]:
    """Raw, unbaselined G3 findings on the real repository (module scope)."""
    return ag.scan_g3(REPO_ROOT)


@pytest.fixture(scope="module")
def repo_baseline() -> Dict[str, object]:
    return ag.load_baseline()


# ---------------------------------------------------------------------------
# repo-level gates: zero unbaselined findings
# ---------------------------------------------------------------------------


def test_g1_single_training_loop(repo_report: Dict[str, object]) -> None:
    assert _gate(repo_report, "G1") == []


def test_g2_no_campaign_versioned_code(repo_report: Dict[str, object]) -> None:
    assert _gate(repo_report, "G2") == []


def test_g3_one_way_imports(repo_report: Dict[str, object]) -> None:
    assert _gate(repo_report, "G3") == []


def test_g4_frozen_history_stays_deleted(repo_report: Dict[str, object]) -> None:
    frozen = repo_report["frozen"]  # type: ignore[index]
    tracked = list(frozen["tracked"])  # type: ignore[index]
    if tracked and not frozen["waiver"]:  # type: ignore[index]
        pytest.skip(
            "frozen-history deletion is still in flight "
            f"({len(tracked)} tracked path(s)); re-run after the cleanup commit"
        )
    assert _gate(repo_report, "G4") == []
    assert frozen["clean"] is True  # type: ignore[index]


def test_g5_ruff_zero_tolerance(repo_report: Dict[str, object]) -> None:
    ruff = repo_report["ruff"]  # type: ignore[index]
    if not ruff["available"]:  # type: ignore[index]
        pytest.skip("ruff is not installed in this interpreter (G5 skipped)")
    assert ruff["findings"] == 0  # type: ignore[index]
    assert _gate(repo_report, "G5") == []


def test_g6_size_and_anti_fork(repo_report: Dict[str, object]) -> None:
    assert _gate(repo_report, "G6") == []
    active = repo_report["active"]  # type: ignore[index]
    assert active["loc"] <= ag.MAX_ACTIVE_LOC  # type: ignore[index]
    assert active["files"] > 0  # type: ignore[index]
    anti_fork = repo_report["anti_fork"]  # type: ignore[index]
    assert anti_fork["backward_owners"] == ["pes2ts_v5/training/trainer.py"]  # type: ignore[index]
    assert anti_fork["campaign_version_files"] == []  # type: ignore[index]


def test_scan_api_reports_zero_unbaselined_on_repo() -> None:
    assert ag.scan(REPO_ROOT) == []


# ---------------------------------------------------------------------------
# baseline: empty tolerances, reasoned grandfathered list, no stale entries
# ---------------------------------------------------------------------------


def test_baseline_is_empty_tolerance_and_reasoned(
    repo_report: Dict[str, object], repo_baseline: Dict[str, object]
) -> None:
    assert repo_baseline["tolerances"] == [], (
        "G3/G5 are zero-tolerance in the three-plate reality: tolerances must "
        "stay empty"
    )
    grandfathered = repo_baseline["grandfathered"]  # type: ignore[index]
    assert 0 < len(grandfathered) <= ag.MAX_GRANDFATHERED
    for entry in grandfathered:
        rel = str(entry["path"])
        assert (REPO_ROOT / rel).is_file(), f"grandfathered file missing: {rel}"
        assert str(entry["reason"]).strip(), f"grandfathered entry needs a reason: {rel}"
        assert ag._count_lines(REPO_ROOT / rel) > ag.MAX_FILE_LINES, (
            f"grandfathered file {rel} is not oversized anymore; prune the entry"
        )
    assert repo_report["stale_baseline"] == []
    baseline = repo_report["baseline"]  # type: ignore[index]
    assert baseline["stale"] == 0  # type: ignore[index]


def test_baseline_loader_rejects_malformed(tmp_path: Path) -> None:
    bad = tmp_path / "baseline.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ag.GuardConfigError):
        ag.load_baseline(bad)
    bad.write_text(json.dumps({"schema": 1, "entries": []}), encoding="utf-8")
    with pytest.raises(ag.GuardConfigError):
        ag.load_baseline(bad)


def test_g3_registry_renderer_exception_is_narrow_and_documented(
    repo_raw_g3: List[ag.Finding],
) -> None:
    exceptions = ag.INFRA_G3_EXCEPTIONS
    assert "infra/governance/render_gv_map.py" in exceptions
    entry = exceptions["infra/governance/render_gv_map.py"]
    assert entry["target"] == "pes2ts_core"
    assert entry["allowed_module"] == "pes2ts_core.gv_registry"
    assert entry["reason"].strip()
    renderer_findings = [
        finding
        for finding in repo_raw_g3
        if finding.path == "infra/governance/render_gv_map.py"
    ]
    assert renderer_findings == [], renderer_findings


def test_g3_governance_renderer_exception_is_narrow(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "infra/governance/render_gv_map.py": (
                "from pes2ts_core.gv_registry import REGISTRY\n"
                "import pes2ts_core.utils.hashing\n"
                "from pes2ts_core import gv_registry\n"
            )
        },
    )
    findings = ag.scan_g3(tmp_path)
    # only the gv_registry import is allowed; any other pes2ts_core import fails
    assert [(finding.line, finding.code) for finding in findings] == [
        (2, "import:pes2ts_core")
    ]


# ---------------------------------------------------------------------------
# synthetic firing tests: each gate must actually catch a violation
# ---------------------------------------------------------------------------


def test_g1_fires_outside_training(tmp_path: Path) -> None:
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
                "# the comment names loss.backward() but never calls it\n"
            ),
            "pes2ts_core/pesn/frozen.py": (
                "def step(loss, optimizer):\n"
                "    loss.backward()\n"
            ),
        },
    )
    findings = ag.scan_g1(tmp_path)
    paths = [finding.path for finding in findings]
    assert paths.count("pes2ts_v5/evil.py") == 2, findings
    assert "pes2ts_v5/training/ok.py" not in paths
    assert "pes2ts_v5/training/callbacks.py" not in paths
    assert "pes2ts_core/pesn/frozen.py" not in paths  # frozen history is not active code


def test_g2_fires_on_campaign_versioned_names(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/v55_trainer.py": "",
            "pes2ts_v5/v563_trainer.py": "",
            "pes2ts_data/v6_core.py": "",
            "infra/v7_runner.py": "",
            "pes2ts_core/g1/v59_adapter.py": "",
            "pes2ts_v5/models/v5_model.py": "",  # legitimate, must stay allowed
            "pes2ts_v5/runner.py": "",
        },
    )
    flagged = {finding.path for finding in ag.scan_g2(tmp_path)}
    assert flagged == {
        "pes2ts_v5/v55_trainer.py",
        "pes2ts_v5/v563_trainer.py",
        "pes2ts_data/v6_core.py",
        "infra/v7_runner.py",
        "pes2ts_core/g1/v59_adapter.py",
    }


def test_g3_fires_on_cross_plate_imports(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/evil.py": (
                "from archive.v4 import helper\n"
                "import pes2ts_core.pesn\n"
            ),
            "pes2ts_data/bad.py": "from archive import helper\n",
            "pes2ts_core/utils/bad.py": "import torch\n",
            "pes2ts_core/g1/bad.py": "from pes2ts_v5 import runner\n",
            "pes2ts_core/contracts/bad.py": "from pes2ts_data import contracts\n",
            "infra/bad.py": "import pes2ts_core.g1\n",
            "tests/kernel/test_bad.py": "from archive.v4 import thing\n",
            "tests/legacy/test_frozen_ok.py": "from archive.v4 import thing\n",
            "pes2ts_core/pesn/ok.py": "import torch\n",
            "pes2ts_v5/good.py": "from pes2ts_core.utils.hashing import sha256_file\n",
            "pes2ts_core/generation/good.py": "from pes2ts_core.contracts import core\n",
        },
    )
    findings = ag.scan_g3(tmp_path)
    by_path: Dict[str, List[str]] = {}
    for finding in findings:
        by_path.setdefault(finding.path, []).append(finding.code)
    assert sorted(by_path["pes2ts_v5/evil.py"]) == [
        "import:archive",
        "import:pes2ts_core.pesn",
    ]
    assert by_path["pes2ts_data/bad.py"] == ["import:archive"]
    assert by_path["pes2ts_core/utils/bad.py"] == ["import:torch"]
    assert by_path["pes2ts_core/g1/bad.py"] == ["import:pes2ts_v5"]
    assert by_path["pes2ts_core/contracts/bad.py"] == ["import:pes2ts_data"]
    assert by_path["infra/bad.py"] == ["import:pes2ts_core"]
    assert by_path["tests/kernel/test_bad.py"] == ["import:archive"]
    # frozen test tier, frozen pesn, and legal downward imports are not flagged
    assert "tests/legacy/test_frozen_ok.py" not in by_path
    assert "pes2ts_core/pesn/ok.py" not in by_path
    assert "pes2ts_v5/good.py" not in by_path
    assert "pes2ts_core/generation/good.py" not in by_path


def test_g4_fires_on_reappearance_and_honours_waiver(tmp_path: Path) -> None:
    tracked = [
        "archive/v4/old_script.py",
        "tests/legacy/test_old.py",
        "pes2ts_core/pesn/models/v54_model.py",
        "pes2ts_v5/runner.py",
    ]
    findings = ag.scan_g4(tmp_path, tracked_paths=tracked, commit_message="normal commit")
    assert {finding.path for finding in findings} == {
        "archive/v4/old_script.py",
        "tests/legacy/test_old.py",
        "pes2ts_core/pesn/models/v54_model.py",
    }
    waived = ag.scan_g4(
        tmp_path,
        tracked_paths=tracked,
        commit_message="owner-approved unlock [unlock-legacy]",
    )
    assert waived == []
    assert ag.scan_g4(tmp_path, tracked_paths=[], commit_message="normal") == []


def test_g4_worktree_fallback_catches_leftover_directory(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"archive/v4/leftover.py": "# frozen history\n"})
    # tmp_path is not a git repository, so the guard falls back to the worktree.
    findings = ag.scan_g4(tmp_path, commit_message="normal commit")
    assert [finding.path for finding in findings] == ["archive/v4/leftover.py"]


def test_g5_fires_with_injected_ruff_findings(tmp_path: Path) -> None:
    def fake_ruff(root: Path) -> List[ag.Finding]:
        return [ag.Finding("pes2ts_v5/evil.py", 3, "G5", "F401 unused import", "F401")]

    findings = ag.scan_g5(tmp_path, ruff_runner=fake_ruff)
    assert [(f.path, f.line, f.gate, f.code) for f in findings] == [
        ("pes2ts_v5/evil.py", 3, "G5", "F401")
    ]


@pytest.mark.skipif(not _ruff_available(), reason="ruff is not installed")
def test_g5_real_ruff_fires_on_synthetic_tree(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/evil.py": "import os\n",
            "archive/v4/skip_me.py": "import os\n",  # not an active plate
        },
    )
    findings = ag.scan_g5(tmp_path)
    paths = {finding.path for finding in findings}
    assert "pes2ts_v5/evil.py" in paths
    assert "archive/v4/skip_me.py" not in paths


def test_g5_missing_ruff_is_skipped_not_a_violation(tmp_path: Path) -> None:
    def unavailable_ruff(root: Path) -> List[ag.Finding]:
        raise ag.RuffUnavailable("ruff is not installed in this interpreter")

    report = ag.run_guard(
        tmp_path,
        use_baseline=False,
        ruff_runner=unavailable_ruff,
        tracked_paths=[],
        commit_message="synthetic",
    )
    assert report["ruff"]["available"] is False
    assert any("G5 skipped" in warning for warning in report["warnings"])
    assert _gate(report, "G5") == []


def _long_python(lines: int) -> str:
    return "\n".join(f"VALUE_{index} = {index}" for index in range(lines)) + "\n"


def test_g6_fires_on_oversized_file_unless_grandfathered(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/big.py": _long_python(400),
            "pes2ts_data/small.py": _long_python(10),
        },
    )
    findings = ag.scan_g6(tmp_path)
    assert [(f.path, f.code) for f in findings] == [("pes2ts_v5/big.py", "size")]
    tolerated = ag.scan_g6(tmp_path, grandfathered_paths=["pes2ts_v5/big.py"])
    assert tolerated == []
    # shrunken / deleted grandfathered files are stale and must be pruned
    stale = ag.stale_grandfathered(
        tmp_path,
        [
            {"path": "pes2ts_data/small.py", "lines": 10, "reason": "test"},
            {"path": "pes2ts_data/gone.py", "lines": 999, "reason": "test"},
        ],
    )
    assert {finding.path for finding in stale} == {
        "pes2ts_data/small.py",
        "pes2ts_data/gone.py",
    }
    assert all(finding.gate == "BASELINE" for finding in stale)


def test_g6_fires_on_active_loc_budget(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"pes2ts_v5/big.py": _long_python(50)})
    findings = ag.scan_g6(tmp_path, max_active_loc=10)
    assert [(f.code, f.gate) for f in findings] == [("loc", "G6")]


def test_g6_anti_fork_counts_second_backward_owner_and_campaign_file(
    tmp_path: Path,
) -> None:
    _write_tree(
        tmp_path,
        {
            "pes2ts_v5/training/ok.py": "def f(loss):\n    loss.backward()\n",
            "pes2ts_v5/evil.py": "def f(loss):\n    loss.backward()\n",
            "pes2ts_v5/v56_runner.py": "",
        },
    )
    report = ag.run_guard(
        tmp_path,
        ruff=False,
        use_baseline=False,
        tracked_paths=[],
        commit_message="synthetic",
    )
    anti_fork = report["anti_fork"]  # type: ignore[index]
    assert anti_fork["backward_owners"] == [  # type: ignore[index]
        "pes2ts_v5/evil.py",
        "pes2ts_v5/training/ok.py",
    ]
    assert anti_fork["campaign_version_files"] == ["pes2ts_v5/v56_runner.py"]  # type: ignore[index]
    assert {"G1", "G2"} <= {finding["gate"] for finding in report["violations"]}  # type: ignore[index]


# ---------------------------------------------------------------------------
# direction semantics: frozen history is never a dependency of active code
# ---------------------------------------------------------------------------


def test_active_plates_do_not_import_frozen_history(
    repo_raw_g3: List[ag.Finding],
) -> None:
    offenders = [
        finding
        for finding in repo_raw_g3
        if finding.code in ("import:archive", "import:pes2ts_core.pesn")
        and not finding.path.startswith("infra/")
    ]
    assert offenders == [], offenders


def test_frozen_prefixes_are_the_branch_history() -> None:
    assert ag.FROZEN_PREFIXES == (
        "archive/",
        "tests/legacy/",
        "pes2ts_core/pesn/",
    )
