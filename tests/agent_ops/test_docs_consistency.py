"""Tests for the ADR-0030 documentation consistency gate (docs_consistency).

Unit tests build minimal tmp repo trees (config JSON + markdown files) and
exercise the three rule sets:

1. forbidden token in an active doc (no exemption marker) -> violation;
2. same token on a line carrying an exemption marker -> clean;
3. historical file without / with the ``ADR-0030`` banner marker;
4. required file missing a required substring.

The final repo-level test runs the gate against the actual repository and is
enforced (exit-asserting) since the ADR-0030 banner lanes merged on
2026-10-21.  Dry-run scope: stdlib + pytest only, no torch/numpy, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import docs_consistency as dc  # noqa: E402


BASE_CONFIG: Dict[str, object] = {
    "forbidden_tokens": [
        "TSStop",
        "TS Stop",
        "TS-neighborhood",
        "approach-side",
        "TS-directed",
        "No-Move candidate",
    ],
    "exemption_markers": [
        "弃用",
        "撤回",
        "Superseded",
        "superseded",
        "取代",
        "禁止",
        "HISTORICAL",
        "Historical",
        "historical",
        "LEGACY",
        "POST-HOC",
        "SUPERSEDED",
    ],
    "historical": [],
    "root_scope": ["README.md"],
    "required_files": {},
    "historical_banner_marker": "ADR-0030",
}


def _write_tree(
    root: Path,
    files: Dict[str, str],
    config: Optional[Dict[str, object]] = None,
) -> Path:
    """Materialize ``{relpath: text}`` under *root*; return the config path."""
    cfg = dict(BASE_CONFIG if config is None else config)
    config_path = root / "docs_consistency_config.json"
    config_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return config_path


def _run(root: Path, config_path: Path):
    return dc.run_checks(root, config_path=config_path)


# ---------------------------------------------------------------------------
# forbidden-token rule
# ---------------------------------------------------------------------------


def test_active_doc_forbidden_token_is_reported(tmp_path: Path) -> None:
    config_path = _write_tree(
        tmp_path,
        {
            "docs/active.md": (
                "# Active\n\nThe stop criterion is TSStop near the saddle.\n"
            ),
        },
    )
    report = _run(tmp_path, config_path)
    assert not report["clean"]
    assert report["violations"] == [
        {
            "category": "forbidden_token",
            "file": "docs/active.md",
            "line": 3,
            "token": "TSStop",
        }
    ]
    assert report["n_files_scanned"] == 1


def test_exemption_marker_on_same_line_suppresses_violation(tmp_path: Path) -> None:
    config_path = _write_tree(
        tmp_path,
        {
            "docs/active.md": (
                "# Active\n\n（弃用）TSStop 是旧语义；现行语义是 Endpoint Stop。\n"
            ),
        },
    )
    report = _run(tmp_path, config_path)
    assert report["clean"]
    assert report["violations"] == []


def test_archive_docs_are_exempt_by_prefix(tmp_path: Path) -> None:
    config_path = _write_tree(
        tmp_path,
        {"docs/archive/old/notes.md": "legacy text: TSStop everywhere\n"},
    )
    report = _run(tmp_path, config_path)
    assert report["clean"]
    assert report["violations"] == []


# ---------------------------------------------------------------------------
# historical banner rule
# ---------------------------------------------------------------------------


def test_historical_file_without_banner_is_violation(tmp_path: Path) -> None:
    config_path = _write_tree(
        tmp_path,
        {"docs/old_contract.md": "# Old contract\n\nTSStop semantics.\n"},
        config={**BASE_CONFIG, "historical": ["docs/old_contract.md"]},
    )
    report = _run(tmp_path, config_path)
    assert not report["clean"]
    assert report["violations"] == [
        {
            "category": "historical_missing_banner",
            "file": "docs/old_contract.md",
            "marker": "ADR-0030",
        }
    ]
    # historical files are out of the forbidden-token scan scope
    assert all(v["category"] != "forbidden_token" for v in report["violations"])


def test_historical_file_with_banner_is_clean(tmp_path: Path) -> None:
    config_path = _write_tree(
        tmp_path,
        {
            "docs/old_contract.md": (
                "# Old contract\n\nSUPERSEDED by ADR-0030 (banner).\n"
                "Old semantics: TSStop.\n"
            )
        },
        config={**BASE_CONFIG, "historical": ["docs/old_contract.md"]},
    )
    report = _run(tmp_path, config_path)
    assert report["clean"]
    assert report["violations"] == []


def test_missing_historical_file_is_violation(tmp_path: Path) -> None:
    config_path = _write_tree(
        tmp_path,
        {"docs/keep.md": "harmless\n"},
        config={**BASE_CONFIG, "historical": ["docs/never_written.md"]},
    )
    report = _run(tmp_path, config_path)
    assert report["violations"] == [
        {"category": "historical_missing_file", "file": "docs/never_written.md"}
    ]


# ---------------------------------------------------------------------------
# required-files rule
# ---------------------------------------------------------------------------


def test_required_file_missing_substring_is_violation(tmp_path: Path) -> None:
    config_path = _write_tree(
        tmp_path,
        {"docs/contract.md": "# Contract\n\nEndpoint Stop at d_end 0.10.\n"},
        config={
            **BASE_CONFIG,
            "required_files": {"docs/contract.md": ["Endpoint Stop", "0.30"]},
        },
    )
    report = _run(tmp_path, config_path)
    assert report["violations"] == [
        {
            "category": "required_substring_missing",
            "file": "docs/contract.md",
            "substring": "0.30",
        }
    ]


def test_required_file_missing_entirely_is_violation(tmp_path: Path) -> None:
    config_path = _write_tree(
        tmp_path,
        {"docs/keep.md": "harmless\n"},
        config={
            **BASE_CONFIG,
            "required_files": {"docs/adr0030.md": ["Endpoint Stop"]},
        },
    )
    report = _run(tmp_path, config_path)
    assert report["violations"] == [
        {"category": "required_missing_file", "file": "docs/adr0030.md"}
    ]


# ---------------------------------------------------------------------------
# repo-level gate (enforced since the ADR-0030 banner lanes merged 2026-10-21)
# ---------------------------------------------------------------------------


def test_token_exempt_prefixes_skip_operational_logs(tmp_path: Path) -> None:
    """token_exempt_prefixes skip the token scan without requiring banners."""
    files = {
        "docs/agent_runs/PARALLEL_LOG.md": "# log\n\n旧 TSStop 标签记录。\n",
        "docs/plans/ACTIVE.md": "# plan\n\n正文。\n",
    }
    config = dict(BASE_CONFIG)
    config["token_exempt_prefixes"] = ["docs/agent_runs/"]
    config_path = _write_tree(tmp_path, files, config)
    report = dc.run_checks(tmp_path, config_path=config_path)
    tokens = [v for v in report["violations"] if v["category"] == "forbidden_token"]
    assert tokens == [], tokens


def test_repo_docs_consistency() -> None:
    report = dc.run_checks(REPO_ROOT)
    assert report["violations"] == [], report["violations"]
