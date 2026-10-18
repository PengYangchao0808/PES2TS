"""Regression checks for the endpoint-only 24-reaction demo audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_demo24_endpoints import audit_demo24

_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "reaction_qm"
pytestmark = pytest.mark.skipif(
    not (_RAW / "B3LYP-RXN_train.csv").is_file()
    or not (_RAW / "B3LYP-RXN_valid.csv").is_file(),
    reason="raw Reaction-QM split CSV is local-only data and is not in a fresh clone",
)


def test_demo24_endpoint_audit_is_consistent_and_still_pending_review():
    report = audit_demo24()

    assert report["audit_status"] == "consistent_pending_human_review"
    assert report["n_candidates"] == 24
    assert report["n_unique_reaction_ids"] == 24
    assert report["split_counts"] == {"train": 16, "valid": 8}
    assert report["review_status_counts"] == {"pending": 24}
    assert report["n_issue_rows"] == 0
    assert report["collection_issues"] == []
    assert set(report["checks_passed_for_all_candidates"].values()) == {24}


def test_symmetry_collapsed_map_is_resolved_but_visible_to_reviewer():
    report = audit_demo24()
    assert report["audit_status"] == "consistent_pending_human_review"
