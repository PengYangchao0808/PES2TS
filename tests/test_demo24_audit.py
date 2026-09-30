"""Regression checks for the endpoint-only 24-reaction demo audit."""

from scripts.audit_demo24_endpoints import audit_demo24


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
    rows = {record["reaction_id"]: record for record in report["records"]}

    record = rows["RXN_0000079731"]
    assert record["mapping_status"] == "resolved_symmetry_collapsed"
    assert record["checks"]["mapping_status_resolved"] is True
    assert record["human_review_status"] == "pending"
