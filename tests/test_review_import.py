from __future__ import annotations

import copy

import pytest

from pes2ts_core.contracts import ContractError, dumps_document, make_document, seal_document, validate_document
from pes2ts_core.g1.review_import import import_review_rows


def review_case():
    return make_document("ReactionCase", "case:review-1", "needs_review",
        dataset_version="test-v1", reaction_id="RXN_REVIEW_0001", case_id="case:review-1", split="train",
        atoms=[{"atom_map_id":3, "element":"C"}, {"atom_map_id":8, "element":"H"}],
        reactant={"charge":0, "multiplicity":None, "geometry":[[0,0,0],[1.1,0,0]]},
        product={"charge":0, "multiplicity":None, "geometry":[[0,0,0],[2.1,0,0]]},
        edits=[{"kind":"broken", "atom_map_ids":[3,8]}], hydrogen_transfers=[],
        source={"dataset":"test", "mapping_provenance":"endpoint_only"},
        review_reasons=["review pending"])


def reviewer_row(case, name):
    return {
        "反应ID":case["reaction_id"], "split":case["split"], "机制层":"A · H transfer",
        "复核者":name, "结论":"accept", "反应中心/机制":"confirmed", "原子映射":"confirmed",
        "电荷与自旋":"confirmed", "几何/装配":"confirmed", "扫描可行性":"1D",
        "R multiplicity":"1", "P multiplicity":"1", "备注":"checked endpoints",
    }


def adjudication_row(case, decision="accept"):
    return {
        "反应ID":case["reaction_id"], "split":case["split"], "机制层":"A · H transfer",
        "复核者1结论":"accept", "复核者2结论":"accept", "最终裁定":decision,
        "裁定者":"adjudicator-1", "R multiplicity":"1" if decision == "accept" else None,
        "P multiplicity":"1" if decision == "accept" else None, "分歧处理说明":"",
        "case_id":case["case_id"],
    }


def import_one(case, reviewer_one=None, reviewer_two=None, adjudication=None):
    return import_review_rows(
        cases={case["reaction_id"]:case},
        reviewer_one=[reviewer_one or reviewer_row(case, "reviewer-A")],
        reviewer_two=[reviewer_two or reviewer_row(case, "reviewer-B")],
        adjudications=[adjudication or adjudication_row(case)],
        workbook_sha256="a" * 64,
    )


def test_review_import_promotes_only_explicit_consensus_case_to_ready_and_audits_it():
    case = review_case()
    reviewed, records = import_one(case)
    ready = reviewed[case["reaction_id"]]
    record = records[case["reaction_id"]]
    assert ready["status"] == "ready"
    assert ready["reactant"]["multiplicity"] == ready["product"]["multiplicity"] == 1
    assert "review_reasons" not in ready
    assert record["status"] == "accepted"
    assert record["case_sha256"] == case["content_sha256"]
    assert record["workbook_sha256"] == "a" * 64
    assert ready["source"]["review_record_id"] == record["object_id"]
    assert ready["source"]["reviewed_from_case_sha256"] == record["case_sha256"]
    assert record["extensions"]["pes2ts.review_output.v1"]["reviewed_case_sha256"] == ready["content_sha256"]
    assert not validate_document(record)
    dumps_document(ready)

    forged = copy.deepcopy(record)
    forged["reviewers"][0]["dimensions"]["atom_mapping"] = "issue"
    forged = seal_document(forged)
    assert any("accepted record requires all dimensions confirmed" in issue
               for issue in validate_document(forged))


def test_review_import_rejects_blank_or_conflicting_decisions():
    case = review_case()
    blank = reviewer_row(case, "reviewer-A")
    blank["几何/装配"] = "not_reviewed"
    with pytest.raises(ContractError, match="dimensions are incomplete"):
        import_one(case, reviewer_one=blank)

    adj = adjudication_row(case)
    adj["复核者2结论"] = "reject"
    with pytest.raises(ContractError, match="does not match"):
        import_one(case, adjudication=adj)


def test_review_import_requires_explicit_spin_consensus_and_1d_for_m1_acceptance():
    case = review_case()
    reviewer = reviewer_row(case, "reviewer-A")
    reviewer["R multiplicity"] = None
    with pytest.raises(ContractError, match="must match each reviewer's explicit values"):
        import_one(case, reviewer_one=reviewer)

    reviewer = reviewer_row(case, "reviewer-A")
    reviewer["扫描可行性"] = "path_or_neb"
    with pytest.raises(ContractError, match="requires 1D feasibility"):
        import_one(case, reviewer_one=reviewer)


def test_rejected_decision_keeps_case_non_executable_and_preserves_reason():
    case = review_case()
    first = reviewer_row(case, "reviewer-A")
    second = reviewer_row(case, "reviewer-B")
    first["结论"] = second["结论"] = "reject"
    adjudication = adjudication_row(case, "reject")
    adjudication["复核者1结论"] = adjudication["复核者2结论"] = "reject"
    adjudication["分歧处理说明"] = "endpoint mapping is inconsistent"
    reviewed, records = import_one(case, first, second, adjudication)
    assert reviewed[case["reaction_id"]]["status"] == "rejected"
    assert reviewed[case["reaction_id"]]["review_reasons"] == ["endpoint mapping is inconsistent"]
    assert records[case["reaction_id"]]["status"] == "rejected"
