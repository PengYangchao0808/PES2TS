from __future__ import annotations

from pes2ts_core.contracts import loads_document, validate_document
from scripts.build_demo24_reaction_cases import DEFAULT_OUTPUT, MANIFEST_PATH, build_review_cases
from scripts.audit_demo24_spin_sources import build_spin_audit


def test_demo24_cases_are_standardized_with_source_backed_spin_but_still_need_review():
    manifest = build_review_cases()

    assert manifest["n_cases"] == 24
    assert manifest["split_counts"] == {"train": 16, "valid": 8}
    assert manifest["case_status_counts"] == {"needs_review": 24}
    assert manifest["status"] == "awaiting_manual_chemistry_review"
    assert MANIFEST_PATH.is_file()
    assert (DEFAULT_OUTPUT / "README.md").is_file()

    for record in manifest["records"]:
        case = loads_document((DEFAULT_OUTPUT / record["relative_path"]).read_text(encoding="utf-8"))
        assert case["reaction_id"] == record["reaction_id"]
        assert case["status"] == "needs_review"
        assert case["reactant"]["multiplicity"] == record["multiplicities"]["reactant"]
        assert case["product"]["multiplicity"] == record["multiplicities"]["product"]
        assert case["source"]["spin_provenance"]["reactant"]["status"] == record["spin_resolution"]["reactant"]
        assert case["source"]["spin_provenance"]["product"]["status"] == record["spin_resolution"]["product"]
        assert case["source"]["g1_export_sha256"] == record["source_export_sha256"]
        assert len(case["source"]["g1_payload_sha256"]) == 64
        assert case["case_id"] == record["case_id"]
        assert not validate_document(case, production_input=True)
        assert "TS/IRC" in manifest["information_boundary"]


def test_single_component_spin_is_traceable_and_never_uses_multiplicity_max():
    report = build_spin_audit()
    assert report["n_reactions"] == 24
    record = next(item for item in report["records"] if item["reaction_id"] == "RXN_0000007104")
    assert record["endpoint_spin"]["reactant"]["multiplicity"] == 1
    assert record["endpoint_spin"]["product"]["multiplicity"] == 1
    assert record["endpoint_spin"]["reactant"]["components"][0]["component_id"] == "R0"
    assert "multiplicity_max" not in record["endpoint_spin"]["reactant"]
