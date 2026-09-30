"""Import completed Demo24 chemistry-review workbooks without guessing decisions."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import posixpath
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

from pes2ts_core.contracts import ContractError, dumps_document, make_document, seal_document

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
REVIEW_SHEETS = ("复核者1", "复核者2")
ADJUDICATION_SHEET = "汇总裁定"
REVIEW_COLUMNS = {
    "reaction_id": "反应ID", "split": "split", "reviewer": "复核者", "decision": "结论",
    "reaction_center": "反应中心/机制", "atom_mapping": "原子映射", "charge_spin": "电荷与自旋",
    "geometry_assembly": "几何/装配", "scan_feasibility": "扫描可行性",
    "reactant_multiplicity": "R multiplicity", "product_multiplicity": "P multiplicity", "note": "备注",
}
ADJ_COLUMNS = {
    "reaction_id": "反应ID", "split": "split", "reviewer1_decision": "复核者1结论",
    "reviewer2_decision": "复核者2结论", "decision": "最终裁定", "adjudicator": "裁定者",
    "reactant_multiplicity": "R multiplicity", "product_multiplicity": "P multiplicity",
    "resolution_note": "分歧处理说明", "case_id": "case_id",
}
DIMENSIONS = ("reaction_center", "atom_mapping", "charge_spin", "geometry_assembly")
FINAL_DECISIONS = {"accept", "reject", "replace", "needs_more_info"}
REVIEWER_DECISIONS = FINAL_DECISIONS | {"pending"}
REVIEW_DIMENSIONS = {"not_reviewed", "confirmed", "issue"}
SCAN_FEASIBILITY = {"pending", "1D", "synchronized", "path_or_neb", "reject"}


def _column_number(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if not letters:
        raise ContractError(f"invalid XLSX cell reference: {cell_ref!r}")
    value = 0
    for char in letters.group(0):
        value = value * 26 + ord(char) - 64
    return value


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    if cell.find(f"{{{MAIN_NS}}}f") is not None:
        raise ContractError("review workbook must contain fixed values, not formulas")
    kind = cell.get("t")
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(f".//{{{MAIN_NS}}}t"))
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    if value_node is None or value_node.text is None:
        return None
    value = value_node.text
    if kind == "s":
        try:
            return shared_strings[int(value)]
        except (ValueError, IndexError) as exc:
            raise ContractError("invalid shared string reference in review workbook") from exc
    if kind == "b":
        return value == "1"
    return value


def _load_sheet_rows(book: ZipFile, target: str, shared_strings: list[str]) -> list[dict[int, Any]]:
    target = target.lstrip("/")
    normalized = posixpath.normpath(target if target.startswith("xl/") else posixpath.join("xl", target))
    if normalized not in book.namelist():
        raise ContractError(f"review workbook sheet is missing: {normalized}")
    root = ET.fromstring(book.read(normalized))
    rows: list[dict[int, Any]] = []
    for row in root.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"):
        values: dict[int, Any] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            values[_column_number(cell.get("r", ""))] = _cell_value(cell, shared_strings)
        rows.append(values)
    return rows


def read_review_workbook(path: str | Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Read the three fixed review sheets from the artifact-tool workbook format.

    This small OOXML reader intentionally has no spreadsheet-library runtime
    dependency. It accepts stored values only and rejects formulas.
    """
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with ZipFile(path) as book:
            workbook = ET.fromstring(book.read("xl/workbook.xml"))
            relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
            relation_targets = {
                rel.get("Id"): rel.get("Target")
                for rel in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
            }
            if "xl/sharedStrings.xml" in book.namelist():
                strings_root = ET.fromstring(book.read("xl/sharedStrings.xml"))
                shared_strings = ["".join(node.text or "" for node in item.findall(f".//{{{MAIN_NS}}}t"))
                                  for item in strings_root.findall(f"{{{MAIN_NS}}}si")]
            else:
                shared_strings = []
            result: dict[str, list[dict[str, Any]]] = {}
            for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
                name = sheet.get("name")
                if name not in (*REVIEW_SHEETS, ADJUDICATION_SHEET):
                    continue
                relation_id = sheet.get(f"{{{DOC_REL_NS}}}id")
                target = relation_targets.get(relation_id)
                if not target:
                    raise ContractError(f"no worksheet relationship for {name}")
                rows = _load_sheet_rows(book, target, shared_strings)
                result[name] = _rows_as_dicts(rows)
            expected = {*REVIEW_SHEETS, ADJUDICATION_SHEET}
            if set(result) != expected:
                raise ContractError(f"review workbook must contain sheets {sorted(expected)}")
            return result, digest
    except (BadZipFile, KeyError, ET.ParseError) as exc:
        raise ContractError(f"invalid review workbook: {exc}") from exc


def _rows_as_dicts(rows: list[dict[int, Any]]) -> list[dict[str, Any]]:
    header_row = next((row for row in rows if row.get(1) == "反应ID"), None)
    if header_row is None:
        raise ContractError("review worksheet is missing its expected header row")
    headers = {column: value for column, value in header_row.items() if isinstance(value, str) and value}
    out = []
    for row in rows:
        if row is header_row or not row.get(1) or not row.get(2):
            continue
        out.append({header: row.get(column) for column, header in headers.items()})
    return out


def _text(row: Mapping[str, Any], column: str, field: str, reaction_id: str) -> str:
    value = row.get(column)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{reaction_id}: {field} is blank")
    return value.strip()


def _multiplicity(row: Mapping[str, Any], column: str, field: str, reaction_id: str, *, required: bool) -> int | None:
    value = row.get(column)
    if value in (None, ""):
        if required:
            raise ContractError(f"{reaction_id}: {field} is required for acceptance")
        return None
    if isinstance(value, bool):
        raise ContractError(f"{reaction_id}: {field} must be an integer from 1 to 10")
    try:
        number = int(str(value))
    except ValueError as exc:
        raise ContractError(f"{reaction_id}: {field} must be an integer from 1 to 10") from exc
    if str(number) != str(value).strip() or not 1 <= number <= 10:
        raise ContractError(f"{reaction_id}: {field} must be an integer from 1 to 10")
    return number


def _index_rows(rows: list[dict[str, Any]], columns: Mapping[str, str], sheet_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        reaction_id = _text(row, columns["reaction_id"], "reaction ID", sheet_name)
        if reaction_id in indexed:
            raise ContractError(f"{sheet_name}: duplicate reaction ID {reaction_id}")
        indexed[reaction_id] = row
    return indexed


def import_review_rows(*, cases: Mapping[str, dict[str, Any]], reviewer_one: list[dict[str, Any]],
                       reviewer_two: list[dict[str, Any]], adjudications: list[dict[str, Any]],
                       workbook_sha256: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate all independent reviews and return reviewed cases and ReviewRecords.

    An accepted case becomes executable only if both reviewers completed every
    dimension, both agree on acceptance and 1D feasibility, the adjudicator
    agrees, and explicit endpoint multiplicities were entered.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", workbook_sha256):
        raise ContractError("workbook_sha256 must be a lowercase SHA256")
    r1 = _index_rows(reviewer_one, REVIEW_COLUMNS, REVIEW_SHEETS[0])
    r2 = _index_rows(reviewer_two, REVIEW_COLUMNS, REVIEW_SHEETS[1])
    adj = _index_rows(adjudications, ADJ_COLUMNS, ADJUDICATION_SHEET)
    expected = set(cases)
    if set(r1) != expected or set(r2) != expected or set(adj) != expected:
        raise ContractError("review workbook reaction IDs must exactly match the ReactionCase manifest")
    processed_cases: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    reviewer_identities: list[set[str]] = []

    for reaction_id, case in cases.items():
        rows = [r1[reaction_id], r2[reaction_id]]
        adjudication_row = adj[reaction_id]
        reviewer_records = []
        identities = set()
        for label, row in zip(REVIEW_SHEETS, rows, strict=True):
            if row.get(REVIEW_COLUMNS["reaction_id"]) != reaction_id or row.get(REVIEW_COLUMNS["split"]) != case.get("split"):
                raise ContractError(f"{reaction_id}: {label} reaction/split identity mismatch")
            identity = _text(row, REVIEW_COLUMNS["reviewer"], f"{label} reviewer identity", reaction_id)
            if identity in identities:
                raise ContractError(f"{reaction_id}: independent reviewer identities must differ")
            identities.add(identity)
            decision = _text(row, REVIEW_COLUMNS["decision"], f"{label} decision", reaction_id)
            if decision not in REVIEWER_DECISIONS or decision == "pending":
                raise ContractError(f"{reaction_id}: {label} decision is incomplete or invalid")
            dimensions = {key: _text(row, REVIEW_COLUMNS[key], f"{label} {key}", reaction_id)
                          for key in DIMENSIONS}
            if any(value not in REVIEW_DIMENSIONS or value == "not_reviewed" for value in dimensions.values()):
                raise ContractError(f"{reaction_id}: {label} review dimensions are incomplete")
            feasibility = _text(row, REVIEW_COLUMNS["scan_feasibility"], f"{label} scan feasibility", reaction_id)
            if feasibility not in SCAN_FEASIBILITY or feasibility == "pending":
                raise ContractError(f"{reaction_id}: {label} scan feasibility is incomplete")
            reviewer_records.append({
                "reviewer": identity,
                "decision": decision,
                "dimensions": dimensions,
                "scan_feasibility": feasibility,
                "multiplicities": {
                    "reactant": _multiplicity(row, REVIEW_COLUMNS["reactant_multiplicity"],
                        f"{label} R multiplicity", reaction_id, required=False),
                    "product": _multiplicity(row, REVIEW_COLUMNS["product_multiplicity"],
                        f"{label} P multiplicity", reaction_id, required=False),
                },
                "note": row.get(REVIEW_COLUMNS["note"]) or "",
            })
        reviewer_identities.append(identities)
        if reviewer_records[0]["reviewer"] == reviewer_records[1]["reviewer"]:
            raise ContractError(f"{reaction_id}: reviewer identities must differ")

        if adjudication_row.get(ADJ_COLUMNS["reaction_id"]) != reaction_id \
                or adjudication_row.get(ADJ_COLUMNS["split"]) != case.get("split") \
                or adjudication_row.get(ADJ_COLUMNS["case_id"]) != case.get("case_id"):
            raise ContractError(f"{reaction_id}: adjudication identity/split/case_id mismatch")
        for reviewer_idx, key in enumerate(("reviewer1_decision", "reviewer2_decision")):
            if adjudication_row.get(ADJ_COLUMNS[key]) != reviewer_records[reviewer_idx]["decision"]:
                raise ContractError(f"{reaction_id}: adjudication does not match {REVIEW_SHEETS[reviewer_idx]}")
        decision = _text(adjudication_row, ADJ_COLUMNS["decision"], "final adjudication", reaction_id)
        if decision not in FINAL_DECISIONS:
            raise ContractError(f"{reaction_id}: final adjudication is incomplete or invalid")
        adjudicator = _text(adjudication_row, ADJ_COLUMNS["adjudicator"], "adjudicator", reaction_id)
        resolution_note = adjudication_row.get(ADJ_COLUMNS["resolution_note"]) or ""
        if not isinstance(resolution_note, str):
            raise ContractError(f"{reaction_id}: resolution note must be text")
        if reviewer_records[0]["decision"] != reviewer_records[1]["decision"] and not resolution_note.strip():
            raise ContractError(f"{reaction_id}: reviewer disagreement requires an adjudication note")
        accepted = decision == "accept"
        final_multiplicities = {
            "reactant": _multiplicity(adjudication_row, ADJ_COLUMNS["reactant_multiplicity"],
                "final R multiplicity", reaction_id, required=accepted),
            "product": _multiplicity(adjudication_row, ADJ_COLUMNS["product_multiplicity"],
                "final P multiplicity", reaction_id, required=accepted),
        }
        if accepted:
            if case.get("status") != "needs_review":
                raise ContractError(f"{reaction_id}: source ReactionCase must be needs_review")
            if reviewer_records[0]["decision"] != "accept" or reviewer_records[1]["decision"] != "accept":
                if not resolution_note.strip():
                    raise ContractError(f"{reaction_id}: overriding a reviewer decision requires an adjudication note")
            if any(any(value != "confirmed" for value in reviewer["dimensions"].values())
                   for reviewer in reviewer_records):
                raise ContractError(f"{reaction_id}: accept requires all review dimensions confirmed")
            if any(reviewer["scan_feasibility"] != "1D" for reviewer in reviewer_records):
                raise ContractError(f"{reaction_id}: M1 executable acceptance requires 1D feasibility from both reviewers")
            for reviewer in reviewer_records:
                if any(reviewer["multiplicities"][endpoint] != final_multiplicities[endpoint]
                       for endpoint in ("reactant", "product")):
                    raise ContractError(f"{reaction_id}: final multiplicities must match each reviewer's explicit values")

        new_case = dict(case)
        if decision == "accept":
            new_case["status"] = "ready"
            new_case["reactant"] = {**case["reactant"], "multiplicity": final_multiplicities["reactant"]}
            new_case["product"] = {**case["product"], "multiplicity": final_multiplicities["product"]}
            new_case.pop("review_reasons", None)
        elif decision == "reject":
            new_case["status"] = "rejected"
            new_case["review_reasons"] = [resolution_note.strip() or "rejected by completed chemistry review"]
        else:
            new_case["status"] = "needs_review"
            new_case["review_reasons"] = [resolution_note.strip() or f"human review disposition: {decision}"]
        review_id = f"review:{reaction_id}:{workbook_sha256[:12]}"
        new_source = dict(new_case.get("source") or {})
        new_source["review_record_id"] = review_id
        new_source["reviewed_from_case_sha256"] = case["content_sha256"]
        new_case["source"] = new_source
        new_case = seal_document(new_case)
        problems = []
        # Rejected plans may retain unresolved endpoint multiplicities.
        from pes2ts_core.contracts import validate_document
        problems = validate_document(new_case, production_input=True)
        if problems:
            raise ContractError(f"{reaction_id}: reviewed ReactionCase invalid: {'; '.join(problems)}")
        dumps_document(new_case)
        review_status = {"accept":"accepted", "reject":"rejected", "needs_more_info":"needs_more_info",
                         "replace":"replacement_required"}[decision]
        record = make_document("ReviewRecord", review_id, review_status,
            dataset_version=case["dataset_version"], reaction_id=reaction_id, case_id=case["case_id"],
            split=case["split"], case_sha256=case["content_sha256"], workbook_sha256=workbook_sha256,
            reviewers=reviewer_records,
            adjudication={"decision": review_status, "adjudicator": adjudicator,
                "reactant_multiplicity": final_multiplicities["reactant"],
                "product_multiplicity": final_multiplicities["product"],
                "resolution_note": resolution_note.strip()},
            decision_source="human",
            extensions={"pes2ts.review_output.v1":{"reviewed_case_sha256":new_case["content_sha256"]}})
        processed_cases[reaction_id] = new_case
        records[reaction_id] = record

    # Reviewer identities must not drift between rows within a sheet.
    first_pair = next(iter(reviewer_identities), set())
    if any(pair != first_pair for pair in reviewer_identities):
        raise ContractError("reviewer identities must stay consistent within their independent workbook tabs")
    return processed_cases, records
