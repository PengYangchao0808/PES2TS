"""Audit the 24 demo candidates using endpoint-only, non-truth inputs.

This checks data consistency and provenance. It does not assess chemical
plausibility and never reads TS/IRC fields or barrier values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CSV = REPO_ROOT / "data/manifests/pes2ts_demo24_candidates_v1.csv"
REVIEW_QUEUE = REPO_ROOT / "data/manifests/demo24_review_queue_v1.json"
TRAIN_IDS = REPO_ROOT / "data/raw/reaction_qm/B3LYP-RXN_train.csv"
VALID_IDS = REPO_ROOT / "data/raw/reaction_qm/B3LYP-RXN_valid.csv"
EXPORT_ROOT = REPO_ROOT / "data/interim/g1_v2/export_contracts_v1"
FORBIDDEN_KEYS = {"endpoint_match", "orientation", "ts_coordinates", "ts_geometry", "irc_frames", "irc_coordinates"}


def _only_first_csv_column(path: Path) -> set[str]:
    """Read reaction IDs only; do not parse other source CSV fields."""
    ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split(",", 1)[0]
        if header != "reaction_id":
            raise ValueError(f"{path.name}: first column must be reaction_id")
        for line in stream:
            reaction_id = line.partition(",")[0].strip()
            if reaction_id:
                ids.add(reaction_id)
    return ids


def _forbidden_keys(value: Any, path: str = "$", found: list[str] | None = None) -> list[str]:
    found = [] if found is None else found
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            _forbidden_keys(child, f"{path}.{key}", found)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_keys(child, f"{path}[{index}]", found)
    return found


def _molecule_bonds(molecule: Chem.Mol) -> dict[tuple[int, int], float]:
    result = {}
    for bond in molecule.GetBonds():
        pair = tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum())))
        result[pair] = 1.5 if bond.GetIsAromatic() else float(bond.GetBondTypeAsDouble())
    return result


def _geometry_min_distance(coordinates: list[list[float]]) -> float:
    distances = [math.dist(coordinates[i], coordinates[j])
                 for i in range(len(coordinates)) for j in range(i + 1, len(coordinates))]
    return min(distances) if distances else math.inf


def audit_demo24() -> dict[str, Any]:
    if not CANDIDATE_CSV.is_file() or not REVIEW_QUEUE.is_file():
        raise FileNotFoundError("demo24 candidate CSV or review queue is missing")
    with CANDIDATE_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    queue = json.loads(REVIEW_QUEUE.read_text(encoding="utf-8"))
    train_ids, valid_ids = _only_first_csv_column(TRAIN_IDS), _only_first_csv_column(VALID_IDS)
    official = {"train": train_ids, "valid": valid_ids}
    queue_by_id = {item["reaction_id"]: item for item in queue["reactions"]}
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    checks = Counter()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        reaction_id = row["reaction_id"]
        issues: list[str] = []
        if reaction_id in seen:
            issues.append("duplicate_candidate_id")
        seen.add(reaction_id)
        split = row["split"]
        if split not in official or reaction_id not in official[split]:
            issues.append("split_not_in_official_source")
        if any(reaction_id in ids for name, ids in official.items() if name != split):
            issues.append("split_conflicts_with_official_source")
        queued = queue_by_id.get(reaction_id)
        if queued is None or queued.get("split") != split:
            issues.append("review_queue_identity_or_split_mismatch")

        numeric_id = int(reaction_id.rsplit("_", 1)[1])
        export_path = EXPORT_ROOT / f"{numeric_id // 1000:05d}" / f"{reaction_id}.json"
        if not export_path.is_file():
            issues.append("sanitized_export_missing")
            records.append({"reaction_id": reaction_id, "split": split, "checks": {}, "issues": issues,
                            "human_review_status": queued.get("review_status") if queued else "missing"})
            continue
        export = json.loads(export_path.read_text(encoding="utf-8"))
        if export.get("reaction_id") != reaction_id:
            issues.append("export_id_mismatch")
        truth_paths = _forbidden_keys(export)
        if truth_paths:
            issues.append("forbidden_truth_key_present")

        maps, elements = export.get("maps", []), export.get("elements", [])
        atom_identity_ok = (len(maps) == len(elements) == int(row["n_atoms"])
                            and len(set(maps)) == len(maps))
        if not atom_identity_ok:
            issues.append("atom_map_or_element_identity_mismatch")
        periodic_table = Chem.GetPeriodicTable()
        expected_numbers = [periodic_table.GetAtomicNumber(element) for element in elements]
        if export.get("r_atomic_numbers") != expected_numbers or export.get("p_atomic_numbers") != expected_numbers:
            issues.append("endpoint_atomic_number_order_mismatch")

        endpoint_checks: dict[str, Any] = {}
        valid_geometry = True
        endpoint_minima = []
        for side, key in (("reactant", "r_coordinates"), ("product", "p_coordinates")):
            xyz = export.get(key, [])
            side_valid = (len(xyz) == len(maps) and all(
                isinstance(atom_xyz, list) and len(atom_xyz) == 3
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in atom_xyz)
                for atom_xyz in xyz))
            if not side_valid:
                issues.append(f"{side}_geometry_shape_or_finiteness_error")
                valid_geometry = False
                continue
            minimum = _geometry_min_distance(xyz)
            endpoint_minima.append(minimum)
            endpoint_checks[side] = {"minimum_pair_distance_angstrom": round(minimum, 6),
                                     "no_severe_overlap_at_0_45_angstrom": minimum >= 0.45}
            if minimum < 0.45:
                issues.append(f"{side}_severe_geometry_overlap")
        if len(endpoint_minima) == 2 and round(min(endpoint_minima), 3) != float(row["min_endpoint_distance_A"]):
            issues.append("candidate_table_min_distance_mismatch")

        parts = row["reaction_smiles"].split(">>")
        molecules = [Chem.MolFromSmiles(part, parser) for part in parts] if len(parts) == 2 else []
        smiles_ok = len(molecules) == 2 and all(molecule is not None for molecule in molecules)
        bond_edits_ok = False
        electronic_summary_ok = False
        components_ok = False
        hydrogen_event_ok = False
        if not smiles_ok:
            issues.append("endpoint_reaction_smiles_parse_error")
        else:
            graphs = []
            components_ok = True
            electronic_summary_ok = True
            for molecule, side, component_column, radical_column, charged_column in (
                (molecules[0], "reactant", "R_components", "radical_electrons_R", "charged_atoms_R"),
                (molecules[1], "product", "P_components", "radical_electrons_P", "charged_atoms_P"),
            ):
                atom_map_elements = {atom.GetAtomMapNum(): atom.GetSymbol() for atom in molecule.GetAtoms()}
                if set(atom_map_elements) != set(maps) or any(
                    atom_map_elements.get(atom_map) != element for atom_map, element in zip(maps, elements)
                ):
                    issues.append(f"{side}_reaction_smiles_map_or_element_mismatch")
                if len(Chem.GetMolFrags(molecule)) != int(row[component_column]):
                    issues.append(f"{side}_component_count_mismatch")
                    components_ok = False
                radicals = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
                charged_atoms = sum(atom.GetFormalCharge() != 0 for atom in molecule.GetAtoms())
                if radicals != int(row[radical_column]) or charged_atoms != int(row[charged_column]):
                    issues.append(f"{side}_charge_or_radical_summary_mismatch")
                    electronic_summary_ok = False
                if sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()) != int(
                    export["charge_total_reactants" if side == "reactant" else "charge_total_products"]
                ):
                    issues.append(f"{side}_total_charge_mismatch")
                    electronic_summary_ok = False
                graphs.append(_molecule_bonds(molecule))

            reactant_graph, product_graph = graphs
            graph_changes = {
                pair: (reactant_graph.get(pair), product_graph.get(pair))
                for pair in set(reactant_graph) | set(product_graph)
                if reactant_graph.get(pair) != product_graph.get(pair)
            }
            listed_changes = {
                tuple(sorted(map(int, edit["pair"]))): (edit.get("r_bond_order"), edit.get("p_bond_order"))
                for edit in export.get("edits", [])
            }
            bond_edits_ok = set(graph_changes) == set(listed_changes) and all(
                all((graph_changes[pair][side] is None and listed_changes[pair][side] is None)
                    or (graph_changes[pair][side] is not None and listed_changes[pair][side] is not None
                        and abs(graph_changes[pair][side] - listed_changes[pair][side]) < 1e-6)
                    for side in (0, 1)) for pair in graph_changes
            )
            if not bond_edits_ok:
                issues.append("mapped_graph_bond_edits_do_not_match_export")

        edit_counts = Counter(edit.get("edit_kind") for edit in export.get("edits", []))
        counts_ok = (edit_counts["formed"] == int(row["F"])
                     and edit_counts["broken"] == int(row["B"])
                     and edit_counts["order_changed"] == int(row["O"]))
        if not counts_ok:
            issues.append("candidate_table_edit_count_mismatch")
        hydrogen_changes = export.get("hydrogen_partner_changes", [])
        heavy_transfers = sum(change.get("kind") == "transfer" for change in hydrogen_changes)
        hh_events = sum(change.get("kind") in {"from_hh", "to_hh"} for change in hydrogen_changes)
        hydrogen_event_ok = heavy_transfers == int(row["n_h_transfer"]) and hh_events == int(row["n_h_hh_events"])
        if not hydrogen_event_ok:
            issues.append("candidate_table_hydrogen_event_count_mismatch")

        mapping_status = row["mapping_status"]
        mapping_resolved = mapping_status in {"resolved_unique", "resolved_symmetry_collapsed"}
        if not mapping_resolved:
            issues.append("mapping_status_not_resolved")
        row_checks = {"official_split_match": reaction_id in official.get(split, set()),
                      "sanitized_export_present": True, "no_forbidden_truth_fields": not truth_paths,
                      "atom_identity_match": atom_identity_ok, "endpoint_geometry_valid": valid_geometry,
                      "reaction_smiles_parse_and_map_match": smiles_ok and not any(
                          issue.endswith("reaction_smiles_map_or_element_mismatch") for issue in issues),
                      "mapped_bond_edits_match": bond_edits_ok, "component_counts_match": components_ok,
                      "charge_and_radical_summaries_match": electronic_summary_ok,
                      "hydrogen_event_counts_match": hydrogen_event_ok, "edit_counts_match": counts_ok,
                      "review_queue_match": queued is not None and queued.get("split") == split,
                      "mapping_status_resolved": mapping_resolved}
        if any(not value for value in row_checks.values()) and not issues:
            issues.append("endpoint_consistency_check_failed")
        checks.update(row_checks)
        records.append({"reaction_id": reaction_id, "split": split, "stratum": row["stratum"],
                        "n_atoms": len(maps), "mapping_status": mapping_status,
                        "endpoint_minimum_distances_angstrom": endpoint_checks,
                        "checks": row_checks,
                        "issues": issues,
                        "human_review_status": queued.get("review_status") if queued else "missing"})

    split_counts = Counter(row["split"] for row in rows)
    all_issues = [(record["reaction_id"], issue) for record in records for issue in record["issues"]]
    review_counts = Counter(record["human_review_status"] for record in records)
    collection_issues = []
    if len(queue["reactions"]) != len(queue_by_id):
        collection_issues.append("review_queue_contains_duplicate_reaction_ids")
    if set(queue_by_id) != seen:
        collection_issues.append("candidate_list_and_review_queue_id_sets_differ")
    audit_status = "consistent_pending_human_review" if not all_issues and not collection_issues else "issues_found"
    return {
        "schema_version": "demo24_endpoint_audit_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "audit_status": audit_status,
        "information_boundary": "Endpoint-only structural consistency checks. No TS/IRC fields, geometries, energies, or labels were read. This is not a chemical plausibility review or acceptance.",
        "sources": {
            "candidate_list": "data/manifests/pes2ts_demo24_candidates_v1.csv",
            "review_queue": "data/manifests/demo24_review_queue_v1.json",
            "official_split_id_columns_only": ["data/raw/reaction_qm/B3LYP-RXN_train.csv", "data/raw/reaction_qm/B3LYP-RXN_valid.csv"],
            "sanitized_endpoint_exports": "data/interim/g1_v2/export_contracts_v1",
            "rdkit_version": rdBase.rdkitVersion,
        },
        "grain": "one row per reaction_id",
        "n_candidates": len(rows),
        "n_unique_reaction_ids": len(seen),
        "split_counts": dict(sorted(split_counts.items())),
        "review_status_counts": dict(sorted(review_counts.items())),
        "checks_passed_for_all_candidates": dict(sorted(checks.items())),
        "n_issue_rows": len(all_issues),
        "collection_issues": collection_issues,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/manifests/demo24_endpoint_audit_v1.json")
    args = parser.parse_args()
    report = audit_demo24()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit_status": report["audit_status"],
                      "n_candidates": report["n_candidates"], "split_counts": report["split_counts"],
                      "n_issue_rows": report["n_issue_rows"], "review_status_counts": report["review_status_counts"]},
                     ensure_ascii=False, sort_keys=True))
    return 0 if report["audit_status"] == "consistent_pending_human_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
