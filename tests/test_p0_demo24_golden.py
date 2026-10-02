"""P0 golden acceptance: reproduce Demo24 strategy evidence from frozen fixtures.

Reads ONLY the version-controlled fixtures under ``tests/fixtures/p0_demo24/``
and the git-tracked evidence JSON
``outputs/pes_generation_strategy_design_v1/demo24_strategy_evidence.json``.
The ``data/`` tree is never opened (enforced by an open() guard test).

Pipeline under test (todo 5-9 modules):

    snapshot -> load_endpoint_materials_from_export
             -> rebuild_endpoint_graphs
             -> aromatic_regions_from_bundle -> build_reaction_edit_graph
             -> build_endpoint_context (+ coordinates + endpoint electronic)
             -> build_event_coupling_graph

Comparisons against the evidence JSON:

* ``records[].features`` field-by-field EXACT equality per record
  (edit_counts, edit_components, connectivity_edit_components, edit_cycle_rank,
  endpoints, atom_attribute_changes, hydrogen_events, aromatic_regions);
* ``records[].edits[]`` field-by-field with build_evidence.py key-presence and
  6-decimal distance rounding;
* every edit belongs to exactly one event or a recorded shared membership;
* zero forbidden truth keys in all P0 output documents (recursive);
* fixture binding: snapshot ``export_sha256`` == manifest mapping == evidence
  (tampering one value in memory must fail the binding);
* running RDKit must equal the fixture manifest pin (pytest.fail otherwise).

Evidence ``proposal``/``PROPOSALS`` blocks are design-time hypotheses, NOT
selector outputs; this test never treats them as produced by the P0 pipeline.
"""

from __future__ import annotations

import builtins
import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from rdkit import rdBase

from pes2ts_core.contracts import FORBIDDEN_TRUTH_KEYS
from pes2ts_core.g1.endpoint_context import (
    EndpointElectronic,
    build_endpoint_context,
    evidence_edit_fields,
)
from pes2ts_core.g1.event_coupling import (
    aromatic_regions_from_bundle,
    build_event_coupling_graph,
)
from pes2ts_core.g1.reaction_edit_graph import build_reaction_edit_graph
from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.generation.planning.graph_rebuild import (
    load_endpoint_materials_from_export,
    rebuild_endpoint_graphs,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests/fixtures/p0_demo24"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
EVIDENCE_PATH = ROOT / "outputs/pes_generation_strategy_design_v1/demo24_strategy_evidence.json"
CSV_PATH = FIXTURE_ROOT / "pes2ts_demo24_candidates_v1.csv"

#: features.* fields that must match the evidence JSON exactly.
FEATURE_FIELDS = (
    "edit_counts",
    "edit_components",
    "connectivity_edit_components",
    "edit_cycle_rank",
    "endpoints",
    "atom_attribute_changes",
    "hydrogen_events",
    "aromatic_regions",
)

#: records[].edits[] fields compared per edit (key presence as in build_evidence).
EDIT_FIELDS = (
    "pair",
    "edit_kind",
    "elements",
    "r_bond_order",
    "p_bond_order",
    "aromatic_region",
    "support_path_R",
    "support_path_P",
    "cross_component_R",
    "cross_component_P",
    "distance_R_A",
    "distance_P_A",
)

#: Keys that must never appear in any P0 output document (recursive).
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {key.lower() for key in FORBIDDEN_TRUTH_KEYS}
    | {key.lower() for key in FORBIDDEN_EXPORT_KEYS}
) | frozenset({"endpoint_match", "orientation", "irc_evidence"})


def _load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_evidence() -> dict[str, Any]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _load_snapshot(reaction_id: str) -> dict[str, Any]:
    path = FIXTURE_ROOT / "records" / f"{reaction_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_rows() -> dict[str, dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return {row["reaction_id"]: row for row in csv.DictReader(handle)}


def _typed_mismatch(
    *, record_id: str, field: str, expected: Any, got: Any, explanation: str
) -> AssertionError:
    return AssertionError(
        f"P0 golden mismatch | record={record_id} field={field}\n"
        f"  expected: {expected!r}\n"
        f"  got:      {got!r}\n"
        f"  typed:    {explanation}"
    )


def _assert_binding(
    snapshot: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    evidence_record: Mapping[str, Any],
    csv_row: Mapping[str, str],
) -> None:
    """Fixture/evidence/CSV binding for one record (tamper target)."""
    rid = str(snapshot["reaction_id"])
    pairs = (
        (
            "export_sha256",
            snapshot.get("export_sha256"),
            manifest_record.get("export_sha256"),
            evidence_record.get("export_sha256"),
        ),
        (
            "reaction_smiles",
            snapshot.get("reaction_smiles"),
            manifest_record.get("reaction_smiles"),
            csv_row.get("reaction_smiles"),
        ),
    )
    for field, snap_val, man_val, ev_val in pairs:
        if not (snap_val == man_val == ev_val):
            raise _typed_mismatch(
                record_id=rid,
                field=f"binding.{field}",
                expected={"manifest": man_val, "evidence_or_csv": ev_val},
                got={"snapshot": snap_val},
                explanation=(
                    "fixture snapshot, manifest mapping and evidence/CSV must "
                    "agree; altered export_sha256 or reaction_smiles breaks "
                    "the frozen-input binding"
                ),
            )


def _rebuild_p0(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Run the full P0 pipeline on one fixture snapshot (pure in-memory)."""
    materials = load_endpoint_materials_from_export(snapshot)
    bundle = rebuild_endpoint_graphs(str(snapshot["reaction_smiles"]), materials)
    aromatic = aromatic_regions_from_bundle(bundle)
    edit_graph = build_reaction_edit_graph(
        bundle, aromatic_regions=aromatic if aromatic else None
    )
    maps = [int(m) for m in snapshot["maps"]]
    r_coordinates = {
        map_id: tuple(float(x) for x in snapshot["r_coordinates"][i])
        for i, map_id in enumerate(maps)
    }
    p_coordinates = {
        map_id: tuple(float(x) for x in snapshot["p_coordinates"][i])
        for i, map_id in enumerate(maps)
    }
    electronic_raw = snapshot["endpoint_electronic"]
    endpoint_electronic = {
        "reactant": EndpointElectronic(
            charge=int(electronic_raw["reactant"]["charge"]),
            multiplicity=int(electronic_raw["reactant"]["multiplicity"]),
        ),
        "product": EndpointElectronic(
            charge=int(electronic_raw["product"]["charge"]),
            multiplicity=int(electronic_raw["product"]["multiplicity"]),
        ),
    }
    context = build_endpoint_context(
        bundle,
        edit_graph,
        r_coordinates=r_coordinates,
        p_coordinates=p_coordinates,
        endpoint_electronic=endpoint_electronic,
    )
    coupling = build_event_coupling_graph(bundle, edit_graph, context)
    return {
        "bundle": bundle,
        "edit_graph": edit_graph,
        "context": context,
        "coupling": coupling,
    }


def _computed_features(p0: Mapping[str, Any]) -> dict[str, Any]:
    edit_doc = p0["edit_graph"].to_doc()
    coupling = p0["coupling"]
    return {
        "edit_counts": edit_doc["edit_counts"],
        "edit_components": edit_doc["edit_components"],
        "connectivity_edit_components": edit_doc["connectivity_edit_components"],
        "edit_cycle_rank": edit_doc["edit_cycle_rank"],
        "endpoints": p0["context"].endpoints_evidence_block(),
        "atom_attribute_changes": edit_doc["atom_attribute_changes"],
        "hydrogen_events": [dict(item) for item in coupling.hydrogen_events],
        "aromatic_regions": {
            region_id: [list(pair) for pair in members]
            for region_id, members in sorted(coupling.aromatic_regions.items())
        },
    }


def _computed_edits(p0: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Merge edit-graph six fields with context evidence fields per pair."""
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    context_by_pair = {
        (edit.pair[0], edit.pair[1]): edit for edit in p0["context"].edits
    }
    for record in p0["edit_graph"].edit_records():
        pair = (int(record["pair"][0]), int(record["pair"][1]))
        ctx_edit = context_by_pair[pair]
        merged = dict(record)
        merged.update(evidence_edit_fields(ctx_edit))
        # evidence_edit_fields re-emits pair/orders/kind from the context copy;
        # restore the edit-graph values so both sources stay independently
        # visible in typed diffs (they must already agree).
        merged["pair"] = record["pair"]
        merged["edit_kind"] = record["edit_kind"]
        merged["r_bond_order"] = record["r_bond_order"]
        merged["p_bond_order"] = record["p_bond_order"]
        by_pair[pair] = merged
    return by_pair


def _normalize_for_compare(field: str, value: Any) -> Any:
    """Project computed values into the evidence JSON JSON-shape."""
    if field == "pair" and isinstance(value, tuple):
        return [int(v) for v in value]
    return value


def _assert_features_match(
    record_id: str, expected: Mapping[str, Any], got: Mapping[str, Any]
) -> None:
    for field in FEATURE_FIELDS:
        if field not in expected:
            raise _typed_mismatch(
                record_id=record_id,
                field=f"features.{field}",
                expected="<present>",
                got="<missing in evidence>",
                explanation="evidence features block lost a required field",
            )
        exp_val = expected[field]
        got_val = got.get(field)
        if got_val != exp_val:
            raise _typed_mismatch(
                record_id=record_id,
                field=f"features.{field}",
                expected=exp_val,
                got=got_val,
                explanation=(
                    "P0 recomputation must reproduce the evidence features "
                    "field exactly (Counter semantics: absent kinds stay absent)"
                ),
            )


def _assert_edits_match(
    record_id: str,
    expected_edits: list[Mapping[str, Any]],
    got_edits: Mapping[tuple[int, int], Mapping[str, Any]],
) -> None:
    expected_pairs = [
        (int(edit["pair"][0]), int(edit["pair"][1])) for edit in expected_edits
    ]
    if len(set(expected_pairs)) != len(expected_pairs):
        raise _typed_mismatch(
            record_id=record_id,
            field="edits",
            expected="unique pairs",
            got=expected_pairs,
            explanation="evidence edit list must carry unique map pairs",
        )
    if set(expected_pairs) != set(got_edits):
        raise _typed_mismatch(
            record_id=record_id,
            field="edits.pair_set",
            expected=sorted(expected_pairs),
            got=sorted(got_edits),
            explanation="P0 edit graph pairs must equal the evidence edit pairs",
        )
    for expected_edit, pair in zip(expected_edits, expected_pairs, strict=True):
        got_edit = got_edits[pair]
        for field in EDIT_FIELDS:
            in_expected = field in expected_edit
            in_got = field in got_edit
            if in_expected != in_got:
                raise _typed_mismatch(
                    record_id=record_id,
                    field=f"edits[{pair}].{field}",
                    expected=expected_edit if in_expected else "<absent>",
                    got=got_edit if in_got else "<absent>",
                    explanation=(
                        "key presence must follow build_evidence.py semantics: "
                        "support_path_R only for formed, support_path_P only "
                        "for broken, neither for order_changed"
                    ),
                )
            if not in_expected:
                continue
            exp_val = _normalize_for_compare(field, expected_edit[field])
            got_val = _normalize_for_compare(field, got_edit[field])
            if field.startswith("distance_"):
                # build_evidence rounds to 6 decimals; both sides already are.
                if got_val != exp_val:
                    raise _typed_mismatch(
                        record_id=record_id,
                        field=f"edits[{pair}].{field}",
                        expected=exp_val,
                        got=got_val,
                        explanation=(
                            "distance must equal round(math.dist(...), 6) "
                            "as in build_evidence.py"
                        ),
                    )
                continue
            if got_val != exp_val:
                raise _typed_mismatch(
                    record_id=record_id,
                    field=f"edits[{pair}].{field}",
                    expected=exp_val,
                    got=got_val,
                    explanation="P0 edit projection must match evidence exactly",
                )


def _assert_membership(record_id: str, p0: Mapping[str, Any]) -> None:
    """Every edit belongs to exactly one event or a recorded shared membership.

    Registry semantics (event_coupling.build_event_coupling_graph): each edit
    pair has exactly one primary owner; additional owners ("extras") are
    recorded with one typed justification each.  ``MembershipRecord.event_ids``
    is the alphabetically sorted union ``{primary, *extras}`` — the primary is
    NOT necessarily the first id; it is the unique id not covered by a
    justification when the row is shared.
    """
    just_rule_ids = {
        "aromatic_region_membership",
        "ring_group_membership",
        "shared_center",
        "h_partner_change",
    }
    edit_pairs = {
        (int(e.pair[0]), int(e.pair[1])) for e in p0["edit_graph"].edits
    }
    membership_rows = {row.edit_pair: row for row in p0["coupling"].membership}
    if set(membership_rows) != edit_pairs:
        raise _typed_mismatch(
            record_id=record_id,
            field="membership.edit_pairs",
            expected=sorted(edit_pairs),
            got=sorted(membership_rows),
            explanation="membership registry must cover exactly the edit pairs",
        )
    event_ids_seen: set[str] = {event.event_id for event in p0["coupling"].events}
    for pair in sorted(edit_pairs):
        row = membership_rows[pair]
        if not row.event_ids:
            raise _typed_mismatch(
                record_id=record_id,
                field=f"membership[{pair}].event_ids",
                expected=">=1 owning event",
                got=[],
                explanation="every edit must belong to at least one event",
            )
        unknown = [eid for eid in row.event_ids if eid not in event_ids_seen]
        if unknown:
            raise _typed_mismatch(
                record_id=record_id,
                field=f"membership[{pair}].event_ids",
                expected=f"ids within {sorted(event_ids_seen)}",
                got=list(row.event_ids),
                explanation=f"membership references unknown events {unknown}",
            )
        justified = {j.event_id for j in row.justifications}
        if len(row.event_ids) == 1:
            if justified:
                raise _typed_mismatch(
                    record_id=record_id,
                    field=f"membership[{pair}].justifications",
                    expected=[],
                    got=sorted(justified),
                    explanation=(
                        "single-owner membership must not carry shared-"
                        "membership justifications"
                    ),
                )
            continue
        if len(justified) != len(row.event_ids) - 1 or not justified < set(
            row.event_ids
        ):
            raise _typed_mismatch(
                record_id=record_id,
                field=f"membership[{pair}].justifications",
                expected=(
                    "one justification per additional owner; primary is the "
                    "unique event id without a justification "
                    f"(event_ids={list(row.event_ids)})"
                ),
                got=sorted(justified),
                explanation=(
                    "shared membership (multiple owning events) requires a "
                    "typed justification per additional membership; "
                    "event_ids is sorted, primary is not event_ids[0]"
                ),
            )
        for justification in row.justifications:
            if justification.rule_id not in just_rule_ids:
                raise _typed_mismatch(
                    record_id=record_id,
                    field=f"membership[{pair}].justifications.rule_id",
                    expected=sorted(just_rule_ids),
                    got=justification.rule_id,
                    explanation="additional membership needs a typed rule_id",
                )
            if justification.event_id not in row.event_ids:
                raise _typed_mismatch(
                    record_id=record_id,
                    field=f"membership[{pair}].justifications.event_id",
                    expected=list(row.event_ids),
                    got=justification.event_id,
                    explanation="justification must name one of the row's events",
                )
    # Events must reference only known edit pairs.
    for event in p0["coupling"].events:
        for pair in event.edit_pairs:
            if pair not in edit_pairs:
                raise _typed_mismatch(
                    record_id=record_id,
                    field=f"events[{event.event_id}].edit_pairs",
                    expected=sorted(edit_pairs),
                    got=[list(p) for p in event.edit_pairs],
                    explanation="event nodes may only claim P0 edit-graph pairs",
                )


def _iter_doc_keys(obj: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_iter_doc_keys(value, path))
    elif isinstance(obj, list | tuple):
        for index, item in enumerate(obj):
            keys.extend(_iter_doc_keys(item, f"{prefix}[{index}]"))
    return keys


def _assert_purity(record_id: str, p0: Mapping[str, Any]) -> None:
    documents = {
        "bundle": p0["bundle"].to_doc(),
        "edit_graph": p0["edit_graph"].to_doc(),
        "context": p0["context"].to_doc(),
        "coupling": p0["coupling"].to_doc(),
    }
    for doc_name, doc in documents.items():
        for path in _iter_doc_keys(doc):
            leaf = path.rsplit(".", 1)[-1]
            leaf = leaf.split("[", 1)[0]
            if leaf.lower() in FORBIDDEN_KEYS:
                raise _typed_mismatch(
                    record_id=record_id,
                    field=f"purity.{doc_name}:{path}",
                    expected="forbidden-free key",
                    got=leaf,
                    explanation=(
                        "P0 outputs must carry zero truth-derived keys "
                        f"(FORBIDDEN set: {sorted(FORBIDDEN_KEYS)})"
                    ),
                )


def _run_record(record: Mapping[str, Any], rows: Mapping[str, Mapping[str, str]]) -> None:
    rid = str(record["reaction_id"])
    manifest = _load_manifest()
    manifest_record = manifest["records"][rid]
    snapshot = _load_snapshot(rid)
    csv_row = rows[rid]
    _assert_binding(snapshot, manifest_record, record, csv_row)
    p0 = _rebuild_p0(snapshot)
    _assert_features_match(rid, record["features"], _computed_features(p0))
    _assert_edits_match(rid, record["edits"], _computed_edits(p0))
    _assert_membership(rid, p0)
    _assert_purity(rid, p0)
    # Snapshot copies of bundle-derived goldens must also agree (binding of
    # aromatic/H projections stored in the fixture to the recomputed values).
    coupling = p0["coupling"]
    if [dict(item) for item in coupling.hydrogen_events] != snapshot[
        "hydrogen_partner_changes"
    ]:
        raise _typed_mismatch(
            record_id=rid,
            field="snapshot.hydrogen_partner_changes",
            expected=snapshot["hydrogen_partner_changes"],
            got=[dict(item) for item in coupling.hydrogen_events],
            explanation="recomputed hydrogen events must equal the frozen snapshot copy",
        )
    snap_regions = {
        region_id: [list(pair) for pair in members]
        for region_id, members in sorted(snapshot["aromatic_regions"].items())
    }
    if coupling.aromatic_regions != snap_regions:
        raise _typed_mismatch(
            record_id=rid,
            field="snapshot.aromatic_regions",
            expected=snap_regions,
            got=coupling.aromatic_regions,
            explanation="recomputed aromatic regions must equal the frozen snapshot copy",
        )


def test_fixture_manifest_and_csv_sha256() -> None:
    """Given frozen fixtures, then CSV + manifest + evidence sha bindings hold."""
    manifest = _load_manifest()
    evidence = _load_evidence()
    csv_sha = _sha256_file(CSV_PATH)
    assert csv_sha == manifest["source_csv_sha256"] == evidence["source_csv_sha256"]
    assert manifest["n_records"] == len(manifest["records"]) == 24
    assert manifest["n_records"] == len(evidence["records"])
    for rid, mapping in manifest["records"].items():
        snapshot_path = FIXTURE_ROOT / mapping["snapshot"]
        assert snapshot_path.is_file(), rid
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert snapshot["reaction_id"] == rid
        assert snapshot["export_sha256"] == mapping["export_sha256"]
    assert set(manifest["records"]) == {
        record["reaction_id"] for record in evidence["records"]
    }


def test_rdkit_version_pin_matches_fixture_manifest() -> None:
    """Deterministic golden requires the pinned RDKit; mismatch fails loudly."""
    manifest = _load_manifest()
    pinned = str(manifest["rdkit_version"])
    running = rdBase.rdkitVersion
    if running != pinned:
        pytest.fail(
            "P0 golden requires rdkit=="
            f"{pinned} (tests/fixtures/p0_demo24/manifest.json pin; "
            f"evidence was built under {manifest.get('evidence_rdkit_version')}); "
            f"running rdkit=={running}. Deterministic graph features depend on "
            "the RDKit pin — run inside the reference conda env `pes2ts`."
        )
    assert pinned == "2026.03.6"


def test_golden_demo24_features_and_edits_exact() -> None:
    """Given all 24 evidence records, then P0 reproduces features+edits exactly."""
    evidence = _load_evidence()
    rows = _csv_rows()
    assert len(evidence["records"]) == 24
    for record in evidence["records"]:
        _run_record(record, rows)


def test_every_edit_has_event_membership_or_shared_justification() -> None:
    """Given each Demo24 record, then every edit has one owner or shared rows."""
    evidence = _load_evidence()
    for record in evidence["records"]:
        rid = str(record["reaction_id"])
        snapshot = _load_snapshot(rid)
        p0 = _rebuild_p0(snapshot)
        _assert_membership(rid, p0)


def test_p0_outputs_contain_zero_forbidden_truth_keys() -> None:
    """Given each Demo24 record, then P0 output docs carry zero truth keys."""
    evidence = _load_evidence()
    for record in evidence["records"]:
        rid = str(record["reaction_id"])
        snapshot = _load_snapshot(rid)
        p0 = _rebuild_p0(snapshot)
        _assert_purity(rid, p0)


def test_tampered_fixture_export_sha256_fails_binding(tmp_path: Path) -> None:
    """Given an in-memory-tampered export_sha256, then golden binding fails."""
    evidence = _load_evidence()
    manifest = _load_manifest()
    rows = _csv_rows()
    record = evidence["records"][0]
    rid = str(record["reaction_id"])
    snapshot = _load_snapshot(rid)
    manifest_record = dict(manifest["records"][rid])
    csv_row = rows[rid]

    # Sanity: untampered binding passes.
    _assert_binding(snapshot, manifest_record, record, csv_row)

    # Tamper snapshot binding field (in-memory copy only; fixture untouched).
    tampered_snapshot = dict(snapshot)
    tampered_snapshot["export_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="binding.export_sha256"):
        _assert_binding(tampered_snapshot, manifest_record, record, csv_row)

    # Tamper the manifest mapping instead.
    tampered_manifest_record = dict(manifest_record)
    tampered_manifest_record["export_sha256"] = "f" * 64
    with pytest.raises(AssertionError, match="binding.export_sha256"):
        _assert_binding(snapshot, tampered_manifest_record, record, csv_row)

    # Tamper a renamed/short sha in a tmp copy on disk — same failure path.
    tmp_snapshot_path = tmp_path / f"{rid}.json"
    tmp_snapshot = dict(snapshot)
    tmp_snapshot["export_sha256"] = "abc123"
    tmp_snapshot_path.write_text(
        json.dumps(tmp_snapshot, ensure_ascii=False), encoding="utf-8"
    )
    loaded_tmp = json.loads(tmp_snapshot_path.read_text(encoding="utf-8"))
    with pytest.raises(AssertionError, match="binding.export_sha256"):
        _assert_binding(loaded_tmp, manifest_record, record, csv_row)


def test_golden_pipeline_reads_no_data_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given the full 24-record golden pipeline, then no data/ path is opened."""
    real_open = builtins.open
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes

    def _reject_data_tree(path: object) -> None:
        text = str(path)
        normalized = text.replace("\\", "/")
        if "/data/" in normalized or normalized.startswith("data/"):
            raise AssertionError(
                f"P0 golden must run offline from fixtures only; refused read: {text}"
            )

    def guarded_open(file: object, *args: Any, **kwargs: Any) -> Any:
        _reject_data_tree(file)
        return real_open(file, *args, **kwargs)

    def guarded_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        _reject_data_tree(self)
        return real_read_text(self, *args, **kwargs)

    def guarded_read_bytes(self: Path) -> bytes:
        _reject_data_tree(self)
        return real_read_bytes(self)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    evidence = _load_evidence()
    rows = _csv_rows()
    assert len(evidence["records"]) == 24
    for record in evidence["records"]:
        _run_record(record, rows)
