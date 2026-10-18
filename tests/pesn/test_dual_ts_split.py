"""Dual-TS split lineage tests (WP0b) on synthetic v1-style split files.

The fixtures mirror the real ``outputs/pesn/direct_egnn_v1_20261015/splits``
structure: ``pesn_irc_split.json`` (``pesn_irc_split_v1``: subsets
train_fit/train_dev/train_calibration, family_holdout, isolation ledger) and
``selection_*.json`` (``pes2ts_direct_selection_v1`` with aligned
reaction_ids/group_ids and a recorded ``split_manifest_sha256``).
No truth/ACP/network data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.dual_ts import DualEndpointTask, EndpointBond, EndpointGraph
from pes2ts_core.pesn.offline.splits.dual_ts_split import (
    BRANCHES_PER_GROUP,
    DUAL_TS_SPLIT_LINEAGE_SCHEMA,
    DualTSSplitLineage,
    DualTSSplitLineageError,
    artifact_digest,
    build_dual_ts_split_lineage,
    validate_lineage_payload,
    write_lineage_artifact,
)


def _rid(n: int) -> str:
    return f"RXN_{n:010d}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_v1_tree(
    tmp_path: Path,
    *,
    n_fit: int = 12,
    n_monitor: int = 3,
    n_dev: int = 2,
    n_sealed: int = 2,
    n_isolated: int = 1,
    duplicate_pair: bool = True,
) -> Path:
    """Synthetic v1 splits dir: manifest + nested prefix selections."""
    base = tmp_path / "v1_splits"
    base.mkdir()
    fit = [_rid(1000 + i) for i in range(n_fit)]
    monitor = [_rid(5000 + i) for i in range(n_monitor)]
    dev = [_rid(7000 + i) for i in range(n_dev)]
    sealed = [_rid(9000 + i) for i in range(n_sealed)]
    isolated = [_rid(4000 + i) for i in range(n_isolated)]

    manifest = {
        "schema_version": "pesn_irc_split_v1",
        "parent_split_sha256": "0" * 64,
        "grouping_version": "pesn_grouping_v1",
        "group_members_sha256": "1" * 64,
        "subsets": {
            "train_fit": fit,
            "train_dev": monitor,
            "train_calibration": dev,
        },
        "family_holdout": {
            "families": [f"FAMSEALED{i}" for i in range(n_sealed)],
            "reaction_ids": sealed,
            "note": "sealed whole families; excluded from fit/monitor/dev",
        },
        "frozen": True,
        "known_group_cross_split_conflicts": 0,
        "direct_scope": {
            "isolation_ledger": {
                "isolated_reaction_ids": isolated,
                "mixed_family_groups": [],
                "missing_family_groups": [],
                "missing_family_labels_all_sets": [],
            },
        },
    }
    manifest_path = base / "pesn_irc_split.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = _sha(manifest_path)

    # duplicate pair: fit[1] is a forward/reverse duplicate of fit[0]
    # (selection group_ids are the v1 canonical group ids)
    def groups_for(ids: list[str]) -> list[str]:
        if not duplicate_pair:
            return list(ids)
        return [fit[0] if r == fit[1] else r for r in ids]

    selections: dict[str, tuple[str, list[str]]] = {
        "direct_fit50": ("fit", fit),
        "direct_fit500": ("fit", fit),
        "direct_fit2000": ("fit", fit),
        "direct_sac_fit12": ("fit", fit[: min(3, len(fit))]),
        "direct_monitor8": ("monitor", monitor),
        "direct_dev8": ("test", dev),
    }
    for name, (role, ids) in selections.items():
        gids = groups_for(ids)
        (base / f"selection_{name}.json").write_text(json.dumps({
            "schema_version": "pes2ts_direct_selection_v1",
            "selection": name,
            "reaction_ids": ids,
            "group_ids": gids,
            "n_reaction_ids": len(ids),
            "n_canonical_groups": len(set(gids)),
            "split_manifest_sha256": manifest_sha,
            "prefix_semantics": "test fixture",
        }))
    return base


def _load_payload(base: Path) -> dict:
    return build_dual_ts_split_lineage(base)


# ---------------------------------------------------------------------------
# Inheritance correctness
# ---------------------------------------------------------------------------


def test_inheritance_matches_v1_subsets_without_rerandomization(tmp_path):
    base = build_v1_tree(tmp_path)
    payload = _load_payload(base)
    assert payload["schema_version"] == DUAL_TS_SPLIT_LINEAGE_SCHEMA
    fit = set(payload["partitions"]["fit"]["group_ids"])
    monitor = set(payload["partitions"]["monitor"]["group_ids"])
    test = set(payload["partitions"]["test"]["group_ids"])
    holdout = set(payload["partitions"]["family_holdout"]["group_ids"])
    # fit reactions keep the fit partition; the duplicate pair shares ONE group
    assert _rid(1000) in fit and _rid(1001) not in fit  # merged into 1000
    assert _rid(1002) in fit and _rid(1011) in fit
    assert _rid(5000) in monitor and _rid(7000) in test
    assert _rid(9000) in holdout and _rid(9001) in holdout
    # no re-randomization: every v1 fit reaction maps into the fit partition
    binding = payload["v1_binding"]["reaction_to_group"]
    assert binding[_rid(1001)] == _rid(1000)
    assert set(fit) == {_rid(1000)} | {_rid(1000 + i) for i in range(2, 12)}
    # partitions are disjoint (rule a at group level)
    assert not (fit & monitor) and not (fit & test) and not (monitor & test)


def test_branch_inherits_group_partition(tmp_path):
    base = build_v1_tree(tmp_path)
    lineage = DualTSSplitLineage.from_v1_splits(base)
    gid = _rid(1000)
    # both branches A/B of one group inherit the group's partition
    assert lineage.partition_for_branch(gid, "A") == "fit"
    assert lineage.partition_for_branch(gid, "B") == "fit"
    with pytest.raises(DualTSSplitLineageError):
        lineage.partition_for_branch(gid, "R")  # identity label only
    # ... and a full contract task resolves via its partition_key()
    def _graph(bond_order: float) -> EndpointGraph:
        return EndpointGraph(
            atomic_numbers=np.array([6, 8]),
            atom_map_ids=("C1", "O1"),
            bonds=(EndpointBond(i=0, j=1, bond_order=bond_order),))
    task = DualEndpointTask(
        reaction_id=_rid(1000),
        reaction_group_id=gid,
        endpoints=(_graph(1.0), _graph(2.0)),
        method_id="test_method",
    )
    task.validate()
    assert lineage.partition_for_task(task) == "fit"
    assert lineage.partition_of("RXN_NEVER_SEEN") is None
    assert lineage.branch_potential()["fit"] == \
        lineage.partition_counts()["fit"] * BRANCHES_PER_GROUP


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_overlap_refusal_across_selection_roles(tmp_path):
    base = build_v1_tree(tmp_path)
    # a fit-listed reaction injected into the dev8 selection -> role overlap
    sel = json.loads((base / "selection_direct_dev8.json").read_text())
    sel["reaction_ids"] = [_rid(7000), _rid(1002)]
    sel["group_ids"] = [_rid(7000), _rid(1002)]
    sel["n_reaction_ids"] = 2
    sel["n_canonical_groups"] = 2
    (base / "selection_direct_dev8.json").write_text(json.dumps(sel))
    with pytest.raises(DualTSSplitLineageError) as excinfo:
        _load_payload(base)
    assert excinfo.value.code == "SELECTION_PARTITION_OVERLAP"


def test_family_conflict_refusal(tmp_path):
    base = build_v1_tree(tmp_path, n_sealed=2)
    # a sealed reaction sitting in train_fit is a family-holdout conflict
    manifest = json.loads((base / "pesn_irc_split.json").read_text())
    manifest["subsets"]["train_fit"].append(_rid(9000))
    (base / "pesn_irc_split.json").write_text(json.dumps(manifest))
    # keep the recorded manifest digest consistent to isolate the rule
    sha = _sha(base / "pesn_irc_split.json")
    for sel_path in sorted(base.glob("selection_*.json")):
        sel = json.loads(sel_path.read_text())
        sel["split_manifest_sha256"] = sha
        sel_path.write_text(json.dumps(sel))
    with pytest.raises(DualTSSplitLineageError) as excinfo:
        _load_payload(base)
    assert excinfo.value.code == "FAMILY_HOLDOUT_CONFLICT"


def test_leakage_refusal_when_group_spans_partitions(tmp_path):
    base = build_v1_tree(tmp_path)
    groups_src = tmp_path / "groups_source.json"
    groups_src.write_text(json.dumps({"groups": [{
        "reaction_group_id": "GRP_NEW",
        "member_reaction_ids": [_rid(1002), _rid(5000)],
    }]}))
    with pytest.raises(DualTSSplitLineageError) as excinfo:
        build_dual_ts_split_lineage(base, groups_src)
    assert excinfo.value.code == "PARTITION_LEAKAGE"


def test_binding_conflict_refusal(tmp_path):
    base = build_v1_tree(tmp_path)
    # fit500 claims the duplicate belongs to a different canonical group,
    # contradicting the fit50 binding evidence for the same reaction
    sel = json.loads((base / "selection_direct_fit500.json").read_text())
    sel["group_ids"] = ["GRP_OTHER" if r == _rid(1001) else r
                        for r in sel["reaction_ids"]]
    sel["n_canonical_groups"] = len(set(sel["group_ids"]))
    (base / "selection_direct_fit500.json").write_text(json.dumps(sel))
    with pytest.raises(DualTSSplitLineageError) as excinfo:
        _load_payload(base)
    assert excinfo.value.code == "BINDING_CONFLICT"


# ---------------------------------------------------------------------------
# Unassigned bucket (rule c)
# ---------------------------------------------------------------------------


def test_unassigned_groups_are_never_silently_assigned(tmp_path):
    base = build_v1_tree(tmp_path, n_isolated=2)
    groups_src = tmp_path / "groups_source.json"
    groups_src.write_text(json.dumps({
        "reaction_group_ids": ["GRP_BRAND_NEW"],
        "groups": [{
            "reaction_group_id": "GRP_MEMBERED",
            "member_reaction_ids": ["RXN_7777777777"],
        }],
    }))
    payload = build_dual_ts_split_lineage(base, groups_src)
    unassigned = {e["reaction_group_id"]: e for e in payload["unassigned"]}
    # isolated v1 reactions: explicit bucket with reason, no partition
    assert _rid(4000) in unassigned
    assert unassigned[_rid(4000)]["reason"] == "v1_isolated_not_in_any_subset"
    assert _rid(4000) not in set(payload["partitions"]["fit"]["group_ids"])
    # unknown groups land unassigned, never silently fit/monitor/test
    assert unassigned["GRP_BRAND_NEW"]["reason"] == "absent_from_v1_split"
    assert unassigned["GRP_MEMBERED"]["reason"] == "reaction_absent_from_v1_split"
    lineage = DualTSSplitLineage.from_payload(payload)
    assert lineage.partition_of("GRP_BRAND_NEW") is None
    assert lineage.unassigned_reason_of("GRP_BRAND_NEW") == "absent_from_v1_split"
    counts = lineage.partition_counts()
    assert counts["unassigned"] == len(payload["unassigned"])


def test_isolated_reaction_recorded_when_inheriting_partition(tmp_path):
    # the real v1 data has exactly one isolated reaction inside a subset;
    # the lineage must inherit (not re-judge) but flag it in the audit
    base = build_v1_tree(tmp_path)
    manifest = json.loads((base / "pesn_irc_split.json").read_text())
    iso = _rid(4000)
    manifest["direct_scope"]["isolation_ledger"]["isolated_reaction_ids"] = [iso]
    manifest["subsets"]["train_dev"] = [_rid(5000 + i) for i in range(3)] + [iso]
    (base / "pesn_irc_split.json").write_text(json.dumps(manifest))
    sha = _sha(base / "pesn_irc_split.json")
    for sel_path in sorted(base.glob("selection_*.json")):
        sel = json.loads(sel_path.read_text())
        sel["split_manifest_sha256"] = sha
        sel_path.write_text(json.dumps(sel))
    payload = _load_payload(base)
    assert iso in payload["audit"]["isolated_but_partitioned_reaction_ids"]
    assert DualTSSplitLineage.from_payload(payload).partition_of(iso) == "monitor"


# ---------------------------------------------------------------------------
# Determinism and digest binding (rule d)
# ---------------------------------------------------------------------------


def test_determinism_same_inputs_same_artifact_digest(tmp_path):
    base = build_v1_tree(tmp_path)
    p1 = _load_payload(base)
    p2 = _load_payload(base)
    assert p1["artifact_sha256"] == p2["artifact_sha256"]
    assert artifact_digest(p1) == artifact_digest(p2)
    # only the timestamp may differ between two runs
    p1.pop("created_utc"), p2.pop("created_utc")
    assert stable_view(p1) == stable_view(p2)


def stable_view(payload: dict) -> str:
    view = {k: v for k, v in payload.items()
            if k not in ("created_utc", "artifact_sha256")}
    view["sources"] = [{"role": s["role"], "sha256": s["sha256"]}
                       for s in payload["sources"]]
    return json.dumps(view, sort_keys=True)


def test_digest_changes_when_input_changes(tmp_path):
    base = build_v1_tree(tmp_path)
    original = _load_payload(base)["artifact_sha256"]
    sel_path = base / "selection_direct_fit50.json"
    sel = json.loads(sel_path.read_text())
    sel["prefix_semantics"] = "tampered content"
    sel_path.write_text(json.dumps(sel))
    mutated = _load_payload(base)["artifact_sha256"]
    assert original != mutated


def test_source_digest_mismatch_is_refused_on_revalidation(tmp_path):
    base = build_v1_tree(tmp_path)
    out = write_lineage_artifact(_load_payload(base), tmp_path / "lin.json")
    payload = json.loads(out.read_text())
    validate_lineage_payload(payload, verify_sources_in=base)  # ok
    sel_path = base / "selection_direct_dev8.json"
    sel_path.write_text(json.dumps({}) )  # content destroyed
    with pytest.raises(DualTSSplitLineageError) as excinfo:
        validate_lineage_payload(payload, verify_sources_in=base)
    assert excinfo.value.code == "SOURCE_DIGEST_MISMATCH"


# ---------------------------------------------------------------------------
# Artifact round trip
# ---------------------------------------------------------------------------


def test_artifact_round_trip_and_tamper_detection(tmp_path):
    base = build_v1_tree(tmp_path)
    payload = _load_payload(base)
    out = write_lineage_artifact(payload, tmp_path / "nested" / "lin.json")
    reloaded = json.loads(out.read_text())
    validate_lineage_payload(reloaded)
    lineage = DualTSSplitLineage.from_payload(reloaded)
    assert lineage.partition_of(_rid(1000)) == "fit"
    assert lineage.wave_groups("fit") and lineage.wave_groups("monitor")
    # wave groups must inherit their role's partition
    assert set(lineage.wave_groups("fit")) <= set(
        payload["partitions"]["fit"]["group_ids"])
    # tampering: move one group between partitions -> digest mismatch
    tampered = json.loads(out.read_text())
    gid = tampered["partitions"]["monitor"]["group_ids"].pop()
    tampered["partitions"]["fit"]["group_ids"].append(gid)
    with pytest.raises(DualTSSplitLineageError) as excinfo:
        validate_lineage_payload(tampered)
    assert excinfo.value.code == "ARTIFACT_DIGEST_MISMATCH"
