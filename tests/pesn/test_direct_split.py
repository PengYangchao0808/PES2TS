"""Direct split freeze tests (WP0 data split) on a synthetic interim tree."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pes2ts_core.pesn.offline.splits.direct_freeze import (
    DirectSplitFreezeError,
    freeze_direct_split,
    load_selection,
)


def _rid(n: int) -> str:
    return f"RXN_{n:010d}"


def build_tree(tmp_path: Path, *, n_candidates: int = 8, n_monitor: int = 3,
               n_dev: int = 2, n_sealed: int = 4, n_exposed: int = 2,
               duplicate_of=None, family_of=None, mixed_group=None):
    """Synthetic G0/G1 interim + audit fixture (metadata only, no truth)."""
    duplicate_of = duplicate_of or {}
    family_of = family_of or {}
    candidates = [_rid(1000 + i) for i in range(n_candidates)]
    monitor = [_rid(5000 + i) for i in range(n_monitor)]
    dev = [_rid(7000 + i) for i in range(n_dev)]
    sealed = [_rid(9000 + i) for i in range(n_sealed)]
    exposed = [_rid(3000 + i) for i in range(n_exposed)]

    interim = tmp_path / "interim"
    interim.mkdir()
    # duplicate groups: one pair shares a canonical group (ledger format)
    pair = {"canonical_reaction_id": candidates[0],
            "member_reaction_ids": [candidates[0], candidates[1]],
            "identity": "testidentity"}
    (interim / "duplicate_groups.json").write_text(
        json.dumps({"schema_version": "g0_duplicate_groups_v1", "groups": [pair]}))
    # family class files (sealed reactions get dedicated families so the
    # synthetic pool is family-disjoint from the sealed sets by default;
    # the duplicate pair shares one family as canonical duplicates should)
    classes = interim / "g1_v2" / "classes"
    all_ids = candidates + monitor + dev + sealed + exposed
    fam = dict(family_of)
    for i, rid in enumerate(all_ids):
        if rid in sealed:
            fam.setdefault(rid, f"FSEAL{sealed.index(rid)}")
        elif rid == candidates[1]:
            fam.setdefault(rid, "FAM00")
        else:
            fam.setdefault(rid, f"FAM{i % 3:02d}")
    if mixed_group:
        fam[mixed_group] = "FAM_SPECIAL"
    for rid, family in fam.items():
        bucket = classes / f"{int(rid.split('_')[-1]) // 1000:05d}"
        bucket.mkdir(parents=True, exist_ok=True)
        (bucket / f"{rid}.json").write_text(json.dumps(
            {"levels": {"l0u_undirected_family": {"cluster_id": family}}}))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "split_manifest.json").write_text("{}")

    ranked = sorted(set(candidates) - {candidates[1]})
    fixture = {
        "schema_version": "pes2ts_direct_training_input_audit_draft_v1",
        "status": "metadata_audited_not_training_qualified",
        "legacy_counts": {
            "monitor250": {"reaction_ids": monitor},
            "dev_effective50": {"reaction_ids": dev},
            "sealed_iid100": {"reaction_ids": sealed[:2]},
            "sealed_family100": {"reaction_ids": sealed[2:]},
            "dev_exposure_union": {"reaction_ids": exposed},
        },
        "candidate_pool": {
            "n_reaction_ids": len(ranked), "n_independent_groups": len(ranked),
            "excluded_reaction_ids": [], "family_overlap_exclusions": [],
            "representative_rule": "test",
            "reaction_ids_ranked": ranked,
        },
    }
    fixture_path = tmp_path / "audit.json"
    fixture_path.write_text(json.dumps(fixture))
    return interim, manifests, fixture_path, candidates, monitor, dev, sealed


def _write_cohorts(tmp_path: Path, candidates, monitor, dev, sealed) -> Path:
    """Cohort files mirroring the audit script inputs (id lists)."""
    exposed = ["RXN_0000300000", "RXN_0000300001"]
    cohorts = tmp_path / "cohorts"
    cohorts.mkdir()
    (cohorts / "irc_fit_cohorts.json").write_text(json.dumps({
        "irc_fit_5000": sorted(set(candidates) - {candidates[1]}),
        "irc_fit_2000_verified": sorted(set(candidates) - {candidates[1]})[:4],
        "irc_dev_250": monitor,
    }))
    (cohorts / "rl_cohorts.json").write_text(json.dumps({
        "rl_dev_200": exposed + ["RXN_0000300099"],
    }))
    (cohorts / "sealed_cohorts.json").write_text(json.dumps({
        "sealed_iid_100": sealed[:2], "sealed_family_100": sealed[2:],
    }))
    (cohorts / "origin_gap_replacements.json").write_text(json.dumps({
        "effective_dev50": dev,
        "replacements": {"dev": [{"reaction_id": "RXN_0000300099"}]},
    }))
    return cohorts


def test_freeze_writes_manifest_selections_and_audit(tmp_path):
    interim, manifests, fixture, candidates, monitor, dev, sealed = build_tree(tmp_path)
    out = tmp_path / "splits"
    cohorts = _write_cohorts(tmp_path, candidates, monitor, dev, sealed)
    written = freeze_direct_split(interim, manifests, fixture, out,
                                  cohorts_dir=cohorts, seed=42)
    manifest = json.loads((out / "pesn_irc_split.json").read_text())
    assert manifest["schema_version"] == "pesn_irc_split_v1"
    assert manifest["frozen"] is True
    assert manifest["known_group_cross_split_conflicts"] == 0
    # duplicate partner stays with its representative (group closure)
    fit = set(manifest["subsets"]["train_fit"])
    assert candidates[0] in fit and candidates[1] in fit
    assert manifest["subsets"]["train_dev"] == monitor
    assert manifest["subsets"]["train_calibration"] == dev
    # sealed never leak into any subset
    for s in manifest["subsets"].values():
        assert not (set(s) & set(sealed))
    audit = json.loads((out / "split_audit.json").read_text())
    assert audit["checks"]["known_group_cross_split_conflicts"] == 0
    # selection prefixes
    fit50 = load_selection(out, "direct_fit50")
    assert fit50  # small synthetic pool: all fit members
    sel = json.loads(written["direct_fit50"].read_text())
    assert sel["schema_version"] == "pes2ts_direct_selection_v1"


def test_selection_prefixes_are_nested_and_deterministic(tmp_path):
    interim, manifests, fixture, candidates, monitor, dev, sealed = build_tree(tmp_path)
    cohorts = _write_cohorts(tmp_path, candidates, monitor, dev, sealed)
    out = tmp_path / "splits"
    freeze_direct_split(interim, manifests, fixture, out, cohorts_dir=cohorts, seed=42)
    fit50 = load_selection(out, "direct_fit50")
    fit500 = load_selection(out, "direct_fit500")
    fit2000 = load_selection(out, "direct_fit2000")
    sac12 = load_selection(out, "direct_sac_fit12")
    assert fit50 == fit500[:len(fit50)]
    assert fit500 == fit2000[:len(fit500)]
    assert sac12 == fit50[:len(sac12)]
    out2 = tmp_path / "splits2"
    freeze_direct_split(interim, manifests, fixture, out2, cohorts_dir=cohorts, seed=42)
    assert load_selection(out2, "direct_fit50") == fit50


def test_overlap_refusal(tmp_path):
    interim, manifests, fixture, candidates, monitor, dev, sealed = build_tree(tmp_path)
    cohorts = _write_cohorts(tmp_path, candidates, monitor, dev, sealed)
    # inject a fit candidate into the dev set via the cohort file
    reps = json.loads((cohorts / "origin_gap_replacements.json").read_text())
    reps["effective_dev50"] = [candidates[0]]
    (cohorts / "origin_gap_replacements.json").write_text(json.dumps(reps))
    with pytest.raises(DirectSplitFreezeError):
        freeze_direct_split(interim, manifests, fixture, tmp_path / "s",
                            cohorts_dir=cohorts, seed=42)


def test_family_conflict_refusal(tmp_path):
    # a candidate sharing a family with sealed reactions must stop the freeze
    interim, manifests, fixture, candidates, monitor, dev, sealed = build_tree(tmp_path)
    fixture_doc = json.loads(fixture.read_text())
    family_of = {candidates[2]: "FAMSEALED", sealed[2]: "FAMSEALED"}
    # rewrite both class files with the conflicting shared family
    classes = interim / "g1_v2" / "classes"
    for rid in (candidates[2], sealed[2]):
        bucket = classes / f"{int(rid.split('_')[-1]) // 1000:05d}"
        (bucket / f"{rid}.json").write_text(json.dumps(
            {"levels": {"l0u_undirected_family": {"cluster_id": "FAMSEALED"}}}))
    cohorts = _write_cohorts(tmp_path, candidates, monitor, dev, sealed)
    with pytest.raises(DirectSplitFreezeError):
        freeze_direct_split(interim, manifests, fixture, tmp_path / "s",
                            cohorts_dir=cohorts, seed=42)


def test_missing_family_labels_are_isolated_not_defaulted(tmp_path):
    interim, manifests, fixture, candidates, monitor, dev, sealed = build_tree(tmp_path)
    # drop the class file of one candidate
    bucket = interim / "g1_v2" / "classes" / f"{int(candidates[3].split('_')[-1]) // 1000:05d}"
    (bucket / f"{candidates[3]}.json").unlink()
    cohorts = _write_cohorts(tmp_path, candidates, monitor, dev, sealed)
    out = tmp_path / "splits"
    freeze_direct_split(interim, manifests, fixture, out, cohorts_dir=cohorts, seed=42)
    manifest = json.loads((out / "pesn_irc_split.json").read_text())
    assert candidates[3] not in set(manifest["subsets"]["train_fit"])
    ledger = manifest["direct_scope"]["isolation_ledger"]
    assert candidates[3] in ledger["isolated_reaction_ids"]
