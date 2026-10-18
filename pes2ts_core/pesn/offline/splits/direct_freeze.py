"""Direct-training split freeze (WP0 data split, plan v3 §2.3–§2.4).

Builds the direct fit/monitor/dev membership from the audited candidate pool
(``pes2ts_direct_training_input_audit_draft_v1`` fixture) with group closure
via ``duplicate_groups.json``, family closure against sealed families, and a
loud isolation ledger for mixed/missing family labels.

The output manifest keeps the ``pesn_irc_split_v1`` schema so the audited
truth bridge (:mod:`pesn.offline.data.truth_source`) can read train-side
members without any guard change: ``train_fit`` = ranked direct candidates,
``train_dev`` = direct monitor set, ``train_calibration`` = direct dev set.
Prefix selections (fit50/500/2000, monitor8, dev8, sac_fit12) are separate
frozen selection files; QC/preparation failures stay in their batch
denominator and are never silently replaced (plan §2.3).

This freeze is a Gate-D *input*: it records isolation and overlap evidence,
it does not by itself qualify any reaction for training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

from pes2ts_core.pesn.offline.splits.freeze import (
    load_parent_manifest_sha256,
)
from pes2ts_core.pesn.offline.splits.grouping import (
    GROUPING_VERSION,
    build_groups,
    groups_content_hash,
    load_canonical_members,
    sha256_text,
)
from pes2ts_core.utils.hashing import JSONValue, sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json

DIRECT_SPLIT_SCHEMA: Final[str] = "pesn_irc_split_v1"
SELECTION_SCHEMA: Final[str] = "pes2ts_direct_selection_v1"
SPLIT_AUDIT_SCHEMA: Final[str] = "pes2ts_direct_split_audit_v1"

class DirectSplitFreezeError(RuntimeError):
    """Typed freeze refusal (overlap, family conflict, malformed inputs)."""


def load_family_labels(
    interim_dir: str | Path, reaction_ids
) -> tuple[dict[str, str], list[str]]:
    """l0u family labels from G1 v2 class files; missing ids are reported."""
    classes_root = Path(interim_dir) / "g1_v2" / "classes"
    families: dict[str, str] = {}
    missing: list[str] = []
    for rid in sorted(set(reaction_ids)):
        path = classes_root / f"{int(rid.split('_')[-1]) // 1000:05d}" / f"{rid}.json"
        if not path.is_file():
            missing.append(rid)
            continue
        doc = read_json(path)
        family = None
        if isinstance(doc, dict):
            levels = doc.get("levels")
            if isinstance(levels, dict):
                entry = levels.get("l0u_undirected_family")
                if isinstance(entry, dict):
                    family = entry.get("cluster_id")
        if not family:
            missing.append(rid)
        else:
            families[rid] = str(family)
    return families, missing


def _group_id_map(groups: Mapping[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for gid, members in groups.items():
        for rid in members:
            out[rid] = gid
    return out


def _cohort_lists(cohorts_dir: str | Path) -> dict[str, list[str]]:
    """Real reaction-id lists from the frozen cohort files (audit inputs)."""
    base = Path(cohorts_dir)

    def _read(name: str) -> dict:
        return read_json(base / name)

    irc = _read("irc_fit_cohorts.json")
    rl = _read("rl_cohorts.json")
    sealed = _read("sealed_cohorts.json")
    replacements = _read("origin_gap_replacements.json")
    dev_extra = [str(x["reaction_id"])
                 for x in replacements["replacements"]["dev"]]
    return {
        "legacy_fit5000": [str(x) for x in irc["irc_fit_5000"]],
        "monitor250": [str(x) for x in irc["irc_dev_250"]],
        "dev_original200": [str(x) for x in rl["rl_dev_200"]],
        "dev_effective50": [str(x) for x in replacements["effective_dev50"]],
        "sealed_iid100": [str(x) for x in sealed["sealed_iid_100"]],
        "sealed_family100": [str(x) for x in sealed["sealed_family_100"]],
        "dev_exposure_union": (
            [str(x) for x in rl["rl_dev_200"]]
            + [str(x) for x in replacements["effective_dev50"]]
            + dev_extra
        ),
    }


def build_direct_split(
    interim_dir: str | Path,
    manifests_dir: str | Path,
    audit_fixture_path: str | Path,
    *,
    cohorts_dir: str | Path,
    seed: int = 42,
) -> dict[str, JSONValue]:
    """Validate and assemble the direct split manifest (in memory)."""
    fixture = read_json(Path(audit_fixture_path))
    if fixture.get("schema_version") != "pes2ts_direct_training_input_audit_draft_v1":
        raise DirectSplitFreezeError("audit fixture schema mismatch")
    if fixture.get("status") != "metadata_audited_not_training_qualified":
        raise DirectSplitFreezeError(
            "audit fixture status changed; re-audit before freezing a direct split"
        )

    legacy = _cohort_lists(cohorts_dir)
    ranked: list[str] = list(fixture["candidate_pool"]["reaction_ids_ranked"])
    monitor250: list[str] = list(legacy["monitor250"])
    dev50: list[str] = list(legacy["dev_effective50"])
    sealed = list(legacy["sealed_iid100"]) + list(legacy["sealed_family100"])
    exposure = list(legacy["dev_exposure_union"])

    canonical_members = load_canonical_members(interim_dir)
    # expand the universe with duplicate partners of every listed reaction so
    # group closure is computed on the real ledger, not on the listing
    ledger_by_canonical: dict[str, list[str]] = {}
    for member, canonical in canonical_members.items():
        ledger_by_canonical.setdefault(canonical, []).append(member)
    listed = set(ranked) | set(monitor250) | set(dev50) | set(sealed) | set(exposure)
    universe_ids = set(listed)
    for rid in sorted(listed):
        canonical = canonical_members.get(rid, rid)
        universe_ids.update(ledger_by_canonical.get(canonical, [rid]))
    universe = ranked + monitor250 + dev50 + sealed + exposure
    universe += sorted(universe_ids - set(universe))
    groups = build_groups(sorted(set(universe)), canonical_members)
    rid_to_group = _group_id_map(groups)

    def _gids(ids: list[str]) -> set[str]:
        return {rid_to_group[rid] for rid in ids if rid in rid_to_group}

    # --- hard overlap checks (reaction and group level) -------------------
    fit_set, mon_set, dev_set = set(ranked), set(monitor250), set(dev50)
    problems: list[str] = []
    for name, a, b in (
        ("fit__monitor", fit_set, mon_set),
        ("fit__dev", fit_set, dev_set),
        ("monitor__dev", mon_set, dev_set),
        ("fit__sealed", fit_set, set(sealed)),
        ("monitor__sealed", mon_set, set(sealed)),
        ("dev__sealed", dev_set, set(sealed)),
        ("fit__exposure", fit_set, set(exposure)),
    ):
        rxn_overlap = a & b
        if rxn_overlap:
            problems.append(f"{name}: {len(rxn_overlap)} reaction ids overlap")
        g_overlap = _gids(list(a)) & _gids(list(b))
        if g_overlap:
            problems.append(f"{name}: {len(g_overlap)} canonical groups overlap")
    if problems:
        raise DirectSplitFreezeError("; ".join(problems))

    # --- family closure (plan §2.4) ---------------------------------------
    # Whole-family sealing applies to the family holdout only; iid-sealed
    # reactions are individually sealed but their families legitimately
    # appear in fit (mirrors the audited pool construction).
    families, missing = load_family_labels(interim_dir, universe)
    sealed_families = {families[r] for r in legacy["sealed_family100"]
                       if r in families}
    family_conflicts = sorted(
        rid for rid in ranked
        if rid in families and families[rid] in sealed_families
    )
    if family_conflicts:
        # The audit pool already excludes these; a re-appearance is a freeze
        # stop, not a silent filter.
        raise DirectSplitFreezeError(
            f"{len(family_conflicts)} candidate reactions sit in sealed families "
            f"(e.g. {family_conflicts[:3]}); re-audit required"
        )

    # mixed-family groups must be isolated, not defaulted to normal training
    mixed_groups: list[str] = []
    isolated_reactions: set[str] = set()
    group_families: dict[str, set[str]] = {}
    for gid, members in groups.items():
        fams = {families[m] for m in members if m in families}
        group_families[gid] = fams
        if len(fams) > 1:
            mixed_groups.append(gid)
            isolated_reactions.update(members)
    missing_fit = sorted(rid for rid in ranked if rid not in families)
    # §2.4: groups without a resolvable family label are isolated from the
    # fit pool and recorded; they are not treated as ordinary train material.
    missing_groups = sorted({rid_to_group[r] for r in missing_fit if r in rid_to_group})
    isolated_reactions.update(missing_fit)

    # group closure: train_fit keeps every member of any group whose ranked
    # representative is fit-eligible (truth-bridge readability + §2.4 closure)
    ranked_groups = {rid_to_group[rid] for rid in ranked if rid in rid_to_group}
    closed = {member for gid in ranked_groups for member in groups.get(gid, [])}
    fit_members = [rid for rid in sorted(closed | set(ranked))
                   if rid not in isolated_reactions]

    parent_sha = load_parent_manifest_sha256(manifests_dir)
    manifest: dict[str, JSONValue] = {
        "schema_version": DIRECT_SPLIT_SCHEMA,
        "parent_split_sha256": parent_sha,
        "grouping_version": GROUPING_VERSION,
        "group_members_sha256": groups_content_hash(groups),
        "subsets": {
            "train_fit": fit_members,
            "train_dev": monitor250,
            "train_calibration": dev50,
        },
        "family_holdout": {
            "families": sorted(sealed_families),
            "reaction_ids": sorted(set(sealed) & set(rid_to_group)),
            "note": "sealed whole families; excluded from fit/monitor/dev",
        },
        "frozen": True,
        "known_group_cross_split_conflicts": 0,
        "direct_scope": {
            "source_audit": str(audit_fixture_path),
            "source_audit_sha256": sha256_file(Path(audit_fixture_path)),
            "seed": seed,
            "status": "gate_d_pending",
            "isolation_ledger": {
                "mixed_family_groups": sorted(mixed_groups),
                "missing_family_groups": missing_groups,
                "isolated_reaction_ids": sorted(isolated_reactions),
                "missing_family_labels_all_sets": missing,
            },
            "counts": {
                "train_fit": len(fit_members),
                "train_dev": len(monitor250),
                "train_calibration": len(dev50),
                "fit_canonical_groups": len(_gids(fit_members)),
                "sealed_families": len(sealed_families),
            },
            "semantics": (
                "train_fit = audited ranked direct candidates after isolation; "
                "prefix selections are separate frozen files; truth reads go "
                "through the existing pesn truth bridge against this split"
            ),
        },
    }
    return manifest


def freeze_direct_split(
    interim_dir: str | Path,
    manifests_dir: str | Path,
    audit_fixture_path: str | Path,
    output_dir: str | Path,
    *,
    cohorts_dir: str | Path,
    seed: int = 42,
) -> dict[str, Path]:
    """Write the manifest, selection files, and the Gate-D audit document."""
    manifest = build_direct_split(interim_dir, manifests_dir, audit_fixture_path,
                                  cohorts_dir=cohorts_dir, seed=seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    split_path = out / "pesn_irc_split.json"
    write_json(split_path, dict(manifest))

    fit_members: list[str] = list(manifest["subsets"]["train_fit"])
    monitor: list[str] = list(manifest["subsets"]["train_dev"])
    dev50: list[str] = list(manifest["subsets"]["train_calibration"])

    selections = {
        "direct_fit50": fit_members[:50],
        "direct_fit500": fit_members[:500],
        "direct_fit2000": fit_members[:2000],
        "direct_monitor8": monitor[:8],
        "direct_dev8": dev50[:8],
        "direct_sac_fit12": fit_members[:12],
    }
    written: dict[str, Path] = {"split": split_path}
    canonical_members = load_canonical_members(interim_dir)
    groups = build_groups(sorted(set(fit_members + monitor + dev50)), canonical_members)
    rid_to_group = _group_id_map(groups)
    for name, members in selections.items():
        doc = {
            "schema_version": SELECTION_SCHEMA,
            "selection": name,
            "reaction_ids": members,
            "group_ids": [rid_to_group[r] for r in members if r in rid_to_group],
            "n_reaction_ids": len(members),
            "n_canonical_groups": len({rid_to_group[r] for r in members if r in rid_to_group}),
            "split_manifest_sha256": sha256_file(split_path),
            "prefix_semantics": (
                "zero-based half-open prefix of the audited candidate ranking; "
                "QC/preparation failures remain in the batch denominator"
            ),
        }
        path = out / f"selection_{name}.json"
        write_json(path, doc)
        written[name] = path

    audit_doc = {
        "schema_version": SPLIT_AUDIT_SCHEMA,
        "split_manifest": "pesn_irc_split.json",
        "split_manifest_sha256": sha256_file(split_path),
        "checks": {
            "reaction_intersections": "empty for fit/monitor/dev/sealed/exposure pairs",
            "group_intersections": "empty for the same pairs",
            "family_intersections": "fit families disjoint from sealed families",
            "known_group_cross_split_conflicts": 0,
        },
        "isolation_ledger": manifest["direct_scope"]["isolation_ledger"],
        "gate_d_status": "evidence_recorded_pending_review",
        "note": (
            "Gate-D additionally requires shared-stepper 100% transfer "
            "verification and source/purpose review; this document is the "
            "membership evidence, not the gate signature"
        ),
    }
    audit_path = out / "split_audit.json"
    write_json(audit_path, audit_doc)
    written["split_audit"] = audit_path
    return written


def load_selection(split_dir: str | Path, name: str) -> list[str]:
    path = Path(split_dir) / f"selection_{name}.json"
    if not path.is_file():
        raise DirectSplitFreezeError(f"selection file missing: {path}")
    doc = read_json(path)
    if doc.get("schema_version") != SELECTION_SCHEMA or doc.get("selection") != name:
        raise DirectSplitFreezeError(f"selection file malformed: {path}")
    return [str(rid) for rid in doc["reaction_ids"]]


__all__ = [
    "DIRECT_SPLIT_SCHEMA",
    "DirectSplitFreezeError",
    "SELECTION_SCHEMA",
    "SPLIT_AUDIT_SCHEMA",
    "build_direct_split",
    "freeze_direct_split",
    "load_family_labels",
    "load_selection",
]
