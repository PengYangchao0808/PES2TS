"""Dual-TS split lineage (WP0b, plan v4 §E; contract doc v2 §1).

Inherits the historical v1 direct split membership (fit / monitor / test /
family holdout) into the dual-endpoint TS task, keyed by
``reaction_group_id`` (contract ``pesn_dual_endpoint_task_v2``), WITHOUT
re-randomizing anything.  Plan v4 §E: 历史split的组/族/暴露归属不因新任务
重置；不重新随机拆组.  Contract doc v2 §1: 历史split的组/族/暴露归属不因
新任务重置，split身份绑定真实内容摘要.

Rules enforced (each refusal is a typed :class:`DualTSSplitLineageError`):

(a) all branches/episodes of one ``reaction_group_id`` land in the same
    partition — a group whose members span partitions is a leakage refusal;
(b) family-level holdout is inherited: a group assigned to fit/monitor/test
    must not contain a reaction recorded in the v1 ``family_holdout``
    (mirrors the v1 sealed-family semantics);
(c) groups absent from the old split are NEVER silently assigned — they land
    in an explicit ``unassigned`` bucket with a registered reason;
(d) the lineage artifact is deterministic and content-digest-bound: it
    records the sha256 of every consumed input file and embeds its own
    content digest (timestamp excluded from the digest);
(e) historical exposure audit: per-partition group counts, branch
    potential (2 branches A/B per group, per contract), family exposure
    summary, and refusal when a v1 reaction appears under selections of
    more than one partition role.

v1 reaction-id → ``reaction_group_id`` binding: v1 canonical group ids ARE
canonical reaction ids (``pesn_grouping_v1`` semantics), so the explicit
binding table is materialized from (i) the aligned ``reaction_ids`` /
``group_ids`` pairs recorded in every v1 selection file (explicit evidence,
including real multi-member duplicate groups) and (ii) identity binding for
reactions without selection evidence.  Because v1 group closure kept every
member of a canonical group inside one subset, identity-bound partners
inherit the same partition as their group; the residual granularity
limitation is recorded verbatim in the artifact audit section.

Read-only on the v1 split files: this module only ever writes the new
lineage artifact.  stdlib + numpy only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Final, Mapping, Sequence

from pes2ts_core.utils.hashing import JSONValue, sha256_file, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json, write_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pes2ts_core.pesn.contracts.dual_ts import DualEndpointTask

# ---------------------------------------------------------------------------
# Frozen identity strings
# ---------------------------------------------------------------------------

#: Schema name of the lineage artifact (embedded in the file).
DUAL_TS_SPLIT_LINEAGE_SCHEMA: Final[str] = "pesn_dual_ts_split_lineage_v1"
#: Version of the tool that produced the artifact.
LINEAGE_TOOL_VERSION: Final[str] = "pes2ts_dual_ts_split_tool_v1"

#: The v1 split manifest schema consumed here.
V1_SPLIT_SCHEMA: Final[str] = "pesn_irc_split_v1"
#: The v1 prefix-selection schema consumed here.
V1_SELECTION_SCHEMA: Final[str] = "pes2ts_direct_selection_v1"

#: Partition roles inherited from the v1 split (frozen vocabulary).
PARTITION_ROLES: Final[tuple[str, ...]] = (
    "fit", "monitor", "test", "family_holdout",
)

#: v1 subset name -> inherited partition role.
V1_SUBSET_ROLE: Final[Mapping[str, str]] = {
    "train_fit": "fit",
    "train_dev": "monitor",
    "train_calibration": "test",
}

#: v1 selection name -> partition role (nested prefixes share the role).
SELECTION_ROLE: Final[Mapping[str, str]] = {
    "direct_fit50": "fit",
    "direct_fit500": "fit",
    "direct_fit2000": "fit",
    "direct_sac_fit12": "fit",
    "direct_monitor8": "monitor",
    "direct_dev8": "test",
}

#: The prefix selections the dual-TS v2 first wave trains on (plan v4 §E:
#: 先fit50双侧数据，再扩量).
WAVE_SELECTIONS: Final[Mapping[str, str]] = {
    "fit": "direct_fit50",
    "monitor": "direct_monitor8",
    "test": "direct_dev8",
}

#: Branches per reaction group: endpoints A and B (contract doc §1/§2 —
#: 完整A→TS与B→TS两条).  Episodes are >= branches and are not yet known at
#: lineage time, so only the branch potential is reported.
BRANCHES_PER_GROUP: Final[int] = 2
ENDPOINT_LABELS: Final[tuple[str, ...]] = ("A", "B")

#: Registered reasons for the explicit ``unassigned`` bucket (rule c).
UNASSIGNED_REASONS: Final[frozenset[str]] = frozenset({
    "v1_isolated_not_in_any_subset",   # v1 isolation ledger, no partition
    "v1_reaction_without_partition",   # present in v1 files, no subset role
    "absent_from_v1_split",            # group id never seen in v1 files
    "reaction_absent_from_v1_split",   # caller-supplied member id unknown
})

#: v1 isolation-ledger field consumed for unassigned classification.
ISOLATED_LEDGER_KEY: Final[str] = "isolated_reaction_ids"


class DualTSSplitLineageError(RuntimeError):
    """Typed lineage refusal (leakage, family conflict, malformed inputs).

    Mirrors the typed-refusal discipline of
    :class:`pes2ts_core.pesn.contracts.dual_ts.DualTSContractError`.
    """

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise DualTSSplitLineageError(code, detail)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iso_utc_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Input loading (strict, read-only)
# ---------------------------------------------------------------------------


def load_v1_split_manifest(path: str | Path) -> dict[str, JSONValue]:
    """Load and structurally validate the v1 split manifest (read-only)."""
    path = Path(path)
    _require(path.is_file(), "SPLIT_MANIFEST_MISSING", f"{path}")
    doc = read_json(path)
    _require(isinstance(doc, dict), "SPLIT_MANIFEST_MALFORMED", f"{path}")
    _require(doc.get("schema_version") == V1_SPLIT_SCHEMA,
             "SCHEMA_UNKNOWN",
             f"expected {V1_SPLIT_SCHEMA}, got {doc.get('schema_version')!r}")
    _require(doc.get("frozen") is True, "SPLIT_NOT_FROZEN",
             f"v1 manifest at {path} is not frozen")
    subsets = doc.get("subsets")
    _require(isinstance(subsets, dict) and set(subsets) == set(V1_SUBSET_ROLE),
             "SPLIT_MANIFEST_MALFORMED",
             f"subsets must be exactly {sorted(V1_SUBSET_ROLE)}")
    for name, members in subsets.items():
        _require(isinstance(members, list) and
                 all(isinstance(x, str) for x in members),
                 "SPLIT_MANIFEST_MALFORMED",
                 f"subset {name} must be a list of reaction ids")
    holdout = doc.get("family_holdout")
    _require(isinstance(holdout, dict)
             and isinstance(holdout.get("families"), list)
             and isinstance(holdout.get("reaction_ids"), list),
             "SPLIT_MANIFEST_MALFORMED",
             "family_holdout with families/reaction_ids required")
    return doc


def _selection_role(name: str) -> str:
    role = SELECTION_ROLE.get(name)
    _require(role is not None, "SELECTION_UNKNOWN",
             f"selection {name!r} is not a registered v1 prefix selection")
    return str(role)


def load_v1_selection(path: str | Path,
                      manifest_path: str | Path) -> dict[str, JSONValue]:
    """Load one v1 selection file and verify its bindings (read-only)."""
    path = Path(path)
    _require(path.is_file(), "SELECTION_MISSING", f"{path}")
    doc = read_json(path)
    _require(isinstance(doc, dict), "SELECTION_MALFORMED", f"{path}")
    _require(doc.get("schema_version") == V1_SELECTION_SCHEMA,
             "SCHEMA_UNKNOWN",
             f"expected {V1_SELECTION_SCHEMA} in {path.name}")
    name = doc.get("selection")
    _require(isinstance(name, str) and path.name == f"selection_{name}.json",
             "SELECTION_MALFORMED",
             f"selection name mismatch in {path.name}")
    _require(_selection_role(name) is not None, "SELECTION_UNKNOWN", name)
    rids, gids = doc.get("reaction_ids"), doc.get("group_ids")
    _require(isinstance(rids, list) and isinstance(gids, list)
             and len(rids) == len(gids)
             and all(isinstance(x, str) for x in rids + gids),
             "SELECTION_MALFORMED",
             f"{path.name}: reaction_ids/group_ids must be aligned str lists")
    _require(doc.get("n_reaction_ids") == len(rids)
             and doc.get("n_canonical_groups") == len(set(gids)),
             "SELECTION_MALFORMED",
             f"{path.name}: counts disagree with the id lists")
    recorded = doc.get("split_manifest_sha256")
    actual = sha256_file(manifest_path)
    _require(recorded == actual, "SELECTION_MANIFEST_DIGEST_MISMATCH",
             f"{path.name} records manifest digest {recorded} but the "
             f"manifest content digests to {actual}")
    return doc


def discover_v1_selections(v1_splits_dir: str | Path,
                           manifest_path: str | Path
                           ) -> dict[str, dict[str, JSONValue]]:
    """Load every ``selection_*.json`` next to the v1 manifest."""
    base = Path(v1_splits_dir)
    found = sorted(base.glob("selection_*.json"))
    _require(bool(found), "SELECTION_NONE_FOUND",
             f"no selection_*.json files under {base}")
    return {str(doc["selection"]): doc
            for doc in (
                load_v1_selection(p, manifest_path) for p in found)}


# ---------------------------------------------------------------------------
# Binding table: v1 reaction id -> reaction_group_id
# ---------------------------------------------------------------------------


def build_reaction_binding(
    selections: Mapping[str, Mapping[str, JSONValue]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Materialize the explicit ``reaction_id -> reaction_group_id`` table.

    Group ids come from the aligned selection pairs (v1 canonical group ids
    are canonical reaction ids).  Returns ``(binding, evidence)`` where
    ``evidence[rid]`` is ``"selection_evidenced"`` or
    ``"identity_no_selection_evidence"``; reactions without any selection
    evidence bind to themselves (v1 identity semantics).
    """
    binding: dict[str, str] = {}
    for name in sorted(selections):
        doc = selections[name]
        for rid, gid in zip(doc["reaction_ids"], doc["group_ids"]):  # type: ignore[arg-type]
            previous = binding.get(rid)
            if previous is not None and previous != gid:
                raise DualTSSplitLineageError(
                    "BINDING_CONFLICT",
                    f"reaction {rid} binds to groups {previous} and {gid} "
                    f"across selections")
            binding[rid] = gid
    # Identity fallback for reactions only seen inside the manifest subsets
    # is applied by the caller (it needs the manifest universe); here we
    # mark what selection evidence exists.
    evidence = {rid: "selection_evidenced" for rid in sorted(binding)}
    return binding, evidence


# ---------------------------------------------------------------------------
# Lineage construction
# ---------------------------------------------------------------------------


def _partition_per_reaction(
    manifest: Mapping[str, JSONValue],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """v1 reaction id -> (partition role, classification reason).

    Reactions inside ``family_holdout.reaction_ids`` that also sit in a
    subset are returned via the ``conflicts`` list (family-holdout rule b).
    """
    subsets = manifest["subsets"]
    role_of: dict[str, str] = {}
    reason_of: dict[str, str] = {}
    for subset_name, role in V1_SUBSET_ROLE.items():
        for rid in subsets[subset_name]:  # type: ignore[index]
            previous = role_of.get(rid)
            _require(previous is None, "SPLIT_MANIFEST_MALFORMED",
                     f"reaction {rid} appears in subsets {previous} and {role}")
            role_of[rid] = role
            reason_of[rid] = f"v1_subset:{subset_name}"
    conflicts: list[str] = []
    sealed = manifest["family_holdout"]["reaction_ids"]  # type: ignore[index]
    for rid in sealed:  # type: ignore[union-attr]
        if rid in role_of:
            conflicts.append(rid)
            continue
        role_of[rid] = "family_holdout"
        reason_of[rid] = "v1_family_holdout"
    return role_of, reason_of, sorted(conflicts)


def build_dual_ts_split_lineage(
    v1_splits_dir: str | Path,
    groups_source: str | Path | None = None,
    *,
    created_utc: str | None = None,
) -> dict[str, JSONValue]:
    """Build the dual-TS lineage artifact payload from the real v1 files.

    Read-only on *v1_splits_dir* and *groups_source*; returns the full
    artifact payload (including ``created_utc`` and ``artifact_sha256``).
    """
    base = Path(v1_splits_dir)
    manifest_path = base / "pesn_irc_split.json"
    manifest = load_v1_split_manifest(manifest_path)
    selections = discover_v1_selections(base, manifest_path)

    # --- rule (e): selection overlap across partition roles ----------------
    role_of_selection = {name: _selection_role(name) for name in selections}
    role_of_reaction: dict[str, set[str]] = {}
    for name, doc in selections.items():
        for rid in doc["reaction_ids"]:  # type: ignore[index]
            role_of_reaction.setdefault(rid, set()).add(role_of_selection[name])
    overlap = {rid: sorted(roles) for rid, roles in role_of_reaction.items()
               if len(roles) > 1}
    _require(not overlap, "SELECTION_PARTITION_OVERLAP",
             f"{len(overlap)} reactions appear under selections of more than "
             f"one partition role (e.g. {dict(list(sorted(overlap.items()))[:3])})")

    # --- selection reactions must sit in their role's v1 subset ------------
    subsets = manifest["subsets"]
    for name, doc in selections.items():
        role = role_of_selection[name]
        subset_name = next(k for k, v in V1_SUBSET_ROLE.items() if v == role)
        outside = sorted(set(doc["reaction_ids"])  # type: ignore[arg-type]
                         - set(subsets[subset_name]))  # type: ignore[index]
        _require(not outside, "SELECTION_SUBSET_MISMATCH",
                 f"{name}: {len(outside)} reactions are not in v1 subset "
                 f"{subset_name} (e.g. {outside[:3]})")

    # --- binding table ------------------------------------------------------
    binding, evidence = build_reaction_binding(selections)
    role_of, reason_of, family_conflicts = _partition_per_reaction(manifest)
    _require(not family_conflicts, "FAMILY_HOLDOUT_CONFLICT",
             f"{len(family_conflicts)} sealed reactions also sit in a v1 "
             f"subset (e.g. {family_conflicts[:3]})")

    # rule (c) input: isolated v1 reactions are part of the audited universe
    # even though they carry no partition role
    isolated = list(
        manifest["direct_scope"]["isolation_ledger"][ISOLATED_LEDGER_KEY])  # type: ignore[index]
    isolated_set = set(isolated)
    universe = sorted(set(role_of) | set(binding) | isolated_set)
    identity_bound: list[str] = []
    for rid in universe:
        if rid not in binding:
            binding[rid] = rid  # v1 identity: group id == canonical id
            evidence[rid] = "identity_no_selection_evidence"
            identity_bound.append(rid)

    # --- aggregate to reaction_group_id (rules a/b) --------------------------
    group_members: dict[str, list[str]] = {}
    for rid, gid in binding.items():
        group_members.setdefault(gid, []).append(rid)
    group_partition: dict[str, str] = {}
    for gid, members in group_members.items():
        roles = sorted({role_of[m] for m in members if m in role_of})
        if len(roles) > 1:
            raise DualTSSplitLineageError(
                "PARTITION_LEAKAGE",
                f"reaction_group {gid} spans v1 partitions {roles}; all "
                "branches/episodes of one group must share one partition")
        if roles:
            group_partition[gid] = roles[0]

    # rule (b): no fit/monitor/test group may contain a sealed reaction
    sealed = set(manifest["family_holdout"]["reaction_ids"])  # type: ignore[arg-type]
    family_exposure = {role: 0 for role in PARTITION_ROLES}
    for gid, members in group_members.items():
        hits = sorted(set(members) & sealed)
        if hits:
            family_exposure[group_partition[gid]] += 1
            _require(group_partition[gid] == "family_holdout",
                     "FAMILY_HOLDOUT_CONFLICT",
                     f"group {gid} inherits partition "
                     f"{group_partition[gid]} but contains sealed reactions "
                     f"{hits[:3]}")

    # rule (c): v1 reactions without any partition role are never assigned
    unassigned: list[dict[str, JSONValue]] = []

    def _unassigned_reason(members: Sequence[str]) -> str:
        if members and all(m in isolated_set for m in members):
            return "v1_isolated_not_in_any_subset"
        return "v1_reaction_without_partition"

    unassigned_groups = sorted(
        gid for gid, members in group_members.items()
        if all(role_of.get(m) not in PARTITION_ROLES for m in members))
    for gid in unassigned_groups:
        members = group_members[gid]
        unassigned.append({
            "reaction_group_id": gid,
            "reason": _unassigned_reason(members),
            "member_reaction_ids": sorted(members),
        })
    assigned_groups = dict(group_partition)

    # --- optional caller-supplied groups (rule c: explicit, never silent) ---
    source_groups: list[dict[str, JSONValue]] = []
    groups_source_doc: JSONValue | None = None
    if groups_source is not None:
        groups_source_path = Path(groups_source)
        _require(groups_source_path.is_file(), "GROUPS_SOURCE_MISSING",
                 f"{groups_source_path}")
        groups_source_doc = read_json(groups_source_path)
        if isinstance(groups_source_doc, dict):
            raw_ids = groups_source_doc.get("reaction_group_ids", [])
            raw_groups = groups_source_doc.get("groups", [])
            _require(isinstance(raw_ids, list) and isinstance(raw_groups, list),
                     "GROUPS_SOURCE_MALFORMED", f"{groups_source_path}")
            source_groups = [{"reaction_group_id": str(g),
                              "member_reaction_ids": []} for g in raw_ids]
            for entry in raw_groups:
                _require(isinstance(entry, dict)
                         and isinstance(entry.get("reaction_group_id"), str),
                         "GROUPS_SOURCE_MALFORMED", f"{groups_source_path}")
                source_groups.append({
                    "reaction_group_id": str(entry["reaction_group_id"]),
                    "member_reaction_ids": [str(m) for m in
                                            entry.get("member_reaction_ids", [])],
                })
        elif isinstance(groups_source_doc, list):
            source_groups = [{"reaction_group_id": str(g),
                              "member_reaction_ids": []}
                             for g in groups_source_doc]
        else:
            raise DualTSSplitLineageError(
                "GROUPS_SOURCE_MALFORMED", f"{groups_source_path}")

    for entry in source_groups:
        gid = str(entry["reaction_group_id"])
        members = list(entry["member_reaction_ids"])
        if not members and gid in group_members:
            members = group_members[gid]
        # a caller-supplied member must agree with the v1 binding evidence;
        # the cross-partition leakage check takes precedence (rule a)
        unknown = [m for m in members if m not in binding]
        member_roles = sorted({role_of[m] for m in members
                               if m in role_of and role_of[m] in PARTITION_ROLES})
        if len(member_roles) > 1:
            raise DualTSSplitLineageError(
                "PARTITION_LEAKAGE",
                f"reaction_group {gid} requested with members spanning "
                f"partitions {member_roles}")
        conflicting = sorted(m for m in members
                             if m in binding and binding[m] != gid)
        _require(not conflicting, "BINDING_CONFLICT",
                 f"group {gid}: caller-supplied members {conflicting[:3]} are "
                 "bound by v1 selection evidence to other groups")
        if member_roles:
            assigned_groups[gid] = member_roles[0]
            continue
        if not members:
            reason = "absent_from_v1_split"
        elif len(unknown) == len(members):
            reason = "reaction_absent_from_v1_split"
        else:
            reason = _unassigned_reason(members)
        unassigned.append({
            "reaction_group_id": gid,
            "reason": reason,
            "member_reaction_ids": sorted(set(members) | set(group_members.get(gid, []))),
        })
    # a caller-supplied group id must not be both assignable and unassigned
    double = sorted({str(e["reaction_group_id"]) for e in unassigned}
                    & set(assigned_groups))
    _require(not double, "BINDING_CONFLICT",
             f"groups {double[:3]} are simultaneously assignable and "
             "unassigned; groups-source contradicts the v1 evidence")
    deduped: list[dict[str, JSONValue]] = []
    for entry in sorted(unassigned, key=lambda e: str(e["reaction_group_id"])):
        gid = str(entry["reaction_group_id"])
        if gid in assigned_groups:
            continue
        if gid in {str(e["reaction_group_id"]) for e in deduped}:
            continue
        deduped.append(entry)
    unassigned = deduped

    # --- dual-TS first-wave selections (plan v4 §E: fit50 first) ------------
    wave: dict[str, JSONValue] = {}
    for role, sel_name in sorted(WAVE_SELECTIONS.items()):
        doc = selections[sel_name]
        gids = sorted(set(doc["group_ids"]))  # type: ignore[arg-type]
        mismatched = [g for g in gids if assigned_groups.get(g) != role]
        _require(not mismatched, "PARTITION_LEAKAGE",
                 f"wave selection {sel_name}: groups {mismatched[:3]} do not "
                 f"inherit partition {role}")
        wave[role] = {
            "selection": sel_name,
            "group_ids": gids,
            "n_groups": len(gids),
            "n_reaction_ids": len(doc["reaction_ids"]),  # type: ignore[arg-type]
        }
    nested = {name: len(sorted(set(selections[name]["group_ids"])))  # type: ignore[index]
              for name in sorted(selections)
              if name not in set(WAVE_SELECTIONS.values())}

    # --- audit counts (rule e) ----------------------------------------------
    partitions_payload: dict[str, JSONValue] = {}
    counts: dict[str, JSONValue] = {}
    for role in PARTITION_ROLES:
        gids = sorted(g for g, r in assigned_groups.items() if r == role)
        partitions_payload[role] = {
            "v1_source": next((k for k, v in V1_SUBSET_ROLE.items() if v == role),
                              "family_holdout"),
            "group_ids": gids,
            "n_groups": len(gids),
            "branch_potential": len(gids) * BRANCHES_PER_GROUP,
        }
        counts[f"n_{role}_groups"] = len(gids)
    counts["n_branches_potential"] = sum(
        partitions_payload[r]["branch_potential"]  # type: ignore[index]
        for r in PARTITION_ROLES)
    counts["n_unassigned_groups"] = len(unassigned)
    counts["n_groups_total"] = (
        sum(int(counts[f"n_{r}_groups"]) for r in PARTITION_ROLES)
        + len(unassigned))
    counts["n_reactions_bound"] = len(binding)
    counts["n_selection_evidenced_reactions"] = sum(
        1 for v in evidence.values() if v == "selection_evidenced")
    counts["n_identity_bound_reactions"] = len(identity_bound)

    isolated_but_partitioned = sorted(
        rid for rid in isolated
        if rid in role_of and role_of[rid] in PARTITION_ROLES)

    sealed_families = sorted(
        {str(f) for f in manifest["family_holdout"]["families"]})  # type: ignore[arg-type]
    sealed_sorted = sorted(sealed)
    sources = [{"role": "v1_split_manifest", "path": str(manifest_path),
                "sha256": sha256_file(manifest_path)}]
    for name in sorted(selections):
        sources.append({"role": f"v1_selection:{name}",
                        "path": str(base / f"selection_{name}.json"),
                        "sha256": sha256_file(base / f"selection_{name}.json")})
    if groups_source is not None:
        sources.append({"role": "groups_source", "path": str(groups_source),
                        "sha256": sha256_file(Path(groups_source))})

    payload: dict[str, JSONValue] = {
        "schema_version": DUAL_TS_SPLIT_LINEAGE_SCHEMA,
        "tool_version": LINEAGE_TOOL_VERSION,
        "created_utc": created_utc or _iso_utc_now(),
        "sources": sources,
        "v1_binding": {
            "binding_rule": (
                "reaction_group_id = v1 canonical group id (pesn_grouping_v1: "
                "group ids are canonical reaction ids); group ids taken from "
                "the aligned reaction_ids/group_ids pairs of the v1 selection "
                "files, identity binding for reactions without selection "
                "evidence; no re-randomization"),
            "grouping_version_inherited": manifest.get("grouping_version"),
            "v1_parent_split_sha256": manifest.get("parent_split_sha256"),
            "v1_group_members_sha256": manifest.get("group_members_sha256"),
            "v1_split_manifest_sha256": sha256_file(manifest_path),
            "reaction_to_group": dict(sorted(binding.items())),
            "reaction_binding_evidence": dict(sorted(evidence.items())),
        },
        "partitions": partitions_payload,
        "dual_ts_wave_selections": {
            "wave": wave,
            "nested_prefix_selections": nested,
            "note": ("first dual-TS wave trains on the frozen prefix "
                     "selections; wave groups must inherit the matching "
                     "v1 partition (verified above)"),
        },
        "family_audit": {
            "sealed_families": sealed_families,
            "n_sealed_families": len(sealed_families),
            "sealed_reaction_ids": sealed_sorted,
            "n_sealed_reactions": len(sealed_sorted),
            "group_exposure_per_partition": family_exposure,
            "conflicts": [],
            "note": (
                "per-reaction l0u family labels live in the G1 interim tree "
                "outside the v1 splits directory; exposure beyond the sealed "
                "list recorded in the v1 manifest is not computable from the "
                "consumed files"),
        },
        "unassigned": sorted(unassigned,
                             key=lambda e: str(e["reaction_group_id"])),
        "audit": {
            "selection_partition_overlap_conflicts": [],
            "group_partition_conflicts": [],
            "isolated_but_partitioned_reaction_ids": isolated_but_partitioned,
            "binding_limitations": (
                "duplicate-group partners without selection evidence bind by "
                "identity, so their dual group key is finer than the v1 "
                "canonical group; v1 group closure kept every partner inside "
                "one subset, therefore the inherited partition (rule a) is "
                "unaffected; the authoritative member ledger (duplicate_"
                "groups.json) lives outside the consumed files"),
            "counts": counts,
        },
        "artifact_sha256": "",
    }
    payload["artifact_sha256"] = artifact_digest(payload)
    return payload


# ---------------------------------------------------------------------------
# Digest, validation, round-trip
# ---------------------------------------------------------------------------


def artifact_digest(payload: Mapping[str, JSONValue]) -> str:
    """Content digest of the lineage artifact.

    SHA-256 over canonical JSON of the payload minus ``created_utc`` and
    ``artifact_sha256`` and minus source *paths* (roles + file digests stay
    in): same consumed content -> same digest, regardless of where the files
    live or when the artifact was written.
    """
    view = {k: v for k, v in payload.items()
            if k not in ("created_utc", "artifact_sha256")}
    view["sources"] = [{"role": s["role"], "sha256": s["sha256"]}
                       for s in payload["sources"]]  # type: ignore[index]
    return _sha256_text(stable_json_dumps(view))


def validate_lineage_payload(
    payload: Mapping[str, JSONValue],
    *,
    verify_sources_in: str | Path | None = None,
) -> None:
    """Round-trip validator: structure, internal consistency, digest.

    With ``verify_sources_in``, every recorded source digest is re-checked
    against the file on disk (content-digest-bound reopening).
    """
    _require(payload.get("schema_version") == DUAL_TS_SPLIT_LINEAGE_SCHEMA,
             "SCHEMA_UNKNOWN",
             f"expected {DUAL_TS_SPLIT_LINEAGE_SCHEMA}, "
             f"got {payload.get('schema_version')!r}")
    _require(isinstance(payload.get("tool_version"), str)
             and bool(payload["tool_version"]), "TOOL_VERSION_MISSING", "")
    recorded = payload.get("artifact_sha256")
    _require(isinstance(recorded, str) and len(recorded) == 64,
             "ARTIFACT_DIGEST_MISSING", "artifact_sha256 must be sha256 hex")
    actual = artifact_digest(dict(payload))
    _require(recorded == actual, "ARTIFACT_DIGEST_MISMATCH",
             f"recorded {recorded} != recomputed {actual}")

    sources = payload.get("sources")
    _require(isinstance(sources, list) and bool(sources)
             and all(isinstance(s, dict) and isinstance(s.get("role"), str)
                     and isinstance(s.get("path"), str)
                     and isinstance(s.get("sha256"), str) for s in sources),
             "SOURCES_MALFORMED", "sources entries malformed")
    if verify_sources_in is not None:
        src_dir = Path(verify_sources_in)
        for source in sources:  # type: ignore[union-attr]
            p = src_dir / Path(str(source["path"])).name
            _require(p.is_file(), "SOURCE_FILE_MISSING", f"{p}")
            _require(sha256_file(p) == source["sha256"],
                     "SOURCE_DIGEST_MISMATCH",
                     f"{p} content changed since the artifact was written")

    partitions = payload.get("partitions")
    _require(isinstance(partitions, dict)
             and set(partitions) == set(PARTITION_ROLES),
             "PARTITIONS_MALFORMED", f"partitions must be {list(PARTITION_ROLES)}")
    seen: dict[str, str] = {}
    for role, block in partitions.items():  # type: ignore[union-attr]
        gids = block.get("group_ids")  # type: ignore[union-attr]
        _require(isinstance(gids, list)
                 and all(isinstance(g, str) for g in gids),
                 "PARTITIONS_MALFORMED", f"{role}: group_ids must be str list")
        _require(block.get("n_groups") == len(gids)  # type: ignore[union-attr]
                 and block.get("branch_potential")
                 == len(gids) * BRANCHES_PER_GROUP,
                 "PARTITIONS_MALFORMED", f"{role}: counts inconsistent")
        for gid in gids:  # type: ignore[union-attr]
            _require(gid not in seen, "PARTITION_LEAKAGE",
                     f"group {gid} appears in partitions {seen.get(gid)!r} "
                     f"and {role!r}")
            seen[gid] = str(role)

    unassigned = payload.get("unassigned")
    _require(isinstance(unassigned, list), "UNASSIGNED_MALFORMED", "")
    for entry in unassigned:  # type: ignore[union-attr]
        gid = entry.get("reaction_group_id")  # type: ignore[union-attr]
        reason = entry.get("reason")  # type: ignore[union-attr]
        _require(isinstance(gid, str) and gid not in seen,
                 "PARTITION_LEAKAGE",
                 f"group {gid} is both partitioned and unassigned")
        _require(reason in UNASSIGNED_REASONS,
                 "UNASSIGNED_REASON_UNKNOWN",
                 f"reason {reason!r} is not registered")
        seen[gid] = "unassigned"  # type: ignore[assignment]

    binding = payload.get("v1_binding", {}).get("reaction_to_group", {})  # type: ignore[union-attr]
    _require(isinstance(binding, dict) and
             all(isinstance(v, str) for v in binding.values()),
             "BINDING_MALFORMED", "reaction_to_group must map str->str")
    bound_groups = set(binding.values())
    for gid in sorted(seen):
        _require(gid in bound_groups or seen[gid] == "unassigned",
                 "BINDING_INCOMPLETE",
                 f"group {gid} has no binding-table support")

    counts = payload.get("audit", {}).get("counts", {})  # type: ignore[union-attr]
    expected_total = sum(int(partitions[r]["n_groups"])  # type: ignore[index,union-attr]
                         for r in PARTITION_ROLES) + len(unassigned)  # type: ignore[arg-type]
    _require(counts.get("n_groups_total") == expected_total,
             "COUNTS_INCONSISTENT",
             f"n_groups_total {counts.get('n_groups_total')} != "
             f"{expected_total}")

    wave = payload.get("dual_ts_wave_selections", {}).get("wave", {})  # type: ignore[union-attr]
    _require(isinstance(wave, dict) and set(wave) == set(WAVE_SELECTIONS),
             "WAVE_MALFORMED", f"wave must cover {sorted(WAVE_SELECTIONS)}")
    for role, block in wave.items():  # type: ignore[union-attr]
        for gid in block.get("group_ids", []):  # type: ignore[union-attr]
            _require(seen.get(gid) == role, "WAVE_PARTITION_MISMATCH",
                     f"wave group {gid} does not inherit partition {role}")


def write_lineage_artifact(payload: Mapping[str, JSONValue],
                           out_path: str | Path) -> Path:
    """Write the lineage artifact (parents created); read-only elsewhere."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(payload))
    return out


# ---------------------------------------------------------------------------
# Facade: the lineage object consumed by later WPs
# ---------------------------------------------------------------------------


class DualTSSplitLineage:
    """Inherited dual-TS split membership keyed by ``reaction_group_id``.

    Construct from the real v1 files (:meth:`from_v1_splits`) or a written
    artifact (:meth:`from_payload`).  Partition lookups never re-randomize
    and never silently assign: unknown groups return ``None``.
    """

    def __init__(self, payload: Mapping[str, JSONValue]):
        validate_lineage_payload(payload)
        self._payload: dict[str, JSONValue] = dict(payload)
        partitions = payload["partitions"]
        self._partition_of: dict[str, str] = {}
        for role, block in partitions.items():  # type: ignore[union-attr]
            for gid in block["group_ids"]:  # type: ignore[index]
                self._partition_of[str(gid)] = str(role)
        self._unassigned = {
            str(e["reaction_group_id"]): str(e["reason"])
            for e in payload["unassigned"]}  # type: ignore[union-attr]

    @classmethod
    def from_v1_splits(cls,
                       v1_splits_dir: str | Path,
                       groups_source: str | Path | None = None,
                       ) -> "DualTSSplitLineage":
        payload = build_dual_ts_split_lineage(v1_splits_dir, groups_source)
        return cls(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, JSONValue]
                     ) -> "DualTSSplitLineage":
        return cls(payload)

    # -- lookups -------------------------------------------------------------
    def partition_of(self, reaction_group_id: str) -> str | None:
        """Inherited partition role, or ``None`` if unassigned/unknown."""
        return self._partition_of.get(str(reaction_group_id))

    def unassigned_reason_of(self, reaction_group_id: str) -> str | None:
        return self._unassigned.get(str(reaction_group_id))

    def partition_for_branch(self, reaction_group_id: str,
                             endpoint_label: str) -> str | None:
        """Partition of one branch (A/B) — branches inherit their group."""
        _require(endpoint_label in ENDPOINT_LABELS, "ENDPOINT_LABEL_INVALID",
                 f"endpoint_label {endpoint_label!r} is 'A' or 'B' only")
        return self.partition_of(reaction_group_id)

    def partition_for_task(self, task: "DualEndpointTask") -> str | None:
        """Partition for a contract task via its ``partition_key()``."""
        return self.partition_of(task.partition_key())

    # -- reporting -----------------------------------------------------------
    def partition_counts(self) -> dict[str, int]:
        counts = {role: 0 for role in PARTITION_ROLES}
        for role in self._partition_of.values():
            counts[role] += 1
        counts["unassigned"] = len(self._unassigned)
        return counts

    def branch_potential(self) -> dict[str, int]:
        return {role: n * BRANCHES_PER_GROUP
                for role, n in self.partition_counts().items()
                if role in PARTITION_ROLES}

    def unassigned_groups(self) -> list[dict[str, JSONValue]]:
        return list(self._payload["unassigned"])  # type: ignore[arg-type]

    def wave_groups(self, role: str) -> list[str]:
        block = self._payload["dual_ts_wave_selections"]["wave"][role]  # type: ignore[index]
        return [str(g) for g in block["group_ids"]]  # type: ignore[index]

    # -- persistence ----------------------------------------------------------
    def to_payload(self) -> dict[str, JSONValue]:
        return dict(self._payload)

    def write_artifact(self, out_path: str | Path) -> Path:
        return write_lineage_artifact(self._payload, out_path)


__all__ = [
    "BRANCHES_PER_GROUP",
    "DUAL_TS_SPLIT_LINEAGE_SCHEMA",
    "ENDPOINT_LABELS",
    "LINEAGE_TOOL_VERSION",
    "PARTITION_ROLES",
    "SELECTION_ROLE",
    "UNASSIGNED_REASONS",
    "V1_SELECTION_SCHEMA",
    "V1_SPLIT_SCHEMA",
    "V1_SUBSET_ROLE",
    "WAVE_SELECTIONS",
    "DualTSSplitLineage",
    "DualTSSplitLineageError",
    "artifact_digest",
    "build_dual_ts_split_lineage",
    "build_reaction_binding",
    "discover_v1_selections",
    "load_v1_selection",
    "load_v1_split_manifest",
    "validate_lineage_payload",
    "write_lineage_artifact",
]
