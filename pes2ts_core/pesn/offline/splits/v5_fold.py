"""V5 sprint split freeze: fit50 group-level 5-fold + sealed candidates.

Implements ADR-0026 §2.4 and sprint plan §1.4 for the V5 sprint window:

* the 50 direct-fit groups become 5 folds (~10 groups each, group-closed —
  every frame of both branches of a group lands in the same fold), stratified
  by reaction family and group size, assigned by a seeded shuffle +
  round-robin within strata (seed=20261018);
* duplicate / reverse / mapping-copy groups are merged with union-find over
  the G0 ``duplicate_groups.json`` ledger and the dual-TS lineage identity
  bindings; every merged component must land in ONE fold (asserted);
* Confirm candidates (≥30 groups from the direct fit pool minus fit50) and
  OOD candidates (≥3 families × ≥10 groups from the lineage ``family_holdout``)
  are registered SEALED — this module only records them, never opens them;
* frozen before any V5 label is inspected; assertions: pairwise-disjoint fold
  group sets, one fold per group, parent sha256 chain (fit50 selection,
  lineage, v1 split manifest) and the historical development-exclusion
  sha256.

The module writes ``data/manifests/v5/split_manifest.json`` (schema
``pesn_v5_split_v1``) only via ``main --write``; the library function returns
the manifest dict in memory so callers/tests can verify without touching the
frozen tree.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Iterable, Mapping

from pes2ts_core.pesn.contracts.key_atom import SPLIT_SCHEMA as V5_SPLIT_SCHEMA
from pes2ts_core.pesn.offline.splits.direct_freeze import load_family_labels
from pes2ts_core.pesn.offline.splits.grouping import sha256_text
from pes2ts_core.utils.hashing import JSONValue, sha256_file, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json, write_json

V5_SPLIT_VERSION: Final[str] = "pes2ts_v5_fold_tool_v1"
DEFAULT_SEED: Final[int] = 20261018
N_FOLDS: Final[int] = 5
CONFIRM_MIN_GROUPS: Final[int] = 30
OOD_MIN_FAMILIES: Final[int] = 3
OOD_MIN_GROUPS_PER_FAMILY: Final[int] = 10


class V5SplitError(RuntimeError):
    """Typed split-freeze refusal (malformed inputs, assertion failure)."""


# ---------------------------------------------------------------------------
# union-find over identity bindings
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        parent = self._parent.setdefault(x, x)
        if parent != x:
            parent = self.find(parent)
            self._parent[x] = parent
        return parent

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: lexicographic min
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo


def merge_identity_components(
    fit_group_ids: Iterable[str],
    *,
    canonical_members: Mapping[str, str],
    lineage_reaction_to_group: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Union fit groups bound to the same identity; return root -> members.

    Bindings used (plan §1.4): G0 duplicate ledger (forward/reverse copies
    share one canonical id) and the dual-TS lineage ``reaction_to_group``
    (v1 canonical group binding).  Groups with no binding stay singletons.
    """
    uf = _UnionFind()
    groups = sorted(set(fit_group_ids))
    for gid in groups:
        uf.find(gid)
    # duplicate-ledger binding: two fit groups that are ledger members of the
    # same canonical reaction id are copies of one reaction
    by_canonical: dict[str, list[str]] = {}
    for gid in groups:
        canonical = canonical_members.get(gid)
        if canonical is not None:
            by_canonical.setdefault(canonical, []).append(gid)
    for members in by_canonical.values():
        for other in members[1:]:
            uf.union(members[0], other)
    # lineage binding: fit group ids that resolve to one v1 group
    by_lineage: dict[str, list[str]] = {}
    for gid in groups:
        v1_group = lineage_reaction_to_group.get(gid)
        if v1_group is not None:
            by_lineage.setdefault(v1_group, []).append(gid)
    for members in by_lineage.values():
        for other in members[1:]:
            uf.union(members[0], other)
    components: dict[str, list[str]] = {}
    for gid in groups:
        components.setdefault(uf.find(gid), []).append(gid)
    root_of = {gid: uf.find(gid) for gid in groups}
    return root_of, components


# ---------------------------------------------------------------------------
# stratified round-robin fold assignment
# ---------------------------------------------------------------------------


def _rank(seed: int, key: str) -> int:
    return int(sha256_text(f"{seed}:{key}")[:16], 16)


def _size_bucket(n_members: int) -> str:
    if n_members <= 1:
        return "singleton"
    if n_members <= 3:
        return "small"
    return "large"


def assign_folds(
    components: Mapping[str, list[str]],
    family_of_component: Mapping[str, str],
    members_per_component: Mapping[str, int],
    *,
    seed: int,
    n_folds: int = N_FOLDS,
) -> dict[str, int]:
    """Seeded shuffle + stratified greedy assignment to folds 0..n_folds-1.

    Components (identity-closed groups) are the assignment units, so a
    merged duplicate component can never straddle two folds.  Within each
    (family, size) stratum components are ranked by
    ``sha256(f"{seed}:{root}")``; processing strata in sorted order and
    assigning each component to the currently smallest fold (ties broken by
    fold index) keeps the assignment seeded while balancing fold sizes — a
    plain per-stratum round-robin would restart at fold 0 every time and
    systematically overload the early folds.
    """
    strata: dict[tuple[str, str], list[str]] = {}
    for root in sorted(components):
        key = (
            str(family_of_component.get(root, "unknown")),
            _size_bucket(int(members_per_component.get(root, 1))),
        )
        strata.setdefault(key, []).append(root)
    fold_of: dict[str, int] = {}
    fold_sizes = [0] * n_folds
    for stratum_key in sorted(strata):
        ranked = sorted(
            strata[stratum_key],
            key=lambda root: (_rank(seed, root), root),
        )
        for root in ranked:
            fold = min(range(n_folds), key=lambda k: (fold_sizes[k], k))
            fold_of[root] = fold
            fold_sizes[fold] += int(members_per_component.get(root, 1))
    return fold_of


# ---------------------------------------------------------------------------
# sealed candidate pools (registered, never opened)
# ---------------------------------------------------------------------------


def select_confirm_candidates(
    fit_pool_group_ids: Iterable[str],
    fit50_group_ids: Iterable[str],
    family_of_group: Mapping[str, str],
    *,
    seed: int,
    target: int = 40,
    minimum: int = CONFIRM_MIN_GROUPS,
) -> list[str]:
    """Family-stratified sealed confirm candidates from the fit pool.

    Pool = direct fit groups minus the 50 development-exposed groups;
    round-robin across families (seeded ranking inside each) until ``target``
    groups are collected.  Raises when the pool cannot reach ``minimum``.
    """
    excluded = set(fit50_group_ids)
    pool = sorted(set(fit_pool_group_ids) - excluded)
    by_family: dict[str, list[str]] = {}
    for gid in pool:
        by_family.setdefault(family_of_group.get(gid, "unknown"), []).append(gid)
    for family in by_family:
        by_family[family].sort(key=lambda gid: (_rank(seed, gid), gid))
    families = sorted(by_family)
    picked: list[str] = []
    idx = {f: 0 for f in families}
    while len(picked) < target:
        progressed = False
        for family in families:
            i = idx[family]
            if i < len(by_family[family]):
                picked.append(by_family[family][i])
                idx[family] = i + 1
                progressed = True
                if len(picked) >= target:
                    break
        if not progressed:
            break
    if len(picked) < minimum:
        raise V5SplitError(
            f"confirm pool insufficient: {len(picked)} < {minimum} groups "
            "(insufficient_evidence; do not freeze a confirm claim)")
    return picked


def select_ood_candidates(
    holdout_group_ids: Iterable[str],
    family_of_group: Mapping[str, str],
    *,
    seed: int,
    min_families: int = OOD_MIN_FAMILIES,
    groups_per_family: int = OOD_MIN_GROUPS_PER_FAMILY,
) -> dict[str, list[str]]:
    """Sealed OOD candidates: ≥3 holdout families × ≥10 groups each.

    Families are ranked by ``sha256(f"{seed}:{family}")`` and the top
    qualifying families (each with ≥ ``groups_per_family`` holdout groups)
    contribute their seeded top-``groups_per_family`` groups.
    """
    by_family: dict[str, list[str]] = {}
    for gid in sorted(set(holdout_group_ids)):
        family = family_of_group.get(gid)
        if family is None:
            continue  # family_holdout members without labels are not usable
        by_family.setdefault(family, []).append(gid)
    qualifying = [
        family for family, members in by_family.items()
        if len(members) >= groups_per_family
    ]
    qualifying.sort(key=lambda f: (_rank(seed, f), f))
    if len(qualifying) < min_families:
        raise V5SplitError(
            f"OOD pool insufficient: {len(qualifying)} families with "
            f">={groups_per_family} groups (need {min_families}); "
            "insufficient_evidence — record, do not claim generalization")
    out: dict[str, list[str]] = {}
    for family in qualifying:
        ranked = sorted(
            by_family[family], key=lambda gid: (_rank(seed, gid), gid))
        out[family] = ranked[:groups_per_family]
    return out


# ---------------------------------------------------------------------------
# manifest builder
# ---------------------------------------------------------------------------


def build_v5_split(
    fit50_selection_path: str | Path,
    lineage_path: str | Path,
    seed: int = DEFAULT_SEED,
    *,
    interim_dir: str | Path = "data/interim",
    v1_split_manifest_path: str | Path | None = (
        "outputs/pesn/direct_egnn_v1_20261015/splits/pesn_irc_split.json"),
    parent_manifest_path: str | Path = "data/manifests/split_manifest.json",
) -> dict[str, JSONValue]:
    """Build the V5 split manifest (in memory; ADR-0026 §2.4).

    Inputs are the frozen direct fit50 selection and the dual-TS split
    lineage; family labels come from the G1 v2 class files under
    *interim_dir* and duplicate bindings from ``interim_dir/duplicate_groups.json``.
    """
    selection = read_json(Path(fit50_selection_path))
    if selection.get("schema_version") != "pes2ts_direct_selection_v1":
        raise V5SplitError("fit50 selection schema mismatch")
    fit50_groups = [str(g) for g in selection.get("group_ids") or []]
    if len(fit50_groups) != 50:
        raise V5SplitError(
            f"fit50 selection must list 50 groups, got {len(fit50_groups)}")
    fit50_reactions = [str(r) for r in selection.get("reaction_ids")
                       if isinstance(r, str)]

    lineage = read_json(Path(lineage_path))
    if lineage.get("schema_version") != "pesn_dual_ts_split_lineage_v1":
        raise V5SplitError("lineage schema mismatch")
    partitions = lineage.get("partitions") or {}
    fit_pool = [str(g) for g in (partitions.get("fit") or {}).get("group_ids") or []]
    holdout = [str(g) for g in
               (partitions.get("family_holdout") or {}).get("group_ids") or []]
    family_audit = lineage.get("family_audit") or {}
    sealed_families = [str(f) for f in family_audit.get("sealed_families") or []]
    sealed_reactions = [str(r) for r in
                        family_audit.get("sealed_reaction_ids") or []]
    lineage_binding = (lineage.get("v1_binding") or {}).get(
        "reaction_to_group") or {}
    lineage_reaction_to_group = {str(k): str(v)
                                 for k, v in lineage_binding.items()}

    # family labels for every reaction we stratify over
    universe = sorted(
        set(fit50_reactions)
        | set(fit50_groups)
        | set(fit_pool)
        | set(holdout))
    families, missing = load_family_labels(interim_dir, universe)
    family_of_group: dict[str, str] = {}
    votes_missing = 0
    for gid in sorted(set(fit50_groups) | set(fit_pool) | set(holdout)):
        family = families.get(gid)
        if family is None:
            votes_missing += 1
            continue
        family_of_group[gid] = family

    # identity closure over the 50 fit groups
    canonical_members: dict[str, str] = {}
    dup_path = Path(interim_dir) / "duplicate_groups.json"
    if dup_path.is_file():
        ledger = read_json(dup_path)
        for group in (ledger.get("groups") or []):
            canonical = group.get("canonical_reaction_id")
            for member in group.get("member_reaction_ids") or []:
                canonical_members[str(member)] = str(canonical)
    root_of, components = merge_identity_components(
        fit50_groups,
        canonical_members=canonical_members,
        lineage_reaction_to_group=lineage_reaction_to_group,
    )

    # component descriptors for stratification
    members_per_component = {
        root: len(members) for root, members in components.items()}
    family_of_component: dict[str, str] = {}
    for root, members in components.items():
        votes: dict[str, int] = {}
        for gid in members:
            family = family_of_group.get(gid)
            if family is not None:
                votes[family] = votes.get(family, 0) + 1
        if votes:
            family_of_component[root] = sorted(
                votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        else:
            family_of_component[root] = "unknown"

    fold_of_component = assign_folds(
        components, family_of_component, members_per_component, seed=seed)

    folds: dict[str, list[str]] = {str(k): [] for k in range(N_FOLDS)}
    group_fold: dict[str, int] = {}
    for root, members in components.items():
        fold = fold_of_component[root]
        for gid in sorted(members):
            folds[str(fold)].append(gid)
            group_fold[gid] = fold
    for fold in folds:
        folds[fold].sort()

    # --- assertions (plan §1.4 freeze gates) ------------------------------
    all_groups = [g for fold in folds for g in folds[fold]]
    if sorted(all_groups) != sorted(fit50_groups):
        raise V5SplitError("fold union != fit50 group set")
    if len(set(all_groups)) != len(all_groups):
        raise V5SplitError("a group appears in more than one fold")
    for root, members in components.items():
        folds_seen = {group_fold[gid] for gid in members}
        if len(folds_seen) != 1:
            raise V5SplitError(
                f"identity component {members} straddles folds {folds_seen}")
    # pairwise-disjoint by construction above; explicit double check
    fold_sets = [set(folds[str(k)]) for k in range(N_FOLDS)]
    for i in range(N_FOLDS):
        for j in range(i + 1, N_FOLDS):
            overlap = fold_sets[i] & fold_sets[j]
            if overlap:
                raise V5SplitError(
                    f"folds {i}/{j} overlap on {sorted(overlap)[:3]}")

    # sealed candidate pools
    confirm = select_confirm_candidates(
        fit_pool, fit50_groups, family_of_group, seed=seed)
    ood = select_ood_candidates(holdout, family_of_group, seed=seed)
    ood_reactions = sorted({gid for members in ood.values() for gid in members})

    # parent hashes
    parents: dict[str, JSONValue] = {
        "fit50_selection": {
            "path": str(fit50_selection_path),
            "sha256": sha256_file(fit50_selection_path),
        },
        "dual_ts_lineage": {
            "path": str(lineage_path),
            "sha256": sha256_file(lineage_path),
        },
    }
    if v1_split_manifest_path is not None and Path(v1_split_manifest_path).is_file():
        parents["v1_split_manifest"] = {
            "path": str(v1_split_manifest_path),
            "sha256": sha256_file(v1_split_manifest_path),
        }
    if parent_manifest_path is not None and Path(parent_manifest_path).is_file():
        parents["g0_parent_split_manifest"] = {
            "path": str(parent_manifest_path),
            "sha256": sha256_file(parent_manifest_path),
        }

    # historical development-exclusion hash: the frozen sealed lists the V5
    # window inherits (lineage family holdout + sealed families).  This is
    # the recorded exclusion ledger the sprint must not re-open.
    history_exclusion = {
        "sealed_families": sorted(sealed_families),
        "sealed_reaction_ids": sorted(sealed_reactions),
        "source": "dual_ts_split_lineage_v1.family_audit",
    }
    history_exclusion_sha256 = sha256_text(stable_json_dumps(
        history_exclusion))  # type: ignore[arg-type]

    manifest: dict[str, JSONValue] = {
        "schema_version": V5_SPLIT_SCHEMA,
        "tool_version": V5_SPLIT_VERSION,
        "seed": int(seed),
        "n_groups": len(fit50_groups),
        "n_folds": N_FOLDS,
        "folds": folds,
        "group_fold": {str(k): int(v) for k, v in sorted(group_fold.items())},
        "components": [
            {
                "root": root,
                "group_ids": sorted(members),
                "family": family_of_component.get(root, "unknown"),
                "size_bucket": _size_bucket(len(members)),
                "binding": "duplicate_ledger+lineage_reaction_to_group",
            }
            for root, members in sorted(components.items())
        ],
        "confirm_candidates": {
            "group_ids": sorted(confirm),
            "n_groups": len(confirm),
            "family_counts": {
                family: sum(1 for gid in confirm
                            if family_of_group.get(gid, "unknown") == family)
                for family in sorted({family_of_group.get(g, "unknown")
                                      for g in confirm})
            },
            "sealed": True,
            "note": ("registered before any V5 label inspection; this round "
                     "does not open confirm (plan §1.4)"),
        },
        "ood_candidates": {
            "families": {family: sorted(members)
                         for family, members in sorted(ood.items())},
            "reaction_ids": ood_reactions,
            "n_families": len(ood),
            "n_groups": sum(len(v) for v in ood.values()),
            "sealed": True,
            "note": ("family_holdout families frozen as OOD candidates; "
                     "opening them is out of scope for this sprint"),
        },
        "parents": parents,
        "history_exclusion": history_exclusion,
        "history_exclusion_sha256": history_exclusion_sha256,
        "family_label_missing_groups": votes_missing,
        "missing_family_label_reactions": missing[:50],
        "assertions": {
            "folds_pairwise_disjoint": True,
            "group_fold_unique": True,
            "components_single_fold": True,
            "confirm_sealed": True,
            "ood_sealed": True,
        },
        "frozen": True,
    }
    return manifest


def write_v5_split_manifest(
    manifest: Mapping[str, JSONValue], output_path: str | Path,
) -> Path:
    """Atomically write the V5 split manifest."""
    path = Path(output_path)
    write_json(path, dict(manifest))
    return path


# ---------------------------------------------------------------------------
# CLI: dry-run by default; --write performs the real freeze
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build (and optionally freeze) the V5 sprint split")
    parser.add_argument(
        "--fit50-selection",
        default="outputs/pesn/direct_egnn_v1_20261015/splits/"
                "selection_direct_fit50.json")
    parser.add_argument(
        "--lineage",
        default="outputs/pesn/dual_ts_v2_20261015/splits/"
                "dual_ts_split_lineage_v1.json")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument(
        "--output", default="data/manifests/v5/split_manifest.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--write", action="store_true",
        help="actually write the manifest (default: dry-run summary only)")
    args = parser.parse_args(argv)

    manifest = build_v5_split(
        args.fit50_selection, args.lineage, seed=args.seed,
        interim_dir=args.interim_dir)
    sizes = [len(manifest["folds"][str(k)]) for k in range(N_FOLDS)]  # type: ignore[index]
    print(f"[dry-run] folds sizes: {sizes}")
    print(f"[dry-run] confirm candidates: "
          f"{manifest['confirm_candidates']['n_groups']} (sealed)")  # type: ignore[index]
    print(f"[dry-run] ood candidates: {manifest['ood_candidates']['n_families']} families "  # type: ignore[index]
          f"x {manifest['ood_candidates']['n_groups']} groups (sealed)")  # type: ignore[index]
    if args.write:
        path = write_v5_split_manifest(manifest, args.output)
        print(f"wrote {path}")
    else:
        print("no --write: manifest not written")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
