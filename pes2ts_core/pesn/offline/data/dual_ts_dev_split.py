"""V4.2 internal development split (Lane A, contract §3).

Deterministic ``fit45_train + fit5_dev`` over the fit-50 identity clusters.
The unit is the IDENTITY CLUSTER: starting from the fit wave's 50 groups in
the splits lineage, identity-bound duplicate-group partners (G0
``duplicate_groups.json``) are merged into one cluster — a cluster never
spans train/dev (contract §3 rule 1).  In this wave no two fit-50 groups are
duplicate partners (only ``RXN_0000000072`` has an out-of-wave partner
``RXN_0000018080``), so there are exactly 50 singleton clusters.

Frozen deterministic rule (contract §3 rule 3):

* clusters sorted by ``min_group_id`` lexicographically;
* dev clusters = sorted positions 5 / 15 / 25 / 35 / 45 (1-based);
* if the dev set's "contains-insertion group count" is 0 or 5, then starting
  from position 46 scan forward (lexicographic) for the first unused cluster
  that carries the complementary insertion property and swap it in place of
  the position-45 cluster (exactly one replacement; the rule is unique);

The artifact ``pesn_dual_ts_dev_split_v42_v1`` records both name lists, every
cluster's metadata (native action count, contains-insertion-before-reconnect,
max atom count, disputed-5 membership), the rule string, ``split_sha256``
(content digest excluding timestamp), generation time, and the lineage parent
hash.  Differences vs the legacy ``wp3_features_v41_a45`` group set are only
recorded, never adopted (contract §3 rule 4).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

from pes2ts_core.pesn.offline.data.dual_ts_native_rebuild import sha256_file
from pes2ts_core.utils.hashing import stable_json_dumps

DEV_SPLIT_SCHEMA: Final[str] = "pesn_dual_ts_dev_split_v42_v1"
DEV_SPLIT_TOOL_VERSION: Final[str] = "pes2ts_dual_ts_dev_split_tool_v1"

#: Frozen dev positions (1-based).
DEV_POSITIONS: Final[tuple[int, ...]] = (5, 15, 25, 35, 45)
#: Replacement scan starts at position 46 (contract §3 rule 3).
REPLACEMENT_SCAN_START: Final[int] = 46

#: Disputed-5 reactions (frozen disputed-bond rule, ADR-0023 / dual_ts_spec).
DISPUTED_5_REACTIONS: Final[frozenset[str]] = frozenset({
    "RXN_0000000160", "RXN_0000000756", "RXN_0000000778",
    "RXN_0000001076", "RXN_0000001174",
})


class DevSplitError(RuntimeError):
    """Typed refusal (never silently splits or drops a cluster)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise DevSplitError(code, detail)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevSplitError("INPUT_UNREADABLE", f"{path}: {exc!r}") from exc
    if not isinstance(doc, dict):
        raise DevSplitError("INPUT_MALFORMED", f"{path}: not a mapping")
    return doc


def _canonical_members(duplicate_groups_path: Path | None) -> dict[str, str] | None:
    """``reaction_id -> canonical_reaction_id`` from G0 duplicate ledger.

    Returns ``None`` when no ledger path is supplied (caller then falls back
    to the lineage's own reaction→group binding for clustering).
    """
    if duplicate_groups_path is None:
        return None
    doc = _load_json(duplicate_groups_path)
    groups = doc.get("groups")
    _require(isinstance(groups, list), "DUP_GROUPS_MALFORMED",
             f"{duplicate_groups_path}: missing groups list")
    members: dict[str, str] = {}
    for group in groups:
        canonical = group.get("canonical_reaction_id")
        member_ids = group.get("member_reaction_ids")
        if not isinstance(canonical, str) or not isinstance(member_ids, list):
            continue
        for member in member_ids:
            if isinstance(member, str):
                members[member] = canonical
    return members


def _identity_clusters(fit_groups: Sequence[str],
                       canonical: dict[str, str] | None,
                       lineage_binding: Mapping[str, str]) -> list[dict]:
    """Cluster the fit-50 groups by identity (duplicate-group partners)."""
    key_of: dict[str, str] = {}
    for g in fit_groups:
        if canonical is not None:
            key_of[g] = canonical.get(g, g)
        else:
            key_of[g] = str(lineage_binding.get(g, g))
    clusters: dict[str, list[str]] = {}
    for g in sorted(fit_groups):
        clusters.setdefault(key_of[g], []).append(g)
    out: list[dict] = []
    for key in sorted(clusters):
        members = sorted(clusters[key])
        out.append({
            "cluster_key": key,
            "min_group_id": min(members),
            "member_group_ids": members,
        })
    return sorted(out, key=lambda c: c["min_group_id"])


def _load_meta(manifest_or_features_dir: str | Path) -> tuple[dict, dict]:
    """Per-reaction metadata from a native manifest (or a features dir)."""
    p = Path(manifest_or_features_dir)
    if p.is_dir():
        manifest = _load_json(p / "features_manifest.json")
    else:
        manifest = _load_json(p)
    rows = manifest.get("per_reaction")
    _require(isinstance(rows, list), "MANIFEST_MALFORMED", "per_reaction missing")
    meta: dict[str, dict] = {}
    for row in rows:
        gid = str(row["reaction_group_id"])
        meta[gid] = {
            "native_action_count": int(row.get("actions_native", 0)),
            "has_insertion": bool(row.get("has_insertion", False)),
            "n_atoms": int(row.get("n_atoms", 0)),
            "disputed": bool(row.get("disputed", False)),
            "reaction_id": str(row.get("reaction_id", gid)),
        }
    return manifest, meta


def _split_digest(payload: Mapping) -> str:
    """Content digest of the dev-split artifact (timestamp + self excluded)."""
    view = {k: v for k, v in payload.items()
            if k not in ("generated_utc", "split_sha256")}
    import hashlib
    return hashlib.sha256(
        stable_json_dumps(view).encode("utf-8")).hexdigest()


def build_dev_split(
    lineage_path: str | Path,
    native_manifest_or_features_dir: str | Path,
    out_path: str | Path,
    *,
    duplicate_groups_path: str | Path | None = None,
) -> dict:
    """Build and write ``wp42_campaign/dev_split.json``; return its payload."""
    lineage_path = Path(lineage_path)
    lineage = _load_json(lineage_path)
    _require(lineage.get("schema_version") == "pesn_dual_ts_split_lineage_v1",
             "LINEAGE_SCHEMA", f"unexpected lineage schema at {lineage_path}")
    wave = lineage.get("dual_ts_wave_selections", {}).get("wave", {})
    fit_block = wave.get("fit", {})
    fit_groups = [str(g) for g in fit_block.get("group_ids", [])]
    _require(len(fit_groups) == 50, "FIT_WAVE_NOT_50",
             f"fit wave has {len(fit_groups)} groups, expected 50")

    binding = lineage.get("v1_binding", {}).get("reaction_to_group", {})
    dup_path = (None if duplicate_groups_path is None
                else Path(duplicate_groups_path))
    canonical = _canonical_members(dup_path)

    manifest, meta = _load_meta(native_manifest_or_features_dir)
    missing = [g for g in fit_groups if g not in meta]
    _require(not missing, "MANIFEST_INCOMPLETE",
             f"native manifest lacks fit groups {missing[:5]}")

    clusters = _identity_clusters(fit_groups, canonical, binding)
    _require(len(clusters) >= 45, "TOO_FEW_CLUSTERS",
             f"{len(clusters)} clusters < 45 (cannot form fit45+fit5)")

    # per-cluster metadata (native action count / insertion / max atoms /
    # disputed-5 membership).
    for cluster in clusters:
        members = cluster["member_group_ids"]
        cluster["native_action_count"] = sum(
            meta[m]["native_action_count"] for m in members)
        cluster["has_insertion"] = any(meta[m]["has_insertion"]
                                       for m in members)
        cluster["max_n_atoms"] = max(meta[m]["n_atoms"] for m in members)
        cluster["disputed5_members"] = sorted(
            m for m in members if m in DISPUTED_5_REACTIONS)
        cluster["identity_partners_outside_wave"] = []

    # record identity partners outside the fit-50 wave (transparency).
    if canonical is not None:
        wave_set = set(fit_groups)
        for cluster in clusters:
            for m in cluster["member_group_ids"]:
                # partners = reactions whose canonical == this cluster key
                partners = sorted(r for r, c in canonical.items()
                                  if c == cluster["cluster_key"]
                                  and r not in wave_set)
                if partners:
                    cluster["identity_partners_outside_wave"] = sorted(
                        set(cluster["identity_partners_outside_wave"])
                        | set(partners))

    dev_indices = [int(p) - 1 for p in DEV_POSITIONS]
    dev_clusters = [clusters[i] for i in dev_indices]
    dev_insertion_count = sum(1 for c in dev_clusters if c["has_insertion"])

    replacement: dict | None = None
    if dev_insertion_count in (0, 5):
        want_insertion = (dev_insertion_count == 0)
        replacement_cluster = None
        replacement_pos = None
        for pos in range(REPLACEMENT_SCAN_START - 1, len(clusters)):
            cluster = clusters[pos]
            if cluster["has_insertion"] == want_insertion:
                replacement_cluster = cluster
                replacement_pos = pos + 1  # 1-based
                break
        _require(replacement_cluster is not None, "REPLACEMENT_UNAVAILABLE",
                 f"dev insertion count {dev_insertion_count} is degenerate but "
                 "no complement cluster exists at position >= 46")
        # swap out the position-45 cluster (index 44), swap in the found one.
        replaced = clusters[dev_indices[-1]]
        clusters[dev_indices[-1]] = replacement_cluster
        replacement = {
            "trigger": ("dev_insertion_count_0"
                        if dev_insertion_count == 0 else "dev_insertion_count_5"),
            "replaced_cluster": replaced["min_group_id"],
            "replaced_position": int(DEV_POSITIONS[-1]),
            "replacement_cluster": replacement_cluster["min_group_id"],
            "replacement_position": replacement_pos,
            "replacement_insertion_property":
                "has_insertion" if want_insertion else "no_insertion",
        }
        dev_clusters = [clusters[i] for i in dev_indices]

    train_clusters = [c for c in clusters if c not in dev_clusters]

    rule_string = (
        "identity_cluster_unit; clusters sorted by min_group_id lexicographic; "
        f"dev = positions {list(DEV_POSITIONS)} (1-based); if dev "
        "contains-insertion count is 0 or 5, scan from position "
        f"{REPLACEMENT_SCAN_START} for the first unused cluster with the "
        "complementary insertion property and swap it in place of position "
        f"{DEV_POSITIONS[-1]} (one replacement, unique rule)")

    payload = {
        "schema_version": DEV_SPLIT_SCHEMA,
        "tool_version": DEV_SPLIT_TOOL_VERSION,
        "generated_utc": _utc_now(),
        "lineage": {
            "path": str(lineage_path),
            "sha256": sha256_file(lineage_path),
            "schema_version": lineage.get("schema_version"),
        },
        "native_manifest": {
            "path": str(Path(native_manifest_or_features_dir)),
            "sha256": sha256_file(
                Path(native_manifest_or_features_dir)
                if Path(native_manifest_or_features_dir).is_file()
                else Path(native_manifest_or_features_dir)
                / "features_manifest.json"),
        },
        "duplicate_groups": {
            "path": (None if dup_path is None else str(dup_path)),
            "sha256": (None if dup_path is None else sha256_file(dup_path)),
            "used": dup_path is not None,
        },
        "n_clusters": len(clusters),
        "dev_positions": list(DEV_POSITIONS),
        "dev_insertion_count": dev_insertion_count,
        "replacement": replacement,
        "train_group_ids": sorted(c["min_group_id"] for c in train_clusters),
        "dev_group_ids": sorted(c["min_group_id"] for c in dev_clusters),
        "clusters": [
            {
                "min_group_id": c["min_group_id"],
                "member_group_ids": c["member_group_ids"],
                "native_action_count": c["native_action_count"],
                "has_insertion_before_reconnect": c["has_insertion"],
                "max_n_atoms": c["max_n_atoms"],
                "disputed5_members": c["disputed5_members"],
                "identity_partners_outside_wave":
                    c["identity_partners_outside_wave"],
            }
            for c in clusters
        ],
        "rule_string": rule_string,
        "legacy_a45_diff": {
            "note": "differences vs legacy wp3_features_v41_a45 group set are "
                    "recorded only, never adopted (contract §3 rule 4)",
            "adopted": False,
        },
        "split_sha256": "",
    }
    payload["split_sha256"] = _split_digest(payload)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=False,
                              ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    return payload


def load_dev_split(path: str | Path) -> dict:
    """Load a dev-split artifact, verifying its ``split_sha256``."""
    doc = _load_json(Path(path))
    _require(doc.get("schema_version") == DEV_SPLIT_SCHEMA, "SCHEMA_UNKNOWN",
             f"expected {DEV_SPLIT_SCHEMA}, got {doc.get('schema_version')!r}")
    recorded = doc.get("split_sha256")
    _require(isinstance(recorded, str) and len(recorded) == 64,
             "DIGEST_MISSING", "split_sha256 must be a sha256 hex")
    actual = _split_digest(doc)
    _require(recorded == actual, "DIGEST_MISMATCH",
             f"recorded {recorded} != recomputed {actual}")
    return doc


__all__ = [
    "DEV_POSITIONS",
    "DEV_SPLIT_SCHEMA",
    "DEV_SPLIT_TOOL_VERSION",
    "DISPUTED_5_REACTIONS",
    "REPLACEMENT_SCAN_START",
    "DevSplitError",
    "build_dev_split",
    "load_dev_split",
]
