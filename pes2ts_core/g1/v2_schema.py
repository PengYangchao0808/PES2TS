"""Schema contract, enumerations, and defaults for the G1 v2 audit-and-repair
layer.

v2 exists because the v1 bond-edit semantics were deliberately **not**
mutually exclusive (a single->double bond change produced one ``formed``, one
``broken``, *and* one ``order_changed`` entry for the same atom pair), the P2
L0/L1 layers consumed those overlapping counters, and the gate released
records carrying IRC event mismatches as ordinary verified records.  v2
defines:

- **Mutually exclusive edits.**  One record per unordered map pair ``(i, j)``
  with ``edit_kind`` in {formed, broken, order_changed}: R-absent/P-present is
  ``formed``, R-present/P-absent is ``broken``, both-present-with-different-
  order is ``order_changed``.  Original bond orders (aromatic ``1.5``
  included) are preserved, and edited aromatic bonds carry the id of their
  conjugated/aromatic region so a whole-ring re-kekulization is never
  mistaken for many independent scan coordinates.
- **Hydrogen partner changes.**  Every hydrogen whose bonded partner differs
  between the sides is one record with a machine-exhaustive ``kind`` derived
  from the partner states (free / bonded-to-H / bonded-to-heavy); H2 events
  are never labelled as ordinary transfers.
- **A three-way gate.**  ``scan_ready`` / ``needs_review`` / ``excluded``
  with typed issue codes in three separated dimensions (structural validity,
  mapping determinism, IRC evidence quality).  Conflicts land in the review
  queue instead of hiding inside a single eligible number.

Nothing in this module reads data or touches the quarantined truth: the v2
rebuild works exclusively from persisted P1/G1 artifacts and the inventory.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Schema versions and taxonomy.
# ---------------------------------------------------------------------------
#: Frozen v1 baseline manifest (the audit's comparison anchor).
V2_FREEZE_SCHEMA_VERSION: Final[str] = "g1_v2_freeze_v1"
#: Per-reaction v2 edit/audit document.
V2_EDITS_SCHEMA_VERSION: Final[str] = "g1_v2_edits_v1"
#: v2 side manifests (build, classification, migration).
V2_MANIFEST_SCHEMA_VERSION: Final[str] = "g1_v2_manifest_v1"
#: Per-reaction v2 classification document.
V2_CLASS_SCHEMA_VERSION: Final[str] = "g1_v2_class_v1"
#: Gate manifest and scan-ready id list.
V2_GATE_SCHEMA_VERSION: Final[str] = "g1_v2_gate_v1"
#: Per-reaction G2 export document (whitelisted fields only).
V2_EXPORT_SCHEMA_VERSION: Final[str] = "g1_v2_export_v1"
#: Taxonomy version embedded in every v2 cluster id.
V2_TAXONOMY_VERSION: Final[str] = "g1p2_taxonomy_v2"
#: Provenance stamp of the atom mapping consumed by v2 (P1 truth-assisted).
MAPPING_PROVENANCE: Final[str] = "truth_assisted_p1"
#: Classification flag: the canonicalization budget was exhausted and the
#: group-based WL-stable fallback supplied the (still invariant) cluster id.
FLAG_BUDGET: Final[str] = "canonical_budget_exhausted"

# ---------------------------------------------------------------------------
# Edit kinds (mutually exclusive per unordered map pair).
# ---------------------------------------------------------------------------
EDIT_FORMED: Final[str] = "formed"
EDIT_BROKEN: Final[str] = "broken"
EDIT_ORDER_CHANGED: Final[str] = "order_changed"
#: Every edit kind, in documentation order.
EDIT_KINDS: Final[tuple[str, ...]] = (
    EDIT_FORMED, EDIT_BROKEN, EDIT_ORDER_CHANGED,
)
#: Bond-order sentinel marking "no bond on this side".
NO_BOND: Final[float] = 0.0
#: Aromatic bond order as read through the raw graph view.
AROMATIC_ORDER: Final[float] = 1.5

# ---------------------------------------------------------------------------
# Hydrogen partner-change kinds.
# ---------------------------------------------------------------------------
#: heavy partner -> different heavy partner (a true donor->acceptor transfer).
H_CHANGE_TRANSFER: Final[str] = "transfer"
#: heavy partner -> no partner (bound H freed).
H_CHANGE_RELEASE: Final[str] = "release"
#: no partner -> heavy partner (free H captured).
H_CHANGE_CAPTURE: Final[str] = "capture"
#: heavy partner -> H partner (this H joins an H-H bond).
H_CHANGE_TO_HH: Final[str] = "to_hh"
#: H partner -> heavy partner (this H leaves an H-H bond).
H_CHANGE_FROM_HH: Final[str] = "from_hh"
#: H partner -> no partner (H2 member freed).
H_CHANGE_HH_RELEASE: Final[str] = "hh_release"
#: no partner -> H partner (free H + free H form H2).
H_CHANGE_HH_FORM_FREE: Final[str] = "hh_form_free"
#: H partner -> different H partner (re-pairing inside an H2 cluster).
H_CHANGE_HH_SWAP: Final[str] = "hh_swap"
#: Every hydrogen-change kind, in documentation order.
H_CHANGE_KINDS: Final[tuple[str, ...]] = (
    H_CHANGE_TRANSFER, H_CHANGE_RELEASE, H_CHANGE_CAPTURE,
    H_CHANGE_TO_HH, H_CHANGE_FROM_HH, H_CHANGE_HH_RELEASE,
    H_CHANGE_HH_FORM_FREE, H_CHANGE_HH_SWAP,
)
#: Kinds that constitute an H-H bond event (H2 formation or cleavage).
HH_EVENT_KINDS: Final[frozenset[str]] = frozenset({
    H_CHANGE_TO_HH, H_CHANGE_FROM_HH, H_CHANGE_HH_RELEASE,
    H_CHANGE_HH_FORM_FREE, H_CHANGE_HH_SWAP,
})

# ---------------------------------------------------------------------------
# Audit dimensions and typed issue codes.
# ---------------------------------------------------------------------------
#: Dimension: graph/index/geometry structure of the record itself.
DIMENSION_STRUCTURE: Final[str] = "structural_validity"
#: Dimension: determinism of the atom mapping (collapse, truncation).
DIMENSION_MAPPING: Final[str] = "mapping_determinism"
#: Dimension: quality of the IRC evidence attached to the record.
DIMENSION_IRC: Final[str] = "irc_evidence_quality"
#: Every audit dimension, in documentation order.
DIMENSIONS: Final[tuple[str, ...]] = (
    DIMENSION_STRUCTURE, DIMENSION_MAPPING, DIMENSION_IRC,
)

# -- structural_validity -----------------------------------------------------
#: map_to_atoms is not a bijection onto the side component rows.
ISSUE_MAP_BIJECTION_BROKEN: Final[str] = "map_bijection_broken"
#: Element at (component, local row) disagrees with the map-space element.
ISSUE_ELEMENT_SEQUENCE_MISMATCH: Final[str] = "element_sequence_mismatch"
#: Component assignment of map_to_atoms disagrees with the graph payload.
ISSUE_COMPONENT_ASSIGNMENT_MISMATCH: Final[str] = "component_assignment_mismatch"
#: The v1 bond_events in the P1 document are not derivable from the graph.
ISSUE_LEGACY_EVENT_INCONSISTENCY: Final[str] = "legacy_event_inconsistency"
#: One map pair carries more than one bond on a side (malformed source graph).
ISSUE_MULTI_BOND_PAIR: Final[str] = "multi_bond_pair"
#: An edit record is internally inconsistent (exclusivity/order/center).
ISSUE_EDIT_CONTRACT_VIOLATION: Final[str] = "edit_contract_violation"

# -- mapping_determinism -----------------------------------------------------
#: Symmetry-collapsed tie with an alternative that changes the edit record.
ISSUE_COLLAPSE_EDIT_AMBIGUOUS: Final[str] = "collapse_edit_ambiguous"
#: Symmetry-collapse audit could not enumerate every tied alternative.
ISSUE_COLLAPSE_AUDIT_TRUNCATED: Final[str] = "collapse_audit_truncated"
#: The G1 component search hit its cap (candidate set provably incomplete).
ISSUE_CANDIDATE_SEARCH_TRUNCATED: Final[str] = "candidate_search_truncated"

# -- irc_evidence_quality ----------------------------------------------------
#: IRC geometry contradicts at least one mapped bond event.
ISSUE_IRC_EVENT_MISMATCH: Final[str] = "irc_event_mismatch"
#: R/P branch orientation stayed unresolved.
ISSUE_ORIENTATION_UNRESOLVED: Final[str] = "orientation_unresolved"

#: Every typed issue code, in documentation order.
ISSUE_CODES: Final[tuple[str, ...]] = (
    ISSUE_MAP_BIJECTION_BROKEN,
    ISSUE_ELEMENT_SEQUENCE_MISMATCH,
    ISSUE_COMPONENT_ASSIGNMENT_MISMATCH,
    ISSUE_LEGACY_EVENT_INCONSISTENCY,
    ISSUE_MULTI_BOND_PAIR,
    ISSUE_EDIT_CONTRACT_VIOLATION,
    ISSUE_COLLAPSE_EDIT_AMBIGUOUS,
    ISSUE_COLLAPSE_AUDIT_TRUNCATED,
    ISSUE_CANDIDATE_SEARCH_TRUNCATED,
    ISSUE_IRC_EVENT_MISMATCH,
    ISSUE_ORIENTATION_UNRESOLVED,
)
#: Issue code -> audit dimension.
ISSUE_DIMENSIONS: Final[dict[str, str]] = {
    ISSUE_MAP_BIJECTION_BROKEN: DIMENSION_STRUCTURE,
    ISSUE_ELEMENT_SEQUENCE_MISMATCH: DIMENSION_STRUCTURE,
    ISSUE_COMPONENT_ASSIGNMENT_MISMATCH: DIMENSION_STRUCTURE,
    ISSUE_LEGACY_EVENT_INCONSISTENCY: DIMENSION_STRUCTURE,
    ISSUE_MULTI_BOND_PAIR: DIMENSION_STRUCTURE,
    ISSUE_EDIT_CONTRACT_VIOLATION: DIMENSION_STRUCTURE,
    ISSUE_COLLAPSE_EDIT_AMBIGUOUS: DIMENSION_MAPPING,
    ISSUE_COLLAPSE_AUDIT_TRUNCATED: DIMENSION_MAPPING,
    ISSUE_CANDIDATE_SEARCH_TRUNCATED: DIMENSION_MAPPING,
    ISSUE_IRC_EVENT_MISMATCH: DIMENSION_IRC,
    ISSUE_ORIENTATION_UNRESOLVED: DIMENSION_IRC,
}

# ---------------------------------------------------------------------------
# Gate statuses.
# ---------------------------------------------------------------------------
#: Ready for the endpoint-only G2 scan stage.
GATE_SCAN_READY: Final[str] = "scan_ready"
#: Carries typed conflicts; enters the adjudication queue, never G2.
GATE_NEEDS_REVIEW: Final[str] = "needs_review"
#: Not a candidate at all (non-eligible P1 status or missing artifacts).
GATE_EXCLUDED: Final[str] = "excluded"
#: Every gate status.
GATE_STATUSES: Final[tuple[str, ...]] = (
    GATE_SCAN_READY, GATE_NEEDS_REVIEW, GATE_EXCLUDED,
)

#: v2 audit outcome stored on the edits document (pre-gate).
AUDIT_CLEAN: Final[str] = "clean"
AUDIT_ISSUES: Final[str] = "issues"
AUDIT_EXCLUDED: Final[str] = "excluded"

# ---------------------------------------------------------------------------
# Directory and artifact names (all under ``paths.interim``/``paths.manifests``,
# never overlapping the v1 trees).
# ---------------------------------------------------------------------------
V2_DIRNAME: Final[str] = "g1_v2"
EDITS_DIRNAME: Final[str] = "edits"
CLASSES_DIRNAME: Final[str] = "classes"
EXPORT_DIRNAME: Final[str] = "export"

FREEZE_MANIFEST_FILENAME: Final[str] = "g1_v2_freeze.json"
V2_MANIFEST_FILENAME: Final[str] = "g1_v2_manifest.json"
V2_SUMMARY_FILENAME: Final[str] = "g1_v2_summary.parquet"
V2_CLASS_MANIFEST_FILENAME: Final[str] = "g1_v2_class_manifest.json"
V2_CLASS_SUMMARY_FILENAME: Final[str] = "g1_v2_class_summary.parquet"
V2_MIGRATION_FILENAME: Final[str] = "g1_v2_migration.json"
V2_GATE_FILENAME: Final[str] = "g1_v2_gate.json"
G2_SCAN_READY_FILENAME: Final[str] = "g2_scan_ready.json"
G2_NEEDS_REVIEW_FILENAME: Final[str] = "g2_needs_review.json"

# ---------------------------------------------------------------------------
# ``g1_v2`` configuration defaults.
# ---------------------------------------------------------------------------
#: Reactions per v2 shard directory.
DEFAULT_V2_SHARD_SIZE: Final[int] = 1000
#: Max alternative candidate combinations enumerated by the collapse audit.
DEFAULT_COLLAPSE_AUDIT_BUDGET: Final[int] = 256
#: Reaction-center shell recorded on the v2 document.
DEFAULT_CENTER_SHELL: Final[int] = 1
#: Per-issue manual-check sample size.
DEFAULT_V2_SAMPLE_SIZE: Final[int] = 20
#: Sample seed.
DEFAULT_V2_SAMPLE_SEED: Final[int] = 42
#: Node-individualization budget of the v2 canonical labeler.
DEFAULT_V2_CANONICAL_BUDGET: Final[int] = 2000

#: v2 hierarchy levels.
LEVEL_L0: Final[str] = "l0_edit_family"
LEVEL_L0U: Final[str] = "l0u_undirected_family"
LEVEL_L1: Final[str] = "l1_center_template"
LEVEL_L2: Final[str] = "l2_context_r1"
LEVEL_L3: Final[str] = "l3_context_r2"
V2_CLASS_LEVELS: Final[tuple[str, ...]] = (
    LEVEL_L0, LEVEL_L0U, LEVEL_L1, LEVEL_L2, LEVEL_L3,
)
#: Rule labels (unchanged nine-label set; semantics recomputed from v2 edits).
FAMILY_LABELS: Final[tuple[str, ...]] = (
    "addition", "substitution", "elimination", "rearrangement",
    "ring_closure", "ring_opening", "fragmentation", "h_transfer", "other",
)


__all__ = [
    "AROMATIC_ORDER",
    "AUDIT_CLEAN",
    "AUDIT_EXCLUDED",
    "AUDIT_ISSUES",
    "CLASSES_DIRNAME",
    "DEFAULT_COLLAPSE_AUDIT_BUDGET",
    "DEFAULT_CENTER_SHELL",
    "DEFAULT_V2_CANONICAL_BUDGET",
    "DEFAULT_V2_SAMPLE_SEED",
    "DEFAULT_V2_SAMPLE_SIZE",
    "DEFAULT_V2_SHARD_SIZE",
    "DIMENSIONS",
    "DIMENSION_IRC",
    "DIMENSION_MAPPING",
    "DIMENSION_STRUCTURE",
    "EDIT_BROKEN",
    "EDIT_FORMED",
    "EDIT_KINDS",
    "EDIT_ORDER_CHANGED",
    "EDITS_DIRNAME",
    "EXPORT_DIRNAME",
    "FAMILY_LABELS",
    "FREEZE_MANIFEST_FILENAME",
    "G2_NEEDS_REVIEW_FILENAME",
    "G2_SCAN_READY_FILENAME",
    "GATE_EXCLUDED",
    "GATE_NEEDS_REVIEW",
    "GATE_SCAN_READY",
    "GATE_STATUSES",
    "H_CHANGE_CAPTURE",
    "H_CHANGE_FROM_HH",
    "H_CHANGE_HH_FORM_FREE",
    "H_CHANGE_HH_RELEASE",
    "H_CHANGE_HH_SWAP",
    "H_CHANGE_KINDS",
    "H_CHANGE_RELEASE",
    "H_CHANGE_TO_HH",
    "H_CHANGE_TRANSFER",
    "HH_EVENT_KINDS",
    "ISSUE_CODES",
    "ISSUE_COLLAPSE_AUDIT_TRUNCATED",
    "ISSUE_COLLAPSE_EDIT_AMBIGUOUS",
    "ISSUE_COMPONENT_ASSIGNMENT_MISMATCH",
    "ISSUE_DIMENSIONS",
    "ISSUE_EDIT_CONTRACT_VIOLATION",
    "ISSUE_ELEMENT_SEQUENCE_MISMATCH",
    "ISSUE_IRC_EVENT_MISMATCH",
    "ISSUE_LEGACY_EVENT_INCONSISTENCY",
    "ISSUE_MAP_BIJECTION_BROKEN",
    "ISSUE_MULTI_BOND_PAIR",
    "ISSUE_ORIENTATION_UNRESOLVED",
    "LEVEL_L0",
    "LEVEL_L0U",
    "LEVEL_L1",
    "LEVEL_L2",
    "LEVEL_L3",
    "MAPPING_PROVENANCE",
    "NO_BOND",
    "V2_CLASS_LEVELS",
    "V2_CLASS_MANIFEST_FILENAME",
    "V2_CLASS_SCHEMA_VERSION",
    "V2_CLASS_SUMMARY_FILENAME",
    "V2_DIRNAME",
    "V2_EDITS_SCHEMA_VERSION",
    "V2_EXPORT_SCHEMA_VERSION",
    "V2_FREEZE_SCHEMA_VERSION",
    "V2_GATE_FILENAME",
    "V2_GATE_SCHEMA_VERSION",
    "V2_MANIFEST_FILENAME",
    "V2_MANIFEST_SCHEMA_VERSION",
    "V2_MIGRATION_FILENAME",
    "V2_SUMMARY_FILENAME",
    "V2_TAXONOMY_VERSION",
]
