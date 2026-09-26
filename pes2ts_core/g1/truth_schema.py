"""Schema contract, status enumerations, and defaults for the truth-assisted
P1/P2 annotation layers of G1.

This module centralizes every constant shared by the P1 mapping builder, the
P2 classifier, their verifiers, and the CLI, so a status name or schema
version is defined exactly once.  Nothing here reads data: the truth-isolation
boundary is untouched (the only truth-reading G1 module is
:mod:`pes2ts_core.g1.p1_truth`, allowlisted in the static guard).
"""

from __future__ import annotations

from typing import Final

#: Schema version stamped into every P1 per-reaction mapping document.
P1_DOCUMENT_SCHEMA_VERSION: Final[str] = "g1_p1_truth_v1"
#: Schema version stamped into every P1 manifest/coverage artifact.
P1_MANIFEST_SCHEMA_VERSION: Final[str] = "g1_p1_truth_manifest_v1"
#: Schema version stamped into every P2 per-reaction class document.
P2_DOCUMENT_SCHEMA_VERSION: Final[str] = "g1_p2_class_v1"
#: Schema version stamped into every P2 manifest/coverage/cluster artifact.
P2_MANIFEST_SCHEMA_VERSION: Final[str] = "g1_p2_class_manifest_v1"
#: Schema version of the join-audit artifact.
JOIN_AUDIT_SCHEMA_VERSION: Final[str] = "g1_p1_join_audit_v1"
#: Schema version of the G2 eligibility gate manifest.
GATE_SCHEMA_VERSION: Final[str] = "g1_gate_v1"

#: Identifier of the mapping algorithm recorded in every P1 document.
MAPPING_ALGORITHM: Final[str] = "ts_map_order_identity_v1"
#: Identifier of the IRC layout detection algorithm.
LAYOUT_ALGORITHM: Final[str] = "irc_step_splice_v1"

#: P1 mapping status: exactly one constraint-satisfying candidate assignment.
STATUS_RESOLVED_UNIQUE: Final[str] = "resolved_unique"
#: P1 mapping status: several candidates tied geometrically (one equivalence
#: class); a canonical representative is chosen and the collapse recorded.
STATUS_RESOLVED_SYMMETRY: Final[str] = "resolved_symmetry_collapsed"
#: Non-equivalent TS-row permutations survive every constraint.
STATUS_UNRESOLVED_CENTER: Final[str] = "unresolved_reactive_center"
#: Candidate enumeration hit its cap; completeness cannot be proven.
STATUS_UNRESOLVED_TRUNCATED: Final[str] = "unresolved_truncated"
#: No candidate makes the TS/IRC endpoints correspond to the given R/P.
STATUS_ENDPOINT_MISMATCH: Final[str] = "ts_irc_endpoint_mismatch"
#: Atom counts or element sequences are not conserved.
STATUS_ATOM_ELEMENT_MISMATCH: Final[str] = "atom_or_element_mismatch"
#: Inventory row has no matching TS and/or IRC record.
STATUS_MISSING_TRUTH_JOIN: Final[str] = "missing_truth_join"
#: The G1 source document itself is rejected (no usable index table).
STATUS_SOURCE_MISMATCH: Final[str] = "source_structure_mismatch"

#: Every P1 status, in documentation order.
P1_STATUSES: Final[tuple[str, ...]] = (
    STATUS_RESOLVED_UNIQUE,
    STATUS_RESOLVED_SYMMETRY,
    STATUS_UNRESOLVED_CENTER,
    STATUS_UNRESOLVED_TRUNCATED,
    STATUS_ENDPOINT_MISMATCH,
    STATUS_ATOM_ELEMENT_MISMATCH,
    STATUS_MISSING_TRUTH_JOIN,
    STATUS_SOURCE_MISMATCH,
)
#: Statuses whose reactions may enter the endpoint-only G2 stage.
G2_ELIGIBLE_STATUSES: Final[frozenset[str]] = frozenset(
    {STATUS_RESOLVED_UNIQUE, STATUS_RESOLVED_SYMMETRY}
)

#: P2 classification statuses.
STATUS_CLASSIFIED: Final[str] = "classified"
STATUS_UNCLASSIFIABLE: Final[str] = "unclassifiable"
P2_STATUSES: Final[tuple[str, ...]] = (STATUS_CLASSIFIED, STATUS_UNCLASSIFIABLE)

#: IRC trajectory layout labels.
LAYOUT_TS_FIRST: Final[str] = "ts_first_dual_branch"
LAYOUT_TS_LAST: Final[str] = "ts_last_dual_branch"
LAYOUT_SINGLE_BRANCH: Final[str] = "single_branch"
LAYOUT_UNUSUAL: Final[str] = "unusual"

#: IRC branch orientation: which side the first branch terminates at.
ORIENTATION_R_FIRST: Final[str] = "R_first"
ORIENTATION_P_FIRST: Final[str] = "P_first"
ORIENTATION_UNRESOLVED: Final[str] = "unresolved"

#: Endpoint match verdicts.
ENDPOINT_PASS: Final[str] = "pass"
ENDPOINT_WEAK: Final[str] = "weak"
ENDPOINT_FAIL: Final[str] = "fail"

#: Per-event IRC support verdicts.
SUPPORT_SUPPORT: Final[str] = "support"
SUPPORT_WEAK: Final[str] = "weak"
SUPPORT_MISMATCH: Final[str] = "mismatch"

#: Directory (under ``paths.interim``) holding every P1/P2 truth annotation.
G1_TRUTH_DIRNAME: Final[str] = "g1_truth"
P1_MAPPING_DIRNAME: Final[str] = "p1_mapping"
REACTION_CLASS_DIRNAME: Final[str] = "reaction_classes"

#: Artifact filenames.
P1_SUMMARY_FILENAME: Final[str] = "g1_p1_summary.parquet"
P1_MANIFEST_FILENAME: Final[str] = "g1_p1_manifest.json"
P1_COVERAGE_FILENAME: Final[str] = "g1_p1_coverage.json"
JOIN_AUDIT_FILENAME: Final[str] = "g1_p1_join_audit.json"
P2_SUMMARY_FILENAME: Final[str] = "g1_reaction_class_summary.parquet"
P2_MANIFEST_FILENAME: Final[str] = "g1_p2_manifest.json"
P2_COVERAGE_FILENAME: Final[str] = "g1_p2_coverage.json"
CLUSTER_REPORT_FILENAME: Final[str] = "g1_cluster_report.json"
GATE_MANIFEST_FILENAME: Final[str] = "g1_gate.json"
G2_ELIGIBLE_FILENAME: Final[str] = "g2_eligible.json"

#: Exit code: a truth-assisted subcommand was invoked without ``--allow-truth``.
EXIT_TRUTH_FLAG_REQUIRED: Final[int] = 23

# ---------------------------------------------------------------------------
# ``g1_truth`` configuration defaults.
# ---------------------------------------------------------------------------
#: Reactions per P1/P2 shard directory.
DEFAULT_SHARD_SIZE: Final[int] = 1000
#: Cap on enumerated candidate combinations per side (truncation beyond).
DEFAULT_MAX_COMBINATIONS: Final[int] = 1024
#: Width (Angstrom) of a Kabsch-RMSD equivalence class for candidate collapse.
DEFAULT_RMSD_CLASS_TOLERANCE: Final[float] = 0.05
#: Aligned RMSD at/below which an endpoint match is ``pass``.
DEFAULT_ENDPOINT_PASS: Final[float] = 1.0
#: Aligned RMSD at/below which an endpoint match is ``weak`` (beyond: ``fail``).
#: Calibrated on the real archive: IRC branch termini routinely sit 1.5-3 A
#: from the *optimized* R/P species (conformer differences), so the endpoint
#: distance is a quality annotation, not the mapping's identity proof -- the
#: identity proof is the exact TS-frame match plus the element hard checks.
DEFAULT_ENDPOINT_WEAK: Final[float] = 3.0
#: Direct RMSD below which an IRC frame is identified with the TS geometry.
DEFAULT_TS_FRAME_TOLERANCE: Final[float] = 1.0e-4
#: A splice step must exceed this factor times the median step.
DEFAULT_SPLICE_FACTOR: Final[float] = 5.0
#: Bonded-distance cutoff: covalent-radius sum plus this tolerance.
DEFAULT_BOND_DISTANCE_TOLERANCE: Final[float] = 0.45
#: Search budget for the TS-row permutation fallback.
DEFAULT_PERMUTATION_BUDGET: Final[int] = 64
#: Minimum RMSD improvement that distinguishes fallback permutations.
DEFAULT_PERMUTATION_TOLERANCE: Final[float] = 0.05
#: Per-status manual-check sample size.
DEFAULT_SAMPLE_SIZE: Final[int] = 20
#: Sample seed when ``g1_truth.sample_seed`` is absent.
DEFAULT_SAMPLE_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# ``g1_class`` configuration defaults.
# ---------------------------------------------------------------------------
#: Default taxonomy version embedded in every cluster id.
DEFAULT_TAXONOMY_VERSION: Final[str] = "g1p2_taxonomy_v1"
#: Node-individualization budget of the canonical labeler.
DEFAULT_CANONICAL_BUDGET: Final[int] = 2000
#: Hierarchy levels produced by the classifier.
CLASS_LEVELS: Final[tuple[str, ...]] = ("l0_edit_family", "l1_center_template", "l2_context_r1", "l3_context_r2")
#: Human-readable rule-based family labels (multi-label).
FAMILY_LABELS: Final[tuple[str, ...]] = (
    "addition", "substitution", "elimination", "rearrangement",
    "ring_closure", "ring_opening", "fragmentation", "h_transfer", "other",
)


__all__ = [
    "CLASS_LEVELS",
    "DEFAULT_BOND_DISTANCE_TOLERANCE",
    "DEFAULT_CANONICAL_BUDGET",
    "DEFAULT_ENDPOINT_PASS",
    "DEFAULT_ENDPOINT_WEAK",
    "DEFAULT_MAX_COMBINATIONS",
    "DEFAULT_PERMUTATION_BUDGET",
    "DEFAULT_PERMUTATION_TOLERANCE",
    "DEFAULT_RMSD_CLASS_TOLERANCE",
    "DEFAULT_SAMPLE_SEED",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SHARD_SIZE",
    "DEFAULT_SPLICE_FACTOR",
    "DEFAULT_TAXONOMY_VERSION",
    "DEFAULT_TS_FRAME_TOLERANCE",
    "FAMILY_LABELS",
    "G1_TRUTH_DIRNAME",
    "G2_ELIGIBLE_FILENAME",
    "G2_ELIGIBLE_STATUSES",
    "GATE_MANIFEST_FILENAME",
    "GATE_SCHEMA_VERSION",
    "JOIN_AUDIT_FILENAME",
    "JOIN_AUDIT_SCHEMA_VERSION",
    "LAYOUT_ALGORITHM",
    "LAYOUT_SINGLE_BRANCH",
    "LAYOUT_TS_FIRST",
    "LAYOUT_TS_LAST",
    "LAYOUT_UNUSUAL",
    "MAPPING_ALGORITHM",
    "ORIENTATION_P_FIRST",
    "ORIENTATION_R_FIRST",
    "ORIENTATION_UNRESOLVED",
    "P1_COVERAGE_FILENAME",
    "P1_DOCUMENT_SCHEMA_VERSION",
    "P1_MANIFEST_FILENAME",
    "P1_MANIFEST_SCHEMA_VERSION",
    "P1_MAPPING_DIRNAME",
    "P1_STATUSES",
    "P1_SUMMARY_FILENAME",
    "P2_COVERAGE_FILENAME",
    "P2_DOCUMENT_SCHEMA_VERSION",
    "P2_MANIFEST_FILENAME",
    "P2_MANIFEST_SCHEMA_VERSION",
    "P2_STATUSES",
    "P2_SUMMARY_FILENAME",
    "REACTION_CLASS_DIRNAME",
    "STATUS_ATOM_ELEMENT_MISMATCH",
    "STATUS_CLASSIFIED",
    "STATUS_ENDPOINT_MISMATCH",
    "STATUS_MISSING_TRUTH_JOIN",
    "STATUS_RESOLVED_SYMMETRY",
    "STATUS_RESOLVED_UNIQUE",
    "STATUS_SOURCE_MISMATCH",
    "STATUS_UNCLASSIFIABLE",
    "STATUS_UNRESOLVED_CENTER",
    "STATUS_UNRESOLVED_TRUNCATED",
    "SUPPORT_MISMATCH",
    "SUPPORT_SUPPORT",
    "SUPPORT_WEAK",
    "ENDPOINT_FAIL",
    "ENDPOINT_PASS",
    "ENDPOINT_WEAK",
    "EXIT_TRUTH_FLAG_REQUIRED",
]
