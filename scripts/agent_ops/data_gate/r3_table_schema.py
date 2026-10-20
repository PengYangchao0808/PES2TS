"""Declarative schema for the six R3 data tables of the 30h plan (**v2**).

Schema version ``r3_6table_v2`` — round 2 (WP-D2), implementing the **ORA-2**
blockers on top of the round-1 (``r3_6table_v1``) scaffold.

Normative sources (read-only references; this module never opens ``data/``):

* ``config/campaigns/pes2ts_30h_20261020.yaml`` → ``data_contract`` block
  (PD-1 six tables, PD-9 flat ``delta_xyz`` encoding, ``d0_primary_angstrom``,
  split unit, truth_access rule) — **normative**.
* ``docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md``
  §3.1–3.2 (six tables + manifests, Data Gate), §7.1 (frozen files),
  §7.4 (test ids T0.1–T0.4; T0.5 comes from ORA-2, not this doc).
* ``docs/contracts/PES2TS_ReactionIntent_Contract_v1.md`` §9 数据合同 /
  骨架表 (L425–445): ``irc_curve`` / ``irc_node`` / ``sample_view`` skeletons.
* ``docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md``
  v3.2 (PD-1 six-table ruling; PD-9 ``delta_xyz`` flat+shape/offset+SHA).

Disagreements between the documents are resolved in favour of the 30h plan's
**six-table** layout (PD-1) and recorded in :data:`SCHEMA_NOTES`; every
validation report embeds those notes so a machine consumer sees the
discrepancy without re-reading the docs.

Scope note (dry-run grade): the validator enforces column presence, FK
integrity, TS uniqueness/order, coordinate + PD-9 encoding finiteness,
intent pairing/involution, reversed-teacher consistency, d0/gauge rules,
weight-sum invariants and manifest/truth-log structure.  Declared ``dtype`` /
``enum`` values are contract for the E1 data builder; value-level enum checks
are enforced only where a gate test names them (label_status / gauge_ref /
units).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

SCHEMA_VERSION = "r3_6table_v2"
VALIDATOR_VERSION = "0.2.0"

SCHEMA_SOURCE: Tuple[str, ...] = (
    "config/campaigns/pes2ts_30h_20261020.yaml#data_contract (PD-1/PD-9, normative)",
    "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md §3.1-3.2",
    "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md §7.1, §7.4",
    "docs/contracts/PES2TS_ReactionIntent_Contract_v1.md §9 数据合同 (骨架表, L425-445)",
    "docs/plans/PES2TS_V5.1-R3_反应意图条件化与端点标签不变重训方案_v3_20261020.md v3.2 (PD-1/PD-9)",
)

SCHEMA_NOTES: Tuple[str, ...] = (
    # --- document discrepancies (round 1, kept) ---
    "table_count_discrepancy: the 30h plan §3.1 mandates six tables "
    "(irc_curve, irc_node, irc_edge, reaction_intent, curve_quality, sample_view) "
    "and explicitly says '共六张，不是五张' (six, not five), while ReactionIntent "
    "Contract v1 §骨架表 (L425-445) only skeletons three tables (irc_curve, "
    "irc_node, sample_view) from the earlier reduced lineage. Implemented: the "
    "6-table 30h layout (PD-1); contract skeleton fields are mapped onto it.",
    "split_manifest_name: plan §3.1 names split_manifest_r3.json, but the §7.1 "
    "freeze list names split_manifest.json. Validator prefers "
    "split_manifest_r3.json and records which file it used.",
    "irc_curve in the contract carries progress[]/nodes[] arrays; the CSV row "
    "flattens them to n_nodes/n_edges/ts_index (per-node rows live in "
    "irc_node, per-step rows in irc_edge). n_edges counts irc_edge rows for "
    "the curve across BOTH intent directions (row-count self-consistency is "
    "not yet gate-enforced).",
    "d0 is not part of curve identity (plan §3.1); it lives in the sample key "
    "(curve_id, node_index, intent_id, d0) of sample_view and in the teacher "
    "protocol, never in irc_curve.",
    "curve_quality.segment_quality/branch_quality are provenance-only (plan "
    "§3.1 '允许保留历史 segment/branch quality 供溯源'): they must never become "
    "split keys, model inputs or identifiers.",
    # --- ORA-2 round-2 additions ---
    "PD-9_flat_encoding (ORA-2 #1): irc_edge dropped the legacy scalar "
    "delta_dx/delta_dy/delta_dz columns. Each row now carries "
    "delta_xyz_flat (JSON flat float list), delta_xyz_shape ('[M,3]'), "
    "delta_xyz_offsets (JSON '[start,end)' into the table-level concatenated "
    "flat buffer, non-decreasing in file order, first row start==0), "
    "atom_map_sha256 (64-hex; format-checked only — the atom-map payload is "
    "not yet a table column), units (fixed literal 'angstrom') and "
    "delta_xyz_sha256. sha256 is defined here as "
    "sha256(struct.pack('<{M*3}d', *flat)) — IEEE-754 float64 little-endian "
    "of the parsed flat list; E1 must reproduce this byte encoding exactly. "
    "Flat length must equal prod(shape)==M*3.",
    "intent_scoped_edges (ORA-2 #4): irc_edge rows are intent-scoped teacher "
    "single-step records (intent_id + same_ref_frame columns), because T0.2b "
    "asserts that the reverse intent stores the mirrored edge (j->i) with "
    "delta_xyz_flat negated within 1e-6 (PD-9 side/negation eps). Physical "
    "adjacent-frame geometry alone cannot express 'edge under reverse intent'.",
    "intent_pairing (ORA-2 #2): reaction_intent carries reverse_of_intent_id "
    "and intent_pair_id; every curve must have exactly 2 intents forming one "
    "pair (forward+reverse, mutual reverses, reverse(reverse(x))==x) unless "
    "flagged intent_underdetermined. Unique key (curve_id, intent_id); "
    "intent_id alone is NOT assumed globally unique.",
    "label_weights (ORA-2 #3): sample_view carries label_weight, curve_weight, "
    "sample_weight, label_status (ok|masked|terminal|conflict), feature_sha, "
    "label_sha. Invariant (T0.1a): per (curve_id, d0) group, "
    "sum(curve_weight) == CURVE_WEIGHT_TARGET (1.0) — one curve is one unit "
    "of weight no matter how many intents/nodes it fans out to, so "
    "bidirectional augmentation never double-counts a curve.",
    "truth_access (ORA-2 #6): T0.5 audits truth_access_log.jsonl — every "
    "entry's accessor must be in TRUTH_ACCESSOR_ALLOWLIST "
    "(pes2ts_core.g0.truth.truth_reader only, per constitution R1 and the "
    "campaign data_contract). Missing log under an explicit --truth-log is "
    "BLOCKED (fail-closed), not failed; without the option an audit that "
    "finds no log is not_run.",
    "T0.5 is not part of the 30h plan §7.4 minimal test list; its id comes "
    "from ORA-2. T0.2a/T0.2b refine plan T0.2 (involution) and T0.3a/T0.3b "
    "refine plan T0.3 (d0/gauge) into machine-checkable halves.",
)


@dataclass(frozen=True)
class Column:
    """One column of an R3 table.

    ``key`` may contain ``;``-separated tokens: ``pk`` (primary key member) and
    ``fk:<table>.<column>`` (single-column foreign key target).
    """

    name: str
    dtype: str  # "string" | "int" | "float" | "bool"
    required: bool = True
    key: str = ""
    nullable: bool = False
    enum: Tuple[str, ...] = ()

    @property
    def key_tokens(self) -> Tuple[str, ...]:
        return tuple(t for t in self.key.split(";") if t)

    @property
    def is_pk(self) -> bool:
        return "pk" in self.key_tokens

    @property
    def fk_refs(self) -> Tuple[Tuple[str, str], ...]:
        refs = []
        for tok in self.key_tokens:
            if tok.startswith("fk:"):
                table, col = tok[3:].split(".", 1)
                refs.append((table, col))
        return tuple(refs)


@dataclass(frozen=True)
class TableSchema:
    """Schema of one ``<name>.csv`` table."""

    name: str
    filename: str
    primary_key: Tuple[str, ...]
    columns: Tuple[Column, ...]
    unique: Tuple[Tuple[str, ...], ...] = ()
    notes: Tuple[str, ...] = ()

    @property
    def column_names(self) -> Tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    @property
    def required_columns(self) -> Tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.required)

    @property
    def fk_columns(self) -> Tuple[Tuple[str, str, str], ...]:
        """(column, ref_table, ref_column) for single-column FKs."""
        out: List[Tuple[str, str, str]] = []
        for col in self.columns:
            for table, ref_col in col.fk_refs:
                out.append((col.name, table, ref_col))
        return tuple(out)

    def column(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"{self.name}: unknown column {name!r}")


# ---------------------------------------------------------------------------
# Normative constants (campaign data_contract / ORA-2)
# ---------------------------------------------------------------------------

#: fixed target for sum(curve_weight) per (curve_id, d0) group — one curve =
#: one unit of weight; forward+reverse must not double count (ORA-2 #3).
CURVE_WEIGHT_TARGET = 1.0
WEIGHT_SUM_TOL = 1e-9

#: PD-9 primary d0 in Angstrom (campaign data_contract.d0_primary_angstrom).
PRIMARY_D0_ANGSTROM = 0.08
D0_TOL = 1e-9

#: PD-9 side-legality / reverse-negation epsilon (data_contract.side_legality_eps).
DELTA_EPS = 1.0e-6

#: fixed units literal for irc_edge (data_contract.delta_xyz_encoding.binds).
CANONICAL_UNITS = "angstrom"

GAUGE_REF_VALUES: Tuple[str, ...] = ("local_frame", "unresolved_masked")
LABEL_STATUS_VALUES: Tuple[str, ...] = ("ok", "masked", "terminal", "conflict")

#: constitution R1 / data_contract.truth_access — the only allowed accessor.
TRUTH_ACCESSOR_ALLOWLIST: Tuple[str, ...] = (
    "truth_reader",
    "pes2ts_core.g0.truth.truth_reader",
)
#: truth log filename checked inside --tables-dir when no --truth-log is given.
TRUTH_LOG_TABLES_DIR_NAME = "truth_access_log.jsonl"
#: repo-relative fallback when --truth-log is not given (cwd-relative).
TRUTH_LOG_DEFAULT_SUBPATH = "docs/agent_runs/truth_access_log.jsonl"


# ---------------------------------------------------------------------------
# The six tables (30h plan §3.1, PD-1)
# ---------------------------------------------------------------------------

IRC_CURVE = TableSchema(
    name="irc_curve",
    filename="irc_curve.csv",
    primary_key=("curve_id",),
    columns=(
        Column("curve_id", "string", key="pk"),
        Column("canonical_irc_id", "string"),
        Column("n_nodes", "int"),
        Column("n_edges", "int"),
        Column("ts_index", "int"),
        Column("endpoint_a_index", "int", required=False, nullable=True),
        Column("endpoint_b_index", "int", required=False, nullable=True),
        Column("source_path", "string", required=False, nullable=True),
        Column("energy_a", "float", required=False, nullable=True),
        Column("energy_b", "float", required=False, nullable=True),
    ),
    notes=("curve identity excludes d0 and excludes branch/segment labels",),
)

IRC_NODE = TableSchema(
    name="irc_node",
    filename="irc_node.csv",
    primary_key=("curve_id", "node_index"),
    columns=(
        Column("curve_id", "string", key="pk;fk:irc_curve.curve_id"),
        Column("node_index", "int", key="pk"),
        Column("x", "float"),
        Column("y", "float"),
        Column("z", "float"),
        Column("progress", "float"),
        Column("is_ts", "bool"),
        Column("is_endpoint", "bool"),
        Column("energy", "float", required=False, nullable=True),
        Column("gradient_norm", "float", required=False, nullable=True),
        Column("quality", "string", required=False, nullable=True),
        Column("atomic_numbers", "string", required=False, nullable=True),
    ),
    notes=("canonical order A -> TS -> B; progress A=-1, TS=0, B=+1",),
)

IRC_EDGE = TableSchema(
    name="irc_edge",
    filename="irc_edge.csv",
    primary_key=("curve_id", "intent_id", "edge_index"),
    columns=(
        Column("curve_id", "string", key="pk;fk:irc_curve.curve_id"),
        # intent-scoped teacher single-step record (ORA-2 #4); src/dst form a
        # composite FK (curve_id, src|dst node_index) into irc_node — see
        # COMPOSITE_FK below.
        Column("intent_id", "string", key="fk:reaction_intent.intent_id"),
        Column("edge_index", "int", key="pk"),
        Column("src_node_index", "int"),
        Column("dst_node_index", "int"),
        # PD-9 flat encoding (ORA-2 #1)
        Column("delta_xyz_flat", "string"),       # JSON list of floats, len M*3
        Column("delta_xyz_shape", "string"),      # e.g. "[2,3]"
        Column("delta_xyz_offsets", "string"),    # JSON "[start,end)" offsets
        Column("atom_map_sha256", "string"),      # 64-hex (format check only)
        Column("delta_xyz_sha256", "string"),     # 64-hex over packed floats
        Column("units", "string", enum=(CANONICAL_UNITS,)),
        Column("same_ref_frame", "bool"),
    ),
    notes=(
        "legacy delta_dx/delta_dy/delta_dz columns are REMOVED (PD-9); a file "
        "carrying them instead is a schema violation",
        "delta_xyz_sha256 = sha256(struct.pack('<{M*3}d', *flat)) — float64 LE",
    ),
)

REACTION_INTENT = TableSchema(
    name="reaction_intent",
    filename="reaction_intent.csv",
    primary_key=("intent_id",),
    columns=(
        Column("intent_id", "string", key="pk"),
        Column("curve_id", "string", key="fk:irc_curve.curve_id"),
        Column("direction", "string", enum=("forward", "reverse")),
        Column("reverse_of_intent_id", "string", key="fk:reaction_intent.intent_id"),
        Column("intent_pair_id", "string"),
        Column("intent_underdetermined", "bool"),
        Column("g_src", "string"),
        Column("g_tgt", "string"),
        Column("delta_b_intent", "string"),
        Column("cv_spec", "string"),
        Column("intent_role", "string", required=False, nullable=True),
        Column("rules", "string", required=False, nullable=True),
        Column("intent_version", "string", required=False, nullable=True),
    ),
    unique=(("curve_id", "intent_id"),),
    notes=(
        "unique key (curve_id, intent_id) — intent_id is not assumed globally "
        "unique (ORA-2 #2)",
        "exactly 2 intents per curve forming one mutual-reverse pair unless "
        "intent_underdetermined (T0.2a)",
    ),
)

CURVE_QUALITY = TableSchema(
    name="curve_quality",
    filename="curve_quality.csv",
    primary_key=("curve_id",),
    columns=(
        Column("curve_id", "string", key="pk;fk:irc_curve.curve_id"),
        Column("quality_status", "string"),
        Column("quality_score", "float", required=False, nullable=True),
        Column("segment_quality", "string", required=False, nullable=True),
        Column("branch_quality", "string", required=False, nullable=True),
        Column("note", "string", required=False, nullable=True),
    ),
    notes=("segment/branch quality columns are provenance-only; no branch id "
           "may enter network inputs, targets or split keys (plan §3.1)",),
)

SAMPLE_VIEW = TableSchema(
    name="sample_view",
    filename="sample_view.csv",
    primary_key=("sample_id",),
    columns=(
        Column("sample_id", "string", key="pk"),
        Column("curve_id", "string", key="fk:irc_curve.curve_id"),
        Column("node_index", "int"),
        Column("intent_id", "string", key="fk:reaction_intent.intent_id"),
        Column("d0", "float"),
        Column("x", "float"),
        Column("y", "float"),
        Column("z", "float"),
        # ORA-2 #3 label / weight block
        Column("label_weight", "float"),
        Column("curve_weight", "float"),
        Column("sample_weight", "float"),
        Column("label_status", "string", enum=LABEL_STATUS_VALUES),
        Column("feature_sha", "string"),
        Column("label_sha", "string"),
        # ORA-2 #5 d0 / gauge block
        Column("sensitivity_only", "bool"),
        Column("gauge_ref", "string", enum=GAUGE_REF_VALUES),
        # teacher targets (optional)
        Column("target_node_index", "int", required=False, nullable=True),
        Column("target_action_u", "string", required=False, nullable=True),
        Column("target_progress_q", "float", required=False, nullable=True),
        Column("stop_label", "int", required=False, nullable=True),
        Column("mask", "bool", required=False, nullable=True),
        Column("fold", "int", required=False, nullable=True),
    ),
    unique=(("curve_id", "node_index", "intent_id", "d0"),),
    notes=(
        "sample_id is the surrogate of the composite key "
        "(curve_id, node_index, intent_id, d0) per contract §9; fold is "
        "informational — split_manifest_r3.json is authoritative",
        "per (curve_id, d0): sum(curve_weight) == CURVE_WEIGHT_TARGET (1.0); "
        "d0==0.08 rows are main-task, other d0 rows need sensitivity_only=true; "
        "gauge_ref=unresolved_masked requires label_status=masked",
    ),
)

TABLES: Dict[str, TableSchema] = {
    t.name: t
    for t in (
        IRC_CURVE,
        IRC_NODE,
        IRC_EDGE,
        REACTION_INTENT,
        CURVE_QUALITY,
        SAMPLE_VIEW,
    )
}

TABLE_ORDER: Tuple[str, ...] = (
    "irc_curve",
    "irc_node",
    "irc_edge",
    "reaction_intent",
    "curve_quality",
    "sample_view",
)

# Composite foreign keys: (child, child_cols, parent, parent_cols).
# Every child column tuple must match a parent primary-key tuple exactly.
COMPOSITE_FK: Tuple[Tuple[str, Tuple[str, ...], str, Tuple[str, ...]], ...] = (
    ("irc_edge", ("curve_id", "src_node_index"), "irc_node", ("curve_id", "node_index")),
    ("irc_edge", ("curve_id", "dst_node_index"), "irc_node", ("curve_id", "node_index")),
    ("sample_view", ("curve_id", "node_index"), "irc_node", ("curve_id", "node_index")),
)

# ---------------------------------------------------------------------------
# Manifests (30h plan §3.1 / §7.1)
# ---------------------------------------------------------------------------

SPLIT_MANIFEST_FILENAME = "split_manifest_r3.json"
SPLIT_MANIFEST_FALLBACK_FILENAME = "split_manifest.json"
REQUIRED_MANIFESTS: Tuple[str, ...] = ("data_manifest.json", "teacher_manifest.json")

# Expected shapes (both accepted for T0.4a):
#   {"curve_folds": {curve_id: fold}}
#   {"folds": {fold: [curve_id, ...]}}
#   {"assignments": [{"curve_id": ..., "fold": ..., "near_dup_group": ...}]}
#   [ {...}, ... ]                       (top-level assignment list)
# Optional near-duplicate grouping:
#   {"near_dup_groups": {group_id: [curve_id, ...]}}
#   or a "near_dup_group" field inside each assignment record.
SPLIT_MANIFEST_SHAPES: Tuple[str, ...] = (
    "curve_folds",
    "folds",
    "assignments",
    "assignment_list",
)

# Expected shape for data_manifest.json / teacher_manifest.json:
#   {"files": [{"path": "<relative to manifest dir>", "sha256": "<64 hex>"}, ...]}
# ({"files": {path: sha256}} mapping also accepted)
MANIFEST_ENTRY_LIST_KEY = "files"
MANIFEST_SHA_KEY = "sha256"


def table_filenames() -> Tuple[str, ...]:
    """Filenames of the six tables in canonical order."""
    return tuple(TABLES[name].filename for name in TABLE_ORDER)


def required_filenames() -> Tuple[str, ...]:
    """All filenames a complete tables-dir must contain (6 CSV + 3 JSON)."""
    return table_filenames() + (
        SPLIT_MANIFEST_FILENAME,
        *REQUIRED_MANIFESTS,
    )
