"""Declarative schema for the six R3 data tables of the 30h plan (dry-run).

Source of truth (referenced read-only; this module never opens ``data/``):

* ``docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md``
  — §3.1 (six tables + three JSON manifests), §3.2 (Data Gate), §7.1 (frozen
  manifest files), §7.4 (test ids T0.1 / T0.4).
* ``docs/contracts/PES2TS_ReactionIntent_Contract_v1.md``
  — §9 数据合同 / 骨架表 (L425–445): ``irc_curve``, ``irc_node``,
  ``sample_view`` skeleton fields.

Where the two documents disagree the 30h plan's **six-table** layout wins and
the discrepancy is recorded in :data:`SCHEMA_NOTES`; every validation report
embeds those notes so a machine consumer sees the discrepancy without
re-reading the docs.

Scope note (dry-run grade): the validator enforces column presence, FK
integrity, TS uniqueness/order, coordinate finiteness and manifest structure.
Declared ``dtype`` / ``enum`` values are contract for the future E1 data
builder but are not yet value-enforced by the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

SCHEMA_VERSION = "r3_6table_v1"
VALIDATOR_VERSION = "0.1.0-dryrun"

SCHEMA_SOURCE: Tuple[str, ...] = (
    "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md §3.1-3.2",
    "docs/plans/PES2TS_30h_R3_V52_V53_分层训练与验收计划_v1.1_20261020.md §7.1, §7.4",
    "docs/contracts/PES2TS_ReactionIntent_Contract_v1.md §9 数据合同 (骨架表, L425-445)",
)

SCHEMA_NOTES: Tuple[str, ...] = (
    "table_count_discrepancy: the 30h plan §3.1 mandates six tables "
    "(irc_curve, irc_node, irc_edge, reaction_intent, curve_quality, sample_view) "
    "and explicitly says '共六张，不是五张' (six, not five), while ReactionIntent "
    "Contract v1 §骨架表 (L425-445) only skeletons three tables (irc_curve, "
    "irc_node, sample_view) from the earlier reduced lineage. Implemented: the "
    "6-table 30h layout; contract skeleton fields are mapped onto it.",
    "split_manifest_name: plan §3.1 names split_manifest_r3.json, but the §7.1 "
    "freeze list names split_manifest.json. Validator prefers "
    "split_manifest_r3.json and records which file it used.",
    "irc_curve in the contract carries progress[]/nodes[] arrays; the CSV row "
    "flattens them to n_nodes/n_edges/ts_index (per-node rows live in "
    "irc_node, per-step rows in irc_edge).",
    "d0 is not part of curve identity (plan §3.1); it lives in the sample key "
    "(curve_id, node_index, intent_id, d0) of sample_view and in the teacher "
    "protocol, never in irc_curve.",
    "curve_quality.segment_quality/branch_quality are provenance-only (plan "
    "§3.1 '允许保留历史 segment/branch quality 供溯源'): they must never become "
    "split keys, model inputs or identifiers.",
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
# The six tables (30h plan §3.1)
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
    primary_key=("curve_id", "edge_index"),
    columns=(
        Column("curve_id", "string", key="pk;fk:irc_curve.curve_id"),
        Column("edge_index", "int", key="pk"),
        # src/dst form a composite FK (curve_id, src|dst node_index) into
        # irc_node; see COMPOSITE_FK below.
        Column("src_node_index", "int"),
        Column("dst_node_index", "int"),
        Column("delta_dx", "float", required=False, nullable=True),
        Column("delta_dy", "float", required=False, nullable=True),
        Column("delta_dz", "float", required=False, nullable=True),
        Column("distance", "float", required=False, nullable=True),
    ),
    notes=("raw irc_edge.delta_xyz is kept separate from derived teacher_action_key "
           "(plan §1.2)",),
)

REACTION_INTENT = TableSchema(
    name="reaction_intent",
    filename="reaction_intent.csv",
    primary_key=("intent_id",),
    columns=(
        Column("intent_id", "string", key="pk"),
        Column("curve_id", "string", key="fk:irc_curve.curve_id"),
        Column("direction", "string", enum=("forward", "reverse")),
        Column("g_src", "string"),
        Column("g_tgt", "string"),
        Column("delta_b_intent", "string"),
        Column("cv_spec", "string"),
        Column("intent_role", "string", required=False, nullable=True),
        Column("rules", "string", required=False, nullable=True),
    ),
    notes=("every curve carries two opposed intents (plan §3.1); direction "
           "'reverse' must be a real reversed task, not a relabel",),
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
        Column("target_node_index", "int", required=False, nullable=True),
        Column("target_action_u", "string", required=False, nullable=True),
        Column("target_progress_q", "float", required=False, nullable=True),
        Column("stop_label", "int", required=False, nullable=True),
        Column("mask", "bool", required=False, nullable=True),
        Column("fold", "int", required=False, nullable=True),
    ),
    unique=(("curve_id", "node_index", "intent_id", "d0"),),
    notes=("sample_id is the surrogate of the composite key "
           "(curve_id, node_index, intent_id, d0) per contract §9; fold is "
           "informational — split_manifest_r3.json is authoritative",),
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
