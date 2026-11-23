#!/usr/bin/env python
"""Build the G1 per-reaction summary Parquet from scattered interim JSON files.

Purpose
-------
`data/interim/` holds ~1.36 M small per-reaction G1 JSON files (6 sources,
~9.1 GiB).  This script streams every source file once, extracts a flat set of
scalar / small-list summary fields, and writes one queryable Parquet table at
`/pes2ts_datastore/02_core_tensors/g1_metadata/g1_reaction_summary.parquet`
(1 row per inventory reaction_id), plus a provenance manifest.

Guarantees
----------
* Source data is READ-ONLY.  Nothing under the repo or datastore input paths is
  deleted, moved or modified.
* Streaming: at most `--chunk-size` rows / `--workers` in-flight files are held
  in memory; the 1.36 M JSONs are never materialised at once.
* Bad JSON / read errors are counted and logged (first 20 paths to /tmp), never
  silently skipped.
* Idempotent: deterministic row order (reaction_id ascending), fixed chunk
  boundaries, no timestamps in the Parquet payload -> two runs produce the same
  sha256.  Re-running appends a run record to the manifest.

Usage
-----
    # canonical build (inside the WSL memory guard)
    bash infra/runtime/wsl_guarded_run.sh --mem 6G --name build-g1-summary -- \
        /root/miniforge3/envs/pes2ts-gpu/bin/python \
        pes2ts_data/builders/build_g1_summary.py

    # smoke run
    python pes2ts_data/builders/build_g1_summary.py --limit 500 --out-dir /tmp/g1smoke

Outputs
-------
    <out-dir>/g1_reaction_summary.parquet
    <out-dir>/g1_summary_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
SCHEMA_VERSION = "g1_reaction_summary_v1"
MANIFEST_SCHEMA_VERSION = "g1_summary_manifest_v1"
DEFAULT_INTERIM = "/mnt/e/Calculations/AI4S_ML_Studys/PES2TS/data/interim"
DEFAULT_OUT_DIR = "/pes2ts_datastore/02_core_tensors/g1_metadata"
DEFAULT_INVENTORY = os.path.join(DEFAULT_INTERIM, "inventory.parquet")
DEFAULT_CHUNK = 4000
DEFAULT_WORKERS = 16
DEFAULT_SAMPLE_N = 100
SAMPLE_SEED = 20261123
BAD_JSON_LOG_MAX = 20

# Arrow types -----------------------------------------------------------------
STR, CAT, INT, FLT, BOOL, SHA = "str", "cat", "int", "float", "bool", "sha"

ARROW_TYPES = {
    CAT: pa.dictionary(pa.int32(), pa.string()),
    STR: pa.string(),
    SHA: pa.string(),
    INT: pa.int64(),
    FLT: pa.float64(),
    BOOL: pa.bool_(),
}

Col = namedtuple("Col", "name typ path kind func rule")


# --------------------------------------------------------------------------- #
# JSON value helpers
# --------------------------------------------------------------------------- #
def _s(v):
    return v if isinstance(v, str) else None


def _i(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    return None


def _f(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _b(v):
    return v if isinstance(v, bool) else None


def _len(v):
    if isinstance(v, (list, dict)):
        return len(v)
    return None


def getpath(obj, path):
    """Traverse `a.b[0].c` style paths; None if any step is missing/typed out."""
    if path is None:
        return None
    cur = obj
    for token in path.split("."):
        if not token:
            continue
        name = token
        idx = None
        if token.endswith("]"):
            base, _, rest = token.partition("[")
            name, idx = base, int(rest[:-1])
        if name:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(name)
        if idx is not None:
            if not isinstance(cur, (list, tuple)) or idx >= len(cur):
                return None
            cur = cur[idx]
    return cur


def coerce(v, typ):
    if typ in (STR, CAT, SHA):
        return _s(v)
    if typ == INT:
        return _i(v)
    if typ == FLT:
        return _f(v)
    if typ == BOOL:
        return _b(v)
    raise ValueError(typ)


def _get(path, typ):
    return lambda j: coerce(getpath(j, path), typ)


def _join_str(vals, sep="|"):
    if not isinstance(vals, (list, tuple)):
        return None
    return sep.join(str(v) for v in vals)


def _cnt_dict(d, key):
    if not isinstance(d, dict):
        return None
    v = d.get(key, 0)
    return _i(v) if _i(v) is not None else 0


def _count_kind(items, kind):
    if not isinstance(items, list):
        return None
    return sum(1 for e in items if isinstance(e, dict) and e.get("edit_kind") == kind)


def _count_status(entries, want_unique):
    if not isinstance(entries, list):
        return None
    n = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        st = e.get("status")
        if (st == "unique") == want_unique:
            n += 1
    return n


def _tags(entries):
    if not isinstance(entries, list):
        return None
    return "|".join(str(e.get("tag")) for e in entries if isinstance(e, dict))


# --------------------------------------------------------------------------- #
# Source column specifications
# --------------------------------------------------------------------------- #
SOURCE_KEYS = [
    "reaction_change",
    "p1_mapping",
    "reaction_classes",
    "v2_classes",
    "v2_edits",
    "v2_export",
]
HAS_COL = {
    "reaction_change": "has_reaction_change",
    "p1_mapping": "has_p1_mapping",
    "reaction_classes": "has_reaction_class",
    "v2_classes": "has_v2_class",
    "v2_edits": "has_v2_edits",
    "v2_export": "has_v2_export",
}
SHA_COL = {
    "reaction_change": "change_file_sha256",
    "p1_mapping": "p1_file_sha256",
    "reaction_classes": "rc_file_sha256",
    "v2_classes": "vc_file_sha256",
    "v2_edits": "ve_file_sha256",
    "v2_export": "ex_file_sha256",
}
SCHEMA_COL = {
    "reaction_change": "change_schema_version",
    "p1_mapping": "p1_schema_version",
    "reaction_classes": "rc_schema_version",
    "v2_classes": "vc_schema_version",
    "v2_edits": "ve_schema_version",
    "v2_export": "ex_schema_version",
}
REL_DIR = {
    "reaction_change": "g1/reaction_change",
    "p1_mapping": "g1_truth/p1_mapping",
    "reaction_classes": "g1_truth/reaction_classes",
    "v2_classes": "g1_v2/classes",
    "v2_edits": "g1_v2/edits",
    "v2_export": "g1_v2/export",
}

# Each entry: Col(name, typ, path, kind, func, rule)
#   kind == "get"  -> value comes from literal JSON path + type coercion
#                     (this is what the verifier re-checks independently)
#   kind == "comp" -> computed from the decoded object via ``func``
CHANGE_COLS = [
    Col("change_schema_version", CAT, "schema_version", "get", None, "JSON `schema_version`"),
    Col("change_dataset_version", CAT, "dataset_version", "get", None, "JSON `dataset_version`"),
    Col("change_generated_at", STR, "generated_at", "get", None, "JSON `generated_at` (ISO-8601 string)"),
    Col("change_reaction_smiles_sha256", SHA, "source.reaction_smiles_sha256", "get", None, "JSON `source.reaction_smiles_sha256`"),
    Col("change_validation_status", CAT, "validation.status", "get", None, "JSON `validation.status` (valid/rejected)"),
    Col("change_validation_failure_code", CAT, "validation.failure_code", "get", None, "JSON `validation.failure_code` (null when valid)"),
    Col("change_ambiguity_index", CAT, "ambiguity.index", "get", None, "JSON `ambiguity.index`"),
    Col("change_ambiguity_pairing", CAT, "ambiguity.pairing", "get", None, "JSON `ambiguity.pairing`"),
    Col("change_n_atoms", INT, "mapping.n_atoms", "get", None, "JSON `mapping.n_atoms`"),
    Col("change_n_reactant_components", INT, "mapping.n_reactant_components", "get", None, "JSON `mapping.n_reactant_components`"),
    Col("change_n_product_components", INT, "mapping.n_product_components", "get", None, "JSON `mapping.n_product_components`"),
    Col("change_charge_consistent", BOOL, "mapping.charge_consistent", "get", None, "JSON `mapping.charge_consistent`"),
    Col("change_spin_consistent", BOOL, "mapping.spin_consistent", "get", None, "JSON `mapping.spin_consistent`"),
    Col("change_map_sets_equal", BOOL, "mapping.map_sets_equal", "get", None, "JSON `mapping.map_sets_equal`"),
    Col("change_maps_unique", BOOL, "mapping.maps_unique", "get", None, "JSON `mapping.maps_unique`"),
    Col("change_n_reactant_entries", INT, None, "comp", lambda j: _len(j.get("reactants")), "len(JSON `reactants`)"),
    Col("change_n_product_entries", INT, None, "comp", lambda j: _len(j.get("products")), "len(JSON `products`)" ),
    Col("change_reactant_tags", STR, None, "comp", lambda j: _tags(j.get("reactants")), "`reactants[].tag` joined with '|'"),
    Col("change_product_tags", STR, None, "comp", lambda j: _tags(j.get("products")), "`products[].tag` joined with '|'"),
    Col("change_n_entries_unique", INT, None, "comp",
        lambda j: None if not isinstance(j.get("reactants"), list) or not isinstance(j.get("products"), list)
        else _count_status(j["reactants"], True) + _count_status(j["products"], True),
        "count of components with status=='unique' over reactants+products"),
    Col("change_n_entries_ambiguous", INT, None, "comp",
        lambda j: None if not isinstance(j.get("reactants"), list) or not isinstance(j.get("products"), list)
        else _count_status(j["reactants"], False) + _count_status(j["products"], False),
        "count of components with status!='unique' over reactants+products"),
    Col("change_cat_both", BOOL, "categories.both", "get", None, "JSON `categories.both`"),
    Col("change_cat_h_migration", BOOL, "categories.h_migration", "get", None, "JSON `categories.h_migration`"),
    Col("change_cat_has_order_change", BOOL, "categories.has_order_change", "get", None, "JSON `categories.has_order_change`"),
    Col("change_cat_multi_component", BOOL, "categories.multi_component", "get", None, "JSON `categories.multi_component`"),
    Col("change_cat_order_change_only", BOOL, "categories.order_change_only", "get", None, "JSON `categories.order_change_only`"),
    Col("change_cat_pure_broken", BOOL, "categories.pure_broken", "get", None, "JSON `categories.pure_broken`"),
    Col("change_cat_pure_formed", BOOL, "categories.pure_formed", "get", None, "JSON `categories.pure_formed`"),
    Col("change_n_bonds_broken", INT, "bond_changes.preview_counters.n_bonds_broken", "get", None, "JSON `bond_changes.preview_counters.n_bonds_broken`"),
    Col("change_n_bonds_formed", INT, "bond_changes.preview_counters.n_bonds_formed", "get", None, "JSON `bond_changes.preview_counters.n_bonds_formed`"),
    Col("change_n_bond_order_changed", INT, "bond_changes.preview_counters.n_bond_order_changed", "get", None, "JSON `bond_changes.preview_counters.n_bond_order_changed`"),
    Col("change_preview_match", BOOL, "bond_changes.preview_match", "get", None, "JSON `bond_changes.preview_match`"),
    Col("change_n_hydrogen_migration", INT, None, "comp", lambda j: _len(getpath(j, "bond_changes.hydrogen_migration")), "len(JSON `bond_changes.hydrogen_migration`)"),
    Col("change_center_core_size", INT, None, "comp", lambda j: _len(getpath(j, "reaction_center.core")), "len(JSON `reaction_center.core`)"),
    Col("change_center_shell_size", INT, None, "comp", lambda j: _len(getpath(j, "reaction_center.with_shell")), "len(JSON `reaction_center.with_shell`)"),
    Col("change_file_sha256", SHA, None, "comp", None, "sha256 of the raw source JSON file bytes"),
]

P1_COLS = [
    Col("p1_schema_version", CAT, "schema_version", "get", None, "JSON `schema_version`"),
    Col("p1_generated_at", STR, "generated_at", "get", None, "JSON `generated_at`"),
    Col("p1_status", CAT, "status", "get", None, "JSON `status`"),
    Col("p1_status_detail", STR, "status_detail", "get", None, "JSON `status_detail`"),
    Col("p1_truth_assisted", BOOL, "truth_assisted", "get", None, "JSON `truth_assisted`"),
    Col("p1_validation_status", CAT, "validation.status", "get", None, "JSON `validation.status`"),
    Col("p1_validation_n_reasons", INT, None, "comp", lambda j: _len(getpath(j, "validation.reasons")), "len(JSON `validation.reasons`)"),
    Col("p1_mapping_algorithm", CAT, "mapping.algorithm", "get", None, "JSON `mapping.algorithm`"),
    Col("p1_map_to_atoms_len", INT, None, "comp", lambda j: _len(getpath(j, "mapping.map_to_atoms")), "len(JSON `mapping.map_to_atoms`)"),
    Col("p1_n_combinations_p", INT, "mapping.n_combinations_p", "get", None, "JSON `mapping.n_combinations_p`"),
    Col("p1_n_combinations_r", INT, "mapping.n_combinations_r", "get", None, "JSON `mapping.n_combinations_r`"),
    Col("p1_n_equivalence_classes_p", INT, "mapping.n_equivalence_classes_p", "get", None, "JSON `mapping.n_equivalence_classes_p`"),
    Col("p1_n_equivalence_classes_r", INT, "mapping.n_equivalence_classes_r", "get", None, "JSON `mapping.n_equivalence_classes_r`"),
    Col("p1_selected_class_size_p", INT, "mapping.selected_class_size_p", "get", None, "JSON `mapping.selected_class_size_p`"),
    Col("p1_selected_class_size_r", INT, "mapping.selected_class_size_r", "get", None, "JSON `mapping.selected_class_size_r`"),
    Col("p1_selected_rmsd_p", FLT, "mapping.selected_rmsd_p", "get", None, "JSON `mapping.selected_rmsd_p`"),
    Col("p1_selected_rmsd_r", FLT, "mapping.selected_rmsd_r", "get", None, "JSON `mapping.selected_rmsd_r`"),
    Col("p1_symmetry_collapsed", BOOL, "mapping.symmetry_collapsed", "get", None, "JSON `mapping.symmetry_collapsed`"),
    Col("p1_ts_row_algorithm", CAT, "mapping.ts_row_algorithm", "get", None, "JSON `mapping.ts_row_algorithm`"),
    Col("p1_n_atoms", INT, "irc_validation.n_atoms", "get", None, "JSON `irc_validation.n_atoms`"),
    Col("p1_n_frames", INT, "irc_validation.n_frames", "get", None, "JSON `irc_validation.n_frames`"),
    Col("p1_n_support", INT, "irc_validation.n_support", "get", None, "JSON `irc_validation.n_support`"),
    Col("p1_n_weak", INT, "irc_validation.n_weak", "get", None, "JSON `irc_validation.n_weak`"),
    Col("p1_n_mismatch", INT, "irc_validation.n_mismatch", "get", None, "JSON `irc_validation.n_mismatch`"),
    Col("p1_endpoint_match", CAT, "irc_validation.endpoint_match", "get", None, "JSON `irc_validation.endpoint_match`"),
    Col("p1_synchrony", CAT, "irc_validation.synchrony", "get", None, "JSON `irc_validation.synchrony`"),
    Col("p1_orientation", CAT, "irc_validation.orientation", "get", None, "JSON `irc_validation.orientation`"),
    Col("p1_layout", CAT, "irc_validation.layout", "get", None, "JSON `irc_validation.layout`"),
    Col("p1_splice_index", INT, "irc_validation.splice_index", "get", None, "JSON `irc_validation.splice_index`"),
    Col("p1_ts_frame_index", INT, "irc_validation.ts_frame_index", "get", None, "JSON `irc_validation.ts_frame_index`"),
    Col("p1_ts_frame_rmsd", FLT, "irc_validation.ts_frame_rmsd", "get", None, "JSON `irc_validation.ts_frame_rmsd`"),
    Col("p1_max_progress_delta", FLT, "irc_validation.max_progress_delta", "get", None, "JSON `irc_validation.max_progress_delta`"),
    Col("p1_endpoint_rmsd_a_to_p", FLT, "irc_validation.endpoint_rmsd.branch_a_to_p", "get", None, "JSON `irc_validation.endpoint_rmsd.branch_a_to_p`"),
    Col("p1_endpoint_rmsd_a_to_r", FLT, "irc_validation.endpoint_rmsd.branch_a_to_r", "get", None, "JSON `irc_validation.endpoint_rmsd.branch_a_to_r`"),
    Col("p1_endpoint_rmsd_b_to_p", FLT, "irc_validation.endpoint_rmsd.branch_b_to_p", "get", None, "JSON `irc_validation.endpoint_rmsd.branch_b_to_p`"),
    Col("p1_endpoint_rmsd_b_to_r", FLT, "irc_validation.endpoint_rmsd.branch_b_to_r", "get", None, "JSON `irc_validation.endpoint_rmsd.branch_b_to_r`"),
    Col("p1_n_quality_flags", INT, None, "comp", lambda j: _len(getpath(j, "irc_validation.quality_flags")), "len(JSON `irc_validation.quality_flags`)"),
    Col("p1_graph_present", BOOL, None, "comp", lambda j: isinstance(j.get("graph"), dict), "JSON `graph` is a dict"),
    Col("p1_n_graph_atoms", INT, None, "comp", lambda j: _len(getpath(j, "graph.elements")), "len(JSON `graph.elements`)"),
    Col("p1_n_graph_r_bonds", INT, None, "comp", lambda j: _len(getpath(j, "graph.r_bonds")), "len(JSON `graph.r_bonds`)"),
    Col("p1_n_graph_p_bonds", INT, None, "comp", lambda j: _len(getpath(j, "graph.p_bonds")), "len(JSON `graph.p_bonds`)"),
    Col("p1_n_graph_r_components", INT, None, "comp", lambda j: _len(getpath(j, "graph.r_components")), "len(JSON `graph.r_components`)"),
    Col("p1_n_graph_p_components", INT, None, "comp", lambda j: _len(getpath(j, "graph.p_components")), "len(JSON `graph.p_components`)"),
    Col("p1_center_core_size", INT, None, "comp", lambda j: _len(getpath(j, "reaction_center.core_maps")), "len(JSON `reaction_center.core_maps`)"),
    Col("p1_center_shell_size", INT, None, "comp", lambda j: _len(getpath(j, "reaction_center.shell1_maps")), "len(JSON `reaction_center.shell1_maps`)"),
    Col("p1_bond_events_broken", INT, None, "comp", lambda j: _len(getpath(j, "bond_events.broken")), "len(JSON `bond_events.broken`)"),
    Col("p1_bond_events_formed", INT, None, "comp", lambda j: _len(getpath(j, "bond_events.formed")), "len(JSON `bond_events.formed`)"),
    Col("p1_bond_events_order_changed", INT, None, "comp", lambda j: _len(getpath(j, "bond_events.order_changed")), "len(JSON `bond_events.order_changed`)"),
    Col("p1_bond_events_hydrogen_migration", INT, None, "comp", lambda j: _len(getpath(j, "bond_events.hydrogen_migration")), "len(JSON `bond_events.hydrogen_migration`)"),
    Col("p1_source_dataset_version", CAT, "sources.dataset_version", "get", None, "JSON `sources.dataset_version`"),
    Col("p1_file_sha256", SHA, None, "comp", None, "sha256 of the raw source JSON file bytes"),
]

RC_COLS = [
    Col("rc_schema_version", CAT, "schema_version", "get", None, "JSON `schema_version`"),
    Col("rc_generated_at", STR, "generated_at", "get", None, "JSON `generated_at`"),
    Col("rc_status", CAT, "status", "get", None, "JSON `status`"),
    Col("rc_status_detail", STR, "status_detail", "get", None, "JSON `status_detail`"),
    Col("rc_p1_status", CAT, "p1_status", "get", None, "JSON `p1_status`"),
    Col("rc_taxonomy_version", CAT, "taxonomy_version", "get", None, "JSON `taxonomy_version`"),
    Col("rc_n_center_atoms", INT, "n_center_atoms", "get", None, "JSON `n_center_atoms`"),
    Col("rc_n_context_r1", INT, "n_context_r1", "get", None, "JSON `n_context_r1`"),
    Col("rc_n_context_r2", INT, "n_context_r2", "get", None, "JSON `n_context_r2`"),
    Col("rc_n_family_labels", INT, None, "comp", lambda j: _len(j.get("family_labels")), "len(JSON `family_labels`)"),
    Col("rc_family_label_primary", CAT, "family_labels[0]", "get", None, "JSON `family_labels[0]`"),
    Col("rc_family_labels", STR, None, "comp", lambda j: _join_str(j.get("family_labels")), "JSON `family_labels` joined with '|'"),
    Col("rc_n_flags", INT, None, "comp", lambda j: _len(j.get("flags")), "len(JSON `flags`)"),
    Col("rc_flags", STR, None, "comp", lambda j: _join_str(j.get("flags")), "JSON `flags` joined with '|'"),
    Col("rc_l0_signature", CAT, "levels.l0_edit_family.signature", "get", None, "JSON `levels.l0_edit_family.signature`"),
    Col("rc_l0_cluster_id", STR, "levels.l0_edit_family.cluster_id", "get", None, "JSON `levels.l0_edit_family.cluster_id`"),
    Col("rc_l1_cluster_id", STR, "levels.l1_center_template.cluster_id", "get", None, "JSON `levels.l1_center_template.cluster_id`"),
    Col("rc_l2_cluster_id", STR, "levels.l2_context_r1.cluster_id", "get", None, "JSON `levels.l2_context_r1.cluster_id`"),
    Col("rc_l3_cluster_id", STR, "levels.l3_context_r2.cluster_id", "get", None, "JSON `levels.l3_context_r2.cluster_id`"),
    Col("rc_l1_n_nodes", INT, "levels.l1_center_template.n_nodes", "get", None, "JSON `levels.l1_center_template.n_nodes`"),
    Col("rc_l2_n_nodes", INT, "levels.l2_context_r1.n_nodes", "get", None, "JSON `levels.l2_context_r1.n_nodes`"),
    Col("rc_l3_n_nodes", INT, "levels.l3_context_r2.n_nodes", "get", None, "JSON `levels.l3_context_r2.n_nodes`"),
    Col("rc_endpoint_match", CAT, "pathway_labels.endpoint_match", "get", None, "JSON `pathway_labels.endpoint_match`"),
    Col("rc_synchrony", CAT, "pathway_labels.synchrony", "get", None, "JSON `pathway_labels.synchrony`"),
    Col("rc_irc_orientation", CAT, "pathway_labels.irc_orientation", "get", None, "JSON `pathway_labels.irc_orientation`"),
    Col("rc_evt_support", INT, "pathway_labels.event_support_histogram.support", "get", None, "JSON `pathway_labels.event_support_histogram.support`"),
    Col("rc_evt_weak", INT, "pathway_labels.event_support_histogram.weak", "get", None, "JSON `pathway_labels.event_support_histogram.weak`"),
    Col("rc_evt_mismatch", INT, "pathway_labels.event_support_histogram.mismatch", "get", None, "JSON `pathway_labels.event_support_histogram.mismatch`"),
    Col("rc_file_sha256", SHA, None, "comp", None, "sha256 of the raw source JSON file bytes"),
]

VC_COLS = [
    Col("vc_schema_version", CAT, "schema_version", "get", None, "JSON `schema_version`"),
    Col("vc_generated_at", STR, "generated_at", "get", None, "JSON `generated_at`"),
    Col("vc_audit_status", CAT, "audit_status", "get", None, "JSON `audit_status`"),
    Col("vc_status", CAT, "status", "get", None, "JSON `status`"),
    Col("vc_status_detail", STR, "status_detail", "get", None, "JSON `status_detail`"),
    Col("vc_p1_status", CAT, "p1_status", "get", None, "JSON `p1_status`"),
    Col("vc_taxonomy_version", CAT, "taxonomy_version", "get", None, "JSON `taxonomy_version`"),
    Col("vc_n_center_atoms", INT, "n_center_atoms", "get", None, "JSON `n_center_atoms`"),
    Col("vc_n_context_r1", INT, "n_context_r1", "get", None, "JSON `n_context_r1`"),
    Col("vc_n_context_r2", INT, "n_context_r2", "get", None, "JSON `n_context_r2`"),
    Col("vc_n_family_labels", INT, None, "comp", lambda j: _len(j.get("family_labels")), "len(JSON `family_labels`)"),
    Col("vc_family_label_primary", CAT, "family_labels[0]", "get", None, "JSON `family_labels[0]`"),
    Col("vc_family_labels", STR, None, "comp", lambda j: _join_str(j.get("family_labels")), "JSON `family_labels` joined with '|'"),
    Col("vc_n_flags", INT, None, "comp", lambda j: _len(j.get("flags")), "len(JSON `flags`)"),
    Col("vc_flags", STR, None, "comp", lambda j: _join_str(j.get("flags")), "JSON `flags` joined with '|'"),
    Col("vc_l0_signature", CAT, "levels.l0_edit_family.signature", "get", None, "JSON `levels.l0_edit_family.signature`"),
    Col("vc_l0u_signature", CAT, "levels.l0u_undirected_family.signature", "get", None, "JSON `levels.l0u_undirected_family.signature`"),
    Col("vc_l0_cluster_id", STR, "levels.l0_edit_family.cluster_id", "get", None, "JSON `levels.l0_edit_family.cluster_id`"),
    Col("vc_l0u_cluster_id", STR, "levels.l0u_undirected_family.cluster_id", "get", None, "JSON `levels.l0u_undirected_family.cluster_id`"),
    Col("vc_l1_cluster_id", STR, "levels.l1_center_template.cluster_id", "get", None, "JSON `levels.l1_center_template.cluster_id`"),
    Col("vc_l2_cluster_id", STR, "levels.l2_context_r1.cluster_id", "get", None, "JSON `levels.l2_context_r1.cluster_id`"),
    Col("vc_l3_cluster_id", STR, "levels.l3_context_r2.cluster_id", "get", None, "JSON `levels.l3_context_r2.cluster_id`"),
    Col("vc_l1_n_nodes", INT, "levels.l1_center_template.n_nodes", "get", None, "JSON `levels.l1_center_template.n_nodes`"),
    Col("vc_l2_n_nodes", INT, "levels.l2_context_r1.n_nodes", "get", None, "JSON `levels.l2_context_r1.n_nodes`"),
    Col("vc_l3_n_nodes", INT, "levels.l3_context_r2.n_nodes", "get", None, "JSON `levels.l3_context_r2.n_nodes`"),
    Col("vc_file_sha256", SHA, None, "comp", None, "sha256 of the raw source JSON file bytes"),
]

VE_COLS = [
    Col("ve_schema_version", CAT, "schema_version", "get", None, "JSON `schema_version`"),
    Col("ve_dataset_version", CAT, "dataset_version", "get", None, "JSON `dataset_version`"),
    Col("ve_generated_at", STR, "generated_at", "get", None, "JSON `generated_at`"),
    Col("ve_p1_status", CAT, "p1_status", "get", None, "JSON `p1_status`"),
    Col("ve_audit_status", CAT, "audit_status", "get", None, "JSON `audit_status`"),
    Col("ve_collapse_state", CAT, "collapse_state", "get", None, "JSON `collapse_state`"),
    Col("ve_legacy_reconciled", BOOL, "legacy_reconciled", "get", None, "JSON `legacy_reconciled`"),
    Col("ve_n_atoms", INT, "n_atoms", "get", None, "JSON `n_atoms`"),
    Col("ve_n_alts_enumerated", INT, "n_alts_enumerated", "get", None, "JSON `n_alts_enumerated`"),
    Col("ve_n_edits", INT, None, "comp", lambda j: _len(j.get("edits")), "len(JSON `edits`)"),
    Col("ve_n_edits_broken", INT, None, "comp", lambda j: _count_kind(j.get("edits"), "broken"), "count edits[] with edit_kind=='broken'"),
    Col("ve_n_edits_formed", INT, None, "comp", lambda j: _count_kind(j.get("edits"), "formed"), "count edits[] with edit_kind=='formed'"),
    Col("ve_n_edits_order_changed", INT, None, "comp", lambda j: _count_kind(j.get("edits"), "order_changed"), "count edits[] with edit_kind=='order_changed'"),
    Col("ve_edit_counts_broken", INT, None, "comp", lambda j: _cnt_dict(j.get("edit_counts"), "broken"), "JSON `edit_counts.broken` (0 when key absent)"),
    Col("ve_edit_counts_formed", INT, None, "comp", lambda j: _cnt_dict(j.get("edit_counts"), "formed"), "JSON `edit_counts.formed` (0 when key absent)"),
    Col("ve_edit_counts_order_changed", INT, None, "comp", lambda j: _cnt_dict(j.get("edit_counts"), "order_changed"), "JSON `edit_counts.order_changed` (0 when key absent)"),
    Col("ve_edit_counts_transfer", INT, None, "comp", lambda j: _cnt_dict(j.get("edit_counts"), "transfer"), "JSON `edit_counts.transfer` (0 when key absent)"),
    Col("ve_edit_counts_present", BOOL, None, "comp", lambda j: isinstance(j.get("edit_counts"), dict), "JSON `edit_counts` is a dict"),
    Col("ve_n_hydrogen_partner_changes", INT, None, "comp", lambda j: _len(j.get("hydrogen_partner_changes")), "len(JSON `hydrogen_partner_changes`)"),
    Col("ve_n_hpc_transfer", INT, None, "comp",
        lambda j: None if not isinstance(j.get("hydrogen_partner_changes"), list)
        else sum(1 for e in j["hydrogen_partner_changes"] if isinstance(e, dict) and e.get("kind") == "transfer"),
        "count hydrogen_partner_changes[] with kind=='transfer'"),
    Col("ve_n_issues", INT, None, "comp", lambda j: _len(j.get("issues")), "len(JSON `issues`)"),
    Col("ve_issues", STR, None, "comp", lambda j: _join_str(j.get("issues")), "JSON `issues` joined with '|'"),
    Col("ve_n_dimensions", INT, None, "comp", lambda j: _len(j.get("dimensions")), "len(JSON `dimensions`)"),
    Col("ve_dimensions", STR, None, "comp", lambda j: _join_str(j.get("dimensions")), "JSON `dimensions` joined with '|'"),
    Col("ve_n_aromatic_regions", INT, None, "comp", lambda j: _len(j.get("aromatic_regions")), "len(JSON `aromatic_regions`)"),
    Col("ve_mapping_bijection_ok", BOOL, "mapping_audit.bijection_ok", "get", None, "JSON `mapping_audit.bijection_ok`"),
    Col("ve_mapping_component_assignment_ok", BOOL, "mapping_audit.component_assignment_ok", "get", None, "JSON `mapping_audit.component_assignment_ok`"),
    Col("ve_mapping_elements_ok", BOOL, "mapping_audit.elements_ok", "get", None, "JSON `mapping_audit.elements_ok`"),
    Col("ve_irc_endpoint_match", CAT, "irc_evidence.endpoint_match", "get", None, "JSON `irc_evidence.endpoint_match`"),
    Col("ve_irc_orientation", CAT, "irc_evidence.orientation", "get", None, "JSON `irc_evidence.orientation`"),
    Col("ve_irc_n_event_mismatch", INT, "irc_evidence.n_event_mismatch", "get", None, "JSON `irc_evidence.n_event_mismatch`"),
    Col("ve_irc_n_quality_flags", INT, "irc_evidence.n_quality_flags", "get", None, "JSON `irc_evidence.n_quality_flags`"),
    Col("ve_center_core_size", INT, None, "comp", lambda j: _len(getpath(j, "reaction_center.core")), "len(JSON `reaction_center.core`)"),
    Col("ve_center_shell_size", INT, None, "comp", lambda j: _len(getpath(j, "reaction_center.with_shell")), "len(JSON `reaction_center.with_shell`)"),
    Col("ve_graph_present", BOOL, None, "comp", lambda j: isinstance(j.get("graph"), dict), "JSON `graph` is a dict"),
    Col("ve_n_graph_atoms", INT, None, "comp", lambda j: _len(getpath(j, "graph.elements")), "len(JSON `graph.elements`)"),
    Col("ve_n_graph_p_bonds", INT, None, "comp", lambda j: _len(getpath(j, "graph.p_bonds")), "len(JSON `graph.p_bonds`)"),
    Col("ve_n_graph_r_bonds", INT, None, "comp", lambda j: _len(getpath(j, "graph.r_bonds")), "len(JSON `graph.r_bonds`)"),
    Col("ve_n_graph_p_components", INT, None, "comp", lambda j: _len(getpath(j, "graph.p_components")), "len(JSON `graph.p_components`)"),
    Col("ve_n_graph_r_components", INT, None, "comp", lambda j: _len(getpath(j, "graph.r_components")), "len(JSON `graph.r_components`)"),
    Col("ve_file_sha256", SHA, None, "comp", None, "sha256 of the raw source JSON file bytes"),
]


def _component_fields(obj, which):
    comps = obj.get("components")
    if not isinstance(comps, dict):
        return None
    tags = sorted(comps)
    if which == "n":
        return len(tags)
    if which == "tags":
        return "|".join(tags)
    vals = []
    for t in tags:
        c = comps.get(t)
        if not isinstance(c, dict):
            vals.append("?")
        elif which == "charges":
            vals.append(f"{t}:{c.get('charge')}")
        elif which == "mults":
            vals.append(f"{t}:{c.get('multiplicity')}")
    return "|".join(vals)


def _count_tag_prefix(obj, prefix):
    comps = obj.get("components")
    if not isinstance(comps, dict):
        return None
    return sum(1 for t in comps if str(t).startswith(prefix))


EX_COLS = [
    Col("ex_schema_version", CAT, "schema_version", "get", None, "JSON `schema_version`"),
    Col("ex_dataset_version", CAT, "dataset_version", "get", None, "JSON `dataset_version`"),
    Col("ex_generated_at", STR, "generated_at", "get", None, "JSON `generated_at`"),
    Col("ex_mapping_provenance", CAT, "mapping_provenance", "get", None, "JSON `mapping_provenance`"),
    Col("ex_atom_order", STR, "atom_order", "get", None, "JSON `atom_order`"),
    Col("ex_n_atoms", INT, None, "comp", lambda j: _len(j.get("p_atomic_numbers")), "len(JSON `p_atomic_numbers`)"),
    Col("ex_n_maps", INT, None, "comp", lambda j: _len(j.get("maps")), "len(JSON `maps`)"),
    Col("ex_n_elements", INT, None, "comp", lambda j: _len(j.get("elements")), "len(JSON `elements`)"),
    Col("ex_n_atom_rows", INT, None, "comp", lambda j: _len(j.get("atom_rows")), "len(JSON `atom_rows`)"),
    Col("ex_charge_total_products", INT, "charge_total_products", "get", None, "JSON `charge_total_products`"),
    Col("ex_charge_total_reactants", INT, "charge_total_reactants", "get", None, "JSON `charge_total_reactants`"),
    Col("ex_multiplicity_max", INT, "multiplicity_max", "get", None, "JSON `multiplicity_max`"),
    Col("ex_n_components", INT, None, "comp", lambda j: _component_fields(j, "n"), "len(JSON `components`)"),
    Col("ex_n_reactant_components", INT, None, "comp", lambda j: _count_tag_prefix(j, "R"), "count `components` tags starting with 'R'"),
    Col("ex_n_product_components", INT, None, "comp", lambda j: _count_tag_prefix(j, "P"), "count `components` tags starting with 'P'"),
    Col("ex_component_tags", STR, None, "comp", lambda j: _component_fields(j, "tags"), "sorted JSON `components` tags joined with '|'"),
    Col("ex_component_charges", STR, None, "comp", lambda j: _component_fields(j, "charges"), "sorted `component=<tag>:<charge>` joined with '|'"),
    Col("ex_component_multiplicities", STR, None, "comp", lambda j: _component_fields(j, "mults"), "sorted `component=<tag>:<multiplicity>` joined with '|'"),
    Col("ex_n_family_labels", INT, None, "comp", lambda j: _len(getpath(j, "classification.family_labels")), "len(JSON `classification.family_labels`)"),
    Col("ex_class_family_label_primary", CAT, "classification.family_labels[0]", "get", None, "JSON `classification.family_labels[0]`"),
    Col("ex_class_family_labels", STR, None, "comp", lambda j: _join_str(getpath(j, "classification.family_labels")), "JSON `classification.family_labels` joined with '|'"),
    Col("ex_class_l0_cluster_id", STR, "classification.l0_cluster_id", "get", None, "JSON `classification.l0_cluster_id`"),
    Col("ex_class_l0u_cluster_id", STR, "classification.l0u_cluster_id", "get", None, "JSON `classification.l0u_cluster_id`"),
    Col("ex_class_l1_cluster_id", STR, "classification.l1_cluster_id", "get", None, "JSON `classification.l1_cluster_id`"),
    Col("ex_status_audit_status", CAT, "status_summary.audit_status", "get", None, "JSON `status_summary.audit_status`"),
    Col("ex_status_p1_status", CAT, "status_summary.p1_status", "get", None, "JSON `status_summary.p1_status`"),
    Col("ex_n_edits", INT, None, "comp", lambda j: _len(j.get("edits")), "len(JSON `edits`)"),
    Col("ex_n_edits_broken", INT, None, "comp", lambda j: _count_kind(j.get("edits"), "broken"), "count edits[] with edit_kind=='broken'"),
    Col("ex_n_edits_formed", INT, None, "comp", lambda j: _count_kind(j.get("edits"), "formed"), "count edits[] with edit_kind=='formed'"),
    Col("ex_n_edits_order_changed", INT, None, "comp", lambda j: _count_kind(j.get("edits"), "order_changed"), "count edits[] with edit_kind=='order_changed'"),
    Col("ex_n_hydrogen_partner_changes", INT, None, "comp", lambda j: _len(j.get("hydrogen_partner_changes")), "len(JSON `hydrogen_partner_changes`)"),
    Col("ex_n_hpc_transfer", INT, None, "comp",
        lambda j: None if not isinstance(j.get("hydrogen_partner_changes"), list)
        else sum(1 for e in j["hydrogen_partner_changes"] if isinstance(e, dict) and e.get("kind") == "transfer"),
        "count hydrogen_partner_changes[] with kind=='transfer'"),
    Col("ex_n_aromatic_regions", INT, None, "comp", lambda j: _len(j.get("aromatic_regions")), "len(JSON `aromatic_regions`)"),
    Col("ex_center_core_size", INT, None, "comp", lambda j: _len(getpath(j, "reaction_center.core")), "len(JSON `reaction_center.core`)"),
    Col("ex_center_shell_size", INT, None, "comp", lambda j: _len(getpath(j, "reaction_center.with_shell")), "len(JSON `reaction_center.with_shell`)"),
    Col("ex_file_sha256", SHA, None, "comp", None, "sha256 of the raw source JSON file bytes"),
]

SOURCE_COLS = {
    "reaction_change": CHANGE_COLS,
    "p1_mapping": P1_COLS,
    "reaction_classes": RC_COLS,
    "v2_classes": VC_COLS,
    "v2_edits": VE_COLS,
    "v2_export": EX_COLS,
}
# Data columns per source (file-sha column is produced from raw bytes, not JSON)
DATA_COLS = {k: [c for c in v if c.name != SHA_COL[k]] for k, v in SOURCE_COLS.items()}
SOURCE_OF_COL = {c.name: k for k in SOURCE_KEYS for c in SOURCE_COLS[k]}

COMMON_COLS = [
    Col("reaction_id", STR, None, "inv", None, "inventory.parquet `reaction_id` (primary key, format RXN_<10 digits>)"),
    Col("in_inventory", BOOL, None, "inv", None, "always true: row is driven by inventory.parquet"),
    Col("n_sources_present", INT, None, "comp", None, "number of the 6 G1 sources whose JSON was present, parsed and extracted"),
    Col("n_source_id_mismatch", INT, None, "comp", None, "number of sources whose JSON `reaction_id` differs from the filename/inventory id"),
    Col("has_reaction_change", BOOL, None, "comp", None, "g1/reaction_change/<shard>/<id>.json present & parsed & extracted"),
    Col("has_p1_mapping", BOOL, None, "comp", None, "g1_truth/p1_mapping/<shard>/<id>.json present & parsed & extracted"),
    Col("has_reaction_class", BOOL, None, "comp", None, "g1_truth/reaction_classes/<shard>/<id>.json present & parsed & extracted"),
    Col("has_v2_class", BOOL, None, "comp", None, "g1_v2/classes/<shard>/<id>.json present & parsed & extracted"),
    Col("has_v2_edits", BOOL, None, "comp", None, "g1_v2/edits/<shard>/<id>.json present & parsed & extracted"),
    Col("has_v2_export", BOOL, None, "comp", None, "g1_v2/export/<shard>/<id>.json present & parsed & extracted"),
    Col("inv_dataset_version", CAT, None, "inv", None, "inventory.parquet `dataset_version`"),
    Col("inv_n_reactant_components", INT, None, "inv", None, "inventory.parquet `n_reactant_components`"),
    Col("inv_n_product_components", INT, None, "inv", None, "inventory.parquet `n_product_components`"),
    Col("inv_total_atoms_reactants", INT, None, "inv", None, "inventory.parquet `total_atoms_reactants`"),
    Col("inv_total_atoms_products", INT, None, "inv", None, "inventory.parquet `total_atoms_products`"),
    Col("inv_charge_total_reactants", INT, None, "inv", None, "inventory.parquet `charge_total_reactants`"),
    Col("inv_charge_total_products", INT, None, "inv", None, "inventory.parquet `charge_total_products`"),
    Col("inv_multiplicity_max", INT, None, "inv", None, "inventory.parquet `multiplicity_max`"),
    Col("inv_n_elements", INT, None, "inv", None, "len(inventory.parquet `elements`)"),
]

ALL_COLS = (
    COMMON_COLS
    + [c for k in SOURCE_KEYS for c in SOURCE_COLS[k]]
)
COL_NAMES = [c.name for c in ALL_COLS]
COL_BY_NAME = {c.name: c for c in ALL_COLS}
SCHEMA = pa.schema(
    [
        pa.field(c.name, ARROW_TYPES[c.typ], nullable=(c.name != "reaction_id") and (c.name != "in_inventory"))
        for c in ALL_COLS
    ]
)

OMITTED_FIELDS = {
    "reaction_change": [
        "reactants[] / products[] (component rows, candidate enumerations, geometry status)",
        "bond_changes.broken|formed|order_changed|hydrogen_migration (per-bond lists; counts kept)",
        "reaction_center.core / with_shell (atom-index lists; sizes kept)",
        "validation.failure_detail (free text)",
    ],
    "p1_mapping": [
        "graph.elements|p_bonds|r_bonds|p_components|r_components (full graph; only counts kept)",
        "irc_validation.event_support (per-event rows), orientation_bond_scores, quality_flags (count kept)",
        "mapping.map_to_atoms (atom order list; length kept)",
        "reaction_center.core_maps / shell1_maps (sizes kept)",
        "sources.* digests (provenance; recorded once in manifest env.recon notes)",
    ],
    "reaction_classes": [
        "levels.*.signature_sha256 (redundant with kept cluster_id / signature)",
        "pathway_labels.event_support_histogram (mismatch/weak/support kept as columns)",
        "flags[] / family_labels[] (joined strings kept)",
    ],
    "v2_classes": [
        "levels.*.signature_sha256 (redundant with kept cluster_id / signature)",
        "flags[] / family_labels[] (joined strings kept)",
    ],
    "v2_edits": [
        "edits[] (per-bond edit rows; counts + kind histogram kept)",
        "graph.* bond/component tables (counts kept)",
        "hydrogen_partner_changes[] (count + transfer count kept)",
        "issues[] / dimensions[] (joined strings kept)",
        "aromatic_regions[] (count kept)",
        "reaction_center.core / with_shell (sizes kept)",
    ],
    "v2_export": [
        "p_coordinates / r_coordinates (3xN float arrays - intentionally excluded from Parquet)",
        "p_atomic_numbers / r_atomic_numbers / maps / elements / atom_rows (lengths kept; atom mapping is reconstructable from JSON)",
        "components.*.atomic_numbers (component charge/multiplicity kept, atom count per component omitted)",
        "edits[] (counts + kind histogram kept)",
        "reaction_center.core / with_shell (sizes kept)",
    ],
}

INVENTORY_FIELDS = [
    ("reaction_id", "s"),
    ("dataset_version", "s"),
    ("n_reactant_components", "i"),
    ("n_product_components", "i"),
    ("total_atoms_reactants", "i"),
    ("total_atoms_products", "i"),
    ("charge_total_reactants", "i"),
    ("charge_total_products", "i"),
    ("multiplicity_max", "i"),
]


# --------------------------------------------------------------------------- #
# Memory sampler (cgroup v2 when available, /proc fallback)
# --------------------------------------------------------------------------- #
class MemorySampler(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self.peak_rss_kb = 0
        self.peak_anon_bytes = 0
        self.peak_current_bytes = 0
        self.n_samples = 0
        self.cgroup_dir = self._resolve_cgroup()
        self.cgroup_peak_supported = os.path.exists(os.path.join(self.cgroup_dir, "memory.peak")) if self.cgroup_dir else False

    @staticmethod
    def _resolve_cgroup():
        """Resolve OUR cgroup dir; return None if it is a machine-wide group.

        When the guarded runner is used we live in a transient
        `pes2ts-<name>-<pid>.scope`; reading its ``memory.stat`` gives our own
        peak anon.  Outside the guard we may be in ``/init.scope`` (the whole
        WSL VM), which would report unrelated memory -> fall back to VmRSS.
        """
        try:
            with open("/proc/self/cgroup") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("0::"):
                        rel = line[3:].lstrip("/")
                        if not rel or rel.endswith("init.scope") or ".scope" not in rel:
                            return None
                        cand = os.path.join("/sys/fs/cgroup", rel)
                        if os.path.isdir(cand):
                            return cand
        except OSError:
            pass
        return None

    def _read(self):
        rss = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
                        break
        except OSError:
            pass
        anon = cur = None
        if self.cgroup_dir:
            try:
                with open(os.path.join(self.cgroup_dir, "memory.current")) as f:
                    cur = int(f.read().strip())
                with open(os.path.join(self.cgroup_dir, "memory.stat")) as f:
                    for line in f:
                        if line.startswith("anon "):
                            anon = int(line.split()[1])
                            break
            except (OSError, ValueError):
                anon = cur = None
        return rss, anon, cur

    def run(self):
        while not self._stop_evt.is_set():
            rss, anon, cur = self._read()
            with self._lock:
                self.peak_rss_kb = max(self.peak_rss_kb, rss)
                if anon is not None:
                    self.peak_anon_bytes = max(self.peak_anon_bytes, anon)
                if cur is not None:
                    self.peak_current_bytes = max(self.peak_current_bytes, cur)
                self.n_samples += 1
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)
        out = {
            "n_samples": self.n_samples,
            "peak_rss_kb": self.peak_rss_kb,
            "peak_anon_bytes": self.peak_anon_bytes,
            "peak_cgroup_memory_current_bytes": self.peak_current_bytes,
            "cgroup_dir": self.cgroup_dir,
            "cgroup_memory_peak_supported": self.cgroup_peak_supported,
            "source": "cgroup-v2 memory.stat anon (sampled)" if self.cgroup_dir else "/proc/self/status VmRSS (no cgroup)",
        }
        if self.cgroup_peak_supported:
            try:
                with open(os.path.join(self.cgroup_dir, "memory.peak")) as f:
                    out["cgroup_memory_peak_bytes"] = int(f.read().strip())
            except (OSError, ValueError):
                pass
        return out


# --------------------------------------------------------------------------- #
# Source scanning (directory listing -> id sets, no per-file stat)
# --------------------------------------------------------------------------- #
def filename_to_id(name):
    if not name.endswith(".json"):
        return None
    stem = name[:-5]
    if not stem.startswith("RXN_") or len(stem) != 14:
        return None
    try:
        n = int(stem[4:])
    except ValueError:
        return None
    if stem != f"RXN_{n:010d}":
        return None
    return n


def scan_source(base):
    """Return dict with listed id set + anomalies + counts (no per-file stat)."""
    t0 = time.time()
    ids = set()
    shards = 0
    bad_names = []
    for shard in sorted(os.listdir(base)):
        d = os.path.join(base, shard)
        if not os.path.isdir(d):
            continue
        shards += 1
        with os.scandir(d) as it:
            for e in it:
                if not e.is_file(follow_symlinks=False):
                    continue
                i = filename_to_id(e.name)
                if i is None:
                    if len(bad_names) < 50:
                        bad_names.append(os.path.join(shard, e.name))
                    continue
                ids.add(i)
    return {
        "base": base,
        "subdirs": shards,
        "ids": ids,
        "files_listed": len(ids),
        "bad_names": bad_names,
        "scan_s": round(time.time() - t0, 2),
    }


def ids_sha256(ids):
    h = hashlib.sha256()
    for i in sorted(ids):
        h.update(f"RXN_{i:010d}\n".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #
def path_for(base, rid_num):
    return os.path.join(base, f"{rid_num // 1000:05d}", f"RXN_{rid_num:010d}.json")


def extract_source(obj, cols):
    """Extract data columns for one source. File-sha columns are filled by caller."""
    row = {}
    for col in cols:
        if col.kind == "get":
            row[col.name] = coerce(getpath(obj, col.path), col.typ)
        elif col.func is not None:
            row[col.name] = col.func(obj)
    return row


def build_row(rid, inv_meta, source_bases):
    """Build one Parquet row (plain Python dict)."""
    rid_num = int(rid[4:])
    row = {c.name: None for c in ALL_COLS}
    row.update({
        "reaction_id": rid,
        "in_inventory": True,
        "n_sources_present": 0,
        "n_source_id_mismatch": 0,
        "inv_dataset_version": inv_meta["dataset_version"],
        "inv_n_reactant_components": inv_meta["n_reactant_components"],
        "inv_n_product_components": inv_meta["n_product_components"],
        "inv_total_atoms_reactants": inv_meta["total_atoms_reactants"],
        "inv_total_atoms_products": inv_meta["total_atoms_products"],
        "inv_charge_total_reactants": inv_meta["charge_total_reactants"],
        "inv_charge_total_products": inv_meta["charge_total_products"],
        "inv_multiplicity_max": inv_meta["multiplicity_max"],
        "inv_n_elements": inv_meta["n_elements"],
    })
    for key in SOURCE_KEYS:
        row[HAS_COL[key]] = False
    statuses = []
    ok_count = 0
    for key in SOURCE_KEYS:
        col_list = SOURCE_COLS[key]
        sha_col = SHA_COL[key]
        path = path_for(source_bases[key], rid_num)
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            statuses.append((key, "missing", 0))
            continue
        except OSError:
            statuses.append((key, "read_error", 0))
            row[sha_col] = None
            continue
        nbytes = len(raw)
        row[sha_col] = hashlib.sha256(raw).hexdigest()
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("top-level JSON is not an object")
        except Exception:
            statuses.append((key, "bad_json", nbytes))
            for col in DATA_COLS[key]:
                row[col.name] = None
            continue
        try:
            row.update(extract_source(obj, col_list))
            if obj.get("reaction_id") != rid:
                row["n_source_id_mismatch"] += 1
            row[HAS_COL[key]] = True
            ok_count += 1
            statuses.append((key, "ok", nbytes))
        except Exception:
            for col in DATA_COLS[key]:
                row[col.name] = None
            statuses.append((key, "extract_error", nbytes))
    row["n_sources_present"] = ok_count
    return row, statuses


# --------------------------------------------------------------------------- #
# Arrow conversion / writing
# --------------------------------------------------------------------------- #
def rows_to_table(rows):
    arrays = []
    for col in ALL_COLS:
        vals = [r[col.name] for r in rows]
        at = ARROW_TYPES[col.typ]
        if col.typ == CAT:
            arrays.append(pa.array(vals, type=pa.string()).dictionary_encode())
        else:
            arrays.append(pa.array(vals, type=at))
    return pa.Table.from_arrays(arrays, schema=SCHEMA)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def sample_ids(all_ids, n, seed=SAMPLE_SEED):
    import random

    rng = random.Random(seed)
    if n >= len(all_ids):
        return sorted(all_ids)
    picked = rng.sample(sorted(all_ids), n)
    return sorted(picked)


def read_parquet_rows_for_ids(parquet_path, wanted):
    """Stream the Parquet file and keep only rows whose reaction_id is wanted."""
    found = {}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=8192):
        d = batch.to_pydict()
        ids = d["reaction_id"]
        for i, rid in enumerate(ids):
            if rid in wanted:
                found[rid] = {k: v[i] for k, v in d.items()}
    return found


def verify_sample(parquet_path, all_ids, inv_lookup, source_bases, n, log):
    t0 = time.time()
    ids = sample_ids(all_ids, n)
    got = read_parquet_rows_for_ids(parquet_path, set(ids))
    missing_rows = [i for i in ids if i not in got]
    per_source = {k: {"fields_compared": 0, "roundtrip_comparisons": 0, "roundtrip_mismatches": 0,
                      "independent_path_checks": 0, "independent_path_mismatches": 0} for k in SOURCE_KEYS}
    inv_cmp = {"comparisons": 0, "mismatches": 0}
    flag_cmp = {"comparisons": 0, "mismatches": 0}
    mismatch_samples = []
    n_roundtrip = n_indep = 0

    for rid in ids:
        if rid not in got:
            continue
        pqrow = got[rid]
        inv_meta = inv_lookup[rid]
        expected, statuses = build_row(rid, inv_meta, source_bases)
        status_by_src = {s[0]: s[1] for s in statuses}
        # flags
        for key in SOURCE_KEYS:
            flag_cmp["comparisons"] += 1
            if bool(pqrow[HAS_COL[key]]) != bool(expected[HAS_COL[key]]):
                flag_cmp["mismatches"] += 1
                if len(mismatch_samples) < 20:
                    mismatch_samples.append(f"{rid} {HAS_COL[key]}: parquet={pqrow[HAS_COL[key]]} expected={expected[HAS_COL[key]]}")
        # inventory-derived columns
        for col in COMMON_COLS:
            if col.kind != "inv":
                continue
            inv_cmp["comparisons"] += 1
            if pqrow[col.name] != expected[col.name]:
                inv_cmp["mismatches"] += 1
                if len(mismatch_samples) < 20:
                    mismatch_samples.append(f"{rid} {col.name}: parquet={pqrow[col.name]!r} expected={expected[col.name]!r}")
        # per-source columns: roundtrip (all cols) + independent literal path checks
        for key in SOURCE_KEYS:
            obj = None
            path = path_for(source_bases[key], int(rid[4:]))
            if status_by_src.get(key) == "ok":
                try:
                    with open(path, "rb") as f:
                        obj = json.loads(f.read())
                except Exception:
                    obj = None
            for col in SOURCE_COLS[key]:
                if col.name not in pqrow:
                    continue
                n_roundtrip += 1
                per_source[key]["fields_compared"] += 1
                per_source[key]["roundtrip_comparisons"] += 1
                if pqrow[col.name] != expected[col.name]:
                    per_source[key]["roundtrip_mismatches"] += 1
                    if len(mismatch_samples) < 20:
                        mismatch_samples.append(
                            f"{rid} {col.name}: parquet={pqrow[col.name]!r} expected={expected[col.name]!r}")
                # Independent check for simple path columns
                if col.kind == "get":
                    per_source[key]["independent_path_checks"] += 1
                    n_indep += 1
                    if obj is None:
                        indep_val = None
                    else:
                        indep_val = coerce(getpath(obj, col.path), col.typ)
                    if pqrow[col.name] != indep_val:
                        per_source[key]["independent_path_mismatches"] += 1
                        if len(mismatch_samples) < 20:
                            mismatch_samples.append(
                                f"{rid} INDEP {col.name}: parquet={pqrow[col.name]!r} json={indep_val!r}")

    total_mismatch = sum(v["roundtrip_mismatches"] for v in per_source.values()) + inv_cmp["mismatches"] + flag_cmp["mismatches"]
    total_roundtrip = n_roundtrip + inv_cmp["comparisons"] + flag_cmp["comparisons"]
    indep_mismatch = sum(v["independent_path_mismatches"] for v in per_source.values())
    result = {
        "n_sampled": len(ids),
        "sample_seed": SAMPLE_SEED,
        "sample_ids": ids,
        "sample_ids_sha256": ids_sha256([int(i[4:]) for i in ids]),
        "parquet_rows_found": len(got),
        "missing_rows": missing_rows,
        "fields_compared_per_source": {k: len(SOURCE_COLS[k]) for k in SOURCE_KEYS},
        "flags_comparisons": flag_cmp,
        "inventory_comparisons": inv_cmp,
        "per_source": per_source,
        "roundtrip_comparisons": total_roundtrip,
        "roundtrip_mismatches": total_mismatch,
        "independent_path_checks": n_indep,
        "independent_path_mismatches": indep_mismatch,
        "consistency_rate": 1.0 if total_roundtrip + n_indep == 0 else
        round(1.0 - (total_mismatch + indep_mismatch) / (total_roundtrip + n_indep), 12),
        "mismatch_samples": mismatch_samples,
        "rule": (
            "For each sampled reaction_id the Parquet row is compared field-by-field against a fresh "
            "read of the six source JSON files: booleans/ints/strings require exact equality; floats "
            "require exact binary equality (JSON double -> Parquet double); a missing JSON path and a "
            "Parquet null are considered equal; joined strings are recomputed from the JSON. "
            "Independently, every column whose extraction rule is a literal JSON path is re-derived by "
            "direct path traversal+type coercion (not by the production extractor's code path)."
        ),
        "wall_s": round(time.time() - t0, 2),
    }
    log(f"[verify] sampled={len(ids)} rows_found={len(got)} roundtrip_cmp={total_roundtrip} "
        f"roundtrip_mismatch={total_mismatch} indep_cmp={n_indep} indep_mismatch={indep_mismatch}")
    return result


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #
def sha256_file(path, buf_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_previous_manifest(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def log_line(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interim", default=DEFAULT_INTERIM, help="interim root containing the 6 source dirs")
    ap.add_argument("--inventory", default=None, help="inventory.parquet path (default: <interim>/inventory.parquet)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="output directory")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="smoke mode: only first N inventory ids, writes *.smoke.parquet, no manifest")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verify-sample-n", type=int, default=DEFAULT_SAMPLE_N)
    ap.add_argument("--bad-json-log", default=None)
    args = ap.parse_args(argv)

    started = time.time()
    started_utc = utcnow()
    smoke = args.limit > 0
    inventory_path = args.inventory or os.path.join(args.interim, "inventory.parquet")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    parquet_name = "g1_reaction_summary.smoke.parquet" if smoke else "g1_reaction_summary.parquet"
    manifest_name = "g1_summary_manifest.json"
    parquet_path = os.path.join(out_dir, parquet_name)
    manifest_path = os.path.join(out_dir, manifest_name)
    bad_log_path = args.bad_json_log or f"/tmp/g1_summary_bad_json_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"

    sampler = MemorySampler(interval=1.0)
    sampler.start()

    timings = {}
    log_line(f"[g1-summary] schema={SCHEMA_VERSION} interim={args.interim} out={out_dir}")
    log_line(f"[g1-summary] python={platform.python_version()} pyarrow={pa.__version__} workers={args.workers} chunk={args.chunk_size}")

    # ---- inventory -------------------------------------------------------- #
    t0 = time.time()
    inv_table = pq.read_table(inventory_path, columns=[f for f, _ in INVENTORY_FIELDS] + ["elements"])
    inv_cols = {name: inv_table.column(name).to_pylist() for name, _ in INVENTORY_FIELDS}
    elements_col = inv_table.column("elements").to_pylist()
    n_inv = inv_table.num_rows
    inv_rows = []
    inv_lookup = {}
    for i in range(n_inv):
        rid = inv_cols["reaction_id"][i]
        meta = {
            "dataset_version": inv_cols["dataset_version"][i],
            "n_reactant_components": inv_cols["n_reactant_components"][i],
            "n_product_components": inv_cols["n_product_components"][i],
            "total_atoms_reactants": inv_cols["total_atoms_reactants"][i],
            "total_atoms_products": inv_cols["total_atoms_products"][i],
            "charge_total_reactants": inv_cols["charge_total_reactants"][i],
            "charge_total_products": inv_cols["charge_total_products"][i],
            "multiplicity_max": inv_cols["multiplicity_max"][i],
            "n_elements": len(elements_col[i]) if isinstance(elements_col[i], list) else None,
        }
        inv_lookup[rid] = meta
        inv_rows.append(rid)
    inv_rows.sort()
    if args.limit:
        inv_rows = inv_rows[: args.limit]
    timings["inventory_s"] = round(time.time() - t0, 2)
    log_line(f"[g1-summary] inventory rows={n_inv} unique={len(inv_lookup)} selected={len(inv_rows)} ({timings['inventory_s']}s)")

    # ---- scan sources (listing only) -------------------------------------- #
    source_bases = {}
    scans = {}
    for key in SOURCE_KEYS:
        base = os.path.join(args.interim, REL_DIR[key])
        source_bases[key] = base
        scans[key] = scan_source(base)
        log_line(f"[g1-summary] scan {key}: files={scans[key]['files_listed']} subdirs={scans[key]['subdirs']} "
                 f"bad_names={len(scans[key]['bad_names'])} ({scans[key]['scan_s']}s)")
    timings["scan_s"] = round(sum(s["scan_s"] for s in scans.values()), 2)

    inv_id_set = set(int(r[4:]) for r in inv_rows)
    source_cov = {}
    for key in SOURCE_KEYS:
        sc = scans[key]
        listed = sc["ids"]
        missing = sorted(inv_id_set - listed)
        extra = sorted(listed - inv_id_set)
        source_cov[key] = {
            "path": sc["base"],
            "subdirs": sc["subdirs"],
            "files_listed": sc["files_listed"],
            "listed_ids_in_inventory": len(listed & inv_id_set),
            "inventory_missing_count": len(missing),
            "inventory_missing_first10": [f"RXN_{i:010d}" for i in missing[:10]],
            "inventory_missing_ids_sha256": ids_sha256(missing),
            "extra_files_not_in_inventory": len(extra),
            "extra_files_first10": [f"RXN_{i:010d}" for i in extra[:10]],
            "bad_filenames": sc["bad_names"][:10],
            "listed_ids_sha256": ids_sha256(listed),
        }
    # free listing sets (only coverage comparisons need them)
    for key in SOURCE_KEYS:
        scans[key]["ids"] = None

    # ---- streaming build -------------------------------------------------- #
    t0 = time.time()
    source_stats = {key: Counter() for key in SOURCE_KEYS}
    source_bytes = {key: 0 for key in SOURCE_KEYS}
    schema_versions = {key: Counter() for key in SOURCE_KEYS}
    bad_json_entries = []
    bad_json_total = 0
    read_error_entries = []
    extract_error_entries = []

    rows_written = 0
    id_mismatch_total = 0
    chunk_size = args.chunk_size
    n_chunks = (len(inv_rows) + chunk_size - 1) // chunk_size
    log_line(f"[g1-summary] building {len(inv_rows)} rows in {n_chunks} chunks of {chunk_size} with {args.workers} workers")

    def worker(item):
        rid = item
        return build_row(rid, inv_lookup[rid], source_bases)

    tmp_parquet = parquet_path + ".tmp"
    if os.path.exists(tmp_parquet):
        os.remove(tmp_parquet)
    t_write = 0.0
    with pq.ParquetWriter(tmp_parquet, SCHEMA, compression="zstd", use_dictionary=True,
                          write_statistics=True) as writer:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for ci in range(n_chunks):
                chunk_ids = inv_rows[ci * chunk_size:(ci + 1) * chunk_size]
                results = list(ex.map(worker, chunk_ids))
                rows = [r for r, _ in results]
                for row, statuses in results:
                    for key, status, nbytes in statuses:
                        source_stats[key][status] += 1
                        source_bytes[key] += nbytes
                        if status == "bad_json":
                            bad_json_total += 1
                            if len(bad_json_entries) < BAD_JSON_LOG_MAX:
                                bad_json_entries.append(f"{key} {row['reaction_id']}")
                        elif status == "read_error" and len(read_error_entries) < 20:
                            read_error_entries.append(f"{key} {row['reaction_id']}")
                        elif status == "extract_error" and len(extract_error_entries) < 20:
                            extract_error_entries.append(f"{key} {row['reaction_id']}")
                    id_mismatch_total += row["n_source_id_mismatch"]
                for r in rows:
                    for key in SOURCE_KEYS:
                        sv = r[SCHEMA_COL[key]]
                        if sv is not None:
                            schema_versions[key][sv] += 1
                tw = time.time()
                writer.write_table(rows_to_table(rows))
                t_write += time.time() - tw
                rows_written += len(rows)
                elapsed = time.time() - t0
                log_line(f"[g1-summary] chunk {ci + 1}/{n_chunks} rows={rows_written} "
                         f"elapsed={elapsed:.1f}s ({rows_written / elapsed:.0f} rows/s) "
                         f"bad_json={bad_json_total} peak_anon={sampler.peak_anon_bytes / 2**20:.0f}MiB")
    timings["build_s"] = round(time.time() - t0, 2)
    timings["parquet_write_s"] = round(t_write, 2)

    # bad JSON log (max 20 paths; explicit, never silent)
    with open(bad_log_path, "w", encoding="utf-8") as f:
        f.write(f"# bad JSON / read errors / extract errors for {SCHEMA_VERSION}\n")
        f.write(f"# generated_at={utcnow()}\n")
        f.write(f"bad_json_total={bad_json_total}\n")
        for line in bad_json_entries:
            f.write(f"bad_json {line}\n")
        for line in read_error_entries:
            f.write(f"read_error {line}\n")
        for line in extract_error_entries:
            f.write(f"extract_error {line}\n")

    # ---- checks + parquet sha --------------------------------------------- #
    size_bytes = os.path.getsize(tmp_parquet)
    parquet_sha = sha256_file(tmp_parquet)
    pf_meta = pq.ParquetFile(tmp_parquet)
    actual_rows = pf_meta.metadata.num_rows
    row_groups = pf_meta.metadata.num_row_groups
    readback_schema = pf_meta.schema_arrow

    has_change_true = source_stats["reaction_change"]["ok"]
    checks = {
        "row_count_equals_inventory": actual_rows == len(inv_rows) if not smoke else None,
        "expected_rows": len(inv_rows),
        "actual_rows": actual_rows,
        "primary_key_unique": None,  # by construction (inventory ids are unique); checked below
        "all_rows_have_reaction_change": has_change_true == actual_rows,
        "reaction_change_ok": source_stats["reaction_change"]["ok"],
        "reaction_change_missing": source_stats["reaction_change"]["missing"],
        "reaction_change_bad_json": source_stats["reaction_change"]["bad_json"],
        "reaction_change_extract_error": source_stats["reaction_change"]["extract_error"],
        "n_source_id_mismatch_total": id_mismatch_total,
    }

    # read back a few columns to confirm integrity + uniqueness
    rb = pq.read_table(tmp_parquet, columns=["reaction_id", "has_reaction_change", "n_sources_present"])
    rb_ids = rb.column("reaction_id").to_pylist()
    checks["primary_key_unique"] = len(rb_ids) == len(set(rb_ids))
    checks["n_sources_present_histogram"] = {str(k): v for k, v in Counter(rb.column("n_sources_present").to_pylist()).items()}
    checks["has_reaction_change_true_count"] = sum(1 for v in rb.column("has_reaction_change").to_pylist() if v)
    sample5 = rb.slice(0, 5).to_pylist()
    del rb

    # ---- verification ----------------------------------------------------- #
    verify = None
    if not args.skip_verify and not smoke:
        t0 = time.time()
        verify = verify_sample(tmp_parquet, inv_rows, inv_lookup, source_bases, args.verify_sample_n, log_line)
        timings["verify_s"] = round(time.time() - t0, 2)
    elif smoke:
        t0 = time.time()
        small_verify = verify_sample(tmp_parquet, inv_rows, inv_lookup, source_bases,
                                     min(20, args.verify_sample_n), log_line)
        verify = small_verify
        timings["verify_s"] = round(time.time() - t0, 2)

    # ---- atomic replace + idempotency ------------------------------------ #
    previous_manifest = load_previous_manifest(manifest_path) if not smoke else None
    previous_sha = None
    previous_runs = []
    if previous_manifest:
        previous_sha = previous_manifest.get("parquet", {}).get("sha256")
        previous_runs = previous_manifest.get("runs", [])
    os.replace(tmp_parquet, parquet_path)
    sha_unchanged = (previous_sha == parquet_sha)

    sampler_stats = sampler.stop()
    timings["total_s"] = round(time.time() - started, 2)

    # ---- build manifest --------------------------------------------------- #
    if not smoke:
        column_manifest = []
        for col in ALL_COLS:
            column_manifest.append({
                "name": col.name,
                "type": col.typ,
                "arrow_type": str(ARROW_TYPES[col.typ]),
                "source": ("inventory.parquet" if col.kind == "inv"
                           else "derived (flags/counts)" if col.name in [c.name for c in COMMON_COLS]
                           else SOURCE_OF_COL[col.name]),
                "json_path": col.path,
                "kind": col.kind,
                "rule": col.rule,
            })

        run_record = {
            "run_index": len(previous_runs) + 1,
            "started_at_utc": started_utc,
            "finished_at_utc": utcnow(),
            "wall_s": timings["total_s"],
            "rows": actual_rows,
            "parquet_sha256": parquet_sha,
            "parquet_size_bytes": size_bytes,
            "parquet_sha256_unchanged_vs_previous_run": sha_unchanged,
            "peak": sampler_stats,
            "timings": timings,
        }
        runs = previous_runs + [run_record]

        env = {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "pyarrow_version": pa.__version__,
            "platform": platform.platform(),
            "kernel": platform.release(),
            "cpu_count": os.cpu_count(),
            "wsl": bool(os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop") or "microsoft" in platform.release().lower()),
            "argv": sys.argv,
            "determinism_note": (
                "Row order = inventory reaction_id ascending; chunk boundaries fixed (--chunk-size); "
                "no wall-clock/timestamp values are written into the Parquet payload; fixed column order; "
                "zstd compression. Same pyarrow/arrow version + same inputs => identical parquet sha256."
            ),
        }

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at_utc": utcnow(),
            "description": (
                "G1 per-reaction summary: one row per inventory.parquet reaction_id, built by streaming the "
                "six scattered G1 JSON sources under data/interim/.  Source files were read-only; nothing was "
                "deleted or moved by this build."
            ),
            "build_script": {
                "path": "pes2ts_data/builders/build_g1_summary.py",
                "sha256": sha256_file(os.path.abspath(__file__)),
            },
            "inputs": {
                "interim_root": args.interim,
                "inventory_parquet": {
                    "path": inventory_path,
                    "sha256": sha256_file(inventory_path),
                    "rows": n_inv,
                    "unique_reaction_ids": len(inv_lookup),
                },
                "source_files": {
                    key: {
                        "path": source_cov[key]["path"],
                        "relative_path": REL_DIR[key],
                        "subdirs": source_cov[key]["subdirs"],
                        "files_listed": source_cov[key]["files_listed"],
                        "bytes_read": source_bytes[key],
                        "bytes_read_human": f"{source_bytes[key] / 2**30:.3f} GiB",
                        "inventory_missing_count": source_cov[key]["inventory_missing_count"],
                        "inventory_missing_first10": source_cov[key]["inventory_missing_first10"],
                        "inventory_missing_ids_sha256": source_cov[key]["inventory_missing_ids_sha256"],
                        "extra_files_not_in_inventory": source_cov[key]["extra_files_not_in_inventory"],
                        "listed_ids_sha256": source_cov[key]["listed_ids_sha256"],
                        "not_read_note": (
                            "g1_v2/export_contracts_v1/ is a sanitized copy of g1_v2/export/ "
                            "(see its own manifest: source_tree_sha256=f237062ebdb0d57535d309d93697468bb1ea97728940c5ad5d16b70e571f31cd, "
                            "n_exports=183460, n_bytes=1043107985) and was NOT read, to avoid double-counting."
                            if key == "v2_export" else None),
                    }
                    for key in SOURCE_KEYS
                },
            },
            "parquet": {
                "path": parquet_path,
                "size_bytes": size_bytes,
                "size_mib": round(size_bytes / 2**20, 2),
                "size_below_500mb": size_bytes < 500 * 2**20,
                "sha256": parquet_sha,
                "sha256_first12": parquet_sha[:12],
                "rows": actual_rows,
                "row_groups": row_groups,
                "compression": "zstd",
                "n_columns": len(ALL_COLS),
                "schema": [{"name": c.name, "type": c.typ, "arrow_type": str(ARROW_TYPES[c.typ]),
                             "nullable": SCHEMA.field(c.name).nullable} for c in ALL_COLS],
                "readback_check": {
                    "pyarrow_schema_fields": len(readback_schema),
                    "first_5_rows": sample5,
                },
            },
            "row_count": {
                "inventory_rows": n_inv,
                "parquet_rows": actual_rows,
                "match": actual_rows == n_inv,
                "grain": "1 row = 1 reaction_id",
            },
            "checks": checks,
            "coverage": {
                "definition": (
                    "Per source: files_listed = files in the shard dirs; inventory_missing = inventory ids with no file; "
                    "bad_json/read_error/extract_error counted during the streaming read (never silently skipped); "
                    "has_true = parsed+extracted rows (has_<source>=True)."
                ),
                "per_source": {
                    key: {
                        "files_listed": source_cov[key]["files_listed"],
                        "bytes_read": source_bytes[key],
                        "missing": source_stats[key]["missing"],
                        "bad_json": source_stats[key]["bad_json"],
                        "read_error": source_stats[key]["read_error"],
                        "extract_error": source_stats[key]["extract_error"],
                        "ok": source_stats[key]["ok"],
                        "has_true": source_stats[key]["ok"],
                        "has_false": actual_rows - source_stats[key]["ok"],
                        "pct_has_true": round(100.0 * source_stats[key]["ok"] / actual_rows, 4) if actual_rows else None,
                        "schema_versions_observed": dict(schema_versions[key]),
                        "columns": len(SOURCE_COLS[key]),
                        "file_sha256_column": SHA_COL[key],
                        "has_flag_column": HAS_COL[key],
                    }
                    for key in SOURCE_KEYS
                },
                "rows_with_all_6_sources": checks["n_sources_present_histogram"].get("6", 0),
                "n_sources_present_histogram": checks["n_sources_present_histogram"],
                "bad_json_total": bad_json_total,
                "bad_json_log_path": bad_log_path,
                "bad_json_log_max_entries": BAD_JSON_LOG_MAX,
                "bad_json_samples": bad_json_entries,
                "read_error_samples": read_error_entries,
                "extract_error_samples": extract_error_entries,
                "n_source_id_mismatch_total": id_mismatch_total,
            },
            "columns": column_manifest,
            "omitted_fields": {
                "policy": (
                    "Variable-length arrays / large objects are not stored in the Parquet. Each source row instead "
                    "carries <prefix>_file_sha256 = sha256 of the raw source JSON file bytes, so any omitted content "
                    "can be integrity-checked or de-duplicated; the original JSON remains the source of truth at "
                    "<source>/<reaction_id//1000:05d>/<reaction_id>.json."
                ),
                "per_source": OMITTED_FIELDS,
            },
            "sampling_verification": verify,
            "runs": runs,
            "idempotency": {
                "previous_parquet_sha256": previous_sha,
                "current_parquet_sha256": parquet_sha,
                "sha_unchanged_vs_previous_run": sha_unchanged,
                "runs_recorded": len(runs),
            },
            "environment": env,
            "open_questions": [
                f"v2_export covers {source_stats['v2_export']['ok']} of {n_inv} inventory reactions "
                f"({source_cov['v2_export']['inventory_missing_count']} missing). The missing set equals the "
                "non-resolved/non-clean reactome set (v2_export status_summary.p1_status only resolved_*), i.e. "
                "export is a resolved-only projection, not a truncated copy; g1_v2/export_contracts_v1/ has the "
                "same 183,460 file count. Confirm downstream consumers treat has_v2_export=False as expected.",
                "g1_v2/export_contracts_v1/ is a sanitized duplicate (removed keys endpoint_match/orientation for "
                "183,460 records per its manifest). This build intentionally reads only g1_v2/export/.",
                "p1_mapping graph / irc_validation / mapping sub-objects can be null or partial for excluded "
                "records (e.g. 155/4000 with graph=None); corresponding p1_* columns are null there.",
                "reaction_change still contains records with validation.status='rejected' (sampled ~5/3000); "
                "has_reaction_change is True for them (file present+parsed) - filter on change_validation_status "
                "when a clean-only subset is required.",
                "Per-source file counts/bytes were measured during this build (all listed files read exactly once; "
                "no per-file pre-stat) so bytes_read equals the on-disk byte total for each source directory.",
                "Determinism of the parquet sha256 assumes the same pyarrow version and platform; verify after any "
                "arrow upgrade (manifest records the version used).",
            ],
        }
        atomic_write_json(manifest_path, manifest)
        log_line(f"[g1-summary] wrote {parquet_path} rows={actual_rows} size={size_bytes/2**20:.1f}MiB sha256={parquet_sha[:12]}")
        log_line(f"[g1-summary] wrote {manifest_path} runs={len(runs)} idempotent={sha_unchanged}")
        if verify:
            log_line(f"[g1-summary] verification consistency={verify['consistency_rate']} "
                     f"roundtrip_mismatches={verify['roundtrip_mismatches']} indep_mismatches={verify['independent_path_mismatches']}")
    else:
        log_line(f"[g1-summary] SMOKE build wrote {parquet_path} rows={actual_rows} sha={parquet_sha[:12]}")
        if verify:
            log_line(f"[g1-summary] SMOKE verification consistency={verify['consistency_rate']}")

    log_line(f"[g1-summary] done in {timings['total_s']}s peak_anon={sampler_stats['peak_anon_bytes']/2**20:.1f}MiB "
             f"peak_rss={sampler_stats['peak_rss_kb']/1024:.0f}MiB")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
