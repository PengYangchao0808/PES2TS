#!/usr/bin/env python3
"""WP2-A fit50 FULL dual-TS teacher build + observation coverage lists.

Takes the fit wave of the frozen dual-TS split lineage (default: ALL 50
group ids) and for each reaction builds the FULL teacher with the frozen
WP2 decisions:

* C5 metric: ``ell`` under the diagonal-mass metric
  ``||delta||_M = sqrt(sum_i m_i ||delta_i||^2)`` with the SAME v1
  ``masses_for`` table used by ``register_path`` (version string
  ``mass_weighted_arc_v1_diag_masses``; the WP1 placeholder artifacts are
  never touched).
* Step-5 insertion: per-branch linear midpoint insertion when any per-atom
  ``|a|`` exceeds ``a_max=0.1 Å`` (C1: step_index renumbered, labels
  re-differenced from EXPORTED coords, identities re-asserted at 1e-8,
  ``inserted:true`` flags + per-insert gaps entries; no clipping, no
  dropping of successors).
* Weak-match policy: authority ``r_bonds``/``p_bonds`` cross-checked
  against covalent-radius perception on the ACTUAL endpoint frames.
  Disagreement on a CHANGED bond → ``bond_authority=disputed`` +
  ``mapping_problem`` — the reaction STAYS in the denominator. Spectator
  disagreement → accept-with-note. Per-reaction outcome recorded.
* Honest failover: every miss is registered item-by-item and the next fit
  wave id is consumed; nothing is ever silently dropped.

Coverage: every teacher state (including ONE shared computation-identity
node per reaction TS) is emitted into ``coverage_nodes.json`` keyed by
FULL computation identity (geometry content digest + atom order/map hash +
GFN2-xTB + charge/multiplicity + E/g single point), and matched honestly
against the v1 ACP receipt cache. ``coverage_summary.json`` carries the
totals, gap list, and the cost/wall estimate with its basis.

NO ACP/xTB execution and no network in this lane: the v1 observations tree
is only read. IRC frames are read ONLY through the audited v1 truth bridge
(``--allow-truth`` gated). Read-only on every v1/ground-truth source;
writes only under ``--out``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    DualEndpointTask,
    DualTSTeacher,
    TASK_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    validate_record,
)
from pes2ts_core.pesn.offline.data.dual_ts_fit50 import (  # noqa: E402
    COVERAGE_NODES_SCHEMA,
    COVERAGE_SUMMARY_SCHEMA,
    GAPS_REGISTRY_SCHEMA_V1,
    METHOD_GFN2_XTB,
    RUN_SUMMARY_SCHEMA,
    WP2_FIT50_VERSION,
    build_cache_index,
    estimate_cost,
    node_records_for_reaction,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (  # noqa: E402
    A_MAX_ANGSTROM,
    ARC_LENGTH_METRIC_V1_DIAG_MASSES,
    BOND_PERCEPTION_AUTHORITY,
    DUAL_TS_PREPROCESS_VERSION,
    DRIFT_RULE,
    DualTSPreprocessError,
    assert_task_teacher_identity,
    build_dual_ts_reaction,
    endpoint_graph_from_bond_list,
    excluded_outcome,
    load_irc_frames_for_reaction,
    masses_for,
    order_frames_and_energy,
)
from pes2ts_core.pesn.offline.data.endpoint_adapter import (  # noqa: E402
    EndpointAdapterError,
    adapt_reaction,
    load_authority_document,
)
from pes2ts_core.pesn.offline.data.direct_preprocess import (  # noqa: E402
    DirectPreprocessError,
)
from pes2ts_core.pesn.offline.splits.dual_ts_split import (  # noqa: E402
    DualTSSplitLineage,
    validate_lineage_payload,
)

DEFAULT_LINEAGE = ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/splits/dual_ts_split_lineage_v1.json")
DEFAULT_OUT = ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_fit50"
DEFAULT_MANIFESTS = ROOT / "data/manifests"       # v1 config paths.manifests
DEFAULT_INTERIM = ROOT / "data/interim"           # v1 config paths.interim
DEFAULT_SPLIT_DIR = ROOT / (
    "outputs/pesn/direct_egnn_v1_20261015/splits")  # v1 config paths.split_dir
DEFAULT_OBSERVATIONS = ROOT / (
    "outputs/pesn/direct_egnn_v1_20261015/observations")  # v1 obs cache

WORKERS = 2                # v1 config limits.max_parallel_reactions
THREADS_PER_WORKER = 2     # v1 config acp.nproc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def irc_index_rows_for(manifests_dir: Path, group_ids: list[str],
                       allow_truth: bool) -> dict[str, dict]:
    """Shape-only IRC index rows via the audited G0 accessor (no trajectory)."""
    from pes2ts_core.g0.truth import truth_reader

    wanted = set(group_ids)
    rows = truth_reader.load_irc_index(allow_truth=allow_truth,
                                       manifests_dir=manifests_dir)
    return {str(r["reaction_id"]): r for r in rows
            if str(r["reaction_id"]) in wanted}


def resolve_group_queue(
    lineage: DualTSSplitLineage,
    groups_arg: str | None,
) -> tuple[list[str], str]:
    """Resolve the build queue from the lineage (pure; additive --groups).

    Default (``groups_arg is None``): the FULL fit wave — semantics
    unchanged.  ``--groups RXN_..,..``: monitor-wave mode — the ONLY
    accepted set is EXACTLY the lineage monitor wave (complete, no
    duplicates), every member must carry partition ``monitor``, and
    fit ∩ monitor must be empty.  Returns ``(queue, mode)``.
    """
    wave_fit = lineage.wave_groups("fit")
    if groups_arg is None:
        return list(wave_fit), "fit_wave"
    requested = [g.strip() for g in str(groups_arg).split(",") if g.strip()]
    if not requested:
        print("[refuse] --groups given but empty after parsing",
              file=sys.stderr)
        raise SystemExit(2)
    if len(set(requested)) != len(requested):
        print("[refuse] --groups contains duplicates", file=sys.stderr)
        raise SystemExit(2)
    wave_monitor = lineage.wave_groups("monitor")
    unknown = sorted(set(requested) - set(wave_monitor))
    if unknown:
        print(f"[refuse] --groups members outside the lineage monitor "
              f"wave: {unknown}", file=sys.stderr)
        raise SystemExit(2)
    missing = sorted(set(wave_monitor) - set(requested))
    if missing:
        print(f"[refuse] --groups must be the COMPLETE monitor wave "
              f"(n={len(wave_monitor)}); missing: {missing}", file=sys.stderr)
        raise SystemExit(2)
    overlap = sorted(set(requested) & set(wave_fit))
    if overlap:
        print(f"[refuse] fit ∩ monitor must be empty; overlap: {overlap}",
              file=sys.stderr)
        raise SystemExit(2)
    for gid in requested:
        if lineage.partition_of(gid) != "monitor":
            print(f"[refuse] {gid}: lineage partition is "
                  f"{lineage.partition_of(gid)!r}, expected 'monitor'",
                  file=sys.stderr)
            raise SystemExit(2)
    return sorted(requested), "monitor_wave"


def prepare_one_group_fit50(
    group_id: str,
    *,
    lineage: DualTSSplitLineage,
    index_rows: dict[str, dict],
    manifests_dir: Path,
    interim_dir: Path,
    split_dir: Path,
    shard_size: int,
    a_max: float,
    allow_truth: bool,
):
    """Full WP2 build of one group; returns (outcome, adapter_or_None).

    Same audited geometry/authority flow as WP1, plus the frozen WP2
    decisions (mass metric, insertion, bond cross-check).  Every failure is
    an honest registered exclusion — the reaction stays in the denominator.
    """
    partition = lineage.partition_of(group_id)
    if partition is None:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition="none", reason="group absent from the lineage artifact",
            error_code="GROUP_UNKNOWN"), None

    # --- geometry: audited IRC frames (before anything else) ---------------
    try:
        frames_doc = load_irc_frames_for_reaction(
            group_id, allow_truth=allow_truth,
            manifests_dir=manifests_dir, split_dir=split_dir)
    except KeyError as exc:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition, reason=str(exc),
            error_code="IRC_ARCHIVE_MISS"), None
    except PermissionError as exc:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition, reason=str(exc),
            error_code="TRUTH_BRIDGE_REFUSED"), None
    frames = np.asarray(frames_doc["coordinates"], dtype=float)
    ehg = frames_doc.get("EHG")
    energies = (None if ehg is None
                else np.asarray(ehg, dtype=float)[:, 0])

    # --- authority documents (G1 bond source) ------------------------------
    adapter = None
    bond_perception = BOND_PERCEPTION_AUTHORITY
    try:
        docs = load_authority_document(interim_dir, group_id,
                                       shard_size=shard_size)
        adapter = adapt_reaction(docs)
    except EndpointAdapterError as exc:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition,
            reason=f"authority document unavailable ({exc}); WP2 keeps the "
                   "honest exclusion (no silent covalent substitution for "
                   "the authority bond source)",
            error_code="AUTHORITY_DOC_MISSING",
            items={"mapping_problems": [
                f"authority document unavailable: {exc}"]}), None

    # --- IRC index verification: exists, atom count -------------------------
    index_row = index_rows.get(group_id)
    index_issues: list[str] = []
    if index_row is None:
        index_issues.append("reaction absent from the IRC index")
    else:
        if int(index_row["n_atoms"]) != len(adapter["atomic_numbers_by_map"]):
            index_issues.append(
                f"index n_atoms {index_row['n_atoms']} != task graph atoms "
                f"{len(adapter['atomic_numbers_by_map'])}")

    layout = str(adapter["layout"])
    orientation = str(adapter["orientation"])
    ts_frame_index = adapter["ts_frame_index"]
    splice_index = adapter["splice_index"]

    if layout != "ts_first_dual_branch" or splice_index is None \
            or ts_frame_index is None:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition,
            reason=f"layout {layout!r} is not a usable two-sided IRC "
                   "(missing branch registered, never fabricated)",
            error_code="MISSING_BRANCH",
            items={"missing_branches": [{
                "side": "unknown", "reason": f"layout {layout!r} not dual-branch",
            }]}), adapter

    quality_notes = [str(f) for f in (adapter.get("quality_flags") or [])]
    if index_issues:
        quality_notes.extend(index_issues)
    if str(adapter.get("endpoint_match")) == "weak":
        quality_notes.append("g1 endpoint_match=weak (inherited from P1 doc)")

    # --- endpoint graphs (v1/G1 authority bond source) ----------------------
    try:
        graph_r = endpoint_graph_from_bond_list(
            np.asarray(adapter["atomic_numbers_by_map"], dtype=int),
            adapter["r_bonds"])
        graph_p = endpoint_graph_from_bond_list(
            np.asarray(adapter["atomic_numbers_by_map"], dtype=int),
            adapter["p_bonds"])
    except DualTSPreprocessError as exc:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition, reason=str(exc),
            error_code="ENDPOINT_BONDS_MISSING",
            items={"missing_branches": [{
                "side": "unknown", "reason": f"endpoint bond list unusable: {exc}",
            }]}), adapter

    # --- order + split + build (frozen WP2 decisions ON) --------------------
    try:
        ordered, ordered_e, ts_position = order_frames_and_energy(
            frames, energies,
            layout=layout, orientation=orientation,
            ts_frame_index=int(ts_frame_index), splice_index=int(splice_index))
        if index_row is not None and int(index_row["n_frames"]) != len(ordered):
            quality_notes.append(
                f"index n_frames {index_row['n_frames']} != archive frames "
                f"{len(ordered)}")
        z = np.asarray(adapter["atomic_numbers_by_map"], dtype=int)
        ell_masses = masses_for(z)  # same v1 table register_path uses
        outcome = build_dual_ts_reaction(
            reaction_id=group_id,
            reaction_group_id=group_id,
            partition=partition,
            atomic_numbers=z,
            ordered_path=ordered,
            authority_ts_position=ts_position,
            energies=ordered_e,
            graph_r=graph_r,
            graph_p=graph_p,
            bond_perception=bond_perception,
            method_id=METHOD_GFN2_XTB,
            charge=int(adapter.get("charge_total_reactants") or 0),
            multiplicity=int(adapter.get("multiplicity_max") or 1),
            a_max_angstrom=a_max,
            arc_length_metric_version=ARC_LENGTH_METRIC_V1_DIAG_MASSES,
            ell_masses=ell_masses,
            insert_on_exceedance=True,
            crosscheck_endpoint_bonds=True,
            versions={
                "budget_version": "wp2_teacher_full_path_v0",
                "environment_version": "wp2_geometry_plus_coverage_v0",
                "reward_version": "unassigned_wp2_v0",
            },
            quality_notes=quality_notes,
        )
        return outcome, adapter
    except DualTSPreprocessError as exc:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition, reason=exc.detail,
            error_code=exc.code,
            items={"geometry_quality_issues": list(quality_notes)}), adapter
    except DirectPreprocessError as exc:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition, reason=str(exc),
            error_code="MASS_TABLE_MISS",
            items={"geometry_quality_issues": list(quality_notes)}), adapter
    except Exception as exc:  # honest failover: never crash the wave
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition,
            reason=f"unexpected builder failure: {exc!r}",
            error_code="BUILD_UNEXPECTED",
            items={"geometry_quality_issues": list(quality_notes)}), adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--max-reactions", type=int, default=None,
                        help="cap on reactions attempted (default: all "
                             "fit-wave groups)")
    parser.add_argument("--groups", type=str, default=None,
                        help="additive monitor-wave mode: comma-separated "
                             "group ids; must be EXACTLY the lineage monitor "
                             "wave (complete, fit∩monitor=0 asserted); "
                             "default (absent) = full fit wave, unchanged")
    parser.add_argument("--truth-source", type=Path, default=DEFAULT_MANIFESTS,
                        help="manifests directory holding truth_manifest.json "
                             "(v1 config default data/manifests)")
    parser.add_argument("--interim", type=Path, default=DEFAULT_INTERIM)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--observations", type=Path,
                        default=DEFAULT_OBSERVATIONS,
                        help="v1 ACP observations cache (read-only audit)")
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--a-max", type=float, default=A_MAX_ANGSTROM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-truth", action="store_true", required=True,
                        help="v1 pattern: audited teacher build requires the "
                             "explicit flag")
    args = parser.parse_args()

    t0 = time.perf_counter()
    lineage_path: Path = args.lineage
    if not lineage_path.is_file():
        print(f"[lineage] missing artifact: {lineage_path}", file=sys.stderr)
        return 2
    lineage_payload = json.loads(lineage_path.read_text())
    validate_lineage_payload(lineage_payload)
    lineage = DualTSSplitLineage.from_payload(lineage_payload)
    wave_fit = lineage.wave_groups("fit")
    queue, groups_mode = resolve_group_queue(lineage, args.groups)
    if groups_mode == "monitor_wave":
        if args.max_reactions is not None:
            print("[refuse] --max-reactions applies to the fit wave only; "
                  "not combinable with --groups", file=sys.stderr)
            return 2
    else:
        queue = list(wave_fit)
        if args.max_reactions is not None:
            queue = queue[:max(0, int(args.max_reactions))]
        # Partition echo discipline: refuse anything outside the fit wave.
        wave_fit_set = set(wave_fit)
        for gid in queue:
            if gid not in wave_fit_set or lineage.partition_of(gid) != "fit":
                print(f"[refuse] {gid}: not a fit-wave group; refusing per "
                      "lineage discipline", file=sys.stderr)
                return 2

    index_rows = irc_index_rows_for(
        args.truth_source, queue, allow_truth=args.allow_truth)
    print(f"[wp2] build mode={groups_mode} "
          f"queue={len(queue)}/{len(wave_fit)} "
          f"metric={ARC_LENGTH_METRIC_V1_DIAG_MASSES} a_max={args.a_max} "
          f"lineage_sha={sha256_file(lineage_path)[:12]} "
          f"allow_truth={args.allow_truth}", flush=True)

    print("[wp2] indexing v1 ACP receipt cache (read-only) ...", flush=True)
    cache_index, cache_audit = build_cache_index(args.observations)
    print(f"[wp2] cache: eval_dirs={cache_audit['eval_dirs']} "
          f"requests={cache_audit['requests_total']} "
          f"bindable+successful={cache_audit['successful_results']} "
          f"no_binding={cache_audit['requests_without_geometry_binding']} "
          f"identities={cache_audit['indexed_identities']}", flush=True)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    gaps_entries: list[dict] = []
    nodes_by_identity: dict[str, dict] = {}
    per_reaction_nodes: dict[str, int] = {}
    per_reaction_ts: dict[str, int] = {}
    per_reaction_insertions: dict[str, dict] = {}
    successes = 0

    for gid in queue:
        outcome, _adapter = prepare_one_group_fit50(
            gid, lineage=lineage, index_rows=index_rows,
            manifests_dir=args.truth_source, interim_dir=args.interim,
            split_dir=args.split_dir, shard_size=args.shard_size,
            a_max=args.a_max, allow_truth=args.allow_truth)
        entry = {
            "reaction_group_id": outcome.reaction_group_id,
            "reaction_id": outcome.reaction_id,
            "partition": outcome.partition,
            "status": outcome.status,
            "reason": outcome.reason,
        }
        gaps_entries.append(dict(outcome.gaps))
        if outcome.status != "ok":
            print(f"[wp2] {gid}: MISS {outcome.reason}", flush=True)
            results.append(entry)
            continue

        # Standalone re-validation before writing (validate_record).
        validate_record(TASK_SCHEMA_VERSION, outcome.task_payload)
        validate_record(TEACHER_SCHEMA_VERSION, outcome.teacher_payload)
        task_file = out / f"{gid}.task.json"
        teacher_file = out / f"{gid}.teacher.json"
        task_file.write_text(json.dumps(
            outcome.task_payload, ensure_ascii=False, indent=1))
        teacher_file.write_text(json.dumps(
            outcome.teacher_payload, ensure_ascii=False, indent=1))

        # HARD GATE: reload the written payloads and re-assert the full
        # cross-record identity (any failure aborts the build loudly).
        reloaded_task = DualEndpointTask.from_payload(
            json.loads(task_file.read_text()))
        reloaded_teacher = DualTSTeacher.from_payload(
            json.loads(teacher_file.read_text()))
        validate_record(TASK_SCHEMA_VERSION, reloaded_task.canonical_payload())
        validate_record(TEACHER_SCHEMA_VERSION, reloaded_teacher.to_payload())
        assert_task_teacher_identity(reloaded_task, reloaded_teacher)

        charge = int(outcome.task.charge)
        multiplicity = int(outcome.task.multiplicity)
        nodes = node_records_for_reaction(
            outcome, method=METHOD_GFN2_XTB, charge=charge,
            multiplicity=multiplicity, cache_index=cache_index)
        for node in nodes:
            key = node["computation_identity"]
            existing = nodes_by_identity.get(key)
            if existing is None:
                nodes_by_identity[key] = node
            else:
                existing["state_refs"].extend(node["state_refs"])

        diag = outcome.diagnostics
        ins_a = int(diag["insertion_count_a"])
        ins_b = int(diag["insertion_count_b"])
        exceed = outcome.gaps["items"]["a_max_exceedances"]
        ts_nodes = sum(1 for n in nodes if n["is_ts"])
        per_reaction_nodes[gid] = len(nodes)
        per_reaction_ts[gid] = ts_nodes
        per_reaction_insertions[gid] = {"A": ins_a, "B": ins_b}
        entry.update({
            "task_file": task_file.name,
            "teacher_file": teacher_file.name,
            "task_digest": diag["task_digest"],
            "teacher_digest": diag["teacher_digest"],
            "branch_a_steps": int(diag["branch_a_steps"]),
            "branch_b_steps": int(diag["branch_b_steps"]),
            "states": int(diag["n_states"]),
            "nodes": len(nodes),
            "ts_nodes": ts_nodes,
            "insertions": {"A": ins_a, "B": ins_b},
            "pre_insertion_exceedances": {
                "A": int(exceed["A"]["count"]),
                "B": int(exceed["B"]["count"]),
            },
            "bond_cross_check": diag["bond_cross_check_outcome"],
        })
        results.append(entry)
        successes += 1
        print(f"[wp2] {gid}: OK states={int(diag['n_states'])} "
              f"nodes={len(nodes)} ts_nodes={ts_nodes} "
              f"insA={ins_a} insB={ins_b} "
              f"bond={diag['bond_cross_check_outcome']}", flush=True)

    # --- coverage nodes artifact --------------------------------------------
    sorted_nodes = sorted(
        nodes_by_identity.values(),
        key=lambda n: (n["reaction_id"],
                       min((r["branch_id"], r["step_index"])
                           for r in n["state_refs"])))
    cache_hits = sum(1 for n in sorted_nodes
                     if n["cache"]["status"] == "cache_hit")
    to_compute = len(sorted_nodes) - cache_hits
    coverage = {
        "schema_version": COVERAGE_NODES_SCHEMA,
        "tool_version": WP2_FIT50_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "lineage_artifact": {
            "path": str(lineage_path.relative_to(ROOT)),
            "sha256": sha256_file(lineage_path),
        },
        "identity_scheme": {
            "geometry_content_digest":
                "sha256 over canonical JSON {'coordinates': round-10 exported "
                "coords} (same bytes both branches for the shared TS)",
            "atom_order_hash":
                "sha256 over canonical JSON {'atomic_numbers', "
                "'atom_map_ids'} with the G1 invariant row i <-> map i+1",
            "method": METHOD_GFN2_XTB,
            "operation": "eg_single_point (E/g single point)",
            "computation_identity":
                "sha256 over canonical JSON of {geometry_content_digest, "
                "atom_order_hash, method, charge, multiplicity, operation}",
            "canonical_json": "sorted keys, compact separators "
                              "(pes2ts_core.utils.hashing.stable_json_dumps)",
            "ts_dedupe": "the shared TS frame is bit-identical across the "
                         "two branches, so one computation-identity node per "
                         "reaction TS is attributed to both terminal states",
        },
        "cache_audit": {k: v for k, v in cache_audit.items()
                        if k != "durations_seconds"},
        "nodes": sorted_nodes,
    }
    coverage_path = out / "coverage_nodes.json"
    coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=1))

    # --- coverage summary + cost ---------------------------------------------
    exclusion_codes: dict[str, int] = {}
    for r in results:
        if r["status"] != "ok":
            code = str(r.get("reason", "")).split(":", 1)[0]
            exclusion_codes[code] = exclusion_codes.get(code, 0) + 1
    bond_counts: dict[str, int] = {}
    disputed_reactions: list[str] = []
    for gaps in gaps_entries:
        cross = (gaps.get("items") or {}).get("bond_cross_check")
        if not cross:
            continue
        outcome_name = str(cross.get("outcome"))
        bond_counts[outcome_name] = bond_counts.get(outcome_name, 0) + 1
        if outcome_name == "disputed":
            disputed_reactions.append(gaps["reaction_group_id"])
    exceed_pre_total = 0
    exceed_pre_max = 0.0
    for gaps in gaps_entries:
        exceed = (gaps.get("items") or {}).get("a_max_exceedances") or {}
        for side in ("A", "B"):
            info = exceed.get(side) or {}
            exceed_pre_total += int(info.get("count") or 0)
            exceed_pre_max = max(exceed_pre_max,
                                 float(info.get("max_abs_per_atom") or 0.0))
    states_total = sum(int(r.get("states") or 0) for r in results)
    inserted_total = sum(sum(v.values()) for v in
                         per_reaction_insertions.values())
    cost = estimate_cost(
        to_compute, cache_audit["durations_seconds"],
        workers=WORKERS, threads_per_worker=THREADS_PER_WORKER)
    summary = {
        "schema_version": COVERAGE_SUMMARY_SCHEMA,
        "tool_version": WP2_FIT50_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "build_mode": groups_mode,
        "groups_override": (
            {"mode": groups_mode, "group_ids": queue}
            if groups_mode == "monitor_wave" else None),
        "lineage_artifact": {
            "path": str(lineage_path.relative_to(ROOT)),
            "sha256": sha256_file(lineage_path),
        },
        "allow_truth": bool(args.allow_truth),
        "a_max_angstrom": float(args.a_max),
        "metric": {
            "arc_length_metric_version": ARC_LENGTH_METRIC_V1_DIAG_MASSES,
            "formula": "ell_t = sum_{k>=t} ||X_{k+1}-X_k||_M with "
                       "||delta||_M = sqrt(sum_i m_i ||delta_i||^2)",
            "masses_source": "v1 masses_for table (direct_preprocess."
                             "MASS_BY_Z) — the same table register_path uses",
            "wp1_note": "WP1 artifacts stay bound to "
                        "mass_weighted_arc_v0_PLACEHOLDER (untouched)",
        },
        "reactions": {
            "wave_fit_total": len(wave_fit),
            "attempted": len(results),
            "ok": successes,
            "excluded": len(results) - successes,
            "exclusion_codes": exclusion_codes,
        },
        "branches": {
            "count": 2 * successes,
            "states_total": states_total,
            "states_per_reaction_avg": round(states_total / successes, 3)
            if successes else 0.0,
        },
        "nodes": {
            "unique_computation_identities": len(sorted_nodes),
            "ts_shared_nodes": sum(per_reaction_ts.values()),
            "per_reaction": per_reaction_nodes,
            "ts_per_reaction": per_reaction_ts,
        },
        "insertions": {
            "reactions_with_insertion": sum(
                1 for v in per_reaction_insertions.values()
                if v["A"] or v["B"]),
            "nodes_inserted_total": inserted_total,
            "per_reaction": per_reaction_insertions,
        },
        "exceedances": {
            "pre_insertion_steps_total": exceed_pre_total,
            "post_insertion_steps_total": 0,
            "max_abs_per_atom_pre_insertion": round(exceed_pre_max, 6),
            "note": "post-insertion per-atom bound asserted in-build "
                    "(INSERTION_INCOMPLETE refusal otherwise)",
        },
        "bond_cross_check": {
            **bond_counts,
            "disputed_reactions": disputed_reactions,
            "policy": "changed-bond dispute -> bond_authority=disputed + "
                      "mapping_problem; reaction stays in the denominator; "
                      "spectator-only mismatch -> accept_with_note",
        },
        "cache": {
            "nodes_cache_hit": cache_hits,
            "nodes_to_compute": to_compute,
            "audit": {k: v for k, v in cache_audit.items()
                      if k != "durations_seconds"},
        },
        "gaps": {"entries": len(gaps_entries)},
        "cost": cost,
        "honesty_notes": [
            "no ACP/xTB execution and no network in this lane; v1 "
            "observations are read-only",
            "teacher states keep observation=null + registered reason; E/g "
            "values are WP2-B scope (computation, not audit)",
            "cache matching uses full geometry content from v1 request.json; "
            "requests without geometry content are never matched",
        ],
        "reactions_detail": results,
    }
    summary_path = out / "coverage_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    # --- gaps registry (new additive schema version; WP1 file untouched) ----
    registry = {
        "schema_version": GAPS_REGISTRY_SCHEMA_V1,
        "tool_version": WP2_FIT50_VERSION,
        "preprocess_version": DUAL_TS_PREPROCESS_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "lineage_artifact": {
            "path": str(lineage_path.relative_to(ROOT)),
            "sha256": sha256_file(lineage_path),
        },
        "a_max_angstrom": float(args.a_max),
        "arc_length_metric_version": ARC_LENGTH_METRIC_V1_DIAG_MASSES,
        "drift_rule": DRIFT_RULE,
        "wp_scope_note": "wp2 inserts midpoint nodes on exceedance, "
                         "cross-checks endpoint bonds, and audits cache "
                         "coverage; actual E/g computation is wp2_b",
        "reactions": gaps_entries,
    }
    registry_path = out / "gaps_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=1))

    wall = time.perf_counter() - t0
    print(f"[done] attempted={len(results)} ok={successes} "
          f"missed={len(results) - successes} "
          f"nodes={len(sorted_nodes)} ts_nodes={sum(per_reaction_ts.values())} "
          f"cache_hits={cache_hits} to_compute={to_compute} "
          f"wall={wall:.1f}s -> {out}", flush=True)
    return 0 if successes >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
