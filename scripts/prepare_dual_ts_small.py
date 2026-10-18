#!/usr/bin/env python3
"""WP1 small package: dual-endpoint TS teacher preprocessing on real IRC data.

Reads the frozen dual-TS split lineage artifact, takes the FIRST N fit-wave
reaction groups (default 3), and for each builds:

* the unordered dual-endpoint task record (``pesn_dual_endpoint_task_v2``),
* the full two-branch TS teacher record (``pesn_dual_ts_teacher_v2``),
* the per-reaction itemized gaps registry entry (missing branches, mapping
  problems, geometry quality, branch step counts, a_max exceedances — flagged,
  never inserted — and missing observations, which are WP2 scope).

IRC frames are read ONLY through the audited v1 truth bridge (allow-truth
gated, frozen-split scope checked).  The truth source location is resolved
from ``truth_manifest.json`` inside the manifests directory (v1 config
default ``data/manifests``); no quarantined path is ever hardcoded here.

A failing reaction is registered honestly and the next fit-wave id is taken;
the deliverable is the honest small package, not forced success.  Read-only
on every v1/ground-truth source; writes only under ``--out``.
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
    TASK_SCHEMA_VERSION,
    TEACHER_SCHEMA_VERSION,
    validate_record,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (  # noqa: E402
    A_MAX_ANGSTROM,
    BOND_PERCEPTION_AUTHORITY,
    BOND_PERCEPTION_COVALENT,
    DUAL_TS_PREPROCESS_VERSION,
    DualTSPreprocessError,
    build_dual_ts_reaction,
    endpoint_graph_from_bond_list,
    endpoint_graph_from_geometry,
    excluded_outcome,
    load_irc_frames_for_reaction,
    order_frames_and_energy,
)
from pes2ts_core.pesn.offline.data.endpoint_adapter import (  # noqa: E402
    EndpointAdapterError,
    adapt_reaction,
    load_authority_document,
)
from pes2ts_core.pesn.offline.splits.dual_ts_split import (  # noqa: E402
    DualTSSplitLineage,
    DualTSSplitLineageError,
    validate_lineage_payload,
)

DEFAULT_LINEAGE = ROOT / (
    "outputs/pesn/dual_ts_v2_20261015/splits/dual_ts_split_lineage_v1.json")
DEFAULT_OUT = ROOT / "outputs/pesn/dual_ts_v2_20261015/wp1_small"
DEFAULT_MANIFESTS = ROOT / "data/manifests"       # v1 config paths.manifests
DEFAULT_INTERIM = ROOT / "data/interim"           # v1 config paths.interim
DEFAULT_SPLIT_DIR = ROOT / (
    "outputs/pesn/direct_egnn_v1_20261015/splits")  # v1 config paths.split_dir

GAPS_REGISTRY_SCHEMA = "pes2ts_dual_ts_gaps_registry_v0"
RUN_SUMMARY_SCHEMA = "pes2ts_dual_ts_wp1_run_summary_v0"


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


def prepare_one_group(
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
) -> tuple:
    """Build one group; returns (outcome, adapter_view_or_None)."""
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
        adapter = None
        bond_perception = BOND_PERCEPTION_COVALENT
        authority_note = f"authority document unavailable ({exc}); covalent fallback"
    else:
        authority_note = ""

    # --- IRC index verification: exists, atom count, two-sided usable ------
    index_row = index_rows.get(group_id)
    index_issues: list[str] = []
    if index_row is None:
        index_issues.append("reaction absent from the IRC index")
    else:
        if adapter is not None and \
                int(index_row["n_atoms"]) != len(adapter["atomic_numbers_by_map"]):
            index_issues.append(
                f"index n_atoms {index_row['n_atoms']} != task graph atoms "
                f"{len(adapter['atomic_numbers_by_map'])}")

    if adapter is None:
        # Covalent fallback path needs a two-sided layout from the index
        # alone; without authority layout metadata a dual split cannot be
        # registered honestly here (layout arrives with the G1 doc).
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition,
            reason=f"{authority_note}; IRC index issues: {index_issues or 'none'}",
            error_code="AUTHORITY_DOC_MISSING",
            items={"mapping_problems": [authority_note] or [],
                   "geometry_quality_issues": list(index_issues)}), None

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

    # --- endpoint graphs (v1/G1 bond source; covalent fallback otherwise) ---
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

    # --- order + split + build ---------------------------------------------
    try:
        ordered, ordered_e, ts_position = order_frames_and_energy(
            frames, energies,
            layout=layout, orientation=orientation,
            ts_frame_index=int(ts_frame_index), splice_index=int(splice_index))
        if index_row is not None and int(index_row["n_frames"]) != len(ordered):
            quality_notes.append(
                f"index n_frames {index_row['n_frames']} != archive frames "
                f"{len(ordered)}")
        outcome = build_dual_ts_reaction(
            reaction_id=group_id,
            reaction_group_id=group_id,
            partition=partition,
            atomic_numbers=np.asarray(adapter["atomic_numbers_by_map"], dtype=int),
            ordered_path=ordered,
            authority_ts_position=ts_position,
            energies=ordered_e,
            graph_r=graph_r,
            graph_p=graph_p,
            bond_perception=bond_perception,
            charge=int(adapter.get("charge_total_reactants") or 0),
            multiplicity=int(adapter.get("multiplicity_max") or 1),
            a_max_angstrom=a_max,
            quality_notes=quality_notes,
        )
        return outcome, adapter
    except DualTSPreprocessError as exc:
        return excluded_outcome(
            reaction_id=group_id, reaction_group_id=group_id,
            partition=partition, reason=exc.detail,
            error_code=exc.code,
            items={"geometry_quality_issues": list(quality_notes)}), adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default=None,
                        help="comma-separated fit-wave group ids "
                             "(default: first 3 fit-wave ids of the lineage)")
    parser.add_argument("--target-count", type=int, default=3,
                        help="number of successful reactions to deliver "
                             "(default 3)")
    parser.add_argument("--max-attempts", type=int, default=8,
                        help="stop after this many attempts (failover "
                             "consumes next fit-wave ids)")
    parser.add_argument("--lineage", type=Path, default=DEFAULT_LINEAGE)
    parser.add_argument("--truth-source", type=Path, default=DEFAULT_MANIFESTS,
                        help="manifests directory holding truth_manifest.json "
                             "(v1 config default data/manifests); the "
                             "quarantined source path is resolved from the "
                             "manifest, never from a literal")
    parser.add_argument("--interim", type=Path, default=DEFAULT_INTERIM)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
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
    if args.groups:
        requested = [g.strip() for g in args.groups.split(",") if g.strip()]
    else:
        requested = wave_fit[:args.target_count]
    # Refuse anything outside the fit wave (partition echo discipline).
    wave_fit_set = set(wave_fit)
    for gid in requested:
        partition = lineage.partition_of(gid)
        if gid not in wave_fit_set or partition != "fit":
            print(f"[refuse] {gid}: not a fit-wave group (partition="
                  f"{partition!r}); refusing per lineage discipline",
                  file=sys.stderr)
            return 2

    # Candidate queue: requested first, then remaining fit-wave ids as
    # honest failover (a miss consumes the next id, per WP1 instructions).
    queue = list(requested) + [g for g in wave_fit if g not in set(requested)]
    queue = queue[:max(args.max_attempts, len(requested))]

    index_rows = irc_index_rows_for(
        args.truth_source, queue, allow_truth=args.allow_truth)
    print(f"[wp1] target={args.target_count} queue={len(queue)} "
          f"lineage_sha={sha256_file(lineage_path)[:12]} "
          f"allow_truth={args.allow_truth}", flush=True)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = []
    gaps_entries = []
    successes = 0
    attempts = 0
    for gid in queue:
        if successes >= args.target_count or attempts >= args.max_attempts:
            break
        attempts += 1
        outcome, _adapter = prepare_one_group(
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
        if outcome.status == "ok":
            # Standalone re-validation before writing (validate_record).
            validate_record(TASK_SCHEMA_VERSION, outcome.task_payload)
            validate_record(TEACHER_SCHEMA_VERSION, outcome.teacher_payload)
            task_file = out / f"{gid}.task.json"
            teacher_file = out / f"{gid}.teacher.json"
            task_file.write_text(json.dumps(
                outcome.task_payload, ensure_ascii=False, indent=1))
            teacher_file.write_text(json.dumps(
                outcome.teacher_payload, ensure_ascii=False, indent=1))
            diag = outcome.diagnostics
            entry.update({
                "task_file": task_file.name,
                "teacher_file": teacher_file.name,
                "task_digest": diag["task_digest"],
                "teacher_digest": diag["teacher_digest"],
                "branch_a_steps": int(diag["branch_a_steps"]),
                "branch_b_steps": int(diag["branch_b_steps"]),
                "ts_position": int(diag["ts_position"]),
                "n_frames_ordered": int(diag["n_frames_ordered"]),
            })
            successes += 1
            print(f"[wp1] {gid}: OK branches="
                  f"{diag['branch_a_steps']:.0f}+{diag['branch_b_steps']:.0f} "
                  f"frames={diag['n_frames_ordered']:.0f} "
                  f"exceedA={outcome.gaps['items']['a_max_exceedances']['A']['count']} "
                  f"exceedB={outcome.gaps['items']['a_max_exceedances']['B']['count']}",
                  flush=True)
        else:
            print(f"[wp1] {gid}: MISS {outcome.reason}", flush=True)
        results.append(entry)
        gaps_entries.append(dict(outcome.gaps))

    wall = time.perf_counter() - t0
    registry = {
        "schema_version": GAPS_REGISTRY_SCHEMA,
        "tool_version": DUAL_TS_PREPROCESS_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "lineage_artifact": {
            "path": str(lineage_path.relative_to(ROOT)),
            "sha256": sha256_file(lineage_path),
        },
        "a_max_angstrom": float(args.a_max),
        "wp_scope_note": "wp1 flags exceedances/missing only; node insertion "
                         "and observation completion are wp2",
        "reactions": gaps_entries,
    }

    summary = {
        "schema_version": RUN_SUMMARY_SCHEMA,
        "tool_version": DUAL_TS_PREPROCESS_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 3),
        "lineage_artifact": {
            "path": str(lineage_path.relative_to(ROOT)),
            "sha256": sha256_file(lineage_path),
        },
        "groups_requested": requested,
        "groups_attempted": [r["reaction_group_id"] for r in results],
        "counts": {
            "requested": len(requested),
            "attempted": len(results),
            "succeeded": successes,
            "missed": len(results) - successes,
        },
        "a_max_angstrom": float(args.a_max),
        "allow_truth": bool(args.allow_truth),
        "reactions": results,
    }
    registry_path = out / "gaps_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=1))
    summary_path = out / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[done] attempted={len(results)} ok={successes} "
          f"missed={len(results) - successes} wall={wall:.1f}s -> {out}",
          flush=True)
    return 0 if successes >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
