"""Truth-free, hash-bound TS proposals from complete OR partial prefixes."""
from __future__ import annotations

import hashlib
import math
import numpy as np

from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM, force_evidence
from pes2ts_core.generation.planning.synchronized_path import align, digest

#: Frozen bundle schema consumed by the validation entry (G2-AB1 WP-4).
GRADIENT_SEED_PROPOSALS_SCHEMA = "pes2ts_gradient_seed_proposals_v1"


def bind_candidate(candidate, result, plan):
    """Normalize legacy frame keys, then enforce geometry/path provenance."""
    index = candidate.get("frame_index", candidate.get("selected_frame"))
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(result["frames"]):
        raise ValueError("INVALID_CANDIDATE_FRAME_INDEX")
    frame = result["frames"][index]
    geometry_hash = digest(frame["geometry"])
    if candidate.get("geometry_sha256", geometry_hash) != geometry_hash:
        raise ValueError("CANDIDATE_GEOMETRY_MISMATCH")
    if candidate.get("frame_id", frame["frame_id"]) != frame["frame_id"]:
        raise ValueError("CANDIDATE_FRAME_ID_MISMATCH")
    path_hash = digest(result)
    if candidate.get("source_path_sha256", path_hash) != path_hash:
        raise ValueError("CANDIDATE_PATH_MISMATCH")
    if candidate.get("plan_sha256", plan["content_sha256"]) != plan["content_sha256"]:
        raise ValueError("CANDIDATE_PLAN_MISMATCH")
    return {**candidate, "frame_index": index, "frame_id": frame["frame_id"],
            "geometry_sha256": geometry_hash, "source_path_sha256": path_hash,
            "plan_sha256": plan["content_sha256"], "completed_interval": result["completed_interval"],
            "reference_geometry_used": False,
            "stationary_point_verified": False, "reaction_connection_verified": False}


def rank_continuation_candidates(result, plan, *, physical_evidence=None, top_k=3,
                                 allow_edge_candidates=True):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("INVALID_CANDIDATE_BUDGET")
    frames = result.get("frames", [])
    forces = {}
    for row in (physical_evidence or {}).get("frames", []):
        if row.get("status") == "bound":
            forces[row["frame_index"]] = row
    for i, frame in enumerate(frames):
        gradient = frame.get("physical_gradient_hartree_per_angstrom")
        if gradient is not None and frame.get("physical_gradient_status") == "bound":
            forces[i] = force_evidence(plan, frame, {"status": "bound",
                "gradient_hartree_per_bohr": (np.asarray(gradient)*BOHR_ANGSTROM).tolist()})
    peaks = []
    for i in range(1, len(frames)-1):
        energies = [frames[j].get("energy_hartree") for j in (i-1, i, i+1)]
        if not all(e is not None and math.isfinite(e) for e in energies):
            continue
        if energies[1] > energies[0] and energies[1] >= energies[2]:
            peaks.append(i)
        elif all(j in forces for j in (i-1, i+1)):
            left = forces[i-1]["reconstructed_energy_slope_hartree_per_lambda"]
            right = forces[i+1]["reconstructed_energy_slope_hartree_per_lambda"]
            if left > 0 > right:
                peaks.append(i)
    proposals = {}
    for peak in peaks:
        for i in range(max(1, peak-1), min(len(frames)-1, peak+2)):
            force = forces.get(i)
            if force is None:
                continue
            gnorm = force["physical_gradient_norm_hartree_per_bohr"]
            if not math.isfinite(gnorm):
                continue
            row = {"frame_index": i, "energy_peak_frame": peak, "evidence_class": "bracketed_peak_neighbourhood",
                   "physical_gradient_norm_hartree_per_bohr": gnorm,
                   "free_gradient_norm_hartree_per_bohr": force["perpendicular_gradient_norm_hartree_per_bohr"],
                   "reconstructed_energy_slope_hartree_per_lambda": force["reconstructed_energy_slope_hartree_per_lambda"],
                   "energy_hartree": frames[i]["energy_hartree"]}
            proposals.setdefault(i, row)
    if allow_edge_candidates and not result.get("completed_interval") and len(frames) >= 3:
        i = len(frames)-1
        if i in forces and frames[i]["lambda"] >= .1 and i not in proposals:
            edge = forces[i]
            proposals[i] = {"frame_index": i, "energy_peak_frame": None,
                            "evidence_class": "unbracketed_prefix_edge",
                            "physical_gradient_norm_hartree_per_bohr": edge["physical_gradient_norm_hartree_per_bohr"],
                            "free_gradient_norm_hartree_per_bohr": edge["perpendicular_gradient_norm_hartree_per_bohr"],
                            "reconstructed_energy_slope_hartree_per_lambda": edge["reconstructed_energy_slope_hartree_per_lambda"],
                            "energy_hartree": frames[i]["energy_hartree"]}
    ordered = sorted(proposals.values(), key=lambda r: (
        r["evidence_class"] != "bracketed_peak_neighbourhood",
        r["physical_gradient_norm_hartree_per_bohr"], -r["energy_hartree"], r["frame_index"]))
    chosen = []
    for proposal in ordered:
        x = np.asarray(frames[proposal["frame_index"]]["geometry"])
        if any(np.sqrt(np.mean(np.sum((align(x, frames[r["frame_index"]]["geometry"])
                                      -frames[r["frame_index"]]["geometry"])**2, axis=1))) < .04 for r in chosen):
            continue
        chosen.append(bind_candidate(proposal, result, plan))
        if len(chosen) >= top_k:
            break
    return {"schema_version": GRADIENT_SEED_PROPOSALS_SCHEMA, "candidate_only": True,
            "rule": "bound_full_gradient_within_peak_neighbourhood_then_distinct_prefix_edge",
            "source_path_sha256": digest(result), "plan_sha256": plan["content_sha256"],
            "completed_interval": result.get("completed_interval", False),
            "candidates": [{**r, "rank": i+1} for i,r in enumerate(chosen)],
            "n_bracketed_peaks": len(peaks), "reference_geometry_used": False}


def continuation_path_bundle(result, plan, *, case_id, plan_id=None, execution_id=None):
    """Project a continuation result into a minimal contract PathBundle.

    G2-AB1 WP-4: partial continuation paths may nominate candidates, so the
    projection marks the bundle usable-as-seed-source while carrying
    ``completed_interval`` as a REPORT-ONLY extension field — path integrity
    and candidate validation are reported separately and neither gates the
    other.  ``acp_task_id`` stays null: a local continuation run is not an
    ACP scheduler task.
    """
    from pes2ts_core.contracts import make_document
    frames = []
    for index, frame in enumerate(result.get("frames", [])):
        energy = frame.get("energy_hartree")
        frames.append({"frame_id": frame["frame_id"], "frame_index": index,
                       "atom_map_ids": list(plan["atom_map_order"]),
                       "geometry": frame["geometry"], "elements": list(plan["elements"]),
                       "energies": ({"constrained_physical": {"value": float(energy),
                                                               "unit": "hartree", "method_id": None}}
                                    if energy is not None and math.isfinite(float(energy)) else {})})
    source_path_sha256 = digest(result)
    return make_document(
        "PathBundle", f"path:continuation:{source_path_sha256[:16]}", "usable",
        reaction_id=plan["reaction_id"], case_id=case_id,
        plan_id=plan_id or f"plan:continuation:{plan['content_sha256'][:16]}",
        candidate_id="candidate:continuation", 
        execution_id=execution_id or f"execution:continuation:{source_path_sha256[:16]}",
        acp_task_id=None, atom_map_ids=list(plan["atom_map_order"]),
        n_atoms=len(plan["elements"]), energy_reference="frame_zero_constrained_physical",
        frames=frames, supersedes=None,
        extensions={"pes2ts.continuation_projection.v1": {
            "source_path_sha256": source_path_sha256,
            "plan_sha256": plan["content_sha256"],
            "completed_interval": bool(result.get("completed_interval", False)),
            "completed_interval_is_report_only": True,
            "path_integrity_reported_separately": True}})


def seed_proposal_from_gradient_ranking(path_bundle, proposals, *, rank=1):
    """Convert the rank-th gradient seed candidate into a contract SeedProposal.

    This is the ONLY bridge from ``pes2ts_gradient_seed_proposals_v1`` into
    the formal validation chain (no ``ranking.json`` fallback): the selected
    candidate's geometry hash must match a frame of ``path_bundle`` verbatim,
    otherwise the conversion is a typed rejection.  The proposal records the
    canonical candidate evidence and marks
    ``preparation_layer_status = absent`` (constitution §8).
    """
    from pes2ts_core.contracts import make_document
    from pes2ts_core.utils.hashing import stable_json_dumps
    if proposals.get("schema_version") != GRADIENT_SEED_PROPOSALS_SCHEMA:
        raise ValueError("INVALID_GRADIENT_SEED_PROPOSALS_SCHEMA")
    candidates = proposals.get("candidates", [])
    if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= len(candidates):
        raise ValueError("NO_BOUND_CANDIDATE")
    candidate = candidates[rank-1]
    frame = next((row for row in path_bundle["frames"] if hashlib.sha256(
        stable_json_dumps(row["geometry"]).encode()).hexdigest() == candidate["geometry_sha256"]), None)
    if frame is None:
        raise ValueError("CANDIDATE_GEOMETRY_NOT_IN_PATH_BUNDLE")
    selected = {"rank": 1, "frame_id": frame["frame_id"], "frame_index": frame["frame_index"],
                "score": candidate["physical_gradient_norm_hartree_per_bohr"],
                "score_unit": "hartree/bohr",
                "reason": candidate["evidence_class"],
                "geometry_sha256": candidate["geometry_sha256"],
                "evidence_class": candidate["evidence_class"],
                "physical_gradient_norm_hartree_per_bohr": candidate["physical_gradient_norm_hartree_per_bohr"],
                "free_gradient_norm_hartree_per_bohr": candidate["free_gradient_norm_hartree_per_bohr"],
                "reconstructed_energy_slope_hartree_per_lambda": candidate["reconstructed_energy_slope_hartree_per_lambda"],
                "plan_sha256": candidate["plan_sha256"],
                "source_path_sha256": candidate["source_path_sha256"],
                "completed_interval": bool(proposals.get("completed_interval", False)),
                "reference_geometry_used": False,
                "stationary_point_verified": False,
                "reaction_connection_verified": False}
    status = "accepted" if path_bundle["status"] == "usable" else "needs_review"
    return make_document(
        "SeedProposal", f"proposal:gradient:{candidate['geometry_sha256'][:20]}", status,
        reaction_id=path_bundle["reaction_id"], case_id=path_bundle["case_id"],
        path_id=path_bundle["object_id"], path_content_sha256=path_bundle["content_sha256"],
        path_status=path_bundle["status"], ranking_version="gradient-seed-v1",
        rule=proposals["rule"], top_k=len(candidates), selection_source="ranking",
        selected_frames=[selected], rejected_reason=None,
        review_note=None if status == "accepted" else "projected path is not usable",
        extensions={"pes2ts.gradient_seed_proposals.v1": {
            "preparation_layer_status": "absent",
            "completed_interval_is_report_only": True}})
