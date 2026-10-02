"""Truth-free, hash-bound TS proposals from complete OR partial prefixes."""
from __future__ import annotations

import math
import numpy as np

from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM, force_evidence
from pes2ts_core.generation.planning.synchronized_path import align, digest


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
            proposals[i] = {"frame_index": i, "energy_peak_frame": None,
                           "evidence_class": "unbracketed_prefix_edge",
                           "physical_gradient_norm_hartree_per_bohr": forces[i]["physical_gradient_norm_hartree_per_bohr"],
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
    return {"schema_version": "pes2ts_gradient_seed_proposals_v1", "candidate_only": True,
            "rule": "bound_full_gradient_within_peak_neighbourhood_then_distinct_prefix_edge",
            "source_path_sha256": digest(result), "plan_sha256": plan["content_sha256"],
            "completed_interval": result.get("completed_interval", False),
            "candidates": [{**r, "rank": i+1} for i,r in enumerate(chosen)],
            "n_bracketed_peaks": len(peaks), "reference_geometry_used": False}
