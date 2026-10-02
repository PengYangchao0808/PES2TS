"""Small, replaceable rule rankers operating only on frozen PathBundle data."""
from __future__ import annotations

import hashlib
from typing import Any

from pes2ts_core.contracts import ContractError, dumps_document, make_document
from pes2ts_core.utils.hashing import stable_json_dumps


def rank_path_bundle(path: dict[str, Any], *, rule: str = "highest_scan_energy", top_k: int = 3) -> dict[str, Any]:
    """Rank frames without recalculating a path; missing energies are skipped."""
    dumps_document(path)
    if rule not in {"highest_scan_energy", "internal_scan_peak"}:
        raise ContractError(f"unsupported ranking rule: {rule}")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ContractError("top_k must be a positive integer")
    proposal_id = "proposal:" + hashlib.sha256(stable_json_dumps({
        "path_content_sha256": path["content_sha256"], "rule": rule,
        "ranking_version": "rule-ranker-v1", "top_k": top_k,
    }).encode()).hexdigest()[:20]
    if path["status"] == "unusable":
        return make_document("SeedProposal", proposal_id, "rejected",
            reaction_id=path["reaction_id"], case_id=path["case_id"], path_id=path["object_id"],
            path_content_sha256=path["content_sha256"], path_status=path["status"],
            ranking_version="rule-ranker-v1", rule=rule, top_k=top_k,
            selection_source="ranking", selected_frames=[], rejected_reason="source PathBundle is unusable")
    energies = []
    for frame in path["frames"]:
        channel = frame.get("energies", {}).get("scan_electronic")
        energies.append(channel["value"] if channel is not None else None)
    if rule == "highest_scan_energy":
        indices = sorted((i for i, e in enumerate(energies) if e is not None), key=lambda i: (-energies[i], i))
    else:
        indices = [i for i in range(1, len(energies)-1)
                   if energies[i] is not None and energies[i-1] is not None and energies[i+1] is not None
                   and energies[i] >= energies[i-1] and energies[i] >= energies[i+1]]
        indices.sort(key=lambda i: (-energies[i], i))
    selected = []
    for rank, index in enumerate(indices[:top_k], start=1):
        frame = path["frames"][index]
        geom_digest = hashlib.sha256(stable_json_dumps(frame["geometry"]).encode()).hexdigest()
        selected.append({"rank": rank, "frame_id": frame["frame_id"], "score": energies[index],
                         "score_unit": "hartree", "reason": rule,
                         "geometry_sha256": geom_digest})
    status = "accepted" if selected and path["status"] == "usable" else "needs_review" if selected else "rejected"
    review_note = (f"source PathBundle status is {path['status']}; ranking is provisional"
                   if selected and path["status"] != "usable" else None)
    return make_document("SeedProposal", proposal_id,
        status, reaction_id=path["reaction_id"], case_id=path["case_id"],
        path_id=path["object_id"], path_content_sha256=path["content_sha256"],
        path_status=path["status"], ranking_version="rule-ranker-v1", rule=rule, top_k=top_k,
        selection_source="ranking", selected_frames=selected,
        rejected_reason=None if selected else "no frames have a comparable scan energy", review_note=review_note)
