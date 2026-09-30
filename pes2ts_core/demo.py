"""Generate an explicitly synthetic seven-object contract demonstration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pes2ts_core.contracts import make_document, seal_document
from pes2ts_core.integration.acp.adapter import acp_result_to_path_bundle
from pes2ts_core.planning import build_minimal_scan_plan
from pes2ts_core.ranking import rank_path_bundle
from pes2ts_core.utils.jsonio import write_json
from pes2ts_core.viewer import render_path_viewer


def synthetic_objects() -> dict[str, dict[str, Any]]:
    case = make_document("ReactionCase", "case:synthetic-h-transfer-v1", "ready",
        dataset_version="synthetic-v1", reaction_id="RXN_SYNTH_0001", case_id="case:synthetic-h-transfer-v1", split="train",
        atoms=[{"atom_map_id":2,"element":"C"},{"atom_map_id":7,"element":"H"},{"atom_map_id":12,"element":"N"}],
        reactant={"charge":0,"multiplicity":1,"geometry":[[0,0,0],[1.10,0,0],[3.50,0,0]]},
        product={"charge":0,"multiplicity":1,"geometry":[[0,0,0],[1.80,0,0],[2.80,0,0]]},
        edits=[{"kind":"broken","atom_map_ids":[2,7]},{"kind":"formed","atom_map_ids":[7,12]}],
        hydrogen_transfers=[{"hydrogen_map_id":7,"old_partner_map_id":2,"new_partner_map_id":12,"kind":"transfer"}],
        source={"dataset":"synthetic","mapping_provenance":"endpoint_only"})
    plan = build_minimal_scan_plan(case, experiment_id="experiment:synthetic-v1")
    # The curve intentionally makes highest-energy and internal-peak rules
    # select different frames, with one missing point retained as a gap.
    frames = []
    energies = [-14.1, -14.8, -14.4, -14.5, None, -15.2]
    for i, energy in enumerate(energies):
        fraction = i / (len(energies) - 1)
        geometry = [[0.0, 0.0, 0.0], [1.1 + 0.7 * fraction, 0.0, 0.0],
                    [3.5 - 0.7 * fraction, 0.0, 0.0]]
        frames.append({"frame_id":f"frame-syn-{i:03d}","atom_map_ids":[2,7,12],
            "geometry_angstrom":geometry,"scan_energy_hartree":energy,
            "scan_method_id":"xtb:gfn2-xTB","refined_energy_hartree":None,
            "refined_method_id":"orca:single-point","converged":i != 3})
    path = acp_result_to_path_bundle(case=case, plan=plan, execution_id="execution:synthetic-1",
        acp_task_id="acp-task:synthetic-1", frames=frames, path_status="needs_review")
    # Make this scan reverse direction relative to R->P and reseal its digest.
    reverse_plan = dict(plan)
    reverse_plan["candidates"] = [{**plan["candidates"][0], "direction":"P_to_R", "start_endpoint":"product"}]
    reverse_plan = seal_document(reverse_plan)
    execution = make_document("ExecutionRecord", "execution:synthetic-1", "completed",
        reaction_id=case["reaction_id"], case_id=case["case_id"], plan_id=plan["plan_id"],
        candidate_id=plan["candidates"][0]["candidate_id"], request_id=plan["candidates"][0]["request_id"],
        acp_task_id="acp-task:synthetic-1", attempts=[
            {"attempt_id":"attempt:1","status":"failed","failure_code":"synthetic_timeout","cpu_seconds":12},
            {"attempt_id":"attempt:2","status":"completed","cpu_seconds":48}],
        total_cpu_seconds=60, cost_complete=True, failure_retained=True,
        work_ref="WORK/attempts/", result_ref="RESULT/path_bundle.json")
    proposal = rank_path_bundle(path, rule="highest_scan_energy", top_k=3)
    validation = make_document("ValidationResult", "validation:synthetic-1", "not_run",
        reaction_id=case["reaction_id"], case_id=case["case_id"], path_id=path["object_id"],
        proposal_id=proposal["object_id"], optts={"status":"not_run"}, frequency={"status":"not_run"},
        irc_forward={"status":"not_run"}, irc_reverse={"status":"not_run"},
        validation_note="synthetic interface example; no physical validation performed")
    experiment = make_document("ExperimentManifest", "experiment:synthetic-v1", "frozen",
        experiment_id="experiment:synthetic-v1", dataset_version="synthetic-v1", splits={"train":[case["reaction_id"]]},
        policy_version="demo-contracts-v1", budgets={"max_attempts":2}, validation_protocol="not_run")
    rejected = make_document("ScanPlan", "plan:synthetic-rejected", "rejected",
        plan_id="plan:synthetic-rejected", experiment_id="experiment:synthetic-v1",
        dataset_version="synthetic-v1", reaction_id=case["reaction_id"], case_id=case["case_id"],
        split="train", plan_version=1, atom_map_ids=[2,7,12], n_atoms=3, candidate_strategy="none",
        source_case_sha256=case["content_sha256"], plan_frozen=False, candidates=[],
        reject_reasons=["synthetic rejection example: ACP capability unavailable"])
    return {"ExperimentManifest":experiment,"ReactionCase":case,"ScanPlan":reverse_plan,
            "ExecutionRecord":execution,"PathBundle":path,"SeedProposal":proposal,
            "ValidationResult":validation,"RejectedPlan":rejected}


def write_synthetic_bundle(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    documents = synthetic_objects()
    for name, document in documents.items():
        write_json(root / f"{name}.json", document)
    write_json(
        root / "SeedProposal.internal_scan_peak.json",
        rank_path_bundle(documents["PathBundle"], rule="internal_scan_peak", top_k=3),
    )
    (root / "README.md").write_text(
        "# Synthetic contracts v1 bundle\n\n"
        "All records here are synthetic. They cover all seven core contracts, a rejected plan, "
        "a failed attempt followed by a successful retry, distinct highest-energy and internal-peak "
        "proposal files for independent ACP display, stable frame identity with a missing energy, reverse scan direction, separate "
        "energy channels, and an unrun physical validation. "
        "No value is evidence of a real calculation or validated TS. Open `path_viewer.html` for the "
        "offline energy/structure view; its distance-based bonds are only visual hints.\n", encoding="utf-8")
    render_path_viewer(synthetic_objects()["PathBundle"], root / "path_viewer.html")
    return root
