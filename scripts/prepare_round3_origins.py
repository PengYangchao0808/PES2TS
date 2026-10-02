"""Independent three-case origin investigation, outside matched path counts."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from run_demo24_round3 import ROOT,OLD,OUT,read,write_json
from pes2ts_core.integration.acp.continuation_backend import ORCAContinuationBackend
from pes2ts_core.generation.planning.connectivity_plan import input_screen
from pes2ts_core.generation.planning.graph_rebuild import load_endpoint_materials_from_export,rebuild_endpoint_graphs
from pes2ts_core.generation.planning.origin_preparation import repair_component_overlap,torsion_hypotheses
from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.generation.planning.ts_checks import mapped_endpoint_identity


def run(rid):
    snapshot=read(ROOT/"tests/fixtures/p0_demo24/records"/f"{rid}.json")
    bundle=rebuild_endpoint_graphs(snapshot["reaction_smiles"],load_endpoint_materials_from_export(snapshot))
    index={m:i for i,m in enumerate(snapshot["maps"])}
    records=[]
    for side,graph in (("R",bundle.r_graph),("P",bundle.p_graph)):
        bonds=[[index[e.map_a],index[e.map_b]] for e in graph.edges]
        x=np.asarray(snapshot["r_coordinates" if side=="R" else "p_coordinates"])
        issues=input_screen(x,snapshot["elements"],bonds)
        preparation={"kind":"original_endpoint"}
        if issues:
            groups=[[index[m] for m in component.map_ids] for component in graph.components]
            x,evidence=repair_component_overlap(x,snapshot["elements"],groups)
            preparation={"kind":"component_translation_repair",**evidence}
            if x is None:
                records.append({"side":side,"status":"PLACEMENT_FAILED","preparation":preparation})
                continue
        hypotheses=[(x,preparation)]
        if rid!="RXN_0000155302":
            single_bonds=[[index[e.map_a],index[e.map_b]] for e in graph.edges if e.bond_order==1.]
            hypotheses.extend(torsion_hypotheses(x,bonds,maximum=2,rotatable_bonds=single_bonds))
        state=snapshot["endpoint_electronic"]["reactant" if side=="R" else "product"]
        for number,(geometry,hypothesis) in enumerate(hypotheses):
            folder=OUT/"origin_investigation"/rid/side/f"hypothesis-{number}"
            variant={**snapshot,"r_coordinates" if side=="R" else "p_coordinates":geometry.tolist()}
            write_json(folder/"input_variant.json",{"snapshot":variant,"source_sha256":digest(snapshot),
                "variant_sha256":digest(variant),"hypothesis":hypothesis,"original_input_replaced":False})
            starting_issues=input_screen(geometry,snapshot["elements"],bonds)
            if starting_issues:
                records.append({"side":side,"hypothesis":number,"status":"INPUT_INVALID","issues":starting_issues})
                continue
            backend=ORCAContinuationBackend(charge=state["charge"],multiplicity=state["multiplicity"],
                elements=snapshot["elements"],folder=folder,opt_level="tight")
            result=backend.call(geometry,[],"free_opt",free=True)
            geometry_issues=input_screen(result["coordinates"],snapshot["elements"],bonds) if result.get("coordinates") else []
            identity=mapped_endpoint_identity(result["coordinates"],snapshot,side) if result.get("coordinates") else {"matched":False}
            row={"reaction_id":rid,"side":side,"hypothesis":number,"preparation":hypothesis,
                 "optimization_success":result["success"],"geometry_issues":geometry_issues,"identity":identity,
                 "valid_origin_candidate":bool(result["success"] and not geometry_issues and identity["matched"]),
                 "minimum_frequency_verified":False,"duration_seconds":result["duration_seconds"]}
            records.append(row);write_json(folder/"assessment.json",row)
    answer={"reaction_id":rid,"records":records,"n_valid_candidates":sum(r.get("valid_origin_candidate",False) for r in records),
            "matched_demo24_denominator_changed":False}
    write_json(OUT/"origin_investigation"/rid/"summary.json",answer)
    print(rid,"origin candidates",answer["n_valid_candidates"],flush=True)
    return answer


if __name__=="__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(run,["RXN_0000155302","RXN_0000079731","RXN_0000091050"]))
    write_json(OUT/"origin_investigation/summary.json",{"records":rows})
