"""Probe physical Lagrangian curvature along the observed rejected movement."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from run_demo24_round3 import ROOT, OLD, OUT, material, read, write_json
from pes2ts_core.integration.acp.gradient_backend import ORCAGradientBackend
from pes2ts_core.generation.planning.constrained_curvature import directional_curvature
from pes2ts_core.generation.planning.synchronized_path import align


def probe(rid):
    _, source, plan = material(rid)
    rejected = [a for a in source["attempts"] if not a["accepted"]]
    if not rejected:
        return
    attempt = rejected[-1]
    folder = OLD/rid/"endpoint/attempts"/attempt["attempt_id"]
    x = np.asarray(read(folder/"request.json")["geometry"])
    output = np.asarray(read(folder/"result.json")["coordinates"])
    evaluator = ORCAGradientBackend(charge=plan["charge"], multiplicity=plan["multiplicity"],
        elements=plan["elements"], folder=OUT/rid/"curvature_probe"/"evaluations")
    result = evaluator(x, "base")
    if not result["success"]:
        row = result
    else:
        row = directional_curvature(x, plan["drivers"], result["gradient_hartree_per_angstrom"],
            align(output, x)-x, evaluator)
    write_json(OUT/rid/"curvature_probe/result.json", row)
    print(rid, row, flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(probe, ["RXN_0000017762", "RXN_0000077619", "RXN_0000161724"]))
