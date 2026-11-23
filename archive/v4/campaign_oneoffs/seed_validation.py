"""Independent, staged physical validation of hash-bound continuation seeds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from continuation_backend import write_json
from gradient_backend_cccp import ORCAGradientBackend
from pes2ts_core.generation.planning.candidate_selection import bind_candidate
from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM
from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.generation.planning.ts_checks import assess_mode, mapped_endpoint_identity


@dataclass(frozen=True)
class ValidationBudgets:
    optts_seconds: float = 600.
    frequency_seconds: float = 600.
    irc_seconds: float = 600.
    endpoint_seconds: float = 300.
    optts_iterations: int = 100
    irc_iterations: int = 80

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("INVALID_VALIDATION_BUDGET")


def validate_seed(plan, snapshot, result, candidate, folder, *, method="HF-3c", basis="", budgets=None):
    """No complete-interval precondition; only accepted-frame provenance matters."""
    from cccp.qc.interfaces.orca_ts import parse_ts_frequency_map, parse_ts_mode_vectors
    budgets = budgets or ValidationBudgets()
    candidate = bind_candidate(candidate, result, plan)
    x = np.asarray(result["frames"][candidate["frame_index"]]["geometry"])
    folder = Path(folder)
    request = {"candidate": candidate, "method": method, "basis": basis,
               "budgets": asdict(budgets), "validation_version": "independent_seed_validation_v1",
               "snapshot_sha256": digest(snapshot), "frequency_threshold_cm1": -30.}
    request_hash = digest(request)
    receipt = folder/"result.json"
    if receipt.exists():
        stored = json.loads(receipt.read_text(encoding="utf-8"))
        if stored["request_sha256"] != request_hash:
            raise ValueError("CACHED_VALIDATION_INPUT_MISMATCH")
        return stored
    write_json(folder/"request.json", {**request, "request_sha256": request_hash})
    record = {"schema_version": "pes2ts_independent_seed_validation_v1", "request_sha256": request_hash,
              "reaction_id": plan["reaction_id"], "candidate": candidate, "method": method, "basis": basis,
              "source_path_complete": result["completed_interval"], "stages": {},
              "target_ts_verified": False, "reaction_connection_verified": False,
              "reference_seed_used": plan.get("reference_seed_used", False)}
    started = time.monotonic()
    extra = ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend'] if method == "GFN2-xTB" else []

    def engine(seconds, directory):
        return ORCAGradientBackend(charge=plan["charge"], multiplicity=plan["multiplicity"],
            elements=plan["elements"], folder=directory, method=method, basis=basis, timeout_seconds=seconds)

    def failure(log, elapsed, limit):
        if elapsed >= .95*limit and "ORCA TERMINATED NORMALLY" not in log:
            return "TIMEOUT"
        if "ORCA TERMINATED NORMALLY" not in log:
            return "PROCESS_FAILED_WITHOUT_NORMAL_TERMINATION"
        return "OPTIMIZATION_NOT_CONVERGED"

    def stage(name, run):
        path = folder/name/"stage.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["validation_request_sha256"] != request_hash:
                raise ValueError("CACHED_VALIDATION_STAGE_MISMATCH")
        else:
            data = run()
            data["validation_request_sha256"] = request_hash
            write_json(path, data)
        record["stages"][name] = data
        write_json(folder/"checkpoint.json", record)
        return data

    def finish(status):
        record["status"] = status
        record["measured_wall_seconds"] = time.monotonic()-started
        record["stage_duration_seconds"] = sum(s.get("duration_seconds",0) for s in record["stages"].values())
        write_json(receipt, record)
        return record

    def optimize_ts():
        e = engine(budgets.optts_seconds, folder/"optts")
        begin = time.monotonic()
        answer = e.interface.transition_state_opt(x, plan["elements"], charge=plan["charge"],
            multiplicity=plan["multiplicity"], output_dir=folder/"optts", output_name="ts",
            method=method, basis=basis, calculate_frequencies=False, initial_hessian="calculate",
            recalc_hess=5, trust_radius=.1, opt_level="tight", geom_maxiter=budgets.optts_iterations,
            extra_blocks=extra, output_callback=lambda line: None)
        elapsed = time.monotonic()-begin
        log = Path(answer.log_file).read_text(errors="replace") if answer.log_file and Path(answer.log_file).exists() else ""
        symbols_ok = list(answer.symbols or []) == list(plan["elements"])
        ok = bool(answer.success and answer.converged and symbols_ok and "ORCA TERMINATED NORMALLY" in log
                  and "THE OPTIMIZATION HAS CONVERGED" in log and answer.coordinates is not None)
        return {"success": ok, "failure_class": None if ok else failure(log,elapsed,budgets.optts_seconds),
                "duration_seconds": elapsed, "message": answer.error_message,
                "geometry": np.asarray(answer.coordinates).tolist() if answer.coordinates is not None else None,
                "energy_hartree": answer.energy_hartree, "log_file": str(answer.log_file),
                "log_sha256": digest(log), "normal_termination": "ORCA TERMINATED NORMALLY" in log,
                "independent_final_frequency_included": False}
    opt = stage("optts", optimize_ts)
    if not opt["success"]:
        return finish("RUN_FAILED:OPTTS:"+opt["failure_class"])
    tx = np.asarray(opt["geometry"])

    def physical_gradient():
        e = engine(120., folder/"ts_gradient")
        answer = e(tx,"bound")
        if answer["success"]:
            g = np.asarray(answer["gradient_hartree_per_angstrom"])*BOHR_ANGSTROM
            answer["full_gradient_rms_hartree_per_bohr"] = float(np.sqrt(np.mean(g*g)))
            answer["full_gradient_max_hartree_per_bohr"] = float(np.abs(g).max())
        return answer
    force = stage("ts_gradient", physical_gradient)
    if not force["success"]:
        return finish("RUN_FAILED:TS_GRADIENT")
    if force["full_gradient_rms_hartree_per_bohr"] > .0001 or force["full_gradient_max_hartree_per_bohr"] > .0003:
        return finish("NOT_STATIONARY_FULL_SPACE")

    def frequencies():
        e = engine(budgets.frequency_seconds, folder/"frequency")
        begin = time.monotonic()
        answer = e.interface.frequency(tx, plan["elements"], charge=plan["charge"], multiplicity=plan["multiplicity"],
            output_dir=folder/"frequency", output_name="freq", method=method, basis=basis, extra_blocks=extra)
        elapsed = time.monotonic()-begin
        log = Path(answer.log_file).read_text(errors="replace") if answer.log_file and Path(answer.log_file).exists() else ""
        freq = parse_ts_frequency_map(log)
        modes = parse_ts_mode_vectors(log)
        return {"success": bool(answer.success and "ORCA TERMINATED NORMALLY" in log and freq),
                "duration_seconds": elapsed, "log_file": str(answer.log_file), "log_sha256": digest(log),
                "frequency_geometry_sha256": digest(tx.tolist()),
                "mode_assessment": assess_mode(tx, plan["drivers"], freq, modes),
                "hessian_file": str(folder/"frequency/freq.hess")}
    freq = stage("frequency", frequencies)
    if not freq["success"]:
        return finish("RUN_FAILED:FREQUENCY")
    assessment = freq["mode_assessment"]
    if not assessment["single_meaningful_imaginary"]:
        return finish("NOT_FIRST_ORDER_SADDLE")
    if not assessment["target_mode_screen_passed"]:
        return finish("NON_TARGET_MODE_OR_MODE_MISSING")
    record["first_order_saddle_verified"] = True

    def irc_run():
        e = engine(budgets.irc_seconds, folder/"irc")
        begin = time.monotonic()
        answer = e.interface.irc(tx, plan["elements"], charge=plan["charge"], multiplicity=plan["multiplicity"],
            output_dir=folder/"irc", output_name="irc", method=method, basis=basis, direction="both",
            max_iter=budgets.irc_iterations, hess_file=Path(freq["hessian_file"]), output_callback=lambda line:None)
        log = Path(answer.log_file).read_text(errors="replace") if answer.log_file and Path(answer.log_file).exists() else ""
        return {"success": bool(answer.success and "ORCA TERMINATED NORMALLY" in log and len(answer.final_geometries)==2),
                "duration_seconds": time.monotonic()-begin, "message": answer.error_message,
                "log_file": str(answer.log_file), "log_sha256": digest(log),
                "final_geometries": {k:np.asarray(v).tolist() for k,v in answer.final_geometries.items()},
                "trajectory_files": {k:str(v) for k,v in (answer.trajectory_files or {}).items()}}
    irc = stage("irc", irc_run)
    if not irc["success"]:
        return finish("RUN_FAILED:IRC")
    assignments = []
    for direction, endpoint in irc["final_geometries"].items():
        def release(endpoint=endpoint,direction=direction):
            e = engine(budgets.endpoint_seconds,folder/f"endpoint_{direction}")
            begin = time.monotonic()
            answer = e.interface.optimize(np.asarray(endpoint),plan["elements"],charge=plan["charge"],
                multiplicity=plan["multiplicity"],output_dir=folder/f"endpoint_{direction}",output_name="minimum",
                method=method,basis=basis,opt_level="tight",geom_maxiter=200,extra_blocks=extra)
            log = Path(answer.log_file).read_text(errors="replace") if answer.log_file and Path(answer.log_file).exists() else ""
            ok = bool(answer.success and answer.converged and "ORCA TERMINATED NORMALLY" in log
                      and "THE OPTIMIZATION HAS CONVERGED" in log)
            return {"success":ok,"duration_seconds":time.monotonic()-begin,"log_file":str(answer.log_file),
                    "log_sha256":digest(log),"geometry":np.asarray(answer.coordinates).tolist() if answer.coordinates is not None else None}
        endpoint_opt = stage(f"endpoint_{direction}",release)
        if not endpoint_opt["success"]:
            return finish("RUN_FAILED:ENDPOINT_OPT")
        ex = np.asarray(endpoint_opt["geometry"])
        identities = {s:mapped_endpoint_identity(ex,snapshot,s) for s in ("R","P")}
        record.setdefault("endpoint_identities",{})[direction] = identities
        matches = [s for s,v in identities.items() if v["matched"]]
        assignments.append(matches)
        def endpoint_freq(ex=ex,direction=direction):
            e = engine(budgets.endpoint_seconds,folder/f"endpoint_{direction}_freq")
            begin = time.monotonic()
            answer=e.interface.frequency(ex,plan["elements"],charge=plan["charge"],multiplicity=plan["multiplicity"],
                output_dir=folder/f"endpoint_{direction}_freq",output_name="minimum_freq",method=method,basis=basis,extra_blocks=extra)
            log=Path(answer.log_file).read_text(errors="replace") if answer.log_file and Path(answer.log_file).exists() else ""
            frequencies=list(parse_ts_frequency_map(log).values())
            return {"success":bool(answer.success and "ORCA TERMINATED NORMALLY" in log and frequencies),
                    "duration_seconds":time.monotonic()-begin,"frequencies_cm1":frequencies,
                    "minimum_frequency_screen_passed":bool(frequencies and min(frequencies) >= -30.),
                    "log_file":str(answer.log_file),"log_sha256":digest(log)}
        ef=stage(f"endpoint_{direction}_freq",endpoint_freq)
        if not ef["success"]:
            return finish("RUN_FAILED:ENDPOINT_FREQUENCY")
        if not ef["minimum_frequency_screen_passed"]:
            return finish("IRC_ENDPOINT_NOT_MINIMUM")
    connected = len(assignments)==2 and (("R" in assignments[0] and "P" in assignments[1])
                                       or ("P" in assignments[0] and "R" in assignments[1]))
    record["reaction_connection_verified"] = connected
    record["target_ts_verified"] = connected
    return finish("TARGET_TS_IRC_VERIFIED" if connected else "NON_TARGET_IRC_CONNECTION")
