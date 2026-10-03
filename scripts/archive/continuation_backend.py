"""Single-point continuation calls through the existing ACP ORCA interface.

No ACP service restart, external source edits, or private subprocess layer.
Every QC call is content-bound and cached in its own attempt directory.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from pes2ts_core.generation.planning.synchronized_path import digest


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


class ORCAContinuationBackend:
    def __init__(self, *, charge, multiplicity, elements, folder, coordinates=(),
                 timeout_seconds=120., trust_radius=None, max_step=None, opt_level="normal"):
        from cccp.config import load_config
        from cccp.qc.interfaces.orca import ORCAInterface
        cfg = load_config()
        cfg["executables"]["orca"]["nproc"] = 2
        cfg["resources"]["nproc"] = 2
        cfg.setdefault("optimization_control", {}).setdefault("timeout", {})["default_seconds"] = timeout_seconds
        if trust_radius is not None or max_step is not None:
            class ControlledInterface(ORCAInterface):
                def _write_input(self, *args, **kwargs):
                    if trust_radius is not None:
                        kwargs["trust_radius"] = trust_radius
                    if max_step is not None:
                        kwargs["geom_extra_lines"] = list(kwargs.get("geom_extra_lines") or [])+[f"  MaxStep {max_step:g}"]
                    return super()._write_input(*args, **kwargs)
            self.interface = ControlledInterface(cfg)
        else:
            self.interface = ORCAInterface(cfg)
        self.charge = charge
        self.multiplicity = multiplicity
        self.elements = elements
        self.folder = Path(folder)
        self.coordinates = list(coordinates)
        self.optimizer_controls = {"trust_radius_au": trust_radius, "max_step_au": max_step,
                                   "opt_level": opt_level, "timeout_seconds": timeout_seconds}

    def call(self, x, targets, name, *, free=False):
        from cccp.qc.interfaces.constraints import AngleConstraint, DihedralConstraint, DistanceConstraint
        request = {"geometry": np.asarray(x).tolist(), "elements": self.elements, "targets": targets,
                   "coordinates": self.coordinates, "charge": self.charge, "multiplicity": self.multiplicity,
                   "method": "GFN2-xTB", "geom_maxiter": 250, "scc_iterations": 2000, "free": free}
        if self.optimizer_controls != {"trust_radius_au": None, "max_step_au": None,
                                      "opt_level": "normal", "timeout_seconds": 120.}:
            request["optimizer_controls"] = self.optimizer_controls
        binding = digest(request)
        folder = self.folder / name
        receipt = folder / "result.json"
        if receipt.exists():
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            if stored["request_sha256"] != binding:
                raise ValueError("CACHED_ATTEMPT_INPUT_MISMATCH")
            return stored
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "request.json", {**request, "request_sha256": binding})
        kwargs = {"charge": self.charge, "multiplicity": self.multiplicity, "output_dir": folder,
                  "output_name": "orca", "method": "GFN2-xTB", "basis": "", "geom_maxiter": 250,
                  "opt_level": self.optimizer_controls["opt_level"],
                  "extra_blocks": ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend']}
        started = time.monotonic()
        if free:
            result = self.interface.optimize(np.asarray(x), self.elements, **kwargs)
        else:
            constructors = {"distance": DistanceConstraint, "angle": AngleConstraint, "dihedral": DihedralConstraint}
            constraints = [constructors[d["kind"]](tuple(d["atoms"]), float(t))
                           for d, t in zip(self.coordinates, targets)]
            result = self.interface.constrained_optimize(np.asarray(x), self.elements, constraints, **kwargs)
        elapsed = time.monotonic() - started
        logfile = Path(result.log_file) if result.log_file else None
        log = logfile.read_text(errors="replace") if logfile and logfile.exists() else ""
        terminated = "ORCA TERMINATED NORMALLY" in log
        optimized = "THE OPTIMIZATION HAS CONVERGED" in log
        finite_energy = result.energy is not None and np.isfinite(result.energy)
        valid_symbols = list(result.symbols or ()) == list(self.elements)
        ok = bool(result.success and result.converged and terminated and optimized and finite_energy and valid_symbols)
        failure = None
        if not ok:
            from cccp.qc.interfaces.orca import classify_orca_failure
            failure = classify_orca_failure(logfile) if logfile and logfile.exists() else "BACKEND_UNAVAILABLE"
            if failure == "unknown":
                failure = "OPT_FAILED"
        record = {"request_sha256": binding, "success": ok, "converged": ok,
                  "energy": float(result.energy) if finite_energy else None,
                  "coordinates": np.asarray(result.coordinates).tolist() if result.coordinates is not None and np.isfinite(result.coordinates).all() else None,
                  "duration_seconds": elapsed, "failure_class": failure,
                  "normal_termination": terminated, "optimization_marker": optimized,
                  "log_file": str(logfile), "log_sha256": digest(log),
                  "message": result.error_message, "input_file": str(result.output_file),
                  "method": "GFN2-xTB", "nproc": 2, "scc_iterations": 2000,
                  "physical_gradient": None, "physical_gradient_status": "not_collected"}
        if ok and record["coordinates"] is not None:
            from pes2ts_core.generation.planning.gradient_evidence import read_bound_engrad
            evidence = read_bound_engrad(folder/"orca.orca_XTB.engrad", record["coordinates"], self.elements, record["energy"])
            record["gradient_evidence"] = evidence
            record["physical_gradient_status"] = evidence["status"]
            record["physical_gradient"] = evidence.get("gradient_hartree_per_bohr")
        write_json(receipt, record)
        return record

    def __call__(self, x, targets, attempt_id):
        return self.call(x, targets, attempt_id)
