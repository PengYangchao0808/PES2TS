"""Auditable ORCA EnGrad evaluations through ACP's existing SP interface."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from pes2ts_core.integration.acp.continuation_backend import ORCAContinuationBackend, write_json
from pes2ts_core.generation.planning.continuation import collision_reason
from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM, read_bound_engrad
from pes2ts_core.generation.planning.local_corrector import LocalCorrectorPolicy, correct_local
from pes2ts_core.generation.planning.synchronized_path import digest


class ORCAGradientBackend:
    def __init__(self, *, charge, multiplicity, elements, folder, method="GFN2-xTB",
                 basis="", timeout_seconds=120., nproc=2):
        engine = ORCAContinuationBackend(charge=charge, multiplicity=multiplicity,
                                         elements=elements, folder=folder)
        self.interface = engine.interface
        # Empty basis is intentional for xTB/composite methods. ACP's public
        # helpers otherwise fall back to the configured def2-TZVPP basis.
        self.interface.basis = basis
        self.interface.method = method
        self.interface.config["optimization_control"]["timeout"]["default_seconds"] = timeout_seconds
        self.interface.nproc = nproc
        self.charge, self.multiplicity = charge, multiplicity
        self.elements, self.folder = list(elements), Path(folder)
        self.method, self.basis = method, basis
        self.timeout_seconds, self.nproc = timeout_seconds, nproc

    def __call__(self, geometry, evaluation_id):
        request = {"geometry": np.asarray(geometry).tolist(), "elements": self.elements,
                   "method": self.method, "basis": self.basis, "charge": self.charge,
                   "multiplicity": self.multiplicity, "route_extras": ["EnGrad"],
                   "timeout_seconds": self.timeout_seconds, "nproc": self.nproc,
                   "scc_iterations": 2000, "schema_version": "pes2ts_physical_gradient_request_v1"}
        binding = digest(request)
        folder = self.folder/evaluation_id
        receipt = folder/"result.json"
        if receipt.exists():
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            if stored["request_sha256"] != binding:
                raise ValueError("CACHED_GRADIENT_INPUT_MISMATCH")
            return stored
        write_json(folder/"request.json", {**request, "request_sha256": binding})
        started = time.monotonic()
        extra = ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend'] if self.method == "GFN2-xTB" else []
        result = self.interface.single_point(np.asarray(geometry), self.elements,
            charge=self.charge, multiplicity=self.multiplicity, output_dir=folder,
            output_name="physical", method=self.method, basis=self.basis,
            route_extras=["EnGrad"], extra_blocks=extra, scf_convergence="tight")
        logfile = Path(result.log_file) if result.log_file else None
        log = logfile.read_text(errors="replace") if logfile and logfile.exists() else ""
        evidence = {"status": "missing"}
        finite_energy = result.energy is not None and np.isfinite(result.energy)
        if result.success and finite_energy and "ORCA TERMINATED NORMALLY" in log:
            # xTB uses a companion name; other methods use the ordinary .engrad.
            for name in ("physical.engrad", "physical.orca_XTB.engrad"):
                candidate = read_bound_engrad(folder/name, geometry, self.elements, result.energy)
                if candidate["status"] != "missing":
                    evidence = candidate
                    break
        success = evidence["status"] == "bound"
        failure = None
        if not success:
            from cccp.qc.interfaces.orca import classify_orca_failure
            failure = classify_orca_failure(logfile) if logfile and logfile.exists() else "BACKEND_UNAVAILABLE"
            if failure == "unknown":
                failure = "PHYSICAL_GRADIENT_"+evidence["status"].upper()
        record = {"success": success, "failure_class": failure, "request_sha256": binding,
                  "energy": float(result.energy) if finite_energy else None,
                  "gradient_evidence": evidence, "gradient_hartree_per_angstrom": None,
                  "duration_seconds": time.monotonic()-started, "log_file": str(logfile),
                  "input_file": str(result.output_file), "normal_termination": "ORCA TERMINATED NORMALLY" in log,
                  "log_sha256": digest(log), "method": self.method, "basis": self.basis}
        if success:
            record["gradient_hartree_per_angstrom"] = (
                np.asarray(evidence["gradient_hartree_per_bohr"])/BOHR_ANGSTROM).tolist()
        write_json(receipt, record)
        return record


class ORCALocalCorrector:
    def __init__(self, plan, folder, *, policy=None, warm_start=False):
        self.plan, self.folder = plan, Path(folder)
        self.policy = policy or LocalCorrectorPolicy()
        self.warm_start = warm_start
        self.inverse_curvature = None

    def on_accept(self, result):
        if self.warm_start and result.get("optimizer_state"):
            self.inverse_curvature = np.asarray(result["optimizer_state"]["inverse_curvature"])

    def model_inverse_curvature(self, guess):
        from itertools import combinations
        from pes2ts_core.generation.planning.continuation import _jacobian
        from pes2ts_core.generation.planning.synchronized_path import coordinate_jacobian, value
        x = np.asarray(guess, float)
        bonds = {tuple(sorted(c["atoms"])) for c in self.plan.get("common_bond_checks", [])}
        for d in self.plan["drivers"]:
            if value(x,"distance",d["atoms"]) < 2.:
                bonds.add(tuple(sorted(d["atoms"])))
        hessian = np.eye(x.size)*.01
        if bonds:
            b = _jacobian(x,[{"kind":"distance","atoms":pair} for pair in sorted(bonds)])
            hessian += .5*b.T@b
        neighbours = {i:set() for i in range(len(x))}
        for a,b in bonds:
            neighbours[a].add(b); neighbours[b].add(a)
        angles=[]
        for centre,ns in neighbours.items():
            for a,b in combinations(sorted(ns),2):
                atoms=[a,centre,b]
                angle=value(x,"angle",atoms)
                if 5. < angle < 175.:
                    angles.append({"kind":"angle","atoms":atoms})
        if angles:
            a=coordinate_jacobian(x,angles)
            hessian += .15*a.T@a
        return np.linalg.inv(hessian)

    def __call__(self, guess, targets, attempt_id):
        folder = self.folder/attempt_id
        initial_curvature = self.inverse_curvature
        if self.warm_start and initial_curvature is None:
            initial_curvature = self.model_inverse_curvature(guess)
        request = {"guess": np.asarray(guess).tolist(), "targets": targets,
                   "plan_sha256": self.plan["content_sha256"], "policy": asdict(self.policy),
                   "optimizer": "projected_bfgs_fixed_neighbourhood_v1"}
        if self.warm_start:
            request["curvature_initialization"] = "valence_model_then_accepted_frame_bfgs"
            request["inverse_curvature_sha256"] = digest(initial_curvature.tolist())
        binding = digest(request)
        receipt = folder/"result.json"
        if receipt.exists():
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            if stored["request_sha256"] != binding:
                raise ValueError("CACHED_LOCAL_CORRECTOR_INPUT_MISMATCH")
            return stored
        write_json(folder/"request.json", {**request, "request_sha256": binding})
        evaluator = ORCAGradientBackend(charge=self.plan["charge"], multiplicity=self.plan["multiplicity"],
            elements=self.plan["elements"], folder=folder/"physical_evaluations")
        def check(x):
            from pes2ts_core.generation.planning.structure_checks import structure_issues
            issues = structure_issues(x, self.plan)
            return issues[0]["reason"] if issues else collision_reason(x, self.plan.get("collision_checks", []))
        result = correct_local(guess, self.plan["drivers"]+self.plan.get("guards", []), targets,
            self.plan["masses"], evaluator, policy=self.policy, geometry_check=check,
            initial_inverse_curvature=initial_curvature,
            save=lambda state: write_json(folder/"local_checkpoint.json", state))
        result["request_sha256"] = binding
        write_json(receipt, result)
        return result
