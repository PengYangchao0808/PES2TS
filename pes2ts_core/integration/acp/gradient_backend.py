"""Auditable ORCA EnGrad evaluations through the ACP ``OrcaGradient`` CLI.

ADR-0002 X4′-B: this backend no longer imports the legacy in-process QC
interface library.  Every gradient acquisition is executed by ACP
(``acp run OrcaGradient``) through
:mod:`pes2ts_core.integration.acp.orca_gradient_transport`, which owns the
claim/receipt/log/timeout/manifest-v2 discipline.  The public call shape of
:class:`ORCAGradientBackend` / :class:`ORCALocalCorrector` is preserved so
consumers need minimal change; ACP wiring (checkout root, interpreter, config)
is resolved from constructor arguments, an optional ``acp_wiring`` mapping,
then the ``PES2TS_ACP_ROOT`` / ``PES2TS_ACP_PYTHON`` / ``PES2TS_ACP_CONFIG``
environment variables.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from pes2ts_core.generation.planning.continuation import collision_reason
from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM
from pes2ts_core.generation.planning.local_corrector import LocalCorrectorPolicy, correct_local
from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.integration.acp.orca_gradient_request import (
    build_gradient_request,
    request_sha256,
)
from pes2ts_core.integration.acp.orca_gradient_transport import (
    parse_gradient_product_evidence,
    run_orca_gradient_attempt,
)

#: GFN2-xTB ORCA input blocks used by the pre-migration local backend.
_XTB_EXTRA_BLOCKS = ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend']
#: Portable identifier pattern shared by the ACP transports.
_PORTABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def write_json(path, document):
    """Atomically write *document* as indented UTF-8 JSON (no NaN/Inf)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    temp.replace(path)


def resolve_acp_wiring(*, acp_root=None, acp_python=None, acp_config=None,
                       acp_wiring=None):
    """Resolve ACP checkout/interpreter/config for the gradient transport.

    Precedence: explicit arguments → ``acp_wiring`` mapping (the ``acp`` block
    of a PES2TS config) → ``PES2TS_ACP_*`` environment variables.  Returns a
    ``(acp_root, acp_python, acp_config)`` tuple; ``acp_root`` may be ``None``
    here and is only required when an attempt actually executes.
    """
    wiring = dict(acp_wiring or {})
    root = acp_root if acp_root is not None else wiring.get("root")
    python = acp_python if acp_python is not None else wiring.get("python")
    config = acp_config if acp_config is not None else wiring.get("config_path")
    if root is None:
        root = os.environ.get("PES2TS_ACP_ROOT") or None
    if python is None:
        python = os.environ.get("PES2TS_ACP_PYTHON") or None
    if config is None:
        config = os.environ.get("PES2TS_ACP_CONFIG") or None
    if root in ("", "null"):
        root = None
    if python in ("", "null"):
        python = None
    if config in ("", "null"):
        config = None
    return root, python, config


def _portable_attempt_id(evaluation_id) -> str:
    value = str(evaluation_id)
    if _PORTABLE_ID.fullmatch(value):
        return value
    return "eval-" + digest(value)[:32]


def _classify_acp_gradient_failure(*, timed_out: bool, error: str | None,
                                   acp_log: str, orca_logs: list[str],
                                   evidence_status: str) -> str:
    """Map an ACP attempt failure onto the historical failure vocabulary.

    The pre-migration backend delegated failure classification to the legacy
    in-process QC library; ACP owns the ORCA process now, so classification is
    derived from the ACP receipt error, the captured CLI log and any ORCA logs
    ACP left under ``WORK``.  Unknown failures keep the honest typed suffixes
    instead of a fabricated legacy class.
    """
    if timed_out:
        return "TIMEOUT"
    haystack = "\n".join([acp_log or "", error or "", *orca_logs])
    if "ORCA TERMINATED NORMALLY" not in haystack:
        return "PROCESS_FAILED_WITHOUT_NORMAL_TERMINATION"
    markers = (
        ("SCF NOT CONVERGED", "SCF_NOT_CONVERGED"),
        ("NOT CONVERGED", "OPTIMIZATION_NOT_CONVERGED"),
        ("IMAGINARY FREQUENCY", "IMAGINARY_FREQUENCY"),
        ("ORCA TERMINATED NORMALLY", None),
    )
    for marker, failure in markers:
        if failure is not None and marker in haystack:
            return failure
    if evidence_status != "bound":
        return "PHYSICAL_GRADIENT_" + str(evidence_status).upper()
    return "GRADIENT_FAILED"


def _collect_orca_logs(attempt_dir: Path) -> list[str]:
    """Return decoded ORCA log/out texts ACP left under the attempt WORK tree."""
    texts: list[str] = []
    work = attempt_dir / "WORK"
    if not work.is_dir():
        return texts
    pes2ts_meta = work / "pes2ts"
    for path in sorted(work.rglob("*")):
        if not path.is_file() or path.is_relative_to(pes2ts_meta):
            continue
        if path.suffix.lower() not in {".log", ".out", ".stdout"} and "orca" not in path.name.lower():
            continue
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return texts


class ORCAGradientBackend:
    """Physical ORCA gradient evaluator backed by ACP ``OrcaGradient``.

    Call shape (unchanged from the pre-migration backend): construct once per
    folder/method, then ``backend(geometry, evaluation_id) -> record``.  The
    record keeps the historical keys (``success``, ``failure_class``,
    ``request_sha256``, ``energy``, ``gradient_evidence``,
    ``gradient_hartree_per_angstrom``, ``duration_seconds``, ``log_file``,
    ``input_file``, ``normal_termination``, ``log_sha256``, ``method``,
    ``basis``) so :func:`correct_local` and experiment scripts keep working.
    """

    def __init__(self, *, charge, multiplicity, elements, folder, method="GFN2-xTB",
                 basis="", timeout_seconds=120., nproc=2,
                 acp_root=None, acp_python=None, acp_config=None, acp_wiring=None):
        self.charge, self.multiplicity = charge, multiplicity
        self.elements, self.folder = list(elements), Path(folder)
        self.method, self.basis = method, basis
        self.timeout_seconds, self.nproc = timeout_seconds, nproc
        self.acp_root, self.acp_python, self.acp_config = resolve_acp_wiring(
            acp_root=acp_root, acp_python=acp_python, acp_config=acp_config,
            acp_wiring=acp_wiring)

    def _extra_blocks(self):
        return list(_XTB_EXTRA_BLOCKS) if self.method == "GFN2-xTB" else []

    def __call__(self, geometry, evaluation_id):
        geometry_rows = np.asarray(geometry, dtype=float).tolist()
        request = build_gradient_request(
            geometry=geometry_rows, elements=self.elements, method=self.method,
            basis=self.basis, charge=self.charge, multiplicity=self.multiplicity,
            route_extras=["EnGrad"], timeout_seconds=self.timeout_seconds,
            nproc=self.nproc, extra_blocks=self._extra_blocks(),
            scf_convergence="tight", output_name="grad")
        binding = digest(request)
        folder = self.folder / str(evaluation_id)
        receipt = folder / "result.json"
        if receipt.exists():
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            if stored["request_sha256"] != binding:
                raise ValueError("CACHED_GRADIENT_INPUT_MISMATCH")
            return stored
        if self.acp_root is None:
            raise ACPCLIError(
                "ORCAGradientBackend requires an ACP checkout root "
                "(constructor acp_root / acp_wiring['root'] / PES2TS_ACP_ROOT)"
            )
        write_json(folder / "request.json", {**request, "request_sha256": binding,
                                             "acp_request_sha256": request_sha256(request)})
        started = time.monotonic()
        attempt_id = _portable_attempt_id(evaluation_id)
        attempt = run_orca_gradient_attempt(
            acp_root=self.acp_root, python_executable=self.acp_python,
            acp_config_path=self.acp_config, request=request, output_root=folder,
            execution_id="pes2ts-orcagradient", attempt_id=attempt_id,
            timeout_seconds=float(self.timeout_seconds) if self.timeout_seconds else 120.)
        acp_log = ""
        if attempt.log_ref:
            log_candidate = folder / attempt.log_ref
            if log_candidate.is_file():
                acp_log = log_candidate.read_text(encoding="utf-8", errors="replace")
        orca_logs = _collect_orca_logs(folder)
        combined_log = "\n".join([acp_log, *orca_logs])
        product = attempt.gradient_product
        evidence = parse_gradient_product_evidence(
            product, gradient_path=attempt.gradient_path, elements=self.elements,
            geometry=geometry_rows,
            energy=float(product["energy_hartree"]) if product and isinstance(product.get("energy_hartree"), (int, float)) else None)
        success = bool(attempt.status == "completed" and evidence["status"] == "bound")
        failure = None
        if not success:
            failure = _classify_acp_gradient_failure(
                timed_out=attempt.timed_out, error=attempt.error, acp_log=acp_log,
                orca_logs=orca_logs, evidence_status=evidence["status"])
        energy = None
        if product is not None and isinstance(product.get("energy_hartree"), (int, float)):
            if math.isfinite(float(product["energy_hartree"])):
                energy = float(product["energy_hartree"])
        normal_termination = "ORCA TERMINATED NORMALLY" in combined_log or (
            success and not orca_logs)
        record = {"success": success, "failure_class": failure,
                  "request_sha256": binding,
                  "acp_request_sha256": attempt.request_sha256,
                  "energy": energy, "gradient_evidence": evidence,
                  "gradient_hartree_per_angstrom": None,
                  "duration_seconds": time.monotonic() - started,
                  "log_file": str(folder / attempt.log_ref) if attempt.log_ref else None,
                  "input_file": attempt.gradient_path,
                  "normal_termination": normal_termination,
                  "log_sha256": digest(combined_log), "method": self.method,
                  "basis": self.basis,
                  "acp": {"workflow": "OrcaGradient", "status": attempt.status,
                          "attempt_id": attempt.attempt_id,
                          "manifest_path": attempt.manifest_path,
                          "gradient_path": attempt.gradient_path,
                          "timed_out": attempt.timed_out,
                          "returncode": attempt.returncode,
                          "wall_seconds": attempt.wall_seconds,
                          "reused": attempt.reused}}
        if success:
            record["gradient_hartree_per_angstrom"] = (
                np.asarray(evidence["gradient_hartree_per_bohr"]) / BOHR_ANGSTROM).tolist()
        write_json(receipt, record)
        return record


class ORCALocalCorrector:
    """Local projected-BFGS corrector driven by the ACP gradient backend."""

    def __init__(self, plan, folder, *, policy=None, warm_start=False,
                 acp_root=None, acp_python=None, acp_config=None, acp_wiring=None):
        self.plan, self.folder = plan, Path(folder)
        self.policy = policy or LocalCorrectorPolicy()
        self.warm_start = warm_start
        self.inverse_curvature = None
        self.acp_root, self.acp_python, self.acp_config = resolve_acp_wiring(
            acp_root=acp_root, acp_python=acp_python, acp_config=acp_config,
            acp_wiring=acp_wiring)

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
            neighbours[a].add(b)
            neighbours[b].add(a)
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
        evaluator = ORCAGradientBackend(
            charge=self.plan["charge"], multiplicity=self.plan["multiplicity"],
            elements=self.plan["elements"], folder=folder/"physical_evaluations",
            acp_root=self.acp_root, acp_python=self.acp_python,
            acp_config=self.acp_config)
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


__all__ = [
    "ORCAGradientBackend",
    "ORCALocalCorrector",
    "resolve_acp_wiring",
    "write_json",
]
