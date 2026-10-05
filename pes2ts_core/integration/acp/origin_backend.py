"""ACP-backed unconstrained origin preparation (G2-AB2 WP-3).

Production adapter behind ``prepare_origin``'s injected ``evaluate_free``:
the free optimization runs as an ACP ``BatchOptimize`` item with a NON-TS
role (ACP-G2-04 pointwise capability), executed through
``python -m acp.cli run BatchOptimize ...`` as an argv-only child process
under the same claim/receipt/log/timeout discipline as the other ACP CLI
adapters (ADR-0002; this module sits on ``SUBPROCESS_IMPORT_ALLOWLIST`` for
the ``subprocess`` import only — every truth check still applies).

Fail-closed capability probing (constitution R3 / plan §1.4): before the
first evaluation the backend verifies the ACP wiring is resolvable and the
ACP CLI accepts the ``BatchOptimize`` workflow.  A failed probe raises
:class:`OriginBackendUnavailable` with a typed needs list — the caller
records a ``blocked`` terminal, and there is NEVER a silent fallback to a
local optimizer.

Evaluation identity is content-addressed (G2-AB1 WP-1): the receipt lives at
``<folder>/<trial>/eval-<content16>/result.json`` bound to the request
sha256; identical content replays the cached receipt (zero QC cost), and a
same-address/different-binding collision is a typed
``IDENTITY_CONFLICT`` — never recomputed in place.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np

from pes2ts_core.generation.planning.synchronized_path import digest
from pes2ts_core.integration.acp.cli_backend import ACPCLIError
from pes2ts_core.integration.acp.gradient_backend import (
    resolve_acp_wiring,
    write_json,
)
from pes2ts_core.integration.acp.stage_results import _verified_manifest

#: Typed fail-closed unavailability code (campaign maps it to ``blocked``).
ORIGIN_BACKEND_UNAVAILABLE = "ORIGIN_BACKEND_UNAVAILABLE"

#: Non-TS batch item id used for the free-optimization structure.
ORIGIN_ITEM_ID = "pes2ts_origin"

#: ACP batch profile for a plain unconstrained optimization (no TS, no freq).
ORIGIN_PROFILE = "opt_only"

_WORKFLOW = "BatchOptimize"
_PORTABLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class OriginBackendUnavailable(RuntimeError):
    """Fail-closed capability probe result (typed, carries a needs list)."""

    def __init__(self, needs: list[str], evidence: dict[str, Any] | None = None):
        self.needs = list(needs)
        self.evidence = evidence or {}
        super().__init__(f"{ORIGIN_BACKEND_UNAVAILABLE}:{';'.join(self.needs)}")


@dataclass(frozen=True)
class OriginEvaluationOutcome:
    """Shape of one ``evaluate_free`` record (consumed by ``prepare_origin``)."""

    success: bool
    failure_class: str | None
    coordinates: list[list[float]] | None
    energy: float | None
    duration_seconds: float
    acp_receipt_ref: str | None
    gradient_hartree_per_angstrom: list[list[float]] | None = None
    acp: dict[str, Any] | None = None


def _xyz_text(elements: list[str], geometry: list[list[float]]) -> str:
    rows = [str(len(elements)), "PES2TS origin free optimization"]
    for element, point in zip(elements, geometry, strict=True):
        rows.append(f"{element} {point[0]:.12f} {point[1]:.12f} {point[2]:.12f}")
    return "\n".join(rows) + "\n"


def _parse_xyz(path: Path, expected_elements: list[str]) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2 or not lines[0].strip().isdigit():
        raise ACPCLIError(f"ACP origin product is not an XYZ file: {path.name}")
    n_atoms = int(lines[0].strip())
    body = [line.split() for line in lines[2:2 + n_atoms] if line.strip()]
    if len(body) != n_atoms:
        raise ACPCLIError(f"ACP origin product has a truncated geometry: {path.name}")
    elements = [row[0] for row in body]
    geometry = [[float(row[1]), float(row[2]), float(row[3])] for row in body]
    if elements != list(expected_elements):
        raise ACPCLIError("ACP origin product element order differs from the input")
    return elements, geometry


def _read_energy_product(folder: Path, item_id: str) -> float | None:
    """Best-effort energy read; a missing product is an honest None.

    ACP ``BatchOptimize`` writes no per-item ``batch_*energy*.json`` for
    ``opt_only`` (audit of ACP_V1_20260811: RESULT JSON writers emit only
    ``result_manifest.json``, ``state_comparison.json`` and
    ``batch_provenance.json``).  The optimized energy is the last finite
    ``cycles[].energy_hartree`` of the optimization trajectory the engine
    copies to ``RESULT/trajectories[/<item_id>]/optimization.json``.  The
    per-item report glob stays as a compatibility fallback.
    """
    for candidate in (folder / "RESULT").glob(f"batch_{item_id}*energy*.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("energy_hartree", "final_energy_hartree", "energy"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value))):
                return float(value)
    for candidate in (
            folder / "RESULT" / "trajectories" / item_id / "optimization.json",
            folder / "RESULT" / "trajectories" / "optimization.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cycles = payload.get("cycles") if isinstance(payload, dict) else None
        if not isinstance(cycles, list):
            continue
        for cycle in reversed(cycles):
            value = cycle.get("energy_hartree") if isinstance(cycle, dict) else None
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value))):
                return float(value)
    return None


def _collect_attempt_logs(folder: Path, cli_log: Path) -> str:
    """Merged ACP CLI stdout + engine logs left under the attempt tree."""
    texts: list[str] = []
    if cli_log.is_file():
        texts.append(cli_log.read_text(encoding="utf-8", errors="replace"))
    pes2ts_meta = folder / "WORK" / "pes2ts"
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.is_relative_to(pes2ts_meta):
            continue
        if path.suffix.lower() not in {".log", ".out"}:
            continue
        try:
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(texts)


class ORCAOriginPreparation:
    """Unconstrained optimization backend driven by the ACP CLI."""

    def __init__(self, *, elements, charge, multiplicity, folder,
                 method: str = "GFN2-xTB", basis: str = "",
                 timeout_seconds: float = 600., nproc: int = 2,
                 acp_root=None, acp_python=None, acp_config=None,
                 acp_wiring: dict[str, Any] | None = None,
                 capability_probe: bool = True,
                 operation: str = "origin_free_optimization"):
        self.elements = [str(e) for e in elements]
        self.charge, self.multiplicity = int(charge), int(multiplicity)
        self.folder = Path(folder)
        self.method, self.basis = method, basis
        self.timeout_seconds = float(timeout_seconds)
        self.nproc = int(nproc)
        self.operation = operation
        self.capability_probe = bool(capability_probe)
        self.acp_root, self.acp_python, self.acp_config = resolve_acp_wiring(
            acp_root=acp_root, acp_python=acp_python, acp_config=acp_config,
            acp_wiring=acp_wiring)
        self._probe: dict[str, Any] | None = None

    # -- capability probe --------------------------------------------------

    def probe_capability(self, *, force: bool = False) -> dict[str, Any]:
        """Fail-closed wiring + workflow probe; never optimistic (R3)."""
        if self._probe is not None and not force:
            return self._probe
        needs: list[str] = []
        evidence: dict[str, Any] = {"workflow": _WORKFLOW, "profile": ORIGIN_PROFILE,
                                    "acp_root": self.acp_root,
                                    "python_executable": self.acp_python}
        if self.acp_root is None:
            needs.append("acp.root / PES2TS_ACP_ROOT must point at the ACP checkout")
        if self.acp_python is None:
            needs.append("acp.python / PES2TS_ACP_PYTHON must name the ACP environment interpreter")
        if not needs:
            command = [self.acp_python, "-m", "acp.cli", "run", _WORKFLOW, "--help"]
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command, cwd=self.acp_root, env=self._child_environment(),
                    stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
                returncode = completed.returncode
                output = (completed.stdout + completed.stderr).decode("utf-8",
                                                                      errors="replace")
            except (OSError, subprocess.TimeoutExpired) as exc:
                returncode, output = None, str(exc)
            evidence["probe_command"] = command
            evidence["probe_returncode"] = returncode
            evidence["probe_log_sha256"] = digest(output[-4000:])
            evidence["probe_seconds"] = time.monotonic() - started
            if returncode != 0:
                needs.append(
                    f"ACP CLI rejected the {_WORKFLOW} workflow probe "
                    f"(returncode={returncode}); an ACP checkout with the "
                    "BatchOptimize non-TS capability (ACP-G2-04) is required")
        self._probe = {"available": not needs, "needs": needs, "evidence": evidence,
                       "fail_closed": True, "local_fallback": False}
        return self._probe

    def _child_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        source = str(Path(self.acp_root) / "src")
        env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"]
                                      if env.get("PYTHONPATH") else "")
        return env

    # -- evaluation ---------------------------------------------------------

    def evaluation_address(self, geometry) -> tuple[str, str]:
        """Content-addressed evaluation identity ``origin-<n>/eval-<content16>``."""
        content = digest({"operation": self.operation,
                          "geometry": np.asarray(geometry, float).tolist(),
                          "method_context": {"method": self.method, "basis": self.basis,
                                             "charge": self.charge,
                                             "multiplicity": self.multiplicity,
                                             "elements": list(self.elements)}})
        counter_file = self.folder / "origin_counter.txt"
        counter_file.parent.mkdir(parents=True, exist_ok=True)
        counter = int(counter_file.read_text().strip() or 0) if counter_file.exists() else 0
        return f"origin-{counter:04d}", "eval-" + content[:16]

    def evaluate_free(self, geometry, evaluation_id) -> dict[str, Any]:
        """Run (or replay) one unconstrained optimization through ACP."""
        if self.capability_probe:
            probe = self.probe_capability()
            if not probe["available"]:
                raise OriginBackendUnavailable(probe["needs"], probe["evidence"])
        geometry_rows = np.asarray(geometry, float).tolist()
        parts = [p for p in str(evaluation_id).split("/") if p]
        if len(parts) != 2 or not all(_PORTABLE_ID.fullmatch(p) for p in parts):
            raise ValueError(f"INVALID_EVALUATION_ID:{evaluation_id}")
        trial_part, content_part = parts
        folder = self.folder / trial_part / content_part
        receipt = folder / "result.json"
        binding = digest({"evaluation_id": str(evaluation_id),
                          "geometry": geometry_rows, "operation": self.operation,
                          "method": self.method, "basis": self.basis,
                          "charge": self.charge, "multiplicity": self.multiplicity,
                          "elements": list(self.elements), "profile": ORIGIN_PROFILE})
        if receipt.exists():
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            if stored.get("request_sha256") != binding:
                raise ValueError(
                    "IDENTITY_CONFLICT:CACHED_ORIGIN_INPUT_MISMATCH:"
                    f"evaluation_id={evaluation_id}:"
                    f"stored_request_sha256={stored.get('request_sha256')}:"
                    f"binding_request_sha256={binding}")
            return stored
        record = self._run_free_optimization(folder, geometry_rows, binding,
                                             str(evaluation_id))
        write_json(receipt, record)
        return record

    def _run_free_optimization(self, folder: Path, geometry_rows, binding: str,
                               evaluation_id: str) -> dict[str, Any]:
        work = folder / "WORK" / "pes2ts"
        work.mkdir(parents=True, exist_ok=True)
        input_xyz = work / "origin_input.xyz"
        payload = _xyz_text(self.elements, geometry_rows).encode("utf-8")
        if input_xyz.exists():
            if input_xyz.read_bytes() != payload:
                raise ACPCLIError("origin attempt input is immutable and already differs")
        else:
            input_xyz.write_bytes(payload)
        batch_request = {"schema_version": "batch_structures_v1", "items": [{
            "id": ORIGIN_ITEM_ID, "candidate_id": f"pes2ts:{self.operation}",
            "role": "minimum", "geometry": str(input_xyz),
            "charge": self.charge, "multiplicity": self.multiplicity}]}
        items_file = work / "batch_request.json"
        items_file.write_text(json.dumps(batch_request, ensure_ascii=False,
                                         sort_keys=True, indent=2), encoding="utf-8")
        args = [_WORKFLOW, "--items-file", str(items_file),
                "--profile", ORIGIN_PROFILE, "--layout-mode", "single_flat",
                "--select", ORIGIN_ITEM_ID, "--method", self.method,
                "--charge", str(self.charge),
                "--multiplicity", str(self.multiplicity),
                "--nproc", str(self.nproc), "--output", str(folder)]
        if self.basis:
            args.extend(["--basis", self.basis])
        if self.acp_config is not None:
            args.extend(["--config", str(self.acp_config)])
        log_path = work / "acp_cli.log"
        claim_path = work / "origin_cli.claim"
        try:
            claim_path.open("x", encoding="utf-8").write(f"pid={os.getpid()}\n")
        except FileExistsError as exc:
            raise ACPCLIError("origin attempt is already claimed") from exc
        command = [self.acp_python, "-m", "acp.cli", "run", *args]
        started = time.monotonic()
        interrupted = False
        with log_path.open("wb") as log_stream:
            popen_kwargs: dict[str, Any] = (
                {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                if os.name == "nt" else {"start_new_session": True})
            try:
                process = subprocess.Popen(command, cwd=self.acp_root,
                                           env=self._child_environment(),
                                           stdin=subprocess.DEVNULL, stdout=log_stream,
                                           stderr=subprocess.STDOUT, **popen_kwargs)
                try:
                    returncode = process.wait(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._terminate(process)
                    returncode = process.wait()
                    error = (f"ACP {_WORKFLOW} exceeded timeout of "
                             f"{self.timeout_seconds:g} seconds")
                except KeyboardInterrupt:
                    interrupted = True
                    self._terminate(process)
                    returncode = process.wait()
                    error = "cancelled by user"
                else:
                    error = None if returncode == 0 else \
                        f"ACP CLI exited with return code {returncode}"
            except OSError as exc:
                error, returncode = f"could not start ACP CLI: {exc}", None
        wall = time.monotonic() - started
        claim_path.unlink(missing_ok=True)
        log_text = _collect_attempt_logs(folder, log_path)
        record = self._collect(folder, evaluation_id, binding, wall, returncode,
                               error, log_text, interrupted)
        return record

    @staticmethod
    def _terminate(process: subprocess.Popen[Any]) -> None:
        try:
            process.terminate()
        except OSError:
            pass

    def _collect(self, folder: Path, evaluation_id: str, binding: str, wall: float,
                 returncode: int | None, error: str | None, log_text: str,
                 interrupted: bool) -> dict[str, Any]:
        acp: dict[str, Any] = {"workflow": _WORKFLOW, "profile": ORIGIN_PROFILE,
                               "wall_seconds": wall, "returncode": returncode,
                               "timed_out": bool(error and "timeout" in error.lower()),
                               "reused": False,
                               "log_sha256": digest(log_text)}
        record = {"success": False, "failure_class": None, "coordinates": None,
                  "energy": None, "duration_seconds": wall,
                  "gradient_hartree_per_angstrom": None,
                  "acp_receipt_ref": str(folder / "result.json"),
                  "request_sha256": binding,
                  "evaluation_identity": {"scheme": "pes2ts_execution_identity_v1",
                                          "evaluation_id": evaluation_id},
                  "acp": acp}
        if interrupted:
            record["failure_class"] = "CANCELLED"
            return record
        if error is not None:
            if "SCF NOT CONVERGED" in log_text:
                record["failure_class"] = "SCF_NOT_CONVERGED"
            elif acp["timed_out"]:
                record["failure_class"] = "TIMEOUT"
            else:
                record["failure_class"] = "PROCESS_FAILED_WITHOUT_NORMAL_TERMINATION"
            return record
        try:
            verified = _verified_manifest(folder, _WORKFLOW)
        except (ACPCLIError, OSError) as exc:
            record["failure_class"] = "PROCESS_FAILED_WITHOUT_NORMAL_TERMINATION"
            record["failure_detail"] = f"ACP result incomplete: {exc}"
            return record
        acp["manifest_sha256"] = verified["manifest_sha256"]
        try:
            product = verified["products"][f"batch_{ORIGIN_ITEM_ID}"]
            _elements, geometry = _parse_xyz(Path(product["absolute_path"]),
                                             self.elements)
        except (KeyError, ACPCLIError, OSError) as exc:
            record["failure_class"] = "ORIGIN_PRODUCT_MISSING"
            record["failure_detail"] = str(exc)
            return record
        if "SCF NOT CONVERGED" in log_text:
            record["failure_class"] = "SCF_NOT_CONVERGED"
            return record
        energy = _read_energy_product(folder, ORIGIN_ITEM_ID)
        record.update({"success": True, "failure_class": None,
                       "coordinates": geometry, "energy": energy})
        if energy is None:
            # Honest typed failure: the campaign's continuation needs a finite
            # origin energy, and nothing may be fabricated (R3).
            record.update({"success": False,
                           "failure_class": "ORIGIN_ENERGY_UNAVAILABLE"})
        return record


__all__ = ["ORIGIN_BACKEND_UNAVAILABLE", "ORCAOriginPreparation",
           "OriginBackendUnavailable"]
