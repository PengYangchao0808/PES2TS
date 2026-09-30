"""Local ACP CLI execution for the OptTS/frequency/IRC validation chain.

Each stage has its own immutable attempt directory. IRC is launched only after
the preceding BatchOptimize result has a valid ACP v2 manifest, verified TS
geometry, and actual method/basis provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from pes2ts_core.integration.acp.cli_backend import ACPCLIBackend, ACPCLIError
from pes2ts_core.integration.acp.stage_results import (
    _verified_manifest,
    collect_batch_ts_frequency_evidence,
    collect_irc_evidence,
)
from pes2ts_core.utils.hashing import stable_json_dumps


@dataclass(frozen=True)
class ACPStageResult:
    workflow: str
    execution_id: str
    attempt_id: str
    status: str
    returncode: int | None
    task_root: str
    request_sha256: str
    manifest_sha256: str | None
    wall_seconds: float
    reused: bool = False
    error: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz_text(elements: list[str], geometry: list[list[float]]) -> str:
    if len(elements) != len(geometry) or not elements:
        raise ACPCLIError("proposal frame atom mapping and geometry are inconsistent")
    rows = [str(len(elements)), "PES2TS selected TS proposal"]
    for element, point in zip(elements, geometry):
        if len(point) != 3 or any(not isinstance(value, (int, float)) or
                                  isinstance(value, bool) or not math.isfinite(value)
                                  for value in point):
            raise ACPCLIError("proposal frame geometry must contain finite N×3 coordinates")
        rows.append(f"{element} {point[0]:.12f} {point[1]:.12f} {point[2]:.12f}")
    return "\n".join(rows) + "\n"


class ACPValidationCLIBackend:
    """Run and collect one selected proposal through ACP BatchOptimize then IRC."""

    def __init__(self, *, acp_root: str | Path,
                 python_executable: str | Path | None = None,
                 config_path: str | Path | None = None,
                 environment: dict[str, str] | None = None) -> None:
        # Reuse ACP checkout/Python validation and child-process cleanup rules.
        self.process_backend = ACPCLIBackend(acp_root=acp_root,
            python_executable=python_executable, config_path=config_path,
            environment=environment)
        self.acp_root = self.process_backend.acp_root
        self.python_executable = self.process_backend.python_executable
        self.config_path = self.process_backend.config_path
        self.environment = self.process_backend.environment

    def run_validation(self, *, case: dict[str, Any], review_record: dict[str, Any],
                       path: dict[str, Any],
                       proposal: dict[str, Any], source_frame_id: str,
                       validation_id: str,
                       expected_method: str, expected_basis: str,
                       output_root: str | Path, batch_execution_id: str,
                       batch_attempt_id: str, irc_execution_id: str,
                       irc_attempt_id: str, batch_item_id: str = "pes2ts_ts",
                       batch_timeout_seconds: float, irc_timeout_seconds: float,
                       nproc: int | None = None, memory: str | None = None,
                       irc_maxpoints: int = 100, irc_step: float = 0.1,
                       endpoint_tolerance_angstrom: float = 0.35) -> dict[str, Any]:
        """Run OptTS+frequency and two-way IRC, returning verified evidence.

        The selected proposal must be the current top-ranked frame in `path`.
        This runner deliberately accepts only the first proposal frame for now.
        """
        from pes2ts_core.contracts import validate_document
        for label, document in (("ReactionCase", case), ("ReviewRecord", review_record),
                                ("PathBundle", path), ("SeedProposal", proposal)):
            issues = validate_document(document)
            if issues:
                raise ACPCLIError(f"{label} contract failed: {issues[0]}")
        for identifier, name in ((batch_execution_id, "batch_execution_id"),
            (batch_attempt_id, "batch_attempt_id"), (irc_execution_id, "irc_execution_id"),
            (irc_attempt_id, "irc_attempt_id"), (batch_item_id, "batch_item_id")):
            if not isinstance(identifier, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", identifier):
                raise ACPCLIError(f"{name} must be a portable identifier")
        if case.get("status") != "ready":
            raise ACPCLIError("validation requires a ready ReactionCase")
        review_output = review_record.get("extensions", {}).get("pes2ts.review_output.v1", {})
        source = case.get("source", {})
        if (review_record.get("schema_name") != "ReviewRecord"
                or review_record.get("status") != "accepted"
                or (review_record.get("reaction_id"), review_record.get("case_id"),
                    review_record.get("dataset_version"), review_record.get("split"))
                   != (case.get("reaction_id"), case.get("case_id"),
                       case.get("dataset_version"), case.get("split"))
                or source.get("review_record_id") != review_record.get("object_id")
                or source.get("reviewed_from_case_sha256") != review_record.get("case_sha256")
                or review_output.get("reviewed_case_sha256") != case.get("content_sha256")):
            raise ACPCLIError("validation requires accepted human review bound to this exact ReactionCase")
        if path.get("status") != "usable" or path.get("reaction_id") != case.get("reaction_id"):
            raise ACPCLIError("validation requires a usable PathBundle for this ReactionCase")
        if (path.get("case_id") != case.get("case_id")
                or proposal.get("status") != "accepted"
                or proposal.get("path_status") != "usable"
                or proposal.get("path_id") != path.get("object_id")
                or proposal.get("path_content_sha256") != path.get("content_sha256")
                or proposal.get("reaction_id") != path.get("reaction_id")
                or proposal.get("case_id") != path.get("case_id")):
            raise ACPCLIError("validation requires an accepted proposal bound to this PathBundle")
        selected = proposal.get("selected_frames")
        if not isinstance(selected, list) or not selected or selected[0].get("frame_id") != source_frame_id:
            raise ACPCLIError("source frame must be the accepted proposal's first selected frame")
        frame = next((row for row in path.get("frames", [])
                      if row.get("frame_id") == source_frame_id), None)
        if frame is None:
            raise ACPCLIError("proposal source frame is missing from PathBundle")
        current_geometry_digest = hashlib.sha256(
            stable_json_dumps(frame.get("geometry")).encode()).hexdigest()
        if selected[0].get("geometry_sha256") != current_geometry_digest:
            raise ACPCLIError("proposal source geometry digest is stale relative to PathBundle")
        elements = [atom["element"] for atom in case["atoms"]]
        if frame.get("atom_map_ids") != [atom["atom_map_id"] for atom in case["atoms"]]:
            raise ACPCLIError("proposal source frame atom map differs from ReactionCase")
        reactant_state = case["reactant"]
        product_state = case["product"]
        if (reactant_state.get("charge") != product_state.get("charge")
                or reactant_state.get("multiplicity") != product_state.get("multiplicity")):
            raise ACPCLIError("TS validation runner currently requires matching R/P charge and multiplicity")
        charge = reactant_state.get("charge")
        multiplicity = reactant_state.get("multiplicity")
        if (not isinstance(charge, int) or isinstance(charge, bool)
                or not isinstance(multiplicity, int) or isinstance(multiplicity, bool)
                or multiplicity < 1):
            raise ACPCLIError("ReactionCase R/P electronic states are incomplete or invalid")
        if not isinstance(expected_method, str) or not expected_method.strip():
            raise ACPCLIError("validation method must be explicit")
        if not isinstance(expected_basis, str):
            raise ACPCLIError("validation basis must be explicit (empty string is allowed)")
        if nproc is not None and (not isinstance(nproc, int) or isinstance(nproc, bool) or nproc < 1):
            raise ACPCLIError("nproc must be a positive integer")
        if memory is not None and (not isinstance(memory, str) or not memory.strip()):
            raise ACPCLIError("memory must be a non-empty resource limit")
        if (not isinstance(irc_maxpoints, int) or isinstance(irc_maxpoints, bool)
                or irc_maxpoints < 1):
            raise ACPCLIError("irc_maxpoints must be a positive integer")
        if (not isinstance(irc_step, (int, float)) or isinstance(irc_step, bool)
                or not math.isfinite(irc_step) or irc_step <= 0):
            raise ACPCLIError("irc_step must be positive and finite")
        if (not isinstance(endpoint_tolerance_angstrom, (int, float))
                or isinstance(endpoint_tolerance_angstrom, bool)
                or not math.isfinite(endpoint_tolerance_angstrom)
                or endpoint_tolerance_angstrom <= 0):
            raise ACPCLIError("endpoint tolerance must be positive and finite")

        root = Path(output_root).expanduser().resolve()
        batch_dir = root / "batch" / "attempts" / batch_attempt_id
        batch_work = batch_dir / "WORK" / "pes2ts"
        batch_work.mkdir(parents=True, exist_ok=True)
        input_xyz = batch_work / "proposal.xyz"
        input_text = _xyz_text(elements, frame["geometry"])
        self._write_attempt_input(input_xyz, input_text.encode("utf-8"))
        batch_request = {"schema_version": "batch_structures_v1", "items": [{
            "id": batch_item_id, "candidate_id": proposal["object_id"],
            "role": "TS", "geometry": str(input_xyz),
            "charge": charge, "multiplicity": multiplicity,
        }]}
        items_file = batch_work / "batch_request.json"
        self._write_attempt_input(items_file, json.dumps(batch_request, ensure_ascii=False,
            sort_keys=True, indent=2).encode("utf-8"))
        batch_args = ["BatchOptimize", "--items-file", str(items_file),
            "--profile", "opt_freq", "--layout-mode", "single_flat",
            "--select", batch_item_id, "--transition-state-method", expected_method,
            "--transition-state-basis", expected_basis, "--charge", str(charge),
            "--multiplicity", str(multiplicity), "--output", str(batch_dir)]
        self._append_resource_args(batch_args, nproc, memory)
        batch_result = self._run_stage(workflow="BatchOptimize",
            execution_id=batch_execution_id, attempt_id=batch_attempt_id,
            task_root=batch_dir, request={"batch": batch_request,
                "source_geometry_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                "method": expected_method,
                "basis": expected_basis, "profile": "opt_freq", "nproc": nproc, "memory": memory},
            args=batch_args, timeout_seconds=batch_timeout_seconds)
        if batch_result.status != "completed":
            return self._failed_validation(validation_id=validation_id, path=path,
                proposal=proposal, batch_result=batch_result,
                note=f"ACP BatchOptimize failed: {batch_result.error or batch_result.status}")

        source_geometry = frame["geometry"]
        source_digest = hashlib.sha256(stable_json_dumps(source_geometry).encode()).hexdigest()
        try:
            optts, frequency = collect_batch_ts_frequency_evidence(
                task_root=batch_dir, item_id=batch_item_id, expected_elements=elements,
                source_frame_id=source_frame_id, source_geometry=source_geometry,
                source_geometry_sha256=source_digest, execution_id=batch_execution_id,
                attempt_id=batch_attempt_id, expected_method=expected_method,
                expected_basis=expected_basis)
        except ACPCLIError as exc:
            return self._failed_validation(validation_id=validation_id, path=path,
                proposal=proposal, batch_result=batch_result,
                note=f"ACP BatchOptimize evidence rejected: {exc}")
        if not optts.get("first_order_saddle") or frequency.get("status") != "passed":
            from pes2ts_core.integration.validation import build_validation_result
            not_run = {"status": "not_run"}
            result = build_validation_result(validation_id=validation_id, path=path,
                proposal=proposal, optts=optts, frequency=frequency,
                irc_forward=not_run, irc_reverse=not_run,
                validation_note="IRC was not started because OptTS/frequency did not establish a first-order saddle.")
            return {"batch": batch_result, "irc": None, "optts": optts,
                "frequency": frequency, "irc_forward": not_run,
                "irc_reverse": not_run, "validation_result": result}
        batch_verified = _verified_manifest(batch_dir, "BatchOptimize")
        ts_product = batch_verified["products"].get(f"batch_{batch_item_id}")
        provenance_path = batch_verified["result_root"] / "batch_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        item_prov = next(row for row in provenance["items"] if row.get("item_id") == batch_item_id)
        effective = item_prov["effective_config"]
        ts_proof = {"schema": "irc_ts_source_v1",
            "geometry_sha256": _sha256(ts_product["absolute_path"]),
            "method": effective["method"], "basis": str(effective.get("basis") or ""),
            "charge": charge, "multiplicity": multiplicity}

        irc_dir = root / "irc" / "attempts" / irc_attempt_id
        irc_work = irc_dir / "WORK" / "pes2ts"
        irc_work.mkdir(parents=True, exist_ok=True)
        irc_input = irc_work / "optimized_ts.xyz"
        optimized_ts_bytes = ts_product["absolute_path"].read_bytes()
        self._write_attempt_input(irc_input, optimized_ts_bytes)
        proof_file = irc_work / "ts_provenance.json"
        proof_bytes = json.dumps(ts_proof, sort_keys=True, indent=2).encode("utf-8")
        self._write_attempt_input(proof_file, proof_bytes)
        irc_args = ["irc", "--input", str(irc_input), "--ts-provenance", str(proof_file),
            "--input-role", "transition_state", "--direction", "both",
            "--method", ts_proof["method"], "--basis", ts_proof["basis"],
            "--charge", str(charge), "--multiplicity", str(multiplicity),
            "--maxpoints", str(irc_maxpoints), "--step", str(irc_step),
            "--output", str(irc_dir)]
        self._append_resource_args(irc_args, nproc, memory, mem_flag="--mem")
        irc_result = self._run_stage(workflow="irc", execution_id=irc_execution_id,
            attempt_id=irc_attempt_id, task_root=irc_dir,
            request={"ts_provenance": ts_proof,
                "optimized_ts_file_sha256": hashlib.sha256(optimized_ts_bytes).hexdigest(),
                "maxpoints": irc_maxpoints,
                "step": irc_step, "direction": "both", "nproc": nproc, "memory": memory},
            args=irc_args, timeout_seconds=irc_timeout_seconds)
        if irc_result.status != "completed":
            failed_irc = {"status": "failed", "execution_id": irc_execution_id,
                "attempt_id": irc_attempt_id}
            from pes2ts_core.integration.validation import build_validation_result
            result = build_validation_result(validation_id=validation_id, path=path,
                proposal=proposal, optts=optts, frequency=frequency,
                irc_forward=failed_irc, irc_reverse=failed_irc,
                validation_note=f"ACP IRC failed: {irc_result.error or irc_result.status}")
            return {"batch": batch_result, "irc": irc_result, "optts": optts,
                "frequency": frequency, "irc_forward": failed_irc,
                "irc_reverse": failed_irc, "validation_result": result}
        try:
            irc_forward, irc_reverse = collect_irc_evidence(task_root=irc_dir, case=case,
                optimized_ts_geometry_sha256=optts["optimized_geometry_sha256"],
                optimized_ts_file_sha256=ts_proof["geometry_sha256"],
                execution_id=irc_execution_id, attempt_id=irc_attempt_id,
                tolerance_angstrom=endpoint_tolerance_angstrom)
        except ACPCLIError as exc:
            from pes2ts_core.integration.validation import build_validation_result
            failed_irc = {"status": "failed", "execution_id": irc_execution_id,
                "attempt_id": irc_attempt_id}
            result = build_validation_result(validation_id=validation_id, path=path,
                proposal=proposal, optts=optts, frequency=frequency,
                irc_forward=failed_irc, irc_reverse=failed_irc,
                validation_note=f"ACP IRC evidence rejected: {exc}")
            return {"batch": batch_result, "irc": irc_result, "optts": optts,
                "frequency": frequency, "irc_forward": failed_irc,
                "irc_reverse": failed_irc, "validation_result": result}
        return {"batch": batch_result, "irc": irc_result, "optts": optts,
                "frequency": frequency, "irc_forward": irc_forward,
                "irc_reverse": irc_reverse}

    @staticmethod
    def _failed_validation(*, validation_id: str, path: dict[str, Any],
                           proposal: dict[str, Any], batch_result: ACPStageResult,
                           note: str) -> dict[str, Any]:
        from pes2ts_core.integration.validation import build_validation_result
        optts = {"status": "failed", "execution_id": batch_result.execution_id,
            "attempt_id": batch_result.attempt_id}
        not_run = {"status": "not_run"}
        result = build_validation_result(validation_id=validation_id, path=path,
            proposal=proposal, optts=optts, frequency=not_run,
            irc_forward=not_run, irc_reverse=not_run, validation_note=note)
        return {"batch": batch_result, "irc": None, "optts": optts,
            "frequency": not_run, "irc_forward": not_run,
            "irc_reverse": not_run, "validation_result": result}

    def _append_resource_args(self, args: list[str], nproc: int | None,
                              memory: str | None, mem_flag: str = "--mem") -> None:
        if nproc is not None:
            args.extend(["--nproc", str(nproc)])
        if memory is not None:
            args.extend([mem_flag, memory])
        if self.config_path is not None:
            args.extend(["--config", str(self.config_path)])

    @staticmethod
    def _write_attempt_input(path: Path, payload: bytes) -> None:
        """Create frozen attempt inputs once; never replace content under an ID."""
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise ACPCLIError(f"attempt input is immutable and already differs: {path.name}")
            return
        path.write_bytes(payload)

    def _run_stage(self, *, workflow: str, execution_id: str, attempt_id: str,
                   task_root: Path, request: dict[str, Any], args: list[str],
                   timeout_seconds: float) -> ACPStageResult:
        for value, label in ((execution_id, "execution_id"), (attempt_id, "attempt_id")):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
                raise ACPCLIError(f"{label} must be a portable identifier")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ACPCLIError("stage timeout must be positive and finite")
        work = task_root / "WORK" / "pes2ts"
        work.mkdir(parents=True, exist_ok=True)
        receipt_path = work / "stage_cli_receipt.json"
        log_path = work / "acp_cli.log"
        claim_path = work / "stage_cli.claim"
        request_digest = hashlib.sha256(stable_json_dumps(request).encode()).hexdigest()
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("execution_id") != execution_id or receipt.get("request_sha256") != request_digest:
                raise ACPCLIError("stage attempt ID already belongs to a different request")
            if receipt.get("status") == "completed":
                verified = _verified_manifest(task_root, workflow)
                if verified["manifest_sha256"] != receipt.get("manifest_sha256"):
                    raise ACPCLIError("ACP result changed after stage receipt was written")
                return ACPStageResult(workflow, execution_id, attempt_id, "completed", 0,
                    str(task_root), request_digest, verified["manifest_sha256"],
                    float(receipt.get("wall_seconds") or 0), True)
            if receipt.get("status") == "running":
                if self.process_backend._pid_alive(receipt.get("pid")):
                    raise ACPCLIError("ACP stage is still running; refusing duplicate submission")
                try:
                    verified = _verified_manifest(task_root, workflow)
                except (ACPCLIError, OSError):
                    pass
                else:
                    receipt.update({"status": "completed", "returncode": 0,
                        "finished_at": receipt.get("finished_at") or _now_utc(),
                        "manifest_sha256": verified["manifest_sha256"], "error": None})
                    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
                    return ACPStageResult(workflow, execution_id, attempt_id, "completed", 0,
                        str(task_root), request_digest, verified["manifest_sha256"],
                        float(receipt.get("wall_seconds") or 0), True)
            raise ACPCLIError("stage attempt is terminal/interrupted; retry with a new attempt ID")
        try:
            with claim_path.open("x", encoding="utf-8") as stream:
                stream.write(f"pid={os.getpid()}\n")
        except FileExistsError as exc:
            raise ACPCLIError("stage attempt is already claimed") from exc
        command = [self.python_executable, "-m", "acp.cli", "run", *args]
        started_at = _now_utc()
        receipt = {"workflow": workflow, "execution_id": execution_id,
            "attempt_id": attempt_id, "request_sha256": request_digest,
            "status": "running", "pid": None, "returncode": None,
            "started_at": started_at, "finished_at": None, "wall_seconds": None,
            "manifest_sha256": None, "command": command,
            "python_executable": self.python_executable,
            "config_sha256": _sha256(self.config_path) if self.config_path else None,
            "error": None}
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env.update(self.environment)
        source = str(self.acp_root / "src")
        env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        started = time.monotonic()
        interrupted = False
        with log_path.open("wb") as log_stream:
            kwargs: dict[str, Any] = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            try:
                process = subprocess.Popen(command, cwd=self.acp_root, env=env,
                    stdin=subprocess.DEVNULL, stdout=log_stream,
                    stderr=subprocess.STDOUT, **kwargs)
                receipt["pid"] = process.pid
                receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    self.process_backend._terminate_tree(process)
                    returncode = process.wait()
                    error = f"ACP {workflow} exceeded timeout of {timeout_seconds:g} seconds"
                except KeyboardInterrupt:
                    interrupted = True
                    self.process_backend._terminate_tree(process)
                    returncode = process.wait()
                    error = "cancelled by user"
                else:
                    error = None if returncode == 0 else f"ACP CLI exited with return code {returncode}"
            except OSError as exc:
                error, returncode = f"could not start ACP CLI: {exc}", None
        elapsed = round(time.monotonic() - started, 3)
        manifest_digest = None
        status = "cancelled" if interrupted else "failed"
        if error is None:
            try:
                manifest_digest = _verified_manifest(task_root, workflow)["manifest_sha256"]
                status = "completed"
            except (ACPCLIError, OSError) as exc:
                error = f"ACP CLI exited successfully but result is incomplete: {exc}"
        receipt.update({"status": status, "returncode": returncode,
            "finished_at": _now_utc(), "wall_seconds": elapsed,
            "manifest_sha256": manifest_digest, "error": error})
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
        claim_path.unlink(missing_ok=True)
        return ACPStageResult(workflow, execution_id, attempt_id, status,
            returncode, str(task_root), request_digest, manifest_digest, elapsed,
            error=error)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


__all__ = ["ACPStageResult", "ACPValidationCLIBackend"]
