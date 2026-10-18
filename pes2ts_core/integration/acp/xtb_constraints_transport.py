"""ACP-CLI transport for the ``XtbConstrainedOpt`` workflow (G2-AB3 WP-4).

Launches ``python -m acp.cli run XtbConstrainedOpt --opt-config <file>
--output <dir>`` with the same claim/receipt/log/timeout discipline as
``xtb_path_transport.py``:

* exclusive ``cli_attempt.claim`` prevents duplicate concurrent submission;
* a terminal ``cli_receipt.json`` records the attempt (execution/attempt id,
  request digest, returncode, wall seconds, timeout flag);
* stdout+stderr land in ``WORK/pes2ts/acp_cli.log``;
* the whole process group is killed on timeout;
* a zero exit alone never counts as success — the ACP v2 result manifest,
  the ``evidence.json`` product, and (when the geometry exists) the
  ``optimized.xyz`` product must validate before ``status="completed"``.

``status="completed"`` here means *the calculation ran and its evidence is
complete*; geometric convergence is reported inside the evidence and the
accept/reject decision belongs to ``native_constraint_results``/the
controller, never to this transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from typing import Any

from pes2ts_core.integration.acp.cli_backend import ACPCLIBackend, ACPCLIError
from pes2ts_core.integration.acp.xtb_constraints_request import (
    ACP_OPT_CONFIG_FLAG,
    ACP_WORKFLOW_NAME,
    request_sha256,
    validate_constraint_request,
)
from pes2ts_core.utils.hashing import sha256_file

_CLI_RUN_COMMAND = "run"
_WORK_DIRNAME = "WORK"
_PES2TS_DIRNAME = "pes2ts"
_MANIFEST_RELATIVE = "result_manifest.json"
_EVIDENCE_RELATIVE = "pes2ts/xtb_constopt/evidence.json"
_OPTIMIZED_RELATIVE = "pes2ts/xtb_constopt/optimized.xyz"
_EVIDENCE_SCHEMA = "pes2ts_xtb_constrained_opt_evidence_v1"


@dataclass(frozen=True)
class XtbConstrainedOptAttemptResult:
    """Typed outcome of one ``XtbConstrainedOpt`` ACP CLI attempt."""

    execution_id: str
    attempt_id: str
    #: ``completed`` = calculation ran + evidence validated (convergence is
    #: reported in the evidence, not implied by this status).
    status: str
    returncode: int | None
    wall_seconds: float
    request_sha256: str
    timed_out: bool
    attempt_dir: str
    manifest_path: str | None
    evidence_path: str | None
    optimized_geometry_path: str | None
    log_ref: str
    error: str | None = None
    reused: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_result_path(raw: Any) -> str:
    if not isinstance(raw, str) or not raw or raw.startswith(("/", "\\")) or ":" in raw:
        raise ACPCLIError("ACP result Product.path must be relative to RESULT")
    value = raw.replace("\\", "/")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ACPCLIError("ACP result Product.path contains an unsafe path component")
    return PurePosixPath(value).as_posix()


def validate_xtb_constopt_result(output_dir: str | Path) -> dict[str, Any]:
    """Verify a completed ``XtbConstrainedOpt`` ACP result.

    Checks the v2 manifest header (``workflow=="XtbConstrainedOpt"``,
    ``status=="completed"``), product paths, the ``evidence.json`` product
    with schema ``pes2ts_xtb_constrained_opt_evidence_v1``, and — when the
    run produced a geometry — the ``optimized.xyz`` product.  Returns a
    summary with resolved paths and digests; raises :class:`ACPCLIError`.
    """
    root = Path(output_dir).expanduser().resolve(strict=True)
    result_root = root / "RESULT"
    manifest_path = result_root / _MANIFEST_RELATIVE
    if not manifest_path.is_file():
        raise ACPCLIError("ACP CLI returned without RESULT/result_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACPCLIError(f"ACP result manifest is unreadable: {exc}") from exc
    if (not isinstance(manifest, dict) or manifest.get("version") != 2
            or manifest.get("workflow") != ACP_WORKFLOW_NAME
            or manifest.get("status") != "completed"
            or not isinstance(manifest.get("products"), list)):
        raise ACPCLIError(
            f"ACP result manifest is not a completed v2 {ACP_WORKFLOW_NAME} manifest"
        )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    products: list[dict[str, Any]] = []
    result_root_resolved = result_root.resolve()
    has_evidence = has_geometry = False
    for product in manifest["products"]:
        if not isinstance(product, dict):
            raise ACPCLIError("ACP result manifest products must be objects")
        product_id = product.get("id")
        if not isinstance(product_id, str) or not product_id or product_id in seen_ids:
            raise ACPCLIError("ACP result manifest contains missing or duplicate product IDs")
        seen_ids.add(product_id)
        portable = _safe_result_path(product.get("path"))
        if portable in seen_paths:
            raise ACPCLIError("ACP result manifest contains duplicate product paths")
        seen_paths.add(portable)
        try:
            target = (result_root / Path(*PurePosixPath(portable).parts)).resolve(strict=True)
        except OSError as exc:
            raise ACPCLIError(f"ACP result product is missing or inaccessible: {portable}") from exc
        if not target.is_relative_to(result_root_resolved):
            raise ACPCLIError(f"ACP result product resolves outside RESULT: {portable}")
        if not target.is_file():
            raise ACPCLIError(f"ACP result product is not a file: {portable}")
        digest = sha256_file(target)
        products.append({"id": product_id, "path": portable, "sha256": digest,
                         "size_bytes": target.stat().st_size})
        if product_id == "evidence":
            has_evidence = True
        if product_id == "optimized_geometry":
            has_geometry = True
    if not has_evidence:
        raise ACPCLIError(
            f"completed {ACP_WORKFLOW_NAME} result does not provide the evidence product"
        )
    evidence_path = result_root / _EVIDENCE_RELATIVE
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACPCLIError(f"ACP constraint-opt evidence is unreadable: {exc}") from exc
    if not isinstance(evidence, dict) or evidence.get("schema_version") != _EVIDENCE_SCHEMA:
        raise ACPCLIError("evidence product does not satisfy the evidence schema")
    optimized_path: str | None = None
    if has_geometry:
        optimized_path = str(result_root / _OPTIMIZED_RELATIVE)
    return {"manifest_sha256": sha256_file(manifest_path), "manifest": manifest,
            "evidence": evidence, "products": products,
            "manifest_path": str(manifest_path), "evidence_path": str(evidence_path),
            "optimized_geometry_path": optimized_path}


def _result_from_receipt(receipt: dict[str, Any], attempt_dir: Path, *,
                         reused: bool = False) -> XtbConstrainedOptAttemptResult:
    completed = receipt.get("status") == "completed"
    manifest_path = evidence_path = optimized_path = None
    if completed:
        result_root = attempt_dir / "RESULT"
        manifest_path = str(result_root / _MANIFEST_RELATIVE)
        evidence_path = str(result_root / _EVIDENCE_RELATIVE)
        optimized_candidate = result_root / _OPTIMIZED_RELATIVE
        optimized_path = str(optimized_candidate) if optimized_candidate.is_file() else None
    return XtbConstrainedOptAttemptResult(
        execution_id=receipt["execution_id"], attempt_id=receipt["attempt_id"],
        status=receipt["status"], returncode=receipt.get("returncode"),
        wall_seconds=float(receipt.get("wall_seconds") or 0.0),
        request_sha256=receipt["request_sha256"],
        timed_out=bool(receipt.get("timed_out")), attempt_dir=str(attempt_dir),
        manifest_path=manifest_path, evidence_path=evidence_path,
        optimized_geometry_path=optimized_path,
        log_ref="WORK/pes2ts/acp_cli.log", error=receipt.get("error"),
        reused=reused)


def run_xtb_constopt_attempt(*, acp_root: str | Path,
                             python_executable: str | Path | None,
                             acp_config_path: str | Path | None,
                             request: dict[str, Any],
                             output_root: str | Path,
                             execution_id: str, attempt_id: str,
                             timeout_seconds: float) -> XtbConstrainedOptAttemptResult:
    """Run or recover one immutable ``XtbConstrainedOpt`` ACP CLI attempt.

    Idempotent recovery semantics (identical to ``run_xtb_path_attempt``):

    * a completed receipt with an unchanged manifest is returned as
      ``reused=True`` without launching anything;
    * a ``running`` receipt whose PID died is recovered by re-validating the
      RESULT tree — if ACP finalized it, the attempt completes without a
      second calculation (billed exactly once).
    """
    validate_constraint_request(request)
    for name, value in (("execution_id", execution_id), ("attempt_id", attempt_id)):
        if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ACPCLIError(f"{name} must be a portable non-empty identifier")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ACPCLIError("timeout_seconds must be positive")
    backend = ACPCLIBackend(acp_root=acp_root, python_executable=python_executable,
                            config_path=acp_config_path)
    attempt_dir = Path(output_root).expanduser().resolve()
    work = attempt_dir / _WORK_DIRNAME / _PES2TS_DIRNAME
    work.mkdir(parents=True, exist_ok=True)
    receipt_path = work / "cli_receipt.json"
    request_path = work / "opt_config.json"
    log_path = work / "acp_cli.log"
    claim_path = work / "cli_attempt.claim"
    request_digest = request_sha256(request)

    old = ACPCLIBackend._read_receipt(receipt_path)
    if old is not None:
        if (old.get("execution_id") != execution_id
                or old.get("request_sha256") != request_digest):
            raise ACPCLIError("attempt ID already belongs to a different execution or request")
        if old.get("status") == "completed":
            verified = validate_xtb_constopt_result(attempt_dir)
            if old.get("manifest_sha256") != verified["manifest_sha256"]:
                raise ACPCLIError("completed ACP result changed after its attempt receipt was written")
            return _result_from_receipt(old, attempt_dir, reused=True)
        if old.get("status") == "running":
            if ACPCLIBackend._pid_alive(old.get("pid")):
                raise ACPCLIError("this ACP attempt is still running; refusing duplicate submission")
            try:
                verified = validate_xtb_constopt_result(attempt_dir)
            except (ACPCLIError, OSError):
                pass
            else:
                old.update({"status": "completed", "returncode": 0,
                            "finished_at": old.get("finished_at") or _now(),
                            "manifest_sha256": verified["manifest_sha256"],
                            "error": None})
                ACPCLIBackend._write_receipt(receipt_path, old)
                return _result_from_receipt(old, attempt_dir, reused=True)
            raise ACPCLIError("attempt is terminal or interrupted; create a new attempt_id to retry")
        raise ACPCLIError("attempt is terminal or interrupted; create a new attempt_id to retry")

    try:
        with claim_path.open("x", encoding="utf-8") as claim_stream:
            claim_stream.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise ACPCLIError("this attempt ID is already claimed by another PES2TS process") from exc
    try:
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8")
    except OSError:
        claim_path.unlink(missing_ok=True)
        raise
    command = [backend.python_executable, "-m", "acp.cli", _CLI_RUN_COMMAND,
               ACP_WORKFLOW_NAME, ACP_OPT_CONFIG_FLAG, str(request_path),
               "--output", str(attempt_dir)]
    if backend.config_path is not None:
        command += ["--config", str(backend.config_path)]
    env = os.environ.copy()
    env.update(backend.environment)
    source = str(backend.acp_root / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    started_at = _now()
    started_clock = time.monotonic()
    receipt = {"execution_id": execution_id, "attempt_id": attempt_id,
               "workflow": ACP_WORKFLOW_NAME, "status": "running", "pid": None,
               "returncode": None, "started_at": started_at, "finished_at": None,
               "wall_seconds": None, "request_sha256": request_digest,
               "manifest_sha256": None, "timed_out": False, "error": None,
               "command": command, "acp_root": str(backend.acp_root),
               "python_executable": backend.python_executable,
               "config_sha256": (sha256_file(backend.config_path)
                                 if backend.config_path else None)}
    try:
        ACPCLIBackend._write_receipt(receipt_path, receipt)
    except OSError:
        claim_path.unlink(missing_ok=True)
        raise
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    process: Any = None
    try:
        with log_path.open("wb") as log_stream:
            process = subprocess.Popen(command, cwd=backend.acp_root, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log_stream,
                                       stderr=subprocess.STDOUT, **popen_kwargs)
            receipt["pid"] = process.pid
            ACPCLIBackend._write_receipt(receipt_path, receipt)
            try:
                returncode = process.wait(timeout=float(timeout_seconds))
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                ACPCLIBackend._terminate_tree(process)
                returncode = process.wait()
            except KeyboardInterrupt:
                receipt["status"] = "cancelled"
                receipt["error"] = "cancelled by user"
                ACPCLIBackend._terminate_tree(process)
                returncode = process.wait()
                receipt["returncode"] = returncode
                receipt["finished_at"] = _now()
                receipt["wall_seconds"] = round(time.monotonic() - started_clock, 3)
                ACPCLIBackend._write_receipt(receipt_path, receipt)
                claim_path.unlink(missing_ok=True)
                raise
        receipt["returncode"] = returncode
        receipt["finished_at"] = _now()
        receipt["wall_seconds"] = round(time.monotonic() - started_clock, 3)
        if receipt["timed_out"]:
            receipt["status"] = "failed"
            receipt["error"] = (
                f"ACP {ACP_WORKFLOW_NAME} exceeded timeout of {timeout_seconds:g} seconds"
            )
        elif returncode != 0:
            receipt["status"] = "failed"
            receipt["error"] = f"ACP CLI exited with return code {returncode}"
        else:
            try:
                verified = validate_xtb_constopt_result(attempt_dir)
                receipt["manifest_sha256"] = verified["manifest_sha256"]
                receipt["status"] = "completed"
            except (ACPCLIError, OSError) as exc:
                receipt["status"] = "failed"
                receipt["error"] = f"ACP CLI exited successfully but its result is incomplete: {exc}"
        ACPCLIBackend._write_receipt(receipt_path, receipt)
        claim_path.unlink(missing_ok=True)
        return _result_from_receipt(receipt, attempt_dir)
    except KeyboardInterrupt:
        claim_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if process is not None and process.poll() is None:
            ACPCLIBackend._terminate_tree(process)
        receipt["status"] = "failed"
        receipt["error"] = f"could not start ACP CLI: {exc}"
        receipt["finished_at"] = _now()
        receipt["wall_seconds"] = round(time.monotonic() - started_clock, 3)
        ACPCLIBackend._write_receipt(receipt_path, receipt)
        claim_path.unlink(missing_ok=True)
        return _result_from_receipt(receipt, attempt_dir)


__all__ = [
    "XtbConstrainedOptAttemptResult",
    "run_xtb_constopt_attempt",
    "validate_xtb_constopt_result",
]
