"""Generic ACP-CLI transport for the ``XtbPathSearch`` workflow.

Launches ``python -m acp.cli run XtbPathSearch --path-config <file>
--output <dir>`` as an auditable local process, following the same
claim/receipt/log/timeout discipline as ``cli_backend.py`` and
``stage_cli.py``: an exclusive ``cli_attempt.claim`` prevents duplicate
concurrent submission, a terminal ``cli_receipt.json`` records the attempt,
stdout+stderr are merged into ``acp_cli.log``, the whole process group is
killed on timeout, and a zero exit alone never counts as success — the ACP
v2 result manifest must validate before the attempt is marked completed.

When ``register=True`` the CLI is also given ``--register`` (ACP X1'-D), which
binds the finished ``--output`` directory into the ACP jobs store under
``ACP_RUN_ROOT`` so the Workbench job list and
``GET /api/v1/jobs/{id}/s2/profile`` resolve it.  ACP registers only after
the workflow itself completed, so a non-zero exit *with* a valid RESULT
manifest means registration failed: the attempt is then failed loudly with a
typed error naming ``--register``/``ACP_RUN_ROOT`` (never a silent
``completed`` the Workbench cannot show; the RESULT tree stays on disk
untouched).  A non-zero exit without a valid manifest is an ordinary workflow
failure either way.

This module owns only the outer PES2TS attempt lifecycle.  ACP still owns
its WORK/RESULT layout, the path search itself, and the result manifest.
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
from pes2ts_core.integration.acp.xtb_path_request import (
    ACP_PATH_CONFIG_FLAG,
    ACP_WORKFLOW_NAME,
    request_sha256,
    validate_path_request,
)
from pes2ts_core.utils.hashing import sha256_file

#: Workflow-parameterized CLI subcommand name.
_CLI_RUN_COMMAND = "run"
#: Receipt/log/claim directory beneath the attempt's ``WORK`` root.
_WORK_DIRNAME = "WORK"
#: PES2TS-owned metadata directory inside ``WORK``.
_PES2TS_DIRNAME = "pes2ts"
#: Relative RESULT locations contracted by plan appendix A.
_PROFILE_RELATIVE = "pes_search/pes_profile.json"
_FRAMES_DIR_RELATIVE = "pes_search/path_frames"
_TRAJECTORY_RELATIVE = "pes_search/xtbpath.xyz"
_MANIFEST_RELATIVE = "result_manifest.json"


@dataclass(frozen=True)
class XtbPathAttemptResult:
    """Typed outcome of one ``XtbPathSearch`` ACP CLI attempt."""

    execution_id: str
    attempt_id: str
    #: ``completed`` only after the ACP v2 manifest validated; otherwise
    #: ``failed``.  Never a silent partial success.
    status: str
    returncode: int | None
    wall_seconds: float
    request_sha256: str
    timed_out: bool
    attempt_dir: str
    manifest_path: str | None
    profile_path: str | None
    frames_dir: str | None
    raw_trajectory_path: str | None
    log_ref: str
    error: str | None = None
    reused: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_result_path(raw: Any) -> str:
    """Return a RESULT-relative posix path, rejecting unsafe references."""
    if not isinstance(raw, str) or not raw or raw.startswith(("/", "\\")) or ":" in raw:
        raise ACPCLIError("ACP result Product.path must be relative to RESULT")
    value = raw.replace("\\", "/")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ACPCLIError("ACP result Product.path contains an unsafe path component")
    return PurePosixPath(value).as_posix()


def validate_xtb_path_result(output_dir: str | Path) -> dict[str, Any]:
    """Verify ACP's completed v2 ``XtbPathSearch`` result and its products.

    Checks the manifest header (``version==2``, ``workflow=="XtbPathSearch"``,
    ``status=="completed"``), every product path (relative, unique, resolving
    inside ``RESULT``, a regular file, SHA-256 matching when declared), and
    the three contracted artifacts under ``RESULT/pes_search/``: the
    ``pes_profile_v2`` profile, the per-frame XYZ directory, and the raw
    ``xtbpath.xyz`` trajectory.  Returns a summary with local digests and
    resolved artifact paths; raises :class:`ACPCLIError` on any violation.
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
            raise ACPCLIError(
                f"ACP result product is missing or inaccessible: {portable}"
            ) from exc
        if not target.is_relative_to(result_root_resolved):
            raise ACPCLIError(f"ACP result product resolves outside RESULT: {portable}")
        if not target.is_file():
            raise ACPCLIError(f"ACP result product is not a file: {portable}")
        metadata = product.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ACPCLIError(f"ACP product metadata is malformed: {product_id}")
        digest = sha256_file(target)
        declared_hash = metadata.get("sha256")
        if declared_hash is not None and declared_hash != digest:
            raise ACPCLIError(f"ACP product SHA256 does not match: {product_id}")
        products.append({"id": product_id, "path": portable, "sha256": digest,
                         "size_bytes": target.stat().st_size})
    profile_path = result_root / _PROFILE_RELATIVE
    frames_dir = result_root / _FRAMES_DIR_RELATIVE
    trajectory_path = result_root / _TRAJECTORY_RELATIVE
    if not profile_path.is_file():
        raise ACPCLIError(
            f"completed {ACP_WORKFLOW_NAME} result does not provide RESULT/{_PROFILE_RELATIVE}"
        )
    if not frames_dir.is_dir():
        raise ACPCLIError(
            f"completed {ACP_WORKFLOW_NAME} result does not provide RESULT/{_FRAMES_DIR_RELATIVE}/"
        )
    if not trajectory_path.is_file():
        raise ACPCLIError(
            f"completed {ACP_WORKFLOW_NAME} result does not provide RESULT/{_TRAJECTORY_RELATIVE}"
        )
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACPCLIError(f"ACP path profile is unreadable: {exc}") from exc
    if not isinstance(profile, dict) or profile.get("schema_version") != "pes_profile_v2":
        raise ACPCLIError("ACP path profile does not satisfy pes_profile_v2")
    return {"manifest_sha256": sha256_file(manifest_path), "manifest": manifest,
            "profile": profile, "products": products,
            "manifest_path": str(manifest_path), "profile_path": str(profile_path),
            "frames_dir": str(frames_dir), "raw_trajectory_path": str(trajectory_path)}


def _result_paths(attempt_dir: Path, *, completed: bool) -> tuple[str | None, str | None, str | None, str | None]:
    if not completed:
        return None, None, None, None
    result_root = attempt_dir / "RESULT"
    return (str(result_root / _MANIFEST_RELATIVE),
            str(result_root / Path(*PurePosixPath(_PROFILE_RELATIVE).parts)),
            str(result_root / Path(*PurePosixPath(_FRAMES_DIR_RELATIVE).parts)),
            str(result_root / Path(*PurePosixPath(_TRAJECTORY_RELATIVE).parts)))


def _result_from_receipt(receipt: dict[str, Any], attempt_dir: Path, *,
                         reused: bool = False) -> XtbPathAttemptResult:
    completed = receipt.get("status") == "completed"
    manifest_path, profile_path, frames_dir, trajectory_path = _result_paths(
        attempt_dir, completed=completed)
    return XtbPathAttemptResult(
        execution_id=receipt["execution_id"], attempt_id=receipt["attempt_id"],
        status=receipt["status"], returncode=receipt.get("returncode"),
        wall_seconds=float(receipt.get("wall_seconds") or 0.0),
        request_sha256=receipt["request_sha256"],
        timed_out=bool(receipt.get("timed_out")), attempt_dir=str(attempt_dir),
        manifest_path=manifest_path, profile_path=profile_path,
        frames_dir=frames_dir, raw_trajectory_path=trajectory_path,
        log_ref="WORK/pes2ts/acp_cli.log", error=receipt.get("error"),
        reused=reused)


def run_xtb_path_attempt(*, acp_root: str | Path,
                         python_executable: str | Path | None,
                         acp_config_path: str | Path | None,
                         request: dict[str, Any],
                         output_root: str | Path,
                         execution_id: str, attempt_id: str,
                         timeout_seconds: float,
                         extra_args: tuple[str, ...] | list[str] = (),
                         register: bool = False) -> XtbPathAttemptResult:
    """Run or recover one immutable ``XtbPathSearch`` ACP CLI attempt.

    Writes the frozen request to ``<output_root>/WORK/pes2ts/path_config.json``,
    claims ``cli_attempt.claim`` against duplicate concurrent submission, and
    launches ``python -m acp.cli run XtbPathSearch --path-config ...
    --output <output_root>`` with ``cwd=acp_root`` and
    ``PYTHONPATH=<acp_root>/src``.  With ``register=True`` the argv also
    carries ``--register`` so ACP binds the output directory into its jobs
    store (``ACP_RUN_ROOT`` must be shared with the ACP server).  The terminal
    state is always recorded in ``cli_receipt.json``; a completed status
    additionally requires a valid ACP v2 ``XtbPathSearch`` manifest.  When
    ``register=True`` and the CLI exits non-zero while that manifest still
    validates, registration is what failed: the attempt is failed with a
    typed ``--register``/``ACP_RUN_ROOT`` error instead of a generic exit-code
    message.  Contract violations raise :class:`ACPCLIError`; attempt-level
    failures return a typed failed result.
    """
    validate_path_request(request)
    for name, value in (("execution_id", execution_id), ("attempt_id", attempt_id)):
        if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ACPCLIError(f"{name} must be a portable non-empty identifier")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ACPCLIError("timeout_seconds must be positive")
    if not isinstance(register, bool):
        raise ACPCLIError("register must be a boolean")
    if isinstance(extra_args, (str, bytes)):
        raise ACPCLIError("extra_args must be a sequence of strings")
    for arg in extra_args:
        if not isinstance(arg, str) or not arg:
            raise ACPCLIError("extra_args items must be non-empty strings")
    # Validates the ACP checkout (src/acp/cli.py) and resolves the Python
    # executable; also supplies _terminate_tree/_pid_alive/receipt helpers.
    backend = ACPCLIBackend(acp_root=acp_root, python_executable=python_executable,
                            config_path=acp_config_path)
    attempt_dir = Path(output_root).expanduser().resolve()
    work = attempt_dir / _WORK_DIRNAME / _PES2TS_DIRNAME
    work.mkdir(parents=True, exist_ok=True)
    receipt_path = work / "cli_receipt.json"
    request_path = work / "path_config.json"
    log_path = work / "acp_cli.log"
    claim_path = work / "cli_attempt.claim"
    request_digest = request_sha256(request)

    old = ACPCLIBackend._read_receipt(receipt_path)
    if old is not None:
        if (old.get("execution_id") != execution_id
                or old.get("request_sha256") != request_digest):
            raise ACPCLIError("attempt ID already belongs to a different execution or request")
        if old.get("status") == "completed":
            verified = validate_xtb_path_result(attempt_dir)
            if old.get("manifest_sha256") != verified["manifest_sha256"]:
                raise ACPCLIError("completed ACP result changed after its attempt receipt was written")
            return _result_from_receipt(old, attempt_dir, reused=True)
        if old.get("status") == "running":
            if ACPCLIBackend._pid_alive(old.get("pid")):
                raise ACPCLIError("this ACP attempt is still running; refusing duplicate submission")
            # The parent may have been interrupted after ACP finalized RESULT
            # but before PES2TS wrote the terminal receipt.
            try:
                verified = validate_xtb_path_result(attempt_dir)
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
               ACP_WORKFLOW_NAME, ACP_PATH_CONFIG_FLAG, str(request_path),
               "--output", str(attempt_dir)]
    if backend.config_path is not None:
        command += ["--config", str(backend.config_path)]
    if register:
        command.append("--register")
    command += list(extra_args)
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
               "register": register,
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
            receipt["error"] = f"ACP {ACP_WORKFLOW_NAME} exceeded timeout of {timeout_seconds:g} seconds"
        elif returncode != 0:
            receipt["status"] = "failed"
            registration_failed = False
            if register:
                # ACP registers only after workflow success, so exit!=0 with a
                # valid RESULT manifest means registration failed, not the path
                # search.  Fail loudly: visualization was explicitly requested.
                try:
                    validate_xtb_path_result(attempt_dir)
                except (ACPCLIError, OSError):
                    pass
                else:
                    registration_failed = True
            if registration_failed:
                receipt["error"] = (
                    f"ACP CLI exited with return code {returncode} after a valid "
                    f"{ACP_WORKFLOW_NAME} result: --register failed, so the run "
                    f"is not in the ACP jobs store (check that the ACP server "
                    f"and this CLI share ACP_RUN_ROOT). RESULT remains on disk "
                    f"at {attempt_dir}"
                )
            else:
                receipt["error"] = f"ACP CLI exited with return code {returncode}"
        else:
            try:
                verified = validate_xtb_path_result(attempt_dir)
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
    "XtbPathAttemptResult",
    "run_xtb_path_attempt",
    "validate_xtb_path_result",
]
