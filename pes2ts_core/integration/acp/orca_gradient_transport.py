"""Generic ACP-CLI transport for the ``OrcaGradient`` workflow.

Launches ``python -m acp.cli run OrcaGradient --gradient-config <file>
--output <dir>`` as an auditable local process, following the same
claim/receipt/log/timeout discipline as ``cli_backend.py``, ``stage_cli.py``
and ``xtb_path_transport.py``: an exclusive ``cli_attempt.claim`` prevents
duplicate concurrent submission, a terminal ``cli_receipt.json`` records the
attempt, stdout+stderr are merged into ``acp_cli.log``, the whole process
group is killed on timeout, and a zero exit alone never counts as success —
the ACP v2 ``OrcaGradient`` manifest must validate before the attempt is
marked completed.

This module owns only the outer PES2TS attempt lifecycle.  ACP still owns
its WORK/RESULT layout, the ORCA ``EnGrad`` single point, and the result
manifest.  Gradient values are the ORCA-printed energy gradient dE/dX in
Hartree/bohr — never forces, never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from typing import Any

from pes2ts_core.integration.acp.cli_backend import ACPCLIBackend, ACPCLIError
from pes2ts_core.integration.acp.orca_gradient_request import (
    ACP_GRADIENT_CONFIG_FLAG,
    ACP_WORKFLOW_NAME,
    GRADIENT_PRODUCT_SCHEMA,
    request_sha256,
    validate_gradient_request,
)
from pes2ts_core.generation.planning.gradient_evidence import BOHR_ANGSTROM
from pes2ts_core.utils.hashing import sha256_file

#: Workflow-parameterized CLI subcommand name.
_CLI_RUN_COMMAND = "run"
#: Receipt/log/claim directory beneath the attempt's ``WORK`` root.
_WORK_DIRNAME = "WORK"
#: PES2TS-owned metadata directory inside ``WORK``.
_PES2TS_DIRNAME = "pes2ts"
#: Relative RESULT locations contracted by the ACP OrcaGradient workflow.
_MANIFEST_RELATIVE = "result_manifest.json"
_GRADIENT_RELATIVE = "gradient/gradient.json"
_GEOMETRY_RELATIVE = "geometry/geometry.xyz"
_ENERGY_RELATIVE = "energy/energy.json"


@dataclass(frozen=True)
class OrcaGradientAttemptResult:
    """Typed outcome of one ``OrcaGradient`` ACP CLI attempt."""

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
    gradient_path: str | None
    geometry_path: str | None
    energy_path: str | None
    gradient_product: dict[str, Any] | None
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


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACPCLIError(f"ACP {label} is unreadable: {exc}") from exc


def _validate_gradient_product(path: Path, *, elements: list[str] | None,
                               geometry: list[list[float]] | None) -> dict[str, Any]:
    """Validate ``RESULT/gradient/gradient.json`` against its frozen schema."""
    product = _read_json(path, "gradient product")
    if not isinstance(product, dict):
        raise ACPCLIError("ACP gradient product must be a JSON object")
    if product.get("schema_version") != GRADIENT_PRODUCT_SCHEMA:
        raise ACPCLIError(
            f"ACP gradient product does not satisfy {GRADIENT_PRODUCT_SCHEMA}"
        )
    gradient = product.get("gradient_hartree_per_bohr")
    if not isinstance(gradient, list) or not gradient:
        raise ACPCLIError("ACP gradient product is missing gradient_hartree_per_bohr")
    rows: list[list[float]] = []
    for row in gradient:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ACPCLIError("ACP gradient_hartree_per_bohr rows must be [x, y, z]")
        try:
            values = [float(v) for v in row]
        except (TypeError, ValueError) as exc:
            raise ACPCLIError("ACP gradient_hartree_per_bohr rows must be numeric") from exc
        if not all(math.isfinite(v) for v in values):
            raise ACPCLIError("ACP gradient_hartree_per_bohr rows must be finite")
        rows.append(values)
    symbols = product.get("symbols")
    if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
        raise ACPCLIError("ACP gradient product symbols must be a list of strings")
    if len(symbols) != len(rows):
        raise ACPCLIError("ACP gradient product symbols/gradient length mismatch")
    energy = product.get("energy_hartree")
    if isinstance(energy, bool) or not isinstance(energy, (int, float)):
        raise ACPCLIError("ACP gradient product energy_hartree must be numeric")
    if not math.isfinite(float(energy)):
        raise ACPCLIError("ACP gradient product energy_hartree must be finite")
    if elements is not None and list(symbols) != list(elements):
        raise ACPCLIError("ACP gradient product symbols do not match the request elements")
    if geometry is not None:
        expected = len(geometry)
        if len(rows) != expected:
            raise ACPCLIError("ACP gradient product length does not match the request geometry")
    bohr_angstrom_product = product.get("gradient_hartree_per_angstrom")
    if isinstance(bohr_angstrom_product, list) and len(bohr_angstrom_product) == len(rows):
        try:
            converted = [[float(v) / BOHR_ANGSTROM for v in row] for row in rows]
            given = [[float(v) for v in row] for row in bohr_angstrom_product]
            if any(
                abs(a - b) > 1e-8
                for ra, rb in zip(converted, given)
                for a, b in zip(ra, rb)
            ):
                raise ACPCLIError(
                    "ACP gradient product bohr/angstrom values are inconsistent"
                )
        except (TypeError, ValueError) as exc:
            raise ACPCLIError("ACP gradient product angstrom gradient is malformed") from exc
    return product


def validate_orca_gradient_result(output_dir: str | Path, *,
                                  elements: list[str] | None = None,
                                  geometry: list[list[float]] | None = None) -> dict[str, Any]:
    """Verify ACP's completed v2 ``OrcaGradient`` result and its products.

    Checks the manifest header (``version==2``, ``workflow=="OrcaGradient"``,
    ``status=="completed"``), every product path (relative, unique, resolving
    inside ``RESULT``, a regular file, SHA-256 matching when declared), and
    the three contracted artifacts: ``RESULT/gradient/gradient.json``
    (``orca_gradient_product_v1``), ``RESULT/geometry/geometry.xyz`` and
    ``RESULT/energy/energy.json``.  Returns a summary with local digests, the
    parsed gradient product and resolved artifact paths; raises
    :class:`ACPCLIError` on any violation.
    """
    root = Path(output_dir).expanduser().resolve(strict=True)
    result_root = root / "RESULT"
    manifest_path = result_root / _MANIFEST_RELATIVE
    if not manifest_path.is_file():
        raise ACPCLIError("ACP CLI returned without RESULT/result_manifest.json")
    manifest = _read_json(manifest_path, "result manifest")
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
                         "size_bytes": target.stat().st_size, "metadata": metadata})
    gradient_path = result_root / Path(*PurePosixPath(_GRADIENT_RELATIVE).parts)
    geometry_path = result_root / Path(*PurePosixPath(_GEOMETRY_RELATIVE).parts)
    energy_path = result_root / Path(*PurePosixPath(_ENERGY_RELATIVE).parts)
    for label, target, is_file in (
        ("gradient", gradient_path, True),
        ("geometry", geometry_path, True),
        ("energy", energy_path, True),
    ):
        if is_file and not target.is_file():
            raise ACPCLIError(
                f"completed {ACP_WORKFLOW_NAME} result does not provide RESULT/"
                f"{_GRADIENT_RELATIVE if label == 'gradient' else _GEOMETRY_RELATIVE if label == 'geometry' else _ENERGY_RELATIVE}"
            )
    gradient_product = _validate_gradient_product(
        gradient_path, elements=elements, geometry=geometry)
    return {"manifest_sha256": sha256_file(manifest_path), "manifest": manifest,
            "products": products, "gradient_product": gradient_product,
            "manifest_path": str(manifest_path), "gradient_path": str(gradient_path),
            "geometry_path": str(geometry_path), "energy_path": str(energy_path)}


def parse_gradient_product_evidence(product: dict[str, Any] | None, *,
                                    gradient_path: str | Path | None,
                                    elements: list[str],
                                    geometry: list[list[float]],
                                    energy: float | None) -> dict[str, Any]:
    """Project an ACP gradient product into PES2TS ``gradient_evidence``.

    The evidence shape mirrors ``read_bound_engrad`` (``status`` /
    ``gradient_hartree_per_bohr`` / ``energy_hartree`` /
    ``maximum_geometry_error_angstrom`` / ``file`` / ``sha256``) so every
    downstream consumer of the old physical-gradient receipt keeps working.
    """
    if product is None or gradient_path is None:
        return {"status": "missing"}
    path = Path(gradient_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {"status": "missing", "file": str(path)}
    sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    try:
        gradient = product.get("gradient_hartree_per_bohr")
        symbols = product.get("symbols")
        saved_energy = product.get("energy_hartree")
        if not isinstance(gradient, list) or len(gradient) != len(elements):
            raise ValueError("INVALID_GRADIENT_LENGTH")
        if list(symbols or []) != list(elements):
            raise ValueError("INVALID_GRADIENT_IDENTITY")
        if energy is not None and isinstance(saved_energy, (int, float)) \
                and abs(float(saved_energy) - float(energy)) > 1e-6:
            raise ValueError("GRADIENT_FRAME_MISMATCH")
        return {"status": "bound",
                "gradient_hartree_per_bohr": gradient,
                "energy_hartree": float(saved_energy) if isinstance(saved_energy, (int, float)) else None,
                "maximum_geometry_error_angstrom": 0.0,
                "file": str(path), "sha256": sha256,
                "source": "acp_orca_gradient_product"}
    except (ValueError, TypeError) as exc:
        return {"status": "rejected", "reason": str(exc), "file": str(path), "sha256": sha256}


def _result_paths(attempt_dir: Path, *, completed: bool) -> tuple[str | None, str | None, str | None, str | None]:
    if not completed:
        return None, None, None, None
    result_root = attempt_dir / "RESULT"
    return (str(result_root / _MANIFEST_RELATIVE),
            str(result_root / Path(*PurePosixPath(_GRADIENT_RELATIVE).parts)),
            str(result_root / Path(*PurePosixPath(_GEOMETRY_RELATIVE).parts)),
            str(result_root / Path(*PurePosixPath(_ENERGY_RELATIVE).parts)))


def _result_from_receipt(receipt: dict[str, Any], attempt_dir: Path, *,
                         reused: bool = False) -> OrcaGradientAttemptResult:
    completed = receipt.get("status") == "completed"
    manifest_path, gradient_path, geometry_path, energy_path = _result_paths(
        attempt_dir, completed=completed)
    gradient_product = receipt.get("gradient_product")
    if completed and gradient_product is None and gradient_path is not None:
        try:
            gradient_product = _read_json(Path(gradient_path), "gradient product")
        except ACPCLIError:
            gradient_product = None
    return OrcaGradientAttemptResult(
        execution_id=receipt["execution_id"], attempt_id=receipt["attempt_id"],
        status=receipt["status"], returncode=receipt.get("returncode"),
        wall_seconds=float(receipt.get("wall_seconds") or 0.0),
        request_sha256=receipt["request_sha256"],
        timed_out=bool(receipt.get("timed_out")), attempt_dir=str(attempt_dir),
        manifest_path=manifest_path, gradient_path=gradient_path,
        geometry_path=geometry_path, energy_path=energy_path,
        gradient_product=gradient_product,
        log_ref="WORK/pes2ts/acp_cli.log", error=receipt.get("error"),
        reused=reused)


def run_orca_gradient_attempt(*, acp_root: str | Path,
                              python_executable: str | Path | None,
                              acp_config_path: str | Path | None,
                              request: dict[str, Any],
                              output_root: str | Path,
                              execution_id: str, attempt_id: str,
                              timeout_seconds: float,
                              extra_args: tuple[str, ...] | list[str] = ()) -> OrcaGradientAttemptResult:
    """Run or recover one immutable ``OrcaGradient`` ACP CLI attempt.

    Writes the frozen request to ``<output_root>/WORK/pes2ts/gradient_request.json``,
    claims ``cli_attempt.claim`` against duplicate concurrent submission, and
    launches ``python -m acp.cli run OrcaGradient --gradient-config ...
    --output <output_root>`` with ``cwd=acp_root`` and
    ``PYTHONPATH=<acp_root>/src``.  The terminal state is always recorded in
    ``cli_receipt.json``; a completed status additionally requires a valid ACP
    v2 ``OrcaGradient`` manifest.  Contract violations raise
    :class:`ACPCLIError`; attempt-level failures return a typed failed result.
    """
    validated = validate_gradient_request(request)
    for name, value in (("execution_id", execution_id), ("attempt_id", attempt_id)):
        if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise ACPCLIError(f"{name} must be a portable non-empty identifier")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ACPCLIError("timeout_seconds must be positive")
    if isinstance(extra_args, (str, bytes)):
        raise ACPCLIError("extra_args must be a sequence of strings")
    for arg in extra_args:
        if not isinstance(arg, str) or not arg:
            raise ACPCLIError("extra_args items must be non-empty strings")
    try:
        backend = ACPCLIBackend(acp_root=acp_root, python_executable=python_executable,
                                config_path=acp_config_path)
    except FileNotFoundError as exc:
        raise ACPCLIError(f"ACP checkout root is missing or unreadable: {acp_root}") from exc
    attempt_dir = Path(output_root).expanduser().resolve()
    work = attempt_dir / _WORK_DIRNAME / _PES2TS_DIRNAME
    work.mkdir(parents=True, exist_ok=True)
    receipt_path = work / "cli_receipt.json"
    request_path = work / "gradient_request.json"
    log_path = work / "acp_cli.log"
    claim_path = work / "cli_attempt.claim"
    binding = request_sha256(validated)

    old = ACPCLIBackend._read_receipt(receipt_path)
    if old is not None:
        if (old.get("execution_id") != execution_id
                or old.get("request_sha256") != binding):
            raise ACPCLIError("attempt ID already belongs to a different execution or request")
        if old.get("status") == "completed":
            verified = validate_orca_gradient_result(
                attempt_dir, elements=validated["elements"],
                geometry=validated["geometry"])
            if old.get("manifest_sha256") != verified["manifest_sha256"]:
                raise ACPCLIError("completed ACP result changed after its attempt receipt was written")
            return _result_from_receipt(old, attempt_dir, reused=True)
        if old.get("status") == "running":
            if ACPCLIBackend._pid_alive(old.get("pid")):
                raise ACPCLIError("this ACP attempt is still running; refusing duplicate submission")
            # The parent may have been interrupted after ACP finalized RESULT
            # but before PES2TS wrote the terminal receipt.
            try:
                verified = validate_orca_gradient_result(
                    attempt_dir, elements=validated["elements"],
                    geometry=validated["geometry"])
            except (ACPCLIError, OSError):
                pass
            else:
                old.update({"status": "completed", "returncode": 0,
                            "finished_at": old.get("finished_at") or _now(),
                            "manifest_sha256": verified["manifest_sha256"],
                            "gradient_product": verified["gradient_product"],
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
            json.dumps(validated, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8")
    except OSError:
        claim_path.unlink(missing_ok=True)
        raise
    command = [backend.python_executable, "-m", "acp.cli", _CLI_RUN_COMMAND,
               ACP_WORKFLOW_NAME, ACP_GRADIENT_CONFIG_FLAG, str(request_path),
               "--output", str(attempt_dir)]
    if backend.config_path is not None:
        command += ["--config", str(backend.config_path)]
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
               "wall_seconds": None, "request_sha256": binding,
               "manifest_sha256": None, "gradient_product": None,
               "timed_out": False, "error": None,
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
            receipt["error"] = f"ACP CLI exited with return code {returncode}"
        else:
            try:
                verified = validate_orca_gradient_result(
                    attempt_dir, elements=validated["elements"],
                    geometry=validated["geometry"])
                receipt["manifest_sha256"] = verified["manifest_sha256"]
                receipt["gradient_product"] = verified["gradient_product"]
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
    "OrcaGradientAttemptResult",
    "parse_gradient_product_evidence",
    "run_orca_gradient_attempt",
    "validate_orca_gradient_result",
]
