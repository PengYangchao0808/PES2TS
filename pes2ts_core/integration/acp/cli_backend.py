"""Auditable local-process backend for ACP's PESsearch CLI.

This module owns only the outer PES2TS attempt lifecycle. ACP still owns its
WORK/RESULT layout, scan implementation, and result manifest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from pes2ts_core.utils.hashing import stable_json_dumps


class ACPCLIError(RuntimeError):
    """An ACP CLI attempt could not be safely started or recovered."""


@dataclass(frozen=True)
class ACPCLIResult:
    execution_id: str
    attempt_id: str
    status: str
    returncode: int | None
    started_at: str
    finished_at: str
    wall_seconds: float
    attempt_dir: str
    request_sha256: str
    manifest_sha256: str | None
    log_ref: str
    error: str | None = None
    reused: bool = False
    timed_out: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_manifest_path(raw: Any) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or raw.startswith(("/", "\\")) or ":" in raw:
        raise ACPCLIError("ACP result Product.path must be relative to RESULT")
    value = raw.replace("\\", "/")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ACPCLIError("ACP result Product.path contains an unsafe path component")
    return path


def validate_cli_result(output_dir: str | Path) -> dict[str, Any]:
    """Verify ACP's v2 manifest and all registered RESULT products.

    ACP's CLI PESsearch products do not currently require per-product hashes;
    this reader always hashes the manifest and files locally and verifies a
    product hash when ACP provides one.
    """
    root = Path(output_dir).resolve(strict=True)
    result_root = root / "RESULT"
    manifest_path = result_root / "result_manifest.json"
    if not manifest_path.is_file():
        raise ACPCLIError("ACP CLI returned without RESULT/result_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACPCLIError(f"ACP result manifest is unreadable: {exc}") from exc
    if (not isinstance(manifest, dict) or manifest.get("version") != 2
            or manifest.get("workflow") != "PESsearch" or manifest.get("status") != "completed"
            or not isinstance(manifest.get("products"), list)):
        raise ACPCLIError("ACP result manifest is not a completed PESsearch v2 manifest")
    product_ids: set[str] = set()
    product_paths: set[str] = set()
    product_path_by_id: dict[str, str] = {}
    products = []
    for product in manifest["products"]:
        if not isinstance(product, dict):
            raise ACPCLIError("ACP result manifest products must be objects")
        product_id = product.get("id")
        if not isinstance(product_id, str) or not product_id or product_id in product_ids:
            raise ACPCLIError("ACP result manifest contains missing or duplicate product IDs")
        product_ids.add(product_id)
        relative = _safe_manifest_path(product.get("path"))
        portable = relative.as_posix()
        if portable in product_paths:
            raise ACPCLIError("ACP result manifest contains duplicate product paths")
        product_paths.add(portable)
        product_path_by_id[product_id] = portable
        target = (result_root / Path(*relative.parts)).resolve(strict=True)
        if not target.is_relative_to(result_root.resolve()):
            raise ACPCLIError("ACP result product resolves outside RESULT")
        if not target.is_file():
            raise ACPCLIError(f"ACP result product is not a file: {portable}")
        product_hash = _sha256(target)
        metadata = product.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ACPCLIError(f"ACP product metadata is malformed: {product_id}")
        declared_hash = metadata.get("sha256")
        if declared_hash is not None and declared_hash != product_hash:
            raise ACPCLIError(f"ACP product SHA256 does not match: {product_id}")
        products.append({"id": product_id, "path": portable, "sha256": product_hash,
                         "size_bytes": target.stat().st_size, "metadata": metadata})
    profile = result_root / "pes_search" / "pes_profile.json"
    if ("pes_profile" not in product_ids
            or product_path_by_id.get("pes_profile") != "pes_search/pes_profile.json"
            or not profile.is_file()):
        raise ACPCLIError("completed PESsearch result does not register its pes_profile")
    try:
        profile_doc = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACPCLIError(f"ACP PES profile is unreadable: {exc}") from exc
    if (not isinstance(profile_doc, dict) or profile_doc.get("schema_version") != "pes_profile_v2"
            or profile_doc.get("workflow") != "PESsearch" or profile_doc.get("status") != "completed"
            or not isinstance(profile_doc.get("frames"), list)):
        raise ACPCLIError("ACP PES profile does not satisfy pes_profile_v2")
    return {"manifest_sha256": _sha256(manifest_path), "manifest": manifest,
            "profile": profile_doc, "products": products}


def collect_cli_path_bundle(*, output_dir: str | Path, case: dict[str, Any],
                            plan: dict[str, Any], execution_id: str,
                            candidate_id: str | None = None,
                            path_status: str = "unchecked") -> dict[str, Any]:
    """Read ACP's RESULT profile and resolve its WORK XYZ frames safely."""
    from pes2ts_core.integration.acp.adapter import acp_s2_profile_to_path_bundle

    root = Path(output_dir).expanduser().resolve(strict=True)
    verified = validate_cli_result(root)
    profile = verified["profile"]
    raw_scan_dir = profile.get("scan_dir")
    if not isinstance(raw_scan_dir, str) or not raw_scan_dir.startswith("WORK/"):
        raise ACPCLIError("ACP PES profile scan_dir must be relative to WORK")
    scan_dir = _resolve_inside(root, raw_scan_dir)
    geometry_by_index: dict[int, list[list[float]]] = {}
    frames_with_task_paths = []
    geometry_refs: dict[int, str] = {}
    for product in verified["products"]:
        metadata = product.get("metadata", {})
        if metadata.get("pes2ts_role") != "scan_frame_geometry":
            continue
        frame_index = metadata.get("frame_index")
        if (not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0
                or frame_index in geometry_refs):
            raise ACPCLIError("PES2TS RESULT geometry products have invalid or duplicate frame indices")
        geometry_refs[frame_index] = f"RESULT/{product['path']}"
    for row in profile["frames"]:
        if not isinstance(row, dict):
            raise ACPCLIError("ACP PES profile frame must be an object")
        index = row.get("index")
        geometry_path = row.get("geometry_path")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ACPCLIError("ACP PES profile frame index must be a non-negative integer")
        if not isinstance(geometry_path, str) or not geometry_path:
            raise ACPCLIError(f"ACP PES frame {index} has no resolved geometry path")
        geometry_file = _resolve_inside(scan_dir, geometry_path)
        geometry_by_index[index] = _read_xyz_geometry(geometry_file, [atom["element"] for atom in case["atoms"]])
        result_geometry_ref = geometry_refs.get(index)
        if not isinstance(result_geometry_ref, str):
            raise ACPCLIError(f"ACP PES frame {index} has no registered RESULT geometry product")
        result_geometry = _read_xyz_geometry(
            _resolve_inside(root, result_geometry_ref), [atom["element"] for atom in case["atoms"]])
        if any(abs(actual - published) > 1e-9
               for row_actual, row_published in zip(geometry_by_index[index], result_geometry, strict=True)
               for actual, published in zip(row_actual, row_published, strict=True)):
            raise ACPCLIError(f"ACP PES frame {index} differs from its registered RESULT geometry")
        frames_with_task_paths.append({**row, "geometry_path": result_geometry_ref})
    if set(geometry_refs) != set(geometry_by_index):
        raise ACPCLIError("PES2TS RESULT geometry products do not exactly match the ACP profile frames")
    task_profile = {**profile, "frames": frames_with_task_paths}
    return acp_s2_profile_to_path_bundle(case=case, plan=plan,
        execution_id=execution_id, acp_task_id=None, profile=task_profile,
        geometry_angstrom_by_index=geometry_by_index, candidate_id=candidate_id,
        path_status=path_status)


def cli_result_to_execution_record(*, case: dict[str, Any], plan: dict[str, Any],
                                   candidate_id: str, result: ACPCLIResult) -> dict[str, Any]:
    """Record CLI process status with unknown CPU cost left explicit."""
    from pes2ts_core.integration.acp.adapter import acp_execution_to_record

    attempt_status = result.status
    attempt: dict[str, Any] = {"attempt_id": result.attempt_id, "status": attempt_status,
        "cpu_seconds": None, "started_at": result.started_at,
        "finished_at": result.finished_at or None, "log_ref": result.log_ref}
    if attempt_status == "failed":
        attempt["failure_code"] = "timeout" if result.timed_out else (
            "invalid_result" if result.returncode == 0 else "acp_cli_failed")
    return acp_execution_to_record(case=case, plan=plan, candidate_id=candidate_id,
        execution_id=result.execution_id, acp_task_id=None, task_status=attempt_status,
        attempts=[attempt], work_ref="WORK/pes2ts", result_ref="RESULT/result_manifest.json")


def _resolve_inside(root: Path, relative: str) -> Path:
    value = relative.replace("\\", "/")
    if value.startswith("/") or ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ACPCLIError("ACP artifact path must be relative and stay inside its task directory")
    target = (root / Path(*PurePosixPath(value).parts)).resolve(strict=True)
    if not target.is_relative_to(root.resolve()):
        raise ACPCLIError("ACP artifact path resolves outside its task directory")
    return target


def _read_xyz_geometry(path: Path, expected_elements: list[str]) -> list[list[float]]:
    if not path.is_file():
        raise ACPCLIError(f"ACP scan geometry is missing: {path.name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        atom_count = int(lines[0].strip())
    except (IndexError, ValueError) as exc:
        raise ACPCLIError("ACP scan geometry is not a valid XYZ file") from exc
    if atom_count != len(expected_elements) or len(lines) < atom_count + 2:
        raise ACPCLIError("ACP XYZ atom count does not match the ReactionCase")
    geometry: list[list[float]] = []
    elements: list[str] = []
    for line in lines[2:2 + atom_count]:
        fields = line.split()
        if len(fields) < 4:
            raise ACPCLIError("ACP XYZ coordinate row is incomplete")
        elements.append(fields[0])
        try:
            xyz = [float(value) for value in fields[1:4]]
        except ValueError as exc:
            raise ACPCLIError("ACP XYZ contains a non-numeric coordinate") from exc
        if any(not math.isfinite(value) for value in xyz):
            raise ACPCLIError("ACP XYZ contains a non-finite coordinate")
        geometry.append(xyz)
    if elements != expected_elements:
        raise ACPCLIError("ACP XYZ element order does not match the ReactionCase atom order")
    return geometry


def _materialize_cli_scan_geometries(output_dir: str | Path,
                                     scan_request: dict[str, Any],
                                     profile: dict[str, Any]) -> None:
    """Publish parsed scan frames as hash-registered RESULT geometry products.

    ACP keeps optimizer frames under WORK. PES2TS copies the exact parsed
    coordinates into its own RESULT namespace so TrajectoryFrame references
    remain consumable after WORK cleanup and pass through ACP's result-file
    serving/cache path. Native ACP frame files and its profile are untouched.
    """
    root = Path(output_dir).resolve(strict=True)
    result_root = root / "RESULT"
    manifest_path = result_root / "result_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ACPCLIError(f"ACP result manifest is unreadable during geometry publication: {exc}") from exc
    source = scan_request.get("source")
    xyz_text = source.get("xyz_text") if isinstance(source, dict) else None
    if not isinstance(xyz_text, str):
        raise ACPCLIError("scan request must include source.xyz_text to publish result geometries")
    xyz_lines = xyz_text.splitlines()
    try:
        expected_count = int(xyz_lines[0].strip())
        expected_elements = [line.split()[0] for line in xyz_lines[2:2 + expected_count]]
    except (IndexError, ValueError) as exc:
        raise ACPCLIError("scan request source.xyz_text is not a valid XYZ structure") from exc
    if expected_count < 1 or len(expected_elements) != expected_count:
        raise ACPCLIError("scan request source.xyz_text has an invalid atom count")
    scan_dir_ref = profile.get("scan_dir")
    rows = profile.get("frames")
    if not isinstance(scan_dir_ref, str) or not scan_dir_ref.startswith("WORK/"):
        raise ACPCLIError("ACP PES profile scan_dir must be relative to WORK")
    if not isinstance(rows, list) or not rows:
        raise ACPCLIError("ACP PES profile must contain frames before geometry publication")
    scan_dir = _resolve_inside(root, scan_dir_ref)
    products = manifest.get("products")
    if not isinstance(products, list):
        raise ACPCLIError("ACP result manifest products must be a list")
    product_by_id = {p.get("id"): p for p in products if isinstance(p, dict)}
    product_by_path = {p.get("path"): p for p in products if isinstance(p, dict)}
    seen_indices: set[int] = set()
    new_products = []
    modified = False
    for row in rows:
        if not isinstance(row, dict):
            raise ACPCLIError("ACP PES profile frame must be an object")
        index = row.get("index")
        geometry_path = row.get("geometry_path")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index in seen_indices:
            raise ACPCLIError("ACP PES profile frame indices must be unique non-negative integers")
        seen_indices.add(index)
        if not isinstance(geometry_path, str) or not geometry_path:
            raise ACPCLIError(f"ACP PES frame {index} has no resolved geometry path")
        source_path = _resolve_inside(scan_dir, geometry_path)
        geometry = _read_xyz_geometry(source_path, expected_elements)
        filename = f"frame_{index:05d}.xyz"
        relative = f"pes2ts/frames/{filename}"
        product_id = f"pes2ts_scan_frame_{index:05d}"
        target = (result_root / "pes2ts" / "frames" / filename).resolve()
        if not target.is_relative_to(result_root.resolve()):
            raise ACPCLIError("PES2TS result geometry resolves outside RESULT")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (f"{expected_count}\nPES2TS scan frame {index}\n" + "".join(
            f"{element} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n"
            for element, xyz in zip(expected_elements, geometry, strict=True)
        )).encode("utf-8")
        if target.exists() and target.read_bytes() != content:
            raise ACPCLIError(f"refusing to overwrite changed RESULT geometry product: {relative}")
        if not target.exists():
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(content)
            os.replace(temporary, target)
        digest = _sha256(target)
        product = {"id": product_id, "label": f"PES2TS scan frame {index}",
                   "path": relative, "kind": "file",
                   "metadata": {"pes2ts_role": "scan_frame_geometry", "frame_index": index,
                                "sha256": digest, "size_bytes": target.stat().st_size}}
        existing_id = product_by_id.get(product_id)
        existing_path = product_by_path.get(relative)
        if ((existing_id is not None and existing_id.get("path") != relative)
                or (existing_path is not None and existing_path.get("id") != product_id)):
            raise ACPCLIError("ACP result manifest conflicts with a PES2TS scan geometry product")
        if existing_id is None:
            new_products.append(product)
        elif existing_id != product:
            products[products.index(existing_id)] = product
            product_by_id[product_id] = product
            modified = True
    if not new_products and not modified:
        return
    products.extend(new_products)
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
                                  encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)


class ACPCLIBackend:
    """Launch and recover one ACP ``python -m acp.cli run PESsearch`` attempt."""

    def __init__(self, *, acp_root: str | Path, python_executable: str | Path | None = None,
                 config_path: str | Path | None = None,
                 environment: dict[str, str] | None = None) -> None:
        self.acp_root = Path(acp_root).expanduser().resolve(strict=True)
        source_root = self.acp_root / "src"
        if not (source_root / "acp" / "cli.py").is_file():
            raise ACPCLIError(f"ACP source checkout does not contain src/acp/cli.py: {self.acp_root}")
        selected_python = str(python_executable or sys.executable)
        resolved_python = shutil.which(selected_python)
        if resolved_python is None:
            raise ACPCLIError(f"ACP Python executable was not found: {selected_python}")
        self.python_executable = str(Path(resolved_python).resolve())
        self.config_path = Path(config_path).expanduser().resolve(strict=True) if config_path else None
        self.environment = dict(environment or {})
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._cancelled_attempts: set[str] = set()
        self._lock = threading.RLock()

    def run_scan(self, *, execution_id: str, attempt_id: str,
                 scan_request: dict[str, Any], output_root: str | Path,
                 timeout_seconds: float, nproc: int | None = None,
                 memory: str | None = None) -> ACPCLIResult:
        """Run or recover one immutable attempt; retries require a new attempt ID."""
        for name, value in (("execution_id", execution_id), ("attempt_id", attempt_id)):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
                raise ACPCLIError(f"{name} must be a portable non-empty identifier")
        if (not isinstance(scan_request, dict) or scan_request.get("mode") != "bond_length_scan"
                or not isinstance(scan_request.get("source"), dict)
                or not isinstance(scan_request.get("coordinate"), dict)
                or not isinstance(scan_request.get("protocol"), dict)):
            raise ACPCLIError("scan_request must be the complete ACP bond_length_scan request")
        if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ACPCLIError("timeout_seconds must be positive")
        if nproc is not None and (not isinstance(nproc, int) or isinstance(nproc, bool) or nproc < 1):
            raise ACPCLIError("nproc must be a positive integer")
        if memory is not None and (not isinstance(memory, str) or not memory.strip()):
            raise ACPCLIError("memory must be null or a non-empty ACP memory limit")
        base = Path(output_root).expanduser().resolve()
        attempt_dir = base / "attempts" / attempt_id
        receipt_path = attempt_dir / "WORK" / "pes2ts" / "cli_receipt.json"
        request_path = attempt_dir / "WORK" / "pes2ts" / "scan_request.json"
        log_path = attempt_dir / "WORK" / "pes2ts" / "acp_cli.log"
        claim_path = attempt_dir / "WORK" / "pes2ts" / "cli_attempt.claim"
        request_sha = hashlib.sha256(stable_json_dumps(scan_request).encode("utf-8")).hexdigest()
        attempt_dir.mkdir(parents=True, exist_ok=True)
        work_meta = receipt_path.parent
        work_meta.mkdir(parents=True, exist_ok=True)

        old = self._read_receipt(receipt_path)
        if old is not None:
            if old.get("execution_id") != execution_id or old.get("request_sha256") != request_sha:
                raise ACPCLIError("attempt ID already belongs to a different execution or request")
            if old.get("status") == "completed":
                verified = validate_cli_result(attempt_dir)
                if old.get("manifest_sha256") != verified["manifest_sha256"]:
                    raise ACPCLIError("completed ACP result changed after its attempt receipt was written")
                _materialize_cli_scan_geometries(attempt_dir, scan_request, verified["profile"])
                verified = validate_cli_result(attempt_dir)
                if old.get("manifest_sha256") != verified["manifest_sha256"]:
                    old["manifest_sha256"] = verified["manifest_sha256"]
                    self._write_receipt(receipt_path, old)
                return self._result_from_receipt(old, attempt_dir, log_path, reused=True)
            if old.get("status") == "running":
                if self._pid_alive(old.get("pid")):
                    raise ACPCLIError("this ACP attempt is still running; refusing duplicate submission")
                # The parent process may have been interrupted after ACP
                # finalized RESULT but before PES2TS wrote the terminal receipt.
                try:
                    verified = validate_cli_result(attempt_dir)
                    _materialize_cli_scan_geometries(attempt_dir, scan_request, verified["profile"])
                    verified = validate_cli_result(attempt_dir)
                except (ACPCLIError, OSError):
                    pass
                else:
                    old.update({"status":"completed", "returncode":0,
                                "finished_at":old.get("finished_at") or _now(),
                                "manifest_sha256":verified["manifest_sha256"],
                                "error":None})
                    self._write_receipt(receipt_path, old)
                    return self._result_from_receipt(old, attempt_dir, log_path, reused=True)
            raise ACPCLIError("attempt is terminal or interrupted; create a new attempt_id to retry")

        try:
            with claim_path.open("x", encoding="utf-8") as claim_stream:
                claim_stream.write(f"pid={os.getpid()}\n")
        except FileExistsError as exc:
            raise ACPCLIError("this attempt ID is already claimed by another PES2TS process") from exc
        try:
            request_path.write_text(json.dumps(scan_request, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        except OSError:
            claim_path.unlink(missing_ok=True)
            raise
        command = [self.python_executable, "-m", "acp.cli", "run", "PESsearch",
                   "--scan-config", str(request_path), "--output", str(attempt_dir)]
        if self.config_path is not None:
            command += ["--config", str(self.config_path)]
        if nproc is not None:
            command += ["--nproc", str(nproc)]
        if memory is not None:
            command += ["--mem", str(memory)]
        env = os.environ.copy()
        env.update(self.environment)
        src = str(self.acp_root / "src")
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        started_at = _now()
        started_clock = time.monotonic()
        receipt = {"execution_id": execution_id, "attempt_id": attempt_id, "status": "running",
                   "pid": None, "returncode": None, "started_at": started_at,
                   "finished_at": None, "wall_seconds": None, "request_sha256": request_sha,
                   "manifest_sha256": None, "error": None, "timed_out": False,
                   "command": command, "acp_root": str(self.acp_root),
                   "python_executable": self.python_executable,
                   "config_sha256": _sha256(self.config_path) if self.config_path else None,
                   "nproc": nproc, "memory": memory}
        try:
            self._write_receipt(receipt_path, receipt)
        except OSError:
            claim_path.unlink(missing_ok=True)
            raise
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        process: subprocess.Popen[Any] | None = None
        try:
            with log_path.open("wb") as log_stream:
                process = subprocess.Popen(command, cwd=self.acp_root, env=env,
                    stdin=subprocess.DEVNULL, stdout=log_stream, stderr=subprocess.STDOUT,
                    **popen_kwargs)
                with self._lock:
                    self._processes[attempt_id] = process
                receipt["pid"] = process.pid
                self._write_receipt(receipt_path, receipt)
                try:
                    returncode = process.wait(timeout=float(timeout_seconds))
                except subprocess.TimeoutExpired:
                    receipt["timed_out"] = True
                    self._terminate_tree(process)
                    returncode = process.wait()
                except KeyboardInterrupt:
                    receipt["status"] = "cancelled"
                    receipt["error"] = "cancelled by user"
                    self._terminate_tree(process)
                    returncode = process.wait()
                    receipt["returncode"] = returncode
                    receipt["finished_at"] = _now()
                    receipt["wall_seconds"] = round(time.monotonic() - started_clock, 3)
                    self._write_receipt(receipt_path, receipt)
                    raise
                finally:
                    with self._lock:
                        self._processes.pop(attempt_id, None)
            receipt["returncode"] = returncode
            receipt["finished_at"] = _now()
            receipt["wall_seconds"] = round(time.monotonic() - started_clock, 3)
            if receipt["timed_out"]:
                receipt["status"] = "failed"
                receipt["error"] = f"ACP CLI exceeded timeout of {timeout_seconds} seconds"
            elif attempt_id in self._cancelled_attempts:
                receipt["status"] = "cancelled"
                receipt["error"] = "cancelled by user"
                self._cancelled_attempts.discard(attempt_id)
            elif returncode != 0:
                receipt["status"] = "failed"
                receipt["error"] = f"ACP CLI exited with return code {returncode}"
            else:
                try:
                    verified = validate_cli_result(attempt_dir)
                    _materialize_cli_scan_geometries(attempt_dir, scan_request, verified["profile"])
                    verified = validate_cli_result(attempt_dir)
                    receipt["manifest_sha256"] = verified["manifest_sha256"]
                    receipt["status"] = "completed"
                except (ACPCLIError, OSError) as exc:
                    receipt["status"] = "failed"
                    receipt["error"] = f"ACP CLI exited successfully but its result is incomplete: {exc}"
            self._write_receipt(receipt_path, receipt)
            claim_path.unlink(missing_ok=True)
            return self._result_from_receipt(receipt, attempt_dir, log_path)
        except KeyboardInterrupt:
            claim_path.unlink(missing_ok=True)
            raise
        except OSError as exc:
            if process is not None and process.poll() is None:
                self._terminate_tree(process)
            receipt["status"] = "failed"
            receipt["error"] = f"could not start ACP CLI: {exc}"
            receipt["finished_at"] = _now()
            receipt["wall_seconds"] = round(time.monotonic() - started_clock, 3)
            self._write_receipt(receipt_path, receipt)
            claim_path.unlink(missing_ok=True)
            return self._result_from_receipt(receipt, attempt_dir, log_path)

    def cancel(self, attempt_id: str) -> bool:
        """Cancel a running local attempt and its child processes, if present."""
        with self._lock:
            process = self._processes.get(attempt_id)
        if process is None or process.poll() is not None:
            return False
        with self._lock:
            self._cancelled_attempts.add(attempt_id)
        self._terminate_tree(process)
        return True

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False

    @staticmethod
    def _read_receipt(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ACPCLIError(f"attempt receipt is unreadable: {path}") from exc
        if not isinstance(payload, dict):
            raise ACPCLIError("attempt receipt root must be an object")
        return payload

    @staticmethod
    def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temp_path, path)

    @staticmethod
    def _result_from_receipt(receipt: dict[str, Any], attempt_dir: Path,
                             log_path: Path | None = None, *, reused: bool = False) -> ACPCLIResult:
        return ACPCLIResult(execution_id=receipt["execution_id"], attempt_id=receipt["attempt_id"],
            status=receipt["status"], returncode=receipt.get("returncode"),
            started_at=receipt["started_at"], finished_at=receipt.get("finished_at") or "",
            wall_seconds=float(receipt.get("wall_seconds") or 0.0), attempt_dir=str(attempt_dir),
            request_sha256=receipt["request_sha256"], manifest_sha256=receipt.get("manifest_sha256"),
            log_ref="WORK/pes2ts/acp_cli.log", error=receipt.get("error"),
            reused=reused, timed_out=bool(receipt.get("timed_out")))


def write_cli_result_json(result: ACPCLIResult, path: str | Path) -> Path:
    """Persist a portable summary, excluding machine-local attempt directory."""
    target = Path(path)
    payload = asdict(result)
    payload.pop("attempt_dir", None)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return target
