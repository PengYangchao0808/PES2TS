"""Frozen PES2TS request builder for the ACP ``OrcaGradient`` workflow.

This module owns the PES2TS side of the frozen request contract
``pes2ts_orca_gradient_request_v1`` (ADR-0002 X4′-B; ACP workflow
``OrcaGradient`` from ``acp run OrcaGradient --gradient-config <file>``).
It is a pure data builder: it never launches a process and never writes files.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

#: Request schema identifier frozen by the ACP ``OrcaGradient`` workflow.
REQUEST_SCHEMA_VERSION: Final[str] = "pes2ts_orca_gradient_request_v1"
#: ACP workflow name launched by the transport for this request shape.
ACP_WORKFLOW_NAME: Final[str] = "OrcaGradient"
#: ACP CLI flag that carries the request file path.
ACP_GRADIENT_CONFIG_FLAG: Final[str] = "--gradient-config"
#: Product schema of ``RESULT/gradient/gradient.json``.
GRADIENT_PRODUCT_SCHEMA: Final[str] = "orca_gradient_product_v1"


def request_sha256(request: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the canonical request payload.

    Mirrors the ACP ``_request_digest`` convention (sorted keys, UTF-8) so a
    PES2TS receipt and an ACP product can be compared on equal footing.
    """
    canonical = json.dumps(request, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_gradient_request(*, geometry: Sequence[Sequence[float]],
                           elements: Sequence[str], method: str, basis: str,
                           charge: int, multiplicity: int,
                           route_extras: Sequence[str] | None = None,
                           timeout_seconds: float | None = None,
                           nproc: int | None = None,
                           extra_blocks: Sequence[str] | None = None,
                           scf_convergence: str | None = None,
                           output_name: str | None = None) -> dict[str, Any]:
    """Build a frozen ``pes2ts_orca_gradient_request_v1`` payload.

    Every knob is explicit — this builder never invents chemistry defaults.
    ``basis`` may be an empty string (legal for GFN/composite methods).
    ``timeout_seconds`` is coerced to an integer ≥ 1 because ACP's schema
    rejects non-integral timeouts.
    """
    rows = [[float(v) for v in row] for row in geometry]
    symbols = [str(e) for e in elements]
    if not rows or len(rows) != len(symbols):
        raise ValueError("geometry and elements must be non-empty and aligned")
    for row in rows:
        if len(row) != 3 or not all(math.isfinite(v) for v in row):
            raise ValueError("geometry rows must be finite [x, y, z]")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method is required")
    if not isinstance(basis, str):
        raise ValueError("basis is required (empty string is legal)")
    if isinstance(charge, bool) or not isinstance(charge, int):
        raise ValueError("charge must be an int")
    if isinstance(multiplicity, bool) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise ValueError("multiplicity must be an int >= 1")
    if nproc is not None and (isinstance(nproc, bool) or not isinstance(nproc, int) or nproc < 1):
        raise ValueError("nproc must be an int >= 1 when set")
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a positive number when set")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number when set")
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "geometry": rows,
        "elements": symbols,
        "method": method.strip(),
        "basis": basis,
        "charge": charge,
        "multiplicity": multiplicity,
        "route_extras": list(route_extras or []),
        "extra_blocks": list(extra_blocks or []),
    }
    if timeout_seconds is not None:
        request["timeout_seconds"] = max(1, int(math.ceil(float(timeout_seconds))))
    if nproc is not None:
        request["nproc"] = nproc
    if scf_convergence is not None:
        request["scf_convergence"] = scf_convergence
    if output_name is not None:
        request["output_name"] = output_name
    return request


def validate_gradient_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a frozen request payload — never default knobs.

    Returns the request unchanged on success; raises :class:`ValueError` with a
    typed reason on any contract violation.  This is a PES2TS pre-flight; ACP
    re-validates the same schema independently.
    """
    if not isinstance(request, Mapping):
        raise ValueError("gradient request must be a mapping")
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {REQUEST_SCHEMA_VERSION!r}, "
            f"got {request.get('schema_version')!r}"
        )
    geometry = request.get("geometry")
    elements = request.get("elements")
    if not isinstance(geometry, list) or not geometry:
        raise ValueError("request requires a geometry list of [x, y, z] rows")
    if not isinstance(elements, list) or len(elements) != len(geometry):
        raise ValueError("elements must be a list matching the geometry length")
    for row, element in zip(geometry, elements):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"geometry rows must be [x, y, z]: {row!r}")
        if not isinstance(element, str) or not element:
            raise ValueError(f"invalid element symbol: {element!r}")
        try:
            coords = [float(v) for v in row]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric geometry row: {row!r}") from exc
        if not all(math.isfinite(v) for v in coords):
            raise ValueError(f"non-finite geometry row: {row!r}")
    method = request.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method is required")
    if not isinstance(request.get("basis"), str):
        raise ValueError("basis is required (empty string is legal)")
    charge = request.get("charge")
    if isinstance(charge, bool) or not isinstance(charge, int):
        raise ValueError("charge must be an int")
    multiplicity = request.get("multiplicity")
    if isinstance(multiplicity, bool) or not isinstance(multiplicity, int) or multiplicity < 1:
        raise ValueError("multiplicity must be an int >= 1")
    route_extras = request.get("route_extras") or []
    if not isinstance(route_extras, list) or not all(isinstance(x, str) for x in route_extras):
        raise ValueError("route_extras must be a list of strings")
    extra_blocks = request.get("extra_blocks") or []
    if not isinstance(extra_blocks, list) or not all(isinstance(x, str) for x in extra_blocks):
        raise ValueError("extra_blocks must be a list of strings")
    timeout_seconds = request.get("timeout_seconds")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds < 1
    ):
        raise ValueError("timeout_seconds must be an int >= 1 when set")
    nproc = request.get("nproc")
    if nproc is not None and (isinstance(nproc, bool) or not isinstance(nproc, int) or nproc < 1):
        raise ValueError("nproc must be an int >= 1 when set")
    scf_convergence = request.get("scf_convergence")
    if scf_convergence is not None and not isinstance(scf_convergence, str):
        raise ValueError("scf_convergence must be a string when set")
    return dict(request)


__all__ = [
    "ACP_GRADIENT_CONFIG_FLAG",
    "ACP_WORKFLOW_NAME",
    "GRADIENT_PRODUCT_SCHEMA",
    "REQUEST_SCHEMA_VERSION",
    "build_gradient_request",
    "request_sha256",
    "validate_gradient_request",
]
