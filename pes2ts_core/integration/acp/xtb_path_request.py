"""Frozen PES2TS request builder for the ACP ``XtbPathSearch`` workflow.

This module owns the PES2TS side of the frozen request contract
``pes2ts_xtb_path_request_v1`` (plan appendix A of
``docs/plans/PES2TS_ACP执行统一与旧代码清理方案_20261003.md``).  It is a pure
data builder: it never launches a process and never writes files.

``build_path_inp_text`` reproduces, byte for byte, the ``$path`` control block
the (now deleted, ADR-0002 X2'-C) local xTB PATH runner used to write.  That
byte-identity is the RecipeEquivalence anchor.  The parameter order/defaults
are duplicated here as frozen copies so this integration module never
imported the generation runner; after the runner's deletion these frozen
copies are the sole source of the block rendering.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from pes2ts_core.utils.hashing import stable_json_dumps

#: Request schema identifier frozen by plan appendix A.
REQUEST_SCHEMA_VERSION: Final[str] = "pes2ts_xtb_path_request_v1"
#: Adapter version echoed in ``provenance.adapter_version``.
ADAPTER_VERSION: Final[str] = "pes2ts_xtb_path_request_v1"
#: ACP workflow name launched by the transport for this request shape.
ACP_WORKFLOW_NAME: Final[str] = "XtbPathSearch"
#: ACP CLI flag that carries the request file path.
ACP_PATH_CONFIG_FLAG: Final[str] = "--path-config"

#: ``$path`` block parameters in write order, with the plan-default values
#: used for any key the ``path_config`` mapping does not set.  Frozen copies
#: of the former ``runner.PATH_PARAM_ORDER`` / ``runner.PATH_PARAM_DEFAULTS``
#: (the local runner was deleted by ADR-0002 X2'-C); the request-builder
#: tests lock the rendering against literal expected blocks.
PATH_PARAM_DEFAULTS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "nrun": 1,
        "npoint": 50,
        "anopt": 10,
        "kpush": 0.003,
        "kpull": -0.015,
        "ppull": 0.05,
        "alp": 0.5,
    }
)
PATH_PARAM_ORDER: Final[tuple[str, ...]] = (
    "nrun",
    "npoint",
    "anopt",
    "kpush",
    "kpull",
    "ppull",
    "alp",
)


def build_path_inp_text(path_config: Mapping[str, Any]) -> str:
    """Return the exact ``$path`` block the local runner writes.

    *path_config* is the flat ``g2.path``-style mapping: keys missing from
    it fall back to :data:`PATH_PARAM_DEFAULTS`, unknown keys are ignored,
    and values are rendered with ``str()`` exactly as
    ``runner.write_path_inp`` does.  The result carries the same three-space
    indent, ``$path``/``$end`` sentinels, and trailing newline.
    """
    if not isinstance(path_config, Mapping):
        raise ValueError("path_config must be a mapping of $path parameters")
    lines = ["$path"]
    for key in PATH_PARAM_ORDER:
        lines.append(f"   {key}={path_config.get(key, PATH_PARAM_DEFAULTS[key])}")
    lines.append("$end")
    return "\n".join(lines) + "\n"


def request_sha256(request: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the canonical request core.

    The digest covers ``{schema_version, reaction_id, source, recipe}`` ONLY
    — ``provenance`` is excluded because it carries derived values (including
    this digest itself).  Serialization is canonical JSON (sorted keys,
    compact separators, literal UTF-8), so the digest is deterministic
    across runs and independent of key insertion order.
    """
    core = {
        "schema_version": request.get("schema_version"),
        "reaction_id": request.get("reaction_id"),
        "source": request.get("source"),
        "recipe": request.get("recipe"),
    }
    return hashlib.sha256(stable_json_dumps(core).encode("utf-8")).hexdigest()


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def build_path_request(
    *,
    reaction_id: str,
    start_xyz_text: str,
    end_xyz_text: str,
    charge: int,
    multiplicity: int,
    path_config: Mapping[str, Any],
    gfn_level: int = 2,
    threads: int = 4,
    timeout_seconds: int = 1800,
    seed: int | None = None,
    extra_args: Sequence[str] = (),
    plan_sha256: str | None = None,
    config_digest: str = "",
) -> dict[str, Any]:
    """Build a complete ``pes2ts_xtb_path_request_v1`` payload.

    ``uhf`` is derived as ``max(0, multiplicity - 1)``; ``request_sha256`` is
    filled by :func:`request_sha256` over the canonical core.  Invalid inputs
    raise ``ValueError`` — this builder never silently defaults anything.
    """
    reaction_id = _require_nonempty_text(reaction_id, "reaction_id")
    start_xyz_text = _require_nonempty_text(start_xyz_text, "start_xyz_text")
    end_xyz_text = _require_nonempty_text(end_xyz_text, "end_xyz_text")
    charge = _require_int(charge, "charge")
    multiplicity = _require_int(multiplicity, "multiplicity")
    if multiplicity < 1:
        raise ValueError("multiplicity must be >= 1")
    gfn_level = _require_int(gfn_level, "gfn_level")
    threads = _require_int(threads, "threads")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ):
        raise ValueError("timeout_seconds must be a positive number")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive number")
    if seed is not None:
        seed = _require_int(seed, "seed")
    if isinstance(extra_args, (str, bytes)) or not isinstance(
        extra_args, Sequence
    ):
        raise ValueError("extra_args must be a sequence of strings")
    normalized_extra = tuple(_require_nonempty_text(arg, "extra_args item")
                             for arg in extra_args)
    if plan_sha256 is not None:
        plan_sha256 = _require_nonempty_text(plan_sha256, "plan_sha256")
    config_digest = _require_nonempty_text(config_digest, "config_digest") \
        if config_digest != "" else ""
    source = {
        "source_type": "xyz_text_pair",
        "start_xyz": start_xyz_text,
        "end_xyz": end_xyz_text,
        "charge": charge,
        "multiplicity": multiplicity,
    }
    recipe = {
        "path_inp_text": build_path_inp_text(path_config),
        "gfn_level": gfn_level,
        "uhf": max(0, multiplicity - 1),
        "threads": threads,
        "timeout_seconds": timeout_seconds,
        "seed": seed,
        "extra_args": list(normalized_extra),
    }
    core = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "reaction_id": reaction_id,
        "source": source,
        "recipe": recipe,
    }
    return {
        **core,
        "provenance": {
            "plan_sha256": plan_sha256,
            "config_digest": config_digest,
            "request_sha256": request_sha256(core),
            "adapter_version": ADAPTER_VERSION,
        },
    }


def validate_path_request(request: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` on any malformed ``pes2ts_xtb_path_request_v1``.

    Checks frozen contract essentials only: schema version, non-empty XYZ
    endpoints, integer charge/multiplicity, and the literal ``$path`` block.
    No silent defaults are applied anywhere — a payload that would need
    one is rejected here.
    """
    if not isinstance(request, Mapping):
        raise ValueError("path request must be a mapping")
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError(
            f"path request schema_version must be {REQUEST_SCHEMA_VERSION!r}"
        )
    _require_nonempty_text(request.get("reaction_id"), "reaction_id")
    source = request.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("path request source must be an object")
    if source.get("source_type") != "xyz_text_pair":
        raise ValueError(
            "path request source.source_type must be 'xyz_text_pair'"
        )
    _require_nonempty_text(source.get("start_xyz"), "source.start_xyz")
    _require_nonempty_text(source.get("end_xyz"), "source.end_xyz")
    _require_int(source.get("charge"), "source.charge")
    multiplicity = _require_int(source.get("multiplicity"), "source.multiplicity")
    if multiplicity < 1:
        raise ValueError("source.multiplicity must be >= 1")
    recipe = request.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("path request recipe must be an object")
    path_inp_text = recipe.get("path_inp_text")
    if not isinstance(path_inp_text, str) or not path_inp_text.strip():
        raise ValueError("path request recipe.path_inp_text must be present")
    if not path_inp_text.startswith("$path"):
        raise ValueError(
            "path request recipe.path_inp_text must be the literal $path block"
        )


__all__ = [
    "ACP_PATH_CONFIG_FLAG",
    "ACP_WORKFLOW_NAME",
    "ADAPTER_VERSION",
    "PATH_PARAM_DEFAULTS",
    "PATH_PARAM_ORDER",
    "REQUEST_SCHEMA_VERSION",
    "build_path_inp_text",
    "build_path_request",
    "request_sha256",
    "validate_path_request",
]
