"""Stage-state sidecar and input digests for the idempotent G0 pipeline.

The pipeline owns every skip decision through ``g0_stage_state.json``: after a
stage succeeds it records a canonical input digest plus the SHA-256 of each
produced artifact, and a stage is skipped only while its recorded digest and
outputs still match.  Source digests come from the source manifest when
available (it proves verification and survives relocation), and JSON artifacts
are digested with the volatile keys removed so a stage that rewrites a manifest
does not invalidate its own recorded digest.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pes2ts_core.g0 import MANIFEST_SCHEMA_VERSION, VOLATILE_KEYS
from pes2ts_core.g0.dedup import DUPLICATE_LEDGER_FILENAME
from pes2ts_core.g0.fetch import source_manifest_path
from pes2ts_core.g0.fingerprints import FINGERPRINTS_PARQUET_FILENAME
from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.neardup import LEAK_AUDIT_FILENAME
from pes2ts_core.g0.split_policy import REMEDIATION_NONE
from pes2ts_core.g0.split_sources import SPLIT_ASSIGNMENT_FILENAME, SPLIT_SUFFIXES
from pes2ts_core.utils.hashing import (
    JSONValue,
    sha256_bytes,
    sha256_file,
    stable_json_dumps,
)
from pes2ts_core.utils.jsonio import read_json, write_json

logger = logging.getLogger(__name__)

#: Stage-state sidecar filename written under ``config["paths"]["manifests"]``.
STAGE_STATE_FILENAME: Final[str] = "g0_stage_state.json"
#: Suffixes identifying the combined archive, the trajectory archive, and the
#: reaction-info table among the configured sources.
MAIN_H5_SUFFIX: Final[str] = "_TZVP.h5"
IRC_H5_SUFFIX: Final[str] = "_IRC.h5"
CSV_SUFFIX: Final[str] = "_reaction_info.csv"


@dataclass(slots=True)
class PipelineContext:
    """Mutable state shared by the stage runners of one pipeline run."""

    config: dict[str, Any]
    state: dict[str, dict[str, Any]]
    force: bool = False
    skip_fetch: bool = False
    remediation: str = REMEDIATION_NONE


def stage_state_path(config: Mapping[str, Any]) -> Path:
    """Return the path of the stage-state sidecar for *config*."""
    return Path(config["paths"]["manifests"]) / STAGE_STATE_FILENAME


def configured_filenames(config: Mapping[str, Any]) -> list[str]:
    """Return every configured source filename, honoring the key fallback."""
    return [
        str(entry.get("filename", key)) for key, entry in config["source"]["files"].items()
    ]


def source_name(config: Mapping[str, Any], suffix: str, *, exclude: str | None = None) -> str:
    """Return the single configured filename ending in *suffix* (not *exclude*)."""
    matches = [
        name
        for name in configured_filenames(config)
        if name.endswith(suffix) and (exclude is None or not name.endswith(exclude))
    ]
    if len(matches) != 1:
        msg = f"Expected exactly one configured source file ending in {suffix!r}, found {matches}"
        raise ValueError(msg)
    return matches[0]


def archive_source_names(config: Mapping[str, Any]) -> tuple[str, str]:
    """Return the ``(combined, trajectory)`` source filenames from *config*."""
    main = source_name(config, MAIN_H5_SUFFIX, exclude=IRC_H5_SUFFIX)
    return main, source_name(config, IRC_H5_SUFFIX)


def stage_input_paths(name: str, config: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return the declared input files of stage *name*."""
    raw = Path(config["paths"]["raw"])
    interim = Path(config["paths"]["interim"])
    manifests = Path(config["paths"]["manifests"])
    inventory = interim / INVENTORY_PARQUET_FILENAME
    if name == "fetch":
        return tuple(raw / filename for filename in configured_filenames(config))
    if name == "inventory":
        return (raw / source_name(config, CSV_SUFFIX), raw / archive_source_names(config)[0])
    if name == "quarantine":
        main_name, irc_name = archive_source_names(config)
        return (raw / main_name, raw / irc_name, raw / source_name(config, CSV_SUFFIX))
    if name == "dedup" or name == "cohorts":
        return (inventory,)
    if name == "split":
        return (
            *(raw / source_name(config, SPLIT_SUFFIXES[label]) for label in SPLIT_SUFFIXES),
            inventory,
        )
    if name == "audit":
        return (
            interim / SPLIT_ASSIGNMENT_FILENAME,
            inventory,
            manifests / DUPLICATE_LEDGER_FILENAME,
        )
    if name == "freeze":
        return (
            manifests / LEAK_AUDIT_FILENAME,
            interim / SPLIT_ASSIGNMENT_FILENAME,
            inventory,
            interim / FINGERPRINTS_PARQUET_FILENAME,
            manifests / DUPLICATE_LEDGER_FILENAME,
        )
    msg = f"Unknown pipeline stage {name!r}"
    raise ValueError(msg)


def source_records(config: Mapping[str, Any]) -> dict[str, str]:
    """Return ``filename -> sha256`` recorded in the source manifest."""
    path = source_manifest_path(config)
    if not path.is_file():
        return {}
    document = read_json(path)
    if not isinstance(document, dict):
        return {}
    files = document.get("files")
    if not isinstance(files, list):
        return {}
    records: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict):
            continue
        filename, digest = entry.get("filename"), entry.get("sha256")
        if isinstance(filename, str) and isinstance(digest, str):
            records[filename] = digest
    return records


def input_component(path: Path, recorded: Mapping[str, str]) -> str:
    """Return one input's digest contribution.

    A recorded source digest wins over the live file; a missing file
    contributes its basename so the digest stays deterministic.
    """
    recorded_digest = recorded.get(path.name)
    if recorded_digest is not None:
        return recorded_digest
    if not path.is_file():
        return f"missing:{path.name}"
    if path.suffix != ".json":
        return sha256_file(path)
    try:
        document = read_json(path)
    except (OSError, ValueError):
        return sha256_file(path)
    if isinstance(document, dict):
        document = {key: value for key, value in document.items() if key not in VOLATILE_KEYS}
    return sha256_bytes(stable_json_dumps(document).encode("utf-8"))


def inputs_digest(paths: Sequence[Path], recorded: Mapping[str, str]) -> str:
    """Return the canonical input digest over *paths* and *recorded* digests."""
    components: list[JSONValue] = [
        f"{path.name}:{input_component(path, recorded)}" for path in paths
    ]
    return sha256_bytes(stable_json_dumps(components).encode("utf-8"))


def fetch_digest(config: Mapping[str, Any]) -> str:
    """Return the fetch input digest over the manifest digests or the config specs."""
    recorded = source_records(config)
    if recorded:
        items: list[JSONValue] = [f"{name}:{recorded[name]}" for name in sorted(recorded)]
    else:
        files = config["source"]["files"]
        items = [
            f"{files[key].get('filename', key)}:{files[key].get('url')}:{files[key].get('md5')}"
            for key in sorted(files)
        ]
    return sha256_bytes(stable_json_dumps(items).encode("utf-8"))


def stage_digest(name: str, config: Mapping[str, Any]) -> str:
    """Return the canonical input digest of stage *name*."""
    if name == "fetch":
        return fetch_digest(config)
    return inputs_digest(stage_input_paths(name, config), source_records(config))


def outputs_with_digests(paths: Sequence[Path]) -> dict[str, str]:
    """Return ``{path: sha256}`` for every existing file in *paths*."""
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def recorded_outputs(ctx: PipelineContext, name: str) -> dict[str, str]:
    """Return the recorded ``{path: sha256}`` of stage *name* (empty if none)."""
    entry = ctx.state.get(name)
    outputs = entry.get("outputs") if isinstance(entry, dict) else None
    if not isinstance(outputs, dict):
        return {}
    return {
        str(path): str(digest)
        for path, digest in outputs.items()
        if isinstance(path, str) and isinstance(digest, str)
    }


def load_stage_state(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Load the stage-state sidecar; untrustworthy state yields ``{}``."""
    path = stage_state_path(config)
    if not path.is_file():
        return {}
    try:
        document = read_json(path)
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable stage state %s: %s", path, exc)
        return {}
    stages = document.get("stages") if isinstance(document, dict) else None
    if not isinstance(stages, dict):
        logger.warning("Ignoring malformed stage state %s", path)
        return {}
    return {
        name: dict(entry)
        for name, entry in stages.items()
        if isinstance(entry, dict)
    }


def write_stage_state(config: Mapping[str, Any], state: Mapping[str, Mapping[str, Any]]) -> None:
    """Atomically write the stage-state sidecar sorted by stage name."""
    document: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stages": {name: dict(state[name]) for name in sorted(state)},
    }
    write_json(stage_state_path(config), document)


def skip_reason(ctx: PipelineContext, name: str, digest: str) -> str | None:
    """Return the skip reason for stage *name*, or ``None`` when it must run."""
    if ctx.force:
        return None
    entry = ctx.state.get(name)
    if not isinstance(entry, dict) or entry.get("inputs_digest") != digest:
        return None
    outputs = recorded_outputs(ctx, name)
    if outputs and all(Path(path).is_file() for path in outputs):
        return "inputs unchanged"
    return None


def record_stage(ctx: PipelineContext, name: str, digest: str, outputs: Mapping[str, str]) -> None:
    """Record stage *name*'s input digest and output digests, then persist."""
    ctx.state[name] = {
        "inputs_digest": digest,
        "outputs": {path: outputs[path] for path in sorted(outputs)},
    }
    write_stage_state(ctx.config, ctx.state)


def sources_relocated(ctx: PipelineContext) -> bool:
    """Return whether the quarantine stage already moved the archives away."""
    if "quarantine" not in ctx.state or not source_manifest_path(ctx.config).is_file():
        return False
    raw = Path(ctx.config["paths"]["raw"])
    main_name, irc_name = archive_source_names(ctx.config)
    return not (raw / main_name).exists() and not (raw / irc_name).exists()


__all__ = [
    "CSV_SUFFIX",
    "IRC_H5_SUFFIX",
    "MAIN_H5_SUFFIX",
    "STAGE_STATE_FILENAME",
    "PipelineContext",
    "archive_source_names",
    "configured_filenames",
    "fetch_digest",
    "input_component",
    "inputs_digest",
    "load_stage_state",
    "outputs_with_digests",
    "record_stage",
    "recorded_outputs",
    "skip_reason",
    "source_name",
    "source_records",
    "sources_relocated",
    "stage_digest",
    "stage_input_paths",
    "stage_state_path",
    "write_stage_state",
]
