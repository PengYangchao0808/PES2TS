"""G–V campaign registry.

The repository has two orthogonal axes:

* **G** — a code/physics capability stage (G0, G1, G2, G2.T, G3.P.0–.3,
  G3.1, G3.2, G4.0–G4.2).  A G stage owns modules, contracts and exit gates.
* **V** — an experiment campaign (V4.2–V4.7, V5.0–V5.2, V6.0–V6.2,
  V7-A/B/C).  A V campaign owns a config, a frozen data manifest, a run set
  and a report.

This module loads the single mapping file
``config/registry/g_v_map.yaml`` and validates campaigns and run records
against it.  It is deliberately dependency-light: PyYAML plus the standard
library; no torch, no offline imports, no filesystem writes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP_PATH = PROJECT_ROOT / "config" / "registry" / "g_v_map.yaml"
GV_MAP_SCHEMA = "pes2ts_gv_map_v1"
CAMPAIGN_STATUSES = frozenset({"proposed", "active", "executed", "archived"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GVMapError(ValueError):
    """Typed rejection of a malformed G–V map or run record."""


@dataclass(frozen=True)
class StageEntry:
    stage_id: str
    title: str
    stage_doc: str | None
    code_globs: tuple[str, ...]


@dataclass(frozen=True)
class CampaignEntry:
    campaign_id: str
    title: str
    status: str
    primary_g: str
    supporting_g: tuple[str, ...]
    data_dependencies: tuple[str, ...]
    code_scope: tuple[str, ...]
    config: str | None
    data_manifest: str | None
    gates: tuple[str, ...]
    evidence: str | None
    comparison_only: tuple[str, ...]
    gate_note: str | None


@dataclass(frozen=True)
class GVMap:
    map_version: str
    path: Path
    stages: Mapping[str, StageEntry]
    campaigns: Mapping[str, CampaignEntry]


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GVMapError(f"{where} must be a YAML mapping")
    return value


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GVMapError(f"{where} must be a non-empty string")
    return value.strip()


def _optional_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, where)


def _str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GVMapError(f"{where} must be a YAML list of strings")
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_require_str(item, f"{where}[{index}]"))
    if len(set(items)) != len(items):
        raise GVMapError(f"{where} contains duplicate entries")
    return tuple(items)


def _parse_stage(stage_id: str, payload: Any) -> StageEntry:
    data = _require_mapping(payload, f"stages.{stage_id}")
    return StageEntry(
        stage_id=stage_id,
        title=_require_str(data.get("title"), f"stages.{stage_id}.title"),
        stage_doc=_optional_str(data.get("stage_doc"), f"stages.{stage_id}.stage_doc"),
        code_globs=_str_tuple(data.get("code_globs"), f"stages.{stage_id}.code_globs"),
    )


def _parse_campaign(campaign_id: str, payload: Any) -> CampaignEntry:
    data = _require_mapping(payload, f"campaigns.{campaign_id}")
    status = _require_str(data.get("status"), f"campaigns.{campaign_id}.status")
    if status not in CAMPAIGN_STATUSES:
        raise GVMapError(
            f"campaigns.{campaign_id}.status {status!r} not in "
            f"{sorted(CAMPAIGN_STATUSES)}"
        )
    return CampaignEntry(
        campaign_id=campaign_id,
        title=_require_str(data.get("title"), f"campaigns.{campaign_id}.title"),
        status=status,
        primary_g=_require_str(
            data.get("primary_g"), f"campaigns.{campaign_id}.primary_g"
        ),
        supporting_g=_str_tuple(
            data.get("supporting_g"), f"campaigns.{campaign_id}.supporting_g"
        ),
        data_dependencies=_str_tuple(
            data.get("data_dependencies"),
            f"campaigns.{campaign_id}.data_dependencies",
        ),
        code_scope=_str_tuple(
            data.get("code_scope"), f"campaigns.{campaign_id}.code_scope"
        ),
        config=_optional_str(data.get("config"), f"campaigns.{campaign_id}.config"),
        data_manifest=_optional_str(
            data.get("data_manifest"), f"campaigns.{campaign_id}.data_manifest"
        ),
        gates=_str_tuple(data.get("gates"), f"campaigns.{campaign_id}.gates"),
        evidence=_optional_str(
            data.get("evidence"), f"campaigns.{campaign_id}.evidence"
        ),
        comparison_only=_str_tuple(
            data.get("comparison_only"),
            f"campaigns.{campaign_id}.comparison_only",
        ),
        gate_note=_optional_str(
            data.get("gate_note"), f"campaigns.{campaign_id}.gate_note"
        ),
    )


def load_gv_map(path: str | Path | None = None) -> GVMap:
    """Load and structurally validate the G–V map (existence not required)."""
    map_path = Path(path) if path is not None else DEFAULT_MAP_PATH
    try:
        raw = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GVMapError(f"cannot read G–V map {map_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise GVMapError(f"invalid YAML in G–V map {map_path}: {exc}") from exc

    document = _require_mapping(raw, str(map_path))
    if document.get("schema") != GV_MAP_SCHEMA:
        raise GVMapError(
            f"schema must be {GV_MAP_SCHEMA!r}, got {document.get('schema')!r}"
        )
    map_version = _require_str(document.get("map_version"), "map_version")

    stages_raw = _require_mapping(document.get("stages"), "stages")
    if not stages_raw:
        raise GVMapError("stages must not be empty")
    stages = {
        _require_str(stage_id, "stages key"): _parse_stage(stage_id, payload)
        for stage_id, payload in stages_raw.items()
    }

    campaigns_raw = _require_mapping(document.get("campaigns"), "campaigns")
    if not campaigns_raw:
        raise GVMapError("campaigns must not be empty")
    campaigns = {
        _require_str(campaign_id, "campaigns key"): _parse_campaign(
            campaign_id, payload
        )
        for campaign_id, payload in campaigns_raw.items()
    }
    return GVMap(
        map_version=map_version,
        path=map_path,
        stages=stages,
        campaigns=campaigns,
    )


def resolve_campaign(gv_map: GVMap, campaign_id: str) -> CampaignEntry:
    """Return one campaign or raise a typed error."""
    try:
        return gv_map.campaigns[campaign_id]
    except KeyError as exc:
        known = ", ".join(sorted(gv_map.campaigns))
        raise GVMapError(
            f"unknown V campaign {campaign_id!r}; known campaigns: {known}"
        ) from exc


def _glob_exists(root: Path, pattern: str) -> bool:
    try:
        return any(root.glob(pattern))
    except (OSError, ValueError):
        return False


def validate_gv_map(
    gv_map: GVMap,
    *,
    repo_root: str | Path = PROJECT_ROOT,
    require_existing: bool | None = None,
) -> None:
    """Validate cross-references and (for active/executed campaigns) paths.

    ``require_existing=None`` means: enforce paths for ``active`` and
    ``executed`` campaigns, allow missing paths for ``proposed``.
    """
    root = Path(repo_root)
    for stage_id, stage in gv_map.stages.items():
        if stage.stage_id != stage_id:
            raise GVMapError(f"stage key {stage_id!r} != stage_id {stage.stage_id!r}")

    for campaign_id, campaign in gv_map.campaigns.items():
        where = f"campaigns.{campaign_id}"
        if campaign.primary_g not in gv_map.stages:
            raise GVMapError(f"{where}.primary_g {campaign.primary_g!r} is not a stage")
        if campaign.primary_g in campaign.supporting_g:
            raise GVMapError(f"{where}.supporting_g must not repeat primary_g")
        for stage_id in campaign.supporting_g:
            if stage_id not in gv_map.stages:
                raise GVMapError(f"{where}.supporting_g {stage_id!r} is not a stage")
        for stage_id in campaign.data_dependencies:
            if stage_id not in gv_map.stages:
                raise GVMapError(
                    f"{where}.data_dependencies {stage_id!r} is not a stage"
                )
        if not campaign.gates:
            raise GVMapError(f"{where}.gates must not be empty")
        if not campaign.code_scope:
            raise GVMapError(f"{where}.code_scope must not be empty")

        enforce = (
            require_existing
            if require_existing is not None
            else campaign.status in {"active", "executed"}
        )
        if not enforce:
            continue
        for pattern in campaign.code_scope:
            if not _glob_exists(root, pattern):
                raise GVMapError(
                    f"{where}.code_scope {pattern!r} matches no file under {root}"
                )
        for label, value in (
            ("config", campaign.config),
            ("data_manifest", campaign.data_manifest),
            ("evidence", campaign.evidence),
        ):
            if value is None:
                continue
            if not (root / value).exists():
                raise GVMapError(f"{where}.{label} does not exist: {value}")
        for pattern in campaign.comparison_only:
            if not _glob_exists(root, pattern):
                raise GVMapError(
                    f"{where}.comparison_only {pattern!r} matches no file"
                )


def validate_gv_run_record(record: Mapping[str, Any], gv_map: GVMap) -> None:
    """Validate the ``gv`` provenance block of one run record.

    The block is required for new (V5+) runs; legacy V4 records may omit it.
    """
    gv = record.get("gv")
    if not isinstance(gv, Mapping):
        raise GVMapError("run record is missing a 'gv' mapping")
    for key in ("map_version", "campaign_id", "primary_g", "supporting_g", "gates"):
        if key not in gv:
            raise GVMapError(f"run record gv block is missing {key!r}")
    if gv["map_version"] != gv_map.map_version:
        raise GVMapError(
            f"run record map_version {gv['map_version']!r} != "
            f"current {gv_map.map_version!r}"
        )
    campaign = resolve_campaign(gv_map, _require_str(gv["campaign_id"], "gv.campaign_id"))
    if _require_str(gv["primary_g"], "gv.primary_g") != campaign.primary_g:
        raise GVMapError(
            f"run record primary_g {gv['primary_g']!r} != "
            f"campaign {campaign.campaign_id} primary {campaign.primary_g!r}"
        )
    supporting = set(_str_tuple(gv["supporting_g"], "gv.supporting_g"))
    unknown = supporting - set(campaign.supporting_g)
    if unknown:
        raise GVMapError(
            f"run record supporting_g not declared for {campaign.campaign_id}: "
            f"{sorted(unknown)}"
        )
    gates = set(_str_tuple(gv["gates"], "gv.gates"))
    unknown_gates = gates - set(campaign.gates)
    if unknown_gates:
        raise GVMapError(
            f"run record gates not declared for {campaign.campaign_id}: "
            f"{sorted(unknown_gates)}"
        )
    for key in ("code_commit", "config_sha256", "data_manifest_sha256", "map_sha256"):
        value = gv.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise GVMapError(f"gv.{key} must be a string when present")
        if key.endswith("_sha256") and not _SHA256.match(value):
            raise GVMapError(f"gv.{key} must be a 64-character lowercase hex digest")


__all__ = [
    "CAMPAIGN_STATUSES",
    "DEFAULT_MAP_PATH",
    "GVMap",
    "GVMapError",
    "GV_MAP_SCHEMA",
    "CampaignEntry",
    "StageEntry",
    "load_gv_map",
    "resolve_campaign",
    "validate_gv_map",
    "validate_gv_run_record",
]
