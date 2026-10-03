"""Backend capability registry and effective-capability intersection (todo 16).

Implements design §9.1
(``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``): a versioned
registry of backend capability claims, an ``effective_capability`` intersection
over the engine / adapter / deployment layers, and typed refusal
(``BACKEND_CAPABILITY_MISSING``) for any request beyond the effective enabled
capability.

Registry file location (repo-convention decision)
-------------------------------------------------
``orca_capabilities_v1.json`` ships as **package data next to this module**
(``Path(__file__).resolve().with_name(...)``), not under ``config/``:

1. No module in this repo ships JSON package data today, so there is no
   conflicting precedent; ``config/`` holds exactly one file,
   ``defaults.yaml``, which is *user-mergeable* configuration
   (``--config`` deep-merge in ``config_loader``).
2. The capability registry is **not** user configuration.  It records what has
   actually been probed; a user YAML override must never be able to inflate
   it.  Keeping it package-adjacent makes that structural.
3. Schema ``orca_capabilities_v1`` is owned by ``contracts_v2`` in this same
   package; co-locating schema, registry, and intersection code keeps one
   ownership unit.
4. ``Path(__file__)`` resolution is cwd-independent — the same guarantee
   ``config_loader`` documents for ``PROJECT_ROOT``.

Field vocabulary alignment with contracts_v2
--------------------------------------------
Entries carry the ``BackendCapability`` OBJECT_FIELDS vocabulary of
``contracts_v2``.  The executable contract wins over informal aliases:

- ``point_limits`` is ``{baseline, max}`` (NOT ``{min, max, default}``).
- ``constraint_support`` is ``{native_scan, per_point_constraints,
  simul_scan}`` (NOT ``{cartesian, internal, per_point}``).
- Probe receipts carry ``{probe_id, date, engine_version, status: pass|fail,
  evidence_ref}``` plus optional ``adapter_version`` / ``modes`` /
  ``receipt_sha256``.  ``contracts_v2`` requires only ``probe_id`` + ``status``
  and rejects no extra receipt keys; the extra fields are this module's probe
  domain.

Enablement semantics (never optimistic)
---------------------------------------
- ``supported_modes`` = strict set intersection of the three layers'
  *declared* modes.  ``coordinate_kinds`` / ``max_scan_coordinates`` /
  ``point_limits`` / booleans intersect the same way (min / AND / max-of-
  baselines-and-min-of-maxes).
- ``method_element_coverage`` = strict three-way intersection: a method appears
  only if **every** layer declares it; effective elements = intersection of
  the three lists.  A layer that declares nothing contributes no methods.
- ``enabled_modes`` = declared modes further gated by **probe receipts**: a
  mode is enabled only when every layer carries at least one ``pass`` receipt
  covering that mode at the layer's own ``engine_version`` (and
  ``adapter_version`` when the receipt names one), and no layer carries a
  ``fail`` receipt covering it.  Receipts with empty/absent ``modes`` record
  evidence but enable nothing.  UNKNOWN = NOT ENABLED.
- The shipped baseline registry has **empty probe_receipts on every entry**:
  declarations are documented, enabled set is empty, and
  ``assert_mode_supported`` refuses every request until gated smoke tasks
  record receipts.  This honest initial state is the point of todo 16.
- The ACP adapter entry records code reality (single uniform B distance,
  ``max_scan_coordinates=1``); the contracts-surface "<=4 coordinates" is NOT
  executable capability and is never inflated into the registry.
- Native xTB PATH metadynamics (``xtb-native-path``) and ORCA-calling-xTB
  (``orca-xtb-bridge``) are separate entries with distinct ids and constraint
  semantics; xTB PATH declares no native scan coordinates
  (``max_scan_coordinates=0``, empty ``coordinate_kinds``).
- Per ADR-0002, xTB PATH executes **through ACP**: the ``xtb-native-path``
  and ``g2-path-adapter`` entries carry ``adapter="acp"`` (repointed from the
  retired local ``g2-path-runner``), and ``acp-xtb-path-adapter`` is the
  canonical adapter-layer record of the ACP-executed XTB_PATH method
  (``method_kind="XTB_PATH"`` planning-plane dispatch, compiled as a recipe).
  All three keep empty ``probe_receipts``: unprobed ACP xTB-path capability
  stays ``unknown``/refused until a real captured ACP version/executable sha
  smoke receipt exists — never silently enabled.
"""

# allow: SIZE_OK — plan-named todo-16 single capability-registry module
# (registry load + effective intersection + typed checks); precedent
# contracts_v2.py / registry.py / endpoint_context.py.

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pes2ts_core.generation.planning.contracts_v2 import (
    CAPABILITY_MODES,
    DRIVER_KINDS,
    MODE_COUPLED_1D,
    MODE_SCHEDULED_1D,
    MODE_SINGLE_1D,
    ContractError,
    make_backend_capability,
)
from pes2ts_core.utils.hashing import stable_json_dumps

# contracts_v2 inlines "PATH_NEB" in CAPABILITY_MODES without a named
# constant; defined locally here (contracts_v2.py must not be modified).
MODE_PATH_NEB: Final[str] = "PATH_NEB"

# ---------------------------------------------------------------------------
# Vocabulary and typed error codes.
# ---------------------------------------------------------------------------

BACKEND_CAPABILITY_MISSING: Final[str] = "BACKEND_CAPABILITY_MISSING"
CAPABILITY_REGISTRY_INVALID: Final[str] = "CAPABILITY_REGISTRY_INVALID"
CAPABILITY_REGISTRY_ENTRY_MISSING: Final[str] = "CAPABILITY_REGISTRY_ENTRY_MISSING"
CAPABILITY_LAYER_MISMATCH: Final[str] = "CAPABILITY_LAYER_MISMATCH"

REGISTRY_FILENAME: Final[str] = "orca_capabilities_v1.json"
REGISTRY_SCHEMA_VERSION: Final[str] = "orca_capabilities_v1_registry"
DEFAULT_REGISTRY_PATH: Final[Path] = Path(__file__).resolve().with_name(REGISTRY_FILENAME)

LAYERS: Final[tuple[str, ...]] = ("engine", "adapter", "deployment")
PROBE_STATUSES: Final[frozenset[str]] = frozenset({"pass", "fail"})

SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


class BackendCapabilityError(ValueError):
    """Typed capability refusal; ``code`` is a stable machine-readable token."""

    def __init__(self, code: str, reasons: Sequence[str]) -> None:
        self.code = code
        self.reasons = tuple(reasons)
        super().__init__(f"{code}: {'; '.join(self.reasons)}")


class CapabilityRegistryError(ValueError):
    """Typed registry load/validation failure."""

    def __init__(self, code: str, reasons: Sequence[str]) -> None:
        self.code = code
        self.reasons = tuple(reasons)
        super().__init__(f"{code}: {'; '.join(self.reasons)}")


# ---------------------------------------------------------------------------
# Value objects.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PointLimits:
    """Point-count window (``contracts_v2`` vocabulary: baseline + max)."""

    baseline: int
    max: int


@dataclass(frozen=True, slots=True)
class ConstraintSupport:
    """Per-field constraint claims (``contracts_v2`` three-boolean vocabulary)."""

    native_scan: bool
    per_point_constraints: bool
    simul_scan: bool


@dataclass(frozen=True, slots=True)
class ProbeReceipt:
    """Version smoke evidence for one (engine, adapter, modes) combination.

    ``modes`` scopes the receipt: only listed modes are attested.  An empty or
    absent ``modes`` list records evidence but enables nothing (never
    optimistic).
    """

    probe_id: str
    date: str
    engine_version: str
    status: str
    evidence_ref: str
    adapter_version: str | None = None
    modes: tuple[str, ...] = ()
    receipt_sha256: str | None = None

    def to_doc(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "probe_id": self.probe_id,
            "date": self.date,
            "engine_version": self.engine_version,
            "status": self.status,
            "evidence_ref": self.evidence_ref,
        }
        if self.adapter_version is not None:
            doc["adapter_version"] = self.adapter_version
        if self.modes:
            doc["modes"] = list(self.modes)
        if self.receipt_sha256 is not None:
            doc["receipt_sha256"] = self.receipt_sha256
        return doc

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> ProbeReceipt:
        if not isinstance(doc, Mapping):
            raise CapabilityRegistryError(
                CAPABILITY_REGISTRY_INVALID, ["probe receipt must be an object"]
            )
        issues: list[str] = []
        for field in ("probe_id", "date", "engine_version", "evidence_ref"):
            value = doc.get(field)
            if not isinstance(value, str) or not value:
                issues.append(f"{field}: must be a non-empty string")
        status = doc.get("status")
        if status not in PROBE_STATUSES:
            issues.append(f"status: must be one of {', '.join(sorted(PROBE_STATUSES))}")
        adapter_version = doc.get("adapter_version")
        if adapter_version is not None and (
            not isinstance(adapter_version, str) or not adapter_version
        ):
            issues.append("adapter_version: must be a non-empty string when present")
        raw_modes = doc.get("modes", [])
        modes: list[str] = []
        if raw_modes is not None:
            if not isinstance(raw_modes, list):
                issues.append("modes: must be an array")
            else:
                for mode in raw_modes:
                    if mode not in CAPABILITY_MODES:
                        issues.append(
                            f"modes: {mode!r} not in capability vocabulary "
                            f"{sorted(CAPABILITY_MODES)}"
                        )
                    else:
                        modes.append(str(mode))
        digest = doc.get("receipt_sha256")
        if digest is not None and (
            not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest)
        ):
            issues.append("receipt_sha256: must be a lowercase SHA256")
        if issues:
            raise CapabilityRegistryError(CAPABILITY_REGISTRY_INVALID, issues)
        return cls(
            probe_id=str(doc["probe_id"]),
            date=str(doc["date"]),
            engine_version=str(doc["engine_version"]),
            status=str(doc["status"]),
            evidence_ref=str(doc["evidence_ref"]),
            adapter_version=str(adapter_version) if adapter_version is not None else None,
            modes=tuple(modes),
            receipt_sha256=str(digest) if digest is not None else None,
        )


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    """One registry layer record: identity plus BackendCapability claims.

    ``method_element_coverage`` maps method -> sorted element tuple.  Treat the
    mapping as immutable by convention (frozen dataclass does not deep-freeze
    dict values).
    """

    entry_id: str
    layer: str
    engine: str
    adapter: str
    notes: str
    engine_version: str
    adapter_version: str
    supported_modes: tuple[str, ...]
    coordinate_kinds: tuple[str, ...]
    max_scan_coordinates: int
    point_limits: PointLimits
    custom_schedule_support: bool
    constraint_support: ConstraintSupport
    method_element_coverage: dict[str, tuple[str, ...]]
    probe_receipts: tuple[ProbeReceipt, ...]

    def to_doc(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "layer": self.layer,
            "engine": self.engine,
            "adapter": self.adapter,
            "capability": {
                "engine_version": self.engine_version,
                "adapter_version": self.adapter_version,
                "supported_modes": list(self.supported_modes),
                "coordinate_kinds": list(self.coordinate_kinds),
                "max_scan_coordinates": self.max_scan_coordinates,
                "point_limits": {
                    "baseline": self.point_limits.baseline,
                    "max": self.point_limits.max,
                },
                "custom_schedule_support": self.custom_schedule_support,
                "constraint_support": {
                    "native_scan": self.constraint_support.native_scan,
                    "per_point_constraints": self.constraint_support.per_point_constraints,
                    "simul_scan": self.constraint_support.simul_scan,
                },
                "method_element_coverage": {
                    method: list(elements)
                    for method, elements in self.method_element_coverage.items()
                },
                "probe_receipts": [receipt.to_doc() for receipt in self.probe_receipts],
            },
            "notes": self.notes,
        }

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> CapabilityEntry:
        """Parse one registry entry, validating via contracts_v2 projection."""
        entry_id = str(doc["entry_id"])
        capability = doc["capability"]
        try:
            make_backend_capability(entry_id, "active", **capability)
        except ContractError as exc:
            raise CapabilityRegistryError(
                CAPABILITY_REGISTRY_INVALID,
                [f"entry {entry_id!r} capability payload rejected by contracts_v2: {exc}"],
            ) from exc
        limits = capability["point_limits"]
        constraints = capability["constraint_support"]
        receipts: list[ProbeReceipt] = []
        for index, receipt_doc in enumerate(capability["probe_receipts"]):
            try:
                receipts.append(ProbeReceipt.from_doc(receipt_doc))
            except CapabilityRegistryError as exc:
                raise CapabilityRegistryError(
                    CAPABILITY_REGISTRY_INVALID,
                    [
                        f"entry {entry_id!r} probe_receipts[{index}]: {reason}"
                        for reason in exc.reasons
                    ],
                ) from exc
        coverage = {
            method: tuple(sorted(str(element) for element in elements))
            for method, elements in sorted(capability["method_element_coverage"].items())
        }
        return cls(
            entry_id=entry_id,
            layer=str(doc["layer"]),
            engine=str(doc["engine"]),
            adapter=str(doc["adapter"]),
            notes=str(doc.get("notes", "")),
            engine_version=str(capability["engine_version"]),
            adapter_version=str(capability["adapter_version"]),
            supported_modes=tuple(str(mode) for mode in capability["supported_modes"]),
            coordinate_kinds=tuple(str(kind) for kind in capability["coordinate_kinds"]),
            max_scan_coordinates=int(capability["max_scan_coordinates"]),
            point_limits=PointLimits(
                baseline=int(limits["baseline"]), max=int(limits["max"])
            ),
            custom_schedule_support=bool(capability["custom_schedule_support"]),
            constraint_support=ConstraintSupport(
                native_scan=bool(constraints["native_scan"]),
                per_point_constraints=bool(constraints["per_point_constraints"]),
                simul_scan=bool(constraints["simul_scan"]),
            ),
            method_element_coverage=coverage,
            probe_receipts=tuple(receipts),
        )


@dataclass(frozen=True, slots=True)
class CapabilityRegistry:
    """Loaded registry: ordered layer entries with identity lookup."""

    registry_id: str
    schema_version: str
    description: str
    entries: tuple[CapabilityEntry, ...]

    def entry(self, entry_id: str) -> CapabilityEntry:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise CapabilityRegistryError(
            CAPABILITY_REGISTRY_ENTRY_MISSING,
            [f"unknown registry entry {entry_id!r}"],
        )

    def to_doc(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "registry_id": self.registry_id,
            "description": self.description,
            "entries": [entry.to_doc() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class EffectiveCapability:
    """Intersection of engine/adapter/deployment claims plus probe gating.

    ``supported_modes`` is the declared intersection; ``enabled_modes`` is the
    probed-ready subset (empty on the shipped baseline).  ``disabled_reasons``
    records why each declared-but-unenabled mode is blocked.
    """

    engine_entry_id: str
    adapter_entry_id: str
    deployment_entry_id: str
    engine: str
    adapter: str
    engine_version: str
    adapter_version: str
    supported_modes: frozenset[str]
    enabled_modes: frozenset[str]
    coordinate_kinds: frozenset[str]
    max_scan_coordinates: int
    point_limits: PointLimits
    custom_schedule_support: bool
    constraint_support: ConstraintSupport
    method_element_coverage: dict[str, frozenset[str]]
    probe_receipts: tuple[ProbeReceipt, ...]
    disabled_reasons: tuple[str, ...]

    def to_doc(self) -> dict[str, Any]:
        return {
            "engine_entry_id": self.engine_entry_id,
            "adapter_entry_id": self.adapter_entry_id,
            "deployment_entry_id": self.deployment_entry_id,
            "engine": self.engine,
            "adapter": self.adapter,
            "engine_version": self.engine_version,
            "adapter_version": self.adapter_version,
            "supported_modes": sorted(self.supported_modes),
            "enabled_modes": sorted(self.enabled_modes),
            "coordinate_kinds": sorted(self.coordinate_kinds),
            "max_scan_coordinates": self.max_scan_coordinates,
            "point_limits": {
                "baseline": self.point_limits.baseline,
                "max": self.point_limits.max,
            },
            "custom_schedule_support": self.custom_schedule_support,
            "constraint_support": {
                "native_scan": self.constraint_support.native_scan,
                "per_point_constraints": self.constraint_support.per_point_constraints,
                "simul_scan": self.constraint_support.simul_scan,
            },
            "method_element_coverage": {
                method: sorted(elements)
                for method, elements in self.method_element_coverage.items()
            },
            "probe_receipts": [receipt.to_doc() for receipt in self.probe_receipts],
            "disabled_reasons": list(self.disabled_reasons),
        }


# ---------------------------------------------------------------------------
# Registry loading.
# ---------------------------------------------------------------------------


def _registry_issues(payload: Any) -> list[str]:
    """Envelope-level issues (entry payloads are checked by contracts_v2)."""
    if not isinstance(payload, dict):
        return ["$: registry root must be an object"]
    issues: list[str] = []
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        issues.append(f"$.schema_version: expected {REGISTRY_SCHEMA_VERSION!r}")
    registry_id = payload.get("registry_id")
    if not isinstance(registry_id, str) or not registry_id:
        issues.append("$.registry_id: must be a non-empty string")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        issues.append("$.entries: must be a non-empty array")
        return issues
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"$.entries[{index}]"
        if not isinstance(entry, dict):
            issues.append(f"{path}: must be an object")
            continue
        entry_id = entry.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            issues.append(f"{path}.entry_id: must be a non-empty string")
        elif entry_id in seen:
            issues.append(f"{path}.entry_id: duplicate entry identity {entry_id!r}")
        else:
            seen.add(entry_id)
        layer = entry.get("layer")
        if layer not in LAYERS:
            issues.append(f"{path}.layer: must be one of {', '.join(LAYERS)}")
        for field in ("engine", "adapter"):
            value = entry.get(field)
            if not isinstance(value, str) or not value:
                issues.append(f"{path}.{field}: must be a non-empty string")
        if not isinstance(entry.get("capability"), dict):
            issues.append(f"{path}.capability: must be an object")
    return issues


def _registry_from_doc(payload: Mapping[str, Any]) -> CapabilityRegistry:
    entries = tuple(CapabilityEntry.from_doc(doc) for doc in payload["entries"])
    return CapabilityRegistry(
        registry_id=str(payload["registry_id"]),
        schema_version=str(payload["schema_version"]),
        description=str(payload.get("description", "")),
        entries=entries,
    )


def load_capability_registry(path: Path | str | None = None) -> CapabilityRegistry:
    """Load and validate the capability registry.

    ``path=None`` loads the shipped package-adjacent baseline
    (``orca_capabilities_v1.json``).  An explicit path loads a deployment
    overlay or test fixture.  Deterministic read: file text -> json.loads ->
    contracts_v2 projection per entry.
    """
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    try:
        text = registry_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CapabilityRegistryError(
            CAPABILITY_REGISTRY_INVALID,
            [f"cannot read registry {registry_path}: {exc}"],
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CapabilityRegistryError(
            CAPABILITY_REGISTRY_INVALID,
            [f"registry {registry_path} is not valid JSON: {exc}"],
        ) from exc
    issues = _registry_issues(payload)
    if issues:
        raise CapabilityRegistryError(CAPABILITY_REGISTRY_INVALID, issues)
    return _registry_from_doc(payload)


def dumps_capability_registry(registry: CapabilityRegistry) -> str:
    """Serialize a registry to stable JSON (deterministic key order)."""
    return stable_json_dumps(registry.to_doc())


def loads_capability_registry(payload: str) -> CapabilityRegistry:
    """Parse, validate, and build a registry from stable JSON."""
    try:
        doc = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CapabilityRegistryError(
            CAPABILITY_REGISTRY_INVALID, [str(exc)]
        ) from exc
    issues = _registry_issues(doc)
    if issues:
        raise CapabilityRegistryError(CAPABILITY_REGISTRY_INVALID, issues)
    return _registry_from_doc(doc)


# ---------------------------------------------------------------------------
# Effective capability intersection.
# ---------------------------------------------------------------------------


def _resolve_capability(
    ref: str | CapabilityEntry,
    registry: CapabilityRegistry | None,
    expected_layer: str,
) -> CapabilityEntry:
    if isinstance(ref, CapabilityEntry):
        entry = ref
    else:
        reg = registry if registry is not None else load_capability_registry()
        entry = reg.entry(str(ref))
    if entry.layer != expected_layer:
        raise CapabilityRegistryError(
            CAPABILITY_LAYER_MISMATCH,
            [
                f"entry {entry.entry_id!r} has layer {entry.layer!r}, "
                f"expected {expected_layer!r}"
            ],
        )
    return entry


def _receipt_covers(receipt: ProbeReceipt, layer: CapabilityEntry, mode: str) -> bool:
    if receipt.engine_version != layer.engine_version:
        return False
    if receipt.adapter_version is not None and receipt.adapter_version != layer.adapter_version:
        return False
    return mode in receipt.modes


def _mode_smoke_state(layer: CapabilityEntry, mode: str) -> tuple[str, str]:
    """Return ``("pass"|"fail"|"unprobed", detail)`` for *mode* on *layer*."""
    covering = [r for r in layer.probe_receipts if _receipt_covers(r, layer, mode)]
    if not covering:
        return (
            "unprobed",
            f"no probe receipt covers {mode} at engine_version {layer.engine_version}",
        )
    if any(receipt.status == "fail" for receipt in covering):
        return (
            "fail",
            f"failed probe receipt covers {mode} at engine_version {layer.engine_version}",
        )
    return "pass", ""


def effective_capability(
    engine: str | CapabilityEntry,
    adapter: str | CapabilityEntry,
    deployment: str | CapabilityEntry,
    *,
    registry: CapabilityRegistry | None = None,
) -> EffectiveCapability:
    """Intersect engine/adapter/deployment capability claims.

    Arguments are registry entry ids (resolved against *registry*, defaulting
    to the shipped baseline) or inline :class:`CapabilityEntry` objects.  Each
    reference must match its expected layer.  Declared fields intersect
    strictly; ``enabled_modes`` additionally requires passing probe receipts on
    every layer at the effective versions (empty on the shipped baseline).
    """
    eng = _resolve_capability(engine, registry, "engine")
    adp = _resolve_capability(adapter, registry, "adapter")
    dep = _resolve_capability(deployment, registry, "deployment")
    layers = (eng, adp, dep)

    supported = set(eng.supported_modes)
    for layer in (adp, dep):
        supported &= set(layer.supported_modes)
    kinds = set(eng.coordinate_kinds) & set(adp.coordinate_kinds) & set(dep.coordinate_kinds)
    max_coords = min(layer.max_scan_coordinates for layer in layers)
    baseline = max(layer.point_limits.baseline for layer in layers)
    max_points = min(layer.point_limits.max for layer in layers)
    custom_schedule = all(layer.custom_schedule_support for layer in layers)
    constraints = ConstraintSupport(
        native_scan=all(layer.constraint_support.native_scan for layer in layers),
        per_point_constraints=all(
            layer.constraint_support.per_point_constraints for layer in layers
        ),
        simul_scan=all(layer.constraint_support.simul_scan for layer in layers),
    )

    # Strict three-way method coverage: a method appears only if every layer
    # declares it; effective elements = intersection of the three lists.
    common_methods = set(eng.method_element_coverage)
    for layer in (adp, dep):
        common_methods &= set(layer.method_element_coverage)
    method_coverage = {
        method: frozenset(
            set(eng.method_element_coverage[method])
            & set(adp.method_element_coverage[method])
            & set(dep.method_element_coverage[method])
        )
        for method in sorted(common_methods)
    }

    enabled: set[str] = set()
    disabled: list[str] = []
    for mode in sorted(supported):
        states = {layer.entry_id: _mode_smoke_state(layer, mode) for layer in layers}
        if all(state == "pass" for state, _ in states.values()):
            enabled.add(mode)
        else:
            details = "; ".join(
                f"{entry_id} [{state}]: {detail}"
                for entry_id, (state, detail) in states.items()
                if state != "pass"
            )
            disabled.append(f"{mode}: {details}")
    if not supported:
        disabled.append("declared engine/adapter/deployment intersection is empty")

    all_receipts = tuple(
        receipt for layer in layers for receipt in layer.probe_receipts
    )
    return EffectiveCapability(
        engine_entry_id=eng.entry_id,
        adapter_entry_id=adp.entry_id,
        deployment_entry_id=dep.entry_id,
        engine=eng.engine,
        adapter=adp.adapter,
        engine_version=eng.engine_version,
        adapter_version=adp.adapter_version,
        supported_modes=frozenset(supported),
        enabled_modes=frozenset(enabled),
        coordinate_kinds=frozenset(kinds),
        max_scan_coordinates=max_coords,
        point_limits=PointLimits(baseline=baseline, max=max_points),
        custom_schedule_support=custom_schedule,
        constraint_support=constraints,
        method_element_coverage=method_coverage,
        probe_receipts=all_receipts,
        disabled_reasons=tuple(disabled),
    )


# ---------------------------------------------------------------------------
# Capability checks (proposal-stage gate; todo 17 consumes these).
# ---------------------------------------------------------------------------


def _structural_mode_problems(
    cap: EffectiveCapability, mode: str, kinds: Sequence[str], n_coords: int
) -> list[str]:
    problems: list[str] = []
    if mode not in CAPABILITY_MODES:
        return [
            f"mode {mode!r} not in capability vocabulary {sorted(CAPABILITY_MODES)}"
        ]
    kind_list = list(kinds)
    if mode == MODE_PATH_NEB:
        if n_coords != 0 or kind_list:
            problems.append(
                "PATH_NEB takes no native scan coordinates "
                "(n_coords must be 0, kinds empty)"
            )
        return problems
    if n_coords < 1:
        problems.append(f"n_coords must be >= 1, got {n_coords}")
    if len(kind_list) != n_coords:
        problems.append(f"len(kinds)={len(kind_list)} must equal n_coords={n_coords}")
    invalid = sorted({kind for kind in kind_list if kind not in DRIVER_KINDS})
    if invalid:
        problems.append(
            f"coordinate kinds {invalid} not in vocabulary {sorted(DRIVER_KINDS)}"
        )
    unsupported = sorted(
        {
            kind
            for kind in kind_list
            if kind in DRIVER_KINDS and kind not in cap.coordinate_kinds
        }
    )
    if unsupported:
        problems.append(
            f"coordinate kinds {unsupported} not in effective coordinate_kinds "
            f"{sorted(cap.coordinate_kinds)}"
        )
    if n_coords > cap.max_scan_coordinates:
        problems.append(
            f"n_coords={n_coords} exceeds effective "
            f"max_scan_coordinates={cap.max_scan_coordinates}"
        )
    if mode == MODE_SINGLE_1D and n_coords != 1:
        problems.append(f"SINGLE_1D requires exactly one coordinate, got {n_coords}")
    if mode in (MODE_COUPLED_1D, MODE_SCHEDULED_1D):
        upper = cap.max_scan_coordinates
        if not (2 <= n_coords <= upper):
            problems.append(f"{mode} requires 2..{upper} coordinates, got {n_coords}")
    return problems


def _enablement_problems(cap: EffectiveCapability, mode: str) -> list[str]:
    if mode not in CAPABILITY_MODES:
        return []
    if mode not in cap.supported_modes:
        return [
            f"mode {mode!r} not in declared engine/adapter/deployment intersection "
            f"{sorted(cap.supported_modes)}"
        ]
    if mode not in cap.enabled_modes:
        return [
            f"mode {mode!r} declared but not enabled (unprobed or failed smoke); "
            "passing probe receipts at the effective versions are required"
        ]
    return []


def assert_mode_supported(
    cap: EffectiveCapability, mode: str, kinds: Sequence[str], n_coords: int
) -> None:
    """Refuse any mode/kind/arity request beyond the enabled capability.

    Raises :class:`BackendCapabilityError` with code
    ``BACKEND_CAPABILITY_MISSING`` when the request is structurally invalid or
    the mode is not enabled (declared-but-unprobed counts as not enabled).
    """
    problems = _structural_mode_problems(cap, mode, kinds, n_coords) + (
        _enablement_problems(cap, mode)
    )
    if problems:
        raise BackendCapabilityError(BACKEND_CAPABILITY_MISSING, problems)


def capability_check(
    cap: EffectiveCapability, mode: str, kinds: Sequence[str], n_coords: int
) -> dict[str, Any]:
    """Contracts-shaped ``capability_check`` dict for proposal candidates.

    ``status`` is ``"pass"`` only when the request is structurally valid AND
    the mode is enabled (probed).  Declared-but-unprobed modes report
    ``"unknown"`` — unsmoked combinations can never yield ready.  Structural
    violations and undeclared modes report ``"fail"``.
    """
    structural = _structural_mode_problems(cap, mode, kinds, n_coords)
    enablement = _enablement_problems(cap, mode)
    undeclared = mode in CAPABILITY_MODES and mode not in cap.supported_modes
    if structural or undeclared:
        return {"status": "fail", "missing": structural + enablement}
    if enablement:
        return {"status": "unknown", "missing": enablement}
    return {"status": "pass", "missing": []}


__all__ = [
    "BACKEND_CAPABILITY_MISSING",
    "CAPABILITY_LAYER_MISMATCH",
    "CAPABILITY_REGISTRY_ENTRY_MISSING",
    "CAPABILITY_REGISTRY_INVALID",
    "DEFAULT_REGISTRY_PATH",
    "REGISTRY_FILENAME",
    "REGISTRY_SCHEMA_VERSION",
    "BackendCapabilityError",
    "CapabilityEntry",
    "CapabilityRegistry",
    "CapabilityRegistryError",
    "ConstraintSupport",
    "EffectiveCapability",
    "PointLimits",
    "ProbeReceipt",
    "assert_mode_supported",
    "capability_check",
    "dumps_capability_registry",
    "effective_capability",
    "load_capability_registry",
    "loads_capability_registry",
]
