"""Chemical target-reaction quality layer with four independent result tiers (todo 22).

Design §11.2 (``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md``):

1. ``execution_complete`` — job and files complete.
2. ``numerically_usable`` — per-frame convergence, constraints, coordinates,
   energies and methods consistent; no missing frames or obvious geometry errors.
3. ``target_path_compatible`` — undriven target edits also complete; terminal
   frames match the target R/P equivalence class and configuration; no
   off-target reaction. Path continuity and in-path side reactions are checked.
4. ``validated_ts`` — separate TS optimization, first-order imaginary frequency
   matching the target motion, and two-way IRC / endpoint verification.

Hard rules encoded here (plan todo 22):

- Bond length is a **continuous** quantity; a fixed integer bond-order threshold
  is never the completion criterion.  A bond in the transition region is labelled
  ``unknown/transition`` — never forced into a binary formed/broken verdict.
- A valid association/dissociation path with **no internal energy peak** is not
  reported as TS; the highest **endpoint** energy is never reported as TS.
- ``validated_ts`` requires typed **evidence references** (OptTS + first-order
  imaginary frequency + two-way IRC).  Without evidence the tier stays
  not-achieved and lists exactly which artifacts are missing.
- Technical ``usable`` ≠ chemical target completion: tiers are recorded
  independently.  A numerically usable path with unfinished target edits has
  ``numerically_usable=true`` and ``target_path_compatible=false``.

Evidence references are opaque strings (receipt paths / object ids) — never
TS/IRC geometries or energies.  ``to_doc`` is scanned against
``FORBIDDEN_TRUTH_KEYS ∪ FORBIDDEN_EXPORT_KEYS``.

New module — frozen single-distance surfaces (``trajectory.py``, ``quality.py``,
``contracts.py``, ``adapter.py``) are **not** modified.  ``quality.py`` keeps
its technical ``usable`` semantics and ``physical_validation="not_run"``; this
module is the additive chemical layer.
"""

# allow: SIZE_OK — plan-named todo-22 single target-path quality module
# (design §11.2 four-tier surface); precedent frame_recovery.py todo 21 /
# compile_orca.py todo 19.

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from pes2ts_core.g1.v2_verify import FORBIDDEN_EXPORT_KEYS
from pes2ts_core.integration.acp.frame_recovery import (
    FrameRecoveryReport,
    MonitorRecord,
)
from pes2ts_core.scan_strategy.geometry_feasibility import covalent_radius_sum
from pes2ts_core.utils.hashing import stable_json_dumps

__all__ = [
    "BOND_BROKEN",
    "BOND_FORMED",
    "BOND_NO_ORDER_EVIDENCE",
    "BOND_NOT_FORMED",
    "BOND_STILL_BONDED",
    "BOND_TRANSITION",
    "CODE_BOND_INCOMPLETE",
    "CODE_BOND_MEASUREMENT_MISSING",
    "CODE_CONTINUITY_GAP",
    "CODE_DRIVER_RESIDUAL_MISSING",
    "CODE_ENDPOINT_MAX_NOT_TS",
    "CODE_EVIDENCE_INCOMPLETE",
    "CODE_ENERGY_INCOMPARABLE",
    "CODE_ENERGY_INCOMPLETE",
    "CODE_CONSTRAINT_RESIDUAL_EXCEEDED",
    "CODE_FRAME_JUMP",
    "CODE_NO_FRAMES",
    "CODE_NO_INTERNAL_PEAK",
    "CODE_NOT_CONVERGED",
    "CODE_ORDER_EVIDENCE_MISSING",
    "CODE_REGION_INCOMPLETE",
    "CODE_REGION_MEASUREMENT_MISSING",
    "CODE_REGION_ORDER_EVIDENCE_MISSING",
    "CODE_STEREO_AT_REACTANT",
    "CODE_STEREO_LABEL_MISSING",
    "CODE_STEREO_UNEXPECTED",
    "CODE_TARGET_INCOMPLETE",
    "DEFAULT_BOND_TOLERANCE_ANGSTROM",
    "DEFAULT_TRANSITION_WIDTH_ANGSTROM",
    "EDIT_BROKEN",
    "EDIT_FORMED",
    "EDIT_ORDER_CHANGED",
    "EVIDENCE_FIRST_ORDER_IMAGINARY",
    "EVIDENCE_IRC_FORWARD",
    "EVIDENCE_IRC_REVERSE",
    "EVIDENCE_OPTTS",
    "EVIDENCE_REQUIRED_KEYS",
    "FORBIDDEN_ASSESSMENT_KEYS",
    "LABEL_ACHIEVED",
    "LABEL_NOT_ACHIEVED",
    "LABEL_NOT_EVALUABLE",
    "PATH_CLASS_ENERGIES_UNAVAILABLE",
    "PATH_CLASS_INTERNAL_MAXIMUM",
    "PATH_CLASS_NO_INTERNAL_PEAK",
    "SCHEMA_TARGET_PATH_ASSESSMENT",
    "TIER_EXECUTION_COMPLETE",
    "TIER_NUMERICALLY_USABLE",
    "TIER_TARGET_PATH_COMPATIBLE",
    "TIER_VALIDATED_TS",
    "TIERS",
    "TargetBondDefinition",
    "TargetDefinitionBundle",
    "TargetPathAssessment",
    "TargetPathError",
    "TargetRegionDefinition",
    "TargetStereoDefinition",
    "TierResult",
    "TsEvidenceBundle",
    "assess_target_path",
    "parse_evidence_bundle",
    "parse_target_definitions",
]

# ---------------------------------------------------------------------------
# Vocabulary.
# ---------------------------------------------------------------------------
SCHEMA_TARGET_PATH_ASSESSMENT: Final[str] = "pes2ts_target_path_assessment_v1"

TIER_EXECUTION_COMPLETE: Final[str] = "execution_complete"
TIER_NUMERICALLY_USABLE: Final[str] = "numerically_usable"
TIER_TARGET_PATH_COMPATIBLE: Final[str] = "target_path_compatible"
TIER_VALIDATED_TS: Final[str] = "validated_ts"
TIERS: Final[tuple[str, ...]] = (
    TIER_EXECUTION_COMPLETE,
    TIER_NUMERICALLY_USABLE,
    TIER_TARGET_PATH_COMPATIBLE,
    TIER_VALIDATED_TS,
)

LABEL_ACHIEVED: Final[str] = "achieved"
LABEL_NOT_ACHIEVED: Final[str] = "not_achieved"
LABEL_NOT_EVALUABLE: Final[str] = "not_evaluable"

EDIT_FORMED: Final[str] = "formed"
EDIT_BROKEN: Final[str] = "broken"
EDIT_ORDER_CHANGED: Final[str] = "order_changed"
EDIT_KINDS: Final[tuple[str, ...]] = (EDIT_FORMED, EDIT_BROKEN, EDIT_ORDER_CHANGED)

#: Continuous bond labels — never a forced binary verdict in the transition zone.
BOND_FORMED: Final[str] = "formed"
BOND_BROKEN: Final[str] = "broken"
BOND_TRANSITION: Final[str] = "unknown/transition"
BOND_STILL_BONDED: Final[str] = "still_bonded"
BOND_NOT_FORMED: Final[str] = "not_formed"
BOND_NO_ORDER_EVIDENCE: Final[str] = "bonded_order_unverified"

#: Typed reason / evidence tokens.
CODE_NO_FRAMES: Final[str] = "NO_FRAMES"
CODE_NOT_CONVERGED: Final[str] = "NOT_ALL_FRAMES_CONVERGED"
CODE_DRIVER_RESIDUAL_MISSING: Final[str] = "DRIVER_RESIDUAL_MISSING"
CODE_CONTINUITY_GAP: Final[str] = "CONTINUITY_GAP"
CODE_FRAME_JUMP: Final[str] = "FRAME_INDEX_JUMP"
CODE_BOND_MEASUREMENT_MISSING: Final[str] = "BOND_MEASUREMENT_MISSING"
CODE_BOND_INCOMPLETE: Final[str] = "TARGET_BOND_INCOMPLETE"
CODE_ORDER_EVIDENCE_MISSING: Final[str] = "ORDER_EVIDENCE_MISSING"
CODE_REGION_MEASUREMENT_MISSING: Final[str] = "REGION_MEASUREMENT_MISSING"
CODE_REGION_INCOMPLETE: Final[str] = "REGION_INCOMPLETE"
CODE_REGION_ORDER_EVIDENCE_MISSING: Final[str] = "REGION_ORDER_EVIDENCE_MISSING"
CODE_STEREO_LABEL_MISSING: Final[str] = "STEREO_LABEL_MISSING"
CODE_STEREO_AT_REACTANT: Final[str] = "STEREO_AT_REACTANT"
CODE_STEREO_UNEXPECTED: Final[str] = "STEREO_UNEXPECTED"
CODE_TARGET_INCOMPLETE: Final[str] = "TARGET_PATH_INCOMPLETE"
CODE_CONSTRAINT_RESIDUAL_EXCEEDED: Final[str] = "CONSTRAINT_RESIDUAL_EXCEEDED"
CODE_ENERGY_INCOMPLETE: Final[str] = "SCAN_ENERGY_INCOMPLETE"
CODE_ENERGY_INCOMPARABLE: Final[str] = "SCAN_ENERGY_CHANNEL_INCOMPARABLE"
CODE_EVIDENCE_INCOMPLETE: Final[str] = "TS_EVIDENCE_INCOMPLETE"
CODE_ENDPOINT_MAX_NOT_TS: Final[str] = "ENDPOINT_MAX_NOT_TS"
CODE_NO_INTERNAL_PEAK: Final[str] = "NO_INTERNAL_PEAK_TS_SEED"

#: validated_ts evidence artifacts (opaque references only).
EVIDENCE_OPTTS: Final[str] = "optts_ref"
EVIDENCE_FIRST_ORDER_IMAGINARY: Final[str] = "first_order_imaginary_ref"
EVIDENCE_IRC_FORWARD: Final[str] = "irc_forward_ref"
EVIDENCE_IRC_REVERSE: Final[str] = "irc_reverse_ref"
EVIDENCE_REQUIRED_KEYS: Final[tuple[str, ...]] = (
    EVIDENCE_OPTTS,
    EVIDENCE_FIRST_ORDER_IMAGINARY,
    EVIDENCE_IRC_FORWARD,
    EVIDENCE_IRC_REVERSE,
)

PATH_CLASS_INTERNAL_MAXIMUM: Final[str] = "internal_maximum"
PATH_CLASS_ENDPOINT_MAX: Final[str] = "endpoint_max"
PATH_CLASS_NO_INTERNAL_PEAK: Final[str] = "no_internal_peak"
PATH_CLASS_ENERGIES_UNAVAILABLE: Final[str] = "energies_unavailable"

#: Continuous-distance policy defaults (Å).  TODO: calibrate (待校准).
DEFAULT_BOND_TOLERANCE_ANGSTROM: Final[float] = 0.45
DEFAULT_TRANSITION_WIDTH_ANGSTROM: Final[float] = 0.5
#: Max |target − actual| still counted as constraint-satisfied (Å).
DEFAULT_CONSTRAINT_TOLERANCE_ANGSTROM: Final[float] = 0.05

FORBIDDEN_ASSESSMENT_KEYS: Final[frozenset[str]] = frozenset(
    key.lower()
    for key in (
        {
            "ts_coordinates", "ts_geometry", "ts_energy", "irc_frames",
            "irc_coordinates", "endpoint_match", "orientation", "irc_evidence",
            "validation_label", "target_ts", "ts_irc_index", "EHG", "ehg",
            "forces",
        }
        | FORBIDDEN_EXPORT_KEYS
    )
)


class TargetPathError(ValueError):
    """Typed refusal; ``code`` is a stable machine-readable token."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


# ---------------------------------------------------------------------------
# Boundary value objects (parse-don't-validate).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TargetBondDefinition:
    """One chemical target edit judged by continuous bond length (Å)."""

    target_id: str
    atom_maps: tuple[int, int]
    edit_kind: str
    element_a: str
    element_b: str
    r_distance_angstrom: float | None = None
    p_distance_angstrom: float | None = None
    driver_id: str | None = None


@dataclass(frozen=True, slots=True)
class TargetRegionDefinition:
    """One target region (e.g. aromatic reorganization) with member bonds."""

    target_id: str
    region_id: str
    bond_targets: tuple[TargetBondDefinition, ...] = ()
    requires_electronic_evidence: bool = False


@dataclass(frozen=True, slots=True)
class TargetStereoDefinition:
    """One stereo target judged by terminal configuration label."""

    target_id: str
    center_maps: tuple[int, ...]
    r_label: str | None
    p_label: str
    final_label: str | None = None


@dataclass(frozen=True, slots=True)
class TargetDefinitionBundle:
    """Parsed target-reaction definitions for one candidate path."""

    bonds: tuple[TargetBondDefinition, ...]
    regions: tuple[TargetRegionDefinition, ...]
    stereo: tuple[TargetStereoDefinition, ...]
    bond_tolerance_angstrom: float = DEFAULT_BOND_TOLERANCE_ANGSTROM
    transition_width_angstrom: float = DEFAULT_TRANSITION_WIDTH_ANGSTROM
    constraint_tolerance_angstrom: float = DEFAULT_CONSTRAINT_TOLERANCE_ANGSTROM
    #: target_id → opaque electronic bond-order evidence ref (order_changed).
    electronic_order_evidence: Mapping[str, str] = field(default_factory=dict)
    #: target_id / region_id → opaque stereo / region evidence ref.
    stereo_evidence: Mapping[str, str] = field(default_factory=dict)
    region_evidence: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TsEvidenceBundle:
    """Typed evidence references for the validated_ts tier (opaque strings)."""

    optts_ref: str | None = None
    first_order_imaginary_ref: str | None = None
    irc_forward_ref: str | None = None
    irc_reverse_ref: str | None = None

    def missing_keys(self) -> tuple[str, ...]:
        return tuple(
            key
            for key, value in (
                (EVIDENCE_OPTTS, self.optts_ref),
                (EVIDENCE_FIRST_ORDER_IMAGINARY, self.first_order_imaginary_ref),
                (EVIDENCE_IRC_FORWARD, self.irc_forward_ref),
                (EVIDENCE_IRC_REVERSE, self.irc_reverse_ref),
            )
            if not (isinstance(value, str) and value.strip())
        )

    def to_record(self) -> dict[str, str | None]:
        return {
            EVIDENCE_OPTTS: self.optts_ref,
            EVIDENCE_FIRST_ORDER_IMAGINARY: self.first_order_imaginary_ref,
            EVIDENCE_IRC_FORWARD: self.irc_forward_ref,
            EVIDENCE_IRC_REVERSE: self.irc_reverse_ref,
        }


@dataclass(frozen=True, slots=True)
class TierResult:
    """One independently recorded quality tier with typed reasons."""

    tier: str
    status: str
    reason_codes: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "missing_evidence": list(self.missing_evidence),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class BondLabelRecord:
    """One target bond judged at the terminal frame with a continuous label."""

    target_id: str
    atom_maps: tuple[int, int]
    edit_kind: str
    label: str
    end_distance_angstrom: float | None
    radius_sum_angstrom: float
    bonded_upper_angstrom: float
    separated_lower_angstrom: float
    completed: bool
    reason_codes: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "atom_maps": list(self.atom_maps),
            "edit_kind": self.edit_kind,
            "label": self.label,
            "end_distance_angstrom": self.end_distance_angstrom,
            "radius_sum_angstrom": self.radius_sum_angstrom,
            "bonded_upper_angstrom": self.bonded_upper_angstrom,
            "separated_lower_angstrom": self.separated_lower_angstrom,
            "completed": self.completed,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class PathShapeDiagnosis:
    """Energy-profile shape diagnostics — never a TS claim by itself."""

    n_frames: int
    energies_available: bool
    max_frame_index: int | None
    max_is_endpoint: bool
    has_internal_maximum: bool
    classification: str

    def to_record(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "energies_available": self.energies_available,
            "max_frame_index": self.max_frame_index,
            "max_is_endpoint": self.max_is_endpoint,
            "has_internal_maximum": self.has_internal_maximum,
            "classification": self.classification,
        }


@dataclass(frozen=True, slots=True)
class TargetPathAssessment:
    """Four-tier chemical quality assessment of one recovered path."""

    schema_version: str
    candidate_id: str
    execution_id: str
    tiers: Mapping[str, TierResult]
    checks: Mapping[str, TierResult]
    bond_labels: tuple[BondLabelRecord, ...]
    path_shape: PathShapeDiagnosis
    evidence: Mapping[str, str | None]
    ts_candidate_from_path_shape: bool
    source: Mapping[str, Any]

    def to_doc(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "execution_id": self.execution_id,
            "tiers": {name: self.tiers[name].to_record() for name in TIERS},
            "checks": {name: self.checks[name].to_record() for name in sorted(self.checks)},
            "bond_labels": [row.to_record() for row in self.bond_labels],
            "path_shape": self.path_shape.to_record(),
            "evidence": dict(self.evidence),
            "ts_candidate_from_path_shape": self.ts_candidate_from_path_shape,
            "source": dict(self.source),
            "semantics": {
                "tiers_independent": True,
                "technical_usable_is_not_chemical_completion": True,
                "bond_labels_continuous": True,
                "transition_label": BOND_TRANSITION,
                "validated_ts_requires_evidence": list(EVIDENCE_REQUIRED_KEYS),
                "endpoint_max_never_ts": True,
                "no_internal_peak_not_ts": True,
            },
        }

    def to_json(self) -> str:
        return stable_json_dumps(self.to_doc())


# ---------------------------------------------------------------------------
# Boundary parsing.
# ---------------------------------------------------------------------------
def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_float(value: Any, name: str) -> float:
    number = _finite_number(value)
    if number is None or number <= 0:
        raise TargetPathError("TARGET_POLICY_INVALID", f"{name} must be finite and positive")
    return number


def _ref_or_none(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TargetPathError("TARGET_POLICY_INVALID", f"{name} must be a non-empty string or null")
    return value.strip()


def _parse_bond(raw: Any, position: str) -> TargetBondDefinition:
    if isinstance(raw, TargetBondDefinition):
        return raw
    if not isinstance(raw, Mapping):
        raise TargetPathError("TARGET_DEFINITION_INVALID", f"{position} must be a mapping")
    target_id = raw.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        raise TargetPathError("TARGET_DEFINITION_INVALID", f"{position}.target_id must be a non-empty string")
    atom_maps = raw.get("atom_maps")
    if (not isinstance(atom_maps, (list, tuple)) or len(atom_maps) != 2
            or not all(_is_int(item) and item > 0 for item in atom_maps)):
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID",
            f"{position}.atom_maps must be a pair of positive integers",
        )
    edit_kind = raw.get("edit_kind")
    if edit_kind not in EDIT_KINDS:
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID",
            f"{position}.edit_kind must be one of {list(EDIT_KINDS)}, got {edit_kind!r}",
        )
    element_a = raw.get("element_a")
    element_b = raw.get("element_b")
    if not isinstance(element_a, str) or not element_a or not isinstance(element_b, str) or not element_b:
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID", f"{position}.element_a/element_b must be non-empty strings"
        )
    r_distance = _finite_number(raw.get("r_distance_angstrom"))
    p_distance = _finite_number(raw.get("p_distance_angstrom"))
    driver_id = raw.get("driver_id")
    if driver_id is not None and (not isinstance(driver_id, str) or not driver_id):
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID", f"{position}.driver_id must be a non-empty string or null"
        )
    return TargetBondDefinition(
        target_id=target_id,
        atom_maps=(int(atom_maps[0]), int(atom_maps[1])),
        edit_kind=str(edit_kind),
        element_a=element_a,
        element_b=element_b,
        r_distance_angstrom=r_distance,
        p_distance_angstrom=p_distance,
        driver_id=driver_id,
    )


def _parse_region(raw: Any, position: str) -> TargetRegionDefinition:
    if isinstance(raw, TargetRegionDefinition):
        return raw
    if not isinstance(raw, Mapping):
        raise TargetPathError("TARGET_DEFINITION_INVALID", f"{position} must be a mapping")
    target_id = raw.get("target_id")
    region_id = raw.get("region_id")
    if not isinstance(target_id, str) or not target_id:
        raise TargetPathError("TARGET_DEFINITION_INVALID", f"{position}.target_id must be a non-empty string")
    if not isinstance(region_id, str) or not region_id:
        raise TargetPathError("TARGET_DEFINITION_INVALID", f"{position}.region_id must be a non-empty string")
    member_raw = raw.get("bond_targets", ())
    if not isinstance(member_raw, (list, tuple)):
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID", f"{position}.bond_targets must be an array"
        )
    members = tuple(
        _parse_bond(item, f"{position}.bond_targets[{index}]")
        for index, item in enumerate(member_raw)
    )
    requires_electronic = raw.get("requires_electronic_evidence", False)
    if not isinstance(requires_electronic, bool):
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID",
            f"{position}.requires_electronic_evidence must be a boolean",
        )
    return TargetRegionDefinition(
        target_id=target_id,
        region_id=region_id,
        bond_targets=members,
        requires_electronic_evidence=requires_electronic,
    )


def _parse_stereo(raw: Any, position: str) -> TargetStereoDefinition:
    if isinstance(raw, TargetStereoDefinition):
        return raw
    if not isinstance(raw, Mapping):
        raise TargetPathError("TARGET_DEFINITION_INVALID", f"{position} must be a mapping")
    target_id = raw.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        raise TargetPathError("TARGET_DEFINITION_INVALID", f"{position}.target_id must be a non-empty string")
    center_maps = raw.get("center_maps")
    if (not isinstance(center_maps, (list, tuple)) or not center_maps
            or not all(_is_int(item) and item > 0 for item in center_maps)):
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID",
            f"{position}.center_maps must be a non-empty array of positive integers",
        )
    p_label = raw.get("p_label")
    if not isinstance(p_label, str) or not p_label:
        raise TargetPathError("TARGET_DEFINITION_INVALID", f"{position}.p_label must be a non-empty string")
    r_label = raw.get("r_label")
    if r_label is not None and (not isinstance(r_label, str) or not r_label):
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID", f"{position}.r_label must be a non-empty string or null"
        )
    final_label = raw.get("final_label")
    if final_label is not None and (not isinstance(final_label, str) or not final_label):
        raise TargetPathError(
            "TARGET_DEFINITION_INVALID", f"{position}.final_label must be a non-empty string or null"
        )
    return TargetStereoDefinition(
        target_id=target_id,
        center_maps=tuple(int(item) for item in center_maps),
        r_label=r_label,
        p_label=p_label,
        final_label=final_label,
    )


def _parse_evidence_map(raw: Any, name: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TargetPathError("TARGET_POLICY_INVALID", f"{name} must be a mapping or null")
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise TargetPathError("TARGET_POLICY_INVALID", f"{name} keys must be non-empty strings")
        ref = _ref_or_none(value, f"{name}[{key}]")
        if ref is not None:
            out[key] = ref
    return out


def parse_target_definitions(raw: Any) -> TargetDefinitionBundle:
    """Parse target definitions into a typed bundle (mapping or dataclass)."""
    if isinstance(raw, TargetDefinitionBundle):
        return raw
    if raw is None:
        return TargetDefinitionBundle(bonds=(), regions=(), stereo=())
    if not isinstance(raw, Mapping):
        raise TargetPathError("TARGET_DEFINITION_INVALID", "target_definitions must be a mapping")
    bonds_raw = raw.get("bonds", ())
    regions_raw = raw.get("regions", ())
    stereo_raw = raw.get("stereo", ())
    if not isinstance(bonds_raw, (list, tuple)):
        raise TargetPathError("TARGET_DEFINITION_INVALID", "bonds must be an array")
    if not isinstance(regions_raw, (list, tuple)):
        raise TargetPathError("TARGET_DEFINITION_INVALID", "regions must be an array")
    if not isinstance(stereo_raw, (list, tuple)):
        raise TargetPathError("TARGET_DEFINITION_INVALID", "stereo must be an array")
    bonds = tuple(_parse_bond(item, f"bonds[{index}]") for index, item in enumerate(bonds_raw))
    regions = tuple(_parse_region(item, f"regions[{index}]") for index, item in enumerate(regions_raw))
    stereo = tuple(_parse_stereo(item, f"stereo[{index}]") for index, item in enumerate(stereo_raw))
    bond_tol = _positive_float(
        raw.get("bond_tolerance_angstrom", DEFAULT_BOND_TOLERANCE_ANGSTROM),
        "bond_tolerance_angstrom",
    )
    transition = _positive_float(
        raw.get("transition_width_angstrom", DEFAULT_TRANSITION_WIDTH_ANGSTROM),
        "transition_width_angstrom",
    )
    constraint_tol = _positive_float(
        raw.get("constraint_tolerance_angstrom", DEFAULT_CONSTRAINT_TOLERANCE_ANGSTROM),
        "constraint_tolerance_angstrom",
    )
    return TargetDefinitionBundle(
        bonds=bonds,
        regions=regions,
        stereo=stereo,
        bond_tolerance_angstrom=bond_tol,
        transition_width_angstrom=transition,
        constraint_tolerance_angstrom=constraint_tol,
        electronic_order_evidence=_parse_evidence_map(
            raw.get("electronic_order_evidence"), "electronic_order_evidence"
        ),
        stereo_evidence=_parse_evidence_map(raw.get("stereo_evidence"), "stereo_evidence"),
        region_evidence=_parse_evidence_map(raw.get("region_evidence"), "region_evidence"),
    )


def parse_evidence_bundle(raw: Any) -> TsEvidenceBundle:
    """Parse validated_ts evidence references (opaque strings only)."""
    if isinstance(raw, TsEvidenceBundle):
        return raw
    if raw is None:
        return TsEvidenceBundle()
    if not isinstance(raw, Mapping):
        raise TargetPathError("EVIDENCE_INVALID", "evidence_bundle must be a mapping")
    for key in raw:
        if str(key).lower() in FORBIDDEN_ASSESSMENT_KEYS:
            raise TargetPathError(
                "EVIDENCE_INVALID", f"evidence_bundle key {key!r} is a forbidden truth field"
            )
    return TsEvidenceBundle(
        optts_ref=_ref_or_none(raw.get(EVIDENCE_OPTTS), EVIDENCE_OPTTS),
        first_order_imaginary_ref=_ref_or_none(
            raw.get(EVIDENCE_FIRST_ORDER_IMAGINARY), EVIDENCE_FIRST_ORDER_IMAGINARY
        ),
        irc_forward_ref=_ref_or_none(raw.get(EVIDENCE_IRC_FORWARD), EVIDENCE_IRC_FORWARD),
        irc_reverse_ref=_ref_or_none(raw.get(EVIDENCE_IRC_REVERSE), EVIDENCE_IRC_REVERSE),
    )


def _parse_report(raw: Any) -> FrameRecoveryReport:
    if isinstance(raw, FrameRecoveryReport):
        return raw
    if not isinstance(raw, Mapping):
        raise TargetPathError(
            "FRAME_RECOVERY_INVALID",
            "frame_recovery_report must be a FrameRecoveryReport or its to_doc() mapping",
        )
    try:
        frames_raw = raw["frames"]
        driver_ids = tuple(raw["driver_ids"])
    except KeyError as exc:
        raise TargetPathError(
            "FRAME_RECOVERY_INVALID", f"frame recovery report missing {exc.args[0]}"
        ) from exc
    if not isinstance(frames_raw, (list, tuple)) or not isinstance(driver_ids, tuple):
        raise TargetPathError("FRAME_RECOVERY_INVALID", "frames/driver_ids have invalid shapes")
    incomplete = tuple(int(index) for index in raw.get("incomplete_frame_indices", ()))
    complete_flag = bool(raw.get("complete", False))
    n_frames = int(raw.get("n_frames", len(frames_raw)))
    source = dict(raw.get("source", {})) if isinstance(raw.get("source"), Mapping) else {}
    shell = FrameRecoveryReport(
        schema_version=str(raw.get("schema_version", "")),
        candidate_id=str(raw.get("candidate_id", "")),
        mode=str(raw.get("mode", "")),
        execution_id=str(raw.get("execution_id", "")),
        acp_task_id=raw.get("acp_task_id"),
        driver_ids=driver_ids,
        driver_units=dict(raw.get("driver_units", {})),
        n_frames=n_frames,
        complete=complete_flag,
        frames=tuple(),
        incomplete_frame_indices=incomplete,
        monitor_coordinate_ids=tuple(raw.get("monitor_coordinate_ids", ())),
        electronic_state_diagnostics=dict(raw.get("electronic_state_diagnostics", {})),
        source=source,
    )
    return _report_from_frames_doc(shell, frames_raw)


# ---------------------------------------------------------------------------
# Continuous bond-length classification.
# ---------------------------------------------------------------------------
def _bond_thresholds(
    elements: tuple[str, str], bundle: TargetDefinitionBundle
) -> tuple[float, float, float]:
    """Return (radius_sum, bonded_upper, separated_lower)."""
    radius_sum = covalent_radius_sum(elements[0], elements[1])
    bonded_upper = radius_sum + bundle.bond_tolerance_angstrom
    separated_lower = bonded_upper + bundle.transition_width_angstrom
    return radius_sum, bonded_upper, separated_lower


def _classify_bond(
    distance: float | None,
    edit_kind: str,
    thresholds: tuple[float, float, float],
    *,
    order_evidence_ref: str | None,
) -> tuple[str, bool, tuple[str, ...]]:
    """Continuous bond verdict; transition zone stays ``unknown/transition``."""
    radius_sum, bonded_upper, separated_lower = thresholds
    if distance is None:
        return BOND_TRANSITION, False, (CODE_BOND_MEASUREMENT_MISSING,)
    if edit_kind == EDIT_FORMED:
        if distance <= bonded_upper:
            return BOND_FORMED, True, ()
        if distance < separated_lower:
            return BOND_TRANSITION, False, (CODE_BOND_INCOMPLETE,)
        return BOND_NOT_FORMED, False, (CODE_BOND_INCOMPLETE,)
    if edit_kind == EDIT_BROKEN:
        if distance >= separated_lower:
            return BOND_BROKEN, True, ()
        if distance > bonded_upper:
            return BOND_TRANSITION, False, (CODE_BOND_INCOMPLETE,)
        return BOND_STILL_BONDED, False, (CODE_BOND_INCOMPLETE,)
    # order_changed: both sides stay bonded; electronic order evidence required.
    if distance > bonded_upper:
        return BOND_TRANSITION, False, (CODE_BOND_INCOMPLETE,)
    if order_evidence_ref is None:
        return BOND_NO_ORDER_EVIDENCE, False, (CODE_ORDER_EVIDENCE_MISSING,)
    return BOND_FORMED, True, ()


def _monitor_distance(record: Any, atom_maps: tuple[int, int]) -> float | None:
    pair = frozenset(atom_maps)
    for monitor in record.monitors:
        if not isinstance(monitor, MonitorRecord):
            continue
        if monitor.kind != "B" or not monitor.measured:
            continue
        if frozenset(monitor.atom_maps) == pair:
            return _finite_number(monitor.value)
    return None


def _driver_distance(record: Any, driver_id: str | None) -> float | None:
    if driver_id is None:
        return None
    return _finite_number(record.actuals_by_driver.get(driver_id))


def _end_distance(record: Any, definition: TargetBondDefinition) -> float | None:
    measured = _monitor_distance(record, definition.atom_maps)
    if measured is not None:
        return measured
    return _driver_distance(record, definition.driver_id)


def _scan_energies(report: FrameRecoveryReport) -> list[float | None]:
    values: list[float | None] = []
    for record in report.frames:
        scan_row = None
        for row in record.energies:
            if row.energy_channel == "scan":
                scan_row = row
                break
        values.append(_finite_number(scan_row.value) if scan_row is not None else None)
    return values


def _diagnose_path_shape(report: FrameRecoveryReport) -> PathShapeDiagnosis:
    energies = _scan_energies(report)
    n_frames = len(report.frames)
    if n_frames == 0:
        return PathShapeDiagnosis(
            n_frames=0,
            energies_available=False,
            max_frame_index=None,
            max_is_endpoint=False,
            has_internal_maximum=False,
            classification=PATH_CLASS_ENERGIES_UNAVAILABLE,
        )
    if any(value is None for value in energies) or not any(value is not None for value in energies):
        return PathShapeDiagnosis(
            n_frames=n_frames,
            energies_available=False,
            max_frame_index=None,
            max_is_endpoint=False,
            has_internal_maximum=False,
            classification=PATH_CLASS_ENERGIES_UNAVAILABLE,
        )
    finite = [value for value in energies if value is not None]
    max_index = max(range(n_frames), key=lambda i: (energies[i] if energies[i] is not None else -math.inf))
    max_is_endpoint = max_index in (0, n_frames - 1)
    has_internal = False
    if 0 < max_index < n_frames - 1:
        left = energies[max_index - 1]
        right = energies[max_index + 1]
        current = energies[max_index]
        if left is not None and right is not None and current is not None:
            has_internal = current >= left and current >= right
    if max_is_endpoint:
        # Monotonic association/dissociation profiles have no interior barrier;
        # the design treats them as endpoint-max / no-internal-peak paths.
        classification = PATH_CLASS_ENDPOINT_MAX
    elif has_internal:
        classification = PATH_CLASS_INTERNAL_MAXIMUM
    else:
        classification = PATH_CLASS_NO_INTERNAL_PEAK
    return PathShapeDiagnosis(
        n_frames=n_frames,
        energies_available=True,
        max_frame_index=max_index,
        max_is_endpoint=max_is_endpoint,
        has_internal_maximum=has_internal,
        classification=classification,
    )


def _assess_execution_complete(report: FrameRecoveryReport) -> TierResult:
    codes: list[str] = []
    if report.n_frames <= 0 or not report.frames:
        codes.append(CODE_NO_FRAMES)
    if not report.complete or report.incomplete_frame_indices:
        codes.append(CODE_CONTINUITY_GAP)
    if codes:
        return TierResult(
            tier=TIER_EXECUTION_COMPLETE,
            status=LABEL_NOT_ACHIEVED,
            reason_codes=tuple(codes),
            details={"n_frames": report.n_frames, "incomplete_frame_indices": list(report.incomplete_frame_indices)},
        )
    return TierResult(
        tier=TIER_EXECUTION_COMPLETE,
        status=LABEL_ACHIEVED,
        details={"n_frames": report.n_frames},
    )


def _assess_numerically_usable(report: FrameRecoveryReport, bundle: TargetDefinitionBundle) -> TierResult:
    codes: list[str] = []
    details: dict[str, Any] = {}
    if not report.frames:
        return TierResult(
            tier=TIER_NUMERICALLY_USABLE,
            status=LABEL_NOT_ACHIEVED,
            reason_codes=(CODE_NO_FRAMES,),
        )
    unconverged = [record.frame_index for record in report.frames if record.converged is not True]
    if unconverged:
        codes.append(CODE_NOT_CONVERGED)
        details["unconverged_frame_indices"] = unconverged
    residual_missing = [
        record.frame_index
        for record in report.frames
        if record.incomplete or record.missing_residual_driver_ids
    ]
    if residual_missing:
        codes.append(CODE_DRIVER_RESIDUAL_MISSING)
        details["residual_missing_frame_indices"] = residual_missing
    indices = [record.frame_index for record in report.frames]
    if indices != list(range(len(indices))):
        codes.append(CODE_FRAME_JUMP)
        details["frame_indices"] = indices
    over_tolerance: list[dict[str, Any]] = []
    for record in report.frames:
        for driver_id, residual in record.residuals_by_driver.items():
            value = _finite_number(residual)
            if value is None:
                continue
            if abs(value) > bundle.constraint_tolerance_angstrom:
                over_tolerance.append(
                    {"frame_index": record.frame_index, "driver_id": driver_id, "residual": value}
                )
    if over_tolerance:
        codes.append(CODE_CONSTRAINT_RESIDUAL_EXCEEDED)
        details["residuals_over_tolerance"] = over_tolerance[:10]
    energies = _scan_energies(report)
    energy_complete = all(value is not None for value in energies)
    method_ids: set[str] = set()
    energy_rows_ok = True
    for record in report.frames:
        scan_rows = [row for row in record.energies if row.energy_channel == "scan"]
        if not scan_rows or _finite_number(scan_rows[0].value) is None:
            energy_rows_ok = False
            continue
        if scan_rows[0].method_id is not None:
            method_ids.add(str(scan_rows[0].method_id))
    if not energy_complete or not energy_rows_ok:
        codes.append(CODE_ENERGY_INCOMPLETE)
        details["scan_energies_available"] = energy_complete
    elif len(method_ids) > 1:
        codes.append(CODE_ENERGY_INCOMPARABLE)
        details["method_ids"] = sorted(method_ids)
    if codes:
        # de-duplicate while preserving order
        unique = tuple(dict.fromkeys(codes))
        return TierResult(
            tier=TIER_NUMERICALLY_USABLE,
            status=LABEL_NOT_ACHIEVED,
            reason_codes=unique,
            details=details,
        )
    return TierResult(
        tier=TIER_NUMERICALLY_USABLE,
        status=LABEL_ACHIEVED,
        details={"n_frames": len(report.frames)},
    )


def _label_bonds(
    report: FrameRecoveryReport, bundle: TargetDefinitionBundle
) -> tuple[BondLabelRecord, ...]:
    if not report.frames:
        return tuple()
    end_frame = max(report.frames, key=lambda record: record.frame_index)
    rows: list[BondLabelRecord] = []
    for definition in bundle.bonds:
        thresholds = _bond_thresholds((definition.element_a, definition.element_b), bundle)
        distance = _end_distance(end_frame, definition)
        order_ref = bundle.electronic_order_evidence.get(definition.target_id)
        label, completed, reasons = _classify_bond(
            distance, definition.edit_kind, thresholds, order_evidence_ref=order_ref
        )
        radius_sum, bonded_upper, separated_lower = thresholds
        rows.append(
            BondLabelRecord(
                target_id=definition.target_id,
                atom_maps=definition.atom_maps,
                edit_kind=definition.edit_kind,
                label=label,
                end_distance_angstrom=distance,
                radius_sum_angstrom=radius_sum,
                bonded_upper_angstrom=bonded_upper,
                separated_lower_angstrom=separated_lower,
                completed=completed,
                reason_codes=reasons,
            )
        )
    return tuple(rows)


def _assess_target_path_compatible(
    report: FrameRecoveryReport,
    bundle: TargetDefinitionBundle,
    bond_labels: tuple[BondLabelRecord, ...],
) -> TierResult:
    codes: list[str] = []
    details: dict[str, Any] = {}
    if not report.frames:
        return TierResult(
            tier=TIER_TARGET_PATH_COMPATIBLE,
            status=LABEL_NOT_ACHIEVED,
            reason_codes=(CODE_NO_FRAMES,),
        )
    indices = [record.frame_index for record in report.frames]
    if indices != list(range(len(indices))):
        codes.append(CODE_FRAME_JUMP)
        details["frame_indices"] = indices
    if report.incomplete_frame_indices or any(record.incomplete for record in report.frames):
        codes.append(CODE_CONTINUITY_GAP)
        details["incomplete_frame_indices"] = list(report.incomplete_frame_indices)
    incomplete_bonds = [row for row in bond_labels if not row.completed]
    if incomplete_bonds:
        codes.append(CODE_TARGET_INCOMPLETE)
        details["incomplete_bond_targets"] = [
            {"target_id": row.target_id, "label": row.label, "reason_codes": list(row.reason_codes)}
            for row in incomplete_bonds
        ]
        for row in incomplete_bonds:
            for code in row.reason_codes:
                if code not in codes:
                    codes.append(code)
    region_failures: list[dict[str, Any]] = []
    for region in bundle.regions:
        member_labels = [
            row for row in bond_labels if row.target_id in {member.target_id for member in region.bond_targets}
        ]
        # Also label member bonds that are not top-level bond targets.
        if region.bond_targets and len(member_labels) < len(region.bond_targets):
            end_frame = max(report.frames, key=lambda record: record.frame_index)
            for member in region.bond_targets:
                if any(row.target_id == member.target_id for row in member_labels):
                    continue
                thresholds = _bond_thresholds((member.element_a, member.element_b), bundle)
                distance = _end_distance(end_frame, member)
                order_ref = bundle.electronic_order_evidence.get(member.target_id)
                label, completed, reasons = _classify_bond(
                    distance, member.edit_kind, thresholds, order_evidence_ref=order_ref
                )
                member_labels.append(
                    BondLabelRecord(
                        target_id=member.target_id,
                        atom_maps=member.atom_maps,
                        edit_kind=member.edit_kind,
                        label=label,
                        end_distance_angstrom=distance,
                        radius_sum_angstrom=thresholds[0],
                        bonded_upper_angstrom=thresholds[1],
                        separated_lower_angstrom=thresholds[2],
                        completed=completed,
                        reason_codes=reasons,
                    )
                )
        if region.bond_targets:
            failed = [row for row in member_labels if not row.completed]
            if failed:
                region_failures.append(
                    {
                        "target_id": region.target_id,
                        "region_id": region.region_id,
                        "code": CODE_REGION_INCOMPLETE,
                        "failed_targets": [row.target_id for row in failed],
                    }
                )
        elif region.requires_electronic_evidence:
            if region.region_id not in bundle.region_evidence and region.target_id not in bundle.region_evidence:
                region_failures.append(
                    {
                        "target_id": region.target_id,
                        "region_id": region.region_id,
                        "code": CODE_REGION_ORDER_EVIDENCE_MISSING,
                        "failed_targets": [],
                    }
                )
    if region_failures:
        codes.append(CODE_REGION_INCOMPLETE)
        details["region_failures"] = region_failures
        for failure in region_failures:
            if failure["code"] not in codes:
                codes.append(str(failure["code"]))
    stereo_failures: list[dict[str, Any]] = []
    for target in bundle.stereo:
        final_label = target.final_label
        if final_label is None:
            stereo_failures.append(
                {"target_id": target.target_id, "code": CODE_STEREO_LABEL_MISSING}
            )
        elif final_label == target.p_label:
            continue
        elif target.r_label is not None and final_label == target.r_label:
            stereo_failures.append(
                {"target_id": target.target_id, "code": CODE_STEREO_AT_REACTANT}
            )
        else:
            stereo_failures.append(
                {"target_id": target.target_id, "code": CODE_STEREO_UNEXPECTED}
            )
    if stereo_failures:
        codes.append("STEREO_TARGET_INCOMPLETE")
        details["stereo_failures"] = stereo_failures
        for failure in stereo_failures:
            if failure["code"] not in codes:
                codes.append(str(failure["code"]))
    if codes:
        return TierResult(
            tier=TIER_TARGET_PATH_COMPATIBLE,
            status=LABEL_NOT_ACHIEVED,
            reason_codes=tuple(dict.fromkeys(codes)),
            details=details,
        )
    return TierResult(
        tier=TIER_TARGET_PATH_COMPATIBLE,
        status=LABEL_ACHIEVED,
        details={
            "n_bond_targets": len(bond_labels),
            "n_regions": len(bundle.regions),
            "n_stereo_targets": len(bundle.stereo),
        },
    )


def _assess_validated_ts(
    evidence: TsEvidenceBundle, path_shape: PathShapeDiagnosis
) -> TierResult:
    missing = evidence.missing_keys()
    codes: list[str] = []
    if missing:
        codes.append(CODE_EVIDENCE_INCOMPLETE)
    if path_shape.classification == PATH_CLASS_ENDPOINT_MAX:
        codes.append(CODE_ENDPOINT_MAX_NOT_TS)
    elif path_shape.classification == PATH_CLASS_NO_INTERNAL_PEAK:
        codes.append(CODE_NO_INTERNAL_PEAK)
    if missing:
        return TierResult(
            tier=TIER_VALIDATED_TS,
            status=LABEL_NOT_ACHIEVED,
            reason_codes=codes,
            missing_evidence=missing,
            details={"path_shape_classification": path_shape.classification},
        )
    # Evidence complete → achieved.  Path-shape codes remain as diagnostics in
    # details; endpoint-max / no-peak alone never grant this tier.
    return TierResult(
        tier=TIER_VALIDATED_TS,
        status=LABEL_ACHIEVED,
        reason_codes=tuple(codes),
        missing_evidence=(),
        details={
            "path_shape_classification": path_shape.classification,
            "ts_candidate_from_path_shape": (
                path_shape.classification == PATH_CLASS_INTERNAL_MAXIMUM
            ),
            "evidence_refs": list(EVIDENCE_REQUIRED_KEYS),
        },
    )


# ---------------------------------------------------------------------------
# Assessment entry point.
# ---------------------------------------------------------------------------
def assess_target_path(
    frame_recovery_report: Any,
    target_definitions: Any,
    evidence_bundle: Any,
) -> TargetPathAssessment:
    """Assess chemical target-path quality across four independent tiers.

    Parameters
    ----------
    frame_recovery_report:
        Todo-21 ``FrameRecoveryReport`` (or its ``to_doc()`` mapping).
    target_definitions:
        ``TargetDefinitionBundle`` or a mapping with ``bonds`` / ``regions`` /
        ``stereo`` / optional evidence maps and continuous-distance policy.
    evidence_bundle:
        ``TsEvidenceBundle`` or a mapping of opaque evidence references
        (``optts_ref`` / ``first_order_imaginary_ref`` / ``irc_forward_ref`` /
        ``irc_reverse_ref``).  Geometries/energies are refused.

    Returns
    -------
    TargetPathAssessment
        Per-tier status + per-check typed details.  Tiers are independent:
        technical ``numerically_usable`` does not imply chemical
        ``target_path_compatible``; ``validated_ts`` requires evidence refs.
    """
    report = _parse_report(frame_recovery_report)
    bundle = parse_target_definitions(target_definitions)
    evidence = parse_evidence_bundle(evidence_bundle)

    execution_tier = _assess_execution_complete(report)
    numerical_tier = _assess_numerically_usable(report, bundle)
    bond_labels = _label_bonds(report, bundle)
    target_tier = _assess_target_path_compatible(report, bundle, bond_labels)
    path_shape = _diagnose_path_shape(report)
    ts_tier = _assess_validated_ts(evidence, path_shape)

    ts_candidate_from_path_shape = (
        path_shape.classification == PATH_CLASS_INTERNAL_MAXIMUM
        and ts_tier.status == LABEL_ACHIEVED
    )

    checks: dict[str, TierResult] = {
        "bond_targets": TierResult(
            tier="bond_targets",
            status=(
                LABEL_ACHIEVED
                if bond_labels and all(row.completed for row in bond_labels)
                else (LABEL_NOT_ACHIEVED if bond_labels else LABEL_NOT_EVALUABLE)
            ),
            reason_codes=tuple(
                dict.fromkeys(code for row in bond_labels for code in row.reason_codes)
            ),
            details={
                "labels": [row.to_record() for row in bond_labels],
            },
        ),
        "path_continuity": TierResult(
            tier="path_continuity",
            status=(
                LABEL_ACHIEVED
                if report.frames
                and [record.frame_index for record in report.frames] == list(range(len(report.frames)))
                and not report.incomplete_frame_indices
                else LABEL_NOT_ACHIEVED
            ),
            reason_codes=(
                ()
                if report.frames
                and [record.frame_index for record in report.frames] == list(range(len(report.frames)))
                and not report.incomplete_frame_indices
                else (CODE_FRAME_JUMP if report.frames else CODE_NO_FRAMES,)
            ),
        ),
        "path_shape": TierResult(
            tier="path_shape",
            status=LABEL_NOT_EVALUABLE
            if path_shape.classification == PATH_CLASS_ENERGIES_UNAVAILABLE
            else LABEL_ACHIEVED,
            reason_codes=(
                ()
                if path_shape.classification != PATH_CLASS_ENERGIES_UNAVAILABLE
                else (CODE_ENERGY_INCOMPLETE,)
            ),
            details=path_shape.to_record(),
        ),
        "ts_evidence": TierResult(
            tier="ts_evidence",
            status=LABEL_ACHIEVED if not evidence.missing_keys() else LABEL_NOT_ACHIEVED,
            reason_codes=(CODE_EVIDENCE_INCOMPLETE,) if evidence.missing_keys() else (),
            missing_evidence=evidence.missing_keys(),
        ),
    }

    return TargetPathAssessment(
        schema_version=SCHEMA_TARGET_PATH_ASSESSMENT,
        candidate_id=report.candidate_id,
        execution_id=report.execution_id,
        tiers={
            TIER_EXECUTION_COMPLETE: execution_tier,
            TIER_NUMERICALLY_USABLE: numerical_tier,
            TIER_TARGET_PATH_COMPATIBLE: target_tier,
            TIER_VALIDATED_TS: ts_tier,
        },
        checks=checks,
        bond_labels=bond_labels,
        path_shape=path_shape,
        evidence=evidence.to_record(),
        ts_candidate_from_path_shape=ts_candidate_from_path_shape,
        source={
            "recovery_schema_version": report.schema_version,
            "recovery_candidate_id": report.candidate_id,
            "recovery_execution_id": report.execution_id,
            "recovery_mode": report.mode,
            "n_frames": report.n_frames,
            "driver_ids": list(report.driver_ids),
            "n_bond_targets": len(bundle.bonds),
            "n_regions": len(bundle.regions),
            "n_stereo_targets": len(bundle.stereo),
        },
    )


def _report_from_frames_doc(report: FrameRecoveryReport, frames_raw: Any) -> FrameRecoveryReport:
    """Rehydrate a FrameRecoveryReport from a to_doc() mapping's frames block."""
    from pes2ts_core.integration.acp.frame_recovery import (
        ContactRecord,
        EnergyChannelRow,
        FrameRecoveryRecord,
    )

    if not isinstance(frames_raw, (list, tuple)):
        return report
    records: list[FrameRecoveryRecord] = []
    for raw in frames_raw:
        if not isinstance(raw, Mapping):
            continue
        monitors = tuple(
            MonitorRecord(
                coordinate_id=str(item.get("coordinate_id", "")),
                kind=str(item.get("kind", "B")),
                atom_maps=tuple(int(m) for m in item.get("atom_maps", ())),
                units=str(item.get("units", "angstrom")),
                role=str(item.get("role", "monitor")),
                value=_finite_number(item.get("value")),
                measured=bool(item.get("measured", False)),
            )
            for item in raw.get("monitors", ())
            if isinstance(item, Mapping)
        )
        contacts = tuple(
            ContactRecord(
                atom_index_a=int(item.get("atom_index_a", 0)),
                atom_index_b=int(item.get("atom_index_b", 0)),
                map_a=int(item.get("map_a", 0)),
                map_b=int(item.get("map_b", 0)),
                distance_angstrom=float(item.get("distance_angstrom", 0.0)),
            )
            for item in raw.get("non_target_contacts", ())
            if isinstance(item, Mapping)
        )
        energies = tuple(
            EnergyChannelRow(
                energy_channel=str(item.get("energy_channel", "none")),
                value=_finite_number(item.get("value")),
                method_id=item.get("method_id"),
            )
            for item in raw.get("energies", ())
            if isinstance(item, Mapping)
        )
        records.append(
            FrameRecoveryRecord(
                frame_index=int(raw.get("frame_index", 0)),
                converged=raw.get("converged"),
                lambda_value=_finite_number(raw.get("lambda_value")),
                stage_id=raw.get("stage_id"),
                geometry_ref=str(raw.get("geometry_ref", "")),
                targets_by_driver=dict(raw.get("targets_by_driver", {})),
                actuals_by_driver=dict(raw.get("actuals_by_driver", {})),
                residuals_by_driver=dict(raw.get("residuals_by_driver", {})),
                backend_residuals_by_driver=dict(raw.get("backend_residuals_by_driver", {})),
                missing_driver_ids=tuple(raw.get("missing_driver_ids", ())),
                missing_residual_driver_ids=tuple(raw.get("missing_residual_driver_ids", ())),
                incomplete=bool(raw.get("incomplete", False)),
                monitors=monitors,
                non_target_contacts=contacts,
                contacts_measured=bool(raw.get("contacts_measured", False)),
                electronic_state_diagnostics=dict(raw.get("electronic_state_diagnostics", {})),
                retry_history=tuple(dict(item) for item in raw.get("retry_history", ())),
                energies=energies,
            )
        )
    records.sort(key=lambda row: row.frame_index)
    incomplete = tuple(row.frame_index for row in records if row.incomplete)
    complete = bool(records) and not incomplete and bool(report.complete)
    return FrameRecoveryReport(
        schema_version=report.schema_version,
        candidate_id=report.candidate_id,
        mode=report.mode,
        execution_id=report.execution_id,
        acp_task_id=report.acp_task_id,
        driver_ids=report.driver_ids,
        driver_units=report.driver_units,
        n_frames=len(records) if records else report.n_frames,
        complete=complete,
        frames=tuple(records),
        incomplete_frame_indices=incomplete,
        monitor_coordinate_ids=report.monitor_coordinate_ids,
        electronic_state_diagnostics=report.electronic_state_diagnostics,
        source=report.source,
    )
