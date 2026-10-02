"""Intermediate-evidence segmentation and plan versioning (design §8.2/§10.1).

Implements the two-level staging protocol of
``docs/design/PES_GENERATION_图论扫描策略选择器设计_v1.md`` §8.2:

- **G1 pre-freezes the intermediate exploration/adjudication rules** as a
  typed rule set (:func:`frozen_intermediate_rules`): candidate-frame energy
  local minimum, constraint-release optimization convergence,
  topology/charge/electronic-state confirmation, and Hessian/frequency for
  the formal conclusion.  The rule set is frozen *before* any G2 execution.
- **G2 returns evidence; G1 creates a NEW plan version** carrying the
  intermediate hash via :func:`stage_new_plan_version`.  The original plan
  document is never mutated; a derived superseded copy completes the
  ``supersedes`` chain that contracts_v2 validates.
- **Segmentation criteria** (design §8.2, operational level): a candidate
  intermediate frame released from constraints optimizes to a *stable*
  structure **and** topology/charge/electronic-state are confirmed → segment.
  The formal conclusion (Hessian/frequency) is recorded as ``pending_evidence``
  when not yet computed; an incomplete Hessian never blocks operational
  segmentation, while a computed Hessian with imaginary modes *contradicts*
  the stability claim and refuses segmentation.
- **Multi-peak paths**: every confirmed intermediate becomes a segment
  boundary; segments are adjacent pairs ``(R,I1), (I1,I2), …, (Ik,P)`` and
  are **never** merged into one TS (design §13.2 P3 gate).
- **"Coordinate temporarily unchanged" ≠ stepwise mechanism**: a
  ``coordinate_unchanged_claim`` alone never substitutes for the frozen rule
  set — segmentation without stable-intermediate evidence is a typed refusal
  (:data:`COORDINATE_UNCHANGED_NOT_MECHANISM`).
- **G2 never self-segments or changes drivers**: staging is structural —
  :func:`stage_new_plan_version` accepts only the frozen original plan (a G1
  artifact) plus returned evidence; it exposes no driver-mutation surface and
  copies every candidate driver verbatim into the new version.
- **Failure tree + budget replay across versions**: the new plan's
  ``extensions.freeze.failure_tree`` and budget denominator are carried from
  the todo-23 vocabulary (:data:`plan_freeze.FAILURE_TREE_CODE_ORDER`,
  :class:`plan_freeze.BudgetLedger` semantics) so failure predicates and
  budget exhaustion remain replayable on every staged version.

Refusals are typed (:class:`IntermediateStagingError` with a stable ``code``);
this module never invents chemistry — absent evidence means no segmentation.
"""

# allow: SIZE_OK — plan-named todo-27 single module (frozen rule set +
# typed evidence/adjudication + staged plan versioning + refusals;
# precedent contracts_v2/plan_freeze/path_request).

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pes2ts_core.contracts import (
    FORBIDDEN_TRUTH_KEYS as _V1_TRUTH_KEYS,
)
from pes2ts_core.contracts import (
    ContractError,
    seal_document,
)
from pes2ts_core.g1.v2_verify import (
    FORBIDDEN_EXPORT_KEYS as _V2_EXPORT_KEYS,
)
from pes2ts_core.generation.planning.contracts_v2 import (
    CANDIDATE_KIND_PATH,
    OBJECT_GENERATION_PLAN,
    SCHEMA_GENERATION_PLAN,
    make_generation_plan,
    validate_v2_document,
)
from pes2ts_core.generation.planning.path_request import (
    INTERMEDIATE_NEXT_STAGE,
)
from pes2ts_core.generation.planning.plan_freeze import (
    BUDGET_ACCOUNTING_CATEGORIES,
    FAILURE_TREE_CODE_ORDER,
    FAILURE_TREE_VERSION,
)
from pes2ts_core.utils.hashing import sha256_bytes, stable_json_dumps

# ---------------------------------------------------------------------------
# Schema vocabulary and typed refusal codes.
# ---------------------------------------------------------------------------
SCHEMA_INTERMEDIATE_STAGING: Final[str] = "g1_intermediate_staging_v1"
OBJECT_INTERMEDIATE_STAGING: Final[str] = "IntermediateStaging"
STAGING_RULES_VERSION: Final[str] = "intermediate_staging_rules_v1"
#: Cross-module handoff: path_request.IntermediateMinimumFlag.next_stage
#: points here (todo-26 seam; test-locked equality).
STAGING_STAGE_ID: Final[str] = INTERMEDIATE_NEXT_STAGE
#: Staging runs structurally on the G1 side only: G2 returns evidence and
#: never segments or mutates drivers itself (design §8.2).
STAGING_RUNS_ON: Final[str] = "G1_side_only"

#: Typed refusal codes (stable machine tokens; never free text).
CODE_STAGING_INPUT_INVALID: Final[str] = "STAGING_INPUT_INVALID"
CODE_PLAN_NOT_FROZEN: Final[str] = "PLAN_NOT_FROZEN"
CODE_NO_STABLE_INTERMEDIATE: Final[str] = "NO_STABLE_INTERMEDIATE"
CODE_ENERGY_LOCAL_MINIMUM_MISSING: Final[str] = "ENERGY_LOCAL_MINIMUM_MISSING"
CODE_INSUFFICIENT_STABILITY_EVIDENCE: Final[str] = "INSUFFICIENT_STABILITY_EVIDENCE"
CODE_INSUFFICIENT_TOPOLOGY_EVIDENCE: Final[str] = "INSUFFICIENT_TOPOLOGY_EVIDENCE"
CODE_COORDINATE_UNCHANGED_NOT_MECHANISM: Final[str] = "COORDINATE_UNCHANGED_NOT_MECHANISM"
CODE_HESSIAN_CONTRADICTS_STABILITY: Final[str] = "HESSIAN_CONTRADICTS_STABILITY"
CODE_INTERMEDIATE_GEOMETRY_MISSING: Final[str] = "INTERMEDIATE_GEOMETRY_MISSING"
CODE_GEOMETRY_SHAPE_INVALID: Final[str] = "GEOMETRY_SHAPE_INVALID"

STAGING_REFUSAL_CODES: Final[frozenset[str]] = frozenset({
    CODE_STAGING_INPUT_INVALID,
    CODE_PLAN_NOT_FROZEN,
    CODE_NO_STABLE_INTERMEDIATE,
    CODE_ENERGY_LOCAL_MINIMUM_MISSING,
    CODE_INSUFFICIENT_STABILITY_EVIDENCE,
    CODE_INSUFFICIENT_TOPOLOGY_EVIDENCE,
    CODE_COORDINATE_UNCHANGED_NOT_MECHANISM,
    CODE_HESSIAN_CONTRADICTS_STABILITY,
    CODE_INTERMEDIATE_GEOMETRY_MISSING,
    CODE_GEOMETRY_SHAPE_INVALID,
})

#: Adjudication verdict vocabulary.
VERDICT_CONFIRMED_STABLE: Final[str] = "confirmed_stable"
VERDICT_REFUSED: Final[str] = "refused"
ADJUDICATION_VERDICTS: Final[frozenset[str]] = frozenset({
    VERDICT_CONFIRMED_STABLE, VERDICT_REFUSED,
})

#: Formal-conclusion vocabulary (Hessian/frequency channel).
FORMAL_CONFIRMED: Final[str] = "confirmed"
FORMAL_PENDING_EVIDENCE: Final[str] = "pending_evidence"
FORMAL_CONTRADICTED: Final[str] = "contradicted"
FORMAL_CONCLUSIONS: Final[frozenset[str]] = frozenset({
    FORMAL_CONFIRMED, FORMAL_PENDING_EVIDENCE, FORMAL_CONTRADICTED,
})

#: Hessian/frequency evidence status vocabulary.
HESSIAN_PENDING: Final[str] = "pending"
HESSIAN_COMPLETE: Final[str] = "complete"
HESSIAN_STATUSES: Final[frozenset[str]] = frozenset({HESSIAN_PENDING, HESSIAN_COMPLETE})

#: Frozen rule-class vocabulary (design §8.2 four rule families).
RULE_CLASS_CANDIDATE_FRAME: Final[str] = "candidate_frame"
RULE_CLASS_CONSTRAINT_RELEASE: Final[str] = "constraint_release"
RULE_CLASS_TOPOLOGY_CHARGE_ELECTRONIC: Final[str] = "topology_charge_electronic"
RULE_CLASS_HESSIAN_FREQUENCY: Final[str] = "hessian_frequency"
RULE_CLASSES: Final[tuple[str, ...]] = (
    RULE_CLASS_CANDIDATE_FRAME,
    RULE_CLASS_CONSTRAINT_RELEASE,
    RULE_CLASS_TOPOLOGY_CHARGE_ELECTRONIC,
    RULE_CLASS_HESSIAN_FREQUENCY,
)

#: Boundary refs used in segment from/to endpoints.
REF_REACTANT: Final[str] = "R"
REF_PRODUCT: Final[str] = "P"

#: Every key that must never appear in staging documents or projections.
FORBIDDEN_STAGING_KEYS: Final[frozenset[str]] = frozenset(
    {key.lower() for key in _V1_TRUTH_KEYS}
    | {key.lower() for key in _V2_EXPORT_KEYS}
)

_SHA256_LEN: Final[int] = 64


class IntermediateStagingError(ValueError):
    """Typed staging refusal; ``code`` is a stable machine token."""

    def __init__(self, code: str, detail: str = "") -> None:
        if code not in STAGING_REFUSAL_CODES:
            raise IntermediateStagingError(
                CODE_STAGING_INPUT_INVALID, f"unknown staging refusal code {code!r}"
            )
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


# ---------------------------------------------------------------------------
# Frozen rule set (G1 pre-freezes; never derived from G2 evidence).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IntermediateRule:
    """One frozen intermediate exploration/adjudication rule record."""

    rule_id: str
    rule_class: str
    description: str
    required_for_segmentation: bool

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection of one rule record."""
        return {
            "rule_id": self.rule_id,
            "rule_class": self.rule_class,
            "description": self.description,
            "required_for_segmentation": self.required_for_segmentation,
        }


def _rule(
    rule_id: str, rule_class: str, description: str, *, required: bool
) -> IntermediateRule:
    return IntermediateRule(
        rule_id=rule_id,
        rule_class=rule_class,
        description=description,
        required_for_segmentation=required,
    )


#: The four rule families of design §8.2, frozen as typed records.
_RULE_RECORDS: Final[tuple[IntermediateRule, ...]] = (
    _rule(
        "R_INTERMEDIATE_CANDIDATE_FRAME_ENERGY_MINIMUM",
        RULE_CLASS_CANDIDATE_FRAME,
        "candidate intermediate frame is an energy local minimum along the "
        "recovered path (interior minimum, strict on at least one side)",
        required=True,
    ),
    _rule(
        "R_INTERMEDIATE_CONSTRAINT_RELEASE_STABLE",
        RULE_CLASS_CONSTRAINT_RELEASE,
        "candidate frame released from reaction constraints optimizes to a "
        "stable structure (released + converged + stable_structure)",
        required=True,
    ),
    _rule(
        "R_INTERMEDIATE_TOPOLOGY_CHARGE_ELECTRONIC",
        RULE_CLASS_TOPOLOGY_CHARGE_ELECTRONIC,
        "topology, total charge and electronic state are confirmed on the "
        "constraint-release optimized structure",
        required=True,
    ),
    _rule(
        "R_INTERMEDIATE_HESSIAN_FREQUENCY_FORMAL",
        RULE_CLASS_HESSIAN_FREQUENCY,
        "formal intermediate conclusion via Hessian/frequency verification "
        "(pending_evidence until computed; imaginary modes contradict stability)",
        required=False,
    ),
)


@dataclass(frozen=True, slots=True)
class IntermediateRuleSet:
    """G1-frozen intermediate exploration/adjudication rule set."""

    rules_version: str
    rules: tuple[IntermediateRule, ...]
    content_sha256: str

    def rule_ids(self) -> tuple[str, ...]:
        """Rule ids in frozen order."""
        return tuple(rule.rule_id for rule in self.rules)

    def rules_by_class(self, rule_class: str) -> tuple[IntermediateRule, ...]:
        """Every rule belonging to *rule_class* (frozen order)."""
        return tuple(rule for rule in self.rules if rule.rule_class == rule_class)

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection including the rule-set digest."""
        return {
            "schema_name": OBJECT_INTERMEDIATE_STAGING,
            "schema_version": SCHEMA_INTERMEDIATE_STAGING,
            "document_kind": "intermediate_rule_set",
            "rules_version": self.rules_version,
            "rules": [rule.to_doc() for rule in self.rules],
            "content_sha256": self.content_sha256,
        }


def _rule_set_digest(rules_version: str, rules: Sequence[IntermediateRule]) -> str:
    payload = stable_json_dumps({
        "rules_version": rules_version,
        "rules": [rule.to_doc() for rule in rules],
    })
    return sha256_bytes(payload.encode("utf-8"))


def _build_rule_set() -> IntermediateRuleSet:
    rules = _RULE_RECORDS
    return IntermediateRuleSet(
        rules_version=STAGING_RULES_VERSION,
        rules=rules,
        content_sha256=_rule_set_digest(STAGING_RULES_VERSION, rules),
    )


#: G1-frozen rule set — immutable, deterministic, pre-execution.
FROZEN_INTERMEDIATE_RULES: Final[IntermediateRuleSet] = _build_rule_set()


def frozen_intermediate_rules() -> IntermediateRuleSet:
    """Return the G1-frozen intermediate rule set (deterministic)."""
    return FROZEN_INTERMEDIATE_RULES


# ---------------------------------------------------------------------------
# Typed evidence records (G2 returns; G1 adjudicates).
# ---------------------------------------------------------------------------
def _finite_xyz(raw: Any, where: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise IntermediateStagingError(CODE_GEOMETRY_SHAPE_INVALID, f"{where} must be a length-3 coordinate")
    coords: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise IntermediateStagingError(
                CODE_GEOMETRY_SHAPE_INVALID, f"{where} coordinates must be numbers"
            )
        value = float(item)
        if not math.isfinite(value):
            raise IntermediateStagingError(
                CODE_GEOMETRY_SHAPE_INVALID, f"{where} coordinates must be finite"
            )
        coords.append(value)
    return (coords[0], coords[1], coords[2])


def _require_bool(raw: Any, where: str) -> bool:
    if not isinstance(raw, bool):
        raise IntermediateStagingError(CODE_STAGING_INPUT_INVALID, f"{where} must be a boolean")
    return raw


def _optional_float(raw: Any, where: str) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise IntermediateStagingError(CODE_STAGING_INPUT_INVALID, f"{where} must be a number or null")
    value = float(raw)
    if not math.isfinite(value):
        raise IntermediateStagingError(CODE_STAGING_INPUT_INVALID, f"{where} must be finite")
    return value


def _optional_int(raw: Any, where: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise IntermediateStagingError(CODE_STAGING_INPUT_INVALID, f"{where} must be an integer or null")
    return raw


@dataclass(frozen=True, slots=True)
class OptimizationEvidence:
    """Constraint-release optimization evidence for one candidate frame."""

    released_from_constraints: bool
    converged: bool
    stable_structure: bool

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "released_from_constraints": self.released_from_constraints,
            "converged": self.converged,
            "stable_structure": self.stable_structure,
        }


@dataclass(frozen=True, slots=True)
class ConfirmationEvidence:
    """Topology / charge / electronic-state confirmation evidence."""

    topology_confirmed: bool
    charge_confirmed: bool
    electronic_state_confirmed: bool

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "topology_confirmed": self.topology_confirmed,
            "charge_confirmed": self.charge_confirmed,
            "electronic_state_confirmed": self.electronic_state_confirmed,
        }

    @property
    def all_confirmed(self) -> bool:
        """True when every confirmation channel holds."""
        return (
            self.topology_confirmed
            and self.charge_confirmed
            and self.electronic_state_confirmed
        )


@dataclass(frozen=True, slots=True)
class HessianEvidence:
    """Hessian/frequency evidence for the formal intermediate conclusion."""

    status: str
    n_imaginary: int | None = None
    reference: str | None = None

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection; optional fields appear only when set."""
        doc: dict[str, Any] = {"status": self.status}
        if self.n_imaginary is not None:
            doc["n_imaginary"] = self.n_imaginary
        if self.reference is not None:
            doc["reference"] = self.reference
        return doc


@dataclass(frozen=True, slots=True)
class IntermediateEvidence:
    """Typed evidence record returned by G2 for one candidate frame."""

    frame_index: int
    energy_local_minimum: bool
    optimization: OptimizationEvidence
    confirmation: ConfirmationEvidence
    hessian: HessianEvidence | None = None
    coordinate_unchanged_claim: bool = False
    depth: float | None = None
    deep_intermediate: bool = False
    optimized_geometry: tuple[tuple[float, float, float], ...] | None = None
    source: str = "neb"

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe evidence projection (deterministic)."""
        doc: dict[str, Any] = {
            "frame_index": self.frame_index,
            "energy_local_minimum": self.energy_local_minimum,
            "optimization": self.optimization.to_doc(),
            "confirmation": self.confirmation.to_doc(),
            "coordinate_unchanged_claim": self.coordinate_unchanged_claim,
            "depth": self.depth,
            "deep_intermediate": self.deep_intermediate,
            "source": self.source,
        }
        if self.hessian is not None:
            doc["hessian"] = self.hessian.to_doc()
        if self.optimized_geometry is not None:
            doc["optimized_geometry"] = [list(row) for row in self.optimized_geometry]
        return doc

    def evidence_sha256(self) -> str:
        """Stable digest of this evidence record."""
        return sha256_bytes(stable_json_dumps(self.to_doc()).encode("utf-8"))


def _parse_optimization(raw: Any, where: str) -> OptimizationEvidence:
    if isinstance(raw, OptimizationEvidence):
        return raw
    if not isinstance(raw, Mapping):
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, f"{where}.optimization must be an object"
        )
    return OptimizationEvidence(
        released_from_constraints=_require_bool(
            raw.get("released_from_constraints"), f"{where}.optimization.released_from_constraints"
        ),
        converged=_require_bool(raw.get("converged"), f"{where}.optimization.converged"),
        stable_structure=_require_bool(
            raw.get("stable_structure"), f"{where}.optimization.stable_structure"
        ),
    )


def _parse_confirmation(raw: Any, where: str) -> ConfirmationEvidence:
    if isinstance(raw, ConfirmationEvidence):
        return raw
    if not isinstance(raw, Mapping):
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, f"{where}.confirmation must be an object"
        )
    return ConfirmationEvidence(
        topology_confirmed=_require_bool(
            raw.get("topology_confirmed"), f"{where}.confirmation.topology_confirmed"
        ),
        charge_confirmed=_require_bool(
            raw.get("charge_confirmed"), f"{where}.confirmation.charge_confirmed"
        ),
        electronic_state_confirmed=_require_bool(
            raw.get("electronic_state_confirmed"),
            f"{where}.confirmation.electronic_state_confirmed",
        ),
    )


def _parse_hessian(raw: Any, where: str) -> HessianEvidence | None:
    if raw is None:
        return None
    if isinstance(raw, HessianEvidence):
        return raw
    if not isinstance(raw, Mapping):
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, f"{where}.hessian must be an object or null"
        )
    status_raw = raw.get("status")
    if not isinstance(status_raw, str) or status_raw not in HESSIAN_STATUSES:
        allowed = ", ".join(sorted(HESSIAN_STATUSES))
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, f"{where}.hessian.status must be one of {allowed}"
        )
    status = status_raw
    n_imaginary = _optional_int(raw.get("n_imaginary"), f"{where}.hessian.n_imaginary")
    reference_raw = raw.get("reference")
    reference: str | None
    if reference_raw is None:
        reference = None
    elif isinstance(reference_raw, str) and reference_raw.strip():
        reference = reference_raw.strip()
    else:
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, f"{where}.hessian.reference must be a non-empty string or null"
        )
    return HessianEvidence(status=status, n_imaginary=n_imaginary, reference=reference)


def _parse_geometry(raw: Any, where: str) -> tuple[tuple[float, float, float], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise IntermediateStagingError(
            CODE_GEOMETRY_SHAPE_INVALID, f"{where} must be a non-empty array of coordinates"
        )
    return tuple(_finite_xyz(row, f"{where}[{index}]") for index, row in enumerate(raw))


def _parse_frame_index(raw: Any, where: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, f"{where}.frame_index must be a non-negative integer"
        )
    return raw


def parse_intermediate_evidence(raw: Any, *, index: int = 0) -> IntermediateEvidence:
    """Boundary-parse one evidence record into a typed
    :class:`IntermediateEvidence` (dataclass passthrough or mapping)."""
    where = f"evidence[{index}]"
    if isinstance(raw, IntermediateEvidence):
        return raw
    if not isinstance(raw, Mapping):
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, f"{where} must be an object or IntermediateEvidence"
        )
    source_raw = raw.get("source", "neb")
    if not isinstance(source_raw, str) or not source_raw.strip():
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, f"{where}.source must be a non-empty string"
        )
    return IntermediateEvidence(
        frame_index=_parse_frame_index(raw.get("frame_index"), where),
        energy_local_minimum=_require_bool(
            raw.get("energy_local_minimum"), f"{where}.energy_local_minimum"
        ),
        optimization=_parse_optimization(raw.get("optimization"), where),
        confirmation=_parse_confirmation(raw.get("confirmation"), where),
        hessian=_parse_hessian(raw.get("hessian"), where),
        coordinate_unchanged_claim=_require_bool(
            raw.get("coordinate_unchanged_claim", False),
            f"{where}.coordinate_unchanged_claim",
        ),
        depth=_optional_float(raw.get("depth"), f"{where}.depth"),
        deep_intermediate=_require_bool(
            raw.get("deep_intermediate", False), f"{where}.deep_intermediate"
        ),
        optimized_geometry=_parse_geometry(raw.get("optimized_geometry"), where),
        source=source_raw.strip(),
    )


# ---------------------------------------------------------------------------
# Adjudication: apply the frozen rule set to one evidence record.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class IntermediateAdjudication:
    """Typed verdict for one evidence record under the frozen rules."""

    frame_index: int
    verdict: str
    refusal_code: str | None
    refusal_detail: str
    formal_conclusion: str
    rule_trace: tuple[str, ...]
    evidence_sha256: str
    deep_intermediate: bool
    depth: float | None

    @property
    def confirmed(self) -> bool:
        """True when the record passes every segmentation-required rule."""
        return self.verdict == VERDICT_CONFIRMED_STABLE

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "frame_index": self.frame_index,
            "verdict": self.verdict,
            "refusal_code": self.refusal_code,
            "refusal_detail": self.refusal_detail,
            "formal_conclusion": self.formal_conclusion,
            "rule_trace": list(self.rule_trace),
            "evidence_sha256": self.evidence_sha256,
            "deep_intermediate": self.deep_intermediate,
            "depth": self.depth,
        }


def _stability_ok(optimization: OptimizationEvidence) -> bool:
    return (
        optimization.released_from_constraints
        and optimization.converged
        and optimization.stable_structure
    )


def _hessian_contradicts(hessian: HessianEvidence | None) -> bool:
    return (
        hessian is not None
        and hessian.status == HESSIAN_COMPLETE
        and hessian.n_imaginary is not None
        and hessian.n_imaginary >= 1
    )


def _hessian_confirms(hessian: HessianEvidence | None) -> bool:
    return (
        hessian is not None
        and hessian.status == HESSIAN_COMPLETE
        and hessian.n_imaginary == 0
    )


def adjudicate_intermediate(
    evidence: IntermediateEvidence,
    rule_set: IntermediateRuleSet | None = None,
) -> IntermediateAdjudication:
    """Apply the frozen rule set to one evidence record.

    Segmentation requires **all** of: energy local minimum
    (``R_INTERMEDIATE_CANDIDATE_FRAME_ENERGY_MINIMUM``), stable
    constraint-release optimization
    (``R_INTERMEDIATE_CONSTRAINT_RELEASE_STABLE``), and topology/charge/
    electronic confirmation
    (``R_INTERMEDIATE_TOPOLOGY_CHARGE_ELECTRONIC``).  The Hessian/frequency
    rule is the formal-conclusion channel: absent/pending →
    ``pending_evidence`` (segmentation still allowed); computed with zero
    imaginary modes → ``confirmed``; computed with ≥1 imaginary modes →
    ``contradicted`` **and** a typed refusal
    (:data:`HESSIAN_CONTRADICTS_STABILITY`).

    A ``coordinate_unchanged_claim`` alone never substitutes for the rule set
    (:data:`COORDINATE_UNCHANGED_NOT_MECHANISM`).
    """
    rules = rule_set if rule_set is not None else FROZEN_INTERMEDIATE_RULES
    stability = _stability_ok(evidence.optimization)
    confirmed_ok = evidence.confirmation.all_confirmed
    energy_ok = evidence.energy_local_minimum
    trace: list[str] = []
    refusal_code: str | None = None
    refusal_detail = ""
    formal = FORMAL_PENDING_EVIDENCE

    trace.append("R_INTERMEDIATE_CANDIDATE_FRAME_ENERGY_MINIMUM")
    trace.append("R_INTERMEDIATE_CONSTRAINT_RELEASE_STABLE")
    trace.append("R_INTERMEDIATE_TOPOLOGY_CHARGE_ELECTRONIC")
    trace.append("R_INTERMEDIATE_HESSIAN_FREQUENCY_FORMAL")

    if _hessian_contradicts(evidence.hessian):
        formal = FORMAL_CONTRADICTED
        refusal_code = CODE_HESSIAN_CONTRADICTS_STABILITY
        refusal_detail = (
            "computed Hessian/frequency reports imaginary modes on a claimed "
            "stable intermediate — formal conclusion contradicts stability"
        )
    elif _hessian_confirms(evidence.hessian):
        formal = FORMAL_CONFIRMED

    if refusal_code is None:
        if energy_ok and stability and confirmed_ok:
            pass  # confirmed; formal stays pending/confirmed as computed above
        elif evidence.coordinate_unchanged_claim and not (energy_ok and stability and confirmed_ok):
            # Design §8.2: a temporarily unchanged coordinate is only a
            # schedule, never a stepwise mechanism on its own.
            if not energy_ok:
                refusal_code = CODE_COORDINATE_UNCHANGED_NOT_MECHANISM
                refusal_detail = (
                    "coordinate temporarily unchanged is a schedule, not a "
                    "stepwise mechanism; no stable-intermediate evidence"
                )
            elif not stability:
                refusal_code = CODE_INSUFFICIENT_STABILITY_EVIDENCE
                refusal_detail = (
                    "coordinate-unchanged claim without constraint-release "
                    "stability evidence cannot segment"
                )
            else:
                refusal_code = CODE_INSUFFICIENT_TOPOLOGY_EVIDENCE
                refusal_detail = (
                    "coordinate-unchanged claim without topology/charge/"
                    "electronic confirmation cannot segment"
                )
        elif not energy_ok:
            refusal_code = CODE_ENERGY_LOCAL_MINIMUM_MISSING
            refusal_detail = "candidate frame is not an energy local minimum along the path"
        elif not stability:
            refusal_code = CODE_INSUFFICIENT_STABILITY_EVIDENCE
            refusal_detail = (
                "constraint-release optimization did not converge to a stable structure"
            )
        else:
            refusal_code = CODE_INSUFFICIENT_TOPOLOGY_EVIDENCE
            refusal_detail = (
                "topology/charge/electronic-state confirmation incomplete"
            )

    verdict = VERDICT_REFUSED if refusal_code is not None else VERDICT_CONFIRMED_STABLE
    # Required-rule trace: record pass/fail per required rule id.
    detailed_trace = [
        f"{rules.rules[0].rule_id}:{'pass' if energy_ok else 'fail'}",
        f"{rules.rules[1].rule_id}:{'pass' if stability else 'fail'}",
        f"{rules.rules[2].rule_id}:{'pass' if confirmed_ok else 'fail'}",
        f"{rules.rules[3].rule_id}:{formal}",
    ]
    return IntermediateAdjudication(
        frame_index=evidence.frame_index,
        verdict=verdict,
        refusal_code=refusal_code,
        refusal_detail=refusal_detail,
        formal_conclusion=formal if refusal_code is None or formal == FORMAL_CONTRADICTED else formal,
        rule_trace=tuple(detailed_trace),
        evidence_sha256=evidence.evidence_sha256(),
        deep_intermediate=evidence.deep_intermediate,
        depth=evidence.depth,
    )


# ---------------------------------------------------------------------------
# Path segments (adjacent pairs; never merged into one TS).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PathSegment:
    """One adjacent path segment between confirmed intermediate boundaries."""

    segment_id: str
    from_ref: str
    to_ref: str
    adjacent: bool
    from_evidence_sha256: str | None
    to_evidence_sha256: str | None
    formal_conclusion_to: str

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe projection."""
        return {
            "segment_id": self.segment_id,
            "from_ref": self.from_ref,
            "to_ref": self.to_ref,
            "adjacent": self.adjacent,
            "from_evidence_sha256": self.from_evidence_sha256,
            "to_evidence_sha256": self.to_evidence_sha256,
            "formal_conclusion_to": self.formal_conclusion_to,
        }


def _intermediate_ref(frame_index: int) -> str:
    return f"I:{frame_index}"


def build_adjacent_segments(
    adjudications: Sequence[IntermediateAdjudication],
) -> tuple[PathSegment, ...]:
    """Segment the directed R→P path at every confirmed intermediate.

    Boundaries are ``[R, I:f1, I:f2, …, P]``; each segment is an adjacent
    pair.  Confirmed deep intermediates never merge into one TS — a path with
    *k* confirmed intermediates yields exactly *k+1* adjacent segments.
    """
    confirmed = sorted(
        (adj for adj in adjudications if adj.confirmed),
        key=lambda adj: adj.frame_index,
    )
    refs = [REF_REACTANT, *(_intermediate_ref(adj.frame_index) for adj in confirmed), REF_PRODUCT]
    segs: list[PathSegment] = []
    for index in range(len(refs) - 1):
        from_ref, to_ref = refs[index], refs[index + 1]
        from_hash: str | None = None
        to_hash: str | None = None
        formal_to = FORMAL_PENDING_EVIDENCE
        if index >= 1:
            from_hash = confirmed[index - 1].evidence_sha256
        if index < len(confirmed):
            to_hash = confirmed[index].evidence_sha256
            formal_to = confirmed[index].formal_conclusion
        segs.append(
            PathSegment(
                segment_id=f"seg-{index:02d}",
                from_ref=from_ref,
                to_ref=to_ref,
                adjacent=True,
                from_evidence_sha256=from_hash,
                to_evidence_sha256=to_hash,
                formal_conclusion_to=formal_to,
            )
        )
    return tuple(segs)


def confirmed_intermediate_sha256(
    adjudications: Sequence[IntermediateAdjudication],
) -> str:
    """Digest over the confirmed adjudications (the intermediate hash)."""
    confirmed = [
        adj.to_doc()
        for adj in sorted(adjudications, key=lambda a: a.frame_index)
        if adj.confirmed
    ]
    return sha256_bytes(stable_json_dumps(confirmed).encode("utf-8"))


# ---------------------------------------------------------------------------
# Staged plan version (G1-side; original plan never mutated).
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StagingResult:
    """Outcome of a successful staging: new plan version + chain + segments."""

    new_plan: dict[str, Any]
    superseded_original: dict[str, Any]
    segments: tuple[PathSegment, ...]
    intermediate_sha256: str
    rules_version: str
    rules_sha256: str
    confirmed_frame_indices: tuple[int, ...]
    formal_conclusions: tuple[str, ...]
    staging_runs_on: str = STAGING_RUNS_ON

    def to_doc(self) -> dict[str, Any]:
        """JSON-safe summary of the staging outcome."""
        return {
            "schema_name": OBJECT_INTERMEDIATE_STAGING,
            "schema_version": SCHEMA_INTERMEDIATE_STAGING,
            "document_kind": "staging_result",
            "staging_runs_on": self.staging_runs_on,
            "rules_version": self.rules_version,
            "rules_sha256": self.rules_sha256,
            "intermediate_sha256": self.intermediate_sha256,
            "confirmed_frame_indices": list(self.confirmed_frame_indices),
            "formal_conclusions": list(self.formal_conclusions),
            "segments": [segment.to_doc() for segment in self.segments],
            "new_plan_id": self.new_plan.get("plan_id"),
            "new_plan_version": self.new_plan.get("plan_version"),
            "new_plan_content_sha256": self.new_plan.get("content_sha256"),
            "superseded_original_plan_id": self.superseded_original.get("plan_id"),
        }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LEN
        and all(char in "0123456789abcdef" for char in value)
    )


def _parse_original_plan(original_plan: Any) -> dict[str, Any]:
    """Boundary-parse the frozen original plan (G1 artifact)."""
    if not isinstance(original_plan, Mapping):
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, "original_plan must be a sealed g1_generation_plan_v2 object"
        )
    if original_plan.get("schema_name") != OBJECT_GENERATION_PLAN:
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID,
            f"original_plan.schema_name must be {OBJECT_GENERATION_PLAN!r}",
        )
    if original_plan.get("schema_version") != SCHEMA_GENERATION_PLAN:
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID,
            f"original_plan.schema_version must be {SCHEMA_GENERATION_PLAN!r}",
        )
    if original_plan.get("status") != "frozen":
        raise IntermediateStagingError(
            CODE_PLAN_NOT_FROZEN,
            f"original_plan.status={original_plan.get('status')!r}; staging requires a frozen plan",
        )
    plan_version = original_plan.get("plan_version")
    if isinstance(plan_version, bool) or not isinstance(plan_version, int) or plan_version < 1:
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, "original_plan.plan_version must be a positive integer"
        )
    candidates = original_plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, "original_plan.candidates must be a non-empty array"
        )
    if not _is_sha256(original_plan.get("content_sha256")):
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, "original_plan.content_sha256 must be a lowercase SHA256"
        )
    for field in ("reaction_id", "case_id", "plan_id"):
        value = original_plan.get(field)
        if not isinstance(value, str) or not value:
            raise IntermediateStagingError(
                CODE_STAGING_INPUT_INVALID, f"original_plan.{field} must be a non-empty string"
            )
    problems = validate_v2_document(dict(original_plan))
    if problems:
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, "; ".join(problems[:5])
        )
    return dict(original_plan)


def _parse_evidence_list(raw: Any) -> tuple[IntermediateEvidence, ...]:
    if isinstance(raw, IntermediateEvidence):
        return (raw,)
    if isinstance(raw, Mapping):
        return (parse_intermediate_evidence(raw),)
    if isinstance(raw, (list, tuple)):
        if not raw:
            raise IntermediateStagingError(
                CODE_NO_STABLE_INTERMEDIATE,
                "intermediate_evidence is empty — segmentation requires confirmed stable evidence",
            )
        return tuple(
            parse_intermediate_evidence(entry, index=index)
            for index, entry in enumerate(raw)
        )
    raise IntermediateStagingError(
        CODE_STAGING_INPUT_INVALID,
        "intermediate_evidence must be an IntermediateEvidence, a mapping, or a non-empty sequence",
    )


def _boundary_geometries(
    original_candidate: Mapping[str, Any],
    confirmed: Sequence[tuple[IntermediateEvidence, IntermediateAdjudication]],
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    """Frozen endpoint blocks ``[R_rows, I1_rows, …, P_rows]`` for path plans."""
    geometries = original_candidate.get("endpoint_geometries")
    if not isinstance(geometries, Mapping):
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, "path candidate endpoint_geometries must be an object"
        )
    r_block = geometries.get("reactant")
    p_block = geometries.get("product")
    if not isinstance(r_block, (list, tuple)) or not isinstance(p_block, (list, tuple)):
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID, "path candidate endpoint_geometries must carry both sides"
        )
    n_atoms = original_candidate.get("n_atoms")
    boundary: list[tuple[tuple[float, float, float], ...]] = [
        tuple(_finite_xyz(row, "endpoint_geometries.reactant") for row in r_block)
    ]
    for evidence, _adj in confirmed:
        if evidence.optimized_geometry is None:
            raise IntermediateStagingError(
                CODE_INTERMEDIATE_GEOMETRY_MISSING,
                f"frame_index={evidence.frame_index}: path-plan staging requires the "
                "constraint-release optimized intermediate geometry (coordinates are "
                "never fabricated)",
            )
        if len(evidence.optimized_geometry) != n_atoms:
            raise IntermediateStagingError(
                CODE_GEOMETRY_SHAPE_INVALID,
                f"frame_index={evidence.frame_index}: optimized_geometry rows "
                f"{len(evidence.optimized_geometry)} != n_atoms {n_atoms}",
            )
        boundary.append(evidence.optimized_geometry)
    boundary.append(
        tuple(_finite_xyz(row, "endpoint_geometries.product") for row in p_block)
    )
    return tuple(boundary)


def _segmented_candidate_graph(
    plan_id: str,
    segment_ids: Sequence[str],
    carried_ids: Sequence[str],
) -> tuple[dict[str, Any], str]:
    """Candidate graph for a staged plan: immutable IDs + explicit terminals."""
    nodes: list[dict[str, Any]] = [
        {"node_id": f"{plan_id}:n-start", "state": "frozen_ready", "terminal": False},
    ]
    for segment_id in segment_ids:
        nodes.append(
            {
                "node_id": f"{plan_id}:n-exec-{segment_id}",
                "state": f"execute:{segment_id}",
                "terminal": False,
            }
        )
    for carried_id in carried_ids:
        nodes.append(
            {
                "node_id": f"{plan_id}:n-carry-{carried_id}",
                "state": f"carried_fallback:{carried_id}",
                "terminal": False,
            }
        )
    nodes.extend([
        {"node_id": f"{plan_id}:n-fallback", "state": "next_segment_or_candidate", "terminal": False},
        {"node_id": f"{plan_id}:n-ok", "state": "path_saved_target_reached", "terminal": True},
        {"node_id": f"{plan_id}:n-budget", "state": "budget_exhausted_preserve_paths", "terminal": True},
        {"node_id": f"{plan_id}:n-fail", "state": "terminal_failure_preserved", "terminal": True},
    ])
    start = nodes[0]["node_id"]
    exec_ids = [f"{plan_id}:n-exec-{segment_id}" for segment_id in segment_ids]
    fallback = f"{plan_id}:n-fallback"
    ok = f"{plan_id}:n-ok"
    budget = f"{plan_id}:n-budget"
    failure = f"{plan_id}:n-fail"
    edges: list[dict[str, str]] = [{"from": start, "to": exec_ids[0]}]
    for index in range(len(exec_ids) - 1):
        edges.append({"from": exec_ids[index], "to": exec_ids[index + 1]})
    edges.append({"from": exec_ids[-1], "to": ok})
    for exec_id in exec_ids:
        edges.extend([
            {"from": exec_id, "to": budget},
            {"from": exec_id, "to": failure},
            {"from": exec_id, "to": fallback},
        ])
    edges.append({"from": fallback, "to": exec_ids[0]})
    for carried_id in carried_ids:
        carried_node = f"{plan_id}:n-carry-{carried_id}"
        edges.extend([
            {"from": fallback, "to": carried_node},
            {"from": carried_node, "to": ok},
            {"from": carried_node, "to": budget},
            {"from": carried_node, "to": failure},
        ])
    graph_sha = sha256_bytes(
        stable_json_dumps({"nodes": nodes, "edges": edges}).encode("utf-8")
    )
    return {"nodes": nodes, "edges": edges, "graph_sha256": graph_sha}, graph_sha


def _segment_scan_candidate(
    candidate: Mapping[str, Any],
    segment: PathSegment,
    index: int,
) -> dict[str, Any]:
    """Copy one ScanCandidateV2 verbatim; drivers are never changed."""
    payload = copy.deepcopy(dict(candidate))
    base_id = str(candidate.get("candidate_id") or "candidate")
    payload["candidate_id"] = f"{base_id}:{segment.segment_id}"
    extensions = payload.get("extensions")
    merged = dict(extensions) if isinstance(extensions, Mapping) else {}
    merged["staging_segment"] = segment.to_doc()
    payload["extensions"] = merged
    payload["failure_reasons"] = []
    _ = index  # segment index encoded in segment_id
    return payload


def _segment_path_candidate(
    candidate: Mapping[str, Any],
    segment: PathSegment,
    boundary: Sequence[tuple[tuple[float, float, float], ...]],
    segment_index: int,
) -> dict[str, Any]:
    """Copy one PathCandidateV1 with segment-local endpoint blocks."""
    payload = copy.deepcopy(dict(candidate))
    base_id = str(candidate.get("candidate_id") or "candidate")
    payload["candidate_id"] = f"{base_id}:{segment.segment_id}"
    payload["endpoint_geometries"] = {
        "reactant": [list(row) for row in boundary[segment_index]],
        "product": [list(row) for row in boundary[segment_index + 1]],
    }
    extensions = payload.get("extensions")
    merged = dict(extensions) if isinstance(extensions, Mapping) else {}
    merged["staging_segment"] = segment.to_doc()
    payload["extensions"] = merged
    payload["failure_reasons"] = []
    return payload


def _staging_block(
    *,
    rules: IntermediateRuleSet,
    intermediate_sha256: str,
    confirmed: Sequence[tuple[IntermediateEvidence, IntermediateAdjudication]],
    segments: Sequence[PathSegment],
    original_plan: Mapping[str, Any],
    segment_candidate_ids: Sequence[str],
    carried_candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """``extensions.staging`` dimension bundle for the new plan version."""
    original_freeze = original_plan.get("extensions")
    freeze_block: Mapping[str, Any] = {}
    if isinstance(original_freeze, Mapping):
        raw = original_freeze.get("freeze")
        if isinstance(raw, Mapping):
            freeze_block = raw
    return {
        "schema_version": SCHEMA_INTERMEDIATE_STAGING,
        "staging_runs_on": STAGING_RUNS_ON,
        "rules_version": rules.rules_version,
        "rules_sha256": rules.content_sha256,
        "intermediate_sha256": intermediate_sha256,
        "confirmed_intermediates": [
            {
                "frame_index": evidence.frame_index,
                "evidence_sha256": adjudication.evidence_sha256,
                "depth": adjudication.depth,
                "deep_intermediate": adjudication.deep_intermediate,
                "formal_conclusion": adjudication.formal_conclusion,
                "source": evidence.source,
            }
            for evidence, adjudication in confirmed
        ],
        "segments": [segment.to_doc() for segment in segments],
        "n_segments": len(segments),
        "never_merged_into_single_ts": True,
        "superseded_plan_id": original_plan.get("plan_id"),
        "superseded_plan_content_sha256": original_plan.get("content_sha256"),
        "segment_candidate_ids": list(segment_candidate_ids),
        "carried_candidate_ids": list(carried_candidate_ids),
        "g2_self_segmentation": False,
        "drivers_changed_by_staging": False,
        "failure_tree_replay": {
            "version": FAILURE_TREE_VERSION,
            "codes": list(FAILURE_TREE_CODE_ORDER),
        },
        "budget_replay": {
            "from_plan_id": original_plan.get("plan_id"),
            "from_plan_content_sha256": original_plan.get("content_sha256"),
            "budget": copy.deepcopy(original_plan.get("budget")),
            "budget_accounting": list(
                freeze_block.get("budget_accounting") or BUDGET_ACCOUNTING_CATEGORIES
            ),
            "note": (
                "failure-tree vocabulary and budget denominator are carried "
                "verbatim from the superseded version; exhaustion semantics "
                "(todo 23 BudgetLedger) remain replayable on this version"
            ),
        },
    }


def stage_new_plan_version(
    original_plan: Any,
    intermediate_evidence: Any,
    *,
    rules: IntermediateRuleSet | None = None,
) -> StagingResult:
    """G1-side staging: create a new plan version at confirmed intermediates.

    ``original_plan`` is the frozen ``g1_generation_plan_v2`` document (a G1
    artifact); ``intermediate_evidence`` is what G2 returned.  Structural
    guarantee (design §8.2): this function runs on the G1 side only — G2
    never self-segments and never changes drivers.  The original plan dict is
    **never mutated**; a derived superseded copy completes the contracts_v2
    ``supersedes`` chain.

    Refusals (typed, raise :class:`IntermediateStagingError`): empty/all-
    refused evidence → ``NO_STABLE_INTERMEDIATE`` (or the first record's
    refusal code); non-frozen/invalid original → ``PLAN_NOT_FROZEN`` /
    ``STAGING_INPUT_INVALID``; path plans without intermediate geometry →
    ``INTERMEDIATE_GEOMETRY_MISSING``; "coordinate unchanged" alone →
    ``COORDINATE_UNCHANGED_NOT_MECHANISM``.
    """
    rule_set = rules if rules is not None else FROZEN_INTERMEDIATE_RULES
    plan = _parse_original_plan(original_plan)
    evidence_items = _parse_evidence_list(intermediate_evidence)

    adjudications = tuple(
        adjudicate_intermediate(evidence, rule_set) for evidence in evidence_items
    )
    confirmed_pairs = [
        (evidence, adjudication)
        for evidence, adjudication in zip(evidence_items, adjudications, strict=True)
        if adjudication.confirmed
    ]
    if not confirmed_pairs:
        first_refusal = next(
            (adj for adj in adjudications if adj.refusal_code is not None), None
        )
        if first_refusal is not None and first_refusal.refusal_code is not None:
            raise IntermediateStagingError(
                first_refusal.refusal_code, first_refusal.refusal_detail
            )
        raise IntermediateStagingError(
            CODE_NO_STABLE_INTERMEDIATE,
            "no evidence record passed the frozen intermediate rules",
        )

    confirmed_pairs.sort(key=lambda pair: pair[0].frame_index)
    confirmed_adjudications = tuple(pair[1] for pair in confirmed_pairs)
    segments = build_adjacent_segments(adjudications)
    intermediate_sha = confirmed_intermediate_sha256(adjudications)

    original_candidates = list(plan["candidates"])
    primary = original_candidates[0]
    is_path = primary.get("candidate_kind") == CANDIDATE_KIND_PATH
    boundary: tuple[tuple[tuple[float, float, float], ...], ...] | None = None
    if is_path:
        boundary = _boundary_geometries(primary, confirmed_pairs)

    segment_candidates: list[dict[str, Any]] = []
    segment_candidate_ids: list[str] = []
    for index, segment in enumerate(segments):
        if is_path:
            assert boundary is not None  # noqa: S101 — narrowed above
            payload = _segment_path_candidate(primary, segment, boundary, index)
        else:
            payload = _segment_scan_candidate(primary, segment, index)
        segment_candidates.append(payload)
        segment_candidate_ids.append(str(payload["candidate_id"]))

    carried_candidates = [copy.deepcopy(dict(c)) for c in original_candidates[1:]]
    carried_candidate_ids = [str(c.get("candidate_id") or "") for c in carried_candidates]

    new_version = int(plan["plan_version"]) + 1
    reaction_id = str(plan["reaction_id"])
    new_plan_id = f"{reaction_id}:gp-v{new_version}"

    endpoint_sha = str(
        (plan.get("policy_hashes") or {}).get("endpoint_graph_sha256") or ""
    )
    graph, graph_sha = _segmented_candidate_graph(
        new_plan_id, segment_candidate_ids, carried_candidate_ids
    )
    complete_graph_sha = sha256_bytes(
        stable_json_dumps(
            {
                "endpoint_graph_sha256": endpoint_sha,
                "candidate_graph_sha256": graph_sha,
            }
        ).encode("utf-8")
    )

    original_freeze_raw = (plan.get("extensions") or {}).get("freeze")
    freeze_block: dict[str, Any] = (
        dict(copy.deepcopy(original_freeze_raw))
        if isinstance(original_freeze_raw, Mapping)
        else {}
    )
    freeze_block["candidate_graph_sha256"] = graph_sha
    freeze_block["graph_sha256"] = complete_graph_sha
    freeze_block["selected_candidate_id"] = segment_candidate_ids[0]
    compiled_request = freeze_block.get("compiled_request")
    if isinstance(compiled_request, dict) and "candidate_id" in compiled_request:
        compiled_request["candidate_id"] = segment_candidate_ids[0]
    fallback_tree: dict[str, list[str]] = {
        segment_candidate_ids[0]: [*segment_candidate_ids[1:], *carried_candidate_ids],
    }
    for segment_id in segment_candidate_ids[1:]:
        fallback_tree[segment_id] = list(segment_candidate_ids)
    for carried_id in carried_candidate_ids:
        fallback_tree[carried_id] = list(segment_candidate_ids)
    freeze_block["fallback_tree"] = fallback_tree
    freeze_block["alternate_candidate_ids"] = [
        cid for cid in segment_candidate_ids[1:]
    ] + carried_candidate_ids
    freeze_block.setdefault("failure_tree", {
        "version": FAILURE_TREE_VERSION,
        "codes": list(FAILURE_TREE_CODE_ORDER),
    })
    freeze_block.setdefault("budget_accounting", list(BUDGET_ACCOUNTING_CATEGORIES))

    policy_hashes = copy.deepcopy(plan.get("policy_hashes") or {})
    policy_hashes["intermediate_staging_rules_sha256"] = rule_set.content_sha256

    staging_block = _staging_block(
        rules=rule_set,
        intermediate_sha256=intermediate_sha,
        confirmed=confirmed_pairs,
        segments=segments,
        original_plan=plan,
        segment_candidate_ids=segment_candidate_ids,
        carried_candidate_ids=carried_candidate_ids,
    )

    fields: dict[str, Any] = {
        "reaction_id": reaction_id,
        "case_id": str(plan["case_id"]),
        "split": str(plan.get("split") or "unassigned"),
        "plan_id": new_plan_id,
        "plan_version": new_version,
        "source_proposal_sha256": str(plan.get("source_proposal_sha256") or ""),
        "source_case_sha256": str(plan.get("source_case_sha256") or ""),
        "policy_hashes": policy_hashes,
        "backend": copy.deepcopy(plan.get("backend")),
        "candidate_graph": graph,
        "candidates": segment_candidates + carried_candidates,
        "budget": copy.deepcopy(plan.get("budget")),
        "compiled": copy.deepcopy(plan.get("compiled")),
        "quality_tests": copy.deepcopy(plan.get("quality_tests")),
        "extensions": {"freeze": freeze_block, "staging": staging_block},
        "supersedes": str(plan["plan_id"]),
    }
    try:
        new_plan = make_generation_plan(new_plan_id, "frozen", **fields)
    except ContractError as exc:
        raise IntermediateStagingError(CODE_STAGING_INPUT_INVALID, str(exc)) from exc

    superseded = copy.deepcopy(plan)
    superseded["status"] = "superseded"
    superseded["supersedes"] = new_plan_id
    superseded = seal_document(superseded)
    superseded_problems = validate_v2_document(superseded)
    if superseded_problems:
        raise IntermediateStagingError(
            CODE_STAGING_INPUT_INVALID,
            "superseded original failed contracts_v2 validation: "
            + "; ".join(superseded_problems[:5]),
        )

    return StagingResult(
        new_plan=new_plan,
        superseded_original=superseded,
        segments=segments,
        intermediate_sha256=intermediate_sha,
        rules_version=rule_set.rules_version,
        rules_sha256=rule_set.content_sha256,
        confirmed_frame_indices=tuple(pair[0].frame_index for pair in confirmed_pairs),
        formal_conclusions=tuple(pair[1].formal_conclusion for pair in confirmed_pairs),
    )


__all__ = [
    "CODE_COORDINATE_UNCHANGED_NOT_MECHANISM",
    "CODE_ENERGY_LOCAL_MINIMUM_MISSING",
    "CODE_GEOMETRY_SHAPE_INVALID",
    "CODE_HESSIAN_CONTRADICTS_STABILITY",
    "CODE_INSUFFICIENT_STABILITY_EVIDENCE",
    "CODE_INSUFFICIENT_TOPOLOGY_EVIDENCE",
    "CODE_INTERMEDIATE_GEOMETRY_MISSING",
    "CODE_NO_STABLE_INTERMEDIATE",
    "CODE_PLAN_NOT_FROZEN",
    "CODE_STAGING_INPUT_INVALID",
    "FROZEN_INTERMEDIATE_RULES",
    "FORBIDDEN_STAGING_KEYS",
    "FORMAL_CONFIRMED",
    "FORMAL_CONTRADICTED",
    "FORMAL_PENDING_EVIDENCE",
    "HESSIAN_COMPLETE",
    "HESSIAN_PENDING",
    "OBJECT_INTERMEDIATE_STAGING",
    "REF_PRODUCT",
    "REF_REACTANT",
    "RULE_CLASS_CANDIDATE_FRAME",
    "RULE_CLASS_CONSTRAINT_RELEASE",
    "RULE_CLASS_HESSIAN_FREQUENCY",
    "RULE_CLASS_TOPOLOGY_CHARGE_ELECTRONIC",
    "RULE_CLASSES",
    "SCHEMA_INTERMEDIATE_STAGING",
    "STAGING_REFUSAL_CODES",
    "STAGING_RULES_VERSION",
    "STAGING_RUNS_ON",
    "STAGING_STAGE_ID",
    "VERDICT_CONFIRMED_STABLE",
    "VERDICT_REFUSED",
    "ConfirmationEvidence",
    "HessianEvidence",
    "IntermediateAdjudication",
    "IntermediateEvidence",
    "IntermediateRule",
    "IntermediateRuleSet",
    "IntermediateStagingError",
    "OptimizationEvidence",
    "PathSegment",
    "StagingResult",
    "adjudicate_intermediate",
    "build_adjacent_segments",
    "confirmed_intermediate_sha256",
    "frozen_intermediate_rules",
    "parse_intermediate_evidence",
    "stage_new_plan_version",
]
