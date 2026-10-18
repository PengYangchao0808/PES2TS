"""Research prediction contract ``pesn_teacher_prediction_v1`` and capabilities.

A :class:`TeacherPrediction` is a research output of the IRC teacher heads —
progress, path proximity, path score, low-dim coordinate direction, and
coordinate activity — each with per-head availability and a typed reason when
unavailable.  It is deliberately distinct from any formal decision estimate:
it carries no chosen action, no execution result, and no success claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

from pes2ts_core.pesn.contracts import PESNError

PREDICTION_SCHEMA_VERSION: Final[str] = "pesn_teacher_prediction_v1"

CAP_ENCODER: Final[str] = "encoder"
CAP_PROGRESS: Final[str] = "progress"
CAP_PATH_PROXIMITY: Final[str] = "path_proximity"
CAP_TEACHER_DIRECTION: Final[str] = "teacher_direction"
CAP_COORDINATE_ACTIVITY: Final[str] = "coordinate_activity"
CAP_P_SEED: Final[str] = "p_seed"
CAP_C_FULL: Final[str] = "c_full"
CAP_Q_STOP: Final[str] = "q_stop"
CAP_Q_MOVE: Final[str] = "q_move"
CAP_PRODUCTION_CONTROL: Final[str] = "production_control"

#: Every capability key a bundle may declare, in canonical order.
CAPABILITY_KEYS: Final[tuple[str, ...]] = (
    CAP_ENCODER,
    CAP_PROGRESS,
    CAP_PATH_PROXIMITY,
    CAP_TEACHER_DIRECTION,
    CAP_COORDINATE_ACTIVITY,
    CAP_P_SEED,
    CAP_C_FULL,
    CAP_Q_STOP,
    CAP_Q_MOVE,
    CAP_PRODUCTION_CONTROL,
)

#: Capabilities that must stay disabled in the first-cycle research package.
MUST_BE_DISABLED_CAPS: Final[frozenset[str]] = frozenset(
    {CAP_P_SEED, CAP_C_FULL, CAP_Q_STOP, CAP_Q_MOVE, CAP_PRODUCTION_CONTROL}
)

CAP_ENABLED: Final[str] = "enabled"
CAP_DISABLED: Final[str] = "disabled"
CAP_RESEARCH_ONLY: Final[str] = "research_only"

HEAD_UNAVAILABLE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "head_disabled",
        "out_of_domain",
        "feature_qualification_failed",
        "backend_missing",
        "coordinate_count_zero",
        "direction_norm_invalid",
    }
)


class UnsupportedCapability(PESNError):
    """A head or capability is not enabled in the loaded model bundle."""

    def __init__(self, capability: str, reason: str) -> None:
        self.capability = capability
        self.reason = reason
        super().__init__(f"capability {capability!r} unsupported: {reason}")


class PredictionRejected(PESNError):
    """Typed rejection of one prediction request (per-sample, never silent)."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"prediction rejected [{code}]: {detail}")


@dataclass(frozen=True, slots=True)
class ScalarEstimate:
    """One scalar head output with explicit availability."""

    available: bool
    value: float | None
    reason: str | None = None

    def validate(self) -> None:
        if self.available:
            if self.value is None:
                raise PredictionRejected("HEAD_INVALID", "available estimate has no value")
            if self.reason is not None:
                raise PredictionRejected("HEAD_INVALID", "available estimate carries a reason")
        elif self.reason not in HEAD_UNAVAILABLE_REASONS:
            raise PredictionRejected(
                "HEAD_INVALID", f"unknown unavailability reason {self.reason!r}"
            )


@dataclass(frozen=True, slots=True)
class VectorEstimate:
    """Per-coordinate head output with per-coordinate mask (never silently 0)."""

    available: bool
    values: tuple[float, ...]
    mask: tuple[bool, ...]
    reason: str | None = None

    def validate(self) -> None:
        if len(self.values) != len(self.mask):
            raise PredictionRejected(
                "HEAD_INVALID", "vector estimate values/mask length mismatch"
            )
        if self.available:
            if self.reason is not None:
                raise PredictionRejected("HEAD_INVALID", "available estimate carries a reason")
        elif self.reason not in HEAD_UNAVAILABLE_REASONS:
            raise PredictionRejected(
                "HEAD_INVALID", f"unknown unavailability reason {self.reason!r}"
            )


@dataclass(frozen=True, slots=True)
class TeacherPrediction:
    """Research prediction for exactly one qualified input."""

    schema_version: str
    input_hash: str
    model_id: str
    model_hash: str
    feature_hash: str
    signed_progress: ScalarEstimate
    path_proximity: ScalarEstimate
    path_score: ScalarEstimate
    teacher_direction: VectorEstimate
    coordinate_activity: VectorEstimate
    research_only: bool = True

    def validate(self) -> None:
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise PredictionRejected(
                "SCHEMA_UNKNOWN", f"prediction schema {self.schema_version!r} unsupported"
            )
        if not self.research_only:
            raise PredictionRejected(
                "HEAD_INVALID", "first-cycle outputs must stay research_only"
            )
        for head in (self.signed_progress, self.path_proximity, self.path_score):
            head.validate()
        self.teacher_direction.validate()
        self.coordinate_activity.validate()
        n_coords = len(self.teacher_direction.values)
        if self.teacher_direction.available and n_coords == 0:
            raise PredictionRejected(
                "coordinate_count_zero", "direction head has no coordinates"
            )


def validate_capability_map(caps: Mapping[str, str]) -> None:
    """Reject malformed or illegally-enabled capability maps."""
    unknown = set(caps) - set(CAPABILITY_KEYS)
    if unknown:
        raise UnsupportedCapability(
            sorted(unknown)[0], f"unknown capability keys: {sorted(unknown)}"
        )
    missing = set(CAPABILITY_KEYS) - set(caps)
    if missing:
        raise UnsupportedCapability(
            sorted(missing)[0], f"missing capability keys: {sorted(missing)}"
        )
    for key in MUST_BE_DISABLED_CAPS:
        if caps[key] != CAP_DISABLED:
            raise UnsupportedCapability(
                key, f"capability must be disabled in the research package, got {caps[key]!r}"
            )
    for key, value in caps.items():
        if value not in (CAP_ENABLED, CAP_DISABLED, CAP_RESEARCH_ONLY):
            raise UnsupportedCapability(key, f"unknown capability status {value!r}")


def research_capability_map(
    *,
    progress: bool,
    path_proximity: bool,
    teacher_direction: bool,
    coordinate_activity: bool,
) -> dict[str, str]:
    """Build the first-cycle capability map (encoder on; decision caps off)."""
    caps: dict[str, str] = {CAP_ENCODER: CAP_ENABLED}
    caps[CAP_PROGRESS] = CAP_ENABLED if progress else CAP_DISABLED
    caps[CAP_PATH_PROXIMITY] = CAP_ENABLED if path_proximity else CAP_DISABLED
    caps[CAP_TEACHER_DIRECTION] = CAP_RESEARCH_ONLY if teacher_direction else CAP_DISABLED
    caps[CAP_COORDINATE_ACTIVITY] = (
        CAP_RESEARCH_ONLY if coordinate_activity else CAP_DISABLED
    )
    for key in MUST_BE_DISABLED_CAPS:
        caps[key] = CAP_DISABLED
    validate_capability_map(caps)
    return caps


def direction_available(values: Sequence[float]) -> bool:
    """A direction vector participates only with a stable non-zero norm."""
    return len(values) > 0 and any(abs(v) > 1e-12 for v in values)


__all__ = [
    "CAPABILITY_KEYS",
    "CAP_DISABLED",
    "CAP_ENABLED",
    "CAP_RESEARCH_ONLY",
    "MUST_BE_DISABLED_CAPS",
    "PREDICTION_SCHEMA_VERSION",
    "PredictionRejected",
    "ScalarEstimate",
    "TeacherPrediction",
    "UnsupportedCapability",
    "VectorEstimate",
    "direction_available",
    "research_capability_map",
    "validate_capability_map",
]
