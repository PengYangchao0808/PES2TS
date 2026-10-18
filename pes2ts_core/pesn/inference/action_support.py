"""Read-only action-support registry query (RL repair plan §5).

Pure inference: given a validated ``pesn_action_support_registry_v1``
document, answer whether a candidate displacement vector has data support at
a compatible decision state.  The verdict is a function of (control-spec
identity, standardized state, standardized action) ONLY — source labels in
the registry are audit fields and are never consulted, so the same command
queried under different source labels always returns the same verdict.

This module imports nothing from ``offline/`` or the execution host; the
offline builder lives in ``pes2ts_core.pesn.offline.action_support``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.action_support import (
    REASON_ACTION_NOT_COVERED,
    REASON_DIMENSION_MISMATCH,
    REASON_SPEC_MISMATCH,
    REASON_STATE_NOT_COVERED,
    validate_action_support_registry,
)


class ActionSupportRegistryError(RuntimeError):
    """Typed registry-query failure."""


@dataclass(frozen=True)
class SupportVerdict:
    supported: bool
    reason: str | None
    nearest_state_distance: float | None
    nearest_action_distance: float | None
    neighbor_count: int
    action_tolerance_std: float
    state_tolerance_angstrom: float
    source_labels_of_neighbors: tuple[str, ...]
    #: Distance to the nearest entry that is jointly geometry- AND
    #: context-compatible (None when no entry is jointly compatible).
    #: ``nearest_state_distance`` stays the raw geometry-nearest distance so
    #: the two are never conflated (review F1).
    nearest_compatible_state_distance: float | None = None


def standardize_geometry(geometry: Sequence[Sequence[float]]) -> np.ndarray:
    """Centroid-translated coordinates (translation-invariant state key)."""
    coords = np.asarray(geometry, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ActionSupportRegistryError("geometry must be a non-empty (N,3) array")
    return coords - coords.mean(axis=0)


def _linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def json_roundtrip(document: Mapping) -> dict:
    """JSON-clean copy of the document; ndarrays or like payloads fail loudly."""
    import json

    return json.loads(json.dumps(dict(document)))


class ActionSupportRegistry:
    """Frozen, read-only view over a validated registry document."""

    def __init__(self, document: Mapping) -> None:
        validate_action_support_registry(document)
        self.document = json_roundtrip(document)
        self.environment_hash = str(document["environment_hash"])
        self.reward_spec_sha256 = str(document["reward_spec_sha256"])
        calibration = document["calibration"]
        self.action_tolerance_std = float(calibration["action_tolerance_std"])
        self.state_tolerance_angstrom = float(
            calibration["state_tolerance_angstrom"]
        )
        self.context_tolerance = float(calibration["context_tolerance"])
        self._group_payloads: dict[tuple[str, tuple[str, ...]], dict] = {
            (str(group["reaction_id"]), tuple(str(d) for d in group["driver_ids"])): group
            for group in document["groups"]
        }
        self._state_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        self._context_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        self._u_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        self._epsilon_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        for key, group in self._group_payloads.items():
            self._state_cache[key] = np.asarray(
                [entry["state_geometry_std"] for entry in group["entries"]],
                dtype=float,
            )
            self._context_cache[key] = np.asarray(
                [entry["context_vector"] for entry in group["entries"]],
                dtype=float,
            )
            self._u_cache[key] = np.asarray(
                [entry["residual_u"] for entry in group["entries"]],
                dtype=float,
            )
            self._epsilon_cache[key] = np.asarray(group["epsilon"], dtype=float)

    @classmethod
    def from_json(cls, path) -> "ActionSupportRegistry":
        import json
        from pathlib import Path

        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def query(
        self,
        *,
        reaction_id: str,
        driver_ids: Sequence[str],
        epsilon: Sequence[float],
        environment_hash: str,
        reward_spec_sha256: str,
        coordinate_spec_sha256: str,
        geometry: Sequence[Sequence[float]],
        context_vector: Sequence[float],
        candidate_u: Sequence[float],
    ) -> SupportVerdict:
        def verdict(
            supported: bool,
            reason: str | None,
            state_distance: float | None,
            action_distance: float | None,
            neighbors: int = 0,
            labels: tuple[str, ...] = (),
            nearest_compatible_state_distance: float | None = None,
        ) -> SupportVerdict:
            return SupportVerdict(
                supported=supported,
                reason=reason,
                nearest_state_distance=state_distance,
                nearest_action_distance=action_distance,
                neighbor_count=neighbors,
                action_tolerance_std=self.action_tolerance_std,
                state_tolerance_angstrom=self.state_tolerance_angstrom,
                source_labels_of_neighbors=labels,
                nearest_compatible_state_distance=nearest_compatible_state_distance,
            )

        if environment_hash != self.environment_hash:
            return verdict(False, REASON_SPEC_MISMATCH, None, None)
        if reward_spec_sha256 != self.reward_spec_sha256:
            return verdict(False, REASON_SPEC_MISMATCH, None, None)
        driver_key = tuple(str(d) for d in driver_ids)
        group_key = (str(reaction_id), driver_key)
        group = self._group_payloads.get(group_key)
        if group is None:
            return verdict(False, REASON_SPEC_MISMATCH, None, None)
        if str(group["coordinate_spec_sha256"]) != str(coordinate_spec_sha256):
            return verdict(False, REASON_SPEC_MISMATCH, None, None)
        if len(candidate_u) != len(driver_key) or len(epsilon) != len(driver_key):
            return verdict(False, REASON_DIMENSION_MISMATCH, None, None)
        if not group["entries"]:
            return verdict(False, REASON_STATE_NOT_COVERED, None, None)

        states = self._state_cache[group_key]
        contexts = self._context_cache[group_key]
        state = standardize_geometry(geometry)
        if state.shape != states.shape[1:]:
            return verdict(False, REASON_DIMENSION_MISMATCH, None, None)
        state_distances = np.asarray(
            [_linf(state, cached) for cached in states]
        )
        nearest_state_distance = float(state_distances.min())
        context = np.asarray(context_vector, dtype=float)
        if context.shape != contexts.shape[1:]:
            return verdict(False, REASON_DIMENSION_MISMATCH, None, None)
        context_distances = np.asarray(
            [_linf(context, cached) for cached in contexts]
        )
        compatible_mask = (
            state_distances <= self.state_tolerance_angstrom
        ) & (context_distances <= self.context_tolerance)
        covered_rows = np.flatnonzero(compatible_mask)
        nearest_compatible = (
            float(state_distances[covered_rows].min())
            if len(covered_rows)
            else None
        )
        if not len(covered_rows):
            return verdict(
                False,
                REASON_STATE_NOT_COVERED,
                nearest_state_distance,
                None,
                nearest_compatible_state_distance=nearest_compatible,
            )

        epsilon_vec = self._epsilon_cache[group_key]
        if not np.allclose(epsilon_vec, np.asarray(epsilon, dtype=float), rtol=0.0, atol=1e-12):
            return verdict(False, REASON_SPEC_MISMATCH, nearest_state_distance, None)
        u_std = np.asarray(candidate_u, dtype=float) / epsilon_vec
        u_rows = self._u_cache[group_key]
        action_distances = np.asarray(
            [
                _linf(u_std, row / epsilon_vec)
                for row in u_rows[covered_rows]
            ]
        )
        within = action_distances <= self.action_tolerance_std
        neighbor_indices = covered_rows[within]
        labels = tuple(
            str(group["entries"][int(i)]["source_label"])
            for i in neighbor_indices
        )
        nearest_action = float(action_distances.min())
        if not within.any():
            return verdict(
                False,
                REASON_ACTION_NOT_COVERED,
                nearest_state_distance,
                nearest_action,
                nearest_compatible_state_distance=nearest_compatible,
            )
        return verdict(
            True,
            None,
            nearest_state_distance,
            nearest_action,
            neighbors=int(within.sum()),
            labels=labels,
            nearest_compatible_state_distance=nearest_compatible,
        )

    def query_actions(
        self,
        *,
        reaction_id: str,
        driver_ids: Sequence[str],
        epsilon: Sequence[float],
        environment_hash: str,
        reward_spec_sha256: str,
        coordinate_spec_sha256: str,
        geometry: Sequence[Sequence[float]],
        context_vector: Sequence[float],
        candidate_actions: Sequence[Sequence[float]],
    ) -> list[SupportVerdict]:
        return [
            self.query(
                reaction_id=reaction_id,
                driver_ids=driver_ids,
                epsilon=epsilon,
                environment_hash=environment_hash,
                reward_spec_sha256=reward_spec_sha256,
                coordinate_spec_sha256=coordinate_spec_sha256,
                geometry=geometry,
                context_vector=context_vector,
                candidate_u=candidate,
            )
            for candidate in candidate_actions
        ]


class LegacySourceSupportGate:
    """Pre-registry semantics: source-label allow-list, explicitly legacy.

    Bundles published before the registry keep their original gate; callers
    must flag results ``legacy_support`` when using this class (plan §5.2).
    """

    legacy_support = True

    def __init__(self, supported_sources: Sequence[str]) -> None:
        self.supported_sources = frozenset(str(s) for s in supported_sources)

    def __call__(
        self,
        _state,
        _candidates: Sequence[Sequence[float]],
        sources: Sequence[str],
    ) -> list[bool]:
        return [str(source) in self.supported_sources for source in sources]


__all__ = [
    "ActionSupportRegistry",
    "ActionSupportRegistryError",
    "LegacySourceSupportGate",
    "SupportVerdict",
    "standardize_geometry",
]
