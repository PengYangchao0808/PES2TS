"""Offline action-support registry builder (RL repair plan §5.1).

Builds a ``pesn_action_support_registry_v1`` document from FIT transitions
only.  Every registry entry is an EXECUTED action with its measured
response (accepted / episode status); offered-but-unexecuted candidates and
source labels never become support.  Tolerances are calibrated on the fit
data itself and frozen into the document:

* ``action_tolerance_std`` — half the minimum pairwise L∞ distance between
  distinct standardized executed actions (so two recorded actions remain
  distinguishable), clamped to [1e-6, 0.25].
* ``state_tolerance_angstrom`` — the 10th percentile of intra-group
  nearest-neighbour standardized-geometry L∞ distances divided by 10,
  clamped to [1e-6, 0.05]; single-state groups use 1e-3.
* ``context_tolerance`` — 1e-6 (budget/target context must match to
  numerical precision in v1; paired-collection states will match exactly by
  construction).

dev data is never used for calibration or registry content (it exists for
model selection only).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.action_support import (
    ACTION_SUPPORT_REGISTRY_SCHEMA,
)
from pes2ts_core.pesn.inference.action_support import standardize_geometry
from pes2ts_core.pesn.offline.transitions import QualifiedTransition

DEFAULT_ACTION_TOLERANCE_STD = 1e-6
DEFAULT_STATE_TOLERANCE_ANGSTROM = 1e-3
CONTEXT_TOLERANCE = 1e-6


def coordinate_spec_sha256(case: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(case["coordinate_spec"], sort_keys=True).encode("utf-8")
    ).hexdigest()


def _calibrate_action_tolerance(groups_u_std: list[np.ndarray]) -> tuple[float, str]:
    distances: list[float] = []
    for matrix in groups_u_std:
        if matrix.shape[0] < 2:
            continue
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[0]):
                distance = float(np.max(np.abs(matrix[i] - matrix[j])))
                if distance > 1e-9:
                    distances.append(distance)
    if not distances:
        return DEFAULT_ACTION_TOLERANCE_STD, "default_no_distinct_actions"
    half_min = 0.5 * min(distances)
    clamped = min(0.25, max(1e-6, half_min))
    return clamped, "min-pairwise-half"


def _calibrate_state_tolerance(group_states: list[np.ndarray]) -> tuple[float, str]:
    nearest: list[float] = []
    for states in group_states:
        if states.shape[0] < 2:
            continue
        for i in range(states.shape[0]):
            others = np.delete(states, i, axis=0)
            nearest.append(float(np.min(np.max(np.abs(others - states[i]), axis=1))))
    if not nearest:
        return DEFAULT_STATE_TOLERANCE_ANGSTROM, "default_single_state"
    quantile = float(np.quantile(nearest, 0.10))
    clamped = min(0.05, max(1e-6, quantile / 10.0))
    return clamped, "nearest-neighbour-q10-over-10"


def build_action_support_registry(
    transitions: Sequence[QualifiedTransition],
    *,
    environment_hash: str,
    reward_spec_sha256: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for transition in transitions:
        doc = transition.doc
        action = doc["action"]
        driver_ids = tuple(str(d) for d in action["driver_ids"])
        epsilon = tuple(float(v) for v in action["epsilon"])
        key = (transition.reaction_id, driver_ids)
        group = grouped.setdefault(
            key,
            {
                "reaction_id": transition.reaction_id,
                "driver_ids": list(driver_ids),
                "epsilon": list(epsilon),
                "coordinate_spec_sha256": coordinate_spec_sha256(transition.case),
                "entries": [],
                "_u_std": [],
                "_states": [],
            },
        )
        if tuple(group["epsilon"]) != epsilon:
            raise ValueError(
                f"{transition.reaction_id}: epsilon changed inside one driver group; "
                "refusing to build a registry over incompatible specs"
            )
        residual = tuple(float(v) for v in action["residual_u"])
        if len(residual) != len(driver_ids):
            raise ValueError(
                f"{transition.reaction_id}: residual dimension != driver count"
            )
        epsilon_vec = np.asarray(epsilon, dtype=float)
        state_std = standardize_geometry(doc["decision_start_geometry"])
        group["entries"].append({
            "transition_id": str(doc.get("transition_id", "")),
            "state_geometry_std": [
                [round(float(x), 9) for x in row] for row in state_std
            ],
            "context_vector": [
                round(float(v), 9)
                for v in (doc.get("decision_context", {}).get("vector") or [])
            ],
            "residual_u": list(residual),
            "accepted": bool(doc.get("accepted")),
            "episode_status": str(doc.get("episode_status")),
            "source_label": str(action.get("source")),
        })
        group["_u_std"].append(
            [float(x) for x in np.asarray(residual) / epsilon_vec]
        )
        group["_states"].append(state_std)

    action_tolerance, action_policy = _calibrate_action_tolerance(
        [np.asarray(g["_u_std"], dtype=float) for g in grouped.values() if g["_u_std"]]
    )
    state_tolerance, state_policy = _calibrate_state_tolerance(
        [np.asarray(g["_states"], dtype=float) for g in grouped.values() if g["_states"]]
    )

    groups_payload: list[dict[str, Any]] = []
    for group in grouped.values():
        entries = group["entries"]
        groups_payload.append({
            "reaction_id": group["reaction_id"],
            "driver_ids": group["driver_ids"],
            "epsilon": group["epsilon"],
            "coordinate_spec_sha256": group["coordinate_spec_sha256"],
            "n_states": len({entry["transition_id"] for entry in entries}),
            "entries": entries,
        })

    return {
        "schema_version": ACTION_SUPPORT_REGISTRY_SCHEMA,
        "environment_hash": str(environment_hash),
        "reward_spec_sha256": str(reward_spec_sha256),
        "built_from": {
            "fit_transition_ids": sorted(
                str(t.doc.get("transition_id", "")) for t in transitions
            ),
            "n_fit_transitions": len(transitions),
        },
        "calibration": {
            "action_tolerance_std": action_tolerance,
            "state_tolerance_angstrom": state_tolerance,
            "context_tolerance": CONTEXT_TOLERANCE,
            "policy": f"{action_policy}+{state_policy}",
        },
        "groups": groups_payload,
    }


__all__ = [
    "build_action_support_registry",
    "coordinate_spec_sha256",
]
