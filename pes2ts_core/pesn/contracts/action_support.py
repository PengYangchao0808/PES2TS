"""Action-support registry contract (``pesn_action_support_registry_v1``).

The registry answers ONE question conservatively: does the recorded data
contain an EXECUTED action, with a measured response, at a compatible
decision state, for the candidate displacement being scored?  It never
decides legality (ChemicalControl/Shield own that), never fabricates
counterfactual responses, and never treats source labels as support — the
same command vector queried under two different source labels must get the
same verdict (RL repair plan §5.1).

First-version scope: same-reaction support only.  A group matches on
environment hash, reward-spec hash, coordinate-spec digest, the ORDERED
driver-id tuple, and the epsilon vector; cross-reaction transfer requires an
independently validated graph/coordinate correspondence and is out of scope
for v1.

Document layout (all calibration values are frozen at build time from fit
data only)::

    {
      "schema_version": "pesn_action_support_registry_v1",
      "environment_hash": str,
      "reward_spec_sha256": str,
      "built_from": {"fit_transition_ids": [...], "n_fit_transitions": int},
      "calibration": {
        "action_tolerance_std": float,
        "state_tolerance_angstrom": float,
        "context_tolerance": float,
        "policy": "min-pairwise-half | defaults"
      },
      "groups": [
        {
          "reaction_id": str,
          "driver_ids": [str, ...],          # ordered driver semantics
          "epsilon": [float, ...],
          "coordinate_spec_sha256": str,
          "n_states": int,
          "entries": [
            {
              "transition_id": str,
              "state_geometry_std": [[x,y,z], ...],   # centroid-translated
              "context_vector": [float, ...],
              "residual_u": [float, ...],
              "accepted": bool,
              "episode_status": str,
              "source_label": str        # audit only, never read for support
            }
          ]
        }
      ]
    }
"""

from __future__ import annotations

from typing import Any, Mapping

ACTION_SUPPORT_REGISTRY_SCHEMA = "pesn_action_support_registry_v1"

REASON_SPEC_MISMATCH = "control_spec_mismatch"
REASON_DIMENSION_MISMATCH = "dimension_mismatch"
REASON_STATE_NOT_COVERED = "state_not_covered"
REASON_ACTION_NOT_COVERED = "action_not_covered"

_REGISTRY_TOP_REQUIRED = (
    "schema_version",
    "environment_hash",
    "reward_spec_sha256",
    "calibration",
    "groups",
)
_REGISTRY_GROUP_REQUIRED = (
    "reaction_id",
    "driver_ids",
    "epsilon",
    "coordinate_spec_sha256",
    "entries",
)
_REGISTRY_ENTRY_REQUIRED = (
    "transition_id",
    "state_geometry_std",
    "context_vector",
    "residual_u",
    "accepted",
    "episode_status",
    "source_label",
)
_CALIBRATION_REQUIRED = (
    "action_tolerance_std",
    "state_tolerance_angstrom",
    "context_tolerance",
    "policy",
)


class ActionSupportContractError(ValueError):
    """Typed registry-document validation failure."""


def validate_action_support_registry(doc: Mapping[str, Any]) -> None:
    if doc.get("schema_version") != ACTION_SUPPORT_REGISTRY_SCHEMA:
        raise ActionSupportContractError(
            f"registry schema {doc.get('schema_version')!r} unsupported"
        )
    for key in _REGISTRY_TOP_REQUIRED:
        if key not in doc:
            raise ActionSupportContractError(f"registry missing key {key!r}")
    calibration = doc["calibration"]
    if not isinstance(calibration, Mapping):
        raise ActionSupportContractError("calibration must be a mapping")
    for key in _CALIBRATION_REQUIRED:
        if key not in calibration:
            raise ActionSupportContractError(f"calibration missing key {key!r}")
    groups = doc["groups"]
    if not isinstance(groups, list):
        raise ActionSupportContractError("groups must be a list")
    seen_groups: set[tuple] = set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise ActionSupportContractError("group must be a mapping")
        for key in _REGISTRY_GROUP_REQUIRED:
            if key not in group:
                raise ActionSupportContractError(f"group missing key {key!r}")
        driver_ids = tuple(group["driver_ids"])
        epsilon = tuple(float(v) for v in group["epsilon"])
        if not driver_ids or len(driver_ids) != len(epsilon):
            raise ActionSupportContractError(
                "driver_ids and epsilon must be parallel non-empty tuples"
            )
        identity = (str(group["reaction_id"]), driver_ids)
        if identity in seen_groups:
            raise ActionSupportContractError(
                f"duplicate group identity {identity}; ambiguous driver order"
            )
        seen_groups.add(identity)
        seen_transitions: set[str] = set()
        for entry in group["entries"]:
            if not isinstance(entry, Mapping):
                raise ActionSupportContractError("entry must be a mapping")
            for key in _REGISTRY_ENTRY_REQUIRED:
                if key not in entry:
                    raise ActionSupportContractError(f"entry missing key {key!r}")
            residual = entry["residual_u"]
            if len(residual) != len(driver_ids):
                raise ActionSupportContractError(
                    "entry residual_u dimension does not match driver_ids"
                )
            transition_id = str(entry["transition_id"])
            if transition_id in seen_transitions:
                raise ActionSupportContractError(
                    f"duplicate transition entry {transition_id!r}"
                )
            seen_transitions.add(transition_id)


__all__ = [
    "ACTION_SUPPORT_REGISTRY_SCHEMA",
    "ActionSupportContractError",
    "REASON_ACTION_NOT_COVERED",
    "REASON_DIMENSION_MISMATCH",
    "REASON_SPEC_MISMATCH",
    "REASON_STATE_NOT_COVERED",
    "validate_action_support_registry",
]
