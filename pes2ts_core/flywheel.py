"""R5/R6 data-flywheel interfaces: cost ledger, label_state, provenance, split firewall.

G2-AB1 WP-7 freezes the OFFLINE interfaces and guards only — no model is
trained and no real holdout is frozen here (that is G3.0).  Constitution
references: R5 (full attempt/cost retention), R6 (evaluation integrity and
the split firewall), §9.1 (data flywheel keeps ALL candidates), §9.2
(``label_state`` seven states — ``unattempted`` is never a failure), §9.3
(provenance four layers), §9.4 (cost ledger: missing values are ``null`` +
reason, never a silent zero; a numerical Hessian's internal gradients are
counted once), §11.2 (reaction-family holdout; no tuning on the holdout).
"""
from __future__ import annotations

import math
from typing import Any

#: Constitution §9.2 — the seven-state label vocabulary (versioned interface).
LABEL_STATES = ("verified_target", "verified_non_target", "optimizer_failed",
                "validation_incomplete", "execution_failed", "unattempted",
                "censored_budget")

#: Constitution §9.3 — provenance layers, all four required.
PROVENANCE_LAYERS = ("geometry", "mapping", "cohort_selection", "development_exposure")

#: R5 counter fields every attempt record must write.
COUNTER_FIELDS = ("N_energy", "N_gradient", "N_Hessian", "N_OptTS_trials")

_SPLITS = ("train", "valid", "test")


def cost_ledger(*, cost_id: str, n_energy: int, n_gradient: int, n_hessian: int,
                n_optts_trials: int, cpu_seconds: float | None,
                cpu_seconds_reason: str | None = None,
                allocated_core_seconds: float | None = None,
                allocated_core_seconds_reason: str | None = None,
                includes_cost_ids: list[str] | None = None,
                numerical_hessian_internal_gradients: int = 0,
                numerical_hessian_gradients_counted_in_N_gradient: bool = False,
                cost_source: str | None = None) -> dict[str, Any]:
    """Validate and freeze one attempt's R5 cost ledger.

    ``cpu_seconds`` (measured) and ``allocated_core_seconds`` (allocated) are
    SEPARATE accounts.  A missing value must be ``None`` WITH a reason —
    writing 0 for "unknown" is rejected.  A numerical Hessian's internal
    gradient evaluations must not be double-counted in ``N_gradient``.
    """
    if not isinstance(cost_id, str) or not cost_id:
        raise ValueError("INVALID_COST_ID")
    counters = {"N_energy": n_energy, "N_gradient": n_gradient,
                "N_Hessian": n_hessian, "N_OptTS_trials": n_optts_trials}
    for name, value in counters.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"INVALID_COST_COUNTER:{name}")
    if isinstance(numerical_hessian_internal_gradients, bool) \
            or not isinstance(numerical_hessian_internal_gradients, int) \
            or numerical_hessian_internal_gradients < 0:
        raise ValueError("INVALID_COST_COUNTER:numerical_hessian_internal_gradients")
    if numerical_hessian_internal_gradients and numerical_hessian_gradients_counted_in_N_gradient:
        raise ValueError("NUMERICAL_HESSIAN_GRADIENTS_DOUBLE_COUNTED")
    ledger: dict[str, Any] = {"cost_id": cost_id, **counters,
                              "includes_cost_ids": list(includes_cost_ids or [])}
    for name, value, reason in (("cpu_seconds", cpu_seconds, cpu_seconds_reason),
                                ("allocated_core_seconds", allocated_core_seconds,
                                 allocated_core_seconds_reason)):
        if value is None:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"MISSING_COST_NEEDS_REASON:{name}")
            ledger[name] = None
            ledger[f"{name}_reason"] = reason
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"INVALID_COST_VALUE:{name}")
            ledger[name] = value
    if numerical_hessian_internal_gradients:
        ledger["numerical_hessian_internal_gradients"] = numerical_hessian_internal_gradients
        ledger["numerical_hessian_gradients_counted_in_N_gradient"] = False
    if cost_source:
        ledger["cost_source"] = cost_source
    return ledger


def aggregate_costs(ledgers: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum R5 ledgers with parent-child deduplication.

    A ledger listed in another ledger's ``includes_cost_ids`` is already
    accounted for by its parent and is never summed twice.
    """
    if not isinstance(ledgers, list) or not ledgers:
        raise ValueError("INVALID_LEDGER_LIST")
    by_id: dict[str, dict[str, Any]] = {}
    for ledger in ledgers:
        if not isinstance(ledger, dict) or "cost_id" not in ledger:
            raise ValueError("INVALID_LEDGER_RECORD")
        cost_id = ledger["cost_id"]
        if cost_id in by_id:
            raise ValueError(f"DUPLICATE_COST_ID:{cost_id}")
        by_id[cost_id] = ledger
    included = {child for ledger in by_id.values()
                for child in ledger.get("includes_cost_ids", [])}
    unknown = included - set(by_id)
    if unknown:
        raise ValueError(f"UNKNOWN_INCLUDED_COST_ID:{sorted(unknown)[0]}")
    roots = [ledger for cost_id, ledger in sorted(by_id.items()) if cost_id not in included]
    totals = {name: 0 for name in COUNTER_FIELDS}
    for ledger in roots:
        for name in COUNTER_FIELDS:
            totals[name] += int(ledger.get(name, 0))
    cpu_values = [ledger["cpu_seconds"] for ledger in roots if ledger.get("cpu_seconds") is not None]
    allocated = [ledger["allocated_core_seconds"] for ledger in roots
                 if ledger.get("allocated_core_seconds") is not None]
    return {**totals,
            "cpu_seconds": sum(cpu_values) if len(cpu_values) == len(roots) else None,
            "allocated_core_seconds": sum(allocated) if len(allocated) == len(roots) else None,
            "n_root_ledgers": len(roots),
            "n_ledgers_total": len(by_id)}


def label_record(reaction_id: str, frame_id: str | None, label_state: str, *,
                 evidence_refs: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate one flywheel label record (constitution §9.2).

    ``unattempted`` is a distinct state: an unrun neighbour frame is never a
    hard negative and never collapses into a failure state.  A
    ``verified_target``/``verified_non_target`` label must carry the OptTS /
    frequency / IRC evidence references that produced it (R2: the label IS
    the validation chain).
    """
    if label_state not in LABEL_STATES:
        raise ValueError(f"INVALID_LABEL_STATE:{label_state}")
    record: dict[str, Any] = {"reaction_id": reaction_id, "frame_id": frame_id,
                              "label_state": label_state,
                              "unattempted_is_not_failure": label_state == "unattempted"}
    if label_state in {"verified_target", "verified_non_target"}:
        refs = evidence_refs or {}
        missing = [stage for stage in ("optts", "frequency", "irc")
                   if not isinstance(refs.get(stage), str) or not refs[stage].strip()]
        if missing:
            raise ValueError(f"VERIFIED_LABEL_MISSING_EVIDENCE:{missing[0]}")
        record["evidence_refs"] = dict(refs)
    return record


def provenance_four_layers(*, geometry: str, mapping: str, cohort_selection: str,
                           development_exposure: str) -> dict[str, str]:
    """Validate the mandatory provenance layers (constitution §9.3).

    Field isolation is not source independence: the mapping layer must state
    its truth assistance (e.g. ``truth_assisted_p1``) instead of pretending
    the pipeline is end-to-end truth-free.
    """
    layers = {"geometry": geometry, "mapping": mapping,
              "cohort_selection": cohort_selection,
              "development_exposure": development_exposure}
    for name, value in layers.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"INVALID_PROVENANCE_LAYER:{name}")
    return layers


def split_firewall_violations(split_assignment: dict[str, str], *,
                              reaction_family: dict[str, str],
                              reverse_pairs: list[tuple[str, str]] | None = None,
                              frame_sibling_pairs: list[tuple[str, str]] | None = None,
                              ) -> list[dict[str, Any]]:
    """R6 split-firewall check over coupled entities.

    - every reaction in a reaction family must share one split;
    - a reaction written in reverse (same transformation, other direction)
      must share the split of its forward twin;
    - adjacent frames of one path, mapping copies, and symmetry copies must
      never straddle two splits.
    """
    if not isinstance(split_assignment, dict):
        raise ValueError("INVALID_SPLIT_ASSIGNMENT")
    violations: list[dict[str, Any]] = []
    for reaction_id, split in sorted(split_assignment.items()):
        if split not in _SPLITS:
            violations.append({"kind": "INVALID_SPLIT", "reaction_id": reaction_id, "split": split})
    families: dict[str, set[str]] = {}
    for reaction_id, family in reaction_family.items():
        if reaction_id in split_assignment:
            families.setdefault(family, set()).add(reaction_id)
    for family, members in sorted(families.items()):
        splits = {split_assignment[m] for m in members}
        if len(splits) > 1:
            violations.append({"kind": "FAMILY_CROSSES_SPLITS", "family": family,
                               "reaction_ids": sorted(members), "splits": sorted(splits)})
    for forward, reverse in reverse_pairs or []:
        if forward in split_assignment and reverse in split_assignment \
                and split_assignment[forward] != split_assignment[reverse]:
            violations.append({"kind": "REVERSE_REACTION_CROSSES_SPLITS",
                               "reaction_ids": [forward, reverse]})
    for left, right in frame_sibling_pairs or []:
        if left in split_assignment and right in split_assignment \
                and split_assignment[left] != split_assignment[right]:
            violations.append({"kind": "COPIES_CROSS_SPLITS", "reaction_ids": [left, right]})
    return violations


def freeze_holdout(family_splits: dict[str, str], *, seed: int) -> dict[str, Any]:
    """Freeze the reaction-family holdout BEFORE any truth/label inspection.

    The frozen manifest is the only legitimate tuning baseline; the guard
    below refuses any later tuning that touches a holdout family (R6).
    """
    if not isinstance(seed, bool) and not isinstance(seed, int):
        raise ValueError("INVALID_HOLDOUT_SEED")
    for family, split in family_splits.items():
        if split not in _SPLITS:
            raise ValueError(f"INVALID_HOLDOUT_SPLIT:{family}")
    return {"schema_version": "pes2ts_holdout_freeze_v1", "seed": seed,
            "family_splits": dict(sorted(family_splits.items())),
            "frozen_before_truth_access": True,
            "tuning_allowed_on": sorted(f for f, s in family_splits.items() if s == "train")}


def assert_no_holdout_tuning(frozen: dict[str, Any], tuned_families: list[str]) -> None:
    """Refuse tuning records that touch any non-train (holdout) family."""
    if frozen.get("schema_version") != "pes2ts_holdout_freeze_v1":
        raise ValueError("INVALID_HOLDOUT_FREEZE_MANIFEST")
    allowed = set(frozen.get("tuning_allowed_on", []))
    offender = next((f for f in tuned_families if f not in allowed), None)
    if offender is not None:
        raise ValueError(f"HOLDOUT_TUNING_FORBIDDEN:{offender}")
