"""G2-AB1 WP-7: label_state seven states and the R5 cost ledger contracts."""
from __future__ import annotations

import pytest

from pes2ts_core.flywheel import (
    COUNTER_FIELDS, LABEL_STATES, aggregate_costs, cost_ledger, label_record,
    provenance_four_layers,
)

EVIDENCE = {"optts": "acp:optts-1", "frequency": "acp:freq-1", "irc": "acp:irc-1"}


def ledger(**overrides):
    base = dict(cost_id="cost-1", n_energy=10, n_gradient=8, n_hessian=1,
                n_optts_trials=2, cpu_seconds=12.5, allocated_core_seconds=48.,
                includes_cost_ids=[], numerical_hessian_internal_gradients=0)
    base.update(overrides)
    return cost_ledger(**base)


def test_label_state_has_exactly_the_seven_constitution_states():
    assert LABEL_STATES == ("verified_target", "verified_non_target", "optimizer_failed",
                            "validation_incomplete", "execution_failed", "unattempted",
                            "censored_budget")


def test_unattempted_is_never_a_failure_state():
    record = label_record("RXN_1", "frame-2", "unattempted")
    assert record["label_state"] == "unattempted"
    assert record["unattempted_is_not_failure"] is True
    for failed_state in ("optimizer_failed", "execution_failed", "censored_budget"):
        assert label_record("RXN_1", "frame-2", failed_state)["unattempted_is_not_failure"] is False
    with pytest.raises(ValueError, match="INVALID_LABEL_STATE"):
        label_record("RXN_1", "frame-2", "hard_negative")


def test_verified_labels_require_the_validation_chain_evidence():
    record = label_record("RXN_1", "frame-5", "verified_target", evidence_refs=EVIDENCE)
    assert record["evidence_refs"] == EVIDENCE
    with pytest.raises(ValueError, match="VERIFIED_LABEL_MISSING_EVIDENCE:frequency"):
        label_record("RXN_1", "frame-5", "verified_non_target",
                     evidence_refs={"optts": "a", "irc": "b"})
    with pytest.raises(ValueError, match="VERIFIED_LABEL_MISSING_EVIDENCE"):
        label_record("RXN_1", "frame-5", "verified_target")


def test_cost_ledger_writes_all_four_counters_and_separates_cpu_accounts():
    record = ledger()
    for field in COUNTER_FIELDS:
        assert field in record
    assert record["cpu_seconds"] == 12.5
    assert record["allocated_core_seconds"] == 48.
    with pytest.raises(ValueError, match="MISSING_COST_NEEDS_REASON:cpu_seconds"):
        ledger(cpu_seconds=None)
    missing = ledger(cpu_seconds=None, cpu_seconds_reason="acp_does_not_report_cpu")
    assert missing["cpu_seconds"] is None
    assert missing["cpu_seconds_reason"] == "acp_does_not_report_cpu"
    with pytest.raises(ValueError, match="INVALID_COST_VALUE"):
        ledger(cpu_seconds=-1)


def test_missing_cost_is_null_plus_reason_never_a_silent_zero():
    unknown = ledger(allocated_core_seconds=None,
                     allocated_core_seconds_reason="scheduler_allocation_not_recorded")
    assert unknown["allocated_core_seconds"] is None
    with pytest.raises(ValueError, match="MISSING_COST_NEEDS_REASON"):
        ledger(allocated_core_seconds=None)


def test_numerical_hessian_internal_gradients_are_counted_once():
    with pytest.raises(ValueError, match="NUMERICAL_HESSIAN_GRADIENTS_DOUBLE_COUNTED"):
        ledger(numerical_hessian_internal_gradients=6,
               numerical_hessian_gradients_counted_in_N_gradient=True)
    honest = ledger(numerical_hessian_internal_gradients=6)
    assert honest["numerical_hessian_internal_gradients"] == 6
    assert honest["numerical_hessian_gradients_counted_in_N_gradient"] is False


def test_aggregate_costs_deduplicates_parent_and_child_ledgers():
    child = ledger(cost_id="child", n_energy=4, n_gradient=4, n_hessian=0,
                   n_optts_trials=0, cpu_seconds=2., allocated_core_seconds=8.,
                   numerical_hessian_internal_gradients=0)
    parent = ledger(cost_id="parent", n_energy=10, n_gradient=8, n_hessian=1,
                    n_optts_trials=2, cpu_seconds=12.5, allocated_core_seconds=48.,
                    includes_cost_ids=["child"])
    totals = aggregate_costs([child, parent])
    assert totals["n_root_ledgers"] == 1
    assert totals["n_ledgers_total"] == 2
    assert totals["N_energy"] == 10
    assert totals["N_gradient"] == 8
    with pytest.raises(ValueError, match="DUPLICATE_COST_ID"):
        aggregate_costs([dict(parent), dict(parent)])
    with pytest.raises(ValueError, match="UNKNOWN_INCLUDED_COST_ID"):
        aggregate_costs([dict(parent, includes_cost_ids=["ghost"])])


def test_aggregate_keeps_missing_costs_explicitly_unknown():
    parent = ledger(cost_id="parent", cpu_seconds=None,
                    cpu_seconds_reason="acp_does_not_report_cpu")
    totals = aggregate_costs([parent])
    assert totals["cpu_seconds"] is None


def test_provenance_requires_all_four_layers_with_truth_assistance_stated():
    layers = provenance_four_layers(geometry="g1_v2_export_v1:abc",
                                    mapping="truth_assisted_p1",
                                    cohort_selection="demo24_train_first_v1",
                                    development_exposure="truth_assisted_development")
    assert set(layers) == {"geometry", "mapping", "cohort_selection", "development_exposure"}
    with pytest.raises(ValueError, match="INVALID_PROVENANCE_LAYER:mapping"):
        provenance_four_layers(geometry="g", mapping=" ", cohort_selection="c",
                               development_exposure="d")
