"""G2 re-review (attempt 2) remediation tests — findings 1/2/3/5.

Covers:
* the selection stage key (full→runs/evidence, short→screening, canary keyed
  with stage; no full/short or fit1/fit5 collision) — finding 1;
* seed aggregation (median metrics + >=2/3 gate-pass rule, per-seed table) —
  finding 1;
* the generic r2 revision writer (--parent/--out/--n/--arm-b-lambda, full
  chain verification, slot fill, out-of-bounds refusal) — finding 2;
* the r0 formal refusal + per-run protocol binding in selection — finding 3;
* the arm-tuple binding (bc_d λ_dir=1.0 formal refusal) — finding 3/5;
* the settings digest including budget/lr/batch/signed flags — finding 5.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    resolve_lambda_registration,
)
from scripts.train_dual_ts_bc import (
    V41_RUN_SCHEMA,
    finalize_selection,
    load_verified_protocol,
    run_settings_digest,
)
from tests.pesn.test_g2_remediation_v41 import (
    FIT50_IDS,
    SPEC_DIGEST,
    _PROTOCOL_R1,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GATE = {
    "median_mse_ratio_max": 1.0,
    "n_groups_ratio_below_one_min": 6,
    "median_cosine_min": 0.3,
    "near_ts_cosine_min": 0.4,
}
SELECTION_RULE = {
    "eligibility_median_cosine_min": 0.0,
    "eligibility_n_cosine_positive_min": 6,
    "simplicity_order": ["bc_nll", "bc_a", "bc_d"],
    "tie_band": 0.01,
}


def _summary(ratio, cos=0.5, near_ts_cos=0.5, n_groups=8):
    return {
        "n_groups": n_groups,
        "mse_ratio_per_group": [ratio] * n_groups,
        "mse_ratio_median": ratio,
        "mse_ratio_min": ratio,
        "mse_ratio_max": ratio,
        "n_groups_ratio_below_one": n_groups if ratio < 1.0 else 0,
        "direction_cosine_per_group": [cos] * n_groups,
        "direction_cosine_median": cos,
        "n_groups_cosine_positive": n_groups if cos > 0 else 0,
        "nll_per_atom_per_group": [1.0] * n_groups,
        "nll_same_density_per_group": [1.0] * n_groups,
        "strata": {
            "near_ts": {"mse_ratio_median": ratio,
                        "direction_cosine_median": near_ts_cos},
            "moving": {"mse_ratio_median": ratio,
                       "direction_cosine_median": near_ts_cos},
            "near_rest": {"mse_ratio_median": ratio,
                          "direction_cosine_median": near_ts_cos},
        },
        "pathology": {"sigma_pinned_at_floor_groups": [],
                      "sigma_dominates_near_rest_groups": []},
    }


def _write_formal_run(tmp_path, *, recipe, seed, stage, ratio, updates_run,
                      protocol_sha256="b" * 64):
    run_dir = tmp_path / f"run_bc_{recipe}_seed{seed}_stage-{stage}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema_version": V41_RUN_SCHEMA,
        "recipe": recipe,
        "settings": {
            "seed": seed, "stage": stage, "diagnostic_run": False,
            "lambda_a": 1.0, "lambda_dir": 1.0,
            "signed_mu_gate": False, "signed_vector_messages": False,
            "protocol_sha256": protocol_sha256,
        },
        "data": {"spec_digest": SPEC_DIGEST},
        "accounting": {"updates_run": updates_run},
    }
    (run_dir / "run_record.json").write_text(json.dumps(rec))
    (run_dir / "monitor_eval.json").write_text(json.dumps({
        "summary": _summary(ratio),
        "c3_trigger": {"status": "not_triggered"},
    }))
    return run_dir


def _thresholds(tmp_path):
    p = tmp_path / "monitor_thresholds_v1.json"
    p.write_text(json.dumps({
        "prereg_sha256": "p" * 64,
        "selection_rule": SELECTION_RULE,
        "gate": GATE,
    }))
    return p


def _protocol(spec_digest=SPEC_DIGEST, sha="b" * 64):
    return {"pipeline": {"spec_digest": spec_digest},
            "freeze": {"sha256": sha}}


# ---------------------------------------------------------------------------
# Finding 1: stage key — full vs short no collision
# ---------------------------------------------------------------------------

def test_selection_full_vs_short_no_collision(tmp_path):
    """full (ratio 0.90) + short (ratio 1.50) same recipe/seed: ``runs`` holds
    the FULL run, ``screening`` holds the short run, and evidence points at the
    full run's monitor_eval (finding 1)."""
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
                      ratio=0.90, updates_run=20000)
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261017, stage="short",
                      ratio=1.50, updates_run=2000)

    selection = finalize_selection(tmp_path, _thresholds(tmp_path), {},
                                   _protocol())
    assert selection["status"] == "SELECTED"
    assert selection["gate_outcome"] == "SELECTED"
    assert selection["relative_best_recipe"] == "bc_nll"
    # full wins the run entry; short is in screening, never in runs
    assert list(selection["runs"].keys()) == \
        ["bc_nll__seed20261017__stage-full"]
    assert selection["runs"]["bc_nll__seed20261017__stage-full"]["updates_run"] \
        == 20000
    assert selection["runs"]["bc_nll__seed20261017__stage-full"]["stage"] \
        == "full"
    assert list(selection["screening"].keys()) == \
        ["bc_nll__seed20261017__stage-short"]
    assert selection["screening"]["bc_nll__seed20261017__stage-short"]["updates_run"] \
        == 2000
    assert selection["screening"]["bc_nll__seed20261017__stage-short"]["stage"] \
        == "short"
    # evidence points at the FULL run's monitor_eval
    ev = selection["evidence"]["monitor_evals"]["bc_nll__seed20261017__stage-full"]
    assert ev.endswith("run_bc_bc_nll_seed20261017_stage-full/monitor_eval.json")
    # per-seed summary is decided from the FULL run (0.90), not the short 1.50
    agg = selection["seed_aggregation"]["bc_nll"]["per_seed"]["20261017"]
    assert agg["summary"]["mse_ratio_median"] == 0.90


def test_selection_canary_stage_key_distinct(tmp_path):
    """fit1 and fit5 canaries are keyed WITH stage — no collision (finding 1)."""
    from tests.pesn.test_g2_remediation_v41 import _write_run_record
    _write_run_record(tmp_path / "run_bc_bc_nll_seed20261017_stage-fit1",
                      schema=V41_RUN_SCHEMA, stage="fit1", diagnostic=False,
                      spec_digest=SPEC_DIGEST, updates_run=30)
    _write_run_record(tmp_path / "run_bc_bc_nll_seed20261017_stage-fit5",
                      schema=V41_RUN_SCHEMA, stage="fit5", diagnostic=False,
                      spec_digest=SPEC_DIGEST, updates_run=50)
    selection = finalize_selection(tmp_path, _thresholds(tmp_path), {},
                                   _protocol())
    assert list(selection["canary_record"].keys()) == [
        "bc_nll__seed20261017__stage-fit1",
        "bc_nll__seed20261017__stage-fit5"]
    assert selection["canary_record"]["bc_nll__seed20261017__stage-fit1"]["stage"] \
        == "fit1"
    assert selection["canary_record"]["bc_nll__seed20261017__stage-fit5"]["stage"] \
        == "fit5"


def test_selection_seed_aggregation_majority_and_median(tmp_path):
    """3 seeds: 2/3 pass → selection_status PASS; median metrics across seeds
    (finding 1 seed aggregation)."""
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
                      ratio=0.80, updates_run=20000)
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261018, stage="full",
                      ratio=0.90, updates_run=20000)
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261019, stage="full",
                      ratio=1.20, updates_run=20000)

    selection = finalize_selection(tmp_path, _thresholds(tmp_path), {},
                                   _protocol())
    agg = selection["seed_aggregation"]["bc_nll"]
    assert agg["n_seeds"] == 3
    assert agg["n_seeds_passing"] == 2
    assert agg["selection_status"] == "PASS"
    assert agg["median_summary"]["mse_ratio_median"] == 0.90
    # full per-seed table present
    assert set(agg["per_seed"].keys()) == {"20261017", "20261018", "20261019"}
    assert agg["per_seed"]["20261017"]["gate_decision"] == "PASS"
    assert agg["per_seed"]["20261019"]["gate_decision"] == "BELOW_GATE"


def test_selection_seed_aggregation_below_gate(tmp_path):
    """1/3 seeds pass → selection_status BELOW_GATE (>=2/3 rule)."""
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
                      ratio=0.80, updates_run=20000)
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261018, stage="full",
                      ratio=1.10, updates_run=20000)
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261019, stage="full",
                      ratio=1.20, updates_run=20000)
    selection = finalize_selection(tmp_path, _thresholds(tmp_path), {},
                                   _protocol())
    agg = selection["seed_aggregation"]["bc_nll"]
    assert agg["n_seeds_passing"] == 1
    assert agg["selection_status"] == "BELOW_GATE"


# ---------------------------------------------------------------------------
# Finding 3: r0 formal refusal + per-run protocol binding
# ---------------------------------------------------------------------------

def test_r0_formal_refusal(tmp_path):
    from scripts.train_dual_ts_bc import compute_protocol_content_digest
    r0 = {"pipeline": {"spec_digest": SPEC_DIGEST},
          "registered_budgets": {"full": {"updates": 20000}}}
    r0_sha = compute_protocol_content_digest(r0)
    r0["freeze"] = {"sha256": r0_sha}
    r0_path = tmp_path / "protocol_v41.json"
    r0_path.write_text(json.dumps(r0, sort_keys=True, indent=2) + "\n")

    # formal refuses r0 (no amendment contract)
    with pytest.raises(SystemExit) as err:
        load_verified_protocol(r0_path, require_formal=True)
    assert "r0" in str(err.value) or "canaries/diagnostics" in str(err.value)

    # non-formal (canary/diagnostic) still loads r0
    payload, digest = load_verified_protocol(r0_path)
    assert digest == r0_sha


def test_r0_run_selection_exclusion(tmp_path):
    """A v2 run recorded under r0 (protocol_sha256 = r0's sha) is EXCLUDED when
    the active protocol is r1 (finding 3)."""
    _write_formal_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
                      ratio=0.90, updates_run=20000,
                      protocol_sha256="0" * 64)  # r0 sha ≠ active "b"*64
    selection = finalize_selection(tmp_path, _thresholds(tmp_path), {},
                                   _protocol(sha="b" * 64))
    assert selection["runs"] == {}
    assert selection["screening"] == {}


# ---------------------------------------------------------------------------
# Finding 3/5: arm-tuple binding
# ---------------------------------------------------------------------------

def test_bc_d_lambda_dir_1_0_formal_refusal():
    """bc_d with λ_dir=1.0 is NOT a registered arm tuple (arm_a is bc_nll) →
    formal refusal (finding 3/5)."""
    d = resolve_lambda_registration(
        _PROTOCOL_R1, recipe="bc_d", lambda_a=1.0, lambda_dir=1.0,
        stage="full", budget_updates=20000, allow_diagnostic_budget=False,
        seed=20261017, features_spec_digest=SPEC_DIGEST,
        population_group_ids=FIT50_IDS, batch_states=64, lr=1e-3,
        signed_vector_messages=False)
    assert not d["accepted"]
    assert any("arm_mapping" in p for p in d["problems"])


def test_arm_tuple_binding_signed_variant():
    """C (bc_nll/1.0/1.0/signed=True) and D (bc_d/1.0/<arm_b>/signed=True) are
    distinct arm tuples; the signed flag is part of the tuple (finding 3/5)."""
    protocol = {
        **_PROTOCOL_R1,
        "arm_mapping": {
            "A": {"recipe": "bc_nll", "lambda_a": 1.0, "lambda_dir": 1.0,
                  "signed_vector_messages": False},
            "B": {"recipe": "bc_d", "lambda_a": 1.0, "lambda_dir": None,
                  "signed_vector_messages": False},
            "C": {"recipe": "bc_nll", "lambda_a": 1.0, "lambda_dir": 1.0,
                  "signed_vector_messages": True},
            "D": {"recipe": "bc_d", "lambda_a": 1.0, "lambda_dir": None,
                  "signed_vector_messages": True},
        },
    }
    base = dict(stage="full", budget_updates=20000,
                allow_diagnostic_budget=False, seed=20261017,
                features_spec_digest=SPEC_DIGEST,
                population_group_ids=FIT50_IDS, batch_states=64, lr=1e-3)
    # C tuple: bc_nll/1.0/1.0/signed=True → accepted
    d = resolve_lambda_registration(protocol, recipe="bc_nll", lambda_a=1.0,
                                    lambda_dir=1.0,
                                    signed_vector_messages=True, **base)
    assert d["accepted"]
    # A tuple: bc_nll/1.0/1.0/signed=False → accepted (distinct tuple)
    d = resolve_lambda_registration(protocol, recipe="bc_nll", lambda_a=1.0,
                                    lambda_dir=1.0,
                                    signed_vector_messages=False, **base)
    assert d["accepted"]
    # D tuple with λ_dir=1.0 (arm_b not yet frozen → None) → refused
    d = resolve_lambda_registration(protocol, recipe="bc_d", lambda_a=1.0,
                                    lambda_dir=1.0,
                                    signed_vector_messages=True, **base)
    assert not d["accepted"]


# ---------------------------------------------------------------------------
# Finding 5: settings digest includes budget/lr/batch/signed
# ---------------------------------------------------------------------------

def test_run_settings_digest_includes_budget_lr_batch_signed():
    base = dict(recipe="bc_nll", seed=20261017, stage="short",
                lambda_a=1.0, lambda_dir=1.0, features_spec_digest=SPEC_DIGEST,
                protocol_sha256="a" * 64)
    assert run_settings_digest(**base) != run_settings_digest(
        **{**base, "budget_updates": 2000})
    assert run_settings_digest(**base) != run_settings_digest(
        **{**base, "lr": 1e-2})
    assert run_settings_digest(**base) != run_settings_digest(
        **{**base, "batch_states": 32})
    assert run_settings_digest(**base) != run_settings_digest(
        **{**base, "signed_vector_messages": True})


# ---------------------------------------------------------------------------
# Finding 2: generic r2 revision writer
# ---------------------------------------------------------------------------

def test_revise_r2_chain_and_arm_b_fill(tmp_path):
    import scripts.freeze_v41_protocol as freeze

    r0_src = PROJECT_ROOT / (
        "outputs/pesn/dual_ts_v2_20261015/wp4_bc/protocol_v41.json")
    r1_src = PROJECT_ROOT / (
        "outputs/pesn/dual_ts_v2_20261015/wp4_bc/protocol_v41_r1.json")
    if not r0_src.is_file() or not r1_src.is_file():
        pytest.skip("real v4.1 protocol chain not present (local outputs)")
    shutil.copy(r0_src, tmp_path / "protocol_v41.json")
    shutil.copy(r1_src, tmp_path / "protocol_v41_r1.json")

    r2 = freeze.revise(parent=tmp_path / "protocol_v41_r1.json",
                       out=tmp_path / "protocol_v41_r2.json",
                       n=2, arm_b_lambda=1.5)
    assert r2["revision"]["n"] == 2
    assert r2["revision"]["parent_file"] == "protocol_v41_r1.json"
    assert r2["arm_mapping"]["B"]["lambda_dir"] == 1.5
    assert r2["arm_mapping"]["D"]["lambda_dir"] == 1.5
    assert r2["lambda_registration"]["slots"]["arm_b"] == {
        "lambda_a": 1.0, "lambda_dir": 1.5, "status": "frozen"}
    assert "seed_aggregation" in r2

    # full chain r2 → r1 → r0 verifies transitively (finding 2)
    payload, digest = load_verified_protocol(tmp_path / "protocol_v41_r2.json")
    assert payload["revision"]["n"] == 2
    assert digest == r2["freeze"]["sha256"]

    # out-of-bounds λ refused
    with pytest.raises(SystemExit) as err:
        freeze.revise(parent=tmp_path / "protocol_v41_r1.json",
                      out=tmp_path / "protocol_v41_r2_oob.json",
                      n=2, arm_b_lambda=5e-4)
    assert "bounds" in str(err.value) or "1e-3" in str(err.value)
    # nothing was written for the refused out
    assert not (tmp_path / "protocol_v41_r2_oob.json").exists()

    # out exists with DIFFERENT content → refusal
    bad = dict(r2)
    bad["arm_mapping"]["B"]["lambda_dir"] = 9.9
    bad["freeze"] = {"sha256": freeze._content_digest(bad)}
    (tmp_path / "protocol_v41_r2_other.json").write_text(
        freeze._serialize(bad))
    with pytest.raises(SystemExit):
        freeze.revise(parent=tmp_path / "protocol_v41_r1.json",
                      out=tmp_path / "protocol_v41_r2_other.json",
                      n=2, arm_b_lambda=1.5)
