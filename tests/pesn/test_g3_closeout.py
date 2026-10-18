"""G3 close-out regression tests — selection-record repair (findings 3/4/5).

Covers the ``finalize_selection`` fixes required before the decision package
is frozen:

* (a) a synthetic all-BELOW_GATE matrix emits top-level ``REGISTER_ABLATION``
  (never an unqualified ``SELECTED``) and separates the RELATIVE argmin
  recipe from the registered gate outcome — finding 3;
* (b) the signed arm (``_signedvecmsg``) appears in ``runs``, ``per_recipe``,
  ``seed_aggregation`` and ``evidence.monitor_evals`` under an arm-unique key
  (no collision with the unsigned recipe) — finding 4;
* (c) ``evidence.fit_features_manifest`` never points at a diagnostic
  (``_a45``/``_diag``) build — finding 5.

Pure synthetic fixtures; no model loading, no GPU, no QC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.train_dual_ts_bc import (
    V41_RUN_SCHEMA,
    finalize_selection,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diagnose_v41_alpha_formal import classify_arm  # noqa: E402

SPEC_DIGEST = "f678f8e5c83cf913980139b48d4836475e11de5d1e1a52035cf6abddf9c1a99c"

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


def _thresholds(tmp_path):
    p = tmp_path / "monitor_thresholds_v1.json"
    p.write_text(json.dumps({
        "prereg_sha256": "p" * 64,
        "selection_rule": SELECTION_RULE,
        "gate": GATE,
    }))
    return p


def _protocol(spec_digest=SPEC_DIGEST, sha="b" * 64, fit_dir=None):
    pipeline = {"spec_digest": spec_digest}
    if fit_dir is not None:
        pipeline["feature_dirs"] = {"fit": fit_dir}
    return {"pipeline": pipeline, "freeze": {"sha256": sha}}


def _write_run(tmp_path, *, recipe, seed, stage, ratio, updates_run,
               cos=0.5, near_ts_cos=0.5,
               signed_mu_gate=False, signed_vector_messages=False,
               protocol_sha256="b" * 64, features_dir=None):
    variant = ""
    if signed_mu_gate:
        variant += "_signedmu"
    if signed_vector_messages:
        variant += "_signedvecmsg"
    run_dir = tmp_path / f"run_bc_{recipe}_seed{seed}{variant}_stage-{stage}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {"spec_digest": SPEC_DIGEST}
    if features_dir is not None:
        data["features_dir"] = features_dir
    rec = {
        "schema_version": V41_RUN_SCHEMA,
        "recipe": recipe,
        "settings": {
            "seed": seed, "stage": stage, "diagnostic_run": False,
            "lambda_a": 1.0, "lambda_dir": 1.0,
            "signed_mu_gate": signed_mu_gate,
            "signed_vector_messages": signed_vector_messages,
            "protocol_sha256": protocol_sha256,
        },
        "data": data,
        "accounting": {"updates_run": updates_run},
    }
    (run_dir / "run_record.json").write_text(json.dumps(rec))
    (run_dir / "monitor_eval.json").write_text(json.dumps({
        "summary": _summary(ratio, cos=cos, near_ts_cos=near_ts_cos),
        "c3_trigger": {"status": "triggered"},
    }))
    return run_dir


# ---------------------------------------------------------------------------
# (a) all-BELOW_GATE → REGISTER_ABLATION, no unqualified SELECTED
# ---------------------------------------------------------------------------

def test_below_gate_matrix_registers_ablation(tmp_path):
    # A (bc_nll) and B (bc_d) both BELOW_GATE: ratio > 1.0, cos/near-TS low.
    _write_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
               ratio=1.05, cos=0.1, near_ts_cos=0.1, updates_run=20000)
    _write_run(tmp_path, recipe="bc_d", seed=20261017, stage="full",
               ratio=1.10, cos=0.1, near_ts_cos=0.1, updates_run=20000)

    selection = finalize_selection(tmp_path, _thresholds(tmp_path), {},
                                   _protocol())
    # the registered non-PASS outcome, not an unqualified SELECTED
    assert selection["gate_outcome"] == "REGISTER_ABLATION"
    assert selection["status"] == "REGISTER_ABLATION"
    assert "selected_recipe" not in selection
    # the RELATIVE argmin is kept, explicitly labeled relative-not-gate
    assert selection["relative_best_recipe"] == "bc_nll"
    assert "RELATIVE" in selection["relative_best_recipe_note"].upper()
    assert selection["per_recipe"]["bc_nll"]["gate_decision"] == "BELOW_GATE"
    assert selection["per_recipe"]["bc_d"]["gate_decision"] == "BELOW_GATE"


def test_pass_matrix_gate_outcome_selected(tmp_path):
    _write_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
               ratio=0.90, updates_run=20000)
    selection = finalize_selection(tmp_path, _thresholds(tmp_path), {},
                                   _protocol())
    assert selection["gate_outcome"] == "SELECTED"
    assert selection["status"] == "SELECTED"


# ---------------------------------------------------------------------------
# (b) signed arm appears in every section under an arm-unique key
# ---------------------------------------------------------------------------

def test_signed_arm_present_in_all_sections(tmp_path):
    _write_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
               ratio=0.90, updates_run=20000)                       # A
    _write_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
               ratio=0.95, signed_vector_messages=True,
               updates_run=20000)                                   # C

    selection = finalize_selection(tmp_path, _thresholds(tmp_path), {},
                                   _protocol())

    a_key = "bc_nll__seed20261017__stage-full"
    c_key = "bc_nll_signedvecmsg__seed20261017__stage-full"
    # runs: both A and C present (no collision)
    assert a_key in selection["runs"]
    assert c_key in selection["runs"]
    assert selection["runs"][c_key]["signed_vector_messages"] is True
    assert selection["runs"][c_key]["arm"] == "bc_nll_signedvecmsg"
    # per_recipe / seed_aggregation: signed arm keyed arm-uniquely
    assert "bc_nll_signedvecmsg" in selection["per_recipe"]
    assert "bc_nll_signedvecmsg" in selection["seed_aggregation"]
    assert selection["seed_aggregation"]["bc_nll_signedvecmsg"]["n_seeds"] == 1
    # evidence.monitor_evals includes the signed arm
    assert c_key in selection["evidence"]["monitor_evals"]
    assert selection["evidence"]["monitor_evals"][c_key].endswith(
        "run_bc_bc_nll_seed20261017_signedvecmsg_stage-full/monitor_eval.json")


# ---------------------------------------------------------------------------
# (c) evidence manifest never points at a diagnostic build
# ---------------------------------------------------------------------------

def test_evidence_manifest_uses_protocol_fit_dir(tmp_path):
    _write_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
               ratio=0.90, updates_run=20000)
    protocol = _protocol(
        fit_dir="outputs/pesn/dual_ts_v2_20261015/wp3_features_v41")
    selection = finalize_selection(
        tmp_path, _thresholds(tmp_path), {}, protocol,
        fit_features_dir=tmp_path / "wp3_features_v41_a45")  # ignored
    manifest = selection["evidence"]["fit_features_manifest"]
    assert manifest.endswith("wp3_features_v41/features_manifest.json")
    assert "a45" not in manifest


def test_evidence_manifest_falls_back_to_formal_run_dir(tmp_path):
    _write_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
               ratio=0.90, updates_run=20000,
               features_dir="outputs/pesn/dual_ts_v2_20261015/wp3_features_v41")
    selection = finalize_selection(
        tmp_path, _thresholds(tmp_path), {}, _protocol(),
        fit_features_dir=tmp_path / "wp3_features_v41_a45")
    manifest = selection["evidence"]["fit_features_manifest"]
    assert "a45" not in manifest and "_diag" not in manifest
    assert manifest.endswith("wp3_features_v41/features_manifest.json")


def test_evidence_manifest_diagnostic_arg_refused(tmp_path):
    _write_run(tmp_path, recipe="bc_nll", seed=20261017, stage="full",
               ratio=0.90, updates_run=20000)  # no features_dir recorded
    with pytest.raises(SystemExit) as err:
        finalize_selection(tmp_path, _thresholds(tmp_path), {}, _protocol(),
                           fit_features_dir=tmp_path / "wp3_features_v41_a45")
    assert "diagnostic" in str(err.value) or "finding 5" in str(err.value)


# ---------------------------------------------------------------------------
# Light unit test for the α-formal diagnostic (finding 9)
# ---------------------------------------------------------------------------

def test_alpha_formal_arm_classification():
    assert classify_arm({"recipe": "bc_nll", "lambda_dir": 1.0,
                         "signed_vector_messages": False}) == "A"
    assert classify_arm({"recipe": "bc_d", "lambda_dir": 559.37,
                         "signed_vector_messages": False}) == "B"
    assert classify_arm({"recipe": "bc_nll", "lambda_dir": 1.0,
                         "signed_vector_messages": True}) == "C"
    # the D tuple (bc_d + signed) is not one of the measured A/B/C arms
    assert classify_arm({"recipe": "bc_d", "lambda_dir": 559.37,
                         "signed_vector_messages": True}) is None
    # bc_a and off-registration tuples classify as None
    assert classify_arm({"recipe": "bc_a", "lambda_dir": 1.0,
                         "signed_vector_messages": False}) is None
    assert classify_arm({"recipe": "bc_d", "lambda_dir": 1.0,
                         "signed_vector_messages": False}) is None
