"""V5.4.1 campaign schema validator tests (WP0, ADR-0032).

Dry-run scope: PyYAML + stdlib only; no torch, no data, no GPU, no network.
Covers every typed rejection required by plan §3.3:

* missing field / wrong schema id / non-unified objective / legacy arm matrix,
* unknown embedding (config registry + run matrix + E6 scheduling),
* illegal forward key (outside FORWARD_VIEW_KEYS_V3 whitelist),
* EdgeAdapter / node-vector core consumption,
* confirm/sealed/QC authorization not false, single-lane/swap guards,
* null §11.2 thresholds blocking ``--formal``,
* G–V registry cross-check mismatch,
* run-matrix shape (V541_ namespace, classes, seeds, diagnostic winner rule),
* run-outcome vocabulary vs. legacy ledger status machine separation.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import v541_config_schema as v  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "agent_ops" / "v541_config_schema.py"
CONFIG = REPO_ROOT / "config" / "campaigns" / "pes2ts_v541_embed_50k.yaml"


@pytest.fixture(scope="module")
def base_cfg() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["schema"] == v.SCHEMA_ID
    return cfg


def _mut(base_cfg: dict, fn) -> dict:
    cfg = copy.deepcopy(base_cfg)
    fn(cfg)
    return cfg


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------
def test_frozen_config_validates_prereg_and_formal(base_cfg):
    prereg = v.validate_config(base_cfg, formal=False)
    assert prereg["ok"] and prereg["mode"] == "prereg"
    formal = v.validate_config(base_cfg, formal=True)
    assert formal["ok"] and formal["mode"] == "formal"
    assert formal["null_thresholds"] == []
    # thresholds are signed frozen 2026-10-27 AND the WP5 W0/W1 profiling
    # budget has since been signed into the frozen config, so nothing blocks
    # T1 at the CONFIG level any more (the T1 GATES themselves stay
    # runner-side evidence).  The null-budget blocking path is exercised
    # explicitly by test_null_budget_blocks_formal_t1 below — never by
    # assuming this file is still unsigned.
    assert formal["t1_blocked"] == []
    assert formal["t1_config_ready"] is True
    # current signed state: a non-empty per-candidate budget table
    budget = base_cfg["budget"]
    assert isinstance(budget["per_candidate_core_time"], dict)
    assert budget["per_candidate_core_time"], "budget table must not be empty"
    assert str(budget["per_candidate_status"]).startswith("signed_")
    for entry in budget["per_candidate_core_time"].values():
        assert isinstance(entry.get("approved_for_formal"), bool)


def test_null_budget_blocks_formal_t1(base_cfg):
    """A null WP5 budget still blocks T1 — on an EXPLICIT temporary copy.

    The frozen repo config is signed now; this test therefore rebuilds the
    pre-signoff state on a copy instead of depending on the real file.
    """
    cfg = _mut(
        base_cfg,
        lambda c: c["budget"].update(
            per_candidate_core_time=None,
            per_candidate_status="pending_wp5_profiling",
        ),
    )
    assert cfg is not base_cfg
    report = v.validate_config(cfg, formal=False)
    assert report["t1_config_ready"] is False
    assert any("per_candidate_core_time" in r for r in report["t1_blocked"])
    # a null budget is a T1 blocker, not a schema rejection
    formal = v.validate_config(cfg, formal=True)
    assert formal["ok"] and formal["mode"] == "formal"
    assert formal["t1_config_ready"] is False
    assert any("per_candidate_core_time" in r for r in formal["t1_blocked"])


def test_cli_exit_zero_on_repo_config():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--formal", "--config", str(CONFIG)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_json_report_is_parseable():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--config", str(CONFIG)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] and report["schema"] == v.SCHEMA_ID


# ---------------------------------------------------------------------------
# schema / required fields / objective
# ---------------------------------------------------------------------------
def test_wrong_schema_id_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.update(schema="pes2ts_v53mp_campaign_v1"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_SCHEMA"


def test_missing_required_field_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.pop("training_contract"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_MISSING_FIELD"
    assert "training_contract" in exc.value.detail


def test_none_required_field_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.update(evaluation_contract=None))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_MISSING_FIELD"


def test_non_unified_objective_rejected(base_cfg):
    for bad in ("unified_move_process_v2", "move_only", None, ""):
        cfg = _mut(base_cfg, lambda c, b=bad: c.update(objective=b))
        with pytest.raises(v.V541SchemaError) as exc:
            v.validate_config(cfg)
        assert exc.value.code == "V541_OBJECTIVE", bad


def test_legacy_arm_matrix_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.update(arms={"A": 1, "B": 2, "C": 3}))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_ARM_MATRIX"


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------
def test_unknown_embedding_in_registry_rejected(base_cfg):
    def _bad(c):
        c["embedding_contract"]["embedding_registry"]["E7"] = {
            "family": "made_up",
            "internal_view": "x",
            "output": "residual_node_delta",
            "in_main_matrix": True,
        }

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_EMBEDDING_UNKNOWN"
    assert "E7" in exc.value.detail


def test_embedding_family_drift_rejected(base_cfg):
    def _bad(c):
        c["embedding_contract"]["embedding_registry"]["E1"]["family"] = "schnet_spatial"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_EMBEDDING_UNKNOWN"


def test_e6_scheduling_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][0]["embedding"] = "E6"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_EMBEDDING_UNKNOWN"


def test_run_matrix_unknown_embedding_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][1]["embedding"] = "E9"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_EMBEDDING_UNKNOWN"


# ---------------------------------------------------------------------------
# forward view / core consumption
# ---------------------------------------------------------------------------
def test_illegal_forward_key_rejected(base_cfg):
    def _bad(c):
        c["embedding_contract"]["forward_view_keys"].append("y_teacher")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_FORWARD_KEY"
    assert "y_teacher" in exc.value.detail


def test_forbidden_input_class_as_forward_key_rejected(base_cfg):
    def _bad(c):
        c["embedding_contract"]["forward_view_keys"].append("ts_coords")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_FORWARD_KEY"


def test_empty_forward_keys_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["embedding_contract"].update(forward_view_keys=[]))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_MISSING_FIELD"


def test_edge_adapter_consumption_rejected(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["embedding_contract"]["core_consumption"].update(edge_adapter=True),
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_EDGE_ADAPTER"


def test_node_vector_consumption_rejected(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["embedding_contract"]["core_consumption"].update(
            node_vector_consumed=True
        ),
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_NODE_VECTOR"


def test_edge_scalar_channel_count_must_stay_legacy_22(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["embedding_contract"]["core_consumption"].update(
            edge_scalar_channels=48
        ),
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_EDGE_ADAPTER"


# ---------------------------------------------------------------------------
# authorization / resources
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["confirm_authorized", "sealed_authorized"])
def test_confirm_or_sealed_authorization_rejected(base_cfg, flag):
    cfg = _mut(base_cfg, lambda c: c["governance"]["authorization"].update(**{flag: True}))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_CONFIRM"


def test_nonzero_qc_calls_rejected(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["governance"]["authorization"].update(new_qc_calls=3),
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_CONFIRM"


def test_single_lane_required(base_cfg):
    cfg = _mut(
        base_cfg, lambda c: c["governance"]["authorization"].update(single_lane=False)
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RESOURCE"


def test_swap_forbidden(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["budget"].update(swap_max=4096))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RESOURCE"


# ---------------------------------------------------------------------------
# thresholds / formal gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", list(v.FORMAL_THRESHOLDS))
def test_null_threshold_blocks_formal(base_cfg, key):
    cfg = _mut(base_cfg, lambda c, k=key: c["thresholds"].update({k: None}))
    # prereg mode: allowed, reported as blocker
    report = v.validate_config(cfg, formal=False)
    assert any(key in r for r in report["t1_blocked"])
    # formal mode: typed rejection
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V541_THRESHOLD_NULL"
    assert key in exc.value.detail


def test_non_numeric_threshold_rejected(base_cfg):
    cfg = _mut(
        base_cfg, lambda c: c["thresholds"].update(max_core_time_ratio="fast")
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_THRESHOLD_NULL"


def test_unfrozen_signoff_blocks_formal(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["thresholds"]["signoff"].update(status="proposed"),
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V541_THRESHOLD_NULL"


def test_signoff_mirror_mismatch_blocks_formal(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["thresholds"]["signoff"].update(max_core_time_ratio=3.0),
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V541_THRESHOLD_NULL"
    assert "max_core_time_ratio" in exc.value.detail


def test_formal_thresholds_match_signoff_block(base_cfg):
    cfg = base_cfg
    signoff = cfg["thresholds"]["signoff"]
    for key in v.FORMAL_THRESHOLDS:
        assert cfg["thresholds"][key] == signoff[key], key
    assert signoff["status"] == "frozen"
    assert signoff["signed_on"] == "2026-10-27"


# ---------------------------------------------------------------------------
# G–V registry cross-check
# ---------------------------------------------------------------------------
def test_gv_campaign_mismatch_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["gv"].update(campaign_id="V5.4"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_GV_MISMATCH"


def test_gv_supporting_g_drift_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["gv"].update(supporting_g=["G3.P.1", "G0"]))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_GV_MISMATCH"


def test_gv_gate_drift_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["gv"]["gates"].append("V541-rogue-gate"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_GV_MISMATCH"


def test_gv_check_can_be_skipped_intentionally(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["gv"].update(map_version="stale"))
    assert v.validate_config(cfg, check_gv=False)["ok"]


# ---------------------------------------------------------------------------
# run matrix / run outcomes
# ---------------------------------------------------------------------------
def test_run_id_must_use_v541_namespace(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][0]["run_id"] = "E1_fold0_s17"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_run_seed_must_be_preregistered(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][0]["seed"] = 7

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_diagnostic_run_cannot_be_winner_eligible(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["class"] == "diagnostic":
                run["winner_eligible"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_run_classes_must_be_exact_three_way_split(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["run_matrix"].update(classes=["nominal"]))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_duplicate_run_id_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][1]["run_id"] = c["run_matrix"]["runs"][0]["run_id"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_run_outcome_vocabulary_frozen(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["run_outcomes"].update(vocabulary=["done", "failed"]),
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_legacy_ledger_status_machine_must_stay_unchanged(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["run_outcomes"].update(ledger_status_machine="extended"),
    )
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_outcome_vocabulary_is_disjoint_from_legacy_ledger_status():
    from scripts.agent_ops.ledger.run_ledger import KNOWN_STATUSES

    ro_vocab = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["run_outcomes"]
    assert set(ro_vocab["vocabulary"]) == {
        "done",
        "invalid_run",
        "scientific_failure",
        "resource_failure",
        "budget_censored",
        "gate_skipped",
    }
    # only "done" is intentionally shared as a label; the state machines are
    # separate — legacy statuses must not appear in the V541 vocabulary otherwise
    legacy_only = set(KNOWN_STATUSES) - {"done"}
    assert not legacy_only & set(ro_vocab["vocabulary"])


# ---------------------------------------------------------------------------
# file-level rejections
# ---------------------------------------------------------------------------
def test_missing_config_file_typed():
    with pytest.raises(v.V541SchemaError) as exc:
        v.load_config(REPO_ROOT / "config" / "campaigns" / "does_not_exist.yaml")
    assert exc.value.code == "V541_CONFIG_MISSING"


def test_non_mapping_yaml_typed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(v.V541SchemaError) as exc:
        v.load_config(bad)
    assert exc.value.code == "V541_CONFIG_MISSING"


def test_cli_exit_nonzero_on_bad_config(tmp_path, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["objective"] = "not_unified"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(bad), "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["code"] == "V541_OBJECTIVE"


# ---------------------------------------------------------------------------
# T3 preregistration (plan §9.1/§9.5) — all selectable candidates listed,
# only inner-val top-2 activates, N_core formula-counted
# ---------------------------------------------------------------------------
def _t3_ids() -> set:
    return {
        f"V541_{eid}_S{seed}" for eid in ("E1", "E2", "E3", "E4", "E5")
        for seed in (29, 43)
    }


def test_t3_preregisters_full_cross_product(base_cfg):
    cfg = base_cfg
    t3 = [r for r in cfg["run_matrix"]["runs"] if r["stage"] == "T3"]
    assert {r["run_id"] for r in t3} == _t3_ids()
    assert len(t3) == 10
    for run in t3:
        assert run["class"] == "conditional"
        assert run["winner_eligible"] is True
        assert run["condition"] == "activated_only_by_inner_val_top2"
        assert run["unactivated_status"] == "gate_skipped"
        assert run["new_training_if_activated"] == 1
        assert "new_training" not in run
        assert run["embedding"] is not None
        if run["embedding"] == "E5":
            assert run["parent_gate"] == (
                "parents_E1_E4_H0_H1_H2_pass_and_budget_approved"
            )
        else:
            assert "parent_gate" not in run


def test_t3_block_declares_full_activation_four(base_cfg):
    block = base_cfg["run_matrix"]["t3_preregistration"]
    assert block["candidates"] == ["E1", "E2", "E3", "E4", "E5"]
    assert block["seeds"] == [29, 43]
    assert block["listed_runs"] == 10
    assert block["full_activation_count"] == 4
    assert block["activation"] == "activated_only_by_inner_val_top2"
    assert block["unactivated_status"] == "gate_skipped"
    assert block["n_core_counting"] == "formula_only"
    # listed runs must never be read as the training count
    assert block["listed_runs"] != block["full_activation_count"]


def test_t3_run_count_does_not_change_n_core_formula(base_cfg):
    """N_core follows §9.1; the 10 listed T3 rows are preregistration only."""
    pin = json.loads(
        (
            REPO_ROOT / "config" / "campaigns" / "v541_templates"
            / "runplan_pin.template.json"
        ).read_text(encoding="utf-8")
    )
    formula = pin["core_training_count_formula"]
    assert "N_core = n_T1 + I(E5) + 2*n_selected + 3*I(E0_retrain)" in formula
    assert "NEVER by summing listed run rows" in formula
    t3_pin = pin["run_classes"]["conditional"]["t3_preregistration"]
    assert t3_pin["listed_runs"] == 10
    assert t3_pin["full_activation_count"] == 4
    assert t3_pin["unactivated_new_trainings"] == 0
    # formula budget: full nominal 8-9 core trainings, not 4+10+...
    assert "8-9" in formula


def test_t3_placeholder_token_rejected(base_cfg):
    def _bad(c):
        runs = c["run_matrix"]["runs"]
        for i, run in enumerate(runs):
            if run["stage"] == "T3":
                runs[i] = {
                    **run,
                    "run_id": f"V541_T3_TOP2_S{run['seed']}",
                    "embedding": None,
                }
                break

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"
    assert "TOP2" in exc.value.detail


@pytest.mark.parametrize("drop", ["V541_E1_S29", "V541_E5_S43", "V541_E3_S29"])
def test_t3_missing_candidate_rejected(base_cfg, drop):
    def _bad(c):
        c["run_matrix"]["runs"] = [
            r for r in c["run_matrix"]["runs"] if r["run_id"] != drop
        ]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"
    assert drop in exc.value.detail


def test_t3_extra_candidate_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"].append(
            {
                "run_id": "V541_E4_S17",
                "stage": "T3",
                "embedding": "E4",
                "seed": 17,
                "class": "conditional",
                "new_training_if_activated": 1,
                "unactivated_status": "gate_skipped",
                "winner_eligible": True,
                "condition": "activated_only_by_inner_val_top2",
            }
        )

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_null_embedding_identity_rejected_anywhere(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["stage"] == "T1":
                run["embedding"] = None
                break

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_EMBEDDING_UNKNOWN"
    assert "deterministic embedding identity" in exc.value.detail


def test_t3_condition_must_gate_on_inner_val_top2(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["stage"] == "T3" and run["embedding"] == "E2":
                run["condition"] = "always_launch"
                break

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"
    assert "activated_only_by_inner_val_top2" in exc.value.detail


def test_t3_unactivated_must_be_gate_skipped(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["stage"] == "T3" and run["embedding"] == "E3":
                run["unactivated_status"] = "failed"
                break

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"
    assert "gate_skipped" in exc.value.detail


def test_t3_must_not_claim_unconditional_new_training(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["stage"] == "T3" and run["embedding"] == "E1":
                run["new_training"] = 1
                break

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"
    assert "new_training" in exc.value.detail


def test_t3_e5_stacks_parent_gate(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["stage"] == "T3" and run["embedding"] == "E5":
                run.pop("parent_gate")
                break

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"
    assert "parent_gate" in exc.value.detail


def test_t3_parent_gate_not_allowed_on_non_e5(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["stage"] == "T3" and run["embedding"] == "E2":
                run["parent_gate"] = "parents_E1_E4_H0_H1_H2_pass_and_budget_approved"
                break

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"
    assert "only valid on E5" in exc.value.detail


@pytest.mark.parametrize(
    "field,value",
    [
        ("listed_runs", 2),
        ("full_activation_count", 10),
        ("activation", "launch_all"),
        ("unactivated_status", "failed"),
        ("n_core_counting", "sum_listed_runs"),
    ],
)
def test_t3_block_invariants_enforced(base_cfg, field, value):
    def _bad(c, f=field, val=value):
        c["run_matrix"]["t3_preregistration"][f] = val

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"


def test_t3_block_missing_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["run_matrix"].pop("t3_preregistration"))
    with pytest.raises(v.V541SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V541_RUN_MATRIX"
    assert "t3_preregistration" in exc.value.detail


# ---------------------------------------------------------------------------
# shared sampler declaration (frozen config must keep plan §7.2 strata)
# ---------------------------------------------------------------------------
def test_config_keeps_declared_frame_strata():
    """WP0 review: the V541 config declares the plan §7.2 strata; the old
    V53MP _curve_batch_positions samples only (curve, direction) pools, so
    equivalence must be proven by the V541 runner / WP5 audit — never assumed.
    The config field itself must stay frozen (ADR-0032 §4)."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    batch = cfg["training_contract"]["batch"]
    assert batch["strata"] == ["start", "pre_TS", "post_TS", "endpoint"]
    assert batch["curves"] == 8 and batch["frames"] == 4
    # ADR must record the verified discrepancy as a WP5 adjudication item
    adr = (
        REPO_ROOT / "docs" / "design" / "decisions"
        / "ADR-0032-V5.4.1-Embedding接口扩展与预注册治理.md"
    ).read_text(encoding="utf-8")
    assert "_curve_batch_positions" in adr
    assert "WP5" in adr
    assert "重训 E0×3" in adr


# ---------------------------------------------------------------------------
# WP0 templates
# ---------------------------------------------------------------------------
def test_templates_parse_and_carry_signed_thresholds():
    tpl_dir = REPO_ROOT / "config" / "campaigns" / "v541_templates"
    signoff = json.loads(
        (tpl_dir / "preregistration_signoff.template.json").read_text(encoding="utf-8")
    )
    budget = json.loads(
        (tpl_dir / "budget_policy.template.json").read_text(encoding="utf-8")
    )
    pin = json.loads(
        (tpl_dir / "runplan_pin.template.json").read_text(encoding="utf-8")
    )
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    signed = signoff["thresholds_signed_frozen"]
    assert signed["status"] == "frozen" and signed["signed_on"] == "2026-10-27"
    for key in v.FORMAL_THRESHOLDS:
        assert signed[key] == cfg["thresholds"][key], key
    auth = signoff["authorization"]
    assert auth["implementation_authorized"] is True
    assert auth["formal_training_authorized"] is True
    assert auth["confirm_authorized"] is False
    assert auth["sealed_authorized"] is False
    assert auth["new_qc_calls"] == 0
    assert auth["single_lane"] is True and auth["swap_allowed"] is False

    assert budget["execution_policy"]["single_lane"] is True
    assert budget["execution_policy"]["swap_max"] == 0
    assert budget["schedule"]["total_updates"] == 37900
    assert budget["cost_acceptance"]["max_core_time_ratio"] == signed[
        "max_core_time_ratio"
    ]

    classes = pin["run_classes"]
    assert set(classes) == {"nominal", "conditional", "diagnostic"}
    assert classes["diagnostic"]["counts_toward_winner"] is False
    assert classes["nominal"]["counts_toward_winner"] is True
    # pin run ids must match the config run matrix exactly
    cfg_ids = {r["run_id"] for r in cfg["run_matrix"]["runs"]}
    pin_ids = {rid for cls in classes.values() for rid in cls["run_ids"]}
    assert cfg_ids == pin_ids, (cfg_ids ^ pin_ids)
    # signoff template must mirror the config T3 preregistration block
    t3_cfg = cfg["run_matrix"]["t3_preregistration"]
    t3_signoff = signoff["t3_preregistration"]
    for key in (
        "listed_runs",
        "candidates",
        "seeds",
        "full_activation_count",
        "activation",
        "unactivated_status",
        "n_core_counting",
    ):
        assert t3_signoff[key] == t3_cfg[key], key
    t3_ids_cfg = {r["run_id"] for r in cfg["run_matrix"]["runs"] if r["stage"] == "T3"}
    assert len(t3_ids_cfg) == 10 and t3_ids_cfg == _t3_ids()


def test_template_run_classes_align_with_config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tpl = json.loads(
        (
            REPO_ROOT
            / "config"
            / "campaigns"
            / "v541_templates"
            / "runplan_pin.template.json"
        ).read_text(encoding="utf-8")
    )
    by_class: dict[str, set[str]] = {}
    for run in cfg["run_matrix"]["runs"]:
        by_class.setdefault(run["class"], set()).add(run["run_id"])
    for name, cls in tpl["run_classes"].items():
        assert set(cls["run_ids"]) == by_class[name], name
