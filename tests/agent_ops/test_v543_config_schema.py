"""V5.4.3 campaign schema validator tests (WP0, ADR-0034).

Dry-run scope: PyYAML + stdlib only; no torch, no data, no GPU, no network.
Covers the WP0 skeleton the plan's WP0 exit criteria require:

* the committed config passes ``prereg`` mode on the real YAML, with the two
  by-design ``t1_blocked`` reasons (WP7 budget null + signoff instance not yet
  pinned),
* typed rejections for schema/status, missing blocks, source checkpoint drift,
  Bridge leak flags, parameter-band arithmetic, training dose, appendix-B
  threshold drift, the 18-run matrix / seed assignment, authorization facts and
  G–V registry drift,
* ``--formal`` gating: null budget ⇒ ``V543_BUDGET_NULL``, malformed budget ⇒
  ``V543_BUDGET_INVALID``, unsigned threshold ⇒ ``V543_THRESHOLD``, missing
  record instance ⇒ ``V543_RECORD_MISSING``, a record whose thresholds or
  ``record_sha256`` do not mirror the frozen config ⇒ ``V543_RECORD_MISMATCH``,
* the happy formal path with a well-formed budget table and a correctly sealed
  signoff instance (canonical ``record_sha256``),
* the five ``v543_templates`` files parse and mirror the frozen config.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import v543_config_schema as v  # noqa: E402

CONFIG = REPO_ROOT / "config" / "campaigns" / "pes2ts_v543_dual_decoder_50k.yaml"
TPL_DIR = REPO_ROOT / "config" / "campaigns" / "v543_templates"
PLAN_DIR = Path("outputs/pesn/v543_dual_decoder_50k/plan")


@pytest.fixture(scope="module")
def base_cfg() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["schema"] == v.SCHEMA_ID
    return cfg


def _mut(base_cfg: dict, fn) -> dict:
    cfg = copy.deepcopy(base_cfg)
    fn(cfg)
    return cfg


def _fill_budget(cfg: dict) -> None:
    cfg["budget"]["per_candidate_core_time"] = {
        did: {
            "core_seconds": 2000.0,
            "w1_core_ratio": 1.0,
            "projected_wall_hours": 0.6,
            "approved_for_formal": True,
            "status": "approved",
        }
        for did in v.FORMAL_DECODER_IDS
    }
    cfg["budget"]["per_candidate_status"] = "signed_vp7_profile"


def _record_payload(cfg: dict) -> dict:
    signed = {
        "status": "frozen",
        "signed_on": "2026-11-02",
        "signed_by": "repository_owner_via_dsh_orchestration_session",
    }
    signed.update({k: cfg["thresholds"][k] for k in v.THRESHOLDS})
    return {
        "schema": v.SIGNOFF_RECORD_SCHEMA_ID,
        "campaign_id": cfg["campaign_id"],
        "gv_campaign_id": cfg["gv"]["campaign_id"],
        "config_path": v.CAMPAIGN_CONFIG_REL,
        "config_sha256": "0" * 64,
        "schema_id": cfg["schema"],
        "thresholds_signed_frozen": signed,
        "signed_by_roles": {
            "authorization": "repository_owner_via_dsh_orchestration_session",
            "governance_owner": "main_session_owner",
            "statistics_owner": "main_session_owner",
            "governance_signoff": "main_session_owner",
            "parity_signoff": None,
            "training_signoff": None,
            "evaluation_signoff": "main_session_owner",
            "resource_signoff": None,
            "oracle_reviewer": "oracle_gate0_reviewer",
            "formal_launch_note": (
                "D1–D8 signed 2026-11-02; reviewer NOT signatory — "
                "oracle_gate0_reviewer"),
        },
    }


def _seal(payload: dict) -> dict:
    body = {k: val for k, val in payload.items() if k != "record_sha256"}
    payload["record_sha256"] = hashlib.sha256(
        v._canonical_json_bytes(body)).hexdigest()
    return payload


def _write_record(root: Path, cfg: dict, *, seal: bool = True) -> Path:
    plan = Path(root) / PLAN_DIR
    plan.mkdir(parents=True, exist_ok=True)
    payload = _record_payload(cfg)
    if seal:
        _seal(payload)
    path = plan / "preregistration_signoff.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# prereg mode on the real committed config
# ---------------------------------------------------------------------------
def test_prereg_passes_on_real_config(base_cfg):
    report = v.validate_config(base_cfg)
    assert report["ok"] is True
    assert report["schema"] == v.SCHEMA_ID
    assert report["campaign_id"] == "PES2TS_V543_DUAL_DECODER_50K"
    assert report["mode"] == "prereg"
    # WP0 has no budget table yet (WP7 profile) — the only remaining blocker;
    # the signed preregistration_signoff INSTANCE is now pinned, so the record
    # requirement no longer contributes a t1_blocked reason.
    assert report["t1_config_ready"] is False
    assert len(report["t1_blocked"]) == 1
    blocked = " ".join(report["t1_blocked"])
    assert "budget.per_candidate_core_time null" in blocked
    assert "preregistration_signoff_record" not in blocked
    for name in (
        "schema", "gv_registry", "governance", "authorization",
        "records_spec", "source", "param_budget", "training", "thresholds",
        "run_matrix", "governance_records",
    ):
        assert name in report["checks"], name
    assert report["null_thresholds"] == []


def test_prereg_accepts_explicit_prereg_flag(base_cfg):
    report = v.validate_config(base_cfg, formal=False)
    assert report["mode"] == "prereg"


def test_numeric_invariants_are_the_plan_numbers():
    assert (v.BAND_LOW, v.BAND_HIGH) == (34200, 37800)
    assert v.PER_DECODER_CAP == 40000
    assert v.TOTAL_MODEL_TARGET == 129619
    assert v.TOTAL_MODEL_BAND_MAX == 133219
    assert v.TOTAL_MODEL_HARD_MAX == 150000
    assert v.ENCODER_FROZEN_PARAMS == 57619 == 55392 + 49 + 2178
    assert v.DOSE_FLAT == 37900 and v.WARMUP_UPDATES == 400
    assert v.DERIVED_POST_WARMUP == 37500
    assert v.FORMAL_RUNS_MAX == 18
    assert v.TOTAL_FORMAL_UPDATES == 682200 == 18 * 37900
    assert len(v.THRESHOLDS) == 19
    assert len(v.EXPECTED_RUN_IDS) == 18
    assert len(v.TYPED_REJECTIONS) == 23
    assert len(set(v.TYPED_REJECTIONS)) == 23


# ---------------------------------------------------------------------------
# typed rejections — identity / structure
# ---------------------------------------------------------------------------
def test_rejects_wrong_schema(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.__setitem__("schema", "pes2ts_v542_campaign_v1"))
    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V543_SCHEMA"


def test_rejects_executed_status_claim(base_cfg):
    def mutate(c):
        c["status"] = "executed"

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_SCHEMA"


@pytest.mark.parametrize(
    "block",
    ["source", "data", "bridge", "tracks", "training", "cache", "screening",
     "evaluation", "thresholds", "resources", "conditional_generation",
     "run_matrix", "budget", "governance", "gv"],
)
def test_rejects_missing_block(base_cfg, block):
    cfg = _mut(base_cfg, lambda c: c.__setitem__(block, None))
    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V543_MISSING_FIELD"


def test_rejects_checkpoint_sha_drift(base_cfg):
    def mutate(c):
        c["source"]["checkpoints"]["29"]["sha256"] = "0" * 64

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_SOURCE"


def test_rejects_confirm_access_enabled(base_cfg):
    def mutate(c):
        c["source"]["allow_confirm_access"] = True

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_SOURCE"


def test_rejects_bridge_supervision_leak_flag(base_cfg):
    def mutate(c):
        c["bridge"]["allow_supervision_leak"] = True

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_BRIDGE"


def test_rejects_decoder_raw_xyz(base_cfg):
    def mutate(c):
        c["bridge"]["allow_decoder_raw_xyz"] = True

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_BRIDGE"


@pytest.mark.parametrize("value", [37800, 34200, 36001, "36000", 36000.0])
def test_rejects_target_param_drift(base_cfg, value):
    def mutate(c):
        c["tracks"]["move"]["target_params"] = value

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code in {"V543_TRACK", "V543_PARAM_BUDGET"}


def test_rejects_band_pct_drift(base_cfg):
    def mutate(c):
        c["tracks"]["process"]["param_band_pct"] = 10.0

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code in {"V543_TRACK", "V543_PARAM_BUDGET"}


def test_rejects_per_decoder_cap_rise(base_cfg):
    def mutate(c):
        c["resources"]["move_refiner_params_max"] = 41000

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_PARAM_BUDGET"


def test_rejects_whole_model_cap_drift(base_cfg):
    def mutate(c):
        c["resources"]["total_model_params_max"] = 160000

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code in {"V543_PARAM_BUDGET", "V543_RESOURCE"}


def test_rejects_loss_lambda_drift(base_cfg):
    def mutate(c):
        c["tracks"]["process"]["lambda_endpoint"] = 0.0124

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_TRACK"


def test_rejects_training_dose_drift(base_cfg):
    def mutate(c):
        c["training"]["updates"] = 37500

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_TRAINING"


def test_rejects_torch_compile(base_cfg):
    def mutate(c):
        c["training"]["torch_compile"] = True

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_TRAINING"


def test_rejects_sampler_spec_drift(base_cfg):
    def mutate(c):
        c["training"]["sampler_spec_sha256"] = "f" * 64

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_TRAINING"


@pytest.mark.parametrize(
    "key,value",
    [
        ("move_rmse_regression_rel_max_pct", 3.0),
        ("process_auprc_drop_abs_max", 0.02),
        ("move_exit_alpha_each", 0.05),
        ("max_latency_ratio_p95", 2.5),
    ],
)
def test_rejects_threshold_drift(base_cfg, key, value):
    def mutate(c):
        c["thresholds"][key] = value

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_THRESHOLD"


def test_rejects_extra_threshold_key(base_cfg):
    def mutate(c):
        c["thresholds"]["rmse_zero_regression_rel_max_pct"] = 2.0  # V542 name

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_THRESHOLD"


def test_rejects_seed_assignment_drift(base_cfg):
    def mutate(c):
        c["run_matrix"]["t3_seed17"] = ["M1", "M2", "P1", "P2", "P3", "P4", "M3"]

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_RUN_MATRIX"


def test_rejects_baseline_in_formal_matrix(base_cfg):
    def mutate(c):
        c["run_matrix"]["t4_seed29"] = ["M0"] + list(c["run_matrix"]["t3_seed17"])[:6]

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_RUN_MATRIX"


def test_rejects_run_count_drift(base_cfg):
    def mutate(c):
        c["run_matrix"]["formal_runs_max"] = 20

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_RUN_MATRIX"


def test_rejects_confirm_authorized_true(base_cfg):
    def mutate(c):
        c["governance"]["authorization"]["confirm_authorized"] = True

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_AUTHORIZATION"


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_qc_calls", 1),
        ("swap_allowed", True),
        ("single_lane", False),
        ("thresholds_signed_frozen", False),
        ("signed_on", "2026-10-29"),
    ],
)
def test_rejects_authorization_drift(base_cfg, key, value):
    def mutate(c):
        c["governance"]["authorization"][key] = value

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_AUTHORIZATION"


def test_rejects_missing_d_decision(base_cfg):
    def mutate(c):
        c["governance"]["authorization"]["signed_decisions"] = [
            "D1", "D2", "D3", "D4", "D5", "D6", "D7"]

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_AUTHORIZATION"


def test_rejects_docs_gate_drift(base_cfg):
    def mutate(c):
        c["governance"]["docs_gate"] = "python scripts/agent_ops/docs_consistency.py"

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_GOVERNANCE"


def test_rejects_gv_gate_drift(base_cfg):
    def mutate(c):
        c["gv"]["gates"] = list(c["gv"]["gates"])[:-1]

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_GV_MISMATCH"


def test_rejects_gv_map_version_drift(base_cfg):
    def mutate(c):
        c["gv"]["map_version"] = "2020-01-01.0"

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate))
    assert exc.value.code == "V543_GV_MISMATCH"


def test_registry_entry_is_proposed(base_cfg):
    registry = v._load_gv_registry()
    entry = registry["campaigns"]["V5.4.3"]
    assert entry["status"] == "proposed"
    assert entry["config"] == v.CAMPAIGN_CONFIG_REL
    assert entry["primary_g"] == "G3.P.2"
    assert set(entry["gates"]) == set(v.GV_EXPECTED_GATES)
    # every canonical T1 gate is a declared registry gate
    assert set(v.T1_CANONICAL_GATES) <= set(entry["gates"])


# ---------------------------------------------------------------------------
# formal-mode gating
# ---------------------------------------------------------------------------
def test_formal_blocks_on_null_budget(base_cfg, tmp_path):
    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(base_cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V543_BUDGET_NULL"


@pytest.mark.parametrize(
    "bad_table",
    [
        ["M1", 2000.0],
        "signed",
        {"M1": 2000.0},
        {"M1": 2000.0, "M2": 2000.0, "M3": 2000.0, "P1": 2000.0,
         "P2": 2000.0, "P3": 2000.0, "P4": 0.0},
        {"M1": 2000.0, "M2": 2000.0, "M3": 2000.0, "P1": 2000.0,
         "P2": 2000.0, "P3": 2000.0, "P4": True},
        {"M1": 2000.0, "M2": 2000.0, "M3": 2000.0, "P1": 2000.0,
         "P2": 2000.0, "P3": 2000.0, "P4": float("nan")},
    ],
)
def test_formal_rejects_malformed_budget(base_cfg, tmp_path, bad_table):
    def mutate(c):
        c["budget"]["per_candidate_core_time"] = bad_table

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate), formal=True, record_root=tmp_path)
    assert exc.value.code == "V543_BUDGET_INVALID"


def test_formal_rejects_unsigned_threshold(base_cfg, tmp_path):
    def mutate(c):
        c["thresholds"]["process_fpr_constraint"] = None

    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(_mut(base_cfg, mutate), formal=True, record_root=tmp_path)
    assert exc.value.code == "V543_THRESHOLD"


def test_formal_rejects_missing_record(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    _fill_budget(cfg)
    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V543_RECORD_MISSING"


def test_formal_passes_with_sealed_record(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    _fill_budget(cfg)
    _write_record(tmp_path, cfg)
    report = v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert report["ok"] is True
    assert report["mode"] == "formal"
    assert report["t1_config_ready"] is True
    assert report["t1_blocked"] == []


def test_formal_rejects_stale_record_sha(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    _fill_budget(cfg)
    path = _write_record(tmp_path, cfg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["thresholds_signed_frozen"]["move_sign_drop_abs_max"] = 0.02
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V543_RECORD_MISMATCH"


def test_formal_rejects_config_path_drift(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    _fill_budget(cfg)
    path = _write_record(tmp_path, cfg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config_path"] = "config/campaigns/pes2ts_v542_backbone_50k.yaml"
    _seal(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V543_RECORD_MISMATCH"


def test_formal_rejects_oracle_as_signatory(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    _fill_budget(cfg)
    path = _write_record(tmp_path, cfg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["signed_by_roles"]["statistics_owner"] = "oracle_gate0_reviewer"
    _seal(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V543_RECORD_MISMATCH"


def test_formal_rejects_empty_roles(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    _fill_budget(cfg)
    path = _write_record(tmp_path, cfg)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["signed_by_roles"] = {}
    _seal(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(v.V543SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V543_RECORD_MISMATCH"


# ---------------------------------------------------------------------------
# CLI (in-process)
# ---------------------------------------------------------------------------
def test_cli_prereg_returns_zero(capsys):
    assert v.main(["--prereg", "--config", str(CONFIG)]) == 0
    out = capsys.readouterr().out
    assert "V543 config OK" in out
    assert "Formal launch not config-ready" in out


def test_cli_json_reports_code_on_rejection(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: nope\n", encoding="utf-8")
    assert v.main(["--json", "--config", str(bad)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["code"] == "V543_SCHEMA"


def test_cli_rejects_prereg_and_formal_together():
    assert v.main(["--prereg", "--formal"]) == 2


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------
def test_all_five_templates_parse():
    names = [
        "preregistration_signoff.template.json",
        "runplan_pin.template.json",
        "budget_policy.template.json",
        "parity_certificate.template.json",
        "decoder_context.template.json",
    ]
    assert len(names) == 5
    for name in names:
        path = TPL_DIR / name
        assert path.is_file(), name
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload.get("_template_note"), name


def test_signoff_template_mirrors_signed_thresholds(base_cfg):
    payload = json.loads(
        (TPL_DIR / "preregistration_signoff.template.json").read_text(
            encoding="utf-8"))
    assert payload["schema"] == v.SIGNOFF_RECORD_SCHEMA_ID
    assert payload["campaign_id"] == base_cfg["campaign_id"]
    assert payload["gv_campaign_id"] == base_cfg["gv"]["campaign_id"]
    assert payload["config_path"] == v.CAMPAIGN_CONFIG_REL
    signed = payload["thresholds_signed_frozen"]
    config_values = {k: base_cfg["thresholds"][k] for k in v.THRESHOLDS}
    mirrored = {k: signed[k] for k in v.THRESHOLDS}
    assert v._canonical_json_bytes(mirrored) == v._canonical_json_bytes(
        config_values)
    assert payload["authorization"]["confirm_authorized"] is False
    assert payload["authorization"]["new_qc_calls"] == 0
    assert payload["authorization"]["signed_decisions"] == [
        f"D{i}" for i in range(1, 9)]
    assert set(payload["signed_by_roles"]) == v.SIGNOFF_ROLE_KEYS


def test_budget_template_carries_param_arithmetic():
    payload = json.loads(
        (TPL_DIR / "budget_policy.template.json").read_text(encoding="utf-8"))
    budget = payload["parameter_budget"]
    assert budget["per_track_band"] == [v.BAND_LOW, v.BAND_HIGH]
    assert budget["single_decoder_hard_cap"] == v.PER_DECODER_CAP
    assert budget["whole_model_target"] == v.TOTAL_MODEL_TARGET
    assert budget["whole_model_in_band_max"] == v.TOTAL_MODEL_BAND_MAX
    assert budget["whole_model_hard_cap"] == v.TOTAL_MODEL_HARD_MAX
    assert set(payload["per_decoder"]) == set(v.KNOWN_DECODERS)
    for did in v.FORMAL_DECODER_IDS:
        assert payload["per_decoder"][did]["core_seconds"] is None


def test_parity_template_pins_three_a0_checkpoints():
    payload = json.loads(
        (TPL_DIR / "parity_certificate.template.json").read_text(
            encoding="utf-8"))
    assert payload["checkpoint_count"] == 3
    assert payload["required_count"] == 3
    assert len(payload["zero_residual_equations"]["required"]) == 7
    by_id = {row["id"]: row for row in payload["checkpoints"]}
    for seed, spec in v.SOURCE_CHECKPOINTS.items():
        row = by_id[f"V542_A0_S{seed}"]
        assert row["path"] == spec["path"]
        assert row["sha256"] == spec["sha256"]
        assert row["required"] is True
    assert payload["not_evidence_for"]


def test_parity_template_separates_the_two_state_dict_counts():
    """ADR-0033 identity 口径 (w_v once = 43) vs plan §1.2 strict-load 口径 (45)."""
    payload = json.loads(
        (TPL_DIR / "parity_certificate.template.json").read_text(
            encoding="utf-8"))
    semantics = payload["state_dict_count_semantics"]
    payload_count = semantics["payload_key_count"]
    expanded = semantics["expanded_target_key_count"]
    assert payload_count["expected"] == 43
    assert expanded["expected"] == 45
    # the w_v dual-write is the only legal expansion and it yields TWO target
    # keys (backbone.w_v + move_decoder.w_v), so 45 - 43 == 2.
    assert expanded["expected"] - payload_count["expected"] == 2
    assert semantics["consistency_rule"].startswith(
        "expanded_target_key_count = payload_key_count + 2")
    assert "ADR-0033" in payload_count["authority"]
    assert "§1.2" in expanded["authority"]
    assert "37 backbone" in payload_count["derivation"]
    assert "2 move_decoder.w_v" in expanded["derivation"]
    assert "6 process_head" in expanded["derivation"]
    for row in payload["checkpoints"]:
        counts = row["state_dict_counts"]
        assert counts["payload_key_count_measured"] is None
        assert counts["expanded_target_key_count_measured"] is None
        assert counts["count_scope_of_this_audit"] is None


def test_preregistration_signoff_instance_is_sealed():
    """The WP0 record instance mirrors the config and carries a valid digest."""
    path = REPO_ROOT / PLAN_DIR / "preregistration_signoff.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == v.SIGNOFF_RECORD_SCHEMA_ID
    assert payload["campaign_id"] == v.CAMPAIGN_ID
    assert payload["config_path"] == v.CAMPAIGN_CONFIG_REL
    assert payload["gv_campaign_id"] == v.GV_CAMPAIGN_ID
    assert payload["schema_id"] == v.SCHEMA_ID
    assert set(payload["signed_by_roles"]) == v.SIGNOFF_ROLE_KEYS
    assert payload["signed_by_roles"]["statistics_owner"] == "main_session_owner"
    assert payload["signed_by_roles"]["evaluation_signoff"] == "main_session_owner"
    assert payload["signed_by_roles"]["oracle_reviewer"] == "oracle_gate0_reviewer"
    assert payload["authorization"]["signed_decisions"] == [
        f"D{i}" for i in range(1, 9)]
    assert payload["authorization"]["confirm_authorized"] is False
    assert payload["record_sha256"] == v._record_canonical_sha256(payload)


def test_runplan_template_matches_18_run_ceiling():
    payload = json.loads(
        (TPL_DIR / "runplan_pin.template.json").read_text(encoding="utf-8"))
    assert payload["formal_runs_max"] == v.FORMAL_RUNS_MAX
    assert payload["total_formal_updates"] == v.TOTAL_FORMAL_UPDATES
    nominal = payload["run_classes"]["nominal"]["run_ids"]
    conditional = payload["run_classes"]["conditional"]["run_ids"]
    assert len(nominal) == 14
    assert len(conditional) == 4
    assert set(nominal) <= v.EXPECTED_RUN_IDS
    assert set(conditional) <= v.EXPECTED_RUN_IDS
    assert payload["dose_semantics"]["updates"] == v.DOSE_FLAT
    assert payload["dose_semantics"]["warmup_updates"] == v.WARMUP_UPDATES


def test_decoder_context_template_lists_frozen_fields():
    payload = json.loads(
        (TPL_DIR / "decoder_context.template.json").read_text(encoding="utf-8"))
    assert payload["context_schema"] == "decoder_context_v1"
    fields = payload["fields"]
    for name in ("graph_s", "key_s", "key_v", "u0", "pair_s",
                 "intent_pair_index", "q0_raw", "endpoint0_logit"):
        assert name in fields, name
    assert fields["graph_s"]["shape"] == ["G", 64]
    assert fields["key_s"]["shape"] == ["K", 48]
    assert fields["key_v"]["shape"] == ["K", 8, 3]
    assert fields["q0_raw"]["source"].startswith("progress_head.fc2")
    move_view = set(payload["read_only_views"]["MoveContextV1"]["fields"])
    process_view = set(payload["read_only_views"]["ProcessContextV1"]["fields"])
    assert "q0_raw" not in move_view and "pair_s" not in move_view
    assert "u0" not in process_view and "key_v" not in process_view
