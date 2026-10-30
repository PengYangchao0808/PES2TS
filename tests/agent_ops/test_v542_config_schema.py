"""V5.4.2 campaign schema validator tests (WP0, ADR-0033 v1.1).

Dry-run scope: PyYAML + stdlib only; no torch, no data, no GPU, no network
(the one exception: the gate-0 A6 pinned-payload cross-check uses
``pytest.importorskip("torch")`` and clean-skips when the pinned file or
torch is absent).
Covers every typed rejection the WP0 v1.1 spec requires (schema / candidate /
budget / threshold / run matrix) plus the interface, dose-block, sampler,
init-source, R3-init and parity-template contracts:

* missing field / wrong schema id / non-unified objective / legacy arm matrix,
* unknown or malformed backbone (A0–A5 / A1-C / A5-C registry + param budget),
* interface drift (BackboneOutput fields + subfields, z/edge_msg widths,
  decoder/PROCESS semantics, A0 capture/w_v/parity assets),
* illegal forward key (outside FORWARD_VIEW_KEYS_V3 whitelist),
* confirm/sealed/QC authorization not false, single-lane/swap/OOM guards,
* null signed thresholds blocking ``--formal``,
* null per-candidate budget: prereg blocker vs ``--formal`` typed block
  ``V542_BUDGET_NULL``,
* G–V registry cross-check (V5.4.2 registered by the main session),
* the exact frozen 26-run matrix (missing/extra/renamed runs rejected),
* v1.1 top-2 / A5-supplement split (A5 never in top2; old
  A1–A5×{29,43}=10 / full_activation_count / h3_pass parent gate rejected),
* seed17_screen_pass predicate (screen, never a three-seed H3 claim),
* H3 statistics contract (bootstrap seed root, sub-seed formula, 10,000
  draws, one-sided p formula, Holm families, frame-level forbidden),
* v1.1 init freeze (seed recipe, construction order, mapping tables,
  A1-C/A5-C init_source_id, A2 V0/phi zero-init, param-group lrs),
* identity/start contract (source_manifest required, canonical
  t1_start_requires, A0 decision INSTANCE before T1),
* smoke 200 / profile 40+40 / diagnostic 400+1600 dose constants, live
  sampler-spec drift,
* run-outcome six-word vocabulary vs. legacy ledger status machine separation,
* gate-2 Ruling 2: per-candidate ``w_g_bias`` (A2/A4 biased
  ``nn.Linear(5,48)``=288; A5/A5-C ``bias=False``=240/cell) and
  ``gru_cell_bias`` (A5 ``[true]``=14,112; A5-C ``[true, false]`` second
  cell bias-free=13,824) exact record, incl. token-preserving
  near-miss rejections and the frozen totals unchanged,
* gate-0 A5 closure (P3 lane D): the two T1 record requirements
  (``a0_reuse_decision_record`` + ``preregistration_signoff_record``)
  are machine-checked governance records — prereg surfaces missing/
  drifted instances as ``t1_blocked`` reasons (WP0 absence by design),
  ``--formal`` typed-rejects (``V542_RECORD_MISSING`` /
  ``V542_RECORD_MISMATCH``) with canonical-file-byte SHA rules,
* gate-3 B2/C2 closure (P3 remediation lane R2): the record gate
  mirrors the frozen config — wrong ``config_path``, empty/drifted
  ``signed_by_roles``, wrong role bindings, Oracle-as-signatory,
  ``schema`` drift, thresholds encoding drift (4 vs 4.0) are typed
  ``V542_RECORD_MISMATCH``; each record declares ``record_sha256``
  verified against the identity module's ``canonical_bytes`` digest
  (missing ⇒ ``V542_RECORD_MISSING``, wrong/stale ⇒
  ``V542_RECORD_MISMATCH``); ``runplan_pin_record`` stays
  WP5-runner-verified (the validator never reads the pin instance),
* gate-3 C4/A1 closure (P3 remediation lane R2): the budget formal
  gate requires a mapping covering every backbone with a finite
  POSITIVE core time — list/string/bool/null-entry/zero/negative/
  NaN/inf/partial/extra-key tables are typed ``V542_BUDGET_INVALID``
  (null/empty stays ``V542_BUDGET_NULL``; prereg never requires
  numbers),
* gate-0 A2 closure (P3 lane D): final-selection eligibility/single/
  no-candidate, ``h3_eligible_candidates``, ``champion_rule``, seed17
  H2 sign threshold and ``core_loop`` are exact/structured — every
  token-preserving near-miss from the Oracle probe tables is rejected,
* the five WP0 templates parse and mirror the frozen config.
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

from scripts.agent_ops import v542_config_schema as v  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "agent_ops" / "v542_config_schema.py"
CONFIG = REPO_ROOT / "config" / "campaigns" / "pes2ts_v542_backbone_50k.yaml"
TPL_DIR = REPO_ROOT / "config" / "campaigns" / "v542_templates"


@pytest.fixture(scope="module")
def base_cfg() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["schema"] == v.SCHEMA_ID
    return cfg


def _mut(base_cfg: dict, fn) -> dict:
    cfg = copy.deepcopy(base_cfg)
    fn(cfg)
    return cfg


def _fill_budget(c: dict, include_attr: bool = True) -> None:
    ids = ["A0", "A1", "A2", "A3", "A4", "A5"]
    if include_attr:
        ids += ["A1-C", "A5-C"]
    c["budget"]["per_candidate_core_time"] = {
        cid: {
            "core_seconds": 2000.0,
            "w1_core_ratio": 1.0,
            "projected_wall_hours": 0.6,
            "approved_for_formal": True,
            "status": "approved",
        }
        for cid in ids
    }
    c["budget"]["per_candidate_status"] = "signed_v542_profile"


def _record_canonical_sha(payload: dict) -> str:
    """The frozen record digest (gate-3 C2): sha256 of the identity
    module's ``canonical_bytes`` over the payload WITHOUT the
    ``record_sha256`` field."""
    import hashlib

    from pes2ts_core.pesn.models.v542.identity import canonical_bytes

    body = {k: v for k, v in payload.items() if k != "record_sha256"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def _seal_record(payload: dict) -> dict:
    """Fill + return the payload with a valid ``record_sha256``."""
    payload["record_sha256"] = _record_canonical_sha(payload)
    return payload


def _write_valid_records(root: Path, cfg: dict | None = None) -> None:
    """Write the two machine-checked record instances under *root*.

    gate-0 A5 + gate-3 B2/C2 test seam: instances live at the frozen
    plan-dir paths relative to the injected ``record_root``, mirror the
    frozen config (config_path/campaign_id/schema/roles/thresholds) and
    carry a valid canonical ``record_sha256``.
    """
    if cfg is None:
        cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    plan = Path(root) / "outputs" / "pesn" / "v542_backbone_50k" / "plan"
    plan.mkdir(parents=True, exist_ok=True)
    decision = _seal_record({
        "schema": v.A0_DECISION_SCHEMA_ID,
        "campaign_id": cfg["campaign_id"],
        "gv_campaign_id": cfg["gv"]["campaign_id"],
        "config_path": v.CAMPAIGN_CONFIG_REL,
        "config_sha256": "0" * 64,
        "decision": "retrain_a0_x3",
        "signed_by": "main_session_owner",
        "runs": list(cfg["a0_baseline"]["runs"]),
    })
    (plan / "a0_reuse_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    signed = {"status": "frozen"}
    signed.update(
        {key: cfg["thresholds"][key] for key in v.FORMAL_THRESHOLDS})
    signoff = _seal_record({
        "schema": v.SIGNOFF_RECORD_SCHEMA_ID,
        "campaign_id": cfg["campaign_id"],
        "gv_campaign_id": cfg["gv"]["campaign_id"],
        "config_path": v.CAMPAIGN_CONFIG_REL,
        "config_sha256": "0" * 64,
        "schema_id": cfg["schema"],
        "thresholds_signed_frozen": signed,
        "signed_by_roles": {
            "authorization": "user_directive_20261029",
            "governance_owner": "main_session_gate0_remediation",
            "statistics_owner": "main_session_owner",
            "governance_signoff": "main_session_gate0_remediation",
            "parity_signoff": None,
            "training_signoff": None,
            "evaluation_signoff": "main_session_owner",
            "resource_signoff": None,
            "oracle_reviewer": "oracle_gate0_reviewer",
            "formal_launch_note": (
                "WP0 authoring complete; formal signoff pending owner "
                "re-affirm; reviewer NOT signatory — oracle_gate0_reviewer"),
        },
    })
    (plan / "preregistration_signoff.json").write_text(
        json.dumps(signoff, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mutate_record(root: Path, name: str, fn, *, seal: bool = True) -> dict:
    """Load a written record instance, mutate it and write it back.

    With ``seal=True`` (default) the payload is RE-SEALED so the
    mutation alone — not a stale SHA — triggers the rejection; with
    ``seal=False`` the declared ``record_sha256`` is left untouched (for
    SHA-drift probes).
    """
    path = (Path(root) / "outputs" / "pesn" / "v542_backbone_50k" / "plan"
            / name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    fn(payload)
    if seal:
        payload = _seal_record(payload)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return payload


def _top2_ids() -> set:
    return {
        f"V542_{cid}_S{seed}" for cid in ("A1", "A2", "A3", "A4")
        for seed in (29, 43)
    }


def _a5_supplement_ids() -> set:
    return {"V542_A5_S29", "V542_A5_S43"}


def _run_by_id(cfg: dict, run_id: str) -> dict:
    for run in cfg["run_matrix"]["runs"]:
        if run["run_id"] == run_id:
            return run
    raise AssertionError(f"run {run_id} not registered")


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------
def test_frozen_config_validates_prereg_with_budget_blocker(base_cfg):
    """WP4: the budget is signed from the same-env profile; only the two
    governance record instances still block T1 at prereg."""
    report = v.validate_config(base_cfg, formal=False)
    assert report["ok"] and report["mode"] == "prereg"
    assert report["null_thresholds"] == []
    assert report["t1_config_ready"] is False
    # gate-0 A5: the two governance record instances are intentionally
    # absent until T1 and surface as t1_blocked reasons — prereg stays
    # green (no typed rejection).
    assert len(report["t1_blocked"]) == 2
    joined = " ".join(report["t1_blocked"])
    assert "per_candidate_core_time" not in joined
    assert "a0_reuse_decision_record" in joined
    assert "preregistration_signoff_record" in joined
    # V5.4.2 is registered in g_v_map.yaml — the registry is NOT a blocker
    assert not any("g_v_map" in r for r in report["t1_blocked"])


def test_repo_config_formal_is_typed_record_block(base_cfg):
    """--formal on the signed-budget config = typed record block (A5)."""
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(base_cfg, formal=True)
    assert exc.value.code == "V542_RECORD_MISSING"
    assert "a0_reuse_decision_record" in exc.value.detail


def test_filled_budget_formal_passes_completely(base_cfg, tmp_path):
    """Signed budget + valid record instances = formal fully config-ready."""
    cfg = _mut(base_cfg, _fill_budget)
    # the two machine-checked record instances must exist and mirror the
    # config (gate-0 A5); write them under a tmp record root
    _write_valid_records(tmp_path)
    formal = v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert formal["ok"] and formal["mode"] == "formal"
    assert formal["null_thresholds"] == []
    assert formal["t1_blocked"] == []
    assert formal["t1_config_ready"] is True
    assert "governance_records" in formal["checks"]


def test_cli_exit_zero_on_repo_config():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(CONFIG)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_formal_typed_block_on_repo_config():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--formal", "--config", str(CONFIG), "--json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert report["ok"] is False
    assert report["code"] == "V542_RECORD_MISSING"


def test_cli_json_report_is_parseable():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--config", str(CONFIG)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] and report["schema"] == v.SCHEMA_ID


def test_typed_rejection_codes_authoritative_count():
    """The module defines the authoritative 22 typed rejection codes."""
    assert len(v.TYPED_REJECTIONS) == 22
    assert len(set(v.TYPED_REJECTIONS)) == 22
    assert all(code.startswith("V542_") for code in v.TYPED_REJECTIONS)
    assert "V542_RECORD_MISSING" in v.TYPED_REJECTIONS
    assert "V542_RECORD_MISMATCH" in v.TYPED_REJECTIONS
    assert "V542_BUDGET_INVALID" in v.TYPED_REJECTIONS


# ---------------------------------------------------------------------------
# schema / required fields / objective
# ---------------------------------------------------------------------------
def test_wrong_schema_id_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.update(schema="pes2ts_v541_campaign_v1"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_SCHEMA"


def test_wrong_campaign_id_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.update(campaign_id="PES2TS_V541_EMBED_50K"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_SCHEMA"


@pytest.mark.parametrize(
    "field",
    ["training_contract", "backbone_contract", "a0_baseline", "diagnostic"],
)
def test_missing_required_field_rejected(base_cfg, field):
    cfg = _mut(base_cfg, lambda c, f=field: c.pop(f))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert field in exc.value.detail


def test_none_required_field_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.update(evaluation_contract=None))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"


@pytest.mark.parametrize("bad", ["unified_move_process_v2", "move_only", None, ""])
def test_non_unified_objective_rejected(base_cfg, bad):
    cfg = _mut(base_cfg, lambda c, b=bad: c.update(objective=b))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_OBJECTIVE"


def test_legacy_arm_matrix_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c.update(arms={"A": 1, "B": 2, "C": 3}))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_ARM_MATRIX"


# ---------------------------------------------------------------------------
# backbone registry + param budget
# ---------------------------------------------------------------------------
def test_unknown_backbone_in_registry_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A6"] = {
            "family": "made_up", "hidden": 48,
            "backbone_active": 1000, "estimated": True,
        }

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"
    assert "A6" in exc.value.detail


def test_missing_frozen_backbone_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["backbone_contract"]["candidates"].pop("A3"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"
    assert "A3" in exc.value.detail


def test_backbone_family_drift_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A1"]["family"] = (
            "painn_lite_single_frame_v32")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"


def test_a0_joint_hard_invariant_enforced(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A0"]["joint"] = 57618

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "57619" in exc.value.detail


def test_a0_active_exact_enforced(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A0"]["backbone_active"] = 55393

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "55392" in exc.value.detail


def test_a0_parameters_cannot_claim_estimated(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A0"]["estimated"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"
    assert "estimated" in exc.value.detail


def test_candidate_estimated_must_be_true(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A2"]["estimated"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"


def test_backbone_active_cap_enforced(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A2"]["backbone_active"] = 71000

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "70000" in exc.value.detail


def test_candidate_hidden_width_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A1"]["hidden"] = 32

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"
    assert "48" in exc.value.detail


def test_common_widths_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidate_common_widths"] = {
            "hidden": 48, "z": 32, "edge_msg": 48, "node_input": 21}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"


def test_a1c_must_stay_diagnostic_only(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A1-C"].pop("diagnostic_only")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"
    assert "diagnostic_only" in exc.value.detail


# ---------------------------------------------------------------------------
# gate-2 Ruling 2: per-candidate W_g / GRUCell bias record
# ---------------------------------------------------------------------------
def test_w_g_bias_frozen_values_in_repo_config(base_cfg):
    """The repo config states the code's truth (read-only cross-check).

    A2/A4: biased nn.Linear(5,48) = 288 params; A5/A5-C:
    nn.Linear(5,48,bias=False) = 240 params per cell.  A5's single
    GRUCell(48,48) is biased (14,112); A5-C keeps cell 0 biased and makes
    the SECOND cell bias-free (13,824).  Frozen totals are unchanged.
    """
    reg = base_cfg["backbone_contract"]["candidates"]
    assert reg["A2"]["w_g_bias"] is True
    assert reg["A4"]["w_g_bias"] is True
    assert reg["A5"]["w_g_bias"] is False
    assert reg["A5-C"]["w_g_bias"] is False
    assert reg["A5"]["gru_cell_bias"] == [True]
    assert reg["A5-C"]["gru_cell_bias"] == [True, False]
    # the frozen preregistered totals stay untouched
    assert reg["A2"]["backbone_active"] == 54896
    assert reg["A4"]["backbone_active"] == 55840
    assert reg["A5"]["backbone_active"] == 31584
    assert reg["A5-C"]["backbone_active"] == 53712
    # no W_g / GRU cells on A0/A1/A1-C/A3 — no bias fields either
    for bid in ("A0", "A1", "A1-C", "A3"):
        assert "w_g_bias" not in reg[bid]
        assert "gru_cell_bias" not in reg[bid]


def test_w_g_bias_matches_frozen_constants(base_cfg):
    reg = base_cfg["backbone_contract"]["candidates"]
    for bid, expected in v.W_G_BIAS.items():
        assert reg[bid]["w_g_bias"] is expected, bid
    for bid, expected in v.GRU_CELL_BIAS.items():
        cells = reg[bid]["gru_cell_bias"]
        assert len(cells) == len(expected), bid
        assert all(g is e for g, e in zip(cells, expected)), bid


@pytest.mark.parametrize(
    "bid,flipped",
    [("A2", False), ("A4", False), ("A5", True), ("A5-C", True)],
)
def test_w_g_bias_drift_rejected(base_cfg, bid, flipped):
    def _bad(c):
        c["backbone_contract"]["candidates"][bid]["w_g_bias"] = flipped

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "w_g_bias" in exc.value.detail


@pytest.mark.parametrize("near_miss", ["true", "false", "True", 1, 0, 1.0])
def test_w_g_bias_token_near_misses_rejected(base_cfg, near_miss):
    """Token-preserving near-misses: string/int/float forms of the boolean."""
    def _bad(c):
        c["backbone_contract"]["candidates"]["A2"]["w_g_bias"] = near_miss

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "w_g_bias" in exc.value.detail


def test_w_g_bias_missing_rejected(base_cfg):
    def _bad(c):
        del c["backbone_contract"]["candidates"]["A5"]["w_g_bias"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "w_g_bias" in exc.value.detail


def test_w_g_bias_on_candidate_without_w_g_rejected(base_cfg):
    """A1 conditions via its node MLP — it has no W_g, so no w_g_bias."""
    def _bad(c):
        c["backbone_contract"]["candidates"]["A1"]["w_g_bias"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "w_g_bias" in exc.value.detail


@pytest.mark.parametrize(
    "cells",
    [[True, True], [False, False], [False], [], [True, False, True],
     ["true", "false"], [1, 0]],
)
def test_a5c_gru_cell_bias_drift_rejected(base_cfg, cells):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A5-C"]["gru_cell_bias"] = cells

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "gru_cell_bias" in exc.value.detail


def test_a5_gru_cell_bias_single_cell_frozen(base_cfg):
    """A5 has ONE cell; a second (even matching) entry is drift."""
    def _bad(c):
        c["backbone_contract"]["candidates"]["A5"]["gru_cell_bias"] = [
            True, False]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "gru_cell_bias" in exc.value.detail


def test_gru_cell_bias_missing_rejected(base_cfg):
    def _bad(c):
        del c["backbone_contract"]["candidates"]["A5-C"]["gru_cell_bias"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "gru_cell_bias" in exc.value.detail


def test_gru_cell_bias_on_non_gru_candidate_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["candidates"]["A3"]["gru_cell_bias"] = [True]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "gru_cell_bias" in exc.value.detail


def test_decoder_params_must_be_49(base_cfg):
    def _bad(c):
        c["backbone_contract"]["move_decoder"]["trainable_params"] = 98

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "49" in exc.value.detail


def test_a0_migration_must_restate_invariants(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["joint_params"] = 57618

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"
    assert "57619" in exc.value.detail


def test_threshold_param_invariant_must_match(base_cfg):
    def _bad(c):
        c["thresholds"]["a0_joint_params"] = 57618

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_PARAM_BUDGET"


# ---------------------------------------------------------------------------
# interface contract
# ---------------------------------------------------------------------------
def test_backbone_output_extra_functional_field_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["backbone_output_fields"] = [
            "z", "edge_msg", "u"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "z" in exc.value.detail and "edge_msg" in exc.value.detail


def test_z_width_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["z"]["shape"] = ["G", 32]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "64" in exc.value.detail


def test_z_tanh_saturated_required(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["z"]["tanh_saturated"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "tanh_saturated" in exc.value.detail


def test_z_order_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["z"]["order"] = "edge_order"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "batch_graph_index" in exc.value.detail


def test_edge_msg_width_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["edge_msg"]["shape"] = ["E", 32]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "48" in exc.value.detail


def test_edge_msg_last_layer_required(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["edge_msg"]["last_layer"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "last_layer" in exc.value.detail


def test_edge_msg_autograd_required(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["edge_msg"]["autograd"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "autograd" in exc.value.detail


def test_edge_msg_order_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["edge_msg"]["order"] = "sorted_edge_index"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "input_edge_index_columns" in exc.value.detail


def test_u_must_stay_out_of_interface(base_cfg):
    def _bad(c):
        c["backbone_contract"]["interface"]["u_in_interface"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_decoder_signature_must_be_five_tensors(base_cfg):
    def _bad(c):
        c["backbone_contract"]["move_decoder"]["inputs"] = ["edge_msg", "batch"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "5-tuple" in exc.value.detail


def test_decoder_must_not_read_supervision_keys(base_cfg):
    def _bad(c):
        c["backbone_contract"]["move_decoder"]["reads_supervision_keys"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_a_ref_must_stay_non_learnable(base_cfg):
    def _bad(c):
        c["backbone_contract"]["move_decoder"]["a_ref_learnable"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_a_ref_value_frozen_at_008(base_cfg):
    def _bad(c):
        c["backbone_contract"]["move_decoder"]["a_ref_angstrom"] = 0.05

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "0.08" in exc.value.detail


def test_decoder_output_order_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["move_decoder"]["output_order"] = "batch_order"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "key_index_order" in exc.value.detail


def test_decoder_non_key_zero_policy_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["move_decoder"]["non_key_policy"] = "masked"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "structural_zero_rows" in exc.value.detail


def test_decoder_forbidden_classes_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["move_decoder"]["forbidden"] = [
            "absolute_coordinates"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_process_head_frozen_class_and_params(base_cfg):
    def _bad(c):
        c["backbone_contract"]["process_head"]["params"] = 4356

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "2178" in exc.value.detail


def test_process_head_reads_z_only(base_cfg):
    def _bad(c):
        c["backbone_contract"]["process_head"]["reads"] = ["z", "edge_msg"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "edge_msg" in exc.value.detail


def test_process_head_reads_edge_msg_forbidden(base_cfg):
    def _bad(c):
        c["backbone_contract"]["process_head"]["reads_edge_msg"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_external_output_keys_frozen_five(base_cfg):
    def _bad(c):
        c["backbone_contract"]["process_head"]["external_output_keys"] = [
            "u", "z", "q", "endpoint_logit", "order_score", "node_h"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_forward_move_keys_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["process_head"]["forward_move_keys"] = ["u"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_process_new_external_keys_forbidden(base_cfg):
    def _bad(c):
        c["backbone_contract"]["process_head"]["new_external_keys_forbidden"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_torch_compile_forbidden(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["torch_compile"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_w_v_must_be_same_tensor(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["w_v_sharing"] = "copied_and_frozen"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "SAME tensor" in exc.value.detail


@pytest.mark.parametrize(
    "claim",
    [
        "not same tensor",
        "backbone.w_v is move_decoder.w_v (NOT the same tensor object)",
        "same tensor? not really",
        "shares storage but not same tensor",
    ],
)
def test_w_v_substring_bypass_rejected(base_cfg, claim):
    """Sub-string claims like 'not same tensor' must never pass the check."""
    def _bad(c, text=claim):
        c["backbone_contract"]["a0_migration"]["w_v_sharing"] = text

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "same tensor object" in exc.value.detail


def test_a0_must_be_staregnnv3_subclass(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["subclass_of"] = "NewNet"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_a0_capture_hook_spec_frozen(base_cfg):
    """Hook: message_mlp[-1], exactly once, finally-removed, no detach."""
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["capture"] = (
            "message_mlp[-1] forward hook, once if convenient")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "exactly once" in exc.value.detail
    assert "finally" in exc.value.detail
    assert "no detach" in exc.value.detail


def test_a0_core_loop_must_be_verbatim(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["core_loop"] = "rewritten forward"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_state_dict_aliases_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["state_dict_aliases"] = ["backbone.w_v"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_alias_assertion_torch_equal_required(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["alias_assertion"] = "allclose"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "torch_equal_on_save_and_load" in exc.value.detail


def test_mapping_must_be_explicit_strict_audit(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["mapping"] = "shape匹配"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_parity_cpu_must_be_bitwise(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["cpu"] = "tolerance"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_gpu_parity_tolerance_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["gpu_atol"] = 1e-3

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "atol=1e-6" in exc.value.detail


def test_parity_real_batch_min_frozen(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"][
            "real_data_min_inner_val_batches"] = 4

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "8" in exc.value.detail


def test_parity_q_equals_order_score_required(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["q_equals_order_score"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_parity_evaluator_rule_required(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["evaluator_parity"] = (
            " eyeball the curves")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "V541" in exc.value.detail


def test_parity_tf32_disabled_required(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["tf32_during_parity"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"


def test_parity_required_assets_count_six(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["required_assets"] = 3

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "6" in exc.value.detail


def test_parity_assets_exact_six_pinned(base_cfg):
    """6/6 assets with the exact frozen paths + SHA-256 digests."""
    parity = base_cfg["backbone_contract"]["a0_migration"]["parity"]
    assets = parity["assets"]
    assert len(assets) == len(v.PARITY_ASSETS) == 6
    for asset, (exp_id, exp_path, exp_sha) in zip(assets, v.PARITY_ASSETS):
        assert asset["id"] == exp_id
        assert asset["path"] == exp_path
        assert asset["sha256"] == exp_sha
        assert asset["required"] is True
        assert "/smoke/" not in asset["path"]
    assert parity["smoke_path_forbidden"]


def test_parity_smoke_path_rejected_in_config(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][0]["path"] = (
            "outputs/pesn/v541_embed_50k/smoke/V541_E0_S17/checkpoint_best.pt")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "smoke" in exc.value.detail.lower()


def test_parity_asset_missing_sha_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][2].pop("sha256")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "sha256" in exc.value.detail


def test_parity_asset_optional_row_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][3][
            "required"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "required must be true" in exc.value.detail


def test_parity_asset_drifted_path_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][4]["path"] = (
            "outputs/pesn/v53mp/v53mp_20k_C_s29/checkpoint_last.pt")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "V53MP_C_S29" in exc.value.detail


def test_parity_missing_asset_row_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"].pop()

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "exactly 6" in exc.value.detail


# ---------------------------------------------------------------------------
# fix-6 C3 + gate-0 A6: parity asset tiers and update provenance
# ---------------------------------------------------------------------------
def test_parity_asset_tiers_frozen_in_config(base_cfg):
    parity = base_cfg["backbone_contract"]["a0_migration"]["parity"]
    for asset in parity["assets"]:
        tier, run_updates, checkpoint_update = v.PARITY_ASSET_TIERS[
            asset["id"]]
        assert asset["source_tier"] == tier, asset["id"]
        assert asset["run_updates"] == run_updates, asset["id"]
        assert asset["checkpoint_update"] == checkpoint_update, asset["id"]
        # gate-0 A6: training_updates is the run-dose alias, NOT the
        # checkpoint update
        assert asset["training_updates"] == run_updates, asset["id"]
        assert asset["checkpoint_update"] <= asset["run_updates"], asset["id"]
        assert asset["required"] is True
    # V541 E0 = best inner-val checkpoint from a 37,900-update run (MAIN);
    # V53MP C = best inner-val checkpoint from a 20k S3 15,400-update run
    for aid in ("V541_E0_S17", "V541_E0_S29", "V541_E0_S43"):
        tier, run_updates, checkpoint_update = v.PARITY_ASSET_TIERS[aid]
        assert tier == "v541_e0_runs_final_37900" and run_updates == 37900
        assert checkpoint_update in (37000, 36400, 37800)
    assert v.PARITY_ASSET_TIERS["V541_E0_S17"][2] == 37000
    assert v.PARITY_ASSET_TIERS["V541_E0_S29"][2] == 36400
    assert v.PARITY_ASSET_TIERS["V541_E0_S43"][2] == 37800
    assert v.PARITY_ASSET_TIERS["V53MP_C_S17"][2] == 15400
    assert v.PARITY_ASSET_TIERS["V53MP_C_S29"][2] == 15300
    assert v.PARITY_ASSET_TIERS["V53MP_C_S43"][2] == 15400
    for aid in ("V53MP_C_S17", "V53MP_C_S29", "V53MP_C_S43"):
        tier, run_updates, checkpoint_update = v.PARITY_ASSET_TIERS[aid]
        assert tier == "v53mp_20k_s3_formal_15400_auxiliary"
        assert run_updates == 15400
        assert checkpoint_update <= 15400
    policy = parity["asset_tier_policy"]
    assert tuple(policy["forbidden"]) == v.PARITY_TIER_POLICY_FORBIDDEN
    assert "37900" in policy["v541_e0_runs_final_37900"].replace(",", "")
    assert "37000/36400/37800" in policy["v541_e0_runs_final_37900"]
    assert "15400" in policy["v53mp_20k_s3_formal_15400_auxiliary"].replace(
        ",", "")
    assert "15400/15300/15400" in policy["v53mp_20k_s3_formal_15400_auxiliary"]
    assert "非钉死" in parity["smoke_path_forbidden"] or (
        "non-pinned" in parity["smoke_path_forbidden"])


@pytest.mark.parametrize("idx", range(6))
def test_parity_asset_tier_drift_rejected(base_cfg, idx):
    """Every row's source_tier is pinned — flipping any one is a reject."""
    def _bad(c, i=idx):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][i][
            "source_tier"] = "wrong_tier_label"

    cfg = _mut(base_cfg, _bad)
    aid = cfg["backbone_contract"]["a0_migration"]["parity"]["assets"][idx]["id"]
    exp_tier = v.PARITY_ASSET_TIERS[aid][0]
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert aid in exc.value.detail and exp_tier in exc.value.detail


@pytest.mark.parametrize("idx", range(6))
def test_parity_asset_updates_drift_rejected(base_cfg, idx):
    """training_updates (run-dose alias) is pinned per row (37,900 vs 15,400)."""
    def _bad(c, i=idx):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][i][
            "training_updates"] = 999

    cfg = _mut(base_cfg, _bad)
    aid = cfg["backbone_contract"]["a0_migration"]["parity"]["assets"][idx]["id"]
    exp_updates = v.PARITY_ASSET_TIERS[aid][1]
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert aid in exc.value.detail and str(exp_updates) in exc.value.detail


@pytest.mark.parametrize("idx", range(6))
def test_parity_asset_run_updates_drift_rejected(base_cfg, idx):
    """run_updates is pinned per row to the tier dose (gate-0 A6)."""
    def _bad(c, i=idx):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][i][
            "run_updates"] = 999

    cfg = _mut(base_cfg, _bad)
    aid = cfg["backbone_contract"]["a0_migration"]["parity"]["assets"][idx]["id"]
    exp_run = v.PARITY_ASSET_TIERS[aid][1]
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert aid in exc.value.detail and "run_updates" in exc.value.detail
    assert str(exp_run) in exc.value.detail


@pytest.mark.parametrize("idx", range(6))
def test_parity_asset_checkpoint_update_drift_rejected(base_cfg, idx):
    """checkpoint_update is pinned to the payload's actual best update."""
    def _bad(c, i=idx):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][i][
            "checkpoint_update"] = 999

    cfg = _mut(base_cfg, _bad)
    aid = cfg["backbone_contract"]["a0_migration"]["parity"]["assets"][idx]["id"]
    exp_ckpt = v.PARITY_ASSET_TIERS[aid][2]
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert aid in exc.value.detail and "checkpoint_update" in exc.value.detail
    assert str(exp_ckpt) in exc.value.detail


@pytest.mark.parametrize("bad_value", [0, -1, "37900", 37900.0, True, None])
def test_parity_run_updates_must_be_positive_int(base_cfg, bad_value):
    """gate-0 A6: run_updates must be a genuine positive int."""
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][0][
            "run_updates"] = bad_value

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "run_updates must be a positive int" in exc.value.detail


@pytest.mark.parametrize("bad_value", [0, -1, "37000", 37000.0, False, None])
def test_parity_checkpoint_update_must_be_positive_int(base_cfg, bad_value):
    """gate-0 A6: checkpoint_update must be a genuine positive int."""
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][0][
            "checkpoint_update"] = bad_value

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "checkpoint_update must be a positive int" in exc.value.detail


def test_parity_checkpoint_update_beyond_run_updates_rejected(base_cfg):
    """A checkpoint update beyond the run dose is impossible (A6)."""
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][0][
            "checkpoint_update"] = 37901  # > run_updates 37900

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "checkpoint_update" in exc.value.detail
    assert "run_updates" in exc.value.detail


def test_parity_training_updates_must_equal_run_updates(base_cfg):
    """training_updates is only the run-dose alias — the checkpoint
    update value there is a reject (gate-0 A6 single-meaning rule)."""
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][0][
            "training_updates"] = 37000  # checkpoint update, not run dose

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "training_updates must equal run_updates" in exc.value.detail


def test_parity_missing_run_updates_field_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][4].pop(
            "run_updates")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "run_updates" in exc.value.detail


def test_parity_missing_checkpoint_update_field_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][4].pop(
            "checkpoint_update")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "checkpoint_update" in exc.value.detail


# ---------------------------------------------------------------------------
# gate-0 A6: pinned payload provenance (read-only; clean skip when absent)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("asset_id", sorted(v.PARITY_ASSET_TIERS))
def test_pinned_payload_update_matches_checkpoint_update(base_cfg, asset_id):
    """Read-only cross-check of the config against the actual payload.

    Loads the pinned ``checkpoint_best.pt`` ONLY when the file exists
    (clean ``pytest.skip`` when absent — the read-only lane owns those
    bytes); hard-fails if ``payload["update"] != checkpoint_update``.
    """
    asset = next(
        a for a in base_cfg["backbone_contract"]["a0_migration"]["parity"]
        ["assets"] if a["id"] == asset_id)
    path = REPO_ROOT / str(asset["path"])
    if not path.is_file():
        pytest.skip(f"pinned parity asset absent in this environment: "
                    f"{asset['path']}")
    torch = pytest.importorskip("torch")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    assert int(payload["update"]) == int(asset["checkpoint_update"]) == \
        v.PARITY_ASSET_TIERS[asset_id][2]


def test_parity_missing_tier_field_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][4].pop(
            "source_tier")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "source_tier" in exc.value.detail


def test_parity_tier_policy_block_required(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"].pop(
            "asset_tier_policy")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "asset_tier_policy" in exc.value.detail


def test_parity_tier_policy_forbids_smoke_and_non_pinned(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["asset_tier_policy"][
            "forbidden"] = ["smoke_checkpoints"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "non_pinned_checkpoints" in exc.value.detail


def test_parity_smoke_forbidden_text_must_mention_non_pinned(base_cfg):
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"][
            "smoke_path_forbidden"] = "smoke checkpoints never count"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "non-pinned" in exc.value.detail


def test_gates_parity_rule_gpu_hard_cap_tokens(base_cfg):
    """fix-6 C3: 1e-6/1e-5 hard cap + floor>cap => inconclusive, no relax."""
    rule = base_cfg["gates"]["parity"]["rule"]
    for token in ("1e-6/1e-5", "floor>cap", "inconclusive"):
        assert token in rule, token
    assert base_cfg["thresholds"]["gpu_parity_atol"] == 1e-6
    assert base_cfg["thresholds"]["gpu_parity_rtol"] == 1e-5


def test_gates_parity_rule_missing_hard_cap_token_rejected(base_cfg):
    def _bad(c):
        c["gates"]["parity"]["rule"] = c["gates"]["parity"]["rule"].replace(
            "floor>cap", "floor<=cap")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "floor>cap" in exc.value.detail


def test_gpu_parity_tolerance_hard_caps_frozen(base_cfg):
    """fix-6: the GPU caps are exact; a looser tolerance is a typed reject."""
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["gpu_atol"] = 1e-5

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "1e-06" in exc.value.detail or "1e-6" in exc.value.detail


def test_parity_schema_error_text_is_tiered_not_37900_only():
    """fix-6 C3: the path-mismatch message must not demand a 37,900 final."""
    import inspect

    source = inspect.getsource(v)
    assert "pin the final 37,900-update checkpoint" not in source
    assert "20k S3 formal" in source
    assert "15,400-update checkpoint" in source


def test_parity_error_message_names_tiered_policy(base_cfg):
    """Drifting a V53MP path reports BOTH tier policies, not '37,900 only'."""
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["parity"]["assets"][3][
            "path"] = "outputs/pesn/v53mp/v53mp_20k_C_s17/checkpoint_last.pt"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert "V53MP_C_S17" in exc.value.detail
    assert "37,900" in exc.value.detail  # V541 tier wording
    assert "15,400" in exc.value.detail  # V53MP tier wording
    assert "non-pinned" in exc.value.detail


# ---------------------------------------------------------------------------
# forward view
# ---------------------------------------------------------------------------
def test_illegal_forward_key_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["forward_view_keys"].append("y_teacher")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_FORWARD_KEY"
    assert "y_teacher" in exc.value.detail


def test_forbidden_input_class_as_forward_key_rejected(base_cfg):
    def _bad(c):
        c["backbone_contract"]["forward_view_keys"].append("ts_coords")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_FORWARD_KEY"


def test_empty_forward_keys_rejected(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["backbone_contract"].update(forward_view_keys=[]),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"


# ---------------------------------------------------------------------------
# authorization / resources
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["confirm_authorized", "sealed_authorized"])
def test_confirm_or_sealed_authorization_rejected(base_cfg, flag):
    cfg = _mut(
        base_cfg,
        lambda c, f=flag: c["governance"]["authorization"].update(**{f: True}),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_CONFIRM"


def test_nonzero_qc_calls_rejected(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["governance"]["authorization"].update(new_qc_calls=3),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_CONFIRM"


def test_single_lane_required(base_cfg):
    cfg = _mut(
        base_cfg, lambda c: c["governance"]["authorization"].update(single_lane=False)
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RESOURCE"


def test_swap_forbidden(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["budget"].update(swap_max=4096))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RESOURCE"


def test_oom_kill_forbidden(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["budget"].update(oom_kill_max=1))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RESOURCE"


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
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_THRESHOLD_NULL"
    assert key in exc.value.detail


def test_non_numeric_threshold_rejected(base_cfg):
    cfg = _mut(
        base_cfg, lambda c: c["thresholds"].update(max_core_time_ratio="fast")
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_THRESHOLD_NULL"


def test_unfrozen_signoff_blocks_formal(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["thresholds"]["signoff"].update(status="proposed"),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_THRESHOLD_NULL"


def test_signoff_mirror_mismatch_blocks_formal(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["thresholds"]["signoff"].update(max_core_time_ratio=3.0),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_THRESHOLD_NULL"
    assert "max_core_time_ratio" in exc.value.detail


def test_formal_thresholds_match_signoff_block(base_cfg):
    cfg = base_cfg
    signoff = cfg["thresholds"]["signoff"]
    assert len(v.FORMAL_THRESHOLDS) == 28  # 24 originals + 4 v1.1 keys
    for key in v.FORMAL_THRESHOLDS:
        assert cfg["thresholds"][key] == signoff[key], key
    assert signoff["status"] == "frozen"
    assert signoff["signed_on"] == "2026-10-29"


def test_h2_h3_cost_constants_frozen_in_config(base_cfg):
    th = base_cfg["thresholds"]
    assert th["sign_drop_abs_max"] == 0.01
    assert th["dead_fraction_increase_abs_max"] == 0.01
    assert th["fixed_coverage_direction_regression_rel_max_pct"] == 2.0
    assert th["rmse_zero_regression_rel_max_pct"] == 2.0
    assert th["h3_move_direction_rel_min_pct"] == 3.0
    assert th["h3_move_direction_abs_min"] == 0.01
    assert th["h3_move_rmse_rel_min_pct"] == 3.0
    assert th["h3_process_delta_auprc_abs_min"] == 0.02
    assert th["h3_process_delta_q_corr_abs_min"] == 0.03
    assert th["move_exit_alpha_each"] == 0.025
    assert th["final_bootstrap_draws"] == 10000
    assert th["bootstrap_seed_root"] == 20261029
    assert th["top2_max_runs"] == 4
    assert th["a5_supplement_max_runs"] == 2
    assert th["t3_max_runs"] == 6
    assert th["max_core_time_ratio"] == 2.0
    assert th["diagnostic_loss_drop_min_pct"] == 5.0
    assert th["diagnostic_dead_fraction_margin_vs_a0"] == 0.20
    assert th["diagnostic_rmse_zero_rel_max_pct"] == 10.0


def test_null_budget_blocks_formal_typed_but_prereg_passes(base_cfg):
    """The pre-profile state is still tested on an in-memory copy."""
    cfg = _mut(base_cfg, lambda c: c["budget"].__setitem__(
        "per_candidate_core_time", None))
    report = v.validate_config(cfg, formal=False)
    assert report["ok"] and report["t1_config_ready"] is False
    assert any("per_candidate_core_time" in r for r in report["t1_blocked"])
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_NULL"


def test_filled_budget_requires_attribution_entries(base_cfg):
    """A1-C/A5-C are full-dose diagnostics and need their own budget entries."""
    cfg = _mut(base_cfg, lambda c: _fill_budget(c, include_attr=False))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_INVALID"
    assert "A1-C" in exc.value.detail and "A5-C" in exc.value.detail


# ---------------------------------------------------------------------------
# gate-3 C4/A1 (2026-10-30, P3 remediation lane R2): the budget formal
# gate is closed — a filled table must be a mapping covering every
# KNOWN_BACKBONES entry with a finite POSITIVE core time.  Oracle gate-3
# probes (list/zeros/neg/nan used to PASS --formal) are now typed
# V542_BUDGET_INVALID rejections; only null/empty keeps V542_BUDGET_NULL.
# ---------------------------------------------------------------------------
def _budget_table(cfg: dict) -> dict:
    return cfg["budget"]["per_candidate_core_time"]


def _full_budget_ids() -> list:
    return ["A0", "A1", "A2", "A3", "A4", "A5", "A1-C", "A5-C"]


def test_formal_empty_list_budget_is_typed_null(base_cfg):
    """[] is an empty (unsigned) table => V542_BUDGET_NULL, not INVALID."""
    def _bad(c):
        c["budget"]["per_candidate_core_time"] = []

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_NULL"


def test_formal_empty_mapping_budget_is_typed_null(base_cfg):
    def _bad(c):
        c["budget"]["per_candidate_core_time"] = {}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_NULL"


@pytest.mark.parametrize("bad_table", ["3600", 3600, ["A0"], True])
def test_formal_non_mapping_budget_typed_invalid(base_cfg, bad_table):
    """A string/number/list/bool table is not a signed budget table."""
    def _bad(c, table=bad_table):
        c["budget"]["per_candidate_core_time"] = table

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_INVALID"


def test_formal_all_null_budget_entries_typed_invalid(base_cfg):
    """Full key coverage but every value null = malformed (gate-3 C4)."""
    def _bad(c):
        c["budget"]["per_candidate_core_time"] = {
            cid: None for cid in _full_budget_ids()}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_INVALID"


@pytest.mark.parametrize("bad_value", [0, 0.0, -1, -2500.5])
def test_formal_non_positive_budget_typed_invalid(base_cfg, bad_value):
    """Zero and negative core times are typed rejections (gate-3 C4)."""
    def _bad(c, value=bad_value):
        c["budget"]["per_candidate_core_time"] = {
            cid: {"core_seconds": value, "status": "approved"}
            for cid in _full_budget_ids()}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_INVALID"
    assert "finite POSITIVE" in exc.value.detail


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"),
                                       float("-inf")])
def test_formal_non_finite_budget_typed_invalid(base_cfg, bad_value):
    """NaN/inf core times are typed rejections (gate-3 C4)."""
    def _bad(c, value=bad_value):
        c["budget"]["per_candidate_core_time"] = {
            cid: value for cid in _full_budget_ids()}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_INVALID"


@pytest.mark.parametrize("bad_value", [True, False, "3600", None])
def test_formal_bool_str_null_entry_typed_invalid(base_cfg, bad_value):
    """bool/string/null entries are typed rejections (gate-3 C4)."""
    def _bad(c, value=bad_value):
        c["budget"]["per_candidate_core_time"] = {
            cid: value for cid in _full_budget_ids()}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_INVALID"


def test_formal_partial_budget_table_typed_invalid(base_cfg):
    """Partial key coverage = V542_BUDGET_INVALID (distinct from NULL)."""
    def _bad(c):
        c["budget"]["per_candidate_core_time"] = {
            cid: {"core_seconds": 2000.0} for cid in ("A0", "A1")}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_INVALID"
    assert "A5-C" in exc.value.detail


def test_formal_extra_key_budget_table_typed_invalid(base_cfg):
    """Extra candidate keys = V542_BUDGET_INVALID (gate-3 C4)."""
    def _bad(c):
        _fill_budget(c)
        table = _budget_table(c)
        table["A9-rogue"] = {"core_seconds": 2000.0}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True)
    assert exc.value.code == "V542_BUDGET_INVALID"
    assert "A9-rogue" in exc.value.detail


def test_formal_bare_number_budget_entries_accepted(base_cfg, tmp_path):
    """A candidate -> bare finite positive number IS a valid core-time
    entry (the spec requires a finite positive numeric value per entry;
    only list/string/None/zero/negative/NaN/bool/partial/extra-key
    tables are rejected)."""
    def _table(c):
        c["budget"]["per_candidate_core_time"] = {
            cid: 2000.0 for cid in _full_budget_ids()}

    cfg = _mut(base_cfg, _table)
    _write_valid_records(tmp_path, cfg)
    formal = v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert formal["ok"] and formal["t1_blocked"] == []


def test_prereg_budget_shape_not_required(base_cfg):
    """prereg never requires numbers: malformed tables stay green (the
    formal gate is where the shape is enforced)."""
    def _bad(c):
        c["budget"]["per_candidate_core_time"] = {
            cid: None for cid in _full_budget_ids()}

    cfg = _mut(base_cfg, _bad)
    report = v.validate_config(cfg, formal=False)
    assert report["ok"] is True
    assert report["null_thresholds"] == []


def test_correct_budget_table_formal_passes(base_cfg, tmp_path):
    """A well-formed signed table + valid records = formal config-ready."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    formal = v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert formal["ok"] and formal["t1_config_ready"] is True
    assert formal["t1_blocked"] == []


def test_h2_rule_must_keep_degradation_word(base_cfg):
    """H2 is a non-inferiority guard — the 退化 (degradation) word is frozen."""
    def _bad(c):
        c["gates"]["H2"]["rule"] = c["gates"]["H2"]["rule"].replace("退化", "变化")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "退化" in exc.value.detail


# ---------------------------------------------------------------------------
# G–V registry cross-check (V5.4.2 registered by the main session)
# ---------------------------------------------------------------------------
def test_gv_campaign_mismatch_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["gv"].update(campaign_id="V5.4.1"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_GV_MISMATCH"


def test_gv_primary_g_drift_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["gv"].update(primary_g="G3.P.0"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_GV_MISMATCH"


def test_gv_gate_drift_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["gv"]["gates"].append("V542-rogue-gate"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_GV_MISMATCH"


def test_gv_check_can_be_skipped_intentionally(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["gv"].update(map_version="stale"))
    assert v.validate_config(cfg, check_gv=False)["ok"]


# ---------------------------------------------------------------------------
# exact 26-run matrix
# ---------------------------------------------------------------------------
def test_run_matrix_is_exact_26_runs(base_cfg):
    cfg = base_cfg
    runs = cfg["run_matrix"]["runs"]
    assert len(runs) == 26 == v.EXPECTED_RUN_COUNT
    assert {r["run_id"] for r in runs} == set(v.EXPECTED_RUN_IDS)
    for run in runs:
        expected = v.EXPECTED_RUN_TABLE[run["run_id"]]
        for field, want in expected.items():
            if isinstance(want, list):
                assert list(run.get(field) or ()) == want, (run["run_id"], field)
            else:
                assert run.get(field) == want, (run["run_id"], field)


def test_run_id_must_use_v542_namespace(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][3]["run_id"] = "V541_A1_S17"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V542_" in exc.value.detail


def test_run_seed_must_be_preregistered(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][3]["seed"] = 7

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_duplicate_run_id_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][4]["run_id"] = c["run_matrix"]["runs"][3]["run_id"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


@pytest.mark.parametrize(
    "drop",
    ["V542_A1_S17", "V542_A5_S29", "V542_DIAG_A3_S17", "V542_A5C_S17",
     "V542_A0_S43"],
)
def test_missing_run_rejected(base_cfg, drop):
    def _bad(c, run_id=drop):
        c["run_matrix"]["runs"] = [
            r for r in c["run_matrix"]["runs"] if r["run_id"] != run_id
        ]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    # missing run named explicitly for non-A0 rows; A0 rows hit the
    # matched-retrain set check whose detail lists the registered A0 ids
    assert drop in exc.value.detail or "A0 retrain" in exc.value.detail


def test_extra_run_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"].append(
            {
                "run_id": "V542_A6_S17",
                "stage": "T1",
                "backbone": "A1",
                "seed": 17,
                "class": "nominal",
                "new_training": 1,
                "winner_eligible": True,
                "dose": "formal",
            }
        )

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V542_A6_S17" in exc.value.detail


def test_renamed_run_rejected(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_DIAG_A2_S17")["run_id"] = "V542_DIAG2_A2_S17"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V542_DIAG2_A2_S17" in exc.value.detail


def test_stage_drift_rejected(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A3_S17")["stage"] = "T3"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "stage" in exc.value.detail


def test_dose_drift_rejected(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A2_S17")["dose"] = "diagnostic_2000"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "dose" in exc.value.detail


def test_diagnostic_run_cannot_be_winner_eligible(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["class"] == "diagnostic":
                run["winner_eligible"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_a0_baseline_cannot_be_winner_eligible(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["backbone"] == "A0" and run["run_id"] == "V542_A0_S17":
                run["winner_eligible"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "baseline" in exc.value.detail


def test_a0_formal_runs_must_declare_baseline_role(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["run_id"] == "V542_A0_S17":
                run.pop("role")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "matched_baseline_retrain" in exc.value.detail


def test_a0_diagnostic_must_declare_dose_reference_role(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["run_id"] == "V542_DIAG_A0_S17":
                run.pop("role")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "matched_dose_reference" in exc.value.detail


def test_run_classes_must_be_exact_three_way_split(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["run_matrix"].update(classes=["nominal"]))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_unknown_backbone_in_run_matrix_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][3]["backbone"] = "A6"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"


def test_null_backbone_identity_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"][3]["backbone"] = None

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_BACKBONE_UNKNOWN"
    assert "deterministic backbone identity" in exc.value.detail


def test_diagnostic_only_backbone_cannot_run_as_nominal(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["run_id"] == "V542_A1C_S17":
                run["class"] = "nominal"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "diagnostic-only" in exc.value.detail


def test_a5_full_dose_condition_frozen(base_cfg):
    def _bad(c):
        for run in c["run_matrix"]["runs"]:
            if run["run_id"] == "V542_A5_S17":
                run["condition"] = "always_launch"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "no_A1_A4_h3_qualified_candidate_after_top2_seeds" in exc.value.detail


def test_a5_full_dose_launch_requires_frozen(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A5_S17")["launch_requires"] = ["diagnostic_gate"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "cost_gate" in exc.value.detail
    assert "H0_legality" in exc.value.detail


def test_a5_activation_block_launch_requires_frozen(base_cfg):
    def _bad(c):
        c["run_matrix"]["a5_activation"]["launch_requires"] = [
            "diagnostic_gate", "cost_gate"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "H0_legality" in exc.value.detail


def test_a5_activation_must_not_gate_on_final_h3(base_cfg):
    """No A1-A4 final-H3 qualification may sit inside A5's launch gates."""
    def _bad(c):
        c["run_matrix"]["a5_activation"]["launch_requires"] = [
            "diagnostic_gate", "cost_gate", "H0_legality", "H1_resource",
            "A1_A4_final_h3_qualified"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "A1_A4_final_h3_qualified" in exc.value.detail


def test_a5_activation_diagnostic_survival_required(base_cfg):
    def _bad(c):
        c["run_matrix"]["a5_activation"]["diagnostic_survival_required"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_run_outcome_vocabulary_frozen_six_words(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["run_outcomes"].update(vocabulary=["done", "failed"]),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_legacy_ledger_status_machine_must_stay_unchanged(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["run_outcomes"].update(ledger_status_machine="extended"),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


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
    legacy_only = set(KNOWN_STATUSES) - {"done"}
    assert not legacy_only & set(ro_vocab["vocabulary"])


# ---------------------------------------------------------------------------
# top-2 / A5-supplement split — A1-A4 only, A5 separate branch
# ---------------------------------------------------------------------------
def test_top2_preregisters_a1_a4_cross_product_only(base_cfg):
    cfg = base_cfg
    top2 = [r for r in cfg["run_matrix"]["runs"]
            if r.get("condition") == "activated_only_by_inner_val_top2"]
    assert {r["run_id"] for r in top2} == _top2_ids()
    assert len(top2) == 8
    for run in top2:
        assert run["backbone"] in ("A1", "A2", "A3", "A4")
        assert run["class"] == "conditional"
        assert run["winner_eligible"] is True
        assert run["unactivated_status"] == "gate_skipped"
        assert run["new_training_if_activated"] == 1
        assert "new_training" not in run
        assert "parent_gate" not in run


def test_a5_supplement_rows_separate_from_top2(base_cfg):
    cfg = base_cfg
    sup = [r for r in cfg["run_matrix"]["runs"]
           if r["run_id"] in _a5_supplement_ids()]
    assert len(sup) == 2
    for run in sup:
        assert run["stage"] == "T3"
        assert run["backbone"] == "A5"
        assert run["class"] == "conditional"
        assert run["winner_eligible"] is True
        assert run["condition"] == v.A5_SUPPLEMENT_CONDITION
        assert run["parent_gate"] == v.A5_SUPPLEMENT_PARENT_GATE
        assert run["unactivated_status"] == "gate_skipped"
        assert run["new_training_if_activated"] == 1
    t3 = [r for r in cfg["run_matrix"]["runs"] if r["stage"] == "T3"]
    assert len(t3) == 10  # 8 top-2 + 2 a5_supplement rows (activation <= 6)


def test_top2_block_declares_v11_split(base_cfg):
    block = base_cfg["run_matrix"]["top2_preregistration"]
    assert block["candidates"] == ["A1", "A2", "A3", "A4"]
    assert block["seeds"] == [29, 43]
    assert block["listed_runs"] == 8
    assert block["top2_max_runs"] == 4
    assert block["a5_supplement_max_runs"] == 2
    assert block["t3_max_runs"] == 6
    assert block["activation"] == "activated_only_by_inner_val_top2"
    assert block["unactivated_status"] == "gate_skipped"
    assert block["n_core_counting"] == "formula_only"
    assert block["seed17_screen_pass_ref"] == v.SCREEN_PREDICATE_REF
    assert "A0_S17" in block["freeze_requires"]
    assert "full_activation_count" not in block  # superseded v1.0 field
    sup = base_cfg["run_matrix"]["a5_supplement"]
    assert sup["runs"] == ["V542_A5_S29", "V542_A5_S43"]
    assert sup["a5_supplement_max_runs"] == 2
    assert sup["condition"] == v.A5_SUPPLEMENT_CONDITION
    assert sup["parent_gate"] == v.A5_SUPPLEMENT_PARENT_GATE


@pytest.mark.parametrize("drop", ["V542_A1_S29", "V542_A3_S43", "V542_A4_S29"])
def test_top2_missing_candidate_rejected(base_cfg, drop):
    def _bad(c):
        c["run_matrix"]["runs"] = [
            r for r in c["run_matrix"]["runs"] if r["run_id"] != drop
        ]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert drop in exc.value.detail


def test_top2_extra_candidate_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["runs"].append(
            {
                "run_id": "V542_A1_S17",
                "stage": "T3",
                "backbone": "A1",
                "seed": 17,
                "class": "conditional",
                "dose": "formal",
                "new_training_if_activated": 1,
                "unactivated_status": "gate_skipped",
                "winner_eligible": True,
                "condition": "activated_only_by_inner_val_top2",
            }
        )

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_top2_placeholder_token_rejected(base_cfg):
    def _bad(c):
        runs = c["run_matrix"]["runs"]
        for i, run in enumerate(runs):
            if run["stage"] == "T3" and run["backbone"] == "A1":
                runs[i] = {
                    **run,
                    "run_id": f"V542_T3_TOP2_S{run['seed']}",
                    "backbone": None,
                }
                break

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "TOP2" in exc.value.detail


def test_top2_condition_must_gate_on_inner_val_top2(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A2_S29")["condition"] = "always_launch"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "activated_only_by_inner_val_top2" in exc.value.detail


def test_top2_unactivated_must_be_gate_skipped(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A3_S43")["unactivated_status"] = "failed"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "gate_skipped" in exc.value.detail


def test_top2_must_not_claim_unconditional_new_training(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A1_S29")["new_training"] = 1

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "new_training" in exc.value.detail


def test_a5_supplement_rows_require_parent_gate(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A5_S29").pop("parent_gate")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "parent_gate" in exc.value.detail


def test_a5_supplement_old_h3_pass_parent_gate_rejected(base_cfg):
    """The v1.0 parent gate 'a5_full_dose_activated_and_seed17_h3_pass' is gone."""
    def _bad(c):
        _run_by_id(c, "V542_A5_S43")["parent_gate"] = (
            "a5_full_dose_activated_and_seed17_h3_pass")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert v.A5_SUPPLEMENT_PARENT_GATE in exc.value.detail


def test_a5_supplement_block_old_parent_gate_rejected(base_cfg):
    def _bad(c):
        c["run_matrix"]["a5_supplement"]["parent_gate"] = (
            "a5_full_dose_activated_and_seed17_h3_pass")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "seed17 SCREEN" in exc.value.detail


def test_top2_parent_gate_not_allowed_on_non_a5(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A2_S29")["parent_gate"] = v.A5_SUPPLEMENT_PARENT_GATE

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "only valid on A5" in exc.value.detail


def test_a5_must_not_enter_top2_candidates(base_cfg):
    def _bad(c):
        c["run_matrix"]["top2_preregistration"]["candidates"] = [
            "A1", "A2", "A3", "A4", "A5"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "NOT include A5" in exc.value.detail


def test_a5_row_with_top2_condition_rejected(base_cfg):
    """Moving an A5 T3 row under the top-2 condition is a rejection."""
    def _bad(c):
        _run_by_id(c, "V542_A5_S29")["condition"] = "activated_only_by_inner_val_top2"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


@pytest.mark.parametrize(
    "field,value",
    [
        ("listed_runs", 10),
        ("top2_max_runs", 10),
        ("a5_supplement_max_runs", 4),
        ("t3_max_runs", 4),
        ("activation", "launch_all"),
        ("unactivated_status", "failed"),
        ("n_core_counting", "sum_listed_runs"),
        ("seed17_screen_pass_ref", "evaluation_contract.seed17_h3"),
        ("freeze_requires", "inner-val only"),
        ("full_activation_count", 4),  # superseded v1.0 key must not return
    ],
)
def test_top2_block_invariants_enforced(base_cfg, field, value):
    def _bad(c, f=field, val=value):
        c["run_matrix"]["top2_preregistration"][f] = val

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_top2_block_missing_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["run_matrix"].pop("top2_preregistration"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "top2_preregistration" in exc.value.detail


def test_a5_supplement_block_missing_rejected(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["run_matrix"].pop("a5_supplement"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a5_supplement" in exc.value.detail


def test_threshold_t3_max_must_match_split(base_cfg):
    def _bad(c):
        c["thresholds"]["t3_max_runs"] = 4
        c["thresholds"]["signoff"]["t3_max_runs"] = 4

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "t3_max_runs" in exc.value.detail


# ---------------------------------------------------------------------------
# A1-C / A5-C activation branches
# ---------------------------------------------------------------------------
def test_a1c_activation_rule_frozen(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A1C_S17")["condition"] = "a1_top_ranked_only"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a1_in_top2_supplement_set_OR_a1_selected_as_backbone" in exc.value.detail


def test_a1c_skip_reason_frozen(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A1C_S17")["skip_reason"] = "no_A1_promotion"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert v.A1C_SKIP_REASON in exc.value.detail


def test_a5c_activation_rule_frozen(base_cfg):
    def _bad(c):
        _run_by_id(c, "V542_A5C_S17")["condition"] = "always_launch"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a5_selected_backbone" in exc.value.detail


def test_a5c_old_promoted_after_h3_predicate_rejected(base_cfg):
    """The v1.0 'a5_promoted_after_h3' predicate is superseded."""
    def _bad(c):
        _run_by_id(c, "V542_A5C_S17")["condition"] = "a5_promoted_after_h3"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a5_promoted_after_h3" in exc.value.detail


def test_a5c_block_condition_frozen(base_cfg):
    def _bad(c):
        c["run_matrix"]["a5c_activation"]["condition"] = "a5_h3_qualified"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a5_selected_backbone" in exc.value.detail


def test_a5c_skip_reason_frozen(base_cfg):
    def _bad(c):
        c["run_matrix"]["a5c_activation"]["skip_reason"] = "no_A5_promotion"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert v.A5C_SKIP_REASON in exc.value.detail


@pytest.mark.parametrize("run_id", ["V542_A1C_S17", "V542_A5C_S17"])
@pytest.mark.parametrize(
    "missing", ["budget_record", "cost_gate", "profile_40_40", "smoke_200_conditional"]
)
def test_attr_activation_requires_budget_and_cost_gates(base_cfg, run_id, missing):
    def _bad(c, rid=run_id, drop=missing):
        _run_by_id(c, rid)["activation_requires"] = [
            item for item in ATTR_REQUIRES_BASE if item != drop]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


ATTR_REQUIRES_BASE = ["H0_fixture", "smoke_200_conditional", "profile_40_40",
                      "cost_gate", "budget_record"]


def test_a5_diagnostic_must_stay_unconditional(base_cfg):
    """A5 smoke/profile/diagnostic always complete before activation decisions."""
    def _bad(c):
        _run_by_id(c, "V542_DIAG_A5_S17")["condition"] = "only_if_activated"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "unconditional" in exc.value.detail


def test_a5_activation_requires_no_discretion(base_cfg):
    def _bad(c):
        c["run_matrix"]["a5_activation"]["no_discretion"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "no_discretion" in exc.value.detail


def test_a1c_not_only_top_ranked_required(base_cfg):
    def _bad(c):
        c["run_matrix"]["a1c_activation"]["not_only_top_ranked"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "not_only_top_ranked" in exc.value.detail


# ---------------------------------------------------------------------------
# seed17_screen_pass predicate (screen, never a three-seed H3 claim)
# ---------------------------------------------------------------------------
def test_seed17_screen_predicate_fields_frozen(base_cfg):
    pred = base_cfg["evaluation_contract"]["seed17_screen_pass_predicate"]
    assert pred["definition_name"] == "seed17_screen_pass"
    assert "H3" in pred["scope"]
    assert tuple(pred["hard_gates_all_pass"]) == (
        "H0_legality", "H1_resource", "diagnostic_gate", "cost_gate")
    assert "A0_S17" in pred["matched_a0_dependency"]
    assert "退化" in pred["h2_no_breach"]
    assert "process_route" in pred["point_effect_requirement"]
    assert "move_route" in pred["point_effect_requirement"]
    assert "A0_S17" in pred["top2_freeze_requires"]
    assert "S29" in pred["h3_final_requires"] and "S43" in pred["h3_final_requires"]


def test_seed17_screen_predicate_missing_rejected(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["evaluation_contract"].pop("seed17_screen_pass_predicate"),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "seed17_screen_pass_predicate" in exc.value.detail


def test_seed17_scope_must_disclaim_h3(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"]["scope"] = (
            "仅用于 top2 eligibility 与 A5 supplement 激活")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "H3" in exc.value.detail


def test_seed17_hard_gates_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "hard_gates_all_pass"] = ["H0_legality", "H1_resource"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "cost_gate" in exc.value.detail


def test_seed17_matched_a0_dependency_required(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "matched_a0_dependency"] = "matched A0 任意 seed 已完成"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "A0_S17" in exc.value.detail


def test_seed17_h2_no_breach_must_keep_degradation_word(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"]["h2_no_breach"] = (
            "seed17 H2 全部 guard 无 breach (RMSE/zero 相对 R3 与 matched A0 <=2%)")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "退化" in exc.value.detail


def test_seed17_point_effect_routes_required(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "point_effect_requirement"] = {"rule": "at least one route"}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "process_route" in exc.value.detail


def test_h3_final_requires_matched_a0_seed_completion(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "h3_final_requires"] = "matched A0 对应 seed complete"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "S29" in exc.value.detail and "S43" in exc.value.detail


# ---------------------------------------------------------------------------
# H3 statistics contract
# ---------------------------------------------------------------------------
def test_statistics_contract_frozen_in_config(base_cfg):
    stats = base_cfg["evaluation_contract"]["statistics"]
    assert stats["bootstrap_seed_root"] == 20261029 == v.BOOTSTRAP_SEED_ROOT
    assert stats["bootstrap_subseed_formula"] == v.BOOTSTRAP_SUBSEED_FORMULA
    assert stats["bootstrap_unit"] == "curve_reaction_cluster"
    assert stats["initial_draws"] == 1000
    assert stats["final_draws"] == 10000 == v.FINAL_DRAWS
    assert stats["ci_and_p_same_draws"] is True
    assert stats["frame_level_bootstrap_forbidden"] is True
    p_rule = stats["p_value_rule"]
    assert p_rule["type"] == v.P_VALUE_TYPE
    assert "(10001)" in p_rule["formula"]
    assert p_rule["draws"] == 10000
    assert p_rule["frame_level_p_forbidden"] is True
    assert p_rule["improvement_direction"]["rmse_zero"] == "a0_minus_candidate"
    holm = stats["holm_family"]
    for exit_name in ("move_direction_exit", "move_rmse_exit"):
        assert holm[exit_name]["alpha"] == 0.025
        assert holm[exit_name]["correction"] == "holm"
        assert holm[exit_name]["scope"] == "eligible_candidate_set_within_exit"
    assert holm["process_route"]["alpha"] == 0.025
    assert holm["process_route"]["family"] == "single_process_family"
    assert tuple(stats["h3_eligible_candidates"]["criteria"]) == (
        v.H3_ELIGIBLE_CRITERIA)
    assert stats["h3_eligible_candidates"]["a5_rule"] == v.H3_ELIGIBLE_A5_RULE
    assert tuple(stats["h3_eligible_candidates"]["excluded"]) == (
        v.H3_ELIGIBLE_EXCLUDED)
    assert stats["champion_rule"]["requires"] == v.CHAMPION_REQUIRES
    assert stats["champion_rule"]["selection_ref"] == (
        v.CHAMPION_SELECTION_REF)
    assert stats["champion_rule"]["report"] == v.CHAMPION_REPORT
    agg = base_cfg["evaluation_contract"]["seed_aggregation"]
    assert agg["min_seeds_same_direction"] == 2
    assert agg["total_seeds"] == 3
    assert isinstance(agg["min_seeds_same_direction"], int)
    assert isinstance(agg["total_seeds"], int)


def test_statistics_bootstrap_seed_root_drift_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["bootstrap_seed_root"] = 1

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "bootstrap_seed_root" in exc.value.detail


def test_statistics_subseed_formula_drift_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["bootstrap_subseed_formula"] = (
            "random.seed(run_seed)")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "sub-seed" in exc.value.detail


def test_statistics_missing_p_value_rule_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"].pop("p_value_rule")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "p_value_rule" in exc.value.detail


def test_statistics_p_formula_denominator_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["p_value_rule"]["formula"] = (
            "p = #{improvement <= 0} / 10000")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "(10001)" in exc.value.detail


def test_statistics_p_draws_must_match_ci(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["p_value_rule"]["draws"] = 1000

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "draws" in exc.value.detail


def test_statistics_final_draws_drift_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["final_draws"] = 1000

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "final" in exc.value.detail


def test_statistics_ci_p_same_draws_required(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["ci_and_p_same_draws"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "ci_and_p_same_draws" in exc.value.detail


@pytest.mark.parametrize("field", ["frame_level_bootstrap_forbidden"])
def test_statistics_frame_level_bootstrap_forbidden(base_cfg, field):
    def _bad(c, f=field):
        c["evaluation_contract"]["statistics"][f] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_statistics_frame_level_p_forbidden(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["p_value_rule"][
            "frame_level_p_forbidden"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_statistics_improvement_direction_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["p_value_rule"][
            "improvement_direction"]["rmse_zero"] = "candidate_minus_a0"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a0_minus_candidate" in exc.value.detail


@pytest.mark.parametrize(
    "family", ["move_direction_exit", "move_rmse_exit", "process_route"])
def test_statistics_holm_family_frozen(base_cfg, family):
    def _bad(c, fam=family):
        c["evaluation_contract"]["statistics"]["holm_family"][fam]["alpha"] = 0.05

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "holm_family" in exc.value.detail


def test_statistics_holm_scope_within_exit_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["holm_family"][
            "move_direction_exit"]["scope"] = "within_candidate_across_exits"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "eligible_candidate_set_within_exit" in exc.value.detail


def test_statistics_process_family_must_be_single(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["holm_family"]["process_route"][
            "family"] = "per_endpoint_families"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "single_process_family" in exc.value.detail


def test_statistics_missing_holm_family_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"].pop("holm_family")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "holm_family" in exc.value.detail


def test_statistics_eligible_excludes_diagnostic_variants(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["h3_eligible_candidates"][
            "excluded"] = ["A0_matched_baseline"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "diagnostic_variants_A1-C_A5-C" in exc.value.detail


def test_statistics_eligible_free_string_form_rejected(base_cfg):
    """gate-0 A2: the old free-string eligibility form is no longer valid."""
    def _bad(c):
        c["evaluation_contract"]["statistics"]["h3_eligible_candidates"] = (
            "H2 diagnostic 三 seed winner-eligible; NOT three seed")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "h3_eligible_candidates" in exc.value.detail


def test_statistics_eligible_criteria_mismatch_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["h3_eligible_candidates"][
            "criteria"] = ["completed_three_seeds", "winner_eligible"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "per_seed_h2_no_breach" in exc.value.detail


def test_statistics_eligible_a5_rule_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["h3_eligible_candidates"][
            "a5_rule"] = "a5_eligible_on_seed17_only"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a5_only_when_three_seeds_complete" in exc.value.detail


def test_champion_rule_requires_adjusted_p(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["champion_rule"]["requires"] = (
            "effect_non_inferiority_conditions_only")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "holm_adjusted_p_below_alpha" in exc.value.detail


def test_champion_rule_free_prose_form_rejected(base_cfg):
    """gate-0 A2: the old token-checked champion prose is no longer valid."""
    def _bad(c):
        c["evaluation_contract"]["statistics"]["champion_rule"] = (
            "冠军须 adjusted p < 对应 alpha 且效应/非劣条件成立 (qualification)")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "champion_rule" in exc.value.detail


def test_min_seeds_same_direction_must_be_numeric(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed_aggregation"]["min_seeds_same_direction"] = (
            "2/3 seeds")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "NUMERIC" in exc.value.detail


def test_h3_gate_rule_must_carry_stats_tokens(base_cfg):
    def _bad(c):
        c["gates"]["H3"]["rule"] = c["gates"]["H3"]["rule"].replace("20261029", "1")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "20261029" in exc.value.detail


# ---------------------------------------------------------------------------
# fix-6 C6: statistics exact structured fields
# ---------------------------------------------------------------------------
def test_statistics_ci_frozen_percentile_95(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["ci"] = "bca"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "percentile_95" in exc.value.detail


def test_statistics_pairing_sentence_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["pairing"] = (
            "independent bootstrap per model")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "resample indices" in exc.value.detail


def test_statistics_final_draws_ref_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["final_draws_ref"] = "hardcoded"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "thresholds.final_bootstrap_draws" in exc.value.detail


def test_statistics_final_draws_must_match_signed_threshold(base_cfg):
    def _bad(c):
        c["thresholds"]["final_bootstrap_draws"] = 5000
        c["thresholds"]["signoff"]["final_bootstrap_draws"] = 5000

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code in ("V542_THRESHOLD_NULL", "V542_RUN_MATRIX")
    assert "5000" in exc.value.detail


@pytest.mark.parametrize("endpoint", ["direction", "endpoint_auprc", "q_corr"])
def test_statistics_improvement_direction_all_four_frozen(
        base_cfg, endpoint):
    """fix-6: every one of the four improvement directions is pinned."""
    def _bad(c, ep=endpoint):
        c["evaluation_contract"]["statistics"]["p_value_rule"][
            "improvement_direction"][ep] = "a0_minus_candidate"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "improvement_direction" in exc.value.detail
    assert "candidate_minus_a0" in exc.value.detail


def test_statistics_improvement_direction_exact_key_set(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["p_value_rule"][
            "improvement_direction"].pop("q_corr")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "improvement_direction" in exc.value.detail


def test_statistics_p_formula_numerator_exact(base_cfg):
    """fix-6: the p numerator/denominator string is frozen verbatim."""
    def _bad(c):
        c["evaluation_contract"]["statistics"]["p_value_rule"]["formula"] = (
            "p = (2 + #{seed-averaged paired bootstrap improvement <= 0}) "
            "/ (10001)")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert v.P_FORMULA_EXACT in exc.value.detail


def test_statistics_champion_rule_references_final_selection(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["champion_rule"][
            "selection_ref"] = "evaluation_contract.seed_aggregation"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "final_selection_rule" in exc.value.detail


def test_statistics_champion_rule_requires_uncorrected_report(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["statistics"]["champion_rule"]["report"] = (
            "holm_adjusted_only")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "uncorrected_and_holm_adjusted" in exc.value.detail


def test_statistics_eligible_requires_winner_eligible(base_cfg):
    def _bad(c):
        elig = c["evaluation_contract"]["statistics"]["h3_eligible_candidates"]
        elig["criteria"] = ["completed_three_seeds", "per_seed_h2_no_breach"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "winner_eligible" in exc.value.detail


def test_statistics_contract_exact_values_in_config(base_cfg):
    stats = base_cfg["evaluation_contract"]["statistics"]
    assert stats["ci"] == v.CI_METHOD == "percentile_95"
    assert stats["pairing"] == v.PAIRING_SENTENCE
    assert stats["final_draws_ref"] == v.FINAL_DRAWS_REF
    assert stats["final_draws"] == 10000 == v.FINAL_DRAWS
    assert stats["final_draws"] == base_cfg["thresholds"]["final_bootstrap_draws"]
    assert dict(stats["p_value_rule"]["improvement_direction"]) == (
        v.IMPROVEMENT_DIRECTIONS)
    assert stats["p_value_rule"]["formula"] == v.P_FORMULA_EXACT
    assert stats["p_value_rule"]["draws"] == stats["final_draws"]
    assert stats["ci_and_p_same_draws"] is True
    assert stats["frame_level_bootstrap_forbidden"] is True
    assert stats["p_value_rule"]["frame_level_p_forbidden"] is True


# ---------------------------------------------------------------------------
# fix-6 C2: structured parenthesized route logic
# ---------------------------------------------------------------------------
def test_route_logic_frozen_in_config(base_cfg):
    rl = base_cfg["evaluation_contract"]["route_logic"]
    assert rl["move_route"]["pass_logic"] == v.MOVE_PASS_LOGIC
    assert rl["process_route"]["pass_logic"] == v.PROCESS_PASS_LOGIC
    for route, comps in v.ROUTE_COMPONENTS.items():
        for comp, want in comps.items():
            assert rl[route][comp] == want, (route, comp)
    assert rl["seed17_screen_reuses_same_parenthesized_logic"] is True


def test_route_logic_missing_block_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"].pop("route_logic")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "route_logic" in exc.value.detail


@pytest.mark.parametrize("route", ["move_route", "process_route"])
def test_route_logic_pass_logic_drift_rejected(base_cfg, route):
    """fix-6 C2: dropping a paren branch (dangling OR) is a typed reject."""
    def _bad(c, r=route):
        c["evaluation_contract"]["route_logic"][r]["pass_logic"] = (
            "direction_pass AND rmse_non_inferior OR rmse_pass")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "pass_logic" in exc.value.detail


@pytest.mark.parametrize(
    "route,comp",
    [("move_route", "direction_pass"),
     ("move_route", "rmse_non_inferior"),
     ("process_route", "auprc_pass"),
     ("process_route", "q_non_inferior")],
)
def test_route_logic_component_threshold_drift_rejected(
        base_cfg, route, comp):
    def _bad(c, r=route, k=comp):
        c["evaluation_contract"]["route_logic"][r][k] = "任意阈值"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert comp in exc.value.detail


def test_route_logic_missing_component_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["route_logic"]["process_route"].pop("q_pass")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "q_pass" in exc.value.detail


def test_route_logic_seed17_reuse_flag_required(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["route_logic"][
            "seed17_screen_reuses_same_parenthesized_logic"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "parenthesized" in exc.value.detail


def test_h3_rule_must_restate_parenthesized_logic(base_cfg):
    def _bad(c):
        c["gates"]["H3"]["rule"] = c["gates"]["H3"]["rule"].replace(
            "(direction_pass AND rmse_non_inferior) OR "
            "(rmse_pass AND direction_non_inferior)",
            "direction pass 且 rmse 非劣, 或 rmse pass")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "parenthesized" in exc.value.detail


def test_seed17_point_route_must_reuse_parenthesized_logic(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "point_effect_requirement"]["move_route"] = (
            "direction 相对 >=3% 且绝对 >=0.01, 或 rmse 相对改善 >=3%, "
            "且另一出口非劣")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "move_route" in exc.value.detail


def test_seed17_dangling_or_forbidden_flag_required(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "point_effect_requirement"]["dangling_or_condition_forbidden"] = (
            False)

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "dangling_or_condition_forbidden" in exc.value.detail


@pytest.mark.parametrize("key,tok", [
    ("process_route", "0.02"),
    ("process_route", "0.03"),
    ("process_route", "0.01"),
    ("move_route", "3%"),
    ("move_route", "0.01"),
    ("move_route", "2%"),
])
def test_seed17_point_threshold_tokens_frozen(base_cfg, key, tok):
    def _bad(c, k=key, token=tok):
        text = c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "point_effect_requirement"][k]
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "point_effect_requirement"][k] = text.replace(token, "9.99")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert tok in exc.value.detail


def test_seed17_rule_must_reference_route_logic(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "point_effect_requirement"]["rule"] = "至少一路 point effect 达标"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "route_logic" in exc.value.detail


def test_seed17_ranking_allows_fewer_than_two(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"]["ranking"] = (
            "满足 predicate 的 A1-A4 按冻结 screen score 排序, 最多 2 个, "
            "不足 2 时补足")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "可少于 2" in exc.value.detail


def test_seed17_scope_must_disclaim_h3_strongly(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"]["scope"] = (
            "用于 top2 eligibility 与 A5 supplement 激活, H3 合格声明")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "NOT a three-seed H3 claim" in exc.value.detail


# ---------------------------------------------------------------------------
# fix-6 C1: frozen final_selection_rule
# ---------------------------------------------------------------------------
def test_final_selection_rule_frozen_in_config(base_cfg):
    rule = base_cfg["evaluation_contract"]["final_selection_rule"]
    assert rule["definition_name"] == v.FINAL_SELECTION_NAME
    assert "H3-qualified" in rule["eligible"]
    assert tuple(rule["excluded"]) == v.FINAL_SELECTION_EXCLUDED
    assert rule["dev_used"] is False
    assert rule["report_all_qualified"] is True
    assert tuple(rule["tie_break_order"]) == v.FINAL_SELECTION_TIE_BREAK
    assert dict(rule["effect_margin_definition"]) == v.FINAL_EFFECT_MARGIN
    assert rule["rule_block_in_statistics_sha"] is True
    assert rule["a5_c_activation_uses_selected_candidate_only"] is True


def test_final_selection_missing_block_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"].pop("final_selection_rule")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "final_selection_rule" in exc.value.detail


def test_final_selection_tie_break_reorder_rejected(base_cfg):
    """fix-6 C1: the lexicographic ORDER is part of the freeze."""
    def _bad(c):
        tb = c["evaluation_contract"]["final_selection_rule"]["tie_break_order"]
        tb[0], tb[3] = tb[3], tb[0]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "tie_break_order" in exc.value.detail


def test_final_selection_missing_step_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "tie_break_order"].pop()  # drop candidate_id lexicographic step

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "7_min_candidate_id_lexicographic" in exc.value.detail


def test_final_selection_extra_step_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "tie_break_order"].append("8_max_reported_effect")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "tie_break_order" in exc.value.detail


def test_final_selection_dev_used_true_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"]["dev_used"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "dev_used" in exc.value.detail


def test_final_selection_report_all_qualified_required(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "report_all_qualified"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "report_all_qualified" in exc.value.detail


def test_final_selection_excluded_set_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"]["excluded"] = [
            "A0_matched_baseline"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "diagnostic_variants_A1-C_A5-C" in exc.value.detail


def test_final_selection_eligible_must_be_h3_qualified_only(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"]["eligible"] = (
            "inner-val 排名靠前的候选")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "H3-qualified" in exc.value.detail


def test_final_selection_no_candidate_rule_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "no_candidate_rule"] = "0 个候选则重跑"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "no_backbone_winner" in exc.value.detail


def test_final_selection_single_candidate_rule_frozen(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "single_candidate_rule"] = "1 个候选则比较 effect size"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "Backbone" in exc.value.detail


def test_final_selection_effect_margin_drift_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "effect_margin_definition"]["move_direction"] = "relative_effect/0.05"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "effect_margin_definition" in exc.value.detail


def test_final_selection_statistics_sha_flag_required(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "rule_block_in_statistics_sha"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "rule_block_in_statistics_sha" in exc.value.detail


def test_final_selection_a5c_uses_selected_only(base_cfg):
    """fix-6 C1: A5-C activation may only read the SELECTED candidate."""
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "a5_c_activation_uses_selected_candidate_only"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a5_c_activation_uses_selected_candidate_only" in exc.value.detail


def test_final_selection_a5c_condition_must_stay_selected(base_cfg):
    """A5C misuse: conditioning on H3-qualification instead of selection."""
    def _bad(c):
        c["run_matrix"]["a5c_activation"]["condition"] = "a5_h3_qualified"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "a5_selected_backbone" in exc.value.detail


# ---------------------------------------------------------------------------
# dose blocks / smoke / profile / diagnostic
# ---------------------------------------------------------------------------
def test_smoke_updates_frozen_at_200(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["smoke"].update(updates=100))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "200" in exc.value.detail


def test_smoke_backbones_frozen_a0_a5(base_cfg):
    def _bad(c):
        c["smoke"]["backbones"] = ["A0", "A1"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_profile_window_frozen_40_40(base_cfg):
    cfg = _mut(
        base_cfg,
        lambda c: c["profile"].update(window={"w0_updates": 20, "w1_updates": 20}),
    )
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "40" in exc.value.detail


def test_profile_conditional_backbones_frozen(base_cfg):
    def _bad(c):
        c["profile"]["conditional_backbones"] = ["A1-C"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "A1-C" in exc.value.detail and "A5-C" in exc.value.detail


def test_cost_gate_denominator_must_be_v542_a0(base_cfg):
    def _bad(c):
        c["profile"]["denominator"] = "v541_e0_archived_core_time"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V542" in exc.value.detail


def test_profile_must_forbid_v541_denominator(base_cfg):
    def _bad(c):
        c["profile"]["forbidden_denominator"] = "none"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_diagnostic_split_frozen_400_1600(base_cfg):
    def _bad(c):
        c["diagnostic"]["w1_updates"] = 1500

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "2000" in exc.value.detail


def test_diagnostic_a0_reference_frozen(base_cfg):
    def _bad(c):
        c["diagnostic"]["a0_reference"] = "V541_E0_S17"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V542_DIAG_A0_S17" in exc.value.detail


def test_diagnostic_must_be_elimination_only(base_cfg):
    def _bad(c):
        c["diagnostic"]["forbids_full_dose_h3_effect_gates"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "H3" in exc.value.detail


def test_cost_acceptance_rule_hard_ratio(base_cfg):
    cfg = _mut(base_cfg, lambda c: c["cost_acceptance"].update(rule="soft_budget"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_cost_acceptance_ratio_references_threshold(base_cfg):
    def _bad(c):
        c["cost_acceptance"]["ratio_expression"] = "candidate/core <= 2.0 fixed"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "max_core_time_ratio" in exc.value.detail


# ---------------------------------------------------------------------------
# selection / tau / sampler / init source / a0 decision
# ---------------------------------------------------------------------------
def test_selection_metric_must_be_inner_val_total_loss(base_cfg):
    def _bad(c):
        c["training_contract"]["selection"]["metric"] = "inner-val H3 direction"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "inner-val total loss" in exc.value.detail


def test_tau_must_calibrate_on_inner_val(base_cfg):
    def _bad(c):
        c["training_contract"]["tau_end"]["calibrated_on"] = "dev"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "inner-val" in exc.value.detail


def test_sampler_id_drift_rejected(base_cfg):
    def _bad(c):
        c["training_contract"]["sampler_binding"]["sampler_id"] = "v999_custom"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_SAMPLER_DRIFT"


def test_sampler_spec_sha_drift_rejected(base_cfg):
    def _bad(c):
        c["training_contract"]["sampler_binding"]["spec_sha256"] = "0" * 64

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_SAMPLER_DRIFT"
    assert "spec_sha256" in exc.value.detail


def test_warmstart_must_be_r3_path(base_cfg):
    def _bad(c):
        c["training_contract"]["backbone_init"]["path"] = (
            "outputs/pesn/v541_embed_50k/runs/V541_E0_S17/checkpoint_best.pt")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "R3" in exc.value.detail


def test_r3_exact_path_required_no_substitute_copy(base_cfg):
    """A different file under the same run directory must not substitute."""
    def _bad(c):
        c["training_contract"]["backbone_init"]["path"] = (
            "outputs/pesn/30h_campaign/r3_runs_alt/5372d9b12807c3f0/"
            "checkpoint_best.pt")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert v.R3_REL_PATH in exc.value.detail


def test_r3_init_sha_must_be_64_hex(base_cfg):
    def _bad(c):
        c["training_contract"]["backbone_init"]["sha256"] = "abc"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"


def test_r3_sha_must_equal_frozen_constant(base_cfg):
    """A valid but different 64-hex digest is a typed R3 identity drift."""
    def _bad(c):
        c["training_contract"]["backbone_init"]["sha256"] = "f" * 64

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_R3_INIT"
    assert "restore the checkpoint" in exc.value.detail


def test_r3_init_hash_drift_detected(base_cfg):
    """The on-disk checkpoint must hash to the frozen R3_SHA256."""
    cfg = base_cfg
    init = cfg["training_contract"]["backbone_init"]
    assert init["sha256"] == v.R3_SHA256
    target = REPO_ROOT / init["path"]
    assert target.is_file()
    import hashlib

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert digest == v.R3_SHA256


def test_r3_init_verify_required(base_cfg):
    def _bad(c):
        c["training_contract"]["backbone_init"]["verify_on_validate"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"


def test_forbidden_sources_must_name_e0_final(base_cfg):
    def _bad(c):
        c["training_contract"]["backbone_init"]["forbidden_sources"] = [
            "any_checkpoint_trained_beyond_R3"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "V541_E0_final" in exc.value.detail


def test_init_policy_mapping_must_forbid_shape_heuristics(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["mapping"] = "runtime_shape_matching"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "shape" in exc.value.detail


# ---------------------------------------------------------------------------
# gate-0 A8: mapping_audit frozen EXACTLY with the w_v dual-write exception
# ---------------------------------------------------------------------------
def test_init_mapping_audit_frozen_exact(base_cfg):
    audit = base_cfg["training_contract"]["init_policy"]["mapping_audit"]
    assert audit == v.INIT_MAPPING_AUDIT_EXACT
    assert "w_v dual-write alias" in audit
    assert "backbone.w_v <-> move_decoder.w_v" in audit


def test_init_mapping_audit_without_wv_exception_rejected(base_cfg):
    """The OLD 'consumed exactly once' string (no alias exception) must be
    rejected — a strict audit would otherwise flag the legitimate w_v
    dual-write as double consumption."""
    def _bad(c):
        c["training_contract"]["init_policy"]["mapping_audit"] = (
            "every source key consumed exactly once; every target key "
            "loaded or fresh-listed with seed; loaded tensors torch.equal; "
            "no empty slots")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "dual-write" in exc.value.detail


def test_init_mapping_audit_token_preserving_near_miss_rejected(base_cfg):
    """Adversarial: keep every meaningful token but reword the alias —
    exact-match enforcement must reject it."""
    def _bad(c):
        c["training_contract"]["init_policy"]["mapping_audit"] = (
            "every source key consumed exactly once except a declared w_v "
            "dual-write alias (backbone.w_v <-> move_decoder.w_v, same "
            "tensor); every target key loaded or fresh-listed with seed; "
            "loaded tensors torch.equal; no empty slots")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "mapping_audit" in exc.value.detail


def test_init_mapping_audit_typo_near_miss_rejected(base_cfg):
    """Adversarial: same tokens, one character changed (w_v -> w)."""
    def _bad(c):
        c["training_contract"]["init_policy"]["mapping_audit"] = (
            v.INIT_MAPPING_AUDIT_EXACT.replace(
                "declared w_v dual-write", "declared w dual-write"))

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"


def test_init_mapping_note_states_37_plus_6_is_43(base_cfg):
    """gate-0 mapping-count clarification: 37 backbone keys (w_v alias) +
    6 process-head keys = 43 joint keys for joint checkpoints."""
    note = base_cfg["training_contract"]["init_policy"]["init_mapping"][
        "note"]
    assert "37" in note and "43" in note and "w_v" in note
    assert "6 process head" in note


def test_init_mapping_note_dropping_joint_43_rejected(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["init_mapping"]["note"] = (
            "R3 37 键 → 每候选语义映射表; 所有 w_v 来自 R3")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "43" in exc.value.detail


def test_init_seed_recipe_frozen(base_cfg):
    fresh = base_cfg["training_contract"]["init_policy"]["candidate_fresh_init"]
    assert fresh["seed_recipe"] == v.INIT_SEED_RECIPE
    assert fresh["seed_recipe_algorithm"] == v.INIT_SEED_RECIPE_ALGORITHM
    assert "sha256" in fresh["seed_recipe"]
    assert "digest()[:8]" in fresh["seed_recipe"]
    assert "2147483647" in fresh["seed_recipe"]


def test_init_seed_recipe_drift_rejected(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["candidate_fresh_init"][
            "seed_recipe"] = "seed_init = run_seed + 1"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "seed_recipe" in exc.value.detail


def test_init_source_id_overrides_frozen(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["candidate_fresh_init"][
            "init_source_id_overrides"] = {"A1-C": "A2", "A5-C": "A5"}

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "A1-C" in exc.value.detail


@pytest.mark.parametrize("bid,wrong", [("A1-C", "A2"), ("A5-C", "A1")])
def test_candidate_init_source_id_frozen(base_cfg, bid, wrong):
    def _bad(c, backbone=bid, src=wrong):
        c["backbone_contract"]["candidates"][backbone]["init_source_id"] = src

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert bid in exc.value.detail


def test_construction_order_frozen(base_cfg):
    order = base_cfg["training_contract"]["init_policy"]["construction_order"]
    assert tuple(order) == v.CONSTRUCTION_ORDER
    assert len(order) == 6


def test_construction_order_drift_rejected(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["construction_order"] = [
            "1_fresh_backbone_modules_seeded_by_seed_init",
            "2_set_seed(run_seed)_before_training",
        ]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "construction_order" in exc.value.detail


def test_r3_semantic_mapping_covers_all_candidates(base_cfg):
    mapping = base_cfg["training_contract"]["init_policy"]["init_mapping"][
        "r3_semantic_mapping"]
    assert set(mapping) == set(v.R3_MAPPING_CANDIDATES)
    assert "V_0=0" in mapping["A2"]
    assert "A1 init stream" in mapping["A1-C"]
    assert "A5 init_stream" not in mapping["A5-C"]  # wording guard
    assert "A5 init stream" in mapping["A5-C"]


def test_r3_semantic_mapping_missing_candidate_rejected(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["init_mapping"][
            "r3_semantic_mapping"].pop("A2")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "A2" in exc.value.detail


def test_a2_vector_seed_choice_frozen(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["init_mapping"][
            "r3_semantic_mapping"]["A2"] = "A1 common 映射集 + 向量通道 fresh seeded"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "V_0=0" in exc.value.detail


def test_a2_phi_output_zero_init_frozen(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["per_candidate_zero_init"]["A2"][
            "phi_u_phi_g_final_output"] = "default_seeded"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "phi_u_phi_g_final_output" in exc.value.detail


def test_a1_w_d_zero_init_frozen(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["per_candidate_zero_init"]["A1"] = (
            "W_d default_seeded")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "W_d" in exc.value.detail


def test_process_head_anchor_must_use_fork_rng(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["process_head_anchor"][
            "method"] = "sequential_construct"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "fork_rng" in exc.value.detail


def test_anchor_seeds_frozen(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["process_head_anchor"]["seeds"] = [17]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"


def test_anchor_must_enter_run_identity(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["process_head_anchor"][
            "included_in_run_identity"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "included_in_run_identity" in exc.value.detail


def test_anchor_must_exclude_w_v(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["process_head_anchor"][
            "excludes_w_v"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "excludes_w_v" in exc.value.detail


def test_param_groups_must_be_by_id(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["param_groups"]["method"] = (
            "name_prefix")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "by_id" in exc.value.detail


def test_param_group_core_lr_frozen(base_cfg):
    groups = base_cfg["training_contract"]["init_policy"]["param_groups"]
    assert groups["core"] == {"members": ["backbone", "move_decoder"], "lr": 1e-4}
    assert groups["head"] == {"members": ["process_head"], "lr": 3e-4}
    assert groups["w0"] == "core frozen, head live"
    assert groups["w1"] == "all joint"


def test_param_group_core_lr_drift_rejected(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["param_groups"]["core"]["lr"] = 1e-3

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "core" in exc.value.detail


def test_param_group_head_members_frozen(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["param_groups"]["head"]["members"] = [
            "process_head", "move_decoder"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"
    assert "head" in exc.value.detail


def test_seed_reset_before_training_required(base_cfg):
    def _bad(c):
        c["training_contract"]["init_policy"]["reset_seed_before_training"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INIT_SOURCE"


def test_a0_decision_frozen_to_retrain(base_cfg):
    def _bad(c):
        c["a0_baseline"]["decision"] = "reuse_e0_x3"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_A0_REUSE"
    assert "retrain_a0_x3" in exc.value.detail


def test_shadow_run_not_required_in_v542(base_cfg):
    """The reuse/shadow-run branch is intentionally NOT implemented."""
    def _bad(c):
        c["a0_baseline"]["shadow_run_required"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_A0_REUSE"
    assert "shadow_run_required" in exc.value.detail


def test_e0_role_keeps_parity_historical_only(base_cfg):
    def _bad(c):
        c["a0_baseline"]["v541_e0_role"] = "h2_h3_comparator"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_A0_REUSE"
    assert "parity" in exc.value.detail


def test_a0_retrain_runs_frozen_set(base_cfg):
    def _bad(c):
        c["a0_baseline"]["runs"] = ["V542_A0_S17"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_A0_REUSE"
    assert "V542_A0_S29" in exc.value.detail


def test_decision_template_is_not_the_instance(base_cfg):
    """decision_record_instance must be the INSTANCE path, never the template."""
    def _bad(c):
        c["a0_baseline"]["decision_record_instance"] = (
            "config/campaigns/v542_templates/a0_reuse_decision.template.json")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_A0_REUSE"
    assert "INSTANCE" in exc.value.detail


def test_decision_instance_required_flag_frozen(base_cfg):
    def _bad(c):
        c["a0_baseline"]["decision_record_instance_required"] = False

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_A0_REUSE"
    assert "instance_required" in exc.value.detail


# ---------------------------------------------------------------------------
# identity / start gates
# ---------------------------------------------------------------------------
def test_identity_required_fields_frozen(base_cfg):
    ic = base_cfg["identity_contract"]
    assert tuple(ic["run_identity_required"]) == v.IDENTITY_REQUIRED_FIELDS
    assert "source_manifest_sha256" in ic["run_identity_required"]
    assert "code_commit" not in ic["run_identity_required"]
    assert tuple(ic["run_identity_optional"]) == ("code_commit",)
    assert ic["code"]["git_commit_authoritative"] is False


def test_identity_source_manifest_required(base_cfg):
    def _bad(c):
        c["identity_contract"]["run_identity_required"] = [
            f for f in c["identity_contract"]["run_identity_required"]
            if f != "source_manifest_sha256"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "source_manifest_sha256" in exc.value.detail


def test_identity_code_commit_must_stay_optional(base_cfg):
    def _bad(c):
        c["identity_contract"]["run_identity_required"].append("code_commit")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "code_commit" in exc.value.detail


def test_git_commit_not_authoritative(base_cfg):
    def _bad(c):
        c["identity_contract"]["code"]["git_commit_authoritative"] = True

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "git_commit_authoritative" in exc.value.detail


def test_t1_start_requires_canonical_gate_ids(base_cfg):
    requires = base_cfg["governance"]["authorization"]["t1_start_requires"]
    assert tuple(requires) == v.T1_START_REQUIRES
    assert "runplan_pin_record" in requires
    assert "V542-H0-legality" in requires
    assert "V542-H1-resource" in requires
    assert "H0" not in requires and "H1" not in requires
    assert "runplan_pin" not in requires  # record ALIAS is forbidden


@pytest.mark.parametrize(
    "bad_entry",
    ["H0", "H1", "runplan_pin", "V542-WP3-runner", "budget_signoff"],
)
def test_t1_start_requires_non_canonical_rejected(base_cfg, bad_entry):
    def _bad(c, entry=bad_entry):
        c["governance"]["authorization"]["t1_start_requires"][0] = entry

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert bad_entry in exc.value.detail


def test_t1_start_requires_must_keep_runplan_pin_record(base_cfg):
    def _bad(c):
        c["governance"]["authorization"]["t1_start_requires"] = [
            f for f in c["governance"]["authorization"]["t1_start_requires"]
            if f != "runplan_pin_record"]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "runplan_pin_record" in exc.value.detail


def test_t1_start_requires_must_keep_h0_h1_diagnostic_gates(base_cfg):
    def _bad(c):
        c["governance"]["authorization"]["t1_start_requires"] = [
            f for f in c["governance"]["authorization"]["t1_start_requires"]
            if f not in ("V542-H0-legality", "V542-diagnostic-2000")]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V542-H0-legality" in exc.value.detail


def test_t1_start_requires_exact_frozen_order(base_cfg):
    """fix-6 + gate-0 A5: the canonical set is frozen EXACTLY, order included.

    9 registry gates + 3 record requirements (runplan_pin_record +
    a0_reuse_decision_record + preregistration_signoff_record).
    """
    requires = base_cfg["governance"]["authorization"]["t1_start_requires"]
    assert tuple(requires) == v.T1_START_REQUIRES
    assert len(requires) == 12  # 9 registry gates + 3 record requirements
    assert tuple(r for r in requires if r not in v.T1_RECORD_WHITELIST) == (
        v.T1_CANONICAL_GATES)
    assert tuple(r for r in requires if r in v.T1_RECORD_WHITELIST) == (
        v.T1_RECORD_REQUIREMENTS)


def test_t1_start_requires_a0_sanity_band_present(base_cfg):
    """fix-6 C5: V542-A0-sanity-band is a hard T1 precondition."""
    requires = base_cfg["governance"]["authorization"]["t1_start_requires"]
    assert v.A0_SANITY_BAND_GATE in requires
    assert requires.index(v.A0_SANITY_BAND_GATE) < requires.index(
        "V542-budget-signoff")


def test_t1_start_requires_reorder_rejected(base_cfg):
    def _bad(c):
        req = c["governance"]["authorization"]["t1_start_requires"]
        req[0], req[1] = req[1], req[0]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V542-A0-sanity-band" in exc.value.detail


def test_t1_start_requires_dropping_a0_sanity_band_rejected(base_cfg):
    def _bad(c):
        c["governance"]["authorization"]["t1_start_requires"] = [
            f for f in c["governance"]["authorization"]["t1_start_requires"]
            if f != v.A0_SANITY_BAND_GATE]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert v.A0_SANITY_BAND_GATE in exc.value.detail


def test_a0_sanity_band_gate_registered_in_gv_map():
    """fix-6: the sanity-band gate must exist in the frozen registry."""
    from pes2ts_core.gv_registry import DEFAULT_MAP_PATH, load_gv_map

    gv_map = load_gv_map(DEFAULT_MAP_PATH)
    from pes2ts_core.gv_registry import resolve_campaign

    entry = resolve_campaign(gv_map, "V5.4.2")
    assert v.A0_SANITY_BAND_GATE in entry.gates


def test_a0_sanity_band_values_frozen(base_cfg):
    def _bad(c):
        c["a0_baseline"]["sanity_band_vs_v541_e0"]["auprc_abs_max"] = 0.05

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "sanity_band_vs_v541_e0" in exc.value.detail


def test_a0_sanity_band_gate_field_frozen(base_cfg):
    def _bad(c):
        c["a0_baseline"]["sanity_band_gate"] = "V542-A0-band"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert v.A0_SANITY_BAND_GATE in exc.value.detail


def test_a0_sanity_band_statistic_is_seed_mean(base_cfg):
    def _bad(c):
        c["a0_baseline"]["sanity_band_statistic"] = "A0 单 seed 检查点对比"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V541 E0×3" in exc.value.detail


def test_a0_breach_procedure_exact_five_steps(base_cfg):
    def _bad(c):
        c["a0_baseline"]["sanity_band_breach_procedure"] = [
            "1_strictly_block_all_T1_formal_candidate_launches",
            "2_migration_audit",
            "6_sign_exception_and_continue",  # the forbidden escape hatch
        ]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "5_sanity_exception_signing_forbidden_under_breach" in exc.value.detail


def test_a0_breach_note_forbids_exception(base_cfg):
    def _bad(c):
        c["a0_baseline"]["sanity_band_note"] = "breach 后可登记 exception 继续"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "exception" in exc.value.detail
    assert "禁止" in exc.value.detail


# ---------------------------------------------------------------------------
# gate-0 A8: breach step 3 names the identity layer per defect class
# ---------------------------------------------------------------------------
def test_a0_breach_step3_names_identity_layer_per_defect(base_cfg):
    raw = base_cfg["a0_baseline"]["sanity_band_breach_procedure"][2]
    # YAML parses the `key: value` row as a mapping — flatten like the
    # validator's _normalize_steps does.
    step3 = (f"{k}: {val}" for k, val in raw.items()) if isinstance(
        raw, dict) else [str(raw)]
    step3 = " ".join(step3)
    for token in ("code_defect=>source_tree_fingerprint",
                  "data_defect=>data_manifest_sha256",
                  "init_defect=>init_mapping_sha256",
                  "anchor_implicated=>process_head_anchor_sha256",
                  "rerun_A0_x3_formal", "invalid_run", "no exception"):
        assert token in step3, token


def test_a0_breach_step3_single_layer_near_miss_rejected(base_cfg):
    """Adversarial: the OLD source_tree_fingerprint-only step 3 (every
    token present, but no per-defect identity layer) must be rejected."""
    def _bad(c):
        c["a0_baseline"]["sanity_band_breach_procedure"][2] = (
            "3_if_code_data_or_init_defect_found: fix_defect_then_bump_"
            "source_tree_fingerprint_and_rerun_A0_x3_formal (code/data/init "
            "all bump source_tree_fingerprint); old_A0x3=>invalid_run, no "
            "exception")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "sanity_band_breach_procedure" in exc.value.detail


def test_a0_breach_step3_dropping_data_layer_rejected(base_cfg):
    """Adversarial: token-preserving near-miss — keeps the code and init
    layers but silently drops the data layer."""
    def _bad(c):
        c["a0_baseline"]["sanity_band_breach_procedure"][2] = (
            "3_if_code_data_or_init_defect_found: fix_defect_then_bump_"
            "matching_identity_layer (code_defect=>source_tree_fingerprint; "
            "init_defect=>init_mapping_sha256; "
            "anchor_implicated=>process_head_anchor_sha256) then "
            "rerun_A0_x3_formal; old_A0x3=>invalid_run, no exception")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"


def test_a0_sanity_band_note_frozen_exact(base_cfg):
    note = base_cfg["a0_baseline"]["sanity_band_note"]
    assert note == v.A0_SANITY_BAND_NOTE_EXACT
    for token in ("source_tree_fingerprint", "data_manifest_sha256",
                  "init_mapping_sha256", "process_head_anchor_sha256",
                  "invalid_run", "禁止", "exception"):
        assert token in note, token


def test_a0_sanity_band_note_single_layer_near_miss_rejected(base_cfg):
    """Adversarial: the OLD note (single bump-code-fingerprint rule) must
    be rejected — it bumps the wrong identity layer for data/init defects."""
    def _bad(c):
        c["a0_baseline"]["sanity_band_note"] = (
            "带宽只抓迁移大 bug, 不要求与 E0 数值相同; breach = 严格阻断 T1 + "
            "迁移审计 + (若发现代码/数据/初始化缺陷) 修复后 bump code "
            "fingerprint 并重跑 A0×3; 禁止签 exception 继续")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "sanity_band_note" in exc.value.detail
    assert "禁止" in exc.value.detail


def test_formal_stage_order_t0_before_t1(base_cfg):
    def _bad(c):
        order = c["run_matrix"]["formal_stage_order"]["order"]
        order[0], order[1] = order[1], order[0]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "T0_V542_A0_S17_S29_S43_formal_plus_V542-A0-sanity-band" in (
        exc.value.detail)


def test_formal_stage_order_frozen_exactly(base_cfg):
    order = tuple(base_cfg["run_matrix"]["formal_stage_order"]["order"])
    assert order == v.FORMAL_STAGE_ORDER
    assert order[0].startswith("T0_")
    assert order[1].startswith("T1_")
    assert order[-1].startswith("T-ATTR_")


def test_top2_freeze_requires_sanity_band(base_cfg):
    def _bad(c):
        c["run_matrix"]["top2_preregistration"]["freeze_requires"] = (
            "A0_S17 complete")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert v.A0_SANITY_BAND_GATE in exc.value.detail


def test_seed17_top2_freeze_requires_sanity_band(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "top2_freeze_requires"] = "A0_S17 complete"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert v.A0_SANITY_BAND_GATE in exc.value.detail


# ---------------------------------------------------------------------------
# file-level rejections
# ---------------------------------------------------------------------------
def test_missing_config_file_typed():
    with pytest.raises(v.V542SchemaError) as exc:
        v.load_config(REPO_ROOT / "config" / "campaigns" / "does_not_exist.yaml")
    assert exc.value.code == "V542_CONFIG_MISSING"


def test_non_mapping_yaml_typed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(v.V542SchemaError) as exc:
        v.load_config(bad)
    assert exc.value.code == "V542_CONFIG_MISSING"


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
    assert json.loads(proc.stdout)["code"] == "V542_OBJECTIVE"


# ---------------------------------------------------------------------------
# WP0 templates
# ---------------------------------------------------------------------------
def test_all_five_templates_exist_and_parse():
    for name in (
        "preregistration_signoff.template.json",
        "budget_policy.template.json",
        "runplan_pin.template.json",
        "parity_certificate.template.json",
        "a0_reuse_decision.template.json",
    ):
        data = json.loads((TPL_DIR / name).read_text(encoding="utf-8"))
        assert data["schema"].startswith("pes2ts_v542_")


def test_signoff_template_mirrors_config_thresholds_and_authorization():
    signoff = json.loads(
        (TPL_DIR / "preregistration_signoff.template.json").read_text(encoding="utf-8")
    )
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    signed = signoff["thresholds_signed_frozen"]
    assert signed["status"] == "frozen" and signed["signed_on"] == "2026-10-29"
    for key in v.FORMAL_THRESHOLDS:
        assert signed[key] == cfg["thresholds"][key], key
    assert signed["top2_max_runs"] == 4
    assert signed["a5_supplement_max_runs"] == 2
    assert signed["t3_max_runs"] == 6
    assert signed["bootstrap_seed_root"] == 20261029
    auth = signoff["authorization"]
    assert auth["user_directive_date"] == "2026-10-29"
    assert auth["implementation_authorized"] is True
    assert auth["formal_training_authorized"] is True
    assert auth["confirm_authorized"] is False
    assert auth["sealed_authorized"] is False
    assert auth["new_qc_calls"] == 0
    assert auth["single_lane"] is True and auth["swap_allowed"] is False

    a0 = signoff["a0_baseline_decision"]
    assert a0["decision"] == "retrain_a0_x3"
    assert a0["e0_role"] == "parity_and_historical_reference_only"
    assert a0["shadow_run_required"] is False
    assert a0["decision_record_instance_required_before_T1"] is True
    assert a0["decision_record_instance"].endswith("a0_reuse_decision.json")

    t2_cfg = cfg["run_matrix"]["top2_preregistration"]
    t2_signoff = signoff["top2_preregistration"]
    for key in (
        "listed_runs",
        "candidates",
        "seeds",
        "activation",
        "top2_max_runs",
        "a5_supplement_max_runs",
        "t3_max_runs",
        "unactivated_status",
        "n_core_counting",
    ):
        assert t2_signoff[key] == t2_cfg[key], key
    assert t2_signoff["a5_supplement_parent_gate"] == (
        cfg["run_matrix"]["a5_supplement"]["parent_gate"])


def test_signoff_template_gate_rules_frozen():
    signoff = json.loads(
        (TPL_DIR / "preregistration_signoff.template.json").read_text(encoding="utf-8")
    )
    rules = signoff["gate_preregistration"]
    assert "退化" in rules["H2_rule"]  # degradation qualifier (finding 7)
    h3 = rules["H3_rule"]
    assert "20261029" in h3
    assert "Holm" in h3
    assert "(10001)" in h3 or "/ (10001)" in h3
    assert "10000" in h3
    assert "adjusted p" in h3
    screen = rules["seed17_screen_pass_rule"]
    assert "非三 seed H3" in screen or "非三 seed" in screen
    assert "a5_selected_backbone" in rules["a5c_rule"] or "A5 被选" in rules["a5c_rule"]
    assert "a5_full_dose_complete_and_seed17_screen_pass" in rules["a5_rule"] or (
        "seed17_screen_pass" in rules["a5_rule"])


def test_signoff_template_manifest_layers_cover_identity():
    signoff = json.loads(
        (TPL_DIR / "preregistration_signoff.template.json").read_text(encoding="utf-8")
    )
    layers = set(signoff["manifest_sha_layers"])
    assert set(v.IDENTITY_REQUIRED_FIELDS) - {"config_sha256"} <= layers
    assert "source_manifest_sha256" in layers
    assert "bootstrap_seed_root" in layers
    assert "statistics_contract_sha256" in layers
    assert "process_head_anchor_sha256" in layers
    assert "init_mapping_sha256" in layers


def test_budget_template_is_pre_profile_and_null():
    """The TEMPLATE stays pre-profile; the campaign config is now signed."""
    budget = json.loads(
        (TPL_DIR / "budget_policy.template.json").read_text(encoding="utf-8")
    )
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert budget["per_candidate_status"] == "pending_wp3_profiling"
    signed = cfg["budget"]["per_candidate_core_time"]
    assert isinstance(signed, dict)
    assert set(signed) == {"A0", "A1", "A2", "A3", "A4", "A5", "A1-C", "A5-C"}
    assert all(isinstance(v, (int, float)) and v > 0 for v in signed.values())
    assert cfg["budget"]["per_candidate_status"] == "signed_wp4_profile"
    assert cfg["budget"]["profile_signoff"]["all_within_ratio"] is True
    for cid in ("A0", "A1", "A2", "A3", "A4", "A5", "A1-C", "A5-C"):
        entry = budget["per_candidate"][cid]
        assert entry["approved_for_formal"] is None
        assert entry["core_seconds"] is None
    assert budget["cost_acceptance"]["max_core_time_ratio"] == 2.0
    assert "V542" in budget["cost_acceptance"]["denominator"]
    assert budget["schedule"]["total_updates"] == 37900
    assert budget["schedule"]["diagnostic_split"] == {"w0": 400, "w1": 1600}
    assert budget["execution_policy"]["single_lane"] is True
    assert budget["execution_policy"]["swap_max"] == 0
    assert budget["execution_policy"]["oom_kill_max"] == 0


def test_budget_template_activation_summary_v11_split():
    budget = json.loads(
        (TPL_DIR / "budget_policy.template.json").read_text(encoding="utf-8")
    )
    summary = budget["activation_summary"]
    assert summary["top2_max_runs"] == 4
    assert summary["a5_supplement_max_runs"] == 2
    assert summary["t3_max_runs"] == 6
    assert "A1-A4 ONLY" in summary["top2_note"]
    assert "a5_full_dose_complete_and_seed17_screen_pass" in summary["top2_note"]
    assert "budget record" in summary["A1C_attribution"] or (
        "budget" in summary["A1C_attribution"])
    assert "SELECTED as Backbone" in summary["A5C_attribution"]


def test_runplan_pin_run_ids_match_config_exactly():
    pin = json.loads(
        (TPL_DIR / "runplan_pin.template.json").read_text(encoding="utf-8")
    )
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    classes = pin["run_classes"]
    assert set(classes) == {"nominal", "conditional", "diagnostic"}
    assert classes["diagnostic"]["counts_toward_winner"] is False
    # winner eligibility is judged PER RUN — never a blanket class claim
    assert classes["nominal"]["counts_toward_winner"] == "per_run_row"
    assert classes["conditional"]["counts_toward_winner"] == "per_run_row"
    cfg_ids = {r["run_id"] for r in cfg["run_matrix"]["runs"]}
    pin_ids = {rid for cls in classes.values() for rid in cls["run_ids"]}
    assert len(cfg_ids) == 26
    assert cfg_ids == pin_ids, (cfg_ids ^ pin_ids)
    formula = pin["core_training_count_formula"]
    assert "NEVER by summing listed run rows" in formula
    t2_pin = classes["conditional"]["top2_preregistration"]
    assert t2_pin["listed_runs"] == 8
    assert t2_pin["candidates"] == ["A1", "A2", "A3", "A4"]
    assert t2_pin["top2_max_runs"] == 4
    assert t2_pin["unactivated_new_trainings"] == 0
    assert "A0_S17" in t2_pin["freeze_requires"]
    sup_pin = classes["conditional"]["a5_supplement"]
    assert sup_pin["runs"] == ["V542_A5_S29", "V542_A5_S43"]
    assert sup_pin["a5_supplement_max_runs"] == 2
    assert sup_pin["parent_gate"] == "a5_full_dose_complete_and_seed17_screen_pass"
    assert "NOT a three-seed H3 claim" in sup_pin["note"]
    assert pin["shared_protocol"]["sampler_spec_sha256"] == (
        cfg["training_contract"]["sampler_binding"]["spec_sha256"]
    )


def test_runplan_pin_statistics_contract_mirrors_config():
    pin = json.loads(
        (TPL_DIR / "runplan_pin.template.json").read_text(encoding="utf-8")
    )
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    stats = cfg["evaluation_contract"]["statistics"]
    contract = pin["statistics_contract"]
    assert contract["bootstrap_seed_root"] == stats["bootstrap_seed_root"] == 20261029
    assert contract["bootstrap_subseed_formula"] == stats["bootstrap_subseed_formula"]
    assert contract["final_draws"] == stats["final_draws"] == 10000
    assert "(10001)" in contract["p_value_rule"]
    assert "frame-level forbidden" in contract["p_value_rule"]
    assert "alpha=0.025" in contract["holm_families"]
    assert "diagnostic variants excluded" in contract["h3_eligible"]


def test_template_run_classes_align_with_config_by_class():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    pin = json.loads(
        (TPL_DIR / "runplan_pin.template.json").read_text(encoding="utf-8")
    )
    by_class: dict[str, set[str]] = {}
    for run in cfg["run_matrix"]["runs"]:
        by_class.setdefault(run["class"], set()).add(run["run_id"])
    for name, cls in pin["run_classes"].items():
        assert set(cls["run_ids"]) == by_class[name], name


def test_parity_template_lists_six_required_checkpoints():
    parity = json.loads(
        (TPL_DIR / "parity_certificate.template.json").read_text(encoding="utf-8")
    )
    assert parity["checkpoint_count"] == 6
    assert parity["required_count"] == 6
    ids = [ck["id"] for ck in parity["checkpoints"]]
    assert ids == [exp[0] for exp in v.PARITY_ASSETS]
    for ck, (exp_id, exp_path, exp_sha) in zip(parity["checkpoints"], v.PARITY_ASSETS):
        assert ck["id"] == exp_id
        assert ck["path"] == exp_path
        assert ck["sha256"] == exp_sha
        assert ck["required"] is True  # 6/6 — no optional rows (finding 16)
        assert "/smoke/" not in ck["path"]
    assert parity["asset_policy"]["all_six_required"] is True
    assert parity["asset_policy"]["smoke_checkpoints_forbidden"] is True
    assert parity["fixture_parity"]["outputs"] == [
        "u", "z", "q", "endpoint_logit", "order_score"]
    assert parity["real_batch_parity"]["min_inner_val_batches"] == 8
    assert parity["gpu_tolerance"]["tf32_disabled"] is True
    assert parity["gpu_tolerance"]["legacy_floor_rule"]
    assert "evaluator" in parity["evaluator_parity"]["rule"]
    assert parity["not_evidence_for"] == [
        "e0_reuse", "training_comparability", "h2_h3_baseline_substitution"]


def test_a0_reuse_template_defaults_to_retrain():
    reuse = json.loads(
        (TPL_DIR / "a0_reuse_decision.template.json").read_text(encoding="utf-8")
    )
    assert reuse["decision"] == "retrain_a0_x3"
    assert reuse["decision_enum"] == ["reuse_e0_x3", "retrain_a0_x3"]
    assert reuse["h3_comparator"] == "matched_a0_only"
    assert reuse["shadow_run_required"] is False
    assert reuse["reuse_branch_implemented"] is False
    assert reuse["v541_e0_role"] == "parity_and_historical_reference_only"
    assert reuse["runs"] == ["V542_A0_S17", "V542_A0_S29", "V542_A0_S43"]
    assert reuse["instance_required_before_formal"] is True
    assert reuse["instance_path"] == v.A0_DECISION_INSTANCE
    band = reuse["sanity_band_vs_v541_e0"]
    assert band["rmse_zero_rel_max_pct"] == 2.0
    assert band["auprc_abs_max"] == 0.01
    assert band["q_corr_abs_max"] == 0.01
    domains = reuse["historical_nine_domains_reference"]
    assert {
        "code", "initialization", "sampler_schedule", "data_view", "loss",
        "evaluator", "checkpoint_identity", "environment", "comparator_consistency",
    } <= set(domains)
    assert domains["shadow_run_stage1_updates"] == 200
    assert domains["shadow_run_stage2_updates"] == 2000


# ---------------------------------------------------------------------------
# fix-6 C4: signoff roles (the Oracle is a reviewer, never a signatory)
# ---------------------------------------------------------------------------
def _signoff_template() -> dict:
    return json.loads(
        (TPL_DIR / "preregistration_signoff.template.json").read_text(
            encoding="utf-8")
    )


def test_signoff_statistics_and_evaluation_owned_by_main_session():
    roles = _signoff_template()["signed_by_roles"]
    assert roles["statistics_owner"] == "main_session_owner"
    assert roles["evaluation_signoff"] == "main_session_owner"


def test_signoff_oracle_is_reviewer_not_signatory():
    roles = _signoff_template()["signed_by_roles"]
    assert roles["oracle_reviewer"] == "oracle_gate0_reviewer"
    # no SIGNATORY role may be attributed to the Oracle (prose notes may
    # mention it only to state that it is NOT a signatory)
    signatory_roles = (
        "authorization", "governance_owner", "statistics_owner",
        "governance_signoff", "parity_signoff", "training_signoff",
        "evaluation_signoff", "resource_signoff",
    )
    for key in signatory_roles:
        value = str(roles[key]).lower()
        assert "oracle" not in value, (key, value)
    assert "NOT signatory" in roles["formal_launch_note"]


def test_signoff_pending_roles_stay_null():
    roles = _signoff_template()["signed_by_roles"]
    for key in ("parity_signoff", "training_signoff", "resource_signoff"):
        assert roles[key] is None, key  # deferred until WP1-WP4 evidence


def test_signoff_template_note_states_pending_owner_reaffirm():
    note = _signoff_template()["_template_note"]
    assert "WP0 authoring complete" in note
    assert "formal signoff pending owner re-affirm" in note
    assert "NOT a signatory" in note
    assert "main_session_owner" in note
    assert "all 28 signed keys" in note  # fix-6: 24 -> 28 count is current


def test_signoff_formal_launch_note_not_oracle():
    roles = _signoff_template()["signed_by_roles"]
    launch = roles["formal_launch_note"]
    assert "pending owner re-affirm" in launch
    assert "NOT signatory" in launch
    assert "oracle_gate0_reviewer" in launch


def test_signoff_template_threshold_count_is_28():
    signed = _signoff_template()["thresholds_signed_frozen"]
    keys = [k for k in signed if k not in ("status", "signed_on", "signed_by",
                                           "cost_acceptance_rule")]
    assert len(keys) == 28
    assert set(keys) == set(v.FORMAL_THRESHOLDS)


# ---------------------------------------------------------------------------
# fix-6: authoritative counts (thresholds 28 / typed rejections 19 / checks)
# ---------------------------------------------------------------------------
def test_threshold_block_exactly_28_signed_keys(base_cfg):
    th = base_cfg["thresholds"]
    signed = {k for k in th if k != "signoff"}
    assert len(signed) == 28
    assert signed == set(v.FORMAL_THRESHOLDS)


def test_typed_rejection_code_count_is_twentytwo():
    """22 codes = the docstring table = the authoritative docs count."""
    assert len(v.TYPED_REJECTIONS) == 22
    assert len(set(v.TYPED_REJECTIONS)) == 22
    import inspect
    import re

    source = inspect.getsource(v)
    # every V542_* code literally raised by _err / V542SchemaError must be
    # a declared member of TYPED_REJECTIONS (no undocumented rejections).
    raised = set(re.findall(r'_err\(\s*"(V542_[A-Z0-9_]+)"', source))
    raised |= set(re.findall(r'V542SchemaError\("(V542_[A-Z0-9_]+)"', source))
    raised |= set(re.findall(r'_err\(\s*\n\s*"(V542_[A-Z0-9_]+)"', source))
    declared = set(v.TYPED_REJECTIONS)
    assert raised <= declared, sorted(raised - declared)
    # the docstring table must list exactly the declared codes
    doc = inspect.getdoc(v)
    table_codes = set(re.findall(r"``(V542_[A-Z0-9_]+)``", doc))
    assert table_codes == declared, sorted(table_codes ^ declared)


def test_module_docstring_counts_are_current():
    """The docstring must state the current 22-code contract + fix-6 scope."""
    import inspect

    doc = inspect.getdoc(v)
    assert "22 codes" in doc
    assert "fix-6" in doc
    assert "final_selection_rule" in doc
    assert "route_logic" in doc
    assert "V542-A0-sanity-band" in doc or "A0 sanity band" in doc
    assert "ADR-0033 v1.2" in doc
    # gate-0 A5/A2 closure is recorded in the module contract
    assert "V542_RECORD_MISSING" in doc
    assert "governance_records" in doc


def test_module_docstring_check_list_matches_runtime(base_cfg):
    """The docstring's named-check list must equal the runtime report."""
    import inspect
    import re

    doc = inspect.getdoc(v)
    report = v.validate_config(base_cfg, formal=False)
    section = doc.split("these named checks in order")[1].split("CLI::")[0]
    listed = set(re.findall(r"``([a-z0-9_]+)``", section))
    assert listed == set(report["checks"]), (
        sorted(listed ^ set(report["checks"])))


def test_validate_config_reports_new_fix6_checks(base_cfg):
    report = v.validate_config(base_cfg, formal=False)
    checks = report["checks"]
    for name in ("route_logic", "final_selection", "seed17_screen",
                 "a0_sanity_band", "statistics", "governance_records"):
        assert name in checks, name
    # the two gate-0 A5 record instances (intentionally absent until T1);
    # the WP4 budget is signed from the same-env profile
    assert len(report["t1_blocked"]) == 2
    joined = " ".join(report["t1_blocked"])
    assert "per_candidate_core_time" not in joined
    assert "a0_reuse_decision_record" in joined
    assert "preregistration_signoff_record" in joined


# ---------------------------------------------------------------------------
# fix-6: exact 26-run set must survive every hardening change
# ---------------------------------------------------------------------------
def test_exact_26_run_set_unchanged(base_cfg):
    ids = [r["run_id"] for r in base_cfg["run_matrix"]["runs"]]
    assert len(ids) == 26 == v.EXPECTED_RUN_COUNT
    assert len(set(ids)) == 26  # no duplicates
    assert set(ids) == set(v.EXPECTED_RUN_IDS)
    # stage/class/dose shape still matches the frozen table row for row
    for run in base_cfg["run_matrix"]["runs"]:
        expected = v.EXPECTED_RUN_TABLE[run["run_id"]]
        for field, want in expected.items():
            got = run.get(field)
            if isinstance(want, list):
                assert list(got or ()) == want, (run["run_id"], field)
            else:
                assert got == want, (run["run_id"], field)


def test_seed17_screen_is_not_h3_claim_in_config(base_cfg):
    pred = base_cfg["evaluation_contract"]["seed17_screen_pass_predicate"]
    assert pred["definition_name"] == "seed17_screen_pass"
    assert "不构成 H3" in pred["scope"] or "NOT a three-seed H3" in pred["scope"]
    assert "三 seed" not in pred["definition_name"]


# ---------------------------------------------------------------------------
# fix-6: template mirrors for tiers / route logic / final selection
# ---------------------------------------------------------------------------
def test_parity_template_rows_carry_source_tier_and_updates():
    parity = json.loads(
        (TPL_DIR / "parity_certificate.template.json").read_text(
            encoding="utf-8")
    )
    for ck in parity["checkpoints"]:
        tier, run_updates, checkpoint_update = v.PARITY_ASSET_TIERS[ck["id"]]
        assert ck["source_tier"] == tier, ck["id"]
        assert ck["run_updates"] == run_updates, ck["id"]
        assert ck["checkpoint_update"] == checkpoint_update, ck["id"]
        assert ck["training_updates"] == run_updates, ck["id"]
        assert ck["checkpoint_update"] <= ck["run_updates"], ck["id"]
        assert ck["required"] is True
        # gate-0 A6: the source wording names the RUN dose and the
        # checkpoint update separately — never "final (37,900 updates)"
        assert "final (37,900 updates)" not in ck["source"], ck["id"]
        assert "checkpoint update" in ck["source"], ck["id"]
    by_id = {ck["id"]: ck for ck in parity["checkpoints"]}
    assert "best inner-val checkpoint from a 37,900-update run" in \
        by_id["V541_E0_S17"]["source"]
    assert "checkpoint update 37000" in by_id["V541_E0_S17"]["source"]
    assert "checkpoint update 36400" in by_id["V541_E0_S29"]["source"]
    assert "checkpoint update 37800" in by_id["V541_E0_S43"]["source"]
    policy = parity["asset_policy"]
    assert policy["all_six_required"] is True
    assert policy["smoke_checkpoints_forbidden"] is True
    assert policy["non_pinned_checkpoints_forbidden"] is True
    note = policy["smoke_paths_rejected_note"]
    assert "37,900" in note and "15,400" in note
    assert "37000/36400/37800" in note
    assert "15400/15300/15400" in note
    assert "非钉死" in note or "non-pinned" in note


def test_parity_template_gpu_caps_and_inconclusive_rule():
    parity = json.loads(
        (TPL_DIR / "parity_certificate.template.json").read_text(
            encoding="utf-8")
    )
    gpu = parity["gpu_tolerance"]
    assert gpu["atol"] == 1e-6 == v.GPU_ATOL
    assert gpu["rtol"] == 1e-5 == v.GPU_RTOL
    assert "hard caps" in gpu["hard_caps_note"]
    assert "no tolerance relaxation" in gpu["hard_caps_note"].lower() or (
        "NEVER relaxed" in gpu["legacy_floor_rule"])
    assert "INCONCLUSIVE" in gpu["legacy_floor_rule"]
    assert gpu["floor_gt_cap_means"].startswith("gpu_parity_inconclusive")
    assert "no_tolerance_relaxation" in gpu["floor_gt_cap_means"]


def test_signoff_template_carries_route_logic_and_final_selection():
    rules = _signoff_template()["gate_preregistration"]
    h3 = rules["H3_rule"]
    assert v.MOVE_PASS_LOGIC in h3
    assert v.PROCESS_PASS_LOGIC in h3
    selection = rules["final_selection_rule"]
    # each frozen tie-break step appears in the template prose, in order
    step_prose = [
        "route priority",
        "Holm-adjusted p 最小",
        "effect margin 最大",
        "core_time_ratio 较小",
        "active backbone params 较小",
        "inner-val total loss 较小",
        "candidate_id 字典序",
    ]
    positions = [selection.index(tok) for tok in step_prose]
    assert positions == sorted(positions), positions
    for margin in v.FINAL_EFFECT_MARGIN.values():
        import re as _re
        assert margin.replace("relative_effect", "MOVE direction").replace(
            "relative_improvement", "MOVE RMSE") in selection or (
            margin in selection) or (
            # prose uses the same numbers
            all(num in selection for num in _re.findall(r"[\d.]+", margin))
        ), margin
    assert "不使用 dev" in selection
    assert "H3-qualified only" in selection
    assert "no_backbone_winner" in selection
    assert "A5-C activation 仅用被选 candidate" in selection
    screen = rules["seed17_screen_pass_rule"]
    assert "(direction_pass AND rmse_non_inferior)" in screen
    assert "(auprc_pass AND q_non_inferior)" in screen
    assert v.A0_SANITY_BAND_GATE in screen


def test_runplan_pin_final_selection_mirrors_config():
    pin = json.loads(
        (TPL_DIR / "runplan_pin.template.json").read_text(encoding="utf-8")
    )
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rule = pin["final_selection_rule"]
    cfg_rule = cfg["evaluation_contract"]["final_selection_rule"]
    assert rule["definition_name"] == cfg_rule["definition_name"]
    assert tuple(rule["tie_break_order"]) == tuple(cfg_rule["tie_break_order"])
    assert dict(rule["effect_margin_definition"]) == dict(
        cfg_rule["effect_margin_definition"])
    assert rule["dev_used"] is False
    assert rule["report_all_qualified"] is True
    assert rule["rule_block_in_statistics_sha"] is True
    assert rule["a5_c_activation_uses_selected_candidate_only"] is True
    assert rule["route_logic_ref"] == "evaluation_contract.route_logic"
    contract = pin["statistics_contract"]
    assert v.MOVE_PASS_LOGIC in contract["route_logic_ref"]
    assert v.PROCESS_PASS_LOGIC in contract["route_logic_ref"]


def test_runplan_pin_t0_sanity_band_before_t1():
    pin = json.loads(
        (TPL_DIR / "runplan_pin.template.json").read_text(encoding="utf-8")
    )
    nominal = pin["run_classes"]["nominal"]
    assert "V542-A0-sanity-band" in nominal["description"]
    assert nominal["description"].index("T0") < nominal["description"].index(
        "T1")
    conditional = pin["run_classes"]["conditional"]
    assert v.A0_SANITY_BAND_GATE in " ".join(conditional["launch_requires"])
    top2 = conditional["top2_preregistration"]
    assert v.A0_SANITY_BAND_GATE in top2["freeze_requires"]
    assert "A0_S17" in top2["freeze_requires"]


def test_signoff_template_a0_sanity_band_block_frozen():
    a0 = _signoff_template()["a0_baseline_decision"]
    assert a0["a0_sanity_band_gate"] == v.A0_SANITY_BAND_GATE
    assert "MUST pass before any T1" in a0["a0_sanity_band_order"]
    assert "sanity-exception signing forbidden" in a0["a0_sanity_band_breach"]
    assert "seed-mean" in a0["a0_sanity_band_statistic"]
    # gate-0 A8: the template must name the identity layer per defect class
    # (no source_tree_fingerprint-only rule)
    for token in ("source_tree_fingerprint", "data_manifest_sha256",
                  "init_mapping_sha256", "process_head_anchor_sha256"):
        assert token in a0["a0_sanity_band_breach"], token


def test_a0_reuse_template_breach_names_identity_layer_per_defect():
    """gate-0 A8: the a0_reuse template mirrors the per-defect-class rule."""
    reuse = json.loads(
        (TPL_DIR / "a0_reuse_decision.template.json").read_text(
            encoding="utf-8"))
    step3 = reuse["sanity_band_vs_v541_e0"]["breach_procedure"][2]
    for token in ("source_tree_fingerprint", "data_manifest_sha256",
                  "init_mapping_sha256", "process_head_anchor_sha256",
                  "invalid_run"):
        assert token in step3, token
    note = reuse["sanity_band_vs_v541_e0"]["note"]
    for token in ("source_tree_fingerprint", "data_manifest_sha256",
                  "init_mapping_sha256", "process_head_anchor_sha256"):
        assert token in note, token


def test_plan_document_carries_frozen_route_logic():
    """docs-level drift guard: the plan must restate both branch logics."""
    plan = (REPO_ROOT / "docs" / "plans" /
            "PES2TS_V5.4.2_Backbone_50K_详细阶段实施方案_v1.0_20261029.md")
    text = plan.read_text(encoding="utf-8")
    assert v.MOVE_PASS_LOGIC in text
    assert v.PROCESS_PASS_LOGIC in text
    assert "final_selection_rule" in text
    assert v.A0_SANITY_BAND_GATE in text
    assert "20261029" in text


def test_adr_document_carries_frozen_route_logic():
    adr = (REPO_ROOT / "docs" / "design" / "decisions" /
           "ADR-0033-V5.4.2-Backbone接口与预注册治理.md")
    text = adr.read_text(encoding="utf-8")
    assert v.MOVE_PASS_LOGIC in text
    assert v.PROCESS_PASS_LOGIC in text
    assert "final_selection_rule" in text
    assert v.A0_SANITY_BAND_GATE in text
    assert "main_session_owner" in text
    assert "oracle_gate0_reviewer" in text


# ---------------------------------------------------------------------------
# gate-0 A2 CLOSED (2026-10-30, P3 lane D): seed17 screen H2 guard is an
# EXACT frozen string + a structured numeric threshold block
# ---------------------------------------------------------------------------
def test_seed17_h2_no_breach_exact_frozen_in_config(base_cfg):
    pred = base_cfg["evaluation_contract"]["seed17_screen_pass_predicate"]
    assert pred["h2_no_breach"] == v.H2_NO_BREACH_EXACT
    assert dict(pred["h2_no_breach_thresholds"]) == v.H2_NO_BREACH_THRESHOLDS


@pytest.mark.parametrize("tok", ["<=2%", "<=0.01", "sign", "dead",
                                 "fixed-coverage direction"])
def test_seed17_h2_guard_tokens_frozen(base_cfg, tok):
    """Every frozen guard token is part of the EXACT string (not presence)."""
    def _bad(c, token=tok):
        text = c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "h2_no_breach"]
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "h2_no_breach"] = text.replace(token, "X")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "h2_no_breach" in exc.value.detail
    assert v.H2_NO_BREACH_EXACT in exc.value.detail


def test_seed17_h2_sign_threshold_near_miss_rejected(base_cfg):
    """Oracle §2.2 near-miss: sign <=0.02 while dead still supplies <=0.01."""
    def _bad(c):
        pred = c["evaluation_contract"]["seed17_screen_pass_predicate"]
        pred["h2_no_breach"] = pred["h2_no_breach"].replace(
            "sign 下降 <=0.01", "sign 下降 <=0.02")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "h2_no_breach" in exc.value.detail


@pytest.mark.parametrize(
    "key,value",
    [("sign_abs_drop_max", 0.02), ("dead_abs_increase_max", 0.02),
     ("rmse_zero_rel_degradation_max", 0.10),
     ("fixed_coverage_rel_degradation_max", 0.05)])
def test_seed17_h2_structured_threshold_drift_rejected(
        base_cfg, key, value):
    def _bad(c, k=key, v_=value):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "h2_no_breach_thresholds"][k] = v_

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert key in exc.value.detail


def test_seed17_h2_structured_block_missing_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"].pop(
            "h2_no_breach_thresholds")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "h2_no_breach_thresholds" in exc.value.detail


def test_seed17_h2_thresholds_must_mirror_signed_thresholds(base_cfg):
    """Structured H2 thresholds must equal thresholds.* (100x for pct)."""
    def _bad(c):
        c["thresholds"]["sign_drop_abs_max"] = 0.02
        c["thresholds"]["signoff"]["sign_drop_abs_max"] = 0.02

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "sign_abs_drop_max" in exc.value.detail
    assert "sign_drop_abs_max" in exc.value.detail


def test_seed17_run_status_requires_finite_metrics(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "run_status"] = "run 已启动"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "run_status" in exc.value.detail


def test_seed17_matched_a0_requires_specific_run_id(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "matched_a0_dependency"] = "matched A0 任意 seed 已完成 且指标有限"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "V542_A0_S17" in exc.value.detail


def test_seed17_screen_hard_gate_cost_is_ratio_2(base_cfg):
    """cost_gate in the screen means core_time_ratio <= 2.0."""
    pred = base_cfg["evaluation_contract"]["seed17_screen_pass_predicate"]
    assert "cost_gate" in pred["hard_gates_all_pass"]
    assert base_cfg["thresholds"]["max_core_time_ratio"] == 2.0
    # the A5 supplement reuses the SAME frozen screen predicate
    sup = base_cfg["run_matrix"]["a5_supplement"]
    assert sup["screen_predicate_ref"] == v.SCREEN_PREDICATE_REF


def test_seed17_predicate_ref_shared_by_top2_and_a5(base_cfg):
    """Both consumers point at the ONE frozen predicate."""
    ref = v.SCREEN_PREDICATE_REF
    top2 = base_cfg["run_matrix"]["top2_preregistration"]
    sup = base_cfg["run_matrix"]["a5_supplement"]
    assert top2["seed17_screen_pass_ref"] == ref
    assert sup["screen_predicate_ref"] == ref
    assert sup["condition"] == "a5_full_dose_complete_and_seed17_screen_pass"
    assert sup["parent_gate"] == sup["condition"]


# ---------------------------------------------------------------------------
# gate-0 A5 closure (2026-10-30, P3 lane D): T1 record gates are
# machine-checked governance records
# ---------------------------------------------------------------------------
def test_t1_start_requires_includes_two_new_record_requirements(base_cfg):
    requires = base_cfg["governance"]["authorization"]["t1_start_requires"]
    assert "a0_reuse_decision_record" in requires
    assert "preregistration_signoff_record" in requires
    # the 9 canonical registry gate IDs are unchanged
    assert v.T1_CANONICAL_GATES == (
        "V542-H0-legality", "V542-H1-resource", "V542-WP1-interface-parity",
        "V542-WP2-candidates", "V542-smoke-200", "V542-profile",
        "V542-diagnostic-2000", "V542-A0-sanity-band", "V542-budget-signoff")
    assert v.T1_RECORD_REQUIREMENTS == (
        "runplan_pin_record", "a0_reuse_decision_record",
        "preregistration_signoff_record")


@pytest.mark.parametrize(
    "record", ["a0_reuse_decision_record", "preregistration_signoff_record"])
def test_t1_start_requires_dropping_new_record_rejected(base_cfg, record):
    def _bad(c, rid=record):
        c["governance"]["authorization"]["t1_start_requires"] = [
            f for f in c["governance"]["authorization"]["t1_start_requires"]
            if f != rid]

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert record in exc.value.detail


@pytest.mark.parametrize(
    "alias",
    ["a0_reuse_decision", "preregistration_signoff", "record_a0_reuse_decision"],
)
def test_t1_start_requires_record_aliases_rejected(base_cfg, alias):
    """Record requirements are exact IDs — aliases are not whitelisted."""
    def _bad(c, entry=alias):
        req = c["governance"]["authorization"]["t1_start_requires"]
        req[req.index("a0_reuse_decision_record")] = entry

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert alias in exc.value.detail


def test_governance_records_block_frozen_in_config(base_cfg):
    records = base_cfg["governance"]["records"]
    assert set(records) == set(v.GOVERNANCE_RECORDS)
    for rid, spec in v.GOVERNANCE_RECORDS.items():
        entry = records[rid]
        assert entry["template"] == spec["template"], rid
        assert entry["instance"] == spec["instance"], rid
        assert entry["sha_rule"] == v.RECORD_SHA_RULE == (
            "sha256_of_canonical_file_bytes"), rid
        assert tuple(entry["required_fields"]) == tuple(
            spec["required_fields"]), rid


def test_governance_records_block_missing_rejected(base_cfg):
    def _bad(c):
        c["governance"].pop("records")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "governance.records" in exc.value.detail


def test_governance_records_missing_entry_rejected(base_cfg):
    def _bad(c):
        c["governance"]["records"].pop("a0_reuse_decision_record")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "a0_reuse_decision_record" in exc.value.detail


@pytest.mark.parametrize(
    "field,value",
    [("instance", "outputs/pesn/v542_backbone_50k/plan/a0_decision.json"),
     ("template", "config/campaigns/v542_templates/other.template.json"),
     ("sha_rule", "sha256_of_reencoded_json")],
)
def test_governance_records_spec_drift_rejected(base_cfg, field, value):
    def _bad(c, f=field, v_=value):
        c["governance"]["records"]["a0_reuse_decision_record"][f] = v_

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert field in exc.value.detail


def test_prereg_missing_records_are_blockers_not_rejections(
        base_cfg, tmp_path):
    """WP0 absence stays by design: t1_blocked reasons, never a raise."""
    report = v.validate_config(base_cfg, formal=False, record_root=tmp_path)
    assert report["ok"] is True
    reasons = [r for r in report["t1_blocked"] if "record " in r]
    assert len(reasons) == 2
    assert any("a0_reuse_decision_record" in r for r in reasons)
    assert any("preregistration_signoff_record" in r for r in reasons)


def test_prereg_record_blocker_cleared_when_instance_valid(
        base_cfg, tmp_path):
    _write_valid_records(tmp_path)
    report = v.validate_config(base_cfg, formal=False, record_root=tmp_path)
    assert report["ok"] is True
    assert not any("record " in r for r in report["t1_blocked"])
    # budget is signed and records are valid: no remaining T1 blocker
    assert report["t1_blocked"] == []
    assert report["t1_config_ready"] is True


def test_formal_missing_record_instances_typed_rejection(base_cfg, tmp_path):
    """--formal with missing record instances = typed V542_RECORD_MISSING."""
    cfg = _mut(base_cfg, _fill_budget)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISSING"
    assert "a0_reuse_decision_record" in exc.value.detail
    assert "preregistration_signoff_record" in exc.value.detail


def test_formal_unreadable_record_typed_rejection(base_cfg, tmp_path):
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    plan = tmp_path / "outputs" / "pesn" / "v542_backbone_50k" / "plan"
    (plan / "a0_reuse_decision.json").write_text(
        "{not json", encoding="utf-8")
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISSING"
    assert "unreadable" in exc.value.detail


def test_formal_none_required_field_typed_rejection(base_cfg, tmp_path):
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, "a0_reuse_decision.json",
        lambda p: p.__setitem__("signed_by", None))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISSING"
    assert "signed_by" in exc.value.detail


def test_formal_decision_drift_typed_mismatch(base_cfg, tmp_path):
    """instance decision != retrain_a0_x3 = typed V542_RECORD_MISMATCH."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, "a0_reuse_decision.json",
        lambda p: p.__setitem__("decision", "reuse_e0_x3"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "retrain_a0_x3" in exc.value.detail


def test_formal_signoff_threshold_drift_typed_mismatch(base_cfg, tmp_path):
    """signoff thresholds that drift from config = typed mismatch."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)

    def _drift(p):
        p["thresholds_signed_frozen"]["sign_drop_abs_max"] = 0.02

    _mutate_record(tmp_path, "preregistration_signoff.json", _drift)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "sign_drop_abs_max" in exc.value.detail


def test_formal_decision_runs_must_mirror_config(base_cfg, tmp_path):
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, "a0_reuse_decision.json",
        lambda p: p.__setitem__("runs", ["V542_A0_S17"]))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "runs" in exc.value.detail


# ---------------------------------------------------------------------------
# gate-3 B2/C2 (2026-10-30, P3 remediation lane R2): the record gate must
# mirror the frozen config and verify the declared SHA.  Oracle gate-3
# probes (wrong config_path / empty signed_by_roles used to PASS --formal
# with a valid budget) are now typed rejections.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "record_name",
    ["a0_reuse_decision.json", "preregistration_signoff.json"])
def test_formal_wrong_config_path_typed_mismatch(base_cfg, tmp_path,
                                                 record_name):
    """Oracle gate-3 B2 probe: config_path=WRONG.yaml used to pass."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, record_name,
        lambda p: p.__setitem__(
            "config_path", "config/campaigns/WRONG.yaml"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "config_path" in exc.value.detail
    assert v.CAMPAIGN_CONFIG_REL in exc.value.detail


def test_formal_empty_signed_by_roles_typed_mismatch(base_cfg, tmp_path):
    """Oracle gate-3 B2 probe: signed_by_roles={} used to pass."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, "preregistration_signoff.json",
        lambda p: p.__setitem__("signed_by_roles", {}))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "signed_by_roles" in exc.value.detail


def test_formal_wrong_role_set_typed_mismatch(base_cfg, tmp_path):
    """A role set missing/extra vs the frozen keys = typed mismatch."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)

    def _drift(p):
        roles = p["signed_by_roles"]
        roles.pop("oracle_reviewer")
        roles["rogue_role"] = "attacker"

    _mutate_record(tmp_path, "preregistration_signoff.json", _drift)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "oracle_reviewer" in exc.value.detail
    assert "rogue_role" in exc.value.detail


def test_formal_wrong_role_binding_typed_mismatch(base_cfg, tmp_path):
    """statistics_owner must be main_session_owner — drift = mismatch."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, "preregistration_signoff.json",
        lambda p: p["signed_by_roles"].__setitem__(
            "statistics_owner", "oracle_gate0_reviewer"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "statistics_owner" in exc.value.detail


def test_formal_oracle_as_signatory_typed_mismatch(base_cfg, tmp_path):
    """No signatory role may name the Oracle (fix-6 C4, now instance-checked)."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, "preregistration_signoff.json",
        lambda p: p["signed_by_roles"].__setitem__(
            "training_signoff", "oracle_gate0_reviewer"))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "training_signoff" in exc.value.detail


@pytest.mark.parametrize(
    "record_name,schema_field",
    [("a0_reuse_decision.json", "pes2ts_v542_a0_reuse_decision_v2"),
     ("preregistration_signoff.json", "pes2ts_v542_preregistration_signoff_v2")])
def test_formal_record_schema_drift_typed_mismatch(
        base_cfg, tmp_path, record_name, schema_field):
    """record `schema` must equal the template schema id (gate-3 B2)."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, record_name,
        lambda p: p.__setitem__("schema", schema_field))
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "schema" in exc.value.detail


def test_formal_missing_record_sha_typed_missing(base_cfg, tmp_path):
    """A record without record_sha256 = V542_RECORD_MISSING (gate-3 C2)."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, "a0_reuse_decision.json",
        lambda p: p.pop("record_sha256"), seal=False)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISSING"
    assert "record_sha256" in exc.value.detail


def test_formal_incorrect_record_sha_typed_mismatch(base_cfg, tmp_path):
    """A declared digest that does not match the canonical digest =
    V542_RECORD_MISMATCH (gate-3 C2)."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    _mutate_record(
        tmp_path, "a0_reuse_decision.json",
        lambda p: p.__setitem__("record_sha256", "0" * 64), seal=False)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "record_sha256" in exc.value.detail


def test_record_sha_uses_identity_canonical_recipe(base_cfg, tmp_path):
    """The digest the validator checks = identity.canonical_sha256 of the
    payload without record_sha256 — no second recipe exists (gate-3 C2)."""
    import hashlib

    from pes2ts_core.pesn.models.v542.identity import (
        canonical_bytes,
        canonical_sha256,
    )

    _write_valid_records(tmp_path, base_cfg)
    plan = tmp_path / "outputs" / "pesn" / "v542_backbone_50k" / "plan"
    payload = json.loads(
        (plan / "a0_reuse_decision.json").read_text(encoding="utf-8"))
    body = {k: v_ for k, v_ in payload.items() if k != "record_sha256"}
    assert payload["record_sha256"] == canonical_sha256(body)
    assert payload["record_sha256"] == hashlib.sha256(
        canonical_bytes(body)).hexdigest()
    assert payload["record_sha256"] == v._record_canonical_sha256(payload)


def test_formal_record_tamper_after_signing_detected(base_cfg, tmp_path):
    """Modify a field after signing WITHOUT updating record_sha256 — the
    stale digest is caught (Oracle gate-3 near-miss)."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    plan = tmp_path / "outputs" / "pesn" / "v542_backbone_50k" / "plan"
    path = plan / "a0_reuse_decision.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["signed_by"] = "attacker"  # tamper, digest stays stale
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "record_sha256" in exc.value.detail


def test_formal_threshold_encoding_drift_typed_mismatch(base_cfg, tmp_path):
    """4 vs 4.0 passes `==` but not canonical-byte equality (gate-3 B2)."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)

    def _drift(p):
        p["thresholds_signed_frozen"]["top2_max_runs"] = 4.0

    _mutate_record(tmp_path, "preregistration_signoff.json", _drift)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert exc.value.code == "V542_RECORD_MISMATCH"
    assert "top2_max_runs" in exc.value.detail


def test_prereg_record_mirror_drift_is_blocker_not_rejection(
        base_cfg, tmp_path):
    """At prereg, config_path drift stays a t1_blocked reason (WP0 green)."""
    _write_valid_records(tmp_path, base_cfg)
    _mutate_record(
        tmp_path, "a0_reuse_decision.json",
        lambda p: p.__setitem__(
            "config_path", "config/campaigns/WRONG.yaml"))
    report = v.validate_config(base_cfg, formal=False, record_root=tmp_path)
    assert report["ok"] is True
    assert any("a0_reuse_decision_record" in r and "config_path" in r
               for r in report["t1_blocked"])


def test_prereg_empty_roles_is_blocker_not_rejection(base_cfg, tmp_path):
    _write_valid_records(tmp_path, base_cfg)
    _mutate_record(
        tmp_path, "preregistration_signoff.json",
        lambda p: p.__setitem__("signed_by_roles", {}))
    report = v.validate_config(base_cfg, formal=False, record_root=tmp_path)
    assert report["ok"] is True
    assert any("preregistration_signoff_record" in r and
               "signed_by_roles" in r for r in report["t1_blocked"])


def test_runplan_pin_is_runner_verified_not_validator(base_cfg, tmp_path):
    """runplan_pin_record is WP5-runner-verified: the validator must NOT
    read the pin instance even when it exists (gate-3 B2 clarification)."""
    cfg = _mut(base_cfg, _fill_budget)
    _write_valid_records(tmp_path, cfg)
    plan = tmp_path / "outputs" / "pesn" / "v542_backbone_50k" / "plan"
    (plan / "runplan.pin.json").write_text(
        json.dumps({"schema": "pes2ts_v542_runplan_pin_v1",
                    "config_path": "config/campaigns/WRONG.yaml"}),
        encoding="utf-8")
    formal = v.validate_config(cfg, formal=True, record_root=tmp_path)
    assert formal["ok"] and formal["t1_blocked"] == []
    assert v.GOVERNANCE_RECORDS["runplan_pin_record"][
        "machine_checked"] is False
    assert "WP5" in v.GOVERNANCE_RECORDS["runplan_pin_record"]["verified_by"]


def test_governance_records_required_fields_include_schema_and_sha(base_cfg):
    """gate-3 B2/C2: the frozen spec requires schema + record_sha256."""
    for rid in ("a0_reuse_decision_record",
                "preregistration_signoff_record"):
        fields = v.GOVERNANCE_RECORDS[rid]["required_fields"]
        assert "schema" in fields, rid
        assert "record_sha256" in fields, rid
        assert "config_path" in fields, rid
    cfg_fields = base_cfg["governance"]["records"][
        "a0_reuse_decision_record"]["required_fields"]
    assert tuple(cfg_fields) == tuple(
        v.GOVERNANCE_RECORDS["a0_reuse_decision_record"]["required_fields"])


def test_prereg_mismatched_record_is_blocker_not_rejection(
        base_cfg, tmp_path):
    _write_valid_records(tmp_path, base_cfg)
    _mutate_record(
        tmp_path, "a0_reuse_decision.json",
        lambda p: p.__setitem__("decision", "reuse_e0_x3"))
    report = v.validate_config(base_cfg, formal=False, record_root=tmp_path)
    assert report["ok"] is True
    assert any("a0_reuse_decision_record" in r and "retrain_a0_x3" in r
               for r in report["t1_blocked"])


def test_record_sha_rule_is_canonical_file_bytes():
    for rid, spec in v.GOVERNANCE_RECORDS.items():
        assert spec["sha_rule"] == v.RECORD_SHA_RULE, rid
    assert v.RECORD_SHA_RULE == "sha256_of_canonical_file_bytes"


def test_signoff_template_t1_record_requirements_mirror_config(base_cfg):
    signoff = _signoff_template()
    records = signoff["t1_record_requirements"]
    cfg_records = base_cfg["governance"]["records"]
    for rid in ("runplan_pin_record", "a0_reuse_decision_record",
                "preregistration_signoff_record"):
        assert records[rid]["template"] == cfg_records[rid]["template"], rid
        assert records[rid]["instance"] == cfg_records[rid]["instance"], rid
    ready = " ".join(signoff["t1_config_ready_requires"])
    assert "a0_reuse_decision.json" in ready
    assert "preregistration_signoff.json" in ready


def test_runplan_pin_final_selection_mirrors_structured_fields(base_cfg):
    pin = json.loads(
        (TPL_DIR / "runplan_pin.template.json").read_text(encoding="utf-8")
    )
    rule = pin["final_selection_rule"]
    cfg_rule = base_cfg["evaluation_contract"]["final_selection_rule"]
    assert rule["eligible_condition"] == cfg_rule["eligible_condition"]
    assert rule["single_candidate_rule_id"] == cfg_rule[
        "single_candidate_rule_id"]
    assert rule["no_candidate_rule_id"] == cfg_rule["no_candidate_rule_id"]


# ---------------------------------------------------------------------------
# gate-0 A2 closure (2026-10-30, P3 lane D): adversarial token-preserving
# near-misses from Oracle gate-0-final §1 probe table / gate-1 §2.2 are now
# typed rejections (they all used to PASS prereg)
# ---------------------------------------------------------------------------
def test_a2_eligible_not_h3_qualified_near_miss_rejected(base_cfg):
    """'NOT H3-qualified; all candidates' used to pass the substring check."""
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"]["eligible"] = (
            "NOT H3-qualified; all candidates")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "eligible" in exc.value.detail


def test_a2_eligible_condition_drift_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "eligible_condition"] = "ALL_CANDIDATES"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "eligible_condition" in exc.value.detail
    assert "H3_QUALIFIED_ONLY" in exc.value.detail


def test_a2_single_candidate_near_miss_rejected(base_cfg):
    """'1 个候选时不选 Backbone' used to pass the token check."""
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "single_candidate_rule"] = "1 个候选时不选 Backbone"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "single_candidate_rule" in exc.value.detail


def test_a2_no_candidate_near_miss_rejected(base_cfg):
    """'0 个候选则 A0 no_backbone_winner but rerun' used to pass."""
    def _bad(c):
        c["evaluation_contract"]["final_selection_rule"][
            "no_candidate_rule"] = (
            "0 个候选则 A0 no_backbone_winner but rerun")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "no_candidate_rule" in exc.value.detail


@pytest.mark.parametrize(
    "field", ["single_candidate_rule_id", "no_candidate_rule_id"])
def test_a2_structured_branch_ids_frozen(base_cfg, field):
    def _bad(c, f=field):
        c["evaluation_contract"]["final_selection_rule"][f] = "drifted_id"

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert field in exc.value.detail


def test_a2_final_selection_structured_fields_frozen_in_config(base_cfg):
    rule = base_cfg["evaluation_contract"]["final_selection_rule"]
    assert rule["eligible"] == v.FINAL_SELECTION_ELIGIBLE_EXACT
    assert rule["eligible_condition"] == v.FINAL_SELECTION_ELIGIBLE_CONDITION
    assert rule["single_candidate_rule"] == v.FINAL_SELECTION_SINGLE_RULE_EXACT
    assert rule["single_candidate_rule_id"] == v.FINAL_SELECTION_SINGLE_RULE_ID
    assert rule["no_candidate_rule"] == (
        v.FINAL_SELECTION_NO_CANDIDATE_RULE_EXACT)
    assert rule["no_candidate_rule_id"] == (
        v.FINAL_SELECTION_NO_CANDIDATE_RULE_ID)
    assert rule["dev_used"] is False
    assert rule["report_all_qualified"] is True


def test_a2_core_loop_not_verbatim_near_miss_rejected(base_cfg):
    """'not verbatim; rewritten forward' used to pass the substring check."""
    def _bad(c):
        c["backbone_contract"]["a0_migration"]["core_loop"] = (
            "not verbatim; rewritten forward")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_INTERFACE"
    assert v.A0_CORE_LOOP in exc.value.detail


def test_a2_core_loop_frozen_enum_in_config(base_cfg):
    core_loop = base_cfg["backbone_contract"]["a0_migration"]["core_loop"]
    assert core_loop == v.A0_CORE_LOOP == "super().forward_verbatim"


def test_a2_h3_eligible_token_soup_string_rejected(base_cfg):
    """'H2 diagnostic 三 seed winner-eligible; NOT three seed' rejected."""
    def _bad(c):
        c["evaluation_contract"]["statistics"]["h3_eligible_candidates"] = (
            "H2 diagnostic 三 seed winner-eligible; NOT three seed")

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_MISSING_FIELD"
    assert "h3_eligible_candidates" in exc.value.detail


def test_a2_h2_sign_threshold_structured_near_miss_rejected(base_cfg):
    def _bad(c):
        c["evaluation_contract"]["seed17_screen_pass_predicate"][
            "h2_no_breach_thresholds"]["sign_abs_drop_max"] = 0.02

    cfg = _mut(base_cfg, _bad)
    with pytest.raises(v.V542SchemaError) as exc:
        v.validate_config(cfg)
    assert exc.value.code == "V542_RUN_MATRIX"
    assert "sign_abs_drop_max" in exc.value.detail
