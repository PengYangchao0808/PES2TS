"""V5.4.3 WP9 — bundle build/verify + parity certificate instance tests
(``scripts/agent_ops/v543_bundle.py``).

CPU-fast and synthetic: a miniature campaign root is built in ``tmp_path``, so
the collection plan, the sha256+size manifest, the five-report skeleton, the
write-once/immutability discipline, the ``verify`` refusals and the certificate
field mapping are all exercised WITHOUT data, WITHOUT torch checkpoints and
WITHOUT touching the real campaign artifacts.

What is proven here (plan §13.3 :1329–1375, §11.2 ``v543_bundle.py`` row,
§11.5 ``test_v543_bundle_strict_load`` / ``test_v543_legacy_files_untouched``):

* the bundle tree follows the §13.3 component layout and every collected file
  carries sha256 + byte size + its source path;
* inputs that do not exist yet are ``pending`` lines (with the
  ``required_for_final`` flag) instead of silent omissions — an intermediate
  bundle is explicitly allowed to be partial;
* the five §13.3 reports are present with a status and their missing evidence
  named, and the controller policy / encoder manifest / data provenance blocks
  are generated from frozen plan text and pinned config, never invented;
* a bundle is write-once (a second build into the same directory is
  ``V543_BUNDLE_EXISTS``) and immutable is declared (``IMMUTABLE.json``);
* ``verify`` re-hashes the whole tree and refuses a changed digest, a missing
  file, an extra file and a manifest that no longer matches its frozen
  declaration;
* the parity certificate generator fills the WP0 template from measured blocks
  (``fill_certificate``) and refuses a config that pins no checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops import v543_bundle as b  # noqa: E402

CAMPAIGN_ROOT_NAME = "outputs/pesn/v543_dual_decoder_50k"


# ---------------------------------------------------------------------------
# fixtures: a miniature campaign root (no real artifacts involved)
# ---------------------------------------------------------------------------
def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")
    return path


@pytest.fixture()
def campaign_root(tmp_path: Path) -> Path:
    root = tmp_path / CAMPAIGN_ROOT_NAME
    _write(root / "plan/runplan.json", {"schema": "synthetic_runplan"})
    _write(root / "plan/runplan.pin.json", {"schema": "synthetic_pin"})
    _write(root / "plan/preregistration_signoff.json", {"schema": "signoff"})
    _write(root / "audit/wp3_go_no_go_decision.json", {"decision": "go"})
    _write(root / "audit/s0_probe_report.json", {"probe": "ok"})
    _write(root / "reports/track_champions.json",
           {"schema": "pesn_v543_stats_select_report_v1", "tracks": {}})
    _write(root / "reports/h2_report.json", {"schema": "h2"})
    _write(root / "reports/h3_report.json", {"schema": "h3"})
    _write(root / "reports/metrics_table.json", {"schema": "metrics"})
    _write(root / "run_ledger.jsonl", '{"run_id": "V543_M1_S17"}\n')
    _write(root / "runs/V543_M1_S17/checkpoint_best.pt", "weights-m1")
    _write(root / "runs/V543_M1_S17/metrics.json", {"schema": "run_metrics"})
    _write(root / "runs/V543_P2_S17/checkpoint_best.pt", "weights-p2")
    _write(root / "profile/profile_report.jsonl", '{"profile": 1}\n')
    _write(root / "smoke/smoke_report.jsonl", '{"smoke": 1}\n')
    return root


def _build(campaign_root: Path, *, immutability: bool = False):
    bundle_root = campaign_root / b.BUNDLES_SUBDIR / (
        b.BUNDLE_DIR_PREFIX + "TEST")
    manifest = b.build_bundle(campaign_root=str(campaign_root),
                              bundle_root=bundle_root,
                              apply_immutability=immutability)
    return bundle_root, manifest


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def test_bundle_layout_and_manifest_digests(campaign_root: Path) -> None:
    bundle_root, manifest = _build(campaign_root)
    assert manifest["schema"] == b.BUNDLE_SCHEMA
    assert manifest["bundle_version"] == "PES2TSBundleV543"
    assert manifest["counts"]["n_files"] >= 10
    for entry in manifest["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
        assert (bundle_root / entry["path"]).is_file()
        assert entry["origin"] in ("copied", "generated")
    paths = {entry["path"] for entry in manifest["files"]}
    # §13.3 component layout
    assert "encoder/a0_checkpoint_manifest.json" in paths
    assert "controller/legal_endpoint_policy.json" in paths
    assert "provenance/data/split_sampler_provenance.json" in paths
    assert "provenance/ledger/run_ledger.jsonl" in paths
    assert "provenance/governance/campaign_config.yaml" in paths
    assert "provenance/governance/ADR-0034.md" in paths
    assert "BUNDLE_REPORT.json" in paths and "BUNDLE_REPORT.md" in paths
    # runs are routed by decoder id into the two expert buckets
    assert any(path.startswith("move_expert/runs/V543_M1_S17/")
               for path in paths)
    assert any(path.startswith("process_expert/runs/V543_P2_S17/")
               for path in paths)
    assert (bundle_root / "IMMUTABLE.json").is_file()
    declaration = json.loads((bundle_root / "IMMUTABLE.json").read_text())
    assert declaration["schema"] == b.IMMUTABLE_SCHEMA
    assert declaration["manifest_sha256"] == b.sha256_file(
        bundle_root / "manifest.json")
    assert declaration["confirm_read"] is False
    assert declaration["new_QC_calls"] == 0


def test_missing_inputs_are_pending_with_the_final_bundle_flag(
        campaign_root: Path) -> None:
    _bundle_root, manifest = _build(campaign_root)
    pending = {entry["path"]: entry for entry in manifest["pending"]}
    # not created by the fixture: these are the WP9 deliverables still missing
    assert "bridge/parity_certificate.json" in pending
    assert pending["bridge/parity_certificate.json"][
        "required_for_final"] is True
    assert "provenance/selection/final_selection.json" in pending
    assert "provenance/rollout/t7_rollout.json" in pending
    for entry in manifest["pending"]:
        assert entry["status"] == "pending"
        assert entry["reason"]


def test_five_reports_name_their_pending_evidence(campaign_root: Path) -> None:
    _bundle_root, manifest = _build(campaign_root)
    reports = manifest["reports"]
    assert set(reports["complete"]) | set(reports["pending"]) == set(
        b.FIVE_REPORTS)
    document = json.loads(
        (campaign_root / b.BUNDLES_SUBDIR / (b.BUNDLE_DIR_PREFIX + "TEST")
         / "BUNDLE_REPORT.json").read_text())
    blocks = document["five_reports"]
    assert blocks["encoder_bridge_parity"]["status"] == "partial"
    assert "certify" in blocks["encoder_bridge_parity"]["pending_reason"]
    assert blocks["integration_rollout"]["missing_evidence"] == [
        "provenance/rollout/t7_rollout.json"]
    assert blocks["move_benchmark"]["status"] == "complete"
    assert document["read_only_declared"] is False


def test_controller_policy_is_the_frozen_plan_transcription(
        campaign_root: Path) -> None:
    bundle_root, _manifest = _build(campaign_root)
    policy = json.loads((bundle_root / "controller/" /
                         "legal_endpoint_policy.json").read_text())
    assert policy["learned"] is False and policy["parameter_count"] == 0
    assert policy["no_move"]["only_reason"].startswith(
        "p_end >= tau_end")
    assert policy["no_move"]["q_may_not_appear_in_trigger_reason"] is True
    assert policy["move_flag"]["tau_end"] is None
    assert policy["not_in_model_forward"] is True


def test_encoder_manifest_records_the_pinned_checkpoint_digests(
        campaign_root: Path) -> None:
    bundle_root, _manifest = _build(campaign_root)
    block = json.loads((bundle_root / "encoder/" /
                        "a0_checkpoint_manifest.json").read_text())
    assert block["schema"] == b.ENCODER_MANIFEST_SCHEMA
    assert sorted(block["checkpoints"]) == ["17", "29", "43"]
    for entry in block["checkpoints"].values():
        assert len(entry["pinned_sha256"]) == 64
        assert entry["maps_to"].startswith("V543_M0_S")


def test_a_bundle_is_write_once(campaign_root: Path) -> None:
    bundle_root, _manifest = _build(campaign_root)
    with pytest.raises(b.BundleError) as error:
        b.build_bundle(campaign_root=str(campaign_root),
                       bundle_root=bundle_root, apply_immutability=False)
    assert error.value.code == b.CODE_EXISTS


def test_a_bundle_must_live_under_the_campaign_root(
        campaign_root: Path, tmp_path: Path) -> None:
    with pytest.raises(b.BundleError) as error:
        b.build_bundle(campaign_root=str(campaign_root),
                       bundle_root=tmp_path / "elsewhere",
                       apply_immutability=False)
    assert error.value.code == b.CODE_ROOT


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def test_verify_accepts_a_fresh_bundle(campaign_root: Path) -> None:
    bundle_root, manifest = _build(campaign_root, immutability=True)
    result = b.verify_bundle(bundle_root)
    assert result["ok"] is True
    assert result["n_files_verified"] == result["n_declared"] == (
        manifest["counts"]["n_files"])
    assert result["immutable_declared"] is True
    assert result["reports_pending"] == manifest["reports"]["pending"]


def test_verify_refuses_a_changed_digest(campaign_root: Path) -> None:
    bundle_root, manifest = _build(campaign_root)
    target = bundle_root / "provenance/ledger/run_ledger.jsonl"
    target.write_text('{"run_id": "tampered"}\n', encoding="utf-8")
    with pytest.raises(b.BundleError) as error:
        b.verify_bundle(bundle_root)
    assert error.value.code == b.CODE_VERIFY_DIGEST
    assert "run_ledger.jsonl" in str(error.value)


def test_verify_refuses_a_missing_file(campaign_root: Path) -> None:
    bundle_root, _manifest = _build(campaign_root)
    (bundle_root / "controller/legal_endpoint_policy.json").unlink()
    with pytest.raises(b.BundleError) as error:
        b.verify_bundle(bundle_root)
    assert "missing" in str(error.value)


def test_verify_refuses_an_extra_file(campaign_root: Path) -> None:
    bundle_root, _manifest = _build(campaign_root)
    (bundle_root / "provenance/extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(b.BundleError) as error:
        b.verify_bundle(bundle_root)
    assert error.value.code == b.CODE_VERIFY_EXTRA


def test_verify_refuses_a_tampered_manifest(campaign_root: Path) -> None:
    bundle_root, _manifest = _build(campaign_root)
    manifest_path = bundle_root / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["pending"] = []
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(b.BundleError) as error:
        b.verify_bundle(bundle_root)
    assert "IMMUTABLE" in str(error.value) or "manifest" in str(error.value)


def test_verify_refuses_a_missing_bundle(tmp_path: Path) -> None:
    with pytest.raises(b.BundleError) as error:
        b.verify_bundle(tmp_path / "nope")
    assert error.value.code == b.CODE_VERIFY_MISSING


# ---------------------------------------------------------------------------
# the S0-A parity certificate instance
# ---------------------------------------------------------------------------
def _measured_block(seed: int, *, status: str = "pass") -> dict:
    return {
        "id": f"V542_A0_S{seed}",
        "path": f"outputs/pesn/v542_backbone_50k/runs/V542_A0_S{seed}/"
                "checkpoint_best.pt",
        "sha256": "a" * 64,
        "pinned_sha256": "a" * 64,
        "sha_matches_pin": True,
        "maps_to": f"V543_M0_S{seed}",
        "equations": {name: True for name in (
            "u0_equals_v542_u", "z_equals_v542_z", "q0_equals_v542_q",
            "tanh_q0_raw_equals_q0", "endpoint0_equals_v542_endpoint_logit",
            "zero_residual_model_equal", "order_score_is_q")},
        "equation_count_passed": 7,
        "cpu_bitwise": True,
        "real_batch_bitwise": True,
        "max_abs_diff": {"u0": 0.0},
        "encoder_forward_count": 16,
        "hook_fire_counts": {"message_mlp[-1]": 1, "norm_h[-1]": 1,
                             "progress_head.fc2": 1},
        "state_dict_counts": {
            "payload_key_count_measured": 45,
            "expanded_target_key_count_measured": 45,
            "count_scope_of_this_audit": "expanded_target_key_count",
        },
        "real_batch": {"status": "executed", "executed": 8},
        "status": status,
    }


def test_fill_certificate_keeps_the_template_and_adds_the_blocks() -> None:
    template = json.loads(
        (REPO_ROOT / b.CERTIFICATE_TEMPLATE).read_text(encoding="utf-8"))
    blocks = [_measured_block(seed) for seed in b.CERTIFICATE_SEEDS]
    document = b.fill_certificate(template, blocks,
                                  extra={"overall_status": "pass"})
    assert document["schema"] == template["schema"]
    assert document["checkpoint_count"] == 3
    assert [entry["id"] for entry in document["checkpoints"]] == [
        "V542_A0_S17", "V542_A0_S29", "V542_A0_S43"]
    # the frozen static blocks survive the merge untouched
    assert document["zero_residual_equations"]["equation_count"] == 7
    assert document["state_dict_count_semantics"]["payload_key_count"][
        "expected"] == 43
    assert document["state_dict_count_semantics"][
        "expanded_target_key_count"]["expected"] == 45
    assert document["failure_procedure"]
    # §7.1 S0-A keeps these explicitly absent
    assert "training_comparability" in document["not_evidence_for"]


def test_generate_certificate_writes_the_instance_from_measured_blocks(
        tmp_path: Path) -> None:
    calls: list[dict] = []

    def measurer(**kwargs):
        calls.append(kwargs)
        return _measured_block(int(kwargs["seed"]))

    output = tmp_path / "plan/parity_certificate.json"
    document = b.generate_parity_certificate(
        output_path=str(output), measurer=measurer)
    assert len(calls) == 3
    assert output.is_file()
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["overall_status"] == "pass_cpu_pending_gpu_and_signoff"
    assert written["signed_by"] is None and written["signed_on"] is None
    assert len(written["config_sha256"]) == 64
    assert len(written["code_sha256"]) == 64
    assert len(written["environment_sha256"]) == 64
    assert len(written["source_manifest_sha256"]) == 64
    assert written["bridge_spec_sha256"]
    assert written["instances"]["written_by_task_brief"] == (
        b.CERTIFICATE_INSTANCE)
    # the measured 45-key payload contradicts the template's 43 expectation and
    # the deviation is REPORTED, not silently fixed
    assert any("43" in item and "45" in item for item in written["ambiguities"])
    assert document["written_sha256"] == b.sha256_file(output)


def test_generate_certificate_refuses_a_config_without_pinned_checkpoints(
        tmp_path: Path) -> None:
    config = _write(tmp_path / "cfg.yaml", "schema: synthetic\n")
    with pytest.raises(b.BundleError) as error:
        b.generate_parity_certificate(config_path=str(config),
                                      output_path=str(tmp_path / "out.json"),
                                      measurer=lambda **_: {})
    assert error.value.code == b.CODE_CERT_CHECKPOINT


def test_collection_plan_covers_the_section_13_3_surface() -> None:
    rules = b.collection_plan()
    sources = {rule.source for rule in rules}
    assert any("ADR-0034" in source for source in sources)
    assert any(source.endswith("pes2ts_v543_dual_decoder_50k.yaml")
               for source in sources)
    assert any(source.endswith("v543_config_schema.py") for source in sources)
    assert any("plan/runplan.json" in source for source in sources)
    assert any("run_ledger.jsonl" in source for source in sources)
    assert any(source.endswith("/audit/") for source in sources)
    assert any(source.endswith("/reports/") for source in sources)
    assert any(source.endswith("/runs/") for source in sources)
    assert any("parity_certificate.json" in source for source in sources)
    assert any("wp3_go_no_go_decision.json" in source for source in sources)
    assert any(rule.required_for_final is False for rule in rules)
