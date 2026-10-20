"""v2 feature-gate tests: allowlist (T1.4b), intent-off mask (T1.4c),
campaign-yaml tolerance binding, and the G1 gate report (CLI included).

Deterministic; requires torch (skipped otherwise, same policy as
``test_feature_harness.py``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip(
    "torch",
    reason="feature-gate tests need torch; run with an interpreter that has "
           "it (e.g. /tmp/opencode/venv/bin/python)",
)

from scripts.agent_ops.feature_gate.dummy_equivariant_model import (  # noqa: E402
    INTENT_FIELD_KEYS,
    PRESERVED_FIELD_KEYS,
    make_dummy_batch,
    make_dummy_model,
    make_intent_off_batch,
)
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    DEFAULT_CAMPAIGN_YAML,
    FORWARD_INPUT_ALLOWLIST,
    FORWARD_INPUT_ALLOWLIST_GROUPS,
    FORWARD_INPUT_ALLOWLIST_SOURCE,
    GATE_REPORT_TESTS,
    INTENT_OFF_MASK_CLASSES,
    PRESERVED_UNORDERED_CONTEXT_KEYS,
    RECORD_ONLY_BANNER,
    PresetContractError,
    build_gate_report,
    check_input_allowlist,
    check_input_legality,
    check_intent_off_mask,
    load_campaign_feature_contract,
    resolve_preset,
    run_feature_gates,
    verify_preset_against_yaml,
)

SEED = 1234
NORMATIVE = "30h_v1.1_normative"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture(seed: int = SEED):
    torch.manual_seed(seed)
    model = make_dummy_model(seed=seed)
    batch = make_dummy_batch(seed=seed)
    return model, batch


# ---------------------------------------------------------------------------
# T1.4b positive input allowlist
# ---------------------------------------------------------------------------

def test_allowlist_constant_cites_campaign_yaml() -> None:
    assert "feature_contract" in FORWARD_INPUT_ALLOWLIST_SOURCE
    assert "pes2ts_30h_20261020.yaml" in FORWARD_INPUT_ALLOWLIST_SOURCE
    # frozen positive set covers current geometry / atomic numbers / graph /
    # reaction-intent groups required by PD-2
    assert {"pos", "z", "key_index", "delta_b_intent", "signed_intent",
            "graph_current", "charge", "mult", "rbf12"} <= FORWARD_INPUT_ALLOWLIST
    for key in FORWARD_INPUT_ALLOWLIST:
        assert key == key.lower() or key in ("degK",)  # only degK keeps casing


def test_clean_batch_passes_allowlist() -> None:
    _, batch = _fixture()
    rep = check_input_allowlist(batch, preset=NORMATIVE)
    assert rep["passed"] is True and rep["violations"] == []
    assert rep["enforcement"] == "allowlist"
    assert set(batch) <= FORWARD_INPUT_ALLOWLIST


def test_every_dummy_batch_key_is_allowlisted() -> None:
    _, batch = _fixture()
    outside = sorted(set(batch) - FORWARD_INPUT_ALLOWLIST)
    assert outside == [], f"synthetic intent-ON batch leaks: {outside}"


@pytest.mark.parametrize("key", ["sneaky_feature", "teacher_label", "q_true"])
def test_non_allowlisted_keys_fail(key: str) -> None:
    _, batch = _fixture()
    batch[key] = torch.tensor([1.0])
    rep = check_input_allowlist(batch, preset=NORMATIVE)
    assert rep["passed"] is False
    assert rep["violations"] == [key]


def test_allowlist_is_subset_not_equality() -> None:
    """A subset of the frozen set is legal; the gate is one-directional."""
    _, batch = _fixture()
    subset = {k for k in FORWARD_INPUT_ALLOWLIST if k in batch}
    rep = check_input_allowlist(batch, allowed_keys=subset, preset=NORMATIVE)
    assert rep["passed"] is True


def test_blacklist_and_allowlist_both_enforced_in_run() -> None:
    """T1.4 report shows both results (ORA-2 item 3)."""
    model, batch = _fixture()
    batch["d_eff"] = torch.tensor([0.1])  # blacklisted -> also not allowed
    report = run_feature_gates(model, batch, preset=NORMATIVE, seed=SEED)
    black = report["checks"]["input_legality"]
    white = report["checks"]["input_allowlist"]
    assert black["passed"] is False and "d_eff" in black["violations"]
    assert white["passed"] is False and "d_eff" in white["violations"]
    assert report["passed"] is False


def test_allowlist_groups_cover_all_keys() -> None:
    grouped = {k for g in FORWARD_INPUT_ALLOWLIST_GROUPS.values() for k in g}
    assert grouped == set(FORWARD_INPUT_ALLOWLIST)


# ---------------------------------------------------------------------------
# T1.4c intent-off full mask (PD-6)
# ---------------------------------------------------------------------------

def test_intent_on_batch_covers_all_five_pd6_classes() -> None:
    """The synthetic batch must exercise 5/5 masked classes."""
    _, batch = _fixture()
    rep = check_intent_off_mask(
        batch, make_intent_off_batch(batch, mode="full"), preset=NORMATIVE)
    assert rep["class_coverage"] == f"5/{len(INTENT_OFF_MASK_CLASSES)}"
    assert rep["classes_absent_from_on_batch"] == []


def test_full_intent_off_mask_passes() -> None:
    _, batch = _fixture()
    rep = check_intent_off_mask(
        batch, make_intent_off_batch(batch, mode="full"), preset=NORMATIVE)
    assert rep["passed"] is True and rep["state"] == "pass"
    assert rep["classes_unmasked"] == []
    assert rep["preserved_ok"] is True
    assert rep["preserved_violations"] == []
    # preserved context really is bit-identical
    off = make_intent_off_batch(batch, mode="full")
    for key in PRESERVED_FIELD_KEYS:
        assert torch.equal(batch[key], off[key])


def test_intent_off_callable_generator_accepted() -> None:
    _, batch = _fixture()
    rep = check_intent_off_mask(
        batch, lambda b: make_intent_off_batch(b, mode="full"),
        preset=NORMATIVE)
    assert rep["passed"] is True


@pytest.mark.parametrize("leaky", [
    {"delta_b_intent"},                     # signed graph edits leak
    {"signed_intent"},                      # implicit direction encoding leaks
    {"direction_encoding"},                 # implicit direction encoding leaks
    {"cv_target_delta"},                    # directed CV descriptor leaks
    {"intent_role"},                        # signed role leaks
    {"b_src", "b_tgt"},                     # ordered source/target leaks
])
def test_partial_intent_off_fails_on_leaking_class(leaky: set[str]) -> None:
    """Every one of the five PD-6 classes is individually load-bearing."""
    _, batch = _fixture()
    off = make_intent_off_batch(batch, mode="full")
    for key in leaky:
        off[key] = batch[key].clone()  # re-leak exactly one class
    rep = check_intent_off_mask(batch, off, preset=NORMATIVE)
    assert rep["passed"] is False and rep["state"] == "fail"
    assert rep["classes_unmasked"] != []
    # the leaked keys are named in the evidence
    named = {k for cls in rep["classes"].values() for k in cls["unmasked_keys"]}
    assert leaky <= named


def test_partial_mode_generator_fails_pd6() -> None:
    _, batch = _fixture()
    rep = check_intent_off_mask(
        batch, make_intent_off_batch(batch, mode="partial"), preset=NORMATIVE)
    assert rep["passed"] is False
    assert set(rep["classes_unmasked"]) >= {
        "signed_graph_edits", "directed_cv_descriptors",
        "implicit_direction_encodings"}


def test_nuke_mode_fails_because_context_destroyed() -> None:
    """Masking must not nuke legal undirected context (PD-6)."""
    _, batch = _fixture()
    rep = check_intent_off_mask(
        batch, make_intent_off_batch(batch, mode="nuke"), preset=NORMATIVE)
    assert rep["passed"] is False and rep["state"] == "fail"
    assert rep["classes_unmasked"] == []          # intent IS masked
    assert rep["preserved_ok"] is False           # ...but context was nuked
    assert set(rep["preserved_violations"]) >= {"z", "edge_index", "b_abs"}


def test_missing_intent_off_variant_is_blocked_not_passed() -> None:
    _, batch = _fixture()
    rep = check_intent_off_mask(batch, None, preset=NORMATIVE)
    assert rep["passed"] is False and rep["state"] == "blocked"
    assert rep["block_reason"] == "intent_off_variant_missing"


def test_preserved_key_dropped_from_off_is_a_violation() -> None:
    _, batch = _fixture()
    off = make_intent_off_batch(batch, mode="full")
    del off["edge_index"]
    rep = check_intent_off_mask(batch, off, preset=NORMATIVE)
    assert rep["preserved_ok"] is False
    assert "edge_index" in rep["preserved_missing_in_off"]


def test_preserved_context_keys_are_undirected_topology() -> None:
    """PD-6 preserved set: unordered chemical topology only, no intent."""
    for key in PRESERVED_UNORDERED_CONTEXT_KEYS:
        assert key not in INTENT_FIELD_KEYS
    assert set(PRESERVED_FIELD_KEYS) <= set(PRESERVED_UNORDERED_CONTEXT_KEYS)


# ---------------------------------------------------------------------------
# Campaign-yaml tolerance binding (PD-2, normative source)
# ---------------------------------------------------------------------------

def test_load_feature_contract_from_campaign_yaml() -> None:
    contract, sha = load_campaign_feature_contract(
        REPO_ROOT / DEFAULT_CAMPAIGN_YAML)
    assert len(sha) == 64
    tol = contract["tolerances"]
    assert float(tol["relabel_invariance_max_abs"]) == 1e-6
    assert float(tol["rotation_translation_permutation"]["rel"]) == 1e-5
    assert float(tol["rotation_translation_permutation"]["abs"]) == 1e-6
    assert float(tol["record_only_reference"]) == 1e-4
    assert len(contract["intent_off_mask"]) == 5
    assert contract["preserved_off_context"]


def test_yaml_tolerances_match_registered_presets() -> None:
    contract, sha = load_campaign_feature_contract(
        REPO_ROOT / DEFAULT_CAMPAIGN_YAML)
    for name in (NORMATIVE, "r3_v3_record"):
        preset, explicit = resolve_preset(name)
        assert explicit is True
        block = verify_preset_against_yaml(preset, contract, sha)
        assert block["record_only_reference"] == 1e-4


def test_yaml_mismatch_is_hard_error() -> None:
    """No automatic relaxation: yaml/preset disagreement refuses."""
    _, _ = resolve_preset(NORMATIVE)
    tampered = {
        "tolerances": {
            "relabel_invariance_max_abs": 1e-6,
            "rotation_translation_permutation": {"rel": 1e-5, "abs": 1e-3},
            "record_only_reference": 1e-4,
        }
    }
    with pytest.raises(PresetContractError, match="PRESET_YAML_MISMATCH"):
        verify_preset_against_yaml(
            resolve_preset(NORMATIVE)[0], tampered, "0" * 64)


def test_missing_campaign_yaml_is_hard_error() -> None:
    with pytest.raises(PresetContractError, match="CAMPAIGN_YAML_MISSING"):
        load_campaign_feature_contract(
            REPO_ROOT / "config/campaigns/does_not_exist.yaml")


def test_malformed_feature_contract_is_hard_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: x\n", encoding="utf-8")
    with pytest.raises(PresetContractError, match="FEATURE_CONTRACT_MISSING"):
        load_campaign_feature_contract(bad)


# ---------------------------------------------------------------------------
# G1 gate report
# ---------------------------------------------------------------------------

def test_gate_report_pass_for_correct_model() -> None:
    model, batch = _fixture()
    report = build_gate_report(
        model, batch, preset=NORMATIVE,
        campaign_yaml=REPO_ROOT / DEFAULT_CAMPAIGN_YAML, seed=SEED,
        intent_off=make_intent_off_batch(batch))

    assert report["schema"] == "pes2ts_feature_gate_report_v1"
    assert report["gate"] == "G1-Feature"
    assert report["state"] == "pass" and report["passed"] is True
    assert report["harness_version"] == "2"

    ids = [t["test_id"] for t in report["tests"]]
    assert ids == [t[0] for t in GATE_REPORT_TESTS]
    assert ids == ["T1.1", "T1.2", "T1.3", "T1.4", "T1.4b", "T1.4c"]
    for entry in report["tests"]:
        assert entry["state"] == "pass"
        assert entry["preset"] == NORMATIVE
        assert entry["yaml_sha256"] == report["yaml_sha256"]
        assert entry["evidence"]
        assert entry["gate_admissible"] is True

    # T1.4 carries BOTH blacklist and allowlist evidence
    t14 = next(t for t in report["tests"] if t["test_id"] == "T1.4")
    assert "input_legality" in t14["evidence"]
    assert "nan_typed_invalid" in t14["evidence"]
    t14b = next(t for t in report["tests"] if t["test_id"] == "T1.4b")
    assert "input_allowlist" in t14b["evidence"]
    t14c = next(t for t in report["tests"] if t["test_id"] == "T1.4c")
    assert t14c["evidence"]["intent_off_mask"]["class_coverage"] == "5/5"

    # yaml sha is real and recorded for the G1 consumer to diff later
    import hashlib
    actual = hashlib.sha256(
        (REPO_ROOT / DEFAULT_CAMPAIGN_YAML).read_bytes()).hexdigest()
    assert report["yaml_sha256"] == actual
    assert report["campaign_yaml"]["sha256"] == actual
    assert report["campaign_yaml"]["feature_contract_present"] is True
    assert json.dumps(report)  # fully JSON-serialisable


def test_gate_report_blocked_on_intent_off_violation() -> None:
    model, batch = _fixture()
    report = build_gate_report(
        model, batch, preset=NORMATIVE,
        campaign_yaml=REPO_ROOT / DEFAULT_CAMPAIGN_YAML, seed=SEED,
        intent_off=make_intent_off_batch(batch, mode="partial"))
    t14c = next(t for t in report["tests"] if t["test_id"] == "T1.4c")
    assert t14c["state"] == "fail"
    assert report["state"] == "fail" and report["passed"] is False


def test_gate_report_fails_on_forbidden_input() -> None:
    model, batch = _fixture()
    batch["pos_ts"] = torch.zeros(3)
    report = build_gate_report(
        model, batch, preset=NORMATIVE,
        campaign_yaml=REPO_ROOT / DEFAULT_CAMPAIGN_YAML, seed=SEED,
        intent_off=make_intent_off_batch(batch))
    t14 = next(t for t in report["tests"] if t["test_id"] == "T1.4")
    assert t14["state"] == "fail"
    assert report["state"] == "fail"


def test_gate_report_record_only_never_passes() -> None:
    model, batch = _fixture()
    report = build_gate_report(
        model, batch, preset="r3_v3_record",
        campaign_yaml=REPO_ROOT / DEFAULT_CAMPAIGN_YAML, seed=SEED,
        intent_off=make_intent_off_batch(batch))
    assert report["preset"]["record_only"] is True
    assert report["gate_admissible"] is False
    assert report["banner"] == RECORD_ONLY_BANNER
    assert report["passed"] is False
    assert report["state"] == "blocked"   # not a pass gate, not a failure


def test_gate_report_blocked_without_campaign_yaml() -> None:
    model, batch = _fixture()
    report = build_gate_report(
        model, batch, preset=NORMATIVE, campaign_yaml=None, seed=SEED,
        intent_off=make_intent_off_batch(batch))
    assert report["state"] == "blocked"
    assert report["yaml_sha256"] is None
    assert report["campaign_yaml"] is None
    assert report["banner"] == "CAMPAIGN_YAML_NOT_SUPPLIED"
    assert report["passed"] is False


def test_gate_report_t14c_blocked_when_intent_off_absent() -> None:
    model, batch = _fixture()
    report = build_gate_report(
        model, batch, preset=NORMATIVE,
        campaign_yaml=REPO_ROOT / DEFAULT_CAMPAIGN_YAML, seed=SEED,
        intent_off=None)
    t14c = next(t for t in report["tests"] if t["test_id"] == "T1.4c")
    assert t14c["state"] == "blocked"
    assert report["state"] == "blocked" and report["passed"] is False


# ---------------------------------------------------------------------------
# CLI binding (--gate-report / --preset / --campaign-yaml)
# ---------------------------------------------------------------------------

def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.feature_gate.equivariance_harness",
         *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300,
    )


def test_cli_refuses_missing_preset() -> None:
    proc = _run_cli([])
    assert proc.returncode == 2
    assert "preset is required" in proc.stderr


def test_cli_gate_report_writes_json(tmp_path: Path) -> None:
    out = tmp_path / "g1_feature.json"
    proc = _run_cli([
        "--gate-report", str(out),
        "--preset", NORMATIVE,
        "--campaign-yaml", DEFAULT_CAMPAIGN_YAML,
        "--seed", str(SEED),
    ])
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["state"] == "pass"
    assert report["gate"] == "G1-Feature"
    assert report["preset"]["name"] == NORMATIVE
    assert len(report["yaml_sha256"]) == 64
    assert [t["test_id"] for t in report["tests"]] == [
        "T1.1", "T1.2", "T1.3", "T1.4", "T1.4b", "T1.4c"]
    assert "state=" in proc.stdout


def test_cli_gate_report_record_only_exit_1(tmp_path: Path) -> None:
    out = tmp_path / "g1_record.json"
    proc = _run_cli([
        "--gate-report", str(out),
        "--preset", "r3_v3_record",
        "--campaign-yaml", DEFAULT_CAMPAIGN_YAML,
    ])
    assert proc.returncode == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["state"] == "blocked"
    assert report["banner"] == RECORD_ONLY_BANNER


def test_cli_intent_off_partial_blocks_gate(tmp_path: Path) -> None:
    out = tmp_path / "g1_partial.json"
    proc = _run_cli([
        "--gate-report", str(out),
        "--preset", NORMATIVE,
        "--campaign-yaml", DEFAULT_CAMPAIGN_YAML,
        "--intent-off-mode", "partial",
    ])
    assert proc.returncode == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["state"] == "fail"


def test_cli_legacy_non_strict_never_passes() -> None:
    proc = _run_cli(["--legacy-non-strict", "--preset", NORMATIVE])
    # explicit preset + legacy flag still fine; the flag alone is the opt-in
    assert proc.returncode == 0
    proc2 = _run_cli(["--legacy-non-strict"])
    # no preset + legacy opt-in -> diagnostics only, non-zero exit
    assert proc2.returncode == 1


def test_cli_unknown_preset_rejected() -> None:
    proc = _run_cli(["--preset", "r3_v3"])
    assert proc.returncode == 2


def test_cli_stdout_report_without_gate_report() -> None:
    proc = _run_cli(["--preset", NORMATIVE, "--seed", str(SEED)])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["harness_version"] == "2"
    assert payload["preset"] == NORMATIVE
    assert payload["passed"] is True
    assert payload["checks"]["intent_off_mask"]["passed"] is True
