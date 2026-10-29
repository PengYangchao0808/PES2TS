"""V5.4.1 H0 legality audit tests (``scripts/campaigns/v541_legality_audit.py``).

H0 is the fixture/restore legality gate of plan §11.1:
schema / no-leak / confirm isolation / restore / E(3) / permutation /
non-key-atom action must ALL pass before any formal T1 run is granted.

These tests do NOT re-run the 50K campaign and do not train anything: they
(a) verify the fail-closed roll-up logic, (b) verify the four committed
artifacts are internally consistent (schemas, per-candidate coverage,
evidence SHAs, scope boundaries), (c) exercise the audit's own typed
refusal paths on the synthetic fixture, and (d) assert the tooling never
touches subprocess / dev / confirm.

The audit itself was EXECUTED (``scripts/campaigns/v541_legality_audit.py
--root outputs/pesn/v541_embed_50k --force`` → ``H0_GRANTED``); the
committed artifacts are the evidence under test here.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.models.v541 import V541Model            # noqa: E402
from pes2ts_core.pesn.offline.training.v541_degree_stats import (  # noqa: E402
    ALLOWED_SPLIT as DEGREE_STATS_SPLIT,
    compute_degree_stats,
)
from scripts.campaigns.v541_legality_audit import (           # noqa: E402
    AUDIT_ALPHA_V541,
    CANDIDATES_V541,
    H0AuditError,
    build_fixture_env,
    h0_verdict,
    run_h0_legality_audit,
)
from scripts.campaigns.v541_legality_audit import (           # noqa: E402
    _tamper_refusals,
)

AUDITS = PROJECT_ROOT / "outputs" / "pesn" / "v541_embed_50k" / "audits"
SCHEMA_H0 = "pes2ts_v541_h0_legality_audit_v1"
SCHEMA_GEO = "pes2ts_v541_geometry_invariance_audit_v1"
SCHEMA_LEAK = "pes2ts_v541_input_leak_audit_v1"
SCHEMA_RESUME = "pes2ts_v541_resume_equivalence_audit_v1"

ARTIFACTS = {
    "h0_legality.json": SCHEMA_H0,
    "geometry_invariance.json": SCHEMA_GEO,
    "input_leak.json": SCHEMA_LEAK,
    "resume_equivalence.json": SCHEMA_RESUME,
}


@pytest.fixture(scope="module")
def artifacts() -> dict:
    """The four committed H0 artifacts, loaded once."""
    out = {}
    for name in ARTIFACTS:
        path = AUDITS / name
        if not path.is_file():
            pytest.skip("H0 artifacts not produced yet: {}".format(path))
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# fail-closed roll-up (pure)
# ---------------------------------------------------------------------------


def test_h0_verdict_is_fail_closed() -> None:
    good = [{"check": "a", "passed": True}, {"check": "b", "passed": True}]
    assert h0_verdict(good)["passed"] is True
    assert h0_verdict(good)["failed_items"] == []

    bad = good + [{"check": "c", "passed": False}]
    roll = h0_verdict(bad)
    assert roll["passed"] is False
    assert roll["failed_items"] == ["c"]
    assert "fail-closed" in roll["decision"]
    assert roll["n_items"] == 3 and roll["n_passed"] == 2

    # a missing/absent ``passed`` flag is a FAILURE, never a pass
    assert h0_verdict([{"check": "x"}])["passed"] is False


# ---------------------------------------------------------------------------
# committed artifacts
# ---------------------------------------------------------------------------


def test_all_four_artifacts_exist_with_their_schema(artifacts) -> None:
    assert set(artifacts) == set(ARTIFACTS)
    for name, schema in ARTIFACTS.items():
        assert artifacts[name]["schema"] == schema, name


def test_h0_verdict_granted_and_every_item_passed(artifacts) -> None:
    doc = artifacts["h0_legality.json"]
    assert doc["passed"] is True
    assert doc["failed_items"] == []
    assert doc["n_items"] == doc["n_passed"] == len(doc["items"]) > 0
    assert "H0 granted" in doc["decision"]
    assert doc["gate"] == "H0"
    assert doc["level"] == "fixture_restore"
    assert doc["objective"] == "unified_move_process"
    assert doc["campaign_id"] == "PES2TS_V541_EMBED_50K"
    # the H0 rule's clauses are each an explicit item
    checks = {i["check"] for i in doc["items"]}
    for required in ("config_schema_formal", "formal_thresholds_signed_frozen",
                     "config_authorization_firewall",
                     "single_objective_no_arm_matrix", "data_audit_ok",
                     "data_audit_confirm_isolated", "geometry_invariance",
                     "input_leak", "resume_equivalence"):
        assert required in checks, required
    assert len([c for c in checks if c.startswith(
        "e0_parity_certificate_")]) == 2


def test_h0_sections_all_passed_without_error(artifacts) -> None:
    sections = artifacts["h0_legality.json"]["sections"]
    assert set(sections) == {"schema_gates", "geometry_invariance",
                             "input_leak", "resume_equivalence"}
    for name, block in sections.items():
        assert block["passed"] is True, name
        assert block["error"] is None, name


def test_artifact_evidence_shas_are_real(artifacts) -> None:
    """Every evidence-bearing item must carry the file's real sha256."""
    import hashlib

    items = artifacts["h0_legality.json"]["items"]
    for item in items:
        if "path" not in item:
            continue                       # pure config/gate items
        assert len(item.get("sha256") or "") == 64, item["check"]
    for name in ("geometry_invariance.json", "input_leak.json",
                 "resume_equivalence.json"):
        path = AUDITS / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        owner = next(i for i in items
                     if i["check"] == name.rsplit(".", 1)[0])
        assert owner["sha256"] == digest, name
    # the data-audit evidence is hashed too
    data = next(i for i in items if i["check"] == "data_audit_ok")
    assert data["sha256"] == hashlib.sha256(
        (AUDITS / "data_audit.json").read_bytes()).hexdigest()


def test_cost_and_scope_boundaries(artifacts) -> None:
    """H0 is fixture/restore level: no 50K run, no dev, no confirm, no QC."""
    cost = artifacts["h0_legality.json"]["cost"]
    assert cost["real_50k_training"] is False
    assert cost["dev_read"] is False
    assert cost["confirm_read"] is False
    assert cost["new_qc_calls"] == 0
    assert cost["device"] == "cpu"
    assert cost["wall_seconds"] > 0

    fixture = artifacts["h0_legality.json"]["fixture"]
    # dev and confirm exist in the fixture and were EXCLUDED, not skipped
    assert fixture["n_dev_positions"] > 0
    assert fixture["n_confirm_positions"] > 0
    assert fixture["n_train_positions"] > 0
    assert fixture["n_inner_val_positions"] > 0
    assert fixture["dev_curve_ids"] and fixture["confirm_curve_ids"]
    assert set(fixture["dev_curve_ids"]).isdisjoint(fixture["confirm_curve_ids"])
    for name, digest in fixture["tables_sha256"].items():
        assert len(digest) == 64, name


def _string_values(obj, *, skip_keys=("check",)) -> list:
    """Every emitted STRING VALUE; ``check`` names are the audit's own
    vocabulary (``single_objective_no_arm_matrix`` PROVES the matrix is
    absent) and are not report content."""
    out = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in skip_keys:
                continue
            out += _string_values(value, skip_keys=skip_keys)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out += _string_values(value, skip_keys=skip_keys)
    elif isinstance(obj, str):
        out.append(obj)
    return out


def test_no_arm_matrix_leak_in_any_artifact(artifacts) -> None:
    """§3.1.4: reports never re-expose the legacy three-arm matrix.

    Key NAMES may mention the concept — ``single_objective_no_arm_matrix``
    and ``arm_matrix_keys`` are the check that PROVES it is empty — but no
    emitted VALUE may.
    """
    for name, doc in artifacts.items():
        for value in _string_values(doc):
            assert "arm" not in value.lower(), (name, value[:120])
            assert "三臂" not in value, (name, value[:120])
    gates = next(i for i in artifacts["h0_legality.json"]["items"]
                 if i["check"] == "single_objective_no_arm_matrix")
    assert gates["arm_matrix_keys"] == []
    assert gates["objective"] == "unified_move_process"


# ---------------------------------------------------------------------------
# geometry artifact
# ---------------------------------------------------------------------------


def test_geometry_covers_every_candidate_at_nonzero_alpha(artifacts) -> None:
    geo = artifacts["geometry_invariance.json"]
    assert geo["passed"] is True
    assert set(geo["per_candidate"]) == set(CANDIDATES_V541)
    for eid in ("E1", "E2", "E3", "E4", "E5"):
        gate = geo["per_candidate"][eid]["gate"]
        assert gate["alpha"] == AUDIT_ALPHA_V541, eid
        assert gate["requested"] == AUDIT_ALPHA_V541
        assert gate["reason"] is None
    # E0 has no branch, hence no gate — never a silent 0.25
    e0_gate = geo["per_candidate"]["E0"]["gate"]
    assert e0_gate["alpha"] is None
    assert e0_gate["reason"] == "e0_no_branch"


def test_geometry_block_carries_every_required_check(artifacts) -> None:
    geo = artifacts["geometry_invariance.json"]
    assert geo["preset"] == "30h_v1.1_normative"
    for eid, block in geo["per_candidate"].items():
        assert block["passed"] is True, eid
        assert block["translation_invariance_u"]["passed"] is True, eid
        assert block["rotation_equivariance_u"]["passed"] is True, eid
        assert block["reflection_equivariance_u"]["passed"] is True, eid
        # improper transform really is improper (det = -1)
        assert block["reflection_equivariance_u"]["det"] < 0.0, eid
        assert block["permutation_equivariance_u"]["passed"] is True, eid
        assert block["batch_consistency"]["passed"] is True, eid
        assert block["non_key_atoms_no_direct_action"]["passed"] is True, eid
        for transform, scalar in block["scalar_invariance"].items():
            assert scalar["passed"] is True, (eid, transform)
            assert set(scalar["scalars"]) == {
                "z", "q", "endpoint_logit", "order_score"}, (eid, transform)
        assert block["batch_consistency"]["independent_graphs"] >= 1
        assert block["batch_consistency"]["mixed_graphs"] > \
            block["batch_consistency"]["independent_graphs"]


# ---------------------------------------------------------------------------
# input-leak artifact
# ---------------------------------------------------------------------------


def test_input_leak_covers_every_required_domain(artifacts) -> None:
    leak = artifacts["input_leak.json"]
    assert leak["passed"] is True
    assert set(leak["per_candidate"]) == set(CANDIDATES_V541)
    for eid, block in leak["per_candidate"].items():
        assert block["passed"] is True, eid
        inter = block["teacher_metadata_intervention"]
        assert inter["passed"] is True, eid
        assert inter["forward_view_identical"] is True
        assert inter["outputs_identical"] is True
        assert inter["supervision_key_refusal"]["raised"] is True
        assert inter["forbidden_metadata_refusal"]["raised"] is True
        finite = block["finite_outputs_and_gradients"]
        assert finite["passed"] is True, eid
        assert finite["loss_finite"] and finite["outputs_finite"]
        assert finite["n_grad_nonfinite"] == 0
        assert finite["n_grad_tensors"] > 0
        assert block["u_rows_equal_n_keys"] is True
        assert block["n_keys"] < block["n_atoms"]   # spectator atoms exist
        assert block["node_vector_consumed"] == 0.0


def test_e6_is_still_refused(artifacts) -> None:
    e6 = artifacts["input_leak.json"]["e6_refused"]
    assert e6["raised"] is True
    assert e6["code"] == "V541_EMBEDDING_NOT_IMPLEMENTED"
    # and the refusal is real, not just recorded
    with pytest.raises(Exception) as exc:
        V541Model(embedding_id="E6", embedding_init_seed=20261028)
    assert getattr(exc.value, "code", "") == "V541_EMBEDDING_NOT_IMPLEMENTED"


def test_e2_degree_stats_discipline(artifacts) -> None:
    degree = artifacts["input_leak.json"]["degree_stats"]
    assert degree["passed"] is True
    ids = {c["check"]: c for c in degree["checks"]}
    for required in ("degree_stats_fitted_on_inner_train",
                     "degree_stats_population_is_inner_train_only",
                     "degree_stats_envelope_digest",
                     "degree_stats_strict_loader_accepts",
                     "degree_stats_fitted_outside_campaign_cache",
                     "degree_stats_forbidden_split_refused",
                     "e2_refuses_missing_degree_stats",
                     "degree_stats_tampered_envelope_refused"):
        assert required in ids and ids[required]["passed"] is True, required
    # the fit really is inner-train and really sits OUTSIDE the campaign cache
    assert ids["degree_stats_population_is_inner_train_only"]["split"] == \
        DEGREE_STATS_SPLIT
    assert ids["degree_stats_population_is_inner_train_only"][
        "leaked_positions"] == []
    assert ids["degree_stats_fitted_outside_campaign_cache"]["passed"] is True
    assert ids["e2_refuses_missing_degree_stats"]["code"] == \
        "DEGREE_STATS_REQUIRED"


# ---------------------------------------------------------------------------
# resume artifact
# ---------------------------------------------------------------------------


def test_resume_protocol_per_candidate(artifacts) -> None:
    resume = artifacts["resume_equivalence.json"]
    assert resume["passed"] is True
    assert set(resume["per_candidate"]) == set(CANDIDATES_V541)
    for eid, block in resume["per_candidate"].items():
        assert block["passed"] is True, eid
        assert block["model_bitwise"] is True, eid
        assert block["optimizer_state"] is True, eid
        assert block["best_checkpoint_digest"] is True, eid
        assert block["last_content_digest"] is True, eid
        assert block["schedule_chain"] is True, eid
        assert block["train_log_semantic_rows"] is True, eid
        assert block["inner_val_by_update"] is True, eid
        assert block["manifest_appended_not_overwritten"] is True, eid
        assert block["leg1_stopped_where_leg2_resumed"] is True, eid
        assert block["manifest_run_complete_events"] == 2, eid
        assert len(block["resume_event"]) > 0, eid
        # the two typed refusals of §7.3
        tamper = block["tamper"]
        assert tamper["passed"] is True, eid
        assert tamper["identity_tamper"]["code"] == \
            "RESUME_IDENTITY_MISMATCH", eid
        assert tamper["content_digest_tamper"]["code"] == \
            "RESUME_CONTENT_DIGEST", eid
        assert tamper["refusals_did_not_append_resume"] is True, eid
    # E0 is the full bitwise anchor; E1-E5 the bounded form
    assert resume["per_candidate"]["E0"]["legs"] == "20+20 vs 40"
    for eid in ("E1", "E2", "E3", "E4", "E5"):
        assert resume["per_candidate"][eid]["legs"] == "2+2 vs 4"
    assert resume["protocol"]["note"].startswith("a live-process kill")


# ---------------------------------------------------------------------------
# the tooling itself
# ---------------------------------------------------------------------------


def test_audit_refuses_to_overwrite_without_force(tmp_path) -> None:
    (tmp_path / "audits").mkdir()
    (tmp_path / "audits" / "h0_legality.json").write_text("{}",
                                                          encoding="utf-8")
    with pytest.raises(H0AuditError) as exc:
        run_h0_legality_audit(root=tmp_path, config_path=None, force=False)
    assert exc.value.code == "H0_OUTPUT_EXISTS"


def test_tamper_refusals_are_typed(tmp_path) -> None:
    """The audit's own restore-refusal probe (fixture level, ~2 s)."""
    env = build_fixture_env(tmp_path / "tables")
    assert env.confirm_pos and env.dev_pos and env.train_pos
    block = _tamper_refusals("E0", env, degree_stats=None)
    assert block["passed"] is True
    assert block["identity_tamper"]["code"] == "RESUME_IDENTITY_MISMATCH"
    assert block["content_digest_tamper"]["code"] == "RESUME_CONTENT_DIGEST"
    assert block["refusals_did_not_append_resume"] is True


def test_degree_stats_forbidden_splits_refused(tmp_path) -> None:
    """dev / inner-val / confirm fits are typed refusals (§6.6)."""
    env = build_fixture_env(tmp_path / "tables")
    legal = compute_degree_stats(env.ds.samples, list(env.train_pos),
                                 split="inner-train", seed=17)
    assert legal.split == DEGREE_STATS_SPLIT
    assert legal.n_positions == len(set(env.train_pos))
    for bad, positions in (("dev", env.dev_pos),
                           ("inner-val", env.inner_val_pos),
                           ("confirm", env.confirm_pos)):
        with pytest.raises(Exception) as exc:
            compute_degree_stats(env.ds.samples, list(positions),
                                 split=bad, seed=17)
        assert getattr(exc.value, "code", "") == \
            "V541_DEGREE_STATS_SPLIT"


def test_confirm_positions_never_enter_the_training_pool(tmp_path) -> None:
    """H0's confirm isolation, exercised on the audit's own fixture."""
    env = build_fixture_env(tmp_path / "tables")
    pool_curves = {str(env.ds.samples[i].curve_id) for i in env.pool_pos}
    assert not (pool_curves & set(env.confirm_curve_ids))
    assert not (pool_curves & set(env.dev_curve_ids))
    assert set(env.train_pos) | set(env.inner_val_pos) == set(env.pool_pos)
    assert not (set(env.train_pos) & set(env.confirm_pos))
    assert not (set(env.train_pos) & set(env.dev_pos))


def test_audit_module_never_imports_subprocess() -> None:
    """The tooling must stay free of dynamic execution (truth guard)."""
    path = PROJECT_ROOT / "scripts" / "campaigns" / "v541_legality_audit.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names
                    if a.name.split(".")[0] in ("subprocess", "importlib")]
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in ("subprocess", "importlib"):
                bad.append(node.module)
    assert bad == []
