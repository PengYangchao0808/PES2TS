"""G2 remediation tests (findings 2/3/4/5).

Covers:
* real-shard BC↔IQL same-seed draw equivalence (finding 3 — the production
  claim, NOT the fixture-shaped one);
* the expanded λ/identity registration gate for FORMAL stages (finding 4);
* protocol verification on read + revision-chain verification (finding 4);
* run-dir identity + selection-record v41 namespace filtering (finding 2);
* IQL exposure snapshots + near-TS integrity refusal (finding 5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pes2ts_core.pesn.contracts.dual_ts import (
    DualTSAction,
    DualTSActionKind,
    DualTSTransition,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (
    load_dual_ts_states,
    resolve_lambda_registration,
)
from pes2ts_core.pesn.offline.training.dual_ts_iql import (
    DualTSIQLTrainerError,
    DualTSTransitionRow,
    EpisodeKeyedBuffer,
)
from pes2ts_core.pesn.offline.training.dual_ts_sampling import (
    DualTSGroupBranchSampler,
    build_sampler_groups,
)
from scripts.train_dual_ts_bc import (
    SELECTION_RECORD_NAME,
    V41_RUN_SCHEMA,
    _record_settings_digest,
    build_run_dir_name,
    compute_protocol_content_digest,
    finalize_selection,
    load_verified_protocol,
    run_settings_digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_V41 = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41"
OBSERVED_FIT = PROJECT_ROOT / "outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed"
A_MAX = 0.1
SPEC_DIGEST = "f678f8e5c83cf913980139b48d4836475e11de5d1e1a52035cf6abddf9c1a99c"

FIT50_IDS = [
    "RXN_0000000072", "RXN_0000000139", "RXN_0000000160", "RXN_0000000187",
    "RXN_0000000237", "RXN_0000000238", "RXN_0000000406", "RXN_0000000535",
    "RXN_0000000575", "RXN_0000000582", "RXN_0000000623", "RXN_0000000733",
    "RXN_0000000756", "RXN_0000000778", "RXN_0000000908", "RXN_0000000990",
    "RXN_0000001076", "RXN_0000001081", "RXN_0000001089", "RXN_0000001094",
    "RXN_0000001106", "RXN_0000001117", "RXN_0000001120", "RXN_0000001156",
    "RXN_0000001174", "RXN_0000001195", "RXN_0000001205", "RXN_0000001211",
    "RXN_0000001242", "RXN_0000001248", "RXN_0000001309", "RXN_0000001423",
    "RXN_0000001484", "RXN_0000001504", "RXN_0000001672", "RXN_0000001675",
    "RXN_0000001678", "RXN_0000001697", "RXN_0000001715", "RXN_0000001735",
    "RXN_0000001745", "RXN_0000001847", "RXN_0000002022", "RXN_0000002062",
    "RXN_0000002064", "RXN_0000002090", "RXN_0000002226", "RXN_0000002286",
    "RXN_0000002309", "RXN_0000002371",
]


# ---------------------------------------------------------------------------
# Finding 3: real-shard BC↔IQL same-seed draw equivalence
# ---------------------------------------------------------------------------

def _fallback_action(n_atoms: int) -> DualTSAction:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n_atoms, 3)) * 0.01
    return DualTSAction(kind=DualTSActionKind.ACCEPTED_FALLBACK,
                        a_max_angstrom=A_MAX, action_angstrom=a,
                        reason="g2_remediation_fixture")


def _iql_buffer_from_states(states: list) -> EpisodeKeyedBuffer:
    """One terminal transition per drawable state, in the GIVEN (natural
    shard) order — NOT pre-sorted.  The fixed ``head_groups()`` must re-order
    to remaining-arc-length ascending to match BC."""
    rows = []
    for s in states:
        n_atoms = int(np.asarray(s["atomic_numbers"]).size)
        t = DualTSTransition(
            transition_id=f"{s['state_id']}-t",
            state_id=s["state_id"],
            next_state_id=None,
            episode_id=s["episode_id"],
            branch_id=s["branch_id"],
            proposed_action=_fallback_action(n_atoms),
            next_observation_missing_reason="teacher_frame_isolated",
            boundary="teacher_endpoint",
            terminal=True,
            head_eligibility={
                "Q": {"eligible": True, "reason": "fixture"},
                "V": {"eligible": True, "reason": "fixture"},
                "actor": {"eligible": True, "reason": "fixture"},
            },
        )
        rows.append(DualTSTransitionRow(
            reaction_group_id=s["reaction_group_id"], transition=t,
            state_inputs={"atomic_numbers": s["atomic_numbers"]},
            reward=0.1,
            remaining_arc_length=s["remaining_arc_length"]))
    return EpisodeKeyedBuffer(rows)


def test_real_shard_bc_iql_identical_draw_sequence():
    """On the REAL fit1 shard (RXN_0000002090), the same seed yields the
    IDENTICAL first-60 drawn state ids from BC and IQL (finding 3)."""
    if not FEATURES_V41.is_dir() or not OBSERVED_FIT.is_dir():
        pytest.skip("real v4.1 feature/observation shards not present (local outputs)")
    states, meta = load_dual_ts_states(
        FEATURES_V41, OBSERVED_FIT, group_filter=["RXN_0000002090"])
    drawable = [s for s in states if not s["is_teacher"]]
    assert len(drawable) == 120  # 122 states − 2 teacher endpoints

    groups_bc = build_sampler_groups(
        drawable,
        group_key=lambda s: s["reaction_group_id"],
        branch_key=lambda s: s["branch_id"],
        arc_key=lambda s, _i: s["remaining_arc_length"],
    )
    buf = _iql_buffer_from_states(drawable)
    groups_iql = buf.head_groups()

    # layout: identical (group, branch) → [state_id] in both views
    assert set(groups_bc) == set(groups_iql)
    for g in groups_bc:
        assert set(groups_bc[g]) == set(groups_iql[g])
        for b in groups_bc[g]:
            bc_ids = [drawable[i]["state_id"] for i in groups_bc[g][b]]
            iql_ids = [buf.rows[i].transition.state_id
                       for i in groups_iql[g][b]]
            assert bc_ids == iql_ids, (g, b)

    seed = 20261017
    bc_sampler = DualTSGroupBranchSampler(seed=seed, groups=groups_bc,
                                          rows_per_group=1)
    iql_sampler = DualTSGroupBranchSampler(seed=seed, groups=groups_iql,
                                           rows_per_group=1)
    n = 60
    bc_draws = bc_sampler.draw_batch(n)
    iql_draws = iql_sampler.draw_batch(n)
    assert len(bc_draws) == len(iql_draws) == n
    bc_ids = [drawable[i]["state_id"] for i in bc_draws]
    iql_ids = [buf.rows[i].transition.state_id for i in iql_draws]
    assert bc_ids == iql_ids


# ---------------------------------------------------------------------------
# Finding 4: expanded registration gate (formal vs canary)
# ---------------------------------------------------------------------------

_PROTOCOL_R1 = {
    "registered_budgets": {
        "fit1": {"updates": None},
        "fit5": {"updates": None},
        "short": {"updates": 2000},
        "full": {"updates": 20000},
    },
    "lambda_registration": {
        "slots": {
            "arm_a": {"lambda_a": 1.0, "lambda_dir": 1.0, "status": "frozen"},
            "arm_b": None,
        },
    },
    "arm_mapping": {
        "A": {"recipe": "bc_nll", "lambda_slot": "arm_a",
              "lambda_a": 1.0, "lambda_dir": 1.0},
        "B": {"recipe": "bc_d", "lambda_slot": "arm_b",
              "lambda_a": 1.0, "lambda_dir": None},
        "C": {"recipe": "bc_nll", "lambda_slot": "arm_a",
              "lambda_a": 1.0, "lambda_dir": 1.0},
        "D": {"recipe": "bc_d", "lambda_slot": "arm_b",
              "lambda_a": 1.0, "lambda_dir": None},
    },
    "matrix_hyperparams": {"batch_states": 64, "lr": 1e-3},
    "pipeline": {"spec_digest": SPEC_DIGEST},
    "seeds": [20261017, 20261018, 20261019],
    "fit50_group_ids": FIT50_IDS,
    "fit1_fit5_subsets": {
        "fit1_group_ids": ["RXN_0000002090"],
        "fit5_group_ids": ["RXN_0000002090", "RXN_0000000623",
                           "RXN_0000000733", "RXN_0000000160",
                           "RXN_0000001094"],
    },
}


def test_resolve_lambda_registration_formal_bindings():
    d = resolve_lambda_registration(
        _PROTOCOL_R1, recipe="bc_nll", lambda_a=1.0, lambda_dir=1.0,
        stage="full", budget_updates=20000, allow_diagnostic_budget=False,
        seed=20261017, features_spec_digest=SPEC_DIGEST,
        population_group_ids=FIT50_IDS, batch_states=64, lr=1e-3)
    assert d["accepted"] and not d["diagnostic"] and not d["problems"]

    def refused(**kw):
        base = dict(recipe="bc_nll", lambda_a=1.0, lambda_dir=1.0,
                    stage="full", budget_updates=20000,
                    allow_diagnostic_budget=False, seed=20261017,
                    features_spec_digest=SPEC_DIGEST,
                    population_group_ids=FIT50_IDS, batch_states=64, lr=1e-3)
        base.update(kw)
        return not resolve_lambda_registration(_PROTOCOL_R1, **base)["accepted"]

    # wrong features dir (old v3 spec digest) → refused (finding 4.2)
    assert refused(features_spec_digest="62d65d16a607b54d59575021958c0056b7")
    # wrong population (a single canary group for a formal stage) → refused
    assert refused(population_group_ids=["RXN_0000002090"])
    # matrix batch/lr mismatches → refused
    assert refused(batch_states=32)
    assert refused(lr=1e-2)
    # seed not in protocol → refused
    assert refused(seed=12345)
    # recipe not in the A/B/C/D mapping → refused
    assert refused(recipe="bc_a")

    # canary: seed free, updates free, but population must match the subset
    d = resolve_lambda_registration(
        _PROTOCOL_R1, recipe="bc_nll", lambda_a=1.0, lambda_dir=1.0,
        stage="fit1", budget_updates=50, allow_diagnostic_budget=False,
        seed=12345, features_spec_digest=SPEC_DIGEST,
        population_group_ids=["RXN_0000002090"], batch_states=8, lr=1e-2)
    assert d["accepted"] and not d["diagnostic"]
    # canary with a wrong subset population → refused
    d = resolve_lambda_registration(
        _PROTOCOL_R1, recipe="bc_nll", lambda_a=1.0, lambda_dir=1.0,
        stage="fit1", budget_updates=50, allow_diagnostic_budget=False,
        seed=12345, features_spec_digest=SPEC_DIGEST,
        population_group_ids=["RXN_0000000623"], batch_states=8, lr=1e-2)
    assert not d["accepted"]


# ---------------------------------------------------------------------------
# Finding 4: protocol verification on read + revision chain
# ---------------------------------------------------------------------------

def test_protocol_verification_on_read_and_chain(tmp_path):
    # r0
    r0 = {"pipeline": {"spec_digest": SPEC_DIGEST},
          "registered_budgets": {"full": {"updates": 20000}}}
    r0_sha = compute_protocol_content_digest(r0)
    r0["freeze"] = {"sha256": r0_sha}
    r0_path = tmp_path / "protocol_v41.json"
    r0_path.write_text(json.dumps(r0, sort_keys=True, indent=2) + "\n")

    # r1 chained to r0
    r1 = {k: v for k, v in r0.items() if k != "freeze"}
    r1["matrix_hyperparams"] = {"batch_states": 64, "lr": 1e-3}
    r1["revision"] = {"n": 1, "parent_freeze_sha256": r0_sha,
                      "parent_file": "protocol_v41.json"}
    r1_sha = compute_protocol_content_digest(r1)
    r1["freeze"] = {"sha256": r1_sha}
    r1_path = tmp_path / "protocol_v41_r1.json"
    r1_path.write_text(json.dumps(r1, sort_keys=True, indent=2) + "\n")

    payload, digest = load_verified_protocol(r1_path)
    assert digest == r1_sha

    # tamper r1 content (not the freeze block) → refusal
    tampered = json.loads(r1_path.read_text())
    tampered["matrix_hyperparams"]["lr"] = 0.5
    tampered_path = tmp_path / "protocol_v41_r1_tampered.json"
    tampered_path.write_text(json.dumps(tampered, sort_keys=True, indent=2) + "\n")
    with pytest.raises(SystemExit) as err:
        load_verified_protocol(tampered_path)
    assert "digest mismatch" in str(err.value)

    # tamper the PARENT (r0) → r1's chain verification refuses
    r0_bad = json.loads(r0_path.read_text())
    r0_bad["registered_budgets"]["full"]["updates"] = 12345
    r0_path.write_text(json.dumps(r0_bad, sort_keys=True, indent=2) + "\n")
    with pytest.raises(SystemExit) as err:
        load_verified_protocol(r1_path)
    assert "parent" in str(err.value) or "digest mismatch" in str(err.value)


# ---------------------------------------------------------------------------
# Finding 2: run-dir identity + selection-record namespace filtering
# ---------------------------------------------------------------------------

def test_run_dir_name_includes_stage_lambda_diag():
    assert build_run_dir_name(
        recipe="bc_nll", seed=20261017, variant="", stage="short",
        lambda_a=1.0, lambda_dir=1.0, diagnostic_run=False) == \
        "run_bc_bc_nll_seed20261017_stage-short"
    assert build_run_dir_name(
        recipe="bc_d", seed=20261018, variant="_signedvecmsg", stage="full",
        lambda_a=1.0, lambda_dir=2.0, diagnostic_run=False) == \
        "run_bc_bc_d_seed20261018_signedvecmsg_stage-full_lambda1_2"
    assert build_run_dir_name(
        recipe="bc_nll", seed=20261017, variant="", stage="fit1",
        lambda_a=1.0, lambda_dir=1.0, diagnostic_run=True) == \
        "run_bc_bc_nll_seed20261017_stage-fit1_diag"


def test_run_settings_digest_detects_stage_and_lambda():
    base = dict(recipe="bc_nll", seed=20261017, stage="short",
                lambda_a=1.0, lambda_dir=1.0, features_spec_digest=SPEC_DIGEST,
                protocol_sha256="a" * 64)
    assert run_settings_digest(**base) == run_settings_digest(**base)
    assert run_settings_digest(**base) != run_settings_digest(
        **{**base, "stage": "full"})
    assert run_settings_digest(**base) != run_settings_digest(
        **{**base, "lambda_dir": 2.0})
    rec = {"settings": {"recipe": "bc_nll", "seed": 20261017, "stage": "short",
                        "lambda_a": 1.0, "lambda_dir": 1.0,
                        "protocol_sha256": "a" * 64},
           "data": {"spec_digest": SPEC_DIGEST}}
    assert _record_settings_digest(rec) == run_settings_digest(**base)


def _write_run_record(run_dir: Path, *, schema, stage, diagnostic, spec_digest,
                      recipe="bc_nll", seed=20261017,
                      protocol_sha256="b" * 64, updates_run=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema_version": schema,
        "recipe": recipe,
        "settings": {
            "seed": seed, "stage": stage, "diagnostic_run": diagnostic,
            "lambda_a": 1.0, "lambda_dir": 1.0,
            "signed_mu_gate": False, "signed_vector_messages": False,
            "protocol_sha256": protocol_sha256,
        },
        "data": {"spec_digest": spec_digest},
        "accounting": {"updates_run": updates_run
                       if updates_run is not None
                       else (2000 if stage in ("short", "full") else 50)},
    }
    (run_dir / "run_record.json").write_text(json.dumps(rec))


def test_selection_record_v41_filters_v4_canary_diagnostic(tmp_path):
    protocol = {"pipeline": {"spec_digest": SPEC_DIGEST},
                "freeze": {"sha256": "b" * 64}}
    thresholds_path = tmp_path / "monitor_thresholds_v1.json"
    thresholds_path.write_text(json.dumps({
        "prereg_sha256": "p" * 64,
        "selection_rule": {"simplicity_order": ["bc_nll", "bc_a", "bc_d"]},
    }))

    # v4-schema record → ignored
    _write_run_record(tmp_path / "run_bc_bc_a_seed20261017",
                      schema="pesn_dual_ts_bc_run_v1", stage="short",
                      diagnostic=False, spec_digest="62d65d16a607b54d")
    # v4.1 formal full → runs
    _write_run_record(tmp_path / "run_bc_bc_nll_seed20261017_stage-full",
                      schema=V41_RUN_SCHEMA, stage="full",
                      diagnostic=False, spec_digest=SPEC_DIGEST)
    # v4.1 formal short → screening (never runs)
    _write_run_record(tmp_path / "run_bc_bc_nll_seed20261017_stage-short",
                      schema=V41_RUN_SCHEMA, stage="short",
                      diagnostic=False, spec_digest=SPEC_DIGEST)
    # v4.1 canary fit1 → canary_record (keyed with stage)
    _write_run_record(tmp_path / "run_bc_bc_nll_seed20261017_stage-fit1",
                      schema=V41_RUN_SCHEMA, stage="fit1",
                      diagnostic=False, spec_digest=SPEC_DIGEST)
    # v4.1 diagnostic → excluded entirely
    _write_run_record(tmp_path / "run_bc_bc_nll_seed20261017_stage-short_diag",
                      schema=V41_RUN_SCHEMA, stage="short",
                      diagnostic=True, spec_digest=SPEC_DIGEST)
    # v4.1 wrong spec digest → ignored
    _write_run_record(tmp_path / "run_bc_bc_nll_seed20261017_stage-short_v3",
                      schema=V41_RUN_SCHEMA, stage="short",
                      diagnostic=False, spec_digest="62d65d16a607b54d")

    selection = finalize_selection(tmp_path, thresholds_path, {}, protocol)
    assert selection["schema_version"] == "pesn_wp4_selection_record_v41"
    assert list(selection["runs"].keys()) == \
        ["bc_nll__seed20261017__stage-full"]
    assert selection["runs"]["bc_nll__seed20261017__stage-full"]["stage"] \
        == "full"
    assert list(selection["screening"].keys()) == \
        ["bc_nll__seed20261017__stage-short"]
    assert selection["screening"]["bc_nll__seed20261017__stage-short"]["stage"] \
        == "short"
    assert list(selection["canary_record"].keys()) == \
        ["bc_nll__seed20261017__stage-fit1"]
    assert selection["canary_record"]["bc_nll__seed20261017__stage-fit1"]["stage"] \
        == "fit1"
    # the NEW record name only; the historical list-format name is untouched
    assert (tmp_path / SELECTION_RECORD_NAME).is_file()
    assert not (tmp_path / "selection_record.json").exists()


# ---------------------------------------------------------------------------
# Finding 5: IQL exposure snapshots + near-TS integrity
# ---------------------------------------------------------------------------

def test_iql_formal_run_refuses_missing_ral(tmp_path):
    from tests.pesn.test_dual_ts_iql import _chain, _settings, _trainer
    buf = EpisodeKeyedBuffer(_chain("R1", "A", "E1", 2, 1))
    settings = _settings(diagnostic_run=False)
    with pytest.raises(DualTSIQLTrainerError) as err:
        _trainer(tmp_path, buf, settings)
    assert err.value.code == "RAL_INTEGRITY_VIOLATION"


def test_iql_exposure_history_snapshots_and_diagnostic_near_ts_none(tmp_path):
    from tests.pesn.test_dual_ts_iql import _settings, _standard_buffer, \
        _trainer
    buf = _standard_buffer()  # rows carry remaining_arc_length=None
    settings = _settings(warmup_updates=4, joint_max_updates=1010)
    trainer = _trainer(tmp_path, buf, settings, name="snap")
    trainer.run_updates(4)     # warmup → joint transition
    trainer.run_updates(996)   # cross global update 1000
    trainer.run_updates(10)
    rec = trainer.run_record()
    kinds = [h["kind"] for h in rec["exposure"]["history"]]
    assert "phase_transition" in kinds
    assert "update_1000" in kinds
    assert "final" in kinds
    assert rec["exposure"]["near_ts_fraction"] is None
    assert rec["exposure"]["near_ts_unknown_rows"] > 0
