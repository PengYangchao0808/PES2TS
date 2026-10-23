"""V5.3-MP closed-loop rollout runner unit tests (gate V53MP-S3-closed-loop).

Driver under test: ``scripts/agent_ops/feature_gate/v53mp_rollout_exec.py``
(the NEW V5.3-MP runner; ``v53_rollout_exec.py`` stays untouched for the
frozen R3/V5.2 campaign).

Coverage (task spec items a–d + discipline):

* (a) reference-normalized budget formula ``max(40, ceil(1.5·L/0.08))``
  incl. the floor behaviour just below the raw-formula boundary;
* (b) curve-selection determinism (input-order independent) + dev-only
  guarantee against the real frozen ``v53mp_split_v1`` manifest
  (pool = training data, confirm = sealed — neither may be selected);
* (c) ``L_key`` on a tiny synthetic arc (Frobenius over KEY atoms only —
  a huge spectator-atom displacement must not contribute) + the budget
  it feeds;
* (d) episode loop with a small real ``KeyAtomV53MPModel`` state_dict:
  record schema fields, JSON-safety, bitwise repeatability, arm C
  online-stop firing when ``endpoint_logit`` is forced high, and the
  typed refusal path when the model emits non-finite output (no action
  ever fabricated);
* aggregation stays per ``arm × budget_mode`` (never a merged success
  rate);
* claim discipline: ``claim_scope`` on every record, forbidden claim
  strings only in the sanctioned negation note;
* campaign-root refusal: outputs must land under ``outputs/pesn/v53mp``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.feature_gate.v53mp_rollout_exec import (  # noqa: E402
    AGGREGATE_SCHEMA_V53MP,
    CLAIM_SCOPE,
    EPISODE_SCHEMA_V53MP,
    EpisodeContextV53MP,
    RolloutExecError,
    aggregate_episode_records,
    check_claim_discipline,
    curve_list_sha256,
    episode_budget,
    l_key_from_arc,
    reference_arc_nodes,
    reference_budget,
    run_episode,
    run_rollouts,
    select_rollout_curves,
)
from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    IntentRecordV3,
)

SPLIT_PATH = (
    PROJECT_ROOT / "outputs" / "pesn" / "v53mp" / "data"
    / "split_manifest_v53mp.json")


# ---------------------------------------------------------------------------
# (a) budget formula
# ---------------------------------------------------------------------------
def test_reference_budget_formula_and_floor() -> None:
    # raw formula below the floor: ceil(1.5 * 2.0 / 0.08) = 38 -> floor 40
    assert reference_budget(2.0) == 40
    assert reference_budget(0.0) == 40
    assert reference_budget(0.05) == 40
    # above the floor: ceil(1.5 * 2.2 / 0.08) = ceil(41.25) = 42
    assert reference_budget(2.2) == 42
    # ceil(1.5 * 6.0 / 0.08) = ceil(112.5) = 113
    assert reference_budget(6.0) == 113
    # monotone non-decreasing
    budgets = [reference_budget(x) for x in
               (0.0, 1.0, 2.0, 2.2, 4.0, 8.0, 20.0)]
    assert budgets == sorted(budgets)


def test_reference_budget_rejects_invalid_lkey() -> None:
    with pytest.raises(RolloutExecError):
        reference_budget(-1.0)
    with pytest.raises(RolloutExecError):
        reference_budget(float("nan"))


def test_episode_budget_modes_and_smoke_cap() -> None:
    assert episode_budget(
        budget_mode="reference_normalized", l_key=6.0,
        autonomous_steps=200) == 113
    assert episode_budget(
        budget_mode="autonomous", l_key=6.0,
        autonomous_steps=200) == 200
    # smoke cap clamps BOTH modes to 20
    assert episode_budget(
        budget_mode="reference_normalized", l_key=50.0,
        autonomous_steps=200, step_cap=20) == 20
    assert episode_budget(
        budget_mode="autonomous", l_key=50.0,
        autonomous_steps=200, step_cap=20) == 20
    with pytest.raises(RolloutExecError):
        episode_budget(budget_mode="merged", l_key=1.0,
                       autonomous_steps=10)


# ---------------------------------------------------------------------------
# (b) curve selection: determinism + dev-only
# ---------------------------------------------------------------------------
def test_curve_selection_deterministic_input_order_invariant() -> None:
    ids = [f"RXN_{i:09d}" for i in range(40)]
    sel_a = select_rollout_curves(ids, 7)
    sel_b = select_rollout_curves(list(reversed(ids)), 7)
    shuffled = ids[::2] + ids[1::2]
    sel_c = select_rollout_curves(shuffled, 7)
    assert sel_a == sel_b == sel_c
    assert len(sel_a) == 7 == len(set(sel_a))
    assert set(sel_a) <= set(ids)
    # list sha: stable for the same list, changes when membership changes
    assert curve_list_sha256(sel_a) == curve_list_sha256(list(sel_a))
    assert curve_list_sha256(sel_a) != curve_list_sha256(sel_a[1:])
    with pytest.raises(RolloutExecError):
        select_rollout_curves(ids, len(ids) + 1)
    with pytest.raises(RolloutExecError):
        select_rollout_curves([], 1)


def test_curve_selection_matches_recorded_rule() -> None:
    """Ordering follows sha256('v53mp-rollout|17|<curve_id>') exactly."""
    import hashlib
    ids = [f"RXN_{i:09d}" for i in range(25)]
    sel = select_rollout_curves(ids, 5)
    expect = sorted(
        ids,
        key=lambda c: hashlib.sha256(
            f"v53mp-rollout|17|{c}".encode("utf-8")).hexdigest())[:5]
    assert sel == expect


@pytest.mark.skipif(
    not SPLIT_PATH.is_file(),
    reason="frozen v53mp split manifest not present in worktree")
def test_curve_selection_dev_only_against_real_split() -> None:
    doc = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    assert doc.get("schema") == "v53mp_split_v1"
    assert doc.get("frozen") is True
    dev = [str(c) for c in doc["subsets"]["dev"]]
    confirm = {str(c) for c in doc["subsets"]["confirm"]}
    pool: set[str] = set()
    for ids in doc["subsets"]["pool"].values():
        pool.update(str(c) for c in ids)

    selected = select_rollout_curves(dev, 24)
    assert len(selected) == 24
    assert set(selected) <= set(dev)
    # dev is disjoint from the sealed confirm + training pool ⇒ the
    # rollout set can never touch either
    assert not set(selected) & confirm
    assert not set(selected) & pool
    # deterministic against the frozen artifact
    assert selected == select_rollout_curves(list(reversed(dev)), 24)


# ---------------------------------------------------------------------------
# (c) L_key on a tiny synthetic arc
# ---------------------------------------------------------------------------
def test_l_key_synthetic_arc_key_atoms_only() -> None:
    # 3 consecutive frames; atom 3 is a HUGE-moving spectator outside K
    f0 = np.array([[0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [3.0, 0.0, 0.0],
                   [0.0, 5.0, 0.0]])
    f1 = np.array([[0.0, 0.0, 0.0],
                   [1.5, 0.0, 0.0],
                   [2.5, 0.0, 0.0],
                   [0.0, 50.0, 0.0]])      # spectator jumps 45 Å — ignored
    f2 = np.array([[0.0, 0.0, 0.0],
                   [2.0, 0.0, 0.0],
                   [2.0, 0.0, 0.0],
                   [0.0, 90.0, 0.0]])      # spectator jumps again — ignored
    key_rows = (0, 1, 2)
    key_idx = np.asarray(key_rows, dtype=np.int64)
    key_frames = [f[key_idx] for f in (f0, f1, f2)]
    # per-step Frobenius over key atoms: sqrt(0.5^2+0.5^2) each → 2·√0.5
    expected = 2.0 * float(np.sqrt(0.5))
    assert abs(l_key_from_arc(key_frames) - expected) < 1e-12
    # the resulting budget uses the same frozen formula
    assert reference_budget(l_key_from_arc(key_frames)) == 40  # floor
    # a long arc escapes the floor: L=8 m → ceil(150)=150
    assert reference_budget(8.0) == 150
    with pytest.raises(RolloutExecError):
        l_key_from_arc([f0])
    assert reference_arc_nodes(0, 2) == [0, 1, 2]
    assert reference_arc_nodes(2, 0) == [2, 1, 0]


# ---------------------------------------------------------------------------
# (d) episode loop with a small model
# ---------------------------------------------------------------------------
def _synthetic_intent() -> IntentRecordV3:
    """A–H breaking, H–B forming; K = {A, H, B}; two bond-length CVs."""
    return IntentRecordV3(
        intent_id="SYN_fwd",
        curve_id="SYN",
        direction="forward",
        reverse_of_intent_id="SYN_rev",
        intent_pair_id="SYN_pair",
        g_src_bonds=((0, 1, 1.0),),
        g_tgt_bonds=((1, 2, 1.0),),
        cv_deltas=(1.0,),
        g_atoms=(6, 1, 8, 6),
        cv_coordinates=((0, 1), (1, 2)),
        cv_kind="graph_edit_bond_lengths",
    )


def _synthetic_frames() -> list[np.ndarray]:
    return [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [3.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                  [2.5, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
    ]


def _synthetic_ctx() -> EpisodeContextV53MP:
    frames = _synthetic_frames()
    intent = _synthetic_intent()
    key_rows = (0, 1, 2)
    cv_pairs = [(0, 1), (1, 2)]

    def dkey(f: np.ndarray) -> np.ndarray:
        return np.array([float(np.linalg.norm(f[j] - f[i]))
                         for i, j in cv_pairs])

    key_idx = np.asarray(key_rows, dtype=np.int64)
    l_key = l_key_from_arc([f[key_idx] for f in frames])
    return EpisodeContextV53MP(
        curve_id="SYN",
        direction="forward",
        intent=intent,
        start_node=0,
        target_node=2,
        ts_index=1,
        atomic_numbers=(6, 1, 8, 6),
        start_xyz=tuple(tuple(float(v) for v in row) for row in frames[0]),
        arc_nodes=(0, 1, 2),
        arc_dkey=(dkey(frames[0]), dkey(frames[1]), dkey(frames[2])),
        target_xyz=tuple(tuple(float(v) for v in row)
                         for row in frames[2]),
        ts_xyz=tuple(tuple(float(v) for v in row) for row in frames[1]),
        l_key=float(l_key),
        reference_budget=reference_budget(l_key),
    )


def _tiny_model(tmp_path: Path, *, force_high_endpoint: bool = False):
    """Real KeyAtomV53MPModel round-tripped through a state_dict file."""
    torch = pytest.importorskip("torch")
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    torch.manual_seed(0)
    model = KeyAtomV53MPModel()
    ckpt = tmp_path / "checkpoint_best.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    from scripts.agent_ops.feature_gate.v53mp_rollout_exec import (
        load_v53mp_checkpoint,
    )
    loaded = load_v53mp_checkpoint(ckpt)  # strict load, eval mode
    if force_high_endpoint:
        # force sigmoid(endpoint_logit) ≈ 0.9997 ≥ tau_end=0.5
        with torch.no_grad():
            loaded.process_head.endpoint_head.fc.weight.zero_()
            loaded.process_head.endpoint_head.fc.bias.fill_(8.0)
    return loaded


def _forced_nan_model(tmp_path: Path):
    torch = pytest.importorskip("torch")
    model = _tiny_model(tmp_path)
    with torch.no_grad():
        model.backbone.node_proj.weight.fill_(float("nan"))
    return model


_EPISODE_REQUIRED_FIELDS = (
    "schema", "curve_id", "direction", "intent_id", "start_node_index",
    "target_node_index", "ts_index_record_only", "arm", "seed", "run_id",
    "checkpoint_sha256", "tau_end", "budget_mode", "budget", "L_key",
    "rollout_seed", "level", "device", "trust_region_max_atom",
    "collision_distance", "n_steps_requested", "n_steps_executed",
    "input_step0_sha256", "halted_online_stop", "online_stop_step",
    "online_stop_role", "claim_scope", "refusal", "steps", "final_xyz",
    "summary", "wall_seconds",
)

_SUMMARY_REQUIRED_FIELDS = (
    "n_steps_executed", "endpoint_reached", "endpoint_reached_step",
    "endpoint_reached_tol30", "endpoint_reached_tol30_step",
    "ts_candidate_crossed", "ts_candidate_crossed_step",
    "arc_progress_series", "final_arc_progress", "first_no_move_step",
    "no_move_distance_max_dkey", "no_move_arc_fraction_remaining",
    "stopped_online", "n_collisions", "n_nonfinite", "n_clip_events",
    "cv_sign_agree_fraction", "total_displacement", "mean_clip_factor",
    "max_atom_excursion", "endpoint_not_reached", "budget_exhausted",
    "refusal_code", "final_dev_target_max", "final_dev_ts_max",
    "final_target_rmsd_all", "final_target_rmsd_key",
    "final_ts_rmsd_all", "final_ts_rmsd_key",
)


def test_episode_record_schema_arm_a(tmp_path: Path) -> None:
    model = _tiny_model(tmp_path)
    rec = run_episode(
        _synthetic_ctx(),
        arm="A",
        model=model,
        budget=3,
        budget_mode="autonomous",
        run_id="v53mp_20k_A_s17",
        checkpoint_sha256="a" * 64,
        seed=17,
        tau_end=None,
    )
    for key in _EPISODE_REQUIRED_FIELDS:
        assert key in rec, key
    assert rec["schema"] == EPISODE_SCHEMA_V53MP
    assert rec["claim_scope"] == CLAIM_SCOPE == "TS_candidate_only"
    assert rec["budget_mode"] == "autonomous"
    assert rec["budget"] == 3 == rec["n_steps_requested"]
    assert rec["n_steps_executed"] == 3
    assert len(rec["steps"]) == 3
    assert rec["refusal"] is None
    assert rec["halted_online_stop"] is False
    assert rec["online_stop_step"] is None
    assert rec["online_stop_role"] is None     # arms A/B: no stop signal
    assert rec["arm"] == "A" and rec["seed"] == 17
    assert rec["run_id"] == "v53mp_20k_A_s17"
    assert rec["checkpoint_sha256"] == "a" * 64
    assert rec["tau_end"] is None
    assert isinstance(rec["L_key"], float)

    for key in _SUMMARY_REQUIRED_FIELDS:
        assert key in rec["summary"], key
    s = rec["summary"]
    # start dev from the target endpoint is 2.0 Å over both CVs; with
    # ≤0.20 Å/atom × 3 steps no episode can enter the 0.10/0.30 windows
    assert s["endpoint_reached"] is False
    assert s["endpoint_reached_tol30"] is False
    assert s["budget_exhausted"] is True      # full budget, no arrival
    assert s["endpoint_not_reached"] is True
    assert s["first_no_move_step"] is None    # only arm C can stop
    assert len(s["arc_progress_series"]) == 3
    assert s["final_arc_progress"] is not None
    assert s["n_nonfinite"] == 0 and s["refusal_code"] is None

    step = rec["steps"][0]
    for key in ("step", "u_max_atom", "clip_factor", "disp_l1_sum",
                "d_key", "d_key_delta", "cv_sign_agree",
                "min_interatomic", "collision", "geometry_finite",
                "endpoint_dev_max", "ts_dev_max", "arc_progress",
                "q_pred", "endpoint_prob"):
        assert key in step, key
    assert len(step["d_key"]) == 2
    assert len(step["cv_sign_agree"]) == 2
    assert step["endpoint_prob"] is not None  # KeyAtomV53MPModel emits it
    assert 0.0 <= step["endpoint_prob"] <= 1.0
    assert step["u_max_atom"] <= 0.20 + 1e-12  # trust region enforced

    # JSON-safe round trip
    assert json.loads(json.dumps(rec, sort_keys=True)) == rec
    # forbidden claim strings never appear in a record
    text = json.dumps(rec).lower()
    assert "verified_ts" not in text
    assert "verified path" not in text

    # determinism: same seed + same inputs ⇒ bitwise-equal record
    rerun = run_episode(
        _synthetic_ctx(),
        arm="A",
        model=model,
        budget=3,
        budget_mode="autonomous",
        run_id="v53mp_20k_A_s17",
        checkpoint_sha256="a" * 64,
        seed=17,
        tau_end=None,
    )
    for key in ("steps", "summary", "input_step0_sha256"):
        assert json.dumps(rec[key], sort_keys=True) == \
            json.dumps(rerun[key], sort_keys=True)


def test_episode_arm_c_online_stop_fires_when_forced_high(
        tmp_path: Path) -> None:
    model = _tiny_model(tmp_path, force_high_endpoint=True)
    rec = run_episode(
        _synthetic_ctx(),
        arm="C",
        model=model,
        budget=5,
        budget_mode="autonomous",
        run_id="v53mp_20k_C_s17",
        checkpoint_sha256="c" * 64,
        seed=17,
        tau_end=0.5,
    )
    assert rec["n_steps_executed"] == 1        # halted at the first step
    assert len(rec["steps"]) == 1
    assert rec["steps"][0]["endpoint_prob"] >= 0.5
    assert rec["halted_online_stop"] is True
    assert rec["online_stop_step"] == 1
    assert rec["online_stop_role"] == "diagnostic_only"
    s = rec["summary"]
    assert s["stopped_online"] is True
    assert s["first_no_move_step"] == 1
    assert s["no_move_distance_max_dkey"] == \
        rec["steps"][0]["endpoint_dev_max"]
    assert s["no_move_arc_fraction_remaining"] is not None
    assert 0.0 <= s["no_move_arc_fraction_remaining"] <= 1.0
    # did NOT exhaust the budget (stopped by the diagnostic signal)
    assert s["budget_exhausted"] is False
    assert s["endpoint_reached"] is False
    assert rec["tau_end"] == 0.5
    assert rec["claim_scope"] == CLAIM_SCOPE
    assert json.loads(json.dumps(rec, sort_keys=True)) == rec


def test_episode_arm_c_requires_tau_end(tmp_path: Path) -> None:
    model = _tiny_model(tmp_path)
    with pytest.raises(RolloutExecError):
        run_episode(
            _synthetic_ctx(), arm="C", model=model, budget=2,
            budget_mode="autonomous", run_id="r", checkpoint_sha256="c" * 64,
            seed=17, tau_end=None)


def test_episode_refuses_nonfinite_model_output(tmp_path: Path) -> None:
    """Non-finite MODEL output = typed refusal; never a fabricated action."""
    model = _forced_nan_model(tmp_path)
    rec = run_episode(
        _synthetic_ctx(),
        arm="A",
        model=model,
        budget=4,
        budget_mode="reference_normalized",
        run_id="v53mp_20k_A_s17",
        checkpoint_sha256="a" * 64,
        seed=17,
        tau_end=None,
    )
    assert rec["refusal"] is not None
    assert rec["refusal"]["code"] == "NONFINITE_MODEL_OUTPUT"
    assert rec["refusal"]["action"] == "no_action_fabricated"
    assert rec["refusal"]["step"] == 1
    assert rec["n_steps_executed"] == 0
    assert rec["steps"] == []
    s = rec["summary"]
    assert s["refusal_code"] == "NONFINITE_MODEL_OUTPUT"
    assert s["budget_exhausted"] is False      # aborted, not exhausted
    assert s["n_nonfinite"] == 0               # geometry never fabricated
    assert rec["claim_scope"] == CLAIM_SCOPE
    assert json.loads(json.dumps(rec, sort_keys=True)) == rec


# ---------------------------------------------------------------------------
# aggregation: separate by budget_mode, never merged
# ---------------------------------------------------------------------------
def _fake_record(arm: str, budget_mode: str, *, reached: bool,
                 stopped: bool = False) -> dict:
    return {
        "schema": EPISODE_SCHEMA_V53MP,
        "claim_scope": CLAIM_SCOPE,
        "arm": arm,
        "seed": 17,
        "curve_id": f"RXN_{arm}_{budget_mode}",
        "direction": "forward",
        "budget_mode": budget_mode,
        "budget": 40,
        "refusal": None,
        "summary": {
            "endpoint_reached": reached,
            "endpoint_reached_tol30": reached,
            "ts_candidate_crossed": False,
            "first_no_move_step": (3 if stopped else None),
            "no_move_distance_max_dkey": (0.4 if stopped else None),
            "no_move_arc_fraction_remaining": (0.5 if stopped else None),
            "n_collisions": 1 if reached else 0,
            "n_nonfinite": 0,
            "budget_exhausted": not reached and not stopped,
            "stopped_online": stopped,
        },
    }


def test_aggregate_separates_budget_modes() -> None:
    records = [
        _fake_record("A", "reference_normalized", reached=True),
        _fake_record("A", "reference_normalized", reached=False),
        _fake_record("A", "autonomous", reached=False),
        _fake_record("C", "autonomous", reached=False, stopped=True),
    ]
    agg = aggregate_episode_records(records)
    assert agg["schema"] == AGGREGATE_SCHEMA_V53MP
    keys = [(g["arm"], g["budget_mode"]) for g in agg["groups"]]
    assert keys == [
        ("A", "autonomous"),
        ("A", "reference_normalized"),
        ("C", "autonomous"),
    ], "groups must be keyed per arm x budget_mode (never merged)"
    # no global/merged success-rate container exists
    assert "overall" not in agg and "success_rate" not in agg

    by_key = {(g["arm"], g["budget_mode"]): g for g in agg["groups"]}
    g_ref = by_key[("A", "reference_normalized")]
    g_auto = by_key[("A", "autonomous")]
    assert abs(g_ref["endpoint_arrival_rate"] - 0.5) < 1e-12
    assert g_auto["endpoint_arrival_rate"] == 0.0
    assert g_ref["collisions_total"] == 1
    assert g_ref["budget_exhaustion_fraction"] == 0.5
    g_c = by_key[("C", "autonomous")]
    assert g_c["online_stop_halted_episodes"] == 1
    dist = g_c["first_no_move_distance_summary"]
    assert dist["n"] == 1 and dist["mean"] == 0.4
    # groups without stops report an empty distribution, not zeros
    assert g_ref["first_no_move_distance_summary"]["n"] == 0
    assert g_ref["first_no_move_distance_summary"]["mean"] is None


def test_claim_discipline_rejects_tampering() -> None:
    good = _fake_record("A", "autonomous", reached=True)
    check_claim_discipline([good], {"verified_ts_claim": "forbidden"})

    missing_scope = dict(good)
    del missing_scope["claim_scope"]
    with pytest.raises(RolloutExecError):
        check_claim_discipline([missing_scope])

    banned = dict(good)
    banned["summary"] = dict(good["summary"])
    banned["summary"]["note"] = "this is a verified path claim"
    with pytest.raises(RolloutExecError):
        check_claim_discipline([banned])

    banned_ts = dict(good)
    banned_ts["extra"] = "see verified_ts evidence"
    with pytest.raises(RolloutExecError):
        check_claim_discipline([banned_ts])

    # manifest: the negation note must exist…
    with pytest.raises(RolloutExecError):
        check_claim_discipline([good], {"verified_ts_claim": "yes"})
    # …and the banned literals may live ONLY inside that negation note
    with pytest.raises(RolloutExecError):
        check_claim_discipline([good], {
            "verified_ts_claim": "forbidden",
            "extra": "see verified_ts evidence"})
    with pytest.raises(RolloutExecError):
        check_claim_discipline([good], {
            "verified_ts_claim": "forbidden",
            "extra": "a verified path was found"})


# ---------------------------------------------------------------------------
# campaign-root discipline
# ---------------------------------------------------------------------------
def test_run_rollouts_refuses_outside_campaign_root() -> None:
    from scripts.agent_ops.supervisor.campaign_root import CampaignRootError

    args = argparse.Namespace(
        out_dir="outputs/pesn/not_v53mp/rollouts",
        runs_root="outputs/pesn/v53mp",
        tables_dir="outputs/pesn/v53mp/data",
        split="outputs/pesn/v53mp/data/split_manifest_v53mp.json",
        arms="A,B", seeds="17", n_curves=2, autonomous_steps=200,
        tier="20k", cache_curves=8, smoke=True,
    )
    with pytest.raises(CampaignRootError):
        run_rollouts(args)
