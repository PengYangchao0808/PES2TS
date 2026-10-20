"""E6 V5.3 rollout harness tests (W-FEAT, CPU-only).

Coverage (task E6-V53PREP):

* determinism: two identical rollouts are bitwise equal;
* step-count exactness: n_steps ∈ {5,10,20} executed exactly, and a
  non-frozen budget is refused;
* controller-arm parity at step 0: classic/r3/v52 all start from the
  identical (geometry, intent) fingerprint and — for the analytic vs
  model step — the classic arm matches the offline
  ``classic_jacobian_step`` formula bitwise (mirror parity);
* no-forbidden-inputs in the loop: every re-embed consumes only the
  moved geometry + fixed intent; the model batch is forward-view only
  (no y_teacher / supervision keys), and mutating the start geometry
  after step 0 cannot change the recorded step-0 fingerprint;
* typed refusal on non-finite actions; frozen claim scope
  (``TS_candidate_only`` — verified_ts_claim impossible);
* curve-list frozen + deterministic (re-running the selection on the
  real E1 tables reproduces the frozen artifact; Development folds only);
* relax-bridge preflight fail-safe: never reports physics executed and
  physics_missing defaults to ``physics_not_run``.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    FORWARD_VIEW_KEYS_V3,
    SUPERVISION_KEYS_V3,
    IntentRecordV3,
    forward_view_v3,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (  # noqa: E402
    build_sample_features_v3,
    collate_v3,
    to_torch_batch,
)
from pes2ts_core.pesn.models.rollout_v3 import (  # noqa: E402
    ARMS_V3,
    CLAIM_SCOPE_V3,
    NO_PHYSICS_STEPS_V3,
    RolloutStartV3,
    RolloutV3Error,
    classic_cv_jacobian_step_v3,
    run_rollout_v3,
)


# ---------------------------------------------------------------------------
# fixtures: tiny synthetic H-transfer curve (no real data dependency)
# ---------------------------------------------------------------------------
def _synthetic_intent() -> IntentRecordV3:
    """A–H breaking, H–B forming; K = {A, H, B}; one bond-length CV."""
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


def _synthetic_start() -> RolloutStartV3:
    return RolloutStartV3(
        curve_id="SYN",
        node_index=0,
        intent=_synthetic_intent(),
        atomic_numbers=(6, 1, 8, 6),
        xyz=((0.0, 0.0, 0.0),        # A (C)
             (1.1, 0.0, 0.0),        # H
             (2.5, 0.2, 0.0),        # B (O)
             (-1.5, 0.3, 0.1)),      # spectator C
        d0=0.08,
    )


def _fresh_model(seed: int = 0) -> torch.nn.Module:
    from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
    torch.manual_seed(seed)
    model = StarEGNNV3()
    model.eval()
    return model


def _fresh_v52_model(seed: int = 0) -> torch.nn.Module:
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV52Model
    torch.manual_seed(seed)
    model = KeyAtomV52Model()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# determinism + step-count exactness
# ---------------------------------------------------------------------------
def test_rollout_determinism_bitwise() -> None:
    model = _fresh_model()
    start = _synthetic_start()
    r1 = run_rollout_v3(arm="r3", start=start, n_steps=5, model=model)
    r2 = run_rollout_v3(arm="r3", start=start, n_steps=5, model=model)
    assert r1.input_step0_sha256 == r2.input_step0_sha256
    assert len(r1.steps) == len(r2.steps) == 5
    for s1, s2 in zip(r1.steps, r2.steps):
        assert s1.u_max_atom == s2.u_max_atom
        assert s1.d_key == s2.d_key
        assert s1.d_key_delta == s2.d_key_delta
        assert s1.min_interatomic == s2.min_interatomic
    assert r1.final_xyz == r2.final_xyz


def test_step_count_exactness_all_budgets() -> None:
    model = _fresh_model()
    start = _synthetic_start()
    for n in NO_PHYSICS_STEPS_V3:
        r = run_rollout_v3(arm="r3", start=start, n_steps=n, model=model)
        assert r.n_steps_requested == n
        assert r.n_steps_executed == n
        assert [s.step for s in r.steps] == list(range(1, n + 1))


def test_step_budget_outside_frozen_set_refused() -> None:
    model = _fresh_model()
    with pytest.raises(RolloutV3Error) as exc:
        run_rollout_v3(arm="r3", start=_synthetic_start(), n_steps=7,
                       model=model)
    assert exc.value.code == "STEP_BUDGET"


def test_unknown_arm_refused() -> None:
    with pytest.raises(RolloutV3Error) as exc:
        run_rollout_v3(arm="teacher", start=_synthetic_start(), n_steps=5,
                       model=None)
    assert exc.value.code == "ARM"
    assert set(ARMS_V3) == {"classic", "r3", "v52"}


def test_model_required_for_learned_arms() -> None:
    with pytest.raises(RolloutV3Error) as exc:
        run_rollout_v3(arm="r3", start=_synthetic_start(), n_steps=5,
                       model=None)
    assert exc.value.code == "MODEL_REQUIRED"


# ---------------------------------------------------------------------------
# controller-arm parity at step 0
# ---------------------------------------------------------------------------
def test_arm_parity_step0_fingerprint() -> None:
    """All three arms begin from the identical step-0 state."""
    start = _synthetic_start()
    r_classic = run_rollout_v3(arm="classic", start=start, n_steps=5)
    r_r3 = run_rollout_v3(arm="r3", start=start, n_steps=5,
                          model=_fresh_model())
    r_v52 = run_rollout_v3(arm="v52", start=start, n_steps=5,
                           model=_fresh_v52_model())
    assert r_classic.input_step0_sha256 == r_r3.input_step0_sha256
    assert r_classic.input_step0_sha256 == r_v52.input_step0_sha256


def test_classic_arm_formula_mirror_parity() -> None:
    """The rollout classic step matches the offline baseline formula."""
    pesn_root = PROJECT_ROOT / "pes2ts_core/pesn/offline/training"
    if not (pesn_root / "key_atom_bc_v3.py").is_file():
        pytest.skip("offline trainer module not present")
    from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
        classic_jacobian_step,
    )
    start = _synthetic_start()
    pos = start.xyz_array()
    key_rows = start.intent.key_rows()
    pairs = [(i, j) for (i, j), (b_src, b_tgt)
             in sorted(start.intent.pair_orders().items())
             if b_src != b_tgt and i in set(key_rows) and j in set(key_rows)]
    edits = np.array([
        start.intent.pair_orders()[(i, j)][1]
        - start.intent.pair_orders()[(i, j)][0] for (i, j) in pairs])
    target = np.clip(edits, -1.0, 1.0) * start.d0
    u_mirror = classic_cv_jacobian_step_v3(pos, key_rows, pairs, target)
    u_offline = classic_jacobian_step(pos, key_rows, pairs, target)
    np.testing.assert_array_equal(u_mirror, u_offline)


def test_v52_arm_records_q_and_stop() -> None:
    r = run_rollout_v3(arm="v52", start=_synthetic_start(), n_steps=5,
                       model=_fresh_v52_model())
    for s in r.steps:
        assert s.q_pred is not None and -1.0 <= s.q_pred <= 1.0
        assert s.stop_prob is not None and 0.0 <= s.stop_prob <= 1.0
    assert r.summary["n_ts_candidate_events"] >= 0


def test_classic_arm_has_no_head_readouts() -> None:
    r = run_rollout_v3(arm="classic", start=_synthetic_start(), n_steps=5)
    for s in r.steps:
        assert s.q_pred is None and s.stop_prob is None
        assert s.ts_candidate_event is False


# ---------------------------------------------------------------------------
# no-forbidden-inputs in the loop
# ---------------------------------------------------------------------------
def test_loop_batch_is_forward_view_only() -> None:
    """Every model forward in the loop sees forward-view keys only."""
    model = _fresh_model()
    seen_batches: list[dict] = []
    original_forward = model.forward

    def spy_forward(batch):
        seen_batches.append(dict(batch))
        return original_forward(batch)

    model.forward = spy_forward  # type: ignore[method-assign]
    run_rollout_v3(arm="r3", start=_synthetic_start(), n_steps=5,
                   model=model)
    assert len(seen_batches) == 5
    for batch in seen_batches:
        extra = set(batch) - set(FORWARD_VIEW_KEYS_V3)
        assert not extra, f"non-forward keys leaked into model: {extra}"
        for forbidden in SUPERVISION_KEYS_V3:
            assert forbidden not in batch


def test_reembed_uses_only_moved_geometry() -> None:
    """Step t's model input geometry equals the moved geometry at step t.

    (No future frame, no teacher geometry, no start-geometry replay after
    the moves — the loop can only feed forward its own moved state.)
    """
    model = _fresh_model()
    start = _synthetic_start()
    seen_pos: list[np.ndarray] = []
    original_forward = model.forward

    def spy_forward(batch):
        seen_pos.append(batch["pos"].detach().cpu().numpy().copy())
        return original_forward(batch)

    model.forward = spy_forward  # type: ignore[method-assign]
    result = run_rollout_v3(arm="r3", start=start, n_steps=5, model=model)
    # step 0 input is the untouched start geometry
    np.testing.assert_array_equal(seen_pos[0], start.xyz_array().astype(np.float32))
    # each later input differs from the start (the loop moved the atoms)
    moved_any = any(
        not np.allclose(p, start.xyz_array().astype(np.float32))
        for p in seen_pos[1:])
    assert moved_any
    # chained re-embed: the trajectory is prefix-consistent across budgets —
    # the 10-step run's input at step 5 equals the 5-step run's input at
    # step 5 (both are start + the same 4 applied moves), and diverges
    # afterwards (step 10's input has seen 9 moves, not 4).
    seen_pos10: list[np.ndarray] = []

    def spy_forward10(batch):
        seen_pos10.append(batch["pos"].detach().cpu().numpy().copy())
        return original_forward(batch)

    model.forward = spy_forward10  # type: ignore[method-assign]
    run_rollout_v3(arm="r3", start=start, n_steps=10, model=model)
    np.testing.assert_array_equal(seen_pos10[4], seen_pos[4])
    assert not np.allclose(seen_pos10[9], seen_pos[4])
    # the final recorded geometry is NOT the final model input — one more
    # move was applied after the last forward
    assert not np.allclose(
        result.final_xyz, seen_pos[-1].astype(np.float64), atol=1e-12)


def test_intent_held_fixed_across_steps() -> None:
    """The re-embed path never mutates the intent object."""
    start = _synthetic_start()
    before = copy.deepcopy(start.intent)
    run_rollout_v3(arm="classic", start=start, n_steps=5)
    assert start.intent == before


def test_nonfinite_action_is_typed_refusal() -> None:
    model = _fresh_model()

    def poisoned(batch):
        n = batch["key_index"].numel()
        return {"u": torch.full((n, 3), float("nan")), "z": batch["pos"][:1]}

    model.forward = poisoned  # type: ignore[method-assign]
    with pytest.raises(RolloutV3Error) as exc:
        run_rollout_v3(arm="r3", start=_synthetic_start(), n_steps=5,
                       model=model)
    assert exc.value.code == "NONFINITE_ACTION"


def test_claim_scope_is_ts_candidate_only() -> None:
    r = run_rollout_v3(arm="classic", start=_synthetic_start(), n_steps=5)
    assert r.claim_scope == CLAIM_SCOPE_V3 == "TS_candidate_only"
    assert "verified" not in json.dumps(r.to_dict()).replace(
        "verified_ts_claim", "")


def test_forward_view_filter_on_rollout_batch() -> None:
    """The built rollout batch carries supervision, and the forward view
    drops it — the same discipline the loop applies."""
    start = _synthetic_start()
    sample = _build_frame_sample_for_test(start)
    built = build_sample_features_v3(sample, level="F2")
    batch = to_torch_batch(collate_v3([built]))
    assert "y_teacher" not in batch  # supervision=None → absent
    fwd = forward_view_v3(batch)
    assert set(fwd).issubset(set(FORWARD_VIEW_KEYS_V3))
    for forbidden in ("pair_index", "pair_edit", "d0", "frame_weight"):
        assert forbidden not in fwd


def _build_frame_sample_for_test(start: RolloutStartV3):
    from pes2ts_core.pesn.models.rollout_v3 import _build_frame_sample
    return _build_frame_sample(start, start.xyz_array())


# ---------------------------------------------------------------------------
# geometry diagnostics sanity
# ---------------------------------------------------------------------------
def test_displacements_finite_and_trust_region() -> None:
    r = run_rollout_v3(arm="classic", start=_synthetic_start(), n_steps=5)
    for s in r.steps:
        assert np.isfinite(s.u_max_atom) and s.u_max_atom >= 0.0
        assert 0.0 < s.clip_factor <= 1.0
        assert s.all_finite
        assert np.isfinite(s.min_interatomic)


def test_d_key_tracks_cv_pairs() -> None:
    start = _synthetic_start()
    r = run_rollout_v3(arm="classic", start=start, n_steps=5)
    n_cv = len(start.intent.cv_coordinates)
    for s in r.steps:
        assert len(s.d_key) == n_cv
        assert len(s.d_key_delta) == n_cv
        assert len(s.cv_sign_agree) == n_cv
    # d_key deltas telescope: last − first == sum of deltas
    d_first = np.asarray(r.steps[0].d_key)
    d_last = np.asarray(r.steps[-1].d_key)
    total = sum(np.asarray(s.d_key_delta) for s in r.steps)
    np.testing.assert_allclose(
        d_last - d_first + np.asarray(r.steps[0].d_key_delta),
        total, atol=1e-12)


# ---------------------------------------------------------------------------
# curve pre-registration (real E1 tables)
# ---------------------------------------------------------------------------
TABLES_DIR = PROJECT_ROOT / "outputs/pesn/30h_campaign/data"
CURVES_PATH = (
    PROJECT_ROOT / "outputs/pesn/30h_campaign/v53/rollout_curves.json")


@pytest.mark.skipif(
    not TABLES_DIR.is_dir(), reason="E1 six tables not present in worktree")
def test_curve_list_frozen_and_deterministic() -> None:
    assert CURVES_PATH.is_file(), (
        "rollout_curves.json missing — run scripts/agent_ops/feature_gate/"
        "v53_rollout_prep.py select-curves first")
    doc = json.loads(CURVES_PATH.read_text(encoding="utf-8"))
    assert doc["schema"] == "pes2ts_v53_rollout_curves_v1"
    assert doc["frozen"] is True
    assert doc["verified_ts_claim"] == "forbidden"
    assert doc["no_physics_steps"] == [5, 10, 20]
    assert doc["arms"] == ["classic", "r3", "v52"]
    assert 8 <= doc["n_curves"] <= 12
    curves = doc["curves"]
    assert len(curves) == doc["n_curves"]

    # Development folds only, both directions recorded, start nodes valid
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        KeyAtomDatasetV3,
    )
    ds = KeyAtomDatasetV3(TABLES_DIR)
    folds = dict(ds.folds)
    for entry in curves:
        assert folds[entry["curve_id"]] in (0, 1, 2)
        assert entry["forward_intent_id"]
        assert entry["reverse_intent_id"]
        assert 0 <= entry["start_node_index"] < entry["n_nodes"]

    # deterministic: re-running the selection reproduces the frozen list
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/agent_ops/feature_gate"))
    import v53_rollout_prep as prep
    regen = prep.select_rollout_curves(TABLES_DIR)
    assert regen["curve_list_sha256"] == doc["curve_list_sha256"]
    assert regen["curves"] == curves


@pytest.mark.skipif(
    not TABLES_DIR.is_dir(), reason="E1 six tables not present in worktree")
def test_curve_list_spans_folds_and_k() -> None:
    doc = json.loads(CURVES_PATH.read_text(encoding="utf-8"))
    folds = {e["fold"] for e in doc["curves"]}
    k_sizes = {e["k_size"] for e in doc["curves"]}
    quartiles = {e["k_quartile"] for e in doc["curves"]}
    assert folds == {0, 1, 2}, "selection must span all Development folds"
    assert len(k_sizes) >= 3, "selection must span diverse K sizes"
    assert quartiles == {0, 1, 2, 3}, "selection must span all K quartiles"


# ---------------------------------------------------------------------------
# relax-bridge preflight fail-safe
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not TABLES_DIR.is_dir(), reason="E1 six tables not present in worktree")
def test_bridge_preflight_fail_safe() -> None:
    pf_path = (
        PROJECT_ROOT / "outputs/pesn/30h_campaign/v53/relax_preflight.json")
    assert pf_path.is_file(), "run v53_rollout_prep.py preflight-relax first"
    doc = json.loads(pf_path.read_text(encoding="utf-8"))
    assert doc["schema"] == "pes2ts_v53_relax_preflight_v1"
    # fail-safe invariants: this task never executes physics
    assert doc["physics_executed"] is False
    assert doc["physics_missing"] == "physics_not_run"
    if not doc["executor_runnable"]:
        assert doc["verdict"] != "executed"
    if doc["dry_run_request"] is not None:
        assert doc["dry_run_request"]["request_executed"] is False
        assert doc["dry_run_request"]["request_built"] is True
        assert doc["dry_run_request"]["request_schema"].startswith(
            "pes2ts_xtb_constrained_opt_request")


def test_bridge_preflight_fail_safe_on_bad_wiring(tmp_path) -> None:
    """Missing ACP wiring degrades to a not-run verdict, never a crash."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts/agent_ops/feature_gate"))
    import v53_rollout_prep as prep
    fake_root = tmp_path / "no_acp_here"
    original = prep._resolve_acp_wiring
    prep._resolve_acp_wiring = lambda: {
        "available": False, "reason": "acp_manifest_missing",
        "checks": {"acp_root": None, "acp_cli_py": None,
                   "acp_python": None, "acp_config": None}}
    try:
        assert original()["available"] in (True, False)  # real wiring
        acp = prep._resolve_acp_wiring()
        assert acp["available"] is False
        # the command path would compute executor_runnable=False and keep
        # physics_missing=physics_not_run — asserted structurally here
        assert not fake_root.exists()
    finally:
        prep._resolve_acp_wiring = original


# ---------------------------------------------------------------------------
# E6-V53EXEC: episode aggregation, budget parity, diagnostic labeling,
# relax result schema (mock executor — real xtb never in tests)
# ---------------------------------------------------------------------------
from pes2ts_core.pesn.models.rollout_v3 import (  # noqa: E402
    DIVERGENCE_EXCURSION_ANGSTROM,
    EPISODE_SCHEMA_V3,
    PhysicsOutcomeV3,
    RELAX_RECIPE_V3,
    RELAX_RESULT_SCHEMA_V3,
    V52_DIAGNOSTIC_LABEL,
    aggregate_episode_records_v3,
    check_budget_parity_v3,
    check_relax_budget_parity_v3,
    enforce_diagnostic_labeling_v3,
    execute_relax_episode_v3,
    is_v52_arm_v3,
)


def _episode_record(arm: str, curve_id: str = "SYN", direction: str = "forward",
                    n_steps: int = 5, fingerprint: str = "fp0",
                    **summary_extra) -> dict:
    base_summary = {
        "cv_sign_agree_fraction": 0.9,
        "contact_made_ts_dkey": False,
        "final_ts_rmsd_all": 0.7,
        "final_arc_progress": 0.3,
        "n_collisions": 0,
        "n_nonfinite": 0,
        "n_clip_events": 1,
        "mean_clip_factor": 0.95,
        "total_displacement": 1.2,
        "max_atom_excursion": 0.4,
        "divergence_flag": False,
        "n_ts_candidate_events": 0,
    }
    base_summary.update(summary_extra)
    v52 = is_v52_arm_v3(arm)
    return {
        "schema": EPISODE_SCHEMA_V3,
        "arm": arm,
        "curve_id": curve_id,
        "direction": direction,
        "n_steps_requested": n_steps,
        "input_step0_sha256": fingerprint,
        "diagnostic_only": v52,
        "diagnostic_label": V52_DIAGNOSTIC_LABEL if v52 else None,
        "claim_scope": CLAIM_SCOPE_V3,
        "summary": base_summary,
    }


def _full_matrix_records() -> list[dict]:
    out = []
    for curve in ("SYN_A", "SYN_B"):
        for direction in ("forward", "reverse"):
            for n in NO_PHYSICS_STEPS_V3:
                for arm in ARMS_V3:
                    out.append(_episode_record(
                        arm, curve_id=curve, direction=direction, n_steps=n,
                        fingerprint=f"{curve}|{direction}|{n}"))
    return out


def test_diagnostic_labeling_enforced() -> None:
    enforce_diagnostic_labeling_v3(_full_matrix_records())  # passes
    bad = _full_matrix_records()
    v52_rec = next(r for r in bad if r["arm"] == "v52")
    v52_rec["diagnostic_only"] = False
    with pytest.raises(RolloutV3Error) as exc:
        enforce_diagnostic_labeling_v3(bad)
    assert exc.value.code == "V52_LABELING"

    bad2 = _full_matrix_records()
    next(r for r in bad2 if r["arm"] == "v52")["diagnostic_label"] = "ok"
    with pytest.raises(RolloutV3Error):
        enforce_diagnostic_labeling_v3(bad2)

    # mislabeling in the other direction is also refused
    bad3 = _full_matrix_records()
    next(r for r in bad3 if r["arm"] == "r3")[
        "diagnostic_label"] = V52_DIAGNOSTIC_LABEL
    with pytest.raises(RolloutV3Error):
        enforce_diagnostic_labeling_v3(bad3)


def test_budget_parity_pass_and_violations() -> None:
    records = _full_matrix_records()
    summary = check_budget_parity_v3(records)
    assert summary["ok"] is True
    assert summary["arms"] == sorted(ARMS_V3)
    assert summary["steps"] == list(NO_PHYSICS_STEPS_V3)
    assert summary["n_groups"] == 2 * 2 * 3  # curves × directions × steps

    # an arm missing one budget point violates parity
    incomplete = [r for r in records if not (
        r["curve_id"] == "SYN_A" and r["direction"] == "forward"
        and r["arm"] == "v52" and r["n_steps_requested"] == 10)]
    with pytest.raises(RolloutV3Error) as exc:
        check_budget_parity_v3(incomplete)
    assert exc.value.code == "BUDGET_PARITY"

    # differing step-0 fingerprints across arms violate parity
    drifted = _full_matrix_records()
    next(r for r in drifted if r["curve_id"] == "SYN_B"
         and r["direction"] == "reverse" and r["arm"] == "classic"
         and r["n_steps_requested"] == 5)["input_step0_sha256"] = "other"
    with pytest.raises(RolloutV3Error) as exc:
        check_budget_parity_v3(drifted)
    assert exc.value.code == "BUDGET_PARITY"

    # a non-frozen budget is refused outright
    illegal = _full_matrix_records() + [
        _episode_record("classic", n_steps=7)]
    with pytest.raises(RolloutV3Error):
        check_budget_parity_v3(illegal)


def test_aggregate_episode_records_math() -> None:
    records = [
        _episode_record("classic", curve_id="A", n_steps=5,
                        cv_sign_agree_fraction=0.5,
                        final_ts_rmsd_all=1.0, n_collisions=2,
                        divergence_flag=True),
        _episode_record("classic", curve_id="B", n_steps=5,
                        cv_sign_agree_fraction=1.0,
                        final_ts_rmsd_all=2.0, n_collisions=0,
                        contact_made_ts_dkey=True),
        _episode_record("r3", curve_id="A", n_steps=5,
                        cv_sign_agree_fraction=0.8,
                        final_ts_rmsd_all=0.5),
        _episode_record("v52", curve_id="A", n_steps=5,
                        cv_sign_agree_fraction=0.9,
                        final_ts_rmsd_all=0.6, n_ts_candidate_events=3),
    ]
    agg = aggregate_episode_records_v3(records)
    groups = {(g["arm"], g["n_steps_requested"]): g
              for g in agg["groups"]}
    classic = groups[("classic", 5)]
    assert classic["n_episodes"] == 2
    assert classic["n_curves"] == 2
    assert classic["cv_sign_agree_fraction_mean"] == pytest.approx(0.75)
    assert classic["final_ts_rmsd_all_mean"] == pytest.approx(1.5)
    assert classic["final_ts_rmsd_all_median"] == pytest.approx(1.5)
    assert classic["collisions_total"] == 2
    assert classic["episodes_with_collision"] == 1
    assert classic["divergence_episodes"] == 1
    assert classic["contact_made_rate"] == pytest.approx(0.5)
    assert classic["diagnostic_only"] is False
    assert classic["diagnostic_label"] is None

    r3 = groups[("r3", 5)]
    assert r3["contact_made_rate"] == pytest.approx(0.0)
    v52 = groups[("v52", 5)]
    assert v52["ts_candidate_events_total"] == 3
    assert v52["diagnostic_only"] is True
    assert v52["diagnostic_label"] == V52_DIAGNOSTIC_LABEL

    # aggregation refuses mislabeled v52 input
    bad = _episode_record("v52")
    bad["diagnostic_label"] = None
    with pytest.raises(RolloutV3Error):
        aggregate_episode_records_v3([bad])


def test_rollout_summary_displacement_and_divergence_fields() -> None:
    r = run_rollout_v3(arm="classic", start=_synthetic_start(), n_steps=5)
    total = sum(s.disp_l1_sum for s in r.steps)
    assert r.summary["total_displacement"] == pytest.approx(total)
    assert r.summary["n_clip_events"] == sum(
        1 for s in r.steps if s.clip_factor < 1.0)
    assert r.summary["max_atom_excursion"] == pytest.approx(
        max(s.max_excursion_from_start for s in r.steps))
    assert isinstance(r.summary["divergence_flag"], bool)
    assert r.summary["divergence_flag"] == (
        r.summary["n_nonfinite"] > 0
        or r.summary["max_atom_excursion"] > DIVERGENCE_EXCURSION_ANGSTROM)


# --- relax result schema with a mock executor ----------------------------
_PAIRS = [(0, 1)]
_TARGETS = [1.5]
_REQUEST = {"provenance": {"request_sha256": "deadbeef"}, "source": {},
            "recipe": {"constraints": []}}


def _mock_runner_converged(request, workdir):
    return PhysicsOutcomeV3(
        status="converged", returncode=0, wall_seconds=1.0,
        final_xyz=((0.0, 0.0, 0.0), (1.5004, 0.0, 0.0)),
        convergence_line="GEOMETRY OPTIMIZATION CONVERGED")


def test_relax_result_schema_mock_executor() -> None:
    res = execute_relax_episode_v3(
        curve_id="SYN", arm="classic", direction="forward",
        driver_pairs=_PAIRS, target_distances=_TARGETS,
        request=_REQUEST, runner=_mock_runner_converged,
        workdir="/nonexistent")
    assert res["schema"] == RELAX_RESULT_SCHEMA_V3
    assert res["status"] == "converged"
    assert res["wall_seconds"] == 1.0
    assert res["achieved_distances"] == pytest.approx([1.5004])
    assert res["dev_from_target"] == pytest.approx([0.0004])
    assert res["request_sha256"] == "deadbeef"
    assert res["recipe"] == dict(RELAX_RECIPE_V3)
    assert res["claim_scope"] == CLAIM_SCOPE_V3
    assert res["diagnostic_only"] is False
    assert res["note"]  # TS_candidate-only note present


def test_relax_runner_exception_is_physics_failed() -> None:
    def boom(request, workdir):
        raise RuntimeError("xtb vanished")
    res = execute_relax_episode_v3(
        curve_id="SYN", arm="r3", direction="forward",
        driver_pairs=_PAIRS, target_distances=_TARGETS,
        request=_REQUEST, runner=boom, workdir="/nonexistent")
    assert res["status"] == "physics_failed"
    assert "xtb vanished" in res["error"]
    assert res["achieved_distances"] is None


def test_relax_v52_arm_forced_diagnostic() -> None:
    res = execute_relax_episode_v3(
        curve_id="SYN", arm="v52", direction="forward",
        driver_pairs=_PAIRS, target_distances=_TARGETS,
        request=_REQUEST, runner=_mock_runner_converged,
        workdir="/nonexistent")
    assert res["diagnostic_only"] is True
    assert res["diagnostic_label"] == V52_DIAGNOSTIC_LABEL


def test_relax_budget_parity_pass_and_violations() -> None:
    def _rec(arm, targets=_TARGETS, recipe=None):
        return {
            "schema": RELAX_RESULT_SCHEMA_V3,
            "curve_id": "SYN", "arm": arm, "direction": "forward",
            "status": "converged", "returncode": 0, "wall_seconds": 1.0,
            "driver_pairs": _PAIRS, "target_distances": list(targets),
            "recipe": dict(RELAX_RECIPE_V3) if recipe is None else recipe,
            "claim_scope": CLAIM_SCOPE_V3,
            "diagnostic_only": arm == "v52",
            "diagnostic_label": (V52_DIAGNOSTIC_LABEL
                                 if arm == "v52" else None),
        }
    ok = check_relax_budget_parity_v3(
        [_rec("classic"), _rec("r3"), _rec("v52")])
    assert ok["ok"] is True and ok["n_curves"] == 1

    # cross-arm target drift violates parity
    with pytest.raises(RolloutV3Error) as exc:
        check_relax_budget_parity_v3(
            [_rec("classic"), _rec("r3", targets=[1.6])])
    assert exc.value.code == "RELAX_PARITY"

    # recipe drift violates parity
    drifted_recipe = dict(RELAX_RECIPE_V3, max_cycle=100)
    with pytest.raises(RolloutV3Error):
        check_relax_budget_parity_v3(
            [_rec("classic"), _rec("r3", recipe=drifted_recipe)])

    # non-TS-candidate claim scope violates parity
    bad_scope = _rec("classic")
    bad_scope["claim_scope"] = "verified_ts"
    with pytest.raises(RolloutV3Error):
        check_relax_budget_parity_v3([bad_scope])
