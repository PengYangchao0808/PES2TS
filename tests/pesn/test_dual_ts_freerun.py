"""WP4-B free-run harness tests (synthetic; NO ACP — G4 §6).

Covers: rollout step math (X' = X + a_det exactly), consume-time feature
recomputation consistency (synthetic + real monitor shard equality), the
no-reference-correction invariant (policy sees only current-state inputs;
bit-identical reruns), guard rails (catastrophic overlap / non-finite /
scale-bound / observation-failure aborts with verbatim offending
geometry), oracle isolation (remaining-arc recorded post-hoc only, never
in features), QC budget cap enforcement with hard abort, receipt reuse
via the content-addressed store (incl. the monitor-store fallback
pattern), the receipt-validator drift guard against the WP2 driver,
model-loading identity gates (bundle/spec/action binding), and record
emission + digest binding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pes2ts_core.generation.planning.synchronized_path import digest  # noqa: E402
from pes2ts_core.integration.acp.gradient_backend import write_json  # noqa: E402
from pes2ts_core.integration.acp.orca_gradient_request import (  # noqa: E402
    build_gradient_request,
    request_sha256,
)
from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    BOUNDARY_TEACHER_ENDPOINT,
    BranchRecord,
    DualTSTeacher,
    EpisodeRecord,
    ObservationRef,
    TeacherLabel,
    TeacherState,
    content_digest_of,
    validate_record,
)
from pes2ts_core.pesn.features.dual_ts_spec import (  # noqa: E402
    build_features_spec,
    save_spec_payload,
)
from pes2ts_core.pesn.models.dual_egnn import (  # noqa: E402
    REQUIRED_STATE_KEYS,
)
from pes2ts_core.pesn.models.dual_ts_policy import (  # noqa: E402
    DEFAULT_A_MAX_ANGSTROM,
    DualTSModelConfig,
    DualTSPolicy,
)
from pes2ts_core.pesn.offline.data.direct_preprocess import (  # noqa: E402
    kabsch_register,
    masses_for,
)
from pes2ts_core.pesn.offline.evaluation.dual_ts_freerun import (  # noqa: E402
    A_MAX_REL_TOL,
    OVERLAP_ABORT_MIN_DISTANCE_ANGSTROM,
    FreerunError,
    ObservationCall,
    FreerunObserver,
    QCBudgetExhausted,
    RolloutSettings,
    RolloutStart,
    bond_validity,
    build_rollout_state_inputs,
    compute_record_digest,
    emit_rollout_artifacts,
    load_frozen_scales,
    load_rollout_start,
    load_selected_policy,
    observation_settings_digest,
    pairwise_min_distance,
    post_hoc_truth_metrics,
    run_rollout,
    seal_record,
    summarize_rollouts,
    validate_receipt_pair,
    verify_record_digest,
)
from pes2ts_core.pesn.offline.training.dual_ts_bc import (  # noqa: E402
    DualTSBCSettings,
    DualTSBCTrainer,
)
from scripts.compute_dual_ts_observations import (  # noqa: E402
    validate_receipt as driver_validate_receipt,
)

XTB_BLOCKS = ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend']

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

Z3 = np.asarray([8, 1, 1], dtype=np.int64)          # O + H + H toy
STEP = 0.14
SCALES = {"s_g": 0.0154, "s_delta_e": 0.0789, "spec_digest": "0" * 64}


def coords_at(t: int) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0],
                     [0.96 + t * STEP, 0.0, 0.0],
                     [1.75, 0.0, 0.0]])


def make_start(coords: np.ndarray | None = None,
               group_id: str = "RXN_T0") -> RolloutStart:
    x = coords_at(0) if coords is None else np.asarray(coords, dtype=float)
    edge_pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    directed = np.concatenate([edge_pairs, edge_pairs[:, ::-1]], axis=0)
    edge_index = np.ascontiguousarray(directed.T)
    return RolloutStart(
        group_id=group_id, reaction_id=group_id,
        branch_id=f"{group_id}#brA", episode_id=f"{group_id}#epA0",
        start_state_id=f"{group_id}|A|0000",
        atomic_numbers=Z3.copy(),
        coordinates=np.asarray(x, dtype=np.float64),
        atom_map_ids=tuple(f"m{i + 1:04d}" for i in range(Z3.size)),
        charge=0, multiplicity=1, edge_pairs=edge_pairs,
        edge_index=edge_index,
        atom_feats_a=np.asarray([[1.0, 2.0, 0.0, 0.0],
                                 [1.0, 1.0, 0.0, 0.0],
                                 [1.0, 1.0, 0.0, 0.0]]),
        atom_feats_b=np.asarray([[1.0, 1.0, 1.0, 0.0],
                                 [1.0, 1.0, 0.0, 0.0],
                                 [1.0, 1.0, 1.0, 0.0]]),
        edge_feats_a=np.asarray([[1.0, 1.0, 0.0, 0.0]] * 4),
        edge_feats_b=np.asarray([[0.0, 0.0, 1.0, 0.0]] * 2 +
                                [[1.0, 1.0, 0.0, 0.0]] * 2),
        bond_source_a="authority", bond_source_b="authority")


def make_policy(seed: int = 0) -> DualTSPolicy:
    torch.manual_seed(seed)
    return DualTSPolicy(
        config=DualTSModelConfig(light=True),
        action_max=DEFAULT_A_MAX_ANGSTROM).to(torch.float32)


class ScriptedObserver:
    """Offline stub with the observer interface (no ACP, no receipts)."""

    def __init__(self, energy_fn=None, gradient_fn=None, fail_after=None):
        self.calls: list[np.ndarray] = []
        self.energy_fn = energy_fn or (lambda x, t: -1.0 - 0.01 * t)
        self.gradient_fn = gradient_fn or (
            lambda x, t: np.full((x.shape[0], 3), 0.01))
        self.fail_after = fail_after
        self.billed_count = 0

    def observe(self, z, coords, maps, *, charge, multiplicity,
                node_label):
        t = len(self.calls)
        self.calls.append(np.asarray(coords, dtype=float).copy())
        if self.fail_after is not None and t >= self.fail_after:
            return ObservationCall(
                digest=f"{t:064x}", status="failed", billed=False,
                energy=None, gradient=None, backend_seconds=None,
                receipt_dir=None, provenance="stub",
                error="stubbed_backend_failure")
        self.billed_count += 1
        return ObservationCall(
            digest=f"{t:064x}", status="computed", billed=True,
            energy=float(self.energy_fn(coords, t)),
            gradient=np.asarray(self.gradient_fn(coords, t),
                                dtype=np.float64),
            backend_seconds=0.001, receipt_dir=f"/stub/{t}",
            provenance="stub", error=None)


class _ActionOverride:
    """Wraps a real policy; overrides the deterministic action carrier."""

    def __init__(self, base: DualTSPolicy, action_fn):
        self.base = base
        self.action_fn = action_fn

    def __call__(self, batch):
        return self.base(batch)

    def deterministic(self, mu):
        return torch.as_tensor(
            np.asarray(self.action_fn(), dtype=np.float64),
            dtype=torch.float32)

    def sample(self, mu, log_sigma, generator=None):
        return self.base.sample(mu, log_sigma, generator=generator)

    def train_mode(self, flag):
        self.base.train_mode(flag)

    def parameters(self):
        return self.base.parameters()


IDENTITY = {"action_max": DEFAULT_A_MAX_ANGSTROM}


def rollout_settings(max_steps=4, **over) -> RolloutSettings:
    return RolloutSettings(max_steps=max_steps,
                           qc_budget=over.pop("qc_budget", 10_000), **over)


# ---------------------------------------------------------------------------
# Rollout step math: X' = X + a_det
# ---------------------------------------------------------------------------

def test_rollout_step_math_x_prime_equals_x_plus_adet():
    policy = make_policy(seed=3)
    observer = ScriptedObserver()
    captured: list[dict] = []

    def probe(t, inputs, features, obs, delta_e, a, x_next):
        captured.append({
            "x": np.asarray(inputs["x_angstrom"],
                            dtype=np.float64).copy(),
            "a": np.asarray(a, dtype=np.float64).copy(),
            "x_next": np.asarray(x_next, dtype=np.float64).copy()})

    record = run_rollout(torch=torch, model=policy, observer=observer,
                         start=make_start(), scales=SCALES, identity=IDENTITY,
                         settings=rollout_settings(max_steps=2),
                         input_probe=probe)
    assert record["n_steps"] == 2
    assert record["termination"]["status"] == "completed_step_cap"
    assert len(observer.calls) == 2
    # the carried state is EXACTLY X' = X + a — nothing else touches it
    for k in range(len(captured) - 1):
        np.testing.assert_array_equal(
            captured[k + 1]["x"], captured[k]["x"] + captured[k]["a"])
        np.testing.assert_array_equal(
            captured[k]["x_next"], captured[k]["x"] + captured[k]["a"])
    # every accepted action respects the a_max bound with zero violations
    for row in record["steps"]:
        a = np.asarray(row["action_round10"])
        assert np.abs(a).max() <= DEFAULT_A_MAX_ANGSTROM * (1 + A_MAX_REL_TOL)
        assert row["geometry_next"]["a_max_violations"] == 0
    # observations happen at the VISITED states (start frame included)
    np.testing.assert_array_equal(observer.calls[0], coords_at(0))


# ---------------------------------------------------------------------------
# Consume-time recompute consistency
# ---------------------------------------------------------------------------

def test_consume_time_recompute_consistency_synthetic():
    start = make_start()
    g_raw = np.asarray([[0.02, -0.01, 0.005],
                        [0.01, 0.03, -0.02],
                        [-0.005, 0.0, 0.015]])
    delta_e = 0.037
    inputs, features = build_rollout_state_inputs(
        start=start, state_id="s", coordinates=coords_at(0),
        gradient_hartree_per_angstrom=g_raw, delta_e_hartree=delta_e,
        scales=SCALES)
    # exact frozen observation layout: [z_id, |g|/s_g, dE/s_dE, charge, mult]
    ns = inputs["node_scalars"]
    assert ns.shape[1] == 5
    np.testing.assert_array_equal(ns[:, 0], Z3)
    np.testing.assert_allclose(
        ns[:, 1], np.linalg.norm(g_raw, axis=1) / SCALES["s_g"], atol=1e-12)
    np.testing.assert_allclose(
        ns[:, 2], delta_e / SCALES["s_delta_e"], atol=1e-12)
    assert set(np.unique(ns[:, 3])) == {0.0}
    assert set(np.unique(ns[:, 4])) == {1.0}
    np.testing.assert_allclose(inputs["g_scaled"], g_raw / SCALES["s_g"],
                               atol=1e-12)
    # the collation layer applies the SAME frozen centroid rule
    x = inputs["x_angstrom"]
    np.testing.assert_allclose(
        x - x.mean(axis=0, keepdims=True), features.x_centered, atol=1e-12)
    assert set(inputs.keys()) == set(REQUIRED_STATE_KEYS)


def test_consume_time_recompute_matches_offline_shard_bytes():
    """REAL monitor data (no QC): recomputing the start state's inputs from
    RAW gradients/energy reproduces the offline shard row (up to the round-10
    gradient precision convention) — the frozen observation-only pipeline is
    reused, not re-implemented (v4.1 WP1: online == offline observation
    builder, 5 channels)."""
    features_dir = ROOT / (
        "outputs/pesn/dual_ts_v2_20261015/wp4_monitor/features_v41")
    observed_dir = ROOT / (
        "outputs/pesn/dual_ts_v2_20261015/wp4_monitor/teachers_observed")
    if not features_dir.is_dir():
        pytest.skip("monitor v41 wave artifacts not present")
    scales = load_frozen_scales(features_dir)
    start = load_rollout_start(features_dir=features_dir,
                               observed_dir=observed_dir,
                               group_id="RXN_0000000798")
    row = np.load(features_dir / "features_RXN_0000000798.npz")
    sids = [str(s) for s in row["state_id"]]
    r = sids.index(start.start_state_id)
    shard_ns = row["node_scalars"][r]
    shard_x = row["x_angstrom"][r]
    shard_g = row["g_scaled"][r]
    shard_dE = float(row["delta_e_hartree"][r])
    g_raw = np.asarray(shard_g, dtype=np.float64) * scales["s_g"]
    inputs, _ = build_rollout_state_inputs(
        start=start, state_id=start.start_state_id,
        coordinates=start.coordinates, gradient_hartree_per_angstrom=g_raw,
        delta_e_hartree=shard_dE, scales=scales)
    ns = inputs["node_scalars"]
    assert ns.shape[1] == 5
    np.testing.assert_array_equal(ns[:, 0], shard_ns[:, 0])
    np.testing.assert_allclose(ns[:, 1], shard_ns[:, 1], atol=1e-9)
    np.testing.assert_allclose(ns[:, 2], shard_ns[:, 2], atol=1e-9)
    np.testing.assert_array_equal(ns[:, 3:5], shard_ns[:, 3:5])
    x = inputs["x_angstrom"]
    np.testing.assert_allclose(x - x.mean(axis=0, keepdims=True),
                               shard_x, atol=1e-12)


# ---------------------------------------------------------------------------
# No-reference-correction invariant
# ---------------------------------------------------------------------------

def test_no_reference_correction_invariant():
    policy = make_policy(seed=5)

    def probe(t, inputs, features, obs, delta_e, a, x_next):
        assert set(inputs.keys()) == set(REQUIRED_STATE_KEYS)
        # observation-only invariant: 5 channels, charge=0, multiplicity=1,
        # and NO teacher/arriving-action channels anywhere.
        ns = np.asarray(inputs["node_scalars"])
        assert ns.shape[1] == 5
        assert set(np.unique(ns[:, 3])) == {0.0}
        assert set(np.unique(ns[:, 4])) == {1.0}

    def run_once() -> dict:
        rec = run_rollout(
            torch=torch, model=policy,
            observer=ScriptedObserver(
                gradient_fn=lambda x, t: 0.001 * x),
            start=make_start(), scales=SCALES, identity=IDENTITY,
            settings=rollout_settings(max_steps=3), input_probe=probe)
        return seal_record(rec)

    rec1 = run_once()
    rec2 = run_once()
    # deterministic given the checkpoint + observations: identical reruns
    # (wall-clock timing is excluded from the comparison; the record digest
    # covers the content once sealed)
    for rec in (rec1, rec2):
        rec["cost"].pop("wall_seconds")
    assert compute_record_digest(rec1) == compute_record_digest(rec2)
    # no truth/reference fields exist anywhere in the pre-posthoc record
    text = json.dumps(rec1)
    assert "remaining_arc" not in text
    assert "post_hoc" not in text
    assert "action_delta" not in text


# ---------------------------------------------------------------------------
# Guard rails (plan §I.3 independent action checks)
# ---------------------------------------------------------------------------

def test_guard_rail_catastrophic_overlap_aborts():
    start = make_start(coords=np.array([[0.0, 0.0, 0.0],
                                        [0.5, 0.0, 0.0],
                                        [2.0, 0.0, 0.0]]))

    def collapse():
        a = np.zeros((3, 3))
        a[0] = [0.1, 0.0, 0.0]
        a[1] = [-0.1, 0.0, 0.0]
        return a

    model = _ActionOverride(make_policy(seed=1), collapse)
    observer = ScriptedObserver()
    record = run_rollout(torch=torch, model=model, observer=observer,
                         start=start, scales=SCALES, identity=IDENTITY,
                         settings=rollout_settings(max_steps=4))
    assert record["termination"]["status"] == "aborted"
    assert record["termination"]["reason"] == "catastrophic_overlap"
    assert record["termination"]["step"] == 0
    offending = np.asarray(
        record["termination"]["offending_geometry_round10"])
    assert pairwise_min_distance(offending) == pytest.approx(0.3, abs=1e-6)
    assert pairwise_min_distance(offending) < \
        OVERLAP_ABORT_MIN_DISTANCE_ANGSTROM
    # no step row was accepted; the offending geometry is saved verbatim
    assert record["n_steps"] == 0 and record["steps"] == []
    # only the start state was observed; the invalid X' is never observed
    assert len(observer.calls) == 1 and record["cost"]["n_billed"] == 1


def test_guard_rail_nonfinite_and_scale_bound_aborts():
    def nan_action():
        a = np.zeros((3, 3))
        a[1, 1] = np.nan
        return a

    record = run_rollout(
        torch=torch, model=_ActionOverride(make_policy(seed=2), nan_action),
        observer=ScriptedObserver(), start=make_start(), scales=SCALES,
        identity=IDENTITY, settings=rollout_settings(max_steps=3))
    assert record["termination"]["reason"] == "action_nonfinite"
    assert np.isnan(np.asarray(
        record["termination"]["offending_action_round10"])).any()

    def huge_action():
        a = np.zeros((3, 3))
        a[0] = [2.0 * DEFAULT_A_MAX_ANGSTROM, 0.0, 0.0]
        return a

    record = run_rollout(
        torch=torch,
        model=_ActionOverride(make_policy(seed=2), huge_action),
        observer=ScriptedObserver(), start=make_start(), scales=SCALES,
        identity=IDENTITY, settings=rollout_settings(max_steps=3))
    assert record["termination"]["reason"] == "action_scale_bound"

    def inf_gradient(x, t):
        g = np.full((x.shape[0], 3), 0.01)
        g[0, 0] = np.inf
        return g

    with pytest.raises(Exception):
        # non-finite observation content must fail loudly in the feature
        # layer (NONFINITE_INPUT), never enter the policy
        run_rollout(torch=torch, model=make_policy(seed=2),
                    observer=ScriptedObserver(gradient_fn=inf_gradient),
                    start=make_start(), scales=SCALES, identity=IDENTITY,
                    settings=rollout_settings(max_steps=2))


def test_guard_rail_observation_failure_aborts():
    record = run_rollout(torch=torch, model=make_policy(seed=4),
                         observer=ScriptedObserver(fail_after=2),
                         start=make_start(), scales=SCALES, identity=IDENTITY,
                         settings=rollout_settings(max_steps=5))
    assert record["termination"]["status"] == "aborted"
    assert record["termination"]["reason"] == "observation_failed"
    assert record["n_steps"] == 2  # steps 0 and 1 observed + accepted


# ---------------------------------------------------------------------------
# Oracle isolation: remaining-arc is post-hoc reporting ONLY
# ---------------------------------------------------------------------------

def build_teacher_payload(reaction_id: str = "RXN_T0",
                          n_steps: int = 4) -> dict:
    def geom(t: int, lab: str) -> np.ndarray:
        x = coords_at(t)
        if lab == "B":
            x = x.copy()
            x[:, 0] *= -1.0
        return x

    branches, episodes, states, labels = [], [], [], []
    for lab in ("A", "B"):
        bid, eid = f"{reaction_id}#br{lab}", f"{reaction_id}#ep{lab}0"
        sids = [f"{reaction_id}|{lab}|{t:04d}" for t in range(n_steps + 1)]
        branches.append(BranchRecord(
            branch_id=bid, endpoint_label=lab, reaction_id=reaction_id,
            episode_ids=(eid,)))
        episodes.append(EpisodeRecord(
            episode_id=eid, branch_id=bid, reaction_id=reaction_id,
            start_state_id=sids[0], budget_version="budget_v0",
            environment_version="ENV_v0", reward_version="REW_v0"))
        for t in range(n_steps + 1):
            x = geom(t, lab)
            terminal = t == n_steps
            states.append(TeacherState(
                state_id=sids[t], episode_id=eid, branch_id=bid,
                step_index=t, atomic_numbers=Z3.copy(), coordinates=x,
                atom_map_ids=tuple(f"m{i + 1:04d}" for i in range(Z3.size)),
                method_id="teacher_v0",
                observation=(None if terminal else ObservationRef(
                    status="ok", energy_hartree=-1.0 - 0.01 * t,
                    gradient_hartree_per_angstrom=np.full((3, 3), 0.01),
                    method_id="teacher_v0", software="toy",
                    settings_digest="a" * 64)),
                observation_missing_reason=(
                    "teacher_frame_isolated" if terminal else None),
                boundary_type=(BOUNDARY_TEACHER_ENDPOINT if terminal
                               else None),
                next_state_id=(None if terminal else sids[t + 1])))
            labels.append(TeacherLabel(
                state_id=sids[t],
                ts_displacement=geom(n_steps, lab) - x,
                remaining_arc_length=(n_steps - t) * STEP,
                teacher_endpoint=terminal,
                action_delta=(None if terminal
                              else geom(t + 1, lab) - x)))
    teacher = DualTSTeacher(
        reaction_id=reaction_id, reaction_group_id=reaction_id,
        branches=tuple(branches), episodes=tuple(episodes),
        states=tuple(states), labels=tuple(labels))
    teacher.validate()
    return teacher.to_payload()


def test_oracle_isolation_post_hoc_only(tmp_path):
    observed_dir = tmp_path / "observed"
    observed_dir.mkdir()
    payload = build_teacher_payload("RXN_T0", n_steps=4)
    (observed_dir / "RXN_T0.teacher.json").write_text(json.dumps(payload))
    record = run_rollout(torch=torch, model=make_policy(seed=7),
                         observer=ScriptedObserver(), start=make_start(),
                         scales=SCALES, identity=IDENTITY,
                         settings=rollout_settings(max_steps=3))
    # pre-posthoc: no truth metric anywhere in the record
    assert "post_hoc_truth_metrics" not in record
    assert all("remaining_arc" not in json.dumps(s)
               for s in record["steps"])
    record = post_hoc_truth_metrics(record, observed_dir=observed_dir)
    post = record["post_hoc_truth_metrics"]
    assert "never fed to the policy" in post["declaration"]
    assert post["n_overlap_steps"] == 3  # min(3 rollout steps, 5 teacher)
    assert post["per_step_deviation_rms_A"][0] == pytest.approx(
        0.0, abs=1e-12)
    assert post["deviation_final_A"] > 0.0
    assert post["deviation_growth_A"] == pytest.approx(
        post["deviation_final_A"], abs=1e-12)
    # teacher label arc at the same indices
    assert post["teacher_label_remaining_arc_A"] == pytest.approx(
        [4 * STEP, 3 * STEP, 2 * STEP])
    assert post["oracle_arc_start_A"] > 0.0
    assert post["oracle_arc_end_A"] > 0.0
    # oracle arc = straight-line mass-weighted distance to the TS frame
    masses = masses_for(Z3)
    x2 = np.asarray(record["steps"][2]["x_round10"], dtype=float)
    ts = np.asarray(payload["states"][4]["coordinates"], dtype=float)
    reg = kabsch_register(x2, ts, masses)
    expect = float(np.sqrt(
        (masses[:, None] * (reg - ts) ** 2).sum() / masses.sum()))
    assert post["oracle_arc_to_ts_A"][2] == pytest.approx(expect, rel=1e-9)


# ---------------------------------------------------------------------------
# QC budget cap + receipt machinery (real FreerunObserver, injected backend)
# ---------------------------------------------------------------------------

class FakeGradientBackend:
    """Writes driver-contract receipts through the REAL request builder."""

    def __init__(self, *, charge, multiplicity, elements, folder, method,
                 timeout_seconds, nproc, acp_wiring=None, **_over):
        self.charge, self.multiplicity = charge, multiplicity
        self.elements = list(elements)
        self.folder = Path(folder)
        self.method = method
        self.timeout_seconds, self.nproc = timeout_seconds, nproc

    def __call__(self, geometry, evaluation_id):
        request = build_gradient_request(
            geometry=np.asarray(geometry, dtype=float).tolist(),
            elements=self.elements, method=self.method, basis="",
            charge=self.charge, multiplicity=self.multiplicity,
            route_extras=["EnGrad"], timeout_seconds=self.timeout_seconds,
            nproc=self.nproc, extra_blocks=XTB_BLOCKS,
            scf_convergence="tight", output_name="grad")
        binding = digest(request)
        folder = self.folder / evaluation_id
        write_json(folder / "request.json",
                   {**request, "request_sha256": binding,
                    "acp_request_sha256": request_sha256(request)})
        coords = np.asarray(geometry, dtype=float)
        write_json(folder / "result.json", {
            "success": True, "request_sha256": binding,
            "energy": float(-1.0 - 0.001 * np.abs(coords).sum()),
            "gradient_hartree_per_angstrom":
                (0.001 * coords).round(10).tolist(),
            "duration_seconds": 0.002,
        })
        return json.loads((folder / "result.json").read_text())


def make_acp(tmp_path) -> dict:
    return {"root": str(tmp_path / "acp_root"),
            "python": "/usr/bin/false",
            "config_path": str(tmp_path / "acp.yaml"),
            "method": "GFN2-xTB", "timeout_seconds": 120.0, "nproc": 1}


def seed_receipt(folder: Path, z, coords, maps) -> str:
    identity = computation_identity_for(z, coords, maps)
    FakeGradientBackend(charge=0, multiplicity=1, elements=["O", "H", "H"],
                        folder=folder, method="GFN2-xTB",
                        timeout_seconds=120, nproc=1)(
        coords, f"{identity[:2]}/{identity}")
    return identity


def computation_identity_for(z, coords, maps) -> str:
    from pes2ts_core.pesn.offline.data.dual_ts_fit50 import (
        atom_order_hash,
        computation_identity,
        geometry_content_digest,
    )
    identity = computation_identity(
        geometry_content_digest(np.asarray(coords, dtype=float)),
        atom_order_hash(np.asarray(z, dtype=int), list(maps)),
        "GFN2-xTB", 0, 1, "eg_single_point")
    return str(identity["digest"])


def test_observer_budget_cap_and_reuse(tmp_path):
    acp = make_acp(tmp_path)
    out_root = tmp_path / "obs"
    fallback = tmp_path / "monitor_store"
    observer = FreerunObserver(out_root=out_root, acp=acp, qc_budget=2,
                               fallback_roots=[fallback],
                               backend_factory=FakeGradientBackend)
    start = make_start()
    z, maps = start.atomic_numbers, start.atom_map_ids
    # start state already receipt-backed in the FALLBACK store (the monitor
    # campaign precedent): content-addressed reuse, zero billing
    seed_receipt(fallback, z, start.coordinates, maps)
    call = observer.observe(z, start.coordinates, maps, charge=0,
                            multiplicity=1, node_label="t0")
    assert call.status == "reused_fallback" and call.billed is False
    assert observer.billed_count == 0
    assert call.energy == pytest.approx(
        -1.0 - 0.001 * float(np.abs(start.coordinates).sum()))
    # a NEW geometry is computed and billed against the budget
    moved = start.coordinates + np.asarray([[0.01, 0.0, 0.0],
                                            [0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0]])
    call2 = observer.observe(z, moved, maps, charge=0, multiplicity=1,
                             node_label="t1")
    assert call2.status == "computed" and call2.billed is True
    assert observer.billed_count == 1
    # same geometry again → resume from the OWN store, still unbilled
    call3 = observer.observe(z, moved, maps, charge=0, multiplicity=1,
                             node_label="t1-again")
    assert call3.status == "resume_valid" and observer.billed_count == 1
    # budget=2: the next NEW geometry bills (2), then the one after is a
    # hard QCBudgetExhausted BEFORE any launch
    moved2 = moved + 0.01
    observer.observe(z, moved2, maps, charge=0, multiplicity=1,
                     node_label="t2")
    assert observer.billed_count == 2
    with pytest.raises(QCBudgetExhausted):
        observer.observe(z, moved2 + 0.01, maps, charge=0, multiplicity=1,
                         node_label="t3")
    # a tampered receipt (invalid gradient) is never zero-filled: a fresh
    # observer with budget headroom recomputes it
    bad = moved2 + 0.02
    d_bad = seed_receipt(out_root, z, bad, maps)
    rd = out_root / d_bad[:2] / d_bad
    result = json.loads((rd / "result.json").read_text())
    result["gradient_hartree_per_angstrom"] = [[0.0, 0.0, "x"]] * 3
    write_json(rd / "result.json", result)
    observer2 = FreerunObserver(out_root=out_root, acp=acp, qc_budget=99,
                                backend_factory=FakeGradientBackend)
    call4 = observer2.observe(z, bad, maps, charge=0, multiplicity=1,
                              node_label="t4")
    assert call4.status == "computed" and call4.billed is True
    observer2.write_ledger(run_id="ledger-test", wall_seconds=1.0)
    ledger = json.loads((out_root / "compute_ledger.json").read_text())
    entry = ledger["runs"][-1]
    assert entry["qc_budget"] == 99
    assert entry["settings_digest"] == observation_settings_digest(acp)
    assert entry["billed"] == 1 and entry["reused"] == 0


def test_rollout_hard_aborts_at_budget_cap(tmp_path):
    acp = make_acp(tmp_path)
    observer = FreerunObserver(out_root=tmp_path / "obs", acp=acp,
                               qc_budget=3,
                               backend_factory=FakeGradientBackend)
    # tiny fixed actions: no geometry guard fires, so the ONLY termination
    # is the hard QC budget gate
    def tiny_action():
        a = np.zeros((3, 3))
        a[0, 2] = 0.01
        return a

    model = _ActionOverride(make_policy(seed=9), tiny_action)
    record = run_rollout(torch=torch, model=model,
                         observer=observer, start=make_start(),
                         scales=SCALES, identity=IDENTITY,
                         settings=rollout_settings(max_steps=5,
                                                   qc_budget=3))
    assert record["termination"]["status"] == "aborted"
    assert record["termination"]["reason"] == "qc_budget_exhausted"
    assert record["termination"]["step"] == 3
    assert record["n_steps"] == 3 and observer.billed_count == 3
    assert record["cost"]["n_billed"] == 3
    # a SECOND rollout cannot bill a single point beyond the cap: it starts
    # from a DIFFERENT geometry (no content-addressed resume possible)
    shifted = make_start(coords=coords_at(0) + 0.02)
    record2 = run_rollout(torch=torch, model=model,
                          observer=observer, start=shifted,
                          scales=SCALES, identity=IDENTITY,
                          settings=rollout_settings(max_steps=2,
                                                    qc_budget=3))
    assert record2["termination"]["reason"] == "qc_budget_exhausted"
    assert record2["n_steps"] == 0 and observer.billed_count == 3


# ---------------------------------------------------------------------------
# Receipt validator drift guard vs the WP2 driver
# ---------------------------------------------------------------------------

def _receipt_pair(n_atoms=3, *, tamper=None):
    coords = np.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0],
                         [1.75, 0.0, 0.0]])
    request = build_gradient_request(
        geometry=coords.tolist(), elements=["O", "H", "H"],
        method="GFN2-xTB", basis="", charge=0, multiplicity=1,
        route_extras=["EnGrad"], timeout_seconds=120, nproc=1,
        extra_blocks=XTB_BLOCKS, scf_convergence="tight", output_name="grad")
    binding = digest(request)
    request = {**request, "request_sha256": binding,
               "acp_request_sha256": request_sha256(request)}
    result = {"success": True, "request_sha256": binding,
              "energy": -1.23,
              "gradient_hartree_per_angstrom":
                  (0.001 * coords).round(10).tolist()}
    if tamper == "energy":
        result["energy"] = "not-a-number"
    elif tamper == "binding":
        result["request_sha256"] = "f" * 64
    elif tamper == "shape":
        result["gradient_hartree_per_angstrom"] = [[0.0, 0.0, 0.0]] * 2
    elif tamper == "success":
        result["success"] = False
    return request, result


@pytest.mark.parametrize("tamper", [None, "energy", "binding", "shape",
                                    "success"])
def test_validate_receipt_pair_drift_guard(tamper):
    request, result = _receipt_pair(tamper=tamper)
    ours = validate_receipt_pair(request, result, 3)
    theirs = driver_validate_receipt(request, result, 3)
    assert ours == theirs
    assert (ours[0] is True) == (tamper is None)


# ---------------------------------------------------------------------------
# Model-loading identity gates
# ---------------------------------------------------------------------------

def _synthetic_states(seed=7):
    rng = np.random.default_rng(seed)
    states = []
    for g in ("RXN_GA", "RXN_GB"):
        for k in range(3):
            n = 3
            direction = rng.normal(size=(n, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            action = direction * rng.uniform(0.012, 0.04, size=(n, 1))
            node_scalars = np.concatenate([
                Z3[:, None].astype(float), rng.normal(size=(n, 1)),
                np.full((n, 1), 0.4), np.zeros((n, 2)),
            ], axis=1)  # 5 channels: z_id, |g|/s_g, dE/s_dE, charge, mult
            near_zero = np.linalg.norm(action, axis=1) < 0.01
            edge_index = np.asarray([[0, 1, 0, 2], [1, 0, 2, 0]],
                                    dtype=np.int64)
            states.append({
                "state_id": f"{g}_{k}", "reaction_group_id": g,
                "branch_id": "brA", "episode_id": "epA", "step_index": k,
                "is_teacher": False,
                "remaining_arc_length": 1.0 - 0.2 * k,
                "action": action,
                "near_zero_mask": near_zero,
                "atomic_numbers": Z3.copy(),
                "node_scalars": node_scalars,
                "x_angstrom": rng.normal(scale=1.0, size=(n, 3)),
                "g_scaled": rng.normal(size=(n, 3)),
                "atom_feats_a": rng.normal(size=(n, 4)),
                "atom_feats_b": rng.normal(size=(n, 4)),
                "edge_feats_a": rng.normal(size=(4, 4)),
                "edge_feats_b": rng.normal(size=(4, 4)),
                "edge_index": edge_index,
            })
    return states


@pytest.fixture(scope="module")
def trained_model_artifacts(tmp_path_factory):
    """Tiny real BC run → checkpoint + bundle + spec-matched features dir."""
    tmp = tmp_path_factory.mktemp("identity")
    spec, payload = build_features_spec(
        s_g=0.0154, s_delta_e=0.0789, element_vocab=[1, 6, 7, 8],
        population={"wave": "synthetic_freerun_tests"},
        n_atoms_range=(3, 8))
    features_dir = tmp / "features"
    features_dir.mkdir()
    save_spec_payload(payload, features_dir / "features_spec.json")
    spec_digest = spec.content_digest()
    write_json(features_dir / "features_manifest.json", {
        "spec_digest": spec_digest,
        "per_reaction": [{"reaction_group_id": "RXN_GA",
                          "reaction_id": "RXN_GA",
                          "npz_file": "features_RXN_GA.npz",
                          "npz_sha256": "0" * 64}],
    })
    settings = DualTSBCSettings(
        recipe="bc_nll", budget_updates=2, batch_states=8, seed=5,
        device="cpu", require_cuda=False, curve_every=1, eval_every=0)
    trainer = DualTSBCTrainer(
        settings=settings, states=_synthetic_states(),
        data_meta={"spec_digest": spec_digest,
                   "shard_digests": ["b" * 64],
                   "norm_params": {"s_g_hartree_per_angstrom": 0.0154,
                                   "s_delta_e": 0.0789}},
        run_dir=tmp / "run")
    record = trainer.train()
    bundle = trainer.build_bundle(record)
    ckpt = tmp / "run" / "checkpoint_last.pt"
    bundle_path = tmp / "run" / "bundle.json"
    bundle_path.write_text(
        json.dumps(bundle.to_payload(), indent=2) + "\n")
    return {"tmp": tmp, "features_dir": features_dir,
            "spec_digest": spec_digest, "ckpt": ckpt,
            "bundle": bundle_path, "bundle_payload": bundle.to_payload()}


def test_identity_gates_happy_path_and_refusals(trained_model_artifacts,
                                                 tmp_path):
    art = trained_model_artifacts
    policy, identity = load_selected_policy(
        torch, checkpoint_path=art["ckpt"], bundle_path=art["bundle"],
        features_dir=art["features_dir"], selected_recipe="bc_nll")
    assert identity["bundle_bc_parent_digest"] == \
        identity["checkpoint_sha256"]
    assert identity["spec_digest"] == art["spec_digest"]
    assert identity["action_version"].startswith(
        "radial_tanh_v0:bc_target=teacher_latent_v1:a_max=0.1")
    # (iv) non-selected recipe refused
    with pytest.raises(FreerunError) as ei:
        load_selected_policy(torch, checkpoint_path=art["ckpt"],
                             bundle_path=art["bundle"],
                             features_dir=art["features_dir"],
                             selected_recipe="bc_a")
    assert ei.value.code == "SELECTED_MODEL_MISMATCH"
    # (i) checkpoint bytes not bound by the bundle → refusal
    other_payload = torch.load(art["ckpt"], map_location="cpu",
                               weights_only=False)
    other_payload["best_metric"] = float(
        other_payload["best_metric"]) + 1.0
    other_ckpt = tmp_path / "other.pt"
    torch.save(other_payload, other_ckpt)
    with pytest.raises(FreerunError) as ei:
        load_selected_policy(torch, checkpoint_path=other_ckpt,
                             bundle_path=art["bundle"],
                             features_dir=art["features_dir"])
    assert ei.value.code == "CHECKPOINT_BUNDLE_MISMATCH"
    # (ii) features dir with a DIFFERENT frozen spec → refusal
    spec2, payload2 = build_features_spec(
        s_g=0.0155, s_delta_e=0.0789, element_vocab=[1, 6, 7, 8],
        population={"wave": "other"}, n_atoms_range=(3, 8))
    features2 = tmp_path / "features2"
    features2.mkdir()
    save_spec_payload(payload2, features2 / "features_spec.json")
    write_json(features2 / "features_manifest.json",
               {"spec_digest": spec2.content_digest(), "per_reaction": []})
    with pytest.raises(FreerunError) as ei:
        load_selected_policy(torch, checkpoint_path=art["ckpt"],
                             bundle_path=art["bundle"],
                             features_dir=features2)
    assert ei.value.code == "SPEC_DIGEST_MISMATCH"
    # (iii) tampered action version → refusal (bundle still schema-valid)
    tampered = dict(art["bundle_payload"])
    tampered["action_version"] = \
        "radial_tanh_v0:bc_target=self_entropy_v0:a_max=0.1:sigma_min=0.02"
    bundle2 = tmp_path / "bundle2.json"
    bundle2.write_text(json.dumps(tampered))
    with pytest.raises(FreerunError) as ei:
        load_selected_policy(torch, checkpoint_path=art["ckpt"],
                             bundle_path=bundle2,
                             features_dir=art["features_dir"])
    assert ei.value.code == "ACTION_VERSION_MISMATCH"


# ---------------------------------------------------------------------------
# Record emission + digest binding
# ---------------------------------------------------------------------------

EMIT_IDENTITY = {
    "checkpoint_sha256": "c" * 64,
    "action_version": "radial_tanh_v0:bc_target=teacher_latent_v1"
                      ":a_max=0.1:sigma_min=0.02",
    "spec_digest": "0" * 64,
    "action_max": DEFAULT_A_MAX_ANGSTROM,
}


def test_record_emission_and_digest_binding(tmp_path):
    record = run_rollout(torch=torch, model=make_policy(seed=11),
                         observer=ScriptedObserver(), start=make_start(),
                         scales=SCALES, identity=EMIT_IDENTITY,
                         settings=rollout_settings(max_steps=2))
    record_path, bundle_path, digest_hex = emit_rollout_artifacts(
        out_dir=tmp_path, record=record, identity=EMIT_IDENTITY,
        scales=dict(SCALES), acp_settings_digest="d" * 64)
    assert record_path.is_file() and bundle_path.is_file()
    record_payload = json.loads(record_path.read_text())
    bundle_payload = json.loads(bundle_path.read_text())
    validate_record("pesn_dual_ts_bundle_v2", bundle_payload)
    assert bundle_payload["bc_parent_checkpoint_digest"] == "c" * 64
    assert bundle_payload["dataset_version"] == \
        f"pesn_dual_ts_features_v3@{EMIT_IDENTITY['spec_digest']}"
    assert bundle_payload["reward_version"].startswith(
        "pesn_dual_ts_reward_v2:none_oracle_diagnostic_only")
    assert bundle_payload["production_decision_open"] is False
    assert content_digest_of(bundle_payload) == \
        record_payload["bundle_content_digest"]
    assert verify_record_digest(record_payload)
    assert record_payload["record_content_digest"] == digest_hex
    # tamper detection: any record mutation breaks the digest binding
    tampered = dict(record_payload)
    tampered["n_steps"] = 99
    assert not verify_record_digest(tampered)
    assert compute_record_digest(record_payload) == digest_hex


def test_summarize_rollouts_aggregate():
    records = []
    for idx, (gid, steps) in enumerate((("RXN_T0", 3), ("RXN_T1", 2))):
        rec = run_rollout(torch=torch, model=make_policy(seed=13),
                          observer=ScriptedObserver(),
                          start=make_start(group_id=gid),
                          scales=SCALES, identity=IDENTITY,
                          settings=rollout_settings(max_steps=steps))
        rec["post_hoc_truth_metrics"] = {
            "deviation_final_A": 0.02 + 0.01 * idx,
            "cumulative_deviation_mean_A": 0.015,
            "oracle_arc_start_A": 0.5,
            "oracle_arc_end_A": 0.4,
        }
        records.append(rec)
    agg = summarize_rollouts(records, qc_budget=320)
    assert agg["n_groups"] == 2
    assert agg["qc"]["billed_total"] == 5
    assert agg["qc"]["billed_within_budget"] is True
    assert agg["qc"]["budget"] == 320
    assert agg["deviation_final_median_A"] == pytest.approx(0.025)
    assert set(agg["oracle_arc_start_end_A_per_group"]) == \
        {"RXN_T0", "RXN_T1"}
    assert agg["termination_reasons"] == {"completed_step_cap": 2}
    assert agg["min_interatomic_distance_global_min_A"] > \
        OVERLAP_ABORT_MIN_DISTANCE_ANGSTROM


def test_bond_validity_frozen_covalent_rule():
    edges = np.asarray([[0, 1], [0, 2]])
    # O–H threshold = 0.66 + 0.31 + 0.45 = 1.42 Å (Cordero + 0.45 tol)
    good = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [1.30, 0.0, 0.0]])
    stats = bond_validity(good, Z3, edges)
    assert stats["broken_fraction"] == 0.0 and stats["n_bonds"] == 2
    stretched = good.copy()
    stretched[1, 0] = 1.55     # O–H distance beyond 1.42 Å by exactly 0.13
    stats = bond_validity(stretched, Z3, edges)
    assert stats["broken_fraction"] == pytest.approx(0.5)
    assert stats["max_excess_A"] == pytest.approx(0.13, abs=1e-9)


def test_load_rollout_start_refusals(tmp_path):
    with pytest.raises(Exception):
        load_rollout_start(features_dir=tmp_path / "nope",
                           observed_dir=tmp_path / "nope2",
                           group_id="RXN_X")


def test_freerun_settings_validation():
    with pytest.raises(Exception):
        RolloutSettings(max_steps=0)
    with pytest.raises(Exception):
        RolloutSettings(qc_budget=-1)
    with pytest.raises(Exception):
        RolloutSettings(action_mode="teleport")
