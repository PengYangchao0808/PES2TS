"""V5.4.1 WP7 no-physics rollout adapter tests
(``scripts/campaigns/v541_rollout.py``).

Coverage (task spec items 1–4):

* (1) checkpoint/model adapter — E0 ``V541Model`` strict-load from a
  fixture checkpoint; E1–E5 (rejected candidates) refused at every
  surface (assert helper, plan line, checkpoint config, metrics); the
  inner-val frozen tau is read from BOTH the checkpoint payload and
  ``metrics.json`` (``endpoint_head.tau_end``) with three-way agreement;
* (2) curve lists — the frozen V5.3-MP rule
  ``sha256('v53mp-rollout|17|<curve_id>')`` reused verbatim (same
  function/salt), 24 ⊂ 96 ⊂ dev machine-checked, list SHAs / source
  split SHA / rule SHA saved, the previously recorded 24-list reused
  exactly; against the REAL frozen dev subset when present;
* (3) discipline — reference_normalized / autonomous strictly separate
  in aggregation (never merged), zero physics cost
  (N_energy/N_gradient/N_Hessian/N_OptTS_trials=0) on every attempt,
  production No-Move closed (frozen tau evaluated as diagnostic only,
  episodes always run to budget), claims limited to
  TS_candidate/endpoint_reached, non-key atom displacement = 0;
* (4) CLI gating — ``plan`` is a dry-run that makes no dev predictions;
  ``run-rollouts`` refuses without ``--execute`` and without a matching
  pinned plan; the dev access ledger entry carries
  ``purpose=candidate_evaluation``.

All fixtures are synthetic (tiny checkpoint + stub runplan/split) — no
real 50K dev access, no GPU, no training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agent_ops.feature_gate import v53mp_rollout_exec as v53  # noqa: E402
from scripts.agent_ops.supervisor.campaign_root import (  # noqa: E402
    CampaignRootError,
)
from scripts.campaigns import v541_rollout as R  # noqa: E402
from pes2ts_core.pesn.contracts.key_atom_v3 import IntentRecordV3  # noqa: E402

REAL_SPLIT = Path("/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
REAL_SPLIT_LOCAL = (
    PROJECT_ROOT / "outputs" / "pesn" / "v53mp" / "data"
    / "split_manifest_v53mp.json")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _dev_ids(n: int = 120) -> list[str]:
    return [f"RXN_{i:09d}" for i in range(n)]


def _synthetic_split_doc(dev: list[str]) -> dict:
    return {
        "schema": "v53mp_split_v1",
        "frozen": True,
        "subsets": {
            "dev": dev,
            "confirm": ["RXN_999999999"],
            "pool": {"50k": ["RXN_888888888", "RXN_888888889"]},
        },
    }


def _write_split(tmp_path: Path, dev: list[str]) -> Path:
    path = tmp_path / "split_manifest_v53mp.json"
    path.write_text(json.dumps(_synthetic_split_doc(dev)), encoding="utf-8")
    return path


def _write_checkpoint(path: Path, *, embedding_id: str = "E0",
                      tau_end: float = 0.7) -> dict:
    """Tiny real V541Model state_dict round-tripped through a ckpt file."""
    from pes2ts_core.pesn.models.v541.model import V541Model

    torch.manual_seed(0)
    model = V541Model(embedding_id="E0")
    payload = {
        "model": model.state_dict(),
        "config": {"embedding_id": embedding_id},
        "tau_end_calibration": {"tau_end": tau_end,
                                "target_precision": 0.9},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": path, "sha256": sha, "payload": payload}


def _write_metrics(run_dir: Path, ckpt_sha: str, *, tau_end: float = 0.7,
                   embedding_id: str = "E0") -> Path:
    metrics = {
        "embedding_id": embedding_id,
        "checkpoint_best_sha256": ckpt_sha,
        "tau_calibrated_on": "inner_val",
        "endpoint_head": {"tau_end": tau_end, "auprc": 0.3, "auroc": 0.7},
        "tau_end_calibration": {"tau_end": tau_end},
    }
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return path


def _write_runplan(root: Path, lines: list[dict]) -> Path:
    plan_dir = root / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    header = {"record_type": "plan_header", "schema": "pes2ts_v541_runplan_v1",
              "campaign_id": "PES2TS_V541_EMBED_50K"}
    path = plan_dir / "runplan.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True) + "\n")
        for ln in lines:
            fh.write(json.dumps(ln, sort_keys=True) + "\n")
    return path


def _fixture_root(tmp_path: Path, *, n_dev: int = 120,
                  tau_ckpt: float = 0.7, tau_metrics: float = 0.7,
                  embedding_id: str = "E0",
                  seed: int = 17, stage: str = "T0-R",
                  run_id: str = "V541_T0R_E0_S17") -> dict:
    """A complete tiny V541 campaign root (plan + runs + split)."""
    root = tmp_path / "outputs" / "pesn" / "v541_embed_50k"
    root.mkdir(parents=True, exist_ok=True)
    dev = _dev_ids(n_dev)
    split = _write_split(tmp_path, dev)
    _write_runplan(root, [{
        "record_type": "v541_run", "run_id": run_id,
        "embedding_id": embedding_id, "seed": seed, "stage": stage,
        "status": "planned",
    }])
    run_dir = root / "runs" / run_id
    ckpt = _write_checkpoint(run_dir / "checkpoint_best.pt",
                             embedding_id=embedding_id, tau_end=tau_ckpt)
    _write_metrics(run_dir, ckpt["sha256"], tau_end=tau_metrics,
                   embedding_id=embedding_id)
    return {"root": root, "split": split, "ckpt": ckpt, "run_id": run_id,
            "dev": dev}


def _synthetic_intent() -> IntentRecordV3:
    return IntentRecordV3(
        intent_id="SYN_fwd", curve_id="SYN", direction="forward",
        reverse_of_intent_id="SYN_rev", intent_pair_id="SYN_pair",
        g_src_bonds=((0, 1, 1.0),), g_tgt_bonds=((1, 2, 1.0),),
        cv_deltas=(1.0,), g_atoms=(6, 1, 8, 6),
        cv_coordinates=((0, 1), (1, 2)),
        cv_kind="graph_edit_bond_lengths")


def _synthetic_ctx() -> v53.EpisodeContextV53MP:
    frames = [
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [3.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0],
                  [2.5, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
    ]
    intent = _synthetic_intent()
    key_rows = (0, 1, 2)
    cv_pairs = [(0, 1), (1, 2)]

    def dkey(f: np.ndarray) -> np.ndarray:
        return np.array([float(np.linalg.norm(f[j] - f[i]))
                         for i, j in cv_pairs])

    key_idx = np.asarray(key_rows, dtype=np.int64)
    l_key = v53.l_key_from_arc([f[key_idx] for f in frames])
    return v53.EpisodeContextV53MP(
        curve_id="SYN", direction="forward", intent=intent,
        start_node=0, target_node=2, ts_index=1,
        atomic_numbers=(6, 1, 8, 6),
        start_xyz=tuple(tuple(float(v) for v in row) for row in frames[0]),
        arc_nodes=(0, 1, 2),
        arc_dkey=(dkey(frames[0]), dkey(frames[1]), dkey(frames[2])),
        target_xyz=tuple(tuple(float(v) for v in row) for row in frames[2]),
        ts_xyz=tuple(tuple(float(v) for v in row) for row in frames[1]),
        l_key=float(l_key),
        reference_budget=v53.reference_budget(l_key))


def _tiny_e0_model(tmp_path: Path, *, force_high_endpoint: bool = False):
    ckpt = _write_checkpoint(tmp_path / "probe.pt")
    model = R.load_v541_e0_checkpoint(ckpt["path"])
    if force_high_endpoint:
        with torch.no_grad():
            model.process_head.endpoint_head.fc.weight.zero_()
            model.process_head.endpoint_head.fc.bias.fill_(8.0)
    return model


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        output_root="outputs/pesn/v541_embed_50k",
        runs_root=None, out_dir=None, runplan=None,
        split="", run_ids="V541_T0R_E0_S17",
        regression_curves=24, rollout_curves=96,
        autonomous_steps=200, prior_manifest=None,
        no_verify_frozen=False, load_model=True, smoke=False,
        out=None, tables_dir="", plan="", execute=False, cache_curves=8,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# (2) curve lists: rule reuse, subsets, SHAs
# ---------------------------------------------------------------------------
def test_lists_follow_frozen_rule_and_subsets() -> None:
    dev = _dev_ids()
    lists = R.build_rollout_lists(dev, verify_frozen=False)
    reg = lists["regression"]["curve_ids"]
    roll = lists["rollout"]["curve_ids"]
    assert len(reg) == 24 and len(roll) == 96
    # ordering follows sha256('v53mp-rollout|17|<curve_id>') verbatim
    expect = sorted(dev, key=lambda c: hashlib.sha256(
        f"v53mp-rollout|17|{c}".encode("utf-8")).hexdigest())
    assert roll == expect[:96]
    assert reg == expect[:24]
    assert roll[:24] == reg                      # 24 is the prefix of 96
    assert set(reg) <= set(roll) <= set(dev)     # 24 ⊂ 96 ⊂ dev
    # list SHAs + rule SHA + salt are recorded
    assert lists["regression"]["curve_list_sha256"] == \
        v53.curve_list_sha256(reg)
    assert lists["rollout"]["curve_list_sha256"] == v53.curve_list_sha256(roll)
    assert lists["selection_salt"] == "17"
    assert lists["selection_rule_sha256"] == \
        hashlib.sha256(v53.SELECTION_RULE.encode("utf-8")).hexdigest()
    assert lists["checks"]["regression_is_rollout_prefix"] is True
    assert lists["checks"]["subset_24_in_96"] is True
    assert lists["checks"]["subset_96_in_dev"] is True
    # input-order invariance (set-deterministic)
    assert R.build_rollout_lists(list(reversed(dev)),
                                 verify_frozen=False) == lists


def test_lists_deterministic_on_real_dev_subset() -> None:
    path = REAL_SPLIT if REAL_SPLIT.is_file() else REAL_SPLIT_LOCAL
    if not path.is_file():
        pytest.skip("frozen v53mp split manifest not present")
    split = R.load_dev_subset(path)
    assert split["dev_size"] == 589
    lists = R.build_rollout_lists(split["dev_ids"], verify_frozen=True)
    # the frozen SHAs recorded at preregistration time must hold
    assert lists["regression"]["curve_list_sha256"] == \
        R.FROZEN_24_LIST_SHA256
    assert lists["rollout"]["curve_list_sha256"] == R.FROZEN_96_LIST_SHA256
    # the previously recorded 24-list is reused EXACTLY (24 ⊂ 96)
    assert tuple(lists["regression"]["curve_ids"]) == R.FROZEN_PRIOR_24_IDS
    assert lists["checks"]["prior_24_list_reused"] is True
    assert lists["checks"]["frozen_24_sha_match"] is True
    assert lists["checks"]["frozen_96_sha_match"] is True
    assert set(lists["rollout"]["curve_ids"]) <= set(split["dev_ids"])
    R.assert_dev_only_selection(lists, split)


def test_lists_membership_drift_detected() -> None:
    dev = _dev_ids()
    base = R.build_rollout_lists(dev, verify_frozen=False)
    # changing the dev membership changes the lists (hash-order rule)
    dev[0] = "RXN_000000120"
    drifted = R.build_rollout_lists(dev, verify_frozen=False)
    assert drifted["rollout"]["curve_ids"] != base["rollout"]["curve_ids"]
    # and the formal default (verify_frozen=True) refuses any synthetic
    # list that does not match the preregistered frozen SHAs
    with pytest.raises(R.V541RolloutError) as exc:
        R.build_rollout_lists(dev)
    assert exc.value.code == "LIST_SHA_DRIFT"


def test_prior_manifest_cross_check(tmp_path: Path) -> None:
    lists = R.build_rollout_lists(_dev_ids(), verify_frozen=False)
    manifest = {
        "split": {
            "curve_ids": lists["regression"]["curve_ids"],
            "curve_list_sha256": lists["regression"]["curve_list_sha256"],
        },
    }
    path = tmp_path / "rollouts_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    out = R.verify_prior_manifest_lists(lists, path)
    assert out["match"] is True
    # tampered prior list → typed refusal
    manifest["split"]["curve_ids"] = manifest["split"]["curve_ids"][1:]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(R.V541RolloutError) as exc:
        R.verify_prior_manifest_lists(lists, path)
    assert exc.value.code == "PRIOR_LIST_MISMATCH"


# ---------------------------------------------------------------------------
# (1) checkpoint / model / tau adapter
# ---------------------------------------------------------------------------
def test_e0_strict_load_and_e1_e5_refused(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "e0.pt")
    model = R.load_v541_e0_checkpoint(ckpt["path"])
    from pes2ts_core.pesn.models.v541.model import V541Model
    assert isinstance(model, V541Model)
    assert model.is_e0 and not model.has_branch
    assert not model.training                    # eval mode
    # E1–E5 (rejected candidates) are refused BEFORE any state-dict work
    bad = _write_checkpoint(tmp_path / "e1.pt", embedding_id="E1")
    with pytest.raises(R.V541RolloutError) as exc:
        R.load_v541_e0_checkpoint(bad["path"])
    assert exc.value.code == "V541_ROLLOUT_EMBEDDING_REFUSED"
    for emb in ("E2", "E3", "E4", "E5"):
        with pytest.raises(R.V541RolloutError) as exc:
            R.assert_e0_only(emb, context="test")
        assert exc.value.code == "V541_ROLLOUT_EMBEDDING_REFUSED"
    assert R.assert_e0_only("E0", context="test") == "E0"
    with pytest.raises(R.V541RolloutError):
        R.assert_e0_only("E7", context="test")     # unknown → refused too


def test_frozen_tau_three_way_agreement(tmp_path: Path) -> None:
    fx = _fixture_root(tmp_path, tau_ckpt=0.7, tau_metrics=0.7)
    from scripts.campaigns.v541_runplan import read_plan
    plan = read_plan(fx["root"] / "plan" / "runplan.jsonl")
    art = R.load_v541_run_artifact(fx["root"] / "runs", fx["run_id"],
                                   plan=plan)
    tau = art["tau"]
    assert tau["tau_end"] == 0.7
    assert tau["match"] is True
    assert tau["tau_calibrated_on"] == "inner_val"
    assert tau["checkpoint_tau_end"] == 0.7
    assert tau["metrics_endpoint_head_tau_end"] == 0.7
    assert tau["metrics_tau_end_calibration_tau_end"] == 0.7
    assert art["embedding_id"] == "E0"
    assert art["seed"] == 17 and art["stage"] == "T0-R"
    assert art["model"] is not None
    assert art["sha256"] == fx["ckpt"]["sha256"]


def test_tau_mismatch_refused(tmp_path: Path) -> None:
    fx = _fixture_root(tmp_path, tau_ckpt=0.7, tau_metrics=0.8)
    with pytest.raises(R.V541RolloutError) as exc:
        R.load_v541_run_artifact(fx["root"] / "runs", fx["run_id"])
    assert exc.value.code == "TAU_MISMATCH"


def test_tau_missing_refused(tmp_path: Path) -> None:
    fx = _fixture_root(tmp_path)
    metrics_path = fx["root"] / "runs" / fx["run_id"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    del metrics["endpoint_head"]["tau_end"]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(R.V541RolloutError) as exc:
        R.load_v541_run_artifact(fx["root"] / "runs", fx["run_id"])
    assert exc.value.code == "TAU_MISSING"


def test_run_artifact_refuses_non_e0_and_unknown_stage(tmp_path: Path) -> None:
    # E1 run (rejected candidate) — refused at the plan line AND metrics
    fx = _fixture_root(tmp_path, embedding_id="E1", run_id="V541_E1_S17",
                       seed=17)
    from scripts.campaigns.v541_runplan import read_plan
    plan = read_plan(fx["root"] / "plan" / "runplan.jsonl")
    with pytest.raises(R.V541RolloutError) as exc:
        R.load_v541_run_artifact(fx["root"] / "runs", "V541_E1_S17",
                                 plan=plan)
    assert exc.value.code == "V541_ROLLOUT_EMBEDDING_REFUSED"
    # E0 but a T4 diagnostic stage — refused
    fx2 = _fixture_root(tmp_path / "t4", run_id="V541_T4_E0_MLP_S17",
                        stage="T4")
    plan2 = read_plan(fx2["root"] / "plan" / "runplan.jsonl")
    with pytest.raises(R.V541RolloutError) as exc:
        R.load_v541_run_artifact(fx2["root"] / "runs",
                                 "V541_T4_E0_MLP_S17", plan=plan2)
    assert exc.value.code == "V541_ROLLOUT_STAGE_REFUSED"
    # run_id outside the V541 namespace — refused
    with pytest.raises(R.V541RolloutError) as exc:
        R.load_v541_run_artifact(fx["root"] / "runs", "v53mp_20k_A_s17")
    assert exc.value.code == "RUN_ID_NAMESPACE"


def test_checkpoint_sha_drift_refused(tmp_path: Path) -> None:
    fx = _fixture_root(tmp_path)
    metrics_path = fx["root"] / "runs" / fx["run_id"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["checkpoint_best_sha256"] = "0" * 64
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(R.V541RolloutError) as exc:
        R.load_v541_run_artifact(fx["root"] / "runs", fx["run_id"])
    assert exc.value.code == "CHECKPOINT_SHA_DRIFT"


# ---------------------------------------------------------------------------
# (3) episode adapter: no-physics, zero cost, diagnostic tau, separate modes
# ---------------------------------------------------------------------------
_EPISODE_EXTRA_FIELDS = (
    "schema", "campaign", "gate", "embedding_id", "tau_scope",
    "production_no_move_open", "no_move_diagnostic", "cost",
    "nonkey_atom_max_displacement", "claim_scope",
)


def test_episode_v541_record_discipline(tmp_path: Path) -> None:
    model = _tiny_e0_model(tmp_path)
    rec = R.run_episode_v541(
        _synthetic_ctx(), model=model, budget=3,
        budget_mode="autonomous", run_id="V541_T0R_E0_S17",
        checkpoint_sha256="a" * 64, seed=17, tau_end=0.7)
    for key in _EPISODE_EXTRA_FIELDS:
        assert key in rec, key
    assert rec["schema"] == R.EPISODE_SCHEMA_V541
    assert rec["campaign"] == R.CAMPAIGN_ID_V541
    assert rec["gate"] == R.GATE_ID_V541
    assert rec["claim_scope"] == "TS_candidate_only"
    assert rec["arm"] == "E0" and rec["embedding_id"] == "E0"
    assert rec["tau_end"] == 0.7 and rec["tau_scope"] == "inner_val_frozen"
    # production No-Move is NOT open: the episode ran the FULL budget
    # even though tau is recorded
    assert rec["production_no_move_open"] is False
    assert rec["online_stop_role"] == "not_open_production"
    assert rec["halted_online_stop"] is False
    assert rec["n_steps_executed"] == 3 == rec["budget"]
    assert rec["summary"]["budget_exhausted"] is True
    # zero-cost physics attestation
    cost = rec["cost"]
    for key in ("N_energy", "N_gradient", "N_Hessian", "N_OptTS_trials",
                "new_QC_calls", "physics_calls"):
        assert cost[key] == 0, key
    assert cost["n_forward_calls"] == 3
    # non-key atom displacement is exactly 0 (spectator atom never moves)
    assert rec["nonkey_atom_max_displacement"] == 0.0
    # JSON-safe + no forbidden claim strings
    assert json.loads(json.dumps(rec, sort_keys=True)) == rec
    text = json.dumps(rec).lower()
    assert "verified_ts" not in text and "verified path" not in text


def test_episode_no_move_diagnostic_forced_high(tmp_path: Path) -> None:
    """Frozen tau would fire — recorded as diagnostic, NOT as a halt."""
    model = _tiny_e0_model(tmp_path, force_high_endpoint=True)
    rec = R.run_episode_v541(
        _synthetic_ctx(), model=model, budget=4,
        budget_mode="reference_normalized", run_id="V541_T0R_E0_S17",
        checkpoint_sha256="c" * 64, seed=17, tau_end=0.5)
    diag = rec["no_move_diagnostic"]
    assert diag["would_stop"] is True
    assert diag["would_stop_step"] == 1
    assert diag["would_stop_endpoint_prob"] >= 0.5
    assert diag["role"] == "diagnostic_only"
    # but the episode STILL ran to budget (no production stop)
    assert rec["n_steps_executed"] == 4
    assert rec["halted_online_stop"] is False
    # the synthetic geometry is 2.0 Å from the target at step 1 → the
    # would-be halt is erroneous (outside the 0.10 Å endpoint window)
    assert diag["would_stop_erroneous"] is True


def test_episode_requires_e0_and_valid_tau(tmp_path: Path) -> None:
    model = _tiny_e0_model(tmp_path)
    with pytest.raises(R.V541RolloutError) as exc:
        R.run_episode_v541(
            _synthetic_ctx(), model=model, budget=2,
            budget_mode="autonomous", run_id="r",
            checkpoint_sha256="b" * 64, seed=17, tau_end=0.7,
            embedding_id="E2")
    assert exc.value.code == "V541_ROLLOUT_EMBEDDING_REFUSED"
    with pytest.raises(R.V541RolloutError):
        R.run_episode_v541(
            _synthetic_ctx(), model=model, budget=2,
            budget_mode="autonomous", run_id="r",
            checkpoint_sha256="b" * 64, seed=17, tau_end=float("nan"))


def test_aggregate_separates_budget_modes_and_zero_cost(tmp_path: Path) -> None:
    model = _tiny_e0_model(tmp_path)
    records = [
        R.run_episode_v541(
            _synthetic_ctx(), model=model, budget=2,
            budget_mode=mode, run_id="V541_T0R_E0_S17",
            checkpoint_sha256="a" * 64, seed=17, tau_end=0.7)
        for mode in ("reference_normalized", "autonomous")
    ]
    agg = R.aggregate_episode_records_v541(records)
    assert agg["schema"] == R.AGGREGATE_SCHEMA_V541
    keys = [(g["arm"], g["budget_mode"]) for g in agg["groups"]]
    assert keys == [("E0", "autonomous"), ("E0", "reference_normalized")]
    # no merged success-rate container exists anywhere
    assert "overall" not in agg and "success_rate" not in agg
    text = json.dumps(agg).lower()
    assert "merged" not in text or "never merged" in text
    # zero-cost totals across all attempts
    totals = agg["physics_cost_totals"]
    for key in ("N_energy", "N_gradient", "N_Hessian", "N_OptTS_trials",
                "new_QC_calls", "physics_calls"):
        assert totals[key] == 0, key
    assert totals["n_forward_calls"] == 4     # 2 episodes × 2 steps
    # per-group diagnostic counters exist and production stop stays closed
    for g in agg["groups"]:
        assert g["production_no_move_open"] is False
        assert "would_stop_episodes" in g
        assert "would_stop_erroneous_episodes" in g
        assert g["nonkey_atom_max_displacement"] == 0.0


def test_aggregate_empty_refused() -> None:
    with pytest.raises(R.V541RolloutError) as exc:
        R.aggregate_episode_records_v541([])
    assert exc.value.code == "NO_RECORDS"


# ---------------------------------------------------------------------------
# (4) CLI: plan dry-run + run gating + ledger
# ---------------------------------------------------------------------------
def test_plan_dry_run_writes_plan(tmp_path: Path, monkeypatch,
                                  capsys) -> None:
    fx = _fixture_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = _args(
        output_root=str(fx["root"]),
        split=str(fx["split"]),
        run_ids=fx["run_id"],
        no_verify_frozen=True,
    )
    assert R.cmd_plan(args) == 0
    plan_path = fx["root"] / "reports" / "rollout_plan.json"
    assert plan_path.is_file()
    doc = json.loads(plan_path.read_text())
    assert doc["schema"] == R.PLAN_SCHEMA_V541
    assert doc["mode"] == "plan_dry_run"
    assert doc["executed"] is False
    assert doc["dev_predictions_made"] is False
    assert doc["production_no_move_open"] is False
    # lists + all identity SHAs are saved
    lists = doc["lists"]
    assert len(lists["regression"]["curve_ids"]) == 24
    assert len(lists["rollout"]["curve_ids"]) == 96
    assert lists["selection_rule_sha256"] == R.selection_rule_sha256()
    assert doc["split"]["sha256"] == hashlib.sha256(
        fx["split"].read_bytes()).hexdigest()
    assert doc["split"]["subset_used"] == "dev"
    ck = doc["checkpoints"][fx["run_id"]]
    assert ck["embedding_id"] == "E0"
    assert ck["sha256"] == fx["ckpt"]["sha256"]
    assert ck["tau"]["tau_end"] == 0.7
    assert ck["model_strict_loaded"] is True
    assert doc["physics_cost_policy"]["N_energy"] == 0
    out = capsys.readouterr().out
    assert "plan (dry-run" in out


def test_plan_refuses_outside_campaign_root(tmp_path: Path, monkeypatch):
    fx = _fixture_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = _args(output_root="outputs/pesn/other_campaign",
                 split=str(fx["split"]))
    with pytest.raises(CampaignRootError):
        R.cmd_plan(args)


def test_plan_refuses_e1_run_id(tmp_path: Path, monkeypatch) -> None:
    fx = _fixture_root(tmp_path, embedding_id="E1", run_id="V541_E1_S17")
    monkeypatch.chdir(tmp_path)
    args = _args(output_root=str(fx["root"]), split=str(fx["split"]),
                 run_ids="V541_E1_S17", no_verify_frozen=True)
    with pytest.raises(R.V541RolloutError) as exc:
        R.cmd_plan(args)
    assert exc.value.code == "V541_ROLLOUT_EMBEDDING_REFUSED"


def test_run_rollouts_refuses_without_execute(tmp_path: Path,
                                              monkeypatch) -> None:
    fx = _fixture_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = _args(
        output_root=str(fx["root"]), split=str(fx["split"]),
        run_ids=fx["run_id"], no_verify_frozen=True,
        tables_dir=str(tmp_path), plan=str(tmp_path / "plan.json"),
        execute=False)
    with pytest.raises(R.V541RolloutError) as exc:
        R.cmd_run_rollouts(args)
    assert exc.value.code == "V541_ROLLOUT_NOT_EXECUTING"
    # nothing was written: no ledger, no reports
    assert not (fx["root"] / "audits" / "dev_access_ledger.jsonl").exists()
    assert not (fx["root"] / "reports" / "rollout_comparison.json").exists()


def test_run_rollouts_requires_matching_pinned_plan(tmp_path: Path,
                                                    monkeypatch) -> None:
    fx = _fixture_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    # a pinned plan whose 96-list sha does not match → typed refusal
    bad_plan = {
        "schema": R.PLAN_SCHEMA_V541, "executed": False,
        "lists": {"rollout": {"curve_list_sha256": "0" * 64},
                  "regression": {"curve_list_sha256": "0" * 64}},
        "checkpoints": {},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(bad_plan), encoding="utf-8")
    args = _args(
        output_root=str(fx["root"]), split=str(fx["split"]),
        run_ids=fx["run_id"], no_verify_frozen=True,
        tables_dir=str(tmp_path), plan=str(plan_path), execute=True)
    with pytest.raises(R.V541RolloutError) as exc:
        R.cmd_run_rollouts(args)
    assert exc.value.code == "PLAN_LIST_DRIFT"


def test_ledger_entry_candidate_evaluation() -> None:
    split = {"sha256": "b" * 64, "dev_size": 589}
    lists = {
        "regression": {"n_curves": 24, "curve_ids": _dev_ids(24),
                       "curve_list_sha256": "c" * 64},
        "rollout": {"n_curves": 96, "curve_ids": _dev_ids(96),
                    "curve_list_sha256": "d" * 64},
    }
    entry = R.ledger_entry(
        run_ids=["V541_T0R_E0_S17"], lists=lists, split=split,
        output_path="outputs/pesn/v541_embed_50k/reports/x.json",
        output_sha256="e" * 64, cost=dict(R.ZERO_COST_BLOCK))
    assert entry["schema"] == R.LEDGER_SCHEMA_V541
    assert entry["purpose"] == "candidate_evaluation"   # WP7, not WP1
    assert entry["wp"] == "WP7"
    assert entry["split"] == "dev"
    assert entry["n_curves"] == 96
    assert entry["curve_set_sha256"] == "d" * 64
    assert entry["regression_curve_set_sha256"] == "c" * 64
    assert entry["split_sha256"] == "b" * 64
    assert entry["new_QC_calls"] == 0
    assert entry["unseal_before_wp7"] is False
    for key in ("N_energy", "N_gradient", "N_Hessian", "N_OptTS_trials"):
        assert entry["cost"][key] == 0


def test_main_cli_exit_codes(tmp_path: Path, monkeypatch, capsys) -> None:
    fx = _fixture_root(tmp_path)
    monkeypatch.chdir(tmp_path)
    rc = R.main([
        "plan", "--output-root", str(fx["root"]),
        "--split", str(fx["split"]),
        "--run-ids", fx["run_id"],
        "--no-verify-frozen",
    ])
    assert rc == 0
    # run-rollouts without --execute exits 2 with the typed code on stderr
    rc = R.main([
        "run-rollouts", "--output-root", str(fx["root"]),
        "--split", str(fx["split"]),
        "--run-ids", fx["run_id"],
        "--tables-dir", str(tmp_path),
        "--plan", str(fx["root"] / "reports" / "rollout_plan.json"),
        "--no-verify-frozen",
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "V541_ROLLOUT_NOT_EXECUTING" in err


# ---------------------------------------------------------------------------
# loader lock (WP7 rollout smoke attempt 1: v1 light-cache SIGTERM)
# ---------------------------------------------------------------------------
def test_open_rollout_dataset_uses_columnar_loader(tmp_path: Path,
                                                   monkeypatch) -> None:
    """open_rollout_dataset MUST construct ColumnarKeyAtomDatasetV3 with the
    same label_statuses/split/cache_curves surface — and never pass the v1
    light_cache materialisation flag."""
    import pes2ts_core.pesn.offline.data.key_atom_dataset_v3 as v1
    import pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col as col
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        LABEL_STATUSES,
    )

    captured: dict = {}
    sentinel = object()

    class _FakeCol:
        def __init__(self, tables_dir, **kwargs):
            captured["tables_dir"] = str(tables_dir)
            captured.update(kwargs)

    monkeypatch.setattr(col, "ColumnarKeyAtomDatasetV3", _FakeCol)
    monkeypatch.setattr(v1, "ShardedKeyAtomDatasetV3",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("v1 sharded loader must not be "
                                           "constructed")))
    out = R.open_rollout_dataset(
        tables_dir=tmp_path, split_manifest=tmp_path / "split.json",
        cache_curves=1500)
    assert isinstance(out, _FakeCol)
    assert captured["tables_dir"] == str(tmp_path)
    assert captured["split_manifest"] == str(tmp_path / "split.json")
    assert captured["label_statuses"] == LABEL_STATUSES
    assert captured["cache_curves"] == 1500
    assert "light_cache" not in captured   # v2 IS light; flag never passed


def test_source_never_constructs_v1_light_cache_loader() -> None:
    """AST lock: the rollout module contains NO ``ShardedKeyAtomDatasetV3``
    construction and no ``light_cache=True`` keyword anywhere in code."""
    import ast

    src = Path(R.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == "ShardedKeyAtomDatasetV3":
                bad_calls.append(f"L{node.lineno}: ShardedKeyAtomDatasetV3")
            for kw in node.keywords:
                if kw.arg == "light_cache":
                    bad_calls.append(f"L{node.lineno}: light_cache kwarg")
    assert bad_calls == [], bad_calls
    # the columnar loader IS referenced (construction + manifest identity)
    assert "ColumnarKeyAtomDatasetV3" in src
    # the attempt-1 failure is preserved as documentation
    assert "SIGTERM" in src and "resource_failure" in src


def test_cmd_run_rollouts_dataset_block_opens_columnar(tmp_path: Path,
                                                       monkeypatch) -> None:
    """The run-rollouts dataset-open block calls open_rollout_dataset with
    exactly (tables_dir, split, cache_curves) — no light_cache."""
    import inspect

    src = inspect.getsource(R.cmd_run_rollouts)
    assert "open_rollout_dataset(" in src
    assert "ShardedKeyAtomDatasetV3" not in src
    assert "light_cache" not in src
    assert "atomic_numbers_map_v541(" in src
    assert "group_intents_by_curve(" in src
    # the old O(n_rows) full-sample scan helper must not be used
    assert "_v53.atomic_numbers_map(" not in src


def test_atomic_numbers_map_v541_uses_curve_table_not_samples() -> None:
    """The selected-96 lookup must go through the eager per-curve table
    (O(96)); touching the LAZY ``samples`` sequence at the 50K tier would
    construct ~14.7M records — the very blowup that SIGTERM'd attempt 1."""

    class _Stub:
        _curve_atomic = {"RXN_A": (6, 1, 8), "RXN_B": (7, 1, 16, 1)}

        @property
        def samples(self):      # pragma: no cover - must never be touched
            raise AssertionError("samples must not be iterated (v2 lazy "
                                 "sequence; use the per-curve table)")

        @property
        def intents(self):
            return {}

    out = R.atomic_numbers_map_v541(_Stub(), ["RXN_B", "RXN_A"])
    assert out == {"RXN_A": (6, 1, 8), "RXN_B": (7, 1, 16, 1)}
    with pytest.raises(R.V541RolloutError) as exc:
        R.atomic_numbers_map_v541(_Stub(), ["RXN_MISSING"])
    assert exc.value.code == "ATOMIC_NUMBERS"


def test_atomic_numbers_map_v541_intent_g_atoms_fallback() -> None:
    """A curve the per-curve table misses falls back to intent g_atoms
    (public API) — never a guessed Z list."""

    class _Stub:
        _curve_atomic = {}

        @property
        def samples(self):      # pragma: no cover
            raise AssertionError("samples must not be iterated")

        @property
        def intents(self):
            intent = _synthetic_intent()      # g_atoms=(6, 1, 8, 6)
            return {("SYN", intent.intent_id): intent}

    out = R.atomic_numbers_map_v541(_Stub(), ["SYN"])
    assert out == {"SYN": (6, 1, 8, 6)}


def test_group_intents_by_curve_partitions_like_frozen_helper() -> None:
    fwd = _synthetic_intent()
    rev = IntentRecordV3(
        intent_id="SYN_rev", curve_id="SYN", direction="reverse",
        reverse_of_intent_id="SYN_fwd", intent_pair_id="SYN_pair",
        g_src_bonds=((1, 2, 1.0),), g_tgt_bonds=((0, 1, 1.0),),
        cv_deltas=(-1.0,), g_atoms=(6, 1, 8, 6),
        cv_coordinates=((0, 1), (1, 2)),
        cv_kind="graph_edit_bond_lengths")
    other = IntentRecordV3(
        intent_id="OTH_fwd", curve_id="OTH", direction="forward",
        reverse_of_intent_id="OTH_rev", intent_pair_id="OTH_pair",
        g_src_bonds=((0, 1, 1.0),), g_tgt_bonds=((1, 2, 1.0),),
        cv_deltas=(1.0,), g_atoms=(6, 1),
        cv_coordinates=((0, 1),), cv_kind="graph_edit_bond_lengths")
    intents = {
        ("SYN", "SYN_fwd"): fwd, ("SYN", "SYN_rev"): rev,
        ("OTH", "OTH_fwd"): other,
    }
    grouped = R.group_intents_by_curve(intents, ["SYN"])
    assert set(grouped) == {"SYN"}
    # the frozen helper consumes the grouped view EXACTLY as the full map
    directions = v53._direction_intents(grouped["SYN"], "SYN")
    assert set(directions) == {"forward", "reverse"}
    assert directions["forward"] is fwd and directions["reverse"] is rev


def test_loader_block_preserves_attempt1_and_authorizes_rerun() -> None:
    blk = R.loader_manifest_block(1500)
    assert blk["class"] == "ColumnarKeyAtomDatasetV3"
    assert blk["light_cache_used"] is False
    assert blk["bounded_cache_curves"] == 1500
    p1 = blk["prior_attempt1"]
    # first resource_failure record preserved verbatim (never rewritten)
    assert p1["status"] == "resource_failure"
    assert p1["exit_code"] == 143 and p1["signal"] == "SIGTERM"
    assert p1["peak_anon_bytes"] == 12260667392
    assert p1["max_swap_current_bytes"] == 0
    assert p1["oom_kill"] == 0
    assert p1["episodes_written"] == 0
    assert p1["ledger_entry_preserved"] is True
    assert p1["rerun_authorized"] is True
    assert "implementation fix" in p1["rerun_reason"]
    # the fix note states the rerun is allowed AFTER the implementation fix
    assert "implementation fix" in blk["fix_note"]
    assert "the rerun is an authorized implementation fix" in blk["fix_note"]


def test_ledger_append_only_preserves_prior_entries(tmp_path: Path) -> None:
    """Appending the rerun entries must NEVER rewrite the attempt-1 record
    (or any prior line) in the dev access ledger."""
    root = tmp_path / "outputs" / "pesn" / "v541_embed_50k"
    split = {"sha256": "b" * 64, "dev_size": 589}
    lists = {
        "regression": {"n_curves": 24, "curve_ids": _dev_ids(24),
                       "curve_list_sha256": "c" * 64},
        "rollout": {"n_curves": 96, "curve_ids": _dev_ids(96),
                    "curve_list_sha256": "d" * 64},
    }
    entry1 = R.ledger_entry(
        run_ids=["V541_T0R_E0_S17"], lists=lists, split=split,
        output_path="outputs/pesn/v541_embed_50k/reports",
        output_sha256="", cost=dict(R.ZERO_COST_BLOCK), status="started")
    path = R.append_ledger(root, entry1)
    line1 = path.read_text(encoding="utf-8").splitlines()[0]
    # rerun appends its own started + done entries…
    entry2 = R.ledger_entry(
        run_ids=["V541_T0R_E0_S17"], lists=lists, split=split,
        output_path="outputs/pesn/v541_embed_50k/reports/rollouts_manifest.json",
        output_sha256="e" * 64, cost=dict(R.ZERO_COST_BLOCK), status="done")
    R.append_ledger(root, entry2)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # …and the attempt-1 line is byte-identical (append-only guarantee)
    assert lines[0] == line1
    assert json.loads(lines[0])["status"] == "started"
    assert json.loads(lines[1])["status"] == "done"
    assert all(json.loads(ln)["purpose"] == "candidate_evaluation"
               for ln in lines)
