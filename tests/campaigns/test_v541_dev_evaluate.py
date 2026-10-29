"""WP7: v541_dev_evaluate unit tests (fixtures only — no real dev access).

What is pinned here (design §11 matrix):

* screen adjudication gate: E*=E0 is the ONLY legal state; any selected
  candidate / non-E0 winner / advanced candidate is a typed refusal;
* freeze schema + binding completeness (screen / E0x3 / tau / runplan /
  code / stats SHA / dev_access=candidate_evaluation / confirm sealed);
* WP1 seal verification passes on a byte-identical fixture and FAILS
  closed on any file / ledger-prefix / manifest drift;
* strict-load refusal surface + E0-only checkpoint gate (FORBIDDEN /
  IDENTITY_MISMATCH raised BEFORE any model construction);
* tau read-only (never recalibrated) + train_q_mean inner-train-only
  provenance flags;
* constant-q baseline formula (median curve MAE) mirror of evaluate_v53mp;
* denominator exact gates fail-closed on SHA mismatch;
* ledger: purpose=candidate_evaluation, WP1 prefix invariant under append;
* aggregate is descriptive-only: dev_veto_applicable=False and
  selection_or_tuning_performed=False;
* confirm never read; no candidate forward path exists.

The REAL-data execution (`freeze` -> cgroup `run` -> `verify`) runs the
script itself, not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.campaigns import v541_dev_evaluate as vde  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _screen(**over):
    doc = {
        "E_star": "E0",
        "selected": [],
        "n_selected": 0,
        "early_termination": True,
        "early_termination_reason": "all_T1_candidates_failed_H2_MOVE_guard",
        "dev_policy": "WP7 evaluates frozen matched E0 x3 only",
        "candidates": [
            {"embedding_id": e, "outcome": "scientific_failure",
             "advance_to_T3": False, "seed": 17}
            for e in ("E1", "E2", "E3")],
        "E4": {"formal_run_started": False, "outcome": "gate_skipped"},
        "E5": {"formal_run_started": False, "outcome": "gate_skipped"},
        "T3": {"outcome": "gate_skipped"},
        "T4": {"outcome": "gate_skipped"},
    }
    doc.update(over)
    return doc


def _seed_doc(seed=17):
    rmse = {"median_rmse": 0.08, "median_rmse_over_zero": 1.0,
            "cos_eff_median": 0.6, "n_effective_atoms": 10}
    sre = {"dead_zone_fraction": 0.09, "n_active_pairs": 100,
           "n_sign_pairs": 100, "sign_accuracy_cluster_median": 0.8}
    return {
        "schema": vde.SCHEMA_SEED, "seed": seed, "split": "dev",
        "model": vde.E0_MODEL_NAME[seed], "run_id": vde.E0_RUN_ID[seed],
        "checkpoint_sha256": "ab" * 32,
        "strict_load": {"mode": "strict", "ok": True},
        "tau": {"tau_end": 0.9, "source":
                "checkpoint_best.pt#tau_end_calibration",
                "calibrated_on": "inner_val", "recomputed": False},
        "train_q_mean": {"value": 0.1, "derived_from_dev": False,
                         "n_rows_used": 1000, "n_q_masked": 900},
        "denominators": {"n_positions": 135926, "n_positions_ok": 116527,
                         "n_curves": 589, "n_curves_move": 557,
                         "positions_sha256": "c" * 64,
                         "devpack_positions_sha256": "d" * 64,
                         "curves_sha256": "e" * 64, "n_forward_batches": 4},
        "move": {
            "after_repaired": {"key_rmse_over_zero": dict(rmse),
                               "sre_and_sign": dict(sre),
                               "fixed_coverage_direction":
                                   {"key_atom_coverage": 0.98}},
            "before_legacy": {"sre_and_sign": dict(sre)},
            "pair_denominators": {"n_pairs_collected_after": 5000},
        },
        "process": {
            "q_head": {"corr_curve_mean": 0.32, "mae_curve_mean": 0.5,
                       "n_eval_curves": 500,
                       "constant_q_baseline_mae": 0.54,
                       "constant_q_value": 0.1,
                       "learnable_above_constant": True},
            "ordering": {"pairwise_accuracy_full": 0.75, "n_pairs_full": 1712,
                         "pairwise_accuracy_cross_ts": 0.86,
                         "n_pairs_cross_ts": 833},
            "endpoint": {"auprc": 0.27, "auroc": 0.7, "prevalence": 0.17,
                         "n_unmasked": 20000, "n_pos": 3400, "n_neg": 16600,
                         "tau_end": 0.9,
                         "start_misstop_rate_at_tau": 0.01,
                         "ts_misstop_rate_at_tau": 0.2},
        },
        "controls": {"confirm_read": False, "new_QC_calls": 0,
                     "selection_or_tuning_performed": False,
                     "tau_recomputed": False},
        "seed_pass": True,
    }


# ---------------------------------------------------------------------------
# T2: screen adjudication gate
# ---------------------------------------------------------------------------


def test_screen_gate_accepts_e0_only():
    blk = vde.screen_adjudication_gate(_screen())
    assert blk["e_star"] == "E0" and blk["selected"] == []
    assert blk["forbidden_candidates"] == ["E1", "E2", "E3", "E4", "E5"]


@pytest.mark.parametrize("bad", [
    {"E_star": "E3"},
    {"selected": ["E1"], "n_selected": 1},
    {"n_selected": 2},
    {"candidates": [{"embedding_id": "E1", "outcome": "winner",
                     "advance_to_T3": False}]},
    {"candidates": [{"embedding_id": "E2", "outcome": "scientific_failure",
                     "advance_to_T3": True}]},
    {"E4": {"formal_run_started": True, "outcome": "gate_skipped"}},
])
def test_screen_gate_refuses_non_e0_states(bad):
    with pytest.raises(vde.WP7Error) as ei:
        vde.screen_adjudication_gate(_screen(**bad))
    assert ei.value.code == "SCREEN_ADJUDICATION_UNEXPECTED"


# ---------------------------------------------------------------------------
# T9: constant-q baseline block (evaluate_v53mp mirror, median-based)
# ---------------------------------------------------------------------------


def _preds(curve_ids, q_pred, q_true, q_mask):
    return [SimpleNamespace(curve_id=c, q_pred=float(p), q_true=float(t),
                            q_mask=bool(m))
            for c, p, t, m in zip(curve_ids, q_pred, q_true, q_mask)]


def test_constant_q_block_median_formula():
    preds = _preds(
        ["A", "A", "B", "B"],
        q_pred=[0.9, 0.7, 0.2, 0.1],
        q_true=[1.0, 0.5, 0.0, 0.2],
        q_mask=[True, True, True, False])
    blk = vde.constant_q_block(preds, 0.1)
    # const_mae over q-masked rows only: mean(|0.1-[1,0.5,0.2]|)=0.4667
    assert blk["constant_q_value"] == 0.1
    assert blk["constant_q_baseline_mae"] == pytest.approx(0.4666666666666667)
    # median curve MAE: A mean(|0.9-1|,|0.7-0.5|)=0.15, B=|0.2-0|=0.2
    # → median 0.175 < 0.4667
    assert blk["learnable_above_constant"] is True
    # train_q_mean=None → const_mae None, learnable None
    blk2 = vde.constant_q_block(preds, None)
    assert blk2["constant_q_baseline_mae"] is None
    assert blk2["learnable_above_constant"] is None


def test_constant_q_block_all_masked_is_none():
    preds = _preds(["A"], [0.5], [1.0], [False])
    blk = vde.constant_q_block(preds, 0.3)
    assert blk["constant_q_baseline_mae"] is None


# ---------------------------------------------------------------------------
# T13: aggregate is descriptive-only (no selection surface)
# ---------------------------------------------------------------------------


def test_aggregate_three_seed_descriptive_only():
    per = {str(s): vde.light_metrics(_seed_doc(s)) for s in (17, 29, 43)}
    # make seeds differ so mean/min/max are exercised
    per["29"]["endpoint_auprc"] = 0.35
    agg = vde.aggregate_three_seed(per)
    m = agg["metrics"]["endpoint_auprc"]
    assert m["n_seeds_defined"] == 3
    assert m["mean"] == pytest.approx(np.mean([0.27, 0.35, 0.27]))
    assert m["min"] == pytest.approx(0.27) and m["max"] == pytest.approx(0.35)
    assert agg["unit"].startswith("seed")
    # no selection/tuning/veto keys exist at the aggregate level
    for key in ("dev_veto_applicable", "selection_or_tuning_performed",
                "winner", "selected"):
        assert key not in agg


def test_light_metrics_reads_frozen_tau_and_train_q_mean_provenance():
    doc = _seed_doc(17)
    lm = vde.light_metrics(doc)
    assert lm["tau_end"] == 0.9
    assert lm["constant_q_baseline_mae"] == 0.54
    assert doc["tau"]["recomputed"] is False
    assert doc["train_q_mean"]["derived_from_dev"] is False
    assert doc["tau"]["source"].endswith("#tau_end_calibration")


# ---------------------------------------------------------------------------
# T6 + T5: strict-load refusal + E0-only checkpoint gate
# ---------------------------------------------------------------------------


def test_strict_load_check_pass_and_refuses_mismatch():
    import torch

    model = torch.nn.Linear(2, 2, bias=False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    info = vde.strict_load_check(model, state)
    assert info["mode"] == "strict" and info["ok"] is True
    bad = {k: torch.zeros_like(v) for k, v in state.items()}
    bad["unexpected"] = torch.zeros(1)
    with pytest.raises(vde.WP7Error) as ei:
        vde.strict_load_check(model, bad)
    assert ei.value.code == "STRICT_LOAD_FAILED"


def _ckpt_fixture(tmp_path, monkeypatch, embedding_id="E1", seed=17):
    """A minimal torch checkpoint + metrics for the loader's GUARD path."""
    import torch

    run_dir = (Path(tmp_path) / vde.RUNS_DIR / vde.E0_RUN_ID[17])
    run_dir.mkdir(parents=True)
    payload = {
        "schema": vde.V541_CKPT_SCHEMA,
        "model": {},
        "config": {"embedding_id": embedding_id, "seed": seed},
        "identity": {"embedding_id": embedding_id, "seed": seed},
        "tau_end_calibration": {"tau_end": 0.5},
    }
    ckpt = run_dir / "checkpoint_best.pt"
    torch.save(payload, ckpt)
    metrics = {
        "schema": vde.V541_CKPT_SCHEMA,
        "checkpoint_best_sha256": vde._sha256_file(ckpt),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics),
                                          encoding="utf-8")
    return vde.Wp7Paths(tmp_path)


def test_load_e0_model_refuses_non_e0_before_construction(tmp_path,
                                                          monkeypatch):
    paths = _ckpt_fixture(tmp_path, monkeypatch, embedding_id="E1")
    with pytest.raises(vde.WP7Error) as ei:
        vde.load_e0_model(paths, 17, "cpu")
    assert ei.value.code == "FORBIDDEN_CANDIDATE"


def test_load_e0_model_refuses_seed_mismatch(tmp_path, monkeypatch):
    paths = _ckpt_fixture(tmp_path, monkeypatch, embedding_id="E0", seed=99)
    with pytest.raises(vde.WP7Error) as ei:
        vde.load_e0_model(paths, 17, "cpu")
    assert ei.value.code == "CKPT_IDENTITY_MISMATCH"


# ---------------------------------------------------------------------------
# T10: denominator exact gates
# ---------------------------------------------------------------------------


def test_denominator_gates_exact_on_sha_mismatch():
    got = {"n_positions": 100, "n_positions_ok": 80, "n_curves": 10,
           "positions_sha256": "a" * 64, "devpack_positions_sha256": "b" * 64,
           "curves_sha256": "c" * 64}
    want = {"n_dev_positions": 100, "n_move_ok_positions": 80,
            "n_dev_curves": 10, "canonical_positions_sha256": "a" * 64,
            "devpack_positions_sha256": "b" * 64, "curves_sha256": "c" * 64}
    assert all(g["pass"] for g in vde._denominator_gates(got, want))
    want2 = dict(want, canonical_positions_sha256="f" * 64)
    gates = vde._denominator_gates(got, want2)
    assert not all(g["pass"] for g in gates)
    failed = [g["name"] for g in gates if not g["pass"]]
    assert failed == ["canonical_positions_sha_exact"]
    want3 = dict(want, n_dev_positions=999)
    assert not all(g["pass"] for g in vde._denominator_gates(got, want3))


# ---------------------------------------------------------------------------
# T11 + T12: ledger schema + WP1 prefix invariance under append
# ---------------------------------------------------------------------------


def _ledger_paths(tmp_path):
    p = vde.Wp7Paths(tmp_path)
    p.audit_dir.mkdir(parents=True, exist_ok=True)
    wp1 = {
        "schema": vde.SCHEMA_LEDGER, "entry_id": "dev_x_1",
        "actor": "root#wp1b_baseline_repair", "purpose": "baseline_repair",
        "wp": "WP1b", "campaign": "V5.4.1", "split": "dev", "model": "C_s17",
        "ts": "2026-10-27T00:00:00Z", "curve_set_sha256": "c" * 64,
        "positions_sha256": "p" * 64, "n_positions": 135926, "n_curves": 589,
        "code_sha256": "0" * 64, "output_path": "x.json",
        "output_sha256": "1" * 64, "new_QC_calls": 0, "status": "done",
    }
    p.ledger_path.write_text(
        json.dumps(wp1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return p, wp1


def test_ledger_append_keeps_wp1_prefix_and_purpose(tmp_path):
    p, wp1 = _ledger_paths(tmp_path)
    prefix_sha = vde._sha256_text(
        p.ledger_path.read_text(encoding="utf-8"))
    entry = vde.make_ledger_entry(
        model="V541_E0_T0R_S17", output_path="r.json",
        output_sha256="2" * 64, freeze_sha256="3" * 64,
        code_sha256="4" * 64, cost={"new_QC_calls": 0},
        positions_sha256="p" * 64, devpack_positions_sha256="q" * 64,
        curve_set_sha256="c" * 64, n_positions=135926, n_curves=589,
        sealed_reference_manifest_sha256="5" * 64)
    assert entry["purpose"] == "candidate_evaluation"
    assert entry["wp"] == "WP7" and entry["confirm_read"] is False
    assert entry["new_QC_calls"] == 0
    assert entry["sealed_reference_manifest_sha256"] == "5" * 64
    vde.append_ledger(p, entry)
    raw = p.ledger_path.read_text(encoding="utf-8").splitlines()
    # WP1 line byte-identical (prefix invariant)
    assert json.loads(raw[0]) == wp1
    assert vde._sha256_text(raw[0] + "\n") == prefix_sha
    assert json.loads(raw[1])["purpose"] == "candidate_evaluation"


# ---------------------------------------------------------------------------
# T1 + T3 + T4: freeze build / verify + WP1 seal drift (fixture tree)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fixture_tree(tmp_path, monkeypatch):
    """A complete miniature campaign tree + frozen R3 sha override."""
    root = Path(tmp_path)
    paths = vde.Wp7Paths(root)

    # screen
    paths.reports_dir.mkdir(parents=True)
    paths.screen_path.write_text(json.dumps(_screen()), encoding="utf-8")

    # fake R3 init whose sha we inject via monkeypatch (BEFORE the run
    # metrics record it in provenance)
    paths.r3_init.parent.mkdir(parents=True, exist_ok=True)
    paths.r3_init.write_bytes(b"R3-INIT")
    monkeypatch.setattr(vde, "R3_INIT_SHA256",
                        vde._sha256_file(paths.r3_init))

    # E0 runs x3 (checkpoint bytes are arbitrary; metrics cross-references)
    for seed in (17, 29, 43):
        rd = paths.e0_run_dir(seed)
        rd.mkdir(parents=True)
        ck = rd / "checkpoint_best.pt"
        ck.write_bytes(("CKPT-%d" % seed).encode())
        tau = {17: 0.9046342048457465, 29: 0.7806053768606613,
               43: 0.9439198254589221}[seed]
        (rd / "metrics.json").write_text(json.dumps({
            "schema": vde.V541_CKPT_SCHEMA,
            "checkpoint_best_sha256": vde._sha256_file(ck),
            "embedding_id": "E0", "seed": seed, "run_outcome": "done",
            "is_formal_run": True, "eval_scope": "inner_val_only",
            "confirm_read": False, "config_full_sha256": "f" * 64,
            "tau_calibrated_on": "inner_val",
            "tau_end_calibration": {"tau_end": tau,
                                    "target_precision": 0.9,
                                    "achieved_precision": 1.0,
                                    "achieved_recall": 0.0017,
                                    "prevalence": 0.169,
                                    "n_inner_val_unmasked": 6789,
                                    "fallback": None},
            "n_train_samples": 1000, "n_inner_val_samples": 100,
            "identity": {"embedding_id": "E0", "seed": seed,
                         "eval_scope": "inner_val_only"},
            "provenance": {"confirm_read": False,
                           "r3_init_sha256": vde.R3_INIT_SHA256},
        }, indent=1), encoding="utf-8")

    # plan pin
    plan_dir = paths.campaign_dir / "plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "runplan.pin.json").write_text(json.dumps({
        "seeds": [17, 29, 43], "runplan_sha256": "r" * 64,
        "code_sha256": "c" * 64, "config_sha256": "g" * 64,
        "budget_sha256": "b" * 64, "source_manifest_sha256": "s" * 64,
        "common_protocol_sha256": "m" * 64,
        "environment_sha256": "e" * 64, "code_commit": "deadbeef",
    }), encoding="utf-8")

    # sealed dir (5 files) + manifest + WP1 ledger + evaluator report
    paths.sealed_dir.mkdir(parents=True)
    sealed_names = ["v53mp_50k_C_s17_dev.json", "v53mp_50k_C_s29_dev.json",
                    "v53mp_50k_C_s43_dev.json", "r3_reference_dev.json",
                    "h2_move_guard_dev.json"]
    files = {}
    for name in sealed_names:
        fp = paths.sealed_dir / name
        fp.write_text(json.dumps({"sealed": True, "name": name}),
                      encoding="utf-8")
        files[name] = vde._sha256_file(fp)
    manifest = {"schema": "v541_dev_seal_manifest_v1", "sealed": True,
                "files": files, "unseal_rule": "WP1 only"}
    mp = paths.sealed_manifest
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    man_sha = vde._sha256_file(mp)

    ev = {"schema": "v541_evaluator_before_after_v1", "status": "ok",
          "dev_sealed_status": {"pass": True, "denominator_sanity": {
              "n_dev_positions": 135926, "n_move_ok_positions": 116527,
              "n_dev_curves": 589}}}
    paths.inner_val_json.write_text(json.dumps(ev), encoding="utf-8")

    def _wp1_entry(model, out_path, sha):
        return {"schema": vde.SCHEMA_LEDGER, "entry_id": "dev_%s_1" % model,
                "actor": "root#wp1b", "purpose": "baseline_repair",
                "wp": "WP1b", "campaign": "V5.4.1", "split": "dev",
                "model": model, "ts": "2026-10-27T00:00:00Z",
                "curve_set_sha256": "c" * 64,
                "positions_sha256": "d" * 64, "n_positions": 135926,
                "n_curves": 589, "code_sha256": "0" * 64,
                "output_path": out_path, "output_sha256": sha,
                "new_QC_calls": 0, "status": "done"}

    entries = [_wp1_entry(n.split("_dev")[0],
                          "audits/dev_baseline_sealed/%s" % n, files[n])
               for n in sealed_names]
    entries.append(_wp1_entry("seal_manifest",
                              "audits/dev_baseline_sealed/manifest.json",
                              man_sha))
    paths.ledger_path.write_text(
        "".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n"
                for e in entries), encoding="utf-8")
    return paths


def test_freeze_build_verify_roundtrip(fixture_tree):
    paths = fixture_tree
    freeze = vde.build_freeze(paths)
    assert freeze["schema"] == vde.SCHEMA_FREEZE
    assert freeze["dev_access"]["type"] == "candidate_evaluation"
    assert freeze["dev_access"]["split"] == "dev"
    assert freeze["confirm"] == {"status": "sealed", "confirm_read": False,
                                 "new_QC_calls": 0}
    assert freeze["screen"]["e_star"] == "E0"
    assert list(freeze["frozen_set"]["per_seed"]) == ["17", "29", "43"]
    for blk in freeze["inner_val_tau"]["per_seed"].values():
        assert blk["calibrated_on"] == "inner_val"
        assert blk["recomputed_on_dev"] is False
        assert blk["source"].endswith("#tau_end_calibration")
    assert freeze["wp1_seal"]["ledger_entries_wp1"] == 6
    assert "scripts/campaigns/r3_v53mp_scale_paired_ci.py" in \
        freeze["code_bindings"]["stats_code_files"]
    assert freeze["dev_reference_binding"]["n_dev_positions"] == 135926
    # roundtrip verify
    vde.verify_freeze(paths, freeze)


def test_freeze_verify_detects_drift(fixture_tree):
    paths = fixture_tree
    freeze = vde.build_freeze(paths)
    # drift the screen (a bound file)
    doc = json.loads(paths.screen_path.read_text())
    doc["E_star"] = "E3"
    paths.screen_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(vde.WP7Error) as ei:
        vde.verify_freeze(paths, freeze)
    assert ei.value.code.startswith("FREEZE_DRIFT") or \
        ei.value.code == "SCREEN_ADJUDICATION_UNEXPECTED"


def test_wp1_seal_verify_and_tamper(fixture_tree):
    paths = fixture_tree
    freeze = vde.build_freeze(paths)
    seal = vde.verify_wp1_seal(paths, freeze)
    assert seal["pass"] is True and seal["n_wp1_entries"] == 6
    # tamper a sealed file
    target = paths.sealed_dir / "r3_reference_dev.json"
    target.write_text(target.read_text() + " ", encoding="utf-8")
    with pytest.raises(vde.WP7Error) as ei:
        vde.verify_wp1_seal(paths, freeze)
    assert ei.value.code == "WP1_SEAL_FILE_DRIFT"


def test_wp1_seal_detects_ledger_prefix_tamper(fixture_tree):
    paths = fixture_tree
    freeze = vde.build_freeze(paths)
    lines = paths.ledger_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["output_sha256"] = "f" * 64           # rewrite a WP1 record
    lines[0] = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    paths.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(vde.WP7Error) as ei:
        vde.verify_wp1_seal(paths, freeze)
    assert ei.value.code == "WP1_LEDGER_PREFIX_DRIFT"


def test_wp1_seal_rejects_foreign_purpose_after_prefix(fixture_tree):
    paths = fixture_tree
    freeze = vde.build_freeze(paths)
    foreign = {"schema": vde.SCHEMA_LEDGER, "purpose": "baseline_repair",
               "model": "sneaky", "wp": "WP7"}
    with open(paths.ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(foreign, sort_keys=True) + "\n")
    with pytest.raises(vde.WP7Error) as ei:
        vde.verify_wp1_seal(paths, freeze)
    # a second baseline_repair line appended AFTER freeze is caught by the
    # post-WP1 purpose guard (only candidate_evaluation may follow)
    assert ei.value.code in ("LEDGER_POST_WP1_PURPOSE",
                             "WP1_SEAL_LEDGER_MISMATCH",
                             "WP1_LEDGER_PREFIX_DRIFT")


def test_build_freeze_tolerates_existing_wp7_entries(fixture_tree):
    """WP7's own candidate_evaluation lines must NOT invalidate a
    (re)freeze: the WP1 block stays the leading byte prefix."""
    paths = fixture_tree
    entry = vde.make_ledger_entry(
        model="V541_E0_T0R_S17", output_path="r.json",
        output_sha256="2" * 64, freeze_sha256="3" * 64,
        code_sha256="4" * 64, cost={"new_QC_calls": 0},
        positions_sha256="p" * 64, devpack_positions_sha256="q" * 64,
        curve_set_sha256="c" * 64, n_positions=135926, n_curves=589)
    vde.append_ledger(paths, entry)
    freeze = vde.build_freeze(paths)
    assert freeze["wp1_seal"]["ledger_entries_wp1"] == 6
    seal = vde.verify_wp1_seal(paths, freeze)
    assert seal["n_wp1_entries"] == 6 and seal["n_entries_total"] == 7


# ---------------------------------------------------------------------------
# T14 + T16: confirm never read; run refuses without freeze / fixture flag
# ---------------------------------------------------------------------------


def test_confirm_split_is_structurally_absent():
    src = (PROJECT_ROOT / "scripts/campaigns/"
           "v541_dev_evaluate.py").read_text(encoding="utf-8")
    # no code path may open the confirm subset
    assert 'split="confirm"' not in src
    assert "'confirm'" not in src or "_subset_curve_ids(split_doc" in src
    assert "confirm_read\": False" in src or '"confirm_read": False' in src


def test_run_refuses_without_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(vde, "PROJECT_ROOT", Path(tmp_path))
    rc = vde.main(["run", "--device", "cpu"])
    assert rc == 1
    # failure record written, dev never touched
    paths = vde.Wp7Paths(Path(tmp_path))
    assert paths.eval_report_path.is_file()
    doc = json.loads(paths.eval_report_path.read_text())
    assert doc["status"] == "failed"
    assert doc["failure"]["code"] == "FREEZE_MISSING"
    assert doc["controls"]["confirm_read"] is False


def test_run_refuses_fixture_flag_without_env(tmp_path, monkeypatch):
    monkeypatch.setattr(vde, "PROJECT_ROOT", Path(tmp_path))
    rc = vde.main(["run", "--device", "cpu", "--limit-positions", "8"])
    assert rc == 1
    paths = vde.Wp7Paths(Path(tmp_path))
    doc = json.loads(paths.eval_report_path.read_text())
    assert doc["failure"]["code"] == "FIXTURE_FLAG_REQUIRED"


def test_run_refuses_non_frozen_seed_set(tmp_path, monkeypatch):
    monkeypatch.setattr(vde, "PROJECT_ROOT", Path(tmp_path))
    rc = vde.main(["run", "--seeds", "17", "--device", "cpu"])
    assert rc == 1
    paths = vde.Wp7Paths(Path(tmp_path))
    doc = json.loads(paths.eval_report_path.read_text())
    assert doc["failure"]["code"] == "SEED_SET_MISMATCH"


def test_freeze_cli_writes_freeze_on_fixture_tree(fixture_tree,
                                                  monkeypatch):
    paths = fixture_tree
    monkeypatch.setattr(vde, "PROJECT_ROOT", paths.project_root)
    rc = vde.main(["freeze"])
    assert rc == 0 and paths.freeze_path.is_file()
    doc = json.loads(paths.freeze_path.read_text())
    assert doc["schema"] == vde.SCHEMA_FREEZE
    # idempotent re-run
    assert vde.main(["freeze"]) == 0
