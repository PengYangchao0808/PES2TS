"""Lane D protocol freeze + evaluation tests (CPU, synthetic 8-group data).

Covers: intersection freeze correctness (subdivided states and their
predecessors excluded, unchanged transfers included, terminals excluded),
write-once semantics, two-protocol MSE, critical-pair distance error + sign
accuracy, near-zero RMSE / alpha, and legacy/native four-gate boundaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.offline.evaluation.dual_ts_protocols import (  # noqa: E402
    ProtocolError,
    apply_gates,
    evaluate_protocol,
    freeze_protocols,
    load_protocol_reference,
    verify_freeze_payload,
)


# ---------------------------------------------------------------------------
# Synthetic arm builder
# ---------------------------------------------------------------------------

def _write_npz(path: Path, **arrays) -> None:
    np.savez(path, **arrays)


def _near_zero(action, is_terminal: bool) -> np.ndarray:
    if action is None:
        return np.ones((2,), dtype=bool)
    return np.linalg.norm(np.asarray(action, dtype=float), axis=1) < 0.01


def _build_synthetic_arms(tmp_path: Path) -> tuple[Path, Path, Path]:
    """8 groups, 2 atoms each; group 0 has a midpoint insertion between s0/s1."""
    legacy_dir = tmp_path / "wp4_monitor" / "features_v41"
    native_dir = tmp_path / "wp42_campaign" / "native_monitor_features"
    legacy_obs = tmp_path / "wp4_monitor" / "teachers_observed"
    for d in (legacy_dir, native_dir, legacy_obs):
        d.mkdir(parents=True, exist_ok=True)

    legacy_rows, native_rows = [], []
    for gi in range(8):
        gid = f"RXN_{gi:010d}"
        rid = gid
        br = f"{gid}#brA"
        z = np.array([6, 1], dtype=int)
        x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        x1 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        x2 = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        xm = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])

        native_states = [
            (f"{gid}|A|0000", br, f"{gid}|A|0001", x0, x1 - x0, 2.0, False),
            (f"{gid}|A|0001", br, f"{gid}|A|0002", x1, x2 - x1, 1.0, False),
            (f"{gid}|A|0002", br, "", x2, None, 0.0, True),
        ]
        if gi == 0:
            legacy_states = [
                (f"{gid}|A|0000", br, f"{gid}|A|0000m", x0, xm - x0, 2.0, False),
                (f"{gid}|A|0000m", br, f"{gid}|A|0001", xm, x1 - xm, 1.5, False),
                (f"{gid}|A|0001", br, f"{gid}|A|0002", x1, x2 - x1, 1.0, False),
                (f"{gid}|A|0002", br, "", x2, None, 0.0, True),
            ]
        else:
            legacy_states = list(native_states)

        def _write_legacy_npz(states):
            S = len(states)
            _write_npz(
                legacy_dir / f"features_{gid}.npz",
                reaction_id=np.array(rid), reaction_group_id=np.array(gid),
                atomic_numbers=z,
                state_id=np.array([s[0] for s in states]),
                next_state_id=np.array([s[2] for s in states]),
                branch_id=np.array([s[1] for s in states]),
                x_angstrom=np.stack([s[3] for s in states]).astype(np.float64),
                near_zero_action_mask=np.stack(
                    [_near_zero(s[4], s[6]) for s in states]).astype(bool),
                is_teacher_endpoint=np.array([s[6] for s in states]),
                edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                edge_feats_a=np.array([[1, 1, 1, 0], [1, 1, 1, 0]],
                                      dtype=np.float32),
                edge_feats_b=np.array([[1, 1, 1, 0], [1, 1, 1, 0]],
                                      dtype=np.float32),
            )
            # teacher labels
            labels = []
            for s in states:
                action = (None if s[4] is None
                          else np.asarray(s[4]).round(10).tolist())
                labels.append({
                    "state_id": s[0],
                    "action_delta": action,
                    "remaining_arc_length": s[5],
                    "teacher_endpoint": s[6],
                    "ts_displacement": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                })
            (legacy_obs / f"{rid}.teacher.json").write_text(
                json.dumps({"schema_version": "pesn_dual_ts_teacher_v2",
                            "labels": labels}))

        def _write_native_npz(states):
            S = len(states)
            _write_npz(
                native_dir / f"features_{gid}.npz",
                reaction_id=np.array(rid), reaction_group_id=np.array(gid),
                atomic_numbers=z,
                state_id=np.array([s[0] for s in states]),
                next_state_id=np.array([s[2] for s in states]),
                branch_id=np.array([s[1] for s in states]),
                x_angstrom=np.stack([s[3] for s in states]).astype(np.float64),
                action_angstrom=np.stack(
                    [(np.zeros((2, 3)) if s[4] is None else s[4])
                     for s in states]).astype(np.float64),
                is_terminal=np.array([s[6] for s in states]),
                remaining_arc_length=np.array([s[5] for s in states],
                                              dtype=np.float64),
                near_zero_action_mask=np.stack(
                    [_near_zero(s[4], s[6]) for s in states]).astype(bool),
                edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                edge_feats_a=np.array([[1, 1, 1, 0], [1, 1, 1, 0]],
                                      dtype=np.float32),
                edge_feats_b=np.array([[1, 1, 1, 0], [1, 1, 1, 0]],
                                      dtype=np.float32),
            )

        _write_legacy_npz(legacy_states)
        _write_native_npz(native_states)
        legacy_rows.append({"reaction_group_id": gid, "npz_file": f"features_{gid}.npz"})
        native_rows.append({"reaction_group_id": gid, "npz_file": f"features_{gid}.npz"})

    (legacy_dir / "features_manifest.json").write_text(json.dumps({
        "schema_version": "pes2ts_dual_ts_wp41_features_manifest_v1",
        "spec_digest": "legacy_spec_digest_000",
        "observed_dir": str(legacy_obs),
        "per_reaction": legacy_rows,
    }))
    (native_dir / "features_manifest.json").write_text(json.dumps({
        "schema_version": "pes2ts_dual_ts_wp41_features_manifest_v1",
        "spec_digest": "native_spec_digest_000",
        "per_reaction": native_rows,
    }))
    return legacy_dir, native_dir, legacy_obs


def _freeze(tmp_path: Path):
    legacy_dir, native_dir, _ = _build_synthetic_arms(tmp_path)
    out = tmp_path / "wp42_campaign" / "protocol_freeze.json"
    frozen = freeze_protocols(legacy_dir, native_dir, out)
    return legacy_dir, native_dir, out, frozen


# ---------------------------------------------------------------------------
# Intersection freeze correctness
# ---------------------------------------------------------------------------

def test_freeze_intersection_correctness(tmp_path):
    _, _, out, frozen = _freeze(tmp_path)
    common_roster = set(frozen["protocols"]["common_unchanged"]["roster"])
    g0 = "RXN_0000000000"
    g1 = "RXN_0000000001"
    # subdivided (inserted) state must NOT be in the intersection
    assert f"{g0}|A|0000m" not in common_roster
    # subdivided PREDECESSOR must NOT be in the intersection
    assert f"{g0}|A|0000" not in common_roster
    # the unchanged state (s1) of group 0 IS in the intersection
    assert f"{g0}|A|0001" in common_roster
    # terminal states never enter the transfer intersection
    assert f"{g1}|A|0002" not in common_roster
    # non-inserted groups keep both non-terminal transfers
    assert f"{g1}|A|0000" in common_roster
    assert f"{g1}|A|0001" in common_roster
    # roster is sorted + unique
    roster = frozen["protocols"]["common_unchanged"]["roster"]
    assert roster == sorted(roster)
    assert len(roster) == len(set(roster))
    # per-group counts: group 0 has exactly 1 unchanged transfer, others 2
    pg = frozen["protocols"]["common_unchanged"]["per_group_counts"]
    assert pg[g0]["n_transfers"] == 1
    for gi in range(1, 8):
        assert pg[f"RXN_{gi:010d}"]["n_transfers"] == 2
    # legacy + native are full 8 groups with all actions
    assert frozen["protocols"]["legacy_full"]["n_groups"] == 8
    assert frozen["protocols"]["native_full"]["n_groups"] == 8
    # payload digest is a 64-hex identity
    assert len(frozen["payload_sha256"]) == 64
    assert verify_freeze_payload(frozen)["ok"] is True
    assert out.is_file()


def test_freeze_write_once(tmp_path):
    legacy_dir, native_dir, _ = _build_synthetic_arms(tmp_path)
    out = tmp_path / "wp42_campaign" / "protocol_freeze.json"
    freeze_protocols(legacy_dir, native_dir, out)
    with pytest.raises(ProtocolError) as exc:
        freeze_protocols(legacy_dir, native_dir, out)
    assert exc.value.code == "FREEZE_EXISTS"


def test_load_protocol_reference_matches_digest(tmp_path):
    legacy_dir, native_dir, _ = _build_synthetic_arms(tmp_path)
    frozen = load_protocol_reference(legacy_dir, native_dir)
    assert frozen["_reference"]["legacy_full"]
    assert frozen["_reference"]["native_full"]
    assert verify_freeze_payload(frozen)["ok"] is True


# ---------------------------------------------------------------------------
# Metric hand-computed examples
# ---------------------------------------------------------------------------

def _transfer(state_id="s1", group_id="g1", branch_id="g1#brA",
              action=None, near_zero=None, is_terminal=False, ral=1.0,
              x=None, atomic_numbers=None, critical_pairs=None):
    a = None if action is None else np.asarray(action, dtype=np.float64)
    n = 2 if atomic_numbers is None else len(atomic_numbers)
    z = np.asarray(atomic_numbers if atomic_numbers is not None
                   else [6, 1], dtype=int)
    if near_zero is None:
        near_zero = (np.ones(n, dtype=bool) if a is None
                     else np.linalg.norm(a, axis=1) < 0.01)
    x = np.zeros((n, 3)) if x is None else np.asarray(x, dtype=np.float64)
    return {
        "state_id": state_id, "group_id": group_id, "branch_id": branch_id,
        "next_state_id": "nxt", "n_atoms": n, "atomic_numbers": z,
        "action": a, "near_zero_mask": np.asarray(near_zero, dtype=bool),
        "is_terminal": is_terminal, "remaining_arc_length": ral,
        "x_angstrom": x, "critical_pairs": critical_pairs or [],
    }


def _frozen_with(*transfers, protocol="legacy_full"):
    return {
        "_reference": {protocol: list(transfers)},
        "protocols": {"common_unchanged": {"missing": {}}},
    }


def test_two_protocol_mse_and_cosine():
    t = _transfer(
        action=[[1.0, 0, 0], [0, 2.0, 0]],
    )
    preds = {"s1": np.array([[0.0, 0, 0], [0, 1.0, 0]])}
    m = evaluate_protocol(preds, "legacy_full", _frozen_with(t))
    # per-atom MSE: atom0 = |[0,0,0]-[1,0,0]|^2 = 1; atom1 = |[0,1,0]-[0,2,0]|^2 = 1
    assert m["mse"]["atom_pooled"]["mse_A2"] == pytest.approx(1.0)
    assert m["mse"]["group_equal_weighted"]["mse_A2"] == pytest.approx(1.0)
    # zero baseline: atom0 = 1, atom1 = 4 → mean 2.5 → ratio 0.4
    assert m["mse"]["atom_pooled"]["zero_A2"] == pytest.approx(2.5)
    assert m["mse"]["atom_pooled"]["ratio_vs_zero"] == pytest.approx(0.4)
    assert m["mse"]["group_equal_weighted"]["ratio_below_one_groups"] == 1
    # direction cosine: atom0 cos 0 (pred=0), atom1 cos 1.0; |a|-weighted
    assert m["direction"]["overall_cosine"] == pytest.approx(2.0 / 3.0)
    assert m["denominator"]["n_direction_atoms"] == 2
    assert m["denominator"]["n_action_states"] == 1


def test_near_zero_rmse_and_alpha():
    t = _transfer(
        action=[[0.0, 0, 0], [0, 2.0, 0]],
        near_zero=[True, False],
    )
    preds = {"s1": np.array([[0.1, 0, 0], [0, 1.0, 0]])}
    m = evaluate_protocol(preds, "legacy_full", _frozen_with(t))
    # near-zero atom0: |pred - a|^2 = 0.01 → RMSE = 0.1
    assert m["near_zero"]["abs_rmse_A"] == pytest.approx(0.1)
    assert m["near_zero"]["n_atoms"] == 1
    # alpha defined only on the direction atom (atom1): |pred|/|a| = 0.5
    assert m["alpha"]["n_atoms"] == 1
    assert m["alpha"]["mean"] == pytest.approx(0.5)
    assert m["alpha"]["near_zero_undefined_note"]


def test_critical_pair_distance_error_and_sign():
    t = _transfer(
        action=[[-0.5, 0, 0], [0.5, 0, 0], [0.0, 0, 0]],
        near_zero=[False, False, True],
        atomic_numbers=[6, 6, 1],
        x=[[0.0, 0, 0], [2.0, 0, 0], [5.0, 5.0, 5.0]],
        critical_pairs=[(0, 1)],
    )
    preds = {"s1": np.array([[-1.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0]])}
    m = evaluate_protocol(preds, "legacy_full", _frozen_with(t))
    crit = m["slices"]["critical_pairs"]
    # d_before = 2, d_true = 3, d_pred = 4 → error = 1
    assert crit["distance_error_mae_A"] == pytest.approx(1.0)
    assert crit["distance_error_rmse_A"] == pytest.approx(1.0)
    assert crit["n_pairs"] == 1
    # dd_true = +1, dd_pred = +2 → sign match
    assert crit["distance_change"]["n_eligible"] == 1
    assert crit["distance_change"]["sign_accuracy"] == pytest.approx(1.0)
    # atom 2 is a spectator (not in any critical pair)
    assert m["slices"]["spectator_atoms"]["n_atoms"] == 1


def test_shape_mismatch_and_missing_prediction_raise():
    t = _transfer(action=[[1.0, 0, 0], [0, 1.0, 0]])
    # wrong shape → typed refusal (no silent truncation)
    with pytest.raises(ProtocolError) as exc:
        evaluate_protocol({"s1": np.zeros((3, 3))}, "legacy_full",
                          _frozen_with(t))
    assert exc.value.code == "SHAPE_MISMATCH"
    # missing prediction → typed refusal
    with pytest.raises(ProtocolError) as exc:
        evaluate_protocol({}, "legacy_full", _frozen_with(t))
    assert exc.value.code == "MISSING_PREDICTION"


# ---------------------------------------------------------------------------
# Four-gate boundaries
# ---------------------------------------------------------------------------

def _gate_metrics(ratio=0.9, below=8, cos=0.35, near_ts=0.45, protocol="legacy_full"):
    return {
        "protocol": protocol,
        "mse": {"group_equal_weighted": {
            "ratio_median": ratio, "ratio_below_one_groups": below}},
        "direction": {"cosine_median": cos},
        "near_ts": {"cosine_median": near_ts},
    }


def test_apply_gates_legacy_pass_and_boundaries():
    # all pass
    assert apply_gates(_gate_metrics())["decision"] == "PASS"
    # ratio == 1.0 is NOT < 1.0 (strict)
    assert apply_gates(_gate_metrics(ratio=1.0))["decision"] == "BELOW_GATE"
    # < 6/8 groups below 1.0 → fail
    assert apply_gates(_gate_metrics(below=5))["decision"] == "BELOW_GATE"
    # cos >= 0.30 passes; 0.299 fails
    assert apply_gates(_gate_metrics(cos=0.30))["decision"] == "PASS"
    assert apply_gates(_gate_metrics(cos=0.299))["decision"] == "BELOW_GATE"
    # near-TS cos >= 0.40 passes; 0.399 fails
    assert apply_gates(_gate_metrics(near_ts=0.40))["decision"] == "PASS"
    assert apply_gates(_gate_metrics(near_ts=0.399))["decision"] == "BELOW_GATE"
    # 6/8 exactly passes (>= 6)
    assert apply_gates(_gate_metrics(below=6))["decision"] == "PASS"


def test_apply_gates_native_new_protocol_flag():
    d = apply_gates(_gate_metrics(protocol="native_full"))
    assert d["decision"] == "PASS"
    assert d["new_protocol"] is True
    assert d["gate_name"] == "native_four_gate_research"


def test_apply_gates_common_not_gated():
    d = apply_gates({"protocol": "common_unchanged", "missing": {}})
    assert d["apply_gate"] is False
    assert d["decision"] == "NOT_GATED"
    assert d["new_protocol"] is False


def test_apply_gates_unknown_protocol_raises():
    with pytest.raises(ProtocolError):
        apply_gates(_gate_metrics(protocol="bogus"))
