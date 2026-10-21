#!/usr/bin/env python3
"""V5.2-R stage D2 phase 1: input-sufficiency probes P0–P4 (CPU, frozen R3).

Implements plan ``docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md``
§4–§5 (lane D2, phase 1).  Writes ONLY under ``--out`` (default
``outputs/pesn/v52r/d2``): ``probe_spec.json`` (PRE-REGISTERED before any
model run), ``evidence_pack.json`` (schema ``pes2ts_v52r_d2_evidence_v1``)
and ``D2_PROBE.md``.  P5 is OUT OF SCOPE here except one stub flag — another
lane owns it and it may not start until D0.1 anomalies are cleared (plan §3.1).

Protocol (plan §4.0 / §5, frozen):

* data — dev folds 0–2 only; sealed folds 3/4 COUNTED ONLY (never encoded,
  never featurized, never probed; asserted in code);
* encoder — frozen cached R3 embeddings from ``--embeddings-dir`` (default
  the D1 cache); no torch, no training, no encoder update;
* split — per eval fold F: train = other two dev folds, eval = F, grouped by
  curve; inner ``GroupShuffleSplit(test_size=0.25, random_state=0)`` for the
  pre-registered threshold (smallest threshold with precision ≥ 0.80 on the
  inner split, else thr=∅ → predict nothing);
* units — 9 (fold × seed) per probe; aggregates as median [min, max];
  block probes are seed-independent (pinned ``random_state=0``) so they are
  FIT ONCE PER FOLD and their identical metrics replicate across the 3 seed
  units — computationally equivalent to 9 fits, documented in the spec;
* probes — P0 (D1-identical baseline), P1 (goal-distance oracle), P1c
  (+chirality), P2 (label oracle, self-check), P3 (topology-only, no
  coordinates), P4 (full single-frame geometry, no goal).  Primary model =
  HistGradientBoostingClassifier(class_weight='balanced', random_state=0);
  secondary logistic / MLP(32,) as D1 (global MLP budget 600 s shared);
  frozen-encoder z/h variants logistic [z64] / [h48-key] / [z⊕h] with D1's
  exact C-grid recipe;
* stats — paired Δ vs P0 per fold-seed unit; curve-cluster bootstrap
  (1000, seed 0) 95% CI of ΔAUPRC for P1−P0, P4−P0 (gating) and P1c−P0,
  P3−P0 (exploratory); P5−P0 recorded PENDING (phase 2); per-curve scatter;
* gate — reports only: P2 self-check, P3 leak verdict, two computable MME
  comparisons (PASS / PENDING-GATE).  ``training_authorized`` is ALWAYS
  false: D2-G is a human/manager review, this script never authorizes GPU.

``--strict`` exits 1 on pipeline-integrity failures only (P0 reproduction
FAIL, P2 self-check FAIL, P3 leak) — MME outcomes are reported, not gated.

Usage::

    python scripts/agent_ops/data_gate/v52r_d2_probe.py [--strict]
        [--tables-dir outputs/pesn/30h_campaign/data]
        [--embeddings-dir outputs/pesn/v52r/d1]
        [--out outputs/pesn/v52r/d2] [--bootstrap 1000]
        [--d1-pack outputs/pesn/v52r/d1/evidence_pack.json] [--p5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ops.data_gate import v52r_d1_probe as d1  # noqa: E402
from pes2ts_core.pesn.offline.data.geometry_stereo import (  # noqa: E402
    MAX_K,
    MAX_K_PAIRS,
    k_feature_blocks,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    DATA_MANIFEST_FILENAME,
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    DEGENERATE_TS_CURVES,
    EndpointStopLabelTable,
    build_endpoint_stop_labels,
)

SCHEMA_ID = "pes2ts_v52r_d2_evidence_v1"
SPEC_SCHEMA = "pes2ts_v52r_d2_probe_spec_v1"
GATE_ID = "V5.2R-D2-PHASE1-PROBE"
DEFAULT_TABLES_DIR = REPO_ROOT / "outputs/pesn/30h_campaign/data"
DEFAULT_EMBEDDINGS_DIR = REPO_ROOT / "outputs/pesn/v52r/d1"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/pesn/v52r/d2"
DEFAULT_D1_PACK = REPO_ROOT / "outputs/pesn/v52r/d1/evidence_pack.json"
CAMPAIGN_CONFIG = d1.CAMPAIGN_CONFIG
LABEL_BUILDER_SRC = REPO_ROOT / "pes2ts_core/pesn/offline/data/key_atom_endpoint_labels_v1.py"
GEOMETRY_STEREO_SRC = REPO_ROOT / "pes2ts_core/pesn/offline/data/geometry_stereo.py"

DEV_FOLDS = d1.DEV_FOLDS
SEALED_FOLDS = d1.SEALED_FOLDS
SEEDS = d1.SEEDS
PROBE_SEED = d1.PROBE_SEED
MLP_BUDGET_S = d1.MLP_TIME_BUDGET_S            # 600 s, GLOBAL across all MLP fits
PRECISION_TARGET = d1.PRECISION_TARGET         # 0.80
BOOTSTRAP_RESAMPLES = d1.BOOTSTRAP_RESAMPLES   # 1000
BOOTSTRAP_SEED = d1.BOOTSTRAP_SEED             # 0
P2_SELF_CHECK_MIN = 0.99
P3_LEAK_MARGIN = 0.05
P0_REPRO_TOL = 1e-6
MME_P1 = 0.10
MME_P4 = 0.05
MME_P5 = 0.10

PROBE_ORDER = ("P0", "P1", "P1c", "P2", "P3", "P4")
GOAL_FREE_PROBES = ("P0", "P4")          # may NOT touch the goal frame
ENCODER_VARIANTS = ("logistic_z", "logistic_h", "logistic_zh", "mlp_z")
#: D1 probe name → D2 key (P0 must reproduce D1 bit-for-bit within tol)
D1_PROBE_MAP = {
    "constant": "P0/constant",
    "logistic_tab": "P0/logistic",
    "hgb_tab": "P0/hgb",
    "logistic_z": "enc/logistic_z",
    "mlp_z": "enc/mlp_z",
    "logistic_h": "enc/logistic_h",
}
P3_ORDER_BINS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
P3_N_FEATURES = 22


# ---------------------------------------------------------------------------
# pre-registration (pure; written BEFORE any model run)
# ---------------------------------------------------------------------------


def feature_layout() -> dict[str, dict[str, Any]]:
    """Frozen per-probe column blocks (exact composition; goes into the spec)."""
    p1_cols = [
        {"block": "k_pair_cur", "start": 0, "stop": 28,
         "desc": "current-frame K pair distances, padded to 28 (D1-identical)"},
        {"block": "direction_onehot", "start": 28, "stop": 30,
         "desc": "intent direction one-hot [forward, reverse]"},
        {"block": "k_pair_goal", "start": 30, "stop": 58,
         "desc": "GOAL-frame K pair distances, same padding"},
        {"block": "delta_cur_minus_goal", "start": 58, "stop": 86,
         "desc": "current − goal per pair distance"},
        {"block": "abs_delta", "start": 86, "stop": 114,
         "desc": "|current − goal| per pair distance"},
    ]
    return {
        "P0": {
            "purpose": "no-goal baseline; EXACTLY the D1 tabular features "
                       "(cross-checked against the D1 evidence pack)",
            "goal_info": "none",
            "n_features": 30,
            "columns": [
                {"block": "k_pair_cur", "start": 0, "stop": 28,
                 "desc": "current-frame K pair distances, padded to 28"},
                {"block": "direction_onehot", "start": 28, "stop": 30,
                 "desc": "intent direction one-hot [forward, reverse]"},
            ],
        },
        "P1": {
            "purpose": "goal-distance oracle (diagnostic, not deployable); "
                       "lower bound of goal-information value (mirror-blind)",
            "goal_info": "goal K distances only",
            "n_features": 114,
            "columns": p1_cols,
        },
        "P1c": {
            "purpose": "P1 + chirality of current AND goal frames over sorted "
                       "K-quads — separates mirror ambiguity",
            "goal_info": "goal K distances + goal chirality",
            "n_features": 394,
            "columns": p1_cols + [
                {"block": "cur_signed_volumes", "start": 114, "stop": 184,
                 "desc": "current-frame signed tetrahedral volumes, C(8,4)=70 padded"},
                {"block": "cur_improper_dihedrals", "start": 184, "stop": 254,
                 "desc": "current-frame improper dihedrals, 70 padded"},
                {"block": "goal_signed_volumes", "start": 254, "stop": 324,
                 "desc": "goal-frame signed tetrahedral volumes, 70 padded"},
                {"block": "goal_improper_dihedrals", "start": 324, "stop": 394,
                 "desc": "goal-frame improper dihedrals, 70 padded"},
            ],
        },
        "P2": {
            "purpose": "label oracle — pipeline self-check (AUPRC ≥ 0.99 or "
                       "the probe pipeline is buggy)",
            "goal_info": "goal K distances + true d_end",
            "n_features": 115,
            "columns": p1_cols + [
                {"block": "d_end_angstrom", "start": 114, "stop": 115,
                 "desc": "true d_end = RMSD_K(current, goal endpoint), proper-rotation Kabsch"},
            ],
        },
        "P3": {
            "purpose": "topology-only reaction-identity features — leak/memory "
                       "check; NO coordinates of any frame, NO intent_id/curve_id "
                       "one-hot (curve-grouped splits make those unlearnable and "
                       "not the point)",
            "goal_info": "none (graph topology of g_src/g_tgt only)",
            "n_features": P3_N_FEATURES,
            "columns": [
                {"block": "n_atoms", "start": 0, "stop": 1, "desc": "graph atom count"},
                {"block": "n_bonds_src", "start": 1, "stop": 2, "desc": "|E(G_src)|"},
                {"block": "n_bonds_tgt", "start": 2, "stop": 3, "desc": "|E(G_tgt)|"},
                {"block": "bond_order_hist_src", "start": 3, "stop": 8,
                 "desc": f"counts by order {list(P3_ORDER_BINS)} + other"},
                {"block": "bond_order_hist_tgt", "start": 8, "stop": 13,
                 "desc": f"counts by order {list(P3_ORDER_BINS)} + other"},
                {"block": "n_changed_bonds", "start": 13, "stop": 14,
                 "desc": "formed + broken + order-changed pairs"},
                {"block": "n_cv_pairs", "start": 14, "stop": 15,
                 "desc": "len(cv_coordinates)"},
                {"block": "abs_delta_b_mean", "start": 15, "stop": 16,
                 "desc": "mean |ΔB| over union bonds"},
                {"block": "abs_delta_b_max", "start": 16, "stop": 17,
                 "desc": "max |ΔB| over union bonds"},
                {"block": "abs_delta_b_sum", "start": 17, "stop": 18,
                 "desc": "sum |ΔB| over union bonds"},
                {"block": "n_nonzero_delta_b", "start": 18, "stop": 19,
                 "desc": "count of pairs with ΔB ≠ 0"},
                {"block": "direction_onehot", "start": 19, "stop": 21,
                 "desc": "intent direction one-hot [forward, reverse]"},
                {"block": "k_size", "start": 21, "stop": 22,
                 "desc": "|K| key-atom set size"},
            ],
        },
        "P4": {
            "purpose": "full legal single-frame geometry (NO goal, NO d_end, NO "
                       "history/energy/gradient) — is 28 distances too weak?",
            "goal_info": "none",
            "n_features": 297,
            "columns": [
                {"block": "k_pair_cur", "start": 0, "stop": 28,
                 "desc": "current-frame K pair distances, padded to 28"},
                {"block": "k_angles", "start": 28, "stop": 84,
                 "desc": "C(8,3)=56 K angles, padded"},
                {"block": "k_dihedrals", "start": 84, "stop": 154,
                 "desc": "C(8,4)=70 K dihedrals, padded"},
                {"block": "signed_volumes", "start": 154, "stop": 224,
                 "desc": "70 signed tetrahedral volumes (chirality), padded"},
                {"block": "cis_trans_flags", "start": 224, "stop": 294,
                 "desc": "70 cis/trans planarity signs, padded"},
                {"block": "k_size", "start": 294, "stop": 295,
                 "desc": "|K| scalar"},
                {"block": "direction_onehot", "start": 295, "stop": 297,
                 "desc": "intent direction one-hot [forward, reverse]"},
            ],
        },
    }


def build_probe_spec(*, hashes: dict[str, Any], n_bootstrap: int,
                     d1_pack_path: str) -> dict[str, Any]:
    """PRE-REGISTERED probe spec (pure — deterministic for fixed inputs)."""
    layout = feature_layout()
    probes: dict[str, Any] = {}
    for name in PROBE_ORDER:
        entry = dict(layout[name])
        entry["models"] = ["hgb_primary", "logistic", "mlp32"]
        probes[name] = entry
    return {
        "schema": SPEC_SCHEMA,
        "stage": "V5.2-R D2 phase 1 — input-sufficiency probes P0–P4 (CPU, "
                 "frozen R3 encoder; P5 pending phase 2)",
        "plan": "docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md §4–§5",
        "hashes": hashes,
        "seeds": {
            "encoder_seeds": list(SEEDS),
            "probe_model_seed": PROBE_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inner_split_seed": PROBE_SEED,
        },
        "fold_protocol": {
            "dev_folds": list(DEV_FOLDS),
            "sealed_folds": list(SEALED_FOLDS),
            "per_unit": "train = other two dev folds, eval = current fold; "
                        "curve-grouped (every row of a curve stays on one side)",
            "inner_split": "GroupShuffleSplit(n_splits=1, test_size=0.25, "
                           "random_state=0) on TRAIN curves, used ONLY for the "
                           "pre-registered threshold",
            "units": "3 folds × 3 seeds = 9 units per probe; aggregate = "
                     "median [min, max]",
            "block_probe_fit_note": "block features + pinned random_state=0 are "
                                    "seed-independent → fitted once per fold, "
                                    "identical metrics replicate across the 3 "
                                    "seed units (equivalent to 9 fits)",
        },
        "threshold_rule": (
            f"smallest threshold with precision ≥ {PRECISION_TARGET:.2f} on the "
            "curve-grouped inner split of TRAIN; if unreachable → thr=∅ "
            "(predict-nothing); never tuned on eval"),
        "probes": probes,
        "models": {
            "hgb_primary": "HistGradientBoostingClassifier(class_weight="
                           "'balanced', random_state=0)",
            "logistic": "StandardScaler → LogisticRegression(C=1.0, penalty='l2', "
                        "max_iter=2000, random_state=0)",
            "mlp32": "StandardScaler → MLPClassifier(hidden_layer_sizes=(32,), "
                     f"early_stopping=True, max_iter=300, random_state=0); "
                     f"GLOBAL budget {MLP_BUDGET_S:.0f} s shared by every MLP fit",
            "constant": "train-prevalence baseline (P0 reference only)",
        },
        "encoder_variants": {
            "inputs": "frozen R3 embeddings from the D1 cache (z 64-d, "
                      "key-pooled h 48-d; no torch, no encoder training)",
            "logistic_z": "D1-verbatim: eval scores from the C=1.0 fit; "
                          "threshold from the _select_c (C ∈ "
                          f"{list(d1.C_GRID)}) best-C inner fit",
            "logistic_h": "D1-verbatim: C=1.0 for scores and inner threshold",
            "logistic_zh": "z⊕h concat (112-d), logistic_z recipe "
                           "(C=1.0 scores, _select_c inner threshold)",
            "mlp_z": "MLP(32) on z; budget shared with block MLPs",
            "scope": "encoder inputs are probe-independent (current frame only) "
                     "→ computed once per (fold, seed) unit and reported under "
                     "the enc/* keys; P0 and P4 reference them as their z/h "
                     "variants",
        },
        "mme_gates": {
            "P1-P0": {"effect": f"ΔAUPRC ≥ +{MME_P1:.2f}",
                      "ci": "paired curve-cluster bootstrap 95% CI lower bound > 0",
                      "point_estimator": "median of paired per-fold-seed ΔAUPRC "
                                         "(primary HGB)",
                      "status": "computable_phase1"},
            "P4-P0": {"effect": f"ΔAUPRC ≥ +{MME_P4:.2f}",
                      "ci": "paired curve-cluster bootstrap 95% CI lower bound > 0",
                      "point_estimator": "median of paired per-fold-seed ΔAUPRC "
                                         "(primary HGB)",
                      "status": "computable_phase1"},
            "P5-P0": {"effect": f"ΔAUPRC ≥ +{MME_P5:.2f}",
                      "ci": "paired curve-cluster bootstrap 95% CI lower bound > 0",
                      "status": "PENDING — phase 2; P5 未启动——待 D0.1 异常清零 "
                                "(plan §3.1); another lane owns P5"},
            "P2_self_check": {"rule": f"P2 aggregate median AUPRC (primary HGB) "
                                      f"≥ {P2_SELF_CHECK_MIN:.2f}; else probe "
                                      "pipeline bug → stop all interpretation"},
            "P3_leak_rule": {"rule": f"P3 ≥ P4 − {P3_LEAK_MARGIN:.2f} (aggregate "
                                     "medians, primary HGB) → reaction-identity "
                                     "leak; P4 interpretation void"},
        },
        "exploratory": ["P1c-P0", "P3-P0"],
        "bootstrap": {"method": "curve cluster bootstrap (resample curves with "
                                "replacement, pooled-row AP)",
                      "n_resamples": int(n_bootstrap),
                      "seed": BOOTSTRAP_SEED,
                      "comparisons": ["P1-P0", "P4-P0", "P5-P0",
                                      "P1c-P0", "P3-P0"]},
        "sealed_firewall": "folds 3/4 are COUNTED ONLY — never encoded, never "
                           "featurized, never probed; every feature matrix is "
                           "built over dev positions only and the run asserts "
                           "fold ∈ {0,1,2} before fitting. No goal coordinates "
                           "in P0/P3/P4 forward features (frame-guard assertion "
                           "at build time).",
        "p5_status": {"state": "pending_phase2",
                      "stub_flag": "--p5 exits without running",
                      "note": "P5 未启动——待 D0.1 异常清零 (plan §3.1)"},
        "d1_reference_pack": d1_pack_path,
        "d1_reference_note": (
            "hard reproduction criterion = in-process re-run of D1's own "
            "probe runner (v52r_d1_probe.run_endpoint_probes) on identical "
            "features/labels/splits; the stored d1/evidence_pack.json is only "
            "an environment-drift audit (it may predate the current sklearn "
            "build)"),
        "runtime_budget_s": {"mlp_global": MLP_BUDGET_S, "total_target": 600},
    }


def spec_text(spec: dict[str, Any]) -> str:
    """Canonical on-disk text of the spec (sort_keys + indent=2, + newline)."""
    return json.dumps(spec, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def spec_sha256(spec: dict[str, Any]) -> str:
    """sha256 of the canonical spec text == sha256 of probe_spec.json bytes."""
    return hashlib.sha256(spec_text(spec).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# feature construction (goal-free guard for P0/P4)
# ---------------------------------------------------------------------------


class _FrameRecorder:
    """Wraps ``dataset.frame_xyz`` and records every (curve, node) request.

    P0/P4 are GOAL-FREE probes: :func:`assert_goal_free` proves that a
    builder only ever asked for CURRENT frames (a goal-frame fetch would
    record a node outside the allowed current-node set and raise).
    """

    def __init__(self, dataset: KeyAtomDatasetV3) -> None:
        self._dataset = dataset
        self.requests: list[tuple[str, int]] = []

    def frame_xyz(self, curve_id: str, node_index: int) -> tuple:
        self.requests.append((str(curve_id), int(node_index)))
        return self._dataset.frame_xyz(curve_id, node_index)


def assert_goal_free(recorder: _FrameRecorder,
                     allowed_current: set[tuple[str, int]],
                     probe_name: str) -> None:
    """Runtime sanity: a goal-free builder touched only current frames."""
    bad = sorted({r for r in recorder.requests if r not in allowed_current})
    if bad:
        raise AssertionError(
            f"{probe_name}: goal-free feature builder requested "
            f"{len(bad)} non-current frame(s), e.g. {bad[:3]} — "
            "goal coordinates must never enter P0/P4 forward features")


def p3_topology_features(intent, *, _frame_xyz: Any = None) -> np.ndarray:
    """P3 topology-only descriptor (22) of a Reaction Intent — NO coordinates.

    ``_frame_xyz`` is ACCEPTED BUT MUST BE IGNORED (negative test hook: the
    D2 test suite feeds shuffled coordinates and asserts identical output).
    Only ``g_src``/``g_tgt`` bonds, CV pairs, direction and |K| enter; there
    is deliberately no intent_id/curve_id one-hot.
    """
    g_src = tuple(intent.g_src_bonds)
    g_tgt = tuple(intent.g_tgt_bonds)
    orders: dict[tuple[int, int], tuple[float, float]] = {}
    for i, j, o in g_src:
        orders[(i, j)] = (float(o), 0.0)
    for i, j, o in g_tgt:
        b_src, _ = orders.get((i, j), (0.0, 0.0))
        orders[(i, j)] = (b_src, float(o))

    def _hist(bonds: tuple[tuple[int, int, float], ...]) -> list[float]:
        counts = [0.0] * (len(P3_ORDER_BINS) + 1)
        for _i, _j, o in bonds:
            try:
                idx = P3_ORDER_BINS.index(float(o))
            except ValueError:
                idx = len(P3_ORDER_BINS)
            counts[idx] += 1.0
        return counts

    deltas = [abs(b_tgt - b_src) for (b_src, b_tgt) in orders.values()]
    n_changed = sum(1 for (b_src, b_tgt) in orders.values() if b_src != b_tgt)
    if intent.g_atoms:
        n_atoms = float(len(intent.g_atoms))
    else:
        atoms = [a for (i, j) in orders for a in (i, j)]
        n_atoms = float(max(atoms) + 1) if atoms else 0.0
    k_size = float(len(intent.key_rows()))
    feats = [
        n_atoms,
        float(len(g_src)),
        float(len(g_tgt)),
        *_hist(g_src),
        *_hist(g_tgt),
        float(n_changed),
        float(len(intent.cv_coordinates)),
        float(np.mean(deltas)) if deltas else 0.0,
        float(max(deltas)) if deltas else 0.0,
        float(sum(deltas)),
        float(sum(1 for d in deltas if d != 0.0)),
        1.0 if intent.direction == "forward" else 0.0,
        1.0 if intent.direction == "reverse" else 0.0,
        k_size,
    ]
    if len(feats) != P3_N_FEATURES:
        raise AssertionError(f"P3 feature width drifted: {len(feats)}")
    return np.asarray(feats, dtype=np.float64)


def build_probe_blocks(dataset: KeyAtomDatasetV3,
                       dev_positions: Sequence[int],
                       labels: EndpointStopLabelTable,
                       ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build every probe matrix over dev rows (sealed rows never enter).

    Returns ``(blocks, info)`` with blocks keyed ``P0..P4`` — all
    ``[n_dev, n_features]`` float64.  Non-finite values (degenerate geometry)
    are counted and replaced by 0.0.  P0/P4 go through the goal-free frame
    guard; P1/P1c/P2 may request exactly the label's target endpoint frame.
    """
    n = len(dev_positions)
    samples = dataset.samples
    layout = feature_layout()
    p1 = np.zeros((n, layout["P1"]["n_features"]), dtype=np.float64)
    p1c = np.zeros((n, layout["P1c"]["n_features"]), dtype=np.float64)
    p2 = np.zeros((n, layout["P2"]["n_features"]), dtype=np.float64)
    p3 = np.zeros((n, layout["P3"]["n_features"]), dtype=np.float64)
    p4 = np.zeros((n, layout["P4"]["n_features"]), dtype=np.float64)

    cur_guard = _FrameRecorder(dataset)
    goal_guard = _FrameRecorder(dataset)
    current_pairs: set[tuple[str, int]] = set()
    max_k = 0
    for pos in dev_positions:
        rec = samples[int(pos)]
        current_pairs.add((str(rec.curve_id), int(rec.node_index)))

    for i, pos in enumerate(dev_positions):
        rec = samples[int(pos)]
        lab = labels.by_sample[rec.sample_id]
        keys = rec.intent.key_rows()
        max_k = max(max_k, len(keys))
        cur = np.asarray(
            cur_guard.frame_xyz(rec.curve_id, rec.node_index), dtype=np.float64)
        kb = k_feature_blocks(cur, keys)
        if len(keys) > MAX_K:
            raise AssertionError(f"|K|={len(keys)} exceeds frozen MAX_K={MAX_K}")

        # P4 (goal-free — frame guard enforces it)
        p4[i, 0:28] = kb["pairs"]
        p4[i, 28:84] = kb["angles"]
        p4[i, 84:154] = kb["dihedrals"]
        p4[i, 154:224] = kb["volumes"]
        p4[i, 224:294] = kb["cis_trans"]
        p4[i, 294] = float(len(keys))
        p4[i, 295 + (0 if rec.intent.direction == "forward" else 1)] = 1.0

        # P1 = P0 layout + goal blocks
        p1[i, 0:28] = kb["pairs"]
        p1[i, 28 + (0 if rec.intent.direction == "forward" else 1)] = 1.0
        goal = np.asarray(
            goal_guard.frame_xyz(rec.curve_id, lab.target_endpoint_node),
            dtype=np.float64)
        gkb = k_feature_blocks(goal, keys)
        p1[i, 30:58] = gkb["pairs"]
        p1[i, 58:86] = kb["pairs"] - gkb["pairs"]
        p1[i, 86:114] = np.abs(kb["pairs"] - gkb["pairs"])

        # P1c = P1 ⊕ chirality of current AND goal frames
        p1c[i, 0:114] = p1[i]
        p1c[i, 114:184] = kb["volumes"]
        p1c[i, 184:254] = kb["impropers"]
        p1c[i, 254:324] = gkb["volumes"]
        p1c[i, 324:394] = gkb["impropers"]

        # P2 = P1 ⊕ true d_end
        p2[i, 0:114] = p1[i]
        p2[i, 114] = float(lab.d_end_angstrom)

        # P3 (no dataset access at all — coordinate-free by construction)
        p3[i] = p3_topology_features(rec.intent)

    # --- goal-free assertions (P0/P4) ---------------------------------------
    assert_goal_free(cur_guard, current_pairs, "P4")
    if not set(goal_guard.requests) <= {
            (str(samples[int(p)].curve_id),
             int(labels.by_sample[samples[int(p)].sample_id].target_endpoint_node))
            for p in dev_positions}:
        raise AssertionError("goal guard recorded an unexpected goal request")

    # P0 via D1 (exact feature parity) + bitwise equality with the guarded
    # current-frame pair block proves P0 never sees goal coordinates.
    p0, p0_info = d1.build_tabular(dataset, dev_positions)
    if p0.shape != (n, MAX_K_PAIRS + 2):
        raise AssertionError(
            f"P0 block shape {p0.shape} != {(n, MAX_K_PAIRS + 2)} — the frozen "
            "D1 tabular width assumption (|K| ≤ 8) is violated")
    if not np.array_equal(p4[:, :MAX_K_PAIRS], p0[:, :MAX_K_PAIRS]):
        diff = int((p4[:, :MAX_K_PAIRS] != p0[:, :MAX_K_PAIRS]).sum())
        raise AssertionError(
            f"P0 vs P4 pair-block mismatch in {diff} cell(s) — "
            "P0 must be D1-identical AND goal-free")
    if not np.array_equal(p1[:, :28], p0[:, :28]):
        raise AssertionError("P1 current-distance prefix drifted from P0")

    blocks = {"P0": p0, "P1": p1, "P1c": p1c, "P2": p2, "P3": p3, "P4": p4}
    nonfinite: dict[str, int] = {}
    for key, mat in blocks.items():
        bad = int((~np.isfinite(mat)).sum())
        if bad:
            nonfinite[key] = bad
            blocks[key] = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    info = {
        "n_rows": int(n),
        "max_k": int(max_k),
        "per_probe": {
            key: {"n_features": int(mat.shape[1]),
                  "composition": _composition_text(key, layout[key]),
                  "columns": layout[key]["columns"],
                  "goal_info": layout[key]["goal_info"],
                  "purpose": layout[key]["purpose"]}
            for key, mat in blocks.items()},
        "nonfinite_replaced": nonfinite,
        "p0_pair_block_bitwise_equal_to_p4_guarded": True,
        "p0_info_d1": p0_info,
        "goal_free_probes": list(GOAL_FREE_PROBES),
        "goal_free_assertion": "cur-frame guard recorded only current "
                               "(curve, node) requests during P0/P4 build; "
                               "P0 pairs verified bitwise-equal to guarded P4 pairs",
    }
    return blocks, info


def _composition_text(key: str, entry: dict[str, Any]) -> str:
    parts = [f"{c['block']}[{c['start']}:{c['stop']}]" for c in entry["columns"]]
    return (f"{entry['n_features']} features = " + " + ".join(parts) +
            f" | goal: {entry['goal_info']}")


# ---------------------------------------------------------------------------
# probing (9 units per probe)
# ---------------------------------------------------------------------------


def _start_and_ts_rows(labels: EndpointStopLabelTable,
                       eval_sids: Sequence[str],
                       ) -> tuple[list[str], list[str], list[str]]:
    start_samples: list[str] = []
    ts_samples: list[str] = []
    ts_deg: list[str] = []
    for sid in eval_sids:
        lab = labels.by_sample[sid]
        csum = labels.curves[lab.curve_id]
        node = int(lab.node_index)
        if ((lab.direction == "forward" and node == csum.endpoint_a_node)
                or (lab.direction == "reverse"
                    and node == csum.endpoint_b_node)):
            start_samples.append(sid)
        if node == csum.ts_index:
            if lab.curve_id in DEGENERATE_TS_CURVES:
                ts_deg.append(sid)
            else:
                ts_samples.append(sid)
    return start_samples, ts_samples, ts_deg


def _unit_metrics(scores_usable: np.ndarray, y_ev: np.ndarray,
                  thr: float | None, thr_info: dict[str, Any],
                  scores_all: np.ndarray, eval_sids: Sequence[str],
                  labels: EndpointStopLabelTable,
                  start_samples: Sequence[str], ts_samples: Sequence[str],
                  ts_deg: Sequence[str]) -> dict[str, Any]:
    m = d1._classification_metrics(scores_usable, y_ev, thr)
    m["threshold_selection"] = thr_info
    m["false_stop"] = d1._false_stop_rates(
        {"sample_ids": list(eval_sids)}, scores_all, thr, labels,
        start_samples, ts_samples, ts_deg)
    return m


def _fit_encoder_variant(variant: str, x_tr: np.ndarray, y_tr: np.ndarray,
                         x_ev: np.ndarray, groups_tr: np.ndarray,
                         mlp_state: dict[str, float],
                         ) -> tuple[np.ndarray | None, float | None,
                                    dict[str, Any]]:
    """Frozen-encoder probe variant with D1's exact C-grid recipe."""
    if variant == "mlp_z":
        if mlp_state["spent"] > MLP_BUDGET_S:
            return None, None, {"skipped": True,
                                "reason": f"MLP time budget {MLP_BUDGET_S}s "
                                          "exhausted"}
        t0 = time.time()
        score_fn, _ = d1._fit_probe("mlp", x_tr, y_tr)
        thr, info = d1._inner_split_threshold(x_tr, y_tr, groups_tr, "mlp", None)
        mlp_state["spent"] += time.time() - t0
        return np.asarray(score_fn(x_ev), dtype=np.float64), thr, info
    if variant == "logistic_h":
        # D1-verbatim: C=1.0 for scores AND inner threshold.
        score_fn, _ = d1._fit_probe("logistic", x_tr, y_tr)
        thr, info = d1._inner_split_threshold(
            x_tr, y_tr, groups_tr, "logistic", None)
        info["best_C"] = None
        info["recipe"] = "D1-verbatim (C=1.0 scores + C=1.0 inner threshold)"
    else:
        # logistic_z / logistic_zh: D1's z recipe — C=1.0 eval scores,
        # _select_c best-C inner threshold.
        c_value = d1._select_c(x_tr, y_tr, groups_tr)
        score_fn, _ = d1._fit_probe("logistic", x_tr, y_tr)
        thr, info = d1._inner_split_threshold(
            x_tr, y_tr, groups_tr, "logistic_z", c_value)
        info["best_C"] = c_value
        info["recipe"] = "D1-verbatim (C=1.0 scores, _select_c best-C threshold)"
    return np.asarray(score_fn(x_ev), dtype=np.float64), thr, info


def run_probe_units(blocks: dict[str, np.ndarray],
                    labels: EndpointStopLabelTable,
                    meta: dict[str, np.ndarray],
                    z_by_unit: dict[tuple[int, int], np.ndarray],
                    h_by_unit: dict[tuple[int, int], np.ndarray],
                    mlp_state: dict[str, float],
                    probe_names: Sequence[str] = PROBE_ORDER,
                    with_encoder: bool = True,
                    ) -> tuple[list[dict[str, Any]],
                               dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Run the 9 (fold × seed) units; returns (units, pooled primary scores).

    Pooled primary scores are per probe: (scores, labels, curves) over the
    USABLE rows of every eval fold — the input to the curve-cluster
    bootstrap and the per-curve scatter table.  ``probe_names`` restricts
    the block probes (D2 phase 2 runs one config per label scheme);
    ``with_encoder=False`` skips the enc/* variants.
    """
    folds = meta["folds"]
    sample_ids = meta["sample_ids"]
    curve_ids = meta["curve_ids"]
    lab_vec = np.array([labels.by_sample[sid].endpoint_stop_label
                        for sid in sample_ids], dtype=np.int64)
    deg_vec = np.array([bool(labels.by_sample[sid].degenerate_flag)
                        for sid in sample_ids], dtype=np.bool_)
    usable = (lab_vec >= 0) & (~deg_vec)

    units: list[dict[str, Any]] = []
    pooled_parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = \
        {probe: [] for probe in probe_names}

    for fold in DEV_FOLDS:
        train_mask = (folds != fold) & usable
        eval_mask = (folds == fold)
        eval_pos = np.flatnonzero(eval_mask)
        usable_local = usable[eval_pos]
        eval_sids = [str(s) for s in sample_ids[eval_mask]]
        start_samples, ts_samples, ts_deg = _start_and_ts_rows(
            labels, eval_sids)
        y_tr = lab_vec[train_mask]
        groups_tr = curve_ids[train_mask]
        y_ev = lab_vec[eval_mask][usable_local]
        x_eval_all = int(eval_mask.sum())

        fold_probes: dict[str, dict[str, Any]] = {}
        fold_scores: dict[str, np.ndarray] = {}

        # constant baseline (P0 reference)
        prev = float(y_tr.mean()) if y_tr.size else 0.0
        score_all = np.full(x_eval_all, prev, dtype=np.float64)
        fold_probes["P0/constant"] = _unit_metrics(
            score_all[usable_local], y_ev, None,
            {"threshold": None, "reason": "constant baseline (no threshold)"},
            score_all, eval_sids, labels, start_samples, ts_samples, ts_deg)

        # block probes — fitted once per fold (seed-independent, pinned seed 0)
        for probe in probe_names:
            x = blocks[probe]
            x_tr = x[train_mask]
            x_ev = x[eval_mask]
            for model in ("hgb", "logistic", "mlp"):
                key = f"{probe}/{model}"
                if model == "mlp" and mlp_state["spent"] > MLP_BUDGET_S:
                    fold_probes[key] = {
                        "skipped": True,
                        "reason": f"MLP time budget {MLP_BUDGET_S}s exhausted"}
                    continue
                t0 = time.time()
                score_fn, _ = d1._fit_probe(model, x_tr, y_tr)
                thr, thr_info = d1._inner_split_threshold(
                    x_tr, y_tr, groups_tr, model, None)
                if model == "mlp":
                    mlp_state["spent"] += time.time() - t0
                score_all = np.asarray(score_fn(x_ev), dtype=np.float64)
                fold_probes[key] = _unit_metrics(
                    score_all[usable_local], y_ev, thr, thr_info, score_all,
                    eval_sids, labels, start_samples, ts_samples, ts_deg)
                fold_scores[key] = score_all

        for probe in probe_names:
            score_all = fold_scores.get(f"{probe}/hgb")
            if score_all is None:
                continue
            pooled_parts[probe].append(
                (score_all[usable_local],
                 lab_vec[eval_mask][usable_local],
                 curve_ids[eval_mask][usable_local]))

        # encoder variants — per (fold, seed), z/h differ across seeds
        for seed in SEEDS:
            z = z_by_unit[(fold, seed)]
            h = h_by_unit[(fold, seed)]
            zh = np.concatenate([z, h], axis=1)
            enc_probes: dict[str, dict[str, Any]] = {}
            if with_encoder:
                for variant, arr in (("logistic_z", z), ("logistic_h", h),
                                     ("logistic_zh", zh), ("mlp_z", z)):
                    scores, thr, info = _fit_encoder_variant(
                        variant, arr[train_mask], y_tr, arr[eval_mask],
                        groups_tr, mlp_state)
                    key = f"enc/{variant}"
                    if scores is None:
                        enc_probes[key] = dict(info)
                        continue
                    enc_probes[key] = _unit_metrics(
                        scores[usable_local], y_ev, thr, info, scores, eval_sids,
                        labels, start_samples, ts_samples, ts_deg)

            units.append({
                "fold": int(fold), "seed": int(seed),
                "n_train": int(train_mask.sum()), "n_eval": int(y_ev.size),
                "train_prevalence": d1._r6(y_tr.mean()) if y_tr.size else None,
                "eval_prevalence": d1._r6(y_ev.mean()) if y_ev.size else None,
                "probes": {**fold_probes, **enc_probes},
            })
    pooled: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        probe: (np.concatenate([p[0] for p in parts]),
                np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]))
        for probe, parts in pooled_parts.items() if parts}
    return units, pooled


def aggregate_units(units: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Median + [min, max] over the 9 units, per probe/model key."""
    keys: list[str] = []
    for u in units:
        for k in u["probes"]:
            if k not in keys:
                keys.append(k)
    out: dict[str, Any] = {}
    for key in keys:
        rows = [u["probes"][key] for u in units
                if key in u["probes"] and not u["probes"][key].get("skipped")]
        if not rows:
            out[key] = {"skipped": True}
            continue
        out[key] = {
            "n_units": len(rows),
            "auroc": d1._minmax(r["auroc"] for r in rows),
            "auprc": d1._minmax(r["auprc"] for r in rows),
            "eval_prevalence": d1._minmax(r["prevalence"] for r in rows),
            "precision_at_threshold": d1._minmax(
                r["precision_at_threshold"] for r in rows),
            "recall_at_threshold": d1._minmax(
                r["recall_at_threshold"] for r in rows),
            "threshold": d1._minmax(r["threshold"] for r in rows),
            "start_fp_rate": d1._minmax(
                r["false_stop"]["start"]["rate"] for r in rows),
            "ts_fp_rate": d1._minmax(
                r["false_stop"]["ts"]["rate"] for r in rows),
            "units_missing_threshold": sum(
                1 for r in rows if r["threshold"] is None),
        }
    return out


def paired_delta_summary(units: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Paired per-fold-seed ΔAUPRC vs P0 (primary HGB) for every probe."""
    out: dict[str, Any] = {}
    for probe in PROBE_ORDER:
        if probe == "P0":
            continue
        per: list[dict[str, Any]] = []
        for u in units:
            a = u["probes"].get(f"{probe}/hgb", {}).get("auprc")
            b = u["probes"].get("P0/hgb", {}).get("auprc")
            if a is None or b is None:
                continue
            per.append({"fold": u["fold"], "seed": u["seed"],
                        "delta": d1._r6(float(a) - float(b))})
        vals = [p["delta"] for p in per]
        out[probe] = {
            "comparison": f"{probe}-P0",
            "n_units": len(vals),
            "median": d1._r6(np.median(vals)) if vals else None,
            "min": d1._r6(min(vals)) if vals else None,
            "max": d1._r6(max(vals)) if vals else None,
            "units": per,
        }
    return out


def paired_cluster_bootstrap_auprc(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray,
    curve_ids: np.ndarray, *, labels_b: np.ndarray | None = None,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired curve-cluster bootstrap of ΔAUPRC = AP(A) − AP(B).

    The SAME curve resample is applied to both score vectors each round, so
    the difference is paired; deterministic for a fixed seed.  ``labels_b``
    supports CROSS-SCHEME comparisons (D2 phase 2: P5 scores evaluated on
    basin labels vs P0 scores on d_end labels over the same intersection
    rows) — each side is scored against its OWN label vector; a resample is
    skipped when either side is one-class.  Returns the full-data point Δ,
    the percentile 95% CI, and resample bookkeeping.
    """
    from sklearn.metrics import average_precision_score

    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    labels_b_vec = labels if labels_b is None else np.asarray(labels_b, dtype=np.int64)
    curve_ids = np.asarray(curve_ids, dtype=object)
    if not (scores_a.size == scores_b.size == labels.size
            == labels_b_vec.size == curve_ids.size):
        raise ValueError("paired bootstrap: score/label/curve lengths differ")
    curves = np.array(sorted(set(curve_ids.tolist())))
    point_a = float(average_precision_score(labels, scores_a)) if labels.size else None
    point_b = float(average_precision_score(labels_b_vec, scores_b)) \
        if labels_b_vec.size else None
    rng = np.random.RandomState(int(seed))
    deltas: list[float] = []
    for _ in range(int(n_resamples)):
        picked = rng.choice(curves, size=curves.size, replace=True)
        parts = [np.flatnonzero(curve_ids == c) for c in picked]
        idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        if idx.size == 0:
            continue
        y_a = labels[idx]
        y_b = labels_b_vec[idx]
        if y_a.min() == y_a.max() or y_b.min() == y_b.max():
            continue               # AP undefined for a one-class resample
        deltas.append(float(average_precision_score(y_a, scores_a[idx])
                            - average_precision_score(y_b, scores_b[idx])))
    if not deltas:
        ci = {"lower": None, "upper": None}
    else:
        ci = {"lower": d1._r6(np.percentile(deltas, 2.5)),
              "upper": d1._r6(np.percentile(deltas, 97.5))}
    point = (d1._r6(point_a - point_b)
             if point_a is not None and point_b is not None else None)
    return {
        "point": point,
        "point_a": d1._r6(point_a),
        "point_b": d1._r6(point_b),
        "prevalence": d1._r6(labels.mean()) if labels.size else None,
        "prevalence_b": d1._r6(labels_b_vec.mean())
        if labels_b_vec.size else None,
        "cross_scheme": labels_b is not None,
        "ci95": ci,
        "n_resamples": int(n_resamples),
        "n_valid_resamples": len(deltas),
        "n_curves": int(curves.size),
        "seed": int(seed),
    }


def per_curve_auprc_table(
    pooled: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, dict[str, float | None]]:
    """Per dev curve × probe AUPRC on pooled eval scores (None when not computable)."""
    from sklearn.metrics import average_precision_score

    out: dict[str, dict[str, float | None]] = {}
    for probe in PROBE_ORDER:
        if probe not in pooled:
            continue
        scores, labels, curves = pooled[probe]
        for cid in sorted(set(curves.tolist())):
            mask = curves == cid
            y = labels[mask]
            if y.size == 0 or y.min() == y.max():
                value = None
            else:
                value = d1._r6(float(average_precision_score(y, scores[mask])))
            out.setdefault(str(cid), {})[probe] = value
    return out


# ---------------------------------------------------------------------------
# P0 ↔ D1 reproduction check (pure)
# ---------------------------------------------------------------------------


def _same_num(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


_COMPARE_FIELDS = ("auprc", "auroc", "prevalence", "threshold",
                   "precision_at_threshold", "recall_at_threshold")
_COMPARE_FALSE_STOP = ("start", "ts")


def _compare_units(my_units: Sequence[dict[str, Any]],
                   d1_units: Sequence[dict[str, Any]],
                   *, tol: float) -> dict[str, Any]:
    """Per-(fold, seed) metric comparison of my units vs a D1 units list."""
    mine = {(u["fold"], u["seed"]): u for u in my_units}
    per_probe: dict[str, float] = {}
    mismatches: list[dict[str, Any]] = []
    n_compared = 0
    max_diff: float | None = None
    for du in d1_units:
        unit = mine.get((du["fold"], du["seed"]))
        for d1_key, my_key in D1_PROBE_MAP.items():
            dm = du.get("probes", {}).get(d1_key)
            if dm is None or dm.get("skipped"):
                continue
            mm = (unit or {}).get("probes", {}).get(my_key)
            if mm is None or mm.get("skipped"):
                mismatches.append(
                    {"fold": du["fold"], "seed": du["seed"],
                     "d1_probe": d1_key, "d2_key": my_key,
                     "field": "<missing in D2>"})
                continue
            for field in _COMPARE_FIELDS:
                dv, mv = dm.get(field), mm.get(field)
                if dv is None and mv is None:
                    continue
                n_compared += 1
                if dv is None or mv is None:
                    diff = float("inf")
                else:
                    diff = abs(float(dv) - float(mv))
                    per_probe[d1_key] = max(per_probe.get(d1_key, 0.0), diff)
                    if max_diff is None or diff > max_diff:
                        max_diff = diff
                if not _same_num(dv, mv, tol):
                    mismatches.append(
                        {"fold": du["fold"], "seed": du["seed"],
                         "d1_probe": d1_key, "d2_key": my_key, "field": field,
                         "d1": dv, "d2": mv})
            for kind in _COMPARE_FALSE_STOP:
                dv = dm.get("false_stop", {}).get(kind, {}).get("rate")
                mv = mm.get("false_stop", {}).get(kind, {}).get("rate")
                n_compared += 1
                if dv is None and mv is None:
                    continue
                if dv is None or mv is None:
                    diff = float("inf")
                else:
                    diff = abs(float(dv) - float(mv))
                    per_probe[d1_key] = max(per_probe.get(d1_key, 0.0), diff)
                    if max_diff is None or diff > max_diff:
                        max_diff = diff
                if not _same_num(dv, mv, tol):
                    mismatches.append(
                        {"fold": du["fold"], "seed": du["seed"],
                         "d1_probe": d1_key, "d2_key": my_key,
                         "field": f"false_stop.{kind}.rate", "d1": dv, "d2": mv})
    if max_diff is not None and not np.isfinite(max_diff):
        max_diff = None
    status = "PASS" if (n_compared > 0 and not mismatches) else (
        "SKIPPED" if n_compared == 0 else "FAIL")
    return {
        "status": status,
        "tolerance": tol,
        "n_compared": n_compared,
        "max_abs_diff": d1._r6(max_diff),
        "per_probe_max_abs_diff": {k: d1._r6(v)
                                   for k, v in sorted(per_probe.items())},
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }


def compare_p0_to_d1(my_units: Sequence[dict[str, Any]],
                     d1_units: Sequence[dict[str, Any]] | None,
                     d1_pack: dict[str, Any] | None,
                     *, tol: float = 1e-12,
                     stored_tol: float = P0_REPRO_TOL) -> dict[str, Any]:
    """P0 ↔ D1 reproduction check (two calibers).

    * ``in_process`` (HARD criterion): D1's OWN probe runner
      (``v52r_d1_probe.run_endpoint_probes``) re-run in this process/env on
      the identical features/labels/splits must produce identical metrics.
      This verifies protocol+code-path parity independent of environment
      drift in the stored pack.
    * ``stored_pack`` (audit): the stored ``d1/evidence_pack.json`` may
      predate the current sklearn build — its diff is REPORTED (with an
      environment-drift note) but does not gate.
    """
    in_process = (_compare_units(my_units, d1_units, tol=tol)
                  if d1_units else
                  {"status": "SKIPPED", "reason": "in-process D1 runner not run"})
    stored = (_compare_units(
        my_units,
        (d1_pack or {}).get("endpoint_probes", {}).get("units", []),
        tol=stored_tol) if d1_pack else
        {"status": "SKIPPED", "reason": "D1 evidence pack not found"})
    stored["note"] = (
        "stored D1 pack may predate the current sklearn build; its diff is an "
        "environment-drift AUDIT — the hard criterion is the in-process D1-runner "
        "parity (PASS proves D2 P0 implements the D1 protocol exactly)")
    return {
        "status": in_process["status"],          # gates the phase-1 gate
        "in_process": in_process,
        "stored_pack": stored,
        "hard_criterion": "in-process D1-runner parity (same env, same inputs)",
    }


# ---------------------------------------------------------------------------
# phase-1 gate (reports; never authorizes training)
# ---------------------------------------------------------------------------


def evaluate_phase1_gate(*, p0_check: dict[str, Any], p2_median: float | None,
                         p3_median: float | None, p4_median: float | None,
                         mme_p1: dict[str, Any],
                         mme_p4: dict[str, Any]) -> dict[str, Any]:
    """D2 phase-1 gate: integrity checks FAIL/LEAK are hard; MME are PASS /
    PENDING-GATE; P5 is PENDING (phase 2).  training_authorized is always False.
    """
    checks: list[dict[str, Any]] = [
        {"id": "p0_reproduces_d1",
         "expect": "P0 metrics reproduce D1's probe runner (hard criterion: "
                   "in-process re-run tol 1e-12; stored pack = env-drift audit "
                   f"tol {P0_REPRO_TOL:g})",
         "actual": f"in_process={p0_check.get('in_process', {}).get('status')} "
                   f"(max_diff="
                   f"{p0_check.get('in_process', {}).get('max_abs_diff')}) "
                   f"stored_pack={p0_check.get('stored_pack', {}).get('status')} "
                   f"(max_diff="
                   f"{p0_check.get('stored_pack', {}).get('max_abs_diff')})",
         "status": p0_check.get("status", "SKIPPED")},
    ]
    p2_pass = p2_median is not None and float(p2_median) >= P2_SELF_CHECK_MIN
    checks.append({
        "id": "p2_self_check",
        "expect": f"P2 aggregate median AUPRC ≥ {P2_SELF_CHECK_MIN:.2f} "
                  "(primary HGB) — else probe pipeline bug",
        "actual": f"median={p2_median}",
        "status": "PASS" if p2_pass else "FAIL",
        "pass": bool(p2_pass)})
    if p3_median is None or p4_median is None:
        leak, bound = False, None
    else:
        bound = float(p4_median) - P3_LEAK_MARGIN
        leak = float(p3_median) >= bound
    checks.append({
        "id": "p3_leak_rule",
        "expect": f"P3 < P4 − {P3_LEAK_MARGIN:.2f} (aggregate medians, primary "
                  "HGB) else reaction-identity leak",
        "actual": f"P3={p3_median} P4={p4_median} bound="
                  f"{d1._r6(bound) if bound is not None else None}",
        "status": "LEAK" if leak else "NO_LEAK",
        "pass": not leak})

    def _mme(check_id: str, res: dict[str, Any], effect: float) -> None:
        med = res.get("median_delta")
        ci_lb = res.get("ci_lower")
        ok = (med is not None and ci_lb is not None
              and float(med) >= effect and float(ci_lb) > 0.0)
        checks.append({
            "id": check_id,
            "expect": f"median paired ΔAUPRC ≥ {effect:+.2f} AND paired "
                      "curve-cluster bootstrap 95% CI lower bound > 0",
            "actual": f"median_delta={med} ci95_lb={ci_lb} "
                      f"(pooled Δ={res.get('point')})",
            "status": "PASS" if ok else "PENDING-GATE",
            "pass": bool(ok)})

    _mme("mme_p1_minus_p0", mme_p1, MME_P1)
    _mme("mme_p4_minus_p0", mme_p4, MME_P4)
    checks.append({
        "id": "mme_p5_minus_p0",
        "expect": f"ΔAUPRC ≥ {MME_P5:+.2f} AND paired CI lower bound > 0",
        "actual": "not computed — P5 未启动（待 D0.1 异常清零，另一 lane）",
        "status": "PENDING",
        "pass": None})

    integrity_ids = {"p0_reproduces_d1", "p2_self_check", "p3_leak_rule"}
    hard_fail = any(
        c["status"] in ("FAIL", "LEAK") for c in checks
        if c["id"] in integrity_ids)
    return {
        "gate_id": GATE_ID,
        "checks": checks,
        "verdict": "FAIL" if hard_fail else "PASS",
        "training_authorized": False,
        "authorization_note": (
            "phase 1 only REPORTS; the single GPU entry point is the D2-G "
            "human/manager review (plan §6/§10) — this gate never authorizes "
            "training"),
        "mme_summary": {
            "p1_p0": d1._r6(mme_p1.get("median_delta")),
            "p1_p0_ci95": [mme_p1.get("ci_lower"), mme_p1.get("ci_upper")],
            "p1_status": next(c["status"] for c in checks
                              if c["id"] == "mme_p1_minus_p0"),
            "p4_p0": d1._r6(mme_p4.get("median_delta")),
            "p4_p0_ci95": [mme_p4.get("ci_lower"), mme_p4.get("ci_upper")],
            "p4_status": next(c["status"] for c in checks
                              if c["id"] == "mme_p4_minus_p0"),
            "p5_status": "PENDING",
        },
    }


# ---------------------------------------------------------------------------
# provenance / loading
# ---------------------------------------------------------------------------


def collect_hashes(tables_dir: Path, embeddings: dict[tuple[int, int], Path],
                   ) -> dict[str, Any]:
    data_manifest = tables_dir / DATA_MANIFEST_FILENAME
    dataset_sha = ""
    if data_manifest.is_file():
        doc = json.loads(data_manifest.read_text(encoding="utf-8"))
        dataset_sha = str(doc.get("dataset_sha256", ""))
    hashes: dict[str, Any] = {
        "code_commit": d1.git_head(),
        "data_manifest_sha256": d1.sha256_file(data_manifest)
        if data_manifest.is_file() else "",
        "dataset_sha256": dataset_sha,
        "campaign_config_sha256": d1.sha256_file(CAMPAIGN_CONFIG)
        if CAMPAIGN_CONFIG.is_file() else "",
        "label_builder_sha256": d1.sha256_file(LABEL_BUILDER_SRC)
        if LABEL_BUILDER_SRC.is_file() else "",
        "geometry_stereo_sha256": d1.sha256_file(GEOMETRY_STEREO_SRC)
        if GEOMETRY_STEREO_SRC.is_file() else "",
        "d1_probe_script_sha256": d1.sha256_file(Path(d1.__file__)),
        "d2_probe_script_sha256": d1.sha256_file(Path(__file__)),
        "embeddings_sha256": {
            f"embeddings_{fold}_{seed}.npz": d1.sha256_file(path)
            for (fold, seed), path in sorted(embeddings.items())},
    }
    return hashes


def load_embeddings(emb_dir: Path,
                    meta: dict[str, np.ndarray],
                    ) -> tuple[dict[tuple[int, int], np.ndarray],
                               dict[tuple[int, int], np.ndarray],
                               dict[tuple[int, int], Path]]:
    z_by_unit: dict[tuple[int, int], np.ndarray] = {}
    h_by_unit: dict[tuple[int, int], np.ndarray] = {}
    paths: dict[tuple[int, int], Path] = {}
    for fold in DEV_FOLDS:
        for seed in SEEDS:
            path = emb_dir / f"embeddings_{fold}_{seed}.npz"
            if not path.is_file():
                raise FileNotFoundError(
                    f"--embeddings-dir {emb_dir} lacks {path.name}; the D2 "
                    "phase-1 probes reuse the frozen D1 R3 embedding cache")
            data = np.load(path, allow_pickle=True)
            if list(data["sample_ids"]) != list(meta["sample_ids"]):
                raise RuntimeError(
                    f"{path}: sample_ids misaligned with dataset order")
            z_by_unit[(fold, seed)] = data["z"].astype(np.float64)
            h_by_unit[(fold, seed)] = data["h"].astype(np.float64)
            paths[(fold, seed)] = path
    return z_by_unit, h_by_unit, paths


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

_AGG_KEYS_BASE = [f"enc/{v}" for v in ENCODER_VARIANTS] + ["P0/constant"]


def _agg_keys(probe_list: Sequence[str]) -> list[str]:
    return [f"{p}/{m}" for p in probe_list
            for m in ("hgb", "logistic", "mlp")] + _AGG_KEYS_BASE


def _fmt(d: dict | None) -> str:
    if not d or d.get("median") is None:
        return "—"
    return f"{d['median']} [{d['min']}, {d['max']}]"


def _render_phase2_section(phase2: dict[str, Any],
                           gate: dict[str, Any]) -> list[str]:
    """§9 body of D2_PROBE.md once the P5 phase has run (report-only)."""
    enr = phase2.get("enrichment", {})
    lab = phase2.get("labels", {})
    val = phase2.get("validity", {})
    pr = phase2.get("probe", {})
    isect = pr.get("intersection", {})
    lines: list[str] = []
    a = lines.append
    a(f"> 阶段 2（P5，plan §4.6）· `p5_spec.json` sha256 "
      f"`{(phase2.get('p5_spec_sha256') or '')[:16]}…` · 前置条件: "
      f"{phase2.get('prerequisite', '')}")
    a("")
    a("### 9.1 标签定义（p5_spec.json 逐字冻结）")
    a("")
    a("- **B_target（每 curve-intent，预注册构造顺序）**: (a) IRC 参考端点帧 "
      "`frame_xyz(curve, target_endpoint_node)`；(b) 其图自同构像（由 "
      "`min_sym_rmsd` 内部按目标端点图 K-perms 处理：forward→g_tgt，"
      "reverse→g_src）；(c) xtb GFN-2 受约束优化构象富集——**仅前 10 条 "
      "dev 曲线（按 curve_id 排序）的 forward 目标端点 basin**，硬预算内；"
      "(d) 本数据集无独立 IRC 端点（跳过）。")
    a("- **d_basin(X)** = min over Y∈B_target 的 sym-min proper-rotation "
      "Kabsch RMSD_K（`min_sym_rmsd`）。")
    a("- **三态标签**: `1` ⟺ d_basin ≤ 0.10 Å **且**键编辑完成；`0` ⟺ "
      "d_basin > 0.30 Å **或**（d_basin ≤ 0.10 且编辑未完成）；`Mask` ⟺ "
      "0.10 < d_basin ≤ 0.30 Å。")
    a("- **键编辑完成（D0 逐字拓扑代理）**: 全部 CV 对距离与目标端点帧偏差 "
      "≤ 0.25 Å（固定对应、无自同构）。→ D0.1 的 22 个 T 行在 P5 下为**负例**"
      "（断言见 9.3）。")
    a("")
    a("### 9.2 构象富集与 xtb 预算（QC only；ADR-0002 已删除 pes2ts_core "
      "本地 runner——subprocess 仅在此脚本此用途获豁免，truth_guard 只扫描 "
      "pes2ts_core/bin）")
    a("")
    a(f"- 子集: 前 10 条 dev 曲线 {enr.get('subset_curves')}；"
      f"seeds primary={enr.get('seeds', {}).get('primary')} / flip="
      f"{enr.get('seeds', {}).get('flip')}（预注册 seed 20261021；flip 复核 "
      f"{enr.get('seeds', {}).get('flip')}）")
    a(f"- 硬上限: ≤{enr.get('caps', {}).get('max_calls')} 次 xtb 调用 · "
      f"≤{enr.get('caps', {}).get('max_wall_s')} s 总墙钟 · 每 curve-intent "
      f"接受 ≤{enr.get('caps', {}).get('max_accepted')} 构象、生成尝试 "
      f"≤{enr.get('caps', {}).get('max_attempts')} 次 · 能量窗 "
      f"±{enr.get('caps', {}).get('energy_window_hartree')} Hartree"
      f"（=5 kcal/mol，参考端点 SP 计入预算）")
    a(f"- 实际: **{enr.get('n_calls')} 次调用 / {enr.get('wall_s')} s**"
      f"（SP {enr.get('n_ref_sp')} · 构象 primary {enr.get('n_conformer_primary')} "
      f"· flip {enr.get('n_conformer_flip')} · pre-flight "
      f"{enr.get('n_pre_flight')}）· 中止: {enr.get('aborted')}"
      + (f"（{enr.get('aborted_reason')}）" if enr.get("aborted_reason") else ""))
    a("")
    if enr.get("call_summary"):
        a("| purpose | 次数 | 墙钟 s | 备注 |")
        a("|---|---:|---:|---|")
        for purpose, s in sorted(enr["call_summary"].items()):
            a(f"| {purpose} | {s['n']} | {s['wall_s']} | {s.get('note', '')} |")
        a("")
    a(f"- 每次调用的秒数/exit code/能量逐条记录在 `evidence_pack.json` → "
      f"`phase2.enrichment.per_call_log`（{len(enr.get('per_call_log', []))} 条）。")
    a("")
    a("### 9.3 有效性检查（全部通过才可用该标签）")
    a("")
    a("| 检查 | 期望 | 实测 | 判定 |")
    a("|---|---|---|:--:|")
    for c in val.get("checks", []):
        mark_s = "PASS" if c.get("pass") else "**FAIL**"
        a(f"| {c.get('id')} | {c.get('expect')} | {c.get('actual')} | "
          f"{mark_s} |")
    a("")
    a(f"**有效性汇总: {val.get('summary')}**")
    if val.get("notes"):
        for note in val["notes"]:
            a(f"- {note}")
    a("")
    a("### 9.4 标签人群（pos/neg/mask）")
    a("")
    pop = lab.get("population", {})
    a("| 人群 | 行数 | positive | negative | mask | usable | usable 正例率 |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for name, key in (("basin / 全表（12034，含 sealed——仅计数）", "full_table"),
                      ("basin / dev folds 0–2", "dev"),
                      ("basin / dev usable（探针人群）", "dev_usable"),
                      ("d_end / 全表（参照，D1）", "d_end_full"),
                      ("d_end / dev usable（参照）", "d_end_dev_usable")):
        p = pop.get(key)
        if not p:
            continue
        rate = p.get("usable_positive_rate")
        if rate is None and p.get("positive") is not None and key in (
                "full_table", "dev", "d_end_full"):
            rate = None
        a(f"| {name} | {p.get('n_rows')} | {p.get('positive')} | "
          f"{p.get('negative')} | {p.get('mask')} | {p.get('usable_rows')} | "
          f"{rate if rate is not None else '—'} |")
    a("")
    t_rows = lab.get("t_rows", {})
    a(f"- D0.1 T 行（22 个，audit 分类全为 `T`）: basin 标签分布 → "
      f"neg {t_rows.get('n_negative')} / mask {t_rows.get('n_mask')} / "
      f"pos {t_rows.get('n_positive')}；断言全部为负例 "
      f"**{'成立' if t_rows.get('pass') else '不成立（见 9.3）'}**。")
    flip = lab.get("flip", {})
    a(f"- seed-flip（{flip.get('seed_a')} vs {flip.get('seed_b')}，分母 = "
      f"primary-seed dev usable {flip.get('n_rows')} 行）: 翻转 "
      f"{flip.get('n_flips')} 行 = **{flip.get('flip_rate')}**"
      f"（阈值 < 0.05）· transitions {flip.get('transitions')}")
    a("")
    a("### 9.5 P5 探针结果（P4 特征构建器原样 297 维，frame guard 激活）")
    a("")
    a(f"- 主比较 P5−P0 在**交集人群**（两方案标签均确定、排除退化行）:"
      f" n = {isect.get('n_rows')} 行；basin 正例 {isect.get('pos_basin')} / "
      f"d_end 正例 {isect.get('pos_dend')}；标签一致 {isect.get('agree')} / "
      f"不一致 {isect.get('disagree')}（交集上两方案人群完全相同 → 可配对）。")
    a("- 每单元/聚合表见 §3（P5 行）与下表；配对 Δ 与 CI 见 §4。")
    a("")
    if isect.get("config_aggregates"):
        a("| 交集配置（同 9 单位） | y 标签 | X | HGB AUPRC median [min,max] |")
        a("|---|---|---:|---|")
        for name, s in isect["config_aggregates"].items():
            meta_cfg = (pr.get("configs") or {}).get(name, {})
            hgb = (s or {}).get("P5/hgb") or (s or {}).get("P0/hgb") \
                or (s or {}).get("P4/hgb") or (s or {}).get("P1/hgb") or {}
            aup = hgb.get("auprc", {}) if hgb else {}
            a(f"| {name} | {meta_cfg.get('label_scheme')} | "
              f"{meta_cfg.get('n_features')} | {aup.get('median')} "
              f"[{aup.get('min')}, {aup.get('max')}] |")
        a("")
    enc = pr.get("enc_aggregate") or {}
    if enc:
        a("- enc/* 变体（冻结编码器，basin 人群）聚合 AUPRC: " + "；".join(
            f"{k.split('/', 1)[1]} {_fmt(v.get('auprc'))}"
            for k, v in sorted(enc.items()) if not v.get("skipped")))
        a("")
    p0_rerun = phase2.get("p0_rerun_check") or {}
    if p0_rerun:
        a(f"- P0 重跑一致性（d_end 全人群 vs phase-1 结果）: "
          f"**{p0_rerun.get('status')}**，max_abs_diff = "
          f"{p0_rerun.get('max_abs_diff')}")
        a("")
    a("### 9.6 结论边界")
    a("")
    for note in phase2.get("conclusion_notes", []):
        a(f"- {note}")
    a("- **本阶段仅报告，不授权训练**（`training_authorized = False` 不变）；"
      "P5−P0 MME 与 6 项有效性检查的结果交由 **D2-G 人工/管理层评审**裁决"
      "（plan §6/§10，唯一 GPU 入口）。")
    return lines


def render_markdown(pack: dict[str, Any]) -> str:
    spec = pack["probe_spec"]
    gate = pack["gate"]
    agg = pack["probe_results"]["aggregate"]
    deltas = pack["paired_deltas"]
    boot = pack["bootstrap"]
    repro = pack["p0_d1_reproduction"]
    feat = pack["feature_disclosure"]
    ds = pack["dataset"]

    def mark(status: str) -> str:
        return status if status in ("PASS", "NO_LEAK") else f"**{status}**"

    lines: list[str] = []
    a = lines.append
    phase2 = pack.get("phase2") or None
    probe_list: list[str] = list(PROBE_ORDER)
    if "P5/hgb" in agg:
        probe_list.append("P5")
    pop_note = ("（P5 行/列 = basin-label 人群，phase 2；phase-1 各行为 "
                "d_end 人群 — 人群不同，勿直接横比）") if "P5" in probe_list else ""
    a("# V5.2-R D2 报告 — 输入充分性探针 P0–P4（阶段 1，CPU，冻结 R3 编码器）")
    a("")
    a(f"> 生成: `scripts/agent_ops/data_gate/v52r_d2_probe.py` · schema "
      f"`{pack['schema']}` · gate `{gate['gate_id']}`")
    a(f"> 预注册: `probe_spec.json` sha256 `{pack['probe_spec_sha256'][:16]}…`"
      f" · commit `{pack['provenance']['code_commit'][:12]}`")
    a("> 划分: dev folds 0–2 only（sealed folds 3/4 仅计数，不编码/不探针）· "
      "9 单元 = 3 折 × 3 seed · 统计单位 = 曲线（n=30），非行")
    if phase2:
        a("> **阶段 2 已运行**: P5 化学终点监督（`p5_spec.json` sha256 "
          f"`{(phase2.get('p5_spec_sha256') or '')[:16]}…`）— 见 §9；"
          "gate 表含 6 项有效性检查 + mme_p5_minus_p0")
    a("")
    a("## 0. 数据与防火墙")
    a("")
    a(f"- dev 行数 {ds['n_dev_rows']} / 总行数 {ds['n_rows']}；sealed 行 "
      f"{ds['n_sealed_rows']}（曲线 {ds['n_sealed_curves']}，**仅计数**）")
    a(f"- dev 曲线 {ds['n_dev_curves']} 条；标签: 正例 "
      f"{ds['primary_window']['n_positive']} / 负例 "
      f"{ds['primary_window']['n_negative']} / Mask "
      f"{ds['primary_window']['n_mask']}（全表 prevalence "
      f"{ds['primary_window']['prevalence']}）")
    a(f"- usable（label ≥ 0 且非退化）: {ds['n_usable_rows']} 行 / 正例 "
      f"{ds['n_usable_positive']}；退化 flag 行 {ds['n_degenerate_rows']} 行 "
      f"（曲线 {list(DEGENERATE_TS_CURVES)}，排除 train/eval，单独计数）")
    a(f"- 编码器: 冻结 R3 缓存 `{pack['embeddings_dir']}`（9 个 npz，sha256 见 "
      "provenance）；无 torch、无编码器训练")
    a(f"- 目标帧防火墙: {feat['goal_free_assertion']}；P0/P4 前向特征零目标坐标")
    a(f"- 非有限特征置 0 计数: {feat['nonfinite_replaced'] or '无'}")
    a("")
    a("## 1. 预注册（probe_spec.json，运行前冻结）")
    a("")
    a(f"- 阈值规则: {spec['threshold_rule']}")
    a(f"- 划分: {spec['fold_protocol']['per_unit']}；"
      f"inner split {spec['fold_protocol']['inner_split']}")
    a(f"- 主模型: `{spec['models']['hgb_primary']}`；次模型 logistic / "
      "MLP(32,)（全局预算 600 s 共享）")
    a(f"- MME: P1−P0 ΔAUPRC ≥ +{MME_P1:.2f} 且配对 CI 下界 > 0；"
      f"P4−P0 ≥ +{MME_P4:.2f} 且 CI 下界 > 0；"
      f"P5−P0 ≥ +{MME_P5:.2f}（**pending 阶段 2**）")
    a(f"- 自检/泄漏: P2 ≥ {P2_SELF_CHECK_MIN:.2f}；P3 ≥ P4 − "
      f"{P3_LEAK_MARGIN:.2f} → 泄漏")
    a(f"- bootstrap: {spec['bootstrap']['method']}，"
      f"{spec['bootstrap']['n_resamples']} 次，seed {BOOTSTRAP_SEED}")
    a("")
    a("## 2. 特征构成披露（每探针列块）")
    a("")
    a("| 探针 | 特征数 | 目标信息 | 列块 |")
    a("|---|---:|---|---|")
    for probe in [p for p in probe_list if p in feat["per_probe"]]:
        f = feat["per_probe"][probe]
        cols = " + ".join(f"{c['block']}[{c['start']}:{c['stop']}]"
                          for c in f["columns"])
        a(f"| {probe} | {f['n_features']} | {f['goal_info']} | {cols} |")
    a("")
    a(f"- P3 为纯拓扑（{P3_N_FEATURES} 维，无坐标、无 intent/curve one-hot）；"
      "z/h 编码器变体见 §4（enc/*，当前帧冻结编码器输入）。")
    a("")
    a("## 3. 每单元结果（主模型 HGB，9 单元）")
    a("")
    a("| fold | seed | probe | AUROC | AUPRC | prevalence | prec@thr | "
      "rec@thr | start FP | TS FP |")
    a("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for u in pack["probe_results"]["units"]:
        for probe in probe_list:
            m = u["probes"].get(f"{probe}/hgb")
            if m is None:
                continue
            if m.get("skipped"):
                a(f"| {u['fold']} | {u['seed']} | {probe} | — | — | — | — | "
                  "— | — | — |")
                continue
            fs = m["false_stop"]
            a(f"| {u['fold']} | {u['seed']} | {probe} | {m['auroc']} | "
              f"{m['auprc']} | {m['prevalence']} | "
              f"{m['precision_at_threshold']} | {m['recall_at_threshold']} | "
              f"{fs['start']['rate']} | {fs['ts']['rate']} |")
    a("")
    a("### 聚合（9 单元 median [min, max]，全部模型）")
    a("")
    a("| probe/model | AUROC | AUPRC | prevalence | prec@thr | rec@thr | "
      "start FP | TS FP | 缺阈值单元 |")
    a("|---|---|---|---|---|---|---|---|---:|")
    for key in _agg_keys(probe_list):
        s = agg.get(key)
        if s is None:
            continue
        if s.get("skipped"):
            a(f"| {key} | — | — | — | — | — | — | — | skipped |")
            continue
        a(f"| {key} | {_fmt(s['auroc'])} | {_fmt(s['auprc'])} | "
          f"{_fmt(s['eval_prevalence'])} | {_fmt(s['precision_at_threshold'])} | "
          f"{_fmt(s['recall_at_threshold'])} | {_fmt(s['start_fp_rate'])} | "
          f"{_fmt(s['ts_fp_rate'])} | {s['units_missing_threshold']} |")
    a("")
    a(f"- 阈值规则（预注册）: 训练集内层曲线分组 split 上 precision ≥ "
      f"{PRECISION_TARGET:.2f} 的最小阈值；达不到则 thr=∅（预测零正例）。")
    if pop_note:
        a(f"- {pop_note}")
    a("- 池化（P0–P4 每探针，主模型，曲线 cluster bootstrap 95% CI）:")
    a("")
    a("| probe | 池化 AUPRC | 95% CI | prevalence |")
    a("|---|---:|---|---:|")
    for probe in probe_list:
        b = boot["per_probe_pooled"].get(probe)
        if b is None:
            continue
        a(f"| {probe} | {b['point']} | [{b['ci95']['lower']}, "
          f"{b['ci95']['upper']}] | {b['prevalence']} |")
    a("")
    a("## 4. 配对比较（Δ vs P0，同折同 seed）与 bootstrap CI")
    a("")
    a("| 比较 | 角色 | 单元 Δ med [min,max] | 池化 ΔAUPRC | 95% CI | MME | 判定 |")
    a("|---|---|---|---:|---|---|:--:|")
    row_spec = [
        ("P1-P0", "primary", MME_P1, "mme_p1_minus_p0"),
        ("P4-P0", "primary", MME_P4, "mme_p4_minus_p0"),
        ("P5-P0", "primary", MME_P5, "mme_p5_minus_p0"),
        ("P1c-P0", "exploratory", None, None),
        ("P3-P0", "exploratory", None, None),
    ]
    comparisons = pack["bootstrap"]["comparisons"]
    _spec_names = {r[0] for r in row_spec}
    rows = [r for r in row_spec if r[0] in comparisons] + [
        (k, "exploratory", None, None)
        for k in sorted(comparisons) if k not in _spec_names]
    check_by_id = {c["id"]: c for c in gate["checks"]}
    for name, role, mme, check_id in rows:
        probe = name.split("-")[0]
        d = deltas.get(name) or deltas.get(probe) or {}
        b = comparisons.get(name, {})
        if b.get("status") == "pending":
            a(f"| {name} | {role} | — | — | — | ≥ +{mme:.2f} | **PENDING** "
              "（阶段 2）|")
            continue
        status = (check_by_id.get(check_id) or {}).get("status") or "exploratory"
        ci = (f"[{b.get('ci95', {}).get('lower')}, "
              f"{b.get('ci95', {}).get('upper')}]")
        delta_fmt = _fmt({"median": d.get("median"), "min": d.get("min"),
                          "max": d.get("max")})
        mme_text = f"≥ +{mme:.2f}" if mme is not None else "—"
        a(f"| {name} | {role} | {delta_fmt} | {b.get('point')} | "
          f"{ci} | {mme_text} | {mark(status)} |")
    a("")
    a(f"- {boot['n_resamples']} 次重采样（有效/比较见 pack），rng seed "
      f"{BOOTSTRAP_SEED}，30 条 dev 曲线；exploratory 比较不参与门控。")
    a(f"- 编码器变体（enc/*，每单元 logistic [z64]/[h48]/[z⊕h] + mlp_z）聚合 "
      f"AUPRC: " + "；".join(
        f"{v} {_fmt(agg.get(f'enc/{v}', {}).get('auprc'))}"
        for v in ENCODER_VARIANTS))
    a("")
    a("## 5. P2 自检与 P3 泄漏判定")
    a("")
    a("| 规则 | 期望 | 实测 | 判定 |")
    a("|---|---|---|:--:|")
    for c in gate["checks"]:
        if c["id"] in ("p2_self_check", "p3_leak_rule"):
            a(f"| {c['id']} | {c['expect']} | {c['actual']} | "
              f"{mark(c['status'])} |")
    a("")
    p2_med = agg.get("P2/hgb", {}).get("auprc", {}).get("median")
    p3_med = agg.get("P3/hgb", {}).get("auprc", {}).get("median")
    p4_med = agg.get("P4/hgb", {}).get("auprc", {}).get("median")
    leak_bound = d1._r6((p4_med or 0.0) - P3_LEAK_MARGIN) \
        if p4_med is not None else None
    a(f"- P2 中位 AUPRC **{p2_med}**（≥ {P2_SELF_CHECK_MIN} 才可信）；"
      f"P3 **{p3_med}** vs P4 **{p4_med}**（泄漏边界 P4 − "
      f"{P3_LEAK_MARGIN} = {leak_bound}）。")
    a("")
    a("## 6. 每曲线散点（池化主模型 AUPRC / 曲线）")
    a("")
    header = "| curve | " + " | ".join(probe_list) + " |"
    a(header)
    a("|---" * (len(probe_list) + 1) + "|")
    for cid, row in sorted(pack["per_curve_auprc"].items()):
        cells = " | ".join(
            ("—" if row.get(p) is None else str(row.get(p)))
            for p in probe_list)
        a(f"| {cid} | {cells} |")
    a("")
    a("- 完整散点数据在 `evidence_pack.json` `per_curve_auprc`；曲线级结论"
      "（统计单位 n=30）不得用行级显著性替代。")
    if pop_note:
        a(f"- {pop_note}")
    a("")
    a("## 7. P0 与 D1 复现核对（双口径）")
    a("")
    rep_in = repro.get("in_process", {})
    rep_st = repro.get("stored_pack", {})
    a(f"- **硬判据 — 进程内 D1 runner 复现**（同环境、同输入、D1 原代码 "
      f"`run_endpoint_probes`）: **{rep_in.get('status')}** · 容差 "
      f"{rep_in.get('tolerance')} · 比较 {rep_in.get('n_compared')} 项 · "
      f"最大绝对差 {rep_in.get('max_abs_diff')} · 不匹配 "
      f"{rep_in.get('n_mismatches')}")
    if rep_in.get("per_probe_max_abs_diff"):
        a("- 进程内每 D1 探针最大绝对差: " + "；".join(
            f"{k}={v}" for k, v in rep_in["per_probe_max_abs_diff"].items()))
    a(f"- 存档 D1 包核对（环境漂移审计）: **{rep_st.get('status')}** · "
      f"容差 {rep_st.get('tolerance')} · 比较 {rep_st.get('n_compared')} 项 "
      f"· 最大绝对差 {rep_st.get('max_abs_diff')} · 不匹配 "
      f"{rep_st.get('n_mismatches')}")
    if rep_st.get("per_probe_max_abs_diff"):
        a("- 存档包每 D1 探针最大绝对差: " + "；".join(
            f"{k}={v}" for k, v in rep_st["per_probe_max_abs_diff"].items()))
    a(f"- 说明: {rep_st.get('note')}")
    if rep_st.get("status") == "FAIL" and rep_in.get("status") == "PASS":
        a("- **结论: D2 P0 与 D1 代码路径完全一致（PASS）；存档 D1 包与当前 "
          "sklearn 构建存在环境漂移，D1 探针在本环境下重跑即复现 D2 P0。**")
    a("")
    a("## 8. Gate 判定（V5.2R-D2-PHASE1-PROBE，仅报告不授权）")
    a("")
    a("| 检查 | 期望 | 实测 | 判定 |")
    a("|---|---|---|:--:|")
    for c in gate["checks"]:
        a(f"| {c['id']} | {c['expect']} | {c['actual']} | "
          f"{mark(c['status'])} |")
    a("")
    if phase2:
        a(f"**Gate 总体（阶段 1 + 阶段 2）: {gate['verdict']}** · "
          f"training_authorized = **{gate['training_authorized']}**")
    else:
        a(f"**Phase-1 总体: {gate['verdict']}** · training_authorized = "
          f"**{gate['training_authorized']}**")
    a("")
    a(f"> {gate['authorization_note']}")
    a("")
    a("## 9. P5 状态")
    a("")
    if phase2:
        lines.extend(_render_phase2_section(phase2, gate))
    else:
        a("- **P5 未启动——待 D0.1 异常清零**（plan §3.1：22 个拓扑异常行分类"
          "清零前 P5 不得启动）；P5 由另一 lane 拥有，本脚本 `--p5` 仅作 stub。")
        a("- 阶段 2 指针: P5（`d_basin` 化学终点监督，plan §4.6）→ P5−P0 MME "
          "（≥ +0.10 且 CI 下界 > 0，spec 已预注册为 PENDING）→ **D2-G 门评审** "
          "（plan §6，人工/管理层，唯一 GPU 入口）→ T1 条件训练（plan §7）。")
    a("")
    a("## 10. 运行时")
    a("")
    rt = pack["runtime_s"]
    a(f"> 运行时: spec {rt['spec_write']} s · 特征构建 {rt['feature_build']} s · "
      f"探针 {rt['probing']} s · D1 复现核对 {rt.get('d1_parity_check')} s · "
      f"统计 {rt['stats']} s · 总计 {rt['total']} s"
      f"（目标 ≤ 600 s）· MLP 已用 {rt['mlp_spent']} / {MLP_BUDGET_S:.0f} s")
    if phase2:
        p2rt = phase2.get("runtime_s", {})
        xt = phase2.get("enrichment", {})
        a(f"> 阶段 2（P5）: 总计 {p2rt.get('total')} s · xtb {xt.get('n_calls')} "
          f"次 / {xt.get('wall_s')} s（上限 50 次 / 600 s）· "
          f"标签+富集 {p2rt.get('labels_and_enrichment')} s · "
          f"探针 {p2rt.get('probing')} s")
    a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run(tables_dir: Path, out_dir: Path, embeddings_dir: Path, *,
        n_bootstrap: int = BOOTSTRAP_RESAMPLES,
        d1_pack_path: Path = DEFAULT_D1_PACK) -> dict[str, Any]:
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = KeyAtomDatasetV3(
        tables_dir,
        label_statuses=("ok", "terminal", "masked", "conflict"))
    samples = dataset.samples
    dev_positions = [i for i, s in enumerate(samples) if s.fold in DEV_FOLDS]
    sealed_positions = [i for i, s in enumerate(samples)
                        if s.fold in SEALED_FOLDS]
    v52_labels = dataset.v52_labels()
    labels = build_endpoint_stop_labels(dataset, v52_labels)
    meta = d1.load_meta(dataset, dev_positions)
    if not set(np.unique(meta["folds"]).tolist()) <= set(DEV_FOLDS):
        raise AssertionError("sealed/unknown fold leaked into dev positions")

    z_by_unit, h_by_unit, emb_paths = load_embeddings(
        Path(embeddings_dir), meta)

    # ---- PRE-REGISTRATION: write probe_spec.json BEFORE any model run ------
    t0 = time.time()
    hashes = collect_hashes(Path(tables_dir), emb_paths)
    spec = build_probe_spec(hashes=hashes, n_bootstrap=n_bootstrap,
                            d1_pack_path=str(d1_pack_path))
    spec_sha = spec_sha256(spec)
    (out_dir / "probe_spec.json").write_text(spec_text(spec), encoding="utf-8")
    t_spec = time.time() - t0

    # ---- features ----------------------------------------------------------
    t0 = time.time()
    blocks, feat_info = build_probe_blocks(dataset, dev_positions, labels)
    lab_vec = np.array([labels.by_sample[sid].endpoint_stop_label
                        for sid in meta["sample_ids"]], dtype=np.int64)
    deg_vec = np.array([bool(labels.by_sample[sid].degenerate_flag)
                        for sid in meta["sample_ids"]], dtype=np.bool_)
    usable = (lab_vec >= 0) & (~deg_vec)
    t_feat = time.time() - t0

    # ---- probes ------------------------------------------------------------
    t0 = time.time()
    mlp_state: dict[str, float] = {"spent": 0.0}
    units, pooled = run_probe_units(blocks, labels, meta,
                                    z_by_unit, h_by_unit, mlp_state)
    t_probe = time.time() - t0

    # ---- statistics --------------------------------------------------------
    t0 = time.time()
    agg = aggregate_units(units)
    deltas = paired_delta_summary(units)
    per_probe_pooled = {
        probe: d1.curve_cluster_bootstrap_auprc(
            scores, labels_arr, curves, n_resamples=n_bootstrap)
        for probe, (scores, labels_arr, curves) in pooled.items()}
    comparisons: dict[str, Any] = {}
    for name, (probe_a, probe_b) in {
            "P1-P0": ("P1", "P0"),
            "P4-P0": ("P4", "P0"),
            "P1c-P0": ("P1c", "P0"),
            "P3-P0": ("P3", "P0")}.items():
        sa, la, ca = pooled[probe_a]
        sb, _lb, _cb = pooled[probe_b]
        res = paired_cluster_bootstrap_auprc(
            sa, sb, la, ca, n_resamples=n_bootstrap)
        res["role"] = ("primary" if name in ("P1-P0", "P4-P0")
                       else "exploratory")
        res["probes"] = [probe_a, probe_b]
        comparisons[name] = res
    comparisons["P5-P0"] = {
        "status": "pending",
        "role": "primary",
        "mme": MME_P5,
        "note": "P5 未启动——待 D0.1 异常清零（阶段 2，另一 lane）",
        "point": None, "ci95": {"lower": None, "upper": None}}
    per_curve = per_curve_auprc_table(pooled)

    def _mme_entry(name: str, effect: float) -> dict[str, Any]:
        b = comparisons[name]
        d = deltas[name.split("-")[0]]
        return {"median_delta": d.get("median"),
                "ci_lower": b.get("ci95", {}).get("lower"),
                "ci_upper": b.get("ci95", {}).get("upper"),
                "point": b.get("point"), "mme": effect,
                "n_units": d.get("n_units")}

    mme_p1 = _mme_entry("P1-P0", MME_P1)
    mme_p4 = _mme_entry("P4-P0", MME_P4)

    # ---- P0 ↔ D1 reproduction ---------------------------------------------
    t0_d1 = time.time()
    # HARD criterion: D1's OWN probe runner re-run in this process/env on the
    # identical features/labels/splits (protocol parity, env-independent).
    d1_units = d1.run_endpoint_probes(
        {"z": z_by_unit, "h": h_by_unit, "tabular": blocks["P0"]},
        labels, meta)
    d1_pack: dict[str, Any] | None = None
    if Path(d1_pack_path).is_file():
        d1_pack = json.loads(Path(d1_pack_path).read_text(encoding="utf-8"))
    p0_check = compare_p0_to_d1(units, d1_units, d1_pack)
    t_d1check = time.time() - t0_d1

    gate = evaluate_phase1_gate(
        p0_check=p0_check,
        p2_median=agg.get("P2/hgb", {}).get("auprc", {}).get("median"),
        p3_median=agg.get("P3/hgb", {}).get("auprc", {}).get("median"),
        p4_median=agg.get("P4/hgb", {}).get("auprc", {}).get("median"),
        mme_p1=mme_p1, mme_p4=mme_p4)
    t_stats = time.time() - t0

    provenance = d1.collect_provenance(Path(tables_dir))
    provenance.update({
        "label_builder_sha256": hashes["label_builder_sha256"],
        "geometry_stereo_sha256": hashes["geometry_stereo_sha256"],
        "d1_probe_script_sha256": hashes["d1_probe_script_sha256"],
        "d2_probe_script_sha256": hashes["d2_probe_script_sha256"],
        "embeddings_sha256": hashes["embeddings_sha256"],
        "probe_spec_sha256": spec_sha,
        "d1_evidence_pack": str(d1_pack_path),
        "d1_evidence_pack_sha256": d1.sha256_file(Path(d1_pack_path))
        if Path(d1_pack_path).is_file() else "",
    })

    pack: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "label_schema": "pes2ts_fullpath_nav_labels_v1",
        "probe_spec": spec,
        "probe_spec_sha256": spec_sha,
        "gate": gate,
        "dataset": {
            "n_rows": len(samples),
            "n_dev_rows": len(dev_positions),
            "n_sealed_rows": len(sealed_positions),
            "n_sealed_curves": len({samples[i].curve_id
                                    for i in sealed_positions}),
            "sealed_folds": list(SEALED_FOLDS),
            "sealed_use": "counted only; never encoded/featurized/probed",
            "n_dev_curves": len(set(meta["curve_ids"].tolist())),
            "n_degenerate_rows": int(deg_vec.sum()),
            "n_usable_rows": int(usable.sum()),
            "n_usable_positive": int(lab_vec[usable].sum()),
            "primary_window": {
                "w_pos": d1.PRIMARY_W_POS,
                "w_neg": d1.W_NEG,
                "n_positive": labels.n_positive,
                "n_negative": labels.n_negative,
                "n_mask": labels.n_mask,
                "prevalence": d1._r6(labels.n_positive
                                     / max(1, labels.n_rows)),
            },
        },
        "embeddings_dir": str(embeddings_dir),
        "feature_disclosure": feat_info,
        "probe_results": {"units": units, "aggregate": agg,
                          "precision_target": PRECISION_TARGET,
                          "threshold_rule": spec["threshold_rule"]},
        "paired_deltas": deltas,
        "bootstrap": {"n_resamples": n_bootstrap, "seed": BOOTSTRAP_SEED,
                      "per_probe_pooled": per_probe_pooled,
                      "comparisons": comparisons},
        "per_curve_auprc": per_curve,
        "p0_d1_reproduction": p0_check,
        "provenance": provenance,
        "runtime_s": {
            "spec_write": d1._r6(t_spec),
            "feature_build": d1._r6(t_feat),
            "probing": d1._r6(t_probe),
            "d1_parity_check": d1._r6(t_d1check),
            "stats": d1._r6(t_stats),
            "mlp_spent": d1._r6(mlp_state["spent"]),
            "total": d1._r6(time.time() - t_start),
        },
    }
    (out_dir / "evidence_pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "D2_PROBE.md").write_text(
        render_markdown(pack), encoding="utf-8")
    return pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    ap.add_argument("--embeddings-dir", type=Path,
                    default=DEFAULT_EMBEDDINGS_DIR,
                    help="frozen R3 embedding cache (default: the D1 cache)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES,
                    help="curve-cluster-bootstrap resamples (default 1000)")
    ap.add_argument("--d1-pack", type=Path, default=DEFAULT_D1_PACK,
                    help="D1 evidence pack for the P0 reproduction check")
    ap.add_argument("--p5", action="store_true",
                    help="STUB: P5 is out of scope for D2 phase 1 "
                         "(phase-2 lane; gated on D0.1 anomaly zeroing)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on pipeline-integrity failure (P0 "
                         "reproduction FAIL / P2 FAIL / P3 LEAK); MME "
                         "PENDING-GATE never fails strict")
    args = ap.parse_args(argv)
    if args.p5:
        print("P5 is OUT OF SCOPE for D2 phase 1: 未启动——待 D0.1 异常清零 "
              "(plan §3.1); another lane owns P5 (phase 2).")
        return 2
    pack = run(args.tables_dir, args.out, args.embeddings_dir,
               n_bootstrap=args.bootstrap, d1_pack_path=args.d1_pack)
    gate = pack["gate"]
    print(f"[{gate['gate_id']}] verdict: {gate['verdict']}  "
          f"(training_authorized={gate['training_authorized']})")
    for c in gate["checks"]:
        print(f"  {c['status']:<12} {c['id']}: {c['actual']}")
    agg = pack["probe_results"]["aggregate"]
    print("aggregate AUPRC (median [min,max], 9 units, primary HGB):")
    for probe in PROBE_ORDER:
        s = agg.get(f"{probe}/hgb", {})
        print(f"  {probe:<3} {_fmt(s.get('auprc'))}")
    print(f"P0-vs-D1: in-process {pack['p0_d1_reproduction']['in_process']['status']} "
          f"(max_diff={pack['p0_d1_reproduction']['in_process']['max_abs_diff']}) | "
          f"stored pack {pack['p0_d1_reproduction']['stored_pack']['status']} "
          f"(max_diff={pack['p0_d1_reproduction']['stored_pack'].get('max_abs_diff')})")
    print(f"spec sha256: {pack['probe_spec_sha256']}")
    print(f"runtime total: {pack['runtime_s']['total']} s "
          f"(target ≤ 600 s; MLP {pack['runtime_s']['mlp_spent']}/600 s)")
    print(f"evidence: {args.out / 'evidence_pack.json'}")
    print(f"report:   {args.out / 'D2_PROBE.md'}")
    if args.strict and gate["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
